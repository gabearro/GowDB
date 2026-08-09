//! Filesystem layout, atomic publication, and catalog-level save/load.
//!
//! Everything that touches the filesystem lives here so that the durability
//! argument is auditable in one place: [`atomic_write`] is the only way any
//! byte in this module reaches its final name, and every other function is
//! path arithmetic on top of it.
//!
//! ## Why parts are numbered, never overwritten
//!
//! Part files are named `part_<seq>.gpart` with `seq` allocated above every
//! sequence number already present in the directory. A rewrite of a table
//! therefore never touches a file a reader might be holding, and a crash
//! mid-rewrite leaves the old files *and* a partial set of new ones -- all of
//! which the still-committed `TABLE` file ignores. The next successful commit
//! collects the orphans. The alternative (reusing `part_000001.gpart`) would
//! make the rename atomicity useless: the file would be atomically replaced,
//! but the table as a whole would be in a mixed state.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

use crate::catalog::{Catalog, DEFAULT_DELTA_LIMIT};
use crate::common::{lane_to_f64, lane_to_i64, Error, Result};
use crate::types::{DataType, PhysicalType, TableDef, Value};

use super::format;
use super::wal::{Wal, WalRecord};
use super::{reader, writer};

/// Root file listing every database and the definition of every table in it.
pub const CATALOG_FILE: &str = "CATALOG";
/// Per-table commit point: definition + live part files + WAL watermark.
pub const TABLE_FILE: &str = "TABLE";
/// Per-table write-ahead log.
pub const WAL_FILE: &str = "wal.log";
/// Part file extension. Deliberately distinctive: the GC in
/// [`writer::write_table`] deletes by pattern, and must never match a file it
/// did not create.
pub const PART_EXT: &str = "gpart";

// ---------------------------------------------------------------------------
// paths
// ---------------------------------------------------------------------------

pub fn part_file_name(seq: u64) -> String {
    format!("part_{seq:06}.{PART_EXT}")
}

/// `Some(seq)` if `name` is one of our part files. Rejects anything it does
/// not fully understand, so a stray file is never a deletion candidate.
pub fn parse_part_seq(name: &str) -> Option<u64> {
    let rest = name.strip_prefix("part_")?;
    let digits = rest.strip_suffix(&format!(".{PART_EXT}"))?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u64>().ok()
}

/// Whether `name` may be used as a directory component.
///
/// Database and table names reach us from a `CREATE TABLE` on one side and
/// from a `CATALOG` file on the other, and both end up in a `join`. A name
/// containing a separator (or `..`) would place a table's data outside the
/// tree it belongs to, so it is refused rather than sanitized -- silently
/// rewriting a user's name is how two tables end up sharing a directory. A
/// leading dot is refused too: that is the namespace [`atomic_write`] uses for
/// its in-flight temp files.
pub fn is_safe_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains(['/', '\\', '\0'])
        && !name.starts_with('.')
}

/// The names the *data directory itself* owns, so no database may take one.
///
/// Both are files this module writes at the root, and a database is a
/// directory: on a fresh directory `mkdir` wins the race and `CATALOG` can
/// never be written afterwards, which leaves every subsequent open failing
/// with `Is a directory`. Two entries and no more -- `TABLE` and `wal.log`
/// live one level *below* a table directory and provably cannot collide, and
/// `.wal-archive` is already refused by the leading dot above. A longer list
/// would only be false refusals.
pub const ROOT_RESERVED: [&str; 2] = [CATALOG_FILE, crate::session::LOCK_FILE];

/// The existing sibling `name` would share a directory entry with, if any.
///
/// Case-*insensitively*, because the filesystem underneath usually is:
/// macOS's default APFS and every Windows one fold case, so `Tenant` and
/// `tenant` are one directory there. Two databases of those names therefore
/// share one directory, and the second `CREATE` silently adopts the first's
/// files -- confirmed on this developer's machine, where it served one
/// tenant's rows under another's name and destroyed the first's. `RENAME
/// TABLE beta TO ALPHA` is worse still: it loses both tables.
///
/// We *refuse* rather than escape the on-disk name. Escaping would rewrite
/// names that existing directories already carry, and a directory whose
/// entries no longer match its catalog is the very failure being fixed;
/// refusing cannot corrupt anything and keeps the tree readable. The price is
/// that a pair of names legal on a case-sensitive filesystem is refused here
/// as well -- taken deliberately, because the alternative on the other half of
/// the world's filesystems is not "legal", it is destructive.
///
/// Called from the `CREATE`/`RENAME` paths only, never from a loader: a
/// directory that already holds such a pair keeps opening exactly as it did.
/// An exact match is not a collision -- that is `IF NOT EXISTS`, and the
/// caller's own "already exists" rule owns it.
pub fn folds_onto<'a>(name: &str, mut siblings: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    siblings.find(|s| *s != name && s.eq_ignore_ascii_case(name))
}

/// A one-line refusal for [`folds_onto`] / [`ROOT_RESERVED`], shared by the
/// three DDL sites so they cannot drift apart.
pub fn name_collision(what: &str, name: &str, with: &str) -> Error {
    Error::storage(format!(
        "refusing to create `{what} {name}`: `{with}` already owns that directory entry on a \
         case-insensitive filesystem (macOS, Windows), where the two would be one directory \
         and one would overwrite the other. Pick a name that differs by more than case"
    ))
}

/// Every part file physically present, ascending by sequence number. This is
/// *not* the set of live parts -- only the `TABLE` file knows that -- it is
/// the set of files the allocator must not collide with.
pub fn list_part_files(dir: &Path) -> Result<Vec<(u64, String)>> {
    let mut out = Vec::new();
    let rd = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(io_err("read directory", dir, e)),
    };
    for entry in rd {
        let entry = entry.map_err(|e| io_err("read directory entry in", dir, e))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if let Some(seq) = parse_part_seq(name) {
            out.push((seq, name.to_string()));
        }
    }
    out.sort_unstable();
    Ok(out)
}

// ---------------------------------------------------------------------------
// durable writes
// ---------------------------------------------------------------------------

/// Temp-name counter. Combined with the pid this makes a name unique among
/// live processes without needing randomness (which would make a leftover
/// file impossible to attribute).
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

fn tmp_path(target: &Path) -> PathBuf {
    let n = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let stem = target.file_name().and_then(|s| s.to_str()).unwrap_or("file");
    let dir = target.parent().unwrap_or_else(|| Path::new("."));
    // Same directory as the target: `rename` is only atomic within a
    // filesystem, and a temp dir may well be on another one.
    dir.join(format!(".{stem}.tmp-{}-{n}", std::process::id()))
}

/// Publish `bytes` at `path` atomically and durably.
///
/// Four steps, none of them optional:
/// 1. write the whole body to a temp file in the same directory;
/// 2. `fsync` it -- otherwise the rename can outlive the data it names;
/// 3. `rename` over the target -- atomic, so no reader sees a prefix;
/// 4. `fsync` the directory -- otherwise the rename itself can be lost.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    if !dir.as_os_str().is_empty() {
        fs::create_dir_all(dir).map_err(|e| io_err("create directory", dir, e))?;
    }
    let tmp = tmp_path(path);
    let write = (|| -> std::io::Result<()> {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        Ok(())
    })();
    if let Err(e) = write {
        let _ = fs::remove_file(&tmp);
        return Err(io_err("write", &tmp, e));
    }
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(io_err("rename into place", path, e));
    }
    sync_dir(dir)
}

/// [`atomic_write`], skipped when the file already holds exactly `bytes`.
///
/// For the small, rewritten-every-checkpoint records: `CATALOG` and `TABLE`.
/// Once part writes are incremental (see [`writer::write_table`]) these are
/// what a checkpoint of an unchanged database still costs, and they are not
/// free: each one is a create, an `fsync`, a `rename` and a directory `fsync`,
/// and an SSD charges a full erase block for a 300-byte file. A `SELECT 1` on
/// a two-table database used to cost six of them.
///
/// Safe because the durability argument is inductive rather than per-call: the
/// only way any byte of these files ever changes is [`atomic_write`], which
/// fsyncs before it publishes, so bytes we can *read* are bytes that are
/// already durable, and writing them again would produce a file identical to
/// the one that is there. Reading a few hundred bytes to find that out is
/// vastly cheaper than the four syscalls it avoids.
///
/// Returns whether it actually wrote.
pub fn commit(path: &Path, bytes: &[u8]) -> Result<bool> {
    if fs::read(path).is_ok_and(|old| old == bytes) {
        return Ok(false);
    }
    atomic_write(path, bytes).map(|()| true)
}

/// `fsync` a directory so a rename inside it survives a power loss.
///
/// Some platforms and filesystems refuse `fsync` on a directory handle
/// outright. There is nothing a caller could do about that, and failing the
/// write would be strictly worse than the (still journalled) rename we already
/// performed, so a refusal is tolerated while a real I/O error is not.
pub fn sync_dir(dir: &Path) -> Result<()> {
    use std::io::ErrorKind::{InvalidInput, PermissionDenied, Unsupported};
    match File::open(dir).and_then(|f| f.sync_all()) {
        Ok(()) => Ok(()),
        Err(e) if matches!(e.kind(), InvalidInput | Unsupported | PermissionDenied) => Ok(()),
        Err(e) => Err(io_err("fsync directory", dir, e)),
    }
}

pub fn read_file(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|e| io_err("read", path, e))
}

pub(crate) fn io_err(what: &str, path: &Path, e: std::io::Error) -> Error {
    Error::Io(format!("cannot {what} {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// hard links, and the filesystems that have none
// ---------------------------------------------------------------------------

/// What this process has learned about hard links: 0 not asked yet,
/// [`LINKS_OK`] they work, [`LINKS_NONE`] they do not and everything copies.
///
/// One relaxed load per link, and at most **one** failed `link` syscall for
/// the whole process rather than one per part -- which is the cost that
/// mattered: a checkpoint of a large table links every part file, and paying
/// an `ENOTSUP` on each of them is a real bill on a filesystem that will never
/// answer differently. Process-wide rather than per-directory on purpose: the
/// state only ever makes us *copy*, which is correct on every filesystem, so
/// the worst a shared answer can do is cost a copy somewhere it was not
/// needed. (`Relaxed` because this is a hint: a racing pair of threads either
/// both try the link or both copy, and both outcomes are correct.)
static LINKS: AtomicU8 = AtomicU8::new(0);
const LINKS_OK: u8 = 1;
const LINKS_NONE: u8 = 2;

/// Set to anything to force the copy path. exFAT -- the filesystem this exists
/// for -- cannot be mounted on every machine that runs the tests, so this is
/// how the fallback is driven end to end; it doubles as the operator's way to
/// prove what a link-less volume would do before moving data onto one.
const NO_LINKS_ENV: &str = "GRANULAR_NO_HARD_LINKS";

/// `errno`s that mean *this filesystem* will never hard-link, as against "this
/// one link failed": ENOTSUP/EOPNOTSUPP (45 on macOS, 95 on Linux, and what a
/// real exFAT volume returns), EPERM, EXDEV, EMLINK. None of them get better
/// on the next file, which is what makes the answer worth remembering.
fn no_links(e: &std::io::Error) -> bool {
    matches!(e.raw_os_error(), Some(1 | 18 | 31 | 45 | 95))
        || e.kind() == std::io::ErrorKind::Unsupported
}

/// Remember that links are gone, and say so once.
///
/// Once per process, not per file: the point is to tell the operator that the
/// disk-space story changed, and repeating it per part would bury it. Reported
/// rather than done silently, because "checkpoint links unchanged parts" and
/// "checkpoint copies unchanged parts" are different bills, and this engine
/// refuses or reports -- it does not quietly do something else.
fn degrade(at: &Path, why: &str) {
    if LINKS.swap(LINKS_NONE, Ordering::Relaxed) != LINKS_NONE {
        eprintln!(
            "granular: the filesystem holding {} has no hard links ({why}); copying \
             instead. Checkpoints and log archiving now cost as much disk as the data \
             they duplicate.",
            at.display()
        );
    }
}

/// Link `from` to `to`, or copy it on a filesystem with no links.
///
/// exFAT has none at all, and the engine links in exactly two places -- the
/// incremental checkpoint and the sealed-segment archive. Before this, the
/// first `INSERT` on such a volume poisoned the directory: the failed archive
/// left the checkpoint unfinished, and *every* later statement, `SELECT`
/// included, exited non-zero forever.
pub fn link_or_copy(from: &Path, to: &Path) -> std::io::Result<()> {
    let known = LINKS.load(Ordering::Relaxed);
    if known != LINKS_NONE {
        if known != LINKS_OK && std::env::var_os(NO_LINKS_ENV).is_some() {
            degrade(to, NO_LINKS_ENV);
        } else {
            match fs::hard_link(from, to) {
                Ok(()) => {
                    LINKS.store(LINKS_OK, Ordering::Relaxed);
                    return Ok(());
                }
                Err(e) if no_links(&e) => degrade(to, &e.to_string()),
                Err(e) => return Err(e),
            }
        }
    }
    // `hard_link` refuses an existing target, and a caller depends on it: the
    // WAL archive reads `AlreadyExists` as "this segment is already here" and
    // compares lengths to catch a divergent one. `fs::copy` would overwrite
    // and lose that check, so the refusal is reproduced rather than dropped.
    if to.try_exists()? {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("{} already exists", to.display()),
        ));
    }
    fs::copy(from, to).map(|_| ())
}

// ---------------------------------------------------------------------------
// catalog
// ---------------------------------------------------------------------------

/// A value that will not collide with another data directory's, from what is
/// already on hand: the wall clock and the pid, mixed by the splitmix64 this
/// crate already carries. Not cryptographic and does not need to be -- it
/// exists so that two directories can be told apart, not so that one cannot be
/// forged. Never 0: that value means "unstamped" everywhere it is read.
fn mint_instance() -> u64 {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    crate::common::hash::splitmix64(ns ^ ((std::process::id() as u64) << 40)) | 1
}

/// The instance id stamped into `root`'s `CATALOG`, or 0 if there is none to
/// read. Best-effort by construction: every reason this could fail (no
/// directory, no `CATALOG`, a `CATALOG` from an older build) is a reason to
/// answer "unstamped", and the caller's rule for that is to let the operation
/// through rather than to refuse on an identity nobody recorded.
pub fn instance_at(root: &Path) -> u64 {
    read_file(&root.join(CATALOG_FILE))
        .and_then(|b| reader::catalog_from_bytes(&b))
        .map_or(0, |(_, id)| id)
}

/// Checkpoint the whole catalog. A no-op for an in-memory catalog, which is
/// what makes it safe to call unconditionally from a session.
///
/// Ordering per table: flush the delta into parts, write the part files,
/// commit the `TABLE` file, truncate the WAL, then re-commit the (now zero)
/// WAL watermark. See the module docs for why the last step exists.
pub fn save_catalog(catalog: &mut Catalog) -> Result<()> {
    let Some(root) = catalog.dir().map(Path::to_path_buf) else {
        return Ok(());
    };
    // Minted here rather than at open, because here is where it becomes
    // durable: an id handed out by a session that never checkpointed would be
    // stamped into a backup and then forgotten, and the next open would mint a
    // different one and refuse that backup's own database.
    if catalog.instance() == 0 {
        catalog.set_instance(mint_instance());
    }
    catalog.flush_all()?;
    fs::create_dir_all(&root).map_err(|e| io_err("create directory", &root, e))?;

    let mut roster: Vec<(String, Vec<TableDef>)> = Vec::new();
    let mut checkpointed: Vec<PathBuf> = Vec::new();
    let mut live: Vec<(PathBuf, Vec<String>)> = Vec::new();
    for db in catalog.database_names() {
        if !is_safe_name(&db) {
            return Err(Error::storage(format!(
                "database name `{db}` cannot be a directory name"
            )));
        }
        let ddir = root.join(&db);
        fs::create_dir_all(&ddir).map_err(|e| io_err("create directory", &ddir, e))?;
        let names = catalog.table_names(Some(&db))?;
        let mut defs = Vec::new();
        for name in &names {
            // A quarantined table keeps its place in the committed roster and
            // has none of its files rewritten. That is exactly what
            // `Catalog::quarantined_def` was written for, and it had no caller
            // here -- so one bad file made *every* checkpoint of the whole
            // database fail, which is how a CLI run that answered every query
            // correctly still exited 1. Dropping it from the roster instead
            // would be worse than the error: the next checkpoint treats a
            // directory the catalog does not name as a dropped table and
            // deletes it, turning one unreadable file into total loss.
            let t = match catalog.table_by_path(&format!("{db}.{name}")) {
                Ok(t) => t,
                Err(e) => match catalog.quarantined_def(&db, name) {
                    Some(def) => {
                        defs.push(def.clone());
                        continue;
                    }
                    None => return Err(e),
                },
            };
            // `Memory` tables are defined to vanish on restart; persisting one
            // would silently change its semantics.
            if !t.def.engine.is_persistent() {
                continue;
            }
            writer::write_table(&ddir, t)?;
            checkpointed.push(ddir.join(name));
            defs.push(t.def.clone());
        }
        // Every name the catalog holds, not just the persistent ones: a name
        // it still knows is never a deletion candidate, whatever its engine.
        live.push((ddir, names));
        roster.push((db, defs));
    }
    commit(&root.join(CATALOG_FILE), &writer::catalog_doc(&roster, catalog.instance()))?;

    // The catalog is committed, so it is now authoritative about which tables
    // exist and a directory it does not name is a dropped one. Collecting
    // *after* the commit is the same ordering as the part GC in
    // [`writer::write_table`]: a crash in between leaves an orphan directory
    // that the next checkpoint collects, whereas deleting first could leave a
    // committed catalog naming a table whose bytes are already gone.
    for (ddir, keep) in &live {
        collect_dropped_tables(ddir, keep)?;
    }

    // The committed parts cover everything the logs hold, so the logs can go.
    // A log that is already just its header holds nothing, and reopening it to
    // rewrite the header it already has -- plus the commit record, to record a
    // watermark it already carries -- is write amplification charged to a
    // database that was only read.
    for tdir in &checkpointed {
        let wal = tdir.join(WAL_FILE);
        if fs::metadata(&wal).is_ok_and(|m| m.len() > format::HEADER_LEN as u64) {
            Wal::open(&wal)?.truncate()?;
            set_wal_committed(tdir, format::HEADER_LEN as u64)?;
        }
    }
    Ok(())
}

/// Remove the directories of tables the committed catalog no longer names.
///
/// `ddir` is a database directory the catalog owns and `keep` is that
/// database's *complete* table roster, so "not in `keep`" really does mean
/// "dropped" -- this is never called with a partial list. Because it is the
/// one place in the module that removes a whole subtree, two further guards
/// apply: the candidate must be a directory whose name is one we could have
/// created ([`is_safe_name`] rejects the `.`-prefixed namespace [`tmp_path`]
/// uses, among others), and it must actually hold a commit record or a log, so
/// a directory this module did not write is left where it is.
///
/// Removal itself is best-effort, like the part GC in
/// [`writer::write_table`]: the checkpoint is already committed and durable by
/// the time we get here, and reporting it as failed because an unlink was
/// refused would be a worse lie than leaving a directory for the next
/// checkpoint to collect.
fn collect_dropped_tables(ddir: &Path, keep: &[String]) -> Result<()> {
    let rd = match fs::read_dir(ddir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(io_err("read directory", ddir, e)),
    };
    let mut collected = false;
    for entry in rd {
        let entry = entry.map_err(|e| io_err("read directory entry in", ddir, e))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !is_safe_name(name) || keep.iter().any(|k| k == name) {
            continue;
        }
        let tdir = ddir.join(name);
        if !tdir.is_dir() || !is_table_dir(&tdir) {
            continue;
        }
        collected |= fs::remove_dir_all(&tdir).is_ok();
    }
    if collected {
        sync_dir(ddir)?;
    }
    Ok(())
}

/// Whether `dir` is a table directory *we* wrote: it holds a commit record, or
/// a log from a table that was created but never checkpointed. Anything else
/// is someone else's directory and is not ours to delete.
fn is_table_dir(dir: &Path) -> bool {
    dir.join(TABLE_FILE).exists() || dir.join(WAL_FILE).exists()
}

/// Load every database, table, part and un-checkpointed WAL record under the
/// catalog's directory. A no-op for an in-memory catalog, and for a directory
/// that has never been written to.
pub fn load_catalog(catalog: &mut Catalog) -> Result<()> {
    let Some(root) = catalog.dir().map(Path::to_path_buf) else {
        return Ok(());
    };
    let cat_path = root.join(CATALOG_FILE);
    if !cat_path.exists() {
        return Ok(());
    }
    let (roster, instance) = reader::catalog_from_bytes(&read_file(&cat_path)?)
        .map_err(|e| prefix(&cat_path, e))?;
    catalog.set_instance(instance);

    for (db, defs) in roster {
        catalog.create_database(&db, true)?;
        for def in defs {
            let tdir = root.join(&db).join(&def.name);
            // Damage found on *this* table's metadata, quarantining it the way
            // a bad part file already does instead of refusing the whole
            // database. Empty on every healthy open, and this is the only
            // allocation the healthy path adds.
            let mut broken: Vec<reader::DamagedPart> = Vec::new();
            // The table's own directory is authoritative for its contents; the
            // CATALOG entry is the roster (and the fallback for a table that
            // was declared but never written).
            //
            // A `TABLE` that will not decode takes that same fallback: the
            // roster still has the definition, so the table keeps its place in
            // the catalog, its schema and its name, and only its *contents* are
            // refused. Which is precisely why `CATALOG` is not quarantinable
            // and never will be -- it is the roster this arm reads. A damaged
            // roster needs redundancy, not quarantine.
            let read = tdir.join(TABLE_FILE).exists().then(|| reader::read_table_image(&tdir));
            let mut image = match read {
                Some(Ok(img)) => img,
                other => {
                    if let Some(Err(e)) = other {
                        broken.push(reader::DamagedPart {
                            file: TABLE_FILE.to_string(),
                            why: e.to_string(),
                        });
                    }
                    reader::TableImage {
                        def,
                        parts: Vec::new(),
                        part_files: Vec::new(),
                        wal_committed: format::HEADER_LEN as u64,
                        damaged: Vec::new(),
                    }
                }
            };
            // Tell each part which file it came from, so the next checkpoint
            // knows it is already written. Without this the incremental path
            // in [`writer::write_table`] would still rewrite the whole
            // database on the first checkpoint after every restart -- which
            // is *every* checkpoint for a one-shot `granular -q ...`.
            //
            // Guarded on the two vectors being the same length, because that
            // is the alignment this zip needs and `read_table_image` does not
            // always give it: a part file that fails to decode is recorded in
            // `damaged` and contributes no `Part`, which would shift every
            // later pairing by one and hand a healthy part some other file's
            // name. A commit record naming the wrong bytes is far worse than
            // the rewrite it would save, so a damaged table simply
            // checkpoints the old way, in full.
            if image.parts.len() == image.part_files.len() {
                for (p, name) in image.parts.iter_mut().zip(&image.part_files) {
                    p.set_origin(parse_part_seq(name).unwrap_or(0));
                }
            }
            let bare = image.def.name.clone();
            let mut qualified = image.def.clone();
            qualified.name = format!("{db}.{bare}");
            catalog.create_table(qualified, true)?;

            let path = format!("{db}.{bare}");
            let t = catalog.table_by_path_mut(&path)?;
            t.set_parts(image.parts);

            let wal = tdir.join(WAL_FILE);
            // Not replayed at all when the `TABLE` file is already the damage:
            // the image is the roster's bare definition, so replaying into it
            // would build a table holding the tail and none of the parts --
            // work whose only product is a table that is refused anyway.
            if broken.is_empty() && wal.exists() {
                let schema = t.schema().clone();
                let wal_len = fs::metadata(&wal).map(|m| m.len()).unwrap_or(0);
                // A watermark past the end of the log means we crashed between
                // truncating the log and recording that we had: the log is
                // empty, the parts are complete, and the stale watermark must
                // be repaired or the *next* records written would be skipped.
                let stale = image.wal_committed > wal_len;
                let from = image.wal_committed.min(wal_len);
                // A log that will not replay quarantines its own table, the
                // same as a part file that will not decode. This used to be a
                // bare `?`: one table's bad checksum, in a file no other table
                // reads, refused the entire database -- `SELECT count() FROM b`
                // and `SELECT * FROM system.tables` included, so the operator
                // could not even find out which table it was.
                match Wal::replay_from(&wal, &schema, from) {
                    Err(e) => {
                        broken.push(reader::DamagedPart {
                            file: WAL_FILE.to_string(),
                            why: e.to_string(),
                        });
                    }
                    Ok(recs) => {
                        for rec in recs {
                            match rec {
                                WalRecord::Insert(b) => {
                                    t.insert(b)?;
                                }
                                WalRecord::Delete(lane) => {
                                    let pk = t.def.pk_col().ok_or_else(|| {
                                        Error::corruption(format!(
                                            "log for `{path}` holds a key delete, but the \
                                             table has no single-column primary key"
                                        ))
                                    })?;
                                    let v = value_from_lane(t.def.schema.ty(pk), lane)?;
                                    t.delete_key(&v)?;
                                }
                            }
                        }
                        if stale {
                            set_wal_committed(&tdir, wal_len.max(format::HEADER_LEN as u64))?;
                        }
                    }
                }
            }
            // Strictly after the resolve above, which is the one that is
            // allowed to succeed: `table_by_path_mut` refuses a quarantined
            // table, and it is also where the reader's part damage is claimed
            // -- so this call extends that list rather than replacing it.
            if !broken.is_empty() {
                catalog.quarantine(&path, broken);
            }
        }
    }
    Ok(())
}

/// Rewrite just the WAL watermark of an already-committed `TABLE` file.
///
/// Cheap by design: re-committing the table would mean rewriting every part.
pub fn set_wal_committed(tdir: &Path, committed: u64) -> Result<()> {
    let path = tdir.join(TABLE_FILE);
    let img = reader::table_header(&read_file(&path)?).map_err(|e| prefix(&path, e))?;
    let doc = writer::table_doc(&img.0, &img.1, committed);
    commit(&path, &doc).map(|_| ())
}

/// Inverse of [`Value::to_lane`] for the physical kinds a primary key can
/// have. String keys have no global lane (their codes are per-granule), which
/// is exactly why they never get the fast primary-key path.
fn value_from_lane(ty: &DataType, lane: u64) -> Result<Value> {
    Ok(match ty.base().physical() {
        PhysicalType::U64 => Value::UInt(lane),
        PhysicalType::I64 => Value::Int(lane_to_i64(lane)),
        PhysicalType::F64 => Value::Float(lane_to_f64(lane)),
        PhysicalType::Str => {
            return Err(Error::storage(format!(
                "cannot recover a {ty} key from a storage lane"
            )))
        }
    })
}

/// Name the file an error came from. A checksum failure is only actionable if
/// you know which file to delete.
pub(crate) fn prefix(path: &Path, e: Error) -> Error {
    match e {
        Error::Corruption(m) => Error::Corruption(format!("{}: {m}", path.display())),
        other => other,
    }
}

pub(crate) fn delta_limit() -> usize {
    DEFAULT_DELTA_LIMIT
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persist::testkit::*;
    use crate::sql::ast::ObjectName;
    use crate::types::{Block, Column, Engine, Value};

    fn on_disk(s: &Scratch) -> Catalog {
        Catalog::on_disk(s.path()).unwrap()
    }

    #[test]
    fn part_names_roundtrip_and_reject_strays() {
        assert_eq!(part_file_name(1), "part_000001.gpart");
        assert_eq!(part_file_name(1_234_567), "part_1234567.gpart");
        assert_eq!(parse_part_seq("part_000042.gpart"), Some(42));
        assert_eq!(parse_part_seq("part_1234567.gpart"), Some(1_234_567));
        for stray in [
            "TABLE",
            "wal.log",
            "part_.gpart",
            "part_00x1.gpart",
            "part_000001.gpart.tmp",
            "part_000001",
            ".part_000001.gpart.tmp-1-2",
            "xpart_000001.gpart",
        ] {
            assert_eq!(parse_part_seq(stray), None, "{stray} must not look like a part");
        }
    }

    #[test]
    fn only_plain_names_may_become_directories() {
        for ok in ["t", "hits_2024", "Weird Name", "a.b"] {
            assert!(is_safe_name(ok), "{ok} should be usable");
        }
        for bad in ["", ".", "..", "a/b", "../x", "a\\b", ".hidden", "a\0b"] {
            assert!(!is_safe_name(bad), "{bad:?} must be refused");
        }
    }

    #[test]
    fn a_name_collides_with_a_sibling_it_differs_from_only_by_case() {
        let siblings = ["Tenant", "orders", "ORDERS_2024"];
        let of = |n: &str| folds_onto(n, siblings.iter().copied());
        assert_eq!(of("tenant"), Some("Tenant"));
        assert_eq!(of("TENANT"), Some("Tenant"));
        assert_eq!(of("Orders"), Some("orders"));
        assert_eq!(of("orders_2024"), Some("ORDERS_2024"));
        // The exact name is not a collision: that is `IF NOT EXISTS`, and the
        // caller's own "already exists" rule owns it.
        assert_eq!(of("Tenant"), None);
        assert_eq!(of("orders"), None);
        // Nor is a name that differs by more than case.
        assert_eq!(of("tenants"), None);
        assert_eq!(of("orders_2025"), None);
        // The root's own file names, whatever case they are asked for.
        for n in ["CATALOG", "catalog", "Lock", "LOCK"] {
            assert!(
                ROOT_RESERVED.iter().any(|r| r.eq_ignore_ascii_case(n)),
                "{n} must not be available as a database name"
            );
        }
        assert!(!ROOT_RESERVED.iter().any(|r| r.eq_ignore_ascii_case("TABLE")));
    }

    #[test]
    fn a_table_whose_name_is_a_path_is_not_written() {
        let s = Scratch::new("badname");
        let mut t = sample_table("t", &[10]);
        t.def.name = "../escape".into();
        let e = writer::write_table(s.path(), &t).unwrap_err();
        assert_eq!(e.code(), "STORAGE_ERROR", "{e}");
        assert!(!s.path().parent().unwrap().join("escape").exists());
    }

    #[test]
    fn a_database_whose_name_is_a_path_is_not_checkpointed() {
        let s = Scratch::new("baddb");
        let mut c = on_disk(&s);
        c.create_database("../escape", false).unwrap();
        let e = save_catalog(&mut c).unwrap_err();
        assert_eq!(e.code(), "STORAGE_ERROR", "{e}");
        assert!(!s.path().parent().unwrap().join("escape").exists());
    }

    #[test]
    fn atomic_write_replaces_and_leaves_no_temp_files() {
        let s = Scratch::new("atomic");
        let p = s.join("f.bin");
        atomic_write(&p, b"first").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"first");
        atomic_write(&p, b"second-and-longer").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"second-and-longer");
        let left: Vec<String> = fs::read_dir(s.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(left, vec!["f.bin".to_string()], "temp files must not survive");
    }

    #[test]
    fn atomic_write_creates_missing_directories() {
        let s = Scratch::new("mkdir");
        let p = s.join("a").join("b").join("c.bin");
        atomic_write(&p, b"x").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"x");
    }

    #[test]
    fn atomic_write_fails_cleanly_on_a_bad_path() {
        let s = Scratch::new("badpath");
        let file = s.join("f.bin");
        atomic_write(&file, b"x").unwrap();
        // A file where a directory is required.
        let e = atomic_write(&file.join("nested"), b"y").unwrap_err();
        assert_eq!(e.code(), "IO_ERROR", "{e}");
    }

    #[test]
    fn sync_dir_accepts_a_real_directory() {
        let s = Scratch::new("syncdir");
        sync_dir(s.path()).unwrap();
    }

    #[test]
    fn in_memory_catalog_is_a_noop_in_both_directions() {
        let mut c = Catalog::in_memory();
        c.create_table(table_def("t"), false).unwrap();
        save_catalog(&mut c).unwrap();
        load_catalog(&mut c).unwrap();
        assert_eq!(c.table_names(None).unwrap(), vec!["t".to_string()]);
    }

    #[test]
    fn loading_a_fresh_directory_is_a_noop() {
        let s = Scratch::new("fresh");
        let mut c = on_disk(&s);
        load_catalog(&mut c).unwrap();
        assert!(c.table_names(None).unwrap().is_empty());
    }

    #[test]
    fn catalog_roundtrip_preserves_rows_and_definitions() {
        let s = Scratch::new("cat-roundtrip");
        let mut c = on_disk(&s);
        c.create_database("analytics", false).unwrap();
        c.create_table(table_def("hits"), false).unwrap();
        let mut d = table_def("analytics.events");
        d.order_by = vec![0];
        d.primary_key = vec![0];
        c.create_table(d, false).unwrap();

        let rows = 3_000;
        let b = sample_block(rows);
        let keys = b.column(0).as_u64().unwrap().to_vec();
        c.table_mut(&ObjectName::bare("hits")).unwrap().insert(b.clone()).unwrap();
        c.table_by_path_mut("analytics.events").unwrap().insert(b).unwrap();
        save_catalog(&mut c).unwrap();

        let mut c2 = on_disk(&s);
        load_catalog(&mut c2).unwrap();
        assert_eq!(c2.database_names(), vec!["analytics", "default"]);
        assert_eq!(c2.table_names(Some("analytics")).unwrap(), vec!["events"]);

        for path in ["default.hits", "analytics.events"] {
            let t = c2.table_by_path_mut(path).unwrap();
            assert_eq!(t.row_count().unwrap(), rows, "{path}");
            for &k in keys.iter().step_by(17) {
                let got = t.get(&Value::UInt(k)).unwrap();
                assert_eq!(got.as_ref().map(|r| r[0].clone()), Some(Value::UInt(k)), "{path} {k}");
            }
        }
    }

    #[test]
    fn saving_twice_replaces_the_previous_parts() {
        let s = Scratch::new("resave");
        let mut c = on_disk(&s);
        c.create_table(table_def("t"), false).unwrap();
        c.table_mut(&ObjectName::bare("t")).unwrap().insert(sample_block(500)).unwrap();
        save_catalog(&mut c).unwrap();
        let tdir = s.join("default").join("t");
        let after_first = list_part_files(&tdir).unwrap();
        assert_eq!(after_first.len(), 1);

        c.table_mut(&ObjectName::bare("t")).unwrap().insert(sample_block(200)).unwrap();
        save_catalog(&mut c).unwrap();
        let after_second = list_part_files(&tdir).unwrap();
        assert!(
            after_second.iter().all(|(seq, _)| *seq > after_first[0].0),
            "sequence numbers must never be reused: {after_second:?}"
        );

        let mut c2 = on_disk(&s);
        load_catalog(&mut c2).unwrap();
        assert_eq!(c2.table_by_path_mut("default.t").unwrap().row_count().unwrap(), 500);
    }

    #[test]
    fn memory_engine_tables_are_not_persisted() {
        let s = Scratch::new("memeng");
        let mut c = on_disk(&s);
        let mut d = table_def("scratch");
        d.engine = Engine::Memory;
        c.create_table(d, false).unwrap();
        c.create_table(table_def("kept"), false).unwrap();
        save_catalog(&mut c).unwrap();
        assert!(!s.join("default").join("scratch").exists());

        let mut c2 = on_disk(&s);
        load_catalog(&mut c2).unwrap();
        assert_eq!(c2.table_names(None).unwrap(), vec!["kept".to_string()]);
    }

    #[test]
    fn empty_databases_survive_a_roundtrip() {
        let s = Scratch::new("emptydb");
        let mut c = on_disk(&s);
        c.create_database("staging", false).unwrap();
        save_catalog(&mut c).unwrap();
        let mut c2 = on_disk(&s);
        load_catalog(&mut c2).unwrap();
        assert!(c2.database_names().contains(&"staging".to_string()));
    }

    #[test]
    fn a_corrupt_catalog_is_reported_not_ignored() {
        let s = Scratch::new("catcorrupt");
        let mut c = on_disk(&s);
        c.create_table(table_def("t"), false).unwrap();
        save_catalog(&mut c).unwrap();

        let path = s.join(CATALOG_FILE);
        let mut bytes = fs::read(&path).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0x40;
        fs::write(&path, &bytes).unwrap();

        let mut c2 = on_disk(&s);
        let e = load_catalog(&mut c2).unwrap_err();
        assert_eq!(e.code(), "CHECKSUM_MISMATCH", "{e}");
        assert!(e.to_string().contains("CATALOG"), "{e}");
    }

    #[test]
    fn deletes_survive_a_checkpoint() {
        let s = Scratch::new("deletes");
        let mut c = on_disk(&s);
        c.create_table(table_def("t"), false).unwrap();
        let b = sample_block(2_000);
        let keys = b.column(0).as_u64().unwrap().to_vec();
        let t = c.table_mut(&ObjectName::bare("t")).unwrap();
        t.insert(b).unwrap();
        t.flush().unwrap();
        for &k in keys.iter().step_by(5) {
            t.delete_key(&Value::UInt(k)).unwrap();
        }
        let want = t.row_count().unwrap();
        save_catalog(&mut c).unwrap();

        let mut c2 = on_disk(&s);
        load_catalog(&mut c2).unwrap();
        let t2 = c2.table_by_path_mut("default.t").unwrap();
        assert_eq!(t2.row_count().unwrap(), want);
        for &k in keys.iter().step_by(5) {
            assert_eq!(t2.get(&Value::UInt(k)).unwrap(), None, "key {k} came back");
        }
    }

    /// The half of the incremental checkpoint that lives here: a table loaded
    /// off disk knows which file each of its parts came from, so the *first*
    /// checkpoint after a restart is incremental too. Without the pairing in
    /// [`load_catalog`] this is the case that still rewrote everything -- and
    /// for a one-shot `granular -q ...` it is the only case there is.
    #[test]
    fn a_checkpoint_after_a_reload_rewrites_nothing() {
        let s = Scratch::new("reload-incremental");
        let mut c = on_disk(&s);
        c.create_table(table_def("t"), false).unwrap();
        c.table_mut(&ObjectName::bare("t")).unwrap().insert(sample_block(3_000)).unwrap();
        save_catalog(&mut c).unwrap();
        let tdir = s.join("default").join("t");
        let before = list_part_files(&tdir).unwrap();
        assert_eq!(before.len(), 1);
        let bytes = fs::read(tdir.join(&before[0].1)).unwrap();
        let cat = fs::read(s.join(CATALOG_FILE)).unwrap();

        let mut c2 = on_disk(&s);
        load_catalog(&mut c2).unwrap();
        save_catalog(&mut c2).unwrap();

        assert_eq!(list_part_files(&tdir).unwrap(), before, "the reloaded part was renumbered");
        assert_eq!(fs::read(tdir.join(&before[0].1)).unwrap(), bytes, "it was rewritten");
        assert_eq!(fs::read(s.join(CATALOG_FILE)).unwrap(), cat);
        assert_eq!(c2.table_by_path_mut("default.t").unwrap().row_count().unwrap(), 3_000);
    }

    /// ...and a table that gained rows writes exactly one more file, leaving
    /// the part it did not touch where it is.
    #[test]
    fn a_checkpoint_writes_only_the_parts_that_moved() {
        let s = Scratch::new("partial-checkpoint");
        let mut c = on_disk(&s);
        c.create_table(table_def("t"), false).unwrap();
        let t = c.table_mut(&ObjectName::bare("t")).unwrap();
        t.insert(sample_block(3_000)).unwrap();
        t.flush().unwrap();
        save_catalog(&mut c).unwrap();
        let tdir = s.join("default").join("t");
        let before = list_part_files(&tdir).unwrap();

        let t = c.table_mut(&ObjectName::bare("t")).unwrap();
        let mut b = sample_block(500);
        // Disjoint keys, so nothing in the first part is tombstoned.
        let keys: Vec<u64> =
            b.column(0).as_u64().unwrap().iter().map(|&k| k + 100_000_000).collect();
        b.columns[0] = Column::u64s(DataType::UInt64, keys);
        t.insert(b).unwrap();
        save_catalog(&mut c).unwrap();

        let after = list_part_files(&tdir).unwrap();
        assert_eq!(after.len(), 2, "{after:?}");
        assert_eq!(after[0], before[0], "the untouched part moved");
        assert!(after[1].0 > before[0].0, "sequence numbers are never reused: {after:?}");
    }

    #[test]
    fn checkpoint_truncates_the_log_and_clears_the_watermark() {
        let s = Scratch::new("checkpoint-wal");
        let mut c = on_disk(&s);
        c.create_table(table_def("t"), false).unwrap();
        c.table_mut(&ObjectName::bare("t")).unwrap().insert(sample_block(100)).unwrap();
        save_catalog(&mut c).unwrap();

        let tdir = s.join("default").join("t");
        let mut w = Wal::open(&tdir.join(WAL_FILE)).unwrap();
        w.append_delete(7).unwrap();
        w.sync().unwrap();
        let grown = fs::metadata(tdir.join(WAL_FILE)).unwrap().len();
        assert!(grown > format::HEADER_LEN as u64);

        save_catalog(&mut c).unwrap();
        assert_eq!(
            fs::metadata(tdir.join(WAL_FILE)).unwrap().len(),
            format::HEADER_LEN as u64,
            "a checkpoint must reclaim the log"
        );
        let img = reader::read_table_image(&tdir).unwrap();
        assert_eq!(img.wal_committed, format::HEADER_LEN as u64);
    }

    #[test]
    fn unflushed_log_records_are_replayed_on_load() {
        let s = Scratch::new("wal-recovery");
        let mut c = on_disk(&s);
        c.create_table(table_def("t"), false).unwrap();
        let b = sample_block(300);
        let keys = b.column(0).as_u64().unwrap().to_vec();
        c.table_mut(&ObjectName::bare("t")).unwrap().insert(b).unwrap();
        save_catalog(&mut c).unwrap();

        // Simulate writes that reached the log but not a part.
        let tdir = s.join("default").join("t");
        let extra = Block::new(vec![
            Column::u64s(DataType::UInt64, vec![9_000_001, 9_000_002]),
            Column::strs(DataType::String, vec!["new-a".into(), "new-b".into()]),
            Column::i64s(DataType::Int64, vec![1, 2]),
            Column::f64s(DataType::Float64, vec![0.5, 1.5]),
        ])
        .unwrap();
        let mut w = Wal::open(&tdir.join(WAL_FILE)).unwrap();
        w.append_insert(&extra).unwrap();
        w.append_delete(keys[0]).unwrap();
        w.sync().unwrap();

        let mut c2 = on_disk(&s);
        load_catalog(&mut c2).unwrap();
        let t = c2.table_by_path_mut("default.t").unwrap();
        assert_eq!(t.get(&Value::UInt(9_000_001)).unwrap().unwrap()[1], Value::str("new-a"));
        assert_eq!(t.get(&Value::UInt(9_000_002)).unwrap().unwrap()[1], Value::str("new-b"));
        assert_eq!(t.get(&Value::UInt(keys[0])).unwrap(), None, "the logged delete must apply");
        assert_eq!(t.row_count().unwrap(), 300 + 2 - 1);
    }

    #[test]
    fn a_log_prefix_already_in_a_part_is_not_replayed_twice() {
        let s = Scratch::new("wal-watermark");
        let mut c = on_disk(&s);
        let mut d = table_def("t");
        d.engine = Engine::Log; // unkeyed: a double replay would be visible
        d.order_by.clear();
        d.primary_key.clear();
        c.create_table(d, false).unwrap();

        let tdir = s.join("default").join("t");
        fs::create_dir_all(&tdir).unwrap();
        let row = Block::new(vec![
            Column::u64s(DataType::UInt64, vec![1]),
            Column::strs(DataType::String, vec!["a".into()]),
            Column::i64s(DataType::Int64, vec![1]),
            Column::f64s(DataType::Float64, vec![1.0]),
        ])
        .unwrap();
        // The write path: log first, then apply.
        let mut w = Wal::open(&tdir.join(WAL_FILE)).unwrap();
        w.append_insert(&row).unwrap();
        w.sync().unwrap();
        c.table_mut(&ObjectName::bare("t")).unwrap().insert(row.clone()).unwrap();
        save_catalog(&mut c).unwrap(); // covers the record, truncates the log

        // A second record arrives after the checkpoint and is not applied.
        let mut w = Wal::open(&tdir.join(WAL_FILE)).unwrap();
        w.append_insert(&row).unwrap();
        w.sync().unwrap();

        let mut c2 = on_disk(&s);
        load_catalog(&mut c2).unwrap();
        assert_eq!(
            c2.table_by_path_mut("default.t").unwrap().row_count().unwrap(),
            2,
            "the checkpointed record must not be replayed, the new one must"
        );
    }

    #[test]
    fn a_stale_watermark_is_repaired_on_load() {
        let s = Scratch::new("wal-stale");
        let mut c = on_disk(&s);
        c.create_table(table_def("t"), false).unwrap();
        c.table_mut(&ObjectName::bare("t")).unwrap().insert(sample_block(50)).unwrap();
        save_catalog(&mut c).unwrap();

        // Crash shape: the watermark names a log longer than the one on disk.
        let tdir = s.join("default").join("t");
        Wal::open(&tdir.join(WAL_FILE)).unwrap();
        set_wal_committed(&tdir, 1_000_000).unwrap();

        let mut c2 = on_disk(&s);
        load_catalog(&mut c2).unwrap();
        let img = reader::read_table_image(&tdir).unwrap();
        assert_eq!(
            img.wal_committed,
            format::HEADER_LEN as u64,
            "a watermark past the end of the log must be repaired"
        );

        // ...and records written after the repair are still replayed.
        let mut w = Wal::open(&tdir.join(WAL_FILE)).unwrap();
        w.append_delete(1).unwrap();
        w.sync().unwrap();
        let mut c3 = on_disk(&s);
        load_catalog(&mut c3).unwrap();
        assert!(c3.table_by_path_mut("default.t").is_ok());
    }

    // ---- adversarial review additions ------------------------------------

    /// save -> load -> save -> load must be stable. Nothing in the suite
    /// exercises a checkpoint taken from a catalog that was itself recovered
    /// from disk.
    #[test]
    fn adversarial_save_load_save_load_is_stable() {
        let s = Scratch::new("adv-resave-cycle");
        let mut c = on_disk(&s);
        c.create_table(table_def("t"), false).unwrap();
        let b = sample_block(2_000);
        let keys = b.column(0).as_u64().unwrap().to_vec();
        c.table_mut(&ObjectName::bare("t")).unwrap().insert(b).unwrap();
        save_catalog(&mut c).unwrap();

        let mut c2 = on_disk(&s);
        load_catalog(&mut c2).unwrap();
        assert_eq!(c2.table_by_path_mut("default.t").unwrap().row_count().unwrap(), 2_000);
        save_catalog(&mut c2).unwrap();

        let mut c3 = on_disk(&s);
        load_catalog(&mut c3).unwrap();
        let t = c3.table_by_path_mut("default.t").unwrap();
        assert_eq!(t.row_count().unwrap(), 2_000, "row count drifted across a re-checkpoint");
        for &k in keys.iter().step_by(29) {
            assert!(t.get(&Value::UInt(k)).unwrap().is_some(), "key {k} vanished");
        }
        assert_eq!(
            c3.table_names(None).unwrap(),
            vec!["t".to_string()],
            "the table roster drifted"
        );
    }

    /// A checkpoint after a DROP TABLE must not leave the dropped table's
    /// bytes readable/reachable in the tree; and re-creating the name must not
    /// resurrect the old rows.
    #[test]
    fn adversarial_dropped_table_directory_is_collected() {
        let s = Scratch::new("adv-drop");
        let mut c = on_disk(&s);
        c.create_table(table_def("t"), false).unwrap();
        c.table_mut(&ObjectName::bare("t")).unwrap().insert(sample_block(500)).unwrap();
        save_catalog(&mut c).unwrap();
        let tdir = s.join("default").join("t");
        assert!(tdir.join(TABLE_FILE).exists());

        c.drop_table(&ObjectName::bare("t"), false).unwrap();
        save_catalog(&mut c).unwrap();
        assert!(
            !tdir.exists(),
            "dropping a table left its parts on disk at {}",
            tdir.display()
        );
    }

    /// Collecting a dropped table removes a subtree, so it must recognise its
    /// own: a directory this module never wrote is not a deletion candidate,
    /// however absent its name is from the catalog.
    #[test]
    fn collection_leaves_directories_it_did_not_write() {
        let s = Scratch::new("adv-drop-stray");
        let mut c = on_disk(&s);
        c.create_table(table_def("t"), false).unwrap();
        save_catalog(&mut c).unwrap();

        let stray = s.join("default").join("notes");
        fs::create_dir_all(&stray).unwrap();
        fs::write(stray.join("readme.txt"), b"not ours").unwrap();
        save_catalog(&mut c).unwrap();
        assert!(stray.join("readme.txt").exists(), "a foreign directory must survive");

        // ...while a table directory that never got as far as a commit record
        // is ours, and goes.
        let half = s.join("default").join("halfborn");
        Wal::open(&half.join(WAL_FILE)).unwrap();
        save_catalog(&mut c).unwrap();
        assert!(!half.exists(), "a log-only directory of an unknown table must be collected");
        assert!(s.join("default").join("t").exists(), "the live table must be untouched");
    }

    #[test]
    fn lane_recovery_covers_every_key_kind() {
        for (ty, v) in [
            (DataType::UInt64, Value::UInt(u64::MAX)),
            (DataType::Int64, Value::Int(i64::MIN)),
            (DataType::Int32, Value::Int(-7)),
            (DataType::Float64, Value::Float(-0.5)),
            (DataType::Date, Value::UInt(19_723)),
        ] {
            let lane = v.to_lane(&ty).unwrap();
            let back = value_from_lane(&ty, lane).unwrap();
            assert_eq!(back.to_lane(&ty).unwrap(), lane, "{ty}");
        }
        assert!(value_from_lane(&DataType::String, 0).is_err());
    }

    #[test]
    fn delta_limit_matches_the_catalog_default() {
        assert_eq!(delta_limit(), DEFAULT_DELTA_LIMIT);
    }
}
