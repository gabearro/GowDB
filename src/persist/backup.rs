//! Backup, restore and verify: a whole database as one self-describing file.
//!
//! ## Why this exists, and why `cp -r` is not it
//!
//! A running database is a moving target. A directory walk reads the `TABLE`
//! commit record of one table and the part files of another, and a writer that
//! commits between those two reads leaves the copy naming a part file that the
//! walk had already passed (or one the GC has since unlinked). Two of eight
//! `cp -r` copies of a live instance were unopenable, and every one of them
//! exited 0. There is no way to fix that from outside the process, because the
//! consistency point is not a filesystem state -- it is a [`Snapshot`].
//!
//! So a backup is taken *from inside*: one [`Snapshot`] per table, pinned
//! before a byte is written. Parts are immutable, so the pin is the whole
//! argument -- once a snapshot is held, the bytes it names cannot change and a
//! concurrent writer publishing a new `PartSet` cannot affect what this archive
//! will contain.
//!
//! ## Layout
//!
//! Deliberately the part file's own shape, for the same reasons:
//!
//! ```text
//!   header                  MAGIC + version
//!   framed part body        | one independently checksummed frame per part,
//!   framed part body        | in the order the manifest names them
//!   ...
//!   framed manifest         databases -> tables -> TABLE docs + part refs
//!   footer                  offset of the manifest + version + checksum
//! ```
//!
//! The manifest comes last and the footer points at it, so `verify` and
//! `restore` open the file, read [`format::FOOTER_LEN`] bytes and jump -- no
//! scan, and no need to hold the archive in memory. The file is mapped, not
//! read: a 100 GB archive costs page cache the kernel can reclaim rather than
//! resident heap it cannot.
//!
//! A table's entry stores the *exact bytes* of a `TABLE` commit record
//! ([`writer::table_doc`]), and part bodies are the exact bytes
//! [`writer::write_part_with`] would publish. Restore is therefore a copy, not
//! a translation: it writes the frames out under the names the record already
//! carries and the result is byte-for-byte a checkpointed database directory,
//! which the ordinary loader opens with no knowledge that a backup happened.
//! There is no second serializer to drift from the first.
//!
//! ## Incremental
//!
//! Falls out of two properties that already hold: parts are immutable, and
//! [`writer::part_bytes_with`] is deterministic (pinned by
//! `writer::encoding_is_deterministic`). So an unchanged part serializes to
//! *byte-identical* output on every backup, and an archive can name it by
//! identity instead of storing it again.
//!
//! `BACKUP ... INCREMENTAL FROM 'base'` records, for every part, the same
//! manifest entry a full backup would -- but stores the body only when the
//! base chain does not already have it. What it saves is archive bytes and the
//! I/O to write them; it does **not** save serializing the part, because the
//! delete mask lives in the `PartSet` rather than in the part file and two
//! runs of a part with different masks are different bytes. Deciding "already
//! there" without producing the bytes would mean trusting `PartSet::origin`,
//! which is documented as a hint a checkpoint must validate against the
//! directory -- fine when the cost of being wrong is a wasted rewrite, not
//! when it is a backup silently missing rows.
//!
//! Identity is `(length, checksum, live rows)`. That is not a cryptographic
//! digest and a determined forger can collide it; two *distinct* parts
//! agreeing on all three by accident is around 2^-64 per pair, which at 10^4
//! parts is ~5e-12. Every resolution is re-checked against all three fields
//! after the body is fetched, so a mismatched index is detected -- a genuine
//! collision is not.
//!
//! ## What verify actually verifies
//!
//! An archive that reports healthy and restores broken is worse than no
//! verify, so `verify` does the whole of what `restore` does except write:
//! every frame's checksum, every part decoded through
//! [`reader::part_from_bytes`] (the same defensive decoder a real open uses),
//! every `TABLE` record parsed, every part the manifest names resolved to a
//! body, and every name it would create checked for being a legal directory
//! entry. It is deliberately not a header check.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::common::{Error, FastMap, Result};
use crate::persist::format::{self, Reader, Writer};
use crate::persist::mmap::Mmap;
use crate::persist::{reader, store, writer};
use crate::storage::part::Snapshot;
use crate::types::TableDef;

/// Conventional extension. Not enforced -- the footer's magic is what says
/// whether a file is an archive -- but it is what the error messages suggest.
pub const ARCHIVE_EXT: &str = "gbak";

/// How many archives a base chain may have. A chain is walked eagerly by
/// `restore` and `verify`, so this is the bound on both the recursion and the
/// number of mappings held at once.
const MAX_CHAIN: usize = 32;

/// Plausibility ceiling on manifest structure counts, matching
/// `reader`'s reasoning: it bounds counts of *structures* and nothing that
/// grows with the data.
const MAX_COUNT: u64 = 1 << 24;

/// One table to archive, with the snapshot its rows are read from.
///
/// The snapshot is taken by the caller, before this struct exists, and every
/// table's is taken in the same critical section: that is what makes an
/// archive one point in time across tables rather than per table.
pub struct Source<'a> {
    pub db: &'a str,
    pub def: &'a TableDef,
    pub snap: Snapshot,
}

/// What a backup, verify or restore touched. Every field is exact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub tables: usize,
    pub parts: usize,
    /// Live rows -- what a `SELECT count()` on the restored database returns.
    pub rows: u64,
    /// Size of the archive file itself.
    pub bytes: u64,
    /// Parts an incremental archive resolved from its base instead of storing.
    pub reused: usize,
}

// ---------------------------------------------------------------------------
// writing
// ---------------------------------------------------------------------------

/// Write the snapshots in `src` to a new archive at `path`.
///
/// `dbs` is the complete database roster and `src` the tables to archive, in
/// any order. The roster is passed separately rather than derived from `src`
/// so that a database holding *no* tables still round-trips: the root
/// `CATALOG` remembers empty databases on purpose, and an archive that dropped
/// them would quietly lose a `CREATE DATABASE`.
///
/// Published the way every other file in this engine is: to a temp name beside
/// the target, fsynced, renamed, and the directory fsynced. A crash at any
/// point leaves either no archive or a complete one -- never the half file that
/// a `cp` interrupted at the wrong moment leaves, which is the failure this
/// whole module exists to replace.
pub fn write_archive(
    path: &Path,
    dbs: &[String],
    src: &[Source<'_>],
    base: Option<&Path>,
) -> Result<Report> {
    // The base is opened first, so a typo in its path fails before anything is
    // written and the operator still has the archive they meant to extend.
    let dedupe = match base {
        Some(b) => Some(Chain::open(&resolve_base(path, b))?),
        None => None,
    };

    let tmp = tmp_path(path);
    let mut sink = Sink::create(&tmp)?;
    let r = (|| -> Result<Report> {
        sink.raw({
            let mut w = Writer::with_capacity(format::HEADER_LEN);
            format::write_header(&mut w);
            &w.finish()
        })?;

        let mut rep = Report::default();
        let mut man = Writer::with_capacity(512 + src.len() * 256);
        man.str(&base.map_or(String::new(), |b| b.to_string_lossy().into_owned()));
        man.u64(now_unix());

        man.varint(dbs.len() as u64);
        for db in dbs {
            if !store::is_safe_name(db) {
                return Err(Error::storage(format!(
                    "database `{db}` cannot be archived: its name cannot be a directory name"
                )));
            }
            man.str(db);
            let mine = src.iter().filter(|s| s.db == db);
            man.varint(mine.clone().count() as u64);
            for s in mine {
                table_into(&mut sink, &mut man, s, dedupe.as_ref(), &mut rep)?;
            }
        }

        let body = man.finish();
        let at = sink.pos;
        sink.frame(&body)?;
        let mut w = Writer::with_capacity(format::FOOTER_LEN);
        format::write_footer(&mut w, at);
        sink.raw(&w.finish())?;
        rep.bytes = sink.pos;
        Ok(rep)
    })();

    match r {
        Ok(rep) => {
            sink.publish(path)?;
            Ok(rep)
        }
        Err(e) => {
            drop(sink);
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

fn table_into(
    sink: &mut Sink,
    man: &mut Writer,
    s: &Source<'_>,
    base: Option<&Chain>,
    rep: &mut Report,
) -> Result<()> {
    if !store::is_safe_name(s.db) || !store::is_safe_name(&s.def.name) {
        return Err(Error::storage(format!(
            "`{}.{}` cannot be archived: neither name may be used as a directory name",
            s.db, s.def.name
        )));
    }
    let set = s.snap.set();
    let n = s.snap.len();
    // Renumbered from 1 rather than carrying the live database's sequence
    // numbers: an archive is a fresh directory, and the numbers only have to
    // be unique and parseable by `store::parse_part_seq` so the first
    // checkpoint after a restore can reuse the files instead of rewriting
    // them. Dedupe is by content, so renaming costs nothing.
    let names: Vec<String> = (1..=n as u64).map(store::part_file_name).collect();
    man.bytes(&writer::table_doc(s.def, &names, format::HEADER_LEN as u64));
    man.varint(n as u64);
    for i in 0..n {
        let body = writer::part_bytes_with(s.snap.part(i), set.deletes(i))?;
        let rows = set.live_rows_of(i) as u64;
        let key = (body.len() as u64, format::checksum(&body), rows);
        man.u64(key.1);
        man.varint(rows);
        // The true length either way -- it is a third of the identity, so a
        // deduped entry that dropped it could not be looked up.
        man.varint(key.0);
        if base.is_some_and(|c| c.has(key)) {
            // Offset 0 is the "ask the base chain" marker. An inline body can
            // never be there: the archive opens with a header, so the first
            // frame starts at `HEADER_LEN`.
            man.varint(0);
            rep.reused += 1;
        } else {
            man.varint(sink.pos);
            sink.frame(&body)?;
        }
        rep.rows += rows;
        rep.parts += 1;
    }
    rep.tables += 1;
    Ok(())
}

/// Append-only file with a byte position, so a part body is streamed rather
/// than concatenated into an archive-sized buffer first.
struct Sink {
    f: std::io::BufWriter<std::fs::File>,
    pos: u64,
    tmp: PathBuf,
}

impl Sink {
    fn create(tmp: &Path) -> Result<Sink> {
        let dir = tmp.parent().unwrap_or_else(|| Path::new("."));
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| store::io_err("create directory", dir, e))?;
        }
        let f = std::fs::File::create(tmp).map_err(|e| store::io_err("create", tmp, e))?;
        Ok(Sink { f: std::io::BufWriter::new(f), pos: 0, tmp: tmp.to_path_buf() })
    }

    fn raw(&mut self, b: &[u8]) -> Result<()> {
        self.f.write_all(b).map_err(|e| store::io_err("write", &self.tmp, e))?;
        self.pos += b.len() as u64;
        Ok(())
    }

    /// `format::write_framed`, split so the body is not copied.
    ///
    /// The layout is that function's and has to track it; `frame_matches_format`
    /// is the test that says so.
    fn frame(&mut self, body: &[u8]) -> Result<()> {
        let mut w = Writer::with_capacity(16);
        w.varint(body.len() as u64);
        w.u64(format::checksum(body));
        self.raw(&w.finish())?;
        self.raw(body)
    }

    /// The four steps [`store::atomic_write`] takes, with the body already
    /// streamed: fsync the temp file, rename it over the target, fsync the
    /// directory.
    fn publish(self, path: &Path) -> Result<()> {
        let Sink { f, tmp, .. } = self;
        // Every failure below unlinks the temp file: a half-written archive
        // left in the target's directory is the exact artefact this module
        // exists to stop anyone finding and trusting.
        let r = (|| -> Result<()> {
            let f = f.into_inner().map_err(|e| store::io_err("flush", &tmp, e.into_error()))?;
            f.sync_all().map_err(|e| store::io_err("fsync", &tmp, e))?;
            drop(f);
            std::fs::rename(&tmp, path)
                .map_err(|e| store::io_err("rename into place", path, e))
        })();
        if r.is_err() {
            let _ = std::fs::remove_file(&tmp);
            return r;
        }
        store::sync_dir(path.parent().unwrap_or_else(|| Path::new(".")))
    }
}

/// A temp name beside the target, in the dot-prefixed namespace
/// [`store::is_safe_name`] refuses for a table, so a leftover can never be
/// mistaken for data. Mirrors `store::tmp_path`, which is private to that
/// module.
fn tmp_path(target: &Path) -> PathBuf {
    let stem = target.file_name().and_then(|s| s.to_str()).unwrap_or("archive");
    let dir = target.parent().unwrap_or_else(|| Path::new("."));
    dir.join(format!(".{stem}.tmp-{}", std::process::id()))
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

// ---------------------------------------------------------------------------
// reading
// ---------------------------------------------------------------------------

/// One archive's manifest, decoded.
struct Manifest {
    base: String,
    created: u64,
    dbs: Vec<(String, Vec<ArchivedTable>)>,
}

struct ArchivedTable {
    /// The `TABLE` commit record, verbatim, ready to write out.
    doc: Vec<u8>,
    def: TableDef,
    files: Vec<String>,
    parts: Vec<PartRef>,
}

#[derive(Clone, Copy)]
struct PartRef {
    crc: u64,
    rows: u64,
    len: u64,
    /// Frame offset in the archive that declares it, or 0 for "not stored
    /// here; resolve from the base chain by identity".
    at: u64,
}

impl PartRef {
    /// What names a part body across archives. See the module header on what
    /// this is and is not proof against.
    fn key(&self) -> (u64, u64, u64) {
        (self.len, self.crc, self.rows)
    }
}

/// An archive and everything its base chain can supply.
///
/// Mapped, not read: the index is offsets into mappings, so resolving a part
/// hands back a slice of the file rather than a copy of it.
struct Chain {
    maps: Vec<Arc<Mmap>>,
    mans: Vec<Manifest>,
    /// `(len, crc, rows) -> (archive, frame offset)`, for every body any
    /// archive in the chain stores inline.
    index: FastMap<(u64, u64, u64), (usize, u64)>,
}

impl Chain {
    /// Open `path` and, transitively, every archive it names as a base.
    fn open(path: &Path) -> Result<Chain> {
        let mut c = Chain { maps: Vec::new(), mans: Vec::new(), index: FastMap::default() };
        let mut next = Some(path.to_path_buf());
        while let Some(p) = next {
            if c.maps.len() >= MAX_CHAIN {
                return Err(Error::corruption(format!(
                    "backup chain is longer than {MAX_CHAIN} archives at {}",
                    p.display()
                )));
            }
            let map = Arc::new(Mmap::open(&p).map_err(|e| store::prefix(&p, e))?);
            let man = decode_manifest(map.as_slice()).map_err(|e| store::prefix(&p, e))?;
            let k = c.maps.len();
            for (_, tables) in &man.dbs {
                for t in tables {
                    for r in &t.parts {
                        if r.at != 0 {
                            // First writer wins: the newest archive in the
                            // chain is opened first, and an identity is by
                            // construction the same bytes wherever it is.
                            c.index.entry(r.key()).or_insert((k, r.at));
                        }
                    }
                }
            }
            next = (!man.base.is_empty()).then(|| resolve_base(&p, Path::new(&man.base)));
            c.maps.push(map);
            c.mans.push(man);
        }
        Ok(c)
    }

    fn has(&self, key: (u64, u64, u64)) -> bool {
        self.index.contains_key(&key)
    }

    /// The bytes of the part `r` names, checksum-verified and re-checked
    /// against all three identity fields.
    fn body(&self, r: &PartRef) -> Result<&[u8]> {
        let (k, at) = if r.at != 0 {
            (0, r.at)
        } else {
            *self.index.get(&r.key()).ok_or_else(|| {
                Error::corruption(format!(
                    "part {:#018x} is not stored in this archive and its base chain does not \
                     have it either; the base archive is missing or is not the one this \
                     backup was taken against",
                    r.crc
                ))
            })?
        };
        let buf = self.maps[k].as_slice();
        let mut rd = Reader::new(buf);
        rd.seek(at as usize)?;
        // `read_framed` verifies the frame's own checksum; the two comparisons
        // after it are what tie *this* body to *this* manifest entry, which is
        // the part the frame cannot know.
        let body = format::read_framed(&mut rd)?;
        if body.len() as u64 != r.len || format::checksum(body) != r.crc {
            return Err(Error::corruption(format!(
                "the body at offset {at} is {} bytes with checksum {:#018x}; the manifest \
                 records {} bytes and {:#018x}",
                body.len(),
                format::checksum(body),
                r.len,
                r.crc
            )));
        }
        Ok(body)
    }
}

fn decode_manifest(buf: &[u8]) -> Result<Manifest> {
    let mut r = Reader::new(buf);
    format::read_header(&mut r)?;
    let at = format::read_footer(buf)?;
    if at < format::HEADER_LEN as u64 {
        return Err(Error::corruption(format!("manifest offset {at} overlaps the header")));
    }
    r.seek(at as usize)?;
    let body = format::read_framed(&mut r)?;

    let mut r = Reader::new(body);
    let base = r.str()?.to_string();
    let created = r.u64()?;
    let ndbs = bounded(r.varint()?, "database count")?;
    let mut dbs = Vec::with_capacity(ndbs.min(64));
    for _ in 0..ndbs {
        let name = r.str()?.to_string();
        let ntables = bounded(r.varint()?, "table count")?;
        let mut tables = Vec::with_capacity(ntables.min(256));
        for _ in 0..ntables {
            let doc = r.bytes()?;
            let (def, files, _) = reader::table_parts_from_bytes(doc)?;
            let nparts = bounded(r.varint()?, "part count")?;
            if nparts != files.len() {
                return Err(Error::corruption(format!(
                    "table `{name}.{}` names {} part files but carries {nparts} part records",
                    def.name,
                    files.len()
                )));
            }
            let mut parts = Vec::with_capacity(nparts.min(4_096));
            for _ in 0..nparts {
                let crc = r.u64()?;
                let rows = r.varint()?;
                let len = r.varint()?;
                let at = r.varint()?;
                parts.push(PartRef { crc, rows, len, at });
            }
            tables.push(ArchivedTable { doc: doc.to_vec(), def, files, parts });
        }
        dbs.push((name, tables));
    }
    Ok(Manifest { base, created, dbs })
}

fn bounded(v: u64, what: &str) -> Result<usize> {
    if v > MAX_COUNT {
        return Err(Error::corruption(format!("{what} of {v} is not plausible")));
    }
    Ok(v as usize)
}

/// A relative base path is relative to the archive that names it, so a pair of
/// archives can be moved together.
fn resolve_base(archive: &Path, base: &Path) -> PathBuf {
    if base.is_absolute() {
        return base.to_path_buf();
    }
    archive.parent().unwrap_or_else(|| Path::new(".")).join(base)
}

/// When the archive was taken, as seconds since the epoch.
pub fn created_at(path: &Path) -> Result<u64> {
    let map = Mmap::open(path).map_err(|e| store::prefix(path, e))?;
    Ok(decode_manifest(map.as_slice())?.created)
}

// ---------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------

/// Check an archive without restoring it.
///
/// Returns the report a restore would produce, or an error naming every
/// problem it found (up to a readable number of them). Everything a restore
/// would rely on is exercised: see the module header.
pub fn verify(path: &Path) -> Result<Report> {
    let chain = Chain::open(path)?;
    let mut bad: Vec<String> = Vec::new();
    let rep = walk(&chain, &mut |_db, _t, _name, body| {
        // The full defensive decode, not a checksum: a body can be intact and
        // still be a part the reader refuses, and an archive that says healthy
        // and restores broken is the one outcome worse than no verify at all.
        reader::part_from_bytes(body).map(|_| ())
    }, &mut bad)?;
    if bad.is_empty() {
        return Ok(rep);
    }
    let shown = bad.len().min(8);
    Err(Error::corruption(format!(
        "{} of {} parts in {} are damaged:\n  {}{}",
        bad.len(),
        rep.parts,
        path.display(),
        bad[..shown].join("\n  "),
        if bad.len() > shown { format!("\n  ... and {} more", bad.len() - shown) } else { String::new() }
    )))
}

/// What [`walk`] hands each part body to: `(database, table, part file, body)`.
type OnPart<'f> = &'f mut dyn FnMut(&str, &str, &str, &[u8]) -> Result<()>;

/// Walk every part of every table, handing each body to `f`. Failures are
/// collected into `bad` rather than propagated, so one damaged part does not
/// hide the other nine.
fn walk(chain: &Chain, f: OnPart<'_>, bad: &mut Vec<String>) -> Result<Report> {
    let mut rep = Report { bytes: chain.maps[0].len() as u64, ..Report::default() };
    let mut seen: Vec<(&str, &str)> = Vec::new();
    for (db, tables) in &chain.mans[0].dbs {
        if !store::is_safe_name(db) {
            bad.push(format!("database `{db}`: not a legal directory name"));
        }
        for t in tables {
            if !store::is_safe_name(&t.def.name) {
                bad.push(format!("table `{db}.{}`: not a legal directory name", t.def.name));
            }
            // Two entries for one name would restore into one directory, and
            // the second commit record would silently hide the first one's
            // parts. This writer cannot produce it -- catalog names are unique
            // -- which is exactly why the reader has to check.
            if !seen.iter().any(|&(a, b)| a == db && b == t.def.name) {
                seen.push((db, &t.def.name));
            } else {
                bad.push(format!("table `{db}.{}` appears twice in the manifest", t.def.name));
            }
            for (i, name) in t.files.iter().enumerate() {
                if store::parse_part_seq(name).is_none() {
                    bad.push(format!("{db}.{}: `{name}` is not a part file name", t.def.name));
                }
                if t.files[..i].contains(name) {
                    bad.push(format!("{db}.{}: `{name}` is named twice", t.def.name));
                }
            }
            rep.tables += 1;
            for (r, name) in t.parts.iter().zip(&t.files) {
                rep.parts += 1;
                rep.rows += r.rows;
                match chain.body(r).and_then(|b| f(db, &t.def.name, name, b).map(|()| b.len())) {
                    Ok(_) => {}
                    Err(e) => bad.push(format!("{db}.{}/{name}: {e}", t.def.name)),
                }
            }
        }
    }
    Ok(rep)
}

// ---------------------------------------------------------------------------
// restore
// ---------------------------------------------------------------------------

/// Unpack `path` into a **new** directory `into`.
///
/// Refuses rather than merges. A restore that wrote into a directory holding a
/// database would interleave two commit records and two sets of part
/// sequence numbers, and the result would be neither database -- which is
/// precisely the failure mode this module was built to end, so it is not
/// available even with a flag. Move the old directory aside and restore beside
/// it; the operator can then swap them with a `rename`, which is atomic, or
/// keep both.
///
/// The archive is verified as it is unpacked -- every frame checksum, every
/// part decoded -- and a failure leaves the target as it was found, because
/// nothing is written until the whole archive has passed.
pub fn restore(path: &Path, into: &Path) -> Result<Report> {
    if let Ok(md) = std::fs::metadata(into) {
        let occupied = md.is_dir()
            && std::fs::read_dir(into)
                .map_err(|e| store::io_err("read directory", into, e))?
                .next()
                .is_some();
        if occupied || !md.is_dir() {
            return Err(Error::storage(format!(
                "refusing to restore into {}: it already exists and is not empty. A restore \
                 never merges into a live database -- restore to a new directory and swap.",
                into.display()
            )));
        }
    }

    let chain = Chain::open(path)?;
    // Two passes. The first proves the archive whole; only then is a byte
    // written, so a bad archive cannot leave a half-populated directory that
    // the next `Session::open` would try to recover.
    let mut bad = Vec::new();
    walk(&chain, &mut |_, _, _, body| reader::part_from_bytes(body).map(|_| ()), &mut bad)?;
    if !bad.is_empty() {
        return Err(Error::corruption(format!(
            "refusing to restore a damaged archive: {} part(s) failed, first is {}",
            bad.len(),
            bad[0]
        )));
    }

    std::fs::create_dir_all(into).map_err(|e| store::io_err("create directory", into, e))?;
    let mut roster: Vec<(String, Vec<TableDef>)> = Vec::new();
    let mut rep = Report { bytes: chain.maps[0].len() as u64, ..Report::default() };
    for (db, tables) in &chain.mans[0].dbs {
        let ddir = into.join(db);
        std::fs::create_dir_all(&ddir).map_err(|e| store::io_err("create directory", &ddir, e))?;
        let mut defs = Vec::with_capacity(tables.len());
        for t in tables {
            let tdir = ddir.join(&t.def.name);
            std::fs::create_dir_all(&tdir)
                .map_err(|e| store::io_err("create directory", &tdir, e))?;
            for (r, name) in t.parts.iter().zip(&t.files) {
                store::atomic_write(&tdir.join(name), chain.body(r)?)?;
                rep.parts += 1;
                rep.rows += r.rows;
            }
            // Last, exactly as a checkpoint does it: the commit record is what
            // makes the parts live, so it is written after every one of them
            // is durable.
            store::commit(&tdir.join(store::TABLE_FILE), &t.doc)?;
            defs.push(t.def.clone());
            rep.tables += 1;
        }
        roster.push((db.clone(), defs));
    }
    // And the root catalog last of all, for the same reason one level up.
    store::commit(&into.join(store::CATALOG_FILE), &writer::catalog_doc(&roster))?;
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persist::testkit::*;
    use crate::storage::Table;

    fn sources(tables: &[Table]) -> Vec<Source<'_>> {
        tables
            .iter()
            .map(|t| Source { db: "default", def: &t.def, snap: t.snapshot() })
            .collect()
    }

    fn roster() -> Vec<String> {
        vec!["default".to_string()]
    }

    /// The one hand-written piece of framing in this module must agree with
    /// the one every other file uses.
    #[test]
    fn frame_matches_format() {
        let s = Scratch::new("bk-frame");
        let p = s.join("f");
        let body: Vec<u8> = (0..1000u32).map(|i| i as u8).collect();
        let mut sink = Sink::create(&p).unwrap();
        sink.frame(&body).unwrap();
        sink.publish(&p).unwrap();
        let mut w = Writer::new();
        format::write_framed(&mut w, &body);
        assert_eq!(std::fs::read(&p).unwrap(), w.finish());
    }

    #[test]
    fn a_full_archive_restores_byte_for_byte() {
        let s = Scratch::new("bk-full");
        let tables = vec![sample_table("hits", &[600, 400]), sample_table("misc", &[120])];
        let arc = s.join("a.gbak");
        let rep = write_archive(&arc, &roster(), &sources(&tables), None).unwrap();
        assert_eq!(rep.tables, 2);
        assert_eq!(rep.parts, 3);
        assert_eq!(rep.reused, 0);

        let out = s.join("out");
        let back = restore(&arc, &out).unwrap();
        assert_eq!((back.tables, back.parts, back.rows), (rep.tables, rep.parts, rep.rows));

        for t in &tables {
            let img = reader::read_table_image(&out.join("default").join(&t.def.name)).unwrap();
            assert!(img.damaged.is_empty());
            assert_eq!(img.def, t.def);
            assert_eq!(
                img.parts.iter().map(|p| p.n_rows).sum::<usize>(),
                t.snapshot().parts().iter().map(|p| p.n_rows).sum::<usize>()
            );
            for (a, b) in img.parts.iter().zip(t.snapshot().parts().iter()) {
                assert_eq!(dump(a), dump(&**b), "a row changed across the archive");
            }
        }
    }

    #[test]
    fn verify_accepts_a_good_archive_and_names_a_bad_one() {
        let s = Scratch::new("bk-verify");
        let tables = vec![sample_table("hits", &[900])];
        let arc = s.join("a.gbak");
        let rep = write_archive(&arc, &roster(), &sources(&tables), None).unwrap();
        assert_eq!(verify(&arc).unwrap(), rep);

        // A byte in the middle of the first part body. The frame checksum has
        // to catch it; a header-only verify would not.
        let mut bytes = std::fs::read(&arc).unwrap();
        let at = bytes.len() / 2;
        bytes[at] ^= 0xff;
        let torn = s.join("torn.gbak");
        std::fs::write(&torn, &bytes).unwrap();
        let e = verify(&torn).expect_err("a corrupted archive must not verify");
        assert_eq!(e.code(), "CHECKSUM_MISMATCH", "{e}");
        assert!(e.to_string().contains("part_000001"), "{e}");

        // And the restore refuses it rather than unpacking what it can.
        let out = s.join("out");
        assert!(restore(&torn, &out).is_err());
        assert!(!out.join("default").exists(), "a refused restore must write nothing");
    }

    #[test]
    fn a_truncated_archive_is_reported_as_truncation() {
        let s = Scratch::new("bk-trunc");
        let tables = vec![sample_table("hits", &[400])];
        let arc = s.join("a.gbak");
        write_archive(&arc, &roster(), &sources(&tables), None).unwrap();
        let mut bytes = std::fs::read(&arc).unwrap();
        bytes.truncate(bytes.len() - 32);
        let cut = s.join("cut.gbak");
        std::fs::write(&cut, &bytes).unwrap();
        let e = verify(&cut).expect_err("a truncated archive must not verify");
        assert_eq!(e.code(), "CHECKSUM_MISMATCH", "{e}");
    }

    #[test]
    fn restore_refuses_a_directory_that_is_not_empty() {
        let s = Scratch::new("bk-occupied");
        let tables = vec![sample_table("hits", &[100])];
        let arc = s.join("a.gbak");
        write_archive(&arc, &roster(), &sources(&tables), None).unwrap();
        let out = s.join("live");
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(out.join("CATALOG"), b"pretend this is a database").unwrap();
        let e = restore(&arc, &out).expect_err("restore must refuse a populated directory");
        assert!(e.to_string().contains("not empty"), "{e}");
        assert_eq!(std::fs::read(out.join("CATALOG")).unwrap(), b"pretend this is a database");
    }

    /// The keystone of the incremental path: an unchanged part is named, not
    /// stored, and the pair still restores whole.
    #[test]
    fn an_incremental_archive_stores_only_what_changed() {
        let s = Scratch::new("bk-incr");
        let mut t = sample_table("hits", &[600, 400]);
        let full = s.join("full.gbak");
        let base_rep = write_archive(&full, &roster(), &sources(std::slice::from_ref(&t)), None).unwrap();
        assert_eq!(base_rep.parts, 2);

        // One more part; the first two are untouched.
        let b = sample_block(200);
        let keys: Vec<u64> = b.column(0).as_u64().unwrap().iter().map(|&k| k + 90_000_000).collect();
        let mut cols = b.columns.clone();
        cols[0] = crate::types::Column::u64s(crate::types::DataType::UInt64, keys);
        t.insert(crate::types::Block::new(cols).unwrap()).unwrap();
        t.flush().unwrap();

        let incr = s.join("incr.gbak");
        let rep =
            write_archive(
                &incr,
                &roster(),
                &sources(std::slice::from_ref(&t)),
                Some(Path::new("full.gbak")),
            )
            .unwrap();
        assert_eq!(rep.parts, 3);
        assert_eq!(rep.reused, 2, "the two unchanged parts must not be stored again");
        assert!(rep.bytes * 2 < base_rep.bytes, "{rep:?} against {base_rep:?}");

        assert_eq!(verify(&incr).unwrap().parts, 3);
        let out = s.join("out");
        restore(&incr, &out).unwrap();
        let img = reader::read_table_image(&out.join("default").join("hits")).unwrap();
        assert_eq!(img.parts.len(), 3);
        assert_eq!(img.parts.iter().map(|p| p.n_rows).sum::<usize>(), 1_200);
    }

    /// An incremental archive is worthless on its own and must say so rather
    /// than restore a database with parts missing.
    #[test]
    fn an_incremental_archive_without_its_base_is_refused() {
        let s = Scratch::new("bk-orphan");
        let t = sample_table("hits", &[600, 400]);
        let full = s.join("full.gbak");
        write_archive(&full, &roster(), &sources(std::slice::from_ref(&t)), None).unwrap();
        let incr = s.join("incr.gbak");
        write_archive(&incr, &roster(), &sources(std::slice::from_ref(&t)), Some(Path::new("full.gbak")))
            .unwrap();
        std::fs::remove_file(&full).unwrap();
        let e = verify(&incr).expect_err("an orphaned incremental must not verify");
        assert!(e.to_string().contains("full.gbak"), "{e}");
        assert!(restore(&incr, &s.join("out")).is_err());
    }

    #[test]
    fn an_empty_database_round_trips() {
        let s = Scratch::new("bk-empty");
        let t = Table::new(table_def("t"), 1 << 20);
        let arc = s.join("a.gbak");
        let rep = write_archive(&arc, &roster(), &sources(std::slice::from_ref(&t)), None).unwrap();
        assert_eq!((rep.tables, rep.parts, rep.rows), (1, 0, 0));
        assert_eq!(verify(&arc).unwrap(), rep);
        let out = s.join("out");
        restore(&arc, &out).unwrap();
        let img = reader::read_table_image(&out.join("default").join("t")).unwrap();
        assert!(img.parts.is_empty());
        assert_eq!(img.def.name, "t");
    }

    /// Deletes live in the `PartSet`, not in the part file, so an archive that
    /// serialized the part alone would resurrect every tombstoned row.
    #[test]
    fn deletes_survive_the_round_trip() {
        let s = Scratch::new("bk-deletes");
        let mut t = sample_table("hits", &[500]);
        let before = t.snapshot().live_rows();
        for pos in (0..300).step_by(7) {
            t.mark_deleted(0, pos);
        }
        let live = t.snapshot().live_rows();
        assert!(live < before, "the test deleted nothing");

        let arc = s.join("a.gbak");
        let rep = write_archive(&arc, &roster(), &sources(std::slice::from_ref(&t)), None).unwrap();
        assert_eq!(rep.rows, live as u64);
        let out = s.join("out");
        restore(&arc, &out).unwrap();
        let img = reader::read_table_image(&out.join("default").join("hits")).unwrap();
        let back: usize = img.parts.iter().map(|p| p.born_live_rows()).sum();
        assert_eq!(back, live);
    }

    #[test]
    fn a_file_that_is_not_an_archive_is_refused() {
        let s = Scratch::new("bk-notarchive");
        let p = s.join("nope.gbak");
        std::fs::write(&p, vec![0u8; 4096]).unwrap();
        assert_eq!(verify(&p).unwrap_err().code(), "CHECKSUM_MISMATCH");
        assert!(Chain::open(&s.join("missing.gbak")).is_err());
    }
}
