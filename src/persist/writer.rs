//! Encoding: in-memory structures -> bytes.
//!
//! ## Part file layout
//!
//! ```text
//!   header                    MAGIC + version
//!   framed granule 0          | one independently checksummed section per
//!   framed granule 1          | granule, in granule order
//!   ...
//!   framed part metadata      row counts, delete bitmap, per-granule zone
//!                             maps, and the file offset of each granule
//!   footer                    offset of the metadata section + version +
//!                             checksum + trailing MAGIC
//! ```
//!
//! Two decisions worth defending:
//!
//! **The metadata section comes last, and the footer points at it.** A reader
//! opens the file, reads the last [`format::FOOTER_LEN`] bytes, and jumps
//! straight to the directory -- no scan. Writing the metadata first would mean
//! either buffering every granule or back-patching offsets into a file we have
//! already streamed out.
//!
//! **Every granule is its own frame.** Corruption stays local (one unreadable
//! granule instead of an unreadable part), and a future mmap reader can verify
//! and decode exactly the granules a query needs. The cost is 9 bytes of
//! framing per granule, against ~4KB of payload.
//!
//! Small structural numbers are LEB128 and bulk payload is fixed-width
//! little-endian, per [`format`]'s contract -- so `base`, `sort_min` and
//! friends are `u64` (they are arbitrary bit patterns with no small-value
//! bias) while counts, widths and column indices are varints.

use std::path::Path;

use crate::common::{Error, Result};
use crate::storage::granule::PkIndexParts;
use crate::storage::part::{self, Deletes};
use crate::storage::{Granule, PackedColumn, Part, Table};
use crate::types::{Block, ColumnData, TableDef};

use super::format::{self, Writer};
use super::reader::MAX_PART_ROWS;
use super::store;

// ---------------------------------------------------------------------------
// parts
// ---------------------------------------------------------------------------

/// Serialize `part` and publish it at `path` atomically. See
/// [`store::atomic_write`] for the durability argument.
///
/// Uses the part's own construction-time delete image, which is the right
/// answer for a part that has never been published into a [`PartSet`] -- a
/// part just decoded, or one a test built by hand. A part inside a table's
/// part set keeps no delete state of its own; [`write_part_with`] is the entry
/// point for those, and `write_table` uses it.
pub fn write_part(path: &Path, part: &Part) -> Result<()> {
    store::atomic_write(path, &part_bytes(part)?)
}

/// [`write_part`] with the delete mask supplied by whoever owns it.
pub fn write_part_with(path: &Path, part: &Part, del: Option<&Deletes>) -> Result<()> {
    store::atomic_write(path, &part_bytes_with(part, del)?)
}

/// The exact bytes [`write_part`] would publish.
///
/// Split out because it is what the tests corrupt, and because a caller
/// packaging parts into something other than a file (a backup stream, a
/// replication frame) needs the body without the filesystem dance.
///
/// Fallible for one reason only: a part the reader would refuse must never
/// reach a file. See [`check_writable`].
pub fn part_bytes(part: &Part) -> Result<Vec<u8>> {
    part_bytes_with(part, part.born_deletes().as_ref())
}

/// [`part_bytes`] with an explicit delete mask.
pub fn part_bytes_with(part: &Part, del: Option<&Deletes>) -> Result<Vec<u8>> {
    check_writable(part)?;
    let mut w = Writer::with_capacity(part.data_bytes() + part.index_bytes() + 1024);
    format::write_header(&mut w);

    let mut offsets: Vec<u64> = Vec::with_capacity(part.granules.len());
    for g in &part.granules {
        offsets.push(w.pos() as u64);
        format::write_framed_aligned(&mut w, &granule_body(g, part.ncols));
    }

    let meta_at = w.pos() as u64;
    format::write_framed_aligned(&mut w, &part_meta(part, del, &offsets));
    format::write_footer(&mut w, meta_at);
    Ok(w.finish())
}

/// Refuse a part the reader could not load back.
///
/// The engine's contract is that a successful `OPTIMIZE`/checkpoint means the
/// rows are safe. Serializing something `reader::decode_meta` will reject
/// breaks that contract in the worst possible way: the write reports success,
/// the superseded parts are collected, and the loss is discovered at the next
/// restart -- as *corruption*, on data that was never corrupt. So the ceiling
/// is asserted here, against the same constant, before a single byte is
/// formatted.
///
/// Not on a hot path: one comparison per part file, against a merge that has
/// just rebuilt every granule and every key index.
fn check_writable(part: &Part) -> Result<()> {
    if part.n_rows as u64 > MAX_PART_ROWS {
        return Err(Error::storage(format!(
            "a part of {} rows cannot be written: the format's limit is {MAX_PART_ROWS} \
             rows per part, so merge or split before committing",
            part.n_rows
        )));
    }
    Ok(())
}

/// Row counts, the delete bitmap, and the directory of granules.
///
/// Zone maps (`sort_min`/`sort_max`) live here rather than in the granule
/// bodies so a reader can prune ranges from the metadata alone, without
/// touching -- or checksumming -- a single granule payload.
fn part_meta(part: &Part, del: Option<&Deletes>, offsets: &[u64]) -> Vec<u8> {
    let words = del.map_or(&[][..], |d| d.words());
    let mut w = Writer::with_capacity(64 + part.granules.len() * 24 + words.len() * 8);
    w.varint(part.n_rows as u64);
    w.varint(part.ncols as u64);
    put_opt_index(&mut w, part.sort_col);
    put_opt_index(&mut w, part.pk_col);
    w.varint(del.map_or(0, |d| d.count()) as u64);
    w.u64_words(words);
    w.varint(part.granules.len() as u64);
    for (g, &off) in part.granules.iter().zip(offsets) {
        w.varint(g.len as u64);
        w.u64(g.sort_min);
        w.u64(g.sort_max);
        w.varint(off);
    }
    w.finish()
}

fn granule_body(g: &Granule, ncols: usize) -> Vec<u8> {
    let mut w = Writer::with_capacity(g.data_bytes() + g.index_bytes() + 64);
    w.varint(g.len as u64);
    w.varint(ncols as u64);
    for c in &g.columns {
        put_column(&mut w, c);
    }
    match &g.pk {
        Some(pk) => {
            w.u8(1);
            put_pk_index(&mut w, &pk.to_parts());
        }
        None => w.u8(0),
    }
    w.finish()
}

/// One packed column: type, FOR metadata, payload words, and the two optional
/// side tables (string dictionary, null mask).
///
/// The type is written as its `Display` form rather than a numeric tag. It
/// costs a handful of bytes per granule and buys two things: `Nullable(...)`
/// and `FixedString(n)` need no separate encoding, and a corrupt file is
/// diagnosable with `strings(1)`.
fn put_column(w: &mut Writer, c: &PackedColumn) {
    w.str(&c.ty.to_string());
    w.varint(c.len() as u64);
    w.u64(c.max_lane());
    let lanes = c.lanes();
    w.u64(lanes.base());
    w.varint(lanes.width() as u64);
    w.u64_words_coded(lanes.words());
    match c.dict() {
        Some(d) => {
            w.u8(1);
            w.bytes(d.blob());
            w.u32_slice(d.offsets());
        }
        None => w.u8(0),
    }
    match c.nulls() {
        Some(n) => {
            w.u8(1);
            w.u64_words(n.words());
        }
        None => w.u8(0),
    }
}

/// The learned-rank point-lookup index, verbatim.
///
/// This is the payload the whole format exists for: the CHD displacement
/// seeds and the fused fingerprint/rank records are minutes of construction
/// across a large table and microseconds to load.
fn put_pk_index(w: &mut Writer, p: &PkIndexParts) {
    w.u64(p.min);
    w.u64(p.max);
    w.u64(p.pmul);
    w.svarint(p.err_bias as i64);
    w.varint(p.ebits as u64);
    w.u64(p.mph_gs);
    w.varint(p.mph_nb as u64);
    w.varint(p.mph_n as u64);
    w.u64(p.seed_base);
    w.varint(p.seed_width as u64);
    w.u64_words(&p.seed_words);
    w.u64(p.fpr_base);
    w.varint(p.fpr_width as u64);
    w.u64_words(&p.fpr_words);
}

fn put_opt_index(w: &mut Writer, v: Option<usize>) {
    match v {
        Some(i) => {
            w.u8(1);
            w.varint(i as u64);
        }
        None => w.u8(0),
    }
}

// ---------------------------------------------------------------------------
// tables
// ---------------------------------------------------------------------------

/// Write the parts of `table` that are not already on disk into
/// `dir/<table name>/` and commit them.
///
/// `dir` is the *database* directory; the table gets a subdirectory of its
/// own. New parts are written under freshly allocated sequence numbers, the
/// `TABLE` file is committed last, and only then are the files no longer named
/// by it removed -- so the table is readable and consistent at every instant,
/// and a crash at any point leaves the previous commit intact.
///
/// ## Incremental, and why that does not weaken the commit
///
/// A part whose bytes are already in a file keeps that file, its sequence
/// number and its inode: it is not read, not rewritten and not renamed. That
/// is the difference between a checkpoint costing O(entire database) and
/// O(what changed) -- before this, a `SELECT 1` against a 100 GB database
/// rewrote 100 GB on exit, needed 100 GB free to do it, and gave every part a
/// new name so no filesystem snapshot could ever be a valid backup.
///
/// The protocol is unchanged in every respect that makes it safe:
///
///   * a part file is still written to a temp name, fsynced, renamed and the
///     directory fsynced ([`store::atomic_write`]);
///   * sequence numbers are still allocated strictly above everything present,
///     so a write can never land on a file the current commit refers to;
///   * the `TABLE` file is still the single commit point, written after every
///     part it names is durable;
///   * files are still unlinked only after that commit, and only ones the new
///     commit does not name -- which now includes leaving the reused ones
///     alone, and still collects orphans from a crashed earlier attempt.
///
/// The one new claim is `PartSet::origin`: "part `i` is already in file
/// `seq`". It is not trusted on its word. Every reuse is checked against the
/// directory listing taken at the top of this function, so the worst a stale
/// origin can cost is a rewrite we did not need -- never a commit record
/// naming a file that is not there. Two consequences of that check are worth
/// stating, because they are what keep a reused name unambiguous:
///
///   * a name can only be reused if it is present, and allocation starts above
///     every present sequence number, so this commit can never hand the same
///     name to two different parts;
///   * a duplicate origin (which would make two entries name one file, and so
///     duplicate rows on reload) is refused the same way -- `taken` lets each
///     file be claimed once, and the loser is rewritten.
pub fn write_table(dir: &Path, table: &Table) -> Result<()> {
    if !store::is_safe_name(&table.def.name) {
        return Err(crate::common::Error::storage(format!(
            "table name `{}` cannot be a directory name",
            table.def.name
        )));
    }
    let tdir = dir.join(&table.def.name);
    std::fs::create_dir_all(&tdir)
        .map_err(|e| store::io_err("create directory", &tdir, e))?;

    let on_disk = store::list_part_files(&tdir)?;
    // Above every sequence number already on disk, so a new part can never
    // land on a file the current commit still refers to -- including the ones
    // this commit is about to reuse.
    let mut next = on_disk.last().map_or(1, |&(s, _)| s + 1);
    let mut taken = vec![false; on_disk.len()];

    // One snapshot for the whole commit: the parts and the delete masks that
    // reach disk must be the same generation, or a restart would resurrect
    // rows a later generation had already tombstoned.
    let snap = table.snapshot();
    let set = snap.set();
    let mut names = Vec::with_capacity(snap.len());
    for (i, p) in snap.parts().iter().enumerate() {
        // `on_disk` is sorted by sequence number, so this is a binary search
        // over the *files*, of which there are hundreds at most -- against a
        // part write of megabytes, it does not exist.
        let claim = match set.origin(i) {
            part::NO_FILE => None,
            seq => match on_disk.binary_search_by_key(&seq, |&(s, _)| s) {
                Ok(k) if !taken[k] => {
                    taken[k] = true;
                    Some(k)
                }
                _ => None,
            },
        };
        match claim {
            Some(k) => names.push(on_disk[k].1.clone()),
            None => {
                let name = store::part_file_name(next);
                write_part_with(&tdir.join(&name), p, set.deletes(i))?;
                // Only now, with the bytes fsynced and the rename durable, is
                // the claim this records actually true.
                set.set_origin(i, next);
                names.push(name);
                next += 1;
            }
        }
    }

    // Everything already in the log is now inside the parts above.
    let wal_len = std::fs::metadata(tdir.join(store::WAL_FILE))
        .map(|m| m.len())
        .unwrap_or(format::HEADER_LEN as u64);

    store::commit(
        &tdir.join(store::TABLE_FILE),
        &table_doc(&table.def, &names, wal_len),
    )?;

    for (k, (_, old)) in on_disk.iter().enumerate() {
        if !taken[k] {
            let _ = std::fs::remove_file(tdir.join(old));
        }
    }
    Ok(())
}

/// The `TABLE` commit record: definition, live parts in order, and the WAL
/// offset those parts already cover.
pub fn table_doc(def: &TableDef, parts: &[String], wal_committed: u64) -> Vec<u8> {
    let mut w = Writer::with_capacity(256 + parts.len() * 24);
    put_table_def(&mut w, def);
    w.varint(parts.len() as u64);
    for p in parts {
        w.str(p);
    }
    w.varint(wal_committed);
    doc(&w.finish())
}

/// The root `CATALOG`: which databases exist and what is in them.
///
/// The definitions are duplicated here and in each table's `TABLE` file. That
/// is deliberate: the catalog answers "what exists" without opening every
/// table directory (and remembers empty databases, which have no directory
/// content at all), while the per-table copy keeps a table directory
/// self-describing, so a single part tree can be recovered even if the root
/// file is lost.
pub fn catalog_doc(roster: &[(String, Vec<TableDef>)]) -> Vec<u8> {
    let mut w = Writer::with_capacity(512);
    w.varint(roster.len() as u64);
    for (db, defs) in roster {
        w.str(db);
        w.varint(defs.len() as u64);
        for d in defs {
            put_table_def(&mut w, d);
        }
    }
    doc(&w.finish())
}

pub fn put_table_def(w: &mut Writer, def: &TableDef) {
    w.str(&def.name);
    w.varint(def.schema.len() as u64);
    for f in def.schema.fields() {
        w.str(&f.name);
        w.str(&f.ty.to_string());
        match f.default_sql() {
            Some(d) => {
                w.u8(1);
                w.str(&d);
            }
            None => w.u8(0),
        }
    }
    w.varint(def.order_by.len() as u64);
    for &c in &def.order_by {
        w.varint(c as u64);
    }
    w.varint(def.primary_key.len() as u64);
    for &c in &def.primary_key {
        w.varint(c as u64);
    }
    put_opt_index(w, def.partition_by);
    w.str(def.engine.name());
}

// ---------------------------------------------------------------------------
// blocks (write-ahead log payloads)
// ---------------------------------------------------------------------------

/// A `Block` in its uncompressed, self-describing form.
///
/// Deliberately *not* the part encoding: a log record is written once, read
/// once during recovery, and must be cheap to append on the latency-critical
/// write path. Bit packing a two-row insert would cost more than it saves.
pub fn put_block(w: &mut Writer, b: &Block) {
    w.varint(b.width() as u64);
    w.varint(b.rows() as u64);
    for c in &b.columns {
        w.str(&c.ty.to_string());
        match &c.data {
            ColumnData::U64(v) => w.u64_slice(v),
            ColumnData::I64(v) => {
                // Raw two's complement, not an order-preserving lane: nothing
                // searches a log record, so the extra transform would be pure
                // cost.
                w.varint(v.len() as u64);
                for &x in v {
                    w.u64(x as u64);
                }
            }
            ColumnData::F64(v) => {
                w.varint(v.len() as u64);
                for &x in v {
                    w.f64(x);
                }
            }
            ColumnData::Str(v) => {
                w.varint(v.len() as u64);
                for s in v {
                    w.str(s);
                }
            }
        }
        match &c.nulls {
            Some(n) => {
                w.u8(1);
                w.u64_slice(n.words());
            }
            None => w.u8(0),
        }
    }
}

// ---------------------------------------------------------------------------
// document envelope
// ---------------------------------------------------------------------------

/// `header | framed body | footer`: the shape of every non-part file we write.
///
/// The frame gives the body a checksum, and the footer gives the file a
/// "complete" marker, so a truncated `CATALOG` is diagnosed as truncation
/// rather than as a body that happens to parse.
pub(crate) fn doc(body: &[u8]) -> Vec<u8> {
    let mut w = Writer::with_capacity(body.len() + format::HEADER_LEN + format::FOOTER_LEN + 16);
    format::write_header(&mut w);
    let at = w.pos() as u64;
    format::write_framed(&mut w, body);
    format::write_footer(&mut w, at);
    w.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persist::reader;
    use crate::persist::testkit::*;
    use crate::storage::Table;
    use crate::types::{Block, Column, DataType, Value};

    #[test]
    fn a_part_file_is_header_frames_and_footer() {
        let bytes = part_bytes(&sample_part(2_500)).unwrap();
        assert!(bytes.len() > format::HEADER_LEN + format::FOOTER_LEN);
        assert_eq!(&bytes[..format::MAGIC.len()], &format::MAGIC);
        assert_eq!(&bytes[bytes.len() - format::MAGIC.len()..], &format::MAGIC);
        // The footer's offset must land on the metadata frame.
        let meta_at = format::read_footer(&bytes).unwrap() as usize;
        assert!(meta_at > format::HEADER_LEN && meta_at < bytes.len() - format::FOOTER_LEN);
    }

    #[test]
    fn encoding_is_deterministic() {
        // Byte-identical output for identical input is what lets a part
        // checksum double as its identity.
        let p = sample_part(1_500);
        assert_eq!(part_bytes(&p).unwrap(), part_bytes(&p).unwrap());
    }

    #[test]
    fn write_part_publishes_atomically() {
        let s = Scratch::new("wp-atomic");
        let path = s.join("p.gpart");
        write_part(&path, &sample_part(300)).unwrap();
        let first = std::fs::read(&path).unwrap();
        write_part(&path, &sample_part(900)).unwrap();
        let second = std::fs::read(&path).unwrap();
        assert_ne!(first, second, "the second write must replace the first");
        assert_eq!(reader::read_part(&path).unwrap().n_rows, 900);
        let names: Vec<String> = std::fs::read_dir(s.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["p.gpart".to_string()]);
    }

    /// The other half of the row-ceiling fix: a part the reader would refuse
    /// must never reach a file.
    ///
    /// The failure this pins is not "the limit is wrong" -- it is that the
    /// limit used to live only in the reader, so `OPTIMIZE TABLE ... FINAL`
    /// reported success, deleted the parts it had just merged, and the rows
    /// were gone. An error here is recoverable; a successful write is not.
    #[test]
    fn a_part_the_reader_would_refuse_is_never_written() {
        let p = sample_part(100);
        // A part carrying a row count past the format's ceiling. Assembled
        // through `from_parts` because building 10^12 real rows to test a
        // bound on the count would be testing the packer.
        let over = Part::from_parts(
            MAX_PART_ROWS as usize + 1,
            p.granules,
            p.deleted,
            p.deleted_count,
            p.sort_col,
            p.pk_col,
            p.ncols,
        );
        let e = part_bytes(&over).expect_err("an unreadable part must not serialize");
        assert_eq!(e.code(), "STORAGE_ERROR", "{e}");
        assert!(e.to_string().contains("limit"), "{e}");

        let s = Scratch::new("wp-overlimit");
        let path = s.join("p.gpart");
        assert!(write_part(&path, &over).is_err());
        assert!(!path.exists(), "a refused write must leave no file behind");
    }

    #[test]
    fn write_table_commits_parts_and_collects_the_old_ones() {
        let s = Scratch::new("wt-commit");
        let t = sample_table("t", &[600, 400, 200]);
        write_table(s.path(), &t).unwrap();
        let tdir = s.join("t");
        assert_eq!(store::list_part_files(&tdir).unwrap().len(), 3);
        assert!(tdir.join(store::TABLE_FILE).exists());

        // A second commit with fewer parts must leave exactly the new ones.
        let t2 = sample_table("t", &[100]);
        write_table(s.path(), &t2).unwrap();
        let files = store::list_part_files(&tdir).unwrap();
        assert_eq!(files.len(), 1, "superseded parts must be collected: {files:?}");
        assert_eq!(files[0].0, 4, "sequence numbers are never reused");
    }

    /// The keystone, at the unit: a second commit of an unchanged table
    /// touches no part file at all.
    #[test]
    fn an_unchanged_part_keeps_its_file() {
        let s = Scratch::new("wt-incremental");
        let t = sample_table("t", &[600, 400]);
        write_table(s.path(), &t).unwrap();
        let tdir = s.join("t");
        let first = store::list_part_files(&tdir).unwrap();
        let bytes: Vec<Vec<u8>> =
            first.iter().map(|(_, n)| std::fs::read(tdir.join(n)).unwrap()).collect();

        write_table(s.path(), &t).unwrap();
        assert_eq!(store::list_part_files(&tdir).unwrap(), first, "a part was renumbered");
        for ((_, n), was) in first.iter().zip(&bytes) {
            assert_eq!(&std::fs::read(tdir.join(n)).unwrap(), was, "{n} was rewritten");
        }
    }

    /// A part whose file has gone missing behind our back must be *written*,
    /// never merely named. This is the guard that keeps a stale provenance a
    /// performance problem instead of a commit record pointing at nothing --
    /// the one failure this mechanism could introduce that loses data.
    #[test]
    fn a_part_whose_file_vanished_is_rewritten_not_named() {
        let s = Scratch::new("wt-vanished");
        let t = sample_table("t", &[600, 400]);
        write_table(s.path(), &t).unwrap();
        let tdir = s.join("t");
        let files = store::list_part_files(&tdir).unwrap();
        std::fs::remove_file(tdir.join(&files[0].1)).unwrap();

        write_table(s.path(), &t).unwrap();
        let img = reader::read_table_image(&tdir).unwrap();
        assert_eq!(img.parts.len(), 2, "the commit lost a part");
        for n in &img.part_files {
            assert!(tdir.join(n).exists(), "the commit names {n}, which is not there");
        }
        assert_eq!(img.parts.iter().map(|p| p.n_rows).sum::<usize>(), 1_000);
    }

    /// Two parts claiming one file would make a commit record name it twice,
    /// and the reload would duplicate every row in it. The claim is
    /// first-come, and the loser is written out.
    #[test]
    fn two_parts_cannot_claim_the_same_file() {
        let s = Scratch::new("wt-dup-origin");
        let tdir = s.join("t");
        std::fs::create_dir_all(&tdir).unwrap();
        // The file both parts will point at, holding the first one's bytes.
        write_part(&tdir.join(store::part_file_name(1)), &sample_part(600)).unwrap();

        let (mut a, mut b) = (sample_part(600), sample_part(400));
        a.set_origin(1);
        b.set_origin(1);
        let t = Table::from_parts(table_def("t"), vec![a, b], 1 << 20);
        write_table(s.path(), &t).unwrap();

        let img = reader::read_table_image(&tdir).unwrap();
        assert_eq!(img.part_files.len(), 2);
        assert_ne!(img.part_files[0], img.part_files[1], "one file was committed twice");
        assert_eq!(img.parts.iter().map(|p| p.n_rows).sum::<usize>(), 1_000);
    }

    #[test]
    fn write_table_of_an_empty_table_commits_nothing_but_the_definition() {
        let s = Scratch::new("wt-empty");
        let t = Table::new(table_def("t"), 1 << 20);
        write_table(s.path(), &t).unwrap();
        let img = reader::read_table_image(&s.join("t")).unwrap();
        assert!(img.parts.is_empty());
        assert_eq!(img.def.name, "t");
    }

    #[test]
    fn stray_files_in_a_table_directory_are_left_alone() {
        let s = Scratch::new("wt-stray");
        let t = sample_table("t", &[100]);
        write_table(s.path(), &t).unwrap();
        let stray = s.join("t").join("notes.txt");
        std::fs::write(&stray, b"hand written").unwrap();
        write_table(s.path(), &t).unwrap();
        assert!(stray.exists(), "the GC must only touch files it named");
    }

    #[test]
    fn table_doc_roundtrips_every_definition_field() {
        use crate::types::{Engine, Field, Schema};
        let def = TableDef {
            name: "weird".into(),
            schema: Schema::new(vec![
                Field::new("a", DataType::UInt8),
                Field::new("b", DataType::Nullable(Box::new(DataType::Float32)))
                    .with_default("1.5")
                    .unwrap(),
                Field::new("c", DataType::LowCardinality(Box::new(DataType::String))),
                Field::new("d", DataType::FixedString(12)),
                Field::new("e", DataType::DateTime),
            ])
            .unwrap(),
            order_by: vec![4, 0],
            primary_key: vec![4],
            partition_by: Some(2),
            engine: Engine::ReplacingMergeTree,
        };
        let bytes = table_doc(&def, &["part_000009.gpart".into()], 4_096);
        let (back, parts, committed) = reader::table_parts_from_bytes(&bytes).unwrap();
        assert_eq!(back, def);
        assert_eq!(parts, vec!["part_000009.gpart".to_string()]);
        assert_eq!(committed, 4_096);
    }

    #[test]
    fn block_encoding_roundtrips_every_physical_kind() {
        let b = sample_block(64);
        let mut w = Writer::new();
        put_block(&mut w, &b);
        let bytes = w.finish();
        let back = reader::block_from_bytes(&bytes, &schema()).unwrap();
        assert_eq!(back, b);
    }

    #[test]
    fn block_encoding_preserves_float_and_null_edge_cases() {
        use crate::types::{ColumnBuilder, Field, Schema};
        let mut nb = ColumnBuilder::new(DataType::Nullable(Box::new(DataType::Int64)));
        nb.push_null();
        nb.push_value(&Value::Int(i64::MIN)).unwrap();
        nb.push_null();
        let b = Block::new(vec![
            Column::f64s(
                DataType::Float64,
                vec![-0.0, f64::INFINITY, f64::MIN_POSITIVE],
            ),
            nb.finish(),
            Column::strs(DataType::String, vec!["".into(), "\u{1F600}".into(), "x".into()]),
        ])
        .unwrap();
        let s = Schema::new(vec![
            Field::new("f", DataType::Float64),
            Field::new("n", DataType::Nullable(Box::new(DataType::Int64))),
            Field::new("s", DataType::String),
        ])
        .unwrap();
        let mut w = Writer::new();
        put_block(&mut w, &b);
        let back = reader::block_from_bytes(&w.finish(), &s).unwrap();
        assert_eq!(back.column(0).as_f64().unwrap()[0].to_bits(), (-0.0f64).to_bits());
        assert!(back.column(1).is_null(0) && back.column(1).is_null(2));
        assert_eq!(back.column(1).value(1), Value::Int(i64::MIN));
        assert_eq!(back.column(2).as_str().unwrap()[1].as_ref(), "\u{1F600}");
    }

    #[test]
    fn an_empty_block_roundtrips() {
        let b = Block::empty(&schema());
        let mut w = Writer::new();
        put_block(&mut w, &b);
        let back = reader::block_from_bytes(&w.finish(), &schema()).unwrap();
        assert_eq!(back.rows(), 0);
        assert_eq!(back.width(), 4);
    }
}
