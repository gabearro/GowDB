//! Decoding: bytes -> in-memory structures, defensively.
//!
//! The writer's job is mechanical. The reader's job is to assume every byte is
//! hostile and still never panic, never allocate on a corrupt length, and
//! never hand back a structure whose invariants the rest of the engine
//! assumes.
//!
//! ## The invariants that are not optional
//!
//! Several hot paths downstream of here index with `get_unchecked`:
//! `PackedU64::get` indexes its word array without a bounds check,
//! `Mph::lookup` indexes the displacement seeds, and `PkIndex::candidate`
//! indexes the fused fingerprint/rank records. That is the right call on a hot
//! path -- the arrays are sized by construction -- but it means a file that
//! *claims* a 40-bit-wide column with two words of payload is not a wrong
//! answer, it is memory unsafety.
//!
//! So every packed array read here is proved large enough for the element
//! count it will be indexed with (`packed_words_needed`, which mirrors
//! `PackedU64::pack`'s own sizing), and every count that feeds such a check is
//! itself tied to a structural quantity that was validated first:
//!
//! ```text
//!   granule len   <= GRANULE_SIZE          (position encoding needs it)
//!   column len    == granule len           (every column covers every row)
//!   mph_n         == granule len           (a slot per key, and the row
//!                                           clamp in `candidate` needs len>0)
//!   seeds         >= mph_nb slots
//!   fpr           >= mph_n slots
//!   pk.min/max    == the pk column's first/last lane
//! ```
//!
//! The last one looks like belt-and-braces next to a checksum, but it is what
//! stops `candidate`'s `key - min` from underflowing on a file whose checksum
//! was recomputed by something other than this writer.
//!
//! Type parsing, schema construction and engine parsing all report their own
//! error kinds (`Bind`, `Unsupported`); those are remapped to `Corruption`
//! here, because "this file is damaged" is the true statement, and callers
//! (and tests) distinguish on the error kind.
//!
//! ## What happens *after* detection: quarantine
//!
//! Detection is not the interesting half. A part file that fails its checksum
//! used to travel up through [`read_table_image`] and `store::load_catalog` as
//! an ordinary `Err`, which meant one bad block took the whole instance
//! offline: `SELECT` on an unrelated table, `SHOW TABLES`, even `CREATE TABLE`
//! all failed, because the process could not finish opening the directory.
//!
//! So [`read_table_image`] now *quarantines* rather than propagates: the parts
//! that decode are returned, the ones that do not are recorded as
//! [`DamagedPart`], and the damage is handed to the catalog, which refuses
//! every read and every write of that table by name. The distinction that
//! matters is per table, not per database.
//!
//! **The table is refused, not served short.** Returning the rows that did
//! decode would be the worst of the three options -- the caller gets a
//! plausible answer that is missing however many rows lived in the bad file,
//! with nothing in the result to say so. A refusal naming the file is an
//! answer an operator can act on. There is deliberately no "read what you can"
//! switch here; if one is ever added it belongs where a user has to type it,
//! not where a default can drift onto it.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::common::{BitSet, Error, Result, GRANULE_SIZE};
use crate::encoding::{PackedU64, StringDict};
use crate::storage::granule::PkIndexParts;
use crate::storage::{Granule, PackedColumn, Part, PkIndex, Table};
use crate::types::{Block, Column, ColumnData, DataType, Engine, Field, Schema, TableDef};

use super::format::{self, Reader};
use super::mmap::Mmap;
use super::store;

/// Sanity ceiling on any count that would otherwise size a `Vec` before the
/// bytes behind it are proved present. Everything below this is still checked
/// against the remaining buffer; this only stops the arithmetic from being
/// silly in the first place.
///
/// It bounds counts of *structures* -- columns in a schema, parts in a commit
/// record, databases in the catalog -- and nothing that grows with the data.
/// A count that scales with rows must use [`MAX_PART_ROWS`] (or a bound derived
/// from it) instead: this ceiling was once applied to the part row count, which
/// made every part above 16,777,216 rows serialize happily and then fail to
/// load, reported as corruption. A plausibility bound that a legitimate write
/// can cross is not a plausibility bound, it is silent data loss.
const MAX_COUNT: u64 = 1 << 24;

/// The real ceiling on rows in one part, and on rows in one log record.
///
/// Picked to be unreachable rather than merely generous: at `GRANULE_SIZE`
/// rows per granule a part this large is 2^30 granules and tens of terabytes,
/// so the file itself runs out long before the count does. Everything below it
/// is still proved against the bytes actually present, exactly as before -- the
/// number only has to be past anything a working engine can produce, and 10^12
/// rows is two orders of magnitude past the largest table this storage layout
/// is dimensioned for.
///
/// `writer::part_bytes` refuses to emit a part above this, so the two sides
/// cannot disagree. That direction matters more than the value: a limit only
/// the reader enforces is discovered at the next restart, on data that is
/// already gone.
pub(crate) const MAX_PART_ROWS: u64 = 1 << 40;

/// Granules one part may claim.
///
/// Not a structure count in the [`MAX_COUNT`] sense even though it counts
/// structures: it is `n_rows / GRANULE_SIZE`, so bounding it independently of
/// the row limit would just reintroduce the same bug one level down.
const MAX_GRANULES: u64 = MAX_PART_ROWS / GRANULE_SIZE as u64;

/// Reserve for a count we have not yet consumed the payload of. The reader
/// still bounds-checks every element, so under-reserving costs a realloc and
/// over-reserving costs nothing but a corrupt file's dream.
fn cap(n: usize) -> usize {
    n.min(4_096)
}

fn bad(msg: impl Into<String>) -> Error {
    Error::corruption(msg)
}

/// Remap a non-corruption error from a parser we borrow (types, schemas,
/// engines) into the one that describes the actual situation.
fn as_corrupt(what: &str, e: Error) -> Error {
    match e {
        Error::Corruption(_) | Error::Io(_) => e,
        other => bad(format!("{what}: {other}")),
    }
}

// ---------------------------------------------------------------------------
// parts
// ---------------------------------------------------------------------------

/// What keeps a mapping alive for as long as the columns pointing into it.
///
/// `None` means the bytes are a transient buffer and every word array has to
/// be copied onto the heap before the decoder returns.
type Owner<'a> = Option<&'a Arc<Mmap>>;

/// Load a part file written by [`super::write_part`].
///
/// Maps the file rather than reading it. The win is not I/O -- both paths
/// touch the same bytes -- it is that packed lanes stay *in the mapping*: a
/// part costs page-cache the kernel can reclaim instead of resident heap it
/// cannot, and opening one no longer scales its cost with the size of its
/// payload. Zone maps, dictionaries and bitsets are still copied; they are
/// small, and they are what pruning reads before deciding to touch a granule.
///
/// Falls back to a plain read if the file cannot be mapped -- a zero-length
/// part, or a filesystem that refuses. The decoded `Part` is identical either
/// way, so this is a performance fallback, not a compatibility one.
pub fn read_part(path: &Path) -> Result<Part> {
    match Mmap::open(path) {
        Ok(m) => part_from_mmap(Arc::new(m)).map_err(|e| store::prefix(path, e)),
        Err(_) => {
            let buf = store::read_file(path)?;
            part_from_bytes(&buf).map_err(|e| store::prefix(path, e))
        }
    }
}

/// Decode a part directly out of a mapping, borrowing its packed lanes.
pub fn part_from_mmap(map: Arc<Mmap>) -> Result<Part> {
    // The borrow of `map` ends with this call; what escapes into the `Part` is
    // clones of the `Arc`, held by the columns that point into its pages.
    decode_part(map.as_slice(), Some(&map))
}

/// Decode a part from an in-memory image, copying everything it needs.
///
/// Reads the header first so that a file from a newer build is reported as a
/// version problem rather than as whatever the footer makes of it.
pub fn part_from_bytes(buf: &[u8]) -> Result<Part> {
    decode_part(buf, None)
}

fn decode_part(buf: &[u8], own: Owner) -> Result<Part> {
    let mut r = Reader::new(buf);
    format::read_header(&mut r)?;
    let meta_at = format::read_footer(buf)?;
    if meta_at < format::HEADER_LEN as u64 {
        return Err(bad(format!("metadata offset {meta_at} overlaps the header")));
    }
    r.seek(meta_at as usize)?;
    let meta = decode_meta(format::read_framed_aligned(&mut r)?, meta_at)?;

    let mut granules = Vec::with_capacity(cap(meta.granules.len()));
    for (gi, d) in meta.granules.iter().enumerate() {
        let mut gr = Reader::new(buf);
        gr.seek(d.offset as usize)?;
        let body = format::read_framed_aligned(&mut gr)
            .map_err(|e| bad(format!("granule {gi}: {e}")))?;
        granules.push(
            decode_granule(body, d, &meta, own).map_err(|e| bad(format!("granule {gi}: {e}")))?,
        );
    }

    let mut p = Part::from_parts(
        meta.n_rows,
        granules,
        meta.deleted,
        meta.deleted_count,
        meta.sort_col,
        meta.pk_col,
        meta.ncols,
    );
    p.pid = meta.pid;
    Ok(p)
}

struct GranuleDesc {
    len: usize,
    sort_min: u64,
    sort_max: u64,
    offset: u64,
}

struct Meta {
    n_rows: usize,
    pid: u64,
    ncols: usize,
    sort_col: Option<usize>,
    pk_col: Option<usize>,
    deleted: BitSet,
    deleted_count: usize,
    granules: Vec<GranuleDesc>,
}

fn decode_meta(body: &[u8], meta_at: u64) -> Result<Meta> {
    let mut r = Reader::new(body);
    let n_rows = row_count(r.varint()?, "part row count")?;
    let pid = r.varint()?;
    let ncols = count(r.varint()?, "part column count")?;
    let sort_col = get_opt_index(&mut r, ncols, "sort column")?;
    let pk_col = get_opt_index(&mut r, ncols, "primary key column")?;
    let deleted_count = row_count(r.varint()?, "deleted count")?;
    let deleted_words = r.u64_words_coded()?.into_vec();
    let ngranules = bounded(r.varint()?, MAX_GRANULES, "granule count")?;

    let mut granules = Vec::with_capacity(cap(ngranules));
    let mut total = 0usize;
    let mut prev_end = format::HEADER_LEN as u64;
    for gi in 0..ngranules {
        let len = count(r.varint()?, "granule row count")?;
        if len > GRANULE_SIZE {
            return Err(bad(format!(
                "granule {gi} claims {len} rows, more than the {GRANULE_SIZE}-row maximum"
            )));
        }
        let sort_min = r.u64()?;
        let sort_max = r.u64()?;
        let offset = r.varint()?;
        if offset < prev_end || offset >= meta_at {
            return Err(bad(format!(
                "granule {gi} is at offset {offset}, outside the data region \
                 [{prev_end}, {meta_at})"
            )));
        }
        prev_end = offset;
        total = total
            .checked_add(len)
            .ok_or_else(|| bad("granule row counts overflow"))?;
        granules.push(GranuleDesc { len, sort_min, sort_max, offset });
    }
    if !r.is_empty() {
        return Err(bad(format!("{} trailing bytes in part metadata", r.remaining())));
    }
    if total != n_rows {
        return Err(bad(format!(
            "part claims {n_rows} rows, its granules hold {total}"
        )));
    }
    if deleted_count > n_rows {
        return Err(bad(format!(
            "part claims {deleted_count} deleted rows out of {n_rows}"
        )));
    }
    // Positions are granule-major (`granule << G_SHIFT | offset`), so the
    // bitmap can never be wider than the position space.
    let max_words = ngranules.saturating_mul(GRANULE_SIZE).div_ceil(64) + 1;
    if deleted_words.len() > max_words {
        return Err(bad(format!(
            "delete bitmap has {} words, more than the {max_words} the part can address",
            deleted_words.len()
        )));
    }
    Ok(Meta {
        n_rows,
        pid,
        ncols,
        sort_col,
        pk_col,
        deleted: BitSet::from_words(deleted_words),
        deleted_count,
        granules,
    })
}

fn decode_granule(body: &[u8], d: &GranuleDesc, meta: &Meta, own: Owner) -> Result<Granule> {
    let mut r = Reader::new(body);
    let len = count(r.varint()?, "granule row count")?;
    let ncols = count(r.varint()?, "granule column count")?;
    if len != d.len {
        return Err(bad(format!(
            "granule holds {len} rows, its metadata entry says {}",
            d.len
        )));
    }
    if ncols != meta.ncols {
        return Err(bad(format!(
            "granule holds {ncols} columns, the part has {}",
            meta.ncols
        )));
    }
    let mut columns = Vec::with_capacity(cap(ncols));
    for ci in 0..ncols {
        columns.push(decode_column(&mut r, len, own).map_err(|e| bad(format!("column {ci}: {e}")))?);
    }
    let pk = match r.u8()? {
        0 => None,
        1 => {
            let pkc = meta.pk_col.ok_or_else(|| {
                bad("granule carries a key index but the part declares no primary key column")
            })?;
            let col = columns
                .get(pkc)
                .ok_or_else(|| bad(format!("primary key column {pkc} is out of range")))?;
            Some(decode_pk_index(&mut r, len, col, own)?)
        }
        other => return Err(bad(format!("unknown granule index tag {other}"))),
    };
    if !r.is_empty() {
        return Err(bad(format!("{} trailing bytes in granule body", r.remaining())));
    }
    Ok(Granule::from_parts(len, columns, pk, d.sort_min, d.sort_max))
}

fn decode_column(r: &mut Reader, rows: usize, own: Owner) -> Result<PackedColumn> {
    let ty = DataType::parse(r.str()?).map_err(|e| as_corrupt("bad column type", e))?;
    let len = count(r.varint()?, "column length")?;
    if len != rows {
        return Err(bad(format!("column holds {len} rows, the granule has {rows}")));
    }
    let max_lane = r.u64()?;
    let lanes = decode_packed(r, len, "column", own)?;

    let dict = match r.u8()? {
        0 => None,
        1 => {
            let blob = r.bytes()?.to_vec();
            let offsets = r.u32_slice()?;
            Some(StringDict::from_parts(blob, offsets)?)
        }
        other => return Err(bad(format!("unknown dictionary tag {other}"))),
    };
    let nulls = match r.u8()? {
        0 => None,
        1 => {
            let words = r.u64_words_coded()?.into_vec();
            // `BitSet::get` is bounds-checked, so an over-long mask is only a
            // structural surprise -- but it is still one this writer cannot
            // produce.
            if words.len() > len.div_ceil(64) + 1 {
                return Err(bad(format!(
                    "null mask has {} words for {len} rows",
                    words.len()
                )));
            }
            Some(BitSet::from_words(words))
        }
        other => return Err(bad(format!("unknown null-mask tag {other}"))),
    };
    // A string column's lanes are dictionary codes and mean nothing without
    // the dictionary; a non-string column has no use for one. Either mismatch
    // means the two halves of the file came from different columns.
    let stringy = ty.physical() == crate::types::PhysicalType::Str;
    if stringy && dict.is_none() && len > 0 {
        return Err(bad(format!("{ty} column of {len} rows has no dictionary")));
    }
    if !stringy && dict.is_some() {
        return Err(bad(format!("{ty} column carries a string dictionary")));
    }
    Ok(PackedColumn::from_parts(ty, lanes, dict, nulls, max_lane, len))
}

/// Read a FOR-packed array and prove its word count covers `n` elements.
fn decode_packed(r: &mut Reader, n: usize, what: &str, own: Owner) -> Result<PackedU64> {
    let base = r.u64()?;
    let width = r.varint()?;
    if width > 64 {
        return Err(bad(format!("{what} claims a {width}-bit payload, the maximum is 64")));
    }
    let words = r.u64_words_coded()?;
    let have = words.len();
    let need = packed_words_needed(width as u32, n);
    if have < need {
        return Err(bad(format!(
            "{what} needs {need} words for {n} values at {width} bits, has {have}"
        )));
    }
    let bytes = match words {
        format::Words::Raw(b) => b,
        // A compressed array is already on the heap and there is nothing left
        // to borrow -- the mapping held the block, not the lanes.
        format::Words::Owned(v) => return Ok(PackedU64::from_parts(base, width as u32, v)),
    };
    // The whole point of the v2 alignment: when the bytes came out of a
    // mapping and landed 8-aligned, the packed lanes *are* the file. Nothing
    // is copied, nothing is decompressed, and the pages stay evictable under
    // memory pressure instead of pinning heap. Anything else -- an owned
    // buffer, a big-endian target -- falls back to a copy.
    match own.and_then(|m| format::as_u64_slice(bytes).map(|w| (m, w))) {
        // SAFETY: `w` points into the mapping that `m` owns, and the clone of
        // `m` stored alongside the pointer keeps that mapping alive for at
        // least as long as the `PackedU64`.
        Some((m, w)) => Ok(unsafe {
            PackedU64::from_mapped(base, width as u32, w.as_ptr(), w.len(), m.clone())
        }),
        None => Ok(PackedU64::from_parts(base, width as u32, format::to_u64_vec(bytes))),
    }
}

/// Word count [`PackedU64::pack`] would produce for `n` values of this width.
///
/// This mirrors the packer exactly, including its trailing pad word -- the
/// straddled `get` reads `words[wi + 1]` unconditionally, so the pad is part
/// of the safety contract, not slack.
fn packed_words_needed(width: u32, n: usize) -> usize {
    if width == 0 {
        // Constant column: no payload, but `pack` still emits the two-word
        // placeholder and `prefetch` indexes `words.len() - 1`.
        return 2;
    }
    if width <= 32 {
        // Word-aligned lanes: `floor(64/width)` per word, plus the pad word.
        let per = 64 / width as usize;
        n.div_ceil(per) + 1
    } else {
        // Straddled: `ceil(n * width / 64)` words of payload, plus the pad
        // word the `u128` load in `get` reads unconditionally. `n` is bounded
        // by GRANULE_SIZE before we get here, so the product cannot overflow;
        // the saturating form keeps that true if a caller ever changes.
        (n.saturating_mul(width as usize).div_ceil(64) + 1).max(2)
    }
}

fn decode_pk_index(
    r: &mut Reader,
    len: usize,
    pk_col: &PackedColumn,
    own: Owner,
) -> Result<PkIndex> {
    if len == 0 {
        return Err(bad("empty granule carries a key index"));
    }
    let min = r.u64()?;
    let max = r.u64()?;
    let pmul = r.u64()?;
    let err_bias = r.svarint()?;
    let ebits = r.varint()?;
    let mph_gs = r.u64()?;
    let mph_nb = count(r.varint()?, "mph bucket count")?;
    let mph_n = count(r.varint()?, "mph key count")?;

    if ebits > 63 {
        return Err(bad(format!("key index error width {ebits} exceeds 63 bits")));
    }
    if err_bias < i32::MIN as i64 || err_bias > i32::MAX as i64 {
        return Err(bad(format!("key index error bias {err_bias} does not fit in an i32")));
    }
    if mph_n != len {
        return Err(bad(format!(
            "key index covers {mph_n} keys, the granule has {len} rows"
        )));
    }
    if mph_nb == 0 || mph_nb > mph_n {
        return Err(bad(format!(
            "key index has {mph_nb} buckets for {mph_n} keys"
        )));
    }
    // `candidate` computes `key - min` unsigned and clamps the resulting row
    // into `[0, len)`. Both are only safe if the bounds really are this
    // granule's first and last key.
    if min != pk_col.lane(0) || max != pk_col.lane(len - 1) {
        return Err(bad(format!(
            "key index bounds [{min}, {max}] do not match the stored key column \
             [{}, {}]",
            pk_col.lane(0),
            pk_col.lane(len - 1)
        )));
    }
    if min > max {
        return Err(bad(format!("key index bounds [{min}, {max}] are inverted")));
    }

    let seeds = decode_packed(r, mph_nb, "mph seeds", own)?;
    let fpr = decode_packed(r, mph_n, "key index records", own)?;

    PkIndex::from_parts(PkIndexParts {
        min,
        max,
        pmul,
        err_bias: err_bias as i32,
        ebits: ebits as u32,
        mph_gs,
        mph_nb: mph_nb as u32,
        mph_n,
        seed_base: seeds.base(),
        seed_width: seeds.width(),
        seed_words: seeds.words().to_vec(),
        fpr_base: fpr.base(),
        fpr_width: fpr.width(),
        fpr_words: fpr.words().to_vec(),
    })
}

fn get_opt_index(r: &mut Reader, ncols: usize, what: &str) -> Result<Option<usize>> {
    Ok(match r.u8()? {
        0 => None,
        1 => {
            let i = count(r.varint()?, what)?;
            if i >= ncols {
                return Err(bad(format!("{what} {i} is out of range for {ncols} columns")));
            }
            Some(i)
        }
        other => return Err(bad(format!("unknown {what} tag {other}"))),
    })
}

/// Narrow an on-disk count of *structures* to `usize` with a sanity ceiling.
fn count(v: u64, what: &str) -> Result<usize> {
    bounded(v, MAX_COUNT, what)
}

/// Narrow an on-disk count of *rows*, or of one-per-row elements, to `usize`.
///
/// Split from [`count`] because these are the counts a legitimate write can
/// drive arbitrarily high; see [`MAX_PART_ROWS`].
fn row_count(v: u64, what: &str) -> Result<usize> {
    bounded(v, MAX_PART_ROWS, what)
}

fn bounded(v: u64, max: u64, what: &str) -> Result<usize> {
    // The second test is not redundant on a 32-bit target, where `as usize`
    // would truncate `MAX_PART_ROWS`-sized values into a small, plausible and
    // completely wrong count. It compiles away on 64-bit.
    if v > max || v > usize::MAX as u64 {
        return Err(bad(format!("{what} of {v} is implausible (the maximum is {max})")));
    }
    Ok(v as usize)
}

// ---------------------------------------------------------------------------
// tables
// ---------------------------------------------------------------------------

/// A part file the reader refused, and why.
///
/// The unit is the *file* and not the granule the damage was found in: a
/// granule is not something anyone can restore from a backup, and a part file
/// is. `why` is the reader's own diagnosis, which already names the granule,
/// the field and the offset.
#[derive(Clone, Debug)]
pub struct DamagedPart {
    /// File name inside the table directory, e.g. `part_000003.gpart`.
    pub file: String,
    pub why: String,
}

/// Everything a committed table directory holds.
pub struct TableImage {
    pub def: TableDef,
    /// The parts that decoded. **Not** necessarily every part the commit
    /// record names -- see `damaged`, and never hand these to a reader without
    /// consulting it.
    pub parts: Vec<Part>,
    /// The part file names, in commit order.
    pub part_files: Vec<String>,
    /// WAL byte offset the parts already cover.
    pub wal_committed: u64,
    /// Part files the commit record names that could not be decoded. Empty on
    /// a healthy table, which is the only shape most callers ever see.
    pub damaged: Vec<DamagedPart>,
}

/// Load the table named `name` from the database directory `dir`.
///
/// Refuses a damaged table outright. This entry point hands back a bare
/// [`Table`] with nowhere to record a quarantine, so serving one with parts
/// missing would be exactly the silent short answer the module docs refuse;
/// the catalog's loader is the caller that can hold the damage instead.
pub fn read_table(dir: &Path, name: &str) -> Result<Table> {
    let img = read_table_image(&dir.join(name))?;
    // Claimed on every path, error included: a record nobody claims would sit
    // in the hand-off until a later load of the same directory replaced it.
    claim_damage(&dir.join(&img.def.name));
    if img.def.name != name {
        return Err(bad(format!(
            "directory `{name}` holds the definition of table `{}`",
            img.def.name
        )));
    }
    if let Some(d) = img.damaged.first() {
        return Err(bad(format!(
            "table `{name}` cannot be loaded: {} of its {} part files are damaged, \
             starting with {} ({})",
            img.damaged.len(),
            img.part_files.len(),
            d.file,
            d.why
        )));
    }
    Ok(Table::from_parts(img.def, img.parts, store::delta_limit()))
}

/// Read a table directory's commit record and every part it names.
///
/// A part that fails to decode is quarantined, not propagated: it lands in
/// `damaged` and in the hand-off [`claim_damage`] reads, and the rest of the
/// table -- and every other table in the database -- still opens. The commit
/// record itself is not optional, though: without it there is no schema and no
/// part list, so nothing about the table is knowable and the error stands.
pub fn read_table_image(tdir: &Path) -> Result<TableImage> {
    let path = tdir.join(store::TABLE_FILE);
    let bytes = store::read_file(&path)?;
    let (def, part_files, wal_committed) =
        table_parts_from_bytes(&bytes).map_err(|e| store::prefix(&path, e))?;
    let mut parts = Vec::with_capacity(cap(part_files.len()));
    let mut damaged = Vec::new();
    for n in &part_files {
        match read_part(&tdir.join(n)) {
            Ok(p) => parts.push(p),
            // The healthy parts are kept rather than dropped on the floor.
            // They are what a repair tool -- or an explicit, typed-out
            // "read what you can" -- would need, and they bound the damage an
            // accidental rewrite of this table could do to the one file.
            Err(e) => damaged.push(DamagedPart { file: n.clone(), why: e.to_string() }),
        }
    }
    // Keyed by the directory the *definition* names, because that is the path
    // the catalog reconstructs from the `TableDef` it is handed. The two are
    // the same directory for every commit record this writer produces; a
    // record naming some other table is the one case where they differ, and
    // following the definition is what keeps a record and its claim in step.
    //
    // A clean table still has to *clear* an earlier load's record, but only if
    // there is one to clear -- so the guard is what keeps the healthy open
    // path free of even the `PathBuf` this join would allocate per table.
    if !damaged.is_empty() || any_pending_damage() {
        set_pending(&sibling(tdir, &def.name), damaged.clone());
    }
    Ok(TableImage { def, parts, part_files, wal_committed, damaged })
}

fn sibling(tdir: &Path, name: &str) -> PathBuf {
    tdir.parent().map_or_else(|| PathBuf::from(name), |p| p.join(name))
}

// ---------------------------------------------------------------------------
// the loader -> catalog hand-off
// ---------------------------------------------------------------------------

thread_local! {
    /// Damage found by [`read_table_image`] and not yet claimed by a catalog.
    ///
    /// `store::load_catalog` hands each table to the catalog as a `TableDef`
    /// and a `Vec<Part>`, and neither has room for the files that did not
    /// decode. The reader leaves the record here instead, and
    /// `Catalog::table_by_path_mut` claims it on the loader's own resolve --
    /// the next thing that happens to that table, and the last one that is
    /// allowed to succeed.
    ///
    /// Thread-local rather than global on purpose: a load runs start to finish
    /// on the thread that called it, so the deposit and the claim are one
    /// sequence with nothing of another session's interleaved. Keyed by the
    /// table directory, so a claim can only pick up its own table's damage.
    ///
    /// It is a stand-in, not a design: the moment `load_catalog` passes the
    /// image's `damaged` list to [`crate::catalog::Catalog::quarantine`]
    /// itself, everything under this heading can go.
    static PENDING: RefCell<Vec<(PathBuf, Vec<DamagedPart>)>> = const { RefCell::new(Vec::new()) };
}

/// How many entries are outstanding across all threads.
///
/// The catalog tests this before touching the thread-local at all, because it
/// tests it on the resolve path -- once per INSERT block -- and a relaxed load
/// of a never-written cache line is a branch the predictor gets right every
/// time on an undamaged database, while a TLS access is not free.
static PENDING_COUNT: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn any_pending_damage() -> bool {
    PENDING_COUNT.load(Ordering::Relaxed) != 0
}

fn set_pending(tdir: &Path, damaged: Vec<DamagedPart>) {
    if damaged.is_empty() {
        clear_pending(tdir);
        return;
    }
    PENDING.with(|p| {
        let mut v = p.borrow_mut();
        match v.iter_mut().find(|(k, _)| k == tdir) {
            Some(slot) => slot.1 = damaged,
            None => {
                v.push((tdir.to_path_buf(), damaged));
                PENDING_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }
    });
}

fn clear_pending(tdir: &Path) {
    if !any_pending_damage() {
        return;
    }
    PENDING.with(|p| {
        let mut v = p.borrow_mut();
        if let Some(i) = v.iter().position(|(k, _)| k == tdir) {
            v.swap_remove(i);
            PENDING_COUNT.fetch_sub(1, Ordering::Relaxed);
        }
    });
}

/// Take the damage recorded for `tdir`, if any. Taking is what keeps the
/// hand-off from being sticky: a table that is dropped and re-created under
/// the same name finds nothing here.
pub(crate) fn claim_damage(tdir: &Path) -> Option<Vec<DamagedPart>> {
    if !any_pending_damage() {
        return None;
    }
    PENDING.with(|p| {
        let mut v = p.borrow_mut();
        let i = v.iter().position(|(k, _)| k == tdir)?;
        PENDING_COUNT.fetch_sub(1, Ordering::Relaxed);
        Some(v.swap_remove(i).1)
    })
}

/// Just the definition and the part list, for a commit-record rewrite.
pub fn table_header(bytes: &[u8]) -> Result<(TableDef, Vec<String>)> {
    let (def, parts, _) = table_parts_from_bytes(bytes)?;
    Ok((def, parts))
}

pub fn table_parts_from_bytes(bytes: &[u8]) -> Result<(TableDef, Vec<String>, u64)> {
    let body = doc_body(bytes)?;
    let mut r = Reader::new(body);
    let def = get_table_def(&mut r)?;
    let n = count(r.varint()?, "part count")?;
    let mut parts = Vec::with_capacity(cap(n));
    for _ in 0..n {
        let name = r.str()?;
        // A commit record names files we are about to open. Anything that is
        // not one of our own part names -- a path, a traversal, a symlink
        // bait -- is refused here rather than passed to the filesystem.
        if store::parse_part_seq(name).is_none() {
            return Err(bad(format!("`{name}` is not a part file name")));
        }
        parts.push(name.to_string());
    }
    let wal_committed = r.varint()?;
    if !r.is_empty() {
        return Err(bad(format!("{} trailing bytes in a commit record", r.remaining())));
    }
    Ok((def, parts, wal_committed))
}

/// The roster, and the directory's instance id -- 0 for a `CATALOG` written
/// before the id existed, which is the one value the id never takes and so
/// reads unambiguously as "unstamped".
pub fn catalog_from_bytes(bytes: &[u8]) -> Result<(Vec<(String, Vec<TableDef>)>, u64)> {
    let body = doc_body(bytes)?;
    let mut r = Reader::new(body);
    let ndbs = count(r.varint()?, "database count")?;
    let mut out = Vec::with_capacity(cap(ndbs));
    for _ in 0..ndbs {
        let db = r.str()?.to_string();
        if !store::is_safe_name(&db) {
            return Err(bad(format!("`{db}` is not a usable database name")));
        }
        let ntables = count(r.varint()?, "table count")?;
        let mut defs = Vec::with_capacity(cap(ntables));
        for _ in 0..ntables {
            defs.push(get_table_def(&mut r)?);
        }
        out.push((db, defs));
    }
    let instance = if r.is_empty() { 0 } else { r.u64()? };
    if !r.is_empty() {
        // Named, because the overwhelmingly likely cause is not damage: this
        // is what a *newer* build's CATALOG looks like to an older one, and an
        // operator who has just rolled a binary back should be told that
        // before they conclude their database is corrupt.
        return Err(bad(format!(
            "{} trailing bytes in the catalog. If this directory was last opened by a \
             newer build of granular, that is the cause and the data is intact -- its \
             catalog carries fields this build does not know about. Re-open it with the \
             newer build",
            r.remaining()
        )));
    }
    Ok((out, instance))
}

fn get_table_def(r: &mut Reader) -> Result<TableDef> {
    let name = r.str()?.to_string();
    // The name becomes a directory component during recovery.
    if !store::is_safe_name(&name) {
        return Err(bad(format!("`{name}` is not a usable table name")));
    }
    let nfields = count(r.varint()?, "column count")?;
    let mut fields = Vec::with_capacity(cap(nfields));
    for _ in 0..nfields {
        let fname = r.str()?.to_string();
        let ty = DataType::parse(r.str()?).map_err(|e| as_corrupt("bad column type", e))?;
        let default = match r.u8()? {
            0 => None,
            1 => Some(r.str()?.to_string()),
            other => return Err(bad(format!("unknown default tag {other}"))),
        };
        let mut f = Field::new(fname, ty);
        if let Some(d) = default {
            // Re-evaluated rather than stored evaluated: the literal is what
            // survives an `ALTER` of the column type, and a default that no
            // longer coerces is a damaged definition, not a usable one.
            f = f.with_default(&d).map_err(|e| as_corrupt("bad column default", e))?;
        }
        fields.push(f);
    }
    let schema = Schema::new(fields).map_err(|e| as_corrupt("bad schema", e))?;
    let order_by = get_index_list(r, schema.len(), "ORDER BY")?;
    let primary_key = get_index_list(r, schema.len(), "PRIMARY KEY")?;
    let partition_by = get_opt_index(r, schema.len().max(1), "PARTITION BY")?;
    let engine = Engine::parse(r.str()?).map_err(|e| as_corrupt("bad engine", e))?;
    Ok(TableDef { name, schema, order_by, primary_key, partition_by, engine })
}

fn get_index_list(r: &mut Reader, ncols: usize, what: &str) -> Result<Vec<usize>> {
    let n = count(r.varint()?, what)?;
    let mut out = Vec::with_capacity(cap(n));
    for _ in 0..n {
        let i = count(r.varint()?, what)?;
        if i >= ncols {
            return Err(bad(format!("{what} column {i} is out of range for {ncols} columns")));
        }
        out.push(i);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// blocks
// ---------------------------------------------------------------------------

pub fn block_from_bytes(bytes: &[u8], schema: &Schema) -> Result<Block> {
    let mut r = Reader::new(bytes);
    let b = get_block(&mut r, schema)?;
    if !r.is_empty() {
        return Err(bad(format!("{} trailing bytes after a block", r.remaining())));
    }
    Ok(b)
}

/// Decode a block, checking it against the schema it claims to belong to.
///
/// The types are read from the record itself (a log written before an `ALTER`
/// still describes itself correctly), but the *physical* kinds must match the
/// schema: a record from another table would otherwise be replayed into
/// columns it does not fit.
pub fn get_block(r: &mut Reader, schema: &Schema) -> Result<Block> {
    use crate::types::PhysicalType;
    let width = count(r.varint()?, "block width")?;
    // A log record is one `INSERT`, and an `INSERT ... SELECT` can hand the
    // session a block of any size at all -- the same shape of bug as the part
    // row count, one layer up.
    let rows = row_count(r.varint()?, "block row count")?;
    if width != schema.len() {
        return Err(bad(format!(
            "record has {width} columns, the table has {}",
            schema.len()
        )));
    }
    let mut cols = Vec::with_capacity(cap(width));
    for ci in 0..width {
        let ty = DataType::parse(r.str()?).map_err(|e| as_corrupt("bad column type", e))?;
        if ty.physical() != schema.ty(ci).physical() {
            return Err(bad(format!(
                "record column {ci} is {ty}, the table declares {}",
                schema.ty(ci)
            )));
        }
        let data = match ty.physical() {
            PhysicalType::U64 => ColumnData::U64(r.u64_slice()?),
            PhysicalType::I64 => {
                ColumnData::I64(r.u64_slice()?.into_iter().map(|x| x as i64).collect())
            }
            PhysicalType::F64 => {
                ColumnData::F64(r.u64_slice()?.into_iter().map(f64::from_bits).collect())
            }
            PhysicalType::Str => {
                let n = row_count(r.varint()?, "string count")?;
                let mut v = Vec::with_capacity(cap(n));
                for _ in 0..n {
                    v.push(r.str()?.into());
                }
                ColumnData::Str(v)
            }
        };
        if data.len() != rows {
            return Err(bad(format!(
                "record column {ci} holds {} values, the record has {rows} rows",
                data.len()
            )));
        }
        let nulls = match r.u8()? {
            0 => None,
            1 => Some(BitSet::from_words(r.u64_slice()?)),
            other => return Err(bad(format!("unknown null-mask tag {other}"))),
        };
        cols.push(match nulls {
            Some(n) => Column::with_nulls(ty, data, n),
            None => Column::new(ty, data),
        });
    }
    Block::new(cols).map_err(|e| as_corrupt("ragged record", e))
}

// ---------------------------------------------------------------------------
// document envelope
// ---------------------------------------------------------------------------

/// Verify and unwrap a `header | framed body | footer` file.
pub(crate) fn doc_body(buf: &[u8]) -> Result<&[u8]> {
    let mut r = Reader::new(buf);
    format::read_header(&mut r)?;
    let at = format::read_footer(buf)?;
    if at != r.pos() as u64 {
        return Err(bad(format!(
            "document body starts at {}, the footer points at {at}",
            r.pos()
        )));
    }
    format::read_framed(&mut r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{hash_key, splitmix64, FP_SEED, G_SHIFT};
    use crate::persist::testkit::*;
    use crate::persist::writer::{self, part_bytes};
    use crate::storage::Stats;
    use crate::types::Value;

    /// `unwrap_err` demands `Debug` on the success type, and `Part`/`Table`
    /// deliberately have none -- printing one would mean printing every packed
    /// word in the file.
    fn must_err<T>(r: Result<T>) -> Error {
        r.err().expect("expected an error")
    }

    fn is_corrupt(e: &Error) -> bool {
        matches!(e, Error::Corruption(_))
    }

    fn roundtrip(p: &Part) -> Part {
        part_from_bytes(&part_bytes(p).expect("serialize")).expect("part must round-trip")
    }

    // -- round trips -------------------------------------------------------

    #[test]
    fn empty_part_roundtrips() {
        let b = crate::types::Block::empty(&schema());
        let p = Part::build(&b, Some(0), Some(0)).unwrap();
        let back = roundtrip(&p);
        assert_eq!(back.n_rows, 0);
        assert_eq!(back.granule_count(), 0);
        assert_eq!(back.ncols, 4);
    }

    #[test]
    fn single_row_part_roundtrips() {
        let p = Part::build(&sample_block(1), Some(0), Some(0)).unwrap();
        let back = roundtrip(&p);
        assert_eq!(back.n_rows, 1);
        assert_eq!(dump(&back), dump(&p));
    }

    #[test]
    fn partial_final_granule_roundtrips() {
        // 1500 rows: one full granule and one 476-row tail.
        let p = sample_part(1_500);
        let back = roundtrip(&p);
        assert_eq!(back.granule_count(), 2);
        assert_eq!(back.granules[0].len, GRANULE_SIZE);
        assert_eq!(back.granules[1].len, 1_500 - GRANULE_SIZE);
        assert_eq!(dump(&back), dump(&p));
    }

    #[test]
    fn part_without_a_sort_or_key_column_roundtrips() {
        let p = Part::build(&sample_block(2_000), None, None).unwrap();
        let back = roundtrip(&p);
        assert_eq!(back.sort_col, None);
        assert_eq!(back.pk_col, None);
        assert!(back.granules.iter().all(|g| g.pk.is_none()));
        assert_eq!(dump(&back), dump(&p));
    }

    #[test]
    fn every_physical_kind_and_type_wrapper_roundtrips() {
        use crate::types::{Column, ColumnBuilder, DataType as T};
        let mut nb = ColumnBuilder::new(T::Nullable(Box::new(T::Int64)));
        for i in 0..8 {
            if i % 3 == 0 {
                nb.push_null();
            } else {
                nb.push_value(&Value::Int(i as i64 - 4)).unwrap();
            }
        }
        let b = crate::types::Block::new(vec![
            Column::u64s(T::UInt64, (0..8).collect()),
            Column::u64s(T::Bool, vec![0, 1, 1, 0, 1, 0, 0, 1]),
            Column::u64s(T::Date, (19_000..19_008).collect()),
            Column::u64s(T::DateTime, (1_700_000_000..1_700_000_008).collect()),
            Column::i64s(T::Int32, (-4..4).collect()),
            Column::f64s(T::Float64, vec![-0.0, 0.0, 1.5, f64::MIN, f64::MAX, 1e-300, -7.25, 3.0]),
            Column::strs(
                T::LowCardinality(Box::new(T::String)),
                vec!["a".into(), "".into(), "\u{4F60}\u{597D}".into(), "a".into(),
                     "z".into(), "a".into(), "mm".into(), "".into()],
            ),
            Column::strs(T::FixedString(4), vec!["abcd".into(); 8]),
            nb.finish(),
        ])
        .unwrap();
        let p = Part::build(&b, Some(0), Some(0)).unwrap();
        let back = roundtrip(&p);
        assert_eq!(dump(&back), dump(&p));
        for c in 0..p.ncols {
            assert_eq!(back.granules[0].columns[c].ty, p.granules[0].columns[c].ty);
        }
    }

    #[test]
    fn constant_columns_roundtrip() {
        use crate::types::{Column, DataType as T};
        // width == 0 is the layout with no payload words at all.
        let b = crate::types::Block::new(vec![
            Column::u64s(T::UInt64, (0..300).collect()),
            Column::u64s(T::UInt32, vec![7; 300]),
            Column::strs(T::String, vec!["same".into(); 300]),
        ])
        .unwrap();
        let p = Part::build(&b, Some(0), Some(0)).unwrap();
        let back = roundtrip(&p);
        assert_eq!(dump(&back), dump(&p));
        assert_eq!(back.granules[0].columns[1].lanes().width(), 0);
    }

    #[test]
    fn wide_random_keys_roundtrip() {
        use crate::types::{Column, DataType as T};
        // 64-bit-wide payloads take the straddled packing layout.
        let mut keys: Vec<u64> = (0..2_000u64).map(splitmix64).collect();
        keys.sort_unstable();
        keys.dedup();
        let b = crate::types::Block::new(vec![Column::u64s(T::UInt64, keys.clone())]).unwrap();
        let p = Part::build(&b, Some(0), Some(0)).unwrap();
        let back = roundtrip(&p);
        assert_eq!(back.granules[0].columns[0].lanes().width(), 64);
        let mut st = Stats::default();
        for &k in &keys {
            assert!(back.find_live(k, hash_key(k, FP_SEED), &mut st, None).is_some(), "key {k}");
        }
    }

    #[test]
    fn deleted_rows_survive() {
        let p = sample_part(3_000);
        let back = roundtrip(&p);
        assert_eq!(back.deleted_count, p.deleted_count);
        assert!(back.deleted_count > 0);
        assert_eq!(deleted_positions(&back), deleted_positions(&p));
        assert_eq!(back.born_live_rows(), p.born_live_rows());
        assert_eq!(back.live_positions(), p.live_positions());
    }

    /// The headline test: a real part, every row, every point lookup, and the
    /// persisted minimal perfect hash still resolving keys.
    #[test]
    fn full_roundtrip_of_a_multi_granule_part() {
        let rows = 5_000;
        let p = sample_part(rows);
        assert!(p.granule_count() >= 5);

        let s = Scratch::new("full-roundtrip");
        let path = s.join("part_000001.gpart");
        writer::write_part(&path, &p).unwrap();
        let back = read_part(&path).unwrap();

        assert_eq!(back.n_rows, p.n_rows);
        assert_eq!(back.ncols, p.ncols);
        assert_eq!(back.granule_count(), p.granule_count());
        assert_eq!(dump(&back), dump(&p), "every cell must survive");

        // Zone maps and the router rebuilt from them.
        for (a, b) in back.granules.iter().zip(&p.granules) {
            assert_eq!((a.sort_min, a.sort_max), (b.sort_min, b.sort_max));
            assert!(a.pk.is_some(), "the key index must be persisted, not dropped");
        }

        // Every key resolves, through the loaded MPH, to the same row.
        let mut st = Stats::default();
        let mut st2 = Stats::default();
        let mut found = 0;
        // Neither part is published, so each still carries its own image.
        let (pd, bd) = (p.born_deletes(), back.born_deletes());
        for (pos, row) in dump(&p) {
            let Value::UInt(key) = row[0] else { panic!("key column changed shape") };
            let fph = hash_key(key, FP_SEED);
            let want = p.find_live(key, fph, &mut st, pd.as_ref());
            assert_eq!(back.find_live(key, fph, &mut st2, bd.as_ref()), want, "key {key} at {pos}");
            found += want.is_some() as usize;
        }
        assert_eq!(found, p.born_live_rows());
        assert!(st2.mph_probes > 0, "lookups must go through the learned-rank index");

        // ...and foreign keys still miss.
        for i in 0..2_000u64 {
            let k = splitmix64(i).saturating_add(1 << 40);
            let fph = hash_key(k, FP_SEED);
            assert_eq!(
                back.find_live(k, fph, &mut st2, bd.as_ref()),
                p.find_live(k, fph, &mut st, pd.as_ref())
            );
        }
    }

    #[test]
    fn the_minimal_perfect_hash_is_stored_verbatim() {
        // Not "an index that works" -- the *same* index. A reader that
        // rebuilt it would be doing a displacement search per granule.
        let p = sample_part(3_000);
        let back = roundtrip(&p);
        for (a, b) in back.granules.iter().zip(&p.granules) {
            let (x, y) = (a.pk.as_ref().unwrap().to_parts(), b.pk.as_ref().unwrap().to_parts());
            assert_eq!((x.min, x.max, x.pmul), (y.min, y.max, y.pmul));
            assert_eq!((x.err_bias, x.ebits), (y.err_bias, y.ebits));
            assert_eq!((x.mph_gs, x.mph_nb, x.mph_n), (y.mph_gs, y.mph_nb, y.mph_n));
            assert_eq!(x.seed_words, y.seed_words, "displacement seeds must be byte-identical");
            assert_eq!(x.fpr_words, y.fpr_words, "fused rank records must be byte-identical");
        }
    }

    #[test]
    fn duplicate_keys_roundtrip_without_an_index() {
        use crate::types::{Column, DataType as T};
        let b = crate::types::Block::new(vec![Column::u64s(
            T::UInt64,
            vec![1, 2, 2, 3, 9, 9, 9],
        )])
        .unwrap();
        let p = Part::build(&b, Some(0), Some(0)).unwrap();
        assert!(p.granules[0].pk.is_none());
        let back = roundtrip(&p);
        assert!(back.granules[0].pk.is_none());
        let mut st = Stats::default();
        assert_eq!(back.find_live(9, hash_key(9, FP_SEED), &mut st, None), Some(4));
    }

    // -- damage ------------------------------------------------------------

    #[test]
    fn every_truncation_is_rejected_without_panicking() {
        let bytes = part_bytes(&sample_part(2_100)).unwrap();
        // Dense near both ends, sampled through the middle: every prefix is
        // ~40k parses otherwise.
        let mut probes: Vec<usize> = (0..600).collect();
        probes.extend((600..bytes.len()).step_by(97));
        probes.extend(bytes.len().saturating_sub(600)..bytes.len());
        for n in probes {
            match part_from_bytes(&bytes[..n]) {
                Ok(_) => panic!("a {n}-byte prefix of a {}-byte part parsed", bytes.len()),
                Err(e) => assert!(is_corrupt(&e), "prefix {n}: wrong error kind: {e}"),
            }
        }
    }

    #[test]
    fn truncation_through_the_file_api_errors_too() {
        let s = Scratch::new("trunc-file");
        let p = sample_part(1_200);
        let path = s.join("p.gpart");
        writer::write_part(&path, &p).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        for n in (0..bytes.len()).step_by(311) {
            std::fs::write(&path, &bytes[..n]).unwrap();
            assert!(read_part(&path).is_err(), "prefix {n} parsed");
        }
    }

    #[test]
    fn a_single_flipped_byte_is_corruption() {
        let s = Scratch::new("bitflip");
        let path = s.join("p.gpart");
        writer::write_part(&path, &sample_part(1_100)).unwrap();
        let good = std::fs::read(&path).unwrap();

        let mid = good.len() / 2;
        let mut bad_bytes = good.clone();
        bad_bytes[mid] ^= 0x01;
        std::fs::write(&path, &bad_bytes).unwrap();
        let e = must_err(read_part(&path));
        assert_eq!(e.code(), "CHECKSUM_MISMATCH", "{e}");
        assert!(e.to_string().contains("p.gpart"), "errors must name the file: {e}");

        std::fs::write(&path, &good).unwrap();
        assert!(read_part(&path).is_ok(), "the undamaged file must still load");
    }

    #[test]
    fn every_single_bit_flip_is_caught() {
        // A small part so the sweep is exhaustive rather than sampled: every
        // byte of the file is covered by the header check, a frame checksum,
        // or the footer checksum, so nothing may slip through.
        let bytes = part_bytes(&sample_part(40)).unwrap();
        for byte in 0..bytes.len() {
            for bit in 0..8 {
                let mut c = bytes.clone();
                c[byte] ^= 1 << bit;
                match part_from_bytes(&c) {
                    Ok(_) => panic!("flip at byte {byte} bit {bit} was accepted"),
                    // The version word is the one field where a flip has a
                    // *diagnosis* rather than just a checksum: it is still
                    // refused, and refused with the right story.
                    Err(e) => assert!(
                        is_corrupt(&e) || e.code() == "FORMAT_VERSION",
                        "byte {byte} bit {bit}: {e}"
                    ),
                }
            }
        }
    }

    #[test]
    fn a_newer_format_version_is_refused() {
        let mut bytes = part_bytes(&sample_part(200)).unwrap();
        let at = format::MAGIC.len();
        bytes[at..at + 4].copy_from_slice(&(format::FORMAT_VERSION + 1).to_le_bytes());
        let e = must_err(part_from_bytes(&bytes));
        assert!(e.to_string().contains("Upgrade granular"), "{e}");
        assert_eq!(e.code(), "FORMAT_VERSION", "{e}");
    }

    #[test]
    fn a_newer_version_in_the_footer_is_refused() {
        // The footer carries its own version and its own checksum, so this
        // has to be forged consistently to prove the version check fires
        // rather than the checksum.
        let mut bytes = part_bytes(&sample_part(200)).unwrap();
        let start = bytes.len() - format::FOOTER_LEN;
        bytes[start + 8..start + 12].copy_from_slice(&(format::FORMAT_VERSION + 3).to_le_bytes());
        let ck = format::checksum(&bytes[start..start + 12]);
        bytes[start + 12..start + 20].copy_from_slice(&ck.to_le_bytes());
        let e = must_err(part_from_bytes(&bytes));
        assert_eq!(e.code(), "FORMAT_VERSION", "{e}");
    }

    #[test]
    fn a_zero_version_is_refused() {
        let mut bytes = part_bytes(&sample_part(64)).unwrap();
        let at = format::MAGIC.len();
        bytes[at..at + 4].copy_from_slice(&0u32.to_le_bytes());
        assert!(part_from_bytes(&bytes).is_err());
    }

    #[test]
    fn random_garbage_never_panics() {
        let mut seed = 0x5EED_1234u64;
        for len in [0usize, 1, 12, 20, 33, 64, 200, 1_000] {
            for _ in 0..64 {
                let buf: Vec<u8> = (0..len)
                    .map(|_| {
                        seed = splitmix64(seed);
                        seed as u8
                    })
                    .collect();
                let _ = part_from_bytes(&buf);
                let _ = catalog_from_bytes(&buf);
                let _ = table_parts_from_bytes(&buf);
                let _ = block_from_bytes(&buf, &schema());
            }
        }
    }

    #[test]
    fn a_part_that_lies_about_its_row_count_is_rejected() {
        // Forge a consistent file whose metadata disagrees with its granules.
        let p = sample_part(100);
        let mut w = format::Writer::new();
        format::write_header(&mut w);
        let goff = w.pos() as u64;
        let bytes = part_bytes(&p).unwrap();
        // Copy the one granule frame verbatim out of a real file.
        let meta_at = format::read_footer(&bytes).unwrap() as usize;
        w.raw(&bytes[format::HEADER_LEN..meta_at]);
        let mut meta = format::Writer::new();
        meta.varint(999); // n_rows, a lie
        meta.varint(p.ncols as u64);
        meta.u8(1);
        meta.varint(0);
        meta.u8(1);
        meta.varint(0);
        meta.varint(0);
        meta.u64_words(&[]);
        meta.varint(1);
        meta.varint(p.granules[0].len as u64);
        meta.u64(p.granules[0].sort_min);
        meta.u64(p.granules[0].sort_max);
        meta.varint(goff);
        let at = w.pos() as u64;
        format::write_framed_aligned(&mut w, meta.as_slice());
        format::write_footer(&mut w, at);
        let e = must_err(part_from_bytes(&w.finish()));
        assert!(e.to_string().contains("granules hold"), "{e}");
    }

    #[test]
    fn an_oversized_granule_is_rejected() {
        let mut meta = format::Writer::new();
        meta.varint(GRANULE_SIZE as u64 + 1);
        meta.varint(1); // part identity
        meta.varint(1);
        meta.u8(0);
        meta.u8(0);
        meta.varint(0);
        meta.u64_words(&[]);
        meta.varint(1);
        meta.varint(GRANULE_SIZE as u64 + 1);
        meta.u64(0);
        meta.u64(0);
        meta.varint(format::HEADER_LEN as u64);
        let e = must_err(decode_meta(meta.as_slice(), 4_096));
        assert!(e.to_string().contains("row maximum"), "{e}");
    }

    // -- the row-count ceiling ---------------------------------------------

    /// Part metadata for `n_rows` rows split into full granules plus a tail,
    /// with every granule descriptor pointing at the same legal offset.
    ///
    /// `decode_meta` never opens a granule body -- it validates the directory
    /// and hands the offsets on -- so this reaches the row-count arithmetic at
    /// any scale for the price of one varint per granule. Building sixteen
    /// million real rows to test a bound on the *count* would test the packer.
    fn meta_for(n_rows: u64) -> Vec<u8> {
        let g = GRANULE_SIZE as u64;
        let full = n_rows / g;
        let tail = n_rows % g;
        let ngranules = full + u64::from(tail != 0);
        let mut m = format::Writer::new();
        m.varint(n_rows);
        m.varint(1); // part identity
        m.varint(1); // ncols
        m.u8(0); // no sort column
        m.u8(0); // no primary key column
        m.varint(0); // deleted count
        m.u64_words(&[]);
        m.varint(ngranules);
        for i in 0..ngranules {
            m.varint(if i < full { g } else { tail });
            m.u64(0);
            m.u64(0);
            // Equal offsets are legal (`offset < prev_end` is what is refused),
            // which is what keeps this cheap.
            m.varint(format::HEADER_LEN as u64);
        }
        m.finish()
    }

    /// The regression: `MAX_COUNT` was a bound on *structure* counts and was
    /// applied to the part row count, so 16,777,217 rows serialized fine and
    /// then loaded as corruption -- discovered at the next restart, after the
    /// merge that produced it had collected the parts it superseded.
    #[test]
    fn a_part_above_the_old_structure_ceiling_still_decodes() {
        let over = MAX_COUNT + 1;
        let meta = decode_meta(&meta_for(over), format::HEADER_LEN as u64 + 1)
            .expect("a part larger than MAX_COUNT rows must load");
        assert_eq!(meta.n_rows as u64, over);
        assert_eq!(meta.granules.len() as u64, over.div_ceil(GRANULE_SIZE as u64));

        // ...and the ceiling that does apply is the row ceiling, reported as
        // such rather than as a granule-count or arithmetic failure. The row
        // count is the first field, so nothing after it needs to be written.
        let mut m = format::Writer::new();
        m.varint(MAX_PART_ROWS + 1);
        let e = must_err(decode_meta(m.as_slice(), 1 << 20));
        assert!(e.to_string().contains("part row count"), "{e}");
        assert!(e.to_string().contains(&MAX_PART_ROWS.to_string()), "{e}");
    }

    /// The header case above proves the arithmetic; this proves the engine.
    ///
    /// 16,777,217 real rows through `Part::build`, `write_part` and `read_part`
    /// -- one row past the ceiling that used to make this exact file load as
    /// corruption. One column and a dense key, because the point is the row
    /// count and not the packer: ~2s and a 48MB scratch file, which is the
    /// cheapest honest version of "the real path works above the old limit".
    #[test]
    fn a_real_part_past_the_old_ceiling_survives_a_write_and_a_read() {
        use crate::types::{Block, Column, DataType as T};
        let n = MAX_COUNT as usize + 1;
        let keys: Vec<u64> = (0..n as u64).collect();
        let p = Part::build(&Block::new(vec![Column::u64s(T::UInt64, keys)]).unwrap(), Some(0), Some(0))
            .unwrap();
        assert_eq!(p.n_rows, n);

        let s = Scratch::new("past-the-ceiling");
        let path = s.join("part_000001.gpart");
        writer::write_part(&path, &p).expect("a 16.7M-row part must write");
        let back = read_part(&path).expect("...and must load back");
        assert_eq!(back.n_rows, n);
        assert_eq!(back.granule_count(), p.granule_count());

        // Not just the header: the far end of the part is really there.
        let mut st = Stats::default();
        for k in [0u64, 1, MAX_COUNT / 2, MAX_COUNT - 1, MAX_COUNT] {
            assert!(back.find_live(k, hash_key(k, FP_SEED), &mut st, None).is_some(), "key {k}");
        }
        assert!(back.find_live(n as u64, hash_key(n as u64, FP_SEED), &mut st, None).is_none());
    }

    /// The granule count is not a structure count either: it is
    /// `n_rows / GRANULE_SIZE`, so bounding it by `MAX_COUNT` would have put
    /// the same cliff back at 17.2 billion rows.
    #[test]
    fn the_granule_ceiling_is_derived_from_the_row_ceiling() {
        assert_eq!(MAX_GRANULES, MAX_PART_ROWS / GRANULE_SIZE as u64);
        const { assert!(MAX_GRANULES > MAX_COUNT) }; // it must not inherit the structure bound
        let mut m = format::Writer::new();
        m.varint(0);
        m.varint(1);
        m.u8(0);
        m.u8(0);
        m.varint(0);
        m.u64_words(&[]);
        m.varint(MAX_GRANULES + 1);
        let e = must_err(decode_meta(m.as_slice(), 1 << 20));
        assert!(e.to_string().contains("granule count"), "{e}");
    }

    /// Structure counts keep the tight bound. Raising every ceiling would have
    /// traded one bug for the allocation-on-a-corrupt-length bug `MAX_COUNT`
    /// exists to prevent.
    #[test]
    fn structure_counts_keep_the_tight_ceiling() {
        assert!(count(MAX_COUNT, "part column count").is_ok());
        assert!(count(MAX_COUNT + 1, "part column count").is_err());
        assert!(row_count(MAX_COUNT + 1, "part row count").is_ok());
        assert!(row_count(MAX_PART_ROWS + 1, "part row count").is_err());

        // The commit record's part list is a structure count: 16M part files
        // in one table is a corrupt length, not a large table.
        let mut w = format::Writer::new();
        writer::put_table_def(&mut w, &table_def("t"));
        w.varint(MAX_COUNT + 1);
        let e = must_err(table_parts_from_bytes(&writer::doc(&w.finish())));
        assert!(e.to_string().contains("part count"), "{e}");
    }

    /// A log record's row count is a row count too: an `INSERT ... SELECT` can
    /// hand the session a block of any size, and a record that will not replay
    /// is data loss discovered at recovery.
    #[test]
    fn a_log_record_may_declare_more_rows_than_the_structure_ceiling() {
        let mut w = format::Writer::new();
        w.varint(1); // one column, matching a one-column schema below
        w.varint(MAX_COUNT + 1); // rows
        w.str("UInt64");
        w.u64_slice(&[1, 2, 3]);
        w.u8(0);
        use crate::types::{Field, Schema};
        let s = Schema::new(vec![Field::new("id", DataType::UInt64)]).unwrap();
        let e = must_err(block_from_bytes(&w.finish(), &s));
        // Rejected for the honest reason -- the column is short -- not for
        // declaring more rows than a structure count may hold.
        assert!(e.to_string().contains("holds 3 values"), "{e}");

        let mut w = format::Writer::new();
        w.varint(1);
        w.varint(MAX_PART_ROWS + 1);
        let e = must_err(block_from_bytes(&w.finish(), &s));
        assert!(e.to_string().contains("block row count"), "{e}");
    }

    #[test]
    fn a_packed_array_too_short_for_its_width_is_rejected() {
        // The check that stands between a corrupt file and an out-of-bounds
        // `get_unchecked` on the point-lookup path.
        let mut w = format::Writer::new();
        w.u64(0); // base
        w.varint(40); // 40 bits/value, straddled layout
        w.u64_words(&[0, 0]); // two words: enough for 3 values, not 1024
        let e = decode_packed(&mut Reader::new(w.as_slice()), 1_024, "column", None).unwrap_err();
        assert!(e.to_string().contains("needs"), "{e}");

        let mut w = format::Writer::new();
        w.u64(0);
        w.varint(65); // wider than a u64
        w.u64_words(&[0, 0]);
        assert!(decode_packed(&mut Reader::new(w.as_slice()), 1, "column", None).is_err());
    }

    #[test]
    fn packed_sizing_matches_the_packer_for_every_width() {
        for width in 0..=64u32 {
            for n in [0usize, 1, 2, 63, 64, 65, 1_000, GRANULE_SIZE] {
                let vals: Vec<u64> = (0..n)
                    .map(|i| match width {
                        0 => 5,
                        64 => splitmix64(i as u64),
                        w => 5 + (splitmix64(i as u64) & ((1u64 << w) - 1)),
                    })
                    .collect();
                let packed = PackedU64::pack(&vals);
                if packed.width() != width {
                    continue; // the sample did not reach the target width
                }
                assert!(
                    packed.words().len() >= packed_words_needed(width, n),
                    "width {width}, n {n}: packer emitted {} words, validator demands {}",
                    packed.words().len(),
                    packed_words_needed(width, n)
                );
            }
        }
    }

    /// Re-encode a key index with individual fields overridden, so each
    /// safety check can be aimed at directly. A checksum-consistent file with
    /// a doctored index is exactly what these checks exist for -- the frame
    /// checksum cannot help, because a forger recomputes it.
    #[allow(clippy::too_many_arguments)]
    fn forge_pk(p: &PkIndexParts, min: u64, max: u64, ebits: u64, nb: u64, n: u64) -> Vec<u8> {
        let mut w = format::Writer::new();
        w.u64(min);
        w.u64(max);
        w.u64(p.pmul);
        w.svarint(p.err_bias as i64);
        w.varint(ebits);
        w.u64(p.mph_gs);
        w.varint(nb);
        w.varint(n);
        w.u64(p.seed_base);
        w.varint(p.seed_width as u64);
        w.u64_words(&p.seed_words);
        w.u64(p.fpr_base);
        w.varint(p.fpr_width as u64);
        w.u64_words(&p.fpr_words);
        w.finish()
    }

    #[test]
    fn a_doctored_key_index_is_rejected_field_by_field() {
        let part = sample_part(500);
        let g = &part.granules[0];
        let col = &g.columns[0];
        let len = g.len;
        let p = g.pk.as_ref().unwrap().to_parts();
        let (nb, n) = (p.mph_nb as u64, p.mph_n as u64);

        // The honest encoding must decode, or the rest proves nothing.
        let good = forge_pk(&p, p.min, p.max, p.ebits as u64, nb, n);
        decode_pk_index(&mut Reader::new(&good), len, col, None).unwrap();

        for (what, bytes, expect) in [
            (
                "shifted lower bound",
                forge_pk(&p, p.min.wrapping_sub(1), p.max, p.ebits as u64, nb, n),
                "do not match the stored key column",
            ),
            (
                "shifted upper bound",
                forge_pk(&p, p.min, p.max.wrapping_add(9), p.ebits as u64, nb, n),
                "do not match the stored key column",
            ),
            (
                "error width past a u64",
                forge_pk(&p, p.min, p.max, 64, nb, n),
                "exceeds 63 bits",
            ),
            (
                "key count disagreeing with the granule",
                forge_pk(&p, p.min, p.max, p.ebits as u64, nb, n - 1),
                "the granule has",
            ),
            (
                "zero buckets",
                forge_pk(&p, p.min, p.max, p.ebits as u64, 0, n),
                "buckets",
            ),
            (
                "more buckets than keys",
                forge_pk(&p, p.min, p.max, p.ebits as u64, n + 1, n),
                "buckets",
            ),
        ] {
            let e = must_err(decode_pk_index(&mut Reader::new(&bytes), len, col, None));
            assert!(is_corrupt(&e), "{what}: {e}");
            assert!(e.to_string().contains(expect), "{what}: {e}");
        }
    }

    #[test]
    fn a_key_index_on_an_empty_granule_is_rejected() {
        // `candidate` clamps a row into `[0, len - 1]`, which has no meaning
        // for an empty granule.
        let part = sample_part(500);
        let p = part.granules[0].pk.as_ref().unwrap().to_parts();
        let col = &part.granules[0].columns[0];
        let bytes = forge_pk(&p, p.min, p.max, p.ebits as u64, p.mph_nb as u64, 0);
        let e = must_err(decode_pk_index(&mut Reader::new(&bytes), 0, col, None));
        assert!(e.to_string().contains("empty granule"), "{e}");
    }

    #[test]
    fn a_key_index_with_a_short_record_array_is_rejected() {
        let part = sample_part(500);
        let g = &part.granules[0];
        let mut p = g.pk.as_ref().unwrap().to_parts();
        p.fpr_words.truncate(2);
        let bytes = forge_pk(&p, p.min, p.max, p.ebits as u64, p.mph_nb as u64, p.mph_n as u64);
        let e = must_err(decode_pk_index(&mut Reader::new(&bytes), g.len, &g.columns[0], None));
        assert!(e.to_string().contains("key index records"), "{e}");
    }

    #[test]
    fn a_missing_file_is_an_io_error_not_a_panic() {
        let s = Scratch::new("missing");
        let e = must_err(read_part(&s.join("nope.gpart")));
        assert_eq!(e.code(), "IO_ERROR", "{e}");
    }

    // -- tables ------------------------------------------------------------

    #[test]
    fn table_roundtrip_preserves_parts_and_rows() {
        let s = Scratch::new("table-roundtrip");
        let t = sample_table("hits", &[1_500, 800, 300]);
        let tsnap = t.snapshot();
        let before: Vec<Vec<(usize, Vec<Value>)>> =
            tsnap.parts().iter().map(|p| dump(p)).collect();
        writer::write_table(s.path(), &t, 0).unwrap();

        let mut back = read_table(s.path(), "hits").unwrap();
        assert_eq!(back.part_count(), 3);
        assert_eq!(back.def, t.def);
        let after: Vec<Vec<(usize, Vec<Value>)>> = {
            let bsnap = back.snapshot();
            bsnap.parts().iter().map(|p| dump(p)).collect()
        };
        assert_eq!(after, before);
        assert_eq!(back.row_count().unwrap(), 1_500 + 800 + 300);
    }

    #[test]
    fn read_table_rejects_a_renamed_directory() {
        let s = Scratch::new("table-renamed");
        let t = sample_table("hits", &[100]);
        writer::write_table(s.path(), &t, 0).unwrap();
        std::fs::rename(s.join("hits"), s.join("clicks")).unwrap();
        let e = must_err(read_table(s.path(), "clicks"));
        assert!(e.to_string().contains("definition of table `hits`"), "{e}");
    }

    #[test]
    fn read_table_of_a_missing_directory_is_an_io_error() {
        let s = Scratch::new("table-missing");
        assert_eq!(must_err(read_table(s.path(), "ghost")).code(), "IO_ERROR");
    }

    #[test]
    fn a_commit_record_naming_a_foreign_file_is_refused() {
        for name in ["../../etc/passwd", "/etc/passwd", "wal.log", "part_1.gpart.bak"] {
            let doc = writer::table_doc(&table_def("t"), &[name.to_string()], 12);
            let e = table_parts_from_bytes(&doc).unwrap_err();
            assert!(e.to_string().contains("is not a part file name"), "{name}: {e}");
        }
    }

    #[test]
    fn a_truncated_commit_record_is_rejected() {
        let doc = writer::table_doc(&table_def("t"), &["part_000001.gpart".into()], 12);
        for n in 0..doc.len() {
            assert!(table_parts_from_bytes(&doc[..n]).is_err(), "prefix {n}");
        }
    }

    #[test]
    fn a_definition_with_out_of_range_key_columns_is_rejected() {
        let mut def = table_def("t");
        def.order_by = vec![0];
        let mut w = format::Writer::new();
        writer::put_table_def(&mut w, &def);
        let mut bytes = w.finish();
        // The ORDER BY column index is the last varint before the key list;
        // rather than patch bytes, rebuild with a bad index directly.
        bytes.clear();
        let mut w = format::Writer::new();
        w.str("t");
        w.varint(1);
        w.str("id");
        w.str("UInt64");
        w.u8(0);
        w.varint(1);
        w.varint(9); // ORDER BY column 9 of a 1-column table
        let e = get_table_def(&mut Reader::new(w.as_slice())).unwrap_err();
        assert!(e.to_string().contains("out of range"), "{e}");
    }

    #[test]
    fn a_duplicate_column_name_is_corruption_not_a_bind_error() {
        let mut w = format::Writer::new();
        w.str("t");
        w.varint(2);
        w.str("id");
        w.str("UInt64");
        w.u8(0);
        w.str("id");
        w.str("UInt64");
        w.u8(0);
        let e = get_table_def(&mut Reader::new(w.as_slice())).unwrap_err();
        assert!(is_corrupt(&e), "{e}");
        assert!(e.to_string().contains("bad schema"), "{e}");
    }

    #[test]
    fn an_unknown_type_or_engine_is_corruption() {
        let mut w = format::Writer::new();
        w.str("t");
        w.varint(1);
        w.str("id");
        w.str("Blob");
        w.u8(0);
        let e = get_table_def(&mut Reader::new(w.as_slice())).unwrap_err();
        assert!(is_corrupt(&e) && e.to_string().contains("bad column type"), "{e}");

        let mut w = format::Writer::new();
        w.str("t");
        w.varint(1);
        w.str("id");
        w.str("UInt64");
        w.u8(0);
        w.varint(0);
        w.varint(0);
        w.u8(0);
        w.str("Kafka");
        let e = get_table_def(&mut Reader::new(w.as_slice())).unwrap_err();
        assert!(is_corrupt(&e) && e.to_string().contains("bad engine"), "{e}");
    }

    #[test]
    fn a_definition_naming_a_path_is_refused() {
        // A name from a `CATALOG` file becomes a directory component during
        // recovery, so it is validated on the way in, not on the way out.
        for name in ["../escape", "a/b", "..", ".", "", ".hidden", "a\\b"] {
            let mut w = format::Writer::new();
            w.str(name);
            w.varint(0);
            let e = must_err(get_table_def(&mut Reader::new(w.as_slice())));
            assert!(e.to_string().contains("not a usable table name"), "{name}: {e}");
        }
        let mut w = format::Writer::new();
        w.varint(1);
        w.str("../evil");
        let doc = writer::doc(&w.finish());
        let e = must_err(catalog_from_bytes(&doc));
        assert!(e.to_string().contains("not a usable database name"), "{e}");
    }

    #[test]
    fn a_dictionary_on_the_wrong_column_kind_is_rejected() {
        let mut w = format::Writer::new();
        w.str("UInt64");
        w.varint(2);
        w.u64(9);
        w.u64(0); // base
        w.varint(0); // width
        w.u64_words(&[0, 0]);
        w.u8(1); // a dictionary, on an integer column
        w.bytes(b"ab");
        w.u32_slice(&[0, 1, 2]);
        w.u8(0);
        let e = must_err(decode_column(&mut Reader::new(w.as_slice()), 2, None));
        assert!(e.to_string().contains("carries a string dictionary"), "{e}");

        let mut w = format::Writer::new();
        w.str("String");
        w.varint(2);
        w.u64(0);
        w.u64(0);
        w.varint(0);
        w.u64_words(&[0, 0]);
        w.u8(0); // no dictionary, on a string column
        w.u8(0);
        let e = must_err(decode_column(&mut Reader::new(w.as_slice()), 2, None));
        assert!(e.to_string().contains("no dictionary"), "{e}");
    }

    #[test]
    fn catalog_documents_roundtrip() {
        let roster = vec![
            ("default".to_string(), vec![table_def("a"), table_def("b")]),
            ("analytics".to_string(), vec![]),
        ];
        let bytes = writer::catalog_doc(&roster, 0xDEAD_BEEF_F00D);
        let (back, instance) = catalog_from_bytes(&bytes).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].0, "default");
        assert_eq!(back[0].1, roster[0].1);
        assert!(back[1].1.is_empty());
        assert_eq!(instance, 0xDEAD_BEEF_F00D, "the instance id round-trips");

        // A `CATALOG` from a build with no instance id: the field is the last
        // thing in the body, so dropping it is exactly what an older writer
        // produced -- and it has to read as "unstamped", not as damage.
        let mut w = format::Writer::new();
        w.varint(0);
        let (back, instance) = catalog_from_bytes(&writer::doc(&w.finish())).unwrap();
        assert!(back.is_empty());
        assert_eq!(instance, 0, "an unstamped catalog reads as 0, not an error");
    }

    #[test]
    fn a_document_whose_footer_points_elsewhere_is_rejected() {
        let mut bytes = writer::catalog_doc(&[], 0);
        let start = bytes.len() - format::FOOTER_LEN;
        bytes[start..start + 8].copy_from_slice(&(format::HEADER_LEN as u64 + 1).to_le_bytes());
        let ck = format::checksum(&bytes[start..start + 12]);
        bytes[start + 12..start + 20].copy_from_slice(&ck.to_le_bytes());
        let e = catalog_from_bytes(&bytes).unwrap_err();
        assert!(e.to_string().contains("footer points at"), "{e}");
    }

    // ---- adversarial review additions ------------------------------------

    /// Build a one-granule, one-string-column part file with a hand-chosen
    /// FOR base / dictionary. Every frame checksum and the footer are computed
    /// honestly, exactly as a forger (or a writer from another build) would.
    fn forge_string_part(base: u64, blob: &[u8], offsets: &[u32]) -> Vec<u8> {
        // Written straight into the granule writer rather than assembled in a
        // detached buffer and spliced: `u64_words` pads relative to the start
        // of the buffer it is writing into, so a column built at offset 0 and
        // pasted in at offset 2 aligns its words to the wrong place. The real
        // writer builds columns directly into the granule for the same reason.
        let mut g = format::Writer::new();
        g.varint(1); // granule rows
        g.varint(1); // granule columns
        g.str("String");
        g.varint(1); // column length
        g.u64(base); // max_lane
        g.u64(base); // FOR base
        g.varint(0); // width 0 => constant column, lane(i) == base
        g.u64_words(&[0, 0]);
        g.u8(1);
        g.bytes(blob);
        g.u32_slice(offsets);
        g.u8(0); // no null mask
        g.u8(0); // no key index
        let g = g.finish();

        let mut w = format::Writer::new();
        format::write_header(&mut w);
        let goff = w.pos() as u64;
        format::write_framed_aligned(&mut w, &g);

        let mut meta = format::Writer::new();
        meta.varint(1); // n_rows
        meta.varint(1); // part identity
        meta.varint(1); // ncols
        meta.u8(0); // no sort column
        meta.u8(0); // no primary key column
        meta.varint(0); // deleted count
        meta.u64_words(&[]);
        meta.varint(1); // one granule
        meta.varint(1); // its row count
        meta.u64(0);
        meta.u64(0);
        meta.varint(goff);

        let meta_at = w.pos() as u64;
        format::write_framed_aligned(&mut w, meta.as_slice());
        format::write_footer(&mut w, meta_at);
        w.finish()
    }

    /// `decode_column` never ties a string column's lane range to the size of
    /// the dictionary it ships with, so a file can declare a dictionary code of
    /// `u64::MAX`. `StringDict::get` computes `code as usize + 1`, which
    /// overflows, and reading the cell panics instead of returning
    /// `Error::Corruption`.
    #[test]
    fn adversarial_out_of_range_dictionary_code_panics_on_read() {
        let bytes = forge_string_part(u64::MAX, b"a", &[0, 1]);
        let part = part_from_bytes(&bytes).expect("the forged part is accepted by the reader");
        assert_eq!(part.n_rows, 1);
        let hit = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            part.value_at(0, 0)
        }));
        assert!(
            hit.is_err(),
            "expected the documented behaviour (no panic); got {:?}",
            hit.ok()
        );
    }

    /// Dictionary offsets that fall inside a multi-byte sequence must be
    /// rejected.
    ///
    /// `StringDict::get` slices the blob at adjacent offsets and calls
    /// `from_utf8_unchecked`, so accepting a split offset would hand out a
    /// `&str` over invalid UTF-8 -- undefined behaviour, reachable from a
    /// corrupt file. Validating the blob as a whole is not sufficient: the
    /// bytes are perfectly good UTF-8, it is the *boundaries* that are wrong.
    #[test]
    fn dictionary_offsets_that_split_a_utf8_sequence_are_rejected() {
        let blob = "\u{65E5}".as_bytes().to_vec(); // 3 bytes: E6 97 A5
        assert_eq!(blob.len(), 3);
        let column = |offsets: &[u32]| {
            let mut w = format::Writer::new();
            w.str("String");
            w.varint(1);
            w.u64(0);
            w.u64(0);
            w.varint(0);
            w.u64_words(&[0, 0]);
            w.u8(1);
            w.bytes(&blob);
            w.u32_slice(offsets);
            w.u8(0);
            decode_column(&mut Reader::new(w.as_slice()), 1, None)
        };

        // entry 0 would end after the first byte of a 3-byte character
        let e = must_err(column(&[0, 1, 3]));
        assert_eq!(e.code(), "CHECKSUM_MISMATCH", "{e}");
        assert!(e.to_string().contains("splits a UTF-8 codepoint"), "{e}");

        // honest boundaries still load, and read back correctly
        let c = column(&[0, 3]).expect("valid boundaries must be accepted");
        assert_eq!(c.dict().unwrap().get(0), "\u{65E5}");
    }

    /// The delete bitmap's population is never checked against the
    /// `deleted_count` the file declares, so a part can load with a live-row
    /// count that contradicts the rows it will actually yield.
    #[test]
    fn adversarial_delete_count_may_contradict_the_bitmap() {
        let p = sample_part(100);
        let mut w = format::Writer::new();
        format::write_header(&mut w);
        let goff = w.pos() as u64;
        let real = part_bytes(&p).unwrap();
        let meta_at_real = format::read_footer(&real).unwrap() as usize;
        w.raw(&real[format::HEADER_LEN..meta_at_real]);

        let mut meta = format::Writer::new();
        meta.varint(p.n_rows as u64);
        meta.varint(p.pid); // part identity
        meta.varint(p.ncols as u64);
        meta.u8(1);
        meta.varint(0);
        meta.u8(1);
        meta.varint(0);
        meta.varint(p.n_rows as u64); // "every row is deleted"
        meta.u64_words(&[]); // ...but the bitmap is empty
        meta.varint(1);
        meta.varint(p.granules[0].len as u64);
        meta.u64(p.granules[0].sort_min);
        meta.u64(p.granules[0].sort_max);
        meta.varint(goff);
        let at = w.pos() as u64;
        format::write_framed_aligned(&mut w, meta.as_slice());
        format::write_footer(&mut w, at);

        let back = part_from_bytes(&w.finish()).expect("accepted");
        assert_eq!(back.born_live_rows(), 0, "the part claims nothing is live");
        assert_eq!(
            back.live_positions().len(),
            100,
            "...but every row is still handed out: live_rows() and the scan disagree"
        );
    }

    #[test]
    fn granule_positions_stay_granule_major_after_a_load() {
        let p = sample_part(2_600);
        let back = roundtrip(&p);
        let bd = back.born_deletes();
        let mut st = Stats::default();
        for (gi, g) in back.granules.iter().enumerate() {
            for i in 0..g.len {
                let pos = (gi << G_SHIFT) + i;
                let Value::UInt(k) = back.value_at(pos, 0) else { panic!() };
                if back.deleted.get(pos) {
                    continue;
                }
                assert_eq!(
                    back.find_live(k, hash_key(k, FP_SEED), &mut st, bd.as_ref()),
                    Some(pos)
                );
            }
        }
    }

    // ---- quarantine -------------------------------------------------------

    /// A three-part table on disk, with part `which` (0-based) bit-flipped in
    /// the middle. Hands back the scratch, the table directory and the name of
    /// the file that was damaged.
    fn damaged_table(tag: &str, which: usize) -> (Scratch, PathBuf, String) {
        let s = Scratch::new(tag);
        let t = sample_table("hits", &[1_500, 800, 2_000]);
        writer::write_table(s.path(), &t, 0).unwrap();
        let tdir = s.join("hits");
        let files = store::list_part_files(&tdir).unwrap();
        assert_eq!(files.len(), 3, "fixture must write three parts");
        let name = files[which].1.clone();
        let p = tdir.join(&name);
        let mut bytes = std::fs::read(&p).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0x20;
        std::fs::write(&p, &bytes).unwrap();
        (s, tdir, name)
    }

    /// The change this module exists for: one bad file is a quarantined part,
    /// not a failed open. The other two parts still decode, and the damage is
    /// reported rather than swallowed.
    #[test]
    fn a_damaged_part_is_quarantined_not_propagated() {
        for which in 0..3 {
            let (_s, tdir, name) = damaged_table("quarantine", which);
            let img = read_table_image(&tdir).expect("a damaged part must not fail the open");
            assert_eq!(img.part_files.len(), 3, "the commit record still names three");
            assert_eq!(img.parts.len(), 2, "the healthy parts must still decode");
            assert_eq!(img.damaged.len(), 1);
            assert_eq!(img.damaged[0].file, name);
            assert!(img.damaged[0].why.contains(&name), "the reason must name the file");
            assert!(
                img.damaged[0].why.contains("checksum"),
                "the reason must be the reader's own: {}",
                img.damaged[0].why
            );
            claim_damage(&tdir);
        }
    }

    /// The hand-off is a queue of one per table directory: the loader claims
    /// it once, and nothing claims it twice.
    #[test]
    fn damage_is_handed_over_exactly_once() {
        let (_s, tdir, name) = damaged_table("handoff", 1);
        read_table_image(&tdir).unwrap();
        let claimed = claim_damage(&tdir).expect("the loader must find the record");
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].file, name);
        assert!(claim_damage(&tdir).is_none(), "a claimed record must not be claimable again");
    }

    /// ...and a table that reads clean clears whatever an earlier read of the
    /// same directory left, or a repaired database would stay quarantined.
    #[test]
    fn a_clean_reload_clears_an_earlier_record() {
        let (s, tdir, name) = damaged_table("repair", 2);
        assert_eq!(read_table_image(&tdir).unwrap().damaged.len(), 1);

        // Put the file back the way `write_table` had it.
        let good = sample_table("hits", &[1_500, 800, 2_000]);
        let s2 = Scratch::new("repair-src");
        writer::write_table(s2.path(), &good, 0).unwrap();
        std::fs::copy(s2.join("hits").join(&name), tdir.join(&name)).unwrap();

        let img = read_table_image(&tdir).unwrap();
        assert!(img.damaged.is_empty(), "the repaired table must read clean");
        assert!(claim_damage(&tdir).is_none(), "a stale record survived the repair");
        drop(s);
    }

    /// `read_table` has nowhere to put a quarantine -- it returns a bare
    /// `Table` -- so it must refuse rather than hand back one short of a part.
    #[test]
    fn read_table_refuses_a_damaged_table() {
        let (s, tdir, name) = damaged_table("read-table", 0);
        let e = must_err(read_table(s.path(), "hits"));
        assert!(is_corrupt(&e), "{e}");
        assert!(e.to_string().contains(&name), "must name the part file: {e}");
        // Drained even though the call failed: a record nobody claims would
        // outlive the load that produced it.
        assert!(claim_damage(&tdir).is_none(), "the refusal must still drain the hand-off");
    }

    /// An undamaged table leaves nothing behind at all: the healthy path must
    /// not so much as touch the hand-off.
    #[test]
    fn a_healthy_table_records_no_damage() {
        let s = Scratch::new("healthy-image");
        writer::write_table(s.path(), &sample_table("hits", &[900]), 0).unwrap();
        let img = read_table_image(&s.join("hits")).unwrap();
        assert!(img.damaged.is_empty());
        assert_eq!(img.parts.len(), 1);
        assert!(claim_damage(&s.join("hits")).is_none());
    }
}

#[cfg(test)]
mod mapped {
    //! The mmap path: same rows, different memory.
    //!
    //! Round-tripping is not the interesting property here -- the copying path
    //! already did that. What these pin is that the mapped path *is* mapped:
    //! that lanes point into the file rather than the heap, that the mapping
    //! outlives the borrow, and that the alignment the format pays for
    //! actually holds all the way down to the address.

    use super::*;
    use crate::persist::testkit::*;
    use crate::persist::write_part;

    fn written(tag: &str, rows: usize) -> (Scratch, std::path::PathBuf, Part) {
        let s = Scratch::new(tag);
        let p = s.join("part.gr");
        let src = sample_part(rows);
        write_part(&p, &src).expect("write");
        (s, p, src)
    }

    /// A part read from disk holds no raw lane bytes on the heap.
    ///
    /// Every column is one of two things: mapped, or compressed. What must
    /// never happen is the third case -- raw words copied onto the heap --
    /// because that is the one where the reader paid for a copy and got
    /// nothing for it.
    #[test]
    fn a_part_read_from_disk_holds_no_raw_lanes_on_the_heap() {
        let (_s, path, _) = written("borrows", 5_000);
        let part = read_part(&path).expect("read");

        let (mut mapped, mut packed, mut total) = (0, 0, 0);
        for g in &part.granules {
            for c in &g.columns {
                total += 1;
                // A mapped column costs a pointer; a compressed one had to be
                // decoded, and is smaller than the words it stands for.
                if c.lanes().is_mapped() {
                    mapped += 1;
                } else {
                    packed += 1;
                }
            }
        }
        assert!(total > 0, "the fixture produced no columns");
        assert_eq!(mapped + packed, total);
        assert!(mapped > 0, "nothing was mapped: the alignment chain is broken");
    }

    /// The mapped part and the copied part are the same part.
    #[test]
    fn mapping_a_part_changes_nothing_a_query_can_see() {
        let (_s, path, src) = written("identical", 5_000);
        let copied = part_from_bytes(&store::read_file(&path).unwrap()).expect("copy path");
        let mapped = read_part(&path).expect("map path");

        assert!(mapped.granules.iter().any(|g| g.columns[0].lanes().is_mapped()));
        assert!(!copied.granules.iter().any(|g| g.columns[0].lanes().is_mapped()));
        assert_eq!(dump(&mapped), dump(&copied));
        assert_eq!(dump(&mapped), dump(&src));
        assert_eq!(mapped.n_rows, src.n_rows);
        assert_eq!(deleted_positions(&mapped), deleted_positions(&src));
    }

    /// The payload stops being heap. This is the whole point: a part's lanes
    /// are the bulk of its bytes, and mapped they cost page cache the kernel
    /// can reclaim rather than resident memory it cannot.
    #[test]
    fn mapping_keeps_the_lane_payload_off_the_heap() {
        let (_s, path, _) = written("heap", 20_000);
        let copied = part_from_bytes(&store::read_file(&path).unwrap()).unwrap();
        let mapped = read_part(&path).unwrap();

        let lanes = |p: &Part| -> usize {
            p.granules.iter().flat_map(|g| &g.columns).map(|c| c.lanes().bytes()).sum()
        };
        let (m, c) = (lanes(&mapped), lanes(&copied));
        // Not a ratio: compressed columns are decoded on both paths and cost
        // the same either way, so the saving is bounded by how much of the
        // part stayed raw. What has to hold is that mapping never costs more.
        assert!(m < c, "mapped lanes cost {m} bytes against {c} copied");
        let per_mapped_col = mapped
            .granules
            .iter()
            .flat_map(|g| &g.columns)
            .filter(|c| c.lanes().is_mapped())
            .map(|c| c.lanes().bytes())
            .max()
            .expect("something must be mapped");
        assert!(per_mapped_col <= 64, "a mapped column costs {per_mapped_col} bytes of heap");
    }

    /// The mapping outlives every borrow of the file handle and the `Arc` the
    /// reader was handed: a column keeps its own clone alive.
    #[test]
    fn lanes_stay_readable_after_the_reader_drops_its_handle() {
        let (_s, path, src) = written("outlives", 3_000);
        let rows = {
            let map = std::sync::Arc::new(Mmap::open(&path).unwrap());
            let part = part_from_mmap(std::sync::Arc::clone(&map)).unwrap();
            drop(map); // the reader's handle goes; the columns' clones remain
            dump(&part)
        };
        assert_eq!(rows, dump(&src));
    }

    /// Deleting the file under an open mapping does not disturb it -- the
    /// pages are attached to the inode, not the directory entry. Compaction
    /// unlinks parts while queries are still reading them.
    #[test]
    fn an_unlinked_part_is_still_readable_through_its_mapping() {
        let (_s, path, src) = written("unlinked", 3_000);
        let part = read_part(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(dump(&part), dump(&src));
    }

    /// Alignment is a runtime property, not a hope: every mapped lane array
    /// has to land on an 8-byte boundary or the cast in `decode_packed` would
    /// be undefined behaviour.
    #[test]
    fn every_mapped_lane_array_is_eight_byte_aligned() {
        let (_s, path, _) = written("aligned", 8_000);
        let part = read_part(&path).unwrap();
        for (gi, g) in part.granules.iter().enumerate() {
            for (ci, c) in g.columns.iter().enumerate() {
                let p = c.lanes().as_slice().as_ptr() as usize;
                assert_eq!(p % 8, 0, "granule {gi} column {ci} at {p:#x}");
            }
        }
    }

    /// A part whose file is truncated must be rejected, not mapped and read
    /// off the end.
    #[test]
    fn a_truncated_part_file_is_rejected_not_mapped() {
        let (s, path, _) = written("truncated", 3_000);
        let full = store::read_file(&path).unwrap();
        for cut in [full.len() / 2, full.len() - 1] {
            let p = s.join("cut.gr");
            std::fs::write(&p, &full[..cut]).unwrap();
            assert!(read_part(&p).is_err(), "accepted a file cut to {cut}");
        }
    }

}

