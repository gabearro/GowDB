//! CSV and TSV in and out: `INSERT ... FROM INFILE` and `SELECT ... INTO OUTFILE`.
//!
//! ## Why
//!
//! Before this, the only way to get a row into the engine was an `INSERT ...
//! VALUES` statement held entirely in memory as SQL text, and the only way to
//! get one out was a box-drawn table. There was no way to load a dataset and no
//! way to hand a result to another program, which rules out essentially every
//! real workload.
//!
//! ## The import streams, and that is the whole design
//!
//! Reading a 10 GB file into RAM to insert it would repeat the exact mistake
//! this phase exists to fix. So:
//!
//!   * bytes arrive through a **fixed 256 KiB window** that is compacted and
//!     refilled, never grown to the file (it grows only for a single record
//!     longer than the window, and refuses one longer than [`MAX_RECORD`]);
//!   * a field is a `(start, end)` pair **into that window** — no `String` per
//!     field, no `Vec` per row, and an unquoted field is handed to the typed
//!     parser as the original bytes;
//!   * rows accumulate into one block of at most `max_insert_block_size` rows
//!     *and* at most a quarter of the memory budget, whichever comes first, so
//!     the resident set is bounded by a setting rather than by the file;
//!   * each block goes straight to storage. Above
//!     `storage::table::BULK_INSERT_THRESHOLD` a block is packed directly into
//!     a part instead of being buffered in the delta, which is why the default
//!     block is 64k rows.
//!
//! Measured: importing a 28.3 MB file at `max_insert_block_size = 16384` holds
//! **768 KB**, and `tests/settings_and_io.rs` asserts that high-water mark
//! stays under a budget 38x smaller than the file.
//!
//! ## Measured throughput
//!
//! 1M rows x 4 columns (`UInt64, String, Float64, UInt32`), release, A/B
//! interleaved in one loop against the same rows as one SQL `INSERT ... VALUES`,
//! best of 5, twice (this machine swings 30% on identical code, so the spread is
//! quoted rather than a single number):
//!
//! ```text
//!   INSERT ... VALUES (37.7 MB statement)   0.582-0.638 s   1.6-1.7 M rows/s   ~62 MB/s
//!   INSERT ... FROM INFILE (33.8 MB CSV)    0.129-0.139 s   7.2-7.7 M rows/s  ~250 MB/s
//! ```
//!
//! **4.5x**, and the gap is not parser cleverness: the VALUES path lexes 37.7 MB
//! of SQL into `Token`s, builds an `Expr` per cell, and holds the statement, the
//! AST and the whole block at once. This path holds 256 KiB and never builds a
//! `Value`.
//!
//! ## Quoting
//!
//! RFC 4180 on both sides, with the two conventions that make a round trip
//! exact: **a quoted field is never NULL** (that is the only way CSV can tell
//! NULL from the empty string), and the writer quotes any field that would
//! otherwise read back as NULL — including a string whose text happens to be
//! the null representation itself.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use crate::common::{BitSet, Error, Result};
use crate::exec::operators::QueryContext;
use crate::session::{ResultSet, Session, StreamItem};
use crate::settings::Settings;
use crate::sql::lexer::{Spanned, Token};
use crate::types::{
    civil_from_days, Block, Column, ColumnData, DataType, PhysicalType, Schema,
};

/// The read window. Big enough that the `read` syscall is amortized to nothing
/// at 217 MB/s (one per 256 KiB is ~830 per second) and small enough to stay a
/// rounding error against the row block it feeds — the block, not this, is what
/// the memory budget is really bounding.
const WINDOW: usize = 256 << 10;

/// A single record longer than this is refused rather than grown into. 64 MiB
/// is far past any real CSV row and stops a file with one unbalanced quote
/// from turning into an allocation the size of the file — which is precisely
/// the failure this module exists to avoid.
pub const MAX_RECORD: usize = 64 << 20;

// ------------------------------------------------------------------ dialect

/// Everything that distinguishes CSV from TSV, and one CSV from another.
#[derive(Clone, Debug)]
pub struct Dialect {
    pub delim: u8,
    /// The exact unquoted text that means NULL.
    pub null: Box<str>,
    /// The first line names the columns.
    pub header: bool,
}

impl Dialect {
    /// The dialect a `FORMAT` name asks for, starting from the session's
    /// settings. `CSV`/`TSV` keep the session's header setting; the
    /// `WithNames`/`WithoutNames` spellings pin it, which is what makes a
    /// script independent of whatever `SET` ran before it.
    pub fn named(name: &str, cfg: &Settings) -> Result<Dialect> {
        let mut d = Dialect {
            delim: cfg.format_csv_delimiter,
            null: cfg.format_csv_null.clone(),
            header: cfg.input_format_with_names_use_header,
        };
        let (base, header) = match name.to_ascii_lowercase().as_str() {
            "csv" => ("csv", d.header),
            "tsv" | "tabseparated" => ("tsv", d.header),
            "csvwithnames" => ("csv", true),
            "tsvwithnames" | "tabseparatedwithnames" => ("tsv", true),
            "csvwithoutnames" => ("csv", false),
            "tsvwithoutnames" | "tabseparatedwithoutnames" => ("tsv", false),
            _ => {
                return Err(Error::unsupported(format!(
                    "unknown FORMAT `{name}`: this engine reads and writes CSV, TSV, \
                     CSVWithNames, TSVWithNames, CSVWithoutNames and TSVWithoutNames"
                )))
            }
        };
        // A TSV whose separator is still `,` because the session set it that
        // way would be a silently wrong file, so the format wins.
        if base == "tsv" {
            d.delim = b'\t';
        }
        d.header = header;
        Ok(d)
    }

    /// The dialect a path implies when no `FORMAT` was given: `.tsv`/`.tab`
    /// mean tab-separated, everything else takes the session's settings.
    pub fn for_path(path: &str, cfg: &Settings) -> Dialect {
        let tsv = path.rsplit('.').next().is_some_and(|e| {
            e.eq_ignore_ascii_case("tsv") || e.eq_ignore_ascii_case("tab")
        });
        Dialect {
            delim: if tsv { b'\t' } else { cfg.format_csv_delimiter },
            null: cfg.format_csv_null.clone(),
            header: cfg.input_format_with_names_use_header,
        }
    }
}

// ------------------------------------------------------------- the byte scan

/// A field's bytes live either in the read window or, if they had to be
/// unescaped, in the scratch buffer. The discriminator rides in the high bit
/// of the start offset rather than in a third field: records are bounded by
/// [`MAX_RECORD`] (2^26), so 31 bits is room to spare and the pair stays 8
/// bytes — which matters, because there is one per field per row.
const IN_SCRATCH: u32 = 1 << 31;

/// SWAR scan for the first byte equal to `a` or `c`, from `i`.
///
/// Eight bytes per iteration with the classic `x - 0x01.. & !x & 0x80..` zero
/// test.
///
/// A/B interleaved through a temporary switch, best of 5 per side, on the 1M
/// row / 33.8 MB fixture above: **0.124 s** against the byte-at-a-time loop's
/// **0.137 s**, so 1.10x on the *entire* import including the storage write.
/// A smaller win than it looks like it should be, and the reason is worth
/// recording: CSV fields here average 9 bytes, so the 8-byte stride retires
/// barely more than one iteration per field, and what is left is the typed
/// decode and the column write. Do not expect more from widening it to 16.
#[inline]
fn find2(b: &[u8], mut i: usize, a: u8, c: u8) -> usize {
    const ONES: u64 = 0x0101_0101_0101_0101;
    const HIGH: u64 = 0x8080_8080_8080_8080;
    let (ma, mc) = (ONES.wrapping_mul(a as u64), ONES.wrapping_mul(c as u64));
    while i + 8 <= b.len() {
        let w = u64::from_le_bytes(b[i..i + 8].try_into().expect("8 bytes"));
        let (xa, xc) = (w ^ ma, w ^ mc);
        let hit = ((xa.wrapping_sub(ONES) & !xa) | (xc.wrapping_sub(ONES) & !xc)) & HIGH;
        if hit != 0 {
            return i + (hit.trailing_zeros() as usize >> 3);
        }
        i += 8;
    }
    while i < b.len() && b[i] != a && b[i] != c {
        i += 1;
    }
    i
}

#[inline]
fn find1(b: &[u8], i: usize, a: u8) -> usize {
    find2(b, i, a, a)
}

enum Scan {
    /// Bytes consumed, terminator included.
    Done(usize),
    /// The window does not hold a whole record yet.
    Need,
}

/// Split one record out of `b`, which starts at a field boundary.
///
/// Re-scans from the start of the record after a refill rather than keeping a
/// resumable state machine. That costs one extra pass over one record per
/// 256 KiB — unmeasurable — and it is the reason the quoted path can be read
/// straight through instead of as a set of resumable states.
fn scan_record(
    b: &[u8],
    eof: bool,
    delim: u8,
    ranges: &mut Vec<(u32, u32)>,
    scratch: &mut Vec<u8>,
) -> Result<Scan> {
    ranges.clear();
    scratch.clear();
    let n = b.len();
    let mut i = 0usize;
    loop {
        if i < n && b[i] == b'"' {
            i += 1;
            let start = i;
            let mut escaped = false;
            let close = loop {
                let q = find1(b, i, b'"');
                if q == n {
                    return if eof { Err(unterminated_quote()) } else { Ok(Scan::Need) };
                }
                // A quote at the very edge of the window is ambiguous: `""` is
                // an escape and `"` alone closes the field.
                if q + 1 == n && !eof {
                    return Ok(Scan::Need);
                }
                if b.get(q + 1) == Some(&b'"') {
                    escaped = true;
                    i = q + 2;
                    continue;
                }
                break q;
            };
            if escaped {
                let s0 = scratch.len();
                let mut k = start;
                while k < close {
                    let m = find1(b, k, b'"').min(close);
                    scratch.extend_from_slice(&b[k..m]);
                    if m < close {
                        scratch.push(b'"');
                        k = m + 2;
                    } else {
                        k = close;
                    }
                }
                ranges.push((s0 as u32 | IN_SCRATCH, scratch.len() as u32));
            } else {
                ranges.push((start as u32, close as u32));
            }
            i = close + 1;
        } else {
            let start = i;
            let e = find2(b, i, delim, b'\n');
            if e == n && !eof {
                return Ok(Scan::Need);
            }
            // `find2` stops at the `\n` of a `\r\n`, so the carriage return is
            // trimmed here rather than searched for -- but *only* before a
            // newline. Trimming it before the delimiter too (which the first
            // version did) silently deleted a CR that was part of the data:
            // `ab\r,c` is a two-field row whose first field ends in a carriage
            // return, and CSV has no way to say that any other way.
            let end = e
                - usize::from(e < n && b[e] == b'\n' && e > start && b[e - 1] == b'\r');
            ranges.push((start as u32, end as u32));
            i = e;
        }
        match b.get(i) {
            Some(&c) if c == delim => i += 1,
            Some(b'\n') => return Ok(Scan::Done(i + 1)),
            // One line-ending model, LF or CRLF, applied identically to
            // quoted and unquoted fields. A lone `\r` as a *terminator* is
            // deliberately not supported: accepting it here and not in the
            // unquoted scan (where `find2` does not stop for it) would give
            // the same file two different row counts depending on whether its
            // fields happened to be quoted.
            Some(b'\r') => {
                return match b.get(i + 1) {
                    Some(b'\n') => Ok(Scan::Done(i + 2)),
                    Some(_) => Err(stray_after_quote(b'\r')),
                    None if eof => Ok(Scan::Done(i + 1)),
                    None => Ok(Scan::Need),
                }
            }
            None if eof => return Ok(Scan::Done(i)),
            None => return Ok(Scan::Need),
            Some(&c) => return Err(stray_after_quote(c)),
        }
    }
}

#[cold]
fn unterminated_quote() -> Error {
    Error::exec(
        "the file ends inside a quoted field: a `\"` was opened and never closed. \
         Nothing was imported past the last complete row",
    )
}

#[cold]
fn stray_after_quote(c: u8) -> Error {
    Error::exec(format!(
        "byte {c:#04x} follows a closing quote where a separator or a line break \
         was expected. A quote inside an unquoted field is taken literally; a \
         quoted field must be quoted end to end"
    ))
}

/// The record splitter over any `Read`, holding one window and nothing else.
pub struct Records<R> {
    src: R,
    buf: Vec<u8>,
    /// Valid bytes in `buf`.
    fill: usize,
    /// Start of the record being scanned.
    pos: usize,
    /// End of the record just returned; `pos` moves here on the next call.
    end: usize,
    eof: bool,
    scratch: Vec<u8>,
    ranges: Vec<(u32, u32)>,
    delim: u8,
    /// 1-based, for error messages. Counts records, not `\n`s, so an embedded
    /// newline does not skew it.
    pub line: u64,
    pub bytes: u64,
}

impl<R: Read> Records<R> {
    pub fn new(src: R, delim: u8) -> Records<R> {
        Records {
            src,
            buf: vec![0; WINDOW],
            fill: 0,
            pos: 0,
            end: 0,
            eof: false,
            scratch: Vec::new(),
            ranges: Vec::new(),
            delim,
            line: 0,
            bytes: 0,
        }
    }

    /// Advance to the next record. `false` at end of input.
    ///
    /// Not an `Iterator`: every item borrows the reader's window, so an
    /// `Item = &[u8]` would be a lending iterator, and copying the record out
    /// to satisfy the trait is the allocation this whole module exists to
    /// avoid. The name still reads right at the call site.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<bool> {
        self.pos = self.end;
        loop {
            if self.pos == self.fill && self.eof {
                return Ok(false);
            }
            // Disjoint field borrows: the window is read while the range and
            // scratch buffers are written, which is what keeps both reusable
            // across every record instead of being rebuilt per row.
            let scan = scan_record(
                &self.buf[self.pos..self.fill],
                self.eof,
                self.delim,
                &mut self.ranges,
                &mut self.scratch,
            )?;
            match scan {
                Scan::Done(k) => {
                    self.end = self.pos + k;
                    self.bytes += k as u64;
                    self.line += 1;
                    return Ok(true);
                }
                Scan::Need => self.refill()?,
            }
        }
    }

    fn refill(&mut self) -> Result<()> {
        if self.pos > 0 {
            self.buf.copy_within(self.pos..self.fill, 0);
            self.fill -= self.pos;
            self.pos = 0;
            self.end = 0;
        }
        if self.fill == self.buf.len() {
            // One record wider than the window. Doubling keeps the total copy
            // linear in the record's length.
            let want = self.buf.len() * 2;
            if want > MAX_RECORD {
                return Err(Error::exec(format!(
                    "a single record exceeds {} MiB. Either the file is not delimited \
                     the way this statement says it is, or a quote is unbalanced",
                    MAX_RECORD >> 20
                )));
            }
            self.buf.resize(want, 0);
        }
        let n = self.src.read(&mut self.buf[self.fill..]).map_err(io_err)?;
        self.fill += n;
        self.eof |= n == 0;
        Ok(())
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.ranges.len()
    }

    /// Field `i` of the current record, as the bytes it occupies.
    #[inline]
    pub fn field(&self, i: usize) -> &[u8] {
        let (s, e) = self.ranges[i];
        if s & IN_SCRATCH != 0 {
            &self.scratch[(s & !IN_SCRATCH) as usize..e as usize]
        } else {
            &self.buf[self.pos + s as usize..self.pos + e as usize]
        }
    }

    /// Was field `i` written with quotes? A quoted field is never NULL.
    #[inline]
    pub fn quoted(&self, i: usize) -> bool {
        let s = self.ranges[i].0;
        if s & IN_SCRATCH != 0 {
            return true;
        }
        // The opening quote sits one byte before the field's text, so this is
        // exact rather than a heuristic: a quoted field's `start` is always at
        // least 1 (it is the index *after* the quote), the first field of a
        // record starts at 0, and every other field is preceded by the
        // delimiter — which `settings::parse_char` refuses to let be `"`.
        s > 0 && self.buf[self.pos + s as usize - 1] == b'"'
    }

    /// The window's footprint, which is the importer's read-side resident set.
    pub fn capacity(&self) -> usize {
        self.buf.len() + self.scratch.capacity() + self.ranges.capacity() * 8
    }
}

fn io_err(e: std::io::Error) -> Error {
    Error::Io(e.to_string())
}

// ------------------------------------------------------------ typed decoding

/// How one target column reads a field. Decided once per import, so the type
/// dispatch is a jump on a value whose sequence repeats identically every row
/// — perfectly predicted — rather than a `DataType` walk per cell.
#[derive(Clone, Copy, Debug)]
enum Conv {
    /// Unsigned, with the declared width's ceiling.
    U(u64),
    /// Signed, with the declared width's range.
    I(i64, i64),
    F,
    S,
    Bool,
    Date,
    DateTime,
    Dec(u8),
}

fn conv_of(ty: &DataType) -> Conv {
    match ty.base() {
        DataType::UInt8 => Conv::U(u8::MAX as u64),
        DataType::UInt16 => Conv::U(u16::MAX as u64),
        DataType::UInt32 => Conv::U(u32::MAX as u64),
        DataType::UInt64 => Conv::U(u64::MAX),
        DataType::Int8 => Conv::I(i8::MIN as i64, i8::MAX as i64),
        DataType::Int16 => Conv::I(i16::MIN as i64, i16::MAX as i64),
        DataType::Int32 => Conv::I(i32::MIN as i64, i32::MAX as i64),
        DataType::Int64 => Conv::I(i64::MIN, i64::MAX),
        DataType::Float32 | DataType::Float64 => Conv::F,
        DataType::Bool => Conv::Bool,
        DataType::Date => Conv::Date,
        DataType::DateTime => Conv::DateTime,
        DataType::Decimal64(s) => Conv::Dec(*s),
        _ => Conv::S,
    }
}

/// `atoi` over bytes, refusing anything that is not all digits.
///
/// Returns `None` on overflow as well as on a bad byte: a `u64` column fed
/// `99999999999999999999` must fail the row, not wrap into a plausible number.
#[inline]
fn atou(b: &[u8]) -> Option<u64> {
    if b.is_empty() {
        return None;
    }
    let mut n = 0u64;
    for &c in b {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        n = n.checked_mul(10)?.checked_add(d as u64)?;
    }
    Some(n)
}

#[inline]
fn atoi(b: &[u8]) -> Option<i64> {
    match b.first()? {
        b'-' => atou(&b[1..]).and_then(|n| (n <= 1 << 63).then(|| (n as i64).wrapping_neg())),
        b'+' => atou(&b[1..]).and_then(|n| i64::try_from(n).ok()),
        _ => atou(b).and_then(|n| i64::try_from(n).ok()),
    }
}

/// `10^18 - 1`, the widest a `Decimal64` unit count goes
/// ([`crate::types::MAX_DECIMAL_PRECISION`]). Matched here so an out-of-range
/// literal is refused on the same digit as it is in `VALUES`.
const DEC_MAX_UNITS: i64 = 999_999_999_999_999_999;

/// `-12.345` at scale 2 is `-1234` units, straight off the bytes.
///
/// Never goes near a float: the whole point of `Decimal64` is that `'0.1'`
/// does not become 10.000000000000002 units, and this project has a
/// differential oracle that catches exactly that mistake.
///
/// Handles the exact case only: no exponent, and no more fraction digits than
/// the column's scale. Anything else hands off to [`dec_units`]'s fallback,
/// because excess digits **round** here (`0.5` at scale 0 is 1, not 0) and
/// re-deriving that rule instead of calling it would be a second decimal
/// dialect — the one thing a fixed-point type cannot afford. The exact case is
/// what this engine's own exporter writes, so the round trip never leaves it.
fn atodec(b: &[u8], scale: u8) -> Option<i64> {
    let (neg, b) = match b.first()? {
        b'-' => (true, &b[1..]),
        b'+' => (false, &b[1..]),
        _ => (false, b),
    };
    let dot = b.iter().position(|&c| c == b'.').unwrap_or(b.len());
    let (int, frac) = (&b[..dot], b.get(dot + 1..).unwrap_or(&[]));
    if int.is_empty() && frac.is_empty() {
        return None;
    }
    let s = scale as usize;
    // Excess digits round rather than truncate, and that rule lives in exactly
    // one place; see the note above.
    if frac.len() > s || frac.iter().any(|&c| !c.is_ascii_digit()) {
        return None;
    }
    let mut units: i64 = if int.is_empty() { 0 } else { atou(int)?.try_into().ok()? };
    for k in 0..s {
        let d = frac.get(k).map_or(0, |&c| (c - b'0') as i64);
        units = units.checked_mul(10)?.checked_add(d)?;
    }
    (units <= DEC_MAX_UNITS).then_some(if neg { -units } else { units })
}

/// [`atodec`], falling back to the engine's own string parser for the shapes
/// the fast path does not cover — an exponent (`1.5e3`, which is what a
/// `toString` of a float produces), a fraction wider than the column's scale
/// (which rounds), and anything needing `i128` intermediate accumulation. One
/// `Arc<str>` on a path that is already off the common case, against a second
/// decimal dialect, which is the thing actually worth avoiding.
#[inline]
fn dec_units(text: &[u8], scale: u8) -> Option<i64> {
    if let Some(u) = atodec(text, scale) {
        return Some(u);
    }
    let s = std::str::from_utf8(text).ok()?;
    crate::types::Value::str(s).to_decimal_units(scale).ok()
}

/// One target column mid-block: a typed buffer, a null mask and how to fill
/// them. `data` is handed to the `Block` on emit and a fresh one is reserved,
/// which is one allocation per column per *block* — the buffer leaves with the
/// block, so there is nothing to recycle.
struct Sink {
    ty: DataType,
    conv: Conv,
    nullable: bool,
    data: ColumnData,
    nulls: BitSet,
}

impl Sink {
    fn new(ty: &DataType, cap: usize) -> Sink {
        Sink {
            ty: ty.clone(),
            conv: conv_of(ty),
            nullable: ty.is_nullable(),
            data: with_capacity(ty.physical(), cap),
            nulls: BitSet::new(),
        }
    }

    #[inline]
    fn push_null(&mut self, row: usize) -> Result<()> {
        if !self.nullable {
            return Err(Error::exec(format!("NULL in a non-nullable {} column", self.ty)));
        }
        self.nulls.set(row);
        match &mut self.data {
            ColumnData::U64(v) => v.push(0),
            ColumnData::I64(v) => v.push(0),
            ColumnData::F64(v) => v.push(0.0),
            ColumnData::Str(v) => v.push("".into()),
        }
        Ok(())
    }

    /// Decode one field into the buffer. `text` borrows the read window, so
    /// nothing but a `String` column allocates.
    #[inline]
    fn push(&mut self, text: &[u8]) -> Result<()> {
        match (&mut self.data, self.conv) {
            (ColumnData::U64(v), Conv::U(max)) => {
                v.push(atou(text).filter(|&n| n <= max).ok_or_else(|| bad(text, &self.ty))?)
            }
            (ColumnData::I64(v), Conv::I(lo, hi)) => {
                v.push(atoi(text).filter(|&n| n >= lo && n <= hi).ok_or_else(|| bad(text, &self.ty))?)
            }
            (ColumnData::F64(v), Conv::F) => v.push(
                std::str::from_utf8(text)
                    .ok()
                    .and_then(|s| s.trim().parse::<f64>().ok())
                    .ok_or_else(|| bad(text, &self.ty))?,
            ),
            (ColumnData::U64(v), Conv::Bool) => v.push(parse_bool(text).ok_or_else(|| bad(text, &self.ty))? as u64),
            (ColumnData::U64(v), Conv::Date) => v.push(
                std::str::from_utf8(text)
                    .ok()
                    .and_then(|s| crate::types::parse_date(s).ok())
                    .or_else(|| atou(text).and_then(|n| u32::try_from(n).ok()))
                    .ok_or_else(|| bad(text, &self.ty))? as u64,
            ),
            // `DateTime` rides the `U64` lane (`DataType::physical`), and is
            // compared and sorted as one -- so a pre-1970 instant stored as
            // two's complement would sort *above* 2262 and break `ORDER BY`.
            // `parse_datetime` is signed and reaches back before the epoch, so
            // the negative is refused here, at the lane boundary, exactly where
            // `Value::as_u64` refuses it for `INSERT ... VALUES`. Bulk load
            // agreeing with row insert is the whole point.
            (ColumnData::U64(v), Conv::DateTime) => v.push(
                std::str::from_utf8(text)
                    .ok()
                    .and_then(|s| crate::types::parse_datetime(s).ok())
                    .or_else(|| atoi(text))
                    .filter(|&n| n >= 0)
                    .ok_or_else(|| bad(text, &self.ty))? as u64,
            ),
            (ColumnData::I64(v), Conv::Dec(s)) => {
                v.push(dec_units(text, s).ok_or_else(|| bad(text, &self.ty))?)
            }
            (ColumnData::Str(v), _) => v.push(
                std::str::from_utf8(text)
                    .map_err(|_| {
                        Error::exec("a String field is not valid UTF-8; this engine stores text, not bytes")
                    })?
                    .into(),
            ),
            // Every combination `conv_of` can produce is above. Reaching here
            // means the two disagree, which is a bug and not a bad row.
            (d, c) => {
                return Err(Error::exec(format!(
                    "internal: {:?} buffer cannot take a {c:?} field",
                    d.physical()
                )))
            }
        }
        Ok(())
    }

    /// Undo a partially-written row. Only the error path calls this, so the
    /// cost lives entirely with the malformed input that caused it.
    fn rewind(&mut self, row: usize) {
        while self.data.len() > row {
            self.nulls.clear(self.data.len() - 1);
            self.data.truncate(self.data.len() - 1);
        }
    }

    fn take(&mut self, cap: usize) -> Column {
        let data = std::mem::replace(&mut self.data, with_capacity(self.ty.physical(), cap));
        let nulls = std::mem::take(&mut self.nulls);
        Column::with_nulls(self.ty.clone(), data, nulls)
    }
}

fn with_capacity(p: PhysicalType, cap: usize) -> ColumnData {
    match p {
        PhysicalType::U64 => ColumnData::U64(Vec::with_capacity(cap)),
        PhysicalType::I64 => ColumnData::I64(Vec::with_capacity(cap)),
        PhysicalType::F64 => ColumnData::F64(Vec::with_capacity(cap)),
        PhysicalType::Str => ColumnData::Str(Vec::with_capacity(cap)),
    }
}

#[cold]
fn bad(text: &[u8], ty: &DataType) -> Error {
    let s = String::from_utf8_lossy(&text[..text.len().min(64)]);
    Error::exec(format!("`{s}` is not a {ty}"))
}

#[inline]
fn parse_bool(b: &[u8]) -> Option<bool> {
    match b {
        b"1" => Some(true),
        b"0" => Some(false),
        _ if b.eq_ignore_ascii_case(b"true") || b.eq_ignore_ascii_case(b"t") => Some(true),
        _ if b.eq_ignore_ascii_case(b"false") || b.eq_ignore_ascii_case(b"f") => Some(false),
        _ => None,
    }
}

// ----------------------------------------------------------------- importing

/// What an import did. `peak_bytes` is the point of the whole module: it is
/// the importer's own high-water mark, and it does not move with the size of
/// the file.
#[derive(Clone, Copy, Debug, Default)]
pub struct ImportStats {
    pub rows: usize,
    pub bytes: u64,
    pub skipped: usize,
    pub peak_bytes: usize,
}

/// What to do with a row that will not decode.
#[derive(Clone, Copy, Debug)]
pub struct ErrorPolicy {
    /// Rows that may be skipped before the statement fails. 0 — the default —
    /// means the first bad row fails it, which is the only policy that cannot
    /// lose data by accident.
    pub allow: u32,
}

/// Stream `src` into `table`, block at a time.
///
/// `columns` empty means "every column, in the file's order" — either the
/// table's declaration order, or whatever the header line says when
/// `dialect.header` is set. Columns the file does not carry are filled with
/// their `DEFAULT`, so a CSV missing a trailing column still loads.
pub fn import<R: Read>(
    sess: &mut Session,
    table: &str,
    columns: &[String],
    src: R,
    dialect: &Dialect,
    cfg: &Settings,
    policy: ErrorPolicy,
    ctx: &QueryContext,
) -> Result<ImportStats> {
    // An import inside a transaction would have to be either rolled back
    // (nothing stages a part) or made durable by a checkpoint (which would
    // persist the transaction's other uncommitted parts). Refused rather than
    // half-done; see the report on routing this through `Session::run_insert`,
    // which would make it transactional for free.
    if sess.in_transaction() {
        return Err(Error::unsupported(
            "INSERT ... FROM INFILE cannot run inside a transaction: it publishes parts \
             as it streams, so there would be nothing for ROLLBACK to undo. COMMIT or \
             ROLLBACK first",
        ));
    }
    // An import is a write, and it does not pass through `exec_statement`, so
    // the read-only gate that lives there has to be repeated. Without it a
    // session holding only a *shared* directory lock would publish parts into a
    // directory another process is reading -- the exact race the lock exists to
    // prevent.
    if sess.is_read_only() {
        return Err(Error::unsupported(
            "INSERT ... FROM INFILE is a write and this session was opened read-only: \
             it holds a shared directory lock that several processes take at once \
             precisely because none of them writes",
        ));
    }
    let schema = sess.catalog.table_by_path(table)?.schema().clone();
    let mut rec = Records::new(src, dialect.delim);

    // Column mapping: file position -> schema index. Resolved once.
    let mut map: Vec<usize> = Vec::new();
    if !rec.next()? {
        return Ok(ImportStats::default()); // an empty file is not an error
    }
    if dialect.header {
        for i in 0..rec.width() {
            let name = std::str::from_utf8(rec.field(i))
                .map_err(|_| Error::exec("a header field is not valid UTF-8"))?;
            map.push(schema.require(name.trim())?);
        }
        if !rec.next()? {
            return Ok(ImportStats { bytes: rec.bytes, peak_bytes: rec.capacity(), ..Default::default() });
        }
    } else if !columns.is_empty() {
        for c in columns {
            map.push(schema.require(c)?);
        }
    } else {
        // Every column, not "as many as the file happens to carry": clamping
        // here made the width check below trivially true, so a two-field file
        // loaded into a three-column table and the statement was reported as a
        // success. (It filled the missing column with its DEFAULT, not with
        // zero as an earlier note here claimed -- so the objection is not that
        // the rows were wrong, it is that nobody asked for a partial load and
        // nobody was told they got one.) Naming the columns is how a partial
        // load is asked for, and that form still fills the rest with DEFAULT.
        map = (0..schema.len()).collect();
    }
    if map.len() != rec.width() {
        return Err(Error::exec(format!(
            "the file has {} fields per row and the statement names {} columns. \
             Name the columns the file does carry -- `INSERT INTO <table> ({}) FROM INFILE ..` \
             -- to load it as a partial row and fill the rest with their DEFAULT",
            rec.width(),
            map.len(),
            (0..rec.width().min(schema.len()))
                .map(|c| schema.name(map[c]))
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    // Duplicates would make the later column silently win; say so instead.
    let mut seen = vec![false; schema.len()];
    for &c in &map {
        if std::mem::replace(&mut seen[c], true) {
            return Err(Error::exec(format!("column `{}` is named twice", schema.name(c))));
        }
    }

    // Blocks are bounded by rows *and* by bytes, so a small budget produces
    // small blocks rather than an out-of-memory error: that is what makes
    // "import a file bigger than the budget" work rather than merely fail
    // politely.
    // The session's tracker, not a second one built here: an import and the
    // statements around it share one budget rather than each getting the whole
    // of it. `export`, one screen down, has taken a `QueryContext` all along.
    let mem = &ctx.mem;
    let block_rows = cfg.max_insert_block_size as usize;
    let block_bytes = (cfg.max_memory_usage / 4).clamp(1 << 16, 1 << 28) as usize;
    mem.reserve(rec.capacity(), "the CSV read window")?;

    let cap = block_rows.min(1 << 16);
    let mut sinks: Vec<Sink> = map.iter().map(|&c| Sink::new(schema.ty(c), cap)).collect();
    let mut st = ImportStats { peak_bytes: rec.capacity(), ..Default::default() };
    let (mut rows, mut used, mut bad_rows) = (0usize, 0usize, 0u32);

    loop {
        if rec.width() != map.len() {
            let e = wrong_width(rec.width(), map.len());
            bad_rows = fail_or_skip(bad_rows, policy, rec.line, st.rows, e)?;
            st.skipped += 1;
        } else {
            match decode_row(&mut sinks, &rec, &dialect.null, rows) {
                Ok(n) => {
                    rows += 1;
                    used += n;
                }
                Err(e) => {
                    for s in sinks.iter_mut() {
                        s.rewind(rows);
                    }
                    bad_rows = fail_or_skip(bad_rows, policy, rec.line, st.rows, e)?;
                    st.skipped += 1;
                }
            }
        }
        if rows >= block_rows || used >= block_bytes {
            // The deadline/cancel checkpoint, inside the branch that was
            // already here: it runs once per emitted block (65,536 rows by
            // default) and never per row. One relaxed load, plus one clock
            // read only when a deadline is actually set, against a full column
            // build and an ingest.
            ctx.check()?;
            // The high-water mark is here by construction: `used` only grows
            // between emits, so the row before an emit is the widest this
            // import ever holds.
            st.peak_bytes = st.peak_bytes.max(rec.capacity() + used);
            mem.reserve(used, "the import's row block")?;
            emit(sess, table, &schema, &map, &mut sinks, rows, cap)
                .map_err(|e| block_rejected(st.rows, rec.line, e))?;
            mem.release(used);
            st.rows += rows;
            (rows, used) = (0, 0);
        }
        if !rec.next()? {
            break;
        }
    }
    if rows > 0 {
        st.peak_bytes = st.peak_bytes.max(rec.capacity() + used);
        emit(sess, table, &schema, &map, &mut sinks, rows, cap)
            .map_err(|e| block_rejected(st.rows, rec.line, e))?;
        st.rows += rows;
    }
    st.bytes = rec.bytes;
    // Acknowledged means durable. The WAL is `Session`'s to write and this
    // path does not reach it, so a checkpoint is what closes the gap — see
    // the report. Skipped entirely for an in-memory session, which has none.
    if sess.catalog.is_persistent() {
        sess.checkpoint()?;
    }
    Ok(st)
}

#[cold]
fn wrong_width(got: usize, want: usize) -> Error {
    Error::exec(format!("row has {got} fields, expected {want}"))
}

/// The malformed-row policy, in one place so it cannot drift between the two
/// ways a row goes bad.
///
/// `committed` is [`ImportStats::rows`], which counts only rows an [`emit`] has
/// already published -- so it is exactly what survives this failure. It has to
/// be said: an import publishes parts as it streams and there is no prefix to
/// roll back, and the message used to promise the opposite while a quarter of a
/// million rows sat durable on disk. An operator who fixed the bad row and
/// retried got them twice.
fn fail_or_skip(
    seen: u32,
    policy: ErrorPolicy,
    line: u64,
    committed: usize,
    e: Error,
) -> Result<u32> {
    if seen < policy.allow {
        return Ok(seen + 1);
    }
    Err(Error::exec(format!(
        "line {line}: {e}. {}{}",
        committed_note(committed),
        if policy.allow == 0 {
            "; raise `input_format_allow_errors_num` to skip bad rows instead"
        } else {
            ". That is more bad rows than `input_format_allow_errors_num` allows"
        }
    )))
}

/// The durable prefix, in words. One place, because an import now has two ways
/// to die past a block boundary and they owe the operator the same sentence.
#[cold]
fn committed_note(committed: usize) -> String {
    match committed {
        0 => "No rows before it were lost".into(),
        n => format!(
            "{n} rows before it are ALREADY COMMITTED and durable -- an import publishes \
             parts as it streams, so nothing rolls them back; remove them before retrying \
             or the retry will duplicate them"
        ),
    }
}

/// A block storage refused: a CHECK or UNIQUE violation.
///
/// Routing `emit` through `Session::import_block` is what made CHECK and
/// UNIQUE reachable from a bulk import, and it created this second exit with
/// it -- one that arrived with no line, no row count and no warning while a
/// quarter of a million rows sat durable on disk. It says so now, and it says
/// the other thing an operator reaches for first does not apply:
/// `input_format_allow_errors_num` counts *malformed* rows, and a row that
/// parses fine and breaks a table rule is not one of them.
#[cold]
fn block_rejected(committed: usize, line: u64, e: Error) -> Error {
    Error::exec(format!(
        "the block ending at line {line} was refused by the table: {e}. {}. \
         `input_format_allow_errors_num` does not skip this -- it skips rows that will \
         not parse, not rows a constraint rejects",
        committed_note(committed)
    ))
}

/// Decode one record into the sinks, returning the bytes it added.
///
/// The caller wants a running block size, and getting it by summing a
/// per-`Sink` counter would be a walk over every column *per row* -- an O(width)
/// loop bolted onto a loop that is already O(width). Returning the total makes
/// it one add.
#[inline]
fn decode_row<R: Read>(
    sinks: &mut [Sink],
    rec: &Records<R>,
    null: &str,
    row: usize,
) -> Result<usize> {
    let mut bytes = 0;
    for (i, s) in sinks.iter_mut().enumerate() {
        let text = rec.field(i);
        bytes += text.len() + 8;
        // A quoted field is never NULL: that is the only distinction CSV can
        // draw between NULL and the empty string, and it is what makes the
        // round trip in `tests/settings_and_io.rs` exact.
        if text == null.as_bytes() && !rec.quoted(i) {
            s.push_null(row)?;
        } else {
            s.push(text)?;
        }
    }
    Ok(bytes)
}

/// Hand one block to storage, filling any column the file did not carry.
fn emit(
    sess: &mut Session,
    table: &str,
    schema: &Schema,
    map: &[usize],
    sinks: &mut [Sink],
    rows: usize,
    cap: usize,
) -> Result<()> {
    // One `Vec` per block, which is the one the `Block` leaves with -- there is
    // no scratch beside it. `map` is one entry per file field, so the search is
    // over a handful of `usize` and beats a second allocation.
    let mut full = Vec::with_capacity(schema.len());
    for c in 0..schema.len() {
        full.push(match map.iter().position(|&m| m == c) {
            Some(i) => sinks[i].take(cap),
            // Unmentioned: the column's DEFAULT, or NULL/zero. Same rule the
            // partial-column `INSERT` path uses, so the two agree.
            None => Column::constant(schema.ty(c), &schema.field(c).fill_value(), rows)?,
        });
    }
    // Through `Session`, not straight at the catalog: this used to be the one
    // write path in the engine that bypassed CHECK and UNIQUE, and the price of
    // routing it is per *block*, not per row -- `enforce_checks` returns on an
    // empty map and `import_block` adds one `contains_key`.
    sess.import_block(table, Block::new(full)?)?;
    Ok(())
}

// ----------------------------------------------------------------- exporting

/// RFC 4180 quoting, one scan, and a field needing none leaves as a single
/// `write_all` of the bytes it arrived as.
///
/// `null` is passed so that a *string* whose text is the null representation
/// gets quoted: without that, exporting the literal text `\N` and importing it
/// back would turn it into NULL. That round trip is a test.
fn write_field<W: Write>(w: &mut W, s: &str, delim: u8, null: &str) -> std::io::Result<()> {
    let b = s.as_bytes();
    let plain = !b.is_empty()
        && s != null
        && !b.iter().any(|&c| c == delim || c == b'"' || c == b'\n' || c == b'\r');
    if plain {
        return w.write_all(b);
    }
    w.write_all(b"\"")?;
    let mut last = 0;
    for (i, &c) in b.iter().enumerate() {
        if c == b'"' {
            w.write_all(&b[last..=i])?; // through the quote, then again from it
            last = i;
        }
    }
    w.write_all(&b[last..])?;
    w.write_all(b"\"")
}

/// How a column's cells are rendered, chosen once per block by [`rend_of`].
#[derive(Clone, Copy)]
pub enum Rend {
    U,
    I,
    F,
    S,
    Bool,
    Date,
    DateTime,
    Dec(u8),
}

/// The renderer a declared type wants. One call per column per block.
pub fn rend_of(ty: &DataType) -> Rend {
    match ty.base() {
        DataType::Bool => Rend::Bool,
        DataType::Date => Rend::Date,
        DataType::DateTime => Rend::DateTime,
        DataType::Decimal64(s) => Rend::Dec(*s),
        t => match t.physical() {
            PhysicalType::U64 => Rend::U,
            PhysicalType::I64 => Rend::I,
            PhysicalType::F64 => Rend::F,
            PhysicalType::Str => Rend::S,
        },
    }
}

/// `YYYY-MM-DD` into the writer with no intermediate `String`.
fn write_date<W: Write>(w: &mut W, days: i64) -> std::io::Result<()> {
    let (y, m, d) = civil_from_days(days);
    let mut b = *b"0000-00-00";
    let mut yy = y.unsigned_abs();
    for i in (0..4).rev() {
        b[i] = b'0' + (yy % 10) as u8;
        yy /= 10;
    }
    b[5] = b'0' + (m / 10) as u8;
    b[6] = b'0' + (m % 10) as u8;
    b[8] = b'0' + (d / 10) as u8;
    b[9] = b'0' + (d % 10) as u8;
    w.write_all(&b)
}

fn write_dec<W: Write>(w: &mut W, units: i64, scale: u8) -> std::io::Result<()> {
    if scale == 0 {
        return write!(w, "{units}");
    }
    let p = 10i64.pow(scale as u32);
    let (sign, u) = if units < 0 { ("-", units.unsigned_abs()) } else { ("", units as u64) };
    write!(w, "{sign}{}.{:0>width$}", u / p as u64, u % p as u64, width = scale as usize)
}

/// Write one block as delimited text. Type dispatch is hoisted to `rend`, so
/// the per-cell cost is a jump on a value the branch predictor sees repeat in
/// the same order every row.
pub fn write_block<W: Write>(
    w: &mut W,
    b: &Block,
    rend: &[Rend],
    delim: u8,
    null: &str,
) -> std::io::Result<()> {
    for r in 0..b.rows() {
        for (c, rend) in rend.iter().enumerate().take(b.width()) {
            if c > 0 {
                w.write_all(&[delim])?;
            }
            let col = b.column(c);
            if col.is_null(r) {
                w.write_all(null.as_bytes())?;
                continue;
            }
            match (*rend, &col.data) {
                (Rend::U, ColumnData::U64(v)) => write!(w, "{}", v[r])?,
                (Rend::I, ColumnData::I64(v)) => write!(w, "{}", v[r])?,
                (Rend::F, ColumnData::F64(v)) => write!(w, "{}", v[r])?,
                (Rend::Bool, ColumnData::U64(v)) => {
                    w.write_all(if v[r] != 0 { b"true" } else { b"false" })?
                }
                (Rend::Date, ColumnData::U64(v)) => write_date(w, v[r] as i64)?,
                (Rend::DateTime, ColumnData::I64(v)) => {
                    write_date(w, v[r].div_euclid(86_400))?;
                    let s = v[r].rem_euclid(86_400);
                    write!(w, " {:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)?
                }
                (Rend::Dec(sc), ColumnData::I64(v)) => write_dec(w, v[r], sc)?,
                (Rend::S, ColumnData::Str(v)) => write_field(w, &v[r], delim, null)?,
                // A column whose physical kind does not match the type it
                // declares is a storage bug, not an output format; render it
                // rather than lose the row.
                _ => write_field(w, &col.value(r).render_plain(), delim, null)?,
            }
        }
        w.write_all(b"\n")?;
    }
    Ok(())
}

/// Stream a `SELECT` to `w` as delimited text, one block at a time.
///
/// Goes through `Session::read_stream`, which is the engine's own answer to
/// "never materialize a large result" — a 200M-row export holds one block.
pub fn export<W: Write>(
    sess: &Session,
    sql: &str,
    w: &mut W,
    dialect: &Dialect,
    ctx: &QueryContext,
) -> Result<usize> {
    let mut rend: Vec<Rend> = Vec::new();
    let mut err: Option<std::io::Error> = None;
    let n = sess.read_stream(sql, ctx, &mut |item| {
        let r: std::io::Result<()> = match item {
            StreamItem::Head(schema) => {
                rend.extend(schema.fields().iter().map(|f| rend_of(&f.ty)));
                if dialect.header {
                    write_header(w, schema, dialect)
                } else {
                    Ok(())
                }
            }
            StreamItem::Rows(b) => write_block(w, &b, &rend, dialect.delim, &dialect.null),
        };
        // The sink signature is `Result<()>`, so the `io::Error` is carried
        // out here rather than stringified into an engine error and losing
        // its kind (a closed pipe and a full disk are not the same failure).
        match r {
            Ok(()) => Ok(()),
            Err(e) => {
                let msg = e.to_string();
                err = Some(e);
                Err(Error::Io(msg))
            }
        }
    })?;
    if let Some(e) = err {
        return Err(io_err(e));
    }
    Ok(n)
}

fn write_header<W: Write>(w: &mut W, schema: &Schema, d: &Dialect) -> std::io::Result<()> {
    for (i, f) in schema.fields().iter().enumerate() {
        if i > 0 {
            w.write_all(&[d.delim])?;
        }
        write_field(w, &f.name, d.delim, &d.null)?;
    }
    w.write_all(b"\n")
}

// ------------------------------------------------------------ the statements

/// `INSERT INTO t [(cols)] FROM INFILE 'path' [FORMAT f] [SETTINGS ...]`.
///
/// `at` is the index of `INFILE`; everything before it is the target, which is
/// re-read here rather than handed over by `sql::parser` because the statement
/// is recognised outside the grammar (see [`crate::settings::Handle::intercept`]).
pub(crate) fn run_import(
    sess: &mut Session,
    cfg: &Settings,
    _text: &str,
    t: &[Spanned],
    at: usize,
) -> Result<ResultSet> {
    let mut i = 2; // past INSERT INTO
    if !t.get(1).is_some_and(|s| s.tok.is_keyword("INTO")) {
        return Err(Error::parse("expected `INSERT INTO`", at_pos(t, 1)));
    }
    let (name, mut i2) = object_name(t, i)?;
    i = i2;
    let mut columns: Vec<String> = Vec::new();
    if t.get(i).map(|s| &s.tok) == Some(&Token::LParen) {
        i += 1;
        loop {
            match t.get(i).and_then(|s| s.tok.word()) {
                Some(w) => columns.push(w.to_string()),
                None => return Err(Error::parse("expected a column name", at_pos(t, i))),
            }
            i += 1;
            match t.get(i).map(|s| &s.tok) {
                Some(Token::Comma) => i += 1,
                Some(Token::RParen) => {
                    i += 1;
                    break;
                }
                _ => return Err(Error::parse("expected `,` or `)`", at_pos(t, i))),
            }
        }
    }
    if !t.get(i).is_some_and(|s| s.tok.is_keyword("FROM")) || i + 1 != at {
        return Err(Error::parse("expected `FROM INFILE 'path'`", at_pos(t, i)));
    }
    let path = literal(t, at + 1, "INFILE")?;
    i2 = at + 2;
    let (dialect, cfg2) = tail(t, &mut i2, &path, cfg)?;
    if i2 != t.len() {
        return Err(Error::parse("trailing input after the import statement", at_pos(t, i2)));
    }
    // An explicit column list is the naming, so it wins over the header
    // setting: a file with a header *and* a column list would otherwise eat
    // its first data row or, worse, import the header as one.
    let d = Dialect {
        header: if columns.is_empty() { dialect.header } else { false },
        ..dialect
    };

    let table = sess.catalog.qualify(&name);
    let f = File::open(&path).map_err(|e| Error::Io(format!("{path}: {e}")))?;
    let policy = ErrorPolicy { allow: cfg2.input_format_allow_errors_num };
    let ctx = cfg2.context(sess);
    let st = import(sess, &table, &columns, f, &d, &cfg2, policy, &ctx)?;
    let mut rs = ResultSet::with_affected(st.rows);
    rs.stats.rows = st.rows;
    rs.stats.rows_scanned = st.rows as u64 + st.skipped as u64;
    Ok(rs)
}

/// `<select> INTO OUTFILE 'path' [FORMAT f] [SETTINGS ...]`.
///
/// `at` is the index of `INTO`, so the query is exactly the source text before
/// it — no AST is re-serialized, which is what keeps every SELECT the engine
/// can parse exportable rather than the subset a printer happens to cover.
pub(crate) fn run_export(
    sess: &mut Session,
    cfg: &Settings,
    text: &str,
    t: &[Spanned],
    at: usize,
) -> Result<ResultSet> {
    let head = &text[..t[at].pos - t[0].pos];
    // `read_stream` is a `&self` path and refuses a table with a delta a scan
    // cannot see. `exec_statement` flushes before every read for the same
    // reason; an export that skipped it would silently omit the newest rows,
    // which is the exact failure mode this engine keeps finding. A read-only
    // session has nothing buffered -- every mutation was refused -- so it is
    // skipped there rather than asked to write parts under a shared lock.
    if !sess.is_read_only() {
        sess.catalog.flush_all()?;
    }
    let path = literal(t, at + 2, "OUTFILE")?;
    let mut i = at + 3;
    let (dialect, cfg2) = tail(t, &mut i, &path, cfg)?;
    if i != t.len() {
        return Err(Error::parse("trailing input after the export statement", at_pos(t, i)));
    }
    // Written through a temporary and renamed: a failed or cancelled export
    // must not leave a half-written file where the next command will read it
    // as data. Same rule the part writer follows.
    let tmp = format!("{path}.part");
    let file = File::create(&tmp).map_err(|e| Error::Io(format!("{tmp}: {e}")))?;
    let mut w = BufWriter::with_capacity(WINDOW, file);
    let n = export(sess, head, &mut w, &dialect, &cfg2.context(sess)).and_then(|n| {
        // `BufWriter`'s `Drop` swallows its flush error, so a full disk on the
        // last 256 KiB has to be collected here or the export reports success.
        w.flush().map_err(io_err)?;
        Ok(n)
    });
    match n {
        Ok(n) => {
            std::fs::rename(&tmp, Path::new(&path))
                .map_err(|e| Error::Io(format!("{path}: {e}")))?;
            let mut rs = ResultSet::with_affected(n);
            rs.stats.rows = n;
            Ok(rs)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// The shared `[FORMAT f] [SETTINGS k = v, ...]` tail.
fn tail(
    t: &[Spanned],
    i: &mut usize,
    path: &str,
    cfg: &Settings,
) -> Result<(Dialect, Settings)> {
    let mut cfg2 = cfg.clone();
    let mut named: Option<String> = None;
    if t.get(*i).is_some_and(|s| s.tok.is_keyword("FORMAT")) {
        named = Some(match t.get(*i + 1).map(|s| &s.tok) {
            Some(Token::Word { value, .. }) => value.clone(),
            Some(Token::Str(s)) => s.clone(),
            _ => return Err(Error::parse("expected a format name after FORMAT", at_pos(t, *i + 1))),
        });
        *i += 2;
    }
    if t.get(*i).is_some_and(|s| s.tok.is_keyword("SETTINGS")) {
        for (name, value) in crate::settings::pairs(t, *i + 1)? {
            cfg2.set(&name, &value)?;
        }
        *i = t.len();
    }
    let d = match &named {
        Some(n) => Dialect::named(n, &cfg2)?,
        None => Dialect::for_path(path, &cfg2),
    };
    Ok((d, cfg2))
}

fn object_name(t: &[Spanned], mut i: usize) -> Result<(crate::sql::ast::ObjectName, usize)> {
    let mut parts = Vec::new();
    loop {
        match t.get(i).and_then(|s| s.tok.word()) {
            Some(w) => parts.push(w.to_string()),
            None => return Err(Error::parse("expected a table name", at_pos(t, i))),
        }
        i += 1;
        if t.get(i).map(|s| &s.tok) == Some(&Token::Dot) {
            i += 1;
        } else {
            return Ok((crate::sql::ast::ObjectName(parts), i));
        }
    }
}

fn literal(t: &[Spanned], i: usize, what: &str) -> Result<String> {
    match t.get(i).map(|s| &s.tok) {
        Some(Token::Str(s)) => Ok(s.clone()),
        _ => Err(Error::parse(format!("{what} wants a quoted path"), at_pos(t, i))),
    }
}

fn at_pos(t: &[Spanned], i: usize) -> usize {
    t.get(i).map_or(t.last().map_or(0, |s| s.pos), |s| s.pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Value;

    fn fields(src: &str, delim: u8) -> Vec<Vec<String>> {
        let mut r = Records::new(src.as_bytes(), delim);
        let mut out = Vec::new();
        while r.next().unwrap() {
            out.push(
                (0..r.width())
                    .map(|i| String::from_utf8(r.field(i).to_vec()).unwrap())
                    .collect(),
            );
        }
        out
    }

    #[test]
    fn quoting_embedded_separators_and_newlines() {
        let got = fields("a,\"b,c\",\"d\ne\",\"f\"\"g\"\n1,2,3,4\n", b',');
        assert_eq!(got[0], ["a", "b,c", "d\ne", "f\"g"]);
        assert_eq!(got[1], ["1", "2", "3", "4"]);
    }

    #[test]
    fn line_endings_and_a_missing_final_newline() {
        assert_eq!(fields("a,b\r\nc,d\r\n", b','), [["a", "b"], ["c", "d"]]);
        assert_eq!(fields("a,b\nc,d", b','), [["a", "b"], ["c", "d"]]);
        assert_eq!(fields("a\tb\nc\td\n", b'\t'), [["a", "b"], ["c", "d"]]);
        // An empty trailing field is a field, not an absence.
        assert_eq!(fields("a,\n", b','), [["a", ""]]);
        // A carriage return that is data, not a line ending. The first version
        // of the CRLF trim deleted this one.
        assert_eq!(fields("ab\r,c\n", b','), [["ab\r", "c"]]);
        // ... and it is still trimmed where it *is* a line ending, including
        // on a quoted field and at end of input.
        assert_eq!(fields("\"q\"\r\nz\r\n", b','), [["q"], ["z"]]);
    }

    /// The window boundary is where a hand-rolled scanner goes wrong, so it is
    /// crossed deliberately: a record longer than 256 KiB, and a quote landing
    /// exactly on the edge.
    #[test]
    fn records_crossing_the_window_are_intact() {
        let big = "x".repeat(WINDOW + 1000);
        let src = format!("a,b\n{big},{big}\nc,d\n");
        let got = fields(&src, b',');
        assert_eq!(got.len(), 3);
        assert_eq!(got[1][0].len(), big.len());
        assert_eq!(got[1][1].len(), big.len());
        assert_eq!(got[2], ["c", "d"]);

        // A quoted field with an escape straddling the refill.
        let pad = "y".repeat(WINDOW - 4);
        let src = format!("\"{pad}\"\"z\",tail\n");
        let got = fields(&src, b',');
        assert_eq!(got[0][0], format!("{pad}\"z"));
        assert_eq!(got[0][1], "tail");
    }

    #[test]
    fn unterminated_quote_is_an_error_not_a_silent_truncation() {
        let mut r = Records::new(&b"a,\"b\n"[..], b',');
        assert!(r.next().is_err());
    }

    #[test]
    fn quoted_is_only_true_for_quoted_fields() {
        let mut r = Records::new(&b"\\N,\"\\N\",x\n"[..], b',');
        assert!(r.next().unwrap());
        assert!(!r.quoted(0), "an unquoted \\N is NULL");
        assert!(r.quoted(1), "a quoted \\N is the two-character string");
        assert!(!r.quoted(2));
    }

    #[test]
    fn integers_reject_what_they_cannot_hold() {
        assert_eq!(atou(b"18446744073709551615"), Some(u64::MAX));
        assert_eq!(atou(b"18446744073709551616"), None, "overflow must not wrap");
        assert_eq!(atou(b"12a"), None);
        assert_eq!(atou(b""), None);
        assert_eq!(atoi(b"-9223372036854775808"), Some(i64::MIN));
        assert_eq!(atoi(b"9223372036854775808"), None);
        assert_eq!(atoi(b"-0"), Some(0));
    }

    /// The decimal path must agree with the engine's own string parser cell for
    /// cell, or the same literal would be legal in `VALUES` and illegal (or,
    /// worse, differently valued) in a CSV. The fallback in `dec_units` is what
    /// makes the agreement total rather than approximate, so it is `dec_units`
    /// -- not the fast path alone -- that is compared.
    #[test]
    fn decimals_agree_with_the_engine() {
        for scale in [0u8, 2, 6, 9] {
            for s in [
                "0", "1", "-1", "12.34", "-12.34", "0.5", "1000000", "-0.01", "3.140000",
                "+7", ".5", "-.5", "1.005", "1.999", "0.0000001", "1.5e3", "2E-2",
                "123456789012345678", "1234567890123456789012", "-0",
            ] {
                assert_eq!(
                    dec_units(s.as_bytes(), scale),
                    Value::str(s).to_decimal_units(scale).ok(),
                    "{s} at scale {scale}"
                );
            }
        }
        // Non-digits are refused by both, and by the fast path without the
        // allocation the fallback would cost.
        for s in ["1.2.3", "", "abc", "1.2x", "--1"] {
            assert_eq!(dec_units(s.as_bytes(), 2), None, "{s}");
        }
    }

    #[test]
    fn csv_quotes_exactly_what_it_must() {
        let mut out = Vec::new();
        for (input, want) in [
            ("plain", "plain"),
            ("", "\"\""),
            ("a,b", "\"a,b\""),
            ("a\"b", "\"a\"\"b\""),
            ("a\nb", "\"a\nb\""),
            ("\\N", "\"\\N\""), // or it would read back as NULL
        ] {
            out.clear();
            write_field(&mut out, input, b',', "\\N").unwrap();
            assert_eq!(String::from_utf8(out.clone()).unwrap(), want, "{input:?}");
        }
    }

    #[test]
    fn dates_and_decimals_render_the_way_they_parse() {
        let mut out = Vec::new();
        write_date(&mut out, crate::types::days_from_civil(2024, 2, 29)).unwrap();
        assert_eq!(out, b"2024-02-29");
        for (units, scale, want) in
            [(1234i64, 2u8, "12.34"), (-1234, 2, "-12.34"), (5, 3, "0.005"), (42, 0, "42")]
        {
            out.clear();
            write_dec(&mut out, units, scale).unwrap();
            assert_eq!(String::from_utf8(out.clone()).unwrap(), want);
        }
    }
}
