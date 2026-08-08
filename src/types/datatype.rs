//! SQL types and the physical representation they collapse to.
//!
//! The engine deliberately has only **four physical kinds** (`U64`, `I64`,
//! `F64`, `Str`). Every SQL type maps onto one of them, which keeps the
//! vectorized expression evaluator to four match arms per operator instead of
//! sixteen, and lets one `PackedU64` codec serve every numeric, temporal and
//! (dictionary-encoded) string column.
//!
//! ## The rule this module exists to enforce
//!
//! The README promises the engine never silently does something other than
//! what was asked, and **a type parameter accepted and then discarded is
//! exactly that promise broken**. It is the shape of every bug this file has
//! ever had: `DEFAULT` stored as text nothing evaluated, `Decimal(38,2)`
//! narrowed to 18 digits, `DateTime64(3)` truncated to whole seconds,
//! `DateTime('America/New_York')` handed back in UTC. Each looked like a
//! working feature, echoed back a lie from `SHOW CREATE TABLE`, and was wrong
//! only in the data.
//!
//! So the standing rule for anything parsed here: **honour it, normalize it
//! visibly, or refuse it by name.** Normalizing is allowed only when nothing
//! is lost and [`fmt::Display`] prints the form actually in force -- that is
//! why `Decimal(10,2)` becomes `Decimal64(2)` (the precision was only a cap,
//! and the cap held) and why `LowCardinality(String)` survives (its storage
//! *is* `String`'s, per-granule dictionaries either way). Where the parameter
//! would change a stored value, the arm returns an `Unsupported` error naming
//! the limitation, because a loud refusal at DDL costs one edit and a silent
//! truncation costs the data.

use super::value::Value;
use crate::common::{Error, Result};
use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PhysicalType {
    /// Unsigned integers, booleans, dates, datetimes, dictionary codes.
    U64,
    /// Signed integers; zigzag-encoded before packing.
    I64,
    /// Floats; bit-cast to u64 before packing.
    F64,
    /// Strings; dictionary-encoded per granule, so physically also u64 codes,
    /// but logically distinct because comparisons need the dictionary.
    Str,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum DataType {
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
    Bool,
    String,
    FixedString(u32),
    /// Days since 1970-01-01.
    Date,
    /// Seconds since 1970-01-01T00:00:00Z.
    DateTime,
    /// Exact fixed-point: an `i64` count of `10^-scale` units, so `Decimal64(2)`
    /// stores $12.34 as 1234. **Not a fifth physical kind** -- it rides
    /// [`PhysicalType::I64`] byte for byte, which is the whole reason it exists:
    /// zigzag, FOR, bitpack, LZ4, the zone maps and the `i128` sum accumulator
    /// all work on it unchanged. An `I128` physical type would instead have to
    /// be threaded through every codec and every match arm in the evaluator.
    ///
    /// The scale lives in the *type*, never in the data. 18 significant digits
    /// is the cap ([`MAX_DECIMAL_PRECISION`]); anything wider needs 128-bit
    /// storage and is rejected at parse time rather than silently truncated.
    Decimal64(u8),
    Nullable(Box<DataType>),
    /// Accepted and tracked for ClickHouse compatibility. It is a no-op hint:
    /// every string column is already dictionary-encoded per granule, so
    /// `LowCardinality(String)` and `String` have identical storage.
    LowCardinality(Box<DataType>),
}

/// Significant digits a `Decimal64` can hold: `10^18 - 1` is the largest
/// magnitude that fits an `i64` (`~9.22e18`) with room for a carry on an
/// addition, so the unit count is capped there rather than at `i64::MAX`.
pub const MAX_DECIMAL_PRECISION: u32 = 18;

impl DataType {
    pub fn physical(&self) -> PhysicalType {
        match self {
            DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Bool
            | DataType::Date
            | DataType::DateTime => PhysicalType::U64,
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Decimal64(_) => PhysicalType::I64,
            DataType::Float32 | DataType::Float64 => PhysicalType::F64,
            DataType::String | DataType::FixedString(_) => PhysicalType::Str,
            DataType::Nullable(inner) | DataType::LowCardinality(inner) => inner.physical(),
        }
    }

    /// Strip `Nullable`/`LowCardinality` wrappers.
    pub fn base(&self) -> &DataType {
        match self {
            DataType::Nullable(i) | DataType::LowCardinality(i) => i.base(),
            other => other,
        }
    }

    pub fn is_nullable(&self) -> bool {
        match self {
            DataType::Nullable(_) => true,
            DataType::LowCardinality(i) => i.is_nullable(),
            _ => false,
        }
    }

    pub fn to_nullable(&self) -> DataType {
        if self.is_nullable() {
            self.clone()
        } else {
            DataType::Nullable(Box::new(self.clone()))
        }
    }

    /// Drop the `Nullable` wrapper but keep everything else.
    pub fn strip_nullable(&self) -> DataType {
        match self {
            DataType::Nullable(i) => i.strip_nullable(),
            DataType::LowCardinality(i) => {
                DataType::LowCardinality(Box::new(i.strip_nullable()))
            }
            other => other.clone(),
        }
    }

    pub fn is_numeric(&self) -> bool {
        matches!(
            self.base().physical(),
            PhysicalType::U64 | PhysicalType::I64 | PhysicalType::F64
        ) && !matches!(self.base(), DataType::Date | DataType::DateTime)
    }

    /// A whole-number type. `Decimal64` is deliberately **not** one even at
    /// scale 0: `is_integer` gates the callers that then read the lane as the
    /// number (`substring` offsets, `intDiv`, `promote`'s int-vs-int arm), and a
    /// decimal's lane is its unit count, not its value.
    pub fn is_integer(&self) -> bool {
        matches!(self.base().physical(), PhysicalType::U64 | PhysicalType::I64)
            && !matches!(
                self.base(),
                DataType::Date | DataType::DateTime | DataType::Bool | DataType::Decimal64(_)
            )
    }

    #[inline]
    pub fn is_decimal(&self) -> bool {
        matches!(self.base(), DataType::Decimal64(_))
    }

    /// The declared scale, for the callers that must rescale before they can
    /// treat two lanes as commensurable.
    #[inline]
    pub fn decimal_scale(&self) -> Option<u8> {
        match self.base() {
            DataType::Decimal64(s) => Some(*s),
            _ => None,
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(self.base().physical(), PhysicalType::F64)
    }

    pub fn is_string(&self) -> bool {
        matches!(self.base().physical(), PhysicalType::Str)
    }

    pub fn is_temporal(&self) -> bool {
        matches!(self.base(), DataType::Date | DataType::DateTime)
    }

    /// Bit width of the declared type, used to range-check literals on INSERT.
    /// Returns `None` for variable-width types.
    pub fn int_bounds(&self) -> Option<(i128, i128)> {
        Some(match self.base() {
            DataType::UInt8 => (0, u8::MAX as i128),
            DataType::UInt16 => (0, u16::MAX as i128),
            DataType::UInt32 => (0, u32::MAX as i128),
            DataType::UInt64 => (0, u64::MAX as i128),
            DataType::Int8 => (i8::MIN as i128, i8::MAX as i128),
            DataType::Int16 => (i16::MIN as i128, i16::MAX as i128),
            DataType::Int32 => (i32::MIN as i128, i32::MAX as i128),
            DataType::Int64 => (i64::MIN as i128, i64::MAX as i128),
            DataType::Bool => (0, 1),
            DataType::Date => (0, u32::MAX as i128),
            DataType::DateTime => (i64::MIN as i128, i64::MAX as i128),
            _ => return None,
        })
    }

    /// The zero of this type: what a non-nullable column with no `DEFAULT`
    /// holds when a row does not mention it.
    ///
    /// Keyed off the physical kind rather than the SQL type so the four-kind
    /// invariant of this module stays the only thing to keep in sync; the
    /// `U64` arm then splits out the types whose zero is not the integer 0
    /// (`Bool`, and the temporals, whose epoch value must render as a date).
    pub fn zero_value(&self) -> Value {
        match self.base().physical() {
            PhysicalType::Str => Value::str(""),
            PhysicalType::F64 => Value::Float(0.0),
            PhysicalType::I64 => match self.base() {
                DataType::Decimal64(s) => Value::Decimal(0, *s),
                _ => Value::Int(0),
            },
            PhysicalType::U64 => match self.base() {
                DataType::Bool => Value::Bool(false),
                DataType::Date => Value::Date(0),
                DataType::DateTime => Value::DateTime(0),
                _ => Value::UInt(0),
            },
        }
    }

    /// Type promotion for binary arithmetic, following ClickHouse's rules
    /// closely enough to be unsurprising: float wins over int, signed wins
    /// over unsigned, wider wins over narrower.
    ///
    /// **The decimal arms below now govern every decimal-point literal in the
    /// language**, not just columns someone declared `Decimal64`: `0.1` lexes
    /// as `Decimal(1, 1)` (see the header of sql/lexer.rs), so `1.5 + 1` is a
    /// decimal and `1.5 + 1.5e0` is a float purely by this table. That is
    /// Postgres's rule -- an unsuffixed decimal constant is `numeric` and
    /// becomes a float only when combined with one -- and it is what makes
    /// `SELECT 0.1 + 0.2` answer `0.3`. It also means a mistake here is no
    /// longer confined to one type: it is the arithmetic of the whole dialect.
    pub fn promote(a: &DataType, b: &DataType) -> Result<DataType> {
        let nullable = a.is_nullable() || b.is_nullable();
        let (ba, bb) = (a.base(), b.base());
        let out = match (ba, bb) {
            // Matched *before* the equal-types shortcut on the next line: two
            // Bools are the one same-type pair whose arithmetic does not fit
            // its own type, and the shortcut used to hand them a Bool result
            // lane. `(1=1)+(2=2)` is 2, which a Bool lane rendered back as
            // `true`; `(1=2)-(1=1)` is -1, which it could not hold at all and
            // killed the query with "-1 is not a Bool" from `Column::constant`.
            // One expression shape, truncating one way and aborting the other.
            //
            // Int64 rather than UInt8/UInt64 because subtraction is the half
            // that has no unsigned answer, and it is what the mixed arm below
            // already yields for `(1=1)+1`, so the Bool row of the table is now
            // monotone. Knock-on: `if`/`CASE`/`coalesce`/`greatest` over two
            // Bools also unify to Int64 and render 1/0 instead of true/false.
            // Accepted -- these all share one promotion table, and a rendering
            // convention is a cheaper thing to lose than an arithmetic answer.
            (DataType::Bool, DataType::Bool) => DataType::Int64,
            _ if ba == bb => ba.clone(),
            (DataType::Float64, x) | (x, DataType::Float64) if x.is_numeric() => DataType::Float64,
            (DataType::Float32, x) | (x, DataType::Float32) if x.is_numeric() => DataType::Float64,
            // Decimal arms sit below Float on purpose: mixing an exact type with
            // an inexact one has to give the inexact one, or `dec + 0.1` would
            // claim an exactness it cannot deliver. Against an integer the
            // decimal wins, because every integer is exactly representable at
            // any scale.
            //
            // Addition and subtraction only need the *wider* scale (1.5 + 2.25
            // is 3.75, scale 2); multiplication needs `s1 + s2` and is therefore
            // **not** routed through here -- see `arith_ty` in
            // exec/functions/scalar.rs.
            (DataType::Decimal64(x), DataType::Decimal64(y)) => DataType::Decimal64(*x.max(y)),
            (DataType::Decimal64(s), x) | (x, DataType::Decimal64(s))
                if x.is_integer() || matches!(x, DataType::Bool) =>
            {
                DataType::Decimal64(*s)
            }
            (x, y) if x.is_integer() && y.is_integer() => {
                let signed = matches!(x.physical(), PhysicalType::I64)
                    || matches!(y.physical(), PhysicalType::I64);
                if signed {
                    DataType::Int64
                } else {
                    DataType::UInt64
                }
            }
            // Date arithmetic: Date +/- Int stays a Date.
            (DataType::Date, x) | (x, DataType::Date) if x.is_integer() => DataType::Date,
            (DataType::DateTime, x) | (x, DataType::DateTime) if x.is_integer() => {
                DataType::DateTime
            }
            (DataType::Bool, x) | (x, DataType::Bool) if x.is_integer() => x.clone(),
            (DataType::String, DataType::FixedString(_))
            | (DataType::FixedString(_), DataType::String) => DataType::String,
            (DataType::FixedString(m), DataType::FixedString(n)) => {
                DataType::FixedString(*m.max(n))
            }
            _ => {
                return Err(Error::bind(format!(
                    "no common type for {a} and {b}"
                )))
            }
        };
        Ok(if nullable { out.to_nullable() } else { out })
    }

    /// Parse a ClickHouse type name.
    ///
    /// Runs once per column per part load, not only at DDL, so the happy path
    /// allocates nothing: the name is ASCII-lowercased into stack space rather
    /// than into a `String`. The old code built *two* `String`s per call --
    /// one for the whole name, one for the head -- and for a parameterized
    /// type the first was then never read.
    pub fn parse(name: &str) -> Result<DataType> {
        let t = name.trim();
        let mut buf = [0u8; LOWER_CAP];
        // Parameterized types: Name(args)
        if let Some(open) = t.find('(') {
            if !t.ends_with(')') {
                return Err(Error::bind(format!("malformed type `{t}`")));
            }
            let head = lower(t[..open].trim(), &mut buf).unwrap_or("");
            let arg = t[open + 1..t.len() - 1].trim();
            return match head {
                "nullable" => Ok(DataType::Nullable(Box::new(DataType::parse(arg)?))),
                "lowcardinality" => Ok(DataType::LowCardinality(Box::new(DataType::parse(arg)?))),
                "fixedstring" => arg
                    .parse::<u32>()
                    .map(DataType::FixedString)
                    .map_err(|_| Error::bind(format!("FixedString length must be an integer, got `{arg}`"))),
                // Both spellings carry arguments this engine cannot keep; see
                // `datetime_args` for which subset is a genuine no-op.
                "datetime" => datetime_args(t, arg, false),
                "datetime64" => datetime_args(t, arg, true),
                // `Decimal(P, S)` names a precision we then have to honour;
                // `Decimal32/64(S)` name it implicitly (9 and 18 digits).
                "decimal" | "numeric" | "dec" => parse_decimal(t, arg, None),
                "decimal32" => parse_decimal(t, arg, Some(9)),
                "decimal64" => parse_decimal(t, arg, Some(18)),
                "decimal128" => Err(too_wide(t, 38)),
                "decimal256" => Err(too_wide(t, 76)),
                _ => Err(Error::bind(format!("unknown type `{t}`"))),
            };
        }
        Ok(match lower(t, &mut buf).unwrap_or("") {
            "uint8" => DataType::UInt8,
            "uint16" => DataType::UInt16,
            "uint32" => DataType::UInt32,
            "uint64" => DataType::UInt64,
            "int8" | "tinyint" => DataType::Int8,
            "int16" | "smallint" => DataType::Int16,
            "int32" | "int" | "integer" => DataType::Int32,
            "int64" | "bigint" => DataType::Int64,
            "float32" | "float" | "real" => DataType::Float32,
            "float64" | "double" => DataType::Float64,
            "bool" | "boolean" => DataType::Bool,
            "string" | "text" | "varchar" => DataType::String,
            "date" => DataType::Date,
            // `Date32` used to be an alias for `Date`, which is a narrowing
            // this engine cannot perform: ClickHouse's `Date32` is a *signed*
            // day count spanning 1900-2299 and ours is unsigned from the
            // epoch, so the whole lower half of the declared range was gone.
            // The failure that produced was loud but nonsensical -- a column
            // explicitly asked for 1950 and then rejected 1950 at INSERT.
            "date32" => {
                return Err(Error::unsupported(
                    "`Date32` spans 1900-2299; `Date` here is an unsigned day count from \
                     1970-01-01 and cannot represent its lower half, so the alias would \
                     narrow the range you declared. Use `Date` (1970-2149), or `DateTime` \
                     if you need instants before the epoch",
                ))
            }
            "datetime" | "timestamp" => DataType::DateTime,
            _ => return Err(Error::bind(format!("unknown type `{t}`"))),
        })
    }
}

/// The longest type name this engine answers to is `LowCardinality` (14), so a
/// name that does not fit cannot be one of ours and short-circuits to the
/// "unknown type" arm without touching the buffer.
const LOWER_CAP: usize = 16;

/// ASCII-lowercase into caller-supplied stack space.
///
/// Exists only to keep [`DataType::parse`] allocation-free: it is on the
/// part-load path (once per column per part), where a `String` per call is a
/// malloc/free pair spent to compare against a fixed set of literals.
///
/// Measured A/B interleaved (12 rounds, best-of, alternating old/new in one
/// loop over a 20-column mix of plain and parameterized names): **87-111
/// ns/parse before, 42-50 ns/parse after, 2.1-2.3x**. The spread is this
/// machine's usual 3x noise; the ratio held on every round.
#[inline]
fn lower<'b>(s: &str, buf: &'b mut [u8; LOWER_CAP]) -> Option<&'b str> {
    let b = s.as_bytes();
    if b.len() > LOWER_CAP {
        return None;
    }
    buf[..b.len()].copy_from_slice(b);
    buf[..b.len()].make_ascii_lowercase();
    // `make_ascii_lowercase` only rewrites 'A'..='Z', so valid UTF-8 in stays
    // valid UTF-8 out; the re-check is over at most 16 bytes and buys `safe`.
    std::str::from_utf8(&buf[..b.len()]).ok()
}

/// The arguments of `DateTime` / `DateTime64` -- the two places a DDL script
/// copied from ClickHouse asks this engine for something its `DateTime` lane
/// (a signed count of **whole UTC seconds**) does not have.
///
/// Both used to be accepted and dropped, which is the `DEFAULT`-stored-as-text
/// bug in a different costume: the column claims a property, `SHOW CREATE
/// TABLE` quietly echoes back a type without it, and every value in it is
/// wrong with nothing to report it. `DateTime64(3)` stored
/// `'2024-01-15 12:00:00.456'` as `12:00:00` -- the fraction was gone at
/// ingest, so no later fix could recover it. `DateTime('America/New_York')`
/// promised local time and returned UTC, off by up to 14 hours depending on
/// the zone, on data that looked entirely plausible.
///
/// Implementing either is a real feature, not a parser fix: sub-second
/// precision needs a scale in the type *and* a `Value`/render/compare path
/// that carries it (the `Decimal64` treatment), and timezones need an IANA
/// database this zero-dependency crate does not ship. Until then the honest
/// answer is a refusal that names the limitation.
///
/// What still parses is the subset that is a genuine no-op, because rejecting
/// those would refuse DDL the engine implements exactly: a UTC-spelled zone
/// (the lane *is* UTC) and `DateTime64(0)` (that *is* second resolution).
fn datetime_args(t: &str, arg: &str, is64: bool) -> Result<DataType> {
    let mut parts = arg.split(',');
    let first = parts.next().unwrap_or("").trim();
    let second = parts.next().map(str::trim);
    if parts.next().is_some() {
        return Err(Error::bind(format!("`{t}` takes at most two arguments")));
    }
    // `DateTime64`'s first argument is the precision and its second the zone;
    // `DateTime` has only the zone.
    let tz = if is64 {
        match first.parse::<u32>() {
            Ok(0) => {}
            Ok(p) => {
                return Err(Error::unsupported(format!(
                    "`{t}`: `DateTime` here is a count of whole seconds, so scale {p} \
                     would store '12:00:00.456' as '12:00:00' and lose the fraction on \
                     the way in, unrecoverably. Use `DateTime` for seconds, or keep the \
                     sub-second count yourself in an Int64/Decimal64 column"
                )))
            }
            Err(_) => {
                return Err(Error::bind(format!(
                    "`{t}`: expected a precision, got `{first}`"
                )))
            }
        }
        second
    } else if second.is_some() {
        return Err(Error::bind(format!("`{t}` takes only a timezone")));
    } else {
        Some(first).filter(|s| !s.is_empty())
    };
    match tz {
        None => Ok(DataType::DateTime),
        Some(z) if is_utc(z) => Ok(DataType::DateTime),
        Some(z) => Err(Error::unsupported(format!(
            "`{t}`: this engine ships no timezone table -- `DateTime` is a UTC second \
             count and renders as UTC -- so a column declared in {z} would read back \
             shifted by that zone's offset with nothing to report it. Declare \
             `DateTime` and convert at the edges"
        ))),
    }
}

/// The zone spellings that name UTC itself, which the lane already is. Quotes
/// are trimmed here rather than by the caller because the type text arrives
/// re-serialized from tokens (`DateTime('UTC')`) at DDL and bare from the
/// catalog on reload.
fn is_utc(z: &str) -> bool {
    let z = z.trim().trim_matches('\'');
    ["utc", "etc/utc", "gmt", "etc/gmt", "z", "zulu", "universal", "uct"]
        .iter()
        .any(|u| z.eq_ignore_ascii_case(u))
}

/// "Your precision does not fit an `i64`", said once and said loudly.
///
/// Truncating instead would be the worst possible failure mode for this type:
/// a `Decimal(38, 2)` quietly narrowed to 18 digits looks correct on every row
/// until the first one that overflows, and by then it is on disk.
#[cold]
fn too_wide(t: &str, digits: u32) -> Error {
    Error::unsupported(format!(
        "`{t}` asks for {digits} significant digits, but decimals live in an i64 \
         lane here and hold at most {MAX_DECIMAL_PRECISION}; use Decimal64(S) \
         (or Float64, if approximation is acceptable)"
    ))
}

/// `Decimal(P, S)`, `Decimal(P)`, `Decimal32(S)`, `Decimal64(S)`.
///
/// `implied` is the precision the *spelling* already fixes, and it doubles as
/// the arity switch: with it the single argument is the scale, without it the
/// first argument is the precision.
fn parse_decimal(t: &str, arg: &str, implied: Option<u32>) -> Result<DataType> {
    let mut parts = arg.split(',');
    let first = parts.next().unwrap_or("").trim();
    let second = parts.next().map(str::trim);
    if parts.next().is_some() {
        return Err(Error::bind(format!("`{t}` takes at most two arguments")));
    }
    let num = |s: &str| {
        s.parse::<u32>()
            .map_err(|_| Error::bind(format!("`{t}`: expected a number, got `{s}`")))
    };
    let (prec, scale) = match (implied, second) {
        (Some(_), Some(_)) => {
            return Err(Error::bind(format!(
                "`{t}` names its precision already; it takes only a scale"
            )))
        }
        (Some(p), None) => (p, num(first)?),
        // ClickHouse's `Decimal(P)` is `Decimal(P, 0)`.
        (None, None) => (num(first)?, 0),
        (None, Some(s)) => (num(first)?, num(s)?),
    };
    if prec == 0 || prec > MAX_DECIMAL_PRECISION {
        return Err(too_wide(t, prec.max(1)));
    }
    if scale > prec {
        return Err(Error::bind(format!(
            "`{t}`: scale {scale} exceeds precision {prec}"
        )));
    }
    Ok(DataType::Decimal64(scale as u8))
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::UInt8 => write!(f, "UInt8"),
            DataType::UInt16 => write!(f, "UInt16"),
            DataType::UInt32 => write!(f, "UInt32"),
            DataType::UInt64 => write!(f, "UInt64"),
            DataType::Int8 => write!(f, "Int8"),
            DataType::Int16 => write!(f, "Int16"),
            DataType::Int32 => write!(f, "Int32"),
            DataType::Int64 => write!(f, "Int64"),
            DataType::Float32 => write!(f, "Float32"),
            DataType::Float64 => write!(f, "Float64"),
            DataType::Bool => write!(f, "Bool"),
            DataType::String => write!(f, "String"),
            DataType::FixedString(n) => write!(f, "FixedString({n})"),
            DataType::Date => write!(f, "Date"),
            DataType::DateTime => write!(f, "DateTime"),
            // Always the `Decimal64(S)` spelling, never the `Decimal(P, S)` one
            // it may have been written as: only the scale survives into the
            // type, so echoing back a precision we no longer track would be a
            // lie the catalog then persists. This is also what makes
            // `parse(t).to_string() == t` hold, which the catalog round trip
            // depends on.
            DataType::Decimal64(s) => write!(f, "Decimal64({s})"),
            DataType::Nullable(i) => write!(f, "Nullable({i})"),
            DataType::LowCardinality(i) => write!(f, "LowCardinality({i})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_roundtrips_through_display() {
        for t in [
            "UInt8", "UInt64", "Int32", "Float64", "String", "Date", "DateTime", "Bool",
            "FixedString(16)", "Nullable(Int64)", "LowCardinality(String)",
            "Nullable(LowCardinality(String))",
            // Only the scale survives into the type, so `Decimal64(S)` is the
            // canonical spelling and the one the catalog persists.
            "Decimal64(0)", "Decimal64(2)", "Decimal64(18)", "Nullable(Decimal64(4))",
        ] {
            let d = DataType::parse(t).unwrap();
            assert_eq!(d.to_string(), t, "roundtrip {t}");
        }
    }

    #[test]
    fn parse_is_case_insensitive_and_aliased() {
        assert_eq!(DataType::parse("uint64").unwrap(), DataType::UInt64);
        assert_eq!(DataType::parse("BIGINT").unwrap(), DataType::Int64);
        assert_eq!(DataType::parse("  Double  ").unwrap(), DataType::Float64);
        assert_eq!(DataType::parse("DateTime('UTC')").unwrap(), DataType::DateTime);
        assert!(DataType::parse("Blob").is_err());
        // Longer than `LOWER_CAP`: must reach the unknown-type arm, not panic
        // or truncate into a match.
        assert!(DataType::parse("SuperLongTypeNameThatIsNotOurs").is_err());
    }

    /// The subset of `DateTime(...)` arguments that is a real no-op, and must
    /// keep parsing: refusing these would reject DDL the engine implements
    /// exactly.
    #[test]
    fn utc_and_second_resolution_are_no_ops_and_still_parse() {
        for t in [
            "DateTime('UTC')",
            "datetime('utc')",
            "DateTime('Etc/UTC')",
            "DateTime('GMT')",
            "DateTime()",
            "DateTime64(0)",
            "DateTime64(0, 'UTC')",
        ] {
            assert_eq!(DataType::parse(t).unwrap(), DataType::DateTime, "{t}");
        }
    }

    /// The bug these pin: `DateTime64(3)` was accepted and stored
    /// `'2024-01-15 12:00:00.456'` as `12:00:00`, and
    /// `DateTime('America/New_York')` was accepted with the zone dropped, so
    /// every value read back was off by the zone's offset. Both echoed a bare
    /// `DateTime` from SHOW CREATE TABLE, so the DDL never matched the data.
    #[test]
    fn subsecond_and_timezone_are_refused_by_name() {
        let e = DataType::parse("DateTime64(3)").unwrap_err();
        assert!(matches!(e, Error::Unsupported(_)), "{e:?} should be NOT_IMPLEMENTED");
        assert!(e.to_string().contains("whole seconds"), "{e}");
        assert!(DataType::parse("DateTime64(9, 'UTC')").is_err());
        // A precision that is not a number is a typo, not a missing feature.
        assert!(DataType::parse("DateTime64(x)").is_err());
        // ...but a zone on a scale-0 column is still judged as a zone.
        let e = DataType::parse("DateTime64(0, 'Europe/Paris')").unwrap_err();
        assert!(e.to_string().contains("timezone table"), "{e}");

        let e = DataType::parse("DateTime('America/New_York')").unwrap_err();
        assert!(matches!(e, Error::Unsupported(_)), "{e:?} should be NOT_IMPLEMENTED");
        assert!(e.to_string().contains("America/New_York"), "{e} must name the zone");
        assert!(DataType::parse("DateTime('UTC', 'UTC')").is_err());

        // `Date32`'s declared range is half unrepresentable here, so the alias
        // was a narrowing too -- it accepted a 1950 column and then refused
        // 1950 at INSERT.
        let e = DataType::parse("Date32").unwrap_err();
        assert!(e.to_string().contains("1900-2299"), "{e}");
        assert_eq!(DataType::parse("Date").unwrap(), DataType::Date);
    }

    /// Every spelling of a decimal collapses to the scale, because the scale is
    /// all the `i64` lane needs. This used to be `Err(unsupported)`.
    #[test]
    fn decimal_spellings_all_reduce_to_a_scale() {
        let p = |s: &str| DataType::parse(s).unwrap();
        assert_eq!(p("Decimal(10, 2)"), DataType::Decimal64(2));
        assert_eq!(p("decimal(10,2)"), DataType::Decimal64(2));
        assert_eq!(p("NUMERIC(9, 4)"), DataType::Decimal64(4));
        assert_eq!(p("Decimal64(6)"), DataType::Decimal64(6));
        assert_eq!(p("Decimal32(4)"), DataType::Decimal64(4));
        // `Decimal(P)` is `Decimal(P, 0)`, as in ClickHouse.
        assert_eq!(p("Decimal(18)"), DataType::Decimal64(0));
        // The full 18 digits, all of them fractional, is legal.
        assert_eq!(p("Decimal(18, 18)"), DataType::Decimal64(18));
    }

    /// A precision that does not fit must *say the limit*, not truncate. A
    /// `Decimal(38, 2)` silently narrowed to 18 digits looks right on every row
    /// until the first one that overflows, and by then it is on disk.
    #[test]
    fn precision_over_the_i64_limit_is_refused_by_name() {
        for t in ["Decimal(19, 2)", "Decimal(38, 10)", "Decimal128(4)", "Decimal256(4)"] {
            let e = DataType::parse(t).unwrap_err().to_string();
            assert!(e.contains("18"), "{t}: {e} should name the 18-digit limit");
        }
        // Scale wider than precision is nonsense whatever the width.
        assert!(DataType::parse("Decimal(4, 6)").is_err());
        // Decimal32 caps at 9 digits, so its scale does too.
        assert!(DataType::parse("Decimal32(12)").is_err());
        // `Decimal64` already names its precision; a second argument is a typo.
        assert!(DataType::parse("Decimal64(10, 2)").is_err());
        assert!(DataType::parse("Decimal(1, 2, 3)").is_err());
        assert!(DataType::parse("Decimal(x)").is_err());
        assert!(DataType::parse("Decimal(0)").is_err());
    }

    #[test]
    fn decimal_is_an_i64_lane_but_not_an_integer() {
        let d = DataType::Decimal64(2);
        // The whole design: byte-identical to Int64 underneath.
        assert_eq!(d.physical(), DataType::Int64.physical());
        assert!(d.is_numeric());
        assert!(!d.is_float());
        assert!(!d.is_temporal());
        // `is_integer` gates callers that then read the lane as the number
        // (`substring` offsets, `promote`'s int-vs-int arm) -- a decimal's lane
        // is a unit count, so it must not qualify even at scale 0.
        assert!(!d.is_integer());
        assert!(!DataType::Decimal64(0).is_integer());
        assert_eq!(d.decimal_scale(), Some(2));
        assert_eq!(d.to_nullable().decimal_scale(), Some(2));
        assert_eq!(DataType::Int64.decimal_scale(), None);
        assert!(matches!(d.zero_value(), Value::Decimal(0, 2)));
        // Range-checked by scale, not by bit width, so `int_bounds` declines.
        assert_eq!(d.int_bounds(), None);
    }

    #[test]
    fn decimal_promotion_keeps_exactness_where_it_can() {
        let p = |a: DataType, b: DataType| DataType::promote(&a, &b).unwrap();
        let d = DataType::Decimal64;
        // Addition only needs the wider scale; multiplication adds them, which
        // `arith_ty` (exec/functions/scalar.rs) overrides this table to do.
        assert_eq!(p(d(2), d(4)), d(4));
        assert_eq!(p(d(4), d(2)), d(4));
        assert_eq!(p(d(2), d(2)), d(2));
        // An integer is exactly representable at any scale, so the decimal wins.
        assert_eq!(p(d(2), DataType::Int64), d(2));
        assert_eq!(p(DataType::UInt8, d(3)), d(3));
        assert_eq!(p(d(2), DataType::Bool), d(2));
        // A float is not, so it wins instead -- claiming exactness we cannot
        // deliver would be worse than admitting the approximation.
        assert_eq!(p(d(2), DataType::Float64), DataType::Float64);
        assert_eq!(p(DataType::Float32, d(2)), DataType::Float64);
        assert_eq!(p(d(2).to_nullable(), DataType::Int64), d(2).to_nullable());
        assert!(DataType::promote(&d(2), &DataType::String).is_err());
        assert!(DataType::promote(&d(2), &DataType::Date).is_err());
    }

    #[test]
    fn physical_mapping_is_four_way() {
        assert_eq!(DataType::UInt8.physical(), PhysicalType::U64);
        assert_eq!(DataType::Date.physical(), PhysicalType::U64);
        assert_eq!(DataType::Int8.physical(), PhysicalType::I64);
        assert_eq!(DataType::Float32.physical(), PhysicalType::F64);
        assert_eq!(DataType::String.physical(), PhysicalType::Str);
        assert_eq!(
            DataType::Nullable(Box::new(DataType::Int32)).physical(),
            PhysicalType::I64
        );
    }

    #[test]
    fn promotion_rules() {
        let p = |a: DataType, b: DataType| DataType::promote(&a, &b).unwrap();
        assert_eq!(p(DataType::Int32, DataType::Int64), DataType::Int64);
        assert_eq!(p(DataType::UInt8, DataType::UInt32), DataType::UInt64);
        assert_eq!(p(DataType::Int32, DataType::UInt32), DataType::Int64);
        assert_eq!(p(DataType::Int32, DataType::Float64), DataType::Float64);
        assert_eq!(p(DataType::Float32, DataType::UInt8), DataType::Float64);
        assert_eq!(p(DataType::Date, DataType::Int32), DataType::Date);
        // Bool must widen against *itself* too, or `(1=1)+(2=2)` gets a result
        // lane that cannot hold 2 and `(1=2)-(1=1)` one that cannot hold -1.
        assert_eq!(p(DataType::Bool, DataType::Bool), DataType::Int64);
        assert_eq!(p(DataType::Bool, DataType::Int32), DataType::Int32);
        assert_eq!(p(DataType::Bool, DataType::UInt8), DataType::UInt8);
        assert_eq!(p(DataType::Bool, DataType::Float64), DataType::Float64);
        // nullability is contagious
        assert_eq!(
            p(DataType::Nullable(Box::new(DataType::Int32)), DataType::Int64),
            DataType::Nullable(Box::new(DataType::Int64))
        );
        assert!(DataType::promote(&DataType::String, &DataType::Int64).is_err());
    }

    #[test]
    fn nullable_helpers() {
        let n = DataType::Nullable(Box::new(DataType::Int32));
        assert!(n.is_nullable());
        assert_eq!(n.strip_nullable(), DataType::Int32);
        assert_eq!(n.to_nullable(), n);
        assert_eq!(DataType::Int32.to_nullable(), n);
        assert_eq!(n.base(), &DataType::Int32);
    }

    #[test]
    fn zero_value_matches_the_physical_kind() {
        // `Value`'s Eq collapses every numeric representation of 0, so these
        // assert on the variant: a `Date` column must get `Date(0)`, not
        // `UInt(0)`, or `Column::constant` builds the wrong lane.
        let z = |t: DataType| t.zero_value();
        assert!(matches!(z(DataType::UInt8), Value::UInt(0)));
        assert!(matches!(z(DataType::Int32), Value::Int(0)));
        assert!(matches!(z(DataType::Float32), Value::Float(f) if f == 0.0));
        assert!(matches!(z(DataType::Bool), Value::Bool(false)));
        assert!(matches!(z(DataType::Date), Value::Date(0)));
        assert!(matches!(z(DataType::DateTime), Value::DateTime(0)));
        assert_eq!(z(DataType::String).render_plain(), "");
        assert!(matches!(z(DataType::String), Value::Str(_)));
        // Wrappers delegate to the inner type; nullability is the caller's call.
        assert!(matches!(
            z(DataType::LowCardinality(Box::new(DataType::String))),
            Value::Str(_)
        ));
        assert!(matches!(z(DataType::Int64.to_nullable()), Value::Int(0)));
    }

    #[test]
    fn int_bounds_gate_literals() {
        assert_eq!(DataType::UInt8.int_bounds(), Some((0, 255)));
        assert_eq!(DataType::Int8.int_bounds(), Some((-128, 127)));
        assert_eq!(DataType::String.int_bounds(), None);
    }
}
