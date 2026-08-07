//! Scalar values: literals, aggregate accumulators, and result-set cells.
//!
//! Bulk data never flows through `Value` -- that is what `Block` is for. This
//! type exists at the edges: parsing literals, rendering results, and holding
//! group keys.

use super::datatype::{DataType, PhysicalType, MAX_DECIMAL_PRECISION};
use crate::common::{Error, Result};
use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum Value {
    Null,
    Bool(bool),
    UInt(u64),
    Int(i64),
    Float(f64),
    Str(Arc<str>),
    /// Days since 1970-01-01.
    Date(u32),
    /// Seconds since the Unix epoch.
    DateTime(i64),
    /// Exact fixed point: `(units, scale)` means `units * 10^-scale`, so $12.34
    /// is `Decimal(1234, 2)`.
    ///
    /// The scale rides along because `Value` is the one place in the engine that
    /// has no column type next to it -- it is what a result cell, a group key
    /// and a folded literal are. Columns keep the scale in
    /// [`DataType::Decimal64`] and store only the `i64`.
    ///
    /// Costs nothing: `Str(Arc<str>)` already makes `Value` 24 bytes, and
    /// `(i64, u8)` fits inside that (asserted in `value_stays_three_words`).
    Decimal(i64, u8),
}

/// `10^k`, `k` in `0..=18`. A table rather than `10i128.pow(k)`: rescaling runs
/// once per block boundary and once per literal, and a lookup keeps it off the
/// multiply chain in the checked paths below.
///
/// `i128` because every exact rescale and every decimal multiply widens through
/// it -- two 18-digit operands make 36 digits, which fits `i128` (39) and
/// nothing narrower.
pub const POW10: [i128; 19] = [
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
    10_000_000_000,
    100_000_000_000,
    1_000_000_000_000,
    10_000_000_000_000,
    100_000_000_000_000,
    1_000_000_000_000_000,
    10_000_000_000_000_000,
    100_000_000_000_000_000,
    1_000_000_000_000_000_000,
];

/// Largest unit count a `Decimal64` may hold: 18 nines.
///
/// Deliberately short of `i64::MAX` (~9.22e18). Clamping at a power of ten is
/// what makes "18 significant digits" a statement about the *decimal* rather
/// than about a binary boundary, and it leaves the top three bits free so a
/// single add of two in-range values cannot wrap before the range check sees it.
pub const DECIMAL_MAX_UNITS: i128 = POW10[MAX_DECIMAL_PRECISION as usize] - 1;

/// Move `units` from scale `from` to scale `to`, exactly.
///
/// Widening is a multiply. Narrowing rounds **half away from zero** -- the rule
/// every invoice in the world is written under, and the one Postgres uses for
/// `CAST(x AS numeric(p,s))`; banker's rounding is right for statistics and
/// wrong for money, and this type exists for money. `None` on overflow, never a
/// wrap: a wrapped price is worse than a failed query.
pub fn decimal_rescale(units: i128, from: u8, to: u8) -> Option<i128> {
    if from == to {
        return Some(units);
    }
    if to > from {
        return units.checked_mul(POW10[(to - from) as usize]);
    }
    let div = POW10[(from - to) as usize];
    let q = units / div;
    let rem = units % div;
    // `rem` carries the sign of `units`, so the half-way test and the bump are
    // both sign-symmetric without a branch on the sign itself.
    Some(if rem.unsigned_abs() * 2 >= div.unsigned_abs() {
        q + if units < 0 { -1 } else { 1 }
    } else {
        q
    })
}

/// [`decimal_rescale`] plus the range check, which is where a decimal actually
/// gets refused.
fn rescale_checked(units: i128, from: u8, to: u8, what: &dyn fmt::Display) -> Result<i64> {
    decimal_rescale(units, from, to)
        .filter(|u| u.abs() <= DECIMAL_MAX_UNITS)
        .map(|u| u as i64)
        .ok_or_else(|| {
            Error::exec(format!(
                "{what} does not fit Decimal64({to}): more than \
                 {MAX_DECIMAL_PRECISION} significant digits"
            ))
        })
}

impl Value {
    pub fn str(s: impl Into<Arc<str>>) -> Value {
        Value::Str(s.into())
    }

    /// A decimal from an already-scaled unit count, range-checked.
    pub fn decimal(units: i128, scale: u8) -> Result<Value> {
        Ok(Value::Decimal(rescale_checked(units, scale, scale, &units)?, scale))
    }

    /// `(units, scale)` when this is a decimal. The units are the *lane*, not
    /// the number: `Decimal(1234, 2).decimal_parts()` is `(1234, 2)`, worth 12.34.
    #[inline]
    pub fn decimal_parts(&self) -> Option<(i64, u8)> {
        match self {
            Value::Decimal(u, s) => Some((*u, *s)),
            _ => None,
        }
    }

    /// This value's unit count at `scale`, widening or rounding as needed.
    ///
    /// Integers convert exactly (every integer is representable at every
    /// scale); floats go through their decimal spelling rather than
    /// `x * 10^s`, because `0.1 * 100.0` is 10.000000000000002 and truncating
    /// that is how a cent goes missing.
    pub fn to_decimal_units(&self, scale: u8) -> Result<i64> {
        match self {
            Value::Decimal(u, s) => rescale_checked(*u as i128, *s, scale, self),
            Value::Float(f) => {
                if !f.is_finite() {
                    return Err(Error::exec(format!("cannot store {f} in Decimal64({scale})")));
                }
                // Through the double's *shortest round-tripping* spelling, so
                // the rounding that matters happens once, in `parse_decimal_str`,
                // under this type's half-away-from-zero rule. Formatting
                // straight to `scale` digits instead would hand the job to
                // `{:.*}`, which rounds half to **even** -- `CAST(2.5 AS
                // Decimal(9,0))` came back 2 while `CAST('2.5' AS ...)` came
                // back 3, one cast disagreeing with the other over one keystroke.
                // Postgres takes the same route for float8 -> numeric.
                parse_decimal_str(&format!("{f}"), scale)
                    .ok_or_else(|| Error::exec(format!("cannot store {f} in Decimal64({scale})")))
            }
            Value::Str(s) => parse_decimal_str(s.trim(), scale)
                .ok_or_else(|| Error::exec(format!("cannot cast '{s}' to Decimal64({scale})"))),
            Value::Null => Err(Error::exec("cannot store NULL in a Decimal64")),
            other => {
                let n = other
                    .as_i64()
                    .map(i128::from)
                    .or_else(|| other.as_u64().map(i128::from))
                    .ok_or_else(|| Error::exec(format!("cannot cast {other} to Decimal64({scale})")))?;
                rescale_checked(n, 0, scale, other)
            }
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// The narrowest type that can hold this value. Literals get widened
    /// later by `DataType::promote`.
    pub fn data_type(&self) -> DataType {
        match self {
            Value::Null => DataType::Nullable(Box::new(DataType::Int64)),
            Value::Bool(_) => DataType::Bool,
            Value::UInt(_) => DataType::UInt64,
            Value::Int(_) => DataType::Int64,
            Value::Float(_) => DataType::Float64,
            Value::Str(_) => DataType::String,
            Value::Date(_) => DataType::Date,
            Value::DateTime(_) => DataType::DateTime,
            Value::Decimal(_, s) => DataType::Decimal64(*s),
        }
    }

    pub fn physical(&self) -> Option<PhysicalType> {
        Some(match self {
            Value::Null => return None,
            Value::Bool(_) | Value::UInt(_) | Value::Date(_) => PhysicalType::U64,
            Value::Int(_) | Value::DateTime(_) | Value::Decimal(..) => PhysicalType::I64,
            Value::Float(_) => PhysicalType::F64,
            Value::Str(_) => PhysicalType::Str,
        })
    }

    /// **The physical lane, not the number.**
    ///
    /// `Date` hands back days and `DateTime` epoch seconds, and `Decimal` is the
    /// same rule taken one step further: it hands back the *unit count*, so
    /// `Decimal(1234, 2)` (which is 12.34) yields 1234. That is deliberate and
    /// load-bearing -- `Column::constant`, `Column::push_value` and
    /// [`Value::to_lane_phys`] all build an `I64` lane straight out of this, and
    /// a "helpfully" descaled 12 there would write the wrong number to disk.
    ///
    /// Everything that wants the *value* instead goes through [`Value::as_f64`],
    /// [`Value::cast_to`] or the `Ord`/`Hash` impls below, all of which treat
    /// the scale as meaningful. Nothing else may read a decimal through here.
    pub fn as_i64(&self) -> Option<i64> {
        Some(match self {
            Value::Bool(b) => *b as i64,
            Value::UInt(u) => {
                if *u > i64::MAX as u64 {
                    return None;
                }
                *u as i64
            }
            Value::Int(i) | Value::Decimal(i, _) => *i,
            Value::Float(f) => {
                if f.is_finite() && *f >= i64::MIN as f64 && *f <= i64::MAX as f64 {
                    *f as i64
                } else {
                    return None;
                }
            }
            Value::Date(d) => *d as i64,
            Value::DateTime(t) => *t,
            _ => return None,
        })
    }

    /// The physical lane again, with the same caveat as [`Value::as_i64`].
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            Value::Bool(b) => *b as u64,
            Value::UInt(u) => *u,
            Value::Int(i) | Value::Decimal(i, _) => {
                if *i < 0 {
                    return None;
                }
                *i as u64
            }
            Value::Date(d) => *d as u64,
            Value::DateTime(t) => {
                if *t < 0 {
                    return None;
                }
                *t as u64
            }
            Value::Float(f) => {
                if f.is_finite() && *f >= 0.0 && *f <= u64::MAX as f64 {
                    *f as u64
                } else {
                    return None;
                }
            }
            _ => return None,
        })
    }

    /// The **numeric value**, unlike [`Value::as_i64`]: a decimal is descaled
    /// here, because every caller of this one is asking "how big is it" rather
    /// than "what is in the lane".
    ///
    /// One divide, not `units * 10^-s`: IEEE division is correctly rounded, so
    /// two decimals that are numerically equal at different scales (1234/100 and
    /// 12340/1000) land on the *same* double, which is what keeps `Hash`
    /// agreeing with `Ord` below.
    pub fn as_f64(&self) -> Option<f64> {
        Some(match self {
            Value::Bool(b) => *b as u8 as f64,
            Value::UInt(u) => *u as f64,
            Value::Int(i) => *i as f64,
            Value::Float(f) => *f,
            Value::Date(d) => *d as f64,
            Value::DateTime(t) => *t as f64,
            Value::Decimal(u, s) => *u as f64 / POW10[*s as usize] as f64,
            _ => return None,
        })
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// SQL truthiness: non-zero / non-empty. NULL is *not* true.
    pub fn truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::UInt(u) => *u != 0,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::Date(d) => *d != 0,
            Value::DateTime(t) => *t != 0,
            // Zero units is zero at every scale, so the lane answers this one
            // without a rescale.
            Value::Decimal(u, _) => *u != 0,
        }
    }

    /// Coerce to a declared column type, range-checking integers so a bad
    /// INSERT is rejected rather than silently wrapping.
    pub fn cast_to(&self, ty: &DataType) -> Result<Value> {
        if self.is_null() {
            return if ty.is_nullable() {
                Ok(Value::Null)
            } else {
                Err(Error::exec(format!("cannot store NULL in non-nullable {ty}")))
            };
        }
        let base = ty.base();
        // Ahead of the physical dispatch: a decimal target is an `I64` lane, but
        // the integer arm below would read the source as a whole number and
        // throw the fraction away.
        if let DataType::Decimal64(s) = base {
            return Ok(Value::Decimal(self.to_decimal_units(*s)?, *s));
        }
        match base.physical() {
            PhysicalType::Str => Ok(Value::Str(self.render_plain().into())),
            PhysicalType::F64 => {
                // The integer arm below parses `Value::Str`, so refusing it
                // here made `DEFAULT '1.5'` on a Float64 column an error while
                // `DEFAULT '42'` on an Int64 one was accepted -- an asymmetry
                // with no reading of the type rules behind it. Postgres and
                // ClickHouse take both.
                //
                // Parsed here rather than by teaching `as_f64` about strings:
                // `as_f64` also backs `Ord` and `Hash`, where a numeric-looking
                // string turning into a number would merge `'1'` and `1` into
                // one GROUP BY key and reorder `ORDER BY` across the Str/number
                // rank boundary. `cast_to` is the one place the coercion is
                // asked for explicitly.
                //
                // `Float` is pulled out ahead of the `as_f64` fallback because
                // it is what an INSERT into a float column almost always
                // carries, and this is a per-value path: taking it here is one
                // match instead of two.
                let f = match self {
                    Value::Float(f) => *f,
                    Value::Str(s) => s
                        .trim()
                        .parse::<f64>()
                        .map_err(|_| Error::exec(format!("cannot cast '{s}' to {ty}")))?,
                    _ => self
                        .as_f64()
                        .ok_or_else(|| Error::exec(format!("cannot cast {self} to {ty}")))?,
                };
                Ok(Value::Float(if matches!(base, DataType::Float32) {
                    f as f32 as f64
                } else {
                    f
                }))
            }
            PhysicalType::U64 | PhysicalType::I64 => {
                let n = match self {
                    Value::Float(f) => {
                        if !f.is_finite() {
                            return Err(Error::exec(format!("cannot cast {f} to {ty}")));
                        }
                        f.trunc() as i128
                    }
                    // Truncates toward zero, exactly as the float arm above
                    // does, so `CAST(-1.9 AS Int)` is -1 whichever exact or
                    // inexact spelling it arrived in.
                    Value::Decimal(u, s) => *u as i128 / POW10[*s as usize],
                    Value::Str(s) => s
                        .trim()
                        .parse::<i128>()
                        .map_err(|_| Error::exec(format!("cannot cast '{s}' to {ty}")))?,
                    other => other
                        .as_i64()
                        .map(|v| v as i128)
                        .or_else(|| other.as_u64().map(|v| v as i128))
                        .ok_or_else(|| Error::exec(format!("cannot cast {self} to {ty}")))?,
                };
                if let Some((lo, hi)) = base.int_bounds() {
                    if n < lo || n > hi {
                        return Err(Error::exec(format!("value {n} out of range for {ty}")));
                    }
                }
                Ok(match base {
                    DataType::Bool => Value::Bool(n != 0),
                    DataType::Date => Value::Date(n as u32),
                    DataType::DateTime => Value::DateTime(n as i64),
                    _ if base.physical() == PhysicalType::I64 => Value::Int(n as i64),
                    _ => Value::UInt(n as u64),
                })
            }
        }
    }

    /// The raw `u64` lane this value occupies in packed storage.
    ///
    /// The mapping is order-preserving (see [`crate::common::lane`]), so a
    /// comparison on lanes is a comparison on values.
    pub fn to_lane(&self, ty: &DataType) -> Result<u64> {
        self.to_lane_phys(ty.base().physical(), ty)
    }

    /// [`Value::to_lane`] with the physical kind already resolved.
    ///
    /// `DataType::physical()` walks `Nullable`/`LowCardinality` wrappers with a
    /// recursive match. That is nothing once, and measurable when it happens
    /// per row on a write path, so callers that write many rows against one
    /// column resolve it once and pass it in.
    pub fn to_lane_phys(&self, phys: PhysicalType, ty: &DataType) -> Result<u64> {
        use crate::common::{f64_to_lane, i64_to_lane};
        // Decimals are settled ahead of the physical dispatch, because a
        // decimal's lane means nothing without a scale next to it and `phys`
        // alone does not carry one. Both directions matter: an `Int(2)` probing
        // a `Decimal64(2)` key column must lane as 200, and a `Decimal(150, 2)`
        // probing an `Int64` one must not lane as 150.
        //
        // Float lanes are left to the arm below: `as_f64` already descales, and
        // an approximate target has no exactness to protect.
        match (self.decimal_parts(), ty.decimal_scale(), phys) {
            (_, Some(s), _) => return self.to_decimal_units(s).map(i64_to_lane),
            (Some((u, s)), None, PhysicalType::U64 | PhysicalType::I64) => {
                // Exact or nothing. `WHERE id = 1.00` names key 1; `id = 1.50`
                // names no key at all, and must say so rather than truncate into
                // a probe for 1 and return a row no scan would have.
                let p = POW10[s as usize];
                let n = u as i128 / p;
                let bad = || Error::exec(format!("{self} is not a valid {ty}"));
                if n * p != u as i128 {
                    return Err(bad());
                }
                return match phys {
                    PhysicalType::U64 => u64::try_from(n).map_err(|_| bad()),
                    _ => Ok(i64_to_lane(n as i64)),
                };
            }
            _ => {}
        }
        match phys {
            PhysicalType::U64 => self
                .as_u64()
                .ok_or_else(|| Error::exec(format!("{self} is not a valid {ty}"))),
            PhysicalType::I64 => self
                .as_i64()
                .map(i64_to_lane)
                .ok_or_else(|| Error::exec(format!("{self} is not a valid {ty}"))),
            PhysicalType::F64 => self
                .as_f64()
                .map(f64_to_lane)
                .ok_or_else(|| Error::exec(format!("{self} is not a valid {ty}"))),
            PhysicalType::Str => Err(Error::exec("string values are lane-encoded via a dictionary")),
        }
    }

    /// The variant's own name, independent of the value it carries.
    ///
    /// Exists because `Value`'s [`Eq`] is deliberately blind to the variant:
    /// `UInt(0)`, `Int(0)`, `Float(0.0)`, `Date(0)` and `Bool(false)` are all
    /// one value (see the `Ord` impl -- `GROUP BY` depends on it). So
    /// `assert_eq!(v, Value::UInt(0))` passes for every one of them, and a test
    /// written that way cannot see a producer handing back the wrong physical
    /// kind -- which is not cosmetic: `Column::constant` builds its lane from
    /// the variant, so a `UInt` where a `Date` belongs is a wrong column.
    ///
    /// Use [`Value::same_variant`] (or `matches!`) when the variant is the
    /// thing under test.
    pub fn variant(&self) -> &'static str {
        match self {
            Value::Null => "Null",
            Value::Bool(_) => "Bool",
            Value::UInt(_) => "UInt",
            Value::Int(_) => "Int",
            Value::Float(_) => "Float",
            Value::Str(_) => "Str",
            Value::Date(_) => "Date",
            Value::DateTime(_) => "DateTime",
            Value::Decimal(..) => "Decimal",
        }
    }

    /// Do these two carry the same variant, ignoring the value in it?
    ///
    /// One discriminant compare, no allocation, so it is cheap enough for a
    /// `debug_assert!` on a per-value path.
    #[inline]
    pub fn same_variant(&self, other: &Value) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }

    /// Same variant *and* equal -- what `assert_eq!` looks like it checks.
    /// `Value::UInt(0).eq_exact(&Value::Int(0))` is false where `==` is true.
    #[inline]
    pub fn eq_exact(&self, other: &Value) -> bool {
        self.same_variant(other) && self == other
    }

    /// Render without SQL quoting, for casts to String and for CSV-ish output.
    pub fn render_plain(&self) -> String {
        match self {
            Value::Null => String::new(),
            Value::Bool(b) => if *b { "true" } else { "false" }.into(),
            Value::UInt(u) => u.to_string(),
            Value::Int(i) => i.to_string(),
            Value::Float(f) => fmt_float(*f),
            Value::Str(s) => s.to_string(),
            Value::Date(d) => fmt_date(*d),
            Value::DateTime(t) => fmt_datetime(*t),
            Value::Decimal(u, s) => fmt_decimal(*u, *s),
        }
    }
}

/// Render a decimal at its declared scale, trailing zeros and all: a
/// `Decimal64(2)` prints `1.50`, not `1.5`. The scale is a property of the
/// column, so dropping it in the output would make two rows of the same column
/// disagree about how much precision was stored.
fn fmt_decimal(units: i64, scale: u8) -> String {
    if scale == 0 {
        return units.to_string();
    }
    // Through `i128`/`u128` because `i64::MIN.abs()` has no `i64` answer, and a
    // lane really can hold it (nothing stops a hand-built column).
    let mag = (units as i128).unsigned_abs();
    let p = POW10[scale as usize] as u128;
    let w = scale as usize;
    format!(
        "{}{}.{:0w$}",
        if units < 0 { "-" } else { "" },
        mag / p,
        mag % p,
        w = w
    )
}

/// Parse an exact decimal literal into a unit count at `scale`.
///
/// Hand-rolled rather than `s.parse::<f64>()? * 10^scale`: that route turns
/// `'0.1'` into 10.000000000000002 units and truncates a cent away, which is
/// precisely the failure this whole type exists to remove. Digits are
/// accumulated in `i128` and rescaled once, so nothing is rounded twice.
///
/// An exponent is accepted (`1.5e3`), because a literal that came back out of
/// `toString` on a float can carry one.
fn parse_decimal_str(s: &str, scale: u8) -> Option<i64> {
    let b = s.as_bytes();
    let mut i = 0usize;
    let neg = match b.first() {
        Some(b'-') => {
            i = 1;
            true
        }
        Some(b'+') => {
            i = 1;
            false
        }
        _ => false,
    };
    let (mut units, mut frac, mut digits, mut point) = (0i128, 0i32, 0u32, false);
    while i < b.len() {
        match b[i] {
            c @ b'0'..=b'9' => {
                digits += 1;
                units = units.checked_mul(10)?.checked_add((c - b'0') as i128)?;
                frac += point as i32;
            }
            b'.' if !point => point = true,
            b'e' | b'E' => break,
            _ => return None,
        }
        i += 1;
    }
    if digits == 0 {
        return None;
    }
    let exp: i32 = if i < b.len() { s[i + 1..].parse().ok()? } else { 0 };
    // The literal currently sits at scale `frac - exp`; move it to `scale`.
    let mut shift = scale as i32 - (frac - exp);
    // Anything this far below the target scale rounds to zero: `units` is at
    // most 39 digits wide, so a divisor of 10^40 cannot leave a nonzero
    // quotient. Bailing here keeps the loop below to at most three passes.
    if shift < -40 {
        return Some(0);
    }
    while shift.abs() > 18 {
        let step = if shift > 0 { 18 } else { -18 };
        units = step_scale(units, step)?;
        shift -= step;
    }
    units = step_scale(units, shift)?;
    let units = if neg { -units } else { units };
    (units.abs() <= DECIMAL_MAX_UNITS).then_some(units as i64)
}

/// One `|shift| <= 18` step of [`parse_decimal_str`]'s rescale.
#[inline]
fn step_scale(units: i128, shift: i32) -> Option<i128> {
    if shift >= 0 {
        units.checked_mul(POW10[shift as usize])
    } else {
        decimal_rescale(units, (-shift) as u8, 0)
    }
}

fn fmt_float(f: f64) -> String {
    if f.is_nan() {
        "nan".into()
    } else if f.is_infinite() {
        if f > 0.0 { "inf".into() } else { "-inf".into() }
    } else if f == f.trunc() && f.abs() < 1e15 {
        format!("{f:.0}")
    } else {
        let s = format!("{f}");
        s
    }
}

// ------------------------------------------------------------ calendar math
// Howard Hinnant's civil-from-days / days-from-civil: branch-free, exact for
// the whole proleptic Gregorian range, and no dependency on a date crate.

pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64; // [0, 399]
    let mp = if m > 2 { m - 3 } else { m + 9 } as u64;
    let doy = (153 * mp + 2) / 5 + d as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe as i64 - 719_468
}

pub fn fmt_date(days: u32) -> String {
    let (y, m, d) = civil_from_days(days as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

pub fn fmt_datetime(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    format!("{y:04}-{m:02}-{d:02} {hh:02}:{mm:02}:{ss:02}")
}

/// Parse `YYYY-MM-DD` into a signed day count, which may be negative.
///
/// Split out from [`parse_date`] because `Date` and `DateTime` have different
/// ranges: `Date` is an unsigned day count starting at the epoch, while
/// `DateTime` is signed seconds and reaches back before 1970.
pub fn parse_civil_days(s: &str) -> Result<i64> {
    let b = s.trim();
    let parts: Vec<&str> = b.split('-').collect();
    if parts.len() != 3 {
        return Err(Error::exec(format!("cannot parse '{s}' as Date")));
    }
    let y: i64 = parts[0].parse().map_err(|_| Error::exec(format!("bad year in '{s}'")))?;
    let m: u32 = parts[1].parse().map_err(|_| Error::exec(format!("bad month in '{s}'")))?;
    let d: u32 = parts[2].parse().map_err(|_| Error::exec(format!("bad day in '{s}'")))?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return Err(Error::exec(format!("'{s}' is not a valid date")));
    }
    Ok(days_from_civil(y, m, d))
}

/// Parse `YYYY-MM-DD`, returning days since epoch.
pub fn parse_date(s: &str) -> Result<u32> {
    let days = parse_civil_days(s)?;
    if days < 0 || days > u32::MAX as i64 {
        return Err(Error::exec(format!("date '{s}' out of Date range")));
    }
    Ok(days as u32)
}

/// Parse `YYYY-MM-DD[ HH:MM:SS]`, returning seconds since epoch.
pub fn parse_datetime(s: &str) -> Result<i64> {
    let t = s.trim();
    let (dpart, tpart) = match t.split_once([' ', 'T']) {
        Some((d, r)) => (d, r),
        None => (t, "00:00:00"),
    };
    // DateTime is signed seconds, so pre-epoch instants are in range here even
    // though they are not representable as a `Date`.
    let days = parse_civil_days(dpart)?;
    let hms: Vec<&str> = tpart.trim().trim_end_matches('Z').split(':').collect();
    if hms.is_empty() || hms.len() > 3 {
        return Err(Error::exec(format!("cannot parse '{s}' as DateTime")));
    }
    let mut secs = 0i64;
    let mult = [3600i64, 60, 1];
    for (i, p) in hms.iter().enumerate() {
        // tolerate fractional seconds by truncating them
        let p = p.split('.').next().unwrap_or(p);
        let v: i64 = p.parse().map_err(|_| Error::exec(format!("bad time in '{s}'")))?;
        secs += v * mult[i];
    }
    Ok(days * 86_400 + secs)
}

// ------------------------------------------------------- ordering & equality
// f64 gets a total order (NaN sorts last, -0.0 == 0.0) so ORDER BY, min/max
// and GROUP BY keys are all well-defined. NULL sorts first, matching
// ClickHouse's default NULLS FIRST for ascending order.

fn rank(v: &Value) -> u8 {
    match v {
        Value::Null => 0,
        Value::Bool(_) | Value::UInt(_) | Value::Int(_) | Value::Float(_) | Value::Decimal(..) => 1,
        Value::Date(_) => 2,
        Value::DateTime(_) => 3,
        Value::Str(_) => 4,
    }
}

/// Exact comparison for anything with a decimal on either side.
///
/// Both sides widen to a common scale in `i128`, which cannot overflow: an
/// in-range decimal is under 10^18 and the largest rescale we can be asked for
/// is another 10^18, for 10^36 against `i128`'s 1.7e38. That exactness is the
/// point -- `price > 0.1` must not be decided by a double, or the type has
/// bought nothing.
///
/// Floats are the one thing it cannot do exactly, and they fall back to the
/// same `as_f64` widening the engine already uses for `Int` vs `Float`. Mixing
/// an exact type with an inexact one has no exact answer to give.
fn cmp_decimal(a: &Value, b: &Value) -> Ordering {
    let (au, asc) = match a.decimal_parts() {
        Some(d) => d,
        // The non-decimal side is only exactly comparable when it is integral.
        None => match a.as_i64().filter(|_| !matches!(a, Value::Float(_))) {
            Some(i) => (i, 0),
            None => return total_cmp_f64(a.as_f64().unwrap_or(f64::NAN), b.as_f64().unwrap_or(f64::NAN)),
        },
    };
    let (bu, bsc) = match b.decimal_parts() {
        Some(d) => d,
        None => match b.as_i64().filter(|_| !matches!(b, Value::Float(_))) {
            Some(i) => (i, 0),
            None => return total_cmp_f64(a.as_f64().unwrap_or(f64::NAN), b.as_f64().unwrap_or(f64::NAN)),
        },
    };
    let hi = asc.max(bsc);
    (au as i128 * POW10[(hi - asc) as usize]).cmp(&(bu as i128 * POW10[(hi - bsc) as usize]))
}

/// **`==` on `Value` ignores the variant.** It defers to `Ord`, which puts
/// every non-null, non-string value into one numeric equivalence class:
/// `UInt(0) == Int(0) == Float(0.0) == Date(0) == Bool(false)`. That is
/// load-bearing -- `GROUP BY`, `DISTINCT` and hash joins all rely on a key
/// comparing equal across the representations a plan may hand them -- and it
/// must not change.
///
/// The trap it sets is in *tests*: `assert_eq!(got, Value::UInt(0))` passes
/// for all five of those, so it does not pin down what it looks like it pins
/// down. A producer that returns the wrong variant is a real defect, because
/// `Column::constant` and `to_lane_phys` pick the physical lane off the
/// variant -- a `UInt(19723)` where a `Date(19723)` belongs builds a column
/// that renders as a number. Assert with [`Value::eq_exact`], `matches!`, or
/// [`Value::variant`] when the variant is what is under test.
///
/// To audit which tests lean on the collapse, temporarily make this `eq` read
/// `self.same_variant(other) && self.cmp(other) == Ordering::Equal` and run the
/// suite. Done here: 10 failures, 8 of them tests that probe the collapse *on
/// purpose* (`probe_*_compare_equal`, `numeric_comparison_crosses_
/// representations`) and one -- `in_between_like` in tests/sql.rs -- that fails
/// because `IN` really does need `UInt(1) == Int(1)`, which is the clearest
/// statement of why this impl cannot change.
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Value {}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Value::Null, Value::Null) => Ordering::Equal,
            (Value::Str(a), Value::Str(b)) => a.cmp(b),
            // A decimal's lane is its unit count, so the shared `as_i64` path
            // below would compare 1234 against 12 for `12.34` vs `12`. Split out
            // ahead of it, and exact.
            (Value::Decimal(..), _) | (_, Value::Decimal(..))
                if (1..=3).contains(&rank(self)) && (1..=3).contains(&rank(other)) =>
            {
                cmp_decimal(self, other)
            }
            // Numeric family compares by value across representations, so
            // `WHERE u = 1` works whether `u` is UInt8 or Float64.
            _ if rank(self) == rank(other) || (rank(self) <= 3 && rank(other) <= 3) => {
                if rank(self) == 0 || rank(other) == 0 {
                    return rank(self).cmp(&rank(other));
                }
                match (self.as_i64(), other.as_i64()) {
                    (Some(a), Some(b)) if !matches!(self, Value::Float(_))
                        && !matches!(other, Value::Float(_)) => a.cmp(&b),
                    _ => match (self.as_f64(), other.as_f64()) {
                        (Some(a), Some(b)) => total_cmp_f64(a, b),
                        _ => rank(self).cmp(&rank(other)),
                    },
                }
            }
            _ => rank(self).cmp(&rank(other)),
        }
    }
}

#[inline]
fn total_cmp_f64(a: f64, b: f64) -> Ordering {
    if a < b {
        Ordering::Less
    } else if a > b {
        Ordering::Greater
    } else if a == b {
        Ordering::Equal // collapses -0.0 == 0.0
    } else {
        // at least one NaN: NaN sorts last, two NaNs are equal
        match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            _ => Ordering::Less,
        }
    }
}

impl std::hash::Hash for Value {
    /// Must agree with [`Value::eq`], or hash-based grouping silently splits
    /// values that compare equal.
    ///
    /// `Ord` puts every non-string, non-null value into **one** numeric
    /// equivalence class — `Date(5)`, `UInt(5)`, `Int(5)`, `Float(5.0)` and
    /// `Bool(true)`-as-1 all compare equal — so they must all hash alike. An
    /// earlier version tagged `Date` and `DateTime` separately, which meant
    /// `GROUP BY`, `DISTINCT` and hash joins put equal keys in different
    /// buckets.
    ///
    /// Integral values hash through `i64` rather than `f64` so that two large
    /// integers a single ULP apart do not collide, while still agreeing with
    /// the `Float` that equals them.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::Null => 0u8.hash(state),
            Value::Str(s) => {
                1u8.hash(state);
                s.hash(state);
            }
            // A decimal that names a whole number *is* that number to `Ord`
            // (`Decimal(500, 2) == Int(5)`), so it must hash as one; only the
            // fractional ones fall through to the float lane, where the divide
            // in `as_f64` is correctly rounded and so agrees across scales.
            Value::Decimal(u, s) => {
                2u8.hash(state);
                let p = POW10[*s as usize];
                if *u as i128 % p == 0 {
                    0u8.hash(state);
                    ((*u as i128 / p) as i64).hash(state);
                } else {
                    1u8.hash(state);
                    self.as_f64().unwrap_or(0.0).to_bits().hash(state);
                }
            }
            v => {
                2u8.hash(state);
                let f = v.as_f64().unwrap_or(0.0);
                // Exact-integer path, shared by every integral representation.
                if f.is_finite()
                    && f.fract() == 0.0
                    && f >= i64::MIN as f64
                    && f <= i64::MAX as f64
                {
                    if let Some(i) = v.as_i64() {
                        0u8.hash(state);
                        i.hash(state);
                        return;
                    }
                }
                1u8.hash(state);
                // -0.0 and 0.0 are one value; all NaNs are one value.
                let f = if f == 0.0 {
                    0.0
                } else if f.is_nan() {
                    f64::NAN
                } else {
                    f
                };
                f.to_bits().hash(state);
            }
        }
    }
}

impl fmt::Display for Value {
    /// SQL-ish rendering: strings quoted, NULL spelled out. Use
    /// `render_plain` for cast-to-String semantics.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "NULL"),
            Value::Str(s) => write!(f, "'{}'", s.replace('\'', "''")),
            other => write!(f, "{}", other.render_plain()),
        }
    }
}

impl From<u64> for Value {
    fn from(v: u64) -> Self {
        Value::UInt(v)
    }
}
impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int(v)
    }
}
impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Float(v)
    }
}
impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Str(v.into())
    }
}
impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::Str(v.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn h(v: &Value) -> u64 {
        let mut s = DefaultHasher::new();
        v.hash(&mut s);
        s.finish()
    }

    #[test]
    fn calendar_roundtrips_over_four_centuries() {
        for d in (0i64..146_097 * 2).step_by(37) {
            let (y, m, dd) = civil_from_days(d);
            assert_eq!(days_from_civil(y, m, dd), d, "day {d}");
        }
    }

    #[test]
    fn known_dates() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 1, 1), 10_957);
        assert_eq!(fmt_date(0), "1970-01-01");
        assert_eq!(fmt_date(19_723), "2024-01-01");
        assert_eq!(parse_date("2024-02-29").unwrap(), 19_782); // leap day
        assert_eq!(fmt_date(parse_date("2024-02-29").unwrap()), "2024-02-29");
    }

    #[test]
    fn datetime_parse_and_format() {
        let t = parse_datetime("2024-01-15 13:45:30").unwrap();
        assert_eq!(fmt_datetime(t), "2024-01-15 13:45:30");
        assert_eq!(fmt_datetime(0), "1970-01-01 00:00:00");
        // date-only defaults to midnight, and the T separator is accepted
        assert_eq!(parse_datetime("2024-01-15").unwrap() % 86_400, 0);
        assert_eq!(
            parse_datetime("2024-01-15T01:00:00Z").unwrap(),
            parse_datetime("2024-01-15 01:00:00").unwrap()
        );
        assert!(parse_datetime("not a date").is_err());
        // negative (pre-epoch) datetimes format correctly
        assert_eq!(fmt_datetime(-1), "1969-12-31 23:59:59");
    }

    #[test]
    fn numeric_comparison_crosses_representations() {
        assert_eq!(Value::UInt(5), Value::Int(5));
        assert_eq!(Value::Int(5), Value::Float(5.0));
        assert_eq!(Value::Bool(true), Value::UInt(1));
        assert!(Value::Int(-1) < Value::UInt(0));
        assert!(Value::Float(1.5) > Value::Int(1));
    }

    #[test]
    fn nulls_sort_first_and_nan_last() {
        let mut v = vec![
            Value::Float(f64::NAN),
            Value::Int(3),
            Value::Null,
            Value::Int(-2),
        ];
        v.sort();
        assert!(v[0].is_null());
        assert_eq!(v[1], Value::Int(-2));
        assert_eq!(v[2], Value::Int(3));
        assert!(matches!(v[3], Value::Float(f) if f.is_nan()));
    }

    #[test]
    fn hash_agrees_with_eq() {
        assert_eq!(h(&Value::UInt(7)), h(&Value::Int(7)));
        assert_eq!(h(&Value::Int(7)), h(&Value::Float(7.0)));
        assert_eq!(h(&Value::Float(0.0)), h(&Value::Float(-0.0)));
        assert_eq!(Value::Float(0.0), Value::Float(-0.0));
        assert_ne!(h(&Value::str("7")), h(&Value::Int(7)));
    }

    #[test]
    fn cast_range_checks() {
        assert_eq!(Value::Int(200).cast_to(&DataType::UInt8).unwrap(), Value::UInt(200));
        assert!(Value::Int(300).cast_to(&DataType::UInt8).is_err());
        assert!(Value::Int(-1).cast_to(&DataType::UInt8).is_err());
        assert!(Value::Null.cast_to(&DataType::UInt8).is_err());
        assert!(Value::Null
            .cast_to(&DataType::Nullable(Box::new(DataType::UInt8)))
            .is_ok());
        assert_eq!(
            Value::Float(3.9).cast_to(&DataType::Int32).unwrap(),
            Value::Int(3)
        );
        assert_eq!(
            Value::str("42").cast_to(&DataType::Int32).unwrap(),
            Value::Int(42)
        );
        assert_eq!(
            Value::Int(42).cast_to(&DataType::String).unwrap(),
            Value::str("42")
        );
    }

    /// A string literal casts to an integer target, so it must cast to a float
    /// target too -- `DEFAULT '1.5'` on Float64 used to be an error while
    /// `DEFAULT '42'` on Int64 was fine.
    #[test]
    fn strings_cast_to_float_targets_like_they_do_to_integer_ones() {
        let f = |s: &str, t: DataType| Value::str(s).cast_to(&t);
        assert!(f("1.5", DataType::Float64).unwrap().eq_exact(&Value::Float(1.5)));
        assert!(f("  -2e3 ", DataType::Float64).unwrap().eq_exact(&Value::Float(-2000.0)));
        // Float32 still narrows through f32, same as any other source variant.
        assert!(f("0.1", DataType::Float32)
            .unwrap()
            .eq_exact(&Value::Float(0.1f32 as f64)));
        assert!(f("42", DataType::Float64).unwrap().eq_exact(&Value::Float(42.0)));
        assert!(f("not a number", DataType::Float64).is_err());
        assert!(f("", DataType::Float64).is_err());
        // `as_f64` itself is untouched: it backs Ord/Hash, where a numeric
        // string turning into a number would merge '1' and 1 into one key.
        assert_eq!(Value::str("1").as_f64(), None);
        assert_ne!(Value::str("1"), Value::Int(1));
    }

    #[test]
    fn eq_collapses_variants_and_eq_exact_does_not() {
        // The collapse, restated as a test so the trap is discoverable from
        // here and not only from the comment on the `PartialEq` impl.
        let zeros = [
            Value::UInt(0),
            Value::Int(0),
            Value::Float(0.0),
            Value::Date(0),
            Value::Bool(false),
        ];
        for a in &zeros {
            for b in &zeros {
                assert_eq!(a, b, "{} vs {}", a.variant(), b.variant());
                assert_eq!(
                    a.eq_exact(b),
                    a.variant() == b.variant(),
                    "{} vs {}",
                    a.variant(),
                    b.variant()
                );
            }
        }
        // Same variant, different value: still not equal.
        assert!(!Value::Int(1).eq_exact(&Value::Int(2)));
        assert!(Value::Int(1).same_variant(&Value::Int(2)));
        assert!(!Value::Null.same_variant(&Value::Int(0)));
        assert_eq!(Value::str("x").variant(), "Str");
        assert_eq!(Value::DateTime(0).variant(), "DateTime");
    }

    // ------------------------------------------------------------- decimals

    fn dec(u: i64, s: u8) -> Value {
        Value::Decimal(u, s)
    }

    /// The variant must not grow `Value`: it is what every result cell, group
    /// key and sort key is, and a fourth word would show up in the aggregation
    /// table's resident set.
    #[test]
    fn value_stays_three_words() {
        assert_eq!(std::mem::size_of::<Value>(), 24);
    }

    #[test]
    fn decimals_render_at_their_declared_scale() {
        // Trailing zeros are kept: the scale is a property of the column, so
        // dropping it would make two rows of one column disagree about how much
        // precision was stored.
        assert_eq!(dec(1234, 2).render_plain(), "12.34");
        assert_eq!(dec(1200, 2).render_plain(), "12.00");
        assert_eq!(dec(5, 2).render_plain(), "0.05");
        assert_eq!(dec(-5, 2).render_plain(), "-0.05");
        assert_eq!(dec(-1234, 2).render_plain(), "-12.34");
        assert_eq!(dec(0, 4).render_plain(), "0.0000");
        assert_eq!(dec(42, 0).render_plain(), "42");
        assert_eq!(dec(1, 18).render_plain(), "0.000000000000000001");
        // A hand-built lane really can hold i64::MIN, and `.abs()` on it panics.
        assert_eq!(dec(i64::MIN, 2).render_plain(), "-92233720368547758.08");
        // Display adds nothing for a number, same as every other numeric.
        assert_eq!(dec(1234, 2).to_string(), "12.34");
    }

    /// The headline case. `0.1` has no `f64`, so the string has to become units
    /// without ever touching a double.
    #[test]
    fn string_to_decimal_is_exact_not_float_rounded() {
        let c = |s: &str, sc: u8| Value::str(s).cast_to(&DataType::Decimal64(sc)).unwrap();
        assert!(c("0.1", 2).eq_exact(&dec(10, 2)));
        assert!(c("0.2", 2).eq_exact(&dec(20, 2)));
        assert!(c("12.34", 2).eq_exact(&dec(1234, 2)));
        assert!(c("-12.34", 2).eq_exact(&dec(-1234, 2)));
        assert!(c(" 7 ", 2).eq_exact(&dec(700, 2)));
        assert!(c("+1.5", 4).eq_exact(&dec(15000, 4)));
        // 0.1 * 100 in f64 is 10.000000000000002; truncating that loses a cent,
        // which is the entire reason this parser is hand-rolled.
        assert_eq!(c("0.07", 2).decimal_parts(), Some((7, 2)));
        assert_eq!(c("1.15", 2).decimal_parts(), Some((115, 2)));
        assert_eq!(c("8.165", 3).decimal_parts(), Some((8165, 3)));
        // Exponents survive, because `toString` of a float can produce one.
        assert!(c("1.5e3", 2).eq_exact(&dec(150_000, 2)));
        assert!(c("15e-1", 2).eq_exact(&dec(150, 2)));
        assert!(c("1e-40", 2).eq_exact(&dec(0, 2)));
        // Rounding on the way in is half away from zero -- the invoice rule.
        assert!(c("1.005", 2).eq_exact(&dec(101, 2)));
        assert!(c("-1.005", 2).eq_exact(&dec(-101, 2)));
        assert!(c("1.004", 2).eq_exact(&dec(100, 2)));
        for bad in ["", "x", "1.2.3", "1,5", "--1", "."] {
            assert!(Value::str(bad).cast_to(&DataType::Decimal64(2)).is_err(), "{bad}");
        }
    }

    #[test]
    fn decimal_casts_round_trip_through_every_family() {
        let d = DataType::Decimal64(2);
        // int -> decimal is exact at any scale
        assert!(Value::Int(5).cast_to(&d).unwrap().eq_exact(&dec(500, 2)));
        assert!(Value::UInt(5).cast_to(&d).unwrap().eq_exact(&dec(500, 2)));
        assert!(Value::Bool(true).cast_to(&d).unwrap().eq_exact(&dec(100, 2)));
        // decimal -> int truncates toward zero, exactly like the float cast
        assert_eq!(dec(1299, 2).cast_to(&DataType::Int64).unwrap(), Value::Int(12));
        assert_eq!(dec(-1299, 2).cast_to(&DataType::Int64).unwrap(), Value::Int(-12));
        assert_eq!(dec(1299, 2).cast_to(&DataType::UInt8).unwrap(), Value::UInt(12));
        // ...and the target's range check still applies to the *number*
        assert!(dec(99_900, 2).cast_to(&DataType::UInt8).is_err());
        // decimal -> string is exact, decimal -> float is not (and says so by
        // being a float)
        assert_eq!(dec(1234, 2).cast_to(&DataType::String).unwrap(), Value::str("12.34"));
        assert_eq!(dec(1234, 2).cast_to(&DataType::Float64).unwrap(), Value::Float(12.34));
        // float -> decimal goes via the float's own decimal spelling, so 0.1
        // lands on 10 units and not on 9.
        assert!(Value::Float(0.1).cast_to(&d).unwrap().eq_exact(&dec(10, 2)));
        // Half away from zero, and *from the shortest spelling of the double*:
        // -2.675 rounds to -2.68 even though the stored double is fractionally
        // below the halfway point, because that is the number the user wrote and
        // what Postgres answers. Ties go away from zero here and not to even, so
        // this agrees with the string cast keystroke for keystroke.
        assert!(Value::Float(-2.675).cast_to(&d).unwrap().eq_exact(&dec(-268, 2)));
        assert!(Value::Float(2.5).cast_to(&DataType::Decimal64(0)).unwrap().eq_exact(&dec(3, 0)));
        assert!(Value::Float(-2.5).cast_to(&DataType::Decimal64(0)).unwrap().eq_exact(&dec(-3, 0)));
        assert!(Value::Float(1e30).cast_to(&d).is_err(), "does not fit 18 digits");
        assert!(Value::Float(f64::NAN).cast_to(&d).is_err());
        assert!(Value::Float(f64::INFINITY).cast_to(&d).is_err());
        // decimal -> decimal rescales, rounding half away from zero
        assert!(dec(1234, 2).cast_to(&DataType::Decimal64(4)).unwrap().eq_exact(&dec(123_400, 4)));
        assert!(dec(1235, 3).cast_to(&d).unwrap().eq_exact(&dec(124, 2)));
        assert!(dec(-1235, 3).cast_to(&d).unwrap().eq_exact(&dec(-124, 2)));
        assert!(dec(1234, 3).cast_to(&d).unwrap().eq_exact(&dec(123, 2)));
    }

    /// 18 digits is the promise; the 19th must be refused, not wrapped.
    #[test]
    fn decimal_range_is_enforced_at_the_edges() {
        let max = DECIMAL_MAX_UNITS as i64; // 18 nines
        assert!(Value::decimal(max as i128, 0).is_ok());
        assert!(Value::decimal(max as i128 + 1, 0).is_err());
        assert!(Value::decimal(-(max as i128), 0).is_ok());
        assert!(Value::decimal(-(max as i128) - 1, 0).is_err());
        // Widening a value that no longer fits is the same refusal.
        assert!(dec(max, 0).cast_to(&DataType::Decimal64(2)).is_err());
        assert!(Value::str("1000000000000000000").cast_to(&DataType::Decimal64(0)).is_err());
        assert!(Value::str("999999999999999999").cast_to(&DataType::Decimal64(0)).is_ok());
        // ...and the message names the limit rather than just failing.
        let e = dec(max, 0).cast_to(&DataType::Decimal64(2)).unwrap_err().to_string();
        assert!(e.contains("18"), "{e}");
    }

    /// The exactness that makes the type worth having: comparison never routes
    /// through a double, so it never gets 0.1 + 0.2 vs 0.3 wrong.
    #[test]
    fn decimal_comparison_is_exact_across_scales_and_families() {
        // Same number, different scales.
        assert_eq!(dec(1234, 2), dec(12_340, 3));
        assert_eq!(dec(1234, 2), dec(1_234_000, 5));
        assert_eq!(dec(0, 2), dec(0, 7));
        assert!(dec(1235, 3) < dec(124, 2));
        assert!(dec(-1, 2) < dec(0, 9));
        // Against integers: the lane is 500 but the value is 5.
        assert_eq!(dec(500, 2), Value::Int(5));
        assert_eq!(dec(500, 2), Value::UInt(5));
        assert_ne!(dec(500, 2), Value::Int(500));
        assert!(dec(505, 2) > Value::Int(5));
        assert!(dec(495, 2) < Value::Int(5));
        assert!(Value::Int(-1) < dec(0, 4));
        // Against floats: the best either side can do, matching how the engine
        // already compares Int against Float.
        assert_eq!(dec(1234, 2), Value::Float(12.34));
        assert!(dec(1234, 2) < Value::Float(12.35));
        // Nulls still sort first and strings still sort last.
        assert!(Value::Null < dec(0, 0));
        assert!(dec(i64::MAX, 0) < Value::str(""));
        // ORDER BY over a mixed bag is a total order with no surprises.
        let mut v = vec![dec(300, 2), Value::Int(1), dec(-1, 2), Value::Null, Value::Float(2.5)];
        v.sort();
        assert!(v[0].is_null());
        assert_eq!(
            v[1..].iter().map(|x| x.render_plain()).collect::<Vec<_>>(),
            ["-0.01", "1", "2.5", "3.00"]
        );
    }

    /// `Hash` must agree with `Eq` or GROUP BY splits keys that compare equal.
    /// A decimal naming a whole number *is* that integer to `Ord`, so it has to
    /// hash as one.
    #[test]
    fn decimal_hash_agrees_with_its_equality() {
        assert_eq!(h(&dec(500, 2)), h(&Value::Int(5)));
        assert_eq!(h(&dec(500, 2)), h(&Value::UInt(5)));
        assert_eq!(h(&dec(500, 2)), h(&Value::Float(5.0)));
        assert_eq!(h(&dec(0, 9)), h(&Value::Int(0)));
        assert_eq!(h(&dec(-700, 2)), h(&Value::Int(-7)));
        // Fractional ones hash through the (correctly rounded) double, which is
        // what makes two spellings of the same number agree.
        assert_eq!(h(&dec(1234, 2)), h(&dec(12_340, 3)));
        assert_eq!(h(&dec(1234, 2)), h(&Value::Float(12.34)));
        assert_ne!(h(&dec(1234, 2)), h(&dec(1235, 2)));
    }

    /// `as_i64` hands back the *lane*, not the number -- `Column::constant`,
    /// `push_value` and `to_lane_phys` all build an I64 lane straight out of it,
    /// so a helpfully descaled answer here would write the wrong number to disk.
    #[test]
    fn decimal_accessors_split_lane_from_value() {
        assert_eq!(dec(1234, 2).as_i64(), Some(1234));
        assert_eq!(dec(1234, 2).as_u64(), Some(1234));
        assert_eq!(dec(-1, 2).as_u64(), None);
        assert_eq!(dec(1234, 2).as_f64(), Some(12.34));
        assert_eq!(dec(1234, 2).decimal_parts(), Some((1234, 2)));
        assert_eq!(Value::Int(1).decimal_parts(), None);
        assert_eq!(dec(1234, 2).data_type(), DataType::Decimal64(2));
        assert_eq!(dec(1234, 2).physical(), Some(PhysicalType::I64));
        assert_eq!(dec(1234, 2).variant(), "Decimal");
        assert!(dec(1, 9).truthy());
        assert!(!dec(0, 9).truthy());
        // Two scales, one number, one double -- this is what keeps `Hash`
        // consistent with the exact `Ord`.
        assert_eq!(dec(1234, 2).as_f64(), dec(12_340, 3).as_f64());
    }

    /// The lane a value occupies depends on the *column* it is going into, and
    /// a decimal is the only kind where that is not a formality.
    #[test]
    fn decimal_lanes_are_resolved_against_the_target_column() {
        use crate::common::i64_to_lane;
        let d2 = DataType::Decimal64(2);
        // An integer probing a decimal key column has to scale up...
        assert_eq!(Value::Int(2).to_lane(&d2).unwrap(), i64_to_lane(200));
        // ...a decimal probing its own column passes through...
        assert_eq!(dec(250, 2).to_lane(&d2).unwrap(), i64_to_lane(250));
        // ...and one at another scale is rescaled, not reinterpreted.
        assert_eq!(dec(2500, 3).to_lane(&d2).unwrap(), i64_to_lane(250));
        // A decimal probing a *plain* integer key is exact or nothing: `id =
        // 1.00` names key 1, `id = 1.50` names no key at all and must say so
        // rather than truncate into a probe for 1.
        assert_eq!(dec(100, 2).to_lane(&DataType::Int64).unwrap(), i64_to_lane(1));
        assert_eq!(dec(-300, 2).to_lane(&DataType::Int64).unwrap(), i64_to_lane(-3));
        assert!(dec(150, 2).to_lane(&DataType::Int64).is_err());
        assert!(dec(175, 2).to_lane(&DataType::UInt64).is_err());
    }

    #[test]
    fn display_quotes_and_escapes() {
        assert_eq!(Value::str("it's").to_string(), "'it''s'");
        assert_eq!(Value::str("it's").render_plain(), "it's");
        assert_eq!(Value::Null.to_string(), "NULL");
        assert_eq!(Value::Float(1.0).to_string(), "1");
        assert_eq!(Value::Date(0).to_string(), "1970-01-01");
    }

    #[test]
    fn truthiness_follows_sql() {
        assert!(!Value::Null.truthy());
        assert!(!Value::UInt(0).truthy());
        assert!(Value::UInt(1).truthy());
        assert!(!Value::str("").truthy());
        assert!(Value::str("x").truthy());
    }
}
