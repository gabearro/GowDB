//! The vectorized scalar function library.
//!
//! Every entry is a `(ret, eval)` pair in one `'static` table. `ret` is the
//! *only* gate: it validates arity and argument types at bind time and yields
//! the result type. `eval` may therefore assume it was handed exactly what
//! `ret` accepted, which is what keeps the inner loops free of per-row type
//! dispatch.
//!
//! ## Why eval re-derives its own output type
//!
//! Each `eval` calls the matching `ret` on its arguments' declared types
//! instead of taking the type as a parameter (the frozen signature has no room
//! for one). That costs a handful of `DataType` clones per *block*, not per
//! row, and it makes "the column `eval` returns is exactly the type `ret`
//! promised" a structural property rather than a convention two functions have
//! to remember to agree on.
//!
//! ## Null semantics
//!
//! The default is strict propagation: the output mask is the union of the
//! input masks, computed once per block with `BitSet::union_with` rather than
//! per row. The exceptions are the functions whose whole purpose is to observe
//! nulls (`isNull`, `ifNull`, `coalesce`, `nullIf`, `assumeNotNull`), the
//! three-valued `and`/`or`, and `if` — where a NULL condition selects the
//! `else` branch (SQL `CASE` semantics) instead of poisoning the row.
//!
//! ## Documented deviations from ClickHouse
//!
//! * **Division by zero returns NULL.** ClickHouse raises for `/` and returns
//!   0 for `intDiv`. Neither is satisfying in a vectorized engine: raising
//!   throws away 8191 good rows for one bad one, and 0 silently corrupts an
//!   average. `divide`, `intDiv` and `modulo` are therefore typed
//!   `Nullable(...)` unconditionally and null the offending rows.
//! * **String → number conversions that fail to parse return NULL** rather
//!   than raising, for the same reason. `toUInt64('x')` is `NULL`, and any
//!   `toXxx` with a `String` argument is typed `Nullable`.
//! * **Number → narrower-number conversions truncate** (`as` casts, which
//!   saturate for float→int in Rust) instead of raising on overflow.
//! * **`toDate` of a bare number reads it as a *day* count**, not a Unix
//!   timestamp. ClickHouse decides from the argument's declared integer width
//!   (`UInt16` → days, `UInt32` → seconds); every integer literal binds as
//!   `Int64` here, so there is nothing to switch on. Days is the choice that
//!   makes `toDate` the exact inverse of `toDateTime` on a `Date` and makes
//!   `toDate` idempotent. **This is the one conversion family that raises
//!   rather than truncating**: see [`date_lane_err`].
//! * **`substring`, `position`, `reverse`, `repeat` and the pad functions are
//!   character-based**, not byte-based. ClickHouse has separate `...UTF8`
//!   variants; we cannot afford the byte-based ones because our string payload
//!   is `Arc<str>` and slicing mid-codepoint is not representable. The `UTF8`
//!   spellings are registered as aliases of the same implementation.
//! * **`splitByChar(sep, s)` returns only the first element.** There is no
//!   Array type in this engine, so returning the full split is not
//!   expressible. The signature and argument order match ClickHouse so that a
//!   query written against it parses; only the arity of the result differs.
//! * **`round` always returns `Float64`** (ClickHouse keeps integer input
//!   integral). It does use ClickHouse's banker's rounding, not Rust's
//!   round-half-away-from-zero. **This is also the one place a `Decimal64`
//!   loses exactness**, and it is a signature problem rather than an oversight:
//!   `round(x, 2)` would have to return `Decimal64(2)`, but `ret` is
//!   `fn(&[DataType])` and cannot see the literal `2`. Use
//!   `CAST(x AS Decimal(18, 2))`, which is resolved by the planner and *is*
//!   exact (half away from zero).
//!
//! ## Decimals
//!
//! `Decimal64(s)` is an `I64` lane holding a count of `10^-s` units, so every
//! function here that touches one obeys the same rule: **the scale is resolved
//! once per block from the declared type, and the inner loop only sees lanes
//! that are already commensurable.** `+`, `-`, `*`, `/`, `%` and `DIV` have
//! exact paths (`dec_arith`, `dec_divide`, `dec_int_binop`) that error on
//! overflow instead of wrapping -- the one place this module refuses to follow
//! its own wrapping-integer convention, because a wrapped price is the exact
//! failure the type was added to prevent. Everything that reads a decimal as a
//! *number* rather than a lane (`toFloat64`, `toInt64`, `toString`, `toDate`)
//! descales first, once, outside the loop, through [`f64_vec`]/[`i64_vec`].
//!
//! The payoff is that a decimal column costs what an `Int64` column costs.
//! Measured over 2M rows, A/B interleaved best-of-7 per side, `Decimal64(2)`
//! against `Int64` holding the identical lanes: `sum` 1.02x, `min`/`max` 1.03x,
//! column-to-column `>` 0.99x -- all inside this machine's noise. The one
//! outlier is `WHERE price > 5000.00` at 1.08x, and it is not scale handling in
//! the loop: the literal lexes as a `Float64` (src/sql/lexer.rs), so the
//! comparison descales the column with one FDIV per row. A decimal literal
//! would put that case on the same lane path as the rest and back at 1.0x.
//! * **`match` (regex) is not implemented.** A regex engine is out of scope
//!   for a zero-dependency crate; `like`/`ilike` cover the wildcard cases.
//! * **`cast` is not registered.** The frozen `ret: fn(&[DataType])` cannot
//!   see the literal naming the target type, so `CAST` has to be resolved by
//!   the planner into the concrete `toXxx` function.
//! * **`now()`/`today()` are evaluated once per block**, not once per row, and
//!   `rand()` is a deterministic splitmix over a process-global counter — see
//!   the note on [`e_rand`].

use super::ScalarFn;
use crate::common::{hash_bytes, hash_key, splitmix64, BitSet, Error, Result};
use crate::types::datatype::MAX_DECIMAL_PRECISION;
use crate::types::value::{decimal_rescale, DECIMAL_MAX_UNITS, POW10};
use crate::types::{
    civil_from_days, days_from_civil, fmt_date, fmt_datetime, parse_datetime, Column, ColumnData,
    DataType, PhysicalType, Value,
};
use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Ceiling on any single constructed string, so `repeat('x', 1e18)` fails
/// loudly instead of taking the machine down.
const MAX_STR: usize = 1 << 24;

// ===========================================================================
// shared helpers
// ===========================================================================

/// Union of the argument null masks. Returns `None` when nothing is null, so
/// the common case costs one pointer check per argument and allocates nothing.
fn nulls_of(args: &[Column], _rows: usize) -> Option<BitSet> {
    let mut acc: Option<BitSet> = None;
    for a in args {
        if let Some(n) = &a.nulls {
            match &mut acc {
                None => acc = Some(n.clone()),
                Some(m) => m.union_with(n),
            }
        }
    }
    acc.filter(|m| !m.is_empty())
}

/// Assemble the result column, widening the declared type to `Nullable` iff
/// the mask actually has a bit set.
fn build(ty: DataType, data: ColumnData, nulls: Option<BitSet>) -> Column {
    match nulls {
        Some(n) if !n.is_empty() => Column { ty: ty.to_nullable(), data, nulls: Some(n) },
        _ => Column::new(ty, data),
    }
}

/// Argument types, for `eval` to feed back into its own `ret`.
#[inline]
fn tys(args: &[Column]) -> Vec<DataType> {
    args.iter().map(|c| c.ty.clone()).collect()
}

/// Nullability is contagious across every strict function.
fn nullable_like(args: &[DataType], base: DataType) -> DataType {
    if args.iter().any(|t| t.is_nullable()) {
        base.to_nullable()
    } else {
        base
    }
}

fn arity(name: &str, args: &[DataType], lo: usize, hi: usize) -> Result<()> {
    if args.len() < lo || args.len() > hi {
        let want = if hi == usize::MAX {
            format!("at least {lo}")
        } else if lo == hi {
            format!("exactly {lo}")
        } else {
            format!("{lo} to {hi}")
        };
        return Err(Error::bind(format!(
            "{name} takes {want} arguments, got {}",
            args.len()
        )));
    }
    Ok(())
}

fn want_num(name: &str, t: &DataType, pos: usize) -> Result<()> {
    if t.is_numeric() {
        Ok(())
    } else {
        Err(Error::bind(format!(
            "{name}: argument {} must be numeric, got {t}",
            pos + 1
        )))
    }
}

fn want_int(name: &str, t: &DataType, pos: usize) -> Result<()> {
    if t.is_integer() {
        Ok(())
    } else {
        Err(Error::bind(format!(
            "{name}: argument {} must be an integer, got {t}",
            pos + 1
        )))
    }
}

fn want_str(name: &str, t: &DataType, pos: usize) -> Result<()> {
    if t.is_string() {
        Ok(())
    } else {
        Err(Error::bind(format!(
            "{name}: argument {} must be a String, got {t}",
            pos + 1
        )))
    }
}

fn want_temporal(name: &str, t: &DataType, pos: usize) -> Result<()> {
    if t.is_temporal() {
        Ok(())
    } else {
        Err(Error::bind(format!(
            "{name}: argument {} must be Date or DateTime, got {t}",
            pos + 1
        )))
    }
}

/// `Column::to_i64_vec`/`to_f64_vec` have no unsigned sibling.
///
/// A decimal is truncated toward zero here rather than handed over as its unit
/// count, matching [`i64_vec`]: every caller of this one is asking for a whole
/// number.
fn to_u64_vec(c: &Column) -> Result<Vec<u64>> {
    if let Some(s) = c.ty.decimal_scale() {
        let p = crate::types::value::POW10[s as usize];
        return Ok(c.as_i64()?.iter().map(|&x| (x as i128 / p).max(0) as u64).collect());
    }
    Ok(match &c.data {
        ColumnData::U64(v) => v.clone(),
        ColumnData::I64(v) => v.iter().map(|&x| x as u64).collect(),
        ColumnData::F64(v) => v.iter().map(|&x| x as u64).collect(),
        ColumnData::Str(_) => {
            return Err(Error::exec("cannot use a String column as an integer"))
        }
    })
}

// =========================================================================
// decimals
// =========================================================================
// `Decimal64(s)` is an `I64` lane holding a count of `10^-s` units, so every
// decimal-aware function here obeys one rule: **the scale is a property of the
// type, resolved once per block, and the inner loop only ever sees `i64`s that
// are already commensurable.** Nothing below multiplies by a power of ten
// inside a per-row loop.

/// The scale a column's lane is denominated in. A plain integer counts as 0,
/// which is exactly right: an `Int64` *is* a decimal with no fractional digits,
/// and saying so collapses the mixed int/decimal cases into the same code.
#[inline]
fn scale_of(t: &DataType) -> u8 {
    t.decimal_scale().unwrap_or(0)
}

/// `10^k` for `k` up to 36 -- what a decimal divide's numerator shift can reach
/// (`rs + os - ls`, each term at most 18). Past that no `i128` product survives
/// anyway, so `None` is the honest answer rather than a wider table.
#[inline]
fn pow10(k: u32) -> Option<i128> {
    match k {
        0..=18 => Some(POW10[k as usize]),
        19..=36 => POW10[18].checked_mul(POW10[(k - 18) as usize]),
        _ => None,
    }
}

#[cold]
fn dec_overflow(who: &str, scale: u8) -> Error {
    Error::exec(format!(
        "{who}: result does not fit Decimal64({scale}) -- more than 18 significant digits"
    ))
}

/// Move a whole lane buffer from `from` scale to `to`, in place.
///
/// In place because the caller already owns the `Vec` that `to_i64_vec` handed
/// back, so a second buffer would be pure waste. The multiplier is hoisted out
/// of the loop; the same-scale case (the overwhelmingly common one, since both
/// operands of a real query share a column type) returns without touching
/// memory at all.
fn rescale_slice(v: &mut [i64], from: u8, to: u8, who: &str) -> Result<()> {
    if from == to {
        return Ok(());
    }
    if to > from {
        let f = POW10[(to - from) as usize] as i64;
        for x in v.iter_mut() {
            *x = x.checked_mul(f).ok_or_else(|| dec_overflow(who, to))?;
        }
    } else {
        for x in v.iter_mut() {
            *x = decimal_rescale(*x as i128, from, to)
                .filter(|u| u.abs() <= DECIMAL_MAX_UNITS)
                .ok_or_else(|| dec_overflow(who, to))? as i64;
        }
    }
    Ok(())
}

/// A column's lanes, brought to `to` scale. Integers enter at scale 0.
fn dec_units(c: &Column, to: u8, who: &str) -> Result<Vec<i64>> {
    let mut v = c.to_i64_vec()?;
    rescale_slice(&mut v, scale_of(&c.ty), to, who)?;
    Ok(v)
}

/// A column as `f64`, **descaling a decimal**.
///
/// `Column::to_f64_vec` reads the buffer, and a decimal's buffer is a unit
/// count: `sqrt(price)` on a `Decimal64(2)` holding $1.50 would otherwise take
/// the square root of 150. Every function in this module that wants a decimal as
/// a *number* rather than a lane goes through here instead.
///
/// The divisor is hoisted, so the loop is one FDIV per row and the non-decimal
/// case is `to_f64_vec` unchanged, allocation for allocation.
///
/// `pub(crate)` so that the comparison path in `exec::expr` and the float-shaped
/// accumulators in `exec::functions::agg` share this one definition of "read a
/// decimal as a number": three copies of a `/ 10^s` are three chances for one of
/// them to be forgotten.
pub(crate) fn f64_vec(c: &Column) -> Result<Vec<f64>> {
    match c.ty.decimal_scale() {
        None => c.to_f64_vec(),
        Some(s) => {
            let d = POW10[s as usize] as f64;
            Ok(c.as_i64()?.iter().map(|&x| x as f64 / d).collect())
        }
    }
}

/// The three lane readers, borrowing when the column is already in the
/// representation asked for.
///
/// `to_i64_vec` and friends copy unconditionally, so `a + b` over two `Int64`
/// columns allocated and filled four buffers to produce one result. These
/// return `Cow`, which for the overwhelmingly common same-representation case
/// is a pointer -- the conversion arms are byte-for-byte the ones they replace.
/// Measured on 2.1M rows, A/B interleaved best-of-9: `Int64 a + b` 0.86x
/// (1.09 vs 0.94 ns/row) and `Float64 a + b` 0.80x.
fn i64_lanes(c: &Column) -> Result<Cow<'_, [i64]>> {
    Ok(match &c.data {
        ColumnData::I64(v) => Cow::Borrowed(v),
        _ => Cow::Owned(c.to_i64_vec()?),
    })
}

fn u64_lanes(c: &Column) -> Result<Cow<'_, [u64]>> {
    match &c.data {
        ColumnData::U64(v) if !c.ty.is_decimal() => Ok(Cow::Borrowed(v)),
        _ => Ok(Cow::Owned(to_u64_vec(c)?)),
    }
}

fn f64_lanes(c: &Column) -> Result<Cow<'_, [f64]>> {
    match &c.data {
        ColumnData::F64(v) => Ok(Cow::Borrowed(v)),
        _ => Ok(Cow::Owned(f64_vec(c)?)),
    }
}

/// A column as whole `i64`s, truncating a decimal toward zero.
///
/// The counterpart to [`f64_vec`] for the integer operations (`DIV`, `%`) that
/// mean "the number", as opposed to [`dec_units`], which means "the lane".
pub(crate) fn i64_vec(c: &Column) -> Result<Vec<i64>> {
    match c.ty.decimal_scale() {
        None => c.to_i64_vec(),
        Some(s) => {
            let p = POW10[s as usize];
            Ok(c.as_i64()?.iter().map(|&x| (x as i128 / p) as i64).collect())
        }
    }
}

/// Promote one column into the representation the result needs.
///
/// Takes the whole `DataType` and not just its physical kind, because two
/// decimal columns can share a lane width and still disagree about what a lane
/// *means*: `Decimal64(2)` and `Decimal64(4)` are both `I64`, and gathering
/// between them without a rescale multiplies a row by 100.
fn coerce_data(c: &Column, ty: &DataType) -> Result<ColumnData> {
    if let Some(to) = ty.decimal_scale() {
        return Ok(ColumnData::I64(dec_units(c, to, "coerce")?));
    }
    let from = match c.ty.decimal_scale() {
        // Same representation on both sides: the plain path below is exact.
        None => return coerce_plain(c, ty.base().physical()),
        Some(s) => s,
    };
    // A decimal flowing into something that is *not* a decimal -- a float won
    // the promotion, or a CAST asked for an integer. Descale here, once, rather
    // than letting `to_f64_vec` hand the raw unit count downstream.
    let p = POW10[from as usize];
    let v = c.as_i64()?;
    Ok(match ty.base().physical() {
        PhysicalType::F64 => {
            let d = p as f64;
            ColumnData::F64(v.iter().map(|&x| x as f64 / d).collect())
        }
        PhysicalType::I64 => ColumnData::I64(v.iter().map(|&x| (x as i128 / p) as i64).collect()),
        PhysicalType::U64 => ColumnData::U64(
            v.iter().map(|&x| (x as i128 / p).max(0) as u64).collect(),
        ),
        PhysicalType::Str => ColumnData::Str(
            v.iter()
                .map(|&x| Arc::from(Value::Decimal(x, from).render_plain()))
                .collect(),
        ),
    })
}

/// A decimal column as a plain `Int64` one, truncated toward zero, nulls kept.
///
/// Cold path. The conversions that want a *whole number* out of a numeric column
/// (`toDate`, `toDateTime`) re-enter through this rather than every arm of a
/// six-way match growing a scale parameter for an argument nobody writes.
fn dec_to_int_column(c: &Column) -> Result<Column> {
    let p = POW10[scale_of(&c.ty) as usize];
    let v: Vec<i64> = c.as_i64()?.iter().map(|&x| (x as i128 / p) as i64).collect();
    let ty = if c.ty.is_nullable() { DataType::Int64.to_nullable() } else { DataType::Int64 };
    Ok(Column { ty, data: ColumnData::I64(v), nulls: c.nulls.clone() })
}

fn coerce_plain(c: &Column, p: PhysicalType) -> Result<ColumnData> {
    Ok(match p {
        PhysicalType::U64 => ColumnData::U64(to_u64_vec(c)?),
        PhysicalType::I64 => ColumnData::I64(c.to_i64_vec()?),
        PhysicalType::F64 => ColumnData::F64(c.to_f64_vec()?),
        PhysicalType::Str => ColumnData::Str(c.as_str()?.to_vec()),
    })
}

/// Row-wise gather: `pick[i]` names which argument supplies row `i`, and
/// `u32::MAX` means "NULL". The output inherits the selected source's own null
/// bit, which is what makes `if`, `ifNull` and `coalesce` fall out of one
/// primitive.
fn gather(
    args: &[Column],
    datas: &[ColumnData],
    pick: &[u32],
    rows: usize,
    p: PhysicalType,
) -> (ColumnData, BitSet) {
    let mut nulls = BitSet::new();
    macro_rules! run {
        ($variant:ident, $zero:expr) => {{
            let src: Vec<&[_]> = datas
                .iter()
                .map(|d| match d {
                    ColumnData::$variant(v) => v.as_slice(),
                    _ => unreachable!("gather sources are pre-coerced"),
                })
                .collect();
            let mut out = Vec::with_capacity(rows);
            for i in 0..rows {
                let k = pick[i];
                if k == u32::MAX || args[k as usize].is_null(i) {
                    nulls.set(i);
                    out.push($zero);
                } else {
                    out.push(src[k as usize][i].clone());
                }
            }
            ColumnData::$variant(out)
        }};
    }
    let data = match p {
        PhysicalType::U64 => run!(U64, 0u64),
        PhysicalType::I64 => run!(I64, 0i64),
        PhysicalType::F64 => run!(F64, 0.0f64),
        PhysicalType::Str => run!(Str, Arc::<str>::from("")),
    };
    (data, nulls)
}

/// SQL truthiness per row, without materializing `Value`s.
fn truth_vec(c: &Column, rows: usize) -> Result<Vec<bool>> {
    Ok(match &c.data {
        ColumnData::U64(v) => (0..rows).map(|i| v[i] != 0).collect(),
        ColumnData::I64(v) => (0..rows).map(|i| v[i] != 0).collect(),
        ColumnData::F64(v) => (0..rows).map(|i| v[i] != 0.0).collect(),
        ColumnData::Str(v) => (0..rows).map(|i| !v[i].is_empty()).collect(),
    })
}

/// The widest type of a numeric family: everything collapses to one of three,
/// except a decimal, which keeps its scale -- widening it to `Int64` would
/// reinterpret the unit count as the number, so `abs(-1.50)` would answer 150.
fn num_wide(t: &DataType) -> DataType {
    match t.base() {
        DataType::Decimal64(s) => DataType::Decimal64(*s),
        b => match b.physical() {
            PhysicalType::F64 => DataType::Float64,
            PhysicalType::I64 => DataType::Int64,
            _ => DataType::UInt64,
        },
    }
}

// ===========================================================================
// math
// ===========================================================================

fn r_abs(a: &[DataType]) -> Result<DataType> {
    arity("abs", a, 1, 1)?;
    want_num("abs", &a[0], 0)?;
    Ok(nullable_like(a, num_wide(&a[0])))
}

fn e_abs(args: &[Column], rows: usize) -> Result<Column> {
    let ty = r_abs(&tys(args))?;
    // abs(i64::MIN) has no i64 representation; wrap rather than panic, which
    // is what every other integer op in the engine does.
    let data = match ty.base().physical() {
        PhysicalType::F64 => {
            let v = f64_vec(&args[0])?;
            ColumnData::F64((0..rows).map(|i| v[i].abs()).collect())
        }
        PhysicalType::I64 => {
            let v = args[0].to_i64_vec()?;
            ColumnData::I64((0..rows).map(|i| v[i].wrapping_abs()).collect())
        }
        _ => ColumnData::U64(to_u64_vec(&args[0])?),
    };
    Ok(build(ty.strip_nullable(), data, nulls_of(args, rows)))
}

fn r_negate(a: &[DataType]) -> Result<DataType> {
    arity("negate", a, 1, 1)?;
    want_num("negate", &a[0], 0)?;
    // negating an unsigned value has to widen to signed; a decimal is already
    // signed and keeps its scale (the unit count negates, the scale does not)
    let base = match a[0].base() {
        DataType::Decimal64(s) => DataType::Decimal64(*s),
        _ if a[0].is_float() => DataType::Float64,
        _ => DataType::Int64,
    };
    Ok(nullable_like(a, base))
}

fn e_negate(args: &[Column], rows: usize) -> Result<Column> {
    let ty = r_negate(&tys(args))?;
    let data = if ty.is_float() {
        let v = f64_vec(&args[0])?;
        ColumnData::F64((0..rows).map(|i| -v[i]).collect())
    } else {
        let v = args[0].to_i64_vec()?;
        ColumnData::I64((0..rows).map(|i| v[i].wrapping_neg()).collect())
    };
    Ok(build(ty.strip_nullable(), data, nulls_of(args, rows)))
}

/// Shared typing for `+`, `-` and `*`. Date arithmetic is allowed for the
/// additive operators only, and Date − Date degrades to a plain day count.
fn arith_ty(name: &str, a: &[DataType], additive: bool) -> Result<DataType> {
    arity(name, a, 2, 2)?;
    for (i, t) in a.iter().enumerate() {
        let ok = t.is_numeric() || (additive && t.is_temporal());
        if !ok {
            return Err(Error::bind(format!(
                "{name}: argument {} must be numeric, got {t}",
                i + 1
            )));
        }
    }
    let (x, y) = (a[0].base(), a[1].base());
    let base = if x.is_temporal() && y.is_temporal() {
        if name == "minus" {
            DataType::Int64
        } else {
            return Err(Error::bind(format!("{name}: cannot combine {x} and {y}")));
        }
    } else if !additive && !x.is_float() && !y.is_float() && (x.is_decimal() || y.is_decimal()) {
        // Multiplication is the one arithmetic op whose result scale is *not*
        // `promote`'s: 1.50 * 1.50 is 2.2500, four digits from two twos, so the
        // scales add. `promote` cannot express that -- it only ever unifies -- so
        // this is the one place `arith_ty` overrides it.
        //
        // Overflowing 18 digits of scale is refused here, at bind time, rather
        // than per row: `Decimal64(10) * Decimal64(10)` has no representable
        // result at all and should not compile into a plan.
        let s = scale_of(x) as u32 + scale_of(y) as u32;
        if s > MAX_DECIMAL_PRECISION {
            return Err(Error::bind(format!(
                "{name}: {x} * {y} needs scale {s}, over the Decimal64 limit of \
                 {MAX_DECIMAL_PRECISION}"
            )));
        }
        DataType::Decimal64(s as u8)
    } else {
        DataType::promote(x, y)?
    };
    Ok(nullable_like(a, base))
}

fn r_plus(a: &[DataType]) -> Result<DataType> {
    arith_ty("plus", a, true)
}
fn r_minus(a: &[DataType]) -> Result<DataType> {
    arith_ty("minus", a, true)
}
fn r_multiply(a: &[DataType]) -> Result<DataType> {
    arith_ty("multiply", a, false)
}

/// Exact `+`, `-` and `*` on decimals.
///
/// The result scale is already decided by `arith_ty`, so all this does is bring
/// both lanes to the scale the *operands* need and run one checked pass:
///   * `+`/`-`: both sides rescale to the result scale, then add unit counts.
///   * `*`: neither side moves -- the unit counts multiply directly, and the
///     product is already denominated in `10^-(s1+s2)`.
///
/// `i128` for the accumulate, then one range check per row. Checked rather than
/// wrapping, unlike every integer op in this module: a wrapped `Int64` is a
/// documented deviation nobody stores money in, whereas a wrapped price is the
/// exact failure this type was added to remove. **A multiply overflows far
/// earlier than the operands suggest** -- two nine-digit values make eighteen --
/// so the check is not theoretical.
fn dec_arith(args: &[Column], rows: usize, out: u8, op: u8, who: &str) -> Result<Column> {
    let (x, y) = if op == b'*' {
        (args[0].to_i64_vec()?, args[1].to_i64_vec()?)
    } else {
        (dec_units(&args[0], out, who)?, dec_units(&args[1], out, who)?)
    };
    let mut o = Vec::with_capacity(rows);
    // One match on the operator per *block*, three flat loops -- the alternative
    // costs a branch per row for something constant across the whole call.
    macro_rules! run {
        ($f:expr) => {{
            let f = $f;
            for i in 0..rows {
                let v: i128 = f(x[i] as i128, y[i] as i128);
                if v.abs() > DECIMAL_MAX_UNITS {
                    return Err(dec_overflow(who, out));
                }
                o.push(v as i64);
            }
        }};
    }
    match op {
        b'+' => run!(|a: i128, b: i128| a + b),
        b'-' => run!(|a: i128, b: i128| a - b),
        _ => run!(|a: i128, b: i128| a * b),
    }
    Ok(build(DataType::Decimal64(out), ColumnData::I64(o), nulls_of(args, rows)))
}

macro_rules! arith {
    ($ev:ident, $ret:ident, $fop:tt, $iop:ident, $dop:literal, $who:literal) => {
        fn $ev(args: &[Column], rows: usize) -> Result<Column> {
            let ty = $ret(&tys(args))?;
            if let Some(s) = ty.decimal_scale() {
                return dec_arith(args, rows, s, $dop, $who);
            }
            let data = match ty.base().physical() {
                // Reached with a decimal operand only when a float won the
                // promotion (`1.50 + 0.5`); `f64_vec` descales it, where
                // `to_f64_vec` would have added 150.0.
                PhysicalType::F64 => {
                    let (x, y) = (f64_lanes(&args[0])?, f64_lanes(&args[1])?);
                    let (x, y) = (&x[..rows], &y[..rows]);
                    ColumnData::F64(x.iter().zip(y).map(|(a, b)| a $fop b).collect())
                }
                PhysicalType::I64 => {
                    let (x, y) = (i64_lanes(&args[0])?, i64_lanes(&args[1])?);
                    let (x, y) = (&x[..rows], &y[..rows]);
                    ColumnData::I64(x.iter().zip(y).map(|(a, b)| a.$iop(*b)).collect())
                }
                PhysicalType::U64 => {
                    let (x, y) = (u64_lanes(&args[0])?, u64_lanes(&args[1])?);
                    let (x, y) = (&x[..rows], &y[..rows]);
                    ColumnData::U64(x.iter().zip(y).map(|(a, b)| a.$iop(*b)).collect())
                }
                PhysicalType::Str => unreachable!("arithmetic on strings is rejected by ret"),
            };
            Ok(build(ty.strip_nullable(), data, nulls_of(args, rows)))
        }
    };
}

arith!(e_plus, r_plus, +, wrapping_add, b'+', "plus");
arith!(e_minus, r_minus, -, wrapping_sub, b'-', "minus");
arith!(e_multiply, r_multiply, *, wrapping_mul, b'*', "multiply");

/// Fractional digits an exact division keeps.
///
/// ClickHouse hands back the left operand's scale, which makes
/// `sum(cents) / count(*)` on a `Decimal64(2)` column answer to the cent -- the
/// one place you actually wanted more digits. Postgres derives a scale from both
/// precisions. Six is the floor here because it is enough for a unit price, a
/// tax rate or an FX quote, and it is what SQL Server's `decimal` division
/// guarantees.
const DIV_MIN_SCALE: u8 = 6;

/// The scale of `a / b`, or `None` when the answer is a float.
///
/// A float on either side poisons exactness -- there is no exact quotient to
/// name -- so mixed float/decimal division keeps the existing `Float64` result
/// rather than pretending.
fn div_scale(a: &DataType, b: &DataType) -> Option<u8> {
    if (!a.is_decimal() && !b.is_decimal()) || a.is_float() || b.is_float() {
        return None;
    }
    Some(scale_of(a).max(DIV_MIN_SCALE))
}

fn r_divide(a: &[DataType]) -> Result<DataType> {
    arity("divide", a, 2, 2)?;
    want_num("divide", &a[0], 0)?;
    want_num("divide", &a[1], 1)?;
    // always Nullable: /0 yields NULL (see module docs)
    Ok(match div_scale(&a[0], &a[1]) {
        Some(s) => DataType::Decimal64(s),
        None => DataType::Float64,
    }
    .to_nullable())
}

/// `a / b` at scale `out`, exactly.
///
/// `a/b = (au * 10^(sb + out - sa)) / bu`, all in `i128`, rounded half away from
/// zero. The numerator shift is a property of the three scales, so it is a
/// single hoisted constant; the loop is a multiply, a divide and a compare.
///
/// The shift reaches 36 in the worst case, and `10^36 * 10^18` does not fit an
/// `i128` -- hence the checked multiply, which reports rather than wraps.
fn dec_divide(args: &[Column], rows: usize, out: u8) -> Result<Column> {
    let (sa, sb) = (scale_of(&args[0].ty), scale_of(&args[1].ty));
    let (x, y) = (args[0].to_i64_vec()?, args[1].to_i64_vec()?);
    let shift = pow10(sb as u32 + out as u32 - sa as u32)
        .ok_or_else(|| dec_overflow("divide", out))?;
    let mut nulls = nulls_of(args, rows).unwrap_or_default();
    let mut o = Vec::with_capacity(rows);
    for i in 0..rows {
        let d = y[i] as i128;
        if d == 0 {
            nulls.set(i);
            o.push(0);
            continue;
        }
        let n = (x[i] as i128)
            .checked_mul(shift)
            .ok_or_else(|| dec_overflow("divide", out))?;
        let (q, rem) = (n / d, n % d);
        // Half away from zero, sign-symmetric: the quotient's sign is the xor of
        // the operands', which is also what a `+1`/`-1` bump has to follow.
        let v = if rem.unsigned_abs() * 2 >= d.unsigned_abs() {
            q + if (n < 0) != (d < 0) { -1 } else { 1 }
        } else {
            q
        };
        if v.abs() > DECIMAL_MAX_UNITS {
            return Err(dec_overflow("divide", out));
        }
        o.push(v as i64);
    }
    Ok(Column {
        ty: DataType::Decimal64(out).to_nullable(),
        data: ColumnData::I64(o),
        nulls: if nulls.is_empty() { None } else { Some(nulls) },
    })
}

fn e_divide(args: &[Column], rows: usize) -> Result<Column> {
    if let Some(s) = div_scale(&args[0].ty, &args[1].ty) {
        return dec_divide(args, rows, s);
    }
    let (x, y) = (f64_vec(&args[0])?, f64_vec(&args[1])?);
    let mut nulls = nulls_of(args, rows).unwrap_or_default();
    let mut out = Vec::with_capacity(rows);
    for i in 0..rows {
        if y[i] == 0.0 {
            nulls.set(i);
            out.push(0.0);
        } else {
            out.push(x[i] / y[i]);
        }
    }
    Ok(Column {
        ty: DataType::Float64.to_nullable(),
        data: ColumnData::F64(out),
        nulls: if nulls.is_empty() { None } else { Some(nulls) },
    })
}

fn r_intdiv(a: &[DataType]) -> Result<DataType> {
    arity("intDiv", a, 2, 2)?;
    want_num("intDiv", &a[0], 0)?;
    want_num("intDiv", &a[1], 1)?;
    Ok(DataType::Int64.to_nullable())
}

/// `%` keeps the operands' type where `DIV` does not: `12.34 % 5` is 2.34, and
/// an `Int64` result lane could only have said 234 or 2.
fn r_modulo(a: &[DataType]) -> Result<DataType> {
    arity("modulo", a, 2, 2)?;
    want_num("modulo", &a[0], 0)?;
    want_num("modulo", &a[1], 1)?;
    Ok(match dec_int_path(&a[0], &a[1]) {
        true => DataType::Decimal64(scale_of(&a[0]).max(scale_of(&a[1]))),
        false => DataType::Int64,
    }
    .to_nullable())
}

/// Do `DIV`/`%` take the exact path? Only when no float is involved: a float
/// operand has already lost the fraction on its way through `to_i64_vec`, and
/// the `Int64` answer this module documents is then the honest one.
#[inline]
fn dec_int_path(a: &DataType, b: &DataType) -> bool {
    (a.is_decimal() || b.is_decimal()) && !a.is_float() && !b.is_float()
}

/// `DIV` and `%` on decimals, exactly.
///
/// Both reduce to arithmetic on unit counts brought to one scale, which is why
/// they share a body: `DIV` truncates the quotient toward zero (Rust's `/` on
/// integers, which is what the `Int64` path already does), and `%` takes the
/// remainder, whose sign follows the dividend exactly as SQL wants.
fn dec_int_binop(args: &[Column], rows: usize, out: Option<u8>) -> Result<Column> {
    let who = if out.is_some() { "modulo" } else { "intDiv" };
    let hi = out.unwrap_or_else(|| scale_of(&args[0].ty).max(scale_of(&args[1].ty)));
    let (x, y) = (dec_units(&args[0], hi, who)?, dec_units(&args[1], hi, who)?);
    let mut nulls = nulls_of(args, rows).unwrap_or_default();
    let mut o = Vec::with_capacity(rows);
    for i in 0..rows {
        if y[i] == 0 {
            nulls.set(i);
            o.push(0);
        } else {
            // Both lanes are at `hi`, so the ratio is scale-free and the
            // remainder is already denominated in `10^-hi`.
            o.push(match out {
                Some(_) => x[i].wrapping_rem(y[i]),
                None => x[i].wrapping_div(y[i]),
            });
        }
    }
    let ty = out.map_or(DataType::Int64, DataType::Decimal64).to_nullable();
    Ok(Column {
        ty,
        data: ColumnData::I64(o),
        nulls: if nulls.is_empty() { None } else { Some(nulls) },
    })
}

macro_rules! int_binop {
    ($ev:ident, $op:ident, $modulo:literal) => {
        fn $ev(args: &[Column], rows: usize) -> Result<Column> {
            if dec_int_path(&args[0].ty, &args[1].ty) {
                let out = if $modulo {
                    Some(scale_of(&args[0].ty).max(scale_of(&args[1].ty)))
                } else {
                    None
                };
                return dec_int_binop(args, rows, out);
            }
            let (x, y) = (i64_vec(&args[0])?, i64_vec(&args[1])?);
            let mut nulls = nulls_of(args, rows).unwrap_or_default();
            let mut out = Vec::with_capacity(rows);
            for i in 0..rows {
                if y[i] == 0 {
                    nulls.set(i);
                    out.push(0);
                } else {
                    // wrapping_* also absorbs i64::MIN / -1
                    out.push(x[i].$op(y[i]));
                }
            }
            Ok(Column {
                ty: DataType::Int64.to_nullable(),
                data: ColumnData::I64(out),
                nulls: if nulls.is_empty() { None } else { Some(nulls) },
            })
        }
    };
}

int_binop!(e_intdiv, wrapping_div, false);
int_binop!(e_modulo, wrapping_rem, true);

/// ClickHouse rounds half to even; Rust's `f64::round` rounds half away from
/// zero. Correct the one case where they disagree.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (r - x).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

fn round_scaled(x: f64, n: i64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let f = 10f64.powi(n.clamp(-18, 18) as i32);
    let y = x * f;
    if !y.is_finite() {
        return x;
    }
    round_half_even(y) / f
}

fn r_round(a: &[DataType]) -> Result<DataType> {
    arity("round", a, 1, 2)?;
    want_num("round", &a[0], 0)?;
    if a.len() == 2 {
        want_int("round", &a[1], 1)?;
    }
    Ok(nullable_like(a, DataType::Float64))
}

fn e_round(args: &[Column], rows: usize) -> Result<Column> {
    let x = f64_vec(&args[0])?;
    let out: Vec<f64> = if args.len() == 2 {
        let n = args[1].to_i64_vec()?;
        (0..rows).map(|i| round_scaled(x[i], n[i])).collect()
    } else {
        (0..rows).map(|i| round_half_even(x[i])).collect()
    };
    Ok(build(DataType::Float64, ColumnData::F64(out), nulls_of(args, rows)))
}

/// One numeric argument in, `Float64` out. Covers most of libm.
fn r_f64_1(a: &[DataType]) -> Result<DataType> {
    arity("math function", a, 1, 1)?;
    want_num("math function", &a[0], 0)?;
    Ok(nullable_like(a, DataType::Float64))
}

macro_rules! unary_f64 {
    ($ev:ident, $f:expr) => {
        fn $ev(args: &[Column], rows: usize) -> Result<Column> {
            let v = f64_vec(&args[0])?;
            let f: fn(f64) -> f64 = $f;
            let out: Vec<f64> = (0..rows).map(|i| f(v[i])).collect();
            Ok(build(DataType::Float64, ColumnData::F64(out), nulls_of(args, rows)))
        }
    };
}

unary_f64!(e_floor, |x| x.floor());
unary_f64!(e_ceil, |x| x.ceil());
// sqrt/log of an out-of-domain value yields NaN rather than NULL: NaN is a
// representable Float64 and already sorts last, so it needs no extra mask.
unary_f64!(e_sqrt, |x| x.sqrt());
unary_f64!(e_exp, |x| x.exp());
unary_f64!(e_log, |x| x.ln());
unary_f64!(e_log2, |x| x.log2());
unary_f64!(e_log10, |x| x.log10());

fn r_f64_2(a: &[DataType]) -> Result<DataType> {
    arity("pow", a, 2, 2)?;
    want_num("pow", &a[0], 0)?;
    want_num("pow", &a[1], 1)?;
    Ok(nullable_like(a, DataType::Float64))
}

fn e_pow(args: &[Column], rows: usize) -> Result<Column> {
    let (x, y) = (f64_vec(&args[0])?, f64_vec(&args[1])?);
    let out: Vec<f64> = (0..rows).map(|i| x[i].powf(y[i])).collect();
    Ok(build(DataType::Float64, ColumnData::F64(out), nulls_of(args, rows)))
}

fn r_sign(a: &[DataType]) -> Result<DataType> {
    arity("sign", a, 1, 1)?;
    want_num("sign", &a[0], 0)?;
    Ok(nullable_like(a, DataType::Int8))
}

fn e_sign(args: &[Column], rows: usize) -> Result<Column> {
    let out: Vec<i64> = match &args[0].data {
        ColumnData::F64(v) => (0..rows)
            .map(|i| {
                if v[i] > 0.0 {
                    1
                } else if v[i] < 0.0 {
                    -1
                } else {
                    0 // also the NaN answer, matching ClickHouse
                }
            })
            .collect(),
        ColumnData::I64(v) => (0..rows).map(|i| v[i].signum()).collect(),
        ColumnData::U64(v) => (0..rows).map(|i| (v[i] != 0) as i64).collect(),
        ColumnData::Str(_) => return Err(Error::exec("sign: not a numeric column")),
    };
    Ok(build(DataType::Int8, ColumnData::I64(out), nulls_of(args, rows)))
}

fn extreme_ty(name: &str, a: &[DataType]) -> Result<DataType> {
    arity(name, a, 1, usize::MAX)?;
    let mut acc = a[0].base().clone();
    for t in &a[1..] {
        acc = DataType::promote(&acc, t.base())?;
    }
    Ok(nullable_like(a, acc))
}

fn r_greatest(a: &[DataType]) -> Result<DataType> {
    extreme_ty("greatest", a)
}
fn r_least(a: &[DataType]) -> Result<DataType> {
    extreme_ty("least", a)
}

fn eval_extreme(args: &[Column], rows: usize, ty: DataType, greatest: bool) -> Result<Column> {
    let p = ty.base().physical();
    let datas: Vec<ColumnData> = args
        .iter()
        .map(|c| coerce_data(c, &ty))
        .collect::<Result<_>>()?;
    let nulls = nulls_of(args, rows);
    // Reduce to a per-row "which argument wins" index, then reuse `gather`.
    let mut pick = vec![0u32; rows];
    for i in 0..rows {
        if nulls.as_ref().is_some_and(|n| n.get(i)) {
            pick[i] = u32::MAX;
            continue;
        }
        let mut best = 0usize;
        for k in 1..datas.len() {
            let better = match (&datas[k], &datas[best]) {
                (ColumnData::U64(a), ColumnData::U64(b)) => {
                    if greatest { a[i] > b[i] } else { a[i] < b[i] }
                }
                (ColumnData::I64(a), ColumnData::I64(b)) => {
                    if greatest { a[i] > b[i] } else { a[i] < b[i] }
                }
                (ColumnData::F64(a), ColumnData::F64(b)) => {
                    if greatest { a[i] > b[i] } else { a[i] < b[i] }
                }
                (ColumnData::Str(a), ColumnData::Str(b)) => {
                    if greatest { a[i] > b[i] } else { a[i] < b[i] }
                }
                _ => unreachable!("extremes are pre-coerced to one representation"),
            };
            if better {
                best = k;
            }
        }
        pick[i] = best as u32;
    }
    let (data, mask) = gather(args, &datas, &pick, rows, p);
    Ok(build(ty.strip_nullable(), data, Some(mask)))
}

fn e_greatest(args: &[Column], rows: usize) -> Result<Column> {
    let ty = r_greatest(&tys(args))?;
    eval_extreme(args, rows, ty, true)
}

fn e_least(args: &[Column], rows: usize) -> Result<Column> {
    let ty = r_least(&tys(args))?;
    eval_extreme(args, rows, ty, false)
}

// ===========================================================================
// strings
// ===========================================================================

fn r_str1_to_u64(a: &[DataType]) -> Result<DataType> {
    arity("length", a, 1, 1)?;
    want_str("length", &a[0], 0)?;
    Ok(nullable_like(a, DataType::UInt64))
}

fn e_length(args: &[Column], rows: usize) -> Result<Column> {
    let v = args[0].as_str()?;
    let out: Vec<u64> = (0..rows).map(|i| v[i].len() as u64).collect();
    Ok(build(DataType::UInt64, ColumnData::U64(out), nulls_of(args, rows)))
}

fn e_length_utf8(args: &[Column], rows: usize) -> Result<Column> {
    let v = args[0].as_str()?;
    let out: Vec<u64> = (0..rows).map(|i| v[i].chars().count() as u64).collect();
    Ok(build(DataType::UInt64, ColumnData::U64(out), nulls_of(args, rows)))
}

fn r_str1_to_str(a: &[DataType]) -> Result<DataType> {
    arity("string function", a, 1, 1)?;
    want_str("string function", &a[0], 0)?;
    Ok(nullable_like(a, DataType::String))
}

/// Case mapping with an `Arc`-sharing fast path: strings that are already in
/// the target case are handed back by refcount bump, which is the common shape
/// of `lower(url)` over a dictionary-encoded column.
macro_rules! case_map {
    ($ev:ident, $needs:expr, $map:expr) => {
        fn $ev(args: &[Column], rows: usize) -> Result<Column> {
            let v = args[0].as_str()?;
            let needs: fn(&str) -> bool = $needs;
            let map: fn(&str) -> String = $map;
            let mut out: Vec<Arc<str>> = Vec::with_capacity(rows);
            for i in 0..rows {
                if needs(&v[i]) {
                    out.push(Arc::from(map(&v[i])));
                } else {
                    out.push(v[i].clone());
                }
            }
            Ok(build(DataType::String, ColumnData::Str(out), nulls_of(args, rows)))
        }
    };
}

case_map!(
    e_lower,
    |s| s.bytes().any(|b| b.is_ascii_uppercase()),
    |s| s.to_ascii_lowercase()
);
case_map!(
    e_upper,
    |s| s.bytes().any(|b| b.is_ascii_lowercase()),
    |s| s.to_ascii_uppercase()
);
case_map!(
    e_lower_utf8,
    |s| s.chars().any(|c| c.is_uppercase()),
    |s| s.to_lowercase()
);
case_map!(
    e_upper_utf8,
    |s| s.chars().any(|c| c.is_lowercase()),
    |s| s.to_uppercase()
);

fn r_concat(a: &[DataType]) -> Result<DataType> {
    arity("concat", a, 1, usize::MAX)?;
    for (i, t) in a.iter().enumerate() {
        if !t.is_string() {
            return Err(Error::bind(format!(
                "concat: argument {} must be a String, got {t} (wrap it in toString)",
                i + 1
            )));
        }
    }
    Ok(nullable_like(a, DataType::String))
}

fn e_concat(args: &[Column], rows: usize) -> Result<Column> {
    let cols: Vec<&[Arc<str>]> = args.iter().map(|c| c.as_str()).collect::<Result<_>>()?;
    let mut out: Vec<Arc<str>> = Vec::with_capacity(rows);
    let mut buf = String::new();
    for i in 0..rows {
        if cols.len() == 1 {
            out.push(cols[0][i].clone());
            continue;
        }
        buf.clear();
        for c in &cols {
            buf.push_str(&c[i]);
        }
        out.push(Arc::from(buf.as_str()));
    }
    Ok(build(DataType::String, ColumnData::Str(out), nulls_of(args, rows)))
}

/// ClickHouse `substring`, in characters: 1-based offset, negative offset
/// counts from the end, negative length stops that many characters before the
/// end.
fn substr_chars(s: &str, off: i64, len: Option<i64>) -> &str {
    // Every arithmetic step here saturates: offsets and lengths come straight
    // from user SQL, so `substring(s, 1, 9223372036854775807)` must clamp to
    // the string rather than overflow.
    let n = s.chars().count() as i64;
    let start = if off > 0 {
        off - 1
    } else if off == 0 {
        0
    } else {
        n.saturating_add(off).max(0)
    };
    let start = start.clamp(0, n);
    let end = match len {
        None => n,
        Some(l) if l >= 0 => start.saturating_add(l).min(n),
        Some(l) => n.saturating_add(l).clamp(start, n),
    };
    let bs = s.char_indices().nth(start as usize).map_or(s.len(), |(b, _)| b);
    let be = s.char_indices().nth(end as usize).map_or(s.len(), |(b, _)| b);
    &s[bs..be.max(bs)]
}

fn r_substring(a: &[DataType]) -> Result<DataType> {
    arity("substring", a, 2, 3)?;
    want_str("substring", &a[0], 0)?;
    want_int("substring", &a[1], 1)?;
    if a.len() == 3 {
        want_int("substring", &a[2], 2)?;
    }
    Ok(nullable_like(a, DataType::String))
}

fn e_substring(args: &[Column], rows: usize) -> Result<Column> {
    let s = args[0].as_str()?;
    let off = args[1].to_i64_vec()?;
    let len = if args.len() == 3 { Some(args[2].to_i64_vec()?) } else { None };
    let mut out: Vec<Arc<str>> = Vec::with_capacity(rows);
    for i in 0..rows {
        out.push(Arc::from(substr_chars(&s[i], off[i], len.as_ref().map(|l| l[i]))));
    }
    Ok(build(DataType::String, ColumnData::Str(out), nulls_of(args, rows)))
}

macro_rules! trim_fn {
    ($ev:ident, $f:expr) => {
        fn $ev(args: &[Column], rows: usize) -> Result<Column> {
            let v = args[0].as_str()?;
            let f: fn(&str) -> &str = $f;
            let mut out: Vec<Arc<str>> = Vec::with_capacity(rows);
            for i in 0..rows {
                let t = f(&v[i]);
                // untouched strings keep sharing the original allocation
                if t.len() == v[i].len() {
                    out.push(v[i].clone());
                } else {
                    out.push(Arc::from(t));
                }
            }
            Ok(build(DataType::String, ColumnData::Str(out), nulls_of(args, rows)))
        }
    };
}

trim_fn!(e_trim, |s| s.trim());
trim_fn!(e_trim_left, |s| s.trim_start());
trim_fn!(e_trim_right, |s| s.trim_end());

fn e_reverse(args: &[Column], rows: usize) -> Result<Column> {
    let v = args[0].as_str()?;
    let mut out: Vec<Arc<str>> = Vec::with_capacity(rows);
    for i in 0..rows {
        // reversed by character, not byte: a byte reversal would not be UTF-8
        out.push(Arc::from(v[i].chars().rev().collect::<String>()));
    }
    Ok(build(DataType::String, ColumnData::Str(out), nulls_of(args, rows)))
}

fn r_replace(a: &[DataType]) -> Result<DataType> {
    arity("replaceAll", a, 3, 3)?;
    for i in 0..3 {
        want_str("replaceAll", &a[i], i)?;
    }
    Ok(nullable_like(a, DataType::String))
}

fn e_replace(args: &[Column], rows: usize) -> Result<Column> {
    let (s, from, to) = (args[0].as_str()?, args[1].as_str()?, args[2].as_str()?);
    let mut out: Vec<Arc<str>> = Vec::with_capacity(rows);
    for i in 0..rows {
        // an empty needle would make str::replace splice `to` between every
        // character; ClickHouse leaves the subject alone, so we do too
        if from[i].is_empty() {
            out.push(s[i].clone());
        } else {
            out.push(Arc::from(s[i].replace(from[i].as_ref(), &to[i])));
        }
    }
    Ok(build(DataType::String, ColumnData::Str(out), nulls_of(args, rows)))
}

fn r_str2_to_bool(a: &[DataType]) -> Result<DataType> {
    arity("string predicate", a, 2, 2)?;
    want_str("string predicate", &a[0], 0)?;
    want_str("string predicate", &a[1], 1)?;
    Ok(nullable_like(a, DataType::Bool))
}

macro_rules! str2_pred {
    ($ev:ident, $f:expr) => {
        fn $ev(args: &[Column], rows: usize) -> Result<Column> {
            let (a, b) = (args[0].as_str()?, args[1].as_str()?);
            let f: fn(&str, &str) -> bool = $f;
            let out: Vec<u64> = (0..rows).map(|i| f(&a[i], &b[i]) as u64).collect();
            Ok(build(DataType::Bool, ColumnData::U64(out), nulls_of(args, rows)))
        }
    };
}

str2_pred!(e_starts_with, |a, b| a.starts_with(b));
str2_pred!(e_ends_with, |a, b| a.ends_with(b));

fn r_position(a: &[DataType]) -> Result<DataType> {
    arity("position", a, 2, 2)?;
    want_str("position", &a[0], 0)?;
    want_str("position", &a[1], 1)?;
    Ok(nullable_like(a, DataType::UInt64))
}

fn e_position(args: &[Column], rows: usize) -> Result<Column> {
    let (h, n) = (args[0].as_str()?, args[1].as_str()?);
    let mut out = Vec::with_capacity(rows);
    for i in 0..rows {
        // 1-based *character* index, 0 when absent
        out.push(match h[i].find(n[i].as_ref()) {
            Some(b) => h[i][..b].chars().count() as u64 + 1,
            None => 0,
        });
    }
    Ok(build(DataType::UInt64, ColumnData::U64(out), nulls_of(args, rows)))
}

fn r_split(a: &[DataType]) -> Result<DataType> {
    arity("splitByChar", a, 2, 2)?;
    want_str("splitByChar", &a[0], 0)?;
    want_str("splitByChar", &a[1], 1)?;
    Ok(nullable_like(a, DataType::String))
}

fn e_split(args: &[Column], rows: usize) -> Result<Column> {
    // (separator, subject) — ClickHouse's argument order. We can only return
    // the first field; see the module docs.
    let (sep, s) = (args[0].as_str()?, args[1].as_str()?);
    let mut out: Vec<Arc<str>> = Vec::with_capacity(rows);
    for i in 0..rows {
        if sep[i].is_empty() {
            out.push(s[i].clone());
        } else {
            match s[i].split(sep[i].as_ref()).next() {
                Some(f) if f.len() == s[i].len() => out.push(s[i].clone()),
                Some(f) => out.push(Arc::from(f)),
                None => out.push(Arc::from("")),
            }
        }
    }
    Ok(build(DataType::String, ColumnData::Str(out), nulls_of(args, rows)))
}

fn r_repeat(a: &[DataType]) -> Result<DataType> {
    arity("repeat", a, 2, 2)?;
    want_str("repeat", &a[0], 0)?;
    want_int("repeat", &a[1], 1)?;
    Ok(nullable_like(a, DataType::String))
}

fn e_repeat(args: &[Column], rows: usize) -> Result<Column> {
    let s = args[0].as_str()?;
    let n = args[1].to_i64_vec()?;
    let mut out: Vec<Arc<str>> = Vec::with_capacity(rows);
    for i in 0..rows {
        let k = n[i].max(0) as usize;
        if s[i].len().saturating_mul(k) > MAX_STR {
            return Err(Error::exec(format!(
                "repeat: result would exceed {MAX_STR} bytes"
            )));
        }
        out.push(Arc::from(s[i].repeat(k)));
    }
    Ok(build(DataType::String, ColumnData::Str(out), nulls_of(args, rows)))
}

fn pad_str(s: &str, n: usize, p: &str, left: bool) -> String {
    let cur = s.chars().count();
    if cur >= n {
        // ClickHouse truncates when the subject is already longer
        return s.chars().take(n).collect();
    }
    let need = n - cur;
    let pc: Vec<char> = if p.is_empty() { vec![' '] } else { p.chars().collect() };
    let fill: String = (0..need).map(|i| pc[i % pc.len()]).collect();
    if left {
        fill + s
    } else {
        let mut o = s.to_string();
        o.push_str(&fill);
        o
    }
}

fn r_pad(a: &[DataType]) -> Result<DataType> {
    arity("leftPad", a, 2, 3)?;
    want_str("leftPad", &a[0], 0)?;
    want_int("leftPad", &a[1], 1)?;
    if a.len() == 3 {
        want_str("leftPad", &a[2], 2)?;
    }
    Ok(nullable_like(a, DataType::String))
}

macro_rules! pad_fn {
    ($ev:ident, $left:expr) => {
        fn $ev(args: &[Column], rows: usize) -> Result<Column> {
            let s = args[0].as_str()?;
            let n = args[1].to_i64_vec()?;
            let p = if args.len() == 3 { Some(args[2].as_str()?) } else { None };
            let mut out: Vec<Arc<str>> = Vec::with_capacity(rows);
            for i in 0..rows {
                let want = n[i].max(0) as usize;
                if want > MAX_STR {
                    return Err(Error::exec(format!("pad: length {want} exceeds {MAX_STR}")));
                }
                let pad = p.map_or(" ", |c| c[i].as_ref());
                out.push(Arc::from(pad_str(&s[i], want, pad, $left)));
            }
            Ok(build(DataType::String, ColumnData::Str(out), nulls_of(args, rows)))
        }
    };
}

pad_fn!(e_left_pad, true);
pad_fn!(e_right_pad, false);

// ===========================================================================
// LIKE matching
// ===========================================================================

/// Length of the UTF-8 sequence a lead byte starts. Continuation bytes and
/// invalid lead bytes report 1, which only matters for malformed input we
/// cannot receive from an `Arc<str>`.
#[inline]
fn utf8_len(b: u8) -> usize {
    match b {
        0x00..=0x7F => 1,
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF4 => 4,
        _ => 1,
    }
}

#[inline]
fn beq(a: u8, b: u8, ci: bool) -> bool {
    if ci {
        a.eq_ignore_ascii_case(&b)
    } else {
        a == b
    }
}

/// SQL `LIKE` with `%`, `_` and backslash escaping.
///
/// This is the classic two-pointer wildcard matcher: on a mismatch we rewind
/// to the most recent `%` and let it swallow one more character. It is
/// **iterative** — no recursion, no call stack proportional to the pattern —
/// and its worst case is O(|s| * |p|) rather than the exponential blowup a
/// naive backtracking regex would hit on `%a%a%a%a%b`.
///
/// `_` consumes one whole *character*, and an escaped literal in the pattern
/// consumes one whole character too, so multi-byte text behaves sensibly even
/// though the scan itself is over bytes. Case folding (`ilike`) is ASCII-only:
/// UTF-8 continuation bytes are all >= 0x80, where `eq_ignore_ascii_case` is
/// the identity, so folding cannot corrupt a multi-byte comparison.
fn like_match(s: &str, pat: &str, ci: bool) -> bool {
    let (sb, pb) = (s.as_bytes(), pat.as_bytes());
    let (mut si, mut pi) = (0usize, 0usize);
    let (mut star_p, mut star_s) = (usize::MAX, 0usize);

    while si < sb.len() {
        let mut advanced = false;
        if pi < pb.len() {
            match pb[pi] {
                b'%' => {
                    star_p = pi;
                    star_s = si;
                    pi += 1;
                    continue;
                }
                b'_' => {
                    pi += 1;
                    si += utf8_len(sb[si]);
                    continue;
                }
                b'\\' if pi + 1 < pb.len() => {
                    let cl = utf8_len(pb[pi + 1]);
                    if pi + 1 + cl <= pb.len()
                        && si + cl <= sb.len()
                        && (0..cl).all(|k| beq(sb[si + k], pb[pi + 1 + k], ci))
                    {
                        pi += 1 + cl;
                        si += cl;
                        continue;
                    }
                }
                c => {
                    if beq(c, sb[si], ci) {
                        pi += 1;
                        si += 1;
                        advanced = true;
                    }
                }
            }
        }
        if advanced {
            continue;
        }
        if star_p == usize::MAX {
            return false;
        }
        // rewind: the last `%` absorbs one more character of the subject
        pi = star_p + 1;
        star_s += utf8_len(sb[star_s]);
        si = star_s;
    }
    while pi < pb.len() && pb[pi] == b'%' {
        pi += 1;
    }
    pi == pb.len()
}

macro_rules! like_fn {
    ($ev:ident, $ci:expr, $neg:expr) => {
        fn $ev(args: &[Column], rows: usize) -> Result<Column> {
            let (s, p) = (args[0].as_str()?, args[1].as_str()?);
            let out: Vec<u64> = (0..rows)
                .map(|i| (like_match(&s[i], &p[i], $ci) ^ $neg) as u64)
                .collect();
            Ok(build(DataType::Bool, ColumnData::U64(out), nulls_of(args, rows)))
        }
    };
}

like_fn!(e_like, false, false);
like_fn!(e_not_like, false, true);
like_fn!(e_ilike, true, false);
like_fn!(e_not_ilike, true, true);

/// `col LIKE 'literal'`, which is every `LIKE` a query actually writes.
///
/// The registry entry takes two columns, so [`crate::exec::expr`] had to build
/// one holding `rows` clones of the same `Arc<str>`: an atomic increment per row
/// on the way in and a decrement per row on the way out, to give every row a
/// pointer to the one pattern it was always going to be matched against.
/// Measured 2.1M rows, A/B interleaved best-of-7: 0.61x.
pub(crate) fn like_const(
    c: &Column,
    pat: &str,
    ci: bool,
    neg: bool,
    rows: usize,
) -> Result<Column> {
    let s = c.as_str()?;
    let out: Vec<u64> =
        s[..rows.min(s.len())].iter().map(|x| (like_match(x, pat, ci) ^ neg) as u64).collect();
    // The splatted pattern column never had a mask, so the union `nulls_of`
    // computed was always just the subject's.
    Ok(build(DataType::Bool, ColumnData::U64(out), c.nulls.clone()))
}

// ===========================================================================
// type conversion
// ===========================================================================

fn r_tostring(a: &[DataType]) -> Result<DataType> {
    arity("toString", a, 1, 1)?;
    Ok(nullable_like(a, DataType::String))
}

fn e_tostring(args: &[Column], rows: usize) -> Result<Column> {
    let c = &args[0];
    let mut out: Vec<Arc<str>> = Vec::with_capacity(rows);
    // One match on the physical kind, then a tight loop — the alternative,
    // `Column::value(i).render_plain()`, would build a `Value` per row.
    match (&c.data, c.ty.base()) {
        (ColumnData::Str(v), _) => out.extend_from_slice(&v[..rows]),
        (ColumnData::U64(v), DataType::Date) => {
            for i in 0..rows {
                out.push(Arc::from(fmt_date(v[i] as u32)));
            }
        }
        (ColumnData::U64(v), DataType::DateTime) => {
            for i in 0..rows {
                out.push(Arc::from(fmt_datetime(v[i] as i64)));
            }
        }
        (ColumnData::U64(v), DataType::Bool) => {
            for i in 0..rows {
                out.push(Arc::from(if v[i] != 0 { "true" } else { "false" }));
            }
        }
        (ColumnData::U64(v), _) => {
            for i in 0..rows {
                out.push(Arc::from(v[i].to_string()));
            }
        }
        (ColumnData::I64(v), DataType::DateTime) => {
            for i in 0..rows {
                out.push(Arc::from(fmt_datetime(v[i])));
            }
        }
        // The scale is fixed for the whole column, so the point lands in the
        // same place on every row -- `Value::Decimal` is built here only to
        // reach one shared renderer, and it allocates nothing extra.
        (ColumnData::I64(v), DataType::Decimal64(s)) => {
            for i in 0..rows {
                out.push(Arc::from(Value::Decimal(v[i], *s).render_plain()));
            }
        }
        (ColumnData::I64(v), _) => {
            for i in 0..rows {
                out.push(Arc::from(v[i].to_string()));
            }
        }
        (ColumnData::F64(v), _) => {
            // reuse Value's float formatting so `toString(1.0)` renders "1"
            // exactly as the result-set renderer would
            for i in 0..rows {
                out.push(Arc::from(Value::Float(v[i]).render_plain()));
            }
        }
    }
    Ok(build(DataType::String, ColumnData::Str(out), nulls_of(args, rows)))
}

/// Conversions from a `String` can fail per row, so they are typed `Nullable`;
/// conversions between numbers cannot, so they are not.
fn conv_ty(name: &str, a: &[DataType], out: DataType) -> Result<DataType> {
    arity(name, a, 1, 1)?;
    let t = &a[0];
    if !(t.is_numeric() || t.is_temporal() || t.is_string()) {
        return Err(Error::bind(format!("{name}: cannot convert {t}")));
    }
    Ok(if t.is_string() {
        out.to_nullable()
    } else {
        nullable_like(a, out)
    })
}

/// Parse a numeric literal leniently: integers first, then a float fallback so
/// `toInt64('3.9')` behaves like the float cast rather than failing.
///
/// Integral text is kept as an integer rather than routed through `f64`:
/// `f64` has a 53-bit mantissa, so `toInt64('123456789012345678')` would come
/// back as `...680` if it round-tripped through a float.
enum ParsedNum {
    I(i64),
    U(u64),
    F(f64),
}

fn parse_num(s: &str) -> Option<ParsedNum> {
    let t = s.trim();
    if let Ok(i) = t.parse::<i64>() {
        return Some(ParsedNum::I(i));
    }
    if let Ok(u) = t.parse::<u64>() {
        return Some(ParsedNum::U(u));
    }
    t.parse::<f64>().ok().filter(|f| f.is_finite()).map(ParsedNum::F)
}

macro_rules! to_int {
    ($ev:ident, $ret:ident, $name:literal, $ty:expr, $prim:ty, $variant:ident, $store:ty, $signed:expr) => {
        fn $ret(a: &[DataType]) -> Result<DataType> {
            conv_ty($name, a, $ty)
        }
        fn $ev(args: &[Column], rows: usize) -> Result<Column> {
            let c = &args[0];
            let mut nulls = nulls_of(args, rows).unwrap_or_default();
            let data: Vec<$store> = if let ColumnData::Str(v) = &c.data {
                let mut out = Vec::with_capacity(rows);
                for i in 0..rows {
                    match parse_num(&v[i]) {
                        // Integral text stays integral, so no mantissa is lost
                        // on the way to a 64-bit target.
                        Some(ParsedNum::I(n)) => out.push(n as $prim as $store),
                        Some(ParsedNum::U(n)) => out.push(n as $prim as $store),
                        Some(ParsedNum::F(f)) => out.push(f as $prim as $store),
                        None => {
                            nulls.set(i);
                            out.push(0);
                        }
                    }
                }
                out
            } else if let Some(s) = c.ty.decimal_scale() {
                // A decimal's lane is its unit count, so the plain integer arms
                // below would answer 1234 for `toInt64(12.34)`. Descale first,
                // truncating toward zero like every other number->int cast here.
                let p = POW10[s as usize];
                c.as_i64()?
                    .iter()
                    .take(rows)
                    .map(|&x| (x as i128 / p) as $prim as $store)
                    .collect()
            } else if $signed {
                c.to_i64_vec()?.iter().take(rows).map(|&x| x as $prim as $store).collect()
            } else {
                to_u64_vec(c)?.iter().take(rows).map(|&x| x as $prim as $store).collect()
            };
            let ty = $ret(&tys(args))?;
            Ok(build(
                ty.strip_nullable(),
                ColumnData::$variant(data),
                if nulls.is_empty() { None } else { Some(nulls) },
            ))
        }
    };
}

// The `as $prim as $store` chain is the documented truncating cast: it narrows
// (`toUInt32` of 2^33 wraps) and float→int saturates, per Rust's `as` rules.
to_int!(e_to_u64, r_to_u64, "toUInt64", DataType::UInt64, u64, U64, u64, false);
to_int!(e_to_u32, r_to_u32, "toUInt32", DataType::UInt32, u32, U64, u64, false);
to_int!(e_to_i64, r_to_i64, "toInt64", DataType::Int64, i64, I64, i64, true);
to_int!(e_to_i32, r_to_i32, "toInt32", DataType::Int32, i32, I64, i64, true);

fn r_to_f64(a: &[DataType]) -> Result<DataType> {
    conv_ty("toFloat64", a, DataType::Float64)
}

fn e_to_f64(args: &[Column], rows: usize) -> Result<Column> {
    let c = &args[0];
    let mut nulls = nulls_of(args, rows).unwrap_or_default();
    let out: Vec<f64> = if let ColumnData::Str(v) = &c.data {
        let mut o = Vec::with_capacity(rows);
        for i in 0..rows {
            match v[i].trim().parse::<f64>() {
                Ok(f) => o.push(f),
                Err(_) => {
                    nulls.set(i);
                    o.push(0.0);
                }
            }
        }
        o
    } else if let Some(s) = c.ty.decimal_scale() {
        // Lossy, and the only conversion here that is lossy *by construction*:
        // 0.1 has no double. The divisor is hoisted; the loop is one FDIV.
        let d = POW10[s as usize] as f64;
        c.as_i64()?.iter().take(rows).map(|&x| x as f64 / d).collect()
    } else {
        c.to_f64_vec()?
    };
    let ty = r_to_f64(&tys(args))?;
    Ok(build(
        ty.strip_nullable(),
        ColumnData::F64(out),
        if nulls.is_empty() { None } else { Some(nulls) },
    ))
}

fn r_to_date(a: &[DataType]) -> Result<DataType> {
    conv_ty("toDate", a, DataType::Date)
}

/// `Date` is physically an *unsigned* 32-bit day count: `Column::value` reads
/// the lane back as `v[i] as u32`, and the part writer packs the same 32 bits.
/// A day number outside `[0, u32::MAX]` therefore does not overflow loudly, it
/// truncates into a different, entirely plausible-looking date -- `-1` day
/// arrived as 11761191-12-31 -- which is then written to disk and is
/// indistinguishable from real data forever after. Every path that can produce
/// an out-of-lane day number is gated on this.
#[inline(always)]
fn in_date_lane(days: i64) -> bool {
    (0..=u32::MAX as i64).contains(&days)
}

/// Raising is against this module's grain (one bad row normally NULLs itself
/// rather than killing 8191 good ones), but `ret` only promises
/// `Nullable(Date)` when the *argument* is nullable, so on the common
/// non-nullable argument there is no NULL to hand back and inventing one would
/// make `eval` contradict `ret`. The string overload keeps the NULL behaviour
/// because `conv_ty` already typed it `Nullable`.
#[cold]
fn date_lane_err(what: &dyn std::fmt::Display) -> Error {
    Error::exec(format!(
        "toDate: {what} is outside the Date range 1970-01-01 .. {} \
         (Date is an unsigned day count from the epoch)",
        fmt_date(u32::MAX)
    ))
}

fn e_to_date(args: &[Column], rows: usize) -> Result<Column> {
    // A decimal lane is a unit count, not a day number: descale first, then
    // re-enter on the plain integer the rest of this function expects.
    if args[0].ty.is_decimal() {
        return e_to_date(&[dec_to_int_column(&args[0])?], rows);
    }
    let c = &args[0];
    let mut nulls = nulls_of(args, rows).unwrap_or_default();
    let out: Vec<u64> = match (&c.data, c.ty.base()) {
        (ColumnData::Str(v), _) => {
            let mut o = Vec::with_capacity(rows);
            for i in 0..rows {
                // parse_datetime also accepts a bare YYYY-MM-DD. A pre-1970
                // literal parses fine and is simply not representable, so it
                // takes the same NULL as an unparseable one.
                match parse_datetime(&v[i]) {
                    Ok(s) if in_date_lane(s.div_euclid(86_400)) => {
                        o.push(s.div_euclid(86_400) as u64)
                    }
                    _ => {
                        nulls.set(i);
                        o.push(0);
                    }
                }
            }
            o
        }
        // Identity, and kept as its own arm so the no-op does not pay for the
        // per-row gate below: a Date lane is in range by construction.
        (_, DataType::Date) => to_u64_vec(c)?.into_iter().take(rows).collect(),
        (_, DataType::DateTime) => {
            let s = c.to_i64_vec()?;
            let mut o = Vec::with_capacity(rows);
            for i in 0..rows {
                // The lane under a NULL is arbitrary -- `build` never reads it
                // -- so range-checking it would raise on a row that has no
                // value at all.
                if c.is_null(i) {
                    o.push(0);
                    continue;
                }
                // div_euclid, not `/`: truncation-toward-zero would map every
                // instant on 1969-12-31 to day 0 and silently gain a day.
                let d = s[i].div_euclid(86_400);
                if !in_date_lane(d) {
                    return Err(date_lane_err(&fmt_datetime(s[i])));
                }
                o.push(d as u64);
            }
            o
        }
        _ => {
            // A bare number is a day count (see the module header). It still
            // has to clear the lane: `toDate(-1)` used to wrap through
            // `as u64` into 18446744073709551615, whose low 32 bits are a
            // perfectly ordinary-looking date.
            let mut o = vec![0u64; rows];
            macro_rules! gate {
                ($v:expr, $days:expr) => {{
                    let (v, days) = ($v, $days);
                    for i in 0..rows {
                        if c.is_null(i) {
                            continue;
                        }
                        match days(v[i]) {
                            Some(d) if in_date_lane(d) => o[i] = d as u64,
                            _ => return Err(date_lane_err(&v[i])),
                        }
                    }
                }};
            }
            match &c.data {
                ColumnData::U64(v) => gate!(v, |x: u64| i64::try_from(x).ok()),
                ColumnData::I64(v) => gate!(v, Some::<i64>),
                // NaN/±inf have no day number; `as i64` would saturate them
                // into a real date, so they are rejected explicitly.
                ColumnData::F64(v) => gate!(v, |x: f64| x.is_finite().then(|| x.trunc() as i64)),
                // Unreachable -- string data is the first arm regardless of
                // declared type -- but an error beats a panic in a server.
                ColumnData::Str(_) => {
                    return Err(Error::exec("toDate: a String column reached the numeric path"))
                }
            }
            o
        }
    };
    let ty = r_to_date(&tys(args))?;
    Ok(build(
        ty.strip_nullable(),
        ColumnData::U64(out),
        if nulls.is_empty() { None } else { Some(nulls) },
    ))
}

fn r_to_datetime(a: &[DataType]) -> Result<DataType> {
    conv_ty("toDateTime", a, DataType::DateTime)
}

fn e_to_datetime(args: &[Column], rows: usize) -> Result<Column> {
    if args[0].ty.is_decimal() {
        return e_to_datetime(&[dec_to_int_column(&args[0])?], rows);
    }
    let c = &args[0];
    let mut nulls = nulls_of(args, rows).unwrap_or_default();
    // DateTime is physically U64 even though it means a signed epoch second;
    // the `as u64` / `as i64` round trip is exact, including pre-1970.
    let out: Vec<u64> = match (&c.data, c.ty.base()) {
        (ColumnData::Str(v), _) => {
            let mut o = Vec::with_capacity(rows);
            for i in 0..rows {
                match parse_datetime(&v[i]) {
                    Ok(s) => o.push(s as u64),
                    Err(_) => {
                        nulls.set(i);
                        o.push(0);
                    }
                }
            }
            o
        }
        (_, DataType::Date) => {
            let d = to_u64_vec(c)?;
            (0..rows).map(|i| d[i].wrapping_mul(86_400)).collect()
        }
        _ => to_u64_vec(c)?.into_iter().take(rows).collect(),
    };
    let ty = r_to_datetime(&tys(args))?;
    Ok(build(
        ty.strip_nullable(),
        ColumnData::U64(out),
        if nulls.is_empty() { None } else { Some(nulls) },
    ))
}

// ===========================================================================
// dates
// ===========================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Unit {
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Quarter,
    Year,
}

fn parse_unit(s: &str) -> Option<Unit> {
    const T: &[(&str, Unit)] = &[
        ("second", Unit::Second), ("seconds", Unit::Second), ("ss", Unit::Second), ("s", Unit::Second),
        ("minute", Unit::Minute), ("minutes", Unit::Minute), ("mi", Unit::Minute), ("n", Unit::Minute),
        ("hour", Unit::Hour), ("hours", Unit::Hour), ("hh", Unit::Hour), ("h", Unit::Hour),
        ("day", Unit::Day), ("days", Unit::Day), ("dd", Unit::Day), ("d", Unit::Day),
        ("week", Unit::Week), ("weeks", Unit::Week), ("wk", Unit::Week), ("ww", Unit::Week),
        ("month", Unit::Month), ("months", Unit::Month), ("mm", Unit::Month), ("m", Unit::Month),
        ("quarter", Unit::Quarter), ("quarters", Unit::Quarter), ("qq", Unit::Quarter), ("q", Unit::Quarter),
        ("year", Unit::Year), ("years", Unit::Year), ("yyyy", Unit::Year), ("yy", Unit::Year),
    ];
    let t = s.trim();
    T.iter().find(|(k, _)| k.eq_ignore_ascii_case(t)).map(|(_, u)| *u)
}

/// Every temporal column reduces to epoch seconds before the calendar math, so
/// `Date` and `DateTime` share one implementation of every extractor.
fn temporal_secs(c: &Column, rows: usize) -> Result<Vec<i64>> {
    let mul = if matches!(c.ty.base(), DataType::Date) { 86_400i64 } else { 1 };
    Ok(match &c.data {
        ColumnData::U64(v) => (0..rows).map(|i| (v[i] as i64).wrapping_mul(mul)).collect(),
        ColumnData::I64(v) => (0..rows).map(|i| v[i].wrapping_mul(mul)).collect(),
        _ => return Err(Error::exec(format!("{} is not a temporal column", c.ty))),
    })
}

fn days_in_month(y: i64, m: u32) -> u32 {
    let (ny, nm) = if m == 12 { (y + 1, 1) } else { (y, m + 1) };
    (days_from_civil(ny, nm, 1) - days_from_civil(y, m, 1)) as u32
}

/// Shift epoch seconds by `n` units, clamping the day-of-month when a calendar
/// month is shorter (Jan 31 + 1 month = Feb 28/29, as ClickHouse does).
fn add_unit(secs: i64, n: i64, u: Unit) -> i64 {
    match u {
        Unit::Second => secs.saturating_add(n),
        Unit::Minute => secs.saturating_add(n.saturating_mul(60)),
        Unit::Hour => secs.saturating_add(n.saturating_mul(3_600)),
        Unit::Day => secs.saturating_add(n.saturating_mul(86_400)),
        Unit::Week => secs.saturating_add(n.saturating_mul(604_800)),
        Unit::Month | Unit::Quarter | Unit::Year => {
            let months = match u {
                Unit::Month => n,
                Unit::Quarter => n.saturating_mul(3),
                _ => n.saturating_mul(12),
            };
            let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
            let (y, m, d) = civil_from_days(days);
            // Saturating throughout: `addYears(d, 9223372036854775807)` is a
            // legal thing to write, and must clamp rather than overflow.
            let total = y
                .saturating_mul(12)
                .saturating_add(m as i64 - 1)
                .saturating_add(months);
            let (ny, nm) = (total.div_euclid(12), total.rem_euclid(12) as u32 + 1);
            let nd = d.min(days_in_month(ny, nm));
            days_from_civil(ny, nm, nd)
                .saturating_mul(86_400)
                .saturating_add(rem)
        }
    }
}

fn r_temporal1(a: &[DataType], out: DataType) -> Result<DataType> {
    arity("date function", a, 1, 1)?;
    want_temporal("date function", &a[0], 0)?;
    Ok(nullable_like(a, out))
}

/// All the extractors and truncators share one shape: temporal in, `u64` out.
macro_rules! date_part {
    ($ev:ident, $ret:ident, $ty:expr, |$s:ident| $body:expr) => {
        fn $ret(a: &[DataType]) -> Result<DataType> {
            r_temporal1(a, $ty)
        }
        fn $ev(args: &[Column], rows: usize) -> Result<Column> {
            let secs = temporal_secs(&args[0], rows)?;
            let mut out = Vec::with_capacity(rows);
            // `Date` is days since 1970 in a u32 lane, so it cannot represent
            // anything before the epoch. A pre-epoch input would produce a
            // negative day count that `as u64` would wrap into a date roughly
            // 50 billion years hence; clamp to 1970-01-01 instead, which is
            // what ClickHouse does for out-of-range Date values.
            let clamp_date = matches!($ty, DataType::Date);
            for i in 0..rows {
                let $s = secs[i];
                let v = $body as i64;
                out.push(if clamp_date { v.max(0) as u64 } else { v as u64 });
            }
            Ok(build($ty, ColumnData::U64(out), nulls_of(args, rows)))
        }
    };
}

date_part!(e_year, r_year, DataType::UInt16, |s| civil_from_days(s.div_euclid(86_400)).0);
date_part!(e_month, r_month, DataType::UInt8, |s| civil_from_days(s.div_euclid(86_400)).1);
date_part!(e_dom, r_dom, DataType::UInt8, |s| civil_from_days(s.div_euclid(86_400)).2);
// 1970-01-01 was a Thursday, so +3 rotates the epoch onto ClickHouse's
// Monday=1 numbering.
date_part!(e_dow, r_dow, DataType::UInt8, |s| s.div_euclid(86_400).wrapping_add(3).rem_euclid(7) + 1);
date_part!(e_doy, r_doy, DataType::UInt16, |s| {
    let d = s.div_euclid(86_400);
    let (y, _, _) = civil_from_days(d);
    d - days_from_civil(y, 1, 1) + 1
});
date_part!(e_quarter, r_quarter, DataType::UInt8, |s| (civil_from_days(s.div_euclid(86_400)).1 - 1) / 3 + 1);
date_part!(e_hour, r_hour, DataType::UInt8, |s| s.rem_euclid(86_400) / 3_600);
date_part!(e_minute, r_minute, DataType::UInt8, |s| s.rem_euclid(86_400) % 3_600 / 60);
date_part!(e_second, r_second, DataType::UInt8, |s| s.rem_euclid(86_400) % 60);

date_part!(e_start_day, r_start_day, DataType::DateTime, |s| s - s.rem_euclid(86_400));
date_part!(e_start_hour, r_start_hour, DataType::DateTime, |s| s - s.rem_euclid(3_600));
date_part!(e_start_min, r_start_min, DataType::DateTime, |s| s - s.rem_euclid(60));
date_part!(e_start_month, r_start_month, DataType::Date, |s| {
    let (y, m, _) = civil_from_days(s.div_euclid(86_400));
    days_from_civil(y, m, 1)
});
date_part!(e_start_quarter, r_start_quarter, DataType::Date, |s| {
    let (y, m, _) = civil_from_days(s.div_euclid(86_400));
    days_from_civil(y, (m - 1) / 3 * 3 + 1, 1)
});
date_part!(e_start_year, r_start_year, DataType::Date, |s| {
    days_from_civil(civil_from_days(s.div_euclid(86_400)).0, 1, 1)
});
date_part!(e_monday, r_monday, DataType::Date, |s| {
    let d = s.div_euclid(86_400);
    d - (d + 3).rem_euclid(7)
});

fn r_unixts(a: &[DataType]) -> Result<DataType> {
    arity("toUnixTimestamp", a, 1, 1)?;
    want_temporal("toUnixTimestamp", &a[0], 0)?;
    Ok(nullable_like(a, DataType::Int64))
}

fn e_unixts(args: &[Column], rows: usize) -> Result<Column> {
    let secs = temporal_secs(&args[0], rows)?;
    Ok(build(DataType::Int64, ColumnData::I64(secs), nulls_of(args, rows)))
}

fn r_now(a: &[DataType]) -> Result<DataType> {
    arity("now", a, 0, 0)?;
    Ok(DataType::DateTime)
}

/// Read the wall clock once and broadcast it. Evaluating `now()` per row would
/// let a single block observe two different "now"s, which breaks
/// `WHERE t < now()` reproducibility inside one query.
fn wall_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn e_now(_args: &[Column], rows: usize) -> Result<Column> {
    Ok(Column::new(DataType::DateTime, ColumnData::U64(vec![wall_secs() as u64; rows])))
}

fn r_today(a: &[DataType]) -> Result<DataType> {
    arity("today", a, 0, 0)?;
    Ok(DataType::Date)
}

fn e_today(_args: &[Column], rows: usize) -> Result<Column> {
    let d = wall_secs().div_euclid(86_400) as u64;
    Ok(Column::new(DataType::Date, ColumnData::U64(vec![d; rows])))
}

fn r_datediff(a: &[DataType]) -> Result<DataType> {
    arity("dateDiff", a, 3, 3)?;
    want_str("dateDiff", &a[0], 0)?;
    want_temporal("dateDiff", &a[1], 1)?;
    want_temporal("dateDiff", &a[2], 2)?;
    Ok(nullable_like(a, DataType::Int64))
}

/// Boundary-crossing count, matching ClickHouse: `dateDiff('day', 23:59, 00:01
/// next day)` is 1, not 0.
fn diff_unit(u: Unit, a: i64, b: i64) -> i64 {
    let (ad, bd) = (a.div_euclid(86_400), b.div_euclid(86_400));
    match u {
        Unit::Second => b - a,
        Unit::Minute => b.div_euclid(60) - a.div_euclid(60),
        Unit::Hour => b.div_euclid(3_600) - a.div_euclid(3_600),
        Unit::Day => bd - ad,
        Unit::Week => (bd + 3).div_euclid(7) - (ad + 3).div_euclid(7),
        Unit::Month | Unit::Quarter | Unit::Year => {
            let (ay, am, _) = civil_from_days(ad);
            let (by, bm, _) = civil_from_days(bd);
            match u {
                Unit::Month => (by - ay) * 12 + bm as i64 - am as i64,
                Unit::Quarter => {
                    (by * 4 + (bm as i64 - 1) / 3) - (ay * 4 + (am as i64 - 1) / 3)
                }
                _ => by - ay,
            }
        }
    }
}

fn e_datediff(args: &[Column], rows: usize) -> Result<Column> {
    let units = args[0].as_str()?;
    let a = temporal_secs(&args[1], rows)?;
    let b = temporal_secs(&args[2], rows)?;
    let nulls = nulls_of(args, rows);
    let mut out = Vec::with_capacity(rows);
    for i in 0..rows {
        if nulls.as_ref().is_some_and(|n| n.get(i)) {
            out.push(0);
            continue;
        }
        let u = parse_unit(&units[i])
            .ok_or_else(|| Error::exec(format!("dateDiff: unknown unit '{}'", units[i])))?;
        out.push(diff_unit(u, a[i], b[i]));
    }
    Ok(build(DataType::Int64, ColumnData::I64(out), nulls))
}

/// Adding a sub-day amount to a `Date` has to widen the result to `DateTime`,
/// otherwise the shift would be silently truncated away.
fn shift_ty(a: &DataType, u: Unit) -> DataType {
    let sub_day = matches!(u, Unit::Second | Unit::Minute | Unit::Hour);
    if matches!(a.base(), DataType::Date) && sub_day {
        DataType::DateTime
    } else {
        a.base().clone()
    }
}

fn r_shift(name: &str, a: &[DataType], u: Unit) -> Result<DataType> {
    arity(name, a, 2, 2)?;
    want_temporal(name, &a[0], 0)?;
    want_int(name, &a[1], 1)?;
    Ok(nullable_like(a, shift_ty(&a[0], u)))
}

fn eval_shift(args: &[Column], rows: usize, u: Unit, sign: i64) -> Result<Column> {
    let secs = temporal_secs(&args[0], rows)?;
    let n = args[1].to_i64_vec()?;
    let ty = shift_ty(&args[0].ty, u);
    let to_days = matches!(ty.base(), DataType::Date);
    let mut out = Vec::with_capacity(rows);
    for i in 0..rows {
        let s = add_unit(secs[i], n[i].saturating_mul(sign), u);
        out.push(if to_days { s.div_euclid(86_400) as u64 } else { s as u64 });
    }
    Ok(build(ty, ColumnData::U64(out), nulls_of(args, rows)))
}

macro_rules! date_shift {
    ($ev:ident, $ret:ident, $name:literal, $unit:expr, $sign:expr) => {
        fn $ret(a: &[DataType]) -> Result<DataType> {
            r_shift($name, a, $unit)
        }
        fn $ev(args: &[Column], rows: usize) -> Result<Column> {
            eval_shift(args, rows, $unit, $sign)
        }
    };
}

date_shift!(e_add_s, r_add_s, "addSeconds", Unit::Second, 1);
date_shift!(e_add_mi, r_add_mi, "addMinutes", Unit::Minute, 1);
date_shift!(e_add_h, r_add_h, "addHours", Unit::Hour, 1);
date_shift!(e_add_d, r_add_d, "addDays", Unit::Day, 1);
date_shift!(e_add_w, r_add_w, "addWeeks", Unit::Week, 1);
date_shift!(e_add_mo, r_add_mo, "addMonths", Unit::Month, 1);
date_shift!(e_add_q, r_add_q, "addQuarters", Unit::Quarter, 1);
date_shift!(e_add_y, r_add_y, "addYears", Unit::Year, 1);
date_shift!(e_sub_s, r_sub_s, "subtractSeconds", Unit::Second, -1);
date_shift!(e_sub_mi, r_sub_mi, "subtractMinutes", Unit::Minute, -1);
date_shift!(e_sub_h, r_sub_h, "subtractHours", Unit::Hour, -1);
date_shift!(e_sub_d, r_sub_d, "subtractDays", Unit::Day, -1);
date_shift!(e_sub_w, r_sub_w, "subtractWeeks", Unit::Week, -1);
date_shift!(e_sub_mo, r_sub_mo, "subtractMonths", Unit::Month, -1);
date_shift!(e_sub_q, r_sub_q, "subtractQuarters", Unit::Quarter, -1);
date_shift!(e_sub_y, r_sub_y, "subtractYears", Unit::Year, -1);

// ===========================================================================
// null handling
// ===========================================================================

fn r_isnull(a: &[DataType]) -> Result<DataType> {
    arity("isNull", a, 1, 1)?;
    Ok(DataType::Bool) // never null itself
}

macro_rules! null_pred {
    ($ev:ident, $neg:expr) => {
        fn $ev(args: &[Column], rows: usize) -> Result<Column> {
            let out: Vec<u64> = match &args[0].nulls {
                Some(n) => (0..rows).map(|i| (n.get(i) ^ $neg) as u64).collect(),
                None => vec![$neg as u64; rows],
            };
            Ok(Column::new(DataType::Bool, ColumnData::U64(out)))
        }
    };
}

null_pred!(e_isnull, false);
null_pred!(e_isnotnull, true);

fn r_assume_not_null(a: &[DataType]) -> Result<DataType> {
    arity("assumeNotNull", a, 1, 1)?;
    Ok(a[0].strip_nullable())
}

fn e_assume_not_null(args: &[Column], rows: usize) -> Result<Column> {
    let mut data = args[0].data.clone();
    data.truncate(rows);
    // The mask is dropped, not consulted: the caller asserted there is nothing
    // to see. Rows that were NULL keep whatever zero the builder wrote.
    Ok(Column::new(args[0].ty.strip_nullable(), data))
}

fn r_ifnull(a: &[DataType]) -> Result<DataType> {
    arity("ifNull", a, 2, 2)?;
    let base = DataType::promote(a[0].base(), a[1].base())?;
    // a's nulls are exactly what gets replaced, so only b can leak one through
    Ok(if a[1].is_nullable() { base.to_nullable() } else { base })
}

fn e_ifnull(args: &[Column], rows: usize) -> Result<Column> {
    let ty = r_ifnull(&tys(args))?;
    let p = ty.base().physical();
    let datas = [coerce_data(&args[0], &ty)?, coerce_data(&args[1], &ty)?];
    let pick: Vec<u32> = (0..rows).map(|i| args[0].is_null(i) as u32).collect();
    let (data, mask) = gather(args, &datas, &pick, rows, p);
    Ok(build(ty.strip_nullable(), data, Some(mask)))
}

fn r_coalesce(a: &[DataType]) -> Result<DataType> {
    arity("coalesce", a, 1, usize::MAX)?;
    let mut acc = a[0].base().clone();
    for t in &a[1..] {
        acc = DataType::promote(&acc, t.base())?;
    }
    // non-nullable as soon as one branch cannot be null
    Ok(if a.iter().all(|t| t.is_nullable()) { acc.to_nullable() } else { acc })
}

fn e_coalesce(args: &[Column], rows: usize) -> Result<Column> {
    let ty = r_coalesce(&tys(args))?;
    let p = ty.base().physical();
    let datas: Vec<ColumnData> = args
        .iter()
        .map(|c| coerce_data(c, &ty))
        .collect::<Result<_>>()?;
    let pick: Vec<u32> = (0..rows)
        .map(|i| {
            args.iter()
                .position(|c| !c.is_null(i))
                .map_or(u32::MAX, |k| k as u32)
        })
        .collect();
    let (data, mask) = gather(args, &datas, &pick, rows, p);
    Ok(build(ty.strip_nullable(), data, Some(mask)))
}

fn r_nullif(a: &[DataType]) -> Result<DataType> {
    arity("nullIf", a, 2, 2)?;
    DataType::promote(a[0].base(), a[1].base())?; // comparability check only
    Ok(a[0].to_nullable())
}

fn e_nullif(args: &[Column], rows: usize) -> Result<Column> {
    let cmp_ty = DataType::promote(args[0].ty.base(), args[1].ty.base())?;
    let (x, y) = (coerce_data(&args[0], &cmp_ty)?, coerce_data(&args[1], &cmp_ty)?);
    let mut nulls = args[0].nulls.clone().unwrap_or_default();
    for i in 0..rows {
        let same = match (&x, &y) {
            (ColumnData::U64(a), ColumnData::U64(b)) => a[i] == b[i],
            (ColumnData::I64(a), ColumnData::I64(b)) => a[i] == b[i],
            (ColumnData::F64(a), ColumnData::F64(b)) => a[i] == b[i],
            (ColumnData::Str(a), ColumnData::Str(b)) => a[i] == b[i],
            _ => unreachable!("nullIf operands are pre-coerced"),
        };
        // a NULL on either side is never "equal", so it cannot trigger a null
        if same && !args[0].is_null(i) && !args[1].is_null(i) {
            nulls.set(i);
        }
    }
    let mut data = args[0].data.clone();
    data.truncate(rows);
    Ok(build(args[0].ty.clone(), data, Some(nulls)))
}

// ===========================================================================
// logic
// ===========================================================================

fn r_if(a: &[DataType]) -> Result<DataType> {
    arity("if", a, 3, 3)?;
    let base = DataType::promote(a[1].base(), a[2].base())?;
    // the condition's own nullability cannot leak: NULL selects the else arm
    Ok(if a[1].is_nullable() || a[2].is_nullable() { base.to_nullable() } else { base })
}

fn e_if(args: &[Column], rows: usize) -> Result<Column> {
    let ty = r_if(&tys(args))?;
    let p = ty.base().physical();
    let branches = &args[1..3];
    let datas = [coerce_data(&args[1], &ty)?, coerce_data(&args[2], &ty)?];
    let cond = truth_vec(&args[0], rows)?;
    let pick: Vec<u32> = (0..rows)
        .map(|i| (!(cond[i] && !args[0].is_null(i))) as u32)
        .collect();
    let (data, mask) = gather(branches, &datas, &pick, rows, p);
    Ok(build(ty.strip_nullable(), data, Some(mask)))
}

fn r_not(a: &[DataType]) -> Result<DataType> {
    arity("not", a, 1, 1)?;
    Ok(nullable_like(a, DataType::Bool))
}

fn e_not(args: &[Column], rows: usize) -> Result<Column> {
    let mut out = truth_lanes(&args[0], rows)?;
    out.iter_mut().for_each(|o| *o ^= 1);
    Ok(build(DataType::Bool, ColumnData::U64(out), nulls_of(args, rows)))
}

fn r_bool_var(a: &[DataType]) -> Result<DataType> {
    arity("and", a, 2, usize::MAX)?;
    Ok(nullable_like(a, DataType::Bool))
}

/// SQL truthiness as 0/1 lanes, ready to be a `Bool` column.
///
/// The `Vec<bool>` [`truth_vec`] hands back has to be widened again before it
/// can be one, so anything that only wanted lanes was paying for two buffers.
fn truth_lanes(c: &Column, rows: usize) -> Result<Vec<u64>> {
    Ok(match &c.data {
        ColumnData::U64(v) => v[..rows].iter().map(|&x| (x != 0) as u64).collect(),
        ColumnData::I64(v) => v[..rows].iter().map(|&x| (x != 0) as u64).collect(),
        ColumnData::F64(v) => v[..rows].iter().map(|&x| (x != 0.0) as u64).collect(),
        ColumnData::Str(v) => v[..rows].iter().map(|x| (!x.is_empty()) as u64).collect(),
    })
}

/// Three-valued logic: `false AND NULL` is `false` (the null never matters),
/// but `true AND NULL` is `NULL`. Same shape mirrored for `or`.
macro_rules! logic_var {
    ($ev:ident, $dominant:expr, $bit:tt) => {
        fn $ev(args: &[Column], rows: usize) -> Result<Column> {
            // No mask on any input means no NULL can reach the output, and the
            // whole three-valued dance collapses to a bitwise fold over
            // truthiness. Decided once per block -- which is the entire point,
            // because the general loop below tests a null bit per row *per
            // argument* whether or not any mask exists. `latency > 500 AND
            // country = 'US'`, both operands fresh from a comparison over
            // non-null columns, is the shape this catches, and it is the
            // shape a WHERE clause almost always has.
            //
            // 2.1M rows, A/B interleaved best-of-9, `a > 500 AND b < 500` as a
            // predicate: 0.21x (4.78 vs 1.02 ns/row).
            if args.iter().all(|c| c.nulls.is_none()) {
                let mut out = truth_lanes(&args[0], rows)?;
                for c in &args[1..] {
                    macro_rules! fold {
                        ($v:expr, $t:expr) => {
                            for (o, x) in out.iter_mut().zip(&$v[..rows]) {
                                *o = *o $bit ($t(x) as u64);
                            }
                        };
                    }
                    match &c.data {
                        ColumnData::U64(v) => fold!(v, |x: &u64| *x != 0),
                        ColumnData::I64(v) => fold!(v, |x: &i64| *x != 0),
                        ColumnData::F64(v) => fold!(v, |x: &f64| *x != 0.0),
                        ColumnData::Str(v) => fold!(v, |x: &Arc<str>| !x.is_empty()),
                    }
                }
                return Ok(Column::bools(out));
            }
            let vals: Vec<Vec<bool>> = args
                .iter()
                .map(|c| truth_vec(c, rows))
                .collect::<Result<_>>()?;
            let mut out = vec![(!$dominant) as u64; rows];
            let mut nulls = BitSet::new();
            for i in 0..rows {
                let mut any_null = false;
                let mut dominated = false;
                for (k, c) in args.iter().enumerate() {
                    if c.is_null(i) {
                        any_null = true;
                    } else if vals[k][i] == $dominant {
                        dominated = true;
                        break;
                    }
                }
                if dominated {
                    out[i] = $dominant as u64;
                } else if any_null {
                    nulls.set(i);
                }
            }
            Ok(build(DataType::Bool, ColumnData::U64(out), Some(nulls)))
        }
    };
}

logic_var!(e_and, false, &);
logic_var!(e_or, true, |);

fn r_xor(a: &[DataType]) -> Result<DataType> {
    arity("xor", a, 2, 2)?;
    Ok(nullable_like(a, DataType::Bool))
}

fn e_xor(args: &[Column], rows: usize) -> Result<Column> {
    let (x, y) = (truth_vec(&args[0], rows)?, truth_vec(&args[1], rows)?);
    let out: Vec<u64> = (0..rows).map(|i| (x[i] ^ y[i]) as u64).collect();
    Ok(build(DataType::Bool, ColumnData::U64(out), nulls_of(args, rows)))
}

// ===========================================================================
// hashing and miscellaneous
// ===========================================================================

const CH_SEED: u64 = 0x9AE1_6A3B_2F90_404F;

fn r_cityhash(a: &[DataType]) -> Result<DataType> {
    arity("cityHash64", a, 1, usize::MAX)?;
    Ok(nullable_like(a, DataType::UInt64))
}

/// Not actually CityHash — it is this engine's `mum`-based mixer, which is
/// what every other hash consumer here (join build side, aggregation table,
/// bloom filters) already uses. The name is kept for ClickHouse
/// source-compatibility; values will not match a real ClickHouse instance.
fn e_cityhash(args: &[Column], rows: usize) -> Result<Column> {
    let mut h = vec![0x9E37_79B9_7F4A_7C15u64; rows];
    for a in args {
        match &a.data {
            ColumnData::U64(v) => {
                for i in 0..rows {
                    h[i] = hash_key(h[i] ^ hash_key(v[i], CH_SEED), CH_SEED);
                }
            }
            ColumnData::I64(v) => {
                for i in 0..rows {
                    h[i] = hash_key(h[i] ^ hash_key(v[i] as u64, CH_SEED), CH_SEED);
                }
            }
            ColumnData::F64(v) => {
                for i in 0..rows {
                    h[i] = hash_key(h[i] ^ hash_key(v[i].to_bits(), CH_SEED), CH_SEED);
                }
            }
            ColumnData::Str(v) => {
                for i in 0..rows {
                    h[i] = hash_key(h[i] ^ hash_bytes(v[i].as_bytes(), CH_SEED), CH_SEED);
                }
            }
        }
    }
    Ok(build(DataType::UInt64, ColumnData::U64(h), nulls_of(args, rows)))
}

/// Monotonic counter behind `rand()`.
static RAND_COUNTER: AtomicU64 = AtomicU64::new(0);

fn r_rand(a: &[DataType]) -> Result<DataType> {
    arity("rand", a, 0, 0)?;
    Ok(DataType::UInt32)
}

fn r_rand64(a: &[DataType]) -> Result<DataType> {
    arity("rand64", a, 0, 0)?;
    Ok(DataType::UInt64)
}

/// `splitmix64` over a process-global counter, bumped once per block rather
/// than once per row. This is **deterministic**: a fresh process replaying the
/// same queries in the same order sees the same stream, which makes
/// `rand()`-driven sampling reproducible in tests. It is emphatically not a
/// source of cryptographic randomness.
fn e_rand(_args: &[Column], rows: usize) -> Result<Column> {
    let base = RAND_COUNTER.fetch_add(rows as u64, Ordering::Relaxed);
    let out: Vec<u64> = (0..rows as u64)
        .map(|i| splitmix64(base.wrapping_add(i)) & 0xFFFF_FFFF)
        .collect();
    Ok(Column::new(DataType::UInt32, ColumnData::U64(out)))
}

fn e_rand64(_args: &[Column], rows: usize) -> Result<Column> {
    let base = RAND_COUNTER.fetch_add(rows as u64, Ordering::Relaxed);
    let out: Vec<u64> = (0..rows as u64)
        .map(|i| splitmix64(base.wrapping_add(i)))
        .collect();
    Ok(Column::new(DataType::UInt64, ColumnData::U64(out)))
}

// ===========================================================================
// registry
// ===========================================================================

const fn f(
    name: &'static str,
    lo: usize,
    hi: usize,
    ret: fn(&[DataType]) -> Result<DataType>,
    eval: fn(&[Column], usize) -> Result<Column>,
) -> ScalarFn {
    ScalarFn { name, arity: (lo, hi), ret, eval }
}

const VAR: usize = usize::MAX;

/// The whole library, in one table. Aliases are separate rows pointing at the
/// same pair of function pointers rather than a name-rewriting map: it costs a
/// few dozen bytes of static data and removes an indirection from lookup.
static FUNCS: &[ScalarFn] = &[
    // ---- math
    f("abs", 1, 1, r_abs, e_abs),
    f("negate", 1, 1, r_negate, e_negate),
    f("plus", 2, 2, r_plus, e_plus),
    f("minus", 2, 2, r_minus, e_minus),
    f("multiply", 2, 2, r_multiply, e_multiply),
    f("divide", 2, 2, r_divide, e_divide),
    f("intDiv", 2, 2, r_intdiv, e_intdiv),
    f("modulo", 2, 2, r_modulo, e_modulo),
    f("round", 1, 2, r_round, e_round),
    f("floor", 1, 1, r_f64_1, e_floor),
    f("ceil", 1, 1, r_f64_1, e_ceil),
    f("ceiling", 1, 1, r_f64_1, e_ceil),
    f("sqrt", 1, 1, r_f64_1, e_sqrt),
    f("exp", 1, 1, r_f64_1, e_exp),
    f("log", 1, 1, r_f64_1, e_log),
    f("ln", 1, 1, r_f64_1, e_log),
    f("log2", 1, 1, r_f64_1, e_log2),
    f("log10", 1, 1, r_f64_1, e_log10),
    f("pow", 2, 2, r_f64_2, e_pow),
    f("power", 2, 2, r_f64_2, e_pow),
    f("sign", 1, 1, r_sign, e_sign),
    f("greatest", 1, VAR, r_greatest, e_greatest),
    f("least", 1, VAR, r_least, e_least),
    // ---- strings
    f("length", 1, 1, r_str1_to_u64, e_length),
    f("lengthUTF8", 1, 1, r_str1_to_u64, e_length_utf8),
    f("char_length", 1, 1, r_str1_to_u64, e_length_utf8),
    f("character_length", 1, 1, r_str1_to_u64, e_length_utf8),
    f("lower", 1, 1, r_str1_to_str, e_lower),
    f("lcase", 1, 1, r_str1_to_str, e_lower),
    f("lowerUTF8", 1, 1, r_str1_to_str, e_lower_utf8),
    f("upper", 1, 1, r_str1_to_str, e_upper),
    f("ucase", 1, 1, r_str1_to_str, e_upper),
    f("upperUTF8", 1, 1, r_str1_to_str, e_upper_utf8),
    f("concat", 1, VAR, r_concat, e_concat),
    f("substring", 2, 3, r_substring, e_substring),
    f("substringUTF8", 2, 3, r_substring, e_substring),
    f("substr", 2, 3, r_substring, e_substring),
    f("mid", 2, 3, r_substring, e_substring),
    f("trim", 1, 1, r_str1_to_str, e_trim),
    f("trimBoth", 1, 1, r_str1_to_str, e_trim),
    f("trimLeft", 1, 1, r_str1_to_str, e_trim_left),
    f("ltrim", 1, 1, r_str1_to_str, e_trim_left),
    f("trimRight", 1, 1, r_str1_to_str, e_trim_right),
    f("rtrim", 1, 1, r_str1_to_str, e_trim_right),
    f("replaceAll", 3, 3, r_replace, e_replace),
    f("replace", 3, 3, r_replace, e_replace),
    f("startsWith", 2, 2, r_str2_to_bool, e_starts_with),
    f("endsWith", 2, 2, r_str2_to_bool, e_ends_with),
    f("position", 2, 2, r_position, e_position),
    f("positionUTF8", 2, 2, r_position, e_position),
    f("splitByChar", 2, 2, r_split, e_split),
    f("reverse", 1, 1, r_str1_to_str, e_reverse),
    f("repeat", 2, 2, r_repeat, e_repeat),
    f("leftPad", 2, 3, r_pad, e_left_pad),
    f("rightPad", 2, 3, r_pad, e_right_pad),
    // ---- matching
    f("like", 2, 2, r_str2_to_bool, e_like),
    f("notLike", 2, 2, r_str2_to_bool, e_not_like),
    f("ilike", 2, 2, r_str2_to_bool, e_ilike),
    f("notILike", 2, 2, r_str2_to_bool, e_not_ilike),
    // ---- type conversion
    f("toString", 1, 1, r_tostring, e_tostring),
    f("toUInt64", 1, 1, r_to_u64, e_to_u64),
    f("toUInt32", 1, 1, r_to_u32, e_to_u32),
    f("toInt64", 1, 1, r_to_i64, e_to_i64),
    f("toInt32", 1, 1, r_to_i32, e_to_i32),
    f("toFloat64", 1, 1, r_to_f64, e_to_f64),
    f("toDate", 1, 1, r_to_date, e_to_date),
    f("toDateTime", 1, 1, r_to_datetime, e_to_datetime),
    // ---- dates
    f("toYear", 1, 1, r_year, e_year),
    f("toMonth", 1, 1, r_month, e_month),
    f("toDayOfMonth", 1, 1, r_dom, e_dom),
    f("toDayOfWeek", 1, 1, r_dow, e_dow),
    f("toDayOfYear", 1, 1, r_doy, e_doy),
    f("toQuarter", 1, 1, r_quarter, e_quarter),
    f("toHour", 1, 1, r_hour, e_hour),
    f("toMinute", 1, 1, r_minute, e_minute),
    f("toSecond", 1, 1, r_second, e_second),
    f("toStartOfDay", 1, 1, r_start_day, e_start_day),
    f("toStartOfHour", 1, 1, r_start_hour, e_start_hour),
    f("toStartOfMinute", 1, 1, r_start_min, e_start_min),
    f("toStartOfMonth", 1, 1, r_start_month, e_start_month),
    f("toStartOfQuarter", 1, 1, r_start_quarter, e_start_quarter),
    f("toStartOfYear", 1, 1, r_start_year, e_start_year),
    f("toMonday", 1, 1, r_monday, e_monday),
    f("toUnixTimestamp", 1, 1, r_unixts, e_unixts),
    f("now", 0, 0, r_now, e_now),
    f("today", 0, 0, r_today, e_today),
    f("dateDiff", 3, 3, r_datediff, e_datediff),
    f("date_diff", 3, 3, r_datediff, e_datediff),
    f("addSeconds", 2, 2, r_add_s, e_add_s),
    f("addMinutes", 2, 2, r_add_mi, e_add_mi),
    f("addHours", 2, 2, r_add_h, e_add_h),
    f("addDays", 2, 2, r_add_d, e_add_d),
    f("addWeeks", 2, 2, r_add_w, e_add_w),
    f("addMonths", 2, 2, r_add_mo, e_add_mo),
    f("addQuarters", 2, 2, r_add_q, e_add_q),
    f("addYears", 2, 2, r_add_y, e_add_y),
    f("subtractSeconds", 2, 2, r_sub_s, e_sub_s),
    f("subtractMinutes", 2, 2, r_sub_mi, e_sub_mi),
    f("subtractHours", 2, 2, r_sub_h, e_sub_h),
    f("subtractDays", 2, 2, r_sub_d, e_sub_d),
    f("subtractWeeks", 2, 2, r_sub_w, e_sub_w),
    f("subtractMonths", 2, 2, r_sub_mo, e_sub_mo),
    f("subtractQuarters", 2, 2, r_sub_q, e_sub_q),
    f("subtractYears", 2, 2, r_sub_y, e_sub_y),
    // ---- nulls
    f("isNull", 1, 1, r_isnull, e_isnull),
    f("isNotNull", 1, 1, r_isnull, e_isnotnull),
    f("ifNull", 2, 2, r_ifnull, e_ifnull),
    f("nullIf", 2, 2, r_nullif, e_nullif),
    f("coalesce", 1, VAR, r_coalesce, e_coalesce),
    f("assumeNotNull", 1, 1, r_assume_not_null, e_assume_not_null),
    // ---- logic
    f("if", 3, 3, r_if, e_if),
    f("not", 1, 1, r_not, e_not),
    f("and", 2, VAR, r_bool_var, e_and),
    f("or", 2, VAR, r_bool_var, e_or),
    f("xor", 2, 2, r_xor, e_xor),
    // ---- hashing / misc
    f("cityHash64", 1, VAR, r_cityhash, e_cityhash),
    f("rand", 0, 0, r_rand, e_rand),
    f("rand64", 0, 0, r_rand64, e_rand64),
];

/// Resolve a scalar function. `name` is already lowercased by
/// [`super::scalar`]; the table stores ClickHouse's canonical casing, so the
/// comparison folds case. A linear scan over ~120 entries with a leading
/// length check is a handful of nanoseconds and happens once per call site at
/// bind time, never per row.
pub fn lookup(name: &str) -> Option<&'static ScalarFn> {
    FUNCS.iter().find(|f| f.name.eq_ignore_ascii_case(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------- fixtures

    fn ui(v: &[u64]) -> Column {
        Column::u64s(DataType::UInt64, v.to_vec())
    }
    fn ii(v: &[i64]) -> Column {
        Column::i64s(DataType::Int64, v.to_vec())
    }
    fn ff(v: &[f64]) -> Column {
        Column::f64s(DataType::Float64, v.to_vec())
    }
    fn ss(v: &[&str]) -> Column {
        Column::strs(DataType::String, v.iter().map(|s| Arc::from(*s)).collect())
    }
    fn date(v: &[i64]) -> Column {
        Column::u64s(DataType::Date, v.iter().map(|&d| d as u64).collect())
    }
    fn dt(v: &[i64]) -> Column {
        Column::u64s(DataType::DateTime, v.iter().map(|&s| s as u64).collect())
    }

    /// Mark rows NULL and widen the declared type accordingly.
    fn nul(mut c: Column, idx: &[usize]) -> Column {
        let mut b = BitSet::new();
        for &i in idx {
            b.set(i);
        }
        c.nulls = Some(b);
        c.ty = c.ty.to_nullable();
        c
    }

    /// Drive a function exactly the way the executor would, and assert the
    /// contract: `ret` and `eval` agree, and the row count is preserved.
    fn call_n(name: &str, args: &[Column], rows: usize) -> Result<Column> {
        let f = lookup(name).unwrap_or_else(|| panic!("no function named {name}"));
        f.check_arity(args.len())?;
        let t: Vec<DataType> = args.iter().map(|c| c.ty.clone()).collect();
        let want = (f.ret)(&t)?;
        let out = (f.eval)(args, rows)?;
        assert_eq!(out.len(), rows, "{name}: wrong row count");
        assert_eq!(
            out.ty.physical(),
            want.physical(),
            "{name}: eval returned {} but ret promised {want}",
            out.ty
        );
        Ok(out)
    }

    fn call(name: &str, args: &[Column]) -> Result<Column> {
        let rows = args.first().map_or(0, |c| c.len());
        call_n(name, args, rows)
    }

    fn ret_of(name: &str, a: &[DataType]) -> Result<DataType> {
        (lookup(name).unwrap().ret)(a)
    }

    fn u(c: &Column) -> Vec<u64> {
        c.as_u64().unwrap().to_vec()
    }
    fn i(c: &Column) -> Vec<i64> {
        c.as_i64().unwrap().to_vec()
    }
    fn fl(c: &Column) -> Vec<f64> {
        c.as_f64().unwrap().to_vec()
    }
    fn st(c: &Column) -> Vec<String> {
        c.as_str().unwrap().iter().map(|x| x.to_string()).collect()
    }

    // ------------------------------------------------------------- registry

    #[test]
    fn lookup_folds_case_and_resolves_aliases() {
        assert!(lookup("intdiv").is_some(), "caller lowercases before lookup");
        assert!(lookup("startswith").is_some());
        assert!(lookup("touint64").is_some());
        // aliases share an implementation
        for (a, b) in [("ln", "log"), ("power", "pow"), ("ceiling", "ceil"), ("substr", "substring")] {
            let (x, y) = (lookup(a).unwrap(), lookup(b).unwrap());
            assert_eq!(x.arity, y.arity, "{a}/{b} arity");
        }
    }

    #[test]
    fn lookup_rejects_unknown_names() {
        assert!(lookup("frobnicate").is_none());
        assert!(lookup("").is_none());
        // regex matching is deliberately absent
        assert!(lookup("match").is_none());
        // CAST is the planner's job, not the registry's
        assert!(lookup("cast").is_none());
    }

    #[test]
    fn every_registered_name_is_unique() {
        for (n, f) in FUNCS.iter().enumerate() {
            for g in &FUNCS[n + 1..] {
                assert!(
                    !f.name.eq_ignore_ascii_case(g.name),
                    "duplicate registration for {}",
                    f.name
                );
            }
        }
    }

    // ----------------------------------------------------------------- math

    #[test]
    fn abs_covers_every_physical_kind() {
        assert_eq!(i(&call("abs", &[ii(&[-3, 5])]).unwrap()), vec![3, 5]);
        assert_eq!(fl(&call("abs", &[ff(&[-1.5, 2.0])]).unwrap()), vec![1.5, 2.0]);
        assert_eq!(u(&call("abs", &[ui(&[7])]).unwrap()), vec![7]);
        // i64::MIN has no positive twin; wrap rather than panic
        assert_eq!(i(&call("abs", &[ii(&[i64::MIN])]).unwrap()), vec![i64::MIN]);
    }

    #[test]
    fn negate_widens_unsigned_to_signed() {
        let out = call("negate", &[ui(&[3])]).unwrap();
        assert_eq!(out.ty, DataType::Int64);
        assert_eq!(i(&out), vec![-3]);
        assert_eq!(fl(&call("negate", &[ff(&[2.5])]).unwrap()), vec![-2.5]);
    }

    #[test]
    fn arithmetic_promotes_operand_types() {
        let out = call("plus", &[ii(&[1, 2]), ff(&[0.5, 0.5])]).unwrap();
        assert_eq!(out.ty, DataType::Float64);
        assert_eq!(fl(&out), vec![1.5, 2.5]);

        let out = call("multiply", &[ui(&[3]), ui(&[4])]).unwrap();
        assert_eq!(out.ty, DataType::UInt64);
        assert_eq!(u(&out), vec![12]);

        let out = call("minus", &[ii(&[1]), ui(&[3])]).unwrap();
        assert_eq!(out.ty, DataType::Int64);
        assert_eq!(i(&out), vec![-2]);
    }

    #[test]
    fn date_arithmetic_keeps_the_temporal_type() {
        let d = date(&[19_723]);
        let out = call("plus", &[d, Column::i64s(DataType::Int32, vec![10])]).unwrap();
        assert_eq!(out.ty, DataType::Date);
        assert_eq!(u(&out), vec![19_733]);
    }

    #[test]
    fn date_minus_date_degrades_to_a_day_count() {
        let out = call("minus", &[date(&[19_733]), date(&[19_723])]).unwrap();
        assert_eq!(out.ty, DataType::Int64);
        assert_eq!(i(&out), vec![10]);
        // but adding two dates is meaningless and must be rejected
        assert!(ret_of("plus", &[DataType::Date, DataType::Date]).is_err());
    }

    #[test]
    fn divide_by_zero_yields_null_not_an_error() {
        let out = call("divide", &[ff(&[1.0, 2.0]), ff(&[2.0, 0.0])]).unwrap();
        assert!(out.ty.is_nullable(), "divide is always Nullable");
        assert_eq!(fl(&out)[0], 0.5);
        assert!(!out.is_null(0));
        assert!(out.is_null(1));
    }

    #[test]
    fn intdiv_and_modulo_null_on_zero_divisor() {
        let d = call("intDiv", &[ii(&[7, 7]), ii(&[2, 0])]).unwrap();
        assert_eq!(i(&d)[0], 3);
        assert!(d.is_null(1));

        let m = call("modulo", &[ii(&[7, 7]), ii(&[3, 0])]).unwrap();
        assert_eq!(i(&m)[0], 1);
        assert!(m.is_null(1));
    }

    #[test]
    fn intdiv_survives_the_overflowing_division() {
        // i64::MIN / -1 is the one division that traps in debug builds
        let d = call("intDiv", &[ii(&[i64::MIN]), ii(&[-1])]).unwrap();
        assert_eq!(i(&d), vec![i64::MIN]);
    }

    #[test]
    fn round_uses_bankers_rounding() {
        let out = call("round", &[ff(&[2.5, 3.5, -2.5, 1.4, 0.5])]).unwrap();
        assert_eq!(fl(&out), vec![2.0, 4.0, -2.0, 1.0, 0.0]);
    }

    #[test]
    fn round_with_a_scale_argument() {
        let out = call("round", &[ff(&[1.2345, 1234.0]), ii(&[2, -2])]).unwrap();
        let got = fl(&out);
        assert!((got[0] - 1.23).abs() < 1e-12, "{got:?}");
        assert_eq!(got[1], 1200.0);
    }

    #[test]
    fn floor_ceil_and_the_transcendentals() {
        assert_eq!(fl(&call("floor", &[ff(&[1.7, -1.2])]).unwrap()), vec![1.0, -2.0]);
        assert_eq!(fl(&call("ceil", &[ff(&[1.2, -1.7])]).unwrap()), vec![2.0, -1.0]);
        assert_eq!(fl(&call("sqrt", &[ff(&[9.0])]).unwrap()), vec![3.0]);
        assert_eq!(fl(&call("log2", &[ff(&[8.0])]).unwrap()), vec![3.0]);
        assert_eq!(fl(&call("log10", &[ff(&[1000.0])]).unwrap()), vec![3.0]);
        assert!((fl(&call("ln", &[ff(&[1.0])]).unwrap())[0]).abs() < 1e-12);
        assert!((fl(&call("exp", &[ff(&[0.0])]).unwrap())[0] - 1.0).abs() < 1e-12);
        // out-of-domain stays NaN rather than becoming NULL
        assert!(fl(&call("sqrt", &[ff(&[-1.0])]).unwrap())[0].is_nan());
    }

    #[test]
    fn pow_and_sign() {
        assert_eq!(fl(&call("power", &[ff(&[2.0]), ff(&[10.0])]).unwrap()), vec![1024.0]);
        let s = call("sign", &[ff(&[-3.0, 0.0, 2.0, f64::NAN])]).unwrap();
        assert_eq!(s.ty, DataType::Int8);
        assert_eq!(i(&s), vec![-1, 0, 1, 0]);
        assert_eq!(i(&call("sign", &[ii(&[-9, 0, 9])]).unwrap()), vec![-1, 0, 1]);
    }

    #[test]
    fn greatest_and_least_over_numbers() {
        let g = call("greatest", &[ii(&[1, 5]), ii(&[4, 2]), ii(&[3, 3])]).unwrap();
        assert_eq!(i(&g), vec![4, 5]);
        let l = call("least", &[ii(&[1, 5]), ii(&[4, 2])]).unwrap();
        assert_eq!(i(&l), vec![1, 2]);
        // mixed kinds promote to the wider representation
        let g = call("greatest", &[ii(&[1]), ff(&[0.5])]).unwrap();
        assert_eq!(g.ty, DataType::Float64);
        assert_eq!(fl(&g), vec![1.0]);
    }

    #[test]
    fn greatest_and_least_over_strings() {
        let g = call("greatest", &[ss(&["a", "z"]), ss(&["b", "c"])]).unwrap();
        assert_eq!(st(&g), vec!["b", "z"]);
        let l = call("least", &[ss(&["a", "z"]), ss(&["b", "c"])]).unwrap();
        assert_eq!(st(&l), vec!["a", "c"]);
    }

    // ------------------------------------------------------------- decimals

    fn dd(scale: u8, v: &[i64]) -> Column {
        Column::i64s(DataType::Decimal64(scale), v.to_vec())
    }

    /// Read a decimal column back as rendered text, which is the only way to
    /// assert on the *number* rather than on the lane.
    fn dtext(c: &Column) -> Vec<String> {
        let s = c.ty.decimal_scale().unwrap_or_else(|| panic!("not a decimal: {}", c.ty));
        i(c).iter().map(|&u| Value::Decimal(u, s).render_plain()).collect()
    }

    /// The disqualifier, gone. `0.1 + 0.2` is `0.3`, not `0.30000000000000004`.
    #[test]
    fn the_canonical_float_embarrassment_is_exact_in_decimal() {
        let out = call("plus", &[dd(1, &[1]), dd(1, &[2])]).unwrap();
        assert_eq!(out.ty, DataType::Decimal64(1));
        assert_eq!(dtext(&out), ["0.3"]);
        // Three tenths, in the lane, with nothing left over. Turning that lane
        // back into a `Value::Decimal` is `Column::value`'s job over in
        // src/types/block.rs, which is why this asserts on the lane.
        assert_eq!(i(&out), vec![3]);
        assert_ne!(0.1f64 + 0.2, 0.3, "the float version really is wrong");
    }

    #[test]
    fn addition_aligns_scales_and_subtraction_follows() {
        // Decimal64(2) + Decimal64(4): the result carries the wider scale and
        // the narrower operand is scaled up, not reinterpreted.
        let out = call("plus", &[dd(2, &[150, -150]), dd(4, &[2500, 2500])]).unwrap();
        assert_eq!(out.ty, DataType::Decimal64(4));
        assert_eq!(dtext(&out), ["1.7500", "-1.2500"]);

        let out = call("minus", &[dd(2, &[1000]), dd(4, &[1])]).unwrap();
        assert_eq!(dtext(&out), ["9.9999"]);

        // An integer is exactly representable at any scale, so it joins in.
        let out = call("plus", &[dd(2, &[150]), ii(&[2])]).unwrap();
        assert_eq!(out.ty, DataType::Decimal64(2));
        assert_eq!(dtext(&out), ["3.50"]);
        let out = call("minus", &[ui(&[10]), dd(2, &[150])]).unwrap();
        assert_eq!(dtext(&out), ["8.50"]);

        // A float poisons the exactness and says so by being a float -- but the
        // decimal still has to *descale* on the way in. Reading the lane
        // straight through `to_f64_vec` made this 150.5.
        let out = call("plus", &[dd(2, &[150]), ff(&[0.5])]).unwrap();
        assert_eq!(out.ty, DataType::Float64);
        assert_eq!(fl(&out), vec![2.0]);
    }

    /// Same trap as the mixed-float addition above, everywhere else a decimal
    /// column can be read as a plain number: the lane is a unit count, and every
    /// one of these would otherwise answer for 150 instead of 1.50.
    #[test]
    fn decimals_descale_before_the_float_and_integer_functions() {
        let d = dd(2, &[150]);
        assert_eq!(fl(&call("round", &[d.clone()]).unwrap()), vec![2.0]);
        assert_eq!(fl(&call("floor", &[d.clone()]).unwrap()), vec![1.0]);
        assert_eq!(fl(&call("ceil", &[d.clone()]).unwrap()), vec![2.0]);
        assert_eq!(fl(&call("sqrt", &[dd(2, &[400])]).unwrap()), vec![2.0]);
        assert_eq!(fl(&call("pow", &[d.clone(), ii(&[2])]).unwrap()), vec![2.25]);
        assert_eq!(fl(&call("divide", &[d.clone(), ff(&[0.5])]).unwrap()), vec![3.0]);
        // `DIV`/`%` against a float give up exactness, but still on the number.
        assert_eq!(i(&call("intDiv", &[dd(2, &[700]), ff(&[2.0])]).unwrap()), vec![3]);
        assert_eq!(u(&call("toUInt64", &[dd(2, &[1299])]).unwrap()), vec![12]);
    }

    /// The scales *add* under multiplication, which `DataType::promote` cannot
    /// express -- it only ever unifies -- so `arith_ty` overrides it.
    #[test]
    fn multiplication_adds_scales() {
        let out = call("multiply", &[dd(2, &[150]), dd(2, &[150])]).unwrap();
        assert_eq!(out.ty, DataType::Decimal64(4));
        assert_eq!(dtext(&out), ["2.2500"]);

        // An integer multiplier does not move the scale.
        let out = call("multiply", &[dd(2, &[199]), ii(&[3])]).unwrap();
        assert_eq!(out.ty, DataType::Decimal64(2));
        assert_eq!(dtext(&out), ["5.97"]);

        // A result scale that cannot exist is refused at bind time, before a
        // plan is built, rather than per row.
        let e = ret_of("multiply", &[DataType::Decimal64(10), DataType::Decimal64(10)])
            .unwrap_err()
            .to_string();
        assert!(e.contains("18"), "{e}");
    }

    /// The case this task exists for: a decimal multiply overflows `i64` while
    /// both operands still look small, and it must error rather than wrap.
    #[test]
    fn decimal_overflow_errors_instead_of_wrapping() {
        // 10^9 * 10^9 = 10^18, one digit past the 18-nine limit.
        let big = 1_000_000_000i64;
        let e = call("multiply", &[dd(0, &[big]), dd(0, &[big])]).unwrap_err();
        assert!(e.to_string().contains("18"), "{e}");
        // Just inside is fine, so the boundary is where it is claimed to be.
        let ok = call("multiply", &[dd(0, &[999_999_999]), dd(0, &[999_999_999])]).unwrap();
        assert_eq!(dtext(&ok), ["999999998000000001"]);
        // Addition overflows too, at the same limit.
        let m = 999_999_999_999_999_999i64;
        assert!(call("plus", &[dd(0, &[m]), dd(0, &[1])]).is_err());
        assert!(call("minus", &[dd(0, &[-m]), dd(0, &[1])]).is_err());
        assert_eq!(dtext(&call("plus", &[dd(0, &[m - 1]), dd(0, &[1])]).unwrap()), [m.to_string()]);
        // Scaling an operand up to meet a wider one can overflow as well.
        assert!(call("plus", &[dd(0, &[m]), dd(2, &[1])]).is_err());
    }

    #[test]
    fn division_is_exact_at_a_decided_scale() {
        // Six fractional digits unless the left operand already wants more.
        let out = call("divide", &[dd(2, &[1000]), dd(2, &[300])]).unwrap();
        assert_eq!(out.ty, DataType::Decimal64(6).to_nullable());
        assert_eq!(dtext(&out), ["3.333333"]);
        // Rounds half away from zero at the last kept digit.
        let out = call("divide", &[dd(0, &[2]), dd(0, &[3])]).unwrap();
        assert_eq!(dtext(&out), ["0.666667"]);
        let out = call("divide", &[dd(0, &[-2]), dd(0, &[3])]).unwrap();
        assert_eq!(dtext(&out), ["-0.666667"]);
        // A wider left operand keeps its own scale.
        let out = call("divide", &[dd(8, &[100_000_000]), ii(&[3])]).unwrap();
        assert_eq!(out.ty, DataType::Decimal64(8).to_nullable());
        assert_eq!(dtext(&out), ["0.33333333"]);
        // /0 nulls the row, same as the float path -- one bad row, not a dead
        // query (module docs).
        let out = call("divide", &[dd(2, &[100, 100]), dd(2, &[0, 200])]).unwrap();
        assert!(out.is_null(0));
        assert_eq!(dtext(&out)[1], "0.500000");
        // A float on either side gives up exactness and returns a float.
        let out = call("divide", &[dd(2, &[100]), ff(&[3.0])]).unwrap();
        assert_eq!(out.ty, DataType::Float64.to_nullable());
    }

    #[test]
    fn intdiv_and_modulo_stay_exact_on_decimals() {
        // `DIV` truncates to a whole number, as it does for every other type.
        let d = call("intDiv", &[dd(2, &[1234]), dd(2, &[500])]).unwrap();
        assert_eq!(d.ty, DataType::Int64.to_nullable());
        assert_eq!(i(&d), vec![2]);
        // `%` keeps the operands' scale: 12.34 % 5 is 2.34, which an Int64
        // result lane could only have called 234 or 2.
        let m = call("modulo", &[dd(2, &[1234]), ii(&[5])]).unwrap();
        assert_eq!(m.ty, DataType::Decimal64(2).to_nullable());
        assert_eq!(dtext(&m), ["2.34"]);
        // Sign follows the dividend, as SQL wants.
        let m = call("modulo", &[dd(2, &[-1234]), ii(&[5])]).unwrap();
        assert_eq!(dtext(&m), ["-2.34"]);
        // Zero divisor still nulls rather than kills.
        let m = call("modulo", &[dd(2, &[1234]), dd(2, &[0])]).unwrap();
        assert!(m.is_null(0));
    }

    #[test]
    fn abs_and_negate_keep_the_scale() {
        for f in ["abs", "negate"] {
            let out = call(f, &[dd(2, &[-1234])]).unwrap();
            assert_eq!(out.ty, DataType::Decimal64(2), "{f}");
            assert_eq!(dtext(&out), ["12.34"], "{f}");
        }
        assert_eq!(dtext(&call("negate", &[dd(3, &[5])]).unwrap()), ["-0.005"]);
        // sign reads the number, and the unit count has the same sign.
        assert_eq!(i(&call("sign", &[dd(4, &[-1, 0, 1])]).unwrap()), vec![-1, 0, 1]);
    }

    /// Every gather-shaped function (`if`, `coalesce`, `ifNull`, `greatest`,
    /// `least`, `nullIf`) shares `coerce_data`, so they all have to rescale --
    /// two `I64` lanes at different scales are not commensurable.
    #[test]
    fn gathering_functions_rescale_rather_than_reinterpret() {
        let g = call("greatest", &[dd(2, &[150]), dd(4, &[14_000])]).unwrap();
        assert_eq!(g.ty, DataType::Decimal64(4));
        assert_eq!(dtext(&g), ["1.5000"]);
        let l = call("least", &[dd(2, &[150]), dd(4, &[14_000])]).unwrap();
        assert_eq!(dtext(&l), ["1.4000"]);

        let c = call("if", &[ui(&[1, 0]), dd(2, &[150, 150]), dd(4, &[1, 1])]).unwrap();
        assert_eq!(dtext(&c), ["1.5000", "0.0001"]);

        let c = call("coalesce", &[nul(dd(2, &[150]), &[0]), dd(4, &[25_000])]).unwrap();
        assert_eq!(dtext(&c), ["2.5000"]);

        let c = call("ifNull", &[nul(dd(2, &[150]), &[0]), ii(&[7])]).unwrap();
        assert_eq!(dtext(&c), ["7.00"]);

        // nullIf compares the *numbers*, so 1.50 and 1.5000 are the same value.
        let n = call("nullIf", &[dd(2, &[150, 150]), dd(4, &[15_000, 1])]).unwrap();
        assert!(n.is_null(0));
        assert!(!n.is_null(1));

        // Mixed with a float, the promotion goes to Float64 and the decimal is
        // descaled on the way -- not handed over as a raw unit count.
        let g = call("greatest", &[dd(2, &[150]), ff(&[1.25])]).unwrap();
        assert_eq!(g.ty, DataType::Float64);
        assert_eq!(fl(&g), vec![1.5]);
    }

    #[test]
    fn conversions_read_the_number_not_the_lane() {
        assert_eq!(st(&call("toString", &[dd(2, &[1234, -5, 0])]).unwrap()), ["12.34", "-0.05", "0.00"]);
        assert_eq!(fl(&call("toFloat64", &[dd(2, &[1234])]).unwrap()), vec![12.34]);
        // Truncating toward zero, like every other number -> int cast here.
        assert_eq!(i(&call("toInt64", &[dd(2, &[1299, -1299])]).unwrap()), vec![12, -12]);
        assert_eq!(u(&call("toUInt64", &[dd(2, &[1299])]).unwrap()), vec![12]);
        // `toDate` of a bare number is a day count (module docs); a decimal
        // descales to that number first rather than offering its unit count.
        let d = call("toDate", &[dd(2, &[1_972_300])]).unwrap();
        assert_eq!(d.ty, DataType::Date);
        assert_eq!(u(&d), vec![19_723]);
        // `concat`/`LIKE` render through the same path, so the point lands.
        assert_eq!(
            st(&call("concat", &[call("toString", &[dd(2, &[500])]).unwrap(), ss(&["!"])]).unwrap()),
            ["5.00!"]
        );
    }

    #[test]
    fn decimal_nulls_and_zero_rows_behave() {
        let out = call("plus", &[nul(dd(2, &[150, 150]), &[1]), dd(2, &[100, 100])]).unwrap();
        assert!(out.ty.is_nullable());
        assert!(!out.is_null(0));
        assert!(out.is_null(1));
        assert_eq!(dtext(&out)[0], "2.50");
        // A zero-row block must not trip the checked arithmetic.
        assert_eq!(call_n("multiply", &[dd(2, &[]), dd(2, &[])], 0).unwrap().len(), 0);
        assert_eq!(call_n("divide", &[dd(2, &[]), dd(2, &[])], 0).unwrap().len(), 0);
    }

    /// The oracle sqlite cannot be.
    ///
    /// `tests/differential.rs` diffs against the sqlite3 CLI, which has no exact
    /// decimal type -- a decimal there is a REAL, which is precisely the thing
    /// this type exists in order not to be. So the check runs against the
    /// *definition* instead: for 20k random operand pairs the engine's unit
    /// count must equal what `i128` arithmetic on scaled integers gives, worked
    /// out here by a different route (explicit powers of ten rather than the
    /// evaluator's hoisted rescale).
    ///
    /// Division is checked as a property rather than by re-deriving it, which
    /// would only restate the implementation: `q` must be the nearest integer to
    /// `N/D`, i.e. `|q*D - N| * 2 <= |D|`. That holds for exactly one `q` except
    /// at a tie, where it holds for two -- and the tie is pinned separately by
    /// `division_is_exact_at_a_decided_scale`.
    #[test]
    fn decimal_arithmetic_matches_exact_scaled_integer_arithmetic() {
        use crate::common::splitmix64;
        let mut seed = 0x5EEDu64;
        let mut next = move || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            splitmix64(seed)
        };
        let p = |k: u8| POW10[k as usize];
        for case in 0..20_000u32 {
            let (sa, sb) = ((next() % 7) as u8, (next() % 7) as u8);
            // Under 10^9 a side, so the product still fits the 18-digit limit
            // and every case is a legal one rather than an overflow test.
            let sign = |x: u64| if x & 1 == 0 { 1i64 } else { -1 };
            let ua = sign(next()) * (next() % 1_000_000_000) as i64;
            let ub = sign(next()) * (next() % 1_000_000_000) as i64;
            let (ca, cb) = (dd(sa, &[ua]), dd(sb, &[ub]));
            let hi = sa.max(sb);
            let (ea, eb) = (ua as i128 * p(hi - sa), ub as i128 * p(hi - sb));

            let got = i(&call("plus", &[ca.clone(), cb.clone()]).unwrap())[0];
            assert_eq!(got as i128, ea + eb, "case {case}: {ua}e-{sa} + {ub}e-{sb}");
            let got = i(&call("minus", &[ca.clone(), cb.clone()]).unwrap())[0];
            assert_eq!(got as i128, ea - eb, "case {case}: {ua}e-{sa} - {ub}e-{sb}");

            let m = call("multiply", &[ca.clone(), cb.clone()]).unwrap();
            assert_eq!(m.ty, DataType::Decimal64(sa + sb));
            assert_eq!(i(&m)[0] as i128, ua as i128 * ub as i128, "case {case}");

            let d = call("divide", &[ca.clone(), cb.clone()]).unwrap();
            let os = sa.max(DIV_MIN_SCALE);
            assert_eq!(d.ty, DataType::Decimal64(os).to_nullable());
            if ub == 0 {
                assert!(d.is_null(0), "case {case}: /0 must null the row");
            } else {
                let (n, dv) = (ua as i128 * p(sb + os - sa), ub as i128);
                let q = i(&d)[0] as i128;
                assert!(
                    (q * dv - n).abs() * 2 <= dv.abs(),
                    "case {case}: {ua}e-{sa} / {ub}e-{sb} gave {q}e-{os}, not nearest"
                );
            }

            // And the text round trip: rendering a decimal and casting the text
            // back must be the identity, or `toString`/`CAST(... AS String)`
            // is not the exact escape hatch it is documented to be.
            let txt = Value::Decimal(ua, sa).render_plain();
            assert_eq!(
                Value::str(txt.as_str()).cast_to(&DataType::Decimal64(sa)).unwrap(),
                Value::Decimal(ua, sa),
                "case {case}: '{txt}' did not round trip at scale {sa}"
            );
        }
    }

    /// A decimal is not an integer, so the functions that index into a string
    /// with one must keep refusing it -- their argument is a position, and a
    /// unit count is not one.
    #[test]
    fn decimals_are_refused_where_an_integer_is_meant() {
        let d = DataType::Decimal64(2);
        assert!(ret_of("substring", &[DataType::String, d.clone()]).is_err());
        assert!(ret_of("repeat", &[DataType::String, d.clone()]).is_err());
        assert!(ret_of("addDays", &[DataType::Date, d.clone()]).is_err());
        assert!(ret_of("leftPad", &[DataType::String, d.clone()]).is_err());
        // ...but every numeric gate still lets it through.
        assert!(ret_of("abs", &[d.clone()]).is_ok());
        assert!(ret_of("round", &[d.clone()]).is_ok());
        assert!(ret_of("sqrt", &[d]).is_ok());
    }

    // ------------------------------------------------------------ ret gates

    #[test]
    fn ret_rejects_wrong_argument_types() {
        assert!(matches!(ret_of("abs", &[DataType::String]), Err(Error::Bind(_))));
        assert!(matches!(ret_of("plus", &[DataType::String, DataType::Int64]), Err(Error::Bind(_))));
        assert!(matches!(ret_of("length", &[DataType::Int64]), Err(Error::Bind(_))));
        assert!(matches!(ret_of("concat", &[DataType::Int64]), Err(Error::Bind(_))));
        assert!(matches!(
            ret_of("substring", &[DataType::String, DataType::String]),
            Err(Error::Bind(_))
        ));
        assert!(matches!(ret_of("toYear", &[DataType::Int64]), Err(Error::Bind(_))));
        assert!(matches!(
            ret_of("dateDiff", &[DataType::String, DataType::Int64, DataType::Date]),
            Err(Error::Bind(_))
        ));
    }

    #[test]
    fn ret_rejects_wrong_arity() {
        assert!(ret_of("abs", &[]).is_err());
        assert!(ret_of("abs", &[DataType::Int64, DataType::Int64]).is_err());
        assert!(ret_of("now", &[DataType::Int64]).is_err());
        assert!(ret_of("substring", &[DataType::String]).is_err());
        assert!(ret_of(
            "substring",
            &[DataType::String, DataType::Int64, DataType::Int64, DataType::Int64]
        )
        .is_err());
        assert!(ret_of("coalesce", &[]).is_err());
    }

    #[test]
    fn ret_propagates_nullability() {
        let n = DataType::Int64.to_nullable();
        assert!(ret_of("abs", &[n.clone()]).unwrap().is_nullable());
        assert!(ret_of("plus", &[n.clone(), DataType::Int64]).unwrap().is_nullable());
        assert!(!ret_of("plus", &[DataType::Int64, DataType::Int64]).unwrap().is_nullable());
        // isNull observes nulls, so it never produces one
        assert!(!ret_of("isNull", &[n]).unwrap().is_nullable());
        // string parsing can fail, so the conversion is Nullable regardless
        assert!(ret_of("toInt64", &[DataType::String]).unwrap().is_nullable());
        assert!(!ret_of("toInt64", &[DataType::UInt8]).unwrap().is_nullable());
    }

    // -------------------------------------------------------------- strings

    #[test]
    fn length_counts_bytes_and_utf8_counts_characters() {
        assert_eq!(u(&call("length", &[ss(&["héllo", ""])]).unwrap()), vec![6, 0]);
        assert_eq!(u(&call("lengthUTF8", &[ss(&["héllo", ""])]).unwrap()), vec![5, 0]);
        assert_eq!(u(&call("char_length", &[ss(&["héllo"])]).unwrap()), vec![5]);
    }

    #[test]
    fn case_mapping_ascii_and_unicode() {
        assert_eq!(st(&call("lower", &[ss(&["ABC", "abc"])]).unwrap()), vec!["abc", "abc"]);
        assert_eq!(st(&call("upper", &[ss(&["abc"])]).unwrap()), vec!["ABC"]);
        // the ASCII variants deliberately leave non-ASCII alone
        assert_eq!(st(&call("lower", &[ss(&["ÄBC"])]).unwrap()), vec!["Äbc"]);
        assert_eq!(st(&call("lowerUTF8", &[ss(&["ÄBC"])]).unwrap()), vec!["äbc"]);
        assert_eq!(st(&call("upperUTF8", &[ss(&["straße"])]).unwrap()), vec!["STRASSE"]);
    }

    #[test]
    fn concat_is_variadic() {
        let out = call("concat", &[ss(&["a", "x"]), ss(&["b", "y"]), ss(&["c", "z"])]).unwrap();
        assert_eq!(st(&out), vec!["abc", "xyz"]);
        assert_eq!(st(&call("concat", &[ss(&["solo"])]).unwrap()), vec!["solo"]);
    }

    #[test]
    fn substring_follows_clickhouse_offsets() {
        let s = ss(&["hello", "hello", "hello", "hello"]);
        let out = call("substring", &[s.clone(), ii(&[2, 1, 0, 9])]).unwrap();
        assert_eq!(st(&out), vec!["ello", "hello", "hello", ""]);
        let out = call("substring", &[s.clone(), ii(&[2, 1, 2, 1]), ii(&[2, 0, 99, -1])]).unwrap();
        assert_eq!(st(&out), vec!["el", "", "ello", "hell"]);
    }

    #[test]
    fn substring_handles_negative_offsets_and_multibyte() {
        let out = call("substring", &[ss(&["hello", "héllo"]), ii(&[-3, 2]), ii(&[2, 2])]).unwrap();
        assert_eq!(st(&out), vec!["ll", "él"]);
    }

    #[test]
    fn trim_family() {
        let s = ss(&["  a  ", "b"]);
        assert_eq!(st(&call("trim", &[s.clone()]).unwrap()), vec!["a", "b"]);
        assert_eq!(st(&call("trimLeft", &[s.clone()]).unwrap()), vec!["a  ", "b"]);
        assert_eq!(st(&call("trimRight", &[s]).unwrap()), vec!["  a", "b"]);
    }

    #[test]
    fn replace_all_including_the_empty_needle() {
        let out = call(
            "replaceAll",
            &[ss(&["aXbXc", "abc"]), ss(&["X", ""]), ss(&["-", "!"])],
        )
        .unwrap();
        assert_eq!(st(&out), vec!["a-b-c", "abc"]);
    }

    #[test]
    fn starts_and_ends_with() {
        let out = call("startsWith", &[ss(&["hello", "hello"]), ss(&["he", "lo"])]).unwrap();
        assert_eq!(out.ty, DataType::Bool);
        assert_eq!(u(&out), vec![1, 0]);
        let out = call("endsWith", &[ss(&["hello", "hello"]), ss(&["lo", "he"])]).unwrap();
        assert_eq!(u(&out), vec![1, 0]);
    }

    #[test]
    fn position_is_one_based_and_character_indexed() {
        let out = call(
            "position",
            &[ss(&["hello", "héllo", "abc", "abc"]), ss(&["ll", "llo", "z", ""])],
        )
        .unwrap();
        assert_eq!(u(&out), vec![3, 3, 0, 1]);
    }

    #[test]
    fn split_by_char_returns_only_the_first_field() {
        let out = call(
            "splitByChar",
            &[ss(&[",", ",", ",", ""]), ss(&["a,b,c", "abc", ",x", "a,b"])],
        )
        .unwrap();
        assert_eq!(st(&out), vec!["a", "abc", "", "a,b"]);
    }

    #[test]
    fn reverse_and_repeat_work_on_characters() {
        assert_eq!(st(&call("reverse", &[ss(&["abc", "héllo"])]).unwrap()), vec!["cba", "olléh"]);
        let out = call("repeat", &[ss(&["ab", "ab", "ab"]), ii(&[3, 0, -1])]).unwrap();
        assert_eq!(st(&out), vec!["ababab", "", ""]);
    }

    #[test]
    fn repeat_refuses_to_build_an_enormous_string() {
        let e = call("repeat", &[ss(&["ab"]), ii(&[1 << 30])]).unwrap_err();
        assert!(matches!(e, Error::Exec(_)), "{e}");
    }

    #[test]
    fn pads_fill_and_truncate() {
        let out = call("leftPad", &[ss(&["ab", "ab", "abcdef"]), ii(&[5, 5, 3])]).unwrap();
        assert_eq!(st(&out), vec!["   ab", "   ab", "abc"]);
        let out = call("leftPad", &[ss(&["ab"]), ii(&[5]), ss(&["xy"])]).unwrap();
        assert_eq!(st(&out), vec!["xyxab"]);
        let out = call("rightPad", &[ss(&["ab"]), ii(&[5]), ss(&["*"])]).unwrap();
        assert_eq!(st(&out), vec!["ab***"]);
    }

    // ------------------------------------------------------------- matching

    #[test]
    fn like_handles_the_basic_wildcards() {
        for (s, p) in [("hello", "hello"), ("hello", "h%o"), ("hello", "h_llo"), ("hello", "%ell%")] {
            assert!(like_match(s, p, false), "{s} LIKE {p}");
        }
        for (s, p) in [("hello", "hell"), ("hello", "h_lo"), ("hello", "ello")] {
            assert!(!like_match(s, p, false), "{s} NOT LIKE {p}");
        }
    }

    #[test]
    fn like_handles_degenerate_patterns() {
        assert!(like_match("", "", false));
        assert!(!like_match("a", "", false));
        assert!(like_match("", "%", false));
        assert!(like_match("", "%%", false));
        assert!(like_match("abc", "%%", false));
        assert!(like_match("abc", "%_%", false));
        assert!(!like_match("", "%_%", false));
        assert!(like_match("a", "%_%", false));
        assert!(like_match("abc", "a%", false));
        assert!(like_match("abc", "%c", false));
        assert!(like_match("abc", "%b%", false));
        assert!(!like_match("abc", "%b", false));
        assert!(!like_match("ab", "%b_", false));
    }

    #[test]
    fn like_backtracking_does_not_explode() {
        // the pattern that kills naive backtrackers; the two-pointer form is
        // linear-ish and, more importantly, terminates
        assert!(like_match("aaa", "%a%a%a%", false));
        assert!(!like_match("aa", "%a%a%a%", false));
        assert!(like_match(&"a".repeat(2000), &"%a".repeat(20), false));
        assert!(!like_match(&"a".repeat(200), &format!("{}b", "%a".repeat(20)), false));
    }

    #[test]
    fn like_supports_backslash_escapes() {
        assert!(like_match("50%", r"50\%", false));
        assert!(!like_match("50x", r"50\%", false));
        assert!(like_match("a_b", r"a\_b", false));
        assert!(!like_match("axb", r"a\_b", false));
        assert!(like_match(r"a\b", r"a\\b", false));
        // a trailing backslash is a literal backslash
        assert!(like_match(r"a\", r"a\", false));
    }

    #[test]
    fn like_underscore_consumes_a_whole_character() {
        assert!(like_match("héllo", "h_llo", false));
        assert!(!like_match("héllo", "h__llo", false));
        assert!(like_match("日本語", "_本_", false));
    }

    #[test]
    fn ilike_folds_ascii_case_only() {
        assert!(like_match("HeLLo", "hello", true));
        assert!(like_match("HELLO", "h%O", true));
        assert!(!like_match("HeLLo", "hello", false));
        // non-ASCII is left alone, as documented
        assert!(!like_match("ÄBC", "äbc", true));
    }

    #[test]
    fn like_family_vectorizes_and_inverts() {
        let s = ss(&["hello", "world"]);
        let p = ss(&["h%", "h%"]);
        assert_eq!(u(&call("like", &[s.clone(), p.clone()]).unwrap()), vec![1, 0]);
        assert_eq!(u(&call("notLike", &[s.clone(), p.clone()]).unwrap()), vec![0, 1]);
        let up = ss(&["H%", "H%"]);
        assert_eq!(u(&call("ilike", &[s.clone(), up.clone()]).unwrap()), vec![1, 0]);
        assert_eq!(u(&call("notILike", &[s, up]).unwrap()), vec![0, 1]);
    }

    // ------------------------------------------------------------ type conv

    #[test]
    fn tostring_renders_each_physical_kind() {
        assert_eq!(st(&call("toString", &[ii(&[-5])]).unwrap()), vec!["-5"]);
        assert_eq!(st(&call("toString", &[ui(&[5])]).unwrap()), vec!["5"]);
        assert_eq!(st(&call("toString", &[ff(&[1.0, 1.5])]).unwrap()), vec!["1", "1.5"]);
        assert_eq!(st(&call("toString", &[ss(&["x"])]).unwrap()), vec!["x"]);
        assert_eq!(st(&call("toString", &[date(&[19_723])]).unwrap()), vec!["2024-01-01"]);
        assert_eq!(
            st(&call("toString", &[dt(&[0])]).unwrap()),
            vec!["1970-01-01 00:00:00"]
        );
        assert_eq!(
            st(&call("toString", &[Column::bools(vec![1, 0])]).unwrap()),
            vec!["true", "false"]
        );
    }

    #[test]
    fn integer_conversions_truncate_rather_than_raise() {
        let out = call("toUInt32", &[ui(&[(1u64 << 33) + 7])]).unwrap();
        assert_eq!(out.ty, DataType::UInt32);
        assert_eq!(u(&out), vec![7]);
        let out = call("toInt32", &[ii(&[i32::MAX as i64 + 1])]).unwrap();
        assert_eq!(i(&out), vec![i32::MIN as i64]);
        // float -> int saturates, per Rust's `as` rules
        assert_eq!(i(&call("toInt64", &[ff(&[1e30, 3.9])]).unwrap()), vec![i64::MAX, 3]);
    }

    #[test]
    fn string_conversions_null_on_a_parse_failure() {
        let out = call("toInt64", &[ss(&["42", " -7 ", "abc", ""])]).unwrap();
        assert!(out.ty.is_nullable());
        assert_eq!(i(&out)[0..2], [42, -7]);
        assert!(!out.is_null(0) && !out.is_null(1));
        assert!(out.is_null(2) && out.is_null(3));

        let out = call("toFloat64", &[ss(&["1.5", "nope"])]).unwrap();
        assert_eq!(fl(&out)[0], 1.5);
        assert!(out.is_null(1));

        // integers parse through the float fallback too
        assert_eq!(i(&call("toInt64", &[ss(&["3.9"])]).unwrap()), vec![3]);
    }

    #[test]
    fn to_date_and_to_datetime_accept_strings_and_temporals() {
        let out = call("toDate", &[ss(&["2024-01-01", "2024-01-15 10:00:00", "junk"])]).unwrap();
        assert_eq!(u(&out)[0..2], [19_723, 19_737]);
        assert!(out.is_null(2));

        let ts = parse_datetime("2024-01-15 13:45:30").unwrap();
        let out = call("toDate", &[dt(&[ts])]).unwrap();
        assert_eq!(out.ty, DataType::Date);
        assert_eq!(u(&out), vec![19_737]);

        let out = call("toDateTime", &[date(&[19_723])]).unwrap();
        assert_eq!(out.ty, DataType::DateTime);
        assert_eq!(u(&out), vec![19_723 * 86_400]);

        let out = call("toDateTime", &[ss(&["2024-01-15 13:45:30"])]).unwrap();
        assert_eq!(u(&out)[0] as i64, ts);
    }

    #[test]
    fn datetime_round_trips_through_the_unsigned_lane() {
        // pre-epoch timestamps are stored as a wrapped u64 and must survive
        let out = call("toDateTime", &[ii(&[-86_400])]).unwrap();
        assert_eq!(out.value(0), Value::DateTime(-86_400));
        assert_eq!(st(&call("toString", &[out]).unwrap()), vec!["1969-12-31 00:00:00"]);
    }

    /// One row per input class. The invariant under test is not "toDate is
    /// right" but the stronger "toDate is right *or* loud": `Date` is an
    /// unsigned 32-bit day count, so every day number it cannot hold used to
    /// truncate into a different, plausible-looking date -- and toDate output
    /// is written to parts, so a wrong date here is permanent.
    #[test]
    fn to_date_is_correct_or_loud_for_every_input_class() {
        #[derive(Debug, PartialEq)]
        enum Want {
            Day(u64),
            Null,
            Raises,
        }
        use Want::*;

        let noon = parse_datetime("2024-01-15 13:45:30").unwrap();
        let post_2038 = parse_datetime("2100-06-01 13:45:30").unwrap();
        let cases: Vec<(&str, Column, Want)> = vec![
            // -- strings: `ret` types these Nullable, so NULL is available
            ("bare YYYY-MM-DD", ss(&["2024-01-01"]), Day(19_723)),
            ("datetime string drops the clock", ss(&["2024-01-15 10:00:00"]), Day(19_737)),
            ("the epoch itself", ss(&["1970-01-01"]), Day(0)),
            ("unparseable", ss(&["junk"]), Null),
            ("parseable but pre-epoch", ss(&["1969-12-31"]), Null),
            ("parseable but far past the lane", ss(&["999999999-01-01"]), Null),
            // -- temporals
            ("Date is the identity", date(&[19_723]), Day(19_723)),
            ("DateTime floors to its day", dt(&[noon]), Day(19_737)),
            ("exactly midnight", dt(&[19_723 * 86_400]), Day(19_723)),
            ("post-2038 DateTime", dt(&[post_2038]), Day((post_2038 / 86_400) as u64)),
            // one second before the epoch is day -1, which used to arrive as
            // 2^64-1 and render as a year-11-million date
            ("one second pre-epoch", dt(&[-1]), Raises),
            ("a whole day pre-epoch", dt(&[-86_400]), Raises),
            ("DateTime beyond the Date lane", dt(&[i64::MAX]), Raises),
            // -- bare numbers are day counts (see the module header)
            ("a positive integer", ii(&[19_723]), Day(19_723)),
            ("zero", ii(&[0]), Day(0)),
            ("a negative integer", ii(&[-1]), Raises),
            ("the last representable day", ui(&[u32::MAX as u64]), Day(u32::MAX as u64)),
            ("one past the last day", ui(&[u32::MAX as u64 + 1]), Raises),
            ("u64::MAX", ui(&[u64::MAX]), Raises),
            ("a float truncates", ff(&[19_723.9]), Day(19_723)),
            ("a negative float", ff(&[-1.5]), Raises),
            ("NaN has no day number", ff(&[f64::NAN]), Raises),
            ("infinity has no day number", ff(&[f64::INFINITY]), Raises),
        ];

        for (what, col, want) in cases {
            let got = match call("toDate", &[col]) {
                Err(_) => Raises,
                Ok(out) if out.is_null(0) => Null,
                Ok(out) => Day(u(&out)[0]),
            };
            assert_eq!(got, want, "toDate: {what}");
        }

        // and the post-2038 day number is the calendar date, not an i32 wrap
        let out = call("toDate", &[dt(&[post_2038])]).unwrap();
        assert_eq!(st(&call("toString", &[out]).unwrap()), vec!["2100-06-01"]);
    }

    #[test]
    fn to_date_leaves_null_rows_alone_whatever_their_lane_holds() {
        // The payload under a NULL is arbitrary, so gating it would raise on a
        // row that carries no value at all.
        let out = call("toDate", &[nul(ii(&[-999_999, 19_723]), &[0])]).unwrap();
        assert!(out.is_null(0));
        assert_eq!(u(&out)[1], 19_723);

        let out = call("toDate", &[nul(dt(&[-86_400, 0]), &[0])]).unwrap();
        assert!(out.is_null(0));
        assert_eq!(u(&out)[1], 0);
    }

    #[test]
    fn to_date_never_emits_a_lane_the_date_type_would_truncate() {
        // `Column::value` reads a Date back as `lane as u32`, so any lane above
        // u32::MAX is silently a *different* date once it round-trips. Sweep
        // the paths that could produce one and require every survivor to be
        // byte-identical after the truncating read.
        let sweep = [
            ii(&[-1, 0, 1]),
            ii(&[0, 19_723, u32::MAX as i64]),
            dt(&[-1, 0, 86_400]),
            dt(&[0, i64::MAX, 86_400]),
            ff(&[-1.5, 2.0, 3.0]),
            ff(&[0.0, 19_723.9, 1e30]),
            date(&[0, 19_723, u32::MAX as i64]),
            ss(&["2024-01-01", "1969-12-31", "junk"]),
        ];
        for c in sweep {
            match call("toDate", &[c]) {
                Err(e) => assert!(
                    e.to_string().contains("Date range"),
                    "the error must name the range, got: {e}"
                ),
                Ok(out) => {
                    for (i, &lane) in u(&out).iter().enumerate() {
                        if out.is_null(i) {
                            continue;
                        }
                        assert!(lane <= u32::MAX as u64, "row {i} lane {lane} truncates");
                        assert_eq!(out.value(i), Value::Date(lane as u32));
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------- dates

    #[test]
    fn date_part_extractors() {
        let d = date(&[19_723]); // 2024-01-01, a Monday
        assert_eq!(u(&call("toYear", &[d.clone()]).unwrap()), vec![2024]);
        assert_eq!(u(&call("toMonth", &[d.clone()]).unwrap()), vec![1]);
        assert_eq!(u(&call("toDayOfMonth", &[d.clone()]).unwrap()), vec![1]);
        assert_eq!(u(&call("toDayOfYear", &[d.clone()]).unwrap()), vec![1]);
        assert_eq!(u(&call("toQuarter", &[d]).unwrap()), vec![1]);
    }

    #[test]
    fn day_of_week_counts_monday_as_one() {
        // 2024-01-01 Mon .. 2024-01-07 Sun
        let d = date(&[19_723, 19_724, 19_727, 19_729]);
        assert_eq!(u(&call("toDayOfWeek", &[d]).unwrap()), vec![1, 2, 5, 7]);
        // and the epoch itself was a Thursday
        assert_eq!(u(&call("toDayOfWeek", &[date(&[0])]).unwrap()), vec![4]);
    }

    #[test]
    fn time_part_extractors() {
        let ts = parse_datetime("2024-01-15 13:45:30").unwrap();
        let c = dt(&[ts]);
        assert_eq!(u(&call("toHour", &[c.clone()]).unwrap()), vec![13]);
        assert_eq!(u(&call("toMinute", &[c.clone()]).unwrap()), vec![45]);
        assert_eq!(u(&call("toSecond", &[c]).unwrap()), vec![30]);
        // a Date is midnight
        assert_eq!(u(&call("toHour", &[date(&[19_723])]).unwrap()), vec![0]);
    }

    #[test]
    fn start_of_truncations() {
        let ts = parse_datetime("2024-01-15 13:45:30").unwrap();
        let c = dt(&[ts]);
        let day = call("toStartOfDay", &[c.clone()]).unwrap();
        assert_eq!(day.ty, DataType::DateTime);
        assert_eq!(u(&day)[0] as i64, parse_datetime("2024-01-15 00:00:00").unwrap());
        assert_eq!(
            u(&call("toStartOfHour", &[c.clone()]).unwrap())[0] as i64,
            parse_datetime("2024-01-15 13:00:00").unwrap()
        );
        assert_eq!(
            u(&call("toStartOfMinute", &[c.clone()]).unwrap())[0] as i64,
            parse_datetime("2024-01-15 13:45:00").unwrap()
        );

        let m = call("toStartOfMonth", &[c.clone()]).unwrap();
        assert_eq!(m.ty, DataType::Date);
        assert_eq!(u(&m), vec![19_723]);
        assert_eq!(u(&call("toStartOfYear", &[c.clone()]).unwrap()), vec![19_723]);
        assert_eq!(u(&call("toStartOfQuarter", &[c.clone()]).unwrap()), vec![19_723]);
        // 2024-01-15 is a Monday already
        assert_eq!(u(&call("toMonday", &[c]).unwrap()), vec![19_737]);
        // and a Thursday rewinds to the Monday before it
        assert_eq!(u(&call("toMonday", &[date(&[19_726])]).unwrap()), vec![19_723]);
    }

    #[test]
    fn to_unix_timestamp_normalizes_both_temporal_types() {
        assert_eq!(i(&call("toUnixTimestamp", &[date(&[1])]).unwrap()), vec![86_400]);
        assert_eq!(i(&call("toUnixTimestamp", &[dt(&[1_700_000_000])]).unwrap()), vec![
            1_700_000_000
        ]);
    }

    #[test]
    fn now_and_today_broadcast_one_reading() {
        let n = call_n("now", &[], 3).unwrap();
        assert_eq!(n.ty, DataType::DateTime);
        let v = u(&n);
        assert_eq!(v.len(), 3);
        assert!(v[0] == v[1] && v[1] == v[2], "now() must not drift within a block");
        assert!(v[0] > 1_600_000_000, "clock looks wrong: {v:?}");

        let t = call_n("today", &[], 2).unwrap();
        assert_eq!(t.ty, DataType::Date);
        assert_eq!(u(&t)[0], v[0] / 86_400);
    }

    #[test]
    fn date_diff_counts_boundary_crossings() {
        let a = date(&[19_723; 6]); // 2024-01-01
        let b = date(&[19_754; 6]); // 2024-02-01
        let units = ss(&["day", "week", "month", "quarter", "year", "hour"]);
        let out = call("dateDiff", &[units, a, b]).unwrap();
        assert_eq!(i(&out), vec![31, 4, 1, 0, 0, 31 * 24]);
    }

    #[test]
    fn date_diff_crosses_a_year_boundary_by_calendar_not_duration() {
        // one second apart, but a different year
        let a = dt(&[parse_datetime("2023-12-31 23:59:59").unwrap()]);
        let b = dt(&[parse_datetime("2024-01-01 00:00:00").unwrap()]);
        assert_eq!(i(&call("dateDiff", &[ss(&["year"]), a.clone(), b.clone()]).unwrap()), vec![1]);
        assert_eq!(i(&call("dateDiff", &[ss(&["second"]), a, b]).unwrap()), vec![1]);
    }

    #[test]
    fn date_diff_rejects_an_unknown_unit() {
        let e = call("dateDiff", &[ss(&["fortnight"]), date(&[0]), date(&[1])]).unwrap_err();
        assert!(matches!(e, Error::Exec(_)), "{e}");
    }

    #[test]
    fn add_months_clamps_the_day_of_month() {
        let jan31 = date(&[days_from_civil(2024, 1, 31)]);
        let out = call("addMonths", &[jan31.clone(), ii(&[1])]).unwrap();
        assert_eq!(out.ty, DataType::Date);
        assert_eq!(u(&out)[0] as i64, days_from_civil(2024, 2, 29));
        // and a non-leap year lands on the 28th
        let out = call("addYears", &[jan31, ii(&[0])]).unwrap();
        assert_eq!(u(&out)[0] as i64, days_from_civil(2024, 1, 31));
        let feb29 = date(&[days_from_civil(2024, 2, 29)]);
        let out = call("addYears", &[feb29, ii(&[1])]).unwrap();
        assert_eq!(u(&out)[0] as i64, days_from_civil(2025, 2, 28));
    }

    #[test]
    fn add_and_subtract_are_mirror_images() {
        let d = date(&[19_723]);
        assert_eq!(u(&call("addDays", &[d.clone(), ii(&[10])]).unwrap()), vec![19_733]);
        assert_eq!(u(&call("subtractDays", &[d.clone(), ii(&[10])]).unwrap()), vec![19_713]);
        assert_eq!(u(&call("addWeeks", &[d.clone(), ii(&[1])]).unwrap()), vec![19_730]);
        let q = call("subtractQuarters", &[d.clone(), ii(&[1])]).unwrap();
        assert_eq!(u(&q)[0] as i64, days_from_civil(2023, 10, 1));
    }

    #[test]
    fn sub_day_shifts_widen_a_date_to_a_datetime() {
        let out = call("addHours", &[date(&[19_723]), ii(&[5])]).unwrap();
        assert_eq!(out.ty, DataType::DateTime);
        assert_eq!(u(&out)[0] as i64, 19_723 * 86_400 + 5 * 3_600);
        // whole-day units keep the Date type
        assert_eq!(
            call("addDays", &[date(&[19_723]), ii(&[1])]).unwrap().ty,
            DataType::Date
        );
    }

    // ---------------------------------------------------------------- nulls

    #[test]
    fn is_null_and_is_not_null_never_produce_nulls() {
        let c = nul(ii(&[1, 2, 3]), &[1]);
        let n = call("isNull", &[c.clone()]).unwrap();
        assert_eq!(n.ty, DataType::Bool);
        assert!(!n.has_nulls());
        assert_eq!(u(&n), vec![0, 1, 0]);
        assert_eq!(u(&call("isNotNull", &[c]).unwrap()), vec![1, 0, 1]);
        // a column with no mask at all
        assert_eq!(u(&call("isNull", &[ii(&[1, 2])]).unwrap()), vec![0, 0]);
    }

    #[test]
    fn if_null_substitutes_the_fallback() {
        let a = nul(ii(&[1, 0]), &[1]);
        let out = call("ifNull", &[a, ii(&[9, 9])]).unwrap();
        assert!(!out.ty.is_nullable(), "a non-null fallback removes nullability");
        assert_eq!(i(&out), vec![1, 9]);
    }

    #[test]
    fn coalesce_picks_the_first_non_null() {
        let a = nul(ii(&[1, 0, 0]), &[1, 2]);
        let b = nul(ii(&[0, 2, 0]), &[0, 2]);
        let c = nul(ii(&[0, 0, 0]), &[0, 1, 2]);
        let out = call("coalesce", &[a, b, c]).unwrap();
        assert_eq!(i(&out)[0..2], [1, 2]);
        assert!(out.is_null(2), "every branch was NULL");
        assert!(out.ty.is_nullable());
    }

    #[test]
    fn coalesce_with_a_non_nullable_tail_is_not_nullable() {
        let a = nul(ii(&[1, 0]), &[1]);
        let out = call("coalesce", &[a, ii(&[7, 7])]).unwrap();
        assert!(!out.ty.is_nullable());
        assert_eq!(i(&out), vec![1, 7]);
    }

    #[test]
    fn null_if_nulls_the_matching_rows() {
        let out = call("nullIf", &[ii(&[1, 2, 3]), ii(&[1, 9, 3])]).unwrap();
        assert!(out.is_null(0) && out.is_null(2));
        assert_eq!(i(&out)[1], 2);
        // a NULL operand is never "equal"
        let out = call("nullIf", &[nul(ii(&[1]), &[0]), ii(&[1])]).unwrap();
        assert!(out.is_null(0));
    }

    #[test]
    fn assume_not_null_strips_the_mask_and_the_wrapper() {
        let c = nul(ii(&[1, 0]), &[1]);
        let out = call("assumeNotNull", &[c]).unwrap();
        assert_eq!(out.ty, DataType::Int64);
        assert!(!out.has_nulls());
        assert_eq!(i(&out), vec![1, 0]);
    }

    // ---------------------------------------------------------------- logic

    #[test]
    fn if_selects_per_row_and_treats_null_as_false() {
        let cond = nul(Column::bools(vec![1, 0, 0]), &[2]);
        let out = call("if", &[cond, ii(&[10, 10, 10]), ii(&[20, 20, 20])]).unwrap();
        assert_eq!(i(&out), vec![10, 20, 20]);
        assert!(!out.has_nulls(), "a NULL condition must not poison the row");
    }

    #[test]
    fn if_inherits_nulls_only_from_the_selected_branch() {
        let cond = Column::bools(vec![1, 0]);
        let a = nul(ii(&[0, 5]), &[0]); // NULL, 5
        let b = nul(ii(&[7, 0]), &[1]); // 7, NULL
        let out = call("if", &[cond, a, b]).unwrap();
        assert!(out.is_null(0) && out.is_null(1));
    }

    #[test]
    fn not_inverts_truthiness_and_propagates_nulls() {
        let out = call("not", &[nul(ii(&[0, 5, 0]), &[2])]).unwrap();
        assert_eq!(u(&out)[0..2], [1, 0]);
        assert!(out.is_null(2));
        // strings are truthy when non-empty
        assert_eq!(u(&call("not", &[ss(&["", "x"])]).unwrap()), vec![1, 0]);
    }

    #[test]
    fn and_or_implement_three_valued_logic() {
        let a = nul(Column::bools(vec![1, 1, 0]), &[0]); // NULL, true, false
        let b = Column::bools(vec![0, 1, 1]); // false, true, true

        let and = call("and", &[a.clone(), b.clone()]).unwrap();
        assert_eq!(u(&and), vec![0, 1, 0]);
        assert!(!and.is_null(0), "false dominates an unknown");

        let or = call("or", &[a, b]).unwrap();
        assert!(or.is_null(0), "NULL OR false is unknown");
        assert_eq!(u(&or)[1..], [1, 1]);
    }

    /// The complete 3x3 table, spelled out rather than spot-checked, because
    /// this is the *reference* the planner's constant folder has to reproduce:
    /// `x AND NULL` must not mean one thing when the optimizer collapses it to
    /// a literal and another thing when the executor evaluates it. Only the
    /// dominant value (`false` for AND, `true` for OR) may absorb an unknown;
    /// every other row that touches a NULL is NULL.
    #[test]
    fn and_or_match_the_full_three_valued_truth_table() {
        const T: [Option<bool>; 3] = [None, Some(true), Some(false)];
        let (mut lv, mut rv, mut ln, mut rn) = (vec![], vec![], vec![], vec![]);
        for (k, a) in T.iter().enumerate() {
            for (j, b) in T.iter().enumerate() {
                let idx = k * 3 + j;
                lv.push(a.unwrap_or(false) as u64);
                rv.push(b.unwrap_or(false) as u64);
                if a.is_none() {
                    ln.push(idx);
                }
                if b.is_none() {
                    rn.push(idx);
                }
            }
        }
        let l = nul(Column::bools(lv), &ln);
        let r = nul(Column::bools(rv), &rn);
        let and = call("and", &[l.clone(), r.clone()]).unwrap();
        let or = call("or", &[l, r]).unwrap();
        let (au, ou) = (u(&and), u(&or));

        for (k, a) in T.iter().enumerate() {
            for (j, b) in T.iter().enumerate() {
                let idx = k * 3 + j;
                let read = |c: &Column, bits: &[u64]| {
                    (!c.is_null(idx)).then(|| bits[idx] != 0)
                };
                let want_and = if *a == Some(false) || *b == Some(false) {
                    Some(false)
                } else if a.is_none() || b.is_none() {
                    None
                } else {
                    Some(true)
                };
                let want_or = if *a == Some(true) || *b == Some(true) {
                    Some(true)
                } else if a.is_none() || b.is_none() {
                    None
                } else {
                    Some(false)
                };
                assert_eq!(read(&and, &au), want_and, "{a:?} AND {b:?}");
                assert_eq!(read(&or, &ou), want_or, "{a:?} OR {b:?}");
            }
        }
    }

    #[test]
    fn and_is_variadic() {
        let out = call(
            "and",
            &[Column::bools(vec![1, 1]), Column::bools(vec![1, 1]), Column::bools(vec![1, 0])],
        )
        .unwrap();
        assert_eq!(u(&out), vec![1, 0]);
    }

    #[test]
    fn xor_propagates_nulls_normally() {
        let out = call("xor", &[nul(Column::bools(vec![1, 0, 1]), &[2]), Column::bools(vec![1, 1, 1])])
            .unwrap();
        assert_eq!(u(&out)[0..2], [0, 1]);
        assert!(out.is_null(2));
    }

    // ----------------------------------------------------------------- misc

    #[test]
    fn city_hash_is_stable_and_order_sensitive() {
        let h1 = u(&call("cityHash64", &[ss(&["abc"])]).unwrap());
        let h2 = u(&call("cityHash64", &[ss(&["abc"])]).unwrap());
        assert_eq!(h1, h2, "hashing must be deterministic");
        let h3 = u(&call("cityHash64", &[ss(&["abd"])]).unwrap());
        assert_ne!(h1, h3);
        // multiple arguments mix, and swapping them changes the digest
        let ab = u(&call("cityHash64", &[ss(&["a"]), ss(&["b"])]).unwrap());
        let ba = u(&call("cityHash64", &[ss(&["b"]), ss(&["a"])]).unwrap());
        assert_ne!(ab, ba);
        assert_ne!(ab, u(&call("cityHash64", &[ss(&["ab"])]).unwrap()));
    }

    #[test]
    fn rand_fills_the_block_with_distinct_values() {
        let r = call_n("rand", &[], 64).unwrap();
        assert_eq!(r.ty, DataType::UInt32);
        let v = u(&r);
        assert!(v.iter().all(|&x| x <= u32::MAX as u64));
        let uniq: std::collections::HashSet<u64> = v.iter().copied().collect();
        assert!(uniq.len() > 60, "splitmix should not collide this hard");
        // successive calls advance the counter
        assert_ne!(v, u(&call_n("rand", &[], 64).unwrap()));
        assert_eq!(call_n("rand64", &[], 4).unwrap().ty, DataType::UInt64);
    }

    // ---------------------------------------------- cross-cutting behaviour

    #[test]
    fn null_propagation_holds_across_the_strict_functions() {
        // one nullable input, row 1 NULL, checked through every family
        let n_i = nul(ii(&[4, 0, -2]), &[1]);
        let n_f = nul(ff(&[4.0, 0.0, 2.0]), &[1]);
        let n_s = nul(ss(&["ab", "", "cd"]), &[1]);
        let n_d = nul(date(&[19_723, 0, 19_724]), &[1]);

        let cases: Vec<(&str, Column)> = vec![
            ("abs", call("abs", &[n_i.clone()]).unwrap()),
            ("negate", call("negate", &[n_i.clone()]).unwrap()),
            ("plus", call("plus", &[n_i.clone(), ii(&[1, 1, 1])]).unwrap()),
            ("divide", call("divide", &[n_f.clone(), ff(&[2.0, 2.0, 2.0])]).unwrap()),
            ("round", call("round", &[n_f.clone()]).unwrap()),
            ("sqrt", call("sqrt", &[n_f.clone()]).unwrap()),
            ("sign", call("sign", &[n_i.clone()]).unwrap()),
            ("greatest", call("greatest", &[n_i.clone(), ii(&[9, 9, 9])]).unwrap()),
            ("length", call("length", &[n_s.clone()]).unwrap()),
            ("lower", call("lower", &[n_s.clone()]).unwrap()),
            ("concat", call("concat", &[n_s.clone(), ss(&["!", "!", "!"])]).unwrap()),
            ("substring", call("substring", &[n_s.clone(), ii(&[1, 1, 1])]).unwrap()),
            ("like", call("like", &[n_s.clone(), ss(&["%", "%", "%"])]).unwrap()),
            ("position", call("position", &[n_s.clone(), ss(&["a", "a", "a"])]).unwrap()),
            ("toString", call("toString", &[n_i.clone()]).unwrap()),
            ("toFloat64", call("toFloat64", &[n_i.clone()]).unwrap()),
            ("toYear", call("toYear", &[n_d.clone()]).unwrap()),
            ("addDays", call("addDays", &[n_d.clone(), ii(&[1, 1, 1])]).unwrap()),
            ("dateDiff", call("dateDiff", &[ss(&["day"; 3]), n_d.clone(), date(&[0, 0, 0])]).unwrap()),
            ("cityHash64", call("cityHash64", &[n_s.clone()]).unwrap()),
            ("xor", call("xor", &[n_i.clone(), ii(&[1, 1, 1])]).unwrap()),
        ];
        assert!(cases.len() >= 10);
        for (name, out) in cases {
            assert!(out.is_null(1), "{name} lost the NULL in row 1");
            assert!(!out.is_null(0), "{name} invented a NULL in row 0");
            assert!(!out.is_null(2), "{name} invented a NULL in row 2");
            assert!(out.ty.is_nullable(), "{name} returned a non-nullable {}", out.ty);
        }
    }

    #[test]
    fn nulls_from_several_arguments_are_unioned() {
        let a = nul(ii(&[1, 0, 3, 0]), &[1]);
        let b = nul(ii(&[1, 1, 0, 0]), &[2, 3]);
        let out = call("plus", &[a, b]).unwrap();
        assert!(!out.is_null(0));
        assert!(out.is_null(1) && out.is_null(2) && out.is_null(3));
    }

    #[test]
    fn an_empty_block_is_handled_by_every_shape_of_function() {
        for (name, args) in [
            ("abs", vec![ii(&[])]),
            ("plus", vec![ii(&[]), ii(&[])]),
            ("concat", vec![ss(&[]), ss(&[])]),
            ("like", vec![ss(&[]), ss(&[])]),
            ("toYear", vec![date(&[])]),
            ("coalesce", vec![ii(&[]), ii(&[])]),
            ("if", vec![ii(&[]), ii(&[]), ii(&[])]),
            ("cityHash64", vec![ss(&[])]),
        ] {
            let out = call(name, &args).unwrap();
            assert_eq!(out.len(), 0, "{name}");
            assert!(!out.has_nulls(), "{name}");
        }
        assert_eq!(call_n("now", &[], 0).unwrap().len(), 0);
        assert_eq!(call_n("rand", &[], 0).unwrap().len(), 0);
    }

    #[test]
    fn a_full_block_stays_correct_at_scale() {
        // exercise the loops at the real block size, not just toy inputs
        let n = crate::common::BLOCK_SIZE;
        let a: Vec<i64> = (0..n as i64).collect();
        let c = Column::i64s(DataType::Int64, a.clone());
        let out = call("plus", &[c.clone(), c]).unwrap();
        assert_eq!(out.len(), n);
        assert_eq!(i(&out)[n - 1], 2 * (n as i64 - 1));
    }
}
