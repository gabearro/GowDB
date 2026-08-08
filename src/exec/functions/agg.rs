//! The aggregate function library.
//!
//! Every entry is a `&'static AggFn`: a name, an arity, a return-type rule the
//! binder can call without touching data, and a constructor for an
//! [`Accumulator`]. One accumulator instance exists per group, so the state
//! footprint per accumulator is a real cost -- it is multiplied by the group
//! cardinality. That constraint drives most of the design decisions below.
//!
//! ## Why these particular implementations
//!
//! * **Integer `sum` accumulates in `i128`.** An OLAP `sum` over a billion
//!   `Int64` rows overflows `i64` routinely, and silently wrapping produces
//!   answers that look plausible. `i128` makes the fold itself exact for any
//!   realistic row count (≈9·10^18 rows of `u64::MAX`), and the only lossy
//!   step is the final narrowing to the declared return type.
//! * **Float `sum`/`avg` use Neumaier compensated summation.** Naive `f64`
//!   accumulation loses ~`log2(n)` bits: at 10^9 rows the low 30 bits of the
//!   answer are noise. Neumaier (Kahan-Babuska) recovers them for one extra
//!   add and one branch per row, and -- crucially for a parallel scan -- its
//!   state merges exactly.
//! * **`uniq` is a real HyperLogLog**, 2^14 registers of `u8`, so it merges
//!   register-wise and costs 16 KiB per group regardless of cardinality.
//!   `uniqExact` keeps every key and is the one to reach for when the group
//!   count is small.
//! * **Variance uses Welford**, not `sum(x^2) - sum(x)^2/n`. The textbook
//!   formula catastrophically cancels when the mean is large relative to the
//!   spread (timestamps, prices); Welford does not, and Chan's parallel
//!   variant merges two partial states in closed form.
//!
//! ## Decimals
//!
//! A `Decimal64(s)` lane is a count of `10^-s` units, so an aggregate that
//! reads the lane as a plain number answers `10^s` times high. The rule here
//! splits by what the aggregate does to the units, not by what it reads:
//!
//! * **Adding and selecting keep the scale.** `sum` is exact *for free* -- the
//!   `i128` fold is already a unit count -- and `min`/`max`/`any`/`argMin`
//!   hand back a lane the output column re-reads at its own declared scale.
//! * **Dividing widens**, to `max(s, 6)` digits, which is what `divide` already
//!   answers a decimal quotient with (`scalar::DIV_MIN_SCALE`). `avg` and the
//!   interpolating `quantile`/`median` divide; `quantileExact` does not.
//! * **`var`/`stddev` descale into `Float64`**, once per group rather than once
//!   per row: scaling every input by `10^-s` scales a variance by `10^-2s` and
//!   a standard deviation by `10^-s`, exactly.
//! * **Counting is scale-blind.** `count`/`uniq`/`uniqExact` compare lanes, and
//!   one column has one scale, so distinct lanes are distinct values.
//!
//! None of this costs the non-decimal path a branch: the scale is read once
//! when the accumulator is built and once more in `finish`, never per row.
//!
//! ## Deliberate deviations from ClickHouse
//!
//! * `groupArray` returns a comma-joined `String`, because this engine has no
//!   Array type. Capped at [`GROUP_ARRAY_LIMIT`] elements.
//! * `sum` **raises** when the exact `i128` total does not fit the declared
//!   return type, where ClickHouse wraps. It used to saturate, for the single
//!   reason that `finish` returned `Value` and had no way to refuse; see
//!   [`Accumulator::finish`] for why that was the worst of the three options.
//!   Overflow of the internal `i128` is still reported from `update`/`merge`,
//!   where it is order-independent.
//! * `quantile` is exact (same machinery as `quantileExact`), not the
//!   reservoir-sampled approximation. Memory is O(rows in the group).
//! * Aggregates over zero rows return `NULL`, `count` alone returns 0. That
//!   includes `sum`, where ClickHouse returns 0 -- see `SumAcc::finish` for
//!   why the ClickHouse rule was dropped rather than extended to the
//!   `Nullable` case.
//! * `avg`/`var*`/`stddev*` over an empty input return `NULL` where ClickHouse
//!   returns `nan`. `NULL` composes better with the rest of this engine.
//!
//! ## The `-If` combinator
//!
//! `sumIf(x, cond)` is not a separate implementation: [`CondAcc`] wraps any
//! inner accumulator, filters the selection vector by the trailing predicate
//! column, and forwards. Because the registry is a table of `&'static AggFn`
//! and `AggFn::new` is a bare `fn` pointer (no closure state), each `-If`
//! variant needs its own static entry; the `if_combinator!` macro generates
//! the two trampoline functions and the static for each base aggregate.
//!
//! `topK` is **not** implemented -- see the module note at the bottom.

use super::{Accumulator, AggFn};
use crate::common::{
    hash_bytes, i64_to_lane, lane_to_i64, splitmix64, Error, FastSet, Result,
};
use crate::types::value::{DECIMAL_MAX_UNITS, POW10};
use crate::types::{Column, ColumnData, DataType, PhysicalType, Value};
use std::any::Any;
use std::cell::Cell;
use std::cmp::Ordering;
use std::sync::Arc;

/// Hard cap on `groupArray` elements, so one pathological group cannot pin
/// unbounded memory.
pub const GROUP_ARRAY_LIMIT: usize = 10_000;

// ---------------------------------------------------------------- helpers

/// Visit the non-NULL rows named by `sel`. The null mask is inspected once per
/// call rather than once per row, so the no-nulls case (the common one) is a
/// flat loop the optimizer can unroll.
///
/// `sel` must hold valid indices into `col`; that is the caller's contract.
#[inline]
fn each_valid(col: &Column, sel: &[u32], mut f: impl FnMut(usize)) {
    debug_assert!(sel.iter().all(|&i| (i as usize) < col.len()), "sel out of range");
    match &col.nulls {
        None => {
            for &r in sel {
                f(r as usize);
            }
        }
        Some(nulls) => {
            for &r in sel {
                let i = r as usize;
                if !nulls.get(i) {
                    f(i);
                }
            }
        }
    }
}

/// Append `f(src[r])` for every non-NULL row `r` in `sel`.
///
/// The no-nulls arm is one `extend` over a `TrustedLen` iterator, so the vector
/// grows once per *block* and the loop that follows has no capacity check and
/// no length store per row -- which `each_valid` plus `Vec::push` cannot give,
/// because the check does not hoist out of the closure. Landed and measured
/// together with the rest of the quantile rework; see [`QuantileAcc`].
///
/// `sel` must hold valid indices into `col`, which is [`Accumulator::update`]'s
/// contract; the bounds check that enforces it here is one predictable branch
/// per row and the only one in the loop.
#[inline]
fn append_lanes<T: Copy>(
    out: &mut Vec<u64>,
    col: &Column,
    sel: &[u32],
    src: &[T],
    f: impl Fn(T) -> u64,
) {
    match &col.nulls {
        None => out.extend(sel.iter().map(|&r| f(src[r as usize]))),
        // Filtering drops `TrustedLen`, so reserve by hand and let `push` run;
        // a column with nulls is paying the mask read per row regardless.
        Some(nulls) => {
            out.reserve(sel.len());
            for &r in sel {
                let i = r as usize;
                if !nulls.get(i) {
                    out.push(f(src[i]));
                }
            }
        }
    }
}

fn downcast<'a, T: 'static>(other: &'a dyn Accumulator, who: &str) -> Result<&'a T> {
    other
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| Error::exec(format!("cannot merge a foreign accumulator into {who}")))
}

fn arg_at<'a>(args: &'a [Column], i: usize, who: &str) -> Result<&'a Column> {
    args.get(i)
        .ok_or_else(|| Error::exec(format!("{who} expected at least {} argument(s)", i + 1)))
}

/// Exactly `n` argument types, or a bind error naming the aggregate.
fn need_args(tys: &[DataType], n: usize, who: &str) -> Result<()> {
    if tys.len() != n {
        return Err(Error::bind(format!(
            "{who} takes exactly {n} argument(s), got {}",
            tys.len()
        )));
    }
    Ok(())
}

fn no_params(params: &[Value], who: &str) -> Result<()> {
    if !params.is_empty() {
        return Err(Error::bind(format!("{who} is not a parametric aggregate")));
    }
    Ok(())
}

fn need_numeric(ty: &DataType, who: &str) -> Result<()> {
    if !ty.is_numeric() {
        return Err(Error::bind(format!("{who} requires a numeric argument, got {ty}")));
    }
    Ok(())
}

/// `-If` predicate columns are numeric truth values (`UInt8`/`Bool` in
/// practice). Strings are rejected at bind time rather than silently taken as
/// "non-empty means true".
fn need_predicate(ty: &DataType, who: &str) -> Result<()> {
    if ty.is_string() {
        return Err(Error::bind(format!(
            "{who}If condition must be a numeric/Bool expression, got {ty}"
        )));
    }
    Ok(())
}

/// Fractional digits an aggregate that *divides* keeps for a `Decimal64(s)`
/// argument. Six is the floor, matching `scalar::DIV_MIN_SCALE` -- `avg(price)`
/// and `sum(price) / count(*)` must not disagree about how many digits an
/// average of money has, and the expression side already picked six.
///
/// Answering at the argument's own scale instead would round the mean of 1.19
/// and 3.80 to 2.50 and present it as exact.
#[inline]
fn div_out_scale(s: u8) -> u8 {
    s.max(6)
}

/// `num / den` rounded half away from zero, `den != 0`.
///
/// The rule every other narrowing in this engine uses (`decimal_rescale`,
/// `scalar::dec_divide`), kept identical here so an aggregate cannot round a
/// cent in the direction an expression would not. Banker's rounding is right
/// for statistics and wrong for money; this type exists for money.
#[inline]
fn div_round(num: i128, den: i128) -> i128 {
    let (q, rem) = (num / den, num % den);
    if rem.unsigned_abs() * 2 >= den.unsigned_abs() {
        q + if (num < 0) != (den < 0) { -1 } else { 1 }
    } else {
        q
    }
}

/// The narrowing at the end of `finish` refused.
///
/// Worded exactly as `scalar::dec_overflow` words it, because `avg(p)` and
/// `sum(p)/count(*)` are the same query and now fail on the same rows: one
/// sentence to recognise, not two.
///
/// `#[cold]` + `#[inline(never)]` keeps the `format!` -- a several-hundred-byte
/// code sequence -- out of `finish`, which is inlined into the per-group emit
/// loop and whose whole body is otherwise a compare and a move.
#[cold]
#[inline(never)]
fn dec_overflow(who: &str, scale: u8) -> Error {
    Error::exec(format!(
        "{who}: result does not fit Decimal64({scale}) -- more than 18 significant digits"
    ))
}

#[cold]
#[inline(never)]
fn int_overflow(who: &str, ty: &str) -> Error {
    Error::exec(format!("{who}: total does not fit {ty}"))
}

/// `units * 10^-scale`, or the refusal above.
///
/// `>` and not `>=`: `DECIMAL_MAX_UNITS` is 18 nines and is itself a legal
/// lane. `unsigned_abs` rather than `abs` so `i128::MIN` cannot panic here --
/// unreachable from any real fold, and free to rule out.
#[inline]
fn fit_dec(units: i128, scale: u8, who: &str) -> Result<Value> {
    if units.unsigned_abs() > DECIMAL_MAX_UNITS as u128 {
        return Err(dec_overflow(who, scale));
    }
    Ok(Value::Decimal(units as i64, scale))
}

/// Total order on `f64` that matches [`Value`]'s: NaN sorts last, `-0.0` ties
/// with `0.0`. Used so a typed min/max scan agrees with a `Value` comparison.
#[inline]
fn cmp_f64(a: &f64, b: &f64) -> Ordering {
    if a < b {
        Ordering::Less
    } else if a > b {
        Ordering::Greater
    } else if a == b {
        Ordering::Equal
    } else {
        match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            _ => Ordering::Less,
        }
    }
}

/// Collapse `-0.0` and every NaN payload so distinct-counting agrees with
/// [`Value`]'s equality, which treats them as equal.
#[inline]
fn canon_f64(x: f64) -> u64 {
    if x == 0.0 {
        0f64.to_bits()
    } else if x.is_nan() {
        f64::NAN.to_bits()
    } else {
        x.to_bits()
    }
}

/// Index of the smallest (or largest) non-NULL selected row, comparing in the
/// column's own physical representation. Ties keep the earliest index, which
/// makes `min`/`max` deterministic under a fixed scan order.
fn extreme_idx(col: &Column, sel: &[u32], want_max: bool) -> Option<usize> {
    macro_rules! scan {
        ($v:expr, $cmp:expr) => {{
            let v = $v;
            let mut best: Option<usize> = None;
            each_valid(col, sel, |i| {
                best = Some(match best {
                    None => i,
                    Some(b) => {
                        let ord = $cmp(&v[i], &v[b]);
                        if ord != Ordering::Equal && (ord == Ordering::Greater) == want_max {
                            i
                        } else {
                            b
                        }
                    }
                });
            });
            best
        }};
    }
    match &col.data {
        ColumnData::U64(v) => scan!(v, u64::cmp),
        ColumnData::I64(v) => scan!(v, i64::cmp),
        ColumnData::F64(v) => scan!(v, cmp_f64),
        ColumnData::Str(v) => scan!(v, |a: &Arc<str>, b: &Arc<str>| a.as_ref().cmp(b.as_ref())),
    }
}

// ------------------------------------------------------------------ count

/// `count()` / `count(x)`. The only aggregate whose empty result is 0, not
/// NULL, and the only one that can run with no argument columns at all --
/// `Block::rows_only` exists precisely so `count(*)` never touches data.
struct CountAcc {
    n: u64,
    star: bool,
}

impl Accumulator for CountAcc {
    fn update(&mut self, args: &[Column], sel: &[u32]) -> Result<()> {
        if self.star || args.is_empty() {
            self.n += sel.len() as u64;
        } else {
            let mut c = 0u64;
            each_valid(&args[0], sel, |_| c += 1);
            self.n += c;
        }
        Ok(())
    }
    fn merge(&mut self, other: &dyn Accumulator) -> Result<()> {
        self.n += downcast::<CountAcc>(other, "count")?.n;
        Ok(())
    }
    fn finish(&self) -> Result<Value> {
        Ok(Value::UInt(self.n))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn boxed_clone(&self) -> Box<dyn Accumulator> {
        Box::new(CountAcc { n: 0, star: self.star })
    }
}

// -------------------------------------------------------------- sum / avg

/// Running total, split by physical kind so integers stay exact.
#[derive(Clone, Copy, PartialEq)]
enum SumState {
    /// `i128` holds any `u64`/`i64` sum for row counts we can physically
    /// store, so the fold never loses a bit.
    Int(i128),
    /// Neumaier compensated sum: `total = sum + comp`.
    Float { sum: f64, comp: f64 },
}

impl SumState {
    #[inline]
    fn add_f64(&mut self, x: f64) {
        if let SumState::Float { sum, comp } = self {
            // Neumaier's refinement of Kahan: the low-order remainder depends
            // on which operand is larger. Plain Kahan loses the compensation
            // entirely when |x| >> |sum|, which is exactly the case that shows
            // up when a huge value lands in the middle of small ones.
            let t = *sum + x;
            if sum.abs() >= x.abs() {
                *comp += (*sum - t) + x;
            } else {
                *comp += (x - t) + *sum;
            }
            *sum = t;
        }
    }
}

/// Shared fold behind `sum`, `avg` and their `-If` forms.
struct SumCore {
    state: SumState,
    /// Non-NULL rows folded so far. Drives the empty-set rule and `avg`.
    n: u64,
    /// The argument's decimal scale, or `None`. Read once here instead of once
    /// per row: a decimal lane is a unit count, so the fold is byte-identical
    /// to the `Int64` one and only `finish` has to know where the point goes.
    ///
    /// It lives in `SumCore` rather than in `SumAcc`/`AvgAcc` because
    /// `SumState`'s `i128` aligns this struct to 16 and leaves 8 spare bytes
    /// after `n`; the field is free here, where on `AvgAcc` it would have
    /// rounded that struct up from 48 bytes per group to 64.
    ///
    /// Measured, A/B interleaved best-of-9 with the order swapped each round
    /// (a duplicated pre-field `SumCore`, since deleted), 3.3M rows through
    /// `add` per side: `sum` over `Int64` 1.002 / 1.000 / 0.956 across three
    /// runs and 1.000 / 1.002 under `--release`, `avg` 0.98-0.99. The fold's
    /// machine code is unchanged, as it has to be -- nothing here reads this.
    scale: Option<u8>,
}

// Both claims above, checked rather than trusted -- one instance of these
// exists per group, so a silent 16-byte growth is a real cost.
const _: () = assert!(std::mem::size_of::<SumCore>() == 48);
const _: () = assert!(std::mem::size_of::<SumAcc>() == 64);
const _: () = assert!(std::mem::size_of::<AvgAcc>() == 48);

impl SumCore {
    fn new(ty: &DataType) -> SumCore {
        let state = match ty.base().physical() {
            PhysicalType::F64 => SumState::Float { sum: 0.0, comp: 0.0 },
            _ => SumState::Int(0),
        };
        SumCore { state, n: 0, scale: ty.decimal_scale() }
    }

    /// An empty core of the same shape, for `boxed_clone`.
    fn reset(&self) -> SumCore {
        SumCore {
            state: match self.state {
                SumState::Int(_) => SumState::Int(0),
                SumState::Float { .. } => SumState::Float { sum: 0.0, comp: 0.0 },
            },
            n: 0,
            scale: self.scale,
        }
    }

    fn add(&mut self, col: &Column, sel: &[u32], who: &str) -> Result<()> {
        let mut count = 0u64;
        // `overflow` is latched rather than returned from the closure so the
        // inner loop stays branch-light; i128 overflow is unreachable in
        // practice but must not corrupt the total silently if it happens.
        let mut overflow = false;
        match (&mut self.state, &col.data) {
            (SumState::Int(total), ColumnData::U64(v)) => each_valid(col, sel, |i| {
                count += 1;
                match total.checked_add(v[i] as i128) {
                    Some(t) => *total = t,
                    None => overflow = true,
                }
            }),
            (SumState::Int(total), ColumnData::I64(v)) => each_valid(col, sel, |i| {
                count += 1;
                match total.checked_add(v[i] as i128) {
                    Some(t) => *total = t,
                    None => overflow = true,
                }
            }),
            (SumState::Int(_), ColumnData::F64(_)) => {
                return Err(Error::exec(format!(
                    "{who}: integer accumulator fed a Float64 column"
                )))
            }
            (st @ SumState::Float { .. }, ColumnData::F64(v)) => each_valid(col, sel, |i| {
                count += 1;
                st.add_f64(v[i]);
            }),
            // An integer column feeding a float accumulator is legal: it is
            // what `avg` over an Int64 column does when the plan widened.
            (st @ SumState::Float { .. }, ColumnData::U64(v)) => each_valid(col, sel, |i| {
                count += 1;
                st.add_f64(v[i] as f64);
            }),
            (st @ SumState::Float { .. }, ColumnData::I64(v)) => each_valid(col, sel, |i| {
                count += 1;
                st.add_f64(v[i] as f64);
            }),
            (_, ColumnData::Str(_)) => {
                return Err(Error::exec(format!("{who} cannot accumulate a String column")))
            }
        }
        if overflow {
            return Err(Error::exec(format!("{who}: 128-bit accumulator overflowed")));
        }
        self.n += count;
        Ok(())
    }

    fn merge_core(&mut self, other: &SumCore, who: &str) -> Result<()> {
        match (&mut self.state, other.state) {
            (SumState::Int(a), SumState::Int(b)) => {
                *a = a
                    .checked_add(b)
                    .ok_or_else(|| Error::exec(format!("{who}: 128-bit accumulator overflowed")))?;
            }
            (st @ SumState::Float { .. }, SumState::Float { sum, comp }) => {
                // Compensations are additive: fold the other partial's high
                // part through the compensated add, then carry its remainder.
                st.add_f64(sum);
                if let SumState::Float { comp: c, .. } = st {
                    *c += comp;
                }
            }
            _ => {
                return Err(Error::exec(format!(
                    "{who}: cannot merge integer and float partial sums"
                )))
            }
        }
        self.n += other.n;
        Ok(())
    }

    fn as_f64(&self) -> f64 {
        match self.state {
            SumState::Int(t) => t as f64,
            // Once the high part has saturated there is nothing left to
            // compensate, and `comp` has itself gone to -inf (the Neumaier
            // remainder of `1e308 + 1e308` is `(1e308 - inf) + 1e308`). Adding
            // them then answers **NaN** for a sum whose honest value is `inf`,
            // which is a silent wrong answer of the same family as the decimal
            // clamps: `sum(x) > 0` was false and `sum(x) <= 0` was false too.
            // One compare per group, never per row.
            SumState::Float { sum, comp } => {
                if sum.is_finite() {
                    sum + comp
                } else {
                    sum
                }
            }
        }
    }
}

/// `sum(x)`. Return type follows the input: signed ints -> `Int64`, unsigned
/// and `Bool` -> `UInt64`, floats -> `Float64`, decimals -> the argument's own
/// `Decimal64(s)`, always `Nullable` (see `finish`).
///
/// The only state beside the fold is one `bool`. It used to be the result's
/// full `DataType` plus a second `bool`, and since `SumCore` is 16-byte
/// aligned (its `i128` total) that padded the struct out: measured
/// `size_of::<SumAcc>()` 80 before, 64 after. One instance exists **per
/// group**, so that is 16 bytes per group off a hash aggregation's resident
/// set, for state `finish` only ever asks one question of. Measured on
/// `SELECT u, sum(v) FROM t GROUP BY u` with 400k groups over 1.6M rows,
/// A/B interleaved best-of-7 (an `AtomicBool` selecting the old struct, since
/// removed): 1.04-1.11x faster across three runs, always in that direction.
/// The 80-byte box also crossed an allocator size class, which is most of it.
struct SumAcc {
    core: SumCore,
    /// Signed integer input, so the narrowed total is an `Int` and not a
    /// `UInt`. Irrelevant on the float path, which `SumState` already tags.
    signed: bool,
}

impl Accumulator for SumAcc {
    fn update(&mut self, args: &[Column], sel: &[u32]) -> Result<()> {
        self.core.add(arg_at(args, 0, "sum")?, sel, "sum")
    }
    fn merge(&mut self, other: &dyn Accumulator) -> Result<()> {
        self.core.merge_core(&downcast::<SumAcc>(other, "sum")?.core, "sum")
    }
    fn finish(&self) -> Result<Value> {
        // Zero rows is NULL, unconditionally -- SQL standard and SQLite.
        //
        // This used to be 0 for a NULL-free argument (ClickHouse) and NULL for
        // a `Nullable` one, so the answer to "what does an empty sum mean"
        // depended on a static property of the *column declaration*. Whichever
        // dialect you read that against, one of the two answers was wrong, and
        // it changed under HAVING: `HAVING sum(x) > 0` over an empty global
        // group emitted a row for the non-nullable column and not for the
        // nullable one. NULL is also what `avg`/`min`/`max` here already
        // return over the same input, so `sum` was the outlier rather than the
        // house style; picking 0 for both would have meant changing those too.
        if self.core.n == 0 {
            return Ok(Value::Null);
        }
        match (self.core.scale, self.core.state) {
            // Summing a decimal is exact for free, and that is the nicest
            // property of storing the scale in the type: the i128 total is
            // already a count of the argument's own units, so the point goes
            // back where it was with no arithmetic and no f64 in the path.
            //
            // The arm costs the other three a tuple test that `finish` runs
            // once per group: 2.13 -> 2.17 ns per group, best-of-9 interleaved
            // over 5M calls (0.978 / 0.958 / 0.926 across three runs, so it is
            // real and it is 0.1 ns). Folding a decimal column and folding the
            // identical `Int64` lanes measure the same, 0.96-1.06 across five
            // runs -- one code path, as intended.
            (Some(s), SumState::Int(t)) => fit_dec(t, s, "sum"),
            // The exact total lives in i128 and only this last step can lose
            // it, so this is the step that refuses. It used to clamp, which
            // meant two rows of 5000000000000000.00 summed to
            // 9999999999999999.99 *and* `sum(p) = 10000000000000000.00`
            // then evaluated TRUE -- a wrong answer consistent with itself.
            (_, SumState::Int(t)) if self.signed => {
                i64::try_from(t).map(Value::Int).map_err(|_| int_overflow("sum", "Int64"))
            }
            (_, SumState::Int(t)) => {
                u64::try_from(t).map(Value::UInt).map_err(|_| int_overflow("sum", "UInt64"))
            }
            // Through `as_f64` so the saturated-high-part rule lives in one
            // place: `avg` reads the same total and must not disagree with
            // `sum` about whether it is `inf` or NaN.
            (_, SumState::Float { .. }) => Ok(Value::Float(self.core.as_f64())),
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn boxed_clone(&self) -> Box<dyn Accumulator> {
        Box::new(SumAcc { core: self.core.reset(), signed: self.signed })
    }
}

/// `avg(x)` -> `Float64`, or `Decimal64(max(s,6))` for a `Decimal64(s)`
/// argument. Kept as sum+count rather than a running mean: the running form
/// (`mean += (x-mean)/n`) costs a divide per row and merges less cleanly, and
/// the compensated sum is already exact enough.
struct AvgAcc {
    core: SumCore,
}

impl Accumulator for AvgAcc {
    fn update(&mut self, args: &[Column], sel: &[u32]) -> Result<()> {
        self.core.add(arg_at(args, 0, "avg")?, sel, "avg")
    }
    fn merge(&mut self, other: &dyn Accumulator) -> Result<()> {
        self.core.merge_core(&downcast::<AvgAcc>(other, "avg")?.core, "avg")
    }
    fn finish(&self) -> Result<Value> {
        if self.core.n == 0 {
            return Ok(Value::Null);
        }
        match (self.core.scale, self.core.state) {
            // One widened integer divide, once per group. The total is already
            // an exact unit count at scale `s`, so the mean at scale `os` never
            // has to touch a double -- routing it through `as_f64` would hand
            // back the imprecision this type exists to remove.
            //
            // `os` must be the scale `ret_avg` promised: the output column is
            // built from that type and takes this lane as-is. That promotion is
            // also why this arm refuses rather than clamps. Clamping happened
            // *after* the widening, so at scale 6 the representable magnitude
            // collapsed to 10^12 whatever the column's declared scale: a single
            // row of 1000000000000.00 averaged to 999999999999.999999 while
            // `max` of the same column answered 1000000000000.00.
            (Some(s), SumState::Int(t)) => {
                let os = div_out_scale(s);
                // The `checked_mul` arm needs ~1.7e20 rows of full-precision
                // units before the divide could pull the mean back into range,
                // so it is not a distinct outcome worth a distinct message.
                let u = t
                    .checked_mul(POW10[(os - s) as usize])
                    .map(|num| div_round(num, self.core.n as i128))
                    .ok_or_else(|| dec_overflow("avg", os))?;
                fit_dec(u, os, "avg")
            }
            _ => Ok(Value::Float(self.core.as_f64() / self.core.n as f64)),
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn boxed_clone(&self) -> Box<dyn Accumulator> {
        Box::new(AvgAcc { core: self.core.reset() })
    }
}

// -------------------------------------------------------------- min / max

/// `min(x)` / `max(x)`. One struct with a direction flag; `merge` refuses to
/// combine opposite directions so a plan bug cannot turn a min into a max.
///
/// Works on every physical kind including `String`: the per-block scan runs on
/// the raw representation, and only the block winner is materialized as a
/// `Value` for the cross-block fold.
///
/// That representation is also why a decimal needs nothing here. `Column::value`
/// reports a `Decimal64(s)` lane as a plain `Int`, but the lane *is* the answer:
/// the scale is fixed per column so lane order is value order, and `ret_same`
/// declares the output `Decimal64(s)`, whose builder puts the point back. The
/// same holds for `any`/`anyLast`/`argMin`/`argMax` below. `groupArray` is the
/// one that cannot rely on it -- it renders instead of handing a lane on.
struct MinMaxAcc {
    best: Option<Value>,
    max: bool,
}

impl MinMaxAcc {
    fn offer(&mut self, v: Value) {
        let better = match &self.best {
            None => true,
            Some(cur) => {
                let ord = v.cmp(cur);
                ord != Ordering::Equal && (ord == Ordering::Greater) == self.max
            }
        };
        if better {
            self.best = Some(v);
        }
    }
}

impl Accumulator for MinMaxAcc {
    fn update(&mut self, args: &[Column], sel: &[u32]) -> Result<()> {
        let col = arg_at(args, 0, if self.max { "max" } else { "min" })?;
        if let Some(i) = extreme_idx(col, sel, self.max) {
            self.offer(col.value(i));
        }
        Ok(())
    }
    fn merge(&mut self, other: &dyn Accumulator) -> Result<()> {
        let who = if self.max { "max" } else { "min" };
        let o = downcast::<MinMaxAcc>(other, who)?;
        if o.max != self.max {
            return Err(Error::exec("cannot merge a min accumulator with a max accumulator"));
        }
        if let Some(v) = o.best.clone() {
            self.offer(v);
        }
        Ok(())
    }
    fn finish(&self) -> Result<Value> {
        // Infallible by construction, and that is the point of the decimal note
        // above: the answer is a lane the argument column already held, so it
        // is in range because the column was.
        Ok(self.best.clone().unwrap_or(Value::Null))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn boxed_clone(&self) -> Box<dyn Accumulator> {
        Box::new(MinMaxAcc { best: None, max: self.max })
    }
}

// ------------------------------------------------------------ any/anyLast

/// `any(x)` (first non-NULL) and `anyLast(x)` (last non-NULL).
///
/// "First" and "last" are defined against the order in which rows are fed and
/// partials are merged: `merge` treats `other` as covering later rows. A
/// parallel scan therefore has to merge partials in scan order for these to be
/// reproducible -- which is the same caveat ClickHouse carries.
struct AnyAcc {
    v: Option<Value>,
    last: bool,
}

impl Accumulator for AnyAcc {
    fn update(&mut self, args: &[Column], sel: &[u32]) -> Result<()> {
        let col = arg_at(args, 0, if self.last { "anyLast" } else { "any" })?;
        if self.last {
            let mut found = None;
            each_valid(col, sel, |i| found = Some(i));
            if let Some(i) = found {
                self.v = Some(col.value(i));
            }
        } else if self.v.is_none() {
            let mut found = None;
            each_valid(col, sel, |i| {
                if found.is_none() {
                    found = Some(i);
                }
            });
            if let Some(i) = found {
                self.v = Some(col.value(i));
            }
        }
        Ok(())
    }
    fn merge(&mut self, other: &dyn Accumulator) -> Result<()> {
        let who = if self.last { "anyLast" } else { "any" };
        let o = downcast::<AnyAcc>(other, who)?;
        if o.last != self.last {
            return Err(Error::exec("cannot merge any with anyLast"));
        }
        if self.last {
            if o.v.is_some() {
                self.v = o.v.clone();
            }
        } else if self.v.is_none() {
            self.v = o.v.clone();
        }
        Ok(())
    }
    fn finish(&self) -> Result<Value> {
        Ok(self.v.clone().unwrap_or(Value::Null))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn boxed_clone(&self) -> Box<dyn Accumulator> {
        Box::new(AnyAcc { v: None, last: self.last })
    }
}

// ------------------------------------------------------- argMin / argMax

/// `argMin(v, k)` / `argMax(v, k)`: the `v` at the row with the extreme `k`.
///
/// Rows with a NULL **key** are skipped; a NULL **value** at the winning row is
/// returned as NULL. Ties go to the first such row.
struct ArgAcc {
    key: Option<Value>,
    val: Option<Value>,
    max: bool,
}

impl Accumulator for ArgAcc {
    fn update(&mut self, args: &[Column], sel: &[u32]) -> Result<()> {
        let who = if self.max { "argMax" } else { "argMin" };
        let vals = arg_at(args, 0, who)?;
        let keys = arg_at(args, 1, who)?;
        if vals.len() != keys.len() {
            return Err(Error::exec(format!("{who}: argument columns have different lengths")));
        }
        if let Some(i) = extreme_idx(keys, sel, self.max) {
            let k = keys.value(i);
            let better = match &self.key {
                None => true,
                Some(cur) => {
                    let ord = k.cmp(cur);
                    ord != Ordering::Equal && (ord == Ordering::Greater) == self.max
                }
            };
            if better {
                self.key = Some(k);
                self.val = Some(vals.value(i));
            }
        }
        Ok(())
    }
    fn merge(&mut self, other: &dyn Accumulator) -> Result<()> {
        let who = if self.max { "argMax" } else { "argMin" };
        let o = downcast::<ArgAcc>(other, who)?;
        if o.max != self.max {
            return Err(Error::exec("cannot merge argMin with argMax"));
        }
        if let Some(k) = &o.key {
            let better = match &self.key {
                None => true,
                Some(cur) => {
                    let ord = k.cmp(cur);
                    ord != Ordering::Equal && (ord == Ordering::Greater) == self.max
                }
            };
            if better {
                self.key = o.key.clone();
                self.val = o.val.clone();
            }
        }
        Ok(())
    }
    fn finish(&self) -> Result<Value> {
        Ok(self.val.clone().unwrap_or(Value::Null))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn boxed_clone(&self) -> Box<dyn Accumulator> {
        Box::new(ArgAcc { key: None, val: None, max: self.max })
    }
}

// ------------------------------------------------------------- uniqExact

/// Distinct keys held in their physical form rather than as `Value`s.
///
/// This is not just a size optimization: `Value`'s `Eq` compares integers
/// through `f64` once they exceed `i64::MAX`, so two distinct `u64` ids above
/// 2^63 would collapse into one. Raw lanes are exact.
///
/// Decimals need no scale here for the same reason `min`/`max` do not: one
/// column has one scale, so two lanes are equal exactly when the two values
/// are, and the count that comes out is a count either way.
enum DistinctSet {
    Num(FastSet<u64>),
    Text(FastSet<Arc<str>>),
}

impl DistinctSet {
    fn for_type(ty: &DataType) -> DistinctSet {
        match ty.base().physical() {
            PhysicalType::Str => DistinctSet::Text(FastSet::default()),
            _ => DistinctSet::Num(FastSet::default()),
        }
    }
    fn len(&self) -> usize {
        match self {
            DistinctSet::Num(s) => s.len(),
            DistinctSet::Text(s) => s.len(),
        }
    }
}

struct UniqExactAcc {
    set: DistinctSet,
}

impl Accumulator for UniqExactAcc {
    fn update(&mut self, args: &[Column], sel: &[u32]) -> Result<()> {
        let col = arg_at(args, 0, "uniqExact")?;
        match (&mut self.set, &col.data) {
            (DistinctSet::Num(s), ColumnData::U64(v)) => each_valid(col, sel, |i| {
                s.insert(v[i]);
            }),
            (DistinctSet::Num(s), ColumnData::I64(v)) => each_valid(col, sel, |i| {
                s.insert(v[i] as u64);
            }),
            (DistinctSet::Num(s), ColumnData::F64(v)) => each_valid(col, sel, |i| {
                s.insert(canon_f64(v[i]));
            }),
            (DistinctSet::Text(s), ColumnData::Str(v)) => each_valid(col, sel, |i| {
                if !s.contains(&v[i]) {
                    s.insert(v[i].clone());
                }
            }),
            _ => return Err(Error::exec("uniqExact: column kind changed between batches")),
        }
        Ok(())
    }
    fn merge(&mut self, other: &dyn Accumulator) -> Result<()> {
        let o = downcast::<UniqExactAcc>(other, "uniqExact")?;
        match (&mut self.set, &o.set) {
            (DistinctSet::Num(a), DistinctSet::Num(b)) => a.extend(b.iter().copied()),
            (DistinctSet::Text(a), DistinctSet::Text(b)) => a.extend(b.iter().cloned()),
            _ => return Err(Error::exec("uniqExact: cannot merge numeric and string key sets")),
        }
        Ok(())
    }
    fn finish(&self) -> Result<Value> {
        Ok(Value::UInt(self.set.len() as u64))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn boxed_clone(&self) -> Box<dyn Accumulator> {
        Box::new(UniqExactAcc {
            set: match self.set {
                DistinctSet::Num(_) => DistinctSet::Num(FastSet::default()),
                DistinctSet::Text(_) => DistinctSet::Text(FastSet::default()),
            },
        })
    }
}

// ------------------------------------------------------------------- uniq

/// Register-address bits. 2^14 registers -> 16 KiB per accumulator and a
/// standard error of `1.04/sqrt(2^14)` ≈ **0.81 %**.
const HLL_P: u32 = 14;
const HLL_M: usize = 1 << HLL_P;
const HLL_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// HyperLogLog. Registers are `u8` rather than packed 6-bit fields: with a
/// 64-bit hash the maximum rank is `64 - 14 + 1 = 51`, which needs 6 bits, but
/// packing them would cost a shift/mask on every row to save 4 KiB per group.
/// Cardinality estimation is a background statistic here, not the hot path, so
/// we buy the speed.
struct HllAcc {
    regs: Vec<u8>,
}

/// One row into the sketch.
///
/// `regs` is a fixed-size **array** reference, not a slice: `idx` is a 14-bit
/// shift so the index is provably below `HLL_M`, and only the array type lets
/// the compiler see that and drop the bounds check plus its panic edge from a
/// loop that runs once per row.
#[inline]
fn hll_add(regs: &mut [u8; HLL_M], h: u64) {
    // Top `p` bits address the register, the remaining 50 pick the rank. The
    // `| 1<<(p-1)` guarantees a set bit inside the payload so `leading_zeros`
    // is bounded by 50 and the +1 rank fits a u8.
    let idx = (h >> (64 - HLL_P)) as usize;
    let rho = ((h << HLL_P) | (1u64 << (HLL_P - 1))).leading_zeros() as u8 + 1;
    // `max`, not `if rho > *r { *r = rho }`. The branch is predicted
    // not-taken (a register that already holds 6 is beaten 1.5% of the time)
    // and the conditional store is what costs: it orders against every later
    // load the compiler cannot prove disjoint. Measured interleaved, the two
    // forms directly against each other, best-of-11 x 3 runs, 2M rows:
    //
    //   uniq(user_id)  100k distinct   max 1.51x 1.27x 1.52x  <- the HLL case
    //   uniq(latency)    900 distinct  max 0.92x 1.15x 0.92x
    //   uniq(bytes%8)      8 distinct  max 0.89x 1.01x 1.01x
    //   uniq(country)      8 distinct  max 1.14x 0.92x 1.04x
    //   controls (sum)                     1.45x 0.95x 0.94x
    //
    // Only the first row is outside the noise band the controls set, and it is
    // the one HLL exists for: at 8 distinct values the store lands on the same
    // 8 bytes every few rows and the dependency chain eats the win back, but
    // an 8-value column should be `uniqExact` anyway.
    let r = &mut regs[idx];
    *r = (*r).max(rho);
}

impl HllAcc {
    fn new() -> HllAcc {
        HllAcc { regs: vec![0u8; HLL_M] }
    }

    fn estimate(&self) -> u64 {
        let m = HLL_M as f64;
        let mut harmonic = 0.0f64;
        let mut zeros = 0usize;
        for &r in &self.regs {
            // 2^-r exactly, r <= 51 so the shift is safe and the reciprocal is
            // a power of two (no rounding).
            harmonic += 1.0 / (1u64 << r) as f64;
            if r == 0 {
                zeros += 1;
            }
        }
        let alpha = 0.7213 / (1.0 + 1.079 / m);
        let raw = alpha * m * m / harmonic;
        // Below ~2.5m the raw estimator is badly biased because most registers
        // are still empty; linear counting on the empty-register count is far
        // more accurate there and is exact for tiny cardinalities.
        if raw <= 2.5 * m && zeros > 0 {
            return (m * (m / zeros as f64).ln()).round() as u64;
        }
        // No large-range correction: that term exists to undo 32-bit hash
        // collisions, and these hashes are 64-bit.
        raw.round() as u64
    }
}

impl Accumulator for HllAcc {
    fn update(&mut self, args: &[Column], sel: &[u32]) -> Result<()> {
        let col = arg_at(args, 0, "uniq")?;
        // Reborrowed as an array once per block; see `hll_add`.
        let regs: &mut [u8; HLL_M] =
            (&mut self.regs[..]).try_into().expect("HllAcc::new sizes regs at HLL_M");
        match &col.data {
            ColumnData::U64(v) => each_valid(col, sel, |i| hll_add(regs, splitmix64(v[i]))),
            ColumnData::I64(v) => each_valid(col, sel, |i| hll_add(regs, splitmix64(v[i] as u64))),
            ColumnData::F64(v) => each_valid(col, sel, |i| hll_add(regs, splitmix64(canon_f64(v[i])))),
            ColumnData::Str(v) => {
                each_valid(col, sel, |i| hll_add(regs, hash_bytes(v[i].as_bytes(), HLL_SEED)))
            }
        }
        Ok(())
    }
    fn merge(&mut self, other: &dyn Accumulator) -> Result<()> {
        let o = downcast::<HllAcc>(other, "uniq")?;
        // The whole point of HLL: the union of two sketches is the register-wise
        // maximum, so partial scans compose with zero error.
        for (a, &b) in self.regs.iter_mut().zip(o.regs.iter()) {
            if b > *a {
                *a = b;
            }
        }
        Ok(())
    }
    fn finish(&self) -> Result<Value> {
        Ok(Value::UInt(self.estimate()))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn boxed_clone(&self) -> Box<dyn Accumulator> {
        Box::new(HllAcc::new())
    }
}

// ------------------------------------------------------------- groupArray

/// `groupArray(x)` -> comma-joined `String`.
///
/// The engine has no Array type, so the collected elements are rendered with
/// `Value::render_plain` and joined with `,`. That is lossy for strings
/// containing commas; it is a deliberate placeholder until an Array type
/// exists, not a format anyone should parse.
struct GroupArrayAcc {
    items: Vec<String>,
    cap: usize,
}

impl Accumulator for GroupArrayAcc {
    fn update(&mut self, args: &[Column], sel: &[u32]) -> Result<()> {
        let col = arg_at(args, 0, "groupArray")?;
        let (items, cap) = (&mut self.items, self.cap);
        // This is the one aggregate that renders instead of handing a lane to a
        // typed builder, so it is the one that has to reattach a decimal's
        // scale itself -- `Column::value` reports the lane as a plain `Int`,
        // and 3.81 would join the list as "381". Matched once per block, so the
        // decimal test never enters the row loop.
        match (col.ty.decimal_scale(), &col.data) {
            (Some(s), ColumnData::I64(v)) => each_valid(col, sel, |i| {
                if items.len() < cap {
                    items.push(Value::Decimal(v[i], s).render_plain());
                }
            }),
            _ => each_valid(col, sel, |i| {
                if items.len() < cap {
                    items.push(col.value(i).render_plain());
                }
            }),
        }
        Ok(())
    }
    fn merge(&mut self, other: &dyn Accumulator) -> Result<()> {
        let o = downcast::<GroupArrayAcc>(other, "groupArray")?;
        let room = self.cap.saturating_sub(self.items.len());
        self.items.extend(o.items.iter().take(room).cloned());
        Ok(())
    }
    fn finish(&self) -> Result<Value> {
        if self.items.is_empty() {
            return Ok(Value::Null);
        }
        Ok(Value::str(self.items.join(",")))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn boxed_clone(&self) -> Box<dyn Accumulator> {
        Box::new(GroupArrayAcc { items: Vec::new(), cap: self.cap })
    }
}

// --------------------------------------------------------------- quantile

/// `quantile(p)(x)`, `quantileExact(p)(x)`, `median(x)`.
///
/// Both spellings are exact -- every value is kept -- and both stay exact here:
/// no t-digest, no reservoir. `quantile` interpolates linearly between the two
/// neighbouring ranks; `quantileExact` returns an actual observed element
/// (rank `floor(p*n)`), which is what makes it "exact" in the ClickHouse sense.
/// The two contracts are unchanged; what changed is that neither one sorts.
///
/// A decimal argument is collected as raw lanes -- the scale is fixed per
/// column, so lanes sort exactly as values do -- and the point goes back on in
/// `finish`.
///
/// ## Selection, not sorting
///
/// An answer needs one order statistic (`quantileExact`) or two adjacent ones
/// (`quantile`), never the sorted sequence, so `finish` runs `select_nth_
/// unstable` -- average `O(n)`, worst case `O(n)` too since std's introselect
/// falls back to median-of-medians -- instead of `sort_unstable_by`. Selection
/// *permutes*, so the multiset survives and `finish` stays repeatable, which is
/// also what lets the buffer be selected **in place** rather than cloned; see
/// `vals` for how that reaches through `&self`.
///
/// Measured interleaved against HEAD's accumulator behind a temporary switch,
/// both sides in one loop with the leading side alternating, best-of-7..11,
/// four runs, 2M rows:
///
/// ```text
///   quantile(0.95)(latency)   UInt32,  900 distinct   2.86x 2.59x 2.69x 2.76x
///   quantileExact(0.95)                               2.60x 2.94x 2.71x 2.55x
///   median(bytes)             Int64, 65536 distinct   4.54x 4.42x 4.61x 4.23x
///   quantile(0.5) GROUP BY country, 8 groups          2.23x 2.11x 2.23x 2.32x
///   controls (ORDER BY, sum, uniq: untouched)         0.94x .. 1.16x
/// ```
///
/// The controls set the noise band on this machine, which is wide. Isolating
/// `finish` in a standalone single-threaded harness over the same 2M words
/// splits the win up, best-of-5:
///
/// ```text
///   clone + sort_unstable_by(total_cmp)   -- what this did   12.0 .. 12.7 ms
///   clone alone                                              0.27 ..  1.18
///   sort_unstable on the lane, no closure                    10.6 .. 11.6
///   clone + encode + select_nth_unstable                      2.7 ..  3.4
///   select_nth_unstable in place, lanes already encoded       2.0 ..  2.3
/// ```
///
/// So the cheap comparator is worth ~10% and *not sorting* is worth 5x; the
/// clone is 0.3 ms of it and is dropped for the memory, not the clock -- it was
/// a second copy of the whole group, 16 MB on this shape.
struct QuantileAcc {
    /// Every observed value, one word each, as an **order-preserving lane**:
    /// `f64_ord_key` bits or `i64_to_lane` decimal lanes, discriminated by
    /// `scale`. Either way a plain `u64` compare is the right compare, so the
    /// selection below runs on `Ord for u64` and not on a closure.
    ///
    /// A two-variant enum would put a tag word on a struct that exists once per
    /// group (40 bytes -> 48, asserted below), and `scale` already answers the
    /// question; the lane codecs are two ALU ops, not work.
    ///
    /// Holding the decimal lanes as integers is a correctness fix, not a
    /// micro-optimization. They used to live in the `f64` store, where a lane
    /// past 2^53 does not survive the round trip: over a `Decimal64(2)` column
    /// holding 1234567890123456.78, `quantileExact` answered
    /// 1234567890123456.80 while `min` and `max` over the same one row answered
    /// .78.
    ///
    /// The `Cell` is what lets `finish(&self)` select in place. `Cell::take`
    /// lends the buffer out and hands it straight back, which is the whole of
    /// the interior mutability -- no `RefCell` flag word (it would cost the 8
    /// bytes the enum tag was rejected for), no `unsafe`. It also makes the
    /// accumulator `!Sync`, which is free: the trait is `Any + Send`, so a
    /// shared `&dyn Accumulator` never crossed a thread anyway.
    vals: Cell<Vec<u64>>,
    p: f64,
    interpolate: bool,
    /// The argument's decimal scale, hoisted out of the row loop, and the
    /// discriminant for `vals` above.
    scale: Option<u8>,
}

const _: () = assert!(std::mem::size_of::<QuantileAcc>() == 40);

/// `f64` bits reordered so that a plain `u64` compare *is* `f64::total_cmp`.
///
/// Negatives descend as bit patterns, so invert them whole; non-negatives only
/// need the sign bit set to sort above every negative. Branchless, and exactly
/// the transformation `total_cmp` documents -- so this is the ordering
/// `QuantileAcc` has always used, not a new one. Deliberately **not**
/// [`crate::common::f64_to_lane`], which additionally folds -0.0 into +0.0 and
/// every NaN into one lane: that is `Value`'s ordering, and adopting it here
/// would move the answer of `quantileExact` over a column containing -0.0 or a
/// negative NaN.
#[inline(always)]
fn f64_ord_key(x: f64) -> u64 {
    let b = x.to_bits();
    b ^ ((((b as i64) >> 63) as u64) | (1 << 63))
}

/// Inverse of [`f64_ord_key`]; exact on every bit pattern, NaN payloads and
/// signed zeroes included.
#[inline(always)]
fn f64_ord_val(k: u64) -> f64 {
    f64::from_bits(k ^ if k & (1 << 63) != 0 { 1 << 63 } else { u64::MAX })
}

/// The `lo`-th and `hi`-th smallest lanes, where `hi` is `lo` or `lo + 1`.
///
/// Two selections would re-partition the whole buffer; the second rank is
/// adjacent, so the tail `select_nth_unstable` already left above `lo` supplies
/// it as a minimum -- one linear pass with no comparisons that can mispredict.
#[inline]
fn nth_pair(v: &mut [u64], lo: usize, hi: usize) -> (u64, u64) {
    debug_assert!(hi == lo || hi == lo + 1, "ranks must be adjacent");
    let (_, &mut a, above) = v.select_nth_unstable(lo);
    if hi == lo {
        return (a, a);
    }
    (a, above.iter().copied().min().unwrap_or(a))
}

/// Denominator for the interpolation weight, which is the one input to the
/// decimal path below that an `i128` cannot hold exactly: `p*(n-1)`'s fraction
/// is an `f64` by construction. Nine digits keep it well under the rounding
/// step of the answer, and the levels people actually ask for (0, ¼, ½, ¾, 1)
/// are exact in binary, so their weights come out exact too.
const QUANTILE_WEIGHT_ONE: i128 = POW10[9];

impl Accumulator for QuantileAcc {
    fn update(&mut self, args: &[Column], sel: &[u32]) -> Result<()> {
        let col = arg_at(args, 0, "quantile")?;
        let vals = self.vals.get_mut();
        // The decimal question is asked once per block and never per row, which
        // is the same shape `SumCore::add` and `GroupArrayAcc::update` use.
        match (self.scale, &col.data) {
            (Some(_), ColumnData::I64(v)) => append_lanes(vals, col, sel, v, i64_to_lane),
            // A `Decimal64` column is physically `I64` everywhere in the
            // engine, so this is a planner bug rather than a user error --
            // but it must not silently mix two encodings in one `vals`.
            (Some(_), _) => {
                return Err(Error::exec("quantile: decimal accumulator fed a non-decimal column"))
            }
            (None, ColumnData::U64(v)) => {
                append_lanes(vals, col, sel, v, |x| f64_ord_key(x as f64))
            }
            (None, ColumnData::I64(v)) => {
                append_lanes(vals, col, sel, v, |x| f64_ord_key(x as f64))
            }
            (None, ColumnData::F64(v)) => append_lanes(vals, col, sel, v, f64_ord_key),
            (None, ColumnData::Str(_)) => {
                return Err(Error::exec("quantile requires a numeric column"))
            }
        }
        Ok(())
    }
    fn merge(&mut self, other: &dyn Accumulator) -> Result<()> {
        let o = downcast::<QuantileAcc>(other, "quantile")?;
        // `Cell` has no `borrow`, so the other side's buffer is lent out and
        // handed straight back. Nothing in between can fail or panic, so `o` is
        // never observed empty.
        let ov = o.vals.take();
        self.vals.get_mut().extend_from_slice(&ov);
        o.vals.set(ov);
        Ok(())
    }
    fn finish(&self) -> Result<Value> {
        // Lent out for the duration and handed back on every exit, including
        // the fallible one at the bottom: `finish` must be repeatable, and a
        // buffer left in the `Cell` empty would answer NULL the second time.
        let mut v = self.vals.take();
        let out = self.answer(&mut v);
        self.vals.set(v);
        out
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn boxed_clone(&self) -> Box<dyn Accumulator> {
        Box::new(QuantileAcc {
            vals: Cell::new(Vec::new()),
            p: self.p,
            interpolate: self.interpolate,
            scale: self.scale,
        })
    }
}

impl QuantileAcc {
    /// The answer over `v`, which is left **permuted** rather than sorted --
    /// the multiset is what the next call needs and permuting preserves it.
    fn answer(&self, v: &mut [u64]) -> Result<Value> {
        if v.is_empty() {
            return Ok(Value::Null);
        }
        let n = v.len();
        // Rank arithmetic is shared; only the decode of `v` differs, and it is
        // decided once here rather than at each of the four reads below.
        let rank = |p: f64| ((p * n as f64).floor() as usize).min(n - 1);
        let pos = self.p * (n - 1) as f64;
        let (lo, frac) = (pos.floor() as usize, pos - pos.floor());
        let hi = (lo + 1).min(n - 1);

        let Some(s) = self.scale else {
            if !self.interpolate {
                let (k, _) = nth_pair(v, rank(self.p), rank(self.p));
                return Ok(Value::Float(f64_ord_val(k)));
            }
            let (a, b) = nth_pair(v, lo, hi);
            let (a, b) = (f64_ord_val(a), f64_ord_val(b));
            // Kept as written, `a + (b - a) * frac`, rather than folded to
            // `a` when `frac` is zero: at `p = 1` over a column holding an
            // infinity the difference is `inf - inf`, and the NaN that falls
            // out of it is the answer this has always given.
            return Ok(Value::Float(a + (b - a) * frac));
        };

        if !self.interpolate {
            // An element that was actually observed, so it keeps the argument's
            // own scale and is in range because the column was: nothing was
            // divided, nothing was widened, and nothing went through an `f64`.
            let (k, _) = nth_pair(v, rank(self.p), rank(self.p));
            return Ok(Value::Decimal(lane_to_i64(k), s));
        }
        // Interpolating between two lanes divides, so it widens like `avg` and
        // `divide` do rather than rounding the answer back onto the argument's
        // last digit -- the median of 1.19 and 3.81 is 2.50, not 3. Everything
        // but the weight stays in `i128`: the widest term is
        // `10^18 * 10^6 * 10^9`, five digits clear of the top.
        //
        // The rounding is applied to the whole interpolated value, not to the
        // `(b-a)` increment: selection makes that increment non-negative, so
        // rounding it away from zero would round -1.5 up to -1 while 1.5 went
        // to 2.
        let os = div_out_scale(s);
        let mul = POW10[(os - s) as usize];
        let (a, b) = nth_pair(v, lo, hi);
        let (a, b) = (lane_to_i64(a) as i128 * mul, lane_to_i64(b) as i128 * mul);
        let w = (frac * QUANTILE_WEIGHT_ONE as f64).round() as i128;
        let u = div_round(a * QUANTILE_WEIGHT_ONE + (b - a) * w, QUANTILE_WEIGHT_ONE);
        // Same widening as `avg`, so the same refusal: at scale 6 the
        // representable magnitude is 10^12, and clamping there answered
        // 999999999999.999999 for a median the column held exactly.
        fit_dec(u, os, "quantile")
    }
}

// -------------------------------------------------- variance / stddev

#[derive(Clone, Copy, PartialEq, Eq)]
enum VarKind {
    VarPop,
    VarSamp,
    StddevPop,
    StddevSamp,
}

impl VarKind {
    fn name(self) -> &'static str {
        match self {
            VarKind::VarPop => "varPop",
            VarKind::VarSamp => "varSamp",
            VarKind::StddevPop => "stddevPop",
            VarKind::StddevSamp => "stddevSamp",
        }
    }
    fn sample(self) -> bool {
        matches!(self, VarKind::VarSamp | VarKind::StddevSamp)
    }
    fn root(self) -> bool {
        matches!(self, VarKind::StddevPop | VarKind::StddevSamp)
    }
}

/// Welford's online variance.
///
/// The naive `E[x^2] - E[x]^2` form loses every significant digit when the
/// mean dwarfs the spread -- variance of unix timestamps a second apart is the
/// canonical disaster. Welford keeps the mean and the centred sum of squares
/// separately, so nothing large is ever subtracted from something else large.
struct WelfordAcc {
    n: u64,
    mean: f64,
    m2: f64,
    kind: VarKind,
    /// The argument's decimal scale, applied to the **result**, not the input.
    /// Scaling every observation by `10^-s` scales a variance by `10^-2s` and a
    /// standard deviation by `10^-s`, so descaling in `finish` is one divide per
    /// group where a per-row `v[i] as f64 / 10^s` would be one per row -- and it
    /// leaves the fold running on integer lanes, where Welford is at its most
    /// accurate. Lands in the padding after `kind`: `size_of` stays 32.
    scale: Option<u8>,
}

impl WelfordAcc {
    #[inline]
    fn push(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        // The second delta uses the *updated* mean; that pairing is what makes
        // the update numerically stable.
        self.m2 += delta * (x - self.mean);
    }
}

impl Accumulator for WelfordAcc {
    fn update(&mut self, args: &[Column], sel: &[u32]) -> Result<()> {
        let who = self.kind.name();
        let col = arg_at(args, 0, who)?;
        // The borrow checker will not let the closure hold `&mut self` while it
        // also calls a method on self, so drive a local copy and write back.
        // Only the three fold fields travel; `scale` is read in `finish` alone,
        // which is the whole point of descaling there rather than per row.
        let mut st =
            WelfordAcc { n: self.n, mean: self.mean, m2: self.m2, kind: self.kind, scale: None };
        match &col.data {
            ColumnData::U64(v) => each_valid(col, sel, |i| st.push(v[i] as f64)),
            ColumnData::I64(v) => each_valid(col, sel, |i| st.push(v[i] as f64)),
            ColumnData::F64(v) => each_valid(col, sel, |i| st.push(v[i])),
            ColumnData::Str(_) => {
                return Err(Error::exec(format!("{who} requires a numeric column")))
            }
        }
        self.n = st.n;
        self.mean = st.mean;
        self.m2 = st.m2;
        Ok(())
    }
    fn merge(&mut self, other: &dyn Accumulator) -> Result<()> {
        let o = downcast::<WelfordAcc>(other, self.kind.name())?;
        if o.kind != self.kind {
            return Err(Error::exec(format!(
                "cannot merge {} into {}",
                o.kind.name(),
                self.kind.name()
            )));
        }
        if o.n == 0 {
            return Ok(());
        }
        if self.n == 0 {
            self.n = o.n;
            self.mean = o.mean;
            self.m2 = o.m2;
            return Ok(());
        }
        // Chan/Golub/LeVeque pairwise combination.
        let (na, nb) = (self.n as f64, o.n as f64);
        let n = na + nb;
        let delta = o.mean - self.mean;
        self.mean += delta * nb / n;
        self.m2 += o.m2 + delta * delta * na * nb / n;
        self.n += o.n;
        Ok(())
    }
    fn finish(&self) -> Result<Value> {
        let denom = if self.kind.sample() {
            if self.n < 2 {
                return Ok(Value::Null);
            }
            (self.n - 1) as f64
        } else {
            if self.n < 1 {
                return Ok(Value::Null);
            }
            self.n as f64
        };
        let var = self.m2 / denom;
        let out = if self.kind.root() { var.max(0.0).sqrt() } else { var };
        // No range to overflow: `ret_var` declares `Float64` for every argument
        // including a decimal one, so the descale below lands in a type that
        // already admits every magnitude the fold can reach.
        Ok(Value::Float(match self.scale {
            // Quadratic in the input for a variance, linear for its root.
            Some(s) => {
                let p = POW10[s as usize] as f64;
                out / if self.kind.root() { p } else { p * p }
            }
            None => out,
        }))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn boxed_clone(&self) -> Box<dyn Accumulator> {
        Box::new(WelfordAcc { n: 0, mean: 0.0, m2: 0.0, kind: self.kind, scale: self.scale })
    }
}

// -------------------------------------------------------- -If combinator

/// Wraps any accumulator with a per-row predicate taken from the **last**
/// argument column. Rows whose predicate is NULL or zero are dropped before
/// the inner accumulator ever sees them, so NULL never counts as true.
struct CondAcc {
    inner: Box<dyn Accumulator>,
}

impl Accumulator for CondAcc {
    fn update(&mut self, args: &[Column], sel: &[u32]) -> Result<()> {
        let n = args.len();
        if n == 0 {
            return Err(Error::exec("-If aggregate invoked without a condition column"));
        }
        let pred = &args[n - 1];
        let mut keep: Vec<u32> = Vec::with_capacity(sel.len());
        match &pred.data {
            ColumnData::U64(v) => each_valid(pred, sel, |i| {
                if v[i] != 0 {
                    keep.push(i as u32);
                }
            }),
            ColumnData::I64(v) => each_valid(pred, sel, |i| {
                if v[i] != 0 {
                    keep.push(i as u32);
                }
            }),
            ColumnData::F64(v) => each_valid(pred, sel, |i| {
                if v[i] != 0.0 {
                    keep.push(i as u32);
                }
            }),
            ColumnData::Str(_) => {
                return Err(Error::exec("-If condition must be numeric, not String"))
            }
        }
        if keep.is_empty() {
            return Ok(());
        }
        self.inner.update(&args[..n - 1], &keep)
    }
    fn merge(&mut self, other: &dyn Accumulator) -> Result<()> {
        let o = downcast::<CondAcc>(other, "-If")?;
        // Delegate the real type check to the inner accumulators.
        self.inner.merge(o.inner.as_ref())
    }
    fn finish(&self) -> Result<Value> {
        self.inner.finish()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn boxed_clone(&self) -> Box<dyn Accumulator> {
        Box::new(CondAcc { inner: self.inner.boxed_clone() })
    }
}

fn ret_if(base: &'static AggFn, tys: &[DataType], params: &[Value]) -> Result<DataType> {
    let n = tys.len();
    if n == 0 {
        return Err(Error::bind(format!(
            "{}If requires a trailing condition argument",
            base.name
        )));
    }
    need_predicate(&tys[n - 1], base.name)?;
    (base.ret)(&tys[..n - 1], params)
}

fn new_if(base: &'static AggFn, tys: &[DataType], params: &[Value]) -> Result<Box<dyn Accumulator>> {
    let n = tys.len();
    if n == 0 {
        return Err(Error::bind(format!(
            "{}If requires a trailing condition argument",
            base.name
        )));
    }
    need_predicate(&tys[n - 1], base.name)?;
    Ok(Box::new(CondAcc { inner: (base.new)(&tys[..n - 1], params)? }))
}

// ------------------------------------------------------- return-type rules

fn ret_count(tys: &[DataType], params: &[Value]) -> Result<DataType> {
    no_params(params, "count")?;
    if tys.len() > 1 {
        return Err(Error::bind("count takes 0 or 1 arguments"));
    }
    Ok(DataType::UInt64)
}

/// `Int64` for signed inputs, `UInt64` for unsigned/Bool, `Float64` for
/// floats -- and always Nullable, because an empty input sums to NULL whatever
/// the argument was declared as (see `SumAcc::finish`). Deriving the wrapper
/// from `tys[0].is_nullable()` instead is what let the empty-set answer depend
/// on the column declaration.
fn ret_sum(tys: &[DataType], params: &[Value]) -> Result<DataType> {
    no_params(params, "sum")?;
    need_args(tys, 1, "sum")?;
    need_numeric(&tys[0], "sum")?;
    Ok(match tys[0].base() {
        // Ahead of the physical dispatch, which would see only `I64` and call a
        // sum of prices an `Int64` count of hundredths.
        DataType::Decimal64(s) => DataType::Decimal64(*s),
        b => match b.physical() {
            PhysicalType::F64 => DataType::Float64,
            PhysicalType::I64 => DataType::Int64,
            _ => DataType::UInt64,
        },
    }
    .to_nullable())
}

fn ret_avg(tys: &[DataType], params: &[Value]) -> Result<DataType> {
    no_params(params, "avg")?;
    need_args(tys, 1, "avg")?;
    need_numeric(&tys[0], "avg")?;
    // This scale and the one `AvgAcc::finish` emits have to be the same number:
    // the output column is built from this type and takes the value's lane
    // as-is, so a disagreement is a silently misplaced decimal point rather
    // than a type error. Both go through `div_out_scale`.
    let base = match tys[0].decimal_scale() {
        Some(s) => DataType::Decimal64(div_out_scale(s)),
        None => DataType::Float64,
    };
    Ok(if tys[0].is_nullable() { base.to_nullable() } else { base })
}

/// `min`/`max`/`any`/`anyLast` all return the argument's own type, Nullable
/// wrapper included -- that wrapper is exactly the "no rows survived" case.
fn ret_same(tys: &[DataType], params: &[Value]) -> Result<DataType> {
    no_params(params, "min/max/any")?;
    need_args(tys, 1, "min/max/any")?;
    Ok(tys[0].clone())
}

fn ret_arg(tys: &[DataType], params: &[Value]) -> Result<DataType> {
    no_params(params, "argMin/argMax")?;
    need_args(tys, 2, "argMin/argMax")?;
    // The result is a value that may not exist (empty group), so it is always
    // potentially NULL.
    Ok(tys[0].to_nullable())
}

fn ret_uniq(tys: &[DataType], params: &[Value]) -> Result<DataType> {
    no_params(params, "uniq")?;
    need_args(tys, 1, "uniq")?;
    Ok(DataType::UInt64)
}

fn ret_group_array(tys: &[DataType], params: &[Value]) -> Result<DataType> {
    if params.len() > 1 {
        return Err(Error::bind("groupArray takes at most one parameter (max size)"));
    }
    need_args(tys, 1, "groupArray")?;
    Ok(DataType::String)
}

fn quantile_p(params: &[Value], who: &str) -> Result<f64> {
    let p = match params.first() {
        None => 0.5,
        Some(v) => v
            .as_f64()
            .ok_or_else(|| Error::bind(format!("{who} level must be a number, got {v}")))?,
    };
    if params.len() > 1 {
        return Err(Error::bind(format!("{who} takes at most one level parameter")));
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(Error::bind(format!("{who} level must be in [0, 1], got {p}")));
    }
    Ok(p)
}

/// `Float64`, except for a decimal argument, where the answer keeps its exact
/// type: the argument's own scale when the aggregate returns an element it saw
/// (`quantileExact`) and `div_out_scale` when it interpolates between two
/// (`quantile`, `median`) -- interpolation is a division, and dividing widens.
///
/// Same invariant as `ret_avg`: the scale here is the scale `QuantileAcc::finish`
/// emits, both through `div_out_scale`.
fn ret_quantile_at(
    tys: &[DataType],
    params: &[Value],
    who: &str,
    interpolate: bool,
) -> Result<DataType> {
    need_args(tys, 1, who)?;
    need_numeric(&tys[0], who)?;
    quantile_p(params, who)?;
    Ok(match tys[0].decimal_scale() {
        Some(s) => DataType::Decimal64(if interpolate { div_out_scale(s) } else { s }),
        None => DataType::Float64,
    })
}

fn ret_quantile(tys: &[DataType], params: &[Value]) -> Result<DataType> {
    ret_quantile_at(tys, params, "quantile", true)
}

fn ret_quantile_exact(tys: &[DataType], params: &[Value]) -> Result<DataType> {
    ret_quantile_at(tys, params, "quantileExact", false)
}

fn ret_median(tys: &[DataType], params: &[Value]) -> Result<DataType> {
    no_params(params, "median")?;
    ret_quantile_at(tys, params, "median", true)
}

fn ret_var(tys: &[DataType], params: &[Value]) -> Result<DataType> {
    no_params(params, "varPop/varSamp/stddevPop/stddevSamp")?;
    need_args(tys, 1, "varPop/varSamp/stddevPop/stddevSamp")?;
    need_numeric(&tys[0], "varPop/varSamp/stddevPop/stddevSamp")?;
    Ok(DataType::Float64)
}

// ------------------------------------------------------------ constructors

fn new_count(tys: &[DataType], params: &[Value]) -> Result<Box<dyn Accumulator>> {
    ret_count(tys, params)?;
    Ok(Box::new(CountAcc { n: 0, star: tys.is_empty() }))
}

fn new_sum(tys: &[DataType], params: &[Value]) -> Result<Box<dyn Accumulator>> {
    ret_sum(tys, params)?;
    Ok(Box::new(SumAcc {
        core: SumCore::new(&tys[0]),
        signed: tys[0].base().physical() == PhysicalType::I64,
    }))
}

fn new_avg(tys: &[DataType], params: &[Value]) -> Result<Box<dyn Accumulator>> {
    ret_avg(tys, params)?;
    // Force the float path: averaging integers through the i128 total would be
    // exact too, but the float state also handles a Float64 input unchanged.
    //
    // A decimal argument is the exception, and keeps the integral fold: its
    // total is an exact unit count that `finish` divides at the decided scale,
    // and an f64 in the middle would give back the digits the type exists for.
    //
    // It is also the faster of the two. A checked `i128` add beats a Neumaier
    // compensated `f64` add (which is an add, an abs compare, two subtracts and
    // an add): averaging a `Decimal64(2)` column measured 1.30 / 1.35 / 1.35x
    // the throughput of averaging the identical lanes as `Int64`, and 1.47-1.64x
    // under `--release`. Widening `avg` over plain integers to the same fold is
    // the obvious follow-up and is deliberately *not* done here -- it would
    // change `avg(Int64)`'s answer from a compensated float mean to an exact
    // rational one, which is a semantic change to argue separately.
    Ok(Box::new(AvgAcc {
        core: SumCore::new(if tys[0].is_decimal() { &tys[0] } else { &DataType::Float64 }),
    }))
}

fn new_min(tys: &[DataType], params: &[Value]) -> Result<Box<dyn Accumulator>> {
    ret_same(tys, params)?;
    Ok(Box::new(MinMaxAcc { best: None, max: false }))
}

fn new_max(tys: &[DataType], params: &[Value]) -> Result<Box<dyn Accumulator>> {
    ret_same(tys, params)?;
    Ok(Box::new(MinMaxAcc { best: None, max: true }))
}

fn new_any(tys: &[DataType], params: &[Value]) -> Result<Box<dyn Accumulator>> {
    ret_same(tys, params)?;
    Ok(Box::new(AnyAcc { v: None, last: false }))
}

fn new_any_last(tys: &[DataType], params: &[Value]) -> Result<Box<dyn Accumulator>> {
    ret_same(tys, params)?;
    Ok(Box::new(AnyAcc { v: None, last: true }))
}

fn new_arg_min(tys: &[DataType], params: &[Value]) -> Result<Box<dyn Accumulator>> {
    ret_arg(tys, params)?;
    Ok(Box::new(ArgAcc { key: None, val: None, max: false }))
}

fn new_arg_max(tys: &[DataType], params: &[Value]) -> Result<Box<dyn Accumulator>> {
    ret_arg(tys, params)?;
    Ok(Box::new(ArgAcc { key: None, val: None, max: true }))
}

fn new_uniq_exact(tys: &[DataType], params: &[Value]) -> Result<Box<dyn Accumulator>> {
    ret_uniq(tys, params)?;
    Ok(Box::new(UniqExactAcc { set: DistinctSet::for_type(&tys[0]) }))
}

fn new_uniq(tys: &[DataType], params: &[Value]) -> Result<Box<dyn Accumulator>> {
    ret_uniq(tys, params)?;
    Ok(Box::new(HllAcc::new()))
}

fn new_group_array(tys: &[DataType], params: &[Value]) -> Result<Box<dyn Accumulator>> {
    ret_group_array(tys, params)?;
    // `groupArray(n)(x)` narrows the cap; the hard limit still applies.
    let cap = params
        .first()
        .and_then(|v| v.as_u64())
        .map_or(GROUP_ARRAY_LIMIT, |n| (n as usize).min(GROUP_ARRAY_LIMIT));
    Ok(Box::new(GroupArrayAcc { items: Vec::new(), cap }))
}

fn new_quantile(tys: &[DataType], params: &[Value]) -> Result<Box<dyn Accumulator>> {
    ret_quantile(tys, params)?;
    Ok(Box::new(QuantileAcc {
        vals: Cell::new(Vec::new()),
        p: quantile_p(params, "quantile")?,
        interpolate: true,
        scale: tys[0].decimal_scale(),
    }))
}

fn new_quantile_exact(tys: &[DataType], params: &[Value]) -> Result<Box<dyn Accumulator>> {
    ret_quantile_exact(tys, params)?;
    Ok(Box::new(QuantileAcc {
        vals: Cell::new(Vec::new()),
        p: quantile_p(params, "quantileExact")?,
        interpolate: false,
        scale: tys[0].decimal_scale(),
    }))
}

fn new_median(tys: &[DataType], params: &[Value]) -> Result<Box<dyn Accumulator>> {
    ret_median(tys, params)?;
    Ok(Box::new(QuantileAcc {
        vals: Cell::new(Vec::new()),
        p: 0.5,
        interpolate: true,
        scale: tys[0].decimal_scale(),
    }))
}

fn new_welford(kind: VarKind, tys: &[DataType], params: &[Value]) -> Result<Box<dyn Accumulator>> {
    ret_var(tys, params)?;
    Ok(Box::new(WelfordAcc { n: 0, mean: 0.0, m2: 0.0, kind, scale: tys[0].decimal_scale() }))
}

// `AggFn::new` is a bare `fn` pointer with no room for captured state, so each
// variance flavour needs its own monomorphic trampoline.
fn new_var_pop(t: &[DataType], p: &[Value]) -> Result<Box<dyn Accumulator>> {
    new_welford(VarKind::VarPop, t, p)
}
fn new_var_samp(t: &[DataType], p: &[Value]) -> Result<Box<dyn Accumulator>> {
    new_welford(VarKind::VarSamp, t, p)
}
fn new_stddev_pop(t: &[DataType], p: &[Value]) -> Result<Box<dyn Accumulator>> {
    new_welford(VarKind::StddevPop, t, p)
}
fn new_stddev_samp(t: &[DataType], p: &[Value]) -> Result<Box<dyn Accumulator>> {
    new_welford(VarKind::StddevSamp, t, p)
}

// ------------------------------------------------------------- the registry

macro_rules! agg {
    ($id:ident, $name:literal, $lo:expr, $hi:expr, $ret:path, $new:path, $dist:expr) => {
        static $id: AggFn = AggFn {
            name: $name,
            arity: ($lo, $hi),
            ret: $ret,
            new: $new,
            supports_distinct: $dist,
        };
    };
}

agg!(COUNT, "count", 0, 1, ret_count, new_count, true);
agg!(SUM, "sum", 1, 1, ret_sum, new_sum, true);
agg!(AVG, "avg", 1, 1, ret_avg, new_avg, true);
agg!(MIN, "min", 1, 1, ret_same, new_min, false);
agg!(MAX, "max", 1, 1, ret_same, new_max, false);
agg!(ANY, "any", 1, 1, ret_same, new_any, false);
agg!(ANY_LAST, "anyLast", 1, 1, ret_same, new_any_last, false);
agg!(ARG_MIN, "argMin", 2, 2, ret_arg, new_arg_min, false);
agg!(ARG_MAX, "argMax", 2, 2, ret_arg, new_arg_max, false);
agg!(UNIQ, "uniq", 1, 1, ret_uniq, new_uniq, true);
agg!(UNIQ_EXACT, "uniqExact", 1, 1, ret_uniq, new_uniq_exact, true);
agg!(GROUP_ARRAY, "groupArray", 1, 1, ret_group_array, new_group_array, true);
agg!(QUANTILE, "quantile", 1, 1, ret_quantile, new_quantile, true);
agg!(QUANTILE_EXACT, "quantileExact", 1, 1, ret_quantile_exact, new_quantile_exact, true);
agg!(MEDIAN, "median", 1, 1, ret_median, new_median, true);
agg!(VAR_POP, "varPop", 1, 1, ret_var, new_var_pop, false);
agg!(VAR_SAMP, "varSamp", 1, 1, ret_var, new_var_samp, false);
agg!(STDDEV_POP, "stddevPop", 1, 1, ret_var, new_stddev_pop, false);
agg!(STDDEV_SAMP, "stddevSamp", 1, 1, ret_var, new_stddev_samp, false);

/// Generate an `-If` variant: two trampolines plus the static entry. Arity is
/// the base arity shifted by the extra trailing condition argument.
macro_rules! if_combinator {
    ($id:ident, $retfn:ident, $newfn:ident, $name:literal, $base:ident, $lo:expr, $hi:expr, $dist:expr) => {
        fn $retfn(tys: &[DataType], params: &[Value]) -> Result<DataType> {
            ret_if(&$base, tys, params)
        }
        fn $newfn(tys: &[DataType], params: &[Value]) -> Result<Box<dyn Accumulator>> {
            new_if(&$base, tys, params)
        }
        static $id: AggFn = AggFn {
            name: $name,
            arity: ($lo, $hi),
            ret: $retfn,
            new: $newfn,
            supports_distinct: $dist,
        };
    };
}

if_combinator!(COUNT_IF, ret_count_if, new_count_if, "countIf", COUNT, 1, 2, true);
if_combinator!(SUM_IF, ret_sum_if, new_sum_if, "sumIf", SUM, 2, 2, true);
if_combinator!(AVG_IF, ret_avg_if, new_avg_if, "avgIf", AVG, 2, 2, true);
if_combinator!(MIN_IF, ret_min_if, new_min_if, "minIf", MIN, 2, 2, false);
if_combinator!(MAX_IF, ret_max_if, new_max_if, "maxIf", MAX, 2, 2, false);
if_combinator!(ANY_IF, ret_any_if, new_any_if, "anyIf", ANY, 2, 2, false);
if_combinator!(ANY_LAST_IF, ret_any_last_if, new_any_last_if, "anyLastIf", ANY_LAST, 2, 2, false);
if_combinator!(ARG_MIN_IF, ret_arg_min_if, new_arg_min_if, "argMinIf", ARG_MIN, 3, 3, false);
if_combinator!(ARG_MAX_IF, ret_arg_max_if, new_arg_max_if, "argMaxIf", ARG_MAX, 3, 3, false);
if_combinator!(UNIQ_IF, ret_uniq_if, new_uniq_if, "uniqIf", UNIQ, 2, 2, true);
if_combinator!(UNIQ_EXACT_IF, ret_uniq_exact_if, new_uniq_exact_if, "uniqExactIf", UNIQ_EXACT, 2, 2, true);
if_combinator!(GROUP_ARRAY_IF, ret_group_array_if, new_group_array_if, "groupArrayIf", GROUP_ARRAY, 2, 2, true);
if_combinator!(QUANTILE_IF, ret_quantile_if, new_quantile_if, "quantileIf", QUANTILE, 2, 2, true);
if_combinator!(QUANTILE_EXACT_IF, ret_quantile_exact_if, new_quantile_exact_if, "quantileExactIf", QUANTILE_EXACT, 2, 2, true);
if_combinator!(MEDIAN_IF, ret_median_if, new_median_if, "medianIf", MEDIAN, 2, 2, true);
if_combinator!(VAR_POP_IF, ret_var_pop_if, new_var_pop_if, "varPopIf", VAR_POP, 2, 2, false);
if_combinator!(VAR_SAMP_IF, ret_var_samp_if, new_var_samp_if, "varSampIf", VAR_SAMP, 2, 2, false);
if_combinator!(STDDEV_POP_IF, ret_stddev_pop_if, new_stddev_pop_if, "stddevPopIf", STDDEV_POP, 2, 2, false);
if_combinator!(STDDEV_SAMP_IF, ret_stddev_samp_if, new_stddev_samp_if, "stddevSampIf", STDDEV_SAMP, 2, 2, false);

/// Look up an aggregate. `name` is **already lowercased** by
/// [`super::aggregate`].
///
/// Base names are tried first so a hypothetical future aggregate whose name
/// ends in `if` cannot be shadowed by the combinator rule.
pub fn lookup(name: &str) -> Option<&'static AggFn> {
    if let Some(f) = base_lookup(name) {
        return Some(f);
    }
    name.strip_suffix("if").and_then(if_lookup)
}

fn base_lookup(name: &str) -> Option<&'static AggFn> {
    Some(match name {
        "count" => &COUNT,
        "sum" => &SUM,
        "avg" => &AVG,
        "min" => &MIN,
        "max" => &MAX,
        // `first_value`/`last_value` are deliberately NOT aliased here: they
        // are window functions and live in `exec::operators::window`, whose
        // `lookup` tries the window-only names first and *then* falls through
        // to this table. Claiming them here would shadow the real ones.
        // (`any`/`anyLast` below are the ClickHouse aggregates with similar
        // meanings; they are not the same functions and are not aliases.)
        "any" => &ANY,
        "anylast" => &ANY_LAST,
        "argmin" => &ARG_MIN,
        "argmax" => &ARG_MAX,
        "uniq" => &UNIQ,
        "uniqexact" => &UNIQ_EXACT,
        "grouparray" => &GROUP_ARRAY,
        "quantile" => &QUANTILE,
        "quantileexact" => &QUANTILE_EXACT,
        "median" => &MEDIAN,
        "varpop" | "var_pop" => &VAR_POP,
        "varsamp" | "var_samp" | "variance" => &VAR_SAMP,
        "stddevpop" | "stddev_pop" => &STDDEV_POP,
        "stddevsamp" | "stddev_samp" | "stddev" => &STDDEV_SAMP,
        _ => return None,
    })
}

/// `name` here is the base stem with the trailing `if` already removed.
fn if_lookup(stem: &str) -> Option<&'static AggFn> {
    Some(match stem {
        "count" => &COUNT_IF,
        "sum" => &SUM_IF,
        "avg" => &AVG_IF,
        "min" => &MIN_IF,
        "max" => &MAX_IF,
        "any" => &ANY_IF,
        "anylast" => &ANY_LAST_IF,
        "argmin" => &ARG_MIN_IF,
        "argmax" => &ARG_MAX_IF,
        "uniq" => &UNIQ_IF,
        "uniqexact" => &UNIQ_EXACT_IF,
        "grouparray" => &GROUP_ARRAY_IF,
        "quantile" => &QUANTILE_IF,
        "quantileexact" => &QUANTILE_EXACT_IF,
        "median" => &MEDIAN_IF,
        "varpop" => &VAR_POP_IF,
        "varsamp" => &VAR_SAMP_IF,
        "stddevpop" => &STDDEV_POP_IF,
        "stddevsamp" => &STDDEV_SAMP_IF,
        _ => return None,
    })
}

// `topK` is intentionally absent. A correct implementation is Space-Saving
// with a bounded counter table, whose merge is only approximate and whose
// error bound needs its own analysis; and with no Array type its result would
// need the same lossy string encoding `groupArray` uses. Not worth shipping
// half of.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ColumnBuilder;

    // ------------------------------------------------------------ fixtures

    fn ints(v: &[i64]) -> Column {
        Column::i64s(DataType::Int64, v.to_vec())
    }
    fn uints(v: &[u64]) -> Column {
        Column::u64s(DataType::UInt64, v.to_vec())
    }
    fn floats(v: &[f64]) -> Column {
        Column::f64s(DataType::Float64, v.to_vec())
    }
    fn strs(v: &[&str]) -> Column {
        Column::strs(DataType::String, v.iter().map(|s| Arc::from(*s)).collect())
    }
    fn bools(v: &[bool]) -> Column {
        Column::bools(v.iter().map(|&b| b as u64).collect())
    }
    fn nullable_ints(v: &[Option<i64>]) -> Column {
        let mut b = ColumnBuilder::new(DataType::Nullable(Box::new(DataType::Int64)));
        for x in v {
            match x {
                Some(i) => b.push_value(&Value::Int(*i)).unwrap(),
                None => b.push_null(),
            }
        }
        b.finish()
    }
    fn all(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    const NI64: DataType = DataType::Int64;
    fn nullable_i64() -> DataType {
        DataType::Nullable(Box::new(DataType::Int64))
    }

    /// Build, feed everything, finish.
    fn run(f: &AggFn, tys: &[DataType], params: &[Value], cols: &[Column]) -> Value {
        try_run(f, tys, params, cols).unwrap()
    }

    /// `run`, but keeping the refusal: `finish` narrows a wider fold to the
    /// declared return type and that is allowed to fail.
    fn try_run(f: &AggFn, tys: &[DataType], params: &[Value], cols: &[Column]) -> Result<Value> {
        let n = cols.first().map_or(0, |c| c.len());
        let mut a = (f.new)(tys, params).unwrap();
        a.update(cols, &all(n)).unwrap();
        a.finish()
    }

    /// Feed disjoint halves into two accumulators, merge, and compare against
    /// one accumulator fed everything. Returns `(whole, merged)`.
    fn merged(f: &AggFn, tys: &[DataType], params: &[Value], cols: &[Column]) -> (Value, Value) {
        let n = cols.first().map_or(0, |c| c.len());
        let idx = all(n);
        let mut whole = (f.new)(tys, params).unwrap();
        whole.update(cols, &idx).unwrap();

        let mid = n / 2;
        let mut a = (f.new)(tys, params).unwrap();
        let mut b = (f.new)(tys, params).unwrap();
        a.update(cols, &idx[..mid]).unwrap();
        b.update(cols, &idx[mid..]).unwrap();
        a.merge(b.as_ref()).unwrap();
        (whole.finish().unwrap(), a.finish().unwrap())
    }

    fn f(v: &Value) -> f64 {
        v.as_f64().expect("expected a numeric value")
    }

    // -------------------------------------------------------------- lookup

    #[test]
    fn lookup_resolves_base_names() {
        for n in [
            "count", "sum", "avg", "min", "max", "any", "anylast", "argmin", "argmax", "uniq",
            "uniqexact", "grouparray", "quantile", "quantileexact", "median", "varpop", "varsamp",
            "stddevpop", "stddevsamp",
        ] {
            assert!(lookup(n).is_some(), "missing aggregate {n}");
        }
        assert!(lookup("nosuchagg").is_none());
        assert!(lookup("").is_none());
        assert!(lookup("if").is_none());
    }

    #[test]
    fn lookup_accepts_standard_sql_spellings_but_not_window_names() {
        assert_eq!(lookup("var_pop").unwrap().name, "varPop");
        assert_eq!(lookup("variance").unwrap().name, "varSamp");
        assert_eq!(lookup("stddev").unwrap().name, "stddevSamp");
        assert_eq!(lookup("stddev_samp").unwrap().name, "stddevSamp");
        // These are window functions. The aggregate registry must not claim
        // them -- `window::lookup` consults the window names first and falls
        // through to here, so an entry added above would shadow the real
        // implementation rather than merely duplicate it.
        assert!(lookup("first_value").is_none());
        assert!(lookup("last_value").is_none());
        assert!(lookup("row_number").is_none());
        assert!(lookup("nth_value").is_none());
    }

    /// The other half of the split, from this side: every aggregate here is
    /// usable as a window function, and none of the window-only names resolves
    /// to an aggregate. This is what makes `sum(x) OVER (...)` reuse
    /// `SumAcc` instead of getting a second implementation.
    #[test]
    fn the_window_registry_falls_through_to_every_aggregate_here() {
        use crate::exec::operators::window::{self, WindowKind};
        for n in [
            "count", "sum", "avg", "min", "max", "any", "anyLast", "argMin", "uniq", "quantile",
            "median", "stddevPop", "sumIf",
        ] {
            let base = super::super::aggregate(n).expect("aggregate exists");
            match window::lookup(n) {
                Some(WindowKind::Agg(w)) => assert_eq!(
                    w.name, base.name,
                    "{n} resolves to a different function as a window than as an aggregate"
                ),
                other => panic!(
                    "{n} is an aggregate but not usable as a window function ({})",
                    other.map(|k| k.name()).unwrap_or("unresolved")
                ),
            }
        }
        for n in ["row_number", "rank", "dense_rank", "lag", "lead", "first_value"] {
            assert!(lookup(n).is_none(), "{n} leaked into the aggregate registry");
        }
    }

    #[test]
    fn lookup_resolves_if_combinator() {
        for (n, want) in [
            ("countif", "countIf"),
            ("sumif", "sumIf"),
            ("avgif", "avgIf"),
            ("minif", "minIf"),
            ("maxif", "maxIf"),
            ("uniqif", "uniqIf"),
            ("uniqexactif", "uniqExactIf"),
            ("anylastif", "anyLastIf"),
            ("stddevpopif", "stddevPopIf"),
        ] {
            assert_eq!(lookup(n).unwrap().name, want);
        }
        assert!(lookup("bogusif").is_none());
    }

    #[test]
    fn lookup_is_reached_through_the_public_facade() {
        // The facade lowercases; `agg::lookup` itself must not have to.
        assert_eq!(super::super::aggregate("SumIf").unwrap().name, "sumIf");
        assert_eq!(super::super::aggregate("uniqExact").unwrap().name, "uniqExact");
        assert!(super::super::is_aggregate("COUNT"));
    }

    #[test]
    fn distinct_flags_match_the_contract() {
        for n in ["count", "sum", "avg", "uniq", "uniqexact"] {
            assert!(lookup(n).unwrap().supports_distinct, "{n} should support DISTINCT");
        }
        for n in ["min", "max", "any", "argmin", "varpop"] {
            assert!(!lookup(n).unwrap().supports_distinct, "{n} should not");
        }
    }

    #[test]
    fn arity_checks_report_the_aggregate_name() {
        assert!(lookup("sum").unwrap().check_arity(1).is_ok());
        let e = lookup("sum").unwrap().check_arity(2).unwrap_err();
        assert!(e.to_string().contains("sum"), "{e}");
        assert!(lookup("count").unwrap().check_arity(0).is_ok());
        assert!(lookup("countif").unwrap().check_arity(0).is_err());
        assert!(lookup("argmin").unwrap().check_arity(2).is_ok());
    }

    // --------------------------------------------------------------- count

    #[test]
    fn count_star_counts_rows() {
        let c = lookup("count").unwrap();
        let mut a = (c.new)(&[], &[]).unwrap();
        a.update(&[], &all(1000)).unwrap();
        a.update(&[], &all(7)).unwrap();
        assert_eq!(a.finish().unwrap(), Value::UInt(1007));
    }

    #[test]
    fn count_arg_skips_nulls() {
        let c = lookup("count").unwrap();
        let col = nullable_ints(&[Some(1), None, Some(3), None]);
        assert_eq!(run(c, &[nullable_i64()], &[], &[col]), Value::UInt(2));
    }

    #[test]
    fn count_over_zero_rows_is_zero_not_null() {
        let c = lookup("count").unwrap();
        assert_eq!(run(c, &[NI64], &[], &[ints(&[])]), Value::UInt(0));
        let mut a = (c.new)(&[], &[]).unwrap();
        assert_eq!(a.finish().unwrap(), Value::UInt(0));
        a.update(&[], &[]).unwrap();
        assert_eq!(a.finish().unwrap(), Value::UInt(0));
    }

    // ----------------------------------------------------------------- sum

    /// The *base* type follows the input; the `Nullable` wrapper does not --
    /// it is unconditional, because an empty input sums to NULL regardless of
    /// how the argument was declared.
    #[test]
    fn sum_picks_return_type_from_input() {
        let s = lookup("sum").unwrap();
        let n = |t: DataType| DataType::Nullable(Box::new(t));
        assert_eq!((s.ret)(&[DataType::Int32], &[]).unwrap(), n(DataType::Int64));
        assert_eq!((s.ret)(&[DataType::UInt8], &[]).unwrap(), n(DataType::UInt64));
        assert_eq!((s.ret)(&[DataType::Float32], &[]).unwrap(), n(DataType::Float64));
        assert_eq!((s.ret)(&[DataType::Bool], &[]).unwrap(), n(DataType::UInt64));
        assert_eq!((s.ret)(&[nullable_i64()], &[]).unwrap(), n(DataType::Int64));
        assert!((s.ret)(&[DataType::String], &[]).is_err());
        assert!((s.ret)(&[DataType::Date], &[]).is_err());
    }

    #[test]
    fn sum_of_signed_and_unsigned() {
        let s = lookup("sum").unwrap();
        assert_eq!(run(s, &[NI64], &[], &[ints(&[1, -2, 3, -4])]), Value::Int(-2));
        assert_eq!(
            run(s, &[DataType::UInt64], &[], &[uints(&[10, 20, 30])]),
            Value::UInt(60)
        );
        assert_eq!(run(s, &[DataType::Bool], &[], &[bools(&[true, false, true])]), Value::UInt(2));
    }

    /// Inverted: this pinned the saturating narrowing, which was only ever a
    /// consequence of `finish` returning `Value` and having no way to refuse.
    /// `sum` of three `i64::MAX` used to answer `i64::MAX`, a number that
    /// compares, sorts and renders like the true total and is not it.
    #[test]
    fn sum_accumulates_beyond_i64_in_i128_and_refuses_to_narrow_what_does_not_fit() {
        let s = lookup("sum").unwrap();
        let e = try_run(s, &[NI64], &[], &[ints(&[i64::MAX, i64::MAX, i64::MAX])]).unwrap_err();
        assert!(e.to_string().contains("Int64"), "{e}");
        assert!(try_run(s, &[NI64], &[], &[ints(&[i64::MIN, i64::MIN])]).is_err());
        // The whole point of the i128 fold: an intermediate excursion past
        // i64 is not an error, because it comes back exactly.
        let v = run(s, &[NI64], &[], &[ints(&[i64::MAX, i64::MAX, i64::MIN, i64::MIN])]);
        assert_eq!(v, Value::Int(-2));
        // Both edges are themselves representable and must still be answered.
        assert_eq!(run(s, &[NI64], &[], &[ints(&[i64::MAX])]), Value::Int(i64::MAX));
        assert_eq!(run(s, &[NI64], &[], &[ints(&[i64::MIN])]), Value::Int(i64::MIN));
        // The unsigned narrowing has the same edge one range up.
        assert!(try_run(s, &[DataType::UInt64], &[], &[uints(&[u64::MAX, u64::MAX])]).is_err());
        assert_eq!(
            run(s, &[DataType::UInt64], &[], &[uints(&[u64::MAX])]),
            Value::UInt(u64::MAX)
        );
    }

    /// Inverted from `sum_of_empty_input_follows_nullability`, which pinned the
    /// bug: the answer used to be 0 for a non-Nullable argument (ClickHouse)
    /// and NULL for a Nullable one, so it turned on a property of the *column
    /// declaration* rather than on the data. Now NULL for both -- SQL standard
    /// and SQLite, and what `avg`/`min`/`max` next door already did.
    #[test]
    fn sum_of_empty_input_is_null_whatever_the_argument_was_declared() {
        let s = lookup("sum").unwrap();
        assert!(run(s, &[NI64], &[], &[ints(&[])]).is_null());
        assert!(run(s, &[DataType::UInt64], &[], &[uints(&[])]).is_null());
        assert!(run(s, &[DataType::Float64], &[], &[floats(&[])]).is_null());
        assert!(run(s, &[nullable_i64()], &[], &[nullable_ints(&[])]).is_null());
        // all-NULL Nullable input is also "zero non-null rows"
        let c = nullable_ints(&[None, None]);
        assert!(run(s, &[nullable_i64()], &[], &[c]).is_null());
        // One real row is enough to get the declared physical kind back, and
        // `eq_exact` is what checks that: plain `==` cannot tell `Int(1)` from
        // `UInt(1)` from `Float(1.0)`.
        assert!(run(s, &[NI64], &[], &[ints(&[1])]).eq_exact(&Value::Int(1)));
        assert!(run(s, &[DataType::UInt64], &[], &[uints(&[1])]).eq_exact(&Value::UInt(1)));
        assert!(run(s, &[DataType::Float64], &[], &[floats(&[1.0])]).eq_exact(&Value::Float(1.0)));
    }

    #[test]
    fn sum_rejects_a_float_column_under_an_integer_accumulator() {
        let s = lookup("sum").unwrap();
        let mut a = (s.new)(&[NI64], &[]).unwrap();
        // Type says Int64, data arrives as F64: a plan bug, not silent garbage.
        assert!(a.update(&[floats(&[1.0])], &[0]).is_err());
    }

    #[test]
    fn float_sum_uses_neumaier_compensation() {
        // Naive left-to-right summation of this sequence returns 0.0: each 1.0
        // is annihilated by the 1e100 magnitude. Neumaier recovers both.
        let data = [1.0, 1e100, 1.0, -1e100];
        let naive: f64 = data.iter().sum();
        assert_eq!(naive, 0.0, "the pathological input must actually break naive summation");

        let s = lookup("sum").unwrap();
        let got = run(s, &[DataType::Float64], &[], &[floats(&data)]);
        assert_eq!(got, Value::Float(2.0));
    }

    #[test]
    fn float_sum_compensation_survives_a_merge() {
        // Split the pathological sequence across two partials: the merge has to
        // carry both compensations, not just the high parts.
        let s = lookup("sum").unwrap();
        let (whole, m) = merged(s, &[DataType::Float64], &[], &[floats(&[1.0, 1e100, 1.0, -1e100])]);
        assert_eq!(f(&whole), 2.0);
        assert_eq!(f(&m), 2.0);
    }

    #[test]
    fn float_sum_beats_naive_on_many_small_addends() {
        // 1e9 rows is impractical in a unit test; 2^20 tiny addends on top of a
        // large base reproduce the same cancellation at a testable scale.
        let mut data = vec![1e9];
        data.extend(std::iter::repeat(1e-8).take(1 << 20));
        let exact = 1e9 + (1u64 << 20) as f64 * 1e-8;
        let naive: f64 = data.iter().sum();
        let s = lookup("sum").unwrap();
        let ours = f(&run(s, &[DataType::Float64], &[], &[floats(&data)]));
        assert!(
            (ours - exact).abs() <= (naive - exact).abs(),
            "compensated {ours} should be no worse than naive {naive} (exact {exact})"
        );
        assert!((ours - exact).abs() < 1e-6, "compensated sum drifted: {ours} vs {exact}");
    }

    // ----------------------------------------------------------------- avg

    #[test]
    fn avg_is_sum_over_count() {
        let a = lookup("avg").unwrap();
        assert_eq!(f(&run(a, &[NI64], &[], &[ints(&[1, 2, 3, 4])])), 2.5);
        assert_eq!((a.ret)(&[NI64], &[]).unwrap(), DataType::Float64);
    }

    #[test]
    fn avg_skips_nulls_and_returns_null_when_empty() {
        let a = lookup("avg").unwrap();
        let c = nullable_ints(&[Some(2), None, Some(4)]);
        assert_eq!(f(&run(a, &[nullable_i64()], &[], &[c])), 3.0);
        assert_eq!(run(a, &[NI64], &[], &[ints(&[])]), Value::Null);
        assert_eq!(run(a, &[nullable_i64()], &[], &[nullable_ints(&[None])]), Value::Null);
    }

    // ----------------------------------------------------------- min / max

    #[test]
    fn min_max_over_numbers() {
        let (mn, mx) = (lookup("min").unwrap(), lookup("max").unwrap());
        let c = ints(&[5, -3, 9, 0]);
        assert_eq!(run(mn, &[NI64], &[], &[c.clone()]), Value::Int(-3));
        assert_eq!(run(mx, &[NI64], &[], &[c]), Value::Int(9));
        let fc = floats(&[1.5, -0.5, 2.25]);
        assert_eq!(f(&run(mn, &[DataType::Float64], &[], &[fc.clone()])), -0.5);
        assert_eq!(f(&run(mx, &[DataType::Float64], &[], &[fc])), 2.25);
    }

    #[test]
    fn min_max_over_strings() {
        let (mn, mx) = (lookup("min").unwrap(), lookup("max").unwrap());
        let c = strs(&["pear", "apple", "quince"]);
        assert_eq!(run(mn, &[DataType::String], &[], &[c.clone()]), Value::str("apple"));
        assert_eq!(run(mx, &[DataType::String], &[], &[c]), Value::str("quince"));
    }

    #[test]
    fn min_max_skip_nulls_and_empty_is_null() {
        let mn = lookup("min").unwrap();
        let c = nullable_ints(&[None, Some(7), None, Some(3)]);
        assert_eq!(run(mn, &[nullable_i64()], &[], &[c]), Value::Int(3));
        assert_eq!(run(mn, &[nullable_i64()], &[], &[nullable_ints(&[None, None])]), Value::Null);
        assert_eq!(run(mn, &[NI64], &[], &[ints(&[])]), Value::Null);
    }

    #[test]
    fn min_max_preserve_the_logical_type() {
        let mx = lookup("max").unwrap();
        let dates = Column::u64s(DataType::Date, vec![19_723, 0, 19_000]);
        assert_eq!(run(mx, &[DataType::Date], &[], &[dates]), Value::Date(19_723));
        assert_eq!((mx.ret)(&[DataType::Date], &[]).unwrap(), DataType::Date);
    }

    #[test]
    fn min_and_max_refuse_to_merge_into_each_other() {
        let (mn, mx) = (lookup("min").unwrap(), lookup("max").unwrap());
        let mut a = (mn.new)(&[NI64], &[]).unwrap();
        let b = (mx.new)(&[NI64], &[]).unwrap();
        let e = a.merge(b.as_ref()).unwrap_err();
        assert!(matches!(e, Error::Exec(_)), "{e}");
    }

    // ------------------------------------------------------ any / anyLast

    #[test]
    fn any_takes_the_first_non_null_and_anylast_the_last() {
        let (an, al) = (lookup("any").unwrap(), lookup("anylast").unwrap());
        let c = nullable_ints(&[None, Some(2), Some(3), None]);
        assert_eq!(run(an, &[nullable_i64()], &[], &[c.clone()]), Value::Int(2));
        assert_eq!(run(al, &[nullable_i64()], &[], &[c]), Value::Int(3));
    }

    #[test]
    fn any_is_stable_across_repeated_updates() {
        let an = lookup("any").unwrap();
        let mut a = (an.new)(&[NI64], &[]).unwrap();
        a.update(&[ints(&[7, 8])], &all(2)).unwrap();
        a.update(&[ints(&[9])], &all(1)).unwrap();
        assert_eq!(a.finish().unwrap(), Value::Int(7));

        let al = lookup("anylast").unwrap();
        let mut b = (al.new)(&[NI64], &[]).unwrap();
        b.update(&[ints(&[7, 8])], &all(2)).unwrap();
        b.update(&[ints(&[9])], &all(1)).unwrap();
        assert_eq!(b.finish().unwrap(), Value::Int(9));
    }

    // ------------------------------------------------------ argMin/argMax

    #[test]
    fn argmin_argmax_pick_by_the_key_column() {
        let (amn, amx) = (lookup("argmin").unwrap(), lookup("argmax").unwrap());
        let vals = strs(&["a", "b", "c", "d"]);
        let keys = ints(&[10, 3, 42, 7]);
        let tys = [DataType::String, NI64];
        assert_eq!(run(amn, &tys, &[], &[vals.clone(), keys.clone()]), Value::str("b"));
        assert_eq!(run(amx, &tys, &[], &[vals, keys]), Value::str("c"));
    }

    #[test]
    fn argmin_breaks_ties_toward_the_first_row_and_skips_null_keys() {
        let amn = lookup("argmin").unwrap();
        let vals = strs(&["first", "second", "third"]);
        let keys = nullable_ints(&[Some(5), Some(5), None]);
        let tys = [DataType::String, nullable_i64()];
        assert_eq!(run(amn, &tys, &[], &[vals, keys]), Value::str("first"));

        // all keys NULL -> nothing selected
        let vals = strs(&["x"]);
        let keys = nullable_ints(&[None]);
        assert_eq!(run(amn, &tys, &[], &[vals, keys]), Value::Null);
    }

    #[test]
    fn argmin_returns_nullable_of_the_value_type() {
        let amn = lookup("argmin").unwrap();
        assert_eq!(
            (amn.ret)(&[DataType::String, NI64], &[]).unwrap(),
            DataType::Nullable(Box::new(DataType::String))
        );
        assert!((amn.ret)(&[DataType::String], &[]).is_err());
    }

    // ----------------------------------------------------- uniq/uniqExact

    #[test]
    fn uniq_exact_counts_distinct_numbers_and_strings() {
        let u = lookup("uniqexact").unwrap();
        let c = ints(&[1, 2, 2, 3, 3, 3]);
        assert_eq!(run(u, &[NI64], &[], &[c]), Value::UInt(3));
        let s = strs(&["a", "b", "a", "c", "b"]);
        assert_eq!(run(u, &[DataType::String], &[], &[s]), Value::UInt(3));
        assert_eq!(run(u, &[NI64], &[], &[ints(&[])]), Value::UInt(0));
    }

    #[test]
    fn uniq_exact_is_exact_above_2_to_the_53() {
        // Values that would collide if we counted distinctness through f64.
        let u = lookup("uniqexact").unwrap();
        let c = uints(&[(1u64 << 63) + 1, (1u64 << 63) + 2, (1u64 << 63) + 3]);
        assert_eq!(run(u, &[DataType::UInt64], &[], &[c]), Value::UInt(3));
    }

    #[test]
    fn uniq_exact_skips_nulls() {
        let u = lookup("uniqexact").unwrap();
        let c = nullable_ints(&[Some(1), None, Some(1), None, Some(2)]);
        assert_eq!(run(u, &[nullable_i64()], &[], &[c]), Value::UInt(2));
    }

    #[test]
    fn hll_is_within_two_percent_on_100k_distinct_values() {
        let u = lookup("uniq").unwrap();
        let vals: Vec<i64> = (0..100_000i64).map(|i| i * 7 + 11).collect();
        let got = run(u, &[NI64], &[], &[ints(&vals)]).as_u64().unwrap() as f64;
        let err = (got - 100_000.0).abs() / 100_000.0;
        assert!(err < 0.02, "HLL error {err:.4} on 100k distinct (got {got})");
    }

    #[test]
    fn hll_uses_linear_counting_for_small_cardinalities() {
        let u = lookup("uniq").unwrap();
        for n in [0i64, 1, 10, 1000] {
            let vals: Vec<i64> = (0..n).collect();
            let got = run(u, &[NI64], &[], &[ints(&vals)]).as_u64().unwrap() as i64;
            // Linear counting is essentially exact while registers are sparse.
            assert!(
                (got - n).abs() <= (n / 50).max(1),
                "uniq({n}) = {got}, expected near-exact"
            );
        }
    }

    #[test]
    fn hll_counts_strings_and_ignores_duplicates() {
        let u = lookup("uniq").unwrap();
        let mut v: Vec<&str> = Vec::new();
        let owned: Vec<String> = (0..500).map(|i| format!("user-{i}")).collect();
        for s in &owned {
            v.push(s);
            v.push(s); // every key twice
        }
        let got = run(u, &[DataType::String], &[], &[strs(&v)]).as_u64().unwrap() as i64;
        assert!((got - 500).abs() <= 10, "uniq over 500 distinct strings = {got}");
    }

    #[test]
    fn hll_merge_is_register_wise_max() {
        // Two sketches over overlapping ranges must estimate the union.
        let u = lookup("uniq").unwrap();
        let a_vals: Vec<i64> = (0..30_000).collect();
        let b_vals: Vec<i64> = (20_000..50_000).collect();
        let mut a = (u.new)(&[NI64], &[]).unwrap();
        let mut b = (u.new)(&[NI64], &[]).unwrap();
        a.update(&[ints(&a_vals)], &all(a_vals.len())).unwrap();
        b.update(&[ints(&b_vals)], &all(b_vals.len())).unwrap();
        a.merge(b.as_ref()).unwrap();
        let got = a.finish().unwrap().as_u64().unwrap() as f64;
        let err = (got - 50_000.0).abs() / 50_000.0;
        assert!(err < 0.02, "merged HLL error {err:.4} (got {got})");
    }

    #[test]
    fn hll_merge_is_idempotent_for_identical_sketches() {
        let u = lookup("uniq").unwrap();
        let vals: Vec<i64> = (0..5000).collect();
        let mut a = (u.new)(&[NI64], &[]).unwrap();
        let mut b = (u.new)(&[NI64], &[]).unwrap();
        a.update(&[ints(&vals)], &all(vals.len())).unwrap();
        b.update(&[ints(&vals)], &all(vals.len())).unwrap();
        let before = a.finish().unwrap();
        a.merge(b.as_ref()).unwrap();
        assert_eq!(before, a.finish().unwrap(), "union with itself must not change the estimate");
    }

    // ---------------------------------------------------------- groupArray

    #[test]
    fn group_array_joins_with_commas() {
        let g = lookup("grouparray").unwrap();
        assert_eq!(
            run(g, &[NI64], &[], &[ints(&[3, 1, 2])]),
            Value::str("3,1,2")
        );
        assert_eq!(
            run(g, &[DataType::String], &[], &[strs(&["a", "b"])]),
            Value::str("a,b")
        );
        assert_eq!((g.ret)(&[NI64], &[]).unwrap(), DataType::String);
    }

    #[test]
    fn group_array_skips_nulls_and_caps_length() {
        let g = lookup("grouparray").unwrap();
        let c = nullable_ints(&[Some(1), None, Some(2)]);
        assert_eq!(run(g, &[nullable_i64()], &[], &[c]), Value::str("1,2"));
        assert_eq!(run(g, &[NI64], &[], &[ints(&[])]), Value::Null);

        let big: Vec<i64> = (0..GROUP_ARRAY_LIMIT as i64 + 500).collect();
        let v = run(g, &[NI64], &[], &[ints(&big)]);
        let n = v.as_str().unwrap().split(',').count();
        assert_eq!(n, GROUP_ARRAY_LIMIT);

        // The explicit parameter narrows the cap.
        let v = run(g, &[NI64], &[Value::UInt(3)], &[ints(&[9, 8, 7, 6, 5])]);
        assert_eq!(v, Value::str("9,8,7"));
    }

    // ------------------------------------------------------------ quantile

    #[test]
    fn quantile_matches_a_sorted_reference() {
        let q = lookup("quantile").unwrap();
        // 0..=100 shuffled by a stride; sorted it is exactly 0..=100, so the
        // p-th interpolated quantile is 100*p for every p.
        let vals: Vec<i64> = (0..101).map(|i: i64| (i * 37) % 101).collect();
        let mut sorted = vals.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..101).collect::<Vec<i64>>());

        for p in [0.0, 0.1, 0.25, 0.5, 0.9, 0.99, 1.0] {
            let got = f(&run(q, &[NI64], &[Value::Float(p)], &[ints(&vals)]));
            let pos = p * (sorted.len() - 1) as f64;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(sorted.len() - 1);
            let want = sorted[lo] as f64 + (sorted[hi] - sorted[lo]) as f64 * (pos - lo as f64);
            assert!((got - want).abs() < 1e-9, "quantile({p}) = {got}, want {want}");
        }
    }

    #[test]
    fn quantile_interpolates_between_neighbours() {
        let q = lookup("quantile").unwrap();
        // n = 4, p = 0.5 -> pos 1.5 -> halfway between 2 and 3.
        assert_eq!(f(&run(q, &[NI64], &[Value::Float(0.5)], &[ints(&[1, 2, 3, 4])])), 2.5);
    }

    #[test]
    fn quantile_exact_returns_an_observed_element() {
        let qe = lookup("quantileexact").unwrap();
        // n = 4, p = 0.5 -> index floor(2.0) = 2 -> the value 3, never 2.5.
        assert_eq!(f(&run(qe, &[NI64], &[Value::Float(0.5)], &[ints(&[1, 2, 3, 4])])), 3.0);
        assert_eq!(f(&run(qe, &[NI64], &[Value::Float(1.0)], &[ints(&[1, 2, 3, 4])])), 4.0);
        assert_eq!(f(&run(qe, &[NI64], &[Value::Float(0.0)], &[ints(&[1, 2, 3, 4])])), 1.0);
    }

    #[test]
    fn median_defaults_to_the_half_quantile() {
        let m = lookup("median").unwrap();
        assert_eq!(f(&run(m, &[NI64], &[], &[ints(&[5, 1, 3])])), 3.0);
        assert_eq!(f(&run(m, &[NI64], &[], &[ints(&[4, 1, 3, 2])])), 2.5);
        assert_eq!(run(m, &[NI64], &[], &[ints(&[])]), Value::Null);
        // median takes no parameters
        assert!((m.ret)(&[NI64], &[Value::Float(0.9)]).is_err());
    }

    #[test]
    fn quantile_validates_its_level() {
        let q = lookup("quantile").unwrap();
        assert!((q.ret)(&[NI64], &[Value::Float(1.5)]).is_err());
        assert!((q.ret)(&[NI64], &[Value::Float(-0.1)]).is_err());
        assert!((q.ret)(&[NI64], &[Value::str("half")]).is_err());
        assert!((q.ret)(&[NI64], &[Value::Float(0.5), Value::Float(0.9)]).is_err());
        assert!((q.ret)(&[DataType::String], &[]).is_err());
        assert!((q.new)(&[NI64], &[Value::Float(2.0)]).is_err());
    }

    // ----------------------------------------------------- variance family

    /// Textbook two-pass variance, used only as a test oracle.
    fn two_pass(v: &[f64], sample: bool) -> f64 {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let ss: f64 = v.iter().map(|x| (x - mean) * (x - mean)).sum();
        ss / if sample { n - 1.0 } else { n }
    }

    #[test]
    fn welford_matches_the_two_pass_formula() {
        let data: Vec<f64> = (0..37).map(|i| (i as f64 * 1.7).sin() * 100.0 + 5.0).collect();
        let col = floats(&data);
        let t = [DataType::Float64];
        let got_pop = f(&run(lookup("varpop").unwrap(), &t, &[], &[col.clone()]));
        let got_samp = f(&run(lookup("varsamp").unwrap(), &t, &[], &[col.clone()]));
        assert!((got_pop - two_pass(&data, false)).abs() < 1e-9, "{got_pop}");
        assert!((got_samp - two_pass(&data, true)).abs() < 1e-9, "{got_samp}");

        let sd_pop = f(&run(lookup("stddevpop").unwrap(), &t, &[], &[col.clone()]));
        let sd_samp = f(&run(lookup("stddevsamp").unwrap(), &t, &[], &[col]));
        assert!((sd_pop - two_pass(&data, false).sqrt()).abs() < 1e-9);
        assert!((sd_samp - two_pass(&data, true).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn welford_survives_a_huge_offset() {
        // The naive E[x^2]-E[x]^2 form returns garbage (often negative) here:
        // the mean is ~1.7e9 and the spread is 1.
        let data: Vec<f64> = (0..1000).map(|i| 1.7e9 + i as f64).collect();
        let got = f(&run(lookup("varpop").unwrap(), &[DataType::Float64], &[], &[floats(&data)]));
        let want = two_pass(&data, false);
        assert!((got - want).abs() / want < 1e-9, "got {got}, want {want}");
        assert!(got > 0.0);
    }

    #[test]
    fn variance_needs_enough_rows() {
        let t = [DataType::Float64];
        let one = floats(&[42.0]);
        assert_eq!(run(lookup("varsamp").unwrap(), &t, &[], &[one.clone()]), Value::Null);
        assert_eq!(run(lookup("stddevsamp").unwrap(), &t, &[], &[one.clone()]), Value::Null);
        assert_eq!(f(&run(lookup("varpop").unwrap(), &t, &[], &[one])), 0.0);
        assert_eq!(run(lookup("varpop").unwrap(), &t, &[], &[floats(&[])]), Value::Null);
    }

    #[test]
    fn variance_flavours_do_not_merge_across_kinds() {
        let mut a = (lookup("varpop").unwrap().new)(&[DataType::Float64], &[]).unwrap();
        let b = (lookup("varsamp").unwrap().new)(&[DataType::Float64], &[]).unwrap();
        assert!(a.merge(b.as_ref()).is_err());
    }

    // ---------------------------------------------------- merge round-trips

    #[test]
    fn every_accumulator_merges_two_halves_into_the_whole() {
        let ints_col = ints(&[5, -3, 9, 0, 12, -7, 4, 4]);
        let f_col = floats(&[1.5, -0.5, 2.25, 8.0, 3.0, -9.5, 0.25, 6.0]);
        let s_col = strs(&["pear", "apple", "fig", "quince", "date", "berry", "kiwi", "lime"]);
        let key_col = ints(&[10, 3, 42, 7, 1, 99, 55, 2]);
        let t_i = [NI64];
        let t_f = [DataType::Float64];
        let t_s = [DataType::String];

        let cases: Vec<(&str, &[DataType], Vec<Value>, Vec<Column>)> = vec![
            ("count", &t_i, vec![], vec![ints_col.clone()]),
            ("sum", &t_i, vec![], vec![ints_col.clone()]),
            ("sum", &t_f, vec![], vec![f_col.clone()]),
            ("avg", &t_i, vec![], vec![ints_col.clone()]),
            ("min", &t_i, vec![], vec![ints_col.clone()]),
            ("max", &t_i, vec![], vec![ints_col.clone()]),
            ("min", &t_s, vec![], vec![s_col.clone()]),
            ("max", &t_s, vec![], vec![s_col.clone()]),
            ("any", &t_i, vec![], vec![ints_col.clone()]),
            ("anylast", &t_i, vec![], vec![ints_col.clone()]),
            ("uniq", &t_i, vec![], vec![ints_col.clone()]),
            ("uniqexact", &t_i, vec![], vec![ints_col.clone()]),
            ("uniqexact", &t_s, vec![], vec![s_col.clone()]),
            ("grouparray", &t_i, vec![], vec![ints_col.clone()]),
            ("quantile", &t_i, vec![Value::Float(0.75)], vec![ints_col.clone()]),
            ("quantileexact", &t_i, vec![Value::Float(0.25)], vec![ints_col.clone()]),
            ("median", &t_i, vec![], vec![ints_col.clone()]),
            ("varpop", &t_f, vec![], vec![f_col.clone()]),
            ("varsamp", &t_f, vec![], vec![f_col.clone()]),
            ("stddevpop", &t_f, vec![], vec![f_col.clone()]),
            ("stddevsamp", &t_f, vec![], vec![f_col.clone()]),
        ];

        for (name, tys, params, cols) in cases {
            let f = lookup(name).unwrap();
            let (whole, m) = merged(f, tys, &params, &cols);
            assert_eq!(whole, m, "{name}: merge disagrees with a single pass");
            assert!(!whole.is_null(), "{name}: fixture should produce a value");
        }

        // Two-argument aggregates need their own fixture shape.
        for name in ["argmin", "argmax"] {
            let f = lookup(name).unwrap();
            let (whole, m) = merged(f, &[NI64, NI64], &[], &[ints_col.clone(), key_col.clone()]);
            assert_eq!(whole, m, "{name}: merge disagrees with a single pass");
        }
    }

    #[test]
    fn merge_rejects_a_foreign_accumulator() {
        let mut s = (lookup("sum").unwrap().new)(&[NI64], &[]).unwrap();
        let c = (lookup("count").unwrap().new)(&[NI64], &[]).unwrap();
        let e = s.merge(c.as_ref()).unwrap_err();
        assert!(matches!(e, Error::Exec(_)), "{e}");
        assert!(e.to_string().contains("sum"), "{e}");

        let mut u = (lookup("uniq").unwrap().new)(&[NI64], &[]).unwrap();
        assert!(u.merge(c.as_ref()).is_err());

        let mut q = (lookup("quantile").unwrap().new)(&[NI64], &[]).unwrap();
        assert!(q.merge(c.as_ref()).is_err());
    }

    #[test]
    fn merging_an_empty_partial_is_a_no_op() {
        for name in ["sum", "avg", "min", "max", "uniq", "uniqexact", "varsamp", "median", "count"] {
            let f = lookup(name).unwrap();
            let mut a = (f.new)(&[NI64], &[]).unwrap();
            a.update(&[ints(&[1, 2, 3, 4])], &all(4)).unwrap();
            let before = a.finish().unwrap();
            let empty = (f.new)(&[NI64], &[]).unwrap();
            a.merge(empty.as_ref()).unwrap();
            let after = a.finish().unwrap();
            assert_eq!(before, after, "{name}: merging an empty partial changed the result");
        }
    }

    #[test]
    fn boxed_clone_yields_an_empty_accumulator_of_the_same_shape() {
        for (name, tys) in [
            ("sum", &[NI64][..]),
            ("avg", &[NI64][..]),
            ("min", &[NI64][..]),
            ("uniq", &[NI64][..]),
            ("uniqexact", &[DataType::String][..]),
            ("median", &[NI64][..]),
            ("varsamp", &[NI64][..]),
            ("grouparray", &[NI64][..]),
        ] {
            let f = lookup(name).unwrap();
            let mut a = (f.new)(tys, &[]).unwrap();
            if name == "uniqexact" {
                a.update(&[strs(&["x", "y"])], &all(2)).unwrap();
            } else {
                a.update(&[ints(&[1, 2, 3])], &all(3)).unwrap();
            }
            let fresh = a.boxed_clone();
            let virgin = (f.new)(tys, &[]).unwrap();
            let empty = virgin.finish().unwrap();
            assert_eq!(fresh.finish().unwrap(), empty, "{name}: clone was not empty");
            // ...and the original is untouched.
            assert_ne!(a.finish().unwrap(), empty, "{name}: original was disturbed");
        }
    }

    #[test]
    fn finish_is_repeatable() {
        for name in ["sum", "avg", "min", "uniq", "uniqexact", "median", "varpop", "grouparray"] {
            let f = lookup(name).unwrap();
            let mut a = (f.new)(&[NI64], &[]).unwrap();
            a.update(&[ints(&[3, 1, 4, 1, 5])], &all(5)).unwrap();
            let (x, y) = (a.finish().unwrap(), a.finish().unwrap());
            assert_eq!(x, y, "{name}: finish is not idempotent");
        }
    }

    #[test]
    fn a_selection_vector_folds_only_the_named_rows() {
        let s = lookup("sum").unwrap();
        let mut a = (s.new)(&[NI64], &[]).unwrap();
        a.update(&[ints(&[1, 2, 3, 4, 5])], &[0, 2, 4]).unwrap();
        assert_eq!(a.finish().unwrap(), Value::Int(9));
        // Repeated indices are folded repeatedly -- `sel` is a list, not a set.
        let mut b = (s.new)(&[NI64], &[]).unwrap();
        b.update(&[ints(&[7])], &[0, 0, 0]).unwrap();
        assert_eq!(b.finish().unwrap(), Value::Int(21));
    }

    // -------------------------------------------------------- -If variants

    #[test]
    fn count_if_counts_only_true_rows() {
        let c = lookup("countif").unwrap();
        let cond = bools(&[true, false, true, true]);
        let mut a = (c.new)(&[DataType::Bool], &[]).unwrap();
        a.update(&[cond], &all(4)).unwrap();
        assert_eq!(a.finish().unwrap(), Value::UInt(3));
    }

    #[test]
    fn sum_if_and_avg_if_respect_the_predicate() {
        let tys = [NI64, DataType::Bool];
        let vals = ints(&[10, 20, 30, 40]);
        let cond = bools(&[true, false, true, false]);
        let s = lookup("sumif").unwrap();
        assert_eq!(run(s, &tys, &[], &[vals.clone(), cond.clone()]), Value::Int(40));
        let a = lookup("avgif").unwrap();
        assert_eq!(f(&run(a, &tys, &[], &[vals, cond])), 20.0);
    }

    #[test]
    fn min_max_uniq_if_variants_work() {
        let tys = [NI64, DataType::Bool];
        let vals = ints(&[5, 1, 9, 2]);
        let cond = bools(&[false, false, true, true]);
        assert_eq!(
            run(lookup("minif").unwrap(), &tys, &[], &[vals.clone(), cond.clone()]),
            Value::Int(2)
        );
        assert_eq!(
            run(lookup("maxif").unwrap(), &tys, &[], &[vals.clone(), cond.clone()]),
            Value::Int(9)
        );
        assert_eq!(
            run(lookup("uniqif").unwrap(), &tys, &[], &[vals.clone(), cond.clone()]),
            Value::UInt(2)
        );
        assert_eq!(
            run(lookup("uniqexactif").unwrap(), &tys, &[], &[vals, cond]),
            Value::UInt(2)
        );
    }

    #[test]
    fn a_null_predicate_is_not_true() {
        let tys = [NI64, nullable_i64()];
        let vals = ints(&[1, 2, 3]);
        // row 1's condition is NULL: neither true nor counted.
        let cond = nullable_ints(&[Some(1), None, Some(1)]);
        assert_eq!(run(lookup("sumif").unwrap(), &tys, &[], &[vals, cond]), Value::Int(4));
    }

    #[test]
    fn an_all_false_predicate_yields_the_empty_aggregate() {
        let tys = [NI64, DataType::Bool];
        let vals = ints(&[1, 2, 3]);
        let cond = bools(&[false, false, false]);
        // `sumIf` inherits `sum`'s empty-set rule, so this is NULL and not 0.
        assert!(run(lookup("sumif").unwrap(), &tys, &[], &[vals.clone(), cond.clone()]).is_null());
        assert_eq!(run(lookup("countif").unwrap(), &[DataType::Bool], &[], &[cond.clone()]), Value::UInt(0));
        assert_eq!(run(lookup("minif").unwrap(), &tys, &[], &[vals.clone(), cond.clone()]), Value::Null);
        assert_eq!(run(lookup("avgif").unwrap(), &tys, &[], &[vals, cond]), Value::Null);
    }

    #[test]
    fn if_variants_report_the_inner_return_type_and_arity() {
        let s = lookup("sumif").unwrap();
        assert_eq!(
            (s.ret)(&[DataType::Int32, DataType::Bool], &[]).unwrap(),
            DataType::Nullable(Box::new(DataType::Int64))
        );
        assert_eq!(s.arity, (2, 2));
        assert_eq!(lookup("countif").unwrap().arity, (1, 2));
        // the trailing condition may not be a String
        assert!((s.ret)(&[NI64, DataType::String], &[]).is_err());
        // and it must be present at all
        assert!((s.ret)(&[], &[]).is_err());
        assert!((s.new)(&[], &[]).is_err());
        // parametric combinators pass their params through
        let q = lookup("quantileif").unwrap();
        assert_eq!((q.ret)(&[NI64, DataType::Bool], &[Value::Float(0.9)]).unwrap(), DataType::Float64);
        assert!((q.ret)(&[NI64, DataType::Bool], &[Value::Float(9.0)]).is_err());
    }

    #[test]
    fn if_variants_merge_through_the_wrapper() {
        let tys = [NI64, DataType::Bool];
        let vals = ints(&[10, 20, 30, 40, 50, 60]);
        let cond = bools(&[true, false, true, true, false, true]);
        let (whole, m) = merged(lookup("sumif").unwrap(), &tys, &[], &[vals.clone(), cond.clone()]);
        assert_eq!(whole, Value::Int(140)); // 10 + 30 + 40 + 60
        assert_eq!(whole, m);

        let (whole, m) = merged(lookup("uniqexactif").unwrap(), &tys, &[], &[vals, cond]);
        assert_eq!(whole, m);
    }

    #[test]
    fn if_variants_reject_a_foreign_merge_partner() {
        let mut a = (lookup("sumif").unwrap().new)(&[NI64, DataType::Bool], &[]).unwrap();
        let b = (lookup("countif").unwrap().new)(&[NI64, DataType::Bool], &[]).unwrap();
        // Both are CondAcc; the inner accumulators must still refuse.
        assert!(a.merge(b.as_ref()).is_err());
        let plain = (lookup("sum").unwrap().new)(&[NI64], &[]).unwrap();
        assert!(a.merge(plain.as_ref()).is_err());
    }

    #[test]
    fn if_variant_boxed_clone_keeps_the_wrapper() {
        let tys = [NI64, DataType::Bool];
        let f = lookup("sumif").unwrap();
        let mut a = (f.new)(&tys, &[]).unwrap();
        a.update(&[ints(&[5]), bools(&[true])], &[0]).unwrap();
        let mut fresh = a.boxed_clone();
        // A fresh clone has seen no rows, which for `sum` is NULL, not 0.
        assert!(fresh.finish().unwrap().is_null());
        fresh.update(&[ints(&[9]), bools(&[true])], &[0]).unwrap();
        assert_eq!(fresh.finish().unwrap(), Value::Int(9));
        assert_eq!(a.finish().unwrap(), Value::Int(5));
    }

    #[test]
    fn if_variant_rejects_a_string_condition_at_runtime() {
        let f = lookup("countif").unwrap();
        let mut a = (f.new)(&[DataType::Bool], &[]).unwrap();
        assert!(a.update(&[strs(&["yes"])], &[0]).is_err());
        assert!(a.update(&[], &[0]).is_err());
    }

    // ------------------------------------------------------------- decimals
    //
    // Every one of these pinned a shipped wrong answer: a `Decimal64(2)` lane
    // is a count of hundredths, and four accumulators here read it as the
    // number itself. `sum(price)` over 3.81 and 1.19 answered 500.
    //
    // The second wave is about *magnitude* rather than scale, and it shipped
    // because these tests all used two- and three-digit lanes. `avg`, `sum` and
    // the interpolating quantiles clamped instead of refusing, and
    // `quantileExact` kept its lanes in an `f64`; between them, four aggregates
    // answered numbers the column did not contain. Every case below that names
    // a magnitude is one of those, and the end-to-end pins are in
    // `tests/decimal_exactness.rs`.

    fn decs(scale: u8, v: &[i64]) -> Column {
        Column::i64s(DataType::Decimal64(scale), v.to_vec())
    }
    const DEC2: DataType = DataType::Decimal64(2);

    #[test]
    fn sum_of_a_decimal_keeps_the_argument_scale() {
        let s = lookup("sum").unwrap();
        assert_eq!(
            (s.ret)(&[DEC2], &[]).unwrap(),
            DataType::Nullable(Box::new(DataType::Decimal64(2)))
        );
        // 381 + 119 = 500 units of 0.01, which is 5.00 -- not 500.
        let got = run(s, &[DEC2], &[], &[decs(2, &[381, 119])]);
        assert!(got.eq_exact(&Value::Decimal(500, 2)), "{got:?}");
        assert_eq!(got.render_plain(), "5.00");
    }

    /// The property that makes this design worth having: the fold is already
    /// an exact `i128` count of the argument's own units, so a decimal sum
    /// costs nothing extra and loses nothing.
    #[test]
    fn sum_of_a_decimal_is_exact_where_the_float_route_is_not() {
        assert_ne!(0.1f64 + 0.2, 0.3, "the canonical f64 failure must actually fail");
        let s = lookup("sum").unwrap();
        let got = run(s, &[DataType::Decimal64(1)], &[], &[decs(1, &[1, 2])]);
        assert!(got.eq_exact(&Value::Decimal(3, 1)), "{got:?}");
        assert_eq!(got.render_plain(), "0.3");
    }

    /// The same lanes under `Int64` must fold to the same number: the decimal
    /// path *is* the integer path, with the point put back at the end.
    #[test]
    fn a_decimal_sums_to_the_same_lane_as_the_identical_int64_column() {
        let s = lookup("sum").unwrap();
        let lanes = [381i64, 119, -7, i32::MAX as i64];
        let d = run(s, &[DEC2], &[], &[decs(2, &lanes)]);
        let i = run(s, &[NI64], &[], &[ints(&lanes)]);
        assert_eq!(d.as_i64(), i.as_i64());
        assert!(!d.same_variant(&i), "only the reported type differs");
    }

    /// Inverted: this pinned a clamp at the decimal range, which is how two
    /// rows of 5000000000000000.00 summed to 9999999999999999.99 *and*
    /// `sum(p) = 10000000000000000.00` evaluated TRUE on the same rows -- a
    /// wrong answer nothing downstream could distinguish from the right one.
    #[test]
    fn sum_of_a_decimal_refuses_to_leave_the_decimal_range() {
        let s = lookup("sum").unwrap();
        let max = crate::types::value::DECIMAL_MAX_UNITS as i64;
        for lanes in [[i64::MAX, i64::MAX], [i64::MIN, i64::MIN], [max, 1], [-max, -1]] {
            let e = try_run(s, &[DEC2], &[], &[decs(2, &lanes)]).unwrap_err();
            assert!(e.to_string().contains("Decimal64(2)"), "{lanes:?}: {e}");
        }
        // The edge itself is representable and is still answered, exactly.
        let got = run(s, &[DEC2], &[], &[decs(2, &[max - 1, 1])]);
        assert!(got.eq_exact(&Value::Decimal(max, 2)), "{got:?}");
        let got = run(s, &[DEC2], &[], &[decs(2, &[-(max - 1), -1])]);
        assert!(got.eq_exact(&Value::Decimal(-max, 2)), "{got:?}");
    }

    #[test]
    fn avg_of_a_decimal_divides_exactly_at_six_digits() {
        let a = lookup("avg").unwrap();
        assert_eq!((a.ret)(&[DEC2], &[]).unwrap(), DataType::Decimal64(6));
        let got = run(a, &[DEC2], &[], &[decs(2, &[381, 119])]);
        assert!(got.eq_exact(&Value::Decimal(2_500_000, 6)), "{got:?}");
        assert_eq!(got.render_plain(), "2.500000");
        // A third, which no binary float lands on: rounded half away from zero
        // at the last kept digit, both directions.
        let got = run(a, &[DEC2], &[], &[decs(2, &[100, 0, 0])]);
        assert!(got.eq_exact(&Value::Decimal(333_333, 6)), "{got:?}");
        let got = run(a, &[DEC2], &[], &[decs(2, &[200, 0, 0])]);
        assert!(got.eq_exact(&Value::Decimal(666_667, 6)), "{got:?}");
        let got = run(a, &[DEC2], &[], &[decs(2, &[-200, 0, 0])]);
        assert!(got.eq_exact(&Value::Decimal(-666_667, 6)), "{got:?}");
    }

    /// Six is a floor, not a rule: an argument that already carries more
    /// digits keeps them.
    #[test]
    fn avg_of_a_high_scale_decimal_keeps_its_own_scale() {
        let a = lookup("avg").unwrap();
        let t = [DataType::Decimal64(9)];
        assert_eq!((a.ret)(&t, &[]).unwrap(), t[0]);
        // 1 unit and 2 units average to 1.5 units, which rounds away from zero.
        let got = run(a, &t, &[], &[decs(9, &[1, 2])]);
        assert!(got.eq_exact(&Value::Decimal(2, 9)), "{got:?}");
    }

    /// `avg(x)` and `sum(x)/count(*)` must not disagree about how many digits
    /// the mean of a decimal has -- the expression side already picked six.
    #[test]
    fn avg_widens_to_the_same_scale_the_divide_function_does() {
        let div = super::super::scalar("divide").unwrap();
        for s in [0u8, 2, 6, 9, 18] {
            let arg = DataType::Decimal64(s);
            let quotient = (div.ret)(&[arg.clone(), DataType::UInt64]).unwrap();
            let mean = (lookup("avg").unwrap().ret)(&[arg], &[]).unwrap();
            assert_eq!(mean, quotient.strip_nullable().clone(), "scale {s}");
        }
    }

    #[test]
    fn avg_of_a_decimal_survives_the_partial_merge() {
        let a = lookup("avg").unwrap();
        let (whole, m) = merged(a, &[DEC2], &[], &[decs(2, &[100, 200, 300, 401])]);
        assert!(whole.eq_exact(&Value::Decimal(2_502_500, 6)), "{whole:?}");
        assert!(whole.eq_exact(&m), "{m:?}");
        let s = lookup("sum").unwrap();
        let (whole, m) = merged(s, &[DEC2], &[], &[decs(2, &[100, 200, 300, 401])]);
        assert!(whole.eq_exact(&Value::Decimal(1001, 2)), "{whole:?}");
        assert!(whole.eq_exact(&m), "{m:?}");
    }

    /// The headline bug: `avg` widens to `max(s,6)` and then used to *clamp* at
    /// 18 digits, so the representable magnitude collapsed to 10^12 whatever
    /// the column's declared scale. One row of 1000000000000.00 averaged to
    /// 999999999999.999999 while `max` of the same row answered
    /// 1000000000000.00, and the fabricated answer was internally consistent.
    #[test]
    fn avg_of_a_decimal_refuses_the_promoted_scale_it_cannot_represent() {
        let a = lookup("avg").unwrap();
        // 10^12 at scale 2 is 10^14 lanes, which at scale 6 needs 10^18 -- one
        // unit past the 18 nines a Decimal64 holds.
        let e = try_run(a, &[DEC2], &[], &[decs(2, &[100_000_000_000_000])]).unwrap_err();
        assert!(e.to_string().contains("Decimal64(6)"), "{e}");
        // The largest mean that does fit still comes back, exactly, and is one
        // lane below the refusal above.
        let got = run(a, &[DEC2], &[], &[decs(2, &[99_999_999_999_999])]);
        assert!(got.eq_exact(&Value::Decimal(999_999_999_999_990_000, 6)), "{got:?}");
        // A scale that does not widen has no promoted range to fall out of, so
        // the identical magnitude is answered rather than refused.
        let got = run(a, &[DataType::Decimal64(6)], &[], &[decs(6, &[100_000_000_000_000])]);
        assert!(got.eq_exact(&Value::Decimal(100_000_000_000_000, 6)), "{got:?}");
    }

    /// `avg(x)` and `sum(x)/count(*)` are the same query, so they must not
    /// disagree about *whether* the mean exists either -- the expression side
    /// has always raised here, and `avg` used to fabricate.
    #[test]
    fn avg_and_the_equivalent_divide_fail_on_the_same_rows() {
        let lanes = [200_000_000_000_000i64, 400_000_000_000_000];
        let mean = try_run(lookup("avg").unwrap(), &[DEC2], &[], &[decs(2, &lanes)]);
        // What `sum(p)/count(*)` does: rescale the exact total to the quotient
        // scale that `divide` picked, which is the same `div_out_scale`.
        let total: i128 = lanes.iter().map(|&l| l as i128).sum();
        let quotient = crate::types::value::decimal_rescale(total, 2, div_out_scale(2))
            .filter(|u| u.unsigned_abs() <= DECIMAL_MAX_UNITS as u128);
        assert!(mean.is_err() && quotient.is_none(), "{mean:?} vs {quotient:?}");
    }

    /// Interpolating divides, so it widens like `avg`; `quantileExact` picks an
    /// element it actually saw, so it keeps the argument's own type.
    #[test]
    fn quantiles_of_a_decimal_split_by_whether_they_interpolate() {
        for n in ["median", "quantile"] {
            let q = lookup(n).unwrap();
            assert_eq!((q.ret)(&[DEC2], &[]).unwrap(), DataType::Decimal64(6), "{n}");
            let got = run(q, &[DEC2], &[], &[decs(2, &[381, 119])]);
            assert!(got.eq_exact(&Value::Decimal(2_500_000, 6)), "{n}: {got:?}");
        }
        let qe = lookup("quantileexact").unwrap();
        assert_eq!((qe.ret)(&[DEC2], &[]).unwrap(), DEC2);
        let got = run(qe, &[DEC2], &[], &[decs(2, &[381, 119])]);
        assert!(got.eq_exact(&Value::Decimal(381, 2)), "{got:?}");
    }

    #[test]
    fn quantile_interpolation_rounds_half_away_from_zero() {
        let q = lookup("quantile").unwrap();
        let t = [DataType::Decimal64(9)];
        // Midway between 1 and 2 units at a scale that cannot widen further.
        let got = run(q, &t, &[], &[decs(9, &[1, 2])]);
        assert!(got.eq_exact(&Value::Decimal(2, 9)), "{got:?}");
        // ... and the same midpoint below zero goes to -2, not -1: the rounding
        // follows the value, not the (always positive) interpolation step.
        let got = run(q, &t, &[], &[decs(9, &[-2, -1])]);
        assert!(got.eq_exact(&Value::Decimal(-2, 9)), "{got:?}");
        // A quarter of the way is exact in binary, so the weight is too.
        let got = run(q, &t, &[Value::Float(0.25)], &[decs(9, &[0, 400])]);
        assert!(got.eq_exact(&Value::Decimal(100, 9)), "{got:?}");
    }

    /// The lanes used to live in an `f64`, so anything past 2^53 came back
    /// rounded: `quantileExact` over one row of 1234567890123456.78 answered
    /// 1234567890123456.80 while `min`/`max` over the same row answered .78.
    /// An element the column actually held must come back bit for bit.
    #[test]
    fn quantile_exact_returns_the_lane_it_saw_past_the_float_mantissa() {
        let qe = lookup("quantileexact").unwrap();
        let big = 123_456_789_012_345_678i64; // 2^53 is ~9.0e15; this is 1.2e17
        assert_ne!(big as f64 as i64, big, "the f64 round trip must actually lose it");
        for lane in [big, -big, DECIMAL_MAX_UNITS as i64] {
            let got = run(qe, &[DEC2], &[], &[decs(2, &[lane])]);
            assert!(got.eq_exact(&Value::Decimal(lane, 2)), "{lane}: {got:?}");
        }
        // Order still has to be value order across the sign, which an unsigned
        // sort of the reinterpreted lanes would get exactly backwards.
        let col = decs(2, &[big, -big, 0]);
        assert!(run(qe, &[DEC2], &[], &[col.clone()]).eq_exact(&Value::Decimal(0, 2)));
        let q0 = lookup("quantileexact").unwrap();
        let got = run(q0, &[DEC2], &[Value::Float(0.0)], &[col]);
        assert!(got.eq_exact(&Value::Decimal(-big, 2)), "{got:?}");
    }

    /// Interpolating widens to `max(s,6)`, so it inherits `avg`'s refusal
    /// rather than `avg`'s old clamp.
    #[test]
    fn interpolating_quantiles_refuse_the_promoted_scale_they_cannot_represent() {
        for n in ["median", "quantile"] {
            let q = lookup(n).unwrap();
            let e = try_run(q, &[DEC2], &[], &[decs(2, &[100_000_000_000_000])]).unwrap_err();
            assert!(e.to_string().contains("Decimal64(6)"), "{n}: {e}");
        }
    }

    /// Welford runs on the raw lanes and the scale comes off the *result*:
    /// scaling every observation by 10^-s scales a variance by 10^-2s and a
    /// standard deviation by 10^-s.
    #[test]
    fn variance_of_a_decimal_descales_by_the_right_power() {
        // mean 2.50, deviations ±1.31
        let got = f(&run(lookup("varpop").unwrap(), &[DEC2], &[], &[decs(2, &[381, 119])]));
        assert!((got - 1.7161).abs() < 1e-12, "{got}");
        let got = f(&run(lookup("stddevpop").unwrap(), &[DEC2], &[], &[decs(2, &[381, 119])]));
        assert!((got - 1.31).abs() < 1e-12, "{got}");
        // The lanes read as plain integers are 10^4 and 10^2 times bigger --
        // that factor is exactly what shipped.
        let lanes = ints(&[381, 119]);
        let raw = f(&run(lookup("varpop").unwrap(), &[NI64], &[], &[lanes]));
        assert!((raw / 10_000.0 - got * got).abs() < 1e-9, "{raw}");
    }

    #[test]
    fn group_array_of_a_decimal_renders_the_point() {
        let g = lookup("grouparray").unwrap();
        let got = run(g, &[DEC2], &[], &[decs(2, &[381, 119, -5, 0])]);
        assert_eq!(got, Value::str("3.81,1.19,-0.05,0.00"));
    }

    /// `min`/`max`/`any`/`argMin`/`argMax` need no scale of their own: the lane
    /// is the answer, lane order is value order because one column has one
    /// scale, and `ret_same` gives the output column the scale that reads it
    /// back. Pinned so a change to `Column::value` cannot quietly break it.
    #[test]
    fn min_max_and_friends_pass_a_decimal_lane_straight_through() {
        for (n, want) in [("min", 119i64), ("max", 381), ("any", 381), ("anylast", 119)] {
            let f = lookup(n).unwrap();
            assert_eq!((f.ret)(&[DEC2], &[]).unwrap(), DEC2, "{n}");
            assert_eq!(run(f, &[DEC2], &[], &[decs(2, &[381, 119])]).as_i64(), Some(want), "{n}");
        }
        let am = lookup("argmax").unwrap();
        let got = run(am, &[DEC2, NI64], &[], &[decs(2, &[381, 119]), ints(&[1, 2])]);
        assert_eq!(got.as_i64(), Some(119), "argMax picks by key, returns the lane");
    }

    /// Counting is scale-blind: one column has one scale, so distinct lanes
    /// are distinct values.
    #[test]
    fn uniq_and_count_of_a_decimal_need_no_scale() {
        let col = || decs(2, &[381, 381, 119]);
        assert_eq!(run(lookup("uniqexact").unwrap(), &[DEC2], &[], &[col()]), Value::UInt(2));
        assert_eq!(run(lookup("uniq").unwrap(), &[DEC2], &[], &[col()]), Value::UInt(2));
        assert_eq!(run(lookup("count").unwrap(), &[DEC2], &[], &[col()]), Value::UInt(3));
    }

    /// The `-If` combinator forwards the argument types, so the decimal return
    /// type has to survive the wrapper.
    #[test]
    fn if_variants_carry_the_decimal_return_type() {
        let cond = DataType::Bool;
        let n = |t: DataType| DataType::Nullable(Box::new(t));
        let s = lookup("sumif").unwrap();
        assert_eq!((s.ret)(&[DEC2, cond.clone()], &[]).unwrap(), n(DataType::Decimal64(2)));
        let a = lookup("avgif").unwrap();
        assert_eq!((a.ret)(&[DEC2, cond.clone()], &[]).unwrap(), DataType::Decimal64(6));
        let got = run(s, &[DEC2, cond], &[], &[decs(2, &[381, 119]), bools(&[true, false])]);
        assert!(got.eq_exact(&Value::Decimal(381, 2)), "{got:?}");
    }
}


