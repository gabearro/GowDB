//! Vectorized expression evaluation: one [`BoundExpr`] over a whole [`Block`].
//!
//! The unit of work is a column, never a row. `a + b * 2` costs three passes
//! over contiguous `&[i64]` slices, not `rows` trips through an interpreter,
//! and that difference is the entire reason a `Block` exists. Nothing in here
//! calls [`Column::value`] in a loop except where the operation is genuinely
//! per-row and untyped (`CASE` branch selection, `IN`-list membership, casts
//! between unrelated families) -- and even those materialize into a
//! [`ColumnBuilder`] rather than a `Vec<Value>`.
//!
//! ## Why so much of this delegates to the scalar registry
//!
//! `plus`, `minus`, `multiply`, `divide`, `intDiv`, `modulo`, `concat`,
//! `negate`, `not`, `and`, `or`, `isNull` and the `LIKE` family all already
//! exist in the [scalar registry](mod@crate::exec::functions::scalar), with
//! their type promotion and NULL rules pinned down by that module's own
//! tests. `a + b` written as
//! `BoundExpr::Binary` and `plus(a, b)` written as `BoundExpr::Scalar` must
//! produce bit-identical results, and the only way to *guarantee* that is to
//! have one implementation rather than two that agree today. So the operators
//! here are sugar over the registry: this module resolves the entry once per
//! row-batch and hands over the columns.
//!
//! What is implemented natively is exactly what the registry has no entry
//! for: comparisons, `CAST`, `CASE`, and `IN`.
//!
//! ## Decimal scale
//!
//! A `Decimal64(s)` lane is a count of `10^-s` units, not a number, so the two
//! sides of a comparison are only commensurable when their scales agree. They
//! usually do, and then the lanes compare as plain `Int64` with nothing added
//! to the inner loop; [`eval_cmp`] splits the disagreeing case out with one
//! `match` per block. Arithmetic needs the same rescale and already has it --
//! `dec_arith` in the scalar registry -- which is one more reason the operators
//! here are sugar over that module rather than a second implementation.
//!
//! ## NULL semantics
//!
//! Strict everywhere: a NULL operand yields a NULL result for arithmetic,
//! comparison, concatenation and casts. `AND`/`OR` are the documented
//! exceptions -- `false AND NULL` is `false` and `true OR NULL` is `true`,
//! because the unknown cannot change the answer. Division by zero produces
//! NULL rather than an error, matching
//! [`crate::planner::optimizer::const_eval`]: a predicate must not mean
//! something different depending on whether the optimizer happened to fold it.

use std::cmp::Ordering;
use std::sync::Arc;

use std::borrow::Cow;

use crate::common::{BitSet, Error, Result};
use crate::exec::functions;
use crate::planner::logical::BoundExpr;
use crate::sql::ast::{BinaryOp, UnaryOp};
use crate::types::value::POW10;
use crate::types::{Block, Column, ColumnBuilder, ColumnData, DataType, PhysicalType, Value};

// ---------------------------------------------------------------- public API

/// How deep a [`BoundExpr`] may nest before evaluation refuses it.
///
/// Nothing the binder produces can reach this: `bind` recurses at least once
/// per level of the tree it builds and stops at its own limit of 200, so a
/// bound expression is never deeper than the binder's own guard allowed. This
/// is the backstop for the trees nobody parsed -- ones assembled directly
/// against this API -- and for the day some rewrite starts *growing* an
/// expression instead of only shrinking it.
///
/// The counter is a `usize` passed by value, which the recursion carries in a
/// register: no cell, no `Drop`, nothing for `?` to leak. The cost is one
/// perfectly-predicted compare-and-branch per *node* -- not per row -- in
/// front of a `match` that already dispatches on the node's tag, so it is
/// invisible next to the per-node work of scanning a whole column.
const MAX_EXPR_DEPTH: usize = 200;

/// Evaluate `e` against every row of `block`, yielding a column of exactly
/// `block.rows()` rows.
pub fn eval(e: &BoundExpr, block: &Block) -> Result<Column> {
    eval_at(e, block, 0)
}

/// `#[cold]` so the formatting and the allocation stay out of `eval_at`'s own
/// frame: the branch that reaches this is never taken in a real query.
#[cold]
fn too_deep() -> Error {
    Error::exec(format!(
        "expression nests more than {MAX_EXPR_DEPTH} levels deep; the evaluator recurses \
         once per level and would run out of stack"
    ))
}

fn eval_at(e: &BoundExpr, block: &Block, depth: usize) -> Result<Column> {
    if depth > MAX_EXPR_DEPTH {
        return Err(too_deep());
    }
    let depth = depth + 1;
    let rows = block.rows();
    match e {
        BoundExpr::Literal { value, ty } => Column::constant(ty, value, rows),

        // A `Vec` clone of an already-decoded column. Cheap relative to the
        // decode that produced it, and it leaves every downstream operator
        // free to consume its inputs without a copy-on-write dance.
        BoundExpr::Column { index, name, .. } => {
            block.columns.get(*index).cloned().ok_or_else(|| {
                Error::exec(format!(
                    "column `{name}` is #{index} but this block is only {} wide",
                    block.width()
                ))
            })
        }

        BoundExpr::Unary { op, expr, .. } => {
            let c = eval_at(expr, block, depth)?;
            call(
                match op {
                    UnaryOp::Neg => "negate",
                    UnaryOp::Not => "not",
                },
                vec![c],
                rows,
            )
        }

        BoundExpr::Binary { left, op, right, .. } => eval_binary(*op, left, right, block, depth),

        BoundExpr::Scalar { func, args, .. } => {
            let cols = eval_all_at(args, block, depth)?;
            func.check_arity(cols.len())?;
            (func.eval)(&cols, rows)
        }

        BoundExpr::Cast { expr, ty } => {
            let c = eval_at(expr, block, depth)?;
            eval_cast(&c, ty, rows)
        }

        BoundExpr::Case { when_then, else_result, ty } => {
            eval_case(when_then, else_result.as_deref(), ty, block, depth)
        }

        BoundExpr::InList { expr, list, negated } => {
            let c = eval_at(expr, block, depth)?;
            eval_in_list(&c, list, *negated, rows)
        }

        BoundExpr::Like { expr, pattern, negated, case_insensitive } => {
            let c = eval_at(expr, block, depth)?;
            let subject = if c.ty.is_string() {
                c
            } else {
                // `x LIKE '1%'` against an integer column: render, then match.
                to_string_column(c, rows)
            };
            let pat: Arc<str> = Arc::from(pattern.as_str());
            let name = match (*case_insensitive, *negated) {
                (false, false) => "like",
                (false, true) => "notLike",
                (true, false) => "ilike",
                (true, true) => "notILike",
            };
            call(
                name,
                vec![subject, Column::strs(DataType::String, vec![pat; rows])],
                rows,
            )
        }

        BoundExpr::IsNull { expr, negated } => {
            let c = eval_at(expr, block, depth)?;
            call(if *negated { "isNotNull" } else { "isNull" }, vec![c], rows)
        }
    }
}

/// Evaluate a list of expressions against one block.
pub fn eval_all(exprs: &[BoundExpr], block: &Block) -> Result<Vec<Column>> {
    eval_all_at(exprs, block, 0)
}

fn eval_all_at(exprs: &[BoundExpr], block: &Block, depth: usize) -> Result<Vec<Column>> {
    exprs.iter().map(|e| eval_at(e, block, depth)).collect()
}

/// Evaluate, borrowing the input column when the expression is a bare column
/// reference.
///
/// `eval` has to return an owned `Column`, so evaluating `GROUP BY country`
/// *clones* the block's column -- and cloning a `Vec<Arc<str>>` is one atomic
/// refcount increment per row, which for a plain projection is pure waste.
/// A bare column reference is by far the most common expression in a plan, so
/// borrowing it is worth the `Cow`.
pub fn eval_all_cow<'b>(
    exprs: &[BoundExpr],
    block: &'b Block,
) -> Result<Vec<Cow<'b, Column>>> {
    exprs
        .iter()
        .map(|e| match e {
            BoundExpr::Column { index, name, .. } => block
                .columns
                .get(*index)
                .map(Cow::Borrowed)
                .ok_or_else(|| {
                    Error::exec(format!(
                        "column `{name}` at index {index} is out of range for a {}-column block",
                        block.width()
                    ))
                }),
            other => eval(other, block).map(Cow::Owned),
        })
        .collect()
}

/// Row indices where `e` is **TRUE**.
///
/// SQL three-valued logic: NULL and FALSE both fail a filter, so the null mask
/// is consulted before truthiness. The result is a selection vector, which is
/// what every operator downstream wants -- `Block::take` is the one reshaping
/// primitive in the engine.
pub fn eval_predicate(e: &BoundExpr, block: &Block) -> Result<Vec<u32>> {
    let rows = block.rows();
    let c = eval(e, block)?;
    let n = rows.min(c.len());
    let mut out = Vec::with_capacity(n);
    match (&c.data, &c.nulls) {
        (ColumnData::U64(v), None) => push_where(&mut out, n, |i| v[i] != 0),
        (ColumnData::U64(v), Some(m)) => push_where(&mut out, n, |i| !m.get(i) && v[i] != 0),
        (ColumnData::I64(v), None) => push_where(&mut out, n, |i| v[i] != 0),
        (ColumnData::I64(v), Some(m)) => push_where(&mut out, n, |i| !m.get(i) && v[i] != 0),
        (ColumnData::F64(v), None) => push_where(&mut out, n, |i| v[i] != 0.0),
        (ColumnData::F64(v), Some(m)) => push_where(&mut out, n, |i| !m.get(i) && v[i] != 0.0),
        (ColumnData::Str(v), None) => push_where(&mut out, n, |i| !v[i].is_empty()),
        (ColumnData::Str(v), Some(m)) => {
            push_where(&mut out, n, |i| !m.get(i) && !v[i].is_empty())
        }
    }
    Ok(out)
}

#[inline]
fn push_where(out: &mut Vec<u32>, n: usize, mut keep_row: impl FnMut(usize) -> bool) {
    for i in 0..n {
        if keep_row(i) {
            out.push(i as u32);
        }
    }
}

// ------------------------------------------------------------------ operators

/// Resolve and invoke a registry entry. The lookup happens once per batch,
/// never once per row.
fn call(name: &str, args: Vec<Column>, rows: usize) -> Result<Column> {
    let f = functions::scalar(name).ok_or_else(|| {
        Error::exec(format!("scalar function `{name}` is missing from the registry"))
    })?;
    (f.eval)(&args, rows)
}

fn eval_binary(
    op: BinaryOp,
    left: &BoundExpr,
    right: &BoundExpr,
    block: &Block,
    depth: usize,
) -> Result<Column> {
    let rows = block.rows();
    let l = eval_at(left, block, depth)?;
    let r = eval_at(right, block, depth)?;
    if op.is_comparison() {
        return eval_cmp(op, &l, &r, rows);
    }
    let name = match op {
        BinaryOp::Plus => "plus",
        BinaryOp::Minus => "minus",
        BinaryOp::Multiply => "multiply",
        BinaryOp::Divide => "divide",
        BinaryOp::IntDiv => "intDiv",
        BinaryOp::Modulo => "modulo",
        BinaryOp::Concat => "concat",
        BinaryOp::And => "and",
        BinaryOp::Or => "or",
        _ => unreachable!("comparisons are handled above"),
    };
    if op == BinaryOp::Concat {
        // `concat` insists on String arguments; SQL's `||` does not.
        let a = to_string_column(l, rows);
        let b = to_string_column(r, rows);
        return call(name, vec![a, b], rows);
    }
    call(name, vec![l, r], rows)
}

fn to_string_column(c: Column, rows: usize) -> Column {
    if c.ty.is_string() {
        return c;
    }
    let data = ColumnData::Str(strs_of(&c, rows));
    match &c.nulls {
        Some(n) => Column { ty: DataType::String.to_nullable(), data, nulls: Some(n.clone()) },
        None => Column::new(DataType::String, data),
    }
}

// ---------------------------------------------------------------- comparison

/// Comparison lowers to an `Ordering` per row, then one branch-free map to
/// 0/1.
///
/// Four same-kind fast paths cover essentially all real traffic. Mixed integer
/// signedness widens to `i128` (exact, unlike routing through `f64`), a float
/// against an integer widens to `f64` exactly as [`Value`] does, and a string
/// against a number falls back to `Value`'s ordering so that comparing across
/// families agrees exactly with what the constant folder would have produced.
///
/// All ten reachable physical pairs are named, so there is no `_` arm that
/// materializes two `Vec<f64>` per block just to walk them once: the mixed
/// float/integer arms zip the two lanes in place. 0.46x on `Float64 > Int64`
/// literal over 2.1M rows, A/B interleaved best-of-9 with the operator dispatch
/// held constant (534 vs 244 M rows/s); 0.19x with [`keep_mask`]'s hoist
/// stacked on top, which is where the other 2.4x comes from.
fn eval_cmp(op: BinaryOp, l: &Column, r: &Column, rows: usize) -> Result<Column> {
    // Equal scales (including the `None == None` of two non-decimals) fall
    // straight through to the lane comparison below: a Decimal64(2) against
    // another Decimal64(2) is byte-identical to Int64 and must stay that way.
    // One `match` per block buys that: measured on the same lanes, the decimal
    // and the Int64 column both run at 1823 M rows/s.
    if l.ty.decimal_scale() != r.ty.decimal_scale() {
        return eval_cmp_decimal(op, l, r, rows);
    }
    let mut bits = Vec::with_capacity(rows);
    macro_rules! run {
        ($a:expr, $b:expr, $cmp:expr) => {{
            let f = $cmp;
            fill(&mut bits, rows, op, $a.iter().zip($b).map(move |(x, y)| f(x, y)))
        }};
    }
    match (&l.data, &r.data) {
        (ColumnData::U64(a), ColumnData::U64(b)) => run!(a, b, |x: &u64, y: &u64| x.cmp(y)),
        (ColumnData::I64(a), ColumnData::I64(b)) => run!(a, b, |x: &i64, y: &i64| x.cmp(y)),
        (ColumnData::F64(a), ColumnData::F64(b)) => {
            run!(a, b, |x: &f64, y: &f64| total_cmp_f64(*x, *y))
        }
        (ColumnData::Str(a), ColumnData::Str(b)) => {
            run!(a, b, |x: &Arc<str>, y: &Arc<str>| x.as_ref().cmp(y.as_ref()))
        }
        (ColumnData::U64(a), ColumnData::I64(b)) => {
            run!(a, b, |x: &u64, y: &i64| (*x as i128).cmp(&(*y as i128)))
        }
        (ColumnData::I64(a), ColumnData::U64(b)) => {
            run!(a, b, |x: &i64, y: &u64| (*x as i128).cmp(&(*y as i128)))
        }
        // One float, one integer: widen to f64, exactly as `Value` does.
        (ColumnData::U64(a), ColumnData::F64(b)) => {
            run!(a, b, |x: &u64, y: &f64| total_cmp_f64(*x as f64, *y))
        }
        (ColumnData::F64(a), ColumnData::U64(b)) => {
            run!(a, b, |x: &f64, y: &u64| total_cmp_f64(*x, *y as f64))
        }
        (ColumnData::I64(a), ColumnData::F64(b)) => {
            run!(a, b, |x: &i64, y: &f64| total_cmp_f64(*x as f64, *y))
        }
        (ColumnData::F64(a), ColumnData::I64(b)) => {
            run!(a, b, |x: &f64, y: &i64| total_cmp_f64(*x, *y as f64))
        }
        (ColumnData::Str(_), _) | (_, ColumnData::Str(_)) => {
            // A string against a number. `Value`'s ordering ranks the
            // families, which is deterministic rather than an error, and is
            // what `const_eval` would have folded to.
            fill(&mut bits, rows, op, values(l, r));
        }
    }
    Ok(bool_column(bits, union_nulls(l, r)))
}

/// The rescaling comparison: the two sides disagree about scale, so their lanes
/// are not commensurable and the equal-scale path above would compare a unit
/// count against a number. `WHERE price > 100` on a `Decimal64(2)` -- 100
/// against a lane a hundred times larger -- is the query that made this exist.
///
/// A non-decimal integer is a decimal of scale 0: every integer is exactly
/// representable at every scale, so "decimal vs Int64" collapses into "decimal
/// vs decimal" and there is one exact loop rather than two. Both lanes widen to
/// `i128` *before* the rescale, which is what makes it exact -- an in-range lane
/// is under 10^18 and the largest factor is another 10^18, for 10^37 against
/// `i128`'s 1.7e38, so a lane near `i64::MAX` still cannot wrap. Narrowing back
/// to `i64` first is precisely how it would go wrong again.
///
/// A float on either side has no exact answer to give, so the decimal descales
/// (`f64_vec`'s rule, fused into the loop instead of allocating a `Vec<f64>`)
/// and the compare happens in `f64`. That is what [`Value::cmp`] does for the
/// same pair, and the two must agree or a predicate would mean something
/// different once the optimizer folded it.
///
/// The honest cost, 2.1M rows A/B interleaved best-of-9 against the same lanes
/// typed `Int64`: 2.8x for the `i128` rescale (655 vs 1823 M rows/s) and 1.5x
/// for the float descale (1198). Both are the slow path by construction -- the
/// scale check in [`eval_cmp`] keeps every same-scale comparison off them.
fn eval_cmp_decimal(op: BinaryOp, l: &Column, r: &Column, rows: usize) -> Result<Column> {
    let (sl, sr) = (scale_of(l), scale_of(r));
    // Hoisted: the scales are a property of the *types*, so the factors are
    // constant for the whole block.
    let hi = sl.max(sr);
    let (fa, fb) = (POW10[(hi - sl) as usize], POW10[(hi - sr) as usize]);
    let mut bits = Vec::with_capacity(rows);
    macro_rules! run {
        ($a:expr, $b:expr, $cmp:expr) => {{
            let f = $cmp;
            fill(&mut bits, rows, op, $a.iter().zip($b).map(move |(x, y)| f(x, y)))
        }};
    }
    // `hi` is one of the two scales, so one factor is always 1 and that side
    // needs no multiply at all. Branching on which, once per block, is 0.72x
    // over the mixed-scale arms (655 vs 462 M rows/s): the factors are runtime
    // values, so LLVM cannot fold the `* 1` away by itself.
    macro_rules! rescale {
        ($a:expr, $b:expr, $x:ty, $y:ty) => {{
            if fb == 1 {
                run!($a, $b, |x: &$x, y: &$y| (*x as i128 * fa).cmp(&(*y as i128)))
            } else {
                run!($a, $b, |x: &$x, y: &$y| (*x as i128).cmp(&(*y as i128 * fb)))
            }
        }};
    }
    match (&l.data, &r.data) {
        (ColumnData::I64(a), ColumnData::I64(b)) => rescale!(a, b, i64, i64),
        (ColumnData::I64(a), ColumnData::U64(b)) => rescale!(a, b, i64, u64),
        (ColumnData::U64(a), ColumnData::I64(b)) => rescale!(a, b, u64, i64),
        // One FDIV per row with the divisor hoisted, which is what a decimal
        // costs against a float literal -- `2.0` lexes as `Float64`, so this is
        // the arm `WHERE price > 2.0` lands on.
        (ColumnData::I64(a), ColumnData::F64(b)) => {
            let d = POW10[sl as usize] as f64;
            run!(a, b, move |x: &i64, y: &f64| total_cmp_f64(*x as f64 / d, *y))
        }
        (ColumnData::F64(a), ColumnData::I64(b)) => {
            let d = POW10[sr as usize] as f64;
            run!(a, b, move |x: &f64, y: &i64| total_cmp_f64(*x, *y as f64 / d))
        }
        // A string against a decimal, which the binder refuses (`cannot compare
        // Decimal64(2) with String`) and only a hand-built plan can reach; and
        // the physically impossible pairs, since a decimal is always an I64
        // lane. `Value`'s family ranking, same as the equal-scale path.
        _ => fill(&mut bits, rows, op, values(l, r)),
    }
    Ok(bool_column(bits, union_nulls(l, r)))
}

/// A column's decimal scale, with a non-decimal reading as 0 -- see
/// [`eval_cmp_decimal`]. Deliberately *not* applied to floats there: a float is
/// not a scale-0 integer and takes the descaling path instead.
#[inline]
fn scale_of(c: &Column) -> u8 {
    c.ty.decimal_scale().unwrap_or(0)
}

/// The untyped comparison, one materialized [`Value`] per side per row. Slow,
/// and reached only by pairs the binder rejects.
fn values<'c>(l: &'c Column, r: &'c Column) -> impl Iterator<Item = Ordering> + 'c {
    (0..l.len().min(r.len())).map(|i| l.value(i).cmp(&r.value(i)))
}

/// Drain up to `rows` orderings into 0/1 lanes.
///
/// `extend` rather than `vec![0; rows]` plus indexed stores: the zeroing pass
/// and the per-row bounds checks both disappear, and a source that ran short
/// pads instead of panicking (`resize` is a no-op in the normal case). Generic
/// over the iterator so the equal-scale and rescaling paths share one
/// definition; monomorphization leaves each caller the flat loop it would have
/// written by hand.
#[inline]
fn fill(bits: &mut Vec<u64>, rows: usize, op: BinaryOp, ord: impl Iterator<Item = Ordering>) {
    let m = keep_mask(op);
    bits.extend(ord.take(rows).map(|o| keep(m, o) as u64));
    bits.resize(rows, 0);
}

/// Which orderings `op` keeps, one bit per outcome: 0 Less, 1 Equal, 2 Greater.
///
/// Resolved once per block, and the largest single win in this file. `op` is
/// loop-invariant, but LLVM will not unswitch a six-way `match` out of an
/// iterator chain by itself, so the old shape paid the dispatch per *row*.
/// 0.22x over 2.1M rows, A/B interleaved best-of-9 (`Int64 > literal`, 1823 vs
/// 399 M rows/s), and 0.41x on the arms whose compare is heavier.
///
/// Null result, so nobody repeats it: pre-slicing both inputs to `rows` and
/// dropping the `.take` in [`fill`] -- on the theory that `Take` is not
/// `TrustedLen` and knocks `extend` off its specialization -- measured 1.000x.
/// The `.take` is free and it is the smaller code, so it stays.
#[inline]
fn keep_mask(op: BinaryOp) -> u8 {
    match op {
        BinaryOp::Eq => 0b010,
        BinaryOp::NotEq => 0b101,
        BinaryOp::Lt => 0b001,
        BinaryOp::LtEq => 0b011,
        BinaryOp::Gt => 0b100,
        BinaryOp::GtEq => 0b110,
        _ => 0,
    }
}

/// [`Ordering`] is `repr(i8)` with `Less == -1`, so `+1` indexes the mask
/// directly and the per-row step is a shift and a test, with no branch at all.
#[inline(always)]
fn keep(mask: u8, o: Ordering) -> bool {
    mask >> (o as i8 + 1) & 1 != 0
}

/// Total order over `f64`, identical to the one [`Value`] uses: `-0.0 == 0.0`
/// and NaN sorts last. Keeping these in step is what stops `ORDER BY x` and
/// `WHERE x > y` from disagreeing about the same pair of values.
#[inline(always)]
fn total_cmp_f64(a: f64, b: f64) -> Ordering {
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

fn union_nulls(a: &Column, b: &Column) -> Option<BitSet> {
    match (&a.nulls, &b.nulls) {
        (None, None) => None,
        (Some(x), None) => Some(x.clone()),
        (None, Some(y)) => Some(y.clone()),
        (Some(x), Some(y)) => {
            let mut m = x.clone();
            m.union_with(y);
            Some(m)
        }
    }
}

fn bool_column(bits: Vec<u64>, nulls: Option<BitSet>) -> Column {
    match nulls {
        Some(n) if !n.is_empty() => Column {
            ty: DataType::Bool.to_nullable(),
            data: ColumnData::U64(bits),
            nulls: Some(n),
        },
        _ => Column::bools(bits),
    }
}

// --------------------------------------------------------------------- CAST

fn eval_cast(c: &Column, ty: &DataType, rows: usize) -> Result<Column> {
    let out_ty = if c.nulls.is_some() { ty.to_nullable() } else { ty.clone() };
    // Same logical base -> a relabel, not a conversion. `Nullable(Int64)` to
    // `Int64` and `String` to `LowCardinality(String)` both land here, and
    // both are free.
    if c.ty.base() == ty.base() {
        return Ok(Column { ty: out_ty, data: c.data.clone(), nulls: c.nulls.clone() });
    }
    let mut b = ColumnBuilder::with_capacity(out_ty, rows);
    for i in 0..rows {
        if c.is_null(i) {
            b.push_null();
        } else {
            b.push_value(&c.value(i).cast_to(ty)?)?;
        }
    }
    Ok(b.finish())
}

// --------------------------------------------------------------------- CASE

/// `CASE WHEN a THEN x WHEN b THEN y ELSE z END`.
///
/// Two passes: the first walks the branches building a `pick` vector (which
/// arm owns each row), the second gathers from the already-evaluated result
/// columns in one typed sweep. So the inner loop stays free of `Value`
/// materialization even though the branch structure is inherently per row.
fn eval_case(
    when_then: &[(BoundExpr, BoundExpr)],
    else_result: Option<&BoundExpr>,
    ty: &DataType,
    block: &Block,
    depth: usize,
) -> Result<Column> {
    let rows = block.rows();
    let mut pick = vec![-1i32; rows];
    let mut arms: Vec<Column> = Vec::with_capacity(when_then.len() + 1);
    for (k, (w, t)) in when_then.iter().enumerate() {
        let cond = eval_at(w, block, depth)?;
        let hit = truthy_mask(&cond, rows);
        for (p, &h) in pick.iter_mut().zip(&hit) {
            if *p < 0 && h {
                *p = k as i32;
            }
        }
        arms.push(eval_at(t, block, depth)?);
    }
    if let Some(e) = else_result {
        let k = arms.len() as i32;
        arms.push(eval_at(e, block, depth)?);
        for p in pick.iter_mut() {
            if *p < 0 {
                *p = k;
            }
        }
    }
    gather_pick(ty, &arms, &pick, rows)
}

/// Per-row SQL truthiness. A NULL condition is not true, which is what makes
/// `CASE WHEN NULL THEN ...` fall through to the next arm.
fn truthy_mask(c: &Column, rows: usize) -> Vec<bool> {
    let n = rows.min(c.len());
    let mut out = vec![false; rows];
    for (i, slot) in out.iter_mut().enumerate().take(n) {
        if c.is_null(i) {
            continue;
        }
        *slot = match &c.data {
            ColumnData::U64(v) => v[i] != 0,
            ColumnData::I64(v) => v[i] != 0,
            ColumnData::F64(v) => v[i] != 0.0,
            ColumnData::Str(v) => !v[i].is_empty(),
        };
    }
    out
}

/// Row-wise gather from several equal-length sources. `pick[i] < 0` means
/// NULL; otherwise the row comes from `sources[pick[i]]`, inheriting that
/// source's own null bit.
fn gather_pick(ty: &DataType, sources: &[Column], pick: &[i32], rows: usize) -> Result<Column> {
    let p = ty.base().physical();
    let mut nulls = BitSet::new();
    macro_rules! run {
        ($conv:ident, $zero:expr) => {{
            let src: Vec<_> = sources.iter().map($conv).collect::<Result<Vec<_>>>()?;
            let mut out = Vec::with_capacity(rows);
            for i in 0..rows {
                let k = pick[i];
                if k < 0 || sources[k as usize].is_null(i) {
                    nulls.set(i);
                    out.push($zero);
                } else {
                    out.push(src[k as usize][i].clone());
                }
            }
            out
        }};
    }
    let data = match p {
        PhysicalType::U64 => ColumnData::U64(run!(u64s_of, 0u64)),
        PhysicalType::I64 => ColumnData::I64(run!(i64s_of, 0i64)),
        PhysicalType::F64 => ColumnData::F64(run!(f64s_of, 0.0f64)),
        PhysicalType::Str => ColumnData::Str(run!(arcs_of, Arc::<str>::from(""))),
    };
    Ok(if nulls.is_empty() {
        Column::new(ty.strip_nullable(), data)
    } else {
        Column { ty: ty.to_nullable(), data, nulls: Some(nulls) }
    })
}

// ----------------------------------------------------------------------- IN

/// `x IN (...)` with the standard NULL rules: a NULL probe is NULL, and a miss
/// against a list that *contains* NULL is also NULL -- the value might have
/// been the unknown one. Only a hit, or a miss against an entirely known list,
/// is decidable.
fn eval_in_list(c: &Column, list: &[Value], negated: bool, rows: usize) -> Result<Column> {
    let has_null = list.iter().any(|v| v.is_null());
    // Long lists get sorted once and binary-searched, turning an
    // O(rows * |list|) scan into O(rows * log |list|). `Value: Ord` is total,
    // so this stays exact.
    let sorted: Option<Vec<Value>> = if list.len() > 16 {
        let mut v: Vec<Value> = list.iter().filter(|x| !x.is_null()).cloned().collect();
        v.sort();
        Some(v)
    } else {
        None
    };
    let mut bits = vec![0u64; rows];
    let mut nulls = BitSet::new();
    for (i, bit) in bits.iter_mut().enumerate() {
        if c.is_null(i) {
            nulls.set(i);
            continue;
        }
        let v = c.value(i);
        let found = match &sorted {
            Some(s) => s.binary_search(&v).is_ok(),
            None => list.iter().any(|x| !x.is_null() && *x == v),
        };
        if found {
            *bit = !negated as u64;
        } else if has_null {
            nulls.set(i);
        } else {
            *bit = negated as u64;
        }
    }
    Ok(bool_column(bits, Some(nulls)))
}

// ------------------------------------------------------------ representation

fn u64s_of(c: &Column) -> Result<Vec<u64>> {
    Ok(match &c.data {
        ColumnData::U64(v) => v.clone(),
        ColumnData::I64(v) => v.iter().map(|&x| x as u64).collect(),
        ColumnData::F64(v) => v.iter().map(|&x| x as u64).collect(),
        ColumnData::Str(_) => return Err(Error::exec("cannot use a String column as an integer")),
    })
}

fn i64s_of(c: &Column) -> Result<Vec<i64>> {
    c.to_i64_vec()
}

fn f64s_of(c: &Column) -> Result<Vec<f64>> {
    c.to_f64_vec()
}

fn arcs_of(c: &Column) -> Result<Vec<Arc<str>>> {
    Ok(strs_of(c, c.len()))
}

/// Render a column as strings, for free when it already is one.
fn strs_of(c: &Column, rows: usize) -> Vec<Arc<str>> {
    match &c.data {
        ColumnData::Str(v) => v.clone(),
        _ => (0..rows.min(c.len()))
            .map(|i| Arc::from(c.value(i).render_plain()))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::logical::BoundExpr as B;

    // ------------------------------------------------------------- fixtures

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
    /// A `Decimal64(s)` column from raw **lanes**: `decs(2, &[381])` is $3.81.
    fn decs(s: u8, v: &[i64]) -> Column {
        Column::i64s(DataType::Decimal64(s), v.to_vec())
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
    /// Nullable Bool column: the raw material for three-valued-logic tests.
    fn tri(v: &[Option<bool>]) -> Column {
        let mut b = ColumnBuilder::new(DataType::Nullable(Box::new(DataType::Bool)));
        for x in v {
            match x {
                Some(t) => b.push_value(&Value::Bool(*t)).unwrap(),
                None => b.push_null(),
            }
        }
        b.finish()
    }

    fn blk(cols: Vec<Column>) -> Block {
        Block::new(cols).unwrap()
    }

    fn col(i: usize, ty: DataType) -> B {
        B::Column { index: i, ty, name: format!("c{i}") }
    }
    fn lit(v: Value) -> B {
        B::lit(v)
    }
    fn bin(l: B, op: BinaryOp, r: B, ty: DataType) -> B {
        B::Binary { left: Box::new(l), op, right: Box::new(r), ty }
    }
    fn cmpx(l: B, op: BinaryOp, r: B) -> B {
        bin(l, op, r, DataType::Bool)
    }

    fn i_of(c: &Column) -> Vec<i64> {
        c.as_i64().unwrap().to_vec()
    }
    fn u_of(c: &Column) -> Vec<u64> {
        c.as_u64().unwrap().to_vec()
    }
    /// Render to `Option<Value>` so NULLs are visible in assertions.
    fn vals(c: &Column) -> Vec<Option<Value>> {
        (0..c.len())
            .map(|i| if c.is_null(i) { None } else { Some(c.value(i)) })
            .collect()
    }

    // --------------------------------------------------------------- basics

    #[test]
    fn literal_expands_to_a_constant_column() {
        let b = blk(vec![ints(&[1, 2, 3])]);
        let c = eval(&lit(Value::Int(7)), &b).unwrap();
        assert_eq!(i_of(&c), vec![7, 7, 7]);

        let n = eval(&lit(Value::Null), &b).unwrap();
        assert_eq!(n.len(), 3);
        assert!((0..3).all(|i| n.is_null(i)));
    }

    #[test]
    fn column_reference_returns_the_input_column() {
        let b = blk(vec![ints(&[1, 2]), strs(&["a", "b"])]);
        assert_eq!(i_of(&eval(&col(0, DataType::Int64), &b).unwrap()), vec![1, 2]);
        let s = eval(&col(1, DataType::String), &b).unwrap();
        assert_eq!(s.value(1), Value::str("b"));
    }

    #[test]
    fn out_of_range_column_is_an_error_not_a_panic() {
        let b = blk(vec![ints(&[1])]);
        assert!(eval(&col(9, DataType::Int64), &b).is_err());
    }

    #[test]
    fn zero_row_block_yields_zero_row_columns() {
        let b = blk(vec![ints(&[])]);
        let e = bin(col(0, DataType::Int64), BinaryOp::Plus, lit(Value::Int(1)), DataType::Int64);
        assert_eq!(eval(&e, &b).unwrap().len(), 0);
        let p = cmpx(col(0, DataType::Int64), BinaryOp::Gt, lit(Value::Int(0)));
        assert!(eval_predicate(&p, &b).unwrap().is_empty());
    }

    // ----------------------------------------------------------- arithmetic

    #[test]
    fn arithmetic_promotes_like_the_type_system_says() {
        let b = blk(vec![ints(&[10, -4]), floats(&[2.5, 4.0])]);
        let e = bin(
            col(0, DataType::Int64),
            BinaryOp::Plus,
            col(1, DataType::Float64),
            DataType::Float64,
        );
        let c = eval(&e, &b).unwrap();
        assert_eq!(c.as_f64().unwrap(), &[12.5, 0.0]);
        assert_eq!(c.ty, DataType::Float64);
    }

    #[test]
    fn integer_arithmetic_stays_integral() {
        let b = blk(vec![ints(&[7, -7])]);
        for (op, want) in [
            (BinaryOp::Plus, vec![10i64, -4]),
            (BinaryOp::Minus, vec![4, -10]),
            (BinaryOp::Multiply, vec![21, -21]),
        ] {
            let e = bin(col(0, DataType::Int64), op, lit(Value::Int(3)), DataType::Int64);
            assert_eq!(i_of(&eval(&e, &b).unwrap()), want, "{op:?}");
        }
    }

    #[test]
    fn divide_by_zero_is_null_not_an_error() {
        let b = blk(vec![ints(&[10, 20, 30]), ints(&[2, 0, 5])]);
        for op in [BinaryOp::Divide, BinaryOp::IntDiv, BinaryOp::Modulo] {
            let e = bin(col(0, DataType::Int64), op, col(1, DataType::Int64), DataType::Int64);
            let c = eval(&e, &b).unwrap();
            assert!(!c.is_null(0), "{op:?} row 0");
            assert!(c.is_null(1), "{op:?} must NULL out x/0");
            assert!(!c.is_null(2), "{op:?} row 2");
        }
    }

    #[test]
    fn divide_matches_const_eval_on_the_same_inputs() {
        use crate::planner::optimizer::const_eval;
        let b = blk(vec![ints(&[7])]);
        let e = bin(lit(Value::Int(7)), BinaryOp::Divide, lit(Value::Int(2)), DataType::Float64);
        assert_eq!(eval(&e, &b).unwrap().value(0), const_eval(&e).unwrap());

        let z = bin(lit(Value::Int(7)), BinaryOp::Divide, lit(Value::Int(0)), DataType::Float64);
        assert!(eval(&z, &b).unwrap().is_null(0));
        assert!(const_eval(&z).unwrap().is_null());
    }

    #[test]
    fn arithmetic_propagates_nulls() {
        let b = blk(vec![nullable_ints(&[Some(1), None, Some(3)])]);
        let e =
            bin(col(0, DataType::Int64), BinaryOp::Multiply, lit(Value::Int(2)), DataType::Int64);
        let c = eval(&e, &b).unwrap();
        assert_eq!(vals(&c), vec![Some(Value::Int(2)), None, Some(Value::Int(6))]);
    }

    #[test]
    fn unary_negate_and_not() {
        let b = blk(vec![ints(&[5, -5]), tri(&[Some(true), None])]);
        let n = eval(
            &B::Unary {
                op: UnaryOp::Neg,
                expr: Box::new(col(0, DataType::Int64)),
                ty: DataType::Int64,
            },
            &b,
        )
        .unwrap();
        assert_eq!(i_of(&n), vec![-5, 5]);

        let not = eval(
            &B::Unary {
                op: UnaryOp::Not,
                expr: Box::new(col(1, DataType::Bool)),
                ty: DataType::Bool,
            },
            &b,
        )
        .unwrap();
        assert_eq!(not.value(0), Value::Bool(false));
        assert!(not.is_null(1), "NOT NULL is NULL");
    }

    #[test]
    fn concat_renders_non_string_operands() {
        let b = blk(vec![strs(&["a", "b"]), ints(&[1, 2])]);
        let e = bin(
            col(0, DataType::String),
            BinaryOp::Concat,
            col(1, DataType::Int64),
            DataType::String,
        );
        let c = eval(&e, &b).unwrap();
        assert_eq!(c.value(0), Value::str("a1"));
        assert_eq!(c.value(1), Value::str("b2"));
    }

    // ----------------------------------------------------------- comparison

    #[test]
    fn comparisons_over_every_same_kind_fast_path() {
        let b = blk(vec![
            uints(&[1, 5, 9]),
            ints(&[1, 5, 9]),
            floats(&[1.0, 5.0, 9.0]),
            strs(&["a", "m", "z"]),
        ]);
        let cases: Vec<(usize, DataType, Value)> = vec![
            (0, DataType::UInt64, Value::UInt(5)),
            (1, DataType::Int64, Value::Int(5)),
            (2, DataType::Float64, Value::Float(5.0)),
            (3, DataType::String, Value::str("m")),
        ];
        for (i, ty, v) in cases {
            let lt = eval(&cmpx(col(i, ty.clone()), BinaryOp::Lt, lit(v.clone())), &b).unwrap();
            assert_eq!(u_of(&lt), vec![1, 0, 0], "col {i} <");
            let ge = eval(&cmpx(col(i, ty.clone()), BinaryOp::GtEq, lit(v.clone())), &b).unwrap();
            assert_eq!(u_of(&ge), vec![0, 1, 1], "col {i} >=");
            let eq = eval(&cmpx(col(i, ty), BinaryOp::Eq, lit(v)), &b).unwrap();
            assert_eq!(u_of(&eq), vec![0, 1, 0], "col {i} =");
        }
    }

    #[test]
    fn comparison_across_signedness_is_exact() {
        // 2^63 as a u64 exceeds every i64; going through f64 would lose the
        // distinction, so the mixed path widens to i128 instead.
        let b = blk(vec![uints(&[1 << 63, 1]), ints(&[-1, 1])]);
        let e = cmpx(col(0, DataType::UInt64), BinaryOp::Gt, col(1, DataType::Int64));
        assert_eq!(u_of(&eval(&e, &b).unwrap()), vec![1, 0]);
    }

    #[test]
    fn comparison_with_null_is_null() {
        let b = blk(vec![nullable_ints(&[Some(1), None, Some(9)])]);
        let e = cmpx(col(0, DataType::Int64), BinaryOp::Gt, lit(Value::Int(5)));
        let c = eval(&e, &b).unwrap();
        assert_eq!(vals(&c), vec![Some(Value::Bool(false)), None, Some(Value::Bool(true))]);
    }

    #[test]
    fn nan_compares_like_value_ordering() {
        let b = blk(vec![floats(&[f64::NAN, 1.0])]);
        let e = cmpx(col(0, DataType::Float64), BinaryOp::Eq, lit(Value::Float(f64::NAN)));
        // `Value::cmp` calls two NaNs equal; the evaluator has to agree.
        assert_eq!(u_of(&eval(&e, &b).unwrap()), vec![1, 0]);
    }

    // ------------------------------------------------- decimal comparison

    /// The bug this path exists for. A `Decimal64(2)` lane is a count of
    /// cents, so every one of these compared 381 against the literal before
    /// [`eval_cmp_decimal`]: `WHERE price > 2.0` returned *both* rows.
    #[test]
    fn decimal_compares_by_value_not_by_lane() {
        let b = blk(vec![decs(2, &[381, 119])]); // 3.81, 1.19
        let d = DataType::Decimal64(2);
        for (r, what) in [
            (lit(Value::Int(2)), "integer literal"),
            (lit(Value::UInt(2)), "unsigned literal"),
            (lit(Value::Float(2.0)), "float literal"),
            (lit(Value::Decimal(2, 0)), "scale-0 decimal"),
            (lit(Value::Decimal(20_000, 4)), "wider decimal"),
        ] {
            let e = cmpx(col(0, d.clone()), BinaryOp::Gt, r);
            assert_eq!(u_of(&eval(&e, &b).unwrap()), vec![1, 0], "price > 2 vs {what}");
        }
        // ...and the mirror image, with the decimal on the right.
        let e = cmpx(lit(Value::Int(2)), BinaryOp::Lt, col(0, d));
        assert_eq!(u_of(&eval(&e, &b).unwrap()), vec![1, 0]);
    }

    /// Equal scales are the fast path: the lanes *are* commensurable, so the
    /// answer must be bit-identical to the same lanes typed `Int64`. This is
    /// the test that fails if the scale check ever turns into a per-row branch
    /// that rescales when it should not.
    #[test]
    fn equal_scales_match_the_int64_path_exactly() {
        let lanes = [-500i64, 0, 119, 381, i64::MAX];
        let b = blk(vec![decs(2, &lanes), ints(&lanes)]);
        for op in [BinaryOp::Lt, BinaryOp::LtEq, BinaryOp::Gt, BinaryOp::GtEq, BinaryOp::Eq] {
            let dec = cmpx(col(0, DataType::Decimal64(2)), op, lit(Value::Decimal(119, 2)));
            let int = cmpx(col(1, DataType::Int64), op, lit(Value::Int(119)));
            assert_eq!(eval(&dec, &b).unwrap(), eval(&int, &b).unwrap(), "{op:?}");
        }
        // Column against column at the same scale, likewise.
        let cc = cmpx(col(0, DataType::Decimal64(2)), BinaryOp::Gt, col(0, DataType::Decimal64(2)));
        assert_eq!(u_of(&eval(&cc, &b).unwrap()), vec![0; 5]);
    }

    #[test]
    fn differing_scales_rescale_in_both_directions() {
        // 1.50 vs 1.5000 vs 1.4999: the same number spelled three ways.
        let b = blk(vec![decs(2, &[150, 150]), decs(4, &[15_000, 14_999])]);
        let (lo, hi) = (col(0, DataType::Decimal64(2)), col(1, DataType::Decimal64(4)));
        assert_eq!(u_of(&eval(&cmpx(lo.clone(), BinaryOp::Eq, hi.clone()), &b).unwrap()), vec![1, 0]);
        assert_eq!(u_of(&eval(&cmpx(lo.clone(), BinaryOp::Gt, hi.clone()), &b).unwrap()), vec![0, 1]);
        // Swapping the operands must swap the answer, not change it: the
        // rescale is symmetric or one of the two factors is on the wrong side.
        assert_eq!(u_of(&eval(&cmpx(hi.clone(), BinaryOp::Lt, lo.clone()), &b).unwrap()), vec![0, 1]);
        assert_eq!(u_of(&eval(&cmpx(hi, BinaryOp::GtEq, lo), &b).unwrap()), vec![1, 0]);
    }

    /// The whole reason the rescale happens in `i128`. At scale 2 these lanes
    /// are ~9.2e16 dollars, and `lane * 100` to reach scale 4 overflows `i64`
    /// by two orders of magnitude -- in release mode that wraps to a negative
    /// number and reverses the comparison.
    #[test]
    fn rescale_near_i64_max_stays_exact() {
        let b = blk(vec![decs(2, &[i64::MAX, i64::MAX - 1, i64::MIN]), decs(4, &[0, 0, 0])]);
        let (a, c) = (col(0, DataType::Decimal64(2)), col(1, DataType::Decimal64(4)));
        assert_eq!(u_of(&eval(&cmpx(a.clone(), BinaryOp::Gt, c), &b).unwrap()), vec![1, 1, 0]);
        // Same magnitudes against a scale-0 integer, where the factor is 10^2
        // on the *other* side.
        let e = cmpx(a, BinaryOp::Gt, lit(Value::Int(i64::MAX)));
        assert_eq!(u_of(&eval(&e, &b).unwrap()), vec![0, 0, 0], "9.2e16 is not > 9.2e18");
    }

    #[test]
    fn decimal_comparison_propagates_nulls_from_either_side() {
        let mut d = ColumnBuilder::new(DataType::Decimal64(2).to_nullable());
        for v in [Some(381i64), None, Some(119)] {
            match v {
                Some(u) => d.push_value(&Value::Decimal(u, 2)).unwrap(),
                None => d.push_null(),
            }
        }
        let b = blk(vec![d.finish(), nullable_ints(&[Some(2), Some(2), None])]);
        let e = cmpx(col(0, DataType::Decimal64(2)), BinaryOp::Gt, col(1, DataType::Int64));
        let c = eval(&e, &b).unwrap();
        assert_eq!(vals(&c), vec![Some(Value::Bool(true)), None, None]);
    }

    /// The evaluator and the constant folder must not disagree about the same
    /// pair of values, or a predicate would mean something different depending
    /// on whether the optimizer got to it first.
    #[test]
    fn decimal_comparison_agrees_with_value_ordering() {
        let b = blk(vec![decs(2, &[381])]);
        let probes = [
            Value::Int(3),
            Value::Int(4),
            Value::UInt(3),
            Value::Float(3.81),
            Value::Float(3.8),
            Value::Decimal(38_100, 4),
            Value::Decimal(3810, 3),
            Value::Decimal(38, 1),
        ];
        for p in probes {
            for op in [BinaryOp::Lt, BinaryOp::Eq, BinaryOp::Gt] {
                let got = eval(&cmpx(col(0, DataType::Decimal64(2)), op, lit(p.clone())), &b)
                    .unwrap();
                let want = keep(keep_mask(op), Value::Decimal(381, 2).cmp(&p)) as u64;
                assert_eq!(u_of(&got), vec![want], "3.81 {op:?} {p}");
            }
        }
    }

    /// Arithmetic is the registry's job (`dec_arith` rescales to the result
    /// scale before it adds), and this pins that the sugar in `eval_binary`
    /// really does reach it -- the same hazard, one operator over.
    #[test]
    fn mixed_scale_arithmetic_rescales_too() {
        let b = blk(vec![decs(2, &[150]), decs(4, &[15_000])]);
        let (lo, hi) = (col(0, DataType::Decimal64(2)), col(1, DataType::Decimal64(4)));
        let sum = bin(lo.clone(), BinaryOp::Plus, hi, DataType::Decimal64(4));
        let c = eval(&sum, &b).unwrap();
        assert_eq!(c.ty, DataType::Decimal64(4));
        assert_eq!(i_of(&c), vec![30_000], "1.50 + 1.5000 is 3.0000, not 1.50 + 15000 units");

        // ...and against a plain integer, where the integer is scale 0.
        let plus = bin(lo, BinaryOp::Plus, lit(Value::Int(2)), DataType::Decimal64(2));
        assert_eq!(i_of(&eval(&plus, &b).unwrap()), vec![350], "1.50 + 2 is 3.50");
    }

    // -------------------------------------------------- three-valued logic

    #[test]
    fn and_or_follow_three_valued_logic() {
        // every combination of TRUE / FALSE / NULL
        let a = tri(&[
            Some(true),
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(false),
            None,
            None,
            None,
        ]);
        let c = tri(&[
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(false),
            None,
        ]);
        let b = blk(vec![a, c]);
        let t = |v: bool| Some(Value::Bool(v));

        let and = eval(
            &bin(col(0, DataType::Bool), BinaryOp::And, col(1, DataType::Bool), DataType::Bool),
            &b,
        )
        .unwrap();
        assert_eq!(
            vals(&and),
            vec![t(true), t(false), None, t(false), t(false), t(false), None, t(false), None]
        );

        let or = eval(
            &bin(col(0, DataType::Bool), BinaryOp::Or, col(1, DataType::Bool), DataType::Bool),
            &b,
        )
        .unwrap();
        assert_eq!(
            vals(&or),
            vec![t(true), t(true), t(true), t(true), t(false), None, t(true), None, None]
        );
    }

    #[test]
    fn predicate_rejects_null_and_false_alike() {
        let b = blk(vec![tri(&[Some(true), Some(false), None, Some(true)])]);
        let sel = eval_predicate(&col(0, DataType::Bool), &b).unwrap();
        assert_eq!(sel, vec![0, 3], "NULL must fail the filter, exactly like FALSE");
    }

    #[test]
    fn predicate_on_a_comparison_with_nulls() {
        let b = blk(vec![nullable_ints(&[Some(1), None, Some(7), Some(9)])]);
        let e = cmpx(col(0, DataType::Int64), BinaryOp::Gt, lit(Value::Int(5)));
        assert_eq!(eval_predicate(&e, &b).unwrap(), vec![2, 3]);
    }

    #[test]
    fn predicate_on_non_bool_uses_sql_truthiness() {
        let b = blk(vec![ints(&[0, 3, 0, -1]), strs(&["", "x", "y", ""])]);
        assert_eq!(eval_predicate(&col(0, DataType::Int64), &b).unwrap(), vec![1, 3]);
        assert_eq!(eval_predicate(&col(1, DataType::String), &b).unwrap(), vec![1, 2]);
    }

    // ----------------------------------------------------------------- CAST

    #[test]
    fn cast_between_families() {
        let b = blk(vec![ints(&[1, 2]), strs(&["42", "7"])]);
        let s = eval(&B::Cast { expr: Box::new(col(0, DataType::Int64)), ty: DataType::String }, &b)
            .unwrap();
        assert_eq!(s.value(0), Value::str("1"));

        let n = eval(&B::Cast { expr: Box::new(col(1, DataType::String)), ty: DataType::Int64 }, &b)
            .unwrap();
        assert_eq!(i_of(&n), vec![42, 7]);
    }

    #[test]
    fn cast_preserves_nulls_and_relabels_for_free() {
        let b = blk(vec![nullable_ints(&[Some(1), None])]);
        let c = eval(&B::Cast { expr: Box::new(col(0, DataType::Int64)), ty: DataType::Int64 }, &b)
            .unwrap();
        assert!(c.is_null(1));
        assert!(c.ty.is_nullable(), "a live mask forces the type to stay Nullable");

        let f =
            eval(&B::Cast { expr: Box::new(col(0, DataType::Int64)), ty: DataType::Float64 }, &b)
                .unwrap();
        assert_eq!(f.value(0), Value::Float(1.0));
        assert!(f.is_null(1));
    }

    #[test]
    fn out_of_range_cast_reports_an_error() {
        let b = blk(vec![ints(&[300])]);
        let e = B::Cast { expr: Box::new(col(0, DataType::Int64)), ty: DataType::UInt8 };
        assert!(eval(&e, &b).is_err());
    }

    // ----------------------------------------------------------------- CASE

    #[test]
    fn case_picks_the_first_true_branch() {
        let b = blk(vec![ints(&[1, 2, 3, 4])]);
        let e = B::Case {
            when_then: vec![
                (
                    cmpx(col(0, DataType::Int64), BinaryOp::Lt, lit(Value::Int(2))),
                    lit(Value::Int(100)),
                ),
                (
                    cmpx(col(0, DataType::Int64), BinaryOp::Lt, lit(Value::Int(4))),
                    lit(Value::Int(200)),
                ),
            ],
            else_result: Some(Box::new(lit(Value::Int(300)))),
            ty: DataType::Int64,
        };
        assert_eq!(i_of(&eval(&e, &b).unwrap()), vec![100, 200, 200, 300]);
    }

    #[test]
    fn case_without_else_is_null_and_null_conditions_never_fire() {
        let b = blk(vec![tri(&[Some(true), Some(false), None])]);
        let e = B::Case {
            when_then: vec![(col(0, DataType::Bool), lit(Value::Int(1)))],
            else_result: None,
            ty: DataType::Int64,
        };
        let c = eval(&e, &b).unwrap();
        assert_eq!(vals(&c), vec![Some(Value::Int(1)), None, None]);
    }

    #[test]
    fn case_gathers_from_columns_not_just_literals() {
        let b = blk(vec![ints(&[1, 2]), ints(&[10, 20]), ints(&[30, 40])]);
        let e = B::Case {
            when_then: vec![(
                cmpx(col(0, DataType::Int64), BinaryOp::Eq, lit(Value::Int(1))),
                col(1, DataType::Int64),
            )],
            else_result: Some(Box::new(col(2, DataType::Int64))),
            ty: DataType::Int64,
        };
        assert_eq!(i_of(&eval(&e, &b).unwrap()), vec![10, 40]);
    }

    // ------------------------------------------------------------------- IN

    #[test]
    fn in_list_and_its_negation() {
        let b = blk(vec![ints(&[1, 2, 3])]);
        let mk = |neg| B::InList {
            expr: Box::new(col(0, DataType::Int64)),
            list: vec![Value::Int(1), Value::Int(3)],
            negated: neg,
        };
        assert_eq!(u_of(&eval(&mk(false), &b).unwrap()), vec![1, 0, 1]);
        assert_eq!(u_of(&eval(&mk(true), &b).unwrap()), vec![0, 1, 0]);
    }

    #[test]
    fn in_list_takes_the_sorted_path_for_long_lists() {
        let b = blk(vec![ints(&[5, 999])]);
        let e = B::InList {
            expr: Box::new(col(0, DataType::Int64)),
            list: (0..40i64).map(Value::Int).collect(),
            negated: false,
        };
        assert_eq!(u_of(&eval(&e, &b).unwrap()), vec![1, 0]);
    }

    #[test]
    fn in_list_null_semantics() {
        let b = blk(vec![nullable_ints(&[Some(1), Some(9), None])]);
        let e = B::InList {
            expr: Box::new(col(0, DataType::Int64)),
            list: vec![Value::Int(1), Value::Null],
            negated: false,
        };
        let c = eval(&e, &b).unwrap();
        // hit -> true; miss against a list holding NULL -> NULL; NULL -> NULL
        assert_eq!(vals(&c), vec![Some(Value::Bool(true)), None, None]);
    }

    // ---------------------------------------------------------- LIKE / NULL

    #[test]
    fn like_and_ilike_with_negation() {
        let b = blk(vec![strs(&["alpha", "beta", "ALPHA"])]);
        let mk = |neg, ci| B::Like {
            expr: Box::new(col(0, DataType::String)),
            pattern: "al%".into(),
            negated: neg,
            case_insensitive: ci,
        };
        assert_eq!(u_of(&eval(&mk(false, false), &b).unwrap()), vec![1, 0, 0]);
        assert_eq!(u_of(&eval(&mk(true, false), &b).unwrap()), vec![0, 1, 1]);
        assert_eq!(u_of(&eval(&mk(false, true), &b).unwrap()), vec![1, 0, 1]);
    }

    #[test]
    fn like_renders_non_string_subjects() {
        let b = blk(vec![ints(&[123, 456])]);
        let e = B::Like {
            expr: Box::new(col(0, DataType::Int64)),
            pattern: "1%".into(),
            negated: false,
            case_insensitive: false,
        };
        assert_eq!(u_of(&eval(&e, &b).unwrap()), vec![1, 0]);
    }

    #[test]
    fn is_null_and_is_not_null() {
        let b = blk(vec![nullable_ints(&[Some(1), None])]);
        let n = eval(&B::IsNull { expr: Box::new(col(0, DataType::Int64)), negated: false }, &b)
            .unwrap();
        assert_eq!(u_of(&n), vec![0, 1]);
        assert!(!n.has_nulls(), "IS NULL is never itself NULL");

        let nn = eval(&B::IsNull { expr: Box::new(col(0, DataType::Int64)), negated: true }, &b)
            .unwrap();
        assert_eq!(u_of(&nn), vec![1, 0]);
    }

    // --------------------------------------------------------------- scalar

    #[test]
    fn scalar_calls_reach_the_registry() {
        let b = blk(vec![strs(&["Abc", "dEf"])]);
        let f = functions::scalar("upper").unwrap();
        let e = B::Scalar { func: f, args: vec![col(0, DataType::String)], ty: DataType::String };
        let c = eval(&e, &b).unwrap();
        assert_eq!(c.value(0), Value::str("ABC"));
        assert_eq!(c.value(1), Value::str("DEF"));
    }

    #[test]
    fn operator_sugar_agrees_with_the_registry_entry() {
        // `a + b` and `plus(a, b)` must be indistinguishable.
        let b = blk(vec![ints(&[1, 2]), ints(&[10, 20])]);
        let sugar =
            bin(col(0, DataType::Int64), BinaryOp::Plus, col(1, DataType::Int64), DataType::Int64);
        let explicit = B::Scalar {
            func: functions::scalar("plus").unwrap(),
            args: vec![col(0, DataType::Int64), col(1, DataType::Int64)],
            ty: DataType::Int64,
        };
        assert_eq!(eval(&sugar, &b).unwrap(), eval(&explicit, &b).unwrap());
    }

    #[test]
    fn nested_expressions_compose() {
        // (a * 2 + b) > 10 AND a IS NOT NULL
        let b = blk(vec![nullable_ints(&[Some(1), Some(6), None]), ints(&[1, 1, 1])]);
        let arith = bin(
            bin(col(0, DataType::Int64), BinaryOp::Multiply, lit(Value::Int(2)), DataType::Int64),
            BinaryOp::Plus,
            col(1, DataType::Int64),
            DataType::Int64,
        );
        let e = bin(
            cmpx(arith, BinaryOp::Gt, lit(Value::Int(10))),
            BinaryOp::And,
            B::IsNull { expr: Box::new(col(0, DataType::Int64)), negated: true },
            DataType::Bool,
        );
        assert_eq!(eval_predicate(&e, &b).unwrap(), vec![1]);
    }

    /// A tree deeper than [`MAX_EXPR_DEPTH`] is refused rather than run: the
    /// evaluator recurses once per node, and "abort the process" is not an
    /// acceptable answer to a query. The binder's own guard means nothing that
    /// *parses* can get here, so the tree is built by hand -- which is exactly
    /// the surface this backstop covers.
    #[test]
    fn a_tree_deeper_than_the_guard_is_an_error_not_a_crash() {
        let b = blk(vec![ints(&[1, 2])]);
        let mut e = col(0, DataType::Int64);
        for _ in 0..MAX_EXPR_DEPTH + 5 {
            e = B::Unary { op: UnaryOp::Neg, expr: Box::new(e), ty: DataType::Int64 };
        }
        let msg = eval(&e, &b).unwrap_err().to_string();
        assert!(msg.contains("nests more than"), "{msg}");
        // `eval_predicate` and `eval_all` share the same entry, so they are
        // guarded by construction rather than by a second check.
        assert!(eval_predicate(&e, &b).is_err());
        assert!(eval_all(std::slice::from_ref(&e), &b).is_err());
    }

    /// The depth is per *path*, not per node visited: a wide `CASE` with a
    /// hundred shallow arms is not deep, and must still evaluate.
    #[test]
    fn the_guard_counts_nesting_not_size() {
        let b = blk(vec![ints(&[7])]);
        let when_then: Vec<(B, B)> = (0..150)
            .map(|k| {
                (
                    cmpx(col(0, DataType::Int64), BinaryOp::Eq, lit(Value::Int(k))),
                    lit(Value::Int(k * 10)),
                )
            })
            .collect();
        let e = B::Case { when_then, else_result: None, ty: DataType::Int64.to_nullable() };
        assert_eq!(vals(&eval(&e, &b).unwrap()), vec![Some(Value::Int(70))]);

        // ...and one level under the ceiling still evaluates.
        let mut deep = col(0, DataType::Int64);
        for _ in 0..MAX_EXPR_DEPTH - 2 {
            deep = B::Unary { op: UnaryOp::Neg, expr: Box::new(deep), ty: DataType::Int64 };
        }
        assert!(eval(&deep, &b).is_ok());
    }
}
