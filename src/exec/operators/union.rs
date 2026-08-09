//! The three ANSI set operations: `UNION`, `INTERSECT` and `EXCEPT`.
//!
//! `UNION ALL` is concatenation and `UNION` is concatenation plus dedup;
//! `UNION DISTINCT` is not a separate implementation but
//! [`super::distinct::Distinct`] stacked on the concatenation, which is exactly
//! what it means. Composing operators instead of special-casing keeps the dedup
//! logic in one place and means `UNION` automatically inherits `DISTINCT`'s
//! streaming behaviour.
//!
//! `INTERSECT` and `EXCEPT` are [`SetDiff`], one operator covering both and all
//! four `ALL`/`DISTINCT` combinations.
//!
//! ## The semantics, which is where these go wrong
//!
//! * **`DISTINCT` is the default**, for all three. `A INTERSECT B` yields each
//!   surviving tuple once however many copies either side held.
//! * **`ALL` keeps multiplicity**, and the two rules are different:
//!   `INTERSECT ALL` emits `min(m, n)` copies, `EXCEPT ALL` emits
//!   `max(m - n, 0)`, where `m` and `n` are the tuple's multiplicity on the
//!   left and right. Both are implemented. Accepting `ALL` and quietly running
//!   `DISTINCT` would be the accept-and-ignore failure this engine has spent
//!   seven waves removing, so it is not an option here.
//! * **`INTERSECT` binds tighter than `UNION` and `EXCEPT`**, which are equal
//!   in precedence and left-associative. `A UNION B INTERSECT C` is
//!   `A UNION (B INTERSECT C)`. That is a parser rule -- see `set_term` in
//!   `sql::parser` -- and getting it wrong changes the answer silently.
//! * **NULLs are not distinct.** Two NULLs *match* each other in a set
//!   operation, which is the opposite of what `=` does and the single most
//!   common bug in set-op implementations. Nothing here implements a second
//!   notion of equality to get that: the key is the same [`GroupKey`] that
//!   `DISTINCT` and `GROUP BY` build, whose `Eq` already treats `NULL` as a
//!   value and already compares numerics across representations, so
//!   `Date(5)` matches `UInt(5)` for the same reason `GROUP BY` puts them in
//!   one group.
//! * **Arity and type compatibility** are checked once, in the binder, by the
//!   same `set_schema` that `UNION` uses -- same widths, `DataType::promote`
//!   per column. No cast is inserted: matching goes through `Value`'s
//!   representation-independent `Eq`, and only the *output* blocks are retyped
//!   (see [`coerce`]).
//!
//! ## Why a dedicated operator rather than a semi/anti join
//!
//! `INTERSECT` is a semi-join and `EXCEPT` is an anti-join over the same
//! inputs, and this engine already lowers `IN (SELECT ...)` / `NOT IN` that way
//! (`LogicalPlan::in_subquery`: an inner join over a `Distinct`, or a left join
//! plus `IS NULL`). Reusing that was tried on paper and rejected for three
//! reasons, in order of severity:
//!
//! 1. **It has the wrong NULL semantics and cannot be given the right ones.**
//!    A join key tuple containing a NULL matches nothing -- `join.rs` skips
//!    such rows when building *and* when probing, because `NULL = NULL` is
//!    unknown. That is correct for a join and wrong for a set operation, and
//!    it is not a flag: the whole `NOT IN` census machinery exists precisely
//!    because SQL's join equality is not set equality.
//! 2. **`ALL` is not expressible.** `min(m, n)` and `max(m - n, 0)` need per
//!    tuple multiplicities. A join over `Distinct` inputs has thrown them
//!    away, and the `ROW_NUMBER() OVER (PARTITION BY every column)` rewrite
//!    that restores them puts a *sort* of both sides under the operation --
//!    the one shape this had to avoid.
//! 3. **It materializes both sides.** The hash join drains left and right
//!    before emitting, so `big INTERSECT small` would cost the big side's
//!    memory. `SetDiff` builds only over the branches being matched *against*
//!    and streams branch 0, so the table is bounded by the small side and the
//!    large side never exists in memory at all.
//!
//! The aggregate-based rewrite (tag each branch, `GROUP BY` every column,
//! filter on the per-branch counts) does get NULLs right, because `GROUP BY`
//! uses the same key -- but it hashes `|L| + |R|` rows into a table of
//! `distinct(L ∪ R)` entries where the operator hashes `|R|` into `distinct(R)`
//! and walks `L` against it. Both alternatives were measured; see the numbers
//! below.
//!
//! ## Shape and cost
//!
//! Branch 0 **streams**; branches 1.. are drained into one
//! `FastMap<GroupKey, u64>` of multiplicities before the first output row.
//! That is where "build over the small side" comes from -- write the small
//! relation on the right and it is the only thing in memory.
//!
//! Steady-state allocation is zero. The probe key is one [`GroupKey`] whose
//! `Vec` is cleared and refilled per row rather than allocated per row (which
//! is what `row_key` does for `DISTINCT`), the selection vector is reused
//! across blocks, and the only allocations after the build are the new entries
//! `EXCEPT DISTINCT` inserts -- one per *distinct output* tuple, not per row.
//!
//! ## Measured
//!
//! Through the CLI in one process, `--release`, A/B interleaved, best-of-9 per
//! side, over `count()` so the numbers are the operation and not the
//! rendering. Two `Int64` tables: `l` has 1M rows over 500k distinct values,
//! `r` has 1M (half of them shared with `l`) or 100.
//!
//! ```text
//!                          1M x 1M      1M x 100
//!   a INTERSECT b          112.9 ms       18.5 ms
//!   a INTERSECT ALL b      112.6          18.4
//!   a EXCEPT b             136.8          81.3
//!   a EXCEPT ALL b         115.9          21.4
//! ```
//!
//! `EXCEPT` at 1M x 100 is the outlier and the reason is the `DISTINCT`, not
//! the subtraction: 499 900 tuples survive and each one's first copy is
//! *inserted* into the table. `EXCEPT ALL` over the same data holds 100
//! entries for the whole query and costs a quarter as much.
//!
//! Against the rewrites -- the same answers spelled with nodes that already
//! existed, which is what a dedicated operator has to beat (best-of-7,
//! interleaved in one process):
//!
//! ```text
//!                                       1M x 1M      1M x 100
//!   EXCEPT      LEFT JOIN + IS NULL     198.8 ms      149.4 ms
//!               SetDiff                 137.7          76.8     1.4x / 1.9x
//!   INTERSECT   JOIN over Distinct      210.3          17.6
//!               tag + GROUP BY          106.1          63.0
//!               SetDiff                 119.6          20.7
//! ```
//!
//! Peak RSS, which is the point of the shape rather than a side effect --
//! 1M x 100, `/usr/bin/time -l`:
//!
//! ```text
//!   EXCEPT ALL       SetDiff  11.0 MB      LEFT JOIN rewrite      84.9 MB
//!   INTERSECT ALL    SetDiff  10.7 MB      tag + GROUP BY rewrite 95.3 MB
//!   (a plain `count() ... WHERE a > k` over the same table:       13.6 MB)
//! ```
//!
//! Both set operations run *below the floor of a plain aggregate over the same
//! table*, because the 1M-row side is never materialized and the 100-row side
//! is a hundred entries.
//!
//! ## Where it is not a win
//!
//! **Building over the small side is not automatic, it is the query's word
//! order.** Branches 1.. are always the ones built, so writing the small
//! relation first costs 4.2x: `small INTERSECT big` 78.4 ms against
//! `big INTERSECT small` 18.6 ms, same data, best-of-9. `INTERSECT` is
//! commutative as a multiset operation so the operator *could* swap them; it
//! does not, because the rows would then come out in the other branch's order
//! and every other set operation here is deterministic in branch-0 order.
//! `EXCEPT` cannot swap at all. A planner that swapped on a cardinality
//! estimate, and only for `INTERSECT`, is the fix -- it belongs in
//! `planner::optimizer` with the join reordering, not here.
//!
//! **`INTERSECT DISTINCT` does not beat either rewrite outright.** The
//! tag-and-`GROUP BY` aggregate is 106 ms against 120 at 1M x 1M and the
//! semi-join is 17.6 against 20.7 at 1M x 100 -- both around 10%, both
//! reversing sign between repeat runs on a machine whose identical-code spread
//! is wider than that, so the honest reading is a wash. Neither is usable
//! anyway: the join drops NULL keys and the aggregate cannot express `ALL`.
//! What the numbers do say is that the probe is not free, and where it goes is
//! recorded on [`fill`].
//!
//! ## What it costs a query that does not use it
//!
//! Nothing. `SetOp::Union` returns from [`build_set`] before `SetDiff` is
//! constructed, and the plan node it hangs off is the one `UNION` already
//! used, so no plan grew a branch. Against the `a1b5baf` binary from a
//! worktree, `EXPLAIN` output is byte-identical for a filtered `count()`, a
//! `GROUP BY`, a hash join and a `UNION ALL`; timings, best-of-21 alternating
//! the two binaries:
//!
//! ```text
//!   count() + filter        0.369 -> 0.372 ms    GROUP BY 500k   39.24 -> 38.07 ms
//!   hash join 1M x 100     15.968 -> 16.075      UNION ALL 1M+1M  6.11 ->  6.10
//!   point lookup            0.018 -> 0.018       UNION DISTINCT 191.7 -> 197.5
//! ```
//!
//! Every ratio is between 0.97 and 1.03 on a machine whose identical-code
//! spread is wider than that.

use crate::catalog::Catalog;
use crate::common::{FastMap, Result};
use crate::planner::logical::LogicalPlan;
use crate::sql::ast::SetOp;
use crate::types::{Block, Column, ColumnBuilder, Schema, Value};

use super::{build, distinct::Distinct, GroupKey, MemGuard, Operator, QueryContext, ScanStats};

pub struct Union<'a> {
    inputs: Vec<Box<dyn Operator + 'a>>,
    schema: &'a Schema,
    cur: usize,
}

/// Build a set-operation pipeline. `UNION` is [`Union`] (plus a `Distinct`
/// unless `ALL`); `INTERSECT` and `EXCEPT` are [`SetDiff`].
pub fn build_set<'a>(
    inputs: &'a [LogicalPlan],
    op: SetOp,
    all: bool,
    schema: &'a Schema,
    catalog: &'a Catalog,
    ctx: &'a QueryContext,
) -> Result<Box<dyn Operator + 'a>> {
    // The context has to reach the branches: a `UNION` of two aggregates is
    // exactly the shape whose memory a budget needs to see.
    let ops: Vec<Box<dyn Operator + 'a>> = inputs
        .iter()
        .map(|p| build(p, catalog, ctx))
        .collect::<Result<_>>()?;
    // A one-branch `INTERSECT`/`EXCEPT` cannot be written in SQL, but a
    // rewrite could produce one, and both mean "the branch itself" -- deduped
    // unless `ALL`. Falling through to the concatenation gets that for free.
    if op == SetOp::Union || ops.len() < 2 {
        let u = Union { inputs: ops, schema, cur: 0 };
        return Ok(if all { Box::new(u) } else { Box::new(Distinct::new(Box::new(u))) });
    }
    let mut branches = ops.into_iter();
    Ok(Box::new(SetDiff {
        left: branches.next().expect("`ops.len() >= 2`, so branch 0 exists"),
        rest: branches.collect(),
        schema,
        mode: Mode::of(op, all),
        counts: FastMap::default(),
        key: GroupKey(Vec::with_capacity(schema.len())),
        sel: Vec::with_capacity(crate::common::BLOCK_SIZE),
        built: false,
        empty: false,
        guard: MemGuard::new(ctx, "set operation"),
        ctx,
    }))
}

impl Operator for Union<'_> {
    fn schema(&self) -> &Schema {
        self.schema
    }

    fn stats(&self) -> ScanStats {
        let mut s = ScanStats::default();
        for i in &self.inputs {
            s.merge(&i.stats());
        }
        s
    }

    fn next(&mut self) -> Result<Option<Block>> {
        while self.cur < self.inputs.len() {
            match self.inputs[self.cur].next()? {
                Some(b) if b.rows() > 0 => return Ok(Some(coerce(b, self.schema)?)),
                Some(_) => continue,
                None => self.cur += 1,
            }
        }
        Ok(None)
    }
}

/// The four variants, resolved once at build time so the probe loop can be
/// chosen once per *block* instead of tested once per row.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    IntersectAll,
    IntersectDistinct,
    ExceptAll,
    ExceptDistinct,
}

impl Mode {
    fn of(op: SetOp, all: bool) -> Mode {
        match (op, all) {
            (SetOp::Intersect, true) => Mode::IntersectAll,
            (SetOp::Intersect, false) => Mode::IntersectDistinct,
            (SetOp::Except, true) => Mode::ExceptAll,
            // `SetOp::Union` never reaches here: `build_set` returns first.
            (_, _) => Mode::ExceptDistinct,
        }
    }

    fn is_intersect(self) -> bool {
        matches!(self, Mode::IntersectAll | Mode::IntersectDistinct)
    }
}

/// `INTERSECT` and `EXCEPT`, with and without `ALL`.
///
/// One table of multiplicities over branches 1.., then one streaming pass over
/// branch 0. The four variants differ only in what the probe does with a
/// count, which is why they are one operator: the build, the key, the
/// selection and the coercion are identical.
pub struct SetDiff<'a> {
    left: Box<dyn Operator + 'a>,
    /// Branches 1.., all of them drained into `counts` before the first row.
    rest: Vec<Box<dyn Operator + 'a>>,
    schema: &'a Schema,
    mode: Mode,
    /// Tuple -> remaining multiplicity. For `EXCEPT` the branches' counts are
    /// *summed*, because `(A - B) - C` removes `B`'s copies and then `C`'s;
    /// for `INTERSECT` they are *minimised*, because a tuple survives only as
    /// often as the scarcest branch holds it.
    counts: FastMap<GroupKey, u64>,
    /// The probe key, cleared and refilled per row. One allocation per query.
    key: GroupKey,
    /// The kept-row indices for the current block. Reused.
    sel: Vec<u32>,
    built: bool,
    /// Set when the build proved the answer empty, so branch 0 is never read.
    empty: bool,
    guard: MemGuard,
    ctx: &'a QueryContext,
}

impl SetDiff<'_> {
    /// Drain branches 1.. into `counts`.
    fn build_table(&mut self) -> Result<()> {
        // Only allocated for a chain of three or more `INTERSECT`s, which is
        // the one shape that needs a second branch's counts before it can
        // minimise against the first's.
        let mut cur: FastMap<GroupKey, u64> = FastMap::default();
        for i in 0..self.rest.len() {
            let refine = self.mode.is_intersect() && i > 0;
            loop {
                self.ctx.check()?;
                let Some(b) = self.rest[i].next()? else { break };
                if b.rows() == 0 {
                    continue;
                }
                let key = &mut self.key;
                if refine {
                    // A tuple absent from `counts` cannot be in the
                    // intersection, so it is never inserted here: `cur` stays
                    // bounded by `counts` rather than by the branch.
                    for r in 0..b.rows() {
                        fill(key, &b.columns, r);
                        if self.counts.contains_key(key) {
                            bump(&mut cur, key);
                        }
                    }
                } else {
                    for r in 0..b.rows() {
                        fill(key, &b.columns, r);
                        bump(&mut self.counts, key);
                    }
                }
                let live = self.counts.len() + cur.len();
                self.guard.grow_to(entry_bytes(self.schema.len()) * live)?;
            }
            if refine {
                self.counts.retain(|k, v| match cur.get(k) {
                    Some(c) => {
                        *v = (*v).min(*c);
                        true
                    }
                    None => false,
                });
                cur.clear();
            }
            // An `INTERSECT` with nothing left to match cannot produce a row,
            // so neither the remaining branches nor branch 0 are read at all.
            // `EXCEPT` has no such shortcut: an empty table means *every* left
            // row survives.
            if self.mode.is_intersect() && self.counts.is_empty() {
                self.empty = true;
                return Ok(());
            }
        }
        Ok(())
    }
}

impl Operator for SetDiff<'_> {
    fn schema(&self) -> &Schema {
        self.schema
    }

    fn stats(&self) -> ScanStats {
        let mut s = self.left.stats();
        for i in &self.rest {
            s.merge(&i.stats());
        }
        s
    }

    fn next(&mut self) -> Result<Option<Block>> {
        if !self.built {
            self.built = true;
            self.build_table()?;
        }
        if self.empty {
            return Ok(None);
        }
        while let Some(b) = self.left.next()? {
            let n = b.rows();
            if n == 0 {
                continue;
            }
            let key = &mut self.key;
            let counts = &mut self.counts;
            let sel = &mut self.sel;
            sel.clear();
            // One match per block. Each arm is a flat walk whose body is three
            // instructions around the hash probe.
            match self.mode {
                Mode::IntersectAll => {
                    for r in 0..n {
                        fill(key, &b.columns, r);
                        if let Some(c) = counts.get_mut(key) {
                            if *c > 0 {
                                *c -= 1;
                                sel.push(r as u32);
                            }
                        }
                    }
                }
                // Zeroing the count is what makes this `DISTINCT` without a
                // second table: the first copy of a matching tuple consumes
                // the entry, every later copy sees a zero and is dropped.
                Mode::IntersectDistinct => {
                    for r in 0..n {
                        fill(key, &b.columns, r);
                        if let Some(c) = counts.get_mut(key) {
                            if *c > 0 {
                                *c = 0;
                                sel.push(r as u32);
                            }
                        }
                    }
                }
                Mode::ExceptAll => {
                    for r in 0..n {
                        fill(key, &b.columns, r);
                        match counts.get_mut(key) {
                            Some(c) if *c > 0 => *c -= 1,
                            _ => sel.push(r as u32),
                        }
                    }
                }
                // The emitted tuple is inserted with count 1, which makes the
                // table do double duty: "the right side had it" and "we have
                // already emitted it" are the same test, so `EXCEPT DISTINCT`
                // needs no `Distinct` above it.
                Mode::ExceptDistinct => {
                    for r in 0..n {
                        fill(key, &b.columns, r);
                        if !counts.contains_key(key) {
                            counts.insert(key.clone(), 1);
                            sel.push(r as u32);
                        }
                    }
                }
            }
            // Only `ExceptDistinct` can grow the table here; for the other
            // three this is one comparison against a high-water mark.
            self.guard.grow_to(entry_bytes(self.schema.len()) * self.counts.len())?;
            if self.sel.is_empty() {
                continue;
            }
            let out = if self.sel.len() == n { b } else { b.take(&self.sel) };
            return Ok(Some(coerce(out, self.schema)?));
        }
        Ok(None)
    }
}

/// Overwrite `key` with row `row` of `cols`. Reusing the `Vec` is the
/// difference between one allocation per query and one per row.
///
/// `Column::value` matches on `(&self.data, self.ty.base())` per value, and
/// `base()` walks a `Box` for a `Nullable` column, so this looks like exactly
/// the per-value type dispatch that belongs once per block. **It is not, and
/// hoisting it lost.** Building a `Vec<(&Column, Lane)>` per block -- one
/// enum per column carrying the already-chosen `Value` variant and a borrowed
/// lane slice -- and walking that instead measured, two binaries alternated,
/// 9 rounds x 3 runs each, `count()` over 1M rows:
///
/// ```text
///                            Column::value      hoisted lanes
///   INTERSECT     1M x 100     22.5 ms            27.1 ms
///   INTERSECT ALL 1M x 100     22.5               24.3
///   EXCEPT        1M x 100     90.4               93.2
///   EXCEPT ALL    1M x 1M     128.1              129.8
///   INTERSECT     1M x 1M     121.6              123.1
/// ```
///
/// Slower on every shape where the table is small, a wash where it is large,
/// and the same sign on a repeat run. `Column::value` is `#[inline]` and the
/// block's type is loop-invariant, so the dispatch was already hoisted by the
/// compiler; what the rewrite added was a `Vec` per block and a pointer pair
/// to load per column per row. Recorded so nobody tries it again.
#[inline]
fn fill(key: &mut GroupKey, cols: &[Column], row: usize) {
    key.0.clear();
    for c in cols {
        key.0.push(c.value(row));
    }
}

/// `counts[key] += 1`, allocating an owned key only for a tuple never seen.
#[inline]
fn bump(counts: &mut FastMap<GroupKey, u64>, key: &GroupKey) {
    match counts.get_mut(key) {
        Some(c) => *c += 1,
        None => {
            counts.insert(key.clone(), 1);
        }
    }
}

/// Bytes a live table entry costs: the key's `Value`s, the `Vec` header inside
/// [`GroupKey`], the count, and the map's own slot. String payloads are shared
/// `Arc<str>` and are not counted, so this is a floor -- enough for the budget
/// to notice a table growing without bound, which is what it is for.
#[inline]
fn entry_bytes(width: usize) -> usize {
    width * std::mem::size_of::<Value>() + std::mem::size_of::<(GroupKey, u64)>() + 16
}

/// Retype a branch's columns to the set operation's schema when the physical
/// representation differs. A no-op in the usual case.
///
/// Branches may disagree about column types where the plan's schema says
/// `Int64` but one branch produced `UInt64` (a literal `VALUES` leg, say). The
/// result set renders each block through its own column types, so a mismatch
/// would show up as inconsistent formatting between rows. Blocks are therefore
/// coerced on the way out -- and only when they actually differ, so the common
/// case costs one pointer comparison per column.
fn coerce(b: Block, schema: &Schema) -> Result<Block> {
    if b.width() != schema.len() {
        return Ok(b);
    }
    let needs = b
        .columns
        .iter()
        .enumerate()
        .any(|(i, c)| c.ty.physical() != schema.ty(i).physical());
    if !needs {
        return Ok(b);
    }
    let rows = b.rows();
    let mut out: Vec<Column> = Vec::with_capacity(b.width());
    for (i, c) in b.columns.iter().enumerate() {
        let want = schema.ty(i);
        if c.ty.physical() == want.physical() {
            out.push(c.clone());
            continue;
        }
        let ty = if c.has_nulls() { want.to_nullable() } else { want.clone() };
        let mut nb = ColumnBuilder::with_capacity(ty, rows);
        for r in 0..rows {
            if c.is_null(r) {
                nb.push_null();
            } else {
                nb.push_value(&c.value(r))?;
            }
        }
        out.push(nb.finish());
    }
    Block::new(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataType, Field, Value};

    fn schema() -> Schema {
        Schema::new(vec![Field::new("a", DataType::Int64)]).unwrap()
    }

    fn values_plan(vs: &[i64], s: &Schema) -> LogicalPlan {
        LogicalPlan::Values {
            rows: vs.iter().map(|&i| vec![Value::Int(i)]).collect(),
            schema: s.clone(),
        }
    }

    /// A plan whose single column is `Nullable(Int64)`, `None` meaning NULL.
    fn null_plan(vs: &[Option<i64>]) -> (LogicalPlan, Schema) {
        let s = Schema::new(vec![Field::new("a", DataType::Int64.to_nullable())]).unwrap();
        let rows = vs
            .iter()
            .map(|v| vec![v.map_or(Value::Null, Value::Int)])
            .collect();
        (LogicalPlan::Values { rows, schema: s.clone() }, s)
    }

    fn run_op(plans: Vec<LogicalPlan>, op: SetOp, all: bool, s: &Schema) -> Vec<Value> {
        let cat = Catalog::in_memory();
        let ctx = QueryContext::new();
        let mut o = build_set(&plans, op, all, s, &cat, &ctx).unwrap();
        let mut out = Vec::new();
        while let Some(b) = o.next().unwrap() {
            for i in 0..b.rows() {
                out.push(b.column(0).value(i));
            }
        }
        out
    }

    fn run(plans: Vec<LogicalPlan>, all: bool, s: &Schema) -> Vec<i64> {
        run_op(plans, SetOp::Union, all, s)
            .into_iter()
            .map(|v| v.as_i64().unwrap())
            .collect()
    }

    fn ints(plans: Vec<LogicalPlan>, op: SetOp, all: bool, s: &Schema) -> Vec<i64> {
        run_op(plans, op, all, s).into_iter().map(|v| v.as_i64().unwrap()).collect()
    }

    #[test]
    fn union_all_concatenates_in_order() {
        let s = schema();
        let plans = vec![values_plan(&[1, 2], &s), values_plan(&[2, 3], &s)];
        assert_eq!(run(plans, true, &s), vec![1, 2, 2, 3]);
    }

    #[test]
    fn union_without_all_deduplicates() {
        let s = schema();
        let plans = vec![values_plan(&[1, 2], &s), values_plan(&[2, 3], &s)];
        assert_eq!(run(plans, false, &s), vec![1, 2, 3]);
    }

    #[test]
    fn a_single_branch_is_passed_through() {
        let s = schema();
        assert_eq!(run(vec![values_plan(&[5, 5], &s)], true, &s), vec![5, 5]);
    }

    #[test]
    fn empty_branches_are_skipped_not_treated_as_end_of_stream() {
        let s = schema();
        let plans = vec![
            values_plan(&[], &s),
            values_plan(&[7], &s),
            values_plan(&[], &s),
            values_plan(&[8], &s),
        ];
        assert_eq!(run(plans, true, &s), vec![7, 8]);
    }

    #[test]
    fn branches_are_coerced_to_the_union_schema() {
        // Branch 2 produces UInt64 where the union schema says Int64.
        let s = schema();
        let plans = vec![
            values_plan(&[1], &s),
            LogicalPlan::Values {
                rows: vec![vec![Value::UInt(2)]],
                schema: Schema::new(vec![Field::new("a", DataType::UInt64)]).unwrap(),
            },
        ];
        let cat = Catalog::in_memory();
        let ctx = QueryContext::new();
        let mut op = build_set(&plans, SetOp::Union, true, &s, &cat, &ctx).unwrap();
        let mut tys = Vec::new();
        while let Some(b) = op.next().unwrap() {
            tys.push(b.column(0).ty.clone());
        }
        assert_eq!(tys, vec![DataType::Int64, DataType::Int64]);
    }

    #[test]
    fn no_branches_at_all() {
        let s = schema();
        assert!(run(vec![], true, &s).is_empty());
    }

    // ------------------------------------------------- intersect and except

    #[test]
    fn intersect_distinct_emits_each_match_once() {
        let s = schema();
        let l = values_plan(&[1, 2, 2, 3, 3, 3], &s);
        let r = values_plan(&[2, 3, 3, 4], &s);
        assert_eq!(ints(vec![l, r], SetOp::Intersect, false, &s), vec![2, 3]);
    }

    #[test]
    fn intersect_all_keeps_the_minimum_multiplicity() {
        let s = schema();
        // 2 appears 2x left / 1x right -> 1; 3 appears 3x / 2x -> 2.
        let l = values_plan(&[1, 2, 2, 3, 3, 3], &s);
        let r = values_plan(&[2, 3, 3, 4], &s);
        assert_eq!(ints(vec![l, r], SetOp::Intersect, true, &s), vec![2, 3, 3]);
    }

    #[test]
    fn except_distinct_emits_each_survivor_once() {
        let s = schema();
        let l = values_plan(&[1, 1, 2, 2, 3], &s);
        let r = values_plan(&[2], &s);
        assert_eq!(ints(vec![l, r], SetOp::Except, false, &s), vec![1, 3]);
    }

    #[test]
    fn except_all_subtracts_multiplicity() {
        let s = schema();
        // 1: 3 left - 1 right = 2; 2: 1 - 2 = 0 (not negative); 3: 1 - 0 = 1.
        let l = values_plan(&[1, 1, 1, 2, 3], &s);
        let r = values_plan(&[1, 2, 2], &s);
        assert_eq!(ints(vec![l, r], SetOp::Except, true, &s), vec![1, 1, 3]);
    }

    /// The rule that separates a set operation from a join: two NULLs match.
    #[test]
    fn nulls_match_each_other_on_both_sides() {
        let (l, s) = null_plan(&[Some(1), None, Some(2), None]);
        let (r, _) = null_plan(&[None, Some(2)]);
        assert_eq!(
            run_op(vec![l, r], SetOp::Intersect, false, &s),
            vec![Value::Null, Value::Int(2)],
            "NULL is a value for a set operation, unlike for `=`"
        );

        let (l, _) = null_plan(&[Some(1), None, Some(2)]);
        let (r, _) = null_plan(&[None]);
        assert_eq!(
            run_op(vec![l, r], SetOp::Except, false, &s),
            vec![Value::Int(1), Value::Int(2)],
            "the NULL row was removed by a NULL on the right"
        );
    }

    #[test]
    fn null_multiplicity_follows_the_all_rules_too() {
        let (l, s) = null_plan(&[None, None, None, Some(1)]);
        let (r, _) = null_plan(&[None, None]);
        assert_eq!(
            run_op(vec![l, r], SetOp::Except, true, &s),
            vec![Value::Null, Value::Int(1)],
            "3 - 2 = 1 NULL survives"
        );
        let (l, _) = null_plan(&[None, None, None, Some(1)]);
        let (r, _) = null_plan(&[None, None]);
        assert_eq!(
            run_op(vec![l, r], SetOp::Intersect, true, &s),
            vec![Value::Null, Value::Null],
            "min(3, 2) = 2 NULLs survive"
        );
    }

    #[test]
    fn empty_inputs_on_either_side() {
        let s = schema();
        let e = || values_plan(&[], &s);
        assert!(ints(vec![e(), values_plan(&[1], &s)], SetOp::Intersect, false, &s).is_empty());
        assert!(ints(vec![values_plan(&[1], &s), e()], SetOp::Intersect, false, &s).is_empty());
        assert!(ints(vec![e(), values_plan(&[1], &s)], SetOp::Except, false, &s).is_empty());
        assert_eq!(ints(vec![values_plan(&[1, 1], &s), e()], SetOp::Except, true, &s), vec![1, 1]);
        assert_eq!(ints(vec![values_plan(&[1, 1], &s), e()], SetOp::Except, false, &s), vec![1]);
    }

    /// An `INTERSECT` whose table came out empty must not read branch 0 at
    /// all -- the shortcut that makes `big INTERSECT (nothing)` free.
    #[test]
    fn an_empty_intersect_table_skips_the_streaming_side() {
        let s = schema();
        let cat = Catalog::in_memory();
        let ctx = QueryContext::new();
        let plans = vec![values_plan(&[1, 2, 3], &s), values_plan(&[], &s)];
        let mut op = build_set(&plans, SetOp::Intersect, true, &s, &cat, &ctx).unwrap();
        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn three_way_intersect_minimises_across_every_branch() {
        let s = schema();
        let plans = vec![
            values_plan(&[1, 1, 1, 2, 3], &s),
            values_plan(&[1, 1, 2], &s),
            values_plan(&[1, 1, 1, 1], &s),
        ];
        // 1: min(3, 2, 4) = 2. 2: absent from branch 3. 3: absent from 2.
        assert_eq!(ints(plans, SetOp::Intersect, true, &s), vec![1, 1]);
    }

    #[test]
    fn three_way_except_sums_the_subtrahends() {
        let s = schema();
        let plans = vec![
            values_plan(&[1, 1, 1, 1, 2], &s),
            values_plan(&[1], &s),
            values_plan(&[1, 1, 2], &s),
        ];
        // 1: 4 - 1 - 2 = 1. 2: 1 - 1 = 0.
        assert_eq!(ints(plans, SetOp::Except, true, &s), vec![1]);
    }

    #[test]
    fn multi_column_tuples_match_on_the_whole_row() {
        let s = Schema::new(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::String),
        ])
        .unwrap();
        let mk = |rows: Vec<(i64, &str)>| LogicalPlan::Values {
            rows: rows.into_iter().map(|(a, b)| vec![Value::Int(a), Value::str(b)]).collect(),
            schema: s.clone(),
        };
        let cat = Catalog::in_memory();
        let ctx = QueryContext::new();
        let plans = vec![mk(vec![(1, "x"), (1, "y"), (2, "x")]), mk(vec![(1, "y"), (2, "y")])];
        let mut op = build_set(&plans, SetOp::Except, false, &s, &cat, &ctx).unwrap();
        let mut got = Vec::new();
        while let Some(b) = op.next().unwrap() {
            for i in 0..b.rows() {
                got.push((b.column(0).value(i), b.column(1).value(i)));
            }
        }
        assert_eq!(got, vec![(Value::Int(1), Value::str("x")), (Value::Int(2), Value::str("x"))]);
    }

    /// Duplicates spread over more than one block must still be counted as one
    /// tuple: the table lives for the whole query, not for a batch.
    #[test]
    fn multiplicity_is_counted_across_block_boundaries() {
        use crate::common::BLOCK_SIZE;
        let s = schema();
        let many: Vec<i64> = (0..BLOCK_SIZE as i64 * 2 + 5).map(|i| i % 3).collect();
        let l = values_plan(&many, &s);
        let r = values_plan(&[0, 1], &s);
        assert_eq!(ints(vec![l, r], SetOp::Intersect, false, &s), vec![0, 1]);
    }

    #[test]
    fn the_build_side_is_charged_to_the_budget() {
        let s = schema();
        let vs: Vec<i64> = (0..2000).collect();
        let plans = vec![values_plan(&[1], &s), values_plan(&vs, &s)];
        let cat = Catalog::in_memory();
        let ctx = QueryContext::with_budget(4096);
        let mut op = build_set(&plans, SetOp::Intersect, true, &s, &cat, &ctx).unwrap();
        let e = op.next().unwrap_err().to_string();
        assert!(e.contains("set operation"), "{e}");
    }
}
