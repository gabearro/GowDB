//! Hash aggregation: `GROUP BY` and the bare-aggregate case.
//!
//! One [`Accumulator`] per aggregate per group, fed through
//! [`Accumulator::update`] with a **selection vector**. That signature is the
//! whole performance story: `update` is called once per group per *block*, not
//! once per row, so `sum` over a group folds a contiguous `&[i64]` in a tight
//! loop instead of paying a virtual call and a hash probe per value.
//!
//! Building those selection vectors takes two passes over each block: one to
//! resolve every row's group (the only per-row hashing that happens), then one
//! to bucket the row ids. Two linear passes plus one vectorized fold per group
//! beats one pass that dispatches per row, because the dispatch is what costs.
//!
//! ## Empty input
//!
//! `SELECT count(*) FROM empty` must return one row containing `0`, not zero
//! rows -- an aggregate with no `GROUP BY` is a fold over the whole relation
//! and a fold always has a result. So the single group is created eagerly at
//! construction time rather than on first sight of a row. With a `GROUP BY`
//! the opposite holds: no rows means no groups means no output.
//!
//! ## DISTINCT aggregates
//!
//! `count(DISTINCT x)` dedups argument tuples **per group**, across the whole
//! query rather than per block, so the seen-set has to live alongside the
//! accumulator. Only rows carrying a first-sighting reach `update`, which
//! keeps the accumulator itself oblivious to distinctness -- `sum(DISTINCT x)`
//! and `sum(x)` share one implementation.
//!
//! ## Output layout
//!
//! `[group columns..., aggregate columns...]`, matching the plan's schema. The
//! group key tuple is already stored (it is the hash key), so emitting it
//! costs nothing extra.
//!
//! ## When the groups do not fit
//!
//! They no longer have to. A table that fills the query's memory budget
//! *freezes* -- it stops admitting new keys, and rows whose key it does not
//! already hold are hash-partitioned to temp files and folded afterwards, one
//! partition at a time. [`accumulate_into`] carries the argument for why that
//! is correct and what it costs; the short version is that a group is never
//! split between memory and disk, so nothing is merged back and no accumulator
//! ever has to be serialized.
//!
//! ## Three separable steps
//!
//! [`accumulate`] folds an input into a [`Groups`], [`Groups::absorb`] combines
//! two of them and [`emit`] turns one into blocks. The operator is just those
//! three in a row; the split exists so
//! [`exchange`](crate::exec::operators::exchange) can run `accumulate` on N
//! threads over disjoint slices of the same scan and fold the partials with
//! `absorb` before a single `emit`. Nothing about the serial path changed --
//! the whole reason the accumulators carry a `merge` is that a partial
//! aggregate is a first-class value here, not a private detail of one operator.

use std::mem::size_of;

use crate::common::{FastSet, Result, BLOCK_SIZE};
use crate::exec::expr;
use crate::exec::functions::Accumulator;
use crate::planner::logical::{BoundAgg, BoundExpr};
use crate::types::{Block, Column, ColumnBuilder, DataType, Schema, Value};

use super::sort::spill;
use super::{GroupKey, MemGuard, Operator, QueryContext, ScanStats};

pub struct Aggregate<'a> {
    input: Box<dyn Operator + 'a>,
    group: &'a [BoundExpr],
    aggs: &'a [BoundAgg],
    schema: &'a Schema,
    ctx: &'a QueryContext,
    /// One empty accumulator per aggregate, cloned per group.
    protos: Vec<Box<dyn Accumulator>>,
    out: Vec<Block>,
    /// Spilled partitions still to be folded. Empty for every aggregate that
    /// fits in the budget, which is the only case that costs anything.
    pending: Vec<Partition>,
    ready: bool,
}

impl<'a> Aggregate<'a> {
    pub fn new(
        input: Box<dyn Operator + 'a>,
        group: &'a [BoundExpr],
        aggs: &'a [BoundAgg],
        schema: &'a Schema,
        ctx: &'a QueryContext,
    ) -> Result<Aggregate<'a>> {
        Ok(Aggregate {
            input,
            group,
            aggs,
            schema,
            // Construct the prototypes now so a bad argument type is a
            // plan-time error rather than something discovered halfway
            // through a scan.
            protos: protos(aggs)?,
            ctx,
            out: Vec::new(),
            pending: Vec::new(),
            ready: false,
        })
    }

    fn materialize(&mut self) -> Result<()> {
        self.ready = true;
        let mut guard = MemGuard::new(self.ctx, guard_name(self.group.len()));
        let mut parts = Partitions::new(0);
        // A bare aggregate is one group that exists before the first row and
        // has no key to partition by, so it is the one shape that still fails
        // rather than spills -- and the one shape that cannot grow by
        // grouping in the first place.
        let spill = (!self.group.is_empty()).then_some(&mut parts);
        let groups = accumulate_into(
            &mut self.input,
            self.group,
            self.aggs,
            &self.protos,
            self.ctx,
            &mut guard,
            spill,
        )?;
        self.out = emit(&groups, self.group, self.aggs, self.schema)?;
        // Handed out back to front so `next` can pop instead of cloning; the
        // same change `sort.rs` carries the measurement for.
        self.out.reverse();
        drop(groups);
        drop(guard);
        self.pending = parts.finish()?;
        Ok(())
    }
}

/// One empty accumulator per aggregate, to be cloned per group.
///
/// Fallible because an aggregate validates its argument types here: `avg` over
/// a `String` has to fail before a single row is read.
pub(crate) fn protos(aggs: &[BoundAgg]) -> Result<Vec<Box<dyn Accumulator>>> {
    aggs.iter()
        .map(|a| {
            let tys: Vec<DataType> = a.args.iter().map(|e| e.ty()).collect();
            (a.func.new)(&tys, &a.params)
        })
        .collect()
}

/// What the memory budget calls this operator's unbounded state.
pub(crate) fn guard_name(ngroup: usize) -> &'static str {
    if ngroup == 0 {
        "the aggregate state"
    } else {
        "the GROUP BY hash table"
    }
}

/// Fold every row `input` produces into a group table.
///
/// Split out of the operator so a parallel exchange can call it once per
/// worker; `guard` is the caller's because the budget has to be charged
/// against the whole query, not per worker.
pub(crate) fn accumulate(
    input: &mut Box<dyn Operator + '_>,
    group: &[BoundExpr],
    aggs: &[BoundAgg],
    protos: &[Box<dyn Accumulator>],
    ctx: &QueryContext,
    guard: &mut MemGuard,
) -> Result<Groups> {
    accumulate_into(input, group, aggs, protos, ctx, guard, None)
}

/// [`accumulate`], optionally allowed to spill.
///
/// With `spill` set, a group table that fills the budget **freezes** instead of
/// failing: it stops admitting new keys and every row whose key it does not
/// already hold is written to a partition file keyed by hash. The consequences
/// are what make this correct rather than merely bounded:
///
/// * a group is either entirely in memory or entirely in one partition, never
///   split -- a key first seen *after* the freeze has no rows before it -- so
///   nothing has to be merged back and the partitions' key sets are disjoint
///   from each other and from the table's;
/// * within a partition, rows keep their input order, so `any`, `anyLast`,
///   `argMin`'s tie-break and `groupArray`'s element order come out exactly as
///   a serial pass would produce them;
/// * `DISTINCT` needs no special handling at all, because a seen-set lives
///   with its group and its group did not move.
///
/// The alternative -- evicting the *table* -- would need an
/// `Accumulator::serialize`, which the trait does not have and which a `uniq`
/// HLL or a `quantile` reservoir would make expensive. Spilling raw rows
/// instead costs re-evaluating the group expressions on the way back, which is
/// one pass over the spilled rows and no new API surface.
///
/// The in-memory path pays for none of it. `spill` is `None` for the exchange
/// (whose workers must return a *complete* partial table), the mode is a
/// latch tested once per block, and the freeze itself is the `grow_to` this
/// loop already made -- its `Err` selects the frozen path instead of
/// returning. Measured interleaved against `accumulate` itself, alternating
/// sides in one loop, best-of-7 over 2M rows, two runs: `GROUP BY` over 100k
/// groups 0.984x and 0.993x, over 8 groups 1.006x and 0.999x. Null.
///
/// What spilling costs: `SELECT g, count() ... GROUP BY g` over the same 2M
/// rows (40 B/row, 80 MB) and 100k groups, best-of-3 per budget:
///
/// ```text
///   budget    246M    82M    61M    31M   7.7M   1.9M
///   GROUP BY    69     68     68     68    139    183  ms
///                                     ^ last budget that holds the table
/// ```
///
/// **A 2.0x cliff**, and shallow below it: what is paid is one write and one
/// read of the rows that missed the frozen table, plus re-hashing them on the
/// way back, so the cost tracks how much of the grouping did *not* fit rather
/// than how small the budget is. A partition that still does not fit splits
/// again on fresh bits, and each extra level is the same linear pass over a
/// smaller share -- the 1.9M column is where that starts to show.
pub(crate) fn accumulate_into(
    input: &mut Box<dyn Operator + '_>,
    group: &[BoundExpr],
    aggs: &[BoundAgg],
    protos: &[Box<dyn Accumulator>],
    ctx: &QueryContext,
    guard: &mut MemGuard,
    mut spill: Option<&mut Partitions>,
) -> Result<Groups> {
    let nagg = aggs.len();
    let ngroup = group.len();
    let mut groups = Groups::new(ngroup, aggs);

    // No GROUP BY: exactly one group, and it has to exist even if no row
    // ever arrives.
    if ngroup == 0 {
        groups.add_empty(protos, aggs);
    }

    // Buffers reused across every block. Between them they are the reason
    // a multi-million-row aggregate allocates a bounded amount rather than
    // an amount proportional to rows x groups.
    let mut probe: Vec<Value> = vec![Value::Null; ngroup];
    let mut row_group: Vec<u32> = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    let mut order: Vec<u32> = Vec::new();
    // Group ids this block actually used. Everything downstream of pass 1
    // walks *this* rather than `0..groups.len`, which is what keeps a block
    // costing its own size instead of the size of the whole grouping.
    let mut touched: Vec<u32> = Vec::new();
    // Row ids carrying a first sighting for a DISTINCT aggregate. Hoisted:
    // the obvious `Vec::new()` inside the fold below allocates once per
    // (group, aggregate) per block.
    let mut fresh: Vec<u32> = Vec::new();
    // Frozen-path scratch: the rows of this block that hit a resident group,
    // and the ones that must be spilled. Both stay empty and unallocated for
    // every aggregate that fits in memory.
    let mut hits: Vec<u32> = Vec::new();
    let mut miss: Vec<u32> = Vec::new();
    let mut mpart: Vec<u32> = Vec::new();
    let mut frozen = false;
    let forced = if spill.is_some() { super::sort::forced_spill_rows() } else { 0 };

    loop {
        // Once per block. The group table is the thing that grows without
        // bound here, and a block adds at most BLOCK_SIZE groups to it, so
        // charging after the block bounds the overshoot at one block's
        // worth of groups while keeping the atomic out of the row loop.
        ctx.check()?;
        let Some(b) = input.next()? else { break };
        let rows = b.rows();
        if rows == 0 {
            continue;
        }
        // Borrow rather than clone: a bare `GROUP BY col` or `sum(col)`
        // would otherwise copy the whole column per block.
        let gcols = expr::eval_all_cow(group, &b)?;
        // Aggregate arguments stay owned: `Accumulator::update` takes
        // `&[Column]`, and materializing the borrows per group would clone
        // once per group instead of once per block.
        let acols: Vec<Vec<Column>> = aggs
            .iter()
            .map(|a| expr::eval_all(&a.args, &b))
            .collect::<Result<_>>()?;

        // Pass 1: resolve each row's group. The only per-row hashing, and
        // it probes the table through `probe` without allocating.
        row_group.clear();
        // `None` on the in-memory path, where a row's id *is* its index into
        // `row_group`; `Some(hits)` once frozen, where the rows that missed
        // have been left out. One branch per block, in pass 2.
        let mut ids: Option<&[u32]> = None;
        if frozen {
            // The frozen path skips the string fast path: it is the slow half
            // of a query that has already lost, and the general loop is the
            // one that has a `find` without an insert.
            hits.clear();
            miss.clear();
            mpart.clear();
            for i in 0..rows {
                for (k, c) in gcols.iter().enumerate() {
                    probe[k] = c.as_ref().value(i);
                }
                let h = super::hash_values(&probe);
                match groups.find(&probe, h) {
                    Some(g) => {
                        row_group.push(g as u32);
                        hits.push(i as u32);
                    }
                    None => {
                        miss.push(i as u32);
                        mpart.push(spill.as_deref().expect("frozen").part_of(h));
                    }
                }
            }
            if !miss.is_empty() {
                let p = spill.as_deref_mut().expect("frozen only with a spill target");
                p.push(&b, &miss, &mpart, ctx)?;
            }
            if row_group.is_empty() {
                continue;
            }
            ids = Some(&hits);
        } else {
            row_group.resize(rows, 0);
            // The single-string-key case gets its own loop: it is the most
            // common shape of a GROUP BY, and it is the one where building a
            // `Value` per row actually costs something (an atomic refcount
            // bump). Everything else shares the general path.
            let str_key =
                ngroup == 1 && matches!(gcols[0].as_ref().data, crate::types::ColumnData::Str(_));
            if str_key {
                let col = gcols[0].as_ref();
                let vals = col.as_str()?;
                let null_h = super::hash_null_key();
                for (i, slot) in row_group.iter_mut().enumerate() {
                    *slot = if col.is_null(i) {
                        groups.find_or_insert(&[Value::Null], null_h, protos, aggs)
                    } else {
                        let h = super::hash_str_key(&vals[i]);
                        groups.find_or_insert_str(&vals[i], h, protos, aggs)
                    } as u32;
                }
            } else if ngroup > 0 {
                for (i, slot) in row_group.iter_mut().enumerate() {
                    for (k, c) in gcols.iter().enumerate() {
                        probe[k] = c.as_ref().value(i);
                    }
                    let h = super::hash_values(&probe);
                    *slot = groups.find_or_insert(&probe, h, protos, aggs) as u32;
                }
            }
        }

        // Pass 2: bucket row ids by group with a counting sort into one
        // flat buffer. The obvious `vec![Vec::new(); ngroups]` allocates a
        // vector per group *per block*, which on a high-cardinality
        // grouping is millions of allocations.
        //
        // Everything here is indexed by group id but sized by *block*. The
        // earlier version cleared, prefix-summed and scanned all `ngroups`
        // slots on every block, and cloned an `ngroups`-long cursor to
        // scatter with -- four O(total groups) passes to process at most
        // 8192 rows. On a 4M-group aggregate that is ~15 billion slot
        // visits and 1200 multi-megabyte allocations to do 10M rows of
        // work, which is where the quadratic came from. `counts` now stays
        // zeroed between blocks (each block resets exactly the slots it
        // dirtied), so it is grown once and never re-cleared.
        //
        // Measured interleaved, 100k groups over 2M rows: 284ms -> 140ms,
        // 2.02x, with an 8-group aggregate unchanged at 46ms. The gap
        // widens with group count, because the term removed was
        // O(blocks x groups) against O(blocks x block).
        let ngroups = groups.len;
        if counts.len() < ngroups {
            counts.resize(ngroups, 0);
        }
        touched.clear();
        for &g in &row_group {
            let c = &mut counts[g as usize];
            if *c == 0 {
                touched.push(g);
            }
            *c += 1;
        }
        // Hand each touched group a contiguous span of `order`, in
        // first-touch order. `counts[g]` becomes the write cursor.
        let mut base = 0u32;
        for &g in &touched {
            let n = std::mem::replace(&mut counts[g as usize], base);
            base += n;
        }
        order.clear();
        order.resize(row_group.len(), 0);
        // Two scatters rather than one indirection: `order` has to hold *block*
        // row ids either way, and paying an `ids[j]` load per row on the path
        // that does not need it would tax every aggregate that fits in memory.
        match ids {
            None => {
                for (i, &g) in row_group.iter().enumerate() {
                    let at = &mut counts[g as usize];
                    order[*at as usize] = i as u32;
                    *at += 1;
                }
            }
            Some(ids) => {
                for (j, &g) in row_group.iter().enumerate() {
                    let at = &mut counts[g as usize];
                    order[*at as usize] = ids[j];
                    *at += 1;
                }
            }
        }

        // Pass 3: one vectorized fold per (group, aggregate). After the
        // scatter `counts[g]` sits at the end of g's span, and the spans
        // were laid out in `touched` order -- so walking `touched` yields
        // each (lo, hi) with a running low-water mark, and the same walk
        // restores the slot to zero for the next block.
        let mut lo = 0usize;
        for &gid in &touched {
            let g = gid as usize;
            let hi = std::mem::replace(&mut counts[g], 0) as usize;
            let s = &order[lo..hi];
            lo = hi;
            let base = g * nagg;
            // The DISTINCT test is hoisted to the whole operator: without
            // a DISTINCT aggregate the `seen` arena is not even allocated,
            // so this is one predictable branch per (group, block) rather
            // than an `Option` probe per aggregate.
            if !groups.has_distinct {
                for (ai, args) in acols.iter().enumerate() {
                    groups.accs[base + ai].update(args, s)?;
                }
                continue;
            }
            for (ai, args) in acols.iter().enumerate() {
                match groups.seen[base + ai].as_mut() {
                    Some(set) => {
                        fresh.clear();
                        for &r in s {
                            let t =
                                GroupKey(args.iter().map(|c| c.value(r as usize)).collect());
                            if set.insert(t) {
                                fresh.push(r);
                            }
                        }
                        if !fresh.is_empty() {
                            groups.accs[base + ai].update(args, &fresh)?;
                        }
                    }
                    None => groups.accs[base + ai].update(args, s)?,
                }
            }
        }
        // The freeze, and the only line the in-memory path did not already
        // run: a `grow_to` that succeeds costs exactly what it did before,
        // plus one compare against a knob that is zero outside the tests.
        let over = guard.grow_to(groups.bytes());
        if over.is_err() || (forced != 0 && groups.len >= forced) {
            match spill.as_deref_mut() {
                None => over?,
                Some(p) => {
                    p.arm(ctx, &b);
                    frozen = true;
                }
            }
        }
    }
    Ok(groups)
}

/// Turn a finished group table into `[group..., aggs...]` blocks, in the
/// table's own group order.
pub(crate) fn emit(
    groups: &Groups,
    group: &[BoundExpr],
    aggs: &[BoundAgg],
    schema: &Schema,
) -> Result<Vec<Block>> {
    let (nagg, ngroup) = (aggs.len(), group.len());
    let width = ngroup + nagg;
    if width == 0 {
        return Ok(if groups.len == 0 {
            Vec::new()
        } else {
            vec![Block::rows_only(groups.len)]
        });
    }
    let ty_at = |i: usize| -> DataType {
        if i < schema.len() {
            schema.ty(i).clone()
        } else if i < ngroup {
            group[i].ty()
        } else {
            aggs[i - ngroup].ty.clone()
        }
    };

    let total = groups.len;
    let mut out = Vec::with_capacity(total.div_ceil(BLOCK_SIZE));
    let mut start = 0;
    while start < total {
        let end = (start + BLOCK_SIZE).min(total);
        let mut builders: Vec<ColumnBuilder> = (0..width)
            .map(|i| ColumnBuilder::with_capacity(ty_at(i), end - start))
            .collect();
        for g in start..end {
            for c in 0..ngroup {
                builders[c].push_value(&groups.keys[g * ngroup + c])?;
            }
            for ai in 0..nagg {
                builders[ngroup + ai].push_value(&groups.accs[g * nagg + ai].finish())?;
            }
        }
        // An aggregate over an empty group finishes as NULL (`min` of
        // nothing), so a column can acquire a mask even where the plan's
        // schema said otherwise. A live mask must never sit on a
        // non-Nullable type, so widen when it happens.
        let cols: Vec<Column> = builders
            .into_iter()
            .map(|b| {
                let mut c = b.finish();
                if c.has_nulls() && !c.ty.is_nullable() {
                    c.ty = c.ty.to_nullable();
                }
                c
            })
            .collect();
        out.push(Block::new(cols)?);
        start = end;
    }
    Ok(out)
}

/// The group table: keys in first-seen order, their accumulators, and the
/// per-group DISTINCT seen-sets.
///
/// Open-addressed by hand rather than a `HashMap<Vec<Value>, usize>`, for one
/// reason: a `HashMap` keyed by an owned tuple has to be *probed* with an owned
/// tuple, so resolving a row's group costs a heap allocation **per input row**.
/// Here the keys live in one row-major arena and probing compares against a
/// borrowed slice, so the steady state allocates nothing at all — only a
/// genuinely new group extends the arena.
///
/// Parallel `Vec`s indexed by group ordinal rather than one `Vec` of structs,
/// so the inner loop can hold `&mut accs[g*nagg+ai]` while reading
/// `seen[g*nagg+ai]` without the borrow checker conflating the two.
///
/// Both of those are **flat arenas**, not `Vec<Vec<_>>`. The nested shape cost
/// two heap allocations per group (one for the accumulator vector, one for the
/// seen vector) and a pointer chase per group per block to reach an
/// accumulator, which on a high-cardinality grouping is a cache miss per
/// group per block. Measured interleaved against the nested shape, best-of-9,
/// 100k groups over 2M rows: 100.5ms -> 67.7ms (1.48x) with one aggregate and
/// 532ms -> 295ms (1.80x) with four -- the gap widens with `nagg` because the
/// nested shape chased a pointer per aggregate. An 8-group aggregate is
/// unchanged at 33.4ms, as expected: two allocations saved eight times is
/// nothing.
#[derive(Default)]
pub(crate) struct Groups {
    /// Number of key columns; 0 for a bare aggregate.
    nkeys: usize,
    /// Row-major key arena: group `g` occupies `keys[g*nkeys..(g+1)*nkeys]`.
    keys: Vec<Value>,
    /// Open-addressing slots holding `group + 1`; 0 means empty.
    slots: Vec<u32>,
    /// Cached hash per group, so growing the table never rehashes a key.
    hashes: Vec<u64>,
    len: usize,
    /// Flat: group `g`'s accumulator for aggregate `a` is `accs[g*nagg+a]`.
    accs: Vec<Box<dyn Accumulator>>,
    /// Same indexing, `Some` only for a DISTINCT aggregate. Left empty (and
    /// never grown) when no aggregate is DISTINCT, which is the usual case --
    /// an `Option<FastSet>` is 48 bytes per group per aggregate to store a
    /// `None`.
    seen: Vec<Option<FastSet<GroupKey>>>,
    has_distinct: bool,
}

/// Feed argument tuples straight into an accumulator, bypassing a block.
///
/// The only place in the engine an aggregate is fed from stored `Value`s
/// rather than from decoded columns, and it exists solely for the DISTINCT
/// half of [`Groups::absorb`]. One `ColumnBuilder` per argument for the whole
/// batch, not per tuple: a `Column` per tuple would be three allocations per
/// distinct value, which on `count(DISTINCT x)` over a million values costs
/// more than the aggregation it is finishing.
fn replay(acc: &mut dyn Accumulator, agg: &BoundAgg, tuples: &[GroupKey]) -> Result<()> {
    let n = tuples.len();
    let mut cols: Vec<Column> = Vec::with_capacity(agg.args.len());
    for (j, a) in agg.args.iter().enumerate() {
        let mut b = ColumnBuilder::with_capacity(a.ty(), n);
        for t in tuples {
            b.push_value(&t.0[j])?;
        }
        let mut c = b.finish();
        // Same widening as `emit`: a NULL argument tuple is a legal member of
        // a seen-set, and a live mask must never sit on a non-Nullable type.
        if c.has_nulls() && !c.ty.is_nullable() {
            c.ty = c.ty.to_nullable();
        }
        cols.push(c);
    }
    let sel: Vec<u32> = (0..n as u32).collect();
    acc.update(&cols, &sel)
}

/// Charged per accumulator on top of its `Box`.
///
/// The `Accumulator` trait does not report its own footprint, so this is a
/// flat estimate: about right for `sum`/`count`/`min`/`max`, an undercount for
/// the stateful ones (`uniq`'s HLL, `quantile`'s reservoir). Making it exact
/// needs an `Accumulator::heap_bytes()` in `exec::functions`; until then a
/// `uniq` over millions of groups is accounted low, which is worth knowing
/// before trusting the budget on that shape of query.
const ACC_BYTES: usize = 48;

impl Groups {
    fn new(nkeys: usize, aggs: &[BoundAgg]) -> Groups {
        Groups {
            nkeys,
            has_distinct: aggs.iter().any(|a| a.distinct),
            slots: vec![0; 64],
            ..Default::default()
        }
    }

    /// Fold a partial table computed over a **later** slice of the same input.
    ///
    /// "Later" is load-bearing, not decoration: `any`, `anyLast`, `argMin`'s
    /// tie-breaking and `groupArray`'s element order are all defined against
    /// feed order, and [`Accumulator::merge`] reads `other` as the side that
    /// came second. The exchange keeps that true by giving worker `k` a
    /// contiguous ascending granule range and absorbing in worker order, which
    /// is why a parallel `GROUP BY` here still returns groups in first-seen
    /// order and still gives `any(x)` the answer a serial scan would.
    ///
    /// DISTINCT is the one aggregate shape that cannot go through `merge`:
    /// each side has already deduplicated *within itself*, so merging
    /// `count(DISTINCT x)` partials of `{1,2}` and `{2,3}` would answer 4. The
    /// seen-sets are unioned instead and only the tuples genuinely new to this
    /// side are fed to the accumulator -- batched into one `update` per
    /// (group, aggregate) rather than one per tuple.
    ///
    /// `other` is consumed rather than borrowed so that a group this side has
    /// never seen can be **adopted**: its accumulators and seen-sets move
    /// across, instead of a fresh clone of the prototype being merged into.
    /// That is what a high-cardinality `GROUP BY` is made of -- when the keys
    /// outnumber the rows per worker, almost every group belongs to exactly
    /// one worker, and cloning-then-merging pays a heap allocation and a
    /// virtual call per group per aggregate to reconstruct state that is
    /// already sitting right there.
    ///
    /// Hence the two passes. Adoption cannot simply pull from an iterator in
    /// the merge loop, because *taking* a group's accumulator also drops it a
    /// moment later when the group turns out to already exist, and 1.3M frees
    /// interleaved with hash probes cost far more than the same 1.3M frees
    /// walked sequentially. So pass one merges out of a borrowed slice and
    /// only records which groups are new; pass two walks the arenas once,
    /// moving those groups' state across and letting the rest drop in order.
    /// Measured interleaved with an `AtomicBool` selecting the clone-and-merge
    /// shape, alternating sides, best-of-12, 10M rows on 14 cores:
    ///
    /// ```text
    ///   GROUP BY country (8 groups)      23.97 -> 23.9 ms   1.00x  (control)
    ///   GROUP BY big     (100k groups)  269.9  -> 268   ms  1.01x
    ///   GROUP BY id      (10M groups)  1515    -> 1010  ms  1.50x
    /// ```
    ///
    /// The single-pass version that took every accumulator through the
    /// iterator measured 0.75x on `GROUP BY big` for the same 1.50x on
    /// `GROUP BY id`; it is not worth re-trying.
    ///
    /// Groups must be visited in ascending order, in both passes: a group
    /// occupies exactly `nagg` contiguous entries of each arena, and the
    /// adopted ones are handed the next `gid`s in the order pass one saw them.
    pub(crate) fn absorb(&mut self, other: Groups, aggs: &[BoundAgg]) -> Result<()> {
        let nagg = aggs.len();
        let Groups { nkeys, keys, hashes, len, accs, mut seen, has_distinct, .. } = other;
        debug_assert_eq!(nkeys, self.nkeys);
        debug_assert_eq!(has_distinct, self.has_distinct);
        // Reused across every group: the argument tuples new to this side.
        let mut fresh: Vec<GroupKey> = Vec::new();
        // Groups of `other` this side has never seen, ascending.
        let mut adopt: Vec<u32> = Vec::new();

        for g in 0..len {
            // The key arena is sliced, never drained: `reserve` clones the key
            // when the group is new, so moving it out first would buy a `Value`
            // move and a drop per key for nothing.
            let key = &keys[g * nkeys..(g + 1) * nkeys];
            let (gid, is_new) = self.reserve(key, hashes[g]);
            if is_new {
                adopt.push(g as u32);
                continue;
            }
            let (base, src) = (gid * nagg, g * nagg);
            for ai in 0..nagg {
                let Some(theirs) = seen.get_mut(src + ai).and_then(|s| s.as_mut()) else {
                    // Not a DISTINCT aggregate: the accumulators compose.
                    self.accs[base + ai].merge(&*accs[src + ai])?;
                    continue;
                };
                let mine = self.seen[base + ai].as_mut().expect("DISTINCT-ness is per aggregate");
                if mine.is_empty() {
                    // This side saw the group but fed this aggregate nothing,
                    // so an empty accumulator merged with theirs *is* theirs,
                    // and the seen-set can be taken whole.
                    self.accs[base + ai].merge(&*accs[src + ai])?;
                    std::mem::swap(mine, theirs);
                    continue;
                }
                fresh.clear();
                fresh.extend(theirs.drain().filter(|t| !mine.contains(t)));
                if !fresh.is_empty() {
                    replay(&mut *self.accs[base + ai], &aggs[ai], &fresh)?;
                    for t in fresh.drain(..) {
                        mine.insert(t);
                    }
                }
            }
        }

        // Pass two. `reserve` already published the adopted groups' keys and
        // hashes, so `self.len` currently runs ahead of the accumulator arena;
        // this closes the gap. (An error out of pass one leaves it open, which
        // is sound only because the caller drops the whole table on that path
        // -- nothing indexes `accs` by group except `emit`, which is never
        // reached.)
        if adopt.is_empty() {
            return Ok(());
        }
        let mut accs = accs.into_iter();
        let mut seen = seen.into_iter();
        let mut cur = 0usize;
        for g in adopt {
            let src = g as usize * nagg;
            // Skip -- and thereby drop -- the groups that merged in place.
            // Sequential, so the frees keep the locality of the bulk drop the
            // borrowing pass above depends on.
            for _ in cur..src {
                accs.next();
                seen.next();
            }
            for _ in 0..nagg {
                self.accs.push(accs.next().expect("nagg accumulators per group"));
                if has_distinct {
                    self.seen.push(seen.next().flatten());
                }
            }
            cur = src + nagg;
        }
        Ok(())
    }

    /// What this table is holding, for the memory budget.
    ///
    /// Capacities rather than lengths: the doubling is the memory that is
    /// actually resident. String group keys are undercounted by the length of
    /// the string, since a `Value::Str` may share its `Arc` with the input
    /// block that produced it.
    pub(crate) fn bytes(&self) -> usize {
        self.keys.capacity() * size_of::<Value>()
            + self.slots.capacity() * size_of::<u32>()
            + self.hashes.capacity() * size_of::<u64>()
            + self.accs.capacity() * (size_of::<Box<dyn Accumulator>>() + ACC_BYTES)
            + self.seen.capacity() * size_of::<Option<FastSet<GroupKey>>>()
    }

    /// Everything a newly created group needs, in one place: the two arenas
    /// have to stay exactly `nagg` entries per group or every later index is
    /// off by an aggregate.
    #[inline]
    fn push_group(&mut self, protos: &[Box<dyn Accumulator>], aggs: &[BoundAgg]) {
        self.accs.extend(protos.iter().map(|p| p.boxed_clone()));
        if self.has_distinct {
            self.seen
                .extend(aggs.iter().map(|a| a.distinct.then(FastSet::default)));
        }
    }

    #[inline(always)]
    fn key_of(&self, g: usize) -> &[Value] {
        &self.keys[g * self.nkeys..(g + 1) * self.nkeys]
    }

    /// Resolve `key` to a group, creating it if new.
    ///
    /// `key` is borrowed from the caller's reusable scratch buffer, which is
    /// what keeps the common path allocation-free.
    #[inline]
    fn find_or_insert(
        &mut self,
        key: &[Value],
        h: u64,
        protos: &[Box<dyn Accumulator>],
        aggs: &[BoundAgg],
    ) -> usize {
        let (g, is_new) = self.reserve(key, h);
        if is_new {
            self.push_group(protos, aggs);
        }
        g
    }

    /// Resolve `key` to an existing group, or `None`. The probe half of
    /// [`Groups::find_or_insert`] without the insert, for a table that has
    /// been frozen because it filled the budget: rows that miss are spilled
    /// rather than growing it further. See [`accumulate_into`].
    #[inline]
    fn find(&self, key: &[Value], h: u64) -> Option<usize> {
        let mask = self.slots.len() - 1;
        let mut i = h as usize & mask;
        loop {
            let s = self.slots[i];
            if s == 0 {
                return None;
            }
            let g = s as usize - 1;
            if self.hashes[g] == h && self.key_of(g) == key {
                return Some(g);
            }
            i = (i + 1) & mask;
        }
    }

    /// `find_or_insert` stopping short of creating the accumulators: the
    /// group's slot, its key and its hash exist, its `nagg` arena entries do
    /// not. `true` means the caller now owes them.
    ///
    /// Split out for [`absorb`](Self::absorb), which supplies the other side's
    /// accumulators rather than fresh ones. Inlined back into `find_or_insert`
    /// for the per-row path, which is the hottest loop in a `GROUP BY`:
    /// measured interleaved before and after the split, 2M rows over 100k
    /// groups, 67.9 vs 67.4 ms -- the same code after inlining, as intended.
    #[inline]
    fn reserve(&mut self, key: &[Value], h: u64) -> (usize, bool) {
        // Keep the load factor under 1/2: past that, linear probing degrades
        // sharply and the extra memory is cheaper than the probe chains.
        if (self.len + 1) * 2 >= self.slots.len() {
            self.grow();
        }
        let mask = self.slots.len() - 1;
        let mut i = h as usize & mask;
        loop {
            let s = self.slots[i];
            if s == 0 {
                let g = self.len;
                self.slots[i] = g as u32 + 1;
                self.keys.extend_from_slice(key);
                self.hashes.push(h);
                self.len += 1;
                return (g, true);
            }
            let g = s as usize - 1;
            // Compare the cached hash first: a mismatch rules the group out
            // without touching the arena, which is the expensive part.
            if self.hashes[g] == h && self.key_of(g) == key {
                return (g, false);
            }
            i = (i + 1) & mask;
        }
    }

    /// Single string key, probed from a borrowed `&str`.
    ///
    /// Same algorithm as `find_or_insert`, but the owned `Value` (and its
    /// `Arc` bump) is built only when a genuinely new group is created --
    /// once per group instead of once per row.
    fn find_or_insert_str(
        &mut self,
        key: &std::sync::Arc<str>,
        h: u64,
        protos: &[Box<dyn Accumulator>],
        aggs: &[BoundAgg],
    ) -> usize {
        debug_assert_eq!(self.nkeys, 1);
        if (self.len + 1) * 2 >= self.slots.len() {
            self.grow();
        }
        let mask = self.slots.len() - 1;
        let mut i = h as usize & mask;
        loop {
            let s = self.slots[i];
            if s == 0 {
                let g = self.len;
                self.slots[i] = g as u32 + 1;
                self.keys.push(Value::Str(key.clone()));
                self.hashes.push(h);
                self.len += 1;
                self.push_group(protos, aggs);
                return g;
            }
            let g = s as usize - 1;
            if self.hashes[g] == h && self.keys[g].as_str() == Some(&**key) {
                return g;
            }
            i = (i + 1) & mask;
        }
    }

    fn grow(&mut self) {
        let cap = (self.slots.len() * 2).max(64);
        let mask = cap - 1;
        let mut slots = vec![0u32; cap];
        for g in 0..self.len {
            let mut i = self.hashes[g] as usize & mask;
            while slots[i] != 0 {
                i = (i + 1) & mask;
            }
            slots[i] = g as u32 + 1;
        }
        self.slots = slots;
    }

    /// Create the single group of a bare aggregate.
    fn add_empty(&mut self, protos: &[Box<dyn Accumulator>], aggs: &[BoundAgg]) -> usize {
        self.find_or_insert(&[], 0, protos, aggs)
    }
}

// ------------------------------------------------------------- spilled rows

/// The partition files a frozen group table is writing its unseen keys to.
///
/// One open [`spill::RunWriter`] per partition, created on first use, so a
/// grouping whose overflow all lands in three buckets does not open sixty-four
/// files. The partition count and the per-partition write buffer both come out
/// of the query's own budget rather than a constant: sixty-four 64 KiB buffers
/// is 4 MiB of write-behind, which is free against 8 GiB and absurd against
/// 256 KiB.
pub(crate) struct Partitions {
    dir: Option<spill::SpillDir>,
    writers: Vec<Option<spill::RunWriter>>,
    /// Recursion depth. Also the mixer seed, which is what stops a partition
    /// that is *still* too big from re-splitting into one bucket.
    level: u32,
    mask: u64,
    flush_at: usize,
    schema: Option<Schema>,
    /// Counting-sort scratch, reused across blocks: spans, write cursors and
    /// the bucketed row ids. Three fields rather than three `Vec::new()`s in
    /// `push`, which would allocate once per block per spilling aggregate.
    counts: Vec<u32>,
    cursor: Vec<u32>,
    sel: Vec<u32>,
}

/// One spilled partition, and its share of the directory's lifetime.
pub(crate) struct Partition {
    path: std::path::PathBuf,
    schema: Schema,
    level: u32,
    /// Shared, because a directory holds every partition cut at one level and
    /// must outlive the last of them -- however early the query stops reading.
    _dir: std::sync::Arc<spill::SpillDir>,
}

impl Partitions {
    pub(crate) fn new(level: u32) -> Partitions {
        Partitions {
            dir: None,
            writers: Vec::new(),
            level,
            mask: 0,
            flush_at: 0,
            schema: None,
            counts: Vec::new(),
            cursor: Vec::new(),
            sel: Vec::new(),
        }
    }

    /// Fix the partition count and buffer size. Called once, at the freeze.
    fn arm(&mut self, ctx: &QueryContext, b: &Block) {
        if self.mask != 0 {
            return;
        }
        // A quarter of the budget for write-behind; the group table -- which
        // has just proved it wants everything -- keeps the rest.
        let cap = ((ctx.mem.limit().max(0) as usize) / 4).max(8 << 10);
        let want = (cap / (32 << 10)).clamp(2, 64).next_power_of_two().min(64);
        // Never more partitions than the block that triggered the freeze has
        // rows: a hundred one-row files cost more in syscalls than they save.
        let n = want.min(b.rows().max(2).next_power_of_two());
        self.mask = n as u64 - 1;
        self.flush_at = (cap / n).clamp(4 << 10, 64 << 10);
        self.writers = (0..n).map(|_| None).collect();
    }

    /// Which partition a group key with hash `h` belongs to.
    ///
    /// Re-mixed with the level rather than shifted by it: shifting runs out of
    /// hash bits after `64/log2(n)` levels and then stops partitioning at all,
    /// which turns a deep recursion into an infinite one.
    #[inline]
    pub(crate) fn part_of(&self, h: u64) -> u32 {
        let seed = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(self.level as u64 + 1);
        (crate::common::mum(h ^ seed, 0xD6E8_FEB8_6659_FD93) & self.mask) as u32
    }

    /// Spill `rows` of `b`, each already assigned a partition in `parts`.
    fn push(&mut self, b: &Block, rows: &[u32], parts: &[u32], ctx: &QueryContext) -> Result<()> {
        // A cancelled query that keeps writing gigabytes to disk is worse than
        // one that keeps burning CPU.
        ctx.check()?;
        if self.schema.is_none() {
            self.schema = Some(spill::schema_of(b));
        }
        // Counting sort into one buffer, the same shape pass 2 above uses and
        // for the same reason: a `Vec` per partition per block would allocate
        // once per bucket per 8192 rows.
        let n = self.writers.len();
        self.counts.clear();
        self.counts.resize(n + 1, 0);
        for &p in parts {
            self.counts[p as usize + 1] += 1;
        }
        for i in 0..n {
            self.counts[i + 1] += self.counts[i];
        }
        self.sel.clear();
        self.sel.resize(rows.len(), 0);
        self.cursor.clear();
        self.cursor.extend_from_slice(&self.counts[..n]);
        for (&r, &p) in rows.iter().zip(parts) {
            let c = &mut self.cursor[p as usize];
            self.sel[*c as usize] = r;
            *c += 1;
        }
        for p in 0..n {
            let (lo, hi) = (self.counts[p] as usize, self.counts[p + 1] as usize);
            if lo == hi {
                continue;
            }
            let sub = b.take(&self.sel[lo..hi]);
            self.writer(p)?.push(&sub)?;
        }
        Ok(())
    }

    fn writer(&mut self, p: usize) -> Result<&mut spill::RunWriter> {
        if self.writers[p].is_none() {
            let dir = match &mut self.dir {
                Some(d) => d,
                None => self.dir.insert(spill::SpillDir::new()?),
            };
            self.writers[p] = Some(dir.create_buffered(self.flush_at)?);
        }
        Ok(self.writers[p].as_mut().expect("just created"))
    }

    /// Close every open partition. Empty when nothing spilled, which is the
    /// answer for every aggregate that fit in memory.
    fn finish(self) -> Result<Vec<Partition>> {
        let Partitions { dir, writers, level, schema, .. } = self;
        let (Some(dir), Some(schema)) = (dir, schema) else { return Ok(Vec::new()) };
        let mut paths = Vec::new();
        for w in writers.into_iter().flatten() {
            paths.push(w.finish()?);
        }
        let dir = std::sync::Arc::new(dir);
        Ok(paths
            .into_iter()
            .map(|path| Partition {
                path,
                schema: schema.clone(),
                level: level + 1,
                _dir: dir.clone(),
            })
            .collect())
    }
}

/// Reads one spilled partition back as an operator, so the recursive pass is
/// the *same* `accumulate_into` and not a second implementation of it.
struct SpillScan {
    src: spill::RunReader,
    schema: Schema,
}

impl SpillScan {
    fn open(p: &Partition) -> Result<SpillScan> {
        Ok(SpillScan {
            src: spill::RunReader::open(&p.path, p.schema.clone())?,
            schema: p.schema.clone(),
        })
    }
}

impl Operator for SpillScan {
    fn schema(&self) -> &Schema {
        &self.schema
    }
    fn next(&mut self) -> Result<Option<Block>> {
        self.src.next()
    }
}

impl Aggregate<'_> {
    /// Fold one spilled partition into `out`; `false` when none are left.
    ///
    /// The partition's keys are disjoint from every key already emitted, so
    /// this appends rather than merges. It may spill again -- a partition that
    /// still does not fit re-partitions on fresh hash bits -- and each level
    /// removes at least the groups that did fit, so it terminates.
    fn next_partition(&mut self) -> Result<bool> {
        let Some(p) = self.pending.pop() else { return Ok(false) };
        self.ctx.check()?;
        let mut guard = MemGuard::new(self.ctx, guard_name(self.group.len()));
        let mut input: Box<dyn Operator> = Box::new(SpillScan::open(&p)?);
        let mut parts = Partitions::new(p.level);
        let groups = accumulate_into(
            &mut input,
            self.group,
            self.aggs,
            &self.protos,
            self.ctx,
            &mut guard,
            Some(&mut parts),
        )?;
        self.out = emit(&groups, self.group, self.aggs, self.schema)?;
        self.out.reverse();
        drop(groups);
        drop(guard);
        // Unlinked as soon as it has been folded rather than with the whole
        // directory, so peak disk tracks what is still owed and not what the
        // query has ever spilled.
        drop(input);
        let _ = std::fs::remove_file(&p.path);
        self.pending.extend(parts.finish()?);
        Ok(true)
    }
}

impl Operator for Aggregate<'_> {
    fn schema(&self) -> &Schema {
        self.schema
    }

    fn stats(&self) -> ScanStats {
        self.input.stats()
    }

    fn next(&mut self) -> Result<Option<Block>> {
        if !self.ready {
            self.materialize()?;
        }
        loop {
            if let Some(b) = self.out.pop() {
                return Ok(Some(b));
            }
            // One call at end of stream, and an immediate `false` for every
            // aggregate that fit in memory.
            if !self.next_partition()? {
                return Ok(None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::functions;
    use crate::exec::operators::values::Values;
    use crate::types::{Field, Value};

    fn agg(name: &str, args: Vec<BoundExpr>, distinct: bool) -> BoundAgg {
        let f = functions::aggregate(name).unwrap();
        let tys: Vec<DataType> = args.iter().map(|a| a.ty()).collect();
        BoundAgg {
            func: f,
            ty: (f.ret)(&tys, &[]).unwrap(),
            args,
            params: vec![],
            distinct,
            name: name.into(),
        }
    }

    fn col(i: usize, ty: DataType) -> BoundExpr {
        BoundExpr::Column { index: i, ty, name: format!("c{i}") }
    }

    /// `(k Int64, v Int64)` rows.
    fn src() -> (Schema, Vec<Vec<Value>>) {
        let s = Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("v", DataType::Int64),
        ])
        .unwrap();
        let rows = vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(2), Value::Int(20)],
            vec![Value::Int(1), Value::Int(30)],
            vec![Value::Int(2), Value::Int(20)],
            vec![Value::Int(1), Value::Int(10)],
        ];
        (s, rows)
    }

    fn out_schema(fields: Vec<(&str, DataType)>) -> Schema {
        Schema::new_unchecked(fields.into_iter().map(|(n, t)| Field::new(n, t)).collect())
    }

    /// Run an aggregate and return rows as `Vec<Value>`, sorted for stability.
    fn run(
        rows: &[Vec<Value>],
        in_schema: &Schema,
        group: &[BoundExpr],
        aggs: &[BoundAgg],
        out: &Schema,
    ) -> Vec<Vec<Value>> {
        let ctx = QueryContext::new();
        let mut a =
            Aggregate::new(Box::new(Values::new(rows, in_schema)), group, aggs, out, &ctx).unwrap();
        let mut got = Vec::new();
        while let Some(b) = a.next().unwrap() {
            for i in 0..b.rows() {
                got.push((0..b.width()).map(|c| b.column(c).value(i)).collect::<Vec<_>>());
            }
        }
        got.sort();
        got
    }

    #[test]
    fn group_by_sums_per_group() {
        let (s, rows) = src();
        let group = vec![col(0, DataType::Int64)];
        let aggs = vec![agg("sum", vec![col(1, DataType::Int64)], false)];
        let out = out_schema(vec![("k", DataType::Int64), ("s", aggs[0].ty.clone())]);
        assert_eq!(
            run(&rows, &s, &group, &aggs, &out),
            vec![
                vec![Value::Int(1), Value::Int(50)],
                vec![Value::Int(2), Value::Int(40)],
            ]
        );
    }

    #[test]
    fn several_aggregates_share_one_pass() {
        let (s, rows) = src();
        let group = vec![col(0, DataType::Int64)];
        let aggs = vec![
            agg("count", vec![], false),
            agg("min", vec![col(1, DataType::Int64)], false),
            agg("max", vec![col(1, DataType::Int64)], false),
        ];
        let out = out_schema(vec![
            ("k", DataType::Int64),
            ("n", aggs[0].ty.clone()),
            ("lo", aggs[1].ty.clone()),
            ("hi", aggs[2].ty.clone()),
        ]);
        let got = run(&rows, &s, &group, &aggs, &out);
        assert_eq!(got[0][0], Value::Int(1));
        assert_eq!(got[0][1], Value::UInt(3));
        assert_eq!(got[0][2], Value::Int(10));
        assert_eq!(got[0][3], Value::Int(30));
        assert_eq!(got[1][1], Value::UInt(2));
    }

    #[test]
    fn no_group_by_folds_the_whole_relation() {
        let (s, rows) = src();
        let aggs = vec![agg("sum", vec![col(1, DataType::Int64)], false)];
        let out = out_schema(vec![("s", aggs[0].ty.clone())]);
        assert_eq!(run(&rows, &s, &[], &aggs, &out), vec![vec![Value::Int(90)]]);
    }

    #[test]
    fn count_over_an_empty_relation_is_zero_not_no_rows() {
        let (s, _) = src();
        let aggs = vec![agg("count", vec![], false)];
        let out = out_schema(vec![("n", aggs[0].ty.clone())]);
        let got = run(&[], &s, &[], &aggs, &out);
        assert_eq!(got, vec![vec![Value::UInt(0)]], "a fold always has a result");
    }

    #[test]
    fn group_by_over_an_empty_relation_yields_no_rows() {
        let (s, _) = src();
        let group = vec![col(0, DataType::Int64)];
        let aggs = vec![agg("count", vec![], false)];
        let out = out_schema(vec![("k", DataType::Int64), ("n", aggs[0].ty.clone())]);
        assert!(run(&[], &s, &group, &aggs, &out).is_empty());
    }

    #[test]
    fn distinct_aggregate_dedups_per_group() {
        let (s, rows) = src();
        let group = vec![col(0, DataType::Int64)];
        let aggs = vec![agg("sum", vec![col(1, DataType::Int64)], true)];
        let out = out_schema(vec![("k", DataType::Int64), ("s", aggs[0].ty.clone())]);
        // group 1 has v = 10, 30, 10 -> distinct {10, 30} = 40
        // group 2 has v = 20, 20     -> distinct {20}     = 20
        assert_eq!(
            run(&rows, &s, &group, &aggs, &out),
            vec![
                vec![Value::Int(1), Value::Int(40)],
                vec![Value::Int(2), Value::Int(20)],
            ]
        );
    }

    #[test]
    fn distinct_dedups_across_batches_not_just_within_one() {
        use crate::common::BLOCK_SIZE;
        let s = Schema::new(vec![Field::new("v", DataType::Int64)]).unwrap();
        // Same 5 values repeated over more than one batch.
        let rows: Vec<Vec<Value>> = (0..BLOCK_SIZE as i64 * 2 + 7)
            .map(|i| vec![Value::Int(i % 5)])
            .collect();
        let aggs = vec![agg("count", vec![col(0, DataType::Int64)], true)];
        let out = out_schema(vec![("n", aggs[0].ty.clone())]);
        assert_eq!(run(&rows, &s, &[], &aggs, &out), vec![vec![Value::UInt(5)]]);
    }

    #[test]
    fn distinct_and_non_distinct_of_the_same_column_coexist() {
        let (s, rows) = src();
        let aggs = vec![
            agg("count", vec![col(1, DataType::Int64)], false),
            agg("count", vec![col(1, DataType::Int64)], true),
        ];
        let out = out_schema(vec![("n", aggs[0].ty.clone()), ("d", aggs[1].ty.clone())]);
        // v = 10, 20, 30, 20, 10 -> 5 total, 3 distinct
        assert_eq!(
            run(&rows, &s, &[], &aggs, &out),
            vec![vec![Value::UInt(5), Value::UInt(3)]]
        );
    }

    #[test]
    fn nulls_are_skipped_by_the_accumulator_but_form_their_own_group() {
        let s = Schema::new(vec![
            Field::new("k", DataType::Nullable(Box::new(DataType::Int64))),
            Field::new("v", DataType::Nullable(Box::new(DataType::Int64))),
        ])
        .unwrap();
        let rows = vec![
            vec![Value::Null, Value::Int(1)],
            vec![Value::Int(1), Value::Null],
            vec![Value::Null, Value::Int(2)],
            vec![Value::Int(1), Value::Int(5)],
        ];
        let group = vec![col(0, s.ty(0).clone())];
        let aggs = vec![
            agg("count", vec![col(1, s.ty(1).clone())], false),
            agg("sum", vec![col(1, s.ty(1).clone())], false),
        ];
        let out = out_schema(vec![
            ("k", s.ty(0).clone()),
            ("n", aggs[0].ty.clone()),
            ("s", aggs[1].ty.clone()),
        ]);
        let got = run(&rows, &s, &group, &aggs, &out);
        // sorted: NULL group first (Value::Null ranks lowest)
        assert_eq!(got.len(), 2);
        assert_eq!(got[0][0], Value::Null);
        assert_eq!(got[0][1], Value::UInt(2));
        assert_eq!(got[0][2], Value::Int(3));
        assert_eq!(got[1][0], Value::Int(1));
        assert_eq!(got[1][1], Value::UInt(1), "the NULL v is not counted");
        assert_eq!(got[1][2], Value::Int(5));
    }

    /// Feeds pre-built blocks, so a test can hand `Aggregate` two batches
    /// whose column types differ -- the shape a `GROUP BY` over a `UNION ALL`
    /// of a `Date` branch and a `UInt64` branch produces (`union::coerce`
    /// leaves both alone, they are physically identical).
    struct Blocks {
        schema: Schema,
        blocks: Vec<Block>,
        pos: usize,
    }
    impl Operator for Blocks {
        fn schema(&self) -> &Schema {
            &self.schema
        }
        fn next(&mut self) -> Result<Option<Block>> {
            if self.pos >= self.blocks.len() {
                return Ok(None);
            }
            self.pos += 1;
            Ok(Some(self.blocks[self.pos - 1].clone()))
        }
    }

    #[test]
    fn probe_group_by_merges_keys_that_compare_equal() {
        let s = Schema::new(vec![Field::new("k", DataType::Date)]).unwrap();
        let dates = Block::new(vec![Column::u64s(DataType::Date, vec![5, 5])]).unwrap();
        let uints = Block::new(vec![Column::u64s(DataType::UInt64, vec![5])]).unwrap();
        assert_eq!(
            dates.column(0).value(0),
            uints.column(0).value(0),
            "the two keys compare equal"
        );
        let group = vec![col(0, DataType::Date)];
        let aggs = vec![agg("count", vec![], false)];
        let out = out_schema(vec![("k", DataType::Date), ("n", aggs[0].ty.clone())]);
        let src = Blocks { schema: s, blocks: vec![dates, uints], pos: 0 };
        let ctx = QueryContext::new();
        let mut a = Aggregate::new(Box::new(src), &group, &aggs, &out, &ctx).unwrap();
        let mut got = Vec::new();
        while let Some(b) = a.next().unwrap() {
            for i in 0..b.rows() {
                got.push((0..b.width()).map(|c| b.column(c).value(i)).collect::<Vec<_>>());
            }
        }
        assert_eq!(got.len(), 1, "equal group keys landed in different groups: {got:?}");
    }

    #[test]
    fn composite_group_keys() {
        let s = Schema::new(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::String),
        ])
        .unwrap();
        let rows = vec![
            vec![Value::Int(1), Value::str("x")],
            vec![Value::Int(1), Value::str("y")],
            vec![Value::Int(2), Value::str("x")],
            vec![Value::Int(1), Value::str("x")],
        ];
        let group = vec![col(0, DataType::Int64), col(1, DataType::String)];
        let aggs = vec![agg("count", vec![], false)];
        let out = out_schema(vec![
            ("a", DataType::Int64),
            ("b", DataType::String),
            ("n", aggs[0].ty.clone()),
        ]);
        let got = run(&rows, &s, &group, &aggs, &out);
        assert_eq!(got.len(), 3);
        assert_eq!(got[0], vec![Value::Int(1), Value::str("x"), Value::UInt(2)]);
    }

    #[test]
    fn grouping_on_an_expression() {
        use crate::sql::ast::BinaryOp;
        let s = Schema::new(vec![Field::new("v", DataType::Int64)]).unwrap();
        let rows: Vec<Vec<Value>> = (0..10i64).map(|i| vec![Value::Int(i)]).collect();
        // GROUP BY v % 3
        let group = vec![BoundExpr::Binary {
            left: Box::new(col(0, DataType::Int64)),
            op: BinaryOp::Modulo,
            right: Box::new(BoundExpr::lit(Value::Int(3))),
            ty: DataType::Int64,
        }];
        let aggs = vec![agg("count", vec![], false)];
        let out = out_schema(vec![
            ("g", DataType::Nullable(Box::new(DataType::Int64))),
            ("n", aggs[0].ty.clone()),
        ]);
        let got = run(&rows, &s, &group, &aggs, &out);
        assert_eq!(got.len(), 3);
        let total: u64 = got.iter().map(|r| r[1].as_u64().unwrap()).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn many_groups_spill_into_several_output_blocks() {
        use crate::common::BLOCK_SIZE;
        let s = Schema::new(vec![Field::new("k", DataType::Int64)]).unwrap();
        let n = BLOCK_SIZE as i64 + 25;
        let rows: Vec<Vec<Value>> = (0..n).map(|i| vec![Value::Int(i)]).collect();
        let group = vec![col(0, DataType::Int64)];
        let aggs = vec![agg("count", vec![], false)];
        let out = out_schema(vec![("k", DataType::Int64), ("n", aggs[0].ty.clone())]);
        let ctx = QueryContext::new();
        let mut a =
            Aggregate::new(Box::new(Values::new(&rows, &s)), &group, &aggs, &out, &ctx).unwrap();
        let mut sizes = Vec::new();
        while let Some(b) = a.next().unwrap() {
            sizes.push(b.rows());
        }
        assert_eq!(sizes, vec![BLOCK_SIZE, 25]);
    }

    #[test]
    fn float_aggregates_use_compensated_summation() {
        let s = Schema::new(vec![Field::new("f", DataType::Float64)]).unwrap();
        let mut rows = vec![vec![Value::Float(1e16)]];
        rows.extend((0..1000).map(|_| vec![Value::Float(1.0)]));
        let aggs = vec![agg("sum", vec![col(0, DataType::Float64)], false)];
        let out = out_schema(vec![("s", aggs[0].ty.clone())]);
        let got = run(&rows, &s, &[], &aggs, &out);
        assert_eq!(got[0][0], Value::Float(1e16 + 1000.0));
    }

    #[test]
    fn a_bad_argument_type_fails_at_build_time() {
        let s = Schema::new(vec![Field::new("s", DataType::String)]).unwrap();
        let rows: Vec<Vec<Value>> = vec![];
        let aggs = vec![BoundAgg {
            func: functions::aggregate("avg").unwrap(),
            args: vec![col(0, DataType::String)],
            params: vec![],
            distinct: false,
            ty: DataType::Float64,
            name: "avg".into(),
        }];
        let out = out_schema(vec![("a", DataType::Float64)]);
        let ctx = QueryContext::new();
        assert!(Aggregate::new(Box::new(Values::new(&rows, &s)), &[], &aggs, &out, &ctx).is_err());
    }

    // ------------------------------------------------ budget & cancellation

    /// `(k Int64)` with `n` distinct keys, spread over several blocks.
    fn wide_rows(n: i64) -> (Schema, Vec<Vec<Value>>) {
        let s = Schema::new(vec![Field::new("k", DataType::Int64)]).unwrap();
        (s, (0..n).map(|i| vec![Value::Int(i)]).collect())
    }

    #[test]
    fn a_group_table_that_outgrows_the_budget_spills_instead_of_dying() {
        // Inverted from `..._is_an_error_not_a_death`, which pinned the step
        // before this one: the budget turned a dead process into an error.
        // Now the same query returns the same rows it returns with room to
        // breathe -- the budget bounds *memory*, not what is answerable.
        let (s, rows) = wide_rows(200_000);
        let group = vec![col(0, DataType::Int64)];
        let aggs = vec![agg("count", vec![], false)];
        let out = out_schema(vec![("k", DataType::Int64), ("n", aggs[0].ty.clone())]);

        // 200k groups at ~96 B each is ~19 MB, so 256 KiB and 4 MiB both have
        // to spill and 64 MiB has to not. All three must agree row for row.
        let mut answers = Vec::new();
        for budget in [256i64 << 10, 4 << 20, 64 << 20] {
            let ctx = QueryContext::with_budget(budget);
            let mut a =
                Aggregate::new(Box::new(Values::new(&rows, &s)), &group, &aggs, &out, &ctx).unwrap();
            let mut got = Vec::new();
            while let Some(b) = a.next().unwrap() {
                for i in 0..b.rows() {
                    got.push((0..b.width()).map(|c| b.column(c).value(i)).collect::<Vec<_>>());
                }
            }
            got.sort();
            assert_eq!(got.len(), 200_000, "budget={budget}");
            drop(a);
            assert_eq!(ctx.mem.used(), 0, "budget={budget}: reservation outlived the operator");
            answers.push(got);
        }
        assert_eq!(answers[0], answers[2], "a spilled aggregate lost rows");
        assert_eq!(answers[1], answers[2]);
        assert!(!spilled_dirs().is_empty(), "nothing spilled, so nothing was tested");
        assert_no_temp_files_left();
    }

    // --------------------------------------------------------------- spilling

    fn spilled_dirs() -> Vec<std::path::PathBuf> {
        spill::SPILLED.with(|s| s.borrow().clone())
    }

    fn assert_no_temp_files_left() {
        for d in spilled_dirs() {
            assert!(!d.exists(), "spill directory {} outlived its query", d.display());
        }
        spill::SPILLED.with(|s| s.borrow_mut().clear());
    }

    /// Aggregate `rows` under `budget` and return the rows, sorted.
    fn under(
        rows: &[Vec<Value>],
        in_schema: &Schema,
        group: &[BoundExpr],
        aggs: &[BoundAgg],
        out: &Schema,
        budget: i64,
    ) -> Vec<Vec<Value>> {
        let ctx = QueryContext::with_budget(budget);
        let mut a =
            Aggregate::new(Box::new(Values::new(rows, in_schema)), group, aggs, out, &ctx).unwrap();
        let mut got = Vec::new();
        while let Some(b) = a.next().unwrap() {
            for i in 0..b.rows() {
                got.push((0..b.width()).map(|c| b.column(c).value(i)).collect::<Vec<_>>());
            }
        }
        drop(a);
        assert_eq!(ctx.mem.used(), 0);
        got.sort();
        got
    }

    #[test]
    fn a_spilled_aggregate_agrees_with_an_in_memory_one_on_every_aggregate() {
        // Every aggregate shape at once, including the order-defined ones and
        // DISTINCT. The property that makes this work is that a group is never
        // split between memory and a partition file, so `groupArray`'s element
        // order and `any`'s choice survive; an implementation that evicted the
        // *table* instead would pass a `count`-only test and fail here.
        spill::SPILLED.with(|s| s.borrow_mut().clear());
        let s = Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("v", DataType::Int64),
            Field::new("t", DataType::String),
        ])
        .unwrap();
        let names = ["ann", "bob", "cyd", "dee"];
        let rows: Vec<Vec<Value>> = (0..120_000i64)
            .map(|i| {
                vec![
                    Value::Int(crate::common::splitmix64(i as u64) as i64 % 20_000),
                    Value::Int(i % 251),
                    Value::str(names[i as usize % 4]),
                ]
            })
            .collect();
        let vv = || col(1, DataType::Int64);
        let group = vec![col(0, DataType::Int64)];
        let aggs = vec![
            agg("count", vec![], false),
            agg("sum", vec![vv()], false),
            agg("min", vec![vv()], false),
            agg("max", vec![vv()], false),
            agg("avg", vec![vv()], false),
            agg("uniq", vec![vv()], false),
            agg("any", vec![col(2, DataType::String)], false),
            agg("anyLast", vec![col(2, DataType::String)], false),
            agg("argMin", vec![col(2, DataType::String), vv()], false),
            agg("groupArray", vec![col(2, DataType::String)], false),
            agg("count", vec![vv()], true),
            agg("sum", vec![vv()], true),
        ];
        let mut fields: Vec<(&str, DataType)> = vec![("k", DataType::Int64)];
        let names: Vec<String> = (0..aggs.len()).map(|i| format!("a{i}")).collect();
        for (i, a) in aggs.iter().enumerate() {
            fields.push((names[i].as_str(), a.ty.clone()));
        }
        let out = out_schema(fields);
        let want = under(&rows, &s, &group, &aggs, &out, 512 << 20);
        assert!(spilled_dirs().is_empty(), "the reference run spilled");
        let got = under(&rows, &s, &group, &aggs, &out, 2 << 20);
        assert!(!spilled_dirs().is_empty(), "nothing spilled");
        assert_eq!(got.len(), want.len());
        assert_eq!(got, want, "a spilled aggregate answered differently");
        assert_no_temp_files_left();
    }

    #[test]
    fn a_spilled_aggregate_repartitions_rather_than_recursing_forever() {
        // A budget small enough that one round of partitioning cannot be
        // enough: each level has to split on *fresh* hash bits, or a partition
        // re-splits into one bucket and the recursion never terminates.
        spill::SPILLED.with(|s| s.borrow_mut().clear());
        let (s, rows) = wide_rows(120_000);
        let group = vec![col(0, DataType::Int64)];
        let aggs = vec![agg("count", vec![], false), agg("sum", vec![col(0, DataType::Int64)], false)];
        let out = out_schema(vec![
            ("k", DataType::Int64),
            ("n", aggs[0].ty.clone()),
            ("s", aggs[1].ty.clone()),
        ]);
        let got = under(&rows, &s, &group, &aggs, &out, 192 << 10);
        assert_eq!(got.len(), 120_000);
        for r in &got {
            assert_eq!(r[1], Value::UInt(1), "one row per group");
            assert_eq!(r[2], r[0], "sum of a single-row group is that row");
        }
        assert!(spilled_dirs().len() > 1, "only one level, so no repartitioning ran");
        assert_no_temp_files_left();
    }

    #[test]
    fn a_spilled_aggregate_keeps_composite_and_string_keys() {
        spill::SPILLED.with(|s| s.borrow_mut().clear());
        let s = Schema::new(vec![
            Field::new("a", DataType::Nullable(Box::new(DataType::String))),
            Field::new("b", DataType::Int64),
        ])
        .unwrap();
        let rows: Vec<Vec<Value>> = (0..80_000i64)
            .map(|i| {
                let a = if i % 11 == 0 {
                    Value::Null
                } else {
                    Value::str(format!("key-{}", i % 4_000))
                };
                vec![a, Value::Int(i % 7)]
            })
            .collect();
        let group = vec![col(0, DataType::String), col(1, DataType::Int64)];
        let aggs = vec![agg("count", vec![], false)];
        let out = out_schema(vec![
            ("a", DataType::Nullable(Box::new(DataType::String))),
            ("b", DataType::Int64),
            ("n", aggs[0].ty.clone()),
        ]);
        let want = under(&rows, &s, &group, &aggs, &out, 512 << 20);
        assert!(spilled_dirs().is_empty(), "the reference run spilled");
        let got = under(&rows, &s, &group, &aggs, &out, 1 << 20);
        assert!(!spilled_dirs().is_empty(), "nothing spilled");
        assert_eq!(got, want);
        let total: u64 = got.iter().map(|r| r[2].as_u64().unwrap()).sum();
        assert_eq!(total, 80_000, "rows were lost or double counted");
        assert_no_temp_files_left();
    }

    #[test]
    fn a_cancelled_spilling_aggregate_stops_and_takes_its_files_with_it() {
        spill::SPILLED.with(|s| s.borrow_mut().clear());
        let (s, rows) = wide_rows(200_000);
        let group = vec![col(0, DataType::Int64)];
        let aggs = vec![agg("count", vec![], false)];
        let out = out_schema(vec![("k", DataType::Int64), ("n", aggs[0].ty.clone())]);
        let ctx = QueryContext::with_budget(256 << 10);
        let mut a =
            Aggregate::new(Box::new(Values::new(&rows, &s)), &group, &aggs, &out, &ctx).unwrap();
        ctx.stop();
        let e = a.next().unwrap_err();
        assert!(e.to_string().contains("cancelled"), "{e}");
        drop(a);
        assert_eq!(ctx.mem.used(), 0);
        assert_no_temp_files_left();
    }

    #[test]
    fn a_group_by_over_a_real_scan_spills_through_the_whole_pipeline() {
        // The same shape `operators::tests::pipeline` governs, but reached
        // through `execute_ctx` and a real table rather than through `Values`:
        // 20k groups against a 64 KiB budget, which is a quarter of what one
        // scan block of `UInt64` costs. It has to repartition several levels
        // deep and still return every group exactly once.
        use crate::catalog::Catalog;
        use crate::planner::logical::ScanNode;
        use crate::types::{Column, Engine, TableDef};
        let n = 20_000u64;
        let mut cat = Catalog::in_memory();
        cat.create_table(
            TableDef {
                name: "t".into(),
                schema: Schema::new(vec![Field::new("id", DataType::UInt64)]).unwrap(),
                order_by: vec![0],
                primary_key: vec![0],
                partition_by: None,
                engine: Engine::MergeTree,
            },
            false,
        )
        .unwrap();
        {
            let t = cat.table_by_path_mut("default.t").unwrap();
            t.insert(Block::new(vec![Column::u64s(DataType::UInt64, (0..n).collect())]).unwrap())
                .unwrap();
            t.flush().unwrap();
        }
        let full = cat.table_by_path("default.t").unwrap().schema().clone();
        let aggs = vec![agg("count", vec![], false)];
        let plan = crate::planner::logical::LogicalPlan::Aggregate {
            input: Box::new(crate::planner::logical::LogicalPlan::Scan(Box::new(ScanNode {
                table: "default.t".into(),
                schema: full.project(&[0]),
                projection: vec![0],
                filters: vec![],
                zone_filters: vec![],
            }))),
            group: vec![col(0, DataType::UInt64)],
            aggs: aggs.clone(),
            schema: out_schema(vec![("id", DataType::UInt64), ("n", aggs[0].ty.clone())]),
        };
        let tight = QueryContext::with_budget(64 << 10);
        let (blocks, _) = super::super::execute_ctx(&plan, &cat, &tight).unwrap();
        let mut ids: Vec<u64> = blocks
            .iter()
            .flat_map(|b| (0..b.rows()).map(move |i| b.column(0).value(i).as_u64().unwrap()))
            .collect();
        ids.sort_unstable();
        assert_eq!(ids.len(), n as usize, "a spilled pipeline lost or duplicated groups");
        assert_eq!(ids, (0..n).collect::<Vec<_>>());
        assert_eq!(tight.mem.used(), 0, "the spilled query kept its reservation");
        assert_no_temp_files_left();
    }

    #[test]
    fn a_bare_aggregate_still_reports_a_budget_it_cannot_meet() {
        // The one shape that must not spill: no GROUP BY means one group and
        // no key to partition on, so a `Partitions` here would have to split a
        // single group across files and merge partial accumulators back.
        let s = Schema::new(vec![Field::new("v", DataType::Int64)]).unwrap();
        let rows: Vec<Vec<Value>> = (0..50_000i64).map(|i| vec![Value::Int(i)]).collect();
        let aggs = vec![agg("count", vec![], false)];
        let out = out_schema(vec![("n", aggs[0].ty.clone())]);
        let ctx = QueryContext::with_budget(64 << 20);
        let mut a =
            Aggregate::new(Box::new(Values::new(&rows, &s)), &[], &aggs, &out, &ctx).unwrap();
        let mut n = 0;
        while let Some(b) = a.next().unwrap() {
            n += b.rows();
        }
        assert_eq!(n, 1, "a fold always has exactly one result");
        assert!(spilled_dirs().is_empty(), "a bare aggregate spilled");
    }

    #[test]
    fn a_cancelled_aggregate_stops_before_the_end_of_the_input() {
        let (s, rows) = wide_rows(100_000);
        let group = vec![col(0, DataType::Int64)];
        let aggs = vec![agg("count", vec![], false)];
        let out = out_schema(vec![("k", DataType::Int64), ("n", aggs[0].ty.clone())]);
        let ctx = QueryContext::new();
        ctx.stop();
        let mut a =
            Aggregate::new(Box::new(Values::new(&rows, &s)), &group, &aggs, &out, &ctx).unwrap();
        assert!(a.next().unwrap_err().to_string().contains("cancelled"));
    }

    // -------------------------------------------------------- partial merges

    /// Aggregate `rows` in `n` contiguous slices and fold the partials, the
    /// way the exchange does.
    fn by_parts(
        rows: &[Vec<Value>],
        in_schema: &Schema,
        group: &[BoundExpr],
        aggs: &[BoundAgg],
        out: &Schema,
        n: usize,
    ) -> Vec<Vec<Value>> {
        let ctx = QueryContext::new();
        let protos = protos(aggs).unwrap();
        let mut base: Option<Groups> = None;
        for i in 0..n {
            let (lo, hi) = (rows.len() * i / n, rows.len() * (i + 1) / n);
            let mut guard = MemGuard::new(&ctx, guard_name(group.len()));
            let mut input: Box<dyn Operator> = Box::new(Values::new(&rows[lo..hi], in_schema));
            let g = accumulate(&mut input, group, aggs, &protos, &ctx, &mut guard).unwrap();
            match &mut base {
                None => base = Some(g),
                Some(b) => b.absorb(g, aggs).unwrap(),
            }
        }
        emit(&base.unwrap(), group, aggs, out)
            .unwrap()
            .iter()
            .flat_map(|b| {
                (0..b.rows())
                    .map(move |i| (0..b.width()).map(|c| b.column(c).value(i)).collect::<Vec<_>>())
            })
            .collect()
    }

    /// One pass over the same rows, for comparison.
    fn whole(
        rows: &[Vec<Value>],
        in_schema: &Schema,
        group: &[BoundExpr],
        aggs: &[BoundAgg],
        out: &Schema,
    ) -> Vec<Vec<Value>> {
        by_parts(rows, in_schema, group, aggs, out, 1)
    }

    #[test]
    fn folding_partials_matches_one_pass_including_group_order() {
        // Row for row and in the same order: the exchange gives worker k a
        // contiguous ascending slice and absorbs in worker order precisely so
        // that first-seen group order survives, and an assertion on the
        // multiset would not notice if it stopped.
        let s = Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("v", DataType::Int64),
            Field::new("s", DataType::String),
        ])
        .unwrap();
        let names = ["ann", "bob", "cyd"];
        let rows: Vec<Vec<Value>> = (0..3_000i64)
            .map(|i| {
                vec![
                    Value::Int(crate::common::splitmix64(i as u64) as i64 % 97),
                    Value::Int(i % 251),
                    Value::str(names[i as usize % 3]),
                ]
            })
            .collect();
        let vv = || col(1, DataType::Int64);
        for group in [vec![], vec![col(0, DataType::Int64)]] {
            let aggs = vec![
                agg("count", vec![], false),
                agg("sum", vec![vv()], false),
                agg("min", vec![vv()], false),
                agg("max", vec![vv()], false),
                agg("avg", vec![vv()], false),
                agg("uniq", vec![vv()], false),
                // Order-defined: these are the ones a shuffled merge breaks.
                agg("any", vec![col(2, DataType::String)], false),
                agg("anyLast", vec![col(2, DataType::String)], false),
                agg("argMin", vec![col(2, DataType::String), vv()], false),
                agg("groupArray", vec![col(2, DataType::String)], false),
                // DISTINCT: the seen-sets union rather than the counts adding.
                agg("count", vec![vv()], true),
                agg("sum", vec![vv()], true),
            ];
            let mut fields: Vec<(&str, DataType)> = Vec::new();
            if !group.is_empty() {
                fields.push(("k", DataType::Int64));
            }
            let names: Vec<String> = (0..aggs.len()).map(|i| format!("a{i}")).collect();
            for (i, a) in aggs.iter().enumerate() {
                fields.push((names[i].as_str(), a.ty.clone()));
            }
            let out = out_schema(fields);
            let want = whole(&rows, &s, &group, &aggs, &out);
            for n in [2usize, 3, 7, 14, 64] {
                assert_eq!(
                    by_parts(&rows, &s, &group, &aggs, &out, n),
                    want,
                    "{n} partials disagree with one pass (group={})",
                    group.len()
                );
            }
        }
    }

    #[test]
    fn a_distinct_partial_unions_rather_than_adding_up() {
        // The bug this pins: two partials that both saw value 2 would count it
        // twice if the accumulators were merged instead of the seen-sets.
        let s = Schema::new(vec![Field::new("v", DataType::Int64)]).unwrap();
        let rows: Vec<Vec<Value>> = [1i64, 2, 2, 3, 1, 3]
            .iter()
            .map(|&v| vec![Value::Int(v)])
            .collect();
        let aggs = vec![
            agg("count", vec![col(0, DataType::Int64)], true),
            agg("sum", vec![col(0, DataType::Int64)], true),
        ];
        let out = out_schema(vec![("n", aggs[0].ty.clone()), ("s", aggs[1].ty.clone())]);
        // Every split, so the overlap between adjacent partials varies.
        for n in 1..=6 {
            assert_eq!(
                by_parts(&rows, &s, &[], &aggs, &out, n),
                vec![vec![Value::UInt(3), Value::Int(6)]],
                "{n} partials"
            );
        }
    }

    #[test]
    fn absorbing_an_empty_partial_changes_nothing() {
        // A worker whose granule range held no live rows still hands back a
        // table -- with the bare aggregate's single group in it, or with
        // nothing when there is a GROUP BY.
        let (s, rows) = src();
        let group = vec![col(0, DataType::Int64)];
        let aggs = vec![agg("count", vec![], false)];
        let out = out_schema(vec![("k", DataType::Int64), ("n", aggs[0].ty.clone())]);
        let ctx = QueryContext::new();
        let protos = protos(&aggs).unwrap();
        let mut guard = MemGuard::new(&ctx, "test");
        let mut full: Box<dyn Operator> = Box::new(Values::new(&rows, &s));
        let mut base = accumulate(&mut full, &group, &aggs, &protos, &ctx, &mut guard).unwrap();
        let want = emit(&base, &group, &aggs, &out).unwrap();
        let mut none: Box<dyn Operator> = Box::new(Values::new(&[], &s));
        let empty = accumulate(&mut none, &group, &aggs, &protos, &ctx, &mut guard).unwrap();
        base.absorb(empty, &aggs).unwrap();
        let got = emit(&base, &group, &aggs, &out).unwrap();
        assert_eq!(got.len(), want.len());
        assert_eq!(got[0].rows(), want[0].rows());
        assert_eq!(got[0].column(1).value(0), want[0].column(1).value(0));
    }

    #[test]
    fn the_flat_arenas_stay_aligned_with_many_aggregates_and_groups() {
        // Guards the arena indexing: a stride bug shows up as one group's
        // aggregate landing in another group's slot, which a single-aggregate
        // test cannot see.
        let s = Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("v", DataType::Int64),
        ])
        .unwrap();
        let rows: Vec<Vec<Value>> = (0..5_000i64)
            .map(|i| vec![Value::Int(i % 500), Value::Int(i)])
            .collect();
        let group = vec![col(0, DataType::Int64)];
        let aggs = vec![
            agg("count", vec![], false),
            agg("min", vec![col(1, DataType::Int64)], false),
            agg("max", vec![col(1, DataType::Int64)], false),
            agg("sum", vec![col(1, DataType::Int64)], false),
        ];
        let out = out_schema(vec![
            ("k", DataType::Int64),
            ("n", aggs[0].ty.clone()),
            ("lo", aggs[1].ty.clone()),
            ("hi", aggs[2].ty.clone()),
            ("s", aggs[3].ty.clone()),
        ]);
        let got = run(&rows, &s, &group, &aggs, &out);
        assert_eq!(got.len(), 500);
        for r in &got {
            let k = r[0].as_i64().unwrap();
            assert_eq!(r[1], Value::UInt(10), "group {k}");
            assert_eq!(r[2], Value::Int(k), "min of group {k}");
            assert_eq!(r[3], Value::Int(k + 4_500), "max of group {k}");
            let want: i64 = (0..10).map(|j| k + j * 500).sum();
            assert_eq!(r[4], Value::Int(want), "sum of group {k}");
        }
    }
}
