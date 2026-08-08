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
//! ## What a row costs on the way in
//!
//! Pass 1 is the only per-row work left, and for a long time it went through
//! [`Value`] whatever the key was: build one per row, hash it with a
//! `MixHasher` (which for an integer means an `as_f64`, a `fract()` and three
//! 128-bit folds), then compare a `&[Value]` slice against the arena. For a
//! single key column -- which is most of them -- none of that is necessary:
//!
//! * an **integer** key is hashed and compared as its own lane
//!   ([`lane_hash`], [`lane_of`]); the two constant folds `Value::hash` makes
//!   before it reaches the integer are compile-time constants, so a row's hash
//!   is one multiply and a genuinely new group is the only thing that ever
//!   builds a `Value`;
//! * a **string** key is memoized by the *address* of its decoded `Arc`
//!   ([`StrMemo`]) -- a granule decodes its dictionary once and clones one
//!   pointer per row, so eight countries over 8192 rows are eight hashes and
//!   8184 pointer compares;
//! * every probe reads its hash **tag out of the slot** rather than out of a
//!   second array, so ruling a colliding group out costs no memory access at
//!   all (`Groups::slots`);
//! * and past the point where the slot array leaves cache, a probe
//!   software-prefetches twelve rows ahead ([`lane_rows`]).
//!
//! Each of those carries its own interleaved measurement where it is decided.
//! End to end, `benches/engine.rs` over 2M rows, two builds of the whole tree
//! differing **only** in this file, run alternately, best-of-12 per side, two
//! runs. Serial (`GRANULAR_THREADS=1`), because fourteen threads on a loaded
//! machine put a +-20% band on every reading and this has to be readable --
//! `top-k by sort`, which this file cannot touch, is the control:
//!
//! ```text
//!                                          ms            three runs
//!   GROUP BY user_id (100k groups)     72.29 -> 40.85   1.89 1.95 1.77
//!   GROUP BY country (8 groups)        37.94 -> 23.43   1.58 1.49 1.62
//!   filter + GROUP BY + ORDER + LIMIT  24.22 -> 17.74   1.42 1.30 1.37
//!   uniq(user_id)          (no GROUP)  11.65 ->  8.89   1.36 1.26 1.31
//!   sum(bytes)             (no GROUP)  10.64 ->  8.17   1.23 1.63 1.30
//!   quantile(0.95)(latency)            12.75 -> 10.97   1.29 1.24 1.16
//!   top-k by sort           (control)   3.22 ->  3.07   1.01 1.00 1.05
//! ```
//!
//! The three bare aggregates move because two of these changes are not about
//! keys at all: aggregate arguments are borrowed rather than cloned per block
//! ([`ArgSrc`]) and the counting sort's `order` buffer stopped being zeroed
//! before it was overwritten. At fourteen threads the same A/B reads 1.31x /
//! 1.23x on the high-cardinality grouping and 1.52x / 1.34x on `country`,
//! against a control that swung 1.18x / 0.96x -- consistent with the serial
//! numbers, and a good illustration of why they were taken serially.
//!
//! Every one of those paths must answer exactly what the general path answers.
//! [`GENERAL_KEYS`] exists so a test can run both over the same rows and
//! compare the bytes; `tests/golf_aggregate.rs` does, across five
//! cardinalities, eight key types and the order-sensitive aggregates.
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
//! **In parallel too.** The exchange's workers each build one of these tables,
//! and until recently none of them could spill: a `GROUP BY` that a serial plan
//! answered under a tight budget was an error as soon as it went wide, and the
//! only thing bounding fourteen partial tables was the whole query failing. A
//! worker now freezes and spills exactly as the serial operator does. What it
//! cannot do is fold what it spilled -- its partitions' keys are disjoint from
//! its *own* resident groups but not from the other workers' -- so the
//! partitions ride along on the table it hands back and [`emit_spilled`] folds
//! them against the merged result. Three things follow, and all three are
//! measured where they are decided: the partition mask has to be the same for
//! every worker ([`Partitions::arm`]), a worker has to stop short of the whole
//! budget so the merge above it can happen at all ([`worker_ceiling`]), and the
//! fold has to treat the merged table as read-only ([`emit_spilled`]).
//!
//! What it costs, 2M rows and 1.1M groups on 14 threads, budget swept downwards,
//! best-of-3 per budget, two runs:
//!
//! ```text
//!   budget      8G    4G     2G     1G   512M   256M   128M    96M    64M
//!   GROUP BY   451   448   1055   1094   1047   1023   1037   1017   fails  ms
//!                       ^ last budget that holds the partials
//! ```
//!
//! **A 2.3x cliff**, flat below it, and the same shape the serial spill has --
//! what is paid is one write and one read of the rows that missed a frozen
//! table. Two boundaries are worth knowing. The cliff arrives at an *eighth* of
//! the budget rather than at the budget, because the partials, the merged table
//! and one folded bucket are resident together (see [`worker_ceiling`]). And
//! below ~96 MiB the query fails however hard it spills: fourteen workers each
//! admit one more block of groups before the per-block check fires, so the
//! fleet's floor is 14 x 8192 groups whatever the budget says. Both of those
//! are the exchange's accounting, not this file's.
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

use crate::common::{BitSet, FastSet, Result, BLOCK_SIZE};
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
        let mut parts = Partitions::new(0, 0);
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

/// Force every group key down the general per-row `Value` path.
///
/// The fast paths below -- the integer lane loop and the string address memo --
/// exist only if they answer *exactly* what the general path answers, and the
/// only honest way to check that is to run the same query both ways and compare
/// the bytes. `tests/golf_aggregate.rs` is the caller; nothing else ever writes
/// it. Costs one relaxed load per **block**, hoisted out of the row loop, which
/// is the same accounting the cancel check already makes.
pub static GENERAL_KEYS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[inline]
fn general_keys() -> bool {
    GENERAL_KEYS.load(std::sync::atomic::Ordering::Relaxed)
}

/// What the memory budget calls this operator's unbounded state.
pub(crate) fn guard_name(ngroup: usize) -> &'static str {
    if ngroup == 0 {
        "the aggregate state"
    } else {
        "the GROUP BY hash table"
    }
}

/// Fold every row `input` produces into a group table, spilling if it must.
///
/// Split out of the operator so a parallel exchange can call it once per
/// worker; `guard` is the caller's because the budget has to be charged
/// against the whole query, not per worker.
///
/// A worker spills exactly as the serial operator does -- before this it could
/// not, so a `GROUP BY` that was answerable serially under a tight budget was
/// an error in parallel, and the fourteen partial tables were bounded only by
/// the whole query failing. What a worker cannot do is *fold* what it spilled:
/// its partitions' keys are disjoint from its own resident groups but not from
/// the other workers', so folding them here would emit a group twice. They ride
/// along on the table instead and [`emit`] folds them once the partials have
/// been merged; see [`emit_spilled`].
///
/// A worker that fits pays nothing for any of it: the freeze is the `grow_to`
/// this loop already made, the ceiling is one relaxed load per block, and
/// `Partitions` allocates nothing until it is armed. Measured interleaved
/// against `accumulate_into(.., None)` -- the shape a worker had before this --
/// alternating sides in one loop, best-of-7 over 2M rows, three runs:
/// 8 groups 1.011x / 0.955x / 1.024x, 1k groups 0.967x / 0.930x / 0.995x, 100k
/// groups 0.983x / 0.982x / 1.052x. Null, with no consistent sign.
pub(crate) fn accumulate(
    input: &mut Box<dyn Operator + '_>,
    group: &[BoundExpr],
    aggs: &[BoundAgg],
    protos: &[Box<dyn Accumulator>],
    ctx: &QueryContext,
    guard: &mut MemGuard,
) -> Result<Groups> {
    let mut parts = Partitions::new(0, worker_ceiling(ctx));
    // Same exception the serial path makes: a bare aggregate's one group exists
    // before the first row and has no key to partition by.
    let spill = (!group.is_empty()).then_some(&mut parts);
    let mut groups = accumulate_into(input, group, aggs, protos, ctx, guard, spill)?;
    let pending = parts.finish()?;
    if !pending.is_empty() {
        groups.over = Some(Box::new(Overflow { ctx: own_ctx(ctx), pending }));
    }
    Ok(groups)
}

/// The size one parallel worker's table stops at, so that the merge above it
/// has somewhere to happen.
///
/// A worker that grows until the shared budget refuses leaves nothing for what
/// comes next, and what comes next needs two more tables' worth: the exchange
/// folds the partials with [`Groups::absorb`] while still holding them (its own
/// accounting is conservative by one partial, so its peak is `partials +
/// merged`), and then [`emit_spilled`] rebuilds one spilled bucket at a time
/// against the merged table. So the fleet has to stop at an **eighth** of the
/// budget: a half would leave the merge exactly nothing, and the check is made
/// once per block, so by the time it fires every worker has taken one more
/// block -- and a block that doubles a `Vec`'s capacity doubles the table.
/// Measured on 14 threads over 1.1M groups, the fleet overshoots by ~2.4x, and
/// an eighth is where the query stops failing: same table, budget swept
/// downward, a third answered only from 384 MiB, an eighth from 96 MiB.
///
/// The share is per *worker* and derived from the pool's width rather than
/// from the shared `used` counter, and that is the whole point: a ceiling read
/// off `used` fires at a moment that depends on how the fourteen threads
/// happened to interleave, so which keys a worker still had resident -- and
/// therefore what `any(x)` and `groupArray(x)` answered -- changed from run to
/// run. This fires at a fixed table size on a fixed slice of the input, so a
/// spilled parallel `GROUP BY` answers the same thing every time. It is also
/// cheaper: the comparison is against `groups.bytes()`, which the `grow_to` on
/// the same line already computed, so the block loop gains no load at all.
///
/// `threads` and not the exchange's actual degree, because a worker is not told
/// how many others there are. Over-dividing is the safe direction -- a narrower
/// fleet spills a little earlier than it had to.
///
/// None of this makes the parallel aggregate stricter than it was: above a half
/// the merge already failed, so the change is that the query spills at three
/// eighths instead of failing at a half.
fn worker_ceiling(ctx: &QueryContext) -> usize {
    let share = (ctx.mem.limit().max(0) as usize) / 8;
    (share / crate::common::pool::global().threads().max(1)).max(1)
}

/// Where one aggregate's argument columns come from, decided once per query.
///
/// `Accumulator::update` wants a `&[Column]`, and the obvious way to get one --
/// `expr::eval_all(&a.args, &b)` -- *clones* the column when the argument is a
/// bare reference, which `sum(bytes)` and `avg(latency)` and almost every
/// aggregate anyone writes are. A one-column window of the block is already a
/// `&[Column]`, so the copy is avoidable outright. What is left is one `match`
/// per (group, aggregate) per block, which against 8192 rows is nothing.
///
/// Worth less than it looks. Measured interleaved against a switch that forced
/// every argument back through `eval_all`, alternating sides in one loop,
/// best-of-9 per side, 2M rows, **serially** (`GRANULAR_THREADS=1`, because at
/// fourteen threads this machine's readings on the same code span 3x and the
/// sign flipped run to run), three runs -- `sum(v)` 1.078 / 1.104 / 1.091,
/// `GROUP BY i` with `count+sum+min` 1.101 / 1.033 / 1.111, `quantile(0.95)(v)`
/// 1.028 / 1.184 / 0.994, `GROUP BY s` with `count+avg` 1.001 / 0.939 / 0.970.
/// So: about **1.09x** where the aggregate reads a plain column, inside the
/// noise everywhere else, and never a loss outside it. The copy it removes is
/// 64 KiB per block, which is a memcpy the prefetcher eats -- what is actually
/// saved is the allocation and the free, once per block per worker. (The
/// larger bare-aggregate numbers in the module header are this *and* the
/// `order` memset; neither alone accounts for them.)
enum ArgSrc {
    /// `count()`: no arguments, and so no columns to find.
    Empty,
    /// A bare column reference: `&block.columns[i..i+1]`.
    Borrow(usize),
    /// An expression, evaluated into the block's owned scratch.
    Eval(usize),
}

impl ArgSrc {
    #[inline(always)]
    fn cols<'b>(&self, b: &'b Block, owned: &'b [Vec<Column>]) -> &'b [Column] {
        match self {
            ArgSrc::Empty => &[],
            // In range: checked once per block, where the error can be raised.
            ArgSrc::Borrow(i) => &b.columns[*i..*i + 1],
            ArgSrc::Eval(k) => &owned[*k],
        }
    }
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
    // Address-keyed memo for the single-string key; see [`StrMemo`]. Costs one
    // allocation for a query that groups by a string and nothing at all for
    // any other shape.
    let mut memo = StrMemo::default();
    // Where each aggregate's argument columns come from, decided once for the
    // query rather than rebuilt per block. See [`ArgSrc`].
    let mut nowned = 0usize;
    let arg_src: Vec<ArgSrc> = aggs
        .iter()
        .map(|a| match a.args.as_slice() {
            [] => ArgSrc::Empty,
            [BoundExpr::Column { index, .. }] => ArgSrc::Borrow(*index),
            _ => {
                nowned += 1;
                ArgSrc::Eval(nowned - 1)
            }
        })
        .collect();
    let mut owned: Vec<Vec<Column>> = vec![Vec::new(); nowned];
    let mut frozen = false;
    let forced = if spill.is_some() { super::sort::forced_spill_rows() } else { 0 };
    // Hoisted out of the block loop: `0` for every serial aggregate and for
    // every worker whose budget is unbounded.
    let soft = spill.as_deref().map_or(0, |p| p.soft);

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
        // Only the arguments that are *expressions* are evaluated; a bare
        // column reference is handed to `update` as a one-column slice of the
        // block itself. `Accumulator::update` takes `&[Column]`, so a
        // one-argument aggregate over a plain column -- `sum(bytes)`,
        // `avg(latency)`, the overwhelming majority -- needs no copy at all,
        // where `eval_all` cloned the whole column once per block.
        // The `Borrow` arm's range check is made here rather than in the fold,
        // so that the fold can slice unconditionally and so that an
        // out-of-range column is one error per block instead of one per group.
        for (a, s) in aggs.iter().zip(&arg_src) {
            match s {
                ArgSrc::Eval(k) => owned[*k] = expr::eval_all(&a.args, &b)?,
                ArgSrc::Borrow(ix) if *ix >= b.width() => {
                    return Err(crate::common::Error::exec(format!(
                        "aggregate {} reads column #{ix} of a {}-column block",
                        a.name,
                        b.width()
                    )))
                }
                _ => {}
            }
        }

        // Pass 1: resolve each row's group. The only per-row hashing, and
        // it probes the table through `probe` without allocating.
        //
        // `None` on the in-memory path, where a row's id *is* its index into
        // `row_group`; `Some(hits)` once frozen, where the rows that missed
        // have been left out. One branch per block, in pass 2.
        let mut ids: Option<&[u32]> = None;
        if frozen {
            row_group.clear();
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
            // Not `clear() + resize`: every one of these slots is written by
            // the loops below, so zeroing them first was a 32 KiB memset per
            // block. `resize` alone writes only what a *shorter* previous
            // block left uncovered, which in steady state is nothing.
            row_group.resize(rows, 0);
            // A single key column gets its own loop per physical shape. Both
            // are the same algorithm as the general path with the per-row
            // `Value` taken out: it is the `Value` -- and the hashing and
            // comparing that follow from it -- that costs, not the probing.
            //
            // What that is worth, measured interleaved against `GENERAL_KEYS`
            // (which forces this whole `match` down its last arm), alternating
            // sides in one loop, best-of-7 per side, 2M rows, serial, three
            // runs per cell, medians:
            //
            // ```text
            //   groups        8        1k      100k
            //   Int64      2.5x      1.85x     1.7x
            //   Int64 NULL 1.7x      1.6x      1.6x      <- nullable, per row
            //   String     2.4x      1.20x     1.2x
            // ```
            //
            // The integer column keeps its factor as the table leaves cache
            // because what it removed -- an `as_f64`, a `fract()`, three
            // 128-bit folds and a `Vec<Value>` compare -- is work, not a miss.
            // The string column's collapses because the memo can only spare
            // the *hash*: past a few hundred distinct values per block the
            // probe is a cache miss either way, and the memo's own lookup is a
            // second one. Parallel (14 threads) reads 1.76x / 1.98x / 1.11x on
            // the same three integer cells, which is the same story with this
            // machine's noise on top.
            let single = (ngroup == 1 && !general_keys()).then(|| gcols[0].as_ref());
            let lane = single.and_then(lane_col);
            match (lane, single) {
                // Integer key: hash and compare the lane itself.
                (Some(lc), Some(col)) => {
                    let (kind, mask, guard) = (lc.kind, lc.mask, lc.guard);
                    let nm = col.nulls.as_ref();
                    let (rg, gs) = (&mut row_group[..], &mut groups);
                    match (lc.lanes, nm.is_some()) {
                        (Lanes::U(v), false) => {
                            lane_pass::<_, false>(v, nm, mask, guard, kind, rg, gs, protos, aggs)
                        }
                        (Lanes::U(v), true) => {
                            lane_pass::<_, true>(v, nm, mask, guard, kind, rg, gs, protos, aggs)
                        }
                        (Lanes::I(v), false) => {
                            lane_pass::<_, false>(v, nm, mask, guard, kind, rg, gs, protos, aggs)
                        }
                        (Lanes::I(v), true) => {
                            lane_pass::<_, true>(v, nm, mask, guard, kind, rg, gs, protos, aggs)
                        }
                    }
                }
                // String key. The owned `Value` (and its `Arc` bump) is built
                // only for a genuinely new group, and the memo usually spares
                // even the hash.
                (None, Some(col)) if matches!(col.data, crate::types::ColumnData::Str(_)) => {
                    let vals = col.as_str()?;
                    let null_h = super::hash_null_key();
                    let nulls = col.has_nulls();
                    memo.reset(rows);
                    for (i, slot) in row_group.iter_mut().enumerate() {
                        if nulls && col.is_null(i) {
                            *slot = groups.find_or_insert(&[Value::Null], null_h, protos, aggs)
                                as u32;
                            continue;
                        }
                        let s = &vals[i];
                        *slot = match memo.get(s) {
                            Some(g) => g,
                            None => {
                                let h = super::hash_str_key(s);
                                let g = groups.find_or_insert_str(s, h, protos, aggs) as u32;
                                memo.put(s, g);
                                g
                            }
                        };
                    }
                }
                _ if ngroup > 0 => {
                    for (i, slot) in row_group.iter_mut().enumerate() {
                        for (k, c) in gcols.iter().enumerate() {
                            probe[k] = c.as_ref().value(i);
                        }
                        let h = super::hash_values(&probe);
                        *slot = groups.find_or_insert(&probe, h, protos, aggs) as u32;
                    }
                }
                _ => {}
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
        // Grown, never cleared, for the same reason `counts` is: the scatter
        // below writes every one of the first `row_group.len()` slots exactly
        // once (a counting sort is a permutation), so `clear() + resize(n, 0)`
        // was a 32 KiB memset per block whose every byte was then overwritten.
        // A *bare* aggregate paid it too -- one group, 8192 zeroes, every
        // block -- which is most of why `sum`, `uniq` and `quantile` moved in
        // the module header's table without their key path changing at all.
        if order.len() < row_group.len() {
            order.resize(row_group.len(), 0);
        }
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
                for (ai, src) in arg_src.iter().enumerate() {
                    groups.accs[base + ai].update(src.cols(&b, &owned), s)?;
                }
                continue;
            }
            for (ai, src) in arg_src.iter().enumerate() {
                let args = src.cols(&b, &owned);
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
        let bytes = groups.bytes();
        let over = guard.grow_to(bytes);
        if over.is_err() || (forced != 0 && groups.len >= forced) || (soft != 0 && bytes > soft) {
            match spill.as_deref_mut() {
                None => over?,
                Some(p) => {
                    p.arm(ctx, super::sort::share_of(guard));
                    frozen = true;
                }
            }
        }
    }
    Ok(groups)
}

/// Turn a finished group table into `[group..., aggs...]` blocks, in the
/// table's own group order.
///
/// [`Accumulator::finish`] is fallible (it narrows a wider fold to the declared
/// return type and may refuse), which widens its return from 24 bytes to 48 and
/// adds a branch per group. Measured, because a `Result` through a vtable is
/// exactly the shape that stops inlining: A/B interleaved with the order
/// swapped each round, best-of-25 per side, **timing the emit loop alone**
/// (the scan and the hash build are ~95% of the query and all of its variance,
/// and neither changed), 400k groups over 800k rows, six rounds, medians of
/// new/old -- `sum(Int64)` 0.91, `avg(Int64)` 0.99, `sum(Float64)` 0.94,
/// `sum(Decimal64)` 0.97, `avg(Decimal64)` 0.97, `count+min+max` 0.96. Net
/// faster on every aggregate, because the `split_at_mut` + `zip` below pays for
/// the `Result` several times over: it drops the per-*cell* bounds check on the
/// key arena, the accumulator arena and the builder vector to one per group.
/// (Load average was ~55 during the run, so single readings spanned 0.72-1.58;
/// the medians are the number. End to end the whole loop is ~5% of the query,
/// so none of this is visible from outside.)
pub(crate) fn emit(
    groups: &Groups,
    group: &[BoundExpr],
    aggs: &[BoundAgg],
    schema: &Schema,
) -> Result<Vec<Block>> {
    // One predictable branch on a null pointer, once per query, for every
    // aggregate that fit in memory.
    if groups.over.is_some() {
        return emit_spilled(groups, group, aggs, schema);
    }
    let (nagg, ngroup) = (aggs.len(), group.len());
    let width = ngroup + nagg;
    if width == 0 {
        return Ok(if groups.len == 0 {
            Vec::new()
        } else {
            vec![Block::rows_only(groups.len)]
        });
    }
    let tys = out_types(group, aggs, schema);

    let total = groups.len;
    let mut out = Vec::with_capacity(total.div_ceil(BLOCK_SIZE));
    let mut start = 0;
    while start < total {
        let end = (start + BLOCK_SIZE).min(total);
        let mut builders: Vec<ColumnBuilder> = tys
            .iter()
            .map(|t| ColumnBuilder::with_capacity(t.clone(), end - start))
            .collect();
        // Split once per block rather than adding `ngroup + ai` per cell, and
        // zip over slices rather than index them: both arenas are proved in
        // range once per group instead of once per column. See the note above.
        let (kb, ab) = builders.split_at_mut(ngroup);
        for g in start..end {
            for (b, k) in kb.iter_mut().zip(&groups.keys[g * ngroup..][..ngroup]) {
                b.push_value(k)?;
            }
            for (b, a) in ab.iter_mut().zip(&groups.accs[g * nagg..][..nagg]) {
                b.push_value(&a.finish()?)?;
            }
        }
        out.push(finish_block(builders)?);
        start = end;
    }
    Ok(out)
}

/// The output column types, in `[group..., aggs...]` order.
///
/// The plan's schema wins where it has an opinion; the expressions' own types
/// are the fallback for a caller that built a narrower schema.
fn out_types(group: &[BoundExpr], aggs: &[BoundAgg], schema: &Schema) -> Vec<DataType> {
    let ngroup = group.len();
    (0..ngroup + aggs.len())
        .map(|i| {
            if i < schema.len() {
                schema.ty(i).clone()
            } else if i < ngroup {
                group[i].ty()
            } else {
                aggs[i - ngroup].ty.clone()
            }
        })
        .collect()
}

/// Close one output block.
///
/// An aggregate over an empty group finishes as NULL (`min` of nothing), so a
/// column can acquire a mask even where the plan's schema said otherwise. A
/// live mask must never sit on a non-Nullable type, so widen when it happens.
fn finish_block(builders: Vec<ColumnBuilder>) -> Result<Block> {
    Block::new(
        builders
            .into_iter()
            .map(|b| {
                let mut c = b.finish();
                if c.has_nulls() && !c.ty.is_nullable() {
                    c.ty = c.ty.to_nullable();
                }
                c
            })
            .collect(),
    )
}

// ---------------------------------------------------- folding a parallel spill

/// [`emit`] for a merged table that still owes the rows its workers spilled.
///
/// The serial spill can *append* its partitions to the answer, because a key it
/// spilled is by construction a key its one table had never seen. A parallel
/// spill cannot: the tables were frozen per worker, so a key worker 0 spilled
/// may be a key worker 1 held resident, and appending would emit that group
/// twice with each half of its rows.
///
/// So the merged table is the authority and the partitions are folded *against*
/// it. Each bucket -- all workers' files for it at once, in worker order -- is
/// aggregated on its own by the same [`accumulate_into`], spilling again if it
/// still does not fit, and the resulting groups are split in two: one the
/// merged table has never seen is emitted as it stands, one it already holds is
/// emitted as the two combined, and the base group is struck off the tail.
///
/// The merged table is only ever **read**. That is what keeps this inside
/// `emit`'s `&Groups` -- widening it would ripple into the exchange -- and,
/// more to the point, what bounds the fold: the table never grows, so the
/// resident set is the table plus one bucket at a time rather than the table
/// plus every group that ever spilled. The combining is done in a scratch
/// accumulator cloned from the *prototype* and merged with both sides, rather
/// than by cloning the base's, because `Accumulator::boxed_clone` promises a
/// fresh accumulator of the same kind and not a copy of its state.
///
/// Cost: one hash probe per spilled group, and for a group the two sides share,
/// two `merge` calls on top. Order: the folded buckets first, then the merged
/// table's remaining groups -- a spilled `GROUP BY` has never promised
/// first-seen order (the serial one emits its partitions after its table for
/// the same reason), and what it does promise, determinism, holds because the
/// exchange's split is static.
///
/// One order caveat that the serial spill does not have: for a group split
/// between a resident table and another worker's spill file, the resident rows
/// are folded first whichever worker they came from. `any(x)` over such a group
/// can therefore answer with a later row than a serial scan would. It is
/// deterministic, and it only arises once a query has spilled.
fn emit_spilled(
    base: &Groups,
    group: &[BoundExpr],
    aggs: &[BoundAgg],
    schema: &Schema,
) -> Result<Vec<Block>> {
    let ov = base.over.as_ref().expect("checked by the caller");
    let ctx = &ov.ctx;
    let protos = protos(aggs)?;
    let (nagg, nkeys) = (aggs.len(), base.nkeys);
    let mut sink = Rows::new(out_types(group, aggs, schema));
    // Base groups a bucket has already emitted, merged with its own rows. One
    // bit per group of a table that has just proved it is as large as the
    // budget allows, so a bitset and not a `Vec<bool>`.
    let mut done = BitSet::with_capacity_bits(base.len);
    let mut vals: Vec<Value> = Vec::with_capacity(nagg);
    let mut more: Vec<Partition> = Vec::new();

    for bucket in buckets(&ov.pending) {
        fold_bucket(&bucket, base, group, aggs, &protos, ctx, &mut sink, &mut done, &mut more)?;
    }
    // A bucket that still did not fit re-partitioned on fresh hash bits, and
    // each level removes at least the groups that did fit, so this terminates.
    // Depth-first (`pop`), so peak disk tracks what is still owed.
    while let Some(p) = more.pop() {
        fold_bucket(&[p], base, group, aggs, &protos, ctx, &mut sink, &mut done, &mut more)?;
    }

    for g in 0..base.len {
        if done.get(g) {
            continue;
        }
        vals.clear();
        for a in &base.accs[g * nagg..][..nagg] {
            vals.push(a.finish()?);
        }
        sink.push(&base.keys[g * nkeys..][..nkeys], &vals)?;
    }
    sink.finish()
}

/// Group the workers' partition files by bucket, keeping worker order inside
/// each and bucket order between them.
///
/// A `Vec<Vec<_>>` indexed by bucket would be tidier and is not worth it: the
/// buckets are at most 64 and the files at most 64 per bucket, so this is a
/// short stable partition of a list that is already nearly sorted.
fn buckets(pending: &[Partition]) -> Vec<Vec<Partition>> {
    let mut out: Vec<Vec<Partition>> = Vec::new();
    for p in pending {
        match out.iter_mut().find(|b| b[0].idx == p.idx && b[0].level == p.level) {
            Some(b) => b.push(p.clone()),
            None => out.push(vec![p.clone()]),
        }
    }
    out
}

/// Fold one bucket's files into `sink`, against the read-only `base`.
#[allow(clippy::too_many_arguments)]
fn fold_bucket(
    bucket: &[Partition],
    base: &Groups,
    group: &[BoundExpr],
    aggs: &[BoundAgg],
    protos: &[Box<dyn Accumulator>],
    ctx: &QueryContext,
    sink: &mut Rows,
    done: &mut BitSet,
    more: &mut Vec<Partition>,
) -> Result<()> {
    ctx.check()?;
    let (nagg, nkeys) = (aggs.len(), base.nkeys);
    let paths: Vec<std::path::PathBuf> = bucket.iter().map(|p| p.path.clone()).collect();
    let mut guard = MemGuard::new(ctx, guard_name(group.len()));
    let mut input: Box<dyn Operator> =
        Box::new(SpillScan::open(bucket[0].schema.clone(), &paths));
    let mut parts = Partitions::new(bucket[0].level, 0);
    let mut t =
        accumulate_into(&mut input, group, aggs, protos, ctx, &mut guard, Some(&mut parts))?;
    // Unlinked as soon as it has been folded rather than with the whole
    // directory, so peak disk tracks what is still owed and not what the query
    // has ever spilled.
    drop(input);
    for p in &paths {
        let _ = std::fs::remove_file(p);
    }
    more.extend(parts.finish()?);

    let mut vals: Vec<Value> = Vec::with_capacity(nagg);
    let mut fresh: Vec<GroupKey> = Vec::new();
    for g in 0..t.len {
        let key = &t.keys[g * nkeys..][..nkeys];
        vals.clear();
        match base.find(key, t.hashes[g]) {
            // Nobody's resident table held this key, so `t` is the whole group.
            None => {
                for a in &t.accs[g * nagg..][..nagg] {
                    vals.push(a.finish()?);
                }
            }
            Some(b) => {
                done.set(b);
                for ai in 0..nagg {
                    let mut acc = protos[ai].boxed_clone();
                    acc.merge(&*base.accs[b * nagg + ai])?;
                    // DISTINCT is the one shape that cannot go through `merge`
                    // twice: both sides deduplicated within themselves, so the
                    // overlap would be counted once each. Replay only the
                    // tuples the base has not already seen -- the same argument
                    // `absorb` makes, read-only on the base's side of it.
                    match (
                        base.seen.get(b * nagg + ai).and_then(|s| s.as_ref()),
                        t.seen.get_mut(g * nagg + ai).and_then(|s| s.as_mut()),
                    ) {
                        (Some(mine), Some(theirs)) => {
                            fresh.clear();
                            fresh.extend(theirs.drain().filter(|x| !mine.contains(x)));
                            if !fresh.is_empty() {
                                replay(&mut *acc, &aggs[ai], &fresh)?;
                            }
                        }
                        _ => acc.merge(&*t.accs[g * nagg + ai])?,
                    }
                    vals.push(acc.finish()?);
                }
            }
        }
        sink.push(key, &vals)?;
    }
    Ok(())
}

/// Row-at-a-time output for the spilled path.
///
/// [`emit`]'s own loop walks one table's arenas and is worth its
/// `split_at_mut`; this one takes rows from three places -- a bucket's own
/// groups, the groups it shares with the merged table, and the merged table's
/// tail -- so it takes them one at a time. Same block cutting and the same
/// nullable widening; the builders are rebuilt per block and nothing else
/// allocates.
struct Rows {
    tys: Vec<DataType>,
    b: Vec<ColumnBuilder>,
    n: usize,
    out: Vec<Block>,
}

impl Rows {
    fn new(tys: Vec<DataType>) -> Rows {
        let b = tys
            .iter()
            .map(|t| ColumnBuilder::with_capacity(t.clone(), BLOCK_SIZE))
            .collect();
        Rows { tys, b, n: 0, out: Vec::new() }
    }

    fn push(&mut self, keys: &[Value], vals: &[Value]) -> Result<()> {
        for (b, v) in self.b.iter_mut().zip(keys.iter().chain(vals)) {
            b.push_value(v)?;
        }
        self.n += 1;
        if self.n == BLOCK_SIZE {
            self.cut()?;
        }
        Ok(())
    }

    fn cut(&mut self) -> Result<()> {
        if self.n == 0 {
            return Ok(());
        }
        let next: Vec<ColumnBuilder> = self
            .tys
            .iter()
            .map(|t| ColumnBuilder::with_capacity(t.clone(), BLOCK_SIZE))
            .collect();
        let b = std::mem::replace(&mut self.b, next);
        self.out.push(finish_block(b)?);
        self.n = 0;
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<Block>> {
        self.cut()?;
        // Handed out back to front by every caller of `emit`; see `next`.
        Ok(self.out)
    }
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
    /// Open-addressing slots holding `(tag << 32) | (group + 1)`; 0 means
    /// empty (an occupied slot always has a non-zero low half).
    ///
    /// The tag is the *high* 32 bits of the key's hash -- disjoint from the low
    /// bits the bucket index uses -- and it is what keeps a probe to one random
    /// access instead of three. Equal keys hash equally, so a tag mismatch rules
    /// a group out without touching `hashes` or `keys`; a tag match still
    /// compares the key exactly, so a 1-in-4-billion false hit costs a compare
    /// and never an answer. The old `Vec<u32>` slot had to load `hashes[g]` and
    /// then `keys[g]` on every probe step, and at a 1/2 load factor half of all
    /// steps are collisions -- two dependent misses to rule out a group.
    slots: Vec<u64>,
    /// Cached hash per group, so growing the table never rehashes a key.
    /// Read by [`Groups::grow`], [`Groups::absorb`] and [`fold_bucket`]; the
    /// per-row probe reads the tag out of the slot instead.
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
    /// Rows this table refused after it filled the budget, hash-partitioned and
    /// waiting to be folded. `None` -- one null pointer per *table*, i.e. once
    /// per worker -- for every aggregate that fit, and for the serial operator,
    /// which folds its own partitions as it streams them. See [`Overflow`].
    over: Option<Box<Overflow>>,
}

/// A parallel worker's spilled rows, and enough of the query context to fold
/// them later.
///
/// The context is owned rather than borrowed because a `Groups` crosses out of
/// `pool::map` on its own and is folded much later by [`emit`], which has no
/// `&QueryContext` -- and widening `emit`'s signature would ripple into the
/// exchange. All three fields are an `Arc` or a `Copy`, so this is three words
/// per worker, paid only by a query that actually spilled.
struct Overflow {
    ctx: QueryContext,
    pending: Vec<Partition>,
}

/// An owned copy of a query's stop conditions: same cancel flag, same deadline,
/// same meter, no borrow.
fn own_ctx(ctx: &QueryContext) -> QueryContext {
    QueryContext { cancel: ctx.cancel.clone(), deadline: ctx.deadline, mem: ctx.mem.clone() }
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

// ------------------------------------------------------- integer group keys

/// What a single integer group column's raw lane means.
///
/// Exactly the arms of [`Column::value`](crate::types::Column::value) that
/// produce an integral `Value`, minus the two whose lane is not the value:
/// `Bool` (which narrows to 0/1) and `Decimal64` (whose `Value::hash` takes a
/// different branch). `Date` keeps a `u32` inside a `u64` lane, hence `MASK`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LaneKind {
    Int,
    UInt,
    Date,
    DateTime,
}

const DATE_MASK: u64 = 0xFFFF_FFFF;

/// The `Value` the general path would have built for this lane. Called once
/// per *group*, never per row.
#[inline]
fn lane_value(kind: LaneKind, lane: u64) -> Value {
    match kind {
        LaneKind::Int => Value::Int(lane as i64),
        LaneKind::UInt => Value::UInt(lane),
        LaneKind::Date => Value::Date(lane as u32),
        LaneKind::DateTime => Value::DateTime(lane as i64),
    }
}

/// The lane a stored group key was built from, or `None` if it was not built
/// from one.
///
/// This is the *exact* comparison the lane probe makes, so it has to reject
/// everything [`lane_hash`] does not describe: `Null`, strings, floats and
/// decimals, and -- the one that is easy to miss -- a `UInt` past `i64::MAX`,
/// whose `Value::hash` falls off the exact-integer branch into the float lane
/// and so does not hash as its lane. The block loop routes those rows to the
/// general path; this keeps the probe from matching them anyway.
///
/// All four integral variants map onto one lane space on purpose: `Int(5)`,
/// `UInt(5)`, `Date(5)` and `DateTime(5)` are one value to `Value`'s `Eq` and
/// one hash to its `Hash`, so they must be one group here too.
#[inline(always)]
fn lane_of(v: &Value) -> Option<u64> {
    match v {
        Value::Int(x) => Some(*x as u64),
        Value::UInt(x) if *x >> 63 == 0 => Some(*x),
        Value::Date(d) => Some(*d as u64),
        Value::DateTime(t) => Some(*t as u64),
        _ => None,
    }
}

/// `hash_values(&[Value::Int(x)])` with everything that does not depend on `x`
/// folded in at compile time.
///
/// `Value::hash` writes three words for an integral value: `2u8` (the numeric
/// equivalence class), `0u8` (the exact-integer discriminant) and then the
/// `i64`. The first two are constant, and *every* `i64` reaches them -- `x as
/// f64` is finite, has a zero fractional part and lies inside
/// `[i64::MIN as f64, i64::MAX as f64]` for all of them -- so a whole
/// `MixHasher`, an `as_f64`, a `fract()` and two of its three 128-bit folds
/// collapse to one multiply. `lane_hash_agrees_with_the_general_path` pins the equality this
/// entire fast path rests on; if `Value::hash` ever changes, that test fails
/// rather than the grouping silently splitting.
const LANE_SEED: u64 = {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    const fn cmum(a: u64, b: u64) -> u64 {
        let r = (a as u128).wrapping_mul(b as u128);
        (r as u64) ^ ((r >> 64) as u64)
    }
    cmum(cmum(0x243F_6A88_85A3_08D3 ^ 0x102, K) ^ 0x100, K)
};

#[inline(always)]
fn lane_hash(lane: u64) -> u64 {
    crate::common::mum(LANE_SEED ^ lane, 0x9E37_79B9_7F4A_7C15)
}

/// One integer group column, classified once per block.
///
/// `guard` is the largest lane that still hashes as its own integer; rows past
/// it (only reachable on a `UInt64` column) fall back to the general path.
struct LaneCol<'c> {
    kind: LaneKind,
    mask: u64,
    guard: u64,
    lanes: Lanes<'c>,
}

enum Lanes<'c> {
    U(&'c [u64]),
    I(&'c [i64]),
}

/// The bit pattern a lane vector's element contributes to a group key. `i64`
/// and `u64` differ only in how the same 64 bits are spelled.
trait Lane: Copy {
    fn bits(self) -> u64;
}
impl Lane for u64 {
    #[inline(always)]
    fn bits(self) -> u64 {
        self
    }
}
impl Lane for i64 {
    #[inline(always)]
    fn bits(self) -> u64 {
        self as u64
    }
}

/// Resolve a whole block's rows against the table, one integer lane at a time.
///
/// Generic over the lane vector *and* over whether the column has a null mask,
/// so both stay out of the row loop: four monomorphizations (doubled again by
/// the prefetch gate below) rather than branches per row on conditions that
/// are constant across a whole block. `zip` rather than an index so the bounds
/// check on `row_group` is proved once.
///
/// The NULL group is resolved at most once per block and then cached -- it is
/// one group, and re-probing it per null row would put the general path's
/// `Value` cost back on exactly the column that a nullable key is made of.
#[allow(clippy::too_many_arguments)]
fn lane_pass<L: Lane, const NULLS: bool>(
    v: &[L],
    nulls: Option<&BitSet>,
    mask: u64,
    guard: u64,
    kind: LaneKind,
    row_group: &mut [u32],
    groups: &mut Groups,
    protos: &[Box<dyn Accumulator>],
    aggs: &[BoundAgg],
) {
    // Past this the slot array is bigger than L2 and every probe is a miss the
    // hardware prefetcher cannot see coming, because the address is a hash.
    // Below it the table is resident and the prefetch is pure overhead --
    // hence a monomorphization rather than a branch. See [`lane_rows`].
    //
    // Measured interleaved against a switch that turned the prefetch off,
    // best-of-7 per side, 2M rows, serial, three runs each. **1M groups**
    // (`count()` 1.233 / 1.112 / 1.155, `count+sum` 1.131 / 1.077 / 1.157) --
    // ~1.15x, and the sign never flipped. **100k groups** (`count()` 1.050 /
    // 0.995 / 1.040) -- null, and a string-keyed control that does not reach
    // this loop at all read 1.004 / 0.977 / 1.048, which is this machine's
    // noise floor and the reason the 100k column is called null rather than
    // 3% up. So the gate is a floor and not a tuning knob: it costs nothing
    // where it fires early and pays where the table leaves cache.
    if groups.slots.len() >= PREFETCH_FROM {
        lane_rows::<L, NULLS, true>(v, nulls, mask, guard, kind, row_group, groups, protos, aggs)
    } else {
        lane_rows::<L, NULLS, false>(v, nulls, mask, guard, kind, row_group, groups, protos, aggs)
    }
}

/// 32768 slots is 256 KiB, which is where a probe stops hitting L2 on the
/// machines this was measured on.
const PREFETCH_FROM: usize = 1 << 15;

/// How far ahead to prefetch. Deep enough to cover a memory round trip at the
/// handful of cycles a probe costs when it hits, shallow enough that the lane
/// it reads to compute the address is one the sequential walk has already
/// pulled in.
const PREFETCH_AHEAD: usize = 12;

#[allow(clippy::too_many_arguments)]
fn lane_rows<L: Lane, const NULLS: bool, const PF: bool>(
    v: &[L],
    nulls: Option<&BitSet>,
    mask: u64,
    guard: u64,
    kind: LaneKind,
    row_group: &mut [u32],
    groups: &mut Groups,
    protos: &[Box<dyn Accumulator>],
    aggs: &[BoundAgg],
) {
    let n = v.len();
    let mut null_gid: Option<u32> = None;
    for (i, (slot, &x)) in row_group.iter_mut().zip(v).enumerate() {
        if PF {
            // Recomputing the future row's hash is one multiply -- cheaper
            // than the second pass and the 64 KiB scratch buffer that storing
            // it would need. `min` rather than a bounds test: the last few
            // rows just prefetch the last row's slot again.
            let j = (i + PREFETCH_AHEAD).min(n - 1);
            let h = lane_hash(v[j].bits() & mask);
            let at = h as usize & (groups.slots.len() - 1);
            crate::common::prefetch_read(&groups.slots[at] as *const u64 as *const u8);
        }
        if NULLS && nulls.is_some_and(|n| n.get(i)) {
            *slot = *null_gid.get_or_insert_with(|| {
                groups.find_or_insert(&[Value::Null], super::hash_null_key(), protos, aggs) as u32
            });
            continue;
        }
        let lane = x.bits() & mask;
        // Never taken outside a `UInt64` column holding a value past
        // `i64::MAX`, which hashes down `Value::hash`'s float branch and so
        // has to go the long way round. Predictable, and `guard` is a
        // register.
        *slot = if lane > guard {
            let key = [Value::UInt(lane)];
            groups.find_or_insert(&key, super::hash_values(&key), protos, aggs) as u32
        } else {
            groups.find_or_insert_lane(lane, lane_hash(lane), kind, protos, aggs) as u32
        };
    }
}

/// Group ids memoized by the *address* of the string that resolved them.
///
/// A granule decodes its dictionary once and then clones one `Arc` per row, so
/// a block of a string column holds one distinct pointer per distinct value
/// and repeats it thousands of times -- `country` over eight countries is
/// eight addresses and 8192 rows. Two rows whose `Arc` names the same
/// allocation are the same string, so a hit skips the hash, the probe and the
/// `memcmp` outright and costs one multiply and one compare.
///
/// **Cleared per block, and that is a correctness requirement, not tidiness.**
/// An address is only a witness for as long as the allocation it names is
/// alive; the block that owns these `Arc`s is dropped at the end of the
/// iteration, and the next block's decode is free to reuse the same address
/// for a different string. Within one block the `Arc`s are all held by the
/// column being walked, so the witness holds.
///
/// Open-addressed rather than direct-mapped because a direct map is only as
/// good as its worst pair: two of eight countries colliding would make every
/// row of both a miss, and the whole point is a table that low cardinality
/// makes vanish.
///
/// A miss costs a load and a store, so a column whose strings genuinely do not
/// repeat would pay for a memo it can never use. One block of that -- more
/// than a quarter of the rows missing -- turns it off for the rest of the
/// query, arena and all.
struct StrMemo {
    /// `(address, group + 1)`; a zero address is empty.
    slots: Vec<(usize, u32)>,
    /// Which slots this block wrote, so clearing costs the block's *distinct*
    /// strings and not the table's size. That is what lets the table be large
    /// enough to hold a thousand-value dictionary without a 64 KiB memset
    /// between blocks eating the win.
    used: Vec<u32>,
    /// Rows offered, and rows that had to hash, for the block in progress.
    rows: usize,
    miss: usize,
    on: bool,
}

/// 4096 entries. Sized to swallow a whole granule dictionary, not just a
/// handful of countries: `GROUP BY` over a thousand distinct strings is common
/// and, at 8192 rows a block, still repays a memo eight times over.
const MEMO_SLOTS: usize = 4096;
/// Stop admitting at half full; past that the probe chains cost more than the
/// hash they save.
const MEMO_FULL: usize = MEMO_SLOTS / 2;

impl Default for StrMemo {
    fn default() -> StrMemo {
        StrMemo { slots: Vec::new(), used: Vec::new(), rows: 0, miss: 0, on: true }
    }
}

impl StrMemo {
    /// Start a block of `rows`, judging the one just finished.
    fn reset(&mut self, rows: usize) {
        if !self.on {
            return;
        }
        if self.miss * 4 > self.rows {
            self.on = false;
            self.slots = Vec::new();
            self.used = Vec::new();
            return;
        }
        // Allocated on the first block of a string grouping and never again.
        if self.slots.is_empty() {
            self.slots.resize(MEMO_SLOTS, (0, 0));
            self.used.reserve(MEMO_FULL);
        }
        for &i in &self.used {
            self.slots[i as usize] = (0, 0);
        }
        self.used.clear();
        self.miss = 0;
        self.rows = rows;
    }

    #[inline(always)]
    fn at(p: usize) -> usize {
        // Multiply-shift: `Arc` allocations of one dictionary are a short
        // stride apart, so the low bits are the informative ones and a plain
        // mask would keep the wrong end. The shift is derived from
        // `MEMO_SLOTS` rather than written out -- a shift that yields fewer
        // bits than the table has slots crowds every start index into the
        // bottom of it, which is a table that still *works* (linear probing
        // wraps over the whole array) but degrades to long chains exactly
        // where the memo is supposed to pay, and nothing fails to say so.
        p.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (usize::BITS - MEMO_SLOTS.trailing_zeros())
    }

    #[inline(always)]
    fn get(&self, s: &std::sync::Arc<str>) -> Option<u32> {
        if !self.on {
            return None;
        }
        let p = s.as_ptr() as usize;
        let mut i = StrMemo::at(p);
        loop {
            let (q, g) = self.slots[i];
            if q == 0 {
                return None;
            }
            if q == p {
                return Some(g - 1);
            }
            i = (i + 1) & (MEMO_SLOTS - 1);
        }
    }

    /// Record the group a *missed* address resolved to. One call per distinct
    /// address per block, not per row.
    fn put(&mut self, s: &std::sync::Arc<str>, g: u32) {
        if !self.on {
            return;
        }
        self.miss += 1;
        if self.used.len() >= MEMO_FULL {
            return;
        }
        let p = s.as_ptr() as usize;
        let mut i = StrMemo::at(p);
        while self.slots[i].0 != 0 {
            i = (i + 1) & (MEMO_SLOTS - 1);
        }
        self.slots[i] = (p, g + 1);
        self.used.push(i as u32);
    }
}

/// Classify a single group column, or refuse it.
fn lane_col<'c>(c: &'c Column) -> Option<LaneCol<'c>> {
    use crate::types::ColumnData as D;
    let (kind, mask, guard) = match (&c.data, c.ty.base()) {
        (D::I64(_), DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64) => {
            (LaneKind::Int, u64::MAX, u64::MAX)
        }
        // A `DateTime` normally rides a `U64` lane; this arm exists because
        // `Column::value` has one, and the two must agree on every shape a
        // column can be in or the fast path would answer a different `Value`
        // than the general path for the same row.
        (D::I64(_), DataType::DateTime) => (LaneKind::DateTime, u64::MAX, u64::MAX),
        (D::U64(_), DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64) => {
            (LaneKind::UInt, u64::MAX, i64::MAX as u64)
        }
        (D::U64(_), DataType::Date) => (LaneKind::Date, DATE_MASK, u64::MAX),
        (D::U64(_), DataType::DateTime) => (LaneKind::DateTime, u64::MAX, u64::MAX),
        _ => return None,
    };
    let lanes = match &c.data {
        D::U64(v) => Lanes::U(v),
        D::I64(v) => Lanes::I(v),
        _ => return None,
    };
    Some(LaneCol { kind, mask, guard, lanes })
}

impl Groups {
    fn new(nkeys: usize, aggs: &[BoundAgg]) -> Groups {
        Groups {
            nkeys,
            has_distinct: aggs.iter().any(|a| a.distinct),
            slots: vec![0; 64],
            ..Default::default()
        }
    }

    /// The half of a slot that identifies the group, and the half that rules
    /// one out. Split out so the three probe loops cannot disagree.
    #[inline(always)]
    fn tag(h: u64) -> u64 {
        h & 0xFFFF_FFFF_0000_0000
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
        let Groups { nkeys, keys, hashes, len, accs, mut seen, has_distinct, over, .. } = other;
        debug_assert_eq!(nkeys, self.nkeys);
        debug_assert_eq!(has_distinct, self.has_distinct);
        // Concatenation, not a merge: every worker cut on the same mask at the
        // same level (see `Partitions::arm`), so partition `p` of one worker
        // and partition `p` of another describe the same key range and no other
        // partition holds a key either of them holds. Worker order is
        // preserved, which is what keeps `any`/`groupArray` reading the earlier
        // slice first when a partition is folded.
        if let Some(o) = over {
            match &mut self.over {
                Some(mine) => mine.pending.extend(o.pending),
                None => self.over = Some(o),
            }
        }
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
            + self.slots.capacity() * size_of::<u64>()
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
        let (tag, mut i) = (Groups::tag(h), h as usize & mask);
        loop {
            let s = self.slots[i];
            if s == 0 {
                return None;
            }
            if Groups::tag(s) == tag {
                let g = s as u32 as usize - 1;
                if self.key_of(g) == key {
                    return Some(g);
                }
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
        let (tag, mut i) = (Groups::tag(h), h as usize & mask);
        loop {
            let s = self.slots[i];
            if s == 0 {
                let g = self.len;
                self.slots[i] = tag | (g as u64 + 1);
                self.keys.extend_from_slice(key);
                self.hashes.push(h);
                self.len += 1;
                return (g, true);
            }
            // Compare the slot's own hash tag first: a mismatch rules the group
            // out without touching either arena, which is the expensive part.
            if Groups::tag(s) == tag {
                let g = s as u32 as usize - 1;
                if self.key_of(g) == key {
                    return (g, false);
                }
            }
            i = (i + 1) & mask;
        }
    }

    /// Single integer key, probed straight off the block's lane.
    ///
    /// The general path builds a `Value` per row, hashes it through
    /// `MixHasher` (three folds, a float round-trip and a `fract()`) and then
    /// compares a `&[Value]` slice. Here the hash is one multiply
    /// ([`lane_hash`]), the probe reads the tag out of the slot, and the
    /// compare is one `u64`; nothing but a genuinely new group ever
    /// materializes a `Value`.
    #[inline]
    fn find_or_insert_lane(
        &mut self,
        lane: u64,
        h: u64,
        kind: LaneKind,
        protos: &[Box<dyn Accumulator>],
        aggs: &[BoundAgg],
    ) -> usize {
        debug_assert_eq!(self.nkeys, 1);
        if (self.len + 1) * 2 >= self.slots.len() {
            self.grow();
        }
        let mask = self.slots.len() - 1;
        let (tag, mut i) = (Groups::tag(h), h as usize & mask);
        loop {
            let s = self.slots[i];
            if s == 0 {
                let g = self.len;
                self.slots[i] = tag | (g as u64 + 1);
                self.keys.push(lane_value(kind, lane));
                self.hashes.push(h);
                self.len += 1;
                self.push_group(protos, aggs);
                return g;
            }
            if Groups::tag(s) == tag {
                let g = s as u32 as usize - 1;
                // Exact, not a tag: `lane_of` is `None` for every key this
                // lane space does not describe, so a `Null` group (or a
                // `UInt` past `i64::MAX`) sharing a tag can never be matched.
                if lane_of(&self.keys[g]) == Some(lane) {
                    return g;
                }
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
        let (tag, mut i) = (Groups::tag(h), h as usize & mask);
        loop {
            let s = self.slots[i];
            if s == 0 {
                let g = self.len;
                self.slots[i] = tag | (g as u64 + 1);
                self.keys.push(Value::Str(key.clone()));
                self.hashes.push(h);
                self.len += 1;
                self.push_group(protos, aggs);
                return g;
            }
            // The tag rules out a colliding group before the `Arc` is chased:
            // on a string key the exact compare is two dependent loads and a
            // `memcmp`, so it is the one this most wants not to reach.
            if Groups::tag(s) == tag {
                let g = s as u32 as usize - 1;
                if self.keys[g].as_str() == Some(&**key) {
                    return g;
                }
            }
            i = (i + 1) & mask;
        }
    }

    fn grow(&mut self) {
        let cap = (self.slots.len() * 2).max(64);
        let mask = cap - 1;
        let mut slots = vec![0u64; cap];
        for g in 0..self.len {
            let h = self.hashes[g];
            let mut i = h as usize & mask;
            while slots[i] != 0 {
                i = (i + 1) & mask;
            }
            slots[i] = Groups::tag(h) | (g as u64 + 1);
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
    /// Table size past which this table freezes even though its own `grow_to`
    /// still succeeds; `0` disables it. Only a parallel worker sets it -- see
    /// [`worker_ceiling`].
    soft: usize,
}

/// One spilled partition, and its share of the directory's lifetime.
///
/// `Clone` is a path, a schema and two refcount bumps, and it exists so
/// [`buckets`] can regroup a borrowed `pending` list without taking the table
/// apart -- [`emit`] only ever has `&Groups`.
#[derive(Clone)]
pub(crate) struct Partition {
    path: std::path::PathBuf,
    schema: Schema,
    level: u32,
    /// Which bucket of the level's hash split this is. Carried because a
    /// parallel aggregate produces one file per (worker, bucket) and the files
    /// of one bucket have to be folded *together*: two workers can both have
    /// spilled the same key, and folding their files separately would create
    /// the group twice and emit it twice.
    idx: u32,
    /// Shared, because a directory holds every partition cut at one level and
    /// must outlive the last of them -- however early the query stops reading.
    _dir: std::sync::Arc<spill::SpillDir>,
}

impl Partitions {
    pub(crate) fn new(level: u32, soft: usize) -> Partitions {
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
            soft,
        }
    }

    /// Fix the partition count and buffer size. Called once, at the freeze.
    ///
    /// The two numbers come from different places on purpose. The partition
    /// **count** is a function of the query's whole budget and of nothing else,
    /// because every worker of a parallel aggregate has to cut on the same
    /// mask: if worker A's bucket 3 and worker B's bucket 3 described different
    /// key ranges, a key could land in two folds and be emitted twice. The
    /// per-partition write **buffer** comes from this operator's own share of
    /// that budget, which legitimately differs per worker and is only a sizing
    /// question -- sixty-four 64 KiB buffers is 4 MiB of write-behind, free
    /// against 8 GiB and absurd against 256 KiB, and fourteen workers each
    /// claiming the query's whole quarter is how a bounded query becomes a 3.5x
    /// overshoot.
    ///
    /// The count used to be clamped by the rows of the block that triggered the
    /// freeze, which made it depend on *where in its input* a worker happened
    /// to run out. It bought nothing even serially: a writer is created on
    /// first use, so a partition that never sees a row never opens a file.
    fn arm(&mut self, ctx: &QueryContext, share: usize) {
        if self.mask != 0 {
            return;
        }
        // A quarter of the budget for write-behind; the group table -- which
        // has just proved it wants everything -- keeps the rest.
        let cap = ((ctx.mem.limit().max(0) as usize) / 4).max(8 << 10);
        let n = (cap / (32 << 10)).clamp(2, 64).next_power_of_two().min(64);
        self.mask = n as u64 - 1;
        self.flush_at = ((share / 4).max(8 << 10) / n).clamp(4 << 10, 64 << 10);
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
        for (i, w) in writers.into_iter().enumerate() {
            if let Some(w) = w {
                paths.push((i as u32, w.finish()?));
            }
        }
        let dir = std::sync::Arc::new(dir);
        Ok(paths
            .into_iter()
            .map(|(idx, path)| Partition {
                path,
                schema: schema.clone(),
                level: level + 1,
                idx,
                _dir: dir.clone(),
            })
            .collect())
    }
}

/// Reads one spilled partition back as an operator, so the recursive pass is
/// the *same* `accumulate_into` and not a second implementation of it.
///
/// A partition can be several files -- one per worker of a parallel aggregate
/// -- and they are read back **in worker order**, one at a time. In order,
/// because `any`, `anyLast`, `argMin`'s tie-break and `groupArray`'s element
/// order are all defined against feed order; one at a time, because a partition
/// of a fourteen-way `GROUP BY` would otherwise hold fourteen read buffers open
/// for a single sequential pass.
struct SpillScan {
    schema: Schema,
    /// Reversed, so `next` pops.
    rest: Vec<std::path::PathBuf>,
    cur: Option<spill::RunReader>,
}

impl SpillScan {
    fn open(schema: Schema, paths: &[std::path::PathBuf]) -> SpillScan {
        SpillScan { schema, rest: paths.iter().rev().cloned().collect(), cur: None }
    }
}

impl Operator for SpillScan {
    fn schema(&self) -> &Schema {
        &self.schema
    }
    fn next(&mut self) -> Result<Option<Block>> {
        loop {
            let r = match &mut self.cur {
                Some(r) => r,
                None => {
                    let Some(p) = self.rest.pop() else { return Ok(None) };
                    self.cur.insert(spill::RunReader::open(&p, self.schema.clone())?)
                }
            };
            match r.next()? {
                Some(b) => return Ok(Some(b)),
                None => self.cur = None,
            }
        }
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
        let mut input: Box<dyn Operator> =
            Box::new(SpillScan::open(p.schema.clone(), std::slice::from_ref(&p.path)));
        let mut parts = Partitions::new(p.level, 0);
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

    // -------------------------------------------------- the integer lane key

    /// The equality the whole lane fast path rests on.
    ///
    /// If `Value::hash` ever stops folding an integral value as
    /// `2u8, 0u8, i64`, this fails -- which is the point. The alternative to
    /// pinning it here is a `GROUP BY` that quietly puts one key in two groups
    /// on whichever build changed it.
    #[test]
    fn lane_hash_agrees_with_the_general_path() {
        let mut lanes: Vec<u64> = vec![
            0,
            1,
            2,
            7,
            255,
            65_535,
            1 << 31,
            (1u64 << 62) - 1,
            i64::MAX as u64,
            (-1i64) as u64,
            (-2i64) as u64,
            (i64::MIN) as u64,
        ];
        for i in 0..512u64 {
            lanes.push(crate::common::splitmix64(i) >> 1);
        }
        for &l in &lanes {
            for kind in [LaneKind::Int, LaneKind::UInt, LaneKind::Date, LaneKind::DateTime] {
                // The two narrow kinds only ever see lanes their `Value` can
                // hold; `lane_col` masks `Date` and the column supplies the
                // rest.
                let l = match kind {
                    LaneKind::Date => l & DATE_MASK,
                    LaneKind::UInt => l & (i64::MAX as u64),
                    _ => l,
                };
                let v = lane_value(kind, l);
                assert_eq!(
                    lane_hash(l),
                    super::super::hash_values(std::slice::from_ref(&v)),
                    "{kind:?} lane {l:#x} ({v})"
                );
                assert_eq!(lane_of(&v), Some(l), "{kind:?} lane {l:#x} does not round-trip");
            }
        }
    }

    /// Everything the lane probe must refuse, because its hash is not
    /// `lane_hash` and a match would merge two different keys.
    #[test]
    fn lane_of_refuses_what_it_does_not_describe() {
        for v in [
            Value::Null,
            Value::str("7"),
            Value::Float(7.0),
            Value::Decimal(700, 2),
            Value::Bool(true),
            Value::UInt(i64::MAX as u64 + 1),
            Value::UInt(u64::MAX),
        ] {
            assert_eq!(lane_of(&v), None, "{v} must not be reachable from a lane probe");
        }
        // And the one that would otherwise be a silent merge: a `UInt` past
        // `i64::MAX` hashes down `Value::hash`'s float branch, so its lane and
        // its hash disagree.
        let big = Value::UInt(1u64 << 63);
        assert_ne!(lane_hash(1u64 << 63), super::super::hash_values(&[big]));
    }

    /// A slot carries both halves, and neither reads the other's bits.
    #[test]
    fn a_slot_holds_a_tag_and_a_group_without_overlapping() {
        for h in [0u64, u64::MAX, 0x1234_5678_9ABC_DEF0] {
            for g in [0u64, 1, 4095, u32::MAX as u64 - 1] {
                let s = Groups::tag(h) | (g + 1);
                assert_eq!(Groups::tag(s), Groups::tag(h));
                assert_eq!(s as u32 as u64 - 1, g);
                assert_ne!(s, 0, "an occupied slot must never read as empty");
            }
        }
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
