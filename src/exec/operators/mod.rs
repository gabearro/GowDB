//! The physical operators, and the pull pipeline they form.
//!
//! Every operator is a `next() -> Option<Block>` iterator over batches. Pull
//! (Volcano) rather than push, but *vectorized*: one `next()` moves up to
//! [`BLOCK_SIZE`] rows, so the per-call virtual dispatch is amortized over
//! thousands of values and the inner loops stay flat slice walks. This is the
//! shape that lets a filter compile to a branchless compare over `&[i64]`
//! instead of an interpreter step per tuple.
//!
//! ## Where a `SELECT`'s time actually goes
//!
//! `Table::scan_fold` and this pipeline read the same parts, and the same work
//! through the two doors used to differ by 8x. That gap is not one thing, and
//! guessing at it produced two wrong answers before it was taken apart. The
//! decomposition below is one process, 2M rows, best-of-15, every stage driven
//! **serially on one thread** so thread scheduling cannot colour it:
//!
//! ```text
//!   storage: scan_each of one Int64 column, body counts rows   1.62 ms
//!   Scan operator, same column, same snapshot                  3.84 ms   2.4x
//!   Scan operator, empty projection (the count(*) shape)       0.04 ms
//!   Scan + Project, one bare column reference                 +1.36 ms  <- was here
//!   Scan + Filter, predicate that keeps every row             +3.16 ms
//!   Scan + Filter, predicate that keeps half                  +4.51 ms
//!   parse + bind + optimize + lower, per query                 0.006-0.06 ms
//!   build the operator tree (serial or exchange), per query     0.001-0.01 ms
//! ```
//!
//! Four things fall out of that, and they are worth more than the ratio was:
//!
//!   * **Planning and building are free.** Under 1% of every query measured,
//!     including the 1.3 ms ones. Nobody needs to cache a plan.
//!   * **The per-block machinery is free.** An empty-projection scan moves 2M
//!     rows in 40 us; that is 244 `Box<dyn Operator>` dispatches, 244
//!     `QueryContext::check`s and 244 blocks built and dropped. The relaxed
//!     atomic per block really is unmeasurable, as the note below claims.
//!   * **`Scan` costs 2.4x the storage decode it wraps**, and that is per
//!     *granule*, not per block: it decodes each granule into a fresh `Block`
//!     and then `extend`s it into an accumulator, so every value is written
//!     twice and every column allocates once per 1024 rows. `ScanScratch`
//!     next door decodes straight into a reused, L1-sized batch. That is the
//!     single largest remaining item and it lives in `scan.rs`/`exchange.rs`.
//!   * **Predicate evaluation costs more than the scan under it**, for the
//!     reason recorded in [`filter`]: `expr` materializes four full buffers per
//!     block to answer `col > lit`.
//!
//! The `+1.36 ms` line is the one this file could reach, and it is gone; see
//! [`project`].
//!
//! ## Where the decisions are, and are not
//!
//! There are none in this file. [`build`] lowers the logical plan through
//! [`crate::planner::physical`] and then maps the result 1:1 onto operators,
//! so which access path a scan gets and whether a sort is bounded are settled
//! before anything here runs. That separation is what made index selection
//! possible at all: `build` used to be the only place a decision *could* have
//! gone, and a decision made while constructing operators is a decision
//! `EXPLAIN` cannot show you.
//!
//! ## Streaming vs. blocking
//!
//! `Scan`, `IndexLookup`, `Filter`, `Project`, `Limit`, `LimitBy`, `Distinct`,
//! `Union` and `Values` are **streaming**: constant memory, first row out
//! before the last row in. `Sort`, `Aggregate` and `Join` are **blocking** by
//! nature -- they cannot answer until they have seen every input row -- so they
//! materialize once on the first `next()` and then hand out precomputed blocks.
//! Doing the materialization lazily keeps [`build`] cheap, which matters
//! because `EXPLAIN` and the binder's own checks construct pipelines they never
//! run. (`IndexLookup` resolves its key set on the first `next()` for the same
//! reason; the set is bounded by the query text, not by the table.)
//!
//! ## Borrowing
//!
//! Operators borrow the plan (`&'a LogicalPlan`, reached through the physical
//! plan, which borrows from it rather than owning a copy) and the catalog
//! (`&'a Catalog`) immutably for the whole execution. The session flushes
//! every table's write buffer *before* planning, so a scan sees only immutable
//! parts and never needs `&mut Table`. That is what makes the whole pipeline
//! shareable and, later, parallelizable without a lock.
//!
//! ## Counters
//!
//! [`ScanStats`] rides up the tree through `Operator::stats`, so a test (or
//! `SELECT`'s own summary line) can assert that zone-map pruning actually
//! fired rather than merely being implemented. Pruning that silently stops
//! working is the single easiest performance regression to ship in an engine
//! like this one, so it is measured, not assumed.
//!
//! ## Stopping a query
//!
//! [`QueryContext`] rides down the tree instead: `build` takes one and every
//! operator keeps a `&QueryContext`. It carries the three things that turn
//! "the process dies" into "the query returns an error" -- a cancel flag any
//! thread may flip, an optional deadline, and the [`MemTracker`] the blocking
//! operators charge their unbounded state to.
//!
//! Both checks are made **once per block**, never per row: a relaxed atomic
//! load amortized over 8192 rows is unmeasurable, and the deadline's
//! `Instant::now` is behind an `Option` that is `None` for every query that did
//! not ask for one. Measured against the same build with the checkpoint
//! switched off, interleaved best-of-9 over 2M rows: `count()` 6.41ms vs
//! 6.43ms, `sum(bytes) WHERE latency > 500` 18.21ms vs 19.74ms, `GROUP BY
//! country` 29.36ms vs 29.54ms. Every one of those is inside the run-to-run
//! swing, and the sign is not even consistent -- which is the answer: 244
//! relaxed loads cannot show up against 2M rows of work.

pub mod aggregate;
pub mod distinct;
pub mod filter;
pub mod join;
pub mod limit;
pub mod project;
pub mod scan;
pub mod sort;
pub mod union;
pub mod values;
pub mod window;

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crate::catalog::Catalog;
use crate::common::{mum, Error, Result, BLOCK_SIZE};
use crate::planner::logical::LogicalPlan;
use crate::planner::physical::{self, PhysicalPlan};
use crate::types::{Block, Schema, Value};

/// A pull-based source of [`Block`]s.
pub trait Operator {
    fn schema(&self) -> &Schema;
    /// The next batch, or `None` at end of stream. A returned block may have
    /// fewer than [`BLOCK_SIZE`] rows; only `None` means "finished".
    fn next(&mut self) -> Result<Option<Block>>;
    /// Scan counters for this operator *and its inputs*. Non-leaf operators
    /// forward; only [`scan::Scan`] and [`scan::IndexLookup`] produce anything.
    fn stats(&self) -> ScanStats {
        ScanStats::default()
    }
}

/// Access-path counters for a query, aggregated over every scan in the plan.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanStats {
    /// Granules never decoded: skipped outright by a zone-map test on a
    /// [`scan::Scan`], or never reached at all by a [`scan::IndexLookup`].
    pub granules_pruned: u64,
    /// Granules actually decoded.
    pub granules_read: u64,
    /// Rows decoded, before the scan's own predicates ran.
    pub rows_read: u64,
}

impl ScanStats {
    pub fn merge(&mut self, o: &ScanStats) {
        self.granules_pruned += o.granules_pruned;
        self.granules_read += o.granules_read;
        self.rows_read += o.rows_read;
    }

    /// Fraction of candidate granules the zone maps eliminated. `0.0` when
    /// there was nothing to prune.
    pub fn prune_ratio(&self) -> f64 {
        let total = self.granules_pruned + self.granules_read;
        if total == 0 {
            0.0
        } else {
            self.granules_pruned as f64 / total as f64
        }
    }
}

// ------------------------------------------------------- resource governance

/// Ceiling on one query's *intermediate* state when the caller sets none.
///
/// A backstop, not a target. Before this existed, a `GROUP BY` whose group
/// table outgrew RAM took the whole process down -- there was no error to
/// return because nothing was counting. 8 GiB is deliberately generous: it has
/// to sit above every legitimate query this engine is fast at (a 4M-group
/// aggregate holds ~1.6 GB) while still being far enough below a workstation's
/// RAM that the failure is a message rather than a swap storm. A session that
/// knows better should pass its own budget; see [`QueryContext::with_budget`].
pub const DEFAULT_MEM_BUDGET: i64 = 8 << 30;

/// The engine's memory meter: one counter, one ceiling, nothing else.
///
/// Deliberately not a per-operator arena or an allocator hook. Everything that
/// grows without bound in this engine is one of four buffers (group table,
/// sort buffer, join build side, join probe fan-out), and each of them can
/// state its own footprint far more cheaply than an allocator shim could
/// measure it. What matters is that the count is charged **once per block**,
/// so the atomic never lands in a per-row or per-group loop.
///
/// Shared behind an `Arc` so a future exchange operator can charge several
/// worker threads against one budget without changing any call site.
#[derive(Debug)]
pub struct MemTracker {
    used: AtomicI64,
    limit: i64,
}

impl MemTracker {
    pub fn with_limit(limit: i64) -> Arc<MemTracker> {
        Arc::new(MemTracker { used: AtomicI64::new(0), limit })
    }

    /// No ceiling. For callers that genuinely want the old behaviour.
    pub fn unlimited() -> Arc<MemTracker> {
        MemTracker::with_limit(i64::MAX)
    }

    pub fn used(&self) -> i64 {
        self.used.load(Ordering::Relaxed)
    }

    pub fn limit(&self) -> i64 {
        self.limit
    }

    /// Charge `bytes` against the budget, naming `what` is being built.
    ///
    /// Adds first and backs out on failure rather than compare-and-swapping in
    /// a loop: the contended case is a budget that is about to fail anyway, so
    /// the uncontended path is what deserves the single instruction.
    pub fn reserve(&self, bytes: usize, what: &str) -> Result<()> {
        let n = bytes as i64;
        let now = self.used.fetch_add(n, Ordering::Relaxed) + n;
        if now > self.limit {
            self.used.fetch_sub(n, Ordering::Relaxed);
            return Err(over_budget(what, bytes, now - n, self.limit));
        }
        Ok(())
    }

    pub fn release(&self, bytes: usize) {
        self.used.fetch_sub(bytes as i64, Ordering::Relaxed);
    }
}

/// An OOM message a user can act on: what was growing, how much more it
/// wanted, what was already held, and the ceiling it hit.
#[cold]
#[inline(never)]
fn over_budget(what: &str, want: usize, held: i64, limit: i64) -> Error {
    Error::exec(format!(
        "query exceeded its memory budget of {} while building {what}: it wanted \
         another {} on top of the {} already held. Raise the budget, add a \
         filter, or reduce the number of groups.",
        human(limit),
        human(want as i64),
        human(held)
    ))
}

fn human(b: i64) -> String {
    for (name, unit) in [("GiB", 1i64 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)] {
        if b.abs() >= unit {
            return format!("{:.2} {name}", b as f64 / unit as f64);
        }
    }
    format!("{b} B")
}

/// Everything an operator needs in order to stop.
///
/// Passed as `&QueryContext` and stored as a shared reference on each
/// operator: threading it must not cost a refcount bump per block, and it does
/// not -- the `Arc`s inside are cloned once per *query*, at build time.
#[derive(Debug)]
pub struct QueryContext {
    /// Flipped by whoever wants the query to stop -- another session, a
    /// signal handler, a client disconnect. Only ever read relaxed: we do not
    /// need to see it the instant it is set, only within one block.
    pub cancel: Arc<AtomicBool>,
    /// Wall-clock stop. `None` for "no deadline", which is what keeps
    /// `Instant::now` out of the loop for queries that did not ask for one.
    pub deadline: Option<Instant>,
    pub mem: Arc<MemTracker>,
}

impl Default for QueryContext {
    fn default() -> Self {
        QueryContext::new()
    }
}

impl QueryContext {
    /// Default budget, no deadline, not cancelled.
    pub fn new() -> QueryContext {
        QueryContext {
            cancel: Arc::new(AtomicBool::new(false)),
            deadline: None,
            mem: MemTracker::with_limit(DEFAULT_MEM_BUDGET),
        }
    }

    pub fn with_budget(bytes: i64) -> QueryContext {
        QueryContext { mem: MemTracker::with_limit(bytes), ..QueryContext::new() }
    }

    /// No budget and no deadline: the pre-governance behaviour, for callers
    /// that would rather be killed by the OS than told no.
    pub fn unlimited() -> QueryContext {
        QueryContext { mem: MemTracker::unlimited(), ..QueryContext::new() }
    }

    pub fn deadline_in(mut self, d: Duration) -> QueryContext {
        self.deadline = Instant::now().checked_add(d);
        self
    }

    /// A handle another thread can flip to stop this query.
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    pub fn stop(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// The per-block checkpoint. One relaxed load plus, only when a deadline
    /// exists, one clock read. Both branches are perfectly predicted, so the
    /// cost in a scan loop is below measurement noise.
    #[inline]
    pub fn check(&self) -> Result<()> {
        if self.cancel.load(Ordering::Relaxed)
            || self.deadline.is_some_and(|d| Instant::now() >= d)
        {
            return Err(self.stopped());
        }
        Ok(())
    }

    #[cold]
    #[inline(never)]
    fn stopped(&self) -> Error {
        if self.cancel.load(Ordering::Relaxed) {
            Error::exec("query cancelled")
        } else {
            Error::exec("query exceeded its deadline")
        }
    }
}

/// The context used by callers that do not supply one. Static so the common
/// `execute(plan, catalog)` path allocates nothing extra and every operator
/// can hold a plain reference.
fn ambient_context() -> &'static QueryContext {
    static C: OnceLock<QueryContext> = OnceLock::new();
    C.get_or_init(QueryContext::new)
}

/// An RAII reservation held by one operator for its lifetime.
///
/// The point of the guard is the `Drop`: a `GROUP BY` that fails halfway --
/// or a pipeline dropped early because a `LIMIT` upstream was satisfied --
/// must give its budget back, and every error path in these operators is a
/// `?`. `grow_to` is the only call in a loop and it is a no-op unless the
/// footprint actually grew, so a steady-state operator pays zero atomics per
/// block.
pub struct MemGuard {
    mem: Arc<MemTracker>,
    held: usize,
    what: &'static str,
}

impl MemGuard {
    pub fn new(ctx: &QueryContext, what: &'static str) -> MemGuard {
        MemGuard { mem: ctx.mem.clone(), held: 0, what }
    }

    /// Raise the reservation to `bytes` total. One atomic when the footprint
    /// grew, none when it did not.
    #[inline]
    pub fn grow_to(&mut self, bytes: usize) -> Result<()> {
        if bytes <= self.held {
            return Ok(());
        }
        self.mem.reserve(bytes - self.held, self.what)?;
        self.held = bytes;
        Ok(())
    }

    pub fn held(&self) -> usize {
        self.held
    }
}

impl Drop for MemGuard {
    fn drop(&mut self) {
        self.mem.release(self.held);
    }
}

/// Compile a logical plan into a runnable pipeline.
///
/// Lowering runs first and unconditionally: [`physical::lower`] is where the
/// access-path and top-K decisions are made, and this function is once again
/// what it was before those decisions existed -- a 1:1 structural mapping,
/// only now from the *physical* plan.
///
/// The physical plan can be a temporary because a `PhysicalPlan<'a>` borrows
/// its expressions and schemas from the `&'a LogicalPlan`, not the other way
/// round: no operator ever borrows from the physical plan itself, so it is free
/// to die at the end of this call.
pub fn build<'a>(
    plan: &'a LogicalPlan,
    catalog: &'a Catalog,
    ctx: &'a QueryContext,
) -> Result<Box<dyn Operator + 'a>> {
    build_physical(physical::lower(plan, catalog)?, catalog, ctx)
}

/// [`build`] for a plan that has already been lowered.
///
/// Takes the plan **by value** so an [`IndexPath`](physical::IndexPath)'s key
/// vector moves into the operator instead of being cloned; an `IN` list folded
/// out of a subquery can be long, and copying it once per query for no reason
/// is exactly the kind of allocation this engine does not make.
pub fn build_physical<'a>(
    plan: PhysicalPlan<'a>,
    catalog: &'a Catalog,
    ctx: &'a QueryContext,
) -> Result<Box<dyn Operator + 'a>> {
    Ok(match plan {
        PhysicalPlan::Scan(s) => Box::new(scan::Scan::new(s, catalog)?),
        PhysicalPlan::IndexLookup(path) => Box::new(scan::IndexLookup::new(*path, catalog)?),
        PhysicalPlan::MetaAggregate(path) => Box::new(scan::MetaAggregate::new(*path, catalog)?),
        PhysicalPlan::Filter { input, predicate } => {
            Box::new(filter::Filter::new(build_physical(*input, catalog, ctx)?, predicate))
        }
        PhysicalPlan::Project { input, exprs, schema } => Box::new(project::Project::new(
            build_physical(*input, catalog, ctx)?,
            exprs,
            schema,
        )),
        PhysicalPlan::Aggregate { input, group, aggs, schema } => {
            Box::new(aggregate::Aggregate::new(
                build_physical(*input, catalog, ctx)?,
                group,
                aggs,
                schema,
                ctx,
            )?)
        }
        PhysicalPlan::Sort { input, keys, fetch } => {
            let inner = build_physical(*input, catalog, ctx)?;
            Box::new(match fetch {
                Some(k) => sort::Sort::top_k(inner, keys, k, ctx),
                None => sort::Sort::new(inner, keys, ctx),
            })
        }
        PhysicalPlan::Window { input, node } => {
            Box::new(window::Window::new(build_physical(*input, catalog, ctx)?, node, ctx))
        }
        PhysicalPlan::Limit { input, limit, offset } => Box::new(limit::Limit::new(
            build_physical(*input, catalog, ctx)?,
            limit,
            offset,
            ctx,
        )),
        PhysicalPlan::LimitBy { input, limit, keys } => Box::new(limit::LimitBy::new(
            build_physical(*input, catalog, ctx)?,
            limit,
            keys,
            ctx,
        )),
        PhysicalPlan::Distinct { input } => {
            Box::new(distinct::Distinct::new(build_physical(*input, catalog, ctx)?))
        }
        PhysicalPlan::Join { left, right, op, on, residual, schema } => Box::new(join::Join::new(
            build_physical(*left, catalog, ctx)?,
            build_physical(*right, catalog, ctx)?,
            op,
            on,
            residual,
            schema,
            ctx,
        )),
        // `build_set` owns the branch construction because the set operators'
        // fields are private to that module, so it re-lowers each branch
        // through `build`. See the `logical` field on `PhysicalPlan::Union`.
        PhysicalPlan::Union { logical, op, all, schema, .. } => {
            union::build_set(logical, op, all, schema, catalog, ctx)?
        }
        PhysicalPlan::Values { rows, schema } => Box::new(values::Values::new(rows, schema)),
        PhysicalPlan::Empty { schema } => Box::new(values::Empty::new(schema)),
        // The serial builder cannot honour a fleet, so it builds the subtree
        // and drops the request. Only `exchange::build` runs an `Exchange`, and
        // `physical::lower` only emits one where that builder will see it --
        // this arm is what the *other* callers (`INSERT ... SELECT`, a `UNION`
        // branch) fall into, and dropping the node there is why their plans are
        // lowered with parallelism switched off rather than lied about.
        PhysicalPlan::Exchange { input, .. } => build_physical(*input, catalog, ctx)?,
    })
}

/// Run a plan to completion, collecting every non-empty batch.
pub fn execute<'a>(plan: &'a LogicalPlan, catalog: &'a Catalog) -> Result<Vec<Block>> {
    Ok(execute_with_stats(plan, catalog)?.0)
}

/// [`execute`], plus the access-path counters the query actually paid.
pub fn execute_with_stats<'a>(
    plan: &'a LogicalPlan,
    catalog: &'a Catalog,
) -> Result<(Vec<Block>, ScanStats)> {
    execute_ctx(plan, catalog, ambient_context())
}

/// [`execute_with_stats`] under a caller-supplied budget, deadline and cancel
/// flag. This is the entry point a session with settings should call.
pub fn execute_ctx<'a>(
    plan: &'a LogicalPlan,
    catalog: &'a Catalog,
    ctx: &'a QueryContext,
) -> Result<(Vec<Block>, ScanStats)> {
    let mut op = build(plan, catalog, ctx)?;
    let mut out = Vec::new();
    // The one checkpoint every query passes through, including the streaming
    // pipelines that never build a blocking operator.
    while let Some(b) = {
        ctx.check()?;
        op.next()?
    } {
        if b.rows() > 0 {
            out.push(b);
        }
    }
    let stats = op.stats();
    Ok((out, stats))
}

// -------------------------------------------------------------- group keys

/// A hash key built from a tuple of [`Value`]s: group-by keys, `DISTINCT`
/// rows, `LIMIT n BY` keys and join keys are all this shape.
///
/// The custom `Hash` exists because [`crate::common::FastHasher`] is an
/// *identity-ish* hasher -- `write_u64` overwrites its state rather than
/// chaining it, which is exactly right for the already-mixed `u64` keys it was
/// built for and exactly wrong for a tuple, where it would reduce the hash of
/// `(a, b)` to the hash of `b` alone. Every `(a, *)` would then land in one
/// bucket. Folding the components through [`mum`] here first, and emitting a
/// single `write_u64`, restores full-tuple discrimination while keeping the
/// map's own hashing free.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GroupKey(pub Vec<Value>);

impl Hash for GroupKey {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(hash_values(&self.0));
    }
}

/// Chaining mixer. Delegating to `Value`'s own `Hash` keeps the result
/// consistent with `Value`'s `Eq` (numerics that compare equal across
/// representations must hash equal), while the chaining fixes the tuple case.
#[derive(Default)]
struct MixHasher(u64);

impl Hasher for MixHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        self.0 = crate::common::hash_bytes(bytes, self.0 | 1);
    }
    #[inline(always)]
    fn write_u64(&mut self, x: u64) {
        self.0 = mum(self.0 ^ x, 0x9E37_79B9_7F4A_7C15);
    }
    #[inline(always)]
    fn write_u32(&mut self, x: u32) {
        self.write_u64(x as u64);
    }
    #[inline(always)]
    fn write_u8(&mut self, x: u8) {
        self.write_u64(x as u64 | 0x100);
    }
    #[inline(always)]
    fn write_usize(&mut self, x: usize) {
        self.write_u64(x as u64);
    }
}

pub fn hash_values(vs: &[Value]) -> u64 {
    let mut h = MixHasher(0x243F_6A88_85A3_08D3);
    for v in vs {
        v.hash(&mut h);
    }
    h.finish()
}

/// `hash_values(&[Value::Str(s)])` computed from a borrowed `&str`.
///
/// Exists so single-column string grouping can probe the group table without
/// materializing a `Value` per row — cloning an `Arc<str>` is an atomic
/// increment, and paying one per input row is most of the cost of a
/// `GROUP BY` over a low-cardinality string.
///
/// This must stay bit-identical to the owned path or a probe would miss its
/// own group; `hash_of_a_borrowed_str_matches_the_owned_value` pins that.
/// It holds because `Arc<str>`'s `Hash` delegates to `str`'s.
#[inline]
pub fn hash_str_key(s: &str) -> u64 {
    let mut h = MixHasher(0x243F_6A88_85A3_08D3);
    1u8.hash(&mut h);
    s.hash(&mut h);
    h.finish()
}

/// `hash_values(&[Value::Null])`, hoisted out of the row loop.
#[inline]
pub fn hash_null_key() -> u64 {
    let mut h = MixHasher(0x243F_6A88_85A3_08D3);
    Value::Null.hash(&mut h);
    h.finish()
}

/// The key tuple for one row of a block, over the given columns.
#[inline]
pub fn row_key(cols: &[crate::types::Column], row: usize) -> GroupKey {
    GroupKey(cols.iter().map(|c| c.value(row)).collect())
}

// ------------------------------------------------------------------ helpers

/// Drain an operator into a single block. Used by the blocking operators,
/// which have to see everything before they can answer.
///
/// This is where a blocking operator spends its time on a large input, so it
/// is also where the cancel/deadline checkpoint has to live: `guard` is
/// consulted once per block and charged the running size of the accumulator,
/// which is exactly the thing that grows without bound here.
///
/// The running size is **summed as blocks arrive** rather than re-measured off
/// the accumulator, and that is not a micro-optimization: `Block::bytes` is
/// O(1) for a fixed-width column but O(rows) for a string one, because it walks
/// the `Vec<Arc<str>>` adding lengths. Asking a *growing* accumulator its size
/// once per block therefore made this loop quadratic in the input -- 244 blocks
/// over 2M rows re-walk 245M entries -- on any plan carrying a string.
///
/// Measured with a temporary switch alternating old and new in one loop,
/// best-of-15 per side, over 2M rows through the one shape that reaches
/// `drain` (a `Sort` with no keys, i.e. `LIMIT k` in input order):
///
/// ```text
///   String column   104.2 -> 40.6 ms   2.57x   (2.57-2.95x over 3 runs)
///   Int64 column      5.63 ->  5.59 ms  1.01x   <- control, bytes() was already O(1)
/// ```
///
/// The two sums are not bit-identical: `a.extend(&b)` may widen the
/// accumulator's null bitmap past the sum of the blocks' own, so this can
/// under-count by a bit per row (1/64 of a fixed-width column, 1/512 of a
/// 64-byte string). `grow_to` feeds a *budget estimate* that already ignores
/// `Vec` slack in the other direction, and no answer depends on it -- only how
/// early a starved query is told no.
pub(crate) fn drain(
    op: &mut Box<dyn Operator + '_>,
    ctx: &QueryContext,
    guard: &mut MemGuard,
) -> Result<Block> {
    let mut acc: Option<Block> = None;
    let mut bytes = 0usize;
    loop {
        ctx.check()?;
        let Some(b) = op.next()? else { break };
        if b.rows() == 0 {
            continue;
        }
        bytes += b.bytes();
        match &mut acc {
            None => acc = Some(b),
            Some(a) => a.extend(&b)?,
        }
        guard.grow_to(bytes)?;
    }
    Ok(match acc {
        Some(a) => a,
        None => Block::empty(op.schema()),
    })
}

/// Split one big block into pipeline-sized batches. Free when it already fits.
pub(crate) fn chunk(block: Block) -> Vec<Block> {
    let n = block.rows();
    if n <= BLOCK_SIZE {
        return if n == 0 { Vec::new() } else { vec![block] };
    }
    let mut out = Vec::with_capacity(n.div_ceil(BLOCK_SIZE));
    let mut s = 0;
    while s < n {
        let e = (s + BLOCK_SIZE).min(n);
        out.push(block.slice(s, e));
        s = e;
    }
    out
}

/// Gather `perm` out of `block` directly into pipeline-sized batches.
///
/// The obvious `chunk(block.take(&perm))` materializes the whole permuted
/// result and then copies it again a block at a time -- two full copies of a
/// sorted output where one will do. Gathering per window costs the same gather
/// and half the memory, and each window's scatter target stays in L2.
pub(crate) fn chunk_take(block: &Block, perm: &[u32]) -> Vec<Block> {
    let n = perm.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n.div_ceil(BLOCK_SIZE));
    let mut s = 0;
    while s < n {
        let e = (s + BLOCK_SIZE).min(n);
        out.push(block.take(&perm[s..e]));
        s = e;
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn hash_of_a_borrowed_str_matches_the_owned_value() {
        // The single-column string grouping fast path probes with a &str while
        // the table stores owned Values. If these ever diverge, a probe misses
        // the group it just inserted and every row becomes its own group.
        for s in ["", "a", "US", "a longer string value", "日本語", "with\0nul"] {
            assert_eq!(
                super::hash_str_key(s),
                super::hash_values(&[Value::str(s)]),
                "mismatch for {s:?}"
            );
        }
        assert_eq!(super::hash_null_key(), super::hash_values(&[Value::Null]));
    }

    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn h(k: &GroupKey) -> u64 {
        let mut s = DefaultHasher::new();
        k.hash(&mut s);
        s.finish()
    }

    #[test]
    fn group_key_discriminates_every_component() {
        // The bug this guards against: a hasher that overwrites its state
        // would make (1, x) and (2, x) collide for every x.
        let a = GroupKey(vec![Value::Int(1), Value::Int(9)]);
        let b = GroupKey(vec![Value::Int(2), Value::Int(9)]);
        let c = GroupKey(vec![Value::Int(1), Value::Int(8)]);
        assert_ne!(h(&a), h(&b));
        assert_ne!(h(&a), h(&c));
        assert_eq!(h(&a), h(&GroupKey(vec![Value::Int(1), Value::Int(9)])));
    }

    #[test]
    fn group_key_hash_agrees_with_eq_across_representations() {
        let a = GroupKey(vec![Value::UInt(7)]);
        let b = GroupKey(vec![Value::Int(7)]);
        assert_eq!(a, b, "Value equates numerics across representations");
        assert_eq!(h(&a), h(&b), "so their hashes must match too");
    }

    #[test]
    fn probe_group_key_hash_agrees_with_eq_for_date_and_datetime() {
        // `Value::cmp` compares Date/DateTime against the plain numeric family
        // by value (rank(Date)=2, rank(Int)=1, both <= 3 -> numeric compare),
        // so these keys are Eq. The Hash/Eq contract therefore demands equal
        // hashes.
        let d = GroupKey(vec![Value::Date(5)]);
        let u = GroupKey(vec![Value::UInt(5)]);
        assert_eq!(d, u);
        assert_eq!(h(&d), h(&u), "Date(5) == UInt(5) but hashes differ");

        let t = GroupKey(vec![Value::DateTime(5)]);
        assert_eq!(t, GroupKey(vec![Value::Int(5)]));
        assert_eq!(h(&t), h(&GroupKey(vec![Value::Int(5)])));
    }

    #[test]
    fn group_key_spreads_over_many_tuples() {
        let mut seen = std::collections::HashSet::new();
        for i in 0..200i64 {
            for j in 0..200i64 {
                seen.insert(hash_values(&[Value::Int(i), Value::Int(j)]));
            }
        }
        assert!(seen.len() > 39_000, "only {} distinct hashes of 40000", seen.len());
    }

    #[test]
    fn scan_stats_merge_and_ratio() {
        let mut a = ScanStats { granules_pruned: 3, granules_read: 1, rows_read: 10 };
        a.merge(&ScanStats { granules_pruned: 1, granules_read: 0, rows_read: 5 });
        assert_eq!(a, ScanStats { granules_pruned: 4, granules_read: 1, rows_read: 15 });
        assert!((a.prune_ratio() - 0.8).abs() < 1e-9);
        assert_eq!(ScanStats::default().prune_ratio(), 0.0);
    }

    #[test]
    fn chunk_take_splits_on_block_size_and_applies_the_permutation() {
        let n = BLOCK_SIZE * 2 + 5;
        let c = crate::types::Column::i64s(
            crate::types::DataType::Int64,
            (0..n as i64).collect(),
        );
        let b = Block::new(vec![c]).unwrap();
        let perm: Vec<u32> = (0..n as u32).rev().collect();
        let blocks = chunk_take(&b, &perm);
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].rows(), BLOCK_SIZE);
        assert_eq!(blocks[2].rows(), 5);
        // The gather has to be applied per window, not per block boundary.
        assert_eq!(blocks[0].column(0).value(0), Value::Int(n as i64 - 1));
        assert_eq!(blocks[2].column(0).value(4), Value::Int(0));
        assert!(chunk_take(&b, &[]).is_empty());
    }

    // ------------------------------------------------ cancellation & budgets

    #[test]
    fn a_cancelled_context_stops_the_query() {
        let ctx = QueryContext::new();
        assert!(ctx.check().is_ok());
        ctx.stop();
        let e = ctx.check().unwrap_err();
        assert!(e.to_string().contains("cancelled"), "{e}");
    }

    #[test]
    fn a_handle_cancels_from_another_thread() {
        let ctx = QueryContext::new();
        let h = ctx.cancel_handle();
        let t = std::thread::spawn(move || h.store(true, Ordering::Relaxed));
        t.join().unwrap();
        assert!(ctx.check().is_err(), "the flag is shared, not copied");
    }

    #[test]
    fn an_elapsed_deadline_fires() {
        let ctx = QueryContext::new().deadline_in(Duration::from_nanos(1));
        std::thread::sleep(Duration::from_millis(2));
        let e = ctx.check().unwrap_err();
        assert!(e.to_string().contains("deadline"), "{e}");
        // and a live deadline does not
        assert!(QueryContext::new()
            .deadline_in(Duration::from_secs(60))
            .check()
            .is_ok());
    }

    #[test]
    fn the_tracker_refuses_what_it_cannot_hold_and_gives_it_back() {
        let m = MemTracker::with_limit(1_000);
        m.reserve(600, "test").unwrap();
        let e = m.reserve(600, "the group table").unwrap_err();
        assert!(e.to_string().contains("group table"), "{e}");
        assert_eq!(m.used(), 600, "a refused reservation must not stay charged");
        m.release(600);
        assert_eq!(m.used(), 0);
        // The message has to name the ceiling and the ask, or it is unactionable.
        assert!(e.to_string().contains("1000 B") || e.to_string().contains("KiB"), "{e}");
    }

    #[test]
    fn a_guard_releases_on_drop_and_only_charges_growth() {
        let ctx = QueryContext::with_budget(10_000);
        {
            let mut g = MemGuard::new(&ctx, "sort buffer");
            g.grow_to(4_000).unwrap();
            g.grow_to(1_000).unwrap();
            assert_eq!(ctx.mem.used(), 4_000, "shrinking must not refund mid-flight");
            g.grow_to(9_000).unwrap();
            assert_eq!(ctx.mem.used(), 9_000);
            assert!(g.grow_to(20_000).is_err());
            assert_eq!(ctx.mem.used(), 9_000, "the failed growth is not held");
        }
        assert_eq!(ctx.mem.used(), 0, "the guard did not release on drop");
    }

    // ------------------------------------------------- whole-pipeline tests
    //
    // `build` / `execute` / `execute_with_stats` are what `session.rs` calls,
    // so they get exercised against a real catalog rather than only through
    // the individual operators.

    mod pipeline {
        use super::super::*;
        use crate::exec::functions;
        use crate::planner::logical::{
            BoundAgg, BoundExpr, CmpOp, ScanNode, SortKey, ZoneFilter,
        };
        use crate::sql::ast::{BinaryOp, JoinOp};
        use crate::types::{Column, DataType, Engine, Field, TableDef, Value};

        /// `orders(id UInt64, cust UInt64, amount Int64)`, `id` sorted 0..n.
        fn catalog(n: u64) -> Catalog {
            let mut c = Catalog::in_memory();
            c.create_table(
                TableDef {
                    name: "orders".into(),
                    schema: Schema::new(vec![
                        Field::new("id", DataType::UInt64),
                        Field::new("cust", DataType::UInt64),
                        Field::new("amount", DataType::Int64),
                    ])
                    .unwrap(),
                    order_by: vec![0],
                    primary_key: vec![0],
                    partition_by: None,
                    engine: Engine::MergeTree,
                },
                false,
            )
            .unwrap();
            let t = c.table_by_path_mut("default.orders").unwrap();
            t.insert(
                Block::new(vec![
                    Column::u64s(DataType::UInt64, (0..n).collect()),
                    Column::u64s(DataType::UInt64, (0..n).map(|i| i % 4).collect()),
                    Column::i64s(DataType::Int64, (0..n).map(|i| (i % 10) as i64).collect()),
                ])
                .unwrap(),
            )
            .unwrap();
            t.flush().unwrap();
            c
        }

        fn scan(cat: &Catalog, projection: Vec<usize>) -> ScanNode {
            let full = cat.table_by_path("default.orders").unwrap().schema().clone();
            ScanNode {
                table: "default.orders".into(),
                schema: full.project(&projection),
                projection,
                filters: vec![],
                zone_filters: vec![],
            }
        }

        fn col(i: usize, ty: DataType) -> BoundExpr {
            BoundExpr::Column { index: i, ty, name: format!("c{i}") }
        }

        fn rows_of(blocks: &[Block]) -> usize {
            blocks.iter().map(|b| b.rows()).sum()
        }

        fn values(blocks: &[Block], c: usize) -> Vec<Value> {
            blocks
                .iter()
                .flat_map(|b| (0..b.rows()).map(move |i| b.column(c).value(i)))
                .collect()
        }

        #[test]
        fn scan_project_limit_pipeline() {
            let cat = catalog(1_000);
            let out_schema = Schema::new(vec![Field::new("double", DataType::Int64)]).unwrap();
            let plan = LogicalPlan::Limit {
                input: Box::new(LogicalPlan::Project {
                    input: Box::new(LogicalPlan::Scan(Box::new(scan(&cat, vec![0])))),
                    exprs: vec![BoundExpr::Binary {
                        left: Box::new(col(0, DataType::UInt64)),
                        op: BinaryOp::Multiply,
                        right: Box::new(BoundExpr::lit(Value::UInt(2))),
                        ty: DataType::UInt64,
                    }],
                    schema: out_schema,
                }),
                limit: Some(3),
                offset: 1,
            };
            let blocks = execute(&plan, &cat).unwrap();
            assert_eq!(
                values(&blocks, 0),
                vec![Value::UInt(2), Value::UInt(4), Value::UInt(6)]
            );
        }

        #[test]
        fn group_by_over_a_scan() {
            let cat = catalog(1_000);
            let count = functions::aggregate("count").unwrap();
            let sum = functions::aggregate("sum").unwrap();
            let aggs = vec![
                BoundAgg {
                    func: count,
                    args: vec![],
                    params: vec![],
                    distinct: false,
                    ty: (count.ret)(&[], &[]).unwrap(),
                    name: "n".into(),
                },
                BoundAgg {
                    func: sum,
                    args: vec![col(1, DataType::Int64)],
                    params: vec![],
                    distinct: false,
                    ty: (sum.ret)(&[DataType::Int64], &[]).unwrap(),
                    name: "total".into(),
                },
            ];
            let schema = Schema::new_unchecked(vec![
                Field::new("cust", DataType::UInt64),
                Field::new("n", aggs[0].ty.clone()),
                Field::new("total", aggs[1].ty.clone()),
            ]);
            let plan = LogicalPlan::Sort {
                input: Box::new(LogicalPlan::Aggregate {
                    input: Box::new(LogicalPlan::Scan(Box::new(scan(&cat, vec![1, 2])))),
                    group: vec![col(0, DataType::UInt64)],
                    aggs,
                    schema,
                }),
                keys: vec![SortKey {
                    expr: col(0, DataType::UInt64),
                    asc: true,
                    nulls_first: true,
                }],
            };
            let blocks = execute(&plan, &cat).unwrap();
            assert_eq!(rows_of(&blocks), 4, "cust cycles through 4 values");
            assert_eq!(values(&blocks, 0), (0..4).map(Value::UInt).collect::<Vec<_>>());
            assert_eq!(values(&blocks, 1), vec![Value::UInt(250); 4]);
            // Every row lands in exactly one group, so the per-group totals
            // have to add back up to the whole table's sum.
            let total: i64 = values(&blocks, 2).iter().map(|v| v.as_i64().unwrap()).sum();
            assert_eq!(total, (0..1_000i64).map(|i| i % 10).sum::<i64>());
        }

        #[test]
        fn stats_ride_all_the_way_up_a_deep_plan() {
            let cat = catalog(20_000);
            let mut s = scan(&cat, vec![0]);
            s.zone_filters =
                vec![ZoneFilter { col: 0, op: CmpOp::GtEq, value: Value::UInt(19_000) }];
            s.filters = vec![BoundExpr::Binary {
                left: Box::new(col(0, DataType::UInt64)),
                op: BinaryOp::GtEq,
                right: Box::new(BoundExpr::lit(Value::UInt(19_000))),
                ty: DataType::Bool,
            }];
            // Scan -> Filter -> Sort -> Limit: every level has to forward stats.
            let plan = LogicalPlan::Limit {
                input: Box::new(LogicalPlan::Sort {
                    input: Box::new(LogicalPlan::Filter {
                        input: Box::new(LogicalPlan::Scan(Box::new(s))),
                        predicate: BoundExpr::Binary {
                            left: Box::new(col(0, DataType::UInt64)),
                            op: BinaryOp::Lt,
                            right: Box::new(BoundExpr::lit(Value::UInt(19_010))),
                            ty: DataType::Bool,
                        },
                    }),
                    keys: vec![SortKey {
                        expr: col(0, DataType::UInt64),
                        asc: false,
                        nulls_first: true,
                    }],
                }),
                limit: Some(2),
                offset: 0,
            };
            let (blocks, st) = execute_with_stats(&plan, &cat).unwrap();
            assert_eq!(values(&blocks, 0), vec![Value::UInt(19_009), Value::UInt(19_008)]);
            assert!(st.granules_pruned >= 17, "pruned only {}", st.granules_pruned);
            assert!(st.rows_read > 0);
        }

        #[test]
        fn distinct_union_and_values_compose() {
            let cat = Catalog::in_memory();
            let s = Schema::new(vec![Field::new("a", DataType::Int64)]).unwrap();
            let plan = LogicalPlan::Union {
                inputs: vec![
                    LogicalPlan::Values {
                        rows: vec![vec![Value::Int(1)], vec![Value::Int(2)]],
                        schema: s.clone(),
                    },
                    LogicalPlan::Values {
                        rows: vec![vec![Value::Int(2)], vec![Value::Int(3)]],
                        schema: s.clone(),
                    },
                    LogicalPlan::Empty { schema: s.clone() },
                ],
                op: crate::sql::ast::SetOp::Union,
                all: false,
                schema: s.clone(),
            };
            let blocks = execute(&plan, &cat).unwrap();
            assert_eq!(
                values(&blocks, 0),
                vec![Value::Int(1), Value::Int(2), Value::Int(3)]
            );
        }

        #[test]
        fn join_of_two_scans() {
            let mut cat = catalog(8);
            cat.create_table(
                TableDef {
                    name: "custs".into(),
                    schema: Schema::new(vec![
                        Field::new("cid", DataType::UInt64),
                        Field::new("name", DataType::String),
                    ])
                    .unwrap(),
                    order_by: vec![0],
                    primary_key: vec![0],
                    partition_by: None,
                    engine: Engine::MergeTree,
                },
                false,
            )
            .unwrap();
            {
                let t = cat.table_by_path_mut("default.custs").unwrap();
                t.insert(
                    Block::new(vec![
                        Column::u64s(DataType::UInt64, vec![0, 1]),
                        Column::strs(DataType::String, vec!["ann".into(), "bob".into()]),
                    ])
                    .unwrap(),
                )
                .unwrap();
                t.flush().unwrap();
            }
            let left = scan(&cat, vec![1]); // orders.cust
            let cust_schema = cat.table_by_path("default.custs").unwrap().schema().clone();
            let right = ScanNode {
                table: "default.custs".into(),
                schema: cust_schema.clone(),
                projection: vec![0, 1],
                filters: vec![],
                zone_filters: vec![],
            };
            let out = left.schema.concat(&cust_schema);
            let plan = LogicalPlan::Join {
                left: Box::new(LogicalPlan::Scan(Box::new(left))),
                right: Box::new(LogicalPlan::Scan(Box::new(right))),
                op: JoinOp::Inner,
                on: vec![(0, 0)],
                residual: None,
                schema: out,
            };
            let blocks = execute(&plan, &cat).unwrap();
            // 8 orders, cust cycles 0..4, only custs 0 and 1 exist -> 4 rows.
            assert_eq!(rows_of(&blocks), 4);
            let names: Vec<String> = values(&blocks, 2)
                .iter()
                .map(|v| v.render_plain())
                .collect();
            assert_eq!(names.iter().filter(|n| *n == "ann").count(), 2);
            assert_eq!(names.iter().filter(|n| *n == "bob").count(), 2);
        }

        #[test]
        fn count_star_needs_no_column_data_at_all() {
            let cat = catalog(5_000);
            let count = functions::aggregate("count").unwrap();
            let ty = (count.ret)(&[], &[]).unwrap();
            let plan = LogicalPlan::Aggregate {
                // An empty projection: the scan reads row counts, nothing else.
                input: Box::new(LogicalPlan::Scan(Box::new(scan(&cat, vec![])))),
                group: vec![],
                aggs: vec![BoundAgg {
                    func: count,
                    args: vec![],
                    params: vec![],
                    distinct: false,
                    ty: ty.clone(),
                    name: "n".into(),
                }],
                schema: Schema::new(vec![Field::new("n", ty)]).unwrap(),
            };
            let blocks = execute(&plan, &cat).unwrap();
            assert_eq!(values(&blocks, 0), vec![Value::UInt(5_000)]);
        }

        #[test]
        fn an_empty_plan_executes_to_nothing() {
            let cat = Catalog::in_memory();
            let s = Schema::new(vec![Field::new("a", DataType::Int64)]).unwrap();
            let plan = LogicalPlan::Empty { schema: s };
            let (blocks, st) = execute_with_stats(&plan, &cat).unwrap();
            assert!(blocks.is_empty());
            assert_eq!(st, ScanStats::default());
        }

        #[test]
        fn build_reports_a_missing_table_rather_than_panicking() {
            let cat = Catalog::in_memory();
            let plan = LogicalPlan::Scan(Box::new(ScanNode {
                table: "default.absent".into(),
                projection: vec![0],
                schema: Schema::empty(),
                filters: vec![],
                zone_filters: vec![],
            }));
            assert!(build(&plan, &cat, &QueryContext::new()).is_err());
        }

        /// A `GROUP BY id` / bare aggregate pair over the same scan, so the two
        /// halves of the budget story below differ in exactly one thing:
        /// whether there is a key to partition the overflow by.
        ///
        /// `uniqExact(id)` rather than `count()`, and the difference is the
        /// whole reason the argument is spelled out: a bare unfiltered
        /// `count()` is answered from part metadata (`physical::meta_path`),
        /// so it never builds an accumulator, never charges the tracker and
        /// can no longer be starved -- it would silently stop testing the half
        /// it is here for. `uniqExact` over 20k distinct ids is the same
        /// one-group shape and answers 20 000 all the same; plain `uniq` is
        /// an HLL sketch and answers 19 881.
        fn agg_plan(cat: &Catalog, group: Vec<BoundExpr>) -> LogicalPlan {
            let uniq = functions::aggregate("uniqExact").unwrap();
            let arg = col(0, DataType::UInt64);
            let ty = (uniq.ret)(&[DataType::UInt64], &[]).unwrap();
            // `[id, n]` with a grouping key, `[n]` without.
            let mut fields: Vec<Field> =
                group.iter().map(|_| Field::new("id", DataType::UInt64)).collect();
            fields.push(Field::new("n", ty.clone()));
            LogicalPlan::Aggregate {
                input: Box::new(LogicalPlan::Scan(Box::new(scan(cat, vec![0])))),
                group,
                aggs: vec![BoundAgg {
                    func: uniq,
                    args: vec![arg],
                    params: vec![],
                    distinct: false,
                    ty,
                    name: "n".into(),
                }],
                schema: Schema::new_unchecked(fields),
            }
        }

        #[test]
        fn a_tight_budget_spills_a_group_by_and_errors_only_where_it_cannot() {
            // Inverted from `..._errors_instead_of_dying`, which pinned the step
            // before this one: 20k groups against a 64 KiB budget used to be an
            // error, and the whole point of that error was that it was not a
            // process the OS killed. `GROUP BY` now spills instead, so the
            // error moved rather than disappearing -- and both halves matter,
            // because "bounded" and "answerable" are different claims.
            let cat = catalog(20_000);
            let plan = agg_plan(&cat, vec![col(0, DataType::UInt64)]);

            // Half one: the budget bounds *memory*, not what is answerable.
            sort::spill::SPILLED.with(|s| s.borrow_mut().clear());
            let tight = QueryContext::with_budget(64 << 10);
            let (blocks, _) = execute_ctx(&plan, &cat, &tight).unwrap();
            assert_eq!(rows_of(&blocks), 20_000);
            let mut ids = values(&blocks, 0);
            ids.sort();
            assert_eq!(ids, (0..20_000u64).map(Value::UInt).collect::<Vec<_>>());
            assert!(
                values(&blocks, 1).iter().all(|v| *v == Value::UInt(1)),
                "`id` is unique, so every group holds exactly one row"
            );
            assert_eq!(tight.mem.used(), 0, "the query kept its reservation");
            let dirs = sort::spill::SPILLED.with(|s| s.borrow().clone());
            assert!(!dirs.is_empty(), "nothing spilled, so nothing was tested");
            for d in &dirs {
                assert!(!d.exists(), "spill directory {} outlived its query", d.display());
            }

            // Half two: the shape that *cannot* spill. A bare aggregate's one
            // group exists before the first row and has no key to partition by,
            // so a budget too small to hold it is still an error the caller
            // sees -- which is the property the old test was really guarding.
            // 64 B does not fit one boxed accumulator, let alone the table it
            // hangs off, so this cannot become answerable by spilling harder.
            let bare = agg_plan(&cat, vec![]);
            let starved = QueryContext::with_budget(64);
            let msg = execute_ctx(&bare, &cat, &starved).unwrap_err().to_string();
            assert!(msg.contains("memory budget"), "{msg}");
            assert!(msg.contains("aggregate state"), "the error must name what grew: {msg}");
            assert_eq!(starved.mem.used(), 0, "the failed query kept its reservation");

            // The negative for both: under the default budget nothing is
            // touched, and the answers match an ungoverned run.
            let ok = execute_ctx(&plan, &cat, &QueryContext::new()).unwrap();
            assert_eq!(rows_of(&ok.0), 20_000);
            assert_eq!(execute(&plan, &cat).unwrap().len(), ok.0.len());
            assert_eq!(values(&execute(&bare, &cat).unwrap(), 0), vec![Value::UInt(20_000)]);
        }

        #[test]
        fn cancelling_stops_a_running_query_promptly() {
            let cat = catalog(200_000);
            let plan = LogicalPlan::Sort {
                input: Box::new(LogicalPlan::Scan(Box::new(scan(&cat, vec![0])))),
                keys: vec![SortKey {
                    expr: col(0, DataType::UInt64),
                    asc: true,
                    nulls_first: true,
                }],
            };
            let ctx = QueryContext::new();
            ctx.stop();
            let e = execute_ctx(&plan, &cat, &ctx).unwrap_err();
            assert!(e.to_string().contains("cancelled"), "{e}");
        }

        #[test]
        fn an_expired_deadline_stops_a_running_query() {
            let cat = catalog(200_000);
            let plan = LogicalPlan::Sort {
                input: Box::new(LogicalPlan::Scan(Box::new(scan(&cat, vec![0])))),
                keys: vec![SortKey {
                    expr: col(0, DataType::UInt64),
                    asc: true,
                    nulls_first: true,
                }],
            };
            let ctx = QueryContext::new().deadline_in(Duration::from_nanos(1));
            std::thread::sleep(Duration::from_millis(2));
            let e = execute_ctx(&plan, &cat, &ctx).unwrap_err();
            assert!(e.to_string().contains("deadline"), "{e}");
        }

        #[test]
        fn top_k_fusion_survives_a_project_and_an_offset() {
            // `Limit -> Project -> Sort` is the shape the binder emits for
            // `SELECT a, b ... ORDER BY c LIMIT n OFFSET m`, and the fused
            // pipeline has to give byte-identical answers to the unfused one.
            let cat = catalog(30_000);
            let out_schema = Schema::new(vec![Field::new("neg", DataType::Int64)]).unwrap();
            let sort = LogicalPlan::Sort {
                input: Box::new(LogicalPlan::Scan(Box::new(scan(&cat, vec![0])))),
                keys: vec![SortKey {
                    expr: col(0, DataType::UInt64),
                    asc: false,
                    nulls_first: true,
                }],
            };
            let project = LogicalPlan::Project {
                input: Box::new(sort),
                exprs: vec![BoundExpr::Binary {
                    left: Box::new(col(0, DataType::UInt64)),
                    op: BinaryOp::Multiply,
                    right: Box::new(BoundExpr::lit(Value::UInt(2))),
                    ty: DataType::UInt64,
                }],
                schema: out_schema,
            };
            let want: Vec<Value> = values(&execute(&project, &cat).unwrap(), 0)[7..17].to_vec();

            let plan = LogicalPlan::Limit {
                input: Box::new(project),
                limit: Some(10),
                offset: 7,
            };
            assert_eq!(values(&execute(&plan, &cat).unwrap(), 0), want);

            // ... and it must actually be bounded, not merely correct: 30k
            // rows through a budget that only fits a few blocks.
            let ctx = QueryContext::with_budget(2 << 20);
            assert!(execute_ctx(&plan, &cat, &ctx).is_ok());
            assert_eq!(ctx.mem.used(), 0);
        }

        #[test]
        fn a_normal_query_leaves_the_budget_where_it_found_it() {
            // Every reservation is guarded, so a completed pipeline must have
            // handed all of it back once the operators are dropped.
            let cat = catalog(50_000);
            let ctx = QueryContext::new();
            let plan = LogicalPlan::Sort {
                input: Box::new(LogicalPlan::Scan(Box::new(scan(&cat, vec![0])))),
                keys: vec![SortKey {
                    expr: col(0, DataType::UInt64),
                    asc: false,
                    nulls_first: true,
                }],
            };
            let (blocks, _) = execute_ctx(&plan, &cat, &ctx).unwrap();
            assert_eq!(rows_of(&blocks), 50_000);
            assert_eq!(ctx.mem.used(), 0);
        }
    }
}
