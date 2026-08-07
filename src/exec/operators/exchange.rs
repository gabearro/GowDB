//! The exchange: N copies of a pipeline over disjoint slices of one scan.
//!
//! `Table::scan_fold` folds a table at 10.58 G rows/s across 14 cores. SQL ran
//! the same work at ~200 M rows/s, because [`Scan`](operators::scan::Scan) is a
//! serial pull iterator and every operator above it is single-threaded. The
//! thread pool and the parallel scan both already existed; nothing but the
//! operator API stood between them and a `SELECT`.
//!
//! ```text
//!            Exchange                merge the partials, emit blocks
//!          /    |    \
//!    Aggregate Aggregate ...         one per worker, own hash table
//!        |      |
//!     Filter  Filter                 replicated, no shared state
//!        |      |
//!   ShardScan ShardScan              disjoint granule ranges, one Snapshot
//! ```
//!
//! ## Why the split is static and contiguous
//!
//! [`pool`] hands work out one index at a time precisely because granule costs
//! are uneven, and the storage-layer scan claims granules from a shared cursor
//! for that reason. This does **not**, and the reason is output, not speed.
//!
//! Give worker `k` a contiguous ascending granule range and fold the partials
//! in worker order, and the merged result is *bit-identical* to the serial one:
//! `GROUP BY` still returns groups in first-seen order, `any(x)` still returns
//! the first `x`, `argMin` still breaks ties toward the earlier row, and a
//! stable `ORDER BY` still breaks ties toward the earlier input row. A shared
//! cursor gives up all four -- and gives them up *nondeterministically*, so the
//! same query answers differently on consecutive runs. The roadmap allows the
//! `GROUP BY` order to become unspecified; it does not follow that it should
//! be, when contiguity buys it back for nothing.
//!
//! What contiguity costs is load balance when granule costs really are uneven,
//! and there is exactly one thing in this engine that makes them uneven by
//! orders of magnitude: a zone map that rejects a granule whole. So the zone
//! maps are evaluated **once, up front and in parallel**, and the workers are
//! given equal shares of the granules that *survive* (see [`Exchange::prune`]).
//! That is the same total work the serial scan does -- it is moved, not added
//! -- and it removes the one gross imbalance a static split would suffer.
//! Residual variance (dictionary vs. bitpacked columns, delete masks) is
//! within a small factor and is the price of a deterministic answer.
//!
//! ## What runs in parallel, and what deliberately does not
//!
//! [`analyze`] admits `Scan -> (Filter | Project)* -> (Aggregate | Sort)` and
//! nothing else. `Filter` and `Project` are pure per-block maps, so replicating
//! them needs no merge at all. The two tops have real merges and they are the
//! two that matter: an aggregate is the shape a 10 G rows/s scan is *for*.
//!
//! **Joins stay serial**, and this operator declines any plan containing one.
//! A correct parallel hash join needs either a replicated build side (memory
//! proportional to workers) or a partitioned one (a shuffle this executor has
//! no operator for), and a subtly wrong join is worth less than no parallel
//! join at all. A join *above* an aggregate still gets a parallel aggregate,
//! because the decision is made per plan node.
//!
//! `Distinct` and `LimitBy` are also declined: both are streaming operators
//! whose output order is first-seen, and neither has a merge that is cheaper
//! than the operator itself.
//!
//! ## Governance
//!
//! One [`QueryContext`] serves every worker, which is what makes the budget a
//! *query* budget: the [`MemTracker`](operators::MemTracker) is one atomic, so N
//! hash tables charge one ceiling. The corollary is that an N-way `GROUP BY`
//! genuinely holds N partial tables at once and can be refused where the
//! serial plan fit -- that is the accounting being right, not a regression, and
//! it is why [`degree`] leaves small queries alone. Cancellation and the
//! deadline reach every worker through the same context, once per block.
//!
//! ## Nesting
//!
//! `Pool::for_each`/`map` run a job submitted from inside a pool worker inline
//! on the submitting thread, so an exchange reached from within a pool job is
//! correct without a guard here. It would still build N pipelines and run them
//! one after another, which is why it is worth saying that nothing in this
//! engine currently executes SQL from inside a pool worker.
//!
//! ## What it is worth
//!
//! 10M rows, six columns, 14 cores. Serial and parallel interleaved in one
//! loop with the sides alternating which runs first, best-of-9 each. This
//! machine swings ±25% run to run, so the ranges in the last column are what
//! five separate runs produced -- the single-run column is the shipped code
//! measured end to end through `Session`'s own planner.
//!
//! ```text
//!   SELECT count(*)                                36.4 ->   6.7 ms  5.5x  (5.5-7.2x)
//!   SELECT sum(bytes)                              47.9 ->   8.5 ms  5.6x  (5.6-7.2x)
//!   SELECT sum(bytes) WHERE latency > 500          98.7 ->  15.2 ms  6.5x  (6.5-8.5x)
//!   SELECT country, count(*), sum(bytes) GROUP BY 164.0 ->  28.2 ms  5.8x  (5.8-9.9x)
//!   SELECT sum(bytes) WHERE id > 9000000 (pruned)  11.5 ->   2.0 ms  5.7x  (5.6-6.6x)
//!   SELECT uniq(big)                               71.1 ->  17.3 ms  4.1x  (4.1-7.2x)
//!   ORDER BY latency DESC LIMIT 5                 133.5 ->  24.5 ms  5.5x  (5.5-8.8x)
//!   ORDER BY country, latency LIMIT 100           282.5 ->  77.4 ms  3.7x  (2.9-3.7x)
//!   ORDER BY country, latency  (full, comparison) 7079  -> 1685  ms  4.2x  (4.2-5.8x)
//!   SELECT count(DISTINCT big)  (100k values)     859.3 -> 400.6 ms  2.1x  (2.1-3.5x)
//!   GROUP BY big               (100k groups)      521.7 -> 321.3 ms  1.6x  (1.6-2.0x)
//!   ORDER BY latency           (full, radix)      156.2 -> 139.0 ms  1.1x  (1.1-1.4x)
//!   GROUP BY id                (10M groups)      1359   -> 1628  ms  0.8x  (0.8-1.4x)
//! ```
//!
//! The last three are the honest part of the table.
//!
//! *A full sort on the radix path* barely moves, and cannot: a serial radix
//! sort of 10M rows is four linear passes over `(lane, row)` pairs, while
//! merging fourteen sorted runs is a heap sift per output row. The merge costs
//! roughly what the sort it replaced did, so all that is left is the scan
//! speedup. The comparison path is 4-6x for the same reason in reverse -- its
//! sort is `n log n` `Value` comparisons and the merge is `n log 14`. Top-K is
//! the shape that wins outright, because each worker keeps `k` rows and the
//! merge is `14k` long regardless of the table.
//!
//! *A high-cardinality `GROUP BY` runs into the merge*, which is serial and
//! proportional to the number of groups, not to the number of rows. At one row
//! per group there is nothing to aggregate and the exchange does the whole
//! grouping twice -- once in parallel, once again while folding partials. The
//! fix is partitioned aggregation: shuffle rows by `hash(key)` so each worker
//! owns a disjoint key range and there is no merge at all. That needs an
//! exchange that repartitions rather than replicates, which is a second
//! operator and a second task. Until then the merge is as cheap as it can be
//! made (see `Groups::absorb`), and 10M distinct groups is a shape to know
//! about rather than one to fear: it is 0.8-1.4x, not 0.1x.

use crate::catalog::Catalog;
use crate::common::{pool, Result, BLOCK_SIZE};
use crate::exec::expr;
use crate::planner::logical::{BoundAgg, BoundExpr, ScanNode, SortKey};
use crate::planner::physical::PhysicalPlan;
use crate::storage::part::Snapshot;
use crate::storage::Part;
use crate::types::{Block, Schema};

// Absolute rather than `super::`, so the module compiles unchanged whether it
// is mounted from `operators/mod.rs` or (as today) from `exec/mod.rs`.
use crate::exec::operators::{
    self, aggregate, distinct, filter, join, limit, project, sort, MemGuard, Operator,
    QueryContext, ScanStats,
};

/// Live rows below which a query stays serial.
///
/// A rendezvous with parked workers costs a few microseconds, and the pipeline
/// setup costs one operator tree per worker on top of that. Below some input
/// size the query is over before the fleet has assembled and going parallel is
/// a pure loss. Measured end to end through the SQL front end on 14 cores,
/// interleaved in one loop with the two sides alternating which goes first
/// (running second is worth a consistent 5-10% here, and a fixed order charges
/// it to the same side every time), best-of-21 per side:
///
/// ```text
///   rows      gran  workers   SELECT sum(bytes)          GROUP BY country (8)
///     4 000      4    --       17.50 us   (declined)     49.67 us  (declined)
///     8 192      8    --       23.00 us   (declined)    108.58 us  (declined)
///    16 384     16     4       44.33 ->  26.38  1.68x   197.12 -> 100.96 1.95x
///    24 000     24     6       63.04 ->  44.29  1.42x   291.08 -> 112.67 2.58x
///    32 768     32     8      131.46 ->  70.08  1.88x   394.88 -> 132.25 2.99x
///    50 000     49    12      141.29 ->  73.17  1.93x   602.21 -> 190.04 3.17x
///   100 000     98    14      266.79 -> 111.33  2.40x     1.23 ms -> 273.92 us
/// 1 000 000    977    14        3.30 ms -> 740.12 us      13.91 ms ->   2.39 ms
/// ```
///
/// The floor sat at 8192 first, and that was measurably wrong: with the two
/// workers eight granules buy, `sum(bytes)` read **0.82x** (55.96 -> 75.08 us)
/// while the `GROUP BY` on the same table read 1.27x. The cheapest aggregate
/// shape is the one that runs out of work to amortize with, so it sets the
/// floor. 16 384 is the next granule-aligned power of two above the last
/// losing measurement, and every shape wins from there up.
///
/// Being wrong toward "stay serial" costs a bounded fraction of a query that
/// was already fast; being wrong toward "go parallel" costs a fixed overhead
/// on every small query, which is the regression an HTAP workload notices.
const MIN_PARALLEL_ROWS: usize = 16 << 10;

/// Granules below which a worker is not worth waking.
///
/// Four granules is 4096 rows, half an execution block: below that a worker
/// spends its share of the query in `Block::extend` and in being woken. It is
/// also what makes [`MIN_PARALLEL_ROWS`] mean four workers rather than one --
/// a floor of 16 would have left a 16k-row table with a single worker, i.e.
/// serial with extra steps.
const MIN_GRANULES_PER_WORKER: usize = 4;

/// How many workers this table's live row count justifies.
///
/// Pure: plan shape and table statistics in, a width out. It is a *planner*
/// decision that happens to be called from the builder -- when
/// [`PhysicalPlan`] grows an `Exchange` node this is the function that
/// belongs behind it, unchanged. 1 means "stay serial", and every caller
/// treats that as a refusal rather than a one-wide fleet.
pub fn degree(rows: usize, granules: usize) -> usize {
    if rows < MIN_PARALLEL_ROWS {
        return 1;
    }
    pool::global()
        .threads()
        .min(granules / MIN_GRANULES_PER_WORKER)
        .max(1)
}

// --------------------------------------------------------------- plan shape

/// A pure per-block stage between the scan and the top: no merge state.
enum Link<'a> {
    Filter(&'a BoundExpr),
    Project { exprs: &'a [BoundExpr], schema: &'a Schema },
}

/// The blocking operator the workers each run a copy of, and whose partials
/// this module knows how to fold.
enum Top<'a> {
    Aggregate { group: &'a [BoundExpr], aggs: &'a [BoundAgg], schema: &'a Schema },
    Sort { keys: &'a [SortKey], fetch: Option<usize> },
}

/// A subtree the exchange can replicate, flattened.
struct Shape<'a> {
    node: &'a ScanNode,
    /// Scan-to-top order, so a worker builds by folding over it.
    links: Vec<Link<'a>>,
    top: Top<'a>,
}

/// Recognize a replicable subtree, or `None` to leave the plan alone.
///
/// Deliberately shallow: it matches a chain, not a tree, so there is no case
/// in which two scans or a join could sneak underneath. Anything unrecognized
/// falls through to the serial builder, which is why adding a plan node
/// elsewhere can never make this wrong -- only less effective.
fn analyze<'a>(plan: &PhysicalPlan<'a>) -> Option<Shape<'a>> {
    let (top, mut p) = match plan {
        PhysicalPlan::Aggregate { input, group, aggs, schema } => {
            (Top::Aggregate { group, aggs, schema }, &**input)
        }
        PhysicalPlan::Sort { input, keys, fetch } => {
            (Top::Sort { keys, fetch: *fetch }, &**input)
        }
        _ => return None,
    };
    let mut links = Vec::new();
    loop {
        match p {
            PhysicalPlan::Filter { input, predicate } => {
                links.push(Link::Filter(predicate));
                p = input;
            }
            PhysicalPlan::Project { input, exprs, schema } => {
                links.push(Link::Project { exprs, schema });
                p = input;
            }
            PhysicalPlan::Scan(node) => {
                links.reverse();
                return Some(Shape { node, links, top });
            }
            // An IndexLookup answers in microseconds and reads one granule per
            // key; there is nothing here to spread. Everything else is either
            // not a leaf or not shardable.
            _ => return None,
        }
    }
}

impl<'a> Shape<'a> {
    /// The schema of the rows reaching the top operator.
    fn input_schema(&self) -> &'a Schema {
        match self.links.iter().rev().find_map(|l| match l {
            Link::Project { schema, .. } => Some(*schema),
            Link::Filter(_) => None,
        }) {
            Some(s) => s,
            None => &self.node.schema,
        }
    }

    /// This subtree's output schema.
    fn schema(&self) -> &'a Schema {
        match &self.top {
            Top::Aggregate { schema, .. } => schema,
            Top::Sort { .. } => self.input_schema(),
        }
    }

    /// One worker's pipeline: a sharded scan with the chain stacked on it.
    fn pipeline<'s>(
        &'s self,
        snap: &'s Snapshot,
        work: &'s [(u32, u32)],
    ) -> Box<dyn Operator + 's>
    where
        'a: 's,
    {
        let mut op: Box<dyn Operator + 's> = Box::new(ShardScan {
            node: self.node,
            snap,
            work,
            at: 0,
            sel: Vec::new(),
            acc: None,
            stats: ScanStats::default(),
        });
        for l in &self.links {
            op = match l {
                Link::Filter(p) => Box::new(filter::Filter::new(op, p)),
                Link::Project { exprs, schema } => {
                    Box::new(project::Project::new(op, exprs, schema))
                }
            };
        }
        op
    }
}

// ------------------------------------------------------------- sharded scan

/// [`Scan`](operators::scan::Scan) restricted to a list of granules.
///
/// The same three stages -- project, prewhere, batch -- minus the pruning
/// stage, which the exchange already ran for the whole table. That absence is
/// the point: a worker's inner loop has no zone-map branch in it at all, and
/// the granules it walks are known to be worth walking.
struct ShardScan<'s> {
    node: &'s ScanNode,
    snap: &'s Snapshot,
    /// This worker's `(part, granule)` pairs, ascending.
    work: &'s [(u32, u32)],
    at: usize,
    /// Live-row selection of the granule in hand, reused across granules.
    sel: Vec<u32>,
    /// Survivors waiting to reach [`BLOCK_SIZE`].
    acc: Option<Block>,
    stats: ScanStats,
}

impl Operator for ShardScan<'_> {
    fn schema(&self) -> &Schema {
        &self.node.schema
    }

    fn stats(&self) -> ScanStats {
        self.stats
    }

    fn next(&mut self) -> Result<Option<Block>> {
        while self.at < self.work.len() {
            let (pi, gi) = self.work[self.at];
            self.at += 1;
            let (pi, gi) = (pi as usize, gi as usize);
            self.stats.granules_read += 1;

            let p = self.snap.part(pi);
            let live = p.live_selection_into(gi, self.snap.deletes(pi), &mut self.sel);
            let mut blk = p.read_columns(gi, &self.node.projection, live)?;
            self.stats.rows_read += blk.rows() as u64;
            if blk.rows() == 0 {
                continue;
            }

            for f in &self.node.filters {
                let sel = expr::eval_predicate(f, &blk)?;
                if sel.len() < blk.rows() {
                    blk = blk.take(&sel);
                }
                if blk.rows() == 0 {
                    break;
                }
            }
            if blk.rows() == 0 {
                continue;
            }

            match &mut self.acc {
                None => self.acc = Some(blk),
                Some(a) => a.extend(&blk)?,
            }
            if self.acc.as_ref().is_some_and(|a| a.rows() >= BLOCK_SIZE) {
                return Ok(self.acc.take());
            }
        }
        Ok(self.acc.take())
    }
}

/// Can every zone filter still be satisfied somewhere in this granule?
///
/// The same test [`Scan`](operators::scan::Scan) makes per granule, hoisted out of
/// the workers so the split can be balanced on what survives it. `zf.col`
/// indexes the *projected* schema and has to be mapped back through
/// `projection` before it can name a packed column -- getting that backwards
/// prunes on the wrong column and silently drops rows.
fn prunes(node: &ScanNode, p: &Part, gi: usize) -> bool {
    let g = &p.granules[gi];
    node.zone_filters.iter().any(|zf| {
        let Some(&table_col) = node.projection.get(zf.col) else { return false };
        let Some(pc) = g.columns.get(table_col) else { return false };
        !zf.may_match(&pc.min_value(), &pc.max_value())
    })
}

// ---------------------------------------------------------------- the merge

/// One worker's contribution.
enum Partial {
    /// A group table covering this worker's granules.
    Groups(aggregate::Groups),
    /// A sorted run. `Block::empty` when the worker's slice held no rows.
    Run(Block),
}

pub struct Exchange<'a> {
    shape: Shape<'a>,
    ctx: &'a QueryContext,
    schema: &'a Schema,
    /// Pinned once, shared by every worker: the answer is a view of the table
    /// at one instant, not a walk over whatever each worker happened to find.
    snap: Snapshot,
    work: Vec<(u32, u32)>,
    workers: usize,
    /// Reversed, so `next` pops instead of cloning a block per batch.
    out: Vec<Block>,
    ready: bool,
    stats: ScanStats,
}

impl<'a> Exchange<'a> {
    /// Drop the granules no zone filter can match, in parallel, and count them.
    ///
    /// Serial would do: at ~60 ns per granule this is 0.6 ms on a 10M-row
    /// table, which is 12% of a parallel full scan and *all* of a heavily
    /// pruned one. So it goes through the pool on the same contiguous split
    /// the scan is about to use, and the survivors are concatenated in order
    /// -- the result is the same list a serial pass would produce, which is
    /// what keeps the granule order (and therefore every merge below)
    /// deterministic.
    fn prune(&mut self) {
        let (work, node) = (&self.work, self.shape.node);
        let snap = &self.snap;
        let n = self.workers.min(work.len()).max(1);
        let kept: Vec<Vec<(u32, u32)>> = pool::global().map(n, |k| {
            let range = &work[span(work.len(), n, k)];
            let mut keep = Vec::new();
            for &(pi, gi) in range {
                if !prunes(node, snap.part(pi as usize), gi as usize) {
                    keep.push((pi, gi));
                }
            }
            keep
        });
        let survivors: usize = kept.iter().map(|v| v.len()).sum();
        self.stats.granules_pruned += (self.work.len() - survivors) as u64;
        let mut out = Vec::with_capacity(survivors);
        for v in kept {
            out.extend_from_slice(&v);
        }
        self.work = out;
    }

    fn materialize(&mut self) -> Result<()> {
        self.ready = true;
        self.ctx.check()?;
        if !self.shape.node.zone_filters.is_empty() {
            self.prune();
        }

        // At least one worker even with nothing to read: `SELECT count(*) FROM
        // empty` is one row holding 0, and that row only exists because a bare
        // aggregate creates its group before it sees any input.
        let n = self.workers.min(self.work.len().max(1));
        let (shape, snap, work, ctx) = (&self.shape, &self.snap, &self.work, self.ctx);
        let parts: Vec<Result<(Partial, ScanStats)>> = pool::global().map(n, |k| {
            let mine = &work[span(work.len(), n, k)];
            let pipe = shape.pipeline(snap, mine);
            run_shard(&shape.top, pipe, ctx)
        });

        let mut partials = Vec::with_capacity(n);
        for r in parts {
            let (p, st) = r?;
            self.stats.merge(&st);
            partials.push(p);
        }
        self.out = self.combine(partials)?;
        // Handed out back to front; see the `out` field.
        self.out.reverse();
        Ok(())
    }

    /// Fold the workers' partials, in worker order, into output blocks.
    fn combine(&self, partials: Vec<Partial>) -> Result<Vec<Block>> {
        match &self.shape.top {
            Top::Aggregate { group, aggs, schema } => {
                let mut guard = MemGuard::new(self.ctx, aggregate::guard_name(group.len()));
                let tables: Vec<aggregate::Groups> = partials
                    .into_iter()
                    .map(|p| match p {
                        Partial::Groups(g) => g,
                        Partial::Run(_) => unreachable!("shard shape follows the top"),
                    })
                    .collect();
                // The workers' own guards died with their frames, so the
                // partials are held but uncharged from the moment `map`
                // returns until here. The window is a few instructions wide
                // and the amount is exactly what was already charged a moment
                // earlier; leaving it uncharged for the whole merge would not
                // be.
                let held: usize = tables.iter().map(|t| t.bytes()).sum();
                guard.grow_to(held)?;
                let mut it = tables.into_iter();
                let mut base = it.next().expect("at least one worker always runs");
                for other in it {
                    base.absorb(other, aggs)?;
                    // Conservative by one partial's worth: `held` still counts
                    // `base`'s pre-merge size. Cheaper than tracking it, and
                    // the budget errs toward refusing.
                    guard.grow_to(held + base.bytes())?;
                }
                aggregate::emit(&base, group, aggs, schema)
            }
            Top::Sort { keys, fetch } => {
                let runs = partials
                    .into_iter()
                    .map(|p| match p {
                        Partial::Run(b) => b,
                        Partial::Groups(_) => unreachable!("shard shape follows the top"),
                    })
                    .collect();
                let mut guard = MemGuard::new(self.ctx, "the sort buffer");
                sort::merge_runs(runs, keys, *fetch, &mut guard)
            }
        }
    }
}

/// Run one worker's copy of the pipeline to completion.
fn run_shard(
    top: &Top<'_>,
    mut pipe: Box<dyn Operator + '_>,
    ctx: &QueryContext,
) -> Result<(Partial, ScanStats)> {
    match top {
        Top::Aggregate { group, aggs, schema: _ } => {
            let protos = aggregate::protos(aggs)?;
            let mut guard = MemGuard::new(ctx, aggregate::guard_name(group.len()));
            let g = aggregate::accumulate(&mut pipe, group, aggs, &protos, ctx, &mut guard)?;
            Ok((Partial::Groups(g), pipe.stats()))
        }
        Top::Sort { keys, fetch } => {
            // A real `Sort`, top-K bound and all: the worker's run is exactly
            // what a serial sort of its slice would produce, which is what
            // makes the k-way merge above a merge rather than a re-sort.
            let mut s: Box<dyn Operator + '_> = match fetch {
                Some(k) => Box::new(sort::Sort::top_k(pipe, keys, *k, ctx)),
                None => Box::new(sort::Sort::new(pipe, keys, ctx)),
            };
            let mut run: Option<Block> = None;
            while let Some(b) = s.next()? {
                match &mut run {
                    None => run = Some(b),
                    Some(a) => a.extend(&b)?,
                }
            }
            let stats = s.stats();
            let run = run.unwrap_or_else(|| Block::empty(s.schema()));
            Ok((Partial::Run(run), stats))
        }
    }
}

/// Worker `k`'s share of `len` items, split into `n` contiguous ranges.
#[inline]
fn span(len: usize, n: usize, k: usize) -> std::ops::Range<usize> {
    len * k / n..len * (k + 1) / n
}

impl Operator for Exchange<'_> {
    fn schema(&self) -> &Schema {
        self.schema
    }

    fn stats(&self) -> ScanStats {
        self.stats
    }

    fn next(&mut self) -> Result<Option<Block>> {
        if !self.ready {
            self.materialize()?;
        }
        Ok(self.out.pop())
    }
}

// ---------------------------------------------------------------- the entry

/// Replace `plan` with a parallel pipeline, or `None` to build it serially.
///
/// Every refusal here is a fall-through, never an error: a shape this cannot
/// run, a table it cannot resolve and a projection it cannot validate all
/// return `None` so the serial builder produces the same answer -- or the same
/// error message -- it always did.
pub fn try_build<'a>(
    plan: &PhysicalPlan<'a>,
    catalog: &'a Catalog,
    ctx: &'a QueryContext,
) -> Result<Option<Box<dyn Operator + 'a>>> {
    let Some(shape) = analyze(plan) else { return Ok(None) };
    let Ok(table) = catalog.table_by_path(&shape.node.table) else { return Ok(None) };
    // `Scan::new` owns the message for an out-of-range projection, and a
    // parallel path that produced a different one would be a second source of
    // truth for the same mistake.
    let ncols = table.schema().len();
    if shape.node.projection.iter().any(|&c| c >= ncols) {
        return Ok(None);
    }

    let snap = table.snapshot();
    let granules: usize = (0..snap.len()).map(|i| snap.part(i).granule_count()).sum();
    let workers = degree(snap.live_rows(), granules);
    if workers <= 1 {
        return Ok(None);
    }

    let work: Vec<(u32, u32)> = (0..snap.len())
        .flat_map(|pi| (0..snap.part(pi).granule_count()).map(move |g| (pi as u32, g as u32)))
        .collect();
    let schema = shape.schema();
    Ok(Some(Box::new(Exchange {
        shape,
        ctx,
        schema,
        snap,
        work,
        workers,
        out: Vec::new(),
        ready: false,
        stats: ScanStats::default(),
    })))
}

/// [`operators::build_physical`], with the exchange consulted at every node.
///
/// Only the operators that can legitimately sit *above* a parallel subtree are
/// recursed through here; everything else is handed to the serial builder
/// whole, which is what stops this from being a second copy of `build_physical`
/// that goes stale. A plan node added elsewhere keeps working and simply stays
/// serial.
///
/// `Sort` is in the list because `SELECT k, count(*) ... GROUP BY k ORDER BY k`
/// wants the *aggregate* parallelized with the sort serial above it, and the
/// per-node decision is what gets that right without a special case.
pub fn build<'a>(
    plan: PhysicalPlan<'a>,
    catalog: &'a Catalog,
    ctx: &'a QueryContext,
) -> Result<Box<dyn Operator + 'a>> {
    if let Some(op) = try_build(&plan, catalog, ctx)? {
        return Ok(op);
    }
    Ok(match plan {
        PhysicalPlan::Filter { input, predicate } => {
            Box::new(filter::Filter::new(build(*input, catalog, ctx)?, predicate))
        }
        PhysicalPlan::Project { input, exprs, schema } => {
            Box::new(project::Project::new(build(*input, catalog, ctx)?, exprs, schema))
        }
        PhysicalPlan::Aggregate { input, group, aggs, schema } => Box::new(
            aggregate::Aggregate::new(build(*input, catalog, ctx)?, group, aggs, schema, ctx)?,
        ),
        PhysicalPlan::Sort { input, keys, fetch } => {
            let inner = build(*input, catalog, ctx)?;
            Box::new(match fetch {
                Some(k) => sort::Sort::top_k(inner, keys, k, ctx),
                None => sort::Sort::new(inner, keys, ctx),
            })
        }
        PhysicalPlan::Limit { input, limit, offset } => {
            Box::new(limit::Limit::new(build(*input, catalog, ctx)?, limit, offset, ctx))
        }
        PhysicalPlan::LimitBy { input, limit, keys } => {
            Box::new(limit::LimitBy::new(build(*input, catalog, ctx)?, limit, keys, ctx))
        }
        PhysicalPlan::Distinct { input } => {
            Box::new(distinct::Distinct::new(build(*input, catalog, ctx)?))
        }
        PhysicalPlan::Join { left, right, op, on, residual, schema } => Box::new(join::Join::new(
            build(*left, catalog, ctx)?,
            build(*right, catalog, ctx)?,
            op,
            on,
            residual,
            schema,
            ctx,
        )),
        // Leaves and the shapes with private constructors: nothing above them
        // to parallelize, so the serial builder is the whole answer.
        other => operators::build_physical(other, catalog, ctx)?,
    })
}

/// [`operators::execute_ctx`] with the exchange in the plan.
///
/// Byte-for-byte the same loop, over a pipeline built by [`build`] instead of
/// `build_physical`. It is here rather than in `operators/mod.rs` for the
/// reason given in `exec/mod.rs`, and `session.rs` reaching this instead of
/// the serial one is the single call site that turns the exchange on.
pub fn execute_ctx<'a>(
    plan: &'a crate::planner::logical::LogicalPlan,
    catalog: &'a Catalog,
    ctx: &'a QueryContext,
) -> Result<(Vec<Block>, ScanStats)> {
    let mut op = build(crate::planner::physical::lower(plan, catalog)?, catalog, ctx)?;
    let mut out = Vec::new();
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

/// [`operators::execute_with_stats`] with the exchange in the plan.
///
/// Exists so the one call site that matters -- `Session::run_query`, which
/// today calls `operators::execute_with_stats(&plan, &self.catalog)` -- can
/// switch by changing the path and nothing else. `operators::ambient_context`
/// is private to that module, so this keeps its own; both are a
/// default-budget, no-deadline context and a query cannot tell them apart.
pub fn execute_with_stats<'a>(
    plan: &'a crate::planner::logical::LogicalPlan,
    catalog: &'a Catalog,
) -> Result<(Vec<Block>, ScanStats)> {
    static C: std::sync::OnceLock<QueryContext> = std::sync::OnceLock::new();
    execute_ctx(plan, catalog, C.get_or_init(QueryContext::new))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::binder::Binder;
    use crate::planner::logical::LogicalPlan;
    use crate::planner::{optimizer, physical};
    use crate::types::Value;
    use crate::Session;

    /// A table wide enough to be worth parallelizing, with every shape the
    /// merges care about: a low-cardinality key, a high-cardinality one, a
    /// string, a float, and a nullable column.
    ///
    /// `ROWS` is chosen so `degree` gives several workers -- 128 granules over
    /// `MIN_GRANULES_PER_WORKER` is 8 -- because a merge bug that only shows up
    /// between two partials is a merge bug that a two-worker test misses.
    const ROWS: i64 = 131_072;

    fn session() -> Session {
        let mut s = Session::in_memory();
        s.execute(
            "CREATE TABLE t (id UInt64, k UInt64, big UInt64, f Float64, \
             s String, n Nullable(Int64)) ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
        )
        .unwrap();
        let names = ["ann", "bob", "cyd", "dee", "eve"];
        let mut sql = String::from("INSERT INTO t VALUES ");
        for i in 0..ROWS {
            if i > 0 {
                sql.push(',');
            }
            let n = if i % 11 == 0 { "NULL".to_string() } else { (i % 977).to_string() };
            sql.push_str(&format!(
                "({i},{},{},{}.5,'{}',{n})",
                i % 8,
                crate::common::splitmix64(i as u64) % 100_000,
                i % 1_000,
                names[(i % 5) as usize]
            ));
        }
        s.execute(&sql).unwrap();
        s.catalog.flush_all().unwrap();
        s
    }

    fn plan_of(s: &mut Session, sql: &str) -> LogicalPlan {
        s.catalog.flush_all().unwrap();
        let stmts = crate::sql::parser::parse(sql).unwrap();
        let q = match &stmts[0] {
            crate::sql::ast::Statement::Query(q) => q.clone(),
            other => panic!("not a query: {other:?}"),
        };
        optimizer::optimize(Binder::new(&s.catalog).bind_query(&q).unwrap()).unwrap()
    }

    fn rows_of(blocks: &[Block]) -> Vec<Vec<Value>> {
        blocks
            .iter()
            .flat_map(|b| {
                (0..b.rows())
                    .map(move |r| (0..b.width()).map(|c| b.column(c).value(r)).collect::<Vec<_>>())
            })
            .collect()
    }

    /// Serial and parallel results for one query, in emission order.
    fn both(s: &mut Session, sql: &str) -> (Vec<Vec<Value>>, Vec<Vec<Value>>) {
        let plan = plan_of(s, sql);
        let ctx = QueryContext::new();
        let serial = operators::execute_ctx(&plan, &s.catalog, &ctx).unwrap().0;
        let parallel = execute_ctx(&plan, &s.catalog, &ctx).unwrap().0;
        (rows_of(&serial), rows_of(&parallel))
    }

    /// Did the exchange actually fire for this query? A test that only checks
    /// agreement passes trivially when the parallel path silently declined.
    fn goes_parallel(s: &mut Session, sql: &str) -> bool {
        let plan = plan_of(s, sql);
        let lowered = physical::lower(&plan, &s.catalog).unwrap();
        fired(&lowered, &s.catalog)
    }

    fn fired(p: &PhysicalPlan<'_>, cat: &Catalog) -> bool {
        let ctx = QueryContext::new();
        if try_build(p, cat, &ctx).unwrap().is_some() {
            return true;
        }
        p.children().iter().any(|c| fired(c, cat))
    }

    // ------------------------------------------------- parallel == serial
    //
    // The oracle in `tests/differential.rs` validates the *serial* engine
    // against sqlite. So `parallel == serial` is what closes the loop: if
    // serial == sqlite and parallel == serial, then parallel == sqlite, and
    // the exchange inherits every case the oracle has ever run. That is the
    // same argument `scan.rs` makes for the index path against the scan.

    #[test]
    fn parallel_agrees_with_serial_row_for_row() {
        let mut s = session();
        // Byte-identical, *including order*: the contiguous split plus an
        // in-order merge is supposed to reproduce first-seen group order and
        // stable sort ties exactly, not merely produce the same multiset.
        for q in [
            // bare aggregates, one per merge shape
            "SELECT count(*) FROM t",
            "SELECT sum(k), min(k), max(k), avg(k) FROM t",
            "SELECT sum(f), avg(f) FROM t",
            "SELECT any(s), anyLast(s) FROM t",
            "SELECT argMin(s, k), argMax(s, k) FROM t",
            "SELECT uniq(big) FROM t",
            "SELECT count(n), sum(n), min(n), max(n) FROM t",
            "SELECT quantile(0.9)(k) FROM t",
            "SELECT median(f) FROM t",
            // GROUP BY, low and high cardinality, and on a string
            "SELECT k, count(*) FROM t GROUP BY k",
            "SELECT s, count(*), sum(k) FROM t GROUP BY s",
            "SELECT big, count(*) FROM t GROUP BY big",
            "SELECT k, s, count(*) FROM t GROUP BY k, s",
            "SELECT n, count(*) FROM t GROUP BY n",
            "SELECT k % 3, count(*) FROM t GROUP BY k % 3",
            "SELECT k, any(s), anyLast(s), argMin(s, id) FROM t GROUP BY k",
            "SELECT k, groupArray(s) FROM t GROUP BY k",
            // DISTINCT aggregates: the seen-sets have to union, not add up
            "SELECT count(DISTINCT k) FROM t",
            "SELECT count(DISTINCT big) FROM t",
            "SELECT sum(DISTINCT k) FROM t",
            "SELECT uniq(DISTINCT s) FROM t",
            "SELECT k, count(DISTINCT s) FROM t GROUP BY k",
            "SELECT s, count(DISTINCT k), sum(DISTINCT k), count(k) FROM t GROUP BY s",
            "SELECT count(DISTINCT n) FROM t",
            // filters and projections between the scan and the top
            "SELECT count(*) FROM t WHERE k = 3",
            "SELECT sum(k) FROM t WHERE id > 100000",
            "SELECT s, count(*) FROM t WHERE n IS NOT NULL GROUP BY s",
            "SELECT sum(k * 2 + 1) FROM t WHERE s != 'ann'",
            // an empty result from a filter nothing survives
            "SELECT count(*), sum(k) FROM t WHERE k = 99",
            "SELECT k, count(*) FROM t WHERE k = 99 GROUP BY k",
            // ORDER BY: full sorts on both strategies, and top-K
            "SELECT id FROM t ORDER BY id DESC",
            "SELECT k, id FROM t ORDER BY k",
            "SELECT s, id FROM t ORDER BY s, id DESC",
            "SELECT n FROM t ORDER BY n",
            "SELECT id FROM t ORDER BY id LIMIT 10",
            "SELECT k, id FROM t ORDER BY k LIMIT 100",
            "SELECT s, k FROM t ORDER BY s, k DESC LIMIT 25",
            "SELECT id FROM t ORDER BY f DESC LIMIT 7",
            "SELECT k FROM t WHERE id % 3 = 0 ORDER BY k LIMIT 50",
            // the sort sits above a parallel aggregate rather than being one
            "SELECT k, count(*) FROM t GROUP BY k ORDER BY k DESC",
            "SELECT s, sum(k) FROM t GROUP BY s ORDER BY sum(k) LIMIT 3",
        ] {
            assert!(goes_parallel(&mut s, q), "the exchange declined `{q}`");
            let (serial, parallel) = both(&mut s, q);
            assert_eq!(parallel, serial, "parallel disagrees with serial on `{q}`");
        }
    }

    #[test]
    fn a_float_sum_lands_within_rounding_of_the_serial_one() {
        // The one answer the exchange is allowed to change. Compensated
        // summation is not associative, so folding fourteen partial sums is
        // not the same sequence of roundings as one long one -- but the whole
        // point of the compensation is that the gap stays at the bottom of the
        // mantissa. Deterministic across runs, because the split is static.
        let mut s = session();
        for q in ["SELECT sum(f) FROM t", "SELECT avg(f) FROM t", "SELECT k, sum(f) FROM t GROUP BY k"] {
            let (serial, parallel) = both(&mut s, q);
            assert_eq!(serial.len(), parallel.len(), "{q}");
            for (a, b) in serial.iter().zip(&parallel) {
                let (x, y) = (a[a.len() - 1].as_f64().unwrap(), b[b.len() - 1].as_f64().unwrap());
                assert!((x - y).abs() <= x.abs() * 1e-12, "{q}: {x} vs {y}");
            }
        }
    }

    // -------------------------------------------------------- the decision

    #[test]
    fn a_small_query_stays_serial() {
        // The threshold has to bite, or every point query pays for a fleet it
        // cannot use. `MIN_PARALLEL_ROWS` rows exactly is the boundary.
        let mut s = Session::in_memory();
        s.execute("CREATE TABLE small (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id")
            .unwrap();
        let mut sql = String::from("INSERT INTO small VALUES ");
        for i in 0..(MIN_PARALLEL_ROWS as i64 - 1) {
            if i > 0 {
                sql.push(',');
            }
            sql.push_str(&format!("({i},{})", i % 7));
        }
        s.execute(&sql).unwrap();
        assert!(!goes_parallel(&mut s, "SELECT count(*) FROM small"));
        assert!(!goes_parallel(&mut s, "SELECT v, count(*) FROM small GROUP BY v"));
        assert!(!goes_parallel(&mut s, "SELECT id FROM small ORDER BY id LIMIT 5"));
        // ... and one more row tips it over.
        s.execute(&format!("INSERT INTO small VALUES ({MIN_PARALLEL_ROWS}, 1)")).unwrap();
        assert!(goes_parallel(&mut s, "SELECT count(*) FROM small"));
    }

    #[test]
    fn degree_refuses_before_it_divides() {
        assert_eq!(degree(0, 0), 1);
        assert_eq!(degree(MIN_PARALLEL_ROWS - 1, 10_000), 1);
        // Enough rows but nowhere near enough granules to spread them.
        assert_eq!(degree(1 << 30, MIN_GRANULES_PER_WORKER - 1), 1);
        assert_eq!(degree(1 << 30, MIN_GRANULES_PER_WORKER * 2), 2.min(pool::global().threads()));
        assert!(degree(1 << 30, 1 << 20) <= pool::global().threads());
    }

    #[test]
    fn the_shapes_the_exchange_must_not_touch() {
        let mut s = session();
        s.execute("CREATE TABLE d (cid UInt64, name String) ENGINE = MergeTree ORDER BY cid")
            .unwrap();
        s.execute("INSERT INTO d VALUES (0,'zero'),(1,'one'),(2,'two')").unwrap();
        for q in [
            // a join anywhere under the top: the build side is not replicable
            "SELECT count(*) FROM t JOIN d ON t.k = d.cid",
            // streaming operators whose output order is first-seen
            "SELECT DISTINCT k FROM t",
            "SELECT id, k FROM t LIMIT 2 BY k",
            // a point lookup is already microseconds
            "SELECT k FROM t WHERE id = 5",
            "SELECT k FROM t WHERE id IN (5, 9, 12)",
            // no blocking top at all: concatenating shards would reorder rows
            // a plain SELECT is entitled to keep
            "SELECT id FROM t",
            "SELECT id FROM t WHERE k = 1",
            "SELECT id FROM t LIMIT 10",
            "SELECT count(*) FROM (SELECT 1) x",
        ] {
            assert!(!goes_parallel(&mut s, q), "the exchange took `{q}`");
        }
        // ... and every one of them still answers the same as before.
        for q in [
            "SELECT DISTINCT k FROM t",
            "SELECT id FROM t LIMIT 10",
            "SELECT k FROM t WHERE id = 5",
        ] {
            let (serial, parallel) = both(&mut s, q);
            assert_eq!(parallel, serial, "{q}");
        }
    }

    // ------------------------------------------------------------- details

    #[test]
    fn counters_add_up_across_the_workers() {
        let mut s = session();
        let plan = plan_of(&mut s, "SELECT count(*) FROM t");
        let ctx = QueryContext::new();
        let (_, serial) = operators::execute_ctx(&plan, &s.catalog, &ctx).unwrap();
        let (_, par) = execute_ctx(&plan, &s.catalog, &ctx).unwrap();
        assert_eq!(par, serial, "a split scan must report the same work");
        assert_eq!(par.rows_read, ROWS as u64);
    }

    #[test]
    fn zone_pruning_happens_once_up_front_and_is_still_counted() {
        // The prune pass is the exchange's, not the workers': the same
        // granules have to be skipped and the same number reported, or the
        // split is balanced on work that does not exist.
        let mut s = session();
        let q = "SELECT count(*) FROM t WHERE id >= 130000";
        let plan = plan_of(&mut s, q);
        let ctx = QueryContext::new();
        let (sb, serial) = operators::execute_ctx(&plan, &s.catalog, &ctx).unwrap();
        let (pb, par) = execute_ctx(&plan, &s.catalog, &ctx).unwrap();
        assert_eq!(rows_of(&pb), rows_of(&sb));
        assert_eq!(par, serial, "pruning moved but did not change");
        assert!(par.granules_pruned > 100, "only {} pruned", par.granules_pruned);
    }

    #[test]
    fn an_empty_table_still_answers_a_bare_aggregate() {
        // A fold always has a result, and the exchange has to run a worker
        // over an empty granule list to produce it.
        let mut s = session();
        s.execute("ALTER TABLE t DELETE WHERE id >= 0").unwrap();
        s.catalog.flush_all().unwrap();
        let plan = plan_of(&mut s, "SELECT count(*), sum(k) FROM t");
        let ctx = QueryContext::new();
        let got = rows_of(&execute_ctx(&plan, &s.catalog, &ctx).unwrap().0);
        assert_eq!(got, vec![vec![Value::UInt(0), Value::Null]]);
        let plan = plan_of(&mut s, "SELECT k, count(*) FROM t GROUP BY k");
        assert!(execute_ctx(&plan, &s.catalog, &ctx).unwrap().0.is_empty());
    }

    #[test]
    fn cancellation_and_the_deadline_reach_every_worker() {
        let mut s = session();
        let plan = plan_of(&mut s, "SELECT big, count(*) FROM t GROUP BY big");
        let ctx = QueryContext::new();
        ctx.stop();
        let e = execute_ctx(&plan, &s.catalog, &ctx).unwrap_err();
        assert!(e.to_string().contains("cancelled"), "{e}");

        let ctx = QueryContext::new().deadline_in(std::time::Duration::from_nanos(1));
        std::thread::sleep(std::time::Duration::from_millis(2));
        let e = execute_ctx(&plan, &s.catalog, &ctx).unwrap_err();
        assert!(e.to_string().contains("deadline"), "{e}");
    }

    #[test]
    fn the_budget_is_one_ceiling_for_the_whole_fleet() {
        // The failure this guards: N workers each holding their own group
        // table against N copies of the budget, so a 14-thread query quietly
        // gets 14x the memory the user allowed.
        let mut s = session();
        let plan = plan_of(&mut s, "SELECT big, count(*) FROM t GROUP BY big");
        let tight = QueryContext::with_budget(256 << 10);
        let e = execute_ctx(&plan, &s.catalog, &tight).unwrap_err();
        assert!(e.to_string().contains("memory budget"), "{e}");
        assert_eq!(tight.mem.used(), 0, "a failed parallel query kept its reservation");

        // ... and a completed one hands all of it back.
        let ctx = QueryContext::new();
        let (blocks, _) = execute_ctx(&plan, &s.catalog, &ctx).unwrap();
        assert!(!blocks.is_empty());
        assert_eq!(ctx.mem.used(), 0);
    }

    #[test]
    fn deletes_are_honoured_by_every_worker() {
        let mut s = session();
        s.execute("ALTER TABLE t DELETE WHERE id % 7 = 0").unwrap();
        s.catalog.flush_all().unwrap();
        for q in ["SELECT count(*) FROM t", "SELECT k, count(*) FROM t GROUP BY k", "SELECT id FROM t ORDER BY id LIMIT 20"] {
            let (serial, parallel) = both(&mut s, q);
            assert_eq!(parallel, serial, "{q}");
        }
    }

    #[test]
    fn the_same_query_answers_identically_every_time() {
        // The claim the static contiguous split exists to make. A shared work
        // cursor would pass every agreement test above on a lucky run and then
        // hand a different `any(s)`, a different group order and a different
        // tie order to the next caller. Repetition is the only way to see it.
        let mut s = session();
        for q in [
            "SELECT any(s), anyLast(s), argMin(s, id) FROM t",
            "SELECT k, any(s), groupArray(s) FROM t GROUP BY k",
            "SELECT s, count(*) FROM t GROUP BY s",
            "SELECT k, id FROM t ORDER BY k LIMIT 40",
            "SELECT sum(f) FROM t",
        ] {
            let plan = plan_of(&mut s, q);
            let ctx = QueryContext::new();
            let first = rows_of(&execute_ctx(&plan, &s.catalog, &ctx).unwrap().0);
            for i in 1..20 {
                let again = rows_of(&execute_ctx(&plan, &s.catalog, &ctx).unwrap().0);
                assert_eq!(again, first, "run {i} of `{q}` differs from run 0");
            }
        }
    }

    #[test]
    fn parts_deletes_and_zone_maps_all_at_once() {
        // Each of these moves the work list on its own; together they are the
        // shape where a split, a prune and a delete mask have to agree about
        // which granule is which.
        let mut s = session();
        for base in 0..3u64 {
            let mut sql = String::from("INSERT INTO t VALUES ");
            for i in 0..30_000u64 {
                if i > 0 {
                    sql.push(',');
                }
                let id = 500_000 + base * 100_000 + i;
                sql.push_str(&format!("({id},{},{i},2.5,'qq',{})", i % 8, i % 5));
            }
            s.execute(&sql).unwrap();
        }
        s.execute("ALTER TABLE t DELETE WHERE id % 13 = 0").unwrap();
        s.catalog.flush_all().unwrap();
        assert!(s.catalog.table_by_path("default.t").unwrap().part_count() >= 2);
        for q in [
            "SELECT count(*) FROM t WHERE id >= 500000",
            "SELECT k, count(*), min(id), max(id) FROM t WHERE id > 100000 GROUP BY k",
            "SELECT id FROM t WHERE id >= 550000 ORDER BY id LIMIT 25",
            "SELECT s, count(DISTINCT k) FROM t WHERE id < 600000 GROUP BY s",
        ] {
            assert!(goes_parallel(&mut s, q), "the exchange declined `{q}`");
            let (serial, parallel) = both(&mut s, q);
            assert_eq!(parallel, serial, "{q}");
        }
    }

    #[test]
    fn several_parts_are_split_across_workers_like_one() {
        // The work list spans parts, so a worker's contiguous range can start
        // mid-part and end in the next one.
        let mut s = session();
        for base in [0u64, 1] {
            let mut sql = String::from("INSERT INTO t VALUES ");
            for i in 0..40_000u64 {
                if i > 0 {
                    sql.push(',');
                }
                let id = 1_000_000 + base * 100_000 + i;
                sql.push_str(&format!("({id},{},{i},1.5,'zz',7)", i % 8));
            }
            s.execute(&sql).unwrap();
            s.catalog.flush_all().unwrap();
        }
        assert!(s.catalog.table_by_path("default.t").unwrap().part_count() >= 2);
        for q in ["SELECT count(*) FROM t", "SELECT k, count(*), any(s) FROM t GROUP BY k", "SELECT id FROM t ORDER BY id DESC LIMIT 30"] {
            let (serial, parallel) = both(&mut s, q);
            assert_eq!(parallel, serial, "{q}");
        }
    }


}
