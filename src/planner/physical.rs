//! The physical plan: where access-path decisions live.
//!
//! The logical plan says *what* rows a query wants. This says *how* the engine
//! will get them. The distinction only earns its keep once there is a choice to
//! make, and until this module existed there was none: `exec::operators::build`
//! was a 1:1 structural mapping from [`LogicalPlan`] to operators, with no
//! place to put a decision even if one had been obvious.
//!
//! Five decisions live here today.
//!
//! ## 1. Index selection
//!
//! `Table::locate` resolves a primary key through a CHD minimal perfect hash
//! plus a split-block bloom filter in ~59 ns. The SQL path never used it: a
//! `WHERE id = <const>` became a [`Scan`](PhysicalPlan::Scan) that walked every
//! granule header in the table testing zone maps. That is O(rows/1024) `Value`
//! comparisons for an answer the storage layer can produce in one probe.
//!
//! So `lower` recognizes the two predicate shapes storage can answer directly,
//! `pk = <const>` and `pk IN (<consts>)`, and lowers the scan to
//! [`PhysicalPlan::IndexLookup`]. Measured through the SQL front end on a
//! 10M-row table, A/B interleaved in one loop with a temporary switch that
//! disabled the decision, best-of-25 per side:
//!
//! ```text
//!   SELECT v FROM t WHERE id = 3133711            138.5 us ->  3.6 us    38x
//!   SELECT v FROM t WHERE id IN (5 constants)     180.4 ms -> 12.4 us  14526x
//!   SELECT v FROM t WHERE id = 3133711 AND ...    144.9 us ->  4.8 us    30x
//! ```
//!
//! The `IN` figure is the one to look at: an `IN` list makes the zone map
//! useless (it can only bound `[min, max]`, so five scattered keys prune
//! nothing) *and* costs the scan an O(list) membership test per row. The
//! remaining 3.6 us is almost entirely front end -- `SELECT 1` costs 1.2 us on
//! the same build.
//!
//! `id = 1 OR id = 2` is the same predicate written the other way and used to
//! reach none of this: `split_conjuncts` stops at an `OR`, so the disjunction
//! arrived at [`key_set`] as one conjunct and matched no shape. It now folds
//! into the same probe list -- 333x on 1M rows, with the measurements and the
//! refusals next to the code.
//!
//! The negative cases matter as much, and were measured the same way: full
//! scans, range scans, `GROUP BY`, a predicate on a non-key column, `id > k`,
//! `id != k` and `ORDER BY ... LIMIT` all land between 0.98x and 1.01x. A
//! decision that fires when it should not is a performance bug in the other
//! direction, so "nothing changed" is the required result, not an afterthought.
//!
//! ## 2. Top-K fusion
//!
//! `ORDER BY x LIMIT n` needs only `n + offset` rows, so the sort under the
//! limit keeps a bounded heap instead of materializing the whole input. That
//! decision used to be a peephole inside `build`; it is a physical property of
//! the sort (`Sort::top_k` vs `Sort::new`), so it is a field on
//! [`PhysicalPlan::Sort`] and shows up in `EXPLAIN` as `TopK` rather than
//! `Sort`.
//!
//! ## 3. Parallelism
//!
//! How many workers a query gets is the third decision, and it used to be
//! taken *inside* `exchange::try_build` at operator-construction time. That
//! made it invisible: `EXPLAIN PIPELINE` for a 400k-row `GROUP BY` was
//! byte-identical to the serial plan, so neither a user nor a benchmark could
//! tell whether a query had gone parallel, and a regression that silently
//! turned the exchange off would not have shown up anywhere. It is now
//! [`PhysicalPlan::Exchange`], produced here by calling the pure
//! [`exchange::degree`], and `exchange::build` obeys the node rather than
//! re-deciding. The width is in the plan text, which is the whole point.
//!
//! The node is emitted only where the builder can honour it. `Window` and the
//! branches of a `UNION` are built by the *serial* builder --
//! `union::build_union` re-lowers each branch through `operators::build`, and
//! `exchange::build` hands `Window` straight to `build_physical` -- so an
//! `Exchange` under either would be a plan node that prints and never runs,
//! which is the exact defect this change exists to remove. `lower` therefore
//! carries a "the builder above me can honour this" flag; see [`lower_at`].
//!
//! Moving the decision must not move any query, and it does not: the same
//! shapes go parallel at the same width, because [`exchange::shard_stats`] runs
//! the same checks against the same snapshot that `try_build` used to. Both
//! variants were built into **one** binary behind a temporary switch and
//! interleaved in a single loop with the sides alternating which ran first,
//! best-of-31 and paired-median per side, 2M rows, 14 cores:
//!
//! ```text
//!                         legacy    node   best  median
//!   count                  3.22    3.28   0.98x  1.00x
//!   sum                    3.98    4.07   0.98x  0.98x
//!   sum + filter           7.45    7.49   0.99x  1.03x
//!   group by country      17.18   16.65   1.03x  1.05x
//!   uniq (HLL)             7.85    7.17   1.09x  1.04x
//!   top-k by sort         10.05   10.49   0.96x  1.00x
//!   full sort             54.44   56.42   0.96x  0.98x
//!   group by high-card   353.7   366.6   0.96x  1.00x
//!   point lookup           0.009   0.009  1.00x  1.00x
//!   small count            0.035   0.036  0.97x  1.00x
//!   small group by         0.134   0.133  1.00x  1.00x
//!   small top-k            0.121   0.122  0.99x  1.00x
//! ```
//!
//! The last four are the ones that carry information. Everything above them
//! disagrees between the two statistics -- `count` best-of says 0.98x and the
//! paired median says 1.00x, `uniq` says 1.09x and 1.04x -- which is what noise
//! looks like on a machine that swings 30%. The bottom four are queries whose
//! whole cost *is* planning, so a snapshot added per plan node would show up
//! there and nowhere else, and they are 1.00x to three digits. That is the
//! measurement that matters: `lower` takes one extra `RwLock::read` + `Arc`
//! clone (~40 ns) per blocking node, against a 35 us query, and the builder
//! now does strictly *less* work than it did -- `try_build` used to re-run the
//! shape match at every one of the plan's nodes, including the leaves.
//!
//! ## 4. Answering from metadata
//!
//! `SELECT count() FROM t` used to read every row of `t` to find out how many
//! there were: 1.29 ms and 1555 M rows/s on 2M rows, a *throughput* number for
//! a question with no rows in the answer. But a part already records how many
//! rows each granule holds and a `PartSet` already tracks its tombstones
//! separately, so an unfiltered count is a fold over metadata --
//! `Σ n_rows - deleted`, one term per part, independent of the table's size.
//! The same is true of `min`/`max`: the zone maps that prune granules are
//! per-granule `(min, max)` pairs, and the extreme of the table is the extreme
//! of those pairs.
//!
//! [`MetaPath`] is that decision, and [`MetaAggregate`] runs it. Both
//! variants were built into **one** binary behind a temporary switch and
//! interleaved in a single loop, the sides alternating which ran first,
//! best-of-N per side over 5-15 rounds; 2M rows / 1954 granules, 14 cores, on
//! a loaded machine. Two independent sessions, reported as a range where they
//! disagree:
//!
//! ```text
//!                                            scanned      folded    speedup
//!   count()                              2.18-2.41 ms   5.8-6.5 us   338-420x
//!   min(bytes), max(bytes)               4.37-4.92 ms    60-64 us     68-82x
//!   min(country), max(country)           9.80-14.2 ms   193-215 us    46-74x
//!   count() WHERE ts >= <50% cut>        1.40-1.41 ms    43-44 us        32x
//!   count() WHERE ts >= <90% cut>          353-433 us    37-50 us    8.7-9.6x
//!   count() WHERE ts < <1% cut>            109-113 us    35-37 us       3.1x
//!   count() WHERE ts BETWEEN <1000 rows>  53.9-58.4 us    46-55 us   1.1-1.2x
//!   count(), table carrying tombstones   2.76-2.88 ms  1.53-1.66 ms  1.7-1.8x
//!   count() inside an open transaction   1.57-1.93 ms   5.1-6.5 us   296-309x
//! ```
//!
//! And the shapes this must *not* move, measured the same way: an undecidable
//! predicate (`latency > 500`, refused) 1.0-1.1x, a predicate no zone test can
//! express (`user_id % 7 = 0`, refused) 0.9-1.2x, `sum(bytes)` 1.1x, `GROUP BY
//! country` 1.1x, `max(bytes)` on a table with tombstones (refused) 1.1x.
//! Nothing moves, which is the required result rather than an afterthought.
//!
//! The claim is really the scaling, not the ratio: 2M / 4M / 8M rows fold in
//! 5.8 / 5.8 / 5.2 us while the scan they replace takes 2.47 / 4.26 / 10.99 ms.
//! `min`/`max` is linear in the *granule* count rather than the row count
//! (1954 zone maps to fold, not 2M values to decode), which is the 1024x
//! reduction the granule size buys; a string column pays three times as much
//! because every bound is decoded through that granule's own dictionary.
//!
//! ### Reaching it through a derived table
//!
//! `SELECT count() FROM (SELECT k FROM t) u` binds to `Aggregate` over
//! `Project` over `Scan`: the subquery is inlined, but its column list survives
//! as a projection, and the fold used to insist on sitting *directly* on the
//! scan. So the commonest way to write a subquery lost the fold entirely, on a
//! query whose answer is a row count. A column projection cannot add or drop a
//! row, so [`meta_source`] looks through it -- 128x on 1M rows, with the table
//! of measurements there.
//!
//! ### Why a filtered count is the general case
//!
//! A granule whose zone map proves that *every* row matches the predicate
//! contributes its live row count without being decoded; one that proves *no*
//! row matches contributes zero; only the granules that straddle the boundary
//! -- at most one per conjunct on a clustered column -- have to be read. An
//! unfiltered count is just the case where every granule is covered.
//!
//! "Every row matches `P`" is not a new kind of reasoning and deliberately is
//! not implemented as one: it is "**no** row matches `¬P`", tested by the same
//! [`ZoneFilter::may_match`] the scan already prunes with. That matters,
//! because pruning is already load-bearing for correctness -- a `may_match`
//! that answered "no" while a matching row existed would drop rows from every
//! query in the engine -- so the covering test inherits exactly the trust the
//! engine already places in it, instead of introducing a second comparison
//! that could disagree with the evaluator. The one extra condition is NULLs:
//! three-valued logic means a NULL row satisfies neither `P` nor `¬P`, so a
//! granule is only *covered* if the predicate's column has no NULL in it.
//!
//! ### What is refused, and why
//!
//! * **`min`/`max` on a table with any tombstone.** A granule's `(min, max)`
//!   bounds the rows it was *built* from, and deleting the row that held the
//!   minimum does not narrow them -- so the fold would answer with a value
//!   that is no longer in the table. Approximate is wrong, so the shortcut is
//!   refused and the scan runs. (`count` has no such problem: the delete
//!   masks are exact and are what makes it exact.)
//! * **A pending delta.** Every read flushes first
//!   (`Session::exec_statement`), so parts *are* the table by the time this
//!   runs, and the scan next door would be equally wrong if they were not.
//!   Refused anyway: it is one `is_empty` against a class of bug a
//!   differential test cannot see.
//! * **`count(x)`, `count(DISTINCT x)`, `sum`, and every other aggregate.**
//!   `count(x)` counts non-NULL values, which no granule header records.
//! * **A conjunct that is not `col <op> literal`.** The predicate the
//!   operator falls back to on a straddling granule is `ScanNode::filters`
//!   itself, so the two must be the *same* predicate: the zone tests are
//!   derived here from the filter list rather than read out of
//!   `ScanNode::zone_filters`, which is a deliberately lossy summary (an `IN`
//!   list becomes a `[min, max]` range) and would over-count if trusted as an
//!   equivalent.
//!
//! ## 5. Reading in the order it is already stored in
//!
//! `SELECT k FROM t ORDER BY k LIMIT 5` used to cost 0.41 ms against 0.009 ms
//! for the same query without the `ORDER BY` -- a `TopK` under a 14-worker
//! `Exchange`, to ask for the order the rows are already lying in. A `MergeTree`
//! part *is* a sorted run and [`Scan`](PhysicalPlan::Scan) walks parts in set
//! order and granules in part order, so for the right `ORDER BY` the sort was
//! being asked to reproduce the order of its own input. [`read_in_order`] is
//! the decision; there is no new operator, because the answer is that the sort
//! should not be there.
//!
//! A/B interleaved in one loop with a temporary switch, the sides alternating
//! which ran first, best-of-25 per side (best-of-9 for the unlimited sorts),
//! 300k rows / 14 cores unless stated:
//!
//! ```text
//!                                              sorted    read in order
//!   ORDER BY k LIMIT 5                         0.414 ms    0.0085 ms   48.9x
//!   ORDER BY k LIMIT 5, two columns            0.445       0.0145      30.7x
//!   ORDER BY k LIMIT 1000                      0.496       0.0090      55.3x
//!   ORDER BY k LIMIT 5 OFFSET 100000           0.606       0.0625       9.7x
//!   ORDER BY k, no limit                       0.751       0.160        4.7x
//!   ORDER BY k, no limit, two columns          1.001       0.399        2.5x
//!   SELECT *, six columns, no limit            2.073       1.183        1.8x
//!   WHERE k >= <99% cut> ORDER BY k LIMIT 5    0.084       0.0095       8.8x
//!   WHERE k >= <99% cut> ORDER BY k            0.063       0.0092       6.9x
//!   ORDER BY k LIMIT 5, four disjoint parts    0.344       0.0082      42.1x
//!   ORDER BY k LIMIT 5, 1M rows                0.471       0.0087      57.9x
//!   ORDER BY k, no limit, 1M rows              2.160       0.574        3.8x
//! ```
//!
//! The `LIMIT` column is the point: with the sort gone the limit reaches the
//! scan, so the query reads one granule instead of the table, and the figure is
//! flat in the table size (0.0085 ms at 300k, 0.0087 ms at 1M) where the sort
//! it replaces is not. Without a `LIMIT` the win is smaller and comes from
//! somewhere else -- the read streams instead of materializing the relation and
//! sorting it.
//!
//! ### The negative case this rule really has, and the gate for it
//!
//! Dropping the sort also drops the `Exchange` that was wrapped round it, so
//! the scan underneath goes from 14 workers to one. That is repaid many times
//! over when the ordered read touches fewer rows -- but a predicate the zone
//! maps cannot decide prunes nothing, so the read walks the whole table either
//! way and the only thing that changed is that it lost 13 cores:
//!
//! ```text
//!   WHERE v = 42 ORDER BY k LIMIT 5   (300k)  0.156 ms -> 0.354 ms  0.44x
//!   WHERE v = 42 ORDER BY k           (300k)  0.162    -> 0.352     0.46x
//!   WHERE v = 42 ORDER BY k LIMIT 5   (1M)    0.341    -> 1.161     0.29x
//! ```
//!
//! The loss grows with the table because it is the parallel scan that was
//! carrying the query: `v = 42` matches 300 of 300k rows, so the sort being
//! replaced was sorting 300 rows and cost nothing to begin with. So the PREWHERE
//! list has to be one the *sort column* decides, or not exist -- the same gate
//! `meta_path` has, for the same reason, and with it every shape above is
//! between 0.97x and 1.07x. The one thing given up is a wide off-key filter with
//! no limit (`WHERE v < 900 ORDER BY k`, 1.29x, where the sort really was the
//! cost); 1.29x on a machine that swings 30% is not worth the 0.29x next to it.
//!
//! ### What this is invisible in
//!
//! `EXPLAIN PIPELINE` shows the sort *absent*, which is the truth but not a
//! statement. Every other decision in this file names itself in the plan text
//! (`IndexLookup`, `MetaAggregate`, `Exchange n workers`) and this one cannot,
//! because saying it needs a pass-through `PhysicalPlan` variant and every
//! variant needs an arm in `exec::operators::build_physical`, which is not this
//! task's file. The tests in `tests/plan_access_paths.rs` assert the absence
//! instead.
//!
//! ## Where the cost model is, and why it is not here
//!
//! This file used to say "not a cost model, and a real one is later, separate
//! work". The later work happened: statistics and cardinality estimation live
//! in [`crate::planner::optimizer::stats`], and the plan search that uses them
//! -- join reordering -- is a pass in `optimizer.rs`.
//!
//! It is one level up rather than in here, and the **Borrowing** section below
//! is the reason. Reordering a join tree produces intermediate schemas and
//! `ON` column pairs that exist nowhere in the input plan, and a
//! `PhysicalPlan<'a>` can only point at things the `&'a LogicalPlan` already
//! contains. So a search that changes the *shape* of a plan has to run while
//! the plan is still owned, which is the logical optimizer; what is left here
//! is the choice of access path for a shape already settled, and none of those
//! choices need a cardinality estimate:
//!
//!   * [`index_path`] fires on a predicate the key index can answer exactly.
//!     Rows do not enter into it -- one probe beats any scan;
//!   * [`meta_path`] answers from headers or does not answer at all;
//!   * `read_in_order` removes a sort that was reproducing its input's order;
//!   * [`exchange::degree`] is the one decision here that reads a row count,
//!     and it reads the exact one out of part metadata rather than an estimate.
//!
//! The measured constants that look like costs --
//! [`SCAN_ROWS_PER_PROBE`] here, `JOIN_ROWS_PER_PROBE` in
//! `exec::operators::join` -- stay measured constants. `stats.rs` says why it
//! does not price the strategy the second one guards.
//!
//! ## Borrowing
//!
//! A `PhysicalPlan<'a>` borrows every expression, schema and scan node from the
//! `&'a LogicalPlan` it was lowered from, and owns only what the decision
//! itself produced (the key lanes). That is what lets `build` keep taking a
//! `&LogicalPlan`: the plan can be a temporary inside `build` because the
//! operators never borrow *from it*, only from `'a`. It is also why lowering is
//! cheap enough to run unconditionally -- it allocates one `Box` per plan node
//! and nothing per row.

use crate::catalog::Catalog;
use crate::common::{lane_to_f64, lane_to_i64, Error, Result};
use crate::sql::ast::{BinaryOp, JoinOp};
use crate::types::{DataType, PhysicalType, Schema, Value};

use crate::exec::exchange;
use crate::exec::operators::window::WindowNode;

use super::logical::{BoundAgg, BoundExpr, CmpOp, LogicalPlan, ScanNode, SortKey, ZoneFilter};

/// How many rows of sequential scan one index probe is worth.
///
/// The gate on `pk IN (...)`: probing 100k keys one at a time to read a 200k-row
/// table is slower than reading it once, and an access-path decision that fires
/// when it should not is a performance bug in the other direction.
///
/// Measured on a 10M-row, 3-column table through the SQL front end, best-of-9:
/// an `IN` lookup costs 4.1 us at one key and 2.24 ms at 4096, i.e. a *marginal*
/// 0.33-0.55 us per additional probe once the fixed front-end cost is out of
/// the way. The scan it replaces runs at 2.6 ns/row for the cheapest possible
/// shape (`count(*)`, no columns decoded) and 17.7 ns/row for one that decodes
/// two columns and evaluates the `IN`.
///
/// 500 ns / 17.7 ns = 28 rows; 500 ns / 2.6 ns = 190 rows. 256 sits just past
/// the pessimistic end of that range on purpose. Being wrong toward "scan it"
/// costs a bounded constant factor on a query that was already going to be
/// slow; being wrong toward "probe it" costs an unbounded one, because the
/// probe count is set by the query text and the row count is not.
const SCAN_ROWS_PER_PROBE: usize = 256;

/// Same ceiling the optimizer enforces, for the same reason: every pass here
/// recurses once per plan level, and `exec::operators::build` recurses over the
/// result. A plan too deep for this is a plan that would overflow the stack a
/// moment later, and the optimizer's own [`MAX_PLAN_DEPTH`] doc already names
/// the physical planner as the reason it caps where it does.
///
/// [`MAX_PLAN_DEPTH`]: super::optimizer
const MAX_PLAN_DEPTH: usize = 200;

/// A primary-key access path: the rows to fetch, named by key rather than
/// found by walking.
///
/// The residual predicates are **not** stripped out of `node.filters`. The
/// index chooses *candidate rows*; the scan's own PREWHERE list then runs over
/// them unchanged, exactly as it would have on a sequentially scanned block.
/// That is the invariant that makes this safe to enable by default: the access
/// path is allowed to over-produce candidates and the filter corrects it, so
/// the only way to a wrong answer is to *miss* a row, which is a property of
/// `Part::find` alone and is pinned by the storage layer's own tests.
pub struct IndexPath<'a> {
    /// The scan this replaces. Projection, output schema and PREWHERE list all
    /// come from here, so everything above the operator is unchanged.
    pub node: &'a ScanNode,
    /// **Table** column index of the primary key.
    pub key_col: usize,
    /// Index of the key column in `node.projection`, i.e. its position in the
    /// projected schema. Only used to name the column in `EXPLAIN`.
    pub key_field: usize,
    /// Key lanes to probe, sorted and deduplicated.
    ///
    /// Sorted because a part is sorted by the same lane, so ascending keys
    /// resolve to ascending row positions and the operator gets scan order for
    /// free. Deduplicated because `IN (5, 5)` must return the row once, and the
    /// residual filter cannot undo a row fetched twice.
    pub keys: Vec<u64>,
}

/// An aggregate answered out of part metadata: no granule decoded, no row read.
///
/// This replaces the whole `Aggregate`-over-`Scan` subtree, not just the scan
/// under it -- the point is that there is no input stream at all. See the
/// module docs for what is refused and why.
pub struct MetaPath<'a> {
    /// The scan this replaces. The table to snapshot, the projection that maps
    /// a predicate column back to storage, and -- for the granules the zone
    /// maps cannot decide -- the PREWHERE list to fall back to.
    pub node: &'a ScanNode,
    /// The aggregate's output schema. One row of it is the entire result.
    pub schema: &'a Schema,
    /// The aggregates, for `EXPLAIN` and for the output column types.
    pub aggs: &'a [BoundAgg],
    /// What to compute, one per output column, parallel to `aggs`.
    pub what: Vec<MetaAgg>,
    /// One entry per conjunct of `node.filters`, or empty for an unfiltered
    /// aggregate. Equivalent to `node.filters` by construction -- see
    /// [`meta_preds`], and *not* the same thing as `node.zone_filters`.
    pub preds: Vec<MetaPred>,
    /// How many threads walk the granules. `1` is serial, and is always the
    /// answer for an unfiltered aggregate, which has no walk. See
    /// [`meta_degree`].
    pub workers: usize,
}

/// One output column of a [`MetaPath`].
pub enum MetaAgg {
    /// `count()` / `count(*)`: live rows.
    Count,
    /// `min(c)` / `max(c)` over **table** column `col`, from the zone maps.
    Extreme { col: usize, max: bool },
}

/// One `col <op> literal` conjunct, in both directions.
///
/// Both `col` fields are **table** column indices -- already mapped through
/// `ScanNode::projection`, unlike the projected-schema indices that
/// `ScanNode::zone_filters` carries.
pub struct MetaPred {
    /// The conjunct. `may_match` false means the granule holds no matching
    /// row, exactly as in the scan's own pruning.
    pub pred: ZoneFilter,
    /// Its negation. `may_match` false means no row *fails* the conjunct, so
    /// -- given the column has no NULL in this granule -- every row matches.
    pub not: ZoneFilter,
}

pub enum PhysicalPlan<'a> {
    /// Sequential scan with zone-map pruning and PREWHERE.
    Scan(&'a ScanNode),
    /// Primary-key point/batch lookup.
    IndexLookup(Box<IndexPath<'a>>),
    /// An aggregate folded out of part metadata instead of rows.
    MetaAggregate(Box<MetaPath<'a>>),
    Filter {
        input: Box<PhysicalPlan<'a>>,
        predicate: &'a BoundExpr,
    },
    Project {
        input: Box<PhysicalPlan<'a>>,
        exprs: &'a [BoundExpr],
        schema: &'a Schema,
    },
    Aggregate {
        input: Box<PhysicalPlan<'a>>,
        group: &'a [BoundExpr],
        aggs: &'a [BoundAgg],
        schema: &'a Schema,
    },
    /// Window functions. No decision to make: the sort a window needs is an
    /// ordinary `Sort` node the binder already placed underneath it.
    Window {
        input: Box<PhysicalPlan<'a>>,
        node: &'a WindowNode,
    },
    Sort {
        input: Box<PhysicalPlan<'a>>,
        keys: &'a [SortKey],
        /// `Some(k)`: keep only the `k` extreme rows (`Sort::top_k`). `None`:
        /// a full sort.
        fetch: Option<usize>,
    },
    Limit {
        input: Box<PhysicalPlan<'a>>,
        limit: Option<usize>,
        offset: usize,
    },
    LimitBy {
        input: Box<PhysicalPlan<'a>>,
        limit: usize,
        keys: &'a [BoundExpr],
    },
    Distinct {
        input: Box<PhysicalPlan<'a>>,
    },
    Join {
        left: Box<PhysicalPlan<'a>>,
        right: Box<PhysicalPlan<'a>>,
        op: JoinOp,
        on: &'a [(usize, usize)],
        residual: Option<&'a BoundExpr>,
        schema: &'a Schema,
    },
    Union {
        /// The lowered branches. Display and traversal only.
        branches: Vec<PhysicalPlan<'a>>,
        /// The same branches, unlowered.
        ///
        /// `union::build_union` takes `&[LogicalPlan]` and constructs a `Union`
        /// whose fields are private to that module, so the physical planner
        /// cannot hand it pre-built branch operators. It therefore lowers each
        /// branch a second time, inside `build`. That is one extra tree walk
        /// per `UNION` branch at plan time and nothing per row; the fix is a
        /// signature change in `union.rs`, which this task does not own.
        logical: &'a [LogicalPlan],
        all: bool,
        schema: &'a Schema,
    },
    Values {
        rows: &'a [Vec<Value>],
        schema: &'a Schema,
    },
    Empty {
        schema: &'a Schema,
    },
    /// `workers` copies of `input`, each over a disjoint contiguous slice of
    /// the one scan underneath it, with their partials merged in worker order.
    ///
    /// `workers` is always >= 2: [`exchange::degree`] answers 1 for "stay
    /// serial", and a one-wide fleet is a serial plan with a rendezvous bolted
    /// on. Emitting the node at all is therefore the statement that the query
    /// *will* run parallel, which is what makes `EXPLAIN PIPELINE` worth
    /// reading.
    Exchange {
        input: Box<PhysicalPlan<'a>>,
        workers: usize,
    },
}

/// Lower a logical plan, choosing an access path for every scan.
pub fn lower<'a>(plan: &'a LogicalPlan, catalog: &Catalog) -> Result<PhysicalPlan<'a>> {
    lower_at(plan, catalog, 0, true)
}

/// `par`: may a parallel node be emitted here?
///
/// False under a `Window` and inside a `UNION` branch, because both are built
/// by the serial builder and an `Exchange` there would print without running.
/// It is a parameter rather than a post-pass because lowering is bottom-up: by
/// the time the `Window` node exists its child has already been wrapped, and
/// unwrapping it again would need a by-value rewrite of every variant.
fn lower_at<'a>(
    plan: &'a LogicalPlan,
    catalog: &Catalog,
    depth: usize,
    par: bool,
) -> Result<PhysicalPlan<'a>> {
    if depth > MAX_PLAN_DEPTH {
        return Err(too_deep());
    }
    let d = depth + 1;
    let down = |p: &'a LogicalPlan| -> Result<Box<PhysicalPlan<'a>>> {
        Ok(Box::new(lower_at(p, catalog, d, par)?))
    };
    Ok(match plan {
        LogicalPlan::Scan(s) => match index_path(s, catalog) {
            Some(p) => PhysicalPlan::IndexLookup(Box::new(p)),
            None => PhysicalPlan::Scan(s),
        },
        LogicalPlan::Filter { input, predicate } => {
            PhysicalPlan::Filter { input: down(input)?, predicate }
        }
        LogicalPlan::Project { input, exprs, schema } => {
            PhysicalPlan::Project { input: down(input)?, exprs, schema }
        }
        LogicalPlan::Aggregate { input, group, aggs, schema } => {
            match meta_path(input, group, aggs, schema, catalog) {
                Some(m) => PhysicalPlan::MetaAggregate(Box::new(m)),
                None => fan_out(
                    PhysicalPlan::Aggregate { input: down(input)?, group, aggs, schema },
                    catalog,
                    par,
                ),
            }
        }
        LogicalPlan::Window { input, node } => {
            PhysicalPlan::Window { input: Box::new(lower_at(input, catalog, d, false)?), node }
        }
        // The rows are already in this order on disk, so the cheapest sort is
        // the one that does not run. See `read_in_order`.
        LogicalPlan::Sort { input, keys } if read_in_order(input, keys, catalog) => *down(input)?,
        LogicalPlan::Sort { input, keys } => fan_out(
            PhysicalPlan::Sort { input: down(input)?, keys, fetch: None },
            catalog,
            par,
        ),
        // `LIMIT 0` is a query that asks for no rows, so it is one that has to
        // produce none -- whatever is underneath. Cutting the subtree here
        // rather than building it and throwing every batch away is the same
        // trade `fuse_top_k` already makes one level down, including its one
        // visible consequence: an expression that would have overflowed on a
        // row nobody asked for no longer gets the chance. `LIMIT` is allowed
        // to spare you the rows you did not ask for.
        LogicalPlan::Limit { input, limit: Some(0), .. } => {
            // `&'a LogicalPlan` in, so the borrow is the caller's, not this
            // frame's -- which is why the schema can be handed to `Empty`.
            PhysicalPlan::Empty { schema: input.schema() }
        }
        LogicalPlan::Limit { input, limit, offset } => {
            let mut inner = lower_at(input, catalog, d, par)?;
            if let Some(l) = limit {
                fuse_top_k(&mut inner, l.saturating_add(*offset));
            }
            PhysicalPlan::Limit { input: Box::new(inner), limit: *limit, offset: *offset }
        }
        LogicalPlan::LimitBy { input, limit, keys } => {
            PhysicalPlan::LimitBy { input: down(input)?, limit: *limit, keys }
        }
        LogicalPlan::Distinct { input } => PhysicalPlan::Distinct { input: down(input)? },
        LogicalPlan::Join { left, right, op, on, residual, schema } => PhysicalPlan::Join {
            left: down(left)?,
            right: down(right)?,
            op: *op,
            on,
            residual: residual.as_ref(),
            schema,
        },
        LogicalPlan::Union { inputs, all, schema } => PhysicalPlan::Union {
            branches: inputs
                .iter()
                .map(|p| lower_at(p, catalog, d, false))
                .collect::<Result<_>>()?,
            logical: inputs,
            all: *all,
            schema,
        },
        LogicalPlan::Values { rows, schema } => PhysicalPlan::Values { rows, schema },
        LogicalPlan::Empty { schema } => PhysicalPlan::Empty { schema },
    })
}

#[cold]
fn too_deep() -> Error {
    Error::unsupported(format!(
        "plan nests more than {MAX_PLAN_DEPTH} levels deep; the physical planner and the \
         operator builder each recurse once per level and would run out of stack"
    ))
}

// -------------------------------------------------------------- 1. top-K

/// Tell the sort under a limit how few rows it actually has to keep.
///
/// `Project` is transparent to this -- the binder puts one between the limit
/// and the sort for every `SELECT a, b ... ORDER BY c LIMIT n`, and it neither
/// drops nor adds rows, so `k` means the same thing on both sides of it.
/// (Getting this wrong is silent: the fusion simply never fires and the only
/// symptom is the sort staying slow. The first version of this matched `Limit`
/// directly over `Sort`, measured 1.00x on every query, and was only found to
/// be dead code by `EXPLAIN`-ing the plan it claimed to match. Match on the
/// shape the binder actually produces.)
///
/// One visible consequence: an expression in that `Project` now evaluates over
/// `k` rows instead of all of them, so a `LIMIT`ed query whose discarded rows
/// would have overflowed a cast no longer fails. That is the same direction
/// ClickHouse takes -- `LIMIT` is allowed to spare you the rows you did not ask
/// for.
fn fuse_top_k(plan: &mut PhysicalPlan<'_>, k: usize) {
    match plan {
        PhysicalPlan::Sort { keys, fetch, .. }
            if crate::exec::operators::sort::Sort::worth_fusing(keys, k) =>
        {
            *fetch = Some(k);
        }
        // `Exchange` is transparent for the same reason `Project` is: it emits
        // exactly the rows the sort under it produced. Missing this arm would
        // silently un-fuse every top-K big enough to go parallel -- which is
        // every top-K worth fusing -- because `fan_out` wraps the sort before
        // the limit above it is lowered. Top-K is the shape the exchange wins
        // on outright (5.5x, see the exchange's own table); a full sort under
        // the same limit is 1.1x.
        PhysicalPlan::Project { input, .. } | PhysicalPlan::Exchange { input, .. } => {
            fuse_top_k(input, k)
        }
        _ => {}
    }
}

// ------------------------------------------------------- 2. index selection

/// Recognize a scan the primary-key index can answer, or `None` to scan.
///
/// Every gate here is a refusal to guess. The path is taken only when the table
/// declares a single-column unique key, the predicate is an equality or `IN` on
/// exactly that column, every literal converts to the stored lane *without
/// loss*, and the number of probes is small enough to beat reading the table.
fn index_path<'a>(node: &'a ScanNode, catalog: &Catalog) -> Option<IndexPath<'a>> {
    // A missing table is not this pass's error to raise: `Scan::new` reports it
    // with the message the tests expect, so fall through silently.
    let table = catalog.table_by_path(&node.table).ok()?;
    // `pk_col` is the whole "is this table keyed" predicate, cached on the
    // table. `None` covers unsorted engines, composite keys, nullable and
    // string keys, and `ORDER BY` without a `PRIMARY KEY` declaration.
    let key_col = table.pk_col()?;
    // The pk has to be in the projection: `node.filters` index the *projected*
    // schema, so a predicate on the key implies the key was projected for it.
    let key_field = node.projection.iter().position(|&c| c == key_col)?;

    let ty = table.schema().ty(key_col);
    let keys = node.filters.iter().find_map(|f| key_set(f, key_field, ty))?;
    if keys.is_empty() {
        // `IN ()` after NULL-stripping. An empty probe list is a correct answer
        // (no row can satisfy it), but it is also exactly what a scan gives for
        // free, and lowering it would make `EXPLAIN` claim an index was used
        // for a query that never touches one.
        return None;
    }

    // Every part must carry the index this path is about to probe. A part built
    // when the table had no key would answer `find` with `None` for every key,
    // which is not a slow answer but a wrong one; a part with no sort column
    // would make the operator's run walk read lane 0 for every row.
    //
    // This is a second `snapshot()` -- the operator takes its own -- and they
    // are guaranteed to be the same set because lowering and building both run
    // under one `&Catalog` borrow, which excludes every mutation. Two
    // uncontended `RwLock::read`s cost ~40 ns against a 3.6 us query, so
    // threading the first snapshot through `IndexPath` would buy 1% and couple
    // the planner to a live storage handle. `AUTO_COMPACT_PARTS` caps the loop
    // at 16 parts.
    let snap = table.snapshot();
    if snap
        .parts()
        .iter()
        .any(|p| p.pk_col != Some(key_col) || p.sort_col != Some(key_col))
    {
        return None;
    }
    // One probe always beats any scan worth the name, so only a batch has to
    // justify itself.
    if keys.len() > 1 && keys.len().saturating_mul(SCAN_ROWS_PER_PROBE) > snap.live_rows() {
        return None;
    }

    Some(IndexPath { node, key_col, key_field, keys })
}

/// The key lanes a predicate pins down, or `None` if it pins none.
///
/// `col` is the key's index in the *projected* schema, which is the space
/// `ScanNode::filters` are written in.
fn key_set(e: &BoundExpr, col: usize, ty: &DataType) -> Option<Vec<u64>> {
    let mut keys = match e {
        BoundExpr::Binary { left, op: BinaryOp::Eq, right, .. } => {
            let v = match (left.as_column(), right.as_column()) {
                (Some(c), _) if c == col => right.as_literal()?,
                (_, Some(c)) if c == col => left.as_literal()?,
                _ => return None,
            };
            vec![exact_lane(v, ty)?]
        }
        // `id = 1 OR id = 2` is `id IN (1, 2)` written the other way, and it is
        // how people write two keys. It never reached the index because
        // `split_conjuncts` stops at an `OR`, so the whole disjunction arrived
        // here as one conjunct and matched neither arm above -- a full table
        // scan for a query that names its rows. A/B interleaved in one loop
        // with a temporary switch, best-of-15 per side, 1M rows:
        //
        // ```text
        //   WHERE id = 1 OR id = 2                    2.07 ms -> 6.2 us   333x
        //   WHERE id = 1 OR id = 2 OR id = 3 OR id = 4 3.24 ms -> 8.5 us   383x
        //   WHERE id IN (1, 2) OR id = 3              3.78 ms -> 6.9 us   550x
        //   WHERE (id = 1 OR id = 2) AND v >= 0       2.17 ms -> 8.3 us   261x
        //   count() WHERE id = 10 OR id = 900000       499 us -> 7.9 us    63x
        // ```
        //
        // and the shapes it must not touch, measured the same way, all 1.0x:
        // `id = 1 OR v = 2`, `id = 1 OR id > 2`, `id = 1 OR id = 5.5`,
        // `NOT (id = 1 OR id = 2)`, plain `id = 1`, `v = 1`.
        //
        // **Every** disjunct has to be a key equality. One that is not admits
        // rows no probe would find, and the index is only allowed to
        // over-produce candidates, never to miss one.
        BoundExpr::Binary { op: BinaryOp::Or, .. } => {
            let mut out = Vec::new();
            let mut ok = true;
            or_disjuncts(e, &mut |d| {
                if !ok {
                    return;
                }
                match key_set(d, col, ty) {
                    // A nested `IN` list is a disjunct like any other, so
                    // `id IN (1, 2) OR id = 3` folds into one probe list.
                    Some(k) => out.extend(k),
                    None => ok = false,
                }
            });
            if !ok {
                return None;
            }
            out
        }
        BoundExpr::InList { expr, list, negated: false } if expr.as_column() == Some(col) => {
            let mut out = Vec::with_capacity(list.len());
            for v in list {
                // A NULL in the list can never make `IN` true -- three-valued
                // logic answers NULL, not TRUE -- so the rows it would admit
                // are exactly none, and dropping it changes no answer. Every
                // *other* literal has to convert exactly or the whole path is
                // refused: see `exact_lane`.
                if v.is_null() {
                    continue;
                }
                out.push(exact_lane(v, ty)?);
            }
            out
        }
        _ => return None,
    };
    keys.sort_unstable();
    keys.dedup();
    Some(keys)
}

/// Visit the leaves of an `OR` tree, the mirror of
/// [`BoundExpr::split_conjuncts`].
///
/// A callback rather than a `Vec<&BoundExpr>`: the caller only folds, so there
/// is no reason to build a list of leaves first. An explicit stack rather than
/// recursion, because `k = 1 OR k = 2 OR ...` nests as deep as the query text is
/// long -- `split_conjuncts` recurses and gets away with it because the parser
/// caps depth, but one `Vec` at plan time is cheaper than depending on that.
fn or_disjuncts(e: &BoundExpr, f: &mut dyn FnMut(&BoundExpr)) {
    let mut stack = vec![e];
    while let Some(cur) = stack.pop() {
        match cur {
            BoundExpr::Binary { left, op: BinaryOp::Or, right, .. } => {
                stack.push(right);
                stack.push(left);
            }
            leaf => f(leaf),
        }
    }
}

/// The stored lane for a literal, but only when the conversion loses nothing.
///
/// `Value::to_lane` is lossy on purpose -- it is the *write* path, where the
/// binder has already coerced the value to the column's type. Reached from a
/// predicate it will happily turn `id = 5.5` on a `UInt64` key into a probe for
/// lane 5 and hand back the row for `id = 5`, which no scan would have
/// returned. Decoding the lane back and comparing under `Value`'s own `Eq` --
/// the same `Eq` the filter about to run over the row uses -- is what turns
/// "this literal names a key" from a hope into a proof.
///
/// A literal that fails the round trip means "no stored row can equal this", so
/// scanning would return nothing either. We still fall back to the scan rather
/// than short-circuiting to an empty result: the reasoning that gets from
/// "inexact" to "matches nothing" runs through `Value`'s cross-representation
/// ordering, and this is not a place to be clever for a query nobody writes.
/// `pub(crate)` for the index-nested-loop join, which turns each row of its
/// build side into a probe key and needs *this* conversion rather than a second
/// one: two spellings of "does this literal name a stored key" that disagreed
/// would be a wrong answer on one path and not the other.
pub(crate) fn exact_lane(v: &Value, ty: &DataType) -> Option<u64> {
    let phys = ty.base().physical();
    let lane = v.to_lane_phys(phys, ty).ok()?;
    let back = match phys {
        PhysicalType::U64 => Value::UInt(lane),
        // `to_lane_phys` already scaled the literal into a unit count, so the
        // decode has to put the scale back or the round trip faces
        // `Decimal(250, 2)` with `Int(250)` -- never equal, so `exact_lane`
        // answered None for *every* decimal key and a decimal PRIMARY KEY
        // silently never reached the MPH index. Correct, and a full scan per
        // point lookup.
        //
        // A/B interleaved through `Session::query`, best-of-15 over 20 keys,
        // `--release`, on a `Decimal64(2)` pk: 10.5us -> 5.8us at 100k rows
        // (1.8x), 30.9us -> 7.4us at 1M (4.2x), 86.1us -> 7.4us at 4M (11.6x).
        // The shape is the point -- the probe side is flat in table size and
        // the scan side is linear, which is the difference between using the
        // index and never using it.
        PhysicalType::I64 => {
            let i = lane_to_i64(lane);
            ty.decimal_scale().map_or(Value::Int(i), |s| Value::Decimal(i, s))
        }
        // Every NaN shares one lane and `Value` equates NaN with NaN, so the
        // round trip would accept `f = NaN` and probe for it. SQL says NaN
        // equals nothing; refuse rather than let the two disagree.
        PhysicalType::F64 => {
            let f = lane_to_f64(lane);
            if !f.is_finite() {
                return None;
            }
            Value::Float(f)
        }
        // Reachable only through a broken `TableDef`: `sort_col` excludes
        // string keys precisely because their lanes are per-granule dictionary
        // codes.
        PhysicalType::Str => return None,
    };
    (back == *v).then_some(lane)
}

// -------------------------------------------------------- 3. parallelism
//
// The two halves of the decision live on opposite sides of the layer on
// purpose. *Can this shape be sharded at all* is a property of the operator
// that would run it, so `exchange::shard_stats` answers it and returns the
// table statistics as its evidence. *How wide should it be* is a cost
// question, so `degree` is called from here -- the one place a real cost model
// would eventually replace it, and the only place that can put the answer in
// the plan text.

/// Wrap a blocking node in an [`Exchange`](PhysicalPlan::Exchange), or leave it
/// alone.
///
/// Not fallible: an unresolvable table, an out-of-range projection and a shape
/// no worker can replicate are all "stay serial", and the serial builder is
/// what owns the error message for the first two. Reporting them twice, in two
/// wordings, is how a planner grows a second source of truth.
fn fan_out<'a>(plan: PhysicalPlan<'a>, catalog: &Catalog, par: bool) -> PhysicalPlan<'a> {
    if !par {
        return plan;
    }
    let Some((rows, granules)) = exchange::shard_stats(&plan, catalog) else { return plan };
    match exchange::degree(rows, granules) {
        // 1 is `degree`'s refusal, and every caller has always treated it as
        // one. Emitting `Exchange { workers: 1 }` would put a rendezvous and N
        // pipeline builds in front of a query that measured *slower* with them.
        0 | 1 => plan,
        workers => PhysicalPlan::Exchange { input: Box::new(plan), workers },
    }
}

// ---------------------------------------------------- 4. metadata answers

/// Recognize an aggregate the part headers can answer, or `None` to run it.
///
/// Every gate is a refusal to approximate. The module docs list them and say
/// why each one is a wrong answer rather than a slow one; this is the code.
fn meta_path<'a>(
    input: &'a LogicalPlan,
    group: &'a [BoundExpr],
    aggs: &'a [BoundAgg],
    schema: &'a Schema,
    catalog: &Catalog,
) -> Option<MetaPath<'a>> {
    // A `GROUP BY` needs per-group counts, which no header records. An empty
    // aggregate list is `SELECT FROM t GROUP BY ()`-shaped and has no output
    // column to put an answer in.
    if !group.is_empty() || aggs.is_empty() {
        return None;
    }
    // Over the scan, not over a `Filter`, a `Limit` or a `Join`: the predicate
    // has to be the one the scan node carries, or the fallback the operator
    // applies to a straddling granule would not be the query's predicate --
    // and a `Limit` makes the count the limit's rather than the table's.
    // Column projections *are* looked through; see [`meta_source`].
    let (node, map) = meta_source(input)?;

    // A missing table is `Scan::new`'s error to raise, with the message the
    // tests expect -- so fall through silently, as `index_path` does.
    let table = catalog.table_by_path(&node.table).ok()?;
    let ncols = table.schema().len();
    if node.projection.iter().any(|&c| c >= ncols) {
        return None;
    }
    // Parts are the whole table only once the write buffer is empty, which
    // every read guarantees by flushing before it plans. See the module docs.
    if table.has_pending_writes() {
        return None;
    }

    let preds = meta_preds(&node.filters, &node.projection)?;
    // Only on the column the table is *sorted* by.
    //
    // Measured, and the reason this gate exists: `count() WHERE latency > 500`
    // over 2M rows, where `latency` is uncorrelated with the sort key, decides
    // exactly zero granules -- every one straddles, so the operator decodes
    // the whole table *serially* while the plan it replaced would have decoded
    // it across 14 workers. A/B interleaved, best-of-5 x 5 rounds:
    // **1.26 ms -> 2.92 ms, 0.43x**. Right answer, 2.3x slower.
    //
    // A zone map only decides a granule when the column is clustered, and the
    // sort column is the one the engine guarantees that for -- so this is the
    // gate that makes the filtered path monotone. With it, the same table
    // measures `ts >= <90% cut>` at 279 us -> 48 us (5.8x) and the undecidable
    // shapes stay on the parallel scan where they belong.
    //
    // Rejected alternative: sampling ~32 granules at plan time and taking the
    // path only if most of them decide. It adapts to correlated-but-unsorted
    // columns, which this gate gives up on -- but it costs a snapshot walk on
    // every aggregate, it needs a threshold nobody can defend, and it makes
    // the plan depend on data the `EXPLAIN` reader cannot see.
    if !preds.is_empty() {
        let sort_col = table.sort_col()?;
        if preds.iter().any(|p| p.pred.col != sort_col) {
            return None;
        }
    }
    // A predicate the primary-key index can answer beats this: the index reads
    // one granule after one MPH probe, while a metadata count still walks
    // every granule's zone map to find the one that straddles. On a 2M-row
    // table that is ~4 us against ~50 us for `count() WHERE id = k` -- right,
    // and 12x slower, which is a performance bug in the other direction. The
    // scan lowering below would have chosen the index; so does this.
    if index_path(node, catalog).is_some() {
        return None;
    }

    // One `RwLock::read` + `Arc` clone (~40 ns), reached only once the shape
    // has matched, and the same argument `index_path` makes applies: lowering
    // and building run under one `&Catalog` borrow, which excludes every
    // mutation, so the operator's own snapshot is this same set.
    //
    // Inside a transaction this is the transaction's overlay -- the same view
    // `Scan` would take -- so read-your-own-writes needs nothing here.
    let snap = table.snapshot();
    // `AUTO_COMPACT_PARTS` caps this at 16 iterations, and it is one null
    // check per part for every table that has never had a row deleted.
    let tombstoned = (0..snap.len()).any(|i| snap.deletes(i).is_some_and(|d| d.count() > 0));

    let mut what = Vec::with_capacity(aggs.len());
    for a in aggs {
        // `count(DISTINCT x)` and `min(x) FILTER`-style parameters are other
        // aggregates wearing the same name.
        if a.distinct || !a.params.is_empty() {
            return None;
        }
        what.push(match a.func.name {
            // `count()` and `count(*)` both bind to an empty argument list;
            // `count(x)` counts non-NULLs and does not.
            "count" if a.args.is_empty() => MetaAgg::Count,
            // A tombstone widens no granule's bounds but can remove the row
            // that set one, so with any delete anywhere the fold is only an
            // envelope. Refuse the whole path rather than answer 8 for a
            // column whose largest live value is 7.
            "min" | "max" if !tombstoned && preds.is_empty() => {
                // The argument has to be a bare column of the scan, and a
                // predicate would mean the extreme is over a subset the
                // bounds do not describe.
                let field = *map.get(a.args.first()?.as_column()?)?;
                MetaAgg::Extreme { col: *node.projection.get(field)?, max: a.func.name == "max" }
            }
            _ => return None,
        });
    }
    let workers = if preds.is_empty() {
        1
    } else {
        meta_degree((0..snap.len()).map(|i| snap.part(i).granule_count()).sum())
    };
    Some(MetaPath { node, schema, aggs, what, preds, workers })
}

/// The scan under an aggregate, seen through the projections a derived table
/// leaves behind, plus the map from the aggregate's input columns back to the
/// scan's projected ones.
///
/// `SELECT count() FROM (SELECT k FROM t) u` binds to `Aggregate` over
/// `Project` over `Scan`: the subquery is inlined, but its column list stays as
/// a projection. That projection neither adds nor drops a row, so the part
/// headers answer the count exactly as they do one level down -- and without
/// this the commonest way to write a subquery loses the fold entirely.
/// Measured through the SQL front end, A/B interleaved in one loop with a
/// temporary switch, best-of-25 per side, 1M rows / 14 cores:
///
/// ```text
///   count() FROM (SELECT k FROM t) u      1.048 ms -> 6.9 us   152x
///   min(k)  FROM (SELECT k FROM t) u      1.310 ms -> 9.6 us   137x
/// ```
///
/// **Bare column references only.** A computed projection cannot change the row
/// count either, but it can *fail* on a row the fold would never look at, and
/// turning a query that raises into one that answers is not a change this
/// decision gets to make on its own. `Limit` and `Filter` are refused for the
/// reasons the caller's own comment gives.
fn meta_source(plan: &LogicalPlan) -> Option<(&ScanNode, Vec<usize>)> {
    // One `Vec` per aggregate at plan time, and nothing per row. `map[c]` is
    // where output column `c` lives in the plan currently in hand, so a
    // projection composes into it by one indirection per column.
    let mut map: Vec<usize> = (0..plan.schema().len()).collect();
    let mut plan = plan;
    loop {
        match plan {
            LogicalPlan::Scan(s) => return Some((s, map)),
            LogicalPlan::Project { input, exprs, .. } => {
                for c in map.iter_mut() {
                    *c = exprs.get(*c)?.as_column()?;
                }
                plan = input;
            }
            _ => return None,
        }
    }
}

/// Threads for a filtered fold's granule walk.
///
/// The walk it parallelizes is the *same* walk the scan does, and the scan's
/// runs inside the exchange's fleet -- so a serial fold was 14x behind on
/// exactly the queries where pruning is the whole cost. `count() WHERE ts
/// BETWEEN <1000 rows>` over 2M rows measured **62 us scanned vs 97 us folded
/// (0.6x)** with the walk serial: the right answer, from metadata, slower than
/// reading the two granules in parallel. With the fleet the same query is
/// 54 us vs 46 us (1.2x), and every other filtered shape gained too -- the
/// 50% cut went 20.6x -> 32x and the 1% cut 2.0x -> 3.1x. That is the whole
/// justification for `workers` existing.
///
/// The floor is per *worker*, not per query: a zone test is ~50 ns, so 512
/// granules is ~25 us of work, comfortably more than waking the pool costs.
/// Below that -- half a million rows -- the walk is over before a fleet
/// finishes assembling, and `1` is a refusal every caller honours.
fn meta_degree(granules: usize) -> usize {
    crate::common::pool::global().threads().min(granules / 512).max(1)
}

/// The scan's PREWHERE list as zone tests, or `None` if any conjunct is not
/// one.
///
/// All-or-nothing on purpose. A conjunct left out would be a predicate the
/// covering test never checks, so a granule could be called covered while some
/// of its rows fail -- which over-counts. `ScanNode::zone_filters` is not used
/// for this precisely because it is *allowed* to be lossy: `x IN (1, 99)`
/// becomes `x >= 1 AND x <= 99` there, which prunes correctly and covers
/// wrongly.
fn meta_preds(filters: &[BoundExpr], projection: &[usize]) -> Option<Vec<MetaPred>> {
    let mut out = Vec::with_capacity(filters.len());
    for f in filters {
        let BoundExpr::Binary { left, op, right, .. } = f else { return None };
        let cmp = CmpOp::from_binary(*op)?;
        let (field, op, value) = match (left.as_column(), right.as_column()) {
            (Some(c), _) => (c, cmp, right.as_literal()?),
            // `5 < x` is `x > 5`, the same flip the optimizer's own
            // extraction makes.
            (_, Some(c)) => (c, cmp.flip(), left.as_literal()?),
            _ => return None,
        };
        // A comparison against NULL is NULL, never TRUE, so no row matches and
        // no row fails: both directions of the zone test would be misleading.
        // Rare enough not to be worth reasoning about; scan it.
        if value.is_null() {
            return None;
        }
        let col = *projection.get(field)?;
        out.push(MetaPred {
            pred: ZoneFilter { col, op, value: value.clone() },
            not: ZoneFilter { col, op: negate(op), value: value.clone() },
        });
    }
    Some(out)
}

/// The comparison that is true exactly where `op` is false, for non-NULL
/// operands.
///
/// Lives here rather than on `CmpOp` because it is only meaningful under that
/// caveat, and `logical.rs` is not this task's file.
fn negate(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Eq => CmpOp::NotEq,
        CmpOp::NotEq => CmpOp::Eq,
        CmpOp::Lt => CmpOp::GtEq,
        CmpOp::LtEq => CmpOp::Gt,
        CmpOp::Gt => CmpOp::LtEq,
        CmpOp::GtEq => CmpOp::Lt,
    }
}

// ------------------------------------------------------- 5. sorted reads

/// Does the stored order already answer this `ORDER BY`?
///
/// A `MergeTree` part *is* a sorted run, and `Scan` walks parts in set order
/// and granules in part order, so for the right `ORDER BY` the sort operator is
/// being asked to reproduce the order its own input arrived in.
///
/// Exactly **one** ascending key, and it has to be [`Table::sort_col`].
///
/// * *One* key, not a prefix of the sort key. A part is built sorted by the
///   whole `order_by` tuple, but `Table::merge_parts` merges runs on the
///   leading sort lane alone and breaks ties by part index -- so a *merged*
///   part is sorted by `order_by[0]` and by nothing after it, and compaction
///   runs off `flush`, which runs off every read. A prefix rule would be right
///   until the first merge and silently wrong after it.
/// * *Ascending*, because storage reads one way. A `DESC` read needs the scan
///   to walk granules and rows backwards; that is a storage change, and the
///   measured 0.5 ms this shape costs today is waiting for it.
/// * `sort_col`, which already excludes an unsorted engine, a nullable key and
///   a string key -- and being non-nullable is what makes `NULLS FIRST/LAST`
///   vacuous here rather than a fourth condition to get wrong.
///
/// [`Table::sort_col`]: crate::storage::Table::sort_col
fn read_in_order(input: &LogicalPlan, keys: &[SortKey], catalog: &Catalog) -> bool {
    let [key] = keys else { return false };
    if !key.asc {
        return false;
    }
    let Some(field) = key.expr.as_column() else { return false };
    let Some((node, col)) = ordered_source(input, field) else { return false };
    let Ok(table) = catalog.table_by_path(&node.table) else { return false };
    if table.sort_col() != Some(col) {
        return false;
    }
    if table.has_pending_writes() {
        return false;
    }
    // The PREWHERE list has to be one the *sort column* decides, or not exist.
    //
    // This is the gate, and it is the same one `meta_path` has for the same
    // reason. Dropping the sort also drops the `Exchange` that was wrapped
    // round it, so the scan underneath goes from 14 workers to one. That is
    // paid back many times over when the ordered read touches fewer rows --
    // but a predicate the zone maps cannot decide prunes nothing, so the read
    // walks the whole table either way and the only thing that changed is that
    // it lost 13 cores. Measured, A/B interleaved in one loop with a temporary
    // switch, best-of-25 per side, 300k rows / 14 cores:
    //
    // ```text
    //   ORDER BY k LIMIT 5                      0.378 -> 0.009   42.8x
    //   ORDER BY k                              0.633 -> 0.155    4.1x
    //   WHERE k >= <99% cut> ORDER BY k LIMIT 5 0.072 -> 0.010    7.6x
    //   WHERE v = 42 ORDER BY k LIMIT 5         0.156 -> 0.354    0.44x
    //   WHERE v = 42 ORDER BY k                 0.162 -> 0.352    0.46x
    // ```
    //
    // and at 1M rows the same off-key shape is **0.34 -> 1.16 ms, 0.29x**: the
    // loss grows with the table because it is the parallel scan that was
    // carrying it. `v = 42` matches 300 of 300k rows, so the sort it replaces
    // was sorting 300 rows and cost nothing to begin with.
    //
    // The one shape this gate gives up is a *wide* off-key filter with no
    // limit -- `WHERE v < 900 ORDER BY k` measured 1.29x, because there the
    // sort really was the cost. 1.29x on a machine that swings 30% is not
    // worth the 0.29x next to it.
    match meta_preds(&node.filters, &node.projection) {
        Some(ps) if ps.iter().all(|p| p.pred.col == col) => {}
        _ => return false,
    }
    parts_concatenate_in_order(&table.snapshot(), col)
}

/// The scan a column of `plan` comes from, and its **table** column index --
/// provided every operator in between emits its rows in the order they arrived.
///
/// `Filter`, `Project` and `Limit` all do: they drop rows, rename them or cut
/// the stream short, and none of them reorders. Everything else either shuffles
/// (`Aggregate`, `Join`, `Distinct`, `Union`) or has no single scan under it,
/// and falls through to `None`.
///
/// Iterative rather than recursive because it runs *before* `lower_at`'s depth
/// guard has walked the subtree it is about to descend.
fn ordered_source(mut plan: &LogicalPlan, mut col: usize) -> Option<(&ScanNode, usize)> {
    loop {
        match plan {
            LogicalPlan::Scan(s) => return Some((s, *s.projection.get(col)?)),
            LogicalPlan::Filter { input, .. } | LogicalPlan::Limit { input, .. } => plan = input,
            LogicalPlan::Project { input, exprs, .. } => {
                col = exprs.get(col)?.as_column()?;
                plan = input;
            }
            _ => return None,
        }
    }
}

/// Do the parts, read back to back in set order, come out non-decreasing on
/// lane `col`?
///
/// Each part is internally sorted, but the set is not: `PartSet::push` appends,
/// and a merge appends its result, so part 0 can hold keys above part 1's. Two
/// lane reads per part, capped at `AUTO_COMPACT_PARTS` parts, and the bounds
/// are already cached on the granule for range pruning.
///
/// The bounds are *birth* bounds -- they describe the rows the part was built
/// from, including any since deleted. That is the safe direction: a tombstone
/// can only narrow the live range, so a set that concatenates in order by birth
/// bounds still does with rows missing.
fn parts_concatenate_in_order(snap: &crate::storage::part::Snapshot, col: usize) -> bool {
    let mut prev: Option<u64> = None;
    for p in snap.parts() {
        // An empty part contributes no row, so it constrains nothing.
        let (Some(lo), Some(hi)) = (p.granules.first(), p.granules.last()) else { continue };
        // `sort_min`/`sort_max` are lanes of *this part's* sort column, so they
        // mean nothing if that is not the column being asked about -- and a
        // part built without one carries zeroes.
        if p.sort_col != Some(col) {
            return false;
        }
        if prev.is_some_and(|end| lo.sort_min < end) {
            return false;
        }
        prev = Some(hi.sort_max);
    }
    true
}

// ------------------------------------------------------------------ EXPLAIN

impl PhysicalPlan<'_> {
    pub fn schema(&self) -> &Schema {
        match self {
            PhysicalPlan::Scan(s) => &s.schema,
            PhysicalPlan::IndexLookup(i) => &i.node.schema,
            PhysicalPlan::MetaAggregate(m) => m.schema,
            PhysicalPlan::Filter { input, .. }
            | PhysicalPlan::Sort { input, .. }
            | PhysicalPlan::Limit { input, .. }
            | PhysicalPlan::LimitBy { input, .. }
            | PhysicalPlan::Exchange { input, .. }
            | PhysicalPlan::Distinct { input } => input.schema(),
            PhysicalPlan::Window { node, .. } => &node.schema,
            PhysicalPlan::Project { schema, .. }
            | PhysicalPlan::Aggregate { schema, .. }
            | PhysicalPlan::Join { schema, .. }
            | PhysicalPlan::Union { schema, .. }
            | PhysicalPlan::Values { schema, .. }
            | PhysicalPlan::Empty { schema } => schema,
        }
    }

    pub fn children(&self) -> Vec<&PhysicalPlan<'_>> {
        match self {
            PhysicalPlan::Scan(_)
            | PhysicalPlan::IndexLookup(_)
            | PhysicalPlan::MetaAggregate(_)
            | PhysicalPlan::Values { .. }
            | PhysicalPlan::Empty { .. } => Vec::new(),
            PhysicalPlan::Filter { input, .. }
            | PhysicalPlan::Project { input, .. }
            | PhysicalPlan::Aggregate { input, .. }
            | PhysicalPlan::Sort { input, .. }
            | PhysicalPlan::Limit { input, .. }
            | PhysicalPlan::LimitBy { input, .. }
            | PhysicalPlan::Window { input, .. }
            | PhysicalPlan::Exchange { input, .. }
            | PhysicalPlan::Distinct { input } => vec![input],
            PhysicalPlan::Join { left, right, .. } => vec![left, right],
            PhysicalPlan::Union { branches, .. } => branches.iter().collect(),
        }
    }

    /// The access path and its parameters, one line, no children.
    ///
    /// `EXPLAIN` is the only way anyone outside this file can tell whether the
    /// index fired or the exchange did, so the label states the decision and
    /// the numbers behind it rather than merely naming the operator.
    ///
    /// `pub(crate)` for `exchange::explain_analyze`, which prints these same
    /// lines with measurements appended. One renderer, so the annotated tree
    /// and the plain one can never disagree about what the plan says.
    pub(crate) fn label(&self) -> String {
        match self {
            PhysicalPlan::Scan(s) => {
                let mut out = format!("Scan {} [{}]", s.table, col_names(&s.schema));
                if !s.filters.is_empty() {
                    let fs: Vec<String> = s.filters.iter().map(|f| f.to_string()).collect();
                    out.push_str(&format!(" prewhere={}", fs.join(" AND ")));
                }
                if !s.zone_filters.is_empty() {
                    out.push_str(&format!(" zonemap={}", s.zone_filters.len()));
                }
                out
            }
            PhysicalPlan::IndexLookup(i) => {
                let s = &i.node;
                let key = s.schema.fields()[i.key_field].name.as_str();
                let mut out = format!(
                    "IndexLookup {} [{}] on {key} ({} key{})",
                    s.table,
                    col_names(&s.schema),
                    i.keys.len(),
                    if i.keys.len() == 1 { "" } else { "s" }
                );
                if !s.filters.is_empty() {
                    let fs: Vec<String> = s.filters.iter().map(|f| f.to_string()).collect();
                    out.push_str(&format!(" residual={}", fs.join(" AND ")));
                }
                out
            }
            // The line has to say *from metadata*, or nobody can tell whether
            // the decision fired: a query that reads no rows and one that
            // reads two million produce the same answer and, without this,
            // the same plan text. It is also how the tests assert it.
            PhysicalPlan::MetaAggregate(m) => {
                let a: Vec<String> = m.aggs.iter().map(agg_call).collect();
                let mut out =
                    format!("MetaAggregate {} [{}] from part metadata", m.node.table, a.join(", "));
                if !m.node.filters.is_empty() {
                    let fs: Vec<String> = m.node.filters.iter().map(|f| f.to_string()).collect();
                    // Named for what it costs: the granules the zone maps
                    // cannot decide either way are still decoded.
                    out.push_str(&format!(
                        " where={} (straddling granules read)",
                        fs.join(" AND ")
                    ));
                }
                // The width belongs in the plan text for the reason the
                // `Exchange` line exists: a fold that quietly stopped fanning
                // out would show up nowhere else.
                if m.workers > 1 {
                    out.push_str(&format!(" {} workers", m.workers));
                }
                out
            }
            PhysicalPlan::Filter { predicate, .. } => format!("Filter {predicate}"),
            PhysicalPlan::Project { exprs, schema, .. } => {
                let items: Vec<String> = exprs
                    .iter()
                    .zip(schema.fields())
                    .map(|(e, f)| format!("{e} AS {}", f.name))
                    .collect();
                format!("Project [{}]", items.join(", "))
            }
            PhysicalPlan::Aggregate { group, aggs, .. } => {
                let g: Vec<String> = group.iter().map(|e| e.to_string()).collect();
                let a: Vec<String> = aggs.iter().map(agg_call).collect();
                format!("Aggregate group=[{}] aggs=[{}]", g.join(", "), a.join(", "))
            }
            PhysicalPlan::Sort { keys, fetch, .. } => {
                let k: Vec<String> = keys
                    .iter()
                    .map(|k| format!("{}{}", k.expr, if k.asc { "" } else { " DESC" }))
                    .collect();
                match fetch {
                    Some(n) => format!("TopK {n} [{}]", k.join(", ")),
                    None => format!("Sort [{}]", k.join(", ")),
                }
            }
            PhysicalPlan::Window { node, .. } => node.label(),
            PhysicalPlan::Limit { limit, offset, .. } => match limit {
                Some(l) => format!("Limit {l} offset {offset}"),
                None => format!("Offset {offset}"),
            },
            PhysicalPlan::LimitBy { limit, keys, .. } => {
                let k: Vec<String> = keys.iter().map(|e| e.to_string()).collect();
                format!("LimitBy {limit} by [{}]", k.join(", "))
            }
            PhysicalPlan::Distinct { .. } => "Distinct".into(),
            PhysicalPlan::Join { op, on, residual, .. } => {
                let pairs: Vec<String> =
                    on.iter().map(|(l, r)| format!("l#{l} = r#{r}")).collect();
                let mut s = format!("{op:?}HashJoin on [{}]", pairs.join(", "));
                if let Some(r) = residual {
                    s.push_str(&format!(" residual={r}"));
                }
                s
            }
            PhysicalPlan::Union { all, .. } => {
                format!("Union{}", if *all { " All" } else { " Distinct" })
            }
            PhysicalPlan::Values { rows, .. } => format!("Values {} rows", rows.len()),
            PhysicalPlan::Empty { .. } => "Empty".into(),
            // The degree is the number a benchmark and a bug report both need,
            // so it is in the line rather than implied by the node's presence.
            PhysicalPlan::Exchange { workers, .. } => format!("Exchange {workers} workers"),
        }
    }

    /// Indented tree, for `EXPLAIN PIPELINE`.
    pub fn explain(&self) -> String {
        let mut out = String::new();
        self.explain_into(0, &mut out);
        out
    }

    fn explain_into(&self, depth: usize, out: &mut String) {
        for _ in 0..depth {
            out.push_str("  ");
        }
        out.push_str(&self.label());
        out.push('\n');
        for c in self.children() {
            c.explain_into(depth + 1, out);
        }
    }
}

/// `name(arg, arg)`, the way both aggregate-bearing labels render a call.
fn agg_call(a: &BoundAgg) -> String {
    let args: Vec<String> = a.args.iter().map(|x| x.to_string()).collect();
    format!("{}({})", a.func.name, args.join(", "))
}

fn col_names(s: &Schema) -> String {
    let cols: Vec<&str> = s.fields().iter().map(|f| f.name.as_str()).collect();
    cols.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::binder::Binder;
    use crate::planner::optimizer;
    use crate::types::{Block, Column, Engine, Field, TableDef};
    use crate::Session;

    /// `t(id <ty>, v Int64)` with `n` rows, keyed unless `keyed` is false.
    fn session(decl: &str, n: u64) -> Session {
        let mut s = Session::in_memory();
        s.execute(&format!("CREATE TABLE t (id {decl}, v Int64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id"))
            .unwrap();
        let t = s.catalog.table_by_path_mut("default.t").unwrap();
        let ty = t.schema().ty(0).clone();
        let ids = match ty.base().physical() {
            PhysicalType::I64 => Column::i64s(ty.clone(), (0..n as i64).collect()),
            PhysicalType::F64 => {
                Column::new(ty.clone(), crate::types::ColumnData::F64((0..n).map(|i| i as f64 / 2.0).collect()))
            }
            _ => Column::u64s(ty.clone(), (0..n).collect()),
        };
        t.insert(
            Block::new(vec![ids, Column::i64s(DataType::Int64, (0..n as i64).map(|i| i * 3).collect())])
                .unwrap(),
        )
        .unwrap();
        t.flush().unwrap();
        s
    }

    /// The physical `EXPLAIN` text for a query, planned exactly as the session
    /// would plan it.
    fn phys(s: &mut Session, sql: &str) -> String {
        let stmts = crate::sql::parser::parse(sql).unwrap();
        let q = match &stmts[0] {
            crate::sql::ast::Statement::Query(q) => q.clone(),
            other => panic!("not a query: {other:?}"),
        };
        s.catalog.flush_all().unwrap();
        let plan = optimizer::optimize(Binder::new(&s.catalog).bind_query(&q).unwrap()).unwrap();
        // Through the same entry point `EXPLAIN PIPELINE` should call, so the
        // rendering these tests assert on is the one a user would see.
        crate::planner::explain_physical(&plan, &s.catalog).unwrap()
    }

    #[test]
    fn equality_on_the_primary_key_lowers_to_an_index_lookup() {
        let mut s = session("UInt64", 20_000);
        let e = phys(&mut s, "SELECT v FROM t WHERE id = 31337");
        assert!(e.contains("IndexLookup default.t"), "{e}");
        assert!(e.contains("on id (1 key)"), "{e}");
        // The predicate stays as a residual: the index picks candidates, the
        // filter still decides.
        assert!(e.contains("residual="), "the prewhere must survive:\n{e}");
        // ... and the literal on the left is the same shape.
        assert!(phys(&mut s, "SELECT v FROM t WHERE 31337 = id").contains("IndexLookup"));
    }

    #[test]
    fn an_in_list_lowers_to_a_batched_index_lookup() {
        let mut s = session("UInt64", 20_000);
        let e = phys(&mut s, "SELECT v FROM t WHERE id IN (5, 9, 5, 12)");
        assert!(e.contains("IndexLookup"), "{e}");
        assert!(e.contains("(3 keys)"), "duplicates must be collapsed: {e}");
    }

    #[test]
    fn an_index_lookup_survives_extra_conjuncts() {
        let mut s = session("UInt64", 20_000);
        let e = phys(&mut s, "SELECT v FROM t WHERE id = 7 AND v > 3");
        assert!(e.contains("IndexLookup"), "{e}");
        assert!(e.contains("residual="), "{e}");
        assert!(e.contains("v#1"), "the other conjunct must still run: {e}");
    }

    #[test]
    fn the_shapes_that_must_still_scan() {
        let mut s = session("UInt64", 20_000);
        for q in [
            // a range, not a point
            "SELECT v FROM t WHERE id > 7",
            "SELECT v FROM t WHERE id >= 7 AND id <= 9",
            // inequality
            "SELECT v FROM t WHERE id != 7",
            // a non-key column
            "SELECT v FROM t WHERE v = 21",
            // negated IN
            "SELECT v FROM t WHERE id NOT IN (1, 2)",
            // no predicate at all
            "SELECT v FROM t",
            // key compared to another column, not a constant
            "SELECT v FROM t WHERE id = v",
            // one disjunct that is not a key equality admits rows no probe
            // would find, so the whole disjunction has to be scanned
            "SELECT v FROM t WHERE id = 1 OR v = 2",
            "SELECT v FROM t WHERE id = 1 OR id > 2",
            "SELECT v FROM t WHERE id = 1 OR (id = 2 AND v = 3)",
            "SELECT v FROM t WHERE NOT (id = 1 OR id = 2)",
        ] {
            let e = phys(&mut s, q);
            assert!(!e.contains("IndexLookup"), "{q} should scan:\n{e}");
            assert!(e.contains("Scan default.t"), "{q}:\n{e}");
        }
    }

    #[test]
    fn an_or_of_key_equalities_is_an_in_list_written_differently() {
        let mut s = session("UInt64", 20_000);
        let e = phys(&mut s, "SELECT v FROM t WHERE id = 1 OR id = 2");
        assert!(e.contains("IndexLookup"), "{e}");
        assert!(e.contains("(2 keys)"), "{e}");
        // Nested, flipped, repeated and mixed with an `IN` -- all one probe
        // list, deduplicated.
        let e = phys(&mut s, "SELECT v FROM t WHERE id IN (1, 2) OR 3 = id OR id = 1");
        assert!(e.contains("(3 keys)"), "{e}");
        // The disjunction is still the residual, so an over-eager probe list
        // could not change the answer even if one got through.
        assert!(e.contains("residual="), "{e}");
        assert_eq!(
            s.query("SELECT v FROM t WHERE id = 1 OR id = 2").unwrap().to_values(),
            [[Value::Int(3)], [Value::Int(6)]]
        );
        // One key repeated is one probe, not two rows.
        assert_eq!(s.query("SELECT v FROM t WHERE id = 5 OR id = 5").unwrap().rows(), 1);
        // A disjunct whose literal does not name a stored key takes the whole
        // path away, rather than probing for the ones that do and losing the
        // rows the scan would have found for the one that does not.
        let e = phys(&mut s, "SELECT v FROM t WHERE id = 1 OR id = 5.5");
        assert!(!e.contains("IndexLookup"), "{e}");
    }

    #[test]
    fn or_disjuncts_flattens_either_nesting() {
        let lit = |i: i64| BoundExpr::lit(Value::Int(i));
        let or = |l: BoundExpr, r: BoundExpr| BoundExpr::Binary {
            left: Box::new(l),
            op: BinaryOp::Or,
            right: Box::new(r),
            ty: DataType::Bool,
        };
        // Left-nested is what the parser builds; right-nested is what a
        // rewrite could leave behind. Both are three leaves, in order.
        for e in [or(or(lit(1), lit(2)), lit(3)), or(lit(1), or(lit(2), lit(3)))] {
            let mut seen = Vec::new();
            or_disjuncts(&e, &mut |d| seen.push(d.to_string()));
            assert_eq!(seen, ["1", "2", "3"]);
        }
        // A non-`OR` is one disjunct, which is what makes the fold in
        // `key_set` uniform.
        let mut seen = 0;
        or_disjuncts(&lit(7), &mut |_| seen += 1);
        assert_eq!(seen, 1);
    }

    #[test]
    fn a_table_without_a_declared_key_always_scans() {
        // `ORDER BY` is a sort key, not a uniqueness declaration, so this table
        // has no pk and duplicates are legal in it.
        let mut s = Session::in_memory();
        s.execute("CREATE TABLE u (id Int64, k Int64) ENGINE = MergeTree ORDER BY (id, k)")
            .unwrap();
        s.execute("INSERT INTO u VALUES (1, 1), (1, 2)").unwrap();
        let stmts = crate::sql::parser::parse("SELECT k FROM u WHERE id = 1").unwrap();
        let q = match &stmts[0] {
            crate::sql::ast::Statement::Query(q) => q.clone(),
            _ => unreachable!(),
        };
        s.catalog.flush_all().unwrap();
        let plan = optimizer::optimize(Binder::new(&s.catalog).bind_query(&q).unwrap()).unwrap();
        let e = lower(&plan, &s.catalog).unwrap().explain();
        assert!(!e.contains("IndexLookup"), "{e}");
    }

    #[test]
    fn a_literal_that_cannot_be_a_key_falls_back_to_the_scan() {
        // Each of these would silently become a probe for a *different* key if
        // `to_lane`'s lossy conversion were trusted: 5.5 truncates to 5, and a
        // negative is not a UInt64 at all.
        let mut s = session("UInt64", 200);
        for q in [
            "SELECT v FROM t WHERE id = 5.5",
            "SELECT v FROM t WHERE id = -1",
            "SELECT v FROM t WHERE id IN (5, 5.5)",
        ] {
            let e = phys(&mut s, q);
            assert!(!e.contains("IndexLookup"), "{q} must not probe:\n{e}");
        }
        // ... but an exactly-representable literal of another variant is fine.
        assert!(phys(&mut s, "SELECT v FROM t WHERE id = 5.0").contains("IndexLookup"));
    }

    #[test]
    fn exact_lane_round_trips_or_refuses() {
        let u = DataType::UInt64;
        let i = DataType::Int64;
        let f = DataType::Float64;
        assert_eq!(exact_lane(&Value::UInt(7), &u), Some(7));
        assert_eq!(exact_lane(&Value::Int(7), &u), Some(7));
        assert_eq!(exact_lane(&Value::Float(7.0), &u), Some(7));
        assert_eq!(exact_lane(&Value::Float(7.5), &u), None, "truncation must be refused");
        assert_eq!(exact_lane(&Value::Int(-1), &u), None);
        assert_eq!(exact_lane(&Value::str("7"), &u), None);
        assert_eq!(exact_lane(&Value::Null, &u), None);
        assert_eq!(exact_lane(&Value::Int(-1), &i), Some(crate::common::i64_to_lane(-1)));
        // -0.0 and 0.0 are one value to SQL and share one lane, so either
        // literal has to find the other's row.
        assert_eq!(exact_lane(&Value::Float(-0.0), &f), exact_lane(&Value::Float(0.0), &f));
        assert_eq!(exact_lane(&Value::Float(f64::NAN), &f), None);
        assert_eq!(exact_lane(&Value::Float(f64::INFINITY), &f), None);
        assert_eq!(exact_lane(&Value::str("x"), &DataType::String), None);

        // A decimal key's lane is a *unit count*, so decoding it back as `Int`
        // made the round trip compare `Int(250)` against `Decimal(250, 2)` --
        // never equal, for any literal, so a decimal pk never reached the index.
        let d = DataType::Decimal64(2);
        let lane = crate::common::i64_to_lane;
        assert_eq!(exact_lane(&Value::Decimal(250, 2), &d), Some(lane(250)));
        // Other scales and plain integers name a key when they land on one...
        assert_eq!(exact_lane(&Value::Decimal(2_500, 3), &d), Some(lane(250)));
        assert_eq!(exact_lane(&Value::Int(2), &d), Some(lane(200)));
        // ... and only then: $0.255 rounds to a *different* stored key, and
        // probing for it would return a row no scan would have.
        assert_eq!(exact_lane(&Value::Decimal(255, 3), &d), None);
        // The converse guard still holds: a decimal literal against a plain
        // integer key must not truncate into a probe for 1.
        assert_eq!(exact_lane(&Value::Decimal(150, 2), &i), None);
        assert_eq!(exact_lane(&Value::Decimal(100, 2), &i), Some(lane(1)));
    }

    #[test]
    fn a_decimal_primary_key_reaches_the_index() {
        // `session` fills `id` with lanes 0..n, which on a `Decimal64(2)` key is
        // $0.00 .. $9.99; row 5 is $0.05 and carries v = 15.
        let mut s = session("Decimal64(2)", 1_000);
        for q in [
            "SELECT v FROM t WHERE id = 0.05",
            "SELECT v FROM t WHERE id = CAST('0.05' AS Decimal64(2))",
            "SELECT v FROM t WHERE id IN (0.05, 0.07)",
        ] {
            let e = phys(&mut s, q);
            assert!(e.contains("IndexLookup"), "{q}:\n{e}");
            assert_eq!(s.query(q).unwrap().blocks[0].column(0).value(0), Value::Int(15), "{q}");
        }
        // $0.055 is not a stored key at scale 2, so the probe has to be refused
        // rather than rounded into one -- the same rule `id = 5.5` obeys on a
        // `UInt64` key.
        let e = phys(&mut s, "SELECT v FROM t WHERE id = 0.055");
        assert!(!e.contains("IndexLookup"), "{e}");
        assert_eq!(s.query("SELECT v FROM t WHERE id = 0.055").unwrap().rows(), 0);
    }

    #[test]
    fn a_float_key_is_probed_by_value_not_by_bits() {
        let mut s = session("Float64", 1_000);
        let e = phys(&mut s, "SELECT v FROM t WHERE id = 2.5");
        assert!(e.contains("IndexLookup"), "{e}");
        let rows = s.query("SELECT v FROM t WHERE id = 2.5").unwrap();
        assert_eq!(rows.rows(), 1);
        // id = i/2, so 2.5 is row 5 and v = 15.
        assert_eq!(rows.blocks[0].column(0).value(0), Value::Int(15));
    }

    #[test]
    fn a_huge_in_list_is_cheaper_to_scan() {
        // 200 rows and 100 keys: 100 probes against 200 rows loses.
        let mut s = session("UInt64", 200);
        let list: Vec<String> = (0..100).map(|i| i.to_string()).collect();
        let q = format!("SELECT v FROM t WHERE id IN ({})", list.join(", "));
        let e = phys(&mut s, &q);
        assert!(!e.contains("IndexLookup"), "100 probes over 200 rows:\n{e}");

        // The same list against a table big enough to pay for it does probe.
        let mut big = session("UInt64", 200_000);
        assert!(phys(&mut big, &q).contains("IndexLookup"));
    }

    #[test]
    fn top_k_shows_up_as_top_k() {
        let mut s = session("UInt64", 20_000);
        let e = phys(&mut s, "SELECT v FROM t ORDER BY id DESC LIMIT 10");
        assert!(e.contains("TopK 10"), "{e}");
        // ... and a sort with no limit above it stays a full sort.
        let e = phys(&mut s, "SELECT v FROM t ORDER BY id DESC");
        assert!(e.contains("Sort ["), "{e}");
        assert!(!e.contains("TopK"), "{e}");
    }

    #[test]
    fn the_tree_still_renders_every_level() {
        let mut s = session("UInt64", 1_000);
        let e = phys(
            &mut s,
            "SELECT id, count(*) FROM t WHERE v > 3 GROUP BY id ORDER BY id LIMIT 5",
        );
        for want in ["Limit 5", "Aggregate", "Scan default.t"] {
            assert!(e.contains(want), "missing {want} in:\n{e}");
        }
    }

    #[test]
    fn the_exchange_is_a_plan_node_with_its_width_on_it() {
        // Was invisible: `EXPLAIN PIPELINE` for a query that fanned out to 14
        // workers was byte-identical to one that stayed serial.
        let mut s = session("UInt64", 100_000);
        let e = phys(&mut s, "SELECT id, count(*) FROM t GROUP BY id");
        let line = e.lines().find(|l| l.contains("Exchange")).unwrap_or_else(|| panic!("{e}"));
        let n: usize = line.split_whitespace().nth(1).unwrap().parse().unwrap();
        assert!(n >= 2, "{line}");
        assert!(line.contains("workers"), "{line}");
        // ... and it wraps the blocking node rather than replacing it.
        assert!(e.contains("Aggregate"), "{e}");
    }

    #[test]
    fn a_node_the_serial_builder_owns_never_gets_an_exchange() {
        // `union::build_union` re-lowers each branch through `operators::build`
        // and `exchange::build` hands `Window` to `build_physical`, so neither
        // can honour a fleet. Printing one there would be the same lie in the
        // other direction: a plan that promises parallelism and runs serially.
        // `sum`, not `count(*)`: a bare unfiltered count no longer reaches the
        // exchange at all -- it is answered from part metadata -- so it would
        // pass the negative assertions for the wrong reason and fail the
        // positive one. Every shape here needs an aggregate that still has to
        // read rows.
        let mut s = session("UInt64", 100_000);
        let u = phys(&mut s, "SELECT sum(v) FROM t UNION ALL SELECT sum(v) FROM t");
        assert!(!u.contains("Exchange"), "a UNION branch is built serially:\n{u}");
        let w = phys(&mut s, "SELECT id, row_number() OVER (ORDER BY v) FROM t");
        assert!(!w.contains("Exchange"), "a window's sort is built serially:\n{w}");
        // The same aggregate outside a UNION still fans out, so the flag is
        // doing the narrow thing and not simply switching parallelism off.
        assert!(phys(&mut s, "SELECT sum(v) FROM t").contains("Exchange"));
    }

    #[test]
    fn top_k_still_fuses_through_the_exchange() {
        // `fan_out` wraps the sort before the `Limit` above it is lowered, so
        // `fuse_top_k` has to see through the new node. Missing that arm turns
        // every parallel top-K into a full parallel sort -- right answer,
        // several times slower, and invisible without this assertion.
        let mut s = session("UInt64", 100_000);
        let e = phys(&mut s, "SELECT v FROM t ORDER BY id DESC LIMIT 10");
        assert!(e.contains("Exchange"), "{e}");
        assert!(e.contains("TopK 10"), "the limit must reach the sort under the exchange:\n{e}");
    }

    // ------------------------------------------------- 4. metadata answers

    #[test]
    fn an_aggregate_the_headers_can_answer_says_so_in_the_plan() {
        let mut s = session("UInt64", 20_000);
        let e = phys(&mut s, "SELECT count() FROM t");
        assert!(e.contains("MetaAggregate default.t [count()] from part metadata"), "{e}");
        // The scan is gone, not merely bypassed -- that is the claim.
        assert!(!e.contains("Scan default.t"), "{e}");
        assert!(!e.contains("Exchange"), "there is nothing left to fan out:\n{e}");
        // The extremes name the columns they fold.
        let x = phys(&mut s, "SELECT min(v), max(id) FROM t");
        assert!(x.contains("[min(v#1), max(id#0)]"), "{x}");
        // A filtered fold prints the predicate it still has to honour, and
        // says that a straddling granule is read rather than folded.
        let f = phys(&mut s, "SELECT count() FROM t WHERE id >= 7");
        assert!(f.contains("MetaAggregate"), "{f}");
        assert!(f.contains("where=(id#0 >= 7)"), "{f}");
        assert!(f.contains("straddling granules read"), "{f}");
        // 20 granules is far under the fan-out floor, so no width is claimed.
        assert!(!f.contains("workers"), "{f}");
    }

    #[test]
    fn the_aggregates_that_must_still_read_rows() {
        let mut s = session("UInt64", 20_000);
        for q in [
            // counts non-NULL values, which no header records
            "SELECT count(v) FROM t",
            "SELECT count(DISTINCT id) FROM t",
            // not a count or an extreme
            "SELECT sum(v) FROM t",
            // one unfoldable column refuses the whole path
            "SELECT count(), sum(v) FROM t",
            // per-group counts are not in any header
            "SELECT id, count() FROM t GROUP BY id",
            // an expression, not a column of the scan
            "SELECT min(v * 2) FROM t",
            // an extreme under a predicate describes a subset the bounds do not
            "SELECT max(id) FROM t WHERE id < 100",
            // a predicate off the sort column decides no granule, so the fold
            // would serialize what the scan spreads across every core
            "SELECT count() FROM t WHERE v > 100",
            // a predicate no zone test can express
            "SELECT count() FROM t WHERE id % 3 = 0",
            "SELECT count() FROM t WHERE id IN (1, 2, 3)",
            // the index answers this in one probe; the fold would still walk
            // every zone map looking for the granule that straddles
            "SELECT count() FROM t WHERE id = 5",
        ] {
            let e = phys(&mut s, q);
            assert!(!e.contains("MetaAggregate"), "`{q}` must read rows:\n{e}");
        }
    }

    #[test]
    fn a_tombstone_takes_min_and_max_off_the_fold_but_not_the_count() {
        let mut s = session("UInt64", 20_000);
        assert!(phys(&mut s, "SELECT max(id) FROM t").contains("MetaAggregate"));
        s.catalog.table_by_path_mut("default.t").unwrap().delete_key(&Value::UInt(19_999)).unwrap();
        // A granule's bounds still describe the row that was deleted, so the
        // fold would answer 19999 for a table whose largest live id is 19998.
        assert!(!phys(&mut s, "SELECT max(id) FROM t").contains("MetaAggregate"));
        // The delete masks are exact, so the count is still exact.
        assert!(phys(&mut s, "SELECT count() FROM t").contains("MetaAggregate"));
    }

    #[test]
    fn meta_degree_refuses_before_it_divides() {
        let threads = crate::common::pool::global().threads();
        assert_eq!(meta_degree(0), 1);
        assert_eq!(meta_degree(511), 1, "one worker's worth of walk is not worth a fleet");
        assert_eq!(meta_degree(1_024), 2.min(threads));
        assert!(meta_degree(1 << 20) <= threads);
    }

    #[test]
    fn a_limit_of_zero_is_lowered_to_nothing_at_all() {
        let mut s = session("UInt64", 20_000);
        for q in [
            "SELECT v FROM t LIMIT 0",
            "SELECT sum(v) FROM t LIMIT 0",
            "SELECT v FROM t ORDER BY id DESC LIMIT 0",
            "SELECT v FROM t LIMIT 0 OFFSET 10",
        ] {
            assert_eq!(phys(&mut s, q).trim(), "Empty", "{q}");
        }
        // ... and one more row asked for is a whole pipeline again.
        assert!(phys(&mut s, "SELECT v FROM t LIMIT 1").contains("Scan default.t"));
    }

    // ---------------------------------------------------- 5. sorted reads

    #[test]
    fn the_sort_that_the_storage_order_already_did() {
        let mut s = session("UInt64", 20_000);
        let e = phys(&mut s, "SELECT v FROM t ORDER BY id LIMIT 5");
        assert!(!e.contains("Sort") && !e.contains("TopK"), "{e}");
        // ... and with the sort gone there is nothing blocking left to fan
        // out, so the limit reaches the scan instead of a fleet.
        assert!(!e.contains("Exchange"), "{e}");
        assert!(e.contains("Scan default.t"), "{e}");
        // A full sort of the same order is the same decision; it is the
        // materialization that goes away rather than the early stop.
        assert!(!phys(&mut s, "SELECT v FROM t ORDER BY id").contains("Sort"));
    }

    #[test]
    fn the_orderings_that_must_still_sort() {
        let mut s = session("UInt64", 20_000);
        for q in [
            // storage reads one way
            "SELECT v FROM t ORDER BY id DESC LIMIT 5",
            // not the sort column
            "SELECT v FROM t ORDER BY v LIMIT 5",
            // the sort column, but not leading
            "SELECT v FROM t ORDER BY v, id LIMIT 5",
            // a prefix is not enough: a merged part is sorted by `order_by[0]`
            // and by nothing after it
            "SELECT id, v FROM t ORDER BY id, v LIMIT 5",
            // an expression, not a column
            "SELECT v FROM t ORDER BY id + 0 LIMIT 5",
            // a predicate no zone map decides: dropping the sort would drop
            // the exchange doing the walk, measured 0.29x
            "SELECT v FROM t WHERE v = 42 ORDER BY id LIMIT 5",
            // rows the scan never produced in any order
            "SELECT id, count() FROM t GROUP BY id ORDER BY id LIMIT 5",
        ] {
            let e = phys(&mut s, q);
            assert!(e.contains("Sort") || e.contains("TopK"), "{q} must sort:\n{e}");
        }
    }

    #[test]
    fn parts_are_only_concatenated_when_they_do_not_overlap() {
        // Two parts, the second entirely above the first: reading them back to
        // back is key order.
        let mut s = session("UInt64", 1_000);
        {
            let t = s.catalog.table_by_path_mut("default.t").unwrap();
            t.insert(
                Block::new(vec![
                    Column::u64s(DataType::UInt64, (1_000..2_000).collect()),
                    Column::i64s(DataType::Int64, (0..1_000).collect()),
                ])
                .unwrap(),
            )
            .unwrap();
            t.flush().unwrap();
        }
        assert!(!phys(&mut s, "SELECT v FROM t ORDER BY id LIMIT 5").contains("Sort"));

        // One more part straddling both: the set no longer concatenates in
        // order, and no rule may pretend otherwise.
        {
            let t = s.catalog.table_by_path_mut("default.t").unwrap();
            t.insert(
                Block::new(vec![
                    Column::u64s(DataType::UInt64, (5_000..6_000).map(|i| i % 1_500).collect()),
                    Column::i64s(DataType::Int64, (0..1_000).collect()),
                ])
                .unwrap(),
            )
            .unwrap();
            t.flush().unwrap();
        }
        let e = phys(&mut s, "SELECT v FROM t ORDER BY id LIMIT 5");
        assert!(e.contains("TopK"), "an overlapping part needs the merge:\n{e}");
    }

    #[test]
    fn a_plan_too_deep_is_refused_rather_than_overflowing() {
        let schema = Schema::new(vec![Field::new("a", DataType::Int64)]).unwrap();
        let mut p = LogicalPlan::Empty { schema };
        for _ in 0..MAX_PLAN_DEPTH + 2 {
            p = LogicalPlan::Distinct { input: Box::new(p) };
        }
        let cat = Catalog::in_memory();
        assert!(lower(&p, &cat).is_err());
    }

    #[test]
    fn a_part_without_the_index_is_not_probed() {
        // Built by hand with no pk column, then adopted by a table that claims
        // one: `find` would answer `None` for every key, which is a wrong
        // answer rather than a slow one.
        let mut cat = Catalog::in_memory();
        cat.create_table(
            TableDef {
                name: "t".into(),
                schema: Schema::new(vec![
                    Field::new("id", DataType::UInt64),
                    Field::new("v", DataType::Int64),
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
        let block = Block::new(vec![
            Column::u64s(DataType::UInt64, (0..2_000).collect()),
            Column::i64s(DataType::Int64, (0..2_000).collect()),
        ])
        .unwrap();
        let unkeyed = crate::storage::Part::build(&block, Some(0), None).unwrap();
        let t = cat.table_by_path_mut("default.t").unwrap();
        t.set_parts(vec![unkeyed]);

        let full = t.schema().clone();
        let node = ScanNode {
            table: "default.t".into(),
            projection: vec![0, 1],
            schema: full,
            filters: vec![BoundExpr::Binary {
                left: Box::new(BoundExpr::Column {
                    index: 0,
                    ty: DataType::UInt64,
                    name: "id".into(),
                }),
                op: BinaryOp::Eq,
                right: Box::new(BoundExpr::lit(Value::UInt(5))),
                ty: DataType::Bool,
            }],
            zone_filters: vec![],
        };
        assert!(index_path(&node, &cat).is_none());
    }
}
