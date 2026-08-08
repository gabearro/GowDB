//! The three table access paths: sequential [`Scan`], [`IndexLookup`], and
//! [`MetaAggregate`], which reads no table at all.
//!
//! Which one a query gets is decided in [`crate::planner::physical`], not here.
//! The first two produce the same thing -- blocks of the scan node's projected
//! schema, already narrowed by its PREWHERE list -- so nothing above them can
//! tell them apart, and swapping one for the other is an access-path decision
//! rather than a plan rewrite. The third replaces the aggregate above the scan
//! as well, because its whole point is that there is no row stream.
//!
//! # Sequential scan
//!
//! This is the operator the whole storage design exists to serve, and it reads
//! bottom-up in three stages, each strictly cheaper than the one after it:
//!
//! 1. **prune** -- for each granule, test the
//!    [`ZoneFilter`](crate::planner::logical::ZoneFilter)s against the
//!    per-column `(min, max)` recorded in the packed column's frame-of-
//!    reference header. If any filter provably cannot match, the granule is
//!    skipped *without decoding a single byte*. A granule is 1024 rows, so on
//!    a selective range query over a sorted key this throws away almost the
//!    whole table for the cost of two `Value` comparisons per granule.
//! 2. **project** -- decode only the columns the plan asked for, and only the
//!    rows the delete bitmap says are live. Columns nobody references are
//!    never touched, which is the entire point of a column store.
//! 3. **prewhere** -- run [`ScanNode::filters`] against the decoded block and
//!    `take()` the survivors, before any operator above sees a row.
//!
//! ## The two index spaces
//!
//! `ScanNode::projection` holds **table** column indices;
//! [`Part::read_columns`] takes those directly. `ScanNode::filters` and
//! `ScanNode::zone_filters` are expressed against the **projected** schema,
//! because that is what the operators above the scan see. So a zone filter's
//! `col` has to be mapped back through `projection` before it can name a
//! packed column. Getting this backwards yields a scan that prunes on the
//! wrong column and silently returns wrong answers, so the mapping lives in
//! exactly one place (`Scan::prunes`) and is asserted in the tests.
//!
//! ## Batching
//!
//! Granules are 1024 rows and blocks are [`BLOCK_SIZE`]; yielding a block per
//! granule would multiply the per-batch overhead of every operator above by
//! eight. So survivors accumulate until the batch is full. The accumulator is
//! also what makes a highly selective filter cheap upstream: a thousand
//! granules that each keep two rows still produce one block, not a thousand.
//!
//! ## The snapshot
//!
//! The operator pins one [`Snapshot`] at construction and reads nothing else
//! for the rest of its life. That is what makes the answer a *view of the
//! table at one instant* rather than a walk over whatever the storage layer
//! happened to hold when each granule came up -- and it costs one lock
//! acquisition per scan, not one per granule.
//!
//! A side effect worth knowing about: the operator no longer borrows the
//! `Table` at all. It holds `Arc`s to the parts, so the only thing tying its
//! lifetime to the catalog is the `&ScanNode` it reads the plan from. Whoever
//! parallelizes the executor or lets a write proceed under a running query
//! starts from here.
//!
//! # Index lookup
//!
//! [`IndexLookup`] answers the same scan node by *naming* its rows instead of
//! walking to them: one CHD minimal-perfect-hash probe per key, guarded by the
//! part's split-block bloom. On a 10M-row table that is 3.6 us against the
//! scan's 138.5 us, and the gap grows with the table because the scan's cost is
//! the granule count and the lookup's is not. See
//! [`crate::planner::physical`] for the full measurement and for the negative
//! cases, which must not move.
//!
//! Three things make it produce the same answer as the scan rather than a
//! faster wrong one:
//!
//!   * **the PREWHERE list still runs.** The index picks candidate rows; every
//!     predicate in `ScanNode::filters`, including the equality that selected
//!     the path, is then evaluated over them exactly as it would have been over
//!     a sequentially decoded block. Over-producing candidates is therefore
//!     harmless and only a *missed* row can be wrong.
//!   * **it reads parts and nothing else,** like the scan next door. The write
//!     buffer is flushed before planning, so the two see the same table; an
//!     index path that consulted the delta would answer a question the scan
//!     does not.
//!   * **it takes the whole run of equal keys,** not the first. `ingest_block`
//!     now dedups every batch, so a part this build writes holds each key once
//!     -- but a part written before that fix and left on disk does not, and the
//!     MPH would then name one row of a run the scan returns whole. Parts are
//!     sorted by that key, so proving the run has length one costs two lane
//!     reads. See [`collect_run`].
//!
//! # Metadata aggregate
//!
//! [`MetaAggregate`] answers `count()`, `min(c)` and `max(c)` by folding the
//! numbers a part already carries -- granule row counts, delete masks, zone
//! maps -- and emits the single row the aggregate above it would have
//! produced. Its cost is the *part* count for an unfiltered `count` and the
//! *granule* count for the rest, so `SELECT count() FROM t` stops being a
//! throughput number (2.2-2.4 ms over 2M rows) and becomes a latency one:
//! 5.8 us at 2M rows, 5.8 us at 4M, 5.2 us at 8M.
//! [`crate::planner::physical`] holds the full table of measurements and,
//! more importantly, the list of shapes it refuses.
//!
//! The one thing it shares with `Scan` is the fallback: a granule whose zone
//! maps cannot decide the predicate either way is decoded and filtered exactly
//! as `Scan::next` would have done it, which is why an over-eager covering
//! test would be the only way to a wrong answer -- and why `covers` is defined
//! as "the pruning test, applied to the negation" rather than as new
//! reasoning.

use crate::catalog::Catalog;
use crate::common::{hash_key, Error, Result, BLOCK_SIZE, FP_SEED, GRANULE_SIZE, G_SHIFT};
use crate::exec::expr;
use crate::planner::logical::ScanNode;
use crate::planner::physical::{IndexPath, MetaAgg, MetaPath};
use crate::storage::part::{Deletes, Snapshot};
use crate::storage::{Part, Stats};
use crate::types::{Block, Column, ColumnBuilder, Schema, Value};

use super::{Operator, ScanStats};

pub struct Scan<'a> {
    node: &'a ScanNode,
    snap: Snapshot,
    part: usize,
    granule: usize,
    /// Live-row selection of the granule in hand, reused across granules: a
    /// table with deletes used to allocate one of these per 1024 rows.
    sel: Vec<u32>,
    /// Survivors waiting to reach [`BLOCK_SIZE`].
    acc: Option<Block>,
    stats: ScanStats,
}

impl<'a> Scan<'a> {
    pub fn new(node: &'a ScanNode, catalog: &'a Catalog) -> Result<Scan<'a>> {
        let table = catalog.table_by_path(&node.table)?;
        let ncols = table.schema().len();
        for &c in &node.projection {
            if c >= ncols {
                return Err(Error::exec(format!(
                    "scan of `{}` projects column #{c}, but the table has {ncols}",
                    node.table
                )));
            }
        }
        Ok(Scan {
            node,
            snap: table.snapshot(),
            part: 0,
            granule: 0,
            sel: Vec::new(),
            acc: None,
            stats: ScanStats::default(),
        })
    }

    pub fn stats(&self) -> ScanStats {
        self.stats
    }

    /// Can every zone filter still be satisfied somewhere in this granule?
    ///
    /// `false` means "read it"; `true` means "provably empty, skip it". Bounds
    /// come from the FOR header, so this costs no I/O and no decode.
    ///
    /// Do not cache `min_value`/`max_value` across the loop. `BETWEEN` reaches
    /// here as two filters on one column, so recomputing the bounds for the
    /// second looks like obvious waste -- it is not, because both are pure
    /// reads of the same header and LLVM already folds them. Measured
    /// interleaved (temporary `AtomicBool` switch, alternating in one loop,
    /// best-of-15 per side, 2M rows / 1954 granules, every one pruned so the
    /// run *is* this loop): one filter 36.0 us, two filters 36.0 us -- the
    /// second filter is free. Carrying the bounds in an `Option<(usize, Value,
    /// Value)>` instead measured **0.956x at one filter and 0.953x at two**,
    /// i.e. 5% slower on both, because the bookkeeping is real and the saving
    /// is not.
    fn prunes(&self, p: &Part, gi: usize) -> bool {
        if self.node.zone_filters.is_empty() {
            return false;
        }
        let g = &p.granules[gi];
        for zf in &self.node.zone_filters {
            // `zf.col` indexes the projected schema; map it back to storage.
            let Some(&table_col) = self.node.projection.get(zf.col) else {
                continue;
            };
            let Some(pc) = g.columns.get(table_col) else {
                continue;
            };
            if !zf.may_match(&pc.min_value(), &pc.max_value()) {
                return true;
            }
        }
        false
    }
}

impl Operator for Scan<'_> {
    fn schema(&self) -> &Schema {
        &self.node.schema
    }

    fn stats(&self) -> ScanStats {
        self.stats
    }

    fn next(&mut self) -> Result<Option<Block>> {
        while self.part < self.snap.len() {
            let p = self.snap.part(self.part);
            if self.granule >= p.granule_count() {
                self.part += 1;
                self.granule = 0;
                continue;
            }
            let gi = self.granule;
            self.granule += 1;

            if self.prunes(p, gi) {
                self.stats.granules_pruned += 1;
                continue;
            }
            self.stats.granules_read += 1;

            let del: Option<&Deletes> = self.snap.deletes(self.part);
            let live = p.live_selection_into(gi, del, &mut self.sel);
            let mut blk = p.read_columns(gi, &self.node.projection, live)?;
            self.stats.rows_read += blk.rows() as u64;
            if blk.rows() == 0 {
                continue;
            }

            // PREWHERE: narrow before anything upstream sees a row.
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

// ---------------------------------------------------------- index lookup

/// A scan answered by the primary-key index: probe, gather, filter.
///
/// The keys were resolved and vetted by `index_path` in
/// [`crate::planner::physical`], so everything left to do here is storage work.
/// Nothing in this operator decides *whether* the index applies.
pub struct IndexLookup<'a> {
    /// The scan node this replaces: projection, output schema and PREWHERE.
    node: &'a ScanNode,
    snap: Snapshot,
    /// Key lanes, sorted ascending and deduplicated by the planner.
    keys: Vec<u64>,
    /// `(part, row position)` of every live candidate row, in scan order.
    /// Resolved once on the first `next()`.
    hits: Vec<(u32, u32)>,
    /// Row offsets within one granule, reused across gathers.
    sel: Vec<u32>,
    cursor: usize,
    resolved: bool,
    /// Survivors waiting to reach [`BLOCK_SIZE`], as in [`Scan`].
    acc: Option<Block>,
    /// Granules in the pinned snapshot, so `stats()` can report how many the
    /// lookup did *not* have to read.
    total_granules: u64,
    stats: ScanStats,
}

impl<'a> IndexLookup<'a> {
    pub fn new(path: IndexPath<'a>, catalog: &Catalog) -> Result<IndexLookup<'a>> {
        let table = catalog.table_by_path(&path.node.table)?;
        let snap = table.snapshot();
        // Refuse a part that cannot answer this key rather than reading `None`
        // out of it. `Part::find` needs the MPH keyed on `key_col`, and
        // `collect_run` reads the *sort* lane; `TableDef::pk_col` only exists
        // when the two columns are the same one, so both have to agree with
        // `path.key_col` or every probe is a silent miss -- a wrong answer, not
        // a slow one. Checked once here, which is also why the operator does
        // not keep `key_col`: after this loop the part's own fields are the
        // authority.
        //
        // The planner already declines to lower such a table -- and declining
        // is the right answer *there*, because a scan still works. Here there is
        // no scan left to fall back to, so it is an error. `IndexPath` is a
        // public type, so this is also what keeps a hand-built one honest.
        // One pass over at most `AUTO_COMPACT_PARTS` parts, at plan time.
        let mut total_granules = 0u64;
        for p in snap.parts() {
            if p.pk_col != Some(path.key_col) || p.sort_col != Some(path.key_col) {
                return Err(Error::exec(format!(
                    "index lookup on `{}` names column #{} as the key, but a part is indexed \
                     on {:?}/{:?} (pk/sort)",
                    path.node.table, path.key_col, p.pk_col, p.sort_col
                )));
            }
            total_granules += p.granule_count() as u64;
        }
        Ok(IndexLookup {
            node: path.node,
            snap,
            // Moved, not cloned: the planner built this vector and has no
            // further use for it, and an `IN` list can be long.
            keys: path.keys,
            hits: Vec::new(),
            sel: Vec::new(),
            cursor: 0,
            resolved: false,
            acc: None,
            total_granules,
            stats: ScanStats::default(),
        })
    }

    /// Report the granules never touched as *pruned*.
    ///
    /// The counters exist so a test can prove an access path did what it
    /// claimed, and `granules_pruned + granules_read == total` is the invariant
    /// the scan's tests assert. An index lookup skips granules by seeking
    /// rather than by testing a zone map, but "granules this query did not have
    /// to decode" means the same thing to whoever reads the number, and leaving
    /// it at zero would make the fastest access path in the engine look like
    /// the one that pruned nothing.
    fn counters(&self) -> ScanStats {
        ScanStats {
            granules_pruned: self.total_granules.saturating_sub(self.stats.granules_read),
            ..self.stats
        }
    }

    pub fn stats(&self) -> ScanStats {
        self.counters()
    }

    /// Probe every key against every part, collecting live candidate rows.
    ///
    /// Parts are visited oldest-first and keys ascending, and a part is sorted
    /// by the same lane, so `hits` comes out in exactly the order a sequential
    /// scan would have produced -- no sort, and `SELECT ... WHERE id IN (...)`
    /// without an `ORDER BY` returns the rows in the same order it did before.
    ///
    /// Every part is probed rather than stopping at the newest hit the way
    /// `Table::locate` does. `locate` may stop because an ingest tombstones the
    /// keys it shadows, so a live key exists in exactly one part; a scan does
    /// not rely on that invariant and neither does this. The cost is bounded:
    /// `AUTO_COMPACT_PARTS` caps the part count at 16 and a foreign part is
    /// rejected by one bloom probe.
    fn resolve(&mut self) {
        let set = self.snap.set();
        // Storage-level counters. The pipeline reads `ScanStats`, not these,
        // and `Part::find` insists on somewhere to put them.
        let mut st = Stats::default();
        // The bloom is only built once a second part exists, and `may_contain`
        // answers "maybe" without one -- so the test is worth making only when
        // there is a foreign part to reject. Hoisted: it is constant for the
        // whole resolution.
        let multi = set.len() > 1;
        self.hits.reserve(self.keys.len());
        for pi in 0..set.len() {
            let p = set.part(pi);
            let del = set.deletes(pi);
            for &lane in &self.keys {
                // Recomputed per part rather than hoisted into a `Vec<u64>` of
                // fingerprints: the hoist would allocate once per query to save
                // two multiplies per (part, key), and the shape it helps -- many
                // keys *and* many parts -- is the one `AUTO_COMPACT_PARTS`
                // exists to prevent. The single-part case, which is every
                // compacted table, would pay the allocation for nothing.
                let fph = hash_key(lane, FP_SEED);
                if multi && !p.may_contain(fph) {
                    continue;
                }
                // `find`, not `find_live`: a deleted row must not hide the rest
                // of its key's run, and `collect_run` checks the delete mask
                // per row anyway. For the unique case the two cost the same.
                let Some(pos) = p.find(lane, fph, &mut st) else { continue };
                collect_run(p, del, pi as u32, pos, lane, &mut self.hits);
            }
        }
        debug_assert!(
            self.hits.windows(2).all(|w| w[0] < w[1]),
            "hits must come out in scan order and be distinct"
        );
    }
}

impl Operator for IndexLookup<'_> {
    fn schema(&self) -> &Schema {
        &self.node.schema
    }

    fn stats(&self) -> ScanStats {
        self.counters()
    }

    fn next(&mut self) -> Result<Option<Block>> {
        if !self.resolved {
            self.resolve();
            self.resolved = true;
        }
        // Outer: one iteration per emitted batch. A batch whose every row was
        // rejected by the residual must not end the stream, so it goes round
        // again instead of returning `None` with hits still pending.
        while self.cursor < self.hits.len() {
            while self.cursor < self.hits.len() {
                // One gather per (part, granule): `read_columns` decodes out of
                // a granule's packed columns, so hits that share one must share
                // a call or the granule is re-entered per row.
                let (pi, first) = self.hits[self.cursor];
                let gi = (first as usize) >> G_SHIFT;
                self.sel.clear();
                while let Some(&(q, pos)) = self.hits.get(self.cursor) {
                    if q != pi || (pos as usize) >> G_SHIFT != gi {
                        break;
                    }
                    self.sel.push(pos & (GRANULE_SIZE as u32 - 1));
                    self.cursor += 1;
                }
                self.stats.granules_read += 1;

                let blk = self.snap.part(pi as usize).read_columns(
                    gi,
                    &self.node.projection,
                    Some(&self.sel),
                )?;
                self.stats.rows_read += blk.rows() as u64;
                match &mut self.acc {
                    None => self.acc = Some(blk),
                    Some(a) => a.extend(&blk)?,
                }
                if self.acc.as_ref().is_some_and(|a| a.rows() >= BLOCK_SIZE) {
                    break;
                }
            }
            if let Some(b) = self.emit()? {
                return Ok(Some(b));
            }
        }
        self.emit()
    }
}

impl IndexLookup<'_> {
    /// Apply the PREWHERE list to the accumulator and hand it over.
    ///
    /// This is the same loop [`Scan`] runs, over the same predicates, and it is
    /// what makes an over-eager candidate harmless. It runs **once per emitted
    /// block**, not once per granule, which is the whole difference between the
    /// two access paths on this step: a scan's granule already holds 1024 rows,
    /// while a lookup's holds as many rows as landed in it -- usually one. A
    /// 1024-key `IN` therefore used to pay 1024 `eval_predicate` calls over
    /// one-row blocks, where the vectorized evaluator has nothing to vectorize
    /// over and the per-call setup *is* the cost -- and each of those calls
    /// walks the whole `IN` list, so it is quadratic in the key count.
    ///
    /// Measured interleaved (temporary `AtomicBool` switch, alternating in one
    /// loop, best-of-9 per side, 10M-row table): per-granule vs per-block is
    /// 6.9 vs 6.2 us at 1 key, 8.0 vs 7.5 us at 5, 92 vs 33 us at 64, 837 vs
    /// 112 us at 256, 8.10 vs 0.36 ms at 1024, and 230 vs 1.5 ms at 4096. A
    /// 156x win at the top of the range and a small one at the bottom; there is
    /// no key count at which the old shape was better.
    ///
    /// Returns `None` for an empty result so the pipeline sees end-of-stream
    /// rather than a zero-row block, matching `Scan`.
    fn emit(&mut self) -> Result<Option<Block>> {
        let Some(mut blk) = self.acc.take() else { return Ok(None) };
        for f in &self.node.filters {
            let sel = expr::eval_predicate(f, &blk)?;
            if sel.len() < blk.rows() {
                blk = blk.take(&sel);
            }
            if blk.rows() == 0 {
                return Ok(None);
            }
        }
        Ok(Some(blk))
    }
}

/// Append every *live* row of `p` whose key lane equals `lane`.
///
/// `Part::find` answers with one row, which is the whole answer for a part that
/// honours its table's unique-key declaration. Every part this build writes
/// does: `ingest_block` used to run `dedup_perm_last_by_key` only when the
/// incoming batch needed sorting, so a bulk insert of >= 4096 rows already in
/// key order carried its duplicates straight into a part, and it now dedups
/// unconditionally (see `a_declared_primary_key_collapses_a_run_of_duplicates`).
///
/// This stays anyway, because parts outlive the code that wrote them: one
/// packed before that fix and still on disk holds the run, and answering it with
/// `find`'s single row is a silently dropped row -- the one failure mode a
/// benchmark cannot see.
///
/// Rows are sorted by the key, so the run is contiguous and the walk stops on
/// the first neighbour that differs -- two lane reads, off the line `find` just
/// touched, to prove the usual case has length one.
#[inline]
fn collect_run(
    p: &Part,
    del: Option<&Deletes>,
    pi: u32,
    pos: usize,
    lane: u64,
    out: &mut Vec<(u32, u32)>,
) {
    // `find` may land anywhere inside the run: on the first row for a granule
    // that fell back to a lower-bound search, on an arbitrary one for a granule
    // with an MPH. Walk back to the start, then forward over the whole run.
    let mut start = pos;
    while let Some(q) = prev_row(p, start) {
        if p.sort_lane_at(q) != lane {
            break;
        }
        start = q;
    }
    let mut q = start;
    loop {
        if del.is_none_or(|d| !d.get(q)) {
            out.push((pi, q as u32));
        }
        match next_row(p, q) {
            Some(n) if p.sort_lane_at(n) == lane => q = n,
            _ => return,
        }
    }
}

/// Row positions are granule-major (`granule << G_SHIFT | offset`) and only the
/// final granule may be partial, so stepping across a boundary is a branch, not
/// an increment.
#[inline]
fn next_row(p: &Part, pos: usize) -> Option<usize> {
    let gi = pos >> G_SHIFT;
    if (pos & (GRANULE_SIZE - 1)) + 1 < p.granules[gi].len {
        return Some(pos + 1);
    }
    let n = gi + 1;
    (n < p.granules.len() && p.granules[n].len > 0).then_some(n << G_SHIFT)
}

#[inline]
fn prev_row(p: &Part, pos: usize) -> Option<usize> {
    if pos & (GRANULE_SIZE - 1) != 0 {
        return Some(pos - 1);
    }
    let gi = pos >> G_SHIFT;
    let g = p.granules.get(gi.checked_sub(1)?)?;
    (g.len > 0).then(|| ((gi - 1) << G_SHIFT) + g.len - 1)
}

// ------------------------------------------------------ metadata aggregate

/// An aggregate answered out of part headers: the numbers, not the rows.
///
/// Emits exactly one block of one row -- the row the [`Aggregate`] it replaced
/// would have produced -- and then end of stream. Which aggregates may take
/// this path, and which are refused because the headers would only
/// approximate them, is decided by `meta_path` in
/// [`crate::planner::physical`].
///
/// [`Aggregate`]: super::aggregate::Aggregate
pub struct MetaAggregate<'a> {
    path: MetaPath<'a>,
    snap: Snapshot,
    /// Live-row selection of the granule in hand, reused across granules --
    /// the same buffer, for the same reason, as `Scan`'s.
    sel: Vec<u32>,
    /// Running `min`/`max`, one slot per output column; a `Count` column's
    /// slot stays `None`. Allocated once here so the fold allocates nothing.
    ext: Vec<Option<Value>>,
    done: bool,
    stats: ScanStats,
}

impl<'a> MetaAggregate<'a> {
    pub fn new(path: MetaPath<'a>, catalog: &Catalog) -> Result<MetaAggregate<'a>> {
        let table = catalog.table_by_path(&path.node.table)?;
        // The same check, with the same message, `Scan::new` makes: this path
        // replaces a scan, so a plan that names a column off the end of the
        // table has to fail the way it always did.
        let ncols = table.schema().len();
        for &c in &path.node.projection {
            if c >= ncols {
                return Err(Error::exec(format!(
                    "scan of `{}` projects column #{c}, but the table has {ncols}",
                    path.node.table
                )));
            }
        }
        let ext = vec![None; path.what.len()];
        Ok(MetaAggregate {
            snap: table.snapshot(),
            path,
            sel: Vec::new(),
            ext,
            done: false,
            stats: ScanStats::default(),
        })
    }

    pub fn stats(&self) -> ScanStats {
        self.stats
    }

    /// Live rows in granule `gi`, by the arithmetic the scan itself uses.
    ///
    /// Deliberately *not* `g.len - d.granule_deleted(gi)`: that counts the
    /// whole 1024-bit window, which for a part's partial final granule
    /// includes the padding past its last row. `live_selection_into` is the
    /// function `Scan::next` picks rows with, so routing the dirty case
    /// through it is what makes "live" mean one thing in this engine rather
    /// than two. The clean case -- one `u16` load -- is the one that has to be
    /// free, and is.
    fn live(p: &Part, gi: usize, del: Option<&Deletes>, sel: &mut Vec<u32>) -> usize {
        let n = p.granules[gi].len;
        match del {
            None => n,
            Some(d) if d.granule_deleted(gi) == 0 => n,
            Some(d) => p.live_selection_into(gi, Some(d), sel).map_or(n, |s| s.len()),
        }
    }

    /// Live rows of the whole snapshot: the unfiltered `count()`.
    ///
    /// `Part::n_rows` is the sum of its granule lengths by construction, so a
    /// table nobody has deleted from costs one add per part and nothing per
    /// granule -- which is what makes this independent of the table's size.
    /// Only a part carrying tombstones pays the granule walk.
    fn count_all(&mut self) -> u64 {
        let mut n = 0u64;
        for pi in 0..self.snap.len() {
            let p = self.snap.part(pi);
            self.stats.granules_pruned += p.granule_count() as u64;
            match self.snap.deletes(pi) {
                Some(d) if d.count() > 0 => {
                    for gi in 0..p.granule_count() {
                        n += Self::live(p, gi, Some(d), &mut self.sel) as u64;
                    }
                }
                _ => n += p.n_rows as u64,
            }
        }
        n
    }

    /// `count()` under a predicate the zone maps can decide for whole
    /// granules.
    ///
    /// The general case of [`count_all`](Self::count_all): a covered granule
    /// contributes its live rows, an excluded one contributes zero, and only
    /// a straddler is decoded -- through the same three steps, in the same
    /// order, that `Scan::next` runs.
    fn count_where(&mut self) -> Result<u64> {
        if self.path.workers > 1 {
            return self.count_where_parallel();
        }
        let mut n = 0u64;
        for pi in 0..self.snap.len() {
            let p = self.snap.part(pi);
            let del = self.snap.deletes(pi);
            for gi in 0..p.granule_count() {
                n += fold_granule(&self.path, p, del, gi, &mut self.sel, &mut self.stats)?;
            }
        }
        Ok(n)
    }

    /// [`count_where`](Self::count_where) across the pool.
    ///
    /// The walk this spreads is the same walk `Scan` does, and `Scan`'s runs
    /// inside the exchange's fleet -- so leaving this serial handed the
    /// parallel scan a 14x head start on precisely the queries whose entire
    /// cost is pruning. The width is [`MetaPath::workers`], chosen at plan
    /// time and printed by `EXPLAIN`; this obeys it rather than re-deciding,
    /// exactly as `exchange::build` obeys its own node.
    fn count_where_parallel(&mut self) -> Result<u64> {
        // `(part, granule)` pairs so a claim can cross a part boundary, the
        // same shape `Table::scan_fold_in` uses. One allocation per query --
        // 16 KB for a 2M-row table, ~0.6 us to build, against the 90 us walk
        // it is spreading.
        let work: Vec<(u32, u32)> = (0..self.snap.len())
            .flat_map(|pi| {
                (0..self.snap.part(pi).granule_count()).map(move |g| (pi as u32, g as u32))
            })
            .collect();
        let cursor = std::sync::atomic::AtomicUsize::new(0);
        let (snap, path, work) = (&self.snap, &self.path, &work);

        let parts: Vec<Result<(u64, ScanStats)>> =
            crate::common::pool::global().map(path.workers, |_| {
                // One scratch buffer per worker for the whole walk, not one
                // per granule: only a straddler touches it at all.
                let mut sel = Vec::new();
                let mut st = ScanStats::default();
                let mut n = 0u64;
                loop {
                    let at = cursor.fetch_add(META_CLAIM, std::sync::atomic::Ordering::Relaxed);
                    if at >= work.len() {
                        break;
                    }
                    let end = (at + META_CLAIM).min(work.len());
                    for &(pi, gi) in &work[at..end] {
                        let p = snap.part(pi as usize);
                        let del = snap.deletes(pi as usize);
                        n += fold_granule(path, p, del, gi as usize, &mut sel, &mut st)?;
                    }
                }
                Ok((n, st))
            });

        let mut n = 0u64;
        for r in parts {
            let (c, st) = r?;
            n += c;
            self.stats.merge(&st);
        }
        Ok(n)
    }

    /// Fold every granule's zone map into the `min`/`max` slots.
    ///
    /// Only reachable with no predicate and no tombstone anywhere in the
    /// snapshot -- the planner refuses otherwise -- so a granule's bounds
    /// describe exactly the rows a scan would have seen.
    fn fold_extremes(&mut self) {
        for pi in 0..self.snap.len() {
            let p = self.snap.part(pi);
            for g in &p.granules {
                for (slot, w) in self.ext.iter_mut().zip(&self.path.what) {
                    let MetaAgg::Extreme { col, max } = w else { continue };
                    let Some(pc) = g.columns.get(*col) else { continue };
                    // These ignore NULL rows -- which is what SQL `min`/`max`
                    // do -- and decode a string through the granule's own
                    // dictionary, so what comes back is a value and the
                    // comparison below is over values, not lanes.
                    let v = if *max { pc.max_value() } else { pc.min_value() };
                    // An empty or all-NULL granule reports `Null` bounds, and
                    // `Value`'s order puts NULL below every number: offering
                    // it would answer `min` with NULL for a table that has a
                    // perfectly good smallest row.
                    if v.is_null() {
                        continue;
                    }
                    offer(slot, v, *max);
                }
            }
        }
    }

    /// The one row, built the way `aggregate::finish_block` builds it.
    fn emit(&self, count: u64) -> Result<Block> {
        let mut cols: Vec<Column> = Vec::with_capacity(self.path.what.len());
        for (i, w) in self.path.what.iter().enumerate() {
            // `aggregate::out_types`' rule: the plan's schema wins where it
            // has an opinion, the aggregate's own type is the fallback.
            let ty = if i < self.path.schema.len() {
                self.path.schema.ty(i).clone()
            } else {
                self.path.aggs[i].ty.clone()
            };
            let v = match w {
                MetaAgg::Count => Value::UInt(count),
                MetaAgg::Extreme { .. } => self.ext[i].clone().unwrap_or(Value::Null),
            };
            let mut b = ColumnBuilder::with_capacity(ty, 1);
            b.push_value(&v)?;
            let mut c = b.finish();
            // `min` of nothing is NULL whatever the column was declared as, so
            // a column can acquire a mask the schema did not promise. A live
            // mask must never sit on a non-Nullable type.
            if c.has_nulls() && !c.ty.is_nullable() {
                c.ty = c.ty.to_nullable();
            }
            cols.push(c);
        }
        Block::new(cols)
    }
}

impl Operator for MetaAggregate<'_> {
    fn schema(&self) -> &Schema {
        self.path.schema
    }

    fn stats(&self) -> ScanStats {
        self.stats
    }

    fn next(&mut self) -> Result<Option<Block>> {
        if self.done {
            return Ok(None);
        }
        self.done = true;
        let n = if self.path.preds.is_empty() {
            let n = self.count_all();
            if self.path.what.iter().any(|w| matches!(w, MetaAgg::Extreme { .. })) {
                self.fold_extremes();
            }
            n
        } else {
            self.count_where()?
        };
        // Exactly one row, always -- a global aggregate over an empty table
        // still answers `0` for `count` and `NULL` for `min`, and returning
        // end-of-stream instead would lose the row rather than the work.
        Ok(Some(self.emit(n)?))
    }
}

/// Granules one worker claims at a time.
///
/// A zone test is ~50 ns, so 64 granules is ~3 us of work per claim: the
/// shared counter costs well under 1% of that, and the tail a straggler can
/// leave is the same 3 us. `Table::scan_fold_in` claims 8, because its unit is
/// a *decoded* granule -- three orders of magnitude more expensive -- and the
/// tail it is protecting against is correspondingly longer.
const META_CLAIM: usize = 64;

/// One granule's contribution to a filtered count.
///
/// The whole body of the walk, so the serial and parallel loops cannot drift:
/// the only difference between them is who calls this and where the counters
/// land. A straddler goes through the same three steps, in the same order,
/// that `Scan::next` runs -- select live rows, decode the projection, apply
/// the PREWHERE list -- because agreeing with the scan is the entire
/// specification.
fn fold_granule(
    path: &MetaPath<'_>,
    p: &Part,
    del: Option<&Deletes>,
    gi: usize,
    sel: &mut Vec<u32>,
    stats: &mut ScanStats,
) -> Result<u64> {
    match verdict(path, p, gi) {
        Verdict::Empty => {
            stats.granules_pruned += 1;
            Ok(0)
        }
        Verdict::All => {
            stats.granules_pruned += 1;
            Ok(MetaAggregate::live(p, gi, del, sel) as u64)
        }
        Verdict::Mixed => {
            stats.granules_read += 1;
            let live = p.live_selection_into(gi, del, sel);
            let mut blk = p.read_columns(gi, &path.node.projection, live)?;
            stats.rows_read += blk.rows() as u64;
            for f in &path.node.filters {
                if blk.rows() == 0 {
                    break;
                }
                let s = expr::eval_predicate(f, &blk)?;
                if s.len() < blk.rows() {
                    blk = blk.take(&s);
                }
            }
            Ok(blk.rows() as u64)
        }
    }
}

/// What one granule's zone maps say about the predicate, before decoding it.
enum Verdict {
    /// No row can match: the granule contributes nothing.
    Empty,
    /// Every row matches: the granule contributes its live row count.
    All,
    /// Neither is provable. Decode it.
    Mixed,
}

/// Decide granule `gi` from its zone maps alone.
///
/// The `Empty` half is the scan's own pruning test, unchanged. The `All` half
/// is *the same test applied to the negated comparison*: no row fails the
/// conjunct, so every row satisfies it. Writing it that way rather than as a
/// fresh `min >= v`-style rule is the point -- pruning is already load-bearing
/// for correctness everywhere else in the engine, so the covering test cannot
/// be wrong in a way the scan is not already wrong, and there is no second
/// comparison to drift from the evaluator's.
///
/// NULLs are the one thing the negation does not cover: under three-valued
/// logic a NULL row satisfies neither the conjunct nor its negation, so a
/// granule with a NULL in a predicate column is never `All` however tight its
/// bounds are. `min_value`/`max_value` already exclude NULL rows, which is why
/// the bounds cannot be trusted to reveal them.
fn verdict(path: &MetaPath<'_>, p: &Part, gi: usize) -> Verdict {
    let g = &p.granules[gi];
    let mut all = true;
    for mp in &path.preds {
        // A part built before an `ALTER TABLE ... ADD COLUMN` is short. Not a
        // reason to guess; decode it.
        let Some(pc) = g.columns.get(mp.pred.col) else { return Verdict::Mixed };
        let (min, max) = (pc.min_value(), pc.max_value());
        if !mp.pred.may_match(&min, &max) {
            return Verdict::Empty;
        }
        // Kept going rather than returning `Mixed` here: a later conjunct may
        // still prove the granule empty, which is the cheaper answer.
        if all && (mp.not.may_match(&min, &max) || has_nulls(pc, g.len)) {
            all = false;
        }
    }
    if all {
        Verdict::All
    } else {
        Verdict::Mixed
    }
}

/// Does this granule hold a NULL in `pc`?
///
/// `count_ones_upto` is 16 popcounts for a full granule, and it is only
/// reached for a column that has a mask at all -- i.e. a `Nullable` one that
/// really did store a NULL somewhere in the part.
#[inline]
fn has_nulls(pc: &crate::storage::PackedColumn, len: usize) -> bool {
    pc.nulls().is_some_and(|b| b.count_ones_upto(len) > 0)
}

/// The `min`/`max` merge rule, character for character the one `MinMaxAcc`
/// uses in `exec::functions::agg`: strictly better replaces, a tie keeps what
/// was already there. Two rules here would show up as `min` disagreeing with
/// itself depending on which access path the planner chose.
#[inline]
fn offer(slot: &mut Option<Value>, v: Value, max: bool) {
    let better = match slot {
        None => true,
        Some(cur) => {
            let ord = v.cmp(cur);
            ord != std::cmp::Ordering::Equal && (ord == std::cmp::Ordering::Greater) == max
        }
    };
    if better {
        *slot = Some(v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::logical::{BoundExpr, CmpOp, LogicalPlan, ZoneFilter};
    use crate::sql::ast::BinaryOp;
    use crate::types::{Column, DataType, Engine, Field, TableDef, Value};

    // A table of 20 000 rows: `id` runs 0..20000 (so ~20 granules of 1024),
    // `v` is `id * 2`, and `cat` cycles through 8 values.
    fn catalog_with_rows(n: i64) -> Catalog {
        let mut c = Catalog::in_memory();
        c.create_table(
            TableDef {
                name: "t".into(),
                schema: Schema::new(vec![
                    Field::new("id", DataType::UInt64),
                    Field::new("v", DataType::Int64),
                    Field::new("cat", DataType::UInt32),
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
        let t = c.table_by_path_mut("default.t").unwrap();
        t.insert(
            Block::new(vec![
                Column::u64s(DataType::UInt64, (0..n as u64).collect()),
                Column::i64s(DataType::Int64, (0..n).map(|i| i * 2).collect()),
                Column::u64s(DataType::UInt32, (0..n as u64).map(|i| i % 8).collect()),
            ])
            .unwrap(),
        )
        .unwrap();
        t.flush().unwrap();
        c
    }

    fn node(projection: Vec<usize>, cat: &Catalog) -> ScanNode {
        let full = cat.table_by_path("default.t").unwrap().schema().clone();
        ScanNode {
            table: "default.t".into(),
            schema: full.project(&projection),
            projection,
            filters: vec![],
            zone_filters: vec![],
        }
    }

    fn run(node: &ScanNode, cat: &Catalog) -> (Vec<Block>, ScanStats) {
        let plan = LogicalPlan::Scan(Box::new(ScanNode {
            table: node.table.clone(),
            projection: node.projection.clone(),
            schema: node.schema.clone(),
            filters: node.filters.clone(),
            zone_filters: node.zone_filters.clone(),
        }));
        // Build directly so the borrow lives long enough for the assertions.
        let mut op = Scan::new(
            match &plan {
                LogicalPlan::Scan(s) => s,
                _ => unreachable!(),
            },
            cat,
        )
        .unwrap();
        let mut out = Vec::new();
        while let Some(b) = op.next().unwrap() {
            out.push(b);
        }
        let st = op.stats();
        (out, st)
    }

    fn total_rows(bs: &[Block]) -> usize {
        bs.iter().map(|b| b.rows()).sum()
    }

    fn col_ref(i: usize, ty: DataType) -> BoundExpr {
        BoundExpr::Column { index: i, ty, name: format!("c{i}") }
    }

    #[test]
    fn full_scan_returns_every_row_in_order() {
        let c = catalog_with_rows(5_000);
        let (blocks, st) = run(&node(vec![0, 1], &c), &c);
        assert_eq!(total_rows(&blocks), 5_000);
        assert_eq!(st.granules_pruned, 0);
        assert_eq!(st.rows_read, 5_000);
        assert_eq!(blocks[0].column(0).as_u64().unwrap()[0], 0);
        let last = blocks.last().unwrap();
        let ids = last.column(0).as_u64().unwrap();
        assert_eq!(ids[ids.len() - 1], 4_999);
    }

    #[test]
    fn batches_are_block_sized_not_granule_sized() {
        let c = catalog_with_rows(20_000);
        let (blocks, _) = run(&node(vec![0], &c), &c);
        // 20 000 rows at BLOCK_SIZE=8192 -> 8192, 8192, 3616 (granules are
        // 1024, so a granule-per-block scan would emit 20 blocks).
        assert!(blocks.len() <= 4, "got {} blocks", blocks.len());
        assert!(blocks[0].rows() >= BLOCK_SIZE);
    }

    #[test]
    fn projection_reads_only_the_requested_columns() {
        let c = catalog_with_rows(2_000);
        let (blocks, _) = run(&node(vec![2], &c), &c);
        assert_eq!(blocks[0].width(), 1);
        assert_eq!(blocks[0].column(0).value(0), Value::UInt(0));
        assert_eq!(blocks[0].column(0).value(9), Value::UInt(1));
    }

    #[test]
    fn empty_projection_still_reports_the_row_count() {
        // `SELECT count(*)` reads no column data at all.
        let c = catalog_with_rows(3_000);
        let (blocks, st) = run(&node(vec![], &c), &c);
        assert_eq!(total_rows(&blocks), 3_000);
        assert_eq!(blocks[0].width(), 0);
        assert_eq!(st.rows_read, 3_000);
    }

    // ------------------------------------------------------ zone-map pruning

    #[test]
    fn zone_maps_skip_granules_on_a_selective_range() {
        // 20 000 rows of a sorted key -> 20 granules. `id >= 19000` can only
        // live in the last two, so 18 granules must be skipped unread.
        let c = catalog_with_rows(20_000);
        let mut n = node(vec![0, 1], &c);
        n.zone_filters = vec![ZoneFilter { col: 0, op: CmpOp::GtEq, value: Value::UInt(19_000) }];
        n.filters = vec![BoundExpr::Binary {
            left: Box::new(col_ref(0, DataType::UInt64)),
            op: BinaryOp::GtEq,
            right: Box::new(BoundExpr::lit(Value::UInt(19_000))),
            ty: DataType::Bool,
        }];
        let (blocks, st) = run(&n, &c);

        assert_eq!(total_rows(&blocks), 1_000, "the answer must still be right");
        assert!(
            st.granules_pruned >= 17,
            "only {} of 20 granules pruned -- zone maps are not firing",
            st.granules_pruned
        );
        assert_eq!(st.granules_pruned + st.granules_read, 20);
        assert!(
            st.rows_read <= 2_048,
            "{} rows decoded for a 1000-row answer",
            st.rows_read
        );
    }

    #[test]
    fn a_point_zone_filter_prunes_almost_everything() {
        let c = catalog_with_rows(20_000);
        let mut n = node(vec![0], &c);
        n.zone_filters = vec![ZoneFilter { col: 0, op: CmpOp::Eq, value: Value::UInt(5_000) }];
        let (_, st) = run(&n, &c);
        assert_eq!(st.granules_read, 1, "exactly one granule can hold id = 5000");
        assert_eq!(st.granules_pruned, 19);
        assert!(st.prune_ratio() > 0.9);
    }

    #[test]
    fn a_zone_filter_that_matches_nothing_prunes_the_whole_table() {
        let c = catalog_with_rows(20_000);
        let mut n = node(vec![0], &c);
        n.zone_filters = vec![ZoneFilter { col: 0, op: CmpOp::Gt, value: Value::UInt(999_999) }];
        let (blocks, st) = run(&n, &c);
        assert!(blocks.is_empty());
        assert_eq!(st.granules_read, 0);
        assert_eq!(st.rows_read, 0);
        assert_eq!(st.granules_pruned, 20);
    }

    #[test]
    fn zone_filter_columns_are_mapped_through_the_projection() {
        // Project [v, id]: zone filter col 1 means `id`, table column 0. If
        // the mapping were skipped it would test `v` and prune the wrong
        // granules -- `v` spans 0..40000 while `id` spans 0..20000.
        let c = catalog_with_rows(20_000);
        let mut n = node(vec![1, 0], &c);
        n.zone_filters = vec![ZoneFilter { col: 1, op: CmpOp::Lt, value: Value::UInt(1_024) }];
        n.filters = vec![BoundExpr::Binary {
            left: Box::new(col_ref(1, DataType::UInt64)),
            op: BinaryOp::Lt,
            right: Box::new(BoundExpr::lit(Value::UInt(1_024))),
            ty: DataType::Bool,
        }];
        let (blocks, st) = run(&n, &c);
        assert_eq!(total_rows(&blocks), 1_024);
        assert_eq!(st.granules_read, 1, "id < 1024 is exactly granule 0");
        assert_eq!(st.granules_pruned, 19);
        // and the columns really are in projected order
        assert_eq!(blocks[0].column(0).value(1), Value::Int(2)); // v
        assert_eq!(blocks[0].column(1).value(1), Value::UInt(1)); // id
    }

    #[test]
    fn a_non_pruning_zone_filter_reads_everything_and_is_still_correct() {
        // `cat` cycles 0..8 within every granule, so its zone map is useless.
        let c = catalog_with_rows(10_000);
        let mut n = node(vec![2], &c);
        n.zone_filters = vec![ZoneFilter { col: 0, op: CmpOp::Eq, value: Value::UInt(3) }];
        n.filters = vec![BoundExpr::Binary {
            left: Box::new(col_ref(0, DataType::UInt32)),
            op: BinaryOp::Eq,
            right: Box::new(BoundExpr::lit(Value::UInt(3))),
            ty: DataType::Bool,
        }];
        let (blocks, st) = run(&n, &c);
        assert_eq!(st.granules_pruned, 0, "nothing is prunable here");
        assert_eq!(total_rows(&blocks), 10_000 / 8);
    }

    #[test]
    fn zone_pruning_never_drops_a_matching_row() {
        // Sweep thresholds across granule boundaries and compare pruned scans
        // against an unpruned reference.
        let c = catalog_with_rows(8_000);
        for t in [0u64, 1, 1_023, 1_024, 1_025, 4_096, 7_999, 8_000] {
            let mut pruned = node(vec![0], &c);
            let pred = BoundExpr::Binary {
                left: Box::new(col_ref(0, DataType::UInt64)),
                op: BinaryOp::GtEq,
                right: Box::new(BoundExpr::lit(Value::UInt(t))),
                ty: DataType::Bool,
            };
            pruned.filters = vec![pred.clone()];
            pruned.zone_filters =
                vec![ZoneFilter { col: 0, op: CmpOp::GtEq, value: Value::UInt(t) }];

            let mut plain = node(vec![0], &c);
            plain.filters = vec![pred];

            let (a, _) = run(&pruned, &c);
            let (b, _) = run(&plain, &c);
            assert_eq!(total_rows(&a), total_rows(&b), "threshold {t}");
            assert_eq!(total_rows(&a), 8_000usize.saturating_sub(t as usize), "threshold {t}");
        }
    }

    // ------------------------------------------------------------- prewhere

    #[test]
    fn prewhere_filters_before_anything_upstream_sees_a_row() {
        let c = catalog_with_rows(5_000);
        let mut n = node(vec![0, 1], &c);
        n.filters = vec![BoundExpr::Binary {
            left: Box::new(col_ref(0, DataType::UInt64)),
            op: BinaryOp::Lt,
            right: Box::new(BoundExpr::lit(Value::UInt(10))),
            ty: DataType::Bool,
        }];
        let (blocks, st) = run(&n, &c);
        assert_eq!(total_rows(&blocks), 10);
        assert_eq!(st.rows_read, 5_000, "no zone filter, so every row is decoded");
    }

    #[test]
    fn multiple_prewhere_conjuncts_all_apply() {
        let c = catalog_with_rows(1_000);
        let mut n = node(vec![0, 2], &c);
        let gt = BoundExpr::Binary {
            left: Box::new(col_ref(0, DataType::UInt64)),
            op: BinaryOp::GtEq,
            right: Box::new(BoundExpr::lit(Value::UInt(100))),
            ty: DataType::Bool,
        };
        let cat = BoundExpr::Binary {
            left: Box::new(col_ref(1, DataType::UInt32)),
            op: BinaryOp::Eq,
            right: Box::new(BoundExpr::lit(Value::UInt(0))),
            ty: DataType::Bool,
        };
        n.filters = vec![gt, cat];
        let (blocks, _) = run(&n, &c);
        // ids 104, 112, ... 992 -> multiples of 8 that are >= 100
        assert_eq!(total_rows(&blocks), (100..1_000).filter(|i| i % 8 == 0).count());
    }

    #[test]
    fn deleted_rows_are_skipped() {
        let mut c = catalog_with_rows(2_000);
        {
            let t = c.table_by_path_mut("default.t").unwrap();
            t.delete_key(&Value::UInt(5)).unwrap();
            t.delete_key(&Value::UInt(6)).unwrap();
            t.flush().unwrap();
        }
        let (blocks, st) = run(&node(vec![0], &c), &c);
        assert_eq!(total_rows(&blocks), 1_998);
        assert_eq!(st.rows_read, 1_998, "deleted rows are never decoded");
        let ids: Vec<u64> = blocks[0].column(0).as_u64().unwrap()[..8].to_vec();
        assert_eq!(ids, vec![0, 1, 2, 3, 4, 7, 8, 9]);
    }

    #[test]
    fn scanning_an_empty_table_yields_nothing() {
        let c = catalog_with_rows(0);
        let (blocks, st) = run(&node(vec![0], &c), &c);
        assert!(blocks.is_empty());
        assert_eq!(st, ScanStats::default());
    }

    #[test]
    fn unknown_table_is_an_error() {
        let c = catalog_with_rows(1);
        let n = ScanNode {
            table: "default.nope".into(),
            projection: vec![0],
            schema: Schema::empty(),
            filters: vec![],
            zone_filters: vec![],
        };
        assert!(Scan::new(&n, &c).is_err());
    }

    #[test]
    fn out_of_range_projection_is_an_error() {
        let c = catalog_with_rows(1);
        let mut n = node(vec![0], &c);
        n.projection = vec![99];
        assert!(Scan::new(&n, &c).is_err());
    }

    #[test]
    fn multiple_parts_are_all_visited() {
        let mut c = catalog_with_rows(1_000);
        {
            let t = c.table_by_path_mut("default.t").unwrap();
            t.insert(
                Block::new(vec![
                    Column::u64s(DataType::UInt64, (5_000..5_500).collect()),
                    Column::i64s(DataType::Int64, (5_000..5_500).map(|i| i * 2).collect()),
                    Column::u64s(DataType::UInt32, vec![1; 500]),
                ])
                .unwrap(),
            )
            .unwrap();
            t.flush().unwrap();
            assert!(t.part_count() >= 2);
        }
        let (blocks, _) = run(&node(vec![0], &c), &c);
        assert_eq!(total_rows(&blocks), 1_500);
    }

    // --------------------------------------------------------- index lookup
    //
    // Every one of these asserts the index path against the *scan* path over
    // the same node, not against a hand-written expectation. That is the only
    // comparison that matters: the two operators answer the same scan node, and
    // an access-path decision is only correct if nobody upstream can tell which
    // one ran.

    /// The rows a scan node produces, as `Value` tuples in emission order.
    fn tuples(bs: &[Block]) -> Vec<Vec<Value>> {
        bs.iter()
            .flat_map(|b| (0..b.rows()).map(move |r| (0..b.width()).map(|c| b.column(c).value(r)).collect()))
            .collect()
    }

    /// Run `node` through the index path. Panics if the planner declined it,
    /// which is what makes "and it really did use the index" part of the test.
    fn run_index(node: &ScanNode, cat: &Catalog) -> (Vec<Block>, ScanStats) {
        let plan = LogicalPlan::Scan(Box::new(ScanNode {
            table: node.table.clone(),
            projection: node.projection.clone(),
            schema: node.schema.clone(),
            filters: node.filters.clone(),
            zone_filters: node.zone_filters.clone(),
        }));
        let path = match crate::planner::physical::lower(&plan, cat).unwrap() {
            crate::planner::PhysicalPlan::IndexLookup(p) => *p,
            _ => panic!("the planner declined the index for this node"),
        };
        let mut op = IndexLookup::new(path, cat).unwrap();
        let mut out = Vec::new();
        while let Some(b) = op.next().unwrap() {
            out.push(b);
        }
        let st = Operator::stats(&op);
        (out, st)
    }

    /// Drive `op` to exhaustion and check it against the scan over the same node.
    fn assert_agrees_with_scan(
        mut op: IndexLookup<'_>,
        node: &ScanNode,
        cat: &Catalog,
    ) -> Vec<Vec<Value>> {
        let mut idx = Vec::new();
        while let Some(b) = op.next().unwrap() {
            assert!(b.rows() > 0, "an empty block is not end-of-stream");
            idx.push(b);
        }
        let (scanned, _) = run(node, cat);
        let (a, b) = (tuples(&idx), tuples(&scanned));
        assert_eq!(a, b, "index and scan disagree");
        for blk in &idx {
            assert_eq!(blk.width(), node.schema.len(), "wrong output width");
            for (c, f) in blk.columns.iter().zip(node.schema.fields()) {
                assert_eq!(c.ty.physical(), f.ty.physical(), "column `{}` retyped", f.name);
            }
        }
        a
    }

    /// The two access paths must agree, row for row and in the same order --
    /// with the path chosen by the planner, so "and it really did use the
    /// index" is part of the assertion.
    fn assert_paths_agree(node: &ScanNode, cat: &Catalog) -> Vec<Vec<Value>> {
        let plan = LogicalPlan::Scan(Box::new(ScanNode {
            table: node.table.clone(),
            projection: node.projection.clone(),
            schema: node.schema.clone(),
            filters: node.filters.clone(),
            zone_filters: node.zone_filters.clone(),
        }));
        let path = match crate::planner::physical::lower(&plan, cat).unwrap() {
            crate::planner::PhysicalPlan::IndexLookup(p) => *p,
            _ => panic!("the planner declined the index for this node"),
        };
        // The borrowed `ScanNode` is the local `plan`'s, which lives as long as
        // this frame; the caller's `node` is an identical copy.
        assert_agrees_with_scan(IndexLookup::new(path, cat).unwrap(), node, cat)
    }

    /// As above but with the key list supplied, so a shape the planner declines
    /// *on cost* still exercises the operator. Correctness does not depend on
    /// the cost gate, and a fuzz that only ran the shapes the gate happens to
    /// admit would leave small tables untested.
    fn assert_forced_index_agrees(
        node: &ScanNode,
        cat: &Catalog,
        ty: &DataType,
        keys: &[Value],
    ) {
        let mut lanes: Vec<u64> = keys.iter().filter_map(|v| v.to_lane(ty).ok()).collect();
        lanes.sort_unstable();
        lanes.dedup();
        let path = IndexPath { node, key_col: 0, key_field: 0, keys: lanes };
        assert_agrees_with_scan(IndexLookup::new(path, cat).unwrap(), node, cat);
    }

    fn eq_pred(col: usize, ty: DataType, v: Value) -> BoundExpr {
        BoundExpr::Binary {
            left: Box::new(col_ref(col, ty)),
            op: BinaryOp::Eq,
            right: Box::new(BoundExpr::lit(v)),
            ty: DataType::Bool,
        }
    }

    /// `node` projecting `[id, v]` with `WHERE id = k` pushed into the scan.
    fn point(cat: &Catalog, k: u64) -> ScanNode {
        let mut n = node(vec![0, 1], cat);
        n.filters = vec![eq_pred(0, DataType::UInt64, Value::UInt(k))];
        n.zone_filters = vec![ZoneFilter { col: 0, op: CmpOp::Eq, value: Value::UInt(k) }];
        n
    }

    #[test]
    fn a_point_lookup_returns_the_row_the_scan_would_have() {
        let c = catalog_with_rows(20_000);
        let rows = assert_paths_agree(&point(&c, 12_345), &c);
        assert_eq!(rows, vec![vec![Value::UInt(12_345), Value::Int(24_690)]]);
    }

    #[test]
    fn a_key_that_is_not_there_yields_nothing() {
        let c = catalog_with_rows(20_000);
        assert!(assert_paths_agree(&point(&c, 20_000), &c).is_empty());
        assert!(assert_paths_agree(&point(&c, u64::MAX), &c).is_empty());
        // ... and neither does an empty table.
        let e = catalog_with_rows(0);
        assert!(assert_paths_agree(&point(&e, 0), &e).is_empty());
    }

    #[test]
    fn the_lookup_reads_one_granule_and_counts_the_rest_as_skipped() {
        let c = catalog_with_rows(20_000);
        let (_, st) = run_index(&point(&c, 12_345), &c);
        assert_eq!(st.granules_read, 1, "one key can only live in one granule");
        assert_eq!(st.rows_read, 1, "and one row is decoded, not 1024");
        // The invariant the scan's own tests assert has to keep holding.
        assert_eq!(st.granules_pruned + st.granules_read, 20);
        assert!(st.prune_ratio() > 0.9);
    }

    #[test]
    fn a_deleted_row_is_not_resurrected_by_the_index() {
        let mut c = catalog_with_rows(20_000);
        {
            let t = c.table_by_path_mut("default.t").unwrap();
            t.delete_key(&Value::UInt(12_345)).unwrap();
            t.flush().unwrap();
        }
        assert!(assert_paths_agree(&point(&c, 12_345), &c).is_empty());
        // its neighbours are untouched
        assert_eq!(assert_paths_agree(&point(&c, 12_344), &c).len(), 1);
        assert_eq!(assert_paths_agree(&point(&c, 12_346), &c).len(), 1);
    }

    #[test]
    fn a_key_shadowed_by_a_newer_part_reads_the_new_row() {
        // Two parts, the second re-inserting a key the first holds. The ingest
        // tombstones the old row, so exactly one is live and both paths have to
        // find that one.
        let mut c = catalog_with_rows(20_000);
        {
            let t = c.table_by_path_mut("default.t").unwrap();
            t.insert(
                Block::new(vec![
                    Column::u64s(DataType::UInt64, vec![12_345]),
                    Column::i64s(DataType::Int64, vec![-7]),
                    Column::u64s(DataType::UInt32, vec![3]),
                ])
                .unwrap(),
            )
            .unwrap();
            t.flush().unwrap();
            assert!(t.part_count() >= 2, "the test needs two parts");
        }
        assert_eq!(
            assert_paths_agree(&point(&c, 12_345), &c),
            vec![vec![Value::UInt(12_345), Value::Int(-7)]]
        );
    }

    /// `default.t` holding `n` rows whose key repeats every `rep` rows -- `id =
    /// i / rep`, `v = i`, `cat = 0` -- written as one bulk insert (>= 4096
    /// rows) that is *already in key order*, which is the shape whose dedup
    /// used to be skipped. `keyed` declares the PRIMARY KEY; without it the
    /// table has a sort key and nothing else.
    fn catalog_of_runs(n: u64, rep: u64, keyed: bool) -> Catalog {
        let mut c = Catalog::in_memory();
        c.create_table(
            TableDef {
                name: "t".into(),
                schema: Schema::new(vec![
                    Field::new("id", DataType::UInt64),
                    Field::new("v", DataType::Int64),
                    Field::new("cat", DataType::UInt32),
                ])
                .unwrap(),
                order_by: vec![0],
                primary_key: if keyed { vec![0] } else { vec![] },
                partition_by: None,
                engine: Engine::MergeTree,
            },
            false,
        )
        .unwrap();
        let t = c.table_by_path_mut("default.t").unwrap();
        t.insert(
            Block::new(vec![
                Column::u64s(DataType::UInt64, (0..n).map(|i| i / rep).collect()),
                Column::i64s(DataType::Int64, (0..n as i64).collect()),
                Column::u64s(DataType::UInt32, vec![0; n as usize]),
            ])
            .unwrap(),
        )
        .unwrap();
        t.flush().unwrap();
        c
    }

    #[test]
    fn a_declared_primary_key_collapses_a_run_of_duplicates() {
        // Inverted from `a_run_of_duplicate_keys_is_returned_whole`, which
        // pinned the bug rather than the fix: `ingest_block` dedup'd only when
        // the batch needed sorting, so a bulk insert already in key order
        // carried duplicates into a part whose MPH index is defined on
        // *distinct* keys alone -- `WHERE id = k` answered one way through the
        // index and another through a scan. Duplicates and the point-lookup
        // index are mutually exclusive by construction, so there is exactly one
        // reconciliation: the run collapses to its last row, which is already
        // what a key re-inserted by a later statement gets.
        //
        // 12 288 is divisible by both run lengths, so every run is whole and
        // the last key is not a special case.
        let n = 12_288u64;
        for rep in [2u64, 3] {
            let c = catalog_of_runs(n, rep, true);
            let (all, _) = run(&node(vec![0], &c), &c);
            assert_eq!(total_rows(&all) as u64, n / rep, "a duplicate survived rep {rep}");
            // Granules are 1024 rows, so an input row at a multiple of 1024 is
            // a seam: at rep 2 key 511 is input rows 1023 and 1024, at rep 3
            // key 341 is 1023..=1025. Both runs straddle it, and the last key
            // is the one whose run ends at the end of the batch.
            for k in [0u64, 1, 341, 511, 512, n / rep - 1] {
                assert_eq!(
                    assert_paths_agree(&point(&c, k), &c),
                    vec![vec![Value::UInt(k), Value::Int((k * rep + rep - 1) as i64)]],
                    "key {k} of a rep-{rep} run"
                );
            }
        }
    }

    #[test]
    fn a_sort_key_without_a_primary_key_keeps_every_copy() {
        // The other half of that fix, and the reason it is not simply "ingest
        // dedups": collapsing is the *unique-key* machinery, so a table that
        // only declares ORDER BY still holds its repeats and the scan has to
        // return all of them, in storage order. There is no index path to
        // agree with here -- the MPH is built over a primary key this table
        // does not have -- which is exactly why both tables are needed.
        let n = 12_288u64;
        let c = catalog_of_runs(n, 2, false);
        let (all, _) = run(&node(vec![0], &c), &c);
        assert_eq!(total_rows(&all) as u64, n, "an unkeyed table dropped a row");
        for k in [0u64, 511, 512, n / 2 - 1] {
            let (blocks, _) = run(&point(&c, k), &c);
            assert_eq!(
                tuples(&blocks),
                vec![
                    vec![Value::UInt(k), Value::Int(k as i64 * 2)],
                    vec![Value::UInt(k), Value::Int(k as i64 * 2 + 1)],
                ],
                "key {k} lost a duplicate"
            );
        }
    }

    #[test]
    fn an_in_list_returns_scan_order_across_parts_and_granules() {
        let mut c = catalog_with_rows(20_000);
        {
            let t = c.table_by_path_mut("default.t").unwrap();
            t.insert(
                Block::new(vec![
                    Column::u64s(DataType::UInt64, (30_000..30_100).collect()),
                    Column::i64s(DataType::Int64, (30_000..30_100).map(|i| i * 2).collect()),
                    Column::u64s(DataType::UInt32, vec![5; 100]),
                ])
                .unwrap(),
            )
            .unwrap();
            t.flush().unwrap();
        }
        let mut n = node(vec![0, 1], &c);
        // Deliberately out of order and with a duplicate and a miss in it.
        let list = vec![
            Value::UInt(30_050),
            Value::UInt(7),
            Value::UInt(99_999),
            Value::UInt(19_999),
            Value::UInt(7),
            Value::UInt(1_024),
        ];
        n.filters = vec![BoundExpr::InList {
            expr: Box::new(col_ref(0, DataType::UInt64)),
            list,
            negated: false,
        }];
        let got = assert_paths_agree(&n, &c);
        let ids: Vec<Value> = got.iter().map(|r| r[0].clone()).collect();
        assert_eq!(
            ids,
            vec![Value::UInt(7), Value::UInt(1_024), Value::UInt(19_999), Value::UInt(30_050)],
            "rows must come out in storage order, deduplicated"
        );
    }

    #[test]
    fn residual_conjuncts_still_reject_the_row() {
        let c = catalog_with_rows(20_000);
        let mut n = node(vec![0, 2], &c);
        n.filters = vec![
            eq_pred(0, DataType::UInt64, Value::UInt(12_345)),
            // 12345 % 8 == 1, so `cat = 2` must reject it and `cat = 1` must not.
            eq_pred(1, DataType::UInt32, Value::UInt(2)),
        ];
        assert!(assert_paths_agree(&n, &c).is_empty());

        n.filters[1] = eq_pred(1, DataType::UInt32, Value::UInt(1));
        assert_eq!(assert_paths_agree(&n, &c).len(), 1);
    }

    #[test]
    fn a_lookup_that_fills_a_block_chunks_like_a_scan() {
        // BLOCK_SIZE + 1 keys spanning nine granules: the operator must hand
        // out one full block and a remainder, never a block per granule.
        //
        // The path is built directly rather than through `lower`, which would
        // decline 8193 probes against a 200k-row table (see
        // `a_huge_in_list_is_cheaper_to_scan` in planner::physical) -- and
        // rightly so. What is under test here is the operator's batching, not
        // the planner's arithmetic.
        let c = catalog_with_rows(200_000);
        let n = node(vec![0, 1], &c);
        let path = IndexPath {
            node: &n,
            key_col: 0,
            key_field: 0,
            // For a `UInt64` key the lane is the value.
            keys: (0..BLOCK_SIZE as u64 + 1).collect(),
        };
        let mut op = IndexLookup::new(path, &c).unwrap();
        let mut blocks = Vec::new();
        while let Some(b) = op.next().unwrap() {
            blocks.push(b);
        }
        assert_eq!(blocks.len(), 2, "got {} blocks", blocks.len());
        assert_eq!(blocks[0].rows(), BLOCK_SIZE);
        assert_eq!(blocks[1].rows(), 1);
        let st = Operator::stats(&op);
        assert_eq!(st.rows_read, BLOCK_SIZE as u64 + 1);
        assert_eq!(st.granules_read, 9, "8193 consecutive keys span 9 granules");
        let ids: Vec<Value> = tuples(&blocks).into_iter().map(|r| r[0].clone()).collect();
        assert_eq!(ids, (0..BLOCK_SIZE as u64 + 1).map(Value::UInt).collect::<Vec<_>>());
    }

    #[test]
    fn a_signed_key_is_probed_through_its_order_preserving_lane() {
        // Signed lanes are sign-flipped, not raw: a negative key that reached
        // `find` unflipped would route to the wrong end of the part.
        let mut c = Catalog::in_memory();
        c.create_table(
            TableDef {
                name: "s".into(),
                schema: Schema::new(vec![
                    Field::new("id", DataType::Int64),
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
        {
            let t = c.table_by_path_mut("default.s").unwrap();
            t.insert(
                Block::new(vec![
                    Column::i64s(DataType::Int64, (-5_000..5_000).collect()),
                    Column::i64s(DataType::Int64, (-5_000..5_000).map(|i| i * 10).collect()),
                ])
                .unwrap(),
            )
            .unwrap();
            t.flush().unwrap();
        }
        let full = c.table_by_path("default.s").unwrap().schema().clone();
        for k in [-5_000i64, -1, 0, 1, 4_999] {
            let n = ScanNode {
                table: "default.s".into(),
                projection: vec![0, 1],
                schema: full.clone(),
                filters: vec![eq_pred(0, DataType::Int64, Value::Int(k))],
                zone_filters: vec![],
            };
            assert_eq!(
                assert_paths_agree(&n, &c),
                vec![vec![Value::Int(k), Value::Int(k * 10)]],
                "key {k}"
            );
        }
    }

    /// Index path vs. scan path over random tables and random keys.
    ///
    /// The SQLite oracle in `tests/differential.rs` validates the *scan*, but it
    /// reaches an index lookup only about once in 300 generated cases: the
    /// predicate has to be an equality, on `id` specifically, against a literal,
    /// on a table that declared `ORDER BY id`. That is thin coverage for the
    /// riskiest change in the engine, so this closes the gap from the other
    /// side -- if `scan == sqlite` and `index == scan`, then `index == sqlite`.
    ///
    /// The shapes here are chosen for the ways an index can go wrong rather
    /// than for variety: runs of duplicate keys that straddle a granule seam,
    /// tombstones sitting inside such a run, a key live in an older part, keys
    /// either side of every boundary, and signed and float lanes whose encoding
    /// is not the identity.
    #[test]
    fn index_and_scan_agree_over_random_tables_and_keys() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut rng = move || {
            seed = crate::common::splitmix64(seed);
            seed
        };

        // (rows, key type, duplicated keys, extra part, deletes)
        for (rows, ty, dup, extra_part, ndel) in [
            (0usize, DataType::UInt64, false, false, 0usize),
            (1, DataType::UInt64, false, false, 0),
            (7, DataType::UInt64, false, false, 3),
            (1_024, DataType::UInt64, false, false, 0),
            (1_025, DataType::UInt64, false, true, 5),
            (2_500, DataType::Int64, false, false, 40),
            (4_096, DataType::UInt64, true, false, 0),
            (5_000, DataType::UInt64, true, false, 60),
            (8_193, DataType::Int64, false, true, 100),
            (3_000, DataType::Float64, false, false, 20),
            (4_096, DataType::Int64, true, true, 30),
        ] {
            let mut c = keyed_catalog(rows, &ty, dup);
            if extra_part {
                let t = c.table_by_path_mut("default.k").unwrap();
                // Half re-inserts (shadowing), half brand-new keys.
                let ids: Vec<i64> = (0..64).map(|i| if i % 2 == 0 { i } else { 100_000 + i }).collect();
                t.insert(key_block(&ids, &ty)).unwrap();
                t.flush().unwrap();
            }
            for _ in 0..ndel {
                let k = (rng() % (rows.max(1) as u64 + 8)) as i64 - 2;
                let t = c.table_by_path_mut("default.k").unwrap();
                let _ = t.delete_key(&key_value(k, &ty));
            }
            c.table_by_path_mut("default.k").unwrap().flush().unwrap();

            let full = c.table_by_path("default.k").unwrap().schema().clone();
            let mk = |filters: Vec<BoundExpr>| ScanNode {
                table: "default.k".into(),
                projection: vec![0, 1],
                schema: full.clone(),
                filters,
                zone_filters: vec![],
            };
            let lit = |k: i64| BoundExpr::lit(key_value(k, &ty));

            for _ in 0..120 {
                // Keys are drawn from a range wider than the table so misses,
                // boundary hits and out-of-range probes all occur.
                let span = rows.max(1) as u64 + 16;
                let k = (rng() % span) as i64 - 4;
                // A single key is never refused on cost, so this arm also
                // asserts the planner keeps choosing the index.
                let node = mk(vec![BoundExpr::Binary {
                    left: Box::new(col_ref(0, ty.clone())),
                    op: BinaryOp::Eq,
                    right: Box::new(lit(k)),
                    ty: DataType::Bool,
                }]);
                assert_paths_agree(&node, &c);

                let n = 1 + (rng() % 5) as usize;
                let list: Vec<Value> = (0..n)
                    .map(|_| key_value((rng() % span) as i64 - 4, &ty))
                    .collect();
                let node = mk(vec![BoundExpr::InList {
                    expr: Box::new(col_ref(0, ty.clone())),
                    list: list.clone(),
                    negated: false,
                }]);
                assert_forced_index_agrees(&node, &c, &ty, &list);
            }
        }
    }

    /// `k(id <ty> PRIMARY KEY, v Int64)`. With `dup`, every key appears twice
    /// via a bulk already-sorted insert, which is the one path that carries
    /// duplicates into a part (`ingest_block` only dedups a batch it had to
    /// sort).
    fn keyed_catalog(rows: usize, ty: &DataType, dup: bool) -> Catalog {
        let mut c = Catalog::in_memory();
        c.create_table(
            TableDef {
                name: "k".into(),
                schema: Schema::new(vec![
                    Field::new("id", ty.clone()),
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
        if rows > 0 {
            let ids: Vec<i64> =
                (0..rows as i64).map(|i| if dup { i / 2 } else { i }).collect();
            c.table_by_path_mut("default.k").unwrap().insert(key_block(&ids, ty)).unwrap();
            c.table_by_path_mut("default.k").unwrap().flush().unwrap();
        }
        c
    }

    fn key_value(k: i64, ty: &DataType) -> Value {
        match ty {
            DataType::Int64 => Value::Int(k - 1_000),
            DataType::Float64 => Value::Float(k as f64 / 2.0),
            _ => Value::UInt(k.max(0) as u64),
        }
    }

    fn key_block(ids: &[i64], ty: &DataType) -> Block {
        let key = match ty {
            DataType::Int64 => Column::i64s(ty.clone(), ids.iter().map(|&i| i - 1_000).collect()),
            DataType::Float64 => Column::new(
                ty.clone(),
                crate::types::ColumnData::F64(ids.iter().map(|&i| i as f64 / 2.0).collect()),
            ),
            _ => Column::u64s(ty.clone(), ids.iter().map(|&i| i.max(0) as u64).collect()),
        };
        Block::new(vec![key, Column::i64s(DataType::Int64, ids.iter().map(|&i| i * 7).collect())])
            .unwrap()
    }

    #[test]
    fn a_path_naming_the_wrong_key_column_is_an_error_not_an_empty_answer() {
        // `IndexPath` is public, so nothing stops a caller building one that
        // names a column the parts are not indexed on. Every probe would then
        // miss and the query would quietly return no rows. The constructor has
        // to refuse it -- there is no scan left to fall back to at that point.
        let c = catalog_with_rows(2_000);
        let n = node(vec![0, 1], &c);
        let path = IndexPath { node: &n, key_col: 1, key_field: 1, keys: vec![5] };
        let e = match IndexLookup::new(path, &c) {
            Err(e) => e,
            Ok(_) => panic!("a path keyed on the wrong column was accepted"),
        };
        assert!(e.to_string().contains("index lookup"), "{e}");
        // ... and the right column still builds.
        assert!(
            IndexLookup::new(
                IndexPath { node: &n, key_col: 0, key_field: 0, keys: vec![5] },
                &c
            )
            .is_ok(),
            "the correctly-keyed path must still construct"
        );
    }

    #[test]
    fn run_walking_steps_across_granule_seams_and_partial_tails() {
        // `next_row`/`prev_row` encode the granule-major layout, including the
        // hole at the end of a partial final granule. Getting either wrong
        // either walks off the end or stops the run early.
        let c = catalog_with_rows(2_500); // 2 full granules + a 452-row tail
        let t = c.table_by_path("default.t").unwrap();
        let snap = t.snapshot();
        let p = snap.part(0);
        assert_eq!(p.granule_count(), 3);

        // Forward from the last row of granule 0 lands on the first of granule 1.
        assert_eq!(next_row(p, GRANULE_SIZE - 1), Some(GRANULE_SIZE));
        assert_eq!(prev_row(p, GRANULE_SIZE), Some(GRANULE_SIZE - 1));
        // The tail granule holds 2500 - 2048 = 452 rows; there is nothing after.
        let last = (2 << G_SHIFT) + 451;
        assert_eq!(next_row(p, last), None);
        assert_eq!(prev_row(p, 0), None);
        // Every step is a live row, and there are exactly as many as the table.
        let mut n = 1;
        let mut pos = 0;
        while let Some(q) = next_row(p, pos) {
            pos = q;
            n += 1;
        }
        assert_eq!(n, 2_500);
    }
}
