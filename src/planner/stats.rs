//! Statistics and the cost model.
//!
//! Until this file existed the planner's entire vocabulary for "how big is
//! this" was `join::SideFacts`, whose own doc comment says it: *"There is no
//! cost model and no statistics in this engine beyond how many live rows does
//! this table have"*. That is enough to pick a join **algorithm** and not
//! enough to pick a join **order**, so join order was FROM-clause order.
//!
//! # Where the numbers come from, and when
//!
//! Four designs were available and only one of them is free:
//!
//! * **maintained on write** -- always fresh, and a tax on the hot path of an
//!   engine whose ingest is measured in bytes per row;
//! * **stored in the part at build time** -- free at query time, and it needs
//!   a part-format version bump plus a migration for every part already on
//!   disk. That is the right *eventual* home for an NDV sketch (see below);
//! * **explicit `ANALYZE`** -- user-controlled, and wrong exactly when it
//!   matters, because the statistic a user forgot to refresh is the one that
//!   plans the query badly;
//! * **derived from part metadata at plan time** -- which is what this does.
//!
//! The fourth wins here because of what a part already is. A part is
//! **immutable** and already records, with no extra byte stored:
//!
//!   * `n_rows` per part and a tombstone count per part set, so an exact live
//!     row count is a fold over ~16 integers. This is the same fact that makes
//!     `SELECT count()` a microsecond operation;
//!   * frame-of-reference `base` and `max_lane` per column per granule -- the
//!     zone maps -- so a column's exact `[min, max]` is one `u64` compare per
//!     granule with **no decode and no page touched**, because both are
//!     already in the granule header.
//!
//! So the statistics are never stale, never missing after a restart, and cost
//! nothing to maintain. What they cannot give is a true distinct count; see
//! [`Facts::span`] for what is estimated instead and how wrong it can be.
//!
//! # What is exact and what is a guess
//!
//! Exact, and marked so in this file:
//!
//!   * relation row counts, for any base table;
//!   * `[min, max]` of any numeric, date, datetime or decimal column;
//!   * NDV of a single-column primary key -- unique by construction, so
//!     `ndv = rows`. This is the one that matters, because it is the dimension
//!     side of every star and chain join.
//!
//! Guesses, all of them flagged at their definition:
//!
//!   * NDV of a non-key column, estimated as `min(rows, max - min + 1)`. Exact
//!     for a dense surrogate key -- which is what a foreign key *is* -- and an
//!     over-estimate for a sparse one, which makes a join look smaller than it
//!     is. [`Facts::ndv`] says what that does to a plan;
//!   * NDV of a string or float column: refused outright rather than guessed.
//!     A per-granule string dictionary is order-preserving *within* the
//!     granule and its codes are not comparable across granules, so there is
//!     no cross-granule span to take;
//!   * predicate selectivity, from [`SEL_EQ`] downward;
//!   * the independence assumption. `WHERE country = 'FR' AND lang = 'fr'`
//!     is estimated as the product of two selectivities and is off by roughly
//!     the correlation. Correlated predicates are where cost models go wrong
//!     and this one is no exception; it is written down rather than hidden.
//!
//! # Cost units
//!
//! One unit is **one row read out of a part and materialized into a block**.
//! Everything else is expressed against that, from measurements already in
//! this repository rather than from first principles:
//!
//! ```text
//!   SCAN         1.0   the unit
//!   OUT          2.0   gather + assemble one output row (join.rs: `assemble`
//!                      measures the typed-copy path at ~1.4x the scan it
//!                      replaced end to end)
//!   PROBE        4.0   hash one key and walk one bucket
//!   BUILD        8.0   hash one key, insert, and pay the table's growth
//! ```
//!
//! Only ratios matter: every candidate order is costed by the same function,
//! so a uniform scale error cancels out of every comparison the search makes.
//! A fifth constant for the primary-key fetch was written and then removed;
//! [`join_cost`] records why, and what to restore when it becomes true.
//!
//! # What is deliberately not here
//!
//! No histogram, no sample, no sketch, no persisted state, no cache that
//! outlives one call to the optimizer. An [`Estimator`] borrows the catalog,
//! answers questions, and is dropped; two queries never share a statistic, so
//! there is no invalidation to get wrong. The whole file allocates a handful
//! of `Vec`s per *query*, never per row and never per granule.

use crate::catalog::Catalog;
use crate::sql::ast::{BinaryOp, JoinOp, SetOp};
use crate::types::{DataType, PhysicalType, Value};

use crate::planner::logical::{BoundExpr, LogicalPlan, ScanNode};

// ------------------------------------------------------------ cost constants

/// Read one row out of a part into a block. The unit everything else is in.
pub const SCAN: f64 = 1.0;
/// Gather and assemble one row of a join's output.
pub const OUT: f64 = 2.0;
/// Hash one key and walk one hash bucket.
pub const PROBE: f64 = 4.0;
/// Hash one key and insert it, amortizing the table's growth.
pub const BUILD: f64 = 8.0;
// ------------------------------------------------------- selectivity defaults

/// `col = <literal>` on a column with no usable NDV.
///
/// The textbook 0.1. Every constant in this block is a guess, and each one is
/// used only when the exact path above it declined.
pub const SEL_EQ: f64 = 0.1;
/// `col < / <= / > / >= <literal>` with no usable `[min, max]`.
pub const SEL_RANGE: f64 = 0.33;
/// `IS NULL`. Its complement covers `IS NOT NULL`.
pub const SEL_NULL: f64 = 0.1;
/// `LIKE`, `NOT LIKE`, and any predicate whose shape this file does not read.
pub const SEL_OTHER: f64 = 0.25;

/// Nothing is ever estimated at zero rows.
///
/// A zero would make one join order infinitely better than every other and
/// would propagate a zero through every product above it. The estimate is a
/// ranking input, not a promise, so it is clamped to one row.
const MIN_ROWS: f64 = 1.0;

/// How deep [`Estimator::rows`] will walk before giving up.
///
/// The optimizer's own `MAX_PLAN_DEPTH` is 200 and it refuses anything deeper
/// before this file is ever called; this is the same backstop one level down,
/// so a hand-built plan cannot recurse here without a bound.
const MAX_DEPTH: usize = 200;

// ------------------------------------------------------------------- estimator

/// Everything known about one base table, resolved once per query.
struct Facts {
    /// Catalog path, as it appears in [`ScanNode::table`].
    path: String,
    /// Exact live rows: `Σ n_rows - tombstones`, from part metadata.
    rows: f64,
    /// Single-column primary key, as a **table** column index. `Some` only for
    /// a unique key that is also the sort key -- which is what makes
    /// `ndv = rows` a fact rather than an assumption.
    pk: Option<usize>,
    /// `[min, max]` **lanes** per table column, folded on first use.
    ///
    /// Lane order is value order for every physical type here
    /// (`common::lane`: signed integers are offset-binary, floats are the
    /// standard order-preserving bit twiddle), so a span can be taken without
    /// decoding anything.
    spans: Vec<Span>,
}

/// One column's zone-map fold: not yet asked, asked and refused, or known.
#[derive(Clone, Copy)]
enum Span {
    Unasked,
    Unknown,
    Known(u64, u64),
}

/// The planner's view of the data, for one call to the optimizer.
///
/// Borrows the catalog and owns a per-query cache. Deliberately not `Clone`
/// and deliberately not stored anywhere: a statistic that outlives the query
/// that computed it is a statistic somebody has to invalidate.
pub struct Estimator<'c> {
    catalog: &'c Catalog,
    /// Interned by table path. Linear scan: a query joins a handful of
    /// relations, and a `HashMap` here would cost more than it saves.
    tables: Vec<Facts>,
}

impl<'c> Estimator<'c> {
    pub fn new(catalog: &'c Catalog) -> Estimator<'c> {
        Estimator { catalog, tables: Vec::new() }
    }

    /// Facts for a table, resolving and caching on first mention.
    ///
    /// `None` for a table the catalog cannot resolve -- a system table, a
    /// table dropped between binding and here, a name a test built by hand.
    /// Every caller treats that as "no statistics", which is what makes an
    /// absent statistic degrade to today's behaviour instead of to a guess.
    fn facts(&mut self, path: &str) -> Option<&mut Facts> {
        if let Some(i) = self.tables.iter().position(|f| f.path == path) {
            return Some(&mut self.tables[i]);
        }
        let table = self.catalog.table_by_path(path).ok()?;
        // Rows buffered in the write delta are *not* counted, and must not be:
        // every read path flushes before it plans, so at plan time the parts
        // are the whole table. Counting them would double-count.
        let rows = table.snapshot().live_rows() as f64;
        let ncols = table.schema().len();
        self.tables.push(Facts {
            path: path.to_string(),
            rows,
            pk: table.pk_col(),
            spans: vec![Span::Unasked; ncols],
        });
        self.tables.last_mut()
    }

    /// Exact live rows of a base table, or `None` if it will not resolve.
    pub fn table_rows(&mut self, path: &str) -> Option<f64> {
        self.facts(path).map(|f| f.rows)
    }

    /// `[min, max]` lanes of a **table** column, folded from the zone maps.
    ///
    /// One `u64` pair per granule, read straight out of the granule header --
    /// `PackedColumn::min_lane` is the frame-of-reference base and `max_lane`
    /// is a field, so neither touches a data page. A 200k-row table is ~196
    /// granules; a 10M-row one is ~9.8k, and the fold is gated on a join
    /// cluster of three or more relations so no small query ever pays it.
    ///
    /// **Granules with a NULL are skipped**, not merged: a NULL row stores
    /// lane 0, which would drag the minimum to zero and turn a tight span into
    /// a meaningless one. Skipping loses precision on a mostly-NULL column and
    /// never invents range that is not there.
    ///
    /// Refused for `Str` (per-granule dictionary codes are not comparable
    /// across granules) and for `F64` (a float span says nothing about how
    /// many distinct values live in it).
    fn span(&mut self, path: &str, col: usize) -> Option<(u64, u64)> {
        let idx = match self.tables.iter().position(|f| f.path == path) {
            Some(i) => i,
            None => {
                self.facts(path)?;
                self.tables.len() - 1
            }
        };
        match self.tables[idx].spans.get(col) {
            None => return None,
            Some(Span::Known(lo, hi)) => return Some((*lo, *hi)),
            Some(Span::Unknown) => return None,
            Some(Span::Unasked) => {}
        }
        let found = self.fold_span(path, col);
        self.tables[idx].spans[col] = match found {
            Some((lo, hi)) => Span::Known(lo, hi),
            None => Span::Unknown,
        };
        found
    }

    fn fold_span(&self, path: &str, col: usize) -> Option<(u64, u64)> {
        let table = self.catalog.table_by_path(path).ok()?;
        match table.schema().fields().get(col)?.ty.physical() {
            PhysicalType::U64 | PhysicalType::I64 => {}
            PhysicalType::Str | PhysicalType::F64 => return None,
        }
        let snap = table.snapshot();
        let (mut lo, mut hi) = (u64::MAX, 0u64);
        let mut seen = false;
        for pi in 0..snap.len() {
            for g in &snap.part(pi).granules {
                // A part built before `ALTER TABLE ... ADD COLUMN` is short.
                let Some(pc) = g.columns.get(col) else { continue };
                if pc.is_empty() || pc.nulls().is_some() {
                    continue;
                }
                lo = lo.min(pc.min_lane());
                hi = hi.max(pc.max_lane());
                seen = true;
            }
        }
        seen.then_some((lo, hi))
    }

    /// Distinct values in a **table** column, or `None` when nothing here can
    /// bound it.
    ///
    /// Exact for a primary key. For anything else this is the span heuristic:
    /// `min(rows, max - min + 1)`, which is
    ///
    ///   * **exact** for a dense integer key, and a foreign key that points at
    ///     a surrogate primary key is exactly that -- which is the case join
    ///     ordering actually has to get right;
    ///   * an **over-estimate** for a sparse column (ids scattered over a wide
    ///     range). An over-estimated NDV divides the join's output by too much
    ///     and makes that join look cheaper than it is, so a sparse foreign
    ///     key can pull a relation earlier into the order than it deserves.
    ///     The damage is bounded by the `rows` clamp, and by the fact that the
    ///     *other* side of an equi-join is usually the key, whose NDV is
    ///     exact and is what `max` picks.
    pub fn table_ndv(&mut self, path: &str, col: usize) -> Option<f64> {
        let f = self.facts(path)?;
        let rows = f.rows;
        if f.pk == Some(col) {
            return Some(rows.max(MIN_ROWS));
        }
        let (lo, hi) = self.span(path, col)?;
        let span = (hi - lo).saturating_add(1) as f64;
        Some(span.min(rows).max(MIN_ROWS))
    }

    // ------------------------------------------------------------ cardinality

    /// Estimated rows out of a plan.
    ///
    /// Never zero (see [`MIN_ROWS`]) and never negative. An unrecognized shape
    /// answers with its input's estimate rather than refusing, because a
    /// refusal here would disable reordering for the whole query while a
    /// pass-through is right for every node that neither adds nor drops rows.
    pub fn rows(&mut self, plan: &LogicalPlan) -> f64 {
        self.rows_at(plan, 0)
    }

    fn rows_at(&mut self, plan: &LogicalPlan, depth: usize) -> f64 {
        if depth > MAX_DEPTH {
            return MIN_ROWS;
        }
        let d = depth + 1;
        let out = match plan {
            LogicalPlan::Scan(s) => {
                let base = self.table_rows(&s.table).unwrap_or(UNKNOWN_TABLE_ROWS);
                base * self.selectivity_all(s, &s.filters)
            }
            LogicalPlan::Filter { input, predicate } => {
                let n = self.rows_at(input, d);
                match scan_of(input) {
                    Some(s) => n * self.selectivity(s, predicate),
                    None => n * SEL_OTHER,
                }
            }
            LogicalPlan::Project { input, .. } | LogicalPlan::Window { input, .. } => {
                self.rows_at(input, d)
            }
            LogicalPlan::Sort { input, .. } => self.rows_at(input, d),
            LogicalPlan::Aggregate { input, group, .. } => {
                let n = self.rows_at(input, d);
                self.groups(input, group, n)
            }
            LogicalPlan::Limit { input, limit, offset } => {
                let n = self.rows_at(input, d);
                match limit {
                    Some(l) => n.min(*l as f64).max(MIN_ROWS),
                    None => (n - *offset as f64).max(MIN_ROWS),
                }
            }
            LogicalPlan::LimitBy { input, limit, keys } => {
                let n = self.rows_at(input, d);
                n.min(self.groups(input, keys, n) * *limit as f64)
            }
            // No NDV for the whole row, so this is the same guess `Aggregate`
            // makes with no group columns it can resolve.
            LogicalPlan::Distinct { input } => distinct_guess(self.rows_at(input, d)),
            LogicalPlan::Join { left, right, op, on, .. } => {
                let (l, r) = (self.rows_at(left, d), self.rows_at(right, d));
                let inner = self.equi_rows(left, right, on, l, r);
                match op {
                    // An outer join emits at least every row of the preserved
                    // side, whatever the equi-join produced.
                    JoinOp::Left => inner.max(l),
                    JoinOp::Right => inner.max(r),
                    JoinOp::Full => inner.max(l).max(r),
                    JoinOp::Cross => l * r,
                    JoinOp::Inner => inner,
                }
            }
            // The three set operations have three different ceilings, and
            // using `UNION`'s for all of them would have told the join
            // reorderer that `big EXCEPT small` is bigger than `big`.
            //   UNION      -- at most the sum of the branches
            //   INTERSECT  -- at most the smallest branch
            //   EXCEPT     -- at most branch 0, which is the one that streams
            LogicalPlan::Union { inputs, op, all, .. } => {
                let mut it = inputs.iter().map(|i| self.rows_at(i, d));
                let n = match op {
                    SetOp::Union => it.sum(),
                    SetOp::Intersect => it.fold(f64::INFINITY, f64::min),
                    SetOp::Except => it.next().unwrap_or(0.0),
                };
                if *all {
                    n
                } else {
                    distinct_guess(n)
                }
            }
            LogicalPlan::Values { rows, .. } => rows.len() as f64,
            LogicalPlan::Empty { .. } => 0.0,
        };
        out.max(MIN_ROWS)
    }

    /// Rows out of an equi-join, by the textbook containment formula.
    ///
    /// `|R ⋈ S| = |R| · |S| / max(ndv(a), ndv(b))`, one factor per equi-pair.
    /// `max` rather than either side alone is what makes the common case
    /// right: joining a fact table's foreign key to a dimension's primary key
    /// divides by the dimension's row count and answers `|fact|`, exactly.
    ///
    /// With no NDV for either column the pair contributes nothing, so an
    /// equi-join between two relations nothing is known about is estimated as
    /// a cross product. That is the pessimistic direction and it is the safe
    /// one: it keeps an unknown relation *late* in the order rather than
    /// promoting it on the strength of a number nobody has.
    fn equi_rows(
        &mut self,
        left: &LogicalPlan,
        right: &LogicalPlan,
        on: &[(usize, usize)],
        l: f64,
        r: f64,
    ) -> f64 {
        let mut out = l * r;
        for &(lc, rc) in on {
            let a = self.col_ndv(left, lc).map(|n| n.min(l));
            let b = self.col_ndv(right, rc).map(|n| n.min(r));
            let div = match (a, b) {
                (Some(x), Some(y)) => x.max(y),
                (Some(x), None) => x,
                (None, Some(y)) => y,
                (None, None) => continue,
            };
            out /= div.max(MIN_ROWS);
        }
        out.max(MIN_ROWS)
    }

    /// NDV of column `col` of a plan's output, seen through the row-preserving
    /// nodes that sit between a join and its scan.
    ///
    /// Stops at anything that could change the column's value set: an
    /// aggregate, a union, a join, a computed projection. Refusing is always
    /// safe here -- see [`Self::equi_rows`] for what a refusal does.
    pub fn col_ndv(&mut self, plan: &LogicalPlan, col: usize) -> Option<f64> {
        let (node, table_col) = resolve_column(plan, col)?;
        let n = self.table_ndv(&node.table, table_col)?;
        // A predicate on the scan cannot add distinct values and usually
        // removes some; clamping to the filtered row count is the cheapest
        // correction that is never an over-estimate in the wrong direction.
        let rows = self.table_rows(&node.table).unwrap_or(UNKNOWN_TABLE_ROWS)
            * self.selectivity_all(node, &node.filters);
        Some(n.min(rows).max(MIN_ROWS))
    }

    /// Distinct groups for a `GROUP BY` / `LIMIT BY` key list.
    fn groups(&mut self, input: &LogicalPlan, keys: &[BoundExpr], rows: f64) -> f64 {
        if keys.is_empty() {
            return MIN_ROWS;
        }
        let mut n = 1.0f64;
        for k in keys {
            n *= match k.as_column().and_then(|c| self.col_ndv(input, c)) {
                Some(v) => v,
                // One unresolvable key column poisons the product, so fall
                // straight to the whole-row guess rather than multiplying a
                // known NDV by a made-up one.
                None => return distinct_guess(rows),
            };
        }
        n.min(rows).max(MIN_ROWS)
    }

    // ------------------------------------------------------------ selectivity

    fn selectivity_all(&mut self, node: &ScanNode, preds: &[BoundExpr]) -> f64 {
        // Independence assumption, and the place it is made. See the module
        // docs: correlated conjuncts are under-estimated by roughly their
        // correlation and nothing here detects that.
        preds.iter().fold(1.0, |acc, p| acc * self.selectivity(node, p))
    }

    /// Fraction of rows a predicate admits, in `[0, 1]`.
    fn selectivity(&mut self, node: &ScanNode, e: &BoundExpr) -> f64 {
        match e {
            BoundExpr::Literal { value, .. } => match value {
                Value::Bool(true) => 1.0,
                Value::Bool(false) | Value::Null => 0.0,
                _ => SEL_OTHER,
            },
            BoundExpr::Binary { left, op, right, .. } => self.sel_binary(node, left, *op, right),
            BoundExpr::InList { expr, list, negated } => {
                let s = match self.column_of(node, expr) {
                    Some((path, c)) => match self.table_ndv(&path, c) {
                        Some(n) => (list.len() as f64 / n).min(1.0),
                        None => (list.len() as f64 * SEL_EQ).min(1.0),
                    },
                    None => (list.len() as f64 * SEL_EQ).min(1.0),
                };
                if *negated {
                    1.0 - s
                } else {
                    s
                }
            }
            BoundExpr::IsNull { negated, .. } => {
                if *negated {
                    1.0 - SEL_NULL
                } else {
                    SEL_NULL
                }
            }
            _ => SEL_OTHER,
        }
    }

    fn sel_binary(
        &mut self,
        node: &ScanNode,
        left: &BoundExpr,
        op: BinaryOp,
        right: &BoundExpr,
    ) -> f64 {
        // `a AND b` / `a OR b` are conjunction and disjunction of two
        // selectivities, under the same independence assumption.
        match op {
            BinaryOp::And => {
                return self.selectivity(node, left) * self.selectivity(node, right);
            }
            BinaryOp::Or => {
                let (a, b) = (self.selectivity(node, left), self.selectivity(node, right));
                return a + b - a * b;
            }
            _ => {}
        }
        // Normalize to `column <op> literal`; `literal <op> column` flips.
        let (col, lit, op) = match (self.column_of(node, left), right.as_literal()) {
            (Some(c), Some(v)) => (c, v, op),
            _ => match (self.column_of(node, right), left.as_literal()) {
                (Some(c), Some(v)) => (c, v, flip(op)),
                _ => return SEL_OTHER,
            },
        };
        let (path, tc) = col;
        match op {
            BinaryOp::Eq => match self.table_ndv(&path, tc) {
                Some(n) => (1.0 / n).min(1.0),
                None => SEL_EQ,
            },
            BinaryOp::NotEq => {
                1.0 - match self.table_ndv(&path, tc) {
                    Some(n) => (1.0 / n).min(1.0),
                    None => SEL_EQ,
                }
            }
            BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq => {
                self.sel_range(&path, tc, op, lit)
            }
            _ => SEL_OTHER,
        }
    }

    /// Range selectivity by linear interpolation over `[min, max]`.
    ///
    /// Uniformity is the assumption, and it is the one range estimation always
    /// makes. It is right for a sort key and a timestamp -- the columns ranges
    /// are actually written against -- and arbitrarily wrong for a skewed one,
    /// which is what a histogram would fix and this does not have.
    ///
    /// The literal is laned **through the column's own type**, not through its
    /// own `Value` variant, and that distinction is the whole correctness of
    /// this function. `WHERE id < 5` on a `UInt64` column binds `5` as
    /// `Value::Int(5)`, whose signed lane is `5 ^ (1 << 63)` -- above every
    /// unsigned lane in the table. Laning it by variant put the literal at the
    /// top of the zone map, made `id < 5` look like it admitted **every** row
    /// of a 20 000-row table, and left the relation last in the join order
    /// when five rows belong first. `Value::to_lane_phys` is the same function
    /// the index path and the scan use, so all three agree by construction.
    fn sel_range(&mut self, path: &str, col: usize, op: BinaryOp, lit: &Value) -> f64 {
        let Some(ty) = self.col_type(path, col) else { return SEL_RANGE };
        // An error is a literal with no lane in this column at all -- `id <
        // 'x'`, or a decimal that does not divide. Not a reason to guess a
        // *number*; fall back to the shape's default.
        let Ok(v) = lit.to_lane_phys(ty.base().physical(), &ty) else { return SEL_RANGE };
        let Some((lo, hi)) = self.span(path, col) else { return SEL_RANGE };
        if hi <= lo {
            // One distinct value: the predicate is all or nothing, and which
            // one is exactly what the comparison against it says.
            return match op {
                BinaryOp::Lt => (v > lo) as u8 as f64,
                BinaryOp::LtEq => (v >= lo) as u8 as f64,
                BinaryOp::Gt => (v < lo) as u8 as f64,
                _ => (v <= lo) as u8 as f64,
            };
        }
        let span = (hi - lo) as f64;
        let below = (v.clamp(lo, hi) - lo) as f64 / span;
        match op {
            BinaryOp::Lt | BinaryOp::LtEq => below,
            _ => 1.0 - below,
        }
        .clamp(0.0, 1.0)
    }

    /// A table column's declared type.
    fn col_type(&self, path: &str, col: usize) -> Option<DataType> {
        let t = self.catalog.table_by_path(path).ok()?;
        Some(t.schema().fields().get(col)?.ty.clone())
    }

    /// `(table path, table column)` if this expression is a bare reference to
    /// a column of `node`'s scan.
    fn column_of(&self, node: &ScanNode, e: &BoundExpr) -> Option<(String, usize)> {
        let projected = e.as_column()?;
        Some((node.table.clone(), *node.projection.get(projected)?))
    }
}

/// Rows assumed for a relation the catalog will not resolve.
///
/// Large on purpose. A relation nothing is known about must not be promoted to
/// the front of a join order by looking small; the estimate that keeps it
/// where the user wrote it is the one that keeps the plan today's plan.
const UNKNOWN_TABLE_ROWS: f64 = 1_000_000.0;

/// Distinct rows of `n`, with nothing to base it on.
///
/// `n^(3/4)` is the shape a distinct count usually has -- sublinear, but not
/// bounded by a constant -- and it is a guess. It only ever ranks two plans
/// against each other, never sizes an allocation.
fn distinct_guess(n: f64) -> f64 {
    n.powf(0.75).max(MIN_ROWS)
}

fn flip(op: BinaryOp) -> BinaryOp {
    match op {
        BinaryOp::Lt => BinaryOp::Gt,
        BinaryOp::LtEq => BinaryOp::GtEq,
        BinaryOp::Gt => BinaryOp::Lt,
        BinaryOp::GtEq => BinaryOp::LtEq,
        other => other,
    }
}

/// The scan under a chain of row-preserving nodes, or `None`.
fn scan_of(plan: &LogicalPlan) -> Option<&ScanNode> {
    match plan {
        LogicalPlan::Scan(s) => Some(s),
        LogicalPlan::Filter { input, .. } | LogicalPlan::Sort { input, .. } => scan_of(input),
        _ => None,
    }
}

/// Follow output column `col` down to the scan column it *is*.
///
/// Only through nodes that neither change a value nor invent a row: a
/// projection of bare columns, a filter, a sort. Anything else returns `None`,
/// and every caller treats that as "no statistic".
fn resolve_column(plan: &LogicalPlan, col: usize) -> Option<(&ScanNode, usize)> {
    let mut plan = plan;
    let mut col = col;
    for _ in 0..MAX_DEPTH {
        match plan {
            LogicalPlan::Scan(s) => return Some((s, *s.projection.get(col)?)),
            LogicalPlan::Project { input, exprs, .. } => {
                col = exprs.get(col)?.as_column()?;
                plan = input;
            }
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::LimitBy { input, .. }
            | LogicalPlan::Distinct { input } => plan = input,
            _ => return None,
        }
    }
    None
}

// ------------------------------------------------------------------ join cost

/// What one join input looks like to the cost model.
#[derive(Clone, Copy)]
pub struct Side {
    /// Estimated rows.
    pub rows: f64,
    /// Cost already accrued producing those rows.
    pub cost: f64,
}

/// Cost of joining two sides that produce `out` rows between them.
///
/// The operator builds its hash table on the **smaller** side and probes with
/// the larger -- its module docs say so and its code does it -- so the model
/// charges `min` for the build and `max` for the probe rather than trusting
/// which one the plan calls "left". That is what makes a join's cost
/// commutative, which is what lets the search consider distinct *sets* of
/// relations rather than both spellings of every pair.
///
/// # The index-nested-loop strategy is deliberately not priced. A null result
///
/// `exec::operators::join` implements a second strategy: when one side is a
/// scan of a table whose primary key is the join key and the other side is
/// small, the keyed side is *fetched by key* instead of read, which its module
/// docs measure at 175x to 540x. Pricing it here is one `if` -- charge
/// `KEY_PROBE` per probe row and drop the keyed side's scan -- and it was
/// written, measured and removed.
///
/// **Because nothing selects it.** `Join::with_index` is the only way to arm
/// that strategy and no planner path calls it: `grep -rn with_index src/`
/// finds the definition and `tests/plan_join_strategy.rs`, which builds the
/// operator by hand precisely because the SQL path cannot. Measured through
/// `Session::query` on a 200k-row keyed table joined to 10 rows on its primary
/// key: 2.2 ms, which is the full scan, not the 20 us a fetch would cost.
///
/// So the branch was a cost model pricing a strategy that will not run, and it
/// changed a plan for the worse. A/B over four shapes with the branch switched
/// in one binary: three chose the same order either way, and `FROM big JOIN
/// small ON small.k = big.id JOIN dim ON small.d = dim.id` chose
/// `[small, big, dim]` with it and `[big, small, dim]` without -- 2.256 ms
/// against 2.188 ms, i.e. no better and possibly worse, because `big` is read
/// in full whichever end of the order it sits at.
///
/// **Put it back when, and only when, the planner can arm the strategy.** At
/// that point the model needs: a `keyed` flag on [`Side`], `KEY_PROBE` per
/// probe row in place of the build and probe terms, and the keyed side's own
/// `cost` dropped rather than added -- the crossover test is
/// `probe.rows * KEY_PROBE <= keyed.rows * SCAN`, which is `join::worth`
/// written in these units.
pub fn join_cost(left: Side, right: Side, out: f64) -> f64 {
    left.cost
        + right.cost
        + out * OUT
        + left.rows.min(right.rows) * BUILD
        + left.rows.max(right.rows) * PROBE
}

/// Cost of reading `rows` rows out of storage.
pub fn scan_cost(rows: f64) -> f64 {
    rows * SCAN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_literal_lanes_through_the_column_type_not_its_own() {
        // The bug this pins: `5` binds as `Value::Int` and a `UInt64` column
        // lanes unsigned, so laning by variant put the literal above every
        // row in the table and `id < 5` estimated at 100% of a 20k-row scan.
        let u = Value::Int(5).to_lane_phys(PhysicalType::U64, &DataType::UInt64).unwrap();
        let i = Value::Int(5).to_lane_phys(PhysicalType::I64, &DataType::Int64).unwrap();
        assert_eq!(u, 5);
        assert_ne!(i, u);
        assert!(Value::Int(-1).to_lane_phys(PhysicalType::U64, &DataType::UInt64).is_err());
    }

    #[test]
    fn join_cost_is_commutative() {
        // The search enumerates sets of relations, not orderings of a pair, so
        // a cost that depended on which side was written first would make the
        // dynamic-programming table wrong rather than merely imprecise.
        let a = Side { rows: 1_000.0, cost: 1_000.0 };
        let b = Side { rows: 40.0, cost: 40.0 };
        assert_eq!(join_cost(a, b, 1_000.0), join_cost(b, a, 1_000.0));
    }

    #[test]
    fn the_small_side_is_the_one_that_is_built() {
        // 1000 x 40 costs the same as 40 x 1000 (above), and both cost less
        // than building the large side would: 1000*BUILD + 40*PROBE.
        let c = join_cost(Side { rows: 1_000.0, cost: 0.0 }, Side { rows: 40.0, cost: 0.0 }, 0.0);
        assert_eq!(c, 40.0 * BUILD + 1_000.0 * PROBE);
    }
}
