//! The physical plan: where access-path decisions live.
//!
//! The logical plan says *what* rows a query wants. This says *how* the engine
//! will get them. The distinction only earns its keep once there is a choice to
//! make, and until this module existed there was none: `exec::operators::build`
//! was a 1:1 structural mapping from [`LogicalPlan`] to operators, with no
//! place to put a decision even if one had been obvious.
//!
//! Two decisions live here today.
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
//! ## What this is deliberately *not*
//!
//! Not a cost model. There are no statistics beyond "how many live rows does
//! this table have", no cardinality estimates and no plan search. The one
//! number that could be called a cost --
//! [`SCAN_ROWS_PER_PROBE`] -- is a measured constant with the measurement
//! written next to it. A real cost model is a later, separate piece of work;
//! what matters here is that there is now somewhere for it to go.
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

use crate::exec::operators::window::WindowNode;

use super::logical::{BoundAgg, BoundExpr, LogicalPlan, ScanNode, SortKey};

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

pub enum PhysicalPlan<'a> {
    /// Sequential scan with zone-map pruning and PREWHERE.
    Scan(&'a ScanNode),
    /// Primary-key point/batch lookup.
    IndexLookup(Box<IndexPath<'a>>),
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
}

/// Lower a logical plan, choosing an access path for every scan.
pub fn lower<'a>(plan: &'a LogicalPlan, catalog: &Catalog) -> Result<PhysicalPlan<'a>> {
    lower_at(plan, catalog, 0)
}

fn lower_at<'a>(
    plan: &'a LogicalPlan,
    catalog: &Catalog,
    depth: usize,
) -> Result<PhysicalPlan<'a>> {
    if depth > MAX_PLAN_DEPTH {
        return Err(too_deep());
    }
    let d = depth + 1;
    let down = |p: &'a LogicalPlan| -> Result<Box<PhysicalPlan<'a>>> {
        Ok(Box::new(lower_at(p, catalog, d)?))
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
            PhysicalPlan::Aggregate { input: down(input)?, group, aggs, schema }
        }
        LogicalPlan::Window { input, node } => {
            PhysicalPlan::Window { input: down(input)?, node }
        }
        LogicalPlan::Sort { input, keys } => {
            PhysicalPlan::Sort { input: down(input)?, keys, fetch: None }
        }
        LogicalPlan::Limit { input, limit, offset } => {
            let mut inner = lower_at(input, catalog, d)?;
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
                .map(|p| lower_at(p, catalog, d))
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
        PhysicalPlan::Project { input, .. } => fuse_top_k(input, k),
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
fn exact_lane(v: &Value, ty: &DataType) -> Option<u64> {
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

// ------------------------------------------------------------------ EXPLAIN

impl PhysicalPlan<'_> {
    pub fn schema(&self) -> &Schema {
        match self {
            PhysicalPlan::Scan(s) => &s.schema,
            PhysicalPlan::IndexLookup(i) => &i.node.schema,
            PhysicalPlan::Filter { input, .. }
            | PhysicalPlan::Sort { input, .. }
            | PhysicalPlan::Limit { input, .. }
            | PhysicalPlan::LimitBy { input, .. }
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
            | PhysicalPlan::Values { .. }
            | PhysicalPlan::Empty { .. } => Vec::new(),
            PhysicalPlan::Filter { input, .. }
            | PhysicalPlan::Project { input, .. }
            | PhysicalPlan::Aggregate { input, .. }
            | PhysicalPlan::Sort { input, .. }
            | PhysicalPlan::Limit { input, .. }
            | PhysicalPlan::LimitBy { input, .. }
            | PhysicalPlan::Window { input, .. }
            | PhysicalPlan::Distinct { input } => vec![input],
            PhysicalPlan::Join { left, right, .. } => vec![left, right],
            PhysicalPlan::Union { branches, .. } => branches.iter().collect(),
        }
    }

    /// The access path and its parameters, one line, no children.
    ///
    /// `EXPLAIN` is the only way anyone outside this file can tell whether the
    /// index fired, so the label states the decision and the numbers behind it
    /// rather than merely naming the operator.
    fn label(&self) -> String {
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
                let a: Vec<String> = aggs
                    .iter()
                    .map(|a| {
                        let args: Vec<String> = a.args.iter().map(|x| x.to_string()).collect();
                        format!("{}({})", a.func.name, args.join(", "))
                    })
                    .collect();
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
            // OR is not a conjunct, so it never reaches the scan's filter list
            "SELECT v FROM t WHERE id = 1 OR id = 2",
        ] {
            let e = phys(&mut s, q);
            assert!(!e.contains("IndexLookup"), "{q} should scan:\n{e}");
            assert!(e.contains("Scan default.t"), "{q}:\n{e}");
        }
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
