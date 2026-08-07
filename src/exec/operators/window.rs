//! Window functions: `f(args) OVER (PARTITION BY ... ORDER BY ... <frame>)`.
//!
//! A window function is neither scalar nor aggregate. It reads a *set* of rows
//! -- the frame -- and yet emits one value per input row, so it cannot live in
//! `Project` (which is row-at-a-time) or in `Aggregate` (which collapses its
//! input). Hence its own operator, appending one column per function to the
//! block it is given and changing nothing else.
//!
//! ## Where the sort comes from
//!
//! Nowhere in this file. The binder puts an ordinary [`LogicalPlan::Sort`] on
//! the partition keys followed by the ORDER BY keys underneath every `Window`
//! node, so partitions arrive contiguous and ordered and this operator only has
//! to find the seams. That is deliberate: the sort already has a radix path, a
//! k-way merge and external spilling, and a second sort written here would have
//! had none of them. It also means `EXPLAIN` shows the sort a window costs,
//! which is the single most surprising thing about window functions'
//! performance.
//!
//! `sum(x) OVER ()` -- no partition, no order -- gets **no** sort node at all;
//! there is nothing to order by, and paying for a sort to compute a grand total
//! would be the obvious way to make the feature look expensive.
//!
//! [`LogicalPlan::Sort`]: crate::planner::logical::LogicalPlan::Sort
//!
//! ## Partitions and peers, found once
//!
//! One `u8` per row carries two bits: "starts a new partition" and "starts a
//! new peer group" (a peer group being a maximal run of rows tying on every
//! ORDER BY key). Both are computed by [`mark_changes`], which dispatches on
//! the column's physical type **once** and then walks a flat slice comparing
//! adjacent elements -- no `Value` materialization, no per-row match. Every
//! function then reads the same two bits, so k window functions cost one
//! boundary pass, not k.
//!
//! One byte per row is the whole per-row footprint this operator adds beyond
//! its output columns. A bitset would be four times smaller and measurably
//! slower to read in the inner loop, which is the wrong trade for a structure
//! that is scanned strictly sequentially exactly once per function.
//!
//! ## How a frame is folded
//!
//! Recomputing an aggregate from scratch at every row is O(rows x frame), and
//! for the overwhelmingly common `UNBOUNDED PRECEDING` running total that is
//! O(rows^2). So the frame's *shape* selects a sweep ([`Sweep::of`]):
//!
//! ```text
//!   whole partition          fold once, broadcast              O(rows)
//!   left edge pinned         one accumulator, fed forwards     O(rows)
//!   right edge pinned        one accumulator, fed backwards    O(rows)
//!   both edges move          refold per row                    O(rows x frame)
//! ```
//!
//! The first three cover every frame anyone writes by accident and most of the
//! ones written on purpose: the two defaults, `UNBOUNDED PRECEDING AND CURRENT
//! ROW`, `CURRENT ROW AND UNBOUNDED FOLLOWING`, and the whole-partition form.
//!
//! This is not a constant factor and the measurement says so. `sum(v) OVER
//! (ORDER BY id)`, A/B interleaved against the same build with the sweep forced
//! to `Refold` (temporary env switch, since removed), best-of-5 per side, three
//! rounds:
//!
//! ```text
//!   rows     incremental     refold        ratio
//!   20 000      0.36 ms      129.2 ms       356x
//!   60 000      1.16 ms     1190.0 ms      1025x
//! ```
//!
//! The ratio tripling when the row count tripled is the whole point: the two
//! sides are O(n) and O(n^2), so quoting a single speedup would be meaningless.
//!
//! The fourth row of the table -- a genuinely sliding frame such as `ROWS
//! BETWEEN 3 PRECEDING AND CURRENT ROW` -- **refolds**, deliberately. Sliding
//! incrementally needs the accumulator to support *removal*, and only some of
//! them can: `sum` and `count` can subtract, `min`/`max` need a monotone deque,
//! `uniq` cannot at all. Rather than split the library in two and maintain a
//! second `sum`, the frame is refolded with one `update` call over a contiguous
//! slice of the shared row-id vector, which is vectorized inside the
//! accumulator. That makes the cost proportional to the frame *width*, which
//! for a moving average is 3 or 7 or 30 -- not to the partition. Measured at 1M
//! rows: `ROWS 30 PRECEDING` runs at 8.5 M rows/s against 28.7 M rows/s for the
//! running total, i.e. the 31-wide refold costs 3.4x, not 31x. A wide sliding
//! frame over a large table is the one shape this operator is bad at, and it is
//! bad at it on purpose.
//!
//! ## What it costs end to end
//!
//! 1M rows, in-memory session, best-of-3, whole result materialized:
//!
//! ```text
//!   SELECT id, v FROM t ORDER BY id                   9.4 ms   106 M rows/s
//!   sum(v) OVER ()                       (no sort)   11.0 ms    91 M rows/s
//!   row_number() OVER (ORDER BY id)                  21.4 ms    47 M rows/s
//!   sum(v) OVER (ORDER BY id)                        34.9 ms    29 M rows/s
//!   lag + lead + rank, one OVER clause               40.0 ms    25 M rows/s
//!   sum(v) OVER (PARTITION BY g ORDER BY id)        348.5 ms     2.9 M rows/s
//! ```
//!
//! The last row is the sort, not the window: `g` is a `String`, which takes the
//! comparison path. The same query partitioned on an integer runs in 19 ms. If
//! a window query is slow, `EXPLAIN` will show a `Sort` and that is where to
//! look -- which is the second reason the sort is a plan node here rather than
//! something this operator does privately.
//!
//! ## What is refused rather than approximated
//!
//! `RANGE` with a numeric offset (`RANGE BETWEEN 3 PRECEDING ...`) and `GROUPS`
//! frames are rejected in the parser. `RANGE` compares *values* of the ORDER BY
//! key, not row positions, so reading it as `ROWS` agrees only when the key is
//! dense and unique -- a wrong answer that looks right on the test data.
//! `RANGE` with `UNBOUNDED`/`CURRENT ROW` bounds *is* implemented, and is not
//! the same as `ROWS`: under `RANGE`, `CURRENT ROW` means "through the last row
//! tied with me", which is exactly what makes `sum(x) OVER (ORDER BY k)` give
//! every row tied on `k` the same running total.
//!
//! `DISTINCT` inside a window call is refused; so does SQLite, and so does
//! Postgres.

use std::borrow::Cow;
use std::sync::Arc;

use crate::common::{Error, Result};
use crate::exec::expr;
use crate::exec::functions::{aggregate, AggFn};
use crate::planner::logical::BoundExpr;
use crate::sql::ast::{FrameBound, FrameUnits, WindowFrame};
use crate::types::{Block, Column, ColumnBuilder, ColumnData, DataType, Field, Schema, Value};

use super::{chunk, drain, MemGuard, Operator, QueryContext, ScanStats};

// ================================================================= registry

/// What a window call computes.
///
/// The aggregates are *not* reimplemented: [`WindowKind::Agg`] holds the very
/// same `&'static AggFn` a `GROUP BY` would use, so `sum(x) OVER (...)` and
/// `sum(x)` fold through one piece of code and cannot disagree.
#[derive(Clone, Copy)]
pub enum WindowKind {
    RowNumber,
    Rank,
    DenseRank,
    PercentRank,
    CumeDist,
    Ntile,
    Lag,
    Lead,
    FirstValue,
    LastValue,
    NthValue,
    Agg(&'static AggFn),
}

impl WindowKind {
    /// The canonical name, for `EXPLAIN` and for the binder's de-duplication
    /// key. An aggregate reports its own registry name, so `sum` and `SUM`
    /// collapse to one column exactly as they do under `GROUP BY`.
    pub fn name(&self) -> &'static str {
        match self {
            WindowKind::RowNumber => "row_number",
            WindowKind::Rank => "rank",
            WindowKind::DenseRank => "dense_rank",
            WindowKind::PercentRank => "percent_rank",
            WindowKind::CumeDist => "cume_dist",
            WindowKind::Ntile => "ntile",
            WindowKind::Lag => "lag",
            WindowKind::Lead => "lead",
            WindowKind::FirstValue => "first_value",
            WindowKind::LastValue => "last_value",
            WindowKind::NthValue => "nth_value",
            WindowKind::Agg(f) => f.name,
        }
    }

    /// True for the functions whose answer depends only on the ordering, never
    /// on the frame. SQL says a frame written on one of these is ignored, so
    /// [`plan_call`] normalizes it away and nothing downstream has to remember
    /// the exception.
    fn ignores_frame(&self) -> bool {
        matches!(
            self,
            WindowKind::RowNumber
                | WindowKind::Rank
                | WindowKind::DenseRank
                | WindowKind::PercentRank
                | WindowKind::CumeDist
                | WindowKind::Ntile
                | WindowKind::Lag
                | WindowKind::Lead
        )
    }
}

/// Resolve a window function name. Case- and underscore-insensitive, matching
/// how [`crate::exec::functions::aggregate`] resolves its own.
///
/// Falls through to the aggregate registry, which is what makes every existing
/// aggregate usable as a window function for free. The order matters: the
/// window-only names are tried first so that a future aggregate named `rank`
/// could not silently take over the ranking function.
pub fn lookup(name: &str) -> Option<WindowKind> {
    let lower = name.to_ascii_lowercase();
    let bare: String = lower.chars().filter(|c| *c != '_').collect();
    Some(match bare.as_str() {
        "rownumber" => WindowKind::RowNumber,
        "rank" => WindowKind::Rank,
        "denserank" => WindowKind::DenseRank,
        "percentrank" => WindowKind::PercentRank,
        "cumedist" => WindowKind::CumeDist,
        "ntile" => WindowKind::Ntile,
        "lag" => WindowKind::Lag,
        "lead" => WindowKind::Lead,
        "firstvalue" => WindowKind::FirstValue,
        "lastvalue" => WindowKind::LastValue,
        "nthvalue" => WindowKind::NthValue,
        _ => return aggregate(&lower).map(WindowKind::Agg),
    })
}

/// A bound window call: everything the operator needs and nothing it does not.
#[derive(Clone)]
pub struct BoundWindow {
    pub kind: WindowKind,
    pub args: Vec<BoundExpr>,
    /// Parametric-aggregate arguments (`quantile(0.9)(x) OVER (...)`).
    pub params: Vec<Value>,
    /// The constant integer argument of `lag`/`lead`/`nth_value`/`ntile`,
    /// folded at bind time because SQL requires it to be constant and the
    /// binder is the last layer that can name the offending expression.
    pub offset: u64,
    pub frame: WindowFrame,
    pub ty: DataType,
    /// Output column name -- the call's own source text, matching how an
    /// unaliased aggregate is named.
    pub name: String,
}

/// The logical payload of a window step: the functions, the keys that group and
/// order them, and the schema that comes out.
///
/// Lives here rather than in `planner::logical` so that the plan node is one
/// line there: everything a window needs to be *described* and everything it
/// needs to be *run* is the same set of fields, and splitting them across two
/// modules would only create a pair that has to be kept in step.
pub struct WindowNode {
    /// Computed in this order, appended to the input in this order.
    pub funcs: Vec<BoundWindow>,
    /// `PARTITION BY` keys. Rows tying on all of them are one partition.
    pub partition: Vec<BoundExpr>,
    /// `ORDER BY` keys. Rows tying on all of them (within a partition) are
    /// peers, which is what `RANGE ... CURRENT ROW` and `rank` are defined in
    /// terms of.
    pub order: Vec<BoundExpr>,
    /// Input schema plus one field per entry of `funcs`.
    pub schema: Schema,
}

impl WindowNode {
    /// One line for `EXPLAIN`.
    pub fn label(&self) -> String {
        let list: Vec<&str> = self.funcs.iter().map(|f| f.name.as_str()).collect();
        let mut s = format!("Window [{}]", list.join(", "));
        if !self.partition.is_empty() {
            let p: Vec<String> = self.partition.iter().map(|e| e.to_string()).collect();
            s.push_str(&format!(" partition=[{}]", p.join(", ")));
        }
        if !self.order.is_empty() {
            let o: Vec<String> = self.order.iter().map(|e| e.to_string()).collect();
            s.push_str(&format!(" order=[{}]", o.join(", ")));
        }
        s
    }
}

// ============================================================ bind-time rules

/// Arity, argument types, constant folding and the result type of one window
/// call, all in one place so the binder holds no window-specific knowledge.
///
/// `args` arrives already bound; `frame` is the *effective* frame, so the
/// nullability rule below can see the real bounds rather than `None`.
pub fn plan_call(
    name: &str,
    mut args: Vec<BoundExpr>,
    params: Vec<Value>,
    distinct: bool,
    frame: WindowFrame,
    display: String,
) -> Result<BoundWindow> {
    let Some(kind) = lookup(name) else {
        return Err(Error::bind(format!("unknown window function `{name}`")));
    };
    if distinct {
        return Err(Error::bind(format!(
            "DISTINCT is not supported inside a window function: `{name}(DISTINCT ...) OVER (...)`"
        )));
    }
    let tys: Vec<DataType> = args.iter().map(|a| a.ty()).collect();
    let mut offset = 0u64;

    let ty = match kind {
        WindowKind::RowNumber | WindowKind::Rank | WindowKind::DenseRank => {
            arity(name, &tys, 0, 0)?;
            DataType::UInt64
        }
        WindowKind::PercentRank | WindowKind::CumeDist => {
            arity(name, &tys, 0, 0)?;
            DataType::Float64
        }
        WindowKind::Ntile => {
            arity(name, &tys, 1, 1)?;
            offset = const_count(&args[0], name, 1)?;
            // The bucket count is consumed here, not at run time: leaving it in
            // `args` would make the operator evaluate a constant column per row.
            args.clear();
            DataType::UInt64
        }
        WindowKind::Lag | WindowKind::Lead => {
            arity(name, &tys, 1, 3)?;
            offset = match args.get(1) {
                Some(e) => const_count(e, name, 0)?,
                None => 1,
            };
            // A default is allowed to be any expression (it is evaluated once
            // per block like any other column); only the offset must fold.
            let base = tys[0].clone();
            match tys.get(2) {
                Some(d) => DataType::promote(&base, d)?.to_nullable(),
                None => base.to_nullable(),
            }
        }
        WindowKind::FirstValue | WindowKind::LastValue => {
            arity(name, &tys, 1, 1)?;
            tys[0].clone().to_nullable()
        }
        WindowKind::NthValue => {
            arity(name, &tys, 2, 2)?;
            offset = const_count(&args[1], name, 1)?;
            args.truncate(1);
            tys[0].clone().to_nullable()
        }
        WindowKind::Agg(f) => {
            f.check_arity(tys.len())?;
            let base = (f.ret)(&tys, &params)?;
            // An aggregate over an *empty* frame is NULL (except `count`, which
            // is 0), so the column has to admit one. A frame that always spans
            // its own row cannot be empty, and those -- the two defaults among
            // them -- keep the aggregate's own type.
            if can_be_empty(&frame) {
                base.to_nullable()
            } else {
                base
            }
        }
    };

    // A frame written on `rank()` or `lag()` is legal and meaningless; folding
    // it to the whole partition here keeps `BoundWindow::frame` uniform, so a
    // reader of that field never has to ask which function it belongs to.
    let frame = if kind.ignores_frame() {
        WindowFrame {
            units: FrameUnits::Range,
            start: FrameBound::UnboundedPreceding,
            end: FrameBound::UnboundedFollowing,
        }
    } else {
        frame
    };
    Ok(BoundWindow { kind, args, params, offset, frame, ty, name: display })
}

fn arity(name: &str, tys: &[DataType], lo: usize, hi: usize) -> Result<()> {
    if tys.len() < lo || tys.len() > hi {
        let want = if lo == hi {
            format!("exactly {lo}")
        } else {
            format!("{lo} to {hi}")
        };
        return Err(Error::bind(format!(
            "window function {name} takes {want} argument(s), got {}",
            tys.len()
        )));
    }
    Ok(())
}

/// A non-negative integer literal argument, with `min` as the smallest legal
/// value (`nth_value`'s n starts at 1, `lag`'s offset at 0).
fn const_count(e: &BoundExpr, name: &str, min: u64) -> Result<u64> {
    let bad = || {
        Error::bind(format!(
            "the count argument of `{name}` must be an integer constant of at least {min}"
        ))
    };
    let v = e.as_literal().ok_or_else(bad)?;
    let n = match v {
        Value::UInt(n) => *n,
        Value::Int(n) if *n >= 0 => *n as u64,
        _ => return Err(bad()),
    };
    if n < min {
        return Err(bad());
    }
    Ok(n)
}

/// Can this frame ever select no rows at all?
///
/// Only when it excludes the current row, i.e. when it lies entirely before or
/// entirely after it. `CURRENT ROW` ranks between the two offset families, so
/// the test is one comparison per bound.
fn can_be_empty(f: &WindowFrame) -> bool {
    let here = FrameBound::CurrentRow.rank();
    !(f.start.rank() <= here && f.end.rank() >= here)
}

// ================================================================= operator

/// `flags` bit: this row begins a new partition.
const NEW_PART: u8 = 1;
/// `flags` bit: this row begins a new peer group. Implied by [`NEW_PART`].
const NEW_PEER: u8 = 2;

pub struct Window<'a> {
    input: Box<dyn Operator + 'a>,
    node: &'a WindowNode,
    ctx: &'a QueryContext,
    /// Reversed once materialization finishes, so `next` is a `pop`.
    out: Vec<Block>,
    ready: bool,
    guard: MemGuard,
}

impl<'a> Window<'a> {
    pub fn new(
        input: Box<dyn Operator + 'a>,
        node: &'a WindowNode,
        ctx: &'a QueryContext,
    ) -> Window<'a> {
        Window {
            input,
            node,
            ctx,
            out: Vec::new(),
            ready: false,
            guard: MemGuard::new(ctx, "a window function's row buffer"),
        }
    }

    fn materialize(&mut self) -> Result<()> {
        self.ready = true;
        let mut block = drain(&mut self.input, self.ctx, &mut self.guard)?;
        let n = block.rows();
        if n == 0 {
            return Ok(());
        }

        // One pass per key column, dispatched on physical type outside the row
        // loop. Partition keys set both bits at once: a new partition is
        // necessarily a new peer group, and folding that in here saves a
        // second sweep over `flags`.
        let mut flags = vec![0u8; n];
        flags[0] = NEW_PART | NEW_PEER;
        for c in expr::eval_all_cow(&self.node.partition, &block)? {
            mark_changes(&c, NEW_PART | NEW_PEER, &mut flags);
        }
        for c in expr::eval_all_cow(&self.node.order, &block)? {
            mark_changes(&c, NEW_PEER, &mut flags);
        }

        // Row ids, built once and shared by every function: an accumulator is
        // fed a *slice* of this, so a frame fold allocates nothing at all.
        let ids: Vec<u32> = (0..n as u32).collect();

        let pass = Pass {
            block: &block,
            flags: &flags,
            ids: &ids,
            ordered: !self.node.order.is_empty(),
            ctx: self.ctx,
        };
        let mut scratch: Vec<Value> = Vec::new();
        let mut cols = Vec::with_capacity(self.node.funcs.len());
        for f in &self.node.funcs {
            self.ctx.check()?;
            cols.push(eval_window(f, &pass, &mut scratch)?);
        }
        block.append_columns(cols)?;
        // Charged once, at the peak: the input rows, the appended columns, and
        // the 5 bytes/row of flags and row ids. `drain` already charged the
        // input, so this only ever grows the reservation.
        self.guard.grow_to(block.bytes() + n * 5)?;

        self.out = chunk(block);
        self.out.reverse();
        Ok(())
    }
}

impl Operator for Window<'_> {
    fn schema(&self) -> &Schema {
        &self.node.schema
    }

    fn stats(&self) -> ScanStats {
        self.input.stats()
    }

    fn next(&mut self) -> Result<Option<Block>> {
        if !self.ready {
            self.materialize()?;
        }
        Ok(self.out.pop())
    }
}

// ------------------------------------------------------------- boundaries

/// OR `bits` into `flags[i]` for every row that differs from its predecessor.
///
/// NULLs compare *equal* to each other here, which is what SQL wants of both
/// `PARTITION BY` (all NULLs are one partition) and peer detection (NULLs tie).
/// The null mask therefore gates the value comparison rather than being ignored
/// -- the data slot under a NULL holds a placeholder, and comparing those would
/// split a partition on garbage.
fn mark_changes(col: &Column, bits: u8, flags: &mut [u8]) {
    let n = flags.len().min(col.len());
    if n < 2 {
        return;
    }
    macro_rules! scan {
        ($vals:expr, $eq:expr) => {{
            let v = $vals;
            match &col.nulls {
                // The common case: one branchless compare per row.
                None => {
                    for i in 1..n {
                        if !$eq(&v[i], &v[i - 1]) {
                            flags[i] |= bits;
                        }
                    }
                }
                Some(nulls) => {
                    for i in 1..n {
                        let (a, b) = (nulls.get(i), nulls.get(i - 1));
                        if a != b || (!a && !$eq(&v[i], &v[i - 1])) {
                            flags[i] |= bits;
                        }
                    }
                }
            }
        }};
    }
    match &col.data {
        ColumnData::U64(v) => scan!(v, |a: &u64, b: &u64| a == b),
        ColumnData::I64(v) => scan!(v, |a: &i64, b: &i64| a == b),
        // Matches `Value`'s equality, not IEEE's: every NaN is one value and
        // -0.0 ties with 0.0, so a partition key does not fracture on a sign
        // bit no comparison can see.
        ColumnData::F64(v) => scan!(v, |a: &f64, b: &f64| a == b || (a.is_nan() && b.is_nan())),
        ColumnData::Str(v) => scan!(v, |a: &Arc<str>, b: &Arc<str>| a == b),
    }
}

/// End (exclusive) of the peer group starting at `gs`, bounded by `pe`.
#[inline]
fn peer_end(flags: &[u8], gs: usize, pe: usize) -> usize {
    let mut e = gs + 1;
    while e < pe && flags[e] & NEW_PEER == 0 {
        e += 1;
    }
    e
}

/// Start of the peer group ending (exclusively) at `ge`, bounded by `ps`.
#[inline]
fn peer_start(flags: &[u8], ge: usize, ps: usize) -> usize {
    let mut s = ge - 1;
    while s > ps && flags[s] & NEW_PEER == 0 {
        s -= 1;
    }
    s
}

// ------------------------------------------------------------------ frames

/// The frame of row `i`, as a half-open row range, given its partition
/// `[ps, pe)` and its peer group `[gs, ge)`.
///
/// Always returns `end >= start`; `end == start` is an empty frame.
#[inline]
fn frame_of(
    fr: &WindowFrame,
    i: usize,
    ps: usize,
    pe: usize,
    gs: usize,
    ge: usize,
) -> (usize, usize) {
    let rows = fr.units == FrameUnits::Rows;
    let s = match fr.start {
        FrameBound::UnboundedPreceding => ps,
        // `RANGE` offsets never reach here: the parser refuses them.
        FrameBound::Preceding(k) => i.saturating_sub(k as usize).max(ps),
        FrameBound::CurrentRow => {
            if rows {
                i
            } else {
                gs
            }
        }
        FrameBound::Following(k) => i.saturating_add(k as usize).min(pe),
        FrameBound::UnboundedFollowing => pe,
    };
    let e = match fr.end {
        FrameBound::UnboundedPreceding => ps,
        FrameBound::Preceding(k) => {
            let k = k as usize;
            if i >= k {
                (i - k + 1).clamp(ps, pe)
            } else {
                ps
            }
        }
        FrameBound::CurrentRow => {
            if rows {
                i + 1
            } else {
                ge
            }
        }
        FrameBound::Following(k) => i.saturating_add(k as usize).saturating_add(1).min(pe),
        FrameBound::UnboundedFollowing => pe,
    };
    (s, e.max(s))
}

/// How an aggregate frame is folded. See the module docs for the table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sweep {
    Whole,
    Forward,
    Backward,
    Refold,
}

impl Sweep {
    /// `ordered` is false when the window has no ORDER BY, in which case every
    /// row of a partition is a peer and `RANGE ... CURRENT ROW` degenerates to
    /// the partition edge -- the `sum(x) OVER (PARTITION BY k)` case, which
    /// would otherwise take the forward path and pay a virtual `update` and a
    /// `finish` per row to compute one constant per partition.
    ///
    /// Small, and honestly so. A/B interleaved against the same build with this
    /// clause disabled, 1M rows, four rounds: on an integer partition key
    /// (radix sort) 18.7/19.0/20.2/20.1 ms against 19.2/19.3/21.8/21.9 ms --
    /// 2-8%, but the sign is the same in every round. On a *string* key the two
    /// sides are indistinguishable (111-122 ms both ways), because the
    /// comparison sort underneath is six times the whole window step. Kept for
    /// the consistent sign and because the branch is free; do not expect it to
    /// show up on a query whose sort is the expensive part.
    fn of(fr: &WindowFrame, ordered: bool) -> Sweep {
        let range_cr = fr.units == FrameUnits::Range && !ordered;
        let opens = matches!(fr.start, FrameBound::UnboundedPreceding)
            || (range_cr && matches!(fr.start, FrameBound::CurrentRow));
        let closes = matches!(fr.end, FrameBound::UnboundedFollowing)
            || (range_cr && matches!(fr.end, FrameBound::CurrentRow));
        match (opens, closes) {
            (true, true) => Sweep::Whole,
            (true, false) => Sweep::Forward,
            (false, true) => Sweep::Backward,
            (false, false) => Sweep::Refold,
        }
    }
}

// -------------------------------------------------------------- evaluation

/// Everything computed once for the whole materialized block and then read by
/// every function: the rows, the partition/peer bits, the shared row-id vector
/// and the query's stop conditions.
///
/// One struct rather than six parameters because it is genuinely one thing --
/// the state of a single materialization pass -- and because k window functions
/// share exactly this and nothing else.
struct Pass<'p> {
    block: &'p Block,
    flags: &'p [u8],
    /// `0..rows`, so a frame fold is `update(args, &ids[s..e])` with no
    /// allocation and no index materialization.
    ids: &'p [u32],
    /// False when the window has no ORDER BY; see [`Sweep::of`].
    ordered: bool,
    ctx: &'p QueryContext,
}

impl Pass<'_> {
    /// End (exclusive) of the partition starting at `ps`.
    #[inline]
    fn part_end(&self, ps: usize) -> usize {
        let n = self.flags.len();
        let mut pe = ps + 1;
        while pe < n && self.flags[pe] & NEW_PART == 0 {
            pe += 1;
        }
        pe
    }
}

/// Compute one window function's output column over the whole materialized
/// block.
fn eval_window(f: &BoundWindow, p: &Pass<'_>, scratch: &mut Vec<Value>) -> Result<Column> {
    match f.kind {
        WindowKind::Agg(af) => eval_agg(f, af, p, scratch),
        WindowKind::Lag | WindowKind::Lead => eval_shift(f, p),
        WindowKind::FirstValue | WindowKind::LastValue | WindowKind::NthValue => eval_pick(f, p),
        _ => Ok(eval_rank(f, p)),
    }
}

/// `row_number`, `rank`, `dense_rank`, `percent_rank`, `cume_dist`, `ntile` --
/// everything that reads only the ordering.
fn eval_rank(f: &BoundWindow, p: &Pass<'_>) -> Column {
    let (n, flags) = (p.block.rows(), p.flags);
    let mut b = ColumnBuilder::with_capacity(f.ty.clone(), n);
    let mut ps = 0usize;
    while ps < n {
        let pe = p.part_end(ps);
        let size = pe - ps;
        let mut gs = ps;
        let mut ge = peer_end(flags, ps, pe);
        let mut dense = 1u64;
        for i in ps..pe {
            if i == ge {
                gs = ge;
                ge = peer_end(flags, gs, pe);
                dense += 1;
            }
            match f.kind {
                WindowKind::RowNumber => b.push_u64((i - ps + 1) as u64),
                WindowKind::Rank => b.push_u64((gs - ps + 1) as u64),
                WindowKind::DenseRank => b.push_u64(dense),
                WindowKind::PercentRank => b.push_f64(if size <= 1 {
                    0.0
                } else {
                    (gs - ps) as f64 / (size - 1) as f64
                }),
                WindowKind::CumeDist => b.push_f64((ge - ps) as f64 / size as f64),
                WindowKind::Ntile => b.push_u64(ntile(i - ps, size, f.offset)),
                _ => unreachable!("eval_rank only sees ordering functions"),
            }
        }
        ps = pe;
    }
    b.finish()
}

/// 1-based bucket of row `j` of `size`, split into `buckets` groups whose sizes
/// differ by at most one, the larger ones first. `buckets` is >= 1 by
/// [`const_count`].
#[inline]
fn ntile(j: usize, size: usize, buckets: u64) -> u64 {
    let k = (buckets as usize).max(1);
    let q = size / k;
    let r = size % k;
    if q == 0 {
        // More buckets than rows: one row each, and the tail stays empty.
        return j as u64 + 1;
    }
    let big = r * (q + 1);
    if j < big {
        (j / (q + 1)) as u64 + 1
    } else {
        ((j - big) / q + r) as u64 + 1
    }
}

/// `lag` / `lead`: a positional gather inside the partition, with a default for
/// the rows that fall off the end. The frame is ignored, per SQL.
fn eval_shift(f: &BoundWindow, p: &Pass<'_>) -> Result<Column> {
    let n = p.block.rows();
    let val = one_col(&f.args[0], p.block)?;
    let dflt = match f.args.get(2) {
        Some(e) => Some(one_col(e, p.block)?),
        None => None,
    };
    let back = matches!(f.kind, WindowKind::Lag);
    let off = f.offset as usize;

    let mut b = ColumnBuilder::with_capacity(f.ty.clone(), n);
    let mut ps = 0usize;
    while ps < n {
        let pe = p.part_end(ps);
        for i in ps..pe {
            let src = if back {
                i.checked_sub(off).filter(|t| *t >= ps)
            } else {
                i.checked_add(off).filter(|t| *t < pe)
            };
            match src {
                Some(t) => b.push_value(&val.value(t))?,
                None => match &dflt {
                    Some(d) => b.push_value(&d.value(i))?,
                    None => b.push_null(),
                },
            }
        }
        ps = pe;
    }
    Ok(b.finish())
}

/// `first_value` / `last_value` / `nth_value`: a positional gather inside the
/// **frame**, which is what makes `last_value(x) OVER (ORDER BY k)` return the
/// current row rather than the partition's last -- the single most reported
/// surprise in every engine that has these.
fn eval_pick(f: &BoundWindow, p: &Pass<'_>) -> Result<Column> {
    let (n, flags) = (p.block.rows(), p.flags);
    let val = one_col(&f.args[0], p.block)?;
    let mut b = ColumnBuilder::with_capacity(f.ty.clone(), n);
    let mut ps = 0usize;
    while ps < n {
        let pe = p.part_end(ps);
        let mut gs = ps;
        let mut ge = peer_end(flags, ps, pe);
        for i in ps..pe {
            if i == ge {
                gs = ge;
                ge = peer_end(flags, gs, pe);
            }
            let (s, e) = frame_of(&f.frame, i, ps, pe, gs, ge);
            let src = match f.kind {
                WindowKind::FirstValue => (s < e).then_some(s),
                WindowKind::LastValue => (s < e).then(|| e - 1),
                // `nth_value(x, k)` counts from the frame start, 1-based.
                _ => s.checked_add(f.offset as usize - 1).filter(|t| *t < e),
            };
            match src {
                Some(t) => b.push_value(&val.value(t))?,
                None => b.push_null(),
            }
        }
        ps = pe;
    }
    Ok(b.finish())
}

/// An aggregate over the frame, folded by whichever sweep the frame allows.
fn eval_agg(
    f: &BoundWindow,
    af: &'static AggFn,
    p: &Pass<'_>,
    scratch: &mut Vec<Value>,
) -> Result<Column> {
    let (n, flags, ids, block) = (p.block.rows(), p.flags, p.ids, p.block);
    // Owned, because `Accumulator::update` wants a contiguous `&[Column]` and
    // there is no borrowed spelling of that. One clone of the argument columns
    // per query, not per row.
    let args = expr::eval_all(&f.args, block)?;
    let tys: Vec<DataType> = f.args.iter().map(|a| a.ty()).collect();
    let sweep = Sweep::of(&f.frame, p.ordered);

    let mut b = ColumnBuilder::with_capacity(f.ty.clone(), n);
    let mut ps = 0usize;
    while ps < n {
        // Once per *partition*, not once per row: the three linear sweeps are
        // fast enough that a per-row check would be the dominant cost, but a
        // `Refold` over a single wide partition is the one thing in this
        // operator that can run long enough for a user to want it stopped.
        p.ctx.check()?;
        let pe = p.part_end(ps);
        match sweep {
            Sweep::Whole => {
                let mut acc = (af.new)(&tys, &f.params)?;
                acc.update(&args, &ids[ps..pe])?;
                let v = acc.finish();
                for _ in ps..pe {
                    b.push_value(&v)?;
                }
            }
            Sweep::Forward => {
                // The accumulator's contents are exactly `[ps, fed)`, and the
                // frame is exactly `[ps, e)` -- so `finish()` is the answer
                // with no bookkeeping, including for an empty frame, where a
                // fresh accumulator already reports the empty-input value.
                let mut acc = (af.new)(&tys, &f.params)?;
                let mut fed = ps;
                let mut gs = ps;
                let mut ge = peer_end(flags, ps, pe);
                for i in ps..pe {
                    if i == ge {
                        gs = ge;
                        ge = peer_end(flags, gs, pe);
                    }
                    let (_, e) = frame_of(&f.frame, i, ps, pe, gs, ge);
                    if e > fed {
                        acc.update(&args, &ids[fed..e])?;
                        fed = e;
                    }
                    b.push_value(&acc.finish())?;
                }
            }
            Sweep::Backward => {
                let mut acc = (af.new)(&tys, &f.params)?;
                let mut fed = pe;
                let mut ge = pe;
                let mut gs = peer_start(flags, pe, ps);
                // Answers arrive last-row-first, so they are stacked and then
                // read back reversed. `push` onto a cleared buffer rather than
                // `resize` + indexed store: the latter writes every slot twice,
                // once with a placeholder and once with the answer.
                scratch.clear();
                for i in (ps..pe).rev() {
                    if i < gs {
                        ge = gs;
                        gs = peer_start(flags, ge, ps);
                    }
                    let (s, _) = frame_of(&f.frame, i, ps, pe, gs, ge);
                    if s < fed {
                        acc.update(&args, &ids[s..fed])?;
                        fed = s;
                    }
                    scratch.push(acc.finish());
                }
                for v in scratch.iter().rev() {
                    b.push_value(v)?;
                }
            }
            Sweep::Refold => {
                let mut gs = ps;
                let mut ge = peer_end(flags, ps, pe);
                for i in ps..pe {
                    if i == ge {
                        gs = ge;
                        ge = peer_end(flags, gs, pe);
                    }
                    let (s, e) = frame_of(&f.frame, i, ps, pe, gs, ge);
                    let mut acc = (af.new)(&tys, &f.params)?;
                    if s < e {
                        acc.update(&args, &ids[s..e])?;
                    }
                    b.push_value(&acc.finish())?;
                }
            }
        }
        ps = pe;
    }
    Ok(b.finish())
}

/// Evaluate one expression, borrowing the block's column when it is a bare
/// reference -- which for `lag(x, 1)` it always is.
fn one_col<'b>(e: &BoundExpr, block: &'b Block) -> Result<Cow<'b, Column>> {
    let mut v = expr::eval_all_cow(std::slice::from_ref(e), block)?;
    Ok(v.remove(0))
}

// ------------------------------------------------------------------- schema

/// The schema a window step produces: its input, then one field per function.
pub fn output_schema(input: &Schema, funcs: &[BoundWindow]) -> Schema {
    let mut fields = input.fields().to_vec();
    fields.extend(funcs.iter().map(|f| Field::new(f.name.clone(), f.ty.clone())));
    Schema::new_unchecked(fields)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------- through the engine
    //
    // The unit tests above pin the arithmetic; these pin that the operator is
    // reachable, streams correctly across block boundaries, and stops when
    // told. Correctness against an independent oracle lives in `tests/window.rs`.

    fn session(rows: usize) -> crate::Session {
        let mut s = crate::Session::in_memory();
        s.execute("CREATE TABLE t (id Int64, g Int64, v Int64) ENGINE = MergeTree ORDER BY id")
            .unwrap();
        let mut sql = String::from("INSERT INTO t VALUES ");
        for i in 0..rows {
            if i > 0 {
                sql.push(',');
            }
            // Three partitions, each far wider than BLOCK_SIZE, and their seams
            // deliberately not on a block boundary.
            let _ = std::fmt::Write::write_fmt(
                &mut sql,
                format_args!("({i},{},{})", i / 7000, i % 5),
            );
        }
        s.execute(&sql).unwrap();
        s
    }

    fn u64s(s: &mut crate::Session, sql: &str) -> Vec<u64> {
        s.query(sql)
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .to_values()
            .iter()
            .map(|r| r[0].as_u64().expect("a numeric column"))
            .collect()
    }

    #[test]
    fn a_partition_that_straddles_many_blocks_is_still_one_partition() {
        const N: usize = 20_000;
        let mut s = session(N);
        // `BLOCK_SIZE` is 8192 and the partitions are 7000 wide, so every seam
        // falls inside a block and one partition spans three of them. A
        // per-block reset -- the obvious way to get this wrong -- would show up
        // as row_number restarting at a block boundary.
        let got = u64s(&mut s, "SELECT row_number() OVER (PARTITION BY g ORDER BY id) FROM t");
        assert_eq!(got.len(), N);
        for (i, r) in got.iter().enumerate() {
            assert_eq!(*r as usize, i % 7000 + 1, "row {i}");
        }
        // And the running total agrees with the closed form: v cycles 0..4.
        let sums = u64s(
            &mut s,
            "SELECT sum(v) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) \
             FROM t",
        );
        let mut want = 0u64;
        for (i, got) in sums.iter().enumerate() {
            want += (i % 5) as u64;
            assert_eq!(*got, want, "row {i}");
        }
    }

    #[test]
    fn a_window_reports_the_scan_counters_of_its_input() {
        // The operator forwards `stats`, so zone-map pruning under a window is
        // still visible. A window that swallowed them would make every window
        // query look like a full scan in the summary line.
        let mut s = session(100);
        let rs = s.query("SELECT rank() OVER (ORDER BY v) FROM t").unwrap();
        assert!(rs.stats.rows_scanned > 0, "the scan counters did not reach the top");
    }

    #[test]
    fn a_cancelled_window_stops_instead_of_folding_the_whole_relation() {
        use crate::planner::{binder::Binder, optimizer};
        use std::sync::atomic::Ordering;

        let mut s = session(20_000);
        // Bind and lower by hand, because cancellation is a property of the
        // context and `Session::query` owns its own.
        s.execute("SYSTEM FLUSH").unwrap();
        let stmt = crate::sql::parse(
            "SELECT sum(v) OVER (ORDER BY id ROWS BETWEEN 200 PRECEDING AND CURRENT ROW) FROM t",
        )
        .unwrap();
        let q = match &stmt[0] {
            crate::sql::ast::Statement::Query(q) => q.clone(),
            other => panic!("{other:?}"),
        };
        let cat = &s.catalog;
        let plan = optimizer::optimize(Binder::new(cat).bind_query(&q).unwrap()).unwrap();
        let ctx = QueryContext::new();
        ctx.cancel.store(true, Ordering::Relaxed);
        let e = super::super::execute_ctx(&plan, cat, &ctx).unwrap_err().to_string();
        assert!(e.contains("cancelled"), "{e}");
    }

    #[test]
    fn lookup_finds_the_ranking_family_and_falls_through_to_aggregates() {
        for n in [
            "row_number",
            "ROW_NUMBER",
            "rank",
            "dense_rank",
            "percent_rank",
            "cume_dist",
            "ntile",
            "lag",
            "lead",
            "first_value",
            "last_value",
            "nth_value",
        ] {
            assert!(lookup(n).is_some(), "missing window function {n}");
        }
        // The aggregate library is reachable unchanged -- this is the whole
        // point of `WindowKind::Agg`, and a regression here would mean a second
        // `sum` had appeared somewhere.
        assert!(matches!(lookup("sum"), Some(WindowKind::Agg(f)) if f.name == "sum"));
        assert!(matches!(lookup("quantileExact"), Some(WindowKind::Agg(_))));
        assert!(lookup("nosuchwindowfn").is_none());
    }

    #[test]
    fn ntile_splits_evenly_with_the_remainder_at_the_front() {
        let buckets = |size: usize, k: u64| -> Vec<u64> {
            (0..size).map(|j| ntile(j, size, k)).collect()
        };
        assert_eq!(buckets(5, 2), vec![1, 1, 1, 2, 2]);
        assert_eq!(buckets(6, 3), vec![1, 1, 2, 2, 3, 3]);
        assert_eq!(buckets(7, 3), vec![1, 1, 1, 2, 2, 3, 3]);
        // More buckets than rows: one row each, tail buckets stay empty.
        assert_eq!(buckets(2, 5), vec![1, 2]);
        assert_eq!(buckets(1, 1), vec![1]);
    }

    fn frame(units: FrameUnits, start: FrameBound, end: FrameBound) -> WindowFrame {
        WindowFrame { units, start, end }
    }

    #[test]
    fn rows_frames_clamp_to_the_partition() {
        // Partition [10, 15), row 12, peer group [12, 13).
        let f = frame(FrameUnits::Rows, FrameBound::Preceding(2), FrameBound::Following(1));
        assert_eq!(frame_of(&f, 12, 10, 15, 12, 13), (10, 14));
        let f = frame(FrameUnits::Rows, FrameBound::Preceding(9), FrameBound::Preceding(1));
        assert_eq!(frame_of(&f, 12, 10, 15, 12, 13), (10, 12));
        // Entirely ahead of the partition end: empty, not clamped to one row.
        let f = frame(FrameUnits::Rows, FrameBound::Following(5), FrameBound::Following(9));
        assert_eq!(frame_of(&f, 12, 10, 15, 12, 13), (15, 15));
        // Entirely behind the first row: also empty.
        let f = frame(FrameUnits::Rows, FrameBound::UnboundedPreceding, FrameBound::Preceding(1));
        assert_eq!(frame_of(&f, 10, 10, 15, 10, 11), (10, 10));
    }

    #[test]
    fn range_current_row_reaches_the_end_of_the_peer_group() {
        // This is the difference that makes RANGE worth implementing: under
        // ROWS the frame stops at the row itself, under RANGE it runs through
        // every row tied with it.
        let rows = frame(FrameUnits::Rows, FrameBound::UnboundedPreceding, FrameBound::CurrentRow);
        let rng = frame(FrameUnits::Range, FrameBound::UnboundedPreceding, FrameBound::CurrentRow);
        assert_eq!(frame_of(&rows, 3, 0, 10, 2, 6), (0, 4));
        assert_eq!(frame_of(&rng, 3, 0, 10, 2, 6), (0, 6));
    }

    #[test]
    fn sweep_picks_the_linear_path_for_every_default_frame() {
        let whole = frame(
            FrameUnits::Range,
            FrameBound::UnboundedPreceding,
            FrameBound::UnboundedFollowing,
        );
        assert_eq!(Sweep::of(&whole, true), Sweep::Whole);
        let running =
            frame(FrameUnits::Range, FrameBound::UnboundedPreceding, FrameBound::CurrentRow);
        assert_eq!(Sweep::of(&running, true), Sweep::Forward);
        // Unordered RANGE ... CURRENT ROW is the whole partition, and must not
        // be walked row by row to discover that.
        assert_eq!(Sweep::of(&running, false), Sweep::Whole);
        let tail = frame(
            FrameUnits::Rows,
            FrameBound::CurrentRow,
            FrameBound::UnboundedFollowing,
        );
        assert_eq!(Sweep::of(&tail, true), Sweep::Backward);
        let sliding =
            frame(FrameUnits::Rows, FrameBound::Preceding(2), FrameBound::CurrentRow);
        assert_eq!(Sweep::of(&sliding, true), Sweep::Refold);
    }

    #[test]
    fn only_a_frame_that_excludes_its_own_row_is_nullable() {
        let f = frame(FrameUnits::Rows, FrameBound::Preceding(1), FrameBound::CurrentRow);
        assert!(!can_be_empty(&f));
        let f = frame(
            FrameUnits::Range,
            FrameBound::UnboundedPreceding,
            FrameBound::UnboundedFollowing,
        );
        assert!(!can_be_empty(&f));
        let f = frame(FrameUnits::Rows, FrameBound::Following(1), FrameBound::Following(3));
        assert!(can_be_empty(&f));
        let f = frame(FrameUnits::Rows, FrameBound::UnboundedPreceding, FrameBound::Preceding(1));
        assert!(can_be_empty(&f));
    }

    fn flags_of(cols: &[&Column], n: usize) -> Vec<u8> {
        let mut flags = vec![0u8; n];
        flags[0] = NEW_PART | NEW_PEER;
        for c in cols {
            mark_changes(c, NEW_PART | NEW_PEER, &mut flags);
        }
        flags
    }

    #[test]
    fn boundaries_land_on_every_change_and_nowhere_else() {
        let c = Column::i64s(DataType::Int64, vec![1, 1, 2, 2, 2, 3]);
        assert_eq!(flags_of(&[&c], 6), vec![3, 0, 3, 0, 0, 3]);

        let s = Column::strs(
            DataType::String,
            vec!["a".into(), "a".into(), "b".into()],
        );
        assert_eq!(flags_of(&[&s], 3), vec![3, 0, 3]);
    }

    #[test]
    fn adjacent_nulls_are_one_partition_not_two() {
        // The trap: the data slot under a NULL is a placeholder, so comparing
        // the raw slice would split on whatever happened to be stored there.
        use crate::types::ColumnBuilder;
        let mut b = ColumnBuilder::new(DataType::Nullable(Box::new(DataType::Int64)));
        b.push_null();
        b.push_null();
        b.push_value(&Value::Int(7)).unwrap();
        b.push_null();
        let c = b.finish();
        assert_eq!(flags_of(&[&c], 4), vec![3, 0, 3, 3]);
    }

    #[test]
    fn float_keys_group_nan_with_nan_and_the_two_zeroes_together() {
        let c = Column::f64s(DataType::Float64, vec![0.0, -0.0, f64::NAN, f64::NAN, 1.0]);
        assert_eq!(flags_of(&[&c], 5), vec![3, 0, 3, 0, 3]);
    }
}
