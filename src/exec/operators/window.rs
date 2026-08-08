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
//! ## The unit is the partition, not the relation
//!
//! This operator used to drain its whole input into one block before it could
//! answer anything, so a window over a relation larger than the budget was
//! refused -- not slow, refused. That was the wrong unit. Rows arrive sorted by
//! `(partition keys, ORDER BY keys)`, so **partitions arrive contiguous and in
//! full**, and a partition that has ended can be computed and handed out while
//! the next one is still being read.
//!
//! So the operator buffers, and when the budget objects it computes every
//! *complete* partition in the buffer, emits them, and keeps only the trailing
//! partial one. Its bound is `max(largest partition, budget)` rather than the
//! relation, and a `PARTITION BY` over a hundred keys is a hundredth of the
//! footprint it used to be.
//!
//! Nothing is written to disk, and that is the point rather than an omission:
//! the natural spill unit is the partition, and once the operator works a
//! partition at a time there is nothing left to write. Rows that have been
//! computed leave immediately, and rows that have not been read yet are still
//! in the operator below.
//!
//! **A single partition larger than the budget** -- `sum(x) OVER ()` over a
//! huge table, or one key holding most of the rows -- has no smaller unit, and
//! it is refused with the message naming the window buffer. Spilling it would
//! not help: every sweep here needs random access to the partition's rows
//! (`lag` reaches backwards, `last_value` forwards, `Refold` re-reads an
//! arbitrary range), so a disk-resident partition needs a second, external
//! implementation of every window function rather than a buffer swap. What is
//! fixed is the case that was actually common -- a large *relation* of ordinary
//! partitions -- and what is left refused is the case that is genuinely hard.
//!
//! The boundary pass moved with it: [`mark_changes`] now runs per input block
//! instead of once over the concatenation, with the seam between two blocks
//! compared through the previous block's last key values. Same total work, and
//! it is what lets the operator know where the last partition starts without
//! having ever held the whole relation.
//!
//! Cutting costs nothing measurable. 1M rows, `sum(v) OVER (PARTITION BY g
//! ORDER BY id)` end to end, best-of-6, cut forced with `GRANULAR_SPILL_ROWS`:
//! one chunk 581.5 ms, six chunks 576.3 ms, twenty-one chunks 527.0 ms. The
//! last one is not a typo and not a claim -- it is inside the noise -- but it
//! does say that the two `Block::slice` copies a cut pays are nothing against
//! the work, and that smaller working sets are not obviously worse.
//!
//! ## Partitions are independent, so the fan-out is free
//!
//! One `PARTITION BY` key means one self-contained problem, and a chunk holding
//! many of them is embarrassingly parallel: worker `k` takes a contiguous,
//! partition-aligned range of rows and computes *every* window function over
//! it, and the columns are concatenated in range order. There is no merge, no
//! partial state and nothing recomputed, so the result is **bit-identical** to
//! the serial one -- including a float `sum`, because each partition is still
//! folded by exactly one accumulator in exactly one order. That is a stronger
//! guarantee than the exchange can make about a parallel aggregate, and it is
//! purely a property of the split being by partition.
//!
//! The split is balanced by *rows* rather than by partitions ([`split_partitions`]),
//! because partition sizes are the one thing a `PARTITION BY` says nothing
//! about. The argument columns are evaluated once for the whole chunk, outside
//! the fan-out, since an accumulator is fed absolute row ids and needs no slice
//! of its own.
//!
//! Measured on 14 cores, 1M rows in 40 integer partitions, `GRANULAR_THREADS`
//! stepped through the pool sizes, best-of-6. This is the **window step
//! alone**, timed inside [`compute`] by a temporary `eprintln` since removed,
//! because the query around it is dominated by something else -- see below.
//!
//! ```text
//!   threads                              1      2      4      8     14
//!   sum(v) OVER (PARTITION BY g       17.7   11.1    7.6    5.6    7.5  ms
//!                ORDER BY id)         1.0x   1.6x   2.3x   3.2x   2.4x
//!   ... ROWS BETWEEN 100 PRECEDING   325.9  232.1  157.8  111.2   71.2  ms
//!                AND CURRENT ROW      1.0x   1.4x   2.1x   2.9x   4.6x
//! ```
//!
//! 56 M rows/s serial and 179 M rows/s at eight threads for the running total;
//! 3.1 M rows/s serial and 14 M rows/s at fourteen for the sliding frame. The
//! cheap window stops scaling at eight because a 17 ms step cannot amortize
//! fourteen rendezvous; the expensive one scales to the end of the machine,
//! which is the shape that needed the threads.
//!
//! **The query around it does not move**, and that is worth stating rather
//! than hiding: `exchange::build` has no arm for `PhysicalPlan::Window`, so the
//! `Sort` the binder puts *underneath* every window is built by the serial
//! builder. On the 1M-row query above that sort is ~470 ms against a 17 ms
//! window step -- 96% of the query, single-threaded, for the want of one match
//! arm. Parallelizing this operator was still the right half to do (it is the
//! half that scales with the frame), but until that arm exists a user will not
//! see it.
//!
//! **A window with no `PARTITION BY` is one partition**, [`split_partitions`]
//! returns one range, and the step stays serial. That is the honest answer, not
//! a missing case: `sum(v) OVER (ORDER BY id)` is a running total, and a
//! prefix scan can only be split by folding each chunk twice -- once to get its
//! partial, once again seeded with the partials before it. `Accumulator` has
//! `merge`, so it could be built; it doubles the work to divide it by the
//! thread count, which is a win only above four or so threads and only for the
//! two linear sweeps (`Whole` and `Forward`). It is a separate task with its
//! own measurement, and pretending otherwise by splitting a single partition
//! would silently break `rank` at the seams.
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

use crate::common::{pool, Error, Result};
use crate::exec::expr;
use crate::exec::functions::{aggregate, AggFn};
use crate::planner::logical::BoundExpr;
use crate::sql::ast::{FrameBound, FrameUnits, WindowFrame};
use crate::types::{Block, Column, ColumnBuilder, ColumnData, DataType, Field, Schema, Value};

use super::{chunk, MemGuard, Operator, QueryContext, ScanStats};

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

/// Rows below which a window step stays single-threaded.
///
/// Assembling the fleet costs a rendezvous with parked workers plus one
/// `ColumnBuilder` per function per worker, and below some size the step is
/// over before that has happened. Measured with the window step timed on its
/// own, 14 cores, a 100-wide sliding frame (the shape with enough work per row
/// to be worth splitting at all), 40 partitions:
///
/// ```text
///   rows     workers   step
///    2 000      1      0.39 ms
///    4 000      1      1.02 ms
///    4 096      2      1.35 ms
///    8 000      3      1.04 ms
///  100 000     14      6.45 ms
/// ```
///
/// 4096 rows is where the fan-out starts and 8000 is where it has already paid
/// for itself. Being wrong toward "stay serial" costs a fraction of a query
/// that was already under a millisecond; being wrong toward "go parallel" taxes
/// every small window in an HTAP workload. The floor is a quarter of the
/// exchange's 16 384 because a window row costs several times what a scanned
/// row does -- there is more work per row to amortize the same fixed cost
/// against.
const MIN_PARALLEL_ROWS: usize = 4 << 10;

/// Rows one worker should get before another is woken.
const MIN_ROWS_PER_WORKER: usize = 2 << 10;

pub struct Window<'a> {
    input: Box<dyn Operator + 'a>,
    node: &'a WindowNode,
    ctx: &'a QueryContext,
    /// Rows read but not yet computed. `None` until the first block arrives,
    /// so the first `extend` is a move rather than `Block::extend`'s clone.
    buf: Option<Block>,
    /// One byte per buffered row; see the module docs. Extended block by
    /// block, which is why the seam below exists.
    flags: Vec<u8>,
    /// The partition keys then the ORDER BY keys of the buffer's last row.
    /// Comparing the next block's first row against these is what lets the
    /// boundary pass run per input block instead of over the concatenation.
    seam: Vec<Value>,
    /// First row of the trailing, possibly incomplete partition -- the only
    /// place the buffer may be cut. Maintained as blocks arrive rather than
    /// found by scanning back, which on a single-partition window would be a
    /// backwards walk of the whole buffer per block.
    part_start: usize,
    /// Reversed, so `next` is a `pop`.
    out: Vec<Block>,
    /// `0..rows`, grown once and shared by every function *and* every worker:
    /// a frame fold is `update(args, &ids[s..e])` with no allocation.
    ids: Vec<u32>,
    eof: bool,
    guard: MemGuard,
    forced: usize,
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
            buf: None,
            flags: Vec::new(),
            seam: Vec::new(),
            part_start: 0,
            out: Vec::new(),
            ids: Vec::new(),
            eof: false,
            guard: MemGuard::new(ctx, "a window function's row buffer"),
            forced: super::sort::forced_spill_rows(),
        }
    }

    /// Read input until there is something to hand out.
    fn pump(&mut self) -> Result<()> {
        loop {
            self.ctx.check()?;
            let Some(b) = self.input.next()? else {
                self.eof = true;
                let n = self.flags.len();
                if n > 0 {
                    self.emit(n)?;
                }
                return Ok(());
            };
            if b.rows() == 0 {
                continue;
            }
            self.absorb(b)?;
            // One `grow_to` per block, exactly as the old `drain` made, and one
            // `usize` compare for the test knob. The `Err` is a signal rather
            // than an error: a relation that does not fit is a query that
            // emits earlier, not one that is refused.
            let n = self.flags.len();
            let over = (self.forced != 0 && n >= self.forced)
                || self.guard.grow_to(self.footprint()).is_err();
            if over {
                if self.part_start == 0 {
                    // The buffer is one partition and there is no smaller unit
                    // to emit -- see the module docs. Charge it for real so the
                    // refusal names the window buffer rather than happening
                    // later somewhere else.
                    self.guard.grow_to(self.footprint())?;
                } else {
                    self.emit(self.part_start)?;
                    return Ok(());
                }
            }
        }
    }

    /// Append one input block and extend the partition/peer bits over it.
    fn absorb(&mut self, b: Block) -> Result<()> {
        let base = self.flags.len();
        let n = b.rows();
        self.flags.resize(base + n, 0);
        if self.ids.len() < base + n {
            let from = self.ids.len() as u32;
            self.ids.extend(from..(base + n) as u32);
        }

        // One pass per key column over *this block*, dispatched on physical
        // type outside the row loop. Partition keys set both bits at once: a
        // new partition is necessarily a new peer group, and folding that in
        // here saves a second sweep over `flags`.
        let pk = expr::eval_all_cow(&self.node.partition, &b)?;
        let ok = expr::eval_all_cow(&self.node.order, &b)?;
        if base == 0 {
            self.flags[0] = NEW_PART | NEW_PEER;
        } else {
            // The seam `mark_changes` cannot see, because it compares adjacent
            // rows of one column and the two rows are in different blocks.
            let np = pk.len();
            let mut bits = 0u8;
            for (i, c) in pk.iter().enumerate() {
                if c.as_ref().value(0) != self.seam[i] {
                    bits |= NEW_PART | NEW_PEER;
                }
            }
            for (i, c) in ok.iter().enumerate() {
                if c.as_ref().value(0) != self.seam[np + i] {
                    bits |= NEW_PEER;
                }
            }
            self.flags[base] |= bits;
        }
        for c in &pk {
            mark_changes(c.as_ref(), NEW_PART | NEW_PEER, &mut self.flags[base..]);
        }
        for c in &ok {
            mark_changes(c.as_ref(), NEW_PEER, &mut self.flags[base..]);
        }
        // Walked backwards and stopped at the first hit: a block that opens no
        // partition costs a full backwards scan of *itself*, never of the
        // buffer, so maintaining this is linear in the input either way.
        for i in (base..base + n).rev() {
            if self.flags[i] & NEW_PART != 0 {
                self.part_start = i;
                break;
            }
        }
        self.seam.clear();
        for c in pk.iter().chain(&ok) {
            self.seam.push(c.as_ref().value(n - 1));
        }
        drop((pk, ok));

        match &mut self.buf {
            None => self.buf = Some(b),
            Some(a) => a.extend(&b)?,
        }
        Ok(())
    }

    /// What the operator is holding: the buffered rows, the byte of flags and
    /// four of row id each, and any computed output not yet collected.
    fn footprint(&self) -> usize {
        self.buf.as_ref().map_or(0, |b| b.bytes())
            + self.flags.len() * 5
            + self.out.iter().map(|b| b.bytes()).sum::<usize>()
    }

    /// Compute the first `cut` rows -- always a whole number of partitions --
    /// and keep the rest.
    fn emit(&mut self, cut: usize) -> Result<()> {
        let buf = self.buf.take().expect("emit with an empty buffer");
        let n = buf.rows();
        debug_assert!(cut <= n && cut > 0);
        // The whole buffer is the overwhelmingly common case (every window
        // that fits takes it), and it must not copy: the two slices below are
        // a full copy of the rows and only a spilling window should pay them.
        let (head, tail) = if cut == n {
            (buf, None)
        } else {
            (buf.slice(0, cut), Some(buf.slice(cut, n)))
        };
        self.out = compute(self.node, self.ctx, head, &self.flags[..cut], &self.ids)?;
        // Handed out back to front; see the `out` field.
        self.out.reverse();

        self.flags.drain(..cut);
        // Saturating because the end-of-stream call cuts *past* the trailing
        // partition's start -- there is no trailing partition left to point at.
        self.part_start = self.part_start.saturating_sub(cut);
        self.buf = tail;
        if self.buf.is_some() {
            // `MemGuard` only ever grows, so a streaming operator that reused
            // one would hold its high-water mark for ever and never feel
            // pressure again -- it would silently buffer the whole relation
            // after the first cut. Replacing it is the release.
            self.guard = MemGuard::new(self.ctx, "a window function's row buffer");
            self.guard.grow_to(self.footprint())?;
        }
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
        loop {
            if let Some(b) = self.out.pop() {
                return Ok(Some(b));
            }
            if self.eof {
                return Ok(None);
            }
            self.pump()?;
        }
    }
}

// --------------------------------------------------------------- fan-out

/// Compute every window function over one partition-aligned block of rows.
fn compute(
    node: &WindowNode,
    ctx: &QueryContext,
    mut block: Block,
    flags: &[u8],
    ids: &[u32],
) -> Result<Vec<Block>> {
    let n = block.rows();
    if n == 0 {
        return Ok(Vec::new());
    }
    let cols = {
        // Evaluated once for the whole block, *outside* the fan-out: an
        // argument column is indexed by absolute row id, so a worker needs no
        // slice of its own and evaluating per worker would repeat the whole
        // expression k times over the whole block.
        let ordered = !node.order.is_empty();
        let prep: Vec<Prepared> = node
            .funcs
            .iter()
            .map(|f| Prepared::of(f, &block, ordered))
            .collect::<Result<_>>()?;
        let pass = Pass { flags, ids, ctx };

        let w = degree(n);
        let ranges = if w > 1 { split_partitions(flags, n, w) } else { Vec::new() };
        if ranges.len() < 2 {
            // Serial: one range, and no `pool` round trip at all.
            let mut scratch: Vec<Value> = Vec::new();
            let mut cols = Vec::with_capacity(node.funcs.len());
            for (f, p) in node.funcs.iter().zip(&prep) {
                ctx.check()?;
                cols.push(eval_window(f, p, &pass, 0, n, &mut scratch)?);
            }
            cols
        } else {
            // Partitions are independent, so this is a pure fan-out with no
            // merge: worker `k` computes every function over its own
            // contiguous, partition-aligned row range, and the answers are
            // concatenated in range order. Nothing is recomputed and no value
            // depends on the split, so the result is bit-identical to the
            // serial one -- including a float `sum`, because each partition is
            // still folded by exactly one accumulator in one order.
            let parts: Vec<Result<Vec<Column>>> = pool::global().map(ranges.len(), |k| {
                let (lo, hi) = ranges[k];
                let mut scratch: Vec<Value> = Vec::new();
                let mut cols = Vec::with_capacity(node.funcs.len());
                for (f, p) in node.funcs.iter().zip(&prep) {
                    ctx.check()?;
                    cols.push(eval_window(f, p, &pass, lo, hi, &mut scratch)?);
                }
                Ok(cols)
            });
            let mut it = parts.into_iter();
            let mut cols = it.next().expect("at least two ranges")?;
            for other in it {
                for (a, b) in cols.iter_mut().zip(other?) {
                    a.extend(&b)?;
                }
            }
            cols
        }
    };
    block.append_columns(cols)?;
    Ok(chunk(block))
}

/// How many workers this many rows justifies. 1 means "stay serial", and the
/// caller treats it as a refusal rather than a one-wide fleet.
fn degree(rows: usize) -> usize {
    if rows < MIN_PARALLEL_ROWS {
        return 1;
    }
    pool::global().threads().min(rows / MIN_ROWS_PER_WORKER).max(1)
}

/// Cut `[0, n)` into at most `w` contiguous ranges that start on a partition
/// boundary and hold roughly equal numbers of *rows*.
///
/// Rows and not partitions, because partition sizes are the one thing a
/// `PARTITION BY` says nothing about: a key with 90% of the rows would give one
/// worker 90% of the work under an equal-partition split, and the whole step
/// would run at serial speed with extra threads watching.
///
/// A window with no `PARTITION BY` is one partition and comes back as a single
/// range, i.e. serial -- see the module docs for why that is the honest answer
/// rather than a missing case.
fn split_partitions(flags: &[u8], n: usize, w: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(w);
    let mut s = 0usize;
    for k in 1..w {
        // Walk forward from the ideal cut to the next partition start. Each
        // walk covers disjoint rows, so the whole split is one pass.
        let mut e = (n * k / w).max(s + 1);
        while e < n && flags[e] & NEW_PART == 0 {
            e += 1;
        }
        if e >= n {
            break;
        }
        out.push((s, e));
        s = e;
    }
    out.push((s, n));
    out
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

/// Everything computed once for one processed chunk and then read by every
/// function *and* every worker: the partition/peer bits, the shared row-id
/// vector and the query's stop conditions.
///
/// One struct rather than three parameters because it is genuinely one thing --
/// the state of a single pass -- and because k window functions across w
/// workers share exactly this and nothing else.
struct Pass<'p> {
    flags: &'p [u8],
    /// `0..rows`, so a frame fold is `update(args, &ids[s..e])` with no
    /// allocation and no index materialization.
    ids: &'p [u32],
    ctx: &'p QueryContext,
}

impl Pass<'_> {
    /// End (exclusive) of the partition starting at `ps`, bounded by the
    /// worker's own range. Ranges are partition-aligned so the bound never
    /// actually bites; it is there so that a range is a self-contained unit and
    /// no worker can read past its own rows.
    #[inline]
    fn part_end(&self, ps: usize, hi: usize) -> usize {
        let mut pe = ps + 1;
        while pe < hi && self.flags[pe] & NEW_PART == 0 {
            pe += 1;
        }
        pe
    }
}

/// One window call's row-independent inputs, evaluated once for the whole
/// chunk and shared by every worker.
///
/// The split exists for the fan-out: an argument column is indexed by absolute
/// row id, so a worker that evaluated `f.args` itself would repeat the whole
/// expression over the whole block once per thread and then use a slice of it.
enum Prepared<'b> {
    Rank,
    Shift { val: Cow<'b, Column>, dflt: Option<Cow<'b, Column>> },
    Pick { val: Cow<'b, Column> },
    Agg { args: Vec<Column>, tys: Vec<DataType>, sweep: Sweep },
}

impl<'b> Prepared<'b> {
    fn of(f: &BoundWindow, block: &'b Block, ordered: bool) -> Result<Prepared<'b>> {
        Ok(match f.kind {
            WindowKind::Agg(_) => Prepared::Agg {
                // Owned, because `Accumulator::update` wants a contiguous
                // `&[Column]` and there is no borrowed spelling of that. One
                // clone of the argument columns per query, not per row.
                args: expr::eval_all(&f.args, block)?,
                tys: f.args.iter().map(|a| a.ty()).collect(),
                sweep: Sweep::of(&f.frame, ordered),
            },
            WindowKind::Lag | WindowKind::Lead => Prepared::Shift {
                val: one_col(&f.args[0], block)?,
                dflt: match f.args.get(2) {
                    Some(e) => Some(one_col(e, block)?),
                    None => None,
                },
            },
            WindowKind::FirstValue | WindowKind::LastValue | WindowKind::NthValue => {
                Prepared::Pick { val: one_col(&f.args[0], block)? }
            }
            _ => Prepared::Rank,
        })
    }
}

/// Compute one window function's output column over the rows `[lo, hi)`, which
/// must start and end on a partition boundary.
fn eval_window(
    f: &BoundWindow,
    prep: &Prepared<'_>,
    p: &Pass<'_>,
    lo: usize,
    hi: usize,
    scratch: &mut Vec<Value>,
) -> Result<Column> {
    match (f.kind, prep) {
        (WindowKind::Agg(af), Prepared::Agg { args, tys, sweep }) => {
            eval_agg(f, af, args, tys, *sweep, p, lo, hi, scratch)
        }
        (_, Prepared::Shift { val, dflt }) => eval_shift(f, val, dflt.as_deref(), p, lo, hi),
        (_, Prepared::Pick { val }) => eval_pick(f, val, p, lo, hi),
        (_, Prepared::Rank) => Ok(eval_rank(f, p, lo, hi)),
        _ => unreachable!("Prepared::of and eval_window disagree about a window kind"),
    }
}

/// `row_number`, `rank`, `dense_rank`, `percent_rank`, `cume_dist`, `ntile` --
/// everything that reads only the ordering.
fn eval_rank(f: &BoundWindow, p: &Pass<'_>, lo: usize, hi: usize) -> Column {
    let flags = p.flags;
    let mut b = ColumnBuilder::with_capacity(f.ty.clone(), hi - lo);
    let mut ps = lo;
    while ps < hi {
        let pe = p.part_end(ps, hi);
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

/// Every value of `c` cast to `ty`, once, so a per-row gather can push lanes
/// without asking what scale they were.
///
/// Only reached when the types genuinely differ; see the note in `eval_shift`.
fn cast_column(c: &Column, ty: &DataType) -> Result<Column> {
    let mut b = ColumnBuilder::with_capacity(ty.clone(), c.len());
    for i in 0..c.len() {
        match c.value(i) {
            Value::Null => b.push_null(),
            v => b.push_value(&v.cast_to(ty)?)?,
        }
    }
    Ok(b.finish())
}

/// `lag` / `lead`: a positional gather inside the partition, with a default for
/// the rows that fall off the end. The frame is ignored, per SQL.
fn eval_shift(
    f: &BoundWindow,
    val: &Column,
    dflt: Option<&Column>,
    p: &Pass<'_>,
    lo: usize,
    hi: usize,
) -> Result<Column> {
    let back = matches!(f.kind, WindowKind::Lag);
    let off = f.offset as usize;

    // `f.ty` is `promote(value, default)`, so either input can be narrower than
    // the output -- and for a decimal that difference is a SCALE, which a
    // `Value` carries and a lane does not. Pushing `Decimal(1000, 1)` into a
    // `Decimal64(2)` column reinterprets the units and renders 100.0 as 10.00.
    // Found by the sqlite oracle: `lag(2.25, 2, 100.0)` answered 10.00 against
    // sqlite's 100.0, on 6 of 40000 generated cases.
    //
    // Hoisted out of the row loop and skipped when the types already agree,
    // which is every non-decimal query and most decimal ones: one comparison
    // per block, nothing per row.
    let need_cast = |c: &Column| c.ty != f.ty;
    let cast_v = if need_cast(val) { Some(cast_column(val, &f.ty)?) } else { None };
    let val = cast_v.as_ref().unwrap_or(val);
    let cast_d = match dflt {
        Some(d) if need_cast(d) => Some(cast_column(d, &f.ty)?),
        _ => None,
    };
    let dflt = cast_d.as_ref().or(dflt);

    let mut b = ColumnBuilder::with_capacity(f.ty.clone(), hi - lo);
    let mut ps = lo;
    while ps < hi {
        let pe = p.part_end(ps, hi);
        for i in ps..pe {
            let src = if back {
                i.checked_sub(off).filter(|t| *t >= ps)
            } else {
                i.checked_add(off).filter(|t| *t < pe)
            };
            match src {
                Some(t) => b.push_value(&val.value(t))?,
                None => match dflt {
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
fn eval_pick(
    f: &BoundWindow,
    val: &Column,
    p: &Pass<'_>,
    lo: usize,
    hi: usize,
) -> Result<Column> {
    let flags = p.flags;
    let mut b = ColumnBuilder::with_capacity(f.ty.clone(), hi - lo);
    let mut ps = lo;
    while ps < hi {
        let pe = p.part_end(ps, hi);
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
#[allow(clippy::too_many_arguments)]
fn eval_agg(
    f: &BoundWindow,
    af: &'static AggFn,
    args: &[Column],
    tys: &[DataType],
    sweep: Sweep,
    p: &Pass<'_>,
    lo: usize,
    hi: usize,
    scratch: &mut Vec<Value>,
) -> Result<Column> {
    let (flags, ids) = (p.flags, p.ids);
    let mut b = ColumnBuilder::with_capacity(f.ty.clone(), hi - lo);
    let mut ps = lo;
    while ps < hi {
        // Once per *partition*, not once per row: the three linear sweeps are
        // fast enough that a per-row check would be the dominant cost, but a
        // `Refold` over a single wide partition is the one thing in this
        // operator that can run long enough for a user to want it stopped.
        p.ctx.check()?;
        let pe = p.part_end(ps, hi);
        match sweep {
            Sweep::Whole => {
                let mut acc = (af.new)(tys, &f.params)?;
                acc.update(args, &ids[ps..pe])?;
                let v = acc.finish()?;
                for _ in ps..pe {
                    b.push_value(&v)?;
                }
            }
            Sweep::Forward => {
                // The accumulator's contents are exactly `[ps, fed)`, and the
                // frame is exactly `[ps, e)` -- so `finish()` is the answer
                // with no bookkeeping, including for an empty frame, where a
                // fresh accumulator already reports the empty-input value.
                let mut acc = (af.new)(tys, &f.params)?;
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
                        acc.update(args, &ids[fed..e])?;
                        fed = e;
                    }
                    b.push_value(&acc.finish()?)?;
                }
            }
            Sweep::Backward => {
                let mut acc = (af.new)(tys, &f.params)?;
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
                        acc.update(args, &ids[s..fed])?;
                        fed = s;
                    }
                    scratch.push(acc.finish()?);
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
                    let mut acc = (af.new)(tys, &f.params)?;
                    if s < e {
                        acc.update(args, &ids[s..e])?;
                    }
                    b.push_value(&acc.finish()?)?;
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

    // ------------------------------------------- streaming and the fan-out

    /// Run one query under a budget, straight through the engine, and return
    /// its rows. Bound by hand because the budget is a property of the context
    /// and `Session::query` owns its own.
    fn under(s: &mut crate::Session, sql: &str, budget: i64) -> Result<Vec<Vec<Value>>> {
        use crate::planner::{binder::Binder, optimizer};
        s.execute("SYSTEM FLUSH").unwrap();
        let stmt = crate::sql::parse(sql).unwrap();
        let q = match &stmt[0] {
            crate::sql::ast::Statement::Query(q) => q.clone(),
            other => panic!("{other:?}"),
        };
        let cat = &s.catalog;
        let plan = optimizer::optimize(Binder::new(cat).bind_query(&q).unwrap()).unwrap();
        let ctx = QueryContext::with_budget(budget);
        let (blocks, _) = super::super::execute_ctx(&plan, cat, &ctx)?;
        let out = blocks
            .iter()
            .flat_map(|b| {
                (0..b.rows()).map(move |r| {
                    (0..b.width()).map(|c| b.column(c).value(r)).collect::<Vec<_>>()
                })
            })
            .collect();
        drop(plan);
        assert_eq!(ctx.mem.used(), 0, "the query kept its reservation");
        Ok(out)
    }

    #[test]
    fn a_window_cut_partition_by_partition_answers_like_a_buffered_one() {
        // The claim: the relation no longer has to fit, only a partition. The
        // budget here is far below the 20 000 rows and comfortably above one
        // 7000-row partition, so the operator has to cut several times -- and
        // the answer has to be identical *including order*, because a window's
        // output order is its input order however the operator chops it up.
        let mut s = session(20_000);
        for q in [
            "SELECT row_number() OVER (PARTITION BY g ORDER BY id) FROM t",
            "SELECT sum(v) OVER (PARTITION BY g ORDER BY id) FROM t",
            "SELECT lag(v, 3, -1) OVER (PARTITION BY g ORDER BY id) FROM t",
            "SELECT last_value(v) OVER (PARTITION BY g ORDER BY id \
             ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM t",
            "SELECT count(*) OVER (PARTITION BY g) FROM t",
            "SELECT sum(v) OVER (PARTITION BY g ORDER BY id ROWS BETWEEN 5 PRECEDING \
             AND 2 FOLLOWING) FROM t",
        ] {
            let want = under(&mut s, q, 512 << 20).unwrap();
            let got = under(&mut s, q, 4 << 20).unwrap();
            assert_eq!(got.len(), 20_000, "{q}: rows went missing");
            assert_eq!(got, want, "{q}: a cut window answered differently");
        }
    }

    #[test]
    fn one_partition_wider_than_the_budget_is_refused_and_says_so() {
        // The shape with no smaller unit. It has to be an error the caller can
        // read rather than a swap storm, and the error has to name the window
        // buffer -- otherwise it looks like the sort underneath ran out.
        let mut s = session(20_000);
        let e = under(&mut s, "SELECT sum(v) OVER (ORDER BY id) FROM t", 192 << 10).unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("memory budget"), "{msg}");

        // ... and the negative: the same query with room answers, so the
        // refusal is about the budget and not about the shape being broken.
        let got = under(&mut s, "SELECT sum(v) OVER (ORDER BY id) FROM t", 512 << 20).unwrap();
        assert_eq!(got.len(), 20_000);
    }

    #[test]
    fn the_split_lands_on_partition_starts_and_balances_rows_not_partitions() {
        // 100 rows: one partition of 70 and three of 10. An equal-partitions
        // split would give one worker 70% of the work.
        let mut flags = vec![0u8; 100];
        for i in [0usize, 70, 80, 90] {
            flags[i] = NEW_PART | NEW_PEER;
        }
        let r = split_partitions(&flags, 100, 4);
        assert!(r.iter().all(|&(s, _)| flags[s] & NEW_PART != 0), "a range cut a partition");
        assert_eq!(r.first().unwrap().0, 0);
        assert_eq!(r.last().unwrap().1, 100);
        for w in r.windows(2) {
            assert_eq!(w[0].1, w[1].0, "the ranges have to tile [0, n)");
        }
        // The first range must swallow the fat partition whole and stop, not
        // creep past 70 in search of a rounder number.
        assert_eq!(r[0], (0, 70));

        // A window with no PARTITION BY is one range whatever the width.
        let mut one = vec![0u8; 100];
        one[0] = NEW_PART | NEW_PEER;
        assert_eq!(split_partitions(&one, 100, 8), vec![(0, 100)]);

        // Every row its own partition: the split is even and uses every worker.
        let all = vec![NEW_PART | NEW_PEER; 100];
        let r = split_partitions(&all, 100, 4);
        assert_eq!(r, vec![(0, 25), (25, 50), (50, 75), (75, 100)]);
    }

    #[test]
    fn degree_leaves_small_windows_alone() {
        assert_eq!(degree(0), 1);
        assert_eq!(degree(MIN_PARALLEL_ROWS - 1), 1, "below the floor, stay serial");
        assert!(degree(MIN_PARALLEL_ROWS) >= 1);
        assert!(degree(1 << 20) <= pool::global().threads());
        // Enough rows to be worth it, but not enough for every thread.
        assert_eq!(degree(MIN_PARALLEL_ROWS).min(2), 2.min(pool::global().threads()));
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
