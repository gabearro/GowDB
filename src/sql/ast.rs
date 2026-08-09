//! Abstract syntax tree.
//!
//! This is the contract between `sql::parser` (which produces it) and
//! `planner::binder` (which consumes it). It is deliberately *untyped*: names
//! are unresolved, `Expr::Function` holds a bare string, and no column indices
//! appear anywhere. All of that is the binder's job.
//!
//! The shape follows ClickHouse's grammar where it diverges from ANSI SQL:
//! `PREWHERE`, `LIMIT n BY`, parametric aggregates (`quantile(0.9)(x)`),
//! `FINAL`, and `GROUP BY ... WITH TOTALS`.

use crate::types::{DataType, Engine, Value};
use std::fmt;
use std::mem;

// ------------------------------------------------------------------ names

/// A possibly-qualified name: `col`, `tbl.col`, `db.tbl.col`.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct ObjectName(pub Vec<String>);

impl ObjectName {
    pub fn bare(s: impl Into<String>) -> Self {
        ObjectName(vec![s.into()])
    }
    /// The last component -- the column or table name proper.
    pub fn last(&self) -> &str {
        self.0.last().map(|s| s.as_str()).unwrap_or("")
    }
    /// Everything before the last component.
    pub fn qualifier(&self) -> Option<&str> {
        if self.0.len() >= 2 {
            Some(&self.0[self.0.len() - 2])
        } else {
            None
        }
    }
}

impl fmt::Display for ObjectName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.join("."))
    }
}

// ------------------------------------------------------------- expressions

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinaryOp {
    Plus,
    Minus,
    Multiply,
    Divide,
    /// Integer division (`intDiv`), distinct from `/`.
    IntDiv,
    Modulo,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
    /// `||` -- string concatenation.
    Concat,
}

impl BinaryOp {
    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            BinaryOp::Eq
                | BinaryOp::NotEq
                | BinaryOp::Lt
                | BinaryOp::LtEq
                | BinaryOp::Gt
                | BinaryOp::GtEq
        )
    }
    pub fn is_logical(&self) -> bool {
        matches!(self, BinaryOp::And | BinaryOp::Or)
    }
    pub fn is_arithmetic(&self) -> bool {
        matches!(
            self,
            BinaryOp::Plus
                | BinaryOp::Minus
                | BinaryOp::Multiply
                | BinaryOp::Divide
                | BinaryOp::IntDiv
                | BinaryOp::Modulo
        )
    }
    /// Mirror a comparison so its operands can be swapped: `a < b` <-> `b > a`.
    pub fn flip(&self) -> BinaryOp {
        match self {
            BinaryOp::Lt => BinaryOp::Gt,
            BinaryOp::LtEq => BinaryOp::GtEq,
            BinaryOp::Gt => BinaryOp::Lt,
            BinaryOp::GtEq => BinaryOp::LtEq,
            other => *other,
        }
    }
    pub fn symbol(&self) -> &'static str {
        match self {
            BinaryOp::Plus => "+",
            BinaryOp::Minus => "-",
            BinaryOp::Multiply => "*",
            BinaryOp::Divide => "/",
            BinaryOp::IntDiv => "DIV",
            BinaryOp::Modulo => "%",
            BinaryOp::Eq => "=",
            BinaryOp::NotEq => "!=",
            BinaryOp::Lt => "<",
            BinaryOp::LtEq => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::GtEq => ">=",
            BinaryOp::And => "AND",
            BinaryOp::Or => "OR",
            BinaryOp::Concat => "||",
        }
    }
    /// [`symbol`](Self::symbol) with its surrounding spaces baked in, so
    /// rendering an infix node is one `write_str` and one pending stack entry
    /// instead of three of each.
    fn spaced(&self) -> &'static str {
        match self {
            BinaryOp::Plus => " + ",
            BinaryOp::Minus => " - ",
            BinaryOp::Multiply => " * ",
            BinaryOp::Divide => " / ",
            BinaryOp::IntDiv => " DIV ",
            BinaryOp::Modulo => " % ",
            BinaryOp::Eq => " = ",
            BinaryOp::NotEq => " != ",
            BinaryOp::Lt => " < ",
            BinaryOp::LtEq => " <= ",
            BinaryOp::Gt => " > ",
            BinaryOp::GtEq => " >= ",
            BinaryOp::And => " AND ",
            BinaryOp::Or => " OR ",
            BinaryOp::Concat => " || ",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntervalUnit {
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Quarter,
    Year,
}

impl IntervalUnit {
    pub fn parse(s: &str) -> Option<IntervalUnit> {
        Some(match s.to_ascii_lowercase().trim_end_matches('s') {
            "second" => IntervalUnit::Second,
            "minute" => IntervalUnit::Minute,
            "hour" => IntervalUnit::Hour,
            "day" => IntervalUnit::Day,
            "week" => IntervalUnit::Week,
            "month" => IntervalUnit::Month,
            "quarter" => IntervalUnit::Quarter,
            "year" => IntervalUnit::Year,
            _ => return None,
        })
    }
    pub fn name(&self) -> &'static str {
        match self {
            IntervalUnit::Second => "SECOND",
            IntervalUnit::Minute => "MINUTE",
            IntervalUnit::Hour => "HOUR",
            IntervalUnit::Day => "DAY",
            IntervalUnit::Week => "WEEK",
            IntervalUnit::Month => "MONTH",
            IntervalUnit::Quarter => "QUARTER",
            IntervalUnit::Year => "YEAR",
        }
    }
}

// ------------------------------------------------------------ window frames

/// What a frame bound counts.
///
/// `ROWS` counts physical rows; `RANGE` counts *peer groups* -- maximal runs of
/// rows that tie on every ORDER BY key. The distinction is not cosmetic, and it
/// is why `GROUPS` and offset `RANGE` bounds are refused by the parser rather
/// than quietly read as `ROWS`: under `sum(x) OVER (ORDER BY k)` every row tied
/// on `k` must see the same running total, and the `ROWS` reading gives each of
/// them a different one. A silently wrong frame is a wrong answer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FrameUnits {
    Rows,
    Range,
}

impl FrameUnits {
    pub fn name(&self) -> &'static str {
        match self {
            FrameUnits::Rows => "ROWS",
            FrameUnits::Range => "RANGE",
        }
    }
}

/// One end of a frame.
///
/// Offsets are already constant-folded to `u64`. SQL requires a non-negative
/// integer literal there, so carrying an `Expr` would only buy the binder a
/// second place to reject one -- and the parser is the layer that still has a
/// byte offset to point at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FrameBound {
    UnboundedPreceding,
    Preceding(u64),
    CurrentRow,
    Following(u64),
    UnboundedFollowing,
}

impl FrameBound {
    /// Position in the `UNBOUNDED PRECEDING < n PRECEDING < CURRENT ROW <
    /// n FOLLOWING < UNBOUNDED FOLLOWING` order, used to reject an inverted
    /// `BETWEEN`. Offsets inside a rank compare numerically, in opposite
    /// directions: `2 PRECEDING` is *earlier* than `1 PRECEDING`.
    pub fn rank(&self) -> (i8, i64) {
        match self {
            FrameBound::UnboundedPreceding => (0, 0),
            FrameBound::Preceding(n) => (1, -(*n as i64)),
            FrameBound::CurrentRow => (2, 0),
            FrameBound::Following(n) => (3, *n as i64),
            FrameBound::UnboundedFollowing => (4, 0),
        }
    }
}

impl fmt::Display for FrameBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrameBound::UnboundedPreceding => f.write_str("UNBOUNDED PRECEDING"),
            FrameBound::Preceding(n) => write!(f, "{n} PRECEDING"),
            FrameBound::CurrentRow => f.write_str("CURRENT ROW"),
            FrameBound::Following(n) => write!(f, "{n} FOLLOWING"),
            FrameBound::UnboundedFollowing => f.write_str("UNBOUNDED FOLLOWING"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WindowFrame {
    pub units: FrameUnits,
    pub start: FrameBound,
    pub end: FrameBound,
}

impl fmt::Display for WindowFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} BETWEEN {} AND {}", self.units.name(), self.start, self.end)
    }
}

/// The contents of an `OVER (...)`.
///
/// A named window (`WINDOW w AS (...)`) never survives parsing: the parser
/// substitutes the definition at every `OVER w`, so nothing downstream has to
/// carry a name table or decide what an unresolved one means.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct WindowSpec {
    pub partition_by: Vec<Expr>,
    pub order_by: Vec<OrderByExpr>,
    /// `None` means "the standard default", which is not one frame but two --
    /// see [`WindowSpec::effective_frame`].
    pub frame: Option<WindowFrame>,
}

impl WindowSpec {
    /// The frame this spec really means.
    ///
    /// With no explicit frame SQL says `RANGE BETWEEN UNBOUNDED PRECEDING AND
    /// CURRENT ROW`, and with no ORDER BY that degenerates to the whole
    /// partition because every row is then its own peer group's neighbour. Both
    /// are spelled out rather than left to the operator, because "the default
    /// frame" is exactly the thing engines get wrong.
    pub fn effective_frame(&self) -> WindowFrame {
        match self.frame {
            Some(f) => f,
            None if self.order_by.is_empty() => WindowFrame {
                units: FrameUnits::Range,
                start: FrameBound::UnboundedPreceding,
                end: FrameBound::UnboundedFollowing,
            },
            None => WindowFrame {
                units: FrameUnits::Range,
                start: FrameBound::UnboundedPreceding,
                end: FrameBound::CurrentRow,
            },
        }
    }
}

impl fmt::Display for WindowSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut sep = "";
        f.write_str("(")?;
        if !self.partition_by.is_empty() {
            f.write_str("PARTITION BY ")?;
            for (i, e) in self.partition_by.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{e}")?;
            }
            sep = " ";
        }
        if !self.order_by.is_empty() {
            write!(f, "{sep}ORDER BY ")?;
            for (i, o) in self.order_by.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{o}")?;
            }
            sep = " ";
        }
        if let Some(fr) = &self.frame {
            write!(f, "{sep}{fr}")?;
        }
        f.write_str(")")
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum Expr {
    Literal(Value),
    Column(ObjectName),
    /// `*` in `count(*)`.
    Wildcard,
    UnaryOp {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    /// A call to a scalar or aggregate function. The binder decides which.
    ///
    /// `params` carries ClickHouse's parametric-aggregate syntax:
    /// `quantile(0.9)(latency)` parses as `name="quantile"`,
    /// `params=[0.9]`, `args=[latency]`.
    Function {
        name: String,
        args: Vec<Expr>,
        params: Vec<Expr>,
        distinct: bool,
    },
    /// `f(args) OVER (spec)`.
    ///
    /// A separate variant rather than an `Option<WindowSpec>` field on
    /// [`Expr::Function`]: the two bind in different places (a window call is
    /// hoisted into an operator that runs *after* GROUP BY and HAVING, an
    /// aggregate into one that runs before), and every `Expr::Function` match
    /// in the engine would otherwise have to remember to test the field. A
    /// missed test would be a silent wrong answer; a missed match arm is a
    /// compile error.
    Window {
        name: String,
        args: Vec<Expr>,
        params: Vec<Expr>,
        distinct: bool,
        spec: Box<WindowSpec>,
    },
    Cast {
        expr: Box<Expr>,
        ty: DataType,
    },
    Case {
        operand: Option<Box<Expr>>,
        when_then: Vec<(Expr, Expr)>,
        else_result: Option<Box<Expr>>,
    },
    InList {
        expr: Box<Expr>,
        list: Vec<Expr>,
        negated: bool,
    },
    InSubquery {
        expr: Box<Expr>,
        subquery: Box<Query>,
        negated: bool,
    },
    Between {
        expr: Box<Expr>,
        low: Box<Expr>,
        high: Box<Expr>,
        negated: bool,
    },
    Like {
        expr: Box<Expr>,
        pattern: Box<Expr>,
        negated: bool,
        case_insensitive: bool,
    },
    IsNull {
        expr: Box<Expr>,
        negated: bool,
    },
    Tuple(Vec<Expr>),
    /// Scalar subquery: `(SELECT max(x) FROM t)`.
    Subquery(Box<Query>),
    Exists {
        subquery: Box<Query>,
        negated: bool,
    },
    Interval {
        value: Box<Expr>,
        unit: IntervalUnit,
    },
}

impl Expr {
    pub fn func(name: impl Into<String>, args: Vec<Expr>) -> Expr {
        Expr::Function { name: name.into(), args, params: Vec::new(), distinct: false }
    }
    pub fn binary(left: Expr, op: BinaryOp, right: Expr) -> Expr {
        Expr::BinaryOp { left: Box::new(left), op, right: Box::new(right) }
    }
    pub fn col(name: &str) -> Expr {
        Expr::Column(ObjectName::bare(name))
    }
    pub fn lit(v: impl Into<Value>) -> Expr {
        Expr::Literal(v.into())
    }

    /// The recursive walk, used while `budget` lasts. `visit` hands it a
    /// budget rather than calling it directly so that a loop-grown chain
    /// falls off into [`Expr::visit_iterative`] instead of off the stack.
    fn visit_within<F: FnMut(&Expr)>(&self, f: &mut F, budget: u32) {
        if budget == 0 {
            return self.visit_iterative(f);
        }
        let budget = budget - 1;
        f(self);
        match self {
            Expr::UnaryOp { expr, .. }
            | Expr::Cast { expr, .. }
            | Expr::IsNull { expr, .. } => expr.visit_within(f, budget),
            Expr::BinaryOp { left, right, .. } => {
                left.visit_within(f, budget);
                right.visit_within(f, budget);
            }
            Expr::Function { args, params, .. } => {
                params.iter().for_each(|e| e.visit_within(f, budget));
                args.iter().for_each(|e| e.visit_within(f, budget));
            }
            // The spec's expressions are real column references and the walk
            // has to reach them: `binder::Demand` is what decides which columns
            // the scan projects, and a `PARTITION BY` key it never saw is a
            // column the operator cannot read.
            Expr::Window { args, params, spec, .. } => {
                params.iter().for_each(|e| e.visit_within(f, budget));
                args.iter().for_each(|e| e.visit_within(f, budget));
                spec.partition_by.iter().for_each(|e| e.visit_within(f, budget));
                spec.order_by.iter().for_each(|o| o.expr.visit_within(f, budget));
            }
            Expr::Case { operand, when_then, else_result } => {
                if let Some(o) = operand {
                    o.visit_within(f, budget);
                }
                for (w, t) in when_then {
                    w.visit_within(f, budget);
                    t.visit_within(f, budget);
                }
                if let Some(e) = else_result {
                    e.visit_within(f, budget);
                }
            }
            Expr::InList { expr, list, .. } => {
                expr.visit_within(f, budget);
                list.iter().for_each(|e| e.visit_within(f, budget));
            }
            Expr::InSubquery { expr, .. } => expr.visit_within(f, budget),
            Expr::Between { expr, low, high, .. } => {
                expr.visit_within(f, budget);
                low.visit_within(f, budget);
                high.visit_within(f, budget);
            }
            Expr::Like { expr, pattern, .. } => {
                expr.visit_within(f, budget);
                pattern.visit_within(f, budget);
            }
            Expr::Tuple(items) => items.iter().for_each(|e| e.visit_within(f, budget)),
            Expr::Interval { value, .. } => value.visit_within(f, budget),
            Expr::Literal(_) | Expr::Column(_) | Expr::Wildcard | Expr::Subquery(_)
            | Expr::Exists { .. } => {}
        }
    }

    /// Every direct `Expr` child, pushed in *reverse* source order so popping
    /// the stack yields them forwards. `Subquery`/`Exists`/`InSubquery`'s
    /// query stay opaque -- the walk has never descended into a nested query.
    fn push_children<'a>(&'a self, out: &mut Vec<&'a Expr>) {
        match self {
            Expr::UnaryOp { expr, .. }
            | Expr::Cast { expr, .. }
            | Expr::IsNull { expr, .. }
            | Expr::InSubquery { expr, .. }
            | Expr::Interval { value: expr, .. } => out.push(expr),
            Expr::BinaryOp { left, right, .. } => {
                out.push(right);
                out.push(left);
            }
            Expr::Function { args, params, .. } => {
                out.extend(args.iter().rev());
                out.extend(params.iter().rev());
            }
            Expr::Window { args, params, spec, .. } => {
                out.extend(spec.order_by.iter().rev().map(|o| &o.expr));
                out.extend(spec.partition_by.iter().rev());
                out.extend(args.iter().rev());
                out.extend(params.iter().rev());
            }
            Expr::Case { operand, when_then, else_result } => {
                out.extend(else_result.as_deref());
                for (w, t) in when_then.iter().rev() {
                    out.push(t);
                    out.push(w);
                }
                out.extend(operand.as_deref());
            }
            Expr::InList { expr, list, .. } => {
                out.extend(list.iter().rev());
                out.push(expr);
            }
            Expr::Between { expr, low, high, .. } => {
                out.push(high);
                out.push(low);
                out.push(expr);
            }
            Expr::Like { expr, pattern, .. } => {
                out.push(pattern);
                out.push(expr);
            }
            Expr::Tuple(items) => out.extend(items.iter().rev()),
            Expr::Literal(_) | Expr::Column(_) | Expr::Wildcard | Expr::Subquery(_)
            | Expr::Exists { .. } => {}
        }
    }

    /// How deep the walk recurses before switching to the worklist. Real
    /// expressions are nowhere near this; a loop-grown chain blows through it
    /// on the first node.
    const VISIT_BUDGET: u32 = 64;

    /// Walk every subexpression, depth-first, parents before children.
    ///
    /// Recursive while the tree is shallow -- which is every hand-written
    /// query -- because the explicit stack costs a `malloc` the recursion does
    /// not: measured 29ns recursive vs 75ns iterative for `contains_aggregate`
    /// on a six-node expression, and the binder runs that per projection.
    /// Past [`Expr::VISIT_BUDGET`] it falls into the worklist, because a
    /// loop-grown `OR` chain is as deep as the query is long and the purely
    /// recursive walk died at 100k terms on a 2 MiB stack.
    pub fn visit<F: FnMut(&Expr)>(&self, f: &mut F) {
        self.visit_within(f, Expr::VISIT_BUDGET)
    }

    /// The worklist walk [`Expr::visit_within`] falls into. A leaf never
    /// pushes, so the `Vec` stays unallocated unless the subtree is deep.
    fn visit_iterative<F: FnMut(&Expr)>(&self, f: &mut F) {
        let mut todo: Vec<&Expr> = Vec::new();
        let mut cur = self;
        loop {
            f(cur);
            cur.push_children(&mut todo);
            match todo.pop() {
                Some(next) => cur = next,
                None => return,
            }
        }
    }

    /// True if this expression tree contains an aggregate call. The binder
    /// uses this to split projections into pre- and post-aggregation halves.
    pub fn contains_aggregate(&self, is_agg: &dyn Fn(&str) -> bool) -> bool {
        let mut found = false;
        self.visit(&mut |e| {
            if let Expr::Function { name, .. } = e {
                if is_agg(name) {
                    found = true;
                }
            }
        });
        found
    }

    /// A stable display name for an unaliased projection, matching
    /// ClickHouse's habit of naming the column after its own source text.
    pub fn display_name(&self) -> String {
        match self {
            Expr::Column(n) => n.last().to_string(),
            other => other.to_string(),
        }
    }
}

// ------------------------------------------------------------------ queries

#[derive(Clone, PartialEq, Debug)]
pub struct OrderByExpr {
    pub expr: Expr,
    pub asc: bool,
    /// `None` means the default: nulls first when ascending, last when
    /// descending (ClickHouse's `NULLS LAST` default is the opposite of
    /// Postgres, so we make it explicit).
    pub nulls_first: Option<bool>,
}

impl OrderByExpr {
    pub fn nulls_first_effective(&self) -> bool {
        self.nulls_first.unwrap_or(self.asc)
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum SelectItem {
    /// `*`
    Wildcard,
    /// `t.*`
    QualifiedWildcard(String),
    Expr { expr: Expr, alias: Option<String> },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum JoinOp {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

#[derive(Clone, PartialEq, Debug)]
pub enum JoinConstraint {
    On(Expr),
    Using(Vec<String>),
    /// `CROSS JOIN` / comma join.
    None,
}

#[derive(Clone, PartialEq, Debug)]
pub enum TableRef {
    Table {
        name: ObjectName,
        alias: Option<String>,
        /// ClickHouse `FINAL`: force a merge-on-read so ReplacingMergeTree
        /// and SummingMergeTree return collapsed rows.
        final_: bool,
    },
    Subquery {
        query: Box<Query>,
        alias: Option<String>,
    },
    Join {
        left: Box<TableRef>,
        right: Box<TableRef>,
        op: JoinOp,
        constraint: JoinConstraint,
    },
}

#[derive(Clone, PartialEq, Debug)]
pub struct Select {
    pub distinct: bool,
    pub projection: Vec<SelectItem>,
    pub from: Option<TableRef>,
    /// ClickHouse `PREWHERE`: a predicate the planner is *required* to push
    /// into the scan ahead of reading other columns.
    pub prewhere: Option<Expr>,
    pub selection: Option<Expr>,
    /// One entry per `GROUP BY` item. A multi-grouping item -- `ROLLUP(a, b)`,
    /// `CUBE(a, b)`, `GROUPING SETS ((a, b), (a), ())` -- rides here as an
    /// ordinary [`Expr::Function`] and is recognized by [`GroupSpec::of`].
    pub group_by: Vec<Expr>,
    pub with_totals: bool,
    pub having: Option<Expr>,
}

/// The spelling of a `GROUP BY` item that asks for more than one grouping.
///
/// These live in [`Select::group_by`] as `Expr::Function` calls rather than in
/// a variant of their own, and that is a deliberate cheat with two payoffs.
/// `ROLLUP(a, b)` and `CUBE(a, b)` *are* function-call syntax, so the
/// expression parser reads them with no grammar at all; and every generic walk
/// over a group key -- view qualification, subquery detection, the depth guard
/// -- keeps working with no arm added, where a new `Expr` variant or a change
/// to `group_by`'s type would have made each of them a place to forget.
///
/// The names are reserved only *here*: nothing else in the engine defines a
/// function called `rollup` or `cube`, so `SELECT cube(x)` is the same unknown
/// function it always was.
pub enum GroupSpec<'a> {
    /// `ROLLUP(a, b, c)`: the four prefixes, longest first.
    Rollup(&'a [Expr]),
    /// `CUBE(a, b, c)`: all eight subsets.
    Cube(&'a [Expr]),
    /// `GROUPING SETS (...)`: each element is one set, always an
    /// [`Expr::Tuple`] -- including the empty one, which is the grand total.
    Sets(&'a [Expr]),
}

/// The name the parser gives a `GROUPING SETS` item. Not writable as a call:
/// the space is what keeps a user's `SELECT "GROUPING SETS"(x)` out.
pub const GROUPING_SETS: &str = "GROUPING SETS";

impl GroupSpec<'_> {
    pub fn of(e: &Expr) -> Option<GroupSpec<'_>> {
        let Expr::Function { name, args, params, distinct } = e else { return None };
        if !params.is_empty() || *distinct {
            return None;
        }
        if name == GROUPING_SETS {
            return Some(GroupSpec::Sets(args));
        }
        if name.eq_ignore_ascii_case("rollup") {
            return Some(GroupSpec::Rollup(args));
        }
        if name.eq_ignore_ascii_case("cube") {
            return Some(GroupSpec::Cube(args));
        }
        None
    }

    pub fn keyword(&self) -> &'static str {
        match self {
            GroupSpec::Rollup(_) => "ROLLUP",
            GroupSpec::Cube(_) => "CUBE",
            GroupSpec::Sets(_) => GROUPING_SETS,
        }
    }
}

/// The three ANSI set operations. `INTERSECT` binds tighter than the other
/// two, which is a parser rule (see `set_term`) rather than anything this enum
/// records.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SetOp {
    Union,
    Except,
    Intersect,
}

impl SetOp {
    /// The keyword, for diagnostics and `EXPLAIN`. Uppercase because every
    /// message that names an operation in this engine spells it as SQL does.
    pub fn keyword(self) -> &'static str {
        match self {
            SetOp::Union => "UNION",
            SetOp::Except => "EXCEPT",
            SetOp::Intersect => "INTERSECT",
        }
    }

    /// Title case, for plan labels: `Union All`, `Except Distinct`.
    pub fn label(self) -> &'static str {
        match self {
            SetOp::Union => "Union",
            SetOp::Except => "Except",
            SetOp::Intersect => "Intersect",
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum SetExpr {
    Select(Box<Select>),
    Query(Box<Query>),
    SetOperation {
        op: SetOp,
        all: bool,
        left: Box<SetExpr>,
        right: Box<SetExpr>,
    },
    /// A bare `VALUES (...), (...)` row set.
    Values(Vec<Vec<Expr>>),
}

#[derive(Clone, PartialEq, Debug)]
pub struct Cte {
    pub name: String,
    pub query: Box<Query>,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Query {
    pub with: Vec<Cte>,
    pub body: SetExpr,
    pub order_by: Vec<OrderByExpr>,
    pub limit: Option<Expr>,
    pub offset: Option<Expr>,
    /// ClickHouse `LIMIT n BY (exprs)`: keep the first `n` rows per distinct
    /// value of the key expressions.
    pub limit_by: Option<(Expr, Vec<Expr>)>,
}

impl Query {
    pub fn simple(select: Select) -> Query {
        Query {
            with: Vec::new(),
            body: SetExpr::Select(Box::new(select)),
            order_by: Vec::new(),
            limit: None,
            offset: None,
            limit_by: None,
        }
    }

    /// Rewrite every *unqualified* table reference to `db.<name>`.
    ///
    /// This is what makes a stored view mean the same thing from any session.
    /// A view body is written against whatever database was current when it
    /// was created, and it is expanded, later, into somebody else's query --
    /// so `CREATE VIEW reports.recent AS SELECT * FROM events` has to keep
    /// meaning `sales.events` after a `USE analytics`, or the view silently
    /// answers from a different table with the same name. Binding it once at
    /// creation is not enough: the binder resolves names against the *current*
    /// database at each use, and the expansion is an AST splice.
    ///
    /// Two things are deliberately left alone. A name that already carries a
    /// qualifier is not touched, and a bare name that matches a CTE in scope
    /// is not either -- `WITH e AS (...) SELECT * FROM e` names the CTE, and
    /// `Binder::table_ref` only consults the CTE stack for a *single*-component
    /// name, so qualifying one would silently redirect it to a base table.
    pub fn qualify_tables(&mut self, db: &str) {
        let mut ctes: Vec<String> = Vec::new();
        qual_query(self, db, &mut ctes);
    }
}

/// `scope` is the CTE names visible here, innermost last. It is grown and
/// truncated in place rather than copied per level: a query nests a handful of
/// levels and this runs once per view, at creation and at load.
fn qual_query(q: &mut Query, db: &str, scope: &mut Vec<String>) {
    let mark = scope.len();
    for cte in q.with.iter_mut() {
        // A CTE may reference the ones declared before it, and not itself:
        // exactly the rule `Binder::table_ref` applies with `&ctes[..pos]`.
        qual_query(&mut cte.query, db, scope);
        scope.push(cte.name.clone());
    }
    qual_setexpr(&mut q.body, db, scope);
    for o in q.order_by.iter_mut() {
        qual_expr(&mut o.expr, db, scope);
    }
    for e in q.limit.iter_mut().chain(q.offset.iter_mut()) {
        qual_expr(e, db, scope);
    }
    if let Some((n, keys)) = q.limit_by.as_mut() {
        qual_expr(n, db, scope);
        for k in keys.iter_mut() {
            qual_expr(k, db, scope);
        }
    }
    scope.truncate(mark);
}

fn qual_setexpr(s: &mut SetExpr, db: &str, scope: &mut Vec<String>) {
    match s {
        SetExpr::Select(sel) => {
            for item in sel.projection.iter_mut() {
                if let SelectItem::Expr { expr, .. } = item {
                    qual_expr(expr, db, scope);
                }
            }
            if let Some(f) = sel.from.as_mut() {
                qual_tableref(f, db, scope);
            }
            for e in sel
                .prewhere
                .iter_mut()
                .chain(sel.selection.iter_mut())
                .chain(sel.having.iter_mut())
            {
                qual_expr(e, db, scope);
            }
            for g in sel.group_by.iter_mut() {
                qual_expr(g, db, scope);
            }
        }
        SetExpr::Query(q) => qual_query(q, db, scope),
        SetExpr::SetOperation { left, right, .. } => {
            qual_setexpr(left, db, scope);
            qual_setexpr(right, db, scope);
        }
        SetExpr::Values(rows) => {
            for e in rows.iter_mut().flatten() {
                qual_expr(e, db, scope);
            }
        }
    }
}

fn qual_tableref(t: &mut TableRef, db: &str, scope: &mut Vec<String>) {
    match t {
        TableRef::Table { name, .. } => {
            if name.0.len() == 1
                && !scope.iter().any(|c| c.eq_ignore_ascii_case(name.last()))
            {
                name.0.insert(0, db.to_string());
            }
        }
        TableRef::Subquery { query, .. } => qual_query(query, db, scope),
        TableRef::Join { left, right, constraint, .. } => {
            qual_tableref(left, db, scope);
            qual_tableref(right, db, scope);
            if let JoinConstraint::On(e) = constraint {
                qual_expr(e, db, scope);
            }
        }
    }
}

/// Exhaustive on purpose -- no wildcard arm. A subquery hiding in an `Expr`
/// variant nobody remembered to walk is a table reference resolved against the
/// reader's database instead of the view's, which is a wrong answer rather than
/// a missed optimization, so a new variant must fail to compile here.
fn qual_expr(e: &mut Expr, db: &str, scope: &mut Vec<String>) {
    match e {
        Expr::Literal(_) | Expr::Column(_) | Expr::Wildcard => {}
        Expr::UnaryOp { expr, .. }
        | Expr::Cast { expr, .. }
        | Expr::IsNull { expr, .. }
        | Expr::Interval { value: expr, .. } => qual_expr(expr, db, scope),
        Expr::BinaryOp { left, right, .. } => {
            qual_expr(left, db, scope);
            qual_expr(right, db, scope);
        }
        Expr::Function { args, params, .. } => {
            for a in args.iter_mut().chain(params.iter_mut()) {
                qual_expr(a, db, scope);
            }
        }
        Expr::Window { args, params, spec, .. } => {
            for a in args.iter_mut().chain(params.iter_mut()) {
                qual_expr(a, db, scope);
            }
            for p in spec.partition_by.iter_mut() {
                qual_expr(p, db, scope);
            }
            for o in spec.order_by.iter_mut() {
                qual_expr(&mut o.expr, db, scope);
            }
        }
        Expr::Case { operand, when_then, else_result } => {
            for o in operand.iter_mut() {
                qual_expr(o, db, scope);
            }
            for (w, t) in when_then.iter_mut() {
                qual_expr(w, db, scope);
                qual_expr(t, db, scope);
            }
            for x in else_result.iter_mut() {
                qual_expr(x, db, scope);
            }
        }
        Expr::InList { expr, list, .. } => {
            qual_expr(expr, db, scope);
            for i in list.iter_mut() {
                qual_expr(i, db, scope);
            }
        }
        Expr::Between { expr, low, high, .. } => {
            qual_expr(expr, db, scope);
            qual_expr(low, db, scope);
            qual_expr(high, db, scope);
        }
        Expr::Like { expr, pattern, .. } => {
            qual_expr(expr, db, scope);
            qual_expr(pattern, db, scope);
        }
        Expr::Tuple(items) => {
            for i in items.iter_mut() {
                qual_expr(i, db, scope);
            }
        }
        Expr::Subquery(q) => qual_query(q, db, scope),
        Expr::InSubquery { expr, subquery, .. } => {
            qual_expr(expr, db, scope);
            qual_query(subquery, db, scope);
        }
        Expr::Exists { subquery, .. } => qual_query(subquery, db, scope),
    }
}

// --------------------------------------------------------------- statements

#[derive(Clone, PartialEq, Debug)]
pub struct ColumnDef {
    pub name: String,
    pub ty: DataType,
    pub default: Option<Expr>,
    pub codec: Option<String>,
    /// `UNIQUE` on the column. Kept per column rather than folded into
    /// `CreateTable::checks` because it is not an expression: the session
    /// enforces it against the unique-key machinery, or refuses the
    /// declaration outright when there is no key to enforce it with.
    pub unique: bool,
}

/// One `CHECK` constraint, however it was spelled: `CHECK (p)` after a column,
/// `CHECK (p)` as its own element of the column list, or
/// `CONSTRAINT c CHECK (p)`. All three end up here, because all three mean the
/// same thing to a write -- a table-wide predicate every row must satisfy.
#[derive(Clone, PartialEq, Debug)]
pub struct CheckDef {
    /// `CONSTRAINT <name>`, or `None` for an unnamed one. Only ever used to
    /// name the constraint in the rejection message and in the catalog.
    pub name: Option<String>,
    pub expr: Expr,
}

#[derive(Clone, PartialEq, Debug)]
pub struct CreateTable {
    pub name: ObjectName,
    pub if_not_exists: bool,
    pub columns: Vec<ColumnDef>,
    pub engine: Engine,
    pub order_by: Vec<Expr>,
    pub primary_key: Vec<Expr>,
    pub partition_by: Option<Expr>,
    /// Every `CHECK` the statement declared, in source order.
    pub checks: Vec<CheckDef>,
    /// `AS SELECT ...`
    pub as_query: Option<Box<Query>>,
}

#[derive(Clone, PartialEq, Debug)]
pub enum InsertSource {
    Values(Vec<Vec<Expr>>),
    Query(Box<Query>),
}

#[derive(Clone, PartialEq, Debug)]
pub struct Insert {
    pub table: ObjectName,
    /// Empty means "all columns, in declaration order".
    pub columns: Vec<String>,
    pub source: InsertSource,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExplainKind {
    /// Logical plan after optimization.
    Plan,
    /// Physical operator tree.
    Pipeline,
    /// Parsed AST, pre-binding.
    Ast,
    /// Physical operator tree, **executed**, annotated with the rows and time
    /// each operator actually cost. The only kind that runs the statement.
    Analyze,
}

#[derive(Clone, PartialEq, Debug)]
pub enum Statement {
    Query(Box<Query>),
    Insert(Insert),
    CreateTable(Box<CreateTable>),
    CreateDatabase {
        name: String,
        if_not_exists: bool,
    },
    DropTable {
        name: ObjectName,
        if_exists: bool,
    },
    DropDatabase {
        name: String,
        if_exists: bool,
    },
    /// A bulk delete: `DELETE FROM t WHERE p`, or its ClickHouse synonym
    /// `ALTER TABLE t DELETE WHERE p`.
    ///
    /// One node for both spellings on purpose -- a mutation is a mutation, and
    /// two nodes would mean two execution paths that have to be kept agreeing.
    /// The variant keeps its `Alter` name because [`crate::session`] dispatches
    /// on it and this crate does not rename across an ownership boundary; read
    /// it as "the delete mutation", not as "the ALTER spelling of it".
    ///
    /// `predicate` is never absent: a missing `WHERE` parses as literal `true`,
    /// so "delete everything" needs no `Option` anywhere downstream.
    AlterDelete {
        table: ObjectName,
        predicate: Expr,
    },
    /// A bulk update: `UPDATE t SET c = e, ... WHERE p`, or its ClickHouse
    /// synonym `ALTER TABLE t UPDATE c = e, ... WHERE p`.
    ///
    /// Every right-hand side reads the row's **pre-update** values, so
    /// `SET a = b, b = a` swaps rather than assigning `b` to both. See
    /// [`AlterDelete`](Statement::AlterDelete) on the naming and on `predicate`.
    AlterUpdate {
        table: ObjectName,
        assignments: Vec<(String, Expr)>,
        predicate: Expr,
    },
    AlterAddColumn {
        table: ObjectName,
        column: ColumnDef,
        if_not_exists: bool,
    },
    AlterDropColumn {
        table: ObjectName,
        column: String,
        if_exists: bool,
    },
    /// `ALTER TABLE t MODIFY COLUMN c <type>`: retype a column in place.
    ///
    /// Only the type changes. `MODIFY COLUMN c <type> DEFAULT x` and the other
    /// per-column clauses are refused by the parser rather than accepted and
    /// dropped -- see `Parser::alter`.
    AlterModifyColumn {
        table: ObjectName,
        column: String,
        ty: DataType,
    },
    /// `RENAME TABLE a TO b`, and its `ALTER TABLE a RENAME TO b` spelling.
    RenameTable {
        from: ObjectName,
        to: ObjectName,
    },
    /// `CREATE VIEW v AS SELECT ...`.
    ///
    /// `body_sql` is the *source text* of the query, sliced out by the parser,
    /// and it is what the catalog stores: this AST has no `Display`, so a view
    /// that round-trips through disk has to round-trip as text. Keeping both
    /// halves means the text is never the thing that gets bound -- the parsed
    /// `query` is -- so the two cannot disagree about what the view means.
    CreateView {
        name: ObjectName,
        query: Box<Query>,
        body_sql: String,
        or_replace: bool,
        if_not_exists: bool,
    },
    DropView {
        name: ObjectName,
        if_exists: bool,
    },
    /// Force a compaction. `final_` merges down to a single part.
    Optimize {
        table: ObjectName,
        final_: bool,
    },
    Truncate {
        table: ObjectName,
    },
    ShowTables {
        database: Option<String>,
    },
    ShowDatabases,
    ShowCreateTable(ObjectName),
    Describe(ObjectName),
    Explain {
        kind: ExplainKind,
        statement: Box<Statement>,
    },
    Use(String),
    /// `SYSTEM FLUSH` — force the delta memtable to a part.
    SystemFlush(Option<ObjectName>),
}

// ----------------------------------------------------------------- teardown
//
// `parser::MAX_DEPTH` bounds recursive *descent*, but the left-associative
// productions -- `+ - * /`, the comparisons, AND, OR, UNION, and the join
// list -- are parsed by a loop, so they are not descent and are not counted.
// A 60k-term `OR` chain (what an ORM emits expanding a large `IN` list) parses
// happily and then kills the process in the compiler-generated `Drop`, which
// recurses once per node. Worse than the nested-paren case fixed in the
// parser: it survives parsing, so the SIGABRT lands at an arbitrary later
// point, possibly after the query already returned rows.
//
// So the three loop-grown spines free themselves with an explicit worklist.
// Everything else -- Query, Select, Cte, Statement::Explain, the
// Expr -> Query -> SetExpr -> Select -> Expr hop -- can only be nested by a
// depth-counted production, so it is already bounded by MAX_DEPTH and is left
// to the derived drop glue.
//
// Two properties keep this off the critical path for ordinary queries:
// `unlink` only moves out children that are themselves interior nodes, so a
// leaf (and a 10k-element literal `IN` list) is freed in place by the normal
// glue and never touches the worklist; and `drop` refuses the worklist
// outright for a node with no interior child, which is nearly all of them.
//
// What it costs, measured interleaved (best of 25, release, the 45-node AST of
// a GROUP BY/HAVING/ORDER BY query): statement teardown 630ns -> 760ns, +20%,
// or +2.3% across parse+drop together. Two cheaper-looking arrangements were
// tried and are worse, so do not re-try them: a fresh `Vec` per expression is
// +45% (the `malloc`/`free` pair costs more than the whole walk), and taking
// the pooled buffer unconditionally is +40% (a TLS access on every node costs
// about what the allocation did). The residue is irreducible -- a manual
// `Drop` is called once per node no matter how little it then does.
//
// STILL RECURSIVE, and measured on a 2 MiB stack: the *derived* `Clone` and
// `PartialEq` abort at ~3k terms and `Debug` at ~5k, all shallower than the
// `Drop` bug this section fixes. `Clone` is reachable -- `binder.rs` clones
// every aliased projection -- so it is a live abort, not a theoretical one;
// undoing it means hand-writing a bottom-up iterative `Clone`, which is a
// bigger change than this one and was left out of it deliberately.

/// A fieldless variant to leave behind when a child is moved out. Chosen
/// because `mem::replace` with it compiles to a discriminant store -- no
/// allocation, and its own drop is a no-op.
const EXPR_HOLE: Expr = Expr::Wildcard;

thread_local! {
    /// The teardown worklist, kept across drops so that steady state allocates
    /// nothing: the buffer comes out of the `Cell` and goes back with its
    /// capacity intact, which costs a TLS access instead of a `malloc`/`free`
    /// pair. Only a node with an interior child gets this far.
    ///
    /// `Cell`, not `RefCell`: teardown reenters itself (an `Expr` owning a
    /// `Query` owning more `Expr`s), and taking the buffer out leaves the
    /// inner drop an empty one to use rather than a borrow panic.
    static TEARDOWN: std::cell::Cell<Vec<Expr>> = const { std::cell::Cell::new(Vec::new()) };
}

impl Expr {
    /// True if this node owns any `Expr` child. `Subquery`/`Exists` count as
    /// leaves here: their children are `Query`, which is depth-bounded.
    #[inline]
    fn has_children(&self) -> bool {
        match self {
            Expr::Literal(_)
            | Expr::Column(_)
            | Expr::Wildcard
            | Expr::Subquery(_)
            | Expr::Exists { .. } => false,
            Expr::Function { args, params, .. } => !args.is_empty() || !params.is_empty(),
            Expr::Window { args, params, spec, .. } => {
                !args.is_empty()
                    || !params.is_empty()
                    || !spec.partition_by.is_empty()
                    || !spec.order_by.is_empty()
            }
            Expr::Tuple(items) => !items.is_empty(),
            Expr::Case { operand, when_then, else_result } => {
                operand.is_some() || !when_then.is_empty() || else_result.is_some()
            }
            _ => true,
        }
    }

    /// True if any direct child is itself an interior node -- i.e. if letting
    /// the compiler's glue free this node could go more than one frame deep.
    #[inline]
    /// Checked before anything else, because it is false for the great
    /// majority of nodes and is the only thing standing between an ordinary
    /// query's teardown and a worklist it has no use for.
    fn has_interior_child(&self) -> bool {
        match self {
            Expr::UnaryOp { expr, .. }
            | Expr::Cast { expr, .. }
            | Expr::IsNull { expr, .. }
            | Expr::InSubquery { expr, .. }
            | Expr::Interval { value: expr, .. } => expr.has_children(),
            Expr::BinaryOp { left, right, .. } => left.has_children() || right.has_children(),
            Expr::Function { args, params, .. } => {
                args.iter().chain(params).any(Expr::has_children)
            }
            Expr::Window { args, params, spec, .. } => {
                args.iter()
                    .chain(params)
                    .chain(&spec.partition_by)
                    .chain(spec.order_by.iter().map(|o| &o.expr))
                    .any(Expr::has_children)
            }
            Expr::Case { operand, when_then, else_result } => {
                operand.as_deref().is_some_and(Expr::has_children)
                    || when_then.iter().any(|(w, t)| w.has_children() || t.has_children())
                    || else_result.as_deref().is_some_and(Expr::has_children)
            }
            Expr::InList { expr, list, .. } => {
                expr.has_children() || list.iter().any(Expr::has_children)
            }
            Expr::Between { expr, low, high, .. } => {
                expr.has_children() || low.has_children() || high.has_children()
            }
            Expr::Like { expr, pattern, .. } => expr.has_children() || pattern.has_children(),
            Expr::Tuple(items) => items.iter().any(Expr::has_children),
            Expr::Literal(_)
            | Expr::Column(_)
            | Expr::Wildcard
            | Expr::Subquery(_)
            | Expr::Exists { .. } => false,
        }
    }

    /// Move every *interior* child into `sink`, leaving [`EXPR_HOLE`] behind.
    fn unlink(&mut self, sink: &mut Vec<Expr>) {
        #[inline]
        fn one(e: &mut Expr, sink: &mut Vec<Expr>) {
            if e.has_children() {
                sink.push(mem::replace(e, EXPR_HOLE));
            }
        }
        #[inline]
        fn many(v: &mut [Expr], sink: &mut Vec<Expr>) {
            for e in v {
                one(e, sink);
            }
        }
        match self {
            Expr::UnaryOp { expr, .. }
            | Expr::Cast { expr, .. }
            | Expr::IsNull { expr, .. }
            | Expr::InSubquery { expr, .. }
            | Expr::Interval { value: expr, .. } => one(expr, sink),
            Expr::BinaryOp { left, right, .. } => {
                one(left, sink);
                one(right, sink);
            }
            Expr::Function { args, params, .. } => {
                many(args, sink);
                many(params, sink);
            }
            Expr::Window { args, params, spec, .. } => {
                many(args, sink);
                many(params, sink);
                many(&mut spec.partition_by, sink);
                for o in &mut spec.order_by {
                    one(&mut o.expr, sink);
                }
            }
            Expr::Case { operand, when_then, else_result } => {
                if let Some(o) = operand {
                    one(o, sink);
                }
                for (w, t) in when_then {
                    one(w, sink);
                    one(t, sink);
                }
                if let Some(e) = else_result {
                    one(e, sink);
                }
            }
            Expr::InList { expr, list, .. } => {
                one(expr, sink);
                many(list, sink);
            }
            Expr::Between { expr, low, high, .. } => {
                one(expr, sink);
                one(low, sink);
                one(high, sink);
            }
            Expr::Like { expr, pattern, .. } => {
                one(expr, sink);
                one(pattern, sink);
            }
            Expr::Tuple(items) => many(items, sink),
            Expr::Literal(_)
            | Expr::Column(_)
            | Expr::Wildcard
            | Expr::Subquery(_)
            | Expr::Exists { .. } => {}
        }
    }
}

impl Expr {
    /// The worklist teardown, kept out of line so that `drop`'s fast path is
    /// small enough for the glue to inline: this runs for a handful of nodes
    /// per query, the fast path runs for every one.
    #[inline(never)]
    #[cold]
    fn tear_down(&mut self) {
        // `try_with`, not `with`: an `Expr` freed during thread teardown can
        // outlive the TLS slot, and that must not turn a drop into a panic.
        let mut work = TEARDOWN.try_with(std::cell::Cell::take).unwrap_or_default();
        self.unlink(&mut work);
        // Each popped node is unlinked before it goes out of scope, so the
        // reentrant `drop` at the end of the iteration finds only leaves and
        // pushes nothing. Stack depth is O(1) for any input.
        while let Some(mut e) = work.pop() {
            e.unlink(&mut work);
        }
        let _ = TEARDOWN.try_with(move |slot| slot.set(work));
    }
}

impl Drop for Expr {
    #[inline]
    fn drop(&mut self) {
        // Almost every node stops here: with no interior child the glue's own
        // recursion is one frame deep, so it is left alone and neither the TLS
        // slot nor the worklist is touched. Taking the pool on every node
        // instead measured +40% on statement teardown against +24% for this.
        if self.has_interior_child() {
            self.tear_down();
        }
    }
}

impl SetExpr {
    /// Only `SetOperation` grows by loop (`a UNION b UNION c ...`); the
    /// `Select`/`Query` arms are reached by depth-counted descent.
    fn unlink(&mut self, sink: &mut Vec<SetExpr>) {
        if let SetExpr::SetOperation { left, right, .. } = self {
            for b in [left, right] {
                if matches!(**b, SetExpr::SetOperation { .. }) {
                    sink.push(mem::replace(&mut **b, SetExpr::Values(Vec::new())));
                }
            }
        }
    }
}

impl Drop for SetExpr {
    fn drop(&mut self) {
        let mut work = Vec::new();
        self.unlink(&mut work);
        while let Some(mut s) = work.pop() {
            s.unlink(&mut work);
        }
    }
}

impl TableRef {
    /// `FROM a, b, c, ...` and `a JOIN b JOIN c ...` are one loop, so the
    /// join tree is left-deep and as tall as the FROM list is long.
    fn unlink(&mut self, sink: &mut Vec<TableRef>) {
        if let TableRef::Join { left, right, .. } = self {
            for b in [left, right] {
                if matches!(**b, TableRef::Join { .. }) {
                    let hole = TableRef::Table {
                        name: ObjectName(Vec::new()),
                        alias: None,
                        final_: false,
                    };
                    sink.push(mem::replace(&mut **b, hole));
                }
            }
        }
    }
}

impl Drop for TableRef {
    fn drop(&mut self) {
        let mut work = Vec::new();
        self.unlink(&mut work);
        while let Some(mut t) = work.pop() {
            t.unlink(&mut work);
        }
    }
}

// ------------------------------------------------------------------ display
// Rendering the AST back to SQL powers EXPLAIN AST, SHOW CREATE TABLE, and
// the default column names for unaliased projections.

/// One unit of pending output for [`Expr`]'s renderer: either a subtree still
/// to be rendered, or literal text that has to land after it.
enum Piece<'a> {
    E(&'a Expr),
    S(&'a str),
    /// `CAST`'s target type -- the only non-`Expr`, non-`str` tail there is.
    T(&'a DataType),
    /// An `OVER (...)` body. Rendered whole by [`WindowSpec`]'s own `Display`
    /// rather than decomposed into pieces: a spec cannot nest inside a spec, so
    /// the one extra frame it costs is bounded at one.
    W(&'a WindowSpec),
}

/// Push `v` rendered as `a, b, c`, reversed so that popping emits it forwards.
fn push_list<'a>(out: &mut Vec<Piece<'a>>, v: &'a [Expr]) {
    for (i, e) in v.iter().enumerate().rev() {
        out.push(Piece::E(e));
        if i > 0 {
            out.push(Piece::S(", "));
        }
    }
}

impl Expr {
    /// Write this node's own text and push whatever must follow, in reverse.
    /// Never touches a child directly -- that is what keeps the frame flat.
    fn render<'a>(&'a self, f: &mut fmt::Formatter<'_>, out: &mut Vec<Piece<'a>>) -> fmt::Result {
        match self {
            Expr::Literal(v) => write!(f, "{v}"),
            Expr::Column(n) => write!(f, "{n}"),
            Expr::Wildcard => f.write_str("*"),
            Expr::UnaryOp { op, expr } => {
                out.push(Piece::E(expr));
                f.write_str(match op {
                    UnaryOp::Neg => "-",
                    UnaryOp::Not => "NOT ",
                })
            }
            Expr::BinaryOp { left, op, right } => {
                // Three pieces, not five: the pre-spaced symbol keeps the
                // pending stack at two entries per level, which for a 200k-term
                // chain is the difference between 6 MB and 13 MB of worklist.
                out.push(Piece::E(right));
                out.push(Piece::S(op.spaced()));
                out.push(Piece::E(left));
                Ok(())
            }
            Expr::Function { name, args, params, distinct } => {
                out.push(Piece::S(")"));
                push_list(out, args);
                if *distinct {
                    out.push(Piece::S("DISTINCT "));
                }
                if !params.is_empty() {
                    out.push(Piece::S(")("));
                    push_list(out, params);
                }
                write!(f, "{name}(")
            }
            Expr::Window { name, args, params, distinct, spec } => {
                out.push(Piece::W(spec));
                out.push(Piece::S(") OVER "));
                push_list(out, args);
                if *distinct {
                    out.push(Piece::S("DISTINCT "));
                }
                if !params.is_empty() {
                    out.push(Piece::S(")("));
                    push_list(out, params);
                }
                write!(f, "{name}(")
            }
            Expr::Cast { expr, ty } => {
                out.push(Piece::S(")"));
                out.push(Piece::T(ty));
                out.push(Piece::S(" AS "));
                out.push(Piece::E(expr));
                f.write_str("CAST(")
            }
            Expr::Case { operand, when_then, else_result } => {
                out.push(Piece::S(" END"));
                if let Some(e) = else_result {
                    out.push(Piece::E(e));
                    out.push(Piece::S(" ELSE "));
                }
                for (w, t) in when_then.iter().rev() {
                    out.push(Piece::E(t));
                    out.push(Piece::S(" THEN "));
                    out.push(Piece::E(w));
                    out.push(Piece::S(" WHEN "));
                }
                if let Some(o) = operand {
                    out.push(Piece::E(o));
                    out.push(Piece::S(" "));
                }
                f.write_str("CASE")
            }
            Expr::InList { expr, list, negated } => {
                out.push(Piece::S(")"));
                push_list(out, list);
                out.push(Piece::S(" IN ("));
                if *negated {
                    out.push(Piece::S(" NOT"));
                }
                out.push(Piece::E(expr));
                Ok(())
            }
            Expr::InSubquery { expr, negated, .. } => {
                out.push(Piece::S(" IN (SELECT ...)"));
                if *negated {
                    out.push(Piece::S(" NOT"));
                }
                out.push(Piece::E(expr));
                Ok(())
            }
            Expr::Between { expr, low, high, negated } => {
                out.push(Piece::E(high));
                out.push(Piece::S(" AND "));
                out.push(Piece::E(low));
                out.push(Piece::S(" BETWEEN "));
                if *negated {
                    out.push(Piece::S(" NOT"));
                }
                out.push(Piece::E(expr));
                Ok(())
            }
            Expr::Like { expr, pattern, negated, case_insensitive } => {
                out.push(Piece::E(pattern));
                out.push(Piece::S(if *case_insensitive { " ILIKE " } else { " LIKE " }));
                if *negated {
                    out.push(Piece::S(" NOT"));
                }
                out.push(Piece::E(expr));
                Ok(())
            }
            Expr::IsNull { expr, negated } => {
                out.push(Piece::S(if *negated { " IS NOT NULL" } else { " IS NULL" }));
                out.push(Piece::E(expr));
                Ok(())
            }
            Expr::Tuple(items) => {
                out.push(Piece::S(")"));
                push_list(out, items);
                f.write_str("(")
            }
            Expr::Subquery(_) => f.write_str("(SELECT ...)"),
            Expr::Exists { negated, .. } => {
                f.write_str(if *negated { "NOT EXISTS (SELECT ...)" } else { "EXISTS (SELECT ...)" })
            }
            Expr::Interval { value, unit } => {
                out.push(Piece::S(unit.name()));
                out.push(Piece::S(" "));
                out.push(Piece::E(value));
                f.write_str("INTERVAL ")
            }
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Explicit worklist for the same reason `Drop` has one, but the
        // threshold is far lower: `write!(f, "{left} {} {right}")` builds a
        // fresh `Arguments` and formatter frame per level, so the recursive
        // version died at 8k terms on an 8 MiB stack -- and `display_name()`
        // runs this on every unaliased projection, so `SELECT 1+1+...` aborted
        // during *binding*, not just under EXPLAIN AST. Measured after: 200k
        // terms render fine.
        //
        // A leaf never pushes, so `Vec::new()` never allocates for the
        // `Column`/`Literal` cases that dominate `display_name()`. Anything
        // deeper pays one allocation for the worklist: `display_name()` on a
        // six-node expression measured 415ns -> 450ns, +9%. Left as is on
        // purpose -- the alternative is keeping the recursive formatter beside
        // this one behind a depth budget (what `visit` does), and two copies of
        // the rendering rules that must not drift is a worse trade for 35ns
        // than it is for the 46ns `visit` was losing on a far hotter path.
        let mut out = Vec::new();
        self.render(f, &mut out)?;
        while let Some(p) = out.pop() {
            match p {
                Piece::E(e) => e.render(f, &mut out)?,
                Piece::S(s) => f.write_str(s)?,
                Piece::T(t) => write!(f, "{t}")?,
                Piece::W(w) => write!(f, "{w}")?,
            }
        }
        Ok(())
    }
}

impl fmt::Display for OrderByExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.expr, if self.asc { "" } else { " DESC" })?;
        // Only when it was written: this feeds the default column name of an
        // unaliased `... OVER (ORDER BY x)`, and echoing a placement the user
        // did not ask for would make the name drift from the source text.
        match self.nulls_first {
            Some(true) => f.write_str(" NULLS FIRST"),
            Some(false) => f.write_str(" NULLS LAST"),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    // ------------------------------------------------ iterative teardown
    //
    // The loop-grown spines. `parser::MAX_DEPTH` does not see these, so the
    // only thing standing between a 60k-term ORM `IN` list and a SIGABRT is
    // the manual `Drop` above.

    /// A stack far too small to hold one frame per node, so a regression back
    /// to recursion fails hard here instead of only on someone else's machine.
    /// 100k nodes at the ~40 bytes/frame the derived glue used would need
    /// 4 MB; this gives it 512 KB.
    fn on_a_small_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(512 << 10)
            .spawn(f)
            .expect("spawn")
            .join()
            .expect("something in the AST still recurses once per node");
    }

    /// Every left-associative production the parser builds with a loop.
    /// Long enough to blow a small stack if anything recurses per node, but
    /// under the parser's `MAX_CHAIN` so these stay tests of the *AST* rather
    /// than of the parser's refusal. The programmatic tests below go far past
    /// the parser's cap, because `Expr` is public and a library caller can
    /// build a chain the parser would never produce.
    const CHAIN_TERMS: usize = 4_000;

    /// An `n`-deep left-leaning `BinaryOp` chain, built without the parser.
    fn deep_expr(n: usize) -> Expr {
        let mut e = Expr::lit(1i64);
        for _ in 0..n {
            e = Expr::binary(e, BinaryOp::Or, Expr::lit(1i64));
        }
        e
    }

    fn chain(sep: &str, term: &str) -> String {
        let mut s = String::with_capacity(CHAIN_TERMS * (term.len() + sep.len()));
        for i in 0..CHAIN_TERMS {
            if i > 0 {
                s.push_str(sep);
            }
            s.push_str(term);
        }
        s
    }

    /// The AST must free a chain deeper than the parser will ever build one.
    ///
    /// `MAX_CHAIN` in the parser stops a *query* from producing this, and that
    /// is the layer users hit. It is not the only layer: `Expr` is public, so a
    /// caller can construct any depth, and the iterative `Drop` here is what
    /// keeps that from aborting the process. Built directly rather than parsed,
    /// precisely so this keeps testing the AST if the parser's policy changes.
    #[test]
    fn the_ast_frees_a_chain_deeper_than_the_parser_will_build() {
        on_a_small_stack(|| {
            drop(deep_expr(100_000));
        });
    }

    /// Same depth, but reached through the containers a query would nest it in
    /// -- a boxed subquery projection and a Vec of select items -- since those
    /// have their own `Drop` glue.
    #[test]
    fn the_ast_frees_a_deep_chain_through_its_containers() {
        on_a_small_stack(|| {
            let items = vec![
                SelectItem::Expr { expr: deep_expr(50_000), alias: None },
                SelectItem::Expr { expr: deep_expr(50_000), alias: Some("a".into()) },
            ];
            drop(items);
            drop(Box::new(deep_expr(100_000)));
            drop(vec![deep_expr(20_000), deep_expr(20_000), deep_expr(20_000)]);
        });
    }

    #[test]
    fn loop_grown_chains_parse_and_free_without_recursing() {
        on_a_small_stack(|| {
            for sql in [
                format!("SELECT 1 WHERE {}", chain(" OR ", "1=1")),
                format!("SELECT 1 WHERE {}", chain(" AND ", "1=1")),
                format!("SELECT {}", chain("+", "1")),
                format!("SELECT {}", chain("||", "'x'")),
                format!("SELECT 1 WHERE {}", chain(" = ", "1")),
                chain(" UNION ALL ", "SELECT 1"),
                format!("SELECT 1 FROM {}", chain(", ", "t")),
                format!("SELECT 1 FROM {}", chain(" JOIN ", "t")),
            ] {
                let stmts = crate::sql::parse(&sql).expect("chains are grammatical");
                assert_eq!(stmts.len(), 1);
                drop(stmts);
            }
        });
    }

    #[test]
    fn a_deep_chain_nested_in_a_subquery_also_frees() {
        on_a_small_stack(|| {
            let inner = chain(" OR ", "1=1");
            for sql in [
                format!("SELECT 1 WHERE 1 IN (SELECT 1 WHERE {inner})"),
                format!("SELECT (SELECT 1 WHERE {inner})"),
                format!("WITH c AS (SELECT 1 WHERE {inner}) SELECT * FROM c"),
                format!("SELECT 1 FROM (SELECT 1 WHERE {inner}) s"),
            ] {
                drop(crate::sql::parse(&sql).expect("subqueries nest one level"));
            }
        });
    }

    #[test]
    fn a_deep_partial_tree_is_freed_on_the_error_path() {
        // The worst version of the bug: the parser has already built the chain
        // when it hits the syntax error, so the tree is dropped while an `Err`
        // unwinds -- a path no successful query ever exercises.
        on_a_small_stack(|| {
            for sql in [
                format!("SELECT {} FROM", chain("+", "1")),
                format!("SELECT 1 WHERE {} AND", chain(" OR ", "1=1")),
                format!("{} UNION ALL", chain(" UNION ALL ", "SELECT 1")),
                format!("SELECT 1 FROM {} JOIN", chain(" JOIN ", "t")),
            ] {
                assert!(crate::sql::parse(&sql).is_err(), "expected a parse error");
            }
        });
    }

    /// The teardown has to reach an `OVER` clause's expressions too. A missed
    /// arm here leaks in silence -- no stack-depth test would notice, because a
    /// window spec cannot be nested deeply enough to overflow anything.
    #[test]
    fn teardown_reaches_into_a_window_spec() {
        use std::sync::Arc;
        let payload: Arc<str> = Arc::from("payload");
        let leaf = || Expr::Literal(Value::Str(Arc::clone(&payload)));
        // One copy in an argument, one in a PARTITION BY key, one in an ORDER
        // BY key, and each buried under an interior node so the worklist rather
        // than the derived glue is what has to find it.
        let deep = || Expr::binary(leaf(), BinaryOp::Plus, leaf());
        let e = Expr::Window {
            name: "sum".into(),
            args: vec![deep()],
            params: vec![deep()],
            distinct: false,
            spec: Box::new(WindowSpec {
                partition_by: vec![deep()],
                order_by: vec![OrderByExpr { expr: deep(), asc: true, nulls_first: None }],
                frame: None,
            }),
        };
        assert_eq!(Arc::strong_count(&payload), 9);
        drop(e);
        assert_eq!(Arc::strong_count(&payload), 1, "the teardown leaked a window node");
    }

    #[test]
    fn a_window_call_survives_the_deep_chain_machinery() {
        on_a_small_stack(|| {
            let sql = format!(
                "SELECT sum({}) OVER (PARTITION BY {}) FROM t",
                chain("+", "1"),
                chain("+", "1")
            );
            drop(crate::sql::parse(&sql).expect("a window over a long chain is grammatical"));
        });
    }

    #[test]
    fn teardown_frees_every_node_rather_than_leaking_it() {
        // A manual `Drop` that forgets a child leaks in silence and no
        // stack-depth test would notice. Hang one shared `Arc` off every
        // literal in the chain and watch the count come all the way back.
        use std::sync::Arc;
        let payload: Arc<str> = Arc::from("payload");
        let leaf = || Expr::Literal(Value::Str(Arc::clone(&payload)));
        let mut e = leaf();
        for _ in 0..50_000 {
            e = Expr::binary(e, BinaryOp::Or, leaf());
        }
        assert_eq!(Arc::strong_count(&payload), 50_002);
        drop(e);
        assert_eq!(Arc::strong_count(&payload), 1, "the teardown leaked a node");
    }

    #[test]
    fn display_and_visit_survive_a_chain_too_deep_to_recurse() {
        on_a_small_stack(|| {
            let sql = chain("+", "1");
            let e = crate::sql::parser::parse_expr(&sql).unwrap();
            // `display_name()` is what the binder calls for an unaliased
            // projection, so this is on the path of `SELECT 1+1+...`.
            assert_eq!(e.display_name(), sql.replace('+', " + "));
            let mut nodes = 0usize;
            e.visit(&mut |_| nodes += 1);
            assert_eq!(nodes, 2 * CHAIN_TERMS - 1);
            assert!(!e.contains_aggregate(&|n| n == "sum"));
        });
    }

    #[test]
    fn object_name_parts() {
        let n = ObjectName(vec!["db".into(), "t".into(), "c".into()]);
        assert_eq!(n.last(), "c");
        assert_eq!(n.qualifier(), Some("t"));
        assert_eq!(n.to_string(), "db.t.c");
        assert_eq!(ObjectName::bare("x").qualifier(), None);
    }

    #[test]
    fn comparison_flip_is_an_involution() {
        for op in [BinaryOp::Lt, BinaryOp::LtEq, BinaryOp::Gt, BinaryOp::GtEq, BinaryOp::Eq] {
            assert_eq!(op.flip().flip(), op);
        }
        assert_eq!(BinaryOp::Lt.flip(), BinaryOp::Gt);
        assert_eq!(BinaryOp::Eq.flip(), BinaryOp::Eq);
    }

    #[test]
    fn visit_reaches_nested_expressions() {
        let e = Expr::binary(
            Expr::func("sum", vec![Expr::col("x")]),
            BinaryOp::Plus,
            Expr::lit(1i64),
        );
        let mut cols = Vec::new();
        e.visit(&mut |x| {
            if let Expr::Column(n) = x {
                cols.push(n.to_string());
            }
        });
        assert_eq!(cols, vec!["x"]);
        assert!(e.contains_aggregate(&|n| n == "sum"));
        assert!(!e.contains_aggregate(&|n| n == "avg"));
    }

    #[test]
    fn display_roundtrips_readably() {
        let e = Expr::binary(Expr::col("a"), BinaryOp::Gt, Expr::lit(3i64));
        assert_eq!(e.to_string(), "a > 3");

        let f = Expr::Function {
            name: "quantile".into(),
            args: vec![Expr::col("latency")],
            params: vec![Expr::lit(0.9)],
            distinct: false,
        };
        assert_eq!(f.to_string(), "quantile(0.9)(latency)");

        let d = Expr::Function {
            name: "count".into(),
            args: vec![Expr::col("x")],
            params: vec![],
            distinct: true,
        };
        assert_eq!(d.to_string(), "count(DISTINCT x)");
    }

    #[test]
    fn nulls_first_default_follows_direction() {
        let asc = OrderByExpr { expr: Expr::col("a"), asc: true, nulls_first: None };
        let desc = OrderByExpr { expr: Expr::col("a"), asc: false, nulls_first: None };
        assert!(asc.nulls_first_effective());
        assert!(!desc.nulls_first_effective());
        let forced = OrderByExpr { expr: Expr::col("a"), asc: false, nulls_first: Some(true) };
        assert!(forced.nulls_first_effective());
    }

    #[test]
    fn interval_unit_parse_tolerates_plurals() {
        assert_eq!(IntervalUnit::parse("days"), Some(IntervalUnit::Day));
        assert_eq!(IntervalUnit::parse("MONTH"), Some(IntervalUnit::Month));
        assert_eq!(IntervalUnit::parse("fortnight"), None);
    }
}
