//! The logical plan and its typed expression language.
//!
//! This is the contract between the binder (which produces it from an AST) and
//! the executor (which runs it). The defining difference from
//! [`crate::sql::ast`] is that everything here is **resolved**: every column is
//! an index into a known schema, every expression knows its own `DataType`,
//! and every function call points at a concrete registry entry. The executor
//! never looks anything up by name.

use crate::common::{Error, Result};
use crate::exec::functions::{aggregate, AggFn, ScalarFn};
use crate::sql::ast::{BinaryOp, JoinOp, UnaryOp};
use crate::types::{DataType, Field, Schema, Value};

// -------------------------------------------------------------- expressions

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CmpOp {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

impl CmpOp {
    pub fn from_binary(op: BinaryOp) -> Option<CmpOp> {
        Some(match op {
            BinaryOp::Eq => CmpOp::Eq,
            BinaryOp::NotEq => CmpOp::NotEq,
            BinaryOp::Lt => CmpOp::Lt,
            BinaryOp::LtEq => CmpOp::LtEq,
            BinaryOp::Gt => CmpOp::Gt,
            BinaryOp::GtEq => CmpOp::GtEq,
            _ => return None,
        })
    }
    pub fn flip(&self) -> CmpOp {
        match self {
            CmpOp::Lt => CmpOp::Gt,
            CmpOp::LtEq => CmpOp::GtEq,
            CmpOp::Gt => CmpOp::Lt,
            CmpOp::GtEq => CmpOp::LtEq,
            other => *other,
        }
    }
}

#[derive(Clone)]
pub enum BoundExpr {
    Literal {
        value: Value,
        ty: DataType,
    },
    /// Index into the input block's columns.
    Column {
        index: usize,
        ty: DataType,
        name: String,
    },
    Unary {
        op: UnaryOp,
        expr: Box<BoundExpr>,
        ty: DataType,
    },
    Binary {
        left: Box<BoundExpr>,
        op: BinaryOp,
        right: Box<BoundExpr>,
        ty: DataType,
    },
    Scalar {
        func: &'static ScalarFn,
        args: Vec<BoundExpr>,
        ty: DataType,
    },
    Cast {
        expr: Box<BoundExpr>,
        ty: DataType,
    },
    Case {
        /// `CASE x WHEN ...` desugars into `CASE WHEN x = ...`, so by this
        /// point there is never an operand.
        when_then: Vec<(BoundExpr, BoundExpr)>,
        else_result: Option<Box<BoundExpr>>,
        ty: DataType,
    },
    /// `x IN (literals)`.
    ///
    /// Only literals. `x IN (SELECT ...)` binds to a semi-join over the
    /// subquery's *plan* instead — see [`LogicalPlan::in_subquery`] — so a
    /// subquery result is never spliced in here as a value list.
    InList {
        expr: Box<BoundExpr>,
        list: Vec<Value>,
        negated: bool,
    },
    Like {
        expr: Box<BoundExpr>,
        pattern: String,
        negated: bool,
        case_insensitive: bool,
    },
    IsNull {
        expr: Box<BoundExpr>,
        negated: bool,
    },
}

impl BoundExpr {
    pub fn ty(&self) -> DataType {
        match self {
            BoundExpr::Literal { ty, .. }
            | BoundExpr::Column { ty, .. }
            | BoundExpr::Unary { ty, .. }
            | BoundExpr::Binary { ty, .. }
            | BoundExpr::Scalar { ty, .. }
            | BoundExpr::Cast { ty, .. }
            | BoundExpr::Case { ty, .. } => ty.clone(),
            BoundExpr::InList { .. } | BoundExpr::Like { .. } | BoundExpr::IsNull { .. } => {
                DataType::Bool
            }
        }
    }

    pub fn lit(v: Value) -> BoundExpr {
        let ty = v.data_type();
        BoundExpr::Literal { value: v, ty }
    }

    /// A literal if this expression is constant, else `None`. Drives constant
    /// folding and predicate analysis.
    pub fn as_literal(&self) -> Option<&Value> {
        match self {
            BoundExpr::Literal { value, .. } => Some(value),
            _ => None,
        }
    }

    pub fn as_column(&self) -> Option<usize> {
        match self {
            BoundExpr::Column { index, .. } => Some(*index),
            _ => None,
        }
    }

    pub fn visit<F: FnMut(&BoundExpr)>(&self, f: &mut F) {
        f(self);
        match self {
            BoundExpr::Unary { expr, .. }
            | BoundExpr::Cast { expr, .. }
            | BoundExpr::InList { expr, .. }
            | BoundExpr::Like { expr, .. }
            | BoundExpr::IsNull { expr, .. } => expr.visit(f),
            BoundExpr::Binary { left, right, .. } => {
                left.visit(f);
                right.visit(f);
            }
            BoundExpr::Scalar { args, .. } => args.iter().for_each(|a| a.visit(f)),
            BoundExpr::Case { when_then, else_result, .. } => {
                for (w, t) in when_then {
                    w.visit(f);
                    t.visit(f);
                }
                if let Some(e) = else_result {
                    e.visit(f);
                }
            }
            BoundExpr::Literal { .. } | BoundExpr::Column { .. } => {}
        }
    }

    /// Column indices this expression reads.
    pub fn referenced_columns(&self) -> Vec<usize> {
        let mut out = Vec::new();
        self.visit(&mut |e| {
            if let BoundExpr::Column { index, .. } = e {
                if !out.contains(index) {
                    out.push(*index);
                }
            }
        });
        out
    }

    /// Rewrite column indices, for pushing an expression through a projection.
    pub fn remap_columns(&mut self, map: &dyn Fn(usize) -> Option<usize>) -> Result<()> {
        let mut err = None;
        self.remap_inner(map, &mut err);
        match err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn remap_inner(&mut self, map: &dyn Fn(usize) -> Option<usize>, err: &mut Option<Error>) {
        match self {
            BoundExpr::Column { index, name, .. } => match map(*index) {
                Some(n) => *index = n,
                None => {
                    *err = Some(Error::bind(format!(
                        "column `{name}` is not available after this operator"
                    )))
                }
            },
            BoundExpr::Unary { expr, .. }
            | BoundExpr::Cast { expr, .. }
            | BoundExpr::InList { expr, .. }
            | BoundExpr::Like { expr, .. }
            | BoundExpr::IsNull { expr, .. } => expr.remap_inner(map, err),
            BoundExpr::Binary { left, right, .. } => {
                left.remap_inner(map, err);
                right.remap_inner(map, err);
            }
            BoundExpr::Scalar { args, .. } => {
                args.iter_mut().for_each(|a| a.remap_inner(map, err))
            }
            BoundExpr::Case { when_then, else_result, .. } => {
                for (w, t) in when_then.iter_mut() {
                    w.remap_inner(map, err);
                    t.remap_inner(map, err);
                }
                if let Some(e) = else_result {
                    e.remap_inner(map, err);
                }
            }
            BoundExpr::Literal { .. } => {}
        }
    }

    /// Split an AND-tree into its conjuncts. Predicate pushdown works on
    /// conjuncts individually, so this is the first thing the optimizer does.
    pub fn split_conjuncts(self) -> Vec<BoundExpr> {
        match self {
            BoundExpr::Binary { left, op: BinaryOp::And, right, .. } => {
                let mut v = left.split_conjuncts();
                v.extend(right.split_conjuncts());
                v
            }
            other => vec![other],
        }
    }

    /// Rebuild an AND-tree from conjuncts. `None` for an empty list, which
    /// callers read as "no predicate at all".
    pub fn join_conjuncts(parts: Vec<BoundExpr>) -> Option<BoundExpr> {
        let mut it = parts.into_iter();
        let first = it.next()?;
        Some(it.fold(first, |acc, e| BoundExpr::Binary {
            left: Box::new(acc),
            op: BinaryOp::And,
            right: Box::new(e),
            ty: DataType::Bool,
        }))
    }
}

impl std::fmt::Display for BoundExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BoundExpr::Literal { value, .. } => write!(f, "{value}"),
            BoundExpr::Column { name, index, .. } => write!(f, "{name}#{index}"),
            BoundExpr::Unary { op, expr, .. } => match op {
                UnaryOp::Neg => write!(f, "-({expr})"),
                UnaryOp::Not => write!(f, "NOT ({expr})"),
            },
            BoundExpr::Binary { left, op, right, .. } => {
                write!(f, "({left} {} {right})", op.symbol())
            }
            BoundExpr::Scalar { func, args, .. } => {
                write!(f, "{}(", func.name)?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{a}")?;
                }
                write!(f, ")")
            }
            BoundExpr::Cast { expr, ty } => write!(f, "CAST({expr} AS {ty})"),
            BoundExpr::Case { when_then, else_result, .. } => {
                write!(f, "CASE")?;
                for (w, t) in when_then {
                    write!(f, " WHEN {w} THEN {t}")?;
                }
                if let Some(e) = else_result {
                    write!(f, " ELSE {e}")?;
                }
                write!(f, " END")
            }
            BoundExpr::InList { expr, list, negated } => {
                write!(f, "{expr}{} IN (", if *negated { " NOT" } else { "" })?;
                for (i, v) in list.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, ")")
            }
            BoundExpr::Like { expr, pattern, negated, case_insensitive } => write!(
                f,
                "{expr}{} {} '{pattern}'",
                if *negated { " NOT" } else { "" },
                if *case_insensitive { "ILIKE" } else { "LIKE" }
            ),
            BoundExpr::IsNull { expr, negated } => {
                write!(f, "{expr} IS{} NULL", if *negated { " NOT" } else { "" })
            }
        }
    }
}

/// A bound aggregate call.
pub struct BoundAgg {
    pub func: &'static AggFn,
    pub args: Vec<BoundExpr>,
    pub params: Vec<Value>,
    pub distinct: bool,
    pub ty: DataType,
    /// Output column name.
    pub name: String,
}

impl Clone for BoundAgg {
    fn clone(&self) -> Self {
        BoundAgg {
            func: self.func,
            args: self.args.clone(),
            params: self.params.clone(),
            distinct: self.distinct,
            ty: self.ty.clone(),
            name: self.name.clone(),
        }
    }
}

#[derive(Clone)]
pub struct SortKey {
    pub expr: BoundExpr,
    pub asc: bool,
    pub nulls_first: bool,
}

/// A `col <op> literal` predicate the scan can evaluate against zone maps
/// before reading any data.
#[derive(Clone, Debug)]
pub struct ZoneFilter {
    /// Index into the scan's *projected* schema.
    pub col: usize,
    pub op: CmpOp,
    pub value: Value,
}

impl ZoneFilter {
    /// Can a granule whose values span `[min, max]` satisfy this predicate?
    ///
    /// `min`/`max` are `Value::Null` for an all-NULL granule, which can never
    /// satisfy a comparison.
    pub fn may_match(&self, min: &Value, max: &Value) -> bool {
        if min.is_null() || max.is_null() {
            // All-NULL granule (or unknown bounds): only `IS NULL`-shaped
            // predicates could match, and those are not ZoneFilters.
            return false;
        }
        match self.op {
            CmpOp::Eq => &self.value >= min && &self.value <= max,
            // `!=` prunes only when every value is the excluded one.
            CmpOp::NotEq => !(min == max && min == &self.value),
            CmpOp::Lt => min < &self.value,
            CmpOp::LtEq => min <= &self.value,
            CmpOp::Gt => max > &self.value,
            CmpOp::GtEq => max >= &self.value,
        }
    }
}

// ------------------------------------------------------------ logical plans

pub struct ScanNode {
    pub table: String,
    /// Table column indices to read, in output order.
    pub projection: Vec<usize>,
    /// Schema after projection: what downstream operators see.
    pub schema: Schema,
    /// Predicates evaluated inside the scan, against the projected schema.
    /// This is ClickHouse's PREWHERE: filtering before the rest of the
    /// pipeline sees a row.
    pub filters: Vec<BoundExpr>,
    /// Granule-level pruning derived from `filters`.
    pub zone_filters: Vec<ZoneFilter>,
}

pub enum LogicalPlan {
    Scan(Box<ScanNode>),
    Filter {
        input: Box<LogicalPlan>,
        predicate: BoundExpr,
    },
    Project {
        input: Box<LogicalPlan>,
        exprs: Vec<BoundExpr>,
        schema: Schema,
    },
    Aggregate {
        input: Box<LogicalPlan>,
        group: Vec<BoundExpr>,
        aggs: Vec<BoundAgg>,
        /// `[group..., agg...]`.
        schema: Schema,
    },
    Sort {
        input: Box<LogicalPlan>,
        keys: Vec<SortKey>,
    },
    /// Window functions: the input's columns, plus one per function.
    ///
    /// The payload lives in `exec::operators::window` because describing a
    /// window step and running one need exactly the same fields; splitting them
    /// across two modules would only create a pair to keep in step. The binder
    /// puts a `Sort` on the partition and order keys directly underneath.
    Window {
        input: Box<LogicalPlan>,
        node: Box<crate::exec::operators::window::WindowNode>,
    },
    Limit {
        input: Box<LogicalPlan>,
        limit: Option<usize>,
        offset: usize,
    },
    /// ClickHouse `LIMIT n BY (keys)`.
    LimitBy {
        input: Box<LogicalPlan>,
        limit: usize,
        keys: Vec<BoundExpr>,
    },
    Distinct {
        input: Box<LogicalPlan>,
    },
    Join {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        op: JoinOp,
        /// Equi-join column pairs: `(left index, right index)`.
        on: Vec<(usize, usize)>,
        /// Residual non-equi predicate, against the concatenated schema.
        residual: Option<BoundExpr>,
        schema: Schema,
    },
    Union {
        inputs: Vec<LogicalPlan>,
        all: bool,
        schema: Schema,
    },
    /// A literal row set: `VALUES`, and the source of `INSERT ... VALUES`.
    Values {
        rows: Vec<Vec<Value>>,
        schema: Schema,
    },
    /// Zero rows with a known schema.
    Empty {
        schema: Schema,
    },
}

impl LogicalPlan {
    pub fn schema(&self) -> &Schema {
        match self {
            LogicalPlan::Scan(s) => &s.schema,
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::LimitBy { input, .. }
            | LogicalPlan::Distinct { input } => input.schema(),
            LogicalPlan::Window { node, .. } => &node.schema,
            LogicalPlan::Project { schema, .. }
            | LogicalPlan::Aggregate { schema, .. }
            | LogicalPlan::Join { schema, .. }
            | LogicalPlan::Union { schema, .. }
            | LogicalPlan::Values { schema, .. }
            | LogicalPlan::Empty { schema } => schema,
        }
    }

    pub fn children(&self) -> Vec<&LogicalPlan> {
        match self {
            LogicalPlan::Scan(_) | LogicalPlan::Values { .. } | LogicalPlan::Empty { .. } => {
                Vec::new()
            }
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Project { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::LimitBy { input, .. }
            | LogicalPlan::Window { input, .. }
            | LogicalPlan::Distinct { input } => vec![input],
            LogicalPlan::Join { left, right, .. } => vec![left, right],
            LogicalPlan::Union { inputs, .. } => inputs.iter().collect(),
        }
    }

    fn label(&self) -> String {
        match self {
            LogicalPlan::Scan(s) => {
                let cols: Vec<&str> = s.schema.fields().iter().map(|f| f.name.as_str()).collect();
                let mut out = format!("Scan {} [{}]", s.table, cols.join(", "));
                if !s.filters.is_empty() {
                    let fs: Vec<String> = s.filters.iter().map(|f| f.to_string()).collect();
                    out.push_str(&format!(" prewhere={}", fs.join(" AND ")));
                }
                if !s.zone_filters.is_empty() {
                    out.push_str(&format!(" zonemap={}", s.zone_filters.len()));
                }
                out
            }
            LogicalPlan::Filter { predicate, .. } => format!("Filter {predicate}"),
            LogicalPlan::Project { exprs, schema, .. } => {
                let items: Vec<String> = exprs
                    .iter()
                    .zip(schema.fields())
                    .map(|(e, f)| format!("{e} AS {}", f.name))
                    .collect();
                format!("Project [{}]", items.join(", "))
            }
            LogicalPlan::Aggregate { group, aggs, .. } => {
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
            LogicalPlan::Sort { keys, .. } => {
                let k: Vec<String> = keys
                    .iter()
                    .map(|k| format!("{}{}", k.expr, if k.asc { "" } else { " DESC" }))
                    .collect();
                format!("Sort [{}]", k.join(", "))
            }
            LogicalPlan::Window { node, .. } => node.label(),
            LogicalPlan::Limit { limit, offset, .. } => match limit {
                Some(l) => format!("Limit {l} offset {offset}"),
                None => format!("Offset {offset}"),
            },
            LogicalPlan::LimitBy { limit, keys, .. } => {
                let k: Vec<String> = keys.iter().map(|e| e.to_string()).collect();
                format!("LimitBy {limit} by [{}]", k.join(", "))
            }
            LogicalPlan::Distinct { .. } => "Distinct".into(),
            LogicalPlan::Join { op, on, residual, .. } => {
                let pairs: Vec<String> =
                    on.iter().map(|(l, r)| format!("l#{l} = r#{r}")).collect();
                let mut s = format!("{op:?}Join on [{}]", pairs.join(", "));
                if let Some(r) = residual {
                    s.push_str(&format!(" residual={r}"));
                }
                s
            }
            LogicalPlan::Union { all, .. } => {
                format!("Union{}", if *all { " All" } else { " Distinct" })
            }
            LogicalPlan::Values { rows, .. } => format!("Values {} rows", rows.len()),
            LogicalPlan::Empty { .. } => "Empty".into(),
        }
    }

    /// Indented tree, for `EXPLAIN`.
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

// -------------------------------------------------- set-membership subqueries

/// `x IN (SELECT ...)` and `EXISTS (SELECT ...)`, with their negations, as a
/// **semi-join** (keep a left row that has a match) or an **anti-join** (keep
/// one that has none).
///
/// The subquery is a *relation* here, not a value. That is the whole point.
/// The old shape ran the subquery before planning and spliced its result in as
/// a literal [`BoundExpr::InList`], which meant a million-row subquery became a
/// million `Value`s held in memory and re-projected into probe lanes once per
/// block; `EXPLAIN` executed it; and the optimizer saw a list rather than a
/// relation, so nothing about it could be pruned, reordered or indexed. As a
/// join it streams, it prunes, and it is the node a decorrelation pass would
/// rewrite -- there is no way to correlate a list that has already forgotten
/// which table it came from.
///
/// ## The four cases set-membership implementations get wrong
///
/// `WHERE` keeps a row only when the predicate is TRUE, so UNKNOWN and FALSE
/// are indistinguishable *there* -- but they are not the same predicate, and
/// the difference surfaces the moment a NULL appears:
///
///   1. `x IN (S)` where `S` yields a NULL and nothing equal to `x`: the answer
///      is **NULL**, not FALSE. A semi-join is exact for it anyway, because a
///      NULL key matches nothing, so both answers drop the row.
///   2. `x NOT IN (S)` with a NULL **anywhere** in `S`: the answer is NULL for
///      *every* row, however unrelated that NULL is, so nothing survives. A
///      plain anti-join would happily return every unmatched row.
///   3. `EXISTS (S)` is **never NULL**. It is a pure existence test and ignores
///      NULLs entirely, which is exactly the keyless shape: `S LIMIT 1`, cross
///      joined on, with no key for a NULL to appear in.
///   4. `S` empty: `IN` is FALSE, `NOT IN` is **TRUE**, `EXISTS` is FALSE.
///      `NOT IN` is the trap -- it is TRUE *even for a NULL `x`*, vacuously,
///      because with no `y` there is nothing to be unknown about. The literal
///      splice got this wrong: `NULL NOT IN ()` became an empty `InList`, whose
///      NULL input answered NULL and dropped the row. Measured against sqlite
///      on four rows including one NULL, `count()` was 3 where it is 4.
///
/// Cases 2 and 4 are both properties of `S` as a whole rather than of any row,
/// and 4 is the one that makes them a single problem: `x IS NOT NULL` is *not*
/// the right guard, because an empty `S` admits a NULL `x`. So `NOT IN` keeps a
/// row exactly when
///
/// ```text
///   no y is NULL   AND   no y = x   AND   (x IS NOT NULL OR S is empty)
/// ```
///
/// and the first and third clauses come from one keyless join against a
/// two-aggregate census of `S` -- `count()` and `count(y)`, whose difference is
/// "S has a NULL" and whose first is "S is empty". One extra pass, not two, and
/// none at all when neither side is nullable, which is the usual case because
/// the subquery is usually a key.
///
/// ## Shape
///
/// ```text
///   x IN (S)         Join Inner on [x = s#0]     left, Distinct(S)
///   EXISTS (S)       Join Cross                  left, Limit 1 (S -> true)
///   NOT EXISTS (S)   Filter g IS NULL
///                      Join Left                 left, Limit 1 (S -> true)
///   x NOT IN (S)     Filter rows = nonnull AND (x IS NOT NULL OR rows = 0)
///                      Join Cross                anti, Aggregate [count(), count(y)] S
///                    where anti is
///                      Filter s#0 IS NULL
///                        Join Left on [x = s#0]  left, Distinct(S)
/// ```
///
/// The census sits *above* the anti-join rather than below it because the
/// hash join materializes both its inputs: feeding it the anti-join's output
/// costs the rows that survived, where feeding the anti-join a widened `left`
/// costs all of them.
///
/// `Distinct` is what a real semi-join operator would not need: without it an
/// inner join multiplies a left row by its number of matches. It also bounds
/// the build side by the subquery's *distinct* count rather than its row count,
/// which is the memory the literal list could not bound at all.
///
/// ## Measured, and where it is not a win
///
/// Two 1M-row `UInt64` tables on disk, `k` the primary key, A/B interleaved
/// against the literal splice in one loop, best of 5 per side:
///
/// ```text
///   SELECT count() FROM a WHERE k IN     (SELECT k FROM b)   2513.9 ->  177.3 ms  14.2x
///   SELECT count() FROM a WHERE k NOT IN (SELECT k FROM b)   2420.3 ->  189.0 ms  12.8x
///     ... the same with a Nullable(UInt64) subquery column   2444.3 ->  187.9 ms  13.0x
///   peak RSS of that first query                              245.3 ->  173.8 MB
///   EXPLAIN PIPELINE of it                                   75.192 ->  0.060 ms  1253x
///   peak RSS of the EXPLAIN                                   162.5 ->   15.8 MB
/// ```
///
/// The EXPLAIN figure is the point of the change rather than a side effect:
/// 15.8 MB is the process floor (16.7 MB with the data mapped), so describing
/// the plan now allocates nothing measurable where it used to run the query.
///
/// It is **not** a win for a small subquery, and the reason is the join
/// operator rather than the rewrite: it materializes both its inputs, so a
/// cross join of a 1M-row left against a *one-row* right already costs 15.9 ms
/// (16 ns/row), where the same scan probing a 9-element literal list costs
/// 2.78 ms and `pk IN (9 literals)` lowers to an `IndexLookup` and costs
/// 0.335 ms. Sweeping the subquery's size against a 1M-row outer:
///
/// ```text
///   rows in the subquery      1     10    100     1k    10k   100k   200k   400k   700k     1M
///   probe is the primary key  .02x  .02x  .01x   .01x   .05x   .26x   .78x  1.14x  9.67x  14.2x
///   probe is not indexed      .14x        .12x                2.97x
/// ```
///
/// So the crossover is ~300k rows when the probe column is the outer table's
/// primary key -- because that is what the literal list buys, an index probe --
/// and ~30k when it is not. Below it the fold is the better plan and the place
/// to choose between them is the fold itself, which is the only code that knows
/// how many rows the subquery produced; above it the list is 300k `Value`s held
/// in memory and re-projected into probe lanes once per block, which is the
/// 2.5 s. The structural fix is the index-nested-loop strategy already in
/// `exec::operators::join::choose` -- `Scan a` is keyed and the subquery is the
/// probe side, which is exactly the shape it fetches by key -- once the planner
/// attaches it; that removes the fold's only remaining advantage.
///
/// `EXISTS` has no such crossover and loses outright when it is uncorrelated
/// and the outer is large: 0.198 -> 14.876 ms for `WHERE EXISTS (SELECT k FROM
/// b WHERE k = 5)` over 1M rows, because the old plan folded it to `true`, the
/// filter vanished and `count()` came out of part metadata. It wins by 32x
/// (4.16 -> 0.13 ms) when the outer is small and the subquery is not, because
/// `LIMIT 1` answers an existence test the fold drains a whole table for. The
/// node is still the right thing to *have* -- a correlated `EXISTS` has no
/// constant to fold to -- but for the uncorrelated case a fold capped at one
/// row dominates it, and that cap belongs in the fold.
impl LogicalPlan {
    /// `probe IN (sub)`, or its negation.
    ///
    /// `sub_nulls` is a **second binding of the same subquery**, required when
    /// [`needs_null_census`] says so and ignored otherwise. It is a separate
    /// plan rather than a clone because `LogicalPlan` is deliberately not
    /// `Clone`; binding twice is a planner-time cost, the cheap kind, and the
    /// executor still makes one streaming pass per plan.
    pub fn in_subquery(
        left: LogicalPlan,
        probe: BoundExpr,
        sub: LogicalPlan,
        sub_nulls: Option<LogicalPlan>,
        negated: bool,
    ) -> Result<LogicalPlan> {
        if sub.schema().len() != 1 {
            return Err(Error::bind(format!(
                "IN (SELECT ...) must select exactly one column, got {}",
                sub.schema().len()
            )));
        }
        let sub_ty = sub.schema().fields()[0].ty.clone();
        // Not a real promotion, only the compatibility check the literal splice
        // used to get from `coerce_literal`: the join matches through `Value`'s
        // cross-representation equality, so `Int64 IN (Float64)` is fine and
        // needs no cast, while `String IN (Int64)` must stay the bind error it
        // has always been rather than become a silently empty answer.
        DataType::promote(&probe.ty(), &sub_ty)?;
        if !negated {
            return Ok(semi(left, Some(probe), distinct(sub)));
        }
        let probe_ty = probe.ty();
        let (left, k) = keyed(left, probe);
        let key = col_at(&left, k);
        let plan = anti(left, Some(k), distinct(sub));
        match sub_nulls {
            Some(n) => census_guard(plan, key, n, &probe_ty, &sub_ty),
            None => Ok(plan),
        }
    }

    /// `EXISTS (sub)`, or its negation.
    pub fn exists_subquery(left: LogicalPlan, sub: LogicalPlan, negated: bool) -> LogicalPlan {
        let one = exists_probe(sub);
        if negated {
            anti(left, None, one)
        } else {
            semi(left, None, one)
        }
    }
}

/// Does `probe NOT IN (sub)` need the census join -- cases 2 and 4 -- or is it
/// a plain anti-join?
pub fn needs_null_census(probe: &DataType, sub: &DataType) -> bool {
    probe.is_nullable() || sub.is_nullable()
}

/// `sub` reduced to "did it produce a row": one non-NULL column, one row at
/// most. `LIMIT 1` is what makes an existence test cost a granule rather than a
/// table, and the literal `true` is what keeps case 3 out of NULL territory --
/// the probed column cannot be NULL, so the anti-join's `IS NULL` test means
/// "no row" and nothing else.
fn exists_probe(sub: LogicalPlan) -> LogicalPlan {
    let one = BoundExpr::lit(Value::Bool(true));
    let schema = Schema::new_unchecked(vec![Field::new("exists", one.ty())]);
    LogicalPlan::Limit {
        input: Box::new(LogicalPlan::Project {
            input: Box::new(sub),
            exprs: vec![one],
            schema,
        }),
        limit: Some(1),
        offset: 0,
    }
}

fn distinct(sub: LogicalPlan) -> LogicalPlan {
    LogicalPlan::Distinct { input: Box::new(sub) }
}

/// Cases 2 and 4 as one row above an anti-join: `[count(), count(y)]` over the
/// subquery, cross joined on, then tested.
///
/// An empty group list is what makes the cross join safe -- an aggregate with
/// no `GROUP BY` emits exactly one row even over no input, which is precisely
/// the "S is empty" fact case 4 needs and which a `LIMIT 1` marker could not
/// deliver (an empty S produces no marker row to test).
fn census_guard(
    left: LogicalPlan,
    key: BoundExpr,
    sub: LogicalPlan,
    probe_ty: &DataType,
    sub_ty: &DataType,
) -> Result<LogicalPlan> {
    let func = aggregate("count").expect("`count` is a registry builtin");
    let y = col_at(&sub, 0);
    let agg = |args: Vec<BoundExpr>, name: &str| -> Result<BoundAgg> {
        let tys: Vec<DataType> = args.iter().map(|a| a.ty()).collect();
        Ok(BoundAgg {
            func,
            ty: (func.ret)(&tys, &[])?,
            args,
            params: Vec::new(),
            distinct: false,
            name: name.into(),
        })
    };
    let aggs = vec![agg(Vec::new(), "rows")?, agg(vec![y], "nonnull")?];
    let schema = Schema::new_unchecked(
        aggs.iter().map(|a| Field::new(a.name.clone(), a.ty.clone())).collect(),
    );
    let census =
        LogicalPlan::Aggregate { input: Box::new(sub), group: Vec::new(), aggs, schema };

    let n = left.schema().len();
    let joined = append_join(left, census, Vec::new(), JoinOp::Cross);
    let (rows, nonnull) = (col_at(&joined, n), col_at(&joined, n + 1));
    let cmp = |l: BoundExpr, op: BinaryOp, r: BoundExpr| BoundExpr::Binary {
        left: Box::new(l),
        op,
        right: Box::new(r),
        ty: DataType::Bool,
    };
    let mut conj = Vec::new();
    if sub_ty.is_nullable() {
        conj.push(cmp(rows.clone(), BinaryOp::Eq, nonnull));
    }
    if probe_ty.is_nullable() {
        // Not `x IS NOT NULL`: case 4 says an empty subquery admits a NULL `x`.
        // Both operands are counts or an `IS NULL`, so neither can be NULL and
        // the OR is two-valued.
        conj.push(cmp(
            BoundExpr::IsNull { expr: Box::new(key), negated: true },
            BinaryOp::Or,
            cmp(rows, BinaryOp::Eq, BoundExpr::lit(Value::UInt(0))),
        ));
    }
    Ok(match BoundExpr::join_conjuncts(conj) {
        Some(predicate) => LogicalPlan::Filter { input: Box::new(joined), predicate },
        // `needs_null_census` is the caller's gate and it is the disjunction of
        // the two tests above, so this is unreachable -- but a census whose
        // every clause folded away is still a correct, if pointless, plan.
        None => joined,
    })
}

fn col_at(plan: &LogicalPlan, i: usize) -> BoundExpr {
    let f = &plan.schema().fields()[i];
    BoundExpr::Column { index: i, ty: f.ty.clone(), name: f.name.clone() }
}

/// Make `probe` a column of `left`, so the join can name it in `on`.
///
/// A bare column -- `WHERE k IN (...)`, which is nearly every real query -- is
/// already one and costs nothing. Anything else buys one `Project`, whose
/// pass-through columns the executor hands on by reference.
fn keyed(left: LogicalPlan, probe: BoundExpr) -> (LogicalPlan, usize) {
    if let Some(i) = probe.as_column() {
        return (left, i);
    }
    let n = left.schema().len();
    let mut exprs: Vec<BoundExpr> = (0..n).map(|i| col_at(&left, i)).collect();
    let mut fields = left.schema().fields().to_vec();
    fields.push(Field::new("in", probe.ty()));
    exprs.push(probe);
    let schema = Schema::new_unchecked(fields);
    (LogicalPlan::Project { input: Box::new(left), exprs, schema }, n)
}

/// `left` joined to a one-column `right`, right's column appended.
///
/// The extra column is left in the output on purpose. Every operator above a
/// `SELECT`'s `WHERE` was bound against the source scope, so indices `0..n`
/// still mean what they meant, and the projection the binder puts on top drops
/// the appendage for free -- where a `Project` inserted here to restore the
/// width would be an operator per row for nothing.
fn append_join(
    left: LogicalPlan,
    right: LogicalPlan,
    on: Vec<(usize, usize)>,
    op: JoinOp,
) -> LogicalPlan {
    let mut fields = left.schema().fields().to_vec();
    // A LEFT join invents a NULL for every unmatched row and the anti-join's
    // entire test is `IS NULL` on exactly that column, so its declared type has
    // to admit one.
    let pads = matches!(op, JoinOp::Left);
    fields.extend(right.schema().fields().iter().map(|f| {
        Field::new(f.name.clone(), if pads { f.ty.to_nullable() } else { f.ty.clone() })
    }));
    let schema = Schema::new_unchecked(fields);
    LogicalPlan::Join {
        left: Box::new(left),
        right: Box::new(right),
        op,
        on,
        residual: None,
        schema,
    }
}

/// Keep left rows that have a match. `None` probe is the keyless existence
/// test, where "a match" means `right` produced any row at all.
fn semi(left: LogicalPlan, probe: Option<BoundExpr>, right: LogicalPlan) -> LogicalPlan {
    match probe {
        Some(p) => {
            let (left, k) = keyed(left, p);
            append_join(left, right, vec![(k, 0)], JoinOp::Inner)
        }
        None => append_join(left, right, Vec::new(), JoinOp::Cross),
    }
}

/// Keep left rows that have none -- an outer join plus "the padding fired".
///
/// The mark column is `right`'s only column, which the LEFT join set to NULL
/// for exactly the unmatched rows. That needs `right` never to yield a NULL of
/// its own where it matters, and it does not: for `NOT EXISTS` the column is a
/// literal `true`, and for `NOT IN` a NULL on the right cannot have matched
/// anything, so a row carrying one is a matched row and would fail the test
/// anyway. A `NULL` *probe* is the case this cannot see, and `census_guard` is
/// where that is settled.
fn anti(left: LogicalPlan, key: Option<usize>, right: LogicalPlan) -> LogicalPlan {
    let mark = left.schema().len();
    let on = key.map(|k| vec![(k, 0)]).unwrap_or_default();
    let joined = append_join(left, right, on, JoinOp::Left);
    LogicalPlan::Filter {
        predicate: BoundExpr::IsNull { expr: Box::new(col_at(&joined, mark)), negated: false },
        input: Box::new(joined),
    }
}

// --------------------------------------------------------------- mutations

/// What a [`MutationPlan`] does with the rows its source selects.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MutationKind {
    Delete,
    Update,
}

/// A bulk `DELETE` or `UPDATE`: a row set, plus what to do with it.
///
/// ## Why this is not a `LogicalPlan` variant
///
/// A mutation is not a relation. It has no schema anyone selects from, it
/// cannot be an operand of a join or a union, and every pass that walks
/// `LogicalPlan` -- pushdown, projection pruning, physical lowering, operator
/// construction -- would need an arm for a node that can only ever be the root.
/// So it *wraps* a plan instead. `source` is an ordinary logical plan and goes
/// through `optimizer::optimize` and `physical::lower` unchanged, which is the
/// entire point: an `UPDATE ... WHERE pk = 3` gets predicate pushdown, zone-map
/// pruning and the primary-key `IndexLookup` because it is running the same
/// code a `SELECT ... WHERE pk = 3` runs, not a second copy of it.
///
/// ## The cost model this shape exists to allow
///
/// `source` selects rows; it does not rewrite them one at a time. Applying it
/// is **O(parts)** in published state: evaluate the predicate block-at-a-time,
/// OR the resulting mask into each part's delete bitmap, publish one new
/// `PartSet` version for the whole statement. Parts are immutable and the set
/// is copy-on-write, so hiding a million rows costs one bitmap per part and one
/// pointer store -- not a million tombstones through the delta, and not a
/// write-ahead record per row.
///
/// Measured, because the difference is not a constant factor. A 200k-row table
/// on a persistent session, `DELETE FROM t WHERE id < n`, A/B interleaved (one
/// round each, five rounds, best per side) against the same statement through
/// `session::run_alter_delete`, which tombstones through the delta one key at a
/// time and logs a record per key:
///
/// ```text
///   rows deleted   per-key path      bitmap sweep     marginal us/row
///        1 000        7.07 ms          0.75 ms         7.07  -> 0.75
///       20 000       57.16 ms          0.81 ms         2.86  -> 0.041
///      100 000      258.72 ms          1.22 ms         2.59  -> 0.012
/// ```
///
/// The sweep figure is publish cost only; a real implementation adds one WAL
/// record and one fsync **per statement** (~4 ms here), which the per-key path
/// already pays once as well -- so end to end the 100k case is 258.7 ms against
/// ~5.2 ms, 50x, and the *marginal* cost per affected row falls 215x. In memory
/// (no WAL at all) the same three rows are 1.2x / 2.3x / 5.9x: the per-key
/// path's real problem is that it makes the row count buy write-ahead records
/// and delta flushes, and neither is a thing a delete needs.
///
/// ## What the executor is promised
///
///   * `table` is a **qualified** path (`db.tbl`), ready for
///     `Catalog::table_by_path_mut`.
///   * `source` reads exactly one table -- `table` -- and produces its matching
///     rows in scan order, so each output row can be paired with the
///     `(part, position)` it came from.
///   * For [`MutationKind::Update`], `source.schema()` **is** the table's
///     schema: one column per table column, in declaration order, already cast
///     to the declared type. The block it yields is the replacement row image
///     and needs no widening before `Table::insert`.
///   * For [`MutationKind::Delete`], `source.schema()` is whatever the
///     predicate happened to need and its *values* are meaningless. Only which
///     rows survive matters, so the executor should read no column it does not
///     need to evaluate the filter.
///
/// An `UPDATE` on a table with a primary key can skip the delete half: the
/// re-insert tombstones the old row by key on its own (that is what makes
/// `ReplacingMergeTree` last-write-wins). Without a key there is no such
/// shortcut and both halves must run, which is exactly why the row set is
/// identified positionally here rather than by key.
pub struct MutationPlan {
    /// Qualified path of the table being changed.
    pub table: String,
    /// The rows to change.
    pub source: LogicalPlan,
    pub kind: MutationKind,
}

impl MutationPlan {
    /// Indented tree, for `EXPLAIN`. The root names the mutation; everything
    /// below it is the ordinary plan that finds the rows, so an index probe or
    /// a zone-map prune shows up here exactly as it would under a `SELECT`.
    pub fn explain(&self) -> String {
        let mut out = match self.kind {
            MutationKind::Delete => format!("Delete {}\n", self.table),
            MutationKind::Update => format!("Update {}\n", self.table),
        };
        self.source.explain_into(1, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Field;

    fn col(i: usize, ty: DataType) -> BoundExpr {
        BoundExpr::Column { index: i, ty, name: format!("c{i}") }
    }

    #[test]
    fn conjunct_split_and_rejoin_roundtrips() {
        let e = BoundExpr::Binary {
            left: Box::new(BoundExpr::Binary {
                left: Box::new(col(0, DataType::Bool)),
                op: BinaryOp::And,
                right: Box::new(col(1, DataType::Bool)),
                ty: DataType::Bool,
            }),
            op: BinaryOp::And,
            right: Box::new(col(2, DataType::Bool)),
            ty: DataType::Bool,
        };
        let parts = e.split_conjuncts();
        assert_eq!(parts.len(), 3);
        let rejoined = BoundExpr::join_conjuncts(parts).unwrap();
        assert_eq!(rejoined.split_conjuncts().len(), 3);
        assert!(BoundExpr::join_conjuncts(vec![]).is_none());
    }

    #[test]
    fn or_is_not_split() {
        let e = BoundExpr::Binary {
            left: Box::new(col(0, DataType::Bool)),
            op: BinaryOp::Or,
            right: Box::new(col(1, DataType::Bool)),
            ty: DataType::Bool,
        };
        assert_eq!(e.split_conjuncts().len(), 1);
    }

    #[test]
    fn referenced_columns_are_deduped() {
        let e = BoundExpr::Binary {
            left: Box::new(col(3, DataType::Int64)),
            op: BinaryOp::Plus,
            right: Box::new(col(3, DataType::Int64)),
            ty: DataType::Int64,
        };
        assert_eq!(e.referenced_columns(), vec![3]);
    }

    #[test]
    fn remap_rewrites_or_reports() {
        let mut e = col(5, DataType::Int64);
        e.remap_columns(&|i| Some(i * 2)).unwrap();
        assert_eq!(e.as_column(), Some(10));

        let mut e = col(5, DataType::Int64);
        assert!(e.remap_columns(&|_| None).is_err());
    }

    #[test]
    fn zone_filter_pruning_logic() {
        let zf = |op, v: i64| ZoneFilter { col: 0, op, value: Value::Int(v) };
        let (min, max) = (Value::Int(10), Value::Int(20));

        assert!(zf(CmpOp::Eq, 15).may_match(&min, &max));
        assert!(!zf(CmpOp::Eq, 5).may_match(&min, &max));
        assert!(!zf(CmpOp::Eq, 25).may_match(&min, &max));
        assert!(zf(CmpOp::Eq, 10).may_match(&min, &max), "boundary is inclusive");

        assert!(zf(CmpOp::Lt, 15).may_match(&min, &max));
        assert!(!zf(CmpOp::Lt, 10).may_match(&min, &max), "no value < min");
        assert!(zf(CmpOp::LtEq, 10).may_match(&min, &max));

        assert!(zf(CmpOp::Gt, 15).may_match(&min, &max));
        assert!(!zf(CmpOp::Gt, 20).may_match(&min, &max), "no value > max");
        assert!(zf(CmpOp::GtEq, 20).may_match(&min, &max));

        // != prunes only a constant granule holding exactly that value
        let c = Value::Int(7);
        assert!(!zf(CmpOp::NotEq, 7).may_match(&c, &c));
        assert!(zf(CmpOp::NotEq, 8).may_match(&c, &c));
        assert!(zf(CmpOp::NotEq, 15).may_match(&min, &max));

        // all-NULL granule never matches a comparison
        assert!(!zf(CmpOp::Eq, 15).may_match(&Value::Null, &Value::Null));
    }

    #[test]
    fn zone_filter_works_on_strings() {
        let zf = ZoneFilter { col: 0, op: CmpOp::Eq, value: Value::str("m") };
        assert!(zf.may_match(&Value::str("a"), &Value::str("z")));
        assert!(!zf.may_match(&Value::str("n"), &Value::str("z")));
    }

    #[test]
    fn explain_renders_a_tree() {
        let schema = Schema::new(vec![Field::new("a", DataType::Int64)]).unwrap();
        let plan = LogicalPlan::Limit {
            input: Box::new(LogicalPlan::Filter {
                input: Box::new(LogicalPlan::Scan(Box::new(ScanNode {
                    table: "t".into(),
                    projection: vec![0],
                    schema: schema.clone(),
                    filters: vec![],
                    zone_filters: vec![],
                }))),
                predicate: col(0, DataType::Bool),
            }),
            limit: Some(10),
            offset: 0,
        };
        let e = plan.explain();
        assert!(e.starts_with("Limit 10 offset 0\n"));
        assert!(e.contains("  Filter"));
        assert!(e.contains("    Scan t [a]"));
    }

    #[test]
    fn cmp_op_flip_is_an_involution() {
        for op in [CmpOp::Eq, CmpOp::NotEq, CmpOp::Lt, CmpOp::LtEq, CmpOp::Gt, CmpOp::GtEq] {
            assert_eq!(op.flip().flip(), op);
        }
        assert_eq!(CmpOp::Lt.flip(), CmpOp::Gt);
    }
}
