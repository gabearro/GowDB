//! The logical plan and its typed expression language.
//!
//! This is the contract between the binder (which produces it from an AST) and
//! the executor (which runs it). The defining difference from
//! [`crate::sql::ast`] is that everything here is **resolved**: every column is
//! an index into a known schema, every expression knows its own `DataType`,
//! and every function call points at a concrete registry entry. The executor
//! never looks anything up by name.

use crate::common::{Error, Result};
use crate::exec::functions::{AggFn, ScalarFn};
use crate::sql::ast::{BinaryOp, JoinOp, UnaryOp};
use crate::types::{DataType, Schema, Value};

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
    /// `x IN (SELECT ...)` never reaches the binder: [`crate::session`]
    /// evaluates an uncorrelated subquery up front and folds its result into
    /// this literal list, which runs it once instead of once per row.
    /// Correlated subqueries are rejected — there is no semi-join operator.
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
