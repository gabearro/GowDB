//! AST -> typed [`LogicalPlan`].
//!
//! Binding is the one place in the engine where names still exist. Above it
//! the parser deals in strings; below it the executor deals in indices. Three
//! things happen here and nowhere else:
//!
//!   * every identifier resolves to a **column index** into the schema of the
//!     operator directly beneath the expression;
//!   * every expression acquires a **`DataType`**, computed the same way the
//!     evaluator will compute its value;
//!   * every function name resolves to a `&'static ScalarFn` / `&'static
//!     AggFn`, so the executor never touches a registry.
//!
//! ## Why types come from the function registry
//!
//! `a + b` could be typed here with `DataType::promote` directly, but the
//! evaluator will run `scalar("plus")`, whose `ret` callback already encodes
//! the ClickHouse-specific corners (`Date + Int` stays a `Date`, `/` is always
//! `Nullable(Float64)` because division by zero yields NULL, `-` on an
//! unsigned widens to `Int64`). Routing operator typing through the *same*
//! callback makes it impossible for the binder's declared type and the
//! evaluator's produced type to drift apart. `promote` is still what does the
//! work; it just does it one level down.
//!
//! ## The aggregation split
//!
//! `SELECT k, sum(v) * 2 FROM t GROUP BY k` cannot be one operator: `sum(v)`
//! is computed per group and `* 2` is computed per output row. The binder
//! therefore splits every select-list expression in two:
//!
//! ```text
//!   Project [k#0, (sum(v)#1 * 2)]
//!     Aggregate group=[k#0] aggs=[sum(v#1)]
//!       Scan t [k, v]
//! ```
//!
//! The split is done on the **bound** tree rather than the AST, because only
//! bound trees are canonical: `t.k` and `k` are different AST nodes but the
//! same `Column { index: 0 }`, so matching a select expression against a
//! GROUP BY expression by `Display` string "just works" regardless of how
//! either was spelled.
//!
//! To make that possible an aggregate call has to survive binding as *some*
//! `BoundExpr`, and the enum (frozen) has no aggregate variant. So a call is
//! hoisted into a side list and left behind as `Column { index: AGG_MARK + j }`
//! -- an index far above any real schema width. The rewrite pass then does one
//! pre-order walk: markers become `Column { n_group + j }`, subtrees equal to a
//! group key become `Column { i }`, and any *surviving* plain column reference
//! is exactly the error case "column is neither grouped nor aggregated".
//!
//! ## Where ORDER BY sits
//!
//! `SELECT a FROM t ORDER BY b` is legal and common, so `Sort` is placed
//! *below* `Project`, where `b` is still in scope. `SELECT DISTINCT` inverts
//! the requirement -- deduplication must happen on the projected rows, and
//! sorting must happen after that -- so in that one case `Sort` moves above
//! `Project` and ORDER BY resolves against the output schema instead. Those
//! are the only two shapes; everything else stacks the same way.
//!
//! ## Why `USING` is a scope rule and not a plan rule
//!
//! `a JOIN b USING (k)` produces **one** `k`, not two, and for an outer join
//! its value is `COALESCE(a.k, b.k)`. The join *operator* cannot express that:
//! it concatenates its two inputs and has no idea that two of the columns
//! share a name. So the merge happens entirely in the [`Scope`] -- the two
//! per-side entries are marked `shadowed` (still reachable as `a.k` / `b.k`,
//! which for an outer join genuinely differ from the merged value, and still
//! expanded by `a.*`) and a third, unqualified entry is added that says which
//! side to read. `*` and an unqualified `k` then see exactly one column, and
//! the coalesce, where one is needed at all, is an ordinary expression in
//! whatever projection or predicate named it.
//!
//! ## What is refused rather than approximated
//!
//! A scalar subquery needs a query *result* at bind time and the binder has no
//! executor to get one from, so it returns `Error::unsupported` instead of a
//! plan that silently means something else. `EXCEPT`/`INTERSECT` and
//! `GROUP BY ... WITH TOTALS` have no corresponding `LogicalPlan` node and are
//! refused for the same reason.
//!
//! `x IN (SELECT ...)` and `EXISTS (SELECT ...)` need no result: as a whole
//! `WHERE` conjunct they are a semi-join or an anti-join over the subquery's
//! own plan (see [`LogicalPlan::in_subquery`]), and `split_membership` is what
//! separates that position from the ones -- inside an `OR`, a `CASE`, a select
//! item -- where the test really does have to produce a value per row and is
//! still refused.
//!
//! `FINAL` is the one thing accepted and dropped: `ScanNode` has no slot for
//! it, so a `ReplacingMergeTree` read is whatever the storage layer gives back.

use std::cell::Cell;

use crate::catalog::Catalog;
use crate::common::{Error, Result};
use crate::exec::functions::{aggregate, scalar, AggFn};
use crate::exec::operators::window::{self, BoundWindow};
use crate::sql::ast::{
    BinaryOp, Expr, IntervalUnit, JoinConstraint, JoinOp, ObjectName, OrderByExpr, Query, Select,
    SelectItem, SetExpr, SetOp, TableRef, UnaryOp, WindowSpec,
};
use crate::types::{parse_date, parse_datetime, DataType, Field, Schema, Value};

use super::logical::{
    needs_null_census, BoundAgg, BoundExpr, LogicalPlan, MutationKind, MutationPlan, ScanNode,
    SortKey,
};

/// Marker base for hoisted aggregate calls. Any real column index is bounded
/// by the widest schema in the query, which is orders of magnitude below this;
/// keeping it well under `usize::MAX` means `AGG_MARK + j` cannot overflow.
const AGG_MARK: usize = 1 << 40;

/// The same trick for hoisted *window* calls, one octave up so the two marker
/// spaces cannot collide.
///
/// They have to be distinguishable rather than merely distinct, because the two
/// rewrites run at different times: `rewrite_over_agg` resolves aggregate
/// markers against the `Aggregate` node's schema and must leave window markers
/// strictly alone, since the `Window` node it will later be given sits *above*
/// that and has not been built yet.
const WIN_MARK: usize = 1 << 41;

/// Guards against an alias cycle that the `expanding` stack cannot see (it
/// only catches direct self-reference).
const MAX_ALIAS_DEPTH: usize = 32;

/// How deep `query` / `set_expr` / `table_ref` / `bind` may nest before the
/// query is refused.
///
/// The parser has its own, independent limit, and it is not enough: a *flat*
/// parse tree can still drive the binder arbitrarily deep. `WITH a1 AS (...),
/// a2 AS (SELECT * FROM a1), ... aN AS (SELECT * FROM aN-1) SELECT * FROM aN`
/// is one `WITH` list to the parser and N nested `query -> table_ref -> query`
/// frames here, because a CTE is inlined at every reference. Same story for
/// `a UNION b UNION c UNION ...`, which the parser folds in a loop but
/// `set_expr` walks recursively.
///
/// 200 is far past any query a human writes, and 200 frames of even the fat
/// one (`select_block`, which holds a dozen `Vec`s of locals) is a small
/// fraction of a default 8 MiB stack. The point is to fail with an error
/// rather than a SIGSEGV, not to find the true ceiling. It is the same value
/// the parser picked, but deliberately not the same constant: neither layer
/// should be able to widen the other's guard by editing its own.
const MAX_BIND_DEPTH: usize = 200;

/// RAII half of [`MAX_BIND_DEPTH`].
///
/// Every recursive binder entry point returns through `?` in a dozen places,
/// so a hand-written decrement at the exits would leak the counter on the
/// first error path anyone forgets -- and a leaked counter turns the *next*
/// query on the same `Binder` into a spurious "nests too deeply". `Drop` is
/// the only placement that survives `?`, `return` and unwinding alike.
struct DepthGuard<'d>(&'d Cell<usize>);

impl<'d> DepthGuard<'d> {
    fn enter(d: &'d Cell<usize>, what: &str) -> Result<DepthGuard<'d>> {
        let n = d.get() + 1;
        if n > MAX_BIND_DEPTH {
            return Err(Error::unsupported(format!(
                "{what} nests more than {MAX_BIND_DEPTH} levels deep; the binder recurses \
                 once per level and would run out of stack"
            )));
        }
        d.set(n);
        Ok(DepthGuard(d))
    }
}

impl Drop for DepthGuard<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}

// =========================================================== name resolution

/// Where a scope column's value comes from.
///
/// Only a `USING` join key is ever anything but [`Src::Block`]: SQL merges the
/// two sides of `JOIN ... USING (k)` into a *single* output column, and which
/// side that column reads depends on which side an outer join may have
/// NULL-padded. The merged entry keeps the left copy's block index either way,
/// so `SELECT *` emits the key where the left table put it (what ClickHouse
/// does) regardless of the join type.
#[derive(Clone, Copy)]
enum Src {
    /// The block column at [`ScopeCol::index`].
    Block,
    /// The block column at *this* index instead -- a `RIGHT JOIN ... USING`
    /// key, whose merged value is the right-hand copy because the left one is
    /// the padded side. `index` then only says where the column sorts.
    At(usize),
    /// `coalesce` of these two -- a `FULL JOIN ... USING` key, where either
    /// side may be the padded one and only the other side has the key.
    Coalesce(usize, usize),
}

impl Src {
    /// Concatenating scopes shifts every right-hand block index, including the
    /// ones a merged key points at.
    fn shift(self, by: usize) -> Src {
        match self {
            Src::Block => Src::Block,
            Src::At(i) => Src::At(i + by),
            Src::Coalesce(a, b) => Src::Coalesce(a + by, b + by),
        }
    }
}

#[derive(Clone)]
struct ScopeCol {
    /// Table alias, or the bare table name when no alias was given. `None` for
    /// an unaliased subquery, whose columns are only reachable unqualified.
    qualifier: Option<String>,
    name: String,
    ty: DataType,
    /// Position in the input block, or `None` for a source column the scan did
    /// not project.
    ///
    /// A `None` here is unreachable by a *legal* reference -- anything the
    /// query names is demanded, and anything demanded is projected -- but
    /// carrying the column anyway is what lets "unknown column `x`" list the
    /// table's real columns instead of the three the scan happened to read.
    index: Option<usize>,
    src: Src,
    /// A per-side copy of a `USING` key, superseded by the merged entry.
    ///
    /// Still reachable as `a.k` (and still expanded by `a.*`), because for an
    /// outer join `a.k` and the merged `k` genuinely differ -- but hidden from
    /// `*` and from unqualified lookups, which is what makes the join emit the
    /// key once and makes a bare `k` unambiguous rather than an error.
    shadowed: bool,
}

/// The columns an expression may name, in source order. `index` is what the
/// `BoundExpr::Column` gets; `width` is how many there actually are.
#[derive(Clone, Default)]
struct Scope {
    cols: Vec<ScopeCol>,
    width: usize,
}

impl Scope {
    fn from_schema(schema: &Schema, qualifier: Option<&str>) -> Scope {
        Scope {
            cols: schema
                .fields()
                .iter()
                .enumerate()
                .map(|(i, f)| ScopeCol {
                    qualifier: qualifier.map(|q| q.to_string()),
                    name: f.name.clone(),
                    ty: f.ty.clone(),
                    index: Some(i),
                    src: Src::Block,
                    shadowed: false,
                })
                .collect(),
            width: schema.len(),
        }
    }

    fn from_table(full: &Schema, projection: &[usize], qualifier: Option<&str>) -> Scope {
        Scope {
            cols: full
                .fields()
                .iter()
                .enumerate()
                .map(|(i, f)| ScopeCol {
                    qualifier: qualifier.map(|q| q.to_string()),
                    name: f.name.clone(),
                    ty: f.ty.clone(),
                    index: projection.iter().position(|&p| p == i),
                    src: Src::Block,
                    shadowed: false,
                })
                .collect(),
            width: projection.len(),
        }
    }

    fn width(&self) -> usize {
        self.width
    }

    /// The columns that really are in the input block, in block order, as `*`
    /// and an ORDER BY ordinal see them: the per-side copies of a `USING` key
    /// are dropped in favour of the one merged entry that supersedes them.
    fn visible(&self) -> Vec<(usize, &ScopeCol)> {
        let mut v: Vec<(usize, &ScopeCol)> = self
            .cols
            .iter()
            .filter(|c| !c.shadowed)
            .filter_map(|c| c.index.map(|i| (i, c)))
            .collect();
        v.sort_by_key(|(i, _)| *i);
        v
    }

    /// [`Scope::visible`] plus the shadowed copies, for `t.*`: a qualified star
    /// means "the columns of `t`", and a `USING` key is one of them. The merged
    /// entry carries no qualifier, so it never matches and cannot double up.
    fn visible_qualified(&self) -> Vec<(usize, &ScopeCol)> {
        let mut v: Vec<(usize, &ScopeCol)> =
            self.cols.iter().filter_map(|c| c.index.map(|i| (i, c))).collect();
        v.sort_by_key(|(i, _)| *i);
        v
    }

    fn concat(&self, other: &Scope) -> Scope {
        let mut cols = self.cols.clone();
        cols.extend(other.cols.iter().map(|c| ScopeCol {
            index: c.index.map(|i| i + self.width),
            src: c.src.shift(self.width),
            ..c.clone()
        }));
        Scope { cols, width: self.width + other.width }
    }

    /// Every column, rendered the way the user would have to write it. This is
    /// the `available: ...` tail that [`Schema::require`] produces, extended
    /// with qualifiers because a join scope has several tables in it.
    fn available(&self) -> String {
        self.cols
            .iter()
            .map(|c| match &c.qualifier {
                Some(q) => format!("{q}.{}", c.name),
                None => c.name.clone(),
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn find(&self, n: &ObjectName) -> Result<&ScopeCol> {
        Ok(&self.cols[self.find_pos(n)?])
    }

    /// Case-sensitive pass first, then case-insensitive -- the same two-step
    /// [`Schema::index_of`] uses, so `SELECT ID` finds `id`.
    ///
    /// Returns the position in `cols`, not the block index: the two differ for
    /// a `USING` key, where three entries (merged, left copy, right copy) share
    /// one block index and only the position tells them apart.
    fn find_pos(&self, n: &ObjectName) -> Result<usize> {
        let qual = n.qualifier();
        let name = n.last();
        for insensitive in [false, true] {
            let mut hit: Option<usize> = None;
            let mut ambiguous = false;
            for (pos, c) in self.cols.iter().enumerate() {
                let q_ok = match qual {
                    None => {
                        // Unqualified, so a shadowed `USING` copy is not a
                        // candidate -- it is exactly the entry whose merged
                        // twin is the answer, and counting it would make every
                        // `USING` key report itself as ambiguous.
                        !c.shadowed
                    }
                    Some(q) => c.qualifier.as_deref().is_some_and(|cq| cq.eq_ignore_ascii_case(q)),
                };
                if !q_ok {
                    continue;
                }
                let n_ok = if insensitive {
                    c.name.eq_ignore_ascii_case(name)
                } else {
                    c.name == name
                };
                if n_ok {
                    if hit.is_some() {
                        ambiguous = true;
                    } else {
                        hit = Some(pos);
                    }
                }
            }
            if ambiguous {
                return Err(Error::bind(format!(
                    "column `{n}` is ambiguous; qualify it. available: {}",
                    self.available()
                )));
            }
            if let Some(pos) = hit {
                return Ok(pos);
            }
        }
        Err(Error::bind(format!(
            "unknown column `{n}`; available: {}",
            self.available()
        )))
    }

    fn resolve(&self, n: &ObjectName) -> Result<BoundExpr> {
        let c = self.find(n)?;
        let index = c.index.ok_or_else(|| {
            Error::bind(format!("internal: column `{n}` was not projected into the scan"))
        })?;
        Ok(col_expr(index, c))
    }
}

/// The expression a reference to a scope column expands to.
///
/// Types are always re-read from `c.ty` rather than cached alongside `src`,
/// because an enclosing outer join re-runs [`nullable_scope`] over the whole
/// scope and a merged `USING` key has to widen with everything else.
fn col_expr(index: usize, c: &ScopeCol) -> BoundExpr {
    let col = |i: usize| BoundExpr::Column { index: i, ty: c.ty.clone(), name: c.name.clone() };
    match c.src {
        Src::Block => col(index),
        Src::At(i) => col(i),
        Src::Coalesce(a, b) => {
            let func = scalar("coalesce").expect("`coalesce` is a registry builtin");
            BoundExpr::Scalar { func, args: vec![col(a), col(b)], ty: c.ty.clone() }
        }
    }
}

// ================================================================== context

/// Everything an expression needs besides the AST node itself.
struct Ctx<'c> {
    scope: &'c Scope,
    /// SELECT-list aliases, AST-level. ClickHouse (unlike ANSI) lets GROUP BY,
    /// HAVING and ORDER BY see them, and lets a later select item see an
    /// earlier one, so the alias is substituted and re-bound in place.
    aliases: &'c [(String, Expr)],
    /// False in WHERE / PREWHERE / GROUP BY / JOIN ON, where an aggregate is a
    /// user error rather than something to hoist.
    allow_agg: bool,
    /// Aggregate calls hoisted out of the expressions bound with this context.
    aggs: Vec<BoundAgg>,
    /// One canonical key per entry of `aggs`, so `sum(v)` written twice binds
    /// to one accumulator.
    agg_keys: Vec<String>,
    in_agg: bool,
    /// Alias names currently being expanded, so `x + 1 AS x` terminates.
    expanding: Vec<String>,
    /// Window calls hoisted out of the expressions bound with this context,
    /// each carrying the `OVER` clause it was written with.
    windows: Vec<PendingWindow>,
    /// De-duplication keys for `windows`, same role as `agg_keys`: the same
    /// call written in the select list and in ORDER BY is computed once.
    window_keys: Vec<String>,
    /// True while binding inside an `OVER` clause or a window call's arguments,
    /// so a nested window function is refused rather than silently hoisted into
    /// a step that runs after the one it sits inside.
    in_window: bool,
}

/// A window call bound but not yet placed: the function, plus the keys of the
/// `OVER` clause it must be grouped with.
struct PendingWindow {
    func: BoundWindow,
    partition: Vec<BoundExpr>,
    order: Vec<SortKey>,
}

impl<'c> Ctx<'c> {
    fn new(scope: &'c Scope, aliases: &'c [(String, Expr)], allow_agg: bool) -> Ctx<'c> {
        Ctx {
            scope,
            aliases,
            allow_agg,
            aggs: Vec::new(),
            agg_keys: Vec::new(),
            in_agg: false,
            expanding: Vec::new(),
            windows: Vec::new(),
            window_keys: Vec::new(),
            in_window: false,
        }
    }

    fn plain(scope: &'c Scope) -> Ctx<'c> {
        Ctx::new(scope, &[], false)
    }
}

/// A CTE stack. Entry `i` may only reference entries `< i`, which is what
/// keeps a self-referential CTE from recursing forever.
type Ctes<'q> = [(&'q str, &'q Query)];

// =================================================================== binder

pub struct Binder<'a> {
    pub catalog: &'a Catalog,
    /// Current nesting of the mutually recursive `query` / `set_expr` /
    /// `table_ref` / `bind`, capped by [`MAX_BIND_DEPTH`].
    ///
    /// A `Cell` rather than a plain field because the guard has to stay alive
    /// *across* the recursive call it protects, and a `&mut self` borrow held
    /// by the guard would be the very borrow that call needs. Nothing else in
    /// the binder is mutable, so the recursive methods take `&self` and the
    /// counter is the only interior mutability in the type.
    depth: Cell<usize>,
}

impl<'a> Binder<'a> {
    pub fn new(catalog: &'a Catalog) -> Self {
        Binder { catalog, depth: Cell::new(0) }
    }

    pub fn bind_query(&mut self, q: &Query) -> Result<LogicalPlan> {
        self.query(q, &[])
    }

    /// Bind one expression against a bare schema. Used by the session layer
    /// for `ALTER ... UPDATE` assignments and column defaults, where there is
    /// no surrounding SELECT to supply a scope.
    pub fn bind_expr_standalone(&mut self, e: &Expr, schema: &Schema) -> Result<BoundExpr> {
        let scope = Scope::from_schema(schema, None);
        let mut ctx = Ctx::plain(&scope);
        self.bind(e, &mut ctx)
    }

    // ------------------------------------------------------------ mutations
    //
    // `DELETE FROM t WHERE p` and `UPDATE t SET c = e WHERE p` bind to the same
    // shape: a scan of `t` with `p` as an ordinary `Filter` above it, wrapped in
    // a [`MutationPlan`] that says what to do with the rows that survive.
    //
    // The predicate is deliberately *not* pre-installed into `ScanNode.filters`
    // here. `optimizer::optimize` is what pushes a filter into the scan, derives
    // its zone filters and folds a constant one away -- and `physical::lower` is
    // what turns `pk = <const>` into an `IndexLookup`. Handing the mutation's
    // predicate to those passes as a plain `Filter` is what makes a mutation get
    // every access-path decision a `SELECT` gets, with no second implementation
    // to keep in step. It is also why `EXPLAIN` over a mutation shows the index
    // probe: there is nothing special to teach it.

    /// `DELETE FROM table WHERE predicate`.
    ///
    /// The scan is narrowed to the columns the predicate reads -- a delete needs
    /// no key and no row *value*, only the identity of the rows, so a delete on
    /// a 40-column table with a one-column predicate decodes one column.
    pub fn bind_delete(&mut self, table: &ObjectName, predicate: &Expr) -> Result<MutationPlan> {
        let mut demand = Demand { all: false, names: Vec::new() };
        demand.walk(predicate);
        let (source, _) = self.mutation_source(table, predicate, &demand)?;
        Ok(MutationPlan {
            table: self.catalog.qualify(table),
            source,
            kind: MutationKind::Delete,
        })
    }

    /// `UPDATE table SET assignments WHERE predicate`.
    ///
    /// The plan produces the **replacement row image**: every table column in
    /// declaration order, with the assigned ones replaced by their expressions
    /// and cast to the declared type. Every right-hand side is bound against the
    /// pre-update row, so `SET a = b, b = a` swaps -- assignments are evaluated
    /// as one simultaneous projection, not applied in sequence.
    pub fn bind_update(
        &mut self,
        table: &ObjectName,
        assignments: &[(String, Expr)],
        predicate: &Expr,
    ) -> Result<MutationPlan> {
        // Every column, unconditionally: the output *is* the new row, so there
        // is nothing to narrow. Reading the whole row is the price of an update
        // on a columnar store, and it is why an UPDATE is worth more thought
        // than a DELETE at the call site.
        let all = Demand { all: true, names: Vec::new() };
        let (source, scope) = self.mutation_source(table, predicate, &all)?;
        let full = self.catalog.table(table)?.schema().clone();

        // Resolve assignment targets first, so `SET nosuch = 1` is reported
        // before anything is bound and `SET a = 1, a = 2` is reported at all --
        // the previous path silently kept whichever the linear search found.
        let mut targets = Vec::with_capacity(assignments.len());
        for (name, _) in assignments {
            let i = full.require(name)?;
            if targets.contains(&i) {
                return Err(Error::bind(format!(
                    "column `{name}` is assigned twice in one UPDATE"
                )));
            }
            targets.push(i);
        }

        let mut exprs = Vec::with_capacity(full.len());
        for (i, f) in full.fields().iter().enumerate() {
            let e = match targets.iter().position(|&t| t == i) {
                Some(a) => {
                    let mut ctx = Ctx::plain(&scope);
                    let bound = self
                        .bind(&assignments[a].1, &mut ctx)
                        .map_err(|e| annotate(e, &f.name))?;
                    // The block goes straight into `Table::insert`, which wants
                    // the declared type. Cast only on a real mismatch: the
                    // common `SET n = n + 1` already types as the column and a
                    // no-op `Cast` would cost a per-row copy of the whole
                    // column for nothing.
                    if bound.ty() == f.ty {
                        bound
                    } else {
                        BoundExpr::Cast { expr: Box::new(bound), ty: f.ty.clone() }
                    }
                }
                // Untouched columns are carried through by index. `all` demand
                // means the scan projected every column in order, so the scope
                // index and the table index are the same number.
                None => BoundExpr::Column { index: i, ty: f.ty.clone(), name: f.name.clone() },
            };
            exprs.push(e);
        }

        Ok(MutationPlan {
            table: self.catalog.qualify(table),
            source: LogicalPlan::Project { input: Box::new(source), exprs, schema: full },
            kind: MutationKind::Update,
        })
    }

    /// The shared half: scan `table`, narrowed by `demand`, with `predicate`
    /// filtering above it. Returns the scope too, so `bind_update` can bind its
    /// assignments against the same pre-update row the predicate saw.
    fn mutation_source(
        &self,
        table: &ObjectName,
        predicate: &Expr,
        demand: &Demand,
    ) -> Result<(LogicalPlan, Scope)> {
        let t = self.catalog.table(table)?;
        let full = t.schema();
        let projection = demand.project(full);
        let schema = full.project(&projection);
        let scope = Scope::from_table(full, &projection, Some(table.last()));
        let scan = LogicalPlan::Scan(Box::new(ScanNode {
            table: self.catalog.qualify(table),
            projection,
            schema,
            filters: Vec::new(),
            zone_filters: Vec::new(),
        }));
        // `Ctx::plain` refuses aggregates, which is the right answer here for
        // the same reason it is in WHERE: `DELETE ... WHERE count(*) > 1` is a
        // user error, not something to hoist into an operator that does not
        // exist on this path.
        let mut ctx = Ctx::plain(&scope);
        let pred = self.bind(predicate, &mut ctx)?;
        Ok((LogicalPlan::Filter { input: Box::new(scan), predicate: pred }, scope))
    }

    // ------------------------------------------------------------- queries

    fn query<'q>(&self, q: &'q Query, outer: &Ctes<'q>) -> Result<LogicalPlan> {
        let _guard = DepthGuard::enter(&self.depth, "query")?;
        let mut ctes: Vec<(&'q str, &'q Query)> = outer.to_vec();
        for c in &q.with {
            ctes.push((c.name.as_str(), &c.query));
        }

        match &q.body {
            // A single SELECT owns the whole query's tail, because ORDER BY
            // wants to sit under the projection (see the module docs) and only
            // `select_block` knows where that is.
            SetExpr::Select(sel) => self.select_block(sel, q, &ctes),
            other => {
                let plan = self.set_expr(other, &ctes)?;
                self.tail_above(plan, q)
            }
        }
    }

    fn set_expr<'q>(&self, e: &'q SetExpr, ctes: &Ctes<'q>) -> Result<LogicalPlan> {
        // Guarded in its own right: `a UNION b UNION c UNION ...` is a loop in
        // the parser but a left-deep `SetOperation` tree here, so this is the
        // one recursion that neither `query` nor `table_ref` sits on.
        let _guard = DepthGuard::enter(&self.depth, "set operation")?;
        match e {
            SetExpr::Select(sel) => {
                let empty = Query::simple((**sel).clone());
                self.select_block(sel, &empty, ctes)
            }
            SetExpr::Query(q) => self.query(q, ctes),
            SetExpr::Values(rows) => self.values(rows),
            SetExpr::SetOperation { op, all, left, right } => {
                if !matches!(op, SetOp::Union) {
                    let name = if matches!(op, SetOp::Except) { "EXCEPT" } else { "INTERSECT" };
                    return Err(Error::unsupported(format!(
                        "{name}: only UNION is implemented; rewrite it as a join or an \
                         anti-join predicate"
                    )));
                }
                let l = self.set_expr(left, ctes)?;
                let r = self.set_expr(right, ctes)?;
                let mut inputs = Vec::new();
                flatten_union(l, *all, &mut inputs);
                flatten_union(r, *all, &mut inputs);
                let schema = union_schema(&inputs)?;
                Ok(LogicalPlan::Union { inputs, all: *all, schema })
            }
        }
    }

    /// `VALUES (...), (...)` as a standalone row set. Every cell must be a
    /// literal: there is no input row for a column reference to point at.
    fn values(&self, rows: &[Vec<Expr>]) -> Result<LogicalPlan> {
        if rows.is_empty() {
            return Ok(LogicalPlan::Empty { schema: Schema::empty() });
        }
        let width = rows[0].len();
        let scope = Scope::default();
        let mut out: Vec<Vec<Value>> = Vec::with_capacity(rows.len());
        // A column's type is a property of *all* the rows: the base type is
        // promoted across every non-NULL cell, and a NULL anywhere makes it
        // Nullable. The two are tracked apart because `Value::Null` reports a
        // placeholder `Nullable(Int64)`, which must not be promoted against a
        // real type -- doing so would make `VALUES ('a'), (NULL)` fail to find
        // a common type. Folding row-by-row with the first row as the seed is
        // what used to make `(1),(NULL)` and `(NULL),(1)` disagree.
        let mut bases: Vec<Option<DataType>> = vec![None; width];
        let mut has_null = vec![false; width];
        for (r, row) in rows.iter().enumerate() {
            if row.len() != width {
                return Err(Error::bind(format!(
                    "VALUES row {} has {} columns, expected {width}",
                    r + 1,
                    row.len()
                )));
            }
            let mut vals = Vec::with_capacity(width);
            for (c, e) in row.iter().enumerate() {
                let mut ctx = Ctx::plain(&scope);
                let b = self.bind(e, &mut ctx)?;
                let v = b.as_literal().cloned().ok_or_else(|| {
                    Error::bind(format!("VALUES entries must be literals, got `{e}`"))
                })?;
                if v.is_null() {
                    has_null[c] = true;
                } else {
                    let ty = b.ty();
                    bases[c] = Some(match bases[c].take() {
                        None => ty,
                        Some(prev) => DataType::promote(&prev, &ty)?,
                    });
                }
                vals.push(v);
            }
            out.push(vals);
        }
        let schema = Schema::new_unchecked(
            bases
                .into_iter()
                .zip(has_null)
                .enumerate()
                .map(|(i, (base, null))| {
                    // An all-NULL column has no evidence of a base type; Int64
                    // is the same placeholder `Value::data_type` picks.
                    let t = base.unwrap_or(DataType::Int64);
                    Field::new(format!("c{}", i + 1), if null { t.to_nullable() } else { t })
                })
                .collect(),
        );
        Ok(LogicalPlan::Values { rows: out, schema })
    }

    /// ORDER BY / LIMIT BY / LIMIT applied to a plan whose schema is already
    /// final -- unions and parenthesized queries.
    fn tail_above(&self, plan: LogicalPlan, q: &Query) -> Result<LogicalPlan> {
        let scope = Scope::from_schema(plan.schema(), None);
        let mut plan = plan;
        if !q.order_by.is_empty() {
            let keys = self.sort_keys(&q.order_by, &scope, &[])?;
            plan = LogicalPlan::Sort { input: Box::new(plan), keys };
        }
        if let Some((n, by)) = &q.limit_by {
            let limit = self.const_count(n, "LIMIT n BY")?;
            let mut ctx = Ctx::plain(&scope);
            let keys = by.iter().map(|e| self.bind(e, &mut ctx)).collect::<Result<Vec<_>>>()?;
            plan = LogicalPlan::LimitBy { input: Box::new(plan), limit, keys };
        }
        self.apply_limit(plan, q)
    }

    fn apply_limit(&self, plan: LogicalPlan, q: &Query) -> Result<LogicalPlan> {
        let limit = match &q.limit {
            Some(e) => Some(self.const_count(e, "LIMIT")?),
            None => None,
        };
        let offset = match &q.offset {
            Some(e) => self.const_count(e, "OFFSET")?,
            None => 0,
        };
        if limit.is_none() && offset == 0 {
            return Ok(plan);
        }
        Ok(LogicalPlan::Limit { input: Box::new(plan), limit, offset })
    }

    /// `LIMIT` / `OFFSET` / `LIMIT n BY` counts. Bound against an empty scope,
    /// so a column reference fails to resolve -- and is reported as the "not a
    /// constant" error it really is rather than as an unknown identifier.
    fn const_count(&self, e: &Expr, what: &str) -> Result<usize> {
        let scope = Scope::default();
        let mut ctx = Ctx::plain(&scope);
        let b = self
            .bind(e, &mut ctx)
            .map_err(|_| Error::bind(format!("{what} must be a constant, got `{e}`")))?;
        let v = b
            .as_literal()
            .ok_or_else(|| Error::bind(format!("{what} must be a constant, got `{e}`")))?;
        // Deliberately not `Value::as_u64`: that truncates, so `LIMIT 2.7`
        // would quietly become `LIMIT 2` instead of being reported.
        match v {
            Value::UInt(n) => Ok(*n as usize),
            Value::Int(n) if *n >= 0 => Ok(*n as usize),
            _ => Err(Error::bind(format!(
                "{what} must be a non-negative integer, got {v}"
            ))),
        }
    }

    // -------------------------------------------------------- select block

    fn select_block<'q>(
        &self,
        sel: &'q Select,
        q: &'q Query,
        ctes: &Ctes<'q>,
    ) -> Result<LogicalPlan> {
        if sel.with_totals {
            return Err(Error::unsupported(
                "GROUP BY ... WITH TOTALS: the logical plan has no TOTALS node",
            ));
        }

        let demand = Demand::of(sel, q);
        let (mut plan, scope) = match &sel.from {
            Some(tr) => self.table_ref(tr, ctes, &demand)?,
            None => {
                if sel.projection.iter().any(|p| !matches!(p, SelectItem::Expr { .. })) {
                    return Err(Error::bind("SELECT * requires a FROM clause"));
                }
                // One row, no columns: `SELECT 1` still has to produce a row
                // for the projection to evaluate against.
                (
                    LogicalPlan::Values { rows: vec![Vec::new()], schema: Schema::empty() },
                    Scope::default(),
                )
            }
        };

        // Aliases are collected before anything is bound so that WHERE,
        // PREWHERE, GROUP BY, HAVING, ORDER BY and later select items can all
        // see them. ClickHouse allows `SELECT a+b AS s FROM t WHERE s > 1`;
        // ANSI does not, and we follow ClickHouse.
        let aliases: Vec<(String, Expr)> = sel
            .projection
            .iter()
            .filter_map(|p| match p {
                SelectItem::Expr { expr, alias: Some(a) } => Some((a.clone(), expr.clone())),
                _ => None,
            })
            .collect();

        // PREWHERE goes straight into the scan: the user asked for it by name,
        // so it is not the optimizer's call to make.
        if let Some(pw) = &sel.prewhere {
            // Aggregates stay banned here — an aggregate in WHERE is a user
            // error, not something to hoist.
            let mut ctx = Ctx::new(&scope, &aliases, false);
            let pred = self.bind(pw, &mut ctx)?;
            plan = match plan {
                LogicalPlan::Scan(mut s) => {
                    s.filters.extend(pred.split_conjuncts());
                    LogicalPlan::Scan(s)
                }
                other => LogicalPlan::Filter { input: Box::new(other), predicate: pred },
            };
        }

        if let Some(w) = &sel.selection {
            // A membership subquery at the top of the AND-spine is a *join*,
            // not a predicate. Everything else in the spine binds as usual and
            // lands **below** the joins, so each join builds over the rows that
            // survived the ordinary filter rather than over the whole table --
            // and pushdown still sinks those conjuncts into the scan, because
            // nothing between them and it has changed.
            let (rest, subs) = split_membership(w);
            let mut ctx = Ctx::new(&scope, &aliases, false);
            let mut conjuncts = Vec::with_capacity(rest.len());
            for r in rest {
                conjuncts.push(self.bind(r, &mut ctx)?);
            }
            if let Some(predicate) = BoundExpr::join_conjuncts(conjuncts) {
                plan = LogicalPlan::Filter { input: Box::new(plan), predicate };
            }
            for s in subs {
                plan = self.membership(plan, s, &scope, &aliases, ctes)?;
            }
        }

        // GROUP BY binds against the source scope, with aggregates banned:
        // `GROUP BY sum(x)` is meaningless.
        let mut group = Vec::with_capacity(sel.group_by.len());
        let mut group_fields = Vec::with_capacity(sel.group_by.len());
        {
            let mut ctx = Ctx::new(&scope, &aliases, false);
            for g in &sel.group_by {
                // `GROUP BY 1` has to bind the select item exactly the way the
                // projection did, alias shadowing included, or the two trees
                // will not match in `rewrite_over_agg`.
                let (g, g_alias) = match ordinal(g, "GROUP BY")? {
                    Some(n) => select_item_at(sel, n)?,
                    None => (g, None),
                };
                let b = self.bind_defining(g, g_alias, &mut ctx)?;
                group_fields.push(Field::new(g.display_name(), b.ty()));
                group.push(b);
            }
        }

        // One context for the select list, HAVING, ORDER BY and LIMIT BY, so
        // that `sum(v)` mentioned in two of them shares one accumulator.
        let mut ctx = Ctx::new(&scope, &aliases, true);

        let mut proj: Vec<BoundExpr> = Vec::new();
        let mut proj_names: Vec<String> = Vec::new();
        for item in &sel.projection {
            match item {
                SelectItem::Wildcard => {
                    for (i, c) in scope.visible() {
                        proj.push(col_expr(i, c));
                        proj_names.push(c.name.clone());
                    }
                }
                SelectItem::QualifiedWildcard(q) => {
                    let want = q.rsplit('.').next().unwrap_or(q);
                    let mut any = false;
                    for (i, c) in scope.visible_qualified() {
                        if c.qualifier.as_deref().is_some_and(|cq| cq.eq_ignore_ascii_case(want)) {
                            any = true;
                            proj.push(col_expr(i, c));
                            proj_names.push(c.name.clone());
                        }
                    }
                    if !any {
                        return Err(Error::bind(format!(
                            "unknown table `{q}` in `{q}.*`; available: {}",
                            scope.available()
                        )));
                    }
                }
                SelectItem::Expr { expr, alias } => {
                    proj.push(self.bind_defining(expr, alias.as_deref(), &mut ctx)?);
                    proj_names
                        .push(alias.clone().unwrap_or_else(|| expr.display_name()));
                }
            }
        }

        let having = match &sel.having {
            Some(h) => {
                let b = self.bind(h, &mut ctx)?;
                // HAVING runs *before* the window step, so a window call there
                // would be a marker no rewrite can resolve. Rejected by name
                // rather than left to produce a wrong column index.
                if has_window_marker(&b) {
                    return Err(Error::bind(
                        "a window function cannot appear in HAVING: windows are computed after \
                         HAVING has run. Wrap the query in a subquery and filter outside it",
                    ));
                }
                Some(b)
            }
            None => None,
        };

        // ORDER BY always *binds* against the source scope and the select list,
        // even under DISTINCT: that is what lets a key be written as an alias,
        // an ordinal, a repeated select-list expression or an aggregate. Where
        // the resulting Sort node ends up is a separate question, settled
        // below -- DISTINCT pushes it above the projection and each key is then
        // rewritten to the output column it already matches.
        let mut sort_keys = self.sort_keys_in(&q.order_by, &mut ctx, &proj)?;
        let mut limit_by = None;
        if !sel.distinct {
            if let Some((n, by)) = &q.limit_by {
                let limit = self.const_count(n, "LIMIT n BY")?;
                let keys = by.iter().map(|e| self.bind(e, &mut ctx)).collect::<Result<Vec<_>>>()?;
                limit_by = Some((limit, keys));
            }
        }

        let aggregating = !group.is_empty() || !ctx.aggs.is_empty();
        let mut having = having;

        if aggregating {
            let aggs = std::mem::take(&mut ctx.aggs);
            let n_group = group.len();
            let group_keys: Vec<String> = group.iter().map(|g| g.to_string()).collect();
            let mut fields = group_fields.clone();
            fields.extend(aggs.iter().map(|a| Field::new(a.name.clone(), a.ty.clone())));
            let schema = Schema::new_unchecked(fields);

            plan = LogicalPlan::Aggregate {
                input: Box::new(plan),
                group,
                aggs,
                schema,
            };

            let gf = &group_fields;
            proj = proj
                .into_iter()
                .map(|e| rewrite_over_agg(e, &group_keys, gf, n_group))
                .collect::<Result<Vec<_>>>()?;
            having = match having {
                Some(h) => Some(rewrite_over_agg(h, &group_keys, gf, n_group)?),
                None => None,
            };
            sort_keys = sort_keys
                .into_iter()
                .map(|k| {
                    Ok(SortKey {
                        expr: rewrite_over_agg(k.expr, &group_keys, gf, n_group)?,
                        asc: k.asc,
                        nulls_first: k.nulls_first,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            limit_by = match limit_by {
                Some((n, keys)) => Some((
                    n,
                    keys.into_iter()
                        .map(|k| rewrite_over_agg(k, &group_keys, gf, n_group))
                        .collect::<Result<Vec<_>>>()?,
                )),
                None => None,
            };
            // A window's own expressions were bound against the source scope
            // too -- `row_number() OVER (ORDER BY sum(v) DESC)` is the reason
            // aggregates are still allowed inside an OVER clause -- so they
            // take the same rewrite. Missing this leaves a hoisted aggregate
            // marker in the window operator's key list.
            for w in ctx.windows.iter_mut() {
                for e in w.func.args.iter_mut().chain(w.partition.iter_mut()) {
                    rewrite_in_place(e, &group_keys, gf, n_group)?;
                }
                for k in w.order.iter_mut() {
                    rewrite_in_place(&mut k.expr, &group_keys, gf, n_group)?;
                }
            }
        }

        // HAVING is a Filter above the Aggregate. Without aggregation it has
        // nothing to filter that WHERE could not, so it lands in the same
        // place WHERE would -- below the projection, against the source scope.
        if let Some(h) = having {
            plan = LogicalPlan::Filter { input: Box::new(plan), predicate: h };
        }

        // Windows sit here and nowhere else: above WHERE, GROUP BY and HAVING,
        // below the projection, DISTINCT, ORDER BY and LIMIT. That is the SQL
        // evaluation order, and it is also the only placement that lets
        // `ORDER BY rank() OVER (...)` and `SELECT DISTINCT ... OVER (...)`
        // both see a column that already exists.
        if !ctx.windows.is_empty() {
            let pending = std::mem::take(&mut ctx.windows);
            let (p, remap) = self.apply_windows(plan, pending)?;
            plan = p;
            for e in proj.iter_mut() {
                resolve_windows(e, &remap)?;
            }
            for k in sort_keys.iter_mut() {
                resolve_windows(&mut k.expr, &remap)?;
            }
            if let Some((_, keys)) = limit_by.as_mut() {
                for k in keys.iter_mut() {
                    resolve_windows(k, &remap)?;
                }
            }
        }

        let out_schema = Schema::new_unchecked(
            proj_names
                .iter()
                .zip(proj.iter())
                .map(|(n, e)| Field::new(n.clone(), e.ty()))
                .collect(),
        );

        if sel.distinct {
            // Deduplication happens on the projected rows, so Sort has to sit
            // above the Project and can only name output columns. The keys were
            // bound below it, against the same expressions the projection was
            // built from, so each one is looked up by its canonical (bound)
            // form and replaced by the output column that computes it. A key
            // that is not in the select list has no such column -- and under
            // DISTINCT it is genuinely meaningless, because the row it would
            // sort by no longer exists once duplicates are collapsed.
            let projected: Vec<String> = proj.iter().map(|e| e.to_string()).collect();
            let out_cols: Vec<BoundExpr> = out_schema
                .fields()
                .iter()
                .enumerate()
                .map(|(i, f)| BoundExpr::Column {
                    index: i,
                    ty: f.ty.clone(),
                    name: f.name.clone(),
                })
                .collect();
            let keys = sort_keys
                .into_iter()
                .map(|k| {
                    let want = k.expr.to_string();
                    let i = projected.iter().position(|p| *p == want).ok_or_else(|| {
                        Error::bind(format!(
                            "ORDER BY `{}` is not in the SELECT DISTINCT list; with DISTINCT \
                             the rows are deduplicated before sorting, so every ORDER BY key \
                             must also be selected",
                            k.expr
                        ))
                    })?;
                    Ok(SortKey {
                        expr: out_cols[i].clone(),
                        asc: k.asc,
                        nulls_first: k.nulls_first,
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            plan = LogicalPlan::Project { input: Box::new(plan), exprs: proj, schema: out_schema };
            plan = LogicalPlan::Distinct { input: Box::new(plan) };
            let out_scope = Scope::from_schema(plan.schema(), None);
            if !keys.is_empty() {
                plan = LogicalPlan::Sort { input: Box::new(plan), keys };
            }
            if let Some((n, by)) = &q.limit_by {
                let limit = self.const_count(n, "LIMIT n BY")?;
                let mut c = Ctx::plain(&out_scope);
                let keys = by.iter().map(|e| self.bind(e, &mut c)).collect::<Result<Vec<_>>>()?;
                plan = LogicalPlan::LimitBy { input: Box::new(plan), limit, keys };
            }
        } else {
            if !sort_keys.is_empty() {
                plan = LogicalPlan::Sort { input: Box::new(plan), keys: sort_keys };
            }
            if let Some((limit, keys)) = limit_by {
                plan = LogicalPlan::LimitBy { input: Box::new(plan), limit, keys };
            }
            plan = LogicalPlan::Project { input: Box::new(plan), exprs: proj, schema: out_schema };
        }

        self.apply_limit(plan, q)
    }

    /// One `WHERE` conjunct that is a membership test, as a join over `left`.
    ///
    /// The subquery binds through [`Binder::query`] like any other, which is
    /// what makes it a first-class relation: it gets the CTE stack, the depth
    /// guard, projection narrowing, and every optimizer pass that runs on the
    /// outer plan. Nothing here executes anything.
    fn membership<'q>(
        &self,
        left: LogicalPlan,
        e: &'q Expr,
        scope: &Scope,
        aliases: &[(String, Expr)],
        ctes: &Ctes<'q>,
    ) -> Result<LogicalPlan> {
        match e {
            Expr::InSubquery { expr, subquery, negated } => {
                let mut ctx = Ctx::new(scope, aliases, false);
                let probe = self.bind(expr, &mut ctx)?;
                let sub = self.query(subquery, ctes)?;
                // Cases 2 and 4 in `LogicalPlan::in_subquery`. Two types that
                // cannot hold a NULL make the census -- and the second pass
                // over the subquery it costs -- provably dead, which is the
                // common case, because the subquery is usually a key.
                let nulls = (*negated
                    && sub.schema().len() == 1
                    && needs_null_census(&probe.ty(), &sub.schema().fields()[0].ty))
                .then(|| self.query(subquery, ctes))
                .transpose()?;
                LogicalPlan::in_subquery(left, probe, sub, nulls, *negated)
            }
            Expr::Exists { subquery, negated } => {
                Ok(LogicalPlan::exists_subquery(left, self.query(subquery, ctes)?, *negated))
            }
            // `split_membership` selects the two arms above and nothing else.
            other => Err(Error::bind(format!("`{other}` is not a membership test"))),
        }
    }

    /// ORDER BY over a plan whose schema is already the output schema, so an
    /// ordinal is simply that output column.
    fn sort_keys(
        &self,
        obs: &[OrderByExpr],
        scope: &Scope,
        aliases: &[(String, Expr)],
    ) -> Result<Vec<SortKey>> {
        let cols: Vec<BoundExpr> =
            scope.visible().into_iter().map(|(i, c)| col_expr(i, c)).collect();
        let mut ctx = Ctx::new(scope, aliases, false);
        self.sort_keys_in(obs, &mut ctx, &cols)
    }

    /// `ORDER BY 2` means "the second select item"; ClickHouse accepts it and
    /// so does everyone's muscle memory. Resolving it against `proj` keeps the
    /// key in the same index space as everything else at this level.
    fn sort_keys_in(
        &self,
        obs: &[OrderByExpr],
        ctx: &mut Ctx<'_>,
        proj: &[BoundExpr],
    ) -> Result<Vec<SortKey>> {
        let mut out = Vec::with_capacity(obs.len());
        for ob in obs {
            let expr = match ordinal(&ob.expr, "ORDER BY")? {
                Some(n) if !proj.is_empty() => {
                    if n == 0 || n > proj.len() {
                        return Err(Error::bind(format!(
                            "ORDER BY position {n} is out of range (1..{})",
                            proj.len()
                        )));
                    }
                    proj[n - 1].clone()
                }
                _ => self.bind(&ob.expr, ctx)?,
            };
            out.push(SortKey {
                expr,
                asc: ob.asc,
                nulls_first: ob.nulls_first_effective(),
            });
        }
        Ok(out)
    }

    // ----------------------------------------------------------- table refs

    fn table_ref<'q>(
        &self,
        tr: &'q TableRef,
        ctes: &Ctes<'q>,
        demand: &Demand,
    ) -> Result<(LogicalPlan, Scope)> {
        let _guard = DepthGuard::enter(&self.depth, "FROM clause")?;
        match tr {
            TableRef::Table { name, alias, .. } => {
                // A CTE shadows a base table of the same name, and is bound
                // afresh at every reference -- inlining, not materialization.
                if name.0.len() == 1 {
                    if let Some(pos) =
                        ctes.iter().rposition(|(n, _)| n.eq_ignore_ascii_case(name.last()))
                    {
                        let (cte_name, q) = ctes[pos];
                        let plan = self.query(q, &ctes[..pos])?;
                        let qual = alias.as_deref().unwrap_or(cte_name);
                        let scope = Scope::from_schema(plan.schema(), Some(qual));
                        return Ok((plan, scope));
                    }
                }
                let table = self.catalog.table(name)?;
                let path = self.catalog.qualify(name);
                let full = table.schema();
                let projection = demand.project(full);
                let schema = full.project(&projection);
                let qual = alias.as_deref().unwrap_or_else(|| name.last());
                let scope = Scope::from_table(full, &projection, Some(qual));
                Ok((
                    LogicalPlan::Scan(Box::new(ScanNode {
                        table: path,
                        projection,
                        schema,
                        filters: Vec::new(),
                        zone_filters: Vec::new(),
                    })),
                    scope,
                ))
            }

            TableRef::Subquery { query, alias } => {
                let plan = self.query(query, ctes)?;
                let scope = Scope::from_schema(plan.schema(), alias.as_deref());
                Ok((plan, scope))
            }

            TableRef::Join { left, right, op, constraint } => {
                let (lp, ls) = self.table_ref(left, ctes, demand)?;
                let (rp, rs) = self.table_ref(right, ctes, demand)?;
                let ln = ls.width();

                // An outer join invents NULLs on the side that may not match,
                // so those columns' declared types have to admit them.
                let (ls, rs) = match op {
                    JoinOp::Left => (ls, nullable_scope(&rs)),
                    JoinOp::Right => (nullable_scope(&ls), rs),
                    JoinOp::Full => (nullable_scope(&ls), nullable_scope(&rs)),
                    _ => (ls, rs),
                };
                let mut joined = ls.concat(&rs);
                // The *physical* schema of the join, which still has both
                // copies of every `USING` key -- the operator concatenates the
                // two sides and knows nothing about merging. Merging is a
                // naming rule, so it happens in the scope, below.
                let schema = scope_schema(&joined);

                let (on, residual) = match constraint {
                    JoinConstraint::None => (Vec::new(), None),
                    JoinConstraint::Using(names) => {
                        let mut on = Vec::with_capacity(names.len());
                        for n in names {
                            let key = ObjectName::bare(n.clone());
                            let (lpos, rpos) = (ls.find_pos(&key)?, rs.find_pos(&key)?);
                            let (l, r) = (ls.resolve(&key)?, rs.resolve(&key)?);
                            let (Some(a), Some(b)) = (l.as_column(), r.as_column()) else {
                                // Only reachable through a *nested* USING whose
                                // inner join was FULL: that key is already a
                                // `coalesce`, and `on` holds column indices, so
                                // there is nothing to point the equi-pair at.
                                return Err(Error::unsupported(format!(
                                    "USING ({n}): `{n}` is the merged key of an enclosed FULL \
                                     JOIN, which is an expression rather than a column; write \
                                     the join condition with ON instead"
                                )));
                            };
                            // `a` / `b` are per-side block indices: `a` is
                            // already in the concatenated space, `b` shifts by
                            // the left width. `on` wants them unshifted.
                            merge_using_key(
                                &mut joined,
                                (lpos, a),
                                (ls.cols.len() + rpos, ln + b),
                                *op,
                            )?;
                            on.push((a, b));
                        }
                        (on, None)
                    }
                    JoinConstraint::On(e) => {
                        let mut ctx = Ctx::plain(&joined);
                        let pred = self.bind(e, &mut ctx)?;
                        split_join_predicate(pred, ln)
                    }
                };

                Ok((
                    LogicalPlan::Join {
                        left: Box::new(lp),
                        right: Box::new(rp),
                        op: *op,
                        on,
                        residual,
                        schema,
                    },
                    joined,
                ))
            }
        }
    }

    // ---------------------------------------------------------- expressions

    /// Bind a select-list expression together with the alias it defines.
    ///
    /// `SELECT lower(url) AS url` gives an alias the same name as a real
    /// column. Elsewhere in the block (GROUP BY / HAVING / ORDER BY) that name
    /// means the select-list expression -- but *inside the defining expression
    /// itself* it has to mean the table column, otherwise the body substitutes
    /// into itself and `lower` is applied twice.
    ///
    /// `Ctx::expanding` already encodes exactly that rule; it was simply never
    /// primed with the alias being defined, so the guard first fired one level
    /// too late. Pushing the name for the duration of the bind makes the
    /// defining item behave like any other level of the expansion.
    fn bind_defining(
        &self,
        expr: &Expr,
        alias: Option<&str>,
        ctx: &mut Ctx<'_>,
    ) -> Result<BoundExpr> {
        let Some(a) = alias else {
            return self.bind(expr, ctx);
        };
        if ctx.expanding.iter().any(|x| x.eq_ignore_ascii_case(a)) {
            return self.bind(expr, ctx);
        }
        ctx.expanding.push(a.to_string());
        let r = self.bind(expr, ctx);
        ctx.expanding.pop();
        r
    }

    fn bind(&self, e: &Expr, ctx: &mut Ctx<'_>) -> Result<BoundExpr> {
        // One guard for the whole expression grammar: `bind_binary`,
        // `bind_function` and the desugarings (`BETWEEN`, `CASE x WHEN`,
        // `INTERVAL`, alias expansion) all come back through here, so no
        // expression path can recurse without passing this line.
        let _guard = DepthGuard::enter(&self.depth, "expression")?;
        match e {
            Expr::Literal(v) => Ok(BoundExpr::lit(v.clone())),

            Expr::Column(name) => {
                if name.qualifier().is_none() {
                    let bare = name.last();
                    let aliases = ctx.aliases;
                    let active = ctx.expanding.iter().any(|a| a.eq_ignore_ascii_case(bare));
                    if !active {
                        if let Some(sub) = aliases
                            .iter()
                            .find(|(a, _)| a.eq_ignore_ascii_case(bare))
                            .map(|(_, x)| x.clone())
                        {
                            if ctx.expanding.len() >= MAX_ALIAS_DEPTH {
                                return Err(Error::bind(format!(
                                    "alias `{bare}` expands into itself"
                                )));
                            }
                            ctx.expanding.push(bare.to_string());
                            let r = self.bind(&sub, ctx);
                            ctx.expanding.pop();
                            return r;
                        }
                    }
                }
                ctx.scope.resolve(name)
            }

            Expr::Wildcard => Err(Error::bind(
                "`*` is only valid as a select item or as the argument of count()",
            )),

            Expr::UnaryOp { op, expr } => {
                let inner = self.bind(expr, ctx)?;
                let name = match op {
                    UnaryOp::Neg => "negate",
                    UnaryOp::Not => "not",
                };
                let ty = ret_of(name, &[inner.ty()])?;
                Ok(BoundExpr::Unary { op: *op, expr: Box::new(inner), ty })
            }

            Expr::BinaryOp { left, op, right } => self.bind_binary(left, *op, right, ctx),

            Expr::Cast { expr, ty } => {
                let inner = self.bind(expr, ctx)?;
                // A cast converts values, it cannot invent one for a NULL, so
                // `CAST(nullable AS Int64)` is `Nullable(Int64)`. `eval_cast`
                // already widens the same way at run time; declaring the bare
                // written type here made the plan disagree with its own output.
                let ty = if inner.ty().is_nullable() { ty.to_nullable() } else { ty.clone() };
                Ok(BoundExpr::Cast { expr: Box::new(inner), ty })
            }

            // `CASE x WHEN a THEN ...` is exactly `CASE WHEN x = a THEN ...`,
            // and desugaring here means the evaluator has one form to run.
            Expr::Case { operand: Some(op), when_then, else_result } => {
                let de = Expr::Case {
                    operand: None,
                    when_then: when_then
                        .iter()
                        .map(|(w, t)| {
                            (
                                Expr::binary((**op).clone(), BinaryOp::Eq, w.clone()),
                                t.clone(),
                            )
                        })
                        .collect(),
                    else_result: else_result.clone(),
                };
                self.bind(&de, ctx)
            }

            Expr::Case { operand: None, when_then, else_result } => {
                let mut wt = Vec::with_capacity(when_then.len());
                let mut ty: Option<DataType> = None;
                for (w, t) in when_then {
                    let bw = self.bind(w, ctx)?;
                    let bt = self.bind(t, ctx)?;
                    ty = Some(match ty {
                        None => bt.ty(),
                        Some(prev) => DataType::promote(&prev, &bt.ty())?,
                    });
                    wt.push((bw, bt));
                }
                let else_result = match else_result {
                    Some(e) => {
                        let b = self.bind(e, ctx)?;
                        ty = Some(match ty {
                            None => b.ty(),
                            Some(prev) => DataType::promote(&prev, &b.ty())?,
                        });
                        Some(Box::new(b))
                    }
                    // No ELSE means an unmatched row yields NULL.
                    None => {
                        ty = ty.map(|t| t.to_nullable());
                        None
                    }
                };
                let ty = ty.ok_or_else(|| Error::bind("CASE needs at least one WHEN"))?;
                Ok(BoundExpr::Case { when_then: wt, else_result, ty })
            }

            Expr::InList { expr, list, negated } => {
                let b = self.bind(expr, ctx)?;
                let target = b.ty();
                let mut vals = Vec::with_capacity(list.len());
                for item in list {
                    let bi = self.bind(item, ctx)?;
                    let v = bi.as_literal().cloned().ok_or_else(|| {
                        Error::bind(format!(
                            "IN list entries must be literals, got `{item}`; a \
                             non-constant list is only expressible as `IN (SELECT ...)`"
                        ))
                    })?;
                    vals.push(coerce_literal(v, &target)?);
                }
                Ok(BoundExpr::InList { expr: Box::new(b), list: vals, negated: *negated })
            }

            // Reachable only *outside* a WHERE conjunct -- inside an OR, a
            // CASE, a select item. There the test has to yield a value per row,
            // and a semi-join yields rows; the mark join that would is not
            // built. `split_membership` takes every position that is a join.
            Expr::InSubquery { .. } => Err(Error::unsupported(
                "`x IN (SELECT ...)` is a semi-join and only binds as a whole WHERE \
                 conjunct; here it would have to be a per-row value",
            )),

            // (x >= lo AND x <= hi), or its negation.
            Expr::Between { expr, low, high, negated } => {
                let ge = Expr::binary((**expr).clone(), BinaryOp::GtEq, (**low).clone());
                let le = Expr::binary((**expr).clone(), BinaryOp::LtEq, (**high).clone());
                let both = Expr::binary(ge, BinaryOp::And, le);
                let de = if *negated {
                    Expr::UnaryOp { op: UnaryOp::Not, expr: Box::new(both) }
                } else {
                    both
                };
                self.bind(&de, ctx)
            }

            Expr::Like { expr, pattern, negated, case_insensitive } => {
                let b = self.bind(expr, ctx)?;
                if !b.ty().is_string() {
                    return Err(Error::bind(format!(
                        "LIKE needs a string on the left, got {}",
                        b.ty()
                    )));
                }
                let bp = self.bind(pattern, ctx)?;
                let pat = bp
                    .as_literal()
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
                    .ok_or_else(|| {
                        Error::bind(format!("LIKE pattern must be a string literal, got `{pattern}`"))
                    })?;
                Ok(BoundExpr::Like {
                    expr: Box::new(b),
                    pattern: pat,
                    negated: *negated,
                    case_insensitive: *case_insensitive,
                })
            }

            Expr::IsNull { expr, negated } => {
                let b = self.bind(expr, ctx)?;
                Ok(BoundExpr::IsNull { expr: Box::new(b), negated: *negated })
            }

            Expr::Tuple(_) => Err(Error::unsupported(
                "tuple expressions outside of IN / LIMIT BY / ORDER BY",
            )),

            Expr::Subquery(_) => Err(Error::unsupported(
                "scalar subqueries: the binder has no executor to evaluate one with",
            )),

            Expr::Exists { .. } => Err(Error::unsupported(
                "EXISTS (SELECT ...) is a semi-join and only binds as a whole WHERE \
                 conjunct; here it would have to be a per-row value",
            )),

            Expr::Interval { .. } => Err(Error::bind(
                "INTERVAL is only meaningful added to or subtracted from a date",
            )),

            Expr::Function { name, args, params, distinct } => {
                self.bind_function(name, args, params, *distinct, e, ctx)
            }

            Expr::Window { name, args, params, distinct, spec } => {
                self.bind_window(name, args, params, *distinct, spec, e, ctx)
            }
        }
    }

    /// Hoist `f(args) OVER (spec)` into [`Ctx::windows`] and leave a marker.
    ///
    /// Same shape as [`Binder::bind_function`]'s aggregate half, and for the
    /// same reason: `BoundExpr` has no window variant, so the call survives
    /// binding as a `Column` index far above any real schema and
    /// [`rewrite_over_window`] resolves it once the operator's output width is
    /// known.
    ///
    /// Everything inside the call -- arguments, `PARTITION BY`, `ORDER BY` --
    /// binds against the *source* scope with aggregates still allowed, because
    /// `row_number() OVER (ORDER BY sum(v) DESC)` is legal and the `sum` has to
    /// hoist into the same `Aggregate` node the select list uses.
    #[allow(clippy::too_many_arguments)]
    fn bind_window(
        &self,
        name: &str,
        args: &[Expr],
        params: &[Expr],
        distinct: bool,
        spec: &WindowSpec,
        whole: &Expr,
        ctx: &mut Ctx<'_>,
    ) -> Result<BoundExpr> {
        if !ctx.allow_agg {
            return Err(Error::bind(format!(
                "window function `{name}` is not allowed here (it belongs in the SELECT list \
                 or ORDER BY; a window is computed after WHERE, GROUP BY and HAVING have run)"
            )));
        }
        if ctx.in_agg {
            return Err(Error::bind(format!(
                "a window function cannot appear inside an aggregate: `{name}` OVER (...)"
            )));
        }
        if ctx.in_window {
            return Err(Error::bind(format!(
                "window functions cannot be nested: `{name}` inside another `OVER` clause"
            )));
        }

        // `count(*) OVER ()`: the wildcard means "no argument", exactly as for
        // the aggregate spelling.
        let real: Vec<&Expr> = args.iter().filter(|a| !matches!(a, Expr::Wildcard)).collect();
        if real.len() != args.len() && !name.eq_ignore_ascii_case("count") {
            return Err(Error::bind(format!("`*` is not a valid argument to `{name}`")));
        }

        ctx.in_window = true;
        let bound = (|| -> Result<_> {
            let mut bound = Vec::with_capacity(real.len());
            for a in real {
                bound.push(self.bind(a, ctx)?);
            }
            let mut pvals = Vec::with_capacity(params.len());
            for p in params {
                let bp = self.bind(p, ctx)?;
                pvals.push(bp.as_literal().cloned().ok_or_else(|| {
                    Error::bind(format!("`{name}` parameters must be constants"))
                })?);
            }
            let mut partition = Vec::with_capacity(spec.partition_by.len());
            for p in &spec.partition_by {
                partition.push(self.bind(p, ctx)?);
            }
            let order = self.sort_keys_in(&spec.order_by, ctx, &[])?;
            Ok((bound, pvals, partition, order))
        })();
        ctx.in_window = false;
        let (bound, pvals, partition, order) = bound?;

        let func = window::plan_call(
            name,
            bound,
            pvals,
            distinct,
            spec.effective_frame(),
            whole.to_string(),
        )?;
        let ty = func.ty.clone();
        let display = func.name.clone();

        // The key has to cover the OVER clause as well as the call: `sum(v)
        // OVER (ORDER BY a)` and `sum(v) OVER (ORDER BY b)` are two different
        // columns that happen to render the same call.
        let key = window_key(&func, &partition, &order);
        let idx = match ctx.window_keys.iter().position(|k| *k == key) {
            Some(i) => i,
            None => {
                ctx.window_keys.push(key);
                ctx.windows.push(PendingWindow { func, partition, order });
                ctx.windows.len() - 1
            }
        };
        Ok(BoundExpr::Column { index: WIN_MARK + idx, ty, name: display })
    }

    /// Stack the `Window` operators a select block needs, and rewrite every
    /// marker to the column it landed in.
    ///
    /// Calls sharing an `OVER` clause share one operator (and therefore one
    /// sort); calls with different clauses get one each, stacked, because each
    /// needs its input ordered differently. `remap[j]` is where hoisted call
    /// `j` ended up, which is *not* its hoist order once the grouping has
    /// shuffled them.
    fn apply_windows(
        &self,
        mut plan: LogicalPlan,
        pending: Vec<PendingWindow>,
    ) -> Result<(LogicalPlan, Vec<usize>)> {
        let mut remap = vec![0usize; pending.len()];
        // Group by OVER clause, preserving first-appearance order so the plan
        // is stable and `EXPLAIN` reads in the order the query was written.
        let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
        for (j, p) in pending.iter().enumerate() {
            let sig = over_key(&p.partition, &p.order);
            match groups.iter_mut().find(|(s, _)| *s == sig) {
                Some((_, v)) => v.push(j),
                None => groups.push((sig, vec![j])),
            }
        }

        let mut pending: Vec<Option<PendingWindow>> = pending.into_iter().map(Some).collect();
        for (_, members) in groups {
            let width = plan.schema().len();
            // Every member of a group has the same OVER clause by construction,
            // so the first one's keys speak for all of them.
            let first = pending[members[0]].as_ref().expect("taken exactly once, below");
            let partition = first.partition.clone();
            let order = first.order.clone();

            // The sort a window needs: partition keys first (they only have to
            // group, so the direction is arbitrary and ascending is as good as
            // any), then the ORDER BY keys with the directions as written.
            let mut keys: Vec<SortKey> = partition
                .iter()
                .map(|e| SortKey { expr: e.clone(), asc: true, nulls_first: true })
                .collect();
            keys.extend(order.iter().cloned());
            if !keys.is_empty() {
                plan = LogicalPlan::Sort { input: Box::new(plan), keys };
            }

            let mut funcs = Vec::with_capacity(members.len());
            for (slot, j) in members.iter().enumerate() {
                remap[*j] = width + slot;
                funcs.push(pending[*j].take().expect("each index appears in one group").func);
            }
            let schema = window::output_schema(plan.schema(), &funcs);
            let node = window::WindowNode {
                funcs,
                partition,
                order: order.into_iter().map(|k| k.expr).collect(),
                schema,
            };
            plan = LogicalPlan::Window { input: Box::new(plan), node: Box::new(node) };
        }
        Ok((plan, remap))
    }

    fn bind_binary(
        &self,
        left: &Expr,
        op: BinaryOp,
        right: &Expr,
        ctx: &mut Ctx<'_>,
    ) -> Result<BoundExpr> {
        // `d + INTERVAL 1 DAY` is `addDays(d, 1)`. Rewriting here means the
        // evaluator never has to know what an interval is.
        if let Expr::Interval { value, unit } = right {
            if matches!(op, BinaryOp::Plus | BinaryOp::Minus) {
                let f = interval_fn(*unit, matches!(op, BinaryOp::Plus));
                let call = Expr::func(f, vec![left.clone(), (**value).clone()]);
                return self.bind(&call, ctx);
            }
        }
        if let Expr::Interval { value, unit } = left {
            if matches!(op, BinaryOp::Plus) {
                let f = interval_fn(*unit, true);
                let call = Expr::func(f, vec![right.clone(), (**value).clone()]);
                return self.bind(&call, ctx);
            }
        }

        let mut l = self.bind(left, ctx)?;
        let mut r = self.bind(right, ctx)?;

        if op.is_comparison() {
            // The single most valuable coercion in the engine: turning
            // `d = '2024-01-01'` into `d = Date(19723)` is what lets the
            // optimizer lift it into a zone filter and skip granules.
            let rt = r.ty();
            coerce_temporal(&mut l, &rt)?;
            let lt = l.ty();
            coerce_temporal(&mut r, &lt)?;
            let (lt, rt) = (l.ty(), r.ty());
            if lt.is_string() != rt.is_string() {
                return Err(Error::bind(format!(
                    "cannot compare {lt} with {rt}"
                )));
            }
            let ty = if lt.is_nullable() || rt.is_nullable() {
                DataType::Bool.to_nullable()
            } else {
                DataType::Bool
            };
            return Ok(BoundExpr::Binary { left: Box::new(l), op, right: Box::new(r), ty });
        }

        let fname = match op {
            BinaryOp::Plus => "plus",
            BinaryOp::Minus => "minus",
            BinaryOp::Multiply => "multiply",
            BinaryOp::Divide => "divide",
            BinaryOp::IntDiv => "intDiv",
            BinaryOp::Modulo => "modulo",
            BinaryOp::And => "and",
            BinaryOp::Or => "or",
            BinaryOp::Concat => "concat",
            _ => unreachable!("comparisons handled above"),
        };
        let ty = ret_of(fname, &[l.ty(), r.ty()])?;
        Ok(BoundExpr::Binary { left: Box::new(l), op, right: Box::new(r), ty })
    }

    fn bind_function(
        &self,
        name: &str,
        args: &[Expr],
        params: &[Expr],
        distinct: bool,
        whole: &Expr,
        ctx: &mut Ctx<'_>,
    ) -> Result<BoundExpr> {
        // Scalar first: the two registries share no names, and the aggregate
        // lookup has a `-If` suffix rule that should not get first refusal.
        if let Some(f) = scalar(name) {
            if !params.is_empty() {
                return Err(Error::bind(format!(
                    "`{name}` is a scalar function and takes no parameter list"
                )));
            }
            if distinct {
                return Err(Error::bind(format!("DISTINCT is only valid on aggregates, not `{name}`")));
            }
            f.check_arity(args.len())?;
            let mut bound = Vec::with_capacity(args.len());
            for a in args {
                bound.push(self.bind(a, ctx)?);
            }
            let tys: Vec<DataType> = bound.iter().map(|b| b.ty()).collect();
            let ty = (f.ret)(&tys).map_err(|e| annotate(e, f.name))?;
            return Ok(BoundExpr::Scalar { func: f, args: bound, ty });
        }

        let Some(f) = aggregate(name) else {
            return Err(Error::bind(format!("unknown function `{name}`")));
        };

        if !ctx.allow_agg {
            return Err(Error::bind(format!(
                "aggregate function `{name}` is not allowed here (it belongs in the \
                 SELECT list, HAVING or ORDER BY)"
            )));
        }
        if ctx.in_agg {
            return Err(Error::bind(format!(
                "aggregate functions cannot be nested: `{name}` inside another aggregate"
            )));
        }
        if distinct && !f.supports_distinct {
            return Err(Error::bind(format!("aggregate `{}` does not support DISTINCT", f.name)));
        }

        // `count(*)` counts rows, so the wildcard is simply no argument at all.
        let is_count = f.name.eq_ignore_ascii_case("count") || f.name.eq_ignore_ascii_case("countIf");
        let mut real_args: Vec<&Expr> = Vec::with_capacity(args.len());
        for a in args {
            if matches!(a, Expr::Wildcard) {
                if !is_count {
                    return Err(Error::bind(format!(
                        "`*` is not a valid argument to `{}`",
                        f.name
                    )));
                }
                continue;
            }
            real_args.push(a);
        }

        let mut pvals = Vec::with_capacity(params.len());
        for p in params {
            let bp = self.bind(p, ctx)?;
            pvals.push(
                bp.as_literal()
                    .cloned()
                    .ok_or_else(|| Error::bind(format!("`{}` parameters must be constants", f.name)))?,
            );
        }

        ctx.in_agg = true;
        let mut bound = Vec::with_capacity(real_args.len());
        for a in real_args {
            match self.bind(a, ctx) {
                Ok(b) => bound.push(b),
                Err(e) => {
                    ctx.in_agg = false;
                    return Err(e);
                }
            }
        }
        ctx.in_agg = false;

        f.check_arity(bound.len())?;
        let tys: Vec<DataType> = bound.iter().map(|b| b.ty()).collect();
        let ty = (f.ret)(&tys, &pvals).map_err(|e| annotate(e, f.name))?;
        let display = whole.to_string();
        let key = agg_key(f, &bound, &pvals, distinct);

        let idx = match ctx.agg_keys.iter().position(|k| *k == key) {
            Some(i) => i,
            None => {
                ctx.agg_keys.push(key);
                ctx.aggs.push(BoundAgg {
                    func: f,
                    args: bound,
                    params: pvals,
                    distinct,
                    ty: ty.clone(),
                    name: display.clone(),
                });
                ctx.aggs.len() - 1
            }
        };
        Ok(BoundExpr::Column { index: AGG_MARK + idx, ty, name: display })
    }
}

// ============================================================ post-aggregate

/// Rewrite an expression bound against the *source* scope so it reads the
/// Aggregate node's output instead.
///
/// Pre-order, because the largest matching subtree wins: with
/// `GROUP BY toStartOfDay(ts)`, the select item `toStartOfDay(ts)` must become
/// one column reference, not a call whose argument `ts` fails to resolve.
fn rewrite_over_agg(
    e: BoundExpr,
    group_keys: &[String],
    group_fields: &[Field],
    n_group: usize,
) -> Result<BoundExpr> {
    if let BoundExpr::Column { index, ty, name } = &e {
        // A window marker is not this pass's business: the operator it names
        // sits *above* the Aggregate and does not exist yet. Left alone, and
        // checked first because `WIN_MARK > AGG_MARK` would otherwise make it
        // look like an aggregate marker with an absurd offset.
        if *index >= WIN_MARK {
            return Ok(e);
        }
        if *index >= AGG_MARK {
            return Ok(BoundExpr::Column {
                index: n_group + (index - AGG_MARK),
                ty: ty.clone(),
                name: name.clone(),
            });
        }
    }
    let rendered = e.to_string();
    if let Some(i) = group_keys.iter().position(|k| *k == rendered) {
        return Ok(BoundExpr::Column {
            index: i,
            ty: group_fields[i].ty.clone(),
            name: group_fields[i].name.clone(),
        });
    }
    let rec = |x: Box<BoundExpr>| -> Result<Box<BoundExpr>> {
        Ok(Box::new(rewrite_over_agg(*x, group_keys, group_fields, n_group)?))
    };
    Ok(match e {
        BoundExpr::Column { name, .. } => {
            return Err(Error::bind(format!(
                "column `{name}` must appear in GROUP BY or be used in an aggregate function"
            )))
        }
        BoundExpr::Literal { value, ty } => BoundExpr::Literal { value, ty },
        BoundExpr::Unary { op, expr, ty } => BoundExpr::Unary { op, expr: rec(expr)?, ty },
        BoundExpr::Binary { left, op, right, ty } => BoundExpr::Binary {
            left: rec(left)?,
            op,
            right: rec(right)?,
            ty,
        },
        BoundExpr::Scalar { func, args, ty } => BoundExpr::Scalar {
            func,
            args: args
                .into_iter()
                .map(|a| rewrite_over_agg(a, group_keys, group_fields, n_group))
                .collect::<Result<Vec<_>>>()?,
            ty,
        },
        BoundExpr::Cast { expr, ty } => BoundExpr::Cast { expr: rec(expr)?, ty },
        BoundExpr::Case { when_then, else_result, ty } => BoundExpr::Case {
            when_then: when_then
                .into_iter()
                .map(|(w, t)| {
                    Ok((
                        rewrite_over_agg(w, group_keys, group_fields, n_group)?,
                        rewrite_over_agg(t, group_keys, group_fields, n_group)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?,
            else_result: match else_result {
                Some(e) => Some(rec(e)?),
                None => None,
            },
            ty,
        },
        BoundExpr::InList { expr, list, negated } => {
            BoundExpr::InList { expr: rec(expr)?, list, negated }
        }
        BoundExpr::Like { expr, pattern, negated, case_insensitive } => BoundExpr::Like {
            expr: rec(expr)?,
            pattern,
            negated,
            case_insensitive,
        },
        BoundExpr::IsNull { expr, negated } => BoundExpr::IsNull { expr: rec(expr)?, negated },
    })
}

/// [`rewrite_over_agg`] applied through a `&mut`, for the expressions that live
/// inside a pending window rather than in a list being rebuilt.
///
/// The hole left while the subtree is out is a fieldless literal, so it costs a
/// discriminant store and no allocation.
fn rewrite_in_place(
    e: &mut BoundExpr,
    group_keys: &[String],
    group_fields: &[Field],
    n_group: usize,
) -> Result<()> {
    let taken = std::mem::replace(e, BoundExpr::lit(Value::Null));
    *e = rewrite_over_agg(taken, group_keys, group_fields, n_group)?;
    Ok(())
}

// ============================================================= post-window

/// Point every hoisted-window marker at the column the operator actually put it
/// in. `remap[j]` is the output index of hoisted call `j`.
fn resolve_windows(e: &mut BoundExpr, remap: &[usize]) -> Result<()> {
    e.remap_columns(&|i| {
        if i >= WIN_MARK {
            remap.get(i - WIN_MARK).copied()
        } else {
            Some(i)
        }
    })
}

/// Does this bound tree still contain an unresolved window marker?
fn has_window_marker(e: &BoundExpr) -> bool {
    let mut found = false;
    e.visit(&mut |x| {
        if matches!(x, BoundExpr::Column { index, .. } if *index >= WIN_MARK) {
            found = true;
        }
    });
    found
}

/// Canonical spelling of an `OVER` clause, over its *bound* keys, so that
/// `PARTITION BY t.k` and `PARTITION BY k` share one operator and one sort.
fn over_key(partition: &[BoundExpr], order: &[SortKey]) -> String {
    let p: Vec<String> = partition.iter().map(|e| e.to_string()).collect();
    let o: Vec<String> = order
        .iter()
        .map(|k| {
            format!(
                "{}{}{}",
                k.expr,
                if k.asc { "+" } else { "-" },
                if k.nulls_first { "n" } else { "N" }
            )
        })
        .collect();
    format!("P[{}]O[{}]", p.join(","), o.join(","))
}

/// Canonical spelling of a whole window call. Two calls share a column only if
/// the function, its arguments, its folded count *and* its frame all match --
/// `sum(v) OVER (ORDER BY a)` and `sum(v) OVER (ORDER BY b)` are two columns
/// that happen to render the same call.
fn window_key(f: &BoundWindow, partition: &[BoundExpr], order: &[SortKey]) -> String {
    let a: Vec<String> = f.args.iter().map(|x| x.to_string()).collect();
    let p: Vec<String> = f.params.iter().map(|x| x.to_string()).collect();
    format!(
        "{}[{}]({})#{}{:?}{}",
        f.kind.name(),
        p.join(","),
        a.join(","),
        f.offset,
        f.frame,
        over_key(partition, order)
    )
}

// ================================================================== helpers

/// A canonical spelling of an aggregate call, used only for de-duplication.
/// It runs on the *bound* arguments, so `sum(v)` and `sum(t.v)` collapse.
fn agg_key(f: &AggFn, args: &[BoundExpr], params: &[Value], distinct: bool) -> String {
    let a: Vec<String> = args.iter().map(|x| x.to_string()).collect();
    let p: Vec<String> = params.iter().map(|x| x.to_string()).collect();
    format!(
        "{}[{}]({}{})",
        f.name,
        p.join(","),
        if distinct { "DISTINCT " } else { "" },
        a.join(",")
    )
}

/// Type of a scalar call, via the registry's own `ret` callback.
fn ret_of(name: &str, args: &[DataType]) -> Result<DataType> {
    let f = scalar(name)
        .ok_or_else(|| Error::bind(format!("internal: operator `{name}` is not registered")))?;
    f.check_arity(args.len())?;
    (f.ret)(args).map_err(|e| annotate(e, f.name))
}

/// Several registry entries share one `ret` callback and so report a generic
/// name ("string function: argument 1 must be a String"). Prefix the call the
/// user actually wrote, because an error that does not name the offending
/// identifier is an error nobody can act on.
fn annotate(e: Error, name: &str) -> Error {
    match e {
        Error::Bind(m) if !m.to_ascii_lowercase().contains(&name.to_ascii_lowercase()) => {
            Error::bind(format!("{name}: {m}"))
        }
        other => other,
    }
}

fn interval_fn(unit: IntervalUnit, add: bool) -> &'static str {
    match (unit, add) {
        (IntervalUnit::Second, true) => "addSeconds",
        (IntervalUnit::Minute, true) => "addMinutes",
        (IntervalUnit::Hour, true) => "addHours",
        (IntervalUnit::Day, true) => "addDays",
        (IntervalUnit::Week, true) => "addWeeks",
        (IntervalUnit::Month, true) => "addMonths",
        (IntervalUnit::Quarter, true) => "addQuarters",
        (IntervalUnit::Year, true) => "addYears",
        (IntervalUnit::Second, false) => "subtractSeconds",
        (IntervalUnit::Minute, false) => "subtractMinutes",
        (IntervalUnit::Hour, false) => "subtractHours",
        (IntervalUnit::Day, false) => "subtractDays",
        (IntervalUnit::Week, false) => "subtractWeeks",
        (IntervalUnit::Month, false) => "subtractMonths",
        (IntervalUnit::Quarter, false) => "subtractQuarters",
        (IntervalUnit::Year, false) => "subtractYears",
    }
}

/// If `target` is temporal and `lit` is a string literal, reinterpret the
/// literal as a date/datetime so the comparison is on lanes, not text.
fn coerce_temporal(lit: &mut BoundExpr, target: &DataType) -> Result<()> {
    if !target.is_temporal() {
        return Ok(());
    }
    let Some(s) = lit.as_literal().and_then(|v| v.as_str()).map(|s| s.to_string()) else {
        return Ok(());
    };
    let (value, ty) = match target.base() {
        DataType::Date => (
            Value::Date(parse_date(&s).map_err(|e| Error::bind(e.to_string()))?),
            DataType::Date,
        ),
        _ => (
            Value::DateTime(parse_datetime(&s).map_err(|e| Error::bind(e.to_string()))?),
            DataType::DateTime,
        ),
    };
    *lit = BoundExpr::Literal { value, ty };
    Ok(())
}

/// Same coercion for `IN` list entries, plus the string-vs-number rejection
/// that `coerce_temporal` gets from the comparison path.
fn coerce_literal(v: Value, target: &DataType) -> Result<Value> {
    if v.is_null() {
        return Ok(Value::Null);
    }
    if target.is_temporal() {
        if let Some(s) = v.as_str() {
            return Ok(match target.base() {
                DataType::Date => Value::Date(parse_date(s).map_err(|e| Error::bind(e.to_string()))?),
                _ => Value::DateTime(parse_datetime(s).map_err(|e| Error::bind(e.to_string()))?),
            });
        }
        return Ok(v);
    }
    if target.is_string() != matches!(v, Value::Str(_)) {
        return Err(Error::bind(format!("cannot compare {target} with {v}")));
    }
    Ok(v)
}

/// Merge the two copies of a `USING` key into the single column standard SQL
/// says the join produces.
///
/// `SELECT * FROM a JOIN b USING (id)` has to yield `id, x, y`, not `id, x,
/// id, y`, and a bare `id` has to resolve rather than report itself ambiguous.
/// Both fall out of one edit to the scope: the per-side entries are marked
/// [`ScopeCol::shadowed`] (still reachable as `a.id` / `b.id`, because for an
/// outer join those genuinely differ from the merged value) and a third entry
/// is appended that both supersedes them and says which side to read.
///
/// Which side that is, is the whole subtlety. The merged column is
/// `COALESCE(left, right)` in general, but a padded NULL can only ever appear
/// on the side the join type makes optional, so:
///
///   * INNER / CROSS -- the two are equal by construction, take the left;
///   * LEFT  -- the right copy is the padded one, take the left;
///   * RIGHT -- the *left* copy is the padded one, so an unmatched right row
///     must show the right key, not NULL. Taking the left here is the bug this
///     exists to prevent;
///   * FULL  -- either side may be padded, and only here is a real `coalesce`
///     call needed. The other three cost exactly one column reference.
///
/// The entry keeps the left copy's *ordering* index whatever the join type, so
/// `*` emits the key in the position the left table gave it.
///
/// `lpos`/`rpos` are positions in `joined.cols`; `lval`/`rval` are the block
/// columns those two entries actually read. The pairs differ whenever a side
/// is itself the merged key of an enclosed join -- `a RIGHT JOIN b USING (k)
/// JOIN c USING (k)` -- and using the entry's own `index` there would silently
/// re-introduce the padded side that the inner merge just excluded.
fn merge_using_key(
    joined: &mut Scope,
    (lpos, lval): (usize, usize),
    (rpos, rval): (usize, usize),
    op: JoinOp,
) -> Result<()> {
    let (l, r) = (joined.cols[lpos].clone(), joined.cols[rpos].clone());
    let (src, ty) = match op {
        JoinOp::Right => (Src::At(rval), r.ty.clone()),
        // Routed through the registry's own `ret` so the declared type is the
        // one `coalesce` will actually produce -- both sides are Nullable here
        // (an outer join widened them), so this is Nullable too, even though no
        // output row can really have the key missing on both sides.
        JoinOp::Full => {
            (Src::Coalesce(lval, rval), ret_of("coalesce", &[l.ty.clone(), r.ty.clone()])?)
        }
        _ => (Src::At(lval), l.ty.clone()),
    };
    joined.cols[lpos].shadowed = true;
    joined.cols[rpos].shadowed = true;
    joined.cols.push(ScopeCol {
        qualifier: None,
        name: l.name.clone(),
        ty,
        index: l.index,
        src,
        shadowed: false,
    });
    Ok(())
}

fn nullable_scope(s: &Scope) -> Scope {
    Scope {
        cols: s
            .cols
            .iter()
            .map(|c| ScopeCol { ty: c.ty.to_nullable(), ..c.clone() })
            .collect(),
        width: s.width,
    }
}

fn scope_schema(s: &Scope) -> Schema {
    Schema::new_unchecked(
        s.visible().into_iter().map(|(_, c)| Field::new(c.name.clone(), c.ty.clone())).collect(),
    )
}

/// Split a `WHERE` clause's AND-spine into ordinary conjuncts and membership
/// tests, in written order.
///
/// Only the *top* of the spine qualifies. `a AND x IN (S)` is two nodes and one
/// of them is a join; `a OR x IN (S)` is one predicate whose value must exist
/// per row, which a semi-join cannot produce -- for those [`Binder::bind`]
/// still refuses the expression, and the session layer's fold is what answers
/// them. Reading the spine rather than recursing into it is what keeps that
/// boundary a property of the shape instead of a guess.
///
/// Iterative for the same reason [`Expr::visit`] is: `a AND b AND c AND ...` is
/// a loop in the parser and a left-deep tree here, so its depth is the user's
/// to choose.
fn split_membership(e: &Expr) -> (Vec<&Expr>, Vec<&Expr>) {
    let (mut rest, mut subs) = (Vec::new(), Vec::new());
    let mut todo = vec![e];
    while let Some(e) = todo.pop() {
        match e {
            Expr::BinaryOp { left, op: BinaryOp::And, right } => {
                todo.push(right);
                todo.push(left);
            }
            Expr::InSubquery { .. } | Expr::Exists { .. } => subs.push(e),
            other => rest.push(other),
        }
    }
    (rest, subs)
}

/// Peel equi-join pairs out of an ON predicate. `on` indices are per-side
/// (`(left index, right index)`); whatever is left over stays expressed
/// against the concatenated schema, which is what the executor evaluates it
/// over once both halves of a row are in hand.
fn split_join_predicate(
    pred: BoundExpr,
    left_len: usize,
) -> (Vec<(usize, usize)>, Option<BoundExpr>) {
    let mut on = Vec::new();
    let mut residual = Vec::new();
    for c in pred.split_conjuncts() {
        if let BoundExpr::Binary { left, op: BinaryOp::Eq, right, .. } = &c {
            if let (Some(a), Some(b)) = (left.as_column(), right.as_column()) {
                if a < left_len && b >= left_len {
                    on.push((a, b - left_len));
                    continue;
                }
                if b < left_len && a >= left_len {
                    on.push((b, a - left_len));
                    continue;
                }
            }
        }
        residual.push(c);
    }
    (on, BoundExpr::join_conjuncts(residual))
}

fn flatten_union(plan: LogicalPlan, all: bool, out: &mut Vec<LogicalPlan>) {
    match plan {
        LogicalPlan::Union { inputs, all: a, .. } if a == all => out.extend(inputs),
        other => out.push(other),
    }
}

fn union_schema(inputs: &[LogicalPlan]) -> Result<Schema> {
    let first = inputs[0].schema();
    let width = first.len();
    let mut tys: Vec<DataType> = first.fields().iter().map(|f| f.ty.clone()).collect();
    for (i, p) in inputs.iter().enumerate().skip(1) {
        let s = p.schema();
        if s.len() != width {
            return Err(Error::bind(format!(
                "UNION branches disagree on width: branch 1 has {width} columns, \
                 branch {} has {}",
                i + 1,
                s.len()
            )));
        }
        for (c, f) in s.fields().iter().enumerate() {
            tys[c] = DataType::promote(&tys[c], &f.ty).map_err(|_| {
                Error::bind(format!(
                    "UNION branches disagree on column {}: {} vs {}",
                    c + 1,
                    tys[c],
                    f.ty
                ))
            })?;
        }
    }
    Ok(Schema::new_unchecked(
        first
            .fields()
            .iter()
            .zip(tys)
            .map(|(f, t)| Field::new(f.name.clone(), t))
            .collect(),
    ))
}

/// A select-list position: the `2` of `ORDER BY 2` / `GROUP BY 2`.
///
/// Only a whole non-negative number is a position. `ORDER BY 1.5` used to go
/// through `Value::as_u64`, which truncates, so it silently became `ORDER BY 1`
/// -- sorting by a column the user never named. A number that is not a position
/// and not a column is a mistake, and is reported as one.
fn ordinal(e: &Expr, what: &str) -> Result<Option<usize>> {
    let Expr::Literal(v) = e else {
        return Ok(None);
    };
    match v {
        Value::UInt(n) => Ok(Some(*n as usize)),
        Value::Int(n) if *n >= 0 => Ok(Some(*n as usize)),
        Value::Float(_) | Value::Int(_) => Err(Error::bind(format!(
            "{what} position must be a whole positive number, got {v}"
        ))),
        _ => Ok(None),
    }
}

/// `GROUP BY 2` -- the second select item, with the alias it defines so the
/// caller can bind it exactly as the projection did. Refused when the
/// projection has a wildcard in it, because then "the second item" depends on
/// a schema the binder has not finished resolving.
fn select_item_at(sel: &Select, n: usize) -> Result<(&Expr, Option<&str>)> {
    if sel.projection.iter().any(|p| !matches!(p, SelectItem::Expr { .. })) {
        return Err(Error::bind(
            "GROUP BY by position is not allowed when the SELECT list contains `*`",
        ));
    }
    match sel.projection.get(n.wrapping_sub(1)) {
        Some(SelectItem::Expr { expr, alias }) if n > 0 => Ok((expr, alias.as_deref())),
        _ => Err(Error::bind(format!(
            "GROUP BY position {n} is out of range (1..{})",
            sel.projection.len()
        ))),
    }
}

// ------------------------------------------------------------------ demand

/// Which of a table's columns this query block could possibly touch.
///
/// The scan is built before the projection is bound, so this is a purely
/// syntactic over-approximation: every identifier mentioned anywhere in the
/// block, matched by bare name. Over-approximating is safe -- the optimizer's
/// projection-pruning pass narrows it further once real indices exist -- and
/// it already removes the common case of a wide table with three columns
/// actually read.
struct Demand {
    all: bool,
    names: Vec<String>,
}

impl Demand {
    fn of(sel: &Select, q: &Query) -> Demand {
        let mut d = Demand { all: false, names: Vec::new() };
        for item in &sel.projection {
            match item {
                // `*` and `t.*` need the whole table; resolving which table a
                // qualified star means is the scope's job, not ours, so both
                // spellings fall back to "everything".
                SelectItem::Wildcard | SelectItem::QualifiedWildcard(_) => d.all = true,
                SelectItem::Expr { expr, .. } => d.walk(expr),
            }
        }
        if let Some(tr) = &sel.from {
            d.walk_from(tr);
        }
        for e in sel.prewhere.iter().chain(sel.selection.iter()).chain(sel.having.iter()) {
            d.walk(e);
        }
        for e in &sel.group_by {
            d.walk(e);
        }
        for ob in &q.order_by {
            d.walk(&ob.expr);
        }
        if let Some((n, by)) = &q.limit_by {
            d.walk(n);
            for e in by {
                d.walk(e);
            }
        }
        d
    }

    fn walk(&mut self, e: &Expr) {
        e.visit(&mut |x| {
            if let Expr::Column(n) = x {
                let lower = n.last().to_ascii_lowercase();
                if !self.names.contains(&lower) {
                    self.names.push(lower);
                }
            }
        });
    }

    fn walk_from(&mut self, tr: &TableRef) {
        if let TableRef::Join { left, right, constraint, .. } = tr {
            self.walk_from(left);
            self.walk_from(right);
            match constraint {
                JoinConstraint::On(e) => self.walk(e),
                JoinConstraint::Using(names) => {
                    for n in names {
                        let lower = n.to_ascii_lowercase();
                        if !self.names.contains(&lower) {
                            self.names.push(lower);
                        }
                    }
                }
                JoinConstraint::None => {}
            }
        }
        // A subquery brings its own Select and computes its own demand.
    }

    fn project(&self, schema: &Schema) -> Vec<usize> {
        if schema.is_empty() {
            return Vec::new();
        }
        if self.all {
            return (0..schema.len()).collect();
        }
        let cols: Vec<usize> = schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, f)| self.names.contains(&f.name.to_ascii_lowercase()))
            .map(|(i, _)| i)
            .collect();
        // `SELECT count(*) FROM t` reads no column by name. Keep one anyway:
        // a zero-column block has no natural row count, and one narrow column
        // is the cheapest possible way to count rows.
        if cols.is_empty() {
            vec![0]
        } else {
            cols
        }
    }
}

// ==================================================================== tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::ast::Statement;
    use crate::types::{Engine, TableDef};

    // ------------------------------------------------------------ fixtures

    fn table(name: &str, cols: &[(&str, &str)], order: &[usize]) -> TableDef {
        TableDef {
            name: name.into(),
            schema: Schema::new(
                cols.iter()
                    .map(|(n, t)| Field::new(*n, DataType::parse(t).unwrap()))
                    .collect(),
            )
            .unwrap(),
            order_by: order.to_vec(),
            primary_key: order.to_vec(),
            partition_by: None,
            engine: Engine::MergeTree,
        }
    }

    fn catalog() -> Catalog {
        let mut c = Catalog::in_memory();
        c.create_table(
            table(
                "events",
                &[
                    ("id", "UInt64"),
                    ("user_id", "UInt64"),
                    ("url", "String"),
                    ("ms", "UInt32"),
                    ("amount", "Float64"),
                    ("d", "Date"),
                    ("ts", "DateTime"),
                ],
                &[0],
            ),
            false,
        )
        .unwrap();
        c.create_table(
            table(
                "users",
                &[("user_id", "UInt64"), ("name", "String"), ("age", "UInt8")],
                &[0],
            ),
            false,
        )
        .unwrap();
        // The only nullable columns in the fixture, and the reason they exist:
        // `NOT IN` grows a NULL census exactly when one side can hold a NULL,
        // so a catalog without one cannot tell the two plans apart.
        c.create_table(
            table(
                "notes",
                &[("id", "UInt64"), ("n", "Nullable(Int64)"), ("tag", "Nullable(String)")],
                &[0],
            ),
            false,
        )
        .unwrap();
        c
    }

    fn plan_of(sql: &str) -> LogicalPlan {
        let c = catalog();
        let mut b = Binder::new(&c);
        let st = crate::sql::parse(sql).unwrap_or_else(|e| panic!("parse {sql}: {e}"));
        match &st[0] {
            Statement::Query(q) => {
                b.bind_query(q).unwrap_or_else(|e| panic!("bind {sql}: {e}"))
            }
            other => panic!("not a query: {other:?}"),
        }
    }

    fn explain(sql: &str) -> String {
        plan_of(sql).explain()
    }

    fn err(sql: &str) -> String {
        let c = catalog();
        let mut b = Binder::new(&c);
        let st = crate::sql::parse(sql).unwrap_or_else(|e| panic!("parse {sql}: {e}"));
        match &st[0] {
            Statement::Query(q) => match b.bind_query(q) {
                Ok(p) => panic!("expected an error for `{sql}`, got:\n{}", p.explain()),
                Err(e) => e.to_string(),
            },
            other => panic!("not a query: {other:?}"),
        }
    }

    /// The schema a query produces, rendered as `name: Type` pairs.
    fn out(sql: &str) -> Vec<String> {
        plan_of(sql)
            .schema()
            .fields()
            .iter()
            .map(|f| format!("{}: {}", f.name, f.ty))
            .collect()
    }

    // ------------------------------------------------------- scans & scope

    #[test]
    fn star_projects_every_column_in_declaration_order() {
        let e = explain("SELECT * FROM events");
        assert!(
            e.contains("Scan default.events [id, user_id, url, ms, amount, d, ts]"),
            "{e}"
        );
        assert_eq!(out("SELECT * FROM events").len(), 7);
    }

    #[test]
    fn scan_projection_is_narrowed_to_referenced_columns() {
        let e = explain("SELECT url FROM events WHERE ms > 10");
        assert!(e.contains("Scan default.events [url, ms]"), "{e}");
        // and the downstream indices are into the *projected* schema
        assert!(e.contains("Filter (ms#1 > 10)"), "{e}");
        assert!(e.contains("Project [url#0 AS url]"), "{e}");
    }

    #[test]
    fn count_star_still_reads_one_column_for_the_row_count() {
        let e = explain("SELECT count(*) FROM events");
        assert!(e.contains("Scan default.events [id]"), "{e}");
        assert!(e.contains("Aggregate group=[] aggs=[count()]"), "{e}");
    }

    #[test]
    fn qualified_and_aliased_column_lookup() {
        let e = explain("SELECT e.url FROM events AS e WHERE e.ms > 1");
        assert!(e.contains("Project [url#0 AS url]"), "{e}");
        // the original table name is replaced by the alias
        assert!(err("SELECT events.url FROM events AS e").contains("unknown column"));
    }

    #[test]
    fn unknown_column_lists_what_is_available() {
        let m = err("SELECT nope FROM users");
        assert!(m.contains("unknown column `nope`"), "{m}");
        assert!(m.contains("available: users.user_id, users.name, users.age"), "{m}");
    }

    #[test]
    fn unknown_table_is_reported_by_name() {
        assert!(err("SELECT * FROM ghosts").contains("ghosts"));
    }

    #[test]
    fn case_insensitive_column_match_is_a_fallback() {
        assert!(explain("SELECT URL FROM events").contains("url#0"));
    }

    #[test]
    fn qualified_wildcard_expands_one_table() {
        let cols = out("SELECT u.* FROM events AS e JOIN users AS u ON e.user_id = u.user_id");
        assert_eq!(cols, vec!["user_id: UInt64", "name: String", "age: UInt8"]);
        assert!(err("SELECT z.* FROM events").contains("unknown table `z`"));
    }

    // -------------------------------------------------------- where/prewhere

    #[test]
    fn prewhere_lands_inside_the_scan_and_where_does_not() {
        let e = explain("SELECT url FROM events PREWHERE ms > 5 WHERE id < 100");
        assert!(e.contains("Scan default.events [id, url, ms] prewhere=(ms#2 > 5)"), "{e}");
        assert!(e.contains("Filter (id#0 < 100)"), "{e}");
    }

    #[test]
    fn where_conjuncts_stay_one_filter_node() {
        let e = explain("SELECT id FROM events WHERE ms > 5 AND ms < 9");
        assert!(e.contains("Filter ((ms#1 > 5) AND (ms#1 < 9))"), "{e}");
    }

    // --------------------------------------------------------------- types

    #[test]
    fn arithmetic_promotes_through_the_registry() {
        // an integer literal is signed, so UInt32 + Int64 widens to Int64
        assert_eq!(out("SELECT ms + 1 FROM events"), vec!["ms + 1: Int64"]);
        assert_eq!(out("SELECT ms + id FROM events"), vec!["ms + id: UInt64"]);
        assert_eq!(out("SELECT ms + amount FROM events"), vec!["ms + amount: Float64"]);
        // `/` is always Nullable(Float64): division by zero yields NULL
        assert_eq!(out("SELECT ms / 2 FROM events"), vec!["ms / 2: Nullable(Float64)"]);
        assert_eq!(out("SELECT -ms FROM events"), vec!["-ms: Int64"]);
    }

    #[test]
    fn comparisons_yield_bool() {
        assert_eq!(out("SELECT ms > 1 FROM events"), vec!["ms > 1: Bool"]);
        assert_eq!(out("SELECT url LIKE 'a%' FROM events"), vec!["url LIKE 'a%': Bool"]);
    }

    #[test]
    fn string_column_against_a_number_is_rejected() {
        let m = err("SELECT id FROM events WHERE url = 42");
        assert!(m.contains("cannot compare"), "{m}");
        assert!(m.contains("String"), "{m}");
    }

    #[test]
    fn date_literal_is_reparsed_so_zone_maps_can_use_it() {
        let e = explain("SELECT id FROM events WHERE d = '2024-01-01'");
        // 2024-01-01 is day 19723 since the epoch; rendering goes through
        // Value::Date, so the literal is a Date and not a string any more.
        assert!(e.contains("(d#1 = 2024-01-01)"), "{e}");
        let e = explain("SELECT id FROM events WHERE ts > '2024-01-15 13:45:30'");
        assert!(e.contains("(ts#1 > 2024-01-15 13:45:30)"), "{e}");
    }

    #[test]
    fn unparseable_date_literal_is_a_bind_error() {
        assert!(err("SELECT id FROM events WHERE d = 'yesterday'").contains("yesterday"));
    }

    #[test]
    fn cast_carries_the_target_type() {
        assert_eq!(out("SELECT CAST(ms AS Int64) FROM events"), vec!["CAST(ms AS Int64): Int64"]);
    }

    // ------------------------------------------------------------ desugaring

    #[test]
    fn between_desugars_to_two_comparisons() {
        let e = explain("SELECT id FROM events WHERE ms BETWEEN 1 AND 9");
        assert!(e.contains("Filter ((ms#1 >= 1) AND (ms#1 <= 9))"), "{e}");
    }

    #[test]
    fn not_between_is_the_negation() {
        let e = explain("SELECT id FROM events WHERE ms NOT BETWEEN 1 AND 9");
        assert!(e.contains("NOT (((ms#1 >= 1) AND (ms#1 <= 9)))"), "{e}");
    }

    #[test]
    fn case_with_an_operand_desugars_to_searched_case() {
        let e = explain("SELECT CASE ms WHEN 1 THEN 'a' ELSE 'b' END FROM events");
        assert!(e.contains("CASE WHEN (ms#0 = 1) THEN 'a' ELSE 'b' END"), "{e}");
    }

    #[test]
    fn case_without_else_is_nullable() {
        assert_eq!(
            out("SELECT CASE WHEN ms > 1 THEN 2 END AS c FROM events"),
            vec!["c: Nullable(Int64)"]
        );
    }

    #[test]
    fn in_list_becomes_a_literal_set() {
        let e = explain("SELECT id FROM events WHERE ms IN (1, 2, 3)");
        assert!(e.contains("ms#1 IN (1, 2, 3)"), "{e}");
        let e = explain("SELECT id FROM events WHERE d IN ('2024-01-01')");
        assert!(e.contains("d#1 IN (2024-01-01)"), "{e}");
    }

    #[test]
    fn in_list_rejects_non_literals() {
        assert!(err("SELECT id FROM events WHERE ms IN (id)").contains("must be literals"));
        assert!(err("SELECT id FROM events WHERE url IN (1)").contains("cannot compare"));
    }

    #[test]
    fn like_requires_a_literal_pattern() {
        assert!(err("SELECT id FROM events WHERE url LIKE url").contains("string literal"));
        assert!(err("SELECT id FROM events WHERE ms LIKE 'x'").contains("LIKE needs a string"));
    }

    #[test]
    fn interval_arithmetic_becomes_a_function_call() {
        let e = explain("SELECT d + INTERVAL 7 DAY FROM events");
        assert!(e.contains("addDays(d#0, 7)"), "{e}");
        let e = explain("SELECT ts - INTERVAL 1 HOUR FROM events");
        assert!(e.contains("subtractHours(ts#0, 1)"), "{e}");
    }

    // ------------------------------------------------------------ functions

    #[test]
    fn scalar_functions_resolve_and_type_check() {
        assert_eq!(out("SELECT lower(url) FROM events"), vec!["lower(url): String"]);
        assert_eq!(out("SELECT toYear(d) FROM events"), vec!["toYear(d): UInt16"]);
        assert!(err("SELECT lower(url, 1) FROM events").contains("takes exactly 1"));
        assert!(err("SELECT nosuchfn(url) FROM events").contains("unknown function `nosuchfn`"));
        assert!(err("SELECT lower(ms) FROM events").contains("lower"));
    }

    #[test]
    fn aggregates_are_rejected_outside_the_places_they_belong() {
        let m = err("SELECT id FROM events WHERE sum(ms) > 1");
        assert!(m.contains("not allowed here"), "{m}");
        assert!(err("SELECT sum(max(ms)) FROM events").contains("cannot be nested"));
        assert!(err("SELECT id FROM events GROUP BY sum(ms)").contains("not allowed here"));
    }

    #[test]
    fn wildcard_is_only_valid_inside_count() {
        assert!(err("SELECT sum(*) FROM events").contains("not a valid argument"));
        assert!(err("SELECT lower(*) FROM events").contains("`*` is only valid"));
    }

    #[test]
    fn distinct_aggregate_is_checked_against_the_registry() {
        assert!(explain("SELECT uniq(DISTINCT url) FROM events").contains("uniq"));
        assert!(err("SELECT min(DISTINCT ms) FROM events").contains("does not support DISTINCT"));
    }

    #[test]
    fn parametric_aggregate_keeps_its_params() {
        let e = explain("SELECT quantile(0.9)(ms) FROM events");
        assert!(e.contains("aggs=[quantile(ms#0)]"), "{e}");
        assert_eq!(out("SELECT quantile(0.9)(ms) FROM events")[0], "quantile(0.9)(ms): Float64");
    }

    // ---------------------------------------------------------- aggregation

    #[test]
    fn group_by_splits_into_aggregate_then_project() {
        let e = explain("SELECT url, sum(ms) * 2 FROM events GROUP BY url");
        assert_eq!(
            e,
            "Project [url#0 AS url, (sum(ms)#1 * 2) AS sum(ms) * 2]\n  \
             Aggregate group=[url#0] aggs=[sum(ms#1)]\n    \
             Scan default.events [url, ms]\n"
        );
    }

    #[test]
    fn implicit_global_group_needs_no_group_by() {
        let e = explain("SELECT count(*), max(ms) FROM events");
        assert!(e.contains("Aggregate group=[] aggs=[count(), max(ms#0)]"), "{e}");
        assert!(e.starts_with("Project [count(*)#0 AS count(*), max(ms)#1 AS max(ms)]"), "{e}");
    }

    #[test]
    fn ungrouped_column_beside_an_aggregate_is_an_error() {
        let m = err("SELECT url, sum(ms) FROM events");
        assert!(m.contains("`url` must appear in GROUP BY"), "{m}");
    }

    #[test]
    fn grouping_on_an_expression_matches_the_same_expression_upstream() {
        let e = explain("SELECT toYear(d), count(*) FROM events GROUP BY toYear(d)");
        assert!(e.contains("Aggregate group=[toYear(d#0)] aggs=[count()]"), "{e}");
        assert!(e.contains("Project [toYear(d)#0 AS toYear(d), count(*)#1 AS count(*)]"), "{e}");
    }

    #[test]
    fn repeated_aggregate_is_computed_once() {
        let e = explain("SELECT sum(ms), sum(ms) + 1 FROM events");
        assert!(e.contains("aggs=[sum(ms#0)]"), "one accumulator only: {e}");
        assert!(e.contains("(sum(ms)#0 + 1)"), "{e}");
    }

    #[test]
    fn a_qualified_and_bare_spelling_share_one_accumulator() {
        let e = explain("SELECT sum(e.ms), sum(ms) FROM events AS e");
        assert!(e.contains("aggs=[sum(ms#0)]"), "{e}");
    }

    #[test]
    fn having_is_a_filter_above_the_aggregate() {
        let e = explain("SELECT url, count(*) FROM events GROUP BY url HAVING count(*) > 3");
        let lines: Vec<&str> = e.lines().collect();
        assert!(lines[0].starts_with("Project"), "{e}");
        assert!(lines[1].trim_start().starts_with("Filter (count(*)#1 > 3)"), "{e}");
        assert!(lines[2].trim_start().starts_with("Aggregate"), "{e}");
    }

    #[test]
    fn select_alias_is_visible_to_group_by_having_and_order_by() {
        let e = explain(
            "SELECT toYear(d) AS y, sum(ms) AS total FROM events \
             GROUP BY y HAVING total > 10 ORDER BY total DESC",
        );
        assert!(e.contains("Aggregate group=[toYear(d#1)] aggs=[sum(ms#0)]"), "{e}");
        assert!(e.contains("Filter (sum(ms)#1 > 10)"), "{e}");
        assert!(e.contains("Sort [sum(ms)#1 DESC]"), "{e}");
        assert_eq!(
            out("SELECT toYear(d) AS y, sum(ms) AS total FROM events GROUP BY y"),
            // `sum` is Nullable even over a non-Nullable column: an empty
            // group has no total, and the answer must not depend on how the
            // argument was declared. See `SumAcc::finish`.
            vec!["y: UInt16", "total: Nullable(UInt64)"]
        );
    }

    #[test]
    fn aggregate_result_types_come_from_the_registry() {
        assert_eq!(
            out("SELECT count(*) AS c, avg(ms) AS a, min(url) AS m FROM events"),
            vec!["c: UInt64", "a: Float64", "m: String"]
        );
    }

    #[test]
    fn having_without_aggregation_filters_below_the_projection() {
        let e = explain("SELECT id FROM events HAVING id > 5");
        assert!(e.contains("Filter (id#0 > 5)"), "{e}");
        assert!(!e.contains("Aggregate"), "{e}");
    }

    // --------------------------------------------------- sort/limit/distinct

    #[test]
    fn order_by_sits_below_the_projection_so_it_can_see_unselected_columns() {
        let e = explain("SELECT url FROM events ORDER BY ms DESC");
        assert_eq!(
            e,
            "Project [url#0 AS url]\n  Sort [ms#1 DESC]\n    Scan default.events [url, ms]\n"
        );
    }

    #[test]
    fn order_by_ordinal_refers_to_the_select_list() {
        let e = explain("SELECT url, ms FROM events ORDER BY 2");
        assert!(e.contains("Sort [ms#1]"), "{e}");
        assert!(err("SELECT url FROM events ORDER BY 5").contains("out of range"));
    }

    #[test]
    fn group_by_ordinal_refers_to_the_select_list() {
        let e = explain("SELECT toYear(d), count(*) FROM events GROUP BY 1");
        assert!(e.contains("Aggregate group=[toYear(d#0)] aggs=[count()]"), "{e}");
        assert!(err("SELECT url FROM events GROUP BY 3").contains("out of range"));
        assert!(err("SELECT * FROM events GROUP BY 1").contains("contains `*`"));
    }

    #[test]
    fn ordinals_also_work_above_a_distinct_and_a_union() {
        assert!(explain("SELECT DISTINCT url FROM events ORDER BY 1").contains("Sort [url#0]"));
        assert!(explain("SELECT ms FROM events UNION ALL SELECT age FROM users ORDER BY 1")
            .contains("Sort [ms#0]"));
    }

    #[test]
    fn distinct_moves_the_sort_above_the_projection() {
        let e = explain("SELECT DISTINCT url FROM events ORDER BY url");
        assert_eq!(
            e,
            "Sort [url#0]\n  Distinct\n    Project [url#0 AS url]\n      \
             Scan default.events [url]\n"
        );
    }

    #[test]
    fn limit_and_offset_become_one_node() {
        let e = explain("SELECT id FROM events LIMIT 10 OFFSET 5");
        assert!(e.starts_with("Limit 10 offset 5\n"), "{e}");
        // ClickHouse's reversed spelling means the same thing
        assert!(explain("SELECT id FROM events LIMIT 5, 10").starts_with("Limit 10 offset 5\n"));
        assert!(!explain("SELECT id FROM events").contains("Limit"));
    }

    #[test]
    fn limit_by_keeps_n_rows_per_key() {
        let e = explain("SELECT url, ms FROM events ORDER BY ms DESC LIMIT 2 BY url");
        assert!(e.contains("LimitBy 2 by [url#0]"), "{e}");
        let lines: Vec<&str> = e.lines().collect();
        assert!(lines[0].starts_with("Project"), "{e}");
        assert!(lines[1].trim_start().starts_with("LimitBy"), "{e}");
        assert!(lines[2].trim_start().starts_with("Sort"), "{e}");
    }

    #[test]
    fn limit_must_be_constant() {
        assert!(err("SELECT id FROM events LIMIT id").contains("LIMIT must be a constant"));
    }

    // ---------------------------------------------------------------- joins

    #[test]
    fn inner_join_on_splits_into_equi_pairs() {
        let e = explain(
            "SELECT u.name, e.ms FROM events AS e \
             JOIN users AS u ON e.user_id = u.user_id",
        );
        assert!(e.contains("InnerJoin on [l#0 = r#0]"), "{e}");
        assert!(e.contains("Project [name#3 AS name, ms#1 AS ms]"), "{e}");
        assert!(!e.contains("residual"), "{e}");
    }

    #[test]
    fn non_equi_conjuncts_become_the_residual() {
        let e = explain(
            "SELECT u.name FROM events AS e JOIN users AS u \
             ON e.user_id = u.user_id AND e.ms > u.age",
        );
        assert!(e.contains("on [l#0 = r#0]"), "{e}");
        assert!(e.contains("residual=(ms#1 > age#4)"), "{e}");
    }

    #[test]
    fn using_becomes_equi_pairs_on_same_named_columns() {
        let e = explain("SELECT name FROM events JOIN users USING (user_id)");
        assert!(e.contains("InnerJoin on [l#0 = r#0]"), "{e}");
    }

    // ------------------------------------------------- USING column merging

    /// `USING` names *one* output column. The join operator still emits both
    /// copies -- it concatenates its inputs and knows nothing about names --
    /// so this is entirely a scope rule, and `*` is where it shows.
    #[test]
    fn using_star_emits_the_join_column_once() {
        let cols = out("SELECT * FROM events JOIN users USING (user_id)");
        assert_eq!(
            cols,
            vec![
                "id: UInt64",
                "user_id: UInt64",
                "url: String",
                "ms: UInt32",
                "amount: Float64",
                "d: Date",
                "ts: DateTime",
                "name: String",
                "age: UInt8",
            ],
            "the key must appear once, in the left table's position"
        );
    }

    /// The merged column is what an unqualified reference means, so naming it
    /// is legal -- without the merge it is two candidates and "ambiguous".
    #[test]
    fn using_key_is_unambiguous_unqualified() {
        let cols = out("SELECT user_id, name FROM events JOIN users USING (user_id)");
        assert_eq!(cols, vec!["user_id: UInt64", "name: String"]);
    }

    /// ...while both per-side copies stay reachable qualified, because for an
    /// outer join they genuinely differ from the merged value.
    #[test]
    fn using_keeps_both_sides_reachable_qualified() {
        let e = explain(
            "SELECT events.user_id, users.user_id FROM events JOIN users USING (user_id)",
        );
        assert!(e.contains("Project [user_id#0 AS user_id, user_id#1 AS user_id]"), "{e}");

        // `t.*` means "the columns of t", key included.
        let cols = out("SELECT users.* FROM events JOIN users USING (user_id)");
        assert_eq!(cols, vec!["user_id: UInt64", "name: String", "age: UInt8"]);
    }

    /// One merged column for every join type, and the padded side is never the
    /// one it reads: LEFT takes the left copy, RIGHT the right copy, FULL the
    /// coalesce of both. INNER's two copies are equal by construction.
    #[test]
    fn using_merges_under_every_join_type() {
        for (sql, want) in [
            ("SELECT * FROM events JOIN users USING (user_id)", "user_id: UInt64"),
            ("SELECT * FROM events LEFT JOIN users USING (user_id)", "user_id: UInt64"),
            ("SELECT * FROM events RIGHT JOIN users USING (user_id)", "user_id: UInt64"),
            (
                "SELECT * FROM events FULL JOIN users USING (user_id)",
                "user_id: Nullable(UInt64)",
            ),
        ] {
            let cols = out(sql);
            assert_eq!(cols.len(), 9, "{sql} -> {cols:?}");
            assert_eq!(cols[1], want, "{sql} -> {cols:?}");
        }
        // CROSS JOIN takes no constraint, so there is nothing to merge and the
        // duplicate name is simply ambiguous, as it always was.
        assert_eq!(out("SELECT * FROM events CROSS JOIN users").len(), 10);
    }

    /// The one case that a column *count* check would not have caught: with a
    /// RIGHT join the left copy is the NULL-padded one, so the merged column
    /// has to be read off the right.
    #[test]
    fn using_reads_the_unpadded_side_of_an_outer_join() {
        let e = explain("SELECT user_id FROM events RIGHT JOIN users USING (user_id)");
        // #0 is events.user_id (padded), #1 is users.user_id.
        assert!(e.contains("Project [user_id#1 AS user_id]"), "right join takes the right: {e}");

        let e = explain("SELECT user_id FROM events LEFT JOIN users USING (user_id)");
        assert!(e.contains("Project [user_id#0 AS user_id]"), "left join takes the left: {e}");

        let e = explain("SELECT user_id FROM events FULL JOIN users USING (user_id)");
        assert!(
            e.contains("Project [coalesce(user_id#0, user_id#1) AS user_id]"),
            "full join coalesces: {e}"
        );
    }

    /// End-to-end: the values, not just the plan. An unmatched right row shows
    /// the *right* key in the merged column, and shows NULL through `a.key`.
    #[test]
    fn using_outer_join_values_come_from_the_matching_side() {
        let mut db = crate::session::Session::in_memory();
        db.execute("CREATE TABLE a (id UInt64, x String) ENGINE = MergeTree ORDER BY id").unwrap();
        db.execute("CREATE TABLE b (id UInt64, y String) ENGINE = MergeTree ORDER BY id").unwrap();
        db.execute("INSERT INTO a VALUES (1, 'a1'), (2, 'a2')").unwrap();
        db.execute("INSERT INTO b VALUES (2, 'b2'), (3, 'b3')").unwrap();

        let rs = db.query("SELECT * FROM a JOIN b USING (id) ORDER BY id").unwrap();
        assert_eq!(
            rs.schema.fields().iter().map(|f| f.name.as_str()).collect::<Vec<_>>(),
            vec!["id", "x", "y"],
            "inner join emits the key once"
        );
        assert_eq!(rs.to_values(), vec![vec![Value::UInt(2), Value::str("a2"), Value::str("b2")]]);

        // Row (3, 'b3') has no match in `a`: the merged id must be 3, not NULL.
        let rs = db.query("SELECT id, x, y FROM a RIGHT JOIN b USING (id) ORDER BY id").unwrap();
        assert_eq!(
            rs.to_values(),
            vec![
                vec![Value::UInt(2), Value::str("a2"), Value::str("b2")],
                vec![Value::UInt(3), Value::Null, Value::str("b3")],
            ]
        );
        // ...and `a.id` still reports the padding, which is why both entries
        // are kept in the scope.
        let rs = db.query("SELECT a.id FROM a RIGHT JOIN b USING (id) ORDER BY id").unwrap();
        assert_eq!(rs.to_values(), vec![vec![Value::UInt(2)], vec![Value::Null]]);

        // FULL: one unmatched row from each side, each showing its own key.
        let rs = db.query("SELECT id FROM a FULL JOIN b USING (id) ORDER BY id").unwrap();
        assert_eq!(
            rs.to_values(),
            vec![vec![Value::UInt(1)], vec![Value::UInt(2)], vec![Value::UInt(3)]],
            "no output row of a FULL join has the key missing on both sides"
        );

        // The merged column is a first-class expression: filterable, groupable.
        let rs = db.query("SELECT count(*) FROM a FULL JOIN b USING (id) WHERE id = 3").unwrap();
        assert_eq!(rs.scalar(), Some(Value::UInt(1)));
    }

    /// Chained `USING` joins merge onto an already-merged key, and the second
    /// merge has to inherit the *value* the first one settled on -- not the
    /// entry's ordering index, which still points at the padded left copy.
    #[test]
    fn a_second_using_merges_onto_the_first_merged_value() {
        let mut db = crate::session::Session::in_memory();
        for t in ["a", "b", "c"] {
            db.execute(&format!(
                "CREATE TABLE {t} (id UInt64, v{t} String) ENGINE = MergeTree ORDER BY id"
            ))
            .unwrap();
        }
        db.execute("INSERT INTO a VALUES (1, 'a1')").unwrap();
        db.execute("INSERT INTO b VALUES (2, 'b2')").unwrap();
        db.execute("INSERT INTO c VALUES (2, 'c2')").unwrap();

        // `a RIGHT JOIN b` leaves one row whose a.id is NULL and whose merged
        // id is 2; joining that to `c` on the merged key must still find 2.
        let rs = db
            .query("SELECT id, vc FROM a RIGHT JOIN b USING (id) JOIN c USING (id)")
            .unwrap();
        assert_eq!(rs.to_values(), vec![vec![Value::UInt(2), Value::str("c2")]]);
    }

    /// A `USING` key that is itself the merged key of an enclosed FULL join is
    /// a `coalesce`, and `Join::on` holds column indices. Refused, not folded
    /// into a silently-wrong equi-pair on one side.
    #[test]
    fn using_on_a_full_joins_merged_key_is_refused_not_guessed() {
        let m = err(
            "SELECT user_id FROM events FULL JOIN users USING (user_id) \
             JOIN users AS u2 USING (user_id)",
        );
        assert!(m.contains("merged key of an enclosed FULL JOIN"), "{m}");
    }

    #[test]
    fn cross_and_comma_joins_have_no_equi_pairs() {
        assert!(explain("SELECT name FROM events, users").contains("CrossJoin on []"));
        assert!(explain("SELECT name FROM events CROSS JOIN users").contains("CrossJoin on []"));
    }

    #[test]
    fn outer_joins_make_the_optional_side_nullable() {
        let cols = out(
            "SELECT e.id, u.name FROM events AS e LEFT JOIN users AS u ON e.user_id = u.user_id",
        );
        assert_eq!(cols, vec!["id: UInt64", "name: Nullable(String)"]);

        let cols = out(
            "SELECT e.id, u.name FROM events AS e RIGHT JOIN users AS u ON e.user_id = u.user_id",
        );
        assert_eq!(cols, vec!["id: Nullable(UInt64)", "name: String"]);

        let cols = out(
            "SELECT e.id, u.name FROM events AS e FULL JOIN users AS u ON e.user_id = u.user_id",
        );
        assert_eq!(cols, vec!["id: Nullable(UInt64)", "name: Nullable(String)"]);
    }

    #[test]
    fn an_unqualified_column_present_on_both_sides_is_ambiguous() {
        let m = err("SELECT user_id FROM events JOIN users ON events.id = users.user_id");
        assert!(m.contains("ambiguous"), "{m}");
        assert!(m.contains("events.user_id"), "{m}");
    }

    // ------------------------------------------------- subqueries and CTEs

    #[test]
    fn from_subquery_exposes_its_output_schema() {
        let e = explain("SELECT s.n FROM (SELECT ms AS n FROM events) AS s WHERE s.n > 1");
        assert!(e.contains("Filter (n#0 > 1)"), "{e}");
        let lines: Vec<&str> = e.lines().collect();
        assert!(lines[2].trim_start().starts_with("Project [ms#0 AS n]"), "{e}");
    }

    #[test]
    fn cte_is_inlined_at_its_reference() {
        let e = explain("WITH big AS (SELECT id, ms FROM events WHERE ms > 100) SELECT id FROM big");
        assert!(e.contains("Filter (ms#1 > 100)"), "{e}");
        assert!(e.contains("Scan default.events"), "{e}");
    }

    #[test]
    fn cte_is_inlined_once_per_reference() {
        let e = explain(
            "WITH c AS (SELECT user_id FROM events) \
             SELECT a.user_id FROM c AS a JOIN c AS b ON a.user_id = b.user_id",
        );
        assert_eq!(e.matches("Scan default.events").count(), 2, "{e}");
    }

    #[test]
    fn a_cte_shadows_a_base_table_of_the_same_name() {
        let e = explain("WITH users AS (SELECT id AS user_id FROM events) SELECT user_id FROM users");
        assert!(e.contains("Scan default.events"), "{e}");
        assert!(!e.contains("Scan default.users"), "{e}");
    }

    #[test]
    fn a_cte_cannot_see_itself() {
        assert!(err("WITH r AS (SELECT id FROM r) SELECT id FROM r").contains("r"));
    }

    // --------------------------------------------------------------- unions

    #[test]
    fn union_all_flattens_and_unifies_types() {
        let e = explain(
            "SELECT ms FROM events UNION ALL SELECT age FROM users \
             UNION ALL SELECT id FROM events",
        );
        assert!(e.starts_with("Union All\n"), "{e}");
        assert_eq!(e.lines().filter(|l| l.starts_with("  Project")).count(), 3, "{e}");
        assert_eq!(
            out("SELECT ms FROM events UNION ALL SELECT age FROM users"),
            vec!["ms: UInt64"]
        );
    }

    #[test]
    fn union_distinct_is_a_different_node_than_union_all() {
        assert!(explain("SELECT ms FROM events UNION SELECT age FROM users")
            .starts_with("Union Distinct\n"));
    }

    #[test]
    fn union_arity_and_type_mismatches_are_reported() {
        let m = err("SELECT id, ms FROM events UNION ALL SELECT age FROM users");
        assert!(m.contains("disagree on width"), "{m}");
        let m = err("SELECT url FROM events UNION ALL SELECT age FROM users");
        assert!(m.contains("disagree on column 1"), "{m}");
    }

    #[test]
    fn query_level_order_by_applies_above_a_union() {
        let e = explain("SELECT ms FROM events UNION ALL SELECT age FROM users ORDER BY ms LIMIT 3");
        assert!(e.starts_with("Limit 3 offset 0\n  Sort [ms#0]\n    Union All\n"), "{e}");
    }

    #[test]
    fn except_and_intersect_are_explicitly_unimplemented() {
        let m = err("SELECT ms FROM events EXCEPT SELECT age FROM users");
        assert!(m.contains("not implemented") && m.contains("EXCEPT"), "{m}");
        let m = err("SELECT ms FROM events INTERSECT SELECT age FROM users");
        assert!(m.contains("INTERSECT"), "{m}");
    }

    // ---------------------------------------------------- unsupported bits

    #[test]
    fn subquery_expressions_are_refused_rather_than_faked() {
        let m = err("SELECT id FROM events WHERE ms = (SELECT max(age) FROM users)");
        assert!(m.contains("not implemented") && m.contains("scalar subqueries"), "{m}");
        // A membership test is refused only where it would have to be a value.
        // As a whole WHERE conjunct it is a join, and has its own tests below.
        let m = err("SELECT id FROM events WHERE user_id IN (SELECT user_id FROM users) OR id = 1");
        assert!(m.contains("semi-join") && m.contains("per-row value"), "{m}");
        let m = err("SELECT NOT EXISTS (SELECT 1 FROM users) FROM events");
        assert!(m.contains("EXISTS") && m.contains("semi-join"), "{m}");
    }

    // ------------------------------------------------- membership subqueries

    #[test]
    fn in_subquery_binds_to_a_semi_join_over_the_subquery_plan() {
        let e = explain("SELECT id FROM events WHERE user_id IN (SELECT user_id FROM users)");
        // The subquery is a *relation* in the tree -- its own Scan, narrowed to
        // its own column -- and not a literal list, which is the whole point.
        assert!(e.contains("InnerJoin on [l#1 = r#0]"), "{e}");
        assert!(e.contains("Distinct"), "{e}");
        assert!(e.contains("Scan default.users [user_id]"), "{e}");
        // No `Project` restoring the outer width: the binder's own projection
        // drops the appended column.
        assert_eq!(e.matches("Project").count(), 2, "{e}");
    }

    #[test]
    fn not_in_over_a_non_nullable_column_needs_no_census() {
        // `id` and `user_id` are both non-nullable here, so cases 2 and 4 in
        // `LogicalPlan::in_subquery` cannot fire and the plan is a plain
        // anti-join: one join, one pass over the subquery.
        let e = explain("SELECT id FROM events WHERE user_id NOT IN (SELECT user_id FROM users)");
        assert!(e.contains("LeftJoin on [l#1 = r#0]"), "{e}");
        assert!(e.contains("Filter user_id#2 IS NULL"), "{e}");
        assert_eq!(e.matches("Join").count(), 1, "no census join: {e}");
        assert_eq!(e.matches("Scan default.users").count(), 1, "one pass: {e}");
    }

    #[test]
    fn not_in_over_a_nullable_column_grows_the_census() {
        let e = explain("SELECT id FROM notes WHERE n NOT IN (SELECT n FROM notes)");
        assert!(e.contains("CrossJoin"), "{e}");
        assert!(e.contains("aggs=[count(), count(n#0)]"), "{e}");
        // Two passes over the subquery, which is what the census costs.
        assert_eq!(e.matches("Scan default.notes").count(), 3, "{e}");
    }

    #[test]
    fn exists_is_a_limit_one_probe_not_a_scan() {
        let e = explain("SELECT id FROM events WHERE EXISTS (SELECT 1 FROM users)");
        assert!(e.contains("CrossJoin"), "{e}");
        assert!(e.contains("Limit 1 offset 0"), "{e}");
        let e = explain("SELECT id FROM events WHERE NOT EXISTS (SELECT * FROM users)");
        assert!(e.contains("LeftJoin on []"), "{e}");
        assert!(e.contains("Limit 1 offset 0"), "{e}");
        // `EXISTS (SELECT *)` is legal: existence does not care how wide a row
        // is. The old fold refused it, because it read column 0 of a result it
        // had already computed.
        assert!(e.contains("Project [true AS exists]"), "{e}");
    }

    #[test]
    fn a_membership_conjunct_leaves_the_others_below_the_join() {
        // The ordinary predicate has to reach the scan, or the join builds over
        // rows a filter was going to throw away.
        let e =
            explain("SELECT id FROM events WHERE ms > 5 AND user_id IN (SELECT user_id FROM users)");
        assert!(e.contains("Filter (ms#2 > 5)"), "{e}");
        let filter = e.find("Filter").expect("a filter");
        let join = e.find("InnerJoin").expect("a join");
        assert!(join < filter, "the filter must sit under the join:\n{e}");
    }

    #[test]
    fn in_subquery_type_and_arity_are_still_checked() {
        let m = err("SELECT id FROM events WHERE id IN (SELECT id, user_id FROM events)");
        assert!(m.contains("exactly one column"), "{m}");
        let m = err("SELECT id FROM events WHERE url IN (SELECT user_id FROM users)");
        assert!(m.contains("String"), "{m}");
    }

    #[test]
    fn with_totals_is_refused() {
        assert!(err("SELECT url, count(*) FROM events GROUP BY url WITH TOTALS")
            .contains("TOTALS"));
    }

    // ------------------------------------------------------------ odds/ends

    #[test]
    fn select_without_from_produces_a_single_row() {
        let e = explain("SELECT 1 + 1 AS two");
        assert_eq!(e, "Project [(1 + 1) AS two]\n  Values 1 rows\n");
        assert!(err("SELECT *").contains("requires a FROM clause"));
    }

    #[test]
    fn bare_values_becomes_a_literal_row_set() {
        let e = explain("VALUES (1, 'a'), (2, 'b')");
        assert_eq!(e, "Values 2 rows\n");
        assert_eq!(
            plan_of("VALUES (1, 'a')")
                .schema()
                .fields()
                .iter()
                .map(|f| f.name.clone())
                .collect::<Vec<_>>(),
            vec!["c1", "c2"]
        );
        assert!(err("VALUES (1), (2, 3)").contains("expected 1"));
    }

    #[test]
    fn is_null_binds_through() {
        let e = explain("SELECT id FROM events WHERE url IS NOT NULL");
        assert!(e.contains("url#1 IS NOT NULL"), "{e}");
    }

    #[test]
    fn alias_cycles_terminate() {
        // `x` on the right resolves to the column, not back to the alias
        let e = explain("SELECT ms + 1 AS ms FROM events ORDER BY ms");
        assert!(e.contains("(ms#0 + 1)"), "{e}");
    }

    #[test]
    fn bind_expr_standalone_uses_the_schema_it_is_handed() {
        let c = catalog();
        let mut b = Binder::new(&c);
        let schema = c.table(&ObjectName::bare("users")).unwrap().schema().clone();
        let e = crate::sql::parser::parse_expr("age + 1").unwrap();
        let bound = b.bind_expr_standalone(&e, &schema).unwrap();
        assert_eq!(bound.to_string(), "(age#2 + 1)");
        assert_eq!(bound.ty(), DataType::Int64);

        let bad = crate::sql::parser::parse_expr("nope").unwrap();
        let m = match b.bind_expr_standalone(&bad, &schema) {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        };
        assert!(m.contains("available: user_id, name, age"), "{m}");
    }

    #[test]
    fn an_aggregate_may_appear_only_in_order_by() {
        let e = explain("SELECT url FROM events GROUP BY url ORDER BY count(*) DESC");
        assert!(e.contains("Aggregate group=[url#0] aggs=[count()]"), "{e}");
        assert!(e.contains("Sort [count(*)#1 DESC]"), "{e}");
        // ... and the projection still only exposes the grouped column
        assert_eq!(out("SELECT url FROM events GROUP BY url ORDER BY count(*)"), vec!["url: String"]);
    }

    #[test]
    fn limit_by_keys_are_rewritten_over_the_aggregate_too() {
        let e = explain(
            "SELECT user_id, url, count(*) FROM events GROUP BY user_id, url LIMIT 1 BY user_id",
        );
        assert!(e.contains("LimitBy 1 by [user_id#0]"), "{e}");
    }

    #[test]
    fn aggregate_over_an_expression() {
        let e = explain("SELECT sum(ms * 2) FROM events");
        assert!(e.contains("aggs=[sum((ms#0 * 2))]"), "{e}");
    }

    #[test]
    fn if_combinator_resolves_like_any_other_aggregate() {
        let e = explain("SELECT countIf(ms > 10) FROM events");
        assert!(e.contains("aggs=[countIf((ms#0 > 10))]"), "{e}");
        assert_eq!(out("SELECT countIf(ms > 10) AS c FROM events"), vec!["c: UInt64"]);
    }

    #[test]
    fn unaliased_subquery_columns_are_reachable_unqualified() {
        let e = explain("SELECT n FROM (SELECT ms AS n FROM events)");
        assert!(e.contains("Project [n#0 AS n]"), "{e}");
    }

    #[test]
    fn nested_subqueries_stack() {
        let e = explain("SELECT n FROM (SELECT m AS n FROM (SELECT ms AS m FROM events) AS i) AS o");
        assert_eq!(e.matches("Project").count(), 3, "{e}");
    }

    #[test]
    fn distinct_without_order_by_is_just_distinct_over_project() {
        let e = explain("SELECT DISTINCT url FROM events");
        assert_eq!(e, "Distinct\n  Project [url#0 AS url]\n    Scan default.events [url]\n");
    }

    #[test]
    fn star_over_a_join_concatenates_both_sides() {
        let cols = out("SELECT * FROM events AS e JOIN users AS u ON e.user_id = u.user_id");
        assert_eq!(cols.len(), 10);
        assert_eq!(cols[7], "user_id: UInt64");
    }

    #[test]
    fn group_by_matches_however_the_column_is_spelled() {
        let e = explain("SELECT e.url, count(*) FROM events AS e GROUP BY url");
        assert!(e.contains("Project [url#0 AS url, count(*)#1 AS count(*)]"), "{e}");
    }

    #[test]
    fn negated_in_list_keeps_its_negation() {
        let e = explain("SELECT id FROM events WHERE ms NOT IN (1, 2)");
        assert!(e.contains("ms#1 NOT IN (1, 2)"), "{e}");
    }

    #[test]
    fn a_cte_may_hold_a_union() {
        let e = explain("WITH u AS (SELECT ms FROM events UNION ALL SELECT age FROM users) \
                         SELECT ms FROM u");
        assert!(e.contains("Union All"), "{e}");
    }

    #[test]
    fn order_by_columns_are_added_to_the_scan_projection() {
        // `ms` appears nowhere but ORDER BY, and the scan still has to read it
        assert!(explain("SELECT url FROM events ORDER BY ms").contains("[url, ms]"));
    }

    #[test]
    fn bound_plan_survives_the_optimizer() {
        // The binder's output is the optimizer's input; every shape it can
        // produce has to make it through all four passes unchanged in meaning.
        for sql in [
            "SELECT url, sum(ms) FROM events WHERE d >= '2024-01-01' GROUP BY url \
             HAVING sum(ms) > 10 ORDER BY url LIMIT 5",
            "SELECT DISTINCT url FROM events ORDER BY url",
            "SELECT u.name, e.ms FROM events AS e LEFT JOIN users AS u \
             ON e.user_id = u.user_id WHERE e.ms > 1",
            "WITH c AS (SELECT id, ms FROM events) SELECT id FROM c WHERE ms IN (1, 2)",
            "SELECT ms FROM events UNION ALL SELECT age FROM users ORDER BY ms",
            "SELECT count(*) FROM events",
        ] {
            let p = plan_of(sql);
            let before = p.schema().clone();
            let opt = super::super::optimizer::optimize(p)
                .unwrap_or_else(|e| panic!("optimize {sql}: {e}"));
            assert_eq!(&before, opt.schema(), "schema changed for {sql}");
        }
    }

    #[test]
    fn zone_filter_extraction_works_on_a_reparsed_date() {
        let p = super::super::optimizer::optimize(plan_of(
            "SELECT id FROM events WHERE d = '2024-01-01'",
        ))
        .unwrap();
        let e = p.explain();
        assert!(e.contains("zonemap=1"), "the date literal must be prunable: {e}");
    }

    // =================================== adversarial review: confirmed bugs

    fn bind_err(sql: &str) -> String {
        let c = catalog();
        let mut b = Binder::new(&c);
        let st = crate::sql::parse(sql).unwrap_or_else(|e| panic!("parse {sql}: {e}"));
        match &st[0] {
            Statement::Query(q) => match b.bind_query(q) {
                Ok(p) => panic!("expected an error for `{sql}`, got:\n{}", p.explain()),
                Err(e) => e.to_string(),
            },
            other => panic!("not a query: {other:?}"),
        }
    }

    /// BUG 1 (critical). The alias table is consulted while binding the very
    /// select item that *defines* the alias, so the alias body is substituted
    /// into itself and the expression is applied twice.
    ///
    /// `ctx.expanding` is only pushed inside the substitution path
    /// (`Binder::bind`, `Expr::Column` arm), so it is empty when the defining
    /// item is bound -- the guard that is supposed to stop this never fires.
    #[test]
    fn bug_self_named_alias_is_applied_twice() {
        let e = explain("SELECT ms + 1 AS ms FROM events");
        assert_eq!(
            e, "Project [(ms#0 + 1) AS ms]\n  Scan default.events [ms]\n",
            "`ms + 1 AS ms` must compute ms+1 once, got: {e}"
        );
    }

    /// Same defect on a function call, and the ORDER BY key ends up computed
    /// from a *different* expression than the projected column.
    #[test]
    fn bug_self_named_alias_desynchronises_sort_and_project() {
        let e = explain("SELECT lower(url) AS url FROM events");
        assert!(!e.contains("lower(lower("), "double application: {e}");

        let e = explain("SELECT ms * 2 AS ms FROM events ORDER BY ms");
        // Sort binds `(ms#0 * 2)` while Project binds `((ms#0 * 2) * 2)`.
        assert!(!e.contains("((ms#0 * 2) * 2)"), "sort/project disagree: {e}");
    }

    /// The same defect turns a perfectly ordinary query into a bind error,
    /// because the alias body is re-entered while `in_agg` is set.
    #[test]
    fn bug_self_named_alias_on_an_aggregate_is_rejected() {
        let p = try_plan("SELECT sum(ms) AS ms FROM events");
        assert!(p.is_ok(), "`sum(ms) AS ms` must bind, got: {:?}", p.err());
    }

    /// End-to-end proof that BUG 1 silently returns wrong numbers.
    #[test]
    fn bug_self_named_alias_returns_wrong_rows() {
        let mut db = crate::session::Session::in_memory();
        db.execute("CREATE TABLE t (id UInt64, ms UInt32) ENGINE = MergeTree ORDER BY id")
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 10), (2, 20)").unwrap();
        let got = db.query("SELECT ms + 1 AS ms FROM t ORDER BY id").unwrap().to_string();
        assert!(got.contains("11") && got.contains("21"), "ms+1 over (10,20) is (11,21): {got}");
    }

    /// BUG 2. With `SELECT DISTINCT`, ORDER BY is re-bound against the output
    /// schema (`select_block`, the `sel.distinct` branch), so an ORDER BY key
    /// that is *literally a select-list expression* no longer resolves. The
    /// same query without DISTINCT binds fine.
    ///
    /// NOTE: as the review wrote it, this test could not pass under *any*
    /// implementation -- it called `bind_err` (which panics when the query
    /// binds) and then panicked unconditionally on the next line. Only that
    /// scaffolding is changed here. What is asserted is what the doc comment
    /// above describes, and it is strictly stronger than "it binds": the Sort
    /// has to survive above the Distinct, and the projection still has to
    /// compute the expression exactly once.
    #[test]
    fn bug_distinct_rejects_order_by_of_a_select_list_expression() {
        // control: without DISTINCT this is fine
        assert!(explain("SELECT ms + 1 FROM events ORDER BY ms + 1").contains("Sort [(ms#0 + 1)]"));

        let e = try_plan("SELECT DISTINCT ms + 1 FROM events ORDER BY ms + 1")
            .unwrap_or_else(|m| {
                panic!("DISTINCT must not break ORDER BY of a select-list expression: {m}")
            })
            .explain();
        let lines: Vec<&str> = e.lines().collect();
        assert!(lines[0].starts_with("Sort ["), "{e}");
        assert!(lines[1].trim_start().starts_with("Distinct"), "{e}");
        assert!(lines[2].trim_start().contains("(ms#0 + 1)"), "{e}");

        // The other half of the rule: a key that is *not* selected has no
        // column left to sort by once duplicates are collapsed.
        assert!(bind_err("SELECT DISTINCT url FROM events ORDER BY ms")
            .contains("SELECT DISTINCT list"));
    }

    /// Same defect for an aggregate: `ORDER BY sum(ms)` where `sum(ms)` is in
    /// the select list is refused only when DISTINCT is present.
    #[test]
    fn bug_distinct_rejects_order_by_of_a_select_list_aggregate() {
        assert!(try_plan("SELECT url, sum(ms) FROM events GROUP BY url ORDER BY sum(ms)").is_ok());
        let p = try_plan(
            "SELECT DISTINCT url, sum(ms) FROM events GROUP BY url ORDER BY sum(ms)",
        );
        assert!(p.is_ok(), "DISTINCT + ORDER BY aggregate: {:?}", p.err());
    }

    /// BUG 3. `Binder::values` only promotes a cell's type when the value is
    /// non-NULL, so a NULL in any row but the first is dropped from the
    /// column's declared type. The result is order-dependent.
    #[test]
    fn bug_values_column_nullability_depends_on_row_order() {
        let a = out("VALUES (1), (NULL)");
        let b = out("VALUES (NULL), (1)");
        assert_eq!(a, b, "the same two rows in the other order type differently");
        assert_eq!(a, vec!["c1: Nullable(Int64)"], "a column holding NULL must be Nullable");
    }

    /// BUG 4. `Expr::Cast` copies the written target type verbatim, dropping
    /// the operand's `Nullable` wrapper, so the plan's declared type says the
    /// column cannot be NULL while the executor happily produces NULLs.
    #[test]
    fn bug_cast_drops_nullability_from_the_declared_type() {
        let c = {
            let mut c = Catalog::in_memory();
            c.create_table(table("n", &[("id", "UInt64"), ("v", "Nullable(Int64)")], &[0]), false)
                .unwrap();
            c
        };
        let mut b = Binder::new(&c);
        let st = crate::sql::parse("SELECT CAST(v AS Int64) FROM n").unwrap();
        let p = match &st[0] {
            Statement::Query(q) => b.bind_query(q).unwrap(),
            _ => unreachable!(),
        };
        assert_eq!(
            p.schema().fields()[0].ty,
            DataType::Nullable(Box::new(DataType::Int64)),
            "CAST of a Nullable operand must stay Nullable"
        );
    }

    /// BUG 5. `ordinal()` and `const_count()` accept any numeric literal and
    /// silently truncate, so `ORDER BY 1.5` becomes `ORDER BY 1` and
    /// `LIMIT 2.7` becomes `LIMIT 2` instead of being rejected.
    #[test]
    fn bug_fractional_ordinals_and_limits_are_silently_truncated() {
        let c = catalog();
        let mut b = Binder::new(&c);
        let st = crate::sql::parse("SELECT id, ms FROM events ORDER BY 1.5").unwrap();
        let r = match &st[0] {
            Statement::Query(q) => b.bind_query(q),
            _ => unreachable!(),
        };
        assert!(r.is_err(), "ORDER BY 1.5 silently became ORDER BY 1");
    }

    fn try_plan(sql: &str) -> std::result::Result<LogicalPlan, String> {
        let c = catalog();
        let mut b = Binder::new(&c);
        let st = crate::sql::parse(sql).map_err(|e| format!("parse: {e}"))?;
        match &st[0] {
            Statement::Query(q) => b.bind_query(q).map_err(|e| e.to_string()),
            other => Err(format!("not a query: {other:?}")),
        }
    }

    // ------------------------------------------------------ recursion depth

    /// The case the parser's own guard cannot see: a chain of CTEs is *flat*
    /// to the parser (one `WITH` list) but nests here, because a CTE is
    /// inlined afresh at every reference. Before the guard this was a stack
    /// overflow -- a SIGSEGV, not an error -- from a few kilobytes of SQL.
    #[test]
    fn a_flat_cte_chain_still_nests_the_binder_and_is_bounded() {
        let mut sql = String::from("WITH c0 AS (SELECT id FROM events)");
        for i in 1..MAX_BIND_DEPTH {
            sql.push_str(&format!(", c{i} AS (SELECT id FROM c{})", i - 1));
        }
        sql.push_str(&format!(" SELECT id FROM c{}", MAX_BIND_DEPTH - 1));
        let m = err(&sql);
        assert!(m.contains("nests more than"), "{m}");

        // A short chain is still perfectly legal, and still inlines.
        let e = explain(
            "WITH c0 AS (SELECT id FROM events), c1 AS (SELECT id FROM c0) SELECT id FROM c1",
        );
        assert!(e.contains("Scan default.events"), "{e}");
    }

    /// Same story one layer down: `a + a + a + ...` is a loop in the parser
    /// and a left-deep tree here, so expression nesting is bounded too.
    #[test]
    fn a_flat_operator_chain_is_bounded() {
        let sql = format!("SELECT {} FROM events", vec!["ms"; MAX_BIND_DEPTH + 5].join(" + "));
        let m = err(&sql);
        assert!(m.contains("nests more than"), "{m}");
    }

    /// Nested set operations are the third flat-parse/deep-bind shape, and the
    /// only one that reaches neither `query` nor `table_ref`.
    #[test]
    fn a_flat_union_chain_is_bounded() {
        let one = "SELECT id FROM events";
        let sql = vec![one; MAX_BIND_DEPTH + 5].join(" UNION ALL ");
        let m = err(&sql);
        assert!(m.contains("nests more than"), "{m}");
    }

    /// What the RAII guard buys over a hand-written decrement: every one of
    /// those refusals returns through `?` from the middle of the recursion, so
    /// a leaked counter would make the *next* query on the same binder fail
    /// for no reason. One binder, a rejected query, then a legal one.
    #[test]
    fn a_refused_query_does_not_poison_the_next_one() {
        let c = catalog();
        let mut b = Binder::new(&c);
        let parse = |sql: &str| match &crate::sql::parse(sql).unwrap()[0] {
            Statement::Query(q) => (**q).clone(),
            other => panic!("not a query: {other:?}"),
        };

        let deep = parse(&format!(
            "SELECT {} FROM events",
            vec!["ms"; MAX_BIND_DEPTH + 5].join(" + ")
        ));
        let ok = parse("SELECT id FROM events");
        for _ in 0..3 {
            assert!(b.bind_query(&deep).is_err());
            assert!(b.bind_query(&ok).is_ok(), "the counter leaked out of a `?` return");
        }
    }

    // -------------------------------------------------------- mutations

    /// Bind a `DELETE`/`UPDATE` statement, whichever spelling it arrived in.
    fn mutation(c: &Catalog, sql: &str) -> Result<MutationPlan> {
        let mut b = Binder::new(c);
        match &crate::sql::parse(sql).unwrap_or_else(|e| panic!("parse {sql}: {e}"))[0] {
            Statement::AlterDelete { table, predicate } => b.bind_delete(table, predicate),
            Statement::AlterUpdate { table, assignments, predicate } => {
                b.bind_update(table, assignments, predicate)
            }
            other => panic!("not a mutation: {other:?}"),
        }
    }

    /// The optimized plan for a mutation, which is what the executor sees.
    fn mut_plan(sql: &str) -> String {
        let c = catalog();
        let mut m = mutation(&c, sql).unwrap_or_else(|e| panic!("bind {sql}: {e}"));
        m.source = crate::planner::optimizer::optimize(m.source).unwrap();
        m.explain()
    }

    fn mut_err(sql: &str) -> String {
        let c = catalog();
        match mutation(&c, sql) {
            Ok(p) => panic!("expected an error for `{sql}`, got:\n{}", p.explain()),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn delete_reads_only_the_columns_its_predicate_needs() {
        // Seven columns in `events`; the predicate names one. A delete needs no
        // row value at all, so the other six must not be decoded.
        let e = mut_plan("DELETE FROM events WHERE ms > 100");
        assert!(e.starts_with("Delete default.events\n"), "{e}");
        assert!(e.contains("Scan default.events [ms]"), "{e}");
        assert!(e.contains("prewhere="), "the predicate should reach the scan: {e}");
    }

    #[test]
    fn delete_and_alter_delete_bind_identically() {
        assert_eq!(
            mut_plan("DELETE FROM events WHERE ms > 100"),
            mut_plan("ALTER TABLE events DELETE WHERE ms > 100"),
            "the two spellings must be one plan, not two implementations"
        );
        assert_eq!(
            mut_plan("UPDATE events SET ms = ms + 1 WHERE id = 7"),
            mut_plan("ALTER TABLE events UPDATE ms = ms + 1 WHERE id = 7"),
        );
    }

    /// The reason the predicate is left as a `Filter` for the optimizer rather
    /// than installed into the scan by hand: it picks up zone-map pruning on
    /// the way through, exactly as a `SELECT` does.
    #[test]
    fn a_mutation_predicate_gets_the_same_zone_filters_as_a_select() {
        let m = mut_plan("DELETE FROM events WHERE id > 500");
        assert!(m.contains("zonemap=1"), "{m}");
        let s = crate::planner::optimizer::optimize(plan_of("SELECT id FROM events WHERE id > 500"))
            .unwrap()
            .explain();
        assert!(s.contains("zonemap=1"), "{s}");
        // Same scan line under both, modulo the `Delete` root: one predicate
        // path, so the mutation cannot silently lose a prune the SELECT gets.
        let scan = |p: &str| p.lines().find(|l| l.contains("Scan ")).unwrap().trim().to_string();
        assert_eq!(scan(&m), scan(&s));
    }

    /// `WHERE` is optional, and its absence must not cost a per-row predicate:
    /// the constant folds away and the scan is left bare.
    #[test]
    fn an_unconditional_delete_folds_its_predicate_away() {
        let e = mut_plan("DELETE FROM events");
        assert!(e.starts_with("Delete default.events\n"), "{e}");
        assert!(!e.contains("prewhere="), "a `WHERE true` should not survive: {e}");
        assert!(!e.contains("Filter"), "{e}");
    }

    #[test]
    fn update_produces_the_whole_replacement_row() {
        let c = catalog();
        let m = mutation(&c, "UPDATE events SET url = 'x', ms = ms * 2 WHERE id = 1").unwrap();
        assert_eq!(m.kind, MutationKind::Update);
        // The plan's output schema is the table's, so the block it yields can
        // go straight into `Table::insert` with no widening.
        let names: Vec<&str> =
            m.source.schema().fields().iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["id", "user_id", "url", "ms", "amount", "d", "ts"]);
        let e = m.explain();
        assert!(e.contains("'x' AS url"), "{e}");
        // `UInt32 * 2` widens to UInt64, so the assignment is narrowed back to
        // the declared type before it is written -- the column it lands in is
        // 4 bytes wide and an 8-byte one would not fit it.
        assert!(e.contains("CAST((ms#3 * 2) AS UInt32) AS ms"), "{e}");
        assert!(e.contains("id#0 AS id"), "untouched columns pass through: {e}");
    }

    /// Assignments are one simultaneous projection over the pre-update row, so
    /// a swap swaps. Sequential application would leave both columns equal.
    #[test]
    fn update_assignments_read_the_pre_update_row() {
        let c = catalog();
        let m = mutation(&c, "UPDATE events SET id = user_id, user_id = id WHERE id > 0").unwrap();
        let e = m.explain();
        assert!(e.contains("user_id#1 AS id"), "{e}");
        assert!(e.contains("id#0 AS user_id"), "{e}");
    }

    /// A literal that is not already the column's type is cast once at plan
    /// time; a matching one is not wrapped at all, because a no-op `Cast` costs
    /// a copy of the whole column per block.
    #[test]
    fn update_casts_only_on_a_real_type_mismatch() {
        let c = catalog();
        let e = mutation(&c, "UPDATE events SET amount = 1 WHERE id = 1").unwrap().explain();
        assert!(e.contains("CAST(1 AS Float64) AS amount"), "{e}");
        let e = mutation(&c, "UPDATE events SET url = 'x' WHERE id = 1").unwrap().explain();
        assert!(!e.contains("CAST"), "same type, no cast: {e}");
    }

    #[test]
    fn mutations_reject_what_a_where_clause_rejects() {
        assert!(mut_err("DELETE FROM events WHERE nosuch = 1").contains("unknown column"));
        assert!(mut_err("DELETE FROM nosuch WHERE id = 1").contains("nosuch"));
        assert!(mut_err("UPDATE events SET nosuch = 1 WHERE id = 1").contains("nosuch"));
        assert!(mut_err("DELETE FROM events WHERE count(*) > 1").contains("aggregate"));
        // Silently keeping one of two assignments to the same column is the
        // kind of thing that loses a write without an error.
        let m = mut_err("UPDATE events SET ms = 1, ms = 2 WHERE id = 1");
        assert!(m.contains("assigned twice"), "{m}");
    }

    /// The payoff of routing the predicate through the ordinary passes: a
    /// mutation on the primary key reaches the MPH index, and `EXPLAIN` proves
    /// it rather than leaving the access path invisible.
    ///
    /// Single-key only here: the `IN` gate weighs probes against live rows, and
    /// the fixture catalog has none, so a batch correctly declines. That gate is
    /// `physical`'s to test; what this pins is that a mutation reaches it.
    #[test]
    fn a_keyed_mutation_predicate_lowers_to_an_index_lookup() {
        let c = catalog();
        for sql in [
            "DELETE FROM events WHERE id = 7",
            "UPDATE events SET ms = 0 WHERE id = 7",
        ] {
            let mut m = mutation(&c, sql).unwrap();
            m.source = crate::planner::optimizer::optimize(m.source).unwrap();
            let phys = crate::planner::physical::lower(&m.source, &c).unwrap().explain();
            assert!(phys.contains("IndexLookup"), "{sql}:\n{phys}");
        }
    }
    // -------------------------------------------------------- window functions

    #[test]
    fn a_window_is_a_node_between_the_source_and_the_projection() {
        // The SQL evaluation order made visible: FROM -> WHERE -> GROUP BY ->
        // HAVING -> WINDOW -> SELECT -> ORDER BY.
        let e = explain(
            "SELECT url, sum(ms) OVER (PARTITION BY url ORDER BY id) FROM events \
             WHERE ms > 1 ORDER BY id",
        );
        let want = ["Project", "Sort", "Window", "Sort", "Filter", "Scan"];
        let got: Vec<&str> =
            e.lines().map(|l| l.trim().split_whitespace().next().unwrap_or("")).collect();
        assert_eq!(got, want, "{e}");
    }

    #[test]
    fn a_window_over_nothing_costs_no_sort() {
        // `sum(x) OVER ()` is a grand total. Ordering it would be pure waste,
        // and the plan is the only place that shows whether it was paid.
        let e = explain("SELECT sum(ms) OVER () FROM events");
        assert!(!e.contains("Sort"), "an unordered window must not sort:\n{e}");
        assert!(e.contains("Window [sum(ms) OVER ()]"), "{e}");
    }

    #[test]
    fn the_window_sort_is_partition_keys_then_order_keys() {
        let e = explain(
            "SELECT sum(ms) OVER (PARTITION BY url, user_id ORDER BY id DESC) FROM events",
        );
        // Partition keys sort ascending (they only have to group); the ORDER
        // BY keys keep the direction that was written.
        assert!(e.contains("Sort [url#2, user_id#1, id#0 DESC]"), "{e}");
    }

    #[test]
    fn calls_sharing_an_over_clause_share_one_operator_and_one_sort() {
        let e = explain(
            "SELECT row_number() OVER (ORDER BY id), sum(ms) OVER (ORDER BY id) FROM events",
        );
        assert_eq!(e.matches("Window ").count(), 1, "{e}");
        assert_eq!(e.matches("Sort ").count(), 1, "{e}");
        // Different clauses get their own step, because each needs its input
        // ordered differently.
        let e = explain(
            "SELECT row_number() OVER (ORDER BY id), sum(ms) OVER (ORDER BY ms) FROM events",
        );
        assert_eq!(e.matches("Window ").count(), 2, "{e}");
        assert_eq!(e.matches("Sort ").count(), 2, "{e}");
    }

    #[test]
    fn the_same_call_written_twice_is_computed_once() {
        // Same rule as `repeated_aggregate_is_computed_once`, and it has to
        // survive the qualified/bare spelling difference for the same reason.
        let e = explain(
            "SELECT sum(ms) OVER (ORDER BY id), sum(events.ms) OVER (ORDER BY events.id) \
             FROM events",
        );
        assert_eq!(e.matches("Window [").count(), 1, "{e}");
        // One operator column, named after whichever spelling was hoisted
        // first, referenced twice by the projection.
        assert_eq!(e.matches("#2").count(), 2, "{e}");
    }

    #[test]
    fn window_output_types_come_from_the_function_and_the_frame() {
        assert_eq!(
            out("SELECT row_number() OVER () AS r, rank() OVER (ORDER BY id) AS k, \
                 percent_rank() OVER (ORDER BY id) AS p FROM events"),
            vec!["r: UInt64", "k: UInt64", "p: Float64"]
        );
        // An aggregate used as a window keeps exactly the type it has as an
        // aggregate -- same registry entry, same `ret` callback.
        assert_eq!(
            out("SELECT sum(ms) OVER (ORDER BY id) AS s FROM events"),
            out("SELECT sum(ms) AS s FROM events")
        );
        // A frame that always contains its own row cannot be empty, so the
        // aggregate's own type stands...
        assert_eq!(
            out("SELECT min(ms) OVER (ORDER BY id) AS m FROM events"),
            vec!["m: UInt32"]
        );
        // ...and one that can be empty has to admit the NULL it then returns.
        assert_eq!(
            out("SELECT min(ms) OVER (ORDER BY id ROWS BETWEEN 2 FOLLOWING AND 3 FOLLOWING) AS m \
                 FROM events"),
            vec!["m: Nullable(UInt32)"]
        );
        // The positional functions are always nullable: the frame or the
        // partition can run out from under them.
        assert_eq!(
            out("SELECT lag(url) OVER (ORDER BY id) AS l, \
                 first_value(ms) OVER (ORDER BY id) AS f FROM events"),
            vec!["l: Nullable(String)", "f: Nullable(UInt32)"]
        );
    }

    #[test]
    fn a_window_may_read_an_aggregate_of_the_group_it_sits_over() {
        // The window binds against the source scope with aggregates allowed,
        // then takes the same post-aggregate rewrite the select list does --
        // so `sum(ms)` here is the *grouped* sum, computed once.
        let e = explain(
            "SELECT url, sum(ms) AS s, rank() OVER (ORDER BY sum(ms) DESC) FROM events \
             GROUP BY url",
        );
        assert!(e.contains("Aggregate group=[url#0] aggs=[sum(ms#1)]"), "{e}");
        // The window's sort key is the aggregate's output column, not a
        // re-evaluation of `ms`.
        assert!(e.contains("Sort [sum(ms)#1 DESC]"), "{e}");
    }

    #[test]
    fn order_by_and_limit_by_can_name_a_window_column() {
        let e = explain("SELECT url FROM events ORDER BY row_number() OVER (ORDER BY id)");
        // The outer Sort sits above the Window, so its key is a plain column
        // reference into the window's output.
        let lines: Vec<&str> = e.lines().collect();
        assert!(lines[1].trim().starts_with("Sort ["), "{e}");
        assert!(lines[2].trim().starts_with("Window ["), "{e}");
        assert!(explain("SELECT url FROM events ORDER BY 1, rank() OVER (ORDER BY id)")
            .contains("Window ["));
    }

    #[test]
    fn distinct_deduplicates_the_window_output() {
        let e = explain("SELECT DISTINCT ntile(4) OVER (ORDER BY id) FROM events");
        let got: Vec<&str> =
            e.lines().map(|l| l.trim().split_whitespace().next().unwrap_or("")).collect();
        assert_eq!(got, ["Distinct", "Project", "Window", "Sort", "Scan"], "{e}");
    }

    #[test]
    fn a_window_spec_pulls_its_columns_into_the_scan() {
        // The scan is built from a syntactic walk, so a PARTITION BY key that
        // the walk cannot see is a column the operator cannot read. Silent
        // when wrong: the window would bind against a column index the block
        // does not have.
        let e = explain("SELECT id FROM events WHERE ms > 0 ORDER BY 1");
        assert!(e.contains("Scan default.events [id, ms]"), "{e}");
        let e = explain(
            "SELECT sum(amount) OVER (PARTITION BY url ORDER BY ts) FROM events",
        );
        assert!(e.contains("Scan default.events [url, amount, ts]"), "{e}");
    }

    #[test]
    fn windows_are_refused_everywhere_they_cannot_be_computed() {
        for (sql, want) in [
            (
                "SELECT id FROM events WHERE row_number() OVER () = 1",
                "not allowed here",
            ),
            (
                "SELECT url FROM events GROUP BY url HAVING rank() OVER (ORDER BY url) > 1",
                "cannot appear in HAVING",
            ),
            (
                "SELECT sum(row_number() OVER ()) FROM events",
                "cannot appear inside an aggregate",
            ),
            (
                "SELECT row_number() OVER (ORDER BY rank() OVER (ORDER BY id)) FROM events",
                "cannot be nested",
            ),
            ("SELECT count(DISTINCT id) OVER () FROM events", "DISTINCT is not supported"),
            ("SELECT nosuchfn() OVER () FROM events", "unknown window function"),
            ("SELECT row_number(id) OVER () FROM events", "takes exactly 0"),
            ("SELECT lag(id, ms) OVER (ORDER BY id) FROM events", "integer constant"),
            ("SELECT nth_value(id, 0) OVER () FROM events", "integer constant"),
            ("SELECT ntile(0) OVER (ORDER BY id) FROM events", "integer constant"),
        ] {
            let e = err(sql);
            assert!(e.contains(want), "`{sql}`\n  got: {e}");
        }
    }

    #[test]
    fn a_query_with_no_window_gets_exactly_the_plan_it_got_before() {
        // The feature has to be free when unused. Every shape the binder
        // special-cases, checked for an unchanged tree.
        for sql in [
            "SELECT url, count(*) FROM events GROUP BY url HAVING count(*) > 1 ORDER BY 2 LIMIT 5",
            "SELECT DISTINCT url FROM events ORDER BY url",
            "SELECT url FROM events ORDER BY ms LIMIT 3 BY url",
            "SELECT * FROM events e JOIN users u ON e.user_id = u.user_id",
        ] {
            let e = explain(sql);
            assert!(!e.contains("Window"), "`{sql}` grew a window node:\n{e}");
        }
    }
}
