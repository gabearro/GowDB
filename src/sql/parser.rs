//! Recursive-descent parser for the ClickHouse-flavoured dialect.
//!
//! Hand-written rather than generated, for three reasons that all matter here:
//! the crate takes no dependencies; the dialect's irregularities
//! (`LIMIT m, n` with the *offset* first, parametric aggregates, `PREWHERE`,
//! `x::T`) are easier to special-case in code than to bend a grammar around;
//! and hand-written descent is the only cheap way to get error messages that
//! name both the expectation and the byte offset.
//!
//! ## Shape
//!
//! The parser owns a `Vec<Spanned>` and an index. Everything is one-token
//! lookahead except three spots that save the index and restore it
//! ([`Parser::try_qualified_wildcard`], the join-modifier prefix, and the
//! `(subquery)`-vs-`(expr)` decision), because the alternative -- threading a
//! "what kind of thing is this" flag through the expression grammar -- is
//! worse than an occasional rewind over a token list that is already in memory.
//!
//! ## Precedence, lowest to highest
//!
//! ```text
//!   OR
//!   AND
//!   NOT                      (prefix, so `NOT a = b` is `NOT (a = b)`)
//!   = != <> < <= > >=  IS [NOT] NULL  [NOT] LIKE/ILIKE  [NOT] IN  [NOT] BETWEEN
//!   ||                       (concat, above comparison so `a || b = c` groups left)
//!   + -
//!   * / % DIV
//!   unary -                  (folded into the literal when the operand is one)
//!   postfix ::T
//!   primary
//! ```
//!
//! `BETWEEN a AND b` parses both bounds at the `||` level, which is what keeps
//! its `AND` from being captured by the logical `AND` two levels below.
//!
//! ## What the parser deliberately does not do
//!
//! No name resolution, no type checking, no folding beyond negative literals,
//! and no rejection of semantically silly-but-grammatical input (`CROSS JOIN
//! ... ON`, aggregates in `WHERE`). All of that is the binder's job, and
//! keeping it out of here means one place to look when a message is wrong.
//!
//! ## Nesting depth
//!
//! Descent is bounded by [`MAX_DEPTH`], because the alternative is a SIGABRT:
//! the release profile sets `panic = "abort"`, so a blown stack is not a
//! catchable error but the death of the host process, and 2 KB of `(((...`
//! was enough to trigger it. See [`Parser::nest`] for the accounting.

use std::cell::Cell;

use crate::common::{Error, Result};
use crate::sql::ast::{
    BinaryOp, ColumnDef, CreateTable, Cte, ExplainKind, Expr, FrameBound, FrameUnits, Insert,
    InsertSource, IntervalUnit, JoinConstraint, JoinOp, ObjectName, OrderByExpr, Query, Select,
    SelectItem, SetExpr, SetOp, Statement, TableRef, UnaryOp, WindowFrame, WindowSpec,
};
use crate::sql::lexer::{is_reserved, tokenize, Spanned, Token};
use crate::types::{DataType, Engine, Value};

// ----------------------------------------------------------------- public API

/// Parse a script into statements, splitting on `;`.
pub fn parse(sql: &str) -> Result<Vec<Statement>> {
    let depth = Cell::new(0);
    let mut p = Parser::new(sql, &depth)?;
    let mut out = Vec::new();
    loop {
        while p.eat(&Token::Semicolon) {}
        if p.peek().is_none() {
            break;
        }
        out.push(p.statement()?);
        if p.peek().is_none() {
            break;
        }
        if !p.at(&Token::Semicolon) {
            return p.err("`;` between statements");
        }
    }
    Ok(out)
}

/// Parse exactly one statement, rejecting trailing input.
pub fn parse_one(sql: &str) -> Result<Statement> {
    let depth = Cell::new(0);
    let mut p = Parser::new(sql, &depth)?;
    while p.eat(&Token::Semicolon) {}
    if p.peek().is_none() {
        return Err(Error::parse("expected a statement, found empty input", 0));
    }
    let st = p.statement()?;
    while p.eat(&Token::Semicolon) {}
    if p.peek().is_some() {
        return p.err("end of input (this entry point accepts a single statement)");
    }
    Ok(st)
}

/// Parse a bare expression. For tests, tools, and anywhere a fragment of SQL
/// needs to become an [`Expr`] without a surrounding query.
pub fn parse_expr(sql: &str) -> Result<Expr> {
    let depth = Cell::new(0);
    let mut p = Parser::new(sql, &depth)?;
    if p.peek().is_none() {
        return Err(Error::parse("expected an expression, found empty input", 0));
    }
    let e = p.expr()?;
    if p.peek().is_some() {
        return p.err("end of expression");
    }
    Ok(e)
}

// -------------------------------------------------------------- token cursor

/// Deepest chain of guarded descents the parser will follow before giving up.
///
/// Measured, not guessed. With the guard lifted, the shape that buys the least
/// nesting per byte of stack is `SELECT 1 WHERE EXISTS (SELECT 1 WHERE ...`:
/// it aborts at 237 levels under the *test* profile on a libtest worker thread
/// (the smallest stack anything here runs on) and at 881 in the release binary
/// on the main thread. Costing three counts a level, it is the shape this
/// limit binds hardest, and 200 stops it at 65 -- a 3.6x margin on the tightest
/// stack measured and 13x on the one that ships. Every other shape has more:
/// nested parens abort at 467/98 allowed, nested subqueries at 422/97, `NOT`
/// chains at 7709/196.
///
/// The other direction matters just as much: 200 has to be far above anything
/// real. It allows 98 levels of parens or nested calls, 97 of nested subquery,
/// 196 of `NOT`. And the machine-generated input that prompted this -- an ORM
/// expanding a 20k-element `IN` list, or the equivalent `OR` chain -- costs
/// *nothing*, because those loops are iterative; only nesting is counted.
const MAX_DEPTH: u32 = 200;

/// Cap on how long a single left-associative chain may grow.
///
/// [`MAX_DEPTH`] bounds recursive *descent*, but `a OR b OR c ...`, the
/// arithmetic operators, the comparisons and `UNION` are all parsed by a loop.
/// They never touch the descent counter, so `parse` used to return `Ok` with a
/// chain as long as the input and the process died later, somewhere else: the
/// binder rejects anything past `MAX_BIND_DEPTH`, but `Expr` derives `Clone`,
/// `PartialEq` and `Debug`, all three of which recurse once per node. A 25k
/// chain overflowed inside the binder's `expr.clone()` *while producing the
/// error that rejects it*. Manual iterative `Drop` in sql::ast fixed teardown;
/// it cannot fix the derives.
///
/// Bounding the chain here fixes all of them at once, because the deep tree is
/// never built. An `IN` list is a `Vec`, not a chain, so the common "ORM
/// expands IN to 60k terms" shape is unaffected -- verified: a 60,000-element
/// `IN` list parses and runs in 1.6ms.
///
/// This is NOT the limit a user meets first. `MAX_BIND_DEPTH` (200, in
/// planner::binder) rejects a chain of ~200 while binding it, because the
/// binder recurses once per level. Both are load-bearing: the binder's bound
/// protects the binder's own stack, and this one protects every path that does
/// *not* go through the binder -- `EXPLAIN AST`, error formatting, and
/// subquery resolution all touch the derived `Clone`/`Debug` first. Raising
/// the binder's limit for flat chains (whose frames are far cheaper than a
/// subquery's) is worth doing and would make 4096 the real ceiling; until
/// then, expect "nests more than 200" for chains between 200 and 4096.
const MAX_CHAIN: usize = 4096;

struct Parser<'d> {
    toks: Vec<Spanned>,
    i: usize,
    /// Byte offset just past the input, so "unexpected end of input" errors
    /// still point somewhere useful.
    eof: usize,
    /// Live count of guarded frames below us. Borrowed from the caller's stack
    /// rather than owned so that [`Nesting`] can hold onto it without holding a
    /// borrow of the `Parser` -- a guard that borrowed `self` would make
    /// `let _n = self.nest()?; self.or_expr()` fail to compile.
    depth: &'d Cell<u32>,
    /// `WINDOW w AS (...)` declarations in scope, in declaration order so a
    /// later one may name an earlier one as its base. Scoped to one `query`,
    /// which saves and restores it.
    windows: Vec<(String, WindowSpec)>,
    /// Whether the token stream contains the word `WINDOW` anywhere; `None`
    /// until something asks.
    ///
    /// The named-window clause is written *after* HAVING but referenced in the
    /// select list, which is parsed first, so [`Parser::select`] has to look
    /// ahead for it. This flag is what keeps that scan off every other query:
    /// one pass over the tokens, memoized, and false for every statement in the
    /// engine's own test suite bar the window ones.
    ///
    /// Lazy rather than computed in [`Parser::new`] so that a statement with no
    /// `SELECT` in it pays *nothing*. That is not a rounding error: the largest
    /// token vectors this parser ever sees are `INSERT ... VALUES` with tens of
    /// thousands of tuples, and scanning one of those for a keyword that cannot
    /// appear in it would be the single most expensive thing the ingest path
    /// asked the parser to do.
    ///
    /// What the scan costs where it does run, A/B interleaved against the same
    /// build with it forced off (temporary env switch, since removed), the 130-
    /// token analytical query at the bottom of this file's tests, best-of-9 per
    /// side, six rounds: 11.49/12.60/12.43/14.02/12.28/11.73 us off against
    /// 11.72/12.45/13.15/14.30/12.13/11.97 us on -- +2% at the best pair, and
    /// two of the six rounds have it the other way round. The calibration for
    /// that: on the `INSERT` side, where the lazy gate means both configurations
    /// execute *identical* instructions, the same measurement still spread
    /// 12.4-14.8 ms. A 2% difference is not visible on this machine.
    window_kw: Cell<Option<bool>>,
}

/// RAII counter for [`Parser::depth`], decrementing on the way out.
///
/// It has to be RAII. Every guarded entry point returns through `?` in a dozen
/// places, and a hand-written decrement after the recursive call is skipped on
/// each of those paths: the count would ratchet up across the failed parses a
/// session accumulates until legitimate queries started failing, and the drift
/// would only show up in long-lived processes. `panic = "abort"` means there is
/// no unwinding path to worry about beyond that.
struct Nesting<'d>(&'d Cell<u32>);

impl Drop for Nesting<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}

impl<'d> Parser<'d> {
    fn new(sql: &str, depth: &'d Cell<u32>) -> Result<Parser<'d>> {
        Ok(Parser {
            toks: tokenize(sql)?,
            i: 0,
            eof: sql.len(),
            depth,
            windows: Vec::new(),
            window_kw: Cell::new(None),
        })
    }

    /// Charge one level of recursion, held for the caller's scope.
    ///
    /// Guarded: [`Parser::statement`] (EXPLAIN wraps a statement),
    /// [`Parser::query`], [`Parser::table_ref`], [`Parser::expr`],
    /// [`Parser::not_expr`], [`Parser::unary`] and [`Parser::primary`]. That
    /// set covers every back edge in the call graph -- the other recursive
    /// descents (`set_primary`, `table_factor`, `in_rest`, `exists_expr`,
    /// `call_args`, `case_expr`, `ctes`, `value_rows`, `key_list`) all reach
    /// themselves only through one of those seven, so guarding them too would
    /// just spend the budget faster.
    ///
    /// The returned guard borrows the counter, not `self`, so the caller can
    /// keep recursing through `&mut self` while holding it.
    ///
    /// Cost, A/B interleaved on the analytical query at the bottom of this
    /// file's tests: 7.90 us per parse with the guards against 7.78 us without,
    /// four rounds each. ~120 ns on a 55-token statement, once per grammar
    /// frame rather than once per token, and no allocation on the success path
    /// -- the `format!` only runs when the parse is being rejected anyway.
    /// Charge one link of a left-associative chain; see [`MAX_CHAIN`].
    #[inline]
    fn chain(&self, n: usize) -> Result<()> {
        if n > MAX_CHAIN {
            return Err(Error::parse(
                format!(
                    "expression chains more than {MAX_CHAIN} operators; \
                     rewrite it as an IN list or a join"
                ),
                self.pos(),
            ));
        }
        Ok(())
    }

    fn nest(&self) -> Result<Nesting<'d>> {
        let d = self.depth.get() + 1;
        if d > MAX_DEPTH {
            return Err(Error::parse(
                format!("nested more than {MAX_DEPTH} levels deep here"),
                self.pos(),
            ));
        }
        self.depth.set(d);
        Ok(Nesting(self.depth))
    }

    /// Does the word `WINDOW` occur anywhere in this statement? Memoized; see
    /// [`Parser::window_kw`].
    fn has_window_kw(&self) -> bool {
        match self.window_kw.get() {
            Some(b) => b,
            None => {
                let b = self.toks.iter().any(|s| s.tok.is_keyword("WINDOW"));
                self.window_kw.set(Some(b));
                b
            }
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.toks.get(self.i).map(|s| &s.tok)
    }

    fn peek_at(&self, n: usize) -> Option<&Token> {
        self.toks.get(self.i + n).map(|s| &s.tok)
    }

    /// Byte offset of the current token, or of end-of-input.
    fn pos(&self) -> usize {
        self.toks.get(self.i).map(|s| s.pos).unwrap_or(self.eof)
    }

    fn bump(&mut self) {
        self.i += 1;
    }

    fn at(&self, t: &Token) -> bool {
        self.peek() == Some(t)
    }

    fn eat(&mut self, t: &Token) -> bool {
        if self.at(t) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, t: &Token) -> Result<()> {
        if self.eat(t) {
            Ok(())
        } else {
            self.err(&format!("`{t}`"))
        }
    }

    fn at_kw(&self, kw: &str) -> bool {
        self.peek().is_some_and(|t| t.is_keyword(kw))
    }

    fn at_kw_at(&self, n: usize, kw: &str) -> bool {
        self.peek_at(n).is_some_and(|t| t.is_keyword(kw))
    }

    fn eat_kw(&mut self, kw: &str) -> bool {
        if self.at_kw(kw) {
            self.bump();
            true
        } else {
            false
        }
    }

    /// Consume a keyword run all-or-nothing, for optional multi-word clauses
    /// like `IF NOT EXISTS` and `WITH TOTALS`.
    fn eat_kws(&mut self, kws: &[&str]) -> bool {
        if kws.iter().enumerate().all(|(n, k)| self.at_kw_at(n, k)) {
            self.i += kws.len();
            true
        } else {
            false
        }
    }

    fn expect_kw(&mut self, kw: &str) -> Result<()> {
        if self.eat_kw(kw) {
            Ok(())
        } else {
            self.err(&format!("`{}`", kw.to_ascii_uppercase()))
        }
    }

    fn err<T>(&self, expected: &str) -> Result<T> {
        let found = match self.peek() {
            Some(t) => format!("`{t}`"),
            None => "end of input".to_string(),
        };
        Err(Error::parse(format!("expected {expected}, found {found}"), self.pos()))
    }

    /// Any word, quoted or not. Callers that need to reject keywords use
    /// [`Parser::opt_alias`] instead.
    fn ident(&mut self) -> Result<String> {
        match self.peek() {
            Some(Token::Word { value, .. }) => {
                let v = value.clone();
                self.bump();
                Ok(v)
            }
            _ => self.err("an identifier"),
        }
    }

    /// `a`, `t.a`, `db.t.a`. Stops before `.*` so `t.*` stays parseable.
    fn object_name(&mut self) -> Result<ObjectName> {
        let mut parts = vec![self.ident()?];
        while self.at(&Token::Dot) && !matches!(self.peek_at(1), Some(Token::Star)) {
            self.bump();
            parts.push(self.ident()?);
        }
        Ok(ObjectName(parts))
    }

    fn ident_list(&mut self) -> Result<Vec<String>> {
        let paren = self.eat(&Token::LParen);
        let mut v = vec![self.ident()?];
        while self.eat(&Token::Comma) {
            v.push(self.ident()?);
        }
        if paren {
            self.expect(&Token::RParen)?;
        }
        Ok(v)
    }
}

// ------------------------------------------------------------- statements

impl Parser<'_> {
    fn statement(&mut self) -> Result<Statement> {
        // `EXPLAIN` takes a statement, so this is a recursive entry point.
        let _nest = self.nest()?;
        if self.at(&Token::LParen)
            || self.at_kw("SELECT")
            || self.at_kw("WITH")
            || self.at_kw("VALUES")
        {
            return Ok(Statement::Query(Box::new(self.query()?)));
        }
        let kw = match self.peek().and_then(|t| t.bare_word()) {
            Some(w) => w.to_ascii_uppercase(),
            None => return self.err("a statement"),
        };
        match kw.as_str() {
            "INSERT" => self.insert(),
            "DELETE" => self.delete_from(),
            "UPDATE" => self.update(),
            "CREATE" => self.create(),
            "DROP" => self.drop_object(),
            "ALTER" => self.alter(),
            "OPTIMIZE" => {
                self.bump();
                self.expect_kw("TABLE")?;
                let table = self.object_name()?;
                let final_ = self.eat_kw("FINAL");
                Ok(Statement::Optimize { table, final_ })
            }
            "TRUNCATE" => {
                self.bump();
                self.eat_kw("TABLE");
                Ok(Statement::Truncate { table: self.object_name()? })
            }
            "SHOW" => self.show(),
            "DESCRIBE" | "DESC" => {
                self.bump();
                self.eat_kw("TABLE");
                Ok(Statement::Describe(self.object_name()?))
            }
            "EXPLAIN" => {
                self.bump();
                let kind = if self.eat_kw("PLAN") {
                    ExplainKind::Plan
                } else if self.eat_kw("PIPELINE") {
                    ExplainKind::Pipeline
                } else if self.eat_kw("AST") || self.eat_kw("SYNTAX") {
                    ExplainKind::Ast
                } else {
                    ExplainKind::Plan
                };
                Ok(Statement::Explain { kind, statement: Box::new(self.statement()?) })
            }
            "USE" => {
                self.bump();
                Ok(Statement::Use(self.ident()?))
            }
            "SYSTEM" => {
                self.bump();
                self.expect_kw("FLUSH")?;
                // `SYSTEM FLUSH` with no table flushes every memtable.
                let target = match self.peek() {
                    Some(Token::Word { .. }) => Some(self.object_name()?),
                    _ => None,
                };
                Ok(Statement::SystemFlush(target))
            }
            _ => self.err(
                "a statement (SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, DROP, ...)",
            ),
        }
    }

    // ------------------------------------------------------------ mutations
    //
    // `DELETE FROM t WHERE p` and `ALTER TABLE t DELETE WHERE p` are the same
    // statement spelled two ways, and they build the same AST node -- there is
    // one execution path, not two, which is the whole point of routing the
    // ClickHouse spelling through here rather than keeping it beside a second
    // implementation. Same for UPDATE. The productions below are shared by both
    // entry points so the two spellings cannot drift in what they accept.

    /// `DELETE FROM t [WHERE p]`.
    fn delete_from(&mut self) -> Result<Statement> {
        self.expect_kw("DELETE")?;
        self.expect_kw("FROM")?;
        let table = self.object_name()?;
        Ok(Statement::AlterDelete { table, predicate: self.mutation_where()? })
    }

    /// `UPDATE t SET c = e, ... [WHERE p]`.
    ///
    /// `SET` is not in the lexer's reserved list, and does not need to be: it
    /// is only ever looked for immediately after a table name, where an
    /// identifier cannot follow anyway.
    fn update(&mut self) -> Result<Statement> {
        self.expect_kw("UPDATE")?;
        let table = self.object_name()?;
        self.expect_kw("SET")?;
        let assignments = self.assignments()?;
        Ok(Statement::AlterUpdate { table, assignments, predicate: self.mutation_where()? })
    }

    fn assignments(&mut self) -> Result<Vec<(String, Expr)>> {
        let mut out = Vec::new();
        loop {
            let col = self.ident()?;
            self.expect(&Token::Eq)?;
            out.push((col, self.expr()?));
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        Ok(out)
    }

    /// A mutation's trailing `WHERE`, defaulted to "every row".
    ///
    /// `DELETE FROM t` with no predicate is legal SQL and means all of it, so
    /// the missing clause becomes a literal `true` rather than an `Option` the
    /// binder would have to special-case: the optimizer's existing constant
    /// folding erases a `Filter true` on its own, so the unconditional form
    /// costs nothing at plan time and the mutation binder stays one path.
    fn mutation_where(&mut self) -> Result<Expr> {
        if self.eat_kw("WHERE") {
            self.expr()
        } else {
            Ok(Expr::Literal(Value::Bool(true)))
        }
    }

    /// The `ALTER TABLE ... DELETE|UPDATE` spelling's mandatory `WHERE`.
    fn require_mutation_where(&self, what: &str) -> Result<()> {
        if self.at_kw("WHERE") {
            return Ok(());
        }
        self.err(&format!(
            "`WHERE` (ALTER TABLE ... {what} requires one; write `{}` to affect every row)",
            if what == "DELETE" { "DELETE FROM t" } else { "UPDATE t SET ..." }
        ))
    }

    fn insert(&mut self) -> Result<Statement> {
        self.expect_kw("INSERT")?;
        self.expect_kw("INTO")?;
        self.eat_kw("TABLE");
        let table = self.object_name()?;

        // `(a, b)` is a column list; `(SELECT ...)` is already the source.
        let mut columns = Vec::new();
        if self.at(&Token::LParen)
            && !self.at_kw_at(1, "SELECT")
            && !self.at_kw_at(1, "WITH")
        {
            columns = self.ident_list()?;
        }

        let source = if self.eat_kw("VALUES") {
            InsertSource::Values(self.value_rows()?)
        } else if self.at_kw("SELECT") || self.at_kw("WITH") || self.at(&Token::LParen) {
            InsertSource::Query(Box::new(self.query()?))
        } else if self.at_kw("FORMAT") {
            return Err(Error::unsupported("INSERT ... FORMAT"));
        } else {
            return self.err("`VALUES` or a SELECT");
        };
        Ok(Statement::Insert(Insert { table, columns, source }))
    }

    fn value_rows(&mut self) -> Result<Vec<Vec<Expr>>> {
        let mut rows = Vec::new();
        loop {
            self.expect(&Token::LParen)?;
            let mut row = Vec::new();
            if !self.at(&Token::RParen) {
                row.push(self.expr()?);
                while self.eat(&Token::Comma) {
                    row.push(self.expr()?);
                }
            }
            self.expect(&Token::RParen)?;
            rows.push(row);
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        Ok(rows)
    }

    fn create(&mut self) -> Result<Statement> {
        self.expect_kw("CREATE")?;
        if self.eat_kw("DATABASE") {
            let if_not_exists = self.eat_kws(&["IF", "NOT", "EXISTS"]);
            return Ok(Statement::CreateDatabase { name: self.ident()?, if_not_exists });
        }
        if self.eat_kw("TABLE") {
            return self.create_table();
        }
        self.err("`TABLE` or `DATABASE` after CREATE")
    }

    fn create_table(&mut self) -> Result<Statement> {
        let if_not_exists = self.eat_kws(&["IF", "NOT", "EXISTS"]);
        let name = self.object_name()?;

        let mut columns = Vec::new();
        if self.eat(&Token::LParen) {
            loop {
                columns.push(self.column_def()?);
                if !self.eat(&Token::Comma) {
                    break;
                }
            }
            self.expect(&Token::RParen)?;
        }

        let mut ct = CreateTable {
            name,
            if_not_exists,
            columns,
            engine: Engine::default(),
            order_by: Vec::new(),
            primary_key: Vec::new(),
            partition_by: None,
            as_query: None,
        };

        // Table-level clauses. ClickHouse fixes their order; we accept any, so
        // a transposed ORDER BY / PARTITION BY is not a syntax error.
        loop {
            if self.eat_kw("ENGINE") {
                self.expect(&Token::Eq)?;
                let name = self.ident()?;
                if self.at(&Token::LParen) {
                    // Engine arguments (`ReplacingMergeTree(ver)`) are parsed
                    // and dropped: `Engine` is a plain enum with nowhere to
                    // keep them.
                    self.paren_text()?;
                }
                // The name table lives in `Engine::parse`, and its error kind
                // (`Unsupported`) is the honest one: the syntax was fine, the
                // engine just is not built. It passes through unchanged.
                ct.engine = Engine::parse(&name)?;
            } else if self.at_kw("ORDER") {
                self.bump();
                self.expect_kw("BY")?;
                ct.order_by = self.key_list()?;
            } else if self.at_kw("PRIMARY") {
                self.bump();
                self.expect_kw("KEY")?;
                ct.primary_key = self.key_list()?;
            } else if self.at_kw("PARTITION") {
                self.bump();
                self.expect_kw("BY")?;
                ct.partition_by = Some(self.expr()?);
            } else if self.at_kw("SAMPLE") {
                self.bump();
                self.expect_kw("BY")?;
                self.expr()?; // accepted, ignored: no sampling in this engine
            } else if self.eat_kw("TTL") {
                self.expr()?; // ditto
            } else if self.eat_kw("SETTINGS") {
                self.skip_settings()?;
            } else if self.eat_kw("AS") {
                if self.at_kw("SELECT") || self.at_kw("WITH") || self.at(&Token::LParen) {
                    ct.as_query = Some(Box::new(self.query()?));
                } else {
                    return Err(Error::unsupported("CREATE TABLE ... AS <other table>"));
                }
            } else {
                break;
            }
        }

        if ct.columns.is_empty() && ct.as_query.is_none() {
            return self.err("a column list or `AS SELECT`");
        }
        Ok(Statement::CreateTable(Box::new(ct)))
    }

    fn column_def(&mut self) -> Result<ColumnDef> {
        let name = self.ident()?;
        let mut ty = self.data_type()?;
        let mut default = None;
        let mut codec = None;
        loop {
            if self.eat_kws(&["NOT", "NULL"]) {
                // no-op: types are non-nullable unless wrapped
            } else if self.eat_kw("NULL") {
                ty = ty.to_nullable();
            } else if self.eat_kw("DEFAULT") {
                default = Some(self.expr()?);
            } else if self.eat_kw("CODEC") {
                // Stored as source text; the storage layer maps it to a codec
                // chain, and unknown chains must survive parsing to be
                // diagnosed there with the column name in hand.
                let text = self.paren_text()?;
                codec = Some(text[1..text.len() - 1].to_string());
            } else if self.eat_kw("COMMENT") {
                match self.peek() {
                    Some(Token::Str(_)) => self.bump(),
                    _ => return self.err("a string after COMMENT"),
                }
            } else if self.eat_kw("TTL") {
                self.expr()?;
            } else {
                break;
            }
        }
        Ok(ColumnDef { name, ty, default, codec })
    }

    fn drop_object(&mut self) -> Result<Statement> {
        self.expect_kw("DROP")?;
        if self.eat_kw("TABLE") {
            let if_exists = self.eat_kws(&["IF", "EXISTS"]);
            return Ok(Statement::DropTable { name: self.object_name()?, if_exists });
        }
        if self.eat_kw("DATABASE") {
            let if_exists = self.eat_kws(&["IF", "EXISTS"]);
            return Ok(Statement::DropDatabase { name: self.ident()?, if_exists });
        }
        self.err("`TABLE` or `DATABASE` after DROP")
    }

    fn alter(&mut self) -> Result<Statement> {
        self.expect_kw("ALTER")?;
        self.expect_kw("TABLE")?;
        let table = self.object_name()?;

        // Same node, same productions, same execution as the `DELETE FROM` /
        // `UPDATE ... SET` spellings above. The one difference is the `WHERE`:
        // ClickHouse makes it mandatory on this spelling, and a bare
        // `ALTER TABLE t DELETE` is far likelier to be a truncated edit than a
        // request to empty the table -- so the ANSI form is where "no predicate
        // means all rows" is allowed to mean that.
        if self.eat_kw("DELETE") {
            self.require_mutation_where("DELETE")?;
            return Ok(Statement::AlterDelete { table, predicate: self.mutation_where()? });
        }
        if self.eat_kw("UPDATE") {
            let assignments = self.assignments()?;
            self.require_mutation_where("UPDATE")?;
            return Ok(Statement::AlterUpdate {
                table,
                assignments,
                predicate: self.mutation_where()?,
            });
        }
        if self.eat_kw("ADD") {
            self.expect_kw("COLUMN")?;
            let if_not_exists = self.eat_kws(&["IF", "NOT", "EXISTS"]);
            return Ok(Statement::AlterAddColumn {
                table,
                column: self.column_def()?,
                if_not_exists,
            });
        }
        if self.eat_kw("DROP") {
            self.expect_kw("COLUMN")?;
            let if_exists = self.eat_kws(&["IF", "EXISTS"]);
            return Ok(Statement::AlterDropColumn { table, column: self.ident()?, if_exists });
        }
        self.err("`DELETE`, `UPDATE`, `ADD COLUMN` or `DROP COLUMN`")
    }

    fn show(&mut self) -> Result<Statement> {
        self.expect_kw("SHOW")?;
        if self.eat_kw("TABLES") {
            let database = if self.eat_kw("FROM") || self.eat_kw("IN") {
                Some(self.ident()?)
            } else {
                None
            };
            return Ok(Statement::ShowTables { database });
        }
        if self.eat_kw("DATABASES") {
            return Ok(Statement::ShowDatabases);
        }
        if self.eat_kw("CREATE") {
            self.eat_kw("TABLE");
            return Ok(Statement::ShowCreateTable(self.object_name()?));
        }
        self.err("`TABLES`, `DATABASES` or `CREATE TABLE`")
    }

    /// `SETTINGS k = v, ...`. Parsed for syntax only: nothing in this engine
    /// reads a setting yet, and silently dropping them beats rejecting queries
    /// copied out of a ClickHouse console.
    fn skip_settings(&mut self) -> Result<()> {
        loop {
            self.ident()?;
            self.expect(&Token::Eq)?;
            self.eat(&Token::Minus);
            match self.peek() {
                Some(Token::Number(_)) | Some(Token::Str(_)) | Some(Token::Word { .. }) => {
                    self.bump()
                }
                _ => return self.err("a setting value"),
            }
            if !self.eat(&Token::Comma) {
                return Ok(());
            }
        }
    }
}

// ------------------------------------------------------------------ queries

impl Parser<'_> {
    fn query(&mut self) -> Result<Query> {
        // The one door every subquery goes through: CTEs, `(SELECT ...)` in a
        // FROM, IN, EXISTS or an expression, and parenthesized set operations.
        let _nest = self.nest()?;
        // A `WINDOW` declaration is scoped to its own query: an inner SELECT
        // must not see the outer one's names, and the outer one must get them
        // back when the inner finishes -- including on the error path, which is
        // why the restore is unconditional rather than at the end of the body.
        // `mem::take` of an empty `Vec` allocates nothing, so a query with no
        // named windows pays a pointer swap.
        let outer = std::mem::take(&mut self.windows);
        let r = self.query_body();
        self.windows = outer;
        r
    }

    fn query_body(&mut self) -> Result<Query> {
        let with = if self.at_kw("WITH") { self.ctes()? } else { Vec::new() };
        let body = self.set_expr()?;
        let mut q =
            Query { with, body, order_by: Vec::new(), limit: None, offset: None, limit_by: None };
        self.query_tail(&mut q)?;
        Ok(q)
    }

    fn ctes(&mut self) -> Result<Vec<Cte>> {
        self.expect_kw("WITH")?;
        let mut out = Vec::new();
        loop {
            let name = self.ident()?;
            // Only the `name AS (query)` form; ClickHouse's scalar
            // `WITH <expr> AS name` is a different shape the AST has no slot
            // for, so it is rejected here rather than silently mangled.
            self.expect_kw("AS")?;
            self.expect(&Token::LParen)?;
            let q = self.query()?;
            self.expect(&Token::RParen)?;
            out.push(Cte { name, query: Box::new(q) });
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        Ok(out)
    }

    /// `ORDER BY` / `LIMIT` / `LIMIT ... BY` / `OFFSET` / `SETTINGS`, all of
    /// which bind to the whole query rather than to the last SELECT of a union.
    fn query_tail(&mut self, q: &mut Query) -> Result<()> {
        if self.at_kw("ORDER") {
            self.bump();
            self.expect_kw("BY")?;
            q.order_by = self.order_by_list()?;
        }
        loop {
            if self.eat_kw("LIMIT") {
                let first = self.expr()?;
                if self.eat(&Token::Comma) {
                    // ClickHouse's reversed form: `LIMIT offset, count`.
                    q.offset = Some(first);
                    q.limit = Some(self.expr()?);
                } else if self.eat_kw("OFFSET") {
                    q.limit = Some(first);
                    q.offset = Some(self.expr()?);
                } else if self.eat_kw("BY") {
                    q.limit_by = Some((first, self.key_list()?));
                } else {
                    q.limit = Some(first);
                }
            } else if self.eat_kw("OFFSET") {
                q.offset = Some(self.expr()?);
            } else if self.eat_kw("SETTINGS") {
                self.skip_settings()?;
            } else {
                return Ok(());
            }
        }
    }

    fn order_by_list(&mut self) -> Result<Vec<OrderByExpr>> {
        let mut out = Vec::new();
        loop {
            let expr = self.expr()?;
            let asc = if self.eat_kw("DESC") {
                false
            } else {
                self.eat_kw("ASC");
                true
            };
            let nulls_first = if self.eat_kw("NULLS") {
                if self.eat_kw("FIRST") {
                    Some(true)
                } else if self.eat_kw("LAST") {
                    Some(false)
                } else {
                    return self.err("`FIRST` or `LAST` after NULLS");
                }
            } else {
                None
            };
            out.push(OrderByExpr { expr, asc, nulls_first });
            if !self.eat(&Token::Comma) {
                return Ok(out);
            }
        }
    }

    /// UNION / EXCEPT, left-associative and below INTERSECT.
    fn set_expr(&mut self) -> Result<SetExpr> {
        let mut left = self.set_term()?;
        let mut n = 0usize;
        loop {
            n += 1;
            self.chain(n)?;
            let op = if self.at_kw("UNION") {
                SetOp::Union
            } else if self.at_kw("EXCEPT") {
                SetOp::Except
            } else {
                return Ok(left);
            };
            self.bump();
            // ClickHouse insists on ALL/DISTINCT after UNION; a bare UNION is
            // accepted here and read as DISTINCT, the ANSI default.
            let all = if self.eat_kw("ALL") {
                true
            } else {
                self.eat_kw("DISTINCT");
                false
            };
            let right = self.set_term()?;
            left = SetExpr::SetOperation {
                op,
                all,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
    }

    fn set_term(&mut self) -> Result<SetExpr> {
        let mut left = self.set_primary()?;
        while self.at_kw("INTERSECT") {
            self.bump();
            let all = if self.eat_kw("ALL") {
                true
            } else {
                self.eat_kw("DISTINCT");
                false
            };
            let right = self.set_primary()?;
            left = SetExpr::SetOperation {
                op: SetOp::Intersect,
                all,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn set_primary(&mut self) -> Result<SetExpr> {
        if self.at_kw("SELECT") {
            return Ok(SetExpr::Select(Box::new(self.select()?)));
        }
        if self.eat_kw("VALUES") {
            return Ok(SetExpr::Values(self.value_rows()?));
        }
        if self.at_kw("WITH") {
            return Ok(SetExpr::Query(Box::new(self.query()?)));
        }
        if self.eat(&Token::LParen) {
            let q = self.query()?;
            self.expect(&Token::RParen)?;
            return Ok(SetExpr::Query(Box::new(q)));
        }
        self.err("`SELECT`, `VALUES` or `(`")
    }

    fn select(&mut self) -> Result<Select> {
        self.expect_kw("SELECT")?;
        let distinct = if self.eat_kw("DISTINCT") {
            true
        } else {
            self.eat_kw("ALL"); // the explicit-but-meaningless ANSI spelling
            false
        };

        // `WINDOW w AS (...)` is written after HAVING and referenced in the
        // select list, so it has to be parsed out of order. The alternative --
        // leaving every `OVER w` unresolved and substituting in a second pass
        // -- needs a mutable walk over the whole expression tree, which is one
        // more recursion over a structure that can be loop-grown to 4096 terms.
        // Reading the clause early costs one linear token scan and leaves the
        // AST with no unresolved names in it at all.
        let win = if self.has_window_kw() { self.window_clause_ahead()? } else { None };

        let mut projection = vec![self.select_item()?];
        while self.eat(&Token::Comma) {
            projection.push(self.select_item()?);
        }

        let from = if self.eat_kw("FROM") { Some(self.table_ref()?) } else { None };
        let prewhere = if self.eat_kw("PREWHERE") { Some(self.expr()?) } else { None };
        let selection = if self.eat_kw("WHERE") { Some(self.expr()?) } else { None };

        let mut group_by = Vec::new();
        let mut with_totals = false;
        if self.at_kw("GROUP") {
            self.bump();
            self.expect_kw("BY")?;
            group_by.push(self.expr()?);
            while self.eat(&Token::Comma) {
                group_by.push(self.expr()?);
            }
            with_totals = self.eat_kws(&["WITH", "TOTALS"]);
        }
        let having = if self.eat_kw("HAVING") { Some(self.expr()?) } else { None };
        // `WITH TOTALS` is legal on either side of HAVING.
        if !with_totals {
            with_totals = self.eat_kws(&["WITH", "TOTALS"]);
        }

        // Step over the clause already consumed above. Guarded on the cursor
        // actually having arrived where the scan found it: a `WINDOW` written
        // somewhere it does not belong then falls through to the ordinary
        // "unexpected token" path instead of silently skipping tokens.
        if let Some((start, end)) = win {
            if self.i == start {
                self.i = end;
            }
        }

        Ok(Select {
            distinct,
            projection,
            from,
            prewhere,
            selection,
            group_by,
            with_totals,
            having,
        })
    }

    fn select_item(&mut self) -> Result<SelectItem> {
        if self.at(&Token::Star) {
            self.bump();
            return Ok(SelectItem::Wildcard);
        }
        if let Some(q) = self.try_qualified_wildcard() {
            return Ok(SelectItem::QualifiedWildcard(q));
        }
        let expr = self.expr()?;
        let alias = self.opt_alias()?;
        Ok(SelectItem::Expr { expr, alias })
    }

    /// `t.*` / `db.t.*`. Needs arbitrary lookahead (the qualifier can be
    /// dotted), so it saves and restores the cursor instead of guessing.
    fn try_qualified_wildcard(&mut self) -> Option<String> {
        let save = self.i;
        let mut parts: Vec<String> = Vec::new();
        loop {
            match self.peek() {
                Some(Token::Word { value, .. }) => parts.push(value.clone()),
                _ => {
                    self.i = save;
                    return None;
                }
            }
            self.bump();
            if !self.eat(&Token::Dot) {
                self.i = save;
                return None;
            }
            if self.eat(&Token::Star) {
                return Some(parts.join("."));
            }
        }
    }

    /// `AS name`, or a bare name that is not a clause keyword.
    fn opt_alias(&mut self) -> Result<Option<String>> {
        if self.eat_kw("AS") {
            return Ok(Some(self.ident()?));
        }
        // `FROM t WINDOW w AS (...)` must not alias the table `WINDOW`.
        // `WINDOW` is deliberately not added to the lexer's reserved list --
        // ClickHouse has no such clause and a column called `window` is
        // perfectly ordinary -- so the clause is recognized by its own shape
        // instead, exactly as [`Parser::window_clause_ahead`] recognizes it.
        if self.at_window_clause(0) {
            return Ok(None);
        }
        match self.peek() {
            Some(Token::Word { value, quoted }) if *quoted || !is_reserved(value) => {
                let v = value.clone();
                self.bump();
                Ok(Some(v))
            }
            _ => Ok(None),
        }
    }

    /// Is the token `n` ahead the start of a `WINDOW <name> AS (` clause?
    ///
    /// Three tokens of lookahead, and no other production can spell them: a
    /// column named `window` is followed by an operator, a comma or a clause
    /// keyword, never by `<ident> AS`.
    fn at_window_clause(&self, n: usize) -> bool {
        self.at_kw_at(n, "WINDOW")
            && matches!(self.peek_at(n + 1), Some(Token::Word { .. }))
            && self.at_kw_at(n + 2, "AS")
    }
}

// ------------------------------------------------------------- table refs

impl Parser<'_> {
    fn table_ref(&mut self) -> Result<TableRef> {
        // `table_factor` descends back into here for a parenthesized join.
        let _nest = self.nest()?;
        let mut left = self.table_factor()?;
        loop {
            if self.eat(&Token::Comma) {
                let right = self.table_factor()?;
                left = TableRef::Join {
                    left: Box::new(left),
                    right: Box::new(right),
                    op: JoinOp::Cross,
                    constraint: JoinConstraint::None,
                };
                continue;
            }

            // Strictness/locality modifiers are recognized so ClickHouse SQL
            // parses, then dropped: this engine has one join implementation.
            let save = self.i;
            while self.at_kw("GLOBAL")
                || self.at_kw("ANY")
                || self.at_kw("ALL")
                || self.at_kw("ASOF")
                || self.at_kw("SEMI")
                || self.at_kw("ANTI")
            {
                self.bump();
            }
            let op = if self.eat_kw("INNER") {
                JoinOp::Inner
            } else if self.eat_kw("LEFT") {
                self.eat_kw("OUTER");
                JoinOp::Left
            } else if self.eat_kw("RIGHT") {
                self.eat_kw("OUTER");
                JoinOp::Right
            } else if self.eat_kw("FULL") {
                self.eat_kw("OUTER");
                JoinOp::Full
            } else if self.eat_kw("CROSS") {
                JoinOp::Cross
            } else if self.at_kw("JOIN") {
                JoinOp::Inner
            } else {
                self.i = save;
                return Ok(left);
            };
            self.expect_kw("JOIN")?;

            let right = self.table_factor()?;
            let constraint = if self.eat_kw("ON") {
                JoinConstraint::On(self.expr()?)
            } else if self.eat_kw("USING") {
                JoinConstraint::Using(self.ident_list()?)
            } else {
                JoinConstraint::None
            };
            left = TableRef::Join {
                left: Box::new(left),
                right: Box::new(right),
                op,
                constraint,
            };
        }
    }

    fn table_factor(&mut self) -> Result<TableRef> {
        if self.at(&Token::LParen) {
            let is_query = self.at_kw_at(1, "SELECT")
                || self.at_kw_at(1, "WITH")
                || self.at_kw_at(1, "VALUES");
            self.bump();
            if is_query {
                let q = self.query()?;
                self.expect(&Token::RParen)?;
                let alias = self.opt_alias()?;
                return Ok(TableRef::Subquery { query: Box::new(q), alias });
            }
            // A parenthesized join: the grouping is already in the tree shape,
            // so the parens leave no trace.
            let inner = self.table_ref()?;
            self.expect(&Token::RParen)?;
            return Ok(inner);
        }
        let name = self.object_name()?;
        // ClickHouse writes FINAL after the table, but after an alias is also
        // in the wild; accept both rather than make the user guess.
        let mut final_ = self.eat_kw("FINAL");
        let alias = self.opt_alias()?;
        if !final_ {
            final_ = self.eat_kw("FINAL");
        }
        Ok(TableRef::Table { name, alias, final_ })
    }
}

// ------------------------------------------------------------- expressions

impl Parser<'_> {
    fn expr(&mut self) -> Result<Expr> {
        // Everything that re-enters the expression grammar -- parens, tuples,
        // call arguments, CASE arms, IN lists -- comes back through here.
        let _nest = self.nest()?;
        self.or_expr()
    }

    fn or_expr(&mut self) -> Result<Expr> {
        let mut e = self.and_expr()?;
        let mut n = 0usize;
        while self.eat_kw("OR") {
            n += 1;
            self.chain(n)?;
            let r = self.and_expr()?;
            e = Expr::binary(e, BinaryOp::Or, r);
        }
        Ok(e)
    }

    fn and_expr(&mut self) -> Result<Expr> {
        let mut e = self.not_expr()?;
        let mut n = 0usize;
        while self.eat_kw("AND") {
            n += 1;
            self.chain(n)?;
            let r = self.not_expr()?;
            e = Expr::binary(e, BinaryOp::And, r);
        }
        Ok(e)
    }

    fn not_expr(&mut self) -> Result<Expr> {
        if self.at_kw("NOT") {
            // Prefix `NOT` stacks on itself without passing back through
            // `expr`, so it needs its own count -- charged inside the branch,
            // because an expression containing no `NOT` recurses no further
            // here and should not spend any of the budget.
            let _nest = self.nest()?;
            self.bump();
            // `NOT EXISTS (...)` folds into one node so the planner sees a
            // single anti-semi-join rather than a negation of a subquery.
            if self.at_kw("EXISTS") {
                return self.exists_expr(true);
            }
            let e = self.not_expr()?;
            return Ok(Expr::UnaryOp { op: UnaryOp::Not, expr: Box::new(e) });
        }
        self.cmp_expr()
    }

    fn cmp_expr(&mut self) -> Result<Expr> {
        let mut e = self.concat_expr()?;
        let mut n = 0usize;
        loop {
            if let Some(op) = self.peek().and_then(cmp_op) {
                self.bump();
                n += 1;
                self.chain(n)?;
                let r = self.concat_expr()?;
                e = Expr::binary(e, op, r);
                continue;
            }
            if self.at_kw("IS") {
                self.bump();
                let negated = self.eat_kw("NOT");
                self.expect_kw("NULL")?;
                e = Expr::IsNull { expr: Box::new(e), negated };
                continue;
            }
            // A `NOT` here can only be the postfix-negating kind: prefix NOT
            // was consumed a level up, before the operand existed.
            let negated = if self.at_kw("NOT")
                && (self.at_kw_at(1, "IN")
                    || self.at_kw_at(1, "LIKE")
                    || self.at_kw_at(1, "ILIKE")
                    || self.at_kw_at(1, "BETWEEN"))
            {
                self.bump();
                true
            } else {
                false
            };
            if self.eat_kw("IN") {
                e = self.in_rest(e, negated)?;
                continue;
            }
            if self.at_kw("LIKE") || self.at_kw("ILIKE") {
                let case_insensitive = self.at_kw("ILIKE");
                self.bump();
                let pattern = self.concat_expr()?;
                e = Expr::Like {
                    expr: Box::new(e),
                    pattern: Box::new(pattern),
                    negated,
                    case_insensitive,
                };
                continue;
            }
            if self.eat_kw("BETWEEN") {
                // Bounds parse above the logical operators, so the `AND` below
                // belongs to BETWEEN and never to a surrounding conjunction.
                let low = self.concat_expr()?;
                self.expect_kw("AND")?;
                let high = self.concat_expr()?;
                e = Expr::Between {
                    expr: Box::new(e),
                    low: Box::new(low),
                    high: Box::new(high),
                    negated,
                };
                continue;
            }
            if negated {
                return self.err("`IN`, `LIKE`, `ILIKE` or `BETWEEN` after NOT");
            }
            return Ok(e);
        }
    }

    fn in_rest(&mut self, e: Expr, negated: bool) -> Result<Expr> {
        self.expect(&Token::LParen)?;
        if self.at_kw("SELECT") || self.at_kw("WITH") {
            let q = self.query()?;
            self.expect(&Token::RParen)?;
            return Ok(Expr::InSubquery {
                expr: Box::new(e),
                subquery: Box::new(q),
                negated,
            });
        }
        let mut list = Vec::new();
        if !self.at(&Token::RParen) {
            list.push(self.expr()?);
            while self.eat(&Token::Comma) {
                list.push(self.expr()?);
            }
        }
        self.expect(&Token::RParen)?;
        Ok(Expr::InList { expr: Box::new(e), list, negated })
    }

    fn concat_expr(&mut self) -> Result<Expr> {
        let mut e = self.additive()?;
        let mut n = 0usize;
        while self.eat(&Token::Concat) {
            n += 1;
            self.chain(n)?;
            let r = self.additive()?;
            e = Expr::binary(e, BinaryOp::Concat, r);
        }
        Ok(e)
    }

    fn additive(&mut self) -> Result<Expr> {
        let mut e = self.multiplicative()?;
        let mut n = 0usize;
        loop {
            let op = if self.eat(&Token::Plus) {
                BinaryOp::Plus
            } else if self.eat(&Token::Minus) {
                BinaryOp::Minus
            } else {
                return Ok(e);
            };
            n += 1;
            self.chain(n)?;
            let r = self.multiplicative()?;
            e = Expr::binary(e, op, r);
        }
    }

    fn multiplicative(&mut self) -> Result<Expr> {
        let mut e = self.unary()?;
        let mut n = 0usize;
        loop {
            let op = if self.eat(&Token::Star) {
                BinaryOp::Multiply
            } else if self.eat(&Token::Slash) {
                BinaryOp::Divide
            } else if self.eat(&Token::Percent) {
                BinaryOp::Modulo
            } else if self.eat_kw("DIV") {
                BinaryOp::IntDiv
            } else {
                return Ok(e);
            };
            n += 1;
            self.chain(n)?;
            let r = self.unary()?;
            e = Expr::binary(e, op, r);
        }
    }

    fn unary(&mut self) -> Result<Expr> {
        if self.at(&Token::Minus) {
            // Same shape as `NOT`: `- - -x` recurses here directly, and the
            // count is charged only on the branch that actually recurses.
            let _nest = self.nest()?;
            self.bump();
            // Fold `-<literal>` into the literal: INSERT VALUES, zone-map
            // pruning and range checks all want a `Value`, not a node to
            // evaluate, and `-1` is far too common to leave folded away.
            if let Some(Token::Number(v)) = self.peek() {
                if let Some(neg) = negate_literal(v) {
                    self.bump();
                    return Ok(Expr::Literal(neg));
                }
            }
            let e = self.unary()?;
            return Ok(Expr::UnaryOp { op: UnaryOp::Neg, expr: Box::new(e) });
        }
        if self.at(&Token::Plus) {
            // Unary plus is identity; drop it rather than model it. It still
            // recurses, though, so `+++...` still has to be counted.
            let _nest = self.nest()?;
            self.bump();
            return self.unary();
        }
        self.postfix()
    }

    fn postfix(&mut self) -> Result<Expr> {
        let mut e = self.primary()?;
        while self.eat(&Token::DoubleColon) {
            let ty = self.data_type()?;
            e = Expr::Cast { expr: Box::new(e), ty };
        }
        Ok(e)
    }

    fn primary(&mut self) -> Result<Expr> {
        // The bottom of the expression ladder and the top of the next one:
        // `(`, a call's arguments, CASE, CAST and EXISTS all restart the
        // grammar from here.
        let _nest = self.nest()?;
        match self.peek() {
            None => self.err("an expression"),
            Some(Token::Number(v)) => {
                let v = v.clone();
                self.bump();
                Ok(Expr::Literal(v))
            }
            Some(Token::Str(s)) => {
                // No Date/DateTime sniffing here: only the binder knows the
                // column type a literal is being compared against.
                let v = Value::str(s.clone());
                self.bump();
                Ok(Expr::Literal(v))
            }
            Some(Token::Star) => {
                self.bump();
                Ok(Expr::Wildcard)
            }
            Some(Token::LParen) => {
                self.bump();
                if self.at_kw("SELECT") || self.at_kw("WITH") {
                    let q = self.query()?;
                    self.expect(&Token::RParen)?;
                    return Ok(Expr::Subquery(Box::new(q)));
                }
                let first = self.expr()?;
                if self.at(&Token::Comma) {
                    let mut items = vec![first];
                    while self.eat(&Token::Comma) {
                        items.push(self.expr()?);
                    }
                    self.expect(&Token::RParen)?;
                    return Ok(Expr::Tuple(items));
                }
                self.expect(&Token::RParen)?;
                Ok(first)
            }
            Some(Token::Word { .. }) => self.word_primary(),
            Some(_) => self.err("an expression"),
        }
    }

    /// Not separately depth-counted: [`Parser::primary`] is its only caller and
    /// already charged one, and a second count here would halve how deeply a
    /// legitimate `f(g(h(...)))` may nest for no extra safety.
    fn word_primary(&mut self) -> Result<Expr> {
        if let Some(w) = self.peek().and_then(|t| t.bare_word()) {
            let upper = w.to_ascii_uppercase();
            match upper.as_str() {
                "NULL" => {
                    self.bump();
                    return Ok(Expr::Literal(Value::Null));
                }
                "TRUE" => {
                    self.bump();
                    return Ok(Expr::Literal(Value::Bool(true)));
                }
                "FALSE" => {
                    self.bump();
                    return Ok(Expr::Literal(Value::Bool(false)));
                }
                "CASE" => return self.case_expr(),
                "INTERVAL" => return self.interval_expr(),
                "CAST" if matches!(self.peek_at(1), Some(Token::LParen)) => {
                    return self.cast_expr()
                }
                "EXISTS" if matches!(self.peek_at(1), Some(Token::LParen)) => {
                    return self.exists_expr(false)
                }
                _ => {}
            }
        }

        // A clause keyword where an operand belongs means the expression is
        // missing, not that someone has a column called `FROM`. Call-shaped
        // uses are exempt: `left(s, 3)` and `any(x)` are real ClickHouse
        // functions whose names happen to be reserved.
        if let Some(w) = self.peek().and_then(|t| t.bare_word()) {
            if is_reserved(w) && !matches!(self.peek_at(1), Some(Token::LParen)) {
                return self.err("an expression");
            }
        }

        let name = self.ident()?;
        if self.at(&Token::LParen) {
            let (args, distinct) = self.call_args()?;
            // A second argument list means the first was ClickHouse's
            // parametric-aggregate prefix: `quantile(0.9)(latency)`.
            if self.at(&Token::LParen) {
                let (real_args, real_distinct) = self.call_args()?;
                let params = args;
                if let Some(spec) = self.over_clause()? {
                    return Ok(Expr::Window {
                        name,
                        args: real_args,
                        params,
                        distinct: real_distinct,
                        spec,
                    });
                }
                return Ok(Expr::Function {
                    name,
                    args: real_args,
                    params,
                    distinct: real_distinct,
                });
            }
            if let Some(spec) = self.over_clause()? {
                return Ok(Expr::Window { name, args, params: Vec::new(), distinct, spec });
            }
            return Ok(Expr::Function { name, args, params: Vec::new(), distinct });
        }
        let mut parts = vec![name];
        while self.at(&Token::Dot) && matches!(self.peek_at(1), Some(Token::Word { .. })) {
            self.bump();
            parts.push(self.ident()?);
        }
        Ok(Expr::Column(ObjectName(parts)))
    }

    fn call_args(&mut self) -> Result<(Vec<Expr>, bool)> {
        self.expect(&Token::LParen)?;
        let distinct = self.eat_kw("DISTINCT");
        let mut args = Vec::new();
        if !self.at(&Token::RParen) {
            args.push(self.expr()?);
            while self.eat(&Token::Comma) {
                args.push(self.expr()?);
            }
        }
        self.expect(&Token::RParen)?;
        Ok((args, distinct))
    }

    /// The `OVER ...` that turns a call into a window function, or `None` when
    /// the next token is not `OVER`.
    ///
    /// Only reached after a complete `f(...)`, so the one-token lookahead
    /// cannot mistake a column named `over` for the keyword unless it directly
    /// follows a call -- where ANSI reserves the word anyway.
    fn over_clause(&mut self) -> Result<Option<Box<WindowSpec>>> {
        if !self.at_kw("OVER") {
            return Ok(None);
        }
        self.bump();
        if self.at(&Token::LParen) {
            return Ok(Some(Box::new(self.window_spec()?)));
        }
        let pos = self.pos();
        let name = self.ident()?;
        match self.named_window(&name) {
            Some(s) => Ok(Some(Box::new(s))),
            None => Err(Error::parse(
                format!("unknown window `{name}`; declare it with `WINDOW {name} AS (...)`"),
                pos,
            )),
        }
    }

    fn named_window(&self, name: &str) -> Option<WindowSpec> {
        self.windows
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, s)| s.clone())
    }

    /// `( [base] [PARTITION BY ...] [ORDER BY ...] [frame] )`.
    fn window_spec(&mut self) -> Result<WindowSpec> {
        self.expect(&Token::LParen)?;
        // ANSI's inherited window: `OVER (w ORDER BY x)` starts from `w` and
        // refines it. Recognized only when the leading word actually names a
        // declared window, so `OVER (x)` -- which is not legal anyway -- still
        // reports the missing clause keyword rather than an unknown name.
        let mut spec = match self.peek().and_then(|t| t.bare_word()) {
            Some(w) => match self.named_window(w) {
                Some(base) => {
                    self.bump();
                    base
                }
                None => WindowSpec::default(),
            },
            None => WindowSpec::default(),
        };

        if self.at_kw("PARTITION") {
            self.bump();
            self.expect_kw("BY")?;
            spec.partition_by = vec![self.expr()?];
            while self.eat(&Token::Comma) {
                spec.partition_by.push(self.expr()?);
            }
        }
        if self.at_kw("ORDER") {
            self.bump();
            self.expect_kw("BY")?;
            spec.order_by = self.order_by_list()?;
        }
        if let Some(f) = self.window_frame()? {
            spec.frame = Some(f);
        }
        self.expect(&Token::RParen)?;
        Ok(spec)
    }

    /// `ROWS`/`RANGE` and its bounds. `None` when no frame was written, which
    /// is *not* the same as the default frame -- see
    /// [`WindowSpec::effective_frame`].
    fn window_frame(&mut self) -> Result<Option<WindowFrame>> {
        let pos = self.pos();
        let units = if self.eat_kw("ROWS") {
            FrameUnits::Rows
        } else if self.eat_kw("RANGE") {
            FrameUnits::Range
        } else if self.at_kw("GROUPS") {
            return Err(Error::parse(
                "GROUPS frames are not implemented; use ROWS for a physical offset",
                pos,
            ));
        } else {
            return Ok(None);
        };

        let (start, end) = if self.eat_kw("BETWEEN") {
            let s = self.frame_bound()?;
            self.expect_kw("AND")?;
            (s, self.frame_bound()?)
        } else {
            // The short form: `ROWS 3 PRECEDING` means `BETWEEN 3 PRECEDING
            // AND CURRENT ROW`.
            (self.frame_bound()?, FrameBound::CurrentRow)
        };
        if start.rank() > end.rank() {
            return Err(Error::parse(
                format!("frame start `{start}` comes after its end `{end}`"),
                pos,
            ));
        }
        // Refused, not silently read as ROWS. A RANGE offset compares *values*
        // of the ORDER BY key, so `RANGE 3 PRECEDING` over `ORDER BY ts` means
        // "the last three units of time", not "the last three rows" -- and the
        // two agree only when the key happens to be dense and unique.
        let offset = |b: &FrameBound| matches!(b, FrameBound::Preceding(_) | FrameBound::Following(_));
        if units == FrameUnits::Range && (offset(&start) || offset(&end)) {
            return Err(Error::parse(
                "RANGE frames with a numeric offset are not implemented; write ROWS for a \
                 physical row offset, or keep the RANGE bounds at UNBOUNDED / CURRENT ROW",
                pos,
            ));
        }
        if self.at_kw("EXCLUDE") {
            return Err(Error::parse(
                "frame EXCLUDE clauses are not implemented",
                self.pos(),
            ));
        }
        Ok(Some(WindowFrame { units, start, end }))
    }

    fn frame_bound(&mut self) -> Result<FrameBound> {
        let pos = self.pos();
        if self.eat_kw("UNBOUNDED") {
            if self.eat_kw("PRECEDING") {
                return Ok(FrameBound::UnboundedPreceding);
            }
            if self.eat_kw("FOLLOWING") {
                return Ok(FrameBound::UnboundedFollowing);
            }
            return self.err("`PRECEDING` or `FOLLOWING` after `UNBOUNDED`");
        }
        if self.eat_kws(&["CURRENT", "ROW"]) {
            return Ok(FrameBound::CurrentRow);
        }
        let n = match self.peek() {
            Some(Token::Number(Value::UInt(n))) => *n,
            Some(Token::Number(Value::Int(n))) if *n >= 0 => *n as u64,
            Some(Token::Number(v)) => {
                return Err(Error::parse(
                    format!("a frame offset must be a non-negative integer, got {v}"),
                    pos,
                ))
            }
            _ => return self.err("`UNBOUNDED`, `CURRENT ROW`, or a row offset"),
        };
        self.bump();
        if self.eat_kw("PRECEDING") {
            return Ok(FrameBound::Preceding(n));
        }
        if self.eat_kw("FOLLOWING") {
            return Ok(FrameBound::Following(n));
        }
        self.err("`PRECEDING` or `FOLLOWING` after a frame offset")
    }

    /// Find and parse this SELECT's `WINDOW` clause without moving the cursor,
    /// returning the token span it occupies so [`Parser::select`] can step over
    /// it when it gets there.
    ///
    /// The scan stays at paren depth 0 and stops at a set operator, so it can
    /// only ever find *this* select's clause: every nested query is inside
    /// parentheses, and a UNION branch parses its own.
    fn window_clause_ahead(&mut self) -> Result<Option<(usize, usize)>> {
        let mut depth = 0i32;
        let mut j = self.i;
        let start = loop {
            let Some(s) = self.toks.get(j) else { return Ok(None) };
            match &s.tok {
                Token::LParen => depth += 1,
                Token::RParen => {
                    if depth == 0 {
                        return Ok(None);
                    }
                    depth -= 1;
                }
                Token::Semicolon if depth == 0 => return Ok(None),
                Token::Word { quoted: false, value } if depth == 0 => {
                    if self.at_window_clause(j - self.i) {
                        break j;
                    }
                    if ["UNION", "EXCEPT", "INTERSECT"].iter().any(|k| value.eq_ignore_ascii_case(k))
                    {
                        return Ok(None);
                    }
                }
                _ => {}
            }
            j += 1;
        };

        let save = self.i;
        self.i = start + 1;
        let r = self.window_defs();
        let end = self.i;
        self.i = save;
        r?;
        Ok(Some((start, end)))
    }

    /// `name AS (spec), name AS (spec), ...`, cursor already past `WINDOW`.
    fn window_defs(&mut self) -> Result<()> {
        loop {
            let pos = self.pos();
            let name = self.ident()?;
            self.expect_kw("AS")?;
            let spec = self.window_spec()?;
            if self.windows.iter().any(|(n, _)| n.eq_ignore_ascii_case(&name)) {
                return Err(Error::parse(format!("window `{name}` is declared twice"), pos));
            }
            self.windows.push((name, spec));
            if !self.eat(&Token::Comma) {
                return Ok(());
            }
        }
    }

    fn case_expr(&mut self) -> Result<Expr> {
        self.expect_kw("CASE")?;
        let operand = if self.at_kw("WHEN") { None } else { Some(Box::new(self.expr()?)) };
        let mut when_then = Vec::new();
        while self.eat_kw("WHEN") {
            let w = self.expr()?;
            self.expect_kw("THEN")?;
            when_then.push((w, self.expr()?));
        }
        if when_then.is_empty() {
            return self.err("`WHEN`");
        }
        let else_result = if self.eat_kw("ELSE") { Some(Box::new(self.expr()?)) } else { None };
        self.expect_kw("END")?;
        Ok(Expr::Case { operand, when_then, else_result })
    }

    fn cast_expr(&mut self) -> Result<Expr> {
        self.expect_kw("CAST")?;
        self.expect(&Token::LParen)?;
        let expr = self.expr()?;
        self.expect_kw("AS")?;
        let ty = self.data_type()?;
        self.expect(&Token::RParen)?;
        Ok(Expr::Cast { expr: Box::new(expr), ty })
    }

    fn exists_expr(&mut self, negated: bool) -> Result<Expr> {
        self.expect_kw("EXISTS")?;
        self.expect(&Token::LParen)?;
        let q = self.query()?;
        self.expect(&Token::RParen)?;
        Ok(Expr::Exists { subquery: Box::new(q), negated })
    }

    fn interval_expr(&mut self) -> Result<Expr> {
        self.expect_kw("INTERVAL")?;
        // The magnitude parses at unary level so `INTERVAL -1 DAY` works while
        // the unit word stays visible to us instead of being eaten as a column.
        let value = self.unary()?;
        let pos = self.pos();
        let unit_word = self.ident()?;
        let unit = IntervalUnit::parse(&unit_word).ok_or_else(|| {
            Error::parse(
                format!("`{unit_word}` is not an interval unit (SECOND, MINUTE, HOUR, DAY, WEEK, MONTH, QUARTER, YEAR)"),
                pos,
            )
        })?;
        Ok(Expr::Interval { value: Box::new(value), unit })
    }
}

// ------------------------------------------------------------ shared pieces

impl Parser<'_> {
    /// A key list where `(a, b)` and `a, b` mean the same thing, as in
    /// `ORDER BY` on CREATE TABLE and `LIMIT n BY`. A lone tuple is flattened;
    /// `(a + b) * 2` is not a tuple so it survives as one key.
    ///
    /// ClickHouse's idiomatic "no sort key" spelling, `ORDER BY tuple()`, is
    /// deliberately **not** flattened to an empty list here: the binder has to
    /// distinguish "the user asked for no ordering" from "there was no ORDER
    /// BY clause at all", and an empty `Vec` cannot express both. It survives
    /// as the `tuple()` call and the binder drops it.
    fn key_list(&mut self) -> Result<Vec<Expr>> {
        let mut list = vec![self.expr()?];
        while self.eat(&Token::Comma) {
            list.push(self.expr()?);
        }
        if list.len() == 1 {
            if let Expr::Tuple(items) = &list[0] {
                return Ok(items.clone());
            }
        }
        Ok(list)
    }

    /// Reassemble a type name and hand it to `DataType::parse`, which owns the
    /// alias table (`BIGINT`, `double`, ...) and the nesting rules.
    fn data_type(&mut self) -> Result<DataType> {
        let pos = self.pos();
        let mut text = match self.peek() {
            Some(Token::Word { value, .. }) => {
                let v = value.clone();
                self.bump();
                v
            }
            _ => return self.err("a type name"),
        };
        if self.at(&Token::LParen) {
            text.push_str(&self.paren_text()?);
        }
        DataType::parse(&text).map_err(|e| match e {
            // A bad type name is a typo and deserves an offset; a valid type we
            // have not built keeps its `Unsupported` kind so the session layer
            // reports "not implemented" rather than "syntax error".
            Error::Bind(m) => Error::parse(m, pos),
            other => other,
        })
    }

    /// Re-serialize a balanced `( ... )` run back to text, parens included.
    /// Type names and CODEC chains are consumed as text by other components,
    /// and the lexer has already dropped the original spacing, so this rebuilds
    /// a normalized form rather than slicing the source.
    ///
    /// Scanning here is a loop, so the parser's own stack is never at risk --
    /// but the string it returns is handed to `DataType::parse`, which *does*
    /// recurse once per `(`, and `Nullable(Nullable(...` 6581 deep aborted the
    /// process through that route. The same ceiling is applied to the text so
    /// the crash is caught at the SQL boundary, where there is still an offset
    /// to report, rather than inside a type parser that has no idea where it is.
    fn paren_text(&mut self) -> Result<String> {
        let open = self.pos();
        self.expect(&Token::LParen)?;
        let mut s = String::from("(");
        let mut depth = 1usize;
        loop {
            let t = match self.peek() {
                Some(t) => t.clone(),
                None => return Err(Error::parse("unterminated `(`", open)),
            };
            self.bump();
            match &t {
                Token::LParen => {
                    depth += 1;
                    if depth > MAX_DEPTH as usize {
                        return Err(Error::parse(
                            format!("nested more than {MAX_DEPTH} levels deep here"),
                            self.pos(),
                        ));
                    }
                    s.push('(');
                }
                Token::RParen => {
                    depth -= 1;
                    s.push(')');
                    if depth == 0 {
                        return Ok(s);
                    }
                }
                Token::Comma => s.push_str(", "),
                other => s.push_str(&other.to_string()),
            }
        }
    }
}

fn cmp_op(t: &Token) -> Option<BinaryOp> {
    Some(match t {
        Token::Eq => BinaryOp::Eq,
        Token::NotEq => BinaryOp::NotEq,
        Token::Lt => BinaryOp::Lt,
        Token::LtEq => BinaryOp::LtEq,
        Token::Gt => BinaryOp::Gt,
        Token::GtEq => BinaryOp::GtEq,
        _ => return None,
    })
}

/// Negate a numeric literal in place where the type allows it. `UInt` only
/// folds at exactly `2^63`, the one unsigned value whose negation is a valid
/// `i64`; everything else falls back to a `UnaryOp` node.
fn negate_literal(v: &Value) -> Option<Value> {
    Some(match v {
        Value::Int(i) => Value::Int(i.checked_neg()?),
        Value::Float(f) => Value::Float(-f),
        Value::UInt(u) if *u == 1u64 << 63 => Value::Int(i64::MIN),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::ast::SelectItem;

    // ---------------------------------------------------------- test helpers

    fn one(sql: &str) -> Statement {
        parse_one(sql).unwrap_or_else(|e| panic!("{sql}\n  -> {e}"))
    }

    fn query_of(sql: &str) -> Query {
        match one(sql) {
            Statement::Query(q) => *q,
            other => panic!("expected a query, got {other:?}"),
        }
    }

    fn select_of(sql: &str) -> Select {
        // `SetExpr` carries a manual `Drop` (ast.rs frees the loop-grown
        // spines iteratively), and Rust forbids moving a field out of a type
        // with a destructor -- so borrow and clone instead of unboxing.
        match &query_of(sql).body {
            SetExpr::Select(s) => (**s).clone(),
            other => panic!("expected a plain SELECT, got {other:?}"),
        }
    }

    fn ex(sql: &str) -> Expr {
        parse_expr(sql).unwrap_or_else(|e| panic!("{sql}\n  -> {e}"))
    }

    /// Fully parenthesized rendering, so precedence is visible in assertions.
    fn sexp(e: &Expr) -> String {
        match e {
            Expr::BinaryOp { left, op, right } => {
                format!("({} {} {})", sexp(left), op.symbol(), sexp(right))
            }
            Expr::UnaryOp { op, expr } => match op {
                UnaryOp::Neg => format!("(- {})", sexp(expr)),
                UnaryOp::Not => format!("(NOT {})", sexp(expr)),
            },
            Expr::IsNull { expr, negated } => {
                format!("({} IS{} NULL)", sexp(expr), if *negated { " NOT" } else { "" })
            }
            other => other.to_string(),
        }
    }

    fn shape(sql: &str) -> String {
        sexp(&ex(sql))
    }

    fn err_of(sql: &str) -> (String, usize) {
        match parse_one(sql) {
            Err(Error::Parse { msg, pos }) => (msg, pos),
            other => panic!("expected a parse error for `{sql}`, got {other:?}"),
        }
    }

    // ------------------------------------------------------------ precedence

    #[test]
    fn or_binds_looser_than_and() {
        assert_eq!(shape("a OR b AND c"), "(a OR (b AND c))");
        assert_eq!(shape("a AND b OR c"), "((a AND b) OR c)");
        assert_eq!(shape("a AND b AND c"), "((a AND b) AND c)");
    }

    #[test]
    fn arithmetic_precedence_and_associativity() {
        assert_eq!(shape("1 + 2 * 3"), "(1 + (2 * 3))");
        assert_eq!(shape("1 * 2 + 3"), "((1 * 2) + 3)");
        assert_eq!(shape("1 - 2 - 3"), "((1 - 2) - 3)");
        assert_eq!(shape("(1 + 2) * 3"), "((1 + 2) * 3)");
        assert_eq!(shape("7 % 3 DIV 2"), "((7 % 3) DIV 2)");
        assert_eq!(shape("a / b / c"), "((a / b) / c)");
    }

    #[test]
    fn not_binds_looser_than_comparison() {
        assert_eq!(shape("NOT a = b"), "(NOT (a = b))");
        assert_eq!(shape("NOT a AND b"), "((NOT a) AND b)");
        assert_eq!(shape("NOT NOT a"), "(NOT (NOT a))");
    }

    #[test]
    fn comparison_sits_between_logic_and_arithmetic() {
        assert_eq!(shape("a + 1 > b * 2"), "((a + 1) > (b * 2))");
        assert_eq!(shape("a = 1 AND b = 2"), "((a = 1) AND (b = 2))");
        // `||` is concatenation and binds tighter than `=`
        assert_eq!(shape("a || b = c"), "((a || b) = c)");
        assert_eq!(shape("a || b || c"), "((a || b) || c)");
    }

    #[test]
    fn unary_minus_folds_into_literals_but_not_columns() {
        assert_eq!(ex("-1"), Expr::Literal(Value::Int(-1)));
        assert_eq!(ex("-1.5"), Expr::Literal(Value::Float(-1.5)));
        assert_eq!(shape("-a"), "(- a)");
        assert_eq!(shape("a - -1"), "(a - -1)");
        assert_eq!(shape("- a + b"), "((- a) + b)");
        assert_eq!(shape("+3"), "3");
        // i64::MIN survives the round trip through an unsigned literal
        assert_eq!(ex("-9223372036854775808"), Expr::Literal(Value::Int(i64::MIN)));
    }

    #[test]
    fn between_swallows_its_own_and() {
        let e = ex("a BETWEEN 1 AND 2 AND b");
        // the outer AND must be the logical one
        match &e {
            Expr::BinaryOp { op: BinaryOp::And, left, right } => {
                assert!(matches!(**left, Expr::Between { negated: false, .. }));
                assert_eq!(**right, Expr::col("b"));
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(ex("a NOT BETWEEN 1 AND 2"), Expr::Between { negated: true, .. }));
    }

    // ------------------------------------------------------------ predicates

    #[test]
    fn is_null_forms() {
        assert!(matches!(ex("a IS NULL"), Expr::IsNull { negated: false, .. }));
        assert!(matches!(ex("a IS NOT NULL"), Expr::IsNull { negated: true, .. }));
        assert_eq!(shape("a IS NULL AND b IS NOT NULL"), "((a IS NULL) AND (b IS NOT NULL))");
    }

    #[test]
    fn like_and_ilike() {
        match &ex("name LIKE 'a%'") {
            Expr::Like { negated, case_insensitive, pattern, .. } => {
                assert!(!negated);
                assert!(!case_insensitive);
                assert_eq!(**pattern, Expr::Literal(Value::str("a%")));
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            ex("name NOT ILIKE 'a%'"),
            Expr::Like { negated: true, case_insensitive: true, .. }
        ));
    }

    #[test]
    fn in_list_and_in_subquery() {
        match &ex("a IN (1, 2, 3)") {
            Expr::InList { list, negated, .. } => {
                assert_eq!(list.len(), 3);
                assert!(!negated);
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(ex("a NOT IN (1)"), Expr::InList { negated: true, .. }));
        assert!(matches!(
            ex("a IN (SELECT id FROM t)"),
            Expr::InSubquery { negated: false, .. }
        ));
        assert!(matches!(
            ex("a NOT IN (SELECT id FROM t)"),
            Expr::InSubquery { negated: true, .. }
        ));
    }

    #[test]
    fn exists_folds_its_negation() {
        assert!(matches!(ex("EXISTS (SELECT 1)"), Expr::Exists { negated: false, .. }));
        assert!(matches!(ex("NOT EXISTS (SELECT 1)"), Expr::Exists { negated: true, .. }));
    }

    // ------------------------------------------------------------- primaries

    #[test]
    fn literals_keep_their_narrowest_value() {
        assert_eq!(ex("1"), Expr::Literal(Value::Int(1)));
        assert_eq!(ex("1.5"), Expr::Literal(Value::Float(1.5)));
        assert_eq!(ex("TRUE"), Expr::Literal(Value::Bool(true)));
        assert_eq!(ex("false"), Expr::Literal(Value::Bool(false)));
        assert_eq!(ex("NULL"), Expr::Literal(Value::Null));
        assert_eq!(ex("'x'"), Expr::Literal(Value::str("x")));
        // date-looking strings stay strings: only the binder knows better
        assert_eq!(ex("'2024-01-01'"), Expr::Literal(Value::str("2024-01-01")));
    }

    #[test]
    fn column_references_can_be_qualified() {
        assert_eq!(ex("a"), Expr::Column(ObjectName::bare("a")));
        assert_eq!(ex("t.a"), Expr::Column(ObjectName(vec!["t".into(), "a".into()])));
        assert_eq!(
            ex("db.t.a"),
            Expr::Column(ObjectName(vec!["db".into(), "t".into(), "a".into()]))
        );
        // quoting lets a keyword be a column
        assert_eq!(ex("`select`"), Expr::Column(ObjectName::bare("select")));
        assert_eq!(ex("\"from\""), Expr::Column(ObjectName::bare("from")));
    }

    #[test]
    fn function_calls_including_clickhouse_shapes() {
        assert_eq!(ex("f(1, 2)"), Expr::func("f", vec![Expr::lit(1i64), Expr::lit(2i64)]));
        assert_eq!(ex("now()"), Expr::func("now", vec![]));
        assert_eq!(ex("count(*)"), Expr::func("count", vec![Expr::Wildcard]));
        assert_eq!(
            ex("count(DISTINCT x)"),
            Expr::Function {
                name: "count".into(),
                args: vec![Expr::col("x")],
                params: vec![],
                distinct: true
            }
        );
        assert_eq!(
            ex("quantile(0.9)(latency)"),
            Expr::Function {
                name: "quantile".into(),
                args: vec![Expr::col("latency")],
                params: vec![Expr::lit(0.9)],
                distinct: false
            }
        );
        // nesting and arithmetic inside arguments
        assert_eq!(shape("sum(a * 2) / count()"), "(sum(a * 2) / count())");
    }

    #[test]
    fn reserved_words_are_operands_only_when_called() {
        // `left` and `any` are ClickHouse functions whose names are reserved
        // for the alias rules; calling them must still work.
        assert_eq!(
            ex("left(s, 3)"),
            Expr::func("left", vec![Expr::col("s"), Expr::lit(3i64)])
        );
        assert!(matches!(ex("any(x)"), Expr::Function { .. }));
        // ...but a bare clause keyword is a missing expression
        let (msg, pos) = err_of("SELECT FROM t");
        assert!(msg.contains("expression") && msg.contains("FROM"), "{msg}");
        assert_eq!(pos, 7);
        // quoting is the escape hatch
        assert_eq!(ex("`from`"), Expr::Column(ObjectName::bare("from")));
    }

    #[test]
    fn casts_in_both_spellings() {
        let want = Expr::Cast { expr: Box::new(Expr::col("x")), ty: DataType::Int64 };
        assert_eq!(ex("CAST(x AS Int64)"), want);
        assert_eq!(ex("x::Int64"), want);
        assert_eq!(
            ex("x::Nullable(Int64)"),
            Expr::Cast {
                expr: Box::new(Expr::col("x")),
                ty: DataType::Nullable(Box::new(DataType::Int64))
            }
        );
        assert_eq!(
            ex("CAST(x AS FixedString(16))"),
            Expr::Cast { expr: Box::new(Expr::col("x")), ty: DataType::FixedString(16) }
        );
        // the cast operator is postfix and binds tighter than arithmetic
        assert_eq!(shape("a::Int64 + 1"), "(CAST(a AS Int64) + 1)");
        // an unknown type names itself in the error
        let e = parse_expr("x::Blob").unwrap_err().to_string();
        assert!(e.contains("Blob"), "{e}");
    }

    #[test]
    fn case_expressions_both_flavours() {
        match &ex("CASE WHEN a THEN 1 ELSE 2 END") {
            Expr::Case { operand, when_then, else_result } => {
                assert!(operand.is_none());
                assert_eq!(when_then.len(), 1);
                assert!(else_result.is_some());
            }
            other => panic!("{other:?}"),
        }
        match &ex("CASE a WHEN 1 THEN 'x' WHEN 2 THEN 'y' END") {
            Expr::Case { operand, when_then, else_result } => {
                assert_eq!(**operand.as_ref().unwrap(), Expr::col("a"));
                assert_eq!(when_then.len(), 2);
                assert!(else_result.is_none());
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn tuples_intervals_and_scalar_subqueries() {
        assert_eq!(ex("(a, b)"), Expr::Tuple(vec![Expr::col("a"), Expr::col("b")]));
        assert_eq!(ex("(a)"), Expr::col("a")); // grouping, not a 1-tuple
        assert_eq!(
            ex("INTERVAL 3 DAY"),
            Expr::Interval { value: Box::new(Expr::lit(3i64)), unit: IntervalUnit::Day }
        );
        assert_eq!(
            ex("INTERVAL 2 months"),
            Expr::Interval { value: Box::new(Expr::lit(2i64)), unit: IntervalUnit::Month }
        );
        assert!(matches!(ex("(SELECT max(x) FROM t)"), Expr::Subquery(_)));
        assert!(parse_expr("INTERVAL 3 FORTNIGHT").is_err());
    }

    // ------------------------------------------------------------ SELECT

    #[test]
    fn select_star_and_qualified_star() {
        let s = select_of("SELECT * FROM t");
        assert_eq!(s.projection, vec![SelectItem::Wildcard]);
        assert!(!s.distinct);
        let s = select_of("SELECT t.*, db.t.*, a FROM t");
        assert_eq!(s.projection[0], SelectItem::QualifiedWildcard("t".into()));
        assert_eq!(s.projection[1], SelectItem::QualifiedWildcard("db.t".into()));
        assert!(matches!(s.projection[2], SelectItem::Expr { .. }));
    }

    #[test]
    fn projection_aliases() {
        let s = select_of("SELECT a AS x, b y, c + 1 AS z, d FROM t");
        let names: Vec<Option<String>> = s
            .projection
            .iter()
            .map(|p| match p {
                SelectItem::Expr { alias, .. } => alias.clone(),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(
            names,
            vec![Some("x".into()), Some("y".into()), Some("z".into()), None]
        );
        // a clause keyword is never eaten as a bare alias
        let s = select_of("SELECT a FROM t WHERE b");
        assert!(matches!(&s.projection[0], SelectItem::Expr { alias: None, .. }));
        assert!(s.selection.is_some());
    }

    #[test]
    fn distinct_and_the_full_clause_set() {
        let s = select_of(
            "SELECT DISTINCT a, sum(b) AS s FROM t FINAL PREWHERE c > 0 WHERE d < 10 \
             GROUP BY a WITH TOTALS HAVING s > 1",
        );
        assert!(s.distinct);
        assert_eq!(s.group_by, vec![Expr::col("a")]);
        assert!(s.with_totals);
        assert!(s.prewhere.is_some());
        assert!(s.selection.is_some());
        assert!(s.having.is_some());
        match s.from.as_ref().unwrap() {
            TableRef::Table { name, final_, alias } => {
                assert_eq!(name.to_string(), "t");
                assert!(*final_);
                assert!(alias.is_none());
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn order_by_directions_and_null_placement() {
        let q = query_of("SELECT a FROM t ORDER BY a, b DESC, c ASC NULLS LAST, d DESC NULLS FIRST");
        let o = &q.order_by;
        assert_eq!(o.len(), 4);
        assert!(o[0].asc && o[0].nulls_first.is_none());
        assert!(!o[1].asc);
        assert_eq!(o[2].nulls_first, Some(false));
        assert_eq!(o[3].nulls_first, Some(true));
        // the AST's default matches the direction
        assert!(o[0].nulls_first_effective());
        assert!(!o[1].nulls_first_effective());
    }

    #[test]
    fn all_four_limit_spellings() {
        let q = query_of("SELECT a FROM t LIMIT 10");
        assert_eq!(q.limit, Some(Expr::lit(10i64)));
        assert!(q.offset.is_none());

        let q = query_of("SELECT a FROM t LIMIT 10 OFFSET 5");
        assert_eq!(q.limit, Some(Expr::lit(10i64)));
        assert_eq!(q.offset, Some(Expr::lit(5i64)));

        // ClickHouse's reversed form puts the offset first
        let q = query_of("SELECT a FROM t LIMIT 5, 10");
        assert_eq!(q.offset, Some(Expr::lit(5i64)));
        assert_eq!(q.limit, Some(Expr::lit(10i64)));

        let q = query_of("SELECT a FROM t LIMIT 2 BY (a, b) LIMIT 100");
        let (n, keys) = q.limit_by.unwrap();
        assert_eq!(n, Expr::lit(2i64));
        assert_eq!(keys, vec![Expr::col("a"), Expr::col("b")]);
        assert_eq!(q.limit, Some(Expr::lit(100i64)));

        // ...and without the parentheses
        let q = query_of("SELECT a FROM t LIMIT 1 BY a");
        assert_eq!(q.limit_by.unwrap().1, vec![Expr::col("a")]);
    }

    #[test]
    fn settings_are_parsed_and_dropped() {
        let s = select_of("SELECT a FROM t SETTINGS max_threads = 8, x = 'y', z = -1");
        assert_eq!(s.projection.len(), 1);
    }

    // -------------------------------------------------------------- joins

    #[test]
    fn join_flavours() {
        let cases = [
            ("SELECT * FROM a JOIN b ON a.x = b.x", JoinOp::Inner),
            ("SELECT * FROM a INNER JOIN b ON a.x = b.x", JoinOp::Inner),
            ("SELECT * FROM a LEFT JOIN b ON a.x = b.x", JoinOp::Left),
            ("SELECT * FROM a LEFT OUTER JOIN b ON a.x = b.x", JoinOp::Left),
            ("SELECT * FROM a RIGHT OUTER JOIN b ON a.x = b.x", JoinOp::Right),
            ("SELECT * FROM a FULL OUTER JOIN b ON a.x = b.x", JoinOp::Full),
            ("SELECT * FROM a CROSS JOIN b", JoinOp::Cross),
            ("SELECT * FROM a, b", JoinOp::Cross),
            ("SELECT * FROM a ANY LEFT JOIN b ON a.x = b.x", JoinOp::Left),
        ];
        for (sql, want) in cases {
            match select_of(sql).from.unwrap() {
                TableRef::Join { op, .. } => assert_eq!(op, want, "{sql}"),
                other => panic!("{sql}: {other:?}"),
            }
        }
    }

    #[test]
    fn join_constraints_and_nesting() {
        match select_of("SELECT * FROM a JOIN b USING (x, y)").from.as_ref().unwrap() {
            TableRef::Join { constraint: JoinConstraint::Using(cols), .. } => {
                assert_eq!(*cols, vec!["x".to_string(), "y".to_string()])
            }
            other => panic!("{other:?}"),
        }
        match select_of("SELECT * FROM a JOIN b USING x").from.as_ref().unwrap() {
            TableRef::Join { constraint: JoinConstraint::Using(cols), .. } => {
                assert_eq!(*cols, vec!["x".to_string()])
            }
            other => panic!("{other:?}"),
        }
        // three-way joins nest to the left
        let three = select_of("SELECT * FROM a JOIN b ON a.x = b.x JOIN c ON b.y = c.y");
        match three.from.as_ref().unwrap() {
            TableRef::Join { left, .. } => assert!(matches!(**left, TableRef::Join { .. })),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn table_factors_take_aliases_and_subqueries() {
        match select_of("SELECT * FROM db.t AS x").from.as_ref().unwrap() {
            TableRef::Table { name, alias, .. } => {
                assert_eq!(name.to_string(), "db.t");
                assert_eq!(alias.as_deref(), Some("x"));
            }
            other => panic!("{other:?}"),
        }
        match select_of("SELECT * FROM t x FINAL").from.as_ref().unwrap() {
            TableRef::Table { alias, final_, .. } => {
                assert_eq!(alias.as_deref(), Some("x"));
                assert!(*final_);
            }
            other => panic!("{other:?}"),
        }
        match select_of("SELECT * FROM (SELECT a FROM t) s").from.as_ref().unwrap() {
            TableRef::Subquery { alias, .. } => assert_eq!(alias.as_deref(), Some("s")),
            other => panic!("{other:?}"),
        }
    }

    // ---------------------------------------------------------- set ops, CTEs

    #[test]
    fn set_operations_and_their_precedence() {
        match query_of("SELECT a FROM t UNION ALL SELECT a FROM u").body {
            SetExpr::SetOperation { op: SetOp::Union, all: true, .. } => {}
            other => panic!("{other:?}"),
        }
        match query_of("SELECT a FROM t UNION DISTINCT SELECT a FROM u").body {
            SetExpr::SetOperation { op: SetOp::Union, all: false, .. } => {}
            other => panic!("{other:?}"),
        }
        match query_of("SELECT a FROM t EXCEPT SELECT a FROM u").body {
            SetExpr::SetOperation { op: SetOp::Except, .. } => {}
            other => panic!("{other:?}"),
        }
        // INTERSECT binds tighter than UNION
        match &query_of("SELECT 1 UNION ALL SELECT 2 INTERSECT SELECT 3").body {
            SetExpr::SetOperation { op: SetOp::Union, right, .. } => {
                assert!(matches!(**right, SetExpr::SetOperation { op: SetOp::Intersect, .. }));
            }
            other => panic!("{other:?}"),
        }
        // ORDER BY after a union belongs to the whole query
        let q = query_of("SELECT 1 UNION ALL SELECT 2 ORDER BY 1");
        assert_eq!(q.order_by.len(), 1);
    }

    #[test]
    fn ctes_bind_to_the_query() {
        let q = query_of("WITH a AS (SELECT 1), b AS (SELECT 2) SELECT * FROM a");
        assert_eq!(q.with.len(), 2);
        assert_eq!(q.with[0].name, "a");
        assert_eq!(q.with[1].name, "b");
        assert!(matches!(q.body, SetExpr::Select(_)));
        // a CTE without its parenthesized body is a clear error
        let (msg, _) = err_of("WITH a AS SELECT 1 SELECT 2");
        assert!(msg.contains('('), "{msg}");
    }

    // ------------------------------------------------------------ DML / DDL

    #[test]
    fn insert_values_and_select() {
        match one("INSERT INTO t (a, b) VALUES (1, 'x'), (2, 'y')") {
            Statement::Insert(i) => {
                assert_eq!(i.table.to_string(), "t");
                assert_eq!(i.columns, vec!["a".to_string(), "b".to_string()]);
                match i.source {
                    InsertSource::Values(rows) => {
                        assert_eq!(rows.len(), 2);
                        assert_eq!(rows[0][0], Expr::lit(1i64));
                        assert_eq!(rows[1][1], Expr::lit("y"));
                    }
                    other => panic!("{other:?}"),
                }
            }
            other => panic!("{other:?}"),
        }
        match one("INSERT INTO db.t VALUES (-1, NULL, 2.5)") {
            Statement::Insert(i) => {
                assert!(i.columns.is_empty());
                match i.source {
                    InsertSource::Values(rows) => {
                        assert_eq!(rows[0][0], Expr::Literal(Value::Int(-1)));
                        assert_eq!(rows[0][1], Expr::Literal(Value::Null));
                    }
                    other => panic!("{other:?}"),
                }
            }
            other => panic!("{other:?}"),
        }
        match one("INSERT INTO t SELECT * FROM u") {
            Statement::Insert(i) => assert!(matches!(i.source, InsertSource::Query(_))),
            other => panic!("{other:?}"),
        }
        match one("INSERT INTO t (a) SELECT a FROM u") {
            Statement::Insert(i) => {
                assert_eq!(i.columns, vec!["a".to_string()]);
                assert!(matches!(i.source, InsertSource::Query(_)));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn create_table_with_every_clause() {
        let st = one(
            "CREATE TABLE IF NOT EXISTS db.hits (
                 id UInt64,
                 url String DEFAULT '' CODEC(ZSTD(3)),
                 ts DateTime,
                 note Nullable(String),
                 n Int32 NULL
             ) ENGINE = ReplacingMergeTree
             PARTITION BY toYYYYMM(ts)
             ORDER BY (id, url)
             PRIMARY KEY id
             SETTINGS index_granularity = 8192",
        );
        let ct = match st {
            Statement::CreateTable(ct) => *ct,
            other => panic!("{other:?}"),
        };
        assert!(ct.if_not_exists);
        assert_eq!(ct.name.to_string(), "db.hits");
        assert_eq!(ct.engine, Engine::ReplacingMergeTree);
        assert_eq!(ct.columns.len(), 5);
        assert_eq!(ct.columns[0].ty, DataType::UInt64);
        assert_eq!(ct.columns[1].default, Some(Expr::lit("")));
        assert_eq!(ct.columns[1].codec.as_deref(), Some("ZSTD(3)"));
        assert_eq!(ct.columns[3].ty, DataType::Nullable(Box::new(DataType::String)));
        assert_eq!(ct.columns[4].ty, DataType::Nullable(Box::new(DataType::Int32)));
        assert_eq!(ct.order_by, vec![Expr::col("id"), Expr::col("url")]);
        assert_eq!(ct.primary_key, vec![Expr::col("id")]);
        assert_eq!(
            ct.partition_by,
            Some(Expr::func("toYYYYMM", vec![Expr::col("ts")]))
        );
        assert!(ct.as_query.is_none());
    }

    #[test]
    fn create_table_as_select_and_empty_sort_key() {
        let ct = match one("CREATE TABLE t ENGINE = Memory AS SELECT a FROM u") {
            Statement::CreateTable(ct) => *ct,
            other => panic!("{other:?}"),
        };
        assert_eq!(ct.engine, Engine::Memory);
        assert!(ct.columns.is_empty());
        assert!(ct.as_query.is_some());

        let ct = match one("CREATE TABLE t (a UInt8) ENGINE = MergeTree() ORDER BY tuple()") {
            Statement::CreateTable(ct) => *ct,
            other => panic!("{other:?}"),
        };
        assert_eq!(ct.engine, Engine::MergeTree);
        // `tuple()` survives so the binder can tell "explicitly no ordering"
        // apart from "no ORDER BY clause" — an empty Vec cannot say both.
        assert_eq!(ct.order_by.len(), 1);
        assert!(
            matches!(&ct.order_by[0], Expr::Function { name, args, .. }
                     if name.eq_ignore_ascii_case("tuple") && args.is_empty()),
            "ORDER BY tuple() should reach the binder as the tuple() call: {:?}",
            ct.order_by[0]
        );

        // ...whereas omitting ORDER BY entirely leaves the list empty.
        let ct = match one("CREATE TABLE t (a UInt8) ENGINE = Memory") {
            Statement::CreateTable(ct) => *ct,
            other => panic!("{other:?}"),
        };
        assert!(ct.order_by.is_empty());
    }

    #[test]
    fn database_and_drop_statements() {
        assert_eq!(
            one("CREATE DATABASE IF NOT EXISTS d"),
            Statement::CreateDatabase { name: "d".into(), if_not_exists: true }
        );
        assert_eq!(
            one("DROP DATABASE d"),
            Statement::DropDatabase { name: "d".into(), if_exists: false }
        );
        assert_eq!(
            one("DROP TABLE IF EXISTS db.t"),
            Statement::DropTable {
                name: ObjectName(vec!["db".into(), "t".into()]),
                if_exists: true
            }
        );
    }

    #[test]
    fn alter_mutations() {
        match one("ALTER TABLE t DELETE WHERE a > 1") {
            Statement::AlterDelete { table, predicate } => {
                assert_eq!(table.to_string(), "t");
                assert_eq!(sexp(&predicate), "(a > 1)");
            }
            other => panic!("{other:?}"),
        }
        match one("ALTER TABLE t UPDATE a = 1, b = b + 1 WHERE c = 2") {
            Statement::AlterUpdate { assignments, predicate, .. } => {
                assert_eq!(assignments.len(), 2);
                assert_eq!(assignments[0].0, "a");
                assert_eq!(sexp(&assignments[1].1), "(b + 1)");
                assert_eq!(sexp(&predicate), "(c = 2)");
            }
            other => panic!("{other:?}"),
        }
        match one("ALTER TABLE t ADD COLUMN IF NOT EXISTS c Nullable(Int64)") {
            Statement::AlterAddColumn { column, if_not_exists, .. } => {
                assert!(if_not_exists);
                assert_eq!(column.name, "c");
                assert_eq!(column.ty, DataType::Nullable(Box::new(DataType::Int64)));
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(
            one("ALTER TABLE t DROP COLUMN IF EXISTS c"),
            Statement::AlterDropColumn {
                table: ObjectName::bare("t"),
                column: "c".into(),
                if_exists: true
            }
        );
        // the mandatory WHERE is enforced on this spelling
        assert!(parse_one("ALTER TABLE t DELETE").is_err());
        assert!(parse_one("ALTER TABLE t UPDATE a = 1").is_err());
    }

    /// The ANSI spellings, and the property that matters most about them: they
    /// are the *same node*, so there is one execution path rather than two.
    #[test]
    fn delete_and_update_parse_to_the_same_node_as_their_alter_synonyms() {
        assert_eq!(
            parse_one("DELETE FROM t WHERE a > 1").unwrap(),
            parse_one("ALTER TABLE t DELETE WHERE a > 1").unwrap()
        );
        assert_eq!(
            parse_one("UPDATE t SET a = 1, b = b + 1 WHERE c = 2").unwrap(),
            parse_one("ALTER TABLE t UPDATE a = 1, b = b + 1 WHERE c = 2").unwrap()
        );
        assert_eq!(
            parse_one("DELETE FROM db.t WHERE a IN (1, 2)").unwrap(),
            parse_one("ALTER TABLE db.t DELETE WHERE a IN (1, 2)").unwrap()
        );
    }

    #[test]
    fn a_mutation_without_a_where_affects_every_row() {
        for sql in ["DELETE FROM t", "UPDATE t SET a = 1"] {
            let p = match parse_one(sql).unwrap() {
                Statement::AlterDelete { predicate, .. } => predicate,
                Statement::AlterUpdate { predicate, .. } => predicate,
                other => panic!("{other:?}"),
            };
            assert_eq!(p, Expr::Literal(Value::Bool(true)), "{sql}");
        }
    }

    #[test]
    fn update_assignment_values_are_full_expressions() {
        match one("UPDATE t SET s = CASE WHEN a > 1 THEN 'big' ELSE 'small' END WHERE b IS NULL") {
            Statement::AlterUpdate { table, assignments, predicate } => {
                assert_eq!(table.to_string(), "t");
                assert_eq!(assignments.len(), 1);
                assert_eq!(assignments[0].0, "s");
                assert!(matches!(assignments[0].1, Expr::Case { .. }));
                assert_eq!(sexp(&predicate), "(b IS NULL)");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn mutation_syntax_errors_name_the_expectation() {
        // `DELETE t` (no FROM) is the commonest slip; so is a missing SET.
        assert!(err_of("DELETE t WHERE a = 1").0.contains("FROM"));
        assert!(err_of("UPDATE t a = 1").0.contains("SET"));
        assert!(err_of("UPDATE t SET a").0.contains('='));
        // `DELETE`/`UPDATE` are not reserved, so they stay usable as names.
        assert!(parse_one("SELECT delete, update FROM t").is_ok());
        assert!(parse_one("SELECT a AS delete FROM t").is_ok());
    }

    #[test]
    fn maintenance_and_introspection_statements() {
        assert_eq!(
            one("OPTIMIZE TABLE t FINAL"),
            Statement::Optimize { table: ObjectName::bare("t"), final_: true }
        );
        assert_eq!(
            one("OPTIMIZE TABLE t"),
            Statement::Optimize { table: ObjectName::bare("t"), final_: false }
        );
        assert_eq!(one("TRUNCATE TABLE t"), Statement::Truncate { table: ObjectName::bare("t") });
        assert_eq!(one("SHOW TABLES"), Statement::ShowTables { database: None });
        assert_eq!(
            one("SHOW TABLES FROM db"),
            Statement::ShowTables { database: Some("db".into()) }
        );
        assert_eq!(one("SHOW DATABASES"), Statement::ShowDatabases);
        assert_eq!(
            one("SHOW CREATE TABLE t"),
            Statement::ShowCreateTable(ObjectName::bare("t"))
        );
        assert_eq!(one("DESCRIBE TABLE t"), Statement::Describe(ObjectName::bare("t")));
        assert_eq!(one("DESC t"), Statement::Describe(ObjectName::bare("t")));
        assert_eq!(one("USE db"), Statement::Use("db".into()));
        assert_eq!(one("SYSTEM FLUSH"), Statement::SystemFlush(None));
        assert_eq!(
            one("SYSTEM FLUSH db.t"),
            Statement::SystemFlush(Some(ObjectName(vec!["db".into(), "t".into()])))
        );
    }

    #[test]
    fn explain_wraps_a_statement() {
        match one("EXPLAIN PIPELINE SELECT 1") {
            Statement::Explain { kind, statement } => {
                assert_eq!(kind, ExplainKind::Pipeline);
                assert!(matches!(*statement, Statement::Query(_)));
            }
            other => panic!("{other:?}"),
        }
        match one("EXPLAIN AST INSERT INTO t VALUES (1)") {
            Statement::Explain { kind, statement } => {
                assert_eq!(kind, ExplainKind::Ast);
                assert!(matches!(*statement, Statement::Insert(_)));
            }
            other => panic!("{other:?}"),
        }
        // no kind defaults to the logical plan
        match one("EXPLAIN SELECT 1") {
            Statement::Explain { kind, .. } => assert_eq!(kind, ExplainKind::Plan),
            other => panic!("{other:?}"),
        }
    }

    // ------------------------------------------------------- entry points

    #[test]
    fn scripts_split_on_semicolons() {
        let sts = parse("SELECT 1; SELECT 2;;\n-- trailing comment\nSELECT 3").unwrap();
        assert_eq!(sts.len(), 3);
        assert!(parse("").unwrap().is_empty());
        assert!(parse("  ;; \n").unwrap().is_empty());
        assert!(parse("-- just a comment").unwrap().is_empty());
    }

    #[test]
    fn parse_one_rejects_zero_or_many() {
        assert!(parse_one("SELECT 1").is_ok());
        assert!(parse_one("SELECT 1;").is_ok());
        assert!(parse_one("").is_err());
        assert!(parse_one("   ").is_err());
        let (msg, pos) = err_of("SELECT 1; SELECT 2");
        assert!(msg.contains("single statement"), "{msg}");
        assert_eq!(pos, 10);
    }

    #[test]
    fn parse_expr_rejects_trailing_input() {
        assert!(parse_expr("1 + 1").is_ok());
        assert!(parse_expr("1 + 1 FROM t").is_err());
        assert!(parse_expr("").is_err());
    }

    #[test]
    fn errors_name_the_expectation_and_the_offset() {
        let (msg, pos) = err_of("SELECT a FROM");
        assert!(msg.contains("identifier"), "{msg}");
        assert_eq!(pos, 13); // end of input

        let (msg, pos) = err_of("SELECT a FROM t WHERE");
        assert!(msg.contains("expression"), "{msg}");
        assert_eq!(pos, 21);

        let (msg, pos) = err_of("SELECT a FROM t GROUP a");
        assert!(msg.contains("`BY`"), "{msg}");
        assert_eq!(pos, 22);

        let (msg, pos) = err_of("SELECT (1, 2");
        assert!(msg.contains("`)`"), "{msg}");
        assert_eq!(pos, 12);

        let (msg, _) = err_of("SELECT a FROM t ORDER BY a NULLS SOON");
        assert!(msg.contains("FIRST"), "{msg}");

        let (msg, _) = err_of("FLUMMOX t");
        assert!(msg.contains("statement"), "{msg}");

        let (msg, _) = err_of("CREATE TABLE t (a UInt8) ENGINE = MergeTree ORDER BY");
        assert!(msg.contains("expression"), "{msg}");
    }

    // --------------------------------------------------------- nesting depth

    /// Deepest `n` a nesting shape still admits, found by walking up from 1.
    /// Walking (rather than bisecting) is the point: every level below the
    /// boundary is parsed, so a shape that breaks *before* the limit, or a
    /// limit that arrives at a different depth than the accounting predicts,
    /// shows up here rather than in a fuzzer months later.
    fn depth_boundary(build: impl Fn(usize) -> String) -> usize {
        for n in 1.. {
            let sql = build(n);
            match parse(&sql) {
                Ok(_) => {}
                Err(Error::Parse { msg, pos }) => {
                    assert!(msg.contains("nested more than"), "at n={n}: {msg}");
                    assert!(pos > 0 && pos < sql.len(), "offset {pos} is outside the input");
                    return n - 1;
                }
                Err(other) => panic!("at n={n}: {other:?}"),
            }
        }
        unreachable!()
    }

    // One builder per nesting shape, so the boundary test and the "does not
    // crash" test agree on what each shape is. `- ` keeps its space: `--`
    // lexes as a line comment and would swallow the rest of the query.
    fn parens(n: usize) -> String {
        format!("SELECT {}1{}", "(".repeat(n), ")".repeat(n))
    }
    fn subqueries(n: usize) -> String {
        format!("SELECT * FROM {}(SELECT 1{}", "(SELECT * FROM ".repeat(n), ")".repeat(n + 1))
    }
    fn calls(n: usize) -> String {
        format!("SELECT {}1{}", "abs(".repeat(n), ")".repeat(n))
    }
    fn nots(n: usize) -> String {
        format!("SELECT {}x", "NOT ".repeat(n))
    }
    fn negs(n: usize) -> String {
        format!("SELECT {}x", "- ".repeat(n))
    }
    fn explains(n: usize) -> String {
        format!("{}SELECT 1", "EXPLAIN ".repeat(n))
    }
    fn join_parens(n: usize) -> String {
        format!("SELECT * FROM {}t{}", "(".repeat(n), ")".repeat(n))
    }
    fn in_lists(n: usize) -> String {
        format!("SELECT 1 WHERE {}1{}", "x IN (".repeat(n), ")".repeat(n))
    }
    fn cases(n: usize) -> String {
        format!("SELECT {}1{}", "CASE WHEN 1 THEN ".repeat(n), " END".repeat(n))
    }
    fn ctes(n: usize) -> String {
        let mut s = String::from("SELECT 1");
        for _ in 0..n {
            s = format!("WITH c AS ({s}) SELECT 1");
        }
        s
    }
    fn exists(n: usize) -> String {
        let mut s = String::from("SELECT 1");
        for _ in 0..n {
            s = format!("SELECT 1 WHERE EXISTS ({s})");
        }
        s
    }

    #[test]
    fn every_recursive_descent_is_depth_limited() {
        // The budget made visible. Shapes whose cycle crosses two guards
        // (expr+primary, query+table_ref) get half of MAX_DEPTH; shapes that
        // cross one (statement, prefix NOT, unary minus) get all of it, less
        // the two or three counts the enclosing statement already holds.
        let d = MAX_DEPTH as usize;

        // Two counts a level: the cycle runs `expr` -> ... -> `primary` -> the
        // nested thing -> `expr`.
        assert_eq!(depth_boundary(parens), d / 2 - 2);
        assert_eq!(depth_boundary(calls), d / 2 - 2);
        assert_eq!(depth_boundary(cases), d / 2 - 2);
        // Ditto, through `query` -> `table_ref`.
        assert_eq!(depth_boundary(subqueries), d / 2 - 3);

        // One count a level: the cycle crosses a single guard, because the
        // frames in between hand their counts back before recursing --
        // `primary` has already returned the `x` when `cmp_expr` calls
        // `in_rest`, and a parenthesized table ref never re-enters `expr`.
        assert_eq!(depth_boundary(join_parens), d - 3);
        assert_eq!(depth_boundary(in_lists), d - 4);
        assert_eq!(depth_boundary(ctes), d - 4);
        assert_eq!(depth_boundary(nots), d - 4);
        assert_eq!(depth_boundary(negs), d - 4);
        assert_eq!(depth_boundary(explains), d - 4);

        // Three: `EXISTS` is reached through `primary`, which is still holding
        // its count when `exists_expr` starts the next `query`. This is the
        // shape MAX_DEPTH binds hardest, and the one its margin is set from.
        assert_eq!(depth_boundary(exists), (d - 5) / 3);
    }

    #[test]
    fn absurd_nesting_is_a_parse_error_not_a_crash() {
        // 3 KB of parens; before the limit existed this aborted the process
        // (`panic = "abort"` in release means a blown stack has no catch, so
        // an embedded caller lost the whole host with it).
        let sql = parens(3000);
        let (msg, pos) = err_of(&sql);
        assert!(msg.contains("nested more than"), "{msg}");
        // The offset points at the paren that broke the limit, as every other
        // error in this parser points at the token it choked on.
        assert!(pos > 7 && pos < sql.len(), "{pos}");

        for sql in [
            subqueries(2000),
            nots(20_000),
            negs(20_000),
            explains(20_000),
            calls(3000),
            join_parens(3000),
            in_lists(3000),
            cases(3000),
            ctes(3000),
            exists(3000),
        ] {
            let (msg, _) = err_of(&sql);
            assert!(msg.contains("nested more than"), "{msg}");
        }
    }

    #[test]
    fn nested_type_text_is_capped_before_it_reaches_the_type_parser() {
        // `paren_text` loops, so this never threatened *our* stack -- but the
        // text goes to `DataType::parse`, which recurses per `(` and aborted
        // the process at 6581 levels of `Nullable(`. Both spellings of a cast
        // and a column definition all funnel through the same helper.
        let deep_ty = format!("{}Int8{}", "Nullable(".repeat(3000), ")".repeat(3000));
        for sql in [
            format!("SELECT CAST(1 AS {deep_ty})"),
            format!("SELECT 1::{deep_ty}"),
            format!("CREATE TABLE t (a {deep_ty}) ENGINE = MergeTree ORDER BY a"),
        ] {
            let (msg, _) = err_of(&sql);
            assert!(msg.contains("nested more than"), "{msg}");
        }
        // The shallow nesting real schemas use is untouched.
        assert!(parse_expr("1::Nullable(Nullable(Int8))").is_ok());
    }

    #[test]
    fn a_flat_chain_is_capped_but_an_in_list_is_not() {
        // The distinction that matters for real clients: an ORM expanding a
        // large `IN` list produces a `Vec`, which costs nothing and must keep
        // working; the `OR` chain some ORMs emit instead produces a left-deep
        // tree, which does not.
        let list: Vec<String> = (0..60_000).map(|i| i.to_string()).collect();
        assert!(parse_expr(&format!("x IN ({})", list.join(", "))).is_ok());

        // Chains are capped. An earlier version of this test asserted 20k
        // chains parse, on the measurement that the AST's recursive `Drop`
        // aborts at 43 946 terms -- true, but not the binding constraint.
        // `Expr` also derives `Clone`, `PartialEq` and `Debug`, all three of
        // which recurse per node, and the binder clones expressions: a 25k
        // chain aborted the process inside `expr.clone()` *while producing the
        // error that rejects it*. Iterative `Drop` cannot fix a derive, so the
        // chain is bounded where it is grown instead.
        for sep in [" OR ", " AND ", " + ", " || "] {
            let terms: Vec<String> =
                (0..MAX_CHAIN + 10).map(|i| format!("x = {i}")).collect();
            let e = parse_expr(&terms.join(sep)).expect_err("a chain past the cap must be refused");
            assert!(e.to_string().contains("chains more than"), "{e}");
        }

        // Just under the cap still parses, so the limit is a cliff and not a
        // slope that has been quietly eroding what works.
        let ok: Vec<String> = (0..MAX_CHAIN - 1).map(|i| format!("x = {i}")).collect();
        assert!(parse_expr(&ok.join(" OR ")).is_ok());
    }

    #[test]
    fn the_depth_counter_is_given_back_on_every_path() {
        // A script is parsed by *one* `Parser` sharing *one* counter, so a
        // count that leaked anywhere -- the obvious way being a hand-written
        // decrement placed after a call that returns through `?` -- ratchets
        // the effective limit down statement by statement. At 88 counts a
        // statement, 500 statements would blow a limit of 200 within three if
        // even one count went missing per statement, and the failure would be
        // an ordinary query rejected, not a crash.
        let deep = parens(80);
        let script = vec![deep.as_str(); 500].join(";\n");
        assert_eq!(parse(&script).unwrap().len(), 500);

        // The error path deserves its own pass even though every `Parser` here
        // is fresh: the parser has three rewind points already, and the day one
        // of them learns to retry a *guarded* descent, a count kept on the
        // abandoned attempt starts leaking within a single parse. This pins the
        // behaviour now, while the counter is still easy to reason about.
        let over = parens(3000);
        for _ in 0..200 {
            assert!(parse_one(&over).is_err());
            assert!(parse_one(&deep).is_ok());
        }
    }

    #[test]
    fn keywords_are_case_insensitive_end_to_end() {
        let a = one("select a from t where b = 1 order by a desc limit 5");
        let b = one("SELECT a FROM t WHERE b = 1 ORDER BY a DESC LIMIT 5");
        let c = one("SeLeCt a FrOm t WhErE b = 1 OrDeR By a DeSc LiMiT 5");
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn comments_and_layout_do_not_change_the_ast() {
        let a = one("SELECT a, b FROM t WHERE a > 1");
        let b = one(
            "SELECT a, /* inline */ b\n  -- pick a few\n  FROM t\n  WHERE a > 1 -- done",
        );
        assert_eq!(a, b);
    }

    #[test]
    fn a_realistic_analytical_query_round_trips() {
        let q = query_of(
            "WITH recent AS (SELECT * FROM hits WHERE ts > now() - INTERVAL 7 DAY)
             SELECT
                 domain,
                 count(*) AS hits,
                 uniq(user_id) AS users,
                 quantile(0.95)(latency_ms) AS p95,
                 sum(CASE WHEN status >= 500 THEN 1 ELSE 0 END) AS errors
             FROM recent AS r
             LEFT JOIN domains AS d ON r.domain = d.name
             PREWHERE status != 304
             WHERE d.active AND domain NOT LIKE '%.test'
             GROUP BY domain
             HAVING hits > 100
             ORDER BY hits DESC NULLS LAST
             LIMIT 3 BY domain
             LIMIT 50 OFFSET 10",
        );
        assert_eq!(q.with.len(), 1);
        let s = match &q.body {
            SetExpr::Select(s) => s,
            other => panic!("{other:?}"),
        };
        assert_eq!(s.projection.len(), 5);
        assert!(s.prewhere.is_some() && s.selection.is_some() && s.having.is_some());
        assert!(matches!(s.from, Some(TableRef::Join { op: JoinOp::Left, .. })));
        assert_eq!(q.order_by[0].nulls_first, Some(false));
        assert_eq!(q.limit_by.as_ref().unwrap().1, vec![Expr::col("domain")]);
        assert_eq!(q.limit, Some(Expr::lit(50i64)));
        assert_eq!(q.offset, Some(Expr::lit(10i64)));
    }
    // ------------------------------------------------------------- windows

    /// The `OVER` clause of the first window call in a query.
    fn over_of(sql: &str) -> WindowSpec {
        let mut found = None;
        for item in &select_of(sql).projection {
            if let SelectItem::Expr { expr, .. } = item {
                expr.visit(&mut |e| {
                    if let Expr::Window { spec, .. } = e {
                        if found.is_none() {
                            found = Some((**spec).clone());
                        }
                    }
                });
            }
        }
        found.unwrap_or_else(|| panic!("no window call in `{sql}`"))
    }

    #[test]
    fn a_call_without_over_is_still_a_plain_function() {
        // The whole grammar change has to be invisible to every query that
        // does not use it.
        assert!(matches!(ex("sum(x)"), Expr::Function { .. }));
        assert!(matches!(ex("quantile(0.9)(x)"), Expr::Function { .. }));
        assert!(matches!(ex("count(DISTINCT x)"), Expr::Function { .. }));
    }

    #[test]
    fn over_turns_any_call_shape_into_a_window() {
        for sql in ["sum(x) OVER ()", "quantile(0.9)(x) OVER ()", "count(*) OVER ()"] {
            assert!(matches!(ex(sql), Expr::Window { .. }), "{sql}");
        }
        // Borrowed rather than destructured: `Expr` has a manual `Drop`, so
        // Rust will not let a field move out of one.
        match &ex("quantile(0.9)(x) OVER (PARTITION BY k)") {
            Expr::Window { name, args, params, spec, .. } => {
                assert_eq!(name, "quantile");
                assert_eq!(args, &[Expr::col("x")]);
                assert_eq!(params, &[Expr::lit(0.9)]);
                assert_eq!(spec.partition_by, vec![Expr::col("k")]);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn partition_order_and_frame_all_parse() {
        let s = over_of(
            "SELECT sum(v) OVER (PARTITION BY a, b ORDER BY c DESC NULLS FIRST \
             ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING) FROM t",
        );
        assert_eq!(s.partition_by, vec![Expr::col("a"), Expr::col("b")]);
        assert_eq!(s.order_by.len(), 1);
        assert!(!s.order_by[0].asc);
        assert_eq!(s.order_by[0].nulls_first, Some(true));
        assert_eq!(
            s.frame,
            Some(WindowFrame {
                units: FrameUnits::Rows,
                start: FrameBound::Preceding(2),
                end: FrameBound::Following(1),
            })
        );
    }

    #[test]
    fn the_short_frame_form_ends_at_the_current_row() {
        let s = over_of("SELECT sum(v) OVER (ORDER BY a ROWS 3 PRECEDING) FROM t");
        assert_eq!(
            s.frame,
            Some(WindowFrame {
                units: FrameUnits::Rows,
                start: FrameBound::Preceding(3),
                end: FrameBound::CurrentRow,
            })
        );
    }

    #[test]
    fn the_default_frame_depends_on_whether_there_is_an_order_by() {
        // Not a parser choice -- the parser records `None` -- but the two
        // meanings of `None` are the thing everyone gets wrong, so they are
        // pinned where the spec that defines them lives.
        let unordered = over_of("SELECT sum(v) OVER (PARTITION BY k) FROM t");
        assert_eq!(unordered.frame, None);
        assert_eq!(unordered.effective_frame().end, FrameBound::UnboundedFollowing);
        let ordered = over_of("SELECT sum(v) OVER (ORDER BY k) FROM t");
        assert_eq!(ordered.effective_frame().end, FrameBound::CurrentRow);
        assert_eq!(ordered.effective_frame().units, FrameUnits::Range);
    }

    #[test]
    fn named_windows_are_substituted_at_every_reference() {
        let q = "SELECT row_number() OVER w, sum(v) OVER w FROM t \
                 WINDOW w AS (PARTITION BY k ORDER BY ts)";
        let s = over_of(q);
        assert_eq!(s.partition_by, vec![Expr::col("k")]);
        assert_eq!(s.order_by.len(), 1);
        // Both references resolved: nothing named survives into the AST, so
        // the binder has no name table to consult.
        let sel = select_of(q);
        assert_eq!(sel.projection.len(), 2);
        for item in &sel.projection {
            if let SelectItem::Expr { expr: Expr::Window { spec, .. }, .. } = item {
                assert_eq!(spec.partition_by, vec![Expr::col("k")]);
            } else {
                panic!("both items are window calls");
            }
        }
    }

    #[test]
    fn a_named_window_can_refine_an_earlier_one() {
        let s = over_of(
            "SELECT sum(v) OVER w2 FROM t \
             WINDOW w AS (PARTITION BY k), w2 AS (w ORDER BY ts DESC)",
        );
        assert_eq!(s.partition_by, vec![Expr::col("k")]);
        assert_eq!(s.order_by.len(), 1);
        assert!(!s.order_by[0].asc);
    }

    #[test]
    fn an_over_clause_may_extend_a_named_window_inline() {
        let s = over_of(
            "SELECT sum(v) OVER (w ROWS UNBOUNDED PRECEDING) FROM t \
             WINDOW w AS (PARTITION BY k ORDER BY ts)",
        );
        assert_eq!(s.partition_by, vec![Expr::col("k")]);
        assert_eq!(s.frame.unwrap().start, FrameBound::UnboundedPreceding);
    }

    #[test]
    fn a_named_window_is_visible_to_the_query_order_by() {
        // ORDER BY is parsed by `query_tail`, after `select` returns, so the
        // name table has to outlive the SELECT that declared it.
        let q = query_of(
            "SELECT v FROM t WINDOW w AS (ORDER BY k) ORDER BY row_number() OVER w",
        );
        assert!(matches!(q.order_by[0].expr, Expr::Window { .. }));
    }

    #[test]
    fn window_is_not_reserved_and_still_works_as_an_identifier() {
        // The clause is recognized by its `WINDOW <name> AS (` shape rather
        // than by reserving the word, so ordinary uses keep working.
        assert!(matches!(ex("window"), Expr::Column(_)));
        let s = select_of("SELECT window FROM t");
        assert_eq!(s.projection.len(), 1);
        let s = select_of("SELECT x AS window FROM t");
        assert!(matches!(&s.projection[0], SelectItem::Expr { alias: Some(a), .. } if a == "window"));
        let s = select_of("SELECT * FROM window");
        assert!(matches!(s.from, Some(TableRef::Table { .. })));
        // And the alias position, which is where the clause could be mistaken
        // for one: `FROM t WINDOW` used to read as `FROM t AS WINDOW`.
        let s = select_of("SELECT * FROM t window");
        assert!(matches!(&s.from, Some(TableRef::Table { alias: Some(a), .. }) if a == "window"));
    }

    #[test]
    fn named_windows_do_not_leak_across_queries() {
        // Declared in the subquery, referenced in the outer one: an error, not
        // a silent inheritance.
        let (msg, _) = err_of(
            "SELECT row_number() OVER w FROM (SELECT x FROM t WINDOW w AS (ORDER BY x)) s",
        );
        assert!(msg.contains("unknown window `w`"), "{msg}");
    }

    #[test]
    fn frames_that_cannot_mean_anything_are_refused_with_an_offset() {
        for (sql, want) in [
            (
                "SELECT sum(v) OVER (ORDER BY a RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
                "RANGE frames with a numeric offset",
            ),
            (
                "SELECT sum(v) OVER (ORDER BY a GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
                "GROUPS frames are not implemented",
            ),
            (
                "SELECT sum(v) OVER (ROWS BETWEEN CURRENT ROW AND 1 PRECEDING) FROM t",
                "comes after its end",
            ),
            (
                "SELECT sum(v) OVER (ROWS BETWEEN 1 FOLLOWING AND 2 PRECEDING) FROM t",
                "comes after its end",
            ),
            // A negative offset never reaches the "non-negative" check: the
            // lexer splits it into `-` and `1`, so it is refused one step
            // earlier as "not a frame bound at all". Same outcome, and both
            // spellings are pinned so neither can start being accepted.
            (
                "SELECT sum(v) OVER (ROWS BETWEEN -1 PRECEDING AND CURRENT ROW) FROM t",
                "or a row offset",
            ),
            (
                "SELECT sum(v) OVER (ROWS BETWEEN 1.5 PRECEDING AND CURRENT ROW) FROM t",
                "non-negative integer",
            ),
            (
                "SELECT sum(v) OVER (ROWS UNBOUNDED) FROM t",
                "`PRECEDING` or `FOLLOWING` after `UNBOUNDED`",
            ),
            (
                "SELECT sum(v) OVER (ORDER BY a ROWS CURRENT ROW EXCLUDE TIES) FROM t",
                "EXCLUDE",
            ),
            ("SELECT sum(v) OVER nosuch FROM t", "unknown window `nosuch`"),
            (
                "SELECT sum(v) OVER w FROM t WINDOW w AS (ORDER BY a), w AS (ORDER BY b)",
                "declared twice",
            ),
        ] {
            let (msg, pos) = err_of(sql);
            assert!(msg.contains(want), "`{sql}`\n  got: {msg}");
            assert!(pos > 0 && pos <= sql.len(), "offset {pos} out of range for `{sql}`");
        }
    }

    #[test]
    fn a_window_call_renders_back_to_the_sql_that_made_it() {
        // The rendering is the default column name, so it has to survive the
        // worklist formatter intact -- including the frame, which is the part
        // that is not an `Expr`.
        for sql in [
            "row_number() OVER ()",
            "sum(v) OVER (PARTITION BY a, b)",
            "sum(v) OVER (ORDER BY a DESC)",
            "sum(v) OVER (PARTITION BY a ORDER BY b ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)",
            "quantile(0.9)(v) OVER (ORDER BY a NULLS LAST)",
            "lag(v, 2, 0) OVER (ORDER BY a)",
        ] {
            assert_eq!(ex(sql).to_string(), sql);
        }
    }

    #[test]
    fn a_window_spec_is_reached_by_the_expression_walk() {
        // `binder::Demand` decides which columns the scan reads by walking the
        // AST, so a PARTITION BY key it cannot see is a column the operator
        // cannot read. This is the cheapest place to catch that.
        let mut cols = Vec::new();
        ex("sum(v) OVER (PARTITION BY p ORDER BY o)").visit(&mut |e| {
            if let Expr::Column(n) = e {
                cols.push(n.to_string());
            }
        });
        cols.sort();
        assert_eq!(cols, vec!["o", "p", "v"]);
    }
}
