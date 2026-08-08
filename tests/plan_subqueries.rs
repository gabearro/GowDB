//! `x IN (SELECT ...)` and `EXISTS (SELECT ...)` as plan nodes.
//!
//! Everything here goes through the binder directly rather than through
//! `Session::query`, and that is deliberate rather than convenient:
//! `Session::resolve_subqueries` still folds a membership test into a literal
//! `IN` list *before* the binder sees it, so a test written against the session
//! would be measuring the fold, not the join. The session hunk that stops it is
//! named in this change's report; when it lands, these queries reach the same
//! plans from `Session::query` and the helpers below can collapse into it.
//!
//! Two of the answers asserted here are ones the fold gets **wrong**, which is
//! the other reason not to route through it:
//!
//!   * `NULL NOT IN (SELECT ... -- no rows)` is TRUE. The fold produces an
//!     empty `InList`, which answers NULL for a NULL input and drops the row.
//!   * `NOT EXISTS (SELECT * FROM t)` is legal for any width of `t`. The fold
//!     reads column 0 of a result it has already computed and refuses more than
//!     one column.
//!
//! Both are checked against sqlite, whose answers are the ones asserted.

use std::time::Instant;

use granular::exec::operators;
use granular::planner::{binder::Binder, explain_physical, optimizer, LogicalPlan};
use granular::sql::Statement;
use granular::types::Value;
use granular::{Result, Session};

// ------------------------------------------------------------------ fixtures

/// `o` is the outer relation, `s*` the subqueries. `x`/`y` are nullable and
/// `id`/`m` are not, because which of the two a column is decides whether the
/// plan grows a NULL census -- so a fixture with only one kind cannot tell the
/// two plans apart.
fn db() -> Session {
    let mut db = Session::in_memory();
    for stmt in [
        "CREATE TABLE o (id UInt64, x Nullable(Int64), s Nullable(String)) \
         ENGINE = MergeTree ORDER BY id",
        // `withnull` yields a NULL, `nonull` does not, `empty` yields nothing.
        "CREATE TABLE withnull (y Nullable(Int64), m UInt64) ENGINE = MergeTree ORDER BY m",
        "CREATE TABLE nonull (y Nullable(Int64), m UInt64) ENGINE = MergeTree ORDER BY m",
        "CREATE TABLE empty (y Nullable(Int64), m UInt64) ENGINE = MergeTree ORDER BY m",
        "INSERT INTO o VALUES (1, 1, 'a'), (2, 2, 'b'), (3, NULL, NULL), (4, 4, 'd'), \
         (5, 2, 'b'), (6, 7, 'g')",
        // Duplicated 2s on purpose: a semi-join that forgot to deduplicate its
        // right side returns row 2 and row 5 twice.
        "INSERT INTO withnull VALUES (2, 1), (2, 2), (NULL, 3), (4, 4)",
        "INSERT INTO nonull VALUES (2, 1), (2, 2), (4, 4)",
    ] {
        db.execute(stmt).unwrap_or_else(|e| panic!("{stmt}: {e}"));
    }
    // `Session::query` flushes the delta before every statement and these
    // tests do not go through it, so an unflushed fixture is one the plans
    // below cannot see at all.
    db.catalog.flush_all().unwrap();
    db
}

fn plan(db: &Session, sql: &str) -> Result<LogicalPlan> {
    let st = granular::sql::parse(sql).unwrap_or_else(|e| panic!("parse {sql}: {e}"));
    let Statement::Query(q) = &st[0] else { panic!("not a query: {sql}") };
    optimizer::optimize(Binder::new(&db.catalog).bind_query(q)?)
}

fn explain(db: &Session, sql: &str) -> String {
    plan(db, sql).unwrap_or_else(|e| panic!("bind {sql}: {e}")).explain()
}

/// Every value the query produces, row-major.
fn rows(db: &Session, sql: &str) -> Vec<Vec<Value>> {
    let plan = plan(db, sql).unwrap_or_else(|e| panic!("bind {sql}: {e}"));
    let blocks = operators::execute(&plan, &db.catalog)
        .unwrap_or_else(|e| panic!("execute {sql}: {e}"));
    let mut out = Vec::new();
    for b in &blocks {
        for r in 0..b.rows() {
            out.push((0..b.width()).map(|c| b.column(c).value(r)).collect());
        }
    }
    out
}

/// The single column of `SELECT id FROM o WHERE ...`, as `u64`s.
fn ids(db: &Session, sql: &str) -> Vec<u64> {
    rows(db, sql)
        .into_iter()
        .map(|r| match r[0] {
            Value::UInt(u) => u,
            ref v => panic!("not an id: {v}"),
        })
        .collect()
}

fn err(db: &Session, sql: &str) -> String {
    match plan(db, sql) {
        Ok(p) => panic!("expected an error for `{sql}`, got:\n{}", p.explain()),
        Err(e) => e.to_string(),
    }
}

// --------------------------------------------- the four three-valued corners

#[test]
fn in_with_a_null_in_the_subquery_is_null_and_not_false() {
    let db = db();
    // Case 1. `1 IN (2, 2, NULL, 4)` is NULL, not FALSE -- but `WHERE` keeps
    // only TRUE, so the observable answer is the same and the semi-join is
    // exact: a NULL key matches nothing. Rows 2 and 5 have x = 2 and match;
    // row 4 has x = 4 and matches; everything else, NULL x included, is out.
    assert_eq!(ids(&db, "SELECT id FROM o WHERE x IN (SELECT y FROM withnull)"), [2, 4, 5]);
    // And the duplicate 2s in the subquery do not duplicate rows 2 and 5.
    assert_eq!(ids(&db, "SELECT id FROM o WHERE x IN (SELECT y FROM nonull)"), [2, 4, 5]);
}

#[test]
fn not_in_with_a_null_anywhere_in_the_subquery_keeps_nothing() {
    let db = db();
    // Case 2, the one that surprises people: `1 NOT IN (2, 2, NULL, 4)` is
    // NULL, not TRUE, because the NULL *might* have been a 1. That holds for
    // every row however unrelated it is to the NULL, so the answer is empty.
    assert_eq!(ids(&db, "SELECT id FROM o WHERE x NOT IN (SELECT y FROM withnull)"), []);
    // Same subquery minus the NULL: now the ordinary anti-join answer.
    assert_eq!(ids(&db, "SELECT id FROM o WHERE x NOT IN (SELECT y FROM nonull)"), [1, 6]);
    // Filtering the NULL out of the *subquery* is the user's way to say what
    // they meant, and it has to work.
    assert_eq!(
        ids(&db, "SELECT id FROM o WHERE x NOT IN (SELECT y FROM withnull WHERE y IS NOT NULL)"),
        [1, 6]
    );
}

#[test]
fn exists_is_never_null() {
    let db = db();
    // Case 3. The subquery yields a NULL and `EXISTS` does not care: it is a
    // row count, not a value. Every outer row survives, NULL x included.
    assert_eq!(ids(&db, "SELECT id FROM o WHERE EXISTS (SELECT y FROM withnull)"), [1, 2, 3, 4, 5, 6]);
    assert_eq!(ids(&db, "SELECT id FROM o WHERE NOT EXISTS (SELECT y FROM withnull)"), []);
    // A subquery that yields only NULLs still *exists*.
    assert_eq!(
        ids(&db, "SELECT id FROM o WHERE EXISTS (SELECT y FROM withnull WHERE y IS NULL)"),
        [1, 2, 3, 4, 5, 6]
    );
    // Any width, because existence does not read a value. The fold refused
    // this one outright.
    assert_eq!(ids(&db, "SELECT id FROM o WHERE NOT EXISTS (SELECT * FROM empty)"), [1, 2, 3, 4, 5, 6]);
}

#[test]
fn an_empty_subquery_is_false_for_in_and_true_for_not_in() {
    let db = db();
    // Case 4, and the row that matters is id 3, whose `x` is NULL:
    // `NULL NOT IN ()` is TRUE, vacuously -- there is no `y` to be unknown
    // about. sqlite agrees; the literal splice answered NULL and dropped it.
    assert_eq!(ids(&db, "SELECT id FROM o WHERE x IN (SELECT y FROM empty)"), []);
    assert_eq!(
        ids(&db, "SELECT id FROM o WHERE x NOT IN (SELECT y FROM empty)"),
        [1, 2, 3, 4, 5, 6]
    );
    assert_eq!(ids(&db, "SELECT id FROM o WHERE EXISTS (SELECT y FROM empty)"), []);
    assert_eq!(
        ids(&db, "SELECT id FROM o WHERE NOT EXISTS (SELECT y FROM empty)"),
        [1, 2, 3, 4, 5, 6]
    );
    // A subquery emptied by its own predicate, not by an empty table: the
    // census has to come from the plan, not from the catalog's row count.
    assert_eq!(
        ids(&db, "SELECT id FROM o WHERE x NOT IN (SELECT y FROM withnull WHERE y > 1000)"),
        [1, 2, 3, 4, 5, 6]
    );
}

#[test]
fn strings_and_expressions_take_the_same_four_answers() {
    let db = db();
    assert_eq!(ids(&db, "SELECT id FROM o WHERE s IN (SELECT t FROM (SELECT 'b' AS t))"), [2, 5]);
    // A probe that is not a bare column gets projected into a key column.
    assert_eq!(ids(&db, "SELECT id FROM o WHERE x + 1 IN (SELECT y FROM nonull)"), [1]);
    assert_eq!(ids(&db, "SELECT id FROM o WHERE x + 1 NOT IN (SELECT y FROM nonull)"), [2, 4, 5, 6]);
}

// -------------------------------------------------------------- plan shapes

#[test]
fn the_subquery_is_a_relation_in_the_plan_not_a_literal_list() {
    let db = db();
    let e = explain(&db, "SELECT id FROM o WHERE x IN (SELECT y FROM nonull)");
    assert!(e.contains("InnerJoin"), "{e}");
    assert!(e.contains("Scan default.nonull"), "the subquery kept its own scan:\n{e}");
    assert!(!e.contains("IN ("), "no literal list survived:\n{e}");
    // Its own projection, narrowed by its own demand: the subquery reads `y`
    // and not `m`, which a spliced value list could never have expressed.
    assert!(e.contains("Scan default.nonull [y]"), "{e}");
}

#[test]
fn the_null_census_is_built_only_when_a_null_is_possible() {
    let mut db = db();
    db.execute("CREATE TABLE plain (k UInt64) ENGINE = MergeTree ORDER BY k").unwrap();
    db.execute("INSERT INTO plain VALUES (1), (2)").unwrap();
    db.catalog.flush_all().unwrap();

    // Neither side can hold a NULL, so cases 2 and 4 cannot fire and the plan
    // is one anti-join over one pass of the subquery.
    let e = explain(&db, "SELECT id FROM o WHERE id NOT IN (SELECT k FROM plain)");
    assert_eq!(e.matches("Join").count(), 1, "no census join wanted:\n{e}");
    assert_eq!(e.matches("Scan default.plain").count(), 1, "one pass wanted:\n{e}");

    // One nullable side and the census appears -- a second pass, which is what
    // being right about `NOT IN` costs.
    let e = explain(&db, "SELECT id FROM o WHERE id NOT IN (SELECT y FROM nonull)");
    assert_eq!(e.matches("Join").count(), 2, "{e}");
    assert!(e.contains("count(), count(y#0)"), "{e}");

    // `IN` never needs it: a NULL on either side simply fails to match.
    let e = explain(&db, "SELECT id FROM o WHERE x IN (SELECT y FROM nonull)");
    assert_eq!(e.matches("Join").count(), 1, "{e}");
}

#[test]
fn an_ordinary_conjunct_stays_below_the_join() {
    let db = db();
    // If it did not, the join would build over rows a filter was about to
    // throw away -- and the scan would lose the predicate it can evaluate
    // before decoding anything else.
    let e = explain(&db, "SELECT id FROM o WHERE id > 3 AND x IN (SELECT y FROM nonull)");
    assert!(e.contains("prewhere=(id#0 > 3)"), "the filter reached the scan:\n{e}");
    assert_eq!(ids(&db, "SELECT id FROM o WHERE id > 3 AND x IN (SELECT y FROM nonull)"), [4, 5]);
}

#[test]
fn several_membership_tests_in_one_where_stack() {
    let db = db();
    let sql = "SELECT id FROM o WHERE x IN (SELECT y FROM nonull) \
               AND id NOT IN (SELECT m FROM nonull) AND EXISTS (SELECT y FROM withnull)";
    let e = explain(&db, sql);
    assert_eq!(e.matches("Join").count(), 3, "a semi, an anti, an exists:\n{e}");
    // x in {2,4} -> ids 2,4,5; minus ids in {1,2,4} -> id 5.
    assert_eq!(ids(&db, sql), [5]);
}

#[test]
fn a_membership_test_nests_inside_another_subquery() {
    let db = db();
    let sql = "SELECT id FROM o WHERE x IN (SELECT y FROM nonull WHERE y IN (SELECT m FROM withnull))";
    // nonull.y in {2,4} intersected with withnull.m {1,2,3,4} -> {2,4}
    assert_eq!(ids(&db, sql), [2, 4, 5]);
    let e = explain(&db, sql);
    assert!(e.contains("Scan default.withnull"), "the inner subquery is a relation too:\n{e}");
}

#[test]
fn a_membership_test_outside_a_where_conjunct_is_still_refused() {
    let db = db();
    // Inside an OR the test has to produce a value per row, and a semi-join
    // produces rows. Refused rather than answered wrongly -- the session layer
    // folds these, which is why the message names the position and not the
    // feature.
    for sql in [
        "SELECT id FROM o WHERE id = 1 OR x IN (SELECT y FROM nonull)",
        "SELECT x IN (SELECT y FROM nonull) FROM o",
        "SELECT id FROM o WHERE NOT (x IN (SELECT y FROM nonull))",
        "SELECT id FROM o GROUP BY id HAVING EXISTS (SELECT y FROM nonull)",
    ] {
        let m = err(&db, sql);
        assert!(m.contains("semi-join") && m.contains("per-row value"), "{sql}: {m}");
    }
}

#[test]
fn arity_and_type_are_checked_where_the_splice_used_to_check_them() {
    let db = db();
    assert!(err(&db, "SELECT id FROM o WHERE x IN (SELECT y, m FROM nonull)")
        .contains("exactly one column"));
    // A String probe against an Int64 subquery was a bind error under the
    // literal splice (`coerce_literal`), and has to stay one rather than
    // become a join that quietly matches nothing.
    assert!(err(&db, "SELECT id FROM o WHERE s IN (SELECT y FROM nonull)").contains("String"));
}

// --------------------------------------------------------- size and EXPLAIN

/// 131072 rows, built by doubling: 17 statements instead of a megabyte of
/// `VALUES`, and it exercises the multi-part path a single insert would not.
fn big(rows: u32) -> Session {
    let mut db = Session::in_memory();
    db.execute("CREATE TABLE big (k UInt64) ENGINE = MergeTree ORDER BY k").unwrap();
    db.execute("CREATE TABLE outer_ (k UInt64) ENGINE = MergeTree ORDER BY k").unwrap();
    db.execute("INSERT INTO big VALUES (0)").unwrap();
    let mut n = 1u32;
    while n < rows {
        db.execute(&format!("INSERT INTO big SELECT k + {n} FROM big")).unwrap();
        n *= 2;
    }
    db.execute("INSERT INTO outer_ SELECT k FROM big").unwrap();
    db.catalog.flush_all().unwrap();
    db
}

#[test]
fn a_subquery_larger_than_any_sane_in_list() {
    let db = big(1 << 17);
    // 131072 values. As a literal `IN` list this is 131072 `Value`s held in
    // memory and re-projected into probe lanes once per block; as a semi-join
    // it is a hash table built once over a stream. The point of the test is
    // that it is an ordinary query rather than a memory event.
    let got = rows(&db, "SELECT count() FROM outer_ WHERE k IN (SELECT k FROM big)");
    assert_eq!(got[0][0], Value::UInt(1 << 17));
    let got = rows(&db, "SELECT count() FROM outer_ WHERE k NOT IN (SELECT k FROM big WHERE k < 1000)");
    assert_eq!(got[0][0], Value::UInt((1 << 17) - 1000));
    // And the plan holds no list at all.
    let e = explain(&db, "SELECT count() FROM outer_ WHERE k IN (SELECT k FROM big)");
    assert!(!e.contains(", 500,"), "a literal list leaked into the plan:\n{e}");
}

#[test]
fn explain_describes_the_subquery_instead_of_running_it() {
    let db = big(1 << 17);
    let sql = "SELECT count() FROM outer_ WHERE k IN (SELECT k FROM big)";

    // The cost of actually answering it, as the yardstick. Measured in the
    // same process on the same data, so no cross-machine constant is pinned.
    let t0 = Instant::now();
    let _ = rows(&db, sql);
    let ran = t0.elapsed();

    let t0 = Instant::now();
    let p = plan(&db, sql).unwrap();
    let text = explain_physical(&p, &db.catalog).unwrap();
    let described = t0.elapsed();

    // The subquery is *in* the description, which is the half that says the
    // node exists...
    assert!(text.contains("Scan default.big"), "{text}");
    // ...and describing it costs a small fraction of running it, which is the
    // half that says nothing ran. A factor of eight, not two: the machine
    // swings and the assertion should fail on a regression, not on load.
    assert!(
        described * 8 < ran,
        "EXPLAIN took {described:?} against a {ran:?} query -- it looks like it ran"
    );
}

#[test]
fn the_same_answer_as_the_join_a_user_would_have_written_by_hand() {
    let db = db();
    // The semi-join is not a new algorithm, it is the join the docs tell you to
    // write instead. If the two ever disagree, one of them is wrong.
    for (sub, hand) in [
        (
            "SELECT id FROM o WHERE x IN (SELECT y FROM nonull)",
            "SELECT o.id FROM o JOIN (SELECT DISTINCT y FROM nonull) z ON o.x = z.y",
        ),
        (
            "SELECT id FROM o WHERE s IN (SELECT t FROM (SELECT 'b' AS t))",
            "SELECT o.id FROM o JOIN (SELECT DISTINCT 'b' AS t) z ON o.s = z.t",
        ),
    ] {
        assert_eq!(ids(&db, sub), ids(&db, hand), "{sub}");
    }
}
