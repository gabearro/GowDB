//! Join reordering must move the plan without moving the answer.
//!
//! Reordering is the first rewrite in this engine that changes a plan on the
//! strength of a *guess*. Predicate pushdown is a theorem; "join `dim` before
//! `big` because `dim` has fifty rows" is an estimate, and an estimate can be
//! wrong. That makes two things worth testing, and they are not the same
//! thing:
//!
//!   * **the answer never moves.** Every shape here is written in several
//!     orders and every spelling must produce the same rows. Inner joins are
//!     the ones that get reordered; outer joins are in the file because they
//!     are the ones that must *not* be, and an outer join reordered as if it
//!     were an inner one does not crash -- it quietly loses or invents the
//!     unmatched rows;
//!   * **the plan actually moves.** A cost model that silently stopped firing
//!     would pass every answer check in this file by leaving FROM order alone,
//!     which is how an optimization becomes dead code. So the order the
//!     planner chose is read back out of `EXPLAIN`.
//!
//! The negative cases carry the same weight. A single-table query, a two-table
//! join and an already-optimal three-table join must come out byte-identical
//! to what the un-costed optimizer produces -- not "equivalent", identical --
//! because that is the only assertion that catches a search which pays for
//! itself on queries that have nothing to gain.
//!
//! ## The reachability pin
//!
//! [`the_session_plans_through_the_cost_model`] is the one test here that is
//! **expected to fail** until one line of `src/session.rs` changes. It exists
//! because everything else in this file can pass while `Session::query` still
//! plans with FROM-clause order: the cost model would be complete, tested, and
//! unreachable -- the failure mode this repository has hit eight times. Its
//! assertion message names the exact edit.

use granular::planner::binder::Binder;
use granular::planner::logical::LogicalPlan;
use granular::planner::optimizer;
use granular::sql::ast::Statement;
use granular::types::Value;
use granular::Session;

// ------------------------------------------------------------------ fixtures

/// A chain: `reg(5) - geo(50) - dim(50) - mid(5k) - big(20k)`.
///
/// Chosen so every adjacent pair is a foreign key onto a primary key, which is
/// the shape the estimator is exact about, and so the sizes span three orders
/// of magnitude -- a chain of equal-sized relations has no order worth
/// choosing and would let a broken search pass.
///
/// Small enough that the whole file runs in a debug build in under a second;
/// the *timing* claims live in the benchmark, not here.
fn chain() -> Session {
    let mut db = Session::in_memory();
    for ddl in [
        "CREATE TABLE big (id UInt64, mid UInt64, v UInt64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
        "CREATE TABLE mid (id UInt64, dim UInt64, w UInt64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
        "CREATE TABLE dim (id UInt64, geo UInt64, name String) ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
        "CREATE TABLE geo (id UInt64, region UInt64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
        "CREATE TABLE reg (id UInt64, label String) ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
    ] {
        db.execute(ddl).unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    bulk(&mut db, "big", 20_000, |i| format!("({i},{},{})", i % 5_000, i * 7 % 100));
    bulk(&mut db, "mid", 5_000, |i| format!("({i},{},{})", i % 50, i * 3 % 97));
    bulk(&mut db, "dim", 50, |i| format!("({i},{},'d{i}')", i % 50));
    bulk(&mut db, "geo", 50, |i| format!("({i},{})", i % 5));
    bulk(&mut db, "reg", 5, |i| format!("({i},'r{i}')"));
    // These tests plan against the catalog directly as well as through
    // `Session`, and only `Session` flushes the write delta first. Statistics
    // come out of part metadata, so an unflushed fixture is a fixture the
    // estimator sizes at zero rows.
    db.catalog.flush_all().unwrap();
    db
}

fn bulk(db: &mut Session, table: &str, n: u64, row: impl Fn(u64) -> String) {
    let mut sql = format!("INSERT INTO {table} VALUES ");
    for i in 0..n {
        if i > 0 {
            sql.push(',');
        }
        sql.push_str(&row(i));
    }
    db.execute(&sql).unwrap_or_else(|e| panic!("insert {table}: {e}"));
}

// ------------------------------------------------------------------- helpers

fn bound(db: &Session, sql: &str) -> LogicalPlan {
    let st = granular::sql::parse(sql).unwrap_or_else(|e| panic!("parse {sql}: {e}"));
    let Statement::Query(q) = &st[0] else { panic!("not a query: {sql}") };
    Binder::new(&db.catalog).bind_query(q).unwrap_or_else(|e| panic!("bind {sql}: {e}"))
}

/// The plan the cost model chooses.
fn costed(db: &Session, sql: &str) -> LogicalPlan {
    optimizer::optimize_costed(bound(db, sql), &db.catalog)
        .unwrap_or_else(|e| panic!("optimize {sql}: {e}"))
}

/// The plan the un-costed optimizer chooses: FROM-clause order.
fn written(db: &Session, sql: &str) -> LogicalPlan {
    optimizer::optimize(bound(db, sql)).unwrap_or_else(|e| panic!("optimize {sql}: {e}"))
}

/// Tables in the order the plan scans them, left to right.
///
/// Read out of `EXPLAIN` rather than by walking the plan, because `EXPLAIN` is
/// what a user has to reason with -- a reordering nobody can see from outside
/// is a reordering nobody can debug.
fn order(explain: &str) -> Vec<String> {
    explain
        .lines()
        .filter_map(|l| l.trim().strip_prefix("Scan default."))
        .map(|t| t.split([' ', '\t']).next().unwrap_or_default().to_string())
        .collect()
}

/// The order `Session` itself plans, through `EXPLAIN`.
fn session_order(db: &mut Session, sql: &str) -> Vec<String> {
    let rs = db
        .query(&format!("EXPLAIN {sql}"))
        .unwrap_or_else(|e| panic!("explain {sql}: {e}"));
    let text: Vec<String> = rs
        .to_values()
        .iter()
        .map(|r| match r.first() {
            Some(Value::Str(s)) => s.to_string(),
            other => panic!("EXPLAIN produced {other:?}"),
        })
        .collect();
    order(&text.join("\n"))
}

/// Every value the query produces through `Session`, row-major, rendered with
/// its `Value` variant so a path that answered `UInt(7)` where another
/// answered `Int(7)` still shows up as a difference.
fn answer(db: &mut Session, sql: &str) -> Vec<String> {
    let rs = db.query(sql).unwrap_or_else(|e| panic!("query {sql}: {e}"));
    let mut rows: Vec<String> = rs
        .to_values()
        .iter()
        .map(|r| r.iter().map(|v| format!("{v:?}")).collect::<Vec<_>>().join("|"))
        .collect();
    // A join's output order is probe-side order, which reordering is entitled
    // to change; the *set* of rows is what must not move.
    rows.sort();
    rows
}

// ------------------------------------------------------------- 1. the answer

/// Four spellings of one three-way inner join, and one answer.
#[test]
fn an_inner_join_answers_the_same_in_every_written_order() {
    let mut db = chain();
    let spellings = [
        "SELECT big.id, mid.w, dim.name FROM big JOIN mid ON big.mid = mid.id JOIN dim ON mid.dim = dim.id WHERE big.id < 40",
        "SELECT big.id, mid.w, dim.name FROM mid JOIN big ON big.mid = mid.id JOIN dim ON mid.dim = dim.id WHERE big.id < 40",
        "SELECT big.id, mid.w, dim.name FROM dim JOIN mid ON mid.dim = dim.id JOIN big ON big.mid = mid.id WHERE big.id < 40",
        "SELECT big.id, mid.w, dim.name FROM mid JOIN dim ON mid.dim = dim.id JOIN big ON big.mid = mid.id WHERE big.id < 40",
    ];
    let want = answer(&mut db, spellings[0]);
    assert_eq!(want.len(), 40, "fixture produced nothing to compare");
    for s in &spellings[1..] {
        assert_eq!(answer(&mut db, s), want, "spelling changed the answer:\n{s}");
    }
}

/// The same, five relations deep, where the search space is big enough that a
/// wrong rebuild has somewhere to hide.
#[test]
fn a_five_way_join_answers_the_same_in_every_written_order() {
    let mut db = chain();
    let a = "SELECT count(), sum(big.v), max(reg.label) FROM big JOIN mid ON big.mid = mid.id JOIN dim ON mid.dim = dim.id JOIN geo ON dim.geo = geo.id JOIN reg ON geo.region = reg.id";
    let b = "SELECT count(), sum(big.v), max(reg.label) FROM reg JOIN geo ON geo.region = reg.id JOIN dim ON dim.geo = geo.id JOIN mid ON mid.dim = dim.id JOIN big ON big.mid = mid.id";
    let c = "SELECT count(), sum(big.v), max(reg.label) FROM dim JOIN geo ON dim.geo = geo.id JOIN reg ON geo.region = reg.id JOIN mid ON mid.dim = dim.id JOIN big ON big.mid = mid.id";
    let want = answer(&mut db, a);
    assert_eq!(want, answer(&mut db, b));
    assert_eq!(want, answer(&mut db, c));
    assert!(want[0].starts_with("UInt(20000)"), "{want:?}");
}

/// A non-equi residual has to survive being re-attached to a different node.
#[test]
fn a_residual_predicate_survives_the_rewrite() {
    let mut db = chain();
    let a = "SELECT count() FROM big JOIN mid ON big.mid = mid.id AND big.v > mid.w JOIN dim ON mid.dim = dim.id";
    let b = "SELECT count() FROM dim JOIN mid ON mid.dim = dim.id JOIN big ON big.mid = mid.id AND big.v > mid.w";
    assert_eq!(answer(&mut db, a), answer(&mut db, b));
    // And it is genuinely selective, or the assertion above proves nothing.
    let all = answer(&mut db, "SELECT count() FROM big JOIN mid ON big.mid = mid.id JOIN dim ON mid.dim = dim.id");
    assert_ne!(answer(&mut db, a), all, "the residual filtered nothing");
}

/// Outer joins are where order is *not* free, so this is the case that catches
/// a cluster which swallowed one.
#[test]
fn outer_joins_keep_their_unmatched_rows() {
    let mut db = chain();
    // `mid.id >= 4000` leaves `big` rows with no partner, so LEFT must pad and
    // INNER must not -- the two answers differ, which is what makes this a
    // test rather than a tautology.
    let left = "SELECT count(), count(mid.w) FROM big LEFT JOIN mid ON big.mid = mid.id AND mid.id >= 4000";
    let inner = "SELECT count(), count(mid.w) FROM big JOIN mid ON big.mid = mid.id AND mid.id >= 4000";
    let l = answer(&mut db, left);
    assert_eq!(l, vec!["UInt(20000)|UInt(4000)".to_string()], "{l:?}");
    assert_eq!(answer(&mut db, inner), vec!["UInt(4000)|UInt(4000)".to_string()]);

    // An inner cluster hanging off an outer join: the inner part may be
    // reordered, the outer join may not move, and the padding must survive.
    let a = "SELECT count(), count(m.name) FROM big LEFT JOIN (SELECT mid.id AS id, dim.name AS name FROM mid JOIN dim ON mid.dim = dim.id WHERE mid.id >= 4000) m ON big.mid = m.id";
    let b = "SELECT count(), count(m.name) FROM big LEFT JOIN (SELECT mid.id AS id, dim.name AS name FROM dim JOIN mid ON mid.dim = dim.id WHERE mid.id >= 4000) m ON big.mid = m.id";
    assert_eq!(answer(&mut db, a), answer(&mut db, b));
    assert_eq!(answer(&mut db, a), vec!["UInt(20000)|UInt(4000)".to_string()]);
}

/// `FULL` preserves both sides and `RIGHT` preserves the right, so neither may
/// be pulled into a cluster.
#[test]
fn full_and_right_joins_are_never_reordered() {
    let db = chain();
    for sql in [
        "SELECT count() FROM dim FULL JOIN geo ON dim.geo = geo.id",
        "SELECT count() FROM big RIGHT JOIN mid ON big.mid = mid.id",
    ] {
        assert_eq!(
            written(&db, sql).explain(),
            costed(&db, sql).explain(),
            "an outer join was rewritten:\n{sql}"
        );
    }
}

// -------------------------------------------------------------- 2. the plan

/// However the three-table chain is written, the planner scans the small end
/// first.
#[test]
fn the_good_order_is_chosen_however_the_query_is_written() {
    let db = chain();
    let good = vec!["dim".to_string(), "mid".to_string(), "big".to_string()];
    for sql in [
        "SELECT count() FROM big JOIN mid ON big.mid = mid.id JOIN dim ON mid.dim = dim.id",
        "SELECT count() FROM mid JOIN big ON big.mid = mid.id JOIN dim ON mid.dim = dim.id",
        "SELECT count() FROM mid JOIN dim ON mid.dim = dim.id JOIN big ON big.mid = mid.id",
        "SELECT count() FROM dim JOIN mid ON mid.dim = dim.id JOIN big ON big.mid = mid.id",
    ] {
        assert_eq!(order(&costed(&db, sql).explain()), good, "wrong order for:\n{sql}");
    }
}

/// Four and five relations, where greedy could go wrong and the search is
/// exhaustive instead.
#[test]
fn four_and_five_way_chains_converge_on_one_order() {
    let db = chain();
    let four = "SELECT count() FROM big JOIN mid ON big.mid = mid.id JOIN dim ON mid.dim = dim.id JOIN geo ON dim.geo = geo.id";
    let four_rev = "SELECT count() FROM geo JOIN dim ON dim.geo = geo.id JOIN mid ON mid.dim = dim.id JOIN big ON big.mid = mid.id";
    assert_eq!(order(&costed(&db, four).explain()), order(&costed(&db, four_rev).explain()));
    assert_eq!(order(&costed(&db, four).explain()).last().map(String::as_str), Some("big"));

    let five = "SELECT count() FROM big JOIN mid ON big.mid = mid.id JOIN dim ON mid.dim = dim.id JOIN geo ON dim.geo = geo.id JOIN reg ON geo.region = reg.id";
    let five_rev = "SELECT count() FROM reg JOIN geo ON geo.region = reg.id JOIN dim ON dim.geo = geo.id JOIN mid ON mid.dim = dim.id JOIN big ON big.mid = mid.id";
    let a = order(&costed(&db, five).explain());
    assert_eq!(a, order(&costed(&db, five_rev).explain()));
    // The 20k-row relation is joined last whatever end it was written at:
    // every earlier position would carry it through another join.
    assert_eq!(a.last().map(String::as_str), Some("big"), "{a:?}");
}

/// A predicate that shrinks one relation must change the order it earns.
#[test]
fn a_selective_predicate_moves_the_relation_it_shrinks() {
    let db = chain();
    // Without the predicate `big` is the largest relation and goes last.
    let plain = "SELECT count() FROM mid JOIN big ON big.mid = mid.id JOIN dim ON mid.dim = dim.id";
    assert_eq!(order(&costed(&db, plain).explain()).last().map(String::as_str), Some("big"));
    // `big.id < 5` leaves five rows, and five rows belong at the front.
    let cut = "SELECT count() FROM mid JOIN big ON big.mid = mid.id JOIN dim ON mid.dim = dim.id WHERE big.id < 5";
    let o = order(&costed(&db, cut).explain());
    assert_eq!(o.first().map(String::as_str), Some("big"), "{o:?}");
}

/// A relation with no predicate connecting it must not be dragged forward: a
/// cross product priced as one is a cross product the search leaves last.
#[test]
fn a_cross_product_is_not_reordered_into_the_middle() {
    let db = chain();
    let sql = "SELECT count() FROM big JOIN mid ON big.mid = mid.id JOIN dim ON mid.dim = dim.id, reg";
    let o = order(&costed(&db, sql).explain());
    assert_eq!(o.len(), 4, "{o:?}");
    // `reg` joins nothing, so it may sit anywhere the connected chain allows
    // -- but the connected relations must still be in size order among
    // themselves, which a search that gave up would not manage.
    let chain_only: Vec<&str> =
        o.iter().map(String::as_str).filter(|t| *t != "reg").collect();
    assert_eq!(chain_only, ["dim", "mid", "big"], "{o:?}");
}

// ---------------------------------------------------------- 3. the negatives

/// Nothing with fewer than three relations may be touched at all.
///
/// `explain()` equality, not "equivalent": the un-costed plan and the costed
/// one have to be the same tree, or the search is paying for itself on queries
/// that cannot gain.
#[test]
fn small_queries_come_out_exactly_as_they_went_in() {
    let db = chain();
    for sql in [
        "SELECT count() FROM big",
        "SELECT id, v FROM big WHERE id = 133",
        "SELECT mid, count() FROM big WHERE v > 40 GROUP BY mid ORDER BY mid LIMIT 5",
        "SELECT count() FROM big JOIN mid ON big.mid = mid.id",
        "SELECT count() FROM big JOIN mid ON big.mid = mid.id WHERE mid.w > 3",
        "SELECT count() FROM big LEFT JOIN mid ON big.mid = mid.id",
        "SELECT count() FROM (SELECT id FROM big UNION ALL SELECT id FROM mid) u",
    ] {
        assert_eq!(written(&db, sql).explain(), costed(&db, sql).explain(), "moved:\n{sql}");
    }
}

/// A cluster already written in the best order is left alone, not rebuilt into
/// an equivalent tree.
#[test]
fn an_already_optimal_order_is_left_untouched() {
    let db = chain();
    for sql in [
        "SELECT count() FROM dim JOIN mid ON mid.dim = dim.id JOIN big ON big.mid = mid.id",
        "SELECT count() FROM reg JOIN geo ON geo.region = reg.id JOIN dim ON dim.geo = geo.id JOIN mid ON mid.dim = dim.id JOIN big ON big.mid = mid.id",
    ] {
        assert_eq!(
            written(&db, sql).explain(),
            costed(&db, sql).explain(),
            "the best order was rewritten:\n{sql}"
        );
    }
}

/// With no statistics the plan must degrade to what it is today, not to
/// something worse.
///
/// Empty tables are the honest version of "absent": every relation estimates
/// at the one-row floor, every order costs the same, and ties go to the
/// incumbent -- so the answer has to be the written order.
#[test]
fn absent_statistics_leave_the_written_order_in_place() {
    let mut db = Session::in_memory();
    for t in ["a", "b", "c"] {
        db.execute(&format!(
            "CREATE TABLE {t} (id UInt64, k UInt64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id"
        ))
        .unwrap();
    }
    db.catalog.flush_all().unwrap();
    let sql = "SELECT count() FROM a JOIN b ON a.k = b.id JOIN c ON b.k = c.id";
    assert_eq!(written(&db, sql).explain(), costed(&db, sql).explain());
    assert_eq!(order(&costed(&db, sql).explain()), ["a", "b", "c"]);
}

/// Statistics that are stale in the direction that matters -- the delta holds
/// rows the parts do not -- must still produce a correct answer.
///
/// It is a *plan* that goes stale, never a result: the estimator reads part
/// metadata, `Session` flushes before it plans, and this is the assertion that
/// the two facts stay connected. A future writer that plans without flushing
/// would size a freshly-loaded table at zero and this is what would notice.
#[test]
fn buffered_writes_do_not_produce_a_wrong_answer() {
    let mut db = chain();
    db.execute("INSERT INTO dim VALUES (60, 0, 'late'), (61, 0, 'later')").unwrap();
    let sql = "SELECT count() FROM dim JOIN geo ON dim.geo = geo.id JOIN reg ON geo.region = reg.id";
    let n = answer(&mut db, sql);
    assert_eq!(n, vec!["UInt(52)".to_string()], "{n:?}");
}

/// The rewrite is an equivalence, so **every** shape it touches must answer
/// exactly what it answered before it.
///
/// This is the broad net rather than a case: one battery of awkward joins, each
/// run twice -- once through `optimize`, once through `optimize_costed` -- and
/// compared row for row through the same executor. It is the assertion that
/// found the one real bug in this pass: `SELECT * FROM a JOIN b USING(k) JOIN
/// c USING(k)` binds to a join whose declared schema is one field *narrower*
/// than the block the operator produces, because `USING` merges the two copies
/// of a key in the binder's scope, and rebuilding the cluster's columns from
/// that schema dropped a column. `survey` now refuses a cluster whose root
/// schema does not describe its own block; without a shape like this in the
/// file, the failure would have been a wrong answer on a query the engine
/// answers correctly today.
#[test]
fn every_awkward_shape_answers_identically_costed_and_not() {
    let mut db = chain();
    db.execute("CREATE TABLE ka (k UInt64, x UInt64) ENGINE = MergeTree ORDER BY k").unwrap();
    db.execute("CREATE TABLE kb (k UInt64, y UInt64) ENGINE = MergeTree ORDER BY k").unwrap();
    db.execute("CREATE TABLE kc (k UInt64, z UInt64) ENGINE = MergeTree ORDER BY k").unwrap();
    db.execute("INSERT INTO ka VALUES (1,10),(2,20),(3,30)").unwrap();
    db.execute("INSERT INTO kb VALUES (1,100),(2,200)").unwrap();
    db.execute("INSERT INTO kc VALUES (1,1000),(2,2000),(3,3000)").unwrap();
    db.catalog.flush_all().unwrap();

    for sql in [
        // `USING`, nested: the schema-width case above.
        "SELECT * FROM ka JOIN kb USING(k) JOIN kc USING(k)",
        "SELECT count() FROM ka JOIN kb USING(k) JOIN kc USING(k)",
        // `USING` mixed with `ON`.
        "SELECT * FROM ka JOIN kb USING(k) JOIN kc ON ka.k = kc.k",
        // A subquery as one relation of the cluster.
        "SELECT count() FROM (SELECT id, dim FROM mid WHERE id < 900) m JOIN dim ON m.dim = dim.id JOIN big ON big.mid = m.id",
        // `IN (SELECT ...)`, which binds to a semi-join spelled as an inner one.
        "SELECT count() FROM big JOIN mid ON big.mid = mid.id WHERE mid.dim IN (SELECT id FROM dim WHERE id < 10)",
        // `NOT IN`, which binds to a LEFT join plus an IS NULL and must not be
        // pulled into a cluster.
        "SELECT count() FROM big JOIN mid ON big.mid = mid.id WHERE mid.dim NOT IN (SELECT id FROM dim WHERE id < 10)",
        // An aggregate as a relation.
        "SELECT count() FROM (SELECT dim AS d, count() AS n FROM mid GROUP BY dim) g JOIN dim ON g.d = dim.id JOIN geo ON dim.geo = geo.id",
        // A union as a relation.
        "SELECT count() FROM (SELECT id FROM dim UNION ALL SELECT id FROM geo) u JOIN mid ON mid.dim = u.id JOIN big ON big.mid = mid.id",
        // Outer join under an inner cluster, and the other way round.
        "SELECT count() FROM (SELECT mid.id AS id FROM mid LEFT JOIN dim ON mid.dim = dim.id) m JOIN big ON big.mid = m.id JOIN geo ON geo.id = m.id",
        "SELECT count() FROM geo LEFT JOIN (SELECT dim.id AS id FROM dim JOIN mid ON mid.dim = dim.id JOIN big ON big.mid = mid.id) t ON t.id = geo.id",
        // A residual, a range predicate and a self-join at once.
        "SELECT count() FROM dim d1 JOIN dim d2 ON d1.geo = d2.id AND d1.id > d2.id JOIN mid ON mid.dim = d1.id JOIN big ON big.mid = mid.id",
        // ORDER BY / LIMIT above the cluster: the projection must not disturb
        // the column indices the sort keys point at.
        "SELECT big.id, dim.name FROM big JOIN mid ON big.mid = mid.id JOIN dim ON mid.dim = dim.id ORDER BY big.id DESC LIMIT 7",
        // A window function above the cluster.
        "SELECT big.id, row_number() OVER (PARTITION BY dim.id ORDER BY big.id) AS r FROM big JOIN mid ON big.mid = mid.id JOIN dim ON mid.dim = dim.id WHERE big.id < 30 ORDER BY big.id LIMIT 9",
    ] {
        let plain = run(&db, &written(&db, sql));
        let cost = run(&db, &costed(&db, sql));
        assert_eq!(plain, cost, "the rewrite changed the answer:\n{sql}");
    }
}

/// Execute a plan through the public executor and render every cell.
fn run(db: &Session, plan: &LogicalPlan) -> Vec<String> {
    let blocks = granular::exec::operators::execute(plan, &db.catalog)
        .unwrap_or_else(|e| panic!("execute: {e}"));
    let mut out = Vec::new();
    for b in &blocks {
        for r in 0..b.rows() {
            out.push(
                (0..b.width())
                    .map(|c| format!("{:?}", b.column(c).value(r)))
                    .collect::<Vec<_>>()
                    .join("|"),
            );
        }
    }
    out.sort();
    out
}

/// A join whose relations repeat has to keep the two copies apart.
#[test]
fn a_self_join_keeps_its_copies_distinct() {
    let mut db = chain();
    let a = "SELECT count() FROM dim d1 JOIN dim d2 ON d1.geo = d2.id JOIN geo ON d2.geo = geo.id";
    let b = "SELECT count() FROM geo JOIN dim d2 ON d2.geo = geo.id JOIN dim d1 ON d1.geo = d2.id";
    assert_eq!(answer(&mut db, a), answer(&mut db, b));
    assert_eq!(answer(&mut db, a), vec!["UInt(50)".to_string()]);
}

// ------------------------------------------------------- 4. the reachability

/// **Expected to fail until `src/session.rs` is wired.**
///
/// Every other test in this file proves the cost model works. This one proves
/// a query typed into `Session` reaches it, which is a different claim and the
/// one this repository has got wrong eight times.
#[test]
fn the_session_plans_through_the_cost_model() {
    let mut db = chain();
    let sql = "SELECT count() FROM big JOIN mid ON big.mid = mid.id JOIN dim ON mid.dim = dim.id";
    let want = order(&costed(&db, sql).explain());
    let got = session_order(&mut db, sql);
    assert_eq!(
        got, want,
        "\n\n`Session` is still planning with FROM-clause order.\n\
         The cost model is complete and unreachable. One line closes it, in \
         `Session::plan_in` in src/session.rs:\n\n  \
         -        optimizer::optimize(plan)\n  \
         +        optimizer::optimize_costed(plan, &self.catalog)\n\n\
         `&self.catalog` is already borrowed on the line above it.\n"
    );
}

/// Planning stays in microseconds.
///
/// Not a benchmark -- the numbers live in the commit message -- but a ceiling
/// loose enough to survive a debug build on a loaded machine and tight enough
/// to catch a search that became exponential. A five-relation cluster is 2^5
/// subsets; a bug that let `MAX_CLUSTER` through unbounded would be 2^60.
#[test]
fn planning_a_five_way_join_stays_cheap() {
    let db = chain();
    let sql = "SELECT count() FROM big JOIN mid ON big.mid = mid.id JOIN dim ON mid.dim = dim.id JOIN geo ON dim.geo = geo.id JOIN reg ON geo.region = reg.id";
    // Warm the catalog and the parser.
    costed(&db, sql);
    let t = std::time::Instant::now();
    for _ in 0..200 {
        costed(&db, sql);
    }
    let each = t.elapsed().as_secs_f64() * 1e6 / 200.0;
    assert!(each < 2_000.0, "planning cost {each:.1} us per query");
}
