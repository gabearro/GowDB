//! End-to-end SQL tests: text in, values out, through the real pipeline.
//!
//! These are the acceptance tests for the engine. Everything here goes through
//! `Session::query`, so a passing test exercises parser -> binder -> optimizer
//! -> executor -> storage.

use granular::types::Value;
use granular::{Result, Session};

// ------------------------------------------------------------------ helpers

fn db() -> Session {
    Session::in_memory()
}

/// A session with a small, fully-known `events` table.
fn seeded() -> Session {
    let mut s = db();
    s.execute(
        "CREATE TABLE events (
            id      UInt64,
            country String,
            latency UInt32,
            bytes   Int64,
            ok      Bool
        ) ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
    )
    .unwrap();
    s.execute(
        "INSERT INTO events VALUES
            (1, 'US', 100, 1000, true),
            (2, 'DE',  50, 2000, true),
            (3, 'US', 300,  500, false),
            (4, 'FR', 150, 1500, true),
            (5, 'DE', 250,  750, false),
            (6, 'US',  75, 3000, true)",
    )
    .unwrap();
    s
}

fn rows(s: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    s.query(sql)
        .unwrap_or_else(|e| panic!("query failed: {sql}\n  {e}"))
        .to_values()
}

fn scalar(s: &mut Session, sql: &str) -> Value {
    s.query(sql)
        .unwrap_or_else(|e| panic!("query failed: {sql}\n  {e}"))
        .scalar()
        .unwrap_or_else(|| panic!("no scalar result for: {sql}"))
}

fn col0(s: &mut Session, sql: &str) -> Vec<Value> {
    rows(s, sql).into_iter().map(|r| r[0].clone()).collect()
}

// -------------------------------------------------------------------- DDL

#[test]
fn create_insert_select_roundtrip() {
    let mut s = seeded();
    assert_eq!(scalar(&mut s, "SELECT count() FROM events"), Value::UInt(6));
    let ids = col0(&mut s, "SELECT id FROM events ORDER BY id");
    assert_eq!(
        ids,
        (1..=6).map(Value::UInt).collect::<Vec<_>>()
    );
}

#[test]
fn describe_and_show() {
    let mut s = seeded();
    let d = rows(&mut s, "DESCRIBE events");
    assert_eq!(d.len(), 5);
    assert_eq!(d[0][0], Value::str("id"));
    assert_eq!(d[0][1], Value::str("UInt64"));
    assert_eq!(d[1][1], Value::str("String"));

    let t = col0(&mut s, "SHOW TABLES");
    assert_eq!(t, vec![Value::str("events")]);

    let ddl = scalar(&mut s, "SHOW CREATE TABLE events");
    let ddl = ddl.as_str().unwrap().to_string();
    assert!(ddl.contains("CREATE TABLE"), "{ddl}");
    assert!(ddl.contains("ENGINE = MergeTree"), "{ddl}");
    assert!(ddl.contains("ORDER BY"), "{ddl}");
}

#[test]
fn drop_and_if_exists() {
    let mut s = seeded();
    s.execute("DROP TABLE events").unwrap();
    assert!(s.query("SELECT count() FROM events").is_err());
    assert!(s.execute("DROP TABLE events").is_err());
    s.execute("DROP TABLE IF EXISTS events").unwrap();
}

#[test]
fn databases_are_isolated() {
    let mut s = db();
    s.execute("CREATE DATABASE analytics").unwrap();
    s.execute("CREATE TABLE analytics.t (id UInt64) ENGINE = MergeTree ORDER BY id")
        .unwrap();
    s.execute("INSERT INTO analytics.t VALUES (1), (2)").unwrap();
    assert_eq!(scalar(&mut s, "SELECT count() FROM analytics.t"), Value::UInt(2));
    assert!(s.query("SELECT count() FROM t").is_err(), "not visible in `default`");
    s.execute("USE analytics").unwrap();
    assert_eq!(scalar(&mut s, "SELECT count() FROM t"), Value::UInt(2));
}

#[test]
fn mergetree_requires_order_by() {
    let mut s = db();
    assert!(s
        .execute("CREATE TABLE t (id UInt64) ENGINE = MergeTree")
        .is_err());
    // tuple() is the ClickHouse escape hatch for "no ordering"
    s.execute("CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY tuple()")
        .unwrap();
}

// ------------------------------------------------------------- projection

#[test]
fn wildcard_and_expressions() {
    let mut s = seeded();
    let r = rows(&mut s, "SELECT * FROM events ORDER BY id LIMIT 1");
    assert_eq!(r[0].len(), 5);
    assert_eq!(r[0][0], Value::UInt(1));
    assert_eq!(r[0][1], Value::str("US"));

    let r = col0(&mut s, "SELECT latency * 2 FROM events ORDER BY id LIMIT 2");
    assert_eq!(r, vec![Value::UInt(200), Value::UInt(100)]);
}

#[test]
fn aliases_are_visible_downstream() {
    let mut s = seeded();
    let r = rows(
        &mut s,
        "SELECT latency AS lat, bytes AS b FROM events WHERE lat > 200 ORDER BY lat",
    );
    assert_eq!(r.len(), 2);
    assert_eq!(r[0][0], Value::UInt(250));
    assert_eq!(r[1][0], Value::UInt(300));
}

#[test]
fn arithmetic_and_precedence() {
    let mut s = db();
    assert_eq!(scalar(&mut s, "SELECT 1 + 2 * 3"), Value::Int(7));
    assert_eq!(scalar(&mut s, "SELECT (1 + 2) * 3"), Value::Int(9));
    assert_eq!(scalar(&mut s, "SELECT 10 - 3 - 2"), Value::Int(5));
    assert_eq!(scalar(&mut s, "SELECT 7 % 3"), Value::Int(1));
    assert_eq!(scalar(&mut s, "SELECT -5 + 10"), Value::Int(5));
}

// ---------------------------------------------------------------- filters

#[test]
fn comparison_and_logic() {
    let mut s = seeded();
    assert_eq!(
        scalar(&mut s, "SELECT count() FROM events WHERE latency > 100"),
        Value::UInt(3)
    );
    assert_eq!(
        scalar(
            &mut s,
            "SELECT count() FROM events WHERE latency > 100 AND country = 'US'"
        ),
        Value::UInt(1)
    );
    assert_eq!(
        scalar(
            &mut s,
            "SELECT count() FROM events WHERE country = 'DE' OR country = 'FR'"
        ),
        Value::UInt(3)
    );
    assert_eq!(
        scalar(&mut s, "SELECT count() FROM events WHERE NOT ok"),
        Value::UInt(2)
    );
}

#[test]
fn in_between_like() {
    let mut s = seeded();
    assert_eq!(
        scalar(&mut s, "SELECT count() FROM events WHERE id IN (1, 3, 5)"),
        Value::UInt(3)
    );
    assert_eq!(
        scalar(&mut s, "SELECT count() FROM events WHERE id NOT IN (1, 3, 5)"),
        Value::UInt(3)
    );
    assert_eq!(
        scalar(&mut s, "SELECT count() FROM events WHERE latency BETWEEN 75 AND 150"),
        Value::UInt(3)
    );
    assert_eq!(
        scalar(&mut s, "SELECT count() FROM events WHERE country LIKE 'U%'"),
        Value::UInt(3)
    );
    assert_eq!(
        scalar(&mut s, "SELECT count() FROM events WHERE country LIKE '_E'"),
        Value::UInt(2)
    );
}

#[test]
fn three_valued_logic_excludes_nulls() {
    let mut s = db();
    s.execute("CREATE TABLE t (id UInt64, v Nullable(Int64)) ENGINE = MergeTree ORDER BY id")
        .unwrap();
    s.execute("INSERT INTO t VALUES (1, 10), (2, NULL), (3, 30)").unwrap();

    // NULL fails every comparison, including `!=`
    assert_eq!(scalar(&mut s, "SELECT count() FROM t WHERE v > 5"), Value::UInt(2));
    assert_eq!(scalar(&mut s, "SELECT count() FROM t WHERE v != 10"), Value::UInt(1));
    assert_eq!(scalar(&mut s, "SELECT count() FROM t WHERE v IS NULL"), Value::UInt(1));
    assert_eq!(
        scalar(&mut s, "SELECT count() FROM t WHERE v IS NOT NULL"),
        Value::UInt(2)
    );
    // count(col) skips nulls, count() does not
    assert_eq!(scalar(&mut s, "SELECT count(v) FROM t"), Value::UInt(2));
    assert_eq!(scalar(&mut s, "SELECT count() FROM t"), Value::UInt(3));
}

// ------------------------------------------------------------- aggregates

#[test]
fn global_aggregates() {
    let mut s = seeded();
    assert_eq!(scalar(&mut s, "SELECT sum(bytes) FROM events"), Value::Int(8750));
    assert_eq!(scalar(&mut s, "SELECT min(latency) FROM events"), Value::UInt(50));
    assert_eq!(scalar(&mut s, "SELECT max(latency) FROM events"), Value::UInt(300));
    let avg = scalar(&mut s, "SELECT avg(latency) FROM events");
    assert!((avg.as_f64().unwrap() - 154.1666).abs() < 0.01, "{avg}");
}

#[test]
fn group_by_with_ordering() {
    let mut s = seeded();
    let r = rows(
        &mut s,
        "SELECT country, count() AS n, sum(bytes) AS b
         FROM events GROUP BY country ORDER BY country",
    );
    assert_eq!(r.len(), 3);
    assert_eq!(r[0], vec![Value::str("DE"), Value::UInt(2), Value::Int(2750)]);
    assert_eq!(r[1], vec![Value::str("FR"), Value::UInt(1), Value::Int(1500)]);
    assert_eq!(r[2], vec![Value::str("US"), Value::UInt(3), Value::Int(4500)]);
}

#[test]
fn aggregate_inside_an_expression() {
    let mut s = seeded();
    // The binder must split this into Aggregate(sum(bytes)) then Project(*2).
    assert_eq!(scalar(&mut s, "SELECT sum(bytes) * 2 FROM events"), Value::Int(17500));
    let r = rows(
        &mut s,
        "SELECT country, sum(bytes) / count() AS avg_bytes
         FROM events GROUP BY country ORDER BY country",
    );
    assert_eq!(r.len(), 3);
    assert!((r[0][1].as_f64().unwrap() - 1375.0).abs() < 1e-6);
}

#[test]
fn having_filters_groups() {
    let mut s = seeded();
    let r = rows(
        &mut s,
        "SELECT country, count() AS n FROM events GROUP BY country HAVING n > 1 ORDER BY country",
    );
    assert_eq!(r.len(), 2);
    assert_eq!(r[0][0], Value::str("DE"));
    assert_eq!(r[1][0], Value::str("US"));
}

#[test]
fn count_star_on_empty_table_is_zero() {
    let mut s = db();
    s.execute("CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id")
        .unwrap();
    assert_eq!(scalar(&mut s, "SELECT count() FROM t"), Value::UInt(0));
    // ...but a grouped aggregate over no rows produces no groups
    assert!(rows(&mut s, "SELECT id, count() FROM t GROUP BY id").is_empty());
}

#[test]
fn conditional_aggregates() {
    let mut s = seeded();
    assert_eq!(
        scalar(&mut s, "SELECT countIf(ok) FROM events"),
        Value::UInt(4)
    );
    assert_eq!(
        scalar(&mut s, "SELECT sumIf(bytes, country = 'US') FROM events"),
        Value::Int(4500)
    );
}

#[test]
fn distinct_counting() {
    let mut s = seeded();
    assert_eq!(
        scalar(&mut s, "SELECT uniqExact(country) FROM events"),
        Value::UInt(3)
    );
    assert_eq!(
        scalar(&mut s, "SELECT count(DISTINCT country) FROM events"),
        Value::UInt(3)
    );
}

// ------------------------------------------------------- sort / limit / distinct

#[test]
fn order_by_directions_and_limit() {
    let mut s = seeded();
    let r = col0(&mut s, "SELECT latency FROM events ORDER BY latency DESC LIMIT 3");
    assert_eq!(r, vec![Value::UInt(300), Value::UInt(250), Value::UInt(150)]);

    let r = col0(&mut s, "SELECT id FROM events ORDER BY id LIMIT 2 OFFSET 2");
    assert_eq!(r, vec![Value::UInt(3), Value::UInt(4)]);

    // ClickHouse's reversed two-argument form: LIMIT offset, count
    let r = col0(&mut s, "SELECT id FROM events ORDER BY id LIMIT 2, 2");
    assert_eq!(r, vec![Value::UInt(3), Value::UInt(4)]);
}

#[test]
fn multi_key_ordering() {
    let mut s = seeded();
    let r = rows(
        &mut s,
        "SELECT country, latency FROM events ORDER BY country ASC, latency DESC",
    );
    assert_eq!(r[0], vec![Value::str("DE"), Value::UInt(250)]);
    assert_eq!(r[1], vec![Value::str("DE"), Value::UInt(50)]);
    assert_eq!(r[2], vec![Value::str("FR"), Value::UInt(150)]);
    assert_eq!(r[3], vec![Value::str("US"), Value::UInt(300)]);
}

#[test]
fn distinct_dedups() {
    let mut s = seeded();
    let r = col0(&mut s, "SELECT DISTINCT country FROM events ORDER BY country");
    assert_eq!(r, vec![Value::str("DE"), Value::str("FR"), Value::str("US")]);
}

// ------------------------------------------------------------- functions

#[test]
fn string_functions() {
    let mut s = db();
    assert_eq!(scalar(&mut s, "SELECT lower('HeLLo')"), Value::str("hello"));
    assert_eq!(scalar(&mut s, "SELECT upper('hello')"), Value::str("HELLO"));
    assert_eq!(scalar(&mut s, "SELECT length('hello')"), Value::UInt(5));
    assert_eq!(scalar(&mut s, "SELECT concat('a', 'b', 'c')"), Value::str("abc"));
    // ClickHouse substring is 1-based
    assert_eq!(scalar(&mut s, "SELECT substring('abcdef', 2, 3)"), Value::str("bcd"));
}

#[test]
fn math_functions() {
    let mut s = db();
    assert_eq!(scalar(&mut s, "SELECT abs(-5)"), Value::Int(5));
    assert_eq!(scalar(&mut s, "SELECT floor(3.7)").as_f64().unwrap(), 3.0);
    assert_eq!(scalar(&mut s, "SELECT ceil(3.2)").as_f64().unwrap(), 4.0);
    assert_eq!(scalar(&mut s, "SELECT round(3.456, 2)").as_f64().unwrap(), 3.46);
    assert_eq!(scalar(&mut s, "SELECT greatest(1, 9, 4)").as_f64().unwrap(), 9.0);
}

#[test]
fn conditionals() {
    let mut s = seeded();
    let r = col0(
        &mut s,
        "SELECT CASE WHEN latency > 200 THEN 'slow' ELSE 'fast' END
         FROM events ORDER BY id",
    );
    assert_eq!(r[0], Value::str("fast"));
    assert_eq!(r[2], Value::str("slow"));

    assert_eq!(scalar(&mut s, "SELECT if(1 > 0, 'yes', 'no')"), Value::str("yes"));
    assert_eq!(scalar(&mut s, "SELECT ifNull(NULL, 7)"), Value::Int(7));
    assert_eq!(scalar(&mut s, "SELECT coalesce(NULL, NULL, 3)"), Value::Int(3));
}

#[test]
fn date_handling() {
    let mut s = db();
    s.execute("CREATE TABLE d (id UInt64, day Date, at DateTime) ENGINE = MergeTree ORDER BY id")
        .unwrap();
    s.execute(
        "INSERT INTO d VALUES
            (1, '2024-01-15', '2024-01-15 13:45:30'),
            (2, '2024-06-30', '2024-06-30 00:00:00')",
    )
    .unwrap();

    assert_eq!(scalar(&mut s, "SELECT toYear(day) FROM d WHERE id = 1"), Value::UInt(2024));
    assert_eq!(scalar(&mut s, "SELECT toMonth(day) FROM d WHERE id = 2"), Value::UInt(6));
    assert_eq!(scalar(&mut s, "SELECT toHour(at) FROM d WHERE id = 1"), Value::UInt(13));

    // A string literal compared to a Date column must be coerced, not rejected.
    assert_eq!(
        scalar(&mut s, "SELECT count() FROM d WHERE day = '2024-01-15'"),
        Value::UInt(1)
    );
    assert_eq!(
        scalar(&mut s, "SELECT count() FROM d WHERE day > '2024-03-01'"),
        Value::UInt(1)
    );
}

#[test]
fn casts() {
    let mut s = db();
    assert_eq!(scalar(&mut s, "SELECT CAST('42' AS Int64)"), Value::Int(42));
    assert_eq!(scalar(&mut s, "SELECT CAST(42 AS String)"), Value::str("42"));
    assert_eq!(scalar(&mut s, "SELECT toString(7)"), Value::str("7"));
    assert_eq!(scalar(&mut s, "SELECT toUInt64('99')"), Value::UInt(99));
}

// -------------------------------------------------------------------- DML

#[test]
fn insert_with_explicit_columns() {
    let mut s = db();
    s.execute(
        "CREATE TABLE t (id UInt64, a Int64, b Nullable(String)) ENGINE = MergeTree ORDER BY id",
    )
    .unwrap();
    s.execute("INSERT INTO t (id, a) VALUES (1, 10), (2, 20)").unwrap();
    let r = rows(&mut s, "SELECT id, a, b FROM t ORDER BY id");
    assert_eq!(r[0][1], Value::Int(10));
    assert_eq!(r[0][2], Value::Null, "unmentioned nullable column defaults to NULL");
}

#[test]
fn insert_select() {
    let mut s = seeded();
    s.execute("CREATE TABLE copy (id UInt64, country String) ENGINE = MergeTree ORDER BY id")
        .unwrap();
    s.execute("INSERT INTO copy SELECT id, country FROM events WHERE country = 'US'")
        .unwrap();
    assert_eq!(scalar(&mut s, "SELECT count() FROM copy"), Value::UInt(3));
}

/// Last-write-wins applies to a *declared* key, and only to one.
///
/// This test used to declare only `ORDER BY id` and expect deduplication —
/// which was the data-loss bug: a sort key is not a unique key, and an INSERT
/// of two rows with the same `id` reported two rows affected and stored one.
/// Both halves are pinned here now, because a fix that kept the duplicate rows
/// by disabling the keyed delta entirely would satisfy the first assertion
/// while silently costing the OLTP path its upsert.
#[test]
fn last_write_wins_on_a_declared_key_only() {
    // Declared: upsert.
    let mut s = db();
    s.execute("CREATE TABLE t (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id")
        .unwrap();
    s.execute("INSERT INTO t VALUES (1, 100)").unwrap();
    s.execute("INSERT INTO t VALUES (1, 200)").unwrap();
    assert_eq!(scalar(&mut s, "SELECT count() FROM t"), Value::UInt(1));
    assert_eq!(scalar(&mut s, "SELECT v FROM t"), Value::Int(200));

    // Sort key alone: both rows survive.
    let mut s = db();
    s.execute("CREATE TABLE u (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id").unwrap();
    s.execute("INSERT INTO u VALUES (1, 100)").unwrap();
    s.execute("INSERT INTO u VALUES (1, 200)").unwrap();
    assert_eq!(scalar(&mut s, "SELECT count() FROM u"), Value::UInt(2));
    assert_eq!(scalar(&mut s, "SELECT sum(v) FROM u"), Value::Int(300));

    // ReplacingMergeTree is the other way to ask for it.
    let mut s = db();
    s.execute("CREATE TABLE r (id UInt64, v Int64) ENGINE = ReplacingMergeTree ORDER BY id")
        .unwrap();
    s.execute("INSERT INTO r VALUES (1, 100)").unwrap();
    s.execute("INSERT INTO r VALUES (1, 200)").unwrap();
    assert_eq!(scalar(&mut s, "SELECT count() FROM r"), Value::UInt(1));
    assert_eq!(scalar(&mut s, "SELECT v FROM r"), Value::Int(200));
}

#[test]
fn alter_delete_and_update() {
    let mut s = seeded();
    s.execute("ALTER TABLE events DELETE WHERE country = 'DE'").unwrap();
    assert_eq!(scalar(&mut s, "SELECT count() FROM events"), Value::UInt(4));

    s.execute("ALTER TABLE events UPDATE latency = 999 WHERE id = 1").unwrap();
    assert_eq!(
        scalar(&mut s, "SELECT latency FROM events WHERE id = 1"),
        Value::UInt(999)
    );
    // the other columns must survive the rewrite
    assert_eq!(
        scalar(&mut s, "SELECT country FROM events WHERE id = 1"),
        Value::str("US")
    );
    assert_eq!(scalar(&mut s, "SELECT count() FROM events"), Value::UInt(4));
}

#[test]
fn truncate_and_optimize() {
    let mut s = seeded();
    s.execute("OPTIMIZE TABLE events FINAL").unwrap();
    assert_eq!(scalar(&mut s, "SELECT count() FROM events"), Value::UInt(6));
    s.execute("TRUNCATE TABLE events").unwrap();
    assert_eq!(scalar(&mut s, "SELECT count() FROM events"), Value::UInt(0));
}

#[test]
fn add_and_drop_column() {
    let mut s = seeded();
    s.execute("ALTER TABLE events ADD COLUMN tag String").unwrap();
    let r = rows(&mut s, "SELECT id, tag FROM events ORDER BY id LIMIT 1");
    assert_eq!(r[0][1], Value::str(""));
    assert_eq!(scalar(&mut s, "SELECT count() FROM events"), Value::UInt(6));

    s.execute("ALTER TABLE events DROP COLUMN tag").unwrap();
    assert!(s.query("SELECT tag FROM events").is_err());
    // dropping a key column must be refused
    assert!(s.execute("ALTER TABLE events DROP COLUMN id").is_err());
}

// ------------------------------------------------------------------ joins

#[test]
fn inner_and_left_join() {
    let mut s = seeded();
    s.execute("CREATE TABLE geo (country String, region String) ENGINE = MergeTree ORDER BY country")
        .unwrap();
    s.execute("INSERT INTO geo VALUES ('US', 'NA'), ('DE', 'EU')").unwrap();

    let r = rows(
        &mut s,
        "SELECT e.id, g.region FROM events e INNER JOIN geo g ON e.country = g.country
         ORDER BY e.id",
    );
    assert_eq!(r.len(), 5, "FR has no match");
    assert_eq!(r[0][1], Value::str("NA"));

    let r = rows(
        &mut s,
        "SELECT e.id, g.region FROM events e LEFT JOIN geo g ON e.country = g.country
         ORDER BY e.id",
    );
    assert_eq!(r.len(), 6);
    let fr = r.iter().find(|row| row[0] == Value::UInt(4)).unwrap();
    assert_eq!(fr[1], Value::Null, "unmatched left row pads with NULL");
}

// ------------------------------------------------------------------- misc

#[test]
fn union_all_and_distinct() {
    let mut s = db();
    s.execute("CREATE TABLE a (x UInt64) ENGINE = MergeTree ORDER BY x").unwrap();
    s.execute("CREATE TABLE b (x UInt64) ENGINE = MergeTree ORDER BY x").unwrap();
    s.execute("INSERT INTO a VALUES (1), (2)").unwrap();
    s.execute("INSERT INTO b VALUES (2), (3)").unwrap();

    let r = col0(&mut s, "SELECT x FROM a UNION ALL SELECT x FROM b ORDER BY x");
    assert_eq!(r.len(), 4);
    let r = col0(&mut s, "SELECT x FROM a UNION DISTINCT SELECT x FROM b ORDER BY x");
    assert_eq!(r, vec![Value::UInt(1), Value::UInt(2), Value::UInt(3)]);
}

#[test]
fn cte_and_subquery_in_from() {
    let mut s = seeded();
    let r = rows(
        &mut s,
        "WITH fast AS (SELECT * FROM events WHERE latency < 200)
         SELECT country, count() FROM fast GROUP BY country ORDER BY country",
    );
    assert_eq!(r.len(), 3);

    let r = col0(
        &mut s,
        "SELECT id FROM (SELECT id FROM events WHERE country = 'US') ORDER BY id",
    );
    assert_eq!(r, vec![Value::UInt(1), Value::UInt(3), Value::UInt(6)]);
}

#[test]
fn uncorrelated_subqueries_are_folded_and_evaluated() {
    let mut s = seeded();

    // scalar subquery
    assert_eq!(scalar(&mut s, "SELECT (SELECT max(latency) FROM events)"), Value::UInt(300));
    assert_eq!(
        scalar(&mut s, "SELECT count() FROM events WHERE latency = (SELECT max(latency) FROM events)"),
        Value::UInt(1)
    );

    // IN (SELECT ...) and its negation
    let r = col0(
        &mut s,
        "SELECT id FROM events WHERE country IN (SELECT country FROM events WHERE latency > 200)
         ORDER BY id",
    );
    assert_eq!(r, vec![Value::UInt(1), Value::UInt(2), Value::UInt(3), Value::UInt(5), Value::UInt(6)]);
    assert_eq!(
        scalar(
            &mut s,
            "SELECT count() FROM events
             WHERE country NOT IN (SELECT country FROM events WHERE latency > 200)"
        ),
        Value::UInt(1)
    );

    // EXISTS / NOT EXISTS
    assert_eq!(
        scalar(&mut s, "SELECT count() FROM events WHERE EXISTS (SELECT 1 FROM events WHERE latency > 200)"),
        Value::UInt(6)
    );
    assert_eq!(
        scalar(&mut s, "SELECT count() FROM events WHERE EXISTS (SELECT 1 FROM events WHERE latency > 9999)"),
        Value::UInt(0)
    );
}

#[test]
fn bad_subqueries_report_clearly() {
    let mut s = seeded();

    // A scalar subquery returning many rows is an error, not a silent pick.
    let e = s
        .query("SELECT (SELECT latency FROM events)")
        .unwrap_err();
    assert!(e.to_string().contains("expected at most 1"), "{e}");

    // Correlated subqueries used to be refused outright, and this asserted the
    // refusal. They decorrelate into joins now, so the equality-correlated form
    // -- which is nearly all of them in practice -- answers.
    let ids = col0(
        &mut s,
        "SELECT id FROM events e
         WHERE latency = (SELECT max(latency) FROM events f WHERE f.country = e.country)",
    );
    assert!(!ids.is_empty(), "an equality-correlated subquery should answer");

    // What is still refused is a correlation that is not an equality, because
    // that is the shape with no join key to decorrelate onto. The refusal has
    // to explain that rather than leak an unknown-column error, since the
    // difference is "rewrite this" and not "you typed it wrong".
    let e = s
        .query(
            "SELECT id FROM events e
             WHERE latency = (SELECT max(latency) FROM events f WHERE f.country <> e.country)",
        )
        .unwrap_err();
    assert_eq!(e.code(), "NOT_IMPLEMENTED", "{e}");
    assert!(e.to_string().contains("correlate"), "{e}");
    assert!(e.to_string().contains("equality"), "should name why it cannot: {e}");
}

#[test]
fn explain_shows_the_plan() {
    let mut s = seeded();
    let plan: Vec<String> = col0(&mut s, "EXPLAIN SELECT id FROM events WHERE id > 3")
        .into_iter()
        .map(|v| v.render_plain())
        .collect();
    let text = plan.join("\n");
    assert!(text.contains("Scan"), "{text}");
    // the predicate must have been pushed into the scan
    assert!(text.contains("prewhere"), "filter did not reach the scan:\n{text}");
}

#[test]
fn errors_are_informative() {
    let mut s = seeded();
    let e = s.query("SELECT nope FROM events").unwrap_err();
    assert!(e.to_string().contains("nope"), "{e}");
    assert!(e.to_string().contains("latency"), "should list available columns: {e}");

    let e = s.query("SELECT * FROM missing_table").unwrap_err();
    assert!(e.to_string().contains("missing_table"), "{e}");

    let e = s.query("SELECT FROM").unwrap_err();
    assert_eq!(e.code(), "SYNTAX_ERROR", "{e}");
}

// ------------------------------------------------------- storage behaviour

#[test]
fn zone_maps_prune_a_selective_range() -> Result<()> {
    let mut s = db();
    s.execute("CREATE TABLE big (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id")?;

    // 50k rows = ~49 granules. A 100-row range must touch almost none of them.
    let values: Vec<String> = (0..50_000u64).map(|i| format!("({i},{i})")).collect();
    s.execute(&format!("INSERT INTO big VALUES {}", values.join(",")))?;
    s.execute("OPTIMIZE TABLE big FINAL")?;

    let rs = s.query("SELECT count() FROM big WHERE id >= 20000 AND id < 20100")?;
    assert_eq!(rs.scalar().unwrap(), Value::UInt(100));
    assert!(
        rs.stats.granules_pruned > 40,
        "expected heavy pruning, got {} pruned / {} read",
        rs.stats.granules_pruned,
        rs.stats.granules_read
    );
    assert!(
        rs.stats.granules_read <= 3,
        "expected to read at most a couple of granules, read {}",
        rs.stats.granules_read
    );
    Ok(())
}

#[test]
fn large_table_aggregates_match_a_reference() -> Result<()> {
    let mut s = db();
    s.execute("CREATE TABLE t (id UInt64, g UInt32, v Int64) ENGINE = MergeTree ORDER BY id")?;
    let n = 20_000u64;
    let values: Vec<String> = (0..n)
        .map(|i| format!("({i},{},{})", i % 7, (i as i64 % 100) - 50))
        .collect();
    s.execute(&format!("INSERT INTO t VALUES {}", values.join(",")))?;

    let expect_sum: i64 = (0..n).map(|i| (i as i64 % 100) - 50).sum();
    assert_eq!(scalar(&mut s, "SELECT sum(v) FROM t"), Value::Int(expect_sum));
    assert_eq!(scalar(&mut s, "SELECT count() FROM t"), Value::UInt(n));

    let r = rows(&mut s, "SELECT g, count() FROM t GROUP BY g ORDER BY g");
    assert_eq!(r.len(), 7);
    let total: u64 = r.iter().map(|row| row[1].as_u64().unwrap()).sum();
    assert_eq!(total, n);
    Ok(())
}

#[test]
fn string_columns_survive_compaction() -> Result<()> {
    let mut s = db();
    s.execute("CREATE TABLE t (id UInt64, name String) ENGINE = MergeTree ORDER BY id")?;
    let names = ["alpha", "beta", "gamma", "delta", "epsilon"];
    let values: Vec<String> = (0..5_000u64)
        .map(|i| format!("({i},'{}')", names[i as usize % names.len()]))
        .collect();
    s.execute(&format!("INSERT INTO t VALUES {}", values.join(",")))?;
    s.execute("OPTIMIZE TABLE t FINAL")?;

    assert_eq!(scalar(&mut s, "SELECT count() FROM t"), Value::UInt(5000));
    let r = rows(&mut s, "SELECT name, count() FROM t GROUP BY name ORDER BY name");
    assert_eq!(r.len(), 5);
    assert_eq!(r[0][0], Value::str("alpha"));
    assert_eq!(r[0][1], Value::UInt(1000));
    // a string range predicate must work through the order-preserving dictionary
    assert_eq!(
        scalar(&mut s, "SELECT count() FROM t WHERE name >= 'delta' AND name < 'epsilon'"),
        Value::UInt(1000)
    );
    Ok(())
}

// ------------------------------------------------------- column DEFAULT

/// `DEFAULT` has to survive the whole round trip: DDL, omitted-column INSERT,
/// a backfilling `ADD COLUMN`, and `SHOW CREATE TABLE`.
///
/// The original bug was the nastiest shape available — the DEFAULT was parsed,
/// persisted and echoed back by `SHOW CREATE TABLE`, so the user had positive
/// confirmation the feature worked while every row silently got the type's
/// zero. It corrupted at ingest, so fixing the code afterwards would not have
/// recovered the data. These assert the value on disk, not the DDL text.
#[test]
fn a_default_is_applied_to_omitted_columns() -> Result<()> {
    let mut s = db();
    s.execute(
        "CREATE TABLE t (id UInt64, s String DEFAULT 'hello', n Int64 DEFAULT 42, f Float64 DEFAULT 1.5)
         ENGINE = MergeTree ORDER BY id",
    )?;
    s.execute("INSERT INTO t (id) VALUES (1)")?;
    // An explicit value still wins over the default.
    s.execute("INSERT INTO t (id, s, n) VALUES (2, 'given', 7)")?;

    assert_eq!(s.query("SELECT s FROM t WHERE id = 1")?.scalar(), Some(Value::str("hello")));
    assert_eq!(s.query("SELECT n FROM t WHERE id = 1")?.scalar(), Some(Value::Int(42)));
    assert_eq!(s.query("SELECT f FROM t WHERE id = 1")?.scalar(), Some(Value::Float(1.5)));
    assert_eq!(s.query("SELECT s FROM t WHERE id = 2")?.scalar(), Some(Value::str("given")));
    assert_eq!(s.query("SELECT n FROM t WHERE id = 2")?.scalar(), Some(Value::Int(7)));
    Ok(())
}

/// `ADD COLUMN ... DEFAULT` must backfill existing rows with the same value a
/// later INSERT would produce. The two paths disagreeing is the bug that hides
/// longest, because each looks right on its own.
#[test]
fn add_column_backfills_with_its_default() -> Result<()> {
    let mut s = db();
    s.execute("CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id")?;
    s.execute("INSERT INTO t VALUES (1), (2)")?;
    s.execute("ALTER TABLE t ADD COLUMN tag String DEFAULT 'backfilled'")?;
    s.execute("INSERT INTO t (id) VALUES (3)")?;

    let n = s.query("SELECT count() FROM t WHERE tag = 'backfilled'")?.scalar();
    assert_eq!(n, Some(Value::UInt(3)), "old rows and new rows must agree");
    Ok(())
}

/// A default that cannot be honoured is a DDL error, not a surprise at insert
/// time — and never a silently-ignored clause.
#[test]
fn an_impossible_default_is_rejected_at_ddl() {
    let mut s = db();
    let e = s
        .execute("CREATE TABLE t (id UInt64, n Int64 DEFAULT 'not a number') ENGINE = MergeTree ORDER BY id")
        .expect_err("a String default on an Int64 column must be refused");
    let msg = e.to_string();
    assert!(msg.contains('n') || msg.contains("DEFAULT"), "unhelpful error: {msg}");

    let mut s2 = db();
    assert!(
        s2.execute("CREATE TABLE u (id UInt64, t DateTime DEFAULT now()) ENGINE = MergeTree ORDER BY id")
            .is_err(),
        "a non-constant default must be refused rather than accepted and ignored"
    );
}

/// SHOW CREATE TABLE round-trips: what it prints must re-create the same table.
#[test]
fn show_create_table_emits_defaults_that_parse_back() -> Result<()> {
    let mut s = db();
    s.execute("CREATE TABLE t (id UInt64, s String DEFAULT 'x', n Int64 DEFAULT -1) ENGINE = MergeTree ORDER BY id")?;
    let ddl = match s.query("SHOW CREATE TABLE t")?.scalar() {
        Some(Value::Str(v)) => v.to_string(),
        other => panic!("SHOW CREATE TABLE returned {other:?}"),
    };
    assert!(ddl.contains("DEFAULT 'x'"), "missing string default in:\n{ddl}");
    assert!(ddl.contains("DEFAULT -1"), "missing negative default in:\n{ddl}");

    // The real test: it parses back, and the reconstructed table behaves the same.
    s.execute("DROP TABLE t")?;
    s.execute(&ddl)?;
    s.execute("INSERT INTO t (id) VALUES (1)")?;
    assert_eq!(s.query("SELECT s FROM t")?.scalar(), Some(Value::str("x")));
    assert_eq!(s.query("SELECT n FROM t")?.scalar(), Some(Value::Int(-1)));
    Ok(())
}

/// A date that does not exist is refused, rather than rolled over into one that
/// does.
///
/// `toDate('2021-02-30')` returned `2021-03-02`, `toDate('2023-11-31')` returned
/// `2023-12-01`, and `toDateTime('2021-02-30 25:99:99')` obliged with
/// `2021-03-03 02:40:39`. Time components were unbounded in *both* directions,
/// so `-5:00:00` walked the instant back into the previous day. Every one of
/// them reported success, through the scalar functions and through both write
/// paths, which is how a typo becomes a row nobody queries for.
#[test]
fn an_impossible_date_is_refused_rather_than_rolled_over() {
    let mut s = db();
    // NULL is how a failed cast reports itself here; the point is that the
    // result is never a different, valid date.
    for bad in [
        "2021-02-30", "2023-11-31", "2021-04-31", "2021-06-31", "2021-09-31", "2021-01-32",
        "2100-02-29", // a century that is not a leap year
        "2021-02-29", // an ordinary year
    ] {
        assert_eq!(
            scalar(&mut s, &format!("SELECT toDate('{bad}')")),
            Value::Null,
            "toDate('{bad}') must not roll over"
        );
    }
    // ...and every real day still parses, leap rules included.
    for good in ["2020-02-29", "2000-02-29", "2400-02-29", "2021-02-28", "2021-04-30", "2021-12-31"]
    {
        assert_eq!(
            scalar(&mut s, &format!("SELECT toString(toDate('{good}'))")),
            Value::str(good),
            "toDate('{good}') must still parse"
        );
    }

    // Time components are bounded at both ends.
    for bad in ["25:00:00", "00:60:00", "00:00:60", "-5:00:00", "00:00:-3600", "00:00:99999"] {
        assert_eq!(
            scalar(&mut s, &format!("SELECT toDateTime('2020-03-01 {bad}')")),
            Value::Null,
            "toDateTime with time `{bad}` must not carry into another day"
        );
    }
    assert_eq!(
        scalar(&mut s, "SELECT toString(toDateTime('2020-03-01 23:59:59'))"),
        Value::str("2020-03-01 23:59:59")
    );

    // And it reaches the write path: the literal is refused, so no row lands.
    s.execute("CREATE TABLE d (v Date) ENGINE = MergeTree ORDER BY v").unwrap();
    assert!(
        s.execute("INSERT INTO d VALUES ('2021-02-30')").is_err(),
        "an impossible date must fail the INSERT, not become 2021-03-02"
    );
    assert_eq!(scalar(&mut s, "SELECT count() FROM d"), Value::UInt(0));
    s.execute("INSERT INTO d VALUES ('2021-02-28')").unwrap();
    assert_eq!(scalar(&mut s, "SELECT count() FROM d"), Value::UInt(1));
}
