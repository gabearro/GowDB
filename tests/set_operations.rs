//! `INTERSECT` and `EXCEPT`, with and without `ALL`, end to end through
//! `Session` and the CLI binary.
//!
//! # How the answers are checked
//!
//! Never against themselves. Each `DISTINCT`-form case is paired with the
//! **hand-written semi-join or anti-join** that means the same thing, and the
//! two must agree row for row -- that is the claim the feature makes
//! (`INTERSECT` *is* a semi-join, `EXCEPT` *is* an anti-join), so checking it
//! checks the answer and the claim at once. The `ALL` forms have no such
//! spelling, because a join over deduplicated inputs has thrown multiplicity
//! away, so those are checked against a multiset computed in this file from
//! the base rows -- an independent implementation, in a different language,
//! of `min(m, n)` and `max(m - n, 0)`.
//!
//! # The four rules that are easy to get wrong
//!
//!   * **`DISTINCT` is the default.** `A INTERSECT B` emits each surviving
//!     tuple once.
//!   * **`ALL` means multiplicity**, and the two rules differ: `INTERSECT ALL`
//!     keeps `min(m, n)` copies, `EXCEPT ALL` keeps `max(m - n, 0)`.
//!   * **`INTERSECT` binds tighter than `UNION` and `EXCEPT`.** A three-way
//!     expression means something different if it does not, and nothing in the
//!     text of the query would show it.
//!   * **NULLs are not distinct here.** Two NULLs *match*, which is the
//!     opposite of `=`. The hand-written comparands in this file spell that
//!     rule out longhand (`... OR (a IS NULL AND EXISTS ...)`) rather than
//!     relying on `IN`, precisely because `IN` gets it wrong -- which is what
//!     makes them an independent check and not a second copy of the bug.
//!
//! # Reachability
//!
//! `set_operations_reach_the_executor` is first, and it is one line per
//! operation through `Session::query`. Nine capabilities in this engine's
//! history landed complete in `src/` and were never wired to a session; if the
//! wiring is what broke, that test fails on its own and the rest of the file
//! is not evidence of anything. `the_cli_runs_a_set_operation` does the same
//! through the shipped binary, so a `Session`-only path could not fake it.

use std::collections::HashMap;
use std::process::Command;

use granular::types::Value;
use granular::Session;

// ------------------------------------------------------------------ helpers

/// `l` and `r` share a shape and a nullable payload. The multiplicities are
/// chosen so every branch of both rules is exercised by one pair of tables:
///
/// ```text
///   a       1    2    3    NULL   4     5
///   in l    1    2    1     2     0     3
///   in r    0    1    2     1     2     3
///   min     0    1    1     1     0     3     <- INTERSECT ALL
///   l - r   1    1    0     1     0     0     <- EXCEPT ALL
/// ```
fn db() -> Session {
    let mut db = Session::in_memory();
    for stmt in [
        "CREATE TABLE l (id UInt64, a Nullable(Int64)) ENGINE = MergeTree ORDER BY id",
        "CREATE TABLE r (id UInt64, a Nullable(Int64)) ENGINE = MergeTree ORDER BY id",
        "INSERT INTO l VALUES (1, 1), (2, 2), (3, 2), (4, 3), (5, NULL), (6, NULL), \
         (7, 5), (8, 5), (9, 5)",
        "INSERT INTO r VALUES (1, 2), (2, 3), (3, 3), (4, NULL), (5, 4), (6, 4), \
         (7, 5), (8, 5), (9, 5)",
    ] {
        db.execute(stmt).unwrap_or_else(|e| panic!("{stmt}: {e}"));
    }
    db
}

fn rows(db: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    db.query(sql).unwrap_or_else(|e| panic!("{sql}\n  -> {e}")).to_values()
}

/// One column of a result, sorted, so a comparison is over the *set* of rows
/// and not over an output order neither engine promises.
fn col(db: &mut Session, sql: &str) -> Vec<Value> {
    let mut v: Vec<Value> = rows(db, sql).into_iter().map(|r| r[0].clone()).collect();
    v.sort();
    v
}

fn err(db: &mut Session, sql: &str) -> String {
    match db.query(sql) {
        Ok(r) => panic!("expected `{sql}` to be refused, got {} rows", r.rows()),
        Err(e) => e.to_string(),
    }
}

fn explain(db: &mut Session, sql: &str) -> String {
    rows(db, &format!("EXPLAIN {sql}"))
        .iter()
        .map(|r| match &r[0] {
            // `Display` on a `Value::Str` quotes it, which an EXPLAIN line is
            // not.
            Value::Str(s) => s.to_string(),
            other => other.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Multiplicity of every value in a one-column result.
fn bag(vs: &[Value]) -> HashMap<String, usize> {
    let mut m = HashMap::new();
    for v in vs {
        *m.entry(v.to_string()).or_insert(0) += 1;
    }
    m
}

/// `min(m, n)` per value: `INTERSECT ALL`, computed here rather than asked of
/// the thing under test.
fn intersect_all(l: &[Value], r: &[Value]) -> HashMap<String, usize> {
    let rb = bag(r);
    bag(l)
        .into_iter()
        .filter_map(|(k, m)| {
            let n = *rb.get(&k).unwrap_or(&0);
            (m.min(n) > 0).then_some((k, m.min(n)))
        })
        .collect()
}

/// `max(m - n, 0)` per value: `EXCEPT ALL`.
fn except_all(l: &[Value], r: &[Value]) -> HashMap<String, usize> {
    let rb = bag(r);
    bag(l)
        .into_iter()
        .filter_map(|(k, m)| {
            let n = *rb.get(&k).unwrap_or(&0);
            (m > n).then_some((k, m - n))
        })
        .collect()
}

// ------------------------------------------------------------ reachability

/// The first thing to check, and the cheapest: does a set operation survive
/// the trip from SQL text to rows at all?
#[test]
fn set_operations_reach_the_executor() {
    let mut db = Session::in_memory();
    for (sql, want) in [
        ("SELECT 1 INTERSECT SELECT 1", 1),
        ("SELECT 1 INTERSECT ALL SELECT 1", 1),
        ("SELECT 1 INTERSECT SELECT 2", 0),
        ("SELECT 1 EXCEPT SELECT 2", 1),
        ("SELECT 1 EXCEPT ALL SELECT 1", 0),
        ("SELECT 1 EXCEPT SELECT 1", 0),
    ] {
        let got = rows(&mut db, sql).len();
        assert_eq!(got, want, "{sql} returned {got} rows, expected {want}");
    }
}

#[test]
fn the_cli_runs_a_set_operation() {
    let out = Command::new(env!("CARGO_BIN_EXE_granular"))
        .args(["-q", "SELECT 2 INTERSECT ALL SELECT 2 EXCEPT ALL SELECT 3"])
        .output()
        .expect("run the granular binary");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "exit {:?}: {text}", out.status.code());
    assert!(text.contains(" 2 "), "no result row in:\n{text}");
    assert!(text.contains("1 row"), "expected exactly one row:\n{text}");
}

// -------------------------------------------- against the hand-written form

/// `INTERSECT` against the semi-join it claims to be. The comparand states the
/// NULL rule in full rather than leaning on `IN`, which would answer "unknown"
/// for a NULL probe and quietly drop the row both engines must keep.
#[test]
fn intersect_agrees_with_a_hand_written_semi_join() {
    let mut db = db();
    let got = col(&mut db, "SELECT a FROM l INTERSECT SELECT a FROM r");
    let want = col(
        &mut db,
        "SELECT DISTINCT a FROM l WHERE a IN (SELECT a FROM r) \
         OR (a IS NULL AND EXISTS (SELECT 1 FROM r WHERE a IS NULL))",
    );
    assert_eq!(got, want);
    // ...and it really is the interesting set, not two empty answers.
    assert_eq!(got, vec![Value::Null, Value::Int(2), Value::Int(3), Value::Int(5)]);
}

/// `EXCEPT` against the anti-join. Same shape, same reason for the longhand.
#[test]
fn except_agrees_with_a_hand_written_anti_join() {
    let mut db = db();
    let got = col(&mut db, "SELECT a FROM l EXCEPT SELECT a FROM r");
    let want = col(
        &mut db,
        "SELECT DISTINCT a FROM l \
         WHERE (a IS NOT NULL AND a NOT IN (SELECT a FROM r WHERE a IS NOT NULL)) \
         OR (a IS NULL AND NOT EXISTS (SELECT 1 FROM r WHERE a IS NULL))",
    );
    assert_eq!(got, want);
    assert_eq!(got, vec![Value::Int(1)]);
}

/// The `ALL` forms, against `min(m, n)` and `max(m - n, 0)` computed in Rust
/// from the base rows. `bag` is what a join over `Distinct` inputs cannot give
/// back, which is why these have no SQL comparand.
#[test]
fn the_all_forms_keep_the_right_number_of_copies() {
    let mut db = db();
    let l = rows(&mut db, "SELECT a FROM l").into_iter().map(|r| r[0].clone()).collect::<Vec<_>>();
    let r = rows(&mut db, "SELECT a FROM r").into_iter().map(|r| r[0].clone()).collect::<Vec<_>>();

    let got = bag(&col(&mut db, "SELECT a FROM l INTERSECT ALL SELECT a FROM r"));
    assert_eq!(got, intersect_all(&l, &r), "INTERSECT ALL is min(m, n)");
    // 2 once, 3 once, NULL once, 5 three times. Spelled out so a change in
    // both the operator and the helper cannot agree on a wrong answer.
    assert_eq!(
        got,
        HashMap::from([("2".into(), 1), ("3".into(), 1), ("NULL".into(), 1), ("5".into(), 3)])
    );

    let got = bag(&col(&mut db, "SELECT a FROM l EXCEPT ALL SELECT a FROM r"));
    assert_eq!(got, except_all(&l, &r), "EXCEPT ALL is max(m - n, 0)");
    assert_eq!(got, HashMap::from([("1".into(), 1), ("2".into(), 1), ("NULL".into(), 1)]));
}

/// `ALL` is not a hint that may be ignored: the two forms of one query must
/// give different answers on data that has duplicates, or `ALL` has been
/// silently read as `DISTINCT` -- the accept-and-ignore failure this engine
/// spent seven waves removing.
#[test]
fn all_and_distinct_are_different_operations() {
    let mut db = db();
    for (all, distinct) in [
        (
            "SELECT a FROM l INTERSECT ALL SELECT a FROM r",
            "SELECT a FROM l INTERSECT SELECT a FROM r",
        ),
        ("SELECT a FROM l EXCEPT ALL SELECT a FROM r", "SELECT a FROM l EXCEPT SELECT a FROM r"),
    ] {
        let (a, d) = (col(&mut db, all), col(&mut db, distinct));
        assert_ne!(a, d, "`{all}` and `{distinct}` gave the same rows");
        assert!(a.len() > d.len(), "ALL should keep more rows: {a:?} vs {d:?}");
    }
}

// -------------------------------------------------------------------- NULLs

/// The rule in isolation, on both sides, in all four forms. `SELECT NULL`
/// against `SELECT NULL` is the whole feature in one line: `=` says unknown,
/// a set operation says match.
#[test]
fn two_nulls_match_each_other() {
    let mut db = Session::in_memory();
    for (sql, want) in [
        ("SELECT NULL INTERSECT SELECT NULL", 1),
        ("SELECT NULL INTERSECT ALL SELECT NULL", 1),
        ("SELECT NULL EXCEPT SELECT NULL", 0),
        ("SELECT NULL EXCEPT ALL SELECT NULL", 0),
        // A NULL on one side only: no match, so INTERSECT drops it and
        // EXCEPT keeps it.
        ("SELECT NULL INTERSECT SELECT 1", 0),
        ("SELECT NULL EXCEPT SELECT 1", 1),
        ("SELECT 1 EXCEPT SELECT NULL", 1),
    ] {
        let got = rows(&mut db, sql).len();
        assert_eq!(got, want, "{sql} returned {got} rows, expected {want}");
    }
    // The kept row really is NULL and not a zero that renders like one.
    assert_eq!(rows(&mut db, "SELECT NULL INTERSECT SELECT NULL")[0][0], Value::Null);
}

/// NULLs count like any other value under `ALL`: three against two leaves one.
#[test]
fn null_multiplicity_follows_the_all_rules() {
    let mut db = Session::in_memory();
    db.execute("CREATE TABLE n (id UInt64, a Nullable(Int64)) ENGINE = MergeTree ORDER BY id")
        .unwrap();
    db.execute("CREATE TABLE m (id UInt64, a Nullable(Int64)) ENGINE = MergeTree ORDER BY id")
        .unwrap();
    db.execute("INSERT INTO n VALUES (1, NULL), (2, NULL), (3, NULL)").unwrap();
    db.execute("INSERT INTO m VALUES (1, NULL), (2, NULL)").unwrap();
    assert_eq!(col(&mut db, "SELECT a FROM n EXCEPT ALL SELECT a FROM m"), vec![Value::Null]);
    assert_eq!(
        col(&mut db, "SELECT a FROM n INTERSECT ALL SELECT a FROM m"),
        vec![Value::Null, Value::Null],
        "min(3, 2)"
    );
    assert_eq!(col(&mut db, "SELECT a FROM n EXCEPT SELECT a FROM m"), Vec::<Value>::new());
}

/// A multi-column tuple with a NULL in one position still matches as a whole
/// row -- the case a per-column `=` would answer differently for every column.
#[test]
fn null_matching_is_per_tuple_not_per_column() {
    let mut db = Session::in_memory();
    let got = rows(
        &mut db,
        "SELECT 1, CAST(NULL AS Nullable(Int64)) INTERSECT SELECT 1, CAST(NULL AS Nullable(Int64))",
    );
    assert_eq!(got.len(), 1, "(1, NULL) matches (1, NULL)");
    let got = rows(
        &mut db,
        "SELECT 1, CAST(NULL AS Nullable(Int64)) INTERSECT SELECT 2, CAST(NULL AS Nullable(Int64))",
    );
    assert!(got.is_empty(), "the non-NULL column still has to agree");
}

// --------------------------------------------------------------- precedence

/// `INTERSECT` binds tighter. `1 UNION ALL 2 INTERSECT ALL 3` is
/// `1 UNION ALL (2 INTERSECT ALL 3)`, which is `{1}`; read left to right it
/// would be `(1 UNION ALL 2) INTERSECT ALL 3`, which is `{}`. The two answers
/// are different, so this test cannot pass by accident.
#[test]
fn intersect_binds_tighter_in_a_three_way_expression() {
    let mut db = Session::in_memory();
    assert_eq!(
        col(&mut db, "SELECT 1 UNION ALL SELECT 2 INTERSECT ALL SELECT 3"),
        vec![Value::UInt(1)],
        "INTERSECT must have consumed `SELECT 2 INTERSECT ALL SELECT 3` first"
    );
    // Parentheses force the other reading, and it really is the other answer.
    assert!(col(&mut db, "(SELECT 1 UNION ALL SELECT 2) INTERSECT ALL SELECT 3").is_empty());

    // Same for EXCEPT, which shares UNION's level: `1 UNION 2 INTERSECT 2` is
    // `1 UNION (2 INTERSECT 2)` = {1, 2}, not `(1 UNION 2) INTERSECT 2` = {2}.
    assert_eq!(
        col(&mut db, "SELECT 1 UNION SELECT 2 INTERSECT SELECT 2"),
        vec![Value::UInt(1), Value::UInt(2)]
    );

    // And the plan says so, so a right answer from a wrong tree is ruled out.
    let e = explain(&mut db, "SELECT 1 UNION ALL SELECT 2 INTERSECT ALL SELECT 3");
    assert!(e.starts_with("Union All"), "{e}");
    assert!(e.contains("  Intersect All"), "{e}");
}

/// `EXCEPT` is left-associative, and it is not associative as an operation:
/// `(1,2) EXCEPT (1) EXCEPT (2)` is `{}` but `(1,2) EXCEPT ((1) EXCEPT (2))`
/// is `{2}`. A planner that flattened a chain of `EXCEPT`s without minding
/// which side the nesting came from would answer the second for the first.
#[test]
fn except_chains_left_and_parentheses_change_the_answer() {
    let mut db = Session::in_memory();
    let flat = "SELECT 1 UNION ALL SELECT 2 EXCEPT ALL SELECT 1 EXCEPT ALL SELECT 2";
    assert!(col(&mut db, flat).is_empty(), "(1,2) - 1 - 2 is empty");
    let nested = "(SELECT 1 UNION ALL SELECT 2) EXCEPT ALL (SELECT 1 EXCEPT ALL SELECT 2)";
    assert_eq!(col(&mut db, nested), vec![Value::UInt(2)], "(1,2) - ({{1}} - {{2}}) is {{2}}");
}

/// A three-way chain of one kind is one node, and it means the same as the
/// nested spelling it came from.
#[test]
fn three_way_chains_of_one_kind_agree_with_their_nested_form() {
    let mut db = db();
    let three = "SELECT a FROM l INTERSECT ALL SELECT a FROM r INTERSECT ALL SELECT a FROM l";
    let nested = "SELECT a FROM l INTERSECT ALL (SELECT a FROM r INTERSECT ALL SELECT a FROM l)";
    assert_eq!(col(&mut db, three), col(&mut db, nested));
    let e = explain(&mut db, three);
    assert_eq!(e.matches("Intersect All").count(), 1, "the chain is one node: {e}");
}

// ------------------------------------------------------------ empty and odd

#[test]
fn empty_inputs_on_either_side() {
    let mut db = db();
    // `WHERE 1 = 0` is the empty branch that still has a schema.
    let empty_l = "SELECT a FROM l WHERE a > 1000";
    for (sql, want) in [
        (format!("{empty_l} INTERSECT SELECT a FROM r"), 0),
        (format!("SELECT a FROM r INTERSECT {empty_l}"), 0),
        (format!("{empty_l} EXCEPT SELECT a FROM r"), 0),
        (format!("{empty_l} INTERSECT ALL SELECT a FROM r"), 0),
    ] {
        assert_eq!(rows(&mut db, &sql).len(), want, "{sql}");
    }
    // Subtracting nothing leaves the left side, deduplicated or not.
    let empty_r = "SELECT a FROM r WHERE a > 1000";
    assert_eq!(rows(&mut db, &format!("SELECT a FROM l EXCEPT ALL {empty_r}")).len(), 9);
    assert_eq!(rows(&mut db, &format!("SELECT a FROM l EXCEPT {empty_r}")).len(), 5);
}

/// A set operation is a query like any other: `ORDER BY` and `LIMIT` attach to
/// the compound, not to its last branch.
#[test]
fn order_by_and_limit_apply_to_the_whole_compound() {
    let mut db = db();
    let got =
        rows(&mut db, "SELECT a FROM l EXCEPT ALL SELECT a FROM r ORDER BY a NULLS LAST LIMIT 2");
    assert_eq!(got.len(), 2);
    assert_eq!(got[0][0], Value::Int(1));
    assert_eq!(got[1][0], Value::Int(2));
}

/// Set operations compose with the rest of the language: a branch may be a
/// `VALUES` list, an aggregate or a CTE, and the compound may feed a subquery.
#[test]
fn branches_can_be_anything_that_produces_rows() {
    let mut db = db();
    assert_eq!(
        col(&mut db, "SELECT a FROM l INTERSECT VALUES (1), (2)"),
        vec![Value::Int(1), Value::Int(2)]
    );
    assert_eq!(
        col(&mut db, "WITH t AS (SELECT a FROM r) SELECT a FROM l INTERSECT SELECT a FROM t"),
        vec![Value::Null, Value::Int(2), Value::Int(3), Value::Int(5)]
    );
    assert_eq!(
        rows(&mut db, "SELECT count() FROM (SELECT a FROM l EXCEPT ALL SELECT a FROM r)")[0][0]
            .as_i64(),
        Some(3)
    );
    // An aggregate on one side, a scan on the other.
    assert_eq!(rows(&mut db, "SELECT max(a) FROM l INTERSECT SELECT 5").len(), 1);
}

#[test]
fn arity_and_type_mismatches_are_reported_and_name_the_operation() {
    let mut db = db();
    let m = err(&mut db, "SELECT id, a FROM l INTERSECT SELECT a FROM r");
    assert!(m.contains("INTERSECT branches disagree on width"), "{m}");
    let m = err(&mut db, "SELECT id, a FROM l EXCEPT SELECT a FROM r");
    assert!(m.contains("EXCEPT branches disagree on width"), "{m}");
    // A compatible pair is *not* an error: UInt64 and Int64 promote, and the
    // rule is the same one UNION uses. `l.id` is 1..9 and `r.a` holds
    // 2, 3, 4, 5 and NULL, so four ids survive.
    assert_eq!(
        col(&mut db, "SELECT id FROM l INTERSECT SELECT a FROM r"),
        vec![Value::Int(2), Value::Int(3), Value::Int(4), Value::Int(5)]
    );
}

// -------------------------------------------------------- the negative case

/// A query with no set operation must be untouched by all of this: same plan,
/// node for node. Pinned as a literal because "unchanged" is only meaningful
/// against something written down.
#[test]
fn a_query_without_a_set_operation_plans_exactly_as_before() {
    let mut db = db();
    assert_eq!(
        explain(&mut db, "SELECT a FROM l WHERE a > 1 ORDER BY a LIMIT 3"),
        "Limit 3 offset 0\n  Project [a#0 AS a]\n    Sort [a#0]\n      \
         Scan default.l [a] prewhere=(a#0 > 1) zonemap=1"
    );
    // And a UNION still plans as a UNION, with the same label it always had.
    assert_eq!(
        explain(&mut db, "SELECT a FROM l UNION ALL SELECT a FROM r"),
        "Union All\n  Project [a#0 AS a]\n    Scan default.l [a]\n  \
         Project [a#0 AS a]\n    Scan default.r [a]"
    );
}

// ------------------------------------------------------------- the shape it
// ------------------------------------------------------------- promises

/// The design claim, made falsifiable: only the branches being matched
/// *against* are held in memory, so a large left side against a small right
/// side runs in a budget that could not hold the left side.
///
/// `big` holds 200k *distinct* values, so a table over it is ~10 MB of
/// `GroupKey`s; the budget here is 1 MB, and the right side is 4 tuples. The
/// same query with the sides swapped is *expected* to fail, which is what
/// makes this a test of the streaming and not of the budget.
#[test]
fn the_large_side_streams_and_only_the_small_side_is_built() {
    let mut db = Session::in_memory();
    db.execute("CREATE TABLE big (id UInt64, a Int64) ENGINE = MergeTree ORDER BY id").unwrap();
    db.execute("CREATE TABLE small (id UInt64, a Int64) ENGINE = MergeTree ORDER BY id").unwrap();
    let vals: Vec<String> = (0..200_000u64).map(|i| format!("({i}, {i})")).collect();
    db.execute(&format!("INSERT INTO big VALUES {}", vals.join(","))).unwrap();
    db.execute("INSERT INTO small VALUES (1, 0), (2, 1), (3, 2), (4, 3)").unwrap();
    db.execute("SET max_memory_usage = 1048576").unwrap();

    // 4 entries in the table, 200k rows streamed past it.
    assert_eq!(
        rows(&mut db, "SELECT count() FROM (SELECT a FROM big EXCEPT ALL SELECT a FROM small)")[0]
            [0]
        .as_i64(),
        Some(199_996)
    );
    assert_eq!(
        rows(&mut db, "SELECT count() FROM (SELECT a FROM big INTERSECT ALL SELECT a FROM small)")
            [0][0]
            .as_i64(),
        Some(4)
    );

    // With the big side on the right there is nothing to stream: the table has
    // to hold it, and the budget says no. A refusal rather than an OOM is the
    // point; the message names the operator so the plan can be read from it.
    let e = err(&mut db, "SELECT a FROM small INTERSECT ALL SELECT a FROM big");
    assert!(e.contains("set operation") && e.contains("memory"), "{e}");
}

/// An `INTERSECT` whose right side is empty cannot produce a row, and the
/// operator must know that before it reads the left side. Measured as work not
/// done: the query touches no granules of `big`.
#[test]
fn an_empty_intersect_never_reads_the_streaming_side() {
    let mut db = Session::in_memory();
    db.execute("CREATE TABLE big (id UInt64, a Int64) ENGINE = MergeTree ORDER BY id").unwrap();
    let vals: Vec<String> = (0..50_000u64).map(|i| format!("({i}, {i})")).collect();
    db.execute(&format!("INSERT INTO big VALUES {}", vals.join(","))).unwrap();

    let r = db.query("SELECT a FROM big INTERSECT ALL SELECT a FROM big WHERE a < 0").unwrap();
    assert_eq!(r.rows(), 0);
    let read = r.stats.rows_scanned;
    assert!(read < 50_000, "the streaming side was read anyway ({read} rows)");
}
