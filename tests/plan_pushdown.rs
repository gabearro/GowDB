//! Predicate pushdown must move a conjunct without moving the answer.
//!
//! Every rewrite in `optimizer::sink_filter` and `optimizer::normalize_
//! predicates` is a bet that two differently-shaped plans compute the same
//! relation. The bet is worth making -- pushing `a.k = 150000` below a
//! 1M x 1M join is 154 ms against 0.063 ms -- and it is also exactly the class
//! of change that returns wrong rows *quietly*, because a pushdown bug does
//! not crash, it drops rows nobody counted.
//!
//! So this file is built around two assertions per rule, and needs both:
//!
//!   * **the answer**, against a reference join written out by hand in Rust
//!     ([`join_ref`]) rather than against another SQL spelling. Two SQL
//!     spellings share an optimizer; a nested loop in this file does not, and
//!     it is the only thing here that knows what `LEFT JOIN` is supposed to
//!     mean when both sides have NULL keys;
//!   * **the plan**, via `EXPLAIN`. A rule that silently stopped firing would
//!     pass every answer check in this file, which is how an optimization
//!     becomes dead code without anybody noticing.
//!
//! The negative cases carry the same weight as the positive ones. A rule that
//! fires where it may not is a wrong answer; the LEFT-JOIN-right-side case
//! below is the specific one that turns an outer join into an inner join, and
//! the fixture has NULLs on both sides of every join precisely so that it
//! shows up as different rows rather than as a different plan.

use granular::types::Value;
use granular::Session;

// ------------------------------------------------------------------ fixtures

/// `(k, v)`; `k` is nullable and is the join key, `v` is a payload.
type Row = (Option<i64>, i64);

/// Deliberately awkward: NULL keys on both sides (never match, must survive
/// outer padding), a key present in one table and not the other (drives the
/// padding path), and duplicate keys on both sides (drives the many-to-many
/// bucket, where a dropped row is easiest to miss).
const L: &[Row] = &[
    (Some(1), 10),
    (Some(2), 20),
    (Some(2), 21),
    (None, 30),
    (Some(3), 40),
    (Some(7), 50),
];
const R: &[Row] = &[
    (Some(2), 200),
    (Some(3), 300),
    (Some(3), 301),
    (None, 400),
    (Some(9), 500),
];

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

impl Op {
    fn sql(self) -> &'static str {
        match self {
            Op::Inner => "l INNER JOIN r ON l.k = r.k",
            Op::Left => "l LEFT OUTER JOIN r ON l.k = r.k",
            Op::Right => "l RIGHT OUTER JOIN r ON l.k = r.k",
            Op::Full => "l FULL OUTER JOIN r ON l.k = r.k",
            // A comma join whose condition lives in the WHERE: the shape the
            // optimizer is expected to turn back into a hash join.
            Op::Cross => "l, r",
        }
    }
    fn name(self) -> &'static str {
        match self {
            Op::Inner => "INNER",
            Op::Left => "LEFT",
            Op::Right => "RIGHT",
            Op::Full => "FULL",
            Op::Cross => "comma",
        }
    }
}

const OPS: [Op; 5] = [Op::Inner, Op::Left, Op::Right, Op::Full, Op::Cross];

/// The join, by hand. `NULL = NULL` is unknown and therefore never a match, an
/// outer join pads the side that did not match, and a comma join is the full
/// product with no padding at all.
fn join_ref(op: Op) -> Vec<(Option<Row>, Option<Row>)> {
    let mut out = Vec::new();
    let mut r_hit = vec![false; R.len()];
    for l in L {
        let mut hit = false;
        for (j, r) in R.iter().enumerate() {
            let matched = match op {
                Op::Cross => true,
                _ => l.0.is_some() && l.0 == r.0,
            };
            if matched {
                out.push((Some(*l), Some(*r)));
                hit = true;
                r_hit[j] = true;
            }
        }
        if !hit && matches!(op, Op::Left | Op::Full) {
            out.push((Some(*l), None));
        }
    }
    for (j, r) in R.iter().enumerate() {
        if !r_hit[j] && matches!(op, Op::Right | Op::Full) {
            out.push((None, Some(*r)));
        }
    }
    out
}

/// One row of the join, as the four values a `SELECT l.k, l.v, r.k, r.v` sees.
/// A padded side is four NULLs, which is what makes the outer cases sharp.
fn cells(row: &(Option<Row>, Option<Row>)) -> [Option<i64>; 4] {
    let (l, r) = row;
    [
        l.and_then(|x| x.0),
        l.map(|x| x.1),
        r.and_then(|x| x.0),
        r.map(|x| x.1),
    ]
}

/// SQL's three-valued AND over the conjuncts of a WHERE clause: a row is kept
/// only when the predicate is TRUE, and UNKNOWN is not TRUE.
type Pred = fn([Option<i64>; 4]) -> Option<bool>;

fn eq(a: Option<i64>, b: i64) -> Option<bool> {
    a.map(|x| x == b)
}

/// `(SQL, the same thing in Rust)`. Every one is a shape some rule wants:
/// left-only, right-only, spanning, IS NULL (which a pushdown must not turn
/// into a row-dropper), and the two the normalizer rewrites before pushdown
/// even sees them.
const PREDS: &[(&str, Pred)] = &[
    ("l.k = 3", |c| eq(c[0], 3)),
    ("r.k = 3", |c| eq(c[2], 3)),
    ("l.k IS NULL", |c| Some(c[0].is_none())),
    ("r.k IS NOT NULL", |c| Some(c[2].is_some())),
    ("l.v > 25", |c| c[1].map(|x| x > 25)),
    ("l.v + r.v > 300", |c| match (c[1], c[3]) {
        (Some(a), Some(b)) => Some(a + b > 300),
        _ => None,
    }),
    ("l.k = 3 AND r.v > 300", |c| and(eq(c[0], 3), c[3].map(|x| x > 300))),
    ("NOT (l.k > 2)", |c| c[0].map(|x| !(x > 2))),
    ("l.k = 1 OR l.k = 3", |c| or(eq(c[0], 1), eq(c[0], 3))),
    ("NOT (l.k > 2 OR l.k < 2)", |c| {
        not(or(c[0].map(|x| x > 2), c[0].map(|x| x < 2)))
    }),
    ("r.k = 3 OR r.k = 9", |c| or(eq(c[2], 3), eq(c[2], 9))),
];

fn and(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}
fn or(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}
fn not(a: Option<bool>) -> Option<bool> {
    a.map(|x| !x)
}

fn db() -> Session {
    let mut s = Session::in_memory();
    for (name, rows) in [("l", L), ("r", R)] {
        s.execute(&format!(
            "CREATE TABLE {name} (k Nullable(Int64), v Int64) ENGINE = MergeTree ORDER BY v"
        ))
        .unwrap();
        let vals: Vec<String> = rows
            .iter()
            .map(|(k, v)| match k {
                Some(k) => format!("({k}, {v})"),
                None => format!("(NULL, {v})"),
            })
            .collect();
        s.execute(&format!("INSERT INTO {name} VALUES {}", vals.join(", "))).unwrap();
    }
    // A keyed pair, big enough that the plan assertions are about a query
    // somebody would actually run, and with a primary key so a sunk conjunct
    // can reach `IndexLookup` rather than merely `prewhere`.
    for name in ["a", "b"] {
        s.execute(&format!(
            "CREATE TABLE {name} (k UInt64, s Int64) ENGINE = MergeTree ORDER BY k PRIMARY KEY k"
        ))
        .unwrap();
        let vals: Vec<String> = (0..2000u64).map(|i| format!("({i}, {i})")).collect();
        s.execute(&format!("INSERT INTO {name} VALUES {}", vals.join(", "))).unwrap();
    }
    // Everything below reads through `&Session`, which cannot flush the delta
    // itself; and a scan only reaches the index and the zone maps once the
    // rows are in a part, which is the whole point of half the assertions.
    s.catalog.flush_all().unwrap();
    s
}

/// Every result row as a sorted list of strings, so a comparison is about the
/// multiset of rows and not about the order a hash join happened to emit them.
fn sorted(s: &Session, sql: &str) -> Vec<String> {
    let mut v: Vec<String> = s
        .read(sql)
        .unwrap_or_else(|e| panic!("`{sql}`: {e}"))
        .to_values()
        .iter()
        .map(|r| {
            r.iter()
                .map(|c| if c.is_null() { "NULL".into() } else { c.render_plain() })
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect();
    v.sort();
    v
}

fn expected(op: Op, pred: Pred, extra: Option<Pred>) -> Vec<String> {
    let mut v: Vec<String> = join_ref(op)
        .iter()
        .map(cells)
        .filter(|c| {
            pred(*c) == Some(true) && extra.is_none_or(|e| e(*c) == Some(true))
        })
        .map(|c| {
            c.iter()
                .map(|x| match x {
                    Some(n) => n.to_string(),
                    None => "NULL".into(),
                })
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect();
    v.sort();
    v
}

fn explain(s: &Session, sql: &str) -> String {
    s.read(&format!("EXPLAIN {sql}"))
        .unwrap_or_else(|e| panic!("EXPLAIN `{sql}`: {e}"))
        .to_values()
        .iter()
        .map(|r| r[0].render_plain())
        .collect::<Vec<_>>()
        .join("\n")
}

fn pipeline(s: &Session, sql: &str) -> String {
    s.read(&format!("EXPLAIN PIPELINE {sql}"))
        .unwrap()
        .to_values()
        .iter()
        .map(|r| r[0].render_plain())
        .collect::<Vec<_>>()
        .join("\n")
}

// ======================================================== 1. the answer holds

/// **The invariant the whole file exists for.**
///
/// Five join types x eleven predicates, against a nested loop that shares no
/// code with the planner. The fixture has NULL keys on both sides, a key only
/// one table has, and duplicate keys on both -- so a rule that sinks a
/// conjunct it may not sink shows up here as missing or extra rows, which is
/// the only symptom such a bug ever has.
#[test]
fn every_join_type_and_predicate_answers_what_a_nested_loop_says() {
    let s = db();
    for op in OPS {
        for (sql, pred) in PREDS {
            // The comma join carries its equi-condition in the WHERE, which is
            // the shape the optimizer turns back into a hash join; the
            // reference has to apply the same restriction.
            let extra: Option<Pred> = (op == Op::Cross).then_some(|c: [Option<i64>; 4]| {
                match (c[0], c[2]) {
                    (Some(a), Some(b)) => Some(a == b),
                    _ => None,
                }
            });
            let where_ = match op {
                Op::Cross => format!("l.k = r.k AND ({sql})"),
                _ => (*sql).to_string(),
            };
            let q = format!(
                "SELECT l.k, l.v, r.k, r.v FROM {} WHERE {where_}",
                op.sql()
            );
            assert_eq!(
                sorted(&s, &q),
                expected(op, *pred, extra),
                "{} JOIN, WHERE {sql}\n{}",
                op.name(),
                explain(&s, &q)
            );
        }
    }
}

/// The same invariant one level up: a conjunct that has to cross an aggregate,
/// a union or a distinct on its way down, over data with NULLs in the grouping
/// key.
#[test]
fn aggregate_union_and_distinct_answer_the_same_filtered_either_way() {
    let s = db();
    for (pushed, natural) in [
        // GROUP BY key, filtered above the aggregate vs below it.
        (
            "SELECT k, count() FROM l WHERE k = 2 GROUP BY k ORDER BY k",
            "SELECT k, count() FROM l GROUP BY k HAVING k = 2 ORDER BY k",
        ),
        // A NULL group is a group; a pushdown must not invent one or lose one.
        (
            "SELECT k, count() FROM l WHERE k IS NULL GROUP BY k ORDER BY k",
            "SELECT k, count() FROM l GROUP BY k HAVING k IS NULL ORDER BY k",
        ),
        // A computed group key: the conjunct has to be restated in terms of
        // the whole expression, not renumbered.
        (
            "SELECT v + 1 AS g, count() FROM l WHERE v + 1 = 31 GROUP BY g ORDER BY g",
            "SELECT v + 1 AS g, count() FROM l GROUP BY g HAVING g = 31 ORDER BY g",
        ),
        (
            "SELECT k FROM (SELECT k FROM l WHERE k = 2 UNION ALL SELECT k FROM r WHERE k = 2) u \
             ORDER BY k",
            "SELECT k FROM (SELECT k FROM l UNION ALL SELECT k FROM r) u WHERE k = 2 ORDER BY k",
        ),
        (
            "SELECT k FROM (SELECT k FROM l WHERE k IS NULL UNION ALL \
             SELECT k FROM r WHERE k IS NULL) u ORDER BY k",
            "SELECT k FROM (SELECT k FROM l UNION ALL SELECT k FROM r) u WHERE k IS NULL \
             ORDER BY k",
        ),
        (
            "SELECT k FROM (SELECT DISTINCT k FROM l WHERE k = 2) d ORDER BY k",
            "SELECT k FROM (SELECT DISTINCT k FROM l) d WHERE k = 2 ORDER BY k",
        ),
        // A distinct union: the conjunct crosses the union *and* the dedup.
        (
            "SELECT count() FROM (SELECT k FROM l WHERE k = 3 UNION DISTINCT \
             SELECT k FROM r WHERE k = 3) u",
            "SELECT count() FROM (SELECT k FROM l UNION DISTINCT SELECT k FROM r) u \
             WHERE k = 3",
        ),
    ] {
        assert_eq!(sorted(&s, pushed), sorted(&s, natural), "`{natural}`");
    }
    // ...and against values, so both spellings agreeing on the wrong thing is
    // still a failure. `l.k` is {1, 2, 2, NULL, 3, 7}.
    assert_eq!(
        s.read("SELECT count() FROM l GROUP BY k HAVING k = 2").unwrap().scalar(),
        Some(Value::UInt(2))
    );
    assert_eq!(
        s.read("SELECT count() FROM l GROUP BY k HAVING k IS NULL").unwrap().scalar(),
        Some(Value::UInt(1))
    );
}

/// The rewrites `normalize_predicates` makes are truth-table identities, not
/// approximations, and the NULL rows are where an approximation would show.
#[test]
fn normalized_predicates_admit_exactly_the_rows_they_did_before() {
    let s = db();
    for (a, b) in [
        ("NOT (k > 2)", "k <= 2"),
        ("NOT (k >= 2)", "k < 2"),
        ("NOT (k = 2)", "k != 2"),
        ("NOT (k IS NULL)", "k IS NOT NULL"),
        ("NOT (k IN (1, 2))", "k NOT IN (1, 2)"),
        ("NOT (k > 2 OR k < 2)", "k <= 2 AND k >= 2"),
        ("NOT (k > 2 AND k < 7)", "k <= 2 OR k >= 7"),
        ("NOT NOT (k > 2)", "k > 2"),
        ("k = 1 OR k = 3", "k IN (1, 3)"),
        ("k = 1 OR k = 3 OR k = 7", "k IN (1, 3, 7)"),
        ("k IN (1, 3) OR k = 7", "k IN (1, 3, 7)"),
        ("v + 0 = 30", "v = 30"),
        ("0 + v = 30", "v = 30"),
        ("v - 0 = 30", "v = 30"),
        ("v * 1 = 30", "v = 30"),
        ("v = 30 + 0", "v = 30"),
        // The one that must NOT be rewritten, asserted as an answer rather
        // than as a plan: `k = 1 OR k = NULL` is TRUE for k = 1 and UNKNOWN
        // everywhere else, and an `IN` list built with the NULL dropped would
        // answer FALSE instead -- which admits the same rows here, but is a
        // different value in a projection.
        ("k = 1 OR k = NULL", "k = 1 OR NULL"),
    ] {
        assert_eq!(
            sorted(&s, &format!("SELECT k, v FROM l WHERE {a} ORDER BY v")),
            sorted(&s, &format!("SELECT k, v FROM l WHERE {b} ORDER BY v")),
            "`{a}` and `{b}` must admit the same rows"
        );
    }
    // Values, so that two spellings agreeing on nothing is caught: `l.v` is
    // {10, 20, 21, 30, 40, 50} with k = {1, 2, 2, NULL, 3, 7}.
    let n = |w: &str| {
        s.read(&format!("SELECT count() FROM l WHERE {w}")).unwrap().scalar().unwrap()
    };
    assert_eq!(n("NOT (k > 2)"), Value::UInt(3), "k in {{1, 2, 2}}; the NULL is UNKNOWN");
    assert_eq!(n("k = 1 OR k = 3"), Value::UInt(2));
    assert_eq!(n("NOT (k > 2 OR k < 2)"), Value::UInt(2), "k = 2 twice");
}

/// `k + 0` is not `k` on an unsigned column, and this is the row that proves
/// it: `promote` makes the sum `Int64`, so the largest `UInt64` renders -1.
/// The rewrite is guarded on the declared type, and this asserts the guard
/// rather than the rewrite.
#[test]
fn an_arithmetic_identity_that_would_change_the_type_is_not_applied() {
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE u (k UInt64) ENGINE = MergeTree ORDER BY k").unwrap();
    s.execute("INSERT INTO u VALUES (1), (18446744073709551615)").unwrap();
    s.catalog.flush_all().unwrap();
    assert_eq!(
        s.read("SELECT count() FROM u WHERE k + 0 = 18446744073709551615").unwrap().scalar(),
        Some(Value::UInt(0)),
        "`k + 0` is Int64 and wraps to -1; stripping the `+ 0` would answer 1"
    );
    assert_eq!(
        s.read("SELECT count() FROM u WHERE k = 18446744073709551615").unwrap().scalar(),
        Some(Value::UInt(1))
    );
    assert!(
        explain(&s, "SELECT count() FROM u WHERE k + 0 = 1").contains("+ 0"),
        "the guard must leave the expression alone"
    );
}

// ========================================================== 2. the plan moved

/// Without this, every assertion above would still pass with the whole pass
/// deleted.
#[test]
fn a_single_side_conjunct_reaches_both_scans() {
    let s = db();
    let q = "SELECT count() FROM a JOIN b ON a.k = b.k WHERE a.k = 1500";
    let e = explain(&s, q);
    assert!(!e.contains("Filter"), "the conjunct should have left the join behind:\n{e}");
    assert_eq!(
        e.matches("prewhere=(k#0 = 1500)").count(),
        2,
        "both scans should carry it -- the right one by inference:\n{e}"
    );
    // ...and it reaches all the way to the index, which is the point.
    let p = pipeline(&s, q);
    assert_eq!(p.matches("IndexLookup").count(), 2, "{p}");
    assert_eq!(s.read(q).unwrap().scalar(), Some(Value::UInt(1)));
}

/// A three-table star: the conjunct is written on the first table, and the
/// third is reached only through the equivalence closure over both `ON`
/// clauses. One hop leaves the third table scanning (measured 3.95 ms against
/// 0.049 ms at 300k rows), so this asserts the closure and not just the hop.
#[test]
fn a_conjunct_reaches_every_table_of_a_three_way_join() {
    let s = db();
    let q = "SELECT count() FROM a x JOIN b y ON x.k = y.k JOIN a z ON y.k = z.k \
             WHERE x.k = 1500";
    let e = explain(&s, q);
    assert_eq!(e.matches("prewhere=(k#0 = 1500)").count(), 3, "{e}");
    assert!(!e.contains("Filter"), "{e}");
    assert_eq!(s.read(q).unwrap().scalar(), Some(Value::UInt(1)));
}

/// The equi-condition of a comma join lives in the WHERE, and leaving it there
/// costs a full cross product. It becomes the join condition instead, which
/// also makes the join an inner join by name as well as by behaviour.
#[test]
fn a_spanning_equality_becomes_the_join_condition() {
    let s = db();
    let e = explain(&s, "SELECT count() FROM a, b WHERE a.k = b.k");
    assert!(e.contains("InnerJoin on [l#0 = r#0]"), "{e}");
    assert!(!e.contains("CrossJoin"), "{e}");
    assert!(!e.contains("Filter"), "{e}");
    assert_eq!(
        s.read("SELECT count() FROM a, b WHERE a.k = b.k").unwrap().scalar(),
        Some(Value::UInt(2000))
    );
    // A comma join with no condition at all is still a cross product, and
    // still says so.
    assert!(explain(&s, "SELECT count() FROM l, r").contains("CrossJoin on []"));
}

#[test]
fn a_group_key_conjunct_leaves_the_aggregate_behind() {
    let s = db();
    for q in [
        "SELECT k, count() FROM a GROUP BY k HAVING k = 1500",
        "SELECT count() FROM (SELECT k, count() c FROM a GROUP BY k) g WHERE k = 1500",
    ] {
        let e = explain(&s, q);
        assert!(!e.contains("Filter"), "{q}:\n{e}");
        assert!(e.contains("prewhere=(k#0 = 1500)"), "{q}:\n{e}");
    }
    // A computed group key is substituted whole, not renumbered.
    let e = explain(&s, "SELECT k + 1 AS g, count() FROM a GROUP BY g HAVING g = 1500");
    assert!(e.contains("prewhere=((k#0 + 1) = 1500)"), "{e}");
}

#[test]
fn a_conjunct_enters_every_union_branch_and_passes_through_distinct() {
    let s = db();
    let e = explain(
        &s,
        "SELECT count() FROM (SELECT k FROM a UNION ALL SELECT k FROM b) u WHERE k = 1500",
    );
    assert_eq!(e.matches("prewhere=(k#0 = 1500)").count(), 2, "{e}");
    assert!(!e.contains("Filter"), "{e}");

    let e = explain(&s, "SELECT count() FROM (SELECT DISTINCT k FROM a) d WHERE k = 1500");
    assert!(e.contains("Distinct"), "{e}");
    assert!(e.contains("prewhere=(k#0 = 1500)"), "{e}");
    assert!(!e.contains("Filter"), "{e}");
}

#[test]
fn a_reshaped_predicate_reaches_the_zone_maps_and_the_index() {
    let s = db();
    for (q, want) in [
        ("SELECT count() FROM a WHERE NOT (k > 1500)", "prewhere=(k#0 <= 1500)"),
        ("SELECT count() FROM a WHERE NOT (k > 1500 OR k < 3)", "zonemap=2"),
        ("SELECT count() FROM a WHERE k = 1 OR k = 2 OR k = 3", "prewhere=k#0 IN (1, 2, 3)"),
        ("SELECT count() FROM a WHERE s + 0 = 1500", "prewhere=(s#0 = 1500)"),
        ("SELECT count() FROM a WHERE k = CAST(1500 AS UInt64)", "prewhere=(k#0 = 1500)"),
    ] {
        let e = explain(&s, q);
        assert!(e.contains(want), "{q} should show `{want}`:\n{e}");
    }
    // An OR-chain on the key becomes an `IN`, and an `IN` is an index probe.
    let p = pipeline(&s, "SELECT s FROM a WHERE k = 1 OR k = 2 OR k = 3");
    assert!(p.contains("IndexLookup"), "{p}");
    assert_eq!(s.read("SELECT count() FROM a WHERE k = 1 OR k = 2 OR k = 3").unwrap().scalar(),
               Some(Value::UInt(3)));
}

// ================================================ 3. the plan does NOT move
//
// Timings cannot prove a rule did not fire on a machine that swings 30% on
// identical code, so every negative case is stated as a plan: the conjunct is
// still a `Filter`, in the place the query put it.

/// The classic pushdown bug, in all four spellings that would produce it. Each
/// of these would silently convert an outer join to an inner one.
#[test]
fn an_outer_join_keeps_the_conjunct_on_its_nullable_side() {
    let s = db();
    for (q, why) in [
        (
            "SELECT count() FROM l LEFT OUTER JOIN r ON l.k = r.k WHERE r.k = 3",
            "a LEFT JOIN's right side is NULL-padded; sinking this drops those rows",
        ),
        (
            "SELECT count() FROM l RIGHT OUTER JOIN r ON l.k = r.k WHERE l.k = 3",
            "the mirror image",
        ),
        (
            "SELECT count() FROM l FULL OUTER JOIN r ON l.k = r.k WHERE l.k = 3",
            "FULL pads both sides, so neither may sink",
        ),
        (
            "SELECT count() FROM l FULL OUTER JOIN r ON l.k = r.k WHERE r.k = 3",
            "FULL pads both sides, so neither may sink",
        ),
    ] {
        let e = explain(&s, q);
        assert!(e.contains("Filter"), "{why}\n{q}:\n{e}");
        assert!(!e.contains("prewhere"), "{why}\n{q}:\n{e}");
    }
}

/// The counterpart, so the pair of tests pins the *direction*: on a LEFT JOIN
/// a left-side conjunct sinks, and its inferred copy is allowed into the right
/// input even though a hand-written one there would not be.
#[test]
fn an_outer_join_still_prunes_the_side_that_is_never_padded() {
    let s = db();
    let q = "SELECT count() FROM a LEFT OUTER JOIN b ON a.k = b.k WHERE a.k = 1500";
    let e = explain(&s, q);
    assert_eq!(e.matches("prewhere=(k#0 = 1500)").count(), 2, "{e}");
    assert!(!e.contains("Filter"), "{e}");
    assert_eq!(s.read(q).unwrap().scalar(), Some(Value::UInt(1)));
}

#[test]
fn the_shapes_no_rule_may_touch_are_planned_exactly_as_written() {
    let s = db();
    for (q, why) in [
        (
            "SELECT count() FROM a JOIN b ON a.k = b.k WHERE a.s + b.s = 8",
            "spans both sides and is not an equality between two columns",
        ),
        (
            "SELECT k, count() c FROM a GROUP BY k HAVING count() > 0",
            "reads an aggregate result, which is what HAVING is for",
        ),
        (
            "SELECT count() FROM (SELECT k FROM a UNION ALL SELECT k FROM b) u WHERE rand() = 7",
            "rand() answers differently once it runs over different rows",
        ),
        (
            "SELECT count() FROM a JOIN b ON a.k = b.k WHERE rand() = 7",
            "same, through a join",
        ),
        (
            "SELECT count() FROM (SELECT k FROM a LIMIT 100) q WHERE k = 1500",
            "LIMIT chooses rows by position, so filtering first changes which",
        ),
        (
            "SELECT count() FROM (SELECT k, row_number() OVER (ORDER BY k) rn FROM a) w \
             WHERE k = 1500",
            "a window reads its whole partition, so it cannot lose rows first",
        ),
    ] {
        let e = explain(&s, q);
        assert!(e.contains("Filter"), "{why}\n{q}:\n{e}");
    }
}

/// The two shaping rules that look applicable and are not.
#[test]
fn the_predicate_shapes_the_normalizer_must_decline() {
    let s = db();
    // `NOT NOT v` where `v` is an integer: the inner `NOT v` is boolean but
    // `v` is not, so collapsing the pair would answer `v` where SQL answers
    // `v != 0`.
    let e = explain(&s, "SELECT count() FROM a WHERE NOT (NOT s)");
    assert!(e.contains("NOT (NOT"), "{e}");
    // Two columns is not an OR-chain on one.
    let e = explain(&s, "SELECT count() FROM a WHERE k = 1 OR s = 2");
    assert!(e.contains("OR"), "{e}");
    assert!(!e.contains(" IN ("), "{e}");
    // A NULL probe stays an OR: an `IN` list would have to carry the NULL, and
    // both `as_zone_filter` and `key_set` would then need a rule for it.
    let e = explain(&s, "SELECT count() FROM a WHERE k = 1 OR k = NULL");
    assert!(!e.contains(" IN ("), "{e}");
}

/// A pushdown that reaches an empty relation must still produce the schema the
/// rest of the plan was built against, and a contradiction under a join must
/// not turn an outer join's padded rows into no rows.
#[test]
fn a_contradiction_under_a_join_still_answers_with_the_right_shape() {
    let s = db();
    assert_eq!(
        s.read("SELECT count() FROM a JOIN b ON a.k = b.k WHERE a.k = 1 AND a.k = 2")
            .unwrap()
            .scalar(),
        Some(Value::UInt(0))
    );
    // The left rows survive; every right column is padded.
    assert_eq!(
        sorted(
            &s,
            "SELECT l.v, r.v FROM l LEFT OUTER JOIN r ON l.k = r.k WHERE l.k = 7 AND 1 = 2"
        ),
        Vec::<String>::new()
    );
    assert_eq!(
        sorted(&s, "SELECT l.v, r.v FROM l LEFT OUTER JOIN r ON l.k = r.k WHERE l.k = 7"),
        vec!["50|NULL".to_string()]
    );
}
