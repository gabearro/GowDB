//! `GROUPING SETS`, `ROLLUP`, `CUBE` and `GROUPING()`, end to end through
//! `Session` and the CLI binary.
//!
//! # How the answers are checked
//!
//! Three ways, none of which is the feature checking itself.
//!
//!   * **Against a hand-written `UNION ALL`** of the plain `GROUP BY`s the
//!     grouping asks for. That is what a user writes without this feature, so
//!     it is both the reference answer and the thing the feature claims to
//!     replace.
//!   * **Against every subset, generated**. `every_grouping_set_equals_the_
//!     plain_group_by_over_its_columns` builds random set lists over random
//!     data and checks each set's rows against `GROUP BY <that subset>` run on
//!     its own. sqlite has no `GROUPING SETS`, so the differential oracle
//!     cannot reach this; a property over the engine's own `GROUP BY` -- which
//!     30 000 differential cases *do* check against sqlite -- is the stronger
//!     statement available, and it covers set lists nobody would think to
//!     write by hand.
//!   * **Against itself under a forced spill**, through the binary with
//!     `GRANULAR_SPILL_ROWS` set, because the spilled path re-reads rows from
//!     disk and has to remember which grouping set each of them missed under.
//!
//! # One pass, not N
//!
//! That is the entire point -- `CUBE(a, b, c)` is eight groupings, and a
//! `UNION ALL` of eight aggregates reads the table eight times.
//! `a_cube_reads_the_table_once` measures it off `system.query_log`'s
//! `rows_scanned` rather than asserting it about the plan text, so it fails if
//! the single pass is ever quietly desugared away.
//!
//! # Reachability
//!
//! `grouping_sets_reach_the_executor` is first, and it is one line per
//! spelling through `Session::query`. Nine capabilities in this engine's
//! history landed complete in `src/` and were never wired to a session.
//! `the_cli_runs_a_rollup` does the same through the shipped binary.

use std::collections::BTreeMap;
use std::process::Command;

use granular::types::Value;
use granular::Session;

// ------------------------------------------------------------------ helpers

/// `(a, b)` with a NULL in each, so an aggregated-away key and a NULL in the
/// data collide in every test that does not use `GROUPING()`.
fn db() -> Session {
    let mut db = Session::in_memory();
    for stmt in [
        "CREATE TABLE t (id UInt64, a Nullable(String), b Nullable(String), v Int64) \
         ENGINE = MergeTree ORDER BY id",
        "INSERT INTO t VALUES (1,'x','p',1), (2,'x','q',2), (3,'y','p',4), (4,'y','q',8), \
         (5,'y','q',16), (6, NULL,'p',32), (7,'x', NULL,64), (8, NULL, NULL,128)",
    ] {
        db.execute(stmt).unwrap_or_else(|e| panic!("{stmt}: {e}"));
    }
    db
}

fn rows(db: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    db.query(sql).unwrap_or_else(|e| panic!("{sql}\n  -> {e}")).to_values()
}

/// One value as comparable text. Rendered, not debug-printed: a hand-written
/// `SELECT NULL` types its column from the literal, so the reference query's
/// `Int(0)` and the grouping set's `UInt(0)` are the same answer and must
/// compare equal. NULL gets a spelling no string literal in these tests uses.
fn text(v: &Value) -> String {
    match v {
        Value::Null => "~".into(),
        v => v.to_string(),
    }
}

/// Rows as comparable text, sorted, so two queries are compared as multisets
/// and not against an output order neither of them promises.
fn bag(db: &mut Session, sql: &str) -> Vec<String> {
    let mut v: Vec<String> =
        rows(db, sql).into_iter().map(|r| r.iter().map(text).collect::<Vec<_>>().join("|")).collect();
    v.sort();
    v
}

fn err(db: &mut Session, sql: &str) -> String {
    match db.query(sql) {
        Ok(_) => panic!("`{sql}` was accepted"),
        Err(e) => e.to_string(),
    }
}

// -------------------------------------------------------------- reachability

/// PIN. Every spelling, through the public API, in one place. If the parser,
/// the binder or the operator is wired up but not to each other, this is the
/// test that says so.
#[test]
fn grouping_sets_reach_the_executor() {
    let mut db = db();
    for (sql, want) in [
        // 7 distinct (a, b) pairs, 3 distinct `a`, 3 distinct `b`, NULLs
        // included -- they are ordinary group keys, which is the whole
        // difficulty `GROUPING()` exists for.
        ("SELECT a, b, count() FROM t GROUP BY GROUPING SETS ((a, b), (a), ())", 7 + 3 + 1),
        ("SELECT a, b, count() FROM t GROUP BY ROLLUP(a, b)", 7 + 3 + 1),
        ("SELECT a, b, count() FROM t GROUP BY CUBE(a, b)", 7 + 3 + 3 + 1),
        ("SELECT a, count(), GROUPING(a) FROM t GROUP BY ROLLUP(a)", 3 + 1),
    ] {
        assert_eq!(rows(&mut db, sql).len(), want, "{sql}");
    }
}

#[test]
fn the_cli_runs_a_rollup() {
    let out = Command::new(env!("CARGO_BIN_EXE_granular"))
        .arg("-q")
        .arg(
            "CREATE TABLE t (k UInt8, v Int64) ENGINE = MergeTree ORDER BY k; \
             INSERT INTO t VALUES (1, 10), (1, 20), (2, 5); \
             SELECT k, sum(v), GROUPING(k) FROM t GROUP BY ROLLUP(k) ORDER BY GROUPING(k), k",
        )
        .output()
        .expect("run the binary");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{text}{}", String::from_utf8_lossy(&out.stderr));
    // Two groups and the grand total, and the total is flagged.
    for want in ["│ 1 │ 30", "│ 2 │ 5", "│ 35     │ 1"] {
        assert!(text.contains(want), "missing `{want}` in\n{text}");
    }
}

// ----------------------------------------------- against a hand-written UNION

/// The desugaring a user writes today. `ROLLUP(a, b)` is these three
/// aggregates concatenated, and the whole claim of the feature is that it is
/// the same answer for one scan instead of three.
const ROLLUP_AB_BY_HAND: &str = "SELECT a, b, sum(v) FROM t GROUP BY a, b \
     UNION ALL SELECT a, CAST(NULL AS Nullable(String)), sum(v) FROM t GROUP BY a \
     UNION ALL SELECT CAST(NULL AS Nullable(String)), CAST(NULL AS Nullable(String)), sum(v) \
     FROM t";

#[test]
fn rollup_is_the_union_of_its_groupings() {
    let mut db = db();
    assert_eq!(
        bag(&mut db, "SELECT a, b, sum(v) FROM t GROUP BY ROLLUP(a, b)"),
        bag(&mut db, ROLLUP_AB_BY_HAND),
    );
}

#[test]
fn cube_is_the_union_of_its_groupings() {
    let mut db = db();
    let n = "CAST(NULL AS Nullable(String))";
    let by_hand = format!(
        "SELECT a, b, sum(v) FROM t GROUP BY a, b \
         UNION ALL SELECT a, {n}, sum(v) FROM t GROUP BY a \
         UNION ALL SELECT {n}, b, sum(v) FROM t GROUP BY b \
         UNION ALL SELECT {n}, {n}, sum(v) FROM t"
    );
    assert_eq!(
        bag(&mut db, "SELECT a, b, sum(v) FROM t GROUP BY CUBE(a, b)"),
        bag(&mut db, &by_hand),
    );
}

/// `GROUP BY a, ROLLUP(b)` is the cross product of the two items, which is
/// two groupings and not three -- `a` is in every one of them.
#[test]
fn a_plain_key_beside_a_rollup_multiplies_out() {
    let mut db = db();
    let by_hand = "SELECT a, b, sum(v) FROM t GROUP BY a, b \
         UNION ALL SELECT a, CAST(NULL AS Nullable(String)), sum(v) FROM t GROUP BY a";
    assert_eq!(
        bag(&mut db, "SELECT a, b, sum(v) FROM t GROUP BY a, ROLLUP(b)"),
        bag(&mut db, by_hand),
    );
}

// ------------------------------------------------------------------ GROUPING

/// The case `GROUPING()` exists for, and it needs both NULLs in one query to
/// be a real test: `a IS NULL` is true for the row whose `a` is genuinely
/// NULL *and* for the row that aggregated `a` away, and only `GROUPING(a)`
/// tells them apart.
#[test]
fn grouping_tells_an_aggregated_null_from_a_data_null() {
    let mut db = db();
    let got = rows(
        &mut db,
        "SELECT GROUPING(a) AS g, sum(v) FROM t GROUP BY ROLLUP(a) HAVING a IS NULL ORDER BY g",
    );
    // Two rows, both with a NULL `a`: the group of the rows that *have* no
    // `a` (32 + 128), and the grand total (all eight rows).
    assert_eq!(
        got,
        vec![
            vec![Value::UInt(0), Value::Int(32 + 128)],
            vec![Value::UInt(1), Value::Int(255)],
        ],
        "an aggregated-away NULL and a data NULL are not being distinguished"
    );
}

/// ANSI packs several columns into one integer, left to right: `GROUPING(a,b)`
/// is 2 when only `a` was dropped, 1 when only `b` was, 3 for the total.
#[test]
fn grouping_of_several_columns_is_a_bitmap() {
    let mut db = db();
    let got = rows(
        &mut db,
        "SELECT g, count() FROM (SELECT GROUPING(a, b) AS g, count() AS n FROM t \
          GROUP BY CUBE(a, b)) GROUP BY g ORDER BY g",
    );
    assert_eq!(
        got,
        vec![
            vec![Value::UInt(0), Value::UInt(7)],
            vec![Value::UInt(1), Value::UInt(3)],
            vec![Value::UInt(2), Value::UInt(3)],
            vec![Value::UInt(3), Value::UInt(1)],
        ]
    );
}

/// A `GROUPING()` whose answer does not depend on the set is a constant, and
/// the binder folds it to one rather than emitting a lookup.
#[test]
fn grouping_over_a_key_every_set_keeps_is_a_constant() {
    let mut db = db();
    let got = rows(&mut db, "SELECT DISTINCT GROUPING(a) FROM t GROUP BY a, ROLLUP(b)");
    assert_eq!(got, vec![vec![Value::UInt(0)]]);
}

#[test]
fn grouping_refuses_what_it_cannot_answer() {
    let mut db = db();
    let e = err(&mut db, "SELECT GROUPING(a) FROM t GROUP BY a");
    assert!(e.contains("GROUPING SETS, ROLLUP or CUBE"), "{e}");
    let e = err(&mut db, "SELECT GROUPING(v) FROM t GROUP BY ROLLUP(a)");
    assert!(e.contains("not one of this query's grouping columns"), "{e}");
    let e = err(&mut db, "SELECT GROUPING(sum(v)) FROM t GROUP BY ROLLUP(a)");
    assert!(e.contains("not allowed here"), "{e}");
}

// -------------------------------------------------- the awkward set lists

/// `()` on its own is the grand total: one row, no grouping at all.
#[test]
fn the_empty_grouping_set_is_the_grand_total() {
    let mut db = db();
    // A key no set keeps is not a key at all, which is why this must name
    // none: ANSI (and Postgres) reject `SELECT a` here for the same reason
    // `SELECT a FROM t` without a GROUP BY is rejected.
    assert_eq!(
        rows(&mut db, "SELECT sum(v) FROM t GROUP BY GROUPING SETS (())"),
        rows(&mut db, "SELECT sum(v) FROM t"),
    );
    let e = err(&mut db, "SELECT a, sum(v) FROM t GROUP BY GROUPING SETS (())");
    assert!(e.contains("must appear in GROUP BY"), "{e}");
}

/// A set list may repeat itself, and it must then answer twice. This is why
/// the executor keys a group on a set **id** and not on the set's column mask:
/// two identical sets are two groupings, and deduplicating them would be a
/// wrong answer that looks tidier.
#[test]
fn a_repeated_grouping_set_answers_twice() {
    let mut db = db();
    let once = bag(&mut db, "SELECT a, count() FROM t GROUP BY GROUPING SETS ((a))");
    let twice = bag(&mut db, "SELECT a, count() FROM t GROUP BY GROUPING SETS ((a), (a))");
    assert_eq!(twice.len(), 2 * once.len(), "a duplicate set was silently deduplicated");
    let mut doubled: Vec<String> = once.iter().chain(once.iter()).cloned().collect();
    doubled.sort();
    assert_eq!(twice, doubled);
}

/// A key written twice is still one column -- and one that every set keeps,
/// so this is an ordinary `GROUP BY a` however elaborately it is spelled.
#[test]
fn a_key_repeated_across_items_is_one_column() {
    let mut db = db();
    assert_eq!(
        bag(&mut db, "SELECT a, count() FROM t GROUP BY GROUPING SETS ((a, a))"),
        bag(&mut db, "SELECT a, count() FROM t GROUP BY a"),
    );
}

/// A bare column in a set list is the one-column set, which is how ANSI lets
/// `GROUPING SETS (a, b)` mean `((a), (b))`.
#[test]
fn a_bare_column_in_a_set_list_is_a_set_of_one() {
    let mut db = db();
    assert_eq!(
        bag(&mut db, "SELECT a, b, count() FROM t GROUP BY GROUPING SETS (a, b)"),
        bag(&mut db, "SELECT a, b, count() FROM t GROUP BY GROUPING SETS ((a), (b))"),
    );
}

/// Grouping by an expression, an alias and an ordinal, all of which the
/// ordinary `GROUP BY` accepts and all of which have to reach the same column.
#[test]
fn a_grouping_set_may_name_a_key_however_group_by_can() {
    let mut db = db();
    let want = bag(
        &mut db,
        "SELECT upper(a) AS ua, sum(v) FROM t GROUP BY upper(a) \
         UNION ALL SELECT CAST(NULL AS Nullable(String)), sum(v) FROM t",
    );
    for sql in [
        "SELECT upper(a) AS ua, sum(v) FROM t GROUP BY ROLLUP(upper(a))",
        "SELECT upper(a) AS ua, sum(v) FROM t GROUP BY ROLLUP(ua)",
        "SELECT upper(a) AS ua, sum(v) FROM t GROUP BY ROLLUP(1)",
    ] {
        assert_eq!(bag(&mut db, sql), want, "{sql}");
    }
}

// ------------------------------------------------------- HAVING and ORDER BY

/// PIN. `HAVING a = 'x'` must run **above** the aggregate. Pushing it below --
/// which the optimizer does for an ordinary `GROUP BY`, and which is a 400x
/// win there -- would pre-filter the rows the grand total sums *and* keep the
/// total's row, whose `a` is NULL and which the filter above removes. The
/// `Aggregate` node's empty `group` list is what makes the rule decline; if
/// that ever changes this test is how it is noticed.
#[test]
fn having_over_a_grouping_key_is_not_pushed_below_the_aggregate() {
    let mut db = db();
    let plan = rows(&mut db, "EXPLAIN SELECT a, sum(v) FROM t GROUP BY ROLLUP(a) HAVING a = 'x'");
    let text: Vec<String> = plan.iter().map(|r| format!("{}", r[0])).collect();
    let text = text.join("\n");
    let (f, a) = (
        text.find("Filter").expect("the HAVING filter"),
        text.find("Aggregate").expect("the aggregate"),
    );
    assert!(f < a, "the HAVING filter sank below the aggregate:\n{text}");
    assert_eq!(
        rows(&mut db, "SELECT a, sum(v) FROM t GROUP BY ROLLUP(a) HAVING a = 'x'"),
        vec![vec![Value::Str("x".into()), Value::Int(1 + 2 + 64)]],
    );
    // The same rule reached the other way: an outer `WHERE` over a derived
    // table, which the planner inlines and pushes through the projection.
    // Both rows here have a NULL `a` -- one is the data's, one is the total's
    // -- so a filter that sank below the aggregate would keep the wrong one.
    assert_eq!(
        bag(&mut db, "SELECT a, sum(v) AS s FROM (SELECT * FROM t) u GROUP BY ROLLUP(a)"),
        bag(&mut db, "SELECT a, sum(v) AS s FROM t GROUP BY ROLLUP(a)"),
    );
    assert_eq!(
        rows(
            &mut db,
            "SELECT s FROM (SELECT a, sum(v) AS s FROM t GROUP BY ROLLUP(a)) u \
             WHERE a IS NULL ORDER BY s"
        ),
        vec![vec![Value::Int(32 + 128)], vec![Value::Int(255)]],
    );
}

#[test]
fn having_and_order_by_may_read_grouping() {
    let mut db = db();
    let got = rows(
        &mut db,
        "SELECT a, sum(v) FROM t GROUP BY CUBE(a, b) HAVING GROUPING(a) = 1 AND GROUPING(b) = 1",
    );
    assert_eq!(got, vec![vec![Value::Null, Value::Int(255)]]);
    let ordered = rows(
        &mut db,
        "SELECT GROUPING(a, b) AS g FROM t GROUP BY CUBE(a, b) ORDER BY GROUPING(a, b) DESC, g",
    );
    assert_eq!(ordered.first(), Some(&vec![Value::UInt(3)]));
}

// ------------------------------------------------------------ the property

/// Every grouping set's rows equal the plain `GROUP BY` over exactly that
/// subset of columns.
///
/// This is the check sqlite cannot give us -- it has no `GROUPING SETS` -- and
/// it is a stronger statement than any example: it holds for set lists nobody
/// would write, over data with NULLs, repeated keys and empty sets, and it is
/// checked against the engine's *own* `GROUP BY`, which 30 000 differential
/// cases already hold to sqlite's answer.
///
/// The set id is what makes the split possible from outside: `GROUPING(a, b,
/// c)` names each row's set exactly, so the result can be cut into the
/// groupings it came from and each piece compared on its own.
#[test]
fn every_grouping_set_equals_the_plain_group_by_over_its_columns() {
    let mut db = Session::in_memory();
    db.execute(
        "CREATE TABLE p (id UInt64, a Nullable(Int64), b Nullable(String), c Int64, v Int64) \
         ENGINE = MergeTree ORDER BY id",
    )
    .unwrap();
    // Deterministic, and deliberately lumpy: a handful of distinct values per
    // column so sets genuinely collapse rows together, plus NULLs in two of
    // the three key columns.
    let mut seed = 0x243F_6A88_85A3_08D3u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut vals = Vec::new();
    for id in 0..400u64 {
        let (x, y, z, v) = (next() % 5, next() % 4, next() % 3, next() % 100);
        let a = if x == 4 { "NULL".to_string() } else { x.to_string() };
        let b = if y == 3 { "NULL".to_string() } else { format!("'s{y}'") };
        vals.push(format!("({id}, {a}, {b}, {z}, {v})"));
    }
    db.execute(&format!("INSERT INTO p VALUES {}", vals.join(", "))).unwrap();

    // A set is a subset of these three, written as the bits it *drops* so it
    // reads the same way `GROUPING()` answers.
    let cols = ["a", "b", "c"];
    let subset = |drop: usize| -> Vec<&str> {
        (0..3).filter(|k| drop >> (2 - k) & 1 == 0).map(|k| cols[k]).collect()
    };

    // Every pair of the eight subsets, plus eight four-set lists with a
    // deliberate duplicate in each. 8^2 + 8 keeps this a second rather than a
    // minute, and every set from the full grouping to the grand total appears.
    let mut lists: Vec<Vec<usize>> = Vec::new();
    for i in 0..8 {
        for j in 0..8 {
            lists.push(vec![i, j]);
        }
    }
    for i in 0..8 {
        lists.push(vec![i, (i + 3) % 8, (i * 5) % 8, i]);
    }

    for list in &lists {
        let sets: Vec<String> =
            list.iter().map(|&m| format!("({})", subset(m).join(", "))).collect();
        // `GROUPING()` may only name a column this query groups by, so ask
        // about exactly the ones some set keeps -- the rest are not grouping
        // columns at all and the binder is right to refuse them.
        let named: Vec<&str> =
            cols.iter().copied().filter(|c| list.iter().any(|&m| subset(m).contains(c))).collect();
        if named.is_empty() {
            continue;
        }
        let w = named.len();
        let sql = format!(
            "SELECT GROUPING({}) AS g, {}, sum(v) AS s, count() AS n FROM p \
             GROUP BY GROUPING SETS ({})",
            named.join(", "),
            named.join(", "),
            sets.join(", ")
        );
        // Split the answer by set, and compare each piece with the plain
        // GROUP BY over that subset. Sets that repeat contribute their rows
        // that many times, which is what the multiplicity here checks.
        let mut got: BTreeMap<u64, Vec<String>> = BTreeMap::new();
        for r in rows(&mut db, &sql) {
            let Value::UInt(g) = r[0] else { panic!("GROUPING is not an integer: {:?}", r[0]) };
            // The dropped columns must come back NULL, whatever the data held.
            for k in 0..w {
                if g >> (w - 1 - k) & 1 == 1 {
                    assert_eq!(r[1 + k], Value::Null, "{sql}\n  key {k} survived its set");
                }
            }
            let mut cells: Vec<String> =
                (0..w).filter(|k| g >> (w - 1 - k) & 1 == 0).map(|k| text(&r[1 + k])).collect();
            cells.push(text(&r[1 + w]));
            cells.push(text(&r[2 + w]));
            got.entry(g).or_default().push(cells.join("|"));
        }
        for (&g, rows_here) in got.iter_mut() {
            rows_here.sort();
            let keys: Vec<&str> =
                (0..w).filter(|k| g >> (w - 1 - k) & 1 == 0).map(|k| named[k]).collect();
            let plain = if keys.is_empty() {
                "SELECT sum(v) AS s, count() AS n FROM p".to_string()
            } else {
                format!(
                    "SELECT {}, sum(v) AS s, count() AS n FROM p GROUP BY {}",
                    keys.join(", "),
                    keys.join(", ")
                )
            };
            let n = list.iter().filter(|&&m| subset(m) == keys).count();
            let mut want: Vec<String> = bag(&mut db, &plain);
            want = want.iter().cycle().take(want.len() * n).cloned().collect();
            want.sort();
            assert_eq!(*rows_here, want, "{sql}\n  set {keys:?} disagrees with `{plain}`");
        }
        // As many groupings came back as the list asked for, duplicates
        // collapsed -- they answer under the same `GROUPING()` bitmap and were
        // checked above by multiplicity.
        let distinct: std::collections::BTreeSet<Vec<&str>> =
            list.iter().map(|&m| subset(m)).collect();
        assert_eq!(got.len(), distinct.len(), "{sql}");
    }
}

// -------------------------------------------------------------- one pass

/// PIN, and the whole reason this is not sugar. `CUBE(a, b, c)` is eight
/// groupings; the `UNION ALL` that means the same thing reads the table eight
/// times. Measured off `system.query_log`, not asserted about the plan text,
/// so a desugaring that produced the right rows the slow way would still fail.
#[test]
fn a_cube_reads_the_table_once() {
    let mut db = Session::in_memory();
    db.execute("CREATE TABLE s (a UInt8, b UInt8, c UInt8, v Int64) ENGINE = MergeTree ORDER BY a")
        .unwrap();
    let vals: Vec<String> =
        (0..2000u32).map(|i| format!("({}, {}, {}, {i})", i % 7, i % 5, i % 3)).collect();
    db.execute(&format!("INSERT INTO s VALUES {}", vals.join(", "))).unwrap();

    let by_hand = "SELECT a,b,c,sum(v) FROM s GROUP BY a,b,c \
         UNION ALL SELECT a,b,NULL,sum(v) FROM s GROUP BY a,b \
         UNION ALL SELECT a,NULL,c,sum(v) FROM s GROUP BY a,c \
         UNION ALL SELECT NULL,b,c,sum(v) FROM s GROUP BY b,c \
         UNION ALL SELECT a,NULL,NULL,sum(v) FROM s GROUP BY a \
         UNION ALL SELECT NULL,b,NULL,sum(v) FROM s GROUP BY b \
         UNION ALL SELECT NULL,NULL,c,sum(v) FROM s GROUP BY c \
         UNION ALL SELECT NULL,NULL,NULL,sum(v) FROM s";
    let cube = "SELECT a, b, c, sum(v) FROM s GROUP BY CUBE(a, b, c)";
    assert_eq!(bag(&mut db, cube), bag(&mut db, by_hand), "the cube and the union disagree");

    let scanned = |db: &mut Session, q: &str| -> i64 {
        let r = rows(
            db,
            &format!(
                "SELECT rows_scanned FROM system.query_log WHERE query LIKE '{q}%' \
                 ORDER BY event_time DESC LIMIT 1"
            ),
        );
        match r.first().and_then(|r| r.first()) {
            Some(Value::UInt(n)) => *n as i64,
            Some(Value::Int(n)) => *n,
            other => panic!("no rows_scanned for `{q}`: {other:?}"),
        }
    };
    let one = scanned(&mut db, "SELECT a, b, c, sum(v) FROM s GROUP BY CUBE");
    let eight = scanned(&mut db, "SELECT a,b,c,sum(v) FROM s GROUP BY a,b,c UNION");
    assert_eq!(one, 2000, "the cube read the table more than once");
    assert!(
        eight >= 8 * one,
        "the hand-written union read {eight} rows, the cube {one}: the comparison is not \
         measuring what it claims to"
    );
}

/// PIN. A query with no grouping sets in it must plan exactly as it did
/// before they existed -- the keys in `group`, not hidden in an aggregate.
/// Everything about the fast paths in the hash aggregate keys off that list.
#[test]
fn a_plain_group_by_plans_exactly_as_it_did() {
    let mut db = db();
    let plan: Vec<String> =
        rows(&mut db, "EXPLAIN SELECT a, sum(v) FROM t GROUP BY a").iter().map(|r| format!("{}", r[0])).collect();
    let text = plan.join("\n");
    assert!(text.contains("Aggregate group=[a#0] aggs=[sum(v#1)]"), "{text}");
    assert!(!text.contains("__grouping"), "a plain GROUP BY grew a grouping-set marker:\n{text}");
}

// ----------------------------------------------------------------- spilling

/// The spilled path re-reads rows from a temp file, and a spilled row belongs
/// to exactly one of its several groups -- the one it missed under. Forced
/// through the binary, because the knob is read once per process.
#[test]
fn a_spilled_cube_answers_exactly_what_a_resident_one_does() {
    let vals: Vec<String> =
        (0..3000u32).map(|i| format!("({}, {}, {}, {i})", i % 13, i % 11, i % 7)).collect();
    let sql = format!(
        "CREATE TABLE s (a UInt8, b UInt8, c UInt8, v Int64) ENGINE = MergeTree ORDER BY a; \
         INSERT INTO s VALUES {}; \
         SELECT count(), sum(s), sum(g) FROM \
           (SELECT a, b, c, sum(v) AS s, GROUPING(a, b, c) AS g FROM s GROUP BY CUBE(a, b, c))",
        vals.join(", ")
    );
    let run = |rows: Option<&str>| -> String {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_granular"));
        cmd.arg("-q").arg(&sql);
        if let Some(n) = rows {
            cmd.env("GRANULAR_SPILL_ROWS", n);
        }
        let out = cmd.output().expect("run the binary");
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        assert!(
            out.status.success(),
            "{text}{}",
            String::from_utf8_lossy(&out.stderr)
        );
        // The answer row, whatever the box drawing around it.
        text.lines().filter(|l| l.starts_with('│')).nth(1).unwrap_or_default().to_string()
    };
    let resident = run(None);
    assert!(resident.contains('│'), "no answer row: {resident}");
    for n in ["512", "64", "8"] {
        assert_eq!(run(Some(n)), resident, "a cube spilling every {n} groups answered differently");
    }
}

// ------------------------------------------------------------------ refusals

#[test]
fn the_shapes_that_cannot_work_are_refused_by_name() {
    let mut db = db();
    for (sql, want) in [
        ("SELECT a FROM t GROUP BY ROLLUP()", "needs at least one column"),
        ("SELECT a FROM t GROUP BY CUBE()", "needs at least one column"),
        (
            "SELECT count() FROM t GROUP BY CUBE(a,b,v,id,a,b,v,id,a,b,v,id,a,b,v,id)",
            "grouping sets",
        ),
    ] {
        let e = err(&mut db, sql);
        assert!(e.contains(want), "`{sql}` said `{e}`, wanted `{want}`");
    }
    // A set is a 64-bit mask, so the 65th distinct key is refused rather than
    // shifted off the end of one -- which would silently drop the column from
    // every set that named it.
    let keys: Vec<String> = (0..70).map(|i| format!("v + {i}")).collect();
    let e = err(
        &mut db,
        &format!("SELECT count() FROM t GROUP BY GROUPING SETS (({}), ())", keys.join(", ")),
    );
    assert!(e.contains("more than 64 grouping columns"), "{e}");
}
