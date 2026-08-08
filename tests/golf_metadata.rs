//! Answers that never touch a row have to be the answers a scan would give.
//!
//! `physical::meta_path` lets `count()`, `min(c)` and `max(c)` be folded out of
//! part metadata -- granule row counts, delete masks, zone maps -- instead of
//! read out of the table. That is worth 380x on `SELECT count() FROM events`,
//! and it is the one optimization in this engine whose failure mode is
//! invisible: a count that is fast and wrong looks exactly like a count that is
//! fast and right, and no benchmark and no `ORDER BY` will ever notice.
//!
//! So every assertion here is the same assertion. Run the query twice --
//! once so the planner takes the shortcut, once forced down the scan -- and
//! demand the *same value*. The forcing device is `min(1)`: an aggregate over
//! a literal, which `meta_path` refuses (its argument is not a column of the
//! scan) and which therefore drags the whole aggregate back onto the
//! `Aggregate`-over-`Scan` pipeline, predicate and all. Nothing else about the
//! query changes, so a disagreement can only be the shortcut.
//!
//! The state space is chosen for the ways "how many rows are there" stops
//! being obvious: an empty table, a table every row of which is deleted, rows
//! still sitting in the write buffer, an open transaction's private overlay,
//! tombstones inside a granule and at a part's ragged tail, a
//! `ReplacingMergeTree` collapsing a re-inserted key, and NULLs -- which are
//! rows for `count` but not values for `min`.

use granular::types::{Block, Column, DataType, Value};
use granular::Session;

// ------------------------------------------------------------------ harness

/// The plan text a query would run, as one string.
fn plan(s: &mut Session, sql: &str) -> String {
    s.query(&format!("EXPLAIN PIPELINE {sql}"))
        .unwrap_or_else(|e| panic!("EXPLAIN `{sql}`: {e}"))
        .to_values()
        .iter()
        .map(|r| r[0].render_plain())
        .collect::<Vec<_>>()
        .join("\n")
}

fn row(s: &mut Session, sql: &str) -> Vec<Value> {
    let rs = s.query(sql).unwrap_or_else(|e| panic!("`{sql}`: {e}"));
    let rows = rs.to_values();
    assert_eq!(rows.len(), 1, "a global aggregate is exactly one row: `{sql}`");
    rows.into_iter().next().expect("just checked")
}

/// Assert that `SELECT <list> <tail>` is answered from metadata, and that the
/// answer is the one the scan gives.
///
/// The reference is the same query with `min(1)` appended: `meta_path` walks
/// the aggregate list and refuses the whole path on the first entry it cannot
/// fold, so one unfoldable column puts the *other* columns back on the scan.
/// The extra output column is dropped before the comparison.
#[track_caller]
fn agrees(s: &mut Session, list: &str, tail: &str) -> Vec<Value> {
    let fast = format!("SELECT {list} {tail}");
    let slow = format!("SELECT {list}, min(1) {tail}");

    let fp = plan(s, &fast);
    assert!(fp.contains("MetaAggregate"), "`{fast}` did not take the shortcut:\n{fp}");
    let sp = plan(s, &slow);
    assert!(!sp.contains("MetaAggregate"), "the reference must not shortcut:\n{sp}");
    assert!(
        sp.contains("Scan ") || sp.contains("IndexLookup"),
        "the reference must actually read the table:\n{sp}"
    );

    let want = row(s, &slow);
    let got = row(s, &fast);
    let n = got.len();
    assert_eq!(got, want[..n], "metadata and scan disagree on `{fast}`");
    got
}

/// The value half of [`agrees`], with no claim about which path ran.
///
/// For the shapes where the planner is entitled to pick a *third* path: an
/// equality on the primary key lowers to an `IndexLookup`, which beats both a
/// scan and a metadata fold, so the only thing left to assert there is that
/// all of them say the same number.
#[track_caller]
fn same(s: &mut Session, list: &str, tail: &str) -> Vec<Value> {
    let want = row(s, &format!("SELECT {list}, min(1) {tail}"));
    let got = row(s, &format!("SELECT {list} {tail}"));
    let n = got.len();
    assert_eq!(got, want[..n], "the access paths disagree on `SELECT {list} {tail}`");
    got
}

/// Same, for a shape the planner is expected to *refuse*.
#[track_caller]
fn scans(s: &mut Session, sql: &str) {
    let p = plan(s, sql);
    assert!(!p.contains("MetaAggregate"), "`{sql}` must not be answered from metadata:\n{p}");
}

/// `t(id UInt64, v Int64, s String, n Nullable(Int64))`, `n` rows, flushed.
///
/// `id` is the key and is dense, `v = id * 3` so its extremes are distinct
/// from `id`'s, `s` cycles through eight strings so the string zone maps are
/// useless, and `n` is NULL on every third row.
fn table(rows: u64) -> Session {
    let mut s = Session::in_memory();
    s.execute(
        "CREATE TABLE t (id UInt64, v Int64, s String, n Nullable(Int64)) \
         ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
    )
    .unwrap();
    fill(&mut s, 0, rows);
    s.execute("SYSTEM FLUSH").unwrap();
    s
}

/// Append `[from, from + n)` to `t`, in batches small enough to parse.
fn fill(s: &mut Session, from: u64, n: u64) {
    let mut i = from;
    while i < from + n {
        let end = (i + 2_000).min(from + n);
        let mut sql = String::from("INSERT INTO t VALUES ");
        for k in i..end {
            if k > i {
                sql.push(',');
            }
            let nn = if k % 3 == 0 { "NULL".to_string() } else { (k as i64 - 500).to_string() };
            sql.push_str(&format!("({k},{},'s{}',{nn})", k as i64 * 3, k % 8));
        }
        s.execute(&sql).unwrap();
        i = end;
    }
}

const COUNT: &str = "count()";
const EXTREMES: &str = "min(id), max(id), min(v), max(v), min(s), max(s), min(n), max(n)";

// ------------------------------------------------------------- the shortcut

#[test]
fn an_unfiltered_count_is_answered_without_reading_a_row() {
    let mut s = table(5_000);
    assert_eq!(agrees(&mut s, COUNT, "FROM t"), vec![Value::UInt(5_000)]);
    // ... and it says so, so the win is provable from outside.
    let p = plan(&mut s, "SELECT count() FROM t");
    assert!(p.contains("from part metadata"), "{p}");
    // The counters are the other half of the claim: nothing was decoded.
    let rs = s.query("SELECT count() FROM t").unwrap();
    assert_eq!(rs.stats.granules_read, 0, "the count decoded a granule");
    assert_eq!(rs.stats.rows_scanned, 0, "the count decoded a row");
    assert!(rs.stats.granules_pruned >= 4, "{:?}", rs.stats);
}

#[test]
fn the_count_is_right_at_every_granule_boundary() {
    // 1024 rows to a granule and 8192 to a block: the interesting sizes are
    // the ones where the last granule is ragged, empty or exactly full.
    for n in [0u64, 1, 2, 1_023, 1_024, 1_025, 2_048, 8_191, 8_192, 8_193] {
        let mut s = table(n);
        assert_eq!(agrees(&mut s, COUNT, "FROM t"), vec![Value::UInt(n)], "{n} rows");
    }
}

#[test]
fn min_and_max_come_out_of_the_zone_maps() {
    let mut s = table(5_000);
    let got = agrees(&mut s, EXTREMES, "FROM t");
    assert_eq!(
        got,
        vec![
            Value::UInt(0),
            Value::UInt(4_999),
            Value::Int(0),
            Value::Int(14_997),
            Value::str("s0"),
            Value::str("s7"),
            // `n` is NULL on every third row and `id - 500` elsewhere, so the
            // extremes have to ignore the NULLs -- row 0 is NULL, and its
            // neighbour is the smallest live value.
            Value::Int(-499),
            Value::Int(4_499),
        ]
    );
}

#[test]
fn an_empty_table_counts_zero_and_has_no_extremes() {
    let mut s = table(0);
    assert_eq!(agrees(&mut s, COUNT, "FROM t"), vec![Value::UInt(0)]);
    // `min` of nothing is NULL, and it is a NULL in a column the schema
    // declared non-Nullable -- which is exactly the widening the aggregate
    // operator does, so the shortcut has to do it too.
    assert_eq!(agrees(&mut s, "min(id), max(v)", "FROM t"), vec![Value::Null, Value::Null]);
}

#[test]
fn a_table_whose_every_row_is_deleted_counts_zero() {
    let mut s = table(3_000);
    s.execute("ALTER TABLE t DELETE WHERE id >= 0").unwrap();
    assert_eq!(agrees(&mut s, COUNT, "FROM t"), vec![Value::UInt(0)]);
}

#[test]
fn tombstones_are_subtracted_wherever_they_sit() {
    // Inside a granule, across a granule seam, and in the ragged tail of the
    // part -- the last one is why the count is not `granule_deleted(gi)`
    // subtracted from `g.len`: that counter spans the whole 1024-bit window,
    // padding included.
    let mut s = table(2_500);
    for (pred, gone) in [
        ("id = 7", 1u64),
        ("id >= 1020 AND id < 1030", 10),
        ("id >= 2495", 5),
        ("id % 7 = 0", 2_500 / 7 + 1 - 1 /* id=7 already gone */ - 2 /* 1022, 2499 too */),
    ] {
        s.execute(&format!("ALTER TABLE t DELETE WHERE {pred}")).unwrap();
        let _ = gone;
        agrees(&mut s, COUNT, "FROM t");
    }
    // and the absolute answer, checked against the predicate by hand
    let live = (0..2_500u64)
        .filter(|k| !(*k == 7 || (1_020..1_030).contains(k) || *k >= 2_495 || k % 7 == 0))
        .count();
    assert_eq!(row(&mut s, "SELECT count() FROM t"), vec![Value::UInt(live as u64)]);
}

#[test]
fn min_and_max_refuse_the_shortcut_once_anything_is_deleted() {
    // A granule's `(min, max)` bounds the rows it was *built* from. Delete the
    // row that held the maximum and the bound does not move, so the fold would
    // answer with a value that is no longer in the table. The planner has to
    // decline -- and the answer has to stay right, which is the point.
    let mut s = table(3_000);
    assert_eq!(agrees(&mut s, "max(id)", "FROM t"), vec![Value::UInt(2_999)]);
    s.execute("ALTER TABLE t DELETE WHERE id >= 2900").unwrap();
    scans(&mut s, "SELECT max(id) FROM t");
    assert_eq!(row(&mut s, "SELECT max(id) FROM t"), vec![Value::UInt(2_899)]);
    // `count` has no such problem: the delete masks are exact, and counting
    // them is what makes it exact.
    assert_eq!(agrees(&mut s, COUNT, "FROM t"), vec![Value::UInt(2_900)]);
}

#[test]
fn rows_still_in_the_write_buffer_are_counted() {
    // Every read flushes before it plans, so this is the shape that would
    // silently lose rows if the shortcut read parts a statement too early.
    let mut s = table(1_000);
    s.execute("INSERT INTO t VALUES (99999, 1, 'z', 5)").unwrap();
    assert!(
        s.catalog.table_by_path("default.t").unwrap().has_pending_writes(),
        "the test needs a dirty delta to be worth running"
    );
    assert_eq!(agrees(&mut s, COUNT, "FROM t"), vec![Value::UInt(1_001)]);
    assert_eq!(agrees(&mut s, "max(id)", "FROM t"), vec![Value::UInt(99_999)]);
}

#[test]
fn a_transaction_counts_its_own_writes_and_forgets_them_on_rollback() {
    let mut s = table(2_000);
    s.execute("BEGIN").unwrap();
    s.execute("INSERT INTO t VALUES (900000, 1, 'z', 1), (900001, 2, 'z', 2)").unwrap();
    s.execute("ALTER TABLE t DELETE WHERE id < 10").unwrap();
    // Read-your-own-writes, through the overlay `Table::snapshot` hands back
    // inside a transaction -- the same one the scan would have taken.
    assert_eq!(agrees(&mut s, COUNT, "FROM t"), vec![Value::UInt(1_992)]);
    s.execute("ROLLBACK").unwrap();
    assert_eq!(agrees(&mut s, COUNT, "FROM t"), vec![Value::UInt(2_000)]);

    // ... and a committed one is visible afterwards.
    s.execute("BEGIN").unwrap();
    s.execute("INSERT INTO t VALUES (900002, 1, 'z', 1)").unwrap();
    s.execute("COMMIT").unwrap();
    assert_eq!(agrees(&mut s, COUNT, "FROM t"), vec![Value::UInt(2_001)]);
}

#[test]
fn a_replacing_engine_counts_the_surviving_row_once() {
    // Last write wins, and the loser is tombstoned at ingest rather than
    // merged away on read -- so the delete masks already say what the count
    // is. Re-insert every key and the table must not double.
    let mut s = Session::in_memory();
    s.execute(
        "CREATE TABLE r (id UInt64, v Int64) ENGINE = ReplacingMergeTree ORDER BY id PRIMARY KEY id",
    )
    .unwrap();
    let mut sql = String::from("INSERT INTO r VALUES ");
    for i in 0..3_000u64 {
        if i > 0 {
            sql.push(',');
        }
        sql.push_str(&format!("({i},{i})"));
    }
    s.execute(&sql).unwrap();
    s.execute("SYSTEM FLUSH").unwrap();
    assert_eq!(agrees(&mut s, COUNT, "FROM r"), vec![Value::UInt(3_000)]);

    // Half of them again, with new values.
    let mut again = String::from("INSERT INTO r VALUES ");
    for i in 0..1_500u64 {
        if i > 0 {
            again.push(',');
        }
        again.push_str(&format!("({i},{})", i + 77));
    }
    s.execute(&again).unwrap();
    s.execute("SYSTEM FLUSH").unwrap();
    assert_eq!(agrees(&mut s, COUNT, "FROM r"), vec![Value::UInt(3_000)]);
    assert_eq!(row(&mut s, "SELECT count() FROM r WHERE v >= 77 AND v < 1577").len(), 1);
}

#[test]
fn several_parts_are_all_folded() {
    let mut s = table(1_500);
    for base in 1..4u64 {
        fill(&mut s, base * 10_000, 1_500);
        s.execute("SYSTEM FLUSH").unwrap();
    }
    assert!(s.catalog.table_by_path("default.t").unwrap().part_count() >= 2);
    assert_eq!(agrees(&mut s, COUNT, "FROM t"), vec![Value::UInt(6_000)]);
    assert_eq!(agrees(&mut s, "min(id), max(id)", "FROM t"), vec![
        Value::UInt(0),
        Value::UInt(31_499)
    ]);
}

// ------------------------------------------------- counts under a predicate

#[test]
fn a_zone_decidable_predicate_counts_from_metadata_and_reads_only_the_straddlers() {
    let mut s = table(20_000);
    // A clustered key: every granule but one is either wholly in or wholly out.
    for cut in [0u64, 1, 1_023, 1_024, 1_025, 9_999, 19_999, 20_000, 40_000] {
        for op in [">=", ">", "<", "<=", "!="] {
            agrees(&mut s, COUNT, &format!("FROM t WHERE id {op} {cut}"));
        }
        // `id = k` is a point lookup, and the index beats the fold at it.
        same(&mut s, COUNT, &format!("FROM t WHERE id = {cut}"));
    }
    // ... and only the granule the boundary falls inside is decoded.
    let rs = s.query("SELECT count() FROM t WHERE id >= 9999").unwrap();
    assert_eq!(rs.stats.granules_read, 1, "{:?}", rs.stats);
    assert!(rs.stats.rows_scanned <= 1_024, "{:?}", rs.stats);
}

#[test]
fn a_predicate_off_the_sort_column_is_left_to_the_parallel_scan() {
    // `s` cycles through eight values inside every granule, so its zone map
    // spans the whole alphabet everywhere and *nothing* is decidable: the fold
    // would decode the table one granule at a time where the plan it replaced
    // spreads it across every core. Measured at 0.43x before this was gated;
    // see `meta_path`. The answer still has to be right, whoever computes it.
    let mut s = table(5_000);
    scans(&mut s, "SELECT count() FROM t WHERE s = 's3'");
    assert_eq!(same(&mut s, COUNT, "FROM t WHERE s = 's3'"), vec![Value::UInt(5_000 / 8)]);
    // ... and one conjunct off the sort column disqualifies the whole
    // predicate, however decidable the others are.
    scans(&mut s, "SELECT count() FROM t WHERE id >= 1000 AND s = 's3'");
}

#[test]
fn several_conjuncts_all_have_to_hold() {
    let mut s = table(20_000);
    for tail in [
        "FROM t WHERE id >= 5000 AND id < 15000",
        "FROM t WHERE id > 100 AND id > 200 AND id < 19000",
        // one conjunct excludes everything the other admits
        "FROM t WHERE id >= 15000 AND id < 5000",
        // a `BETWEEN` reaches the scan as two conjuncts
        "FROM t WHERE id BETWEEN 900 AND 1100",
        // the boundary lands inside one granule, which is the case the whole
        // covering test exists to isolate: 19 granules decided, 1 decoded
        "FROM t WHERE id >= 5100 AND id <= 5100",
    ] {
        agrees(&mut s, COUNT, tail);
    }
}

#[test]
fn a_null_in_a_predicate_column_is_never_counted_as_a_match() {
    // The covering test is "no row fails the negation", which under
    // three-valued logic is not the same as "every row matches": a NULL row
    // does neither. A granule whose `n` column holds a NULL therefore has to
    // be decoded however tight its bounds are, and if it were not, this
    // count would come out 50% high.
    let mut s = table(6_000);
    let live = (0..6_000u64).filter(|k| k % 3 != 0).count() as u64;
    assert_eq!(same(&mut s, COUNT, "FROM t WHERE n > -100000"), vec![Value::UInt(live)]);
    // The sort column is the one the fold is allowed to reason about, so make
    // it nullable and ask again: with a NULL in every granule, not one of them
    // may be called covered, and the count has to come out at two thirds
    // rather than at the table's row count.
    let mut u = Session::in_memory();
    u.execute(
        "CREATE TABLE u (k Nullable(Int64), v Int64) ENGINE = MergeTree ORDER BY k",
    )
    .unwrap();
    let mut sql = String::from("INSERT INTO u VALUES ");
    for i in 0..4_000i64 {
        if i > 0 {
            sql.push(',');
        }
        let k = if i % 3 == 0 { "NULL".to_string() } else { i.to_string() };
        sql.push_str(&format!("({k},{i})"));
    }
    u.execute(&sql).unwrap();
    u.execute("SYSTEM FLUSH").unwrap();
    let want = (0..4_000i64).filter(|i| i % 3 != 0 && *i > -100_000).count() as u64;
    assert_eq!(same(&mut u, COUNT, "FROM u WHERE k > -100000"), vec![Value::UInt(want)]);
}

#[test]
fn a_filtered_count_over_a_table_with_tombstones_agrees_with_the_scan() {
    let mut s = table(12_000);
    s.execute("ALTER TABLE t DELETE WHERE id % 11 = 0").unwrap();
    s.execute("ALTER TABLE t DELETE WHERE id >= 11990").unwrap();
    for tail in ["FROM t", "FROM t WHERE id >= 6000", "FROM t WHERE id < 3000", "FROM t WHERE id >= 11000"] {
        agrees(&mut s, COUNT, tail);
    }
    // ... and the shapes it declines still answer the same.
    same(&mut s, COUNT, "FROM t WHERE s = 's5'");
}

// -------------------------------------------------------- what is refused

#[test]
fn the_shapes_that_must_still_run_the_aggregate() {
    let mut s = table(5_000);
    for q in [
        // counts non-NULL values, which no header records
        "SELECT count(n) FROM t",
        "SELECT count(DISTINCT id) FROM t",
        // not a count or an extreme
        "SELECT sum(v) FROM t",
        "SELECT avg(v) FROM t",
        "SELECT uniq(id) FROM t",
        "SELECT quantile(0.9)(v) FROM t",
        // one unfoldable column refuses the whole path
        "SELECT count(), sum(v) FROM t",
        // a group needs per-group counts
        "SELECT s, count() FROM t GROUP BY s",
        // an expression, not a column of the scan
        "SELECT min(v * 2) FROM t",
        // an extreme under a predicate: the bounds describe the granule, not
        // the subset the predicate keeps
        "SELECT max(id) FROM t WHERE id < 100",
        // a predicate on a column the table is not sorted by: no granule
        // decides, so the fold would serialize what the scan parallelizes
        "SELECT count() FROM t WHERE v > 100",
        "SELECT count() FROM t WHERE n < 0",
        // a predicate no zone test can express, so the covering test would
        // have nothing equivalent to fall back on
        "SELECT count() FROM t WHERE id % 3 = 0",
        "SELECT count() FROM t WHERE id + v > 10",
        "SELECT count() FROM t WHERE s LIKE 's%'",
        "SELECT count() FROM t WHERE id IN (1, 2, 3)",
        "SELECT count() FROM t WHERE id > 5 OR v < 2",
        "SELECT count() FROM t WHERE n IS NULL",
        // an equality on the primary key: the index answers it in one probe
        // and one granule, where a metadata fold would still walk every zone
        // map looking for the granule that straddles.
        "SELECT count() FROM t WHERE id = 5",
        // not directly over a scan
        "SELECT count() FROM (SELECT id FROM t LIMIT 10)",
    ] {
        scans(&mut s, q);
    }
}

#[test]
fn a_refused_shape_is_still_the_same_answer_as_before() {
    // The negative cases have to keep working, not merely keep scanning.
    let mut s = table(3_000);
    assert_eq!(row(&mut s, "SELECT count(n) FROM t"), vec![Value::UInt(2_000)]);
    assert_eq!(
        row(&mut s, "SELECT count(), sum(v) FROM t"),
        vec![Value::UInt(3_000), Value::Int((0..3_000i64).map(|i| i * 3).sum())]
    );
    assert_eq!(row(&mut s, "SELECT max(id) FROM t WHERE id < 100"), vec![Value::UInt(99)]);
}

// ------------------------------------------------------- the other cut-offs

#[test]
fn limit_zero_never_builds_the_query_under_it() {
    let mut s = table(5_000);
    for q in [
        "SELECT id FROM t LIMIT 0",
        "SELECT sum(v) FROM t LIMIT 0",
        "SELECT id FROM t ORDER BY v DESC LIMIT 0",
        "SELECT s, count() FROM t GROUP BY s ORDER BY s LIMIT 0",
    ] {
        let p = plan(&mut s, q);
        assert_eq!(p.trim(), "Empty", "`{q}` still builds a pipeline:\n{p}");
        assert_eq!(s.query(q).unwrap().rows(), 0, "{q}");
        // The schema still has to describe the rows that are not there: a
        // client that declares its columns before sending them cannot wait.
        assert!(!s.query(q).unwrap().schema.fields().is_empty(), "{q}");
    }
    // `LIMIT 0 OFFSET n` is still nothing, and a non-zero limit is untouched.
    assert_eq!(plan(&mut s, "SELECT id FROM t LIMIT 0 OFFSET 10").trim(), "Empty");
    assert!(plan(&mut s, "SELECT id FROM t LIMIT 1").contains("Scan "));
}

#[test]
fn a_provably_empty_filter_needs_no_table_and_no_metadata() {
    // The optimizer already folds `WHERE false` to `Empty`; what matters here
    // is that the aggregate above it still answers, and answers 0 rather than
    // the table's row count.
    let mut s = table(5_000);
    let p = plan(&mut s, "SELECT count() FROM t WHERE false");
    assert!(p.contains("Empty"), "{p}");
    assert!(!p.contains("MetaAggregate"), "{p}");
    assert_eq!(row(&mut s, "SELECT count() FROM t WHERE false"), vec![Value::UInt(0)]);
    assert_eq!(row(&mut s, "SELECT count() FROM t WHERE 1 = 0"), vec![Value::UInt(0)]);
}

#[test]
fn a_table_with_no_parts_at_all_is_answered_from_the_absence_of_them() {
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE e (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id").unwrap();
    assert_eq!(agrees(&mut s, COUNT, "FROM e"), vec![Value::UInt(0)]);
    assert_eq!(agrees(&mut s, COUNT, "FROM e WHERE id > 5"), vec![Value::UInt(0)]);
    assert_eq!(agrees(&mut s, "min(id), max(v)", "FROM e"), vec![Value::Null, Value::Null]);
}

#[test]
fn the_fold_agrees_with_itself_once_it_fans_out() {
    // `meta_degree` wants 512 granules per worker, so the walk only goes
    // parallel past ~1M rows -- which is why every other test in this file
    // runs it serially and this one has to exist. Built as a `Block` rather
    // than as `INSERT ... VALUES`, because 1.1M rows of SQL text is 20 MB to
    // lex and this test is about the fold, not the parser.
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE big (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id").unwrap();
    let n = 1_100_000u64;
    {
        let t = s.catalog.table_by_path_mut("default.big").unwrap();
        t.insert(
            Block::new(vec![
                Column::u64s(DataType::UInt64, (0..n).collect()),
                Column::i64s(DataType::Int64, (0..n as i64).map(|i| i % 977).collect()),
            ])
            .unwrap(),
        )
        .unwrap();
        t.flush().unwrap();
    }
    let p = plan(&mut s, "SELECT count() FROM big WHERE id >= 500000");
    assert!(p.contains("workers"), "the fold stayed serial, so this proves nothing:\n{p}");

    for tail in [
        "FROM big",
        "FROM big WHERE id >= 500000",
        "FROM big WHERE id < 1000",
        "FROM big WHERE id BETWEEN 300000 AND 300500",
        "FROM big WHERE id != 7",
        "FROM big WHERE id > 2000000",
    ] {
        agrees(&mut s, COUNT, tail);
    }
    // ... and once every worker has tombstones to subtract as well.
    s.execute("ALTER TABLE big DELETE WHERE id % 1000 = 3").unwrap();
    agrees(&mut s, COUNT, "FROM big");
    agrees(&mut s, COUNT, "FROM big WHERE id >= 500000");
    assert_eq!(
        row(&mut s, "SELECT count() FROM big"),
        vec![Value::UInt(n - (0..n).filter(|k| k % 1_000 == 3).count() as u64)]
    );
}

// --------------------------------------------------------------- the fuzz

#[test]
fn metadata_and_scan_agree_over_random_tables_predicates_and_deletes() {
    // The suite above pins the cases someone thought of. This one is for the
    // ones nobody did: random row counts across every granule boundary,
    // random tombstones, random parts, and predicates on a clustered key, an
    // unclustered one and a string, at cut points drawn from the whole range
    // and from just outside it.
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    for round in 0..12u64 {
        let n = 1 + (rng() % 4_200);
        let mut s = table(n);
        // A second part, sometimes, overlapping the first.
        if round % 3 == 0 {
            fill(&mut s, n / 2, 700);
            s.execute("SYSTEM FLUSH").unwrap();
        }
        // Tombstones, sometimes.
        if round % 4 != 0 {
            let m = 2 + rng() % 9;
            s.execute(&format!("ALTER TABLE t DELETE WHERE id % {m} = 1")).unwrap();
        }
        // Unflushed rows, sometimes.
        if round % 5 == 0 {
            s.execute("INSERT INTO t VALUES (777777, 1, 's1', 3)").unwrap();
        }

        agrees(&mut s, COUNT, "FROM t");
        for _ in 0..8 {
            let cut = (rng() % (n + 64)) as i64 - 32;
            let op = ["<", "<=", "=", "!=", ">", ">="][(rng() % 6) as usize];
            // `id` is the primary key, so an equality on it lowers to the
            // index rather than to either of the paths under test.
            same(&mut s, COUNT, &format!("FROM t WHERE id {op} {cut}"));
            // `v` and `s` are off the sort column, so the planner declines
            // them -- but the answers still have to agree, and this is where a
            // gate that declined too much or too little would show up.
            same(&mut s, COUNT, &format!("FROM t WHERE v {op} {}", cut * 3));
            same(&mut s, COUNT, &format!("FROM t WHERE s {op} 's{}'", rng() % 9));
            agrees(
                &mut s,
                COUNT,
                &format!("FROM t WHERE id >= {cut} AND id < {}", cut + (rng() % 3_000) as i64),
            );
        }
    }
}

#[test]
fn extremes_and_scan_agree_over_random_tables() {
    // Separate, because `min`/`max` are only taken on a table with no
    // tombstone at all -- so the interesting axis is the shape of the data,
    // not the shape of the deletes.
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for round in 0..10u64 {
        let n = rng() % 3_300;
        let mut s = table(n);
        if round % 2 == 0 && n > 0 {
            fill(&mut s, 50_000 + rng() % 1_000, 1 + rng() % 900);
            s.execute("SYSTEM FLUSH").unwrap();
        }
        agrees(&mut s, EXTREMES, "FROM t");
        agrees(&mut s, COUNT, "FROM t");
    }
}
