//! Access paths, end to end: does the planner reach the machinery the engine
//! already has?
//!
//! Every test asserts the **answer** before it asserts the plan. A plan
//! assertion on its own only proves the planner changed its mind, and the
//! failure mode of every decision here is a *fast wrong answer* -- for a sorted
//! read, specifically a fast wrong order, which a test that checks the row set
//! would not see at all.
//!
//! So the reference is a real sort over the same rows. Two ways to force one,
//! used where each fits:
//!
//!   * a **woven twin**: the same keys stored in four parts that each span the
//!     whole range, so concatenating the parts is not sorted and the planner
//!     has to sort. Keys are unique in these tables, so the twin owes the same
//!     answer row for row.
//!   * `ORDER BY k + 0`: the same order -- adding zero to a `UInt64` reorders
//!     nothing -- but `BoundExpr::as_column()` answers `None` for a `Binary`,
//!     so no rule can match it. Only used *with* a `LIMIT`, because a full
//!     parallel sort on a lone expression key panics today; see the note on
//!     `the_orderings_the_stored_one_does_not_answer`.
//!
//! The engine's sort is stable, so over an input already in key order it is the
//! identity: the reference and the elided read must agree on the order of tied
//! keys too, not merely on the set of rows.

use granular::types::{Block, Column, DataType, Value};
use granular::Session;

// ------------------------------------------------------------------ helpers

fn empty(ddl: &str) -> Session {
    let mut s = Session::in_memory();
    s.execute(ddl).unwrap();
    s
}

/// One part holding exactly `ks`, in the order given.
fn part(s: &mut Session, ks: Vec<u64>) {
    let t = s.catalog.table_by_path_mut("default.a").unwrap();
    let vs: Vec<i64> = ks.iter().map(|&k| k as i64 % 1_000).collect();
    t.insert(
        Block::new(vec![Column::u64s(DataType::UInt64, ks), Column::i64s(DataType::Int64, vs)])
            .unwrap(),
    )
    .unwrap();
    t.flush().unwrap();
}

const DDL: &str =
    "CREATE TABLE a (k UInt64, v Int64) ENGINE = MergeTree ORDER BY k PRIMARY KEY k";

/// `0..n` in `parts` parts whose key ranges do not overlap.
fn table(n: u64, parts: u64) -> Session {
    let mut s = empty(DDL);
    let per = n / parts;
    for p in 0..parts {
        part(&mut s, (p * per..p * per + per).collect());
    }
    s
}

/// The same `0..n` keys, stored so that no concatenation of the parts is
/// sorted: four strided parts, each spanning the whole range.
fn woven(n: u64) -> Session {
    let mut s = empty(DDL);
    for p in 0..4u64 {
        part(&mut s, (0..n / 4).map(|i| i * 4 + p).collect());
    }
    s
}

fn plan(s: &mut Session, sql: &str) -> String {
    s.query(&format!("EXPLAIN PIPELINE {sql}"))
        .unwrap()
        .to_values()
        .iter()
        .map(|r| r[0].to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

fn rows(s: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    s.query(sql).unwrap().to_values()
}

/// Did the sort disappear? The claim is that the operator is *gone*, not that
/// it got cheaper, so both spellings of it have to be absent.
fn elided(plan: &str) -> bool {
    !plan.contains("Sort [") && !plan.contains("TopK")
}

/// `sql` must answer exactly what the woven twin answers, and must do it
/// without a sort while the twin does it with one.
fn agrees_with_twin(a: &mut Session, b: &mut Session, sql: &str) {
    let want = rows(b, sql);
    assert!(!elided(&plan(b, sql)), "the twin must really sort `{sql}`:\n{}", plan(b, sql));
    assert_eq!(rows(a, sql), want, "`{sql}` answered differently from the sorted reference");
    let p = plan(a, sql);
    assert!(elided(&p), "`{sql}` should read in order:\n{p}");
}

// --------------------------------------------------- 1. the sort that is free

#[test]
fn ordering_by_the_sort_key_reads_in_order_instead_of_sorting() {
    let (mut s, mut t) = (table(20_000, 1), woven(20_000));
    for q in [
        "SELECT k FROM a ORDER BY k LIMIT 5",
        "SELECT k, v FROM a ORDER BY k LIMIT 5",
        "SELECT k FROM a ORDER BY k",
        "SELECT k, v FROM a ORDER BY k",
        "SELECT k FROM a ORDER BY k LIMIT 5 OFFSET 9997",
        // A predicate the zone maps decide on the sort column keeps the read
        // ordered *and* keeps the pruning.
        "SELECT k FROM a WHERE k >= 19000 ORDER BY k LIMIT 5",
        "SELECT k FROM a WHERE k >= 19000 AND k < 19500 ORDER BY k",
        // `NULLS FIRST/LAST` cannot change anything here: `TableDef::sort_col`
        // refuses a nullable column, so the key this fires on has no NULL in
        // it and both spellings name the same order. Asserted rather than
        // assumed -- a rule that ignored the flag on a key that *could* be
        // NULL would pass every other test in this file.
        "SELECT k FROM a ORDER BY k NULLS FIRST LIMIT 5",
        "SELECT k FROM a ORDER BY k NULLS LAST LIMIT 5",
        "SELECT k FROM a ORDER BY k ASC NULLS LAST",
    ] {
        agrees_with_twin(&mut s, &mut t, q);
    }
    // Spelled out once, so a test that agreed with a broken reference twice
    // would still fail.
    assert_eq!(
        rows(&mut s, "SELECT k FROM a ORDER BY k LIMIT 5"),
        (0..5).map(|k| vec![Value::UInt(k)]).collect::<Vec<_>>()
    );
    // ... and the plan is a bare read under the limit: no sort, and no
    // exchange either, because the node the exchange wrapped is the one that
    // went away.
    let p = plan(&mut s, "SELECT k FROM a ORDER BY k LIMIT 5");
    assert!(p.contains("Scan default.a"), "{p}");
    assert!(!p.contains("Exchange"), "nothing left to fan out:\n{p}");
}

#[test]
fn several_parts_are_only_concatenated_when_their_ranges_do_not_overlap() {
    // Disjoint and ascending: reading them back to back *is* key order.
    let (mut disjoint, mut t) = (table(20_000, 4), woven(20_000));
    agrees_with_twin(&mut disjoint, &mut t, "SELECT k FROM a ORDER BY k LIMIT 5");
    agrees_with_twin(&mut disjoint, &mut t, "SELECT k FROM a ORDER BY k");

    // Interleaved -- the shape a real ingest produces whenever keys do not
    // arrive in order -- needs a merge, and a merge is what `Sort` is.
    let p = plan(&mut t, "SELECT k FROM a ORDER BY k LIMIT 5");
    assert!(!elided(&p), "overlapping parts still need a merge:\n{p}");

    // Parts written in descending order are each sorted and exactly backwards
    // as a set, which the range test has to catch as surely as interleaving.
    let mut backwards = empty(DDL);
    for p in (0..4u64).rev() {
        part(&mut backwards, (p * 5_000..p * 5_000 + 5_000).collect());
    }
    let p = plan(&mut backwards, "SELECT k FROM a ORDER BY k LIMIT 5");
    assert!(!elided(&p), "a descending part set is not a sorted read:\n{p}");
    assert_eq!(
        rows(&mut backwards, "SELECT k FROM a ORDER BY k LIMIT 5"),
        (0..5).map(|k| vec![Value::UInt(k)]).collect::<Vec<_>>()
    );

    // Touching ranges are still in order: part 0 ends at 4999 and part 1
    // starts at 5000, so `<=` is the test and `<` would refuse a set that is
    // perfectly ordered.
    let mut touching = empty(DDL);
    part(&mut touching, (0..5_000).collect());
    part(&mut touching, (5_000..10_000).collect());
    assert!(elided(&plan(&mut touching, "SELECT k FROM a ORDER BY k LIMIT 5")));
}

#[test]
fn pending_deletions_do_not_move_a_row_out_of_order() {
    let (mut s, mut t) = (table(20_000, 1), woven(20_000));
    for k in [0u64, 1, 2, 7, 19_999] {
        for db in [&mut s, &mut t] {
            db.catalog
                .table_by_path_mut("default.a")
                .unwrap()
                .delete_key(&Value::UInt(k))
                .unwrap();
        }
    }
    // A tombstone removes rows from a sorted run; what is left is still one.
    agrees_with_twin(&mut s, &mut t, "SELECT k FROM a ORDER BY k LIMIT 5");
    agrees_with_twin(&mut s, &mut t, "SELECT k FROM a ORDER BY k");
    assert_eq!(
        rows(&mut s, "SELECT k FROM a ORDER BY k LIMIT 5"),
        [3u64, 4, 5, 6, 8].map(|k| vec![Value::UInt(k)]).to_vec()
    );
}

#[test]
fn an_open_transaction_reads_its_own_writes_in_the_right_order() {
    // Even keys only, so an odd key is a genuine insert rather than the
    // last-write-wins replacement a repeated primary key would be.
    let mut s = empty(DDL);
    part(&mut s, (0..10_000).map(|i| i * 2).collect());

    s.execute("BEGIN").unwrap();
    // Lands inside the committed range, so the overlay part overlaps it and
    // the concatenation is no longer sorted. The rule has to see that.
    s.execute("INSERT INTO a VALUES (9999, 7)").unwrap();
    let p = plan(&mut s, "SELECT k FROM a ORDER BY k LIMIT 5");
    assert!(!elided(&p), "an overlapping overlay part needs the sort:\n{p}");
    assert_eq!(
        rows(&mut s, "SELECT k FROM a WHERE k >= 9996 ORDER BY k LIMIT 3"),
        [9_996u64, 9_998, 9_999].map(|k| vec![Value::UInt(k)]).to_vec()
    );
    s.execute("ROLLBACK").unwrap();

    // An overlay above everything committed keeps the set ordered, so the read
    // stays ordered -- and still sees the uncommitted row.
    s.execute("BEGIN").unwrap();
    s.execute("INSERT INTO a VALUES (30000, 1)").unwrap();
    let p = plan(&mut s, "SELECT k FROM a ORDER BY k LIMIT 5");
    assert!(elided(&p), "a disjoint overlay part is still a sorted read:\n{p}");
    assert_eq!(
        rows(&mut s, "SELECT k FROM a ORDER BY k LIMIT 3"),
        [0u64, 2, 4].map(|k| vec![Value::UInt(k)]).to_vec()
    );
    assert_eq!(
        rows(&mut s, "SELECT k FROM a WHERE k >= 19998 ORDER BY k"),
        [19_998u64, 30_000].map(|k| vec![Value::UInt(k)]).to_vec()
    );
    s.execute("ROLLBACK").unwrap();
    // ... and after the rollback the uncommitted row is gone from the ordered
    // read too, rather than surviving in a stale snapshot.
    assert_eq!(rows(&mut s, "SELECT k FROM a WHERE k >= 19998 ORDER BY k"), [[Value::UInt(19_998)]]);
}

#[test]
fn tied_keys_come_back_in_the_order_a_stable_sort_would_have_produced() {
    // No PRIMARY KEY, so duplicates are legal and every key appears twice.
    // This is where "a fast wrong order" hides: the key column looks right
    // whatever happens to the payload, so the payload is what has to be
    // pinned. The expected order is the stored one, which is also what a
    // *stable* sort of the stored order produces -- the two must not diverge.
    let mut s = empty("CREATE TABLE a (k UInt64, v Int64) ENGINE = MergeTree ORDER BY k");
    let t = s.catalog.table_by_path_mut("default.a").unwrap();
    t.insert(
        Block::new(vec![
            Column::u64s(DataType::UInt64, (0..4_000).map(|i| i / 2).collect()),
            Column::i64s(DataType::Int64, (0..4_000).collect()),
        ])
        .unwrap(),
    )
    .unwrap();
    t.flush().unwrap();

    let got = rows(&mut s, "SELECT k, v FROM a ORDER BY k");
    assert!(elided(&plan(&mut s, "SELECT k, v FROM a ORDER BY k")));
    let want: Vec<Vec<Value>> =
        (0..4_000i64).map(|i| vec![Value::UInt(i as u64 / 2), Value::Int(i)]).collect();
    assert_eq!(got, want, "ties must keep the stored order");
    // The reference agrees, through the top-K path which no rule elides.
    assert_eq!(rows(&mut s, "SELECT k, v FROM a ORDER BY k + 0 LIMIT 6"), want[..6].to_vec());
}

// ----------------------------------------------- 2. and the shapes it refuses

#[test]
fn the_orderings_the_stored_one_does_not_answer() {
    let mut s = table(20_000, 1);
    for (q, why) in [
        // Storage reads one way. A DESC read needs the scan to walk granules
        // and rows backwards, which is a storage change, not a planner one.
        ("SELECT k FROM a ORDER BY k DESC LIMIT 5", "descending"),
        // Not the sort column.
        ("SELECT k FROM a ORDER BY v LIMIT 5", "another column"),
        // The sort column, but not first: rows tied on `v` are in `k` order,
        // which says nothing about the order of `v`.
        ("SELECT k FROM a ORDER BY v, k LIMIT 5", "not a leading key"),
        // A *prefix* of the stored order is refused too, and deliberately:
        // `Table::merge_parts` merges runs on the leading sort lane alone and
        // breaks ties by part index, so a merged part is sorted by
        // `order_by[0]` and by nothing after it.
        ("SELECT k, v FROM a ORDER BY k, v LIMIT 5", "second key not guaranteed"),
        ("SELECT k, v FROM a ORDER BY k ASC, v DESC LIMIT 5", "mixed directions"),
        // An expression, not a column: `k + 0` is the same order, but proving
        // that is a theorem the planner does not have.
        ("SELECT k FROM a ORDER BY k + 0 LIMIT 5", "an expression"),
        // A predicate no zone map decides prunes nothing, so the read walks
        // the whole table either way -- and dropping the sort drops the
        // exchange that was doing that walk across every core. Measured at
        // 0.29x on 1M rows; see `read_in_order`.
        ("SELECT k FROM a WHERE v = 42 ORDER BY k LIMIT 5", "an undecidable predicate"),
        ("SELECT k FROM a WHERE v < 900 ORDER BY k", "an undecidable predicate"),
        ("SELECT k FROM a WHERE k % 7 = 0 ORDER BY k LIMIT 5", "an inexpressible predicate"),
        // Every group is one output row, in whatever order the hash table
        // hands them over.
        ("SELECT k, count() FROM a GROUP BY k ORDER BY k LIMIT 5", "an aggregate between"),
        ("SELECT DISTINCT k FROM a ORDER BY k LIMIT 5", "a distinct between"),
    ] {
        let p = plan(&mut s, q);
        assert!(!elided(&p), "{q} must still sort ({why}):\n{p}");
    }
    // ... and each still answers correctly.
    assert_eq!(
        rows(&mut s, "SELECT k FROM a ORDER BY k DESC LIMIT 3"),
        [19_999u64, 19_998, 19_997].map(|k| vec![Value::UInt(k)]).to_vec()
    );
    assert_eq!(
        rows(&mut s, "SELECT k FROM a WHERE v = 42 ORDER BY k LIMIT 3"),
        [42u64, 1_042, 2_042].map(|k| vec![Value::UInt(k)]).to_vec()
    );
    assert_eq!(
        rows(&mut s, "SELECT k, v FROM a ORDER BY k, v LIMIT 2"),
        [vec![Value::UInt(0), Value::Int(0)], vec![Value::UInt(1), Value::Int(1)]]
    );
}

#[test]
fn a_table_that_is_not_stored_in_order_never_reads_in_order() {
    for ddl in [
        // No order at all.
        "CREATE TABLE a (k UInt64, v Int64) ENGINE = MergeTree ORDER BY tuple()",
        // `Memory` keeps insertion order whatever the declaration says.
        "CREATE TABLE a (k UInt64, v Int64) ENGINE = Memory",
        // A nullable key: `sort_col` refuses it, because where NULL sorts is a
        // decision the storage layer never made.
        "CREATE TABLE a (k Nullable(UInt64), v Int64) ENGINE = MergeTree ORDER BY k",
        // A string key: its lanes are per-granule dictionary codes, so they
        // order within a granule and mean nothing across granules.
        "CREATE TABLE a (k String, v Int64) ENGINE = MergeTree ORDER BY k",
    ] {
        let mut s = empty(ddl);
        s.execute("INSERT INTO a VALUES (3, 1), (1, 2), (2, 3)").unwrap();
        let p = plan(&mut s, "SELECT k FROM a ORDER BY k LIMIT 2");
        assert!(!elided(&p), "{ddl}:\n{p}");
        assert_eq!(rows(&mut s, "SELECT k FROM a ORDER BY k LIMIT 2").len(), 2, "{ddl}");
    }
}

// ------------------------------------------------------- 3. the index's reach

#[test]
fn the_index_answers_a_point_lookup_and_nothing_it_should_not() {
    let mut s = table(20_000, 1);
    assert!(plan(&mut s, "SELECT v FROM a WHERE k = 7").contains("IndexLookup"));
    assert!(plan(&mut s, "SELECT v FROM a WHERE k IN (7, 9)").contains("IndexLookup"));
    for q in [
        "SELECT v FROM a WHERE k > 7",
        "SELECT v FROM a WHERE k >= 7 AND k <= 9",
        "SELECT v FROM a WHERE k != 7",
        "SELECT v FROM a WHERE v = 21",
        "SELECT v FROM a WHERE k NOT IN (1, 2)",
        "SELECT v FROM a",
    ] {
        let p = plan(&mut s, q);
        assert!(!p.contains("IndexLookup"), "{q} must scan:\n{p}");
    }
    assert_eq!(rows(&mut s, "SELECT v FROM a WHERE k = 7"), [[Value::Int(7)]]);
    assert_eq!(rows(&mut s, "SELECT v FROM a WHERE k > 19997").len(), 2);

    // A composite key and a string key have no MPH index to reach: storage
    // builds one only over a single non-string, non-nullable lane, so
    // `TableDef::pk_col` is `None` and there is nothing for the planner to
    // choose. Pinned here so that the day storage grows one, this is the test
    // that says the planner still has to be taught.
    let mut c = empty(
        "CREATE TABLE a (x UInt64, y UInt64, v Int64) ENGINE = MergeTree \
         ORDER BY (x, y) PRIMARY KEY (x, y)",
    );
    c.execute("INSERT INTO a VALUES (1, 2, 30), (1, 3, 40), (2, 1, 50)").unwrap();
    let p = plan(&mut c, "SELECT v FROM a WHERE x = 1 AND y = 2");
    assert!(!p.contains("IndexLookup"), "a composite key has no index to probe:\n{p}");
    // The leading key still prunes granules, which is what carries this shape.
    assert!(p.contains("zonemap="), "{p}");
    assert_eq!(rows(&mut c, "SELECT v FROM a WHERE x = 1 AND y = 2"), [[Value::Int(30)]]);

    let mut t = empty(
        "CREATE TABLE a (s String, v Int64) ENGINE = MergeTree ORDER BY s PRIMARY KEY s",
    );
    t.execute("INSERT INTO a VALUES ('a', 1), ('b', 2)").unwrap();
    let p = plan(&mut t, "SELECT v FROM a WHERE s = 'b'");
    assert!(!p.contains("IndexLookup"), "a string key has no index to probe:\n{p}");
    assert_eq!(rows(&mut t, "SELECT v FROM a WHERE s = 'b'"), [[Value::Int(2)]]);
}

// ------------------------------------------- 4. what the headers can answer

#[test]
fn the_metadata_fold_reaches_through_a_projection() {
    let mut s = table(20_000, 1);
    assert!(plan(&mut s, "SELECT count() FROM a").contains("MetaAggregate"));
    assert_eq!(rows(&mut s, "SELECT count() FROM a"), [[Value::UInt(20_000)]]);

    // A derived table is inlined, but its column list stays as a `Project`
    // between the aggregate and the scan. That projection cannot change the
    // row count, so the headers still answer it.
    for (q, want) in [
        ("SELECT count() FROM (SELECT k FROM a) u", Value::UInt(20_000)),
        ("SELECT count() FROM (SELECT v, k FROM a) u", Value::UInt(20_000)),
        ("SELECT max(k) FROM (SELECT v, k FROM a) u", Value::UInt(19_999)),
        ("SELECT min(v) FROM (SELECT v, k FROM a) u", Value::Int(0)),
        ("SELECT count() FROM (SELECT k FROM (SELECT k, v FROM a) x) u", Value::UInt(20_000)),
    ] {
        let p = plan(&mut s, q);
        assert!(p.contains("MetaAggregate"), "`{q}` should fold:\n{p}");
        assert_eq!(rows(&mut s, q), [[want]], "{q}");
    }

    // A trivially-true predicate is not a special case: the optimizer folds
    // `1 = 1` away before this is reached, so the fold fires for the ordinary
    // reason and the plan carries no `where=` at all.
    let t = plan(&mut s, "SELECT count() FROM a WHERE 1 = 1");
    assert!(t.contains("MetaAggregate") && !t.contains("where="), "{t}");
    assert_eq!(rows(&mut s, "SELECT count() FROM a WHERE 1 = 1"), [[Value::UInt(20_000)]]);

    for q in [
        // A join can drop rows and duplicate them, and nothing in the catalog
        // says it will not: there are no foreign keys, so `count()` over a
        // join is `|a JOIN b|`, which no header records.
        "SELECT count() FROM a JOIN a AS b ON a.k = b.k",
        // A `LIMIT` below makes the count the limit's, not the table's.
        "SELECT count() FROM (SELECT k FROM a LIMIT 10) u",
        // A `WHERE` inside the derived table lands as the scan's PREWHERE and
        // is folded normally -- but one the zone maps cannot decide is
        // refused, exactly as it is without the subquery.
        "SELECT count() FROM (SELECT k FROM a WHERE v = 42) u",
        // A computed projection could raise on a row the fold never reads.
        "SELECT count() FROM (SELECT k * 2 AS k FROM a) u",
        "SELECT count(v) FROM a",
        "SELECT sum(v) FROM a",
    ] {
        let p = plan(&mut s, q);
        assert!(!p.contains("MetaAggregate"), "`{q}` must read rows:\n{p}");
    }
    assert_eq!(rows(&mut s, "SELECT count() FROM (SELECT k FROM a LIMIT 10) u"), [[Value::UInt(10)]]);
    assert_eq!(rows(&mut s, "SELECT count() FROM (SELECT k * 2 AS k FROM a) u"), [[Value::UInt(20_000)]]);
    assert_eq!(rows(&mut s, "SELECT count() FROM a JOIN a AS b ON a.k = b.k"), [[Value::UInt(20_000)]]);
}
