//! The hybrid claim, asserted.
//!
//! Being good at OLAP and good at OLTP separately is not the same as being
//! both on one table at once. The interesting failure is structural, not
//! arithmetic: a scan flushes the write buffer before reading, so an
//! interleaved read/write workload creates a part per query. Unchecked, that
//! grows without bound — every point lookup then probes one more bloom filter
//! and every scan reads one more set of undersized granules, and the engine
//! quietly stops being either thing.
//!
//! These tests pin the properties that make the combination hold:
//! writes are visible to queries immediately, the part count stays bounded
//! under interleaving, point lookups keep resolving through it, and
//! compaction does not lose or duplicate a row.

use granular::sql::ast::ObjectName;
use granular::types::Value;
use granular::{Result, Session};

fn parts(db: &Session, table: &str) -> usize {
    db.catalog
        .table(&ObjectName::bare(table))
        .expect("table exists")
        .part_count()
}

fn scalar(s: &mut Session, sql: &str) -> Value {
    s.query(sql)
        .unwrap_or_else(|e| panic!("query failed: {sql}\n  {e}"))
        .scalar()
        .unwrap_or_else(|| panic!("no scalar for {sql}"))
}

/// Writes and analytical queries alternating on one table.
#[test]
fn interleaved_writes_and_scans_stay_bounded() -> Result<()> {
    let mut db = Session::in_memory();
    db.execute("CREATE TABLE t (id UInt64, g UInt32, v Int64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id")?;

    let mut written = 0u64;
    let mut worst_parts = 0usize;

    for _round in 0..40 {
        for _ in 0..200 {
            db.execute(&format!(
                "INSERT INTO t VALUES ({written}, {}, {})",
                written % 8,
                written as i64
            ))?;
            written += 1;
        }

        // Every query must see every write that preceded it.
        let rs = db.query("SELECT g, count() FROM t GROUP BY g")?;
        let seen: u64 = rs.to_values().iter().map(|r| r[1].as_u64().unwrap()).sum();
        assert_eq!(seen, written, "an analytical query missed buffered writes");

        worst_parts = worst_parts.max(parts(&db, "t"));
    }

    // The property: a query triggers a flush, so without bounded compaction
    // this would be ~40 parts and climbing. Auto-compaction has to hold it.
    assert!(
        worst_parts <= 20,
        "part count reached {worst_parts} under interleaving; \
         scan-triggered flushes are not being compacted"
    );

    // ...and the OLTP half still works through whatever parts remain.
    for id in [0u64, 1, written / 2, written - 1] {
        assert_eq!(
            scalar(&mut db, &format!("SELECT v FROM t WHERE id = {id}")),
            Value::Int(id as i64),
            "point lookup failed for id {id}"
        );
    }
    assert_eq!(scalar(&mut db, "SELECT count() FROM t"), Value::UInt(written));
    Ok(())
}

/// Updates and deletes interleaved with queries: the same structural risk,
/// plus tombstones that compaction has to actually drop.
///
/// Checked against a `BTreeMap` rather than incremental arithmetic. The first
/// version of this test tracked the expected sum by adding and subtracting as
/// it went, and got it wrong: overwrite ranges overlap between rounds, so a
/// key rewritten twice was counted twice. A reference model cannot drift.
#[test]
fn interleaved_mutations_stay_consistent() -> Result<()> {
    use std::collections::BTreeMap;

    let mut db = Session::in_memory();
    db.execute("CREATE TABLE t (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id")?;

    let n = 4_000u64;
    let vals: Vec<String> = (0..n).map(|i| format!("({i},{i})")).collect();
    db.execute(&format!("INSERT INTO t VALUES {}", vals.join(",")))?;

    let mut model: BTreeMap<u64, i64> = (0..n).map(|i| (i, i as i64)).collect();

    for round in 0..25u64 {
        // overwrite a slice of keys (last write wins, no row growth)
        let base = round * 37 % (n - 100);
        for k in base..base + 50 {
            let v = 1_000_000 + k as i64;
            db.execute(&format!("INSERT INTO t VALUES ({k}, {v})"))?;
            model.insert(k, v);
        }
        // delete a few
        let dbase = 3_000 + round * 10;
        if dbase + 5 < n {
            db.execute(&format!(
                "ALTER TABLE t DELETE WHERE id >= {dbase} AND id < {}",
                dbase + 5
            ))?;
            for k in dbase..dbase + 5 {
                model.remove(&k);
            }
        }

        assert_eq!(
            scalar(&mut db, "SELECT count() FROM t"),
            Value::UInt(model.len() as u64),
            "row count drifted at round {round}"
        );
        assert_eq!(
            scalar(&mut db, "SELECT sum(v) FROM t"),
            Value::Int(model.values().sum::<i64>()),
            "sum drifted at round {round}"
        );
    }

    assert!(parts(&db, "t") <= 20, "parts unbounded under mutation");

    // A full compaction must change nothing observable.
    db.execute("OPTIMIZE TABLE t FINAL")?;
    assert_eq!(parts(&db, "t"), 1);
    assert_eq!(
        scalar(&mut db, "SELECT count() FROM t"),
        Value::UInt(model.len() as u64)
    );
    assert_eq!(
        scalar(&mut db, "SELECT sum(v) FROM t"),
        Value::Int(model.values().sum::<i64>())
    );
    // And every individual row still reads back correctly.
    for (&k, &v) in model.iter().step_by(97) {
        assert_eq!(
            scalar(&mut db, &format!("SELECT v FROM t WHERE id = {k}")),
            Value::Int(v),
            "key {k}"
        );
    }
    Ok(())
}

/// Partial compaction merges an arbitrary subset of parts. That is only sound
/// because a live key exists in exactly one part — this pins that invariant.
#[test]
fn partial_compaction_preserves_last_write_wins() -> Result<()> {
    let mut db = Session::in_memory();
    db.execute("CREATE TABLE t (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id")?;

    // Rewrite the same small key range many times, forcing many parts, each
    // holding a newer version of keys that older parts also once held.
    for round in 0..60i64 {
        for k in 0..20u64 {
            db.execute(&format!("INSERT INTO t VALUES ({k}, {})", round * 100 + k as i64))?;
        }
        // a query between rounds, so flushes are scan-triggered too
        let _ = db.query("SELECT count() FROM t")?;
    }

    assert_eq!(scalar(&mut db, "SELECT count() FROM t"), Value::UInt(20));
    for k in 0..20u64 {
        assert_eq!(
            scalar(&mut db, &format!("SELECT v FROM t WHERE id = {k}")),
            Value::Int(59 * 100 + k as i64),
            "key {k} did not keep its newest value across partial merges"
        );
    }
    Ok(())
}

/// Both access paths on one table, each still doing its job: a selective
/// range must prune granules, and a point lookup must not read the table.
#[test]
fn both_access_paths_work_on_one_table() -> Result<()> {
    let mut db = Session::in_memory();
    db.execute("CREATE TABLE t (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id")?;
    let vals: Vec<String> = (0..40_000u64).map(|i| format!("({i},{i})")).collect();
    db.execute(&format!("INSERT INTO t VALUES {}", vals.join(",")))?;

    // OLAP: full aggregate reads everything.
    let full = db.query("SELECT sum(v) FROM t")?;
    assert_eq!(full.scalar().unwrap(), Value::Int((0..40_000i64).sum()));
    assert!(full.stats.granules_read > 30, "a full scan should read the table");

    // OLAP: a selective range prunes nearly all of it.
    let sel = db.query("SELECT count() FROM t WHERE id >= 20000 AND id < 20100")?;
    assert_eq!(sel.scalar().unwrap(), Value::UInt(100));
    assert!(
        sel.stats.granules_pruned > sel.stats.granules_read * 10,
        "zone maps stopped pruning: {} pruned vs {} read",
        sel.stats.granules_pruned,
        sel.stats.granules_read
    );

    // OLTP: a point lookup touches almost nothing.
    let point = db.query("SELECT v FROM t WHERE id = 31337")?;
    assert_eq!(point.scalar().unwrap(), Value::Int(31337));
    assert!(
        point.stats.granules_read <= 2,
        "a point lookup read {} granules",
        point.stats.granules_read
    );

    // ...and a write lands immediately, visible to both.
    db.execute("INSERT INTO t VALUES (31337, -1)")?;
    assert_eq!(
        scalar(&mut db, "SELECT v FROM t WHERE id = 31337"),
        Value::Int(-1)
    );
    assert_eq!(scalar(&mut db, "SELECT count() FROM t"), Value::UInt(40_000));
    Ok(())
}
