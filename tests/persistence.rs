//! End-to-end durability tests: write with one `Session`, drop it, reopen, and
//! assert the data (and the indexes) survived.
//!
//! The interesting claim being tested is not just "bytes round-trip" but that
//! the *indexes* round-trip: the minimal perfect hash and learned-rank records
//! are persisted rather than rebuilt, so reopening a large table is an I/O
//! cost rather than a re-indexing cost.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use granular::types::Value;
use granular::{Result, Session};

/// A unique scratch directory per test, removed on drop. Derived from the pid
/// and a counter rather than randomness so failures are reproducible.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "granular-it-{}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed),
            tag
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Scratch(d)
    }
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scalar(s: &mut Session, sql: &str) -> Value {
    s.query(sql)
        .unwrap_or_else(|e| panic!("query failed: {sql}\n  {e}"))
        .scalar()
        .unwrap_or_else(|| panic!("no scalar for {sql}"))
}

#[test]
fn data_survives_a_reopen() -> Result<()> {
    let dir = Scratch::new("reopen");

    {
        let mut db = Session::open(dir.path())?;
        db.execute(
            "CREATE TABLE t (id UInt64, name String, v Int64)
             ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
        )?;
        let values: Vec<String> = (0..5_000u64)
            .map(|i| format!("({i},'name-{}',{})", i % 50, (i as i64 % 200) - 100))
            .collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")))?;
        db.checkpoint()?;
    }

    let mut db = Session::open(dir.path())?;
    assert_eq!(scalar(&mut db, "SELECT count() FROM t"), Value::UInt(5_000));

    let expect_sum: i64 = (0..5_000i64).map(|i| (i % 200) - 100).sum();
    assert_eq!(scalar(&mut db, "SELECT sum(v) FROM t"), Value::Int(expect_sum));

    // point lookups must still resolve through the persisted index
    assert_eq!(
        scalar(&mut db, "SELECT name FROM t WHERE id = 1234"),
        Value::str("name-34")
    );
    // string dictionary survived
    assert_eq!(
        scalar(&mut db, "SELECT uniqExact(name) FROM t"),
        Value::UInt(50)
    );
    Ok(())
}

#[test]
fn schema_and_engine_survive_a_reopen() -> Result<()> {
    let dir = Scratch::new("schema");
    {
        let mut db = Session::open(dir.path())?;
        db.execute(
            "CREATE TABLE t (
                id  UInt64,
                d   Date,
                f   Float64,
                opt Nullable(Int32)
             ) ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
        )?;
        db.execute("INSERT INTO t VALUES (1, '2024-03-01', 1.5, NULL), (2, '2024-03-02', -2.5, 7)")?;
        db.checkpoint()?;
    }

    let mut db = Session::open(dir.path())?;
    let d = db.query("DESCRIBE t")?.to_values();
    assert_eq!(d.len(), 4);
    assert_eq!(d[1][1], Value::str("Date"));
    assert_eq!(d[3][1], Value::str("Nullable(Int32)"));

    assert_eq!(scalar(&mut db, "SELECT toYear(d) FROM t WHERE id = 1"), Value::UInt(2024));
    assert_eq!(scalar(&mut db, "SELECT f FROM t WHERE id = 2"), Value::Float(-2.5));
    assert_eq!(scalar(&mut db, "SELECT opt FROM t WHERE id = 1"), Value::Null);
    assert_eq!(scalar(&mut db, "SELECT opt FROM t WHERE id = 2"), Value::Int(7));
    assert_eq!(scalar(&mut db, "SELECT count(opt) FROM t"), Value::UInt(1));
    Ok(())
}

#[test]
fn deletes_and_updates_survive_a_reopen() -> Result<()> {
    let dir = Scratch::new("mutations");
    {
        let mut db = Session::open(dir.path())?;
        db.execute("CREATE TABLE t (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id")?;
        let values: Vec<String> = (0..2_000u64).map(|i| format!("({i},{i})")).collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")))?;
        db.execute("ALTER TABLE t DELETE WHERE id < 500")?;
        db.execute("ALTER TABLE t UPDATE v = 999 WHERE id = 1000")?;
        db.checkpoint()?;
    }

    let mut db = Session::open(dir.path())?;
    assert_eq!(scalar(&mut db, "SELECT count() FROM t"), Value::UInt(1_500));
    assert_eq!(scalar(&mut db, "SELECT v FROM t WHERE id = 1000"), Value::Int(999));
    assert!(db.query("SELECT v FROM t WHERE id = 100")?.is_empty());
    Ok(())
}

#[test]
fn multiple_databases_and_tables_survive() -> Result<()> {
    let dir = Scratch::new("multi");
    {
        let mut db = Session::open(dir.path())?;
        db.execute("CREATE DATABASE analytics")?;
        db.execute("CREATE TABLE a (id UInt64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id")?;
        db.execute("CREATE TABLE analytics.b (id UInt64, s String) ENGINE = MergeTree ORDER BY id PRIMARY KEY id")?;
        db.execute("INSERT INTO a VALUES (1), (2), (3)")?;
        db.execute("INSERT INTO analytics.b VALUES (10, 'x'), (20, 'y')")?;
        db.checkpoint()?;
    }

    let mut db = Session::open(dir.path())?;
    assert_eq!(scalar(&mut db, "SELECT count() FROM a"), Value::UInt(3));
    assert_eq!(scalar(&mut db, "SELECT count() FROM analytics.b"), Value::UInt(2));
    assert_eq!(
        scalar(&mut db, "SELECT s FROM analytics.b WHERE id = 20"),
        Value::str("y")
    );

    let dbs: Vec<Value> = db
        .query("SHOW DATABASES")?
        .to_values()
        .into_iter()
        .map(|r| r[0].clone())
        .collect();
    assert!(dbs.contains(&Value::str("analytics")));
    assert!(dbs.contains(&Value::str("default")));
    Ok(())
}

#[test]
fn a_large_table_reopens_and_still_prunes() -> Result<()> {
    let dir = Scratch::new("large");
    let n = 30_000u64;
    {
        let mut db = Session::open(dir.path())?;
        db.execute("CREATE TABLE big (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id")?;
        let values: Vec<String> = (0..n).map(|i| format!("({i},{i})")).collect();
        db.execute(&format!("INSERT INTO big VALUES {}", values.join(",")))?;
        db.execute("OPTIMIZE TABLE big FINAL")?;
        db.checkpoint()?;
    }

    let mut db = Session::open(dir.path())?;
    assert_eq!(scalar(&mut db, "SELECT count() FROM big"), Value::UInt(n));

    // Zone maps must survive the round-trip, not just the data.
    let rs = db.query("SELECT count() FROM big WHERE id >= 10000 AND id < 10100")?;
    assert_eq!(rs.scalar().unwrap(), Value::UInt(100));
    assert!(
        rs.stats.granules_pruned > 20,
        "pruning did not survive reopen: {} pruned / {} read",
        rs.stats.granules_pruned,
        rs.stats.granules_read
    );

    // Every key must still resolve through the persisted index.
    for id in [0u64, 1, 12_345, n - 1] {
        assert_eq!(
            scalar(&mut db, &format!("SELECT v FROM big WHERE id = {id}")),
            Value::Int(id as i64),
            "id {id}"
        );
    }
    Ok(())
}

#[test]
fn writes_after_reopen_append_correctly() -> Result<()> {
    let dir = Scratch::new("append");
    {
        let mut db = Session::open(dir.path())?;
        db.execute("CREATE TABLE t (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id")?;
        db.execute("INSERT INTO t VALUES (1, 10), (2, 20)")?;
        db.checkpoint()?;
    }
    {
        let mut db = Session::open(dir.path())?;
        db.execute("INSERT INTO t VALUES (3, 30)")?;
        // overwriting an existing key must still be last-write-wins across the
        // persistence boundary
        db.execute("INSERT INTO t VALUES (1, 111)")?;
        db.checkpoint()?;
    }

    let mut db = Session::open(dir.path())?;
    assert_eq!(scalar(&mut db, "SELECT count() FROM t"), Value::UInt(3));
    assert_eq!(scalar(&mut db, "SELECT v FROM t WHERE id = 1"), Value::Int(111));
    assert_eq!(scalar(&mut db, "SELECT v FROM t WHERE id = 3"), Value::Int(30));
    assert_eq!(scalar(&mut db, "SELECT sum(v) FROM t"), Value::Int(161));
    Ok(())
}

/// Dropping a `Session` without calling `checkpoint()` is what a crash looks
/// like from the filesystem's point of view: parts are whatever the last
/// checkpoint wrote, and everything since then exists only in the log. If the
/// log is not actually fed by the write path, this test loses data.
#[test]
fn acknowledged_writes_survive_a_crash_without_checkpoint() -> Result<()> {
    let dir = Scratch::new("crash");
    {
        let mut db = Session::open(dir.path())?;
        db.execute("CREATE TABLE t (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id")?;
        db.execute("INSERT INTO t VALUES (1, 10), (2, 20)")?;
        db.checkpoint()?; // this much is in parts
        db.execute("INSERT INTO t VALUES (3, 30)")?;
        db.execute("INSERT INTO t VALUES (1, 111)")?;
        db.execute("ALTER TABLE t DELETE WHERE id = 2")?;
        // ...and then the process dies. No checkpoint.
    }

    let mut db = Session::open(dir.path())?;
    assert_eq!(
        scalar(&mut db, "SELECT count() FROM t"),
        Value::UInt(2),
        "post-checkpoint writes were lost"
    );
    assert_eq!(scalar(&mut db, "SELECT v FROM t WHERE id = 1"), Value::Int(111));
    assert_eq!(scalar(&mut db, "SELECT v FROM t WHERE id = 3"), Value::Int(30));
    assert!(db.query("SELECT v FROM t WHERE id = 2")?.is_empty(), "delete was lost");
    Ok(())
}

/// Recovery has to be idempotent: replaying the same log twice must not
/// double-apply anything.
#[test]
fn recovery_is_idempotent_across_repeated_crashes() -> Result<()> {
    let dir = Scratch::new("crash-twice");
    {
        let mut db = Session::open(dir.path())?;
        db.execute("CREATE TABLE t (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id")?;
        db.execute("INSERT INTO t VALUES (1, 10)")?;
    }
    for _ in 0..3 {
        let mut db = Session::open(dir.path())?;
        assert_eq!(scalar(&mut db, "SELECT count() FROM t"), Value::UInt(1));
        assert_eq!(scalar(&mut db, "SELECT v FROM t WHERE id = 1"), Value::Int(10));
        // crash again without checkpointing
    }
    // and one clean shutdown, which folds the log into parts
    {
        let mut db = Session::open(dir.path())?;
        db.checkpoint()?;
    }
    let mut db = Session::open(dir.path())?;
    assert_eq!(scalar(&mut db, "SELECT count() FROM t"), Value::UInt(1));
    Ok(())
}

#[test]
fn an_empty_database_reopens_cleanly() -> Result<()> {
    let dir = Scratch::new("empty");
    {
        let mut db = Session::open(dir.path())?;
        db.checkpoint()?;
    }
    let mut db = Session::open(dir.path())?;
    assert!(db.query("SHOW TABLES")?.is_empty());
    db.execute("CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id")?;
    assert_eq!(scalar(&mut db, "SELECT count() FROM t"), Value::UInt(0));
    Ok(())
}
