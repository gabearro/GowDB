//! Are `UPDATE`, `DELETE` and the result renderer actually reachable, and do
//! they mean what they say?
//!
//! Every test here drives the public [`Session`] API against a **real data
//! directory**, because the defects it covers were all invisible in memory:
//!
//!   * `apply_sweep` refused any mutation on a table with no single-column
//!     PRIMARY KEY *when write-ahead logging was on*, which is every persistent
//!     session and therefore the default MergeTree shape;
//!   * an `UPDATE` that mapped several live keys onto one collapsed them and
//!     reported the pre-collapse count as "rows affected";
//!   * a result cell wider than 65535 characters aborted the process inside
//!     `std::fmt`; and
//!   * a transaction survived a failed statement, absorbing the ones after it.
//!
//! Reopening the directory without an explicit `checkpoint()` is the point of
//! most of them: what survives is exactly what the statement itself made
//! durable, which is the claim the refusal used to be protecting.

use std::path::PathBuf;

use granular::common::{Error, Result};
use granular::session::{ResultSet, Session};
use granular::types::Value;

// ---------------------------------------------------------------- fixtures

/// A data directory that removes itself. Named from the pid plus a counter, so
/// a rerun cannot collide with a live process and a leftover is attributable.
struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Dir {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "granular-mutation-{}-{}-{tag}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("create the scratch directory");
        Dir(p)
    }
    /// Reopen the directory as a fresh session, the way a restart would.
    fn open(&self) -> Session {
        Session::open(&self.0).expect("reopen the data directory")
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Every row of a query, as `Value`s.
fn rows(s: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    s.query(sql).unwrap_or_else(|e| panic!("{sql}: {e}")).to_values()
}

/// The single cell of a 1x1 result.
fn one(s: &mut Session, sql: &str) -> Value {
    s.query(sql).unwrap_or_else(|e| panic!("{sql}: {e}")).scalar().expect("a 1x1 result")
}

/// `Int` values of a one-column query, in order.
fn ints(s: &mut Session, sql: &str) -> Vec<i64> {
    rows(s, sql)
        .into_iter()
        .map(|r| match r[0] {
            Value::Int(v) => v,
            Value::UInt(v) => v as i64,
            ref other => panic!("{sql}: not an integer: {other:?}"),
        })
        .collect()
}

fn affected(r: Result<ResultSet>) -> Option<usize> {
    r.expect("statement failed").affected
}

// ------------------------------------------------ 1. the three table shapes

/// The headline defect: on a logging session, `DELETE` and `UPDATE` were
/// refused for every table without a *single-column* PRIMARY KEY -- which is
/// the shape `CREATE TABLE ... ENGINE = MergeTree ORDER BY id` produces, and
/// the shape a composite key produces. All three shapes now mutate, and the
/// mutation survives a reopen with no `checkpoint()` anywhere.
#[test]
fn delete_and_update_work_on_every_persistent_table_shape() {
    // (a) ORDER BY only -- no PRIMARY KEY at all, the default MergeTree shape.
    // (b) a composite PRIMARY KEY.
    // (c) a single-column PRIMARY KEY, the one shape that already worked; it
    //     is here so a regression on the keyed path cannot hide.
    let shapes: [(&str, &str); 3] = [
        ("order_by_only", "ENGINE = MergeTree ORDER BY id"),
        ("composite_pk", "ENGINE = MergeTree PRIMARY KEY (id, v) ORDER BY (id, v)"),
        ("single_pk", "ENGINE = MergeTree PRIMARY KEY id ORDER BY id"),
    ];
    for (tag, engine) in shapes {
        let dir = Dir::new(tag);
        {
            let mut db = dir.open();
            db.execute(&format!("CREATE TABLE t (id Int64, v Int64) {engine}"))
                .unwrap_or_else(|e| panic!("{tag}: create: {e}"));
            db.execute("INSERT INTO t VALUES (1,10),(2,20),(3,30),(4,40)")
                .unwrap_or_else(|e| panic!("{tag}: insert: {e}"));

            let n = affected(db.query("DELETE FROM t WHERE id = 1"));
            assert_eq!(n, Some(1), "{tag}: DELETE reported the wrong count");
            assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY id"), [2, 3, 4], "{tag}");

            let n = affected(db.query("UPDATE t SET v = v + 100 WHERE id = 2"));
            assert_eq!(n, Some(1), "{tag}: UPDATE reported the wrong count");
            assert_eq!(
                ints(&mut db, "SELECT v FROM t ORDER BY id"),
                [120, 30, 40],
                "{tag}: UPDATE wrote the wrong rows"
            );

            // And a bulk shape, so the sweep is not only exercised one row at
            // a time: everything above id 2 goes.
            let n = affected(db.query("DELETE FROM t WHERE id > 2"));
            assert_eq!(n, Some(2), "{tag}: bulk DELETE count");
            assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY id"), [2], "{tag}");
        }
        // Dropped without a checkpoint. Whatever is here now is what the
        // statements made durable on their own.
        let mut db = dir.open();
        assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY id"), [2], "{tag}: after reopen");
        assert_eq!(ints(&mut db, "SELECT v FROM t ORDER BY id"), [120], "{tag}: after reopen");
    }
}

/// The fold writes into `<root>/<db>/<table>/`, so a table outside `default`
/// is where a path built from the wrong half would go silently wrong: the
/// parts would land beside the wrong directory and the reopen would show the
/// pre-mutation rows with no error anywhere.
#[test]
fn an_unkeyed_mutation_in_a_named_database_lands_in_its_own_directory() {
    let dir = Dir::new("named-db");
    {
        let mut db = dir.open();
        db.execute("CREATE DATABASE shop").unwrap();
        db.execute("CREATE TABLE shop.t (id Int64, v Int64) ENGINE = MergeTree ORDER BY id")
            .unwrap();
        db.execute("INSERT INTO shop.t VALUES (1,10),(2,20),(3,30)").unwrap();
        db.execute("DELETE FROM shop.t WHERE id = 2").unwrap();
        db.execute("UPDATE shop.t SET v = 111 WHERE id = 1").unwrap();
        assert_eq!(ints(&mut db, "SELECT v FROM shop.t ORDER BY id"), [111, 30]);
    }
    let mut db = dir.open();
    assert_eq!(ints(&mut db, "SELECT v FROM shop.t ORDER BY id"), [111, 30], "after reopen");
    // The default database must not have grown a shadow copy.
    assert!(db.query("SELECT count() FROM default.t").is_err());
}

/// A sort key is not a unique key, so an unkeyed table may hold several rows
/// under one `ORDER BY` value -- and a positional sweep has to hit exactly the
/// ones the predicate names, not "the rows with that sort key".
#[test]
fn a_sweep_on_duplicate_sort_keys_hits_only_the_matching_rows() {
    let dir = Dir::new("dup-sort-keys");
    {
        let mut db = dir.open();
        db.execute("CREATE TABLE t (id Int64, v Int64) ENGINE = MergeTree ORDER BY id").unwrap();
        db.execute("INSERT INTO t VALUES (1,10),(1,20),(1,30),(2,40)").unwrap();
        assert_eq!(affected(db.query("DELETE FROM t WHERE v = 20")), Some(1));
        assert_eq!(ints(&mut db, "SELECT v FROM t ORDER BY v"), [10, 30, 40]);

        assert_eq!(affected(db.query("UPDATE t SET v = 99 WHERE v = 30")), Some(1));
        assert_eq!(ints(&mut db, "SELECT v FROM t ORDER BY v"), [10, 40, 99]);
    }
    let mut db = dir.open();
    assert_eq!(ints(&mut db, "SELECT v FROM t ORDER BY v"), [10, 40, 99], "after reopen");
    assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY v"), [1, 2, 1], "after reopen");
}

/// The sweep is a bulk statement: 20k rows out of 40k, on the shape that used
/// to be refused outright, and the survivors have to be exactly the others.
#[test]
fn a_large_unkeyed_delete_lands_and_survives_a_reopen() {
    let dir = Dir::new("bulk-unkeyed");
    const N: i64 = 40_000;
    {
        let mut db = dir.open();
        db.execute("CREATE TABLE t (id Int64, v Int64) ENGINE = MergeTree ORDER BY id").unwrap();
        let vals: Vec<String> = (0..N).map(|i| format!("({i},{})", i * 2)).collect();
        db.execute(&format!("INSERT INTO t VALUES {}", vals.join(","))).unwrap();
        assert_eq!(
            affected(db.query(&format!("DELETE FROM t WHERE id < {}", N / 2))),
            Some((N / 2) as usize)
        );
        assert_eq!(one(&mut db, "SELECT count() FROM t"), Value::UInt((N / 2) as u64));
    }
    let mut db = dir.open();
    assert_eq!(one(&mut db, "SELECT count() FROM t"), Value::UInt((N / 2) as u64));
    assert_eq!(one(&mut db, "SELECT min(id) FROM t"), Value::Int(N / 2));
    assert_eq!(one(&mut db, "SELECT max(id) FROM t"), Value::Int(N - 1));
    assert_eq!(one(&mut db, "SELECT sum(v) FROM t"), Value::Int((N / 2..N).map(|i| i * 2).sum()));
}

/// The unkeyed mutation is durable *per statement*, so a session that never
/// checkpoints and never shuts down cleanly still keeps every one of them --
/// including the last, which is the one a log-only design would lose.
#[test]
fn a_run_of_unkeyed_mutations_all_survive() {
    let dir = Dir::new("run-of-mutations");
    {
        let mut db = dir.open();
        db.execute("CREATE TABLE t (id Int64, v Int64) ENGINE = MergeTree ORDER BY id").unwrap();
        for i in 0..8i64 {
            db.execute(&format!("INSERT INTO t VALUES ({i},{i})")).unwrap();
            if i % 2 == 1 {
                db.execute(&format!("DELETE FROM t WHERE id = {}", i - 1)).unwrap();
            }
        }
        assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY id"), [1, 3, 5, 7]);
    }
    let mut db = dir.open();
    assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY id"), [1, 3, 5, 7], "after reopen");
}

/// Mutations inside an explicit transaction: ROLLBACK must leave every row,
/// COMMIT must publish *and* make durable. The unkeyed path defers its
/// durability to COMMIT, so this is the case where deferring could go wrong.
#[test]
fn an_unkeyed_mutation_obeys_the_transaction_it_is_in() {
    let dir = Dir::new("unkeyed-txn");
    {
        let mut db = dir.open();
        db.execute("CREATE TABLE t (id Int64, v Int64) ENGINE = MergeTree ORDER BY id").unwrap();
        db.execute("INSERT INTO t VALUES (1,10),(2,20),(3,30)").unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("DELETE FROM t WHERE id = 1").unwrap();
        assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY id"), [2, 3], "visible inside");
        db.execute("ROLLBACK").unwrap();
        assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY id"), [1, 2, 3], "rolled back");

        db.execute("BEGIN").unwrap();
        db.execute("DELETE FROM t WHERE id = 1").unwrap();
        db.execute("UPDATE t SET v = 222 WHERE id = 2").unwrap();
        db.execute("COMMIT").unwrap();
        assert_eq!(ints(&mut db, "SELECT v FROM t ORDER BY id"), [222, 30]);
    }
    let mut db = dir.open();
    assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY id"), [2, 3], "after reopen");
    assert_eq!(ints(&mut db, "SELECT v FROM t ORDER BY id"), [222, 30], "after reopen");
}

/// An INSERT and a DELETE of the *same rows* in one statement stream is the
/// case a log-plus-fold design gets wrong if the two halves have different
/// commit points: replay the insert, lose the delete, and the rows come back.
/// Every prefix of this script has to reopen to a state the script passed
/// through, and the final one has to be the last statement's.
#[test]
fn an_insert_then_delete_never_resurrects_the_rows() {
    let dir = Dir::new("insert-then-delete");
    {
        let mut db = dir.open();
        db.execute("CREATE TABLE t (id Int64, v Int64) ENGINE = MergeTree ORDER BY id").unwrap();
        db.execute("INSERT INTO t VALUES (1,10),(2,20)").unwrap();
        db.execute("INSERT INTO t VALUES (3,30),(4,40)").unwrap();
        db.execute("DELETE FROM t WHERE id % 2 = 1").unwrap();
        assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY id"), [2, 4]);
    }
    let mut db = dir.open();
    assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY id"), [2, 4], "after reopen");

    // The same inside one transaction, where the insert's log record and the
    // sweep's tombstones have to become durable together or not at all.
    let dir = Dir::new("insert-then-delete-txn");
    {
        let mut db = dir.open();
        db.execute("CREATE TABLE t (id Int64, v Int64) ENGINE = MergeTree ORDER BY id").unwrap();
        db.execute("INSERT INTO t VALUES (1,10)").unwrap();
        db.execute("BEGIN").unwrap();
        db.execute("INSERT INTO t VALUES (2,20),(3,30)").unwrap();
        db.execute("DELETE FROM t WHERE id < 3").unwrap();
        db.execute("COMMIT").unwrap();
        assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY id"), [3]);
    }
    let mut db = dir.open();
    assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY id"), [3], "after reopen");

    // ...and a ROLLBACK of the same shape leaves the pre-transaction rows,
    // with nothing of the transaction on disk.
    let dir = Dir::new("insert-then-delete-rollback");
    {
        let mut db = dir.open();
        db.execute("CREATE TABLE t (id Int64, v Int64) ENGINE = MergeTree ORDER BY id").unwrap();
        db.execute("INSERT INTO t VALUES (1,10)").unwrap();
        db.execute("BEGIN").unwrap();
        db.execute("INSERT INTO t VALUES (2,20)").unwrap();
        db.execute("DELETE FROM t WHERE id = 1").unwrap();
        db.execute("ROLLBACK").unwrap();
        assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY id"), [1]);
    }
    let mut db = dir.open();
    assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY id"), [1], "after reopen");
}

// ------------------------------------------------- 2. colliding primary keys

/// `UPDATE t SET id = 9` over three rows used to print "Ok. 3 rows affected."
/// and leave one row. Mapping distinct live keys onto one is a unique-key
/// violation, not a write, so it must fail and change nothing.
#[test]
fn an_update_that_collides_primary_keys_errors_and_changes_nothing() {
    let dir = Dir::new("collide-pk");
    {
        let mut db = dir.open();
        db.execute("CREATE TABLE t (id Int64, v Int64) ENGINE = MergeTree PRIMARY KEY id ORDER BY id")
            .unwrap();
        db.execute("INSERT INTO t VALUES (1,10),(2,20),(3,30)").unwrap();

        let err = db.query("UPDATE t SET id = 9").expect_err("a colliding UPDATE must fail");
        assert!(err.to_string().contains("primary key"), "{err}");
        assert_eq!(
            rows(&mut db, "SELECT id, v FROM t ORDER BY id"),
            vec![
                vec![Value::Int(1), Value::Int(10)],
                vec![Value::Int(2), Value::Int(20)],
                vec![Value::Int(3), Value::Int(30)],
            ],
            "the refused UPDATE must leave the table exactly as it was"
        );

        // Two rows onto one is the same violation, and so is landing on a key
        // the statement did not name -- the row at id = 3 is not in the
        // predicate, so overwriting it would be pure loss.
        assert!(db.query("UPDATE t SET id = 5 WHERE id < 3").is_err());
        assert!(db.query("UPDATE t SET id = 3 WHERE id = 1").is_err());
        assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY id"), [1, 2, 3]);

        // A key assignment that stays injective is a perfectly good UPDATE,
        // and every row it names lands -- including the ones that move onto
        // keys the same statement vacated.
        assert_eq!(affected(db.query("UPDATE t SET id = id + 10")), Some(3));
        assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY id"), [11, 12, 13]);
        assert_eq!(ints(&mut db, "SELECT v FROM t ORDER BY id"), [10, 20, 30]);
        assert_eq!(affected(db.query("UPDATE t SET id = id - 1")), Some(3));
        assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY id"), [10, 11, 12]);
        assert_eq!(ints(&mut db, "SELECT v FROM t ORDER BY id"), [10, 20, 30]);
    }
    let mut db = dir.open();
    assert_eq!(ints(&mut db, "SELECT id FROM t ORDER BY id"), [10, 11, 12], "after reopen");
    assert_eq!(ints(&mut db, "SELECT v FROM t ORDER BY id"), [10, 20, 30], "after reopen");
}

/// The count a mutation reports has to be the number of rows that landed.
/// `UPDATE t SET id = 9` reporting 3 while storing 1 was the visible half of
/// the collapse; every count below is checked against a `count()` of the table.
#[test]
fn a_mutation_reports_the_rows_that_actually_landed() {
    let dir = Dir::new("honest-counts");
    let mut db = dir.open();
    db.execute("CREATE TABLE t (id Int64, v Int64) ENGINE = MergeTree PRIMARY KEY id ORDER BY id")
        .unwrap();
    db.execute("INSERT INTO t VALUES (1,10),(2,20),(3,30),(4,40)").unwrap();

    assert_eq!(affected(db.query("UPDATE t SET v = 0 WHERE id <= 2")), Some(2));
    assert_eq!(one(&mut db, "SELECT count() FROM t"), Value::UInt(4));
    assert_eq!(one(&mut db, "SELECT sum(v) FROM t"), Value::Int(70));

    assert_eq!(affected(db.query("DELETE FROM t WHERE id = 99")), Some(0));
    assert_eq!(affected(db.query("UPDATE t SET v = 1 WHERE id = 99")), Some(0));
    assert_eq!(one(&mut db, "SELECT count() FROM t"), Value::UInt(4));

    assert_eq!(affected(db.query("DELETE FROM t")), Some(4));
    assert_eq!(one(&mut db, "SELECT count() FROM t"), Value::UInt(0));
}

// ------------------------------------------------------- 3. the wide cell

/// `{:w$}` stores its runtime width in a `u16`, so rendering a cell (or a
/// column name) longer than 65535 characters panicked inside `std::fmt` --
/// and with `panic = "abort"` in the release profile that is a SIGABRT of the
/// whole process, from one long JSON blob.
#[test]
fn a_cell_wider_than_a_u16_renders_instead_of_aborting() {
    let mut db = Session::in_memory();
    db.execute("CREATE TABLE t (s String) ENGINE = Log").unwrap();
    for n in [65_534usize, 65_535, 65_536, 70_000] {
        db.execute("TRUNCATE TABLE t").unwrap();
        db.execute(&format!("INSERT INTO t VALUES ('{}')", "x".repeat(n))).unwrap();
        let text = db.query("SELECT s FROM t").unwrap().to_string();
        assert!(text.contains(&"x".repeat(n)), "the {n}-char cell was not rendered whole");
        // The frame is still a frame: header, rule, one row, rule, footer.
        assert_eq!(text.lines().count(), 6, "n = {n}");
    }
}

/// The same ceiling applies to a column *name*, which reaches the formatter by
/// the same route.
#[test]
fn a_column_name_wider_than_a_u16_renders_instead_of_aborting() {
    let mut db = Session::in_memory();
    let name = "n".repeat(70_000);
    let text = db.query(&format!("SELECT 1 AS \"{name}\"")).unwrap().to_string();
    assert!(text.contains(&name), "the long column name was truncated");
}

/// Padding is by display width, not bytes: a multi-byte cell must still line
/// the columns up, which is what the hand-rolled pad has to preserve.
#[test]
fn multibyte_cells_still_line_up() {
    let mut db = Session::in_memory();
    db.execute("CREATE TABLE t (s String) ENGINE = Log").unwrap();
    db.execute("INSERT INTO t VALUES ('é'), ('日本語'), ('abcdef')").unwrap();
    let text = db.query("SELECT s FROM t").unwrap().to_string();
    let widths: Vec<usize> = text
        .lines()
        .filter(|l| l.starts_with('│'))
        .map(|l| l.chars().count())
        .collect();
    assert!(widths.windows(2).all(|w| w[0] == w[1]), "ragged rows: {widths:?}\n{text}");
}

// ------------------------------------------- 4. the transaction state machine

/// A nested `BEGIN` errors, leaves the outer transaction open -- and poisons
/// it, so the inner block's `COMMIT` can no longer durably commit the outer
/// transaction's uncommitted work.
#[test]
fn a_nested_begin_cannot_commit_the_outer_transaction() {
    let dir = Dir::new("nested-begin");
    {
        let mut db = dir.open();
        db.execute("CREATE TABLE t (id Int64, v Int64) ENGINE = MergeTree PRIMARY KEY id ORDER BY id")
            .unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("INSERT INTO t VALUES (1,10)").unwrap();
        // The inner block, which believes it is opening its own transaction.
        assert!(db.execute("BEGIN").is_err(), "nesting must be refused");
        assert!(db.in_transaction(), "and must not close the outer transaction");
        db.execute("INSERT INTO t VALUES (2,20)").unwrap_err();
        let e = db.execute("COMMIT").expect_err("the inner COMMIT must not publish");
        assert!(e.to_string().contains("failed"), "{e}");
        assert!(!db.in_transaction(), "a refused COMMIT still ends the transaction");
        assert_eq!(one(&mut db, "SELECT count() FROM t"), Value::UInt(0));
    }
    let mut db = dir.open();
    assert_eq!(one(&mut db, "SELECT count() FROM t"), Value::UInt(0), "after reopen");
}

/// A statement that fails inside a transaction poisons it: every later
/// statement is refused until ROLLBACK, rather than returning `Ok` over work
/// that is then discarded at exit.
#[test]
fn a_failed_statement_poisons_the_transaction_until_rollback() {
    let dir = Dir::new("poison");
    let mut db = dir.open();
    db.execute("CREATE TABLE t (id Int64, v Int64) ENGINE = MergeTree PRIMARY KEY id ORDER BY id")
        .unwrap();
    db.execute("INSERT INTO t VALUES (1,10)").unwrap();

    db.execute("BEGIN").unwrap();
    db.execute("INSERT INTO t VALUES (2,20)").unwrap();
    // Any error will do; an unknown column is the most ordinary one.
    let first = db.execute("SELECT nosuchcolumn FROM t").unwrap_err();
    assert_eq!(first.code(), "UNKNOWN_IDENTIFIER", "{first}");

    for sql in ["INSERT INTO t VALUES (3,30)", "SELECT count() FROM t", "DELETE FROM t"] {
        let e = db.execute(sql).unwrap_err();
        assert!(e.to_string().contains("nosuchcolumn"), "{sql}: {e}");
    }
    // A syntax error is a failed statement too, and poisons the same way.
    assert!(db.execute("COMMIT").is_err(), "COMMIT must not publish a poisoned transaction");
    assert!(!db.in_transaction());
    assert_eq!(one(&mut db, "SELECT count() FROM t"), Value::UInt(1), "nothing was published");

    // ROLLBACK is the way out, and it always works.
    db.execute("BEGIN").unwrap();
    db.execute("INSERT INTO t VALUES (2,20)").unwrap();
    assert!(db.execute("SELECT nosuchcolumn FROM t").is_err());
    db.execute("ROLLBACK").unwrap();
    assert!(!db.in_transaction());
    // ...and the session is ordinary again afterwards.
    db.execute("BEGIN").unwrap();
    db.execute("INSERT INTO t VALUES (2,20)").unwrap();
    db.execute("COMMIT").unwrap();
    assert_eq!(one(&mut db, "SELECT count() FROM t"), Value::UInt(2));
}

// -------------------------------------------------- 5. through the real binary

/// Everything above drives the library. This drives the **process**, because
/// two of the defects are only fully visible there: `panic = "abort"` is set
/// for the release profile, so the 65535-character ceiling was a SIGABRT
/// rather than an error, and a mutation refused inside `Session` is a
/// non-zero exit for any script that shells out.
mod cli {
    use std::path::Path;
    use std::process::{Command, Output};

    const BIN: &str = env!("CARGO_BIN_EXE_granular");

    fn run(dir: &Path, sql: &str) -> Output {
        Command::new(BIN)
            .arg("--data")
            .arg(dir)
            .arg("-q")
            .arg(sql)
            .output()
            .expect("spawn the granular binary")
    }

    /// Exit status, not just "did it print something": a script's only signal.
    fn ok(dir: &Path, sql: &str) -> String {
        let out = run(dir, sql);
        assert!(
            out.status.success(),
            "`{sql}` exited {:?} (signal {:?})\nstderr: {}",
            out.status.code(),
            signal_of(&out),
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    #[cfg(unix)]
    fn signal_of(out: &Output) -> Option<i32> {
        use std::os::unix::process::ExitStatusExt;
        out.status.signal()
    }
    #[cfg(not(unix))]
    fn signal_of(_: &Output) -> Option<i32> {
        None
    }

    /// The data rows of a rendered result, header and footer dropped.
    ///
    /// Substring matching on the whole output does not work: the footer
    /// carries an elapsed time, so `"20"` matches `0.200 ms` and a test that
    /// asserts a row is *gone* passes for the wrong reason. Ask for the cells.
    fn cells(out: &str) -> Vec<Vec<String>> {
        out.lines()
            .filter(|l| l.starts_with('│'))
            .skip(1) // the header row
            .map(|l| l.trim_matches('│').split('│').map(|c| c.trim().to_string()).collect())
            .collect()
    }

    /// The whole lifecycle through separate processes: create, insert, mutate,
    /// then read back from a *new* process. Nothing checkpoints explicitly, so
    /// the last process only sees what the mutating ones made durable.
    #[test]
    fn delete_and_update_survive_across_processes() {
        let dir = super::Dir::new("cli-lifecycle");
        let d = &dir.0;
        ok(d, "CREATE TABLE t (id Int64, v Int64) ENGINE = MergeTree ORDER BY id");
        ok(d, "INSERT INTO t VALUES (1,10),(2,20),(3,30)");
        assert!(
            ok(d, "DELETE FROM t WHERE id = 2").contains("1 row affected"),
            "the DELETE did not report one row"
        );
        ok(d, "UPDATE t SET v = 333 WHERE id = 3");

        let rows = cells(&ok(d, "SELECT id, v FROM t ORDER BY id"));
        assert_eq!(rows, vec![vec!["1", "10"], vec!["3", "333"]], "after two fresh processes");
        assert_eq!(cells(&ok(d, "SELECT count() FROM t")), vec![vec!["2"]]);
    }

    /// The abort. In the release profile the panic this used to raise had no
    /// unwinder behind it, so the process died on SIGABRT (exit 134) with no
    /// output at all -- which is what a library embedding would have done to
    /// its host. Success plus a stdout longer than the cell is the assertion.
    #[test]
    fn a_70k_character_cell_does_not_kill_the_process() {
        let dir = super::Dir::new("cli-wide-cell");
        let d = &dir.0;
        ok(d, "CREATE TABLE t (s String) ENGINE = Log");
        ok(d, &format!("INSERT INTO t VALUES ('{}')", "x".repeat(70_000)));
        let out = run(d, "SELECT s FROM t");
        assert_eq!(signal_of(&out), None, "the process was killed by a signal");
        assert!(out.status.success(), "exited {:?}", out.status.code());
        assert!(
            out.stdout.len() > 70_000,
            "only {} bytes of output -- the cell was not rendered",
            out.stdout.len()
        );
    }
}

/// A syntax error never reaches `exec_statement`, so it has to be poisoned at
/// the same place the parse happens or it would slip through.
#[test]
fn a_syntax_error_poisons_the_transaction_too() {
    let mut db = Session::in_memory();
    db.execute("CREATE TABLE t (id Int64) ENGINE = MergeTree PRIMARY KEY id ORDER BY id").unwrap();
    db.execute("BEGIN").unwrap();
    db.execute("INSERT INTO t VALUES (1)").unwrap();
    assert!(db.execute("SELEKT 1").is_err());
    assert!(db.execute("INSERT INTO t VALUES (2)").is_err(), "poisoned by the syntax error");
    assert!(db.execute("COMMIT").is_err());
    assert_eq!(one(&mut db, "SELECT count() FROM t"), Value::UInt(0));
}

/// One script, several statements, run through `Session::run` in a single
/// call -- the shape a `-f script.sql` invocation takes. The failure has to
/// stop the batch rather than let the rest of it accumulate into a commit.
#[test]
fn a_script_stops_at_the_first_failure_inside_a_transaction() {
    let mut db = Session::in_memory();
    db.execute("CREATE TABLE t (id Int64) ENGINE = MergeTree PRIMARY KEY id ORDER BY id").unwrap();
    let err: Error = db
        .run("BEGIN; INSERT INTO t VALUES (1); SELECT nosuchcolumn FROM t; INSERT INTO t VALUES (2); COMMIT;")
        .expect_err("the script must fail");
    assert_eq!(err.code(), "UNKNOWN_IDENTIFIER", "{err}");
    // The transaction is still open, poisoned, and holding nothing anybody can
    // commit. Rolling it back is the only way on.
    assert!(db.in_transaction());
    db.execute("ROLLBACK").unwrap();
    assert_eq!(one(&mut db, "SELECT count() FROM t"), Value::UInt(0));
}
