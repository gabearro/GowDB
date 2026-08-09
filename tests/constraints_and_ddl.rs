//! Constraints and schema changes, driven through the shipped binary.
//!
//! Everything here runs `granular` as a child process against a real data
//! directory, and almost everything reopens that directory in a *second*
//! process before it asserts. That is the point of the file. A `CHECK` that
//! rejects a bad row in the session that declared it and forgets the
//! constraint on restart is not a constraint, and this project has shipped
//! that shape of defect enough times to have a name for it: the capability
//! lands complete in `src/` and never reaches a user.
//!
//! The four claims:
//!
//!   1. **A refused write leaves nothing behind.** Not a row, not a log
//!      record, not a half-applied batch -- and `kill -9` immediately after a
//!      refusal must not resurrect it. `a_refused_unique_insert_is_not_replayed`
//!      is the sharp one: the record is staged and never committed, so
//!      recovery has nothing to apply.
//!   2. **A constraint is part of the table, not of the session.** It survives
//!      a reopen, a `RENAME`, and a `BACKUP`/`RESTORE` round trip into a
//!      different directory -- which is why the metadata lives in a table
//!      rather than in a file beside the catalog.
//!   3. **Schema changes are all-or-nothing.** `MODIFY COLUMN` refuses the
//!      whole statement over one value that does not fit, naming the row, and
//!      a `kill -9` in the middle of one leaves the table fully old or fully
//!      new -- never half migrated.
//!   4. **What is not enforceable is refused, loudly.** `UNIQUE` on a column
//!      with no index behind it is a DDL error naming what is missing, not an
//!      accepted declaration that does nothing.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

const BIN: &str = env!("CARGO_BIN_EXE_granular");

// ------------------------------------------------------------------ harness

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "granular-ddl-{}-{}-{tag}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("scratch dir");
        Scratch(d)
    }
    fn s(&self) -> String {
        self.0.to_string_lossy().into_owned()
    }
    /// A sibling path that does not exist yet -- a restore target, or a second
    /// database.
    fn sibling(&self, name: &str) -> String {
        let p = self.0.with_file_name(format!(
            "{}-{name}",
            self.0.file_name().unwrap().to_string_lossy()
        ));
        let _ = std::fs::remove_dir_all(&p);
        p.to_string_lossy().into_owned()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
        // Restore targets and archives are named `<dir>-<tag>`; sweep them too.
        if let Some(parent) = self.0.parent() {
            if let Ok(rd) = std::fs::read_dir(parent) {
                let prefix = format!("{}-", self.0.file_name().unwrap().to_string_lossy());
                for e in rd.flatten() {
                    if e.file_name().to_string_lossy().starts_with(&prefix) {
                        let _ = std::fs::remove_dir_all(e.path());
                        let _ = std::fs::remove_file(e.path());
                    }
                }
            }
        }
    }
}

struct Run {
    code: i32,
    out: String,
    err: String,
}

impl Run {
    fn of(o: Output) -> Run {
        Run {
            code: o.status.code().unwrap_or(-1),
            out: String::from_utf8_lossy(&o.stdout).into_owned(),
            err: String::from_utf8_lossy(&o.stderr).into_owned(),
        }
    }
    fn ok(self) -> Run {
        assert_eq!(self.code, 0, "expected success\nstdout:\n{}\nstderr:\n{}", self.out, self.err);
        self
    }
    /// Failed, and the message says `needle`. Both halves matter: a refusal
    /// with the wrong reason is a different bug wearing the right exit code.
    fn fails_with(self, needle: &str) -> Run {
        assert_eq!(self.code, 1, "expected failure\nstdout:\n{}\nstderr:\n{}", self.out, self.err);
        assert!(
            self.err.contains(needle),
            "expected an error mentioning `{needle}`, got:\n{}",
            self.err
        );
        self
    }
}

/// One CLI invocation, in its own process, against `dir`.
fn run(dir: &str, sql: &str) -> Run {
    Run::of(
        Command::new(BIN)
            .args(["--data", dir, "-q", sql])
            .stdin(Stdio::null())
            .output()
            .expect("spawn granular"),
    )
}

/// The single value a one-cell query returns, with the box drawing stripped.
fn cell(dir: &str, sql: &str) -> String {
    let r = run(dir, sql).ok();
    r.out
        .lines()
        .filter(|l| l.starts_with('│'))
        .nth(1)
        .unwrap_or_else(|| panic!("no data row in:\n{}", r.out))
        .trim_matches(|c| c == '│' || c == ' ')
        .trim()
        .to_string()
}

/// Every data cell of a query, row-major, `|`-joined per row.
fn rows(dir: &str, sql: &str) -> Vec<String> {
    let r = run(dir, sql).ok();
    r.out
        .lines()
        .filter(|l| l.starts_with('│'))
        .skip(1)
        .map(|l| {
            l.trim_matches('│')
                .split('│')
                .map(|c| c.trim().to_string())
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect()
}

/// A table of `n` rows: `id` 0..n, `v` = id, `s` = 's<id>'.
fn fill(dir: &str, table: &str, n: usize) {
    // In chunks, because the whole point of some of these tables is to be big
    // enough to kill a rewrite in the middle of, and one `-q` argument holding
    // 60k tuples is past `ARG_MAX`.
    for base in (0..n).step_by(4096) {
        let vals: Vec<String> =
            (base..n.min(base + 4096)).map(|i| format!("({i},{i},'s{i}')")).collect();
        run(dir, &format!("INSERT INTO {table} VALUES {}", vals.join(","))).ok();
    }
}

// ================================================================ TASK A: CHECK

/// A violated CHECK refuses the statement and leaves the table byte-identical
/// -- in this process and in the next one.
#[test]
fn a_violated_check_refuses_the_write_and_changes_nothing() {
    let d = Scratch::new("check-refuses");
    let dir = d.s();
    run(
        &dir,
        "CREATE TABLE k (id UInt64, v Int64 CHECK (v > 0), s String) \
         ENGINE = MergeTree ORDER BY id",
    )
    .ok();
    run(&dir, "INSERT INTO k VALUES (1, 10, 'a'), (2, 20, 'b')").ok();

    // One bad row in the middle of a good batch fails the whole batch: a
    // partially applied INSERT would be the worse answer by far.
    run(&dir, "INSERT INTO k VALUES (3, 30, 'c'), (4, -1, 'd'), (5, 50, 'e')")
        .fails_with("CHECK constraint");
    assert_eq!(cell(&dir, "SELECT count() FROM k"), "2", "a refused batch left rows behind");

    // And a single bad row on its own.
    run(&dir, "INSERT INTO k VALUES (9, 0, 'zero')").fails_with("CHECK constraint");
    assert_eq!(rows(&dir, "SELECT id, v FROM k ORDER BY id"), ["1|10", "2|20"]);

    // The rejection names the constraint, the row, and the predicate, because
    // "constraint violated" without those is a support ticket.
    let e = run(&dir, "INSERT INTO k VALUES (9, -7, 'x')").fails_with("check_v").err;
    assert!(e.contains("id=9") && e.contains("v=-7"), "{e}");
    assert!(e.contains("v > 0"), "{e}");

    // Second process: the constraint is in the catalog, not in the session.
    assert_eq!(cell(&dir, "SELECT count() FROM k"), "2");
    run(&dir, "INSERT INTO k VALUES (6, -6, 'f')").fails_with("CHECK constraint");
    run(&dir, "INSERT INTO k VALUES (6, 60, 'f')").ok();
    assert_eq!(cell(&dir, "SELECT count() FROM k"), "3");
}

/// UPDATE is the other way a stored row comes to violate a constraint.
#[test]
fn a_check_is_enforced_on_update_too() {
    let d = Scratch::new("check-update");
    let dir = d.s();
    run(
        &dir,
        "CREATE TABLE k (id UInt64, v Int64, CONSTRAINT positive CHECK (v > 0)) \
         ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
    )
    .ok();
    run(&dir, "INSERT INTO k VALUES (1, 10), (2, 20)").ok();

    run(&dir, "UPDATE k SET v = -1 WHERE id = 1").fails_with("positive");
    assert_eq!(rows(&dir, "SELECT id, v FROM k ORDER BY id"), ["1|10", "2|20"]);
    // The legal one still works, so the check is not simply refusing UPDATEs.
    run(&dir, "UPDATE k SET v = 11 WHERE id = 1").ok();
    assert_eq!(rows(&dir, "SELECT id, v FROM k ORDER BY id"), ["1|11", "2|20"]);
}

/// SQL's rule, kept deliberately: a CHECK is violated only by **FALSE**, so a
/// NULL passes it. Pinned because the other reading is defensible and someone
/// will eventually "fix" this: declare the column non-nullable if NULL is not
/// allowed -- nullability is part of the type here, and the type refuses it.
#[test]
fn a_null_passes_a_check_and_a_non_nullable_column_is_how_you_refuse_it() {
    let d = Scratch::new("check-null");
    let dir = d.s();
    run(
        &dir,
        "CREATE TABLE n (id UInt64, v Nullable(Int64) CHECK (v > 0)) \
         ENGINE = MergeTree ORDER BY id",
    )
    .ok();
    run(&dir, "INSERT INTO n VALUES (1, NULL)").ok();
    run(&dir, "INSERT INTO n VALUES (2, -1)").fails_with("CHECK constraint");
    assert_eq!(cell(&dir, "SELECT count() FROM n"), "1");

    run(&dir, "CREATE TABLE m (id UInt64, v Int64 CHECK (v > 0)) ENGINE = MergeTree ORDER BY id")
        .ok();
    run(&dir, "INSERT INTO m VALUES (1, NULL)").fails_with("NULL");
}

/// The declaration itself is checked when it is written, not when a row
/// arrives: an unknown column, a non-condition and an aggregate are all the
/// user's mistake, and all three are cheaper to hear about now.
#[test]
fn an_unenforceable_check_is_refused_at_ddl_time() {
    let d = Scratch::new("check-ddl");
    let dir = d.s();
    let ddl = |body: &str| {
        format!("CREATE TABLE bad (id UInt64, v Int64, {body}) ENGINE = MergeTree ORDER BY id")
    };
    run(&dir, &ddl("CHECK (nosuch > 0)")).fails_with("nosuch");
    run(&dir, &ddl("CHECK (v)")).fails_with("condition");
    run(&dir, &ddl("CHECK (count() > 0)")).fails_with("aggregate");
    // Nothing was created by any of them.
    assert!(rows(&dir, "SHOW TABLES").is_empty(), "a refused CREATE TABLE left a table behind");
}

/// A constraint is table metadata, so it has to be *in the backup*. This is
/// the test the storage decision was made for: a sidecar file next to
/// `CATALOG` would restore into a database that had quietly lost its
/// constraints, and the first write after that restore would be accepted where
/// the original refuses it.
#[test]
fn constraints_and_views_survive_a_backup_and_restore() {
    let d = Scratch::new("backup");
    let dir = d.s();
    let archive = d.sibling("archive.gbak");
    let restored = d.sibling("restored");
    run(
        &dir,
        "CREATE TABLE k (id UInt64 UNIQUE, v Int64 CHECK (v > 0)) \
         ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
    )
    .ok();
    run(&dir, "INSERT INTO k VALUES (1, 10), (2, 20)").ok();
    run(&dir, "CREATE VIEW big AS SELECT id FROM k WHERE v >= 20").ok();
    run(&dir, &format!("BACKUP TO '{archive}'")).ok();

    run(&dir, &format!("RESTORE FROM '{archive}' TO '{restored}'")).ok();
    assert_eq!(cell(&restored, "SELECT count() FROM k"), "2");
    // The CHECK, the UNIQUE and the view all came with it.
    run(&restored, "INSERT INTO k VALUES (3, -3)").fails_with("CHECK constraint");
    run(&restored, "INSERT INTO k VALUES (1, 99)").fails_with("would be shared");
    assert_eq!(rows(&restored, "SELECT * FROM big"), ["2"]);
}

// ============================================================== TASK A: UNIQUE

/// `UNIQUE` on the key turns the upsert into a refusal, and the row that was
/// already there is untouched.
#[test]
fn unique_refuses_a_repeated_key_and_keeps_the_row_that_had_it() {
    let d = Scratch::new("unique");
    let dir = d.s();
    run(
        &dir,
        "CREATE TABLE u (id UInt64 UNIQUE, name String) \
         ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
    )
    .ok();
    run(&dir, "INSERT INTO u VALUES (1, 'first'), (2, 'second')").ok();

    run(&dir, "INSERT INTO u VALUES (1, 'usurper')").fails_with("would be shared");
    assert_eq!(rows(&dir, "SELECT id, name FROM u ORDER BY id"), ["1|first", "2|second"]);
    // Twice in one statement is the same violation.
    run(&dir, "INSERT INTO u VALUES (7, 'a'), (7, 'b')").fails_with("would be shared");
    assert_eq!(cell(&dir, "SELECT count() FROM u"), "2");

    // Survives a reopen: without the constraint the same INSERT would be an
    // upsert that silently replaced `first`.
    run(&dir, "INSERT INTO u VALUES (1, 'usurper')").fails_with("would be shared");
    assert_eq!(cell(&dir, "SELECT name FROM u WHERE id = 1"), "first");
    // ...and the same table without UNIQUE still upserts, which is what makes
    // the constraint a choice rather than a change of engine.
    run(&dir, "CREATE TABLE p (id UInt64, name String) ENGINE = MergeTree ORDER BY id PRIMARY KEY id")
        .ok();
    run(&dir, "INSERT INTO p VALUES (1, 'first')").ok();
    run(&dir, "INSERT INTO p VALUES (1, 'second')").ok();
    assert_eq!(cell(&dir, "SELECT name FROM p WHERE id = 1"), "second");
}

/// A `UNIQUE` this engine cannot enforce is a DDL error naming what is
/// missing. Accepting it silently is the failure mode this whole wave exists
/// to remove: the application would believe an invariant nothing was checking.
#[test]
fn unique_without_an_index_behind_it_is_refused_with_the_reason() {
    let d = Scratch::new("unique-refused");
    let dir = d.s();
    // No PRIMARY KEY at all: ORDER BY is a sort key, not a uniqueness claim.
    run(
        &dir,
        "CREATE TABLE a (id UInt64 UNIQUE, v Int64) ENGINE = MergeTree ORDER BY id",
    )
    .fails_with("PRIMARY KEY");
    // A key exists, but not on this column.
    run(
        &dir,
        "CREATE TABLE b (id UInt64, email String UNIQUE) \
         ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
    )
    .fails_with("email");
    // Multi-column UNIQUE has no single lane to index.
    run(
        &dir,
        "CREATE TABLE c (id UInt64, x Int64, UNIQUE (id, x)) \
         ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
    )
    .fails_with("multi-column");
    assert!(rows(&dir, "SHOW TABLES").is_empty(), "a refused CREATE TABLE left a table behind");
}

/// A refused write must not come back from the dead.
///
/// The record for a UNIQUE table is *staged* and only committed once the batch
/// is accepted, so a `kill -9` between the refusal and the exit has nothing to
/// replay. Logging first and refusing afterwards -- the ordinary order --
/// would leave a record that recovery applies with last-write-wins, landing
/// exactly the row the constraint rejected on top of the one it protected.
#[test]
fn a_refused_unique_insert_is_not_replayed_after_a_kill() {
    let d = Scratch::new("unique-replay");
    let dir = d.s();
    run(
        &dir,
        "CREATE TABLE u (id UInt64 UNIQUE, name String) \
         ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
    )
    .ok();
    run(&dir, "INSERT INTO u VALUES (1, 'original')").ok();

    // A child that takes the refusal and is then killed before it can exit --
    // so no checkpoint runs and recovery is driven entirely by the log.
    let mut child = Command::new(BIN)
        .args([
            "--data",
            &dir,
            "-q",
            "INSERT INTO u VALUES (1, 'usurper'); SELECT sleepEachRow(30) FROM u",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn");
    std::thread::sleep(Duration::from_millis(150));
    let _ = child.kill();
    let _ = child.wait();

    assert_eq!(cell(&dir, "SELECT count() FROM u"), "1", "the refused row was replayed");
    assert_eq!(cell(&dir, "SELECT name FROM u WHERE id = 1"), "original");
}

// ============================================================== TASK B: VIEWS

/// A view is a stored query: it survives a restart, it sees new rows, and it
/// can be built on another view.
#[test]
fn a_view_survives_a_reopen_and_follows_its_source() {
    let d = Scratch::new("view");
    let dir = d.s();
    run(&dir, "CREATE TABLE t (id UInt64, v Int64, s String) ENGINE = MergeTree ORDER BY id").ok();
    fill(&dir, "t", 10);
    run(&dir, "CREATE VIEW evens AS SELECT id, v FROM t WHERE v % 2 = 0").ok();

    // A different process: the view came out of the catalog.
    assert_eq!(cell(&dir, "SELECT count() FROM evens"), "5");
    // It is a query, not a copy -- new rows show up.
    run(&dir, "INSERT INTO t VALUES (100, 100, 'x')").ok();
    assert_eq!(cell(&dir, "SELECT count() FROM evens"), "6");
    // It composes: joins, aggregates and views over views all work, because a
    // reference is rewritten into the derived table it stands for.
    run(&dir, "CREATE VIEW big_evens AS SELECT id FROM evens WHERE id >= 100").ok();
    assert_eq!(rows(&dir, "SELECT * FROM big_evens"), ["100"]);
    assert_eq!(cell(&dir, "SELECT count() FROM evens e JOIN t ON e.id = t.id"), "6");

    // It is listed, and it says what it is.
    assert!(rows(&dir, "SHOW TABLES").iter().any(|r| r == "evens"));
    assert!(run(&dir, "SHOW CREATE VIEW evens").ok().out.contains("CREATE VIEW"));
    assert_eq!(rows(&dir, "DESCRIBE evens"), ["id|UInt64", "v|Int64"]);

}

/// A view body means what it meant when it was written, from any session.
///
/// `CREATE VIEW` records the database its unqualified names resolve in and
/// qualifies the stored body with it, so a `USE other` in the reader's session
/// cannot redirect the view at a different table that happens to share a name.
/// Without that the view silently answers from the wrong table.
#[test]
fn a_view_resolves_against_the_database_it_was_created_in() {
    let d = Scratch::new("view-db");
    let dir = d.s();
    run(&dir, "CREATE TABLE t (id UInt64, v Int64, s String) ENGINE = MergeTree ORDER BY id").ok();
    fill(&dir, "t", 3);
    run(&dir, "CREATE VIEW v AS SELECT count() AS n FROM t").ok();

    // A second database with a table of the same name and different contents.
    run(&dir, "CREATE DATABASE other").ok();
    run(
        &dir,
        "CREATE TABLE other.t (id UInt64, v Int64, s String) ENGINE = MergeTree ORDER BY id",
    )
    .ok();
    run(&dir, "INSERT INTO other.t VALUES (1,1,'a'),(2,2,'b'),(3,3,'c'),(4,4,'d')").ok();

    assert_eq!(cell(&dir, "SELECT n FROM v"), "3");
    assert_eq!(cell(&dir, "USE other; SELECT n FROM default.v"), "3", "the view followed USE");
}

/// The namespace is shared with tables, and a name that resolves to two things
/// is a wrong answer waiting to happen.
#[test]
fn a_view_cannot_shadow_a_table_or_be_shadowed_by_one() {
    let d = Scratch::new("view-shadow");
    let dir = d.s();
    run(&dir, "CREATE TABLE t (id UInt64, v Int64, s String) ENGINE = MergeTree ORDER BY id").ok();
    run(&dir, "CREATE VIEW t AS SELECT 1").fails_with("is a table");
    run(&dir, "CREATE VIEW w AS SELECT id FROM t").ok();
    run(&dir, "CREATE TABLE w (id UInt64) ENGINE = MergeTree ORDER BY id")
        .fails_with("is a view");
    // Replacing one on purpose is allowed, and takes effect immediately.
    run(&dir, "CREATE OR REPLACE VIEW w AS SELECT 42 AS id").ok();
    assert_eq!(cell(&dir, "SELECT id FROM w"), "42");
    run(&dir, "DROP VIEW w").ok();
    run(&dir, "SELECT * FROM w").fails_with("does not exist");
    run(&dir, "DROP VIEW IF EXISTS w").ok();
}

// ============================================================= TASK B: RENAME

/// A rename keeps every row, moves the constraints with the table, and leaves
/// no directory behind for the old name.
#[test]
fn rename_moves_the_table_its_rows_and_its_constraints() {
    let d = Scratch::new("rename");
    let dir = d.s();
    run(
        &dir,
        "CREATE TABLE a (id UInt64 UNIQUE, v Int64 CHECK (v >= 0), s String) \
         ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
    )
    .ok();
    fill(&dir, "a", 4000);

    run(&dir, "RENAME TABLE a TO b").ok();
    assert_eq!(cell(&dir, "SELECT count() FROM b"), "4000");
    assert_eq!(cell(&dir, "SELECT sum(v) FROM b"), "7998000");
    run(&dir, "SELECT count() FROM a").fails_with("does not exist");
    assert_eq!(rows(&dir, "SHOW TABLES").iter().filter(|r| *r == "a").count(), 0);
    assert!(!Path::new(&dir).join("default").join("a").exists(), "the old directory is still there");

    // Both constraints came across, in a new process.
    run(&dir, "INSERT INTO b VALUES (5000, -1, 'x')").fails_with("CHECK constraint");
    run(&dir, "INSERT INTO b VALUES (1, 1, 'dup')").fails_with("would be shared");
    run(&dir, "INSERT INTO b VALUES (5000, 1, 'ok')").ok();
    assert_eq!(cell(&dir, "SELECT count() FROM b"), "4001");

    // The obvious refusals.
    run(&dir, "CREATE TABLE c (id UInt64) ENGINE = MergeTree ORDER BY id").ok();
    run(&dir, "RENAME TABLE b TO c").fails_with("already exists");
    run(&dir, "RENAME TABLE nosuch TO d").fails_with("does not exist");
    // And the ALTER spelling of the same statement.
    run(&dir, "ALTER TABLE c RENAME TO e").ok();
    assert!(rows(&dir, "SHOW TABLES").iter().any(|r| r == "e"));
}

// ====================================================== TASK B: MODIFY COLUMN

/// A widening rewrite works and survives a reopen; a narrowing one that cannot
/// hold a stored value refuses the *whole* statement and says which row.
#[test]
fn modify_column_converts_or_refuses_naming_the_row() {
    let d = Scratch::new("modify");
    let dir = d.s();
    run(&dir, "CREATE TABLE t (id UInt64, v Int64, s String) ENGINE = MergeTree ORDER BY id").ok();
    fill(&dir, "t", 500);

    run(&dir, "ALTER TABLE t MODIFY COLUMN v Int32").ok();
    assert_eq!(rows(&dir, "DESCRIBE t")[1], "v|Int32");
    assert_eq!(cell(&dir, "SELECT sum(v) FROM t"), "124750");

    // 499 does not fit in an Int8, and the message says so about that row.
    let e = run(&dir, "ALTER TABLE t MODIFY COLUMN v Int8").fails_with("row 128").err;
    assert!(e.contains("out of range"), "{e}");
    // Refused whole: the type and every row are exactly as they were, in this
    // process and the next.
    assert_eq!(rows(&dir, "DESCRIBE t")[1], "v|Int32");
    assert_eq!(cell(&dir, "SELECT sum(v) FROM t"), "124750");
    assert_eq!(cell(&dir, "SELECT count() FROM t"), "500");

    // Lossy is refused even where a CAST would happily truncate: a schema
    // change has no expression to blame it on and no way back.
    run(&dir, "CREATE TABLE f (id UInt64, x Float64) ENGINE = MergeTree ORDER BY id").ok();
    run(&dir, "INSERT INTO f VALUES (1, 1.5), (2, 2.0)").ok();
    run(&dir, "ALTER TABLE f MODIFY COLUMN x Int64").fails_with("cannot represent exactly");
    assert_eq!(rows(&dir, "DESCRIBE f")[1], "x|Float64");

    // A NULL cannot be stored in a type that has none, and that names the row too.
    run(&dir, "CREATE TABLE n (id UInt64, x Nullable(Int64)) ENGINE = MergeTree ORDER BY id").ok();
    run(&dir, "INSERT INTO n VALUES (1, 5), (2, NULL)").ok();
    run(&dir, "ALTER TABLE n MODIFY COLUMN x Int64").fails_with("row 1 is NULL");

    // A key column is refused rather than silently rebuilt: the parts on disk
    // are sorted and indexed by its current lane.
    run(&dir, "ALTER TABLE t MODIFY COLUMN id Int64").fails_with("key");
}

/// `MODIFY COLUMN` is a rewrite, and a rewrite that half-happened is
/// unrecoverable. Kill one in the middle, at a swept point, and the table must
/// come back either fully old or fully new -- with every row either way.
#[test]
fn a_kill_during_modify_column_leaves_the_table_fully_old_or_fully_new() {
    let d = Scratch::new("modify-kill");
    let dir = d.s();
    run(&dir, "CREATE TABLE t (id UInt64, v Int64, s String) ENGINE = MergeTree ORDER BY id").ok();
    fill(&dir, "t", 60_000);
    let rows_before = cell(&dir, "SELECT count() FROM t");
    let sum_before = cell(&dir, "SELECT sum(v) FROM t");

    // Calibrate: how long does the statement take when nothing kills it?
    let t0 = std::time::Instant::now();
    run(&dir, "ALTER TABLE t MODIFY COLUMN v Int64").ok(); // a no-op retype, same cost shape
    let base = t0.elapsed().max(Duration::from_millis(4));

    let mut old = 0;
    let mut new = 0;
    for (i, frac) in [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95, 1.05].iter().enumerate() {
        let want = if i % 2 == 0 { "Int32" } else { "Int64" };
        let mut child = Command::new(BIN)
            .args(["--data", &dir, "-q", &format!("ALTER TABLE t MODIFY COLUMN v {want}")])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn");
        std::thread::sleep(base.mul_f64(*frac));
        let _ = child.kill();
        let _ = child.wait();

        // Whatever survived has to be a whole table: every row, the right sum,
        // and one of the two types -- never a mixture, and never unreadable.
        let ty = rows(&dir, "DESCRIBE t")[1].clone();
        assert!(ty == "v|Int32" || ty == "v|Int64", "half-migrated column: {ty}");
        if ty.ends_with("Int32") {
            new += 1;
        } else {
            old += 1;
        }
        assert_eq!(cell(&dir, "SELECT count() FROM t"), rows_before, "rows lost at frac {frac}");
        assert_eq!(cell(&dir, "SELECT sum(v) FROM t"), sum_before, "values lost at frac {frac}");
        // Whatever it is now, it still takes writes and still reads back.
        run(&dir, "INSERT INTO t VALUES (999999, 1, 'z')").ok();
        run(&dir, "DELETE FROM t WHERE id = 999999").ok();
    }
    eprintln!("  MODIFY COLUMN killed at 8 points: {old} came back old, {new} came back new");
}

/// The same question for `RENAME`: one name or the other, with all the rows,
/// and never a name that resolves to a directory that is not there.
#[test]
fn a_kill_during_rename_leaves_one_name_or_the_other() {
    let d = Scratch::new("rename-kill");
    let dir = d.s();
    // With a constraint on it, because the rename publishes the metadata and
    // the table separately: whichever name survives the kill has to still be
    // enforcing, or a crash would be a way to quietly drop a CHECK.
    run(
        &dir,
        "CREATE TABLE a (id UInt64, v Int64 CHECK (v >= 0), s String) \
         ENGINE = MergeTree ORDER BY id",
    )
    .ok();
    fill(&dir, "a", 40_000);
    let sum = cell(&dir, "SELECT sum(v) FROM a");

    let t0 = std::time::Instant::now();
    run(&dir, "RENAME TABLE a TO b").ok();
    let base = t0.elapsed().max(Duration::from_millis(3));
    run(&dir, "RENAME TABLE b TO a").ok();

    let mut here = 0;
    let mut there = 0;
    for frac in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.1] {
        let (from, to) = if here % 2 == 0 { ("a", "b") } else { ("b", "a") };
        let mut child = Command::new(BIN)
            .args(["--data", &dir, "-q", &format!("RENAME TABLE {from} TO {to}")])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn");
        std::thread::sleep(base.mul_f64(frac));
        let _ = child.kill();
        let _ = child.wait();

        let names = rows(&dir, "SHOW TABLES");
        let has_from = names.iter().any(|n| n == from);
        let has_to = names.iter().any(|n| n == to);
        assert!(has_from ^ has_to, "rename at {frac} left `{names:?}`");
        let live = if has_to { to } else { from };
        assert_eq!(cell(&dir, &format!("SELECT sum(v) FROM {live}")), sum, "rows lost at {frac}");
        run(&dir, &format!("INSERT INTO {live} VALUES (99999, -1, 'x')"))
            .fails_with("CHECK constraint");
        if has_to {
            there += 1;
        } else {
            here += 1;
        }
        // Put it back under `a` for the next trial.
        if live != "a" {
            run(&dir, &format!("RENAME TABLE {live} TO a")).ok();
        }
    }
    eprintln!("  RENAME killed at 7 points: {here} kept the old name, {there} took the new one");
}

// ======================================================= the metadata itself

/// The catalog's own table is readable and not writable: enforcement reads the
/// engine's copy in memory, so a hand-written row would be a table whose
/// contents and whose behaviour disagree until the next restart.
#[test]
fn the_metadata_table_is_readable_but_not_writable() {
    let d = Scratch::new("meta");
    let dir = d.s();
    run(
        &dir,
        "CREATE TABLE k (id UInt64, v Int64 CHECK (v > 0)) ENGINE = MergeTree ORDER BY id",
    )
    .ok();
    // Readable, and it says what it holds.
    let r = rows(&dir, "SELECT kind, object, name FROM _granular_ddl");
    assert_eq!(r, ["CHECK|k|check_v"], "the metadata row is not what the engine stored");

    run(&dir, "INSERT INTO _granular_ddl VALUES ('CHECK','k','x','','1 = 1')")
        .fails_with("catalog's own table");
    run(&dir, "DROP TABLE _granular_ddl").fails_with("catalog's own table");
    run(&dir, "ALTER TABLE _granular_ddl MODIFY COLUMN sql UInt64").fails_with("catalog's own table");
    // Still enforcing after all that.
    run(&dir, "INSERT INTO k VALUES (1, -1)").fails_with("CHECK constraint");
}

/// Dropping a table takes its constraints with it, so a table re-created under
/// the same name is not silently governed by the old one's rules.
#[test]
fn a_re_created_table_does_not_inherit_the_old_ones_constraints() {
    let d = Scratch::new("recreate");
    let dir = d.s();
    run(&dir, "CREATE TABLE k (id UInt64, v Int64 CHECK (v > 0)) ENGINE = MergeTree ORDER BY id")
        .ok();
    run(&dir, "INSERT INTO k VALUES (1, -1)").fails_with("CHECK constraint");
    run(&dir, "DROP TABLE k").ok();
    run(&dir, "CREATE TABLE k (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id").ok();
    // Same name, no constraint -- in this process and after a reopen.
    run(&dir, "INSERT INTO k VALUES (1, -1)").ok();
    assert_eq!(cell(&dir, "SELECT v FROM k"), "-1");
    run(&dir, "INSERT INTO k VALUES (2, -2)").ok();
    assert_eq!(cell(&dir, "SELECT count() FROM k"), "2");
    // And the metadata table goes away with the last constraint in the database.
    assert!(!rows(&dir, "SHOW TABLES").iter().any(|r| r == "_granular_ddl"));
}

/// The inversion of `a_bulk_import_is_refused_while_the_database_has
/// _constraints`, which is what this test used to be called.
///
/// The CSV bulk import was the one write path that did not go through
/// `run_insert`: `io::emit` handed blocks straight to the catalog, so it
/// enforced nothing, and the whole feature was therefore refused on any
/// database holding a single constraint -- including imports into tables that
/// had none. `emit` calls `Session::import_block` now, so the constraint is
/// enforced where it belongs and the database-wide ban is gone.
#[test]
fn a_bulk_import_enforces_check_constraints() {
    let d = Scratch::new("import");
    let dir = d.s();
    let csv = format!("{dir}/rows.csv");
    std::fs::write(&csv, "id,v,s\n1,1,a\n2,-9,b\n").unwrap();
    // The same rows without the column `k` does not have.
    let bad = format!("{dir}/bad.csv");
    std::fs::write(&bad, "id,v\n1,1\n2,-9\n").unwrap();
    let good = format!("{dir}/good.csv");
    std::fs::write(&good, "id,v\n3,3\n4,4\n").unwrap();

    run(&dir, "CREATE TABLE plain (id UInt64, v Int64, s String) ENGINE = MergeTree ORDER BY id")
        .ok();
    run(&dir, "CREATE TABLE k (id UInt64, v Int64 CHECK (v > 0)) ENGINE = MergeTree ORDER BY id")
        .ok();

    // The violating row fails the import, by the constraint's own message --
    // not by a blanket "bulk load is not available here".
    run(&dir, &format!("INSERT INTO k FROM INFILE '{bad}' FORMAT CSVWithNames"))
        .fails_with("CHECK constraint");
    assert_eq!(cell(&dir, "SELECT count() FROM k"), "0");

    // A file the constraint accepts loads, into the same constrained table.
    run(&dir, &format!("INSERT INTO k FROM INFILE '{good}' FORMAT CSVWithNames")).ok();
    assert_eq!(cell(&dir, "SELECT count() FROM k"), "2");

    // And the unconstrained table in the same database is importable, which is
    // exactly what the database-wide ban used to take away.
    run(&dir, &format!("INSERT INTO plain FROM INFILE '{csv}' FORMAT CSVWithNames")).ok();
    assert_eq!(cell(&dir, "SELECT count() FROM plain"), "2");
}

/// The other half of the bypass: a `UNIQUE` key was unenforced on the import
/// path, so the ban covered it too. It is a real constraint now, and it holds
/// both within one file and against rows an earlier statement stored.
#[test]
fn a_bulk_import_enforces_unique_keys() {
    let d = Scratch::new("import-unique");
    let dir = d.s();
    let dupe = format!("{dir}/dupe.csv");
    std::fs::write(&dupe, "id,v\n1,a\n2,b\n1,c\n").unwrap();
    let uniq = format!("{dir}/uniq.csv");
    std::fs::write(&uniq, "id,v\n1,a\n2,b\n3,c\n").unwrap();

    run(
        &dir,
        "CREATE TABLE u (id UInt64, v String, UNIQUE (id)) \
         ENGINE = MergeTree PRIMARY KEY id ORDER BY id",
    )
    .ok();
    run(&dir, &format!("INSERT INTO u FROM INFILE '{dupe}' FORMAT CSVWithNames"))
        .fails_with("shared with another row in the same statement");
    assert_eq!(cell(&dir, "SELECT count() FROM u"), "0");

    run(&dir, &format!("INSERT INTO u FROM INFILE '{uniq}' FORMAT CSVWithNames")).ok();
    assert_eq!(cell(&dir, "SELECT count() FROM u"), "3");
    // Against rows already stored, in a second process, so it is the key index
    // answering and not session state.
    run(&dir, &format!("INSERT INTO u FROM INFILE '{uniq}' FORMAT CSVWithNames"))
        .fails_with("shared with a row already stored");
    assert_eq!(cell(&dir, "SELECT count() FROM u"), "3");
}

/// A view stores a query, so it has no rows to write to -- and the error says
/// that rather than "table does not exist" about a name that plainly does.
#[test]
fn inserting_into_a_view_is_refused_by_name() {
    let d = Scratch::new("view-insert");
    let dir = d.s();
    run(&dir, "CREATE TABLE t (id UInt64, v Int64, s String) ENGINE = MergeTree ORDER BY id").ok();
    run(&dir, "CREATE VIEW v AS SELECT id FROM t").ok();
    run(&dir, "INSERT INTO v VALUES (1)").fails_with("it is a view");
}

/// A schema change that would leave a constraint dangling is refused, rather
/// than leaving a CHECK that fails every write with a message about binding.
#[test]
fn a_column_a_check_depends_on_cannot_be_dropped() {
    let d = Scratch::new("drop-col");
    let dir = d.s();
    run(
        &dir,
        "CREATE TABLE k (id UInt64, v Int64, w Int64, CHECK (v > 0)) \
         ENGINE = MergeTree ORDER BY id",
    )
    .ok();
    run(&dir, "INSERT INTO k VALUES (1, 1, 1)").ok();
    run(&dir, "ALTER TABLE k DROP COLUMN v").fails_with("no longer bind");
    // A column nothing depends on still goes.
    run(&dir, "ALTER TABLE k DROP COLUMN w").ok();
    assert_eq!(rows(&dir, "DESCRIBE k").len(), 2);
    run(&dir, "INSERT INTO k VALUES (2, -1)").fails_with("CHECK constraint");
}
