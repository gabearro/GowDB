//! W4: who owns a transaction, what a connection handle costs to make, and
//! what an operator can see and set about durability.
//!
//! Every test here drives the shipped facade (`Db`/`Session`) or the
//! `granular` binary. The engine already ruled the "commit somebody else's
//! transaction" failure a bug at the nested-`BEGIN` boundary and closed it
//! there; these are the same bug at the connection boundary.

use std::path::{Path, PathBuf};
use std::process::Command;

use granular::{Db, Session, Value};

// ------------------------------------------------------------------ harness

fn dir(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("granular-w4-{name}"));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).expect("create test data dir");
    p
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        Scratch(dir(name))
    }
    fn path(&self) -> &Path {
        &self.0
    }
    fn s(&self) -> String {
        self.0.display().to_string()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn bin() -> PathBuf {
    let mut p = std::env::current_exe().expect("test exe");
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("granular")
}

/// Run the shipped binary. Returns (status code, stdout, stderr).
fn run(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(bin()).args(args).output().expect("run granular");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn q(dir: &str, sql: &str) -> (i32, String, String) {
    run(&["--data", dir, "-q", sql, "--format", "tsv", "--no-header"])
}

const DDL: &str = "CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id";

fn ids(db: &Db) -> Vec<u64> {
    db.reader()
        .query("SELECT id FROM t ORDER BY id")
        .expect("read")
        .to_values()
        .iter()
        .map(|r| match r[0] {
            Value::UInt(n) => n,
            ref v => panic!("not a uint: {v:?}"),
        })
        .collect()
}

// ------------------------------------------------------------- (a) ownership

/// A second connection's `COMMIT` must not durably publish a transaction it
/// never opened.
///
/// Connection A leaves an uncommitted transaction open between writer guards
/// -- the wire-server shape, which is a supported and separately tested thing
/// to do. Connection B then runs a plain autocommit `INSERT` and a `COMMIT`.
/// B's `COMMIT` used to return `Ok(())` and publish A's row.
#[test]
fn a_second_connection_cannot_commit_a_transaction_it_did_not_open() {
    let a = Db::in_memory();
    let b = a.clone(); // a second connection to the same database
    a.execute(DDL).unwrap();

    a.writer().execute("BEGIN").unwrap();
    a.writer().execute("INSERT INTO t VALUES (111)").unwrap();
    assert!(a.writer().in_transaction(), "the transaction must survive the guard drop");

    let e = b
        .writer()
        .commit()
        .expect_err("B committed a transaction it never opened");
    assert!(
        e.to_string().contains("another connection"),
        "the refusal must name whose transaction it is: {e}"
    );

    // A's row is still uncommitted, and A can still finish its own work.
    a.writer().rollback().unwrap();
    b.execute("INSERT INTO t VALUES (222)").unwrap();
    assert_eq!(ids(&a), vec![222], "A's uncommitted row must not be durable");
}

/// A second connection's ordinary statement must not silently *enlist* in a
/// transaction it did not open.
///
/// This is the half underneath the other two: one shared `Session` holds one
/// transaction, so B's autocommit `INSERT` used to land in A's overlay. B was
/// told `Ok` -- and A's `ROLLBACK` then erased the row, or A's `COMMIT`
/// published it at a boundary B never chose. Both directions are wrong, so
/// the statement is refused rather than given either.
#[test]
fn a_second_connection_cannot_write_into_a_transaction_it_did_not_open() {
    let a = Db::in_memory();
    let b = a.clone();
    a.execute(DDL).unwrap();

    a.writer().execute("BEGIN").unwrap();
    a.writer().execute("INSERT INTO t VALUES (111)").unwrap();

    let e = b.execute("INSERT INTO t VALUES (222)").expect_err("B enlisted in A's transaction");
    assert!(e.to_string().contains("another connection"), "{e}");
    // And a read is refused for the same reason `Reader` refuses one: it
    // would be a dirty read of A's overlay.
    assert!(b.writer().query("SELECT count() FROM t").is_err());
    // B's own BEGIN must not poison A's transaction on its way out.
    assert!(b.writer().begin().is_err());

    a.writer().execute("COMMIT").unwrap();
    assert_eq!(ids(&a), vec![111], "A's COMMIT is still A's to make");
    b.execute("INSERT INTO t VALUES (222)").unwrap();
    assert_eq!(ids(&b), vec![111, 222]);
}

/// A second connection's `ROLLBACK` must not discard its own committed write.
///
/// The mirror image, and the worse half: B was told `Ok` twice -- once for an
/// `INSERT` it believed was autocommitting, once for a `ROLLBACK` -- and the
/// row was gone.
#[test]
fn a_second_connection_cannot_roll_back_a_transaction_it_did_not_open() {
    let a = Db::in_memory();
    let b = a.clone();
    a.execute(DDL).unwrap();
    a.execute("INSERT INTO t VALUES (1)").unwrap();
    // B's own write, autocommitted and acknowledged before A opens anything.
    b.execute("INSERT INTO t VALUES (444)").unwrap();

    a.writer().execute("BEGIN").unwrap();
    a.writer().execute("INSERT INTO t VALUES (333)").unwrap();

    let e = b.writer().rollback().expect_err("B rolled back a transaction it never opened");
    assert!(e.to_string().contains("another connection"), "{e}");

    a.writer().rollback().unwrap();
    assert_eq!(ids(&a), vec![1, 444], "B's own row must survive its refused ROLLBACK");
}

/// The owner of a transaction can still drive it across separate guards.
///
/// The refusal above is about *identity*, not about parking a transaction
/// between acquisitions: `db.writer().execute("BEGIN")` ... later ...
/// `db.writer().execute("COMMIT")` on the same handle is the wire-server
/// shape and must keep working.
#[test]
fn the_connection_that_opened_a_transaction_can_still_commit_it() {
    let db = Db::in_memory();
    db.execute(DDL).unwrap();
    db.writer().execute("BEGIN").unwrap();
    db.writer().execute("INSERT INTO t VALUES (7)").unwrap();
    db.writer().execute("COMMIT").unwrap();
    assert_eq!(ids(&db), vec![7]);
}

/// A bare `Session` -- the CLI's whole world -- is one connection by
/// construction, so nothing about it changes.
#[test]
fn a_bare_session_is_unaffected_by_the_owner_token() {
    let mut s = Session::in_memory();
    s.execute(DDL).unwrap();
    s.execute("BEGIN").unwrap();
    s.execute("INSERT INTO t VALUES (5)").unwrap();
    s.execute("COMMIT").unwrap();
    assert_eq!(s.query("SELECT count() FROM t").unwrap().scalar().unwrap().to_string(), "1");

    // And through the binary, end to end.
    let d = Scratch::new("bare-session");
    let (c, _, err) = q(&d.s(), "CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id");
    assert_eq!(c, 0, "{err}");
    let (c, _, err) = q(&d.s(), "BEGIN; INSERT INTO t VALUES (1),(2); COMMIT");
    assert_eq!(c, 0, "{err}");
    let (c, out, err) = q(&d.s(), "SELECT count() FROM t");
    assert_eq!((c, out.trim()), (0, "2"), "{err}");
}

/// A thread that panics holding the writer must not leave a transaction
/// another connection could adopt -- and must not leave one at all when the
/// panic unwinds.
///
/// The shipped binary is `panic = "abort"`, so this face only bites an
/// embedder built with unwind; the two above need no panic at all. Both are
/// closed by the same token, and this one additionally by `Writer`'s `Drop`.
#[test]
fn a_panicking_writer_does_not_leave_an_adoptable_transaction() {
    let a = Db::in_memory();
    let b = a.clone();
    a.execute(DDL).unwrap();
    a.execute("INSERT INTO t VALUES (1)").unwrap();

    let a2 = a.clone();
    let died = std::thread::spawn(move || {
        let mut w = a2.writer();
        w.begin().unwrap();
        w.execute("INSERT INTO t VALUES (555)").unwrap();
        panic!("thread A dies holding the writer");
    })
    .join();
    assert!(died.is_err(), "the thread was supposed to panic");

    // Unwinding through `Writer` rolls the transaction back, so there is
    // nothing left to adopt and B's COMMIT is a plain "no transaction" error.
    let e = b.writer().commit().expect_err("B committed a dead thread's transaction");
    let msg = e.to_string();
    assert!(
        msg.contains("without an open transaction") || msg.contains("another connection"),
        "{msg}"
    );
    b.execute("INSERT INTO t VALUES (2)").unwrap();
    assert_eq!(ids(&b), vec![1, 2], "the panicked thread's row must not be durable");
}

/// A connection that opens a transaction and is then simply *dropped* -- a
/// client disconnect, a cancelled task, no panic anywhere -- must not wedge
/// the database for every other connection forever.
///
/// This is the panic orphan's far more common face, and the owner token made
/// it strictly worse before this test existed: owners are monotonic, so once
/// the owning `Db` was gone nothing in the process could ever produce that id
/// again, and `COMMIT`, `ROLLBACK`, `BEGIN`, every read and every write from
/// every connection -- including the original -- were refused permanently.
#[test]
fn a_dropped_connection_does_not_wedge_the_database() {
    let a = Db::in_memory();
    a.execute(DDL).unwrap();
    a.execute("INSERT INTO t VALUES (1)").unwrap();

    let b = a.clone();
    b.execute("BEGIN").unwrap();
    b.execute("INSERT INTO t VALUES (555)").unwrap();
    drop(b); // the client hangs up

    // The abandoned transaction is gone, not adopted: B's row is not durable.
    assert!(!a.writer().in_transaction(), "the abandoned transaction is still open");
    a.execute("INSERT INTO t VALUES (2)").unwrap();
    assert_eq!(ids(&a), vec![1, 2], "the abandoned connection's row must not be durable");

    // And the database still works for a transaction opened after it.
    let c = a.clone();
    c.execute("BEGIN").unwrap();
    c.execute("INSERT INTO t VALUES (3)").unwrap();
    c.execute("COMMIT").unwrap();
    assert_eq!(ids(&a), vec![1, 2, 3]);
}

/// The same on a real directory, where rolling the orphan back also has to
/// rewind the write-ahead log -- and where a second process is the one that
/// proves nothing was published.
#[test]
fn a_dropped_connection_leaves_no_trace_on_disk() {
    let d = Scratch::new("orphan-on-disk");
    let a = Db::open(d.path()).unwrap();
    a.execute(DDL).unwrap();
    a.execute("INSERT INTO t VALUES (1)").unwrap();

    let b = a.clone();
    b.execute("BEGIN").unwrap();
    b.execute("INSERT INTO t VALUES (555)").unwrap();
    b.execute("INSERT INTO t VALUES (556)").unwrap();
    drop(b);

    a.execute("INSERT INTO t VALUES (2)").unwrap();
    a.writer().checkpoint().unwrap();
    drop(a);

    let (c, out, err) = q(&d.s(), "SELECT id FROM t ORDER BY id");
    assert_eq!(c, 0, "{err}");
    assert_eq!(out.split_whitespace().collect::<Vec<_>>(), vec!["1", "2"], "{err}");
}

/// A connection dropped *without* a transaction of its own must not disturb
/// one another connection is legitimately holding open across guards -- the
/// wire-server shape. The orphan sweep keys on the owner, not on the fact
/// that some connection somewhere went away.
#[test]
fn dropping_an_idle_connection_leaves_another_connections_transaction_alone() {
    let a = Db::in_memory();
    a.execute(DDL).unwrap();
    a.writer().execute("BEGIN").unwrap();
    a.writer().execute("INSERT INTO t VALUES (7)").unwrap();

    let idle = a.clone();
    drop(idle);

    assert!(a.writer().in_transaction(), "an unrelated drop ended A's transaction");
    a.writer().commit().unwrap();
    assert_eq!(ids(&a), vec![7]);
}

// --------------------------------------------------------- (b) cheap liveness

/// Constructing a read handle must not take the session lock.
///
/// It used to, only to copy `Limits` out; a pool opening a connection stalled
/// for the whole length of an unrelated `INSERT`, and taking one on a thread
/// that already held the writer deadlocked outright with no query in sight.
#[test]
fn a_reader_handle_can_be_taken_while_this_thread_holds_the_writer() {
    let db = Db::in_memory();
    db.execute(DDL).unwrap();
    let w = db.writer();
    // No query -- just construction. This is the call that never returned.
    let r = db.reader();
    drop(w);
    assert_eq!(r.query("SELECT count() FROM t").unwrap().scalar().unwrap().to_string(), "0");
}

/// The same, from inside a transaction that holds the guard for its whole
/// length: `Db::transaction` is the documented way to write, and a pool that
/// hands out connections underneath one must not block.
#[test]
fn a_reader_handle_can_be_taken_from_inside_a_transaction() {
    let db = Db::in_memory();
    db.execute(DDL).unwrap();
    let taken = db
        .transaction(|s| {
            s.execute("INSERT INTO t VALUES (1)")?;
            Ok(db.reader())
        })
        .expect("taking a reader inside a transaction must not deadlock");
    assert_eq!(taken.query("SELECT count() FROM t").unwrap().scalar().unwrap().to_string(), "1");
}

// ------------------------------------------------- (c) automatic checkpoints

/// A row wide enough that a few thousand of them make a log worth folding.
const WIDE: &str = "CREATE TABLE t (id UInt64, s String) ENGINE = MergeTree ORDER BY id";

fn wide_rows(from: u64, n: u64) -> String {
    let mut sql = String::from("INSERT INTO t VALUES ");
    for i in from..from + n {
        if i > from {
            sql.push(',');
        }
        sql.push_str(&format!("({i},'padpadpadpadpadpadpadpadpadpad-{i}')"));
    }
    sql
}

fn wal_bytes(d: &Path) -> u64 {
    std::fs::metadata(d.join("default").join("t").join("wal.log")).map_or(0, |m| m.len())
}

fn part_files(d: &Path) -> usize {
    std::fs::read_dir(d.join("default").join("t"))
        .map(|rd| {
            rd.flatten().filter(|e| e.file_name().to_string_lossy().starts_with("part_")).count()
        })
        .unwrap_or(0)
}

/// Nothing auto-checkpointed: a writer that only ran `INSERT`s grew `wal.log`
/// without bound and wrote no part file at all, so disk-full arrived far ahead
/// of the data volume and recovery had the whole history to replay.
///
/// Driven through `Session` on a real directory and deliberately *never*
/// calling `checkpoint`, which is the shape the defect needs -- the CLI's exit
/// checkpoint would hide it.
#[test]
fn a_long_running_writer_folds_its_log_without_being_told() {
    let d = Scratch::new("fold-on");
    let (peak, parts) = {
        let mut s = Session::open(d.path()).unwrap();
        s.execute(WIDE).unwrap();
        s.execute("SET wal_fold_bytes = '256K'").unwrap();
        let mut peak = 0;
        for b in 0..12u64 {
            s.execute(&wide_rows(b * 2000, 2000)).unwrap();
            peak = peak.max(wal_bytes(d.path()));
        }
        assert_eq!(
            s.query("SELECT count() FROM t").unwrap().scalar().unwrap().to_string(),
            "24000"
        );
        (peak, part_files(d.path()))
    };
    // The log never ran away, and parts were published along the way.
    assert!(peak < 700_000, "the log peaked at {peak} bytes: it is not being folded");
    assert!(parts > 1, "no part was written by the fold ({parts} files)");

    // And the rows are all there in a second process, from the parts the fold
    // wrote plus whatever tail is left in the log.
    let (c, out, err) = q(&d.s(), "SELECT count(), sum(id) FROM t");
    assert_eq!((c, out.trim()), (0, "24000\t287988000"), "{err}");
}

/// `wal_fold_bytes = 0` is exactly what the engine did before: no automatic
/// fold at all. Kept reachable, and pinned, because a workload that
/// checkpoints on its own schedule should be able to say so.
#[test]
fn wal_fold_bytes_zero_keeps_the_old_behaviour() {
    let d = Scratch::new("fold-off");
    let mut s = Session::open(d.path()).unwrap();
    s.execute(WIDE).unwrap();
    s.execute("SET wal_fold_bytes = 0").unwrap();
    for b in 0..12u64 {
        s.execute(&wide_rows(b * 2000, 2000)).unwrap();
    }
    assert!(
        wal_bytes(d.path()) > 1 << 20,
        "with the fold disabled the log must grow: {} bytes",
        wal_bytes(d.path())
    );
    assert_eq!(part_files(d.path()), 0, "0 must mean no automatic checkpoint at all");
}

/// The same fold, on a table declared `UNIQUE`.
///
/// That path stages its record and writes the commit marker by hand instead of
/// routing through `log_insert`, so it never consulted `wal_fold_bytes` at all
/// and the log grew without bound however small the threshold was set. It is
/// the shape that matters most for this: a `UNIQUE` key is what an OLTP writer
/// declares, and an OLTP writer is exactly the process that stays open for a
/// week inserting one row at a time.
#[test]
fn a_unique_keyed_table_folds_its_log_too() {
    let d = Scratch::new("fold-unique");
    let (peak, parts) = {
        let mut s = Session::open(d.path()).unwrap();
        s.execute(
            "CREATE TABLE t (id UInt64, s String, UNIQUE (id)) \
             ENGINE = MergeTree PRIMARY KEY id ORDER BY id",
        )
        .unwrap();
        s.execute("SET wal_fold_bytes = '256K'").unwrap();
        let mut peak = 0;
        for b in 0..12u64 {
            s.execute(&wide_rows(b * 2000, 2000)).unwrap();
            peak = peak.max(wal_bytes(d.path()));
        }
        assert_eq!(
            s.query("SELECT count() FROM t").unwrap().scalar().unwrap().to_string(),
            "24000"
        );
        (peak, part_files(d.path()))
    };
    assert!(peak < 700_000, "the log peaked at {peak} bytes: it is not being folded");
    assert!(parts > 1, "no part was written by the fold ({parts} files)");

    // And the constraint still holds after the fold, in a second process: the
    // key index is rebuilt from the parts the fold wrote, not from the log.
    let (c, out, err) = q(&d.s(), "SELECT count(), sum(id) FROM t");
    assert_eq!((c, out.trim()), (0, "24000\t287988000"), "{err}");
    let (c, _, err) = q(&d.s(), "INSERT INTO t VALUES (7,'again')");
    assert_ne!(c, 0, "the UNIQUE key must survive its own fold");
    assert!(err.contains("shared with a row already stored"), "{err}");
}

/// The threshold is a real setting: it round-trips through `SHOW SETTINGS`,
/// and `SET` refuses to invent a value for it.
#[test]
fn wal_fold_bytes_is_a_documented_setting() {
    let d = Scratch::new("fold-setting");
    let (c, out, err) = q(&d.s(), "SHOW SETTINGS LIKE 'wal_fold_bytes'");
    assert_eq!(c, 0, "{err}");
    let f: Vec<&str> = out.trim().split('\t').collect();
    assert_eq!((f[0], f[1], f[2], f[3]), ("wal_fold_bytes", "64M", "64M", "bytes"), "{out}");
    let (c, out, err) = q(&d.s(), "SET wal_fold_bytes = '8M'; SHOW SETTINGS LIKE 'wal_fold_bytes'");
    assert_eq!(c, 0, "{err}");
    assert!(out.contains("8M"), "{out}");
}

// --------------------------------------------------- (d) archive retention

/// `SET wal_archive_retention` was documented in `wal.rs` and did not exist;
/// `set_archive_retention` had zero non-test callers.
///
/// Asserted against the *mechanism* rather than the registry: the value has to
/// land in the process-global the archiver actually trims by, or this is a
/// setting that reports itself and changes nothing.
#[test]
fn set_wal_archive_retention_reaches_the_archive() {
    let before = granular::persist::wal::archive_retention();
    let mut s = Session::in_memory();
    s.execute("SET wal_archive_retention = '128M'").unwrap();
    assert_eq!(granular::persist::wal::archive_retention(), 128 << 20);
    s.execute("SET wal_archive_retention = 0").unwrap();
    assert_eq!(granular::persist::wal::archive_retention(), 0, "0 keeps every segment");
    granular::persist::wal::set_archive_retention(before);
}

/// The same name through the binary, which is where an operator types it, and
/// the `default` column must keep telling the truth after a `SET`.
#[test]
fn wal_archive_retention_is_reachable_from_the_cli() {
    let d = Scratch::new("retention-cli");
    let (c, out, err) = q(&d.s(), "SET wal_archive_retention = '128M'; SHOW SETTINGS LIKE 'wal_archive%'");
    assert_eq!(c, 0, "{err}");
    let f: Vec<&str> = out.trim().split('\t').collect();
    assert_eq!(
        (f[0], f[1], f[2]),
        ("wal_archive_retention", "128M", "64M"),
        "value must move and default must not: {out}"
    );
}

/// A statement-scoped `SETTINGS` clause must not outlive its statement -- and
/// this is the setting for which that mattered most, because the next archive
/// tick acts on it and pruned segments do not come back.
///
/// It was the one entry in the registry with no `Settings` field: `set` wrote
/// the process-wide static directly, so `saved.apply_to` had nothing to put
/// back. `SELECT 1 SETTINGS wal_archive_retention='1'` on a read query then
/// trimmed point-in-time recovery to the newest segment and left every later
/// statement in the process at that value.
#[test]
fn a_scoped_wal_archive_retention_does_not_outlive_its_statement() {
    let d = Scratch::new("retention-scope");
    let (c, out, err) = q(
        &d.s(),
        "SELECT 1 SETTINGS wal_archive_retention='1M'; SHOW SETTINGS LIKE 'wal_archive_retention'",
    );
    assert_eq!(c, 0, "{err}");
    let f: Vec<&str> = out.trim().lines().last().unwrap().split('\t').collect();
    assert_eq!((f[1], f[2]), ("64M", "64M"), "the clause stuck to the session: {out}");

    // And the archive it governs is still whole. Five sealed segments, then a
    // read carrying the clause and a write that runs the archive tick.
    let (c, _, err) = q(&d.s(), DDL);
    assert_eq!(c, 0, "{err}");
    // One process per insert: the exit checkpoint is what seals a segment.
    for i in 1..=5 {
        let (c, _, err) = q(&d.s(), &format!("INSERT INTO t VALUES ({i})"));
        assert_eq!(c, 0, "{err}");
    }
    let segs = |d: &str| -> u64 {
        let (c, out, err) = q(d, "SELECT segments FROM system.wal");
        assert_eq!(c, 0, "{err}");
        out.trim().parse().expect("segment count")
    };
    let before = segs(&d.s());
    assert!(before >= 5, "expected sealed segments to trim, got {before}");
    let (c, _, err) = q(
        &d.s(),
        "SELECT count() FROM t SETTINGS wal_archive_retention='1'; INSERT INTO t VALUES (6)",
    );
    assert_eq!(c, 0, "{err}");
    assert!(
        segs(&d.s()) > before,
        "the scoped clause pruned the archive: {before} segments before, {} after",
        segs(&d.s())
    );

    // A failed `SET` must apply none of its pairs, including this one.
    let (c, _, _) = q(&d.s(), "SET wal_archive_retention='4M', bogus=1");
    assert_ne!(c, 0, "an unknown setting must fail the statement");
    let (c, out, err) = q(&d.s(), "SHOW SETTINGS LIKE 'wal_archive_retention'");
    assert_eq!(c, 0, "{err}");
    assert_eq!(
        out.trim().split('\t').nth(1),
        Some("64M"),
        "a failed SET applied one of its pairs anyway: {out}"
    );
}

// ------------------------------------------------- (e) system.wal visibility

/// An operator could not see WAL bytes per table, the recovery watermark, the
/// segment count, the horizon or the last checkpoint from SQL at all: every
/// function behind them existed and was called only from `backup.rs`.
#[test]
fn system_wal_shows_the_log_an_operator_has_to_reason_about() {
    let d = Scratch::new("system-wal");
    let mut s = Session::open(d.path()).unwrap();
    s.execute(WIDE).unwrap();
    s.execute("SET wal_fold_bytes = 0").unwrap();
    s.execute(&wide_rows(0, 2000)).unwrap();

    let row = |s: &mut Session| -> Vec<String> {
        s.query(
            "SELECT wal_bytes, wal_committed, replay_bytes, last_checkpoint, segments, lags \
             FROM system.wal WHERE table = 't'",
        )
        .unwrap()
        .to_values()
        .remove(0)
        .iter()
        .map(|v| v.to_string())
        .collect()
    };

    let before = row(&mut s);
    let (bytes, replay) = (before[0].parse::<u64>().unwrap(), before[2].parse::<u64>().unwrap());
    assert!(bytes > 10_000, "an unfolded log must show its size, got {bytes}");
    assert_eq!(replay, bytes - before[1].parse::<u64>().unwrap());
    assert!(replay > 0, "everything logged since the last checkpoint has to replay");

    s.checkpoint().unwrap();
    let after = row(&mut s);
    assert_eq!(after[2], "0", "after a checkpoint there is nothing left to replay: {after:?}");
    assert!(after[0].parse::<u64>().unwrap() < bytes, "the log must have been truncated");
    assert_ne!(after[3], "1970-01-01 00:00:00", "last_checkpoint must be a real time");
    // The archive picked the segment up, and the live log no longer lags it.
    assert_eq!(after[4], "1", "the checkpoint archives one segment: {after:?}");
    assert_eq!(after[5], "0", "an emptied log does not lag its archive: {after:?}");
}

/// Reachable by name from the shipped binary, with the columns it promises.
#[test]
fn system_wal_is_reachable_from_the_cli() {
    let d = Scratch::new("system-wal-cli");
    let (c, _, err) = q(&d.s(), WIDE);
    assert_eq!(c, 0, "{err}");
    let (c, out, err) =
        run(&["--data", &d.s(), "-q", "SELECT * FROM system.wal", "--format", "tsv"]);
    assert_eq!(c, 0, "{err}");
    let head = out.lines().next().unwrap_or_default();
    for col in [
        "database", "table", "wal_bytes", "wal_committed", "replay_bytes", "last_checkpoint",
        "segments", "archive_bytes", "archive_first_seq", "archive_last_seq", "horizon_seq",
        "lags",
    ] {
        assert!(head.split('\t').any(|c| c == col), "system.wal has no `{col}`: {head}");
    }
}

// ------------------------------------------------ (f) quarantine, widened

/// Seed a directory with two healthy tables and return it.
fn seed_two(name: &str) -> Scratch {
    let d = Scratch::new(name);
    let mut s = Session::open(d.path()).unwrap();
    s.execute("CREATE TABLE a (id UInt64, s String) ENGINE = MergeTree ORDER BY id").unwrap();
    s.execute("CREATE TABLE b (id UInt64) ENGINE = MergeTree ORDER BY id").unwrap();
    let mut rows = String::from("INSERT INTO a VALUES ");
    for i in 0..3000u64 {
        if i > 0 {
            rows.push(',');
        }
        rows.push_str(&format!("({i},'padpadpadpadpadpadpad-{i}')"));
    }
    s.execute(&rows).unwrap();
    s.execute("INSERT INTO b VALUES (7),(8),(9)").unwrap();
    s.checkpoint().unwrap();
    // Live post-checkpoint records, so `a`'s log has something to replay. Four
    // of them, not one: the replay treats a bad frame at the *end* of a log as
    // a torn write and stops there, which is correct and is not the case under
    // test -- damage in the middle is.
    //
    // No checkpoint after them: the session is dropped exactly as a crashed
    // writer would leave the directory.
    for i in 0..4u64 {
        s.execute(&format!("INSERT INTO a VALUES ({},'tail{i}')", 90_000 + i)).unwrap();
    }
    d
}

fn scribble(p: &Path, off: u64, n: usize) {
    let mut b = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    assert!(b.len() as u64 > off, "{} is only {} bytes", p.display(), b.len());
    for i in off as usize..(off as usize + n).min(b.len()) {
        b[i] ^= 0xFF;
    }
    std::fs::write(p, b).unwrap();
}

/// A `wal.log` that will not replay must quarantine its own table, not refuse
/// the whole database.
///
/// It used to be a bare `?` in the loader, so a bad checksum in a file no
/// other table reads took every healthy table down with it -- including
/// `system.tables`, which is where an operator would look to find out which
/// table it was.
#[test]
fn a_damaged_wal_quarantines_one_table_and_leaves_the_rest() {
    let d = seed_two("quar-wal");
    // Inside the first record's *body*, past its length varint and checksum,
    // so this is damage the reader can prove rather than a torn tail.
    scribble(&d.path().join("default").join("a").join("wal.log"), 24, 4);

    let (c, out, err) = q(&d.s(), "SELECT count() FROM b");
    assert_eq!((c, out.trim()), (0, "3"), "a healthy table must still answer: {err}");

    let (c, out, err) = q(&d.s(), "SELECT name, quarantined FROM system.tables");
    assert_eq!(c, 0, "{err}");
    assert_eq!(out.trim(), "a\t1\nb\t0", "the roster must name the damaged table");

    let (c, _, err) = q(&d.s(), "SELECT count() FROM a");
    assert_eq!(c, 1, "the damaged table must be refused");
    assert!(err.contains("wal.log"), "the refusal must name the file to restore: {err}");
    assert!(err.contains("No other table is affected"), "{err}");
}

/// The same for a `TABLE` file, which falls through to the roster's own copy
/// of the definition -- and which is precisely why `CATALOG` is out of scope:
/// it *is* that roster.
#[test]
fn a_damaged_table_file_quarantines_one_table_and_leaves_the_rest() {
    let d = seed_two("quar-table");
    scribble(&d.path().join("default").join("a").join("TABLE"), 40, 8);

    let (c, out, err) = q(&d.s(), "SELECT count() FROM b");
    assert_eq!((c, out.trim()), (0, "3"), "a healthy table must still answer: {err}");

    let (c, out, err) = q(&d.s(), "SELECT name, quarantined FROM system.tables");
    assert_eq!(c, 0, "{err}");
    assert_eq!(out.trim(), "a\t1\nb\t0");

    let (c, _, err) = q(&d.s(), "SELECT count() FROM a");
    assert_eq!(c, 1);
    assert!(err.contains("TABLE"), "the refusal must name the file: {err}");
}

/// A quarantined table keeps its place in the roster, so a later checkpoint
/// neither rewrites it nor collects its directory as a dropped table.
///
/// This is what makes the quarantine survivable rather than a slow deletion:
/// a table the committed `CATALOG` stops naming is one the *next* checkpoint
/// `remove_dir_all`s.
#[test]
fn a_quarantined_table_survives_the_checkpoints_that_follow_it() {
    let d = seed_two("quar-survive");
    scribble(&d.path().join("default").join("a").join("TABLE"), 40, 8);
    // Three separate processes, each of which checkpoints on the way out.
    for _ in 0..3 {
        let (c, out, err) = q(&d.s(), "INSERT INTO b VALUES (10)");
        assert_eq!(c, 0, "a write to the healthy table must succeed: {out}{err}");
    }
    assert!(d.path().join("default").join("a").join("TABLE").exists(), "`a` was collected");
    let (c, out, err) = q(&d.s(), "SELECT name, quarantined FROM system.tables");
    assert_eq!((c, out.trim()), (0, "a\t1\nb\t0"), "{err}");
    let (c, out, err) = q(&d.s(), "SELECT count() FROM b");
    assert_eq!((c, out.trim()), (0, "6"), "{err}");
}

/// The one damage the engine must *not* degrade around: `_granular_ddl` holds
/// the CHECK and UNIQUE rules the write path enforces, so a database whose
/// copy will not read must refuse to open rather than accept writes under
/// rules it cannot see. Widening the quarantine to logs must route into that
/// refusal, not around it.
#[test]
fn a_damaged_constraint_log_still_refuses_to_open_the_database() {
    let d = Scratch::new("quar-ddl");
    {
        let mut s = Session::open(d.path()).unwrap();
        s.execute(
            "CREATE TABLE t (id UInt64, CONSTRAINT pos CHECK (id > 0)) \
             ENGINE = MergeTree ORDER BY id",
        )
        .unwrap();
        s.execute("INSERT INTO t VALUES (1)").unwrap();
        s.checkpoint().unwrap();
    }
    let ddl = d.path().join("default").join("_granular_ddl");
    assert!(ddl.exists(), "the constraint table was not created");
    // Whichever of the two files carries it, the refusal must be the loud one.
    scribble(&ddl.join("TABLE"), 40, 8);

    let (c, _, err) = q(&d.s(), "SELECT count() FROM t");
    assert_eq!(c, 1, "a database with unreadable constraints must not open");
    assert!(
        err.contains("_granular_ddl") && err.contains("Refusing to open"),
        "the refusal must be the constraint one, not a bare quarantine: {err}"
    );
}

// ------------------------------------------------------------ (g) read-only

/// `--read-only` did not exist, `Session::open_read_only` was reachable only
/// from the library, and the unconditional exit checkpoint would have failed
/// every run that used it.
#[test]
fn read_only_answers_queries_refuses_writes_and_exits_zero() {
    let d = Scratch::new("read-only");
    let (c, _, err) = q(&d.s(), &format!("{DDL}; INSERT INTO t VALUES (1),(2)"));
    assert_eq!(c, 0, "{err}");

    let ro = |sql: &str| {
        run(&["--read-only", "--data", &d.s(), "-q", sql, "--format", "tsv", "--no-header"])
    };
    let (c, out, err) = ro("SELECT count() FROM t");
    assert_eq!((c, out.trim()), (0, "2"), "a read-only run must answer and exit 0: {err}");

    let (c, _, err) = ro("INSERT INTO t VALUES (3)");
    assert_eq!(c, 1, "a read-only session must refuse a write");
    assert!(err.contains("read-only"), "{err}");

    // The shared lock is shared: two of them at once, and a writer alongside.
    let (c, out, _) = ro("SELECT count() FROM t");
    assert_eq!((c, out.trim()), (0, "2"));
    let (c, _, err) = q(&d.s(), "INSERT INTO t VALUES (3)");
    assert_eq!(c, 0, "the directory must still be writable afterwards: {err}");
}

/// A forensic copy on read-only media could not be opened at all: the lock
/// file was always opened `write(true).create(true)`, even for `LOCK_SH`.
///
/// And it has to answer a query that *spills*, which is the half this test
/// originally missed: spill files moved from `env::temp_dir()` to
/// `<data>/.spill`, so on the very media this flag exists for, every read big
/// enough to spill failed with `Permission denied` while a plain `count()`
/// succeeded. A read-only session spills to the temp directory again.
#[test]
#[cfg(unix)]
fn read_only_opens_a_copy_on_read_only_media() {
    let src = Scratch::new("ro-media-src");
    let ddl = "CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id";
    let (c, _, err) = q(&src.s(), &format!("{ddl}; INSERT INTO t VALUES (1),(2),(3)"));
    assert_eq!(c, 0, "{err}");
    // A second table with enough distinct keys that an 8 MiB budget has to
    // spill the GROUP BY.
    let csv = std::env::temp_dir().join("granular-w4-ro-media.csv");
    let mut body = String::from("v\n");
    for i in 0..200_000u64 {
        body.push_str(&format!("{i}\n"));
    }
    std::fs::write(&csv, body).unwrap();
    let (c, _, err) = q(
        &src.s(),
        &format!(
            "CREATE TABLE big (v UInt64) ENGINE = MergeTree ORDER BY v; \
             INSERT INTO big FROM INFILE '{}'",
            csv.display()
        ),
    );
    let _ = std::fs::remove_file(&csv);
    assert_eq!(c, 0, "{err}");

    let copy = std::env::temp_dir().join("granular-w4-ro-media-copy");
    let _ = std::fs::remove_dir_all(&copy);
    let st = Command::new("cp").args(["-R", &src.s(), &copy.display().to_string()]).status().unwrap();
    assert!(st.success());
    Command::new("chmod").args(["-R", "a-w", &copy.display().to_string()]).status().unwrap();

    let ro = |sql: &str| {
        run(&[
            "--read-only",
            "--data",
            &copy.display().to_string(),
            "-q",
            sql,
            "--format",
            "tsv",
            "--no-header",
        ])
    };
    let plain = ro("SELECT count() FROM t");
    let spilling = ro("SET max_memory_usage='8M'; SELECT count() FROM (SELECT v, count() c FROM big GROUP BY v)");
    let left_behind = copy.join(granular::session::SPILL_DIR).exists();
    Command::new("chmod").args(["-R", "u+w", &copy.display().to_string()]).status().unwrap();
    let _ = std::fs::remove_dir_all(&copy);

    assert_eq!((plain.0, plain.1.trim()), (0, "3"), "a read-only copy must open: {}", plain.2);
    assert_eq!(
        (spilling.0, spilling.1.trim()),
        (0, "200000"),
        "a spilling read on read-only media must answer: {}",
        spilling.2
    );
    assert!(!left_behind, "a read-only session wrote a spill directory into the data directory");
}

/// A flag that cannot do anything is a flag that lies. `--read-only` without
/// `--data` is a usage error, and the flag is in `--help`.
#[test]
fn read_only_needs_a_data_directory_and_is_documented() {
    let (c, _, err) = run(&["--read-only", "-q", "SELECT 1"]);
    assert_eq!(c, 2, "{err}");
    assert!(err.contains("--read-only needs --data"), "{err}");

    let (c, out, _) = run(&["--help"]);
    assert_eq!(c, 0);
    assert!(out.contains("--read-only"), "a flag missing from --help is unreachable: {out}");
}
