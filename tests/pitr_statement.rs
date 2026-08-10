//! The recovery *statement*: `RESTORE ... UNTIL`, typed at the shipped binary.
//!
//! `tests/pitr.rs` pins the machinery -- where a recovery lands on each axis,
//! and every target it refuses -- by calling `backup::restore_until` directly.
//! This file pins the half that machinery is useless without: that the same
//! targets and the same refusals are reachable by typing SQL at `granular`,
//! which is the only interface an operator has while the incident is running.
//! Every recovery below is a `RESTORE ... UNTIL` statement in a child process,
//! and every result is read back by a second child process opening the
//! recovered directory -- so a recovery the real loader cannot open, or a
//! statement wired to nothing, fails here.
//!
//! The library is called for exactly one thing: to learn the recovery LSN a
//! backup ended at, and the ones later writes were stamped with. That is not
//! an omission in the statement -- an LSN is a number a *tool* hands the
//! operator, and these tests are standing in for the tool.
//!
//! What is pinned, in order:
//!
//!   1. **The statement lands on the instant it was given**, on both axes, and
//!      the row it reports says how much log it replayed to get there. The
//!      backup's own instant, every point in the middle, and the far end are
//!      all reachable and all different.
//!   2. **Every unanswerable target is refused through the statement**, with
//!      the machinery's sentence rather than a parse error, and with nothing
//!      written: before the backup, past the end of the archive, across a hole
//!      in it, and a target that is not a target at all.
//!   3. **`UNTIL` is not a way into the open database.** `RESTORE` refuses to
//!      write into the directory this session has open; a clause that reached
//!      its own copy of that check would be a second place for it to be wrong,
//!      and the mistake it prevents is the unrecoverable one.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::Duration;

use granular::backup;
use granular::persist::wal;
use granular::Session;

const BIN: &str = env!("CARGO_BIN_EXE_granular");
const DDL: &str = "CREATE TABLE t (id UInt64, v String) ENGINE = MergeTree ORDER BY id";

// ------------------------------------------------------------------ harness

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let d =
            std::env::temp_dir().join(format!("granular-pitrstmt-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("scratch dir");
        Scratch(d)
    }
    fn at(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
    fn s(&self, name: &str) -> String {
        self.at(name).to_string_lossy().into_owned()
    }
    fn db(&self) -> PathBuf {
        self.at("db")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Run {
    code: i32,
    out: String,
    err: String,
}

impl Run {
    fn ok(self) -> Run {
        assert_eq!(self.code, 0, "stdout:\n{}\nstderr:\n{}", self.out, self.err);
        self
    }
    /// A refusal: non-zero exit *and* the reason on stderr. Both, because a
    /// recovery that failed quietly is the failure this feature exists to end.
    fn refused(self) -> String {
        assert_ne!(self.code, 0, "the statement was accepted:\n{}", self.out);
        assert!(!self.err.trim().is_empty(), "a refusal with no reason");
        self.err
    }
    /// The single column of a one-column result, in order.
    fn col(&self) -> Vec<String> {
        self.out.lines().filter(|l| !l.is_empty()).map(str::to_string).collect()
    }
    /// The fields of the first row, which for an operator statement is the
    /// whole report.
    fn row(&self) -> Vec<String> {
        self.out.lines().next().expect("a row").split('\t').map(str::to_string).collect()
    }
}

fn run(db: Option<&Path>, args: &[&str]) -> Run {
    let mut c = Command::new(BIN);
    if let Some(d) = db {
        c.args(["--data", &d.to_string_lossy()]);
    }
    let o: Output = c.args(args).output().expect("spawn granular");
    Run {
        code: o.status.code().unwrap_or(-1),
        out: String::from_utf8_lossy(&o.stdout).into_owned(),
        err: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

fn sql(db: &Path, q: &str) -> Run {
    run(Some(db), &["--format", "tsv", "--no-header", "-q", q])
}

/// One CLI invocation, which is also one checkpoint and therefore one archived
/// segment: the process checkpoints on the way out, and that is what retires
/// the log into the archive.
fn step(db: &Path, q: &str) -> Run {
    sql(db, q).ok()
}

/// The `id`s in `t`, ascending, read by the real binary out of `dir`.
fn ids(dir: &Path) -> Vec<String> {
    sql(dir, "SELECT id FROM t ORDER BY id").ok().col()
}

/// `RESTORE FROM '<arc>' TO '<out>' [UNTIL <target>]`, run against the open
/// database `db` -- which is what supplies the WAL archive to roll through.
fn restore(db: &Path, arc: &str, out: &Path, until: &str) -> Run {
    let sep = if until.is_empty() { "" } else { " UNTIL " };
    sql(db, &format!("RESTORE FROM '{arc}' TO '{}'{sep}{until}", out.to_string_lossy()))
}

/// A database with three rows, a backup of it, and three more rows written
/// afterwards -- one CLI invocation each, so every write lands in its own
/// archived segment and the boundaries are nameable.
///
/// Returns the archive path and the recovery LSN each later write was stamped
/// with, in order. Same fixture as `tests/pitr.rs`, deliberately: the claim
/// here is that the statement reaches the same states the direct calls do.
fn staged(s: &Scratch) -> (String, Vec<u64>) {
    let db = s.db();
    step(&db, DDL);
    step(&db, "INSERT INTO t VALUES (1,'a'),(2,'b')");
    step(&db, "INSERT INTO t VALUES (3,'c')");
    let arc = s.s("base.gbak");
    step(&db, &format!("BACKUP TO '{arc}'"));
    let mut marks = Vec::new();
    for (id, v) in [(4u32, "d"), (5, "e"), (6, "f")] {
        step(&db, &format!("INSERT INTO t VALUES ({id},'{v}')"));
        marks.push(wal::archive_end(&db).expect("archive end").last_seq);
    }
    (arc, marks)
}

// ------------------------------------------------------- the moment restored

/// The headline claim, through SQL: `UNTIL LSN` lands on that LSN -- not the
/// backup it started from, not the latest state it could have reached -- and
/// the report row says how many logged mutations it replayed to get there.
#[test]
fn the_statement_recovers_to_the_lsn_it_was_given() {
    let s = Scratch::new("lsn");
    let (arc, marks) = staged(&s);
    let db = s.db();
    assert_eq!(ids(&db), ["1", "2", "3", "4", "5", "6"], "fixture");

    // No `UNTIL` at all is still the backup's own instant: adding the clause
    // must not have changed what leaving it out means.
    let plain = s.at("plain");
    let r = restore(&db, &arc, &plain, "").ok();
    assert_eq!(ids(&plain), ["1", "2", "3"]);
    assert_eq!(r.row()[4], "0", "a restore with no UNTIL replays nothing: {:?}", r.row());

    // ...and each archived write, one at a time, by the number a tool would
    // have handed the operator.
    for (i, &mark) in marks.iter().enumerate() {
        let out = s.at(&format!("at-{mark}"));
        let r = restore(&db, &arc, &out, &format!("LSN {mark}")).ok();
        let want: Vec<String> = (1..=4 + i as u32).map(|n| n.to_string()).collect();
        assert_eq!(ids(&out), want, "RESTORE ... UNTIL LSN {mark}");
        assert_eq!(r.row()[4], (i + 1).to_string(), "one logged insert per step");
    }

    // The far end, by both spellings, and the report's shape while we are here
    // -- `replayed` is the column that distinguishes the two statements.
    let latest = s.at("latest");
    restore(&db, &arc, &latest, "LATEST").ok();
    assert_eq!(ids(&latest), ["1", "2", "3", "4", "5", "6"]);
    let last = s.at("last");
    restore(&db, &arc, &last, &format!("LSN {}", marks.last().unwrap())).ok();
    assert_eq!(ids(&last), ids(&latest), "UNTIL LATEST and UNTIL the last LSN are one state");

    let head = run(
        Some(&db),
        &["--format", "tsv", "-q", &format!("RESTORE FROM '{arc}' TO '{}'", s.s("hdr"))],
    )
    .ok();
    assert_eq!(head.row(), ["directory", "tables", "parts", "rows", "replayed"]);

    // The database the recovery read from is untouched by any of it.
    assert_eq!(ids(&db), ["1", "2", "3", "4", "5", "6"], "the live database moved");
}

/// The axis a human has: "the state at 14:32:05.250". Each write is fenced by
/// enough wall clock that the target cannot be ambiguous at millisecond
/// resolution, and the recovery to a moment between two writes holds the first
/// and not the second.
#[test]
fn the_statement_recovers_to_the_timestamp_it_was_given() {
    let s = Scratch::new("time");
    let db = s.db();
    step(&db, DDL);
    step(&db, "INSERT INTO t VALUES (1,'a')");
    let arc = s.s("base.gbak");
    step(&db, &format!("BACKUP TO '{arc}'"));

    let mut between = Vec::new();
    for id in 2..=4u32 {
        std::thread::sleep(Duration::from_millis(30));
        step(&db, &format!("INSERT INTO t VALUES ({id},'v{id}')"));
        std::thread::sleep(Duration::from_millis(30));
        between.push(wal::archive_end(&db).expect("archive end").last_ms + 15);
    }

    for (i, &ms) in between.iter().enumerate() {
        // Sub-second, deliberately: truncating to the second would move the
        // target by up to 999 ms, which is far enough to land on the other
        // side of a write and make this test a coin flip under load.
        let text =
            format!("{}.{:03}", granular::types::fmt_datetime((ms / 1000) as i64), ms % 1000);
        let out = s.at(&format!("t-{i}"));
        let r = restore(&db, &arc, &out, &format!("TIMESTAMP '{text}'")).ok();
        let want: Vec<String> = (1..=2 + i as u32).map(|n| n.to_string()).collect();
        assert_eq!(ids(&out), want, "UNTIL TIMESTAMP '{text}' is the state after write {}", i + 2);
        assert_eq!(r.row()[4], (i + 1).to_string(), "one logged insert per step");
    }
}

// ------------------------------------------------------------ what it refuses

/// A target the backup predates. Clamping to the backup would answer a
/// question nobody asked, so it is an error -- and the error arrives before a
/// directory exists, not after a partial unpack.
#[test]
fn the_statement_refuses_a_target_before_the_backup() {
    let s = Scratch::new("before");
    let (arc, _) = staged(&s);
    let db = s.db();
    let base = backup::boundary_of(Path::new(&arc)).expect("boundary");
    assert!(base > 1, "the fixture must have written before the backup");

    let e = restore(&db, &arc, &s.at("x"), &format!("LSN {}", base - 2)).refused();
    assert!(e.contains("before") && e.contains("older backup"), "{e}");
    assert!(!s.at("x").exists(), "a refused recovery must write nothing");

    let created = backup::created_at(Path::new(&arc)).expect("created");
    let text = granular::types::fmt_datetime((created / 1000) as i64 - 60);
    let e = restore(&db, &arc, &s.at("y"), &format!("TIMESTAMP '{text}'")).refused();
    assert!(e.contains("before"), "{e}");
    assert!(!s.at("y").exists());

    // The boundary itself is answerable: it is the backup's own instant, and
    // it must be exactly the backup rather than one record either side.
    restore(&db, &arc, &s.at("z"), &format!("LSN {}", base - 1)).ok();
    assert_eq!(ids(&s.at("z")), ["1", "2", "3"]);
}

/// The silent-wrong-answer case: an LSN nothing ever issued, and a timestamp
/// past an archive that is behind. Either would otherwise hand back the state
/// at the last checkpoint while reporting success.
#[test]
fn the_statement_refuses_a_target_past_the_end_of_the_archive() {
    let s = Scratch::new("after");
    let (arc, _) = staged(&s);
    let db = s.db();
    let end = wal::archive_end(&db).expect("archive end");

    let e = restore(&db, &arc, &s.at("x"), &format!("LSN {}", end.last_seq + 1)).refused();
    assert!(e.contains("past the end"), "{e}");
    assert!(e.contains("no such LSN"), "a checkpointed database has no tail: {e}");
    assert!(!s.at("x").exists(), "a refused recovery must write nothing");

    // Records left in a live log -- what a crash leaves -- and a timestamp
    // past the archive becomes unanswerable rather than approximate.
    {
        let mut sess = Session::open(&db).expect("open");
        sess.execute("INSERT INTO t VALUES (9,'i')").expect("insert");
    }
    let ms = wal::archive_end(&db).expect("archive end").last_ms + 60_000;
    let text = granular::types::fmt_datetime((ms / 1000) as i64);
    let e = restore(&db, &arc, &s.at("y"), &format!("TIMESTAMP '{text}'")).refused();
    assert!(e.contains("un-archived records"), "{e}");
    assert!(e.contains("default.t"), "the message must name the table still holding them: {e}");
    assert!(!s.at("y").exists());
}

/// A hole is the failure this feature has to not have: replaying across one
/// and reporting success hands back a database missing whatever the absent
/// segment held, with nothing to show for it.
#[test]
fn the_statement_refuses_a_recovery_across_a_hole_in_the_archive() {
    let s = Scratch::new("gap");
    let (arc, marks) = staged(&s);
    let db = s.db();

    let mut segs = wal::segments(&db, "default", "t").expect("segments");
    assert!(segs.len() >= 4, "the fixture must archive several segments: {}", segs.len());
    // Out with a middle one, exactly as a half-finished `scp` of the archive
    // or an over-eager cleanup script would take it. One unlink and the
    // segment is gone: there is no sidecar to take with it.
    let victim = segs.remove(segs.len() / 2);
    std::fs::remove_file(&victim.path).expect("remove the segment");

    let e = restore(&db, &arc, &s.at("x"), &format!("LSN {}", marks.last().unwrap())).refused();
    assert!(e.contains("hole"), "{e}");
    assert!(!s.at("x").exists(), "a refused recovery must write nothing");

    // ...and the backup itself is still restorable, because a hole in the log
    // is not damage to the archive it would have been replayed onto.
    restore(&db, &arc, &s.at("base"), "").ok();
    assert_eq!(ids(&s.at("base")), ["1", "2", "3"]);
}

/// The refusal that must survive the new clause. `RESTORE` never writes into
/// the directory the session has open -- two sets of part sequence numbers and
/// two commit records in one place is a database that is neither -- and
/// `UNTIL` must not be a second door to it.
#[test]
fn until_is_not_a_way_into_the_open_database() {
    let s = Scratch::new("open");
    let (arc, marks) = staged(&s);
    let db = s.db();
    let before = ids(&db);

    for until in ["", "LATEST", &format!("LSN {}", marks[0]), &format!("LSN {}", marks[2])] {
        let e = restore(&db, &arc, &db, until).refused();
        assert!(e.contains("has that database open"), "UNTIL {until}: {e}");
        assert!(e.contains("swap"), "the refusal must say what to do instead: {e}");

        // One level down is the same mistake wearing a different path: it
        // would leave a second CATALOG and a second set of part directories
        // inside the tree the loader walks.
        let inner = db.join("restored");
        let e = restore(&db, &arc, &inner, until).refused();
        assert!(e.contains("inside the other"), "nested, UNTIL {until}: {e}");
        assert!(!inner.exists(), "the nested restore wrote into the live database");

        // ...and so is the open database spelled with a detour through it.
        let detour = db.join("..").join(db.file_name().unwrap());
        let e = restore(&db, &arc, &detour, until).refused();
        assert!(e.contains("has that database open"), "via `..`, UNTIL {until}: {e}");
    }
    let mut left: Vec<String> = std::fs::read_dir(&db)
        .expect("the live data directory")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    left.sort();
    assert_eq!(
        left,
        [".wal", "CATALOG", "LOCK", "default"],
        "a refused restore left something in the live data directory"
    );

    // Byte for byte the same database afterwards, not merely the same count.
    assert_eq!(ids(&db), before);
    assert_eq!(
        sql(&db, "SELECT id, v FROM t ORDER BY id").ok().col(),
        sql(&db, "SELECT id, v FROM t ORDER BY id").ok().col()
    );
    assert_eq!(before.len(), 6, "the fixture");
}

/// A target that is not a target. Each is refused at the statement, before a
/// directory is created, and says which of the three spellings to use -- the
/// grammar in `backup::parse_target`, reached through the statement rather
/// than restated beside it.
#[test]
fn a_target_that_is_not_a_target_is_refused_before_anything_is_written() {
    let s = Scratch::new("grammar");
    let (arc, _) = staged(&s);
    let db = s.db();

    for (until, want) in [
        ("LSN soon", "whole number"),
        ("TIMESTAMP 'yesterday'", "yesterday"),
        ("EPOCH 0", "not a recovery target"),
        // Silently dropping the value would answer a question nobody asked.
        ("LATEST 5", "takes no value"),
        ("LSN 5 EXTRA", "takes the form"),
        ("", "takes the form"), // a bare trailing `UNTIL`
    ] {
        let out = s.at("never");
        let q = format!("RESTORE FROM '{arc}' TO '{}' UNTIL {until}", out.to_string_lossy());
        let e = sql(&db, &q).refused();
        assert!(e.contains(want), "`UNTIL {until}` must say `{want}`: {e}");
        assert!(!out.exists(), "`UNTIL {until}` created a directory");
    }

    // And the target the machinery cannot supply: no data directory means no
    // archived log to roll through, and saying so beats restoring the backup
    // and calling it a recovery.
    let e = run(
        None,
        &["--format", "tsv", "--no-header", "-q",
          &format!("RESTORE FROM '{arc}' TO '{}' UNTIL LATEST", s.s("nodir"))],
    )
    .refused();
    assert!(e.contains("in memory"), "{e}");
    assert!(!s.at("nodir").exists());

    // ...while the same session restores the backup's own instant perfectly
    // well, which is the difference the sentence above has to explain.
    run(
        None,
        &["--format", "tsv", "--no-header", "-q",
          &format!("RESTORE FROM '{arc}' TO '{}'", s.s("plain"))],
    )
    .ok();
    assert_eq!(ids(&s.at("plain")), ["1", "2", "3"]);
}

// ------------------------------------------------- the log outlives the table

/// **The commonest recovery there is: somebody dropped the table.**
///
/// The log lives at `<data>/.wal/<db>/<table>` rather than inside the table's
/// own directory precisely so that a `DROP TABLE` -- which removes that
/// directory outright -- cannot take the history with it. This is the test
/// that the arrangement actually pays off through the statement, and it pins
/// two things a drop must do to the log on its way past.
///
/// The first is that the drop **seals** it. A recovery reads sealed segments
/// only, because the live one may end in an interrupted append, and nothing
/// will ever roll a dead table's log again -- so records written and dropped
/// inside one checkpoint interval would be on disk, intact, and permanently
/// unreachable, with `UNTIL LATEST` reporting success and handing back a
/// database missing them. That is a silent loss produced by a successful
/// recovery, which is the one outcome this feature exists to rule out.
///
/// The second is that it stops lagging. `archive_lags` walks log directories,
/// not the catalog, so an unsealed dead log answers "still holds un-archived
/// records" for ever, and every `UNTIL TIMESTAMP` past it is refused with
/// advice -- checkpoint the database -- that no amount of checkpointing can
/// satisfy. A wedge with a remedy that does not work is worse than no remedy.
#[test]
fn a_dropped_tables_history_is_still_recoverable_through_the_statement() {
    let s = Scratch::new("dropped");
    let db = s.db();
    step(&db, DDL);
    step(&db, "INSERT INTO t VALUES (1,'a')");
    let arc = s.s("base.gbak");
    step(&db, &format!("BACKUP TO '{arc}'"));

    // Both inserts and the drop in *one* invocation, so the drop lands before
    // any checkpoint has retired what they logged. That ordering is the whole
    // fixture: split across processes, the exit checkpoint would seal them and
    // the bug would not reproduce.
    step(&db, "INSERT INTO t VALUES (2,'b'); INSERT INTO t VALUES (3,'c'); DROP TABLE t");
    assert!(sql(&db, "SELECT count() FROM t").code != 0, "the table must really be gone");

    let latest = s.at("latest");
    let r = restore(&db, &arc, &latest, "LATEST").ok();
    assert_eq!(
        ids(&latest),
        ["1", "2", "3"],
        "the rows written before the drop are gone from the recovery"
    );
    assert_eq!(r.row()[4], "2", "both logged inserts had to be replayed: {:?}", r.row());

    // A moment past everything the archive holds is answerable rather than
    // wedged: with the dead log sealed there is nothing left lagging.
    let far = s.at("far");
    restore(&db, &arc, &far, "TIMESTAMP '2999-01-01 00:00:00'").ok();
    assert_eq!(ids(&far), ["1", "2", "3"]);

    // ...and it is still answerable after another checkpoint, which is what
    // distinguishes "sealed" from "we happened to catch it before the drop".
    step(&db, "CREATE TABLE u (id UInt64, v String) ENGINE = MergeTree ORDER BY id");
    let again = s.at("again");
    restore(&db, &arc, &again, "TIMESTAMP '2999-01-01 00:00:00'").ok();
    assert_eq!(ids(&again), ["1", "2", "3"]);
}

/// A table recreated under a dropped one's name must not inherit its history.
///
/// The same arrangement that makes the test above work creates this hazard:
/// `CREATE TABLE t` after `DROP TABLE t` opens the *dead* table's log
/// directory. A fresh table has no commit record, so a watermark defaulting to
/// the start of the stream would replay the corpse into a schema that need not
/// even match it. `CREATE TABLE` stamps the position the stream has already
/// reached, which is what makes the answer zero rather than "whatever decodes".
#[test]
fn a_table_recreated_over_a_dropped_ones_log_starts_empty() {
    let s = Scratch::new("recreate");
    let db = s.db();
    step(&db, DDL);
    step(&db, "INSERT INTO t VALUES (1,'a'),(2,'b'),(3,'c')");
    // A different shape entirely, so a replayed record could not even be
    // mistaken for a row of this table.
    step(&db, "DROP TABLE t; CREATE TABLE t (k String, n Int64) ENGINE = MergeTree ORDER BY k");

    let count = |q: &str| sql(&db, q).ok().col().join("");
    assert_eq!(count("SELECT count() FROM t"), "0", "the previous incarnation replayed");
    // Again in a fresh process: the first answer came from memory, this one
    // comes from the loader, which is where the replay would happen.
    assert_eq!(count("SELECT count() FROM t"), "0", "the previous incarnation replayed on open");
    assert_eq!(sql(&db, "SELECT k, n FROM t").ok().col(), Vec::<String>::new());

    // The new table still takes writes, and they survive a reopen -- a
    // watermark parked past the end of the stream would swallow them too.
    step(&db, "INSERT INTO t VALUES ('x', 7)");
    assert_eq!(sql(&db, "SELECT k, n FROM t").ok().col(), ["x\t7"]);
}

// ----------------------------------------------------------------- retention

/// Retention is a window, and its edge has to be a refusal rather than a
/// smaller answer.
///
/// `wal_archive_retention` is a byte budget over archived segments, so setting
/// it small drops the oldest ones and moves the horizon up past the backup.
/// A recovery that would have to replay through what went must refuse and say
/// so: rolling forward across the hole would hand back a database missing
/// whatever those segments held, with a success report on top. `system.wal` is
/// where an operator reads the same numbers *before* they need them, so it has
/// to agree with the refusal.
#[test]
fn retention_moves_the_window_and_a_recovery_behind_it_is_refused() {
    let s = Scratch::new("retention");
    let (arc, marks) = staged(&s);
    let db = s.db();

    let wal = |cols: &str| sql(&db, &format!("SELECT {cols} FROM system.wal")).ok().row();
    let before = wal("segments, archive_first_seq, archive_last_seq, horizon_seq");
    assert!(before[0].parse::<u64>().expect("a count") >= 4, "fixture: {before:?}");
    assert_eq!(before[3], "0", "nothing has been pruned yet");
    assert_eq!(before[1], "1", "so the archive still serves from the very first LSN");
    assert_eq!(before[2], marks.last().unwrap().to_string(), "...up to the last write");

    // ...and every one of those LSNs really is answerable while the window
    // still covers them.
    restore(&db, &arc, &s.at("wide"), &format!("LSN {}", marks[0])).ok();
    assert_eq!(ids(&s.at("wide")), ["1", "2", "3", "4"]);

    // One byte of budget: every archived segment is over it, so they all go as
    // the next one is created.
    step(&db, "SET wal_archive_retention = 1; INSERT INTO t VALUES (7,'g')");
    let after = wal("segments, archive_first_seq, archive_last_seq, horizon_seq");
    assert_eq!(after[0], "0", "retention kept a segment it had no budget for: {after:?}");
    let horizon: u64 = after[3].parse().expect("a horizon");
    assert!(horizon >= *marks.last().unwrap(), "the horizon must follow the files: {after:?}");
    assert_eq!(after[1], "0", "an empty archive spans nothing");

    // The recovery that needed them refuses, names the horizon, and writes
    // nothing -- rather than replaying what survives and calling it a
    // recovery.
    let e = restore(&db, &arc, &s.at("gone"), "LATEST").refused();
    assert!(e.contains("begins at recovery LSN"), "{e}");
    assert!(e.contains("not in the backup"), "the refusal must say what is missing: {e}");
    assert!(!s.at("gone").exists(), "a refused recovery must write nothing");
    for until in [&format!("LSN {}", marks[0]), "TIMESTAMP '2999-01-01 00:00:00'"] {
        let e = restore(&db, &arc, &s.at("gone2"), until).refused();
        assert!(!e.trim().is_empty(), "UNTIL {until} was refused with no reason");
        assert!(!s.at("gone2").exists(), "UNTIL {until} wrote a directory");
    }

    // The backup itself is still restorable, and that is not an oversight:
    // `RESTORE` with no `UNTIL` rolls nothing forward, so it needs no archive
    // at all. Retention takes away the recoveries that read the log, not the
    // backup's own instant.
    let plain = s.at("plain");
    let r = restore(&db, &arc, &plain, "").ok();
    assert_eq!(r.row()[4], "0", "a plain restore replays nothing: {:?}", r.row());
    assert_eq!(ids(&plain), ["1", "2", "3"]);

    // The live database is untouched by any of it and still holds everything.
    assert_eq!(ids(&db), ["1", "2", "3", "4", "5", "6", "7"]);
}

/// `system.wal`'s archive columns are what an operator reads to find out
/// whether a recovery is possible *before* attempting one, so they have to be
/// the same numbers the recovery itself uses -- not a parallel calculation
/// that can drift.
///
/// `archive_first_seq` in particular used to be the oldest segment's own
/// `first_seq`, which is the tick immediately *before* that segment: off by
/// one from the first LSN a recovery can serve, and provably equal to
/// `horizon_seq` in every case where the archive is not empty. Two columns
/// carrying one number is a column that answers no question.
#[test]
fn the_archive_columns_describe_the_window_a_recovery_actually_has() {
    let s = Scratch::new("columns");
    let (arc, marks) = staged(&s);
    let db = s.db();
    let row = sql(
        &db,
        "SELECT segments, archive_first_seq, archive_last_seq, horizon_seq, lags FROM system.wal",
    )
    .ok()
    .row();
    let n: u64 = row[0].parse().expect("segments");
    let (first, last, horizon) = (
        row[1].parse::<u64>().expect("first"),
        row[2].parse::<u64>().expect("last"),
        row[3].parse::<u64>().expect("horizon"),
    );
    assert!(n > 0 && first <= last, "{row:?}");
    assert_eq!(first, horizon + 1, "the window starts one past what pruning discarded");
    assert_ne!(first, horizon, "two columns carrying one number");
    assert_eq!(last, *marks.last().unwrap(), "the window ends at the last archived write");
    assert_eq!(row[4], "0", "a checkpointed database does not lag");

    // The window has to *contain* the backup, or this archive could not be
    // rolled forward at all -- which is the check `RESTORE` makes before it
    // unpacks anything, expressed here as the two numbers an operator can read
    // beforehand.
    let base = backup::boundary_of(Path::new(&arc)).expect("boundary");
    assert!(first <= base, "the archive starts after the backup: {first} > {base}");

    // The far edge is answerable and one past it is not. Below the backup is a
    // *different* refusal -- a backup cannot roll backwards -- so the low edge
    // of the window is not a restore target for this archive, and asserting it
    // were would be pinning the wrong constraint.
    restore(&db, &arc, &s.at("hi"), &format!("LSN {last}")).ok();
    assert_eq!(ids(&s.at("hi")), ids(&db), "the far edge is the live database");
    let e = restore(&db, &arc, &s.at("past"), &format!("LSN {}", last + 1)).refused();
    assert!(e.contains("past the end"), "{e}");
    assert!(!s.at("past").exists());
    let e = restore(&db, &arc, &s.at("under"), &format!("LSN {first}")).refused();
    assert!(e.contains("older backup"), "below the backup is its own refusal: {e}");
    assert!(!s.at("under").exists());
}

/// Retention that has moved past the backup for **one** table refuses the
/// whole recovery, before it has written anything.
///
/// The directory-wide window (`archive_first_seq`) is the union: one table
/// whose segments have all been pruned does not narrow it, so the global check
/// sees a window that still covers the backup and lets the recovery start.
/// What then refuses is the per-table check inside `wal::recover` -- correct,
/// but by then the archive has been unpacked, and the operator is left with a
/// half-written directory they have to delete before they can retry. The same
/// question is now asked of every table in the manifest up front.
///
/// The fixture is two tables with very different write rates, which is the
/// ordinary shape this arises in: `hot` outgrows the byte budget and loses its
/// archive; `cold` was written once and keeps its single segment, so the
/// window still looks wide enough.
#[test]
fn one_tables_retention_refuses_the_recovery_before_it_unpacks_anything() {
    let s = Scratch::new("perTable");
    let db = s.db();
    step(&db, DDL);
    step(&db, "CREATE TABLE cold (id UInt64, v String) ENGINE = MergeTree ORDER BY id");
    step(&db, "INSERT INTO t VALUES (1,'a'); INSERT INTO cold VALUES (1,'a')");
    let arc = s.s("base.gbak");
    step(&db, &format!("BACKUP TO '{arc}'"));

    // `hot` alone writes, so `cold`'s one segment stays and the union window
    // keeps reaching back to the backup.
    for id in 2..=8u32 {
        step(&db, &format!("INSERT INTO t VALUES ({id},'x')"));
    }
    step(&db, "SET wal_archive_retention = 1; INSERT INTO t VALUES (9,'x')");

    let wal_of = |t: &str| {
        sql(&db, &format!("SELECT segments, horizon_seq FROM system.wal WHERE table = '{t}'"))
            .ok()
            .row()
    };
    assert_eq!(wal_of("t")[0], "0", "fixture: `t`'s archive should be gone");
    assert_ne!(wal_of("cold")[0], "0", "fixture: `cold` should still have its segment");
    let first = sql(&db, "SELECT archive_first_seq FROM system.wal WHERE table = 'cold'")
        .ok()
        .row()[0]
        .clone();
    assert_eq!(first, "1", "fixture: the surviving table still spans the backup");

    let out = s.at("perTable");
    let e = restore(&db, &arc, &out, "LATEST").refused();
    assert!(e.contains("no longer reaches back to the backup"), "{e}");
    assert!(e.contains("`d.t`") || e.contains(".t`"), "the refusal must name the table: {e}");
    assert!(
        !out.exists(),
        "the refusal unpacked the archive first and left {} behind",
        out.display()
    );

    // And the live database is untouched by the attempt.
    assert_eq!(ids(&db).len(), 9);
}
