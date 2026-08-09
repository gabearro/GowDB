//! Point-in-time recovery, end to end: the log between two backups.
//!
//! Everything that *produces* an archive here goes through the shipped binary,
//! because that is the half that has to be wired into the engine. A checkpoint
//! is what retires a log, the CLI takes one on the way out, and if archiving
//! were not reachable from that path every test below would find an empty
//! `.wal-archive` and fail. The recovery itself is a function of the
//! filesystem and is called directly (`granular::backup::restore_until`); its
//! *result* is then read back through the binary, so a restored directory that
//! the real loader cannot open is a failure here.
//!
//! What is pinned:
//!
//!   1. **A recovery lands on the instant asked for.** Not the backup, not the
//!      latest state: exactly the rows that existed at the target, on both the
//!      LSN axis and the timestamp axis. Twice over, because a recovery that
//!      is not deterministic is a recovery you cannot check.
//!   2. **Every unanswerable target is refused, loudly.** Before the backup,
//!      past the end of the archive, across a hole in the archive, through a
//!      segment retention has dropped, and through a segment a crash caught
//!      mid-archive. Each one is a state where the alternative is reporting
//!      success while silently skipping records.
//!   3. **A `kill -9` never leaves the archive silently short.** Whatever the
//!      process was doing, what is on disk afterwards is either whole or
//!      visibly incomplete -- and a clean restart repairs it.
//!
//! The one thing not driven from SQL is the recovery target itself: `RESTORE
//! FROM ... TO ...` has no `UNTIL` clause yet, and the statement is parsed in
//! `session.rs`, which this wave does not own. `pin_restore_until_is_not_yet_a
//! _statement` fails the day that lands and says what to do about it.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use granular::backup::{self, Target};
use granular::persist::wal;
use granular::Session;

const BIN: &str = env!("CARGO_BIN_EXE_granular");

// ------------------------------------------------------------------ harness

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let d = std::env::temp_dir().join(format!("granular-pitr-{}-{tag}", std::process::id()));
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
    fn of(o: Output) -> Run {
        Run {
            code: o.status.code().unwrap_or(-1),
            out: String::from_utf8_lossy(&o.stdout).into_owned(),
            err: String::from_utf8_lossy(&o.stderr).into_owned(),
        }
    }
    fn ok(self) -> Run {
        assert_eq!(self.code, 0, "stdout:\n{}\nstderr:\n{}", self.out, self.err);
        self
    }
    /// The single column of a one-column result, in order.
    fn col(&self) -> Vec<String> {
        self.out.lines().filter(|l| !l.is_empty()).map(str::to_string).collect()
    }
}

fn sql(db: &Path, q: &str) -> Run {
    Run::of(
        Command::new(BIN)
            .args(["--data", &db.to_string_lossy(), "--format", "tsv", "--no-header", "-q", q])
            .output()
            .expect("spawn granular"),
    )
}

/// One CLI invocation, which is also one checkpoint and therefore one archived
/// segment: the process checkpoints on the way out, and that is what retires
/// the log into the archive.
fn step(db: &Path, q: &str) -> Run {
    sql(db, q).ok()
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap()
}

/// The `id`s in `t`, ascending, read by the real binary out of `dir`.
fn ids(dir: &Path) -> Vec<String> {
    sql(dir, "SELECT id FROM t ORDER BY id").ok().col()
}

const DDL: &str = "CREATE TABLE t (id UInt64, v String) ENGINE = MergeTree ORDER BY id";

/// Held by anything that turns the process-global retention budget down, and
/// by every test that would notice if it were.
static ARCHIVE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    ARCHIVE.lock().unwrap_or_else(|e| e.into_inner())
}

/// A database with three rows, a backup of it, and three more rows written
/// afterwards -- one CLI invocation each, so every write lands in its own
/// archived segment and the boundaries are nameable.
///
/// Returns the archive path and the recovery LSN each later write was stamped
/// with, in order.
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

/// The headline claim: a recovery to a point in the middle is that point --
/// neither the backup it started from nor the latest state it could have
/// reached.
#[test]
fn a_recovery_lands_on_the_lsn_it_was_given() {
    let _x = exclusive();
    let s = Scratch::new("lsn");
    let (arc, marks) = staged(&s);
    let db = s.db();
    assert_eq!(ids(&db), ["1", "2", "3", "4", "5", "6"], "fixture");

    // The backup alone: three rows, whatever the archive holds.
    let plain = s.at("plain");
    backup::restore(Path::new(&arc), &plain).expect("restore the base");
    assert_eq!(ids(&plain), ["1", "2", "3"]);

    // ...and each archived write, one at a time.
    for (i, &mark) in marks.iter().enumerate() {
        let out = s.at(&format!("at-{mark}"));
        let rep = backup::restore_until(Path::new(&arc), &db, &out, Target::Lsn(mark))
            .unwrap_or_else(|e| panic!("recover to LSN {mark}: {e}"));
        let want: Vec<String> = (1..=4 + i as u32).map(|n| n.to_string()).collect();
        assert_eq!(ids(&out), want, "LSN {mark} replayed {} records", rep.replayed);
        assert_eq!(rep.replayed as usize, i + 1, "one logged insert per step");
    }

    // The far end, by both spellings.
    for (tag, target) in [("latest", Target::Latest), ("last", Target::Lsn(*marks.last().unwrap()))]
    {
        let out = s.at(tag);
        backup::restore_until(Path::new(&arc), &db, &out, target).expect("recover");
        assert_eq!(ids(&out), ["1", "2", "3", "4", "5", "6"], "{tag}");
    }
}

/// Same target, twice, into two directories: byte-identical logs and identical
/// answers. A recovery whose result depends on when it ran is one nobody can
/// check against anything.
#[test]
fn replaying_to_the_same_target_twice_gives_the_same_state() {
    let _x = exclusive();
    let s = Scratch::new("determinism");
    let (arc, marks) = staged(&s);
    let db = s.db();
    let target = Target::Lsn(marks[1]);

    let (a, b) = (s.at("a"), s.at("b"));
    let ra = backup::restore_until(Path::new(&arc), &db, &a, target).expect("first");
    let rb = backup::restore_until(Path::new(&arc), &db, &b, target).expect("second");
    assert_eq!(ra.replayed, rb.replayed);
    assert_eq!(ids(&a), ["1", "2", "3", "4", "5"]);
    assert_eq!(ids(&b), ids(&a));

    let log = |d: &Path| std::fs::read(d.join("default").join("t").join("wal.log")).unwrap();
    assert_eq!(log(&a), log(&b), "the recovered log is a byte range of the archive, not a rebuild");
}

/// The axis a human has. Each write is separated by enough wall clock to be
/// nameable, and the recovery to a moment between two of them holds the first
/// and not the second.
#[test]
fn a_recovery_lands_on_the_timestamp_it_was_given() {
    let _x = exclusive();
    let s = Scratch::new("time");
    let db = s.db();
    step(&db, DDL);
    step(&db, "INSERT INTO t VALUES (1,'a')");
    let arc = s.s("base.gbak");
    step(&db, &format!("BACKUP TO '{arc}'"));

    // Three writes with a gap either side of the instant we will name, so the
    // target cannot be ambiguous at millisecond resolution.
    let mut between = Vec::new();
    for id in 2..=4u32 {
        std::thread::sleep(Duration::from_millis(30));
        step(&db, &format!("INSERT INTO t VALUES ({id},'v{id}')"));
        std::thread::sleep(Duration::from_millis(30));
        between.push(now_ms());
    }

    for (i, &t) in between.iter().enumerate() {
        let out = s.at(&format!("t-{i}"));
        backup::restore_until(Path::new(&arc), &db, &out, Target::Time(t))
            .unwrap_or_else(|e| panic!("recover to {t}: {e}"));
        let want: Vec<String> = (1..=2 + i as u32).map(|n| n.to_string()).collect();
        assert_eq!(ids(&out), want, "recovery to the moment after write {}", i + 2);
    }

    // ...and the same target spelled the way an operator would type it.
    // Sub-second, deliberately: truncating to the second would move the target
    // by up to 999 ms, which is far enough to land on the other side of a
    // write and make this test a coin flip under load.
    let text = format!(
        "{}.{:03}",
        granular::types::fmt_datetime((between[0] / 1000) as i64),
        between[0] % 1000
    );
    let target = backup::parse_target("timestamp", &text).expect("parse the target");
    assert_eq!(target, Target::Time(between[0]));
    let out = s.at("typed");
    backup::restore_until(Path::new(&arc), &db, &out, target).expect("recover");
    assert_eq!(ids(&out), ["1", "2"], "`{text}` must be the state just after the first write");
}

// ------------------------------------------------------------ what it refuses

#[test]
fn a_target_before_the_backup_is_an_error_not_a_clamp() {
    let _x = exclusive();
    let s = Scratch::new("before");
    let (arc, _) = staged(&s);
    let db = s.db();
    let base = backup::boundary_of(Path::new(&arc)).expect("boundary");
    assert!(base > 1, "the fixture must have written before the backup");

    let e = backup::restore_until(Path::new(&arc), &db, &s.at("x"), Target::Lsn(base - 2))
        .expect_err("a target before the backup must be refused");
    assert!(e.to_string().contains("before"), "{e}");
    assert!(!s.at("x").exists(), "a refused recovery must write nothing");

    let created = backup::created_at(Path::new(&arc)).expect("created");
    let e = backup::restore_until(
        Path::new(&arc),
        &db,
        &s.at("y"),
        Target::Time(created.saturating_sub(60_000)),
    )
    .expect_err("a timestamp before the backup must be refused");
    assert!(e.to_string().contains("before"), "{e}");

    // The boundary itself is answerable: it is the backup's own instant.
    backup::restore_until(Path::new(&arc), &db, &s.at("z"), Target::Lsn(base - 1))
        .expect("the backup's own LSN is a state it can be restored to");
    assert_eq!(ids(&s.at("z")), ["1", "2", "3"]);
}

/// The silent-wrong-answer case this guard exists for: the last few minutes
/// are still in the live log, so "restore to five minutes ago" would quietly
/// hand back the state at the last checkpoint instead.
#[test]
fn a_target_past_the_end_of_the_archive_is_an_error() {
    let _x = exclusive();
    let s = Scratch::new("after");
    let (arc, marks) = staged(&s);
    let db = s.db();
    let end = wal::archive_end(&db).expect("archive end");
    assert_eq!(end.last_seq, *marks.last().unwrap());

    // An LSN nothing ever issued is a mistake whether or not the archive is
    // behind, because an LSN is a number a tool handed the operator.
    let e = backup::restore_until(Path::new(&arc), &db, &s.at("x"), Target::Lsn(end.last_seq + 1))
        .expect_err("an LSN past the archive must be refused");
    assert!(e.to_string().contains("past the end"), "{e}");
    assert!(e.to_string().contains("no such LSN"), "a checkpointed database has no tail: {e}");
    assert!(!s.at("x").exists(), "a refused recovery must write nothing");

    // With every log archived, a timestamp after the last write is simply the
    // latest state -- there is nothing missing to be wrong about.
    backup::restore_until(Path::new(&arc), &db, &s.at("idle"), Target::Time(end.last_ms + 60_000))
        .expect("an idle database is recoverable to any instant since its last write");
    assert_eq!(ids(&s.at("idle")), ["1", "2", "3", "4", "5", "6"]);

    // Now leave records in a live log -- a session dropped without a
    // checkpoint, which is what a crash leaves -- and the same target becomes
    // unanswerable.
    {
        let mut sess = Session::open(&db).expect("open");
        sess.execute("INSERT INTO t VALUES (9,'i')").expect("insert");
    }
    let e = backup::restore_until(
        Path::new(&arc),
        &db,
        &s.at("y"),
        Target::Time(wal::archive_end(&db).unwrap().last_ms + 60_000),
    )
    .expect_err("a timestamp past an archive that is behind must be refused");
    assert!(e.to_string().contains("un-archived records"), "{e}");
    assert!(e.to_string().contains("default.t"), "the message must name the table: {e}");
    assert!(!s.at("y").exists());
}

/// A hole is the failure this whole feature has to not have: replaying across
/// one and reporting success would hand back a database missing whatever the
/// missing segment held, with nothing to show for it.
#[test]
fn a_gap_in_the_archive_is_refused_and_names_the_missing_range() {
    let _x = exclusive();
    let s = Scratch::new("gap");
    let (arc, marks) = staged(&s);
    let db = s.db();

    let dir = wal::archive_dir(&db, "default", "t");
    let mut segs = wal::segments(&db, "default", "t").expect("segments");
    assert!(segs.len() >= 4, "the fixture must archive several segments: {}", segs.len());
    // Take out a middle one, exactly as a half-finished `scp` of the archive
    // or an over-eager cleanup script would.
    let victim = segs.remove(segs.len() / 2);
    std::fs::remove_file(&victim.path).expect("remove the segment");
    let _ = std::fs::remove_file(dir.join(
        victim.path.file_name().unwrap().to_string_lossy().replace(".gwal", ".gseal"),
    ));

    let e = wal::segments(&db, "default", "t").expect_err("a hole must be reported");
    let msg = e.to_string();
    assert!(msg.contains("hole"), "{msg}");
    assert!(
        msg.contains(&victim.origin.to_string()) && msg.contains(&victim.end.to_string()),
        "the message must name the missing byte range {}..{}: {msg}",
        victim.origin,
        victim.end
    );

    let e = backup::restore_until(Path::new(&arc), &db, &s.at("x"), Target::Lsn(*marks.last().unwrap()))
        .expect_err("a recovery over a hole must be refused");
    assert!(e.to_string().contains("hole"), "{e}");
    assert!(!s.at("x").exists(), "a refused recovery must write nothing");
}

/// The crash window archiving has: the log is linked into the archive and the
/// seal is not written yet. That link is not a short segment -- it is not a
/// segment at all, because the seal is what publishes one, and the records it
/// holds are still in the log the interrupted checkpoint never replaced.
#[test]
fn an_unsealed_link_is_not_part_of_the_archive_and_is_superseded() {
    let _x = exclusive();
    let s = Scratch::new("unsealed");
    let db = s.db();
    step(&db, DDL);
    step(&db, "INSERT INTO t VALUES (1,'a')");
    let arc = s.s("base.gbak");
    step(&db, &format!("BACKUP TO '{arc}'"));
    step(&db, "INSERT INTO t VALUES (2,'b')");

    // The exact state a `kill -9` between `link` and the seal leaves: the link
    // is there, the seal is not, and the log was never replaced -- so the
    // records exist in both places and neither copy has been lost.
    let segs = wal::segments(&db, "default", "t").expect("segments");
    let newest = segs.last().expect("a segment").clone();
    let seal = PathBuf::from(newest.path.to_string_lossy().replace(".gwal", ".gseal"));
    std::fs::copy(&newest.path, db.join("default").join("t").join("wal.log")).expect("un-truncate");
    std::fs::remove_file(&seal).expect("remove the seal");
    assert_eq!(
        wal::segments(&db, "default", "t").expect("segments").len(),
        segs.len() - 1,
        "an unsealed link must not read as a segment"
    );

    // Its records are still in the archive-less half of the world, so a
    // recovery to the latest archived state simply does not have them yet.
    let out = s.at("mid-crash");
    backup::restore_until(Path::new(&arc), &db, &out, Target::Latest).expect("recover");
    assert_eq!(ids(&out), ["1"], "an unsealed link must contribute nothing");

    // The next checkpoint takes that position back and archives the log as it
    // now stands, which is a superset -- so nothing is written twice and
    // nothing is lost.
    step(&db, "INSERT INTO t VALUES (3,'c')");
    let after = wal::segments(&db, "default", "t").expect("segments");
    assert_eq!(
        after.len(),
        segs.len(),
        "the retry must reuse the position, not chain past it: {after:?}"
    );
    assert_eq!(after.last().unwrap().origin, newest.origin);
    let out = s.at("healed");
    backup::restore_until(Path::new(&arc), &db, &out, Target::Latest).expect("recover");
    assert_eq!(ids(&out), ["1", "2", "3"], "the superseded records must appear exactly once");
}

/// The other half of that contract, and the one with teeth: a segment that
/// *is* sealed and is shorter than its seal claims is damage, not a shorter
/// history. Reading it would stop early and report success.
#[test]
fn a_sealed_segment_that_is_short_is_reported_not_replayed() {
    let _x = exclusive();
    let s = Scratch::new("short");
    let (arc, _) = staged(&s);
    let db = s.db();
    let seg = wal::segments(&db, "default", "t").expect("segments").pop().expect("a segment");
    let bytes = std::fs::read(&seg.path).expect("segment");
    std::fs::write(&seg.path, &bytes[..bytes.len() - 8]).expect("truncate the segment");

    let e = backup::restore_until(Path::new(&arc), &db, &s.at("x"), Target::Latest)
        .expect_err("a short segment must not be replayed");
    assert!(e.to_string().contains("report success"), "{e}");
    assert!(!s.at("x").exists());
}

// ------------------------------------------------------------------ retention

/// Unbounded growth is how this feature becomes an outage, so the archive has
/// a byte budget -- and dropping a segment has to be *recorded*, or a recovery
/// that needed it would replay a shorter history and report success.
#[test]
fn retention_prunes_the_oldest_segments_and_a_recovery_that_needed_them_refuses() {
    let s = Scratch::new("retention");
    let db = s.db();
    step(&db, DDL);
    step(&db, "INSERT INTO t VALUES (1,'a')");
    let arc = s.s("base.gbak");
    step(&db, &format!("BACKUP TO '{arc}'"));

    // In process, because the budget is a library setting and the statement
    // that reaches it (`SET wal_archive_retention`) is one line of session
    // wiring this wave does not own. What is being pinned is the mechanism.
    //
    // The budget is process-global -- it describes a data directory, and there
    // is one writer per directory -- so this holds the other tests in this
    // binary off while it is turned down.
    let _x = exclusive();
    let restore_default = Guard(wal::archive_retention());
    wal::set_archive_retention(1);
    {
        let mut sess = Session::open(&db).expect("open");
        for id in 2..=8u32 {
            sess.execute(&format!("INSERT INTO t VALUES ({id},'v{id}')")).expect("insert");
            sess.checkpoint().expect("checkpoint");
        }
    }
    drop(restore_default);

    let segs = wal::segments(&db, "default", "t").expect("segments");
    assert!(segs.len() <= 2, "a 1-byte budget must keep almost nothing: {}", segs.len());
    assert!(!segs.is_empty(), "the newest segment is never pruned -- it carries the numbering");

    let e = backup::restore_until(Path::new(&arc), &db, &s.at("x"), Target::Latest)
        .expect_err("a recovery that needs a pruned segment must refuse");
    let msg = e.to_string();
    assert!(msg.contains("retention") || msg.contains("begins at"), "{msg}");
    assert!(!s.at("x").exists());

    // ...and the live database is untouched by any of it.
    assert_eq!(ids(&db).len(), 8);
}

/// Restores the retention budget however the test above ends.
struct Guard(u64);

impl Drop for Guard {
    fn drop(&mut self) {
        wal::set_archive_retention(self.0);
    }
}

// ---------------------------------------------------------------------- crash

/// `kill -9` at a spread of moments, including inside the exit checkpoint that
/// does the archiving. Afterwards the archive is either whole or visibly
/// incomplete; it is never a segment that parses and is short.
#[test]
fn a_kill_during_archiving_never_leaves_the_archive_silently_short() {
    let _x = exclusive();
    let s = Scratch::new("kill");
    let db = s.db();
    step(&db, DDL);
    step(&db, "INSERT INTO t VALUES (1,'a')");
    let arc = s.s("base.gbak");
    step(&db, &format!("BACKUP TO '{arc}'"));

    for round in 0..12u64 {
        let mut child = Command::new(BIN)
            .args(["--data", &db.to_string_lossy(), "--format", "tsv", "--no-header"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn granular");
        let mut sin = child.stdin.take().expect("stdin");
        let _ = writeln!(sin, "INSERT INTO t VALUES ({}, 'k{round}');", 100 + round);
        let _ = sin.flush();
        drop(sin);
        // Sweeping the delay walks the kill across the statement, the exit
        // checkpoint, the link and the seal.
        std::thread::sleep(Duration::from_micros(200 + round * 900));
        let _ = child.kill();
        let _ = child.wait();

        // Whatever state that left, the archive is readable or it says why.
        let segs = match wal::segments(&db, "default", "t") {
            Ok(segs) => segs,
            Err(e) => panic!("round {round}: the archive lost its shape: {e}"),
        };
        for w in segs.windows(2) {
            assert_eq!(w[0].end, w[1].origin, "round {round}: a hole appeared with no error");
        }
        for seg in &segs {
            let len = std::fs::metadata(&seg.path).expect("segment").len();
            assert_eq!(
                seg.end - seg.origin + granular::persist::format::HEADER_LEN as u64,
                len,
                "round {round}: {} is not the length its seal claims",
                seg.path.display()
            );
            // The bug this test found: a writer killed between its append and
            // its `fsync` leaves records that the next open replays anyway,
            // and a read-only session's exit checkpoint would then archive a
            // whole segment carrying no tick -- unplaceable in time, and
            // silently skipped by every recovery.
            assert!(
                !seg.span.is_empty(),
                "round {round}: {} carries {} bytes of log and no tick to place them by",
                seg.path.display(),
                seg.end - seg.origin
            );
        }
        // ...and whatever survived is recoverable, every round.
        let out = s.at("crash");
        let _ = std::fs::remove_dir_all(&out);
        backup::restore_until(Path::new(&arc), &db, &out, Target::Latest)
            .unwrap_or_else(|e| panic!("round {round}: the archive stopped being usable: {e}"));
    }

    // A clean restart repairs the archive, and the recovery then agrees with
    // the database that survived.
    let live = ids(&db);
    step(&db, "SELECT 1");
    let out = s.at("after");
    backup::restore_until(Path::new(&arc), &db, &out, Target::Latest).expect("recover");
    assert_eq!(ids(&out), live, "a restart-repaired archive must reproduce the survivor");
}

// ------------------------------------------------------------------- the pins

/// PIN. The recovery target has no SQL spelling yet: `admin_stmt` in
/// `session.rs` accepts `RESTORE FROM '<archive>' TO '<dir>'` and nothing
/// else, and that file is not this wave's to change. The exact two-hunk patch
/// is in this wave's `outsideMyFiles`.
///
/// When it lands this test starts failing. Invert it: assert that the
/// statement recovers to the LSN, which is what
/// `a_recovery_lands_on_the_lsn_it_was_given` already does through
/// `backup::restore_until` -- the statement is a three-line wrapper over it.
#[test]
fn pin_restore_until_is_not_yet_a_statement() {
    let _x = exclusive();
    let s = Scratch::new("pin");
    let (arc, marks) = staged(&s);
    let db = s.db();
    let r = sql(
        &db,
        &format!("RESTORE FROM '{arc}' TO '{}' UNTIL LSN {}", s.s("out"), marks[0]),
    );
    assert_eq!(r.code, 1, "RESTORE ... UNTIL now parses: invert this pin");
    assert!(
        r.err.contains("takes the form"),
        "RESTORE ... UNTIL reached the engine -- the session.rs patch has landed, so this \
         pin is stale. Replace it with an assertion that the statement recovers to the LSN.\n{}",
        r.err
    );
}

/// The target grammar itself is this module's, so it is checked here even
/// though nothing types it yet.
#[test]
fn a_recovery_target_is_parsed_or_refused_with_a_reason() {
    assert_eq!(backup::parse_target("LATEST", "").unwrap(), Target::Latest);
    assert_eq!(backup::parse_target("lsn", "42").unwrap(), Target::Lsn(42));
    let ms = 1_770_000_000_000u64;
    let text = granular::types::fmt_datetime((ms / 1000) as i64);
    assert_eq!(backup::parse_target("timestamp", &text).unwrap(), Target::Time(ms));
    // Sub-second, which is the resolution "just before the bad deploy" needs.
    assert_eq!(
        backup::parse_target("timestamp", &format!("{text}.250")).unwrap(),
        Target::Time(ms + 250)
    );
    for (kind, value) in [("lsn", "soon"), ("timestamp", "yesterday"), ("epoch", "0")] {
        assert!(backup::parse_target(kind, value).is_err(), "`UNTIL {kind} {value}`");
    }
}

