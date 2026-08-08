//! A transaction across several tables commits all of itself, or none of it.
//!
//! Logs are per table, so an N-table transaction has N markers in N files and
//! N `fsync`s. Nothing about that ordering makes the transaction atomic on its
//! own: a crash after the second `fsync` of three leaves two tables holding
//! the transaction and one not, with no error to anyone and nothing in the
//! recovered database that says so. `Wal::prepare`/`Wal::decide` fix it by
//! putting the whole transaction's fate in one record in one file, and this
//! file is the check that the fix is *reachable* -- that a `BEGIN` typed at the
//! engine's own front door really does get it.
//!
//! ## Two crash oracles, because one of them cannot be trusted to aim
//!
//! **The prefix sweep** (`every_prefix_of_a_three_table_commit_...`) is the
//! one that proves things. The commit sequence is a series of appends in
//! program order -- table 1's marker, then table 2's, then table 3's -- so the
//! set of states a crash can leave is exactly the set of *prefixes* of that
//! byte stream. Building the committed directory once and then cutting each
//! log back to each offset in turn reproduces every one of them, at every
//! byte, deterministically and with no scheduler involved. It is the same
//! technique `stress_crash.rs` uses on a single log, widened to the three
//! files a transaction spans.
//!
//! **The kill sweep** (`killing_a_three_table_commit_...`) is the one that
//! proves it is really the engine doing this and not a test fixture. A child
//! process runs the transaction through the CLI and is `kill -9`ed at a
//! randomised instant inside the commit; the parent then reopens the directory
//! in a fresh process and asks the three tables what happened. It is
//! probabilistic about *where* it lands, which is why it is not the primary
//! oracle -- but it is a real process dying in a real commit, and a fixture
//! that has quietly stopped exercising the window cannot fake it.
//!
//! Every test prints its seed and its kill point, so a failure reproduces.
//!
//! ## Why the assertion is "all three agree" and not "all three committed"
//!
//! Either answer is correct after a crash: the transaction is entitled to be
//! lost right up until the last `fsync` returns, and entitled to be present
//! from that instant on. What is never correct is *disagreement*, and that is
//! what is asserted. The sweep separately checks that it saw both answers, so
//! a run in which nothing ever committed -- which would make "they agree"
//! trivially true -- fails instead of passing quietly.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use granular::{Session, Value};

// ---------------------------------------------------------------- fixtures

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "granular-txn-{}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed),
            tag
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Scratch(d)
    }
    fn path(&self) -> &Path {
        &self.0
    }
    fn s(&self) -> &str {
        self.0.to_str().unwrap()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_granular")
}

fn scale() -> u64 {
    std::env::var("GRANULAR_STRESS").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
}

fn seed(tag: &str) -> u64 {
    let s = std::env::var("GRANULAR_STRESS_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0x2545_F491_4F6C_DD1D);
    eprintln!("txn_multitable::{tag}: GRANULAR_STRESS_SEED={s} GRANULAR_STRESS={}", scale());
    s
}

struct Rng(u64);

impl Rng {
    fn new(s: u64) -> Rng {
        Rng(s | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// The three tables a transaction here spans.
const TABLES: [&str; 3] = ["a", "b", "c"];

fn ddl(t: &str) -> String {
    format!("CREATE TABLE {t} (id UInt64, s String) ENGINE = MergeTree ORDER BY id PRIMARY KEY id")
}

fn wal_of(dir: &Path, t: &str) -> PathBuf {
    dir.join("default").join(t).join("wal.log")
}

/// `SELECT count(*)` from each of the three tables, in a **new** session over
/// `dir`, so recovery has to run to answer.
fn recovered(dir: &Path) -> Result<[u64; 3], String> {
    let mut s = Session::open(dir).map_err(|e| e.to_string())?;
    let mut out = [0u64; 3];
    for (i, t) in TABLES.iter().enumerate() {
        let rs =
            s.query(&format!("SELECT count(*) FROM {t}")).map_err(|e| format!("{t}: {e}"))?;
        out[i] = match rs.to_values().first().and_then(|r| r.first()) {
            Some(Value::UInt(n)) => *n,
            Some(Value::Int(n)) => *n as u64,
            other => return Err(format!("count(*) on {t} came back as {other:?}")),
        };
    }
    Ok(out)
}

/// The assertion the whole file exists for. `n` is what each table holds after
/// recovery; they have to agree, whatever they agree on.
fn assert_atomic(n: [u64; 3], base: u64, ctx: &str) -> bool {
    assert!(
        n[0] == n[1] && n[1] == n[2],
        "{ctx}: a transaction committed a PREFIX of itself -- \
         a={} b={} c={} rows, and every table was written exactly once by it",
        n[0],
        n[1],
        n[2]
    );
    assert!(
        n[0] == base || n[0] == base + 1,
        "{ctx}: {} rows per table, expected {base} (rolled back) or {} (committed)",
        n[0],
        base + 1
    );
    n[0] == base + 1
}

// ------------------------------------------------------- the prefix sweep

/// The state of the three logs on either side of one three-table COMMIT.
struct Committed {
    dir: Scratch,
    /// Length of each log when the commit sequence started, i.e. with the
    /// transaction's rows staged and no marker written yet.
    pre: [u64; 3],
    /// The whole of each log once the commit returned.
    post: [Vec<u8>; 3],
    /// Rows per table before the transaction.
    base: u64,
}

/// Run a three-table transaction to a successful COMMIT and keep the logs.
///
/// In process and dropped without a checkpoint, for the same reason
/// `stress_crash.rs` does it: `Session` has no `Drop` checkpoint, but the CLI
/// checkpoints at exit and would fold the very records under test into parts.
fn commit_three_tables(base: u64) -> Committed {
    let dir = Scratch::new("prefix");
    let mut s = Session::open(dir.path()).unwrap();
    for t in TABLES {
        s.execute(&ddl(t)).unwrap();
        for i in 0..base {
            s.execute(&format!("INSERT INTO {t} VALUES ({i},'base')")).unwrap();
        }
    }
    // Fold the setup into parts so the logs hold the transaction and nothing
    // else -- the sweep is over the commit, not over the fixture.
    s.checkpoint().unwrap();

    s.execute("BEGIN").unwrap();
    for t in TABLES {
        s.execute(&format!("INSERT INTO {t} VALUES ({base},'txn')")).unwrap();
    }
    let mut pre = [0u64; 3];
    for (i, t) in TABLES.iter().enumerate() {
        pre[i] = std::fs::metadata(wal_of(dir.path(), t)).unwrap().len();
    }
    s.execute("COMMIT").unwrap();

    let post = TABLES.map(|t| std::fs::read(wal_of(dir.path(), t)).unwrap());
    drop(s);
    for (i, t) in TABLES.iter().enumerate() {
        assert!(
            post[i].len() as u64 > pre[i],
            "{t}: COMMIT wrote nothing to the log, so this sweeps nothing"
        );
    }
    Committed { dir, pre, post, base }
}

/// Every crash point in the commit sequence, at every byte, over all three
/// logs at once.
///
/// The commit appends to table 1's log, then table 2's, then table 3's, so a
/// crash leaves a prefix of that stream: the logs before the one being written
/// are whole, the one being written is cut somewhere, and the ones after it
/// are still at their pre-commit length. Each such state is reconstructed in a
/// copy of the directory and handed to a fresh `Session`.
///
/// This is the test that fails against the pre-two-phase engine, and it fails
/// at the first byte of table 2's marker: table 1 has committed and table 2
/// has not.
#[test]
fn every_prefix_of_a_three_table_commit_is_all_or_nothing() {
    seed("every_prefix_of_a_three_table_commit_is_all_or_nothing");
    let c = commit_three_tables(3);
    let step = if scale() >= 4 { 1 } else { 3 };
    let (mut saw_commit, mut saw_abort, mut points) = (false, false, 0usize);

    for i in 0..3 {
        let mut cuts: Vec<usize> = (c.pre[i] as usize..c.post[i].len()).step_by(step).collect();
        cuts.push(c.post[i].len());
        for cut in cuts {
            let dst = Scratch::new("cut");
            copy_tree(c.dir.path(), dst.path());
            for (j, t) in TABLES.iter().enumerate() {
                // Logs before this one are whole, this one is cut, the ones
                // after it never got their marker.
                let n = match j.cmp(&i) {
                    std::cmp::Ordering::Less => c.post[j].len(),
                    std::cmp::Ordering::Equal => cut,
                    std::cmp::Ordering::Greater => c.pre[j] as usize,
                };
                std::fs::write(wal_of(dst.path(), t), &c.post[j][..n]).unwrap();
            }
            points += 1;
            let ctx = format!("log {} of 3 cut to {cut} of {}", i + 1, c.post[i].len());
            match recovered(dst.path()) {
                // A refusal is allowed -- damage that is not a torn tail is
                // documented as reported rather than swallowed. A disagreement
                // is not, and neither is a panic.
                Err(e) => assert!(
                    e.contains("corrupt") || e.contains("checksum"),
                    "{ctx}: recovery failed with something other than corruption: {e}"
                ),
                Ok(n) => {
                    if assert_atomic(n, c.base, &ctx) {
                        saw_commit = true;
                    } else {
                        saw_abort = true;
                    }
                }
            }
        }
    }
    assert!(points > 3, "the sweep covered {points} points, which is not a sweep");
    assert!(saw_abort, "no cut left the transaction rolled back");
    assert!(saw_commit, "no cut left the transaction committed -- the sweep never crossed it");
    eprintln!("  swept {points} crash points across 3 logs, step {step}");
}

/// Copy a data directory, skipping the lock and any in-flight temp file.
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap().flatten() {
        let name = e.file_name();
        let n = name.to_string_lossy();
        if n == "LOCK" || n.contains(".tmp-") {
            continue;
        }
        let (src, dst) = (e.path(), to.join(&name));
        if e.file_type().unwrap().is_dir() {
            copy_tree(&src, &dst);
        } else {
            std::fs::copy(&src, &dst).unwrap();
        }
    }
}

// ------------------------------------------------ the coordinator recycled

/// The other way a two-phase commit can lose data, and it needs no crash at
/// all: the log holding the decision gets recycled while somebody is still
/// citing it.
///
/// A positional sweep -- `ALTER ... DELETE` on a table with no single-column
/// primary key -- is a mutation the log cannot describe, so the engine makes
/// it durable by writing that *one* table's parts and truncating that one
/// table's log. If the table it does that to happens to be the coordinator of
/// an earlier transaction, and the truncation took the decision with it, the
/// other two tables' prepares become unresolvable -- which reads exactly like
/// "the decision was never written", which is an abort, which silently drops
/// rows that were committed and acknowledged.
///
/// So: commit across three tables, sweep the coordinator, reopen, and expect
/// all three to still hold the transaction.
#[test]
fn recycling_the_coordinators_log_keeps_the_transaction() {
    seed("recycling_the_coordinators_log_keeps_the_transaction");
    let dir = Scratch::new("recycle");
    {
        let mut s = Session::open(dir.path()).unwrap();
        s.execute(&ddl("a")).unwrap();
        s.execute(&ddl("b")).unwrap();
        // No PRIMARY KEY, so a DELETE on it is a positional sweep -- and it is
        // enlisted last, which makes it the transaction's coordinator.
        s.execute("CREATE TABLE c (id UInt64, s String) ENGINE = MergeTree ORDER BY id").unwrap();
        // A row for the sweep to find later: a sweep that deletes nothing is
        // not a mutation and does not fold.
        s.execute("INSERT INTO c VALUES (1,'sweep me')").unwrap();
        s.execute("BEGIN").unwrap();
        for t in TABLES {
            s.execute(&format!("INSERT INTO {t} VALUES (7,'txn')")).unwrap();
        }
        s.execute("COMMIT").unwrap();

        // Folds `c` into parts and recycles its log, on its own, with `a` and
        // `b` still holding staged records and a prepare each.
        let before = std::fs::metadata(wal_of(dir.path(), "c")).unwrap().len();
        s.execute("ALTER TABLE c DELETE WHERE id = 1").unwrap();
        let after = std::fs::metadata(wal_of(dir.path(), "c")).unwrap().len();
        assert!(
            after < before,
            "the sweep did not recycle the coordinator's log ({before} -> {after}), \
             so this tests nothing"
        );
        assert!(
            after > header_len(),
            "the decision was recycled along with the log `a` and `b` are citing"
        );
    }
    let n = recovered(dir.path()).expect("reopen after the sweep");
    assert_atomic(n, 0, "after the coordinator's log was recycled");
    assert_eq!(n, [1, 1, 1], "the committed transaction was dropped by a later fold");
}

/// ...and the same thing once every log has been folded, which is the state a
/// full `CHECKPOINT` leaves and the point at which the decisions may finally
/// go. The rows are in parts by then, so the only way to fail this is to lose
/// them on the way.
#[test]
fn a_checkpoint_after_a_multi_table_commit_keeps_every_table() {
    seed("a_checkpoint_after_a_multi_table_commit_keeps_every_table");
    let dir = Scratch::new("ckpt");
    {
        let mut s = Session::open(dir.path()).unwrap();
        for t in TABLES {
            s.execute(&ddl(t)).unwrap();
        }
        s.execute("BEGIN").unwrap();
        for t in TABLES {
            s.execute(&format!("INSERT INTO {t} VALUES (7,'txn')")).unwrap();
        }
        s.execute("COMMIT").unwrap();
        s.checkpoint().unwrap();
        for t in TABLES {
            assert_eq!(
                std::fs::metadata(wal_of(dir.path(), t)).unwrap().len(),
                header_len(),
                "{t}: a checkpoint with nothing left to cite must reclaim the whole log"
            );
        }
    }
    assert_eq!(recovered(dir.path()).unwrap(), [1, 1, 1]);
}

// --------------------------------------------------------- the kill sweep

/// `kill -9` a real child in the middle of a real three-table COMMIT.
///
/// The child is driven over a pipe so the parent can put it in the state it
/// wants and then let it commit: DDL and the baseline first, then `BEGIN` and
/// three `INSERT`s, and the parent waits for all three logs to grow, which is
/// the only signal it needs that the rows are staged and nothing has been
/// marked yet. Then `COMMIT` goes down the pipe and a timer starts.
///
/// The kill point is randomised over a window calibrated from a run that was
/// allowed to finish, so the sweep spans the whole commit rather than piling
/// up in its longest phase. Trials that miss -- the child died before the
/// commit started, or after it ended -- are still checked (they must agree
/// too) and counted separately, because a run in which every trial missed
/// proves nothing and says so.
#[test]
fn killing_a_three_table_commit_leaves_all_of_it_or_none() {
    let mut rng = Rng::new(seed("killing_a_three_table_commit_leaves_all_of_it_or_none"));
    const BASE: u64 = 2;
    // One uninterrupted run to find out how long a commit takes here; this
    // machine's fsync is worth milliseconds and a fixed sleep would be a guess.
    let window = calibrate(BASE).max(Duration::from_micros(200));
    let trials = 24 * scale();
    let (mut committed, mut aborted) = (0u32, 0u32);

    for k in 0..trials {
        let dir = Scratch::new("kill");
        let mut child = staged_child(&dir, BASE);
        // Somewhere in [0, 2x] the commit's own duration: short enough to land
        // between two markers, long enough to reach the last one.
        let at = Duration::from_nanos((rng.next() % (2 * window.as_nanos() as u64 + 1)).max(1));
        let live = {
            use std::io::Write;
            child.stdin.as_mut().unwrap().write_all(b"COMMIT;\n").unwrap();
            child.stdin.as_mut().unwrap().flush().unwrap();
            std::thread::sleep(at);
            let live = child.try_wait().unwrap().is_none();
            let _ = child.kill();
            let _ = child.wait();
            live
        };
        let ctx = format!("trial {k}: killed {at:?} into COMMIT (window {window:?}, live={live})");
        match recovered(dir.path()) {
            Err(e) => panic!("{ctx}: the directory would not reopen: {e}"),
            Ok(n) => {
                if assert_atomic(n, BASE, &ctx) {
                    committed += 1;
                } else {
                    aborted += 1;
                }
            }
        }
    }
    eprintln!("  {trials} kills: {committed} committed, {aborted} rolled back, none split");
    // Not an assertion on the split -- either answer is legal at any instant.
    // What would be worthless is a run that never got near the boundary, and
    // that shows up as every trial landing on the same side.
    assert!(
        committed > 0 && aborted > 0,
        "every one of {trials} kills landed on the same side of the commit \
         ({committed} committed, {aborted} rolled back): the window was never crossed, \
         so this proved nothing. Raise GRANULAR_STRESS."
    );
}

/// A child with the three tables created, the baseline checkpointed, and a
/// transaction open with one staged row in each table -- stopped there, with
/// `COMMIT` still to come down the pipe.
fn staged_child(dir: &Scratch, base: u64) -> Child {
    // Setup in its own process, so its exit checkpoint folds the baseline into
    // parts and the logs under test hold only the transaction.
    let mut setup = String::new();
    for t in TABLES {
        setup.push_str(&ddl(t));
        setup.push_str(";\n");
        for i in 0..base {
            setup.push_str(&format!("INSERT INTO {t} VALUES ({i},'base');\n"));
        }
    }
    let out = Command::new(bin())
        .args(["--data", dir.s(), "-q", &setup])
        .output()
        .expect("spawn granular");
    assert!(out.status.success(), "setup: {}", String::from_utf8_lossy(&out.stderr));

    let mut child = Command::new(bin())
        .args(["--data", dir.s()])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn granular");
    {
        use std::io::Write;
        let w = child.stdin.as_mut().unwrap();
        w.write_all(b"BEGIN;\n").unwrap();
        for t in TABLES {
            writeln!(w, "INSERT INTO {t} VALUES ({base},'txn');").unwrap();
        }
        w.flush().unwrap();
    }
    // Every log holding a staged record is the state "the rows are down, no
    // marker is". Polling the directory rather than stdout because the CLI
    // wraps stdout in a 64 KiB buffer that a handful of statements never fills.
    let t0 = Instant::now();
    while TABLES.iter().any(|t| {
        std::fs::metadata(wal_of(dir.path(), t)).map_or(true, |m| m.len() <= header_len())
    }) {
        assert!(t0.elapsed() < Duration::from_secs(20), "the child never staged its inserts");
        std::thread::sleep(Duration::from_micros(200));
    }
    child
}

/// How long one three-table COMMIT takes on this machine, best of three.
fn calibrate(base: u64) -> Duration {
    let mut best = Duration::from_secs(9999);
    for _ in 0..3 {
        let dir = Scratch::new("calib");
        let mut child = staged_child(&dir, base);
        let t0 = Instant::now();
        {
            use std::io::Write;
            let w = child.stdin.as_mut().unwrap();
            w.write_all(b"COMMIT;\n").unwrap();
            w.flush().unwrap();
        }
        // The commit is over once the last log has its marker.
        let last = wal_of(dir.path(), TABLES[TABLES.len() - 1]);
        let before = std::fs::metadata(&last).unwrap().len();
        while std::fs::metadata(&last).map_or(0, |m| m.len()) == before {
            assert!(t0.elapsed() < Duration::from_secs(20), "the calibration commit never landed");
            std::hint::spin_loop();
        }
        best = best.min(t0.elapsed());
        let _ = child.kill();
        let _ = child.wait();
    }
    best
}

/// The log header the engine writes before any record; a file this size holds
/// nothing. Read from a throwaway directory rather than hard-coded, so a
/// format change cannot quietly turn the wait above into a no-op.
fn header_len() -> u64 {
    let dir = Scratch::new("hdr");
    let mut s = Session::open(dir.path()).unwrap();
    s.execute(&ddl("a")).unwrap();
    s.execute("INSERT INTO a VALUES (0,'x')").unwrap();
    s.checkpoint().unwrap();
    std::fs::metadata(wal_of(dir.path(), "a")).unwrap().len()
}
