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

/// The three tables most of the transactions here span.
const TABLES: [&str; 3] = ["a", "b", "c"];

/// Four, for the tests about the *shape* of the commit rather than its
/// atomicity. Three tables mean two prepares; four mean three, which is the
/// first width at which the prepares' barriers are neither "the only one" nor
/// "a pair" -- and the width at which issuing them concurrently instead of one
/// after another is worth anything. See `commit_durable`.
const FOUR: [&str; 4] = ["a", "b", "c", "d"];

fn ddl(t: &str) -> String {
    format!("CREATE TABLE {t} (id UInt64, s String) ENGINE = MergeTree ORDER BY id PRIMARY KEY id")
}

/// A table's **active** log segment.
///
/// The log is a directory of numbered segments under `<root>/.wal/<db>/<t>`,
/// and only the highest-numbered one is appended to; a checkpoint seals that
/// one where it stands and starts the next. Every reading here -- "did the
/// COMMIT write anything", "did the fold recycle the log" -- is about the file
/// a writer is actually using, which is this one.
fn wal_of(dir: &Path, t: &str) -> PathBuf {
    let d = dir.join(".wal").join("default").join(t);
    let mut names: Vec<String> = std::fs::read_dir(&d)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.ends_with(".gwal"))
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    d.join(names.pop().unwrap_or_else(|| "none.gwal".into()))
}

/// `SELECT count(*)` from each of the three tables, in a **new** session over
/// `dir`, so recovery has to run to answer.
fn recovered(dir: &Path) -> Result<[u64; 3], String> {
    let n = counts(dir, &TABLES)?;
    Ok([n[0], n[1], n[2]])
}

/// The same over any set of tables. One session, so one recovery.
fn counts(dir: &Path, tables: &[&str]) -> Result<Vec<u64>, String> {
    let mut s = Session::open(dir).map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(tables.len());
    for t in tables {
        let rs =
            s.query(&format!("SELECT count(*) FROM {t}")).map_err(|e| format!("{t}: {e}"))?;
        out.push(match rs.to_values().first().and_then(|r| r.first()) {
            Some(Value::UInt(n)) => *n,
            Some(Value::Int(n)) => *n as u64,
            other => return Err(format!("count(*) on {t} came back as {other:?}")),
        });
    }
    Ok(out)
}

/// The assertion the whole file exists for. `n` is what each table holds after
/// recovery; they have to agree, whatever they agree on.
fn assert_atomic(n: [u64; 3], base: u64, ctx: &str) -> bool {
    assert_agreed(&n, &TABLES, base, ctx)
}

/// [`assert_atomic`] over any width.
fn assert_agreed(n: &[u64], tables: &[&str], base: u64, ctx: &str) -> bool {
    assert!(
        n.iter().all(|k| *k == n[0]),
        "{ctx}: a transaction committed a PREFIX of itself -- {}, \
         and every table was written exactly once by it",
        tables
            .iter()
            .zip(n)
            .map(|(t, k)| format!("{t}={k}"))
            .collect::<Vec<_>>()
            .join(" ")
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

/// The state of the logs on either side of one multi-table COMMIT.
struct Committed<'a> {
    dir: Scratch,
    tables: &'a [&'a str],
    /// Length of each log when the commit sequence started, i.e. with the
    /// transaction's rows staged and no marker written yet.
    pre: Vec<u64>,
    /// The whole of each log once the commit returned.
    post: Vec<Vec<u8>>,
    /// Rows per table before the transaction.
    base: u64,
}

/// Run a transaction across `tables` to a successful COMMIT and keep the logs.
///
/// In process and dropped without a checkpoint, for the same reason
/// `stress_crash.rs` does it: `Session` has no `Drop` checkpoint, but the CLI
/// checkpoints at exit and would fold the very records under test into parts.
fn commit_tables<'a>(tables: &'a [&'a str], base: u64) -> Committed<'a> {
    let dir = Scratch::new("prefix");
    let mut s = Session::open(dir.path()).unwrap();
    for t in tables {
        s.execute(&ddl(t)).unwrap();
        for i in 0..base {
            s.execute(&format!("INSERT INTO {t} VALUES ({i},'base')")).unwrap();
        }
    }
    // Fold the setup into parts so the logs hold the transaction and nothing
    // else -- the sweep is over the commit, not over the fixture.
    s.checkpoint().unwrap();

    s.execute("BEGIN").unwrap();
    for t in tables {
        s.execute(&format!("INSERT INTO {t} VALUES ({base},'txn')")).unwrap();
    }
    let pre: Vec<u64> =
        tables.iter().map(|t| std::fs::metadata(wal_of(dir.path(), t)).unwrap().len()).collect();
    s.execute("COMMIT").unwrap();

    let post: Vec<Vec<u8>> =
        tables.iter().map(|t| std::fs::read(wal_of(dir.path(), t)).unwrap()).collect();
    drop(s);
    for (i, t) in tables.iter().enumerate() {
        assert!(
            post[i].len() as u64 > pre[i],
            "{t}: COMMIT wrote nothing to the log, so this sweeps nothing"
        );
    }
    Committed { dir, tables, pre, post, base }
}

/// Reconstruct every crash point of `c`'s commit sequence and assert each one
/// is all or nothing.
///
/// The commit appends to log 1, then log 2, ... so a crash leaves a prefix of
/// that stream: the logs before the one being written are whole, the one being
/// written is cut somewhere, and the ones after it are still at their
/// pre-commit length.
fn sweep_every_prefix(c: &Committed<'_>) {
    let step = if scale() >= 4 { 1 } else { 3 };
    let (mut saw_commit, mut saw_abort, mut points) = (false, false, 0usize);
    let width = c.tables.len();

    for i in 0..width {
        let mut cuts: Vec<usize> = (c.pre[i] as usize..c.post[i].len()).step_by(step).collect();
        cuts.push(c.post[i].len());
        for cut in cuts {
            let dst = Scratch::new("cut");
            copy_tree(c.dir.path(), dst.path());
            for (j, t) in c.tables.iter().enumerate() {
                let n = match j.cmp(&i) {
                    std::cmp::Ordering::Less => c.post[j].len(),
                    std::cmp::Ordering::Equal => cut,
                    std::cmp::Ordering::Greater => c.pre[j] as usize,
                };
                std::fs::write(wal_of(dst.path(), t), &c.post[j][..n]).unwrap();
            }
            points += 1;
            let ctx = format!("log {} of {width} cut to {cut} of {}", i + 1, c.post[i].len());
            match counts(dst.path(), c.tables) {
                // A refusal is allowed -- damage that is not a torn tail is
                // documented as reported rather than swallowed. A disagreement
                // is not, and neither is a panic.
                Err(e) => assert!(
                    e.contains("corrupt") || e.contains("checksum"),
                    "{ctx}: recovery failed with something other than corruption: {e}"
                ),
                Ok(n) => {
                    if assert_agreed(&n, c.tables, c.base, &ctx) {
                        saw_commit = true;
                    } else {
                        saw_abort = true;
                    }
                }
            }
        }
    }
    assert!(points > width, "the sweep covered {points} points, which is not a sweep");
    assert!(saw_abort, "no cut left the transaction rolled back");
    assert!(saw_commit, "no cut left the transaction committed -- the sweep never crossed it");
    eprintln!("  swept {points} crash points across {width} logs, step {step}");
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
    sweep_every_prefix(&commit_tables(&TABLES, 3));
}

/// The same sweep one table wider, which is where the prepares stopped being
/// issued one after another.
///
/// Three tables mean two prepares; four mean three, and three barriers are
/// issued **concurrently** and joined before the decision is appended. The
/// reordering is real -- every prepare record is now written before any of
/// them is `fsync`ed, where before each was written and flushed in turn -- so
/// the set of byte-level crash states is not the same set the three-table
/// sweep covers, and it gets its own sweep rather than an argument.
///
/// What is unchanged, and what this asserts, is the outcome: a crash anywhere
/// in the sequence leaves all four tables agreeing.
#[test]
fn every_prefix_of_a_four_table_commit_is_all_or_nothing() {
    seed("every_prefix_of_a_four_table_commit_is_all_or_nothing");
    sweep_every_prefix(&commit_tables(&FOUR, 3));
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

/// The state the sweep above leaves -- a segment whose header carries
/// decisions forward -- has to survive being *checkpointed*, and then being
/// opened again, and again.
///
/// The previous test reopens once, and once was not enough. A carried decision
/// is the only thing that makes a segment header's `carry_len` non-zero, and
/// every path that reads a header without the segment behind it read such a
/// header as damage: the checkpoint could not measure the log, so it recorded
/// a watermark of **0**, and the next open replayed the whole stream on top of
/// the parts that already held it. Measured on the shipped binary: the
/// coordinator went 1 -> 15 -> 17 -> 19 rows over successive opens, with a
/// `DELETE`d row back among them, `system.wal` reporting `segments = 0`, and
/// `RESTORE ... UNTIL LATEST` refusing the healthy database as corrupt.
///
/// So: reach that state, checkpoint it, and then assert the two things that
/// were false -- the counts do not move across repeated opens, and the archive
/// still describes itself.
#[test]
fn a_carried_decision_survives_the_checkpoint_that_measures_it() {
    seed("a_carried_decision_survives_the_checkpoint_that_measures_it");
    let dir = Scratch::new("carry-open");
    {
        let mut s = Session::open(dir.path()).unwrap();
        s.execute(&ddl("a")).unwrap();
        s.execute(&ddl("b")).unwrap();
        // No PRIMARY KEY, so `ALTER ... DELETE` on it is a positional sweep:
        // one table's parts written and one table's log rolled, with `a` and
        // `b` still holding a prepare each. That roll is what carries.
        s.execute("CREATE TABLE c (id UInt64, s String) ENGINE = MergeTree ORDER BY id")
            .unwrap();
        s.execute("INSERT INTO c VALUES (1,'sweep me')").unwrap();
        s.execute("BEGIN").unwrap();
        for t in TABLES {
            s.execute(&format!("INSERT INTO {t} VALUES (7,'txn')")).unwrap();
        }
        s.execute("COMMIT").unwrap();
        s.execute("ALTER TABLE c DELETE WHERE id = 1").unwrap();
        // The checkpoint the CLI runs on the way out, which is where the
        // watermark is recorded.
        s.checkpoint().unwrap();
    }
    let carry = {
        let seg = std::fs::read(wal_of(dir.path(), "c")).unwrap();
        u32::from_le_bytes(seg[12..16].try_into().unwrap())
    };
    assert!(carry > 0, "the fixture carried no decision, so this proves nothing");

    // Three opens, not one. Each is a fresh `Session` over the same directory,
    // so a watermark that failed to advance shows up as growth.
    let first = recovered(dir.path()).expect("reopen after the sweep");
    assert_eq!(first, [1, 1, 1], "the transaction is in all three tables exactly once");
    for i in 2..=3 {
        assert_eq!(recovered(dir.path()).unwrap(), first, "open {i} replayed an already-folded log");
    }

    // ...and the archive can still be read, which is what a recovery does
    // before it unpacks anything.
    let segs = granular::persist::wal::segments(dir.path(), "default", "c")
        .expect("a carried decision is not a hole in the archive");
    assert!(!segs.is_empty(), "the sweep rolled, so `c` has an archive");
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
    let window = calibrate(&TABLES, BASE).max(Duration::from_micros(200));
    let trials = 24 * scale();
    let (mut committed, mut aborted) = (0u32, 0u32);

    for k in 0..trials {
        let dir = Scratch::new("kill");
        let mut child = staged_child(&dir, &TABLES, BASE);
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

/// The same kill sweep at four tables, where three barriers overlap.
///
/// Same oracle, one width up, and it is the width that matters: at three
/// tables the concurrent join has two participants, at four it has three, and
/// three is the first count at which the pool really distributes the work
/// rather than handing the caller its own share and one other.
///
/// Half the trials of the three-table sweep, because the point here is the
/// wider commit and not a second helping of the same statistics.
#[test]
fn killing_a_four_table_commit_leaves_all_of_it_or_none() {
    let mut rng = Rng::new(seed("killing_a_four_table_commit_leaves_all_of_it_or_none"));
    const BASE: u64 = 2;
    let window = calibrate(&FOUR, BASE).max(Duration::from_micros(200));
    let trials = 12 * scale();
    let (mut committed, mut aborted) = (0u32, 0u32);

    for k in 0..trials {
        let dir = Scratch::new("kill4");
        let mut child = staged_child(&dir, &FOUR, BASE);
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
        match counts(dir.path(), &FOUR) {
            Err(e) => panic!("{ctx}: the directory would not reopen: {e}"),
            Ok(n) => {
                if assert_agreed(&n, &FOUR, BASE, &ctx) {
                    committed += 1;
                } else {
                    aborted += 1;
                }
            }
        }
    }
    eprintln!("  {trials} kills across 4 tables: {committed} committed, {aborted} rolled back");
    assert!(
        committed > 0 && aborted > 0,
        "every one of {trials} kills landed on the same side of the commit \
         ({committed} committed, {aborted} rolled back): the window was never crossed, \
         so this proved nothing. Raise GRANULAR_STRESS."
    );
}

/// A four-table transaction typed at the CLI really takes the two-phase path,
/// and the whole of it is durable after the process that wrote it is gone.
///
/// The barrier change lives one function below the front door, so the thing
/// worth asserting from outside is that the front door still reaches it: three
/// participants each holding a prepare that names the fourth, a decision in
/// the fourth, and four tables that agree in a process that never saw the
/// transaction run.
///
/// The log bytes are read *before* the child exits, because the CLI
/// checkpoints on the way out and would fold the records under test into
/// parts. `staged_child` leaves the child stopped with the rows staged; the
/// COMMIT goes down the pipe and the parent waits for the coordinator's log to
/// grow, which is the commit's own completion signal.
#[test]
fn a_four_table_commit_through_the_cli_prepares_three_and_decides_once() {
    seed("a_four_table_commit_through_the_cli_prepares_three_and_decides_once");
    const BASE: u64 = 2;
    let dir = Scratch::new("cli4");
    let mut child = staged_child(&dir, &FOUR, BASE);
    let pre: Vec<u64> =
        FOUR.iter().map(|t| std::fs::metadata(wal_of(dir.path(), t)).unwrap().len()).collect();
    {
        use std::io::Write;
        let w = child.stdin.as_mut().unwrap();
        w.write_all(b"COMMIT;\n").unwrap();
        w.flush().unwrap();
    }
    // The commit is over once the coordinator -- the last table enlisted --
    // has its decision.
    let last = wal_of(dir.path(), FOUR[FOUR.len() - 1]);
    let t0 = Instant::now();
    while std::fs::metadata(&last).map_or(0, |m| m.len()) == pre[FOUR.len() - 1] {
        assert!(t0.elapsed() < Duration::from_secs(20), "the four-table COMMIT never landed");
        std::thread::sleep(Duration::from_micros(200));
    }
    for (i, t) in FOUR.iter().enumerate() {
        let now = std::fs::metadata(wal_of(dir.path(), t)).unwrap().len();
        assert!(
            now > pre[i],
            "{t}: COMMIT wrote nothing to its log, so this transaction was not four-table"
        );
    }
    // Kill rather than close: a clean exit checkpoints, and then the assertion
    // below would be about parts instead of about recovery.
    let _ = child.kill();
    let _ = child.wait();

    let n = counts(dir.path(), &FOUR).expect("the directory must reopen");
    assert_eq!(
        n,
        vec![BASE + 1; FOUR.len()],
        "a COMMIT that returned did not survive the process that ran it"
    );
}

/// A child with the tables created, the baseline checkpointed, and a
/// transaction open with one staged row in each table -- stopped there, with
/// `COMMIT` still to come down the pipe.
fn staged_child(dir: &Scratch, tables: &[&str], base: u64) -> Child {
    // Setup in its own process, so its exit checkpoint folds the baseline into
    // parts and the logs under test hold only the transaction.
    let mut setup = String::new();
    for t in tables {
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
        for t in tables {
            writeln!(w, "INSERT INTO {t} VALUES ({base},'txn');").unwrap();
        }
        w.flush().unwrap();
    }
    // Every log holding a staged record is the state "the rows are down, no
    // marker is". Polling the directory rather than stdout because the CLI
    // wraps stdout in a 64 KiB buffer that a handful of statements never fills.
    let t0 = Instant::now();
    while tables.iter().any(|t| {
        std::fs::metadata(wal_of(dir.path(), t)).map_or(true, |m| m.len() <= header_len())
    }) {
        assert!(t0.elapsed() < Duration::from_secs(20), "the child never staged its inserts");
        std::thread::sleep(Duration::from_micros(200));
    }
    child
}

/// How long one COMMIT across `tables` takes on this machine, best of three.
fn calibrate(tables: &[&str], base: u64) -> Duration {
    let mut best = Duration::from_secs(9999);
    for _ in 0..3 {
        let dir = Scratch::new("calib");
        let mut child = staged_child(&dir, tables, base);
        let t0 = Instant::now();
        {
            use std::io::Write;
            let w = child.stdin.as_mut().unwrap();
            w.write_all(b"COMMIT;\n").unwrap();
            w.flush().unwrap();
        }
        // The commit is over once the last log has its marker.
        let last = wal_of(dir.path(), tables[tables.len() - 1]);
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

// ------------------------------------------- two commit points in one COMMIT

/// A transaction may hold at most one commit point, and a folding table brings
/// its own.
///
/// ## The hole this closes, and how it was found
///
/// An unkeyed `DELETE`/`UPDATE` that hides rows living in a part with no
/// durable home cannot be described by any log record -- there is nothing to
/// name the rows with -- so the engine makes it durable by writing that
/// table's parts out at `COMMIT`. `Session::commit` runs those folds *after*
/// `Session::commit_durable` has already returned `Ok`, and `commit_durable`
/// excludes folding tables from the two-phase protocol outright (its
/// coordinator is `rposition(|e| !e.fold && e.seq.is_some())` and `parties` is
/// counted over `!e.fold`). So such a transaction had **two** commit points
/// with a window between them.
///
/// `kill -9` in that window commits half a transaction. Measured on the tree
/// this refusal landed on, 30-round scripts killed at a random instant:
///
/// ```text
///   BEGIN; INSERT INTO k; DELETE FROM u; COMMIT     16 of 25 trials half-committed
///   BEGIN; DELETE FROM u; DELETE FROM u2; COMMIT     8 of 25 trials diverged
/// ```
///
/// The second shape is the worse one: with every participant folding,
/// `commit_durable` finds no participant at all and returns having written no
/// marker and performed no `fsync`, so the two folds are simply two
/// independent checkpoints. Both are zero of 25 with the refusal in place.
///
/// There is no ordering that fixes it. A `TABLE` rename cannot be made
/// conditional on a decision written after it, and the decision cannot be made
/// conditional on a rename replay cannot reproduce -- the rows the fold exists
/// for are exactly the ones no record can name. Writing a marker anyway would
/// release the group's `Insert` records with the sweep's tombstones still only
/// in memory, resurrecting the rows the statement deleted. So it is refused.
///
/// ## What is asserted here
///
/// Both orders reach the rule, and -- the assertion that keeps the refusal
/// honest -- the *citable* case still succeeds. A multi-table transaction
/// whose unkeyed `DELETE` hits rows that are already in a part file logs
/// `TAG_MASK_RUN`, never folds, and must be unaffected. If that stops working
/// the refusal has become a blanket ban on unkeyed deletes in transactions,
/// which is a different and much worse change.
#[test]
fn a_folding_table_may_not_share_a_transaction() {
    let dir = Scratch::new("two-commit-points");
    let mut s = Session::open(dir.path()).unwrap();
    s.execute("CREATE TABLE k (id UInt64, v UInt64) ENGINE = MergeTree PRIMARY KEY id ORDER BY id")
        .unwrap();
    s.execute("CREATE TABLE u (id UInt64, v UInt64) ENGINE = MergeTree ORDER BY id").unwrap();
    s.execute("CREATE TABLE u2 (id UInt64, v UInt64) ENGINE = MergeTree ORDER BY id").unwrap();
    // Checkpointed, so these rows are in part files and deletes against them
    // are citable -- that is the case that must keep working.
    s.execute("INSERT INTO u VALUES (1,1),(2,2),(3,3)").unwrap();
    s.execute("INSERT INTO u2 VALUES (1,1),(2,2),(3,3)").unwrap();
    s.checkpoint().unwrap();

    // ORDER 1: a durable table is enlisted first, then a table wants to fold.
    // Reached in `Session::mark_fold`.
    s.execute("INSERT INTO u VALUES (90,90)").unwrap(); // dark: no durable home
    s.execute("BEGIN").unwrap();
    s.execute("INSERT INTO k VALUES (1,1)").unwrap();
    let e = s.execute("DELETE FROM u WHERE id = 90").unwrap_err().to_string();
    assert!(
        e.contains("default.u") && e.contains("default.k") && e.contains("PRIMARY KEY"),
        "the refusal must name both tables and the fix; got: {e}"
    );
    s.execute("ROLLBACK").unwrap();
    assert_eq!(one(&mut s, "SELECT count() FROM k"), 0, "the refused transaction left rows in `k`");
    assert_eq!(one(&mut s, "SELECT count() FROM u WHERE id = 90"), 1, "row 90 was hidden anyway");

    // ORDER 2: a table is already folding, then another wants in. Reached in
    // `Session::enlist`, which is a different call site and a different scan.
    s.execute("BEGIN").unwrap();
    s.execute("DELETE FROM u WHERE id = 90").unwrap();
    let e = s.execute("INSERT INTO k VALUES (2,2)").unwrap_err().to_string();
    assert!(e.contains("default.u"), "the refusal must name the folding table; got: {e}");
    s.execute("ROLLBACK").unwrap();

    // ORDER 2, both unkeyed: the shape where `commit_durable` used to write
    // nothing at all.
    s.execute("INSERT INTO u VALUES (91,91)").unwrap();
    s.execute("INSERT INTO u2 VALUES (91,91)").unwrap();
    s.execute("BEGIN").unwrap();
    s.execute("DELETE FROM u WHERE id = 91").unwrap();
    assert!(s.execute("DELETE FROM u2 WHERE id = 91").is_err(), "two folding tables were allowed");
    s.execute("ROLLBACK").unwrap();

    // THE NARROWNESS ASSERTION. Same statement shape, citable rows: it must
    // still commit, across two tables, with no fold anywhere.
    s.execute("BEGIN").unwrap();
    s.execute("INSERT INTO k VALUES (3,3)").unwrap();
    s.execute("DELETE FROM u WHERE id = 1").unwrap();
    s.execute("DELETE FROM u2 WHERE id = 2").unwrap();
    s.execute("COMMIT").expect("a multi-table transaction over CITABLE unkeyed rows must commit");
    assert_eq!(one(&mut s, "SELECT count() FROM k"), 1);
    assert_eq!(one(&mut s, "SELECT count() FROM u WHERE id = 1"), 0);
    assert_eq!(one(&mut s, "SELECT count() FROM u2 WHERE id = 2"), 0);

    // And a folding table on its own is still fine -- that shape is atomic and
    // is not what was broken.
    s.execute("BEGIN").unwrap();
    s.execute("DELETE FROM u WHERE id = 90").unwrap();
    s.execute("DELETE FROM u WHERE id = 91").unwrap();
    s.execute("COMMIT").expect("a single folding table is atomic on its own");
    assert_eq!(one(&mut s, "SELECT count() FROM u WHERE id > 89"), 0);
}

/// One `UInt64` out of a one-row, one-column result.
fn one(s: &mut Session, q: &str) -> u64 {
    let rs = s.query(q).unwrap();
    match rs.to_values().first().and_then(|r| r.first()) {
        Some(Value::UInt(n)) => *n,
        Some(Value::Int(n)) => *n as u64,
        other => panic!("`{q}` returned {other:?}"),
    }
}

/// The same hole, under a real `kill -9`, through the CLI.
///
/// The in-process test above pins the refusal; this pins the thing the
/// refusal exists for. Every round inserts a row into an unkeyed table (so it
/// is dark -- no part file holds it yet), then opens a transaction that writes
/// the keyed table and deletes that row. Before the refusal, the keyed insert
/// was durable and the unkeyed delete was not, in 16 of 25 trials; the id
/// appearing in **both** tables is the proof, because the same transaction
/// wrote one and removed the other.
///
/// Six trials rather than twenty-five: the shape reproduced at ~64% per trial,
/// so six is a 1-in-1400 chance of a run that would have missed it, and this
/// belongs in a `cargo test` that people actually run. `GRANULAR_STRESS`
/// multiplies it.
#[test]
fn killing_a_transaction_that_mixes_a_folding_table_commits_all_or_none() {
    let s0 = seed("killing_a_transaction_that_mixes_a_folding_table_commits_all_or_none");
    let trials = 6 * scale();
    const ROUNDS: u64 = 30;

    let mut body = String::new();
    for r in 0..ROUNDS {
        let i = 1000 + r;
        // Its own statement, so the row is in a runtime part with no durable
        // home by the time the transaction below sweeps it.
        body.push_str(&format!("INSERT INTO u VALUES ({i},{i});\n"));
        body.push_str(&format!(
            "BEGIN; INSERT INTO k VALUES ({i},{i}); DELETE FROM u WHERE id = {i}; COMMIT;\n"
        ));
    }
    let ddl = "CREATE TABLE k (id UInt64, v UInt64) ENGINE = MergeTree PRIMARY KEY id ORDER BY id;\
               CREATE TABLE u (id UInt64, v UInt64) ENGINE = MergeTree ORDER BY id";

    let base = {
        let dir = Scratch::new("cal-fold-mix");
        let p = seed_and_script(&dir, ddl, &body);
        let t0 = Instant::now();
        let _ = script_child(&dir, &p).wait();
        t0.elapsed()
    };

    let mut rng = s0 | 1;
    let mut killed = 0u64;
    for trial in 0..trials {
        let dir = Scratch::new("fold-mix");
        let p = seed_and_script(&dir, ddl, &body);
        // xorshift, inline: this file has no `Rng` of its own and one call
        // site does not earn one.
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        let frac = ((rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64) / (1u64 << 53) as f64;
        let at = base.mul_f64(0.05 + frac * 0.95);
        let mut c = script_child(&dir, &p);
        std::thread::sleep(at);
        let live = c.try_wait().expect("try_wait").is_none();
        if live {
            let _ = c.kill();
        }
        let _ = c.wait();
        killed += live as u64;

        let ks = ids(&dir, "k").unwrap_or_else(|e| panic!("trial {trial}: `k` would not open: {e}"));
        let us = ids(&dir, "u").unwrap_or_else(|e| panic!("trial {trial}: `u` would not open: {e}"));
        let both: Vec<u64> = ks.iter().copied().filter(|i| us.contains(i)).collect();
        assert!(
            both.is_empty(),
            "trial {trial} (kill at {at:?}): {} transaction(s) committed only their keyed half \
             -- ids {both:?} were inserted into `k` and deleted from `u` by the SAME \
             transaction, and the delete is gone",
            both.len()
        );
        // The floor: a row `u` was given and no statement ever removed cannot
        // vanish, so "both tables empty" cannot pass this quietly.
        assert!(
            ks.len() as u64 <= ROUNDS && us.len() as u64 <= ROUNDS,
            "trial {trial}: more rows recovered than were ever sent"
        );
    }
    assert!(killed > 0, "every trial finished before its kill -- this tested nothing");
    eprintln!("  {killed}/{trials} trials killed inside a mixed folding/durable commit");
}

/// Every `id` in `t`, after reopening in a fresh process.
fn ids(dir: &Scratch, t: &str) -> Result<Vec<u64>, String> {
    let out = Command::new(bin())
        .args(["--data", dir.s(), "-q", &format!("SELECT id FROM {t} ORDER BY id")])
        .args(["--format", "tsv", "--no-header"])
        .output()
        .expect("spawn granular");
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(out
        .stdout
        .split(|b| b.is_ascii_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).parse().expect("an id"))
        .collect())
}

/// Create the tables in their own process -- so its exit checkpoint is what
/// puts the seeded rows in part files -- and drop `body` beside them as a
/// script for the child to run.
fn seed_and_script(dir: &Scratch, ddl: &str, body: &str) -> PathBuf {
    let out = Command::new(bin())
        .args(["--data", dir.s(), "-q", ddl])
        .output()
        .expect("spawn granular");
    assert!(out.status.success(), "setup: {}", String::from_utf8_lossy(&out.stderr));
    let p = dir.path().join("w.sql");
    std::fs::write(&p, body).unwrap();
    p
}

fn script_child(dir: &Scratch, script: &Path) -> Child {
    Command::new(bin())
        .args(["--data", dir.s(), "-f", script.to_str().unwrap()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn granular")
}
