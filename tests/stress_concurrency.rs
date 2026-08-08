//! Concurrency stress: many readers against a writer, with an oracle that
//! makes a *wrong answer* fail rather than only a crash.
//!
//! ## The oracle, and why it needs no shared state
//!
//! Every writer here only ever appends, in atomic batches of `K` rows, ids
//! `b*K .. (b+1)*K`. So the set of committed rows after batch `b` is exactly
//! `{0 .. (b+1)*K}` -- and therefore *any* consistent snapshot a reader can
//! legally observe is `{0..n}` for some `n` that is a multiple of `K`. That
//! collapses the reference model to arithmetic:
//!
//! ```text
//!   count()       == n           and  n % K == 0
//!   uniqExact(id) == n                (no duplicates)
//!   sum(id)       == n*(n-1)/2        (no torn batch, no missing row)
//!   min(id)       == 0, max(id) == n-1
//! ```
//!
//! A `BTreeMap` reference would need a lock the reader shares with the writer,
//! which is exactly the contention the test is trying to *create* rather than
//! serialize. The closed form is the same reference model with the lock
//! removed: a reader that saw half of batch `b` fails the `sum` check, one
//! that saw a duplicate fails `uniqExact`, one that saw a row from an
//! uncommitted transaction fails `n % K`. All five aggregates come out of a
//! **single** statement, because one statement is one snapshot -- two
//! statements would be two, and the test would be checking nothing.
//!
//! Each writer batch is deliberately **two** `INSERT`s inside one
//! `BEGIN`/`COMMIT`. One multi-row `INSERT` is already atomic via
//! `atomic_stmt`, so a single-statement batch would prove only that
//! statement-level atomicity works; two statements are what make "never a row
//! from a half-applied transaction" a claim about *transactions*.
//!
//! ## Runtime
//!
//! A plain `cargo test` runs this file in 6.8 s (7.4 s under `--release`), and
//! it is bounded by construction rather than by a timer: every writer loop is
//! `for round in 0..ROUNDS` with `ROUNDS` scaled by `GRANULAR_STRESS`
//! (default 1), and every reader stops when the writer sets its done flag. The
//! deep run is `GRANULAR_STRESS=25 cargo test --release --test
//! stress_concurrency` -- ~9000 checked queries per chaos seed, ~50 s.
//! `GRANULAR_STRESS_SEED` replays a failure; the seed is printed by every test
//! that uses randomness.
//!
//! Every test also carries a [`Watchdog`]. A concurrency bug that hangs is
//! still a bug, and a hung test that never returns is a bug report nobody
//! reads.

// `write!` into the reused SQL buffer rather than `push_str(&format!(..))`:
// the writer loops build 32 rows per batch and the chaos test runs 1500 of
// them, so the temporary `String` per row is the one allocation in this file
// that is actually on a path that repeats.
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::time::{Duration, Instant};

use granular::types::Value;
use granular::{Db, Reader, Session};

// ---------------------------------------------------------------- fixtures

/// A unique scratch directory per test, removed on drop. Same shape as the one
/// in `tests/persistence.rs`: pid plus a counter, so a failure is reproducible
/// and two tests never collide.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "granular-sc-{}-{}-{}",
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

/// Work multiplier. 1 keeps the whole file under ~6 s in `--release`; the deep
/// sweep is `GRANULAR_STRESS=20`.
fn scale() -> u64 {
    std::env::var("GRANULAR_STRESS").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
}

/// Seed, printed by every randomized test so a failure replays with
/// `GRANULAR_STRESS_SEED=<n>`.
fn seed(tag: &str) -> u64 {
    let s = std::env::var("GRANULAR_STRESS_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    eprintln!("stress_concurrency::{tag}: GRANULAR_STRESS_SEED={s} GRANULAR_STRESS={}", scale());
    s
}

/// xorshift64*, so a "random" schedule is a *replayable* one. No dependency,
/// and the quality needed here is "decorrelated small integers".
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
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

fn threads_available() -> usize {
    std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1)
}

/// A hang is a bug, and a hung test that never returns is a bug report nobody
/// reads. Every test in this file runs its threads under this: if the work has
/// not finished by `limit`, print what was reached and take the process down.
///
/// `std::process::exit` rather than `panic!` on purpose -- a deadlock leaves
/// live threads parked on a lock, and unwinding one test thread out of a
/// harness that then waits on the others turns a diagnosable failure into a
/// silent hang at the end of the run.
struct Watchdog {
    done: Arc<(Mutex<bool>, Condvar)>,
    _t: std::thread::JoinHandle<()>,
}

impl Watchdog {
    fn new(what: &'static str, limit: Duration) -> Watchdog {
        let done = Arc::new((Mutex::new(false), Condvar::new()));
        let d = Arc::clone(&done);
        let t = std::thread::spawn(move || {
            let (m, cv) = &*d;
            let mut g = m.lock().unwrap_or_else(|e| e.into_inner());
            let start = Instant::now();
            while !*g && start.elapsed() < limit {
                let (ng, _) = cv
                    .wait_timeout(g, limit.saturating_sub(start.elapsed()))
                    .unwrap_or_else(|e| e.into_inner());
                g = ng;
            }
            if !*g {
                eprintln!(
                    "\nWATCHDOG: `{what}` made no progress for {:?} -- deadlock or livelock. \
                     Re-run with GRANULAR_STRESS_SEED printed above.",
                    limit
                );
                std::process::exit(101);
            }
        });
        Watchdog { done, _t: t }
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        let (m, cv) = &*self.done;
        *m.lock().unwrap_or_else(|e| e.into_inner()) = true;
        cv.notify_all();
    }
}

// ------------------------------------------------------------- the oracle

/// Rows per atomic batch. Two `INSERT`s of `K/2` inside one transaction.
const K: u64 = 32;

const DDL: &str = "CREATE TABLE t (id UInt64, b UInt64, s String) \
                   ENGINE = MergeTree ORDER BY id";

/// The five-aggregate snapshot probe. One statement, therefore one snapshot.
const PROBE: &str = "SELECT count(), uniqExact(id), sum(id), min(id), max(id) FROM t";

fn u64_of(v: &Value) -> u64 {
    match v {
        Value::UInt(n) => *n,
        Value::Int(n) => *n as u64,
        Value::Null => 0,
        other => panic!("expected a number, got {other}"),
    }
}

/// Check one probe result against the closed form. Returns the row count so a
/// caller can assert monotonicity.
///
/// `who` and `round` are in every message because a concurrency failure that
/// cannot be attributed to a thread is a failure that cannot be minimized.
fn check_prefix(rows: &[Value], who: &str) -> u64 {
    let (n, uniq, sum, lo, hi) =
        (u64_of(&rows[0]), u64_of(&rows[1]), u64_of(&rows[2]), u64_of(&rows[3]), u64_of(&rows[4]));
    assert_eq!(
        n % K,
        0,
        "{who}: saw {n} rows, not a whole number of {K}-row transactions -- \
         a half-applied transaction is visible"
    );
    assert_eq!(uniq, n, "{who}: {n} rows but {uniq} distinct ids -- a row was published twice");
    assert_eq!(
        sum,
        n * n.saturating_sub(1) / 2,
        "{who}: {n} rows sum to {sum}, want {} -- the snapshot is not a prefix {{0..n}}",
        n * n.saturating_sub(1) / 2
    );
    if n > 0 {
        assert_eq!(lo, 0, "{who}: min(id) = {lo}, want 0");
        assert_eq!(hi, n - 1, "{who}: {n} rows but max(id) = {hi}, want {}", n - 1);
    }
    n
}

/// Prefix tables for "the ids below `n` whose decimal form contains `digit`":
/// `(count[n], sum[n])`.
///
/// Built once per test, because the alternative is what the first draft did --
/// re-derive the answer with `id.to_string()` inside the reader loop, which is
/// one `String` per row per query and made the *oracle* the bottleneck: 40
/// batches took 1587 ms of which the engine was a minority. A reader loop that
/// spends its time in the reference model is not stressing anything.
fn digit_tables(hi: u64, digit: u8) -> (Vec<u32>, Vec<u64>) {
    let (mut counts, mut sums) = (Vec::with_capacity(hi as usize + 1), Vec::with_capacity(hi as usize + 1));
    let (mut c, mut s, mut buf) = (0u32, 0u64, String::with_capacity(24));
    for id in 0..=hi {
        counts.push(c);
        sums.push(s);
        buf.clear();
        write!(buf, "{id}").unwrap();
        if buf.as_bytes().contains(&digit) {
            c += 1;
            s += id;
        }
    }
    (counts, sums)
}

fn probe(r: &Reader, who: &str) -> u64 {
    let rs = r.query(PROBE).unwrap_or_else(|e| panic!("{who}: probe failed: {e}"));
    let v = rs.to_values();
    assert_eq!(v.len(), 1, "{who}: probe returned {} rows", v.len());
    check_prefix(&v[0], who)
}

/// `BEGIN; INSERT half; INSERT half; COMMIT;` for batch `b`.
///
/// The SQL is built into a reused `String` -- a batch is 32 rows and the point
/// of the test is the *reader* side, so a writer that allocates two `String`s
/// per batch is spending the machine on the wrong thread.
fn write_batch(db: &Db, sql: &mut String, b: u64) {
    let half = K / 2;
    db.transaction(|s| {
        for chunk in 0..2u64 {
            sql.clear();
            sql.push_str("INSERT INTO t VALUES ");
            for j in 0..half {
                let id = b * K + chunk * half + j;
                if j > 0 {
                    sql.push(',');
                }
                write!(sql, "({id},{b},'r{id}')").unwrap();
            }
            s.execute(sql.as_str())?;
        }
        Ok(())
    })
    .unwrap_or_else(|e| panic!("batch {b} failed: {e}"));
}

// ---------------------------------------------------------------- the tests

/// N readers against one writer: every snapshot a reader sees is a whole
/// number of committed transactions.
///
/// The reader count is deliberately `2 x cores`: oversubscription is what puts
/// a reader on the CPU *between* the writer's two `INSERT`s rather than only
/// before or after them.
#[test]
fn readers_see_whole_transactions_only() {
    let _w = Watchdog::new("readers_see_whole_transactions_only", Duration::from_secs(120));
    let dir = Scratch::new("whole-txn");
    let db = Db::open(dir.path()).unwrap();
    db.execute(DDL).unwrap();

    let batches = 60 * scale();
    let nreaders = (threads_available() * 2).max(4);
    let done = Arc::new(AtomicBool::new(false));
    let probes = Arc::new(AtomicU64::new(0));
    let start = Arc::new(Barrier::new(nreaders + 1));

    let readers: Vec<_> = (0..nreaders)
        .map(|i| {
            let r = db.reader();
            let (done, probes, start) = (Arc::clone(&done), Arc::clone(&probes), Arc::clone(&start));
            std::thread::spawn(move || {
                let who = format!("reader {i}");
                start.wait();
                // Monotonic: the writer only appends, so a snapshot that went
                // *backwards* means a reader was handed a stale part set --
                // a different bug from a torn read, and invisible to the
                // closed form on its own.
                let mut last = 0;
                let mut n = 0u64;
                while !done.load(Ordering::Relaxed) {
                    let seen = probe(&r, &who);
                    assert!(seen >= last, "{who}: snapshot went backwards, {last} -> {seen}");
                    last = seen;
                    n += 1;
                }
                probes.fetch_add(n, Ordering::Relaxed);
                last
            })
        })
        .collect();

    let mut sql = String::with_capacity(1024);
    start.wait();
    for b in 0..batches {
        write_batch(&db, &mut sql, b);
    }
    done.store(true, Ordering::Relaxed);

    for (i, h) in readers.into_iter().enumerate() {
        h.join().unwrap_or_else(|_| panic!("reader {i} panicked"));
    }
    // The final state is the whole thing, checked from a fresh handle.
    assert_eq!(probe(&db.reader(), "final"), batches * K);
    assert!(
        probes.load(Ordering::Relaxed) > 0,
        "no reader ever ran a probe -- the test proved nothing"
    );
}

/// The same claim, with the writer's transactions *deleting* as well as
/// inserting, so a torn read of a `DELETE` is detectable too.
///
/// Batch `b` inserts `b*K .. (b+1)*K` and deletes the single id `b*K` in the
/// same transaction, so a legal snapshot after `b` batches is
/// `{0..b*K} \ {0, K, 2K, ...}`: `count = b*(K-1)` and the sum is the closed
/// form minus the deleted arithmetic progression. A reader that sees the
/// `INSERT` without the `DELETE` -- the shape that a positional sweep
/// published outside its transaction would produce -- fails on `count`.
#[test]
fn readers_see_whole_transactions_with_deletes() {
    let _w = Watchdog::new("readers_see_whole_transactions_with_deletes", Duration::from_secs(120));
    let dir = Scratch::new("whole-txn-del");
    let db = Db::open(dir.path()).unwrap();
    db.execute(DDL).unwrap();

    let batches = 25 * scale();
    let nreaders = threads_available().max(2);
    let done = Arc::new(AtomicBool::new(false));
    let start = Arc::new(Barrier::new(nreaders + 1));

    let readers: Vec<_> = (0..nreaders)
        .map(|i| {
            let r = db.reader();
            let (done, start) = (Arc::clone(&done), Arc::clone(&start));
            std::thread::spawn(move || {
                let who = format!("reader {i}");
                start.wait();
                while !done.load(Ordering::Relaxed) {
                    let v = r.query(PROBE).unwrap_or_else(|e| panic!("{who}: {e}")).to_values();
                    let (n, uniq, sum) = (u64_of(&v[0][0]), u64_of(&v[0][1]), u64_of(&v[0][2]));
                    assert_eq!(uniq, n, "{who}: {n} rows, {uniq} distinct ids");
                    assert_eq!(
                        n % (K - 1),
                        0,
                        "{who}: {n} rows is not a whole number of {}-row batches -- \
                         a transaction's INSERT is visible without its DELETE (or vice versa)",
                        K - 1
                    );
                    let b = n / (K - 1);
                    let all = b * K;
                    let want = all * all.saturating_sub(1) / 2 - K * b * b.saturating_sub(1) / 2;
                    assert_eq!(
                        sum, want,
                        "{who}: {n} rows (batch {b}) sum to {sum}, want {want}"
                    );
                }
            })
        })
        .collect();

    let mut sql = String::with_capacity(1024);
    start.wait();
    for b in 0..batches {
        db.transaction(|s| {
            sql.clear();
            sql.push_str("INSERT INTO t VALUES ");
            for j in 0..K {
                let id = b * K + j;
                if j > 0 {
                    sql.push(',');
                }
                write!(sql, "({id},{b},'r{id}')").unwrap();
            }
            s.execute(sql.as_str())?;
            s.execute(&format!("DELETE FROM t WHERE id = {}", b * K))
        })
        .unwrap_or_else(|e| panic!("batch {b} failed: {e}"));
    }
    done.store(true, Ordering::Relaxed);
    for (i, h) in readers.into_iter().enumerate() {
        h.join().unwrap_or_else(|_| panic!("reader {i} panicked"));
    }
    let v = db.reader().query(PROBE).unwrap().to_values();
    assert_eq!(u64_of(&v[0][0]), batches * (K - 1));
}

/// Readers scanning while parts are merged out from under them and the files
/// behind them are unlinked.
///
/// The seed is written, checkpointed and the session **dropped** before the
/// readers start, so the parts they scan are `mmap`ed straight out of
/// `part_*.gpart`. The compactor then runs `OPTIMIZE TABLE t FINAL` and
/// `checkpoint`, and `write_table` `remove_file`s every superseded part while
/// those mappings are live. The design says a mapping survives its unlink;
/// this is the load that proves it rather than the comment that asserts it.
#[test]
fn readers_race_compaction_and_unlink() {
    let _w = Watchdog::new("readers_race_compaction_and_unlink", Duration::from_secs(180));
    let dir = Scratch::new("compact-race");
    let rounds = 12 * scale();

    // 24 parts on disk before anyone reads: above AUTO_COMPACT_PARTS (16), so
    // the very first flush after reopen also triggers auto-compaction.
    let seed_batches = 24u64;
    {
        let mut s = Session::open(dir.path()).unwrap();
        s.execute(DDL).unwrap();
        let mut sql = String::new();
        for b in 0..seed_batches {
            sql.clear();
            sql.push_str("INSERT INTO t VALUES ");
            for j in 0..K {
                let id = b * K + j;
                if j > 0 {
                    sql.push(',');
                }
                write!(sql, "({id},{b},'r{id}')").unwrap();
            }
            s.execute(&sql).unwrap();
            s.execute("SYSTEM FLUSH").unwrap();
        }
        s.checkpoint().unwrap();
    }

    let db = Db::open(dir.path()).unwrap();
    assert_eq!(probe(&db.reader(), "reopen"), seed_batches * K);

    // `like7[n]` is the sum of the ids below `n` whose decimal form contains a
    // `7` -- the answer `sumIf(id, s LIKE '%7%')` must give for a snapshot of
    // `n` rows.
    let like7 = Arc::new(digit_tables((seed_batches + rounds) * K, b'7').1);

    let nreaders = (threads_available() * 2).max(4);
    let done = Arc::new(AtomicBool::new(false));
    let start = Arc::new(Barrier::new(nreaders + 1));
    let readers: Vec<_> = (0..nreaders)
        .map(|i| {
            let r = db.reader();
            let (done, start, like7) =
                (Arc::clone(&done), Arc::clone(&start), Arc::clone(&like7));
            std::thread::spawn(move || {
                let who = format!("reader {i}");
                start.wait();
                let mut last = 0;
                while !done.load(Ordering::Relaxed) {
                    // Six aggregates, one statement: the `LIKE` sum and the
                    // row count it is checked against have to come out of the
                    // *same* snapshot. Splitting them was the first version of
                    // this test and it failed instantly against a correct
                    // engine -- the second query simply saw a later snapshot.
                    // The `LIKE` is what forces the whole mapping to be
                    // touched instead of answered out of a zone map.
                    let v = r
                        .query(
                            "SELECT count(), uniqExact(id), sum(id), min(id), max(id), \
                                    sumIf(id, s LIKE '%7%') FROM t",
                        )
                        .unwrap_or_else(|e| panic!("{who}: probe: {e}"))
                        .to_values();
                    let n = check_prefix(&v[0], &who);
                    assert!(n >= last, "{who}: snapshot went backwards, {last} -> {n}");
                    last = n;
                    let (got, want) = (u64_of(&v[0][5]), like7[n as usize]);
                    assert_eq!(got, want, "{who}: LIKE scan over {n} rows gave {got}, want {want}");
                }
            })
        })
        .collect();

    let mut sql = String::with_capacity(1024);
    start.wait();
    for round in 0..rounds {
        let b = seed_batches + round;
        write_batch(&db, &mut sql, b);
        db.execute("SYSTEM FLUSH").unwrap();
        if round % 4 == 3 {
            db.execute("OPTIMIZE TABLE t FINAL").unwrap();
            // Rewrites every part file under new sequence numbers and unlinks
            // the old ones -- the unlink the readers must survive.
            db.writer().checkpoint().unwrap();
        }
    }
    done.store(true, Ordering::Relaxed);
    for (i, h) in readers.into_iter().enumerate() {
        h.join().unwrap_or_else(|_| panic!("reader {i} panicked"));
    }
    assert_eq!(probe(&db.reader(), "final"), (seed_batches + rounds) * K);
}

/// Readers racing a `ROLLBACK`: rolled-back rows must never appear, during or
/// after.
///
/// The rolled-back batches use ids from `POISON` upwards, so "did a reader
/// ever see one" is a `WHERE id >= POISON` count and not an inference. Both
/// halves are checked in one statement, because two statements would be two
/// snapshots and a row could hide between them.
#[test]
fn readers_race_rollback() {
    const POISON: u64 = 1 << 40;
    let _w = Watchdog::new("readers_race_rollback", Duration::from_secs(120));
    let dir = Scratch::new("rollback-race");
    let db = Db::open(dir.path()).unwrap();
    db.execute(DDL).unwrap();

    let rounds = 40 * scale();
    let nreaders = threads_available().max(2);
    let done = Arc::new(AtomicBool::new(false));
    let start = Arc::new(Barrier::new(nreaders + 1));

    let readers: Vec<_> = (0..nreaders)
        .map(|i| {
            let r = db.reader();
            let (done, start) = (Arc::clone(&done), Arc::clone(&start));
            std::thread::spawn(move || {
                let who = format!("reader {i}");
                start.wait();
                while !done.load(Ordering::Relaxed) {
                    let v = r
                        .query(
                            "SELECT countIf(id < 1099511627776), countIf(id >= 1099511627776), \
                                    sumIf(id, id < 1099511627776) FROM t",
                        )
                        .unwrap_or_else(|e| panic!("{who}: {e}"))
                        .to_values();
                    let (good, bad, sum) = (u64_of(&v[0][0]), u64_of(&v[0][1]), u64_of(&v[0][2]));
                    assert_eq!(bad, 0, "{who}: {bad} rolled-back rows are visible");
                    assert_eq!(good % K, 0, "{who}: {good} committed rows, not a multiple of {K}");
                    assert_eq!(
                        sum,
                        good * good.saturating_sub(1) / 2,
                        "{who}: {good} rows sum to {sum}"
                    );
                }
            })
        })
        .collect();

    let mut sql = String::with_capacity(1024);
    start.wait();
    let mut committed = 0u64;
    for round in 0..rounds {
        if round % 2 == 0 {
            write_batch(&db, &mut sql, committed);
            committed += 1;
        } else {
            // An error out of `f` is what `Db::transaction` rolls back on, so
            // the rollback is driven the way an application would drive it
            // rather than by calling `rollback` directly.
            let e = db.transaction::<()>(|s| {
                sql.clear();
                sql.push_str("INSERT INTO t VALUES ");
                for j in 0..K {
                    let id = POISON + round * K + j;
                    if j > 0 {
                        sql.push(',');
                    }
                    write!(sql, "({id},{round},'poison')").unwrap();
                }
                s.execute(&sql)?;
                // Read-your-own-writes inside the transaction: the rows the
                // rollback is about to erase must be visible *here*, or the
                // test would pass on a transaction that never wrote anything.
                let n = u64_of(
                    &s.query("SELECT count() FROM t WHERE id >= 1099511627776")?.scalar().unwrap(),
                );
                assert_eq!(n, K, "the open transaction cannot see its own {K} rows");
                Err(granular::Error::exec("deliberate rollback"))
            });
            assert!(e.is_err(), "the rollback transaction reported success");
        }
    }
    done.store(true, Ordering::Relaxed);
    for (i, h) in readers.into_iter().enumerate() {
        h.join().unwrap_or_else(|_| panic!("reader {i} panicked"));
    }
    assert_eq!(probe(&db.reader(), "final"), committed * K);
    // ... and on disk, after a reopen: a rollback leaves no trace in the log
    // either, and replay is the only thing that would resurrect it.
    db.writer().checkpoint().unwrap();
    drop(db);
    let mut s = Session::open(dir.path()).unwrap();
    let n = u64_of(&s.query("SELECT count() FROM t WHERE id >= 1099511627776").unwrap().scalar().unwrap());
    assert_eq!(n, 0, "{n} rolled-back rows came back after a reopen");
}

/// Nested parallelism: parallel queries from many threads, while the writer is
/// building parts big enough that `Part::build_sel` fans out on its own.
///
/// `Part::build_sel` spawns `available_parallelism()` scoped threads whenever a
/// part has >= 8 granules (>= 8192 rows) and **does not consult**
/// `pool::in_pool`, unlike every other fan-out in the tree. This test is the
/// load that would expose the resulting oversubscription: `4 x cores` reader
/// threads whose queries are each sharded by the exchange, against a writer
/// whose every flush is a 16-granule part build. The watchdog is the assertion
/// -- a thread explosion shows up as a stall, not as a wrong answer.
#[test]
fn nested_parallelism_stays_bounded() {
    let _w = Watchdog::new("nested_parallelism_stays_bounded", Duration::from_secs(180));
    let dir = Scratch::new("nested");
    let db = Db::open(dir.path()).unwrap();
    db.execute(DDL).unwrap();

    // 16 granules per flush: over `Part::build_sel`'s `ranges.len() >= 8`
    // threshold, so every flush below really does fan out.
    const ROWS: u64 = 16 * 1024;
    let rounds = 4 * scale();
    let nreaders = (threads_available() * 4).max(8);

    let mut sql = String::with_capacity(ROWS as usize * 24);
    let mut base = 0u64;
    let fill = |db: &Db, base: &mut u64, sql: &mut String| {
        sql.clear();
        sql.push_str("INSERT INTO t VALUES ");
        for j in 0..ROWS {
            let id = *base + j;
            if j > 0 {
                sql.push(',');
            }
            write!(sql, "({id},{},'r{id}')", id % 97).unwrap();
        }
        db.execute(sql).unwrap();
        db.execute("SYSTEM FLUSH").unwrap();
        *base += ROWS;
    };
    fill(&db, &mut base, &mut sql);

    let done = Arc::new(AtomicBool::new(false));
    let start = Arc::new(Barrier::new(nreaders + 1));
    let slowest = Arc::new(AtomicU64::new(0));
    let readers: Vec<_> = (0..nreaders)
        .map(|i| {
            let r = db.reader();
            let (done, start, slowest) =
                (Arc::clone(&done), Arc::clone(&start), Arc::clone(&slowest));
            std::thread::spawn(move || {
                let who = format!("reader {i}");
                start.wait();
                while !done.load(Ordering::Relaxed) {
                    let t = Instant::now();
                    // GROUP BY over the whole table: the exchange shards this
                    // across the pool, so every reader thread is itself a
                    // fan-out.
                    let rs = r
                        .query("SELECT b, count(), sum(id) FROM t GROUP BY b")
                        .unwrap_or_else(|e| panic!("{who}: {e}"));
                    let groups = rs.rows() as u64;
                    assert_eq!(groups, 97, "{who}: {groups} groups, want 97");
                    let n = probe(&r, &who);
                    assert_eq!(n % ROWS, 0, "{who}: {n} rows is not a whole number of flushes");
                    slowest.fetch_max(t.elapsed().as_millis() as u64, Ordering::Relaxed);
                }
            })
        })
        .collect();

    start.wait();
    for _ in 0..rounds {
        fill(&db, &mut base, &mut sql);
    }
    done.store(true, Ordering::Relaxed);
    for (i, h) in readers.into_iter().enumerate() {
        h.join().unwrap_or_else(|_| panic!("reader {i} panicked"));
    }
    eprintln!(
        "nested_parallelism_stays_bounded: {nreaders} readers, slowest query pair {} ms",
        slowest.load(Ordering::Relaxed)
    );
    assert_eq!(probe(&db.reader(), "final"), base);
}

/// Deliberate contention: a small table, many threads, and long and short
/// queries mixed, with the schedule driven by a printed seed.
///
/// Small on purpose. A big table hides lock behaviour behind work; a table
/// that answers in microseconds makes the lock the whole cost, which is where
/// a livelock or a starved writer shows up. The mix matters too: a long
/// `LIKE` scan holds the shared lock long enough for a writer to queue behind
/// it, and the short point lookups then have to get past that writer.
#[test]
fn mixed_short_and_long_queries_under_contention() {
    let _w =
        Watchdog::new("mixed_short_and_long_queries_under_contention", Duration::from_secs(120));
    let s0 = seed("mixed_short_and_long_queries_under_contention");
    let dir = Scratch::new("contention");
    let db = Db::open(dir.path()).unwrap();
    db.execute(DDL).unwrap();

    let batches = 40 * scale();
    let nthreads = (threads_available() * 3).max(6);
    let done = Arc::new(AtomicBool::new(false));
    let start = Arc::new(Barrier::new(nthreads + 1));
    let shape = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0)]);
    let like3 = Arc::new(digit_tables(batches * K, b'3').0);

    let readers: Vec<_> = (0..nthreads)
        .map(|i| {
            let r = db.reader();
            let (done, start, shape, like3) =
                (Arc::clone(&done), Arc::clone(&start), Arc::clone(&shape), Arc::clone(&like3));
            std::thread::spawn(move || {
                let who = format!("thread {i}");
                let mut rng = Rng::new(s0 ^ (i as u64).wrapping_mul(0x9E37_79B9));
                start.wait();
                while !done.load(Ordering::Relaxed) {
                    match rng.below(3) {
                        // Short: a point lookup through the index.
                        0 => {
                            let n = probe(&r, &who);
                            if n > 0 {
                                let id = rng.below(n);
                                let got = r
                                    .query(&format!("SELECT s FROM t WHERE id = {id}"))
                                    .unwrap_or_else(|e| panic!("{who}: point {id}: {e}"));
                                assert_eq!(
                                    got.scalar(),
                                    Some(Value::str(format!("r{id}"))),
                                    "{who}: point lookup of {id} in a {n}-row snapshot"
                                );
                            }
                            shape[0].fetch_add(1, Ordering::Relaxed);
                        }
                        // Long: a full scan with a substring predicate, which
                        // nothing can prune -- and the row count beside it in
                        // the *same* statement, so the check is exact rather
                        // than a bound. Two statements would be two snapshots
                        // and the answer could only be bracketed.
                        1 => {
                            let v = r
                                .query(
                                    "SELECT count(), uniqExact(id), sum(id), min(id), max(id), \
                                            countIf(s LIKE '%3%') FROM t",
                                )
                                .unwrap_or_else(|e| panic!("{who}: like: {e}"))
                                .to_values();
                            let n = check_prefix(&v[0], &who);
                            let (got, want) = (u64_of(&v[0][5]), like3[n as usize] as u64);
                            assert_eq!(
                                got, want,
                                "{who}: LIKE '%3%' matched {got} of {n} rows, want {want}"
                            );
                            shape[1].fetch_add(1, Ordering::Relaxed);
                        }
                        // A streaming read, which holds the shared lock for
                        // the length of the stream rather than one query.
                        _ => {
                            let mut n = 0usize;
                            let mut sum = 0u64;
                            r.stream("SELECT id FROM t", &mut |item| {
                                if let granular::StreamItem::Rows(b) = item {
                                    n += b.rows();
                                    for row in 0..b.rows() {
                                        sum += u64_of(&b.column(0).value(row));
                                    }
                                }
                                Ok(())
                            })
                            .unwrap_or_else(|e| panic!("{who}: stream: {e}"));
                            let n = n as u64;
                            assert_eq!(n % K, 0, "{who}: streamed {n} rows, not a multiple of {K}");
                            assert_eq!(
                                sum,
                                n * n.saturating_sub(1) / 2,
                                "{who}: streamed {n} rows summing to {sum}"
                            );
                            shape[2].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            })
        })
        .collect();

    let mut sql = String::with_capacity(1024);
    start.wait();
    let t0 = Instant::now();
    for b in 0..batches {
        write_batch(&db, &mut sql, b);
    }
    let write_ms = t0.elapsed().as_millis();
    done.store(true, Ordering::Relaxed);
    for (i, h) in readers.into_iter().enumerate() {
        h.join().unwrap_or_else(|_| panic!("thread {i} panicked"));
    }
    eprintln!(
        "mixed contention: {nthreads} threads, {batches} batches in {write_ms} ms, \
         {} point / {} scan / {} stream",
        shape[0].load(Ordering::Relaxed),
        shape[1].load(Ordering::Relaxed),
        shape[2].load(Ordering::Relaxed)
    );
    // A starved writer is a failure even when every answer is right: the whole
    // reason `Reader` exists is that a read must not stop a write.
    assert!(
        shape[0].load(Ordering::Relaxed) + shape[2].load(Ordering::Relaxed) > 0,
        "no short query or stream ever completed"
    );
    assert_eq!(probe(&db.reader(), "final"), batches * K);
}

/// A `Cursor` open across a writer's commits.
///
/// The cursor holds the shared lock on its own thread until it is drained, so
/// this is the one shape where a reader can *block* the writer. Two things
/// have to be true: the rows it yields are one consistent snapshot (not a
/// moving target), and the writer parked behind it makes progress once it is
/// dropped -- neither of which a single-threaded test can observe.
#[test]
fn cursor_snapshot_does_not_move_and_does_not_wedge_the_writer() {
    let _w = Watchdog::new("cursor_snapshot", Duration::from_secs(120));
    let dir = Scratch::new("cursor");
    let db = Db::open(dir.path()).unwrap();
    db.execute(DDL).unwrap();
    let mut sql = String::with_capacity(1024);
    for b in 0..8 {
        write_batch(&db, &mut sql, b);
    }
    // Buffered rows would make `Reader::cursor` take the *exclusive* lock, so
    // the test would be measuring a different code path than the one it is
    // about. Flushing first is the documented way to avoid it.
    db.execute("SYSTEM FLUSH").unwrap();

    let rounds = 5 * scale();
    for round in 0..rounds {
        let before = probe(&db.reader(), "before");
        let cur = db.reader().cursor("SELECT id FROM t").unwrap();

        // A writer parked behind the cursor. It cannot commit until the
        // cursor is dropped; if it ever does, the snapshot below moves.
        let wdb = db.clone();
        let committed = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&committed);
        let b = 8 + round;
        let w = std::thread::spawn(move || {
            let mut sql = String::with_capacity(1024);
            write_batch(&wdb, &mut sql, b);
            flag.store(true, Ordering::Relaxed);
        });

        let mut n = 0u64;
        let mut sum = 0u64;
        for blk in cur {
            let blk = blk.unwrap_or_else(|e| panic!("cursor block: {e}"));
            for r in 0..blk.rows() {
                sum += u64_of(&blk.column(0).value(r));
            }
            n += blk.rows() as u64;
            // Deliberately slow, so the writer really is waiting rather than
            // racing us to the end.
            std::thread::yield_now();
        }
        assert_eq!(n, before, "cursor yielded {n} rows for a {before}-row snapshot");
        assert_eq!(sum, n * n.saturating_sub(1) / 2, "cursor rows do not form {{0..{n}}}");

        w.join().expect("the writer behind the cursor panicked");
        assert!(committed.load(Ordering::Relaxed));
        assert_eq!(probe(&db.reader(), "after"), before + K);
    }
}

/// How many OS threads this process has right now.
///
/// `/proc/self/task` on Linux, `ps -M` on macOS -- two syscalls' worth of
/// portability rather than a dependency, and it is only called from a sampler
/// running every 2 ms.
fn live_threads() -> usize {
    if let Ok(rd) = std::fs::read_dir("/proc/self/task") {
        return rd.count();
    }
    let out = std::process::Command::new("ps")
        .args(["-M", "-p", &std::process::id().to_string()])
        .output();
    match out {
        // One header line plus one line per thread.
        Ok(o) => String::from_utf8_lossy(&o.stdout).lines().count().saturating_sub(1),
        Err(_) => 0,
    }
}

/// Part construction fans out with `std::thread::scope` and asks nobody's
/// permission first.
///
/// `Part::build_sel` spawns `min(cores, granules)` threads whenever a part has
/// >= 8 granules, and unlike every other fan-out in the tree it does not
/// consult `pool::in_pool` or any process-wide budget. One writer is fine --
/// that is the ~15 us per flush the pool's module docs price in. `N`
/// independent sessions in one process are not: nothing serializes them, so
/// the spawns are `N x min(cores, granules)` and they are all in flight at
/// once.
///
/// N independent in-memory sessions is not a contrived shape -- it is one
/// process holding several databases, which is what `Session::in_memory` is
/// *for*.
///
/// **Measured**, 42 writers flushing 16-granule parts on a 14-core machine,
/// three runs: the live thread count peaks at 123/169/154 against a 3-thread
/// baseline, i.e. **+120 to +166 threads, 9-12x the core count**, from a
/// workload whose useful parallelism is 14. Each build spawns 8 (16 granules
/// over 14 threads rounds to 2 per thread, so 8 chunks), and nothing caps how
/// many builds are in flight. It is oversubscription rather than a leak -- the
/// scope joins -- and the
/// bound below is the one that matters for a regression: if part construction
/// ever runs *inside* a pool job, each of the pool's own threads fans out
/// again and the peak becomes `writers x cores` (~588 here). That is what this
/// refuses. The +122 is recorded rather than asserted, because it is a
/// scheduling observation and would flake.
#[test]
fn concurrent_part_builds_do_not_multiply_threads() {
    let _w = Watchdog::new("concurrent_part_builds", Duration::from_secs(180));
    let cores = threads_available();
    let writers = (cores * 3).max(6);
    // 16 granules per part: over `Part::build_sel`'s `ranges.len() >= 8` gate.
    const ROWS: u64 = 16 * 1024;
    let rounds = 3 * scale();

    let baseline = live_threads();
    let peak = Arc::new(AtomicUsize::new(baseline));
    let done = Arc::new(AtomicBool::new(false));
    let start = Arc::new(Barrier::new(writers + 1));

    let sampler = {
        let (peak, done) = (Arc::clone(&peak), Arc::clone(&done));
        std::thread::spawn(move || {
            while !done.load(Ordering::Relaxed) {
                peak.fetch_max(live_threads(), Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(2));
            }
        })
    };

    let hands: Vec<_> = (0..writers)
        .map(|i| {
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                // One private session each: no lock between them, which is the
                // whole point.
                let mut s = Session::in_memory();
                s.execute(DDL).unwrap();
                let mut sql = String::with_capacity(ROWS as usize * 24);
                start.wait();
                for r in 0..rounds {
                    sql.clear();
                    sql.push_str("INSERT INTO t VALUES ");
                    for j in 0..ROWS {
                        let id = r * ROWS + j;
                        if j > 0 {
                            sql.push(',');
                        }
                        write!(sql, "({id},{},'r{id}')", id % 97).unwrap();
                    }
                    s.execute(sql.as_str()).unwrap();
                    s.execute("SYSTEM FLUSH").unwrap();
                }
                let n = s.query("SELECT count() FROM t").unwrap().scalar().unwrap();
                assert_eq!(u64_of(&n), rounds * ROWS, "writer {i} lost rows");
            })
        })
        .collect();

    start.wait();
    for (i, h) in hands.into_iter().enumerate() {
        h.join().unwrap_or_else(|_| panic!("writer {i} panicked"));
    }
    done.store(true, Ordering::Relaxed);
    sampler.join().unwrap();

    let peak = peak.load(Ordering::Relaxed);
    let extra = peak.saturating_sub(baseline);
    eprintln!(
        "  {writers} concurrent part builds on {cores} cores: {baseline} threads before, \
         {peak} at peak (+{extra}); a per-build fan-out with no budget would be ~{}",
        writers * cores
    );
    // Two thirds of `writers * cores`. The measured peak is +120 alone and up
    // to +218 when the rest of this file is running beside it, so the bound
    // has ~1.6x headroom over the observed worst case and sits ~1.5x below the
    // shape a nested fan-out would produce. Anything tighter flakes; anything
    // looser stops catching the regression it exists for.
    let bound = (writers * cores * 2 / 3).max(6 * cores);
    assert!(
        extra <= bound,
        "part construction spawned {extra} extra threads for {writers} concurrent builds \
         (bound {bound}): `Part::build_sel` fans out per call without consulting \
         `pool::in_pool` or any process-wide budget, so a build that runs inside a pool job \
         multiplies by the pool width",
    );
}

/// Everything at once, on a seeded schedule.
///
/// The narrow tests above each hold one variable still. This one does not: N
/// reader threads pick uniformly from six query shapes while the writer picks
/// from six mutations -- commit, rollback, flush, compact, checkpoint, reopen
/// a cursor -- so the interleavings that only appear when a `ROLLBACK` lands
/// inside a compaction inside a reader's buffered-write flush get generated
/// rather than imagined. The closed-form oracle is what makes the resulting
/// mess checkable: whatever order things happen in, a reader's snapshot is
/// still `{0..n}` with `n` a multiple of `K`.
///
/// Seeded, and the seed is printed, so a failure is a rerun and not a story.
#[test]
fn chaos_readers_and_a_chaotic_writer() {
    const POISON: u64 = 1 << 40;
    let _w = Watchdog::new("chaos_readers_and_a_chaotic_writer", Duration::from_secs(180));
    let s0 = seed("chaos_readers_and_a_chaotic_writer");
    let dir = Scratch::new("chaos");
    let db = Db::open(dir.path()).unwrap();
    db.execute(DDL).unwrap();

    let rounds = 60 * scale();
    let nreaders = (threads_available() * 2).max(4);
    let done = Arc::new(AtomicBool::new(false));
    let start = Arc::new(Barrier::new(nreaders + 1));
    let ops = Arc::new(AtomicUsize::new(0));

    let readers: Vec<_> = (0..nreaders)
        .map(|i| {
            let r = db.reader();
            let (done, start, ops) = (Arc::clone(&done), Arc::clone(&start), Arc::clone(&ops));
            std::thread::spawn(move || {
                let who = format!("chaos reader {i}");
                let mut rng = Rng::new(s0 ^ (i as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
                start.wait();
                while !done.load(Ordering::Relaxed) {
                    match rng.below(6) {
                        0 => {
                            probe(&r, &who);
                        }
                        // The rolled-back rows must never appear, whatever
                        // else is going on.
                        1 => {
                            let v = r
                                .query(
                                    "SELECT count(), countIf(id >= 1099511627776) FROM t",
                                )
                                .unwrap_or_else(|e| panic!("{who}: {e}"))
                                .to_values();
                            assert_eq!(u64_of(&v[0][1]), 0, "{who}: rolled-back rows are visible");
                            assert_eq!(u64_of(&v[0][0]) % K, 0, "{who}: torn snapshot");
                        }
                        2 => {
                            let n = probe(&r, &who);
                            if n > 0 {
                                let id = rng.below(n);
                                assert_eq!(
                                    r.query(&format!("SELECT s FROM t WHERE id = {id}"))
                                        .unwrap_or_else(|e| panic!("{who}: point {id}: {e}"))
                                        .scalar(),
                                    Some(Value::str(format!("r{id}"))),
                                    "{who}: point lookup of {id} below a {n}-row snapshot"
                                );
                            }
                        }
                        3 => {
                            let groups = r
                                .query("SELECT b, count() FROM t GROUP BY b")
                                .unwrap_or_else(|e| panic!("{who}: group by: {e}"))
                                .rows() as u64;
                            let n = probe(&r, &who);
                            // Batch `b` contributes group `b`, so the group
                            // count is the batch count of *some* snapshot at or
                            // before the probe.
                            assert!(
                                groups <= n / K,
                                "{who}: {groups} groups over a snapshot of {n} rows"
                            );
                        }
                        4 => {
                            let mut n = 0u64;
                            let mut sum = 0u64;
                            r.stream("SELECT id FROM t", &mut |item| {
                                if let granular::StreamItem::Rows(b) = item {
                                    n += b.rows() as u64;
                                    for row in 0..b.rows() {
                                        sum += u64_of(&b.column(0).value(row));
                                    }
                                }
                                Ok(())
                            })
                            .unwrap_or_else(|e| panic!("{who}: stream: {e}"));
                            assert_eq!(n % K, 0, "{who}: streamed {n} rows");
                            assert_eq!(sum, n * n.saturating_sub(1) / 2, "{who}: streamed sum");
                        }
                        _ => {
                            let mut n = 0u64;
                            let mut sum = 0u64;
                            for blk in r.cursor("SELECT id FROM t").unwrap_or_else(|e| {
                                panic!("{who}: cursor: {e}")
                            }) {
                                let blk = blk.unwrap_or_else(|e| panic!("{who}: cursor block: {e}"));
                                for row in 0..blk.rows() {
                                    sum += u64_of(&blk.column(0).value(row));
                                }
                                n += blk.rows() as u64;
                            }
                            assert_eq!(n % K, 0, "{who}: cursor yielded {n} rows");
                            assert_eq!(sum, n * n.saturating_sub(1) / 2, "{who}: cursor sum");
                        }
                    }
                    ops.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    let mut rng = Rng::new(s0 ^ 0xF1EA_5EED);
    let mut sql = String::with_capacity(1 << 12);
    let mut committed = 0u64;
    start.wait();
    for round in 0..rounds {
        match rng.below(6) {
            // Committed batches dominate, so the table actually grows and the
            // readers have something to be wrong about.
            0 | 1 | 2 => {
                write_batch(&db, &mut sql, committed);
                committed += 1;
            }
            3 => {
                let _ = db.transaction::<()>(|s| {
                    sql.clear();
                    sql.push_str("INSERT INTO t VALUES ");
                    for j in 0..K {
                        let id = POISON + round * K + j;
                        if j > 0 {
                            sql.push(',');
                        }
                        write!(sql, "({id},{round},'poison')").unwrap();
                    }
                    s.execute(sql.as_str())?;
                    Err(granular::Error::exec("deliberate rollback"))
                });
            }
            4 => db.execute("SYSTEM FLUSH").unwrap(),
            _ => {
                db.execute("OPTIMIZE TABLE t FINAL").unwrap();
                if round % 3 == 0 {
                    db.writer().checkpoint().unwrap();
                }
            }
        }
    }
    done.store(true, Ordering::Relaxed);
    for (i, h) in readers.into_iter().enumerate() {
        h.join().unwrap_or_else(|_| panic!("chaos reader {i} panicked"));
    }
    eprintln!(
        "  chaos: {nreaders} readers ran {} queries against {rounds} writer rounds \
         ({committed} committed batches)",
        ops.load(Ordering::Relaxed)
    );
    assert_eq!(probe(&db.reader(), "final"), committed * K);
    assert!(ops.load(Ordering::Relaxed) > nreaders as usize, "the readers barely ran");
}

/// A transaction driven through *separate* `writer()` acquisitions, which is
/// the shape a wire server has: one statement, one call, no guard parked in a
/// local.
///
/// `Db::transaction` holds the exclusive lock for the whole transaction, so a
/// reader structurally cannot see inside it. `db.writer().execute("BEGIN")`
/// does not: the guard drops at the end of the statement and the session is
/// left with an open transaction and **nothing holding the lock**, so a
/// `Reader` gets in while an uncommitted overlay is attached. Reading through
/// it would be a dirty read, and rolling it back would make the reader's
/// earlier answer retroactively false.
#[test]
fn a_reader_cannot_see_a_transaction_left_open_between_writer_guards() {
    let _w = Watchdog::new("reader_vs_open_txn", Duration::from_secs(60));
    let db = Db::in_memory();
    db.execute(DDL).unwrap();
    db.execute("INSERT INTO t VALUES (0,0,'r0'),(1,0,'r1')").unwrap();

    db.writer().execute("BEGIN").unwrap();
    db.writer().execute("INSERT INTO t VALUES (2,9,'poison'),(3,9,'poison')").unwrap();
    assert!(db.writer().in_transaction(), "the transaction did not survive the guard drop");

    // The reader is on another thread, so this is not "the same thread that
    // wrote" reading its own uncommitted work -- it is a different connection.
    let r = db.reader();
    let seen = std::thread::spawn(move || {
        (r.query("SELECT count() FROM t").map(|rs| u64_of(&rs.scalar().unwrap())),
         r.query("SELECT count() FROM t WHERE b = 9").map(|rs| u64_of(&rs.scalar().unwrap())))
    })
    .join()
    .expect("reader thread");

    // Either answer is defensible -- refuse, or show the pre-transaction state
    // -- and both are checked here so the test pins whichever one the engine
    // chose. What is not defensible is showing the uncommitted rows.
    if let Ok(n) = seen.0 {
        assert_eq!(
            n, 2,
            "a concurrent reader saw {n} rows while a transaction that inserted 2 more was \
             still open: that is a dirty read"
        );
    }
    if let Ok(n) = seen.1 {
        assert_eq!(n, 0, "a concurrent reader saw {n} uncommitted rows");
    }

    db.writer().execute("ROLLBACK").unwrap();
    assert_eq!(u64_of(&db.reader().query("SELECT count() FROM t").unwrap().scalar().unwrap()), 2);
}

/// Two cursors open at once, with a writer queued behind them.
///
/// This is a connection pool with two open portals and one `INSERT`, and it is
/// the one shape in the facade where the locking can invert: each `Cursor`
/// holds the shared lock on its own thread until it is drained, so a writer
/// that arrives between the two `cursor()` calls sits in the `RwLock`'s queue
/// -- and on a writer-preferring lock the *second* cursor then queues behind
/// the writer, which is waiting for the *first* cursor, which the client will
/// not drain until the second one has produced a row.
///
/// Detected with a bounded `recv_timeout` rather than by blocking the test
/// thread, so a real deadlock is reported as a failure instead of hanging the
/// run. The stuck threads are left parked; the harness's `main` returning ends
/// the process regardless.
#[test]
fn two_cursors_and_a_queued_writer_do_not_deadlock() {
    let _w = Watchdog::new("two_cursors_and_a_queued_writer", Duration::from_secs(90));
    let dir = Scratch::new("two-cursors");
    let db = Db::open(dir.path()).unwrap();
    db.execute(DDL).unwrap();
    let mut sql = String::with_capacity(1 << 16);
    for b in 0..16 {
        write_batch(&db, &mut sql, b);
    }
    // Flushed, so `cursor()` takes the *shared* lock: with rows still buffered
    // it would take the exclusive one and the first cursor alone would block
    // everything, which is a documented behaviour and a different test.
    db.execute("SYSTEM FLUSH").unwrap();

    let (tx, rx) = std::sync::mpsc::channel::<Result<(u64, u64), String>>();
    let d = db.clone();
    std::thread::spawn(move || {
        let out = (|| -> Result<(u64, u64), String> {
            let a = d.reader().cursor("SELECT id FROM t").map_err(|e| e.to_string())?;
            // The writer lands between the two cursors, which is what puts it
            // in the queue ahead of the second one.
            let w = d.clone();
            let writer = std::thread::spawn(move || {
                let mut sql = String::with_capacity(1024);
                write_batch(&w, &mut sql, 16);
            });
            std::thread::sleep(Duration::from_millis(50));
            let b = d.reader().cursor("SELECT id FROM t").map_err(|e| e.to_string())?;

            // Interleaved on purpose: neither cursor is drained before the
            // other has produced.
            let (mut na, mut nb) = (0u64, 0u64);
            let (mut ia, mut ib) = (a.into_iter(), b.into_iter());
            loop {
                let ga = ia.next();
                let gb = ib.next();
                if ga.is_none() && gb.is_none() {
                    break;
                }
                if let Some(g) = ga {
                    na += g.map_err(|e| e.to_string())?.rows() as u64;
                }
                if let Some(g) = gb {
                    nb += g.map_err(|e| e.to_string())?.rows() as u64;
                }
            }
            writer.join().map_err(|_| "writer panicked".to_string())?;
            Ok((na, nb))
        })();
        let _ = tx.send(out);
    });

    match rx.recv_timeout(Duration::from_secs(30)) {
        Ok(Ok((na, nb))) => {
            for (who, n) in [("cursor a", na), ("cursor b", nb)] {
                assert_eq!(n % K, 0, "{who} yielded {n} rows, not a whole number of batches");
                assert!(n >= 16 * K, "{who} yielded {n} rows, fewer than the 16 batches it opened over");
            }
        }
        Ok(Err(e)) => panic!("interleaved cursors failed: {e}"),
        Err(_) => panic!(
            "DEADLOCK: two open cursors with an INSERT queued between them made no progress \
             in 30 s. A pool with two open portals and one write wedges."
        ),
    }
}

/// One reader's cancellation must not stop another's query.
///
/// `Reader::cancel_handle` hands out the *session's* flag by default, so two
/// handles from one `Db` share it -- documented, and the reason
/// `with_own_cancel` exists. This pins both halves: that the shared flag really
/// is shared (so `KILL` reaches a pool), that `with_own_cancel` really does
/// isolate (so it does not reach the whole pool), and that a cancelled handle
/// recovers with `resume` instead of staying dead.
#[test]
fn cancellation_is_per_handle_when_asked_for() {
    let _w = Watchdog::new("cancellation_is_per_handle", Duration::from_secs(60));
    let db = Db::in_memory();
    db.execute(DDL).unwrap();
    let mut sql = String::with_capacity(1 << 16);
    for b in 0..8 {
        write_batch(&db, &mut sql, b);
    }

    let shared_a = db.reader();
    let shared_b = db.reader();
    let own = db.reader().with_own_cancel();

    shared_a.cancel_handle().store(true, Ordering::Relaxed);
    assert!(shared_a.query(PROBE).is_err(), "a cancelled handle answered anyway");
    assert!(
        shared_b.query(PROBE).is_err(),
        "the second handle kept running: the session-wide flag is not shared, so a pool's \
         KILL QUERY would miss every other connection"
    );
    assert!(
        own.query(PROBE).is_ok(),
        "`with_own_cancel` did not isolate the handle: one connection's KILL stopped another"
    );

    shared_a.resume();
    assert!(shared_a.query(PROBE).is_ok(), "`resume` did not revive the handle");
    assert!(shared_b.query(PROBE).is_ok(), "`resume` on one handle did not revive its twin");

    // The flag is raised on a *different* thread from the one running the
    // query -- which is what a `KILL QUERY` from a control connection is.
    //
    // The helper is joined before the claim is checked, deliberately. The
    // first version of this raced a 5 ms timer against a loop of probes and
    // passed alone and failed under `cargo test`'s parallelism: with 56 other
    // threads on the machine the helper simply did not get scheduled inside
    // the loop's budget. "The flag is set, therefore the next query refuses"
    // is the same claim without the timer in it.
    let victim = db.reader().with_own_cancel();
    let flag = victim.cancel_handle();
    std::thread::spawn(move || flag.store(true, Ordering::Relaxed)).join().unwrap();
    assert!(
        victim.query(PROBE).is_err(),
        "a cancel raised on another thread is not visible to the query"
    );
    victim.resume();
    assert!(victim.query(PROBE).is_ok());
}

/// A per-handle memory ceiling and deadline must reach the *parallel*
/// operators, and must not leak into the handle next to them.
///
/// This is the Phase 2 failure mode stated as a test: the budget and the
/// deadline are built, and the question is only whether a `Reader` -- the type
/// a pool actually holds -- can impose them on a query that the exchange has
/// sharded across the pool. A limit that the serial path honours and the
/// parallel path ignores is worse than no limit at all, because the query it
/// fails to stop is the big one.
#[test]
fn per_handle_limits_reach_the_parallel_operators() {
    let _w = Watchdog::new("per_handle_limits", Duration::from_secs(120));
    let db = Db::in_memory();
    db.execute(DDL).unwrap();

    // 120k rows, every one its own group: the aggregate has to hold 120k
    // entries, which is megabytes, so a 64 KiB ceiling is unambiguous.
    const ROWS: u64 = 120_000;
    let mut sql = String::with_capacity(ROWS as usize * 26);
    sql.push_str("INSERT INTO t VALUES ");
    for i in 0..ROWS {
        if i > 0 {
            sql.push(',');
        }
        write!(sql, "({i},{i},'r{i}')").unwrap();
    }
    db.execute(&sql).unwrap();
    db.execute("SYSTEM FLUSH").unwrap();

    const WIDE: &str = "SELECT b, count(), sum(id) FROM t GROUP BY b";
    let plain = db.reader();
    assert_eq!(plain.query(WIDE).unwrap().rows() as u64, ROWS, "the unconstrained query is wrong");

    let tight = db.reader().with_memory_limit(64 << 10);
    let e = tight
        .query(WIDE)
        .err()
        .expect("a 120k-group aggregate ran inside a 64 KiB budget: the ceiling never reached the exchange");
    assert!(
        e.to_string().to_lowercase().contains("memor"),
        "the refusal was not about memory: {e}"
    );

    let quick = db.reader().with_timeout(Duration::from_nanos(1));
    let e = quick
        .query(WIDE)
        .err()
        .expect("a 1 ns deadline did not stop a 120k-group aggregate");
    let msg = e.to_string().to_lowercase();
    assert!(
        msg.contains("timeout") || msg.contains("deadline") || msg.contains("cancel"),
        "the refusal was not about the deadline: {e}"
    );

    // Per handle, not per session: the unconstrained handle beside them is
    // still fine, and so is a fresh one taken after the refusals.
    assert_eq!(plain.query(WIDE).unwrap().rows() as u64, ROWS, "one handle's limit leaked");
    assert_eq!(db.reader().query(WIDE).unwrap().rows() as u64, ROWS);

    // And under concurrency: eight constrained handles failing at once must
    // not stop eight unconstrained ones running at once.
    let n = threads_available().max(2);
    let bad: Vec<_> = (0..n)
        .map(|_| {
            let r = db.reader().with_memory_limit(64 << 10);
            std::thread::spawn(move || r.query(WIDE).is_err())
        })
        .collect();
    let good: Vec<_> = (0..n)
        .map(|_| {
            let r = db.reader();
            std::thread::spawn(move || r.query(WIDE).map(|rs| rs.rows() as u64))
        })
        .collect();
    for h in bad {
        assert!(h.join().unwrap(), "a constrained handle succeeded under contention");
    }
    for h in good {
        assert_eq!(
            h.join().unwrap().expect("an unconstrained handle failed"),
            ROWS,
            "a neighbouring handle's memory refusal changed this one's answer"
        );
    }
}

/// Two processes, one directory: the second must be refused, not allowed to
/// interleave writes into the first one's files.
///
/// This is the concurrency case the in-process tests structurally cannot
/// reach, and the one that destroyed committed writes before the `flock`
/// landed. Driven through the binary because a lock that only holds inside one
/// process is not a lock.
#[test]
fn a_second_writer_process_is_refused() {
    let _w = Watchdog::new("a_second_writer_process_is_refused", Duration::from_secs(60));
    let dir = Scratch::new("two-procs");
    let bin = env!("CARGO_BIN_EXE_granular");
    let path = dir.path().to_str().unwrap();

    let out = std::process::Command::new(bin)
        .args(["--data", path, "-q", DDL])
        .output()
        .expect("spawn");
    assert!(out.status.success(), "setup failed: {}", String::from_utf8_lossy(&out.stderr));

    // Hold the directory open in-process, then ask the binary for it.
    let db = Db::open(dir.path()).unwrap();
    let out = std::process::Command::new(bin)
        .args(["--data", path, "-q", "INSERT INTO t VALUES (1,1,'x')"])
        .output()
        .expect("spawn");
    assert!(
        !out.status.success(),
        "a second writer process opened a locked directory and reported success"
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("already open by another granular process"),
        "the refusal did not say why: {err}"
    );

    // Released on drop, so the next process gets in.
    drop(db);
    let out = std::process::Command::new(bin)
        .args(["--data", path, "-q", "INSERT INTO t VALUES (1,1,'x'); SELECT count() FROM t"])
        .args(["--format", "tsv", "--no-header"])
        .output()
        .expect("spawn");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "1");
}
