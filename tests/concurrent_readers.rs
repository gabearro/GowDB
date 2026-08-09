//! Two threads must be able to query at once, and the answers must be the ones
//! a single thread would have given.
//!
//! This file exists because the capability was *built* and never reachable.
//! `Table::committed_snapshot`, whose own doc says it is what a reader that is
//! not the writing session must see, had zero non-test callers in the tree;
//! `Table::freeze_delta` had none at all. Underneath, an 8-thread run straight
//! against the exec and storage layers produced 0 mismatches. Everything above
//! them took `&mut Session`, because `Session::plan` opened with
//! `catalog.flush_all()` -- so a read-only query on `a` took exclusive write
//! access to `b`, `c` and `d`. Eight identical 2M-row queries behind an
//! `Arc<Mutex<Session>>` measured 33.03 ms on one thread, 32.34 on two, 31.58
//! on four. Perfectly flat. Not contention: the type system.
//!
//! So every test here drives the *public* facade -- `Db`, `Reader`, `Cursor`,
//! `Session` -- from real threads. A regression that made the read path take
//! `&mut` again would fail to compile, and one that made it serialize again
//! would fail `readers_overlap_and_scale`.
//!
//! The audit's own experiment, re-run against both facades in one binary --
//! eight identical 2M-row `GROUP BY` queries, best of three per width:
//!
//! ```text
//!   threads   Arc<Mutex<Session>>    Reader
//!         1              49.69 ms   50.15 ms
//!         2              53.97 ms   43.21 ms
//!         4              64.01 ms   45.20 ms
//!         8              62.81 ms   46.67 ms
//!        14              62.02 ms   46.21 ms
//! ```
//!
//! The mutex column is the flat curve, and slightly *worse* with more threads
//! -- that is lock handoff, not work. The gain is only 1.35x rather than the
//! 5.9x in `readers_overlap_and_scale` because a 2M-row query is already
//! sharded across every core by the exchange, so concurrent queries are
//! sharing the cores they were going to use anyway; what overlaps is the
//! serial head and tail of each one. The N-way win is for the workload a
//! connection pool actually has -- many queries that are individually small.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use granular::types::Value;
use granular::{Db, Session, StreamItem};

// ---------------------------------------------------------------- fixtures

/// A unique scratch directory per test, removed on drop. Same shape as the one
/// in `tests/persistence.rs` -- pid plus a counter, so a failure is
/// reproducible and two tests never collide.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "granular-cr-{}-{}-{}",
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

/// `rows` rows of `(id, grp, s)`, bulk-loaded and flushed into parts.
///
/// One `INSERT ... VALUES` with every row in it: the point of these tests is
/// the read path, and a row-at-a-time load would spend the whole runtime in
/// the writer.
fn seed(db: &Db, rows: u64) {
    db.execute(
        "CREATE TABLE t (id UInt64, grp UInt32, s String) ENGINE = MergeTree ORDER BY id",
    )
    .unwrap();
    let mut sql = String::with_capacity(rows as usize * 24);
    sql.push_str("INSERT INTO t VALUES ");
    for i in 0..rows {
        if i > 0 {
            sql.push(',');
        }
        sql.push_str(&format!("({i},{},'row-{i}-abc')", i % 97));
    }
    db.execute(&sql).unwrap();
    db.writer().execute("SYSTEM FLUSH").unwrap();
}

fn scalar_u64(rs: &granular::ResultSet) -> u64 {
    match rs.scalar().expect("one cell") {
        Value::UInt(n) => n,
        Value::Int(n) => n as u64,
        other => panic!("expected a number, got {other}"),
    }
}

/// The measured query. Deliberately *not* one the exchange will shard -- see
/// `readers_overlap_and_scale`.
const HEAVY: &str = "SELECT count() FROM t WHERE s LIKE '%7-abc%' AND grp < 90";

fn threads_available() -> usize {
    std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1)
}

/// Held by every test in this file that saturates the machine.
///
/// `cargo test` runs the tests *within* a binary in parallel, and every test
/// here spawns one thread per core: eight of them at once means each measures
/// a machine that is already full. That is fine for the correctness tests and
/// fatal for `readers_overlap_and_scale`, which caught it -- a scaling ratio
/// measured against sibling tests hammering the same cores failed roughly one
/// run in three. Serializing them costs about a second of wall clock for the
/// whole file and makes every number in it mean what it says.
///
/// `into_inner` rather than `unwrap`: a panic in one test must report *that*
/// test's failure, not turn every later one into a poisoned-lock panic.
fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

// ------------------------------------------------- (a) correctness, threaded

/// N threads, one shared `Reader`, and every answer identical to the one a
/// single thread produced first.
///
/// The point is not that counting is hard. It is that the whole read path --
/// parse, bind, optimize, lower, pin a snapshot, decode, aggregate -- runs
/// from `&self` on several threads at once and shares no mutable state. A
/// `Reader` that pinned a per-thread scratch buffer, or a planner that cached
/// into the catalog, would show up here as a wrong answer or a data race.
#[test]
fn concurrent_readers_agree_with_a_serial_reference() {
    let _one = exclusive();
    let db = Db::in_memory();
    seed(&db, 20_000);

    let queries = [
        "SELECT count() FROM t",
        "SELECT sum(id) FROM t WHERE id % 3 = 0",
        "SELECT count() FROM t WHERE s LIKE '%7-abc%'",
        "SELECT grp, count() FROM t GROUP BY grp ORDER BY grp LIMIT 5",
        "SELECT id FROM t ORDER BY id DESC LIMIT 3",
        "SELECT count() FROM t a JOIN t b ON a.id = b.id WHERE a.id < 500",
        "SELECT count() FROM (SELECT id FROM t WHERE id IN (SELECT id FROM t WHERE id < 40))",
    ];

    // The reference, taken serially through the same facade.
    let reader = db.reader();
    let want: Vec<Vec<Vec<Value>>> = queries
        .iter()
        .map(|q| reader.query(q).unwrap().to_values())
        .collect();

    let n = threads_available().clamp(2, 8);
    let start = Arc::new(Barrier::new(n));
    std::thread::scope(|scope| {
        for _ in 0..n {
            let r = reader.clone();
            let start = Arc::clone(&start);
            let want = &want;
            scope.spawn(move || {
                start.wait();
                for _ in 0..8 {
                    for (i, q) in queries.iter().enumerate() {
                        let got = r.query(q).unwrap().to_values();
                        assert_eq!(got, want[i], "`{q}` differed under concurrency");
                    }
                }
            });
        }
    });
}

// ------------------------------------------------------- (b) they overlap

/// Overlap proved without a clock: N queries stop *inside* the engine and wait
/// for each other.
///
/// Each thread streams a query and, on its first block -- with the read path
/// entered, the plan built, the snapshot pinned and the scan running --
/// announces itself and waits for the other N-1. A facade that serialized
/// reads cannot get past one announcement, so the wait times out and this
/// fails; a facade that runs them at once passes in microseconds. Unlike a
/// throughput measurement this cannot be fooled by a loaded machine, which is
/// why it is the assertion and the curve below is the number.
#[test]
fn n_queries_are_inside_the_engine_at_the_same_instant() {
    let _one = exclusive();
    let db = Db::in_memory();
    seed(&db, 4_000);
    let reader = db.reader();

    let n = threads_available().clamp(2, 6);
    let arrived = Arc::new(AtomicUsize::new(0));
    let passed = Arc::new(AtomicUsize::new(0));
    std::thread::scope(|scope| {
        for _ in 0..n {
            let r = reader.clone();
            let (arrived, passed) = (Arc::clone(&arrived), Arc::clone(&passed));
            scope.spawn(move || {
                let mut announced = false;
                r.stream("SELECT id FROM t WHERE id < 2000", &mut |item| {
                    if let StreamItem::Rows(_) = item {
                        if !announced {
                            announced = true;
                            arrived.fetch_add(1, Ordering::AcqRel);
                            // Generous: this is a liveness bound, not a
                            // measurement. A serialized facade blocks here
                            // forever and fails; a concurrent one is through
                            // in microseconds.
                            let t0 = Instant::now();
                            while arrived.load(Ordering::Acquire) < n
                                && t0.elapsed() < Duration::from_secs(10)
                            {
                                std::thread::yield_now();
                            }
                            if arrived.load(Ordering::Acquire) >= n {
                                passed.fetch_add(1, Ordering::AcqRel);
                            }
                        }
                    }
                    Ok(())
                })
                .unwrap();
            });
        }
    });
    assert_eq!(
        passed.load(Ordering::Acquire),
        n,
        "only {} of {n} queries were ever executing at the same instant: the read \
         path is still serialized",
        passed.load(Ordering::Acquire)
    );
}

/// The audit's flat 33/32/31 ms curve, disproved twice over: by wall clock,
/// and by counting how many queries were inside the engine at once.
///
/// Two things make the measurement mean what it says:
///
///   * the query is one the exchange declines to shard, asserted through
///     `EXPLAIN PIPELINE` below. If it went parallel, one thread would already
///     saturate the machine and a flat curve would prove nothing.
///   * the answer is checked, so a "fast" run that skipped the work fails.
///
/// The assertion is deliberately loose (1.6x at four threads against a
/// theoretical 4x) because this machine swings 30% on identical code. The
/// failure it is built to catch is 1.0x, which is what a `Mutex<Session>` --
/// or a read path that took the writer lock -- produces.
///
/// Measured, 14 cores (10 performance + 4 efficiency), best of three per width
/// over six runs:
///
/// ```text
///   threads   queries/s   speedup
///         1        1538      1.00
///         2        2786      1.81
///         4        5103      3.32
///         8        8520      5.54
///        14        9270      6.03
/// ```
///
/// Sublinear and rising, which is the win. The ceiling is not this facade: an
/// identical fan-out over a bare `&Session`, with no lock at all and no
/// `has_pending_writes` test, measures 1560 / 2712 / 4067 / 7339 / 8894 -- the
/// same curve inside the noise, so the shared `RwLock` costs nothing at
/// fourteen threads. What runs out is the machine (four of the fourteen cores
/// are efficiency cores) and the process allocator, both below anything in
/// this change.
#[test]
fn readers_overlap_and_scale() {
    let _one = exclusive();
    let db = Db::in_memory();
    seed(&db, 12_000);
    let reader = db.reader();

    let pipeline = reader.query(&format!("EXPLAIN PIPELINE {HEAVY}")).unwrap().to_string();
    assert!(
        !pipeline.contains("Exchange"),
        "this test measures reader concurrency, so the query must be serial \
         inside one reader; the planner sharded it:\n{pipeline}"
    );
    let want = scalar_u64(&reader.query(HEAVY).unwrap());
    assert!(want > 0, "the measured query must actually match rows");

    // How many `query` calls were inside the engine simultaneously. A serial
    // facade cannot get this above 1 no matter how many threads call it.
    let inflight = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));

    let run = |threads: usize, per_thread: usize| -> Duration {
        let start = Arc::new(Barrier::new(threads));
        let t0 = Instant::now();
        std::thread::scope(|scope| {
            for _ in 0..threads {
                let r = reader.clone();
                let start = Arc::clone(&start);
                let (inflight, peak) = (Arc::clone(&inflight), Arc::clone(&peak));
                scope.spawn(move || {
                    start.wait();
                    for _ in 0..per_thread {
                        let now = inflight.fetch_add(1, Ordering::AcqRel) + 1;
                        peak.fetch_max(now, Ordering::AcqRel);
                        let got = scalar_u64(&r.query(HEAVY).unwrap());
                        inflight.fetch_sub(1, Ordering::AcqRel);
                        assert_eq!(got, want);
                    }
                });
            }
        });
        t0.elapsed()
    };

    // Warm: first touch of the pool, the parts and the string dictionary.
    run(1, 20);

    let widths: Vec<usize> = [1usize, 2, 4, 8, 14]
        .into_iter()
        .filter(|&w| w <= threads_available().max(1))
        .collect();
    const PER_THREAD: usize = 60;
    // Widths interleaved rather than one width at a time, best-of-N per width:
    // the other tests in this binary run alongside this one, so load drifts,
    // and measuring all of width 1 before any of width 4 would attribute that
    // drift to the width. Same total work, honest comparison.
    let mut best = vec![Duration::MAX; widths.len()];
    for _ in 0..3 {
        for (i, &w) in widths.iter().enumerate() {
            best[i] = best[i].min(run(w, PER_THREAD));
        }
    }
    let curve: Vec<(usize, f64)> = widths
        .iter()
        .zip(&best)
        .map(|(&w, dt)| (w, (w * PER_THREAD) as f64 / dt.as_secs_f64()))
        .collect();
    for ((w, qps), dt) in curve.iter().zip(&best) {
        println!("{w:>3} threads  {:>8.2} ms  {qps:>10.0} queries/s", dt.as_secs_f64() * 1e3);
    }

    assert!(
        peak.load(Ordering::Acquire) >= 2,
        "no two queries were ever inside the engine at the same time"
    );
    if widths.contains(&4) {
        let one = curve[0].1;
        let four = curve.iter().find(|(w, _)| *w == 4).unwrap().1;
        // 1.5x against the 3.05x measured below, because this machine swings
        // 30% on identical code even with `exclusive()` keeping the siblings
        // out. The failure worth catching is 1.0x -- what a `Mutex<Session>`,
        // or a read path that took the writer lock, produces however quiet the
        // machine is -- and
        // `n_queries_are_inside_the_engine_at_the_same_instant` proves the
        // same claim without a clock at all.
        assert!(
            four >= one * 1.5,
            "reads did not scale: {one:.0} q/s on 1 thread, {four:.0} on 4 \
             ({:.2}x). The pre-split facade measured 1.0x.",
            four / one
        );
    }
}

// ------------------------------ (c) a snapshot, while a writer commits

/// A reader sees the database before a transaction or after it, never inside.
///
/// The writer commits `BATCH` rows as one transaction, over and over, on its
/// own thread. Every reader asserts the row count is a multiple of `BATCH` --
/// a count of `BATCH * k + 3` would mean a reader got into the middle of a
/// commit, which is the torn read the pinned-snapshot design exists to make
/// impossible. It also asserts it saw the count *move*, so a run where the
/// writer never got scheduled cannot pass by accident.
#[test]
fn a_reader_sees_a_consistent_snapshot_while_a_writer_commits() {
    let _one = exclusive();
    const BATCH: u64 = 8;
    const COMMITS: u64 = 60;

    let db = Db::in_memory();
    db.execute("CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id")
        .unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let reader = db.reader();
    let seen = Arc::new(AtomicUsize::new(0));

    std::thread::scope(|scope| {
        // The writer: `COMMITS` transactions of `BATCH` single-row inserts.
        {
            let db = db.clone();
            let stop = Arc::clone(&stop);
            scope.spawn(move || {
                for c in 0..COMMITS {
                    db.transaction(|s| {
                        for r in 0..BATCH {
                            s.execute(&format!("INSERT INTO t VALUES ({})", c * BATCH + r))?;
                        }
                        Ok(())
                    })
                    .unwrap();
                    std::thread::yield_now();
                }
                stop.store(true, Ordering::Release);
            });
        }
        for _ in 0..threads_available().clamp(2, 4) {
            let r = reader.clone();
            let stop = Arc::clone(&stop);
            let seen = Arc::clone(&seen);
            scope.spawn(move || {
                let mut distinct = std::collections::BTreeSet::new();
                while !stop.load(Ordering::Acquire) {
                    let n = scalar_u64(&r.query("SELECT count() FROM t").unwrap());
                    assert_eq!(
                        n % BATCH,
                        0,
                        "a reader saw {n} rows, which is {} rows into a transaction that \
                         had not committed",
                        n % BATCH
                    );
                    // Every row up to the count must be there: a reader that
                    // saw the count of one snapshot and the rows of another
                    // would report a hole.
                    let m = scalar_u64(
                        &r.query(&format!("SELECT count() FROM t WHERE id < {n}")).unwrap(),
                    );
                    assert!(m >= n, "count said {n} but only {m} of those ids were visible");
                    distinct.insert(n);
                }
                seen.fetch_max(distinct.len(), Ordering::AcqRel);
            });
        }
    });

    assert_eq!(
        scalar_u64(&reader.query("SELECT count() FROM t").unwrap()),
        BATCH * COMMITS
    );
    assert!(
        seen.load(Ordering::Acquire) >= 2,
        "no reader ever observed the table change, so nothing was concurrent"
    );
}

/// Autocommit writes, not transactions: a reader must never fail *because* a
/// writer is busy.
///
/// A small `INSERT` lands in the delta, which a scan cannot see, so the reader
/// has to flush before it can answer. Doing that as "take the writer lock,
/// flush, drop it, take the reader lock" loses to a writer that buffers again
/// in the gap, and the reader would report "this session has buffered writes"
/// -- a failure with nothing the caller could do about it. Every query here
/// must succeed and count a multiple of the batch size.
#[test]
fn a_reader_never_fails_because_a_writer_is_buffering() {
    let _one = exclusive();
    const BATCH: u64 = 4;

    let db = Db::in_memory();
    db.execute("CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id")
        .unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let reads = Arc::new(AtomicUsize::new(0));

    std::thread::scope(|scope| {
        {
            let (db, stop, reads) = (db.clone(), Arc::clone(&stop), Arc::clone(&reads));
            scope.spawn(move || {
                for c in 0..200u64 {
                    // One statement, one autocommit, four rows into the delta.
                    let vals: Vec<String> =
                        (0..BATCH).map(|r| format!("({})", c * BATCH + r)).collect();
                    let before = reads.load(Ordering::Acquire);
                    db.execute(&format!("INSERT INTO t VALUES {}", vals.join(",")))
                        .unwrap();
                    // Pace the writer on the readers rather than racing them.
                    // 200 inserts of four rows is a few milliseconds, and on a
                    // loaded machine the whole loop can finish before a reader
                    // thread is scheduled at all -- which used to fail the run
                    // for "the readers never ran", having disproved nothing.
                    // Waiting for one read per insert makes the interleaving a
                    // property of the test instead of the scheduler.
                    //
                    // Bounded, so a reader that panics ends the run at its own
                    // assertion instead of hanging this thread in `scope`.
                    let mut spins = 0u32;
                    while reads.load(Ordering::Acquire) == before && spins < 10_000_000 {
                        spins += 1;
                        std::thread::yield_now();
                    }
                }
                stop.store(true, Ordering::Release);
            });
        }
        for _ in 0..threads_available().clamp(2, 4) {
            let r = db.reader();
            let (stop, reads) = (Arc::clone(&stop), Arc::clone(&reads));
            scope.spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    let n = scalar_u64(
                        &r.query("SELECT count() FROM t")
                            .expect("a read must not fail because a writer is buffering"),
                    );
                    assert_eq!(n % BATCH, 0, "a reader saw part of an INSERT: {n} rows");
                    reads.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
    // One read per insert is guaranteed by the pacing above, so this is a
    // floor on the pacing having worked, not a hope about scheduling.
    assert!(reads.load(Ordering::Relaxed) > 10, "the readers never ran");
    assert_eq!(
        scalar_u64(&db.reader().query("SELECT count() FROM t").unwrap()),
        800
    );
}

// ------------------------------------------------------- streaming results

/// A result that never exists all at once.
///
/// `Cursor` is the pull side: the producer parks until the consumer takes the
/// previous block, so the peak is two blocks whatever the row count. The test
/// checks the rows are all there and in order, that the schema is known
/// *before* the first block (a portal has to answer `Describe` first), and
/// that abandoning a cursor half way releases the database rather than
/// wedging the next writer.
#[test]
fn a_cursor_streams_and_an_abandoned_one_releases_the_writer() {
    let _one = exclusive();
    let db = Db::in_memory();
    seed(&db, 30_000);
    let reader = db.reader();

    let mut cur = reader.cursor("SELECT id FROM t ORDER BY id").unwrap();
    assert_eq!(cur.schema().len(), 1, "the shape is known before the first row");
    let mut blocks = 0usize;
    let mut next = 0u64;
    for b in cur.by_ref() {
        let b = b.unwrap();
        for r in 0..b.rows() {
            assert_eq!(b.column(0).value(r), Value::UInt(next));
            next += 1;
        }
        blocks += 1;
    }
    assert_eq!(next, 30_000);
    assert!(blocks > 1, "a 30k-row result arrived in one block, so nothing streamed");

    // Abandoned after one block: `Drop` cancels the query, joins the producer
    // and releases the shared lock. If it did not, this write would hang.
    let mut cur = reader.cursor("SELECT id FROM t ORDER BY id").unwrap();
    assert!(cur.next().is_some());
    drop(cur);
    db.execute("INSERT INTO t VALUES (999999, 1, 'late')").unwrap();
    assert_eq!(
        scalar_u64(&reader.query("SELECT count() FROM t").unwrap()),
        30_001
    );
}

/// The push side: `Session::read_stream` hands over one block at a time and
/// announces the schema first, even for an empty result.
#[test]
fn read_stream_announces_its_schema_before_any_row() {
    let db = Db::in_memory();
    seed(&db, 5_000);
    let reader = db.reader();

    let mut heads = 0;
    let mut rows = 0;
    let mut widest = 0;
    reader
        .stream("SELECT id, s FROM t WHERE id < 4000", &mut |item| {
            match item {
                StreamItem::Head(s) => {
                    assert_eq!(rows, 0, "the head must arrive before any row");
                    assert_eq!(s.len(), 2);
                    heads += 1;
                }
                StreamItem::Rows(b) => {
                    rows += b.rows();
                    widest = widest.max(b.rows());
                }
            }
            Ok(())
        })
        .unwrap();
    assert_eq!((heads, rows), (1, 4000));
    assert!(widest <= 8192, "a block is bounded, so the peak is bounded");

    // An empty result still describes itself, which is the case a wire
    // protocol cannot get from "the first block".
    let mut heads = 0;
    let n = reader
        .stream("SELECT id FROM t WHERE id > 1000000", &mut |item| {
            if let StreamItem::Head(_) = item {
                heads += 1;
            }
            Ok(())
        })
        .unwrap();
    assert_eq!((heads, n), (1, 0));
}

// ------------------------------------------------- per-query governance

/// A budget, a deadline and a cancel flag, all reachable from the facade.
///
/// Each of these existed in `operators::QueryContext` and was reachable only
/// from the operator tests: every `Session` entry point built a fresh default
/// context, so a memory limit could not be set and a running query could not
/// be stopped. That is exactly the "landed in src/, never handed to Session"
/// defect this phase is about.
#[test]
fn limits_and_cancellation_reach_the_engine_from_the_facade() {
    let db = Db::in_memory();
    seed(&db, 40_000);

    // Budget: a grouped aggregate over 40k distinct keys cannot fit in 64 KiB.
    let tight = db.reader().with_memory_limit(64 << 10);
    let e = tight
        .query("SELECT id, count() FROM t GROUP BY id")
        .expect_err("a 40k-group aggregate must not fit in 64 KiB");
    assert!(e.to_string().contains("memory budget"), "{e}");
    // ...and the same reader answers a query that does fit, so the budget is a
    // limit and not a switch.
    assert_eq!(scalar_u64(&tight.query("SELECT count() FROM t").unwrap()), 40_000);

    // Deadline.
    let brief = db.reader().with_timeout(Duration::from_nanos(1));
    let e = brief
        .query("SELECT count() FROM t")
        .expect_err("a 1 ns deadline must stop the query");
    assert!(e.to_string().contains("deadline"), "{e}");

    // Cancellation, from another thread, while the query runs.
    let slow = db.reader().with_own_cancel();
    let flag = slow.cancel_handle();
    let killer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(2));
        flag.store(true, Ordering::Relaxed);
    });
    let mut refused = 0;
    for _ in 0..200 {
        if slow
            .query("SELECT id, count() FROM t GROUP BY id ORDER BY id")
            .is_err()
        {
            refused += 1;
        }
    }
    killer.join().unwrap();
    assert!(refused > 0, "setting the cancel flag stopped nothing");
    // The flag is the reader's own, so the rest of the database is unaffected.
    assert_eq!(
        scalar_u64(&db.reader().query("SELECT count() FROM t").unwrap()),
        40_000
    );
    slow.resume();
    assert!(slow.query("SELECT count() FROM t").is_ok(), "resume must clear it");
}

// ------------------------------------------------------------- read-only

/// Several read-only sessions share one directory; a writer excludes them.
///
/// The exclusive `flock` is what stops two writers from allocating the same
/// part sequence number and overwriting each other's committed data. A
/// session that cannot write cannot do that, so it takes `LOCK_SH` and several
/// of them coexist -- which is what a query fleet over a checkpointed database
/// wants.
#[test]
fn read_only_sessions_share_a_directory_and_refuse_writes() {
    let _one = exclusive();
    let dir = Scratch::new("ro");
    {
        let mut w = Session::open(dir.path()).unwrap();
        w.execute("CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id")
            .unwrap();
        w.execute("INSERT INTO t VALUES (1),(2),(3)").unwrap();
        w.checkpoint().unwrap();
    }

    let a = Session::open_read_only(dir.path()).unwrap();
    let b = Session::open_read_only(dir.path()).unwrap();
    assert!(a.is_read_only() && b.is_read_only());
    for s in [&a, &b] {
        assert_eq!(scalar_u64(&s.read("SELECT count() FROM t").unwrap()), 3);
    }

    // Refused, and by name.
    let e = a
        .read("INSERT INTO t VALUES (4)")
        .expect_err("a read path must refuse a write");
    assert!(e.to_string().contains("read-only"), "{e}");
    let mut c = Session::open_read_only(dir.path()).unwrap();
    let e = c.execute("INSERT INTO t VALUES (4)").unwrap_err();
    assert!(e.to_string().contains("read-only"), "{e}");
    assert!(c.checkpoint().is_err(), "a read-only session must not checkpoint");

    // A writer cannot join them, and they cannot join a writer.
    assert!(
        Session::open(dir.path()).is_err(),
        "a writer must not open a directory held by readers"
    );
    drop((a, b, c));
    let w = Session::open(dir.path()).unwrap();
    assert!(
        Session::open_read_only(dir.path()).is_err(),
        "a reader must not open a directory held by a writer"
    );
    drop(w);

    // Concurrent read-only sessions in one process, on real threads.
    let shared = Db::open_read_only(dir.path()).unwrap();
    let r = shared.reader();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let r = r.clone();
            scope.spawn(move || {
                for _ in 0..50 {
                    assert_eq!(scalar_u64(&r.query("SELECT count() FROM t").unwrap()), 3);
                }
            });
        }
    });
}

// -------------------------------------------- the read path never lies

/// A read that cannot see buffered rows says so instead of answering short.
///
/// `Session::read` takes `&self` and therefore cannot flush. The engine's
/// scans read parts, so a table with a non-empty delta would answer with
/// however many rows happened to be packed already -- the exact shape of
/// silent data loss this project keeps finding. The `&self` path refuses; the
/// `Reader` takes the writer lock for one flush and answers correctly.
#[test]
fn a_read_that_cannot_flush_refuses_rather_than_answering_short() {
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id")
        .unwrap();
    s.execute("INSERT INTO t VALUES (1),(2),(3)").unwrap();
    // Buffered in the delta, not yet in a part: the `&self` read cannot answer.
    let e = s.read("SELECT count() FROM t").expect_err("must refuse");
    assert!(e.to_string().contains("buffered writes"), "{e}");
    // The `&mut` path flushes and answers.
    assert_eq!(scalar_u64(&s.query("SELECT count() FROM t").unwrap()), 3);
    assert_eq!(scalar_u64(&s.read("SELECT count() FROM t").unwrap()), 3);

    // Through `Db`, the flush is taken care of: the reader upgrades to the
    // writer lock once, then answers.
    let db = Db::in_memory();
    db.execute("CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id")
        .unwrap();
    db.execute("INSERT INTO t VALUES (1),(2),(3)").unwrap();
    assert_eq!(
        scalar_u64(&db.reader().query("SELECT count() FROM t").unwrap()),
        3,
        "a reader must see every acknowledged write, buffered or not"
    );

    // And a write is refused by the read path whatever it is spelled like.
    let r = db.reader();
    for sql in [
        "INSERT INTO t VALUES (9)",
        "CREATE TABLE u (id UInt64) ENGINE = MergeTree ORDER BY id",
        "ALTER TABLE t DELETE WHERE id = 1",
        "OPTIMIZE TABLE t",
        "SYSTEM FLUSH",
        "USE default",
    ] {
        assert!(r.query(sql).is_err(), "`{sql}` is not a read");
    }
    assert_eq!(scalar_u64(&r.query("SELECT count() FROM t").unwrap()), 3);
}

/// A reader must not see a transaction's private overlay.
///
/// `Table::snapshot` hands the overlay to whoever asks once `begin_txn` has
/// run, so a read that landed there mid-transaction would be a dirty read of
/// rows a `ROLLBACK` is still entitled to erase. `Db::transaction` holds the
/// writer lock so a reader cannot get in; driving `BEGIN` through a raw
/// `Db::writer` guard and dropping it is the one way to leave one open, and
/// the reader refuses rather than reading through it.
#[test]
fn a_reader_refuses_to_read_through_an_open_transaction() {
    let db = Db::in_memory();
    db.execute("CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id")
        .unwrap();
    db.execute("INSERT INTO t VALUES (1),(2)").unwrap();

    {
        let mut w = db.writer();
        w.begin().unwrap();
        w.execute("INSERT INTO t VALUES (3)").unwrap();
        // The writer sees its own writes; the guard is dropped below with the
        // transaction still open.
        assert_eq!(scalar_u64(&w.query("SELECT count() FROM t").unwrap()), 3);
    }
    let e = db
        .reader()
        .query("SELECT count() FROM t")
        .expect_err("reading through an open transaction is a dirty read");
    assert!(e.to_string().contains("transaction is open"), "{e}");

    db.writer().rollback().unwrap();
    assert_eq!(
        scalar_u64(&db.reader().query("SELECT count() FROM t").unwrap()),
        2,
        "the rolled-back row must be gone"
    );
}
