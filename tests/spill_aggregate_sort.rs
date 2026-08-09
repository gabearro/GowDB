//! Spilling has to be reachable from the front door, in parallel, and it has to
//! take its files with it.
//!
//! The defect this file exists for: `GROUP BY` and `ORDER BY` both learned to
//! spill, and then the exchange fanned them out to fourteen workers and none of
//! the workers could. A `GROUP BY` a serial plan answered under a tight budget
//! became an error the moment the table was big enough to go parallel -- which
//! is to say, the moment spilling was worth having. Every test here therefore
//! runs the **parallel** pipeline (`exec::execute_parallel`, which is what
//! `Session::run_query` reaches) or the real binary, never an operator in
//! isolation.
//!
//! The budget is not settable from SQL yet -- another change is wiring settings
//! this wave -- so the in-process tests drive `exec::execute_parallel` with a
//! `QueryContext::with_budget`, which is public and is exactly what a `SET
//! max_memory_usage` will hand it. The table and the reference answers come
//! from `Session` itself, and the last test drives the whole stack through
//! `CARGO_BIN_EXE_granular`, so no layer between SQL text and a temp file is
//! assumed rather than exercised.
//!
//! Correctness is asserted **by value**: every group, every aggregate and every
//! sorted row is compared against the same query with room to spare. A test
//! that only checked that the query completed would have passed against a spill
//! that dropped half its partitions.
//!
//! The tests hold one lock and share one table. Both for the same reason: the
//! only way to see a spill that worked is to catch its directory while it
//! exists, spill directories are named after the *process*, and `cargo test`
//! runs a file's tests as threads of one process -- so two tests spilling at
//! once would each see the other's files and prove nothing.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use granular::exec::operators::QueryContext;
use granular::planner::binder::Binder;
use granular::planner::logical::LogicalPlan;
use granular::planner::optimizer;
use granular::types::Value;
use granular::{Block, Session};

const BIN: &str = env!("CARGO_BIN_EXE_granular");

/// Rows enough to make the exchange fan out and to make a tight budget bite.
/// Kept as small as that allows, because this file builds its table from SQL
/// text: 2M rows carrying 1.1M distinct group keys.
const ROWS: u64 = 2_000_000;

/// The budget the `GROUP BY` tests run under. Against ~1.5 GiB of group table
/// it is deep into the spilling region.
///
/// It used to say "the parallel aggregate's floor is around 96 MiB and not
/// lower". That number came from the old core-count-dependent
/// `worker_ceiling`; the floor is now a function of the budget alone
/// (`MIN_WORKER_TABLE`, plus one block's worth of accumulator heap for an
/// aggregate that has any), so it is both lower and hardware-independent.
/// 128 MiB is kept because it is what these tests were calibrated against,
/// not because it is the edge.
const TIGHT_AGG: i64 = 128 << 20;

/// The budget the `ORDER BY` test runs under.
const TIGHT_SORT: i64 = 96 << 20;

// ----------------------------------------------------------------- harness

/// One table, one lock. Built on first use.
fn db() -> MutexGuard<'static, Session> {
    static DB: OnceLock<Mutex<Session>> = OnceLock::new();
    DB.get_or_init(|| {
        let mut s = Session::in_memory();
        s.execute(
            "CREATE TABLE t (id UInt64, big UInt64, k UInt64, v Int64, s String, \
             m Nullable(Int64)) ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
        )
        .unwrap();
        let names = ["ann", "bob", "cyd", "dee", "eve"];
        // One statement per 100k rows: the parser is not what is under test and
        // a single 2M-row literal list is a 100 MB string.
        let mut sql = String::new();
        for i in 0..ROWS {
            if i % 100_000 == 0 {
                if !sql.is_empty() {
                    s.execute(&sql).unwrap();
                }
                sql = String::from("INSERT INTO t VALUES ");
            } else {
                sql.push(',');
            }
            let m = if i % 11 == 0 { "NULL".to_string() } else { (i % 977).to_string() };
            sql.push_str(&format!(
                "({i},{},{},{},'{}',{m})",
                granular::common::splitmix64(i) % 1_500_000,
                i % 64,
                i % 97,
                names[(i % 5) as usize]
            ));
        }
        s.execute(&sql).unwrap();
        s.catalog.flush_all().unwrap();
        Mutex::new(s)
    })
    .lock()
    .unwrap_or_else(|e| e.into_inner())
}

fn plan(s: &mut Session, sql: &str) -> LogicalPlan {
    s.catalog.flush_all().unwrap();
    let stmts = granular::sql::parser::parse(sql).unwrap();
    let q = match &stmts[0] {
        granular::sql::ast::Statement::Query(q) => q.clone(),
        other => panic!("not a query: {other:?}"),
    };
    optimizer::optimize(Binder::new(&s.catalog).bind_query(&q).unwrap()).unwrap()
}

fn rows_of(blocks: &[Block]) -> Vec<Vec<Value>> {
    blocks
        .iter()
        .flat_map(|b| {
            (0..b.rows())
                .map(move |r| (0..b.width()).map(|c| b.column(c).value(r)).collect::<Vec<_>>())
        })
        .collect()
}

/// Spill directories belonging to this process. `SpillDir` names them
/// `granular-spill-<pid>-<n>`.
fn spill_dirs() -> Vec<PathBuf> {
    let prefix = format!("granular-spill-{}-", std::process::id());
    let Ok(rd) = std::fs::read_dir(std::env::temp_dir()) else { return Vec::new() };
    rd.flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(&prefix)))
        .collect()
}

/// Run `f`, watching the spill directory throughout.
///
/// The watcher is the only way to see a spill that worked: the directories
/// unlink themselves on the way out, so afterwards there is by design nothing
/// left to find. Polling during the query answers both halves at once -- "did
/// it really spill" and "did it clean up" -- and it is what stops these tests
/// from passing vacuously on a budget that turned out to be generous.
fn watch<T>(f: impl FnOnce() -> T) -> (T, bool) {
    let stop = Arc::new(AtomicBool::new(false));
    let seen = Arc::new(AtomicBool::new(false));
    let (s2, seen2) = (stop.clone(), seen.clone());
    let h = std::thread::spawn(move || {
        while !s2.load(Ordering::Relaxed) {
            if !spill_dirs().is_empty() {
                seen2.store(true, Ordering::Relaxed);
            }
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
    });
    let out = f();
    stop.store(true, Ordering::Relaxed);
    h.join().unwrap();
    (out, seen.load(Ordering::Relaxed))
}

fn assert_clean(what: &str) {
    let left = spill_dirs();
    assert!(left.is_empty(), "{what}: spill directories outlived the query: {left:?}");
}

/// The same query, in parallel, with room to spare.
fn reference(s: &Session, p: &LogicalPlan) -> Vec<Vec<Value>> {
    let ctx = QueryContext::with_budget(8 << 30);
    let out = rows_of(&granular::exec::execute_parallel(p, &s.catalog, &ctx).unwrap().0);
    assert_eq!(ctx.mem.used(), 0, "the reference kept its reservation");
    out
}

// -------------------------------------------------------------- the tests

#[test]
fn a_parallel_group_by_spills_under_a_small_budget_and_answers_exactly() {
    let mut s = db();
    // 1.1M distinct keys out of 2M rows: enough that the fourteen partial
    // tables cannot all be resident, and enough that the fold has to put a key
    // one worker spilled together with the same key another worker held.
    // The budget is per query because one of them has a floor the others do
    // not. `uniq` allocates a fixed 16 KiB HLL register array *per group*, and
    // since `Accumulator::heap_bytes` exists the budget can finally see it:
    // 1.1M groups is 18 GiB of registers, and the irreducible minimum is one
    // block's worth of new groups -- 8192 x 16 KiB = 128 MiB -- because the
    // freeze check is once per block and a block creates its groups before it.
    // So that query gets room for a few workers' worth of that floor and still
    // spills by three orders of magnitude; the rest keep the original 128 MiB.
    // Before `heap_bytes` this query ran at 128 MiB while actually holding
    // ~1.9 GiB, which is the defect, not the budget.
    for (q, budget) in [
        ("SELECT big, count(*), sum(v), min(v), max(v) FROM t GROUP BY big", TIGHT_AGG),
        ("SELECT big, count(DISTINCT k), sum(DISTINCT k), uniq(s) FROM t GROUP BY big", 4 << 30),
        ("SELECT big, k, count(*) FROM t GROUP BY big, k", TIGHT_AGG),
        ("SELECT big, count(m), quantile(0.9)(m), max(s) FROM t GROUP BY big", TIGHT_AGG),
        ("SELECT big, count(*) FROM t WHERE k < 32 GROUP BY big", TIGHT_AGG),
    ] {
        let p = plan(&mut s, q);
        let mut want = reference(&s, &p);
        want.sort();
        assert!(want.len() > 500_000, "`{q}` produced only {} groups", want.len());

        let tight = QueryContext::with_budget(budget);
        let (mut got, spilled) = watch(|| {
            rows_of(&granular::exec::execute_parallel(&p, &s.catalog, &tight).unwrap().0)
        });
        got.sort();
        assert!(spilled, "nothing spilled for `{q}`, so nothing was tested");
        // Every value, not just the group count: a bucket folded twice would
        // double a `count`, and a bucket folded against the wrong side would
        // lose a `min`.
        assert_eq!(got.len(), want.len(), "`{q}` lost or duplicated groups");
        assert_eq!(got, want, "the spilled run disagrees with the in-memory one on `{q}`");
        assert_eq!(tight.mem.used(), 0, "`{q}` kept its reservation");
        assert_clean(q);
    }
}

#[test]
fn a_spilled_parallel_group_by_is_deterministic_where_it_cannot_be_identical() {
    // The one thing a parallel spill gives up, stated as a test rather than
    // left to be discovered. `any`, `anyLast`, `argMin`'s tie-break and
    // `groupArray`'s element order are defined against feed order, and a group
    // one worker spilled while another held it resident is fed the resident
    // rows first whichever worker they came from. So these are *not* asserted
    // equal to the in-memory answer -- they are asserted stable, which is what
    // the exchange's static split does promise and what a shared work queue
    // would have taken away as well.
    let mut s = db();
    let q = "SELECT big, any(s), anyLast(s), argMin(s, id), groupArray(k) FROM t GROUP BY big";
    let p = plan(&mut s, q);
    let mut want = reference(&s, &p);
    want.sort();

    let mut first: Option<Vec<Vec<Value>>> = None;
    for i in 0..3 {
        let tight = QueryContext::with_budget(TIGHT_AGG);
        let (mut got, spilled) = watch(|| {
            rows_of(&granular::exec::execute_parallel(&p, &s.catalog, &tight).unwrap().0)
        });
        got.sort();
        assert!(spilled, "run {i} of `{q}` did not spill");
        // The groups themselves are exact; only the order-sensitive columns
        // may move, and every one of them still has to be a value from its own
        // group -- which the key comparison below would not catch on its own.
        let keys: Vec<&Value> = got.iter().map(|r| &r[0]).collect();
        let want_keys: Vec<&Value> = want.iter().map(|r| &r[0]).collect();
        assert_eq!(keys, want_keys, "run {i} of `{q}` lost a group");
        match &first {
            None => first = Some(got),
            Some(f) => assert_eq!(&got, f, "run {i} of `{q}` differs from run 0"),
        }
        assert_eq!(tight.mem.used(), 0);
        assert_clean(q);
    }
}

#[test]
fn a_large_order_by_spills_under_a_small_budget_and_answers_exactly() {
    // The budget-driven half of the sort story runs **serially**, and that is a
    // statement about the exchange rather than about the sort. Each of the
    // fourteen workers opens its own `RunMerge` and sizes it from what the
    // budget has left at that instant, so how much the fleet collectively
    // claims depends on how the threads interleave: the same query at the same
    // budget answers on a quiet machine and is refused on a loaded one. That is
    // a real defect and it is one the exchange has to fix -- it is the only
    // party that knows the degree -- so pinning a test to the edge of it would
    // pin the flake, not the behaviour. See `outsideMyFiles`.
    //
    // What is asserted here is the sort's own spilling, exactly, against the
    // parallel in-memory answer. The *parallel* spilling sort is asserted by
    // `the_binary_spills_a_group_by_and_a_sort_and_answers_the_same`, which
    // forces the spill rather than starving it and so does not depend on
    // fourteen threads agreeing about memory.
    let mut s = db();
    for q in [
        // A full sort on a key that is not the table's own order, so the
        // workers' runs interleave and the merge has real work to do.
        "SELECT k, id FROM t ORDER BY big, id",
        // The comparison path: a string key, a second key, mixed directions.
        "SELECT s, big FROM t ORDER BY s, big DESC",
        // A nullable key, so NULL placement has to survive the round trip
        // through a temp file and back out of the merge.
        "SELECT m, id FROM t ORDER BY m NULLS FIRST, id",
    ] {
        let p = plan(&mut s, q);
        let want = reference(&s, &p);
        assert_eq!(want.len(), ROWS as usize, "`{q}` produced {} rows", want.len());

        let tight = QueryContext::with_budget(TIGHT_SORT);
        let (got, spilled) = watch(|| {
            rows_of(&granular::exec::operators::execute_ctx(&p, &s.catalog, &tight).expect(q).0)
        });
        assert!(spilled, "nothing spilled for `{q}`, so nothing was tested");
        assert_eq!(got.len(), want.len(), "`{q}` lost rows");
        // Position by position, not as a multiset: a merge that broke ties by
        // run rather than by input position would pass a multiset check and
        // still reorder every tie.
        assert_eq!(got, want, "the spilled sort disagrees with the in-memory one on `{q}`");
        assert_eq!(tight.mem.used(), 0, "`{q}` kept its reservation");
        assert_clean(q);
    }
}

#[test]
fn a_cancelled_spilling_query_takes_its_temp_files_with_it() {
    // The production incident this pins: a spill that leaks its directory on
    // the error path fills /tmp one abandoned query at a time, and nothing
    // notices until the disk does. Both blocking operators get it, and the
    // cancel fires *once a spill file exists* rather than after a sleep, so it
    // lands in the middle of the spilling and not before it.
    let mut s = db();
    for (q, budget) in [
        ("SELECT big, count(*), sum(v) FROM t GROUP BY big", TIGHT_AGG),
        // Cancelled long before the merge, so the fleet-sizing wobble the sort
        // test above documents cannot reach this one.
        ("SELECT big, id FROM t ORDER BY big, id", TIGHT_SORT),
    ] {
        let p = plan(&mut s, q);
        let ctx = QueryContext::with_budget(budget);
        let cancel = ctx.cancel_handle();
        let stop = Arc::new(AtomicBool::new(false));
        let s2 = stop.clone();
        let armed = std::thread::spawn(move || {
            while !s2.load(Ordering::Relaxed) {
                if !spill_dirs().is_empty() {
                    cancel.store(true, Ordering::Relaxed);
                    return true;
                }
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
            false
        });
        let out = granular::exec::execute_parallel(&p, &s.catalog, &ctx);
        stop.store(true, Ordering::Relaxed);
        assert!(
            armed.join().unwrap(),
            "`{q}` never spilled, so the cancel had nothing to interrupt"
        );
        let e = out.expect_err("a cancelled query must not return an answer").to_string();
        assert!(e.contains("cancelled"), "`{q}`: {e}");
        assert_eq!(ctx.mem.used(), 0, "`{q}` kept its reservation on the error path");
        // The directories die with the operator that owns them, and the
        // operator dies with the `?` that failed.
        assert_clean(q);
    }
}

#[test]
fn the_binary_spills_a_group_by_and_a_sort_and_answers_the_same() {
    // End to end through the shipped binary: SQL text in, rows out, with
    // `GRANULAR_SPILL_ROWS` forcing every group table and every sort buffer to
    // spill regardless of the budget. Nothing between the front door and the
    // temp file is assumed. The knob exists because the budget is not settable
    // from SQL yet; when it is, this test should set that instead.
    let mut sql = String::from(
        "CREATE TABLE t (id UInt64, big UInt64, k UInt64, s String) ENGINE = MergeTree \
         ORDER BY id PRIMARY KEY id;\n",
    );
    // 40k rows: past the exchange's 16384-row floor, so this really is the
    // parallel path, and small enough to be a test.
    for i in 0..40_000u64 {
        sql.push_str(if i % 10_000 == 0 { "INSERT INTO t VALUES " } else { "," });
        sql.push_str(&format!(
            "({i},{},{},'n{}')",
            granular::common::splitmix64(i) % 20_000,
            i % 32,
            i % 5
        ));
        if i % 10_000 == 9_999 {
            sql.push_str(";\n");
        }
    }
    // Order-insensitive aggregates only: `any` and `groupArray` are defined
    // against feed order, and a parallel spill is allowed to change it. See
    // `a_spilled_parallel_group_by_is_deterministic_where_it_cannot_be_identical`.
    sql.push_str(
        "SELECT big, count(*), sum(k), max(s) FROM t GROUP BY big ORDER BY big LIMIT 40;\n\
         SELECT k, count(DISTINCT s), uniq(big), min(id) FROM t GROUP BY k ORDER BY k;\n\
         SELECT s, big, id FROM t ORDER BY s, big, id LIMIT 60;\n\
         SELECT id, big FROM t ORDER BY big DESC, id LIMIT 25;\n",
    );

    let run = |spill: Option<&str>| -> String {
        let mut c = Command::new(BIN);
        match spill {
            Some(n) => c.env("GRANULAR_SPILL_ROWS", n),
            None => c.env_remove("GRANULAR_SPILL_ROWS"),
        };
        let out = c.arg("-q").arg(&sql).output().expect("run granular");
        assert!(
            out.status.success(),
            "exit {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
        // The trailing "N rows in X ms" line is a clock reading, not an answer.
        String::from_utf8(out.stdout)
            .unwrap()
            .lines()
            .filter(|l| !l.contains(" rows in "))
            .map(|l| format!("{l}\n"))
            .collect()
    };

    let plain = run(None);
    assert!(plain.contains('│'), "no result table in:\n{plain}");
    // Snapshotted around the spilling run, not scanned afterwards: this machine
    // may well be carrying spill directories from a build that was killed, and
    // a test that failed on someone else's litter would say nothing about this
    // one. What is asserted is that the child added nothing that outlived it.
    let before = all_spill_dirs();
    assert_eq!(run(Some("500")), plain, "the binary answers differently when its operators spill");
    let leaked: Vec<PathBuf> =
        all_spill_dirs().into_iter().filter(|p| !before.contains(p)).collect();
    assert!(leaked.is_empty(), "the binary left {leaked:?} behind");
}

/// Every process's spill directories, for the before/after diff above.
fn all_spill_dirs() -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(std::env::temp_dir()) else { return Vec::new() };
    rd.flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("granular-spill-"))
        })
        .collect()
}
