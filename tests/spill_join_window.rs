//! End to end: the grace hash join and the partition-at-a-time window.
//!
//! Both features are *inside* operators that `Session` already reaches, which
//! is exactly the shape of defect this phase exists to stop -- code that is
//! complete in `src/` and unreachable from the facade. So the assertions here
//! are made from outside the crate: half through the public [`Session`] API and
//! half through the `granular` binary, and none of them can be satisfied by a
//! unit test on the operator.
//!
//! ## Getting a spilling query out of a facade with no budget setting
//!
//! `Session` fixes every query's budget at `DEFAULT_MEM_BUDGET` (8 GiB) and
//! exposes no way to change it -- so a test that wanted a *spilling* join
//! through `Session` would have to insert 8 GiB of rows. The engine's own
//! answer to that is `GRANULAR_SPILL_ROWS`, the knob the differential harness
//! uses, which makes the blocking operators spill after N buffered rows
//! whatever the budget says. It is read once per process through a `OnceLock`,
//! so a reference run and a spilling run cannot share one; that is why the
//! spilling half of this file drives the CLI with
//! [`std::process::Command`](std::process::Command) rather than `Session`.
//!
//! Two things follow, and both are asserted here rather than assumed:
//!
//! * **the spilling run must engage.** A knob that did nothing would leave the
//!   answers identical and every comparison below would pass vacuously. A grace
//!   join emits partition by partition instead of in probe order, so an
//!   unordered join whose row *order* is unchanged is a run that never spilled
//!   -- that inequality is the engagement proof, and it is asserted;
//! * **the temp files must go.** The child gets its own `TMPDIR`, so what it
//!   left behind is exactly what is in that directory afterwards, with no
//!   guessing about which of the suite's spill directories belong to whom.

use std::path::{Path, PathBuf};
use std::process::Command;

use granular::{Session, Value};

// ------------------------------------------------------------------ harness

const BIN: &str = env!("CARGO_BIN_EXE_granular");

struct Run {
    stdout: String,
    ok: bool,
    stderr: String,
    /// The child's private temp directory, still on disk so it can be
    /// inspected after the process is gone.
    tmp: PathBuf,
}

impl Run {
    /// Every `granular-spill-*` directory the child left behind. Empty is the
    /// only acceptable answer once the process has exited, on success *and* on
    /// failure: a spill directory unlinks itself when the operator that owns it
    /// is dropped, and every exit from a query drops its operators.
    fn leftover_spill_dirs(&self) -> Vec<PathBuf> {
        let Ok(rd) = std::fs::read_dir(&self.tmp) else { return Vec::new() };
        rd.flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("granular-spill-"))
            })
            .collect()
    }

    fn rows(&self) -> Vec<&str> {
        self.stdout.lines().collect()
    }

    fn sorted_rows(&self) -> Vec<&str> {
        let mut r = self.rows();
        r.sort_unstable();
        r
    }
}

/// Run one SQL script through the binary, in a scratch directory of its own.
fn cli(tag: &str, sql: &str, envs: &[(&str, &str)]) -> Run {
    let root = std::env::temp_dir().join(format!(
        "gr-e2e-{}-{tag}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("scratch dir");
    let script = root.join("q.sql");
    std::fs::write(&script, sql).expect("write script");
    // The child's TMPDIR *is* the observation: `SpillDir` builds its path from
    // `env::temp_dir()`, so redirecting it makes "did this query clean up"
    // answerable without racing every other test that spills.
    let tmp = root.join("tmp");
    std::fs::create_dir_all(&tmp).expect("tmp dir");

    let mut cmd = Command::new(BIN);
    cmd.arg("-f")
        .arg(&script)
        .arg("--format")
        .arg("tsv")
        .arg("--no-header")
        .env("TMPDIR", &tmp);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("run granular");
    Run {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        ok: out.status.success(),
        tmp,
    }
}

/// The reference: no knob, so every operator stays in memory.
fn memory(tag: &str, sql: &str) -> Run {
    let r = cli(tag, sql, &[]);
    assert!(r.ok, "the in-memory reference failed: {}", r.stderr);
    assert!(
        r.leftover_spill_dirs().is_empty(),
        "the in-memory reference spilled at all: {:?}",
        r.leftover_spill_dirs()
    );
    r
}

/// The same script with the operators forced onto their spilling paths.
fn spilling(tag: &str, sql: &str, rows: usize) -> Run {
    cli(tag, sql, &[("GRANULAR_SPILL_ROWS", &rows.to_string())])
}

// --------------------------------------------------------------- the tables

/// `l(id, k, v)` and `r(id, k, w)`, joined on `k`.
///
/// The key domains overlap only partly and in both directions, so a `FULL
/// OUTER JOIN` has unmatched rows on each side; keys repeat, so a partition
/// holds real fan-out rather than one row per bucket; and every 23rd left key
/// is NULL, which matches nothing and still has to come back out of an outer
/// join exactly once.
fn join_tables(nl: i64, nr: i64) -> String {
    let mut s = String::with_capacity((nl + nr) as usize * 24);
    s.push_str(
        "CREATE TABLE l (id Int64, k Nullable(Int64), v Int64) ENGINE = MergeTree ORDER BY id;\n\
         CREATE TABLE r (id Int64, k Int64, w Int64) ENGINE = MergeTree ORDER BY id;\n",
    );
    s.push_str("INSERT INTO l VALUES ");
    for i in 0..nl {
        if i > 0 {
            s.push(',');
        }
        if i % 23 == 0 {
            s.push_str(&format!("({i},NULL,{})", i % 7));
        } else {
            s.push_str(&format!("({i},{},{})", (i * 7919) % 5000, i % 7));
        }
    }
    s.push_str(";\nINSERT INTO r VALUES ");
    for i in 0..nr {
        if i > 0 {
            s.push(',');
        }
        // 1000..4999 -- so left keys 0..999 have no match and right keys
        // 4000..4999 have several.
        s.push_str(&format!("({i},{},{})", 1000 + (i * 104_729) % 4000, i % 11));
    }
    s.push_str(";\n");
    s
}

/// `w(id, g, v)`: `g` is the partition key, wide enough that the parallel split
/// has several partitions to hand out and uneven enough that an
/// equal-partitions split would be a bad one.
fn window_table(n: i64) -> String {
    let mut s = String::with_capacity(n as usize * 20);
    s.push_str("CREATE TABLE w (id Int64, g Int64, v Int64) ENGINE = MergeTree ORDER BY id;\n");
    s.push_str("INSERT INTO w VALUES ");
    for i in 0..n {
        if i > 0 {
            s.push(',');
        }
        // Partition sizes 1x, 2x, 3x ... so no split is even.
        let g = ((i % 45) as f64).sqrt() as i64;
        s.push_str(&format!("({i},{g},{})", i % 13));
    }
    s.push_str(";\n");
    s
}

// ================================================================ the joins

#[test]
fn a_join_whose_build_side_far_exceeds_the_budget_still_answers_correctly() {
    // The headline claim: the same rows, from a run that could not hold either
    // side. Compared against a reference run of the *same script* rather than
    // against hand-computed numbers, so the two differ in exactly one thing.
    let q = "SELECT count(*), sum(l.v), sum(r.w), min(l.id), max(r.id) \
             FROM l JOIN r ON l.k = r.k;\n";
    let sql = join_tables(40_000, 40_000) + q;
    let want = memory("join-mem", &sql);
    let got = spilling("join-spill", &sql, 2_000);
    assert!(got.ok, "the spilling join failed: {}", got.stderr);
    assert_eq!(got.rows(), want.rows(), "a spilled join answered differently");
    assert!(
        got.leftover_spill_dirs().is_empty(),
        "a successful spilling join left {:?}",
        got.leftover_spill_dirs()
    );
    assert!(!want.stdout.trim().is_empty(), "the join produced nothing to compare");
}

#[test]
fn the_spilling_join_really_is_a_different_plan_and_not_a_knob_that_does_nothing() {
    // Engagement, not correctness. A grace join hands its rows out partition by
    // partition; the in-memory one hands them out in probe order. So an
    // unordered projection has to come back as the same *multiset* in a
    // different *sequence* -- and if the sequence matched, the spilling run
    // never ran and every other assertion in this file is vacuous.
    let q = "SELECT l.id, r.id FROM l JOIN r ON l.k = r.k;\n";
    let sql = join_tables(20_000, 20_000) + q;
    let want = memory("join-order-mem", &sql);
    let got = spilling("join-order-spill", &sql, 2_000);
    assert!(got.ok, "{}", got.stderr);
    assert_eq!(got.sorted_rows(), want.sorted_rows(), "the multiset changed");
    assert_ne!(
        got.rows(),
        want.rows(),
        "the row order is identical, so the join never took the grace path"
    );
}

#[test]
fn an_outer_join_across_partition_boundaries_pads_every_row_exactly_once() {
    // The case grace joins break. Partitioning splits the unmatched rows across
    // buckets, and the two ways to get it wrong are visible in these three
    // counts: padding a row once per partition inflates `count(*)`, and losing
    // the padding entirely deflates it. Every join type at once, because LEFT,
    // RIGHT and FULL exercise a different pair of padding phases.
    let counts = "SELECT count(*), count(l.id), count(r.id), sum(l.v), sum(r.w) FROM l";
    let mut q = String::new();
    for op in ["LEFT OUTER JOIN", "RIGHT OUTER JOIN", "FULL OUTER JOIN", "INNER JOIN"] {
        q.push_str(&format!("{counts} {op} r ON l.k = r.k;\n"));
    }
    // ... and one whole-row comparison, ordered, so a wrong *value* rather than
    // a wrong count cannot hide behind the aggregate.
    q.push_str(
        "SELECT l.id, l.k, r.id, r.w FROM l FULL OUTER JOIN r ON l.k = r.k \
         ORDER BY l.id, r.id, r.w;\n",
    );
    let sql = join_tables(20_000, 20_000) + &q;
    let want = memory("outer-mem", &sql);
    let got = spilling("outer-spill", &sql, 1_500);
    assert!(got.ok, "the spilling outer join failed: {}", got.stderr);
    assert_eq!(got.rows(), want.rows(), "a spilled outer join padded differently");
    assert!(got.leftover_spill_dirs().is_empty());
}

#[test]
fn a_spilling_join_that_fails_takes_its_temp_files_with_it() {
    // The `?` path. `CAST('abc' AS Int64)` is evaluated per output block, so
    // the query dies with partitions still pending and the operator dropped by
    // unwinding the pipeline rather than by finishing it.
    let sql = join_tables(20_000, 20_000)
        + "SELECT CAST('abc' AS Int64) FROM l JOIN r ON l.k = r.k;\n";
    let got = spilling("join-err", &sql, 1_500);
    assert!(!got.ok, "the failing query succeeded: {}", got.stdout);
    assert!(got.stderr.contains("cannot cast"), "wrong failure: {}", got.stderr);
    assert!(
        got.leftover_spill_dirs().is_empty(),
        "a failed spilling join left {:?}",
        got.leftover_spill_dirs()
    );
}

#[test]
fn a_spilling_join_abandoned_by_a_limit_takes_its_temp_files_with_it() {
    // The other early exit: nothing failed, the consumer simply stopped asking.
    // The operator is dropped with most of its partitions unread.
    let sql = join_tables(20_000, 20_000)
        + "SELECT l.id FROM l JOIN r ON l.k = r.k LIMIT 3;\n";
    let got = spilling("join-limit", &sql, 1_500);
    assert!(got.ok, "{}", got.stderr);
    assert_eq!(got.rows().len(), 3);
    assert!(
        got.leftover_spill_dirs().is_empty(),
        "an abandoned spilling join left {:?}",
        got.leftover_spill_dirs()
    );
}

#[test]
fn the_in_memory_join_still_answers_through_session() {
    // The other half of the reachability claim: the fast path is what every
    // real query takes, and the spill machinery must not have moved it. Driven
    // through `Session` because that is the API a caller has.
    let mut s = Session::in_memory();
    for stmt in join_tables(5_000, 5_000).split(";\n") {
        if !stmt.trim().is_empty() {
            s.execute(stmt).unwrap_or_else(|e| panic!("{stmt:.60}: {e}"));
        }
    }
    let inner = s
        .query("SELECT count(*) FROM l JOIN r ON l.k = r.k")
        .unwrap()
        .scalar()
        .unwrap();
    let left = s
        .query("SELECT count(*), count(r.id) FROM l LEFT OUTER JOIN r ON l.k = r.k")
        .unwrap()
        .to_values();
    // A LEFT JOIN emits every left row at least once, so its total is the
    // matched rows plus one row per unmatched left row -- and the matched
    // half is exactly the inner join.
    assert_eq!(left[0][1], inner, "the matched half of a LEFT JOIN is the inner join");
    let unmatched = s
        .query(
            "SELECT count(*) FROM l WHERE l.k IS NULL \
             OR l.k NOT IN (SELECT k FROM r)",
        )
        .unwrap()
        .scalar()
        .unwrap();
    let (total, matched, un) = (
        left[0][0].as_u64().unwrap(),
        inner.as_u64().unwrap(),
        unmatched.as_u64().unwrap(),
    );
    assert_eq!(total, matched + un, "{total} != {matched} + {un}");
}

// =============================================================== the windows

#[test]
fn a_window_over_a_large_input_is_computed_partition_at_a_time() {
    // The relation no longer has to fit; a partition does. Forced to cut every
    // 2000 rows -- far below one partition-worth of the input -- the answer has
    // to be byte-identical, *including order*, because a window's output order
    // is its input order no matter how the operator chops it up.
    let q = "SELECT id, g, row_number() OVER (PARTITION BY g ORDER BY id), \
             sum(v) OVER (PARTITION BY g ORDER BY id), \
             lag(v, 2, -1) OVER (PARTITION BY g ORDER BY id), \
             last_value(v) OVER (PARTITION BY g ORDER BY id \
                                 ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) \
             FROM w ORDER BY id;\n";
    let sql = window_table(60_000) + q;
    let want = memory("win-mem", &sql);
    let got = spilling("win-spill", &sql, 2_000);
    assert!(got.ok, "the streaming window failed: {}", got.stderr);
    assert_eq!(got.rows().len(), 60_000, "rows went missing");
    assert_eq!(got.rows(), want.rows(), "a streamed window answered differently");
    assert!(
        got.leftover_spill_dirs().is_empty(),
        "the window left {:?}",
        got.leftover_spill_dirs()
    );
}

#[test]
fn a_parallel_window_answers_exactly_what_a_serial_one_does() {
    // Partitions are independent and each is still folded by one accumulator in
    // one order, so the fan-out is required to be *bit*-identical -- floats
    // included, which is the assertion an exchange-style split could not make.
    let q = "SELECT id, sum(v) OVER (PARTITION BY g ORDER BY id), \
             avg(toFloat64(v)) OVER (PARTITION BY g ORDER BY id), \
             rank() OVER (PARTITION BY g ORDER BY v), \
             ntile(7) OVER (PARTITION BY g ORDER BY id) \
             FROM w ORDER BY id;\n";
    let sql = window_table(60_000) + q;
    let serial = cli("win-1t", &sql, &[("GRANULAR_THREADS", "1")]);
    assert!(serial.ok, "{}", serial.stderr);
    let parallel = cli("win-nt", &sql, &[]);
    assert!(parallel.ok, "{}", parallel.stderr);
    assert_eq!(parallel.rows().len(), 60_000);
    assert_eq!(parallel.rows(), serial.rows(), "the fan-out changed an answer");

    // ... and with the streaming cut on top of it, which is the combination
    // that actually ships: several chunks, each fanned out independently.
    let both = cli(
        "win-nt-spill",
        &sql,
        &[("GRANULAR_SPILL_ROWS", "3000")],
    );
    assert!(both.ok, "{}", both.stderr);
    assert_eq!(both.rows(), serial.rows(), "streaming plus fan-out changed an answer");
}

#[test]
fn a_window_with_no_partition_by_is_one_partition_and_still_correct() {
    // The shape the fan-out cannot split and the streaming cut cannot cut: one
    // partition covering the whole relation. It has to answer, and it has to
    // answer the same way whatever the knobs say -- the operator falls back to
    // buffering the lot, which is the documented plan.
    let q = "SELECT id, sum(v) OVER (ORDER BY id), count(*) OVER () FROM w ORDER BY id;\n";
    let sql = window_table(30_000) + q;
    let want = memory("win-nopart-mem", &sql);
    let got = spilling("win-nopart-spill", &sql, 2_000);
    assert!(got.ok, "{}", got.stderr);
    assert_eq!(got.rows(), want.rows());
    let serial = cli("win-nopart-1t", &sql, &[("GRANULAR_THREADS", "1")]);
    assert_eq!(got.rows(), serial.rows());
}

#[test]
fn a_window_computes_the_closed_form_through_session() {
    // Reachability plus arithmetic, from outside the crate and against an
    // oracle the operator cannot influence: `row_number` inside a partition of
    // known width, and a running total of a known series.
    const N: i64 = 40_000;
    const P: i64 = 500;
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE t (id Int64, g Int64, v Int64) ENGINE = MergeTree ORDER BY id")
        .unwrap();
    let mut sql = String::from("INSERT INTO t VALUES ");
    for i in 0..N {
        if i > 0 {
            sql.push(',');
        }
        sql.push_str(&format!("({i},{},{i})", i / P));
    }
    s.execute(&sql).unwrap();

    let rows = s
        .query(
            "SELECT row_number() OVER (PARTITION BY g ORDER BY id), \
             sum(v) OVER (PARTITION BY g ORDER BY id) FROM t ORDER BY id",
        )
        .unwrap()
        .to_values();
    assert_eq!(rows.len() as i64, N);
    for (i, row) in rows.iter().enumerate() {
        let i = i as i64;
        assert_eq!(row[0], Value::UInt((i % P + 1) as u64), "row_number at {i}");
        // Running total of `id` from the partition's first row to this one.
        let first = i - i % P;
        let want = (first + i) * (i - first + 1) / 2;
        assert_eq!(row[1].as_i64().unwrap(), want, "running sum at {i}");
    }

    // A relation far below the parallel floor takes the serial path and must
    // give the same shape of answer -- the threshold is an optimization, never
    // a second implementation.
    let small = s
        .query(
            "SELECT row_number() OVER (PARTITION BY g ORDER BY id) FROM t \
             WHERE id < 100 ORDER BY id",
        )
        .unwrap()
        .to_values();
    assert_eq!(small.len(), 100);
    assert_eq!(small[99][0], Value::UInt(100), "one partition of 100 rows");
}

// -------------------------------------------------------------------- misc

#[test]
fn the_binary_under_test_is_the_one_that_was_built() {
    // Cheap, and it turns "every CLI test failed mysteriously" into one clear
    // failure when the harness is pointed at nothing.
    assert!(Path::new(BIN).exists(), "{BIN} does not exist");
}
