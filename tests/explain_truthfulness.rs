//! Does `EXPLAIN` describe what actually runs, and does it cost nothing?
//!
//! Two defects, both of them the same defect at different layers.
//!
//!   * **The plan did not mention parallelism.** The exchange was not a
//!     `PhysicalPlan` variant at all -- it was decided inside
//!     `exchange::try_build` while operators were being constructed -- so
//!     `EXPLAIN PIPELINE` for a 400k-row `GROUP BY` was byte-identical to the
//!     serial plan. Nobody outside the executor could tell whether a query went
//!     parallel: not a user, not a benchmark, and not a regression test. A
//!     silent loss of the 5-9x the exchange is worth would have shipped.
//!   * **`EXPLAIN ANALYZE` did not exist.** `Session::run_query` computes
//!     `QueryStats { granules_read, granules_pruned, rows_scanned }` on every
//!     query and throws all three away; the only renderer prints a row count
//!     and a millisecond total.
//!
//! Everything here drives the *public* surface -- `Session::query` and the
//! `granular` binary -- because the pattern this phase exists to break is a
//! capability that is complete in `src/` and unreachable from outside it. A
//! unit test on the planner would have passed against the broken build: the
//! decision was already correct, it was just invisible.

use std::process::Command;
use std::time::{Duration, Instant};

use granular::types::{Block, Column, DataType};
use granular::Session;

const BIN: &str = env!("CARGO_BIN_EXE_granular");

/// Rows enough to clear `exchange::MIN_PARALLEL_ROWS` (16 384) with room to
/// spare on the granule side too: 400k rows is 391 granules, which is more
/// than four per worker on any machine this runs on.
const BIG: u64 = 400_000;

/// One `EXPLAIN`'s text. The result is a one-column relation of lines, which
/// is what a shell sees, so joining them back is what a shell would read.
fn explain(s: &mut Session, sql: &str) -> String {
    let rs = s.query(sql).unwrap_or_else(|e| panic!("`{sql}`: {e}"));
    rs.to_values()
        .iter()
        .map(|r| r[0].render_plain())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The physical plan for a query, which is the rendering `EXPLAIN PIPELINE`
/// exists for -- plain `EXPLAIN` shows the logical tree, where neither the
/// access path nor the parallelism exists yet.
fn pipeline(s: &mut Session, q: &str) -> String {
    explain(s, &format!("EXPLAIN PIPELINE {q}"))
}

/// `t(id UInt64, k UInt64, v Int64)` with `n` rows.
///
/// Built as a `Block` and handed to the table rather than as an `INSERT ...
/// VALUES` string: 400k rows of SQL text is 8 MB to lex and parse, and this
/// file is measuring `EXPLAIN`, not the parser.
fn session(n: u64) -> Session {
    let mut s = Session::in_memory();
    s.execute(
        "CREATE TABLE t (id UInt64, k UInt64, v Int64) \
         ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
    )
    .unwrap();
    let t = s.catalog.table_by_path_mut("default.t").unwrap();
    t.insert(
        Block::new(vec![
            Column::u64s(DataType::UInt64, (0..n).collect()),
            Column::u64s(DataType::UInt64, (0..n).map(|i| i % 8).collect()),
            Column::i64s(DataType::Int64, (0..n).map(|i| i as i64 % 500 - 250).collect()),
        ])
        .unwrap(),
    )
    .unwrap();
    t.flush().unwrap();
    s
}

/// The `workers` figure out of an `Exchange <n> workers` line.
fn workers(plan: &str) -> usize {
    let line = plan
        .lines()
        .find(|l| l.trim_start().starts_with("Exchange "))
        .unwrap_or_else(|| panic!("no Exchange line in:\n{plan}"));
    line.split_whitespace()
        .nth(1)
        .and_then(|w| w.parse().ok())
        .unwrap_or_else(|| panic!("unparsable Exchange line: {line}"))
}

/// Best of `n` -- this machine swings 30% on identical code, and every timing
/// assertion below is an order-of-magnitude claim that the minimum makes
/// robust and the mean does not.
fn best_of(n: usize, mut f: impl FnMut()) -> Duration {
    (0..n)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed()
        })
        .min()
        .unwrap()
}

// ------------------------------------------------- 1. the plan names the fleet

#[test]
fn explain_pipeline_names_the_exchange_and_its_width() {
    let mut s = session(BIG);
    let p = pipeline(&mut s, "SELECT k, count(*) FROM t GROUP BY k");
    // The bug, stated: the plan for a query that goes parallel used to be
    // character-for-character the plan for one that does not.
    assert!(p.contains("Exchange"), "no exchange in the plan:\n{p}");
    let n = workers(&p);
    assert!(n >= 2, "an exchange must be at least 2 wide, got {n}:\n{p}");
    assert!(
        n <= std::thread::available_parallelism().map_or(64, |p| p.get()),
        "more workers than cores: {n}"
    );
    // ... and it sits above the aggregate it replicates, not somewhere else.
    let lines: Vec<&str> = p.lines().collect();
    let xi = lines.iter().position(|l| l.trim_start().starts_with("Exchange")).unwrap();
    assert!(
        lines[xi + 1].trim_start().starts_with("Aggregate"),
        "the exchange must be over the aggregate:\n{p}"
    );
}

#[test]
fn every_parallel_shape_says_so_and_no_serial_shape_does() {
    let mut s = session(BIG);
    // The shapes `exchange::analyze` admits: an aggregate or a sort over a
    // chain of filters and projections.
    for q in [
        // `count(*)` alone is no longer here: `physical::meta_path` answers a
        // bare count from part metadata, so there is nothing left to spread
        // and the honest plan has no `Exchange` in it. Paired with a `sum` it
        // is a parallel aggregate again, which is what this line is testing.
        "SELECT count(*), sum(v) FROM t",
        "SELECT sum(v) FROM t WHERE k = 3",
        "SELECT k, count(*), sum(v) FROM t GROUP BY k",
        "SELECT id FROM t ORDER BY v DESC LIMIT 5",
        "SELECT id, v FROM t WHERE v > 0 ORDER BY v",
    ] {
        let p = pipeline(&mut s, q);
        assert!(p.contains("Exchange"), "`{q}` runs parallel but the plan hides it:\n{p}");
    }
    // ... and the ones it declines. A plan that claims a fleet it does not get
    // is the same lie in the other direction.
    for q in [
        // a point lookup answers in microseconds; nothing to spread
        "SELECT v FROM t WHERE id = 5",
        // streaming operators whose output order is first-seen
        "SELECT DISTINCT k FROM t",
        "SELECT id, k FROM t LIMIT 2 BY k",
        // no blocking top at all
        "SELECT id FROM t",
        "SELECT id FROM t WHERE k = 1 LIMIT 10",
    ] {
        let p = pipeline(&mut s, q);
        assert!(!p.contains("Exchange"), "`{q}` is serial but the plan claims a fleet:\n{p}");
    }
}

#[test]
fn a_small_table_gets_no_exchange_line() {
    // The threshold has to be visible from outside, or "we stay serial below
    // 16k rows" is an unverifiable claim in a doc comment.
    let mut s = session(4_000);
    for q in [
        "SELECT count(*) FROM t",
        "SELECT k, count(*) FROM t GROUP BY k",
        "SELECT id FROM t ORDER BY v LIMIT 5",
    ] {
        let p = pipeline(&mut s, q);
        assert!(!p.contains("Exchange"), "4k rows must stay serial:\n{p}");
    }
}

#[test]
fn the_threshold_is_where_it_is_documented_to_be() {
    // Measured, not guessed: bisect the row count at which the plan flips, and
    // pin it. `exchange::MIN_PARALLEL_ROWS` is 16 << 10, and a change to it
    // that nobody meant should fail here rather than in a benchmark six months
    // later. (The granule floor can only push the flip *up*, never down: 16384
    // rows is 16 granules, which is 4 workers at MIN_GRANULES_PER_WORKER.)
    let (mut lo, mut hi) = (1_024u64, 65_536u64);
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        let mut s = session(mid);
        // Not a bare `count(*)`: that is answered from part metadata at every
        // size, so it is serial on both sides of the threshold and the bisect
        // below would never converge on one.
        if pipeline(&mut s, "SELECT sum(v) FROM t").contains("Exchange") {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    assert_eq!(hi, 16 << 10, "the parallel threshold moved to {hi}");
}

#[test]
fn a_top_k_under_a_limit_keeps_its_bound_when_it_goes_parallel() {
    // The regression this change could most easily have introduced: `lower`
    // wraps the sort in an `Exchange` *before* the `Limit` above it is lowered,
    // so the top-K fusion has to see through the new node. If it does not, the
    // plan silently degrades from `TopK 5` to a full sort of 400k rows -- a
    // correct answer, several times slower, and invisible without this.
    let mut s = session(BIG);
    let p = pipeline(&mut s, "SELECT id, v FROM t ORDER BY v DESC LIMIT 5");
    assert!(p.contains("TopK 5"), "the limit must still fuse through the exchange:\n{p}");
    assert!(p.contains("Exchange"), "{p}");
}

// -------------------------------------------------------- 2. EXPLAIN ANALYZE

#[test]
fn explain_analyze_reports_per_operator_rows_and_time() {
    let mut s = session(BIG);
    let a = explain(&mut s, "EXPLAIN ANALYZE SELECT k, count(*) FROM t GROUP BY k");

    // Every measured line carries all three, and the counters the session used
    // to compute and discard are on the line that earned them.
    let measured: Vec<&str> = a.lines().filter(|l| l.contains("rows=")).collect();
    assert!(!measured.is_empty(), "nothing was measured:\n{a}");
    for l in &measured {
        assert!(l.contains("blocks="), "{l}");
        assert!(l.contains("time="), "{l}");
    }
    assert!(a.contains("Total "), "no wall-clock total:\n{a}");

    // Non-zero rows, which is the assertion that fails if `EXPLAIN ANALYZE`
    // renders a plan without running it.
    let root = measured[0];
    let n: u64 = field(root, "rows=").parse().unwrap();
    assert_eq!(n, 8, "8 groups, got {n} on `{root}`");

    // ... and the access-path counters reached the surface.
    assert!(a.contains("decoded="), "the scan counters are still discarded:\n{a}");
    let decoded: u64 = field(&a, "decoded=").parse().unwrap();
    assert_eq!(decoded, BIG, "every row should have been decoded once");
    // `granules=391r/0p` -- read before the slash, pruned after.
    let g = field(&a, "granules=");
    let read: u64 = g.split('r').next().unwrap().parse().unwrap();
    assert!(read > 100, "400k rows is ~391 granules, reported {read}:\n{a}");
}

#[test]
fn explain_analyze_measures_the_plan_it_prints() {
    // The whole value of ANALYZE is that the tree and the numbers describe one
    // execution. A parallel query must therefore report its rows on the
    // `Exchange` line -- the nodes underneath run inside the workers and are
    // honestly left unmeasured rather than given invented figures.
    let mut s = session(BIG);
    let a = explain(&mut s, "EXPLAIN ANALYZE SELECT k, count(*) FROM t GROUP BY k");
    let lines: Vec<&str> = a.lines().collect();
    let xi = lines.iter().position(|l| l.trim_start().starts_with("Exchange")).unwrap();
    assert!(lines[xi].contains("rows=8"), "{}", lines[xi]);
    assert!(
        !lines[xi + 1].contains("rows="),
        "a node inside the fleet must not claim its own measurement: {}",
        lines[xi + 1]
    );

    // A serial plan measures every level instead, which is the point of doing
    // it per operator at all.
    let mut small = session(4_000);
    let a = explain(&mut small, "EXPLAIN ANALYZE SELECT k, count(*) FROM t GROUP BY k");
    let measured = a.lines().filter(|l| l.contains("rows=")).count();
    assert!(measured >= 2, "a serial plan should measure each operator:\n{a}");
    assert!(a.contains("Scan default.t"), "{a}");
}

#[test]
fn explain_analyze_and_the_query_agree_about_the_answer() {
    let mut s = session(BIG);
    for (q, want) in [
        ("SELECT count(*) FROM t", 1u64),
        ("SELECT k, count(*) FROM t GROUP BY k", 8),
        ("SELECT id FROM t ORDER BY v DESC LIMIT 5", 5),
        ("SELECT id FROM t WHERE id = 7", 1),
    ] {
        let rows = s.query(q).unwrap().rows() as u64;
        assert_eq!(rows, want, "`{q}`");
        let a = explain(&mut s, &format!("EXPLAIN ANALYZE {q}"));
        let got: u64 = field(a.lines().find(|l| l.contains("rows=")).unwrap(), "rows=")
            .parse()
            .unwrap();
        assert_eq!(got, want, "ANALYZE disagrees with the query on `{q}`:\n{a}");
    }
}

// ------------------------------------------------------ 3. EXPLAIN is cheap

#[test]
fn explain_describes_the_query_instead_of_running_it() {
    let mut s = session(BIG);
    // Two shapes whose execution is milliseconds and whose planning is
    // microseconds. If EXPLAIN ran the query the ratio would be ~1x by
    // definition, so any factor comfortably above 1 settles it; 8x is what a
    // 30%-swing machine at load 40 can still be held to, and the measured
    // margin cold is 18x on the aggregate and 57x on the sort.
    for q in ["SELECT k, count(*) FROM t GROUP BY k", "SELECT id FROM t ORDER BY v DESC"] {
        let run = best_of(9, || {
            s.query(q).unwrap();
        });
        let described = best_of(9, || {
            s.query(&format!("EXPLAIN PIPELINE {q}")).unwrap();
        });
        assert!(
            described * 8 < run,
            "EXPLAIN of `{q}` cost {described:?} against {run:?} to run it -- \
             it is executing what it was asked to describe"
        );
    }
}

#[test]
fn explain_analyze_is_the_only_kind_that_runs_the_query() {
    let mut s = session(BIG);
    let q = "SELECT k, count(*) FROM t GROUP BY k";
    let run = best_of(9, || {
        s.query(q).unwrap();
    });
    let analyzed = best_of(9, || {
        s.query(&format!("EXPLAIN ANALYZE {q}")).unwrap();
    });
    // Within a factor of four of the query itself: ANALYZE builds the same
    // pipeline and pays one `Instant::now` pair per block on top. Anything
    // wildly cheaper would mean it had not really run.
    assert!(analyzed * 4 > run, "ANALYZE {analyzed:?} vs query {run:?} -- it did not run");
    assert!(analyzed < run * 4, "ANALYZE {analyzed:?} vs query {run:?} -- probes cost too much");
}

/// The one part of the contract this change could not close, pinned as a
/// number so it cannot quietly get worse and so the fix has a tripwire.
///
/// `EXPLAIN PIPELINE SELECT count() FROM t WHERE id IN (SELECT id FROM t)`
/// **executes the inner query**, because `Session::plan` folds every
/// uncorrelated subquery into a literal list before binding and `run_explain`
/// calls `Session::plan`. That fold is in `session.rs`, which this change does
/// not own; the accompanying report names the exact lines and the design that
/// replaces it (bind the subquery to a symbolic node and render its plan
/// nested, rather than running it for its values).
///
/// Asserted in the direction that is true *today* so the suite stays green.
/// **When the fold is fixed, invert this**: the assertion becomes
/// `described * 4 < run`, and the name loses its `still_`.
#[test]
fn explain_still_executes_an_uncorrelated_subquery() {
    let mut s = session(BIG);
    let inner = "SELECT id FROM t WHERE k = 3";
    let outer = format!("SELECT count(*) FROM t WHERE id IN ({inner})");
    let run = best_of(3, || {
        s.query(inner).unwrap();
    });
    let described = best_of(3, || {
        s.query(&format!("EXPLAIN PIPELINE {outer}")).unwrap();
    });
    assert!(
        described * 4 > run,
        "EXPLAIN no longer runs the subquery ({described:?} vs {run:?}) -- \
         invert this test, the defect it pins is fixed"
    );
}

// ----------------------------------------------------------- 4. from a shell

#[test]
fn the_binary_shows_the_exchange_and_the_measurements() {
    // The layer the previous seven failures were invisible at: everything can
    // be right in `src/` and still not reach a user typing at a prompt.
    let dir = std::env::temp_dir().join(format!("granular-explain-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // A quarter of `BIG` is still 48 granules, i.e. 12 workers, and it keeps
    // the `VALUES` text this test has to lex under a megabyte.
    let mut values = String::with_capacity(1 << 20);
    for i in 0..(BIG / 4) {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i},{})", i % 8));
    }
    let script = format!(
        "CREATE TABLE t (id UInt64, k UInt64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id;\n\
         INSERT INTO t VALUES {values};\n\
         SYSTEM FLUSH;\n\
         EXPLAIN PIPELINE SELECT k, count(*) FROM t GROUP BY k;\n\
         EXPLAIN ANALYZE SELECT k, count(*) FROM t GROUP BY k;\n"
    );
    let f = dir.join("script.sql");
    std::fs::write(&f, script).unwrap();

    let out = Command::new(BIN)
        .arg("--data")
        .arg(&dir)
        .arg("-f")
        .arg(&f)
        .output()
        .expect("run granular");
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(out.status.success(), "exit {:?}\n{text}", out.status.code());
    assert!(text.contains("Exchange"), "the shell cannot see the fleet:\n{text}");
    assert!(text.contains("rows="), "the shell cannot see the measurements:\n{text}");
    assert!(text.contains("decoded="), "{text}");

    let _ = std::fs::remove_dir_all(&dir);
}

// ----------------------------------------------------------------- 5. answers

#[test]
fn making_the_exchange_a_plan_node_did_not_change_any_answer() {
    // The plan grew a level; the rows must not have moved. Cheap end-to-end
    // insurance against the one thing a planner change is most likely to break.
    let mut s = session(BIG);
    assert_eq!(s.query("SELECT count(*) FROM t").unwrap().scalar().unwrap().as_u64(), Some(BIG));
    let g = s.query("SELECT k, count(*) FROM t GROUP BY k").unwrap().to_values();
    assert_eq!(g.len(), 8);
    for (i, row) in g.iter().enumerate() {
        assert_eq!(row[0].as_u64(), Some(i as u64), "group order must stay first-seen");
        assert_eq!(row[1].as_u64(), Some(BIG / 8));
    }
    let top = s.query("SELECT id FROM t ORDER BY id DESC LIMIT 3").unwrap().to_values();
    assert_eq!(
        top.iter().map(|r| r[0].as_u64().unwrap()).collect::<Vec<_>>(),
        vec![BIG - 1, BIG - 2, BIG - 3]
    );
}

/// The text after `key`, up to the next space.
fn field<'a>(s: &'a str, key: &str) -> &'a str {
    let rest = &s[s.find(key).unwrap_or_else(|| panic!("no `{key}` in:\n{s}")) + key.len()..];
    rest.split_whitespace().next().unwrap_or("")
}
