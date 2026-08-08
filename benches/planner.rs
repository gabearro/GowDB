//! Planner quality benchmark: what the engine *chose* against the best plan
//! available for the same answer.
//!
//! `benches/engine.rs` measures how fast the machinery runs. Nothing measured
//! whether the planner *reaches* it, which is how three separate gaps -- no
//! predicate pushdown through a join, no index-nested-loop join, no sort
//! elimination -- survived four waves of optimization work. Every one of them
//! is a query where a fast path exists and the planner walks past it, so every
//! one of them is invisible to a benchmark that only times the plan the
//! planner picked.
//!
//! The unit here is a **ratio**, not a time. Each shape is a pair of queries
//! that must return the identical answer:
//!
//!   * `chosen` -- the natural way to write it, which is what the planner gets
//!     to optimize; and
//!   * `best` -- the same answer hand-written into the plan the optimizer
//!     should have produced.
//!
//! `chosen / best` is the planner's quality score for that shape. 1.0 means
//! the rewrite is already in the optimizer. 10x means a user who knows the
//! trick beats the planner by 10x, and the shape is a bug with a number
//! attached. Machine noise is 30%+ here, so the thresholds are set to catch a
//! regression of an order of magnitude, never a percentage.
//!
//! **Below 1.0 the shape has gone stale**, and the run says so in its own
//! output rather than printing a green line: the optimizer has found something
//! the hand-written half does not do, so the pair is now understating the win
//! and its `best` should be rewritten around whatever the planner found. That
//! is not hypothetical -- `filter above join` reads 0.0051x against the
//! in-flight index-nested-loop join, because the natural spelling lowers to
//! `InnerHashJoin < IndexLookup x2` while the hand-pushed subquery this file
//! calls "best" does not reach the index at all.
//!
//! Both halves of every pair are executed and fingerprinted *before* either is
//! timed, and both are checked against an answer computed here in Rust from
//! the fixture's own arithmetic. A ratio between two different answers is not
//! a measurement, and a fast plan for an empty result is not a fast plan, so
//! an answer that does not match fails the run rather than printing a number.
//!
//! ## The baseline, 2026-08-08 at 336ca5c
//!
//! 300k x 300k, best of three full runs on a 14-core machine. Every number is
//! also stored per shape in `Shape::measured` and printed next to the live
//! one, so a wave that closes a gap shows the fall in its own output.
//!
//! ```text
//!   pk lookup through a join            1843x    the two join gaps compounded
//!   filter above distinct                829x
//!   filter above group by                698x
//!   pk lookup through a union            109x
//!   filter above union                    77x
//!   order by sort key + limit             44x
//!   range written as OR of ranges         15x
//!   order by sort key, descending         14x
//!   range written as NOT(...)             10x
//!   order by sort key, aggregated        6.8x
//!   filter above join                    4.2x    the shape the audit reported
//!   order by sort key                    3.9x
//!   IN (SELECT) vs semi-join             3.4x
//!   count() with always-true filter      2.8x    7 us absolute; not a gap
//!   pk lookup through a subquery         1.6x    the rule that already works
//!   3-join, worst order                  1.5x    no reordering exists
//!   wide dictionary vs narrow            1.5x
//!   String vs LowCardinality(String)     1.0x
//!   order by non-sort-key (control)      1.0x    identical both sides
//!   filter on an aggregate (control)     1.0x
//! ```
//!
//! ## What each of this wave's fixes should move
//!
//! Three of these gaps are being closed concurrently, so the handoff is which
//! lines each one owns. The same mapping is printed at the end of every run,
//! derived from `Shape::gap`, so it cannot drift from the shapes themselves.
//!
//!   * **`sink_filter` arms for Join / Union / Aggregate / Distinct** owns the
//!     seven `pushdown` lines, and the four largest ratios in the table are
//!     among them.
//!   * **an index-nested-loop join** owns `pk lookup through a join` -- 1843x,
//!     the worst line here, because it is the pushdown gap and the join gap
//!     compounded. Note `filter above join` (4.2x) is the *pushdown* half
//!     alone, so the two lines together separate the two fixes.
//!   * **sort elimination** owns the four `sort-elim` lines. The `descending`
//!     one is deliberately included: a rule that matches `ORDER BY <sort key>`
//!     without checking direction returns the table backwards, and the pin on
//!     that shape is what catches it.
//!
//! `IN (SELECT)` and join reordering have no owner this wave. The two
//! `(control)` lines and the four other `closed` lines must all still read
//! ~1.0x afterwards; that is what says a new rule fired only where it should.
//!
//! ## The ratios are not constants: they grow with the table
//!
//! The same run at `ROWS=1000000`, which is why closing these matters more at
//! scale than the 300k column suggests -- the cost of a plan that prunes
//! nothing is quadratic in what it failed to prune:
//!
//! ```text
//!   filter above distinct         829x -> 1917x      filter above union    77x -> 225x
//!   pk lookup through a join     1843x -> 2131x      filter above join    4.2x -> 5.1x
//!   filter above group by         698x ->  978x      IN (SELECT)          3.4x -> 6.8x
//!   pk lookup through a union     109x ->  352x      3-join worst order   1.5x -> 1.8x
//! ```
//!
//! Because of that, the thresholds are only enforced at `BASELINE_ROWS`. Any
//! other size prints its table and asserts nothing.
//!
//! ## The run knows when not to assert
//!
//! Ratios here are asymmetric by construction -- a wide `Exchange` as
//! `chosen`, a pruned near-nothing as `best` -- and a forked plan waits on the
//! slowest of its workers, so shared cores inflate one side and not the other.
//! Left alone that produces false regressions: fourteen busy loops on the
//! other cores took `order by sort key + limit` from 44x to 176x. So the run
//! calibrates first (see [`fanout_probe`]) and prints its table either way,
//! withholding only the panic. A gate that cries wolf on a busy machine gets
//! muted, and a muted gate catches nothing.
//!
//! Environment overrides:
//!   `ROWS=1000000`   rows per side (default `BASELINE_ROWS`)
//!   `REPS=7`         A/B interleaved rounds, best-of per side
//!   `PLANS=1`        also dump `EXPLAIN PIPELINE` for both halves of each pair
//!   `NOASSERT=1`     print the table without enforcing the thresholds

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use granular::common::{hash_bytes, splitmix64};
use granular::sql::ast::ObjectName;
use granular::types::{Block, Column, DataType, Value};
use granular::{Result, ResultSet, Session};

// ------------------------------------------------------------------ fixtures

/// The eight-value dimension. `cat`/`cs` hold these, so a granule's dictionary
/// is 3 bits/row wide -- what ClickHouse calls `LowCardinality(String)` and
/// this engine applies unconditionally.
const CATS: [&str; 8] = ["US", "DE", "FR", "JP", "BR", "IN", "GB", "CA"];
/// The row count every threshold in this file was measured at. Ratios grow
/// with `n` -- some of them quadratically, because the cost of a plan that
/// prunes nothing is quadratic in what it failed to prune -- so a run at any
/// other size prints its table and enforces nothing.
const BASELINE_ROWS: u64 = 300_000;

/// A loose sanity bound on the identical-query controls: 1.01 - 1.16x quiet,
/// 1.4x with a `cargo test` on the other cores. This catches the harness
/// breaking, not contention -- see [`fanout_probe`] for that.
const NOISE_GATE: f64 = 1.5;

/// [`fanout_probe`] on a quiet machine at [`BASELINE_ROWS`], and how far above
/// it the run still trusts itself. Quiet it reads 4 - 5x and under load 18x,
/// so the gate has a factor of four to sit in and 2.0x is not a fine judgment.
const FANOUT_QUIET: f64 = 5.0;
const FANOUT_GATE: f64 = 2.0;

/// Planted exactly once, at row `probe`. Gives the low-cardinality column a
/// predicate with the same 1-of-N selectivity as one on the unique `tag`
/// column, so the two are comparable as a ratio.
const RARE: &str = "ZZ";

/// The dataset, plus every answer the shapes are pinned to.
///
/// The pins are worked out here, in Rust, from the same arithmetic that
/// generated the rows -- never by asking the engine what it thinks. A harness
/// that asks the engine for the expected answer and then checks the engine
/// against it agrees with any bug that is consistent.
struct Fix {
    n: u64,
    /// The key every point-lookup shape probes for. Mid-table, so it is not
    /// the first or last granule and cannot be found by accident.
    probe: u64,
    /// `a.v` at `probe`.
    v_probe: i64,
    /// Rows of `a` whose `cat` is `US`.
    us: u64,
    /// Distinct `cat` values with more than 1000 rows: 8, because `RARE` has
    /// exactly one. The control that a filter on an aggregate *result* must
    /// not be pushed below the aggregate.
    big_cats: u64,
    /// The `n/2`-th smallest `v`. `v` has ~n/1000 duplicates of every value,
    /// so which row this comes from is unobservable and the value is not.
    v_at_mid: i64,
    /// Rows of `c`, the small side of the three-way join.
    c_rows: u64,
}

fn v_of(i: u64) -> i64 {
    (splitmix64(i) % 1000) as i64
}

impl Fix {
    fn cat_of(&self, i: u64) -> &'static str {
        if i == self.probe {
            RARE
        } else {
            CATS[(splitmix64(i) >> 13) as usize % CATS.len()]
        }
    }

    /// `a`: the fact table. Sorted *and* keyed on `k`, because `pk_col()` --
    /// the gate on every index access path -- answers `None` for `ORDER BY`
    /// without a `PRIMARY KEY` declaration, and a benchmark of the index path
    /// that never reaches the index would be measuring nothing.
    fn build(db: &mut Session, n: u64) -> Result<Fix> {
        let mut f = Fix {
            n,
            probe: n / 2,
            v_probe: 0,
            us: 0,
            big_cats: 0,
            v_at_mid: 0,
            c_rows: 0,
        };
        f.v_probe = v_of(f.probe);

        db.execute(
            "CREATE TABLE a (
                k   UInt64,
                v   Int64,
                cat LowCardinality(String),
                cs  String,
                tag String
             ) ENGINE = MergeTree ORDER BY k PRIMARY KEY k",
        )?;
        db.execute(
            "CREATE TABLE b (k UInt64, w Int64) ENGINE = MergeTree ORDER BY k PRIMARY KEY k",
        )?;
        db.execute(
            "CREATE TABLE c (k UInt64, z Int64) ENGINE = MergeTree ORDER BY k PRIMARY KEY k",
        )?;

        let cats: Vec<Arc<str>> = (0..n).map(|i| f.cat_of(i).into()).collect();
        let vs: Vec<i64> = (0..n).map(v_of).collect();
        let a = Block::new(vec![
            Column::u64s(DataType::UInt64, (0..n).collect()),
            Column::i64s(DataType::Int64, vs.clone()),
            // Declared two ways over identical values: the pair is the whole
            // point of the `String` vs `LowCardinality(String)` shape.
            Column::strs(DataType::LowCardinality(Box::new(DataType::String)), cats.clone()),
            Column::strs(DataType::String, cats.clone()),
            Column::strs(DataType::String, (0..n).map(|i| format!("k{i:09}").into()).collect()),
        ])?;
        let b = Block::new(vec![
            Column::u64s(DataType::UInt64, (0..n).collect()),
            Column::i64s(DataType::Int64, (0..n).map(|i| i as i64 * 2).collect()),
        ])?;
        // The small side of the three-way join. 256 rows against 300k is the
        // ratio that makes join order matter: build over 256 or over 300k.
        let step = (n / 256).max(1);
        let ck: Vec<u64> = (0..256).map(|i| i * step).filter(|&k| k < n).collect();
        f.c_rows = ck.len() as u64;
        let c = Block::new(vec![
            Column::u64s(DataType::UInt64, ck.clone()),
            Column::i64s(DataType::Int64, ck.iter().map(|&k| k as i64).collect()),
        ])?;

        for (name, blk) in [("a", a), ("b", b), ("c", c)] {
            db.catalog.table_mut(&ObjectName::bare(name))?.insert(blk)?;
        }
        db.catalog.flush_all()?;

        f.us = cats.iter().filter(|s| &***s == "US").count() as u64;
        f.big_cats = CATS
            .iter()
            .chain(std::iter::once(&RARE))
            .filter(|w| cats.iter().filter(|s| &***s == **w).count() > 1000)
            .count() as u64;
        let mut sorted = vs;
        sorted.sort_unstable();
        f.v_at_mid = sorted[(n / 2) as usize];
        Ok(f)
    }
}

// ------------------------------------------------------------ answer pinning

/// Order-sensitive fingerprint of a whole result: `(rows, hash)`.
///
/// Order-sensitive on purpose. Two of the shapes here are about `ORDER BY`,
/// and a multiset comparison would call an unsorted result equal to a sorted
/// one -- which is exactly the bug a sort-elimination rule can introduce.
fn fingerprint(rs: &ResultSet) -> (usize, u64) {
    let mut h = 0x243f_6a88_85a3_08d3u64;
    for blk in &rs.blocks {
        for r in 0..blk.rows() {
            for c in 0..blk.width() {
                h = splitmix64(h ^ hash_value(&blk.column(c).value(r)));
            }
        }
    }
    (rs.rows(), h)
}

fn hash_value(v: &Value) -> u64 {
    match v {
        Value::Null => 0x9e37_79b9_7f4a_7c15,
        Value::Bool(b) => splitmix64(*b as u64 + 1),
        Value::UInt(u) => splitmix64(*u),
        Value::Int(i) | Value::DateTime(i) => splitmix64(*i as u64),
        Value::Float(f) => splitmix64(f.to_bits()),
        Value::Date(d) => splitmix64(*d as u64),
        Value::Decimal(u, s) => splitmix64(*u as u64) ^ splitmix64(*s as u64),
        Value::Str(s) => hash_bytes(s.as_bytes(), 0x5bf0_3635),
    }
}

// -------------------------------------------------------------- the shapes

/// Which of this wave's gaps a shape is measuring, so the table says what a
/// number means rather than leaving it to be rediscovered.
#[derive(Clone, Copy, PartialEq)]
enum Gap {
    /// `sink_filter` has no Join / Union / Aggregate / Distinct arms.
    Pushdown,
    /// No index-nested-loop join: one row on the left still builds a hash
    /// table over the whole right side.
    IndexJoin,
    /// `ORDER BY` on the table's own sort key still sorts.
    SortElim,
    /// `IN (SELECT ...)` materializes the subquery into a literal list.
    Decorrelate,
    /// No join reordering exists and none is planned in this wave. The ratio
    /// is the standing penalty for writing the joins in the wrong order.
    NoFix,
    /// A rule that already works. Here to prove it keeps working, and to prove
    /// the shapes above are not just measuring noise.
    Closed,
}

impl Gap {
    fn tag(self) -> &'static str {
        match self {
            Gap::Pushdown => "pushdown",
            Gap::IndexJoin => "index-join",
            Gap::SortElim => "sort-elim",
            Gap::Decorrelate => "decorrelate",
            Gap::NoFix => "NO FIX YET",
            Gap::Closed => "closed",
        }
    }

    /// The change that would take this group's ratios to 1.0.
    fn fix(self) -> &'static str {
        match self {
            Gap::Pushdown => "sink_filter arms for Join, Union, Aggregate and Distinct",
            Gap::IndexJoin => "an index-nested-loop join over the primary key",
            Gap::SortElim => "recognise that ORDER BY on the sort key is already satisfied",
            Gap::Decorrelate => "lower IN (SELECT ...) to a semi-join instead of a literal list",
            Gap::NoFix => "join reordering: does not exist, and has no owner this wave",
            Gap::Closed => "nothing: these are the controls, and 1.0x is the pass",
        }
    }
}

struct Shape {
    name: &'static str,
    chosen: String,
    best: String,
    /// The ratio this shape measured on 2026-08-08 at 336ca5c, best of three
    /// full runs. Printed beside the live ratio: a wave that closes a gap
    /// should show the number fall, and the fall is the deliverable.
    measured: f64,
    /// Fails the run above this. Set ~2.5x above `measured`, so this machine's
    /// 30% swing cannot trip it and a 10x regression cannot hide under it.
    max_ratio: f64,
    /// Rows out, and the first cell, both computed from the fixture's
    /// arithmetic rather than from the engine.
    ///
    /// Comparing the two halves against each other is not enough on its own:
    /// a rewrite that is wrong in the *same* way on both sides -- and a
    /// pushdown or sort-elimination rule fires on both sides here, because
    /// both are legal SQL for the same question -- would agree with itself.
    /// The pin is the independent witness.
    pin: (usize, Value),
    gap: Gap,
    why: &'static str,
}

fn shapes(f: &Fix) -> Vec<Shape> {
    let (n, probe) = (f.n, f.probe);
    // A 1000-key window in the middle of the table: wide enough that the
    // answer is not one granule, narrow enough that zone maps prune ~99.9%.
    let (lo, hi) = (probe, probe + 999);
    let tag = format!("k{probe:09}");
    let mid = n / 2;
    let (one, count) = (1usize, |k: u64| Value::UInt(k));
    let v_probe = Value::Int(f.v_probe);

    vec![
        // ---------------------------------------------------- gap 1: pushdown
        Shape {
            name: "filter above join",
            chosen: format!("SELECT count() FROM a JOIN b ON a.k = b.k WHERE a.k = {probe}"),
            best: format!(
                "SELECT count() FROM (SELECT k FROM a WHERE k = {probe}) AS s \
                 JOIN b ON s.k = b.k"
            ),
            measured: 4.2,
            max_ratio: 12.0,
            pin: (one, count(1)),
            gap: Gap::Pushdown,
            why: "predicate stays above InnerHashJoin; both sides build in full",
            // Deliberately paired against the hand-pushed subquery rather than
            // against the bare lookup, so this line isolates the *pushdown*
            // half of the join story and `pk lookup through a join` -- same
            // query, paired against the index probe -- carries both halves.
            // Once an index join lands, that split inverts: `best` here stops
            // being the best plan and the run flags the pair as stale.
        },
        Shape {
            name: "filter above union",
            chosen: format!(
                "SELECT count() FROM (SELECT k, v FROM a UNION ALL SELECT k, w FROM b) AS u \
                 WHERE u.k = {probe}"
            ),
            best: format!(
                "SELECT count() FROM (SELECT k, v FROM a WHERE k = {probe} \
                 UNION ALL SELECT k, w FROM b WHERE k = {probe}) AS u"
            ),
            measured: 77.0,
            max_ratio: 250.0,
            pin: (one, count(2)),
            gap: Gap::Pushdown,
            why: "each branch is a keyed table; pushed, both are index probes",
        },
        Shape {
            name: "filter above group by",
            chosen: format!(
                "SELECT n FROM (SELECT k, count() AS n FROM a GROUP BY k) AS g WHERE g.k = {probe}"
            ),
            best: format!(
                "SELECT count() AS n FROM (SELECT k FROM a WHERE k = {probe}) AS g GROUP BY k"
            ),
            measured: 698.0,
            max_ratio: 2000.0,
            pin: (one, count(1)),
            gap: Gap::Pushdown,
            why: "a grouping key is a pass-through; the filter can sink below it",
        },
        Shape {
            name: "filter above distinct",
            chosen: format!("SELECT k FROM (SELECT DISTINCT k FROM a) AS d WHERE d.k = {probe}"),
            best: format!("SELECT DISTINCT k FROM (SELECT k FROM a WHERE k = {probe}) AS d"),
            measured: 829.0,
            max_ratio: 2000.0,
            pin: (one, count(probe)),
            gap: Gap::Pushdown,
            why: "DISTINCT does not invent rows, so a filter commutes with it",
        },
        // ------------------------------------------------- gap 2: index joins
        Shape {
            name: "pk lookup through a join",
            chosen: format!("SELECT a.v FROM a JOIN b ON a.k = b.k WHERE a.k = {probe}"),
            best: format!("SELECT v FROM a WHERE k = {probe}"),
            measured: 1843.0,
            max_ratio: 4000.0,
            pin: (one, v_probe.clone()),
            gap: Gap::IndexJoin,
            why: "one left row against a primary key should be one MPH probe",
        },
        Shape {
            name: "pk lookup through a union",
            chosen: format!(
                "SELECT v FROM (SELECT k, v FROM a UNION ALL SELECT k, w FROM b) AS u \
                 WHERE u.k = {probe}"
            ),
            best: format!(
                "SELECT v FROM a WHERE k = {probe} UNION ALL SELECT w FROM b WHERE k = {probe}"
            ),
            measured: 109.0,
            max_ratio: 300.0,
            pin: (2, v_probe.clone()),
            gap: Gap::Pushdown,
            why: "same as the union count, projecting rows rather than folding",
        },
        Shape {
            name: "pk lookup through a subquery",
            chosen: format!("SELECT v FROM (SELECT k, v FROM a) AS s WHERE s.k = {probe}"),
            best: format!("SELECT v FROM a WHERE k = {probe}"),
            measured: 1.6,
            max_ratio: 4.0,
            pin: (one, v_probe),
            gap: Gap::Closed,
            why: "the Project arm of sink_filter already handles this",
        },
        // -------------------------------------------- gap 3: sort elimination
        Shape {
            name: "order by sort key + limit",
            chosen: "SELECT k FROM a ORDER BY k LIMIT 5".into(),
            best: "SELECT k FROM a LIMIT 5".into(),
            measured: 43.8,
            max_ratio: 120.0,
            pin: (5, count(0)),
            gap: Gap::SortElim,
            why: "TopK under an Exchange for the first 5 rows of granule 0",
        },
        Shape {
            name: "order by sort key",
            chosen: "SELECT k FROM a ORDER BY k".into(),
            best: "SELECT k FROM a".into(),
            measured: 3.9,
            max_ratio: 12.0,
            pin: (n as usize, count(0)),
            gap: Gap::SortElim,
            why: "a full sort of data that is already stored in that order",
        },
        Shape {
            name: "order by sort key, aggregated",
            chosen: "SELECT sum(k) FROM (SELECT k FROM a ORDER BY k) AS s".into(),
            best: "SELECT sum(k) FROM (SELECT k FROM a) AS s".into(),
            measured: 6.8,
            max_ratio: 20.0,
            pin: (one, count(n * (n - 1) / 2)),
            gap: Gap::SortElim,
            why: "sorting under a fold that cannot observe order at all",
        },
        Shape {
            name: "order by sort key, descending",
            chosen: "SELECT k FROM a ORDER BY k DESC LIMIT 5".into(),
            // The last granule, then a five-row sort. `max(k)` is in the part
            // metadata, so this bound is one the planner could derive too --
            // which makes it a fair "best available", not a cheat.
            best: format!("SELECT k FROM a WHERE k >= {} ORDER BY k DESC", n - 5),
            measured: 13.8,
            max_ratio: 40.0,
            pin: (5, count(n - 1)),
            gap: Gap::SortElim,
            why: "descending on the sort key is the same scan read backwards",
        },
        // ------------------------------------------------------ decorrelation
        Shape {
            name: "IN (SELECT) vs semi-join",
            chosen: "SELECT count() FROM a WHERE k IN (SELECT k FROM b)".into(),
            best: "SELECT count() FROM a JOIN b ON a.k = b.k".into(),
            measured: 3.4,
            max_ratio: 10.0,
            pin: (one, count(n)),
            gap: Gap::Decorrelate,
            why: "the subquery becomes N literals, re-projected once per block",
        },
        // ----------------------------------------------------- join ordering
        //
        // All five distinct left-deep spellings of the same three-way join,
        // 300k x 300k x 256, best of five each (2026-08-08, 336ca5c):
        //
        //     b JOIN a JOIN c   15.92 ms      a JOIN c JOIN b   11.13 ms
        //     a JOIN b JOIN c   15.76 ms      c JOIN b JOIN a   11.26 ms
        //                                     c JOIN a JOIN b   10.89 ms
        //
        // 1.46x, not the two orders of magnitude the textbook promises, and
        // the reason is worth writing down before someone builds a cost model
        // to fix it: `join.rs` already picks the *smaller* side to build,
        // adaptively, at run time. That caps the hash table at min(|L|,|R|) no
        // matter how the query was written, so the only thing a bad order
        // still costs is the intermediate -- 300k rows through the second join
        // instead of 256. Both spellings scan all three tables in full either
        // way, and at this size the scans are most of the 11 ms floor.
        //
        // So the join-reordering gap here is real but bounded, and it is the
        // *cheapest* of the gaps this file measures, not the dearest. Anyone
        // ranking work off this table should read the top line, not this one.
        Shape {
            name: "3-join, worst order",
            chosen: "SELECT count() FROM a JOIN b ON a.k = b.k JOIN c ON b.k = c.k".into(),
            best: "SELECT count() FROM c JOIN a ON c.k = a.k JOIN b ON a.k = b.k".into(),
            measured: 1.5,
            max_ratio: 4.0,
            pin: (one, count(f.c_rows)),
            gap: Gap::NoFix,
            why: "written order is executed order, but the adaptive build side caps the damage",
        },
        // ------------------------------------------------- predicate spelling
        Shape {
            name: "count() with an always-true filter",
            chosen: "SELECT count() FROM a WHERE k >= 0".into(),
            best: "SELECT count() FROM a".into(),
            measured: 2.8,
            max_ratio: 8.0,
            pin: (one, count(n)),
            gap: Gap::Closed,
            why: "a per-granule zone-map test the bare fold skips: 293 granules, ~7 us",
        },
        Shape {
            name: "range written as NOT(...)",
            chosen: format!("SELECT count() FROM a WHERE NOT (k < {lo} OR k > {hi})"),
            best: format!("SELECT count() FROM a WHERE k >= {lo} AND k <= {hi}"),
            measured: 10.2,
            max_ratio: 30.0,
            pin: (one, count(1000)),
            gap: Gap::Pushdown,
            why: "De Morgan would hand the zone maps a range they can prune on",
        },
        Shape {
            name: "range written as OR of ranges",
            chosen: format!(
                "SELECT count() FROM a WHERE (k >= {lo} AND k <= {}) \
                 OR (k >= {} AND k <= {hi})",
                lo + 499,
                lo + 500
            ),
            best: format!("SELECT count() FROM a WHERE k >= {lo} AND k <= {hi}"),
            measured: 14.7,
            max_ratio: 40.0,
            pin: (one, count(1000)),
            gap: Gap::Pushdown,
            why: "a disjunction of ranges bounds the granule just as tightly",
        },
        // ------------------------------------------------ dictionary encoding
        Shape {
            name: "String vs LowCardinality(String)",
            chosen: "SELECT count() FROM a WHERE cs = 'US'".into(),
            best: "SELECT count() FROM a WHERE cat = 'US'".into(),
            measured: 1.0,
            max_ratio: 2.0,
            pin: (one, count(f.us)),
            gap: Gap::Closed,
            why: "identical storage by design; the declaration must cost nothing",
        },
        Shape {
            name: "wide dictionary vs narrow",
            chosen: format!("SELECT count() FROM a WHERE tag = '{tag}'"),
            best: format!("SELECT count() FROM a WHERE cat = '{RARE}'"),
            measured: 1.5,
            max_ratio: 4.0,
            pin: (one, count(1)),
            gap: Gap::Closed,
            why: "one row out of both; ~19-bit codes over a 3 MB blob vs 3-bit codes",
        },
        // ------------------------------------------------------ negative cases
        //
        // Shapes the rules being written this wave must NOT fire on. Each one
        // is identical on both sides, so its ratio is the harness's own noise
        // floor -- what a "1.0x" is worth on this machine. The load-bearing
        // half is the pin: a sort-elimination rule that ignores direction, or
        // a pushdown that sinks a predicate below the aggregate that computes
        // its operand, changes the answer *on both sides at once* and only an
        // independently computed expectation catches it.
        Shape {
            name: "order by non-sort-key (control)",
            chosen: format!("SELECT v FROM a ORDER BY v LIMIT 5 OFFSET {mid}"),
            best: format!("SELECT v FROM a ORDER BY v LIMIT 5 OFFSET {mid}"),
            measured: 1.0,
            max_ratio: 2.5,
            pin: (5, Value::Int(f.v_at_mid)),
            gap: Gap::Closed,
            why: "v is not the sort key, so no rewrite may remove this sort",
        },
        Shape {
            name: "filter on an aggregate (control)",
            chosen: "SELECT count() FROM (SELECT cat, count() AS m FROM a GROUP BY cat) AS g \
                     WHERE g.m > 1000"
                .into(),
            best: "SELECT count() FROM (SELECT cat, count() AS m FROM a GROUP BY cat) AS g \
                   WHERE g.m > 1000"
                .into(),
            measured: 1.0,
            max_ratio: 2.5,
            pin: (one, count(f.big_cats)),
            gap: Gap::Closed,
            why: "m is an aggregate output; pushing it below the GROUP BY is wrong",
        },
    ]
}

// ------------------------------------------------------------- measurement

/// How badly this machine is inflating a fan-out plan relative to a pruned
/// one, right now.
///
/// The identical-query controls cannot see this and it took a false alarm to
/// notice: both halves of an identical pair are equally parallel, so
/// contention divides out of their ratio. It does not divide out of the
/// shapes that matter here, which are asymmetric by construction -- a wide
/// `Exchange` over every granule as `chosen`, a zone-map-pruned near-nothing
/// as `best`. A forked plan waits on the *slowest* of its workers, so when
/// something else owns the cores it inflates super-linearly while a plan that
/// touches two granules barely moves.
///
/// Measured at 300k, best of five, against fourteen busy loops on the other
/// cores:
///
/// ```text
///                              quiet          loaded      separates?
///   fan-out probe            4 - 5x        18 - 18x      yes, 4x apart
///   identical-pair floor  1.01 - 1.16x   1.16 - 1.22x    no, overlapping
/// ```
///
/// The floor is kept and printed -- it caught a `cargo test` sharing the
/// machine, where it read 1.4x -- but it is not the gate, because under the
/// load above it stayed *below* its own threshold while three shapes tripped
/// theirs. `order by sort key + limit` read 176x against a 44x baseline in
/// that run. A gate that misses the case it was added for is not a gate.
///
/// The two halves answer different questions, so this cannot be a `Shape`;
/// it is calibration, not a measurement of the planner.
fn fanout_probe(db: &mut Session, probe: u64, reps: u32) -> Result<f64> {
    let pruned = format!("SELECT sum(v) FROM a WHERE k >= {probe} AND k <= {}", probe + 999);
    let (mut wide, mut narrow) = (f64::MAX, f64::MAX);
    for _ in 0..reps {
        let t = Instant::now();
        black_box(db.query("SELECT sum(v) FROM a")?);
        wide = wide.min(t.elapsed().as_secs_f64());

        let t = Instant::now();
        black_box(db.query(&pruned)?);
        narrow = narrow.min(t.elapsed().as_secs_f64());
    }
    Ok(wide / narrow)
}

/// One shape, A/B interleaved, best-of-`reps` per side.
///
/// Interleaved rather than run-all-of-A-then-all-of-B because this machine
/// drifts 30%+ over the length of a benchmark: a serial layout charges that
/// drift entirely to whichever side ran second. Best-of rather than mean
/// because the noise here is one-sided -- another process can only ever make a
/// run slower.
fn race(db: &mut Session, s: &Shape, reps: u32) -> Result<(f64, f64)> {
    let (mut chosen, mut best) = (f64::MAX, f64::MAX);
    for _ in 0..reps {
        let t = Instant::now();
        black_box(db.query(&s.chosen)?);
        chosen = chosen.min(t.elapsed().as_secs_f64());

        let t = Instant::now();
        black_box(db.query(&s.best)?);
        best = best.min(t.elapsed().as_secs_f64());
    }
    Ok((chosen, best))
}

/// The operator chain `EXPLAIN PIPELINE` reports, as one line.
///
/// Derived rather than hand-written next to each shape: a note that says
/// "Filter above InnerHashJoin" is only true until someone fixes it, and a
/// benchmark whose commentary silently goes stale is how the gaps got here.
/// `Project` is dropped as noise -- it appears in nearly every plan and never
/// distinguishes two of them.
fn plan_chain(db: &mut Session, sql: &str) -> Result<String> {
    let rs = db.query(&format!("EXPLAIN PIPELINE {sql}"))?;
    let mut out: Vec<String> = Vec::new();
    for row in rs.to_values() {
        let Some(line) = row.first().and_then(|v| v.as_str()) else { continue };
        let Some(op) = line.trim_start().split(' ').next() else { continue };
        if op == "Project" || op.is_empty() {
            continue;
        }
        // Repeated leaves say nothing: a two-way join reads `Scan Scan`.
        match out.last_mut() {
            Some(prev) if prev.starts_with(op) => {
                let n: u32 = prev[op.len()..].trim_start_matches('x').parse().unwrap_or(1);
                *prev = format!("{op}x{}", n + 1);
            }
            _ => out.push(op.to_string()),
        }
    }
    Ok(out.join(" < "))
}

/// One measured shape. Borrows its `Shape` rather than copying the strings out
/// of it: the table wants the threshold and the baseline next to the number.
struct Row<'a> {
    shape: &'a Shape,
    chosen: f64,
    best: f64,
    ratio: f64,
    chain: String,
}

fn rule(title: &str) {
    println!("\n\x1b[1m== {title} ==\x1b[0m");
}

fn ms(t: f64) -> String {
    if t < 1e-3 {
        format!("{:.3} ms", t * 1e3)
    } else {
        format!("{:.2} ms", t * 1e3)
    }
}

/// Ratios span five orders of magnitude here, and a fixed `{:.1}` prints the
/// two interesting extremes -- 1843x and 0.007x -- as "1843.0x" and "0.0x".
fn ratio_str(r: f64) -> String {
    match r {
        x if x >= 100.0 => format!("{x:.0}x"),
        x if x >= 1.0 => format!("{x:.1}x"),
        x if x >= 0.01 => format!("{x:.2}x"),
        x => format!("{x:.4}x"),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("\x1b[31mplanner bench failed: {e}\x1b[0m");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let n: u64 =
        std::env::var("ROWS").ok().and_then(|s| s.parse().ok()).unwrap_or(BASELINE_ROWS);
    let reps: u32 = std::env::var("REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let show_plans = std::env::var("PLANS").is_ok();
    let enforce = std::env::var("NOASSERT").is_err();

    rule(&format!("planner quality: {n} x {n} rows, {reps} interleaved rounds"));
    let mut db = Session::in_memory();
    let t0 = Instant::now();
    let fix = Fix::build(&mut db, n)?;
    let probe = fix.probe;
    println!("load: {:?}", t0.elapsed());

    // Fixture gates. Every ratio below is meaningless if the data is not the
    // shape the queries assume, and each of these has a specific way to be
    // wrong: `a` and `b` must cover the same key space or the joins go empty,
    // `c` must stay small or the join-order shape stops measuring order, and
    // the planted rare value must be unique or the dictionary shapes compare
    // two different selectivities.
    let one = |db: &mut Session, sql: &str| -> Result<u64> {
        db.query(sql)?
            .scalar()
            .and_then(|v| v.as_u64())
            .ok_or_else(|| granular::Error::exec(format!("`{sql}` produced no scalar")))
    };
    assert_eq!(one(&mut db, "SELECT count() FROM a")?, n);
    assert_eq!(one(&mut db, "SELECT count() FROM b")?, n);
    assert_eq!(one(&mut db, "SELECT count() FROM c")?, fix.c_rows);
    assert!(fix.c_rows * 64 < n, "`c` is not small enough to make join order matter");
    assert_eq!(one(&mut db, "SELECT count() FROM a JOIN b ON a.k = b.k")?, n);
    assert_eq!(one(&mut db, &format!("SELECT count() FROM a WHERE cat = '{RARE}'"))?, 1);
    assert_eq!(one(&mut db, &format!("SELECT count() FROM a WHERE tag = 'k{probe:09}'"))?, 1);
    assert_eq!(one(&mut db, "SELECT uniqExact(cat) FROM a")?, CATS.len() as u64 + 1);
    assert_eq!(one(&mut db, "SELECT uniqExact(tag) FROM a")?, n);
    // The premise of the whole sort-elimination section: storage order already
    // *is* key order, so `ORDER BY k` is asking for what the scan hands over.
    let unordered = db.query("SELECT k FROM a")?;
    assert_eq!(unordered.rows(), n as usize);
    let mut last = None;
    for blk in &unordered.blocks {
        for r in 0..blk.rows() {
            let k = blk.column(0).value(r).as_u64().unwrap();
            assert!(last.is_none_or(|p| p <= k), "a bare scan of `a` is not in key order");
            last = Some(k);
        }
    }
    println!("fixture gates passed ✔");

    // Calibration first, so a machine that is already busy is known to be busy
    // before any shape is timed against it.
    let fanout = fanout_probe(&mut db, probe, reps)?;

    let shapes = shapes(&fix);
    let mut rows: Vec<Row> = Vec::with_capacity(shapes.len());

    for s in &shapes {
        // Pin the answer before timing anything. Both halves run once here, so
        // this doubles as the warm-up the timing loop would otherwise need.
        let (rc, rb) = (db.query(&s.chosen)?, db.query(&s.best)?);
        assert_eq!(
            fingerprint(&rc),
            fingerprint(&rb),
            "`{}`: the two halves disagree.\n  chosen: {}\n  best:   {}\n\
             A ratio between two different answers is not a measurement.",
            s.name,
            s.chosen,
            s.best
        );
        for (half, rs) in [("chosen", &rc), ("best", &rb)] {
            assert_eq!(
                (rs.rows(), rs.scalar().unwrap_or(Value::Null)),
                s.pin,
                "`{}` ({half}) does not match the answer computed from the fixture; \
                 a ratio for the wrong answer is worse than no ratio",
                s.name
            );
        }

        let (chosen, best) = race(&mut db, s, reps)?;
        let chain = plan_chain(&mut db, &s.chosen)?;
        if show_plans {
            println!("\n  {}\n    chosen: {}\n      {}", s.name, s.chosen, chain);
            println!("    best:   {}\n      {}", s.best, plan_chain(&mut db, &s.best)?);
        }
        rows.push(Row { shape: s, chosen, best, ratio: chosen / best, chain });
    }

    // Worst first: the top line of this table is the next thing to fix.
    rows.sort_by(|x, y| y.ratio.total_cmp(&x.ratio));

    rule("chosen plan vs best available plan, worst first");
    let baseline = n == BASELINE_ROWS;
    println!(
        "{:<34} {:>10} {:>10} {:>8} {:>9}  {:<12} {}",
        "shape",
        "chosen",
        "best",
        "ratio",
        if baseline { "336ca5c" } else { "" },
        "gap",
        "what the engine chose"
    );
    for r in &rows {
        let colour = match r.ratio {
            x if x >= 10.0 => "\x1b[31m",
            x if x >= 3.0 => "\x1b[33m",
            // Below 1.0 the planner beat the plan this file calls "best",
            // which is a result about the *harness*: the hand-written half has
            // gone stale and is understating the win. Blue, so it does not
            // read as another passing green line.
            x if x < 0.9 => "\x1b[36m",
            _ => "\x1b[32m",
        };
        // The baseline column is only meaningful at the row count it was taken
        // at; every one of these ratios grows with n, several of them
        // quadratically, so printing it next to a `ROWS=1000000` run would
        // invite exactly the comparison it cannot support.
        let was = if baseline { ratio_str(r.shape.measured) } else { String::new() };
        println!(
            "{:<34} {:>10} {:>10} {colour}{:>8}\x1b[0m {was:>9}  {:<12} {}",
            r.shape.name,
            ms(r.chosen),
            ms(r.best),
            ratio_str(r.ratio),
            r.shape.gap.tag(),
            r.chain,
        );
    }

    println!("\nwhy each ratio is what it is:");
    for r in &rows {
        if r.ratio >= 2.0 || r.shape.gap == Gap::NoFix {
            println!("  {:<34} {}", r.shape.name, r.shape.why);
        }
    }

    // What should move next time this runs, and what must not. Grouped by gap
    // so the integrator can read one wave's worth of work off the output
    // instead of reconstructing which shapes a change was supposed to touch.
    // A ratio below 1 is not a pass, it is a stale shape: the optimizer found
    // something the hand-written half does not do, so the number understates
    // the win and the pair should be rewritten around the new best plan.
    let stale: Vec<&Row> = rows.iter().filter(|r| r.ratio < 0.9).collect();
    if !stale.is_empty() {
        println!("\n\x1b[36mthe planner beat the hand-written plan -- update these pairs:\x1b[0m");
        for r in stale {
            println!(
                "  {:<34} {} ({} chosen vs {} best)",
                r.shape.name,
                ratio_str(r.ratio),
                ms(r.chosen),
                ms(r.best)
            );
        }
    }

    println!("\nwhat closes each group:");
    for gap in [Gap::IndexJoin, Gap::Pushdown, Gap::SortElim, Gap::Decorrelate, Gap::NoFix, Gap::Closed]
    {
        let hit: Vec<&Row> = rows.iter().filter(|r| r.shape.gap == gap).collect();
        let worst = hit.iter().map(|r| r.ratio).fold(0.0f64, f64::max);
        println!(
            "  {:<12} {:<62} {:>2} shape{}, worst {}",
            gap.tag(),
            gap.fix(),
            hit.len(),
            if hit.len() == 1 { " " } else { "s" },
            ratio_str(worst),
        );
    }

    // How flat the noise floor actually was, measured rather than assumed.
    //
    // The identical-query controls race a query against itself, so anything
    // they report other than 1.0 is this machine. That makes them a live
    // contention detector, and it earns its keep: with a `cargo test` running
    // on the other cores they read 1.4x and 0.7x, while `order by sort key,
    // descending` -- a 14-worker TopK raced against a serial 5-row scan, so
    // contention lands on one side only -- inflated from 13.8x to 107x and
    // tripped a threshold it has never been near on a quiet machine.
    //
    // Refusing to assert is the right answer there. A benchmark that cries
    // regression whenever the machine is busy gets muted, and a muted gate
    // catches nothing. The table still prints; only the panic is withheld.
    let floor = rows
        .iter()
        .filter(|r| r.shape.chosen == r.shape.best)
        .map(|r| r.ratio.max(1.0 / r.ratio))
        .fold(1.0f64, f64::max);
    let skew = fanout / FANOUT_QUIET;
    let quiet = floor <= NOISE_GATE && skew <= FANOUT_GATE;
    println!(
        "\nnoise floor {floor:.2}x (identical query against itself), \
         fan-out skew {skew:.2}x ({fanout:.0}x vs {FANOUT_QUIET:.0}x quiet)  {}",
        if quiet { "-- quiet enough to assert" } else { "-- TOO NOISY, not asserting" }
    );

    // Thresholds last, and all of them, so one regression does not hide the
    // rest of the table behind a panic.
    let over: Vec<String> = rows
        .iter()
        .filter(|r| r.ratio > r.shape.max_ratio)
        .map(|r| {
            format!(
                "  {:<34} {} > {} (was {})",
                r.shape.name,
                ratio_str(r.ratio),
                ratio_str(r.shape.max_ratio),
                ratio_str(r.shape.measured)
            )
        })
        .collect();
    let enforced = enforce && baseline && quiet;
    if !over.is_empty() {
        let head = format!("{} shape(s) past their threshold:\n{}", over.len(), over.join("\n"));
        assert!(
            !enforced,
            "{head}\nThresholds sit ~2.5x above the 2026-08-08 measurement at 336ca5c, and \
             the machine measured quiet (floor {floor:.2}x, fan-out skew {skew:.2}x), \
             so this is not contention."
        );
        println!("\n\x1b[33m{head}\n(not enforced)\x1b[0m");
    }
    if !baseline {
        println!(
            "ROWS={n}, not the {BASELINE_ROWS} the thresholds were taken at -- \
             ratios printed, nothing enforced."
        );
    }
    if enforced {
        println!("\n\x1b[32mAll planner-quality thresholds held.\x1b[0m");
    }
    Ok(())
}
