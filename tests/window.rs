//! Randomized differential testing of **window functions** against the
//! `sqlite3` CLI.
//!
//! # Why this is a separate file from `differential.rs`
//!
//! Not because window functions deserve their own harness in principle -- they
//! do not -- but because the thing that has to be generated is different in
//! kind. `differential.rs` varies a schema, rows and a whole query; what makes
//! a window function wrong is the *frame*, and a frame is only interesting in
//! combination with ties, partition edges, NULLs and tiny partitions. So this
//! generator holds the schema fixed and varies the one axis that matters,
//! which gets far more frame coverage per case than bolting an `OVER` clause
//! onto the existing query generator would.
//!
//! # The determinism problem, and how it is handled
//!
//! A window function's answer is only well defined up to the ordering its
//! `OVER` clause gives. `row_number() OVER (ORDER BY k)` over rows that tie on
//! `k` may legitimately number them either way, in either engine, so comparing
//! it would test the two sort implementations rather than the two window
//! implementations. Every case therefore declares itself either:
//!
//!   * **totally ordered** -- the window's ORDER BY ends with the unique `id`,
//!     so no ties exist and *every* function and frame is fair game; or
//!   * **tied** -- the ORDER BY is deliberately low-cardinality, and the case
//!     is then restricted to the functions and frames that are defined in terms
//!     of peer *groups* rather than row positions: `rank`, `dense_rank`,
//!     `percent_rank`, `cume_dist`, and aggregates under a `RANGE` frame.
//!
//! The tied half is the valuable half. `RANGE ... CURRENT ROW` means "through
//! the last row tied with me", and reading it as `ROWS` -- which is the obvious
//! mistake, and the one this engine's parser refuses to let you make silently
//! -- gives every tied row a different running total. Only a tied case can see
//! that.
//!
//! # Running it
//!
//! ```text
//!   cargo test --test window                    DEFAULT_CASES cases
//!   GRANULAR_WINDOW_CASES=20000 cargo test --release --test window
//!   GRANULAR_WINDOW_SEED=12345                  case n uses seed+n
//!   GRANULAR_DIFF_SQLITE=/path                  which sqlite3 (shared with
//!                                               differential.rs on purpose)
//! ```
//!
//! Without a `sqlite3` on the box every test here skips rather than fails: an
//! oracle that is not installed is not a defect in the engine.

use std::fmt::Write as _;
use std::io::Write as _;
use std::process::{Command, Stdio};

use granular::common::splitmix64;
use granular::types::Value;
use granular::Session;

/// Enough to exercise every menu entry several times over while `cargo test`
/// stays interactive. The volume runs are what the env var is for.
const DEFAULT_CASES: usize = 400;

/// Cases per `sqlite3` process. Spawning dominates; the rows are tiny.
const BATCH: usize = 40;

/// Rows per case. Deliberately small: a 6-row table makes single-row
/// partitions, empty frames and partition edges the *common* case rather than
/// something a large random table reaches by accident.
const MAX_ROWS: usize = 12;

/// Emitted between cases in a batch so the reader can attribute rows. No
/// generated value can produce it -- the text pool is single lowercase letters.
const SENTINEL: &str = "<<granular-window-boundary>>";

/// Relative tolerance for float comparison. `avg`, `percent_rank` and
/// `cume_dist` are divisions, so the two engines can differ in the last bit
/// without either being wrong.
const FLOAT_REL_TOL: f64 = 1e-12;

// ============================================================ random source

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = splitmix64(self.0);
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn pick<'a>(&mut self, v: &'a [&'a str]) -> &'a str {
        v[self.below(v.len())]
    }
    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }
}

// ================================================================ the case

/// One row of the fixed schema `(id, g, v, f)`.
///
/// `v` is nullable on purpose: NULLs in a `PARTITION BY` key must group
/// together, NULLs in an `ORDER BY` key must tie with each other, and NULLs in
/// an aggregate argument must be skipped. Three separate rules, one column.
struct Row {
    id: i64,
    g: &'static str,
    v: Option<i64>,
    f: f64,
}

struct Case {
    seed: u64,
    rows: Vec<Row>,
    query: String,
}

impl Case {
    /// The full script for one dialect, DDL included.
    fn script(&self, sqlite: bool) -> String {
        let mut s = String::new();
        if sqlite {
            s.push_str("DROP TABLE IF EXISTS w;\nCREATE TABLE w (id INTEGER, g TEXT, v INTEGER, f REAL);\n");
        } else {
            s.push_str(
                "CREATE TABLE w (id Int64, g String, v Nullable(Int64), f Float64) \
                 ENGINE = MergeTree ORDER BY id;\n",
            );
        }
        if !self.rows.is_empty() {
            s.push_str("INSERT INTO w VALUES ");
            for (i, r) in self.rows.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                let v = match r.v {
                    Some(x) => x.to_string(),
                    None => "NULL".into(),
                };
                let _ = write!(s, "({}, '{}', {}, {:?})", r.id, r.g, v, r.f);
            }
            s.push_str(";\n");
        }
        let _ = writeln!(s, "{};", self.query);
        s
    }
}

// ============================================================== generation

/// Frames whose bounds are row *positions*. Only ever paired with a total
/// order -- under ties they are as arbitrary as the sort that produced them.
const ROWS_FRAMES: &[&str] = &[
    "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
    "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
    "ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING",
    "ROWS BETWEEN 2 PRECEDING AND CURRENT ROW",
    "ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING",
    "ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING",
    // The two that can select *nothing*: an aggregate over them is NULL
    // (except count, which is 0), and the row it happens on is the first or
    // the last of its partition.
    "ROWS BETWEEN 1 FOLLOWING AND 3 FOLLOWING",
    "ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING",
    "ROWS 2 PRECEDING",
    "ROWS UNBOUNDED PRECEDING",
];

/// Frames whose bounds are peer *groups*. Safe under ties, which is the whole
/// reason `RANGE` exists.
const RANGE_FRAMES: &[&str] = &[
    "",
    "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
    "RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
    "RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING",
];

/// Functions defined in terms of peer groups, so a tie in the ORDER BY cannot
/// make them ambiguous.
const TIE_SAFE_FNS: &[&str] = &[
    "rank()",
    "dense_rank()",
    "percent_rank()",
    "cume_dist()",
    "count(*)",
    "count(v)",
    "sum(v)",
    "avg(v)",
    "min(v)",
    "max(v)",
    "sum(f)",
];

/// Functions that read a row *position*, and are therefore only well defined
/// once the ordering is total.
const POSITIONAL_FNS: &[&str] = &[
    "row_number()",
    "ntile(2)",
    "ntile(3)",
    "ntile(4)",
    "lag(v)",
    "lag(v, 2)",
    "lag(v, 1, -1)",
    "lead(v)",
    "lead(v, 2, 0)",
    "lead(v, 3)",
    "first_value(v)",
    "last_value(v)",
    "nth_value(v, 2)",
    "nth_value(v, 3)",
];

const PARTITIONS: &[&str] = &["", "PARTITION BY g", "PARTITION BY g, v % 2"];

fn gen_case(seed: u64) -> Case {
    let mut r = Rng(seed);
    let n = 1 + r.below(MAX_ROWS);
    let letters = ["a", "b", "c"];
    let mut rows = Vec::with_capacity(n);
    for id in 0..n {
        // A tiny value pool, so ties are the norm rather than a rarity.
        let v = if r.chance(20) { None } else { Some((r.below(5) as i64) - 1) };
        rows.push(Row {
            id: id as i64,
            g: {
                // Sometimes collapse every row into one group, so the
                // one-partition and many-partition shapes both show up.
                let pool = if r.chance(30) { 1 } else { letters.len() };
                letters[r.below(pool)]
            },
            v,
            f: (r.below(7) as f64) / 2.0 - 1.5,
        });
    }

    // Total order: the window's ORDER BY ends with the unique `id`, so no two
    // rows are peers and every function is deterministic. Tied: deliberately
    // low-cardinality, and the menus below shrink to match.
    let total = r.chance(60);
    let part = r.pick(PARTITIONS);
    let (order, fns, frames): (String, &[&str], &[&str]) = if total {
        let lead = ["", "g, ", "v, ", "v DESC, ", "f, "];
        let dir = if r.chance(30) { " DESC" } else { "" };
        (
            format!("ORDER BY {}id{dir}", r.pick(&lead)),
            if r.chance(50) { POSITIONAL_FNS } else { TIE_SAFE_FNS },
            if r.chance(70) { ROWS_FRAMES } else { RANGE_FRAMES },
        )
    } else {
        let key = r.pick(&["g", "v", "v % 3", "f"]);
        let dir = if r.chance(30) { " DESC" } else { "" };
        // No ORDER BY at all is also a tie case: every row of the partition is
        // then one peer group, which is exactly the whole-partition frame.
        let o = if r.chance(20) { String::new() } else { format!("ORDER BY {key}{dir}") };
        (o, TIE_SAFE_FNS, RANGE_FRAMES)
    };

    let frame = if order.is_empty() { "" } else { r.pick(frames) };
    let call = r.pick(fns);
    let mut over = String::new();
    for piece in [part, order.as_str(), frame] {
        if piece.is_empty() {
            continue;
        }
        if !over.is_empty() {
            over.push(' ');
        }
        over.push_str(piece);
    }

    // The outer ORDER BY is always the unique key, so the two engines' row
    // order is comparable no matter what the window did.
    let query = format!("SELECT id, {call} OVER ({over}) FROM w ORDER BY id");
    Case { seed, rows, query }
}

// ================================================================ comparing

#[derive(Debug, Clone, PartialEq)]
enum Cell {
    Null,
    Int(i64),
    Real(f64),
    Text(String),
}

impl Cell {
    /// Equal *as a SQL value*, not as a Rust value.
    ///
    /// Integer and real are compared across the divide because the two engines
    /// legitimately disagree on the width of an answer -- SQLite's `sum` of
    /// integers is an INTEGER, granular's is a `UInt64`, and `avg` is a REAL in
    /// both but reached by different arithmetic.
    fn agrees(&self, other: &Cell) -> bool {
        match (self, other) {
            (Cell::Null, Cell::Null) => true,
            (Cell::Text(a), Cell::Text(b)) => a == b,
            (Cell::Int(a), Cell::Int(b)) => a == b,
            _ => match (self.as_f64(), other.as_f64()) {
                (Some(a), Some(b)) => close(a, b),
                _ => false,
            },
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Cell::Int(i) => Some(*i as f64),
            Cell::Real(f) => Some(*f),
            _ => None,
        }
    }
}

fn close(a: f64, b: f64) -> bool {
    if a == b || (a.is_nan() && b.is_nan()) {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() <= FLOAT_REL_TOL * scale
}

fn cell_of(v: &Value) -> Cell {
    match v {
        Value::Null => Cell::Null,
        Value::Int(i) => Cell::Int(*i),
        Value::UInt(u) => Cell::Int(*u as i64),
        Value::Bool(b) => Cell::Int(*b as i64),
        Value::Float(f) => Cell::Real(*f),
        Value::Str(s) => Cell::Text(s.to_string()),
        other => Cell::Text(other.to_string()),
    }
}

// ============================================================ the two engines

fn sqlite_path() -> Option<&'static str> {
    static FOUND: std::sync::OnceLock<Option<&'static str>> = std::sync::OnceLock::new();
    *FOUND.get_or_init(|| {
        let runs = |p: &str| {
            Command::new(p)
                .arg("-version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        };
        if let Ok(p) = std::env::var("GRANULAR_DIFF_SQLITE") {
            return runs(&p).then(|| &*Box::leak(p.into_boxed_str()));
        }
        ["/usr/bin/sqlite3", "sqlite3"].into_iter().find(|p| runs(p))
    })
}

fn skip_without_sqlite() -> bool {
    if sqlite_path().is_some() {
        return false;
    }
    eprintln!("skipping: no sqlite3 on this machine, so there is no oracle to diff against");
    true
}

const PREAMBLE: &str = "\
.bail on
.mode quote
.separator \"\\t\"
.headers off
";

/// `.mode quote` rather than TSV for the same reason `differential.rs` uses it:
/// TSV cannot tell NULL from the string `'NULL'`, and a window function that
/// returned the wrong one of those would look identical.
fn sqlite_raw(script: &str) -> Option<(String, String)> {
    let bin = sqlite_path()?;
    let mut child = Command::new(bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    // Written from a helper thread: a big result can fill the stdout pipe
    // while we are still writing stdin, and that deadlock is exactly the flake
    // a test harness must not have.
    let mut stdin = child.stdin.take().expect("piped");
    let owned = script.to_string();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(owned.as_bytes());
    });
    let out = child.wait_with_output().ok()?;
    let _ = writer.join();
    Some((
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    ))
}

fn parse_quote_row(line: &str) -> Vec<Cell> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\'' {
            let mut s = String::new();
            i += 1;
            while i < b.len() {
                if b[i] == b'\'' {
                    if i + 1 < b.len() && b[i + 1] == b'\'' {
                        s.push('\'');
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                s.push(b[i] as char);
                i += 1;
            }
            out.push(Cell::Text(s));
        } else {
            let start = i;
            while i < b.len() && b[i] != b'\t' {
                i += 1;
            }
            out.push(parse_scalar(&line[start..i]));
        }
        if i < b.len() && b[i] == b'\t' {
            i += 1;
        }
    }
    out
}

fn parse_scalar(tok: &str) -> Cell {
    if tok == "NULL" {
        return Cell::Null;
    }
    if !tok.contains(['.', 'e', 'E']) {
        if let Ok(i) = tok.parse::<i64>() {
            return Cell::Int(i);
        }
    }
    match tok.parse::<f64>() {
        Ok(f) => Cell::Real(f),
        Err(_) => Cell::Text(tok.to_string()),
    }
}

/// Run a whole batch in one `sqlite3`, splitting the output on [`SENTINEL`].
/// `None` when the batch hit an error, in which case the caller re-runs its
/// cases one at a time to find out which.
fn sqlite_batch(cases: &[Case]) -> Option<Vec<Vec<Vec<Cell>>>> {
    let mut script = String::from(PREAMBLE);
    for c in cases {
        script.push_str(&c.script(true));
        let _ = writeln!(script, "SELECT '{SENTINEL}';");
    }
    let (out, err) = sqlite_raw(&script)?;
    if !err.trim().is_empty() {
        return None;
    }
    let mut all = Vec::with_capacity(cases.len());
    let mut cur = Vec::new();
    for line in out.lines() {
        if line.contains(SENTINEL) {
            all.push(std::mem::take(&mut cur));
        } else {
            cur.push(parse_quote_row(line));
        }
    }
    (all.len() == cases.len()).then_some(all)
}

fn sqlite_one(case: &Case) -> Result<Vec<Vec<Cell>>, String> {
    let mut script = String::from(PREAMBLE);
    script.push_str(&case.script(true));
    let (out, err) = sqlite_raw(&script).ok_or("sqlite3 would not run")?;
    if !err.trim().is_empty() {
        return Err(err.trim().to_string());
    }
    Ok(out.lines().map(parse_quote_row).collect())
}

fn granular_run(case: &Case) -> Result<Vec<Vec<Cell>>, String> {
    let mut s = Session::in_memory();
    for stmt in case.script(false).split(";\n") {
        let stmt = stmt.trim();
        if stmt.is_empty() {
            continue;
        }
        if stmt.starts_with("SELECT") {
            return s
                .query(stmt)
                .map(|rs| rs.to_values().iter().map(|r| r.iter().map(cell_of).collect()).collect())
                .map_err(|e| e.to_string());
        }
        s.execute(stmt).map_err(|e| e.to_string())?;
    }
    Err("no query in the case".into())
}

fn report(case: &Case, got: &[Vec<Cell>], want: &[Vec<Cell>]) -> String {
    let mut s = format!(
        "\nwindow differential mismatch (GRANULAR_WINDOW_SEED={} GRANULAR_WINDOW_CASES=1)\n\n\
         --- granular ---\n{}\n--- sqlite ---\n{}\n",
        case.seed,
        case.script(false),
        case.script(true)
    );
    let _ = writeln!(s, "row | granular | sqlite");
    for i in 0..got.len().max(want.len()) {
        let g = got.get(i).map(|r| format!("{r:?}")).unwrap_or_else(|| "<missing>".into());
        let w = want.get(i).map(|r| format!("{r:?}")).unwrap_or_else(|| "<missing>".into());
        let flag = if g == w { "  " } else { "**" };
        let _ = writeln!(s, "{flag}{i:3} | {g} | {w}");
    }
    s
}

fn compare(case: &Case, want: &[Vec<Cell>]) -> Option<String> {
    let got = match granular_run(case) {
        Ok(g) => g,
        Err(e) => {
            return Some(format!(
                "\ngranular refused a query sqlite3 accepted: {e}\n{}",
                case.script(false)
            ))
        }
    };
    if got.len() != want.len() {
        return Some(report(case, &got, want));
    }
    for (g, w) in got.iter().zip(want) {
        if g.len() != w.len() || !g.iter().zip(w).all(|(a, b)| a.agrees(b)) {
            return Some(report(case, &got, want));
        }
    }
    None
}

// ==================================================================== tests

#[test]
fn window_functions_agree_with_sqlite() {
    if skip_without_sqlite() {
        return;
    }
    let cases: usize = std::env::var("GRANULAR_WINDOW_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_CASES);
    let seed0: u64 = std::env::var("GRANULAR_WINDOW_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0x5EED_0000_0000_0001);

    let mut checked = 0usize;
    let mut n = 0usize;
    while n < cases {
        let batch: Vec<Case> =
            (n..(n + BATCH).min(cases)).map(|i| gen_case(seed0.wrapping_add(i as u64))).collect();
        n += batch.len();
        match sqlite_batch(&batch) {
            Some(results) => {
                for (c, want) in batch.iter().zip(&results) {
                    if let Some(msg) = compare(c, want) {
                        panic!("{msg}");
                    }
                    checked += 1;
                }
            }
            // The batch aborted somewhere. Re-run its cases individually so a
            // query sqlite3 dislikes is attributed instead of poisoning 39
            // innocent neighbours.
            None => {
                for c in &batch {
                    match sqlite_one(c) {
                        Ok(want) => {
                            if let Some(msg) = compare(c, &want) {
                                panic!("{msg}");
                            }
                            checked += 1;
                        }
                        Err(e) => panic!(
                            "\nsqlite3 rejected a generated query -- the generator has left the \
                             dialect intersection:\n{e}\n{}",
                            c.script(true)
                        ),
                    }
                }
            }
        }
    }
    assert_eq!(checked, cases, "every generated case must have been compared");
}

/// The generator's own contract, checked without an oracle.
///
/// A generator that quietly stopped emitting `RANGE` frames, or tied cases, or
/// empty frames, would keep passing forever while testing nothing. This is the
/// cheapest guard against that, and it runs even on a machine with no sqlite3.
#[test]
fn the_generator_covers_the_shapes_it_claims_to() {
    let mut seen_range = 0;
    let mut seen_rows = 0;
    let mut seen_empty_capable = 0;
    let mut seen_unordered = 0;
    let mut seen_partitioned = 0;
    let mut seen_positional = 0;
    let mut seen_single_row = 0;
    for i in 0..2000u64 {
        let c = gen_case(0x5EED_0000_0000_0001u64.wrapping_add(i));
        let q = &c.query;
        seen_range += q.contains("RANGE") as usize;
        seen_rows += q.contains("ROWS") as usize;
        seen_empty_capable +=
            (q.contains("1 FOLLOWING AND 3 FOLLOWING") || q.contains("AND 1 PRECEDING")) as usize;
        seen_unordered += !q.contains("ORDER BY id, ") as usize * q.contains("OVER ()") as usize;
        seen_partitioned += q.contains("PARTITION BY") as usize;
        seen_positional +=
            (q.contains("row_number") || q.contains("lag(") || q.contains("nth_value")) as usize;
        seen_single_row += (c.rows.len() == 1) as usize;
    }
    for (what, n) in [
        ("RANGE frames", seen_range),
        ("ROWS frames", seen_rows),
        ("frames that can be empty", seen_empty_capable),
        ("unordered windows", seen_unordered),
        ("partitioned windows", seen_partitioned),
        ("positional functions", seen_positional),
        ("single-row tables", seen_single_row),
    ] {
        assert!(n > 10, "the generator emitted only {n} cases with {what}");
    }
}

/// The frame semantics the generator is built around, stated directly.
///
/// If this ever disagrees with sqlite3 the generator's whole tied half is
/// meaningless, so it is worth one hand-written case that says what the answer
/// must be rather than only what the two engines must share.
#[test]
fn range_and_rows_differ_exactly_where_the_ties_are() {
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE t (id Int64, k Int64) ENGINE = MergeTree ORDER BY id").unwrap();
    s.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 20), (4, 30)").unwrap();

    let mut col = |sql: &str| -> Vec<i64> {
        s.query(sql)
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .to_values()
            .iter()
            .map(|r| match &r[1] {
                Value::Int(i) => *i,
                Value::UInt(u) => *u as i64,
                other => panic!("{other:?}"),
            })
            .collect()
    };

    // RANGE runs through the last peer, so the two rows tied on k=20 both see
    // 10+20+20. ROWS stops at the row itself, so they see 30 and 50.
    assert_eq!(col("SELECT id, sum(k) OVER (ORDER BY k) FROM t ORDER BY id"), vec![10, 50, 50, 80]);
    assert_eq!(
        col("SELECT id, sum(k) OVER (ORDER BY k ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) \
             FROM t ORDER BY id"),
        vec![10, 30, 50, 80]
    );
    // The same distinction, seen through the ranking functions.
    assert_eq!(col("SELECT id, rank() OVER (ORDER BY k) FROM t ORDER BY id"), vec![1, 2, 2, 4]);
    assert_eq!(col("SELECT id, dense_rank() OVER (ORDER BY k) FROM t ORDER BY id"), vec![1, 2, 2, 3]);
    assert_eq!(col("SELECT id, row_number() OVER (ORDER BY id) FROM t ORDER BY id"), vec![1, 2, 3, 4]);
}
