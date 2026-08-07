//! `Decimal64` aggregates must be exact or refuse. Never fabricate.
//!
//! # What this file pins, and why it exists at the `Session` level
//!
//! `Decimal64` is the one type in this engine whose entire reason to exist is
//! exactness, and four accumulators used to hand back numbers the data does not
//! contain. Every case below was verified wrong against a release build before
//! the fix:
//!
//! ```text
//!   CREATE TABLE d (p Decimal64(2)) ...; INSERT INTO d VALUES (1000000000000.00);
//!   SELECT avg(p), max(p) FROM d;
//!     avg -> 999999999999.999999   fabricated
//!     max -> 1000000000000.00      correct, same column, same row
//!
//!   INSERT INTO ledger VALUES (2000000000000.00), (4000000000000.00);
//!   SELECT avg(m)          FROM ledger;  -> 999999999999.999999   fabricated
//!   SELECT sum(m)/count(*) FROM ledger;  -> error
//! ```
//!
//! Three separate clamps produced those: `avg` and the interpolating
//! `quantile`/`median` clamped at 18 digits *after* promoting the result to
//! scale `max(s,6)`, which collapses the representable magnitude to 10^12
//! whatever the column declared; `sum` clamped at the decimal range, so two
//! rows of 5000000000000000.00 summed to 9999999999999999.99 and
//! `sum(p) = 10000000000000000.00` then evaluated TRUE. A fourth route,
//! `quantileExact`, kept its lanes in an `f64`, so anything past 2^53 came back
//! rounded to a value the column never held.
//!
//! The house defect this file is really guarding against is different, though:
//! seven times a capability has landed complete in `src/` and never been wired
//! to `Session`. So every assertion here goes through the public API -- and
//! `runs_through_the_shipped_binary` goes through the CLI as well, because
//! `Session` is not what a user runs either.
//!
//! # The oracle
//!
//! Mostly not a table of expected strings. `avg(x)` and `sum(x)/count(*)` are
//! the same query, and the expression side has always raised where the
//! aggregate side fabricated, so **they must agree** -- both succeeding with
//! the same value or both failing. That property needs no hand-computed
//! constants, catches a fabricated answer *and* a spuriously refused one, and
//! is exactly the self-contradiction that made the original bug undetectable
//! from SQL. Where a literal answer is checked it is one a human can verify by
//! reading it.

use granular::types::Value;
use granular::Session;

/// A fresh session holding one `Decimal64(scale)` column named `p`, loaded from
/// decimal *string* literals.
///
/// Strings rather than bare numeric literals on purpose: a numeric literal
/// still goes through an `f64` on its way in (a separate, known defect), so a
/// lane past 2^53 would arrive already rounded and the test would be measuring
/// the parser instead of the aggregate. `CAST('...' AS Decimal64(s))` is the
/// one path that is exact today.
fn table(scale: u8, vals: &[&str]) -> Session {
    let mut s = Session::in_memory();
    s.execute(&format!("CREATE TABLE d (p Decimal64({scale})) ENGINE=MergeTree ORDER BY tuple()"))
        .expect("create");
    let rows: Vec<String> = vals.iter().map(|v| format!("('{v}')")).collect();
    s.execute(&format!("INSERT INTO d VALUES {}", rows.join(", "))).expect("insert");
    s
}

/// The one scalar a query returns, rendered exactly, or the error text.
///
/// `render_plain` and not `to_string`: it is the digits the user would see, so
/// a misplaced decimal point fails the comparison rather than hiding inside a
/// numeric equality that rescales.
fn cell(s: &mut Session, sql: &str) -> Result<String, String> {
    match s.query(sql) {
        Ok(rs) => Ok(rs.scalar().unwrap_or(Value::Null).render_plain()),
        Err(e) => Err(e.to_string()),
    }
}

/// Every column of every row, rendered, or the error text.
fn rows(s: &mut Session, sql: &str) -> Result<Vec<Vec<String>>, String> {
    match s.query(sql) {
        Ok(rs) => Ok(rs
            .to_values()
            .iter()
            .map(|r| r.iter().map(|v| v.render_plain()).collect())
            .collect()),
        Err(e) => Err(e.to_string()),
    }
}

/// Magnitudes from 10^11 to 10^18, each at a scale that lets the column hold
/// it, paired with the exact mean of the single row -- which is the value
/// itself, at whatever scale `avg` promotes to.
///
/// `(scale, literal, exact mean or None if it cannot be represented)`.
const SPAN: &[(u8, &str, Option<&str>)] = &[
    // Comfortably inside the promoted range: 10^11 at scale 2 is 10^13 lanes,
    // and 10^17 at scale 6 still fits the 18 digits a Decimal64 holds.
    (2, "100000000000.00", Some("100000000000.000000")),
    (2, "999999999999.99", Some("999999999999.990000")),
    (0, "100000000000", Some("100000000000.000000")),
    // The promotion to scale 6 is what runs out, not the column: every one of
    // these renders back through `max(p)` unharmed.
    (2, "1000000000000.00", None),
    (2, "9999999999999.99", None),
    (2, "1234567890123456.78", None),
    (2, "9999999999999999.99", None),
    (0, "999999999999999999", None),
    (4, "99999999999999.9999", None),
    // Scale 6 and above are already at or past `div_out_scale`, so nothing is
    // promoted and the mean of one row is that row, to the last digit.
    (6, "999999999999.999999", Some("999999999999.999999")),
    (6, "0.000001", Some("0.000001")),
    (9, "999999999.999999999", Some("999999999.999999999")),
    (18, "0.999999999999999999", Some("0.999999999999999999")),
];

/// The headline bug. For one row, `avg(p)` *is* `p` -- it cannot be anything
/// else -- so either the exact value comes back or the query fails. There is no
/// third answer, and 999999999999.999999 for a row holding 1000000000000.00 was
/// the third answer.
#[test]
fn avg_of_a_single_row_is_that_row_or_an_error_never_a_third_number() {
    for &(scale, lit, want) in SPAN {
        let mut s = table(scale, &[lit]);
        // The row really did arrive intact; otherwise this tests the loader.
        assert_eq!(cell(&mut s, "SELECT max(p) FROM d").as_deref(), Ok(lit), "scale {scale}");
        let got = cell(&mut s, "SELECT avg(p) FROM d");
        match want {
            Some(exact) => assert_eq!(got.as_deref(), Ok(exact), "avg of {lit} at scale {scale}"),
            None => {
                let e = got
                    .expect_err(&format!("avg of {lit} at {scale} must not fabricate"));
                assert!(e.contains("Decimal64"), "{lit}: {e}");
            }
        }
    }
}

/// `avg(x)` and `sum(x)/count(*)` are algebraically the same query and the
/// engine used to contradict itself on them: one fabricated, the other raised.
/// They must now succeed together, on the same digits, or fail together.
#[test]
fn avg_and_sum_over_count_never_disagree() {
    let cases: &[(u8, &[&str])] = &[
        (2, &["3.81", "1.19"]),
        (2, &["2000000000000.00", "4000000000000.00"]),
        (2, &["1000000000000.00"]),
        (2, &["999999999999.99", "999999999999.99"]),
        (2, &["-5000000000000000.00", "5000000000000000.00"]),
        (0, &["1", "2", "3"]),
        (6, &["0.000001", "0.000002"]),
        (2, &["9999999999999999.99", "-9999999999999999.99", "0.01"]),
    ];
    for &(scale, vals) in cases {
        let mut s = table(scale, vals);
        let mean = cell(&mut s, "SELECT avg(p) FROM d");
        let ratio = cell(&mut s, "SELECT sum(p) / count(*) FROM d");
        match (&mean, &ratio) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "scale {scale} {vals:?}"),
            (Err(_), Err(_)) => {}
            _ => panic!("scale {scale} {vals:?}: avg -> {mean:?} but sum/count -> {ratio:?}"),
        }
    }
}

/// `sum` clamped at the decimal range, which is worse than it sounds: the
/// clamped total compared *equal* to the true one, so the wrong answer was
/// consistent with itself and invisible from SQL.
#[test]
fn a_sum_past_eighteen_digits_is_refused_rather_than_clamped() {
    let mut s = table(2, &["5000000000000000.00", "5000000000000000.00"]);
    let e = cell(&mut s, "SELECT sum(p) FROM d").expect_err("saturated total must be refused");
    assert!(e.contains("Decimal64(2)"), "{e}");
    // The comparison that used to answer TRUE against a total that was not the
    // total. It must now fail rather than agree with either side.
    for lit in ["10000000000000000.00", "9999999999999999.99"] {
        let q = format!("SELECT sum(p) = CAST('{lit}' AS Decimal64(2)) FROM d");
        assert!(cell(&mut s, &q).is_err(), "sum(p) = {lit} still answers");
    }
    // One unit below the edge is representable and must still be answered --
    // the refusal has to be the range, not a blanket fear of large sums.
    let mut s = table(2, &["9999999999999999.98", "0.01"]);
    assert_eq!(cell(&mut s, "SELECT sum(p) FROM d").as_deref(), Ok("9999999999999999.99"));
    assert_eq!(cell(&mut s, "SELECT sum(p) = max(p) + min(p) FROM d").as_deref(), Ok("true"));
}

/// `median`/`quantile` interpolate, so they divide, so they widen to
/// `max(s,6)` and inherit `avg`'s promoted range -- and used to inherit `avg`'s
/// clamp with it.
#[test]
fn interpolating_quantiles_are_exact_or_refused() {
    // Exact where the promotion fits: the median of 1.19 and 3.81 is 2.50, at
    // six digits because interpolation is a division.
    let mut s = table(2, &["3.81", "1.19"]);
    assert_eq!(cell(&mut s, "SELECT median(p) FROM d").as_deref(), Ok("2.500000"));
    assert_eq!(cell(&mut s, "SELECT quantile(0.5)(p) FROM d").as_deref(), Ok("2.500000"));
    // ... and refused where it does not, on the same rows `avg` is refused on.
    let mut s = table(2, &["1000000000000.00"]);
    for q in ["SELECT median(p) FROM d", "SELECT quantile(0.5)(p) FROM d"] {
        let e = cell(&mut s, q).expect_err(q);
        assert!(e.contains("Decimal64(6)"), "{q}: {e}");
    }
}

/// `quantileExact` returns an element it actually saw, so its answer is always
/// a value the column holds. It kept its lanes in an `f64`, where anything past
/// 2^53 does not survive the round trip: over one row of 1234567890123456.78 it
/// answered 1234567890123456.80 while `min` and `max` answered .78.
#[test]
fn quantile_exact_returns_a_value_the_column_actually_holds() {
    let vals = ["1234567890123456.78", "-9999999999999999.99", "0.00", "9999999999999999.99"];
    let mut s = table(2, &vals);
    for (level, want) in [(0.0, vals[1]), (0.25, vals[2]), (0.5, vals[0]), (0.75, vals[3])] {
        let q = format!("SELECT quantileExact({level})(p) FROM d");
        assert_eq!(cell(&mut s, &q).as_deref(), Ok(want), "{q}");
    }
    // The invariant that makes it "exact", stated against the column rather
    // than against a literal: over one row, every level is that row.
    let mut s = table(2, &["1234567890123456.78"]);
    for level in ["0", "0.5", "1"] {
        let q = format!("SELECT quantileExact({level})(p) = max(p) FROM d");
        assert_eq!(cell(&mut s, &q).as_deref(), Ok("true"), "{q}");
    }
}

/// Every frame `eval_agg` can sweep, on both sides of the range.
///
/// The window spellings reuse the same accumulators and so were wrong the same
/// way, but the check is not simply "agree with `avg(p)`": a window evaluates
/// one mean *per frame*, and a frame can legitimately overflow where the
/// partition's own mean does not. So the in-range fixture asserts the exact
/// answer for every frame, and the out-of-range one asserts that every frame is
/// refused -- a fabricated 999999999999.999999 fails both.
///
/// The frame list is chosen to reach all four of `eval_agg`'s sweeps -- `Whole`,
/// `Forward`, `Backward`, `Refold` -- because each one calls `finish` on its own
/// line, so covering one proves nothing about the other three.
const FRAMES: [&str; 5] = [
    "OVER ()",
    "OVER (ORDER BY p ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)",
    "OVER (ORDER BY p ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)",
    "OVER (ORDER BY p ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)",
    "OVER (ORDER BY p ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING)",
];

#[test]
fn the_window_forms_are_exact_or_refused_frame_by_frame() {
    // Means of every frame over [1.19, 3.81]: each row alone, and both.
    let mut s = table(2, &["3.81", "1.19"]);
    for over in FRAMES {
        let q = format!("SELECT avg(p) {over} AS a FROM d ORDER BY a");
        let got = rows(&mut s, &q).unwrap_or_else(|e| panic!("{q}: {e}"));
        let mut seen: Vec<&str> = got.iter().map(|r| r[0].as_str()).collect();
        seen.dedup();
        assert!(
            seen.iter().all(|v| ["1.190000", "2.500000", "3.810000"].contains(v)),
            "{q}: {seen:?}"
        );
    }
    // Every frame here averages 10^12, which no `Decimal64(6)` holds.
    let mut s = table(2, &["1000000000000.00", "1000000000000.00"]);
    assert!(cell(&mut s, "SELECT avg(p) FROM d").is_err());
    for over in FRAMES {
        let q = format!("SELECT avg(p) {over} AS a FROM d");
        let e = rows(&mut s, &q).expect_err(&q);
        assert!(e.contains("Decimal64(6)"), "{q}: {e}");
    }
    // `sum` over a frame has the same story one range up.
    let mut s = table(2, &["5000000000000000.00", "5000000000000000.00"]);
    for over in FRAMES {
        let q = format!("SELECT sum(p) {over} AS a FROM d");
        assert!(rows(&mut s, &q).is_err(), "{q} still clamps");
    }
}

/// One bad group must fail the query, not vanish into a table of good rows.
/// A per-group `finish` that swallowed its own error would leave 999 correct
/// rows and one fabricated one, which is the hardest kind of wrong to notice.
#[test]
fn an_overflowing_group_fails_the_whole_query() {
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE g (k UInt64, p Decimal64(2)) ENGINE=MergeTree ORDER BY tuple()")
        .expect("create");
    let mut vals: Vec<String> = (0..500).map(|k| format!("({k}, '1.25')")).collect();
    vals.push("(999, '1000000000000.00')".into());
    s.execute(&format!("INSERT INTO g VALUES {}", vals.join(", "))).expect("insert");

    assert_eq!(cell(&mut s, "SELECT avg(p) FROM g WHERE k = 0").as_deref(), Ok("1.250000"));
    let e = cell(&mut s, "SELECT k, avg(p) FROM g GROUP BY k ORDER BY k")
        .expect_err("the bad group must not be swallowed");
    assert!(e.contains("Decimal64(6)"), "{e}");
    // Excluding it puts every remaining group back in range.
    let ok = rows(&mut s, "SELECT k, avg(p) FROM g WHERE k < 500 GROUP BY k ORDER BY k")
        .expect("the good groups still aggregate");
    assert_eq!(ok.len(), 500);
    assert!(ok.iter().all(|r| r[1] == "1.250000"), "{:?}", &ok[..3]);
}

/// `sumIf`/`avgIf` wrap another accumulator and forward its answer, and
/// `DISTINCT` reaches `finish` down a different road again (the per-group
/// seen-set is replayed into a fresh accumulator). Both must carry the refusal
/// rather than unwrap it into a fabricated cell.
#[test]
fn the_if_combinator_and_distinct_forward_the_refusal() {
    let mut s = table(2, &["1000000000000.00", "0.01"]);
    assert_eq!(cell(&mut s, "SELECT avgIf(p, p < 1) FROM d").as_deref(), Ok("0.010000"));
    let e = cell(&mut s, "SELECT avgIf(p, p > 1) FROM d").expect_err("avgIf must not fabricate");
    assert!(e.contains("Decimal64(6)"), "{e}");
    // Both rows are distinct, so DISTINCT changes neither answer here: the mean
    // of 10^12 and a cent is half of 10^12, which still fits scale 6.
    assert_eq!(cell(&mut s, "SELECT avg(DISTINCT p) FROM d").as_deref(), Ok("500000000000.005000"));
    assert_eq!(cell(&mut s, "SELECT sum(DISTINCT p) FROM d").as_deref(), Ok("1000000000000.01"));
    // Two distinct values whose *mean* is past 10^12 is the refusal, reached
    // through the replay path rather than through `update`.
    let mut s = table(2, &["1000000000000.00", "1000000000000.00", "2000000000000.00"]);
    let e = cell(&mut s, "SELECT avg(DISTINCT p) FROM d").expect_err("distinct avg must refuse");
    assert!(e.contains("Decimal64(6)"), "{e}");
    // ...while the DISTINCT sum of the same rows is in range and still answers,
    // and is not the sum of all three.
    assert_eq!(cell(&mut s, "SELECT sum(DISTINCT p) FROM d").as_deref(), Ok("3000000000000.00"));
    assert_eq!(cell(&mut s, "SELECT sum(p) FROM d").as_deref(), Ok("4000000000000.00"));
}

/// The fix must not have bought exactness by making ordinary money queries
/// fail. Everything a ledger actually contains still answers, to the cent.
#[test]
fn ordinary_decimal_aggregation_is_untouched() {
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE t (r String, amt Decimal64(2)) ENGINE=MergeTree ORDER BY tuple()")
        .expect("create");
    s.execute(
        "INSERT INTO t VALUES ('a','19.99'),('a','5.01'),('b','-3.25'),('b','3.25'),('b','100.00')",
    )
    .expect("insert");
    let got = rows(
        &mut s,
        "SELECT r, sum(amt), avg(amt), min(amt), max(amt), median(amt), count(*) \
         FROM t GROUP BY r ORDER BY r",
    )
    .expect("query");
    assert_eq!(
        got,
        vec![
            vec!["a", "25.00", "12.500000", "5.01", "19.99", "12.500000", "2"],
            vec!["b", "100.00", "33.333333", "-3.25", "100.00", "3.250000", "3"],
        ]
    );
    // And the property the type exists for: no f64 anywhere in that path.
    let q = "SELECT sum(amt) = CAST('125.00' AS Decimal64(2)) FROM t";
    assert_eq!(cell(&mut s, q).as_deref(), Ok("true"));
    let mut s = table(1, &["0.1", "0.2"]);
    assert_eq!(cell(&mut s, "SELECT sum(p) FROM d").as_deref(), Ok("0.3"));
}

/// Integer `sum` had the same clamp for the same reason -- `finish` returned a
/// `Value` and could not refuse -- so three `i64::MAX` rows totalled `i64::MAX`.
/// Same policy, and the same reason: a saturated total is indistinguishable
/// from the true one. (SQLite raises here too, which is what lets the
/// differential oracle widen its integer pool.)
#[test]
fn an_integer_sum_past_the_return_type_is_refused_too() {
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE i (v Int64) ENGINE=MergeTree ORDER BY tuple()").expect("create");
    s.execute("INSERT INTO i VALUES (9223372036854775807), (9223372036854775807)")
        .expect("insert");
    let e = cell(&mut s, "SELECT sum(v) FROM i").expect_err("clamped total must be refused");
    assert!(e.contains("Int64"), "{e}");
    // The i128 fold is still the point: an excursion past i64 that comes back
    // is exact, not an error.
    s.execute("INSERT INTO i VALUES (-9223372036854775808), (-9223372036854775808)")
        .expect("insert");
    assert_eq!(cell(&mut s, "SELECT sum(v) FROM i").as_deref(), Ok("-2"));
    // Float sums have a range no total can leave, and must not have acquired a
    // refusal along the way.
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE f (v Float64) ENGINE=MergeTree ORDER BY tuple()").expect("create");
    s.execute("INSERT INTO f VALUES (1e300), (1e300)").expect("insert");
    // Compared as a number: `render_plain` spells a float out in full, and 2e300
    // written out is 301 digits.
    let got = cell(&mut s, "SELECT sum(v) FROM f").expect("float sums still answer");
    assert_eq!(got.parse::<f64>().ok(), Some(2e300), "{got}");
    // Past the range it saturates to `inf`, which is the float answer. It used
    // to be NaN: Neumaier's remainder of `1e308 + 1e308` is `(1e308 - inf) +
    // 1e308`, so `comp` went to -inf and `sum + comp` cancelled to NaN --
    // found by this file, fixed in `SumCore::as_f64`.
    s.execute("INSERT INTO f VALUES (1e308), (1e308), (1e308)").expect("insert");
    assert_eq!(cell(&mut s, "SELECT sum(v) FROM f").as_deref(), Ok("inf"));
    assert_eq!(cell(&mut s, "SELECT avg(v) FROM f").as_deref(), Ok("inf"));
}

/// `Session` is not what a user runs. This is the same fix seen from the far
/// side of the binary, and it is here because the recurring failure in this
/// project is a capability that is complete in `src/` and never reached.
#[test]
fn runs_through_the_shipped_binary() {
    use std::process::Command;
    let sql = "CREATE TABLE d (p Decimal64(2)) ENGINE=MergeTree ORDER BY tuple(); \
               INSERT INTO d VALUES ('1000000000000.00'); \
               SELECT avg(p) FROM d;";
    let out = Command::new(env!("CARGO_BIN_EXE_granular"))
        .args(["-q", sql])
        .output()
        .expect("run the granular binary");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !text.contains("999999999999.999999"),
        "the CLI still reports the fabricated average:\n{text}"
    );
    assert!(text.contains("Decimal64(6)"), "expected a refusal, got:\n{text}");
}
