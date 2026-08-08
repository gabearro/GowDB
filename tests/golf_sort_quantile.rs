//! Contract tests for the two operators the sort/quantile golfing pass moved:
//! `quantile`/`quantileExact`/`median` (which now select instead of sorting)
//! and `ORDER BY` (whose radix path grew nullable keys and whose top-K filter
//! grew a lane fast path).
//!
//! Everything here is a *contract*, not an implementation detail, so the tests
//! stay meaningful if the implementations move again:
//!
//! * a quantile is checked against a reference computed from a sorted copy in
//!   the test, at every percentile from 0 to 100 inclusive, on even and odd row
//!   counts, with and without NULLs;
//! * the two spellings are checked to keep *different* contracts --
//!   `quantileExact` always answers with an element the column holds,
//!   `quantile` is allowed not to and must actually interpolate somewhere;
//! * `ORDER BY` is checked against a reference order over every key type, in
//!   both directions, with NULLS FIRST and NULLS LAST, and for stability;
//! * top-K is checked to equal the head of the full sort, which is the only
//!   definition of "right" a fused `ORDER BY ... LIMIT` has.

use granular::types::Value;
use granular::Session;

// ------------------------------------------------------------------ helpers

fn rows(s: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    s.query(sql)
        .unwrap_or_else(|e| panic!("query failed: {sql}\n  {e}"))
        .to_values()
}

fn scalar(s: &mut Session, sql: &str) -> Value {
    s.query(sql)
        .unwrap_or_else(|e| panic!("query failed: {sql}\n  {e}"))
        .scalar()
        .unwrap_or_else(|| panic!("no scalar result for: {sql}"))
}

fn col0(s: &mut Session, sql: &str) -> Vec<Value> {
    rows(s, sql).into_iter().map(|r| r[0].clone()).collect()
}

/// Ids of a single-column result, in order.
fn ids(s: &mut Session, sql: &str) -> Vec<u64> {
    col0(s, sql)
        .iter()
        .map(|v| v.as_u64().unwrap_or_else(|| panic!("not an id: {v:?}")))
        .collect()
}

/// A table of one column of `ty`, loaded with `vals` (SQL literals, `NULL`
/// allowed) and an `id` that records insertion order.
fn one_col(ty: &str, vals: &[&str]) -> Session {
    let mut s = Session::in_memory();
    s.execute(&format!("CREATE TABLE t (id UInt64, v {ty}) ENGINE = MergeTree ORDER BY id"))
        .unwrap();
    if !vals.is_empty() {
        let tuples: Vec<String> =
            vals.iter().enumerate().map(|(i, v)| format!("({i}, {v})")).collect();
        s.execute(&format!("INSERT INTO t VALUES {}", tuples.join(","))).unwrap();
    }
    s
}

fn as_f64(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not numeric: {v:?}"))
}

/// The reference `quantile` is measured against: linear interpolation between
/// the two neighbouring ranks of the sorted non-NULL values.
fn ref_interpolated(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let pos = p * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - pos.floor())
}

/// The reference `quantileExact` is measured against: the element at rank
/// `floor(p*n)`, clamped to the last one.
fn ref_exact(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    sorted[((p * n as f64).floor() as usize).min(n - 1)]
}

// ------------------------------------------------------------ quantile: ranks

/// Every percentile from 0 to 100 inclusive, on odd *and* even row counts, for
/// both spellings.
///
/// The parity matters: an even count puts the median between two elements,
/// where the interpolating form has to divide and the exact form must not.
#[test]
fn quantile_matches_a_sorted_reference_at_every_percentile() {
    for n in [1usize, 2, 3, 4, 7, 8, 9, 50, 101] {
        // Deliberately out of order on the way in, with repeats, so nothing
        // can pass by accident of insertion order.
        let raw: Vec<i64> = (0..n).map(|i| ((i * 37) % n) as i64 * 3 - 11).collect();
        let lits: Vec<String> = raw.iter().map(|v| v.to_string()).collect();
        let refs: Vec<&str> = lits.iter().map(|s| s.as_str()).collect();
        let mut s = one_col("Int64", &refs);
        let mut sorted: Vec<f64> = raw.iter().map(|&v| v as f64).collect();
        sorted.sort_by(f64::total_cmp);

        for pct in 0..=100 {
            let p = pct as f64 / 100.0;
            let got = as_f64(&scalar(&mut s, &format!("SELECT quantile({p})(v) FROM t")));
            let want = ref_interpolated(&sorted, p);
            assert!((got - want).abs() < 1e-9, "n={n} quantile({p}) = {got}, want {want}");

            let got = as_f64(&scalar(&mut s, &format!("SELECT quantileExact({p})(v) FROM t")));
            assert_eq!(got, ref_exact(&sorted, p), "n={n} quantileExact({p})");
        }

        // `median` is `quantile(0.5)`, not `quantileExact(0.5)`.
        let med = as_f64(&scalar(&mut s, "SELECT median(v) FROM t"));
        assert!((med - ref_interpolated(&sorted, 0.5)).abs() < 1e-9, "n={n} median");
    }
}

/// Floats, including negatives and both zeroes, at every percentile.
#[test]
fn quantile_over_floats_at_every_percentile() {
    let raw: [f64; 12] = [3.5, -1.25, 0.0, 7.75, -0.0, 2.0, -9.5, 4.25, 1.5, 6.0, -3.0, 8.125];
    let lits: Vec<String> = raw.iter().map(|v| format!("{v:?}")).collect();
    let refs: Vec<&str> = lits.iter().map(|s| s.as_str()).collect();
    let mut s = one_col("Float64", &refs);
    let mut sorted = raw.to_vec();
    sorted.sort_by(f64::total_cmp);

    for pct in 0..=100 {
        let p = pct as f64 / 100.0;
        let got = as_f64(&scalar(&mut s, &format!("SELECT quantile({p})(v) FROM t")));
        assert!((got - ref_interpolated(&sorted, p)).abs() < 1e-12, "quantile({p}) = {got}");
        let got = as_f64(&scalar(&mut s, &format!("SELECT quantileExact({p})(v) FROM t")));
        assert_eq!(got, ref_exact(&sorted, p), "quantileExact({p})");
    }
}

/// NULLs are skipped, not counted and not sorted to an end: the answer must be
/// the answer over the non-NULL rows alone, at every percentile.
#[test]
fn quantile_ignores_nulls_at_every_percentile() {
    let mut s = one_col(
        "Nullable(Int64)",
        &["5", "NULL", "1", "9", "NULL", "3", "7", "NULL", "11", "13"],
    );
    let sorted: Vec<f64> = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0];
    for pct in 0..=100 {
        let p = pct as f64 / 100.0;
        let got = as_f64(&scalar(&mut s, &format!("SELECT quantile({p})(v) FROM t")));
        assert!(
            (got - ref_interpolated(&sorted, p)).abs() < 1e-9,
            "quantile({p}) with NULLs = {got}"
        );
        let got = as_f64(&scalar(&mut s, &format!("SELECT quantileExact({p})(v) FROM t")));
        assert_eq!(got, ref_exact(&sorted, p), "quantileExact({p}) with NULLs");
    }
    // All-NULL and empty inputs answer NULL, not 0 and not an error.
    let mut e = one_col("Nullable(Int64)", &["NULL", "NULL"]);
    assert_eq!(scalar(&mut e, "SELECT quantile(0.5)(v) FROM t"), Value::Null);
    assert_eq!(scalar(&mut e, "SELECT quantileExact(0.9)(v) FROM t"), Value::Null);
    let mut z = one_col("Int64", &[]);
    assert_eq!(scalar(&mut z, "SELECT median(v) FROM t"), Value::Null);
}

/// UInt64 lanes above 2^53, where the two spellings differ *by contract*:
/// `quantileExact` hands back the observed lane and `quantile` is a `Float64`
/// that cannot hold it. Pins the direction of that loss.
#[test]
fn quantile_over_wide_unsigned_lanes() {
    let a = 9_007_199_254_740_993u64; // 2^53 + 1, not representable in f64
    let b = 9_007_199_254_740_995u64;
    let mut s = one_col("UInt64", &[&a.to_string(), &b.to_string()]);
    assert_eq!(scalar(&mut s, "SELECT quantileExact(0.0)(v) FROM t").as_f64(), Some(a as f64));
    assert_eq!(scalar(&mut s, "SELECT quantileExact(1.0)(v) FROM t").as_f64(), Some(b as f64));
    let mid = as_f64(&scalar(&mut s, "SELECT median(v) FROM t"));
    assert!(mid >= a as f64 && mid <= b as f64, "median {mid} outside [{a}, {b}]");
}

// ------------------------------------------ quantile: the two distinct contracts

/// The whole point of having both spellings: `quantileExact` answers with an
/// element of the column, `quantile` interpolates and may not.
#[test]
fn the_interpolating_and_exact_forms_keep_distinct_contracts() {
    // Two values ten apart, so the halfway point is neither of them.
    let mut s = one_col("Int64", &["10", "20"]);
    assert_eq!(as_f64(&scalar(&mut s, "SELECT quantile(0.5)(v) FROM t")), 15.0);
    assert_eq!(as_f64(&scalar(&mut s, "SELECT median(v) FROM t")), 15.0);
    // `floor(0.5 * 2) = 1`, so the exact form is the upper of the two.
    assert_eq!(as_f64(&scalar(&mut s, "SELECT quantileExact(0.5)(v) FROM t")), 20.0);

    // Over 1000 rows with no repeats, every `quantileExact` answer must be a
    // value the column holds; `quantile` must land off that grid somewhere, or
    // it is not interpolating at all.
    let lits: Vec<String> = (0..1000).map(|i| (i * 7).to_string()).collect();
    let refs: Vec<&str> = lits.iter().map(|s| s.as_str()).collect();
    let mut s = one_col("Int64", &refs);
    let mut off_grid = 0;
    for pct in 0..=100 {
        let p = pct as f64 / 100.0;
        let e = as_f64(&scalar(&mut s, &format!("SELECT quantileExact({p})(v) FROM t")));
        assert_eq!(e % 7.0, 0.0, "quantileExact({p}) = {e} is not an observed value");
        let q = as_f64(&scalar(&mut s, &format!("SELECT quantile({p})(v) FROM t")));
        assert!((0.0..=6993.0).contains(&q), "quantile({p}) = {q} out of range");
        if q % 7.0 != 0.0 {
            off_grid += 1;
        }
    }
    assert!(
        off_grid > 0,
        "quantile never left the observed grid; it is behaving like quantileExact"
    );

    // Both agree at the ends: p=0 is the minimum, p=1 the maximum, for either
    // spelling and whatever the row count.
    for n in [1usize, 2, 5, 6] {
        let lits: Vec<String> = (0..n).map(|i| ((n - i) * 4).to_string()).collect();
        let refs: Vec<&str> = lits.iter().map(|s| s.as_str()).collect();
        let mut s = one_col("Int64", &refs);
        for f in ["quantile", "quantileExact"] {
            assert_eq!(
                as_f64(&scalar(&mut s, &format!("SELECT {f}(0.0)(v) FROM t"))),
                4.0,
                "{f}(0) over n={n}"
            );
            assert_eq!(
                as_f64(&scalar(&mut s, &format!("SELECT {f}(1.0)(v) FROM t"))),
                (n * 4) as f64,
                "{f}(1) over n={n}"
            );
        }
    }
}

/// Decimals split by whether the aggregate divides: the exact form keeps the
/// column's scale and hands back a stored lane, the interpolating form widens
/// like `avg` does.
#[test]
fn quantile_over_decimals_splits_by_whether_it_divides() {
    let mut s = one_col("Decimal64(2)", &["1.19", "3.81"]);
    // Interpolating: (1.19 + 3.81) / 2 = 2.50, at the widened divide scale.
    assert_eq!(scalar(&mut s, "SELECT median(v) FROM t"), Value::Decimal(2_500_000, 6));
    assert_eq!(scalar(&mut s, "SELECT quantile(0.5)(v) FROM t"), Value::Decimal(2_500_000, 6));
    // Exact: an observed lane, at the column's own scale.
    assert_eq!(scalar(&mut s, "SELECT quantileExact(0.5)(v) FROM t"), Value::Decimal(381, 2));
    assert_eq!(scalar(&mut s, "SELECT quantileExact(0.0)(v) FROM t"), Value::Decimal(119, 2));
    assert_eq!(scalar(&mut s, "SELECT quantileExact(1.0)(v) FROM t"), Value::Decimal(381, 2));

    // Negative lanes select as signed integers, not as unsigned words: an
    // unsigned compare would rank every negative price above every positive.
    let mut s = one_col("Decimal64(2)", &["-5.00", "-1.00", "2.00", "-3.00", "4.00"]);
    assert_eq!(scalar(&mut s, "SELECT quantileExact(0.0)(v) FROM t"), Value::Decimal(-500, 2));
    assert_eq!(scalar(&mut s, "SELECT quantileExact(1.0)(v) FROM t"), Value::Decimal(400, 2));
    assert_eq!(scalar(&mut s, "SELECT quantileExact(0.5)(v) FROM t"), Value::Decimal(-100, 2));
    assert_eq!(scalar(&mut s, "SELECT median(v) FROM t"), Value::Decimal(-1_000_000, 6));

    // A lane past the f64 mantissa survives the aggregate, which is why the
    // accumulator holds integers rather than float bits. Compared against
    // `max` over the same rows rather than against the literal: the decimal
    // *literal* still rounds through an f64 on the way in (a separate, known
    // defect), and this test is about what the aggregate does to a lane the
    // column already holds, whatever that lane turned out to be.
    let mut s = one_col("Decimal64(2)", &["1234567890123456.78", "1234567890123456.79"]);
    for p in ["0.0", "0.5", "1.0"] {
        let q = scalar(&mut s, &format!("SELECT quantileExact({p})(v) FROM t"));
        let extreme = scalar(
            &mut s,
            if p == "0.0" { "SELECT min(v) FROM t" } else { "SELECT max(v) FROM t" },
        );
        assert_eq!(q, extreme, "quantileExact({p}) lost a lane min/max kept");
        assert!(matches!(q, Value::Decimal(u, 2) if u > (1i64 << 53)), "{q:?}");
    }

    // Every percentile over an odd count, against the stored lanes.
    let lanes: [i64; 7] = [-450, -120, 0, 75, 310, 999, 1000];
    let lits: Vec<String> =
        lanes.iter().map(|l| format!("{}.{:02}", l / 100, (l % 100).abs())).collect();
    let refs: Vec<&str> = lits.iter().map(|s| s.as_str()).collect();
    let mut s = one_col("Decimal64(2)", &refs);
    let mut sorted = lanes;
    sorted.sort_unstable();
    for pct in 0..=100 {
        let p = pct as f64 / 100.0;
        let want = sorted[((p * 7.0).floor() as usize).min(6)];
        assert_eq!(
            scalar(&mut s, &format!("SELECT quantileExact({p})(v) FROM t")),
            Value::Decimal(want, 2),
            "quantileExact({p}) over decimal lanes"
        );
        // Interpolating stays inside the observed range at every level.
        let q = as_f64(&scalar(&mut s, &format!("SELECT quantile({p})(v) FROM t")));
        assert!((-4.5..=10.0).contains(&q), "quantile({p}) = {q} left the column's range");
    }
}

/// `finish` must be callable more than once, and it now works in the
/// accumulator's own buffer rather than a copy. A second read of one group must
/// not answer NULL, or a different number.
#[test]
fn a_quantile_answers_the_same_twice_in_one_query() {
    let mut s = one_col("Int64", &["4", "8", "15", "16", "23", "42"]);
    let r = rows(
        &mut s,
        "SELECT quantile(0.25)(v), quantile(0.25)(v), quantileExact(0.5)(v), \
                quantileExact(0.5)(v), median(v), median(v) FROM t",
    );
    assert_eq!(r.len(), 1);
    assert_eq!(r[0][0], r[0][1], "two quantile(0.25) in one select disagree");
    assert_eq!(r[0][2], r[0][3], "two quantileExact(0.5) in one select disagree");
    assert_eq!(r[0][4], r[0][5], "two median() in one select disagree");
    // And across statements, on the same session.
    let first = scalar(&mut s, "SELECT quantile(0.75)(v) FROM t");
    let second = scalar(&mut s, "SELECT quantile(0.75)(v) FROM t");
    assert_eq!(first, second);
}

/// Grouped, where there is one accumulator per group and the partial states are
/// merged across workers before anything is read.
#[test]
fn quantile_per_group_matches_the_same_quantile_over_the_group_alone() {
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE t (id UInt64, g UInt32, v Int64) ENGINE = MergeTree ORDER BY id")
        .unwrap();
    let tuples: Vec<String> =
        (0..20_000u64).map(|i| format!("({i}, {}, {})", i % 4, (i * 7919) % 1000)).collect();
    for chunk in tuples.chunks(10_000) {
        s.execute(&format!("INSERT INTO t VALUES {}", chunk.join(","))).unwrap();
    }

    let grouped = rows(
        &mut s,
        "SELECT g, quantile(0.9)(v), quantileExact(0.9)(v), median(v) FROM t GROUP BY g ORDER BY g",
    );
    assert_eq!(grouped.len(), 4);
    for (i, r) in grouped.iter().enumerate() {
        let alone = rows(
            &mut s,
            &format!(
                "SELECT quantile(0.9)(v), quantileExact(0.9)(v), median(v) FROM t WHERE g = {i}"
            ),
        );
        assert_eq!(r[1], alone[0][0], "group {i} quantile");
        assert_eq!(r[2], alone[0][1], "group {i} quantileExact");
        assert_eq!(r[3], alone[0][2], "group {i} median");
    }
}

/// A level outside `[0, 1]` is refused rather than clamped -- clamping is what
/// this used to do.
#[test]
fn quantile_still_validates_its_level() {
    let mut s = one_col("Int64", &["1", "2", "3"]);
    for bad in ["-0.1", "1.5", "2"] {
        assert!(
            s.query(&format!("SELECT quantile({bad})(v) FROM t")).is_err(),
            "quantile({bad}) was accepted"
        );
    }
    assert!(s.query("SELECT quantile(0.5)(id) FROM t").is_ok());
}

// ------------------------------------------------------------------- ORDER BY

/// The descending order of the same keys: values reversed, ties left in
/// insertion order. Derived from the engine's own ascending answer, so what it
/// pins is *stability*, not the key type's comparator.
fn reverse_keeping_ties(s: &mut Session, asc_ids: &[u64]) -> Vec<u64> {
    let keys: Vec<Value> = asc_ids
        .iter()
        .map(|id| scalar(s, &format!("SELECT v FROM t WHERE id = {id}")))
        .collect();
    let mut out: Vec<u64> = Vec::with_capacity(asc_ids.len());
    let mut hi = asc_ids.len();
    while hi > 0 {
        let mut lo = hi - 1;
        while lo > 0 && keys[lo - 1] == keys[hi - 1] {
            lo -= 1;
        }
        out.extend_from_slice(&asc_ids[lo..hi]);
        hi = lo;
    }
    out
}

/// Every key type, both directions, both NULL placements, against a reference
/// order spelled out here -- and the fused top-K against every prefix of it.
///
/// Each case has repeats (so stability has something to prove), NULLs (so
/// placement has something to prove) and, where the type allows, a value that
/// orders differently as a signed integer than as an unsigned word.
#[test]
fn order_by_every_key_type_in_both_directions_with_both_null_placements() {
    // (type, literals in insertion order, ids in ascending key order excluding
    // NULLs, ids that are NULL).
    let cases: &[(&str, &[&str], &[u64], &[u64])] = &[
        (
            "Nullable(Int64)",
            &["3", "NULL", "-7", "3", "0", "NULL", "-7", "9223372036854775807"],
            &[2, 6, 4, 0, 3, 7],
            &[1, 5],
        ),
        (
            "Nullable(UInt64)",
            &["3", "NULL", "18446744073709551615", "3", "0", "9223372036854775808"],
            &[4, 0, 3, 5, 2],
            &[1],
        ),
        (
            "Nullable(Float64)",
            // -0.0 and 0.0 compare equal, so they hold insertion order.
            &["1.5", "NULL", "-2.5", "1.5", "0.0", "-0.0", "1e308"],
            &[2, 4, 5, 0, 3, 6],
            &[1],
        ),
        (
            "Nullable(String)",
            &["'pear'", "NULL", "'apple'", "'pear'", "''", "'Pear'"],
            &[4, 5, 2, 0, 3],
            &[1],
        ),
        (
            "Nullable(Decimal64(2))",
            &["1.50", "NULL", "-2.50", "1.50", "0.00"],
            &[2, 4, 0, 3],
            &[1],
        ),
        (
            "Nullable(DateTime)",
            &["1700000000", "NULL", "1600000000", "1700000000", "0"],
            &[4, 2, 0, 3],
            &[1],
        ),
        (
            "Nullable(Date)",
            &["'2024-01-02'", "NULL", "'1970-01-01'", "'2024-01-02'", "'2000-06-15'"],
            &[2, 4, 0, 3],
            &[1],
        ),
        ("Nullable(Bool)", &["true", "NULL", "false", "true", "false"], &[2, 4, 0, 3], &[1]),
        // The non-nullable shapes run the same code with no mask at all.
        ("Int64", &["3", "-7", "3", "0", "-7"], &[1, 4, 3, 0, 2], &[]),
        ("String", &["'b'", "'a'", "'b'", "'A'"], &[3, 1, 0, 2], &[]),
        ("Float64", &["1.5", "-2.5", "1.5", "0.0"], &[1, 3, 0, 2], &[]),
        ("UInt64", &["3", "18446744073709551615", "3", "0"], &[3, 0, 2, 1], &[]),
    ];

    for (ty, lits, asc_ids, null_ids) in cases {
        let mut s = one_col(ty, lits);
        let desc_ids = reverse_keeping_ties(&mut s, asc_ids);

        for (dir, order) in [("ASC", asc_ids.to_vec()), ("DESC", desc_ids)] {
            for nulls_first in [true, false] {
                let clause = if nulls_first { "NULLS FIRST" } else { "NULLS LAST" };
                let mut want: Vec<u64> = Vec::new();
                if nulls_first {
                    want.extend_from_slice(null_ids);
                }
                want.extend(order.iter().copied());
                if !nulls_first {
                    want.extend_from_slice(null_ids);
                }
                let got = ids(&mut s, &format!("SELECT id FROM t ORDER BY v {dir} {clause}"));
                assert_eq!(got, want, "{ty} ORDER BY v {dir} {clause}");

                // Top-K is the head of that same order, at every prefix length.
                for k in 0..=want.len() {
                    let got = ids(
                        &mut s,
                        &format!("SELECT id FROM t ORDER BY v {dir} {clause} LIMIT {k}"),
                    );
                    assert_eq!(got, want[..k], "{ty} ORDER BY v {dir} {clause} LIMIT {k}");
                }
            }
        }
    }
}

/// Stability at a size that goes through the block machinery and the parallel
/// exchange, on keys with heavy repeats: within one key value the ids must come
/// out in insertion order, ascending and descending alike.
#[test]
fn a_sort_with_many_ties_stays_stable_through_the_exchange() {
    let n = 60_000u64;
    let mut s = Session::in_memory();
    s.execute(
        "CREATE TABLE t (id UInt64, k UInt32, nk Nullable(Int64), sk String) \
         ENGINE = MergeTree ORDER BY id",
    )
    .unwrap();
    let tuples: Vec<String> = (0..n)
        .map(|i| {
            let nk = if i % 7 == 0 { "NULL".to_string() } else { (i % 23).to_string() };
            format!("({i}, {}, {nk}, '{}')", i % 16, (b'a' + (i % 5) as u8) as char)
        })
        .collect();
    for chunk in tuples.chunks(10_000) {
        s.execute(&format!("INSERT INTO t VALUES {}", chunk.join(","))).unwrap();
    }

    for (key, dir) in [
        ("k", "ASC"),
        ("k", "DESC"),
        ("nk", "ASC NULLS FIRST"),
        ("nk", "ASC NULLS LAST"),
        ("nk", "DESC NULLS FIRST"),
        ("nk", "DESC NULLS LAST"),
        ("sk", "ASC"),
        ("sk", "DESC"),
    ] {
        let out = rows(&mut s, &format!("SELECT {key}, id FROM t ORDER BY {key} {dir}"));
        assert_eq!(out.len() as u64, n, "{key} {dir} lost rows");
        let mut prev: Option<(Value, u64)> = None;
        for r in &out {
            let id = r[1].as_u64().unwrap();
            if let Some((pk, pid)) = &prev {
                if pk == &r[0] {
                    assert!(id > *pid, "{key} {dir} is not stable: {pid} then {id}");
                }
            }
            prev = Some((r[0].clone(), id));
        }
        // And the fused top-K is the head of exactly that order.
        for k in [1usize, 5, 100, 9_000] {
            let head = ids(&mut s, &format!("SELECT id FROM t ORDER BY {key} {dir} LIMIT {k}"));
            let want: Vec<u64> = out[..k].iter().map(|r| r[1].as_u64().unwrap()).collect();
            assert_eq!(head, want, "{key} {dir} LIMIT {k}");
        }
    }
}

/// The top-K filter drops any block that cannot beat the current k-th best. A
/// key whose winners arrive *last* catches a threshold compared the wrong way
/// round; a key whose values are all equal catches a non-strict compare, which
/// would keep every row and defeat the bound.
#[test]
fn top_k_agrees_with_a_full_sort_whichever_end_the_winners_arrive_at() {
    let n = 40_000i64;
    for shape in ["ascending", "descending", "constant", "v-shaped", "two-valued"] {
        let mut s = Session::in_memory();
        s.execute("CREATE TABLE t (id UInt64, v Int64, f Float64) ENGINE = MergeTree ORDER BY id")
            .unwrap();
        let tuples: Vec<String> = (0..n)
            .map(|i| {
                let v = match shape {
                    "ascending" => i,
                    "descending" => n - i,
                    "constant" => 7,
                    "v-shaped" => (n / 2 - i).abs(),
                    _ => i % 2,
                };
                format!("({i}, {v}, {}.5)", -v)
            })
            .collect();
        for chunk in tuples.chunks(10_000) {
            s.execute(&format!("INSERT INTO t VALUES {}", chunk.join(","))).unwrap();
        }
        for key in ["v", "f"] {
            for dir in ["ASC", "DESC"] {
                let full = ids(&mut s, &format!("SELECT id FROM t ORDER BY {key} {dir}"));
                assert_eq!(full.len() as i64, n);
                for k in [1usize, 2, 5, 17, 1000, 8193, 12_289] {
                    let head =
                        ids(&mut s, &format!("SELECT id FROM t ORDER BY {key} {dir} LIMIT {k}"));
                    assert_eq!(head, full[..k], "{shape} {key} {dir} LIMIT {k}");
                }
            }
        }
    }
}

/// Several keys, mixed directions and mixed NULL placement -- the shape with no
/// single lane, which keeps taking the comparison path and must keep agreeing
/// with its own top-K.
#[test]
fn multi_key_sorts_and_their_top_k_agree() {
    let mut s = Session::in_memory();
    s.execute(
        "CREATE TABLE t (id UInt64, a Nullable(String), b Nullable(Int64)) \
         ENGINE = MergeTree ORDER BY id",
    )
    .unwrap();
    let tuples: Vec<String> = (0..5_000u64)
        .map(|i| {
            let a = if i % 11 == 0 { "NULL".into() } else { format!("'g{}'", i % 6) };
            let b = if i % 7 == 0 { "NULL".into() } else { (i % 13).to_string() };
            format!("({i}, {a}, {b})")
        })
        .collect();
    s.execute(&format!("INSERT INTO t VALUES {}", tuples.join(","))).unwrap();

    for clause in [
        "a ASC, b DESC",
        "a DESC NULLS FIRST, b ASC NULLS LAST",
        "b ASC NULLS FIRST, a DESC",
        "b DESC, id ASC",
    ] {
        let full = ids(&mut s, &format!("SELECT id FROM t ORDER BY {clause}"));
        assert_eq!(full.len(), 5_000);
        for k in [1usize, 3, 100, 4_000] {
            let head = ids(&mut s, &format!("SELECT id FROM t ORDER BY {clause} LIMIT {k}"));
            assert_eq!(head, full[..k], "ORDER BY {clause} LIMIT {k}");
        }
    }
}

/// An expression key, which is materialized per block rather than borrowed --
/// the top-K filter has to reach it through the same path the sort does.
#[test]
fn expression_keys_sort_and_top_k_alike() {
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE t (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id").unwrap();
    let tuples: Vec<String> =
        (0..20_000u64).map(|i| format!("({i}, {})", (i * 31) % 977)).collect();
    for chunk in tuples.chunks(10_000) {
        s.execute(&format!("INSERT INTO t VALUES {}", chunk.join(","))).unwrap();
    }
    for clause in ["v % 10 ASC", "v % 10 DESC", "-v ASC", "v * 2 DESC, id ASC"] {
        let full = ids(&mut s, &format!("SELECT id FROM t ORDER BY {clause}"));
        assert_eq!(full.len(), 20_000);
        for k in [1usize, 4, 9_000] {
            let head = ids(&mut s, &format!("SELECT id FROM t ORDER BY {clause} LIMIT {k}"));
            assert_eq!(head, full[..k], "ORDER BY {clause} LIMIT {k}");
        }
    }
}
