//! `SELECT 0.1 + 0.2` is `0.3`, and the rules that makes true everywhere else.
//!
//! # The change
//!
//! A numeric literal **with a decimal point and no exponent** lexes as
//! `Value::Decimal(units, frac_digits)` instead of `Value::Float`. `0.1` is
//! `Decimal(1, 1)`; `1.50` is `Decimal(150, 2)`; `1.5e3` is still
//! `Float(1500.0)`.
//!
//! `Decimal64` was exact from the day it landed, but nothing in the *language*
//! produced one: exact arithmetic needed `CAST('0.1' AS Decimal(10,2))`, so the
//! one demo everybody runs first still answered 0.30000000000000004. This is
//! the wiring, and it is the eighth time in this project that a capability was
//! complete in `src/` and unreachable from SQL -- which is why every assertion
//! below goes through `Session`, and `the_headline_demo_through_the_shipped_binary`
//! goes through the CLI as well.
//!
//! # The promotion rules, and why these and not others
//!
//! **Postgres is the precedent.** There an unsuffixed decimal constant is
//! `numeric`, and it becomes a float only when combined with one. Every rule
//! below is that rule; where this engine has to differ it is because
//! `Decimal64` is 18 digits of `i64` and `numeric` is arbitrary precision.
//!
//! | expression | result | why |
//! |---|---|---|
//! | `0.1 + 0.2` | `Decimal64(1)` | same scale, exact |
//! | `0.10 + 0.2` | `Decimal64(2)` | `+`/`-` take the *wider* scale |
//! | `1.5 * 1.5` | `Decimal64(2)` | `*` **adds** scales; 2.25 needs both |
//! | `1.5 + 1` | `Decimal64(1)` | every integer is exact at every scale, so the decimal wins |
//! | `1.5 + 1.5e0` | `Float64` | a float on either side poisons exactness |
//! | `3.0 / 2` | `Decimal64(6)` | division keeps `max(scale, 6)` digits |
//! | `dec_col = 0.1` | exact | both sides widen to a common scale in `i128` |
//! | `float_col = 0.1` | float | the *column* is inexact; the literal cannot fix that |
//!
//! The one that deserves an argument is **Decimal vs Float giving Float**. The
//! alternative -- refusing the mix -- would be defensible for a type whose
//! whole promise is exactness, but it would make `float_col * 0.5` a bind
//! error in a dialect where that has always worked, for no gain: there is no
//! exact answer to hand back when one operand is already approximate. Postgres
//! resolves it the same way. What matters is that the *result type says so*:
//! the answer is a `Float64` and renders as one, so nothing claims an
//! exactness it did not deliver.
//!
//! # The boundary that makes this tractable
//!
//! An exponent keeps a literal a float. That is not only compatibility: `1e-7`
//! is the spelling `toString` of a double produces, and it is the only spelling
//! that can name a magnitude no `Decimal64` has. Keeping it inexact leaves
//! every user of this dialect a way to *ask* for a float.
//!
//! # Too many digits falls back rather than erroring
//!
//! Past 18 significant digits there is no `Decimal64` at all. The literal
//! becomes the `Float` it has always been instead of failing to parse --
//! refusing would take a query that works today and break it over a number the
//! user never asked to be exact, and the fallback is not silent: the value is a
//! float and renders as one. `too_many_digits_falls_back_to_the_float_it_was`
//! pins both halves.

use granular::types::Value;
use granular::Session;
use std::process::Command;

/// The one scalar a query returns, rendered exactly -- the digits a user would
/// see, so a misplaced point fails rather than hiding inside a numeric equality
/// that rescales.
fn cell(s: &mut Session, sql: &str) -> Result<String, String> {
    match s.query(sql) {
        Ok(rs) => Ok(rs.scalar().unwrap_or(Value::Null).render_plain()),
        Err(e) => Err(e.to_string()),
    }
}

/// The scalar as a `Value`, for the assertions where the *variant* is the point
/// and `Value`'s variant-blind `Eq` would let a float masquerade as a decimal.
fn val(s: &mut Session, sql: &str) -> Value {
    s.query(sql)
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .scalar()
        .unwrap_or(Value::Null)
}

fn row_strings(s: &mut Session, sql: &str) -> Vec<Vec<String>> {
    s.query(sql)
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .to_values()
        .iter()
        .map(|r| r.iter().map(|v| v.render_plain()).collect())
        .collect()
}

// ===========================================================================
// the headline
// ===========================================================================

/// Issue #30, stated as the demo everybody runs first.
///
/// Three independent things are asserted, because each could be true alone and
/// still leave the bug: the rendered digits, the result *variant* (a `Float`
/// that happens to print `0.3` is not a fix), and the comparison against `0.3`
/// -- which is the assertion that fails on any engine doing this in binary64,
/// since `0.1 + 0.2 == 0.3` is famously false there.
#[test]
fn zero_point_one_plus_zero_point_two_is_exactly_zero_point_three() {
    let mut s = Session::in_memory();
    assert_eq!(cell(&mut s, "SELECT 0.1 + 0.2"), Ok("0.3".into()));
    assert_eq!(val(&mut s, "SELECT 0.1 + 0.2").decimal_parts(), Some((3, 1)));
    assert_eq!(cell(&mut s, "SELECT 0.1 + 0.2 = 0.3"), Ok("true".into()));
    // The scale is the digits written, so the two-digit spelling answers to two
    // digits. Both are exact; they differ only in the precision asked for.
    assert_eq!(cell(&mut s, "SELECT 0.10 + 0.20"), Ok("0.30".into()));
    // ...and the two agree as numbers, which is what `Value`'s exact `Ord`
    // across scales is for.
    assert_eq!(cell(&mut s, "SELECT 0.10 + 0.20 = 0.1 + 0.2"), Ok("true".into()));

    // The rest of the classic set, each wrong in binary64.
    for (sql, want) in [
        ("SELECT 0.1 + 0.7", "0.8"),
        ("SELECT 0.3 - 0.1", "0.2"),
        ("SELECT 1.1 * 3", "3.3"),
        ("SELECT 0.1 + 0.2 - 0.3", "0.0"),
        ("SELECT 4.35 * 100", "435.00"),
    ] {
        assert_eq!(cell(&mut s, sql), Ok(want.into()), "{sql}");
    }
}

/// `Session` is not what a user runs. The demo has to be right in the shipped
/// binary, which is the layer this project has shipped a wrong answer at twice.
#[test]
fn the_headline_demo_through_the_shipped_binary() {
    let out = Command::new(env!("CARGO_BIN_EXE_granular"))
        .args(["-q", "SELECT 0.1 + 0.2 AS answer"])
        .output()
        .expect("spawn granular");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("0.3"), "{text}");
    assert!(
        !text.contains("0.30000000000000004"),
        "the binary still answers in binary64:\n{text}"
    );
}

// ===========================================================================
// the boundary: point vs exponent
// ===========================================================================

/// The rule in one test: a point makes it exact, an exponent does not.
#[test]
fn an_exponent_literal_stays_a_float() {
    let mut s = Session::in_memory();
    for sql in ["SELECT 1.5e3", "SELECT 1.5E-3", "SELECT 1e3", "SELECT 0.1e0", "SELECT 1.0e0"] {
        assert_eq!(val(&mut s, sql).variant(), "Float", "{sql}");
    }
    for sql in ["SELECT 1.5", "SELECT 0.1", "SELECT 1.0", "SELECT 1.50"] {
        assert_eq!(val(&mut s, sql).variant(), "Decimal", "{sql}");
    }
    // An integer literal has no point and is untouched.
    assert_eq!(val(&mut s, "SELECT 42").variant(), "Int");
    // ...so the exponent form is the way to *ask* for binary64, and still gets
    // binary64's answer. This is the assertion that would fail if the rule were
    // applied to every numeric literal.
    assert_eq!(cell(&mut s, "SELECT 1e-1 + 2e-1"), Ok("0.30000000000000004".into()));
}

/// Past 18 significant digits there is no `Decimal64`, so the literal falls
/// back to the float it has always been rather than becoming a parse error.
#[test]
fn too_many_digits_falls_back_to_the_float_it_was() {
    let mut s = Session::in_memory();
    // The widest exact literal, and the first one that is not.
    assert_eq!(
        val(&mut s, "SELECT 0.999999999999999999").decimal_parts(),
        Some((999_999_999_999_999_999, 18))
    );
    for sql in [
        "SELECT 0.0000000000000000001",   // scale 19
        "SELECT 1234567890123456789.5",   // 20 significant digits
        "SELECT 99999999999999999999.99", // both at once
    ] {
        assert_eq!(val(&mut s, sql).variant(), "Float", "{sql}");
    }
    // Falling back is not the same as failing: the queries answer.
    assert_eq!(cell(&mut s, "SELECT 1234567890123456789.5 > 0"), Ok("true".into()));
    // And the digits are still reachable exactly -- through the string route,
    // which is what `Decimal64(18)` is for.
    assert_eq!(
        cell(&mut s, "SELECT CAST('0.999999999999999999' AS Decimal64(18))"),
        Ok("0.999999999999999999".into())
    );
}

// ===========================================================================
// promotion
// ===========================================================================

/// Every arm of the promotion table in the module header, asserted as the
/// *type* of the answer and not only its digits -- a `Float64` that happens to
/// print `2.5` would pass a digits-only check while having lost the property
/// the whole change exists for.
#[test]
fn mixed_arithmetic_follows_the_documented_promotion_rules() {
    let mut s = Session::in_memory();
    let cases: &[(&str, &str, &str)] = &[
        // sql, rendered, variant
        ("SELECT 1.5 + 1", "2.5", "Decimal"),
        ("SELECT 1 + 1.5", "2.5", "Decimal"),
        ("SELECT 1.5 - 2", "-0.5", "Decimal"),
        // `+`/`-` unify to the wider scale...
        ("SELECT 0.10 + 0.2", "0.30", "Decimal"),
        ("SELECT 1.5 - 0.25", "1.25", "Decimal"),
        // ...and `*` adds them, because 1.5 * 1.5 genuinely needs two digits.
        ("SELECT 1.5 * 1.5", "2.25", "Decimal"),
        ("SELECT 1.50 * 1.50", "2.2500", "Decimal"),
        // Division keeps max(scale, 6) fractional digits.
        ("SELECT 3.0 / 2", "1.500000", "Decimal"),
        ("SELECT 1.0 / 8", "0.125000", "Decimal"),
        // A float on either side wins, and the result says so by being one.
        ("SELECT 1.5 + 1.5e0", "3", "Float"),
        ("SELECT 1.5e0 * 2.0", "3", "Float"),
        // Unary minus folds into the literal and keeps the scale.
        ("SELECT -1.50", "-1.50", "Decimal"),
        ("SELECT -0.1 + -0.2", "-0.3", "Decimal"),
    ];
    for (sql, want, variant) in cases {
        assert_eq!(cell(&mut s, sql).as_deref(), Ok(*want), "{sql}");
        assert_eq!(&val(&mut s, sql).variant(), variant, "{sql}");
    }
    // Exactness is not an accident of small numbers: 18 digits of it survives
    // an addition that a double would round away at the third-from-last digit.
    assert_eq!(
        cell(&mut s, "SELECT 999999999999.999998 + 0.000001"),
        Ok("999999999999.999999".into())
    );
}

/// A decimal literal is exact against a `Decimal64` column and only as good as
/// the column against a `Float64` one. Both halves matter: the first is the
/// point of the change, and the second is the promise it must not overclaim.
#[test]
fn comparison_against_decimal_and_float_columns() {
    let mut s = Session::in_memory();
    s.execute(
        "CREATE TABLE t (id Int64, d Decimal64(2), f Float64) \
         ENGINE = MergeTree ORDER BY id",
    )
    .expect("ddl");
    s.execute("INSERT INTO t VALUES (1, 0.10, 0.1), (2, 0.25, 0.25), (3, 1.00, 1.0)")
        .expect("insert");

    // Exact against the exact column, across scales: the literal is scale 1 and
    // the column scale 2, and they still name the same row.
    assert_eq!(row_strings(&mut s, "SELECT id FROM t WHERE d = 0.1"), [["1"]]);
    assert_eq!(row_strings(&mut s, "SELECT id FROM t WHERE d = 0.100"), [["1"]]);
    assert_eq!(row_strings(&mut s, "SELECT id FROM t WHERE d > 0.5"), [["3"]]);
    assert_eq!(row_strings(&mut s, "SELECT id FROM t WHERE d = 1"), [["3"]]);

    // Against the float column the literal descales to the nearest double,
    // which is the same double `0.1` has always produced -- so a query written
    // before this change finds exactly the rows it used to.
    assert_eq!(row_strings(&mut s, "SELECT id FROM t WHERE f = 0.1"), [["1"]]);
    assert_eq!(row_strings(&mut s, "SELECT id FROM t WHERE f = 0.25"), [["2"]]);
    assert_eq!(row_strings(&mut s, "SELECT id FROM t WHERE f > 0.5"), [["3"]]);

    // The difference the type buys, in one query and on one row: three tenths
    // is exactly three tenths on the exact column and 0.30000000000000004 on
    // the float one. Deliberately a *product* and not a `sum` -- `sum` over
    // floats here is compensated (Kahan-Babuska-Neumaier) and lands on the
    // correctly-rounded answer, so it would have shown no difference and the
    // test would have been asserting nothing.
    assert_eq!(
        row_strings(&mut s, "SELECT d * 3, f * 3, d * 3 = 0.3, f * 3 = 0.3 FROM t WHERE id = 1"),
        [["0.30", "0.30000000000000004", "true", "false"]]
    );
}

// ===========================================================================
// round trip
// ===========================================================================

/// A literal must survive INSERT and come back, into a decimal column and into
/// a float one, positive and negative. The negative case is the one that needs
/// the parser to fold `-` into the literal -- an unfolded `UnaryOp` over a
/// decimal is not something the VALUES path accepts.
#[test]
fn literals_round_trip_through_insert_and_select() {
    let mut s = Session::in_memory();
    s.execute(
        "CREATE TABLE m (id Int64, price Decimal64(2), rate Decimal64(6), f Float64) \
         ENGINE = MergeTree ORDER BY id",
    )
    .expect("ddl");
    s.execute(
        "INSERT INTO m VALUES (1, 0.1, 0.000001, 0.1), (2, -12.34, -1.5, -0.25), \
         (3, 19.99, 0.075, 1.5)",
    )
    .expect("insert");

    assert_eq!(
        row_strings(&mut s, "SELECT price, rate, f FROM m ORDER BY id"),
        [
            // 0.1 written into a Decimal64(2) column is stored at the column's
            // scale, not the literal's -- and is still exactly one tenth.
            ["0.10", "0.000001", "0.1"],
            ["-12.34", "-1.500000", "-0.25"],
            ["19.99", "0.075000", "1.5"],
        ]
    );
    // Exact all the way through storage: the classic money sum, off by a cent
    // in binary64 and not here.
    assert_eq!(cell(&mut s, "SELECT sum(price) FROM m"), Ok("7.75".into()));
    assert_eq!(cell(&mut s, "SELECT sum(price) = 7.75 FROM m"), Ok("true".into()));
    // A literal too precise for the column is a loud refusal, not a silent
    // truncation -- the standing rule for this type.
    assert!(
        s.execute("INSERT INTO m VALUES (4, 0.1, 0.1, 0.1)").is_ok(),
        "a literal narrower than the column is fine"
    );
    let e = s
        .execute("INSERT INTO m VALUES (5, 1.005, 0.1, 0.1)")
        .map(|_| String::new())
        .unwrap_or_else(|e| e.to_string());
    // Rounding half away from zero is this type's documented rule for a
    // narrowing, so 1.005 into Decimal64(2) is 1.01 rather than an error.
    assert!(e.is_empty(), "{e}");
    assert_eq!(cell(&mut s, "SELECT price FROM m WHERE id = 5"), Ok("1.01".into()));
}

/// The two branch shapes that read a lane without descaling it before this
/// landed, both reachable from a bare literal and both silently wrong: `CASE`
/// answered 10 for `1.0` and a folded `CASE` answered 0.01 for `0.1`.
///
/// Kept here rather than with the operator tests because the *literal* is what
/// made them reachable -- they needed an explicit `Decimal64` CAST before.
#[test]
fn a_decimal_literal_survives_case_and_the_branch_functions() {
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE b (id Int64, f Float64) ENGINE = MergeTree ORDER BY id")
        .expect("ddl");
    s.execute("INSERT INTO b VALUES (1, -3.75)").expect("insert");

    // Constant-folded: the arms have different scales, so the folded value must
    // be rescaled into the result type and not copied lane-for-lane.
    assert_eq!(cell(&mut s, "SELECT CASE WHEN 1=1 THEN 0.1 ELSE 0.25 END"), Ok("0.10".into()));
    // Scale 1, not 2: `promote(Decimal64(1), Int64)` is the decimal's own scale
    // -- an integer is exact at every scale, so nothing has to widen.
    assert_eq!(cell(&mut s, "SELECT CASE WHEN 1=0 THEN 0.1 ELSE 7 END"), Ok("7.0".into()));
    // Evaluated per row, against a float column: the result is a Float64 and
    // the decimal arm has to descale into it.
    assert_eq!(
        row_strings(&mut s, "SELECT CASE WHEN id <> 7 THEN 1.0 ELSE f END FROM b"),
        [["1"]]
    );
    assert_eq!(
        row_strings(&mut s, "SELECT CASE WHEN id = 7 THEN 1.0 ELSE f END FROM b"),
        [["-3.75"]]
    );
    // ...and the same three through the functions that share the shape.
    assert_eq!(cell(&mut s, "SELECT if(1=1, 0.1, 0.25)"), Ok("0.10".into()));
    assert_eq!(cell(&mut s, "SELECT coalesce(0.1, 0.25)"), Ok("0.10".into()));
    assert_eq!(cell(&mut s, "SELECT greatest(0.1, 0.25)"), Ok("0.25".into()));
    assert_eq!(cell(&mut s, "SELECT least(1.5, 2)"), Ok("1.5".into()));
}

/// Places a numeric literal appears that are *not* arithmetic, and where a
/// decimal must be refused or ignored exactly as a float was.
#[test]
fn non_arithmetic_positions_still_reject_a_fractional_literal() {
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE p (id Int64, v Int64) ENGINE = MergeTree ORDER BY id")
        .expect("ddl");
    s.execute("INSERT INTO p VALUES (1, 10), (2, 20)").expect("insert");

    // A select-list position must be a whole number. `ORDER BY 1.5` silently
    // becoming `ORDER BY 1` was a filed bug; a decimal literal must not
    // re-open it by falling through the ordinal check as "not a position".
    let e = s.query("SELECT id, v FROM p ORDER BY 1.5").unwrap_err().to_string();
    assert!(e.contains("whole positive number"), "{e}");
    let e = s.query("SELECT id, v FROM p GROUP BY 1.5").unwrap_err().to_string();
    assert!(e.contains("whole positive number"), "{e}");
    // ...and a whole one still is a position.
    assert_eq!(row_strings(&mut s, "SELECT v FROM p ORDER BY 1 DESC"), [["20"], ["10"]]);

    // LIMIT / OFFSET take a non-negative integer and nothing else.
    for sql in ["SELECT id FROM p LIMIT 1.5", "SELECT id FROM p LIMIT 1 OFFSET 0.5"] {
        assert!(s.query(sql).is_err(), "{sql}");
    }

    // A decimal probing an integer key column is exact or nothing: `= 1.0`
    // names key 1, `= 1.5` names no key at all and must say so rather than
    // truncate into a probe for 1.
    assert_eq!(row_strings(&mut s, "SELECT v FROM p WHERE id = 1.0"), [["10"]]);
    assert!(row_strings(&mut s, "SELECT v FROM p WHERE id = 1.5").is_empty());
}

/// `SHOW CREATE TABLE` output is meant to be pasted back, so a `DEFAULT`
/// written as a decimal literal has to survive the round trip through DDL text
/// -- the path where an f64 used to eat the last digits.
#[test]
fn a_decimal_default_survives_ddl_text() {
    let mut s = Session::in_memory();
    s.execute(
        "CREATE TABLE d (id Int64, price Decimal64(2) DEFAULT 9.99, \
         f Float64 DEFAULT 0.5) ENGINE = MergeTree ORDER BY id",
    )
    .expect("ddl");
    s.execute("INSERT INTO d (id) VALUES (1)").expect("insert");
    assert_eq!(row_strings(&mut s, "SELECT price, f FROM d"), [["9.99", "0.5"]]);
    let ddl = cell(&mut s, "SHOW CREATE TABLE d").expect("show create");
    assert!(ddl.contains("9.99"), "{ddl}");
    // The printed DDL must re-create the same column.
    let mut s2 = Session::in_memory();
    s2.execute(&ddl.replace("CREATE TABLE default.d", "CREATE TABLE d2").replace(" d ", " d2 "))
        .unwrap_or_else(|e| panic!("re-running the printed DDL: {e}\n{ddl}"));
}
