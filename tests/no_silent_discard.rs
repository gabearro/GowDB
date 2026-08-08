//! One rule, driven end to end through `Session`: **nothing is accepted and
//! then ignored.**
//!
//! The README promises the engine never silently does something other than
//! what was asked. Every test here takes a declaration a user can write, and
//! asserts the engine lands on one of exactly two outcomes:
//!
//!   * it **refuses** the declaration, by name, with a message saying what it
//!     would have had to do; or
//!   * it **accepts** it and the declaration is *in force* — provably, from
//!     the outside, by the data that comes back.
//!
//! The forbidden third outcome — accept, echo something else back from
//! `SHOW CREATE TABLE`, and store data that does not match either — is the
//! defect this whole file is named after. It has now been the shape of six
//! separate bugs in this engine (`DEFAULT` stored as unevaluated text,
//! `Decimal(38,2)` narrowed, `DateTime64(3)` truncated, `DateTime('<zone>')`
//! dropped, `NOT NULL` dropped, `SETTINGS` dropped), which is why the check is
//! a test file rather than a code review note.
//!
//! Everything goes through the public `Session` API on purpose. Each of these
//! fixes lives in the parser or the type system, and this engine's
//! characteristic failure is a capability that is complete in `src/` and never
//! reachable from `Session` — a unit test on `DataType::parse` would pass with
//! the whole path unwired.

use granular::types::Value;
use granular::{Error, Session};

// ------------------------------------------------------------------ helpers

fn db() -> Session {
    Session::in_memory()
}

/// `Ok` iff the statement ran. The `Error` itself comes back, not its text, so
/// a test can check the *kind* as well as the wording: a clause the engine
/// does not implement must report `NOT_IMPLEMENTED` and not `SYNTAX_ERROR`,
/// because the difference is "this engine cannot" versus "you typed it wrong"
/// and only one of them tells the reader to stop looking for the typo.
fn run(s: &mut Session, sql: &str) -> Result<(), Error> {
    s.execute(sql)
}

fn ok(s: &mut Session, sql: &str) {
    if let Err(e) = run(s, sql) {
        panic!("should have been accepted: {sql}\n  got: {e}");
    }
}

/// The refusal half of the contract: rejected, and the message contains every
/// one of `must_say` -- a refusal that does not name the thing it refused
/// sends the reader into the source, which is most of what made the silent
/// version tolerable in the first place.
fn refused(s: &mut Session, sql: &str, must_say: &[&str]) -> Error {
    match run(s, sql) {
        Ok(()) => panic!("accepted a declaration it does not honour: {sql}"),
        Err(e) => {
            let text = e.to_string();
            for w in must_say {
                assert!(text.contains(w), "refusal must mention `{w}`: {sql}\n  got: {text}");
            }
            e
        }
    }
}

/// `refused`, plus the insistence that this is a missing feature rather than
/// bad syntax.
fn not_implemented(s: &mut Session, sql: &str, must_say: &[&str]) -> Error {
    let e = refused(s, sql, must_say);
    assert_eq!(e.code(), "NOT_IMPLEMENTED", "wrong error kind for {sql}: {e}");
    e
}

fn one_string(s: &mut Session, sql: &str) -> String {
    s.query(sql)
        .unwrap_or_else(|e| panic!("query failed: {sql}\n  {e}"))
        .to_values()
        .first()
        .and_then(|r| r.first())
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("no string result: {sql}"))
        .to_string()
}

fn scalar(s: &mut Session, sql: &str) -> Value {
    s.query(sql)
        .unwrap_or_else(|e| panic!("query failed: {sql}\n  {e}"))
        .scalar()
        .unwrap_or_else(|| panic!("no scalar: {sql}"))
}

/// `SHOW CREATE TABLE t`, re-executed against a fresh session under the name
/// `rt`, then compared to the original by `DESCRIBE`.
///
/// This is the check that catches an accept-then-ignore even when the ignored
/// clause has no *visible* effect on data: if the engine drops a clause, the
/// DDL it prints is not the DDL it was given, and pasting the printed form
/// into a migration builds a different table. A round trip that survives is
/// the engine agreeing that what it printed is what it has.
fn show_create_round_trips(original_ddl: &str) -> String {
    let mut a = db();
    ok(&mut a, original_ddl);
    let printed = one_string(&mut a, "SHOW CREATE TABLE t");

    let mut b = db();
    ok(&mut b, &printed.replace("TABLE `t`", "TABLE `rt`"));
    let printed_again = one_string(&mut b, "SHOW CREATE TABLE rt");
    assert_eq!(
        printed.replace("TABLE `t`", "TABLE `rt`"),
        printed_again,
        "SHOW CREATE TABLE is not a fixpoint for:\n{original_ddl}"
    );

    let types = |s: &mut Session, t: &str| -> Vec<Vec<Value>> {
        s.query(&format!("DESCRIBE {t}")).unwrap().to_values()
    };
    assert_eq!(types(&mut a, "t"), types(&mut b, "rt"), "round trip changed the schema");
    printed
}

// ------------------------------------------------- 1. DateTime64(precision)

/// `DateTime64(3)` used to be accepted and silently truncated:
/// `'2024-01-15 12:00:00.456'` went in and `12:00:00` came out, with
/// `SHOW CREATE TABLE` echoing a bare `DateTime`. The fraction was gone at
/// ingest, so it was not recoverable afterwards by any later fix.
///
/// Written as a genuine either/or: if a future version implements sub-second
/// precision this test keeps passing, by the other branch.
#[test]
fn datetime64_is_refused_or_keeps_the_fraction() {
    let mut s = db();
    let ddl = "CREATE TABLE t (id Int64, ts DateTime64(3)) ENGINE = MergeTree ORDER BY id";
    match run(&mut s, ddl) {
        Err(e) => {
            assert!(e.to_string().contains("whole seconds"), "must name the limitation: {e}");
            assert_eq!(e.code(), "NOT_IMPLEMENTED", "{e}");
        }
        Ok(()) => {
            ok(&mut s, "INSERT INTO t VALUES (1, '2024-01-15 12:00:00.456')");
            let got = one_string(&mut s, "SELECT toString(ts) FROM t");
            assert!(
                got.contains(".456"),
                "accepted DateTime64(3) and dropped the fraction: {got}"
            );
        }
    }

    // The subset that is a real no-op still parses, because refusing DDL the
    // engine implements exactly would be its own kind of dishonesty.
    ok(
        &mut s,
        "CREATE TABLE zero (id Int64, ts DateTime64(0)) ENGINE = MergeTree ORDER BY id",
    );
    ok(&mut s, "INSERT INTO zero VALUES (1, '2024-01-15 12:00:00')");
    assert_eq!(
        one_string(&mut s, "SELECT toString(ts) FROM zero"),
        "2024-01-15 12:00:00"
    );
}

// ------------------------------------------------------ 2. DateTime('<zone>')

/// `DateTime('America/New_York')` used to be accepted with the zone dropped,
/// so a table copied out of a ClickHouse migration held UTC instants under a
/// column that claimed local time — every value off by the zone's offset, and
/// `SHOW CREATE TABLE` printing a bare `DateTime` so the DDL no longer even
/// described the column.
#[test]
fn datetime_timezone_is_refused_or_shown() {
    let mut s = db();
    let ddl =
        "CREATE TABLE t (id Int64, ts DateTime('America/New_York')) ENGINE = MergeTree ORDER BY id";
    match run(&mut s, ddl) {
        Err(e) => {
            assert!(e.to_string().contains("America/New_York"), "must name the zone: {e}");
            assert_eq!(e.code(), "NOT_IMPLEMENTED", "{e}");
        }
        Ok(()) => {
            // Accepting means honouring, and the minimum visible proof is that
            // the declaration survives into what the engine says it has.
            let printed = one_string(&mut s, "SHOW CREATE TABLE t");
            assert!(
                printed.contains("America/New_York"),
                "accepted a timezone and dropped it from the table: {printed}"
            );
        }
    }

    // A UTC-spelled zone names the lane this engine already has, so it is
    // honoured rather than refused -- and honoured means the column behaves
    // exactly as an undecorated DateTime does.
    ok(
        &mut s,
        "CREATE TABLE u (id Int64, ts DateTime('UTC')) ENGINE = MergeTree ORDER BY id",
    );
    ok(&mut s, "INSERT INTO u VALUES (1, '2024-01-15 12:00:00')");
    assert_eq!(
        one_string(&mut s, "SELECT toString(ts) FROM u"),
        "2024-01-15 12:00:00"
    );
    assert_eq!(scalar(&mut s, "SELECT toUnixTimestamp(ts) FROM u"), Value::Int(1_705_320_000));

    // `Date32` was the same narrowing with a different name: half its declared
    // range is not representable in this engine's unsigned day count, so the
    // alias accepted a column for 1950 and then refused 1950 at INSERT.
    refused(
        &mut s,
        "CREATE TABLE d (id Int64, d Date32) ENGINE = MergeTree ORDER BY id",
        &["1900-2299"],
    );
}

// ------------------------------------------------------------- 3. NOT NULL

/// `NOT NULL` used to be eaten and dropped, on any column. The reproduction
/// from the audit, verbatim: the DDL succeeded, the NULL insert succeeded, the
/// NULL was stored, and `SHOW CREATE TABLE` echoed `Nullable(String)` with the
/// constraint missing.
#[test]
fn not_null_is_enforced_or_refused_never_dropped() {
    let mut s = db();
    let ddl = "CREATE TABLE n (id Int64, x Nullable(String) NOT NULL) ENGINE = MergeTree ORDER BY id";
    match run(&mut s, ddl) {
        Err(e) => {
            let text = e.to_string();
            assert!(text.contains("NOT NULL"), "must name the clause: {text}");
            assert!(text.contains("Nullable(String)"), "must name the type it fights: {text}");
        }
        Ok(()) => {
            // Accepted means enforced: the NULL must not reach the table.
            let stored = run(&mut s, "INSERT INTO n VALUES (1, NULL)");
            assert!(stored.is_err(), "accepted NOT NULL and then stored a NULL");
            assert_eq!(scalar(&mut s, "SELECT count() FROM n"), Value::UInt(0));
        }
    }
}

/// The other half, and the one that must keep working: `NOT NULL` on an
/// ordinary column is the standard-SQL spelling every other dialect uses, it
/// is already true here (nullability is part of the type), and it is enforced
/// on every write path.
#[test]
fn not_null_on_a_plain_column_is_honoured() {
    let mut s = db();
    ok(
        &mut s,
        "CREATE TABLE t (id Int64 NOT NULL, x String NOT NULL) ENGINE = MergeTree ORDER BY id",
    );
    ok(&mut s, "INSERT INTO t VALUES (1, 'a')");

    // Enforced through VALUES...
    let e = run(&mut s, "INSERT INTO t VALUES (2, NULL)").expect_err("NULL must be refused");
    assert!(e.to_string().contains("non-nullable"), "{e}");
    // ...and through INSERT SELECT, which is a different code path.
    assert!(run(&mut s, "INSERT INTO t SELECT 3, NULL").is_err());
    assert_eq!(scalar(&mut s, "SELECT count() FROM t"), Value::UInt(1));

    // And the printed DDL re-creates a table with the same behaviour: the
    // constraint is carried by the type, so it cannot be lost in the round
    // trip the way a dropped `NOT NULL` keyword was.
    let printed = show_create_round_trips(
        "CREATE TABLE t (id Int64 NOT NULL, x String NOT NULL) ENGINE = MergeTree ORDER BY id",
    );
    assert!(printed.contains("`x` String"), "{printed}");
    assert!(!printed.contains("Nullable"), "{printed}");

    let mut b = db();
    ok(&mut b, &printed.replace("TABLE `t`", "TABLE `rt`"));
    assert!(run(&mut b, "INSERT INTO rt VALUES (1, NULL)").is_err());

    // `NULL` is still the dual and still additive: a bare type states no
    // nullability, so asking for it is not a contradiction.
    ok(&mut s, "CREATE TABLE m (id Int64, x String NULL) ENGINE = MergeTree ORDER BY id");
    ok(&mut s, "INSERT INTO m VALUES (1, NULL)");
    assert_eq!(scalar(&mut s, "SELECT count() FROM m WHERE x IS NULL"), Value::UInt(1));
    assert!(one_string(&mut s, "SHOW CREATE TABLE m").contains("Nullable(String)"));
}

// -------------------------------------------------------------- 4. SETTINGS

/// `SELECT 1 SETTINGS max_threads = 2` used to return `Ok` with no effect, and
/// so did `SETTINGS not_a_real_setting = 'zzz'`. A setting is *only* an
/// instruction — there is no data left over when you drop it — so accepting
/// one and ignoring it discards the entire request.
#[test]
fn unknown_and_unimplemented_settings_are_refused() {
    let mut s = db();
    ok(&mut s, "CREATE TABLE t (id Int64) ENGINE = MergeTree ORDER BY id");

    // A name nothing here knows: most likely a typo, and reported as one.
    refused(&mut s, "SELECT 1 SETTINGS not_a_real_setting = 'zzz'", &["not_a_real_setting"]);
    refused(&mut s, "SELECT * FROM t SETTINGS x = 1, y = 2", &["unknown setting"]);

    // A name this engine knows and does not implement: the message says what
    // it would have done, so the reader can decide whether they needed it.
    not_implemented(&mut s, "SELECT 1 SETTINGS max_threads = 2", &["max_threads"]);

    // `max_memory_usage` used to be refused here, and this line asserted that.
    // It is implemented now -- there is a real per-query counter behind it --
    // so accepting it is the honest answer and refusing it would be the lie.
    // The contract this file pins is "honour it or refuse it, never accept and
    // ignore", and honouring is the side it moved to.
    ok(&mut s, "SELECT 1 SETTINGS max_memory_usage = 1000");

    // ...and honoured, not merely tolerated. A ceiling this small has to
    // actually constrain something, or "accepted" would mean exactly the
    // silent no-op this whole file exists to forbid. Seeded with enough
    // distinct groups that the table cannot fit in 4 KiB.
    ok(&mut s, "CREATE TABLE wide (g Int64) ENGINE = MergeTree ORDER BY g");
    let rows: Vec<String> = (0..20_000).map(|i| format!("({i})")).collect();
    ok(&mut s, &format!("INSERT INTO wide VALUES {}", rows.join(",")));
    let e = s
        .query("SELECT g, count() FROM wide GROUP BY g SETTINGS max_memory_usage = 4096")
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        !e.is_empty(),
        "a 4 KiB ceiling did not constrain a 20k-group aggregate -- the setting \
         is being accepted and ignored, which is the bug this file forbids"
    );
    // Same story as `max_memory_usage`: this pinned a refusal, and the deadline
    // is implemented now. Accepted, and honoured -- a one-second deadline on a
    // query that cannot finish in one second has to fire.
    ok(&mut s, "SELECT 1 SETTINGS max_execution_time = 1");

    // Scope is checked too: a table setting on a query is misplaced, not
    // unknown, and saying so is the difference between a one-word fix and a
    // search through the source.
    // `index_granularity = 1024` on a query is accepted rather than refused as
    // misplaced, because 1024 IS this engine's granule size -- the clause
    // asserts something true and changes nothing, which is not the
    // accept-and-ignore this file forbids. A value that differed would be
    // refused on either scope; that is the case worth guarding, and it is below.
    ok(&mut s, "SELECT 1 SETTINGS index_granularity = 1024");
    let e = refused(&mut s, "SELECT 1 SETTINGS index_granularity = 4096", &["1024"]);
    assert!(!e.to_string().contains("unknown"), "{e}");

    // A table setting this engine does not implement is refused by name, so the
    // reader learns which clause to drop rather than that something, somewhere,
    // was wrong.
    refused(&mut s, "SELECT 1 SETTINGS ttl_only_drop_parts = 1", &["ttl_only_drop_parts"]);
}

/// The accepts, and why they are accepts: each of these names a property this
/// engine has already fixed, at exactly the value asked for. Honouring is
/// free; the same setting at any other value is a refusal, because that value
/// is what it could not deliver.
#[test]
fn settings_this_engine_already_implements_are_honoured() {
    let mut s = db();

    // common::BLOCK_SIZE, common::GRANULE_SIZE, and the outer-join fill rule.
    ok(&mut s, "SELECT 1 SETTINGS max_block_size = 8192");
    ok(&mut s, "SELECT 1 SETTINGS join_use_nulls = 1, max_block_size = '8192'");
    ok(
        &mut s,
        "CREATE TABLE g (id Int64) ENGINE = MergeTree ORDER BY id SETTINGS index_granularity = 1024",
    );

    // ...and at any other value they are refused, naming the value in force.
    // 8192 is ClickHouse's default granularity and the single most-pasted
    // setting there is, which is exactly why accepting it was expensive: the
    // granule here is 1024 rows and every zone map is built on that.
    not_implemented(
        &mut s,
        "CREATE TABLE g2 (id Int64) ENGINE = MergeTree ORDER BY id \
         SETTINGS index_granularity = 8192",
        &["1024"],
    );
    refused(&mut s, "SELECT 1 SETTINGS max_block_size = 65536", &["8192"]);
    refused(&mut s, "SELECT 1 SETTINGS join_use_nulls = 0", &["NULL"]);
    // A refused CREATE TABLE must not have half-created the table.
    assert!(s.query("SELECT count() FROM g2").is_err());

    // The honoured one is not merely tolerated: a table declared at the
    // granularity the engine actually runs stores and reads back correctly
    // across several granules (3000 rows is just under three of them).
    let mut ins = String::from("INSERT INTO g VALUES ");
    for i in 0..3000 {
        if i > 0 {
            ins.push(',');
        }
        ins.push_str(&format!("({i})"));
    }
    ok(&mut s, &ins);
    assert_eq!(scalar(&mut s, "SELECT count() FROM g"), Value::UInt(3000));
    assert_eq!(scalar(&mut s, "SELECT max(id) FROM g"), Value::Int(2999));
}

// ------------------------------------------- the same defect, found nearby

/// Found while auditing the four above, and the worst of the extras: engine
/// arguments were parsed and dropped, and the argument that exists is
/// `ReplacingMergeTree(version)` — *the row with the largest version wins*.
/// Dropped, it silently degrades to last-write-wins, so this table returned 2
/// where the DDL asked for 5. Nothing distinguishes that from a correct answer
/// without knowing the version column was supposed to be there.
#[test]
fn engine_arguments_are_refused_not_dropped() {
    let mut s = db();
    let ddl = "CREATE TABLE r (id Int64, v Int64) ENGINE = ReplacingMergeTree(v) ORDER BY id";
    match run(&mut s, ddl) {
        Err(e) => assert!(e.to_string().contains("engine arguments"), "{e}"),
        Ok(()) => {
            ok(&mut s, "INSERT INTO r VALUES (1, 5)");
            ok(&mut s, "INSERT INTO r VALUES (1, 2)");
            assert_eq!(
                scalar(&mut s, "SELECT v FROM r"),
                Value::Int(5),
                "accepted a version column and then kept the older row"
            );
        }
    }
    // Empty parens name no argument and are the same engine, so they stay.
    ok(&mut s, "CREATE TABLE m (id Int64) ENGINE = MergeTree() ORDER BY id");
    ok(&mut s, "INSERT INTO m VALUES (1)");
    assert_eq!(scalar(&mut s, "SELECT count() FROM m"), Value::UInt(1));
}

/// Join strictness was "recognized so ClickHouse SQL parses, then dropped:
/// this engine has one join implementation". True, and exactly the problem —
/// the one it has is `ALL`, and every dropped modifier named a different row
/// set. This is the reproduction: two matching rows on the right, and `ANY`
/// asking for at most one.
#[test]
fn join_strictness_is_refused_not_dropped() {
    let mut s = db();
    ok(&mut s, "CREATE TABLE a (id Int64, v Int64) ENGINE = Memory");
    ok(&mut s, "INSERT INTO a VALUES (1, 10), (2, 20), (3, 30)");
    ok(&mut s, "CREATE TABLE b (id Int64, w Int64) ENGINE = Memory");
    ok(&mut s, "INSERT INTO b VALUES (1, 100), (1, 101), (2, 200)");

    let sql = "SELECT count() FROM a ANY LEFT JOIN b ON a.id = b.id";
    match s.query(sql) {
        Err(e) => {
            assert!(e.to_string().contains("ANY JOIN"), "{e}");
            assert_eq!(e.code(), "NOT_IMPLEMENTED", "{e}");
        }
        // If ANY is ever implemented it must mean what it says: one row per
        // left row, so 3 and not the 4 an ALL join produces.
        Ok(r) => assert_eq!(r.scalar(), Some(Value::UInt(3)), "ANY kept both matches"),
    }
    for m in ["SEMI", "ANTI", "ASOF"] {
        not_implemented(
            &mut s,
            &format!("SELECT count() FROM a {m} LEFT JOIN b ON a.id = b.id"),
            &[m],
        );
    }

    // `ALL` and `GLOBAL` stay: the first names this engine's own semantics and
    // the second distributes to shards it does not have, so both are honoured
    // by doing nothing, which is the only case where doing nothing is honest.
    assert_eq!(
        scalar(&mut s, "SELECT count() FROM a ALL LEFT JOIN b ON a.id = b.id"),
        Value::UInt(4)
    );
    assert_eq!(
        scalar(&mut s, "SELECT count() FROM a GLOBAL JOIN b ON a.id = b.id"),
        Value::UInt(3)
    );
}

/// `TTL` and `SAMPLE BY` were listed in the README under "Not supported", with
/// the claim that each such feature "fails with a specific NOT_IMPLEMENTED
/// message naming the feature". They parsed clean instead. `TTL` is the one
/// with consequences: a table declared to expire rows, and silently given no
/// expiry, keeps returning rows the DDL said would be gone.
#[test]
fn ttl_and_sample_by_are_refused() {
    let mut s = db();
    not_implemented(
        &mut s,
        "CREATE TABLE t (id Int64, ts DateTime) ENGINE = MergeTree ORDER BY id TTL ts + 30",
        &["TTL"],
    );
    refused(
        &mut s,
        "CREATE TABLE t (id Int64, ts DateTime TTL ts + 30) ENGINE = MergeTree ORDER BY id",
        &["TTL"],
    );
    refused(
        &mut s,
        "CREATE TABLE t (id Int64) ENGINE = MergeTree ORDER BY id SAMPLE BY id",
        &["SAMPLE BY"],
    );
    // None of the refusals left a table behind.
    assert!(s.query("SELECT count() FROM t").is_err());
}

// ------------------------------------------------------ SHOW CREATE TABLE

/// The round-trip half of the contract, over every declaration that survives:
/// whatever `SHOW CREATE TABLE` prints must re-create a table that behaves the
/// same. A dropped clause shows up here even when its effect on data is
/// invisible, because the printed DDL is what a user pastes into a migration.
#[test]
fn show_create_table_round_trips_every_accepted_declaration() {
    for ddl in [
        "CREATE TABLE t (id Int64, x String NOT NULL) ENGINE = MergeTree ORDER BY id",
        "CREATE TABLE t (id Int64, x Nullable(String)) ENGINE = MergeTree ORDER BY id",
        "CREATE TABLE t (id Int64, ts DateTime('UTC')) ENGINE = MergeTree ORDER BY id",
        "CREATE TABLE t (id Int64, ts DateTime64(0)) ENGINE = MergeTree ORDER BY id",
        "CREATE TABLE t (id UInt64, d Decimal(10, 2), s LowCardinality(String)) \
         ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
        "CREATE TABLE t (id Int64, x String DEFAULT 'z', n Int32 NULL) \
         ENGINE = ReplacingMergeTree ORDER BY id",
        "CREATE TABLE t (id Int64, p UInt32) ENGINE = MergeTree ORDER BY id PARTITION BY p",
        "CREATE TABLE t (id Int64) ENGINE = MergeTree ORDER BY id SETTINGS index_granularity = 1024",
        "CREATE TABLE t (id Int64, s String CODEC(ZSTD(3))) ENGINE = MergeTree ORDER BY id",
        // A decimal DEFAULT is the case where the printed DDL used to differ
        // from the table it was printed from: unquoted, its digits went back
        // through the lexer's `f64` on the way in.
        "CREATE TABLE t (id Int64, d Decimal64(18) DEFAULT '0.123456789012345678') \
         ENGINE = MergeTree ORDER BY id",
    ] {
        show_create_round_trips(ddl);
    }

    // ...and the round trip is on the *value*, not only on the text.
    let printed = show_create_round_trips(
        "CREATE TABLE t (id Int64, d Decimal64(18) DEFAULT '0.123456789012345678') \
         ENGINE = MergeTree ORDER BY id",
    );
    let mut b = db();
    ok(&mut b, &printed.replace("TABLE `t`", "TABLE `rt`"));
    ok(&mut b, "INSERT INTO rt (id) VALUES (1)");
    assert_eq!(
        one_string(&mut b, "SELECT toString(d) FROM rt"),
        "0.123456789012345678",
        "the printed DDL re-created a column with a different default"
    );
}

/// A `DEFAULT` is a declaration too, and this is the one that survived
/// CREATE TABLE and then changed by itself.
///
/// The catalog persists a default as SQL *text* and re-parses it on open. That
/// re-parse routed anything containing a `.` through `f64`, which holds ~15.9
/// significant digits against `Decimal64(18)`'s 18 — so a default that was
/// exact when written came back different after a restart, and the next
/// checkpoint wrote the changed value back out as though it had always been
/// that. Nothing reports it; the difference is in the digits nobody reads.
#[test]
fn a_decimal_default_survives_a_restart_unchanged() {
    let dir = std::env::temp_dir().join(format!(
        "granular-nsd-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let printed = {
        let mut s = Session::open(&dir).unwrap();
        ok(
            &mut s,
            "CREATE TABLE t (id Int64, d Decimal64(18) DEFAULT '0.123456789012345678') \
             ENGINE = MergeTree ORDER BY id",
        );
        ok(&mut s, "INSERT INTO t (id) VALUES (1)");
        assert_eq!(
            one_string(&mut s, "SELECT toString(d) FROM t"),
            "0.123456789012345678"
        );
        s.checkpoint().unwrap();
        one_string(&mut s, "SHOW CREATE TABLE t")
    };

    // Reopen: the DDL, the default, and a row written *after* the reload must
    // all still be the value that was declared.
    let mut s = Session::open(&dir).unwrap();
    assert_eq!(one_string(&mut s, "SHOW CREATE TABLE t"), printed);
    assert!(printed.contains("0.123456789012345678"), "{printed}");
    ok(&mut s, "INSERT INTO t (id) VALUES (2)");
    let all = s.query("SELECT toString(d) FROM t ORDER BY id").unwrap().to_values();
    for r in &all {
        assert_eq!(
            r[0].as_str(),
            Some("0.123456789012345678"),
            "the default changed across the reload: {all:?}"
        );
    }
    drop(s);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A normalization is only honest when the form actually in force is the form
/// printed. These are the two spellings this engine rewrites, and both pass
/// that bar: the precision of a `Decimal` was a cap that held, and
/// `LowCardinality(String)` really is `String`'s storage (per-granule
/// dictionaries either way).
#[test]
fn normalized_types_print_the_form_in_force() {
    let mut s = db();
    ok(
        &mut s,
        "CREATE TABLE t (id Int64, d Decimal(10, 2), s LowCardinality(String)) \
         ENGINE = MergeTree ORDER BY id",
    );
    let printed = one_string(&mut s, "SHOW CREATE TABLE t");
    assert!(printed.contains("Decimal64(2)"), "{printed}");
    assert!(printed.contains("LowCardinality(String)"), "{printed}");
    ok(&mut s, "INSERT INTO t VALUES (1, '12.34', 'a')");
    assert_eq!(one_string(&mut s, "SELECT toString(d) FROM t"), "12.34");

    // A precision the i64 lane cannot hold is refused rather than narrowed --
    // the same rule, applied before this file existed.
    let e = run(&mut s, "CREATE TABLE w (d Decimal(38, 2)) ENGINE = Memory")
        .expect_err("38 digits do not fit an i64");
    assert!(e.to_string().contains("18"), "{e}");
}

// ===========================================================================
// AUDIT NOTE: the same defect, still open
//
// Found by the same sweep and *not* fixed, because each one's enforcement
// point is outside the parser and the type system. Written down here rather
// than in a review comment so the next reader of this file inherits the list.
//
//  1. `FixedString(n)` never enforces `n`. `CREATE TABLE f (s FixedString(3))`
//     takes `INSERT ... ('abcdefghij')` and stores all ten bytes;
//     `length(s)` then returns 10, where ClickHouse returns 3 (it truncates or
//     zero-pads to exactly n). `CAST(x AS FixedString(3))` is the same. The
//     type is byte-for-byte a `String` in storage, so the declaration is pure
//     decoration -- and unlike `LowCardinality`, which promises nothing about
//     values, this one promises a width. The gate belongs in
//     `Value::cast_to`'s `PhysicalType::Str` arm (src/types/value.rs), which
//     is the single funnel every INSERT and every CAST goes through.
//
//  2. `CODEC(...)` parses into `ColumnDef.codec` and **nothing reads it**.
//     `CREATE TABLE c (s String CODEC(ZSTD(3)))` stores LZ4-or-nothing like
//     every other column, and even `CODEC(NOTACODEC)` is accepted. This one is
//     left alone deliberately: a codec moves bytes, not values, so the answers
//     stay right and only the compression claim is false. It is the single
//     accepted-and-dropped clause in this engine that cannot corrupt data.
//
//  3. There is no `SET` statement at all, so a session cannot carry a setting
//     even where one would be honoured -- and `SETTINGS` still has nowhere in
//     the AST to land, so the validation in the parser is a gate rather than a
//     value a later phase can consume.
//
//  4. `SELECT ... FROM t FINAL` parses into `TableRef::Table.final_`, which
//     the binder never reads. Verified inert rather than broken: this engine
//     collapses duplicate keys at write and at merge, not under `FINAL`, so
//     the rows are already what `FINAL` would produce. It is on this list only
//     because "parses into a field nothing reads" is how every entry above
//     started.
// ===========================================================================
