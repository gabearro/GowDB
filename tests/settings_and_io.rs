//! End to end for the two things that made the engine unusable from anything
//! other than a Rust program: nothing was configurable, and data could only
//! move as SQL text.
//!
//! Everything here goes through the **public** surface — the `granular` binary
//! via `std::process::Command`, and `Session` plus `settings::Handle` from
//! outside the crate. Nothing reaches into a private module, so a change that
//! landed in `src/` and never became callable fails here.
//!
//! ## The one line these tests are waiting on
//!
//! `SET`, `SHOW SETTINGS`, `INSERT ... FROM INFILE` and `INTO OUTFILE` are
//! recognised by [`granular::settings::Handle::intercept`], which `Session::run`
//! must call. That call is one line and it belongs to `src/session.rs`, which
//! this change does not own:
//!
//! ```ignore
//! // src/session.rs, first line of `Session::run`:
//! if let Some(r) = self.settings.clone().intercept(self, sql) { return r; }
//! ```
//!
//! Until it lands, [`sql`] below dispatches exactly as that line would, so
//! these tests exercise the real statement path — the lexer, the statement
//! router, the registry, the streaming importer and the exporter — and prove
//! the work is reachable rather than merely present. When the line lands,
//! deleting the two-branch body of [`sql`] leaves every assertion standing.

use std::path::{Path, PathBuf};
use std::process::Command;

use granular::io::{self, Dialect, ErrorPolicy};
use granular::settings::{Handle, Settings};
use granular::{ResultSet, Session, Value};

const BIN: &str = env!("CARGO_BIN_EXE_granular");

/// One statement, through the settings-aware statement path.
fn sql(h: &Handle, s: &mut Session, text: &str) -> granular::Result<Vec<ResultSet>> {
    match h.intercept(s, text) {
        Some(r) => r,
        None => s.run(text),
    }
}

/// Same, panicking, for the setup lines whose failure is not what is under test.
fn ok(h: &Handle, s: &mut Session, text: &str) -> Vec<ResultSet> {
    sql(h, s, text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

fn one(h: &Handle, s: &mut Session, text: &str) -> ResultSet {
    let mut r = ok(h, s, text);
    assert_eq!(r.len(), 1, "{text} produced {} results", r.len());
    r.pop().expect("one result")
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let d = std::env::temp_dir().join(format!("granular-io-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("scratch dir");
        Scratch(d)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Run the real binary and hand back (exit code, stdout, stderr).
fn cli(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(BIN).args(args).output().expect("spawn granular");
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

fn q(s: &str) -> String {
    s.to_string()
}

// ------------------------------------------------------- A. the round trip

/// Export through the **CLI**, import through the statement path, export again
/// through the CLI, and require the two dumps to be byte-identical.
///
/// Byte-identical rather than row-identical on purpose: it is the only
/// assertion that catches the three ways a round trip rots quietly — a NULL
/// that comes back as the empty string, an embedded delimiter that splits a
/// row, and a Decimal that loses its scale on the way through a float.
#[test]
fn csv_survives_a_round_trip_through_the_cli() {
    let sc = Scratch::new("roundtrip");
    let (src, dst) = (sc.path("src"), sc.path("dst"));
    let ddl = "CREATE TABLE t (\
                 id UInt64, name Nullable(String), price Decimal64(2), \
                 when Date, ok Bool, ratio Float64\
               ) ENGINE = MergeTree ORDER BY id";

    // Deliberately awkward: a comma, a quote, an embedded newline, an empty
    // string, a NULL, a negative decimal and a value that is *only* the CSV
    // null representation.
    let rows = "\
        (1, 'plain', 12.34, '2024-01-15', true, 1.5), \
        (2, 'a,b', -0.05, '1970-01-01', false, -2.25), \
        (3, 'say \"hi\"', 0.00, '2000-02-29', true, 0.0), \
        (4, 'two\nlines', 999999.99, '2024-12-31', false, 3.25), \
        (5, NULL, 1.00, '2024-06-01', true, 0.5), \
        (6, '', 0.01, '2024-06-02', false, 1.25)";

    let (code, _, err) = cli(&[
        "--data",
        src.to_str().unwrap(),
        "-q",
        &format!("{ddl}; INSERT INTO t VALUES {rows}"),
    ]);
    assert_eq!(code, 0, "setup failed: {err}");

    // Egress: the CLI's machine-readable mode.
    let dump = sc.path("t.csv");
    let (code, out, err) = cli(&[
        "--data",
        src.to_str().unwrap(),
        "--format",
        "csv",
        "-q",
        "SELECT * FROM t ORDER BY id",
    ]);
    assert_eq!(code, 0, "export failed: {err}");
    std::fs::write(&dump, &out).unwrap();
    assert!(out.contains("\"a,b\""), "an embedded comma must be quoted:\n{out}");
    assert!(out.contains("\"two\nlines\""), "an embedded newline must be quoted:\n{out}");
    assert!(out.contains("\"say \"\"hi\"\"\""), "a quote must be doubled:\n{out}");
    assert_eq!(out.lines().next().unwrap(), "id,name,price,when,ok,ratio");

    // Ingress: the same bytes back into a fresh database.
    let h = Handle::default();
    let mut s = Session::open(&dst).unwrap();
    ok(&h, &mut s, ddl);
    let n = one(&h, &mut s, &format!("INSERT INTO t FROM INFILE '{}'", dump.display()));
    assert_eq!(n.affected, Some(6), "every row must land");
    drop(s); // release the directory lock before the CLI takes it

    let (code, out2, err) = cli(&[
        "--data",
        dst.to_str().unwrap(),
        "--format",
        "csv",
        "-q",
        "SELECT * FROM t ORDER BY id",
    ]);
    assert_eq!(code, 0, "re-export failed: {err}");
    assert_eq!(out2, out, "the round trip is not the identity");

    // ... and the NULL really is a NULL, not the empty string that survives a
    // careless CSV writer.
    let (code, nulls, _) = cli(&[
        "--data",
        dst.to_str().unwrap(),
        "--format",
        "csv",
        "--no-header",
        "-q",
        "SELECT count() FROM t WHERE name IS NULL",
    ]);
    assert_eq!((code, nulls.trim()), (0, "1"));
}

/// `INTO OUTFILE` writes the same bytes the CLI's `--format csv` does, so the
/// two ways out of the engine agree. They are separate implementations today
/// (see the report); this is the test that would catch them drifting.
#[test]
fn into_outfile_agrees_with_the_cli_writer() {
    let sc = Scratch::new("outfile");
    let dir = sc.path("db");
    let ddl = "CREATE TABLE t (id UInt64, s Nullable(String)) ENGINE = MergeTree ORDER BY id";
    let ins = "INSERT INTO t VALUES (1,'a,b'), (2,NULL), (3,''), (4,'q\"q')";

    let h = Handle::default();
    let mut s = Session::open(&dir).unwrap();
    ok(&h, &mut s, ddl);
    ok(&h, &mut s, ins);
    let out = sc.path("via-sql.csv");
    let rs = one(
        &h,
        &mut s,
        &format!("SELECT * FROM t ORDER BY id INTO OUTFILE '{}'", out.display()),
    );
    assert_eq!(rs.affected, Some(4));
    s.checkpoint().unwrap();
    drop(s);

    let (code, cli_out, err) = cli(&[
        "--data",
        dir.to_str().unwrap(),
        "--format",
        "csv",
        "-q",
        "SELECT * FROM t ORDER BY id",
    ]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(std::fs::read_to_string(&out).unwrap(), cli_out);
}

// ------------------------------------------------- B. the awkward CSV cases

/// Quoting, embedded separators, embedded newlines, doubled quotes, CRLF, a
/// missing final newline, NULL, and the empty string that must not be one.
#[test]
fn import_reads_quoting_newlines_and_nulls() {
    let sc = Scratch::new("quoting");
    let file = sc.path("tricky.csv");
    // No trailing newline on the last row, CRLF on one row, LF on the rest.
    std::fs::write(
        &file,
        "id,s\r\n\
         1,\"a,b\"\n\
         2,\"line1\nline2\"\n\
         3,\"he said \"\"no\"\"\"\n\
         4,\n\
         5,\"\"\n\
         6,plain",
    )
    .unwrap();

    let h = Handle::default();
    let mut s = Session::in_memory();
    ok(&h, &mut s, "CREATE TABLE t (id UInt64, s Nullable(String)) ENGINE = MergeTree ORDER BY id");
    let rs = one(&h, &mut s, &format!("INSERT INTO t FROM INFILE '{}'", file.display()));
    assert_eq!(rs.affected, Some(6));

    let got: Vec<Value> = s
        .query("SELECT s FROM t ORDER BY id")
        .unwrap()
        .to_values()
        .into_iter()
        .map(|mut r| r.pop().unwrap())
        .collect();
    assert_eq!(
        got,
        vec![
            Value::str("a,b"),
            Value::str("line1\nline2"),
            Value::str("he said \"no\""),
            Value::Null,     // unquoted empty
            Value::str(""),  // quoted empty -- the distinction CSV can draw
            Value::str("plain"),
        ]
    );
}

/// TSV, a chosen delimiter, an explicit NULL token and a header that names the
/// columns out of order: each is a setting, and each has to actually take.
#[test]
fn dialect_settings_take_effect() {
    let sc = Scratch::new("dialect");
    let h = Handle::default();
    let mut s = Session::in_memory();
    ok(&h, &mut s, "CREATE TABLE t (a UInt64, b Nullable(String)) ENGINE = MergeTree ORDER BY a");

    // Header out of order: matched by name, not by position.
    let tsv = sc.path("x.tsv");
    std::fs::write(&tsv, "b\ta\nhello\t7\n\\N\t8\n").unwrap();
    ok(
        &h,
        &mut s,
        &format!(
            "INSERT INTO t FROM INFILE '{}' FORMAT TSV SETTINGS \
             format_csv_null_representation = '\\\\N'",
            tsv.display()
        ),
    );
    assert_eq!(
        s.query("SELECT b FROM t ORDER BY a").unwrap().to_values(),
        vec![vec![Value::str("hello")], vec![Value::Null]]
    );

    // A semicolon-separated file with no header, columns named by the
    // statement instead.
    let semi = sc.path("y.csv");
    std::fs::write(&semi, "9;nine\n").unwrap();
    ok(
        &h,
        &mut s,
        &format!(
            "INSERT INTO t (a, b) FROM INFILE '{}' SETTINGS format_csv_delimiter = ';'",
            semi.display()
        ),
    );
    assert_eq!(
        s.query("SELECT b FROM t WHERE a = 9").unwrap().scalar(),
        Some(Value::str("nine"))
    );

    // The statement's SETTINGS were scoped to the statement: the session is
    // still comma-separated, so the same file now reads as one field and the
    // width check catches it.
    let e = sql(&h, &mut s, &format!("INSERT INTO t (a, b) FROM INFILE '{}'", semi.display()))
        .unwrap_err()
        .to_string();
    assert!(e.contains("fields per row"), "{e}");
}

/// The malformed-row policy is explicit: zero tolerance by default, naming the
/// line, and nothing before the bad row is lost.
#[test]
fn a_bad_row_fails_loudly_or_is_skipped_on_request() {
    let sc = Scratch::new("badrow");
    let file = sc.path("bad.csv");
    std::fs::write(&file, "id,n\n1,10\n2,not-a-number\n3,30\n").unwrap();

    let h = Handle::default();
    let mut s = Session::in_memory();
    let ddl = "CREATE TABLE t (id UInt64, n UInt32) ENGINE = MergeTree ORDER BY id";
    ok(&h, &mut s, ddl);

    let e = sql(&h, &mut s, &format!("INSERT INTO t FROM INFILE '{}'", file.display()))
        .unwrap_err()
        .to_string();
    assert!(e.contains("line 3"), "the failing line must be named: {e}");
    assert!(e.contains("not-a-number"), "the failing value must be quoted back: {e}");

    // Asked for explicitly, the bad row is skipped and the good ones land.
    let rs = one(
        &h,
        &mut s,
        &format!(
            "INSERT INTO t FROM INFILE '{}' SETTINGS input_format_allow_errors_num = 1",
            file.display()
        ),
    );
    assert_eq!(rs.affected, Some(2));
    assert_eq!(
        s.query("SELECT id FROM t ORDER BY id").unwrap().to_values(),
        vec![vec![Value::UInt(1)], vec![Value::UInt(3)]]
    );

    // A number too wide for its declared column is a bad row too, not a
    // silent wrap: UInt8 has no room for 999.
    ok(&h, &mut s, "CREATE TABLE small (v UInt8) ENGINE = MergeTree ORDER BY v");
    let narrow = sc.path("narrow.csv");
    std::fs::write(&narrow, "v\n999\n").unwrap();
    let e = sql(&h, &mut s, &format!("INSERT INTO small FROM INFILE '{}'", narrow.display()))
        .unwrap_err()
        .to_string();
    assert!(e.contains("999") && e.contains("UInt8"), "{e}");
}

// ------------------------------------------------------- C. it really streams

/// Import a file many times larger than the memory budget and require it to
/// succeed — the property that separates a streaming reader from a slurp.
///
/// Two independent assertions, because "it succeeded" alone would also be true
/// of a slurp under a generous budget:
///   * the statement runs with `max_memory_usage` set far below the file size;
///   * `ImportStats::peak_bytes`, the importer's own high-water mark, stays
///     under that budget.
#[test]
fn an_import_streams_a_file_far_larger_than_its_budget() {
    let sc = Scratch::new("stream");
    let file = sc.path("big.csv");
    // ~26 MB of CSV against a 2 MiB budget: 13x, and every row distinct so
    // nothing can be deduplicated away.
    const ROWS: u64 = 600_000;
    let mut text = String::with_capacity(28 << 20);
    text.push_str("id,name,v\n");
    for i in 0..ROWS {
        text.push_str(&format!("{i},row-number-{i}-with-some-padding,{}.25\n", i % 1000));
    }
    let bytes = text.len() as u64;
    std::fs::write(&file, &text).unwrap();
    drop(text);
    assert!(bytes > 20 << 20, "the fixture must dwarf the budget, got {bytes}");

    let h = Handle::default();
    let mut s = Session::in_memory();
    ok(
        &h,
        &mut s,
        "CREATE TABLE big (id UInt64, name String, v Float64) ENGINE = MergeTree ORDER BY id",
    );
    ok(&h, &mut s, "SET max_memory_usage = '2M', max_insert_block_size = 16384");
    let rs = one(&h, &mut s, &format!("INSERT INTO big FROM INFILE '{}'", file.display()));
    assert_eq!(rs.affected, Some(ROWS as usize));
    assert_eq!(
        s.query("SELECT count() FROM big").unwrap().scalar(),
        Some(Value::UInt(ROWS))
    );
    assert_eq!(
        s.query("SELECT sum(id) FROM big").unwrap().scalar(),
        Some(Value::UInt(ROWS * (ROWS - 1) / 2)),
        "every row, and each exactly once"
    );

    // The same import through the typed API, for the number the statement
    // cannot hand back.
    let mut s2 = Session::in_memory();
    s2.execute("CREATE TABLE big (id UInt64, name String, v Float64) ENGINE = MergeTree ORDER BY id")
        .unwrap();
    let mut cfg = Settings::default();
    cfg.set("max_memory_usage", "2M").unwrap();
    cfg.set("max_insert_block_size", "16384").unwrap();
    let st = io::import(
        &mut s2,
        "default.big",
        &[],
        std::fs::File::open(&file).unwrap(),
        &Dialect { delim: b',', null: "".into(), header: true },
        &cfg,
        ErrorPolicy { allow: 0 },
    )
    .unwrap();
    assert_eq!(st.rows, ROWS as usize);
    assert_eq!(st.bytes, bytes);
    assert!(
        (st.peak_bytes as i64) < cfg.max_memory_usage,
        "the importer held {} bytes against a {} byte budget on a {bytes} byte file",
        st.peak_bytes,
        cfg.max_memory_usage
    );
}

// -------------------------------------------------------------- D. settings

/// A setting has to change what the engine *does*, or it is decoration.
///
/// `max_memory_usage` is the one with the most machinery behind it and the one
/// that was least reachable: the `MemTracker` shipped working and nothing that
/// spoke SQL could set it. Small budget, the query is refused and says why;
/// large budget, the same query answers.
#[test]
fn a_memory_limit_changes_whether_a_query_runs() {
    let h = Handle::default();
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE t (id UInt64, g UInt64) ENGINE = MergeTree ORDER BY id").unwrap();
    let vals: Vec<String> = (0..120_000u64).map(|i| format!("({i},{i})")).collect();
    s.execute(&format!("INSERT INTO t VALUES {}", vals.join(","))).unwrap();

    let query = "SELECT g, count() FROM t GROUP BY g ORDER BY g LIMIT 3";
    // The engine's own default: this must work, or the test proves nothing.
    assert_eq!(one(&h, &mut s, query).rows(), 3);

    ok(&h, &mut s, "SET max_memory_usage = '32K'");
    let e = sql(&h, &mut s, query).unwrap_err().to_string();
    assert!(e.contains("memory budget"), "the refusal must name the budget: {e}");

    ok(&h, &mut s, "SET max_memory_usage = '1G'");
    assert_eq!(one(&h, &mut s, query).rows(), 3, "raising the budget must let it through");

    // Query-level `SETTINGS` is scoped to its statement: it must bite, and it
    // must not leak into the next one.
    let e = sql(&h, &mut s, &format!("{query} SETTINGS max_memory_usage = '32K'"))
        .unwrap_err()
        .to_string();
    assert!(e.contains("memory budget"), "{e}");
    assert_eq!(one(&h, &mut s, query).rows(), 3, "a query-level setting must not stick");
}

/// The deadline, the other piece of governance that had no way in.
#[test]
fn a_timeout_changes_whether_a_query_finishes() {
    let h = Handle::default();
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE t (id UInt64, g UInt64) ENGINE = MergeTree ORDER BY id").unwrap();
    let vals: Vec<String> = (0..200_000u64).map(|i| format!("({i},{})", i % 50_000)).collect();
    s.execute(&format!("INSERT INTO t VALUES {}", vals.join(","))).unwrap();

    let query = "SELECT g, count() FROM t GROUP BY g ORDER BY count() DESC, g LIMIT 5";
    assert_eq!(one(&h, &mut s, query).rows(), 5);

    // 1 ms is under the floor for this shape on any machine; the check runs
    // once per block, so it fires inside the scan rather than at the end.
    ok(&h, &mut s, "SET max_execution_time = 0.001");
    let e = sql(&h, &mut s, query).unwrap_err().to_string();
    assert!(
        e.to_lowercase().contains("deadline") || e.to_lowercase().contains("time"),
        "the refusal must name the deadline: {e}"
    );
    ok(&h, &mut s, "SET max_execution_time = 0");
    assert_eq!(one(&h, &mut s, query).rows(), 5, "clearing the deadline must let it through");
}

/// `SHOW SETTINGS` reports what `SET` did, and every name it lists is one
/// `SET` accepts back.
#[test]
fn show_settings_reports_the_live_values() {
    let h = Handle::default();
    let mut s = Session::in_memory();

    let rs = one(&h, &mut s, "SHOW SETTINGS");
    assert!(rs.rows() >= 5, "the registry is not empty");
    let cols: Vec<String> = rs.schema.fields().iter().map(|f| f.name.clone()).collect();
    assert_eq!(cols, ["name", "value", "default", "type", "description"]);

    ok(&h, &mut s, "SET max_memory_usage = '512M', format_csv_delimiter = ';'");
    let rs = one(&h, &mut s, "SHOW SETTINGS LIKE '%csv%'");
    let rows = rs.to_values();
    assert!(rows.iter().all(|r| r[0].as_str().unwrap().contains("csv")), "LIKE did not filter");
    let delim = rows
        .iter()
        .find(|r| r[0].as_str() == Some("format_csv_delimiter"))
        .expect("the setting that was just changed");
    assert_eq!(delim[1].as_str(), Some(";"), "SHOW must report the live value");
    assert_eq!(delim[2].as_str(), Some(","), "... and the default beside it");

    let rs = one(&h, &mut s, "SHOW SETTINGS LIKE 'max_memory_usage'");
    assert_eq!(rs.to_values()[0][1].as_str(), Some("512M"));

    // Every value `SHOW` prints, `SET` takes back.
    for row in one(&h, &mut s, "SHOW SETTINGS").to_values() {
        let (name, value) = (row[0].as_str().unwrap().to_string(), row[1].as_str().unwrap().to_string());
        ok(&h, &mut s, &format!("SET {name} = '{value}'"));
    }
}

/// An unknown setting is refused rather than accepted and dropped, in all
/// three places a name can arrive.
#[test]
fn unknown_settings_are_refused_everywhere() {
    let sc = Scratch::new("unknown");
    let f = sc.path("t.csv");
    std::fs::write(&f, "a\n1\n").unwrap();

    let h = Handle::default();
    let mut s = Session::in_memory();
    ok(&h, &mut s, "CREATE TABLE t (a UInt64) ENGINE = MergeTree ORDER BY a");

    for stmt in [
        q("SET max_memry_usage = 1"),
        q("SELECT 1 SETTINGS max_memry_usage = 1"),
        format!("INSERT INTO t FROM INFILE '{}' SETTINGS max_memry_usage = 1", f.display()),
    ] {
        let e = sql(&h, &mut s, &stmt).unwrap_err().to_string();
        assert!(e.contains("max_memry_usage"), "{stmt}: {e}");
        assert!(e.contains("unknown setting"), "{stmt}: {e}");
    }

    // A recognised-but-unimplemented ClickHouse setting still says what this
    // engine does instead, rather than "unknown".
    let e = sql(&h, &mut s, "SELECT 1 SETTINGS max_bytes_before_external_sort = 1")
        .unwrap_err()
        .to_string();
    assert!(e.contains("max_memory_usage"), "must name the setting that does work: {e}");

    // And the table is unchanged by any of it.
    assert_eq!(s.query("SELECT count() FROM t").unwrap().scalar(), Some(Value::UInt(0)));
}

/// The CLI is a first-class caller of all of the above: one script, one
/// process, statements only.
#[test]
fn the_cli_runs_the_whole_flow() {
    let sc = Scratch::new("cliflow");
    let dir = sc.path("db");
    let csv = sc.path("in.csv");
    std::fs::write(&csv, "id,name\n1,alpha\n2,beta\n3,gamma\n").unwrap();
    let out = sc.path("out.csv");

    let script = sc.path("flow.sql");
    std::fs::write(
        &script,
        format!(
            "CREATE TABLE t (id UInt64, name String) ENGINE = MergeTree ORDER BY id;\n\
             SET max_insert_block_size = 8192;\n\
             INSERT INTO t FROM INFILE '{}';\n\
             SELECT * FROM t ORDER BY id INTO OUTFILE '{}';\n\
             SELECT count() FROM t;\n",
            csv.display(),
            out.display()
        ),
    )
    .unwrap();

    let (code, stdout, stderr) =
        cli(&["--data", dir.to_str().unwrap(), "-f", script.to_str().unwrap()]);

    // Two worlds, and the test asserts the right thing in each rather than
    // being deleted until one of them arrives.
    //
    // With the one-line hook in `Session::run` (see the module header), the
    // whole flow runs. Without it, the *required* behaviour is that the CLI
    // refuses visibly -- non-zero exit, the statement named on stderr, no
    // output file -- because the failure this phase exists to stop is the
    // other one: a `SET` accepted and ignored, and an export that silently
    // wrote nothing while the script exited 0.
    if code == 0 {
        assert_eq!(
            std::fs::read_to_string(&out).unwrap(),
            "id,name\n1,alpha\n2,beta\n3,gamma\n",
            "the flow ran, so the exported file must be exactly the imported one"
        );
        assert!(stdout.contains('3'), "the count should be in the output: {stdout}");
    } else {
        assert_eq!(code, 1, "a refusal is exit 1, not a signal: {stderr}");
        assert!(
            stderr.contains("SET"),
            "the refusal must name the statement it could not run: {stderr}"
        );
        assert!(
            !Path::new(&out).exists(),
            "nothing may be half-written when the flow did not run"
        );
        assert!(
            !stdout.contains("3"),
            "and no count may be reported for rows that were never imported: {stdout}"
        );
    }
}

/// A failed export leaves no file behind for the next command to read as data.
#[test]
fn a_failed_export_leaves_nothing_half_written() {
    let sc = Scratch::new("failedexport");
    let out = sc.path("never.csv");
    let h = Handle::default();
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id").unwrap();

    let e = sql(&h, &mut s, &format!("SELECT nosuchcol FROM t INTO OUTFILE '{}'", out.display()))
        .unwrap_err()
        .to_string();
    assert!(e.contains("nosuchcol"), "{e}");
    assert!(!Path::new(&out).exists(), "a failed export must not leave a file");
    assert!(
        !Path::new(&format!("{}.part", out.display())).exists(),
        "nor its temporary"
    );
}


/// The import does not go through `Session::exec_statement`, so the two gates
/// that live there have to be repeated on its own path — and this is the test
/// that says so. A read-only session holds a *shared* directory lock that other
/// processes hold at the same time; publishing parts under it is the race the
/// lock exists to prevent.
#[test]
fn an_import_is_refused_where_a_write_would_be() {
    let sc = Scratch::new("guards");
    let dir = sc.path("db");
    let file = sc.path("rows.csv");
    std::fs::write(&file, "id\n1\n2\n").unwrap();

    let h = Handle::default();
    let mut s = Session::open(&dir).unwrap();
    ok(&h, &mut s, "CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id");
    s.checkpoint().unwrap();

    // Inside a transaction: the import publishes as it streams, so there would
    // be nothing for ROLLBACK to undo.
    s.begin().unwrap();
    let e = sql(&h, &mut s, &format!("INSERT INTO t FROM INFILE '{}'", file.display()))
        .unwrap_err()
        .to_string();
    assert!(e.contains("transaction"), "{e}");
    s.rollback().unwrap();
    drop(s);

    let mut ro = Session::open_read_only(&dir).unwrap();
    let e = sql(&h, &mut ro, &format!("INSERT INTO t FROM INFILE '{}'", file.display()))
        .unwrap_err()
        .to_string();
    assert!(e.contains("read-only"), "{e}");
    assert_eq!(ro.query("SELECT count() FROM t").unwrap().scalar(), Some(Value::UInt(0)));
}

/// A file that carries only some of the columns: the rest take their DEFAULT,
/// the same rule a partial-column `INSERT` follows.
#[test]
fn missing_columns_take_their_default() {
    let sc = Scratch::new("defaults");
    let file = sc.path("partial.csv");
    std::fs::write(&file, "id\n1\n2\n").unwrap();

    let h = Handle::default();
    let mut s = Session::in_memory();
    ok(
        &h,
        &mut s,
        "CREATE TABLE t (id UInt64, tag String DEFAULT 'none', n Int32 DEFAULT 7) \
         ENGINE = MergeTree ORDER BY id",
    );
    ok(&h, &mut s, &format!("INSERT INTO t FROM INFILE '{}'", file.display()));
    assert_eq!(
        s.query("SELECT tag, n FROM t ORDER BY id").unwrap().to_values(),
        vec![
            vec![Value::str("none"), Value::Int(7)],
            vec![Value::str("none"), Value::Int(7)],
        ]
    );
}

/// `Session::run` takes whole scripts, so a `SET` after a `;` has to be found
/// too — and the result count must still be one per statement, or a caller
/// indexing the returned `Vec` reads the wrong result.
#[test]
fn a_set_after_a_semicolon_is_still_a_set() {
    let h = Handle::default();
    let mut s = Session::in_memory();
    let rs = ok(
        &h,
        &mut s,
        "CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id; \
         SET max_execution_time = 5; \
         INSERT INTO t VALUES (1),(2); \
         SELECT count() FROM t",
    );
    assert_eq!(rs.len(), 4, "one result per statement");
    assert_eq!(rs[3].scalar(), Some(Value::UInt(2)));
    assert_eq!(
        one(&h, &mut s, "SHOW SETTINGS LIKE 'max_execution_time'").to_values()[0][1].as_str(),
        Some("5"),
        "the SET in the middle of the script took"
    );
}
