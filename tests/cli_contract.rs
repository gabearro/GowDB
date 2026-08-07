//! The contract `granular` owes a shell, tested by running the real binary.
//!
//! Every test here spawns `CARGO_BIN_EXE_granular` with `std::process::Command`
//! and looks at the three things a script can actually see: the exit status,
//! stdout, and stderr. Nothing in this file touches the library — a fix that
//! landed in `src/` but never reached `main` would fail here, which is the
//! whole point of testing at this layer.
//!
//! The two defects being pinned:
//!
//!   * **exit 0 on every error.** A syntax error, a missing table, a failed
//!     INSERT, a bad command line — all of them printed a message and exited
//!     0, so `granular -q ... && next` ran `next` after a failure and no CI
//!     job, Makefile or pipeline could tell whether anything had worked.
//!   * **line-based statement splitting.** Any line whose trimmed end was `;`
//!     ended a statement, so `VALUES ('a;` / `b')` split a string literal in
//!     half and `SELECT 1 + -- note ;` ended before its operand. Boundaries
//!     now come from the SQL lexer, which is what the parser uses.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const BIN: &str = env!("CARGO_BIN_EXE_granular");

// ------------------------------------------------------------------ harness

/// A scratch directory that removes itself. Every test gets its own so the
/// data-directory `flock` is never contended and the tests can run in
/// parallel, which is how `cargo test` runs them.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        // pid + tag is unique: tags are distinct per test and tests share a
        // process. No randomness, so a leftover directory is reproducible.
        let d = std::env::temp_dir().join(format!("granular-cli-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("scratch dir");
        Scratch(d)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
    /// Write a script and hand back its path.
    fn script(&self, name: &str, body: &str) -> PathBuf {
        let p = self.path(name);
        std::fs::write(&p, body).expect("write script");
        p
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Run {
    code: i32,
    out: String,
    err: String,
}

impl Run {
    fn of(o: Output) -> Run {
        Run {
            // A `None` code means a signal killed it -- an abort on a broken
            // pipe, say. -1 makes that fail loudly instead of looking like 0.
            code: o.status.code().unwrap_or(-1),
            out: String::from_utf8_lossy(&o.stdout).into_owned(),
            err: String::from_utf8_lossy(&o.stderr).into_owned(),
        }
    }
}

fn run(args: &[&str]) -> Run {
    Run::of(Command::new(BIN).args(args).output().expect("spawn granular"))
}

/// Same, with `stdin` fed from a string rather than inherited: this is the
/// piped-script path, and it must not be a terminal.
fn run_stdin(args: &[&str], input: &str) -> Run {
    let mut child = Command::new(BIN)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn granular");
    child.stdin.take().unwrap().write_all(input.as_bytes()).expect("feed stdin");
    Run::of(child.wait_with_output().expect("wait"))
}

fn db(s: &Scratch) -> String {
    s.path("db").to_str().expect("utf-8 path").to_string()
}

fn p(path: &Path) -> String {
    path.to_str().expect("utf-8 path").to_string()
}

// -------------------------------------------------------------- exit status

#[test]
fn success_exits_zero() {
    let r = run(&["-q", "SELECT 1 AS a", "--format", "tsv"]);
    assert_eq!(r.code, 0, "stderr: {}", r.err);
    assert_eq!(r.out, "a\n1\n");
    assert!(r.err.is_empty(), "nothing belongs on stderr: {}", r.err);
}

#[test]
fn syntax_error_exits_one() {
    let r = run(&["-q", "SELEKT 1"]);
    assert_eq!(r.code, 1, "a syntax error must not exit 0");
    assert!(r.err.contains("SYNTAX_ERROR"), "stderr: {}", r.err);
}

#[test]
fn missing_table_exits_one() {
    let r = run(&["-q", "SELECT * FROM does_not_exist"]);
    assert_eq!(r.code, 1);
    assert!(r.err.contains("does_not_exist"), "stderr: {}", r.err);
}

#[test]
fn failed_insert_exits_one() {
    let s = Scratch::new("failed-insert");
    let r = run(&["--data", &db(&s), "-q", "INSERT INTO nowhere VALUES (1)"]);
    assert_eq!(r.code, 1);
    assert!(!r.err.is_empty());
}

/// The shape that motivated all of this: `granular -q ... && next_step`.
#[test]
fn shell_and_and_does_not_run_after_a_failure() {
    let r = Run::of(
        Command::new("/bin/sh")
            .arg("-c")
            .arg(format!("{BIN:?} -q 'SELECT * FROM nope' 2>/dev/null && echo RAN_ANYWAY"))
            .output()
            .expect("spawn sh"),
    );
    assert!(!r.out.contains("RAN_ANYWAY"), "the `&&` branch ran after a failure");
}

#[test]
fn errors_go_to_stderr_not_stdout() {
    let r = run(&["-q", "SELECT * FROM nope"]);
    assert!(r.out.is_empty(), "an error leaked onto stdout: {:?}", r.out);
    assert!(r.err.contains("error ["), "stderr: {:?}", r.err);
}

#[test]
fn unknown_argument_exits_two() {
    let r = run(&["--not-a-flag"]);
    assert_eq!(r.code, 2, "a usage error is 2, not 1 and not 0");
    assert!(r.err.contains("USAGE"), "usage goes to stderr: {}", r.err);
    assert!(r.out.is_empty());
}

/// `--data` with nothing after it used to leave `dir` as `None`, which opened
/// an *in-memory* database and dropped every write on exit.
#[test]
fn option_without_a_value_exits_two() {
    for flag in ["--data", "-q", "--query", "-f", "--file", "--format"] {
        let r = run(&[flag]);
        assert_eq!(r.code, 2, "`{flag}` with no value must be a usage error");
        assert!(r.err.contains("needs a value") || r.err.contains("unknown --format"));
    }
}

#[test]
fn conflicting_and_repeated_options_exit_two() {
    let s = Scratch::new("conflict");
    let f = s.script("x.sql", "SELECT 1;\n");
    assert_eq!(run(&["-q", "SELECT 1", "-f", &p(&f)]).code, 2);
    assert_eq!(run(&["-q", "SELECT 1", "-q", "SELECT 2"]).code, 2);
    assert_eq!(run(&["--format", "yaml"]).code, 2);
    // A value handed to a flag that takes none is dropped silently otherwise.
    assert_eq!(run(&["--no-header=please", "-q", "SELECT 1"]).code, 2);
    assert_eq!(run(&["--nonsense=1"]).code, 2);
}

#[test]
fn help_exits_zero_on_stdout() {
    let r = run(&["--help"]);
    assert_eq!(r.code, 0);
    assert!(r.out.contains("USAGE"));
    assert!(r.out.contains("EXIT STATUS"), "the contract is documented");
    assert!(r.err.is_empty());
}

#[test]
fn missing_script_file_exits_one() {
    let r = run(&["-f", "/nonexistent/definitely/not/here.sql"]);
    assert_eq!(r.code, 1);
    assert!(r.out.is_empty());
}

#[test]
fn piped_stdin_reports_failure() {
    assert_eq!(run_stdin(&[], "SELECT * FROM nope;\n").code, 1);
    let ok = run_stdin(&["--format", "tsv", "--no-header"], "SELECT 7 AS x;\n");
    assert_eq!(ok.code, 0, "stderr: {}", ok.err);
    assert_eq!(ok.out, "7\n");
}

// -------------------------------------------------- lexer-based line splits

/// A `;` inside a string literal is not a statement boundary. The old splitter
/// cut this INSERT in half at the end of the first line and the row was never
/// stored; the run still exited 0.
#[test]
fn semicolon_inside_a_string_literal_is_not_a_boundary() {
    let s = Scratch::new("semi-in-string");
    let f = s.script(
        "in.sql",
        "CREATE TABLE t (id UInt64, s String) ENGINE = MergeTree ORDER BY id;\n\
         INSERT INTO t VALUES (1, 'a;\nb'), (2, 'plain;text');\n\
         SELECT s FROM t ORDER BY id;\n",
    );
    let r = run(&["--data", &db(&s), "-f", &p(&f), "--format", "tsv", "--no-header"]);
    assert_eq!(r.code, 0, "stderr: {}", r.err);
    // The literal newline is escaped by TSV, the `;` survives verbatim.
    assert_eq!(r.out, "a;\\nb\nplain;text\n");
}

/// A `;` inside a `--` comment is not a boundary either. The old splitter saw
/// a line ending in `;` and submitted `SELECT 1 + -- note ;`, a parse error.
#[test]
fn semicolon_inside_a_comment_is_not_a_boundary() {
    let r = run(&["-q", "SELECT 1 + -- one plus ;\n 2 AS n", "--format", "tsv"]);
    assert_eq!(r.code, 0, "stderr: {}", r.err);
    assert_eq!(r.out, "n\n3\n");
}

#[test]
fn semicolon_inside_a_block_comment_is_not_a_boundary() {
    let r = run(&["-q", "SELECT /* a ; b\n c ; d */ 5 AS v", "--format", "tsv"]);
    assert_eq!(r.code, 0, "stderr: {}", r.err);
    assert_eq!(r.out, "v\n5\n");
}

#[test]
fn empty_statements_and_comment_only_input_are_not_errors() {
    for sql in [";;;", "-- nothing at all", "   ", "/* only a comment */"] {
        let r = run(&["-q", sql]);
        assert_eq!(r.code, 0, "{sql:?} should be a no-op, stderr: {}", r.err);
        assert!(r.out.is_empty(), "{sql:?} printed {:?}", r.out);
    }
}

#[test]
fn a_trailing_statement_without_its_semicolon_still_runs() {
    let r = run(&["-q", "SELECT 1 AS a;\nSELECT 2 AS b", "--format", "tsv"]);
    assert_eq!(r.code, 0, "stderr: {}", r.err);
    assert_eq!(r.out, "a\n1\nb\n2\n");
}

/// Unlexable text is still reported, with the right code, and does not hang
/// waiting for a `;` that will never come.
#[test]
fn unterminated_literal_is_a_reported_error() {
    for sql in ["SELECT 'oops", "SELECT 1 /* oops", "SELECT `oops"] {
        let r = run(&["-q", sql]);
        assert_eq!(r.code, 1, "{sql:?}");
        assert!(r.err.contains("SYNTAX_ERROR"), "{sql:?}: {}", r.err);
    }
}

// ------------------------------------------------------------ script policy

/// A script stops at its first error. The statement after the failure must not
/// have run, and the exit status must say so — but the statements *before* it
/// were acknowledged and must still be durable.
#[test]
fn a_script_stops_at_the_first_error_and_keeps_what_committed() {
    let s = Scratch::new("bail");
    let f = s.script(
        "in.sql",
        "CREATE TABLE a (id UInt64) ENGINE = MergeTree ORDER BY id;\n\
         INSERT INTO a VALUES (1);\n\
         SELECT * FROM missing_table;\n\
         CREATE TABLE b (id UInt64) ENGINE = MergeTree ORDER BY id;\n",
    );
    let r = run(&["--data", &db(&s), "-f", &p(&f)]);
    assert_eq!(r.code, 1, "stderr: {}", r.err);

    // A second process, so this reads what actually reached the disk.
    let after = run(&["--data", &db(&s), "-q", ".tables", "--format", "tsv", "--no-header"]);
    assert_eq!(after.code, 0, "stderr: {}", after.err);
    assert!(after.out.contains('a'), "the committed table is gone: {:?}", after.out);
    assert!(!after.out.contains('b'), "the script continued past its error: {:?}", after.out);

    let rows = run(&["--data", &db(&s), "-q", "SELECT id FROM a", "--format", "tsv", "--no-header"]);
    assert_eq!(rows.out, "1\n", "the acknowledged row was not checkpointed");
}

/// Statements before an error keep their output, and it arrives *before* the
/// error on the merged stream — stdout is buffered and stderr is not, so this
/// only holds because stdout is flushed before the error is printed.
#[test]
fn output_and_errors_interleave_in_order() {
    let r = Run::of(
        Command::new("/bin/sh")
            .arg("-c")
            .arg(format!(
                "{BIN:?} --format tsv --no-header -q 'SELECT 1 AS a; SELECT * FROM nope;' 2>&1"
            ))
            .output()
            .expect("spawn sh"),
    );
    let one = r.out.find('1').expect("the first statement's output");
    let err = r.out.find("error [").expect("the error");
    assert!(one < err, "the error jumped ahead of the output: {:?}", r.out);
}

// ------------------------------------------------------- machine-readable output

#[test]
fn tsv_carries_rows_and_nothing_else() {
    let s = Scratch::new("tsv");
    let f = s.script(
        "in.sql",
        "CREATE TABLE t (id UInt64, s String) ENGINE = MergeTree ORDER BY id;\n\
         INSERT INTO t VALUES (1, 'x'), (2, 'y');\n",
    );
    assert_eq!(run(&["--data", &db(&s), "-f", &p(&f), "--format", "tsv"]).out, "",
        "DDL and DML print no prose in a machine format");

    let r = run(&["--data", &db(&s), "-q", "SELECT id, s FROM t ORDER BY id", "--format", "tsv"]);
    assert_eq!(r.out, "id\ts\n1\tx\n2\ty\n");
    assert!(!r.out.contains("rows in"), "no timing footer in a machine format");

    let bare = run(&[
        "--data", &db(&s), "-q", "SELECT id FROM t ORDER BY id", "--format=tsv", "--no-header",
    ]);
    assert_eq!(bare.out, "1\n2\n", "`--flag=value` works and --no-header drops the header");
}

#[test]
fn tsv_escapes_what_would_forge_a_boundary() {
    let r = run(&["-q", "SELECT 'a\tb' AS t, 'c\\\\d' AS u", "--format", "tsv", "--no-header"]);
    assert_eq!(r.code, 0, "stderr: {}", r.err);
    assert_eq!(r.out, "a\\tb\tc\\\\d\n", "one tab, and it is the separator");
    assert_eq!(r.out.matches('\t').count(), 1);
}

#[test]
fn csv_quotes_only_what_needs_it_and_keeps_null_apart_from_empty() {
    let r = run(&[
        "-q",
        "SELECT 'plain' AS a, 'x,y' AS b, 'q\"z' AS c, '' AS d, NULL AS e",
        "--format",
        "csv",
        "--no-header",
    ]);
    assert_eq!(r.code, 0, "stderr: {}", r.err);
    // NULL is nothing at all; the empty string is `""`. That distinction is
    // the only reason to quote an empty field.
    assert_eq!(r.out, "plain,\"x,y\",\"q\"\"z\",\"\",\n");
}

#[test]
fn the_default_format_is_still_the_pretty_table() {
    let r = run(&["-q", "SELECT 1 AS a"]);
    assert_eq!(r.code, 0);
    assert!(r.out.contains('┌') && r.out.contains("1 row in"), "{:?}", r.out);
}

// -------------------------------------------------------------- durability

/// The whole point of `--data`: a script writes, a separate process reads.
#[test]
fn a_script_persists_for_the_next_process() {
    let s = Scratch::new("persist");
    let f = s.script(
        "in.sql",
        "CREATE TABLE t (id UInt64, s String) ENGINE = MergeTree ORDER BY id;\n\
         INSERT INTO t VALUES (1, 'one'), (2, 'two');\n",
    );
    assert_eq!(run(&["--data", &db(&s), "-f", &p(&f)]).code, 0);
    let r = run(&["--data", &db(&s), "-q", "SELECT s FROM t ORDER BY id", "--format", "tsv", "--no-header"]);
    assert_eq!(r.code, 0, "stderr: {}", r.err);
    assert_eq!(r.out, "one\ntwo\n");
}

/// `granular ... | head -1` must not abort the process. `println!` panics on a
/// closed pipe -- "failed printing to stdout: Broken pipe" -- and this crate is
/// built with `panic = "abort"`, so the old shell died on SIGABRT the moment
/// the reader went away. Exit 0, no signal, and the run still reaches its
/// checkpoint.
///
/// The result has to outrun the pipe's own buffer or nothing is ever written
/// to a closed fd and the test proves nothing: 400 rows x 700 bytes is ~280 KB
/// against a 64 KB pipe.
#[test]
fn a_closed_pipe_is_not_a_crash() {
    let s = Scratch::new("pipe");
    let wide = "x".repeat(700);
    let mut sql = String::from(
        "CREATE TABLE t (id UInt64, s String) ENGINE = MergeTree ORDER BY id;\n\
         INSERT INTO t VALUES ",
    );
    for i in 0..400 {
        if i > 0 {
            sql.push(',');
        }
        sql.push_str(&format!("({i},'{wide}')"));
    }
    sql.push_str(";\n");
    let f = s.script("in.sql", &sql);
    assert_eq!(run(&["--data", &db(&s), "-f", &p(&f)]).code, 0);

    // The pipeline's status is `head`'s, so granular's own is echoed out of
    // the subshell. `${PIPESTATUS}` would be shorter and is not portable.
    let r = Run::of(
        Command::new("/bin/sh")
            .arg("-c")
            .arg(format!(
                "{{ {BIN:?} --data {:?} -q 'SELECT s FROM t ORDER BY id' --format tsv; \
                  echo \"granular-exit=$?\" >&2; }} | head -1",
                db(&s)
            ))
            .output()
            .expect("spawn sh"),
    );
    assert!(!r.err.contains("panicked"), "it aborted on the closed pipe: {}", r.err);
    assert!(r.err.contains("granular-exit=0"), "stderr: {}", r.err);
    assert_eq!(r.out, "s\n");
}

// ----------------------------------------------------------- dot commands

#[test]
fn dot_commands_work_in_a_script_and_respect_the_format() {
    let s = Scratch::new("dot");
    let f = s.script(
        "in.sql",
        "CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id;\n\
         .tables\n\
         .quit\n\
         SELECT * FROM never_reached;\n",
    );
    let r = run(&["--data", &db(&s), "-f", &p(&f), "--format", "tsv", "--no-header"]);
    assert_eq!(r.code, 0, ".quit ends the script cleanly; stderr: {}", r.err);
    assert_eq!(r.out, "t\n");
}

#[test]
fn an_unknown_dot_command_says_so_on_stderr() {
    let r = run_stdin(&[], ".nonsense\n");
    assert!(r.err.contains("unknown command"), "stderr: {}", r.err);
    assert!(r.out.is_empty());
}
