//! `granular` — a SQL shell over the engine.
//!
//! ```text
//!   granular                          in-memory REPL
//!   granular --data ./db              persistent REPL
//!   granular -q "SELECT 1"            one shot
//!   granular --data ./db -f setup.sql run a script
//!   echo "SELECT 1" | granular        piped input
//!   granular -q "SELECT ..." --format tsv | cut -f2
//! ```
//!
//! ## The contract this file owes a shell
//!
//! Four things, none of which it used to honour.
//!
//! **The exit status is the truth.** 0 only when every statement succeeded, 1
//! when one failed or the database could not be opened, written or
//! checkpointed, 2 when the command line was wrong. Every error path used to
//! print a message and exit 0, which made `granular -q ... && deploy` deploy
//! on a syntax error and made the binary unusable from a Makefile, a CI job or
//! any pipeline that tests `$?`.
//!
//! **Statements end at a `;` the *lexer* found.** The old splitter took any
//! line whose trimmed end was `;`, so `INSERT INTO t VALUES ('a;b')` was cut
//! in half and a `-- drop this ;` comment ended a statement early. Boundaries
//! now come from [`granular::sql::lexer::tokenize`], the same tokenizer
//! `Session::run_mixed` splits transactions with — the only way to agree with
//! the parser about what a boundary is, is to ask the thing that parses.
//!
//! **Results go to stdout and errors to stderr**, with stdout flushed before
//! anything reaches stderr so a script's output and its error interleave in
//! the order they happened.
//!
//! **There is a machine-readable output.** `--format tsv|csv` prints result
//! rows and nothing else — no `Ok.`, no timing footer, no box drawing — so the
//! binary can be piped into `cut`, `awk` or a test harness. The format is
//! chosen once per result, never per cell.

use std::fs::File;
use std::io::{self, BufRead, BufWriter, IsTerminal, StdoutLock, Write};

use granular::sql::lexer::{tokenize, Token};
use granular::{Error, ResultSet, Session, Value};

const HELP: &str = "\
granular — hybrid OLAP + OLTP database

USAGE:
    granular [OPTIONS]

OPTIONS:
    --data <DIR>     open a persistent database in DIR (default: in-memory)
    --read-only      open --data under a shared lock: queries only, no writes,
                     no checkpoint, and several of these (or a live writer) may
                     hold the same directory at once
    -q, --query SQL  run SQL and exit
    -f, --file PATH  run the statements in PATH and exit
    --format FMT     output as `table` (default), `tsv` or `csv`
    --no-header      omit the header row from tsv/csv output
    -h, --help       show this message

REPL COMMANDS:
    .help            this message
    .tables          list tables
    .schema TABLE    show a table's DDL
    .stats TABLE     compression and index footprint
    .quit / .exit    leave (Ctrl-D also works)

EXIT STATUS:
    0  every statement succeeded
    1  a statement failed, or the database could not be opened or written
    2  the command line was wrong

Statements are terminated by `;`, located with the SQL lexer: a `;` inside a
string literal or a `--` comment does not end one. A statement may span lines.
Scripts (-q, -f, piped stdin) stop at the first error; `tsv` and `csv` print
result rows only, so their output pipes into anything.";

/// 0 fine, 1 the database said no, 2 the command line was wrong. The
/// convention every shell, Makefile and CI runner already assumes.
const EXIT_OK: i32 = 0;
const EXIT_FAIL: i32 = 1;
const EXIT_USAGE: i32 = 2;

fn main() {
    std::process::exit(real_main());
}

fn real_main() -> i32 {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&argv) {
        Ok(Some(a)) => a,
        Ok(None) => return EXIT_OK, // --help
        Err(msg) => {
            eprintln!("granular: {msg}\n\n{HELP}");
            return EXIT_USAGE;
        }
    };

    // Opened before the database so a typo in the path does not leave a
    // freshly created data directory behind. Opened, not read: the script is
    // streamed a line at a time, so its size stops being resident memory.
    // Measured on a 93 MB script whose statements are terminated normally,
    // peak RSS 2.9 MB against the slurping version's 103 MB. The pending
    // buffer still holds one *statement*, so the floor is the longest
    // statement, not the longest line.
    let script = match &args.file {
        Some(p) => match File::open(p) {
            Ok(f) => Some(f),
            Err(e) => {
                eprintln!("granular: {p}: {e}");
                return EXIT_FAIL;
            }
        },
        None => None,
    };

    let opened = match (&args.dir, args.read_only) {
        (Some(d), false) => Session::open(d),
        // A shared directory lock and every mutation refused by name. There is
        // no read-only in-memory database to open: `parse_args` has already
        // refused the combination.
        (Some(d), true) => Session::open_read_only(d),
        (None, _) => Ok(Session::in_memory()),
    };
    let mut session = match opened {
        Ok(s) => s,
        Err(e) => return die(&e),
    };

    // A terminal on stdin with no work named on the command line is the only
    // interactive case; everything else is a script and stops at its first
    // error, the way `psql -v ON_ERROR_STOP=1` and `sqlite3 -bail` do. A
    // script's statements depend on each other, so ploughing on past a failed
    // CREATE buries the one real error under a cascade of derivative ones and
    // half-applies the migration. The prompt keeps going instead: an error
    // there has already been seen and recovered from. It still counts toward
    // the exit status, so there is one rule to state and one to test -- every
    // mode reports failure, only the stopping differs.
    let interactive = args.query.is_none() && script.is_none() && io::stdin().is_terminal();

    let mut code = EXIT_OK;
    {
        let mut sh = Shell::new(&mut session, Out::new(args.fmt, args.header), !interactive);
        // All four modes are the same loop over lines, differing only in where
        // the lines come from -- which is why `;` means exactly the same thing
        // typed, piped, passed to `-q` and read from a file.
        let r = if let Some(sql) = &args.query {
            sh.stream(sql.as_bytes())
        } else if let Some(f) = script {
            sh.stream(io::BufReader::new(f))
        } else if !interactive {
            sh.stream(io::stdin().lock())
        } else {
            repl(&mut sh, args.dir.is_some())
        };
        if let Err(e) = r {
            eprintln!("granular: {e}");
            code = EXIT_FAIL;
        }
        // Explicit, because `BufWriter`'s `Drop` flush swallows its error: a
        // full disk on the last 64 KiB has to reach the exit status, not
        // vanish into a destructor.
        sh.out.flush();
        if sh.failed || sh.out.err {
            code = EXIT_FAIL;
        }
    }

    // Checkpointed even after a failure: the statements that did succeed were
    // acknowledged, and dropping them here would turn a reported error into
    // silent data loss. `Session` deliberately has no `Drop` checkpoint, so
    // this call is the only one.
    //
    // Except on a read-only session, which refuses `CHECKPOINT` by design --
    // unguarded, this line would `die` on every `--read-only` run that had
    // just answered every query correctly.
    if !session.is_read_only() {
        if let Err(e) = session.checkpoint() {
            die(&e);
            code = EXIT_FAIL;
        }
    }
    code
}

/// Report an engine error on stderr and hand back the failure status, so the
/// caller reads as `return die(&e)`.
fn die(e: &Error) -> i32 {
    eprintln!("error [{}]: {e}", e.code());
    EXIT_FAIL
}

// ------------------------------------------------------------- command line

#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Table,
    Tsv,
    Csv,
}

struct Args {
    dir: Option<String>,
    query: Option<String>,
    file: Option<String>,
    fmt: Format,
    header: bool,
    read_only: bool,
}

/// Parse the command line. `Ok(None)` means `--help` was asked for and
/// printed; `Err` carries the message and earns [`EXIT_USAGE`].
fn parse_args(argv: &[String]) -> Result<Option<Args>, String> {
    let mut a = Args {
        dir: None,
        query: None,
        file: None,
        fmt: Format::Table,
        header: true,
        read_only: false,
    };

    let mut i = 0;
    while i < argv.len() {
        // `--flag=value` as well as `--flag value`. Only long flags split:
        // `-q` takes SQL, and SQL is full of `=`.
        let (name, inline) = match argv[i].split_once('=') {
            Some((n, v)) if n.starts_with("--") => (n, Some(v)),
            _ => (argv[i].as_str(), None),
        };
        // A missing value used to leave the option unset, so `granular --data`
        // silently opened an *in-memory* database and threw away every write
        // at exit -- the same lie as exiting 0 on an error, one layer up.
        let value = |i: &mut usize| -> Result<String, String> {
            match inline {
                Some(v) => Ok(v.to_string()),
                None => {
                    *i += 1;
                    argv.get(*i).cloned().ok_or_else(|| format!("`{name}` needs a value"))
                }
            }
        };
        // Repeats are rejected rather than last-one-wins: silently running one
        // of the two queries the user asked for is the defect this file exists
        // to stop making.
        let once = |slot: &mut Option<String>, v: String| -> Result<(), String> {
            match slot {
                Some(_) => Err(format!("`{name}` given twice")),
                None => {
                    *slot = Some(v);
                    Ok(())
                }
            }
        };

        match name {
            // `--no-header=please` would otherwise be accepted and the value
            // dropped, which is the pattern this whole file is here to stop.
            "-h" | "--help" | "--no-header" if inline.is_some() => {
                return Err(format!("`{name}` takes no value"))
            }
            "-h" | "--help" => {
                println!("{HELP}");
                return Ok(None);
            }
            "--data" => once(&mut a.dir, value(&mut i)?)?,
            "-q" | "--query" => once(&mut a.query, value(&mut i)?)?,
            "-f" | "--file" => once(&mut a.file, value(&mut i)?)?,
            "--format" => {
                a.fmt = match value(&mut i)?.as_str() {
                    "table" => Format::Table,
                    "tsv" => Format::Tsv,
                    "csv" => Format::Csv,
                    o => return Err(format!("unknown --format `{o}`: want table, tsv or csv")),
                }
            }
            "--no-header" => a.header = false,
            "--read-only" => a.read_only = true,
            // `argv[i]`, not `name`: the message should quote what was typed,
            // `=value` and all.
            _ => return Err(format!("unknown argument `{}`", argv[i])),
        }
        i += 1;
    }

    if a.query.is_some() && a.file.is_some() {
        // The old parser let -f win and dropped -q without a word.
        return Err("-q and -f are mutually exclusive".into());
    }
    if a.read_only && a.dir.is_none() {
        // An in-memory database that refuses writes can only ever be empty,
        // so the flag is silently useless there -- which is the class of quiet
        // wrongness this binary reports instead.
        return Err("--read-only needs --data: an in-memory database has nothing to read".into());
    }
    Ok(Some(a))
}

// ---------------------------------------------------------------- rendering

type Sink = BufWriter<StdoutLock<'static>>;

/// Everything written to stdout goes through here.
///
/// One `BufWriter` over a held `StdoutLock` for the whole run. `println!`
/// takes the stdout mutex per call and `Stdout`'s `LineWriter` flushes at
/// every newline, so rendering an N-row table used to cost N write syscalls
/// and N lock acquisitions; buffered, it costs one syscall per 64 KiB.
///
/// That is the single biggest win in this file. A/B interleaved, best-of-N,
/// `--release`, stdout redirected to a file: 200k rows x 2 columns 306 ms
/// against 2508 ms (0.122), 20k rows x 1 column 65.5 ms against 213 ms
/// (0.308), and a 1-row result 163 ms against 157 ms (1.03 -- inside the
/// noise, so nothing regressed at the small end where process startup and the
/// checkpoint dominate).
struct Out {
    w: Sink,
    fmt: Format,
    header: bool,
    /// The reader closed the pipe (`| head -1`). Not a failure: stop writing,
    /// let the run reach its checkpoint, exit 0. Restoring `SIGPIPE` to
    /// `SIG_DFL` instead would be more unix-like and is deliberately not done
    /// -- dying mid-script would skip that checkpoint and lose acknowledged
    /// writes to a `| head`.
    gone: bool,
    /// A write failed for a reason that is not a closed pipe -- a full disk.
    /// That one is a failure and must reach the exit status.
    err: bool,
}

impl Out {
    fn new(fmt: Format, header: bool) -> Out {
        // 64 KiB: one page-aligned syscall per ~1500 rendered rows.
        Out {
            w: BufWriter::with_capacity(64 << 10, io::stdout().lock()),
            fmt,
            header,
            gone: false,
            err: false,
        }
    }

    /// Still worth writing to?
    fn alive(&self) -> bool {
        !self.gone && !self.err
    }

    fn put(&mut self, r: io::Result<()>) {
        match r {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::BrokenPipe => self.gone = true,
            Err(e) => {
                self.err = true;
                eprintln!("granular: stdout: {e}");
            }
        }
    }

    fn flush(&mut self) {
        if self.alive() {
            let r = self.w.flush();
            self.put(r);
        }
    }

    /// A whole line of prose (the banner, `.help`, `.stats`). Generic over
    /// `Display` so callers can hand it `format_args!` and render into the
    /// buffer directly, with no intermediate `String`.
    fn line<D: std::fmt::Display>(&mut self, d: D) {
        if self.alive() {
            let r = writeln!(self.w, "{d}");
            self.put(r);
        }
    }

    fn prompt(&mut self, s: &str) {
        if self.alive() {
            let r = write!(self.w, "{s}");
            self.put(r);
            self.flush(); // a prompt the reader cannot see is not a prompt
        }
    }

    /// The format dispatch happens exactly here -- once per result set, never
    /// per row and never per cell. `esc` is a generic parameter rather than a
    /// `fn` pointer so each format monomorphizes into its own inlined loop.
    fn result(&mut self, rs: &ResultSet) {
        if !self.alive() {
            return;
        }
        let r = match self.fmt {
            Format::Table => writeln!(self.w, "{rs}"),
            Format::Tsv => self.rows(rs, b'\t', tsv_field),
            Format::Csv => self.rows(rs, b',', csv_field),
        };
        self.put(r);
    }

    fn rows<E: Fn(&mut Sink, &str) -> io::Result<()>>(
        &mut self,
        rs: &ResultSet,
        sep: u8,
        esc: E,
    ) -> io::Result<()> {
        // Machine formats carry rows and nothing else: no `Ok.` for DDL, no
        // `N rows in X ms` footer. A consumer parsing the stream must not have
        // to filter prose out of it.
        if rs.schema.is_empty() {
            return Ok(());
        }
        if self.header {
            for (i, f) in rs.schema.fields().iter().enumerate() {
                if i > 0 {
                    self.w.write_all(&[sep])?;
                }
                esc(&mut self.w, &f.name)?;
            }
            self.w.write_all(b"\n")?;
        }
        for b in &rs.blocks {
            let width = b.width();
            for r in 0..b.rows() {
                for c in 0..width {
                    if c > 0 {
                        self.w.write_all(&[sep])?;
                    }
                    // `Column::value` costs a 24-byte stack `Value` and, for
                    // strings, one `Arc` refcount bump -- no allocation on the
                    // integer and string paths, which is every wide dump. The
                    // four variants below it have no writer-shaped renderer on
                    // `Value`, so they pay one `String` per cell; see
                    // `render_plain`.
                    //
                    // `write!` through `core::fmt` for the integers, not a
                    // hand-rolled decimal writer, and the measurement is why:
                    // 200k rows x 2 columns takes 231 ms as tsv against a
                    // 195 ms floor for the same command returning one row, so
                    // materializing *and* rendering 400k cells is 36 ms of it.
                    // Nothing in there is worth more than a couple of percent
                    // of the command.
                    match b.column(c).value(r) {
                        // An unquoted empty field is NULL and a quoted one is
                        // the empty string: the only way CSV can tell them
                        // apart. TSV cannot, and does not pretend to.
                        Value::Null => {}
                        Value::Bool(t) => {
                            self.w.write_all(if t { b"true" } else { b"false" })?
                        }
                        Value::UInt(u) => write!(self.w, "{u}")?,
                        Value::Int(n) => write!(self.w, "{n}")?,
                        Value::Str(s) => esc(&mut self.w, &s)?,
                        other => write!(self.w, "{}", other.render_plain())?,
                    }
                }
                self.w.write_all(b"\n")?;
            }
        }
        Ok(())
    }
}

/// TSV has no quoting, so the bytes that would forge a row or column boundary
/// are backslash-escaped the way `sqlite3 -tsv` and `mysqldump` do. One scan,
/// and a field containing none of them -- essentially all of them -- leaves as
/// a single `write_all` of the original slice.
fn tsv_field(w: &mut Sink, s: &str) -> io::Result<()> {
    let b = s.as_bytes();
    let mut last = 0;
    for (i, &c) in b.iter().enumerate() {
        let e: &[u8] = match c {
            b'\t' => b"\\t",
            b'\n' => b"\\n",
            b'\r' => b"\\r",
            b'\\' => b"\\\\",
            _ => continue,
        };
        w.write_all(&b[last..i])?;
        w.write_all(e)?;
        last = i + 1;
    }
    w.write_all(&b[last..])
}

/// RFC 4180: quote only a field that would otherwise be ambiguous, and double
/// any `"` inside it. The empty string is quoted so it stays distinguishable
/// from NULL, which is written as nothing at all.
fn csv_field(w: &mut Sink, s: &str) -> io::Result<()> {
    let b = s.as_bytes();
    if !b.is_empty() && !b.iter().any(|c| matches!(c, b',' | b'"' | b'\n' | b'\r')) {
        return w.write_all(b);
    }
    w.write_all(b"\"")?;
    let mut last = 0;
    for (i, &c) in b.iter().enumerate() {
        if c == b'"' {
            // Through the quote, then again from it: that is the doubling.
            w.write_all(&b[last..=i])?;
            last = i;
        }
    }
    w.write_all(&b[last..])?;
    w.write_all(b"\"")
}

// ------------------------------------------------------- statement plumbing

/// What the caller should do after feeding a line.
#[derive(PartialEq, Eq)]
enum Flow {
    Go,
    /// `.quit`, a closed pipe, or -- in a script -- the first error.
    Quit,
}

/// The line-to-statement machine shared by every input mode, so a `;` means
/// the same thing typed, piped and read from a file.
struct Shell<'a> {
    session: &'a mut Session,
    out: Out,
    /// Text accepted since the last statement boundary.
    buf: String,
    /// Byte ranges of the complete statements found in `buf`. A field, not a
    /// local, so the driver allocates one `Vec` per process rather than one
    /// per line.
    spans: Vec<(usize, usize)>,
    /// A `;` has arrived that no lex has yet accounted for.
    dirty: bool,
    /// Any statement failed. Decides the exit status.
    failed: bool,
    /// Stop at the first error (scripts) or carry on (the prompt).
    bail: bool,
}

impl<'a> Shell<'a> {
    fn new(session: &'a mut Session, out: Out, bail: bool) -> Shell<'a> {
        Shell {
            session,
            out,
            buf: String::new(),
            spans: Vec::new(),
            dirty: false,
            failed: false,
            bail,
        }
    }

    /// Run every statement `r` yields.
    ///
    /// A line at a time, with one reused `String`, so a script costs its
    /// longest line and not its length. `&[u8]` is a `BufRead`, which is how
    /// `-q` shares this loop without a file behind it.
    fn stream<R: BufRead>(&mut self, mut r: R) -> io::Result<()> {
        let mut line = String::new();
        loop {
            line.clear();
            if r.read_line(&mut line)? == 0 {
                break;
            }
            if self.feed(&line) == Flow::Quit {
                return Ok(());
            }
        }
        self.finish();
        Ok(())
    }

    /// Accept one line, with or without its newline.
    fn feed(&mut self, line: &str) -> Flow {
        if self.buf.trim().is_empty() {
            let t = line.trim();
            // Dot commands are line commands, and only at the start of a
            // statement -- `t.col` continued onto its own line is still SQL.
            if t.starts_with('.') {
                self.buf.clear();
                return self.dot(t);
            }
            if t.is_empty() {
                self.buf.clear(); // keep blank lines out of the prompt state
                return Flow::Go;
            }
        }
        self.buf.push_str(line);
        if !line.ends_with('\n') {
            self.buf.push('\n'); // a `--` comment must not swallow what follows
        }
        // Lexing is the expensive half of finding a boundary, so it is gated
        // on a byte that could be one: a 10k-line `INSERT ... VALUES` with no
        // interior `;` is lexed once, at its last line, instead of once per
        // line. `dirty` stays set through a [`Split::Open`] answer, so a `;`
        // that landed inside an unclosed literal is re-examined on every line
        // until the literal closes -- the only case that re-lexes per line,
        // and the only case where it is necessary.
        self.dirty |= line.as_bytes().contains(&b';');
        if !self.dirty {
            return Flow::Go;
        }
        self.drain()
    }

    /// Run the complete statements sitting in `buf` and keep the rest.
    fn drain(&mut self) -> Flow {
        // Taken out so the statement slices borrow a local while `run_sql`
        // borrows `self`. A move, not a copy, and the buffer's capacity is
        // handed back at the end of the call.
        let mut text = std::mem::take(&mut self.buf);
        let flow = match complete_statements(&text, &mut self.spans) {
            Split::At(cut) => {
                self.dirty = false;
                let mut flow = Flow::Go;
                for k in 0..self.spans.len() {
                    let (s, e) = self.spans[k];
                    if self.run_sql(&text[s..e]) == Flow::Quit {
                        // Nothing after this point is going to run; drop it so
                        // `finish` does not run it either.
                        text.clear();
                        flow = Flow::Quit;
                        break;
                    }
                }
                if flow == Flow::Go {
                    text.drain(..cut);
                }
                flow
            }
            // Still inside a literal or a comment. Any `;` already buffered is
            // unaccounted for, so `dirty` stays set and the next line re-lexes
            // whether or not it brings a `;` of its own -- a boundary is
            // deferred here, never lost.
            Split::Open => Flow::Go,
            // Unlexable, and no further line can change that, so hand it to
            // the parser now: at a prompt the caret arrives while the typo is
            // still on screen, and in a script the run stops at the statement
            // that is actually wrong rather than at the next `;`.
            Split::Bad => {
                self.dirty = false;
                let flow = self.run_sql(&text);
                text.clear();
                flow
            }
        };
        self.buf = text;
        flow
    }

    /// A trailing statement with no `;` still runs: scripts that end without
    /// one are ordinary, and dropping the last statement in silence is the
    /// same class of lie as exiting 0 on an error.
    fn finish(&mut self) {
        if self.buf.trim().is_empty() {
            self.buf.clear();
            return;
        }
        let text = std::mem::take(&mut self.buf);
        self.run_sql(&text);
        self.buf = text;
        self.buf.clear();
    }

    fn pending(&self) -> bool {
        !self.buf.trim().is_empty()
    }

    fn run_sql(&mut self, sql: &str) -> Flow {
        match self.session.run(sql) {
            Ok(results) => {
                for r in &results {
                    self.out.result(r);
                }
                if self.out.alive() {
                    Flow::Go
                } else {
                    Flow::Quit
                }
            }
            Err(e) => self.fail(&e),
        }
    }

    fn fail(&mut self, e: &Error) -> Flow {
        self.failed = true;
        // stdout is buffered and stderr is not, so without this the output of
        // the statements that worked would appear *after* the error that
        // stopped them.
        self.out.flush();
        eprintln!("error [{}]: {e}", e.code());
        if self.bail {
            Flow::Quit
        } else {
            Flow::Go
        }
    }

    fn dot(&mut self, cmd: &str) -> Flow {
        let mut parts = cmd.splitn(2, char::is_whitespace);
        let head = parts.next().unwrap_or("");
        let arg = parts.next().unwrap_or("").trim();

        let r = match head {
            ".quit" | ".exit" => return Flow::Quit,
            ".help" => {
                self.out.line(HELP);
                Ok(())
            }
            ".tables" => self.session.query("SHOW TABLES").map(|rs| self.out.result(&rs)),
            ".schema" if arg.is_empty() => {
                eprintln!("usage: .schema TABLE");
                Ok(())
            }
            ".schema" => self
                .session
                .query(&format!("SHOW CREATE TABLE {arg}"))
                .map(|rs| self.out.result(&rs)),
            ".stats" if arg.is_empty() => {
                eprintln!("usage: .stats TABLE");
                Ok(())
            }
            ".stats" => self.stats(arg),
            other => {
                eprintln!("unknown command `{other}` — try .help");
                Ok(())
            }
        };
        match r {
            Ok(()) if self.out.alive() => Flow::Go,
            Ok(()) => Flow::Quit,
            Err(e) => self.fail(&e),
        }
    }

    fn stats(&mut self, arg: &str) -> granular::Result<()> {
        use granular::sql::ast::ObjectName;
        self.session.catalog.flush_all()?;
        let name = ObjectName(arg.split('.').map(|s| s.to_string()).collect());
        let t = self.session.catalog.table(&name)?;
        let report = t.compression_report();
        let (parts, delta) = (t.part_count(), t.delta_len());
        self.out.line(report);
        self.out.line(format_args!("  parts: {parts}, buffered writes: {delta}"));
        Ok(())
    }
}

/// What a buffer holds.
enum Split {
    /// Boundaries are known; `spans` is filled and the buffer should be drained
    /// to this offset. Possibly `At(0)` — well-formed text with no `;` yet.
    At(usize),
    /// The buffer ends inside an unclosed literal or comment, so no boundary in
    /// it can be trusted until more input arrives.
    Open,
    /// Unlexable, and no further input will change that.
    Bad,
}

/// Locate the complete statements at the front of `buf`.
///
/// Fills `spans` with the byte range of each one — first token to the `;` that
/// ends it, exclusive, which is what `parse` wants.
///
/// The split is the tokenizer's, not a scan for `;`, for the reason
/// `Session::run_mixed` gives: a semicolon inside a string literal or a comment
/// is not a boundary, and the only way to agree with the parser about that is
/// to ask the same lexer. `tokenize` re-lexes text that `Session::run` will lex
/// again — the same trade `run_mixed` already makes, and it does not show up:
/// A/B interleaved against the old line-based splitter, best-of-5, a 200k-tuple
/// `INSERT` script through `-f`, 632 ms against 637 ms with string literals
/// (0.992) and 794 ms against 915 ms without (0.868). Parsing and ingesting
/// those tuples costs orders of magnitude more than lexing them twice.
///
/// So the obvious optimization — a `mentions_txn_keyword`-style byte prefilter
/// that proves a buffer holds no quote and no comment opener, in which case
/// every `;` in it is provably top level and the lex can be skipped — was
/// measured and is not worth its lines. Recorded so nobody tries it again.
fn complete_statements(buf: &str, spans: &mut Vec<(usize, usize)>) -> Split {
    spans.clear();
    let toks = match tokenize(buf) {
        Ok(t) => t,
        Err(e) => return if extendable(buf, &e) { Split::Open } else { Split::Bad },
    };
    let mut start: Option<usize> = None;
    let mut cut = 0usize;
    for t in &toks {
        if t.tok == Token::Semicolon {
            // `;;` and a trailing `;` produce no statement, only a new cut.
            if let Some(s) = start.take() {
                spans.push((s, t.pos));
            }
            cut = t.pos + 1;
        } else if start.is_none() {
            start = Some(t.pos);
        }
    }
    Split::At(cut)
}

/// Could another line still turn this lex error into a valid statement?
///
/// Two ways, and only two. A quote or `/*` left open runs off the end of the
/// buffer by definition and the next line can close it. And a byte the lexer
/// rejects *as the last thing in the buffer* may be half of a two-byte
/// operator — `|` before `|`, `!` before `=`, `:` before `:` — which is what a
/// statement broken across a line inside `||` looks like. Everything else is
/// about text that is already complete, and waiting for more of it would hang
/// the prompt with the answer already known.
fn extendable(buf: &str, e: &Error) -> bool {
    let Error::Parse { pos, .. } = e else { return false };
    let tail = &buf.as_bytes()[(*pos).min(buf.len())..];
    matches!(tail.first(), Some(b'\'' | b'"' | b'`' | b'/')) || tail.trim_ascii_end().len() <= 1
}

// --------------------------------------------------------------------- repl

fn repl(sh: &mut Shell, persistent: bool) -> io::Result<()> {
    sh.out.line(format_args!(
        "granular {} — {} database. `.help` for commands, `.quit` to exit.",
        env!("CARGO_PKG_VERSION"),
        if persistent { "persistent" } else { "in-memory" }
    ));

    let stdin = io::stdin();
    // Both the lock and the line buffer are hoisted out of the loop: one mutex
    // acquisition and one allocation for the whole session instead of two per
    // line typed.
    let mut input = stdin.lock();
    let mut line = String::new();

    loop {
        sh.out.prompt(if sh.pending() { "     ...> " } else { "granular> " });
        line.clear();
        if input.read_line(&mut line)? == 0 {
            sh.out.line(""); // EOF: leave the cursor on its own line
            break;
        }
        if sh.feed(&line) == Flow::Quit {
            return Ok(());
        }
    }

    sh.finish();
    Ok(())
}
