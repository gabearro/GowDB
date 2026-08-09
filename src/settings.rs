//! Named, typed, documented runtime settings — and the three statements that
//! reach them from SQL.
//!
//! ## Why this module exists
//!
//! `MemTracker`, the per-query deadline and the cancel flag all landed
//! working, with `Session::set_memory_limit` / `set_timeout` /
//! `cancel_handle` sitting on the facade — and *nothing that speaks SQL could
//! touch any of them*. A user with the CLI in front of them had no way to cap
//! a query, and `SELECT ... SETTINGS max_memory_usage = ...` was refused with
//! "there is no per-query memory accounting", a note that had been false since
//! the governance work shipped. So the capability existed twice over (built,
//! and documented as absent) and was reachable zero times.
//!
//! ## The shape
//!
//! One [`SPECS`] table is the whole registry: name, type, default, which field
//! it writes and one line of prose. Every entry point reads it —
//! [`Settings::set`] for `SET`, [`show`] for `SHOW SETTINGS`, the parser's
//! query-level `SETTINGS` check, and the `SETTINGS` clause on an import — so a
//! setting is added in exactly one place and cannot be accepted by one path
//! and rejected by another.
//!
//! **Unknown names are refused.** ClickHouse has ~1000 settings; this engine
//! implements seven. Accepting a name it does not implement is the precise
//! defect this phase exists to stop: the query then runs with something other
//! than what it asked for and says nothing. A name that is recognised but not
//! implemented gets a message naming what this engine does instead
//! (`sql::parser::SETTINGS_TABLE` holds those); a name nothing knows is
//! reported as a typo, with an offset.
//!
//! ## Reaching a `Session`
//!
//! [`Handle`] is the one hook. `Session::run` needs a single line —
//!
//! ```text
//! if let Some(r) = self.settings.clone().intercept(self, sql) { return r; }
//! ```
//!
//! — and the clone is what makes it one line: it ends the borrow of `self`
//! before `intercept` takes `&mut Session`. The state is behind an `Arc` for
//! the same reason, at a cost of two atomics per *statement* (against a
//! statement that is about to be lexed, parsed, bound and lowered), and
//! `intercept` returns `None` after a byte sniff for everything that is not
//! one of its three statements.
//!
//! A process-global registry was tried on paper and rejected: two `Session`s
//! in one process (which `Db`, and every test file here, create) would then
//! report each other's values from `SHOW SETTINGS`, which is exactly the class
//! of quiet wrongness the module is here to remove.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::common::{Error, Result};
use crate::session::{ResultSet, Session};
use crate::sql::lexer::{tokenize, Spanned, Token};
use crate::types::{Block, Column, DataType, Field, Schema};

// --------------------------------------------------------------- the values

/// One session's settings.
///
/// Sixteen bytes of numbers, a byte, a bool and one `Box<str>` — small enough
/// to clone into a `ResultSet` renderer without anybody thinking about it, and
/// small enough that the `Arc<Mutex<..>>` in [`Handle`] is about sharing
/// rather than about size.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Ceiling on one query's intermediate state. Also the *spill* trigger:
    /// the external sort and the external `GROUP BY` start writing runs when
    /// this budget refuses them, so there is deliberately no second threshold
    /// setting to keep consistent with it.
    pub max_memory_usage: i64,
    /// Ceiling on what one query may write to the spill directory. 0 -- the
    /// default -- means no ceiling, the same convention `max_execution_time`
    /// uses, and it is also the value that keeps the counter untouched: the
    /// charge returns before it reads the atomic.
    ///
    /// Separate from `max_memory_usage` because spilling *amplifies*: an 8 MiB
    /// budget measured 272 MB on disk, 34x, with nothing to stop it.
    pub max_temporary_data_on_disk: u64,
    /// Per-statement wall clock. 0 means no deadline, which is the only value
    /// that keeps `Instant::now` out of the operator loop entirely.
    pub max_execution_time_ms: u64,
    /// Rows per block the streaming importer hands to storage. Above
    /// `storage::table::BULK_INSERT_THRESHOLD` (4096) each block is packed
    /// straight into a part instead of being buffered in the delta, which is
    /// what keeps a 10 GB import's resident set flat.
    pub max_insert_block_size: u32,
    /// Malformed input rows to skip before an import gives up. 0 — the
    /// default — means the first bad row fails the statement, naming the line.
    pub input_format_allow_errors_num: u32,
    /// Field separator for CSV read and write. TSV is this set to `\t`.
    pub format_csv_delimiter: u8,
    /// The text that means NULL on input, and is written for NULL on output.
    ///
    /// Empty by default, which is not an arbitrary choice: it is exactly the
    /// convention the CLI's `--format csv` writer already uses ("an unquoted
    /// empty field is NULL and a quoted one is the empty string: the only way
    /// CSV can tell them apart"), so a table dumped by the binary imports back
    /// through this module without a setting. `\N` is one `SET` away for
    /// files that come from somewhere else.
    pub format_csv_null: Box<str>,
    /// Read the first line of an import as column names and match by name
    /// rather than by position.
    pub input_format_with_names_use_header: bool,
    /// Write-ahead log bytes, per table, that trigger an automatic fold into
    /// parts at the next statement boundary. 0 disables it, which is what the
    /// engine did unconditionally before this existed: nothing auto-
    /// checkpointed, so a long-running writer grew `wal.log` without bound and
    /// wrote no part file at all until a DDL, a BACKUP, an import or process
    /// exit happened to call one.
    pub wal_fold_bytes: u64,
    /// Archived write-ahead log bytes to keep under `<data>/.wal-archive`.
    ///
    /// The value that governs is a process-wide static in `persist::wal` --
    /// the archive tick runs where no `Settings` is in scope -- and this field
    /// is the session's *mirror* of it: [`Handle::snapshot`] seeds it from the
    /// static and [`Settings::apply_to`] pushes it back. That is not
    /// bookkeeping for its own sake. Without a field, `set` wrote the static
    /// directly, so this was the one setting a statement-scoped `SETTINGS`
    /// clause could not un-scope: `SELECT count() FROM t SETTINGS
    /// wal_archive_retention='1'` trimmed the archive to one segment and left
    /// every later statement in the process at that value. Seeding from the
    /// static rather than from `default()` is what keeps a second session in
    /// the same process from pushing its default over the first one's `SET`.
    pub wal_archive_retention: u64,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            max_memory_usage: crate::exec::operators::DEFAULT_MEM_BUDGET,
            max_temporary_data_on_disk: 0,
            max_execution_time_ms: 0,
            // 64k rows: 16x `BULK_INSERT_THRESHOLD`, so every block bypasses
            // the delta, and small enough that one block of a 40-column table
            // is a few MB.
            max_insert_block_size: 65_536,
            input_format_allow_errors_num: 0,
            format_csv_delimiter: b',',
            format_csv_null: Box::from(""),
            input_format_with_names_use_header: true,
            wal_fold_bytes: crate::persist::wal::DEFAULT_FOLD_BYTES,
            wal_archive_retention: crate::persist::wal::DEFAULT_ARCHIVE_BYTES,
        }
    }
}

/// Which field of [`Settings`] a spec writes.
///
/// The table carries this rather than a `fn(&mut Settings, ..)` pointer so
/// that `set` and `get` are two matches on one enum that the compiler checks
/// for exhaustiveness — a new setting cannot be added to the table and
/// forgotten by the writer, which is how registries usually rot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Slot {
    MaxMemory,
    MaxTempDisk,
    MaxTime,
    InsertBlock,
    AllowErrors,
    CsvDelim,
    CsvNull,
    CsvHeader,
    FoldBytes,
    /// The one slot that writes nothing on `Settings`. The archive's retention
    /// budget is a *process* global (one `.wal-archive` directory per data
    /// directory, and the archiver that trims it has no session), so it lives
    /// in the static it has always lived in and this row is only the name that
    /// reaches it. Keeping it off `Settings` also keeps `Settings` the size it
    /// is, which matters because it is cloned into every `SHOW SETTINGS`.
    WalRetention,
}

/// How a setting's text is read, and how its value is written back.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// A byte count, with an optional `K`/`M`/`G`/`T` suffix (binary: `1K` is
    /// 1024). Rendered back with the suffix, because `8589934592` in a
    /// `SHOW SETTINGS` column tells nobody it is 8 GiB.
    Bytes,
    /// Seconds, fractional allowed (`0.25`). Stored as milliseconds.
    Seconds,
    /// A row count.
    Rows,
    /// `0`/`1`, `true`/`false`, `on`/`off`, `yes`/`no`.
    Bool,
    /// Exactly one byte. `'\t'` and `\t` both name the tab.
    Char,
    /// Free text, taken verbatim from a quoted literal.
    Text,
}

/// One row of the registry.
pub struct Spec {
    pub name: &'static str,
    pub kind: Kind,
    slot: Slot,
    pub doc: &'static str,
}

/// Every setting this engine actually applies.
///
/// Short on purpose, and it grows only when something behind it is real: an
/// entry here is a promise that the value changes what the engine does, which
/// is checked by a test per setting in `tests/settings_and_io.rs`.
pub static SPECS: &[Spec] = &[
    Spec {
        name: "max_memory_usage",
        kind: Kind::Bytes,
        slot: Slot::MaxMemory,
        doc: "ceiling on one query's intermediate state (group table, sort buffer, \
              join build side). The per-query memory accounting behind it is a real \
              counter, so exceeding the ceiling fails the query -- and the external \
              sort and GROUP BY spill to disk rather than fail where they can. \
              0 means no limit",
    },
    Spec {
        name: "max_temporary_data_on_disk",
        kind: Kind::Bytes,
        slot: Slot::MaxTempDisk,
        doc: "ceiling on the bytes one query may write to the spill directory \
              (<data>/.spill), charged per write as the external sort, GROUP BY and \
              join flush their runs. Spilling amplifies -- a tight max_memory_usage \
              buys a large spill -- so this is a separate number, not a multiple of \
              that one. 0 means no limit",
    },
    Spec {
        name: "max_execution_time",
        kind: Kind::Seconds,
        slot: Slot::MaxTime,
        doc: "wall-clock deadline for a single statement, checked once per block so \
              a long scan stops inside itself rather than at the end. 0 disables the \
              deadline, which also keeps the clock read out of the operator loop",
    },
    Spec {
        name: "max_insert_block_size",
        kind: Kind::Rows,
        slot: Slot::InsertBlock,
        doc: "rows per block a streaming import (INSERT ... FROM INFILE) hands to \
              storage. Bounds the importer's resident set; values above 4096 pack \
              straight into parts instead of buffering in the delta",
    },
    Spec {
        name: "input_format_allow_errors_num",
        kind: Kind::Rows,
        slot: Slot::AllowErrors,
        doc: "malformed input rows an import may skip before it fails. 0 means the \
              first bad row aborts the statement, naming its line number",
    },
    Spec {
        name: "format_csv_delimiter",
        kind: Kind::Char,
        slot: Slot::CsvDelim,
        doc: "field separator for CSV read and write",
    },
    Spec {
        name: "format_csv_null_representation",
        kind: Kind::Text,
        slot: Slot::CsvNull,
        doc: "the exact unquoted text that means NULL on input and is written for \
              NULL on output; empty by default, matching the CLI's csv writer. A \
              quoted field is never NULL, which is the only way CSV can tell NULL \
              from the empty string",
    },
    Spec {
        name: "input_format_with_names_use_header",
        kind: Kind::Bool,
        slot: Slot::CsvHeader,
        doc: "read an import's first line as column names and match columns by name \
              rather than by position",
    },
    Spec {
        name: "wal_fold_bytes",
        kind: Kind::Bytes,
        slot: Slot::FoldBytes,
        doc: "write-ahead log bytes, per table, above which the next statement boundary \
              folds that table's log into parts and truncates it. This is the only \
              automatic checkpoint there is: without it a long-running writer never \
              writes a part, and disk-full arrives far ahead of the data volume. Large \
              on purpose -- a fold costs a whole-table rewrite, which is O(table) and \
              not O(log), so a small threshold turns every few MB of log into one. \
              0 disables it",
    },
    Spec {
        name: "wal_archive_retention",
        kind: Kind::Bytes,
        slot: Slot::WalRetention,
        doc: "byte budget for the archived write-ahead log under <data>/.wal-archive, \
              which is what BACKUP ... INCREMENTAL and RESTORE ... UNTIL roll forward \
              through. Segments older than the budget are trimmed as new ones arrive, \
              so this is the window a point-in-time recovery can reach back into. \
              Process-wide, not per session. 0 keeps every segment forever",
    },
];

/// The spec for `name`, case-insensitively.
pub fn spec(name: &str) -> Option<&'static Spec> {
    SPECS.iter().find(|s| s.name.eq_ignore_ascii_case(name))
}

impl Settings {
    /// Apply `name = value`, parsing `value` per the spec's [`Kind`].
    ///
    /// The error for an unknown name deliberately does *not* list the
    /// alternatives: the message a typo needs is "this is not a setting", and
    /// `SHOW SETTINGS` is one statement away for the rest.
    pub fn set(&mut self, name: &str, value: &str) -> Result<()> {
        let Some(sp) = spec(name) else {
            // A ClickHouse name this engine recognises but cannot honour gets
            // the sentence that says what it does instead; only a name nothing
            // knows is reported as a typo.
            // `None` back from `compat_note` means the value matches what the
            // engine already does, so there is nothing to store and nothing to
            // complain about.
            // `compat_note` answers `None` for two different reasons, and they
            // are not the same answer: a value that matches a fixed setting is
            // accepted, a name nothing has heard of is a typo.
            return match crate::sql::parser::compat_note(name, value) {
                Some(e) => Err(e),
                None if crate::sql::parser::knows_setting(name) => Ok(()),
                None => Err(unknown(name)),
            };
        };
        match sp.slot {
            Slot::MaxMemory => {
                // 0 is "no limit" and has to become the tracker's saturation
                // value, not a budget of zero bytes -- which would fail every
                // query, including the one that tries to raise it back.
                let n = parse_bytes(sp.name, value)?;
                self.max_memory_usage = if n == 0 { i64::MAX } else { n };
            }
            Slot::MaxTempDisk => {
                self.max_temporary_data_on_disk = parse_bytes(sp.name, value)?.max(0) as u64
            }
            Slot::MaxTime => self.max_execution_time_ms = parse_millis(sp.name, value)?,
            Slot::InsertBlock => {
                let n = parse_u64(sp.name, value)?;
                if n == 0 {
                    return Err(bad(sp.name, value, "a block of 0 rows would never finish"));
                }
                self.max_insert_block_size = n.min(u32::MAX as u64) as u32;
            }
            Slot::AllowErrors => {
                self.input_format_allow_errors_num =
                    parse_u64(sp.name, value)?.min(u32::MAX as u64) as u32
            }
            Slot::CsvDelim => self.format_csv_delimiter = parse_char(sp.name, value)?,
            Slot::CsvNull => self.format_csv_null = Box::from(value),
            Slot::CsvHeader => self.input_format_with_names_use_header = parse_bool(sp.name, value)?,
            Slot::FoldBytes => {
                self.wal_fold_bytes = parse_bytes(sp.name, value)?.max(0) as u64
            }
            Slot::WalRetention => {
                self.wal_archive_retention = parse_bytes(sp.name, value)?.max(0) as u64
            }
        }
        Ok(())
    }

    /// The current value of `sp`, rendered the way `SET` would accept it back.
    /// Round-tripping is a property test, not a hope: `SHOW SETTINGS` output
    /// that `SET` refuses is a documentation bug with a straight face.
    pub fn get(&self, sp: &Spec) -> String {
        match sp.slot {
            Slot::MaxMemory => {
                if self.max_memory_usage == i64::MAX {
                    "0".into()
                } else {
                    render_bytes(self.max_memory_usage)
                }
            }
            Slot::MaxTempDisk => render_zero_or_bytes(self.max_temporary_data_on_disk),
            Slot::MaxTime => render_secs(self.max_execution_time_ms),
            Slot::InsertBlock => self.max_insert_block_size.to_string(),
            Slot::AllowErrors => self.input_format_allow_errors_num.to_string(),
            Slot::CsvDelim => render_char(self.format_csv_delimiter),
            Slot::CsvNull => self.format_csv_null.to_string(),
            Slot::CsvHeader => if self.input_format_with_names_use_header { "1" } else { "0" }.into(),
            Slot::FoldBytes => render_zero_or_bytes(self.wal_fold_bytes),
            Slot::WalRetention => render_zero_or_bytes(self.wal_archive_retention),
        }
    }

    /// Push the two session-scoped settings into `sess`.
    ///
    /// The others are read where they are used (the importer, the CSV writer),
    /// so this is the whole of the coupling to `Session`: two calls that
    /// already existed on the facade and had no caller that spoke SQL.
    pub fn apply_to(&self, sess: &mut Session) {
        sess.set_memory_limit(self.max_memory_usage);
        sess.set_temp_disk_limit(self.max_temporary_data_on_disk);
        sess.set_timeout(match self.max_execution_time_ms {
            0 => None,
            ms => Some(Duration::from_millis(ms)),
        });
        sess.set_wal_fold_bytes(self.wal_fold_bytes);
        // The one push that leaves the session: process-wide by design, and
        // written here rather than in `set` so that it takes effect at the
        // same instant as everything else -- which is what gives a statement-
        // scoped `SETTINGS` clause its scope back, and what stops a `SET` that
        // fails on a later pair from having applied this one.
        crate::persist::wal::set_archive_retention(self.wal_archive_retention);
    }

    /// A query context carrying this session's budget, deadline and cancel
    /// flag — what `Session::read_stream` needs, built from the same numbers
    /// `apply_to` gave the session so an `INTO OUTFILE` runs under exactly
    /// the governance a plain `SELECT` would.
    pub fn context(&self, sess: &Session) -> crate::exec::operators::QueryContext {
        let ctx = crate::exec::operators::QueryContext {
            cancel: sess.cancel_handle(),
            deadline: None,
            mem: crate::exec::operators::MemTracker::with_limit(self.max_memory_usage),
            spill: sess.spill_budget(self.max_temporary_data_on_disk),
        };
        match self.max_execution_time_ms {
            0 => ctx,
            ms => ctx.deadline_in(Duration::from_millis(ms)),
        }
    }
}

// ------------------------------------------------------------- value parsing

#[cold]
fn unknown(name: &str) -> Error {
    Error::bind(format!(
        "unknown setting `{name}`. This engine implements the settings it can \
         actually honour and refuses the rest, because a setting accepted and \
         dropped is a query running under something other than what it asked \
         for. `SHOW SETTINGS` lists every name"
    ))
}

#[cold]
fn bad(name: &str, value: &str, why: &str) -> Error {
    Error::bind(format!("`{name} = {value}` is not usable: {why}"))
}

/// `1048576`, `1M`, `64Mi`, `2GiB`. Binary throughout — a memory budget spelled
/// `1G` and enforced as 10^9 would be off by 7%, and nobody would notice.
fn parse_bytes(name: &str, v: &str) -> Result<i64> {
    let t = v.trim();
    let digits = t.trim_end_matches(|c: char| c.is_ascii_alphabetic());
    let suffix = t[digits.len()..].trim();
    let shift = match suffix.trim_end_matches(['b', 'B']) {
        "" => 0,
        s if s.eq_ignore_ascii_case("k") || s.eq_ignore_ascii_case("ki") => 10,
        s if s.eq_ignore_ascii_case("m") || s.eq_ignore_ascii_case("mi") => 20,
        s if s.eq_ignore_ascii_case("g") || s.eq_ignore_ascii_case("gi") => 30,
        s if s.eq_ignore_ascii_case("t") || s.eq_ignore_ascii_case("ti") => 40,
        _ => return Err(bad(name, v, "want a byte count, optionally suffixed K, M, G or T")),
    };
    let n: i64 = digits
        .trim()
        .parse()
        .map_err(|_| bad(name, v, "want a byte count, optionally suffixed K, M, G or T"))?;
    n.checked_shl(shift)
        .filter(|&x| x >= 0 && (shift == 0 || (x >> shift) == n))
        .ok_or_else(|| bad(name, v, "overflows a 64-bit byte count"))
}

/// `render_bytes` for the settings whose 0 means "off" rather than "no bytes":
/// `0K` is a value `SET` would take back and a reader would misread.
fn render_zero_or_bytes(n: u64) -> String {
    match n {
        0 => "0".into(),
        n => render_bytes(n as i64),
    }
}

fn render_bytes(n: i64) -> String {
    for (unit, shift) in [("G", 30), ("M", 20), ("K", 10)] {
        if n >= 1 << shift && n & ((1 << shift) - 1) == 0 {
            return format!("{}{unit}", n >> shift);
        }
    }
    n.to_string()
}

/// Seconds in, milliseconds out. Fractional is accepted because a 250 ms
/// deadline is a thing people want and `0.25` is how they write it.
fn parse_millis(name: &str, v: &str) -> Result<u64> {
    let f: f64 = v
        .trim()
        .parse()
        .map_err(|_| bad(name, v, "want seconds, e.g. 30 or 0.25; 0 disables"))?;
    if !(f.is_finite() && f >= 0.0) {
        return Err(bad(name, v, "want a non-negative number of seconds"));
    }
    Ok((f * 1000.0).round() as u64)
}

fn render_secs(ms: u64) -> String {
    if ms % 1000 == 0 {
        (ms / 1000).to_string()
    } else {
        format!("{}", ms as f64 / 1000.0)
    }
}

fn parse_u64(name: &str, v: &str) -> Result<u64> {
    v.trim().parse().map_err(|_| bad(name, v, "want a non-negative whole number"))
}

fn parse_bool(name: &str, v: &str) -> Result<bool> {
    let t = v.trim();
    for (yes, words) in [(true, ["1", "true", "on", "yes"]), (false, ["0", "false", "off", "no"])] {
        if words.iter().any(|w| t.eq_ignore_ascii_case(w)) {
            return Ok(yes);
        }
    }
    Err(bad(name, v, "want 0/1, true/false, on/off or yes/no"))
}

/// One byte. `\t`, `\\` and a bare character are all accepted; the escapes are
/// here because the lexer already resolved `'\t'` inside a string literal, but
/// `SET format_csv_delimiter = '\\t'` from a shell that ate the backslash
/// arrives as the two characters and should still mean tab.
fn parse_char(name: &str, v: &str) -> Result<u8> {
    let b = v.as_bytes();
    let c = match b {
        [c] => *c,
        [b'\\', b't'] => b'\t',
        [b'\\', b'\\'] => b'\\',
        _ => return Err(bad(name, v, "want exactly one byte, e.g. ',' or '\\t'")),
    };
    if c == b'"' || c == b'\n' || c == b'\r' {
        return Err(bad(name, v, "a quote or a line terminator cannot also be the separator"));
    }
    Ok(c)
}

fn render_char(c: u8) -> String {
    match c {
        b'\t' => "\\t".into(),
        _ => (c as char).to_string(),
    }
}

// ---------------------------------------------------------------- SHOW output

/// `SHOW SETTINGS [LIKE 'pattern']` as a result set.
pub fn show(cfg: &Settings, like: Option<&str>) -> Result<ResultSet> {
    let schema = Schema::new(vec![
        Field::new("name", DataType::String),
        Field::new("value", DataType::String),
        Field::new("default", DataType::String),
        Field::new("type", DataType::String),
        Field::new("description", DataType::String),
    ])?;
    let def = Settings::default();
    let mut cols: [Vec<std::sync::Arc<str>>; 5] = Default::default();
    for sp in SPECS.iter().filter(|s| like.is_none_or(|p| like_match(p, s.name))) {
        cols[0].push(sp.name.into());
        cols[1].push(cfg.get(sp).into());
        cols[2].push(default_of(sp, &def).into());
        cols[3].push(kind_name(sp.kind).into());
        cols[4].push(sp.doc.into());
    }
    let block = Block::new(
        cols.into_iter().map(|v| Column::strs(DataType::String, v)).collect(),
    )?;
    let rows = block.rows();
    let mut rs = ResultSet { schema, blocks: vec![block], ..ResultSet::empty() };
    rs.stats.rows = rows;
    Ok(rs)
}

/// What a fresh process would report for `sp`.
///
/// `Settings::default()` answers it for every session-scoped setting. It
/// cannot answer for [`Slot::WalRetention`], which has no field to default:
/// reading it back through `get` would return whatever the process is
/// currently set to and make the `default` column agree with `value` forever.
fn default_of(sp: &Spec, def: &Settings) -> String {
    match sp.slot {
        Slot::WalRetention => render_zero_or_bytes(crate::persist::wal::DEFAULT_ARCHIVE_BYTES),
        _ => def.get(sp),
    }
}

fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::Bytes => "bytes",
        Kind::Seconds => "seconds",
        Kind::Rows => "rows",
        Kind::Bool => "bool",
        Kind::Char => "char",
        Kind::Text => "text",
    }
}

/// SQL `LIKE` over ASCII, case-insensitive, `%` and `_` only.
///
/// Iterative with one backtrack point rather than recursive: the pattern comes
/// from a user and `%a%a%a%...` is the standard way to make a recursive
/// matcher take exponential time.
fn like_match(pat: &str, s: &str) -> bool {
    let (p, t) = (pat.as_bytes(), s.as_bytes());
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star, mut mark) = (usize::MAX, 0usize);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == b'_' || p[pi].eq_ignore_ascii_case(&t[ti])) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == b'%' {
            star = pi;
            pi += 1;
            mark = ti;
        } else if star != usize::MAX {
            pi = star + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|&c| c == b'%')
}

// ----------------------------------------------------------------- the hook

/// A session's settings, shared so that wiring them up costs one call.
///
/// See the module header for the line `Session::run` needs. `Clone` is an
/// `Arc` bump and is what lets that line end its borrow of `self` before
/// `intercept` takes `&mut Session`.
#[derive(Clone, Default, Debug)]
pub struct Handle(Arc<Mutex<Settings>>);

impl Handle {
    pub fn new(s: Settings) -> Handle {
        Handle(Arc::new(Mutex::new(s)))
    }

    /// A copy of the current values. Cheap, and it is a copy on purpose: the
    /// importer and the CSV writer read settings for the length of a whole
    /// statement, and holding the lock across a 10 GB import would deadlock
    /// the `SET` that wanted to change them.
    pub fn snapshot(&self) -> Settings {
        let mut s = self.0.lock().unwrap_or_else(|e| e.into_inner()).clone();
        // The static is the truth for this one; the field is its mirror. Read
        // it here so `SHOW SETTINGS` reports what the archive tick will
        // actually use, and so the value restored after a scoped `SETTINGS`
        // clause is the one that was really in force.
        s.wal_archive_retention = crate::persist::wal::archive_retention();
        s
    }

    pub fn store(&self, s: Settings) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = s;
    }

    /// Run `sql` if it is one of the statements this module owns.
    ///
    /// `None` means "not mine" and is the answer for every statement the
    /// engine already had — decided by [`sniff`], a byte scan, before anything
    /// is tokenized.
    ///
    /// The statements:
    ///
    /// ```text
    ///   SET name = value [, ...]
    ///   SHOW SETTINGS [LIKE 'pattern']
    ///   INSERT INTO t [(cols)] FROM INFILE 'path' [FORMAT fmt] [SETTINGS ...]
    ///   <select> INTO OUTFILE 'path' [FORMAT fmt] [SETTINGS ...]
    ///   <select> SETTINGS name = value [, ...]
    /// ```
    ///
    /// They are recognised here rather than in `sql::parser` for a mechanical
    /// reason worth writing down: `Statement` is matched exhaustively in
    /// `session.rs`, so a new variant would not compile in a file this change
    /// does not own. Transaction control is recognised the same way and for
    /// the same kind of reason, so the shape is not new.
    pub fn intercept(&self, sess: &mut Session, sql: &str) -> Option<Result<Vec<ResultSet>>> {
        if !sniff(sql) {
            return None;
        }
        let toks = match tokenize(sql) {
            Ok(t) => t,
            // A `SET` that will not even lex is ours to explain: the lexer has
            // no suffixed-number literal, so `= 512M` dies here rather than in
            // `parse_bytes`, and "invalid numeric literal" alone would send the
            // reader looking in the wrong place. Everything else falls through
            // to the engine's own error, unchanged.
            Err(e) if starts_word(sql.trim_start().as_bytes(), b"set") => {
                return Some(Err(Error::bind(format!(
                    "{e}. A setting value that is not a plain number has to be quoted:                      `SET max_memory_usage = '512M'`"
                ))))
            }
            Err(_) => return None,
        };
        let spans = split_statements(&toks);
        // Nothing extended in there after all: hand it back whole so the
        // engine's own path lexes it exactly once more, as it always did.
        if !spans.iter().any(|&(a, b)| classify(&toks[a..b]).is_some()) {
            return None;
        }
        Some(self.run_spans(sess, sql, &toks, &spans))
    }

    fn run_spans(
        &self,
        sess: &mut Session,
        sql: &str,
        toks: &[Spanned],
        spans: &[(usize, usize)],
    ) -> Result<Vec<ResultSet>> {
        let mut out = Vec::with_capacity(spans.len());
        for &(a, b) in spans {
            let text = span_text(sql, toks, a, b);
            match classify(&toks[a..b]) {
                Some(k) => out.push(self.run_one(sess, k, text, &toks[a..b])?),
                // Not ours; the engine's own dispatcher owns it, including
                // every error message it would have produced.
                None => out.extend(sess.run(text)?),
            }
        }
        Ok(out)
    }

    fn run_one(
        &self,
        sess: &mut Session,
        kind: Ext,
        text: &str,
        toks: &[Spanned],
    ) -> Result<ResultSet> {
        match kind {
            Ext::Set => {
                let mut cfg = self.snapshot();
                for (name, value) in pairs(toks, 1)? {
                    cfg.set(&name, &value)?;
                }
                cfg.apply_to(sess);
                self.store(cfg);
                Ok(ResultSet::empty())
            }
            Ext::Show => {
                // `SHOW SETTINGS LIKE 'x'` — the only tail this takes.
                let like = match toks.get(2) {
                    None => None,
                    Some(t) if t.tok.is_keyword("LIKE") => match toks.get(3).map(|t| &t.tok) {
                        Some(Token::Str(p)) => Some(p.clone()),
                        _ => return Err(Error::parse("SHOW SETTINGS LIKE wants a pattern", pos(toks, 3))),
                    },
                    Some(t) => {
                        return Err(Error::parse(
                            format!("expected end of statement or LIKE, found `{}`", t.tok),
                            t.pos,
                        ))
                    }
                };
                if like.is_some() && toks.len() > 4 {
                    return Err(Error::parse("trailing input after SHOW SETTINGS", pos(toks, 4)));
                }
                show(&self.snapshot(), like.as_deref())
            }
            Ext::Import(at) => crate::io::run_import(sess, &self.snapshot(), text, toks, at),
            Ext::Export(at) => crate::io::run_export(sess, &self.snapshot(), text, toks, at),
            Ext::Scoped(at) => {
                // Query-level `SETTINGS`: apply, run the query without its
                // tail, put the session back. Scoped rather than sticky
                // because that is what the clause means everywhere else, and
                // a `SELECT` that quietly re-tuned the session for every
                // later statement would be a new way to get a wrong answer.
                let saved = self.snapshot();
                let mut cfg = saved.clone();
                for (name, value) in pairs(toks, at + 1)? {
                    cfg.set(&name, &value)?;
                }
                cfg.apply_to(sess);
                let head = &text[..toks[at].pos - toks[0].pos];
                let r = sess.query(head);
                saved.apply_to(sess);
                r
            }
        }
    }
}

/// Which extended statement a token span is, and where its keyword sits.
#[derive(Clone, Copy, Debug)]
enum Ext {
    Set,
    Show,
    /// Index of `INFILE`.
    Import(usize),
    /// Index of `INTO` in `INTO OUTFILE`.
    Export(usize),
    /// Index of a trailing top-level `SETTINGS`.
    Scoped(usize),
}

/// Could this text possibly contain one of our statements?
///
/// One pass, and it answers "no" for every statement the engine has ever run
/// without tokenizing anything. `SET`/`SHOW` are checked only where a
/// *statement* can start -- the front of the text, or after a `;` -- so
/// `UPDATE t SET x = 1` and `LIMIT 10 OFFSET 5` never reach the tokenizer.
/// `infile`, `outfile` and `settings` are searched anywhere, so a column named
/// `filename` costs one wasted lex and nothing else.
///
/// A `;` inside a string literal makes this say "maybe" when the answer is
/// "no", which is the safe direction: the tokenizer runs, `classify` finds
/// nothing, and the statement goes back to the engine untouched.
fn sniff(sql: &str) -> bool {
    let b = sql.as_bytes();
    if statement_starts_with(b, 0) {
        return true;
    }
    if b.iter().enumerate().any(|(i, &c)| c == b';' && statement_starts_with(b, i + 1)) {
        return true;
    }
    contains_ci(b, b"infile") || contains_ci(b, b"outfile") || contains_ci(b, b"settings")
}

/// Does a statement beginning at `from` (past whitespace and comments) open
/// with `SET` or `SHOW`?
fn statement_starts_with(b: &[u8], from: usize) -> bool {
    let mut i = from;
    // Leading whitespace and `--`/`/* */` comments, so that a commented header
    // does not hide the first word.
    loop {
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        if b[i..].starts_with(b"--") {
            i += b[i..].iter().position(|&c| c == b'\n').map_or(b.len() - i, |p| p + 1);
        } else if b[i..].starts_with(b"/*") {
            i += b[i..].windows(2).position(|w| w == b"*/").map_or(b.len() - i, |p| p + 2);
        } else {
            break;
        }
    }
    starts_word(&b[i..], b"set") || starts_word(&b[i..], b"show")
}

/// `w` at the front of `b`, followed by something that cannot continue an
/// identifier — so `SETTINGS` does not read as `SET`.
fn starts_word(b: &[u8], w: &[u8]) -> bool {
    b.len() >= w.len()
        && b[..w.len()].eq_ignore_ascii_case(w)
        && b.get(w.len()).is_none_or(|c| !c.is_ascii_alphanumeric() && *c != b'_')
}

/// Case-insensitive substring. Gated on the first byte so the inner compare
/// runs only where it can match; `needle` is never empty.
fn contains_ci(hay: &[u8], needle: &[u8]) -> bool {
    let (f0, f1) = (needle[0], needle[0].to_ascii_uppercase());
    hay.len() >= needle.len()
        && (0..=hay.len() - needle.len())
            .any(|i| (hay[i] == f0 || hay[i] == f1) && hay[i..i + needle.len()].eq_ignore_ascii_case(needle))
}

/// Top-level statement spans, split on `;` the *lexer* found — a `;` inside a
/// string literal or a comment is not a boundary, and the only way to agree
/// with the parser about that is to ask the same tokenizer.
fn split_statements(toks: &[Spanned]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0;
    for i in 0..=toks.len() {
        if i == toks.len() || toks[i].tok == Token::Semicolon {
            if i > start {
                out.push((start, i));
            }
            start = i + 1;
        }
    }
    out
}

/// The source text of `toks[a..b]`, from the first token to whatever ended it.
fn span_text<'s>(sql: &'s str, toks: &[Spanned], a: usize, b: usize) -> &'s str {
    let end = toks.get(b).map_or(sql.len(), |t| t.pos);
    &sql[toks[a].pos..end]
}

fn pos(toks: &[Spanned], i: usize) -> usize {
    toks.get(i).map_or(toks.last().map_or(0, |t| t.pos), |t| t.pos)
}

/// Is this span one of ours, and where is its keyword?
fn classify(t: &[Spanned]) -> Option<Ext> {
    if t.is_empty() {
        return None;
    }
    if t[0].tok.is_keyword("SET") {
        return Some(Ext::Set);
    }
    if t[0].tok.is_keyword("SHOW") && t.get(1).is_some_and(|x| x.tok.is_keyword("SETTINGS")) {
        return Some(Ext::Show);
    }
    // Depth-tracked so `... WHERE x IN (SELECT ... SETTINGS ...)` is not
    // mistaken for a clause on the outer query. The last top-level match wins,
    // which is the one the grammar puts at the end.
    let (mut depth, mut found) = (0i32, None);
    for (i, s) in t.iter().enumerate() {
        match s.tok {
            Token::LParen => depth += 1,
            Token::RParen => depth -= 1,
            _ if depth != 0 => {}
            Token::Word { quoted: false, .. } => {
                if s.tok.is_keyword("INFILE") && t[0].tok.is_keyword("INSERT") {
                    found = Some(Ext::Import(i));
                } else if s.tok.is_keyword("OUTFILE") && i > 0 && t[i - 1].tok.is_keyword("INTO") {
                    found = Some(Ext::Export(i - 1));
                } else if s.tok.is_keyword("SETTINGS")
                    && i > 0
                    && t[i - 1].tok != Token::Dot
                    && found.is_none()
                {
                    // Not after a dot: `system.settings` and `mydb.settings`
                    // are qualified *names*, and claiming them made every
                    // table called `settings` unqueryable -- `SELECT * FROM
                    // mydb.settings LIMIT 1` was parsed as a settings clause
                    // and died on "expected a setting name".
                    found = Some(Ext::Scoped(i));
                }
            }
            _ => {}
        }
    }
    // A `SETTINGS` tail belongs to whichever file statement precedes it, and
    // those parse it themselves; only a bare query's tail is ours to scope.
    match found {
        Some(Ext::Scoped(i)) if t[..i].iter().any(|s| s.tok.is_keyword("INFILE") || s.tok.is_keyword("OUTFILE")) => None,
        f => f,
    }
}

/// `k = v [, k = v]*` starting at token `i`, to the end of the span.
///
/// Values arrive as tokens rather than as sliced source so that `= 1024` and
/// `= '1024'` are the same request and a negative number keeps its sign.
pub(crate) fn pairs(t: &[Spanned], mut i: usize) -> Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    loop {
        let name = match t.get(i).map(|s| &s.tok) {
            Some(Token::Word { value, .. }) => value.clone(),
            _ => return Err(Error::parse("expected a setting name", pos(t, i))),
        };
        if t.get(i + 1).map(|s| &s.tok) != Some(&Token::Eq) {
            return Err(Error::parse(format!("expected `=` after `{name}`"), pos(t, i + 1)));
        }
        i += 2;
        let neg = t.get(i).map(|s| &s.tok) == Some(&Token::Minus);
        i += neg as usize;
        let value = match t.get(i).map(|s| &s.tok) {
            Some(Token::Str(s)) => s.clone(),
            Some(tok @ (Token::Number(_) | Token::Word { .. })) => tok.to_string(),
            _ => return Err(Error::parse(format!("expected a value for `{name}`"), pos(t, i))),
        };
        out.push((name, if neg { format!("-{value}") } else { value }));
        i += 1;
        match t.get(i).map(|s| &s.tok) {
            Some(Token::Comma) => i += 1,
            None => return Ok(out),
            Some(tok) => {
                return Err(Error::parse(format!("unexpected `{tok}` in a SETTINGS list"), pos(t, i)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_spec_round_trips_through_set() {
        // A `SHOW SETTINGS` value column that `SET` will not take back is a
        // lie in a table whose whole job is to be believed.
        let mut cfg = Settings::default();
        for sp in SPECS {
            let shown = cfg.get(sp);
            let mut c2 = Settings::default();
            c2.set(sp.name, &shown).unwrap_or_else(|e| panic!("{}: {e}", sp.name));
            assert_eq!(c2.get(sp), shown, "{} did not round-trip", sp.name);
        }
        // ... and after a change, not just at the defaults.
        cfg.set("max_memory_usage", "512M").unwrap();
        assert_eq!(cfg.max_memory_usage, 512 << 20);
        assert_eq!(cfg.get(spec("max_memory_usage").unwrap()), "512M");
    }

    #[test]
    fn unknown_names_are_refused() {
        let mut cfg = Settings::default();
        assert!(cfg.set("max_memry_usage", "1").is_err());
        assert!(cfg.set("allow_experimental_everything", "1").is_err());
        // ... and a known name with an unusable value is refused too, rather
        // than clamping silently.
        assert!(cfg.set("max_memory_usage", "lots").is_err());
        assert!(cfg.set("max_insert_block_size", "0").is_err());
        assert!(cfg.set("format_csv_delimiter", "ab").is_err());
        assert!(cfg.set("format_csv_delimiter", "\"").is_err());
        assert!(cfg.set("input_format_with_names_use_header", "maybe").is_err());
    }

    #[test]
    fn byte_suffixes_are_binary() {
        let mut c = Settings::default();
        for (text, want) in
            [("1024", 1024i64), ("1K", 1 << 10), ("64Mi", 64 << 20), ("2GiB", 2 << 30)]
        {
            c.set("max_memory_usage", text).unwrap();
            assert_eq!(c.max_memory_usage, want, "{text}");
        }
        c.set("max_memory_usage", "0").unwrap();
        assert_eq!(c.max_memory_usage, i64::MAX, "0 must mean no limit, not a zero budget");
    }

    #[test]
    fn seconds_accept_fractions() {
        let mut c = Settings::default();
        c.set("max_execution_time", "0.25").unwrap();
        assert_eq!(c.max_execution_time_ms, 250);
        assert_eq!(c.get(spec("max_execution_time").unwrap()), "0.25");
        c.set("max_execution_time", "30").unwrap();
        assert_eq!(c.get(spec("max_execution_time").unwrap()), "30");
    }

    /// The sniff must not wake up for ordinary SQL, or every statement in the
    /// engine pays for a second tokenize.
    #[test]
    fn sniff_is_quiet_for_ordinary_sql() {
        for q in [
            "SELECT a FROM t WHERE b = 1",
            "UPDATE t SET x = 1 WHERE id = 2",
            "SELECT a FROM t ORDER BY a LIMIT 10 OFFSET 5",
            "INSERT INTO t VALUES (1, 'x')",
            "CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id",
        ] {
            assert!(!sniff(q), "woke for `{q}`");
        }
        for q in [
            "SET max_memory_usage = 1",
            "  \n SHOW SETTINGS",
            "-- comment\nSET x = 1",
            "SELECT 1 SETTINGS max_memory_usage = 1",
            "INSERT INTO t FROM INFILE 'x.csv'",
            "SELECT * FROM t INTO OUTFILE 'x.csv'",
            // A statement boundary is a place a statement can start, so a
            // `SET` after one is found -- `Session::run` takes whole scripts.
            "SELECT 1; SET max_memory_usage = 1",
            "CREATE TABLE t (a UInt64) ENGINE = Memory;\n  show settings",
        ] {
            assert!(sniff(q), "slept through `{q}`");
        }
    }

    #[test]
    fn like_matches_the_way_sql_does() {
        assert!(like_match("max\\_%", "max_memory_usage") || like_match("max%", "max_memory_usage"));
        assert!(like_match("%csv%", "format_csv_delimiter"));
        assert!(like_match("MAX%", "max_memory_usage"), "LIKE is case-insensitive here");
        assert!(!like_match("%csv%", "max_memory_usage"));
        assert!(like_match("%", "anything"));
        // The pattern a recursive matcher takes exponential time on: 20 stars
        // each of which can float, over a subject that matches every prefix and
        // fails only at the last byte.
        assert!(!like_match(&format!("{}b", "%a".repeat(20)), &"a".repeat(30)));
    }

    #[test]
    fn classify_ignores_a_nested_settings_clause() {
        let t = tokenize("SELECT a FROM (SELECT b FROM t SETTINGS max_memory_usage = 1)").unwrap();
        assert!(classify(&t).is_none(), "a subquery's tail is not the outer query's");
        let t = tokenize("SELECT a FROM t SETTINGS max_memory_usage = 1").unwrap();
        assert!(matches!(classify(&t), Some(Ext::Scoped(_))));
    }
}
