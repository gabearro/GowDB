//! The public entry point: SQL text in, results out.
//!
//! A `Session` owns a [`Catalog`] and turns statement text into effects. The
//! pipeline for a query is:
//!
//! ```text
//!   sql -> parse -> bind -> optimize -> execute -> ResultSet
//! ```
//!
//! DDL and DML take shortcuts through the same machinery: `INSERT ... SELECT`
//! runs the SELECT through the executor and hands the blocks to
//! [`crate::storage::Table::insert`], and `ALTER ... DELETE` runs its predicate
//! as a SELECT of primary keys.
//!
//! ## Transactions
//!
//! `BEGIN` / `COMMIT` / `ROLLBACK` are three lines of bookkeeping over two
//! things that already existed: the per-table overlay in
//! [`crate::storage::Table::begin_txn`], and the staged records in
//! [`crate::persist::Wal`]. A transaction here is
//!
//!   * a set of *enlisted* tables, each holding a private `Arc<PartSet>` that
//!     its writes go into and its own reads come out of; and
//!   * one staging group per enlisted table's log, plus the LSN that log stood
//!     at when the table was enlisted.
//!
//! `COMMIT` makes the logs durable and then stores each overlay over its
//! published set -- durability first, visibility second, and the second half
//! cannot fail. `ROLLBACK` drops the overlays and rewinds each log to its
//! enlistment LSN, which leaves both memory and disk exactly as they were.
//!
//! Enlistment is **lazy**: `BEGIN` writes one `Option` and touches no table, so
//! a transaction that only reads costs nothing, and a transaction over one
//! table does not drag the others into it.
//!
//! ### A transaction that has failed stays failed
//!
//! The first statement to return an error while a transaction is open
//! *poisons* it: every later statement is refused, and `COMMIT` rolls back and
//! reports rather than publishing. Two holes close together. A statement that
//! failed used to leave the transaction open, so the statements after it
//! returned `Ok` over writes the session was already committed to discarding.
//! And a nested `BEGIN` errored while leaving the outer transaction open, so
//! the inner block's `COMMIT` -- which believed it was ending its own
//! transaction -- durably committed the outer one's uncommitted work at a
//! boundary the outer writer never chose.
//!
//! `ROLLBACK` is the way out and always runs. `COMMIT` on a poisoned
//! transaction rolls back and returns an error rather than the `Ok` PostgreSQL
//! reports for the same case: a client that gets `Ok` from `COMMIT` is
//! entitled to believe its writes landed.
//!
//! ### What autocommit pays
//!
//! Nothing. A statement outside a transaction takes the same path it always
//! did: `txn` is `None`, no table has an overlay, and the WAL keeps logging
//! plain committed records with an fsync each. The one deliberate change is
//! that a *multi-block* `INSERT` now runs inside an implicit transaction --
//! see [`Session::atomic_stmt`] -- which makes it atomic and, as a side
//! effect, costs one fsync instead of one per block. Single-block statements,
//! which is every `INSERT ... VALUES`, never open one.
//!
//! ## One writer, many readers
//!
//! Every SQL entry point used to take `&mut self`, and not because a query
//! mutates: because [`Session::plan`] opened with `catalog.flush_all()`, so a
//! `SELECT` on one table took exclusive write access to every other one. The
//! cost was not contention, it was the type system -- eight identical 2M-row
//! queries behind an `Arc<Mutex<Session>>` measured 33.03 ms on one thread,
//! 32.34 on two and 31.58 on four. Perfectly flat, because nothing overlapped.
//!
//! The split is three things:
//!
//!   * [`Session::read`] and friends take `&self`. They never flush; the
//!     `&mut` half does that, and [`Catalog::has_pending_writes`] is what
//!     tells a reader when it must.
//!   * [`Db`] holds one `Session` behind an `RwLock` and hands out [`Reader`]s
//!     (`Send + Sync + Clone`, `'static`) and a [`Writer`] guard. N readers
//!     run at once under the shared lock; the writer takes it exclusively for
//!     the duration of one statement.
//!   * every read carries a [`QueryContext`] built from the session's own
//!     limits, so a memory budget, a deadline and a cancel flag are finally
//!     reachable from the facade rather than only from the operator tests.
//!
//! Isolation is snapshot, not MVCC: `Scan` pins one `Arc<PartSet>` per table
//! at build time and reads nothing else, and a writer publishes by storing a
//! new `Arc` over the old one. There are no version chains, no per-row
//! visibility test and no reader/writer interlock beyond the `RwLock` that
//! keeps `&Catalog` and `&mut Catalog` apart.
//!
//! ### Several tables, one commit point
//!
//! Logs are per table, so a transaction spanning N tables touches N files. It
//! used to write N commit markers and fsync N files, which is N commit points:
//! a crash between two of them left a *prefix* of the transaction durable,
//! committing some tables and not others with no error to anyone.
//!
//! [`Session::commit_durable`] is a two-phase commit instead. The last
//! enlisted table is the coordinator; every earlier one logs a *prepare* that
//! cites the coordinator's decision, and the coordinator's own marker -- the
//! last append and the last fsync of the COMMIT -- releases all of them at
//! once. Recovery resolves each prepare against the decision it names, so a
//! crash anywhere in the sequence commits every table or none. It costs the
//! same N fsyncs the broken version did, because the decision doubles as the
//! coordinator's own marker; a single-table transaction writes no prepare at
//! all and is unchanged, which is the shape the OLTP path actually has. See
//! [`crate::persist::Wal::prepare`].
//!
//! The exception is a table that took a mutation the log cannot describe: its
//! commit point is the `TABLE` rename in [`Session::fold_to_parts`] rather
//! than a marker, so it is not a participant and its atomicity with the rest
//! of the transaction is still the rename's.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock, RwLockWriteGuard, Weak};
use std::time::{Duration, Instant};

use crate::catalog::Catalog;
use crate::common::{Error, Result};
use crate::exec::operators;
use crate::exec::operators::{MemGuard, QueryContext};
use crate::planner::{
    binder::Binder,
    logical::{BoundExpr, LogicalPlan, ZoneFilter},
    optimizer,
};
use crate::sql::ast::{
    ColumnDef, CreateTable, ExplainKind, Insert, InsertSource, ObjectName, Statement,
};
use crate::sql::parse;
use crate::storage::table::KeyConflict;
use crate::storage::{MaskRuns, SweepLog};
use crate::types::{Block, Column, ColumnBuilder, DataType, Field, Schema, TableDef, Value};

#[derive(Debug, Default, Clone, Copy)]
pub struct QueryStats {
    pub rows: usize,
    pub elapsed_us: u128,
    pub granules_read: u64,
    pub granules_pruned: u64,
    pub rows_scanned: u64,
}

/// What a streaming read hands its sink.
///
/// `Head` arrives exactly once, before any row and even when there are none:
/// a sink that has to describe the result before sending it -- a wire
/// protocol's `RowDescription`, a CSV header, a portal -- must not have to
/// wait for a first block that may never come. `Rows` blocks are owned, so a
/// sink can keep one without copying it.
pub enum StreamItem<'a> {
    Head(&'a Schema),
    Rows(Block),
}

/// A materialized result. Small by construction: anything large should be
/// streamed through [`Session::read_stream`] instead.
#[derive(Debug)]
pub struct ResultSet {
    pub schema: Schema,
    pub blocks: Vec<Block>,
    pub stats: QueryStats,
    /// Set for statements that report a count rather than rows (INSERT, ALTER).
    pub affected: Option<usize>,
}

impl ResultSet {
    pub fn empty() -> ResultSet {
        ResultSet {
            schema: Schema::empty(),
            blocks: Vec::new(),
            stats: QueryStats::default(),
            affected: None,
        }
    }

    pub fn with_affected(n: usize) -> ResultSet {
        ResultSet { affected: Some(n), ..ResultSet::empty() }
    }

    pub fn rows(&self) -> usize {
        self.blocks.iter().map(|b| b.rows()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.rows() == 0
    }

    /// Every row as `Value`s. Convenience for tests and small results.
    pub fn to_values(&self) -> Vec<Vec<Value>> {
        let mut out = Vec::with_capacity(self.rows());
        for b in &self.blocks {
            for r in 0..b.rows() {
                out.push((0..b.width()).map(|c| b.column(c).value(r)).collect());
            }
        }
        out
    }

    /// The single cell of a 1x1 result, for `SELECT count(*)`-shaped queries.
    pub fn scalar(&self) -> Option<Value> {
        let b = self.blocks.first()?;
        if b.rows() == 0 || b.width() == 0 {
            return None;
        }
        Some(b.column(0).value(0))
    }

    fn from_rows(schema: Schema, rows: Vec<Vec<Value>>) -> Result<ResultSet> {
        let mut builders: Vec<ColumnBuilder> = schema
            .fields()
            .iter()
            .map(|f| ColumnBuilder::with_capacity(f.ty.clone(), rows.len()))
            .collect();
        for r in &rows {
            for (c, b) in builders.iter_mut().enumerate() {
                b.push_value(r.get(c).unwrap_or(&Value::Null))?;
            }
        }
        let n = rows.len();
        Ok(ResultSet {
            schema,
            blocks: vec![Block::new(builders.into_iter().map(|b| b.finish()).collect())?],
            stats: QueryStats { rows: n, ..Default::default() },
            affected: None,
        })
    }

    fn one_string_column(name: &str, values: Vec<String>) -> Result<ResultSet> {
        let schema = Schema::new(vec![Field::new(name, DataType::String)])?;
        ResultSet::from_rows(
            schema,
            values.into_iter().map(|v| vec![Value::str(v)]).collect(),
        )
    }
}

/// Fixed-width text rendering, close enough to ClickHouse's `PrettyCompact` to
/// be readable in a terminal.
impl std::fmt::Display for ResultSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(n) = self.affected {
            return write!(f, "Ok. {n} row{} affected.", if n == 1 { "" } else { "s" });
        }
        if self.schema.is_empty() {
            return write!(f, "Ok.");
        }
        let ncols = self.schema.len();
        let mut widths: Vec<usize> = self
            .schema
            .fields()
            .iter()
            .map(|f| f.name.chars().count())
            .collect();
        // Rendered straight out of the blocks. The old shape went through
        // `to_values()` first, which materialized a `Vec<Vec<Value>>` of the
        // whole result *in addition* to the strings -- two full copies live at
        // once, one of them thrown away unread.
        let mut cells: Vec<Vec<String>> = Vec::with_capacity(self.rows());
        for b in &self.blocks {
            for r in 0..b.rows() {
                let row: Vec<String> = (0..ncols)
                    .map(|c| {
                        if c < b.width() {
                            b.column(c).value(r).render_plain()
                        } else {
                            String::new()
                        }
                    })
                    .collect();
                for (c, s) in row.iter().enumerate() {
                    widths[c] = widths[c].max(s.chars().count());
                }
                cells.push(row);
            }
        }
        let rule =
            |f: &mut std::fmt::Formatter<'_>, l: &str, m: &str, r: &str| -> std::fmt::Result {
                write!(f, "{l}")?;
                for (i, w) in widths.iter().enumerate() {
                    if i > 0 {
                        write!(f, "{m}")?;
                    }
                    fill(f, RULE_RUN, '─'.len_utf8(), w + 2)?;
                }
                writeln!(f, "{r}")
            };
        rule(f, "┌", "┬", "┐")?;
        write!(f, "│")?;
        for (i, fl) in self.schema.fields().iter().enumerate() {
            pad_cell(f, &fl.name, widths[i])?;
        }
        writeln!(f)?;
        rule(f, "├", "┼", "┤")?;
        for r in &cells {
            write!(f, "│")?;
            for (i, s) in r.iter().enumerate() {
                pad_cell(f, s, widths[i])?;
            }
            writeln!(f)?;
        }
        rule(f, "└", "┴", "┘")?;
        write!(
            f,
            "{} row{} in {:.3} ms",
            self.stats.rows,
            if self.stats.rows == 1 { "" } else { "s" },
            self.stats.elapsed_us as f64 / 1000.0
        )
    }
}

/// 64 spaces and 64 box-drawing dashes, the padding units [`fill`] copies from.
const PAD_RUN: &str = "                                                                ";
const RULE_RUN: &str = "────────────────────────────────────────────────────────────────";

/// Write `n` copies of a one-character unit, `chunk.len() / unit` at a time.
///
/// Replaces `"─".repeat(w + 2)`, which allocated a `String` per column per rule
/// line -- three per column for every rendered result, and on a wide cell a
/// multi-hundred-kilobyte one.
fn fill(f: &mut std::fmt::Formatter<'_>, chunk: &str, unit: usize, mut n: usize) -> std::fmt::Result {
    let per = chunk.len() / unit;
    while n > 0 {
        let k = n.min(per);
        f.write_str(&chunk[..k * unit])?;
        n -= k;
    }
    Ok(())
}

/// One `│`-terminated cell: a space, `s` left-aligned in `w` columns, a space.
///
/// Hand-padded rather than `{:w$}`, and this is a crash fix rather than a
/// tidy-up: `std::fmt` packs a *runtime* width into a `u16`, so a cell or a
/// column name longer than 65535 characters -- one 64 KiB JSON blob, one long
/// log line -- panicked inside the formatter. With `panic = "abort"` in the
/// release profile that is a SIGABRT with no unwinding, which in a library
/// embedding takes the host process down. Padding by hand has no ceiling, and
/// as a side effect skips the formatting machinery's per-argument runtime
/// dispatch.
fn pad_cell(f: &mut std::fmt::Formatter<'_>, s: &str, w: usize) -> std::fmt::Result {
    f.write_str(" ")?;
    f.write_str(s)?;
    // `chars`, not `len`: the widths were measured that way, so a multi-byte
    // cell must be measured the same way or the columns stop lining up.
    fill(f, PAD_RUN, 1, w.saturating_sub(s.chars().count()))?;
    f.write_str(" │")
}

pub struct Session {
    pub catalog: Catalog,
    /// One open write-ahead log per persistent table, keyed by `db.table`.
    /// Empty for an in-memory session.
    wals: crate::common::FastMap<String, crate::persist::Wal>,
    /// When false, writes are only durable at [`Session::checkpoint`].
    wal_enabled: bool,
    /// The exclusive `flock` on `<root>/LOCK`, held open for exactly as long
    /// as this session exists. Never read: closing the fd is what releases the
    /// lock, so the field's only job is to keep the descriptor alive. `None`
    /// for an in-memory session, and on a non-unix target (see
    /// `lock_data_dir`).
    ///
    /// This is also the answer to "does `Session` need a `Drop`?" -- it does
    /// not. Dropping the `File` releases the lock; the write-ahead log is
    /// fsynced before every acknowledgement, so nothing durable is pending;
    /// and an implicit checkpoint in `Drop` would be actively wrong, because
    /// it would turn an abandoned or errored session into a commit and would
    /// have nowhere to report an I/O failure.
    _lock: Option<File>,
    /// The open transaction, if any. `None` on every autocommit path, and the
    /// only thing `BEGIN` has to write.
    txn: Option<Txn>,
    /// Which connection is driving right now: stamped by [`Db::writer`] as it
    /// takes the lock, and left at 0 for a session nobody shared. `COMMIT` and
    /// `ROLLBACK` compare it against [`Txn::owner`], which is the whole of the
    /// mechanism that stops one connection ending another's transaction.
    owner: u64,
    /// Whether that connection is still *alive*, stamped alongside it. `BEGIN`
    /// copies it into the transaction, and that is what tells "another
    /// connection is still working on this" apart from "the connection that
    /// opened this is gone and nothing will ever end it" -- see
    /// [`Session::reap_txn`]. `None` for a bare `Session`, which is one
    /// connection by construction and cannot orphan anything.
    owner_tok: Option<Weak<u64>>,
    /// Session-scoped settings, and the hook `SET` / `SHOW SETTINGS` /
    /// `SETTINGS` reach the engine through. Per-session rather than global on
    /// purpose: two `Session`s in one process -- which `Db` and most of the
    /// test files create -- would otherwise report each other's values.
    settings: crate::settings::Handle,
    /// Opened with a *shared* directory lock: every mutating statement is
    /// refused, so several such sessions -- in this process or another -- can
    /// read one directory at once. See [`Session::open_read_only`].
    read_only: bool,
    /// Budget, deadline and cancel flag handed to every query this session
    /// runs. Three words, read on the read path and nowhere else.
    limits: Limits,
    /// The bounded ring `system.query_log` reads.
    ///
    /// Shared rather than owned so a [`Reader`] -- which runs statements
    /// through `&self` on another thread -- appends to the same log the writer
    /// does. One uncontended lock and one `memcpy` per *statement*, and no
    /// allocation at all in steady state: see [`crate::system::QueryLog`].
    log: crate::system::QueryLog,
    /// CHECK constraints and stored views, loaded from `<db>._granular_ddl` at
    /// open. Empty -- and so free to consult -- on a database that declares
    /// neither.
    ext: Extensions,
    /// `wal_fold_bytes`: the per-table log size above which a statement
    /// boundary folds that log into parts. 0 disables it. Pushed in by
    /// [`crate::settings::Settings::apply_to`], the way the memory budget and
    /// the timeout are.
    fold_bytes: u64,
    /// Tables whose log has outgrown `fold_bytes` and have not been folded
    /// yet. Empty on every path that has not crossed the threshold, so the
    /// drain at each statement boundary is one length check.
    ///
    /// Deferred rather than folded where the threshold is noticed, and that is
    /// the whole of the care this needs: the insert path is *log before
    /// apply*, so a fold inside the append would `write_table` a table that is
    /// missing the record it is about to truncate the log for -- a durability
    /// hole with a checkpoint's name on it.
    fold_due: Vec<String>,
    /// The unkeyed sweep's encode buffer, kept across statements so a session
    /// deleting in a loop allocates once. See [`Session::apply_sweep`].
    masks: MaskRuns,
}

/// The per-query governance a session applies, as *settings* rather than as a
/// live [`QueryContext`].
///
/// A `QueryContext` cannot be reused across queries and that is deliberate:
/// its deadline is an absolute `Instant`, and its `MemTracker` is a single
/// atomic whose reservations would accumulate across queries (a query that
/// failed between `grow_to` and the guard's drop leaks its charge forever).
/// So the session stores the *policy* and mints one context per query --
/// exactly one `Arc` allocation, against a query that has already parsed,
/// bound and lowered a plan.
///
/// `cancel` is the one thing shared rather than minted: a handle taken once
/// must be able to stop whatever the session runs next, and every query after
/// it, which is what a client-disconnect or a Ctrl-C handler needs.
#[derive(Clone, Debug)]
struct Limits {
    mem: i64,
    timeout: Option<Duration>,
    cancel: Arc<AtomicBool>,
    /// The data directory, when there is one. Spill files go under
    /// `<root>/.spill` rather than into `env::temp_dir()`, so an operator who
    /// sized the data volume for the database sized it for the spill too. A
    /// leading dot, like `.wal`: `store::valid_name` already refuses
    /// those as database names, so it can never collide with one.
    spill_root: Option<Arc<Path>>,
    /// `max_temporary_data_on_disk`, 0 for unlimited.
    temp_disk: u64,
}

/// Where a data directory keeps its spill files.
pub const SPILL_DIR: &str = ".spill";

fn spill_root(root: &Path) -> Arc<Path> {
    root.join(SPILL_DIR).into()
}

impl Default for Limits {
    fn default() -> Limits {
        Limits {
            mem: crate::exec::operators::DEFAULT_MEM_BUDGET,
            timeout: None,
            cancel: Arc::new(AtomicBool::new(false)),
            spill_root: None,
            temp_disk: 0,
        }
    }
}

impl Limits {
    /// One context for one query. `deadline_in` is applied here rather than
    /// stored, so a session-wide timeout means "per statement" and not "from
    /// whenever the setting was made".
    fn context(&self) -> QueryContext {
        let ctx = QueryContext {
            cancel: Arc::clone(&self.cancel),
            deadline: None,
            mem: crate::exec::operators::MemTracker::with_limit(self.mem),
            spill: self.spill(self.temp_disk),
        };
        match self.timeout {
            Some(d) => ctx.deadline_in(d),
            None => ctx,
        }
    }

    /// The spill root and ceiling for one query. `ambient()` -- shared, no
    /// allocation -- for an in-memory session with no ceiling, which is what
    /// most of the test suite is; one small `Arc` otherwise.
    fn spill(&self, limit: u64) -> Arc<crate::exec::operators::SpillBudget> {
        match (&self.spill_root, limit) {
            (None, 0) => crate::exec::operators::SpillBudget::ambient(),
            (Some(r), l) => crate::exec::operators::SpillBudget::new(Arc::clone(r), l),
            (None, l) => {
                crate::exec::operators::SpillBudget::new(std::env::temp_dir().into(), l)
            }
        }
    }
}

/// The tables an open transaction has written to.
///
/// A `Vec` rather than a map: a transaction touches a handful of tables at
/// most, and the linear scan over a few short strings beats hashing one on
/// every write. Enlistment order is preserved, which is the order COMMIT
/// writes its markers in and the order a partial multi-table commit would
/// have happened in.
#[derive(Default)]
struct Txn {
    tables: Vec<Enlisted>,
    /// The connection that ran `BEGIN`. Copied from [`Session::owner`], so it
    /// is 0 -- and therefore always a match -- for a bare `Session`, which is
    /// one connection by construction and is what the CLI has.
    owner: u64,
    /// A handle on that connection's *liveness*, copied from
    /// [`Session::owner_tok`]. Dead means the `Db` clone that ran `BEGIN` has
    /// been dropped, so no `COMMIT` and no `ROLLBACK` can ever arrive: the
    /// transaction is swept rather than left to refuse every other connection
    /// forever. `None` for a bare `Session`, whose transaction is never
    /// foreign and must never be swept.
    tok: Option<Weak<u64>>,
    /// The first error a statement raised while this transaction was open.
    ///
    /// Once set the transaction is *poisoned*: every later statement is
    /// refused and COMMIT rolls back instead of committing. Without it a
    /// failed statement left the transaction open and the ones after it
    /// returned `Ok` over work that was then discarded at exit -- a client
    /// that never checks a second time is told its writes landed when they
    /// did not. A `String` rather than an `Error`, because carrying the
    /// original would mean cloning it on every read of the flag; the message
    /// is the only part worth repeating.
    poisoned: Option<String>,
}

struct Enlisted {
    /// `db.table`, the same key the WAL cache uses.
    path: String,
    /// Staging group in that table's log. `None` when logging is off (an
    /// in-memory session, or one with `set_wal_enabled(false)`), in which case
    /// the transaction is purely an in-memory overlay.
    seq: Option<u64>,
    /// The log's LSN when this table was enlisted. ROLLBACK rewinds here.
    lsn: u64,
    /// Set when this table took a mutation the log cannot express -- a
    /// positional sweep on a table with no single-column primary key. COMMIT
    /// makes it durable by folding the parts to disk instead; see
    /// [`Session::fold_to_parts`].
    fold: bool,
}

/// What the subquery folder carries down the AST: how much nesting is left,
/// and the context whose budget and deadline the nested runs answer to.
///
/// Two words passed by `&mut` through a recursion that already existed, rather
/// than a second parameter on eight signatures.
struct Sub<'a> {
    left: usize,
    ctx: &'a QueryContext,
}

/// Transaction control. Not in the SQL grammar -- `src/sql` is not this
/// module's to extend -- so [`Session::run`] recognises it ahead of the parser.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TxnStmt {
    Begin,
    Commit,
    Rollback,
}

impl TxnStmt {
    fn name(self) -> &'static str {
        match self {
            TxnStmt::Begin => "BEGIN",
            TxnStmt::Commit => "COMMIT",
            TxnStmt::Rollback => "ROLLBACK",
        }
    }
}

/// Operator statements, recognised ahead of the parser for exactly the reason
/// transaction control and `SET` are: `Statement` is matched exhaustively in
/// this file and `src/sql` is not this module's to extend.
///
/// ```text
///   BACKUP TO '<archive>' [INCREMENTAL FROM '<base archive>']
///   RESTORE FROM '<archive>' [TO '<directory>'] [UNTIL <recovery target>]
///   VERIFY BACKUP '<archive>'
/// ```
///
/// The recovery target is spelled `LSN <n>`, `TIMESTAMP '<ts>'` or `LATEST`,
/// and it is turned into a [`crate::backup::Target`] here at parse time --
/// before a directory is created -- so `UNTIL LSN yesterday` costs nothing but
/// the sentence that says why.
#[derive(Debug)]
enum Admin {
    Backup { to: String, base: Option<String> },
    Restore { from: String, to: Option<String>, until: Option<crate::backup::Target> },
    Verify { archive: String },
}

// ------------------------------------------------- constraints and views
//
// Two things a table can now carry that `TableDef` has no room for: `CHECK`
// constraints, and -- for a database rather than a table -- stored views. Both
// are text plus a parsed AST, and both have to survive a restart, a backup and
// a restore.
//
// ## Where they live, and why it is a table
//
// In a table. `<db>._granular_ddl`, one row per constraint or view, created
// the first time a database has one and dropped when it has none.
//
// The alternative was a file beside `CATALOG`, and it was rejected for one
// reason: `backup.rs` archives *tables*. A sidecar file is not in an archive,
// so `RESTORE` would produce a database whose CHECK constraints had silently
// disappeared -- and the first write after that restore would be accepted
// where the original would have refused it. Storing them as rows means backup,
// incremental backup, restore, verify, the WAL, the checkpoint's atomicity and
// `collect_dropped_tables` all already handle them, with no second path to
// keep in step. The cost is that the table is visible: `SHOW TABLES` lists it,
// and a `SELECT` reads it. That is the honest trade -- there is no hidden
// state -- and writes to it are refused (see `guard_ddl_table`), because the
// engine's copy in memory is what enforcement reads.
//
// ## Durability without its own log
//
// Nothing here writes to the metadata table's write-ahead log. Every statement
// that changes it is DDL, and `Session::dispatch` checkpoints after DDL, so the
// commit point is the checkpoint's `CATALOG` rename -- one atomic publication
// covering the table that gained the constraint *and* the row that records it.
// Logging it as well would open a window the other way: a crash between the log
// append and the checkpoint would replay a row set that is a superset of the
// intended one, and since a DROP is expressed by a row's *absence*, a superset
// resurrects dropped views.

/// The metadata table's name, in each database that needs one.
const DDL_TABLE: &str = "_granular_ddl";

/// One `CHECK` constraint, as stored.
struct Check {
    /// `CONSTRAINT <name>`, or a generated one. Only ever printed.
    name: String,
    /// The predicate's source text, which is what the catalog row holds.
    /// `Expr`'s `Display` is fully parenthesized, so this round-trips through
    /// `parse_expr` unchanged -- the same property `DEFAULT` relies on.
    sql: String,
    /// The parsed predicate, bound afresh against the table's schema on each
    /// statement that writes. Kept as an `Expr` rather than a `BoundExpr`
    /// because a bound one holds column *indices*, and `ALTER TABLE ... DROP
    /// COLUMN` renumbers them: a cached index that survived a DDL statement
    /// would check the wrong column, silently.
    expr: crate::sql::ast::Expr,
}

/// One stored view.
struct View {
    /// The database an unqualified name in `sql` resolves against -- the
    /// session's current database when the view was created. `query` is
    /// already qualified with it; this is kept so the row can be written back
    /// out and re-qualified identically on the next open.
    scope: String,
    sql: String,
    /// The body, parsed and fully qualified. Cloned into the outer statement
    /// at every reference: a view is inlined, never materialized.
    query: crate::sql::ast::Query,
}

/// Everything the catalog cannot hold, keyed the way the write path asks for
/// it.
#[derive(Default)]
struct Extensions {
    /// `db.table` -> its constraints. Declaration order within a statement,
    /// and by name after a reload (the catalog rows are sorted so that two
    /// checkpoints of an unchanged database produce identical bytes); order
    /// only decides which of two violated constraints is named first. Empty on
    /// every database that declares none, which is what makes the check on the
    /// insert path one `is_empty` against a `HashMap` field.
    checks: crate::common::FastMap<String, Vec<Check>>,
    /// `db.view` -> the view.
    views: crate::common::FastMap<String, View>,
    /// `db.table` -> the column declared `UNIQUE`.
    ///
    /// One per table by construction: `check_unique_declarations` accepts the
    /// declaration only on the table's own unique key, and there is one of
    /// those. What it buys is the difference between an upsert and an error on
    /// a repeated key, so it has to survive a restart like any other part of
    /// the DDL -- a table that silently went back to last-write-wins after a
    /// reopen would lose rows exactly where the user asked to be told.
    uniques: crate::common::FastMap<String, String>,
}

impl Extensions {
    fn is_empty(&self) -> bool {
        self.checks.is_empty() && self.views.is_empty() && self.uniques.is_empty()
    }

    /// The view a table reference names, if it is one.
    ///
    /// Two probes at most, and the first is skipped entirely while no view
    /// exists. `db` is the session's current database, for a bare name.
    fn view(&self, name: &ObjectName, db: &str) -> Option<&View> {
        if self.views.is_empty() {
            return None;
        }
        self.views.get(&view_key(name, db))
    }
}

/// What the fold detector needs beyond the AST: whether a bare table name is a
/// view. Two words, copied down four levels of walk rather than borrowed
/// separately at each.
#[derive(Clone, Copy)]
struct Names<'a> {
    ext: &'a Extensions,
    db: &'a str,
}

impl Session {
    pub fn in_memory() -> Session {
        Session {
            catalog: Catalog::in_memory(),
            fold_bytes: crate::persist::wal::DEFAULT_FOLD_BYTES,
            fold_due: Vec::new(),
            masks: MaskRuns::default(),
            owner: 0,
            owner_tok: None,
            wals: Default::default(),
            wal_enabled: false,
            // Nothing on disk to guard: an in-memory session shares no files
            // with anyone, so it must never take (or contend for) the lock.
            _lock: None,
            txn: None,
            settings: crate::settings::Handle::new(Default::default()),
            read_only: false,
            limits: Limits::default(),
            log: crate::system::QueryLog::new(),
            ext: Extensions::default(),
        }
    }

    /// Open (or create) a persistent database rooted at `dir`.
    ///
    /// The directory is claimed exclusively for this process first (an
    /// `flock` on `<dir>/LOCK`), then recovery runs: parts are loaded, and each
    /// table's write-ahead log is replayed from the watermark its last
    /// checkpoint recorded, so writes acknowledged since that checkpoint are
    /// restored.
    pub fn open(dir: impl AsRef<Path>) -> Result<Session> {
        let mut catalog = Catalog::on_disk(dir)?;
        // Before `load_catalog`, not after: recovery rewrites the WAL
        // watermark and can truncate a log, which is already a mutation a
        // second process must not be racing.
        let root = catalog.dir().expect("on_disk always sets a directory").to_path_buf();
        let lock = lock_data_dir(&root, LockMode::Exclusive)?;
        crate::persist::load_catalog(&mut catalog)?;
        let mut s = Session {
            catalog,
            fold_bytes: crate::persist::wal::DEFAULT_FOLD_BYTES,
            fold_due: Vec::new(),
            masks: MaskRuns::default(),
            owner: 0,
            owner_tok: None,
            wals: Default::default(),
            wal_enabled: true,
            _lock: lock,
            txn: None,
            settings: crate::settings::Handle::new(Default::default()),
            read_only: false,
            limits: Limits { spill_root: Some(spill_root(&root)), ..Limits::default() },
            log: crate::system::QueryLog::new(),
            ext: Extensions::default(),
        };
        // Before the first query, and only from the writer: a spill directory
        // whose owner is gone is unlinked here, because `SpillDir`'s `Drop`
        // cannot run on a `SIGKILL`, a panic-abort or a power loss.
        crate::exec::operators::sort::spill::reap(&spill_root(&root));
        s.load_extensions()?;
        Ok(s)
    }

    /// Open `dir` for queries only, under a **shared** directory lock.
    ///
    /// Several read-only sessions -- in this process or in others -- hold the
    /// lock together; a writer's `LOCK_EX` excludes them all and they exclude
    /// it, which is the same single-writer rule [`Session::open`] enforces,
    /// only now with the reader side allowed to be plural. That is the whole
    /// of it: the exclusion the exclusive lock exists for is *two writers
    /// allocating the same part sequence number*, and a session that cannot
    /// write cannot do that.
    ///
    /// Every mutating statement is refused with an error naming the mode, and
    /// the write-ahead log is never opened, so nothing here can create or
    /// extend a file. Recovery still runs -- a log left behind by a crashed
    /// writer is replayed **into memory only**, so this session sees the
    /// acknowledged writes without persisting anything.
    pub fn open_read_only(dir: impl AsRef<Path>) -> Result<Session> {
        let mut catalog = Catalog::on_disk(dir)?;
        let root = catalog.dir().expect("on_disk always sets a directory").to_path_buf();
        let lock = lock_data_dir(&root, LockMode::Shared)?;
        crate::persist::load_catalog(&mut catalog)?;
        let mut s = Session {
            catalog,
            fold_bytes: crate::persist::wal::DEFAULT_FOLD_BYTES,
            fold_due: Vec::new(),
            masks: MaskRuns::default(),
            owner: 0,
            owner_tok: None,
            wals: Default::default(),
            wal_enabled: false,
            _lock: lock,
            txn: None,
            settings: crate::settings::Handle::new(Default::default()),
            read_only: true,
            // No `spill_root`, so spilling falls back to `env::temp_dir()`.
            // A session that has promised not to write the data directory must
            // not write scratch into it either: on the read-only media this
            // mode exists to open, `<data>/.spill` is not creatable and every
            // spilling query failed with a permission error where a plain read
            // succeeded -- and on writable media it left spill directories
            // nothing collects, because `open_read_only` runs no reaper.
            limits: Limits::default(),
            log: crate::system::QueryLog::new(),
            ext: Extensions::default(),
        };
        // A read-only session reads views and constraints like any other: the
        // load only touches the metadata table's own rows, which the recovery
        // above has already put in memory.
        s.load_extensions()?;
        Ok(s)
    }

    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// This session's spill root under a caller-supplied ceiling. The settings
    /// layer holds `max_temporary_data_on_disk` and the session holds the
    /// root, so neither can build the pair alone.
    pub(crate) fn spill_budget(
        &self,
        limit: u64,
    ) -> Arc<crate::exec::operators::SpillBudget> {
        self.limits.spill(limit)
    }

    /// `max_temporary_data_on_disk`, pushed in by `Settings::apply_to` the way
    /// the memory budget and the timeout are.
    pub fn set_temp_disk_limit(&mut self, bytes: u64) {
        self.limits.temp_disk = bytes;
    }

    // ------------------------------------------------------- query governance

    /// Cap the memory one query of this session may reserve. The default is
    /// [`operators::DEFAULT_MEM_BUDGET`].
    ///
    /// Per *query*, not per session: two queries running concurrently on two
    /// readers each get this much, which is the only reading under which the
    /// number means anything -- a shared budget makes two innocent queries
    /// refuse each other.
    pub fn set_memory_limit(&mut self, bytes: i64) {
        self.limits.mem = bytes;
    }

    /// Stop any query of this session that runs longer than `d`. Measured
    /// per statement, from the moment it starts.
    pub fn set_timeout(&mut self, d: Option<Duration>) {
        self.limits.timeout = d;
    }

    /// `wal_fold_bytes`, pushed in by `Settings::apply_to`. 0 disables the
    /// automatic fold, which is what the engine did before it existed.
    pub fn set_wal_fold_bytes(&mut self, bytes: u64) {
        self.fold_bytes = bytes;
    }

    /// A flag another thread can set to stop this session's queries.
    ///
    /// Shared, not snapshotted: setting it stops the query in flight *and*
    /// every one after it, until [`Session::resume`] clears it. That is what a
    /// disconnected client wants; a one-shot cancel would race the next
    /// statement.
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.limits.cancel)
    }

    /// Clear a cancellation, so the session takes queries again.
    pub fn resume(&self) {
        self.limits.cancel.store(false, Ordering::Relaxed);
    }

    /// The ring `system.query_log` reads, for an embedder that would rather
    /// ship these somewhere than query them.
    pub fn query_log(&self) -> &crate::system::QueryLog {
        &self.log
    }

    /// The view namespace, for the walk that decides whether a statement has
    /// anything to fold.
    fn names(&self) -> Names<'_> {
        Names { ext: &self.ext, db: self.catalog.current_database() }
    }

    /// `db.view` and the view itself, for a name that is one.
    fn view_of(&self, name: &ObjectName) -> Option<(String, &View)> {
        if self.ext.views.is_empty() {
            return None;
        }
        let key = view_key(name, self.catalog.current_database());
        self.ext.views.get(&key).map(|v| (key, v))
    }

    /// Record one finished statement.
    ///
    /// Called from both halves of the session -- the `&mut` dispatcher and the
    /// `&self` read path -- which is why the log is behind a handle rather
    /// than in the struct: a `Reader` running on another thread has only
    /// `&Session` and must still be visible here.
    ///
    /// ## What it costs a statement that never looks at the log
    ///
    /// One uncontended mutex, one `SystemTime::now`, and a `memcpy` of the
    /// statement text into a buffer the ring recycled -- no allocation once
    /// `LOG_CAPACITY` statements have run. A/B interleaved through a temporary
    /// switch, 12 runs a side of 200k point statements each (the shortest
    /// statement there is, so the worst case for a fixed per-statement cost):
    /// **5.042 us against 5.169 us best-of-12, 5.33 against 5.38 by median**,
    /// i.e. 50-130 ns on a 5.3 us statement and inside this machine's own
    /// run-to-run spread. On a 1.2 ms analytic query it is not measurable at
    /// all -- both sides moved 5-9% between runs of identical code.
    ///
    /// The first cut allocated the text with `Box<str>` and measured 129 ns;
    /// recycling the evicted entry's buffers (see
    /// [`crate::system::QueryLog::record`]) is what halved it.
    fn log_stmt(
        &self,
        sql: &str,
        kind: &'static str,
        started: Instant,
        r: &Result<ResultSet>,
    ) {
        let (rows, s) = match r {
            Ok(rs) => (rs.affected.unwrap_or(rs.stats.rows) as u64, rs.stats),
            Err(_) => (0, QueryStats::default()),
        };
        self.log.record(
            sql,
            kind,
            crate::system::Counters {
                rows,
                // From the caller's clock rather than `stats.elapsed_us`,
                // which is stamped before this point on some paths and not at
                // all on others. A log whose durations disagree with the
                // shell's timing footer is a log nobody trusts twice.
                elapsed_us: started.elapsed().as_micros().min(u64::MAX as u128) as u64,
                granules_read: s.granules_read,
                granules_pruned: s.granules_pruned,
                rows_scanned: s.rows_scanned,
            },
            // Formatted only on the failure path, which is not the one being
            // measured -- and where an allocation is nothing next to whatever
            // just went wrong.
            r.as_ref().err().map(|e| e.to_string()).as_deref(),
        );
    }

    /// Turn write-ahead logging off for this session.
    ///
    /// Bulk loading is the case that wants this: an `fsync` per statement is
    /// the dominant cost when the whole job is "ingest a billion rows and
    /// checkpoint once", and the log buys nothing if you would re-run the load
    /// after a crash anyway. Writes are then durable only at
    /// [`Session::checkpoint`].
    pub fn set_wal_enabled(&mut self, on: bool) {
        self.wal_enabled = on && self.catalog.is_persistent() && !self.read_only;
    }

    /// Persist everything: flush write buffers, rewrite parts, and truncate
    /// the logs whose records are now inside those parts.
    ///
    /// Refused inside a transaction, and not out of caution: a checkpoint
    /// writes the parts a table's `snapshot()` reports, which inside a
    /// transaction is the *uncommitted* overlay -- so it would make a
    /// transaction durable that ROLLBACK is still entitled to erase, and then
    /// truncate the log that was the only record of the boundary.
    pub fn checkpoint(&mut self) -> Result<()> {
        if self.read_only {
            return Err(read_only_err("CHECKPOINT"));
        }
        if self.txn.is_some() {
            return Err(Error::unsupported(
                "cannot checkpoint inside a transaction: it would persist \
                 uncommitted parts. COMMIT or ROLLBACK first",
            ));
        }
        if !self.catalog.is_persistent() {
            return Ok(());
        }
        self.catalog.flush_all()?;
        // This checkpoint covers every table, so any pending per-table fold is
        // work that is about to be done twice.
        self.fold_due.clear();
        // Drop the cached handles first: `save_catalog` rolls each log behind
        // this session's back, which leaves a cached `Wal` holding the segment
        // that was just sealed -- and an append into a segment that is no
        // longer the newest is a record replay will never reach.
        self.wals.clear();
        crate::persist::save_catalog(&mut self.catalog)
    }

    // ------------------------------------------- constraints and views: I/O

    /// Read every database's `_granular_ddl` into [`Session::ext`].
    ///
    /// Runs once, at open, before any statement. A row naming a table that no
    /// longer exists is dropped *in memory only* -- see
    /// [`Session::run_rename_table`], which deliberately leaves such a row
    /// behind for a window so that a crash mid-rename cannot lose a constraint.
    fn load_extensions(&mut self) -> Result<()> {
        for db in self.catalog.database_names() {
            let path = format!("{db}.{DDL_TABLE}");
            // A quarantined metadata table is the one damage this engine
            // cannot degrade around: the rows it could not read are the rules
            // the write path enforces, so opening anyway would accept writes
            // the database was declared to refuse. Loud, and recoverable --
            // restore the file, or delete the directory to give up the
            // constraints deliberately.
            if self.catalog.is_quarantined(&path) {
                return Err(Error::corruption(format!(
                    "`{path}` did not decode, and it holds this database's CHECK and \
                     UNIQUE constraints and its views. Refusing to open `{db}` with its \
                     rules unknown: restore the file from a backup, or remove \
                     `{db}/{DDL_TABLE}` from the data directory to drop the constraints \
                     on purpose"
                )));
            }
            if self.catalog.table_by_path(&path).is_err() {
                continue;
            }
            let cols: Vec<usize> = (0..DDL_COLUMNS.len()).collect();
            // Recovery may have replayed rows into the delta, which a scan
            // does not see.
            self.catalog.table_by_path_mut(&path)?.flush()?;
            let blocks = self.catalog.table_by_path_mut(&path)?.scan(&cols)?;
            for b in &blocks {
                for r in 0..b.rows() {
                    let cell = |c: usize| match b.column(c).value(r) {
                        Value::Str(s) => s.to_string(),
                        other => other.to_string(),
                    };
                    self.install_ext_row(&db, &cell(0), cell(1), cell(2), cell(3), cell(4))?;
                }
            }
        }
        // The metadata table names the objects it describes; the catalog says
        // which of them exist. Anything the catalog does not have is a leftover
        // from a rename or a drop that was interrupted, and applying it would
        // mean a constraint attached to nothing.
        self.prune_ext();
        Ok(())
    }

    /// Turn one metadata row into an entry. Errors here are corruption: the
    /// rows were written by this code and re-parsed by the same parser.
    fn install_ext_row(
        &mut self,
        db: &str,
        kind: &str,
        object: String,
        name: String,
        scope: String,
        sql: String,
    ) -> Result<()> {
        let key = format!("{db}.{object}");
        match kind {
            "CHECK" => {
                let expr = crate::sql::parser::parse_expr(&sql).map_err(|e| {
                    Error::corruption(format!(
                        "`{db}.{DDL_TABLE}` holds an unparseable CHECK for `{key}`: {e}"
                    ))
                })?;
                self.ext.checks.entry(key).or_default().push(Check { name, sql, expr });
            }
            "VIEW" => {
                let mut query = parse_view_body(&sql).map_err(|e| {
                    Error::corruption(format!(
                        "`{db}.{DDL_TABLE}` holds an unparseable view `{key}`: {e}"
                    ))
                })?;
                query.qualify_tables(&scope);
                self.ext.views.insert(key, View { scope, sql, query });
            }
            "UNIQUE" => {
                self.ext.uniques.insert(key, name);
            }
            other => {
                return Err(Error::corruption(format!(
                    "`{db}.{DDL_TABLE}` holds a row of unknown kind `{other}`"
                )))
            }
        }
        Ok(())
    }

    /// Forget every constraint whose table, and every view whose database, the
    /// catalog no longer has.
    fn prune_ext(&mut self) {
        if self.ext.is_empty() {
            return;
        }
        let catalog = &self.catalog;
        self.ext.checks.retain(|k, _| catalog.table_by_path(k).is_ok());
        self.ext.uniques.retain(|k, _| catalog.table_by_path(k).is_ok());
        let dbs = catalog.database_names();
        self.ext
            .views
            .retain(|k, _| dbs.iter().any(|d| k.starts_with(d) && k[d.len()..].starts_with('.')));
    }

    /// Refuse a name that would take a directory entry something else already
    /// owns: `None` checks a database against the root's siblings and the
    /// root's own two file names, `Some(db)` checks a table against that
    /// database's siblings.
    ///
    /// Case-folded, which is the whole point -- see [`store::folds_onto`] for
    /// what the fold prevents and why the answer is a refusal. One pass over a
    /// name list that is DDL-sized, on statements that are about to `mkdir`
    /// and `fsync`; nothing per row and nothing in a loop.
    fn guard_dir_name(&self, db: Option<&str>, name: &str) -> Result<()> {
        use crate::persist::store;
        let hit = match db {
            None => {
                if let Some(r) = store::ROOT_RESERVED.iter().find(|r| r.eq_ignore_ascii_case(name))
                {
                    return Err(Error::storage(format!(
                        "refusing to create `DATABASE {name}`: the data directory keeps its \
                         own `{r}` file under that name. Creating a directory there wins the \
                         race against the file and leaves the database permanently \
                         unopenable"
                    )));
                }
                store::folds_onto(name, self.catalog.db_names())
            }
            Some(db) => store::folds_onto(name, self.catalog.table_names_in(db)),
        };
        let what = if db.is_none() { "DATABASE" } else { "TABLE" };
        match hit {
            Some(other) => Err(store::name_collision(what, name, other)),
            None => Ok(()),
        }
    }

    /// Rewrite `<db>._granular_ddl` from what is in memory.
    ///
    /// Whole-table, not incremental: the row set for one database is a handful
    /// of short strings, and rewriting it makes "what is in the table" and
    /// "what is in memory" the same statement rather than two that can drift.
    /// The table is dropped outright when the last constraint or view in a
    /// database goes, so an ordinary database has no such table at all.
    ///
    /// Not logged; see the module note above. The caller is DDL and
    /// [`Session::dispatch`] checkpoints behind it.
    fn persist_ext(&mut self, db: &str) -> Result<()> {
        if !self.catalog.is_persistent() {
            return Ok(());
        }
        let mut rows: Vec<[String; 5]> = Vec::new();
        let prefix = format!("{db}.");
        for (k, checks) in &self.ext.checks {
            let Some(object) = k.strip_prefix(&prefix) else { continue };
            for c in checks {
                rows.push([
                    "CHECK".into(),
                    object.into(),
                    c.name.clone(),
                    String::new(),
                    c.sql.clone(),
                ]);
            }
        }
        for (k, v) in &self.ext.views {
            let Some(object) = k.strip_prefix(&prefix) else { continue };
            rows.push([
                "VIEW".into(),
                object.into(),
                String::new(),
                v.scope.clone(),
                v.sql.clone(),
            ]);
        }
        for (k, col) in &self.ext.uniques {
            let Some(object) = k.strip_prefix(&prefix) else { continue };
            rows.push([
                "UNIQUE".into(),
                object.into(),
                col.clone(),
                String::new(),
                String::new(),
            ]);
        }
        // Two writes of the same database must produce the same table, or an
        // unchanged catalog would checkpoint differently every time.
        rows.sort_unstable();

        let name = ObjectName(vec![db.to_string(), DDL_TABLE.to_string()]);
        let path = format!("{db}.{DDL_TABLE}");
        self.catalog.drop_table(&name, true)?;
        if rows.is_empty() {
            return Ok(());
        }
        self.catalog.create_table(ddl_table_def(&path)?, false)?;
        let mut cols: Vec<ColumnBuilder> = (0..DDL_COLUMNS.len())
            .map(|_| ColumnBuilder::with_capacity(DataType::String, rows.len()))
            .collect();
        for r in &rows {
            for (i, cell) in r.iter().enumerate() {
                cols[i].push_value(&Value::str(cell.as_str()))?;
            }
        }
        let block = Block::new(cols.into_iter().map(|b| b.finish()).collect())?;
        self.catalog.table_by_path_mut(&path)?.insert(block)?;
        Ok(())
    }

    // ------------------------------------------------------ backup / restore

    fn run_admin(&mut self, a: &Admin) -> Result<ResultSet> {
        match a {
            Admin::Backup { to, base } => self.run_backup(to, base.as_deref()),
            Admin::Restore { from, to, until } => self.run_restore(from, to.as_deref(), *until),
            Admin::Verify { archive } => {
                let r = crate::backup::verify(Path::new(archive))?;
                report(
                    &["archive", "tables", "parts", "rows", "bytes"],
                    Value::str(archive.as_str()),
                    &[r.tables as u64, r.parts as u64, r.rows, r.bytes],
                )
            }
        }
    }

    /// `BACKUP TO '<archive>' [INCREMENTAL FROM '<base>']`.
    ///
    /// One [`crate::storage::part::Snapshot`] per table, all of them taken
    /// here, under the write borrow this statement already holds -- so the
    /// archive is one instant of the whole database rather than one instant
    /// per table. From there on nothing in the catalog is read: parts are
    /// immutable, so a writer that publishes a new `PartSet` while the bytes
    /// are going out cannot change what this archive contains. That is the
    /// property a `cp -r` cannot have, and the reason two of eight such copies
    /// of a live instance were unopenable.
    ///
    /// Allowed on a read-only session: it writes nothing to the database.
    fn run_backup(&mut self, to: &str, base: Option<&str>) -> Result<ResultSet> {
        if self.txn.is_some() {
            return Err(Error::unsupported(
                "cannot BACKUP inside a transaction: `snapshot()` would hand it the \
                 transaction's uncommitted overlay, which ROLLBACK is still entitled to \
                 erase. COMMIT or ROLLBACK first",
            ));
        }
        // Buffered rows are in the delta and an archive is made of parts, so
        // "acknowledged" and "archived" are only the same set after this line.
        self.catalog.flush_all()?;
        if let Some(&(path, dp)) = self.catalog.damaged_parts().first() {
            return Err(Error::corruption(format!(
                "refusing to back up `{path}`: it is quarantined because `{}` did not \
                 decode, so an archive taken now would be silently missing whatever rows \
                 that file held. Repair or DROP the table first. ({})",
                dp.file, dp.why
            )));
        }

        let mut dbs = self.catalog.database_names();
        dbs.sort();
        let mut roster: Vec<(String, String)> = Vec::new();
        for db in &dbs {
            for t in self.catalog.table_names(Some(db))? {
                roster.push((db.clone(), t));
            }
        }
        let mut src = Vec::with_capacity(roster.len());
        for (db, name) in &roster {
            let t = self.catalog.table_by_path(&format!("{db}.{name}"))?;
            // A `Memory` table is defined to vanish on restart; archiving one
            // would restore it as something that does not, which is the same
            // silent change of semantics `save_catalog` declines to make.
            if !t.def.engine.is_persistent() {
                continue;
            }
            src.push(crate::backup::Source { db, def: &t.def, snap: t.snapshot() });
        }
        let r = crate::backup::write_archive(
            Path::new(to),
            &dbs,
            &src,
            base.map(Path::new),
            self.catalog.instance(),
        )?;
        report(
            &["archive", "tables", "parts", "rows", "bytes", "reused_parts"],
            Value::str(to),
            &[r.tables as u64, r.parts as u64, r.rows, r.bytes, r.reused as u64],
        )
    }

    /// `RESTORE FROM '<archive>' TO '<directory>' [UNTIL <recovery target>]`.
    ///
    /// The target directory is mandatory, and it may not be the directory this
    /// session has open. Both refusals are the same rule: a restore that wrote
    /// into a live database would interleave its part sequence numbers and
    /// commit records with the ones already there, and the result would be
    /// neither database. Restore beside it and swap the directories --
    /// `rename` is atomic and the old copy survives the mistake.
    ///
    /// `UNTIL` rolls the unpacked copy forward through the *open* database's
    /// WAL archive, which it reads and never writes -- that is what makes a
    /// point-in-time recovery legal while the source is serving. Which is also
    /// why the two directories being distinct is checked once, above, before
    /// either branch: an `UNTIL` that reached its own copy of the check would
    /// be a second place for it to be wrong, and the failure it guards against
    /// is unrecoverable by construction.
    ///
    /// Deliberately *not* checkpointed first. A recovery statement that wrote
    /// to the database it is recovering from would be refused on a read-only
    /// session and would change the source's archive under an operator who is
    /// mid-incident; when the tail really is needed, `check_target` says so and
    /// names the table still holding it.
    fn run_restore(
        &mut self,
        from: &str,
        to: Option<&str>,
        until: Option<crate::backup::Target>,
    ) -> Result<ResultSet> {
        let Some(to) = to else {
            return Err(Error::unsupported(
                "RESTORE needs a target: `RESTORE FROM '<archive>' TO '<directory>'`. It is \
                 never the open database -- restore beside it and swap the directories.",
            ));
        };
        let target = Path::new(to);
        if let Some(root) = self.catalog.dir() {
            if let Some(exact) = overlaps(root, target) {
                return Err(Error::storage(format!(
                    "refusing to restore into {to}: this session has that database open{}. \
                     Restore to a new directory and swap.",
                    if exact {
                        String::new()
                    } else {
                        format!(
                            " at {}, and one of the two directories is inside the other -- \
                             which would leave a second CATALOG and a second set of part \
                             directories inside the tree the loader walks",
                            root.display()
                        )
                    }
                )));
            }
        }
        let r = match until {
            None => crate::backup::restore(Path::new(from), target)?,
            Some(t) => {
                let Some(root) = self.catalog.dir() else {
                    return Err(Error::unsupported(
                        "RESTORE ... UNTIL rolls forward through the archived write-ahead \
                         log of the open database, and this session has none -- it is \
                         in memory. Open the data directory whose archive holds the log \
                         and run it there; `RESTORE FROM ... TO ...` without `UNTIL` \
                         unpacks the archive's own instant anywhere.",
                    ));
                };
                crate::backup::restore_until(Path::new(from), root, target, t)?
            }
        };
        // `replayed` is reported whether or not `UNTIL` was given, exactly as
        // `BACKUP` reports `reused_parts` without `INCREMENTAL`: one shape per
        // statement is one thing for a script to read, and "replayed 0" is the
        // answer a plain restore should give rather than a missing column.
        report(
            &["directory", "tables", "parts", "rows", "replayed"],
            Value::str(to),
            &[r.tables as u64, r.parts as u64, r.rows, r.replayed],
        )
    }

    /// Open (or reuse) the cached log handle for `path`.
    fn wal_for(&mut self, path: &str) -> Result<Option<&mut crate::persist::Wal>> {
        if !self.wal_enabled {
            return Ok(None);
        }
        let Some(root) = self.catalog.dir() else { return Ok(None) };
        if !self.wals.contains_key(path) {
            let (db, tbl) = path.split_once('.').unwrap_or(("default", path));
            let p = crate::persist::wal::wal_dir(root, db, tbl);
            self.wals
                .insert(path.to_string(), crate::persist::Wal::open(&p)?);
        }
        Ok(Some(self.wals.get_mut(path).expect("just inserted")))
    }

    /// Seal a dropped table's log, so what it still holds joins the archive.
    ///
    /// The log lives at `<root>/.wal/<db>/<table>` and deliberately outlives
    /// `DROP TABLE`: "restore to the moment before the drop" is the commonest
    /// point-in-time recovery there is, and it needs those records. What the
    /// drop must not leave behind is an **active** segment holding them,
    /// because two things follow from it and both are wrong.
    ///
    /// The first is a silent loss. [`crate::persist::wal::recover`] reads
    /// sealed segments only -- the active one may end in an interrupted
    /// append -- and nothing will ever roll a dead table's log again, so
    /// records written and then dropped inside one checkpoint interval become
    /// permanently unreachable. `RESTORE ... UNTIL LATEST` reports success and
    /// hands back a database missing them.
    ///
    /// The second is a wedge. [`crate::persist::wal::archive_lags`] walks log
    /// directories rather than the catalog, so the dead table answers "still
    /// holds un-archived records" forever, and every `RESTORE ... UNTIL
    /// TIMESTAMP` past it is refused with advice -- checkpoint first -- that
    /// cannot ever help.
    ///
    /// One roll at the drop settles both: the records move into the archive,
    /// where they are exactly as recoverable and no longer lag.
    ///
    /// **What it costs**, measured, because it is not free: two `fsync`-class
    /// operations -- the successor segment's atomic publish -- for a table
    /// whose log still holds records, and *nothing* for one whose does not.
    /// On a loop that creates, writes to and drops a table 120 times that is
    /// +15% against the same build without it (best-of-9, interleaved, A/A
    /// floor 6%); on a loop that drops tables with empty logs it is inside the
    /// noise. The trade is those two `fsync`s against the records being
    /// unrecoverable, and a `DROP TABLE` already checkpoints, so it is not a
    /// new order of cost on that path.
    fn retire_log(&mut self, db: &str, table: &str) -> Result<()> {
        if !self.wal_enabled {
            return Ok(());
        }
        let Some(root) = self.catalog.dir().map(Path::to_path_buf) else { return Ok(()) };
        // The handle goes either way: after the roll it names a segment that
        // is no longer the newest, and an append into one of those is a record
        // replay would never reach.
        //
        // Rolling *through* it when this session has one is the whole reason
        // the cache is consulted here -- it is already open at the right
        // segment, so the retire costs the fsyncs it cannot avoid and not a
        // `read_dir` plus a rescan of the active segment on top. Which is the
        // common shape: a table with anything worth sealing was written to by
        // this session, and writing to it is what opened the handle.
        match self.wals.remove(&format!("{db}.{table}")) {
            Some(mut w) => w.roll(),
            // Never written through this session, so open the directory only
            // if something is actually there -- a `Memory` table or one that
            // was created and dropped unused must not be given a log on its
            // way out.
            None => crate::persist::wal::roll_for_checkpoint(&crate::persist::wal::wal_dir(
                &root, db, table,
            ))
            .map(drop),
        }
    }

    /// Append a block to a table's log.
    ///
    /// Outside a transaction the record is committed the instant it is framed,
    /// and fsynced before the mutation it describes is acknowledged, so a crash
    /// between the two replays the write rather than losing it.
    ///
    /// Inside one the record is *staged* under the transaction's sequence
    /// number and deliberately **not** synced. Nothing is acknowledged until
    /// COMMIT, and COMMIT's marker plus its single fsync is what makes the
    /// whole group durable at once -- so a thousand-statement transaction
    /// pays one fsync rather than a thousand, and a crash in the middle of it
    /// replays nothing at all, because an unreleased staging group is dropped
    /// by [`crate::persist::Wal::replay`] by construction.
    fn log_insert(&mut self, path: &str, b: &Block) -> Result<()> {
        let seq = self.enlist(path)?;
        let t = self.fold_bytes;
        let Some(w) = self.wal_for(path)? else { return Ok(()) };
        match seq {
            Some(s) => {
                w.append_insert_staged(s, b)?;
            }
            None => {
                w.append_insert(b)?;
                w.sync()?;
            }
        }
        // One `u64` compare per logged *block*, against an append and (outside
        // a transaction) an fsync the same block already pays. `w.pending()`
        // is a field subtraction, not a `stat` -- and it is `pending` rather
        // than `len` because an LSN is a *stream* position now and never
        // restarts, so a threshold compared against `len` would be permanently
        // true after the first 64 MiB and would queue a fold on every
        // statement forever.
        if t != 0 && w.pending() >= t {
            self.mark_fold_due(path);
        }
        Ok(())
    }

    /// Append one key delete per lane, enlisting once for the batch.
    ///
    /// The delete counterpart to [`Session::log_insert`], with the same two
    /// durability rules -- and one more that only a bulk statement needs: the
    /// enlistment, the log-handle lookup *and the write* are hoisted out of
    /// the loop. Doing the first two per record costs a linear scan of the
    /// transaction's table list and a string compare; doing the third per
    /// record costs a syscall, which on a nineteen-byte record is the entire
    /// operation. Measured on a 50 000-row `DELETE` through a persistent
    /// session, best-of-5 interleaved: 881 ms per-record, 6.9 ms batched --
    /// **127x**, and the residue is the sweep plus the single fsync. The
    /// table of sizes is next to the mutation section below.
    fn log_deletes(&mut self, path: &str, lanes: &[u64]) -> Result<()> {
        let seq = self.enlist(path)?;
        let t = self.fold_bytes;
        let Some(w) = self.wal_for(path)? else { return Ok(()) };
        match seq {
            Some(s) => {
                w.append_deletes_staged(s, lanes)?;
            }
            None => {
                w.append_deletes(lanes)?;
                w.sync()?;
            }
        }
        if t != 0 && w.pending() >= t {
            self.mark_fold_due(path);
        }
        Ok(())
    }

    /// [`Session::log_deletes`] for an unkeyed table: one `TAG_MASK_RUN` per
    /// part the sweep touched, naming positions instead of key lanes.
    ///
    /// Same two durability rules and the same fold threshold. What it replaces
    /// is not a cheaper record but a whole table rewrite -- see
    /// [`Session::apply_sweep`].
    fn log_masks(&mut self, path: &str, masks: &MaskRuns) -> Result<()> {
        let seq = self.enlist(path)?;
        let t = self.fold_bytes;
        let Some(w) = self.wal_for(path)? else { return Ok(()) };
        w.append_masks(seq, masks)?;
        if seq.is_none() {
            w.sync()?;
        }
        if t != 0 && w.pending() >= t {
            self.mark_fold_due(path);
        }
        Ok(())
    }

    /// Note that `path`'s log has outgrown `wal_fold_bytes`.
    ///
    /// `#[cold]` and allocating on purpose: this runs once per threshold-worth
    /// of log -- 64 MiB by default -- not once per row or once per block, and
    /// the linear scan is over a list that holds one entry per *table* a
    /// statement has written to.
    #[cold]
    fn mark_fold_due(&mut self, path: &str) {
        if !self.fold_due.iter().any(|p| p == path) {
            self.fold_due.push(path.to_string());
        }
    }

    /// Fold every table whose log has outgrown the threshold.
    ///
    /// Called at a statement boundary with no transaction open, which is the
    /// only place it is safe: the log is written *before* the mutation it
    /// describes is applied (see the ordering note on `apply_insert`), so a
    /// fold any earlier would write a table out and then truncate a log whose
    /// last record had not reached it. Inside a transaction the overlays are
    /// still private, so the entries simply wait for `COMMIT`.
    ///
    /// One `Vec` length check per statement when nothing is due, which is
    /// every statement between two folds.
    fn drain_folds(&mut self) -> Result<()> {
        while let Some(p) = self.fold_due.pop() {
            self.fold_to_parts(&p).map_err(|e| {
                Error::storage(format!(
                    "the statement succeeded and its write is durable in the log, but \
                     folding `{p}`'s log into parts afterwards failed: {e}. Nothing is \
                     lost -- the log replays at the next open -- but it cannot be \
                     truncated until this succeeds, so it will keep growing"
                ))
            })?;
        }
        Ok(())
    }

    // --------------------------------------------------------- transactions

    /// True between `BEGIN` and its `COMMIT`/`ROLLBACK`.
    pub fn in_transaction(&self) -> bool {
        self.txn.is_some()
    }

    /// Open a transaction. Writes go to per-table overlays until `commit`.
    ///
    /// Costs one `Option` write: no table is touched and no log is opened
    /// until the transaction actually writes to one.
    ///
    /// A nested `BEGIN` is refused **and poisons the transaction it found**.
    /// Leaving the outer one merely open was the whole bug: the inner block
    /// then ran to its own `COMMIT`, which durably committed the outer
    /// transaction's uncommitted work at a boundary nobody had asked for.
    /// Poisoning turns that premature commit into a refusal, which is the one
    /// answer that cannot silently publish somebody else's rows.
    pub fn begin(&mut self) -> Result<()> {
        if let Some(txn) = self.txn.as_mut() {
            // Somebody else's transaction is not this connection's to poison:
            // the refusal below is a diagnosis of *this* block's mistake, and
            // stamping it on another connection would let any caller kill a
            // transaction it cannot even see.
            if txn.owner != self.owner {
                return Err(foreign_txn("BEGIN"));
            }
            let msg = "a transaction is already open; nested transactions are not supported";
            txn.poisoned.get_or_insert_with(|| msg.to_string());
            return Err(Error::unsupported(msg));
        }
        self.txn = Some(Txn { owner: self.owner, tok: self.owner_tok.clone(), ..Txn::default() });
        Ok(())
    }

    /// Discard an open transaction that nothing can ever end.
    ///
    /// Reached from exactly two places, both of which have already decided
    /// that the `Db` clone which ran `BEGIN` is gone: [`Db::drop`], which is
    /// the prompt path, and [`Db::writer`], which is the backstop for when the
    /// prompt one could not take the lock. Without it the owner token turned a
    /// client disconnect into a permanent wedge -- owners are monotonic, so
    /// the identity that could have ended the transaction was unrecoverable
    /// and every statement from every connection was refused forever.
    ///
    /// A rewind failure is swallowed on purpose: the half that matters is
    /// detaching the overlay, which is infallible, and the caller here is a
    /// `drop` or an unrelated statement, neither of which authored the
    /// transaction or has any action to take. Dead bytes in the log are what
    /// replay already discards.
    #[cold]
    fn reap_txn(&mut self) {
        if let Some(txn) = self.txn.take() {
            let _ = self.discard(txn);
        }
    }

    /// Refuse to end a transaction this connection did not open.
    ///
    /// The same rule the nested-`BEGIN` refusal enforces, one scope out: there
    /// the inner block's `COMMIT` would have published the outer block's
    /// uncommitted work, here a second connection's would. One `u64` compare
    /// per `COMMIT`/`ROLLBACK`, and none at all outside a transaction.
    #[inline]
    fn check_owner(&self, what: &str) -> Result<()> {
        match &self.txn {
            Some(t) if t.owner != self.owner => Err(foreign_txn(what)),
            _ => Ok(()),
        }
    }

    /// Make the transaction's writes durable and then visible, in that order.
    ///
    /// A failure anywhere in the durable half rolls the whole thing back, so
    /// COMMIT either happens or does not -- it never half-happens and then
    /// reports an error over a table that has already moved.
    ///
    /// A poisoned transaction is rolled back and the failure reported. It is
    /// deliberately **not** an `Ok` that quietly discards (which is what
    /// `COMMIT` did before, and what PostgreSQL reports as `ROLLBACK`): a
    /// client that gets `Ok` from `COMMIT` is entitled to believe its writes
    /// landed, and here they did not.
    pub fn commit(&mut self) -> Result<()> {
        // Before the `take`, because a refused COMMIT must leave the other
        // connection's transaction exactly where it found it.
        self.check_owner("COMMIT")?;
        // Taken up front so the durable half can borrow the roster while it
        // holds `&mut self` for the catalog and the logs -- the alternative is
        // a `Vec` of cloned paths per COMMIT, and a transaction is allowed to
        // be as small as one statement.
        let Some(txn) = self.txn.take() else {
            return Err(Error::exec("COMMIT without an open transaction"));
        };
        if let Some(why) = &txn.poisoned {
            let e = Error::exec(format!(
                "COMMIT refused and the transaction rolled back: an earlier statement \
                 in it failed ({why})"
            ));
            self.txn = Some(txn);
            let _ = self.rollback();
            return Err(e);
        }
        match self.commit_durable(&txn.tables) {
            Ok(()) => {
                // Infallible: one pointer store per enlisted table. This is
                // why COMMIT is O(parts published) -- in fact O(tables) --
                // rather than O(rows).
                for e in &txn.tables {
                    if let Ok(t) = self.catalog.table_by_path_mut(&e.path) {
                        t.commit_txn();
                    }
                }
                // Strictly after the publish, and this is the one place where
                // visibility precedes durability: `write_table` persists what
                // `snapshot()` reports, which before `commit_txn` is still the
                // private overlay, so folding first would put uncommitted rows
                // on disk. Nothing can observe the gap -- `commit` is
                // synchronous and a crash inside it takes the observer with it
                // -- and a fold that fails reports an error over a
                // transaction that is visible but not yet durable, which is
                // the same shape as the multi-table caveat in the module docs.
                for e in txn.tables.iter().filter(|e| e.fold) {
                    self.fold_to_parts(&e.path)?;
                }
                Ok(())
            }
            Err(e) => {
                // Put it back so the ordinary rollback path can undo whatever
                // the durable half managed before it failed.
                self.txn = Some(txn);
                let _ = self.rollback();
                Err(e)
            }
        }
    }

    /// Refuse a statement that must not run against the open transaction:
    /// one belonging to another connection, or one that has already failed.
    ///
    /// One `Option` test on the fast path -- `None` for every autocommit
    /// statement -- and one `u64` compare inside a healthy transaction.
    ///
    /// The owner arm is not a nicety. This engine holds **one** transaction
    /// per shared `Session`, so a second connection's statement used to
    /// *enlist in it*: its rows went into an overlay the other connection was
    /// free to roll back, and it was told they had landed. That is the
    /// nested-`BEGIN` failure again with connections in place of blocks, and
    /// it is refused for the same reason. Refusing reads too matches
    /// [`Reader`], which already declines to read through an open transaction
    /// rather than serve a dirty row.
    #[inline]
    fn check_txn(&self, kind: &str) -> Result<()> {
        match self.txn.as_ref() {
            None => Ok(()),
            Some(t) if t.owner != self.owner => Err(foreign_txn(kind)),
            Some(t) => match t.poisoned.as_deref() {
                None => Ok(()),
                Some(why) => Err(poisoned_err(why)),
            },
        }
    }

    /// Record `e` as the reason the open transaction (if any) is poisoned, and
    /// hand it back unchanged.
    /// Only ever this connection's transaction. A statement refused *because*
    /// the open transaction belongs to somebody else must not then poison it:
    /// that would hand any caller a way to kill a transaction it is not even
    /// allowed to read.
    #[cold]
    fn poison(&mut self, e: Error) -> Error {
        let owner = self.owner;
        if let Some(txn) = self.txn.as_mut().filter(|t| t.owner == owner) {
            txn.poisoned.get_or_insert_with(|| e.to_string());
        }
        e
    }

    /// The fallible half of COMMIT: flush the buffered rows into each overlay,
    /// then make every enlisted log durable behind one commit point.
    ///
    /// Durability strictly before visibility. Nothing has been published when
    /// this returns -- the overlays are still private -- so an error here is
    /// undone by dropping them.
    ///
    /// A *folding* table is the one exception, and it is deliberate: its
    /// commit point is the `TABLE` rename inside [`Session::fold_to_parts`],
    /// not a marker. Writing a marker as well would open a window in which the
    /// log's `Insert` records are released but the sweep's tombstones are not
    /// yet in a part -- a crash there would replay the inserts and resurrect
    /// exactly the rows the statement deleted. With no marker, a crash before
    /// the fold drops the whole staged group, which is what a COMMIT that did
    /// not finish means.
    fn commit_durable(&mut self, tables: &[Enlisted]) -> Result<()> {
        for e in tables {
            self.catalog.table_by_path_mut(&e.path)?.flush()?;
        }
        // A table whose COMMIT folds writes no marker: the records its group
        // staged are going into the parts instead, and replay drops the group
        // by construction. Telling the log so is what keeps `Wal::roll`'s
        // "no transaction spans a roll" guard from refusing the fold that
        // immediately follows -- which is the default shape of an unkeyed
        // DELETE or UPDATE inside a transaction.
        for e in tables.iter().filter(|e| e.fold && e.seq.is_some()) {
            if let Some(w) = self.wals.get_mut(&e.path) {
                w.drop_group();
            }
        }
        // Across several tables this is a two-phase commit, because N markers
        // in N files fsynced one after another are N commit points, and a
        // crash between two of them left a *prefix* of the transaction durable
        // -- some tables holding it and some not, with no error to anyone.
        // The last participant is the coordinator: every earlier one logs a
        // prepare citing its decision, and the coordinator's own marker, the
        // last append and the last fsync of the whole COMMIT, releases all of
        // them at once. See [`crate::persist::Wal::prepare`].
        //
        // A single-table transaction pays nothing for it: no prepare, no extra
        // record, the same one marker and one fsync it always had -- which is
        // the shape every autocommit statement and the whole OLTP path have.
        let Some(last) = tables.iter().rposition(|e| !e.fold && e.seq.is_some()) else {
            return Ok(());
        };
        let seq = tables[last].seq.expect("rposition matched on `is_some`");
        let parties = tables.iter().filter(|e| !e.fold && e.seq.is_some()).count();
        let missing = || Error::storage("an enlisted table with a sequence number has no log");
        if parties > 1 {
            // One `PathBuf` for the whole transaction, and only when there is
            // more than one participant to cite it.
            let coord = self.wals.get(&tables[last].path).ok_or_else(missing)?.path().to_owned();
            // Append and stamp every prepare first, in enlistment order, so
            // each log's tick sequence is exactly what it was when this loop
            // also fsynced. Nothing is durable yet; a crash anywhere in here
            // leaves prepares citing a decision that was never written, which
            // is abort, which is what an unfinished COMMIT means.
            for e in tables.iter().take(last).filter(|e| !e.fold) {
                let Some(s) = e.seq else { continue };
                let w = self.wals.get_mut(&e.path).ok_or_else(missing)?;
                w.prepare(s, &coord, seq)?;
                w.stage_sync()?;
            }
            self.prepare_barrier(tables, last)?;
        }
        let w = self.wals.get_mut(&tables[last].path).ok_or_else(missing)?;
        if parties > 1 {
            w.decide(seq)?;
        } else {
            w.commit(seq)?;
        }
        w.sync()?;
        Ok(())
    }

    /// Make every staged prepare durable, concurrently, and return only once
    /// all of them have finished.
    ///
    /// The protocol wants the N-1 prepares durable **before the decision is
    /// written**; it does not want them durable in any order relative to each
    /// other. That is the whole opportunity, and on this platform it is worth
    /// taking: a barrier here is `F_FULLFSYNC`, a *device* cache flush of
    /// ~4 ms, and flushes issued concurrently on distinct files share one.
    /// Measured on the development machine, best-of-21, order alternated:
    /// three sequential barriers 11.78 ms against 3.84 ms concurrent, eight
    /// 31.78 against 12.08. So an N-table COMMIT stops costing N barriers of
    /// latency and starts costing two -- the prepares, then the decision --
    /// whatever N is.
    ///
    /// What it deliberately does **not** do is weaken any prepare. Every one
    /// still gets a real `F_FULLFSYNC` that has *returned* before the decision
    /// record is appended. The tempting version -- plain `fsync(2)` on the
    /// prepares, leaning on the decision's full barrier to flush the device
    /// for all of them -- is unsound: `F_FULLFSYNC` guarantees a post-condition
    /// of its return, not an ordering during its execution, so a power cut
    /// inside that last flush can persist the decision while a participant's
    /// bytes are still in the device's cache. Replay would then release the
    /// prepared groups and find one participant's records missing, which is
    /// the half-committed transaction two-phase commit exists to prevent.
    /// The concurrent form is at worst 2x the cost of the unsound one and
    /// introduces no new assumption at all.
    ///
    /// The join is unconditional, and that is load-bearing rather than tidy:
    /// [`Pool::map`](crate::common::pool::Pool::map) returns only once every
    /// index has run, so there is no path on which a failed barrier is
    /// discovered after the decision has been written. The first error aborts
    /// the COMMIT with every table still private, and the caller rolls back.
    ///
    /// Costs one `Vec` of N `&Wal` and one of N results per multi-table
    /// COMMIT. Single-table transactions -- every autocommit statement and the
    /// whole OLTP path -- never reach it.
    fn prepare_barrier(&self, tables: &[Enlisted], last: usize) -> Result<()> {
        let missing = || Error::storage("an enlisted table with a sequence number has no log");
        let staged: Vec<&crate::persist::Wal> = tables
            .iter()
            .take(last)
            .filter(|e| !e.fold && e.seq.is_some())
            .map(|e| self.wals.get(&e.path).ok_or_else(missing))
            .collect::<Result<_>>()?;
        // One participant has nothing to overlap with, and the pool would
        // charge a job setup to discover that.
        if let [w] = staged[..] {
            return w.barrier();
        }
        let n = staged.len();
        crate::common::pool::global().map(n, |i| staged[i].barrier()).into_iter().collect()
    }

    /// Discard the transaction, in memory and on disk.
    ///
    /// Dropping an overlay is a pointer store -- parts are immutable, so
    /// nothing has to be un-written -- and rewinding each log to the LSN the
    /// table was enlisted at leaves the file byte-identical to its
    /// pre-transaction state. Replay would have dropped those staged records
    /// anyway; the rewind is what makes "no trace" true of the disk too.
    pub fn rollback(&mut self) -> Result<()> {
        // A foreign ROLLBACK is the worse half of the pair: the connection
        // that runs it has been told `Ok` for an autocommitted write, and
        // discarding somebody else's overlay would take that write with it.
        self.check_owner("ROLLBACK")?;
        let Some(txn) = self.txn.take() else {
            return Err(Error::exec("ROLLBACK without an open transaction"));
        };
        self.discard(txn)
    }

    /// The body of `ROLLBACK`, over a transaction already detached from the
    /// session. Split out so [`Session::reap_txn`] can undo an orphan without
    /// going back through the owner check that would refuse it.
    fn discard(&mut self, txn: Txn) -> Result<()> {
        // Every table is rolled back even if one of the log rewinds fails:
        // leaving an overlay attached would keep uncommitted parts visible,
        // which is far worse than a log with dead bytes in it. The first
        // error is reported once the state is clean.
        let mut first_err = None;
        for e in &txn.tables {
            if let Ok(t) = self.catalog.table_by_path_mut(&e.path) {
                t.rollback_txn();
            }
            if e.seq.is_none() {
                continue;
            }
            if let Some(w) = self.wals.get_mut(&e.path) {
                if let Err(err) = w.rewind_to(e.lsn) {
                    first_err.get_or_insert(err);
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Bring `path` into the open transaction on its first write, and hand
    /// back the staging sequence number its log records must carry.
    ///
    /// `Ok(None)` means "not transactional": either no transaction is open, or
    /// one is but this session does not log.
    fn enlist(&mut self, path: &str) -> Result<Option<u64>> {
        let Some(txn) = &self.txn else { return Ok(None) };
        if let Some(e) = txn.tables.iter().find(|e| e.path == path) {
            return Ok(e.seq);
        }
        // The second half of the two-commit-points refusal; see
        // [`Session::mark_fold`], which is the same rule reached from the
        // other order. A scan of a `Vec` that is one entry long for every
        // single-table transaction, and the clone happens only on the way to
        // an error.
        if let Some(f) = txn.tables.iter().find(|e| e.fold).map(|e| e.path.clone()) {
            return Err(two_commit_points(&f, path));
        }
        // `begin_txn` flushes first, so the buffered rows that were already
        // there are published as committed data *before* the overlay exists --
        // which is what lets `rollback_txn` discard the delta wholesale.
        self.catalog.table_by_path_mut(path)?.begin_txn()?;
        let (seq, lsn) = match self.wal_for(path)? {
            Some(w) => (Some(w.begin()), w.lsn()),
            None => (None, 0),
        };
        self.txn
            .as_mut()
            .expect("checked above")
            .tables
            .push(Enlisted { path: path.to_string(), seq, lsn, fold: false });
        Ok(seq)
    }

    /// Note that `path` took a mutation the write-ahead log cannot describe,
    /// so COMMIT has to make it durable by writing the parts out instead.
    ///
    /// `enlist` has always run first, so the entry is the last one pushed --
    /// the reverse scan finds it on the first compare. With no transaction to
    /// defer to (a direct API caller rather than `atomic_stmt`) the statement
    /// is already committed, so the fold happens now instead: deferring to a
    /// COMMIT that will never arrive is how a durability hole is built.
    ///
    /// ## A folding table may not share a transaction, and this is where that
    /// is enforced
    ///
    /// A folding table's commit point is the `TABLE` rename inside
    /// [`Session::fold_to_parts`], which [`Session::commit`] runs **after**
    /// [`Session::commit_durable`] has already returned `Ok` -- and
    /// `commit_durable` excludes folding tables from the two-phase protocol
    /// outright, selecting its coordinator with `rposition(|e| !e.fold ...)`
    /// and counting `parties` over `!e.fold`. So a transaction holding one
    /// folding table and anything else has **two** commit points with a
    /// window between them, and `kill -9` in that window commits one half.
    ///
    /// Measured on this tree, before this refusal existed, 30-round scripts
    /// killed at a random instant: `BEGIN; INSERT INTO k; DELETE FROM u;
    /// COMMIT` over a keyed `k` and an unkeyed `u` left the insert durable and
    /// the delete undone in **16 of 25** trials; two unkeyed tables deleting
    /// under one `COMMIT` -- where `commit_durable` finds no participant at
    /// all, writes no marker and performs no fsync -- diverged in **6 of 25**.
    /// In a 130-trial randomized campaign over mixed workloads it was the only
    /// failure shape that appeared, 7 times, and nothing else failed at all.
    ///
    /// There is no marker-ordering fix. A rename cannot be made conditional on
    /// a decision written after it, and a decision cannot be made conditional
    /// on a rename that replay is unable to reproduce -- the rows the fold
    /// exists for are exactly the ones no log record can name. Writing a
    /// marker as well would release the group's `Insert` records with the
    /// sweep's tombstones still only in memory, which resurrects precisely the
    /// rows the statement deleted. So the transaction is refused instead, at
    /// the statement that would have created the second commit point rather
    /// than at `COMMIT`: the message names the statement the user just ran,
    /// and the transaction is still intact and still rollable.
    ///
    /// Both orders are covered -- this one for "a durable table is already
    /// enlisted and now a table wants to fold", [`Session::enlist`] for "a
    /// table is already folding and now another wants in".
    ///
    /// The refusal is narrow by construction: it fires only for a sweep that
    /// hid rows in a part with **no durable home**, since `apply_sweep` only
    /// reaches here when `MaskRuns::dark > 0`. A multi-table transaction whose
    /// unkeyed `DELETE` hits checkpointed rows logs `TAG_MASK_RUN`, never
    /// folds, and is unaffected.
    fn mark_fold(&mut self, path: &str) -> Result<()> {
        if let Some(txn) = self.txn.as_mut() {
            if let Some(other) = txn.tables.iter().find(|e| e.path != path).map(|e| e.path.clone())
            {
                return Err(two_commit_points(path, &other));
            }
            if let Some(e) = txn.tables.iter_mut().rev().find(|e| e.path == path) {
                e.fold = true;
                return Ok(());
            }
        }
        self.fold_to_parts(path)
    }

    /// Checkpoint **one** table: flush it, write its parts and commit record,
    /// then discard the log those parts now cover.
    ///
    /// This is the durability device for a positional sweep -- see
    /// [`Session::apply_sweep`]. It is deliberately not [`Session::checkpoint`],
    /// which walks every table in the catalog: the mutation touched one, and
    /// rewriting the rest would make an unrelated table's size a cost of this
    /// statement.
    ///
    /// The write ordering is `save_catalog`'s: flush the delta, **roll the
    /// log**, then commit the parts with the stream position the roll
    /// produced.
    ///
    /// The roll comes first because it destroys nothing -- the sealed segment
    /// stays exactly where it is -- so the "commit the parts before you
    /// discard the log" rule the old ordering existed for has nothing left to
    /// protect. What that buys is that the watermark and the fresh segment's
    /// origin are the same number by construction rather than by arithmetic:
    /// there is no `set_wal_committed` afterwards, and no stale-watermark
    /// repair on the next open. A crash between the two leaves the old
    /// watermark and no new parts, so replay replays everything from it --
    /// correct, and the one case that exercises multi-segment replay.
    fn fold_to_parts(&mut self, path: &str) -> Result<()> {
        let Some(root) = self.catalog.dir().map(Path::to_path_buf) else { return Ok(()) };
        let (db, _) = path.split_once('.').unwrap_or(("default", path));
        {
            let t = self.catalog.table_by_path_mut(path)?;
            // A `Memory` table is defined to vanish on restart, so there is
            // nothing to make durable and nowhere to write it.
            if !t.def.engine.is_persistent() {
                return Ok(());
            }
            t.flush()?;
        }
        // Through the cached handle rather than a fresh `Wal::open`, so the
        // session's idea of the stream stays correct; reopening behind the
        // cache is what forces `Session::checkpoint` to drop it.
        let committed = match self.wal_for(path)? {
            Some(w) => {
                w.roll()?;
                w.origin()
            }
            None => 0,
        };
        let t = self.catalog.table_by_path(path)?;
        crate::persist::write_table(&root.join(db), t, committed)
    }

    /// Run `f` as one atomic statement.
    ///
    /// Inside an explicit transaction this is a plain call: the enclosing
    /// transaction already supplies the atomicity, and committing here would
    /// end it early. Outside one it opens an implicit transaction, so a
    /// statement that publishes several parts publishes all of them or none.
    ///
    /// Used only where a statement really can publish more than once -- see
    /// `run_insert`. Wrapping the single-block case would put a `begin_txn`
    /// flush in front of every `INSERT ... VALUES`, forcing the buffered write
    /// path to build a part per statement, in exchange for atomicity a single
    /// publish already has.
    fn atomic_stmt<F>(&mut self, f: F) -> Result<usize>
    where
        F: FnOnce(&mut Session) -> Result<usize>,
    {
        if self.txn.is_some() {
            return f(self);
        }
        self.begin()?;
        match f(self) {
            Ok(n) => {
                self.commit()?;
                Ok(n)
            }
            Err(e) => {
                // The statement's error is the one worth reporting; a rollback
                // failure on top of it would only hide it.
                let _ = self.rollback();
                Err(e)
            }
        }
    }

    /// Run one or more statements, discarding results. For DDL and DML.
    pub fn execute(&mut self, sql: &str) -> Result<()> {
        self.run(sql).map(|_| ())
    }

    /// Run exactly one statement and return its result.
    pub fn query(&mut self, sql: &str) -> Result<ResultSet> {
        let mut rs = self.run(sql)?;
        match rs.len() {
            0 => Ok(ResultSet::empty()),
            1 => Ok(rs.pop().unwrap()),
            n => Err(Error::exec(format!("expected a single statement, got {n}"))),
        }
    }

    /// Run every statement in `sql`, returning one result per statement.
    pub fn run(&mut self, sql: &str) -> Result<Vec<ResultSet>> {
        // Transaction control is recognised here rather than in the grammar,
        // and the fast test is what keeps that free: `mentions_txn_keyword` is
        // one pass that answers "no" for every statement this engine has ever
        // run, and then the text reaches `parse` byte for byte as it did
        // before transactions existed. Only a hit pays for the second lex.
        if mentions_txn_keyword(sql) {
            return self.run_mixed(sql);
        }
        // `SET` / `SHOW SETTINGS` / `SETTINGS` are recognised here rather than
        // in the grammar, for the same mechanical reason transaction control is
        // just above: `Statement` is matched exhaustively across this file, so a
        // new variant would not compile in the modules that would have to
        // introduce it. `intercept` byte-sniffs first and answers `None` for
        // everything else, so ordinary SQL reaches `parse` unchanged.
        //
        // The clone ends the borrow of `self` before `intercept` takes
        // `&mut Session`; the state is behind an `Arc`, so it is two atomics
        // per *statement* against a statement about to be lexed, parsed, bound
        // and lowered.
        //
        // `INSERT ... FROM INFILE` used to be refused outright here whenever
        // the database had any CHECK or UNIQUE, because `io::emit` published
        // blocks straight at the catalog and passed none of the checks
        // `run_insert` applies. It routes through `Session::import_block` now,
        // so the gate -- and the byte scan that fed it -- is gone.
        if let Some(r) = self.settings.clone().intercept(self, sql) {
            return r;
        }
        // `BACKUP` / `RESTORE` / `VERIFY BACKUP` are recognised by the same
        // splitter transaction control uses, for the same reason and at the
        // same price: one byte scan that answers "no" for every statement this
        // engine had before, and a second lex only for the ones it does not.
        //
        // Strictly *after* the settings hook and not beside the txn one:
        // `INSERT INTO t FROM INFILE 'backup.csv'` mentions the word, and
        // `run_mixed` does not run the settings extensions, so claiming it
        // here first would turn a working import into a parse error.
        if mentions_admin_keyword(sql) {
            return self.run_mixed(sql);
        }
        // Poisoning is applied here rather than inside `exec_statement`,
        // because a parse error is a statement that failed too -- and one that
        // never reaches `exec_statement`.
        let stmts = match parse(sql) {
            Ok(s) => s,
            Err(e) => return Err(self.poison(e)),
        };
        let texts = statement_texts(sql, stmts.len());
        let mut out = Vec::with_capacity(stmts.len());
        for (i, s) in stmts.iter().enumerate() {
            let text = texts.as_ref().map_or(sql, |t| t[i]);
            match self.exec_statement(s, text) {
                Ok(rs) => out.push(rs),
                Err(e) => return Err(self.poison(e)),
            }
        }
        Ok(out)
    }

    /// `run` for input that may contain transaction control.
    ///
    /// Split on top-level semicolons using the *lexer* rather than a scan for
    /// `;`, because those are not the same thing: a semicolon inside a string
    /// literal or a comment is not a statement boundary, and the only way to
    /// agree with the parser about that is to ask the same tokenizer.
    fn run_mixed(&mut self, sql: &str) -> Result<Vec<ResultSet>> {
        use crate::sql::lexer::{tokenize, Token};
        let toks = tokenize(sql)?;
        let mut out = Vec::new();
        let mut start = 0usize;
        // One extra iteration for the tail, which has no closing semicolon.
        for i in 0..=toks.len() {
            let is_end = i == toks.len() || toks[i].tok == Token::Semicolon;
            if !is_end {
                continue;
            }
            let span = &toks[start..i];
            start = i + 1;
            if span.is_empty() {
                continue; // `;;`, or a trailing `;`
            }
            // The statement's own text, from its first token to the semicolon
            // that ended it (or the end of the input).
            let end = if i == toks.len() { sql.len() } else { toks[i].pos };
            let text = sql[span[0].pos..end].trim();
            if let Some(t) = txn_stmt(span) {
                let t0 = Instant::now();
                // ROLLBACK is the way *out* of a poisoned transaction, so it
                // is the one statement that must still run; BEGIN and COMMIT
                // poison or report on their own.
                let r = match t {
                    TxnStmt::Begin => self.begin(),
                    TxnStmt::Commit => self.commit(),
                    TxnStmt::Rollback => self.rollback(),
                }
                .map(|()| {
                    let mut rs = ResultSet::empty();
                    rs.stats.elapsed_us = t0.elapsed().as_micros();
                    rs
                });
                self.log_stmt(text, t.name(), t0, &r);
                let rs = r?;
                // A COMMIT is the statement boundary a transaction's logs have
                // been waiting for; see `Session::drain_folds`.
                if self.txn.is_none() && !self.fold_due.is_empty() {
                    self.drain_folds()?;
                }
                out.push(rs);
                continue;
            }
            if let Some(a) = admin_stmt(span) {
                let t0 = Instant::now();
                let r = a.and_then(|a| self.run_admin(&a));
                self.log_stmt(text, "ADMIN", t0, &r);
                match r {
                    Ok(mut rs) => {
                        rs.stats.elapsed_us = t0.elapsed().as_micros();
                        out.push(rs);
                    }
                    Err(e) => return Err(self.poison(e)),
                }
                continue;
            }
            let stmts = match parse(text) {
                Ok(s) => s,
                Err(e) => return Err(self.poison(e)),
            };
            let texts = statement_texts(text, stmts.len());
            for (k, s) in stmts.iter().enumerate() {
                match self.exec_statement(s, texts.as_ref().map_or(text, |t| t[k])) {
                    Ok(rs) => out.push(rs),
                    Err(e) => return Err(self.poison(e)),
                }
            }
        }
        Ok(out)
    }

    /// Run one statement and record it in the query log.
    ///
    /// The wrapper exists so that *every* return path of the dispatcher below
    /// -- including the early refusals -- lands in the log. A log that only
    /// holds the statements that got as far as the executor would be missing
    /// exactly the ones an operator is looking for.
    fn exec_statement(&mut self, stmt: &Statement, sql: &str) -> Result<ResultSet> {
        let t0 = Instant::now();
        // Once, and handed down: the dispatcher's two refusals and the log
        // entry all want the same `&'static str`.
        let kind = stmt_kind(stmt);
        let r = self.dispatch(stmt, kind, t0);
        self.log_stmt(sql, kind, t0, &r);
        // The statement is over, applied and logged, and no transaction is
        // holding an overlay: the first moment a log that crossed
        // `wal_fold_bytes` can safely be folded and truncated. Two loads and a
        // predicted branch when nothing is due, which is every statement
        // between two folds.
        if r.is_ok() && self.txn.is_none() && !self.fold_due.is_empty() {
            self.drain_folds()?;
        }
        r
    }

    fn dispatch(
        &mut self,
        stmt: &Statement,
        kind: &'static str,
        t0: Instant,
    ) -> Result<ResultSet> {
        // Nothing runs in a transaction that has already failed. Returning
        // `Ok` here is the shape of the bug: the statement's writes go into an
        // overlay the session is committed to discarding, and the client is
        // told they landed.
        self.check_txn(kind)?;
        // DDL persists itself immediately (see the checkpoint below), and a
        // checkpoint inside a transaction would write out uncommitted parts.
        // Refused rather than silently promoted to an implicit commit, which
        // is the other common answer and the one that loses a ROLLBACK the
        // user still believed in.
        if self.txn.is_some() && is_ddl(stmt) {
            return Err(Error::unsupported(
                "DDL is not allowed inside a transaction: it checkpoints, which \
                 would persist the transaction's uncommitted parts. COMMIT or \
                 ROLLBACK first",
            ));
        }
        // `USE` is not a read -- it moves the session's current database --
        // but it writes nothing, and a read-only session that could not
        // change database would be able to query only one of them.
        if self.read_only && !is_read(stmt) && !matches!(stmt, Statement::Use(_)) {
            return Err(read_only_err(kind));
        }
        // The read set goes through the `&self` path, and the only thing the
        // `&mut` half adds is the flush -- which is exactly the line that used
        // to make a read exclusive. Two entry points, one implementation.
        if is_read(stmt) {
            self.catalog.flush_all()?;
            let mut rs = self.read_statement(stmt, &self.limits.context())?;
            rs.stats.elapsed_us = t0.elapsed().as_micros();
            if rs.stats.rows == 0 {
                rs.stats.rows = rs.rows();
            }
            return Ok(rs);
        }
        let mut rs = match stmt {
            Statement::Insert(i) => self.run_insert(i)?,
            Statement::CreateTable(c) => self.run_create_table(c)?,
            Statement::CreateDatabase { name, if_not_exists } => {
                self.guard_dir_name(None, name)?;
                self.catalog.create_database(name, *if_not_exists)?;
                ResultSet::empty()
            }
            Statement::CreateView { name, query, body_sql, or_replace, if_not_exists } => {
                self.run_create_view(name, query, body_sql, *or_replace, *if_not_exists)?
            }
            Statement::DropView { name, if_exists } => self.run_drop_view(name, *if_exists)?,
            Statement::RenameTable { from, to } => self.run_rename_table(from, to)?,
            Statement::AlterModifyColumn { table, column, ty } => {
                self.run_modify_column(table, column, ty)?
            }
            Statement::DropTable { name, if_exists } => {
                self.guard_ddl_table(name)?;
                let path = self.catalog.qualify(name);
                let (db, tbl) = self.catalog.resolve(name);
                self.catalog.drop_table(name, *if_exists)?;
                // After the catalog agrees the table is gone, so `IF EXISTS`
                // on a name that was never there touches no log.
                self.retire_log(&db, &tbl)?;
                // The constraints go with the table, and they go *now*: a
                // table re-created under the same name has not declared them,
                // and a stale entry would enforce a constraint the new table
                // never had.
                let mut changed = self.ext.checks.remove(&path).is_some();
                changed |= self.ext.uniques.remove(&path).is_some();
                if changed {
                    let (db, _) = self.catalog.resolve(name);
                    self.persist_ext(&db)?;
                }
                ResultSet::empty()
            }
            Statement::DropDatabase { name, if_exists } => {
                // Every table in it is being dropped, so every log in it needs
                // sealing -- read the roster while the catalog still has one.
                let doomed: Vec<String> =
                    self.catalog.table_names_in(name).map(str::to_string).collect();
                self.catalog.drop_database(name, *if_exists)?;
                for t in &doomed {
                    self.retire_log(name, t)?;
                }
                // The metadata table went with the database; the copy in
                // memory has to go too, and there is nothing left to persist
                // it to.
                let prefix = format!("{name}.");
                self.ext.checks.retain(|k, _| !k.starts_with(&prefix));
                self.ext.uniques.retain(|k, _| !k.starts_with(&prefix));
                self.ext.views.retain(|k, _| !k.starts_with(&prefix));
                ResultSet::empty()
            }
            Statement::Use(db) => {
                self.catalog.use_database(db)?;
                ResultSet::empty()
            }
            Statement::Optimize { table, final_ } => {
                let t = self.catalog.table_mut(table)?;
                if *final_ {
                    t.compact()?;
                } else {
                    t.flush()?;
                }
                ResultSet::empty()
            }
            Statement::Truncate { table } => {
                self.guard_ddl_table(table)?;
                let (db, tbl) = self.catalog.resolve(table);
                // Re-qualified, and this is not tidiness. `Catalog::create_table`
                // strips the database off the name it is given and stores the
                // bare one, so handing the stored definition straight back
                // recreates the table in the *current* database:
                // `TRUNCATE TABLE m.u` left `default.u` behind and destroyed
                // `m.u` with its rows, in one statement, exit 0.
                let mut def = self.catalog.table(table)?.def.clone();
                def.name = format!("{db}.{tbl}");
                self.catalog.drop_table(table, false)?;
                self.catalog.create_table(def, false)?;
                // A truncated table is a new incarnation with an empty part
                // list, so its identity counter has to start above the one the
                // old incarnation was minting from -- see `PartSet::seed_pids`.
                if let Some(root) = self.catalog.dir().map(Path::to_path_buf) {
                    let end = crate::persist::wal::stream_end(&root, &db, &tbl);
                    self.catalog.table_mut(table)?.seed_pids(end);
                }
                // Deliberately *not* touching `ext`: TRUNCATE empties a table,
                // it does not redefine it, so the constraints it was created
                // with still apply to everything written after.
                ResultSet::empty()
            }
            Statement::SystemFlush(target) => {
                match target {
                    Some(t) => self.catalog.table_mut(t)?.flush()?,
                    None => self.catalog.flush_all()?,
                }
                ResultSet::empty()
            }
            Statement::AlterDelete { table, predicate } => {
                self.run_alter_delete(table, predicate)?
            }
            Statement::AlterUpdate { table, assignments, predicate } => {
                self.run_alter_update(table, assignments, predicate)?
            }
            Statement::AlterAddColumn { table, column, if_not_exists } => {
                self.run_add_column(table, column, *if_not_exists)?
            }
            Statement::AlterDropColumn { table, column, if_exists } => {
                self.run_drop_column(table, column, *if_exists)?
            }
            // Every read arm lives in `read_statement`, which the branch above
            // took. Reaching one here would mean `is_read` and the dispatcher
            // disagree, which is a bug rather than a statement.
            other => return Err(Error::exec(format!("unhandled statement: {other:?}"))),
        };
        // DDL is persisted immediately. A table that exists only in memory but
        // already has a write-ahead log on disk is an orphan: the log records
        // reference a table the catalog does not know about, so recovery
        // cannot apply them. DDL is rare enough that a catalog write per
        // statement is the right trade, and on a fresh CREATE TABLE there is
        // nothing to write out anyway.
        if is_ddl(stmt) && self.catalog.is_persistent() {
            self.checkpoint()?;
        }
        rs.stats.elapsed_us = t0.elapsed().as_micros();
        if rs.stats.rows == 0 {
            rs.stats.rows = rs.rows();
        }
        Ok(rs)
    }

    // ------------------------------------------------------------ the read set
    //
    // `&self`, all of it. What each one is allowed to touch is the whole
    // design: a read may pin an `Arc<PartSet>`, decode from it and read the
    // catalog's metadata, and it may not flush, publish, log or checkpoint.
    // That is the line that lets N of these run at once on one catalog.

    /// Run one read-only statement: SELECT, EXPLAIN, SHOW, DESCRIBE.
    ///
    /// Refuses anything else rather than falling through to a `&mut` path it
    /// does not have -- a reader that silently did nothing would be the same
    /// class of lie as a query that silently missed rows.
    pub fn read(&self, sql: &str) -> Result<ResultSet> {
        self.read_with(sql, &self.limits.context())
    }

    /// [`Session::read`] under a caller-supplied budget, deadline and cancel
    /// flag, for a pool that governs each connection separately.
    pub fn read_with(&self, sql: &str, ctx: &QueryContext) -> Result<ResultSet> {
        let t0 = Instant::now();
        // `SELECT` is the kind a statement that never parsed is filed under:
        // this path refuses everything else anyway, so it is the truth for
        // every entry that can reach the log from here.
        let mut kind = "SELECT";
        let r = self.one_read_stmt(sql).and_then(|stmt| {
            kind = stmt_kind(&stmt);
            let mut rs = self.read_statement(&stmt, ctx)?;
            // The `&mut` dispatcher has always stamped this; the `&self` half
            // never did, so a `Reader::query` reported an elapsed time of zero
            // and `system.query_log` would have inherited the same blank.
            rs.stats.elapsed_us = t0.elapsed().as_micros();
            Ok(rs)
        });
        self.log_stmt(sql, kind, t0, &r);
        r
    }

    /// Stream a `SELECT`'s blocks to `sink` without ever holding more than one.
    ///
    /// The `Vec<Block>` a `ResultSet` carries is the last unbounded buffer in
    /// the engine; this is the way past it, and the thing a `COPY TO`, a wire
    /// protocol's portal, or any client that consumes as it goes should call.
    /// Returns the row count the sink saw.
    pub fn read_stream(
        &self,
        sql: &str,
        ctx: &QueryContext,
        sink: &mut dyn FnMut(StreamItem<'_>) -> Result<()>,
    ) -> Result<usize> {
        let stmt = self.one_read_stmt(sql)?;
        let Statement::Query(q) = &stmt else {
            // SHOW/DESCRIBE/EXPLAIN produce one small block by construction;
            // routing them through here would buy nothing and would give the
            // caller a second shape to handle.
            return Err(Error::unsupported(
                "streaming is for SELECT; use `read` for SHOW, DESCRIBE and EXPLAIN",
            ));
        };
        let mut rows = 0;
        self.stream_in(q, ctx, &mut |item| {
            if let StreamItem::Rows(b) = &item {
                rows += b.rows();
            }
            sink(item)
        })?;
        Ok(rows)
    }

    /// Parse `sql` down to exactly one statement and refuse it unless it reads.
    ///
    /// The gate is here, on the parsed statement, rather than on the text: a
    /// scan for "SELECT" cannot tell `SELECT` from `/* SELECT */ DROP TABLE`.
    fn one_read_stmt(&self, sql: &str) -> Result<Statement> {
        let mut stmts = parse(sql)?;
        match stmts.len() {
            1 => {}
            0 => return Err(Error::exec("no statement to run")),
            n => return Err(Error::exec(format!("expected a single statement, got {n}"))),
        }
        let stmt = stmts.pop().expect("length checked");
        if !is_read(&stmt) {
            return Err(Error::unsupported(format!(
                "{} is a write and this is a read-only path: run it through a \
                 `&mut Session` (or `Db::writer`)",
                stmt_kind(&stmt)
            )));
        }
        // The two gates below are about *rows*, so they are asked only of the
        // statements that read rows. `SHOW TABLES` and `DESCRIBE` answer out
        // of the catalog, which neither a buffered write nor an open
        // transaction can change -- DDL inside a transaction is refused --
        // and failing them because an unrelated table has an unflushed delta
        // would be a refusal with no cause.
        if !reads_rows(&stmt) {
            return Ok(stmt);
        }
        // A reader must not see an open transaction's private overlay:
        // `Table::snapshot` hands the overlay to whoever asks once `begin_txn`
        // has run, so a read here would be a dirty read of uncommitted rows.
        // `Db::transaction` holds the writer lock across the whole block,
        // which is why this is reachable only by driving BEGIN through a raw
        // `Db::writer` guard and then dropping it.
        if self.catalog.any_in_txn() {
            return Err(Error::unsupported(
                "a transaction is open on this database: a concurrent read would see \
                 its uncommitted rows. COMMIT or ROLLBACK first, or hold the writer \
                 for the whole transaction with `Db::transaction`",
            ));
        }
        // Scans read parts. Buffered rows are invisible to them, and a reader
        // cannot flush -- so it says so instead of answering short. The
        // `Reader` on the other side of `Db` takes the writer lock for one
        // flush and retries; see `Reader::with_session`.
        if self.catalog.has_pending_writes() {
            return Err(Error::exec(PENDING_WRITES));
        }
        Ok(stmt)
    }

    /// The read set's dispatcher. `&self` end to end.
    fn read_statement(&self, stmt: &Statement, ctx: &QueryContext) -> Result<ResultSet> {
        match stmt {
            Statement::Query(q) => self.query_in(q, ctx),
            Statement::ShowDatabases => {
                ResultSet::one_string_column("name", self.catalog.database_names())
            }
            Statement::ShowTables { database } => {
                let db = database.as_deref().unwrap_or(self.catalog.current_database());
                let mut names = self.catalog.table_names(Some(db))?;
                // Views are listed beside tables because that is the namespace
                // they share: a `SELECT` names either the same way, and a
                // `SHOW TABLES` that omitted them would be a list you cannot
                // trust to tell you what a name will resolve to.
                let prefix = format!("{db}.");
                names.extend(
                    self.ext
                        .views
                        .keys()
                        .filter_map(|k| k.strip_prefix(&prefix).map(str::to_string)),
                );
                names.sort();
                ResultSet::one_string_column("name", names)
            }
            Statement::ShowCreateTable(name) => {
                let statement = match self.view_of(name) {
                    Some((key, v)) => {
                        let (db, bare) = key.split_once('.').unwrap_or(("", &key));
                        format!("CREATE VIEW `{db}`.`{bare}` AS\n{}", v.sql)
                    }
                    None => {
                        let t = self.catalog.table(name)?;
                        let path = self.catalog.qualify(name);
                        render_create_table(
                            t.schema(),
                            &t.def,
                            self.ext.checks.get(&path).map_or(&[][..], Vec::as_slice),
                            self.ext.uniques.get(&path).map(String::as_str),
                        )
                    }
                };
                ResultSet::one_string_column("statement", vec![statement])
            }
            Statement::Describe(name) => {
                // A view's columns are its query's, and the cheapest exact way
                // to say what they are is to bind it -- which is also the only
                // way that cannot drift from what a SELECT would return.
                if let Some((_, v)) = self.view_of(name) {
                    let plan = self.plan_in(&v.query, ctx)?;
                    let schema = Schema::new(vec![
                        Field::new("name", DataType::String),
                        Field::new("type", DataType::String),
                    ])?;
                    let rows = plan
                        .schema()
                        .fields()
                        .iter()
                        .map(|f| vec![Value::str(f.name.clone()), Value::str(f.ty.to_string())])
                        .collect();
                    return ResultSet::from_rows(schema, rows);
                }
                let t = self.catalog.table(name)?;
                let schema = Schema::new(vec![
                    Field::new("name", DataType::String),
                    Field::new("type", DataType::String),
                ])?;
                let rows = t
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| vec![Value::str(f.name.clone()), Value::str(f.ty.to_string())])
                    .collect();
                ResultSet::from_rows(schema, rows)
            }
            Statement::Explain { kind, statement } => self.run_explain(*kind, statement, ctx),
            other => Err(Error::unsupported(format!(
                "{} is not a read",
                stmt_kind(other)
            ))),
        }
    }

    // --------------------------------------------------------------- queries

    /// Plan a query, flushing first. The `&mut self` half of the split.
    ///
    /// Scans read parts, not the write buffer, so everything buffered has to
    /// land in a part first. See the storage::table module docs for why this
    /// beats teaching every operator to merge a hash map.
    ///
    /// The flush is *here* rather than inside [`Session::plan_in`], and that
    /// one line is the whole reason reads used to serialize: it made a
    /// `SELECT` on `a` take exclusive write access to `b`, `c` and `d`, and
    /// may synchronously rewrite them.
    fn plan(&mut self, q: &crate::sql::ast::Query) -> Result<LogicalPlan> {
        self.catalog.flush_all()?;
        self.plan_in(q, &self.limits.context())
    }

    /// Plan a query against the catalog exactly as it stands.
    ///
    /// The read path's planner: `&self`, no flush, no mutation of anything.
    /// The caller is responsible for the buffered rows -- either it flushed
    /// (the `&mut` path) or it checked [`Catalog::has_pending_writes`] and
    /// refused (the `&self` path).
    fn plan_in(&self, q: &crate::sql::ast::Query, ctx: &QueryContext) -> Result<LogicalPlan> {
        let q = self.resolve_subqueries(q, ctx)?;
        let plan = Binder::new(&self.catalog).bind_query(&q)?;
        // Costed, not plain: the catalog is what makes a join order choosable,
        // because cardinality is a property of the data and not of the plan.
        // The mutation call sites below stay on `optimize` deliberately -- they
        // are single-table by construction and have no order to choose, so they
        // would pay the estimator for nothing.
        optimizer::optimize_costed(plan, &self.catalog)
    }

    /// Evaluate uncorrelated subqueries and fold them into literals.
    ///
    /// The binder resolves names against a schema and has no executor, so it
    /// cannot evaluate `x IN (SELECT ...)`. But an *uncorrelated* subquery does
    /// not depend on the outer row at all, so it can simply be run first and
    /// replaced by its result — which is also strictly better than a
    /// per-row-correlated evaluation would be, since it runs once.
    ///
    ///   * `(SELECT ...)` in scalar position -> the single value it produced
    ///   * `x IN (SELECT ...)`               -> `x IN (v1, v2, ...)`
    ///   * `EXISTS (SELECT ...)`             -> `true` / `false`
    ///
    /// A *correlated* subquery references an outer column, so running it alone
    /// fails to bind; that failure is caught and reported as the unsupported
    /// feature it is, rather than as a confusing unknown-column error.
    /// ## Nothing is cloned unless something is folded
    ///
    /// The rewrite is in-place, so it needs an owned AST -- and this used to
    /// clone the whole query on *every* statement, subqueries or not, which is
    /// 330 ns and a dozen allocations for `SELECT id FROM t WHERE id = ?` and
    /// 1.07 us for an ordinary analytic query (measured, best of 5 over 200k
    /// clones). [`has_subquery`] answers the same question by walking the
    /// borrowed AST with no allocation at all, and the overwhelming majority
    /// of statements answer "no".
    fn resolve_subqueries<'q>(
        &self,
        q: &'q crate::sql::ast::Query,
        ctx: &QueryContext,
    ) -> Result<std::borrow::Cow<'q, crate::sql::ast::Query>> {
        // `system.*` rides the same walk and the same clone: both rewrites
        // replace a node in place, both need an owned AST to do it, and a
        // query that needs neither -- which is almost every query -- pays one
        // borrowed traversal and nothing else. Folding the system-table test
        // into `has_subquery` rather than adding a second walk is what keeps
        // "must not cost anything when unused" true.
        if !has_subquery(q, self.names()) {
            return Ok(std::borrow::Cow::Borrowed(q));
        }
        let mut out = q.clone();
        let mut st = Sub { left: 64, ctx };
        self.rewrite_query(&mut out, &mut st)?;
        Ok(std::borrow::Cow::Owned(out))
    }

    fn rewrite_query(&self, q: &mut crate::sql::ast::Query, budget: &mut Sub<'_>) -> Result<()> {
        use crate::sql::ast::SetExpr;
        for cte in q.with.iter_mut() {
            self.rewrite_query(&mut cte.query, budget)?;
        }
        self.rewrite_setexpr(&mut q.body, budget)?;
        for o in q.order_by.iter_mut() {
            self.rewrite_expr(&mut o.expr, budget)?;
        }
        for e in q.limit.iter_mut().chain(q.offset.iter_mut()) {
            self.rewrite_expr(e, budget)?;
        }
        if let Some((n, keys)) = q.limit_by.as_mut() {
            self.rewrite_expr(n, budget)?;
            for k in keys.iter_mut() {
                self.rewrite_expr(k, budget)?;
            }
        }
        let _ = SetExpr::Values(Vec::new()); // keep the import honest
        Ok(())
    }

    fn rewrite_setexpr(&self, s: &mut crate::sql::ast::SetExpr, budget: &mut Sub<'_>) -> Result<()> {
        use crate::sql::ast::{SelectItem, SetExpr};
        match s {
            SetExpr::Select(sel) => {
                for item in sel.projection.iter_mut() {
                    if let SelectItem::Expr { expr, .. } = item {
                        self.rewrite_expr(expr, budget)?;
                    }
                }
                if let Some(f) = sel.from.as_mut() {
                    self.rewrite_tableref(f, budget)?;
                }
                for e in sel
                    .prewhere
                    .iter_mut()
                    .chain(sel.selection.iter_mut())
                    .chain(sel.having.iter_mut())
                {
                    self.rewrite_expr(e, budget)?;
                }
                for g in sel.group_by.iter_mut() {
                    self.rewrite_expr(g, budget)?;
                }
            }
            SetExpr::Query(q) => self.rewrite_query(q, budget)?,
            SetExpr::SetOperation { left, right, .. } => {
                self.rewrite_setexpr(left, budget)?;
                self.rewrite_setexpr(right, budget)?;
            }
            SetExpr::Values(rows) => {
                for r in rows.iter_mut() {
                    for e in r.iter_mut() {
                        self.rewrite_expr(e, budget)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn rewrite_tableref(&self, t: &mut crate::sql::ast::TableRef, budget: &mut Sub<'_>) -> Result<()> {
        use crate::sql::ast::{JoinConstraint, TableRef};
        match t {
            // A `system.` or `information_schema.` reference becomes the
            // derived table it is equivalent to, computed now, from this
            // session's catalog. Everything downstream -- binder, optimizer,
            // executor -- then treats it as an ordinary relation, which is
            // what makes it joinable and filterable without a plan node that
            // could drift from the real one.
            TableRef::Table { name, alias, .. } => {
                if let Some(kind) = crate::system::classify(name, &self.catalog) {
                    let q = crate::system::derived(
                        kind,
                        &self.catalog,
                        &self.settings,
                        &self.log,
                    )?;
                    // Unaliased, the table is still addressable by its bare
                    // name -- `system.parts p` and `system.parts` must both
                    // qualify as `parts`.
                    let alias = alias.clone().or_else(|| Some(name.last().to_string()));
                    *t = TableRef::Subquery { query: Box::new(q), alias };
                    return Ok(());
                }
                // A view becomes the derived table it is defined as, exactly
                // as a `system.` reference does -- one rewrite, and everything
                // downstream sees an ordinary subquery. Inlining rather than
                // materializing is what makes a view cost nothing to keep and
                // always agree with its sources.
                //
                // The stored body is already fully qualified, so it means the
                // same thing spliced into a statement from any database.
                let Some(v) = self.ext.view(name, self.catalog.current_database()) else {
                    return Ok(());
                };
                // Views can nest, and after a DROP + CREATE they can nest in a
                // cycle (`a` over `b`, then `b` re-created over `a`). The
                // subquery budget bounds both: it is the one counter that is
                // already decremented per level of folding, and running out of
                // it says so rather than overflowing the stack.
                if budget.left == 0 {
                    return Err(Error::unsupported(format!(
                        "view `{}` nests too deeply -- if two views reference each other, \
                         one of them has to go",
                        name.last()
                    )));
                }
                budget.left -= 1;
                let mut q = v.query.clone();
                let alias = alias.clone().or_else(|| Some(name.last().to_string()));
                self.rewrite_query(&mut q, budget)?;
                *t = TableRef::Subquery { query: Box::new(q), alias };
            }
            TableRef::Subquery { query, .. } => self.rewrite_query(query, budget)?,
            TableRef::Join { left, right, constraint, .. } => {
                self.rewrite_tableref(left, budget)?;
                self.rewrite_tableref(right, budget)?;
                if let JoinConstraint::On(e) = constraint {
                    self.rewrite_expr(e, budget)?;
                }
            }
        }
        Ok(())
    }

    fn rewrite_expr(&self, e: &mut crate::sql::ast::Expr, budget: &mut Sub<'_>) -> Result<()> {
        use crate::sql::ast::Expr;
        // Depth-first, so a subquery nested inside another is resolved first.
        match e {
            Expr::UnaryOp { expr, .. } | Expr::Cast { expr, .. } | Expr::IsNull { expr, .. } => {
                self.rewrite_expr(expr, budget)?
            }
            Expr::BinaryOp { left, right, .. } => {
                self.rewrite_expr(left, budget)?;
                self.rewrite_expr(right, budget)?;
            }
            Expr::Function { args, params, .. } => {
                for a in args.iter_mut().chain(params.iter_mut()) {
                    self.rewrite_expr(a, budget)?;
                }
            }
            // A window call's OVER clause is ordinary expression territory, so
            // a subquery inside `PARTITION BY (SELECT ...)` has to be folded
            // here like any other -- the binder sees only literals by then.
            Expr::Window { args, params, spec, .. } => {
                for a in args.iter_mut().chain(params.iter_mut()) {
                    self.rewrite_expr(a, budget)?;
                }
                for p in spec.partition_by.iter_mut() {
                    self.rewrite_expr(p, budget)?;
                }
                for o in spec.order_by.iter_mut() {
                    self.rewrite_expr(&mut o.expr, budget)?;
                }
            }
            Expr::Case { operand, when_then, else_result } => {
                if let Some(o) = operand {
                    self.rewrite_expr(o, budget)?;
                }
                for (w, t) in when_then.iter_mut() {
                    self.rewrite_expr(w, budget)?;
                    self.rewrite_expr(t, budget)?;
                }
                if let Some(x) = else_result {
                    self.rewrite_expr(x, budget)?;
                }
            }
            Expr::InList { expr, list, .. } => {
                self.rewrite_expr(expr, budget)?;
                for i in list.iter_mut() {
                    self.rewrite_expr(i, budget)?;
                }
            }
            Expr::Between { expr, low, high, .. } => {
                self.rewrite_expr(expr, budget)?;
                self.rewrite_expr(low, budget)?;
                self.rewrite_expr(high, budget)?;
            }
            Expr::Like { expr, pattern, .. } => {
                self.rewrite_expr(expr, budget)?;
                self.rewrite_expr(pattern, budget)?;
            }
            Expr::Tuple(items) => {
                for i in items.iter_mut() {
                    self.rewrite_expr(i, budget)?;
                }
            }
            Expr::Interval { value, .. } => self.rewrite_expr(value, budget)?,
            Expr::Literal(_) | Expr::Column(_) | Expr::Wildcard => {}
            // --- the three that actually get folded ---
            //
            // `None` means the subquery is *correlated*: it names a column of
            // the query enclosing it and has no value of its own to fold to.
            // The node is left exactly where it is and the binder decorrelates
            // it into a join. Folding stays what it always was -- the
            // uncorrelated fast path -- and keeps its measured plans.
            Expr::Subquery(q) => {
                if let Some(vals) = self.eval_subquery(q, budget, "scalar subquery", 1)? {
                    *e = Expr::Literal(vals.into_iter().next().unwrap_or(Value::Null));
                }
            }
            Expr::InSubquery { expr, subquery, negated } => {
                self.rewrite_expr(expr, budget)?;
                if let Some(vals) =
                    self.eval_subquery(subquery, budget, "IN (SELECT ...)", usize::MAX)?
                {
                    *e = Expr::InList {
                        expr: expr.clone(),
                        list: vals.into_iter().map(Expr::Literal).collect(),
                        negated: *negated,
                    };
                }
            }
            Expr::Exists { subquery, negated } => {
                if let Some(vals) = self.eval_subquery(subquery, budget, "EXISTS", usize::MAX)? {
                    *e = Expr::Literal(Value::Bool(!vals.is_empty() != *negated));
                }
            }
        }
        Ok(())
    }

    /// Run a subquery and return column 0. `max_rows` caps a scalar subquery
    /// at one row, per SQL semantics.
    fn eval_subquery(
        &self,
        q: &crate::sql::ast::Query,
        budget: &mut Sub<'_>,
        what: &str,
        max_rows: usize,
    ) -> Result<Option<Vec<Value>>> {
        if budget.left == 0 {
            return Err(Error::unsupported(format!(
                "{what}: subquery nesting is too deep"
            )));
        }
        budget.left -= 1;

        // `plan_in`, not `plan`: the outer statement flushed already if it was
        // going to, and a subquery must not be the thing that decides a read
        // needs `&mut`.
        // A correlated subquery cannot bind on its own: it names a column of
        // the query around it, which is not in scope out here. That is no
        // longer an error -- the binder decorrelates it into a join -- so
        // folding declines and leaves the node alone. A *genuine* bind error
        // inside the subquery is not lost either: the binder meets it a moment
        // later, in the scope where it can say what is actually wrong.
        let plan = match self.plan_in(q, budget.ctx) {
            Ok(p) => p,
            Err(Error::Bind(_)) | Err(Error::Storage(_)) => return Ok(None),
            Err(other) => return Err(other),
        };
        if plan.schema().len() != 1 {
            return Err(Error::bind(format!(
                "{what} must select exactly one column, got {}",
                plan.schema().len()
            )));
        }
        // Streamed rather than collected: an `IN (SELECT ...)` over a million
        // rows used to hold the blocks *and* the values at once, and the
        // blocks were thrown away one line later. This also puts the budget
        // and the deadline around a subquery, which `operators::execute`'s
        // ambient context could not.
        let mut out = Vec::new();
        let mut op = operators::build(&plan, &self.catalog, budget.ctx)?;
        while let Some(b) = {
            budget.ctx.check()?;
            op.next()?
        } {
            out.reserve(b.rows());
            let col = b.column(0);
            for r in 0..b.rows() {
                out.push(col.value(r));
            }
            // Bounded here rather than after the loop: a scalar subquery over
            // a huge table stopped only once every value was materialized.
            if max_rows == 1 && out.len() > 1 {
                break;
            }
        }
        if max_rows == 1 && out.len() > 1 {
            // No row count in the message any more, because the loop above
            // stops at the second row rather than draining the table to find
            // out how wrong it was.
            return Err(Error::exec(format!(
                "{what} returned more than one row, expected at most 1"
            )));
        }
        Ok(Some(out))
    }

    /// The read path, end to end, from `&self`: plan, run, collect.
    ///
    /// One function behind every reader -- [`Session::read`], [`Reader::query`]
    /// and the `&mut` [`Session::query`] all land here -- so there is no second
    /// pipeline to drift, which is the defect this phase exists to stop.
    fn query_in(&self, q: &crate::sql::ast::Query, ctx: &QueryContext) -> Result<ResultSet> {
        let mut blocks = Vec::new();
        let mut schema = Schema::empty();
        // The result set is the one buffer in the engine that grew without a
        // ceiling: every operator above `BLOCK_SIZE` charges its footprint to
        // the tracker, and then the rows it produced were accumulated here
        // uncharged. `SELECT * FROM t` on a table larger than RAM was an OOM
        // rather than an error. The guard is released when this returns, which
        // is right: from there on the memory is the caller's, and the budget
        // describes one query's peak, not the lifetime of what it handed back.
        let mut held = MemGuard::new(ctx, "the result set");
        let mut bytes = 0usize;
        let stats = self.stream_in(q, ctx, &mut |item| {
            match item {
                StreamItem::Head(s) => schema = s.clone(),
                StreamItem::Rows(b) => {
                    bytes += retained_bytes(&b);
                    held.grow_to(bytes)?;
                    blocks.push(b);
                }
            }
            Ok(())
        })?;
        Ok(ResultSet {
            schema,
            stats: QueryStats {
                rows: blocks.iter().map(|b| b.rows()).sum(),
                elapsed_us: 0,
                granules_read: stats.granules_read,
                granules_pruned: stats.granules_pruned,
                rows_scanned: stats.rows_read,
            },
            blocks,
            affected: None,
        })
    }

    /// Run `q` and hand each block to `sink` as it is produced.
    ///
    /// The streaming primitive: nothing above one block is held by this
    /// function, so exporting a billion rows costs a block of memory rather
    /// than a billion rows of it. A blocking operator (a sort, a hash
    /// aggregate) still builds its own state -- that is the operator's
    /// footprint, charged to `ctx` where it is built -- but the *result* no
    /// longer has to exist all at once, which is what a `COPY TO`, a portal
    /// fetch, or any client that reads faster than it can buffer needs.
    ///
    /// The schema is delivered *before* the first block and unconditionally,
    /// including for an empty result: a client that has to describe the rows
    /// before it sends them (every wire protocol) cannot wait to find out.
    fn stream_in(
        &self,
        q: &crate::sql::ast::Query,
        ctx: &QueryContext,
        sink: &mut dyn FnMut(StreamItem<'_>) -> Result<()>,
    ) -> Result<crate::exec::operators::ScanStats> {
        let plan = self.plan_in(q, ctx)?;
        sink(StreamItem::Head(plan.schema()))?;
        // Through the exchange, which decides per query whether to go parallel
        // (see `exchange::degree`) and falls back to the serial pipeline below
        // its row threshold. The decision is in the operator, not here.
        let mut op = crate::exec::exchange::build(
            crate::planner::physical::lower(&plan, &self.catalog)?,
            &self.catalog,
            ctx,
        )?;
        while let Some(b) = {
            ctx.check()?;
            op.next()?
        } {
            if b.rows() > 0 {
                sink(StreamItem::Rows(b))?;
            }
        }
        Ok(op.stats())
    }

    fn run_explain(
        &self,
        kind: ExplainKind,
        stmt: &Statement,
        ctx: &QueryContext,
    ) -> Result<ResultSet> {
        let text = match (kind, stmt) {
            (ExplainKind::Ast, s) => format!("{s:#?}"),
            // PIPELINE renders the *physical* plan, which is the only place the
            // access path is visible: whether a predicate on the key lowered to
            // an index probe or stayed a scan is a physical decision, and PLAN
            // shows the logical tree where that choice does not exist yet.
            // Without this, index selection is unprovable from the outside.
            (ExplainKind::Pipeline, Statement::Query(q)) => {
                let logical = self.plan_in(q, ctx)?;
                crate::planner::physical::lower(&logical, &self.catalog)?.explain()
            }
            // The one EXPLAIN that runs the statement, and the only way to see
            // the counters `run_query` computes and drops on the floor. Same
            // plan, same builder, same loop as `Session::query` -- a diagnostic
            // that measured a differently built pipeline would be measuring
            // something nobody runs.
            (ExplainKind::Analyze, Statement::Query(q)) => {
                // The session's own context, so an ANALYZE honours the same
                // budget, deadline and cancel flag the query it measures would
                // have got. It used to mint a default one, which made EXPLAIN
                // ANALYZE the one way to run a query the session could not stop.
                let logical = self.plan_in(q, ctx)?;
                crate::exec::exchange::explain_analyze(&logical, &self.catalog, ctx)?
            }
            (_, Statement::Query(q)) => self.plan_in(q, ctx)?.explain(),
            // A mutation is not a relation, so it is not a `LogicalPlan` and
            // does not reach the arms above -- but the plan that finds its rows
            // is an ordinary one, and this is the only way to see from outside
            // that it got the same treatment a SELECT's would. `prewhere=` in
            // the scan line is the pushdown, `zonemap=` the granule pruning.
            //
            // No `PIPELINE` arm on purpose: `physical::lower` decides between a
            // sequential scan and an `IndexLookup`, and `Table::delete_where`
            // takes neither -- it sweeps positions, because a mutation needs
            // the `(part, row)` of a match and an operator pipeline yields
            // values. Rendering the physical plan here would advertise an
            // access path the statement does not run.
            (_, Statement::AlterDelete { table, predicate }) => {
                let mut m = Binder::new(&self.catalog).bind_delete(table, predicate)?;
                m.source = optimizer::optimize(m.source)?;
                m.explain()
            }
            (_, Statement::AlterUpdate { table, assignments, predicate }) => {
                let mut m =
                    Binder::new(&self.catalog).bind_update(table, assignments, predicate)?;
                m.source = optimizer::optimize(m.source)?;
                m.explain()
            }
            (_, other) => format!("{other:#?}"),
        };
        ResultSet::one_string_column("explain", text.lines().map(|l| l.to_string()).collect())
    }

    // ------------------------------------------------------------------ DML

    fn run_insert(&mut self, ins: &Insert) -> Result<ResultSet> {
        self.guard_ddl_table(&ins.table)?;
        // Named before the catalog is asked, or the error would be "table does
        // not exist" about a name that plainly does exist.
        if let Some((key, _)) = self.view_of(&ins.table) {
            return Err(Error::unsupported(format!(
                "cannot INSERT into `{key}`: it is a view, which stores a query rather \
                 than rows. Insert into the table the view selects from"
            )));
        }
        let path = self.catalog.qualify(&ins.table);
        let target = self.catalog.table_by_path(&path)?.def.schema.clone();

        let order: Vec<usize> = if ins.columns.is_empty() {
            (0..target.len()).collect()
        } else {
            ins.columns
                .iter()
                .map(|c| target.require(c))
                .collect::<Result<_>>()?
        };

        let source_blocks = match &ins.source {
            InsertSource::Values(rows) => vec![self.values_to_block(rows, &target, &order)?],
            InsertSource::Query(q) => {
                let plan = self.plan(q)?;
                if plan.schema().len() != order.len() {
                    return Err(Error::bind(format!(
                        "INSERT expects {} columns, the SELECT produces {}",
                        order.len(),
                        plan.schema().len()
                    )));
                }
                // The session's real budget, deadline and cancel flag, not
                // the process-static tracker this used to reach. The identical
                // aggregate was refused as a `SELECT` and accepted here.
                operators::execute_ctx(&plan, &self.catalog, &self.limits.context())?.0
            }
        };

        // More than one block means more than one publish, and until this
        // wrapper existed the second one could fail with the first already
        // irreversibly in the table -- parts published, rows tombstoned, and
        // an error returned over the top. One block cannot do that (a single
        // `Table::insert` is already failure-atomic), and wrapping it would
        // cost a flush per `INSERT ... VALUES`, so the gate is exact.
        //
        // The multi-block case reaches here only from `INSERT ... SELECT`,
        // whose planning has already flushed every table -- so the `begin_txn`
        // inside `enlist` finds an empty delta and costs nothing.
        if source_blocks.len() > 1 {
            let (t, o) = (target, order);
            return self
                .atomic_stmt(move |s| s.insert_blocks(&path, source_blocks, &t, &o))
                .map(ResultSet::with_affected);
        }
        let n = self.insert_blocks(&path, source_blocks, &target, &order)?;
        Ok(ResultSet::with_affected(n))
    }

    /// Widen, log and apply each source block. The body of an `INSERT`, split
    /// out so [`Session::atomic_stmt`] can wrap it.
    fn insert_blocks(
        &mut self,
        path: &str,
        blocks: Vec<Block>,
        target: &Schema,
        order: &[usize],
    ) -> Result<usize> {
        // `UNIQUE` on the key column is the difference between an upsert and a
        // refusal, and it is decided once per statement: one map probe, on a
        // map that is empty unless some table in this database declared one.
        let unique = self.ext.uniques.contains_key(path);
        let mut n = 0;
        for b in blocks {
            if b.rows() == 0 {
                continue;
            }
            let full = self.widen_to_schema(b, target, order)?;
            // Checked before anything is logged: a refused row must leave no
            // record behind, and a record in the log is a row a crash would
            // replay. A CHECK can say no from the block alone, so it does.
            self.enforce_checks(path, &full)?;
            if unique {
                n += self.insert_unique(path, full)?;
                continue;
            }
            // Log-before-apply: the record is durable before the write is
            // acknowledged, so a crash between the two replays the insert
            // rather than losing it. Inside a transaction it is staged
            // instead, and the commit marker is what makes it replayable.
            self.log_insert(path, &full)?;
            n += self.catalog.table_by_path_mut(path)?.insert(full)?;
        }
        Ok(n)
    }

    /// One block from a bulk importer, checked the way an `INSERT` is.
    ///
    /// What `io::emit` calls instead of reaching for
    /// `catalog.table_by_path_mut(..).insert(..)`, which was the one write path
    /// in the engine that bypassed constraints -- and which is why an import
    /// into a database with any constraint at all used to be refused outright.
    pub(crate) fn import_block(&mut self, path: &str, b: Block) -> Result<usize> {
        self.enforce_checks(path, &b)?;
        if self.ext.uniques.contains_key(path) {
            return self.insert_unique(path, b);
        }
        self.catalog.table_by_path_mut(path)?.insert(b)
    }

    /// Insert into a table whose key is declared `UNIQUE`: refuse a repeated
    /// key instead of replacing the row that already has it.
    ///
    /// ## Why this cannot be log-before-apply
    ///
    /// A refusal is the *expected* outcome here, not an I/O accident -- it is
    /// what the constraint is for. Logging first and then refusing leaves an
    /// `Insert` record in the log for a statement the caller was told failed,
    /// and replay applies records with last-write-wins, so a crash after a
    /// refused insert would land exactly the row the constraint rejected, on
    /// top of the row it was protecting. Silent, and only after a crash.
    ///
    /// So the record is **staged** instead, under a group of its own, and the
    /// commit marker is written only once `insert_with` has accepted the
    /// batch. A staged group with no marker is never replayed -- that is what
    /// the form exists for, and it is the same mechanism a transaction uses --
    /// so every crash point has one of two outcomes and no third. The rewind
    /// on the failure path is hygiene, not correctness: it keeps a refused
    /// statement from leaving bytes in the log, and the log would ignore them
    /// either way.
    ///
    /// It costs one extra record (the marker, ~10 bytes) per statement against
    /// the plain path's one, and the same single `fsync`.
    fn insert_unique(&mut self, path: &str, full: Block) -> Result<usize> {
        // Inside a transaction the ordinary path is already this shape: the
        // record is staged under the transaction's group, and a refusal
        // poisons the transaction so the group is never committed.
        if self.txn.is_some() {
            self.log_insert(path, &full)?;
            return self
                .catalog
                .table_by_path_mut(path)?
                .insert_with(full, KeyConflict::Reject);
        }
        let staged = match self.wal_for(path)? {
            Some(w) => {
                let seq = w.begin();
                let lsn = w.lsn();
                w.append_insert_staged(seq, &full)?;
                Some((seq, lsn))
            }
            None => None,
        };
        let applied = self
            .catalog
            .table_by_path_mut(path)?
            .insert_with(full, KeyConflict::Reject);
        match (&applied, staged) {
            (Ok(_), Some((seq, _))) => {
                let t = self.fold_bytes;
                let w = self.wal_for(path)?.expect("the log that staged the record");
                w.commit(seq)?;
                w.sync()?;
                // The same threshold `log_insert` checks, and it has to be
                // repeated because this path stages and commits by hand rather
                // than routing through it. Without it a table declared UNIQUE
                // is the one shape `wal_fold_bytes` does not reach: measured at
                // a 128 KiB threshold, 200 single-row inserts left 414356 bytes
                // of log against 16484 for the same inserts into a table with
                // no UNIQUE. One `u64` load and one compare, beside the `fsync`
                // on the line above.
                let over = t != 0 && w.pending() >= t;
                if over {
                    self.mark_fold_due(path);
                }
            }
            (Err(_), Some((_, lsn))) => {
                if let Some(w) = self.wal_for(path)? {
                    // Best effort, and safe either way: an uncommitted group
                    // is inert.
                    let _ = w.rewind_to(lsn);
                }
            }
            _ => {}
        }
        applied
    }

    /// Evaluate literal VALUES rows into a block covering only `order`.
    fn values_to_block(
        &self,
        rows: &[Vec<crate::sql::ast::Expr>],
        target: &Schema,
        order: &[usize],
    ) -> Result<Block> {
        use crate::sql::ast::{Expr, UnaryOp};
        let mut builders: Vec<ColumnBuilder> = order
            .iter()
            .map(|&c| ColumnBuilder::with_capacity(target.ty(c).clone(), rows.len()))
            .collect();
        for r in rows {
            if r.len() != order.len() {
                return Err(Error::bind(format!(
                    "VALUES row has {} entries, expected {}",
                    r.len(),
                    order.len()
                )));
            }
            for (i, e) in r.iter().enumerate() {
                let ty = target.ty(order[i]);
                let v = match e {
                    Expr::Literal(v) => coerce_literal(v, ty)?,
                    // The lexer emits `-1` as unary minus over a literal.
                    Expr::UnaryOp { op: UnaryOp::Neg, expr } => match &**expr {
                        Expr::Literal(v) => {
                            let neg = match v {
                                Value::Int(i) => Value::Int(-i),
                                Value::UInt(u) => Value::Int(-(*u as i64)),
                                Value::Float(f) => Value::Float(-f),
                                // `-2.5` in a VALUES row: the literal is
                                // exact, and negating it through `f64` would
                                // be the one place an INSERT threw the
                                // exactness away.
                                Value::Decimal(u, s) => Value::Decimal(-u, *s),
                                other => {
                                    return Err(Error::bind(format!("cannot negate {other}")))
                                }
                            };
                            coerce_literal(&neg, ty)?
                        }
                        other => {
                            return Err(Error::bind(format!(
                                "VALUES only accepts literals, got `{other}`"
                            )))
                        }
                    },
                    other => {
                        return Err(Error::bind(format!(
                            "VALUES only accepts literals, got `{other}`"
                        )))
                    }
                };
                builders[i].push_value(&v)?;
            }
        }
        Block::new(builders.into_iter().map(|b| b.finish()).collect())
    }

    /// Expand a partial-column block to the table's full width, filling
    /// unmentioned columns with NULL or a type default.
    fn widen_to_schema(&self, b: Block, target: &Schema, order: &[usize]) -> Result<Block> {
        if order.len() == target.len() && order.iter().enumerate().all(|(i, &c)| i == c) {
            return Ok(b);
        }
        let n = b.rows();
        let mut slots: Vec<Option<Column>> = vec![None; target.len()];
        for (i, &c) in order.iter().enumerate() {
            slots[c] = Some(b.columns[i].clone());
        }
        let mut out = Vec::with_capacity(target.len());
        for (c, slot) in slots.into_iter().enumerate() {
            match slot {
                Some(col) => out.push(col),
                None => {
                    // A declared DEFAULT wins over the type's zero. It was cast
                    // to the column type once at DDL, so this is a borrow and a
                    // fill -- no parse, no bind, nothing per row.
                    let ty = target.ty(c);
                    let fill = match target.field(c).default_value() {
                        Some(v) => v.clone(),
                        None if ty.is_nullable() => Value::Null,
                        None => ty.zero_value(),
                    };
                    out.push(Column::constant(ty, &fill, n)?);
                }
            }
        }
        Block::new(out)
    }

    // -------------------------------------------------------------- mutations
    //
    // `DELETE` and `UPDATE` are bulk statements here, not loops. Both bind
    // through `Binder::bind_delete`/`bind_update` and run the *ordinary*
    // optimizer over the result, so a mutation's predicate is pushed into the
    // scan as PREWHERE, has its zone maps derived and its constants folded by
    // the same passes a `SELECT` gets -- there is no second predicate
    // implementation to drift, and `EXPLAIN` over a mutation is what shows it.
    // `Table::delete_where` then evaluates that predicate block-at-a-time and
    // publishes **one** new part-set version for the whole statement.
    //
    // What that replaced, and why the shape mattered more than the constant
    // factor: the old path selected the matching primary keys into a `Vec`,
    // buffered a tombstone per key through the delta, and framed a log record
    // per key. So the row count bought delta churn and per-key index probes,
    // neither of which a delete needs. Measured in memory, A/B interleaved
    // (both implementations alternating in one loop, best of 7 per side),
    // 200k-row keyed table:
    //
    // ```text
    //   DELETE FROM t WHERE id < n        per-key       sweep     speedup
    //         n =   1 000                 1.28 ms     0.051 ms      25.1x
    //         n =  50 000                 6.22 ms     0.496 ms      12.5x
    //         n = 200 000 (all of it)    19.05 ms     1.668 ms      11.4x
    //         n =  50 000, 12 columns     6.15 ms     0.495 ms      12.4x
    // ```
    //
    // Per *affected row* that is 0.095 us down to 0.008 us. The residue is the
    // predicate scan itself, which is the same work the matching `SELECT` does
    // and cannot be removed; what is gone is everything that was proportional
    // to the row count on the write side.
    //
    // UPDATE keeps a re-insert, so it stays O(rows rewritten) by construction
    // -- but coalescing that re-insert (see `update_blocks`) took the same
    // three sizes to 0.99x / 1.31x / 2.08x, i.e. 55.80 ms -> 26.80 ms for a
    // whole-table update. A `SELECT` over the same predicate shares the scan
    // and is untouched: 0.890 ms for `count()` over the full 200k either way.
    //
    // Rejected, and worth recording so it is not retried blind: lowering the
    // sweep through `physical::lower` so `DELETE ... WHERE pk = c` becomes an
    // `IndexLookup`. The operator yields *values*, and a sweep needs the
    // `(part, row)` a value came from, so it would be a second access path to
    // keep in step. The zone maps already answer that case by header check:
    // deleting one row by key out of 200 000 measures 0.053 ms end to end,
    // parse and publish included.
    //
    // ## What a *persistent* session pays on top, measured
    //
    // The sweep above is the in-memory half. On disk the statement also has to
    // become durable, and that is where the time actually went. Numbers below
    // are A/B interleaved (best-of-5 per side, alternating) on a table
    // checkpointed first, so the log starts empty:
    //
    // ```text
    //   persistent DELETE, rows hidden    per-record log     batched      speedup
    //         1 000                          20.80 ms       5.81 ms        3.6x
    //         5 000                          91.86 ms      10.00 ms        9.2x
    //        50 000                         880.67 ms       6.91 ms      127.5x
    // ```
    //
    // The old shape framed and `write(2)`-ed one nineteen-byte record per
    // hidden row, so the syscall *was* the statement -- 17.7 us per row, four
    // orders of magnitude above the 0.008 us the sweep itself spends. One
    // buffer and one write per statement leaves the cost flat in the row
    // count, which is what "bulk" was supposed to mean. See
    // `Wal::append_deletes`.
    //
    // The unkeyed route pays a table fold instead of a log write, and it is
    // dominated by fsync rather than by size -- 53.6 ms to delete 5 000 rows
    // of 10 000, 61.3 ms to delete 50 000 of 100 000, against 0.12 ms and
    // 0.81 ms for the same sweeps with logging off. That is the price of
    // durability without a durable row identity, and at bulk sizes it is
    // still an order of magnitude *under* what the keyed log path cost before
    // this change.

    fn run_alter_delete(
        &mut self,
        table: &ObjectName,
        predicate: &crate::sql::ast::Expr,
    ) -> Result<ResultSet> {
        let path = self.catalog.qualify(table);
        let Some(sweep) = self.plan_sweep(table, predicate)? else {
            // The optimizer proved the predicate can never be TRUE and folded
            // the source to `Empty`. No row matches; nothing to publish.
            return Ok(ResultSet::with_affected(0));
        };
        // Atomic as a statement: the sweep is one publish, but the enlistment
        // and the log records around it are not, and a failure between them
        // must leave no rows deleted rather than an arbitrary prefix.
        self.atomic_stmt(move |s| s.apply_sweep(&path, &sweep))
            .map(ResultSet::with_affected)
    }

    fn run_alter_update(
        &mut self,
        table: &ObjectName,
        assignments: &[(String, crate::sql::ast::Expr)],
        predicate: &crate::sql::ast::Expr,
    ) -> Result<ResultSet> {
        // The sweep reads parts, and so does the scan under the new row
        // images, so anything still buffered has to be in one first. `plan`
        // does this for a query; a mutation does not go through `plan`.
        self.catalog.flush_all()?;
        let path = self.catalog.qualify(table);
        let mut m = Binder::new(&self.catalog).bind_update(table, assignments, predicate)?;

        // An UPDATE that carries the primary key through unchanged needs no
        // delete half at all: the re-insert tombstones the original *by key*
        // on its way through the keyed delta, which is what makes a mutation
        // here last-write-wins. Anything else has nothing for the insert to
        // shadow -- an unkeyed table has no key to shadow by, and an UPDATE
        // that assigns the key writes a row under a *different* one -- so the
        // old row has to be hidden positionally or it stays live alongside its
        // replacement. `bind_update` emits a bare column reference for every
        // column it does not assign, and its scan projects the full schema in
        // order, so this reads the assignment for the key exactly.
        let carried = self.catalog.table_by_path(&path)?.pk_col().is_some_and(|pk| {
            matches!(&m.source, LogicalPlan::Project { exprs, .. }
                if exprs.get(pk).and_then(|e| e.as_column()) == Some(pk))
        });

        m.source = optimizer::optimize(m.source)?;
        // Strictly before the sweep: hiding the rows first would leave the
        // scan under this nothing to read.
        let blocks = operators::execute_ctx(&m.source, &self.catalog, &self.limits.context())?.0;
        if blocks.iter().all(|b| b.rows() == 0) {
            return Ok(ResultSet::with_affected(0));
        }
        // Bound a second time, as a DELETE. The update's own source projects
        // every column -- it *is* the replacement row -- while the delete half
        // only needs the columns the predicate reads, so this is what keeps a
        // one-column predicate on a forty-column table from decoding forty
        // columns twice. Two binds per statement, nothing per row.
        let sweep = match carried {
            true => None,
            false => self.plan_sweep(table, predicate)?,
        };
        self.atomic_stmt(move |s| {
            // Deletes before the insert, in the log as well as in memory: a
            // replay that appended the new row first and then applied a key
            // delete for it would remove the row the statement wrote.
            if let Some(sw) = &sweep {
                s.apply_sweep(&path, sw)?;
            }
            // `carried` is exactly "the key is written back unchanged", which
            // is the only case where the re-insert is *meant* to land on top
            // of an existing key. Everything else assigns the key, so the
            // sweep has already retired every row this statement rewrites, and
            // a key that still has a live row belongs to a row the statement
            // did not name -- overwriting it is data loss, not an update.
            let mode = match carried {
                true => KeyConflict::Replace,
                false => KeyConflict::Reject,
            };
            s.update_blocks(&path, blocks, mode)
        })
        .map(ResultSet::with_affected)
    }

    /// Bind `DELETE FROM table WHERE predicate` and lower it to a positional
    /// sweep. `None` means no row can match.
    fn plan_sweep(
        &mut self,
        table: &ObjectName,
        predicate: &crate::sql::ast::Expr,
    ) -> Result<Option<Sweep>> {
        // Same preamble as `plan`, for the same reason: positions only exist
        // inside parts, so the write buffer has to be folded into one first.
        self.catalog.flush_all()?;
        let m = Binder::new(&self.catalog).bind_delete(table, predicate)?;
        Sweep::of(optimizer::optimize(m.source)?)
    }

    /// Apply one bulk delete: hide the rows, make them durable, one publish.
    ///
    /// ## A row's identity is not always a key, so there are two names for it
    ///
    /// The log's delete record names a primary-key *lane*
    /// ([`crate::persist::Wal::append_delete`]), which is a value and therefore
    /// survives anything that moves the row. When the table has one that is
    /// the whole story: hide the rows, log a lane each, done in O(parts).
    ///
    /// A table with only `ORDER BY`, or with a composite `PRIMARY KEY`, has no
    /// such lane -- and that is the *default* MergeTree shape. The sweep hides
    /// rows by position, so the record is the position; what a position needs
    /// is something that pins the rows it indexes. The obvious candidate,
    /// `(part index, row)`, does not: replay reconstructs the table as
    /// *checkpointed parts + the log*, and a part built after the checkpoint
    /// exists only as the `Insert` records that fed it, so whether those rows
    /// land in one part or three depends on flush timing and compaction --
    /// neither of which is in the log.
    ///
    /// [`crate::storage::Part::pid`] is a candidate that does. It is written
    /// inside the part file, so a part decoded off disk knows it without a
    /// sidecar; and a rewrite preserves it along with every row position,
    /// because `write_part_with` copies granules verbatim and only appends a
    /// mask. So `(pid, pos)` names the same row after a crash, after the
    /// checkpoint that moved the part to a new file, and after a restore --
    /// the difference between this and `(part index, row)` is that a part file
    /// is a *fact* while a part index is a reconstruction.
    ///
    /// That leaves exactly one gap, and it is the one the reasoning above
    /// identified: a part built since the last checkpoint has no file, so
    /// nothing names it. Those rows -- and only those -- take the old route,
    /// [`Session::mark_fold`]: hide them in memory and let COMMIT write the
    /// table's parts out, then discard the log they cover. One table rewrite,
    /// for the statements that need it instead of for all of them.
    ///
    /// Measured, 120 `BEGIN`/`DELETE`/`COMMIT` transactions on an unkeyed
    /// table: 4 durability barriers per transaction and a full table rewrite
    /// each, against 1 barrier and ~18 log bytes when the rows are citable --
    /// which is what the keyed path costs.
    fn apply_sweep(&mut self, path: &str, sweep: &Sweep) -> Result<usize> {
        let keyed = self.catalog.table_by_path(path)?.pk_col().is_some();
        // Enlisted unconditionally, and this is load-bearing:
        // `Table::edit`/`publish` redirect into the transaction's private
        // overlay only once `begin_txn` has run on that table, and `enlist` is
        // what runs it. Without this an in-memory session -- which logs
        // nothing, so never reached `enlist` on this path -- would sweep
        // straight into the committed set and ROLLBACK would have nothing to
        // drop.
        self.enlist(path)?;
        // An in-memory delete of a million rows pays for neither channel: a
        // lane is a packed-lane read per hidden row, a position is a varint,
        // and a session with no log needs neither.
        let mut keys = Vec::new();
        // Taken and put back so the buffer's capacity survives the statement:
        // a session deleting in a loop allocates once. A statement that fails
        // forfeits it, which costs one allocation and keeps the happy path
        // straight.
        let mut masks = std::mem::take(&mut self.masks);
        masks.clear();
        let sink = match (self.wal_enabled, keyed) {
            (false, _) => SweepLog::None,
            (true, true) => SweepLog::Keys(&mut keys),
            (true, false) => SweepLog::Masks(&mut masks),
        };
        let n = self.catalog.table_by_path_mut(path)?.delete_where_keys(
            &sweep.projection,
            sweep.pred.as_ref(),
            &sweep.zone,
            sink,
        )?;
        if !keys.is_empty() {
            self.log_deletes(path, &keys)?;
        } else if !masks.is_empty() {
            self.log_masks(path, &masks)?;
        }
        // Rows hidden where no record could name them. Only these need the
        // table written out, and a sweep over aged rows has none.
        if masks.dark > 0 {
            self.mark_fold(path)?;
        }
        self.masks = masks;
        Ok(n)
    }

    fn update_blocks(
        &mut self,
        path: &str,
        blocks: Vec<Block>,
        mode: KeyConflict,
    ) -> Result<usize> {
        // Coalesced into one insert, not one per executor block. `Table::insert`
        // packs any batch at or above `BULK_INSERT_THRESHOLD` straight into a
        // part of its own, so feeding it 8192-row blocks one at a time built a
        // part per block and then tripped `maybe_auto_compact` part way through
        // the statement -- rewriting 200k rows cost twenty-five part builds
        // plus a full merge of the table. One block is one sort, one pack, one
        // part, and one log record. Sources are dropped as they are folded in,
        // so the peak is the result plus the `Vec` growth slack rather than
        // twice the result.
        let mut it = blocks.into_iter().filter(|b| b.rows() > 0);
        let Some(mut acc) = it.next() else { return Ok(0) };
        for b in it {
            acc.extend(&b)?;
        }
        // The replacement row images, checked exactly as an INSERT's are --
        // and before the log record, so a refused UPDATE leaves nothing to
        // replay. An UPDATE is the *other* way a row can come to violate a
        // constraint, and a table that enforced its CHECKs on INSERT only
        // would be a constraint in name.
        self.enforce_checks(path, &acc)?;
        // An UPDATE is a re-insert of the changed rows, so logging the insert
        // is enough to replay it: the primary key makes it idempotent, and
        // where there is no key the sweep's tombstones were logged just above.
        self.log_insert(path, &acc)?;
        let t = self.catalog.table_by_path_mut(path)?;
        // Logged before it is attempted, and rejected *after*: that ordering
        // is exactly what the staged-record form exists for. The statement
        // runs inside `atomic_stmt`, so a refusal here rolls the transaction
        // back and `Wal::rewind_to` erases the record the write never earned.
        let n = t.insert_with(acc, mode)?;
        t.flush()?;
        Ok(n)
    }

    // -------------------------------------------- constraints: enforcement

    /// Refuse `block` if any row fails a CHECK declared on `path`.
    ///
    /// ## Vectorized, because this is the write path
    ///
    /// One `eval` per constraint per *block*, not per row: the predicate goes
    /// through the same evaluator a `WHERE` does, so a check over an
    /// `Int64` column is a specialized comparison over a decoded lane and
    /// costs about what filtering the same block would. What it must never
    /// become is a per-row bind or a per-row `Value` walk -- a buffered insert
    /// is ~33 ns/row, and either of those is an order of magnitude more than
    /// the row itself.
    ///
    /// Measured, A/B interleaved through the CLI, release build, best of 7 per
    /// side, `INSERT INTO t SELECT * FROM src` over 1M rows:
    ///
    /// ```text
    ///   one CHECK (v > 0)      0.58 s      no constraint    0.60 s
    ///   eight CHECKs           0.58 s      no constraint    0.56 s
    /// ```
    ///
    /// One constraint is **not measurable** -- it came out ahead in five of
    /// the seven rounds, which is what this machine's +-40 ms spread on
    /// identical code looks like. Eight of them cost about 20 ms per million
    /// rows, i.e. ~2.5 ns per row per constraint, still inside the noise of a
    /// single side. The same insert re-parsed from 200k literal `VALUES` rows
    /// measured 1.12 s against 1.11 s, where the parser is the statement and
    /// the check is invisible.
    ///
    /// ## NULL passes
    ///
    /// SQL's rule, and deliberately kept: a constraint is violated only when
    /// the predicate is **FALSE**, so `CHECK (v > 0)` accepts a NULL `v`. The
    /// alternative reading -- "not TRUE is a violation" -- would make a
    /// `Nullable` column with a CHECK unwritable without a DEFAULT, and would
    /// disagree with every other engine. Declare the column non-nullable if
    /// NULL is not allowed; nullability is a type here, and the type already
    /// refuses it.
    fn enforce_checks(&self, path: &str, block: &Block) -> Result<()> {
        // The gate. One load and a branch on a database that declares no
        // constraints, which is every database that had one before this
        // existed.
        if self.ext.checks.is_empty() {
            return Ok(());
        }
        let Some(checks) = self.ext.checks.get(path) else {
            return Ok(());
        };
        // Borrowed, not cloned: this runs once per statement on the write
        // path, and both borrows below are shared ones of the same catalog.
        let schema = &self.catalog.table_by_path(path)?.def.schema;
        let mut binder = Binder::new(&self.catalog);
        for c in checks {
            let bound = binder.bind_expr_standalone(&c.expr, schema).map_err(|e| {
                Error::bind(format!(
                    "constraint `{}` on `{path}` no longer binds against the table: {e}",
                    c.name
                ))
            })?;
            let col = crate::exec::expr::eval(&bound, block)?;
            if let Some(row) = first_false(&col) {
                return Err(Error::exec(format!(
                    "CHECK constraint `{}` on `{path}` is violated by {}: {}",
                    c.name,
                    render_row(block, schema, row),
                    c.sql
                )));
            }
        }
        Ok(())
    }

    /// Refuse a schema change that would leave a CHECK unable to bind.
    ///
    /// The alternative is a constraint that silently stops being enforced --
    /// or, worse, one that fails every write afterwards with a message about
    /// binding rather than about the column that went away. Asked by binding,
    /// so it is exactly the question the write path will ask.
    fn rebind_checks(&self, path: &str, schema: &Schema, what: &str) -> Result<()> {
        let Some(checks) = self.ext.checks.get(path) else { return Ok(()) };
        for c in checks {
            Binder::new(&self.catalog)
                .bind_expr_standalone(&c.expr, schema)
                .map_err(|e| {
                    Error::bind(format!(
                        "cannot {what}: constraint `{}` on `{path}` would no longer bind \
                         ({e}). Re-create the table without the constraint first",
                        c.name
                    ))
                })?;
        }
        Ok(())
    }

    /// Refuse a statement that would write the engine's own metadata table.
    ///
    /// Enforcement reads the copy in memory, so a hand-written `INSERT` into
    /// `_granular_ddl` would produce a table whose rows and whose behaviour
    /// disagree -- and the disagreement would only show up after a restart.
    fn guard_ddl_table(&self, name: &ObjectName) -> Result<()> {
        if !name.last().eq_ignore_ascii_case(DDL_TABLE) {
            return Ok(());
        }
        Err(Error::unsupported(format!(
            "`{DDL_TABLE}` is the catalog's own table -- it holds this database's CHECK \
             constraints and views, and the engine keeps a copy in memory that a direct \
             write would not update. Use CREATE/DROP VIEW and CREATE TABLE ... CHECK; \
             read it with SELECT"
        )))
    }

    // ------------------------------------------------------------------ DDL

    /// `CREATE VIEW v AS <query>`.
    ///
    /// The body is **validated by binding it**, here, against the catalog as
    /// it stands -- a view that cannot be planned is refused at creation
    /// rather than at the first `SELECT` from it. It is also qualified here
    /// (see [`crate::sql::ast::Query::qualify_tables`]), so what is stored
    /// means the same thing from any session.
    fn run_create_view(
        &mut self,
        name: &ObjectName,
        query: &crate::sql::ast::Query,
        body_sql: &str,
        or_replace: bool,
        if_not_exists: bool,
    ) -> Result<ResultSet> {
        self.guard_ddl_table(name)?;
        let (db, view) = self.catalog.resolve(name);
        let key = format!("{db}.{view}");
        // Asked before anything is installed: the view is recorded in a table
        // *in that database*, so a name in a database that does not exist
        // would leave a view in memory that nothing could persist.
        self.catalog.table_names(Some(&db))?;
        // A view and a table share one namespace, because a query names them
        // the same way. Shadowing either direction is a silent wrong answer:
        // the reference would resolve to whichever the rewrite happened to
        // check first.
        if self.catalog.table_by_path(&key).is_ok() {
            return Err(Error::bind(format!(
                "`{key}` is a table; a view cannot shadow it"
            )));
        }
        if self.ext.views.contains_key(&key) && !or_replace {
            return if if_not_exists {
                Ok(ResultSet::empty())
            } else {
                Err(Error::bind(format!(
                    "view `{key}` already exists (use CREATE OR REPLACE VIEW)"
                )))
            };
        }
        let scope = self.catalog.current_database().to_string();
        let mut q = query.clone();
        q.qualify_tables(&scope);
        // Binding is the validation, and it has to happen with the view *not*
        // yet installed: otherwise `CREATE OR REPLACE VIEW v AS SELECT * FROM
        // v` would validate against its own previous definition and then
        // recurse at every use.
        self.plan(&q)?;
        let replaced = self
            .ext
            .views
            .insert(key.clone(), View { scope, sql: body_sql.to_string(), query: q });
        // `persist_ext` reads what is in memory, so the entry has to go in
        // first -- and come back out if the write fails, or the session would
        // enforce a view the catalog does not have.
        if let Err(e) = self.persist_ext(&db) {
            match replaced {
                Some(v) => self.ext.views.insert(key, v),
                None => self.ext.views.remove(&key),
            };
            return Err(e);
        }
        Ok(ResultSet::empty())
    }

    fn run_drop_view(&mut self, name: &ObjectName, if_exists: bool) -> Result<ResultSet> {
        let (db, view) = self.catalog.resolve(name);
        let key = format!("{db}.{view}");
        let Some(gone) = self.ext.views.remove(&key) else {
            return if if_exists {
                Ok(ResultSet::empty())
            } else {
                Err(Error::storage(format!("view `{key}` does not exist")))
            };
        };
        if let Err(e) = self.persist_ext(&db) {
            self.ext.views.insert(key, gone);
            return Err(e);
        }
        Ok(ResultSet::empty())
    }

    /// `RENAME TABLE a TO b`.
    ///
    /// ## Atomic, and it is the checkpoint that makes it so
    ///
    /// The rename itself is three map operations on the catalog. What has to
    /// be atomic is the *directory*, and the trick is that nothing here
    /// renames one: the table's parts are hard-linked under the new name
    /// first, the new directory's `TABLE` record is committed, and only then
    /// does the root `CATALOG` -- the single commit point of a checkpoint --
    /// start naming the new table. A crash before that publication leaves the
    /// old name with its own untouched directory and an orphan beside it that
    /// the next checkpoint's `collect_dropped_tables` removes; a crash after
    /// it leaves the new name complete. There is no third state, and no
    /// window in which a name resolves to a directory that is not there.
    ///
    /// Hard links rather than a copy because a rename must not be proportional
    /// to the table: parts are immutable, and two directory entries for one
    /// inode is exactly what "the same part, under a new table name" means.
    /// Measured through the CLI, A/B interleaved against a build that skipped
    /// the linking and let the checkpoint rewrite the parts (temporary env
    /// switch, since removed), 1M rows / 14 MB, best of 5 per side:
    /// **0.08 s linked against 0.23 s copied**, whole process included. The
    /// gap is the part copy and grows with the table; the linked side does
    /// not.
    ///
    /// The constraints move with the table, and they move *first*, under both
    /// names at once. A crash between the two publications then leaves
    /// whichever table survived still carrying its CHECKs; the row for the
    /// name that lost is pruned at the next open.
    fn run_rename_table(&mut self, from: &ObjectName, to: &ObjectName) -> Result<ResultSet> {
        let (from_db, from_name) = self.catalog.resolve(from);
        let (to_db, to_name) = self.catalog.resolve(to);
        let from_path = format!("{from_db}.{from_name}");
        let to_path = format!("{to_db}.{to_name}");
        if from_path == to_path {
            return Ok(ResultSet::empty());
        }
        self.guard_ddl_table(from)?;
        self.guard_ddl_table(to)?;
        // Resolve both ends before touching anything.
        self.catalog.table_by_path(&from_path)?;
        if self.catalog.table_by_path(&to_path).is_ok() {
            return Err(Error::storage(format!("table `{to_path}` already exists")));
        }
        if self.ext.views.contains_key(&to_path) {
            return Err(Error::bind(format!("`{to_path}` is a view; a table cannot shadow it")));
        }
        if !crate::persist::store::is_safe_name(&to_name) {
            return Err(Error::storage(format!(
                "table name `{to_name}` cannot be a directory name"
            )));
        }
        // Includes `from_name` itself when the two databases are the same, so
        // `RENAME TABLE beta TO ALPHA` beside an `alpha` is refused -- and so
        // is `RENAME TABLE alpha TO ALPHA`, which has no safe spelling on a
        // case-insensitive filesystem. Both destroyed *both* tables before
        // this line existed: the link and the drop resolve to one directory.
        self.guard_dir_name(Some(&to_db), &to_name)?;

        // Everything the old name owns has to be on disk before the parts can
        // be linked under the new one, and the delta has to be empty because
        // buffered rows live in neither directory.
        self.checkpoint()?;

        // Phase 1: the constraints, under *both* names. Published on its own,
        // so no crash can leave the surviving table without them.
        let mut moved_meta = false;
        if let Some(cs) = self.ext.checks.get(&from_path) {
            let copy: Vec<Check> = cs
                .iter()
                .map(|c| Check { name: c.name.clone(), sql: c.sql.clone(), expr: c.expr.clone() })
                .collect();
            self.ext.checks.insert(to_path.clone(), copy);
            moved_meta = true;
        }
        if let Some(u) = self.ext.uniques.get(&from_path).cloned() {
            self.ext.uniques.insert(to_path.clone(), u);
            moved_meta = true;
        }
        if moved_meta {
            self.persist_ext(&to_db)?;
            if to_db != from_db {
                self.persist_ext(&from_db)?;
            }
            self.checkpoint()?;
        }

        // Phase 2: the parts, linked under the new name, with the new
        // directory's own commit record -- all of it invisible until the
        // `CATALOG` below names it.
        if let Some(root) = self.catalog.dir().map(Path::to_path_buf) {
            let t = self.catalog.table_by_path(&from_path)?;
            let mut def = t.def.clone();
            def.name = to_name.clone();
            let snap = t.snapshot();
            let dbdir = root.join(&to_db);
            std::fs::create_dir_all(&dbdir)
                .map_err(|e| Error::Io(format!("cannot create {}: {e}", dbdir.display())))?;
            // The stream position the *destination name's* log has already
            // reached, for the same reason `CREATE TABLE` stamps one: a log
            // directory outlives its table, so renaming onto a dropped name
            // meets a stream that does not start at zero.
            let end = crate::persist::wal::stream_end(&root, &to_db, &to_name);
            link_table_dir(
                &root.join(&from_db).join(&from_name),
                &dbdir.join(&to_name),
                &def,
                &snap,
                end,
            )?;
        }

        // Phase 3: the catalog. `Table` has no public move, so the old entry
        // is swapped out for a placeholder that is dropped a line later --
        // this never clones a part.
        let mut def = self.catalog.table_by_path(&from_path)?.def.clone();
        def.name = to_name.clone();
        let empty = crate::storage::Table::new(def.clone(), crate::catalog::DEFAULT_DELTA_LIMIT);
        let mut moved = std::mem::replace(self.catalog.table_by_path_mut(&from_path)?, empty);
        moved.def.name = to_name.clone();
        self.wals.remove(&from_path);
        self.catalog.drop_table(from, false)?;
        let mut qualified = def;
        qualified.name = to_path.clone();
        self.catalog.create_table(qualified, false)?;
        *self.catalog.table_by_path_mut(&to_path)? = moved;

        // Phase 4: the old name's constraints go, now that nothing names it.
        let mut dropped = self.ext.checks.remove(&from_path).is_some();
        dropped |= self.ext.uniques.remove(&from_path).is_some();
        if dropped {
            self.persist_ext(&from_db)?;
        }
        Ok(ResultSet::empty())
    }

    /// `ALTER TABLE t MODIFY COLUMN c <type>`: rewrite the column, or refuse.
    ///
    /// Every value is cast up front, and **the first one that does not fit
    /// fails the whole statement**, naming the row and the value. That is the
    /// only defensible answer for a schema change: the alternatives are
    /// storing a saturated or truncated value (silent, permanent data loss) or
    /// leaving the table half-migrated, and a `MODIFY COLUMN` that half-ran is
    /// unrecoverable because the old bytes are gone.
    ///
    /// The rewrite itself is the same shape as ADD/DROP COLUMN: build a fresh
    /// table, insert the recast blocks, and swap it in only once every block
    /// has been converted -- so a failure leaves the original untouched, and
    /// the checkpoint that follows is what makes the new one durable.
    fn run_modify_column(
        &mut self,
        table: &ObjectName,
        name: &str,
        ty: &DataType,
    ) -> Result<ResultSet> {
        self.guard_ddl_table(table)?;
        let path = self.catalog.qualify(table);
        let t = self.catalog.table_by_path_mut(&path)?;
        let idx = t.def.schema.require(name)?;
        if t.def.schema.ty(idx) == ty {
            return Ok(ResultSet::empty());
        }
        // A key column's lane is what the sparse index, the router and the
        // zone maps are built on, and its physical width is baked into every
        // part already written. Retyping one would need the index rebuilt in
        // the same statement; refusing says so.
        if t.def.order_by.contains(&idx) || t.def.primary_key.contains(&idx) {
            return Err(Error::unsupported(format!(
                "cannot MODIFY `{name}`: it is part of `{path}`'s key, and the parts on \
                 disk are sorted and indexed by its current type. Create a table with the \
                 type you want and `INSERT ... SELECT` into it"
            )));
        }
        let cols: Vec<usize> = (0..t.def.schema.len()).collect();
        let blocks = t.scan(&cols)?;
        let mut def = t.def.clone();
        def.schema = retyped_schema(&def.schema, idx, ty)?;
        self.rebind_checks(&path, &def.schema, &format!("MODIFY `{name}`"))?;

        // Cast first, publish second. `Column::cast_to` is per value and
        // reports the first failure; the row number it is given is the row
        // within the table, counted across blocks, because that is the number
        // that lets someone go and look at it.
        let mut recast = Vec::with_capacity(blocks.len());
        let mut seen = 0usize;
        for b in blocks {
            let rows = b.rows();
            let mut cs = b.columns;
            cs[idx] = cast_column(&cs[idx], ty, name, seen)?;
            recast.push(Block::new(cs)?);
            seen += rows;
        }
        let mut fresh = crate::storage::Table::new(def, crate::catalog::DEFAULT_DELTA_LIMIT);
        let mut n = 0;
        for b in recast {
            n += fresh.insert(b)?;
        }
        fresh.flush()?;
        *self.catalog.table_by_path_mut(&path)? = fresh;
        Ok(ResultSet::with_affected(n))
    }

    fn run_create_table(&mut self, c: &CreateTable) -> Result<ResultSet> {
        // CREATE TABLE ... AS SELECT takes its schema from the query.
        let (fields, as_blocks) = match &c.as_query {
            Some(q) => {
                let plan = self.plan(q)?;
                let s = plan.schema().clone();
                let blocks =
                    operators::execute_ctx(&plan, &self.catalog, &self.limits.context())?.0;
                (s.fields().to_vec(), Some(blocks))
            }
            None => (
                c.columns
                    .iter()
                    .map(|cd| {
                        let f = Field::new(cd.name.clone(), cd.ty.clone());
                        match &cd.default {
                            // The parser already produced a `Value` for a bare
                            // literal, so hand it over rather than rendering
                            // it to SQL text for `with_default` to re-parse.
                            Some(crate::sql::ast::Expr::Literal(v)) => {
                                f.with_default_value(v.clone())
                            }
                            // Anything else (`-1` arrives as unary minus over
                            // a literal) goes through the text path, which
                            // also rejects the non-constant ones.
                            Some(e) => f.with_default(&e.to_string()),
                            None => Ok(f),
                        }
                    })
                    .collect::<Result<Vec<_>>>()?,
                None,
            ),
        };
        if fields.is_empty() {
            return Err(Error::bind("CREATE TABLE requires at least one column"));
        }
        let schema = Schema::new(fields)?;

        let order_by = resolve_key_exprs(&c.order_by, &schema, "ORDER BY")?;
        // An undeclared PRIMARY KEY stays empty; it does NOT default to
        // ORDER BY. Defaulting it is what silently turned a sort key into a
        // unique key: `ORDER BY id` routed writes into the keyed delta, where
        // `put_keyed` overwrites the slot an existing key owns, so
        // `INSERT ... VALUES (4,1),(4,2)` reported two rows affected and
        // stored one. ClickHouse's ORDER BY is a sort key and duplicates are
        // legal; uniqueness has to be *declared* -- an explicit PRIMARY KEY,
        // or ReplacingMergeTree, both of which `TableDef::pk_col` honours.
        let primary_key = resolve_key_exprs(&c.primary_key, &schema, "PRIMARY KEY")?;
        if c.engine.is_sorted() && c.order_by.is_empty() {
            return Err(Error::bind(format!(
                "{} requires ORDER BY (use `ORDER BY tuple()` for none)",
                c.engine.name()
            )));
        }
        // Refused rather than approximated. Ingest already tombstones an
        // existing primary key, so a SummingMergeTree here would quietly
        // behave as a replacing engine and return last-write-wins where the
        // user asked for a sum — wrong answers dressed up as working ones.
        if c.engine == crate::types::Engine::SummingMergeTree {
            return Err(Error::unsupported(
                "SummingMergeTree: rows are collapsed by last-write-wins on the \
                 primary key, so summing at merge time is not implemented. Use \
                 MergeTree and aggregate with `sum()` at query time",
            ));
        }
        // A PRIMARY KEY that does not lead ORDER BY would route point lookups
        // to the wrong granule, so reject it rather than silently disabling
        // the index.
        if !primary_key.is_empty() && !order_by.is_empty() && primary_key[0] != order_by[0] {
            return Err(Error::bind("PRIMARY KEY must be a prefix of ORDER BY"));
        }
        let name = self.catalog.qualify(&c.name);
        // The reverse of the check in `run_create_view`: one namespace, so a
        // table may not take a name a view already answers to. Without this
        // the reference resolves to whichever the rewrite looks at first,
        // which is a wrong answer rather than an error.
        if self.ext.views.contains_key(&name) {
            return Err(Error::bind(format!(
                "`{name}` is a view; a table cannot shadow it (DROP VIEW it first)"
            )));
        }
        // `PARTITION BY` is refused by the parser, so no table this engine
        // creates has one. The slot stays on `TableDef` because it is a fixed
        // field of the TABLE file format: a database written before the
        // refusal must still open and still describe itself truthfully.
        let def =
            TableDef { name, schema, order_by, primary_key, partition_by: None, engine: c.engine };

        // Both constraint kinds are decided *before* the table exists, so a
        // declaration this engine cannot enforce leaves nothing behind.
        let checks = self.bind_checks(&c.checks, &def)?;
        check_unique_declarations(&c.columns, &def)?;

        let path = self.catalog.qualify(&c.name);
        if self.catalog.table_by_path(&path).is_ok() {
            // `IF NOT EXISTS` on an existing table: `create_table` is about to
            // do nothing, so nothing here may either -- least of all replace
            // the live table's constraints with this statement's.
            self.catalog.create_table(def, c.if_not_exists)?;
            return Ok(ResultSet::empty());
        }
        // Past the exact-name branch above, so this only ever sees a *new*
        // name -- `IF NOT EXISTS` on the table that is already there never
        // reaches it.
        let (db, tbl) = self.catalog.resolve(&c.name);
        self.guard_dir_name(Some(&db), &tbl)?;
        self.catalog.create_table(def, c.if_not_exists)?;
        // Stamp a commit record immediately, carrying the stream position this
        // table's log directory has *already* reached.
        //
        // The log lives at `<root>/.wal/<db>/<table>` and deliberately
        // survives `DROP TABLE`, which is what makes "restore to just before
        // the drop" work. So a table created under a dropped one's name would
        // otherwise open with no `TABLE` file, default to a watermark of zero,
        // and replay the previous incarnation's entire stream into a schema
        // that need not match it. One `atomic_write` on a DDL statement closes
        // that, and it also means a missing `TABLE` file can no longer mean
        // "young table" -- only damage.
        if let Some(root) = self.catalog.dir().map(Path::to_path_buf) {
            let end = crate::persist::wal::stream_end(&root, &db, &tbl);
            // Same reasoning for part *identities*: this table holds no parts,
            // so its counter would start at 1 and reissue the numbers a
            // dropped predecessor's archived parts still carry. The stream
            // position is monotone across the drop; seeding from it makes a
            // stale citation name an identity that does not exist, which is
            // refused, rather than one that resolves to the wrong row.
            let t = self.catalog.table_by_path_mut(&path)?;
            t.seed_pids(end);
            let t = self.catalog.table_by_path(&path)?;
            if t.def.engine.is_persistent() {
                crate::persist::write_table(&root.join(&db), t, end)?;
            }
        }
        // A re-created table must not inherit the constraints of the one that
        // had the name before it: a CREATE TABLE that declares none has
        // declared none.
        let mut changed = self.ext.checks.remove(&path).is_some();
        changed |= self.ext.uniques.remove(&path).is_some();
        if !checks.is_empty() {
            self.ext.checks.insert(path.clone(), checks);
            changed = true;
        }
        if let Some(u) = c.columns.iter().find(|c| c.unique) {
            self.ext.uniques.insert(path.clone(), u.name.clone());
            changed = true;
        }
        if changed {
            let (db, _) = self.catalog.resolve(&c.name);
            self.persist_ext(&db)?;
        }

        let mut n = 0;
        if let Some(blocks) = as_blocks {
            for b in blocks {
                if b.rows() > 0 {
                    self.enforce_checks(&path, &b)?;
                    n += self.catalog.table_by_path_mut(&path)?.insert(b)?;
                }
            }
        }
        Ok(if n > 0 { ResultSet::with_affected(n) } else { ResultSet::empty() })
    }

    /// Bind every declared CHECK against the table being created, and keep the
    /// ones that bind.
    ///
    /// Binding here is what makes a constraint refuse at DDL time rather than
    /// at the first insert: `CHECK (nosuch > 0)` names a column that does not
    /// exist, `CHECK (v)` is not a predicate, and `CHECK (count(*) > 0)` is an
    /// aggregate -- all three are the user's mistake, and all three are far
    /// cheaper to report now than on a write six months later.
    fn bind_checks(
        &self,
        decls: &[crate::sql::ast::CheckDef],
        def: &TableDef,
    ) -> Result<Vec<Check>> {
        let mut out: Vec<Check> = Vec::with_capacity(decls.len());
        for (i, d) in decls.iter().enumerate() {
            let name = match &d.name {
                Some(n) => n.clone(),
                None => format!("check_{}", i + 1),
            };
            if out.iter().any(|c| c.name.eq_ignore_ascii_case(&name)) {
                return Err(Error::bind(format!(
                    "two constraints on `{}` are both named `{name}`",
                    def.name
                )));
            }
            let bound = Binder::new(&self.catalog)
                .bind_expr_standalone(&d.expr, &def.schema)
                .map_err(|e| Error::bind(format!("CHECK `{name}`: {e}")))?;
            if bound.ty().base() != &DataType::Bool {
                return Err(Error::bind(format!(
                    "CHECK `{name}` is {} rather than a condition: write a comparison, \
                     e.g. `CHECK ({} <> 0)`",
                    bound.ty(),
                    d.expr
                )));
            }
            // Stored as text, because that is what the catalog row holds and
            // what has to survive a restart. Rendered from the AST rather than
            // sliced out of the statement so that what is stored is exactly
            // what was bound.
            out.push(Check { name, sql: d.expr.to_string(), expr: d.expr.clone() });
        }
        Ok(out)
    }

    /// `ALTER TABLE ... ADD COLUMN`: rebuild with the new column appended.
    fn run_add_column(
        &mut self,
        table: &ObjectName,
        col: &ColumnDef,
        if_not_exists: bool,
    ) -> Result<ResultSet> {
        self.guard_ddl_table(table)?;
        let path = self.catalog.qualify(table);
        let t = self.catalog.table_by_path_mut(&path)?;
        if t.def.schema.index_of(&col.name).is_some() {
            return if if_not_exists {
                Ok(ResultSet::empty())
            } else {
                Err(Error::bind(format!("column `{}` already exists", col.name)))
            };
        }
        let cols: Vec<usize> = (0..t.def.schema.len()).collect();
        let blocks = t.scan(&cols)?;
        let mut def = t.def.clone();
        // Backfill the existing rows with the DEFAULT, not the type's zero --
        // otherwise ADD COLUMN ... DEFAULT silently disagrees with what the
        // same DEFAULT does on a subsequent INSERT.
        let base = Field::new(col.name.clone(), col.ty.clone());
        let field = match &col.default {
            Some(crate::sql::Expr::Literal(v)) => base.with_default_value(v.clone())?,
            Some(e) => base.with_default(&e.to_string())?,
            None => base,
        };
        let fill = match field.default_value() {
            Some(v) => v.clone(),
            None if col.ty.is_nullable() => Value::Null,
            None => col.ty.zero_value(),
        };
        def.schema.push(field);
        let mut fresh = crate::storage::Table::new(def, crate::catalog::DEFAULT_DELTA_LIMIT);
        let mut n = 0;
        for b in blocks {
            let rows = b.rows();
            let mut cs = b.columns.clone();
            cs.push(Column::constant(&col.ty, &fill, rows)?);
            n += fresh.insert(Block::new(cs)?)?;
        }
        fresh.flush()?;
        *self.catalog.table_by_path_mut(&path)? = fresh;
        Ok(ResultSet::with_affected(n))
    }

    fn run_drop_column(
        &mut self,
        table: &ObjectName,
        name: &str,
        if_exists: bool,
    ) -> Result<ResultSet> {
        self.guard_ddl_table(table)?;
        let path = self.catalog.qualify(table);
        let t = self.catalog.table_by_path_mut(&path)?;
        let Some(idx) = t.def.schema.index_of(name) else {
            return if if_exists {
                Ok(ResultSet::empty())
            } else {
                Err(Error::bind(format!("column `{name}` does not exist")))
            };
        };
        if t.def.order_by.contains(&idx) || t.def.primary_key.contains(&idx) {
            return Err(Error::bind(format!(
                "cannot drop `{name}`: it is part of the table's key"
            )));
        }
        let keep: Vec<usize> = (0..t.def.schema.len()).filter(|&i| i != idx).collect();
        let blocks = t.scan(&keep)?;
        let mut def = t.def.clone();
        def.schema = def.schema.project(&keep);
        self.rebind_checks(&path, &def.schema, &format!("drop `{name}`"))?;
        let remap = |i: usize| keep.iter().position(|&k| k == i).unwrap();
        def.order_by = def.order_by.iter().map(|&i| remap(i)).collect();
        def.primary_key = def.primary_key.iter().map(|&i| remap(i)).collect();
        def.partition_by = def.partition_by.map(remap);

        let mut fresh = crate::storage::Table::new(def, crate::catalog::DEFAULT_DELTA_LIMIT);
        let mut n = 0;
        for b in blocks {
            n += fresh.insert(b)?;
        }
        fresh.flush()?;
        *self.catalog.table_by_path_mut(&path)? = fresh;
        Ok(ResultSet::with_affected(n))
    }
}

// ----------------------------------------------------- one writer, N readers

/// One database, shared: a single writer and any number of concurrent readers.
///
/// ## Why an `RwLock<Session>` and not an `Arc<Catalog>` snapshot
///
/// The reader has to hand the executor a `&Catalog` that outlives the query --
/// `operators::build` takes one, `physical::lower` takes one, and `Scan`
/// resolves its table through one. So a reader owns either a *borrow* of the
/// writer's catalog or a *copy* of it, and both alternatives were tried on
/// paper first:
///
///   * **A copy** (`Arc<Catalog>` snapshot). Building one means building a
///     `Table` per table over the pinned `Arc<PartSet>`, and the only public
///     constructors take `Vec<Part>` by value while a `PartSet` holds
///     `Arc<Part>` *plus the delete masks beside them*. A copy through
///     `from_parts` silently drops the masks, which resurrects every deleted
///     row. Sharing a part set into a fresh `Table` needs one constructor
///     `storage::table` does not expose (see the report).
///   * **A borrow** (`Reader<'a>`). Free, `Send + Sync`, and it makes the
///     writer unreachable for as long as any reader exists -- the borrow
///     checker enforces single-writer by *forbidding* the concurrency the
///     phase is about. A pool cannot write while a connection is open.
///
/// So the shared thing is the `Session` itself, behind one `RwLock`. Readers
/// take it shared and run at once; the writer takes it exclusively for the
/// length of one statement. That is the decided single-writer / multi-reader
/// position expressed with the one primitive that can express it, and it costs
/// a reader one uncontended `read()` per *query* -- against a query that has
/// already parsed, bound, lowered and scanned.
///
/// A `Reader` is `Send + Sync + Clone` and owns no lifetime, which is what a
/// connection pool and a wire server's per-connection model need: hand each
/// connection a clone, and they run in parallel.
///
/// ## A `Db` *is* a connection
///
/// Cloning one mints a new identity, and that identity is what owns a
/// transaction: `COMMIT` and `ROLLBACK` are refused when the open transaction
/// belongs to a different clone. Without it a second connection's `COMMIT`
/// durably published work it never authored and its `ROLLBACK` silently
/// discarded a write it had been told was committed -- the same failure the
/// engine already ruled a bug at the nested-`BEGIN` boundary, one scope out.
/// So: **one clone per connection**, and share a clone only where you would
/// share a connection.
///
/// The identity is the clone rather than the [`Writer`] guard because a wire
/// server takes one guard per *statement*: `writer().execute("BEGIN")` and a
/// later `writer().execute("COMMIT")` are one transaction on one connection,
/// and a per-guard token would refuse them.
///
/// **Dropping a `Db` closes that connection**, which ends its transaction if
/// one is open -- the same thing every other database does when a client hangs
/// up, and mandatory once identity exists: connection ids are monotonic, so
/// without it an abandoned handle left a transaction nothing in the process
/// could ever name again, and every statement from every connection was
/// refused forever.
pub struct Db {
    inner: Arc<RwLock<Session>>,
    /// This connection's budget and deadline, held here rather than read out
    /// of the session so that [`Db::reader`] takes no lock at all. It used to:
    /// opening a connection waited out an unrelated `INSERT`, and taking one
    /// on a thread already holding the writer deadlocked with no query in
    /// sight. The one honest consequence is that
    /// `writer().set_memory_limit(..)` after `into_shared` no longer reaches
    /// later `Reader`s -- set it on the `Reader`, which is where a per-
    /// connection budget belongs.
    limits: Limits,
    /// This connection's identity *and* its liveness, in one allocation: the
    /// `u64` is the id every ownership check compares, and the `Arc` is what
    /// an open transaction holds a `Weak` on so a dropped connection can be
    /// told from a busy one. Ids are monotonic and never reused, so without
    /// the second half a disconnect while a transaction was open wedged the
    /// database permanently -- nothing in the process could produce that id
    /// again, and every statement from every connection was refused.
    owner: Arc<u64>,
}

/// Connection identities. Never 0: that is the bare-`Session` caller, whose
/// transactions are its own by construction.
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

fn mint_owner() -> Arc<u64> {
    Arc::new(NEXT_OWNER.fetch_add(1, Ordering::Relaxed))
}

impl Clone for Db {
    /// A new connection, not a second name for this one. See the type's docs.
    fn clone(&self) -> Db {
        Db {
            inner: Arc::clone(&self.inner),
            limits: self.limits.clone(),
            owner: mint_owner(),
        }
    }
}

impl Drop for Db {
    /// Closing a connection ends its transaction, which is what every other
    /// database does and what the owner token made mandatory: a `Db` clone
    /// *is* a connection, so a client that hangs up mid-transaction -- no
    /// panic, just a dropped handle -- must not leave one nothing can end.
    ///
    /// `try_write` rather than `write`, always: this drop can run on a thread
    /// that already holds the writer through *another* clone, and blocking
    /// there would trade a wedge for a deadlock. When it loses the race the
    /// transaction is swept by the next [`Db::writer`] instead, which is why
    /// the token is the guarantee and this is only the prompt path -- worth
    /// having because a reader refuses while any transaction is open, and
    /// readers alone would otherwise wait for a writer that may never come.
    fn drop(&mut self) {
        let Ok(mut g) = self.inner.try_write() else { return };
        if g.txn.as_ref().is_some_and(|t| t.owner == *self.owner) {
            g.reap_txn();
        }
    }
}

impl Db {
    pub fn in_memory() -> Db {
        Session::in_memory().into_shared()
    }

    pub fn open(dir: impl AsRef<Path>) -> Result<Db> {
        Ok(Session::open(dir)?.into_shared())
    }

    /// [`Session::open_read_only`], shared. Several of these coexist in one
    /// process and across processes; a writer excludes them all.
    pub fn open_read_only(dir: impl AsRef<Path>) -> Result<Db> {
        Ok(Session::open_read_only(dir)?.into_shared())
    }

    /// A handle that can run reads concurrently with every other one.
    ///
    /// Cheap, and now *unconditionally* cheap: one `Arc` clone and a copy of
    /// this connection's limits, with no lock taken. Take one per connection
    /// and give each its own budget and deadline.
    pub fn reader(&self) -> Reader {
        Reader { inner: Arc::clone(&self.inner), limits: self.limits.clone() }
    }

    /// Exclusive access, for the write set. Held until the guard drops, so a
    /// caller that wants several statements to be one unit simply keeps it.
    ///
    /// Readers block for exactly this long, which is why the guard should not
    /// be parked in a local across a `BEGIN`: use [`Db::transaction`].
    ///
    /// One rule: do not run a [`Reader`] *query* on the thread that holds this
    /// guard -- it would wait for a lock the same thread already owns. Taking
    /// the handle is free (see [`Db::reader`]); it is the query that blocks.
    /// The guard derefs to `Session`, so `writer().read(sql)` answers the same
    /// question without a second acquisition.
    ///
    /// The one store here is this connection's identity, which is what makes
    /// `COMMIT` and `ROLLBACK` able to tell whose transaction they are ending.
    pub fn writer(&self) -> Writer<'_> {
        let mut g = self.inner.write().unwrap_or_else(|e| e.into_inner());
        // The backstop for a connection whose own `Drop` could not take the
        // lock. One `Option` test on every acquisition, one `u64` compare when
        // a transaction is open, and the `Weak` is only touched for one that
        // belongs to somebody else -- the path that was about to raise an
        // error anyway.
        if g.txn.as_ref().is_some_and(|t| {
            t.owner != *self.owner && t.tok.as_ref().is_some_and(|w| w.strong_count() == 0)
        }) {
            g.reap_txn();
        }
        if g.owner != *self.owner {
            g.owner = *self.owner;
            g.owner_tok = Some(Arc::downgrade(&self.owner));
        }
        Writer { g }
    }

    /// Run `f` with exclusive access, wrapped in `BEGIN`/`COMMIT`.
    ///
    /// The writer lock is held across the whole transaction, which is what
    /// makes a concurrent reader see the transaction as one step: it cannot
    /// acquire the shared lock while the overlay exists, so it observes the
    /// state either wholly before or wholly after. Without this the reader
    /// would have to *refuse* -- an uncommitted overlay is what
    /// `Table::snapshot` hands out, so reading through it is a dirty read.
    ///
    /// `f` gets the `Session` itself, so it can read its own writes with
    /// [`Session::query`]. It must not reach for a [`Reader`]: that waits on
    /// the lock this call is holding, on this thread.
    pub fn transaction<R>(&self, f: impl FnOnce(&mut Session) -> Result<R>) -> Result<R> {
        let mut w = self.writer();
        w.begin()?;
        match f(&mut w) {
            Ok(v) => {
                w.commit()?;
                Ok(v)
            }
            Err(e) => {
                let _ = w.rollback();
                Err(e)
            }
        }
    }

    /// Run one or more write statements. Convenience for `writer().execute`.
    pub fn execute(&self, sql: &str) -> Result<()> {
        self.writer().execute(sql)
    }
}

impl Session {
    /// Move this session behind a shared handle: one writer, many readers.
    ///
    /// By value, because that is the honest signature: the whole point is that
    /// nobody keeps a `&mut Session` on the side while readers are running.
    pub fn into_shared(self) -> Db {
        // The limits are copied out here rather than read per `reader()`, and
        // `cancel` is an `Arc`, so a handle taken from the session before this
        // call still stops the queries a `Reader` runs after it.
        let limits = self.limits.clone();
        Db { inner: Arc::new(RwLock::new(self)), limits, owner: mint_owner() }
    }
}

/// Exclusive access to the session behind a [`Db`]. Derefs to `Session`, so
/// the entire existing write API is reachable unchanged.
pub struct Writer<'a> {
    g: RwLockWriteGuard<'a, Session>,
}

impl Drop for Writer<'_> {
    /// Roll back this connection's open transaction when the stack is
    /// unwinding.
    ///
    /// **Only** when unwinding. A guard that drops normally with a
    /// transaction still open is the wire-server shape -- one guard per
    /// statement, `BEGIN` in one and `COMMIT` in a later one -- and rolling
    /// back there would break it. A thread that dies mid-transaction is the
    /// other case entirely: nobody is coming back for it, and the row it left
    /// staged is one an unrelated connection used to be able to commit.
    ///
    /// One `thread::panicking()` read per guard drop, against the `RwLock`
    /// release the same drop is already paying. Nothing at all in the shipped
    /// binary, which is `panic = "abort"` -- which is why the owner token, not
    /// this, is the fix for that case; this is the second line.
    fn drop(&mut self) {
        if std::thread::panicking() && self.g.txn.as_ref().is_some_and(|t| t.owner == self.g.owner)
        {
            let _ = self.g.rollback();
        }
    }
}

impl std::ops::Deref for Writer<'_> {
    type Target = Session;
    fn deref(&self) -> &Session {
        &self.g
    }
}

impl std::ops::DerefMut for Writer<'_> {
    fn deref_mut(&mut self) -> &mut Session {
        &mut self.g
    }
}

/// A concurrent read handle: `Send + Sync + Clone`, no lifetime.
///
/// Every method takes `&self`, so one `Reader` shared by N threads -- or N
/// clones, which cost an `Arc` bump each -- run their queries at the same
/// time. This is the type the flat 33/32/31 ms scaling curve was about.
#[derive(Clone)]
pub struct Reader {
    inner: Arc<RwLock<Session>>,
    limits: Limits,
}

impl Reader {
    /// Run one read statement.
    pub fn query(&self, sql: &str) -> Result<ResultSet> {
        let ctx = self.limits.context();
        self.with_session(|s| s.read_with(sql, &ctx))
    }

    /// [`Reader::query`] under a caller-supplied context, for a pool that
    /// governs a statement rather than a connection.
    pub fn query_with(&self, sql: &str, ctx: &QueryContext) -> Result<ResultSet> {
        self.with_session(|s| s.read_with(sql, ctx))
    }

    /// Stream a `SELECT` to `sink`, one block at a time. See
    /// [`Session::read_stream`].
    ///
    /// The shared lock is held for the whole stream, so the sink is on the
    /// critical path of every writer: a sink that blocks on a slow socket
    /// blocks the next `INSERT`. [`Reader::cursor`] is the version that does
    /// not, at the cost of a thread.
    pub fn stream(
        &self,
        sql: &str,
        sink: &mut dyn FnMut(StreamItem<'_>) -> Result<()>,
    ) -> Result<usize> {
        let ctx = self.limits.context();
        self.with_session(move |s| s.read_stream(sql, &ctx, sink))
    }

    /// Per-query memory ceiling for this handle only.
    pub fn with_memory_limit(mut self, bytes: i64) -> Reader {
        self.limits.mem = bytes;
        self
    }

    /// Per-statement deadline for this handle only.
    pub fn with_timeout(mut self, d: Duration) -> Reader {
        self.limits.timeout = Some(d);
        self
    }

    /// A private cancel flag for this handle, not shared with the session it
    /// came from. Without this a pool's `KILL QUERY` would stop every
    /// connection at once.
    pub fn with_own_cancel(mut self) -> Reader {
        self.limits.cancel = Arc::new(AtomicBool::new(false));
        self
    }

    /// Set this to stop the queries this handle is running.
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.limits.cancel)
    }

    pub fn resume(&self) {
        self.limits.cancel.store(false, Ordering::Relaxed);
    }

    /// Run `f` under the shared lock, draining buffered writes first if a
    /// writer left any.
    ///
    /// The flush is the one thing a read cannot do for itself, so it is done
    /// here -- once, under the writer lock, and only when
    /// `has_pending_writes` says a scan would otherwise answer short. In the
    /// steady state (an analytic workload, or an OLTP one between writes) the
    /// test is a handful of `usize` compares and the query runs entirely
    /// under the shared lock, which is what lets N of them overlap.
    ///
    /// The slow path flushes **and answers** under the exclusive lock rather
    /// than dropping back to the shared one, and that is not laziness: a
    /// writer that buffers again in the gap would leave the retry facing the
    /// same buffered rows, and reporting that as an error would be a spurious
    /// failure with no action the caller could take. Holding the lock across
    /// both makes progress unconditional. The price is that the first read
    /// after a write serializes with the other readers -- one query, and only
    /// until the delta is empty again.
    ///
    /// A table mid-transaction skips straight to `f`, which refuses: flushing
    /// would push the buffered rows into the writer's private overlay, which
    /// is a mutation of somebody else's transaction to produce an answer that
    /// is going to be refused anyway.
    fn with_session<R>(&self, f: impl FnOnce(&Session) -> Result<R>) -> Result<R> {
        {
            let g = self.inner.read().unwrap_or_else(|e| e.into_inner());
            if !g.catalog.has_pending_writes() || g.catalog.any_in_txn() {
                return f(&g);
            }
        }
        let mut g = self.inner.write().unwrap_or_else(|e| e.into_inner());
        g.catalog.flush_all()?;
        f(&g)
    }

    /// Read rows incrementally, pulling instead of pushing.
    ///
    /// [`Reader::stream`] is the cheaper API and the one to prefer; this is
    /// the shape a wire protocol's portal needs, where the *client* decides
    /// when the next batch is wanted. It costs one thread, which holds the
    /// shared lock until the cursor is drained or dropped -- so a writer on
    /// this thread waits for it. Dropping the cursor cancels the query and
    /// releases the lock, and `Drop` joins, so "drop it before you write" is
    /// the whole rule.
    ///
    /// One sharp edge worth stating: a cursor opened while a writer has rows
    /// buffered takes the *exclusive* lock instead (see
    /// [`Reader::with_session`]) and holds it, so it blocks other readers too
    /// until it is drained. Opening cursors on a quiet database, or after a
    /// `SYSTEM FLUSH`, avoids it entirely.
    pub fn cursor(&self, sql: &str) -> Result<Cursor> {
        Cursor::spawn(Arc::clone(&self.inner), self.limits.clone(), sql.to_string())
    }
}

/// A pull-based result stream. `Iterator<Item = Result<Block>>`.
///
/// One block is in flight at a time: the producer parks on a rendezvous send
/// until the consumer takes the previous one, so a cursor over a billion rows
/// holds two blocks, not a billion rows. That backpressure is the point -- an
/// unbounded channel would move the unbounded buffer rather than remove it.
pub struct Cursor {
    rx: std::sync::mpsc::Receiver<Result<Chunk>>,
    /// Flipped by `Drop`, which is what unblocks a producer parked in `send`
    /// and stops the query it is running mid-block.
    cancel: Arc<AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
    schema: Schema,
    done: bool,
}

/// [`StreamItem`] with the head owned, because it crosses a channel.
enum Chunk {
    Head(Schema),
    Rows(Block),
}

impl Cursor {
    fn spawn(inner: Arc<RwLock<Session>>, limits: Limits, sql: String) -> Result<Cursor> {
        // Its own flag, never the session's: dropping one cursor must not
        // cancel the session's other queries.
        let limits = Limits { cancel: Arc::new(AtomicBool::new(false)), ..limits };
        let cancel = Arc::clone(&limits.cancel);
        // Capacity 1, not 0: `sync_channel(0)` makes every block a full
        // rendezvous, so the producer and the consumer alternate in lockstep
        // and neither ever runs ahead. One slot lets the scan decode block
        // n+1 while the client is still reading block n, which is the whole
        // of the pipelining a portal wants, and costs one block of memory.
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Chunk>>(1);
        let reader = Reader { inner, limits };
        let join = std::thread::Builder::new()
            .name("granular-cursor".into())
            .spawn(move || {
                let ctx = reader.limits.context();
                let out = reader.with_session(|s| {
                    s.read_stream(&sql, &ctx, &mut |item| {
                        let msg = match item {
                            StreamItem::Head(s) => Chunk::Head(s.clone()),
                            StreamItem::Rows(b) => Chunk::Rows(b),
                        };
                        // A closed receiver is a dropped cursor, which is a
                        // cancellation and not an error to report to anyone.
                        tx.send(Ok(msg)).map_err(|_| Error::exec("cursor closed"))
                    })
                });
                if let Err(e) = out {
                    let _ = tx.send(Err(e));
                }
            })
            .map_err(|e| Error::Io(format!("cannot start a cursor thread: {e}")))?;
        // Block for the head, so a cursor either fails to start or knows its
        // shape -- a portal has to answer `Describe` before it fetches, and an
        // API that made the schema arrive with the first row could not answer
        // it at all for an empty result.
        let schema = match rx.recv() {
            Ok(Ok(Chunk::Head(s))) => s,
            Ok(Err(e)) => {
                let _ = join.join();
                return Err(e);
            }
            // Neither can happen -- the producer's first send is the head --
            // but the alternative to handling them is `unwrap` on a channel.
            Ok(Ok(Chunk::Rows(_))) | Err(_) => {
                let _ = join.join();
                return Err(Error::exec("cursor produced no schema"));
            }
        };
        Ok(Cursor { rx, cancel, join: Some(join), schema, done: false })
    }

    /// The result's shape. Known before the first row, always.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}

impl Iterator for Cursor {
    type Item = Result<Block>;

    fn next(&mut self) -> Option<Result<Block>> {
        if self.done {
            return None;
        }
        match self.rx.recv() {
            Ok(Ok(Chunk::Rows(b))) => Some(Ok(b)),
            // The producer reported a failure, or finished and hung up. Either
            // way this cursor is over: a second `recv` would return the same
            // disconnect forever, and an error must be reported once.
            Ok(Err(e)) => {
                self.done = true;
                Some(Err(e))
            }
            Ok(Ok(Chunk::Head(_))) | Err(_) => {
                self.done = true;
                None
            }
        }
    }
}

impl Drop for Cursor {
    fn drop(&mut self) {
        // Order matters: cancel first so a producer *inside* the query stops
        // at its next block boundary, then drop the receiver so one parked in
        // `send` wakes with an error. Joining last is what makes the shared
        // lock provably released when this returns -- otherwise a writer taken
        // on the next line would race a thread still holding it.
        self.cancel.store(true, Ordering::Relaxed);
        let (_, rx) = std::sync::mpsc::sync_channel::<Result<Chunk>>(1);
        drop(std::mem::replace(&mut self.rx, rx));
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

// ------------------------------------------------------- single-writer lock

/// Name of the lock file in the data directory's root.
///
/// Its *contents* are only the holder's pid in ASCII, for the error message.
/// The exclusion is the `flock`, never the file's existence, so a lock file
/// left behind by a killed process is inert: the kernel dropped the lock when
/// the fd closed, and the next opener simply overwrites the stale pid. There
/// is deliberately no "remove the stale lock file" recovery step for a user to
/// get wrong.
pub const LOCK_FILE: &str = "LOCK";

/// `flock(2)`, declared by hand for the same reason as `mmap` in
/// [`crate::persist::mmap`]: the crate has no dependencies, and this is one
/// stable C symbol with a fixed ABI.
///
/// Flag values read off `<sys/fcntl.h>` on macOS (`LOCK_EX` 0x02, `LOCK_NB`
/// 0x04) and `<bits/fcntl-linux.h>` on glibc, which agree -- both inherited
/// them from 4.2BSD, as did every other unix that has `flock`.
#[cfg(unix)]
mod flock_sys {
    use std::ffi::c_int;

    pub const LOCK_SH: c_int = 1;
    pub const LOCK_EX: c_int = 2;
    pub const LOCK_NB: c_int = 4;

    extern "C" {
        pub fn flock(fd: c_int, op: c_int) -> c_int;
    }
}

/// Take `LOCK_EX|LOCK_NB` on an already-open file, reporting only whether it
/// was taken.
///
/// The spill reaper's whole test: a spill directory holds this on its `LOCK`
/// for its lifetime, so success here means the owner's descriptor is closed
/// and the tree is orphaned. Non-unix returns `false`, so the reaper there
/// deletes nothing rather than deleting blind -- the same shape the non-unix
/// `lock_data_dir` already takes.
#[cfg(unix)]
pub(crate) fn try_lock_exclusive(f: &File) -> bool {
    use std::os::unix::io::AsRawFd;
    // SAFETY: `f` owns a valid open descriptor for the whole call, and `flock`
    // only inspects the descriptor -- it neither retains it nor touches user
    // memory.
    unsafe { flock_sys::flock(f.as_raw_fd(), flock_sys::LOCK_EX | flock_sys::LOCK_NB) == 0 }
}

#[cfg(not(unix))]
pub(crate) fn try_lock_exclusive(_f: &File) -> bool {
    false
}

/// Which claim a session makes on the data directory.
///
/// `Shared` is what makes [`Session::open_read_only`] plural: `flock` lets any
/// number of `LOCK_SH` holders coexist and excludes them all from a `LOCK_EX`
/// holder, which is precisely the single-writer / many-reader rule this phase
/// is about -- enforced by the kernel, across processes, for free.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LockMode {
    Shared,
    Exclusive,
}

impl LockMode {
    #[cfg(unix)]
    fn op(self) -> std::ffi::c_int {
        match self {
            LockMode::Shared => flock_sys::LOCK_SH,
            LockMode::Exclusive => flock_sys::LOCK_EX,
        }
    }

    fn why(self) -> &'static str {
        match self {
            LockMode::Shared => {
                "another granular process has it open for writing. A read-only session \
                 shares the directory with other readers, but not with a writer"
            }
            LockMode::Exclusive => {
                "Only one process may have a data directory open for writing at a time -- \
                 concurrent writers allocate colliding part file names and overwrite each \
                 other's committed data. Close the other process, or point this one at a \
                 different --data directory"
            }
        }
    }
}

/// Claim `root` for this process, or fail naming the process that holds it.
///
/// Two writers on one directory do not merely interleave badly, they destroy
/// committed data: each allocates part sequence numbers from its own snapshot
/// of the table directory, so both pick the *same* `part_NNNNNN.gpart` name
/// and `rename` over one another, and then the last `CATALOG` write commits a
/// part list describing files the other process wrote. Measured on this
/// engine before the lock existed: eight concurrent one-`INSERT` processes,
/// every one of which returned success and fsynced, left 7 / 0 / 4 of the 8
/// rows across three runs, and one run left the table unreadable. The lock is
/// what makes that acknowledgement mean anything.
///
/// `flock` rather than an `O_EXCL` pid file because the kernel releases it
/// when the descriptor closes -- including on `SIGKILL`, on a panic, and on a
/// process that forgot to clean up -- so a crash can never leave a directory
/// that needs manual unlocking. `LOCK_NB` because blocking here would turn a
/// misconfiguration into a hang.
///
/// Cost is one `open` and one `flock` per `Session::open`, both off any hot
/// path.
#[cfg(unix)]
fn lock_data_dir(root: &Path, mode: LockMode) -> Result<Option<File>> {
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::os::unix::io::AsRawFd;

    let path = root.join(LOCK_FILE);
    let shared = mode == LockMode::Shared;
    let open = |write: bool| {
        std::fs::OpenOptions::new()
            .read(true)
            .write(write)
            .create(write)
            // Emphatically not `truncate`: the file is opened before the lock
            // is held, and erasing the incumbent's pid would destroy the only
            // thing that makes the failure diagnosable.
            .truncate(false)
            .open(&path)
    };
    // The read-only-media fallback below fires for *this* and nothing else.
    // "Any error at all" was too wide: a transient `EMFILE` or a bad path on a
    // perfectly writable directory then produced a session holding no lock,
    // which is exactly the race the lock exists to prevent -- reported as
    // success.
    const EROFS: i32 = 30;
    let unwritable = |e: &std::io::Error| {
        e.kind() == std::io::ErrorKind::PermissionDenied || e.raw_os_error() == Some(EROFS)
    };
    let failed = |e: &std::io::Error| {
        Error::Io(format!("cannot open lock file {}: {e}", path.display()))
    };
    let mut f = match open(true) {
        Ok(f) => f,
        // A forensic copy on read-only media has no writable `LOCK`, and a
        // shared holder never writes to one anyway -- it does not stamp its
        // pid (see below), it only needs a descriptor to `flock`. So fall
        // back to a read-only open, and then to no lock at all: a directory
        // nobody can write cannot have a writer to exclude, which is the
        // entire thing the lock is for.
        Err(ref e) if shared && unwritable(e) => match open(false) {
            Ok(f) => f,
            Err(ref e) if unwritable(e) => return Ok(None),
            Err(e) => return Err(failed(&e)),
        },
        Err(e) => return Err(failed(&e)),
    };

    // SAFETY: `f` owns a valid open descriptor for the whole call, and `flock`
    // only inspects the descriptor -- it neither retains it nor touches user
    // memory.
    let rc = unsafe { flock_sys::flock(f.as_raw_fd(), mode.op() | flock_sys::LOCK_NB) };
    if rc != 0 {
        // EWOULDBLOCK is the expected case (someone holds it); anything else
        // (EOPNOTSUPP on an exotic filesystem, EBADF) is reported the same
        // way, because opening the directory anyway would be the bug.
        let os = std::io::Error::last_os_error();
        let mut holder = String::new();
        let _ = f.read_to_string(&mut holder);
        let holder = holder.trim();
        let who = if holder.is_empty() {
            String::new()
        } else {
            format!(" (pid {holder})")
        };
        return Err(Error::storage(format!(
            "data directory `{}` is already open by another granular process{who}: {os}. {}.",
            root.display(),
            mode.why()
        )));
    }

    // A shared holder must not stamp its pid over the file: several of them
    // hold the lock at once, so the last writer would win a race nobody
    // arbitrates, and the pid is only there to name the *exclusive* holder in
    // the message above.
    if shared {
        return Ok(Some(f));
    }

    // Only now, holding the lock, is it ours to write to.
    let stamp = |f: &mut File| -> std::io::Result<()> {
        f.set_len(0)?;
        f.seek(SeekFrom::Start(0))?;
        f.write_all(format!("{}\n", std::process::id()).as_bytes())?;
        f.flush()
    };
    if let Err(e) = stamp(&mut f) {
        // The lock is held and that is what matters; the pid is a comment. But
        // a failure to write it means the filesystem is read-only or full, and
        // a database that cannot write is better refused here than at the
        // first INSERT.
        return Err(Error::Io(format!(
            "cannot record this process in lock file {}: {e}",
            path.display()
        )));
    }
    Ok(Some(f))
}

/// No `flock` off unix, and no dependency budget for the `LockFileEx` dance,
/// so a non-unix build is documented as unlocked rather than pretending. This
/// is a real limitation, not an oversight: see the unix arm for what the lock
/// prevents.
#[cfg(not(unix))]
fn lock_data_dir(_root: &Path, _mode: LockMode) -> Result<Option<File>> {
    Ok(None)
}

// ----------------------------------------------------------------- mutations

/// A bulk delete, lowered out of its plan and detached from the catalog.
///
/// Owned rather than borrowed. Applying it needs `&mut Catalog` to reach the
/// table, and the plan it came from borrows the catalog through the binder --
/// so carrying the plan would deadlock the borrow checker for the sake of
/// three small `Vec`s built once per statement and none per row.
///
/// This is `ScanNode`'s two index spaces, kept honest: `projection` holds
/// *table* column indices, while `pred` and `zone` are expressed against the
/// projected schema. `Table::delete_where` maps between them exactly as
/// `operators::scan::Scan` does.
struct Sweep {
    projection: Vec<usize>,
    /// `None` when the optimizer folded the whole predicate away: every live
    /// row matches, which is what `DELETE FROM t` with no `WHERE` means.
    pred: Option<BoundExpr>,
    zone: Vec<ZoneFilter>,
}

impl Sweep {
    /// Peel the optimized source of a `MutationKind::Delete` plan.
    ///
    /// `optimize` sinks the mutation's `WHERE` into the scan as PREWHERE and
    /// derives its zone filters, so the shape is a bare `Scan`. A conjunct a
    /// future pass declines to push would stay as a `Filter` above it: it is
    /// ANDed back on rather than dropped, because dropping one deletes rows
    /// the statement did not name. Anything else is a plan `bind_delete` did
    /// not build, and is refused rather than swept.
    fn of(plan: LogicalPlan) -> Result<Option<Sweep>> {
        let (node, above) = match plan {
            LogicalPlan::Scan(n) => (n, None),
            LogicalPlan::Filter { input, predicate } => match *input {
                LogicalPlan::Scan(n) => (n, Some(predicate)),
                other => return Err(not_a_sweep(&other)),
            },
            // Constantly FALSE -- or constantly UNKNOWN, which a filter treats
            // the same way. No row matches.
            LogicalPlan::Empty { .. } => return Ok(None),
            other => return Err(not_a_sweep(&other)),
        };
        let mut node = *node;
        let mut conjuncts = std::mem::take(&mut node.filters);
        conjuncts.extend(above);
        Ok(Some(Sweep {
            projection: node.projection,
            pred: BoundExpr::join_conjuncts(conjuncts),
            zone: node.zone_filters,
        }))
    }
}

/// One refusal for every way a connection can reach another connection's
/// transaction: ending it (`COMMIT`/`ROLLBACK`), nesting inside it (`BEGIN`),
/// or quietly enlisting in it (any other statement).
#[cold]
fn foreign_txn(what: &str) -> Error {
    Error::unsupported(format!(
        "{what} refused: a transaction opened by another connection to this database is \
         still in progress, and this connection has none of its own. Running {what} here \
         would publish, discard or read writes this connection did not author -- the same \
         failure a nested BEGIN is refused for, one scope out. This engine has a single \
         writer, so wait for the other connection to COMMIT or ROLLBACK"
    ))
}

/// A transaction was about to acquire a second commit point. See
/// [`Session::mark_fold`] for why that cannot be ordered away.
///
/// `folding` is the table whose sweep has no durable row identity; `other` is
/// whatever else the transaction already holds. Both are named because the
/// user's fix depends on which is which, and both workarounds are given
/// because "unsupported" without one is just a wall.
#[cold]
fn two_commit_points(folding: &str, other: &str) -> Error {
    Error::unsupported(format!(
        "refusing to put this DELETE/UPDATE on `{folding}` in the same transaction as \
         `{other}`: `{folding}` has no single-column PRIMARY KEY, and this statement hid \
         rows that are not yet in any part file, so nothing in the log can name them. That \
         table can only be made durable by writing its parts out, which happens after \
         COMMIT has already made `{other}` durable -- so a crash between the two would \
         commit one and lose the other. Either give `{folding}` a single-column PRIMARY \
         KEY, or run this statement in a transaction of its own (a single table folding on \
         its own is atomic)"
    ))
}

#[cold]
fn poisoned_err(why: &str) -> Error {
    Error::exec(format!(
        "the transaction cannot continue: an earlier statement in it failed ({why}). \
         ROLLBACK to start again"
    ))
}

#[cold]
fn not_a_sweep(p: &LogicalPlan) -> Error {
    let root = p.explain();
    Error::exec(format!(
        "a mutation's row set lowered to `{}`, which names no single table to \
         delete positions from",
        root.lines().next().unwrap_or("?").trim()
    ))
}

// ------------------------------------------------------ transaction control

/// True when `sql` contains a word that could begin a transaction-control
/// statement.
///
/// The gate on [`Session::run_mixed`], and the reason autocommit pays nothing
/// for transactions existing: one pass with no allocation that answers `false`
/// for every statement in this engine's vocabulary, after which the text goes
/// to `parse` exactly as before. A false positive -- `INSERT ... VALUES
/// ('commit')` -- only costs one extra lex, never a wrong answer, because the
/// split that follows is done by the real tokenizer.
///
/// The first letter selects *which* keyword to compare, rather than gating a
/// run of four comparisons: `SELECT`, `count`, `sum`, `BY` and `ORDER` all
/// begin with one of `b`/`c`/`r`/`s`, so the letter test fires on roughly half
/// the words in an ordinary query and what happens next is most of the cost.
///
/// No word-boundary tracking either. It would tighten the answer -- `x_commit`
/// currently trips this -- but a false positive costs one extra lex and
/// nothing else, while the bookkeeping costs two operations on every byte.
///
/// Measured on the 76-byte query the differential harness generates most of
/// (best-of-9, alternating): four comparisons per candidate with word-boundary
/// tracking 70.0 ns, one comparison with tracking 60.3 ns, one comparison
/// without 54.6 ns. Through `Session::run` all three measure 0.988-1.005
/// against the pre-change body -- the statement itself is 48 us, so this is
/// 0.1% either way and the reason to pick the last one is that it is also the
/// smallest.
fn mentions_txn_keyword(sql: &str) -> bool {
    #[inline]
    fn starts(hay: &[u8], kw: &[u8]) -> bool {
        hay.len() >= kw.len() && hay[..kw.len()].eq_ignore_ascii_case(kw)
    }
    let b = sql.as_bytes();
    for (i, &c) in b.iter().enumerate() {
        let hit = match c | 0x20 {
            b'b' => starts(&b[i..], b"begin"),
            b'c' => starts(&b[i..], b"commit"),
            b'r' => starts(&b[i..], b"rollback"),
            b's' => starts(&b[i..], b"start"),
            _ => false,
        };
        if hit {
            return true;
        }
    }
    false
}

/// Classify one statement's tokens as transaction control, or `None` to let
/// the parser have it.
///
/// Deliberately exact: only the complete standard spellings are claimed, so
/// anything else beginning with one of these words still reaches the parser
/// and still gets the parser's error message rather than a confusing one from
/// here. None of the four can begin a statement in this dialect, so nothing
/// legal is being shadowed.
fn txn_stmt(span: &[crate::sql::lexer::Spanned]) -> Option<TxnStmt> {
    let head = span[0].tok.bare_word()?;
    // `BEGIN`/`COMMIT`/`ROLLBACK` optionally followed by the noise word;
    // `START` only in `START TRANSACTION`.
    let noise = |rest: &[crate::sql::lexer::Spanned]| match rest {
        [] => true,
        [t] => t.tok.is_keyword("work") || t.tok.is_keyword("transaction"),
        _ => false,
    };
    let rest = &span[1..];
    if head.eq_ignore_ascii_case("start") {
        return match rest {
            [t] if t.tok.is_keyword("transaction") => Some(TxnStmt::Begin),
            _ => None,
        };
    }
    if !noise(rest) {
        return None;
    }
    if head.eq_ignore_ascii_case("begin") {
        Some(TxnStmt::Begin)
    } else if head.eq_ignore_ascii_case("commit") {
        Some(TxnStmt::Commit)
    } else if head.eq_ignore_ascii_case("rollback") {
        Some(TxnStmt::Rollback)
    } else {
        None
    }
}

/// A one-row result: a name, then a row of counts. What every operator
/// statement in this file reports, so they cannot describe themselves in three
/// different shapes.
fn report(cols: &[&str], name: Value, counts: &[u64]) -> Result<ResultSet> {
    let schema = Schema::new(
        std::iter::once(Field::new(cols[0], DataType::String))
            .chain(cols[1..].iter().map(|c| Field::new(*c, DataType::UInt64)))
            .collect(),
    )?;
    let row = std::iter::once(name).chain(counts.iter().map(|&n| Value::UInt(n))).collect();
    ResultSet::from_rows(schema, vec![row])
}

/// Whether two paths name the same directory, or one contains the other.
///
/// Containment and not just equality, because `TO '<open db>/restored'` is the
/// same mistake spelled one level down: it leaves a second `CATALOG` and a
/// second set of part directories inside the tree the loader walks, which is
/// the state the "a restore never merges into a live database" rule exists to
/// prevent. Checked on a live root: the scan in `unaccounted_table_dirs` looks
/// one level too shallow to notice, so nothing else would have, and the
/// operator would find out the next time somebody copied the data directory.
///
/// `None` when the two are disjoint. Otherwise `Some(true)` for the same
/// directory and `Some(false)` for one inside the other -- the refusal reads
/// differently for each, and answering here is what keeps it from resolving
/// both paths a second time to find out.
fn overlaps(a: &Path, b: &Path) -> Option<bool> {
    let (a, b) = (resolved(a), resolved(b));
    (a.starts_with(&b) || b.starts_with(&a)).then(|| a == b)
}

/// `p` with its deepest *existing* ancestor canonicalized.
///
/// Plain `canonicalize` is not enough here: a restore target has not been
/// created yet, which is precisely the input it refuses -- and comparing the
/// raw text instead would let `../db` or a symlinked parent walk straight past
/// the refusal above.
fn resolved(p: &Path) -> PathBuf {
    let abs;
    let mut p = p;
    if p.is_relative() {
        if let Ok(cwd) = std::env::current_dir() {
            abs = cwd.join(p);
            p = &abs;
        }
    }
    let mut cur = p;
    loop {
        if let Ok(c) = std::fs::canonicalize(cur) {
            return p.strip_prefix(cur).map_or(c.clone(), |tail| c.join(tail));
        }
        match cur.parent() {
            Some(up) if up != cur => cur = up,
            _ => return p.to_path_buf(),
        }
    }
}

/// [`mentions_txn_keyword`] for the three operator statements, and with the
/// same reasoning about false positives: `SELECT 'backup'` costs one extra
/// lex, `admin_stmt` then declines it, and the parser gets it unchanged.
fn mentions_admin_keyword(sql: &str) -> bool {
    #[inline]
    fn starts(hay: &[u8], kw: &[u8]) -> bool {
        hay.len() >= kw.len() && hay[..kw.len()].eq_ignore_ascii_case(kw)
    }
    let b = sql.as_bytes();
    for (i, &c) in b.iter().enumerate() {
        let hit = match c | 0x20 {
            b'b' => starts(&b[i..], b"backup"),
            b'r' => starts(&b[i..], b"restore"),
            b'v' => starts(&b[i..], b"verify"),
            _ => false,
        };
        if hit {
            return true;
        }
    }
    false
}

/// Classify one statement's tokens as an operator statement.
///
/// `None` hands it to the parser untouched. `Some(Err(..))` is for a head word
/// that is unambiguously ours with a tail that is not: none of these three can
/// begin a statement in this dialect, so "BACKUP" followed by anything else is
/// a mistyped backup and deserves to be told what the syntax is, rather than
/// the parser's "unexpected token" on a word it has never heard of.
fn admin_stmt(span: &[crate::sql::lexer::Spanned]) -> Option<Result<Admin>> {
    use crate::sql::lexer::Token;
    use std::borrow::Cow;
    let head = span[0].tok.bare_word()?;
    let text = |i: usize| match span.get(i).map(|s| &s.tok) {
        Some(Token::Str(s)) => Some(s.clone()),
        _ => None,
    };
    let kw = |i: usize, w: &str| span.get(i).is_some_and(|s| s.tok.is_keyword(w));
    let usage = |what: &str, form: &str| {
        Some(Err(Error::parse(format!("{what} takes the form `{form}`"), span[0].pos)))
    };

    if head.eq_ignore_ascii_case("backup") {
        let Some(to) = kw(1, "to").then(|| text(2)).flatten() else {
            return usage("BACKUP", "BACKUP TO '<archive>' [INCREMENTAL FROM '<base>']");
        };
        return match span.len() {
            3 => Some(Ok(Admin::Backup { to, base: None })),
            6 if kw(3, "incremental") && kw(4, "from") => match text(5) {
                Some(base) => Some(Ok(Admin::Backup { to, base: Some(base) })),
                None => usage("BACKUP", "BACKUP TO '<archive>' INCREMENTAL FROM '<base>'"),
            },
            _ => usage("BACKUP", "BACKUP TO '<archive>' [INCREMENTAL FROM '<base>']"),
        };
    }
    if head.eq_ignore_ascii_case("restore") {
        const FORM: &str = "RESTORE FROM '<archive>' [TO '<directory>'] \
                            [UNTIL LSN <n> | UNTIL TIMESTAMP '<ts>' | UNTIL LATEST]";
        let Some(from) = kw(1, "from").then(|| text(2)).flatten() else {
            return usage("RESTORE", FORM);
        };
        // `TO` stays optional in the grammar so that omitting it is answered by
        // `run_restore`'s sentence about why the target is never the open
        // database, rather than by a form line that does not say why.
        let (to, rest) = match (kw(3, "to"), text(4)) {
            (true, Some(to)) => (Some(to), &span[5..]),
            (true, None) => return usage("RESTORE", FORM),
            _ => (None, &span[3..]),
        };
        let until = match rest {
            [] => None,
            [u, tail @ ..] if u.tok.is_keyword("until") => {
                let Some(kind) = tail.first().and_then(|s| s.tok.bare_word()) else {
                    return usage("RESTORE", FORM);
                };
                // The value goes to `backup::parse_target` as text rather than
                // being decoded here: the grammar of a recovery target belongs
                // to the module that acts on one, so the statement and the
                // machinery cannot come to disagree about what a timestamp
                // means. `LATEST` carries no value, hence the empty borrow.
                //
                // A bare word is handed over too, rather than refused as the
                // wrong token: `UNTIL LSN soon` deserves the sentence naming
                // where a recovery LSN comes from, not a form line -- and this
                // clause is only ever typed by someone whose day has already
                // gone wrong.
                let value = match &tail[1..] {
                    [] => Cow::Borrowed(""),
                    [v] => match &v.tok {
                        Token::Str(s) | Token::Word { value: s, .. } => Cow::Borrowed(s.as_str()),
                        // `UNTIL LSN 5` lexes as a number, and `render_plain`
                        // is the spelling that round-trips it -- `Display`
                        // would quote a string, which this arm never sees.
                        Token::Number(n) => Cow::Owned(n.render_plain()),
                        _ => return usage("RESTORE", FORM),
                    },
                    _ => return usage("RESTORE", FORM),
                };
                match crate::backup::parse_target(kind, &value) {
                    Ok(t) => Some(t),
                    Err(e) => return Some(Err(e)),
                }
            }
            _ => return usage("RESTORE", FORM),
        };
        return Some(Ok(Admin::Restore { from, to, until }));
    }
    if head.eq_ignore_ascii_case("verify") {
        // Only `VERIFY BACKUP`, so the word stays available for whatever a
        // later wave wants to verify.
        if !kw(1, "backup") {
            return None;
        }
        return match (span.len(), text(2)) {
            (3, Some(archive)) => Some(Ok(Admin::Verify { archive })),
            _ => usage("VERIFY BACKUP", "VERIFY BACKUP '<archive>'"),
        };
    }
    None
}

/// The text of each statement in `sql`, when there is more than one.
///
/// [`parse`] returns statements with no spans, and the query log wants the
/// text that was typed rather than a reconstruction from the AST. Only a batch
/// pays the second lex; a single statement *is* the whole input, which is what
/// the CLI and every `query`/`execute` call submit. A split that disagrees
/// with the parse gives up rather than pairing texts with the wrong
/// statements.
fn statement_texts(sql: &str, n: usize) -> Option<Vec<&str>> {
    use crate::sql::lexer::{tokenize, Token};
    if n < 2 {
        return None;
    }
    let toks = tokenize(sql).ok()?;
    let mut out = Vec::with_capacity(n);
    let mut start = 0usize;
    for i in 0..=toks.len() {
        if i != toks.len() && toks[i].tok != Token::Semicolon {
            continue;
        }
        let span = &toks[start..i];
        start = i + 1;
        if span.is_empty() {
            continue;
        }
        let end = if i == toks.len() { sql.len() } else { toks[i].pos };
        out.push(sql[span[0].pos..end].trim());
    }
    (out.len() == n).then_some(out)
}

/// What *holding* `b` costs this process, in bytes.
///
/// Deliberately not [`Block::bytes`], on two counts. It walks every string in
/// the block to add its body's length: 1332 ns per 8192-row string block
/// against 3 ns here (measured, best of 7 over 25 blocks of a 200k-row
/// result), which is per-row work on a path that must not have any. And it
/// charges that body once *per row*, while a scan's strings are `Arc<str>`
/// clones of the part dictionary's -- one body, N pointers -- so it reported
/// 7 MiB for a result whose real cost was 4.
///
/// The pointer is what the result owns, so the pointer is what it is charged.
/// A string column *computed* by an expression does own its bodies and is
/// undercharged by their length; the ceiling this defends is 8 GiB and the
/// thing it is defending against is a result of unbounded *blocks*, which
/// 24 bytes per row per column bounds exactly.
fn retained_bytes(b: &Block) -> usize {
    b.columns
        .iter()
        .map(|c| {
            let per = match &c.data {
                crate::types::ColumnData::Str(_) => std::mem::size_of::<std::sync::Arc<str>>(),
                _ => 8,
            };
            c.data.len() * per + c.nulls.as_ref().map_or(0, |n| n.bytes())
        })
        .sum()
}

// ------------------------------------------------ is there anything to fold?
//
// The mirror of `Session::rewrite_*`, over a borrowed AST and answering one
// bit instead of rewriting. It exists so the common statement -- no subquery
// anywhere in it -- plans without cloning itself first.
//
// **Every one of these is conservative in the same direction**: an arm this
// code has not been taught about answers `true`, which costs a clone and takes
// the path that was there before. So a variant added to the AST cannot make a
// subquery go unfolded; the worst it can do is lose an optimization. That is
// the only reading under which a second traversal is safe to keep beside the
// first, and it is why the catch-alls are wildcards rather than exhaustive
// matches the compiler would force somebody to extend.

fn has_subquery(q: &crate::sql::ast::Query, n: Names<'_>) -> bool {
    q.with.iter().any(|c| has_subquery(&c.query, n))
        || set_has_subquery(&q.body, n)
        || q.order_by.iter().any(|o| expr_has_subquery(&o.expr))
        || q.limit.iter().chain(q.offset.iter()).any(expr_has_subquery)
        || q.limit_by.as_ref().is_some_and(|(n, keys)| {
            expr_has_subquery(n) || keys.iter().any(expr_has_subquery)
        })
}

fn set_has_subquery(s: &crate::sql::ast::SetExpr, n: Names<'_>) -> bool {
    use crate::sql::ast::{SelectItem, SetExpr};
    match s {
        SetExpr::Select(sel) => {
            sel.projection.iter().any(|i| match i {
                SelectItem::Expr { expr, .. } => expr_has_subquery(expr),
                _ => false,
            }) || sel.from.as_ref().is_some_and(|f| tableref_has_subquery(f, n))
                || sel
                    .prewhere
                    .iter()
                    .chain(sel.selection.iter())
                    .chain(sel.having.iter())
                    .any(expr_has_subquery)
                || sel.group_by.iter().any(expr_has_subquery)
        }
        SetExpr::Query(q) => has_subquery(q, n),
        SetExpr::SetOperation { left, right, .. } => {
            set_has_subquery(left, n) || set_has_subquery(right, n)
        }
        SetExpr::Values(rows) => rows.iter().flatten().any(expr_has_subquery),
    }
}

fn tableref_has_subquery(t: &crate::sql::ast::TableRef, n: Names<'_>) -> bool {
    use crate::sql::ast::{JoinConstraint, TableRef};
    match t {
        // Not a subquery, but the other thing `rewrite_tableref` turns into
        // one, and it needs the same owned AST. Deliberately only the two
        // qualifier compares here and not the full `system::classify`: a
        // reference to a *real* table in a database called `system` answers
        // yes and costs a clone the rewrite then declines to use, which is
        // exactly the trade the rest of this walk already makes.
        TableRef::Table { name, .. } => {
            name.qualifier().is_some_and(|q| {
                q.eq_ignore_ascii_case(crate::system::SYSTEM_DB)
                    || q.eq_ignore_ascii_case(crate::system::INFO_SCHEMA_DB)
            })
            // The third thing `rewrite_tableref` turns into a subquery. This
            // one *is* an exact test rather than a cheap over-approximation,
            // because a view's name has no distinguishing qualifier -- and it
            // costs nothing on a database with no views, where the map is
            // empty and `Extensions::view` returns before hashing anything.
            || n.ext.view(name, n.db).is_some()
        }
        // A derived table is bound, not folded -- but one nested *inside* it
        // is folded, so the walk goes through.
        TableRef::Subquery { query, .. } => has_subquery(query, n),
        TableRef::Join { left, right, constraint, .. } => {
            tableref_has_subquery(left, n)
                || tableref_has_subquery(right, n)
                || matches!(constraint, JoinConstraint::On(e) if expr_has_subquery(e))
        }
    }
}

fn expr_has_subquery(e: &crate::sql::ast::Expr) -> bool {
    use crate::sql::ast::Expr;
    match e {
        Expr::Subquery(_) | Expr::InSubquery { .. } | Expr::Exists { .. } => true,
        Expr::Literal(_) | Expr::Column(_) | Expr::Wildcard => false,
        Expr::UnaryOp { expr, .. }
        | Expr::Cast { expr, .. }
        | Expr::IsNull { expr, .. }
        | Expr::Interval { value: expr, .. } => expr_has_subquery(expr),
        Expr::BinaryOp { left, right, .. } => {
            expr_has_subquery(left) || expr_has_subquery(right)
        }
        Expr::Function { args, params, .. } => {
            args.iter().chain(params.iter()).any(expr_has_subquery)
        }
        Expr::Window { args, params, spec, .. } => {
            args.iter().chain(params.iter()).any(expr_has_subquery)
                || spec.partition_by.iter().any(expr_has_subquery)
                || spec.order_by.iter().any(|o| expr_has_subquery(&o.expr))
        }
        Expr::Case { operand, when_then, else_result } => {
            operand.iter().any(|o| expr_has_subquery(o))
                || when_then
                    .iter()
                    .any(|(w, t)| expr_has_subquery(w) || expr_has_subquery(t))
                || else_result.iter().any(|x| expr_has_subquery(x))
        }
        Expr::InList { expr, list, .. } => {
            expr_has_subquery(expr) || list.iter().any(expr_has_subquery)
        }
        Expr::Between { expr, low, high, .. } => {
            expr_has_subquery(expr) || expr_has_subquery(low) || expr_has_subquery(high)
        }
        Expr::Like { expr, pattern, .. } => {
            expr_has_subquery(expr) || expr_has_subquery(pattern)
        }
        Expr::Tuple(items) => items.iter().any(expr_has_subquery),
        // See the note above: unknown means "assume yes".
        #[allow(unreachable_patterns)]
        _ => true,
    }
}

/// Statements that change the catalog's shape rather than a table's contents.
///
/// Every one of them checkpoints on the way out, which is what makes the
/// constraint and view metadata durable at the same instant as the table it
/// describes -- see the `Extensions` note. A new DDL statement that is missing
/// from this list would leave its metadata row in memory only.
fn is_ddl(s: &Statement) -> bool {
    matches!(
        s,
        Statement::CreateTable(_)
            | Statement::CreateDatabase { .. }
            | Statement::CreateView { .. }
            | Statement::DropView { .. }
            | Statement::DropTable { .. }
            | Statement::DropDatabase { .. }
            | Statement::AlterAddColumn { .. }
            | Statement::AlterDropColumn { .. }
            | Statement::AlterModifyColumn { .. }
            | Statement::RenameTable { .. }
            | Statement::Truncate { .. }
    )
}

/// Statements a `&self` session may run: they read the catalog and the parts,
/// and touch nothing else.
///
/// `EXPLAIN` is in, including `EXPLAIN ANALYZE`, because the statement it
/// wraps is only ever *planned* here -- and ANALYZE runs it exactly as a
/// `SELECT` runs, through the same builder. `EXPLAIN` over a mutation binds
/// the plan that would find the rows and stops there. `USE` is out: it moves
/// the session's current database, which is state, however small.
///
/// The one that has to stay out and looks like it belongs: `SYSTEM FLUSH`. It
/// is spelled like a read and it drains every write buffer.
fn is_read(s: &Statement) -> bool {
    matches!(
        s,
        Statement::Query(_)
            | Statement::ShowDatabases
            | Statement::ShowTables { .. }
            | Statement::ShowCreateTable(_)
            | Statement::Describe(_)
            | Statement::Explain { .. }
    )
}

/// Read statements that touch a table's *rows* rather than only its metadata.
///
/// `EXPLAIN` is in even where it does not execute: `PIPELINE` reads granule
/// counts off the pinned snapshot to decide the exchange width, so a plan
/// rendered over an unflushed delta would describe a fleet the real query
/// would not get.
fn reads_rows(s: &Statement) -> bool {
    matches!(s, Statement::Query(_) | Statement::Explain { .. })
}

/// The statement's name, for an error that has to say what it refused.
///
/// A `&'static str` rather than `format!("{stmt:?}")`: the refusal messages
/// are the only caller, and dumping an entire AST into one is how a rejected
/// `INSERT ... SELECT` produces a megabyte of error text.
fn stmt_kind(s: &Statement) -> &'static str {
    match s {
        Statement::Query(_) => "SELECT",
        Statement::Insert(_) => "INSERT",
        Statement::CreateTable(_) => "CREATE TABLE",
        Statement::CreateDatabase { .. } => "CREATE DATABASE",
        Statement::DropTable { .. } => "DROP TABLE",
        Statement::DropDatabase { .. } => "DROP DATABASE",
        Statement::Use(_) => "USE",
        Statement::Optimize { .. } => "OPTIMIZE",
        Statement::Truncate { .. } => "TRUNCATE",
        Statement::SystemFlush(_) => "SYSTEM FLUSH",
        Statement::AlterDelete { .. } => "ALTER TABLE ... DELETE",
        Statement::AlterUpdate { .. } => "ALTER TABLE ... UPDATE",
        Statement::AlterAddColumn { .. } => "ALTER TABLE ... ADD COLUMN",
        Statement::AlterDropColumn { .. } => "ALTER TABLE ... DROP COLUMN",
        Statement::AlterModifyColumn { .. } => "ALTER TABLE ... MODIFY COLUMN",
        Statement::RenameTable { .. } => "RENAME TABLE",
        Statement::CreateView { .. } => "CREATE VIEW",
        Statement::DropView { .. } => "DROP VIEW",
        Statement::ShowDatabases => "SHOW DATABASES",
        Statement::ShowTables { .. } => "SHOW TABLES",
        Statement::ShowCreateTable(_) => "SHOW CREATE TABLE",
        Statement::Describe(_) => "DESCRIBE",
        Statement::Explain { .. } => "EXPLAIN",
    }
}

/// What a `&self` read says when a table still holds buffered rows.
///
/// It is an error and not a shrug because the alternative is the defect this
/// engine keeps finding: an answer that is short by however many rows happened
/// to be in the delta, returned as if it were complete.
const PENDING_WRITES: &str =
    "this session has buffered writes that a scan cannot see, and a read-only path \
     cannot flush them. Run the query through `&mut Session`, or use a `Reader` from \
     `Db`, which takes the writer lock for one flush and retries";

#[cold]
fn read_only_err(what: &str) -> Error {
    Error::unsupported(format!(
        "{what} is not allowed on a read-only session: it was opened with \
         `open_read_only`, which takes a shared directory lock that several \
         processes hold at once precisely because none of them writes"
    ))
}


// ------------------------------------------ constraints and views: helpers

/// Accept `UNIQUE` only where this engine can actually enforce it, and say
/// exactly what is missing where it cannot.
///
/// ## What is enforced, and by what
///
/// A `UNIQUE` column is enforced **iff it is the table's unique key** --
/// [`TableDef::pk_col`], i.e. a single-column `PRIMARY KEY` (or a
/// `ReplacingMergeTree` sort key) whose lane is order-preserving. That is not
/// a coincidence of implementation: that column, and only that column, has the
/// MPH index and the keyed delta behind it, so "does this value already exist"
/// is one probe rather than a scan. `Table::insert_with(KeyConflict::Reject)`
/// then asks it of the whole batch, against the batch itself and against every
/// live row, and refuses the statement if the answer is yes.
///
/// What that changes is the *meaning* of an insert on a keyed table, which is
/// why it is opt-in: without `UNIQUE`, a repeated primary key is
/// last-write-wins (an upsert -- the OLTP path this engine is built around).
/// With it, a repeated key is an error.
///
/// ## What is refused, and why not "accept and scan"
///
/// `UNIQUE` on any other column is refused at DDL time. Enforcing it would
/// mean a full scan of the column per insert -- turning a 33 ns/row write into
/// a table scan -- or a second index this engine does not have. Both are real
/// answers; neither is *this* answer, and the one thing that must not happen
/// is accepting the declaration and enforcing nothing, which is precisely the
/// silent-acceptance defect four waves of work have been removing. The error
/// says which column would have to become the key.
fn check_unique_declarations(cols: &[ColumnDef], def: &TableDef) -> Result<()> {
    for c in cols.iter().filter(|c| c.unique) {
        let idx = def.schema.require(&c.name)?;
        if def.pk_col() == Some(idx) {
            continue;
        }
        let why = if def.primary_key.is_empty() {
            format!(
                "`{}` has no PRIMARY KEY. Declare `PRIMARY KEY ({0})` (and put it first \
                 in ORDER BY) and the constraint is enforced by the key index",
                c.name
            )
        } else if !def.primary_key.contains(&idx) {
            format!(
                "the table's key is `{}`, not `{}`. A UNIQUE constraint is enforced here \
                 only on the unique key itself, which is the one column with an index \
                 behind it",
                def.schema.name(def.primary_key[0]),
                c.name
            )
        } else {
            format!(
                "`{}` is part of a key this engine cannot index on its own -- a unique \
                 key must be a single non-nullable, non-string column that leads ORDER BY",
                c.name
            )
        };
        return Err(Error::unsupported(format!(
            "UNIQUE on `{}`: not enforceable, so it is refused rather than accepted and \
             ignored. {why}. Every other column would need a scan per insert or a \
             secondary index, and neither exists here",
            c.name
        )));
    }
    Ok(())
}

/// `db.name` for a possibly-bare object name, against the current database.
/// The one place the view namespace's key shape is written down.
fn view_key(name: &ObjectName, db: &str) -> String {
    match name.qualifier() {
        Some(q) => format!("{q}.{}", name.last()),
        None => format!("{db}.{}", name.last()),
    }
}

/// The metadata table's columns. Five strings, self-describing on purpose:
/// `SELECT * FROM _granular_ddl` has to be readable by whoever is looking at a
/// database they did not create.
const DDL_COLUMNS: [&str; 5] = ["kind", "object", "name", "scope", "sql"];

/// `_granular_ddl`'s definition. `Log`: it is appended, read whole, and
/// rewritten whole -- there is nothing to sort by and nothing to index.
fn ddl_table_def(path: &str) -> Result<TableDef> {
    Ok(TableDef {
        name: path.to_string(),
        schema: Schema::new(
            DDL_COLUMNS.iter().map(|n| Field::new(*n, DataType::String)).collect(),
        )?,
        order_by: Vec::new(),
        primary_key: Vec::new(),
        partition_by: None,
        engine: crate::types::Engine::Log,
    })
}

/// Re-parse a stored view body. It went in as the text of one query, so
/// anything else in the row is corruption rather than a user error.
fn parse_view_body(sql: &str) -> Result<crate::sql::ast::Query> {
    match crate::sql::parser::parse_one(sql)? {
        Statement::Query(q) => Ok(*q),
        other => Err(Error::corruption(format!(
            "a view body must be a query, got {}",
            stmt_kind(&other)
        ))),
    }
}

/// The first row where `c` is FALSE, skipping NULLs.
///
/// SQL's rule for a CHECK: only FALSE violates it. One pass over the lane with
/// no allocation, and the null mask is consulted only for the rows that
/// already failed -- a column with no nulls never touches it at all.
fn first_false(c: &Column) -> Option<usize> {
    let crate::types::ColumnData::U64(v) = &c.data else {
        // Guarded at declaration: a non-boolean CHECK is refused there, so
        // reaching this would be a bound expression whose type changed under
        // us. Report no violation rather than guess at truthiness.
        return None;
    };
    let mut from = 0;
    while let Some(k) = v[from..].iter().position(|&x| x == 0) {
        let row = from + k;
        if !c.is_null(row) {
            return Some(row);
        }
        from = row + 1;
    }
    None
}

/// `column = value, column = value` for one row, for a rejection message.
///
/// Long values are cut short: the point is to identify the row, and an error
/// carrying a 64 KB JSON blob is one nobody can read.
fn render_row(b: &Block, schema: &Schema, row: usize) -> String {
    let mut s = String::from("row (");
    for (i, col) in b.columns.iter().enumerate().take(schema.len()) {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(schema.name(i));
        s.push('=');
        let v = col.value(row).to_string();
        match v.char_indices().nth(40) {
            Some((cut, _)) => {
                s.push_str(&v[..cut]);
                s.push_str("...");
            }
            None => s.push_str(&v),
        }
    }
    s.push(')');
    s
}

/// `schema` with column `idx` retyped.
///
/// The column's `DEFAULT` is re-coerced to the new type rather than carried
/// across, so `MODIFY COLUMN c UInt8` on a column defaulting to 300 fails here
/// instead of leaving a default the column can no longer hold.
fn retyped_schema(schema: &Schema, idx: usize, ty: &DataType) -> Result<Schema> {
    let mut fields = schema.fields().to_vec();
    let old = &fields[idx];
    let mut f = Field::new(old.name.clone(), ty.clone());
    if let Some(v) = old.default_value() {
        f = f.with_default_value(v.clone())?;
    }
    fields[idx] = f;
    Schema::new(fields)
}

/// Cast every value of `c` to `ty`, refusing the first one the new type cannot
/// hold. `base_row` is the row number of `c`'s first row within the table, so
/// the message names a row somebody can go and look at.
///
/// **Lossy is refused, not accepted.** `Value::cast_to` is the `CAST(...)`
/// operator's coercion and truncates a float toward zero, which is right for
/// an expression and wrong for a schema change: `MODIFY COLUMN v Int64` on a
/// column holding 1.9 would silently store 1 and there would be no way back.
/// So each value is cast *back* and compared, and anything that does not
/// survive the round trip fails the statement.
fn cast_column(c: &Column, ty: &DataType, name: &str, base_row: usize) -> Result<Column> {
    let n = c.data.len();
    // The declared type exactly: a `Nullable` source going to a non-nullable
    // target is not widened back here, it is refused below, row by row.
    let mut b = ColumnBuilder::with_capacity(ty.clone(), n);
    for i in 0..n {
        if c.is_null(i) {
            if !ty.is_nullable() {
                return Err(Error::exec(format!(
                    "cannot MODIFY `{name}` to {ty}: row {} is NULL, and {ty} has no NULL. \
                     Use `Nullable({ty})`, or delete the row first",
                    base_row + i
                )));
            }
            b.push_null();
            continue;
        }
        let v = c.value(i);
        let cast = v.cast_to(ty).map_err(|e| {
            Error::exec(format!(
                "cannot MODIFY `{name}` to {ty}: row {} holds {v} ({e})",
                base_row + i
            ))
        })?;
        if !round_trips(&v, &cast, &c.ty) {
            return Err(Error::exec(format!(
                "cannot MODIFY `{name}` to {ty}: row {} holds {v}, which {ty} cannot \
                 represent exactly (it would become {cast}). Nothing was changed",
                base_row + i
            )));
        }
        b.push_value(&cast)?;
    }
    Ok(b.finish())
}

/// Does `cast` still mean `original`? Asked by casting back to the old type.
fn round_trips(original: &Value, cast: &Value, old: &DataType) -> bool {
    match cast.cast_to(old) {
        // NaN is never equal to itself, so a float that is still NaN has in
        // fact survived; every other inequality is a real loss.
        Ok(back) => {
            back == *original
                || matches!((&back, original), (Value::Float(a), Value::Float(b))
                    if a.is_nan() && b.is_nan())
        }
        Err(_) => false,
    }
}

/// Publish a table's part files under a second name, without copying them.
///
/// `from` and `to` are table directories, and `snap` is the part set the
/// caller has just checkpointed. Every live part is hard-linked across under
/// the file name it already has, and a `TABLE` record naming them **in part
/// order** is committed in `to` -- which leaves `to` a complete, openable
/// table directory that nothing yet refers to, because the root `CATALOG`
/// still names only `from`. See [`Session::run_rename_table`] for why that
/// ordering is the whole atomicity argument.
///
/// Part *order* is the thing that must not be lost: newest-part-wins is how a
/// keyed table resolves a repeated key, so a record that listed the same parts
/// in a different order would silently resurrect superseded rows. It comes
/// from `PartSet::origin`, which pairs each live part with the file the last
/// checkpoint wrote it to -- not from the directory listing, whose sequence
/// order is allocation order and need not match.
///
/// Returns `false` if the parts cannot be linked (any part not yet written, or
/// the source directory absent), in which case the caller lets the checkpoint
/// write the new directory from scratch.
fn link_table_dir(
    from: &Path,
    to: &Path,
    def: &TableDef,
    snap: &crate::storage::part::Snapshot,
    wal_committed: u64,
) -> Result<bool> {
    use crate::persist::{store, writer};
    let set = snap.set();
    let mut names = Vec::with_capacity(snap.len());
    for i in 0..snap.len() {
        match set.origin(i) {
            crate::storage::part::NO_FILE => return Ok(false),
            seq => names.push(store::part_file_name(seq)),
        }
    }
    if !from.is_dir() || !names.iter().all(|n| from.join(n).exists()) {
        return Ok(false);
    }
    if to.exists() {
        // An orphan from a rename interrupted before its `CATALOG` was
        // published. Nothing can be referring to it -- the roster that
        // survived the crash cannot name it -- so it is ours to replace.
        std::fs::remove_dir_all(to)
            .map_err(|e| Error::Io(format!("cannot remove {}: {e}", to.display())))?;
    }
    std::fs::create_dir_all(to)
        .map_err(|e| Error::Io(format!("cannot create {}: {e}", to.display())))?;
    for n in &names {
        // A copy on a filesystem with no links, decided once for the process
        // rather than re-attempted per part -- a big table is a lot of parts.
        // Preferred over the `Ok(false)` contract below, which is also a legal
        // answer here: that path makes the checkpoint re-encode every part
        // from memory, where a copy just moves the bytes that are already
        // right.
        store::link_or_copy(&from.join(n), &to.join(n)).map_err(|e| {
            Error::Io(format!("cannot link {} into {}: {e}", n, to.display()))
        })?;
    }
    // The table *directory* is fresh; the log stream under the destination
    // name need not be. Logs live at `<root>/.wal/<db>/<table>` and outlive
    // their tables, so a rename onto a name that was dropped meets a stream
    // that has already run -- and a watermark of zero would ask replay to
    // fold the dead incarnation's records into this one. The caller passes
    // where that stream has reached, exactly as `CREATE TABLE` does. (The
    // parts above cover everything: the caller checkpointed, so there is
    // nothing left in the source's log either.)
    store::commit(
        &to.join(store::TABLE_FILE),
        &writer::table_doc(def, &names, wal_committed),
    )?;
    store::sync_dir(to)?;
    Ok(true)
}

/// String literals in VALUES adopt the column's type, so
/// `INSERT INTO t VALUES ('2024-01-01')` into a Date column works.
fn coerce_literal(v: &Value, ty: &DataType) -> Result<Value> {
    if v.is_null() {
        return if ty.is_nullable() {
            Ok(Value::Null)
        } else {
            Err(Error::exec(format!("cannot store NULL in non-nullable {ty}")))
        };
    }
    match (v, ty.base()) {
        (Value::Str(s), DataType::Date) => Ok(Value::Date(crate::types::parse_date(s)?)),
        (Value::Str(s), DataType::DateTime) => {
            Ok(Value::DateTime(crate::types::parse_datetime(s)?))
        }
        _ => v.cast_to(ty),
    }
}

/// Turn `ORDER BY (a, b)` / `ORDER BY a` into column indices. Only bare column
/// references and tuples of them are supported: an expression key would need a
/// materialized computed column, which this engine does not have.
fn resolve_key_exprs(
    exprs: &[crate::sql::ast::Expr],
    schema: &Schema,
    what: &str,
) -> Result<Vec<usize>> {
    use crate::sql::ast::Expr;
    let mut out = Vec::new();
    for e in exprs {
        match e {
            Expr::Column(n) => out.push(schema.require(n.last())?),
            Expr::Tuple(items) => {
                for i in items {
                    match i {
                        Expr::Column(n) => out.push(schema.require(n.last())?),
                        other => {
                            return Err(Error::unsupported(format!(
                                "{what} only accepts column names, got `{other}`"
                            )))
                        }
                    }
                }
            }
            // `ORDER BY tuple()` is ClickHouse for "no ordering at all".
            Expr::Function { name, args, .. }
                if name.eq_ignore_ascii_case("tuple") && args.is_empty() => {}
            other => {
                return Err(Error::unsupported(format!(
                    "{what} only accepts column names, got `{other}`"
                )))
            }
        }
    }
    Ok(out)
}

/// `checks` and `unique` are the table's constraints, which live beside the
/// `TableDef` rather than in it. They are printed for the same reason the
/// `PRIMARY KEY` is: this output is what somebody pastes into a migration, and
/// DDL that silently dropped a constraint would produce a table that accepts
/// writes the original refuses.
fn render_create_table(
    schema: &Schema,
    def: &TableDef,
    checks: &[Check],
    unique: Option<&str>,
) -> String {
    let mut cols: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| {
            let u = if unique.is_some_and(|u| u.eq_ignore_ascii_case(&f.name)) {
                " UNIQUE"
            } else {
                ""
            };
            match f.default_sql() {
                Some(d) => format!("    `{}` {}{u} DEFAULT {d}", f.name, f.ty),
                None => format!("    `{}` {}{u}", f.name, f.ty),
            }
        })
        .collect();
    cols.extend(
        checks
            .iter()
            .map(|c| format!("    CONSTRAINT `{}` CHECK ({})", c.name, c.sql)),
    );
    let key = |idx: &[usize]| -> String {
        idx.iter()
            .map(|&i| format!("`{}`", schema.name(i)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut s = format!(
        "CREATE TABLE `{}`\n(\n{}\n)\nENGINE = {}",
        def.name.rsplit('.').next().unwrap_or(&def.name),
        cols.join(",\n"),
        def.engine.name()
    );
    if !def.order_by.is_empty() {
        s.push_str(&format!("\nORDER BY ({})", key(&def.order_by)));
    }
    // Emitted whenever it exists, even when it equals ORDER BY: since a sort
    // key no longer implies uniqueness, omitting an equal PRIMARY KEY would
    // make the printed DDL re-create an *unkeyed* table. SHOW CREATE output is
    // exactly what a user pastes into a migration.
    if !def.primary_key.is_empty() {
        s.push_str(&format!("\nPRIMARY KEY ({})", key(&def.primary_key)));
    }
    if let Some(p) = def.partition_by {
        s.push_str(&format!("\nPARTITION BY `{}`", schema.name(p)));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persist::store::{CATALOG_FILE, TABLE_FILE};
    use crate::persist::testkit::Scratch;

    const DDL: &str = "CREATE TABLE t (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id";

    /// The safety property of planning without cloning: a false *negative*
    /// from `has_subquery` leaves an `Expr::Subquery` in the tree for the
    /// binder to trip over, so every position the rewriter folds in has to be
    /// a position this detector looks in.
    ///
    /// One entry per arm of `Session::rewrite_*`. A false positive costs a
    /// clone and is not a bug, which is why the negatives below are only
    /// spot-checks -- they exist to prove the detector is not `|| true`.
    #[test]
    fn every_position_the_folder_rewrites_is_a_position_the_detector_looks_in() {
        let empty = Extensions::default();
        let names = Names { ext: &empty, db: crate::catalog::DEFAULT_DATABASE };
        let with = [
            "SELECT (SELECT max(id) FROM t) FROM t",
            "SELECT * FROM t WHERE id IN (SELECT id FROM t)",
            "SELECT * FROM t WHERE EXISTS (SELECT 1 FROM t)",
            "SELECT * FROM t PREWHERE id > (SELECT min(id) FROM t)",
            "SELECT id, count() FROM t GROUP BY id HAVING count() > (SELECT 1)",
            "SELECT * FROM t GROUP BY id + (SELECT 0)",
            "WITH c AS (SELECT (SELECT 1) AS x FROM t) SELECT * FROM c",
            "SELECT * FROM (SELECT id FROM t WHERE id IN (SELECT id FROM t)) s",
            "SELECT * FROM t a JOIN t b ON a.id = b.id + (SELECT 0)",
            "SELECT * FROM t ORDER BY id + (SELECT 0)",
            "SELECT * FROM t LIMIT (SELECT 1)",
            "SELECT * FROM t WHERE CASE WHEN id > (SELECT 0) THEN 1 ELSE 0 END = 1",
            "SELECT abs(id - (SELECT 0)) FROM t",
            "SELECT sum(v) OVER (PARTITION BY id + (SELECT 0)) FROM t",
            "SELECT * FROM t WHERE id BETWEEN (SELECT 0) AND 9",
            "SELECT * FROM t WHERE id IN (1, (SELECT 2))",
            "SELECT id FROM t UNION ALL SELECT (SELECT 1) FROM t",
        ];
        for sql in with {
            let stmts = parse(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
            let Statement::Query(q) = &stmts[0] else { panic!("{sql}: not a query") };
            assert!(has_subquery(q, names), "missed the subquery in `{sql}`");
        }
        for sql in [
            "SELECT id, v FROM t WHERE id = 1",
            "SELECT id, sum(v) FROM t GROUP BY id ORDER BY id LIMIT 10",
            "SELECT id FROM t UNION ALL SELECT id FROM t",
            "SELECT * FROM t a JOIN t b ON a.id = b.id",
        ] {
            let stmts = parse(sql).unwrap();
            let Statement::Query(q) = &stmts[0] else { panic!("{sql}: not a query") };
            assert!(!has_subquery(q, names), "`{sql}` has no subquery to fold");
        }
    }

    /// Two sessions over one directory used to both report success and then
    /// overwrite each other's parts. The second must now fail loudly.
    ///
    /// Both sessions living in this one process is the sharp end of the test,
    /// not a shortcut: `flock` excludes per *open file description*, not per
    /// process, and each `Session::open` does its own `open(2)`, so the second
    /// call contends for real. The pid assertion is what rules out a
    /// same-process shortcut passing by accident -- only the bytes in the file
    /// on disk can supply the holder's identity.
    #[test]
    fn a_second_session_on_one_directory_is_refused() {
        let s = Scratch::new("session-lock");
        let mut first = Session::open(s.path()).unwrap();
        first.execute(DDL).unwrap();
        first.execute("INSERT INTO t VALUES (1, 10)").unwrap();
        first.checkpoint().unwrap();

        let Err(err) = Session::open(s.path()) else {
            panic!("a second session on {} must be refused", s.path().display())
        };
        assert_eq!(err.code(), "STORAGE_ERROR", "{err}");
        let msg = err.to_string();
        assert!(msg.contains(&format!("pid {}", std::process::id())), "{msg}");
        assert!(msg.contains(&s.path().display().to_string()), "{msg}");

        // The release is the fd close, so dropping the session is the whole
        // unlock protocol; the stale LOCK file it leaves behind must not need
        // removing before the next open.
        drop(first);
        assert!(s.join(LOCK_FILE).exists(), "the lock file is left in place");
        let mut second = Session::open(s.path()).unwrap();
        assert_eq!(
            second.query("SELECT count() FROM t").unwrap().scalar(),
            Some(Value::UInt(1)),
            "the surviving row is the one the first session committed"
        );
    }

    #[test]
    fn the_lock_file_records_the_holding_process() {
        let s = Scratch::new("session-lock-pid");
        let held = Session::open(s.path()).unwrap();
        let txt = std::fs::read_to_string(s.join(LOCK_FILE)).unwrap();
        assert_eq!(txt.trim().parse::<u32>().unwrap(), std::process::id());
        drop(held);
    }

    /// An in-memory session shares no file with anyone, so it must neither
    /// take the lock nor contend for one a persistent session holds.
    #[test]
    fn an_in_memory_session_takes_no_lock() {
        let s = Scratch::new("session-inmem");
        let held = Session::open(s.path()).unwrap();
        assert!(held._lock.is_some());
        let a = Session::in_memory();
        let b = Session::in_memory();
        assert!(a._lock.is_none() && b._lock.is_none());
        drop((a, b, held));
    }

    // ---- transactions ------------------------------------------------------

    const KEYED: &str =
        "CREATE TABLE t (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id";

    fn count(s: &mut Session) -> u64 {
        match s.query("SELECT count() FROM t").unwrap().scalar() {
            Some(Value::UInt(n)) => n,
            other => panic!("count() returned {other:?}"),
        }
    }

    fn v_of(s: &mut Session, id: u64) -> Option<i64> {
        let rs = s.query(&format!("SELECT v FROM t WHERE id = {id}")).unwrap();
        match rs.scalar() {
            Some(Value::Int(v)) => Some(v),
            None => None,
            other => panic!("v returned {other:?}"),
        }
    }

    /// The SQL spelling, end to end, in memory.
    #[test]
    fn begin_commit_and_rollback_run_as_sql() {
        let mut s = Session::in_memory();
        s.execute(KEYED).unwrap();
        s.execute("INSERT INTO t VALUES (1, 10)").unwrap();

        s.execute("BEGIN").unwrap();
        assert!(s.in_transaction());
        s.execute("INSERT INTO t VALUES (2, 20)").unwrap();
        assert_eq!(count(&mut s), 2, "read-your-own-writes inside the transaction");
        s.execute("ROLLBACK").unwrap();
        assert!(!s.in_transaction());
        assert_eq!(count(&mut s), 1, "the rolled-back row is gone");
        assert_eq!(v_of(&mut s, 2), None);

        // Every accepted spelling, and a multi-statement batch in one call.
        for open in ["BEGIN", "BEGIN WORK", "BEGIN TRANSACTION", "START TRANSACTION"] {
            let before = count(&mut s);
            s.run(&format!("{open}; INSERT INTO t VALUES (9, 90); ROLLBACK")).unwrap();
            assert_eq!(count(&mut s), before, "{open}");
        }
        s.run("BEGIN; INSERT INTO t VALUES (3, 30); COMMIT").unwrap();
        assert_eq!(count(&mut s), 2);
        assert_eq!(v_of(&mut s, 3), Some(30));
        for close in ["COMMIT WORK", "COMMIT TRANSACTION", "ROLLBACK WORK"] {
            s.execute("BEGIN").unwrap();
            s.execute(close).unwrap();
            assert!(!s.in_transaction(), "{close}");
        }
    }

    /// A transaction must be able to overwrite and delete committed rows and
    /// still see the result, and a rollback must put every one of them back.
    #[test]
    fn a_rolled_back_transaction_restores_updates_and_deletes() {
        let mut s = Session::in_memory();
        s.execute(KEYED).unwrap();
        s.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)").unwrap();

        s.execute("BEGIN").unwrap();
        s.execute("INSERT INTO t VALUES (1, 111)").unwrap();
        s.execute("ALTER TABLE t DELETE WHERE id = 2").unwrap();
        s.execute("ALTER TABLE t UPDATE v = 333 WHERE id = 3").unwrap();
        s.execute("INSERT INTO t VALUES (4, 40)").unwrap();
        assert_eq!(count(&mut s), 3);
        assert_eq!(v_of(&mut s, 1), Some(111));
        assert_eq!(v_of(&mut s, 2), None);
        assert_eq!(v_of(&mut s, 3), Some(333));
        assert_eq!(v_of(&mut s, 4), Some(40));

        s.execute("ROLLBACK").unwrap();
        assert_eq!(count(&mut s), 3);
        assert_eq!(v_of(&mut s, 1), Some(10));
        assert_eq!(v_of(&mut s, 2), Some(20));
        assert_eq!(v_of(&mut s, 3), Some(30));
        assert_eq!(v_of(&mut s, 4), None);
    }

    /// A committed transaction has to be indistinguishable from the same
    /// statements run in autocommit -- otherwise it is a different engine.
    #[test]
    fn a_committed_transaction_matches_autocommit() {
        let stmts = [
            "INSERT INTO t VALUES (1, 10), (2, 20)",
            "INSERT INTO t VALUES (2, 22), (3, 30)",
            "ALTER TABLE t DELETE WHERE id = 1",
            "ALTER TABLE t UPDATE v = 99 WHERE id = 3",
        ];
        let mut plain = Session::in_memory();
        plain.execute(KEYED).unwrap();
        for st in stmts {
            plain.execute(st).unwrap();
        }
        let mut txn = Session::in_memory();
        txn.execute(KEYED).unwrap();
        txn.execute("BEGIN").unwrap();
        for st in stmts {
            txn.execute(st).unwrap();
        }
        txn.execute("COMMIT").unwrap();

        let want = plain.query("SELECT id, v FROM t ORDER BY id").unwrap().to_values();
        let got = txn.query("SELECT id, v FROM t ORDER BY id").unwrap().to_values();
        assert_eq!(got, want);
        assert!(!want.is_empty());
    }

    /// The isolation claim, at the level the substrate actually models it: a
    /// reader holding a `Snapshot` -- which is exactly what a concurrent
    /// reader thread holds, and what a second session on this table would take
    /// -- sees none of the transaction, and then sees all of it at once.
    ///
    /// `committed_snapshot` is the other half: it is what a reader that is not
    /// the writing session asks for, and it never reports an overlay at all.
    #[test]
    fn an_outside_reader_sees_nothing_until_commit() {
        let mut s = Session::in_memory();
        s.execute(KEYED).unwrap();
        s.execute("INSERT INTO t VALUES (1, 10)").unwrap();
        s.execute("SYSTEM FLUSH").unwrap();
        let pinned = s.catalog.table_by_path("default.t").unwrap().snapshot();

        s.execute("BEGIN").unwrap();
        s.execute("INSERT INTO t VALUES (2, 20), (3, 30)").unwrap();
        s.execute("SYSTEM FLUSH").unwrap();
        assert_eq!(count(&mut s), 3, "the writer sees its own writes");
        {
            let t = s.catalog.table_by_path("default.t").unwrap();
            assert_eq!(pinned.live_rows(), 1, "the pinned reader moved");
            assert_eq!(t.committed_snapshot().live_rows(), 1, "uncommitted parts published");
        }

        s.execute("COMMIT").unwrap();
        assert_eq!(
            s.catalog.table_by_path("default.t").unwrap().committed_snapshot().live_rows(),
            3,
            "COMMIT must publish the whole transaction"
        );
        assert_eq!(pinned.live_rows(), 1, "and the pinned reader stays pinned");
        assert_eq!(count(&mut s), 3);
    }

    /// A rollback must leave the on-disk log byte-identical, not merely
    /// semantically empty: the staged records replay would drop are still
    /// bytes every later open re-scans.
    #[test]
    fn rollback_leaves_no_trace_on_disk() {
        let s = Scratch::new("session-txn-rollback-disk");
        let mut db = Session::open(s.path()).unwrap();
        db.execute(KEYED).unwrap();
        db.execute("INSERT INTO t VALUES (1, 10)").unwrap();
        let wal = crate::persist::wal::wal_dir(s.path(), "default", "t");
        // The *records*, not the whole file: the segment header carries a
        // durability watermark that legitimately moves when a transaction
        // syncs and is then rolled back. "No trace" is a claim about the log's
        // contents and its length, and both are asserted.
        let body = |d: &std::path::Path| -> Vec<(String, Vec<u8>)> {
            let mut v: Vec<(String, Vec<u8>)> = std::fs::read_dir(d)
                .unwrap()
                .flatten()
                .map(|e| {
                    let b = std::fs::read(e.path()).unwrap();
                    let head = crate::persist::wal::SEG_HEADER_LEN as usize;
                    (e.file_name().to_string_lossy().into_owned(), b[head.min(b.len())..].to_vec())
                })
                .collect();
            v.sort();
            v
        };
        let before = body(&wal);

        db.execute("BEGIN").unwrap();
        for i in 2..40u64 {
            db.execute(&format!("INSERT INTO t VALUES ({i}, {})", i * 10)).unwrap();
        }
        db.execute("ALTER TABLE t DELETE WHERE id = 1").unwrap();
        assert!(
            body(&wal)[0].1.len() > before[0].1.len(),
            "the test needs the transaction to have logged something"
        );
        db.execute("ROLLBACK").unwrap();

        assert_eq!(body(&wal), before, "the log kept the aborted records");
        assert_eq!(count(&mut db), 1);
        assert_eq!(v_of(&mut db, 1), Some(10));

        // ...and it survives a real restart.
        db.checkpoint().unwrap();
        drop(db);
        let mut back = Session::open(s.path()).unwrap();
        assert_eq!(count(&mut back), 1);
        assert_eq!(v_of(&mut back, 1), Some(10));
    }

    /// A crash between BEGIN and COMMIT replays nothing: the staged records
    /// have no marker behind them, which is the definition of a write that was
    /// never acknowledged. Dropping the session without committing is exactly
    /// the state a killed process leaves the *files* in.
    #[test]
    fn a_crash_mid_transaction_replays_nothing() {
        let s = Scratch::new("session-txn-crash");
        {
            let mut db = Session::open(s.path()).unwrap();
            db.execute(KEYED).unwrap();
            db.execute("INSERT INTO t VALUES (1, 10)").unwrap();
            db.checkpoint().unwrap();

            db.execute("BEGIN").unwrap();
            db.execute("INSERT INTO t VALUES (2, 20), (3, 30)").unwrap();
            db.execute("INSERT INTO t VALUES (1, 999)").unwrap();
            db.execute("ALTER TABLE t DELETE WHERE id = 1").unwrap();
            assert_eq!(count(&mut db), 2);
            // No COMMIT, no ROLLBACK: the process dies here.
            std::mem::forget(db.txn.take());
            drop(db);
        }
        let mut back = Session::open(s.path()).unwrap();
        assert_eq!(count(&mut back), 1, "an uncommitted transaction was replayed");
        assert_eq!(v_of(&mut back, 1), Some(10), "the pre-transaction row was altered");
        assert_eq!(v_of(&mut back, 2), None);
        assert_eq!(v_of(&mut back, 3), None);
    }

    /// The other side: a committed transaction survives a crash before the
    /// next checkpoint, because the marker is fsynced before COMMIT returns.
    #[test]
    fn a_committed_transaction_survives_a_crash() {
        let s = Scratch::new("session-txn-durable");
        {
            let mut db = Session::open(s.path()).unwrap();
            db.execute(KEYED).unwrap();
            db.execute("BEGIN").unwrap();
            db.execute("INSERT INTO t VALUES (1, 10), (2, 20)").unwrap();
            db.execute("INSERT INTO t VALUES (2, 22)").unwrap();
            db.execute("COMMIT").unwrap();
            // No checkpoint: only the log can carry this across the restart.
            drop(db);
        }
        let mut back = Session::open(s.path()).unwrap();
        assert_eq!(count(&mut back), 2);
        assert_eq!(v_of(&mut back, 1), Some(10));
        assert_eq!(v_of(&mut back, 2), Some(22));
    }

    /// Statement-level atomicity: a multi-block `INSERT ... SELECT` that fails
    /// part way through used to leave the blocks it had already published in
    /// the table, with an error returned on top. Failure is injected in the
    /// storage layer, which is where the real ones (a codec error, an
    /// allocation) come from.
    #[test]
    fn a_failed_multi_block_insert_publishes_nothing() {
        let mut s = Session::in_memory();
        s.execute(
            "CREATE TABLE src (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
        )
        .unwrap();
        s.execute(KEYED).unwrap();
        // Three `BLOCK_SIZE` blocks out of the scan, each one a separate bulk
        // ingest and so a separate publish -- which is the shape the atomicity
        // is about.
        const N: u64 = 3 * crate::common::BLOCK_SIZE as u64;
        let rows: Vec<String> = (0..N).map(|i| format!("({i},{i})")).collect();
        for chunk in rows.chunks(4096) {
            s.execute(&format!("INSERT INTO src VALUES {}", chunk.join(","))).unwrap();
        }
        s.execute("INSERT INTO t VALUES (0, -1)").unwrap();
        s.execute("SYSTEM FLUSH").unwrap(); // so the arming below counts only this statement
        let before = count(&mut s);
        assert_eq!(before, 1);

        // The first publish succeeds and the second fails. Before this
        // statement ran inside an implicit transaction, the first stayed in
        // the table with the error returned over the top of it.
        crate::storage::table::arm_build_failure_after(1);
        let err = s.execute("INSERT INTO t SELECT id, v FROM src").unwrap_err();
        crate::storage::table::disarm_build_failure();
        assert!(err.to_string().contains("injected"), "{err}");

        assert!(!s.in_transaction(), "the implicit transaction must have been closed");
        assert_eq!(count(&mut s), before, "a failed statement published rows");
        assert_eq!(v_of(&mut s, 0), Some(-1), "and it altered a row it should not have");
        // ...and a retry still does the whole job.
        s.execute("INSERT INTO t SELECT id, v FROM src").unwrap();
        assert_eq!(count(&mut s), N);
        assert_eq!(v_of(&mut s, 0), Some(0));
    }

    /// DDL checkpoints, and a checkpoint inside a transaction would persist
    /// parts a ROLLBACK is still entitled to erase. Both doors are shut.
    ///
    /// Partly inverted: the refusal is unchanged, but it no longer leaves the
    /// transaction usable. A statement that fails inside a transaction poisons
    /// it, so each refusal is tested in a transaction of its own -- the old
    /// shape ran all four in one and then committed, which is exactly the
    /// "statements after a failure return Ok and are discarded" hole.
    #[test]
    fn ddl_and_checkpoint_are_refused_inside_a_transaction() {
        let s = Scratch::new("session-txn-ddl");
        let mut db = Session::open(s.path()).unwrap();
        db.execute(KEYED).unwrap();

        for ddl in [
            "CREATE TABLE u (id UInt64) ENGINE = MergeTree ORDER BY id",
            "DROP TABLE t",
            "TRUNCATE TABLE t",
            "ALTER TABLE t ADD COLUMN w Int64",
        ] {
            db.execute("BEGIN").unwrap();
            db.execute("INSERT INTO t VALUES (1, 10)").unwrap();
            let e = db.execute(ddl).unwrap_err();
            assert_eq!(e.code(), "NOT_IMPLEMENTED", "{ddl}: {e}");
            // Poisoned, so the rows it wrote can only be discarded.
            assert_eq!(db.execute("COMMIT").unwrap_err().code(), "EXECUTION_ERROR");
            assert!(!db.in_transaction(), "{ddl}: a refused COMMIT still ends the transaction");
            assert_eq!(count(&mut db), 0, "{ddl}: the poisoned transaction published rows");
        }

        db.execute("BEGIN").unwrap();
        db.execute("INSERT INTO t VALUES (1, 10)").unwrap();
        assert_eq!(db.checkpoint().unwrap_err().code(), "NOT_IMPLEMENTED");
        // `checkpoint` is the direct API, not a statement, so it reports
        // without poisoning -- and the transaction it declined to persist
        // still commits.
        assert_eq!(count(&mut db), 1);
        db.execute("COMMIT").unwrap();
        assert_eq!(count(&mut db), 1);
        db.checkpoint().unwrap();
    }

    /// Enlistment is per table and lazy, so a transaction over two of them has
    /// to publish (or discard) both, and must leave a third alone.
    #[test]
    fn a_transaction_over_two_tables_commits_and_rolls_back_together() {
        let mut s = Session::in_memory();
        for n in ["a", "b", "c"] {
            s.execute(&KEYED.replace(" t ", &format!(" {n} "))).unwrap();
            s.execute(&format!("INSERT INTO {n} VALUES (1, 1)")).unwrap();
        }
        let n = |s: &mut Session, t: &str| -> u64 {
            match s.query(&format!("SELECT count() FROM {t}")).unwrap().scalar() {
                Some(Value::UInt(v)) => v,
                other => panic!("{other:?}"),
            }
        };

        s.execute("BEGIN").unwrap();
        s.execute("INSERT INTO a VALUES (2, 2)").unwrap();
        s.execute("INSERT INTO b VALUES (2, 2), (3, 3)").unwrap();
        assert_eq!((n(&mut s, "a"), n(&mut s, "b"), n(&mut s, "c")), (2, 3, 1));
        s.execute("ROLLBACK").unwrap();
        assert_eq!((n(&mut s, "a"), n(&mut s, "b"), n(&mut s, "c")), (1, 1, 1));

        s.execute("BEGIN").unwrap();
        s.execute("INSERT INTO a VALUES (2, 2)").unwrap();
        s.execute("INSERT INTO b VALUES (2, 2), (3, 3)").unwrap();
        s.execute("COMMIT").unwrap();
        assert_eq!((n(&mut s, "a"), n(&mut s, "b"), n(&mut s, "c")), (2, 3, 1));
    }

    /// A prepare whose barrier fails must abort the COMMIT **before** the
    /// decision is written -- the one property the parallel prepare could have
    /// broken.
    ///
    /// The barriers now run concurrently, so the error does not arrive at the
    /// statement that caused it; it arrives from a join. A `?` inside that
    /// scope, or a join that only ran on the happy path, would let the
    /// decision be appended over a participant whose bytes are not on the
    /// platter -- and replay would then release every prepared group and find
    /// one participant's records missing, which is exactly the half-committed
    /// transaction two-phase commit exists to prevent.
    ///
    /// Four tables, so there are three prepares and the failure is in the
    /// middle one rather than the first or the last. The assertion is made
    /// twice: in the session that failed, and in a fresh one over the same
    /// directory, because "nothing moved" has to be true of the disk too.
    #[test]
    fn a_prepare_whose_barrier_fails_stops_the_commit_before_the_decision() {
        let s = Scratch::new("prepare-barrier");
        let names = ["a", "b", "c", "d"];
        let mut db = Session::open(s.path()).unwrap();
        for n in names {
            db.execute(&KEYED.replace(" t ", &format!(" {n} "))).unwrap();
            db.execute(&format!("INSERT INTO {n} VALUES (1, 1)")).unwrap();
        }
        db.checkpoint().unwrap();
        let rows = |db: &mut Session| {
            names.map(|n| match db.query(&format!("SELECT count() FROM {n}")).unwrap().scalar() {
                Some(Value::UInt(v)) => v,
                other => panic!("{n}: {other:?}"),
            })
        };
        assert_eq!(rows(&mut db), [1, 1, 1, 1]);

        // Participant 2 of 3. `wal_for` keys on the qualified path and the
        // transaction below enlists in `names` order, so `b` is the second
        // prepare and `d` is the coordinator.
        db.wal_for("default.b").unwrap().expect("a logging session").refuse_barriers();

        db.execute("BEGIN").unwrap();
        for n in names {
            db.execute(&format!("INSERT INTO {n} VALUES (2, 2)")).unwrap();
        }
        let e = db.execute("COMMIT").unwrap_err().to_string();
        assert!(e.contains("fsync"), "the refused barrier must be what COMMIT reports: {e}");
        assert_eq!(rows(&mut db), [1, 1, 1, 1], "a table moved under a COMMIT that failed");

        drop(db);
        let mut again = Session::open(s.path()).unwrap();
        assert_eq!(
            rows(&mut again),
            [1, 1, 1, 1],
            "recovery released a transaction whose prepare was never made durable"
        );
    }

    /// Partly inverted: the last line used to be `COMMIT` succeeding after a
    /// refused nested `BEGIN`, which is the bug. A block that opens its own
    /// transaction, finds one already open, and then commits was committing
    /// the *outer* transaction's uncommitted work at a boundary the outer
    /// writer never chose. The nested `BEGIN` now poisons, so the only way on
    /// is ROLLBACK.
    #[test]
    fn commit_or_rollback_without_a_transaction_is_an_error() {
        let mut s = Session::in_memory();
        s.execute(KEYED).unwrap();
        assert!(s.execute("COMMIT").is_err());
        assert!(s.execute("ROLLBACK").is_err());
        s.execute("BEGIN").unwrap();
        s.execute("INSERT INTO t VALUES (1, 10)").unwrap();
        assert!(s.execute("BEGIN").is_err(), "nesting is refused");
        assert!(s.in_transaction(), "and leaves the outer transaction open");
        assert!(s.execute("COMMIT").is_err(), "which it must not publish");
        assert!(!s.in_transaction());
        assert_eq!(count(&mut s), 0, "the nested block's COMMIT committed the outer work");

        // ROLLBACK is the way out, and after it the session is ordinary again.
        s.execute("BEGIN").unwrap();
        assert!(s.execute("BEGIN").is_err());
        s.execute("ROLLBACK").unwrap();
        s.execute("BEGIN").unwrap();
        s.execute("INSERT INTO t VALUES (1, 10)").unwrap();
        s.execute("COMMIT").unwrap();
        assert_eq!(count(&mut s), 1);
    }

    /// The interception must not shadow ordinary SQL. A string literal, a
    /// column and an identifier that merely *contain* the keywords all have to
    /// keep working, and a semicolon inside a literal is not a boundary.
    #[test]
    fn transaction_keywords_inside_ordinary_sql_are_left_alone() {
        let mut s = Session::in_memory();
        s.execute(
            "CREATE TABLE logs (id UInt64, commit_msg String, started UInt64) \
             ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
        )
        .unwrap();
        s.execute("INSERT INTO logs VALUES (1, 'begin; rollback', 5)").unwrap();
        s.execute("INSERT INTO logs VALUES (2, 'COMMIT', 7)").unwrap();
        let rs = s
            .query("SELECT commit_msg FROM logs WHERE started = 5")
            .unwrap();
        assert_eq!(rs.scalar(), Some(Value::str("begin; rollback")));
        assert_eq!(
            s.query("SELECT count() FROM logs WHERE commit_msg = 'COMMIT'")
                .unwrap()
                .scalar(),
            Some(Value::UInt(1))
        );
        assert!(!s.in_transaction(), "nothing above opened a transaction");
        // ...and a `--` comment naming one is inert too.
        s.run("-- begin here\nSELECT 1; -- commit\nSELECT 2").unwrap();
        assert!(!s.in_transaction());
    }

    /// The gate on the whole interception. It must answer `false` for the
    /// statements this engine actually runs, or every query pays for a second
    /// lex it does not need.
    #[test]
    fn the_transaction_prefilter_is_false_for_ordinary_sql() {
        for sql in [
            "SELECT count() FROM hits WHERE region = 3 GROUP BY user_id ORDER BY 1 DESC LIMIT 10",
            "INSERT INTO t VALUES (1, 2), (3, 4)",
            "CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id",
            "ALTER TABLE t UPDATE v = 1 WHERE id = 2",
            "SELECT bs, crumble, sortie FROM t",
        ] {
            assert!(!mentions_txn_keyword(sql), "{sql}");
        }
        for sql in ["BEGIN", "  commit  ", "x;RollBack", "start transaction"] {
            assert!(mentions_txn_keyword(sql), "{sql}");
        }
        // The documented false positives: no word-boundary tracking, so a
        // keyword embedded in an identifier trips the filter. That costs one
        // extra lex and cannot change an answer, because the split that
        // follows is done by the real tokenizer -- which is what
        // `transaction_keywords_inside_ordinary_sql_are_left_alone` pins.
        for sql in ["SELECT restart FROM t", "SELECT x_commit FROM t"] {
            assert!(mentions_txn_keyword(sql), "{sql}");
            let mut s = Session::in_memory();
            s.execute("CREATE TABLE t (restart UInt64, x_commit UInt64) ENGINE = Log").unwrap();
            assert!(s.query(sql).is_ok(), "{sql}");
            assert!(!s.in_transaction());
        }
    }

    /// Losing the root `CATALOG` must not be reinterpreted as "empty
    /// database": the next checkpoint would collect every table directory as
    /// dropped and delete data that is still entirely intact.
    #[test]
    fn a_lost_catalog_is_refused_rather_than_emptied() {
        let s = Scratch::new("session-lost-catalog");
        {
            let mut db = Session::open(s.path()).unwrap();
            db.execute(DDL).unwrap();
            db.execute("INSERT INTO t VALUES (1, 10), (2, 20)").unwrap();
            db.checkpoint().unwrap();
        }
        let tdir = s.join("default").join("t");
        assert!(tdir.join(TABLE_FILE).exists());
        std::fs::remove_file(s.join(CATALOG_FILE)).unwrap();

        let Err(err) = Session::open(s.path()) else {
            panic!("a database whose CATALOG is missing must be refused")
        };
        assert!(err.to_string().contains("default/t"), "{err}");
        // The entire point of refusing: everything needed to recover is still
        // on disk afterwards.
        assert!(tdir.join(TABLE_FILE).exists(), "the commit record was destroyed");
        assert!(
            !crate::persist::store::list_part_files(&tdir).unwrap().is_empty(),
            "the parts were destroyed"
        );
    }

    // ---------------------------------------------------------- mutations

    fn vals(s: &mut Session, sql: &str) -> Vec<Vec<Value>> {
        s.query(sql).unwrap().to_values()
    }

    /// The transaction guarantee for a bulk delete, end to end through the
    /// public API. A sweep publishes into the table's overlay, so ROLLBACK
    /// drops it whole -- and the interesting half is that an in-memory session
    /// writes no log at all, so nothing but `apply_sweep`'s explicit `enlist`
    /// opens that overlay in the first place. Without it the sweep goes
    /// straight into the committed set and this test loses 2500 rows.
    #[test]
    fn a_delete_inside_a_rolled_back_transaction_leaves_every_row() {
        let mut s = Session::in_memory();
        s.execute(
            "CREATE TABLE t (id UInt64, v Int64) ENGINE = MergeTree PRIMARY KEY id ORDER BY id",
        )
        .unwrap();
        let rows: Vec<String> = (0..3_000u64).map(|i| format!("({i},{i})")).collect();
        s.execute(&format!("INSERT INTO t VALUES {}", rows.join(","))).unwrap();

        s.execute("BEGIN").unwrap();
        assert_eq!(
            s.query("DELETE FROM t WHERE id < 2500").unwrap().affected,
            Some(2_500)
        );
        assert_eq!(vals(&mut s, "SELECT count() FROM t")[0][0], Value::UInt(500));
        s.execute("ROLLBACK").unwrap();
        assert_eq!(vals(&mut s, "SELECT count() FROM t")[0][0], Value::UInt(3_000));
        assert_eq!(vals(&mut s, "SELECT count() FROM t WHERE id < 2500")[0][0], Value::UInt(2_500));

        // The same statement committed does publish, and an UPDATE alongside
        // it rolls back too -- a mutation's two halves are one unit.
        s.execute("BEGIN").unwrap();
        s.execute("DELETE FROM t WHERE id < 2500").unwrap();
        s.execute("UPDATE t SET v = -1 WHERE id = 2500").unwrap();
        s.execute("ROLLBACK").unwrap();
        assert_eq!(vals(&mut s, "SELECT count() FROM t")[0][0], Value::UInt(3_000));
        assert_eq!(vals(&mut s, "SELECT v FROM t WHERE id = 2500")[0][0], Value::Int(2_500));

        s.execute("BEGIN").unwrap();
        s.execute("DELETE FROM t WHERE id < 2500").unwrap();
        s.execute("UPDATE t SET v = -1 WHERE id = 2500").unwrap();
        s.execute("COMMIT").unwrap();
        assert_eq!(vals(&mut s, "SELECT count() FROM t")[0][0], Value::UInt(500));
        assert_eq!(vals(&mut s, "SELECT v FROM t WHERE id = 2500")[0][0], Value::Int(-1));
    }

    /// BUG 7's fix, from the mutation side. An UPDATE with no primary key to
    /// shadow by used to append the rewritten rows and leave the originals
    /// live; the delete half is a bitmap write, which needs no key at all --
    /// a row's identity is its position.
    ///
    /// The duplicate `id` is deliberate: without a unique key an UPDATE has to
    /// rewrite *every* matching row, so a fix that quietly reintroduced
    /// key semantics would collapse these two into one and be caught here.
    #[test]
    fn an_unkeyed_update_replaces_its_rows_instead_of_duplicating_them() {
        let mut s = Session::in_memory();
        s.execute("CREATE TABLE t (id UInt64, v UInt64) ENGINE = MergeTree ORDER BY tuple()")
            .unwrap();
        s.execute("INSERT INTO t VALUES (1,10),(2,20),(2,21),(3,30)").unwrap();
        s.execute("UPDATE t SET v = 99 WHERE id = 2").unwrap();
        assert_eq!(vals(&mut s, "SELECT count() FROM t")[0][0], Value::UInt(4));
        assert_eq!(
            vals(&mut s, "SELECT id, v FROM t ORDER BY id, v"),
            vec![
                vec![Value::UInt(1), Value::UInt(10)],
                vec![Value::UInt(2), Value::UInt(99)],
                vec![Value::UInt(2), Value::UInt(99)],
                vec![Value::UInt(3), Value::UInt(30)],
            ]
        );
        // ...and the delete half on its own.
        s.execute("DELETE FROM t WHERE id = 2").unwrap();
        assert_eq!(vals(&mut s, "SELECT count() FROM t")[0][0], Value::UInt(2));
        s.execute("DELETE FROM t").unwrap();
        assert_eq!(vals(&mut s, "SELECT count() FROM t")[0][0], Value::UInt(0));
    }

    /// BUG 8's fix. The predicate used to be bound against a synthesized
    /// select list, where this dialect lets `WHERE` see an alias -- so
    /// `WHERE id = 1` became `WHERE id + 10 = 1` and matched nothing.
    /// `Binder::bind_update` binds it against the table's own scope, which has
    /// no select list in it to shadow anything.
    #[test]
    fn an_update_may_assign_the_column_its_predicate_reads() {
        let mut s = Session::in_memory();
        s.execute(
            "CREATE TABLE t (id Int64, v Int64) ENGINE = MergeTree PRIMARY KEY id ORDER BY id",
        )
        .unwrap();
        s.execute("INSERT INTO t VALUES (1,10),(2,20)").unwrap();
        s.execute("UPDATE t SET id = id + 10 WHERE id = 1").unwrap();
        assert_eq!(
            vals(&mut s, "SELECT id, v FROM t ORDER BY id"),
            vec![
                vec![Value::Int(2), Value::Int(20)],
                vec![Value::Int(11), Value::Int(10)],
            ],
            "the old row must be gone, not living beside its replacement"
        );
        // Assignments are one simultaneous projection, not a sequence.
        s.execute("UPDATE t SET id = v, v = id WHERE id = 11").unwrap();
        assert_eq!(
            vals(&mut s, "SELECT id, v FROM t ORDER BY id"),
            vec![
                vec![Value::Int(2), Value::Int(20)],
                vec![Value::Int(10), Value::Int(11)],
            ]
        );
    }

    /// `affected` is the rows the statement changed, which for a delete is the
    /// rows it *newly* hid. Counting matches instead would report work the
    /// table did not do the second time a predicate is run.
    #[test]
    fn a_delete_reports_only_the_rows_it_newly_hid() {
        let mut s = Session::in_memory();
        s.execute(
            "CREATE TABLE t (id UInt64, v Int64) ENGINE = MergeTree PRIMARY KEY id ORDER BY id",
        )
        .unwrap();
        s.execute("INSERT INTO t VALUES (1,1),(2,2),(3,3),(4,4)").unwrap();
        assert_eq!(s.query("DELETE FROM t WHERE id <= 2").unwrap().affected, Some(2));
        assert_eq!(s.query("DELETE FROM t WHERE id <= 2").unwrap().affected, Some(0));
        assert_eq!(s.query("DELETE FROM t WHERE id > 99").unwrap().affected, Some(0));
        // A predicate the optimizer proves can never be TRUE never reaches
        // storage at all -- the source folds to `Empty`.
        assert_eq!(s.query("DELETE FROM t WHERE 1 = 0").unwrap().affected, Some(0));
        assert_eq!(vals(&mut s, "SELECT id FROM t ORDER BY id").len(), 2);
    }

    /// The only way to see from outside that a mutation gets a SELECT's
    /// planning. `prewhere=` is the pushdown and `zonemap=` the granule
    /// pruning, both produced by `optimizer::optimize` over the plan
    /// `Binder::bind_delete` built -- not by anything mutation-specific.
    #[test]
    fn explain_over_a_mutation_shows_the_planned_predicate() {
        let mut s = Session::in_memory();
        s.execute(
            "CREATE TABLE t (id UInt64, a UInt64, b String) \
             ENGINE = MergeTree PRIMARY KEY id ORDER BY id",
        )
        .unwrap();
        let plan = |s: &mut Session, sql: &str| {
            s.query(sql)
                .unwrap()
                .to_values()
                .iter()
                .map(|r| r[0].to_string())
                .collect::<Vec<_>>()
                .join("\n")
        };

        let d = plan(&mut s, "EXPLAIN DELETE FROM t WHERE id = 5");
        assert!(d.starts_with("'Delete default.t"), "{d}");
        assert!(d.contains("prewhere=(id#0 = 5)"), "{d}");
        assert!(d.contains("zonemap=1"), "{d}");
        // A delete needs no row value, only which rows -- so it projects the
        // predicate's columns and nothing else.
        assert!(d.contains("Scan default.t [id]"), "{d}");

        let d = plan(&mut s, "EXPLAIN DELETE FROM t WHERE b = 'q'");
        assert!(d.contains("Scan default.t [b]"), "{d}");

        // An update's own source is the replacement row, so it reads the whole
        // table -- but its predicate is pushed just the same.
        let u = plan(&mut s, "EXPLAIN UPDATE t SET a = a + 1 WHERE id > 100");
        assert!(u.starts_with("'Update default.t"), "{u}");
        assert!(u.contains("prewhere=(id#0 > 100)"), "{u}");
        assert!(u.contains("Scan default.t [id, a, b]"), "{u}");
    }

    /// Inverted. This test used to assert that a logging session **refuses**
    /// an unkeyed mutation, because a positional sweep has no write-ahead
    /// representation. That was true of the log and false of durability as a
    /// whole: `apply_sweep` now makes such a statement durable by folding the
    /// table's parts to disk at COMMIT (see [`Session::fold_to_parts`]), so
    /// the most ordinary persistent shape -- `ENGINE = MergeTree ORDER BY id`,
    /// no PRIMARY KEY -- can be mutated at all. What the test pins now is that
    /// the mutation lands *and survives a reopen with no explicit checkpoint*.
    #[test]
    fn an_unkeyed_mutation_is_durable_on_a_logging_session() {
        let s = Scratch::new("session-unkeyed-mutation");
        {
            let mut db = Session::open(s.path()).unwrap();
            db.execute("CREATE TABLE u (id UInt64, v Int64) ENGINE = MergeTree ORDER BY tuple()")
                .unwrap();
            db.execute("INSERT INTO u VALUES (1,10),(2,20)").unwrap();
            db.execute("DELETE FROM u WHERE id = 1").unwrap();
            assert_eq!(vals(&mut db, "SELECT count() FROM u")[0][0], Value::UInt(1));
            db.execute("UPDATE u SET v = 99 WHERE id = 2").unwrap();
            assert_eq!(vals(&mut db, "SELECT v FROM u WHERE id = 2")[0][0], Value::Int(99));

            // With a key, the same shapes work through the log instead.
            db.execute(
                "CREATE TABLE k (id UInt64, v Int64) ENGINE = MergeTree PRIMARY KEY id ORDER BY id",
            )
            .unwrap();
            db.execute("INSERT INTO k VALUES (1,10),(2,20),(3,30)").unwrap();
            db.execute("DELETE FROM k WHERE id = 2").unwrap();
        }
        // No checkpoint, no clean shutdown: whatever survives is what the
        // statements themselves made durable.
        let mut db = Session::open(s.path()).unwrap();
        assert_eq!(vals(&mut db, "SELECT count() FROM u")[0][0], Value::UInt(1));
        assert_eq!(vals(&mut db, "SELECT v FROM u WHERE id = 2")[0][0], Value::Int(99));
        assert_eq!(vals(&mut db, "SELECT count() FROM k")[0][0], Value::UInt(2));
    }

    /// The bulk sweep still has to be replayable: a crash with no checkpoint
    /// leaves only the log, and every row the statement hid has to be in it.
    #[test]
    fn a_bulk_delete_survives_a_crash_without_a_checkpoint() {
        let s = Scratch::new("session-bulk-delete-crash");
        {
            let mut db = Session::open(s.path()).unwrap();
            db.execute(
                "CREATE TABLE t (id UInt64, v Int64) \
                 ENGINE = MergeTree PRIMARY KEY id ORDER BY id",
            )
            .unwrap();
            let rows: Vec<String> = (0..2_000u64).map(|i| format!("({i},{i})")).collect();
            db.execute(&format!("INSERT INTO t VALUES {}", rows.join(","))).unwrap();
            db.checkpoint().unwrap();
            db.execute("DELETE FROM t WHERE id < 500").unwrap();
            db.execute("UPDATE t SET id = id + 10000 WHERE id = 1000").unwrap();
            // ...and the process dies.
        }
        let mut db = Session::open(s.path()).unwrap();
        assert_eq!(vals(&mut db, "SELECT count() FROM t")[0][0], Value::UInt(1_500));
        assert!(db.query("SELECT v FROM t WHERE id = 100").unwrap().is_empty());
        assert!(db.query("SELECT v FROM t WHERE id = 1000").unwrap().is_empty());
        assert_eq!(vals(&mut db, "SELECT v FROM t WHERE id = 11000")[0][0], Value::Int(1_000));
    }

    // ---- the unkeyed sweep's durable row identity -------------------------

    /// Every part file in `db.t`, by name.
    fn part_files(root: &Path, table: &str) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(root.join("default").join(table))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| crate::persist::store::parse_part_seq(n).is_some())
            .collect();
        v.sort();
        v
    }

    fn log_bytes(root: &Path, table: &str) -> u64 {
        std::fs::read_dir(root.join(".wal").join("default").join(table))
            .map(|rd| rd.map(|e| e.unwrap().metadata().unwrap().len()).sum())
            .unwrap_or(0)
    }

    /// The claim the whole design exists to make: hiding a row that lives in a
    /// checkpointed part costs a log record, not a table rewrite.
    ///
    /// The part file is the observable. `write_table` only ever *adds* a file
    /// -- it never edits one in place -- so an unchanged listing is proof no
    /// checkpoint ran, and the rows still being gone after a reopen with no
    /// clean shutdown is proof the log carried them.
    #[test]
    fn an_unkeyed_delete_of_checkpointed_rows_logs_instead_of_rewriting() {
        let s = Scratch::new("session-mask-no-rewrite");
        let before;
        {
            let mut db = Session::open(s.path()).unwrap();
            db.execute("CREATE TABLE u (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id")
                .unwrap();
            let rows: Vec<String> = (0..2_000u64).map(|i| format!("({i},{i})")).collect();
            db.execute(&format!("INSERT INTO u VALUES {}", rows.join(","))).unwrap();
            db.checkpoint().unwrap();
            before = part_files(s.path(), "u");
            assert_eq!(before.len(), 1, "one part after the checkpoint");

            db.execute("DELETE FROM u WHERE id % 100 = 7").unwrap();
            db.execute("UPDATE u SET v = -1 WHERE id = 3").unwrap();
            assert_eq!(
                part_files(s.path(), "u"),
                before,
                "an unkeyed DELETE and UPDATE over checkpointed rows must not rewrite the \
                 table"
            );
            // ...and the process dies with no checkpoint and no shutdown.
        }
        let mut db = Session::open(s.path()).unwrap();
        assert_eq!(vals(&mut db, "SELECT count() FROM u")[0][0], Value::UInt(1_980));
        assert!(db.query("SELECT v FROM u WHERE id = 107").unwrap().is_empty());
        assert_eq!(vals(&mut db, "SELECT v FROM u WHERE id = 3")[0][0], Value::Int(-1));
        assert_eq!(vals(&mut db, "SELECT v FROM u WHERE id = 4")[0][0], Value::Int(4));
    }

    /// The residual, pinned rather than hidden: rows that reached a part only
    /// since the last checkpoint have no file to be named in, so hiding them
    /// still writes the table out. This is the case `TAG_FLUSH`/`TAG_MERGE`
    /// would close and this wave does not.
    #[test]
    fn an_unkeyed_delete_of_rows_inserted_in_the_same_transaction_still_folds() {
        let s = Scratch::new("session-mask-residual");
        let mut db = Session::open(s.path()).unwrap();
        db.execute("CREATE TABLE u (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id").unwrap();
        db.execute("INSERT INTO u VALUES (1,1)").unwrap();
        db.checkpoint().unwrap();
        let before = part_files(s.path(), "u");

        db.execute("BEGIN").unwrap();
        db.execute("INSERT INTO u VALUES (2,2)").unwrap();
        db.execute("DELETE FROM u WHERE id = 2").unwrap();
        db.execute("COMMIT").unwrap();
        assert_ne!(part_files(s.path(), "u"), before, "the fresh part has no durable home");
        assert_eq!(vals(&mut db, "SELECT count() FROM u")[0][0], Value::UInt(1));
    }

    /// A rolled-back sweep must leave the log exactly as it found it -- the
    /// mask record included, since it is appended before COMMIT like every
    /// other staged record.
    #[test]
    fn a_rolled_back_unkeyed_delete_leaves_the_log_byte_identical() {
        let s = Scratch::new("session-mask-rollback");
        let mut db = Session::open(s.path()).unwrap();
        db.execute("CREATE TABLE u (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id").unwrap();
        db.execute("INSERT INTO u VALUES (1,1),(2,2),(3,3)").unwrap();
        db.checkpoint().unwrap();
        let seg = s.path().join(".wal").join("default").join("u");
        let name = std::fs::read_dir(&seg).unwrap().next().unwrap().unwrap().path();
        let img = std::fs::read(&name).unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("DELETE FROM u WHERE id = 2").unwrap();
        db.execute("ROLLBACK").unwrap();
        assert_eq!(std::fs::read(&name).unwrap(), img, "rewind must restore the log's bytes");
        assert_eq!(vals(&mut db, "SELECT count() FROM u")[0][0], Value::UInt(3));
    }

    /// Re-hiding an already-hidden row changes nothing, so it must log
    /// nothing. That is what makes replaying a record twice free, which a
    /// coarse checkpoint watermark guarantees will happen.
    #[test]
    fn deleting_the_same_unkeyed_row_twice_logs_nothing_the_second_time() {
        let s = Scratch::new("session-mask-idempotent");
        let mut db = Session::open(s.path()).unwrap();
        db.execute("CREATE TABLE u (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id").unwrap();
        db.execute("INSERT INTO u VALUES (1,1),(2,2),(3,3)").unwrap();
        db.checkpoint().unwrap();
        db.execute("DELETE FROM u WHERE id = 2").unwrap();

        // Against a control that matches nothing at all: both statements pay
        // the commit marker and the tick an autocommit transaction always
        // pays, and the question is whether the second logs a *record* on top.
        let a = log_bytes(s.path(), "u");
        db.execute("DELETE FROM u WHERE id = 999").unwrap();
        let empty = log_bytes(s.path(), "u") - a;
        let b = log_bytes(s.path(), "u");
        let rs = db.query("DELETE FROM u WHERE id = 2").unwrap();
        assert_eq!(rs.affected, Some(0), "no row was live to hide");
        assert_eq!(log_bytes(s.path(), "u") - b, empty, "and no record was written");
    }

    /// Scattered single-row deletes across many parts: every one is citable,
    /// none rewrites the table, and the visible set survives a crash. The
    /// shape the old design was worst at -- one statement straddling several
    /// parts rewrote all of them.
    #[test]
    fn a_sweep_across_many_checkpointed_parts_is_replayed_exactly() {
        let s = Scratch::new("session-mask-many-parts");
        {
            let mut db = Session::open(s.path()).unwrap();
            db.execute("CREATE TABLE u (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id")
                .unwrap();
            for g in 0..6u64 {
                let rows: Vec<String> =
                    (0..500u64).map(|i| format!("({},{i})", g * 1_000 + i)).collect();
                db.execute(&format!("INSERT INTO u VALUES {}", rows.join(","))).unwrap();
                db.checkpoint().unwrap();
            }
            let parts = part_files(s.path(), "u");
            assert!(parts.len() >= 6, "{parts:?}");
            db.execute("DELETE FROM u WHERE id % 1000 = 250").unwrap();
            assert_eq!(part_files(s.path(), "u"), parts, "six parts touched, none rewritten");
        }
        let mut db = Session::open(s.path()).unwrap();
        assert_eq!(vals(&mut db, "SELECT count() FROM u")[0][0], Value::UInt(2_994));
        assert!(db.query("SELECT v FROM u WHERE id = 3250").unwrap().is_empty());
    }

    /// The claim the design turns on: a checkpoint moves a part to a new file
    /// and the identity inside it does not move, so a citation minted before
    /// the checkpoint still resolves after it.
    ///
    /// This is the difference between naming the part *file* and naming the
    /// part. A file-sequence citation would be unresolvable the moment the
    /// first delete's rewrite lands, which -- since a clean shutdown
    /// checkpoints -- is the steady state.
    #[test]
    fn a_parts_identity_survives_the_checkpoint_that_rewrites_it() {
        let s = Scratch::new("session-mask-identity-survives");
        let pid;
        {
            let mut db = Session::open(s.path()).unwrap();
            db.execute("CREATE TABLE u (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id")
                .unwrap();
            let rows: Vec<String> = (0..500u64).map(|i| format!("({i},{i})")).collect();
            db.execute(&format!("INSERT INTO u VALUES {}", rows.join(","))).unwrap();
            db.checkpoint().unwrap();
            let first = part_files(s.path(), "u");
            pid = db.catalog.table_by_path("default.u").unwrap().snapshot().part(0).pid;

            // A delete, then the checkpoint that rewrites the part it touched.
            db.execute("DELETE FROM u WHERE id = 1").unwrap();
            db.checkpoint().unwrap();
            assert_ne!(part_files(s.path(), "u"), first, "the mask moved, so the file did");
            assert_eq!(
                db.catalog.table_by_path("default.u").unwrap().snapshot().part(0).pid,
                pid,
                "...but the identity a log record cites did not"
            );

            // A second delete cites the same identity against the new file,
            // and is a log record rather than a third rewrite.
            let second = part_files(s.path(), "u");
            db.execute("DELETE FROM u WHERE id = 2").unwrap();
            assert_eq!(part_files(s.path(), "u"), second);
        }
        let mut db = Session::open(s.path()).unwrap();
        assert_eq!(vals(&mut db, "SELECT count() FROM u")[0][0], Value::UInt(498));
        assert!(db.query("SELECT v FROM u WHERE id = 2").unwrap().is_empty());
        assert_eq!(
            db.catalog.table_by_path("default.u").unwrap().snapshot().part(0).pid,
            pid,
            "and it round-trips through the part file"
        );
    }

    /// Part file numbers and part identities are both monotone across an
    /// incarnation boundary. `TRUNCATE`, and `DROP` then `CREATE`, both empty
    /// the directory, so the listing alone would reissue `part_000001` and
    /// identity 1 to a new table while a backup still holds the old ones.
    #[test]
    fn part_numbers_and_identities_never_restart_across_an_incarnation() {
        let s = Scratch::new("session-mask-incarnation");
        let mut db = Session::open(s.path()).unwrap();
        let ddl = "CREATE TABLE u (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id";
        db.execute(ddl).unwrap();
        db.execute("INSERT INTO u VALUES (1,1),(2,2)").unwrap();
        db.checkpoint().unwrap();
        let seq = |v: &[String]| {
            v.iter().filter_map(|n| crate::persist::store::parse_part_seq(n)).max().unwrap()
        };
        let high = seq(&part_files(s.path(), "u"));
        let pid = db.catalog.table_by_path("default.u").unwrap().snapshot().part(0).pid;

        for sql in ["TRUNCATE TABLE u", "DROP TABLE u"] {
            db.execute(sql).unwrap();
            if sql.starts_with("DROP") {
                db.execute(ddl).unwrap();
            }
            db.execute("INSERT INTO u VALUES (9,9)").unwrap();
            db.checkpoint().unwrap();
            assert!(
                seq(&part_files(s.path(), "u")) > high,
                "{sql} reissued a part file number"
            );
            assert!(
                db.catalog.table_by_path("default.u").unwrap().snapshot().part(0).pid > pid,
                "{sql} reissued a part identity"
            );
        }
    }

    /// `TRUNCATE` empties a table; it must not move it to another database.
    ///
    /// `Catalog::create_table` stores the *bare* name, so recreating from the
    /// stored definition put the table in whatever database was current --
    /// `TRUNCATE TABLE m.u` left a `default.u` behind and destroyed `m.u` with
    /// its rows, silently, exit 0.
    #[test]
    fn truncate_keeps_the_table_in_its_own_database() {
        let s = Scratch::new("session-truncate-db");
        let mut db = Session::open(s.path()).unwrap();
        db.execute("CREATE DATABASE m").unwrap();
        db.execute("CREATE TABLE m.u (id UInt64) ENGINE = MergeTree ORDER BY id").unwrap();
        db.execute("INSERT INTO m.u VALUES (1),(2)").unwrap();
        db.execute("TRUNCATE TABLE m.u").unwrap();
        assert_eq!(vals(&mut db, "SELECT count() FROM m.u")[0][0], Value::UInt(0));
        assert!(
            db.query("SELECT count() FROM default.u").is_err(),
            "TRUNCATE must not create a table in another database"
        );
    }

    /// A citation that does not resolve is corruption, not something to skip.
    ///
    /// Skipping would resurrect a row that was deleted and acknowledged --
    /// silently, and only on the machine that crashed -- which is the exact
    /// failure the positional record was refused for in the first place. Both
    /// ways a citation can fail to resolve get a case, and the assertion is on
    /// the *refusal* so it cannot quietly become a `continue`.
    #[test]
    fn a_mask_record_that_does_not_resolve_is_refused() {
        for (bogus_pid, pos, why) in [(true, 0u64, "does not hold"), (false, 9_999, "rows")] {
            let s = Scratch::new("session-mask-corrupt");
            let pid;
            {
                let mut db = Session::open(s.path()).unwrap();
                db.execute("CREATE TABLE u (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id")
                    .unwrap();
                db.execute("INSERT INTO u VALUES (1,1),(2,2),(3,3)").unwrap();
                db.checkpoint().unwrap();
                let real = db.catalog.table_by_path("default.u").unwrap().snapshot().part(0).pid;
                pid = if bogus_pid { real + 9_999 } else { real };
            }
            let dir = crate::persist::wal::wal_dir(s.path(), "default", "u");
            let mut w = crate::persist::Wal::open(&dir).unwrap();
            let mut m = MaskRuns::default();
            m.hide(pid, pos);
            w.append_masks(None, &m).unwrap();
            w.sync().unwrap();
            drop(w);

            let mut db = Session::open(s.path()).unwrap();
            let e = db.query("SELECT count() FROM u").unwrap_err();
            assert!(e.to_string().contains(why), "pid {pid} pos {pos}: {e}");
        }
    }

    /// Replay must not compact: a merge retires the part identity a later
    /// record cites, and the two are not in step. Enough flushes to cross
    /// `AUTO_COMPACT_PARTS`, with a mask record behind them naming a part from
    /// before the first.
    ///
    /// **MEASURED, AND THIS TEST DOES NOT CURRENTLY DISTINGUISH THE TWO
    /// ENGINES.** With the `if self.replaying { return Ok(()) }` guard deleted
    /// from `maybe_auto_compact`, this test still passes, and so does a
    /// 12-trial `kill -9` campaign built to provoke it (14 checkpointed parts,
    /// deletes against them interleaved with bulk inserts that pack a part
    /// each). Instrumenting the merge showed why: the compaction *does* fire
    /// during replay without the guard -- observed `replaying=true merging 8
    /// of 16 parts` -- but it takes the smallest parts, and the parts it took
    /// were the ones replay had just built, never a cited one.
    ///
    /// The reason looks structural rather than lucky. Replay's part list is
    /// the checkpointed one plus whatever the `Insert` records rebuild, and
    /// runtime flushes strictly more often than replay does -- every `DELETE`
    /// goes through `plan_sweep`'s `flush_all` and replay's mask arm flushes
    /// nothing -- so runtime reaches any given part count at or before the
    /// point replay does. Runtime therefore compacts first, and a runtime
    /// merge retires the part's durable home, which makes the next sweep over
    /// those rows `dark` and stops the citation being minted at all.
    ///
    /// So the guard is defence in depth against a hazard nobody has yet
    /// reached, not a fix for a reproduced one. Keep it -- it is one branch on
    /// a path that runs once per open, and the failure it guards is a silently
    /// resurrected row -- but do not read this test as evidence that removing
    /// it breaks anything, because it is not.
    #[test]
    fn replay_does_not_compact_away_a_part_a_later_record_cites() {
        let s = Scratch::new("session-mask-replay-compact");
        {
            let mut db = Session::open(s.path()).unwrap();
            db.execute("CREATE TABLE u (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id")
                .unwrap();
            db.execute("INSERT INTO u VALUES (1,1),(2,2),(3,3)").unwrap();
            db.checkpoint().unwrap();
            // Twenty logged inserts, each of which becomes a part of its own on
            // replay -- past `AUTO_COMPACT_PARTS` twice over.
            for i in 0..20u64 {
                db.execute(&format!("INSERT INTO u VALUES ({}, {i})", 100 + i)).unwrap();
                db.execute("SELECT count() FROM u").unwrap();
            }
            // Cites the part written by the checkpoint above, which sits at
            // index 0 and is what an unsuppressed compaction would swallow.
            db.execute("DELETE FROM u WHERE id = 2").unwrap();
        }
        let mut db = Session::open(s.path()).unwrap();
        assert_eq!(vals(&mut db, "SELECT count() FROM u")[0][0], Value::UInt(22));
        assert!(db.query("SELECT v FROM u WHERE id = 2").unwrap().is_empty());
    }

}
