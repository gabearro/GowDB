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
//! `COMMIT` fsyncs a commit marker into each log and then stores each overlay
//! over its published set -- durability first, visibility second, and the
//! second half cannot fail. `ROLLBACK` drops the overlays and rewinds each log
//! to its enlistment LSN, which leaves both memory and disk exactly as they
//! were.
//!
//! Enlistment is **lazy**: `BEGIN` writes one `Option` and touches no table, so
//! a transaction that only reads costs nothing, and a transaction over one
//! table does not drag the others into it.
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
//! ### The multi-table caveat
//!
//! Logs are per table, so a transaction spanning N tables writes N commit
//! markers and fsyncs N files. A crash *between* those fsyncs can leave a
//! prefix of them durable, which commits some tables and not others. Making
//! that atomic needs one log for the whole database rather than one per table;
//! it is the right fix and it is not in this change. A single-table
//! transaction -- one marker, one fsync -- is fully atomic, and that is the
//! shape the OLTP path actually has.

use std::fs::File;
use std::path::Path;
use std::time::Instant;

use crate::catalog::Catalog;
use crate::common::{Error, Result};
use crate::exec::operators;
use crate::planner::{
    binder::Binder,
    logical::{BoundExpr, LogicalPlan, ZoneFilter},
    optimizer,
};
use crate::sql::ast::{
    ColumnDef, CreateTable, ExplainKind, Insert, InsertSource, ObjectName, Statement,
};
use crate::sql::parse;
use crate::types::{Block, Column, ColumnBuilder, DataType, Field, Schema, TableDef, Value};

#[derive(Debug, Default, Clone, Copy)]
pub struct QueryStats {
    pub rows: usize,
    pub elapsed_us: u128,
    pub granules_read: u64,
    pub granules_pruned: u64,
    pub rows_scanned: u64,
}

/// A materialized result. Small by construction: anything large should be
/// streamed through [`operators::build`] instead.
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
        let rows = self.to_values();
        let ncols = self.schema.len();
        let mut widths: Vec<usize> = self
            .schema
            .fields()
            .iter()
            .map(|f| f.name.chars().count())
            .collect();
        let cells: Vec<Vec<String>> = rows
            .iter()
            .map(|r| {
                (0..ncols)
                    .map(|c| r.get(c).map(|v| v.render_plain()).unwrap_or_default())
                    .collect()
            })
            .collect();
        for r in &cells {
            for (c, s) in r.iter().enumerate() {
                widths[c] = widths[c].max(s.chars().count());
            }
        }
        let rule =
            |f: &mut std::fmt::Formatter<'_>, l: &str, m: &str, r: &str| -> std::fmt::Result {
                write!(f, "{l}")?;
                for (i, w) in widths.iter().enumerate() {
                    if i > 0 {
                        write!(f, "{m}")?;
                    }
                    write!(f, "{}", "─".repeat(w + 2))?;
                }
                writeln!(f, "{r}")
            };
        rule(f, "┌", "┬", "┐")?;
        write!(f, "│")?;
        for (i, fl) in self.schema.fields().iter().enumerate() {
            write!(f, " {:w$} │", fl.name, w = widths[i])?;
        }
        writeln!(f)?;
        rule(f, "├", "┼", "┤")?;
        for r in &cells {
            write!(f, "│")?;
            for (i, s) in r.iter().enumerate() {
                write!(f, " {:w$} │", s, w = widths[i])?;
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
}

/// Transaction control. Not in the SQL grammar -- `src/sql` is not this
/// module's to extend -- so [`Session::run`] recognises it ahead of the parser.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TxnStmt {
    Begin,
    Commit,
    Rollback,
}

impl Session {
    pub fn in_memory() -> Session {
        Session {
            catalog: Catalog::in_memory(),
            wals: Default::default(),
            wal_enabled: false,
            // Nothing on disk to guard: an in-memory session shares no files
            // with anyone, so it must never take (or contend for) the lock.
            _lock: None,
            txn: None,
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
        let lock = lock_data_dir(&root)?;
        crate::persist::load_catalog(&mut catalog)?;
        Ok(Session {
            catalog,
            wals: Default::default(),
            wal_enabled: true,
            _lock: lock,
            txn: None,
        })
    }

    /// Turn write-ahead logging off for this session.
    ///
    /// Bulk loading is the case that wants this: an `fsync` per statement is
    /// the dominant cost when the whole job is "ingest a billion rows and
    /// checkpoint once", and the log buys nothing if you would re-run the load
    /// after a crash anyway. Writes are then durable only at
    /// [`Session::checkpoint`].
    pub fn set_wal_enabled(&mut self, on: bool) {
        self.wal_enabled = on && self.catalog.is_persistent();
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
        // Drop the cached handles first: `save_catalog` truncates each log and
        // resets its watermark, which would leave a cached `Wal`'s idea of the
        // file length stale and make the next append land at the wrong offset.
        self.wals.clear();
        crate::persist::save_catalog(&mut self.catalog)
    }

    /// Open (or reuse) the cached log handle for `path`.
    fn wal_for(&mut self, path: &str) -> Result<Option<&mut crate::persist::Wal>> {
        if !self.wal_enabled {
            return Ok(None);
        }
        let Some(root) = self.catalog.dir() else { return Ok(None) };
        if !self.wals.contains_key(path) {
            let (db, tbl) = path.split_once('.').unwrap_or(("default", path));
            let p = root.join(db).join(tbl).join(crate::persist::store::WAL_FILE);
            self.wals
                .insert(path.to_string(), crate::persist::Wal::open(&p)?);
        }
        Ok(Some(self.wals.get_mut(path).expect("just inserted")))
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
        let Some(w) = self.wal_for(path)? else { return Ok(()) };
        match seq {
            Some(s) => w.append_insert_staged(s, b).map(|_| ()),
            None => {
                w.append_insert(b)?;
                w.sync()
            }
        }
    }

    /// Append one key delete per lane, enlisting once for the batch.
    ///
    /// The delete counterpart to [`Session::log_insert`], with the same two
    /// durability rules -- and one more that only a bulk statement needs: the
    /// enlistment and the log-handle lookup are hoisted out of the loop. Doing
    /// them per record costs a linear scan of the transaction's table list and
    /// a string compare, which was free when a statement logged one record and
    /// is not when it logs a million.
    fn log_deletes(&mut self, path: &str, lanes: &[u64]) -> Result<()> {
        let seq = self.enlist(path)?;
        let Some(w) = self.wal_for(path)? else { return Ok(()) };
        match seq {
            Some(s) => {
                for &l in lanes {
                    w.append_delete_staged(s, l)?;
                }
                Ok(())
            }
            None => {
                for &l in lanes {
                    w.append_delete(l)?;
                }
                w.sync()
            }
        }
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
    pub fn begin(&mut self) -> Result<()> {
        if self.txn.is_some() {
            return Err(Error::unsupported(
                "a transaction is already open; nested transactions are not supported",
            ));
        }
        self.txn = Some(Txn::default());
        Ok(())
    }

    /// Make the transaction's writes durable and then visible, in that order.
    ///
    /// A failure anywhere in the durable half rolls the whole thing back, so
    /// COMMIT either happens or does not -- it never half-happens and then
    /// reports an error over a table that has already moved.
    pub fn commit(&mut self) -> Result<()> {
        // Taken up front so the durable half can borrow the roster while it
        // holds `&mut self` for the catalog and the logs -- the alternative is
        // a `Vec` of cloned paths per COMMIT, and a transaction is allowed to
        // be as small as one statement.
        let Some(txn) = self.txn.take() else {
            return Err(Error::exec("COMMIT without an open transaction"));
        };
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

    /// The fallible half of COMMIT: flush the buffered rows into each overlay,
    /// then fsync a commit marker into each enlisted log.
    ///
    /// Durability strictly before visibility. Nothing has been published when
    /// this returns -- the overlays are still private -- so an error here is
    /// undone by dropping them.
    fn commit_durable(&mut self, tables: &[Enlisted]) -> Result<()> {
        for e in tables {
            self.catalog.table_by_path_mut(&e.path)?.flush()?;
        }
        for e in tables {
            let Some(seq) = e.seq else { continue };
            let w = self
                .wals
                .get_mut(&e.path)
                .expect("an enlisted table with a sequence number has an open log");
            w.commit(seq)?;
            w.sync()?;
        }
        Ok(())
    }

    /// Discard the transaction, in memory and on disk.
    ///
    /// Dropping an overlay is a pointer store -- parts are immutable, so
    /// nothing has to be un-written -- and rewinding each log to the LSN the
    /// table was enlisted at leaves the file byte-identical to its
    /// pre-transaction state. Replay would have dropped those staged records
    /// anyway; the rewind is what makes "no trace" true of the disk too.
    pub fn rollback(&mut self) -> Result<()> {
        let Some(txn) = self.txn.take() else {
            return Err(Error::exec("ROLLBACK without an open transaction"));
        };
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
            .push(Enlisted { path: path.to_string(), seq, lsn });
        Ok(seq)
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
        let stmts = parse(sql)?;
        let mut out = Vec::with_capacity(stmts.len());
        for s in &stmts {
            out.push(self.exec_statement(s)?);
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
            match txn_stmt(span) {
                Some(t) => {
                    let t0 = Instant::now();
                    match t {
                        TxnStmt::Begin => self.begin()?,
                        TxnStmt::Commit => self.commit()?,
                        TxnStmt::Rollback => self.rollback()?,
                    }
                    let mut rs = ResultSet::empty();
                    rs.stats.elapsed_us = t0.elapsed().as_micros();
                    out.push(rs);
                }
                None => {
                    // The statement's own text, from its first token to the
                    // semicolon that ended it (or the end of the input).
                    let end = if i == toks.len() { sql.len() } else { toks[i].pos };
                    for s in &parse(&sql[span[0].pos..end])? {
                        out.push(self.exec_statement(s)?);
                    }
                }
            }
        }
        Ok(out)
    }

    fn exec_statement(&mut self, stmt: &Statement) -> Result<ResultSet> {
        let t0 = Instant::now();
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
        let mut rs = match stmt {
            Statement::Query(q) => self.run_query(q)?,
            Statement::Insert(i) => self.run_insert(i)?,
            Statement::CreateTable(c) => self.run_create_table(c)?,
            Statement::CreateDatabase { name, if_not_exists } => {
                self.catalog.create_database(name, *if_not_exists)?;
                ResultSet::empty()
            }
            Statement::DropTable { name, if_exists } => {
                self.catalog.drop_table(name, *if_exists)?;
                ResultSet::empty()
            }
            Statement::DropDatabase { name, if_exists } => {
                self.catalog.drop_database(name, *if_exists)?;
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
                let def = self.catalog.table(table)?.def.clone();
                self.catalog.drop_table(table, false)?;
                self.catalog.create_table(def, false)?;
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
            Statement::ShowDatabases => {
                ResultSet::one_string_column("name", self.catalog.database_names())?
            }
            Statement::ShowTables { database } => ResultSet::one_string_column(
                "name",
                self.catalog.table_names(database.as_deref())?,
            )?,
            Statement::ShowCreateTable(name) => {
                let t = self.catalog.table(name)?;
                let ddl = render_create_table(t.schema(), &t.def);
                ResultSet::one_string_column("statement", vec![ddl])?
            }
            Statement::Describe(name) => {
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
                ResultSet::from_rows(schema, rows)?
            }
            Statement::Explain { kind, statement } => self.run_explain(*kind, statement)?,
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

    // --------------------------------------------------------------- queries

    fn plan(&mut self, q: &crate::sql::ast::Query) -> Result<LogicalPlan> {
        // Scans read parts, not the write buffer, so everything buffered has to
        // land in a part first. See the storage::table module docs for why this
        // beats teaching every operator to merge a hash map.
        self.catalog.flush_all()?;
        let q = self.resolve_subqueries(q)?;
        let plan = Binder::new(&self.catalog).bind_query(&q)?;
        optimizer::optimize(plan)
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
    fn resolve_subqueries(&mut self, q: &crate::sql::ast::Query) -> Result<crate::sql::ast::Query> {
        let mut out = q.clone();
        let mut budget = 64usize;
        self.rewrite_query(&mut out, &mut budget)?;
        Ok(out)
    }

    fn rewrite_query(&mut self, q: &mut crate::sql::ast::Query, budget: &mut usize) -> Result<()> {
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

    fn rewrite_setexpr(&mut self, s: &mut crate::sql::ast::SetExpr, budget: &mut usize) -> Result<()> {
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

    fn rewrite_tableref(&mut self, t: &mut crate::sql::ast::TableRef, budget: &mut usize) -> Result<()> {
        use crate::sql::ast::{JoinConstraint, TableRef};
        match t {
            TableRef::Table { .. } => {}
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

    fn rewrite_expr(&mut self, e: &mut crate::sql::ast::Expr, budget: &mut usize) -> Result<()> {
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
            Expr::Subquery(q) => {
                let vals = self.eval_subquery(q, budget, "scalar subquery", 1)?;
                *e = Expr::Literal(vals.into_iter().next().unwrap_or(Value::Null));
            }
            Expr::InSubquery { expr, subquery, negated } => {
                self.rewrite_expr(expr, budget)?;
                let vals = self.eval_subquery(subquery, budget, "IN (SELECT ...)", usize::MAX)?;
                *e = Expr::InList {
                    expr: expr.clone(),
                    list: vals.into_iter().map(Expr::Literal).collect(),
                    negated: *negated,
                };
            }
            Expr::Exists { subquery, negated } => {
                let vals = self.eval_subquery(subquery, budget, "EXISTS", usize::MAX)?;
                *e = Expr::Literal(Value::Bool(!vals.is_empty() != *negated));
            }
        }
        Ok(())
    }

    /// Run a subquery and return column 0. `max_rows` caps a scalar subquery
    /// at one row, per SQL semantics.
    fn eval_subquery(
        &mut self,
        q: &crate::sql::ast::Query,
        budget: &mut usize,
        what: &str,
        max_rows: usize,
    ) -> Result<Vec<Value>> {
        if *budget == 0 {
            return Err(Error::unsupported(format!(
                "{what}: subquery nesting is too deep"
            )));
        }
        *budget -= 1;

        let plan = self.plan(q).map_err(|e| match e {
            // A correlated subquery cannot bind on its own, and the resulting
            // "unknown column" is a confusing way to say so.
            Error::Bind(m) => Error::unsupported(format!(
                "{what}: correlated subqueries are not supported ({m})"
            )),
            other => other,
        })?;
        if plan.schema().len() != 1 {
            return Err(Error::bind(format!(
                "{what} must select exactly one column, got {}",
                plan.schema().len()
            )));
        }
        let blocks = operators::execute(&plan, &self.catalog)?;
        let mut out = Vec::new();
        for b in &blocks {
            for r in 0..b.rows() {
                out.push(b.column(0).value(r));
            }
        }
        if max_rows == 1 && out.len() > 1 {
            return Err(Error::exec(format!(
                "{what} returned {} rows, expected at most 1",
                out.len()
            )));
        }
        Ok(out)
    }

    fn run_query(&mut self, q: &crate::sql::ast::Query) -> Result<ResultSet> {
        let plan = self.plan(q)?;
        let schema = plan.schema().clone();
        // Through the exchange, which decides per query whether to go parallel
        // (see `exchange::degree`) and falls back to the serial pipeline below
        // its row threshold. Identical signature and return type to the serial
        // entry point it replaced -- the decision is in the operator, not here.
        let (blocks, st) = crate::exec::execute_parallel_stats(&plan, &self.catalog)?;
        let rows = blocks.iter().map(|b| b.rows()).sum();
        Ok(ResultSet {
            schema,
            blocks,
            stats: QueryStats {
                rows,
                elapsed_us: 0,
                granules_read: st.granules_read,
                granules_pruned: st.granules_pruned,
                rows_scanned: st.rows_read,
            },
            affected: None,
        })
    }

    fn run_explain(&mut self, kind: ExplainKind, stmt: &Statement) -> Result<ResultSet> {
        let text = match (kind, stmt) {
            (ExplainKind::Ast, s) => format!("{s:#?}"),
            // PIPELINE renders the *physical* plan, which is the only place the
            // access path is visible: whether a predicate on the key lowered to
            // an index probe or stayed a scan is a physical decision, and PLAN
            // shows the logical tree where that choice does not exist yet.
            // Without this, index selection is unprovable from the outside.
            (ExplainKind::Pipeline, Statement::Query(q)) => {
                let logical = self.plan(q)?;
                crate::planner::physical::lower(&logical, &self.catalog)?.explain()
            }
            (_, Statement::Query(q)) => self.plan(q)?.explain(),
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
                operators::execute(&plan, &self.catalog)?
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
        let mut n = 0;
        for b in blocks {
            if b.rows() == 0 {
                continue;
            }
            let full = self.widen_to_schema(b, target, order)?;
            // Log-before-apply: the record is durable before the write is
            // acknowledged, so a crash between the two replays the insert
            // rather than losing it. Inside a transaction it is staged
            // instead, and the commit marker is what makes it replayable.
            self.log_insert(path, &full)?;
            n += self.catalog.table_by_path_mut(path)?.insert(full)?;
        }
        Ok(n)
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
        let blocks = operators::execute(&m.source, &self.catalog)?;
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
            s.update_blocks(&path, blocks)
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

    /// Apply one bulk delete: hide the rows, log the keys, one publish.
    fn apply_sweep(&mut self, path: &str, sweep: &Sweep) -> Result<usize> {
        // Refused before anything is touched. A positional delete has no
        // write-ahead representation -- the log's only delete record names a
        // primary-key *lane* (`Wal::append_delete`), and a table with no
        // single-column key has no lane to name. In memory that is fine, there
        // being no log and nothing to recover; on a logging session,
        // acknowledging it would mean a crash silently resurrects the rows,
        // and for an UPDATE replays the append without its tombstone and
        // duplicates them. Refused rather than corrupted.
        if self.wal_enabled && self.catalog.table_by_path(path)?.pk_col().is_none() {
            return Err(Error::unsupported(format!(
                "`{path}` has no single-column primary key, so its deleted rows cannot be \
                 written to the log; the mutation would not survive a crash. Add \
                 `PRIMARY KEY <col>`, or run it on an in-memory session"
            )));
        }
        // Enlisted unconditionally, and this is load-bearing:
        // `Table::edit`/`publish` redirect into the transaction's private
        // overlay only once `begin_txn` has run on that table, and `enlist` is
        // what runs it. Without this an in-memory session -- which logs
        // nothing, so never reached `enlist` on this path -- would sweep
        // straight into the committed set and ROLLBACK would have nothing to
        // drop.
        self.enlist(path)?;
        // Only a logging session needs the lanes, and asking for them costs a
        // packed-lane read per hidden row plus a `Vec` that grows to the
        // affected count. An in-memory delete of a million rows should pay
        // neither, so the sink is `None` unless there is a log to feed; the
        // guard above has already established that a log implies a key.
        let mut keys = Vec::new();
        let n = self.catalog.table_by_path_mut(path)?.delete_where_keys(
            &sweep.projection,
            sweep.pred.as_ref(),
            &sweep.zone,
            self.wal_enabled.then_some(&mut keys),
        )?;
        if !keys.is_empty() {
            self.log_deletes(path, &keys)?;
        }
        Ok(n)
    }

    fn update_blocks(&mut self, path: &str, blocks: Vec<Block>) -> Result<usize> {
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
        // An UPDATE is a re-insert of the changed rows, so logging the insert
        // is enough to replay it: the primary key makes it idempotent, and
        // where there is no key the sweep's tombstones were logged just above.
        self.log_insert(path, &acc)?;
        let t = self.catalog.table_by_path_mut(path)?;
        let n = t.insert(acc)?;
        t.flush()?;
        Ok(n)
    }

    // ------------------------------------------------------------------ DDL

    fn run_create_table(&mut self, c: &CreateTable) -> Result<ResultSet> {
        // CREATE TABLE ... AS SELECT takes its schema from the query.
        let (fields, as_blocks) = match &c.as_query {
            Some(q) => {
                let plan = self.plan(q)?;
                let s = plan.schema().clone();
                let blocks = operators::execute(&plan, &self.catalog)?;
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
        let partition_by = match &c.partition_by {
            Some(e) => {
                Some(resolve_key_exprs(std::slice::from_ref(e), &schema, "PARTITION BY")?[0])
            }
            None => None,
        };

        let name = self.catalog.qualify(&c.name);
        let def = TableDef { name, schema, order_by, primary_key, partition_by, engine: c.engine };
        self.catalog.create_table(def, c.if_not_exists)?;

        let mut n = 0;
        if let Some(blocks) = as_blocks {
            let path = self.catalog.qualify(&c.name);
            for b in blocks {
                if b.rows() > 0 {
                    n += self.catalog.table_by_path_mut(&path)?.insert(b)?;
                }
            }
        }
        Ok(if n > 0 { ResultSet::with_affected(n) } else { ResultSet::empty() })
    }

    /// `ALTER TABLE ... ADD COLUMN`: rebuild with the new column appended.
    fn run_add_column(
        &mut self,
        table: &ObjectName,
        col: &ColumnDef,
        if_not_exists: bool,
    ) -> Result<ResultSet> {
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

    pub const LOCK_EX: c_int = 2;
    pub const LOCK_NB: c_int = 4;

    extern "C" {
        pub fn flock(fd: c_int, op: c_int) -> c_int;
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
fn lock_data_dir(root: &Path) -> Result<Option<File>> {
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::os::unix::io::AsRawFd;

    let path = root.join(LOCK_FILE);
    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        // Emphatically not `truncate`: the file is opened before the lock is
        // held, and erasing the incumbent's pid would destroy the only thing
        // that makes the failure diagnosable.
        .truncate(false)
        .open(&path)
        .map_err(|e| Error::Io(format!("cannot open lock file {}: {e}", path.display())))?;

    // SAFETY: `f` owns a valid open descriptor for the whole call, and `flock`
    // only inspects the descriptor -- it neither retains it nor touches user
    // memory.
    let rc = unsafe { flock_sys::flock(f.as_raw_fd(), flock_sys::LOCK_EX | flock_sys::LOCK_NB) };
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
            "data directory `{}` is already open by another granular process{who}: {os}. \
             Only one process may have a data directory open at a time -- concurrent \
             writers allocate colliding part file names and overwrite each other's \
             committed data. Close the other process, or point this one at a different \
             --data directory.",
            root.display()
        )));
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
fn lock_data_dir(_root: &Path) -> Result<Option<File>> {
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

/// Statements that change the catalog's shape rather than a table's contents.
fn is_ddl(s: &Statement) -> bool {
    matches!(
        s,
        Statement::CreateTable(_)
            | Statement::CreateDatabase { .. }
            | Statement::DropTable { .. }
            | Statement::DropDatabase { .. }
            | Statement::AlterAddColumn { .. }
            | Statement::AlterDropColumn { .. }
            | Statement::Truncate { .. }
    )
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

fn render_create_table(schema: &Schema, def: &TableDef) -> String {
    let cols: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| match f.default_sql() {
            Some(d) => format!("    `{}` {} DEFAULT {d}", f.name, f.ty),
            None => format!("    `{}` {}", f.name, f.ty),
        })
        .collect();
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
        let wal = s.join("default").join("t").join(crate::persist::store::WAL_FILE);
        let before = std::fs::read(&wal).unwrap();

        db.execute("BEGIN").unwrap();
        for i in 2..40u64 {
            db.execute(&format!("INSERT INTO t VALUES ({i}, {})", i * 10)).unwrap();
        }
        db.execute("ALTER TABLE t DELETE WHERE id = 1").unwrap();
        assert!(
            std::fs::read(&wal).unwrap().len() > before.len(),
            "the test needs the transaction to have logged something"
        );
        db.execute("ROLLBACK").unwrap();

        assert_eq!(std::fs::read(&wal).unwrap(), before, "the log kept the aborted records");
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
    #[test]
    fn ddl_and_checkpoint_are_refused_inside_a_transaction() {
        let s = Scratch::new("session-txn-ddl");
        let mut db = Session::open(s.path()).unwrap();
        db.execute(KEYED).unwrap();
        db.execute("BEGIN").unwrap();
        db.execute("INSERT INTO t VALUES (1, 10)").unwrap();

        for ddl in [
            "CREATE TABLE u (id UInt64) ENGINE = MergeTree ORDER BY id",
            "DROP TABLE t",
            "TRUNCATE TABLE t",
            "ALTER TABLE t ADD COLUMN w Int64",
        ] {
            let e = db.execute(ddl).unwrap_err();
            assert_eq!(e.code(), "NOT_IMPLEMENTED", "{ddl}: {e}");
        }
        assert_eq!(db.checkpoint().unwrap_err().code(), "NOT_IMPLEMENTED");

        // The transaction is untouched by all of that, and still commits.
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

    #[test]
    fn commit_or_rollback_without_a_transaction_is_an_error() {
        let mut s = Session::in_memory();
        s.execute(KEYED).unwrap();
        assert!(s.execute("COMMIT").is_err());
        assert!(s.execute("ROLLBACK").is_err());
        s.execute("BEGIN").unwrap();
        assert!(s.execute("BEGIN").is_err(), "nesting is refused");
        s.execute("COMMIT").unwrap();
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

    /// A positional delete has no write-ahead representation, so a logging
    /// session must refuse it rather than acknowledge a mutation a crash would
    /// silently undo. The same statement is fine in memory, and fine on disk
    /// once the table has a key to log by.
    #[test]
    fn an_unkeyed_mutation_is_refused_on_a_logging_session() {
        let s = Scratch::new("session-unkeyed-mutation");
        let mut db = Session::open(s.path()).unwrap();
        db.execute("CREATE TABLE u (id UInt64, v Int64) ENGINE = MergeTree ORDER BY tuple()")
            .unwrap();
        db.execute("INSERT INTO u VALUES (1,10),(2,20)").unwrap();

        let err = db.execute("DELETE FROM u WHERE id = 1").unwrap_err();
        assert_eq!(err.code(), "NOT_IMPLEMENTED", "{err}");
        assert!(err.to_string().contains("PRIMARY KEY"), "{err}");
        // Refused means refused: nothing was hidden on the way out.
        assert_eq!(vals(&mut db, "SELECT count() FROM u")[0][0], Value::UInt(2));
        assert!(db.execute("UPDATE u SET v = 1 WHERE id = 1").is_err());
        assert_eq!(vals(&mut db, "SELECT v FROM u WHERE id = 1")[0][0], Value::Int(10));

        // With a key, the same shapes work and survive a checkpoint.
        db.execute(
            "CREATE TABLE k (id UInt64, v Int64) ENGINE = MergeTree PRIMARY KEY id ORDER BY id",
        )
        .unwrap();
        db.execute("INSERT INTO k VALUES (1,10),(2,20),(3,30)").unwrap();
        db.execute("DELETE FROM k WHERE id = 2").unwrap();
        db.checkpoint().unwrap();
        drop(db);
        let mut db = Session::open(s.path()).unwrap();
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

}
