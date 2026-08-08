//! `system.*` and `information_schema.*`: the engine, queryable.
//!
//! Everything in here is a **virtual** table. Nothing is materialized, nothing
//! is maintained, and no write path knows these exist -- a row of
//! `system.parts` is computed from the catalog and the part metadata at the
//! instant it is read, from the same [`Snapshot`](crate::storage::part::Snapshot)
//! a `SELECT` on the table itself would pin. A `system.parts` that disagrees
//! with the files on disk is worse than no `system.parts`, and the only way to
//! be sure it cannot is to have no second copy of the truth.
//!
//! ## Tables, not dot-commands
//!
//! A REPL command cannot be joined, filtered, aggregated or scripted, and at
//! 3am those are the four things you need. So these are relations:
//!
//! ```sql
//!   SELECT database, table, sum(data_bytes) FROM system.parts
//!    WHERE state = 'active' GROUP BY database, table ORDER BY 3 DESC
//! ```
//!
//! works, and so does joining `system.parts` to `system.tables`.
//!
//! ## How they reach the planner without a new plan node
//!
//! A reference to one is rewritten, before binding, into the derived table it
//! is equivalent to:
//!
//! ```text
//!   FROM system.parts p   ->   FROM (SELECT c1 AS database, c2 AS table, ...
//!                                      FROM (VALUES (...), (...))) AS p
//! ```
//!
//! That buys three things for no new machinery. The rows go through the same
//! binder, optimizer and executor as any other relation, so filters, joins and
//! aggregates over them are not special cases that can drift. The rewrite runs
//! on `&self`, so a [`Reader`](crate::Reader) on another thread sees values as
//! fresh as the writer's. And there is no `Table` in the catalog to be stale,
//! to be checkpointed to disk, or to be found by `SHOW TABLES` as if a user
//! had created it.
//!
//! The cost when unused is two string compares per table reference, folded
//! into the walk `session::has_subquery` already does and sharing the clone it
//! already decides on -- so an ordinary query pays nothing new. The cost when
//! used is one `Value` per cell of the result, which for these tables is
//! bounded by the number of parts.
//!
//! ## A real table wins
//!
//! If a database actually named `system` holds a table actually named `parts`,
//! that table is what `system.parts` means. A virtual table that shadowed
//! user data would be a silent wrong answer, and the reserved-name rule
//! belongs in `CREATE TABLE` rather than here.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::catalog::Catalog;
use crate::common::Result;
use crate::settings::Settings;
use crate::sql::ast::{Expr, ObjectName, Query, Select, SelectItem, SetExpr, TableRef};
use crate::storage::part::NO_FILE;
use crate::types::{TableDef, Value};

/// Which virtual table a name refers to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Tables,
    Parts,
    Columns,
    Settings,
    QueryLog,
    InfoTables,
    InfoColumns,
}

/// The database qualifier a virtual table lives under.
pub const SYSTEM_DB: &str = "system";
/// The SQL-standard alias namespace, because that is what tools look for.
pub const INFO_SCHEMA_DB: &str = "information_schema";

/// Resolve `name` to a virtual table, or `None` if it is an ordinary one.
///
/// Two string compares against the qualifier for the overwhelming majority of
/// references, which never match. The catalog probe only happens on a name
/// that already looks like ours.
pub fn classify(name: &ObjectName, catalog: &Catalog) -> Option<Kind> {
    let db = name.qualifier()?;
    let kind = if db.eq_ignore_ascii_case(SYSTEM_DB) {
        match_lower(name.last(), &[
            ("tables", Kind::Tables),
            ("parts", Kind::Parts),
            ("columns", Kind::Columns),
            ("settings", Kind::Settings),
            ("query_log", Kind::QueryLog),
        ])?
    } else if db.eq_ignore_ascii_case(INFO_SCHEMA_DB) {
        match_lower(name.last(), &[
            ("tables", Kind::InfoTables),
            ("columns", Kind::InfoColumns),
        ])?
    } else {
        return None;
    };
    // A real table of that name is the user's, not ours.
    if catalog.table_by_path(&format!("{db}.{}", name.last())).is_ok() {
        return None;
    }
    Some(kind)
}

fn match_lower(s: &str, table: &[(&str, Kind)]) -> Option<Kind> {
    table.iter().find(|(n, _)| s.eq_ignore_ascii_case(n)).map(|&(_, k)| k)
}

// ---------------------------------------------------------------------------
// the query log
// ---------------------------------------------------------------------------

/// Statements kept for `system.query_log`.
///
/// A ring, so the memory this costs is bounded by the constant and not by
/// uptime. 512 is enough to cover the burst that preceded whatever you are
/// looking at without being a place large query texts accumulate.
pub const LOG_CAPACITY: usize = 512;

/// Statement text kept per entry. A generated `INSERT ... VALUES` can be
/// megabytes and there is nothing in the tail of one worth 512 copies of.
pub const LOG_TEXT_MAX: usize = 4_000;

/// One statement, as it ran.
///
/// The two variable-length fields are `String` rather than `Box<str>` because
/// the ring *recycles* them: once it is full, appending an entry pops the
/// oldest, clears its buffers and writes into them, so a steady-state engine
/// does one malloc and one free per statement fewer. Measured at the
/// statement, below.
#[derive(Clone, Debug, Default)]
pub struct Entry {
    /// Seconds since the epoch, at completion.
    pub at: i64,
    pub kind: &'static str,
    pub sql: String,
    pub rows: u64,
    pub elapsed_us: u64,
    pub granules_read: u64,
    pub granules_pruned: u64,
    pub rows_scanned: u64,
    /// The message the statement failed with; empty when it succeeded.
    pub error: String,
}

/// The counters an entry carries, so `record` takes one argument per *idea*
/// rather than nine positional numbers.
#[derive(Clone, Copy, Debug, Default)]
pub struct Counters {
    pub rows: u64,
    pub elapsed_us: u64,
    pub granules_read: u64,
    pub granules_pruned: u64,
    pub rows_scanned: u64,
}

/// The log itself, shared so a [`Reader`](crate::Reader) on another thread
/// records into the same ring the writer does.
///
/// One uncontended lock and one `memcpy` of the statement text per
/// *statement*, and no allocation once the ring is full. That is the whole
/// cost, and it is charged where a statement has already been lexed, bound and
/// executed -- deliberately not per row, per block or per granule, which is
/// the line this engine draws everywhere else. `Session::log_stmt` carries the
/// measurement.
#[derive(Clone, Default, Debug)]
pub struct QueryLog(Arc<Mutex<VecDeque<Entry>>>);

impl QueryLog {
    pub fn new() -> QueryLog {
        QueryLog(Arc::new(Mutex::new(VecDeque::with_capacity(LOG_CAPACITY))))
    }

    /// Append one statement, evicting the oldest once the ring is full.
    ///
    /// The evicted entry is not dropped, it is *refilled*: its two `String`s
    /// keep their capacity and the statement text is copied into them. So the
    /// steady state -- a ring that has been full since the 512th statement --
    /// allocates only when a statement is longer than any of the last 512.
    ///
    /// A poisoned mutex is recovered from rather than propagated: losing the
    /// diagnostic log is not a reason to fail the statement it describes.
    pub fn record(&self, sql: &str, kind: &'static str, c: Counters, error: Option<&str>) {
        let mut g = self.0.lock().unwrap_or_else(|p| p.into_inner());
        let mut e = match g.len() {
            LOG_CAPACITY.. => g.pop_front().unwrap_or_default(),
            _ => Entry::default(),
        };
        e.at = now_unix();
        e.kind = kind;
        e.sql.clear();
        clip_into(sql, &mut e.sql);
        e.rows = c.rows;
        e.elapsed_us = c.elapsed_us;
        e.granules_read = c.granules_read;
        e.granules_pruned = c.granules_pruned;
        e.rows_scanned = c.rows_scanned;
        e.error.clear();
        if let Some(m) = error {
            e.error.push_str(m);
        }
        g.push_back(e);
    }

    /// Newest first, which is the order anyone reading a log wants.
    fn snapshot(&self) -> Vec<Entry> {
        let g = self.0.lock().unwrap_or_else(|p| p.into_inner());
        g.iter().rev().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Append `sql`, trimmed and length-capped, to `out`.
///
/// Truncated on a character boundary: a statement is a `&str`, and slicing one
/// at a byte offset is a panic waiting for the first multi-byte literal.
pub fn clip_into(sql: &str, out: &mut String) {
    let s = sql.trim();
    if s.len() <= LOG_TEXT_MAX {
        out.push_str(s);
        return;
    }
    let mut end = LOG_TEXT_MAX;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    out.push_str(&s[..end]);
    out.push_str("...");
}

pub fn now_unix() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
}

// ---------------------------------------------------------------------------
// the rewrite
// ---------------------------------------------------------------------------

/// The derived table `kind` is equivalent to, ready to stand in for a
/// `TableRef::Table` in the AST.
pub fn derived(
    kind: Kind,
    catalog: &Catalog,
    settings: &crate::settings::Handle,
    log: &QueryLog,
) -> Result<Query> {
    let (cols, rows) = rows_of(kind, catalog, settings, log)?;
    Ok(as_query(cols, rows))
}

/// `SELECT c1 AS <name>, ... FROM (VALUES ...)`, with the aliases carrying the
/// column names the binder would otherwise have no source for.
///
/// An empty table still needs a schema, so it is one row of the right shapes
/// with a constant-false filter over it rather than a bare `VALUES ()`, which
/// binds to a plan of no columns and would make `SELECT database FROM
/// system.parts` fail to resolve on a database that happens to have no parts.
fn as_query(cols: Columns, rows: Vec<Vec<Value>>) -> Query {
    let empty = rows.is_empty();
    let rows = if empty { vec![cols.iter().map(|(_, s)| s.zero()).collect()] } else { rows };
    let values = rows
        .into_iter()
        .map(|r| r.into_iter().map(Expr::Literal).collect())
        .collect();
    let inner = Query {
        with: Vec::new(),
        body: SetExpr::Values(values),
        order_by: Vec::new(),
        limit: None,
        offset: None,
        limit_by: None,
    };
    let select = Select {
        distinct: false,
        projection: cols
            .iter()
            .enumerate()
            .map(|(i, (name, _))| SelectItem::Expr {
                expr: Expr::Column(ObjectName::bare(format!("c{}", i + 1))),
                alias: Some((*name).to_string()),
            })
            .collect(),
        from: Some(TableRef::Subquery { query: Box::new(inner), alias: None }),
        prewhere: None,
        selection: empty.then(|| Expr::BinaryOp {
            left: Box::new(Expr::Literal(Value::Int(0))),
            op: crate::sql::ast::BinaryOp::Eq,
            right: Box::new(Expr::Literal(Value::Int(1))),
        }),
        group_by: Vec::new(),
        with_totals: false,
        having: None,
    };
    Query::simple(select)
}

/// The value a column takes in the placeholder row of an empty virtual table.
///
/// Its only job is to fix the column's type, because `VALUES` infers the type
/// from the literals -- so a table's schema must not depend on whether it
/// happens to have rows today.
#[derive(Clone, Copy)]
pub enum Shape {
    Text,
    Num,
    Time,
}

impl Shape {
    fn zero(self) -> Value {
        match self {
            Shape::Text => Value::str(""),
            Shape::Num => Value::UInt(0),
            Shape::Time => Value::DateTime(0),
        }
    }
}

use Shape::{Num, Text, Time};

const TABLES: Columns = &[
    ("database", Text),
    ("name", Text),
    ("engine", Text),
    ("sorting_key", Text),
    ("primary_key", Text),
    ("partition_key", Text),
    ("columns", Num),
    ("parts", Num),
    ("rows", Num),
    ("delta_rows", Num),
    ("data_bytes", Num),
    ("index_bytes", Num),
    ("quarantined", Num),
];

const PARTS: Columns = &[
    ("database", Text),
    ("table", Text),
    ("name", Text),
    ("state", Text),
    ("rows", Num),
    ("live_rows", Num),
    ("deleted_rows", Num),
    ("granules", Num),
    ("data_bytes", Num),
    ("index_bytes", Num),
    ("reason", Text),
];

const COLUMNS: Columns = &[
    ("database", Text),
    ("table", Text),
    ("name", Text),
    ("position", Num),
    ("type", Text),
    ("default_expression", Text),
    ("is_nullable", Num),
    ("is_in_primary_key", Num),
    ("is_in_sorting_key", Num),
];

const SETTINGS: Columns =
    &[("name", Text), ("value", Text), ("default", Text), ("type", Text), ("description", Text)];

const QUERY_LOG: Columns = &[
    ("event_time", Time),
    ("kind", Text),
    ("query", Text),
    ("rows", Num),
    ("duration_us", Num),
    ("granules_read", Num),
    ("granules_pruned", Num),
    ("rows_scanned", Num),
    ("error", Text),
];

const INFO_TABLES: Columns = &[
    ("table_catalog", Text),
    ("table_schema", Text),
    ("table_name", Text),
    ("table_type", Text),
];

const INFO_COLUMNS: Columns = &[
    ("table_catalog", Text),
    ("table_schema", Text),
    ("table_name", Text),
    ("column_name", Text),
    ("ordinal_position", Num),
    ("column_default", Text),
    ("is_nullable", Text),
    ("data_type", Text),
];

/// A virtual table's column list: name and the value shape that fixes its
/// type when the table happens to be empty.
pub type Columns = &'static [(&'static str, Shape)];

/// The column list of a virtual table, without computing its rows.
pub fn schema_of(kind: Kind) -> Columns {
    match kind {
        Kind::Tables => TABLES,
        Kind::Parts => PARTS,
        Kind::Columns => COLUMNS,
        Kind::Settings => SETTINGS,
        Kind::QueryLog => QUERY_LOG,
        Kind::InfoTables => INFO_TABLES,
        Kind::InfoColumns => INFO_COLUMNS,
    }
}

fn rows_of(
    kind: Kind,
    catalog: &Catalog,
    settings: &crate::settings::Handle,
    log: &QueryLog,
) -> Result<(Columns, Vec<Vec<Value>>)> {
    let rows = match kind {
        Kind::Tables => tables(catalog)?,
        Kind::Parts => parts(catalog)?,
        Kind::Columns => columns(catalog, false)?,
        // The lock is taken only in this arm: every other virtual table would
        // otherwise pay for a settings snapshot it never looks at.
        Kind::Settings => settings_rows(&settings.snapshot())?,
        Kind::QueryLog => query_log(log),
        Kind::InfoTables => info_tables(catalog)?,
        Kind::InfoColumns => columns(catalog, true)?,
    };
    Ok((schema_of(kind), rows))
}

// ---------------------------------------------------------------------------
// the rows
// ---------------------------------------------------------------------------

fn text(s: impl AsRef<str>) -> Value {
    Value::str(s.as_ref())
}

fn num(n: usize) -> Value {
    Value::UInt(n as u64)
}

/// Column names for a list of indices, as `a, b`. Empty stays empty rather
/// than becoming `()`, so a filter on `sorting_key = ''` means what it says.
fn key_names(def: &TableDef, cols: &[usize]) -> String {
    let mut s = String::new();
    for (i, &c) in cols.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(def.schema.name(c));
    }
    s
}

/// Every `(database, table name)` the catalog knows, sorted, **including
/// quarantined tables**.
///
/// Hiding a damaged table here would be the same mistake `SHOW TABLES` already
/// refuses to make: the one table an operator needs to find at 3am is the one
/// that stopped answering.
fn roster(catalog: &Catalog) -> Result<Vec<(String, String)>> {
    let mut dbs = catalog.database_names();
    dbs.sort();
    let mut out = Vec::new();
    for db in dbs {
        for t in catalog.table_names(Some(&db))? {
            out.push((db.clone(), t));
        }
    }
    Ok(out)
}

/// The definition of a table whether or not it is quarantined.
///
/// `Catalog::table_by_path` refuses a quarantined table by design -- that
/// refusal is what stops a short answer reaching a query -- so a describing
/// view has to come in through the door the checkpoint uses.
fn def_of<'c>(catalog: &'c Catalog, db: &str, name: &str) -> Option<&'c TableDef> {
    catalog.quarantined_def(db, name).or_else(|| {
        catalog.table_by_path(&format!("{db}.{name}")).ok().map(|t| &t.def)
    })
}

fn tables(catalog: &Catalog) -> Result<Vec<Vec<Value>>> {
    let mut out = Vec::new();
    for (db, name) in roster(catalog)? {
        let path = format!("{db}.{name}");
        let quarantined = catalog.is_quarantined(&path);
        let Some(def) = def_of(catalog, &db, &name) else { continue };
        let head = [
            text(&db),
            text(&name),
            text(def.engine.name()),
            text(key_names(def, &def.order_by)),
            text(key_names(def, &def.primary_key)),
            text(def.partition_by.map_or(String::new(), |c| def.schema.name(c).to_string())),
            num(def.schema.len()),
        ];
        // A quarantined table's `Table` is missing however many rows lived in
        // the file that failed, so its counters are refused rather than
        // reported short -- the `quarantined` column is the answer, and
        // `system.parts` names the file.
        let body = match (quarantined, catalog.table_by_path(&path)) {
            (false, Ok(t)) => {
                let snap = t.snapshot();
                [
                    num(snap.len()),
                    num(snap.live_rows() + t.delta_len()),
                    num(t.delta_len()),
                    num(t.data_bytes()),
                    num(t.index_bytes()),
                    num(0),
                ]
            }
            _ => [num(0), num(0), num(0), num(0), num(0), num(1)],
        };
        out.push(head.into_iter().chain(body).collect());
    }
    Ok(out)
}

fn parts(catalog: &Catalog) -> Result<Vec<Vec<Value>>> {
    let mut out = Vec::new();
    for (db, name) in roster(catalog)? {
        let path = format!("{db}.{name}");
        if let Ok(t) = catalog.table_by_path(&path) {
            let snap = t.snapshot();
            let set = snap.set();
            for i in 0..snap.len() {
                let p = snap.part(i);
                let live = set.live_rows_of(i);
                // The file that already holds these bytes, which is what an
                // operator needs to correlate a row here with `ls`. Empty when
                // the part has not been checkpointed yet -- it exists only in
                // this process and in the write-ahead log.
                let file = match set.origin(i) {
                    NO_FILE => String::new(),
                    seq => crate::persist::store::part_file_name(seq),
                };
                out.push(vec![
                    text(&db),
                    text(&name),
                    text(file),
                    text("active"),
                    num(p.n_rows),
                    num(live),
                    num(p.n_rows - live),
                    num(p.granule_count()),
                    num(p.data_bytes()),
                    num(p.index_bytes()),
                    text(""),
                ]);
            }
        }
    }
    // The parts that did not decode. They are not in any `PartSet` -- that is
    // what quarantine means -- so they can only come from the record the
    // reader handed the catalog, and leaving them out would make this view
    // disagree with the directory in exactly the case it is consulted.
    for (path, dp) in catalog.damaged_parts() {
        let (db, name) = path.split_once('.').unwrap_or(("", path));
        out.push(vec![
            text(db),
            text(name),
            text(&dp.file),
            text("damaged"),
            num(0),
            num(0),
            num(0),
            num(0),
            num(0),
            num(0),
            text(&dp.why),
        ]);
    }
    Ok(out)
}

fn columns(catalog: &Catalog, info_schema: bool) -> Result<Vec<Vec<Value>>> {
    let mut out = Vec::new();
    for (db, name) in roster(catalog)? {
        let Some(def) = def_of(catalog, &db, &name) else { continue };
        for (i, f) in def.schema.fields().iter().enumerate() {
            let default = f.default_sql().unwrap_or_default();
            let nullable = f.ty.is_nullable();
            out.push(if info_schema {
                vec![
                    text(&db),
                    text(&db),
                    text(&name),
                    text(&f.name),
                    num(i + 1),
                    text(default),
                    text(if nullable { "YES" } else { "NO" }),
                    text(f.ty.to_string()),
                ]
            } else {
                vec![
                    text(&db),
                    text(&name),
                    text(&f.name),
                    num(i + 1),
                    text(f.ty.to_string()),
                    text(default),
                    num(nullable as usize),
                    num(def.primary_key.contains(&i) as usize),
                    num(def.order_by.contains(&i) as usize),
                ]
            });
        }
    }
    Ok(out)
}

fn info_tables(catalog: &Catalog) -> Result<Vec<Vec<Value>>> {
    Ok(roster(catalog)?
        .into_iter()
        .map(|(db, name)| vec![text(&db), text(&db), text(name), text("BASE TABLE")])
        .collect())
}

/// Exactly what `SHOW SETTINGS` renders, through the same builder, so the two
/// cannot drift.
fn settings_rows(cfg: &Settings) -> Result<Vec<Vec<Value>>> {
    Ok(crate::settings::show(cfg, None)?.to_values())
}

fn query_log(log: &QueryLog) -> Vec<Vec<Value>> {
    log.snapshot()
        .into_iter()
        .map(|e| {
            vec![
                Value::DateTime(e.at),
                text(e.kind),
                text(&e.sql),
                Value::UInt(e.rows),
                Value::UInt(e.elapsed_us),
                Value::UInt(e.granules_read),
                Value::UInt(e.granules_pruned),
                Value::UInt(e.rows_scanned),
                text(&e.error),
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_virtual_table_has_a_row_shape_of_the_right_width() {
        let cat = Catalog::in_memory();
        let cfg = crate::settings::Handle::default();
        let log = QueryLog::new();
        for k in [
            Kind::Tables,
            Kind::Parts,
            Kind::Columns,
            Kind::Settings,
            Kind::QueryLog,
            Kind::InfoTables,
            Kind::InfoColumns,
        ] {
            let (cols, rows) = rows_of(k, &cat, &cfg, &log).unwrap();
            assert_eq!(cols.len(), schema_of(k).len());
            for r in &rows {
                assert_eq!(r.len(), cols.len(), "{k:?} row width");
            }
        }
    }

    #[test]
    fn the_log_is_a_ring_and_reports_newest_first() {
        let log = QueryLog::new();
        for i in 0..LOG_CAPACITY + 10 {
            log.record(&format!("q{i}"), "SELECT", Counters::default(), None);
        }
        assert_eq!(log.len(), LOG_CAPACITY, "the ring must bound its own memory");
        let s = log.snapshot();
        assert_eq!(s[0].sql, format!("q{}", LOG_CAPACITY + 9));
        assert_eq!(s[LOG_CAPACITY - 1].sql, format!("q{}", 10));
    }

    /// A recycled buffer must not leave the previous statement's tail behind,
    /// which is the one way this optimization could produce a wrong row.
    #[test]
    fn a_recycled_entry_keeps_nothing_of_the_one_it_replaced() {
        let log = QueryLog::new();
        for i in 0..LOG_CAPACITY {
            log.record(
                &format!("SELECT {i} FROM a_very_long_table_name_to_grow_the_buffer"),
                "SELECT",
                Counters::default(),
                Some("something went wrong in a long and detailed way"),
            );
        }
        log.record("SELECT 1", "SELECT", Counters { rows: 7, ..Counters::default() }, None);
        let s = log.snapshot();
        assert_eq!(s[0].sql, "SELECT 1");
        assert_eq!(s[0].error, "");
        assert_eq!(s[0].rows, 7);
    }

    #[test]
    fn clip_never_splits_a_character() {
        let long = "é".repeat(LOG_TEXT_MAX);
        let mut c = String::new();
        clip_into(&long, &mut c);
        assert!(c.len() <= LOG_TEXT_MAX + 3);
        assert!(c.ends_with("..."));
        let mut t = String::new();
        clip_into("  SELECT 1  ", &mut t);
        assert_eq!(t, "SELECT 1");
    }

    #[test]
    fn an_unqualified_or_foreign_name_is_not_a_system_table() {
        let cat = Catalog::in_memory();
        assert!(classify(&ObjectName::bare("parts"), &cat).is_none());
        assert!(classify(&ObjectName(vec!["mydb".into(), "parts".into()]), &cat).is_none());
        assert!(classify(&ObjectName(vec!["system".into(), "nope".into()]), &cat).is_none());
        assert_eq!(
            classify(&ObjectName(vec!["SYSTEM".into(), "Parts".into()]), &cat),
            Some(Kind::Parts)
        );
        assert_eq!(
            classify(&ObjectName(vec!["information_schema".into(), "columns".into()]), &cat),
            Some(Kind::InfoColumns)
        );
    }
}
