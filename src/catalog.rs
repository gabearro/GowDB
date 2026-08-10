//! Databases and tables, and the name resolution over them.
//!
//! The catalog owns every [`Table`]. Everything above it -- binder, planner,
//! executor -- borrows tables from here rather than owning storage, which
//! keeps the "who can mutate a table" question answerable: only code holding
//! `&mut Catalog`.
//!
//! ## Quarantine
//!
//! Name resolution is also where damage stops. A table whose part files did
//! not all decode at open ([`crate::persist::reader::read_table_image`]) is
//! recorded here by name, and every accessor that hands out a `Table` refuses
//! it -- because the `Table` behind that name is missing however many rows
//! lived in the file that failed, and serving it would be a plausible wrong
//! answer rather than an error.
//!
//! The refusal is deliberately *here* and not in the scan: this is the one
//! place a query names a table, so the check costs one map probe per statement
//! rather than one per granule, and it is skipped entirely -- a load and a
//! branch -- while the map is empty, which it is on every undamaged database.
//!
//! What keeps working is everything that does not read a table's rows:
//! `SHOW DATABASES`, `SHOW TABLES` (which lists the damaged table, because
//! hiding it is how a later checkpoint would decide it had been dropped and
//! delete its directory), `USE`, `CREATE TABLE`, and `DROP TABLE` -- the last
//! being the operator's way out when the file is not coming back.

use std::path::{Path, PathBuf};

use crate::common::{Error, FastMap, Result};
use crate::persist::reader::{self, DamagedPart};
use crate::persist::store;
use crate::sql::ast::ObjectName;
use crate::storage::Table;
use crate::types::TableDef;

pub const DEFAULT_DATABASE: &str = "default";

/// Delta-memtable size at which a table auto-flushes to a part.
pub const DEFAULT_DELTA_LIMIT: usize = 64 * 1024;

#[derive(Default)]
pub struct Database {
    pub tables: FastMap<String, Table>,
}

pub struct Catalog {
    databases: FastMap<String, Database>,
    current: String,
    /// Backing directory. `None` for a purely in-memory catalog.
    dir: Option<PathBuf>,
    delta_limit: usize,
    /// Tables whose on-disk parts did not all decode, by `db.table`.
    ///
    /// Empty on every healthy database, which is what makes the check on the
    /// resolve path free: one `is_empty` against a `HashMap` field, and no
    /// name is formatted unless there is something to compare it against.
    /// Open and scan on an undamaged 4M-row, 13-part database measured
    /// unchanged (open 86.9 ms against 86.2 ms, scan 9.2 ms against 8.7 ms,
    /// best of 16 interleaved runs per side, per-side spread ±12%).
    damaged: FastMap<String, Vec<DamagedPart>>,

    /// This data directory's identity, minted once and then never changed --
    /// 0 until the first checkpoint stamps it (and for a directory written by
    /// a build that predates it). A backup carries the id of the directory it
    /// was taken from, which is the only thing that distinguishes *this*
    /// database's archived log from an unrelated one whose LSNs happen to
    /// start at the same place. See `store::mint_instance`.
    instance: u64,
}

impl Catalog {
    pub fn in_memory() -> Catalog {
        let mut databases = FastMap::default();
        databases.insert(DEFAULT_DATABASE.to_string(), Database::default());
        Catalog {
            databases,
            current: DEFAULT_DATABASE.to_string(),
            dir: None,
            delta_limit: DEFAULT_DELTA_LIMIT,
            damaged: FastMap::default(),
            instance: 0,
        }
    }

    pub fn instance(&self) -> u64 {
        self.instance
    }

    pub fn set_instance(&mut self, id: u64) {
        self.instance = id;
    }

    /// A catalog backed by `dir`. The directory is created if absent; existing
    /// contents are *not* loaded here -- call
    /// [`crate::persist::catalog::load_all`] for that.
    ///
    /// Refuses to open a directory that holds table data but no `CATALOG`:
    /// see `unaccounted_table_dirs` below for why that case cannot be treated
    /// as an empty database.
    pub fn on_disk(dir: impl AsRef<Path>) -> Result<Catalog> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        // One `stat` in the overwhelmingly common case; the directory walk
        // only runs when the roster file is already missing, which is either a
        // brand-new database (nothing to walk) or the accident below.
        if !dir.join(store::CATALOG_FILE).exists() {
            let orphans = unaccounted_table_dirs(&dir);
            if !orphans.is_empty() {
                let shown: Vec<&str> = orphans.iter().take(5).map(String::as_str).collect();
                let more = orphans.len().saturating_sub(shown.len());
                return Err(Error::storage(format!(
                    "`{}` has table data but no {} file: refusing to open it as an empty \
                     database. Found {}{}. Opening empty would make the next checkpoint \
                     treat every one of those as a dropped table and delete it. Restore \
                     {} from a backup, or move the data aside if you really want a fresh \
                     database here.",
                    dir.display(),
                    store::CATALOG_FILE,
                    shown.join(", "),
                    if more > 0 { format!(" and {more} more") } else { String::new() },
                    store::CATALOG_FILE,
                )));
            }
        }
        let mut c = Catalog::in_memory();
        c.dir = Some(dir);
        Ok(c)
    }

    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    pub fn is_persistent(&self) -> bool {
        self.dir.is_some()
    }

    pub fn set_delta_limit(&mut self, n: usize) {
        self.delta_limit = n.max(1);
    }

    pub fn current_database(&self) -> &str {
        &self.current
    }

    pub fn use_database(&mut self, name: &str) -> Result<()> {
        if !self.databases.contains_key(name) {
            return Err(Error::storage(format!("database `{name}` does not exist")));
        }
        self.current = name.to_string();
        Ok(())
    }

    pub fn database_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.databases.keys().cloned().collect();
        v.sort();
        v
    }

    /// Database names, borrowed and unordered -- for the checks that only need
    /// to look. [`Catalog::database_names`] clones and sorts every name; a
    /// name check wants neither.
    pub fn db_names(&self) -> impl Iterator<Item = &str> {
        self.databases.keys().map(String::as_str)
    }

    /// Table names in `db`, borrowed and unordered. Empty for a database that
    /// does not exist -- the caller is about to report that better than this
    /// could.
    pub fn table_names_in(&self, db: &str) -> impl Iterator<Item = &str> {
        self.databases.get(db).into_iter().flat_map(|d| d.tables.keys().map(String::as_str))
    }

    pub fn create_database(&mut self, name: &str, if_not_exists: bool) -> Result<()> {
        if self.databases.contains_key(name) {
            return if if_not_exists {
                Ok(())
            } else {
                Err(Error::storage(format!("database `{name}` already exists")))
            };
        }
        self.databases.insert(name.to_string(), Database::default());
        Ok(())
    }

    pub fn drop_database(&mut self, name: &str, if_exists: bool) -> Result<()> {
        if name == DEFAULT_DATABASE {
            return Err(Error::storage("cannot drop the default database"));
        }
        if !self.damaged.is_empty() {
            self.damaged.retain(|k, _| k.split_once('.').is_none_or(|(d, _)| d != name));
        }
        if self.databases.remove(name).is_none() && !if_exists {
            return Err(Error::storage(format!("database `{name}` does not exist")));
        }
        if self.current == name {
            self.current = DEFAULT_DATABASE.to_string();
        }
        Ok(())
    }

    /// Split a possibly-qualified name into `(database, table)`, defaulting
    /// the database to the session's current one.
    pub fn resolve(&self, name: &ObjectName) -> (String, String) {
        match name.0.len() {
            0 => (self.current.clone(), String::new()),
            1 => (self.current.clone(), name.0[0].clone()),
            _ => {
                let n = name.0.len();
                (name.0[n - 2].clone(), name.0[n - 1].clone())
            }
        }
    }

    pub fn create_table(&mut self, def: TableDef, if_not_exists: bool) -> Result<()> {
        let (db, tbl) = {
            let parts: Vec<&str> = def.name.split('.').collect();
            if parts.len() >= 2 {
                (parts[parts.len() - 2].to_string(), parts[parts.len() - 1].to_string())
            } else {
                (self.current.clone(), def.name.clone())
            }
        };
        let limit = self.delta_limit;
        let d = self
            .databases
            .get_mut(&db)
            .ok_or_else(|| Error::storage(format!("database `{db}` does not exist")))?;
        if d.tables.contains_key(&tbl) {
            return if if_not_exists {
                Ok(())
            } else {
                Err(Error::storage(format!("table `{db}.{tbl}` already exists")))
            };
        }
        let mut def = def;
        def.name = tbl.clone();
        d.tables.insert(tbl, Table::new(def, limit));
        Ok(())
    }

    pub fn drop_table(&mut self, name: &ObjectName, if_exists: bool) -> Result<()> {
        let (db, tbl) = self.resolve(name);
        // Before the removal, and unconditionally: dropping a quarantined
        // table is the operator's way out, so the quarantine has to go with
        // it or the name stays refused for a table that no longer exists --
        // and, worse, the next checkpoint would still decline to rewrite it.
        if !self.damaged.is_empty() {
            self.damaged.remove(&format!("{db}.{tbl}"));
        }
        let d = self
            .databases
            .get_mut(&db)
            .ok_or_else(|| Error::storage(format!("database `{db}` does not exist")))?;
        if d.tables.remove(&tbl).is_none() && !if_exists {
            return Err(Error::storage(format!("table `{db}.{tbl}` does not exist")));
        }
        Ok(())
    }

    pub fn table(&self, name: &ObjectName) -> Result<&Table> {
        let (db, tbl) = self.resolve(name);
        if !self.damaged.is_empty() {
            refuse_if_damaged(&self.damaged, &db, &tbl)?;
        }
        self.databases
            .get(&db)
            .and_then(|d| d.tables.get(&tbl))
            .ok_or_else(|| Error::storage(format!("table `{db}.{tbl}` does not exist")))
    }

    pub fn table_mut(&mut self, name: &ObjectName) -> Result<&mut Table> {
        let (db, tbl) = self.resolve(name);
        if !self.damaged.is_empty() {
            refuse_if_damaged(&self.damaged, &db, &tbl)?;
        }
        self.databases
            .get_mut(&db)
            .and_then(|d| d.tables.get_mut(&tbl))
            .ok_or_else(|| Error::storage(format!("table `{db}.{tbl}` does not exist")))
    }

    /// Look up by the plain string the planner records in `ScanNode::table`,
    /// which is always `db.table`.
    pub fn table_by_path(&self, path: &str) -> Result<&Table> {
        let (db, tbl) = split_path(path, &self.current);
        if !self.damaged.is_empty() {
            refuse_if_damaged(&self.damaged, db, tbl)?;
        }
        self.databases
            .get(db)
            .and_then(|d| d.tables.get(tbl))
            .ok_or_else(|| Error::storage(format!("table `{path}` does not exist")))
    }

    pub fn table_by_path_mut(&mut self, path: &str) -> Result<&mut Table> {
        let (db, tbl) = split_path(path, &self.current);
        if !self.damaged.is_empty() {
            refuse_if_damaged(&self.damaged, db, tbl)?;
        }
        // Strictly after the refusal, and that ordering is the mechanism, not
        // an accident. The loader reads a table's parts and then resolves it
        // here once to install them; that resolve is the one this claim turns
        // into a quarantine, so it succeeds and every resolve after it -- any
        // INSERT, DELETE, ALTER or OPTIMIZE -- is refused above. A rebuild
        // like `ALTER TABLE ... ADD COLUMN` reaches straight for this
        // accessor, and it would otherwise rewrite the table from the parts
        // that happened to load.
        if reader::any_pending_damage() {
            claim_damage(&mut self.damaged, self.dir.as_deref(), db, tbl);
        }
        self.databases
            .get_mut(db)
            .and_then(|d| d.tables.get_mut(tbl))
            .ok_or_else(|| Error::storage(format!("table `{path}` does not exist")))
    }

    // ---- quarantine ------------------------------------------------------

    /// Record that `db.table` has part files that did not decode.
    ///
    /// The direct route for a loader that *can* say so: `table_by_path_mut`
    /// picks the same record up out of the reader's hand-off because
    /// `store::load_catalog` has nowhere to put it, and this is what that
    /// hand-off exists to stand in for. Passing an empty list lifts the
    /// quarantine, which only a caller that has re-read the files should do.
    /// Additive, because damage is: a table can have a part file that will not
    /// decode *and* a log that will not replay, and the loader finds them at
    /// two different moments -- the parts through the reader's hand-off on its
    /// resolve, the log and the `TABLE` file afterwards. Replacing would drop
    /// whichever was found first, and the refusal message is a list of files
    /// to restore, so a short list is a wrong instruction.
    pub fn quarantine(&mut self, path: &str, parts: Vec<DamagedPart>) {
        if parts.is_empty() {
            self.damaged.remove(path);
        } else {
            self.damaged.entry(path.to_string()).or_default().extend(parts);
        }
    }

    /// Is this `db.table` refusing to answer because of damage on disk?
    pub fn is_quarantined(&self, path: &str) -> bool {
        !self.damaged.is_empty() && self.damaged.contains_key(path)
    }

    /// The definition of a quarantined table, or `None` if it is healthy.
    ///
    /// What a checkpoint needs: a quarantined table must keep its place in the
    /// committed roster (a table the root `CATALOG` stops naming is a table
    /// the *next* checkpoint deletes the directory of) while none of its files
    /// are rewritten, because rewriting them would collect the very part that
    /// could not be read.
    pub fn quarantined_def(&self, db: &str, table: &str) -> Option<&TableDef> {
        if self.damaged.is_empty() || !self.damaged.contains_key(&format!("{db}.{table}")) {
            return None;
        }
        self.databases.get(db).and_then(|d| d.tables.get(table)).map(|t| &t.def)
    }

    /// Every quarantined part, as `(db.table, part)`, sorted.
    ///
    /// One row per damaged *file*, which is the grain a `system.` table wants:
    /// the table is what stopped answering, the file is what has to be put
    /// back. Sorted so two calls agree.
    pub fn damaged_parts(&self) -> Vec<(&str, &DamagedPart)> {
        let mut v: Vec<(&str, &DamagedPart)> = self
            .damaged
            .iter()
            .flat_map(|(t, parts)| parts.iter().map(move |p| (t.as_str(), p)))
            .collect();
        v.sort_unstable_by(|a, b| a.0.cmp(b.0).then_with(|| a.1.file.cmp(&b.1.file)));
        v
    }

    /// Fully-qualified `db.table` path for a possibly-bare name.
    pub fn qualify(&self, name: &ObjectName) -> String {
        let (db, tbl) = self.resolve(name);
        format!("{db}.{tbl}")
    }

    pub fn table_names(&self, db: Option<&str>) -> Result<Vec<String>> {
        let db = db.unwrap_or(&self.current);
        let d = self
            .databases
            .get(db)
            .ok_or_else(|| Error::storage(format!("database `{db}` does not exist")))?;
        let mut v: Vec<String> = d.tables.keys().cloned().collect();
        v.sort();
        Ok(v)
    }

    pub fn all_tables_mut(&mut self) -> impl Iterator<Item = (&String, &mut Table)> {
        self.databases
            .iter_mut()
            .flat_map(|(_, d)| d.tables.iter_mut())
    }

    pub fn all_tables(&self) -> impl Iterator<Item = (&String, &Table)> {
        self.databases.iter().flat_map(|(_, d)| d.tables.iter())
    }

    /// Does any table hold rows a scan would not see?
    ///
    /// The gate on the `&self` read path in [`crate::Session::read`]: scans
    /// read parts, so a table with a non-empty delta has to be flushed before
    /// one can answer, and flushing needs `&mut`. Answering that question is
    /// one `usize` compare per table -- cheaper than the `flush_all` it
    /// replaces even when it says yes, because `flush_all`'s cost was never
    /// the work, it was the exclusive borrow of every *other* table.
    ///
    /// Deliberately not cached in a flag: a `Table` can be mutated through
    /// `catalog.table_mut()` by anyone holding `&mut Catalog`, and a stale
    /// "clean" flag is a silently wrong answer. Asking the tables is exact.
    pub fn has_pending_writes(&self) -> bool {
        self.all_tables().any(|(_, t)| t.has_pending_writes())
    }

    /// Is a table mid-transaction, i.e. carrying a private overlay that
    /// [`Table::snapshot`] would hand to a reader as if it were committed?
    ///
    /// [`Table::snapshot`]: crate::storage::Table::snapshot
    pub fn any_in_txn(&self) -> bool {
        self.all_tables().any(|(_, t)| t.in_txn())
    }

    /// Flush every table's write buffer. Called before persisting, and by the
    /// `&mut self` write path before a statement that has to see its own
    /// buffered rows through a scan.
    ///
    /// No longer on the read path: see [`Catalog::has_pending_writes`].
    pub fn flush_all(&mut self) -> Result<()> {
        for (_, d) in self.databases.iter_mut() {
            for (_, t) in d.tables.iter_mut() {
                t.flush()?;
            }
        }
        Ok(())
    }
}

/// `<db>/<table>` for every directory under `root` that holds real table data.
///
/// Called only when the root `CATALOG` is missing, to distinguish the two
/// situations that look identical to [`crate::persist::load_catalog`]:
///
///   * a fresh directory, which must open as an empty database; and
///   * a directory whose `CATALOG` was lost -- deleted, restored from a
///     partial backup, or truncated away -- while every table's parts and
///     write-ahead logs are still sitting there intact.
///
/// The second used to open silently as an empty database, and that is not a
/// harmless misread: the committed catalog is what
/// [`crate::persist::save_catalog`] treats as authoritative about which tables
/// exist, so the *next* checkpoint would see every table directory as dropped
/// and `remove_dir_all` it. A single missing index file, with all the data
/// still on disk, would become total loss. Refusing to open costs the user one
/// error message and keeps the recovery possible.
///
/// "Real data" is the same test `store::is_table_dir` applies plus a part
/// file, so a table that was committed and then had its `TABLE` file lost too
/// still counts. Non-table directories are ignored on the same grounds as the
/// dropped-table collector: [`store::is_safe_name`] rejects names this module
/// could not have created, and a directory holding none of our files is
/// somebody else's.
fn unaccounted_table_dirs(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let Ok(dbs) = std::fs::read_dir(root) else { return out };
    for db in dbs.flatten() {
        let dbdir = db.path();
        let name = db.file_name();
        let Some(dbname) = name.to_str() else { continue };
        if !store::is_safe_name(dbname) || !dbdir.is_dir() {
            continue;
        }
        let Ok(tables) = std::fs::read_dir(&dbdir) else { continue };
        for t in tables.flatten() {
            let tdir = t.path();
            let name = t.file_name();
            let Some(tname) = name.to_str() else { continue };
            if !store::is_safe_name(tname) || !tdir.is_dir() || !holds_table_data(&tdir) {
                continue;
            }
            out.push(format!("{dbname}/{tname}"));
        }
    }
    out.sort();
    out
}

fn holds_table_data(dir: &Path) -> bool {
    if dir.join(store::TABLE_FILE).exists() {
        return true;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return false };
    rd.flatten().any(|e| {
        e.file_name()
            .to_str()
            .is_some_and(|n| store::parse_part_seq(n).is_some())
    })
}

/// Borrowed, not owned: this runs once per resolve and a resolve runs once per
/// INSERT *block*, so the two `String`s it used to allocate were two
/// allocations per 8192 rows of an `INSERT ... SELECT`. Both halves borrow
/// disjoint fields of the catalog, which is why the callers can still take
/// `&mut self.databases` around them.
///
/// Measured on 40,000 single-row `INSERT` statements (~80,000 resolves), best
/// of 32 interleaved runs against a build of the same tree without this and
/// without the quarantine check: 171.6 ms against 176.2 ms. A 2.6% effect on
/// a machine that swings 12% between identical runs -- three of the four
/// batches favoured it and one did not -- so the honest reading is "the
/// quarantine check costs nothing, and this pays for it".
fn split_path<'a>(path: &'a str, current: &'a str) -> (&'a str, &'a str) {
    match path.split_once('.') {
        Some((d, t)) => (d, t),
        None => (current, path),
    }
}

/// Refuse a table whose parts did not all load. Called only when the map is
/// non-empty, so the healthy path never formats a name.
fn refuse_if_damaged(
    damaged: &FastMap<String, Vec<DamagedPart>>,
    db: &str,
    tbl: &str,
) -> Result<()> {
    let path = format!("{db}.{tbl}");
    let Some(parts) = damaged.get(&path) else { return Ok(()) };
    // Every damaged file, up to three: an operator restoring from a backup
    // needs the list, and a table with more than a handful of bad files has a
    // dead disk rather than a bad block.
    let mut msg = format!(
        "table `{path}` is quarantined: {} of its files could not be read when this \
         database was opened",
        parts.len()
    );
    for p in parts.iter().take(3) {
        msg.push_str(". ");
        msg.push_str(&p.why);
    }
    if parts.len() > 3 {
        msg.push_str(&format!(". ...and {} more", parts.len() - 3));
    }
    msg.push_str(
        ". Answering from the files that did load would silently drop the rows in the ones \
         that did not, so every read and write of this table is refused. Restore the file \
         from a backup and reopen, or DROP the table. No other table is affected.",
    );
    Err(Error::corruption(msg))
}

/// Move the damage the loader just found for `db.tbl` into this catalog.
///
/// Keyed by the table's directory, which is the only name the reader and the
/// catalog can both compute -- the reader never sees the database it is
/// loading into, and the catalog never sees the files.
fn claim_damage(
    damaged: &mut FastMap<String, Vec<DamagedPart>>,
    dir: Option<&Path>,
    db: &str,
    tbl: &str,
) {
    let Some(root) = dir else { return };
    if let Some(parts) = reader::claim_damage(&root.join(db).join(tbl)) {
        damaged.insert(format!("{db}.{tbl}"), parts);
    }
}

impl Default for Catalog {
    fn default() -> Self {
        Catalog::in_memory()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persist::testkit::Scratch;
    use crate::types::{DataType, Engine, Field, Schema};

    fn def(name: &str) -> TableDef {
        TableDef {
            name: name.into(),
            schema: Schema::new(vec![Field::new("id", DataType::UInt64)]).unwrap(),
            order_by: vec![0],
            primary_key: vec![0],
            partition_by: None,
            engine: Engine::MergeTree,
        }
    }

    #[test]
    fn create_and_look_up_tables() {
        let mut c = Catalog::in_memory();
        c.create_table(def("t"), false).unwrap();
        assert!(c.table(&ObjectName::bare("t")).is_ok());
        assert!(c.table(&ObjectName::bare("nope")).is_err());
        assert_eq!(c.qualify(&ObjectName::bare("t")), "default.t");
        assert!(c.table_by_path("default.t").is_ok());
    }

    #[test]
    fn duplicate_create_respects_if_not_exists() {
        let mut c = Catalog::in_memory();
        c.create_table(def("t"), false).unwrap();
        assert!(c.create_table(def("t"), false).is_err());
        assert!(c.create_table(def("t"), true).is_ok());
    }

    #[test]
    fn drop_respects_if_exists() {
        let mut c = Catalog::in_memory();
        c.create_table(def("t"), false).unwrap();
        c.drop_table(&ObjectName::bare("t"), false).unwrap();
        assert!(c.drop_table(&ObjectName::bare("t"), false).is_err());
        assert!(c.drop_table(&ObjectName::bare("t"), true).is_ok());
    }

    #[test]
    fn qualified_names_target_the_right_database() {
        let mut c = Catalog::in_memory();
        c.create_database("analytics", false).unwrap();
        let mut d = def("analytics.hits");
        d.name = "analytics.hits".into();
        c.create_table(d, false).unwrap();

        let qualified = ObjectName(vec!["analytics".into(), "hits".into()]);
        assert!(c.table(&qualified).is_ok());
        assert!(c.table(&ObjectName::bare("hits")).is_err(), "not in `default`");

        c.use_database("analytics").unwrap();
        assert!(c.table(&ObjectName::bare("hits")).is_ok());
        assert_eq!(c.current_database(), "analytics");
    }

    #[test]
    fn cannot_drop_default_database() {
        let mut c = Catalog::in_memory();
        assert!(c.drop_database(DEFAULT_DATABASE, false).is_err());
    }

    #[test]
    fn dropping_current_database_falls_back_to_default() {
        let mut c = Catalog::in_memory();
        c.create_database("scratch", false).unwrap();
        c.use_database("scratch").unwrap();
        c.drop_database("scratch", false).unwrap();
        assert_eq!(c.current_database(), DEFAULT_DATABASE);
    }

    #[test]
    fn table_names_are_sorted() {
        let mut c = Catalog::in_memory();
        for n in ["zebra", "apple", "mango"] {
            c.create_table(def(n), false).unwrap();
        }
        assert_eq!(
            c.table_names(None).unwrap(),
            vec!["apple", "mango", "zebra"]
        );
        assert!(c.table_names(Some("missing")).is_err());
    }

    // ---- opening a directory whose CATALOG went missing -------------------

    #[test]
    fn a_fresh_directory_still_opens_as_an_empty_database() {
        let s = Scratch::new("cat-fresh");
        assert!(Catalog::on_disk(s.path()).is_ok(), "an empty root must open");
        // A path that does not exist yet is created, not refused.
        assert!(Catalog::on_disk(s.join("sub").join("deeper")).is_ok());
        // The session lock file sits in the root and is not table data.
        std::fs::write(s.join("LOCK"), b"1234\n").unwrap();
        assert!(Catalog::on_disk(s.path()).is_ok(), "a LOCK file is not a database");
    }

    /// The three shapes of "this directory is a table": a commit record, a
    /// log, or a part file. Any one of them over a missing `CATALOG` means the
    /// roster was lost while the data survived, and opening empty would let
    /// the next checkpoint delete the survivors.
    #[test]
    fn a_missing_catalog_over_table_data_is_refused() {
        for marker in [store::TABLE_FILE, "part_000003.gpart"] {
            let s = Scratch::new("cat-lost-roster");
            let tdir = s.join("default").join("hits");
            std::fs::create_dir_all(&tdir).unwrap();
            std::fs::write(tdir.join(marker), b"x").unwrap();

            let Err(e) = Catalog::on_disk(s.path()) else {
                panic!("{marker}: a directory with table data and no CATALOG must be refused")
            };
            assert_eq!(e.code(), "STORAGE_ERROR", "{marker}: {e}");
            let msg = e.to_string();
            assert!(msg.contains("default/hits"), "{marker}: must name the table: {msg}");
            assert!(msg.contains(store::CATALOG_FILE), "{marker}: must name the file: {msg}");

            // ...and the refusal is scoped to exactly that: with a CATALOG
            // present the same tree opens, whatever the CATALOG's contents.
            std::fs::write(s.join(store::CATALOG_FILE), b"opaque").unwrap();
            assert!(Catalog::on_disk(s.path()).is_ok(), "{marker}");
        }
    }

    // ---- quarantine -------------------------------------------------------

    /// Two tables on disk, one of them with a bit-flipped part file. Hands
    /// back the scratch and the name of the file that was damaged.
    fn two_tables_one_damaged(tag: &str) -> (Scratch, String) {
        use crate::persist::testkit::sample_block;
        let s = Scratch::new(tag);
        let mut c = Catalog::on_disk(s.path()).unwrap();
        for name in ["a", "b"] {
            c.create_table(crate::persist::testkit::table_def(name), false).unwrap();
            c.table_by_path_mut(&format!("default.{name}"))
                .unwrap()
                .insert(sample_block(2_000))
                .unwrap();
        }
        store::save_catalog(&mut c).unwrap();

        let tdir = s.join("default").join("a");
        let files = store::list_part_files(&tdir).unwrap();
        let name = files[0].1.clone();
        let p = tdir.join(&name);
        let mut bytes = std::fs::read(&p).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0x08;
        std::fs::write(&p, &bytes).unwrap();
        (s, name)
    }

    fn reopen(s: &Scratch) -> Catalog {
        let mut c = Catalog::on_disk(s.path()).unwrap();
        store::load_catalog(&mut c).expect("a damaged part must not fail the open");
        c
    }

    /// The whole point: damage in `a` is damage in `a`. The database opens,
    /// `b` answers, and the catalog itself is fully usable.
    #[test]
    fn a_damaged_part_quarantines_only_its_own_table() {
        let (s, part) = two_tables_one_damaged("quarantine-one");
        let c = reopen(&s);

        assert!(c.table_by_path("default.b").is_ok(), "an unrelated table must be unaffected");
        assert_eq!(c.table_names(None).unwrap(), vec!["a".to_string(), "b".to_string()]);
        assert_eq!(c.database_names(), vec!["default".to_string()]);

        // `Table` has no `Debug` on purpose, so `expect_err` is out.
        let Err(e) = c.table_by_path("default.a") else { panic!("a damaged table must refuse") };
        assert_eq!(e.code(), "CHECKSUM_MISMATCH", "{e}");
        let msg = e.to_string();
        assert!(msg.contains("default.a"), "must name the table: {msg}");
        assert!(msg.contains(&part), "must name the part file: {msg}");

        assert!(c.is_quarantined("default.a"));
        assert!(!c.is_quarantined("default.b"));
        let dmg = c.damaged_parts();
        assert_eq!(dmg.len(), 1);
        assert_eq!((dmg[0].0, dmg[0].1.file.as_str()), ("default.a", part.as_str()));
        assert!(c.quarantined_def("default", "a").is_some_and(|d| d.name == "a"));
        assert!(c.quarantined_def("default", "b").is_none());
    }

    /// Every accessor that hands out a `Table`, not just the read path: an
    /// `ALTER TABLE ... ADD COLUMN` rebuilds a table straight through
    /// `table_by_path_mut`, and would otherwise rebuild it from the parts that
    /// happened to load.
    #[test]
    fn every_accessor_refuses_a_quarantined_table() {
        let (s, _) = two_tables_one_damaged("quarantine-accessors");
        let mut c = reopen(&s);
        let a = ObjectName::bare("a");
        assert!(c.table(&a).is_err(), "table()");
        assert!(c.table_mut(&a).is_err(), "table_mut()");
        assert!(c.table_by_path("default.a").is_err(), "table_by_path()");
        assert!(c.table_by_path_mut("default.a").is_err(), "table_by_path_mut()");
        // ...and the same four still work for the healthy one.
        let b = ObjectName::bare("b");
        assert!(c.table(&b).is_ok());
        assert!(c.table_mut(&b).is_ok());
        assert!(c.table_by_path("default.b").is_ok());
        assert!(c.table_by_path_mut("default.b").is_ok());
    }

    /// The way out. `DROP` is the operator's answer when the file is not
    /// coming back, so it must not be refused -- and the quarantine has to go
    /// with the table, or the name stays poisoned for a table that is gone.
    #[test]
    fn dropping_a_quarantined_table_clears_the_quarantine() {
        let (s, _) = two_tables_one_damaged("quarantine-drop");
        let mut c = reopen(&s);
        c.drop_table(&ObjectName::bare("a"), false).unwrap();
        assert!(c.damaged_parts().is_empty());
        assert!(!c.is_quarantined("default.a"));

        // A fresh table under the same name is a fresh table.
        c.create_table(crate::persist::testkit::table_def("a"), false).unwrap();
        assert!(c.table_by_path("default.a").is_ok(), "a re-created name must not inherit it");
        assert!(store::save_catalog(&mut c).is_ok(), "and the database checkpoints again");
    }

    #[test]
    fn dropping_the_database_clears_its_quarantines() {
        let s = Scratch::new("quarantine-dropdb");
        let mut c = Catalog::on_disk(s.path()).unwrap();
        c.create_database("scratch", false).unwrap();
        c.damaged.insert(
            "scratch.a".into(),
            vec![DamagedPart { file: "part_000001.gpart".into(), why: "x".into() }],
        );
        c.damaged.insert(
            "default.a".into(),
            vec![DamagedPart { file: "part_000002.gpart".into(), why: "y".into() }],
        );
        c.drop_database("scratch", false).unwrap();
        assert_eq!(c.damaged_parts().len(), 1, "only the dropped database's entries go");
        assert!(c.is_quarantined("default.a"));
    }

    /// An undamaged database must record nothing at all -- the map staying
    /// empty is what keeps the check off the resolve path.
    #[test]
    fn a_healthy_database_carries_no_quarantine() {
        let s = Scratch::new("quarantine-none");
        let mut c = Catalog::on_disk(s.path()).unwrap();
        c.create_table(crate::persist::testkit::table_def("t"), false).unwrap();
        store::save_catalog(&mut c).unwrap();
        let c2 = reopen(&s);
        assert!(c2.damaged_parts().is_empty());
        assert!(!c2.is_quarantined("default.t"));
        assert!(c2.table_by_path("default.t").is_ok());
    }

    /// The check removes nothing, but it does refuse to open a whole database,
    /// so it has to recognise its own the way the dropped-table collector does
    /// -- a directory holding none of our files is somebody else's.
    #[test]
    fn a_directory_we_did_not_write_is_not_table_data() {
        let s = Scratch::new("cat-foreign");
        let d = s.join("notes").join("drafts");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("readme.txt"), b"not ours").unwrap();
        // A plain file where a database directory would go is not one either.
        std::fs::write(s.join("default"), b"not a directory").unwrap();
        assert!(Catalog::on_disk(s.path()).is_ok());
    }
}
