//! Databases and tables, and the name resolution over them.
//!
//! The catalog owns every [`Table`]. Everything above it -- binder, planner,
//! executor -- borrows tables from here rather than owning storage, which
//! keeps the "who can mutate a table" question answerable: only code holding
//! `&mut Catalog`.

use std::path::{Path, PathBuf};

use crate::common::{Error, FastMap, Result};
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
        }
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
        self.databases
            .get(&db)
            .and_then(|d| d.tables.get(&tbl))
            .ok_or_else(|| Error::storage(format!("table `{db}.{tbl}` does not exist")))
    }

    pub fn table_mut(&mut self, name: &ObjectName) -> Result<&mut Table> {
        let (db, tbl) = self.resolve(name);
        self.databases
            .get_mut(&db)
            .and_then(|d| d.tables.get_mut(&tbl))
            .ok_or_else(|| Error::storage(format!("table `{db}.{tbl}` does not exist")))
    }

    /// Look up by the plain string the planner records in `ScanNode::table`,
    /// which is always `db.table`.
    pub fn table_by_path(&self, path: &str) -> Result<&Table> {
        let (db, tbl) = split_path(path, &self.current);
        self.databases
            .get(&db)
            .and_then(|d| d.tables.get(&tbl))
            .ok_or_else(|| Error::storage(format!("table `{path}` does not exist")))
    }

    pub fn table_by_path_mut(&mut self, path: &str) -> Result<&mut Table> {
        let (db, tbl) = split_path(path, &self.current);
        self.databases
            .get_mut(&db)
            .and_then(|d| d.tables.get_mut(&tbl))
            .ok_or_else(|| Error::storage(format!("table `{path}` does not exist")))
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
    if dir.join(store::TABLE_FILE).exists() || dir.join(store::WAL_FILE).exists() {
        return true;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return false };
    rd.flatten().any(|e| {
        e.file_name()
            .to_str()
            .is_some_and(|n| store::parse_part_seq(n).is_some())
    })
}

fn split_path(path: &str, current: &str) -> (String, String) {
    match path.split_once('.') {
        Some((d, t)) => (d.to_string(), t.to_string()),
        None => (current.to_string(), path.to_string()),
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
        for marker in [store::TABLE_FILE, store::WAL_FILE, "part_000003.gpart"] {
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
