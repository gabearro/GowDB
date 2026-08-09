//! Operability, end to end: taking a copy of a running database, and looking
//! inside one.
//!
//! Everything here drives the shipped binary with `std::process::Command`, or
//! the public `Session`/`Db` API, because both features had a specific way of
//! being built and never reaching a user. A `backup.rs` full of green unit
//! tests that no statement calls is worth nothing, and a `system.parts` that
//! is right about a `Catalog` in a test fixture and wrong about the files on
//! disk is worse than nothing.
//!
//! The four claims being pinned:
//!
//!   1. **A backup is one instant.** Taken against a pinned snapshot, not a
//!      directory walk that races a writer -- which is exactly why two of
//!      eight `cp -r` copies of a live instance were unopenable, with exit 0.
//!      Tested twice: once with writes on both sides of the `BACKUP` in one
//!      script, and once with a writer thread actually running against the
//!      same `Db` while backups are taken.
//!   2. **Restore refuses rather than half-clobbers.** Into the live database,
//!      into any non-empty directory, and from a damaged archive.
//!   3. **Verify actually verifies.** A single flipped byte anywhere in an
//!      archive is reported, with a non-zero exit status, and the restore of
//!      that archive is refused.
//!   4. **The system tables agree with reality.** `system.parts` is checked
//!      against the `.gpart` files present in the data directory, and its row
//!      counts against `count()` on the table itself.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use granular::{Db, Session};

const BIN: &str = env!("CARGO_BIN_EXE_granular");

// ------------------------------------------------------------------ harness

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let d = std::env::temp_dir().join(format!("granular-backup-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("scratch dir");
        Scratch(d)
    }
    fn at(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
    fn s(&self, name: &str) -> String {
        self.at(name).to_string_lossy().into_owned()
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
            code: o.status.code().unwrap_or(-1),
            out: String::from_utf8_lossy(&o.stdout).into_owned(),
            err: String::from_utf8_lossy(&o.stderr).into_owned(),
        }
    }
    fn ok(self) -> Run {
        assert_eq!(self.code, 0, "stdout:\n{}\nstderr:\n{}", self.out, self.err);
        self
    }
    fn fails(self) -> Run {
        assert_eq!(self.code, 1, "expected failure, got:\n{}\n{}", self.out, self.err);
        self
    }
    /// The statements succeeded, but the process may still exit 1 because the
    /// checkpoint it takes on the way out cannot rewrite a quarantined table.
    /// That is `degraded_open`'s documented behaviour, not this wave's, and a
    /// query about the damage still has to answer.
    fn ok_despite_quarantine(self) -> Run {
        assert!(
            self.code == 0 || self.err.contains("quarantined"),
            "exit {}\nstdout:\n{}\nstderr:\n{}",
            self.code,
            self.out,
            self.err
        );
        self
    }
    /// Every cell of a `tsv` result, one row per line.
    fn rows(&self) -> Vec<Vec<String>> {
        self.out
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| l.split('\t').map(str::to_string).collect())
            .collect()
    }
    fn one(&self) -> String {
        let r = self.rows();
        assert_eq!(r.len(), 1, "expected one row, got {r:?}");
        r[0].join("\t")
    }
}

/// Run SQL against `db` (or in memory when `db` is `None`), machine-readable.
fn sql(db: Option<&Path>, q: &str) -> Run {
    let mut c = Command::new(BIN);
    if let Some(d) = db {
        c.arg("--data").arg(d);
    }
    Run::of(
        c.args(["--format", "tsv", "--no-header", "-q", q])
            .output()
            .expect("spawn granular"),
    )
}

/// The piped-script path: statements arrive on stdin while the process runs,
/// which is the closest a single-writer engine gets to "live" from outside.
fn pipe(db: &Path, script: &str) -> Run {
    let mut child = Command::new(BIN)
        .args(["--data", &db.to_string_lossy(), "--format", "tsv", "--no-header"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn granular");
    child.stdin.take().expect("stdin").write_all(script.as_bytes()).expect("feed script");
    Run::of(child.wait_with_output().expect("wait"))
}

const DDL: &str = "CREATE TABLE hits (id UInt64, host String, ms UInt32) \
                   ENGINE = MergeTree ORDER BY id PRIMARY KEY id";

/// `INSERT` of `n` rows starting at `from`, as one statement.
fn insert(from: u64, n: u64) -> String {
    let mut s = String::from("INSERT INTO hits VALUES ");
    for i in 0..n {
        let id = from + i;
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("({id},'h{}',{})", id % 7, id * 3));
    }
    s.push(';');
    s
}

/// The `.gpart` files actually present under a data directory, as
/// `db/table/file`.
fn part_files(root: &Path) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let Ok(dbs) = std::fs::read_dir(root) else { return out };
    for db in dbs.flatten() {
        let Ok(tables) = std::fs::read_dir(db.path()) else { continue };
        for t in tables.flatten() {
            let Ok(files) = std::fs::read_dir(t.path()) else { continue };
            for f in files.flatten() {
                let name = f.file_name().to_string_lossy().into_owned();
                if name.ends_with(".gpart") {
                    out.insert(format!(
                        "{}/{}/{name}",
                        db.file_name().to_string_lossy(),
                        t.file_name().to_string_lossy()
                    ));
                }
            }
        }
    }
    out
}

/// Flip one byte in the middle of a file. Enough to fail a frame checksum and
/// not enough to be visible in the header or the footer, which is the case a
/// header-only "verify" would wave through.
fn flip_middle(p: &Path) {
    let mut b = std::fs::read(p).expect("read");
    let at = b.len() / 2;
    b[at] ^= 0xff;
    std::fs::write(p, &b).expect("write");
}

// ------------------------------------------------------------------ backup

/// The headline: back up a database that is being written to, restore the
/// archive somewhere else, and get back exactly what the database held at the
/// moment the backup was taken -- not before, and not after.
#[test]
fn a_backup_taken_mid_stream_restores_the_database_as_it_was() {
    let s = Scratch::new("midstream");
    let db = s.at("db");
    let arc = s.s("a.gbak");

    // Writes before the backup, the backup, the count at that instant, then
    // more writes after it -- all inside one live process, on one connection,
    // with the archive taken between two of them.
    let mut script = format!("{DDL};\n");
    for k in 0..6 {
        script.push_str(&insert(k * 500 + 1, 500));
        script.push('\n');
    }
    script.push_str(&format!("BACKUP TO '{arc}';\n"));
    script.push_str("SELECT count(), sum(id) FROM hits;\n");
    for k in 6..12 {
        script.push_str(&insert(k * 500 + 1, 500));
        script.push('\n');
    }
    script.push_str("SELECT count(), sum(id) FROM hits;\n");
    let r = pipe(&db, &script).ok();
    let rows = r.rows();
    // The backup's own report row, then the two counts.
    let at_backup = rows[rows.len() - 2].clone();
    let at_end = rows[rows.len() - 1].clone();
    assert_eq!(at_backup, vec!["3000".to_string(), "4501500".to_string()]);
    assert_ne!(at_backup, at_end, "the writes after the backup did nothing");

    let out = s.s("restored");
    sql(None, &format!("RESTORE FROM '{arc}' TO '{out}'")).ok();
    let back = sql(Some(Path::new(&out)), "SELECT count(), sum(id) FROM hits").ok();
    assert_eq!(back.rows()[0], at_backup, "the restore is not the database at backup time");

    // Row for row, not just the aggregate: an archive that lost a column or
    // reordered a granule would still add up.
    let want = sql(Some(&db), "SELECT id, host, ms FROM hits WHERE id <= 3000 ORDER BY id").ok();
    let got = sql(Some(Path::new(&out)), "SELECT id, host, ms FROM hits ORDER BY id").ok();
    assert_eq!(got.out, want.out);
}

/// The same claim, with a writer that is genuinely running: a thread inserting
/// through `Db` while the main thread takes backups of the same instance.
///
/// The ids are consecutive from 1 and each `INSERT` is one statement, so the
/// table holds `1..=k` at every instant a statement boundary can be observed.
/// A backup that raced a writer -- read one table's commit record and another
/// table's parts, or half of a `PartSet` -- would restore with a hole, and
/// `count() = max(id)` is what says there is none.
///
/// The overlap is *waited for*, not hoped for. Six backups of a 1000-row table
/// can all finish before the writer's first batch lands, and a trial that saw
/// no rows disproved nothing; it used to fail the run anyway. Now each backup
/// is paced on an observed advance of the writer, and a writer that still
/// outruns the backups is given more work rather than called a failure.
#[test]
fn a_backup_taken_while_a_writer_runs_is_a_consistent_prefix() {
    for batches in [40u64, 160, 640] {
        if live_backup_race(batches) {
            return;
        }
    }
    panic!("the writer outran the backups at every size: the race never happened");
}

/// One trial of the above. Returns false if the writer finished before a single
/// backup could overlap it -- nothing was observed, so the caller escalates.
fn live_backup_race(batches: u64) -> bool {
    use std::sync::atomic::{AtomicU64, Ordering};
    const PER: u64 = 25;

    let s = Scratch::new(&format!("live-{batches}"));
    let db = Db::open(s.at("db")).expect("open");
    db.execute(DDL).expect("ddl");

    let landed = AtomicU64::new(0);
    let taken = std::thread::scope(|scope| {
        let (w, l) = (&db, &landed);
        scope.spawn(move || {
            for k in 0..batches {
                w.execute(&insert(k * PER + 1, PER)).expect("insert");
                l.store(k + 1, Ordering::Release);
            }
        });
        // One backup per observed advance of the writer: every archive is a
        // distinct instant, and every one of them was taken with the writer
        // still going. Backups race the writer for the same lock, so each is
        // whatever the table was between two of its statements.
        let (mut taken, mut at) = (Vec::new(), 0);
        while taken.len() < 6 {
            let now = landed.load(Ordering::Acquire);
            if now >= batches {
                break; // writer done: a further backup would race nothing
            }
            if now == at {
                std::thread::yield_now();
                continue;
            }
            at = now;
            let p = s.s(&format!("live-{}.gbak", taken.len()));
            db.writer().query(&format!("BACKUP TO '{p}'")).expect("backup");
            taken.push(p);
        }
        taken
    });
    drop(db);

    let (mut seen, mut last) = (false, 0u64);
    for (i, p) in taken.iter().enumerate() {
        sql(None, &format!("VERIFY BACKUP '{p}'")).ok();
        let out = s.s(&format!("live-out-{i}"));
        sql(None, &format!("RESTORE FROM '{p}' TO '{out}'")).ok();
        let r = sql(Some(Path::new(&out)), "SELECT count(), max(id), min(id) FROM hits").ok();
        let row = r.rows().remove(0);
        let (n, hi) = (row[0].parse::<u64>().unwrap(), row[1].parse::<u64>().unwrap_or(0));
        if n == 0 {
            continue; // taken before the first batch landed
        }
        assert_eq!(row[2], "1", "archive {i} lost the head of the table");
        assert_eq!(n, hi, "archive {i} has {n} rows but a max id of {hi}: it has a hole");
        assert_eq!(n % PER, 0, "archive {i} caught half of an INSERT: {n} rows");
        // Taken in order against a table that only grows.
        assert!(n >= last, "archive {i} went backwards: {n} rows after {last}");
        (seen, last) = (true, n);
    }
    seen
}

#[test]
fn verify_reports_a_corrupted_archive_and_the_restore_of_one_is_refused() {
    let s = Scratch::new("corrupt");
    let db = s.at("db");
    let arc = s.s("a.gbak");
    pipe(&db, &format!("{DDL};\n{}\nBACKUP TO '{arc}';\n", insert(1, 3_000))).ok();
    sql(None, &format!("VERIFY BACKUP '{arc}'")).ok();

    flip_middle(Path::new(&arc));
    let bad = sql(None, &format!("VERIFY BACKUP '{arc}'")).fails();
    assert!(bad.err.contains("damaged"), "{}", bad.err);
    assert!(bad.err.contains(".gpart"), "the report must name the part: {}", bad.err);

    let out = s.s("restored");
    let r = sql(None, &format!("RESTORE FROM '{arc}' TO '{out}'")).fails();
    assert!(r.err.contains("damaged"), "{}", r.err);
    assert!(
        !Path::new(&out).join("default").exists(),
        "a refused restore must leave nothing behind"
    );

    // And a file that is not an archive at all.
    let junk = s.s("junk.gbak");
    std::fs::write(&junk, vec![0u8; 8192]).unwrap();
    sql(None, &format!("VERIFY BACKUP '{junk}'")).fails();
}

#[test]
fn restore_refuses_the_live_database_and_any_occupied_directory() {
    let s = Scratch::new("refuse");
    let db = s.at("db");
    let arc = s.s("a.gbak");
    pipe(&db, &format!("{DDL};\n{}\nBACKUP TO '{arc}';\n", insert(1, 200))).ok();

    // Into the database this very session has open.
    let r = sql(
        Some(&db),
        &format!("RESTORE FROM '{arc}' TO '{}'", db.to_string_lossy()),
    )
    .fails();
    assert!(r.err.contains("has that database open"), "{}", r.err);

    // Into a directory that merely has something in it.
    let occupied = s.at("occupied");
    std::fs::create_dir_all(&occupied).unwrap();
    std::fs::write(occupied.join("CATALOG"), b"someone else's database").unwrap();
    let r = sql(None, &format!("RESTORE FROM '{arc}' TO '{}'", occupied.to_string_lossy())).fails();
    assert!(r.err.contains("not empty"), "{}", r.err);
    assert_eq!(
        std::fs::read(occupied.join("CATALOG")).unwrap(),
        b"someone else's database",
        "the refused restore wrote into it anyway"
    );

    // Without a target at all.
    let r = sql(Some(&db), &format!("RESTORE FROM '{arc}'")).fails();
    assert!(r.err.contains("TO '<directory>'"), "{}", r.err);

    // The original is untouched throughout.
    assert_eq!(sql(Some(&db), "SELECT count() FROM hits").ok().one(), "200");
}

/// Incremental: the parts the base already holds are named, not stored, and
/// the pair still restores whole.
#[test]
fn an_incremental_backup_stores_only_the_new_parts() {
    let s = Scratch::new("incr");
    let db = s.at("db");
    let full = s.s("full.gbak");
    let incr = s.s("incr.gbak");

    let mut script = format!("{DDL};\n");
    for k in 0..4 {
        script.push_str(&insert(k * 2_000 + 1, 2_000));
        script.push_str("\nOPTIMIZE TABLE hits;\n");
    }
    script.push_str(&format!("BACKUP TO '{full}';\n"));
    for k in 4..6 {
        script.push_str(&insert(k * 2_000 + 1, 2_000));
        script.push_str("\nOPTIMIZE TABLE hits;\n");
    }
    script.push_str(&format!("BACKUP TO '{incr}' INCREMENTAL FROM '{full}';\n"));
    let r = pipe(&db, &script).ok();
    let reports: Vec<Vec<String>> = r.rows().into_iter().filter(|row| row.len() == 6).collect();
    assert_eq!(reports.len(), 2, "expected two BACKUP reports, got {:?}", r.rows());
    let (base, delta) = (&reports[0], &reports[1]);
    let reused: u64 = delta[5].parse().unwrap();
    assert!(reused > 0, "nothing was reused from the base: {delta:?}");
    let (base_bytes, delta_bytes): (u64, u64) =
        (base[4].parse().unwrap(), delta[4].parse().unwrap());
    assert!(
        delta_bytes < base_bytes,
        "the incremental archive ({delta_bytes}) is not smaller than the full one ({base_bytes})"
    );

    sql(None, &format!("VERIFY BACKUP '{incr}'")).ok();
    let out = s.s("restored");
    sql(None, &format!("RESTORE FROM '{incr}' TO '{out}'")).ok();
    assert_eq!(
        sql(Some(Path::new(&out)), "SELECT count(), sum(id) FROM hits").ok().one(),
        sql(Some(&db), "SELECT count(), sum(id) FROM hits").ok().one()
    );

    // And an incremental archive is worthless without its base, which it says
    // rather than restoring a database with parts missing.
    std::fs::remove_file(&full).unwrap();
    let e = sql(None, &format!("VERIFY BACKUP '{incr}'")).fails();
    assert!(e.err.contains("full.gbak"), "{}", e.err);
}

/// The rows must come from the snapshot, not from what happens to be on disk:
/// a table with rows still in the delta must archive them.
#[test]
fn rows_that_are_still_buffered_reach_the_archive() {
    let s = Scratch::new("delta");
    let db = s.at("db");
    let arc = s.s("a.gbak");
    // Small enough that nothing auto-flushes, and no OPTIMIZE anywhere.
    pipe(&db, &format!("{DDL};\n{}\nBACKUP TO '{arc}';\n", insert(1, 5))).ok();
    let out = s.s("restored");
    sql(None, &format!("RESTORE FROM '{arc}' TO '{out}'")).ok();
    assert_eq!(sql(Some(Path::new(&out)), "SELECT count() FROM hits").ok().one(), "5");
}

/// Schema is data too. A database with no tables, and a table with no rows,
/// both have to come back -- the root `CATALOG` remembers empty databases on
/// purpose, so an archive that dropped them would silently lose a
/// `CREATE DATABASE`.
#[test]
fn empty_databases_and_empty_tables_survive_the_round_trip() {
    let s = Scratch::new("emptydb");
    let db = s.at("db");
    let arc = s.s("a.gbak");
    pipe(
        &db,
        &format!(
            "CREATE DATABASE analytics;\n\
             CREATE DATABASE staging;\n\
             CREATE TABLE analytics.events (id UInt64, kind String) \
               ENGINE = MergeTree ORDER BY id;\n\
             BACKUP TO '{arc}';\n"
        ),
    )
    .ok();
    let out = s.s("restored");
    sql(None, &format!("RESTORE FROM '{arc}' TO '{out}'")).ok();
    let o = Path::new(&out);
    let mut dbs = sql(Some(o), "SHOW DATABASES").ok().rows();
    dbs.sort();
    assert_eq!(dbs, vec![vec!["analytics"], vec!["default"], vec!["staging"]]);
    assert_eq!(sql(Some(o), "SELECT count() FROM analytics.events").ok().one(), "0");
    assert_eq!(
        sql(Some(o), "SELECT name, type FROM system.columns WHERE table = 'events'")
            .ok()
            .rows(),
        vec![vec!["id", "UInt64"], vec!["kind", "String"]]
    );
}

/// A quarantined table cannot be archived: the parts that decoded are not the
/// table, and an archive of them would be silently short.
#[test]
fn a_backup_of_a_damaged_database_is_refused_rather_than_taken_short() {
    let s = Scratch::new("damaged-backup");
    let db = s.at("db");
    pipe(&db, &format!("{DDL};\n{}\nOPTIMIZE TABLE hits;\n", insert(1, 3_000))).ok();
    let victim = part_files(&db).into_iter().next().expect("a part file");
    flip_middle(&db.join(&victim));

    let r = sql(Some(&db), &format!("BACKUP TO '{}'", s.s("nope.gbak"))).fails();
    assert!(r.err.contains("quarantined"), "{}", r.err);
    assert!(!s.at("nope.gbak").exists(), "a refused backup must leave no archive");
}

// ------------------------------------------------------------ system tables

/// The claim that matters most: `system.parts` and the directory listing are
/// the same set of names, and the row counts are the same numbers `count()`
/// reports.
#[test]
fn system_parts_agrees_with_the_files_on_disk() {
    let s = Scratch::new("parts");
    let db = s.at("db");
    let mut script = format!("{DDL};\n");
    for k in 0..3 {
        script.push_str(&insert(k * 4_000 + 1, 4_000));
        script.push_str("\nOPTIMIZE TABLE hits;\n");
    }
    pipe(&db, &script).ok();

    let on_disk = part_files(&db);
    assert!(on_disk.len() >= 3, "expected several parts, found {on_disk:?}");

    let named: BTreeSet<String> = sql(
        Some(&db),
        "SELECT database || '/' || table || '/' || name FROM system.parts \
         WHERE state = 'active' ORDER BY 1",
    )
    .ok()
    .rows()
    .into_iter()
    .map(|r| r[0].clone())
    .collect();
    assert_eq!(named, on_disk, "system.parts disagrees with the directory");

    // The counters, against the table itself.
    let totals = sql(
        Some(&db),
        "SELECT sum(live_rows), sum(granules) > 0, sum(data_bytes) > 0 FROM system.parts",
    )
    .ok();
    let want = sql(Some(&db), "SELECT count() FROM hits").ok().one();
    assert_eq!(totals.rows()[0][0], want);
    assert_eq!(totals.rows()[0][1], "true");
    assert_eq!(totals.rows()[0][2], "true");

    // And `system.tables` agrees with both.
    let t = sql(
        Some(&db),
        "SELECT parts, rows, engine, sorting_key, primary_key, columns \
         FROM system.tables WHERE name = 'hits'",
    )
    .ok();
    let row = t.rows().remove(0);
    assert_eq!(row[0].parse::<usize>().unwrap(), on_disk.len());
    assert_eq!(row[1], want);
    assert_eq!(row[2], "MergeTree");
    assert_eq!(row[3], "id");
    assert_eq!(row[4], "id");
    assert_eq!(row[5], "3");
}

/// A deleted row must show up as a deleted row, because "why is this table
/// still 4 GB" is the question `system.parts` exists to answer.
#[test]
fn system_parts_reports_tombstones() {
    let s = Scratch::new("tombstones");
    let db = s.at("db");
    pipe(
        &db,
        &format!(
            "{DDL};\n{}\nOPTIMIZE TABLE hits;\nALTER TABLE hits DELETE WHERE id <= 300;\n",
            insert(1, 3_000)
        ),
    )
    .ok();
    let r = sql(
        Some(&db),
        "SELECT sum(rows), sum(live_rows), sum(deleted_rows) FROM system.parts",
    )
    .ok();
    let row = r.rows().remove(0);
    assert_eq!(row[0], "3000");
    assert_eq!(row[1], "2700");
    assert_eq!(row[2], "300");
    assert_eq!(sql(Some(&db), "SELECT count() FROM hits").ok().one(), "2700");
}

/// The quarantine the reader tracks has to be visible from SQL, or an operator
/// with a refusing table has nowhere to look.
#[test]
fn system_parts_and_tables_show_a_quarantined_file() {
    let s = Scratch::new("quarantine");
    let db = s.at("db");
    pipe(&db, &format!("{DDL};\n{}\nOPTIMIZE TABLE hits;\n", insert(1, 3_000))).ok();
    let victim = part_files(&db).into_iter().next().expect("a part file");
    let file = victim.rsplit('/').next().unwrap().to_string();
    flip_middle(&db.join(&victim));

    let r = sql(
        Some(&db),
        "SELECT database, table, name, state, reason != '' FROM system.parts \
         WHERE state = 'damaged'",
    )
    .ok_despite_quarantine();
    let row = r.rows().remove(0);
    assert_eq!(row[0], "default");
    assert_eq!(row[1], "hits");
    assert_eq!(row[2], file);
    assert_eq!(row[3], "damaged");
    assert_eq!(row[4], "true", "the damage must carry the reader's diagnosis");

    assert_eq!(
        sql(Some(&db), "SELECT quarantined FROM system.tables WHERE name = 'hits'")
            .ok_despite_quarantine()
            .one(),
        "1"
    );
    // The table itself still refuses, which is the behaviour these rows explain.
    sql(Some(&db), "SELECT count() FROM hits").fails();
}

#[test]
fn system_columns_matches_the_catalog_and_information_schema_mirrors_it() {
    let s = Scratch::new("columns");
    let db = s.at("db");
    sql(
        Some(&db),
        "CREATE TABLE t (id UInt64, note Nullable(String) DEFAULT 'x', at DateTime) \
         ENGINE = MergeTree ORDER BY (id, at) PRIMARY KEY id",
    )
    .ok();

    let r = sql(
        Some(&db),
        "SELECT name, position, type, default_expression, is_nullable, \
                is_in_primary_key, is_in_sorting_key \
           FROM system.columns WHERE table = 't' ORDER BY position",
    )
    .ok();
    assert_eq!(
        r.rows(),
        vec![
            vec!["id", "1", "UInt64", "", "0", "1", "1"],
            vec!["note", "2", "Nullable(String)", "'x'", "1", "0", "0"],
            vec!["at", "3", "DateTime", "", "0", "0", "1"],
        ]
    );

    // The standard-conforming alias carries the same facts under the names a
    // tool looks for.
    let i = sql(
        Some(&db),
        "SELECT table_catalog, table_schema, table_name, column_name, ordinal_position, \
                is_nullable, data_type \
           FROM information_schema.columns WHERE table_name = 't' ORDER BY ordinal_position",
    )
    .ok();
    assert_eq!(
        i.rows(),
        vec![
            vec!["default", "default", "t", "id", "1", "NO", "UInt64"],
            vec!["default", "default", "t", "note", "2", "YES", "Nullable(String)"],
            vec!["default", "default", "t", "at", "3", "NO", "DateTime"],
        ]
    );
    assert_eq!(
        sql(Some(&db), "SELECT table_name, table_type FROM information_schema.tables").ok().one(),
        "t\tBASE TABLE"
    );
}

/// The counters the engine has always computed and always thrown away.
#[test]
fn system_query_log_carries_the_stats_the_engine_already_computed() {
    let s = Scratch::new("querylog");
    let db = s.at("db");
    let script = format!(
        "{DDL};\n{}\nOPTIMIZE TABLE hits;\n\
         SELECT count() FROM hits WHERE id = 17;\n\
         SELECT nope FROM hits;\n\
         SELECT kind, rows, granules_read > 0, granules_pruned > 0, error \
           FROM system.query_log WHERE query LIKE '%id = 17%' OR error != '' \
           ORDER BY event_time;\n",
        insert(1, 20_000)
    );
    // The bad statement makes the script fail as a whole; the log query before
    // it does not run, so this is two passes.
    pipe(&db, &script);
    let r = pipe(
        &db,
        "SELECT kind, rows, granules_read > 0, granules_pruned > 0, error != '' \
           FROM system.query_log WHERE query != '' ORDER BY event_time;",
    )
    .ok();
    // A fresh process: the only statement in its log is the one it just ran,
    // which proves the log is per-session and not persisted anywhere.
    assert_eq!(r.rows().len(), 0, "the log must not survive the process: {:?}", r.rows());

    let r = pipe(
        &db,
        "SELECT count() FROM hits WHERE id = 17;\n\
         SELECT nope FROM hits;\n",
    );
    assert_eq!(r.code, 1, "the bad column must fail the script");

    // Now in one session: run both, then read the log.
    let mut sess = Session::open(&db).expect("open");
    sess.query("SELECT count() FROM hits WHERE id = 17").expect("point query");
    assert!(sess.query("SELECT nope FROM hits").is_err());
    let log = sess
        .query(
            "SELECT kind, rows, granules_read, granules_pruned, error \
               FROM system.query_log ORDER BY event_time",
        )
        .expect("query log");
    let rows = log.to_values();
    // Newest first, and the log query itself is not in its own answer.
    let text = format!("{rows:?}");
    assert!(text.contains("SELECT"), "{text}");
    let failed = rows.iter().find(|r| format!("{:?}", r[4]).contains("nope"));
    assert!(failed.is_some(), "a failed statement must be logged: {text}");
    let scanned = rows
        .iter()
        .filter_map(|r| match (&r[2], &r[3]) {
            (granular::Value::UInt(a), granular::Value::UInt(b)) => Some(a + b),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    assert!(scanned > 0, "granule counters never reached the log: {text}");
}

#[test]
fn system_settings_is_queryable_and_reflects_a_set() {
    let s = Scratch::new("settings");
    let db = s.at("db");
    let r = pipe(
        &db,
        "SET max_memory_usage = 123456789;\n\
         SELECT value, default != value FROM system.settings WHERE name = 'max_memory_usage';\n",
    )
    .ok();
    assert_eq!(r.one(), "123456789\ttrue");
    // Every setting `SHOW SETTINGS` knows is in the table, and vice versa.
    let n = sql(Some(&db), "SELECT count() FROM system.settings").ok().one();
    let shown = Run::of(
        Command::new(BIN)
            .args(["--format", "tsv", "--no-header", "-q", "SHOW SETTINGS"])
            .output()
            .expect("spawn"),
    )
    .ok();
    assert_eq!(n.parse::<usize>().unwrap(), shown.rows().len());
}

/// A qualified name whose last component is `settings` is a *name*.
///
/// The extended-statement hook claimed any unquoted `SETTINGS` token as the
/// start of a settings clause, so `system.settings` -- and, already before
/// this wave, any user table called `settings` -- died on "expected a setting
/// name". Both directions are pinned here: the name resolves, and a real
/// trailing `SETTINGS` clause still does what it always did.
#[test]
fn a_table_named_settings_is_not_mistaken_for_a_settings_clause() {
    let s = Scratch::new("settingsname");
    let db = s.at("db");
    pipe(
        &db,
        "CREATE TABLE \"settings\" (id UInt64) ENGINE = MergeTree ORDER BY id;\n\
         INSERT INTO \"settings\" VALUES (7);\n",
    )
    .ok();
    assert_eq!(sql(Some(&db), "SELECT id FROM default.settings LIMIT 1").ok().one(), "7");
    // The clause itself is untouched: an unsettable name still reports as one.
    assert!(sql(None, "SELECT 1 SETTINGS max_memory_usage = 64000000").ok().code == 0);
}

/// The whole reason these are tables and not dot-commands.
#[test]
fn system_tables_can_be_joined_filtered_and_aggregated() {
    let s = Scratch::new("joins");
    let db = s.at("db");
    let mut script = format!("{DDL};\nCREATE TABLE small (id UInt64) ENGINE = MergeTree ORDER BY id;\n");
    script.push_str(&insert(1, 5_000));
    script.push_str("\nOPTIMIZE TABLE hits;\nINSERT INTO small VALUES (1);\n");
    pipe(&db, &script).ok();

    let r = sql(
        Some(&db),
        "SELECT t.name, count(), sum(p.live_rows) FROM system.tables t \
           INNER JOIN system.parts p ON t.name = p.table \
          WHERE p.state = 'active' GROUP BY t.name ORDER BY t.name",
    )
    .ok();
    assert_eq!(r.rows().len(), 2, "{:?}", r.rows());
    let hits = r.rows().into_iter().find(|x| x[0] == "hits").expect("hits");
    assert_eq!(hits[2], "5000");

    // A subquery over one, and a bare aggregate over another.
    assert_eq!(
        sql(
            Some(&db),
            "SELECT count() FROM system.columns \
              WHERE table IN (SELECT name FROM system.tables WHERE parts > 0)"
        )
        .ok()
        .one(),
        // hits (3) + small (1): the read path flushes, so `small`'s single
        // buffered row is in a part by the time `parts > 0` is evaluated.
        "4"
    );
}

/// An empty virtual table still has to have a schema: a database with no parts
/// must answer `SELECT database FROM system.parts` with zero rows, not with an
/// unknown-column error.
#[test]
fn an_empty_system_table_still_binds_its_columns() {
    let s = Scratch::new("emptysys");
    let db = s.at("db");
    let r = pipe(
        &db,
        "SELECT database, table, name, live_rows FROM system.parts;\n\
         SELECT count() FROM system.parts;\n\
         SELECT count() FROM system.tables;\n\
         SELECT count() FROM system.query_log WHERE error != '';\n",
    )
    .ok();
    assert_eq!(r.rows(), vec![vec!["0"], vec!["0"], vec!["0"]]);
}

/// A real table wins over the virtual one of the same name, so a virtual table
/// can never shadow a user's data.
#[test]
fn a_real_table_named_like_a_system_one_is_not_shadowed() {
    let s = Scratch::new("shadow");
    let db = s.at("db");
    pipe(
        &db,
        "CREATE DATABASE system;\n\
         CREATE TABLE system.parts (mine UInt64) ENGINE = MergeTree ORDER BY mine;\n\
         INSERT INTO system.parts VALUES (42);\n\
         SELECT mine FROM system.parts;\n",
    )
    .ok()
    .rows()
    .iter()
    .for_each(|r| assert_eq!(r[0], "42"));
    // The other virtual tables in that namespace are untouched.
    assert!(sql(Some(&db), "SELECT count() FROM system.tables").ok().one().parse::<u64>().unwrap() > 0);
}

/// `system.*` reads run on the `&self` path too, which is where a `Reader`
/// lives: a thread pool must be able to introspect without the writer lock.
#[test]
fn a_reader_thread_can_query_the_system_tables() {
    let s = Scratch::new("reader");
    let db = Db::open(s.at("db")).expect("open");
    db.execute(DDL).expect("ddl");
    db.execute(&insert(1, 2_000)).expect("insert");
    db.writer().checkpoint().expect("checkpoint");

    let reader = db.reader();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let r = reader.clone();
            scope.spawn(move || {
                let rs = r.query("SELECT sum(live_rows) FROM system.parts").expect("parts");
                assert_eq!(rs.scalar(), Some(granular::Value::UInt(2_000)));
                r.query("SELECT count() FROM information_schema.columns").expect("columns");
            });
        }
    });
}
