//! On-disk identity: whose directory is this, and who owns that name?
//!
//! Every failure this file pins had the same shape -- the rows carried no
//! provenance, so nothing could be reconciled after the fact. One tenant's
//! table served under another tenant's name, a `RENAME` that lost both sides,
//! a database that could never be opened again, and a point-in-time recovery
//! that merged two unrelated databases and exited 0. All four were confirmed
//! on this machine before anything here was written.
//!
//! Driven through the shipped binary, in a child process, against a real data
//! directory, and read back by a *second* child process -- because the defect
//! this project keeps repeating is capability that lands in `src/` and never
//! reaches a user. A refusal that only the library knows about is not a
//! refusal.
//!
//! What is pinned, in order:
//!
//!   1. **A name that differs only by case is refused**, for a database, for a
//!      table, and for a `RENAME` target -- and the rows that were already
//!      there are still readable afterwards. On a case-insensitive filesystem
//!      (APFS's default, and every Windows one) the two names are one
//!      directory, and the second `CREATE` adopted the first one's files.
//!   2. **The data directory's own file names are not available** as database
//!      names. `CREATE DATABASE "catalog"` on a fresh directory used to win
//!      the race against the `CATALOG` file and leave the database permanently
//!      unopenable.
//!   3. **An archive knows which database it came from.** `RESTORE ... UNTIL`
//!      pointed at the wrong `--data` directory is refused instead of rolling
//!      one tenant's log forward onto another's parts.
//!   4. **A torn roll-forward publishes nothing.** The root `CATALOG` is the
//!      commit point, and it is written after the last log record rather than
//!      before the first -- so a failure part-way leaves a directory the
//!      loader refuses rather than one that opens with half a transaction.
//!   5. **A filesystem with no hard links still works.** exFAT has none, and
//!      the first `INSERT` used to poison the directory so thoroughly that
//!      every later statement, `SELECT` included, exited 1 forever.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};

const BIN: &str = env!("CARGO_BIN_EXE_granular");
const DDL: &str = "ENGINE = MergeTree ORDER BY id";

// ------------------------------------------------------------------ harness

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "granular-ident-{}-{}-{tag}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
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
    fn ok(self) -> Run {
        assert_eq!(self.code, 0, "expected success\nstdout:\n{}\nstderr:\n{}", self.out, self.err);
        self
    }
    /// Refused, *and* it said why. A non-zero exit with an empty reason is the
    /// same defect wearing a better exit code.
    fn refused(self, needle: &str) -> Run {
        assert_ne!(self.code, 0, "the statement was accepted:\n{}", self.out);
        assert!(
            self.err.contains(needle),
            "expected a refusal mentioning `{needle}`, got:\n{}",
            self.err
        );
        self
    }
    /// Every line of a `--format tsv --no-header` result.
    fn lines(&self) -> Vec<String> {
        self.out.lines().filter(|l| !l.is_empty()).map(str::to_string).collect()
    }
}

/// One CLI invocation in its own process -- which is also one checkpoint, and
/// therefore one archived WAL segment.
fn sql(dir: &Path, q: &str) -> Run {
    let o: Output = Command::new(BIN)
        .args(["--data", &dir.to_string_lossy(), "--format", "tsv", "--no-header", "-q", q])
        .stdin(Stdio::null())
        .output()
        .expect("spawn granular");
    Run {
        code: o.status.code().unwrap_or(-1),
        out: String::from_utf8_lossy(&o.stdout).into_owned(),
        err: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

fn step(dir: &Path, q: &str) -> Run {
    sql(dir, q).ok()
}

// -------------------------------------------------------------- case folding

/// The confirmed data loss, end to end: two `CREATE DATABASE`s differing only
/// by case became one directory, and the second tenant's table replaced the
/// first tenant's rows while both names kept answering.
#[test]
fn a_database_that_differs_only_by_case_is_refused_and_the_first_tenant_survives() {
    let s = Scratch::new("dbfold");
    let db = s.at("data");
    step(&db, "CREATE DATABASE Tenant");
    step(&db, &format!("CREATE TABLE Tenant.t (id Int64) {DDL}"));
    step(&db, "INSERT INTO Tenant.t VALUES (111)");

    sql(&db, "CREATE DATABASE tenant").refused("already owns that directory entry");
    // `IF NOT EXISTS` is not a way around it: it is a different database, and
    // creating it is what destroys the first one.
    sql(&db, "CREATE DATABASE IF NOT EXISTS tenant").refused("already owns that directory entry");
    // ...while the exact name still means "the one that is already there".
    step(&db, "CREATE DATABASE IF NOT EXISTS Tenant");

    assert_eq!(step(&db, "SELECT id FROM Tenant.t").lines(), ["111"], "the tenant's row survives");
    assert_eq!(step(&db, "SHOW DATABASES").lines(), ["Tenant", "default"]);
    // One directory on disk, and it is the one the catalog names.
    let names: Vec<String> = std::fs::read_dir(&db)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| !n.starts_with('.') && n != "CATALOG" && n != "LOCK")
        .collect();
    assert_eq!(names.len(), 2, "default and Tenant, nothing else: {names:?}");
}

/// The same fold one level down, on both statements that create a table
/// directory. `RENAME` was the worse of the two: it destroyed the rows on
/// *both* sides and dropped one name from the catalog, exit 0 throughout.
#[test]
fn a_table_that_differs_only_by_case_is_refused_by_create_and_by_rename() {
    let s = Scratch::new("tblfold");
    let db = s.at("data");
    step(&db, &format!("CREATE TABLE alpha (id Int64) {DDL}"));
    step(&db, "INSERT INTO alpha VALUES (1)");
    step(&db, &format!("CREATE TABLE beta (id Int64) {DDL}"));
    step(&db, "INSERT INTO beta VALUES (2)");

    sql(&db, &format!("CREATE TABLE ALPHA (id Int64) {DDL}"))
        .refused("already owns that directory entry");
    sql(&db, "RENAME TABLE beta TO ALPHA").refused("already owns that directory entry");
    // A pure case change of a table's own name has no safe spelling here: the
    // link and the drop resolve to one directory.
    sql(&db, "RENAME TABLE alpha TO ALPHA").refused("already owns that directory entry");
    // A rename that is not a fold is untouched.
    step(&db, "RENAME TABLE beta TO gamma");

    assert_eq!(
        step(&db, "SELECT 'alpha', id FROM alpha UNION ALL SELECT 'gamma', id FROM gamma")
            .lines()
            .len(),
        2,
        "both tables kept their rows"
    );
    assert_eq!(step(&db, "SELECT id FROM alpha").lines(), ["1"]);
    assert_eq!(step(&db, "SELECT id FROM gamma").lines(), ["2"]);
}

/// `CREATE DATABASE "catalog"` on a *fresh* directory used to brick it: the
/// directory won the race against the `CATALOG` file that had not been written
/// yet, and every later open failed with `Is a directory`. Permanently.
#[test]
fn a_database_named_for_the_data_directorys_own_files_is_refused() {
    for name in ["catalog", "CATALOG", "lock", "LOCK"] {
        let s = Scratch::new(&format!("res-{name}"));
        let db = s.at("data");
        // A brand-new directory, which is the case that used to be fatal.
        sql(&db, &format!("CREATE DATABASE \"{name}\"")).refused("keeps its own");
        // ...and it is still a healthy database afterwards.
        assert_eq!(step(&db, "SHOW DATABASES").lines(), ["default"], "{name}");
        step(&db, &format!("CREATE TABLE t (id Int64) {DDL}"));
        step(&db, "INSERT INTO t VALUES (7)");
        assert_eq!(step(&db, "SELECT id FROM t").lines(), ["7"], "{name}");
    }
}

// ------------------------------------------------------- archive provenance

/// Two unrelated databases, each with a table of the same name. Rolling A's
/// archive forward against B's log used to succeed and hand back a merge of
/// the two, because the only thing tying an archive to a directory was a
/// counter that starts at 1 everywhere.
#[test]
fn an_archive_cannot_be_rolled_forward_against_a_foreign_data_directory() {
    let s = Scratch::new("foreign");
    let (a, b) = (s.at("dbA"), s.at("dbB"));
    for (db, rows) in [(&a, ["(1,'A-one')", "(2,'A-two')"]), (&b, ["(99,'B')", "(98,'B')"])] {
        step(db, &format!("CREATE TABLE acme (id Int64, s String) {DDL}"));
        step(db, &format!("INSERT INTO acme VALUES {}", rows[0]));
        if std::ptr::eq(db, &a) {
            step(db, &format!("BACKUP TO '{}'", s.s("A.gbak")));
        }
        step(db, &format!("INSERT INTO acme VALUES {}", rows[1]));
    }

    sql(&b, &format!("RESTORE FROM '{}' TO '{}' UNTIL LATEST", s.s("A.gbak"), s.s("wrong")))
        .refused("was taken from a different database");
    assert!(!s.at("wrong").join("CATALOG").exists(), "nothing was published");

    // The same statement against the directory the archive *did* come from
    // still rolls forward, so the check refuses foreignness and not recovery.
    step(&a, &format!("RESTORE FROM '{}' TO '{}' UNTIL LATEST", s.s("A.gbak"), s.s("right")));
    assert_eq!(step(&s.at("right"), "SELECT id FROM acme ORDER BY id").lines(), ["1", "2"]);
}

/// The identity is minted once and then never changes: reopening, writing and
/// checkpointing must not renumber the directory, or every archive taken
/// before the restart would become foreign to it.
#[test]
fn a_data_directorys_identity_survives_reopening_it() {
    let s = Scratch::new("stable");
    let db = s.at("data");
    step(&db, &format!("CREATE TABLE t (id Int64) {DDL}"));
    step(&db, "INSERT INTO t VALUES (1)");
    let arc = s.s("early.gbak");
    step(&db, &format!("BACKUP TO '{arc}'"));
    // Four more processes, each of which opens, writes and checkpoints.
    for i in 2..=5 {
        step(&db, &format!("INSERT INTO t VALUES ({i})"));
    }
    step(&db, &format!("RESTORE FROM '{arc}' TO '{}' UNTIL LATEST", s.s("out")));
    assert_eq!(
        step(&s.at("out"), "SELECT id FROM t ORDER BY id").lines(),
        ["1", "2", "3", "4", "5"],
        "an archive from five checkpoints ago is still this directory's own"
    );
}

// ------------------------------------------------------------ torn recovery

/// A roll-forward that fails part-way must publish nothing.
///
/// The tear is induced honestly: one table's archived segment is corrupted, so
/// `RESTORE ... UNTIL` writes the first table's recovered log and then fails on
/// the second. Before the fix the root `CATALOG` had already been committed --
/// the directory opened clean, with one table at the recovery target and the
/// rest at the backup instant, which is half of any transaction that touched
/// both. Now the commit point is last and the loader's existing "table data
/// but no CATALOG" refusal catches it.
#[test]
fn a_torn_roll_forward_leaves_a_directory_the_loader_refuses() {
    let s = Scratch::new("torn");
    let db = s.at("data");
    for t in ["t1", "t2", "t3"] {
        step(&db, &format!("CREATE TABLE {t} (id Int64) {DDL}"));
        step(&db, &format!("INSERT INTO {t} VALUES (1)"));
    }
    let arc = s.s("base.gbak");
    step(&db, &format!("BACKUP TO '{arc}'"));
    for t in ["t1", "t2", "t3"] {
        step(&db, &format!("INSERT INTO {t} VALUES (2)"));
    }

    // Damage the *body* of t2's oldest archived segment: the chain still
    // meets, so the archive looks whole from the directory listing that
    // `check_target` walks, and the failure happens inside the roll-forward
    // loop, which is the window this test is about.
    let seg = db.join(".wal").join("default").join("t2");
    let mut segs: Vec<PathBuf> = std::fs::read_dir(&seg)
        .expect("t2 has an archive")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "gwal"))
        .collect();
    segs.sort();
    // Every segment but the newest is archived; damage them all, so whichever
    // one the recovery reaches fails.
    assert!(segs.len() > 1, "no sealed segment to damage");
    segs.pop();
    for p in &segs {
        let mut bytes = std::fs::read(p).unwrap();
        // The last record's body, not the header: the segment's length and
        // its chain position are untouched, so `check_target`'s walk still
        // sees a whole archive and the failure lands inside the roll-forward.
        let n = bytes.len();
        bytes[n - 8..].fill(0xA5);
        std::fs::write(p, &bytes).unwrap();
    }

    let out = s.at("out");
    let r = sql(&db, &format!("RESTORE FROM '{arc}' TO '{}' UNTIL LATEST", out.to_string_lossy()));
    assert_ne!(r.code, 0, "the damaged segment must fail the recovery:\n{}", r.out);

    // The tear is real -- the first table's work is on disk...
    assert!(out.join("default").join("t1").join("TABLE").exists(), "t1 was unpacked");
    // ...and yet nothing is published, so the loader refuses the directory
    // rather than serving a half-recovered database.
    assert!(!out.join("CATALOG").exists(), "no commit point was written");
    sql(&out, "SELECT count() FROM t1").refused("no CATALOG file");
}

// ----------------------------------------------------- filesystems with no
// ----------------------------------------------------- hard links (exFAT)

/// exFAT has no hard links at all, and this engine links in exactly two
/// places: archiving a sealed WAL segment, and pointing a renamed table's
/// directory at the parts it already has. Both used to fail, and the first
/// `INSERT` then poisoned the directory for good -- a pure `SELECT` exited 1.
///
/// Driven by the same seam an operator can use to rehearse the move, because
/// a real exFAT volume is not mountable everywhere the suite runs. What it
/// exercises is the production path: the probe, the fallback, and the notice.
#[test]
fn a_filesystem_without_hard_links_still_inserts_selects_and_renames() {
    let s = Scratch::new("nolinks");
    let db = s.at("data");
    let go = |q: &str| -> Run {
        let o: Output = Command::new(BIN)
            .args(["--data", &db.to_string_lossy(), "--format", "tsv", "--no-header", "-q", q])
            .env("GRANULAR_NO_HARD_LINKS", "1")
            .stdin(Stdio::null())
            .output()
            .expect("spawn granular");
        Run {
            code: o.status.code().unwrap_or(-1),
            out: String::from_utf8_lossy(&o.stdout).into_owned(),
            err: String::from_utf8_lossy(&o.stderr).into_owned(),
        }
    };

    go(&format!("CREATE TABLE t (id Int64, s String) {DDL}")).ok();
    go("INSERT INTO t VALUES (1,'a')").ok();
    go("INSERT INTO t VALUES (2,'b')").ok();
    // The statement that used to exit 1 on a directory an archive had poisoned.
    assert_eq!(go("SELECT s FROM t ORDER BY id").ok().lines(), ["a", "b"]);
    // The one remaining link site, which only a RENAME reaches -- the WAL no
    // longer has one at all: a segment *is* its own archive file, so nothing
    // is linked or copied to retire it, on any filesystem.
    let ren = go("RENAME TABLE t TO t2").ok();
    assert!(
        ren.err.contains("no hard links"),
        "the operator is told the disk-space story changed, got:\n{}",
        ren.err
    );
    assert_eq!(go("SELECT s FROM t2 ORDER BY id").ok().lines(), ["a", "b"]);

    // Copies, not links: every part file is its own inode. And the directory
    // opens exactly the same from a process that never heard of the seam.
    assert_eq!(step(&db, "SELECT s FROM t2 ORDER BY id").lines(), ["a", "b"]);
    let parts = db.join("default").join("t2");
    for e in std::fs::read_dir(&parts).unwrap().flatten() {
        if e.path().extension().is_some_and(|x| x == "gpart") {
            assert!(e.metadata().unwrap().len() > 0, "a copied part is not empty");
        }
    }

    // The WAL archive was written by copy too, and it is still a usable
    // archive: a point-in-time recovery rolls it forward.
    let arc = s.s("nl.gbak");
    step(&db, &format!("BACKUP TO '{arc}'"));
    step(&db, "INSERT INTO t2 VALUES (3,'c')");
    step(&db, &format!("RESTORE FROM '{arc}' TO '{}' UNTIL LATEST", s.s("out")));
    assert_eq!(step(&s.at("out"), "SELECT s FROM t2 ORDER BY id").lines(), ["a", "b", "c"]);
}
