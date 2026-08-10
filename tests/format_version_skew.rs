//! Version skew, at the binary: what an operator sees when the on-disk format
//! moved out from under their data.
//!
//! v3 replaced the single `wal.log` per table with numbered segments under
//! `<data>/.wal`, and `MIN_READ_VERSION` went up with `FORMAT_VERSION`, so a
//! v2 data directory and a v2 `.gbak` are both refused. That refusal is the
//! entire migration path, which makes its *wording* load-bearing: the files
//! are intact, and an operator told they have corruption will go looking for
//! damage, delete the directory, or restore over a good backup with an older
//! one. Each test below therefore pins three things -- the exit status, the
//! error code, and the absence of the words that would send someone the wrong
//! way.
//!
//! Every fixture is a real file this build wrote with its version word
//! rewritten, which is stronger than a hand-assembled header in two ways: the
//! rest of the file is genuine, so a refusal proves the version is checked
//! *before* any checksum or body; and the rewrite is the only difference, so
//! nothing else can be what refused it. `MAGIC ++ u32le(2)` at offset 0 is
//! byte-for-byte what a v2 build wrote there -- verified against artifacts
//! produced by the previous binary, whose real v2 `CATALOG` and `.gbak` give
//! these same messages.
//!
//! What is pinned:
//!
//!   1. **An older data directory is refused before anything is touched.**
//!      `CATALOG` is the first file read, so no table directory is opened, no
//!      table is filed as quarantined, and nothing is rewritten.
//!   2. **A newer one is refused with the opposite remedy.** Recreate and
//!      upgrade are not interchangeable, and a build that wrote to a directory
//!      it cannot read would destroy it.
//!   3. **Every archive verb refuses an older archive**, and leaves it where
//!      it is -- telling someone to recreate an archive is telling them to
//!      delete their only copy.

use std::path::{Path, PathBuf};
use std::process::Command;

use granular::persist::format;

const BIN: &str = env!("CARGO_BIN_EXE_granular");

// ------------------------------------------------------------------ harness

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let d = std::env::temp_dir().join(format!("granular-skew-{}-{tag}", std::process::id()));
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

fn sql(db: &Path, q: &str) -> Run {
    let o = Command::new(BIN)
        .args(["--data", &db.to_string_lossy(), "--format", "tsv", "--no-header", "-q", q])
        .output()
        .expect("spawn granular");
    Run {
        code: o.status.code().unwrap_or(-1),
        out: String::from_utf8_lossy(&o.stdout).into_owned(),
        err: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

/// The whole claim about a refusal's wording, in one place: it exits non-zero,
/// it is a `FORMAT_VERSION` error rather than a corruption one, it says which
/// version it found and which this build speaks, and it reports nothing on
/// stdout as if it had worked.
///
/// The damage check is about the *claim*, not the vocabulary. These messages
/// use "corrupt" and "damaged" on purpose -- to deny them -- so banning the
/// words would fail the very sentences that do the job. What must never appear
/// is a corruption error code, or a checksum the operator is invited to go
/// hunting for.
fn refused_as_version_skew(r: &Run, found: u32) -> String {
    assert_ne!(r.code, 0, "accepted:\nstdout:\n{}\nstderr:\n{}", r.out, r.err);
    assert!(r.out.trim().is_empty(), "a refusal that also printed a result: {}", r.out);
    let e = r.err.clone();
    assert!(e.contains("[FORMAT_VERSION]"), "not filed as a version skew: {e}");
    assert!(e.contains(&format!("version {found}")), "does not say what it found: {e}");
    assert!(
        e.contains(&format!("version {}", format::FORMAT_VERSION)),
        "does not say what it reads: {e}"
    );
    for code in ["CHECKSUM_MISMATCH", "CORRUPTION", "STORAGE_ERROR", "IO_ERROR"] {
        assert!(!e.contains(code), "a version skew filed as `{code}`: {e}");
    }
    let lower = e.to_lowercase();
    assert!(!lower.contains("checksum"), "nothing here failed a checksum: {e}");
    // Any mention of damage has to be a denial of it. Both sentences that
    // carry one are pinned by name, so a rewrite that turns the denial into an
    // accusation fails here rather than at 3am.
    if lower.contains("damage") || lower.contains("corrupt") {
        assert!(
            e.contains("Nothing here is damaged") || e.contains("intact, not corrupt"),
            "the message mentions damage without denying it: {e}"
        );
    }
    e
}

/// A real data directory with one table in it, and a copy of every file in it
/// taken before anything is pointed at it.
fn seeded(s: &Scratch, name: &str) -> PathBuf {
    let db = s.at(name);
    let r = sql(
        &db,
        "CREATE TABLE t (id UInt64, v String) ENGINE = MergeTree ORDER BY id; \
         INSERT INTO t VALUES (1,'a'),(2,'b')",
    );
    assert_eq!(r.code, 0, "fixture: {}", r.err);
    db
}

/// Rewrite the 4-byte version word that follows `MAGIC` at the head of a file.
/// One `u32`, nothing else, so the file is otherwise exactly what this build
/// wrote -- which is what makes a refusal attributable to the version alone.
fn set_version(path: &Path, v: u32) {
    let mut b = std::fs::read(path).expect("read the fixture");
    assert_eq!(&b[..format::MAGIC.len()], &format::MAGIC, "{} is not ours", path.display());
    b[format::MAGIC.len()..format::HEADER_LEN].copy_from_slice(&v.to_le_bytes());
    std::fs::write(path, &b).expect("write the fixture");
}

/// Every file under `dir`, by relative path, with its bytes -- so "nothing was
/// touched" can be asserted rather than eyeballed.
fn snapshot(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if let Ok(b) = std::fs::read(&p) {
                out.push((p.strip_prefix(dir).expect("under dir").to_path_buf(), b));
            }
        }
    }
    out.sort();
    out
}

// ------------------------------------------------------------- a data directory

/// The headline: a directory this build cannot read is refused with the story
/// an operator needs, and it is refused at `CATALOG` -- the first file the
/// loader opens -- so the tables underneath are never reached.
///
/// The part files matter here beyond "unchanged". `load_catalog` quarantines a
/// table whose own files will not decode, and filing a format change in that
/// list would tell someone to restore a file that is perfectly good. Refusing
/// at the root makes that unreachable rather than merely unlikely.
#[test]
fn an_older_data_directory_is_refused_as_a_version_change_not_as_damage() {
    let s = Scratch::new("older");
    let db = seeded(&s, "db");
    // Everything except `CATALOG`, which this test edits, and `LOCK`, which
    // holds the pid of whichever process last took the directory -- a refused
    // open still takes the lock, which is how it earns the right to look.
    let owned = |d: &Path| -> Vec<(PathBuf, Vec<u8>)> {
        snapshot(d)
            .into_iter()
            .filter(|(p, _)| p != Path::new("CATALOG") && p != Path::new("LOCK"))
            .collect()
    };
    let before = owned(&db);
    assert!(before.iter().any(|(p, _)| p.to_string_lossy().ends_with(".gpart")), "fixture");
    assert!(before.iter().any(|(p, _)| p.to_string_lossy().contains(".wal")), "fixture");

    set_version(&db.join("CATALOG"), format::FORMAT_VERSION - 1);

    let r = sql(&db, "SELECT count() FROM t");
    let e = refused_as_version_skew(&r, format::FORMAT_VERSION - 1);
    assert!(e.contains("CATALOG"), "the refusal must name the file: {e}");
    assert!(e.contains("no migration path"), "{e}");
    assert!(e.contains("must be recreated"), "the remedy is missing: {e}");
    assert!(e.contains("write-ahead log layout changed"), "the reason is missing: {e}");

    assert_eq!(before, owned(&db), "a refused open rewrote the data directory");
    // The `CATALOG` too, byte for byte apart from the word this test changed:
    // a half-open that rewrote the roster is how a directory ends up readable
    // by neither version.
    let cat = std::fs::read(db.join("CATALOG")).expect("CATALOG");
    assert_eq!(
        u32::from_le_bytes(cat[format::MAGIC.len()..format::HEADER_LEN].try_into().unwrap()),
        format::FORMAT_VERSION - 1,
        "a refused open rewrote the version word"
    );

    // ...and it stays refused. A second attempt must not have been "fixed" by
    // the first.
    let again = sql(&db, "SELECT count() FROM t");
    refused_as_version_skew(&again, format::FORMAT_VERSION - 1);
    assert_eq!(before, owned(&db), "a second refused open rewrote the data directory");
}

/// The other direction, which is the dangerous one: this build must not write
/// to a directory a newer build owns. The remedies are opposites -- recreate
/// versus upgrade -- so one message for both would be wrong half the time.
#[test]
fn a_newer_data_directory_says_upgrade_rather_than_recreate() {
    let s = Scratch::new("newer");
    let db = seeded(&s, "db");
    set_version(&db.join("CATALOG"), format::FORMAT_VERSION + 1);
    // `CATALOG` is the byte this test edited and `LOCK` is the one file a
    // refused open is *expected* to write -- taking the lock is how it earns
    // the right to look at all. Everything else must come out identical.
    let owned = |d: &Path| -> Vec<(PathBuf, Vec<u8>)> {
        snapshot(d)
            .into_iter()
            .filter(|(p, _)| p != Path::new("CATALOG") && p != Path::new("LOCK"))
            .collect()
    };
    let before = owned(&db);
    assert!(before.iter().any(|(p, _)| p.to_string_lossy().ends_with(".gpart")), "fixture");

    // A read, a write and a DDL: the last two are the ones that would destroy
    // a directory a newer build owns, so all three have to stop at the door.
    for stmt in ["SELECT count() FROM t", "INSERT INTO t VALUES (3,'c')", "DROP TABLE t"] {
        let r = sql(&db, stmt);
        let e = refused_as_version_skew(&r, format::FORMAT_VERSION + 1);
        assert!(e.contains("Upgrade granular"), "{stmt}: {e}");
        assert!(e.contains("must not write"), "{stmt}: {e}");
        assert!(!e.contains("must be recreated"), "a newer directory must not be recreated: {e}");
        assert_eq!(before, owned(&db), "`{stmt}` wrote into a directory a newer build owns");
    }
}

// ----------------------------------------------------------------- an archive

/// Every verb that reads an archive refuses an older one, with the remedy
/// inverted: **keep** the file and take a fresh backup. `RESTORE`, `VERIFY
/// BACKUP` and `BACKUP ... INCREMENTAL FROM` all reach the same reader, and
/// all three are things someone types during an incident.
#[test]
fn an_older_archive_is_refused_by_every_verb_and_is_left_where_it_is() {
    let s = Scratch::new("archive");
    let db = seeded(&s, "db");
    let arc = s.s("old.gbak");
    assert_eq!(sql(&db, &format!("BACKUP TO '{arc}'")).code, 0, "fixture");
    set_version(Path::new(&arc), format::FORMAT_VERSION - 1);
    let bytes = std::fs::read(&arc).expect("the archive");

    for stmt in [
        format!("RESTORE FROM '{arc}' TO '{}'", s.s("out")),
        format!("RESTORE FROM '{arc}' TO '{}' UNTIL LATEST", s.s("out")),
        format!("RESTORE FROM '{arc}' TO '{}' UNTIL LSN 2", s.s("out")),
        format!("VERIFY BACKUP '{arc}'"),
        format!("BACKUP TO '{}' INCREMENTAL FROM '{arc}'", s.s("inc.gbak")),
    ] {
        let r = sql(&db, &stmt);
        let e = refused_as_version_skew(&r, format::FORMAT_VERSION - 1);
        assert!(e.contains("archive"), "{stmt}: it is an archive, not a database: {e}");
        assert!(e.contains("fresh backup"), "{stmt}: the remedy is missing: {e}");
        assert!(!e.contains("must be recreated"), "{stmt}: never delete the only copy: {e}");
        assert_eq!(std::fs::read(&arc).expect("still there"), bytes, "{stmt} rewrote the archive");
        assert!(!s.at("out").exists(), "{stmt} created the restore target");
        assert!(!s.at("inc.gbak").exists(), "{stmt} wrote an archive it could not chain onto");
    }

    // The live database is untouched by all of it and still answers.
    let r = sql(&db, "SELECT count() FROM t");
    assert_eq!(r.code, 0, "{}", r.err);
    assert_eq!(r.out.trim(), "2");
}

/// A newer archive says upgrade, and **only** upgrade.
///
/// This is the one that was wrong. The archive refusal used to be built by
/// rewriting the data-directory refusal's text, and the rewrite only matched
/// the older direction -- so a newer archive got both stories concatenated:
/// "Upgrade granular rather than downgrading" and then "take a fresh backup
/// with this build". The second is the sentence an operator acts on, and doing
/// it replaces an archive this build cannot read with one it can, losing
/// whatever the newer format held.
#[test]
fn a_newer_archive_says_upgrade_and_nothing_else() {
    let s = Scratch::new("newarc");
    let db = seeded(&s, "db");
    let arc = s.s("new.gbak");
    assert_eq!(sql(&db, &format!("BACKUP TO '{arc}'")).code, 0, "fixture");
    set_version(Path::new(&arc), format::FORMAT_VERSION + 1);

    for stmt in [
        format!("RESTORE FROM '{arc}' TO '{}'", s.s("out")),
        format!("VERIFY BACKUP '{arc}'"),
        format!("BACKUP TO '{}' INCREMENTAL FROM '{arc}'", s.s("inc.gbak")),
    ] {
        let r = sql(&db, &stmt);
        let e = refused_as_version_skew(&r, format::FORMAT_VERSION + 1);
        assert!(e.contains("Upgrade granular"), "{stmt}: {e}");
        assert!(e.contains("archive"), "{stmt}: it is an archive: {e}");
        assert!(
            !e.contains("fresh backup"),
            "a newer archive must never be told to overwrite itself: {e}"
        );
        assert!(!s.at("out").exists(), "{stmt} created the restore target");
        assert!(!s.at("inc.gbak").exists(), "{stmt} wrote an archive it could not chain onto");
    }
}
