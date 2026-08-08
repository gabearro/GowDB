//! One corrupt granule must not take the instance offline.
//!
//! Before this, a single bad block anywhere under the data directory made the
//! whole database unopenable: `store::load_catalog` propagated the first part
//! that failed its checksum, `Session::open` never returned, and so `SELECT`
//! on an unrelated table, `SHOW TABLES`, `SHOW DATABASES` and `DROP TABLE` all
//! failed together. The damage was in one file; the outage was total.
//!
//! Everything here runs the real binary over a real data directory with real
//! bytes overwritten in a real part file, because the defect was never in the
//! decoder -- it detected the damage perfectly -- it was in what the layers
//! above it did with the error. A unit test on the reader cannot see that.
//!
//! The three properties, in the order they matter:
//!
//!   1. **The instance opens.** The healthy table answers, the catalog is
//!      fully usable, and the damage is confined to the table that owns it.
//!   2. **The damaged table refuses, and names the file.** It does not answer
//!      short. Returning the rows that did decode would be worse than any
//!      error: `SELECT count(*)` would hand back a plausible number that is
//!      missing however many rows lived in the bad file, with nothing in the
//!      result to say so. Every read *and* every write is refused -- an
//!      `ALTER TABLE ... ADD COLUMN` rebuilds a table from what it can scan,
//!      which would quietly make the loss permanent.
//!   3. **Nothing is rewritten under the damage.** The part file the reader
//!      refused is still there, byte for byte, after any number of sessions.
//!      It is the only copy of those rows, and a restore from backup is the
//!      operator's way out.
//!
//! The damage is applied at four offsets -- the file header, a granule body,
//! the footer, and the footer's own checksum -- because they fail in four
//! different places in the decoder and the quarantine has to be the same
//! whichever one fires.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use granular::persist::format::{FOOTER_LEN, HEADER_LEN};

const BIN: &str = env!("CARGO_BIN_EXE_granular");

/// Rows per table. Above `GRANULE_SIZE` so a part has several granules and
/// "corrupt a granule body" hits one that is not the first.
const ROWS: u64 = 2_500;

// ------------------------------------------------------------------ harness

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let d = std::env::temp_dir().join(format!("granular-degraded-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("scratch dir");
        Scratch(d)
    }
    fn db(&self) -> PathBuf {
        self.0.join("db")
    }
    fn tdir(&self, table: &str) -> PathBuf {
        self.db().join("default").join(table)
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

fn run(db: &Path, sql: &str) -> Run {
    let o = Command::new(BIN)
        .args(["--data", db.to_str().expect("utf-8 path"), "-q", sql])
        .args(["--format", "tsv", "--no-header"])
        .output()
        .expect("spawn granular");
    Run {
        code: o.status.code().unwrap_or(-1),
        out: String::from_utf8_lossy(&o.stdout).into_owned(),
        err: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

fn script(db: &Path, sql: &str) -> Run {
    let f = db.with_extension("sql");
    std::fs::write(&f, sql).expect("write script");
    let o = Command::new(BIN)
        .args(["--data", db.to_str().expect("utf-8 path")])
        .args(["-f", f.to_str().expect("utf-8 path")])
        .args(["--format", "tsv", "--no-header"])
        .output()
        .expect("spawn granular");
    Run {
        code: o.status.code().unwrap_or(-1),
        out: String::from_utf8_lossy(&o.stdout).into_owned(),
        err: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

/// A statement that succeeded, allowing for the one complaint a quarantined
/// database is entitled to make on the way out.
///
/// A checkpoint refuses to rewrite a quarantined table -- that refusal is what
/// keeps the unreadable file on disk -- and the shell checkpoints on exit and
/// reports the failure in its exit status. So a *query* that worked can still
/// leave a non-zero status while damage is present, and this asserts the part
/// that is actually about the query: it produced its answer, and nothing was
/// wrong with it but the quarantine.
///
/// The tolerance is temporary. Once `save_catalog` skips a quarantined table
/// by name instead of refusing at it (`Catalog::quarantined_def` is there for
/// exactly that), a session over a damaged database exits 0 like any other and
/// this can tighten to `assert_eq!(r.code, 0)`.
fn worked(r: &Run, what: &str) {
    assert!(
        r.code == 0 || r.err.contains("quarantined"),
        "{what} failed for a reason other than the quarantine: [{}] {}",
        r.code,
        r.err
    );
}

/// A statement that must have been refused, with the refusal naming `part`.
fn refused(r: &Run, part: &str, what: &str) {
    assert_ne!(r.code, 0, "{what} must not succeed on a damaged table: {}", r.out);
    assert!(
        r.err.contains("quarantined") && r.err.contains(part),
        "{what} must be refused by name and by file: {}",
        r.err
    );
    assert!(
        !r.out.chars().any(|c| c.is_ascii_digit()),
        "{what} printed rows as well as failing -- a short answer is the one \
         outcome worse than an error: {}",
        r.out
    );
}

// ------------------------------------------------------------------ fixture

fn ddl(name: &str) -> String {
    format!(
        "CREATE TABLE {name} (id UInt64, host String, ms Int64) \
         ENGINE = MergeTree ORDER BY id PRIMARY KEY id;\n"
    )
}

/// `n` rows as one `INSERT`, which is one part.
fn rows(name: &str, lo: u64, n: u64) -> String {
    let mut s = format!("INSERT INTO {name} VALUES ");
    for i in lo..lo + n {
        if i > lo {
            s.push(',');
        }
        s.push_str(&format!("({i},'h{}',{})", i % 97, i * 3));
    }
    s.push_str(";\n");
    s
}

/// A database with two tables of [`ROWS`] rows each. `hits` is the one that
/// gets damaged; `events` is the bystander that has to keep working.
///
/// `hits` is loaded in two sessions, which is what gives it two part files:
/// with only one, "does not answer short" would be the weak claim that a table
/// whose every part is unreadable returns nothing. With two, half the rows are
/// sitting decoded in memory, and the engine has to refuse to hand them over.
fn build(s: &Scratch) {
    let mut sql = String::new();
    sql.push_str(&ddl("hits"));
    sql.push_str(&ddl("events"));
    sql.push_str(&rows("hits", 0, ROWS / 2));
    sql.push_str(&rows("events", 1_000_000, ROWS));
    let r = script(&s.db(), &sql);
    assert_eq!(r.code, 0, "fixture: {}", r.err);

    let r = script(&s.db(), &rows("hits", ROWS / 2, ROWS / 2));
    assert_eq!(r.code, 0, "fixture: {}", r.err);
    assert_eq!(
        part_files(&s.tdir("hits")).len(),
        2,
        "the fixture must leave two parts, or the short-answer tests prove nothing"
    );
}

fn part_files(tdir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(tdir)
        .expect("table directory")
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("part_") && n.ends_with(".gpart"))
        .collect();
    v.sort();
    v
}

/// Where in a part file to write, and what each offset breaks.
#[derive(Clone, Copy)]
enum At {
    /// The format version in the file header: rejected before anything is
    /// decoded at all.
    Header,
    /// Payload inside a granule frame: caught by that frame's checksum.
    GranuleBody,
    /// The footer's pointer to the metadata section.
    Footer,
    /// The footer's own checksum, so the footer disagrees with itself.
    Checksum,
}

impl At {
    fn name(self) -> &'static str {
        match self {
            At::Header => "header",
            At::GranuleBody => "granule body",
            At::Footer => "footer",
            At::Checksum => "footer checksum",
        }
    }
    fn offset(self, len: usize) -> usize {
        match self {
            At::Header => 8,                  // the version word after MAGIC
            At::GranuleBody => HEADER_LEN + 64, // inside the first granule frame
            At::Footer => len - FOOTER_LEN,   // the metadata offset
            At::Checksum => len - FOOTER_LEN + 12,
        }
    }
}

/// Flip a bit in the last part file of `hits` and hand back its name.
///
/// The *last*, so the first part -- half the table -- is still perfectly
/// readable and sitting in memory when the query arrives.
fn damage(s: &Scratch, at: At) -> String {
    let tdir = s.tdir("hits");
    let files = part_files(&tdir);
    assert!(!files.is_empty(), "the fixture wrote no parts");
    let name = files.last().expect("a part file").clone();
    let p = tdir.join(&name);
    let mut bytes = std::fs::read(&p).expect("read part");
    let off = at.offset(bytes.len());
    assert!(off < bytes.len(), "{} is past the end of a {}-byte part", at.name(), bytes.len());
    bytes[off] ^= 0x40;
    std::fs::write(&p, &bytes).expect("write part");
    name
}

/// Every file under `root`, by path, as (length, contents hash).
fn tree(root: &Path) -> BTreeMap<String, (u64, u64)> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, (u64, u64)>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(root, &p, out);
            } else if let Ok(b) = std::fs::read(&p) {
                // FNV-1a: this file has no business importing the engine's
                // hasher to compare two byte strings.
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                for &x in &b {
                    h = (h ^ x as u64).wrapping_mul(0x100_0000_01b3);
                }
                let rel = p.strip_prefix(root).unwrap_or(&p).to_string_lossy().into_owned();
                out.insert(rel, (b.len() as u64, h));
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

// -------------------------------------------------------------------- tests

/// The headline: the instance opens, and only the damaged table is missing.
#[test]
fn a_corrupt_part_quarantines_its_table_and_nothing_else() {
    for at in [At::Header, At::GranuleBody, At::Footer, At::Checksum] {
        let s = Scratch::new(&format!("one-{}", at.name().replace(' ', "-")));
        build(&s);
        let part = damage(&s, at);
        let db = s.db();

        let healthy = run(&db, "SELECT count(*) FROM events");
        worked(&healthy, at.name());
        assert_eq!(
            healthy.out.trim(),
            ROWS.to_string(),
            "{}: the healthy table must still answer in full",
            at.name()
        );

        let show = run(&db, "SHOW TABLES");
        worked(&show, "SHOW TABLES");
        assert_eq!(
            show.out.split_whitespace().collect::<Vec<_>>(),
            vec!["events", "hits"],
            "{}: SHOW TABLES must list both, damaged included",
            at.name()
        );

        let dbs = run(&db, "SHOW DATABASES");
        worked(&dbs, "SHOW DATABASES");
        assert!(dbs.out.contains("default"), "{}: {}", at.name(), dbs.out);

        let desc = run(&db, "DESCRIBE events");
        worked(&desc, "DESCRIBE");
        assert!(desc.out.contains("host"), "{}: {}", at.name(), desc.out);

        refused(&run(&db, "SELECT count(*) FROM hits"), &part, &format!("{}: count", at.name()));
        refused(&run(&db, "SELECT * FROM hits LIMIT 5"), &part, &format!("{}: scan", at.name()));
    }
}

/// The trap this exists to avoid. Every one of these has a plausible short
/// answer available -- `count(*)` is answered from part metadata without
/// reading a row, and the scan could just skip the granule -- and every one of
/// them must refuse instead.
#[test]
fn a_damaged_table_never_answers_short() {
    let s = Scratch::new("never-short");
    build(&s);
    let part = damage(&s, At::GranuleBody);
    let db = s.db();

    for sql in [
        "SELECT count(*) FROM hits",
        "SELECT count() FROM hits",
        "SELECT sum(ms) FROM hits",
        "SELECT max(id) FROM hits",
        "SELECT * FROM hits",
        "SELECT * FROM hits WHERE id = 7",
        "SELECT * FROM hits ORDER BY id LIMIT 1",
        "SELECT host, count(*) FROM hits GROUP BY host",
        "SELECT count(*) FROM hits JOIN events USING (id)",
        "SELECT count(*) FROM events WHERE id IN (SELECT id FROM hits)",
    ] {
        refused(&run(&db, sql), &part, sql);
    }
}

/// Writes too, and for a sharper reason than reads: a rebuild reads what it
/// can and writes the result back, so `ALTER TABLE ... ADD COLUMN` on a
/// quarantined table is how a recoverable bad block becomes permanent loss.
#[test]
fn every_write_to_a_damaged_table_is_refused() {
    let s = Scratch::new("no-writes");
    build(&s);
    let part = damage(&s, At::Checksum);
    let db = s.db();

    for sql in [
        "INSERT INTO hits VALUES (999999, 'z', 1)",
        "INSERT INTO hits SELECT id, host, ms FROM events",
        "ALTER TABLE hits DELETE WHERE id < 10",
        "ALTER TABLE hits UPDATE ms = 0 WHERE id < 10",
        "ALTER TABLE hits ADD COLUMN extra UInt64",
        "ALTER TABLE hits DROP COLUMN ms",
        "OPTIMIZE TABLE hits",
        "OPTIMIZE TABLE hits FINAL",
        "SYSTEM FLUSH hits",
    ] {
        refused(&run(&db, sql), &part, sql);
    }
}

/// The bystander is not degraded, only spared: it still takes writes, and they
/// are still durable across a restart.
#[test]
fn the_healthy_table_still_takes_writes() {
    let s = Scratch::new("healthy-writes");
    build(&s);
    damage(&s, At::GranuleBody);
    let db = s.db();

    worked(&run(&db, "INSERT INTO events VALUES (5, 'new', 1), (6, 'new', 2)"), "INSERT");
    let r = run(&db, "SELECT count(*) FROM events");
    worked(&r, "count after INSERT");
    assert_eq!(r.out.trim(), (ROWS + 2).to_string(), "the write must survive the restart");

    let deleted = run(&db, "ALTER TABLE events DELETE WHERE id = 5");
    worked(&deleted, "DELETE");
    let r = run(&db, "SELECT count(*) FROM events");
    worked(&r, "count after DELETE");
    assert_eq!(r.out.trim(), (ROWS + 1).to_string());
}

/// The way out when the file is not coming back. `DROP` is refused by nothing,
/// it clears the quarantine, and what it leaves behind is an ordinary healthy
/// database -- exit status included.
#[test]
fn dropping_the_damaged_table_repairs_the_instance() {
    let s = Scratch::new("drop-repair");
    build(&s);
    damage(&s, At::Footer);
    let db = s.db();

    let dropped = run(&db, "DROP TABLE hits");
    assert_eq!(dropped.code, 0, "DROP must clear the quarantine outright: {}", dropped.err);
    assert!(!s.tdir("hits").exists(), "the dropped table's files must be collected");

    let r = run(&db, "SHOW TABLES");
    assert_eq!(r.code, 0, "a repaired database must exit clean: {}", r.err);
    assert_eq!(r.out.split_whitespace().collect::<Vec<_>>(), vec!["events"]);
    let r = run(&db, "SELECT count(*) FROM events");
    assert_eq!(r.code, 0, "{}", r.err);
    assert_eq!(r.out.trim(), ROWS.to_string(), "the surviving table must be whole");

    // And the name is free again, with nothing of the old table inherited.
    let r = run(&db, "CREATE TABLE hits (id UInt64) ENGINE = MergeTree ORDER BY id");
    assert_eq!(r.code, 0, "{}", r.err);
    let r = run(&db, "SELECT count(*) FROM hits");
    assert_eq!(r.code, 0, "{}", r.err);
    assert_eq!(r.out.trim(), "0");
}

/// The bytes that could not be read are the only copy of those rows. Nothing
/// -- not a query, not a write to another table, not the checkpoint every
/// session takes on the way out -- may rewrite or collect that file.
#[test]
fn the_damaged_bytes_are_never_rewritten() {
    let s = Scratch::new("bytes-survive");
    build(&s);
    let part = damage(&s, At::GranuleBody);
    let db = s.db();
    let hits = s.tdir("hits");
    let before = tree(&hits);
    let damaged_bytes = std::fs::read(hits.join(&part)).expect("read damaged part");

    for sql in [
        "SELECT count(*) FROM events",
        "INSERT INTO events VALUES (77, 'x', 1)",
        "SELECT count(*) FROM hits",
        "SHOW TABLES",
        "OPTIMIZE TABLE events FINAL",
    ] {
        let _ = run(&db, sql);
        assert_eq!(
            std::fs::read(hits.join(&part)).ok().as_deref(),
            Some(&damaged_bytes[..]),
            "`{sql}` rewrote the damaged part"
        );
    }
    assert_eq!(tree(&hits), before, "the quarantined table's directory must be untouched");
}

/// The library path, not the shell: a read-only session over a damaged
/// directory is the shape an operator reaches for while deciding what to
/// restore, and it has to behave the same -- and, because it never
/// checkpoints, it does so with nothing to complain about at all.
#[test]
fn a_read_only_session_opens_and_quarantines_the_same_way() {
    let s = Scratch::new("read-only");
    build(&s);
    let part = damage(&s, At::GranuleBody);

    let mut sess = granular::Session::open_read_only(s.db()).expect("a damaged directory must open");
    let rs = sess.query("SELECT count(*) FROM events").expect("the healthy table must answer");
    assert!(rs.to_string().contains(&ROWS.to_string()), "{rs}");

    let e = sess.query("SELECT count(*) FROM hits").expect_err("the damaged one must not");
    assert_eq!(e.code(), "CHECKSUM_MISMATCH", "{e}");
    assert!(e.to_string().contains(&part), "must name the part file: {e}");
    assert!(e.to_string().contains("default.hits"), "must name the table: {e}");

    // The whole point of holding the damage in the catalog rather than in the
    // reader: it is a list, and something can show it to an operator.
    let dmg = sess.catalog.damaged_parts();
    assert_eq!(dmg.len(), 1);
    assert_eq!(dmg[0].0, "default.hits");
    assert_eq!(dmg[0].1.file, part);
    assert!(dmg[0].1.why.contains(&part), "{}", dmg[0].1.why);
    assert!(sess.catalog.is_quarantined("default.hits"));
    assert!(!sess.catalog.is_quarantined("default.events"));
}

/// Quarantine is per table, and a second database proves it is not per
/// process: the damage is in `default.hits` and nothing outside it notices.
#[test]
fn damage_in_one_database_leaves_the_others_alone() {
    let s = Scratch::new("other-db");
    build(&s);
    let db = s.db();
    let r = script(
        &db,
        "CREATE DATABASE analytics;\n\
         CREATE TABLE analytics.wide (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id;\n\
         INSERT INTO analytics.wide VALUES (1, 10), (2, 20), (3, 30);\n",
    );
    assert_eq!(r.code, 0, "fixture: {}", r.err);
    let part = damage(&s, At::Header);

    let r = run(&db, "SELECT sum(v) FROM analytics.wide");
    worked(&r, "another database");
    assert_eq!(r.out.trim(), "60");

    let r = run(&db, "SHOW TABLES FROM analytics");
    worked(&r, "SHOW TABLES FROM");
    assert_eq!(r.out.trim(), "wide");

    refused(&run(&db, "SELECT count(*) FROM default.hits"), &part, "qualified read");
}
