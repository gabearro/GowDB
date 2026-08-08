//! The checkpoint only writes what changed -- and is still exactly as safe.
//!
//! Every test here drives the shipped binary, because the defect this file
//! exists for was reachable in precisely one way: running `granular` twice.
//! A part that had not changed since it was written was rewritten anyway,
//! under a fresh sequence number, into a fresh inode. That made a checkpoint
//! O(entire database) rather than O(what changed), so a read-only query
//! against a 100 GB database wrote 100 GB, needed 100 GB free to do it, and
//! wedged the instance permanently on a nearly-full volume. It also meant no
//! file in the tree kept its identity from one run to the next, which is
//! exactly the property a filesystem-snapshot backup needs.
//!
//! ## What is asserted, and why in this shape
//!
//! **Identity, not just content.** The interesting claim is not "the rows are
//! still right" -- the old code got that right while rewriting everything --
//! it is "this is the same file". So these fixtures compare inode numbers,
//! and for the read-only case the bytes as well. An inode is the one thing a
//! rewrite cannot preserve.
//!
//! **Crashes at the structural points, deterministically, and then at swept
//! points for real.** An incremental commit adds a way to lose data that a
//! full rewrite did not have: it *keeps* files, so it now depends on files
//! written by an earlier run still being there and still being named. Two of
//! the three windows are reconstructed byte-for-byte from copies of the tree
//! (a crash after the parts are written but before the commit; a crash after
//! the commit but before the superseded files are unlinked), because racing
//! for them is unreliable while the states themselves are not. The third form
//! is a genuine `kill -9` of a live checkpoint at three polled trigger points,
//! which is the only way to reach the windows nobody thought to enumerate.
//!
//! The oracle for the kills is `stress_crash.rs`'s, with one simplification
//! available here: the workload is plain `INSERT`s, and a plain `INSERT` never
//! writes a part file -- it appends to the log and fsyncs before it
//! acknowledges. So the first part-file write in the run *is* the exit
//! checkpoint, and observing one proves every statement in the script had
//! already been acknowledged. "Exactly the acknowledged writes" is then
//! checkable with no channel back from the child: the recovered ids must be
//! exactly `0..sent`, no holes and no extras.

#![cfg(unix)]

use std::collections::{BTreeMap, BTreeSet};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------- fixtures

const BIN: &str = env!("CARGO_BIN_EXE_granular");
const DDL: &str =
    "CREATE TABLE t (id UInt64, s String, v Int64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id";
/// Rows per bulk load. Above `BULK_INSERT_THRESHOLD`, so each load becomes a
/// part of its own instead of being buffered into the delta with its
/// neighbours -- the fixtures need parts they can name.
const CHUNK: u64 = 5_000;

struct Scratch(PathBuf);

impl Scratch {
    /// `tag` must not contain `begin`, `commit`, `rollback` or `start`: the
    /// tag ends up inside the quoted path of an `INSERT ... FROM INFILE`, and
    /// `Session::run` routes any statement whose *bytes* hold one of those
    /// words to the transaction splitter, which parses with the grammar and
    /// never reaches the importer. That is a real bug, it is not this file's,
    /// and it costs an hour to rediscover from the syntax error it produces.
    fn new(tag: &str) -> Scratch {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "granular-ckpt-{}-{}-{tag}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Scratch(d)
    }
    fn path(&self) -> &Path {
        &self.0
    }
    /// The database root, one level down so the fixture can keep its scripts,
    /// CSV files and tree copies out of it.
    fn db(&self) -> PathBuf {
        self.0.join("db")
    }
    fn tdir(&self) -> PathBuf {
        self.db().join("default").join("t")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Run `sql` against `db` in a fresh process, returning its rows. Panics on a
/// non-zero status: every use is a fixture step, and a failure there would
/// otherwise surface as a baffling assertion three lines later.
fn run_at(db: &Path, sql: &str) -> String {
    let out = Command::new(BIN)
        .args(["--data", db.to_str().unwrap(), "-q", sql])
        .args(["--format", "tsv", "--no-header"])
        .output()
        .expect("spawn granular");
    assert!(out.status.success(), "`{sql}` failed: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn run(s: &Scratch, sql: &str) -> String {
    run_at(&s.db(), sql)
}

/// `count, sum(id), sum(v), max(s)` -- enough that a lost, duplicated or
/// resurrected row cannot hide.
fn fingerprint(db: &Path) -> String {
    run_at(db, "SELECT count(*), sum(id), sum(v), max(s) FROM t")
}

fn csv(s: &Scratch, name: &str, lo: u64, n: u64) -> PathBuf {
    let mut body = String::with_capacity(n as usize * 24);
    body.push_str("id,s,v\n");
    for i in lo..lo + n {
        body.push_str(&format!("{i},r{},{}\n", i % 97, i * 3));
    }
    let p = s.path().join(name);
    std::fs::write(&p, body).unwrap();
    p
}

/// A database of `chunks` parts holding ids `0..chunks*CHUNK`.
fn seed(s: &Scratch, chunks: u64) {
    run(s, DDL);
    let mut sql = String::new();
    for c in 0..chunks {
        let f = csv(s, &format!("c{c}.csv"), c * CHUNK, CHUNK);
        sql.push_str(&format!("INSERT INTO t FROM INFILE '{}';\n", f.display()));
    }
    run(s, &sql);
}

/// Every part file in a table directory: name -> (inode, length).
fn parts_in(tdir: &Path) -> BTreeMap<String, (u64, u64)> {
    let mut out = BTreeMap::new();
    let Ok(rd) = std::fs::read_dir(tdir) else { return out };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with("part_") && name.ends_with(".gpart") {
            let m = e.metadata().unwrap();
            out.insert(name, (m.ino(), m.len()));
        }
    }
    out
}

fn parts(s: &Scratch) -> BTreeMap<String, (u64, u64)> {
    parts_in(&s.tdir())
}

fn ino_of(p: &Path) -> Option<u64> {
    std::fs::metadata(p).ok().map(|m| m.ino())
}

/// The bytes of every part file: the strongest available form of "untouched".
fn part_bytes(s: &Scratch) -> BTreeMap<String, Vec<u8>> {
    parts(s).keys().map(|n| (n.clone(), std::fs::read(s.tdir().join(n)).unwrap())).collect()
}

fn names_in(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .map(|rd| rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default()
}

// ------------------------------------------------------- identity of files

/// The headline: a query that changes nothing must write nothing.
///
/// This is the defect reproduced at the granularity it was found. The part
/// files keep their names, their inodes *and* their bytes across a read-only
/// invocation -- and so does the `TABLE` commit record, which is small but
/// still costs a create, an fsync, a rename and a directory fsync per table
/// per run, and a whole erase block on the SSD underneath.
#[test]
fn a_read_only_query_rewrites_nothing() {
    let s = Scratch::new("readonly");
    seed(&s, 4);
    let before = parts(&s);
    let bytes = part_bytes(&s);
    let table = ino_of(&s.tdir().join("TABLE"));
    assert_eq!(before.len(), 4, "the fixture should be four parts: {before:?}");

    assert_eq!(run(&s, "SELECT 1"), "1");
    assert_eq!(parts(&s), before, "a read-only query renamed or rewrote a part");
    assert_eq!(part_bytes(&s), bytes, "a read-only query changed a part's bytes");
    assert_eq!(ino_of(&s.tdir().join("TABLE")), table, "the commit record was rewritten");

    // ...and a read that actually touches every row is no different.
    assert_eq!(run(&s, "SELECT count(*) FROM t"), "20000");
    assert_eq!(parts(&s), before, "a full scan rewrote a part");
    assert_eq!(part_bytes(&s), bytes, "a full scan changed a part's bytes");
}

/// An insert writes its own part and leaves every other one alone. This is
/// what turns checkpoint cost from O(database) into O(change).
#[test]
fn an_insert_writes_one_part_and_keeps_the_rest() {
    let s = Scratch::new("insert");
    seed(&s, 4);
    let before = parts(&s);

    let f = csv(&s, "extra.csv", 90_000, CHUNK);
    run(&s, &format!("INSERT INTO t FROM INFILE '{}'", f.display()));

    let after = parts(&s);
    assert_eq!(after.len(), 5, "expected exactly one new part: {after:?}");
    for (name, id) in &before {
        assert_eq!(
            after.get(name),
            Some(id),
            "part {name} was rewritten by an insert that did not touch it"
        );
    }
    assert_eq!(run(&s, "SELECT count(*) FROM t"), "25000");
}

/// A delete moves a part's *mask* without touching its rows, and that part --
/// and only that part -- is rewritten.
///
/// A deliberate decision rather than an oversight: the delete bitmap lives
/// inside the part file, so a moved mask invalidates the file. See
/// `PartSet::tombstone`, which drops the file provenance exactly when a
/// tombstone actually lands, and the rejected sidecar-mask alternative
/// recorded there.
#[test]
fn a_delete_rewrites_only_the_part_it_touched() {
    let s = Scratch::new("delete");
    seed(&s, 4);
    let before = parts(&s);

    // An id in the third chunk, so untouched parts sit on both sides of it.
    run(&s, "DELETE FROM t WHERE id = 10007");

    let after = parts(&s);
    assert_eq!(after.len(), 4, "the superseded part was not collected: {after:?}");
    let kept = before.iter().filter(|(n, id)| after.get(*n) == Some(*id)).count();
    assert_eq!(kept, 3, "exactly one part should have been rewritten: {before:?} -> {after:?}");
    assert!(
        after.keys().any(|n| !before.contains_key(n)),
        "the rewritten part must land in a new file: {after:?}"
    );

    assert_eq!(run(&s, "SELECT count(*) FROM t"), "19999");
    // Twice: once against the just-written parts, once after a further reopen
    // and checkpoint, which is where a mask that failed to reach disk would
    // show up.
    assert_eq!(run(&s, "SELECT count(*) FROM t WHERE id = 10007"), "0");
    assert_eq!(run(&s, "SELECT count(*) FROM t WHERE id = 10007"), "0");
}

/// Re-deleting an already-dead row leaves the mask byte-identical, so it must
/// dirty nothing. The narrow case that separates "the mask moved" from "a
/// DELETE ran".
#[test]
fn a_delete_that_hides_nothing_rewrites_nothing() {
    let s = Scratch::new("delete-noop");
    seed(&s, 2);
    run(&s, "DELETE FROM t WHERE id = 5");
    let before = parts(&s);
    run(&s, "DELETE FROM t WHERE id = 5");
    assert_eq!(parts(&s), before, "a delete that hid no row rewrote a part");
    run(&s, "DELETE FROM t WHERE id = 999999");
    assert_eq!(parts(&s), before, "a delete that matched nothing rewrote a part");
}

/// When everything really has changed, everything is written -- and the old
/// files go. The incremental path must not turn into a leak.
#[test]
fn a_merge_replaces_every_part_and_collects_the_old_ones() {
    let s = Scratch::new("optimize");
    seed(&s, 4);
    let before = parts(&s);
    let fp = fingerprint(&s.db());

    run(&s, "OPTIMIZE TABLE t FINAL");

    let after = parts(&s);
    assert_eq!(after.len(), 1, "a full merge should leave one part: {after:?}");
    for name in before.keys() {
        assert!(!after.contains_key(name), "merged-away part {name} is still on disk");
    }
    assert_eq!(fingerprint(&s.db()), fp, "a merge changed the data");
}

/// Twenty rounds of write-then-checkpoint: files must not accumulate, ids must
/// not drift, and a part that was already correct must not be rewritten again
/// and again.
#[test]
fn many_checkpoints_neither_leak_files_nor_drift() {
    let s = Scratch::new("rounds");
    seed(&s, 2);
    let mut rewrites = 0usize;
    let mut prev = parts(&s);
    for r in 0..20u64 {
        let f = csv(&s, &format!("r{r}.csv"), 1_000_000 + r * CHUNK, CHUNK);
        run(&s, &format!("INSERT INTO t FROM INFILE '{}'", f.display()));
        let now = parts(&s);
        rewrites += prev.iter().filter(|(n, id)| now.get(*n) != Some(*id)).count();
        prev = now;
    }
    // The floor is not zero: auto-compaction merges the small parts once there
    // are sixteen of them, and a merged part legitimately replaces its inputs.
    // Measured at 8 -- one compaction round, nothing else -- against 2 + 3 +
    // ... + 21 = 230 when every checkpoint rewrote every part. The bound is
    // loose enough that a change of compaction policy moves it rather than
    // breaking it, and tight enough that a return to full rewrites cannot pass.
    eprintln!("  {rewrites} part rewrites over 20 rounds, {} files left", prev.len());
    assert!(rewrites < 30, "{rewrites} parts were rewritten across 20 incremental checkpoints");
    assert_eq!(run(&s, "SELECT count(*) FROM t"), (22 * CHUNK).to_string());
    assert!(
        parts(&s).len() <= 22,
        "part files accumulated: {:?}",
        parts(&s).keys().collect::<Vec<_>>()
    );
}

// ------------------------------------------------------------ crash safety

/// Copy a directory tree. The only way to freeze a crash state without racing
/// for it.
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap().flatten() {
        let (src, dst) = (e.path(), to.join(e.file_name()));
        if src.is_dir() {
            copy_tree(&src, &dst);
        } else {
            std::fs::copy(&src, &dst).unwrap();
        }
    }
}

/// Reopen `db` in a fresh process and read the ids back.
fn recovered(db: &Path) -> Result<Vec<u64>, String> {
    let out = Command::new(BIN)
        .args(["--data", db.to_str().unwrap()])
        .args(["-q", "SELECT id FROM t ORDER BY id"])
        .args(["--format", "tsv", "--no-header"])
        .output()
        .expect("spawn granular");
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .split_ascii_whitespace()
        .map(|v| v.parse().expect("an integer id"))
        .collect())
}

/// A crash after the new part files are written but before the `TABLE` file
/// names them.
///
/// Reconstructed at the file level: the tree as it was before the round, plus
/// the part files the round produced, with the old commit record left in
/// place. (The *log* in this reconstruction is the old, already-truncated one,
/// so the rows in the orphan are not recoverable from anywhere -- which is the
/// point. What a real crash leaves in the log is the other axis, and the kill
/// trials below cover it.) The rule under test is that the previous commit is
/// the whole truth: a part file nothing names cannot leak a single row into
/// the answer, and the next commit collects it.
#[test]
fn a_crash_before_the_commit_falls_back_to_the_previous_one() {
    let s = Scratch::new("crash-unnamed");
    seed(&s, 3);
    let fp = fingerprint(&s.db());
    let before_tree = s.path().join("before");
    copy_tree(&s.db(), &before_tree);
    let old_files: BTreeSet<String> = parts(&s).into_keys().collect();

    let f = csv(&s, "more.csv", 500_000, CHUNK);
    run(&s, &format!("INSERT INTO t FROM INFILE '{}'", f.display()));
    let orphans: Vec<String> =
        parts(&s).into_keys().filter(|n| !old_files.contains(n)).collect();
    assert_eq!(orphans.len(), 1, "the fixture produced {orphans:?}");

    let crashed = s.path().join("crashed");
    let ctdir = crashed.join("default/t");
    copy_tree(&before_tree, &crashed);
    std::fs::copy(s.tdir().join(&orphans[0]), ctdir.join(&orphans[0])).unwrap();
    let staged = parts_in(&ctdir);
    assert_eq!(staged.len(), old_files.len() + 1, "the fixture built no orphan");

    let ids = recovered(&crashed).expect("the previous commit must still open");
    assert_eq!(ids.len(), 3 * CHUNK as usize, "the previous commit was not intact");
    assert!(ids.iter().enumerate().all(|(i, &id)| id == i as u64), "recovered ids have a hole");
    assert_eq!(run_at(&crashed, "SELECT count(*), sum(id), sum(v), max(s) FROM t"), fp);

    // That reopen checkpointed on the way out: the orphan is gone and every
    // committed part is still the same file.
    let after = parts_in(&ctdir);
    assert!(!after.contains_key(&orphans[0]), "the orphan was never collected: {after:?}");
    for name in &old_files {
        assert_eq!(
            after.get(name),
            staged.get(name),
            "collecting the orphan rewrote committed part {name}"
        );
    }
}

/// A crash after the commit but before the superseded files are unlinked.
///
/// This is the window the incremental path widened: unlinking used to be
/// "every file that was here when I started", and is now a decision about
/// which files the new commit does *not* name. A file wrongly kept must still
/// be invisible, and a file wrongly unlinked would be unrecoverable -- so the
/// fixture puts the superseded file back and insists the committed answer is
/// unchanged, deleted row included.
#[test]
fn a_crash_before_the_collection_keeps_the_new_commit() {
    let s = Scratch::new("crash-unswept");
    seed(&s, 3);
    let before_tree = s.path().join("before");
    copy_tree(&s.db(), &before_tree);
    let old_files = parts(&s);

    // A delete supersedes exactly one part: this round both keeps files and
    // drops one, which is the mixed case.
    run(&s, "DELETE FROM t WHERE id = 7001");
    let fp = fingerprint(&s.db());

    let crashed = s.path().join("crashed");
    let ctdir = crashed.join("default/t");
    copy_tree(&s.db(), &crashed);
    let now: BTreeSet<String> = parts(&s).into_keys().collect();
    let superseded: Vec<&String> = old_files.keys().filter(|n| !now.contains(*n)).collect();
    assert_eq!(superseded.len(), 1, "the fixture superseded {} parts", superseded.len());
    let stale = superseded[0].clone();
    std::fs::copy(before_tree.join("default/t").join(&stale), ctdir.join(&stale)).unwrap();
    let staged = parts_in(&ctdir);
    assert_eq!(staged.len(), now.len() + 1, "the fixture put no stale file back");

    let ids = recovered(&crashed).expect("the new commit must open");
    assert_eq!(ids.len(), 3 * CHUNK as usize - 1, "the committed delete did not survive");
    assert!(!ids.contains(&7001), "the stale file resurrected a deleted row");
    assert_eq!(
        run_at(&crashed, "SELECT count(*), sum(id), sum(v), max(s) FROM t"),
        fp,
        "the data moved across the crash"
    );

    // The reopen tidied up on its way out, and left the live parts alone.
    let after = parts_in(&ctdir);
    assert!(!after.contains_key(&stale), "the stale file was not collected: {after:?}");
    for name in &now {
        assert_eq!(
            after.get(name),
            staged.get(name),
            "collecting the stale file rewrote live part {name}"
        );
    }
}

/// `kill -9` a live checkpoint at three polled trigger points and check what
/// recovered.
///
/// Polled rather than timed because the windows are one fsync wide and a
/// uniform sleep finds them about never:
///
///   * a `.tmp-` file exists in the table directory -- somewhere inside
///     `atomic_write`, between the create and the rename;
///   * a part file the previous commit did not name exists -- parts are being
///     written and the commit record has not moved yet;
///   * the commit record's inode changed -- the commit landed and what remains
///     is unlinking the superseded files.
///
/// Each trial's workload is plain `INSERT`s, every one of which fsyncs its log
/// record before returning, and none of which writes a part file. So reaching
/// any of these points proves all sixty were acknowledged, and the recovered
/// ids must be exactly `0..sent`.
#[test]
fn killing_a_checkpoint_at_swept_points_loses_no_acknowledged_write() {
    const POINTS: [&str; 3] = ["tmp-file", "new-part", "commit-record"];
    const ACKED: u64 = 60;
    let mut killed = 0;
    for trial in 0..9usize {
        let point = POINTS[trial % POINTS.len()];
        let s = Scratch::new(&format!("kill{trial}"));
        seed(&s, 2);
        let sent = 2 * CHUNK + ACKED;
        let old_parts: BTreeSet<String> = parts(&s).into_keys().collect();
        let table_ino = ino_of(&s.tdir().join("TABLE"));

        let mut script = String::new();
        for i in 0..ACKED {
            script.push_str(&format!("INSERT INTO t VALUES ({}, 'k', {i});\n", 2 * CHUNK + i));
        }
        let sp = s.path().join("w.sql");
        std::fs::write(&sp, &script).unwrap();

        let tdir = s.tdir();
        let mut trigger: Box<dyn FnMut() -> bool> = match point {
            "tmp-file" => Box::new(move || names_in(&tdir).iter().any(|n| n.contains(".tmp-"))),
            "new-part" => Box::new(move || {
                names_in(&tdir).iter().any(|n| n.starts_with("part_") && !old_parts.contains(n))
            }),
            _ => {
                let f = tdir.join("TABLE");
                Box::new(move || ino_of(&f) != table_ino)
            }
        };

        let child = Command::new(BIN)
            .args(["--data", s.db().to_str().unwrap(), "-f", sp.to_str().unwrap()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn granular");
        let hit = kill_when(child, &mut trigger, Duration::from_secs(30));
        killed += hit as usize;

        let ids = recovered(&s.db())
            .unwrap_or_else(|e| panic!("trial {trial} at {point}: reopen failed: {e}"));
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(
                id, i as u64,
                "trial {trial} at {point}: id {i} is missing while {id} survived -- an \
                 acknowledged write was lost ({} of {sent} recovered)",
                ids.len()
            );
        }
        assert!(
            ids.len() as u64 <= sent,
            "trial {trial} at {point}: recovered {} rows but only {sent} were written",
            ids.len()
        );
        if hit {
            assert_eq!(
                ids.len() as u64,
                sent,
                "trial {trial} at {point}: the kill landed inside the checkpoint, so every \
                 insert had already been acknowledged, but only {} of {sent} came back",
                ids.len()
            );
        }
    }
    assert!(killed > 0, "no trial was killed while checkpointing -- nothing was tested");
    eprintln!("  {killed}/9 trials were killed inside the checkpoint");
}

/// Poll `pred` every 50 us and `kill -9` the instant it holds. Returns whether
/// a live process was actually signalled.
fn kill_when(mut c: Child, pred: &mut dyn FnMut() -> bool, limit: Duration) -> bool {
    let t0 = Instant::now();
    loop {
        if pred() {
            let live = c.try_wait().expect("try_wait").is_none();
            if live {
                let _ = c.kill();
            }
            let _ = c.wait();
            return live;
        }
        if c.try_wait().expect("try_wait").is_some() {
            return false;
        }
        if t0.elapsed() > limit {
            let _ = c.kill();
            let _ = c.wait();
            return false;
        }
        std::thread::sleep(Duration::from_micros(50));
    }
}
