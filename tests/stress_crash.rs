//! Crash stress: `kill -9` a writing child at a swept point, reopen, and check
//! what survived against what was acknowledged.
//!
//! ## What "acknowledged" means here, and how it is observed
//!
//! Two oracles, because neither alone is enough.
//!
//! **The prefix oracle.** Every workload writes ids `0, 1, 2, ...` in order,
//! one statement each. Statement `i` returns `Ok` before statement `j > i`
//! starts, so if the recovered set contains `j` but not `i`, then `i` was
//! acknowledged and then lost. A *hole* is therefore a proof of data loss, and
//! it needs no channel back from the child at all. It is one-sided: a lost
//! **suffix** of acknowledged writes leaves no hole, which is what the second
//! oracle is for.
//!
//! **The ack oracle** (`every_acknowledged_insert_survives_a_kill`). The child
//! is made to emit more than 64 KiB after each `INSERT`, which is exactly the
//! capacity of the `BufWriter` the shell wraps stdout in -- so the buffer
//! flushes on every statement and the parent, reading the pipe as it goes,
//! observes a marker that the engine could only have written *after* it
//! returned `Ok` for that insert. A marker the parent has read is an
//! acknowledgement the parent can hold the engine to, and bytes already in the
//! pipe survive the child's death. That turns "everything the engine said Ok
//! to is present" into a checkable claim rather than an inference.
//!
//! The other half -- "and nothing it did not" -- is `max(id) < sent`: a row
//! that was never sent must never appear. Note that it is *not* a violation
//! for an unacknowledged row to be present: the log record is fsynced before
//! the acknowledgement is returned, so a crash in that window legitimately
//! recovers a write the client never heard about.
//!
//! ## Sweeping the kill point
//!
//! Timing alone is a poor sweep -- the phases of a write are wildly uneven, so
//! a uniform delay lands in the longest one almost every time. Three
//! techniques instead, in increasing order of precision:
//!
//!   * **calibrated timing** -- run once to completion, then kill at
//!     `frac * base` for a swept `frac`, with workloads shaped so that one
//!     phase (flush, compaction, DDL, transaction) dominates the run;
//!   * **state-triggered kills** -- poll the data directory and kill the
//!     instant it enters the state of interest: a `.tmp-` file exists
//!     (mid-rename), a log has been truncated (mid-checkpoint), one table's
//!     commit marker is down and the next one's is not (mid-commit). These are
//!     deterministic and reproduce every time;
//!   * **offline truncation** -- copy a crashed directory and cut `wal.log`
//!     back to every byte offset in turn. That is a crash at *every* point in
//!     the append path, exhaustively, with no scheduler involved.
//!
//! ## Runtime
//!
//! A plain `cargo test` runs this file in 26.5 s (31 s under `--release`),
//! which is the price of `kill -9` needing a real process and a real fsync --
//! ~60 child processes, most of the time spent in `F_FULLFSYNC`.
//! `GRANULAR_STRESS=n` multiplies every trial count and turns the truncation
//! sweep from every 64th byte into every byte; `GRANULAR_STRESS=6` is 126 s
//! and ~180 kill trials. `GRANULAR_STRESS_SEED` replays a run; every test
//! prints its seed and every failure prints its kill point.

#![cfg(unix)]

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicI64, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use granular::Session;

// ---------------------------------------------------------------- fixtures

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "granular-cr-{}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed),
            tag
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Scratch(d)
    }
    fn path(&self) -> &Path {
        &self.0
    }
    fn s(&self) -> &str {
        self.0.to_str().unwrap()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_granular")
}

fn scale() -> u64 {
    std::env::var("GRANULAR_STRESS").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
}

fn seed(tag: &str) -> u64 {
    let s = std::env::var("GRANULAR_STRESS_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0xA076_1D64_78BD_642F);
    eprintln!("stress_crash::{tag}: GRANULAR_STRESS_SEED={s} GRANULAR_STRESS={}", scale());
    s
}

struct Rng(u64);

impl Rng {
    fn new(s: u64) -> Rng {
        Rng(s | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// A fraction in `[0, 1)`, for placing a kill inside a calibrated run.
    fn frac(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ------------------------------------------------------------ child driver

/// Run the binary to completion. Returns `(exit ok, stdout)`.
fn cli(dir: &Scratch, args: &[&str]) -> (bool, String) {
    let out = Command::new(bin())
        .args(["--data", dir.s()])
        .args(args)
        .output()
        .expect("spawn granular");
    (out.status.success(), String::from_utf8_lossy(&out.stdout).into_owned())
}

fn setup(dir: &Scratch, ddl: &str) {
    let out =
        Command::new(bin()).args(["--data", dir.s(), "-q", ddl]).output().expect("spawn granular");
    assert!(out.status.success(), "setup failed: {}", String::from_utf8_lossy(&out.stderr));
}

fn write_script(dir: &Scratch, name: &str, body: &str) -> PathBuf {
    let p = dir.path().join(name);
    std::fs::write(&p, body).unwrap();
    p
}

/// Spawn a child running `script` against `dir`, with stdout discarded.
fn spawn(dir: &Scratch, script: &Path) -> Child {
    Command::new(bin())
        .args(["--data", dir.s(), "-f", script.to_str().unwrap()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn granular")
}

/// `kill -9` after `d`, reaping either way. Returns true if the signal was
/// actually delivered to a live process (a child that had already finished
/// tests nothing).
fn kill_after(mut c: Child, d: Duration) -> bool {
    std::thread::sleep(d);
    let live = c.try_wait().expect("try_wait").is_none();
    if live {
        let _ = c.kill();
    }
    let _ = c.wait();
    live
}

/// Poll `pred` every ~50 us and `kill -9` the instant it holds.
///
/// Precision beats randomness for the narrow windows -- a rename is one fsync
/// wide, and a uniform sleep would find it about never.
fn kill_when(mut c: Child, mut pred: impl FnMut() -> bool, limit: Duration) -> bool {
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
            return false; // finished before the state ever appeared
        }
        if t0.elapsed() > limit {
            let _ = c.kill();
            let _ = c.wait();
            return false;
        }
        std::thread::sleep(Duration::from_micros(50));
    }
}

/// Every `id` in `t`, in order, after reopening the directory in a *new*
/// process. A fresh process is the point: recovery has to run.
fn recovered(dir: &Scratch, table: &str) -> Result<Vec<u64>, String> {
    let out = Command::new(bin())
        .args(["--data", dir.s(), "-q", &format!("SELECT id FROM {table} ORDER BY id")])
        .args(["--format", "tsv", "--no-header"])
        .output()
        .expect("spawn granular");
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .split_ascii_whitespace()
        .map(|s| s.parse().expect("an integer id"))
        .collect())
}

/// The prefix oracle. `sent` is how many ids the workload offered; `acked` is
/// how many the parent *saw* acknowledged (0 when there is no ack channel).
fn assert_prefix(ids: &[u64], sent: u64, acked: u64, ctx: &str) {
    for (i, &id) in ids.iter().enumerate() {
        assert_eq!(
            id, i as u64,
            "{ctx}: recovered ids are not a prefix -- id {} is missing while {id} survived, \
             so an acknowledged write was lost. Recovered {} of {sent}.",
            i,
            ids.len()
        );
    }
    assert!(
        ids.len() as u64 <= sent,
        "{ctx}: recovered {} rows but only {sent} were ever sent",
        ids.len()
    );
    assert!(
        ids.len() as u64 >= acked,
        "{ctx}: {acked} inserts were acknowledged on stdout but only {} survived recovery",
        ids.len()
    );
}

const T_DDL: &str = "CREATE TABLE t (id UInt64, s String) ENGINE = MergeTree ORDER BY id";

/// `n` autocommit inserts, one statement each -- so every one is a log append
/// plus an fsync plus an acknowledgement, which is the shape the prefix oracle
/// needs.
fn insert_script(n: u64) -> String {
    let mut s = String::with_capacity(n as usize * 40);
    for i in 0..n {
        s.push_str(&format!("INSERT INTO t VALUES ({i},'r{i}');\n"));
    }
    s
}

// ------------------------------------------------------------------- tests

/// The broad sweep: kill an inserting child at a fraction of its calibrated
/// runtime, from just after start to just past the end, and check the prefix
/// oracle every time.
///
/// The tail of the sweep (`frac > 1`) lands inside the exit checkpoint, which
/// is where the parts are written and the logs truncated -- the most dangerous
/// window in the file, because it is the only one that *deletes* durable
/// records.
#[test]
fn kill_sweep_across_the_insert_path() {
    let s0 = seed("kill_sweep_across_the_insert_path");
    let mut rng = Rng::new(s0);
    let trials = 6 * scale();
    // Each autocommit insert is an append plus an fsync, ~4 ms on this
    // machine, so `N` is the run length in units of 4 ms: 200 gives a ~0.8 s
    // window to place a kill in, which is enough resolution and keeps the
    // default `cargo test` honest.
    const N: u64 = 200;

    // Calibrate once, on its own directory, so the sweep is in units of "this
    // machine right now" rather than a hard-coded millisecond count that would
    // be wrong on any other one.
    let base = {
        let dir = Scratch::new("cal");
        setup(&dir, T_DDL);
        let script = write_script(&dir, "w.sql", &insert_script(N));
        let t0 = Instant::now();
        let mut c = spawn(&dir, &script);
        assert!(c.wait().expect("wait").success());
        t0.elapsed()
    };
    eprintln!("  calibrated {N} inserts + checkpoint at {} ms", base.as_millis());

    let mut killed = 0;
    for trial in 0..trials {
        let dir = Scratch::new("sweep");
        setup(&dir, T_DDL);
        let script = write_script(&dir, "w.sql", &insert_script(N));
        // Up to 1.15x so the last few trials land inside the exit checkpoint.
        let frac = 0.02 + rng.frac() * 1.13;
        let at = base.mul_f64(frac);
        let live = kill_after(spawn(&dir, &script), at);
        killed += live as u64;
        let ids = recovered(&dir, "t")
            .unwrap_or_else(|e| panic!("trial {trial} (kill at {at:?}): reopen failed: {e}"));
        assert_prefix(&ids, N, 0, &format!("trial {trial}, kill at {at:?} ({frac:.3} of base)"));
    }
    assert!(killed > 0, "every trial finished before its kill -- the sweep tested nothing");
    eprintln!("  {killed}/{trials} trials were killed mid-write");
}

/// The precise form: everything the engine acknowledged on stdout is present
/// after recovery.
///
/// The child interleaves `INSERT k` with a `SELECT` that renders more than the
/// shell's 64 KiB stdout buffer, so each insert's acknowledgement is *pushed*
/// through the pipe instead of sitting in a buffer that `kill -9` discards.
/// Without that, the parent's view of "acked" would be up to 3000 statements
/// behind and the invariant would be vacuous.
#[test]
fn every_acknowledged_insert_survives_a_kill() {
    let s0 = seed("every_acknowledged_insert_survives_a_kill");
    let mut rng = Rng::new(s0 ^ 0x5DEE_CE66);
    let trials = 6 * scale();
    const N: u64 = 400;
    // 96 rows x ~900 bytes ~= 86 KiB per marker, comfortably over the 64 KiB
    // `BufWriter` in `main.rs`, which is what forces the flush.
    const PAD_ROWS: u64 = 96;

    let mut best_acked = 0u64;
    for trial in 0..trials {
        let dir = Scratch::new("ack");
        let pad = "p".repeat(900);
        let mut ddl = String::from(
            "CREATE TABLE t (id UInt64, s String) ENGINE = MergeTree ORDER BY id;\
             CREATE TABLE pad (p String) ENGINE = MergeTree ORDER BY p;\
             INSERT INTO pad VALUES ",
        );
        for i in 0..PAD_ROWS {
            if i > 0 {
                ddl.push(',');
            }
            ddl.push_str(&format!("('{i}{pad}')"));
        }
        setup(&dir, &ddl);

        let mut body = String::with_capacity(N as usize * 64);
        for i in 0..N {
            body.push_str(&format!("INSERT INTO t VALUES ({i},'r{i}');\nSELECT {i},p FROM pad;\n"));
        }
        let script = write_script(&dir, "w.sql", &body);

        let mut child = Command::new(bin())
            .args(["--data", dir.s(), "-f", script.to_str().unwrap()])
            .args(["--format", "tsv", "--no-header"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn granular");
        let out = child.stdout.take().expect("piped");

        // Drained on its own thread: a 64 KiB pipe would otherwise fill and
        // block the child, which would turn the ack channel into a throttle.
        let acked = Arc::new(AtomicI64::new(-1));
        let a = Arc::clone(&acked);
        let pump = std::thread::spawn(move || {
            let mut r = BufReader::with_capacity(1 << 16, out);
            let mut line = String::new();
            // `read_line` into one reused `String`: the marker rows are ~900
            // bytes each and there are tens of thousands of them.
            while r.read_line(&mut line).unwrap_or(0) > 0 {
                if let Some(k) = line.split('\t').next().and_then(|f| f.trim().parse::<i64>().ok())
                {
                    a.fetch_max(k, Ordering::Relaxed);
                }
                line.clear();
            }
        });

        // Somewhere in the first three quarters of the workload, so the kill
        // is genuinely mid-stream.
        let at = Duration::from_millis(80 + (rng.frac() * 1200.0) as u64);
        std::thread::sleep(at);
        let live = child.try_wait().expect("try_wait").is_none();
        if live {
            let _ = child.kill();
        }
        let _ = child.wait();
        pump.join().expect("pump");

        let seen = acked.load(Ordering::Relaxed);
        let n_acked = (seen + 1).max(0) as u64;
        best_acked = best_acked.max(n_acked);
        let ids = recovered(&dir, "t")
            .unwrap_or_else(|e| panic!("trial {trial} (kill at {at:?}): reopen failed: {e}"));
        assert_prefix(
            &ids,
            N,
            n_acked,
            &format!("trial {trial}, kill at {at:?}, {n_acked} acknowledged"),
        );
        assert!(live || n_acked == N, "trial {trial}: child finished early with {n_acked} acked");
    }
    assert!(
        best_acked > 0,
        "no insert was ever observed acknowledged -- the ack channel is not working, \
         so this test proves nothing"
    );
    eprintln!("  deepest observed acknowledgement: {best_acked} inserts");
}

/// Kill inside a flush and inside a compaction.
///
/// `SYSTEM FLUSH` after every batch turns the run into a sequence of part
/// builds, and `OPTIMIZE TABLE t FINAL` every eighth batch makes it a sequence
/// of k-way merges that unlink their inputs. Both rewrite the part set; a
/// crash inside either must leave the *previous* set intact, because a part is
/// only ever published by a `TABLE` rename.
#[test]
fn kill_during_flush_and_compaction() {
    let s0 = seed("kill_during_flush_and_compaction");
    let mut rng = Rng::new(s0 ^ 0x1D8E_3D75);
    let trials = 8 * scale();
    const BATCHES: u64 = 40;
    const PER: u64 = 8;

    let body = {
        let mut s = String::new();
        for b in 0..BATCHES {
            s.push_str("INSERT INTO t VALUES ");
            for j in 0..PER {
                let id = b * PER + j;
                if j > 0 {
                    s.push(',');
                }
                s.push_str(&format!("({id},'r{id}')"));
            }
            s.push_str(";\nSYSTEM FLUSH;\n");
            if b % 8 == 7 {
                s.push_str("OPTIMIZE TABLE t FINAL;\n");
            }
        }
        s
    };

    let base = {
        let dir = Scratch::new("cal-flush");
        setup(&dir, T_DDL);
        let script = write_script(&dir, "w.sql", &body);
        let t0 = Instant::now();
        assert!(spawn(&dir, &script).wait().expect("wait").success());
        t0.elapsed()
    };

    let mut killed = 0;
    for trial in 0..trials {
        let dir = Scratch::new("flush");
        setup(&dir, T_DDL);
        let script = write_script(&dir, "w.sql", &body);
        let frac = 0.05 + rng.frac() * 1.1;
        let at = base.mul_f64(frac);
        killed += kill_after(spawn(&dir, &script), at) as u64;
        let ids = recovered(&dir, "t")
            .unwrap_or_else(|e| panic!("trial {trial} (kill at {at:?}): reopen failed: {e}"));
        // Batches are atomic, so a recovered set is a whole number of them.
        assert_prefix(&ids, BATCHES * PER, 0, &format!("trial {trial}, kill at {at:?}"));
        assert_eq!(
            ids.len() as u64 % PER,
            0,
            "trial {trial} (kill at {at:?}): recovered {} rows, not a whole number of \
             {PER}-row statements",
            ids.len()
        );
    }
    assert!(killed > 0, "every trial finished before its kill");
}

/// Kill while the checkpoint is deleting durable records.
///
/// The exit checkpoint writes each table's parts and then **truncates its
/// log**, which is the only operation in the engine that destroys a record it
/// has already acknowledged. The trigger is exact: poll for the log to shrink,
/// and kill on the same millisecond. If the parts were not committed first,
/// this loses everything the log held.
#[test]
fn kill_during_checkpoint_log_truncation() {
    seed("kill_during_checkpoint_log_truncation");
    let trials = 4 * scale();
    const N: u64 = 150;

    let mut killed = 0;
    for trial in 0..trials {
        let dir = Scratch::new("ckpt");
        setup(&dir, T_DDL);
        let script = write_script(&dir, "w.sql", &insert_script(N));
        let wal = dir.path().join("default").join("t").join("wal.log");
        let mut high = 0u64;
        let live = kill_when(
            spawn(&dir, &script),
            || {
                let n = std::fs::metadata(&wal).map(|m| m.len()).unwrap_or(0);
                high = high.max(n);
                // Shrunk from its high-water mark: `Wal::truncate` has run and
                // the process is inside `save_catalog`.
                // 2048 is well above the 12-byte header and well below the
                // ~6 KiB the 150 records here occupy (~42 bytes each,
                // measured), so the gate cannot fire on a log that has not
                // been written to yet and cannot fail to fire on one that has.
                high > 2048 && n < high
            },
            Duration::from_secs(60),
        );
        killed += live as u64;
        let ids = recovered(&dir, "t")
            .unwrap_or_else(|e| panic!("trial {trial}: reopen after checkpoint kill failed: {e}"));
        assert_prefix(&ids, N, 0, &format!("trial {trial}, killed mid-checkpoint (live={live})"));
        // The checkpoint only starts once every insert has been acknowledged,
        // so nothing may be missing at all.
        assert_eq!(
            ids.len() as u64,
            N,
            "trial {trial}: a kill during the checkpoint lost {} acknowledged rows",
            N - ids.len() as u64
        );
    }
    assert!(killed > 0, "the log never shrank -- the trigger never fired");
}

/// Kill in the one-instruction window between the temp file and the rename.
///
/// `store::atomic_write` writes `.TABLE.tmp-<pid>-<n>`, fsyncs it, renames it
/// over `TABLE` and fsyncs the directory. Every `DELETE` here folds to parts
/// and goes through it, so the window recurs; the parent polls for the temp
/// file and kills inside it. Either version of `TABLE` is a correct outcome --
/// a *torn* one is not, and neither is a leftover temp file that stops the
/// directory reopening.
#[test]
fn kill_between_the_temp_file_and_the_rename() {
    seed("kill_between_the_temp_file_and_the_rename");
    let trials = 4 * scale();
    const N: u64 = 60;

    let mut fired = 0;
    for trial in 0..trials {
        let dir = Scratch::new("rename");
        setup(&dir, T_DDL);
        let mut body = insert_script(N);
        body.push_str("SYSTEM FLUSH;\n");
        for i in 0..N {
            body.push_str(&format!("DELETE FROM t WHERE id = {i};\n"));
        }
        let script = write_script(&dir, "w.sql", &body);
        let tdir = dir.path().join("default").join("t");
        let live = kill_when(
            spawn(&dir, &script),
            || {
                std::fs::read_dir(&tdir).is_ok_and(|rd| {
                    rd.flatten().any(|e| e.file_name().to_string_lossy().contains(".tmp-"))
                })
            },
            Duration::from_secs(60),
        );
        fired += live as u64;
        let ids = recovered(&dir, "t").unwrap_or_else(|e| {
            panic!("trial {trial}: a kill between temp file and rename wedged the directory: {e}")
        });
        // The deletes run front to back, so the survivors are a *suffix*:
        // check that shape directly rather than through `assert_prefix`.
        let lo = N - ids.len() as u64;
        for (k, &id) in ids.iter().enumerate() {
            assert_eq!(
                id,
                lo + k as u64,
                "trial {trial}: survivors are not a suffix of the delete order: {ids:?}"
            );
        }
    }
    eprintln!("  {fired}/{trials} trials caught the rename window");
}

/// Kill in the middle of DDL. Recovery must find a catalog, not a half-written
/// one, and every table it lists must open.
///
/// `TABLES` is 8 and not 60, and the reason is a defect rather than a taste:
/// `Session::run` checkpoints the **whole catalog** after every DDL statement
/// (`session.rs:1184`), so creating `n` tables costs `O(n^2)` fsyncs. Measured
/// on this machine, `CREATE TABLE` x n from a cold directory: n=5 0.90 s,
/// n=10 2.62 s, n=20 8.61 s, n=40 32.67 s -- a clean quadratic, and the reason
/// the first version of this test ran for four minutes. Eight tables is the
/// largest schema this can sweep and still belong in `cargo test`.
#[test]
fn kill_during_ddl() {
    let s0 = seed("kill_during_ddl");
    let mut rng = Rng::new(s0 ^ 0xB5AD_4ECE);
    let trials = 2 * scale();
    const TABLES: u64 = 6;

    let body = {
        let mut s = String::new();
        for k in 0..TABLES {
            s.push_str(&format!(
                "CREATE TABLE d{k} (id UInt64, s String) ENGINE = MergeTree ORDER BY id;\n\
                 INSERT INTO d{k} VALUES ({k},'r{k}');\n"
            ));
            if k % 5 == 4 {
                s.push_str(&format!("DROP TABLE d{};\n", k - 4));
            }
        }
        s
    };

    let base = {
        let dir = Scratch::new("cal-ddl");
        setup(&dir, T_DDL);
        let script = write_script(&dir, "w.sql", &body);
        let t0 = Instant::now();
        assert!(spawn(&dir, &script).wait().expect("wait").success());
        t0.elapsed()
    };

    for trial in 0..trials {
        let dir = Scratch::new("ddl");
        setup(&dir, T_DDL);
        let script = write_script(&dir, "w.sql", &body);
        let at = base.mul_f64(0.05 + rng.frac() * 1.1);
        kill_after(spawn(&dir, &script), at);

        // `SHOW TABLES` lists what the catalog claims; every one of them must
        // then answer. One process for the whole check, not one per table:
        // each open re-reads the catalog *and* checkpoints on exit, so the
        // per-table spelling was quadratic and took longer than the crashes.
        let (ok, listed) = cli(&dir, &["-q", "SHOW TABLES", "--format", "tsv", "--no-header"]);
        assert!(ok, "trial {trial} (kill at {at:?}): the catalog would not open");
        let names: Vec<&str> = listed.split_ascii_whitespace().collect();
        assert!(!names.is_empty(), "trial {trial}: the catalog came back empty");
        let probe = names
            .iter()
            .map(|n| format!("SELECT count() FROM {n}"))
            .collect::<Vec<_>>()
            .join(";");
        // A script bails at its first error, so a failure here is "one of
        // these tables is listed and does not open".
        let (ok, _) = cli(&dir, &["-q", &probe, "--format", "tsv", "--no-header"]);
        assert!(
            ok,
            "trial {trial} (kill at {at:?}): the catalog lists {names:?} but one does not open"
        );
    }
}

/// A single-table transaction is all or nothing across a crash.
///
/// The transaction stages `ROWS` rows under one sequence number and releases
/// them with one commit marker, so the only two recoverable states are 0 and
/// `ROWS`. Anything in between means a staging group was released without its
/// marker.
#[test]
fn single_table_transaction_is_all_or_nothing() {
    let s0 = seed("single_table_transaction_is_all_or_nothing");
    let mut rng = Rng::new(s0 ^ 0x2545_F491);
    let trials = 6 * scale();
    const TXNS: u64 = 12;
    const ROWS: u64 = 40;

    let body = {
        let mut s = String::new();
        for t in 0..TXNS {
            s.push_str("BEGIN;\n");
            for j in 0..ROWS {
                let id = t * ROWS + j;
                s.push_str(&format!("INSERT INTO t VALUES ({id},'r{id}');\n"));
            }
            s.push_str("COMMIT;\n");
        }
        s
    };

    let base = {
        let dir = Scratch::new("cal-txn");
        setup(&dir, T_DDL);
        let script = write_script(&dir, "w.sql", &body);
        let t0 = Instant::now();
        assert!(spawn(&dir, &script).wait().expect("wait").success());
        t0.elapsed()
    };

    let mut killed = 0;
    for trial in 0..trials {
        let dir = Scratch::new("txn");
        setup(&dir, T_DDL);
        let script = write_script(&dir, "w.sql", &body);
        let at = base.mul_f64(0.05 + rng.frac() * 1.1);
        killed += kill_after(spawn(&dir, &script), at) as u64;
        let ids = recovered(&dir, "t")
            .unwrap_or_else(|e| panic!("trial {trial} (kill at {at:?}): reopen failed: {e}"));
        assert_prefix(&ids, TXNS * ROWS, 0, &format!("trial {trial}, kill at {at:?}"));
        assert_eq!(
            ids.len() as u64 % ROWS,
            0,
            "trial {trial} (kill at {at:?}): recovered {} rows -- a {ROWS}-row transaction was \
             committed in part",
            ids.len()
        );
    }
    assert!(killed > 0, "every trial finished before its kill");
}

/// **Known limitation, documented here rather than left as folklore.**
///
/// The log is per table, and `commit_durable` walks the enlisted tables
/// appending a commit marker and fsyncing each one in turn. A crash between
/// two of those fsyncs releases the staging groups of the tables already
/// walked and drops the rest -- so a transaction over `T` tables can commit a
/// **prefix** of itself.
///
/// The trigger needs no timing at all. Every table's log is the same size
/// after the staged inserts, and the markers go down in enlistment order, so
/// "the first table's log has grown and the last one's has not" is exactly the
/// mid-commit state, and it holds for as long as the remaining fsyncs take.
/// The reproduction rate is 100%.
///
/// This test asserts the **current** behaviour. When the two-phase commit that
/// fixes it lands, this test fails, and the assertion below is the one to
/// invert: `first == PER && last == 0` becomes `first == last`.
#[test]
fn multi_table_transaction_commits_a_prefix_on_crash() {
    seed("multi_table_transaction_commits_a_prefix_on_crash");
    const TABLES: usize = 12;
    // Big enough that each per-table fsync at commit is milliseconds, which is
    // what makes the mid-commit state observable from another process.
    const PER: u64 = 400;

    let dir = Scratch::new("mtxn");
    let ddl: Vec<String> = (0..TABLES)
        .map(|k| format!("CREATE TABLE t{k} (id UInt64, s String) ENGINE = MergeTree ORDER BY id"))
        .collect();
    setup(&dir, &ddl.join(";"));

    let mut body = String::from("BEGIN;\n");
    for k in 0..TABLES {
        body.push_str(&format!("INSERT INTO t{k} VALUES "));
        for i in 0..PER {
            if i > 0 {
                body.push(',');
            }
            body.push_str(&format!("({i},'padpadpadpadpadpadpadpadpad{i}')"));
        }
        body.push_str(";\n");
    }
    body.push_str("COMMIT;\n");
    let script = write_script(&dir, "w.sql", &body);

    let wal = |k: usize| {
        std::fs::metadata(dir.path().join("default").join(format!("t{k}")).join("wal.log"))
            .map(|m| m.len())
            .unwrap_or(0)
    };
    let mut staged = 0u64;
    let live = kill_when(
        spawn(&dir, &script),
        || {
            let (first, last) = (wal(0), wal(TABLES - 1));
            if staged == 0 {
                // All logs the same non-trivial size: the staged inserts are
                // down and COMMIT has not started.
                if first > 4096 && first == last {
                    staged = first;
                }
                return false;
            }
            first > staged && last == staged
        },
        Duration::from_secs(60),
    );
    assert!(live, "never reached the mid-commit state -- the trigger needs revisiting");

    // One reopen for all twelve counts: twelve would each replay twelve logs
    // and checkpoint twelve tables on the way out.
    let probe = (0..TABLES).map(|k| format!("SELECT count() FROM t{k}")).collect::<Vec<_>>();
    let (ok, out) = cli(&dir, &["-q", &probe.join(";"), "--format", "tsv", "--no-header"]);
    assert!(ok, "the directory would not reopen after a crash inside COMMIT");
    let counts: Vec<u64> =
        out.split_ascii_whitespace().map(|s| s.parse().expect("a count")).collect();
    assert_eq!(counts.len(), TABLES);
    eprintln!("  per-table row counts after a crash inside COMMIT: {counts:?}");

    // The bug, stated as an assertion so the fix has a target: the first table
    // committed and the last did not, out of one transaction.
    assert_eq!(counts[0], PER, "table 0 did not commit; the trigger fired too early");
    assert_eq!(
        *counts.last().unwrap(),
        0,
        "the last table committed too -- if this now holds for every trial, the per-table \
         commit has been replaced and this test should be inverted to `counts[0] == \
         *counts.last().unwrap()`"
    );
    assert!(
        counts.iter().any(|&c| c != counts[0]),
        "KNOWN LIMITATION no longer reproduces: the transaction was atomic across tables"
    );
}

/// A process that *exits normally* with a transaction still open.
///
/// The durability half is right and is asserted: the uncommitted rows are
/// gone, the staged group is orphaned, and -- the part that is easy to get
/// wrong -- a later transaction's commit marker does **not** release it, so
/// the orphan stays dropped across any number of subsequent commits.
///
/// The contract half is wrong, and is asserted so the fix has a target: the
/// shell exits 1 with `NOT_IMPLEMENTED: cannot checkpoint inside a
/// transaction`, because `main.rs` runs `session.checkpoint()` unconditionally
/// and `Session::checkpoint` refuses inside a transaction. Every statement the
/// script contained succeeded; the failure is the shell's own epilogue, and
/// the message describes an internal call rather than what the user did.
/// `psql` rolls back and says so. Rolling back before the exit checkpoint
/// turns this into exit 0, and then the assertion below is the one to invert.
#[test]
fn a_script_that_ends_inside_a_transaction() {
    seed("a_script_that_ends_inside_a_transaction");
    let dir = Scratch::new("open-txn");
    setup(&dir, T_DDL);
    let (ok, _) = cli(&dir, &["-q", "INSERT INTO t VALUES (0,'r0')"]);
    assert!(ok);

    let out = Command::new(bin())
        .args(["--data", dir.s(), "-q", "BEGIN; INSERT INTO t VALUES (1,'r1'); \
                                         INSERT INTO t VALUES (2,'r2')"])
        .output()
        .expect("spawn");
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    eprintln!("  exit {:?}, stderr: {}", out.status.code(), err.trim());

    // Durability: the uncommitted rows are not there.
    assert_eq!(recovered(&dir, "t").unwrap(), vec![0], "an unterminated transaction was durable");

    // And a later, properly committed transaction does not release the
    // orphaned staging group behind it -- the sequence numbers are resumed
    // from the file for exactly this reason.
    let (ok, _) = cli(&dir, &["-q", "BEGIN; INSERT INTO t VALUES (3,'r3'); COMMIT"]);
    assert!(ok);
    assert_eq!(
        recovered(&dir, "t").unwrap(),
        vec![0, 3],
        "a later COMMIT released an orphaned staging group"
    );

    assert!(
        !out.status.success() && err.contains("cannot checkpoint inside a transaction"),
        "the shell no longer reports the exit checkpoint's refusal for a script that ended \
         inside a transaction -- if it now rolls back and exits 0, invert this assertion"
    );
}

/// A crash in the middle of a log append, at **every** byte offset.
///
/// Timing cannot sweep this: an append is a few hundred nanoseconds. Cutting a
/// copy of a real crashed log back to offset `L` produces exactly the file a
/// crash at that instant would have left, for every `L` in turn, and recovery
/// has to answer the same way each time -- a prefix of the records, never a
/// hole, never a panic, never a row that was not written.
///
/// The default run steps 64 bytes (a record here is ~40), so every record
/// boundary and several interiors are covered; `GRANULAR_STRESS>=8` steps 1
/// and covers literally every offset.
#[test]
fn truncated_wal_recovers_a_prefix_at_every_offset() {
    seed("truncated_wal_recovers_a_prefix_at_every_offset");
    const N: u64 = 120;
    let src = Scratch::new("wal-src");

    // In process, and dropped without a checkpoint: `Session` has no `Drop`
    // checkpoint, so the log still holds every record. Going through the CLI
    // would checkpoint at exit and truncate the very file under test.
    {
        let mut s = Session::open(src.path()).unwrap();
        s.execute(T_DDL).unwrap();
        for i in 0..N {
            s.execute(&format!("INSERT INTO t VALUES ({i},'r{i}')")).unwrap();
        }
    }
    let wal_src = src.path().join("default").join("t").join("wal.log");
    let bytes = std::fs::read(&wal_src).expect("read wal");
    assert!(bytes.len() > 1000, "the log is empty -- the fixture wrote nothing to it");

    let step = if scale() >= 8 { 1 } else { 64 };
    let mut widest = 0usize;
    let mut cut = (0..bytes.len()).step_by(step).collect::<Vec<_>>();
    cut.push(bytes.len());
    for &len in &cut {
        let dst = Scratch::new("wal-cut");
        copy_tree(src.path(), dst.path());
        // The LOCK file is a lock, not state; copying it is harmless, but the
        // stale `.tmp-` files a copy might pick up are not, so copy_tree skips
        // them.
        std::fs::write(dst.path().join("default").join("t").join("wal.log"), &bytes[..len])
            .unwrap();

        let mut s = match Session::open(dst.path()) {
            Ok(s) => s,
            // A refusal is allowed -- corruption that is *not* a torn tail is
            // documented as reported rather than swallowed. A wrong answer is
            // not allowed, and neither is a panic, which would fail here.
            Err(_) => continue,
        };
        let rs = s.query("SELECT id FROM t ORDER BY id").expect("query after truncation");
        let ids: Vec<u64> = rs
            .to_values()
            .iter()
            .map(|r| match r[0] {
                granular::Value::UInt(n) => n,
                ref o => panic!("id came back as {o}"),
            })
            .collect();
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(
                id, i as u64,
                "wal cut to {len} of {}: recovered ids are not a prefix ({ids:?})",
                bytes.len()
            );
        }
        assert!(
            ids.len() as u64 <= N,
            "wal cut to {len}: recovered {} rows, more than the {N} written",
            ids.len()
        );
        widest = widest.max(ids.len());
    }
    assert_eq!(widest as u64, N, "the untruncated copy did not recover every row");
    eprintln!("  swept {} truncation points, step {step}", cut.len());
}

/// Copy a data directory, skipping the lock file and any temp files.
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap().flatten() {
        let name = e.file_name();
        let n = name.to_string_lossy();
        if n == "LOCK" || n.contains(".tmp-") {
            continue;
        }
        let (src, dst) = (e.path(), to.join(&name));
        if e.file_type().unwrap().is_dir() {
            copy_tree(&src, &dst);
        } else {
            std::fs::copy(&src, &dst).unwrap();
        }
    }
}

// ------------------------------------------------------------- disk full

/// `setrlimit(RLIMIT_FSIZE)` and `SIGXFSZ`, hand-declared for the same reason
/// as the `flock` in `session.rs` and the `mmap` in `persist/mmap.rs`: the
/// crate has no dependencies and this is three symbols.
///
/// A file-size rlimit rather than a real full volume because it needs no
/// privilege, no `hdiutil`, and no cleanup that a killed test could skip -- and
/// the engine cannot tell the difference: both surface as a failed `write`.
/// `SIGXFSZ` has to be ignored or the kernel kills the child at the first
/// oversized write and we would be testing signal delivery instead of the
/// engine's error path.
mod rlimit_sys {
    pub type CInt = i32;

    #[repr(C)]
    pub struct RLimit {
        pub cur: u64,
        pub max: u64,
    }

    /// 1 on every unix that has it -- 4.2BSD assigned it and Linux copied the
    /// numbering.
    pub const RLIMIT_FSIZE: CInt = 1;
    pub const SIGXFSZ: CInt = 25;
    pub const SIG_IGN: usize = 1;

    extern "C" {
        pub fn setrlimit(resource: CInt, rlp: *const RLimit) -> CInt;
        pub fn signal(sig: CInt, handler: usize) -> usize;
    }
}

/// Spawn the binary with every file it writes capped at `cap` bytes.
fn spawn_capped(dir: &Scratch, args: &[&str], cap: u64) -> Child {
    use std::os::unix::process::CommandExt;
    let mut cmd = Command::new(bin());
    cmd.args(["--data", dir.s()]).args(args).stdout(Stdio::null()).stderr(Stdio::piped());
    // SAFETY: runs in the child between fork and exec. Both calls are plain
    // syscalls with no allocation and no lock, which is the whole requirement.
    unsafe {
        cmd.pre_exec(move || {
            rlimit_sys::signal(rlimit_sys::SIGXFSZ, rlimit_sys::SIG_IGN);
            let l = rlimit_sys::RLimit { cur: cap, max: cap };
            if rlimit_sys::setrlimit(rlimit_sys::RLIMIT_FSIZE, &l) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    cmd.spawn().expect("spawn granular")
}

/// A full disk must fail the write, not the database.
///
/// The child writes under a file-size cap until a write fails. Then, with the
/// cap lifted, the directory must still open, still answer, and hold a prefix
/// of what was sent -- an ENOSPC that leaves an unopenable directory turns a
/// transient condition into permanent loss.
#[test]
fn a_full_disk_fails_the_write_and_not_the_database() {
    seed("a_full_disk_fails_the_write_and_not_the_database");
    let dir = Scratch::new("nospc");
    setup(&dir, T_DDL);

    const N: u64 = 400;
    let script = write_script(&dir, "w.sql", &insert_script(N));
    // Above the header and a few records, below what 400 of them need, so the
    // failure lands mid-workload rather than at the first byte.
    let mut child = spawn_capped(&dir, &["-f", script.to_str().unwrap()], 4096);
    let mut err = String::new();
    if let Some(mut e) = child.stderr.take() {
        use std::io::Read;
        let _ = e.read_to_string(&mut err);
    }
    let st = child.wait().expect("wait");
    assert!(
        !st.success(),
        "the child reported success while every file it wrote was capped at 4 KiB"
    );
    assert!(
        err.contains("File too large") || err.contains("EFBIG") || err.contains("cannot"),
        "the out-of-space failure did not name an I/O error: {err}"
    );

    let ids = recovered(&dir, "t")
        .unwrap_or_else(|e| panic!("a directory that hit ENOSPC will not reopen: {e}"));
    for (i, &id) in ids.iter().enumerate() {
        assert_eq!(id, i as u64, "post-ENOSPC recovery is not a prefix: {ids:?}");
    }
    assert!(ids.len() as u64 <= N);
    eprintln!("  {} of {N} inserts survived the cap", ids.len());

    // And it keeps working once there is room again.
    let (ok, _) = cli(&dir, &["-q", &format!("INSERT INTO t VALUES ({N},'after')")]);
    assert!(ok, "the directory stayed wedged after the cap was lifted");
    assert_eq!(recovered(&dir, "t").unwrap().len(), ids.len() + 1);
}

/// A read on a full disk must still answer.
///
/// The audit's report was that a near-full volume wedges the instance
/// "including for read-only queries", and this is the reproduction: a
/// directory whose part file is already larger than the cap, and a child that
/// runs one `SELECT` and nothing else.
///
/// The claim under test is about the **shell**, not the engine: `main.rs`
/// checkpoints unconditionally on exit, so a pure `SELECT` rewrites every
/// table's parts, and rewriting a part that is bigger than the remaining space
/// fails. The engine's own read path never writes.
#[test]
fn a_read_only_query_on_a_full_disk() {
    seed("a_read_only_query_on_a_full_disk");
    let dir = Scratch::new("nospc-read");
    setup(&dir, T_DDL);

    // ~64 KiB of part, comfortably over the 16 KiB cap below.
    let mut ins = String::from("INSERT INTO t VALUES ");
    for i in 0..2000u64 {
        if i > 0 {
            ins.push(',');
        }
        ins.push_str(&format!("({i},'row-{i}-padpadpadpad')"));
    }
    let (ok, _) = cli(&dir, &["-q", &ins]);
    assert!(ok, "fixture insert failed");
    let part_bytes: u64 = std::fs::read_dir(dir.path().join("default").join("t"))
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".gpart"))
        .map(|e| e.metadata().unwrap().len())
        .sum();
    assert!(part_bytes > 16 * 1024, "fixture part is only {part_bytes} bytes");

    // Three times, because the interesting question is not only whether one
    // fails but whether the failures *accumulate*: a `.tmp-` file left behind
    // by each aborted part write is how a transient full disk becomes a
    // permanent one.
    let mut codes = Vec::new();
    let mut err = String::new();
    for _ in 0..3 {
        let mut child = spawn_capped(&dir, &["-q", "SELECT count() FROM t"], 16 * 1024);
        err.clear();
        if let Some(mut e) = child.stderr.take() {
            use std::io::Read;
            let _ = e.read_to_string(&mut err);
        }
        codes.push(child.wait().expect("wait").code());
        assert!(
            !err.contains("panicked"),
            "a full disk panicked the shell instead of reporting an error: {err}"
        );
    }
    eprintln!(
        "  read-only `SELECT count()` under a 16 KiB file cap, x3: exits {codes:?}, \
         last stderr: {}",
        err.trim()
    );

    let leftovers: Vec<String> = std::fs::read_dir(dir.path().join("default").join("t"))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(".tmp-"))
        .collect();
    assert!(leftovers.is_empty(), "aborted checkpoints left temp files behind: {leftovers:?}");

    let ids = recovered(&dir, "t")
        .unwrap_or_else(|e| panic!("a read-only query on a full disk destroyed the directory: {e}"));
    assert_eq!(ids.len(), 2000, "a read-only query lost rows on a full disk");

    // **The defect, asserted so it has a target.** Nothing in `SELECT
    // count()` needs to write, but `main.rs` runs `session.checkpoint()`
    // unconditionally before exiting, so a query that reads rewrites every
    // part -- and on a volume with no room, a pure read fails. Give the shell
    // a read-only mode (or make the exit checkpoint conditional on the session
    // having written) and this assertion becomes `codes == [Some(0); 3]`.
    assert!(
        codes.iter().all(|c| *c == Some(1)),
        "a read-only query no longer fails on a full disk (exits {codes:?}) -- if the exit \
         checkpoint has been made conditional, invert this assertion to `Some(0)`"
    );
}
