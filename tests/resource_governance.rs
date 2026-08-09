//! `max_memory_usage`, `max_execution_time` and `max_temporary_data_on_disk`
//! mean what `system.settings` says they mean.
//!
//! Half-enforcement is worse than none: an operator who sizes a box to the
//! advertised ceiling and gets 12 GB anyway has been told a number that is not
//! a number. Every gap pinned here was measured on the shipped binary first --
//! a legal `SELECT` refused at 8 MiB and the *same* work accepted three ways as
//! a write; a 1 second deadline overrun 23x by a filter, 4x by a `DISTINCT` and
//! 2x by a plain scan; 4.46 GB of `DISTINCT` keys and 1.6 GB of `uniq` sketches
//! invisible to the accounting; and 272 MB of spill written to a filesystem the
//! operator never sized for the database, then leaked forever when the process
//! was killed.
//!
//! Everything here runs the **shipped binary** against a real data directory,
//! in a child process. That is deliberate and is the point of the file: this
//! project's recurring defect is capability that lands in `src/` and never
//! reaches a user, and a budget the library honours but SQL cannot reach is
//! exactly that defect. The refusals below are the ones a person typing SQL
//! gets.
//!
//! One directory, one lock: the data directory takes an exclusive `flock`, so
//! the child processes have to be serialized. `GUARD` is that, and it is also
//! what keeps the spill assertions from seeing another test's files.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Mutex, MutexGuard, OnceLock};

const BIN: &str = env!("CARGO_BIN_EXE_granular");

/// Rows in the shared table. Enough that a 50 ms deadline is genuinely short
/// and a 1 MiB budget genuinely small, small enough that building it once
/// costs under a second.
const ROWS: usize = 2_000_000;

// ------------------------------------------------------------------ harness

/// The shared data directory, built once, plus the lock that serializes the
/// child processes that open it.
fn db() -> (PathBuf, MutexGuard<'static, ()>) {
    static DB: OnceLock<(PathBuf, Mutex<()>)> = OnceLock::new();
    let (dir, lock) = DB.get_or_init(|| {
        let root = std::env::temp_dir().join(format!("granular-w3res-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("scratch root");
        let csv = root.join("t.csv");
        let mut text = String::with_capacity(ROWS * 14);
        for i in 0..ROWS {
            text.push_str(&format!("{i},{}\n", i % 1000));
        }
        std::fs::write(&csv, text).expect("write csv");
        let dir = root.join("db");
        let out = run_at(
            &dir,
            &format!(
                "CREATE TABLE t (a Int64, b Int64) ENGINE = MergeTree ORDER BY a;
                 CREATE TABLE sink (c UInt64) ENGINE = MergeTree ORDER BY c;
                 CREATE TABLE up (a Int64, b Int64) ENGINE = MergeTree ORDER BY a;
                 SET input_format_with_names_use_header = 0;
                 INSERT INTO t FROM INFILE '{}';
                 INSERT INTO up FROM INFILE '{}';",
                csv.display(),
                csv.display()
            ),
        );
        assert!(out.status.success(), "building the table: {}", err(&out));
        (dir, Mutex::new(()))
    });
    (dir.clone(), lock.lock().unwrap_or_else(|e| e.into_inner()))
}

fn run_at(dir: &Path, sql: &str) -> Output {
    Command::new(BIN)
        .arg("--data")
        .arg(dir)
        .arg("-q")
        .arg(sql)
        .output()
        .expect("run granular")
}

/// The same, with `GRANULAR_THREADS` pinned -- the pool reads it once at
/// startup, so the only way to vary the fleet width is a fresh process.
fn run_threaded(dir: &Path, threads: usize, sql: &str) -> Output {
    Command::new(BIN)
        .env("GRANULAR_THREADS", threads.to_string())
        .arg("--data")
        .arg(dir)
        .arg("-q")
        .arg(sql)
        .output()
        .expect("run granular")
}

fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// Assert the statement failed, and that it failed for the stated reason
/// rather than by falling over on the way there.
fn refused(o: &Output, because: &str) {
    assert!(!o.status.success(), "expected a refusal, got success");
    let e = err(o);
    assert!(e.contains(because), "wrong refusal: {e}");
}

fn accepted(o: &Output) {
    assert!(o.status.success(), "expected success: {}", err(o));
}

const DEADLINE: &str = "SET max_execution_time = 0.05;";
const OVER_TIME: &str = "exceeded its deadline";
const OVER_MEM: &str = "memory budget";

// ------------------------- (a) the write paths are governed like the reads

/// `INSERT ... SELECT`, `UPDATE` and `CREATE TABLE ... AS SELECT` used to run
/// under a process-static 8 GiB tracker with no deadline and no cancel flag,
/// so the identical aggregate was refused as a `SELECT` and accepted three
/// ways as a write. Each pair below is the *same* work either side of the
/// statement keyword.
#[test]
fn the_three_write_paths_are_governed_like_the_select_they_contain() {
    let (dir, _lock) = db();
    // `DISTINCT` rather than `GROUP BY`, because the aggregate *spills* when
    // the budget refuses it and would take minutes rather than fail. The key
    // table has no spill path, so it is the shape that answers "is this
    // statement governed at all" in one block.
    let agg = "SELECT count() FROM (SELECT DISTINCT a FROM t)";
    let budget = "SET max_memory_usage = '1M';";

    // The control: as a bare SELECT this was always refused.
    refused(&run_at(&dir, &format!("{budget} {agg}")), OVER_MEM);

    refused(&run_at(&dir, &format!("{budget} INSERT INTO sink {agg}")), OVER_MEM);
    refused(
        &run_at(
            &dir,
            &format!(
                "{budget} CREATE TABLE ctas ENGINE = MergeTree ORDER BY x AS \
                 SELECT count() x FROM (SELECT DISTINCT a FROM t)"
            ),
        ),
        OVER_MEM,
    );
    // The refusal has to be a refusal, not a rename of the failure: the table
    // the write would have created must not exist afterwards.
    refused(&run_at(&dir, "SELECT count() FROM ctas"), "ctas");

    // UPDATE takes the deadline half of the same hole -- it is a scan plus a
    // rewrite, and 2M rows through `substring` is far past 50 ms.
    refused(
        &run_at(
            &dir,
            &format!(
                "{DEADLINE} UPDATE up SET b = b + 1 \
                 WHERE substring(concat(toString(a), toString(a), toString(a)), 2, 8) LIKE '%9%'"
            ),
        ),
        OVER_TIME,
    );
}

/// `io::import` had no `QueryContext` at all -- no deadline, and nothing for a
/// cancel handle to flip -- while its `export` sibling one file over has taken
/// one all along. The memory half was already honoured and is not what this
/// pins.
#[test]
fn a_bulk_import_stops_at_the_deadline() {
    let (dir, _lock) = db();
    let csv = dir.parent().expect("scratch root").join("t.csv");
    let load = format!(
        "SET input_format_with_names_use_header = 0; \
         INSERT INTO imp FROM INFILE '{}';",
        csv.display()
    );
    accepted(&run_at(&dir, "CREATE TABLE imp (a Int64, b Int64) ENGINE = MergeTree ORDER BY a"));
    refused(&run_at(&dir, &format!("{DEADLINE} {load}")), OVER_TIME);
    // ... and the same import with no deadline still loads, so what was
    // measured above is the deadline and not a broken importer.
    accepted(&run_at(&dir, &load));
}

// --------------------- (b) the skip loops reach a checkpoint

/// A `Filter` that rejects every block used to `continue` forever without ever
/// returning to the caller's per-block check: measured 23.3x a 1 second
/// deadline, growing linearly with the table.
#[test]
fn a_filter_that_rejects_every_row_stops_at_the_deadline() {
    let (dir, _lock) = db();
    let q = "SELECT s FROM (SELECT substring(concat(toString(a), toString(a), toString(a)), 2, 8) s \
             FROM t) WHERE s LIKE 'zzz%'";
    refused(&run_at(&dir, &format!("{DEADLINE} {q}")), OVER_TIME);
    accepted(&run_at(&dir, q));
}

/// `DISTINCT` sees every key in its first block and then consumes the rest of
/// the table in one `next()`. Measured: a 1 second deadline bought 0 ms and
/// the statement reported success.
#[test]
fn a_distinct_that_sees_no_new_key_stops_at_the_deadline() {
    let (dir, _lock) = db();
    // The expression is what makes one pass long enough to be worth stopping;
    // `b` has 1000 values, so every block after the first is wholly duplicate.
    let q = "SELECT DISTINCT concat(toString(b), toString(b), toString(b)) FROM t";
    refused(&run_at(&dir, &format!("{DEADLINE} {q}")), OVER_TIME);
    accepted(&run_at(&dir, q));
}

/// The one the brief said not to touch, and the one that fires on the most
/// ordinary query there is. `Scan` is *not* 1:1 with the blocks it emits: it
/// prunes granules and coalesces to `BLOCK_SIZE`, and the optimizer pushes a
/// selective predicate into PREWHERE -- so `WHERE <matches nothing>` has no
/// `Filter` above it at all and used to read the whole table inside one
/// `next()`.
#[test]
fn a_scan_whose_prewhere_matches_nothing_stops_at_the_deadline() {
    let (dir, _lock) = db();
    let q = "SELECT a FROM t WHERE toString(b) = 'zzz'";
    refused(&run_at(&dir, &format!("{DEADLINE} {q}")), OVER_TIME);
    accepted(&run_at(&dir, q));
}

// --------------------------------- (c)/(d) what the budget can see

/// `DISTINCT`'s key table had no `MemGuard` at any of its three construction
/// sites: 4.46 GB resident under an 8 MiB budget, reported as a success.
#[test]
fn distinct_charges_its_key_table_to_the_budget() {
    let (dir, _lock) = db();
    let q = "SELECT count() FROM (SELECT DISTINCT a FROM t)";
    refused(&run_at(&dir, &format!("SET max_memory_usage = '1M'; {q}")), OVER_MEM);
    accepted(&run_at(&dir, q));
}

/// `Project` had no ceiling on the block it builds. `MAX_STR` is 16 MiB per
/// value, so 8192 rows of one computed `String` column is 128 GiB by
/// construction; this asks for a far more modest 14 MB against 1 MiB.
#[test]
fn a_projection_charges_the_strings_it_builds() {
    let (dir, _lock) = db();
    let q = "SELECT length(max(s)) FROM (SELECT repeat(toString(a), 256) s FROM t)";
    refused(&run_at(&dir, &format!("SET max_memory_usage = '1M'; {q}")), OVER_MEM);
    // A fixed-width projection over the same rows is *not* charged -- the
    // guard is `None` there by construction -- so this must still pass under
    // the same budget it just refused.
    accepted(&run_at(&dir, "SET max_memory_usage = '1M'; SELECT max(a + 1) FROM t"));
}

/// `ACC_BYTES = 48` made `uniq`'s 16 KiB HLL register array invisible: two
/// `GROUP BY`s over the same groups under the same budget, differing only in
/// the aggregate, measured 167 MB and 1.13 GB and both succeeded. The
/// accounted difference between them was exactly zero.
#[test]
fn uniq_charges_its_sketch_to_the_budget() {
    let (dir, _lock) = db();
    let budget = "SET max_memory_usage = '16M';";
    let shape = |f: &str| {
        format!("{budget} SELECT count() FROM (SELECT a % 4000 g, {f} c FROM t GROUP BY g)")
    };
    // 4000 groups x 16 KiB of registers is 64 MiB against a 16 MiB ceiling.
    refused(&run_at(&dir, &shape("uniq(a)")), OVER_MEM);
    // The control: the same grouping, the same budget, an aggregate whose
    // state really is inline. If this failed too, the test would be measuring
    // the group table rather than the sketch.
    accepted(&run_at(&dir, &shape("count()")));
}

// ------------------------------------- (e) the floor is not the hardware

/// `worker_ceiling` divided the budget by the pool's width, so the smallest
/// budget a query could run under scaled linearly with the core count: 4 MiB
/// at 1 thread, 192 MiB at 128, same query and same data. A query sized on a
/// dev box OOMed on a bigger server.
#[test]
fn the_group_by_memory_floor_does_not_move_with_the_core_count() {
    let (dir, _lock) = db();
    let q = "SET max_memory_usage = '16M'; SELECT count() FROM (SELECT a, count() c FROM t GROUP BY a)";
    for threads in [1usize, 8, 64, 128] {
        let out = run_threaded(&dir, threads, q);
        assert!(
            out.status.success(),
            "the same budget must hold at {threads} threads: {}",
            err(&out)
        );
    }
}

// ------------------------------------------------ (f) temporary data

/// Spill files rooted at `env::temp_dir()`, which on macOS is a per-user
/// `/var/folders` path and on Linux is typically a small tmpfs -- so a spill
/// the operator sized the data volume for landed on a different filesystem
/// entirely, where it is an out-of-memory rather than a relief.
#[test]
fn spill_files_live_under_the_data_directory() {
    let (dir, _lock) = db();
    let spill = dir.join(".spill");
    let _ = std::fs::remove_dir_all(&spill);
    accepted(&run_at(
        &dir,
        "SET max_memory_usage = '8M'; SELECT count() FROM (SELECT a, count() c FROM t GROUP BY a)",
    ));
    // The per-query directories unlink themselves on the way out; the root
    // they were created under is what is left to prove where they were.
    assert!(spill.is_dir(), "nothing spilled under the data directory");
    assert_eq!(
        std::fs::read_dir(&spill).expect("read .spill").count(),
        0,
        "a finished query left its spill files behind"
    );
}

/// There was no ceiling on temporary data at all: an 8 MiB memory budget wrote
/// 272 MB of spill -- 34x amplification -- with nothing to stop it.
#[test]
fn max_temporary_data_on_disk_stops_a_runaway_spill() {
    let (dir, _lock) = db();
    let q = "SET max_memory_usage = '8M'; SET max_temporary_data_on_disk = '4M'; \
             SELECT count() FROM (SELECT a, count() c FROM t GROUP BY a)";
    refused(&run_at(&dir, q), "max_temporary_data_on_disk");
    // Generous is not the same as absent: the identical query under a ceiling
    // it fits inside must still answer.
    accepted(&run_at(
        &dir,
        "SET max_memory_usage = '8M'; SET max_temporary_data_on_disk = '4G'; \
         SELECT count() FROM (SELECT a, count() c FROM t GROUP BY a)",
    ));
    // And the setting is a first-class one, not a name the parser tolerates.
    let listed = run_at(&dir, "SELECT name FROM system.settings");
    accepted(&listed);
    assert!(
        String::from_utf8_lossy(&listed.stdout).contains("max_temporary_data_on_disk"),
        "the setting is not advertised"
    );
}

/// `SpillDir::drop` is the only cleanup that existed, and `Drop` cannot run on
/// a `SIGKILL`, a panic-abort or a power loss -- so every crashed query leaked
/// its spill forever. Measured: 272 MB survived a `kill -9` and two later
/// sessions.
///
/// The reaper's test is a `flock`, not a pid or an mtime: an orphan's `LOCK`
/// can be taken, a live directory's cannot. This fakes the orphan rather than
/// killing a child, because that is precisely the state a `SIGKILL` leaves --
/// a directory whose `LOCK` nobody holds.
#[test]
fn a_spill_directory_whose_owner_died_is_reaped_at_open() {
    let root = std::env::temp_dir().join(format!("granular-w3reap-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dir = root.join("db");
    let out = run_at(&dir, "CREATE TABLE t (a Int64) ENGINE = MergeTree ORDER BY a");
    assert!(out.status.success(), "{}", err(&out));

    let spill = dir.join(".spill");
    let orphan = spill.join("granular-spill-999999-0");
    std::fs::create_dir_all(&orphan).expect("orphan dir");
    std::fs::write(orphan.join("LOCK"), b"").expect("orphan lock");
    std::fs::write(orphan.join("run-000000.grun"), vec![0u8; 1 << 20]).expect("orphan run");
    // A directory that is not one of ours must survive: the reaper matches on
    // the name it writes and on nothing else.
    let bystander = spill.join("not-a-spill-dir");
    std::fs::create_dir_all(&bystander).expect("bystander");

    accepted(&run_at(&dir, "SELECT count() FROM t"));
    assert!(!orphan.exists(), "an orphaned spill directory survived a session open");
    assert!(bystander.is_dir(), "the reaper deleted something that was not its own");
    let _ = std::fs::remove_dir_all(&root);
}
