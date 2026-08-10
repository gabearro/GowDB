//! The append path allocates nothing per record. This is the test that says so.
//!
//! "Nothing new allocated per record on the append path" is a standing rule
//! for this engine, and until now nothing in the tree measured it -- so it had
//! quietly become false: four allocations per insert record (the framing
//! buffer, a second buffer for the eighteen-byte frame header, and a `String`
//! per column for the type name, twice over). A rule nobody measures is a
//! comment.
//!
//! An integration test is its own binary, so it may install a
//! `#[global_allocator]`; the library under test then routes every allocation
//! through the counter below.
//!
//! The number asserted is a *slope* -- the difference between appending 1100
//! records and appending 100, divided by 1000 -- not a total. Everything that
//! happens once (opening the file, listing the segments, growing the buffer to
//! its working size) cancels, and what is left is the marginal cost of one
//! more record, which is the quantity the rule is about.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static ON: AtomicBool = AtomicBool::new(false);

struct Counting;

// `realloc` counts too: growing a `Vec` in place is still a trip through the
// allocator, and a per-record buffer that doubles is exactly the shape this
// test exists to catch.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if ON.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(l.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        if ON.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(n as u64, Ordering::Relaxed);
        }
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn counted<T>(f: impl FnOnce() -> T) -> (T, u64, u64) {
    ALLOCS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    ON.store(true, Ordering::Relaxed);
    let t = f();
    ON.store(false, Ordering::Relaxed);
    (t, ALLOCS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed))
}

use granular::persist::Wal;
use granular::types::{Block, Column, DataType};

fn block(n: i64) -> Block {
    Block::new(vec![
        Column::u64s(DataType::UInt64, (0..n as u64).collect()),
        Column::i64s(DataType::Int64, (0..n).map(|x| x * 7).collect()),
    ])
    .unwrap()
}

fn tmp(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "gr-alloc-{tag}-{}-{:?}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Allocations charged to one more record, for a workload that appends `n`.
fn slope(tag: &str, lo: u64, hi: u64, mut one: impl FnMut(&mut Wal, u64)) -> (f64, f64) {
    let run = |n: u64, one: &mut dyn FnMut(&mut Wal, u64)| -> (u64, u64) {
        let d = tmp(tag);
        let mut w = Wal::open(&d).unwrap();
        // Warm every lazy path -- and the framing buffer -- outside the count.
        one(&mut w, 0);
        let (_, a, b) = counted(|| {
            for i in 0..n {
                one(&mut w, i + 1);
            }
        });
        drop(w);
        std::fs::remove_dir_all(&d).ok();
        (a, b)
    };
    let (a1, b1) = run(lo, &mut one);
    let (a2, b2) = run(hi, &mut one);
    let d = (hi - lo) as f64;
    ((a2 - a1) as f64 / d, (b2 - b1) as f64 / d)
}

#[test]
fn appending_a_record_allocates_nothing() {
    let small = block(1);
    let big = block(512);

    let (ai, bi) = slope("ins", 100, 1100, |w, _| {
        w.append_insert(&small).unwrap();
    });
    // A 512-row block is ~8 kB of body: if anything on this path still sized a
    // buffer by the record, it would show here as thousands of bytes and not
    // as a rounding error.
    let (ag, bg) = slope("insbig", 50, 550, |w, _| {
        w.append_insert(&big).unwrap();
    });
    let (ad, bd) = slope("del", 100, 1100, |w, i| {
        w.append_delete(i).unwrap();
    });
    // A run record: one frame for the whole batch, and still no allocation.
    let lanes: Vec<u64> = (0..64).collect();
    let (ar, br) = slope("run", 100, 1100, |w, _| {
        w.append_deletes(&lanes).unwrap();
    });

    let report = format!(
        "insert(1 row) {ai:.3} allocs / {bi:.1} bytes\n\
         insert(512 rows) {ag:.3} allocs / {bg:.1} bytes\n\
         delete(1 lane) {ad:.3} allocs / {bd:.1} bytes\n\
         delete run(64 lanes) {ar:.3} allocs / {br:.1} bytes"
    );
    for (what, allocs, bytes) in
        [("insert", ai, bi), ("big insert", ag, bg), ("delete", ad, bd), ("delete run", ar, br)]
    {
        assert_eq!(allocs, 0.0, "{what} allocates per record\n{report}");
        assert_eq!(bytes, 0.0, "{what} allocates bytes per record\n{report}");
    }
}
