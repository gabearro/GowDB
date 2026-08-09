//! Generational fuzzing of every boundary the engine has to treat as hostile.
//!
//! Three things reach this engine from outside its own memory: SQL text typed
//! by a user, and part/catalog/WAL bytes read off a disk that can be rotted,
//! truncated or deliberately forged. Each has a hard invariant:
//!
//!   * **SQL** returns `Ok` or `Err`. It never panics, never recurses off the
//!     stack, never hangs and never sizes an allocation from the input.
//!   * **Part, `TABLE`, `CATALOG`, WAL frames** return `Ok` or
//!     `Error::Corruption`. Same three nevers, plus: a length field read out
//!     of the file must never become a `Vec` capacity before it has been
//!     checked against the bytes that are actually there.
//!   * **Anything the engine writes it reads back identically.**
//!
//! ## Why this file exists next to the `adversarial_*` unit tests
//!
//! Those tests pin *specific* bugs found by review, at fixed seeds and short
//! lengths. They are regression tests and they should stay that way. What they
//! cannot do is find the next bug: 576 fixed inputs at ≤512 bytes is a
//! vanishingly small slice of the space, and nothing in them grows.
//!
//! This file is the search half. It is a generational fuzzer -- generate,
//! evaluate, keep what is new -- written from scratch because the alternative
//! is not available: `cargo-fuzz`/`libFuzzer` needs a nightly toolchain and a
//! crates.io dependency, and this crate's `[dependencies]` is empty on purpose.
//!
//! ### Coverage without instrumentation
//!
//! A real fuzzer keeps an input when it lights up a new edge. With no
//! sanitizer coverage to read, this one uses the next best observable: the
//! *diagnostic* the engine produced. Error code plus the message with every
//! run of digits collapsed to `#` identifies the rejection site almost
//! exactly -- every `Err` in `persist::reader` carries a distinct string, and
//! "offset 41" versus "offset 93" is the same site, which is why the digits go.
//! Inputs with a fingerprint never seen before are written to `tests/corpus/`
//! under that fingerprint, so the file name *is* the novelty check and reruns
//! converge instead of churning.
//!
//! The corpus is replayed first on every run, so coverage accumulates across
//! runs rather than restarting cold, and a crash found on a lucky seed at 3am
//! stays found. It is bounded to [`CORPUS_FILES`] files of [`CORPUS_BYTES`]
//! bytes each per target so it stays committable.
//!
//! ### Detecting the failures a `#[test]` cannot catch by itself
//!
//! `assert!` catches wrong answers. The three interesting failure modes here
//! are not wrong answers:
//!
//!   * **Panic** -- caught with `catch_unwind` so the loop can report the
//!     exact seed rather than dying on case 40,000 with no context.
//!   * **Hang** -- a global progress counter and one watchdog thread. If no
//!     fuzz case anywhere advances for [`STALL_SECS`], the watchdog prints the
//!     target and seed that are stuck and exits the process non-zero. An
//!     infinite loop in a parser otherwise shows up as a CI job that times out
//!     with no idea which input did it.
//!   * **Unbounded allocation** -- a `#[global_allocator]` shim records the
//!     largest single allocation *request* made on the current thread.
//!     `Vec::with_capacity(bogus_varint)` is a request for 8 exabytes; the
//!     process dies before any assert runs, so nothing but an allocator hook
//!     can see it coming. Per-thread and request-sized (not
//!     outstanding-bytes) so that the other tests running concurrently in this
//!     binary cannot perturb the reading.
//!
//! ## Miri
//!
//! This crate has ~46 raw-pointer operations across 14 modules, and Miri is
//! the only checker for them that costs no dependency. It does not run from
//! `cargo test` -- it needs nightly and it is ~1000x slower -- so it is a
//! manual pass. Setup once, then one module at a time:
//!
//! ```text
//! cd "…/OLAP:OLTP database"
//! CARGO_TARGET_DIR=/tmp/gr-miri cargo +nightly miri setup
//! MIRIFLAGS="-Zmiri-disable-isolation" CARGO_TARGET_DIR=/tmp/gr-miri \
//!     cargo +nightly miri test --lib common::pool
//! ```
//!
//! `-Zmiri-disable-isolation` is required: the fixtures create temp files and
//! read the clock. Per module, one at a time, because a whole-crate run is
//! hours and one module's stop hides the rest.
//!
//! ### Results, run against `rustc 1.95.0-nightly` / `miri 0.1.0 (3daae5e42e)`
//!
//! | module | tests | verdict |
//! |--------|-------|---------|
//! | `common::pool`     | 63 | clean, 56s |
//! | `common::hash`     | 3  | clean, 4s |
//! | `encoding::bitpack`| 6  | clean, 172s |
//! | `encoding::dict`   | 8  | clean, 5s |
//! | `index::filter`    | 3  | clean, 82s |
//! | `storage::column`  | 11 | clean, 12s |
//! | `storage::delta`   | 17 | clean, 25s |
//! | `persist::format`  | 51 | clean, 186s |
//! | `sort`             | 4  | clean; see the note on the two large-N cases |
//! | `common::bitset`   | 5  | clean, 5s |
//! | `encoding::lz4`    | 2  | clean, sampled (`roundtrip_every_short_length`, `roundtrip_all_identical_run`) |
//! | `storage::part`    | 2  | clean, sampled (`point_lookups_hit_every_key` alone is 839s) |
//! | `persist::mmap`    | -- | **stops at the syscall**, see below |
//!
//! `encoding::lz4`, `storage::part` and `persist::reader` are sampled rather
//! than run whole: their fixtures build megabyte buffers and multi-granule
//! parts, and Miri's interpreter makes each of those a multi-hour run. The
//! `unsafe` in all three is the same shape as what the clean rows above cover
//! (`get_unchecked` behind a proved bound, a `&[u64]` cast over an aligned
//! buffer), so the sampling is over inputs, not over unsafe blocks.
//! `storage::part`'s sample is the important one: `point_lookups_hit_every_key`
//! drives the CHD minimal perfect hash through `Granule`'s packed-word reads
//! for every key in a multi-granule part, built by the thread pool, which is
//! the raw-pointer path with the most surface in the crate.
//!
//! `sort`'s two remaining tests (`matches_std_sort`,
//! `soa_matches_the_pair_form_and_is_stable`) both loop to n = 100,000 and did
//! not finish inside a 40-minute Miri budget. That is a budget result, not a
//! verdict -- and it costs nothing, because the `unsafe` they would exercise
//! is the `get_unchecked_mut(counts[d])` scatter, and
//! `is_stable_on_duplicate_keys` reaches it: n = 1,000 is above the
//! 128-element cutover to the radix path, it is all duplicate keys (so every
//! bucket is deep), and it is clean.
//!
//! No Undefined Behaviour, no leak, no data race reported anywhere. Two
//! results are worth calling out because they were the ones most likely to be
//! real problems:
//!
//!   * **`common::pool`'s `mem::transmute::<&Job<'_>, &'static Job<'static>>`**
//!     is clean under Stacked Borrows across all 63 tests, including the
//!     nested-pool, panic-in-`f` and drop-panic cases. Miri's concurrency
//!     checking is not a proof for *all* interleavings, only the ones it
//!     schedules, but the mutex discipline the safety comment describes
//!     (removed from `state.jobs` under the lock, then `active` waited to zero
//!     under the same lock) is exactly what makes the laundering sound, and
//!     nothing contradicted it.
//!   * **`PackedU64`'s raw word reads** (`storage::column`, `encoding::bitpack`)
//!     are clean, including the `get_unchecked` paths and the mapped backing
//!     exercised through `Mmap::slurp`.
//!
//! ### The one module Miri cannot run as written
//!
//! `persist::mmap` stops with:
//!
//! ```text
//! error: unsupported operation: Miri does not support file-backed memory mappings
//!    --> src/persist/mmap.rs:174:13
//! ```
//!
//! The module *already contains* the heap-backed stand-in the check needs --
//! `Mmap::slurp`, compiled under `#[cfg(any(not(unix), test))]` -- it is just
//! not reachable, because on unix `Mmap::load` is `#[cfg(unix)]` and calls the
//! syscall. Widening two gates makes the whole module checkable with no
//! behaviour change off Miri:
//!
//! ```text
//! -    #[cfg(unix)]                            fn load(…)   // real mmap
//! +    #[cfg(all(unix, not(miri)))]            fn load(…)
//! -    #[cfg(not(unix))]                       fn load(…)   // slurp
//! +    #[cfg(any(not(unix), miri))]            fn load(…)
//! ```
//!
//! That edit is in `src/persist/mmap.rs`, which this file does not own; it is
//! reported rather than made. What *can* be checked today already is, and it
//! is clean: `persist::mmap::tests::adv_miri_slurp_lifecycle` (the raw-pointer
//! lifecycle, `from_raw_parts`, and a move across threads over a `slurp`ed
//! buffer), `adv_zero_length_never_reaches_slurp`, `fallback_buffer_is_word_aligned`
//! and the three `empty_*` tests all pass under Miri unmodified.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fmt::Write as _;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Once;

use granular::common::{hash_bytes, splitmix64, GRANULE_SIZE};
use granular::persist::{format, reader, writer, Wal, WalRecord};
use granular::storage::Part;
use granular::types::{
    Block, Column, ColumnBuilder, DataType, Engine, Field, Schema, TableDef, Value,
};
use granular::{Error, Session};

// ---------------------------------------------------------------------------
// allocation watchdog
// ---------------------------------------------------------------------------

/// Wraps the system allocator to record the largest single allocation
/// *request* the current thread has made.
///
/// Request size, not outstanding bytes, and per-thread, not global: the
/// failure being hunted is one `Vec::with_capacity(n)` where `n` came from a
/// corrupt length field, and the thread that decodes the bytes is the thread
/// that asks for the memory. Both choices make the reading immune to the other
/// tests running concurrently in this binary, which an outstanding-bytes gauge
/// would not be.
struct Watch;

thread_local! {
    /// Reset by [`peak_request`], read after. `const` init and no destructor,
    /// so `try_with` cannot allocate and cannot fail except during TLS
    /// teardown -- either would be a re-entrant allocation, i.e. a hang.
    static MAX_REQ: Cell<usize> = const { Cell::new(0) };
}

#[inline]
fn note(n: usize) {
    let _ = MAX_REQ.try_with(|c| {
        if n > c.get() {
            c.set(n);
        }
    });
}

unsafe impl GlobalAlloc for Watch {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        note(l.size());
        System.alloc(l)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        note(l.size());
        System.alloc_zeroed(l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        note(new);
        System.realloc(p, l, new)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
}

#[global_allocator]
static ALLOC: Watch = Watch;

/// Run `f`, returning its value and the largest single allocation it requested
/// on this thread. Blind to allocations made on threads `f` spawns -- every
/// decoder measured here is single-threaded, and `Part::build` (which is not)
/// is deliberately never measured.
fn peak_request<R>(f: impl FnOnce() -> R) -> (R, usize) {
    MAX_REQ.with(|c| c.set(0));
    let r = f();
    (r, MAX_REQ.with(|c| c.get()))
}

// ---------------------------------------------------------------------------
// hang watchdog
// ---------------------------------------------------------------------------

/// Seconds with no fuzz case anywhere completing before the watchdog declares
/// a hang. Generous: this machine's timings swing 3x under load, and a false
/// positive here kills the whole test binary.
const STALL_SECS: u64 = 180;

static PROGRESS: AtomicU64 = AtomicU64::new(0);
/// Index into [`TARGETS`] of the case in flight, and the seed that produced
/// it -- enough to reproduce without storing (and allocating) a label.
static CUR_TARGET: AtomicUsize = AtomicUsize::new(0);
static CUR_SEED: AtomicU64 = AtomicU64::new(0);

const TARGETS: [&str; 6] = ["sql", "part", "doc", "wal", "block", "roundtrip"];

/// Mark the start of one case. Two relaxed stores and an increment: cheap
/// enough to sit in the innermost fuzz loop, which is the only place it is
/// useful.
#[inline]
fn tick(target: usize, seed: u64) {
    CUR_TARGET.store(target, Ordering::Relaxed);
    CUR_SEED.store(seed, Ordering::Relaxed);
    PROGRESS.fetch_add(1, Ordering::Relaxed);
}

/// Start the watchdog once per process.
///
/// It exits rather than panicking: a panic on a detached thread is swallowed
/// by the test harness, and the point is to fail the run loudly with the seed
/// still on screen.
fn arm_watchdog() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        std::thread::Builder::new()
            .name("fuzz-watchdog".into())
            .spawn(|| {
                let mut last = PROGRESS.load(Ordering::Relaxed);
                let mut idle = 0u64;
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    let now = PROGRESS.load(Ordering::Relaxed);
                    if now != last {
                        last = now;
                        idle = 0;
                        continue;
                    }
                    idle += 1;
                    if idle >= STALL_SECS {
                        let t = TARGETS
                            .get(CUR_TARGET.load(Ordering::Relaxed))
                            .copied()
                            .unwrap_or("?");
                        eprintln!(
                            "\nFUZZ HANG: no case completed for {STALL_SECS}s.\n  \
                             target = {t}, seed = 0x{:016x}\n  \
                             reproduce with GRANULAR_FUZZ_SEED=0x{:016x}\n",
                            CUR_SEED.load(Ordering::Relaxed),
                            CUR_SEED.load(Ordering::Relaxed),
                        );
                        std::process::exit(101);
                    }
                }
            })
            .expect("watchdog thread");
    });
}

// ---------------------------------------------------------------------------
// budget and rng
// ---------------------------------------------------------------------------

/// Cases per target.
///
/// `cargo test` must stay fast, so the defaults total ~27,000 cases and run in
/// under three seconds. The soak is
///
/// ```text
/// GRANULAR_FUZZ_CASES=100000 cargo test --release --test robustness
/// ```
///
/// which is 100,000 cases on each of the four byte-level fuzzers plus the
/// capped heavy loops, and takes 51s (measured). Everything is seeded from
/// `GRANULAR_FUZZ_SEED`, so a soak that finds something reproduces from the
/// seed the failure prints.
fn cases(default: usize) -> usize {
    if cfg!(miri) {
        // Miri is ~1000x slower and is not what this file is for; keep the
        // loops alive (so nothing is `#[ignore]`d) but token-sized.
        return 2;
    }
    match std::env::var("GRANULAR_FUZZ_CASES") {
        Ok(v) => v.trim().parse().unwrap_or(default),
        Err(_) => default,
    }
}

/// Case count for a loop whose per-case cost is *milliseconds* rather than
/// microseconds -- a part build, an `fsync`, a row-by-row comparison of a
/// 2,500-row block.
///
/// `GRANULAR_FUZZ_CASES` still raises these, but only to `ceiling`. Measured
/// the hard way on the first soak of this file: at 250,000 the three heavy
/// loops ran for over an hour while the four byte-level fuzzers -- drawing
/// from the same generators, over the same damage operators -- finished all
/// 250,000 cases each in minutes. Coverage per second is two orders of
/// magnitude better in the light loops, so the heavy ones are capped and the
/// budget goes where it buys something.
fn heavy_cases(default: usize, ceiling: usize) -> usize {
    cases(default).min(ceiling)
}

/// Base seed. Fixed by default so a failure is reproducible and a green run
/// means the same thing twice; override to explore new ground.
fn base_seed(salt: u64) -> u64 {
    let s = std::env::var("GRANULAR_FUZZ_SEED")
        .ok()
        .and_then(|v| {
            let t = v.trim();
            match t.strip_prefix("0x") {
                Some(h) => u64::from_str_radix(h, 16).ok(),
                None => t.parse().ok(),
            }
        })
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    splitmix64(s ^ salt)
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    #[inline]
    fn next(&mut self) -> u64 {
        self.0 = splitmix64(self.0);
        self.0
    }
    /// Uniform-enough in `0..n`. Multiply-shift, not modulo: the bias is below
    /// 2^-64 relative and it costs one `mulhi`.
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        ((self.next() as u128 * n as u128) >> 64) as usize
    }
    #[inline]
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
    /// `pick` for string tables. Separate because `pick` hands back `&&str`
    /// and every caller here wants the `&str`.
    #[inline]
    fn pick_str<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[self.below(xs.len())]
    }
    #[inline]
    fn chance(&mut self, one_in: usize) -> bool {
        self.below(one_in) == 0
    }
}

// ---------------------------------------------------------------------------
// behaviour fingerprints
// ---------------------------------------------------------------------------

/// Two reusable buffers so fingerprinting a case allocates nothing after the
/// first one. Threaded through the loops by hand rather than hidden in a
/// thread-local, because the loops already own scratch and one more field is
/// cheaper than a TLS lookup per case.
#[derive(Default)]
struct Fp {
    raw: String,
    norm: String,
}

impl Fp {
    /// Fingerprint one outcome: the target tag, the error code, and the
    /// message with digit runs collapsed. See the module docs -- this is the
    /// coverage signal the corpus is keyed by.
    fn of(&mut self, tag: &str, err: Option<&Error>) -> u64 {
        self.norm.clear();
        self.norm.push_str(tag);
        match err {
            None => self.norm.push_str(":ok"),
            Some(e) => {
                self.raw.clear();
                let _ = write!(self.raw, "{e}");
                self.norm.push(':');
                self.norm.push_str(e.code());
                self.norm.push(':');
                let mut digits = false;
                for c in self.raw.chars() {
                    if c.is_ascii_digit() {
                        if !digits {
                            self.norm.push('#');
                            digits = true;
                        }
                    } else {
                        digits = false;
                        self.norm.push(c);
                    }
                }
            }
        }
        hash_bytes(self.norm.as_bytes(), 0x00C0_FFEE_D15E_A5E5)
    }
}

// ---------------------------------------------------------------------------
// corpus
// ---------------------------------------------------------------------------

/// Per-target file ceiling. The corpus is meant to be committed, so it is
/// capped by count and by size rather than allowed to grow to whatever the
/// fuzzer finds interesting.
const CORPUS_FILES: usize = 64;
/// Measured: a healthy 1-row part image is 407 bytes and a `CATALOG` for three
/// databases is 580, so 8 KiB keeps every document target's damaged images and
/// the small-part ones. Multi-granule part images (~40 KiB at 2,100 rows) are
/// deliberately over the line: they are regenerated from the fixture on every
/// run anyway, and persisting them would put a megabyte in the repository to
/// re-test what the fixture already re-tests.
const CORPUS_BYTES: usize = 8 * 1024;

struct Corpus {
    dir: PathBuf,
    ext: &'static str,
    /// The signals already on disk, sorted. `offer` is called once per fuzz
    /// case -- a quarter of a million times in a soak -- and the overwhelming
    /// majority of those calls rediscover a signal that is already here. A
    /// sorted `Vec` of at most [`CORPUS_FILES`] `u64`s answers that in a
    /// binary search over one cache line's worth of data; the `path.exists()`
    /// this replaces was a `stat(2)` syscall per case.
    seen: std::sync::Mutex<Vec<u64>>,
    writable: bool,
}

impl Corpus {
    fn open(kind: &str, ext: &'static str) -> Corpus {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus").join(kind);
        let writable = std::env::var_os("GRANULAR_FUZZ_NO_WRITE").is_none()
            && std::fs::create_dir_all(&dir).is_ok();
        // A hand-added file whose name is not a signal still counts against
        // the cap (it is one more file in the directory) but can never be
        // matched by a generated signal, which is the behaviour you want: it
        // is pinned input, not a claimed slot.
        let mut seen: Vec<u64> = Vec::new();
        let mut extra = 0usize;
        if let Ok(d) = std::fs::read_dir(&dir) {
            for e in d.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                match name.split('.').next().and_then(|h| u64::from_str_radix(h, 16).ok()) {
                    Some(sig) => seen.push(sig),
                    None => extra += 1,
                }
            }
        }
        seen.sort_unstable();
        // Occupied slots that carry no signal are represented by pushing the
        // count out of range rather than by a second counter.
        for i in 0..extra {
            seen.push(u64::MAX - i as u64);
        }
        seen.sort_unstable();
        seen.dedup();
        Corpus { dir, ext, seen: std::sync::Mutex::new(seen), writable }
    }

    /// Everything already on disk, sorted so a failure reproduces in the same
    /// order twice.
    fn load(&self) -> Vec<(String, Vec<u8>)> {
        let mut out: Vec<(String, Vec<u8>)> = match std::fs::read_dir(&self.dir) {
            Ok(d) => d
                .flatten()
                .filter(|e| e.path().is_file())
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().into_owned();
                    // Cap what a replay will read: a corpus file that grew
                    // beyond the budget (by hand, or by an older build) must
                    // not turn every run into a slow one.
                    let b = std::fs::read(e.path()).ok()?;
                    (b.len() <= CORPUS_BYTES * 4).then_some((name, b))
                })
                .collect(),
            Err(_) => Vec::new(),
        };
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Keep `bytes` if `signal` is a behaviour this corpus has not recorded.
    ///
    /// Content-addressed by the *signal*, not by the bytes: the file name is
    /// the novelty check, so a rerun that rediscovers the same rejection site
    /// with a different input is a no-op instead of a new file. `create_new`
    /// makes the check and the claim one atomic step, which matters because
    /// several fuzz tests run concurrently.
    fn offer(&self, signal: u64, bytes: &[u8]) {
        if !self.writable || bytes.len() > CORPUS_BYTES {
            return;
        }
        {
            let mut seen = self.seen.lock().expect("corpus lock");
            if seen.len() >= CORPUS_FILES {
                return;
            }
            match seen.binary_search(&signal) {
                Ok(_) => return,
                Err(at) => seen.insert(at, signal),
            }
        }
        // Written outside the lock, and with `create_new` so the check and the
        // claim are one atomic step against a *concurrent process* -- another
        // `cargo test` in the same checkout -- which the in-memory set cannot
        // see. A lost race just means the file is already there.
        use std::io::Write;
        let _ = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.dir.join(format!("{signal:016x}{}", self.ext)))
            .and_then(|mut f| f.write_all(bytes));
    }
}

// ---------------------------------------------------------------------------
// scratch directories
// ---------------------------------------------------------------------------

/// Named from the pid and a counter rather than randomness, matching
/// `persist::testkit`: a rerun cannot collide with a live process and a name
/// is reproducible while debugging.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        static N: AtomicUsize = AtomicUsize::new(0);
        let d = std::env::temp_dir().join(format!(
            "granular-fuzz-{}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed),
            tag
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("scratch dir");
        Scratch(d)
    }
    fn join(&self, s: &str) -> PathBuf {
        self.0.join(s)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---------------------------------------------------------------------------
// panic capture
// ---------------------------------------------------------------------------

thread_local! {
    /// Set only while a fuzz case is inside [`no_panic`]. The hook installed
    /// by [`Quiet`] drops panic output for threads that have it set and
    /// forwards everything else, so a caught fuzz panic costs no output while
    /// a real assertion failure -- here or in a test running concurrently in
    /// the same binary -- still prints normally.
    static EXPECTING: Cell<bool> = const { Cell::new(false) };
}

/// Arms the hook described above, once per process.
///
/// The hook is process-global, which is why it discriminates by thread instead
/// of being a blanket silencer: with tens of thousands of caught panics a
/// blanket hook is the only way to keep the output readable, and a blanket
/// hook is also how you lose the one failure you cared about.
///
/// The value is inert and has no `Drop`. A fuzz loop binds it for its scope so
/// the code reads as if the hook were scoped, but installing and restoring per
/// test would race the other tests running in this binary -- `set_hook` is
/// global state and the last restore would win.
struct Quiet;

impl Quiet {
    fn on() -> Quiet {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                if EXPECTING.try_with(|c| c.get()).unwrap_or(false) {
                    return;
                }
                prev(info);
            }));
        });
        Quiet
    }
}

/// Run `f`, converting a panic into `Err(message)`.
fn no_panic<R>(f: impl FnOnce() -> R) -> std::result::Result<R, String> {
    let _ = EXPECTING.try_with(|c| c.set(true));
    let r = catch_unwind(AssertUnwindSafe(f));
    let _ = EXPECTING.try_with(|c| c.set(false));
    r.map_err(|p| {
        p.downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| p.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "<non-string panic payload>".into())
    })
}

/// Stack for every fuzz-loop thread, and the reason there is one.
///
/// `libtest` runs each `#[test]` on a `std::thread` with the default 2 MiB
/// stack. That is not enough for this file, because `sql::parse` returns an
/// `Expr` tree whose depth is unbounded for left-associative operator chains
/// and whose `Drop` is recursive -- see
/// [`sql_ast_drop_recurses_once_per_operator`], which measures exactly where
/// it dies. Until that is fixed in the parser, the generators here can
/// manufacture a chain long enough to abort the process, and a fuzz harness
/// that aborts on its own inputs is useless. At the measured ~120 bytes per
/// `SetExpr` level, 64 MiB buys ~550,000 levels; the generators here are
/// capped at 64 KiB of text, i.e. at most ~16,000 operators, so the margin is
/// 30x and does not depend on the optimization level the way the cliff does.
const FUZZ_STACK: usize = 64 * 1024 * 1024;

/// Run a test body on a thread with [`FUZZ_STACK`], re-raising its panic in
/// the caller so `libtest` still sees a normal failure.
fn on_big_stack(f: impl FnOnce() + Send + 'static) {
    let h = std::thread::Builder::new()
        .stack_size(FUZZ_STACK)
        .name("fuzz".into())
        .spawn(f)
        .expect("fuzz thread");
    if let Err(p) = h.join() {
        std::panic::resume_unwind(p);
    }
}

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

fn fixture_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::UInt64),
        Field::new("host", DataType::String),
        Field::new("ms", DataType::Nullable(Box::new(DataType::Int64))),
        Field::new("ratio", DataType::Float64),
    ])
    .unwrap()
}

fn fixture_def(name: &str) -> TableDef {
    TableDef {
        name: name.into(),
        schema: fixture_schema(),
        order_by: vec![0],
        primary_key: vec![0],
        partition_by: None,
        engine: Engine::MergeTree,
    }
}

/// `n` rows matching [`fixture_schema`]: unique ascending keys, a
/// low-cardinality string, a nullable column with holes, and a float. All four
/// physical kinds plus a null mask, which is what makes a mutated part reach
/// every decoder branch.
fn fixture_block(n: usize) -> Block {
    let mut keys: Vec<u64> = (0..n as u64 * 3).map(|i| splitmix64(i) % 4_000_000).collect();
    keys.sort_unstable();
    keys.dedup();
    assert!(keys.len() >= n, "not enough distinct keys for {n} rows");
    keys.truncate(n);
    let hosts: Vec<std::sync::Arc<str>> =
        keys.iter().map(|k| format!("host-{:02}", k % 37).into()).collect();
    let mut ms = ColumnBuilder::new(DataType::Nullable(Box::new(DataType::Int64)));
    for (i, &k) in keys.iter().enumerate() {
        if i % 7 == 0 {
            ms.push_null();
        } else {
            ms.push_value(&Value::Int((k % 10_000) as i64 - 5_000)).unwrap();
        }
    }
    let ratio: Vec<f64> = keys.iter().map(|&k| (k % 1000) as f64 / 8.0 - 60.0).collect();
    Block::new(vec![
        Column::u64s(DataType::UInt64, keys),
        Column::strs(DataType::String, hosts),
        ms.finish(),
        Column::f64s(DataType::Float64, ratio),
    ])
    .unwrap()
}

fn fixture_part(n: usize) -> Part {
    Part::build(&fixture_block(n), Some(0), Some(0)).unwrap()
}

/// Every `(position, row)` of a part, for exact comparison across a round
/// trip. Mirrors `persist::testkit::dump`, which integration tests cannot see.
fn dump(p: &Part) -> Vec<Vec<Value>> {
    let mut out = Vec::with_capacity(p.n_rows);
    for (gi, g) in p.granules.iter().enumerate() {
        // Row positions are `granule << G_SHIFT | offset`, not a flat count:
        // a granule short of `GRANULE_SIZE` leaves a hole in the numbering.
        let base = gi * GRANULE_SIZE;
        for i in 0..g.len {
            out.push((0..p.ncols).map(|c| p.value_at(base + i, c)).collect());
        }
    }
    out
}

fn is_corrupt(e: &Error) -> bool {
    matches!(e, Error::Corruption(_) | Error::Io(_))
}

// ===========================================================================
// target 1: the SQL front end
// ===========================================================================

/// Hand-written inputs that already broke something, or that sit on a boundary
/// worth never losing. Written into the corpus on first run so a fresh
/// checkout starts warm rather than cold.
const SQL_SEEDS: &[&str] = &[
    "",
    ";",
    ";;;;;;;;",
    "SELECT",
    "SELECT 1",
    "SELECT * FROM t WHERE a IN (SELECT b FROM u WHERE c IN (SELECT d FROM v))",
    "SELECT CASE WHEN a THEN CASE WHEN b THEN 1 ELSE 2 END ELSE 3 END FROM t",
    "SELECT a FROM t JOIN u USING (a) JOIN v USING (a) JOIN w USING (a)",
    "SELECT -----------1",
    "SELECT NOT NOT NOT NOT NOT NOT NOT NOT 1",
    "SELECT 1e",
    "SELECT 1e999999999999999999999",
    "SELECT 0x",
    "SELECT '",
    "SELECT \"",
    "SELECT `",
    "SELECT /*",
    "SELECT -- \n 1",
    "SELECT 1 /*/ 2",
    "INSERT INTO t VALUES ()",
    "INSERT INTO t VALUES (,)",
    "CREATE TABLE t (a UInt64) ENGINE = MergeTree ORDER BY a",
    "CREATE TABLE t (a Nullable(Nullable(Nullable(UInt64)))) ENGINE = Memory",
    "CREATE TABLE t (a FixedString(4294967295)) ENGINE = Memory",
    "SELECT toDate(9223372036854775807)",
    "SELECT count(*) FROM t GROUP BY a WITH TOTALS ORDER BY 1 LIMIT 18446744073709551615",
    "WITH x AS (SELECT 1) SELECT * FROM x",
    "SELECT INTERVAL 1 DAY + INTERVAL 2 MONTH",
    "SELECT a::UInt64::String::Float64 FROM t",
    "SELECT * FROM t UNION ALL SELECT * FROM t UNION ALL SELECT * FROM t",
    "SELECT [1,2,3][1]",
    "SELECT {1:2}",
    "\u{feff}SELECT 1",
    "SELECT '\u{0}\u{1}\u{2}'",
    "SELECT 'ünïcödé' = '𝔘𝔫𝔦'",
];

/// Fragments the token-soup generator draws from. Deliberately biased towards
/// the characters that change the *shape* of a parse -- brackets, quotes,
/// comment openers -- rather than uniform over bytes, which mostly produces
/// inputs the lexer rejects at position 0.
const SQL_ATOMS: &[&str] = &[
    " ", "(", ")", ",", ";", "*", "'", "\"", "`", "--", "/*", "*/", "::", ".", "-", "+", "/", "%",
    "=", "<", ">", "<=", ">=", "!=", "<>", "|", "&", "^", "~", "?", "@", "$", "\\", "[", "]", "{",
    "}", "\n", "\t", "\0", "SELECT", "FROM", "WHERE", "GROUP", "BY", "ORDER", "LIMIT", "OFFSET",
    "JOIN", "LEFT", "INNER", "ON", "USING", "AS", "AND", "OR", "NOT", "NULL", "CASE", "WHEN",
    "THEN", "ELSE", "END", "IN", "BETWEEN", "LIKE", "IS", "DISTINCT", "UNION", "ALL", "WITH",
    "INSERT", "INTO", "VALUES", "CREATE", "TABLE", "DROP", "ALTER", "ENGINE", "MergeTree",
    "Nullable", "UInt64", "String", "count", "sum", "toDate", "1", "0", "-1", "1.5", "1e9",
    "0xFF", "'s'", "t", "a", "b", "x_1", "\u{e9}", "\u{4e2d}", "\u{1f600}",
];

/// Random tokens with no grammar at all. Cheap, and it is what finds the
/// lexer's edges (unterminated literals, comment nesting, stray bytes).
fn gen_soup(rng: &mut Rng, out: &mut String) {
    out.clear();
    let n = 1 + rng.below(60);
    for _ in 0..n {
        out.push_str(rng.pick_str(SQL_ATOMS));
    }
}

/// A grammar-directed generator: valid-ish SQL, so the *parser* is exercised
/// rather than the lexer. `budget` bounds both depth and total size; it is
/// decremented on the way down so a runaway expansion is impossible.
fn gen_expr(rng: &mut Rng, out: &mut String, budget: &mut usize) {
    if *budget == 0 {
        out.push('1');
        return;
    }
    *budget -= 1;
    match rng.below(12) {
        0 => out.push_str(rng.pick_str(&["a", "b", "t.a", "u.b", "\"odd name\"", "*"])),
        1 => out.push_str(rng.pick_str(&["1", "-1", "0", "1.5", "'s'", "NULL", "TRUE", "0xFF"])),
        2 => {
            out.push('(');
            gen_expr(rng, out, budget);
            out.push(')');
        }
        3 => {
            gen_expr(rng, out, budget);
            out.push(' ');
            out.push_str(rng.pick_str(&["+", "-", "*", "/", "%", "=", "<", ">", "AND", "OR", "LIKE"]));
            out.push(' ');
            gen_expr(rng, out, budget);
        }
        4 => {
            out.push_str(rng.pick_str(&["NOT ", "-", "+", "~"]));
            gen_expr(rng, out, budget);
        }
        5 => {
            out.push_str(rng.pick_str(&["count", "sum", "max", "toDate", "substring", "if"]));
            out.push('(');
            let args = rng.below(4);
            for i in 0..args {
                if i > 0 {
                    out.push_str(", ");
                }
                gen_expr(rng, out, budget);
            }
            out.push(')');
        }
        6 => {
            out.push_str("CASE WHEN ");
            gen_expr(rng, out, budget);
            out.push_str(" THEN ");
            gen_expr(rng, out, budget);
            out.push_str(" ELSE ");
            gen_expr(rng, out, budget);
            out.push_str(" END");
        }
        7 => {
            gen_expr(rng, out, budget);
            out.push_str(" IN (");
            gen_select(rng, out, budget);
            out.push(')');
        }
        8 => {
            gen_expr(rng, out, budget);
            out.push_str(rng.pick_str(&[" IS NULL", " IS NOT NULL", " IS NOT DISTINCT FROM 1"]));
        }
        9 => {
            gen_expr(rng, out, budget);
            out.push_str("::");
            out.push_str(rng.pick_str(&["UInt64", "String", "Float64", "Nullable(Int8)", "Date"]));
        }
        10 => {
            gen_expr(rng, out, budget);
            out.push_str(" BETWEEN ");
            gen_expr(rng, out, budget);
            out.push_str(" AND ");
            gen_expr(rng, out, budget);
        }
        _ => {
            out.push_str("(SELECT ");
            gen_expr(rng, out, budget);
            out.push(')');
        }
    }
}

fn gen_select(rng: &mut Rng, out: &mut String, budget: &mut usize) {
    if *budget == 0 {
        out.push_str("SELECT 1");
        return;
    }
    *budget -= 1;
    out.push_str("SELECT ");
    if rng.chance(6) {
        out.push_str("DISTINCT ");
    }
    let cols = 1 + rng.below(4);
    for i in 0..cols {
        if i > 0 {
            out.push_str(", ");
        }
        gen_expr(rng, out, budget);
        if rng.chance(4) {
            out.push_str(" AS c");
            let _ = write!(out, "{i}");
        }
    }
    if rng.chance(2) {
        out.push_str(" FROM ");
        match rng.below(4) {
            0 => out.push('t'),
            1 => {
                out.push('(');
                gen_select(rng, out, budget);
                out.push_str(") sub");
            }
            2 => {
                out.push_str("t ");
                out.push_str(rng.pick_str(&["JOIN", "LEFT JOIN", "INNER JOIN", "CROSS JOIN"]));
                out.push_str(" u ");
                out.push_str(rng.pick_str(&["ON t.a = u.a", "USING (a)", ""]));
            }
            _ => out.push_str("t, u, v"),
        }
    }
    for (chance, kw) in
        [(2usize, " WHERE "), (3, " GROUP BY "), (4, " HAVING "), (3, " ORDER BY ")]
    {
        if rng.chance(chance) {
            out.push_str(kw);
            gen_expr(rng, out, budget);
        }
    }
    if rng.chance(3) {
        out.push_str(" LIMIT ");
        out.push_str(rng.pick_str(&["1", "0", "18446744073709551615", "-1", "1e9"]));
    }
    if rng.chance(5) {
        out.push_str(rng.pick_str(&[" UNION ALL ", " UNION ", " UNION DISTINCT "]));
        gen_select(rng, out, budget);
    }
}

fn gen_sql(rng: &mut Rng, out: &mut String) {
    out.clear();
    let mut budget = 4 + rng.below(40);
    match rng.below(8) {
        0 => {
            out.push_str("CREATE TABLE t (");
            let n = 1 + rng.below(6);
            for i in 0..n {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "c{i} ");
                out.push_str(rng.pick_str(&[
                    "UInt64",
                    "Int8",
                    "Float64",
                    "String",
                    "Bool",
                    "Date",
                    "DateTime",
                    "FixedString(8)",
                    "Nullable(Int64)",
                    "LowCardinality(String)",
                ]));
                if rng.chance(4) {
                    out.push_str(" DEFAULT ");
                    gen_expr(rng, out, &mut budget);
                }
            }
            out.push_str(") ENGINE = ");
            out.push_str(rng.pick_str(&["MergeTree ORDER BY c0", "Memory", "Log", "Nonsense"]));
        }
        1 => {
            out.push_str("INSERT INTO t VALUES ");
            let rows = 1 + rng.below(5);
            for r in 0..rows {
                if r > 0 {
                    out.push_str(", ");
                }
                out.push('(');
                let n = 1 + rng.below(4);
                for c in 0..n {
                    if c > 0 {
                        out.push_str(", ");
                    }
                    gen_expr(rng, out, &mut budget);
                }
                out.push(')');
            }
        }
        2 => {
            out.push_str(rng.pick_str(&[
                "DROP TABLE t",
                "DROP TABLE IF EXISTS t",
                "ALTER TABLE t ADD COLUMN z UInt8",
                "USE db",
                "SHOW TABLES",
                "DESCRIBE t",
                "EXPLAIN ",
                "OPTIMIZE TABLE t",
            ]));
            if out.ends_with("EXPLAIN ") {
                gen_select(rng, out, &mut budget);
            }
        }
        3 => {
            out.push_str("WITH cte AS (");
            gen_select(rng, out, &mut budget);
            out.push_str(") ");
            gen_select(rng, out, &mut budget);
        }
        _ => gen_select(rng, out, &mut budget),
    }
    // A script, not just a statement: the `;` splitting loop in `parse` is its
    // own state machine and deserves the traffic.
    if rng.chance(5) {
        out.push_str("; ");
        gen_select(rng, out, &mut budget);
    }
}

/// Byte-level mutation of an existing input. This is the half that finds
/// things the grammar generator never will: a valid query with one byte wrong
/// gets deep into the parser before it fails.
fn mutate(rng: &mut Rng, seed: &str, other: &str, out: &mut String) {
    out.clear();
    out.push_str(seed);
    let rounds = 1 + rng.below(4);
    for _ in 0..rounds {
        // Everything below indexes by char boundary: `String` must stay UTF-8
        // for `parse(&str)`, and slicing mid-codepoint would panic in the
        // fuzzer rather than in the engine.
        let bounds: Vec<usize> = out.char_indices().map(|(i, _)| i).chain([out.len()]).collect();
        if bounds.len() < 2 {
            out.push_str(rng.pick_str(SQL_ATOMS));
            continue;
        }
        let a = bounds[rng.below(bounds.len())];
        let b = bounds[rng.below(bounds.len())];
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        match rng.below(6) {
            // Delete a span.
            0 => {
                out.replace_range(lo..hi, "");
            }
            // Duplicate a span in place -- the cheapest way to manufacture
            // deep nesting out of a shallow input.
            1 => {
                let dup = out[lo..hi].to_string();
                out.insert_str(hi, &dup);
            }
            // Splice in a slice of a different input.
            2 => {
                let ob: Vec<usize> =
                    other.char_indices().map(|(i, _)| i).chain([other.len()]).collect();
                let x = ob[rng.below(ob.len())];
                let y = ob[rng.below(ob.len())];
                let (p, q) = if x <= y { (x, y) } else { (y, x) };
                out.insert_str(lo, &other[p..q]);
            }
            // Insert a token.
            3 => out.insert_str(lo, rng.pick_str(SQL_ATOMS)),
            // Replace a span with a token.
            4 => out.replace_range(lo..hi, rng.pick_str(SQL_ATOMS)),
            // Repeat a token many times: length attacks without needing a
            // separate generator.
            _ => {
                let tok = rng.pick_str(SQL_ATOMS);
                let times = 1 << rng.below(9);
                out.reserve(tok.len() * times);
                for _ in 0..times {
                    out.insert_str(lo, tok);
                }
            }
        }
        // Length ceiling: a mutation chain that doubles every round would
        // otherwise reach hundreds of megabytes and measure the allocator
        // rather than the parser.
        if out.len() > 64 * 1024 {
            out.truncate(bounds[bounds.len() / 2]);
        }
    }
}

/// One parse, with every invariant checked. Returns the fingerprint so the
/// caller can decide whether the input is corpus-worthy.
fn check_parse(sql: &str, fp: &mut Fp) -> u64 {
    let (res, peak) = peak_request(|| no_panic(|| granular::sql::parse(sql)));
    match res {
        Err(msg) => panic!("PANIC in sql::parse: {msg}\n  input ({} bytes): {sql:?}", sql.len()),
        Ok(Ok(_)) => {
            // An allocation ceiling relative to the input, not absolute: an
            // AST is a few pointers per token and a token is at least one
            // byte, so a request two orders of magnitude past the input is a
            // length field being trusted, not a parse.
            let ceiling = (sql.len().max(4096)) * 256;
            assert!(
                peak <= ceiling,
                "sql::parse asked for {peak} bytes on a {}-byte input (ceiling {ceiling}): {sql:?}",
                sql.len()
            );
            fp.of("sql", None)
        }
        Ok(Err(e)) => {
            // A text parser has no files and no checksums. Corruption or Io
            // out of here would mean a disk error path was reached from a
            // string, which is a real finding, not a style nit.
            assert!(
                matches!(e, Error::Parse { .. } | Error::Unsupported(_)),
                "sql::parse returned {} ({e}) for {sql:?}",
                e.code()
            );
            fp.of("sql", Some(&e))
        }
    }
}

#[test]
fn fuzz_sql_parser() {
    on_big_stack(fuzz_sql_parser_body);
}

fn fuzz_sql_parser_body() {
    arm_watchdog();
    let corpus = Corpus::open("sql", ".sql");
    for s in SQL_SEEDS {
        corpus.offer(hash_bytes(s.as_bytes(), 0), s.as_bytes());
    }
    let saved = corpus.load();
    let _quiet = Quiet::on();
    let mut fp = Fp::default();

    // Replay first: the corpus is the accumulated coverage, and a regression
    // in it is worth finding before spending the budget on new ground.
    for (name, bytes) in &saved {
        let Ok(s) = std::str::from_utf8(bytes) else { continue };
        tick(0, hash_bytes(bytes, 0));
        check_parse(s, &mut fp);
        let _ = name;
    }

    let mut rng = Rng::new(base_seed(1));
    let mut buf = String::new();
    let mut alt = String::new();
    let n = cases(6_000);
    for i in 0..n {
        let seed = rng.next();
        tick(0, seed);
        let mut r = Rng::new(seed);
        // Rotate the three generators rather than picking randomly, so a run
        // of any length gets all three in proportion.
        match i % 3 {
            0 => gen_soup(&mut r, &mut buf),
            1 => gen_sql(&mut r, &mut buf),
            _ => {
                // Mutate something that already exists: a stored corpus entry
                // when there is one, otherwise a freshly generated statement.
                gen_sql(&mut r, &mut alt);
                let base: &str = if !saved.is_empty() && r.chance(2) {
                    std::str::from_utf8(&saved[r.below(saved.len())].1).unwrap_or(&alt)
                } else {
                    SQL_SEEDS[r.below(SQL_SEEDS.len())]
                };
                let base = base.to_string();
                mutate(&mut r, &base, &alt, &mut buf);
            }
        }
        let sig = check_parse(&buf, &mut fp);
        corpus.offer(sig, buf.as_bytes());
    }
}

#[test]
fn sql_recursion_guards_hold_under_generated_input() {
    on_big_stack(sql_recursion_guards_hold_under_generated_input_body);
}

fn sql_recursion_guards_hold_under_generated_input_body() {
    arm_watchdog();
    let _quiet = Quiet::on();
    let mut fp = Fp::default();

    // The guard is `MAX_DEPTH = 200` in `sql::parser`. These are the shapes
    // that reach it -- one per recursive production, because a guard that
    // covers `primary` but not `type` still aborts the process.
    //
    // 20_000 is deliberately two orders of magnitude past the limit: an
    // off-by-one in the guard survives 201, not 20_000.
    let n = 20_000;
    let deep: Vec<String> = vec![
        "SELECT ".to_string() + &"(".repeat(n) + "1" + &")".repeat(n),
        "SELECT ".to_string() + &"NOT ".repeat(n) + "1",
        "SELECT ".to_string() + &"-".repeat(n) + "1",
        "SELECT ".to_string() + &"CASE WHEN 1 THEN ".repeat(n) + "1" + &" END".repeat(n),
        "SELECT ".to_string() + &"f(".repeat(n) + "1" + &")".repeat(n),
        "SELECT ".to_string() + &"(SELECT ".repeat(n) + "1" + &")".repeat(n),
        "CREATE TABLE t (a ".to_string()
            + &"Nullable(".repeat(n)
            + "UInt64"
            + &")".repeat(n)
            + ") ENGINE = Memory",
        "SELECT 1".to_string() + &" UNION ALL SELECT 1".repeat(n),
        "SELECT ".to_string() + &"1 + ".repeat(n) + "1",
        "SELECT 1::UInt64".to_string() + &"::UInt64".repeat(n),
        "SELECT a".to_string() + &"[1]".repeat(n),
        "WITH ".to_string() + &"x AS (SELECT 1), ".repeat(n) + "y AS (SELECT 1) SELECT 1",
        "SELECT * FROM t".to_string() + &" JOIN u USING (a)".repeat(n),
        "SELECT 1 IN (".to_string() + &"1,".repeat(n) + "1)",
        // Unbalanced: the guard must fire on the way *down*, before the
        // matching close would have unwound it.
        "SELECT ".to_string() + &"(".repeat(n),
        "SELECT ".to_string() + &")".repeat(n),
        "SELECT ".to_string() + &"/*".repeat(n),
        "SELECT ".to_string() + &"'".repeat(n),
        ";".repeat(n),
        "SELECT 1".to_string() + &";SELECT 1".repeat(n),
    ];

    for (i, sql) in deep.iter().enumerate() {
        tick(0, i as u64);
        // Reaching here at all proves no stack overflow: an overflow is a
        // SIGSEGV that `catch_unwind` cannot see, so the test dying is the
        // signal, and surviving is the assertion.
        let sig = check_parse(sql, &mut fp);
        let _ = sig;
    }

    // The other half of the claim: the guard rejects rather than truncating.
    // A parser that silently stopped descending would return `Ok` with a
    // wrong AST, which is worse than the abort it replaced.
    let over = "SELECT ".to_string() + &"(".repeat(500) + "1" + &")".repeat(500);
    let e = granular::sql::parse(&over).expect_err("500 levels must be refused");
    assert!(e.to_string().contains("nested more than"), "wrong rejection: {e}");

    // …and that it does not reject *under* the limit, or the guard is a
    // correctness bug of its own. Measured: a parenthesis costs two levels
    // (`expr` then `primary`), so the wall sits at 99 parens, not 200. That is
    // the guard being conservative rather than wrong -- but it is worth
    // pinning, because a refactor that makes any production cost a third level
    // silently halves the depth of query users can write.
    granular::sql::parse(&("SELECT ".to_string() + &"(".repeat(98) + "1" + &")".repeat(98)))
        .expect("98 nested parentheses must still parse");
    let e = granular::sql::parse(&("SELECT ".to_string() + &"(".repeat(99) + "1" + &")".repeat(99)))
        .expect_err("99 nested parentheses reach the 200-level guard");
    assert!(e.to_string().contains("nested more than"), "wrong rejection: {e}");
}

#[test]
fn sql_parse_is_linear_enough_to_finish() {
    on_big_stack(sql_parse_is_linear_enough_to_finish_body);
}

fn sql_parse_is_linear_enough_to_finish_body() {
    arm_watchdog();
    let _quiet = Quiet::on();
    // Not a timing assertion -- this machine swings 3x under load, so a
    // wall-clock bound would be flaky. The watchdog is the real check; this
    // test exists to feed it the shapes where a quadratic or exponential
    // parser would actually hang: long flat inputs, not deep ones.
    let mut fp = Fp::default();
    for (i, sql) in [
        "SELECT ".to_string() + &"a, ".repeat(50_000) + "a FROM t",
        "SELECT * FROM t WHERE ".to_string() + &"a = 1 AND ".repeat(50_000) + "a = 1",
        "INSERT INTO t VALUES ".to_string() + &"(1),".repeat(50_000) + "(1)",
        "SELECT '".to_string() + &"x".repeat(1_000_000) + "'",
        "SELECT ".to_string() + &"x".repeat(1_000_000),
        "-- ".to_string() + &"x".repeat(1_000_000),
        "/* ".to_string() + &"x".repeat(1_000_000) + " */ SELECT 1",
        " ".repeat(1_000_000) + "SELECT 1",
    ]
    .iter()
    .enumerate()
    {
        tick(0, i as u64);
        check_parse(sql, &mut fp);
    }
}

// ===========================================================================
// damage: the mutation operators every on-disk target shares
// ===========================================================================

/// Overwrite `buf[at..]` in place with the canonical LEB128 encoding of `v`,
/// clipped at the end of the buffer.
///
/// In place and clipped rather than spliced: a length field lives at a fixed
/// offset inside a structure, and inserting bytes would shift everything after
/// it, which is a different (and much less interesting) kind of damage.
fn put_varint_at(buf: &mut [u8], at: usize, v: u64) {
    let mut enc = [0u8; 10];
    let mut n = 0;
    let mut x = v;
    while x >= 0x80 {
        enc[n] = (x as u8) | 0x80;
        n += 1;
        x >>= 7;
    }
    enc[n] = x as u8;
    n += 1;
    let end = (at + n).min(buf.len());
    buf[at..end].copy_from_slice(&enc[..end - at]);
}

/// The length-field values worth forcing: `u64::MAX` and the powers of two
/// that clear a naive `< isize::MAX` check while still asking the allocator
/// for hundreds of gigabytes, plus one past `MAX_COUNT` (2^24), which is the
/// bound `persist::reader` actually applies to structure counts.
const EXTREME_LENGTHS: [u64; 6] =
    [u64::MAX, 1 << 60, 1 << 40, 1 << 32, (1 << 24) + 1, usize::MAX as u64];

/// Apply one round of damage to a serialized image.
///
/// The strategies are the ones that actually break file formats, in rough
/// order of how often they occur in the wild: a flipped bit (rot), a short
/// file (interrupted write), a zeroed block (a page the filesystem allocated
/// and never wrote back), and -- the hostile cases -- a length field driven to
/// an extreme and a byte replaced with one of the values a forger reaches for.
fn damage(rng: &mut Rng, buf: &mut Vec<u8>) {
    if buf.is_empty() {
        buf.push(rng.next() as u8);
        return;
    }
    match rng.below(9) {
        // Bit flip.
        0 => {
            let i = rng.below(buf.len());
            buf[i] ^= 1 << rng.below(8);
        }
        // Byte set to a value a forger would choose. 0x00/0xFF are the
        // all-clear/all-set patterns; 0x80 is the varint continuation bit,
        // which is what turns a 1-byte length into a 10-byte one.
        1 => {
            let i = rng.below(buf.len());
            buf[i] = *rng.pick(&[0x00u8, 0x01, 0x7F, 0x80, 0xFF, 0xFE]);
        }
        // Truncate. Biased to the ends, where the header and the footer are.
        2 => {
            let n = match rng.below(3) {
                0 => rng.below(64.min(buf.len())),
                1 => buf.len() - rng.below(64.min(buf.len())),
                _ => rng.below(buf.len()),
            };
            buf.truncate(n);
        }
        // Extend with junk: a file longer than its own structure claims.
        3 => {
            let n = 1 + rng.below(64);
            for _ in 0..n {
                buf.push(rng.next() as u8);
            }
        }
        // Zero a run. A suffix run is the "hole" shape the WAL's `is_tail`
        // exists for; an interior run is unambiguous rot.
        4 => {
            let at = rng.below(buf.len());
            let end = (at + 1 + rng.below(4096)).min(buf.len());
            buf[at..end].fill(0);
        }
        // A length field at its extreme. Drawn twice as often as the other
        // operators because it is the only one that targets the specific bug
        // class this file exists to rule out -- a `Vec` sized from a number the
        // file supplied.
        5 | 6 => {
            let at = rng.below(buf.len());
            let v = *rng.pick(&EXTREME_LENGTHS);
            put_varint_at(buf, at, v);
        }
        // Swap two 8-byte words: reorders structure without changing any
        // byte, which a checksum catches and a length check does not.
        7 => {
            if buf.len() >= 16 {
                let a = rng.below(buf.len() - 8) & !7;
                let b = rng.below(buf.len() - 8) & !7;
                for k in 0..8 {
                    buf.swap(a + k, b + k);
                }
            }
        }
        // Splice: copy a run from elsewhere in the same file. Produces
        // structurally plausible garbage that random bytes never do.
        _ => {
            let n = 1 + rng.below(64.min(buf.len()));
            let src = rng.below(buf.len() - n + 1);
            let dst = rng.below(buf.len() - n + 1);
            let run: Vec<u8> = buf[src..src + n].to_vec();
            buf[dst..dst + n].copy_from_slice(&run);
        }
    }
}

/// Recompute a `doc`'s footer checksum. The footer covers only its own first
/// 12 bytes (offset + version), so this is a fixed rewrite that does not
/// depend on the body.
fn repair_footer(buf: &mut [u8]) -> Option<()> {
    let start = buf.len().checked_sub(format::FOOTER_LEN)?;
    let ck = format::checksum(&buf[start..start + 12]);
    buf[start + 12..start + 20].copy_from_slice(&ck.to_le_bytes());
    buf[start + 20..].copy_from_slice(&format::MAGIC);
    Some(())
}

/// Recompute the frame checksum of the single body a `writer::doc` wraps.
///
/// Without this, ~every mutation dies at the frame checksum and the semantic
/// validators underneath -- the ones that decide whether a part name is a path
/// traversal, whether a column index is in range, whether a row count is
/// believable -- are never reached. Repairing the checksum is what a forger
/// would do, and it is the only way to fuzz the layer that matters.
fn repair_doc(buf: &mut [u8]) -> Option<()> {
    repair_footer(buf)?;
    let (sum_at, body_at, end) = {
        let at = format::read_footer(buf).ok()? as usize;
        let mut r = format::Reader::new(buf);
        r.seek(at).ok()?;
        let len = r.varint().ok()? as usize;
        r.u64().ok()?;
        let sum_at = r.pos().checked_sub(8)?;
        let body_at = r.pos();
        (sum_at, body_at, body_at.checked_add(len)?)
    };
    if end > buf.len() {
        return None;
    }
    let ck = format::checksum(&buf[body_at..end]);
    buf[sum_at..sum_at + 8].copy_from_slice(&ck.to_le_bytes());
    Some(())
}

/// The same trick for a part file's metadata frame, which is written with
/// `write_framed_aligned` (body padded to 8 so packed lanes can be read out of
/// a mapping in place), so the body starts at the next multiple of 8 rather
/// than immediately after the checksum.
fn repair_part_meta(buf: &mut [u8]) -> Option<()> {
    repair_footer(buf)?;
    let (sum_at, body_at, end) = {
        let at = format::read_footer(buf).ok()? as usize;
        let mut r = format::Reader::new(buf);
        r.seek(at).ok()?;
        let len = r.varint().ok()? as usize;
        r.u64().ok()?;
        let sum_at = r.pos().checked_sub(8)?;
        let body_at = r.pos().next_multiple_of(8);
        (sum_at, body_at, body_at.checked_add(len)?)
    };
    if end > buf.len() {
        return None;
    }
    let ck = format::checksum(&buf[body_at..end]);
    buf[sum_at..sum_at + 8].copy_from_slice(&ck.to_le_bytes());
    Some(())
}

/// Every decoder measured here is single-threaded and reads from a buffer it
/// does not own, so the only thing it should ever size from the file is
/// bounded by the file. 16 MiB is two orders of magnitude above what any
/// fixture in this file legitimately needs (measured: 1.6 MiB peak for a
/// healthy 20k-row part, and 64 KiB for a `TABLE`/`CATALOG` document), and
/// four orders below what an unchecked `varint` would ask for.
const ALLOC_CEILING: usize = 16 * 1024 * 1024;

/// Run one decoder over one hostile image and check the shared contract:
/// `Ok` or `Corruption`, no panic, no allocation sized from the input.
fn check_decode<T>(
    tag: &str,
    bytes: &[u8],
    fp: &mut Fp,
    f: impl FnOnce(&[u8]) -> granular::Result<T>,
) -> u64 {
    let (res, peak) = peak_request(|| no_panic(|| f(bytes)));
    assert!(
        peak <= ALLOC_CEILING,
        "{tag} asked for {peak} bytes from a {}-byte image (ceiling {ALLOC_CEILING})",
        bytes.len()
    );
    match res {
        Err(msg) => panic!(
            "PANIC in {tag}: {msg}\n  image ({} bytes): {:02x?}",
            bytes.len(),
            &bytes[..bytes.len().min(256)]
        ),
        Ok(Ok(_)) => fp.of(tag, None),
        Ok(Err(e)) => {
            assert!(
                is_corrupt(&e),
                "{tag} rejected a damaged image with {} ({e}), not Corruption",
                e.code()
            );
            fp.of(tag, Some(&e))
        }
    }
}

// ===========================================================================
// target 2: part files
// ===========================================================================

#[test]
fn fuzz_part_reader() {
    on_big_stack(fuzz_part_reader_body);
}

fn fuzz_part_reader_body() {
    arm_watchdog();
    let corpus = Corpus::open("part", ".gpart");
    let _quiet = Quiet::on();
    let mut fp = Fp::default();

    for (_, bytes) in corpus.load() {
        tick(1, hash_bytes(&bytes, 0));
        check_decode("part", &bytes, &mut fp, reader::part_from_bytes);
    }

    // Three sizes: one granule, a granule boundary, and several granules with
    // a router and a bloom filter, because the decoder's shape changes at each.
    let images: Vec<Vec<u8>> = [1usize, 1024, 2_100]
        .iter()
        .map(|&n| writer::part_bytes(&fixture_part(n)).expect("serialize"))
        .collect();

    // The healthy image must round-trip, or every damaged verdict below is
    // measuring the wrong thing.
    for img in &images {
        let (r, peak) = peak_request(|| reader::part_from_bytes(img));
        r.expect("a freshly written part must load");
        assert!(peak <= ALLOC_CEILING, "healthy part decode peaked at {peak}");
    }

    let mut rng = Rng::new(base_seed(2));
    let mut buf = Vec::new();
    let n = cases(4_000);
    for _ in 0..n {
        let seed = rng.next();
        tick(1, seed);
        let mut r = Rng::new(seed);
        buf.clear();
        buf.extend_from_slice(&images[r.below(images.len())]);
        for _ in 0..1 + r.below(3) {
            damage(&mut r, &mut buf);
        }
        // Half the cases get their checksums put back, so the run is split
        // between "does the checksum layer hold" and "does the layer under it
        // hold when the checksum has been forged".
        if r.chance(2) {
            let _ = repair_part_meta(&mut buf);
        }
        let sig = check_decode("part", &buf, &mut fp, reader::part_from_bytes);
        corpus.offer(sig, &buf);
    }
}

#[test]
fn part_truncation_at_every_boundary_is_rejected() {
    on_big_stack(part_truncation_at_every_boundary_is_rejected_body);
}

fn part_truncation_at_every_boundary_is_rejected_body() {
    arm_watchdog();
    let _quiet = Quiet::on();
    let mut fp = Fp::default();
    let img = writer::part_bytes(&fixture_part(1_500)).unwrap();

    // Dense at both ends (header, footer, metadata frame) and sampled through
    // the granule payload, which is homogeneous and where every prefix behaves
    // the same. Sampling the middle is what keeps this test under a second.
    let mut probes: Vec<usize> = (0..768).collect();
    probes.extend((768..img.len()).step_by(101));
    probes.extend(img.len().saturating_sub(768)..img.len());
    for n in probes {
        tick(1, n as u64);
        let cut = &img[..n];
        let sig = check_decode("part-trunc", cut, &mut fp, reader::part_from_bytes);
        let _ = sig;
        // A strict prefix is never a whole part: the footer is at the end.
        assert!(
            reader::part_from_bytes(cut).is_err(),
            "a {n}-byte prefix of a {}-byte part parsed",
            img.len()
        );
    }
}

#[test]
fn part_length_fields_at_their_extremes_do_not_allocate() {
    on_big_stack(part_length_fields_at_their_extremes_do_not_allocate_body);
}

fn part_length_fields_at_their_extremes_do_not_allocate_body() {
    arm_watchdog();
    let _quiet = Quiet::on();
    let mut fp = Fp::default();
    let img = writer::part_bytes(&fixture_part(200)).unwrap();

    // Walk a maximal varint across every offset of the metadata frame and
    // repair the checksum behind it. This is the direct attack on the
    // "size a Vec from a length prefix" bug class: at some offset that varint
    // *is* the granule count, the row count, the dictionary length or a word
    // count, and `check_decode`'s allocation ceiling is what proves each one
    // is bounded before it is believed.
    let meta_at = format::read_footer(&img).unwrap() as usize;
    let mut buf = Vec::with_capacity(img.len());
    for at in meta_at..img.len() - format::FOOTER_LEN {
        for &v in &EXTREME_LENGTHS {
            tick(1, at as u64);
            buf.clear();
            buf.extend_from_slice(&img);
            put_varint_at(&mut buf, at, v);
            let _ = repair_part_meta(&mut buf);
            check_decode("part-len", &buf, &mut fp, reader::part_from_bytes);
        }
    }
}

// ===========================================================================
// target 3: TABLE and CATALOG documents
// ===========================================================================

fn fixture_catalog() -> Vec<(String, Vec<TableDef>)> {
    let mut wide = fixture_def("wide");
    wide.schema = Schema::new(
        (0..24)
            .map(|i| {
                Field::new(
                    format!("c{i}"),
                    match i % 6 {
                        0 => DataType::UInt64,
                        1 => DataType::String,
                        2 => DataType::Nullable(Box::new(DataType::Int64)),
                        3 => DataType::Float64,
                        4 => DataType::FixedString(8),
                        _ => DataType::Date,
                    },
                )
            })
            .collect(),
    )
    .unwrap();
    wide.order_by = vec![0];
    wide.primary_key = vec![0];
    let mut defaulted = fixture_def("defaulted");
    defaulted.schema = Schema::new(vec![
        Field::new("id", DataType::UInt64),
        Field::new("n", DataType::Int64).with_default("42").unwrap(),
        Field::new("s", DataType::String).with_default("'x'").unwrap(),
    ])
    .unwrap();
    vec![
        ("default".to_string(), vec![fixture_def("hits"), wide]),
        ("other".to_string(), vec![defaulted]),
        ("empty".to_string(), Vec::new()),
    ]
}

#[test]
fn fuzz_catalog_and_table_documents() {
    on_big_stack(fuzz_catalog_and_table_documents_body);
}

fn fuzz_catalog_and_table_documents_body() {
    arm_watchdog();
    let corpus = Corpus::open("doc", ".doc");
    let _quiet = Quiet::on();
    let mut fp = Fp::default();

    let roster = fixture_catalog();
    let cat = writer::catalog_doc(&roster, 0x5EED_1234_5678_9ABC);
    let parts: Vec<String> = (1..=9).map(|i| format!("part_{i:06}.gpart")).collect();
    let tbl = writer::table_doc(&fixture_def("hits"), &parts, 4096);

    // Round-trip first: a document the writer just produced must decode to
    // exactly what went in, or nothing below means anything.
    let (back, instance) = reader::catalog_from_bytes(&cat).expect("catalog round-trip");
    assert_eq!(instance, 0x5EED_1234_5678_9ABC, "the instance id survives the round trip");
    assert_eq!(back.len(), roster.len());
    for ((a, ad), (b, bd)) in back.iter().zip(&roster) {
        assert_eq!(a, b);
        assert_eq!(ad.len(), bd.len());
        for (x, y) in ad.iter().zip(bd) {
            assert_eq!(x.name, y.name);
            assert_eq!(x.schema.fields().len(), y.schema.fields().len());
            for (f, g) in x.schema.fields().iter().zip(y.schema.fields()) {
                assert_eq!(f.name, g.name, "column names must survive");
                assert_eq!(f.ty, g.ty, "column types must survive");
                assert_eq!(f.default_sql(), g.default_sql(), "DEFAULTs must survive");
            }
            assert_eq!(x.order_by, y.order_by);
            assert_eq!(x.primary_key, y.primary_key);
            assert_eq!(x.partition_by, y.partition_by);
            assert_eq!(x.engine.name(), y.engine.name());
        }
    }
    let (tdef, tparts, twal) = reader::table_parts_from_bytes(&tbl).expect("table round-trip");
    assert_eq!(tparts, parts);
    assert_eq!(twal, 4096);
    assert_eq!(tdef.name, "hits");

    for (_, bytes) in corpus.load() {
        tick(2, hash_bytes(&bytes, 0));
        check_decode("catalog", &bytes, &mut fp, reader::catalog_from_bytes);
        check_decode("table", &bytes, &mut fp, reader::table_parts_from_bytes);
    }

    let mut rng = Rng::new(base_seed(3));
    let mut buf = Vec::new();
    let n = cases(5_000);
    for i in 0..n {
        let seed = rng.next();
        tick(2, seed);
        let mut r = Rng::new(seed);
        buf.clear();
        buf.extend_from_slice(if i % 2 == 0 { &cat } else { &tbl });
        for _ in 0..1 + r.below(3) {
            damage(&mut r, &mut buf);
        }
        if r.chance(2) {
            let _ = repair_doc(&mut buf);
        }
        // Both readers, on both images: a `CATALOG` fed to the `TABLE` reader
        // is exactly the confusion a directory mix-up produces, and neither
        // may do worse than reject it.
        let a = check_decode("catalog", &buf, &mut fp, reader::catalog_from_bytes);
        let b = check_decode("table", &buf, &mut fp, reader::table_parts_from_bytes);
        corpus.offer(a ^ b.rotate_left(32), &buf);
    }
}

#[test]
fn a_forged_commit_record_can_never_name_a_path() {
    on_big_stack(a_forged_commit_record_can_never_name_a_path_body);
}

fn a_forged_commit_record_can_never_name_a_path_body() {
    arm_watchdog();
    let _quiet = Quiet::on();
    // The `TABLE` file names files that recovery will open. A forged name that
    // escapes the table directory is the difference between a corrupt database
    // and an arbitrary-file-read, so it gets a targeted generator rather than
    // being left to chance in the byte fuzzer.
    let hostile = [
        "../../../../etc/passwd",
        "/etc/passwd",
        "part_000001.gpart/../../x",
        "..",
        ".",
        "",
        "part_000001.gpart\0.txt",
        "part_-00001.gpart",
        "part_18446744073709551616.gpart",
        "PART_000001.GPART",
        "part_000001.gpart ",
        "wal.log",
        "TABLE",
        "CATALOG",
        "\u{202e}tragp.100000_trap",
    ];
    for name in hostile {
        let doc = writer::table_doc(&fixture_def("hits"), &[name.to_string()], 0);
        match reader::table_parts_from_bytes(&doc) {
            Ok((_, parts, _)) => {
                // Whatever survives must be a name the store itself would
                // mint, not merely a name that happens to decode.
                assert_eq!(parts.len(), 1);
                assert!(
                    parts[0].starts_with("part_") && parts[0].ends_with(".gpart"),
                    "commit record accepted `{name}`"
                );
                assert!(!parts[0].contains('/'), "commit record accepted a path: `{name}`");
            }
            Err(e) => assert!(is_corrupt(&e), "`{name}` rejected with {}: {e}", e.code()),
        }
    }
}

// ===========================================================================
// target 4: the WAL, and the torn-tail / bit-rot boundary
// ===========================================================================

/// Write a log holding `n` plain (immediately committed) records, alternating
/// inserts and deletes, and return its bytes together with what a healthy
/// replay yields.
fn fixture_wal(dir: &Scratch, n: usize) -> (PathBuf, Vec<u8>, Vec<WalRecord>) {
    let path = dir.join("wal.log");
    let schema = fixture_schema();
    {
        let mut w = Wal::open(&path).expect("open");
        for i in 0..n {
            if i % 3 == 2 {
                w.append_delete(i as u64 * 7).expect("delete");
            } else {
                w.append_insert(&fixture_block(1 + i % 5)).expect("insert");
            }
        }
        w.sync().expect("sync");
    }
    let bytes = std::fs::read(&path).expect("read");
    let healthy = Wal::replay(&path, &schema).expect("healthy replay");
    (path, bytes, healthy)
}

#[test]
fn wal_truncation_always_yields_a_prefix() {
    on_big_stack(wal_truncation_always_yields_a_prefix_body);
}

fn wal_truncation_always_yields_a_prefix_body() {
    arm_watchdog();
    let _quiet = Quiet::on();
    let dir = Scratch::new("wal-trunc");
    let (path, bytes, healthy) = fixture_wal(&dir, 24);
    let schema = fixture_schema();

    // The claim in `wal.rs`: a crash during `write` leaves a partial record at
    // the end, that is not corruption, and replay stops cleanly at it. So
    // *every* suffix truncation must succeed and return a prefix of the
    // healthy replay -- not merely "not error", a prefix, because returning
    // the wrong records would be worse than refusing.
    for n in 0..=bytes.len() {
        tick(3, n as u64);
        std::fs::write(&path, &bytes[..n]).unwrap();
        let got = Wal::replay(&path, &schema)
            .unwrap_or_else(|e| panic!("a {n}-byte prefix of the log was rejected: {e}"));
        assert!(
            got.len() <= healthy.len() && got[..] == healthy[..got.len()],
            "a {n}-byte prefix replayed {} records that are not a prefix of the {} healthy ones",
            got.len(),
            healthy.len()
        );
    }
}

#[test]
fn wal_trailing_zeros_are_a_hole_not_rot() {
    on_big_stack(wal_trailing_zeros_are_a_hole_not_rot_body);
}

fn wal_trailing_zeros_are_a_hole_not_rot_body() {
    arm_watchdog();
    let _quiet = Quiet::on();
    let dir = Scratch::new("wal-zeros");
    let (path, bytes, healthy) = fixture_wal(&dir, 10);
    let schema = fixture_schema();

    // The subtle half of `is_tail`: a filesystem that allocated a block and
    // never wrote it back leaves zeros, not a short file. Measuring the end of
    // the data at EOF instead of at the last non-zero byte turns the most
    // ordinary crash shape into permanent, unopenable rot. Zeroing every
    // suffix is the exhaustive form of that check.
    // From the first byte a record can start at: zeroing the header is not a
    // hole, it is a destroyed file, and `format::read_header` is right to
    // refuse it. The claim under test is about record bytes.
    for n in format::HEADER_LEN..bytes.len() {
        tick(3, n as u64);
        let mut c = bytes.clone();
        c[n..].fill(0);
        std::fs::write(&path, &c).unwrap();
        let got = Wal::replay(&path, &schema)
            .unwrap_or_else(|e| panic!("zeroing from byte {n} was reported as rot: {e}"));
        assert!(
            got.len() <= healthy.len() && got[..] == healthy[..got.len()],
            "zeroing from {n} replayed records that are not a prefix"
        );

        // …and the log must still be openable afterwards, which is the part
        // that actually costs data when it is wrong: `Wal::open` truncates
        // the torn tail, and an open that fails here is an outage.
        //
        // Sampled, not exhaustive: a truncating `open` does two `fsync`s and a
        // `sync_dir`, and at one per byte offset this single line was 14 of
        // the suite's 20 seconds. The replay above already covers every
        // offset, and `wal_reopen_after_damage_is_append_clean` covers the
        // reopen contract against damage that is not a clean suffix of zeros.
        if n % 16 == 0 {
            Wal::open(&path)
                .unwrap_or_else(|e| panic!("zeroing from byte {n} bricked the log: {e}"));
        }
    }
}

#[test]
fn wal_damage_never_invents_a_record() {
    on_big_stack(wal_damage_never_invents_a_record_body);
}

fn wal_damage_never_invents_a_record_body() {
    arm_watchdog();
    let _quiet = Quiet::on();
    let dir = Scratch::new("wal-damage");
    let (path, bytes, healthy) = fixture_wal(&dir, 20);
    let schema = fixture_schema();
    let mut fp = Fp::default();
    let mut rng = Rng::new(base_seed(4));
    let mut buf = Vec::new();
    let n = cases(3_000);

    for _ in 0..n {
        let seed = rng.next();
        tick(3, seed);
        let mut r = Rng::new(seed);
        buf.clear();
        buf.extend_from_slice(&bytes);
        for _ in 0..1 + r.below(3) {
            damage(&mut r, &mut buf);
        }
        std::fs::write(&path, &buf).unwrap();

        let (res, peak) = peak_request(|| no_panic(|| Wal::replay(&path, &schema)));
        assert!(peak <= ALLOC_CEILING, "replay asked for {peak} bytes");
        let sig = match res {
            Err(msg) => panic!("PANIC in Wal::replay: {msg}\n  seed 0x{seed:016x}"),
            Ok(Err(e)) => {
                assert!(is_corrupt(&e), "replay rejected with {}: {e}", e.code());
                fp.of("wal", Some(&e))
            }
            Ok(Ok(got)) => {
                // The invariant that makes the torn-tail rule safe: framing
                // only ever *shortens* the history. A frame whose checksum
                // passes is a frame we wrote, so anything replay returns must
                // be a prefix of what was written -- if damage could ever make
                // it return a different record, "stop at the first bad frame"
                // would be silently losing acknowledged writes instead of
                // reporting them.
                assert!(
                    got.len() <= healthy.len() && got[..] == healthy[..got.len()],
                    "damaged log replayed {} records that are not a prefix of the {} written \
                     (seed 0x{seed:016x})",
                    got.len(),
                    healthy.len()
                );
                fp.of("wal", None)
            }
        };
        let _ = sig;
    }
}

#[test]
fn wal_reopen_after_damage_is_append_clean() {
    on_big_stack(wal_reopen_after_damage_is_append_clean_body);
}

fn wal_reopen_after_damage_is_append_clean_body() {
    arm_watchdog();
    let _quiet = Quiet::on();
    let dir = Scratch::new("wal-reopen");
    let (path, bytes, _) = fixture_wal(&dir, 16);
    let schema = fixture_schema();
    let mut rng = Rng::new(base_seed(5));
    let mut buf = Vec::new();
    // Three `fsync`s per case (open's truncate, the append, the drop): ~5ms,
    // all of it waiting on the disk. Every one of these cases is a `damage`
    // draw that `wal_damage_never_invents_a_record` takes at full speed.
    let n = heavy_cases(300, 5_000);

    for _ in 0..n {
        let seed = rng.next();
        tick(3, seed);
        let mut r = Rng::new(seed);
        buf.clear();
        buf.extend_from_slice(&bytes);
        for _ in 0..1 + r.below(2) {
            damage(&mut r, &mut buf);
        }
        std::fs::write(&path, &buf).unwrap();

        // The rule `Wal::open` exists to enforce: after opening, appending
        // must be *visible*. If open left a byte of unparseable tail behind,
        // the next record would sit behind bytes that never parse and the
        // following replay would stop in front of it -- silently losing an
        // acknowledged write, which is the one failure a log may not have.
        let Ok(mut w) = Wal::open(&path) else { continue };
        let Ok(before) = Wal::replay(&path, &schema) else { continue };
        w.append_delete(0xDEAD_BEEF).expect("append after reopen");
        w.sync().expect("sync after reopen");
        drop(w);

        let after = Wal::replay(&path, &schema).unwrap_or_else(|e| {
            panic!("a record appended after reopen was unreadable (seed 0x{seed:016x}): {e}")
        });
        assert_eq!(
            after.len(),
            before.len() + 1,
            "append after reopen changed the history (seed 0x{seed:016x})"
        );
        assert_eq!(after[..before.len()], before[..], "reopen rewrote history");
        assert_eq!(
            after[before.len()],
            WalRecord::Delete(0xDEAD_BEEF),
            "the appended record did not survive"
        );
    }
}

#[test]
fn wal_staged_records_need_their_commit_marker() {
    on_big_stack(wal_staged_records_need_their_commit_marker_body);
}

fn wal_staged_records_need_their_commit_marker_body() {
    arm_watchdog();
    let _quiet = Quiet::on();
    let dir = Scratch::new("wal-staged");
    let path = dir.join("wal.log");
    let schema = fixture_schema();

    // A staged group with no marker is a write that was logged and then failed
    // -- replay must drop it. Truncating the log to every byte offset walks the
    // marker in and out of existence, and the record count may only ever step
    // by exactly the size of the group.
    {
        let mut w = Wal::open(&path).unwrap();
        w.append_insert(&fixture_block(3)).unwrap();
        let seq = w.begin();
        w.append_insert_staged(seq, &fixture_block(2)).unwrap();
        w.append_delete_staged(seq, 99).unwrap();
        w.commit(seq).unwrap();
        let orphan = w.begin();
        w.append_insert_staged(orphan, &fixture_block(4)).unwrap();
        w.append_insert(&fixture_block(1)).unwrap();
        w.sync().unwrap();
    }
    let bytes = std::fs::read(&path).unwrap();

    // 4 committed records: the leading insert, the two released by the marker,
    // and the trailing insert. The orphan is dropped by construction.
    let full = Wal::replay(&path, &schema).unwrap();
    assert_eq!(full.len(), 4, "an uncommitted staged record was replayed: {full:?}");

    let mut seen = Vec::new();
    for n in 0..=bytes.len() {
        tick(3, n as u64);
        std::fs::write(&path, &bytes[..n]).unwrap();
        let got = Wal::replay(&path, &schema)
            .unwrap_or_else(|e| panic!("prefix {n} of a staged log was rejected: {e}"));
        // Every prefix must be explicable: no record may appear that the full
        // replay does not contain, in the same relative order.
        let mut it = full.iter();
        for rec in &got {
            assert!(
                it.any(|f| f == rec),
                "prefix {n} replayed a record the full log does not contain: {rec:?}"
            );
        }
        seen.push(got.len());
    }
    // The counts must be monotone: a longer log can only ever release more.
    assert!(
        seen.windows(2).all(|w| w[0] <= w[1]),
        "record count is not monotone in log length: {seen:?}"
    );
}

// ===========================================================================
// target 5: block records (the WAL payload, with no checksum in front of it)
// ===========================================================================

#[test]
fn fuzz_block_records() {
    on_big_stack(fuzz_block_records_body);
}

fn fuzz_block_records_body() {
    arm_watchdog();
    let corpus = Corpus::open("block", ".blk");
    let _quiet = Quiet::on();
    let mut fp = Fp::default();
    let schema = fixture_schema();

    // `put_block`/`get_block` is the one encoding in the format with *no*
    // checksum of its own -- inside a log record the frame carries it, so the
    // decoder is the only thing standing between a mutated body and the
    // column builders. That makes it the highest-yield fuzz target in the
    // whole persistence layer: every mutation reaches a semantic check.
    let mut w = format::Writer::new();
    writer::put_block(&mut w, &fixture_block(700));
    let img = w.finish();
    reader::block_from_bytes(&img, &schema).expect("a written block must decode");

    for (_, bytes) in corpus.load() {
        tick(4, hash_bytes(&bytes, 0));
        check_decode("block", &bytes, &mut fp, |b| reader::block_from_bytes(b, &schema));
    }

    let mut rng = Rng::new(base_seed(6));
    let mut buf = Vec::new();
    let n = cases(8_000);
    for _ in 0..n {
        let seed = rng.next();
        tick(4, seed);
        let mut r = Rng::new(seed);
        buf.clear();
        // A quarter of the cases are pure noise rather than damaged truth:
        // the decoder must survive a body that was never a block at all,
        // which is what a log record from a different table looks like.
        if r.chance(4) {
            let len = r.below(512);
            buf.reserve(len);
            for _ in 0..len {
                buf.push(r.next() as u8);
            }
        } else {
            buf.extend_from_slice(&img);
            for _ in 0..1 + r.below(3) {
                damage(&mut r, &mut buf);
            }
        }
        let sig = check_decode("block", &buf, &mut fp, |b| reader::block_from_bytes(b, &schema));
        corpus.offer(sig, &buf);
    }
}

// ===========================================================================
// target 6: round-trip properties
// ===========================================================================

/// A random block over a random schema, biased hard towards the values that
/// break codecs: the ends of every integer range, NaN and the infinities,
/// empty and multi-byte strings, all-null and all-same columns.
fn random_block(rng: &mut Rng) -> (Schema, Block) {
    let types = [
        DataType::UInt64,
        DataType::UInt32,
        DataType::Int64,
        DataType::Int8,
        DataType::Float64,
        DataType::Bool,
        DataType::String,
        DataType::Date,
        DataType::DateTime,
        DataType::Nullable(Box::new(DataType::Int64)),
        DataType::Nullable(Box::new(DataType::String)),
        DataType::Nullable(Box::new(DataType::Float64)),
    ];
    let ncols = 1 + rng.below(5);
    let rows = 1 + rng.below(2_500);

    // Column 0 is the sort key and must be unique and ascending, so it is
    // built separately -- a part with a non-unique key takes a different
    // (interpolation-search) path, which the duplicate-key case below covers.
    let mut keys: Vec<u64> = Vec::with_capacity(rows);
    let mut k = rng.next() % 1_000_000;
    for _ in 0..rows {
        keys.push(k);
        k = k.wrapping_add(1 + rng.next() % 97);
    }
    let mut fields = vec![Field::new("k", DataType::UInt64)];
    let mut cols = vec![Column::u64s(DataType::UInt64, keys)];

    for c in 1..ncols {
        let ty = rng.pick(&types).clone();
        fields.push(Field::new(format!("c{c}"), ty.clone()));
        let mut b = ColumnBuilder::with_capacity(ty.clone(), rows);
        let nullable = matches!(ty, DataType::Nullable(_));
        let inner = match &ty {
            DataType::Nullable(t) => (**t).clone(),
            t => t.clone(),
        };
        for _ in 0..rows {
            if nullable && rng.chance(5) {
                b.push_null();
                continue;
            }
            let v = match inner {
                DataType::UInt64 | DataType::UInt32 => Value::UInt(match rng.below(6) {
                    0 => 0,
                    1 => u32::MAX as u64,
                    2 => u64::MAX,
                    3 => 1,
                    _ => rng.next(),
                }),
                DataType::Int64 | DataType::Int8 => Value::Int(match rng.below(6) {
                    0 => 0,
                    1 => i64::MIN,
                    2 => i64::MAX,
                    3 => -1,
                    _ => rng.next() as i64,
                }),
                DataType::Float64 => Value::Float(match rng.below(8) {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f64::NAN,
                    3 => f64::INFINITY,
                    4 => f64::NEG_INFINITY,
                    5 => f64::MIN_POSITIVE,
                    6 => f64::MAX,
                    _ => f64::from_bits(rng.next()),
                }),
                DataType::Bool => Value::Bool(rng.chance(2)),
                DataType::String => Value::str(match rng.below(6) {
                    0 => String::new(),
                    1 => "\u{0}\u{1}".to_string(),
                    2 => "\u{4e2d}\u{6587}\u{1f600}".to_string(),
                    3 => "x".repeat(1 + rng.below(300)),
                    4 => format!("lo{}", rng.below(4)),
                    _ => format!("{:x}", rng.next()),
                }),
                DataType::Date => Value::Date(rng.next() as u32),
                DataType::DateTime => Value::DateTime(rng.next() as i64),
                _ => Value::Null,
            };
            // A generator that produced an out-of-range value for the type is
            // the generator's bug, not the engine's; fall back to null/zero
            // rather than failing the round-trip for it.
            if b.push_value(&v).is_err() {
                if nullable {
                    b.push_null();
                } else {
                    b.push_value(&Value::UInt(0))
                        .or_else(|_| b.push_value(&Value::Int(0)))
                        .or_else(|_| b.push_value(&Value::Float(0.0)))
                        .or_else(|_| b.push_value(&Value::Bool(false)))
                        .or_else(|_| b.push_value(&Value::str("")))
                        .or_else(|_| b.push_value(&Value::Date(0)))
                        .or_else(|_| b.push_value(&Value::DateTime(0)))
                        .expect("some value must be pushable");
                }
            }
        }
        cols.push(b.finish());
    }
    (Schema::new(fields).unwrap(), Block::new(cols).unwrap())
}

/// `Value` equality that treats the two payloads a bit pattern can carry.
/// `NaN != NaN` under `PartialEq`, but a round trip must preserve the bits,
/// so floats are compared by their representation; `-0.0` and `0.0` are
/// distinct here for the same reason.
fn same_value(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Float(x), Value::Float(y)) => x.to_bits() == y.to_bits(),
        _ => a == b,
    }
}

#[test]
fn part_roundtrip_is_exact_for_random_data() {
    on_big_stack(part_roundtrip_is_exact_for_random_data_body);
}

fn part_roundtrip_is_exact_for_random_data_body() {
    arm_watchdog();
    let mut rng = Rng::new(base_seed(7));
    // Heavy: a part build fans out across every core, and the comparison below
    // walks every cell of a block that can be 2,500 rows wide.
    let n = heavy_cases(120, 4_000);
    for _ in 0..n {
        let seed = rng.next();
        tick(5, seed);
        let mut r = Rng::new(seed);
        let (_, block) = random_block(&mut r);
        let part = Part::build(&block, Some(0), Some(0)).expect("build");
        let img = writer::part_bytes(&part).expect("serialize");
        let back = reader::part_from_bytes(&img).expect("deserialize");

        assert_eq!(back.n_rows, part.n_rows, "row count (seed 0x{seed:016x})");
        assert_eq!(back.ncols, part.ncols, "column count (seed 0x{seed:016x})");
        assert_eq!(back.granule_count(), part.granule_count(), "granules");
        let (before, after) = (dump(&part), dump(&back));
        assert_eq!(before.len(), after.len(), "row count (seed 0x{seed:016x})");
        for (i, (x, y)) in before.iter().zip(&after).enumerate() {
            assert_eq!(x.len(), y.len(), "row {i} width (seed 0x{seed:016x})");
            for (c, (a, b)) in x.iter().zip(y).enumerate() {
                assert!(
                    same_value(a, b),
                    "row {i} column {c}: {a:?} became {b:?} (seed 0x{seed:016x})"
                );
            }
        }

        // The whole point of persisting a part rather than re-inserting it:
        // the minimal perfect hash comes back rather than being rebuilt.
        for (gi, (g, h)) in part.granules.iter().zip(&back.granules).enumerate() {
            assert_eq!(g.len, h.len, "granule {gi} length");
            assert_eq!(g.pk.is_some(), h.pk.is_some(), "granule {gi} lost its index");
            assert_eq!((g.sort_min, g.sort_max), (h.sort_min, h.sort_max), "zone map {gi}");
        }

        // Serialization must be a function of the part, not of the run: part
        // checksums are used as identity, so two writers emitting the same
        // logical part have to agree byte for byte.
        let again = writer::part_bytes(&back).expect("re-serialize");
        assert_eq!(img, again, "re-serializing a decoded part changed the bytes");
    }
}

#[test]
fn wal_roundtrip_is_exact_for_random_blocks() {
    on_big_stack(wal_roundtrip_is_exact_for_random_blocks_body);
}

fn wal_roundtrip_is_exact_for_random_blocks_body() {
    arm_watchdog();
    let dir = Scratch::new("wal-rt");
    let mut rng = Rng::new(base_seed(8));
    // Heavy: one log file, one `fsync` and one full re-read per case.
    let n = heavy_cases(60, 2_000);
    for case in 0..n {
        let seed = rng.next();
        tick(5, seed);
        let mut r = Rng::new(seed);
        let (schema, block) = random_block(&mut r);
        let path = dir.join(&format!("rt-{case}.log"));
        let _ = std::fs::remove_file(&path);
        let mut want = Vec::new();
        {
            let mut w = Wal::open(&path).expect("open");
            for i in 0..1 + r.below(4) {
                if r.chance(3) {
                    let lane = r.next();
                    w.append_delete(lane).expect("delete");
                    want.push(WalRecord::Delete(lane));
                } else {
                    w.append_insert(&block).expect("insert");
                    want.push(WalRecord::Insert(block.clone()));
                }
                let _ = i;
            }
            w.sync().expect("sync");
        }
        let got = Wal::replay(&path, &schema).expect("replay");
        assert_eq!(got.len(), want.len(), "record count (seed 0x{seed:016x})");
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            match (a, b) {
                (WalRecord::Delete(x), WalRecord::Delete(y)) => {
                    assert_eq!(x, y, "record {i} lane (seed 0x{seed:016x})")
                }
                (WalRecord::Insert(x), WalRecord::Insert(y)) => {
                    assert_eq!(x.rows(), y.rows(), "record {i} rows");
                    assert_eq!(x.width(), y.width(), "record {i} width");
                    for c in 0..x.width() {
                        for row in 0..x.rows() {
                            let (p, q) = (x.column(c).value(row), y.column(c).value(row));
                            assert!(
                                same_value(&p, &q),
                                "record {i} col {c} row {row}: {q:?} came back as {p:?} \
                                 (seed 0x{seed:016x})"
                            );
                        }
                    }
                }
                _ => panic!("record {i} changed kind (seed 0x{seed:016x})"),
            }
        }
        let _ = std::fs::remove_file(&path);
    }
}

#[test]
fn session_roundtrip_survives_a_reopen() {
    on_big_stack(session_roundtrip_survives_a_reopen_body);
}

fn session_roundtrip_survives_a_reopen_body() {
    arm_watchdog();
    let dir = Scratch::new("session-rt");
    let mut rng = Rng::new(base_seed(9));
    let rows = 3_000usize;

    // Values chosen so the SQL text is unambiguous: the round trip being
    // tested is storage, not literal parsing, and a float printed at 17
    // digits would confuse the two.
    let mut want: Vec<(u64, String, i64)> = Vec::with_capacity(rows);
    let mut k = 0u64;
    for _ in 0..rows {
        k += 1 + rng.next() % 13;
        let s = match rng.below(5) {
            0 => String::new(),
            1 => "\u{4e2d}\u{6587}".to_string(),
            2 => "x".repeat(1 + rng.below(120)),
            3 => format!("lo{}", rng.below(4)),
            _ => format!("{:x}", rng.next()),
        };
        want.push((k, s, rng.next() as i64));
    }

    let mut sql = String::with_capacity(rows * 48);
    sql.push_str("INSERT INTO t VALUES ");
    for (i, (k, s, v)) in want.iter().enumerate() {
        if i > 0 {
            sql.push_str(", ");
        }
        let _ = write!(sql, "({k}, '{}', {v})", s.replace('\'', "''"));
    }

    {
        let mut db = Session::open(dir.path()).expect("open");
        db.execute("CREATE TABLE t (k UInt64, s String, v Int64) ENGINE = MergeTree ORDER BY k")
            .expect("create");
        db.execute(&sql).expect("insert");
    }
    // Reopened, so the comparison runs against bytes that went through the
    // part writer and the WAL rather than against the in-memory delta.
    let mut db = Session::open(dir.path()).expect("reopen");
    let got = db.query("SELECT k, s, v FROM t ORDER BY k").expect("select").to_values();
    assert_eq!(got.len(), want.len(), "row count after reopen");
    for (i, (row, (k, s, v))) in got.iter().zip(&want).enumerate() {
        assert_eq!(row[0], Value::UInt(*k), "row {i} key");
        assert_eq!(row[1], Value::str(s.as_str()), "row {i} string");
        assert_eq!(row[2], Value::Int(*v), "row {i} value");
    }
}

// ===========================================================================
// found by this file: the AST's Drop is recursive and unbounded
// ===========================================================================

/// **Open bug, found by `sql_parse_is_linear_enough_to_finish` on its first
/// run.** `MAX_DEPTH = 200` in `sql::parser` bounds the parser's *descent*.
/// Left-associative productions -- `+ - * / %`, `AND`, `OR`, the comparisons,
/// `UNION [ALL]` -- are parsed by a loop, not by descent, so they never touch
/// the counter. `parse` therefore returns `Ok` with an `Expr`/`SetExpr` chain
/// as deep as the input is long, and the compiler-generated `Drop` for that
/// chain is recursive. Freeing it overflows the stack: a `SIGSEGV` the caller
/// cannot catch, on the path where it has *already decided to throw the query
/// away*.
///
/// Measured on this machine, `n` = number of operators, abort = stack overflow
/// while freeing the `Ok` the parser just returned:
///
/// | shape | profile | stack | survives | aborts |
/// |-------|---------|-------|----------|--------|
/// | `SELECT ` + `"1 + "*n` + `1` | test (`opt-level=2`) | 2 MiB | 25,000 | 30,000 |
/// | `SELECT ` + `"1 + "*n` + `1` | test (`opt-level=2`) | 8 MiB | 100,000 | 150,000 |
/// | `SELECT 1` + `" UNION ALL SELECT 1"*n` | test | 2 MiB | 16,000 | 20,000 |
/// | `SELECT 1` + `" UNION ALL SELECT 1"*n` | release | 2 MiB | 18,000 | 20,000 |
///
/// Linear, no ceiling: ~70 bytes of stack per `Expr` level and ~120 per
/// `SetExpr` level, and the constant moves with the optimization level, so
/// there is no input length that is safe by construction. `mem::forget` on the
/// parsed statement survives n = 200,000 on a 2 MiB stack, which is what
/// isolates it to `Drop` rather than to the parser. `Session::query` inherits
/// it: the binder rejects the query with `NOT_IMPLEMENTED` and then aborts the
/// process freeing the AST it just rejected.
///
/// Minimal reproducer:
///
/// ```text
/// std::thread::spawn(|| {                                  // 2 MiB stack
///     let sql = "SELECT ".to_string() + &"1 + ".repeat(30_000) + "1";
///     let ast = granular::sql::parse(&sql).unwrap();       // Ok
///     drop(ast);                                           // stack overflow
/// }).join();
/// ```
///
/// The fix belongs in `src/sql/ast.rs` (a hand-written iterative `Drop` that
/// walks the chain into a worklist) or in `src/sql/parser.rs` (count the
/// iterative loops against `MAX_DEPTH` too, which caps the tree at 200 and
/// costs nothing). Not made here: this file does not own those.
///
/// What this test can assert without aborting the run is the boundary as it
/// stands -- the shapes that are safe today must stay safe, and the guard that
/// *does* work must keep working. Raise the constants when the `Drop` is
/// fixed; the test is written so that is the only edit needed.
#[test]
fn sql_ast_drop_recurses_once_per_operator() {
    // 2 MiB exactly: the default this bug is actually dangerous on, not the
    // 8 MiB main thread that hides it. Deliberately *not* `on_big_stack`.
    let h = std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(|| {
            // A quarter of the worst measured cliff (16,000, for `UNION ALL`
            // at `opt-level=2`). The margin is this wide on purpose: the
            // per-level cost is an optimizer artefact and moved by 25% between
            // `--release` and the test profile during this investigation, so a
            // constant chosen just under a measured cliff is a test that
            // aborts the whole binary on somebody else's toolchain.
            const SAFE: usize = 4_000;
            for chain in [
                "SELECT ".to_string() + &"1 + ".repeat(SAFE) + "1",
                "SELECT ".to_string() + &"1 AND ".repeat(SAFE) + "1",
                "SELECT ".to_string() + &"1 OR ".repeat(SAFE) + "1",
                "SELECT ".to_string() + &"1 = ".repeat(SAFE) + "1",
                "SELECT 1".to_string() + &" UNION ALL SELECT 1".repeat(SAFE),
            ] {
                // Parsing *and* freeing. Surviving the `drop` is the assertion;
                // an overflow is an abort, not a failed `assert!`.
                let ast = granular::sql::parse(&chain).expect("a flat chain must parse");
                assert_eq!(ast.len(), 1);
                drop(ast);
            }

            // The descent guard is unaffected and must stay that way: nesting
            // that *does* recurse is still refused long before any stack is at
            // risk, and it is refused rather than silently truncated.
            let nested = "SELECT ".to_string() + &"(".repeat(100_000) + "1";
            let e = granular::sql::parse(&nested).expect_err("100k parens must be refused");
            assert!(e.to_string().contains("nested more than"), "wrong rejection: {e}");
        })
        .expect("2 MiB probe thread");
    h.join().expect("a 20,000-term operator chain must parse and free on a 2 MiB stack");
}
