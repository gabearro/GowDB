//! A persistent worker pool for data-parallel query execution.
//!
//! Part construction (see [`crate::storage::Part::build_sel`]) spawns threads
//! with `std::thread::scope` and pays ~15us of thread creation per flush. That
//! is invisible next to the megabytes of packing it wraps. A scan cannot make
//! the same trade: fanning out per query would put those 15us **on every
//! call**, and a point-ish query that touches four granules costs less than
//! that in total. So the threads have to outlive the query, and `for_each` is
//! a rendezvous with threads that already exist rather than a spawn.
//!
//! ```text
//!   for_each(n, f):
//!
//!     caller ---- publish job ----> [ shared job list ]
//!            |                        ^        ^
//!            |                        |        |
//!            +-- drains indices ------+        +-- workers drain indices
//!                                                  (parked on a condvar
//!                                                   until a job appears)
//!         .. caller returns only after every index has been run ..
//! ```
//!
//! ## The caller works too
//!
//! A pool of N spawns N-1 threads and runs the Nth share on the calling
//! thread. Two reasons: N-way parallelism is what the caller asked for, not
//! N+1-way oversubscription of the machine; and a 1-thread pool then needs no
//! threads at all, so `GRANULAR_THREADS=1` is a genuinely serial engine with
//! no synchronization on any path (useful when a benchmark or a bug hunt
//! wants the scheduler out of the picture).
//!
//! ## Indices, not ranges
//!
//! Work is handed out one index at a time from a single `AtomicUsize`. Static
//! partitioning would be one fewer atomic per item, but granule costs are wildly
//! uneven -- a granule whose zone map lets the filter reject it wholesale costs
//! ~0, the one next to it decodes 1024 rows through a dictionary -- so a static
//! split leaves threads idle at the tail while one thread finishes the
//! expensive quarter. One `fetch_add` per item (a few ns, uncontended most of
//! the time) buys automatic load balance. The counter is per job, so nothing
//! needs resetting between calls.
//!
//! ## Nested calls run inline
//!
//! A job body that itself calls `for_each` runs its indices serially on the
//! current thread. The alternative -- letting a worker block waiting for its
//! nested job to be picked up -- can deadlock the moment every thread is
//! blocked inside a nested submission, and the fix (a real continuation-based
//! scheduler) is not worth it: the engine's nesting is scan-inside-scan, where
//! the outer level already saturates the pool. Inline is always correct and
//! never worse than the parallelism already in flight.
//!
//! ## Panics
//!
//! Each item runs under `catch_unwind`. A panicking item does not stop the
//! job, does not poison the pool, and does not strand the waiter: the payload
//! is stashed on the job and re-raised on the calling thread once all work has
//! finished, so a panic in a scan surfaces to the query that caused it and the
//! pool stays usable afterwards. (Under `panic = "abort"`, which the release
//! profile sets, the catch is inert and the process aborts -- that is the
//! crate's chosen policy, not something this module can override.)
//!
//! ## Why the borrow is sound
//!
//! `f` and everything it captures live on the caller's stack; the workers are
//! `'static`. The job descriptor is published to the workers with its lifetime
//! transmuted away, which is sound because of one invariant, maintained under
//! the shared mutex:
//!
//!   1. a worker may only obtain a `&Job` while holding the mutex, and it
//!      increments `job.active` **before** releasing it;
//!   2. `for_each` removes the job from the shared list, and then waits for
//!      `job.active == 0`, both under that same mutex, before returning.
//!
//! (1) means there is no instant at which a worker holds a reference without
//! having announced it; (2) means the frame that owns the `Job` outlives every
//! announced reference. `std::thread::scope` proves the same thing by joining;
//! this proves it by counting, because the threads must not be joined.

use std::any::Any;
use std::cell::Cell;
use std::mem;
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::{self, JoinHandle};

/// Thread count override, e.g. `GRANULAR_THREADS=1 cargo bench`.
pub const THREADS_ENV: &str = "GRANULAR_THREADS";

/// Sanity clamp on the env override so a typo cannot try to spawn 10^6
/// threads and take the process down with it.
const MAX_THREADS: usize = 1024;

/// The process-wide pool, started on first use.
///
/// Threads are spawned when this is first called and live until the process
/// exits (a `static` is never dropped, which is exactly what we want -- a pool
/// torn down at exit would only add shutdown latency).
pub fn global() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let raw = std::env::var(THREADS_ENV).ok();
        Pool::with_threads(resolve_threads(raw.as_deref()))
    })
}

/// `GRANULAR_THREADS` if it parses to something sane, else the machine's
/// parallelism, else 1. Split out from `global` so it is testable without
/// mutating process environment from a test thread.
fn resolve_threads(raw: Option<&str>) -> usize {
    if let Some(v) = raw {
        if let Ok(n) = v.trim().parse::<usize>() {
            if n > 0 {
                return n.min(MAX_THREADS);
            }
        }
    }
    thread::available_parallelism().map(|p| p.get()).unwrap_or(1).min(MAX_THREADS)
}

// ---------------------------------------------------------------------------
// Job
// ---------------------------------------------------------------------------

/// The erased job body. `Sync` is the load-bearing bound: several threads call
/// it at once.
type Body<'a> = dyn Fn(usize) + Send + Sync + 'a;

/// One `for_each` in flight. Lives on the submitting thread's stack.
struct Job<'a> {
    f: &'a Body<'a>,
    n: usize,
    /// Next index to claim. Overshoots `n` by at most one per participant.
    next: AtomicUsize,
    /// Workers currently inside `drain` for this job. The submitter is not
    /// counted -- it is the one doing the waiting.
    active: AtomicUsize,
    /// Written only when an item panics, so the happy path never allocates
    /// and never touches this lock.
    failure: Mutex<Option<Failure>>,
}

/// What a panicking job leaves behind.
struct Failure {
    /// The first payload seen; later ones are dropped (there is only one
    /// calling thread to re-raise on).
    payload: Box<dyn Any + Send>,
    /// Every index whose closure panicked, i.e. every result slot `map` must
    /// *not* treat as initialized. Unsorted while collecting.
    failed: Vec<usize>,
}

impl<'a> Job<'a> {
    fn new(n: usize, f: &'a Body<'a>) -> Self {
        Job {
            f,
            n,
            next: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            failure: Mutex::new(None),
        }
    }

    /// Claim and run indices until the job is exhausted.
    ///
    /// `Relaxed` is enough for the counter: uniqueness comes from the atomicity
    /// of the read-modify-write, not from ordering, and everything the closure
    /// writes is published to the submitter by the mutex it takes on the way
    /// out of the pool.
    fn drain(&self) {
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= self.n {
                return;
            }
            // AssertUnwindSafe: the payload is re-raised rather than swallowed,
            // so a caller can never observe a half-updated world without also
            // observing the panic that made it.
            if let Err(p) = panic::catch_unwind(AssertUnwindSafe(|| (self.f)(i))) {
                self.record(i, p);
            }
        }
    }

    #[cold]
    fn record(&self, i: usize, payload: Box<dyn Any + Send>) {
        let mut slot = lock(&self.failure);
        match &mut *slot {
            Some(f) => f.failed.push(i),
            None => *slot = Some(Failure { payload, failed: vec![i] }),
        }
    }

    /// Takes by shared reference rather than by value: the laundered
    /// `&'static Job` is still in scope at the call site, and moving the job
    /// out from under it would invalidate a reference the borrow checker
    /// cannot see.
    fn take_failure(&self) -> Option<Failure> {
        lock(&self.failure).take()
    }
}

// ---------------------------------------------------------------------------
// Pool
// ---------------------------------------------------------------------------

struct State {
    /// Jobs in flight, oldest first. Entries point into the stack frames of
    /// the threads that submitted them; see the module header for why that is
    /// sound. Capacity is retained across jobs, so publishing does not
    /// allocate after the first few calls.
    jobs: Vec<&'static Job<'static>>,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    /// Workers park here when there is nothing to do. Parking rather than
    /// spinning is the whole point: a spinning worker steals a core from the
    /// caller's own share of the same query.
    work: Condvar,
    /// Submitters park here waiting for their helpers to drop out.
    done: Condvar,
    /// Revolutions of the worker loop that found nothing to run. Ticked once
    /// per park by the implementation below, once per *revolution* by a
    /// hypothetical spinning one -- which is the entire difference the
    /// idleness test is looking for, expressed as a counter instead of as a
    /// CPU-time reading. Per pool, so the rest of the test binary cannot
    /// pollute it the way it pollutes `getrusage(RUSAGE_SELF)`.
    #[cfg(test)]
    idle_polls: AtomicUsize,
}

pub struct Pool {
    shared: Arc<Shared>,
    /// `threads() - 1` entries: the calling thread is the missing one.
    workers: Vec<JoinHandle<()>>,
}

impl Pool {
    /// A private pool with `threads`-way parallelism (clamped to `1..=1024`).
    /// Prefer [`global`]; this exists for benchmarks and tests that need a
    /// known width.
    pub fn with_threads(threads: usize) -> Pool {
        let threads = threads.clamp(1, MAX_THREADS);
        let shared = Arc::new(Shared {
            state: Mutex::new(State { jobs: Vec::with_capacity(8), shutdown: false }),
            work: Condvar::new(),
            done: Condvar::new(),
            #[cfg(test)]
            idle_polls: AtomicUsize::new(0),
        });
        let mut workers = Vec::with_capacity(threads - 1);
        for i in 0..threads - 1 {
            let sh = Arc::clone(&shared);
            // A refused spawn (thread limit, no memory) degrades the pool
            // rather than killing the process: fewer threads is still correct.
            match thread::Builder::new()
                .name(format!("granular-pool-{i}"))
                .spawn(move || {
                    // Identity for the work-distribution tests; see `SLOT`.
                    #[cfg(test)]
                    SLOT.with(|c| c.set(i + 1));
                    worker(&sh)
                })
            {
                Ok(h) => workers.push(h),
                Err(_) => break,
            }
        }
        Pool { shared, workers }
    }

    /// Width of the pool, counting the calling thread.
    #[inline]
    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Idle revolutions of this pool's worker loops so far; see
    /// [`Shared::idle_polls`].
    #[cfg(test)]
    fn idle_polls(&self) -> usize {
        self.shared.idle_polls.load(Ordering::Relaxed)
    }

    /// Run `f(i)` for i in 0..n across the pool, returning when all are done.
    ///
    /// Runs work on the calling thread too. If any invocation panics, all
    /// indices still run and the first payload is re-raised here.
    pub fn for_each<F>(&self, n: usize, f: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        if let Some(fail) = self.run(n, &f) {
            panic::resume_unwind(fail.payload);
        }
    }

    /// Same, collecting a result per index. `out[i] == f(i)`, always.
    pub fn map<T, F>(&self, n: usize, f: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize) -> T + Send + Sync,
    {
        // One allocation for the whole call: workers write their results
        // straight into the caller's final buffer at disjoint offsets, so
        // there is no per-thread staging vector and no merge pass.
        let mut out: Vec<T> = Vec::with_capacity(n);
        let base = SendPtr(out.as_mut_ptr());
        // `base.slot(i)` and not `base.0.add(i)`: edition-2021 closures capture
        // the *field*, and a bare `*mut T` is neither Send nor Sync.
        let body = move |i: usize| {
            let v = f(i);
            // SAFETY: index `i` is claimed by exactly one participant (single
            // fetch_add counter), `i < n <= capacity`, and the slot is
            // uninitialized, so this write neither races nor overwrites a live
            // value.
            unsafe { base.slot(i).write(v) };
        };

        match self.run(n, &body) {
            None => {
                // SAFETY: `run` executed every index in 0..n to completion and
                // reported no failure, so every slot below `n` was written.
                unsafe { out.set_len(n) };
                out
            }
            Some(mut fail) => {
                // Exactly the panicking indices are uninitialized; everything
                // else must be dropped by hand, since `out` is still empty and
                // will only free the buffer.
                fail.failed.sort_unstable();
                if mem::needs_drop::<T>() {
                    for i in 0..n {
                        if fail.failed.binary_search(&i).is_err() {
                            // SAFETY: slot `i` was written by a completed item
                            // and has not been dropped or moved out of.
                            unsafe { ptr::drop_in_place(base.slot(i)) };
                        }
                    }
                }
                drop(out);
                panic::resume_unwind(fail.payload)
            }
        }
    }

    /// The shared body of `for_each`/`map`: run every index, report the first
    /// panic instead of raising it (so `map` can clean up its buffer first).
    fn run(&self, n: usize, f: &Body<'_>) -> Option<Failure> {
        if n == 0 {
            return None;
        }
        let job = Job::new(n, f);

        // Serial paths, in order of how often they hit: a single item is not
        // worth a wakeup, a 1-wide pool has nobody to wake, and a nested call
        // must not wait on threads that may all be inside this same job.
        if n == 1 || self.workers.is_empty() || in_pool() {
            let _nested = InPool::enter();
            job.drain();
            return job.take_failure();
        }

        // SAFETY: laundering `&Job<'_>` to `&'static Job<'static>` so it can be
        // handed to threads that are not scoped. Sound because this frame does
        // not return while the pointer is reachable: it is removed from
        // `state.jobs` under the mutex (no worker can newly acquire it), and
        // then `active` is waited to zero under the same mutex (every worker
        // that did acquire it announced itself under that mutex before using
        // it, and clears its count under the mutex after its last use). Both
        // conditions hold before the `Job` -- and therefore the caller's `f`
        // and its captures -- goes out of scope.
        let handle: &'static Job<'static> =
            unsafe { mem::transmute::<&Job<'_>, &'static Job<'static>>(&job) };
        lock(&self.shared.state).jobs.push(handle);

        // Wake at most as many workers as there is work for. `notify_one` in a
        // loop rather than `notify_all`: with thousands of tiny jobs a
        // broadcast wakes the whole pool to discover there are two items.
        let wake = (n - 1).min(self.workers.len());
        for _ in 0..wake {
            self.shared.work.notify_one();
        }

        // Retirement runs from a `Drop`, not from straight-line code at the
        // end of this function. `drain` catches panics from `f`, so the only
        // way out of it is normal return -- *almost*: a panic payload whose
        // own `Drop` panics unwinds past the catch. If that unwound through
        // here, this frame would free the `Job` while workers still held the
        // laundered `&'static` pointer to it, which is a use-after-free rather
        // than merely a lost result. Tying retirement to scope exit makes the
        // unwind path identical to the normal one.
        struct Retire<'a> {
            shared: &'a Shared,
            job: &'static Job<'static>,
        }
        impl Drop for Retire<'_> {
            fn drop(&mut self) {
                let mut st = lock(&self.shared.state);
                let me = self.job as *const Job<'static>;
                // Removed under the lock first, so no worker can newly acquire
                // it; then wait out the ones that already did.
                st.jobs.retain(|j| !ptr::eq(*j as *const Job<'static>, me));
                while self.job.active.load(Ordering::Relaxed) != 0 {
                    st = self.shared.done.wait(st).unwrap_or_else(|e| e.into_inner());
                }
            }
        }
        let retire = Retire { shared: &self.shared, job: handle };

        // The caller's own share. Marked as in-pool so a nested `for_each`
        // from inside `f` takes the serial path above.
        {
            let _nested = InPool::enter();
            job.drain();
        }

        drop(retire);
        job.take_failure()
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        lock(&self.shared.state).shutdown = true;
        self.shared.work.notify_all();
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

/// Worker main loop. Never spins: it either has a job or it is parked.
fn worker(shared: &Arc<Shared>) {
    // Everything this thread ever runs is a job body, so any `for_each` it
    // makes is nested by definition.
    IN_POOL.with(|c| c.set(true));

    let mut st = lock(&shared.state);
    loop {
        if st.shutdown {
            return;
        }
        // Oldest job with work left. FIFO keeps the query that has been
        // waiting longest from being starved by a stream of newer ones.
        let job = st.jobs.iter().copied().find(|j| j.next.load(Ordering::Relaxed) < j.n);
        match job {
            Some(job) => {
                // Announced before the lock is released -- this is invariant
                // (1) from the module header, and the reason the submitter's
                // `active == 0` means "nobody holds a reference" rather than
                // "nobody held one a moment ago".
                job.active.fetch_add(1, Ordering::Relaxed);
                drop(st);

                // Symmetric to `Retire` on the submitter side: if `drain`
                // unwinds -- which only a panicking panic-payload `Drop` can
                // cause -- the decrement still has to happen, or the submitter
                // waits on `active` forever and the whole pool wedges.
                struct Clear<'a> {
                    shared: &'a Shared,
                    job: &'static Job<'static>,
                }
                impl Drop for Clear<'_> {
                    fn drop(&mut self) {
                        let _st = lock(&self.shared.state);
                        self.job.active.fetch_sub(1, Ordering::Relaxed);
                        // `job` is dangling from here on: the submitter may
                        // return the instant this lock is released.
                        self.shared.done.notify_all();
                    }
                }
                {
                    let _clear = Clear { shared: &shared, job };
                    job.drain();
                }

                st = lock(&shared.state);
            }
            // No work: park. The mutex is released atomically with the wait,
            // so a job published between the scan and here cannot be missed.
            None => {
                #[cfg(test)]
                shared.idle_polls.fetch_add(1, Ordering::Relaxed);
                st = shared.work.wait(st).unwrap_or_else(|e| e.into_inner());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

thread_local! {
    /// Set while this thread is executing job items. Process-wide rather than
    /// per-pool: a job on one pool submitting to another is just as capable of
    /// deadlocking as a job submitting to its own.
    static IN_POOL: Cell<bool> = const { Cell::new(false) };
}

#[inline]
fn in_pool() -> bool {
    IN_POOL.with(|c| c.get())
}

#[cfg(test)]
thread_local! {
    /// Which participant this thread is, for a job it helps run: 0 on the
    /// submitting thread, `i + 1` on worker `i`. Set once at worker start,
    /// never cleared -- a thread belongs to at most one pool, and a thread
    /// that is a worker somewhere can only ever submit down the serial path
    /// (`in_pool`), so a slot is unambiguous for the job that reads it.
    ///
    /// Exists so a test can assert *who ran what* -- the property the
    /// speedup-ratio tests were reaching for -- without measuring anything.
    static SLOT: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
#[inline]
fn slot() -> usize {
    SLOT.with(|c| c.get())
}

/// Sets the in-pool flag for a scope, restoring the previous value (nested
/// entries on the calling thread must not clear it on the way out).
struct InPool(bool);

impl InPool {
    #[inline]
    fn enter() -> InPool {
        InPool(IN_POOL.with(|c| c.replace(true)))
    }
}

impl Drop for InPool {
    #[inline]
    fn drop(&mut self) {
        IN_POOL.with(|c| c.set(self.0));
    }
}

/// Poisoning is not a failure mode here: user panics are caught inside
/// `drain`, well away from any lock, and the pool's own invariants are plain
/// integers. Take the data back either way rather than cascading a panic into
/// every later query.
#[inline]
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A raw pointer that may cross into the workers.
struct SendPtr<T>(*mut T);

// Hand-written, because `derive` would demand `T: Copy` on the pointee.
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// # Safety
    /// `i` must be within the allocation the pointer came from.
    #[inline]
    unsafe fn slot(&self, i: usize) -> *mut T {
        self.0.add(i)
    }
}

// SAFETY: the pointer targets a buffer owned by the submitting frame, which
// outlives the job (see `Pool::run`), and each index writes a disjoint slot.
// `T: Send` is what makes moving the produced values across threads legal.
unsafe impl<T: Send> Send for SendPtr<T> {}
unsafe impl<T: Send> Sync for SendPtr<T> {}

/// Nothing in here asserts on elapsed time, and nothing sleeps.
///
/// It used to. Two of these tests timed a job on a 1-wide pool against the
/// same job on a 4-wide one and asserted a ratio, and several arranged overlap
/// by sleeping long enough that a wakeup could not possibly compete. Both
/// idioms encode an assumption about how fast the machine is, which is exactly
/// the assumption coverage, Miri and the sanitizers break: they dilate
/// execution 5-50x and unevenly, so the ratios stop holding and the sleeps stop
/// dominating. The properties themselves have nothing to do with time --
/// "four items ran on four threads", "the expensive prefix did not all land on
/// one worker", "an idle worker is not spinning" -- so they are stated as
/// facts about counters and rendezvous, which mean the same thing at any
/// execution speed.
///
/// What is left of the clock: [`STUCK`], a deadline that only fires when the
/// pool has genuinely wedged (turning a hang into a failed assertion instead of
/// a hung harness), and the one test in [`tests::wallclock`], which is a
/// budget and says so.
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
    use std::thread::ThreadId;
    use std::time::{Duration, Instant};

    fn pool(n: usize) -> Pool {
        Pool::with_threads(n)
    }

    /// Iteration counts for the loops that exist only to shake races.
    ///
    /// Miri explores interleavings systematically rather than statistically and
    /// runs two orders of magnitude slower, so the full counts would turn
    /// `cargo miri test` into an overnight job while finding nothing the first
    /// couple of dozen rounds did not. Outside Miri this is the identity.
    fn reps(n: usize) -> usize {
        if cfg!(miri) {
            n.min(24)
        } else {
            n
        }
    }

    /// How long a rendezvous waits before declaring the pool wedged.
    ///
    /// Not a budget: no working pool comes anywhere near it, and no failing one
    /// ever finishes. It exists so a lost wakeup fails the test instead of
    /// hanging the harness, which means it only has to sit above the worst
    /// plausible scheduling delay -- 50x slower under a sanitizer is still 50x
    /// below this.
    const STUCK: Duration = Duration::from_secs(60);

    /// A one-shot latch. `wait` blocks until some other thread calls `open`.
    struct Gate {
        open: Mutex<bool>,
        go: Condvar,
    }

    impl Gate {
        fn new() -> Gate {
            Gate { open: Mutex::new(false), go: Condvar::new() }
        }

        fn open(&self) {
            *lock(&self.open) = true;
            self.go.notify_all();
        }

        /// False if it never opened; see [`STUCK`].
        fn wait(&self) -> bool {
            let mut open = lock(&self.open);
            let deadline = Instant::now() + STUCK;
            while !*open {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return false;
                }
                open = self.go.wait_timeout(open, left).unwrap_or_else(|e| e.into_inner()).0;
            }
            true
        }
    }

    /// An n-way meeting point: every arriving thread blocks until `want` of
    /// them are inside at once.
    ///
    /// This is what replaced the `work(30)` sleeps. A sleep only makes overlap
    /// *likely*, and only on a machine where the sleep outlasts a wakeup;
    /// blocking until the other participants show up makes the overlap a
    /// precondition for the job to finish at all. So "the pool ran these four
    /// items in parallel" stops being a measurement and becomes a fact: either
    /// four threads claimed them, or `arrive` reports that they did not.
    ///
    /// One generation is enough: once `want` have arrived the meeting stays
    /// open, so a job with more items than participants rendezvouses on its
    /// first `want` and streams the rest.
    struct Meet {
        want: usize,
        give_up: Duration,
        arrived: Mutex<usize>,
        go: Condvar,
    }

    impl Meet {
        fn new(want: usize) -> Meet {
            Meet::with_deadline(want, STUCK)
        }

        /// Only for the vacuity guard below, which *expects* the meeting to
        /// fail and would otherwise sit out the full [`STUCK`] to find out.
        fn with_deadline(want: usize, give_up: Duration) -> Meet {
            Meet { want, give_up, arrived: Mutex::new(0), go: Condvar::new() }
        }

        /// Blocks until `want` threads are inside. False = they never came,
        /// i.e. the pool handed `want` items to fewer than `want` threads.
        fn arrive(&self) -> bool {
            let mut n = lock(&self.arrived);
            *n += 1;
            if *n >= self.want {
                self.go.notify_all();
                return true;
            }
            let deadline = Instant::now() + self.give_up;
            while *n < self.want {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return false;
                }
                n = self.go.wait_timeout(n, left).unwrap_or_else(|e| e.into_inner()).0;
            }
            true
        }
    }

    /// Per-participant claim counter: slot 0 is the submitting thread, slot
    /// `i + 1` is worker `i` (see [`super::SLOT`]).
    ///
    /// The counters live here rather than inside `Job` so the hot loop keeps
    /// exactly the instructions it has today -- `drain` is the same code in a
    /// test build as in a release one, which is worth more than saving each
    /// test a `claim()` call.
    struct Claims(Vec<AtomicUsize>);

    impl Claims {
        fn new(p: &Pool) -> Claims {
            Claims((0..p.threads()).map(|_| AtomicUsize::new(0)).collect())
        }

        /// Record that the running participant claimed one item. Silently
        /// skips a participant with no slot here, which is only possible for a
        /// worker of some *other* pool running a nested job inline.
        fn claim(&self) {
            if let Some(c) = self.0.get(slot()) {
                c.fetch_add(1, SeqCst);
            }
        }

        fn counts(&self) -> Vec<usize> {
            self.0.iter().map(|c| c.load(SeqCst)).collect()
        }

        /// Participants that claimed at least one item.
        fn participants(&self) -> usize {
            self.0.iter().filter(|c| c.load(SeqCst) > 0).count()
        }

        fn total(&self) -> usize {
            self.0.iter().map(|c| c.load(SeqCst)).sum()
        }
    }

    // -- shape ------------------------------------------------------------

    #[test]
    fn threads_counts_the_caller() {
        for n in [1, 2, 3, 8] {
            let p = pool(n);
            assert_eq!(p.threads(), n, "a pool of {n} must give {n}-way parallelism");
            assert_eq!(p.workers.len(), n - 1, "the caller is the missing thread");
        }
    }

    #[test]
    fn single_thread_pool_spawns_nothing() {
        let p = pool(1);
        assert!(p.workers.is_empty());
        let hits = AtomicUsize::new(0);
        let me = thread::current().id();
        p.for_each(64, |_| {
            assert_eq!(thread::current().id(), me, "a 1-wide pool must stay on the caller");
            hits.fetch_add(1, SeqCst);
        });
        assert_eq!(hits.load(SeqCst), 64);
    }

    #[test]
    fn zero_threads_is_clamped_to_one() {
        let p = pool(0);
        assert_eq!(p.threads(), 1);
        assert_eq!(p.map(4, |i| i), vec![0, 1, 2, 3]);
    }

    // -- degenerate n ------------------------------------------------------

    #[test]
    fn n_zero_runs_nothing() {
        let p = pool(4);
        let hits = AtomicUsize::new(0);
        p.for_each(0, |_| {
            hits.fetch_add(1, SeqCst);
        });
        assert_eq!(hits.load(SeqCst), 0);
    }

    #[test]
    fn n_zero_map_is_empty() {
        let p = pool(4);
        let out: Vec<usize> = p.map(0, |i| i);
        assert!(out.is_empty());
    }

    #[test]
    fn n_one_runs_once_on_the_caller() {
        let p = pool(4);
        let me = thread::current().id();
        let seen = Mutex::new(Vec::new());
        p.for_each(1, |i| seen.lock().unwrap().push((i, thread::current().id())));
        assert_eq!(*seen.lock().unwrap(), vec![(0, me)]);
    }

    #[test]
    fn n_one_map() {
        let p = pool(4);
        assert_eq!(p.map(1, |i| i * 7), vec![0]);
    }

    // -- coverage ----------------------------------------------------------

    #[test]
    fn every_index_runs_exactly_once() {
        let p = pool(4);
        let n = reps(10_000);
        let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        p.for_each(n, |i| {
            counts[i].fetch_add(1, SeqCst);
        });
        for (i, c) in counts.iter().enumerate() {
            assert_eq!(c.load(SeqCst), 1, "index {i} ran {} times", c.load(SeqCst));
        }
    }

    #[test]
    fn n_far_larger_than_the_pool() {
        let p = pool(3);
        let sum = AtomicUsize::new(0);
        let n = reps(100_000);
        p.for_each(n, |i| {
            sum.fetch_add(i, SeqCst);
        });
        assert_eq!(sum.load(SeqCst), (0..n).sum::<usize>());
    }

    #[test]
    fn every_width_covers_every_index() {
        // The interesting boundaries are n == 1 (serial path) and n <= threads
        // (fewer items than helpers to wake).
        for threads in 1..=6 {
            let p = pool(threads);
            for n in 0..reps(40) {
                let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                p.for_each(n, |i| {
                    counts[i].fetch_add(1, SeqCst);
                });
                assert!(
                    counts.iter().all(|c| c.load(SeqCst) == 1),
                    "threads={threads} n={n} left an index unvisited or doubled"
                );
            }
        }
    }

    #[test]
    fn consecutive_jobs_each_start_from_zero() {
        // The claim counter is per job; a leaked counter would silently skip
        // the whole second job.
        let p = pool(4);
        for round in 0..reps(50) {
            let counts: Vec<AtomicUsize> = (0..64).map(|_| AtomicUsize::new(0)).collect();
            p.for_each(64, |i| {
                counts[i].fetch_add(1, SeqCst);
            });
            assert!(
                counts.iter().all(|c| c.load(SeqCst) == 1),
                "round {round} did not cover 0..64"
            );
        }
    }

    // -- map ---------------------------------------------------------------

    #[test]
    fn map_preserves_index_order() {
        let p = pool(4);
        let n = reps(1000);
        let out = p.map(n, |i| i * i);
        assert_eq!(out.len(), n);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, i * i, "slot {i} holds another index's result");
        }
    }

    #[test]
    fn map_of_owned_values() {
        // Non-Copy, heap-owning, drop-having: exercises the move across
        // threads into the caller's buffer.
        let p = pool(4);
        let out = p.map(500, |i| format!("row-{i}"));
        assert_eq!(out.len(), 500);
        assert_eq!(out[0], "row-0");
        assert_eq!(out[499], "row-499");
        for (i, s) in out.iter().enumerate() {
            assert_eq!(s, &format!("row-{i}"));
        }
    }

    #[test]
    fn map_of_boxes() {
        let p = pool(4);
        let out = p.map(256, |i| Box::new(i as u64));
        assert_eq!(out.iter().map(|b| **b).sum::<u64>(), (0..256u64).sum::<u64>());
    }

    #[test]
    fn map_of_zero_sized_values() {
        let p = pool(4);
        let out = p.map(1000, |_| ());
        assert_eq!(out.len(), 1000);
    }

    #[test]
    fn map_result_has_exact_len() {
        let p = pool(4);
        let out = p.map(37, |i| i as u8);
        assert_eq!(out.len(), 37);
        assert_eq!(out.capacity(), 37, "map must not over-allocate the result");
    }

    #[test]
    fn map_from_a_single_wide_pool_matches_a_wide_one() {
        let serial = pool(1).map(300, |i| i * 3 + 1);
        let parallel = pool(6).map(300, |i| i * 3 + 1);
        assert_eq!(serial, parallel);
    }

    // -- borrowing ---------------------------------------------------------

    #[test]
    fn closures_may_borrow_the_callers_stack() {
        // The whole point of the lifetime laundering: `input` and `out` are
        // locals, and the workers touch both.
        let p = pool(4);
        let input: Vec<u64> = (0..2048).map(|i| i * 3).collect();
        let out = p.map(input.len(), |i| input[i] + 1);
        assert_eq!(out.first(), Some(&1));
        assert_eq!(out.last(), Some(&(2047 * 3 + 1)));

        let total = AtomicUsize::new(0);
        p.for_each(input.len(), |i| {
            total.fetch_add(input[i] as usize, SeqCst);
        });
        assert_eq!(total.load(SeqCst), input.iter().sum::<u64>() as usize);
    }

    #[test]
    fn borrowed_state_is_visible_after_the_call_returns() {
        // Publication in the other direction: writes made by workers must be
        // visible to the caller the moment `for_each` returns.
        let p = pool(4);
        for _ in 0..reps(200) {
            let cells: Vec<AtomicUsize> = (0..64).map(|_| AtomicUsize::new(0)).collect();
            p.for_each(64, |i| cells[i].store(i + 1, Ordering::Relaxed));
            for (i, c) in cells.iter().enumerate() {
                assert_eq!(c.load(Ordering::Relaxed), i + 1);
            }
        }
    }

    // -- the instrumentation itself ----------------------------------------

    #[test]
    fn the_rendezvous_notices_when_overlap_is_impossible() {
        // Vacuity guard for every `arrive()` assertion in this module: they are
        // only worth anything if the meeting can actually fail. A 1-wide pool
        // runs its items strictly one after another, so a 2-way meeting must
        // report that it never filled -- if `arrive` were unconditionally true,
        // the distribution tests would pass on a pool with no parallelism at
        // all, which is exactly the failure mode they exist to catch.
        //
        // The short deadline is sound *here specifically*: the expected result
        // is the timeout, and no amount of slowing the machine down can make a
        // serial pool overlap, so a slow host makes this test slower and never
        // wrong.
        // Note which way round the signal runs: the *last* thread to arrive
        // always succeeds, since by then the meeting is full -- serially, that
        // is item 1 arriving long after item 0 gave up. So overlap means "no
        // participant was stranded", which is what the tests above assert, and
        // this one asserts the converse can happen.
        let p = pool(1);
        let met = Meet::with_deadline(2, Duration::from_millis(50));
        let stranded = AtomicBool::new(false);
        p.for_each(2, |_| {
            if !met.arrive() {
                stranded.store(true, SeqCst);
            }
        });
        assert!(stranded.load(SeqCst), "a 1-wide pool cannot run two items at once");
    }

    #[test]
    fn the_idle_counter_ticks_when_workers_park() {
        // Vacuity guard for `idle_workers_do_not_spin`: that test asserts a
        // *small* delta, so a counter that never moved would satisfy it while
        // the pool spun itself hoarse. Workers run out of work here by
        // construction, so every one of them must eventually record it.
        let p = pool(4);
        let workers = p.threads() - 1;
        if workers == 0 {
            return;
        }
        p.for_each(64, |_| {});
        let deadline = Instant::now() + STUCK;
        while p.idle_polls() < workers && Instant::now() < deadline {
            thread::yield_now();
        }
        assert!(
            p.idle_polls() >= workers,
            "{workers} workers ran out of work but only {} parks were recorded",
            p.idle_polls()
        );
    }

    // -- participation -----------------------------------------------------

    #[test]
    fn the_caller_and_the_workers_both_run_items() {
        let p = pool(4);
        if p.threads() < 4 {
            return;
        }
        let me = thread::current().id();
        let seen: Mutex<HashSet<ThreadId>> = Mutex::new(HashSet::new());
        let claims = Claims::new(&p);
        // Exactly one item per participant, and each one blocks until all four
        // are running: the caller *cannot* drain the job alone, so "the workers
        // took a share" is enforced by the shape of the job rather than made
        // likely by a sleep.
        let met = Meet::new(4);
        let overlapped = AtomicBool::new(true);
        p.for_each(4, |_| {
            claims.claim();
            seen.lock().unwrap().insert(thread::current().id());
            if !met.arrive() {
                overlapped.store(false, SeqCst);
            }
        });
        assert!(overlapped.load(SeqCst), "4 items on a 4-wide pool ran on fewer than 4 threads");
        let seen = seen.into_inner().unwrap();
        assert!(seen.contains(&me), "the calling thread must take a share");
        assert_eq!(seen.len(), 4, "a pool of 4 must use exactly its 4 threads");
        assert_eq!(claims.counts(), vec![1, 1, 1, 1], "one item each: caller, then workers 0..3");
    }

    #[test]
    fn parallel_work_overlaps() {
        // Was: run the job on a 1-wide pool, run it again on a 4-wide one,
        // assert the second was 2x faster. What that was reaching for is that
        // the items run *at the same time* and that every thread gets some, so
        // that is what this states -- a rendezvous the job cannot get past
        // without four threads inside it, plus the claim counts for the tail.
        let p = pool(4);
        if p.threads() < 4 {
            return;
        }
        const N: usize = 64;
        let claims = Claims::new(&p);
        let met = Meet::new(4);
        let overlapped = AtomicBool::new(true);
        p.for_each(N, |_| {
            claims.claim();
            if !met.arrive() {
                overlapped.store(false, SeqCst);
            }
        });
        assert!(overlapped.load(SeqCst), "no four items ever overlapped");
        assert_eq!(claims.total(), N, "someone lost an item");
        assert_eq!(claims.participants(), 4, "claims landed on {:?}", claims.counts());
    }

    #[test]
    fn uneven_work_does_not_strand_threads() {
        // Four expensive items bunched into the first eighth, the shape a
        // static split handles worst: it hands the whole prefix to one thread.
        // Here "expensive" means "does not return until all four are running",
        // so a static split would not merely be slow, it would never finish --
        // and the claim counts say directly that the prefix reached four
        // different participants, which is the load balance the one-index-at-
        // a-time counter exists to buy.
        let p = pool(4);
        if p.threads() < 4 {
            return;
        }
        const N: usize = 64;
        let prefix = Claims::new(&p);
        let all = Claims::new(&p);
        let met = Meet::new(4);
        let spread = AtomicBool::new(true);
        p.for_each(N, |i| {
            all.claim();
            if i < 4 {
                prefix.claim();
                if !met.arrive() {
                    spread.store(false, SeqCst);
                }
            }
        });
        assert!(spread.load(SeqCst), "the expensive prefix did not reach 4 threads");
        // A thread that has claimed an expensive index is stuck in it until all
        // four are claimed, so the four cannot share a participant.
        assert_eq!(prefix.counts(), vec![1, 1, 1, 1], "one expensive item each");
        assert_eq!(all.total(), N);
        assert_eq!(all.participants(), 4, "cheap tail landed on {:?}", all.counts());
    }

    #[test]
    fn workers_are_reused_not_respawned() {
        // Reuse means the *same* threads serve two jobs. The rendezvous forces
        // every thread in the pool into each job, so each id set is complete
        // and a pool that respawned per job would show two disjoint ones. (The
        // old form asserted 500 jobs finish inside five seconds, which is a
        // measurement of the machine: a spawn is ~15us, so on a loaded box the
        // budget and the signal are the same order of magnitude.)
        let p = pool(4);
        if p.threads() < 4 {
            return;
        }
        let round = |p: &Pool| -> HashSet<(ThreadId, String)> {
            let seen = Mutex::new(HashSet::new());
            let met = Meet::new(4);
            p.for_each(4, |_| {
                let t = thread::current();
                let name = t.name().unwrap_or("caller").to_string();
                seen.lock().unwrap().insert((t.id(), name));
                assert!(met.arrive(), "the pool did not run 4 items on 4 threads");
            });
            seen.into_inner().unwrap()
        };

        let first = round(&p);
        assert_eq!(first.len(), 4);
        assert_eq!(
            first.iter().filter(|(_, n)| n.starts_with("granular-pool-")).count(),
            3,
            "3 of the 4 participants must be the pool's own threads"
        );
        for _ in 0..reps(200) {
            p.for_each(16, |_| {});
        }
        assert_eq!(round(&p), first, "the pool replaced its threads between jobs");
    }

    // -- nesting -----------------------------------------------------------

    #[test]
    fn nested_for_each_does_not_deadlock() {
        let p = pool(4);
        let total = AtomicUsize::new(0);
        p.for_each(16, |_| {
            p.for_each(16, |_| {
                total.fetch_add(1, SeqCst);
            });
        });
        assert_eq!(total.load(SeqCst), 256);
    }

    #[test]
    fn nested_for_each_runs_inline() {
        let p = pool(4);
        p.for_each(8, |_| {
            let outer = thread::current().id();
            p.for_each(8, |_| {
                assert_eq!(
                    thread::current().id(),
                    outer,
                    "a nested job must stay on the thread that submitted it"
                );
            });
        });
    }

    #[test]
    fn deeply_nested_for_each() {
        let p = pool(4);
        let total = AtomicUsize::new(0);
        p.for_each(6, |_| {
            p.for_each(6, |_| {
                p.for_each(6, |_| {
                    total.fetch_add(1, SeqCst);
                });
            });
        });
        assert_eq!(total.load(SeqCst), 216);
    }

    #[test]
    fn nested_map_inside_for_each() {
        let p = pool(4);
        let sums: Vec<AtomicUsize> = (0..16).map(|_| AtomicUsize::new(0)).collect();
        p.for_each(16, |i| {
            let inner = p.map(32, |j| i * j);
            sums[i].store(inner.iter().sum(), SeqCst);
        });
        for i in 0..16 {
            assert_eq!(sums[i].load(SeqCst), i * (0..32).sum::<usize>());
        }
    }

    #[test]
    fn nesting_onto_a_different_pool_is_also_inline() {
        let outer = pool(4);
        let inner = pool(4);
        let hits = AtomicUsize::new(0);
        outer.for_each(8, |_| {
            let me = thread::current().id();
            inner.for_each(8, |_| {
                assert_eq!(thread::current().id(), me);
                hits.fetch_add(1, SeqCst);
            });
        });
        assert_eq!(hits.load(SeqCst), 64);
    }

    #[test]
    fn the_in_pool_flag_is_restored() {
        let p = pool(4);
        assert!(!in_pool());
        p.for_each(8, |_| {
            assert!(in_pool(), "items run with the nested flag set");
        });
        assert!(!in_pool(), "the flag must not leak past the call");
    }

    // -- panics ------------------------------------------------------------

    #[cfg(panic = "unwind")]
    #[test]
    fn a_panicking_item_propagates_to_the_caller() {
        let p = pool(4);
        let r = panic::catch_unwind(AssertUnwindSafe(|| {
            p.for_each(64, |i| {
                if i == 13 {
                    panic!("item {i} exploded");
                }
            });
        }));
        let payload = r.expect_err("the panic must reach the caller");
        let msg = payload.downcast_ref::<String>().expect("payload preserved");
        assert_eq!(msg, "item 13 exploded");
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn all_work_still_runs_before_the_panic_is_raised() {
        let p = pool(4);
        let done = AtomicUsize::new(0);
        let r = panic::catch_unwind(AssertUnwindSafe(|| {
            p.for_each(200, |i| {
                if i % 40 == 7 {
                    panic!("boom");
                }
                done.fetch_add(1, SeqCst);
            });
        }));
        assert!(r.is_err());
        assert_eq!(done.load(SeqCst), 195, "the other 195 items must still have run");
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn the_pool_is_usable_after_a_panic() {
        let p = pool(4);
        for _ in 0..5 {
            let r = panic::catch_unwind(AssertUnwindSafe(|| {
                p.for_each(32, |i| {
                    if i == 0 {
                        panic!("boom");
                    }
                });
            }));
            assert!(r.is_err());
            // Same pool, immediately afterwards: no poisoned mutex, no lost
            // workers, no stuck job in the list.
            let counts: Vec<AtomicUsize> = (0..256).map(|_| AtomicUsize::new(0)).collect();
            p.for_each(256, |i| {
                counts[i].fetch_add(1, SeqCst);
            });
            assert!(counts.iter().all(|c| c.load(SeqCst) == 1));
        }
        assert_eq!(p.map(8, |i| i), vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn a_panic_on_a_worker_does_not_wedge_the_waiter() {
        // Every index panics, so whichever thread runs them, the submitter has
        // to be released by the panic path rather than the normal one. There
        // is no deadline here any more: the old `elapsed() < 10s` could only be
        // evaluated *after* `for_each` returned, so it never had anything to
        // say about the stranding it was named for -- a stranded waiter hangs
        // right here instead. `adv_worker_unwind_strands_the_submitter` is the
        // test that turns that hang into a failure.
        let p = pool(4);
        let r = panic::catch_unwind(AssertUnwindSafe(|| {
            p.for_each(reps(500), |_| panic!("boom"));
        }));
        assert!(r.is_err());
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn a_panic_in_the_serial_path_propagates() {
        let p = pool(1);
        let r = panic::catch_unwind(AssertUnwindSafe(|| p.for_each(4, |_| panic!("serial boom"))));
        assert!(r.is_err());
        let r = panic::catch_unwind(AssertUnwindSafe(|| pool(4).for_each(1, |_| panic!("n=1"))));
        assert!(r.is_err());
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn a_panic_in_a_nested_job_propagates_through_both_levels() {
        let p = pool(4);
        let r = panic::catch_unwind(AssertUnwindSafe(|| {
            p.for_each(8, |_| {
                p.for_each(8, |j| {
                    if j == 3 {
                        panic!("nested boom");
                    }
                });
            });
        }));
        assert!(r.is_err());
        assert_eq!(p.map(4, |i| i), vec![0, 1, 2, 3], "pool still usable");
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn map_panics_without_leaking_or_double_dropping() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        struct Tracked(#[allow(dead_code)] usize);
        impl Drop for Tracked {
            fn drop(&mut self) {
                DROPS.fetch_add(1, SeqCst);
            }
        }

        let p = pool(4);
        let r = panic::catch_unwind(AssertUnwindSafe(|| {
            let _v: Vec<Tracked> = p.map(64, |i| {
                if i == 7 || i == 40 {
                    panic!("no value for {i}");
                }
                Tracked(i)
            });
        }));
        assert!(r.is_err());
        // 62 values were produced and must be dropped exactly once each; the
        // two panicking slots hold nothing and must not be touched.
        assert_eq!(DROPS.load(SeqCst), 62, "leaked or double-dropped map results");
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn map_survives_an_all_panicking_job() {
        let p = pool(4);
        let r = panic::catch_unwind(AssertUnwindSafe(|| {
            let _v: Vec<String> = p.map(100, |i| panic!("{i}"));
        }));
        assert!(r.is_err());
        assert_eq!(p.map(3, |i| i.to_string()), vec!["0", "1", "2"]);
    }

    // -- concurrency -------------------------------------------------------

    #[test]
    fn thousands_of_small_jobs() {
        // The race-shaking test: publish/retire churn with jobs small enough
        // that workers are constantly parking and waking.
        let p = pool(4);
        let mut expected = 0usize;
        let total = AtomicUsize::new(0);
        for round in 0..reps(4000) {
            let n: usize = round % 9;
            expected += n * n.saturating_sub(1) / 2;
            p.for_each(n, |i| {
                total.fetch_add(i, SeqCst);
            });
        }
        assert_eq!(total.load(SeqCst), expected);
    }

    #[test]
    fn thousands_of_small_maps() {
        let p = pool(4);
        for round in 0..reps(2000) {
            let n = round % 7;
            let out = p.map(n, |i| i * 2);
            assert_eq!(out, (0..n).map(|i| i * 2).collect::<Vec<_>>());
        }
    }

    #[test]
    fn concurrent_callers_share_the_pool() {
        // Several external threads submitting at once: jobs coexist in the
        // shared list and every one of them must still cover its range.
        let p = pool(4);
        let ok = AtomicBool::new(true);
        thread::scope(|s| {
            for t in 0..6 {
                let p = &p;
                let ok = &ok;
                s.spawn(move || {
                    for round in 0..reps(150) {
                        let n = 1 + (t * 7 + round) % 50;
                        let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                        p.for_each(n, |i| {
                            counts[i].fetch_add(1, SeqCst);
                        });
                        if counts.iter().any(|c| c.load(SeqCst) != 1) {
                            ok.store(false, SeqCst);
                        }
                    }
                });
            }
        });
        assert!(ok.load(SeqCst), "a concurrently submitted job lost or doubled an index");
    }

    #[test]
    fn concurrent_maps_from_several_threads() {
        let p = pool(4);
        thread::scope(|s| {
            for t in 0..4 {
                let p = &p;
                s.spawn(move || {
                    for _ in 0..reps(100) {
                        let out = p.map(64, |i| i * (t + 1));
                        for (i, v) in out.iter().enumerate() {
                            assert_eq!(*v, i * (t + 1));
                        }
                    }
                });
            }
        });
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn a_panic_on_one_caller_does_not_disturb_another() {
        let p = pool(4);
        thread::scope(|s| {
            let p = &p;
            s.spawn(move || {
                for _ in 0..reps(100) {
                    let r = panic::catch_unwind(AssertUnwindSafe(|| {
                        p.for_each(32, |i| {
                            if i == 5 {
                                panic!("boom");
                            }
                        });
                    }));
                    assert!(r.is_err());
                }
            });
            s.spawn(move || {
                for _ in 0..reps(100) {
                    let out = p.map(48, |i| i + 1);
                    assert_eq!(out, (1..=48).collect::<Vec<_>>());
                }
            });
        });
    }

    #[test]
    fn pools_can_be_created_and_dropped_repeatedly() {
        // Drop has to join, or a test binary that does this leaks 200 threads.
        for _ in 0..reps(50) {
            let p = pool(4);
            p.for_each(8, |_| {});
        }
    }

    /// The one place left in this module that asserts on elapsed time.
    ///
    /// Everything else here was rewritten to state its property without a
    /// clock; this one cannot be, because "immediate" *is* the property -- a
    /// shutdown that waits for a worker to notice the flag on its next
    /// revolution instead of being woken out of the condvar is still correct,
    /// just slow, and only a budget tells the two apart.
    ///
    /// So it is the one test that has to be skippable. `#[cfg(sanitize = ..)]`
    /// would be the natural gate but it is nightly-only and a hard error
    /// (E0658) on the stable toolchain this crate builds with, so the escape
    /// hatch is an env var instead -- set it for coverage, ASAN or TSAN runs.
    /// Miri gets the compile-time gate, since there the whole module would
    /// otherwise pay for a test it cannot evaluate.
    #[cfg(not(miri))]
    mod wallclock {
        use super::*;

        /// Set to anything to skip the wall-clock budget below.
        const NO_TIMING_ENV: &str = "GRANULAR_NO_TIMING";

        #[test]
        fn dropping_a_pool_while_it_is_idle_is_immediate() {
            if std::env::var_os(NO_TIMING_ENV).is_some() {
                return;
            }
            let p = pool(8);
            p.for_each(64, |_| {});
            let t = Instant::now();
            drop(p);
            assert!(t.elapsed() < Duration::from_secs(2), "shutdown took {:?}", t.elapsed());
        }
    }

    // -- global ------------------------------------------------------------

    #[test]
    fn global_is_a_single_pool() {
        let a = global();
        let b = global();
        assert!(ptr::eq(a, b), "global() must hand back the same pool");
        assert!(a.threads() >= 1);
    }

    #[test]
    fn global_runs_work() {
        let p = global();
        let n = reps(5000);
        let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        p.for_each(n, |i| {
            counts[i].fetch_add(1, SeqCst);
        });
        assert!(counts.iter().all(|c| c.load(SeqCst) == 1));
        assert_eq!(p.map(64, |i| i * 2).iter().sum::<usize>(), (0..64).map(|i| i * 2).sum());
    }

    #[test]
    fn thread_count_resolution() {
        let auto = resolve_threads(None);
        assert!(auto >= 1 && auto <= MAX_THREADS);
        assert_eq!(resolve_threads(Some("1")), 1);
        assert_eq!(resolve_threads(Some("7")), 7);
        assert_eq!(resolve_threads(Some(" 3 ")), 3, "surrounding space is tolerated");
        assert_eq!(resolve_threads(Some("99999")), MAX_THREADS, "clamped, not obeyed");
        // Nonsense falls back to the machine rather than to a broken pool.
        assert_eq!(resolve_threads(Some("0")), auto);
        assert_eq!(resolve_threads(Some("")), auto);
        assert_eq!(resolve_threads(Some("many")), auto);
        assert_eq!(resolve_threads(Some("-2")), auto);
    }

    // -- idleness ----------------------------------------------------------

    /// Workers must park, not spin. A spinning pool of 8 burns 8 cores for as
    /// long as it is idle, which is invisible in every other test here and
    /// catastrophic in the engine.
    ///
    /// This used to read `getrusage(RUSAGE_SELF)` across alternating 100ms
    /// windows. `RUSAGE_SELF` is process-wide, so it sees every other test in
    /// the binary and no amount of denoising fully isolates it -- and it is a
    /// CPU-time budget, so it means nothing under a tool that inflates CPU time
    /// by 50x. The counter it replaces it with is exact, per pool, and has no
    /// clock in it: the worker loop ticks `idle_polls` once each time it looks
    /// for work and finds none. Parked on a condvar that happens once per job
    /// and then the thread is unschedulable; spinning, it happens once per
    /// revolution, forever.
    #[test]
    fn idle_workers_do_not_spin() {
        let p = pool(8);
        let workers = p.threads() - 1;
        if workers == 0 {
            return;
        }
        p.for_each(64, |_| {}); // every worker wakes, works, and re-parks
        let before = p.idle_polls();

        // Hand the machine to whoever wants it, over and over. A parked worker
        // cannot be scheduled by any of these; a spinning one is scheduled by
        // every one of them and ticks the counter each time round its loop.
        for _ in 0..10_000 {
            thread::yield_now();
        }

        // One tick per worker may still be in flight: a worker that had not
        // re-parked when `for_each` returned registers its "nothing left to
        // run" after `before` was read. Doubled for a spurious condvar wakeup,
        // which is legal and vanishingly rare -- and a spin loop overshoots
        // either bound by four orders of magnitude, so the slack costs no
        // discriminating power.
        let spun = p.idle_polls() - before;
        assert!(
            spun <= 2 * workers,
            "{workers} idle workers went round their loop {spun} times while nothing was \
             submitted; parked workers cannot run at all"
        );
    }

    // -- adversarial review ------------------------------------------------

    #[test]
    fn adv_for_each_returns_only_after_every_item_finished() {
        // The soundness core: the caller's frame owns `f` and its captures, so
        // no worker may still be inside an item when `for_each` returns.
        //
        // The old form staggered the items with 20ms sleeps so the caller
        // would finish first and have something left to wait for. The
        // rendezvous is the same idea taken to its limit and without a clock:
        // all eight items are inside the closure simultaneously, so whichever
        // participant leaves first, `for_each` still has seven live borrows to
        // wait out.
        let p = pool(8);
        if p.threads() < 8 {
            return;
        }
        for round in 0..reps(30) {
            let inflight = AtomicUsize::new(0);
            let met = Meet::new(8);
            let all_in = AtomicBool::new(true);
            p.for_each(8, |_| {
                inflight.fetch_add(1, SeqCst);
                if !met.arrive() {
                    all_in.store(false, SeqCst);
                }
                inflight.fetch_sub(1, SeqCst);
            });
            assert!(all_in.load(SeqCst), "round {round}: 8 items did not reach 8 threads");
            assert_eq!(
                inflight.load(SeqCst),
                0,
                "round {round}: for_each returned with items still in flight"
            );
        }
    }

    #[test]
    fn adv_map_returns_only_after_every_slot_is_written() {
        let p = pool(8);
        if p.threads() < 8 {
            return;
        }
        for _ in 0..reps(30) {
            let inflight = AtomicUsize::new(0);
            let met = Meet::new(8);
            let all_in = AtomicBool::new(true);
            let out = p.map(8, |i| {
                inflight.fetch_add(1, SeqCst);
                if !met.arrive() {
                    all_in.store(false, SeqCst);
                }
                inflight.fetch_sub(1, SeqCst);
                i * 5
            });
            assert!(all_in.load(SeqCst), "8 items did not reach 8 threads");
            assert_eq!(inflight.load(SeqCst), 0, "map returned mid-write");
            assert_eq!(out, (0..8).map(|i| i * 5).collect::<Vec<_>>());
        }
    }

    #[test]
    fn adv_no_job_is_left_in_the_shared_list() {
        // A job that outlives its `run` frame in `state.jobs` is a dangling
        // `&'static Job` the next worker will dereference.
        let p = pool(4);
        for _ in 0..reps(200) {
            p.for_each(32, |_| {});
            let _ = p.map(16, |i| i);
        }
        assert!(lock(&p.shared.state).jobs.is_empty(), "a retired job stayed published");
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn adv_no_job_is_left_in_the_shared_list_after_a_panic() {
        let p = pool(4);
        for _ in 0..reps(50) {
            let r = panic::catch_unwind(AssertUnwindSafe(|| {
                p.for_each(32, |i| {
                    if i % 3 == 0 {
                        panic!("boom {i}");
                    }
                });
            }));
            assert!(r.is_err());
        }
        assert!(lock(&p.shared.state).jobs.is_empty(), "a panicking job stayed published");
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn adv_map_drop_accounting_with_many_panicking_indices() {
        static D: AtomicUsize = AtomicUsize::new(0);
        struct T(#[allow(dead_code)] usize);
        impl Drop for T {
            fn drop(&mut self) {
                D.fetch_add(1, SeqCst);
            }
        }
        let n = reps(4096);
        let p = pool(8);
        let r = panic::catch_unwind(AssertUnwindSafe(|| {
            let _v: Vec<T> = p.map(n, |i| {
                if i % 7 == 3 {
                    panic!("no value for {i}");
                }
                T(i)
            });
        }));
        assert!(r.is_err());
        let panicking = (0..n).filter(|i| i % 7 == 3).count();
        assert_eq!(D.load(SeqCst), n - panicking, "map mis-accounted its result slots");
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn adv_map_serial_path_panic_cleans_up() {
        static D2: AtomicUsize = AtomicUsize::new(0);
        struct T(#[allow(dead_code)] usize);
        impl Drop for T {
            fn drop(&mut self) {
                D2.fetch_add(1, SeqCst);
            }
        }
        let p = pool(1); // serial path
        let r = panic::catch_unwind(AssertUnwindSafe(|| {
            let _v: Vec<T> = p.map(32, |i| {
                if i == 9 {
                    panic!("serial");
                }
                T(i)
            });
        }));
        assert!(r.is_err());
        assert_eq!(D2.load(SeqCst), 31);
    }

    #[test]
    fn adv_randomized_concurrent_stress() {
        // Mixed widths, nesting and result collection from six submitters at
        // once: shakes publish/retire/park transitions harder than the
        // fixed-shape tests above.
        let p = pool(4);
        thread::scope(|s| {
            for t in 0..6usize {
                let p = &p;
                s.spawn(move || {
                    let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ (t as u64 + 1);
                    for _ in 0..reps(300) {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        let n = (x as usize) % 33;
                        let out = p.map(n, |i| i * 3);
                        assert_eq!(out.len(), n);
                        for (i, v) in out.iter().enumerate() {
                            assert_eq!(*v, i * 3);
                        }
                        if x % 5 == 0 {
                            let hits = AtomicUsize::new(0);
                            p.for_each(n, |_| {
                                p.for_each(3, |_| {
                                    hits.fetch_add(1, SeqCst);
                                });
                            });
                            assert_eq!(hits.load(SeqCst), n * 3);
                        }
                    }
                });
            }
        });
        assert!(lock(&p.shared.state).jobs.is_empty());
    }

    /// `drain` catches the item's panic but then calls `record` *outside* the
    /// catch. `record` drops the payload of every panic after the first, so a
    /// payload whose `Drop` panics unwinds straight out of `drain`:
    ///   - on a worker: `active` is never decremented -> the submitter waits
    ///     forever on `done`;
    ///   - on the caller: `run` unwinds with the job still in `state.jobs`,
    ///     leaving a dangling `&'static Job` pointing at a dead stack frame.
    /// Run in a child process, because either outcome wedges or corrupts the
    /// test binary.
    #[cfg(panic = "unwind")]
    #[test]
    fn adv_panic_payload_with_a_panicking_drop() {
        struct Nasty;
        impl Drop for Nasty {
            fn drop(&mut self) {
                if !thread::panicking() {
                    panic!("payload drop");
                }
            }
        }
        let p = pool(4);
        let r = panic::catch_unwind(AssertUnwindSafe(|| {
            p.for_each(64, |_| panic::panic_any(Nasty));
        }));
        assert!(r.is_err());
        assert!(
            lock(&p.shared.state).jobs.is_empty(),
            "the job was left published after run() unwound"
        );
    }

    /// Same hole, worker side. Only items that land on a worker panic, so the
    /// worker unwinds out of `drain` (and out of `worker`) with `active` still
    /// at 1, and the submitter's `while job.active != 0` never terminates.
    ///
    /// The worker used to be given its two panics by keeping the caller busy
    /// with 5ms sleeps and hoping; now the caller holds every item it claims
    /// until the worker says it has done the damage, so the interleaving the
    /// test needs is the only one it can have.
    #[cfg(panic = "unwind")]
    #[test]
    fn adv_worker_unwind_strands_the_submitter() {
        struct Nasty;
        impl Drop for Nasty {
            fn drop(&mut self) {
                if !thread::panicking() {
                    panic!("payload drop");
                }
            }
        }
        // Fires on normal return *and* on unwind, so it distinguishes "for_each
        // gave control back" from "for_each is blocked forever".
        struct Returned(Arc<Gate>);
        impl Drop for Returned {
            fn drop(&mut self) {
                self.0.open();
            }
        }

        let returned = Arc::new(Gate::new());
        let flag = Arc::clone(&returned);
        let h = thread::spawn(move || {
            let p = pool(2);
            let _guard = Returned(flag); // drops before `p`, so before the join
            if p.threads() < 2 {
                return;
            }
            let hits = AtomicUsize::new(0);
            let damaged = Gate::new();
            let _ = panic::catch_unwind(AssertUnwindSafe(|| {
                p.for_each(8, |_| {
                    let on_worker = thread::current()
                        .name()
                        .map(|n| n.starts_with("granular-pool-"))
                        .unwrap_or(false);
                    if on_worker {
                        // Two of these on the worker: the first payload is
                        // stashed, the second is dropped inside `record` and
                        // takes the unwind out through `Clear`.
                        if hits.fetch_add(1, SeqCst) == 1 {
                            damaged.open();
                        }
                        panic::panic_any(Nasty);
                    }
                    damaged.wait();
                });
            }));
        });
        assert!(
            returned.wait(),
            "for_each never returned: a worker unwound out of drain without clearing `active`"
        );
        let _ = h.join();
    }

    /// A panic that unwinds out of `run` must not leave its `Job` published.
    ///
    /// `run` launders `&Job` to `&'static Job` and pushes it into `state.jobs`
    /// so workers can reach it. If an unwind skipped the removal, that entry
    /// would outlive `run`'s stack frame and the *next* call's workers would
    /// read `j.next` / `j.n` through it -- a use-after-free, and the reason
    /// retirement is tied to a `Drop` rather than to the end of the function.
    #[cfg(panic = "unwind")]
    #[test]
    fn adv_dangling_job_is_not_left_behind_by_an_unwind() {
        struct Nasty;
        impl Drop for Nasty {
            fn drop(&mut self) {
                if !thread::panicking() {
                    panic!("payload drop");
                }
            }
        }
        let p = pool(2);
        let _ = panic::catch_unwind(AssertUnwindSafe(|| {
            p.for_each(6, |_| panic::panic_any(Nasty));
        }));
        assert!(
            lock(&p.shared.state).jobs.is_empty(),
            "a job outlived the frame that owns it; the next call would read freed memory"
        );
        // Which makes this safe, and the pool still usable afterwards.
        let n = AtomicUsize::new(0);
        p.for_each(6, |_| {
            n.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(n.load(Ordering::Relaxed), 6, "the pool did not recover");
    }

    /// Watchdog wrapper: a hang inside the pool should fail this test rather
    /// than block the harness forever. The deadline is [`STUCK`], which is a
    /// liveness guard and not a throughput budget -- the stress below is four
    /// orders of magnitude away from it on any machine that is not deadlocked.
    #[test]
    fn adv_stress_completes_within_a_deadline() {
        let done = Arc::new(Gate::new());
        let flag = Arc::clone(&done);
        let h = thread::spawn(move || {
            let p = pool(6);
            for _ in 0..reps(2000) {
                p.for_each(5, |_| {});
                let _ = p.map(3, |i| i);
            }
            flag.open();
        });
        assert!(done.wait(), "pool stress did not finish -- deadlock");
        h.join().unwrap();
    }

    /// Heap-side face of the same hole: when `run` unwinds, `map`'s `out`
    /// buffer is freed by the unwind while workers are still writing results
    /// through `SendPtr` into it.
    ///
    /// The old version assumed indices 0 and 1 would fall to the caller
    /// because it starts draining before the worker wakes, and used 50ms items
    /// plus a 300ms tail sleep to keep the worker inside its item across the
    /// unwind. Both are races. Here the panic is chosen by *who is running the
    /// item* rather than by its number, and two gates pin the order outright:
    ///
    ///   worker claims an item, says so, and waits  ->  caller's first item
    ///   panics (payload stashed)  ->  caller's second item releases the worker
    ///   and panics again, and *that* payload is dropped inside `record`, which
    ///   unwinds out of `run` with the worker still inside the closure, holding
    ///   a `SendPtr` into the buffer `map` is about to free.
    #[cfg(panic = "unwind")]
    #[test]
    fn adv_map_unwind_frees_the_buffer_under_the_workers() {
        struct Nasty;
        impl Drop for Nasty {
            fn drop(&mut self) {
                if !thread::panicking() {
                    panic!("payload drop");
                }
            }
        }
        let p = pool(2);
        if p.threads() < 2 {
            return;
        }
        let caller = thread::current().id();
        let claimed = Gate::new(); // worker: "I am inside an item"
        let unwinding = Gate::new(); // caller: "my next panic escapes `run`"
        let mine = AtomicUsize::new(0);
        let helped = AtomicBool::new(false);
        let _ = panic::catch_unwind(AssertUnwindSafe(|| {
            let _v: Vec<u64> = p.map(6, |i| {
                if thread::current().id() == caller {
                    claimed.wait();
                    if mine.fetch_add(1, SeqCst) == 1 {
                        unwinding.open();
                    }
                    panic::panic_any(Nasty);
                }
                helped.store(true, SeqCst);
                claimed.open();
                unwinding.wait();
                i as u64 // ... and this write lands mid-unwind
            });
        }));
        // Asserted out here because a failure inside the closure would just be
        // another panic for `map` to swallow.
        assert!(helped.load(SeqCst), "no worker took an item: the race was never set up");
        assert!(mine.load(SeqCst) >= 2, "the caller took {} items, need 2 to escape `run`", mine.load(SeqCst));
        // No settling sleep: `Retire` is what guarantees the worker is out, and
        // a build where that is not true is the bug this test exists to catch.
    }

    /// Compact enough for `miri --many-seeds` to explore many interleavings of
    /// publish / claim / retire.
    #[test]
    fn adv_tiny_parallel_roundtrip() {
        let p = pool(3);
        let src: Vec<usize> = (0..5).collect();
        let out = p.map(5, |i| src[i] + 1);
        assert_eq!(out, vec![1, 2, 3, 4, 5]);
        let hits = AtomicUsize::new(0);
        p.for_each(5, |_| {
            hits.fetch_add(1, SeqCst);
        });
        assert_eq!(hits.load(SeqCst), 5);
        assert!(lock(&p.shared.state).jobs.is_empty());
    }

    #[test]
    fn adv_wide_pool_narrow_jobs() {
        // n == 2 on a wide pool: only one worker is ever notified. Repeated
        // heavily, a lost wakeup or a miscounted `active` shows up as a hang.
        let p = pool(16);
        for round in 0..reps(3000) {
            let hits = AtomicUsize::new(0);
            p.for_each(2, |_| {
                hits.fetch_add(1, SeqCst);
            });
            assert_eq!(hits.load(SeqCst), 2, "round {round}");
        }
    }
}
