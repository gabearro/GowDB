//! `ORDER BY`.
//!
//! Blocking: every row has to be seen before the first can be emitted. What is
//! sorted is a **permutation** of `u32` row ids, not the rows themselves, so
//! each pass moves 12 bytes per row regardless of how wide the table is and
//! every column is gathered exactly once at the end, by `Block::take`.
//!
//! ## Two strategies
//!
//! *Radix* when there is a single key that is non-nullable and non-string. The
//! key goes through [`crate::common`]'s order-preserving lane codec, so four
//! branchless 16-bit LSD passes over `(lane, row)` pairs replace `n log n`
//! comparisons -- and [`radix_sort`] skips any pass whose 16 bits are uniform,
//! which for clustered data is usually the top two.
//!
//! *Comparison* otherwise: nullable keys, string keys, or several keys. NULL
//! placement and per-key direction are the reason -- a lane cannot encode
//! "NULLS LAST on a descending key", and string lanes are per-granule
//! dictionary codes with no global meaning.
//!
//! ## Descending without losing stability
//!
//! Reversing an ascending sort gives descending order but also reverses ties,
//! which turns a stable sort into an unstable one and makes `ORDER BY a DESC`
//! non-deterministic across equal `a`. So after reversing, each run of equal
//! keys is reversed back. That is one extra linear pass and it buys a genuine
//! stable descending sort, which matters as soon as a second `ORDER BY` column
//! or a `LIMIT` is involved.
//!
//! ## Top-K
//!
//! `ORDER BY x LIMIT n` does not need a sorted input, only its n extreme rows,
//! and [`super::build`] fuses the pair so [`Sort::top_k`] hears about it. The
//! buffer is then capped at `n + BLOCK_SIZE` rows instead of the whole
//! relation, and once it is full each incoming row is tested against the
//! current n-th best -- one comparison per row against a hoisted threshold --
//! so blocks that cannot contribute never reach a sort at all. Measured
//! interleaved against the same build with the fusion disabled, best-of-7 over
//! 2M rows: `ORDER BY latency DESC LIMIT 5` 34.8ms -> 17.2ms (2.0x, and that
//! side was already on the fast radix path), `ORDER BY country, ts LIMIT 100`
//! 592ms -> 45ms (13.1x). Peak sort memory falls from the whole relation to
//! roughly `1.5k` rows, which is the part that matters here.
//!
//! It is not unconditional: past a certain `k` the fusion starts losing, and
//! [`Sort::worth_fusing`] carries the table of where.
//!
//! ## Spilling
//!
//! A sort whose buffer will not fit the query's memory budget no longer fails.
//! [`Sort::materialize_all`] sorts what it has, writes it to a temp file as a
//! **run**, releases the reservation and carries on; the runs are then merged
//! back with [`RunMerge`], which streams so the merged relation never has to be
//! resident either. That is a textbook external merge sort, and the only
//! interesting decisions are where the threshold comes from and what the
//! in-memory path pays for it.
//!
//! The threshold is not a constant: it is the [`MemGuard`] itself refusing to
//! grow. Whatever budget the query was given -- and whatever the rest of the
//! plan is already holding against it -- is what decides, which is the only
//! definition that stays right when two operators spill in the same query.
//!
//! The in-memory path pays **nothing**. The check is the `grow_to` this loop
//! already made once per block; all that changed is that its `Err` selects a
//! spill instead of returning. A query that fits reaches exactly the code it
//! reached before, with the same buffer, the same permutation and the same
//! `chunk_take`. Measured interleaved against the pre-spill body -- kept
//! verbatim behind a temporary env-var switch, alternating sides in one loop,
//! best-of-7 over 2M rows, two runs: `ORDER BY v` (radix) 0.992x and 1.000x,
//! `ORDER BY s, v` (comparison) 0.989x and 0.995x. Null, as intended.
//!
//! ## What spilling costs
//!
//! Same 2M rows (40 B/row, 80 MB), best-of-3 per budget, the budget swept from
//! "comfortably fits" downwards:
//!
//! ```text
//!   budget       246M    82M    61M    31M   7.7M   1.9M
//!   ORDER BY v    124    297    282    279    447   1081  ms
//!   ORDER BY s,v  786    817    788    746    912   1680  ms
//!                   ^ last budget that holds the whole relation
//! ```
//!
//! **The cliff is 2.3x on the radix path and ~1.0x on the comparison path**,
//! and the difference is the point: spilling costs one write and one read of
//! the relation plus a heap sift per output row, which is most of what a radix
//! sort costs and almost none of what an `n log n` sort over `Value`s costs.
//! It then stays flat as the budget keeps falling, because the work does not
//! change until the run count outgrows the fan-in -- the 7.7M and 1.9M columns
//! are two and three merge passes, and each pass is one more write and read.
//!
//! Top-K deliberately does not spill: it is memory-bounded by construction
//! (`1.5k + BLOCK_SIZE` rows), so a `k` large enough to exhaust the budget is a
//! `k` that should have been a plain sort, and [`Sort::worth_fusing`] already
//! declines those.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::mem::size_of;
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::common::{f64_to_lane, i64_to_lane, Error, Result, BLOCK_SIZE};
use crate::exec::expr;
use crate::planner::logical::SortKey;
use crate::sort::radix_sort;
use crate::types::{Block, Column, ColumnData, PhysicalType, Schema, Value};

use super::{chunk, chunk_take, drain, MemGuard, Operator, QueryContext, ScanStats};

pub struct Sort<'a> {
    input: Box<dyn Operator + 'a>,
    keys: &'a [SortKey],
    ctx: &'a QueryContext,
    /// `Some(k)`: only the first `k` rows of the sorted order are ever read,
    /// because a `LIMIT` sits directly above. `None`: sort everything.
    fetch: Option<usize>,
    /// Reversed once materialization finishes; see `next`.
    out: Vec<Block>,
    /// `Some` only for a sort that spilled. A query that fit in RAM never
    /// looks at this field except once, at end of stream.
    merge: Option<RunMerge<'a>>,
    ready: bool,
}

impl<'a> Sort<'a> {
    pub fn new(
        input: Box<dyn Operator + 'a>,
        keys: &'a [SortKey],
        ctx: &'a QueryContext,
    ) -> Sort<'a> {
        Sort { input, keys, ctx, fetch: None, out: Vec::new(), merge: None, ready: false }
    }

    /// Is fusing a fetch of `k` into a sort over `keys` worth it?
    ///
    /// Top-K wins by not sorting rows that cannot make the answer, so it stops
    /// paying once `k` approaches the input size -- and the point where that
    /// happens depends entirely on how fast the underlying sort is. Measured
    /// interleaved against the unfused plan, best-of-7 (best-of-3 above 200k),
    /// two runs, 2M rows:
    ///
    /// ```text
    ///   k        radix (ORDER BY latency)   comparison (ORDER BY country, ts)
    ///   5        1.89x  2.02x               --
    ///   100      --                         13.84x  13.10x
    ///   10 000   1.58x  1.46x               --
    ///   50 000   1.22x  1.16x               --
    ///   100 000  1.08x  1.01x               4.60x   5.12x
    ///   200 000  1.03x  0.76x               --
    ///   500 000  0.62x  <- regression       1.46x   1.39x
    /// ```
    ///
    /// The radix path is four linear passes, so re-running it on a 750k-row
    /// buffer costs more than one pass over the whole 2M; the comparison path
    /// is `n log n` over `Value` comparisons, so it is still well ahead at
    /// 500k. Hence two ceilings rather than one, each set an octave below its
    /// measured crossover: this machine swings ±20% run to run (the 500k radix
    /// row above is *the same code on both sides* once the ceiling took
    /// effect, and it still read 0.84x), so a ceiling placed at the crossover
    /// would be a coin flip. Above them the plain sort runs and little is
    /// lost -- at `k = n/4` the memory saving was small anyway.
    pub(crate) fn worth_fusing(keys: &[SortKey], k: usize) -> bool {
        let radix = keys.len() == 1
            && keys[0].expr.ty().physical() != PhysicalType::Str
            && !keys[0].expr.ty().is_nullable();
        k <= if radix { 64 << 10 } else { 1 << 20 }
    }

    /// A sort that only has to be right about its first `k` rows.
    pub fn top_k(
        input: Box<dyn Operator + 'a>,
        keys: &'a [SortKey],
        k: usize,
        ctx: &'a QueryContext,
    ) -> Sort<'a> {
        Sort { input, keys, ctx, fetch: Some(k), out: Vec::new(), merge: None, ready: false }
    }

    fn materialize(&mut self) -> Result<()> {
        self.ready = true;
        let r = self.materialize_into();
        // Handed out back to front so `next` can pop instead of cloning; see
        // there for why.
        self.out.reverse();
        r
    }

    fn materialize_into(&mut self) -> Result<()> {
        match self.fetch {
            // A `LIMIT 0` still has to consume nothing, not everything.
            Some(k) if self.keys.is_empty() || k == 0 => {
                self.out = Vec::new();
                if k > 0 {
                    // No keys means input order, so the first k rows will do.
                    let mut guard = MemGuard::new(self.ctx, "the sort buffer");
                    let all = drain(&mut self.input, self.ctx, &mut guard)?;
                    self.out = chunk(all.slice(0, k.min(all.rows())));
                }
                Ok(())
            }
            Some(k) => self.materialize_top_k(k),
            None => self.materialize_all(),
        }
    }

    /// Sort everything, spilling sorted runs to disk if the buffer will not fit.
    ///
    /// The accumulation loop is [`drain`] with one line added: the `grow_to`
    /// that used to be the query's cause of death now decides, on failure,
    /// that what is buffered becomes a run. A query that never trips it takes
    /// the identical path it took before -- one buffer, one permutation, one
    /// `chunk_take` -- which is the whole reason the spill state lives in
    /// locals here rather than in the operator.
    fn materialize_all(&mut self) -> Result<()> {
        let mut guard = MemGuard::new(self.ctx, "the sort buffer");
        // Charged alongside the rows so the reservation tracks the peak the
        // sort will actually reach, not just the buffer it has filled so far.
        // Derived from the keys' *declared* types, which is what a buffer
        // still being filled has to go on, and which agrees with
        // `scratch_bytes` on every shape but one: a key declared nullable
        // whose blocks turn out to hold no nulls is charged the comparison
        // path's 28 B/row where the radix path will want 16. That direction is
        // the safe one and it is 24 MB on a 2M-row sort.
        let per_row = est_scratch_per_row(self.keys);
        let forced = forced_spill_rows();
        let mut acc: Option<Block> = None;
        let mut runs: Option<RunSet> = None;

        loop {
            self.ctx.check()?;
            let Some(b) = self.input.next()? else { break };
            if b.rows() == 0 {
                continue;
            }
            match &mut acc {
                None => acc = Some(b),
                Some(a) => a.extend(&b)?,
            }
            let a = acc.as_ref().expect("just filled");
            let (rows, need) = (a.rows(), a.bytes() + a.rows() * per_row);
            if guard.grow_to(need).is_ok() && (forced == 0 || rows < forced) {
                continue;
            }
            // Out of budget. Sort what is here, push it out as a run, and hand
            // the reservation back by replacing the guard -- `MemGuard` only
            // ever grows, because a mid-flight refund is what would let two
            // operators both believe they have the headroom.
            let b = acc.take().expect("just filled");
            let perm = self.order_of(&b)?;
            let set = match &mut runs {
                Some(s) => s,
                None => runs.insert(RunSet::new()?),
            };
            set.push_run(&b, &perm, self.ctx)?;
            drop(b);
            guard = MemGuard::new(self.ctx, "the sort buffer");
        }

        let Some(mut set) = runs else {
            // The common case: everything fit, so this is the pre-spill code.
            let Some(all) = acc else { return Ok(()) };
            if all.rows() == 0 {
                return Ok(());
            }
            let cols = key_columns(self.keys, &all)?;
            let refs: Vec<&Column> = cols.iter().map(|c| c.as_ref()).collect();
            guard.grow_to(all.bytes() + scratch_bytes(all.rows(), self.keys, &refs))?;
            let perm = permutation(&refs, self.keys, all.rows())?;
            drop(refs);
            drop(cols);
            self.out = chunk_take(&all, &perm);
            return Ok(());
        };

        if let Some(tail) = acc {
            let perm = self.order_of(&tail)?;
            set.push_run(&tail, &perm, self.ctx)?;
        }
        drop(guard);
        self.merge = Some(RunMerge::open(set, self.keys, self.ctx)?);
        Ok(())
    }

    /// The sorted order of one buffer.
    fn order_of(&self, b: &Block) -> Result<Vec<u32>> {
        let cols = key_columns(self.keys, b)?;
        let refs: Vec<&Column> = cols.iter().map(|c| c.as_ref()).collect();
        permutation(&refs, self.keys, b.rows())
    }

    /// Bounded top-K: keep at most `k` rows, merging blocks into the buffer
    /// and sorting it only when it has outgrown `k` by a margin.
    ///
    /// Because the merge is a *stable* sort of `already-kept ++ new`, ties
    /// break toward the earlier input row exactly as a full sort would; that
    /// equivalence is pinned by `top_k_agrees_with_a_full_sort_including_ties`.
    ///
    /// The margin is what keeps this linear. Sorting and trimming on *every*
    /// block -- the obvious version, and the first one written here -- sorts
    /// the whole buffer `k / BLOCK_SIZE` times before it is even full, which
    /// is `O(k^2 / BLOCK_SIZE)` of wasted comparisons: a `LIMIT 1000000` over
    /// 2M rows would sort ~61M elements where one plain sort does 2M. Letting
    /// the buffer run `k/2` past `k` between trims makes the total `O(n log k)`
    /// again, at a peak of `1.5k + BLOCK_SIZE` rows instead of `k`.
    fn materialize_top_k(&mut self, k: usize) -> Result<()> {
        let mut guard = MemGuard::new(self.ctx, "the top-K sort buffer");
        let nk = self.keys.len();
        let trim_at = k.saturating_add((k / 2).max(crate::common::BLOCK_SIZE));
        // Upper bound on what a trim allocates per row: the permutation, plus
        // the key tuples if it takes the comparison path. Charged without
        // dispatching on the key type, because being 12 bytes/row pessimistic
        // on the radix path is cheaper than deciding per block.
        let scratch_per_row = 4 + nk * size_of::<Value>();
        let mut top: Option<Block> = None;
        // Reused across blocks: the survivor list, the per-row key probe, and
        // the threshold tuple. Nothing here allocates per row.
        let mut sel: Vec<u32> = Vec::new();
        let mut probe: Vec<Value> = vec![Value::Null; nk];
        let mut worst: Vec<Value> = Vec::new();

        loop {
            self.ctx.check()?;
            let Some(b) = self.input.next()? else { break };
            if b.rows() == 0 {
                continue;
            }

            // The buffer is full, so only rows that beat the current k-th best
            // can change the answer. Strictly-less is what keeps this stable:
            // a row that ties the k-th best sorts after it and would be
            // dropped again anyway.
            let b = if worst.is_empty() {
                b
            } else {
                let kc = key_columns(self.keys, &b)?;
                sel.clear();
                for i in 0..b.rows() {
                    for (slot, c) in probe.iter_mut().zip(kc.iter()) {
                        *slot = c.as_ref().value(i);
                    }
                    if compare_keys(&probe, &worst, self.keys) == Ordering::Less {
                        sel.push(i as u32);
                    }
                }
                if sel.is_empty() {
                    continue;
                }
                let kept = b.take(&sel);
                drop(kc);
                kept
            };

            let mut combined = match top.take() {
                None => b,
                Some(mut t) => {
                    t.extend(&b)?;
                    t
                }
            };
            guard.grow_to(combined.bytes() + combined.rows() * scratch_per_row)?;
            if combined.rows() >= trim_at {
                combined = self.trim(combined, k, &mut worst)?;
            }
            top = Some(combined);
        }

        // The buffer holds up to `trim_at` rows in whatever order the last
        // merge left them, so the final trim is not optional.
        if let Some(all) = top {
            self.out = chunk(self.trim(all, k, &mut worst)?);
        }
        Ok(())
    }

    /// Sort the buffer, keep its first `k` rows, and republish the threshold.
    fn trim(&self, b: Block, k: usize, worst: &mut Vec<Value>) -> Result<Block> {
        let cols = key_columns(self.keys, &b)?;
        let refs: Vec<&Column> = cols.iter().map(|c| c.as_ref()).collect();
        let perm = permutation(&refs, self.keys, b.rows())?;
        drop(refs);
        drop(cols);
        let out = b.take(&perm[..k.min(perm.len())]);
        if out.rows() >= k {
            // Refresh the threshold. Once per trim, never per row -- which is
            // why it is worth materializing as a `Vec<Value>` at all.
            let kc = key_columns(self.keys, &out)?;
            worst.clear();
            for c in &kc {
                worst.push(c.as_ref().value(k - 1));
            }
        }
        Ok(out)
    }
}

/// The key columns of a block, borrowed where the key is a plain column.
///
/// A bare `ORDER BY col` would otherwise clone the whole column -- for a top-N
/// over millions of rows, a full copy of the data purely in order to read it.
fn key_columns<'b>(keys: &[SortKey], b: &'b Block) -> Result<Vec<Cow<'b, Column>>> {
    keys.iter()
        .map(|k| match &k.expr {
            crate::planner::logical::BoundExpr::Column { index, .. } if *index < b.width() => {
                Ok(Cow::Borrowed(b.column(*index)))
            }
            other => expr::eval(other, b).map(Cow::Owned),
        })
        .collect()
}

/// What [`permutation`] will allocate for `rows`: the id vector always, plus
/// either the radix `(lane, row)` pairs or the materialized key tuples.
fn scratch_bytes(rows: usize, keys: &[SortKey], cols: &[&Column]) -> usize {
    let per_row = if keys.len() == 1 && radix_eligible(cols[0]) {
        12
    } else {
        keys.len() * size_of::<Value>()
    };
    rows * (4 + per_row)
}

impl Operator for Sort<'_> {
    fn schema(&self) -> &Schema {
        self.input.schema()
    }

    fn stats(&self) -> ScanStats {
        self.input.stats()
    }

    fn next(&mut self) -> Result<Option<Block>> {
        if !self.ready {
            self.materialize()?;
        }
        // Popped from a reversed buffer, not cloned out of an indexed one.
        // `out` is the whole sorted relation, so cloning each block on the way
        // to the caller meant both copies were live at once -- peak memory of
        // two full results where one will do. Measured interleaved against the
        // cloning form with an `AtomicBool`, alternating sides, best-of-11 over
        // 2M rows: `ORDER BY v` 14.48 -> 14.76 ms, `ORDER BY s, v` 509.0 ->
        // 498.0 ms, `ORDER BY v LIMIT 10` 13.09 -> 13.01 ms. That is a null
        // result on time, and it is kept for the memory, not the clock.
        if let Some(b) = self.out.pop() {
            return Ok(Some(b));
        }
        // Reached once per query, at end of stream, and `None` for every sort
        // that fit in RAM.
        let Some(m) = self.merge.as_mut() else { return Ok(None) };
        match m.next()? {
            Some(b) => Ok(Some(b)),
            // Dropped rather than left exhausted: that is what unlinks the
            // spill files, and a pipeline above may hold the operator long
            // after it stopped reading from it.
            None => {
                self.merge = None;
                Ok(None)
            }
        }
    }
}

/// Per-row scratch [`permutation`] will want, from the keys' declared types.
///
/// Deliberately the same test [`Sort::worth_fusing`] makes rather than
/// [`radix_eligible`]'s: this is charged while the buffer is still filling, so
/// it cannot look at a block, and it has to agree with [`scratch_bytes`] or the
/// peak reservation of a sort that does *not* spill would move.
fn est_scratch_per_row(keys: &[SortKey]) -> usize {
    if keys.is_empty() {
        return 4;
    }
    let radix = keys.len() == 1
        && keys[0].expr.ty().physical() != PhysicalType::Str
        && !keys[0].expr.ty().is_nullable();
    4 + if radix { 12 } else { keys.len() * size_of::<Value>() }
}

/// Test knob: spill every `n` buffered rows (or, in the hash aggregate, every
/// `n` groups) regardless of the budget.
///
/// Exists because the budget a `Session` runs under is fixed at
/// [`super::DEFAULT_MEM_BUDGET`], so there is otherwise no way to point the
/// differential harness -- the thing most likely to catch a wrong external
/// sort -- at the spilling path:
///
/// ```text
///   GRANULAR_SPILL_ROWS=2000 GRANULAR_DIFF_CASES=20000 cargo test --release \
///       --test differential
/// ```
///
/// Read once per process; zero when unset, which is the only value that costs
/// anything, and it costs one predictable compare per block.
pub(crate) fn forced_spill_rows() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("GRANULAR_SPILL_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

// --------------------------------------------------------------- k-way merge

/// Merge already-sorted runs into one sorted stream, keeping at most `fetch`.
///
/// This is the top half of a parallel `ORDER BY`: each exchange worker sorts
/// its own slice of the scan and this folds the results. Runs must arrive in
/// **input order** -- worker 0's slice first -- because ties break toward the
/// lower absolute position, which is what makes the merged order identical to
/// the stable serial sort's rather than merely equivalent to it.
///
/// The storage layer already has a k-way merge (`Table::merge_parts`), and it
/// gets to be four lines: its keys are single `u64` lanes, so a
/// `BinaryHeap<Reverse<(u64, part, row)>>` can own them outright. A sort key
/// here is a tuple of `Value`s, and an owning heap item would mean one heap
/// allocation per *output row*. Hence [`kway`] below, which stores positions
/// and compares through a closure that reads an arena in place -- and hence the
/// two comparators, mirroring [`permutation`]'s own split: a single
/// non-nullable non-string key merges on `u64` lanes, everything else on
/// materialized key tuples.
///
/// Worth knowing before reaching for this: a merge costs a heap sift *per
/// output row*, which is a lot next to a radix sort's four linear passes.
/// Measured through the exchange on 10M rows and 14 cores, `ORDER BY latency`
/// (single non-null integer key, so the radix path) is 156 -> 139 ms, 1.1x --
/// the merge gives back most of what the parallel sort won. `ORDER BY country,
/// latency`, which sorts by `n log n` `Value` comparisons instead, is 7079 ->
/// 1685 ms, 4.2x. And `ORDER BY latency DESC LIMIT 5` is 133 -> 24 ms, because
/// there the runs are five rows long and the merge is free. The shape of the
/// win is set by how expensive the sort being parallelized was, not by the
/// merge.
pub(crate) fn merge_runs(
    runs: Vec<Block>,
    keys: &[SortKey],
    fetch: Option<usize>,
    guard: &mut MemGuard,
) -> Result<Vec<Block>> {
    let mut runs: Vec<Block> = runs.into_iter().filter(|b| b.rows() > 0).collect();
    let total: usize = runs.iter().map(|b| b.rows()).sum();
    let want = fetch.map_or(total, |k| k.min(total));
    if want == 0 {
        return Ok(Vec::new());
    }
    // One run is already the answer -- and no ORDER BY keys means input order,
    // so concatenation is too.
    if runs.len() == 1 || keys.is_empty() {
        let mut all = runs.remove(0);
        for r in &runs {
            all.extend(r)?;
        }
        return Ok(chunk(if want < all.rows() { all.slice(0, want) } else { all }));
    }

    // `ends[i]` is where run `i` stops inside the concatenation, so a heap
    // entry needs only its absolute position and its run.
    let mut ends: Vec<u32> = Vec::with_capacity(runs.len());
    let mut heads: Vec<(u32, u32)> = Vec::with_capacity(runs.len());
    let mut at = 0u32;
    for (i, r) in runs.iter().enumerate() {
        heads.push((at, i as u32));
        at += r.rows() as u32;
        ends.push(at);
    }
    let mut all = runs.remove(0);
    for r in &runs {
        all.extend(r)?;
    }
    drop(runs);

    let cols = key_columns(keys, &all)?;
    let refs: Vec<&Column> = cols.iter().map(|c| c.as_ref()).collect();
    guard.grow_to(all.bytes() + scratch_bytes(all.rows(), keys, &refs))?;

    let perm = if keys.len() == 1 && radix_eligible(refs[0]) {
        let lanes = lanes_of(refs[0])?;
        let asc = keys[0].asc;
        kway(&mut heads, &ends, want, |a, b| {
            let (x, y) = (lanes[a as usize], lanes[b as usize]);
            // Ties fall through to the position, which is the stable
            // tie-break; `asc` never flips it, because "earlier input row
            // first" is not a direction.
            if x != y {
                (x < y) == asc
            } else {
                a < b
            }
        })
    } else {
        let k = keys.len();
        let mut vals: Vec<Value> = Vec::with_capacity(all.rows() * k);
        for i in 0..all.rows() {
            for c in &refs[..k] {
                vals.push(c.value(i));
            }
        }
        kway(&mut heads, &ends, want, |a, b| {
            let (x, y) = (a as usize * k, b as usize * k);
            match compare_keys(&vals[x..x + k], &vals[y..y + k], keys) {
                Ordering::Less => true,
                Ordering::Greater => false,
                Ordering::Equal => a < b,
            }
        })
    };
    drop(refs);
    drop(cols);
    Ok(chunk_take(&all, &perm))
}

/// The lane vector [`radix_permutation`] would have built, without the sort.
fn lanes_of(c: &Column) -> Result<Vec<u64>> {
    Ok(match &c.data {
        ColumnData::U64(v) => v.clone(),
        ColumnData::I64(v) => v.iter().map(|&x| i64_to_lane(x)).collect(),
        ColumnData::F64(v) => v.iter().map(|&x| f64_to_lane(x)).collect(),
        ColumnData::Str(_) => return Err(Error::exec("string columns have no global sort lane")),
    })
}

/// Pop the least head `want` times. `heads` is `(position, run)` per live run.
///
/// `less` must be a strict total order -- ties broken by position -- or the
/// merge stops being a stable sort. Sift-down rather than `BinaryHeap` so the
/// comparator can close over the key arena; at fourteen runs the whole heap is
/// two cache lines and the sift is four comparisons.
fn kway<F>(heads: &mut Vec<(u32, u32)>, ends: &[u32], want: usize, mut less: F) -> Vec<u32>
where
    F: FnMut(u32, u32) -> bool,
{
    for i in (0..heads.len() / 2).rev() {
        sift_down(heads, i, &mut less);
    }
    let mut out = Vec::with_capacity(want);
    while out.len() < want {
        let Some(&(pos, run)) = heads.first() else { break };
        out.push(pos);
        let next = pos + 1;
        if next < ends[run as usize] {
            heads[0] = (next, run);
            sift_down(heads, 0, &mut less);
        } else {
            // The run is spent. Moving the tail into the root is the standard
            // shrink; when it *is* the root the pop already emptied the heap.
            let last = heads.pop().unwrap_or_default();
            if !heads.is_empty() {
                heads[0] = last;
                sift_down(heads, 0, &mut less);
            }
        }
    }
    out
}

fn sift_down<F>(h: &mut [(u32, u32)], mut i: usize, less: &mut F)
where
    F: FnMut(u32, u32) -> bool,
{
    let n = h.len();
    loop {
        let mut m = i;
        let l = 2 * i + 1;
        if l < n && less(h[l].0, h[m].0) {
            m = l;
        }
        if l + 1 < n && less(h[l + 1].0, h[m].0) {
            m = l + 1;
        }
        if m == i {
            return;
        }
        h.swap(i, m);
        i = m;
    }
}

/// Row order that sorts `cols` by `keys`.
pub(crate) fn permutation(cols: &[&Column], keys: &[SortKey], rows: usize) -> Result<Vec<u32>> {
    if keys.is_empty() || rows <= 1 {
        return Ok((0..rows as u32).collect());
    }
    if keys.len() == 1 && radix_eligible(cols[0]) {
        return radix_permutation(cols[0], keys[0].asc);
    }
    // Comparison fallback. Materializing the key tuples once up front turns
    // `O(n log n)` `Column::value` calls into `O(n)` of them.
    //
    // One flat arena, `vals[r*k .. (r+1)*k]`, not a `Vec<Value>` per row: the
    // per-row shape cost one heap allocation *per input row* (2M of them on
    // the benchmark's string sort) and made every comparison chase a pointer
    // out of the row vector before it could look at a key. Measured
    // interleaved against the old shape, best-of-9, `ORDER BY country, ts`
    // over 2M rows: 585ms -> 524ms (1.11x). Less than the allocation count
    // suggests, because `sort_by`'s comparisons dominate -- but it is free,
    // and it takes 2M allocations out of a query that makes one pass.
    let k = keys.len();
    let mut vals: Vec<Value> = Vec::with_capacity(rows * k);
    for i in 0..rows {
        for c in &cols[..k] {
            vals.push(c.value(i));
        }
    }
    let mut idx: Vec<u32> = (0..rows as u32).collect();
    idx.sort_by(|&a, &b| {
        let (a, b) = (a as usize * k, b as usize * k);
        compare_keys(&vals[a..a + k], &vals[b..b + k], keys)
    });
    Ok(idx)
}

fn radix_eligible(c: &Column) -> bool {
    // Strings have no global lane (dictionary codes are per granule), and a
    // NULL's lane is a stored zero that would sort as a real value.
    !c.has_nulls() && c.ty.physical() != PhysicalType::Str
}

fn radix_permutation(c: &Column, asc: bool) -> Result<Vec<u32>> {
    let mut keyed: Vec<(u64, u32)> = match &c.data {
        ColumnData::U64(v) => v.iter().enumerate().map(|(i, &x)| (x, i as u32)).collect(),
        ColumnData::I64(v) => v
            .iter()
            .enumerate()
            .map(|(i, &x)| (i64_to_lane(x), i as u32))
            .collect(),
        ColumnData::F64(v) => v
            .iter()
            .enumerate()
            .map(|(i, &x)| (f64_to_lane(x), i as u32))
            .collect(),
        ColumnData::Str(_) => {
            return Err(Error::exec("string columns have no global sort lane"))
        }
    };
    radix_sort(&mut keyed);
    if asc {
        return Ok(keyed.into_iter().map(|(_, i)| i).collect());
    }
    // Descending: reverse, then un-reverse each run of equal keys so the sort
    // stays stable.
    keyed.reverse();
    let mut s = 0;
    while s < keyed.len() {
        let mut e = s + 1;
        while e < keyed.len() && keyed[e].0 == keyed[s].0 {
            e += 1;
        }
        keyed[s..e].reverse();
        s = e;
    }
    Ok(keyed.into_iter().map(|(_, i)| i).collect())
}

/// Lexicographic comparison honouring each key's `asc` and `nulls_first`.
///
/// `NULLS FIRST` is absolute, not relative to the direction: it means "at the
/// top of the output", so it is applied *before* the descending flip rather
/// than being reversed along with the values. That matches ClickHouse and the
/// SQL standard's `NULLS FIRST | LAST` clause.
fn compare_keys(a: &[Value], b: &[Value], keys: &[SortKey]) -> Ordering {
    for (i, k) in keys.iter().enumerate() {
        let (x, y) = (&a[i], &b[i]);
        let o = match (x.is_null(), y.is_null()) {
            (true, true) => Ordering::Equal,
            (true, false) => {
                if k.nulls_first {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
            (false, true) => {
                if k.nulls_first {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
            (false, false) => {
                let o = x.cmp(y);
                if k.asc {
                    o
                } else {
                    o.reverse()
                }
            }
        };
        if o != Ordering::Equal {
            return o;
        }
    }
    Ordering::Equal
}

// -------------------------------------------------------------- spilled runs

/// The sorted runs one spilling operator has produced, and the directory they
/// live in.
///
/// Owning the directory is what makes the error path safe: every way out of a
/// spilling operator -- a `?` from the middle of a merge, a cancel, a `LIMIT`
/// above that stops reading -- drops this, and dropping it unlinks the files.
pub(crate) struct RunSet {
    dir: spill::SpillDir,
    runs: Vec<PathBuf>,
    /// Column types of the spilled blocks, taken from the first one written.
    /// Every run in a set comes from one operator, so one schema serves all.
    schema: Option<Schema>,
    /// Rows per spilled block. Not [`BLOCK_SIZE`]: see [`spill_block_rows`].
    block_rows: usize,
    row_bytes: usize,
    rows: usize,
}

impl RunSet {
    pub(crate) fn new() -> Result<RunSet> {
        Ok(RunSet {
            dir: spill::SpillDir::new()?,
            runs: Vec::new(),
            schema: None,
            block_rows: BLOCK_SIZE,
            row_bytes: 1,
            rows: 0,
        })
    }

    /// Write one already-sorted buffer as a run, cut into merge-sized blocks.
    pub(crate) fn push_run(&mut self, b: &Block, perm: &[u32], ctx: &QueryContext) -> Result<()> {
        if perm.is_empty() {
            return Ok(());
        }
        if self.schema.is_none() {
            self.schema = Some(spill::schema_of(b));
            self.row_bytes = (b.bytes() / b.rows().max(1)).max(1);
            self.block_rows = spill_block_rows(ctx, self.row_bytes);
        }
        let mut w = self.dir.create()?;
        let mut s = 0;
        while s < perm.len() {
            // A cancelled query that keeps writing gigabytes to disk is worse
            // than one that keeps burning CPU, so the checkpoint is inside the
            // write loop and not merely around it.
            ctx.check()?;
            let e = (s + self.block_rows).min(perm.len());
            // Gathered per window rather than once for the whole run: the same
            // reason `chunk_take` does it, one copy instead of two.
            w.push(&b.take(&perm[s..e]))?;
            s = e;
        }
        self.rows += perm.len();
        self.runs.push(w.finish()?);
        Ok(())
    }

    pub(crate) fn len(&self) -> usize {
        self.runs.len()
    }
}

/// Rows per spilled block, so that a merge fits in the budget that forced the
/// spill in the first place.
///
/// A merge holds one of these per open run, plus the concatenation it
/// assembles from and the output block it gathers into, so at the smallest
/// useful fan-in of two that is four blocks; the divisor is eight, which
/// leaves room for an expression key's evaluated column and the permutation.
/// Sizing them from the budget rather than pinning them at [`BLOCK_SIZE`] is
/// what lets a query with a 200 KiB budget spill *and then merge* instead of
/// trading one out-of-memory error for another; the cost is smaller output
/// batches, which the pipeline already tolerates.
fn spill_block_rows(ctx: &QueryContext, row_bytes: usize) -> usize {
    let budget = ctx.mem.limit().max(0) as usize;
    (budget / (row_bytes * 8)).clamp(64, BLOCK_SIZE)
}

/// Where one sort key's values live inside a spilled block.
///
/// A plain `ORDER BY col` reads them straight out of the block, which is the
/// point: the merge then carries no copy of the key column at all. Only a key
/// that is an *expression* has to be evaluated and kept, and it is kept
/// alongside the block so `fill`'s slicing keeps the two in step.
enum KeyAt {
    Col(usize),
    Expr(usize),
}

fn key_positions(keys: &[SortKey], width: usize) -> Vec<KeyAt> {
    let mut nx = 0;
    keys.iter()
        .map(|k| match &k.expr {
            crate::planner::logical::BoundExpr::Column { index, .. } if *index < width => {
                KeyAt::Col(*index)
            }
            _ => {
                nx += 1;
                KeyAt::Expr(nx - 1)
            }
        })
        .collect()
}

/// One run being read back, positioned at the row the merge will take next.
struct Cursor {
    src: spill::RunReader,
    /// The one spilled block this run is reading from. Replaced whole by
    /// `fill`, never grown: see there.
    cur: Block,
    /// The expression sort keys evaluated over `cur`, in [`KeyAt::Expr`] order.
    /// Sliced and extended in lockstep with `cur`.
    kx: Vec<Column>,
    pos: usize,
    /// `pos` at the start of the current output block.
    start: usize,
    done: bool,
}

/// Streaming k-way merge of sorted runs read back from disk.
///
/// The in-memory [`merge_runs`] cannot be reused: it merges runs that are
/// already one concatenated block, which is exactly the thing a spilled sort
/// does not have. What *is* shared is the shape -- a hand-rolled sift-down
/// (see [`kway`]) rather than a `BinaryHeap`, so the comparator can read a key
/// arena in place instead of owning a tuple per heap item.
///
/// Ties break toward the lower run index, which is the stable tie-break: runs
/// are written in input order, so every row of run `i` precedes every row of
/// run `j > i`. That is what makes a spilled sort return *the same* order as
/// the in-memory one and not merely an equivalent one --
/// `a_spilled_sort_matches_the_in_memory_one_exactly` pins it.
pub(crate) struct RunMerge<'a> {
    keys: &'a [SortKey],
    at: Vec<KeyAt>,
    ctx: &'a QueryContext,
    cursors: Vec<Cursor>,
    /// The head row's key tuple per run, `nk` apart. Refreshed once per
    /// advance, i.e. once per output row -- not once per comparison, which is
    /// `log(fanin)` times more often, and not once per *loaded row*, which was
    /// the first version here and cost a `Value` (24 B) per row per key of
    /// resident memory for rows the merge might never reach.
    heads: Vec<Value>,
    /// Live runs. Only `.0` is compared; `sift_down` is shared with [`kway`].
    heap: Vec<(u32, u32)>,
    /// Per output row: which piece of the concatenation it came from, and
    /// where in it. Packed so the two survive a piece closing mid-block.
    src: Vec<u64>,
    piece_base: Vec<u32>,
    perm: Vec<u32>,
    /// The piece each run currently has open, or [`NO_PIECE`]. A field and not
    /// a local so the merge allocates nothing per output block.
    open: Vec<u32>,
    block_rows: usize,
    guard: MemGuard,
    /// Held so the files outlive the merge and die with it.
    _set: RunSet,
}

/// A cursor with no piece open in the block being assembled.
const NO_PIECE: u32 = u32::MAX;

impl<'a> RunMerge<'a> {
    /// Open a merge over `set`, collapsing it to a mergeable fan-in first.
    fn open(mut set: RunSet, keys: &'a [SortKey], ctx: &'a QueryContext) -> Result<RunMerge<'a>> {
        let f = fanin(ctx, &set);
        while set.len() > f {
            set = merge_pass(set, keys, ctx, f)?;
        }
        RunMerge::over(set, keys, ctx)
    }

    /// Open a merge over a set that already fits in one pass.
    fn over(set: RunSet, keys: &'a [SortKey], ctx: &'a QueryContext) -> Result<RunMerge<'a>> {
        let schema = set.schema.clone().unwrap_or_else(Schema::empty);
        let at = key_positions(keys, schema.len());
        let mut cursors = Vec::with_capacity(set.len());
        for p in &set.runs {
            cursors.push(Cursor {
                src: spill::RunReader::open(p, schema.clone())?,
                cur: Block::empty(&schema),
                kx: Vec::new(),
                pos: 0,
                start: 0,
                done: false,
            });
        }
        let n = cursors.len();
        let mut m = RunMerge {
            keys,
            at,
            ctx,
            heads: vec![Value::Null; n * keys.len()],
            heap: Vec::with_capacity(n),
            src: Vec::with_capacity(set.block_rows),
            piece_base: Vec::new(),
            perm: Vec::with_capacity(set.block_rows),
            open: vec![NO_PIECE; n],
            block_rows: set.block_rows,
            cursors,
            guard: MemGuard::new(ctx, "the sort merge buffers"),
            _set: set,
        };
        for i in 0..n {
            m.fill(i)?;
            if !m.cursors[i].done {
                m.refresh(i);
                m.heap.push((i as u32, 0));
            }
        }
        let (heap, heads, keys) = (&mut m.heap, &m.heads, m.keys);
        for i in (0..heap.len() / 2).rev() {
            sift_down(heap, i, &mut |a, b| less(heads, keys, a, b));
        }
        Ok(m)
    }

    /// Publish run `i`'s head key tuple for the comparator.
    #[inline]
    fn refresh(&mut self, i: usize) {
        let nk = self.keys.len();
        let c = &self.cursors[i];
        for (k, a) in self.at.iter().enumerate() {
            self.heads[i * nk + k] = match a {
                KeyAt::Col(j) => c.cur.column(*j).value(c.pos),
                KeyAt::Expr(j) => c.kx[*j].value(c.pos),
            };
        }
    }

    /// Advance run `i` to its next spilled block.
    ///
    /// Every caller has just closed the run's piece, so `cur` is fully
    /// consumed *and* the rows this output block took from it have already
    /// been copied into the concatenation -- which is what lets the block be
    /// replaced outright rather than grown, and what bounds an open run at one
    /// spilled block. The first version kept the tail and extended, which cost
    /// a second copy of every row for a case that cannot arise.
    fn fill(&mut self, i: usize) -> Result<()> {
        let (keys, at) = (self.keys, &self.at);
        let c = &mut self.cursors[i];
        debug_assert_eq!(c.pos, c.start, "fill on a run with an open piece");
        loop {
            let Some(b) = c.src.next()? else {
                // Spent, but `cur` is deliberately left alone: `pos` and
                // `start` still describe an empty contribution to the block
                // being assembled, and the assembly walks every run.
                c.done = true;
                return Ok(());
            };
            if b.rows() == 0 {
                continue;
            }
            c.kx.clear();
            for (k, a) in keys.iter().zip(at) {
                if matches!(a, KeyAt::Expr(_)) {
                    c.kx.push(expr::eval(&k.expr, &b)?);
                }
            }
            c.cur = b;
            c.pos = 0;
            c.start = 0;
            return Ok(());
        }
    }

    /// The next merged block, or `None` once every run is spent.
    pub(crate) fn next(&mut self) -> Result<Option<Block>> {
        self.ctx.check()?;
        if self.heap.is_empty() {
            return Ok(None);
        }
        self.src.clear();
        self.piece_base.clear();
        // One piece per (run, loaded block) touched by this output block. A
        // run whose block runs out mid-block closes its piece and opens
        // another, which is why the piece id has to ride along with the row.
        self.open.fill(NO_PIECE);
        let mut scratch: Option<Block> = None;
        let mut at = 0u32;
        for c in self.cursors.iter_mut() {
            c.start = c.pos;
        }

        while self.src.len() < self.block_rows {
            let Some(&(top, _)) = self.heap.first() else { break };
            let i = top as usize;
            if self.open[i] == NO_PIECE {
                self.open[i] = self.piece_base.len() as u32;
                self.piece_base.push(0);
            }
            let off = (self.cursors[i].pos - self.cursors[i].start) as u64;
            self.src.push(((self.open[i] as u64) << 32) | off);
            self.cursors[i].pos += 1;

            if self.cursors[i].pos >= self.cursors[i].cur.rows() {
                // The block is exhausted. Close the piece *before* `fill` can
                // slice the block out from under it.
                let p = self.open[i];
                close_piece(&mut self.cursors[i], p, &mut self.piece_base, &mut at, &mut scratch)?;
                self.open[i] = NO_PIECE;
                self.fill(i)?;
            }
            let done = self.cursors[i].done;
            if !done {
                self.refresh(i);
            }
            let (heap, heads, keys) = (&mut self.heap, &self.heads, self.keys);
            if done {
                // The run is spent. Standard shrink: the tail takes the root.
                let last = heap.pop().unwrap_or_default();
                if !heap.is_empty() {
                    heap[0] = last;
                    sift_down(heap, 0, &mut |a, b| less(heads, keys, a, b));
                }
            } else {
                sift_down(heap, 0, &mut |a, b| less(heads, keys, a, b));
            }
        }
        if self.src.is_empty() {
            return Ok(None);
        }
        for i in 0..self.cursors.len() {
            let p = self.open[i];
            if p != NO_PIECE {
                close_piece(&mut self.cursors[i], p, &mut self.piece_base, &mut at, &mut scratch)?;
            }
        }
        let Some(scratch) = scratch else { return Ok(None) };

        // Two vectorized copies and no `Value` round trip: the concatenation
        // above, then one gather. Feeding a `ColumnBuilder` a value per cell
        // -- the obvious way to interleave rows from several runs -- costs a
        // heap round trip per string and cannot use `Column::take` at all.
        let (perm, src, base) = (&mut self.perm, &self.src, &self.piece_base);
        perm.clear();
        perm.extend(src.iter().map(|&x| base[(x >> 32) as usize] + x as u32));
        let live = self
            .cursors
            .iter()
            .map(|c| c.cur.bytes() + c.kx.iter().map(|x| x.bytes()).sum::<usize>())
            .sum::<usize>()
            + scratch.bytes()
            + self.perm.capacity() * size_of::<u32>()
            + self.src.capacity() * size_of::<u64>();
        self.guard.grow_to(live)?;
        Ok(Some(scratch.take(&self.perm)))
    }
}

/// Append run `c`'s contribution `[start, pos)` to the concatenation and
/// record where it landed.
fn close_piece(
    c: &mut Cursor,
    piece: u32,
    base: &mut [u32],
    at: &mut u32,
    scratch: &mut Option<Block>,
) -> Result<()> {
    base[piece as usize] = *at;
    if c.pos == c.start {
        return Ok(());
    }
    *at += (c.pos - c.start) as u32;
    let part = c.cur.slice(c.start, c.pos);
    match scratch {
        None => *scratch = Some(part),
        Some(s) => s.extend(&part)?,
    }
    c.start = c.pos;
    Ok(())
}

/// Strict order over two live runs, ties broken toward the lower run index.
#[inline]
fn less(heads: &[Value], keys: &[SortKey], a: u32, b: u32) -> bool {
    let nk = keys.len();
    let (i, j) = (a as usize * nk, b as usize * nk);
    match compare_keys(&heads[i..i + nk], &heads[j..j + nk], keys) {
        Ordering::Less => true,
        Ordering::Greater => false,
        Ordering::Equal => a < b,
    }
}

/// How many runs may be open at once.
///
/// Each open run holds one spilled block, charged at two here so an
/// expression sort key's evaluated column has somewhere to live. The fan-in is
/// therefore a memory question and it answers it out of the same budget
/// everything else here comes from: a tight budget merges two runs at a time
/// and takes more passes,
/// a generous one merges 64 and takes a single pass. Capped at 64 because past
/// that the heap stops fitting in cache and the sift costs more than the extra
/// pass it saves.
fn fanin(ctx: &QueryContext, set: &RunSet) -> usize {
    let per_run = 2 * set.block_rows * set.row_bytes;
    (((ctx.mem.limit().max(0) as usize) / 2) / per_run.max(1)).clamp(2, 64)
}

/// Collapse `set` to at most `fanin` runs by merging groups of them.
///
/// The extra passes an external sort needs when the run count outgrows what
/// one merge can hold open. Each pass is the same streaming merge, writing to
/// a run file instead of to the caller, and it unlinks each group as soon as
/// it has been consumed so peak disk stays near one copy of the relation
/// rather than two.
fn merge_pass(set: RunSet, keys: &[SortKey], ctx: &QueryContext, fanin: usize) -> Result<RunSet> {
    let mut out = RunSet::new()?;
    out.rows = set.rows;
    out.schema = set.schema.clone();
    out.block_rows = set.block_rows;
    out.row_bytes = set.row_bytes;
    for group in set.runs.chunks(fanin) {
        let part = RunSet {
            dir: spill::SpillDir::borrowed(&set.dir),
            runs: group.to_vec(),
            schema: set.schema.clone(),
            block_rows: set.block_rows,
            row_bytes: set.row_bytes,
            rows: set.rows,
        };
        let mut m = RunMerge::over(part, keys, ctx)?;
        let mut w = out.dir.create()?;
        while let Some(b) = m.next()? {
            w.push(&b)?;
        }
        out.runs.push(w.finish()?);
        drop(m);
        for p in group {
            let _ = std::fs::remove_file(p);
        }
    }
    Ok(out)
}
/// Temp files for the operators that no longer have to fit in RAM.
///
/// Blocks go out through [`crate::persist::writer::put_block`] -- the same
/// self-describing codec the write-ahead log uses -- behind an eight-byte
/// length/row frame. What is deliberately *not* borrowed from
/// [`crate::persist::store`] is `atomic_write`'s durability: a spill file is
/// private to one query, is read exactly once, and must **not** survive the
/// process, so a temp copy, an `fsync` and a rename would be three costs with
/// nothing to buy. What is borrowed is the part that matters -- the pid-plus-
/// counter naming, so a directory left behind by a `SIGKILL` is attributable,
/// and `io_err`, so a full disk reads the same here as everywhere else.
pub(crate) mod spill {
    use std::fs::File;
    use std::io::{BufReader, Read, Write};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::common::Result;
    use crate::persist::format::{Reader, Writer};
    use crate::persist::{reader, store, writer};
    use crate::types::{Block, Field, Schema};

    /// Bytes buffered before a run file sees a `write` syscall.
    const FLUSH_AT: usize = 256 << 10;

    static SEQ: AtomicU64 = AtomicU64::new(0);

    #[cfg(test)]
    thread_local! {
        /// Every spill directory this thread has created. Thread-local rather
        /// than global precisely so a test can assert "it spilled, and it took
        /// its files with it" while the rest of the suite spills in parallel.
        pub(crate) static SPILLED: std::cell::RefCell<Vec<PathBuf>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    /// A directory of spill files that unlinks itself when dropped.
    pub(crate) struct SpillDir {
        root: PathBuf,
        next: u64,
        /// A second handle on someone else's directory: it may create files
        /// there but must not delete the tree. See [`SpillDir::borrowed`].
        owned: bool,
    }

    impl SpillDir {
        pub(crate) fn new() -> Result<SpillDir> {
            let n = SEQ.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir()
                .join(format!("granular-spill-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&root)
                .map_err(|e| store::io_err("create the spill directory", &root, e))?;
            #[cfg(test)]
            SPILLED.with(|s| s.borrow_mut().push(root.clone()));
            Ok(SpillDir { root, next: 0, owned: true })
        }

        /// A non-owning view, so a merge pass can read run files out of a
        /// directory whose lifetime belongs to the set it is consuming.
        pub(crate) fn borrowed(d: &SpillDir) -> SpillDir {
            SpillDir { root: d.root.clone(), next: u64::MAX / 2, owned: false }
        }

        /// A fresh, empty run file.
        pub(crate) fn create(&mut self) -> Result<RunWriter> {
            self.create_buffered(FLUSH_AT)
        }

        /// [`SpillDir::create`] with a caller-chosen write buffer. A hash
        /// aggregate holds one writer per partition open at once, so its
        /// buffers have to be sized against the budget rather than each
        /// claiming the 256 KiB a single sorted run happily takes.
        pub(crate) fn create_buffered(&mut self, flush_at: usize) -> Result<RunWriter> {
            let path = self.root.join(format!("run-{:06}.grun", self.next));
            self.next += 1;
            RunWriter::create(path, flush_at)
        }
    }

    impl Drop for SpillDir {
        fn drop(&mut self) {
            // Best effort by necessity: `Drop` cannot report, and a leaked
            // temp directory is a worse outcome than a lost error message.
            if self.owned {
                let _ = std::fs::remove_dir_all(&self.root);
            }
        }
    }

    /// Append-only writer over one run file.
    pub(crate) struct RunWriter {
        file: File,
        path: PathBuf,
        w: Writer,
        flush_at: usize,
    }

    impl RunWriter {
        fn create(path: PathBuf, flush_at: usize) -> Result<RunWriter> {
            let file =
                File::create(&path).map_err(|e| store::io_err("create the spill file", &path, e))?;
            let w = Writer::with_capacity(flush_at + (flush_at >> 2));
            Ok(RunWriter { file, path, w, flush_at })
        }

        pub(crate) fn push(&mut self, b: &Block) -> Result<()> {
            if b.rows() == 0 {
                return Ok(());
            }
            let at = self.w.pos();
            self.w.u32(0);
            // The row count is framed separately because `put_block` cannot
            // carry it for a zero-column block, and `SELECT count(*)` scans
            // produce exactly those.
            self.w.u32(b.rows() as u32);
            writer::put_block(&mut self.w, b);
            let len = self.w.pos() - at - 8;
            self.w.patch_u32(at, len as u32)?;
            if self.w.pos() >= self.flush_at {
                self.flush()?;
            }
            Ok(())
        }

        fn flush(&mut self) -> Result<()> {
            let cap = self.flush_at + (self.flush_at >> 2);
            let bytes = std::mem::replace(&mut self.w, Writer::with_capacity(cap)).finish();
            if bytes.is_empty() {
                return Ok(());
            }
            self.file
                .write_all(&bytes)
                .map_err(|e| store::io_err("write the spill file", &self.path, e))
        }

        /// Close the run and hand back its path.
        pub(crate) fn finish(mut self) -> Result<PathBuf> {
            self.flush()?;
            Ok(self.path)
        }
    }

    /// Sequential reader over one run file, one block per `next`.
    pub(crate) struct RunReader {
        file: BufReader<File>,
        path: PathBuf,
        schema: Schema,
        body: Vec<u8>,
    }

    impl RunReader {
        pub(crate) fn open(path: &Path, schema: Schema) -> Result<RunReader> {
            let f =
                File::open(path).map_err(|e| store::io_err("open the spill file", path, e))?;
            Ok(RunReader {
                file: BufReader::with_capacity(64 << 10, f),
                path: path.to_path_buf(),
                schema,
                body: Vec::new(),
            })
        }

        pub(crate) fn next(&mut self) -> Result<Option<Block>> {
            let mut hdr = [0u8; 8];
            match self.file.read_exact(&mut hdr) {
                Ok(()) => {}
                // The only clean end: a run file is a whole number of frames.
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
                Err(e) => return Err(store::io_err("read the spill file", &self.path, e)),
            }
            let len = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as usize;
            let rows = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
            // `resize` and not `clear` + `resize`: `read_exact` overwrites
            // every byte, so zeroing the ones it is about to fill is waste.
            self.body.resize(len, 0);
            self.file
                .read_exact(&mut self.body)
                .map_err(|e| store::io_err("read the spill file", &self.path, e))?;
            let mut r = Reader::new(&self.body);
            let mut b = reader::get_block(&mut r, &self.schema)?;
            if b.width() == 0 {
                b.set_rows(rows);
            }
            Ok(Some(b))
        }
    }

    /// The schema a spilled block will be read back against. Only the physical
    /// kinds are checked on the way in, and the names are never looked at.
    pub(crate) fn schema_of(b: &Block) -> Schema {
        Schema::new_unchecked(
            b.columns
                .iter()
                .enumerate()
                .map(|(i, c)| Field::new(format!("c{i}"), c.ty.clone()))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::Values;
    use crate::planner::logical::BoundExpr;
    use crate::types::{DataType, Field, Value};

    fn key(i: usize, ty: DataType, asc: bool, nulls_first: bool) -> SortKey {
        SortKey {
            expr: BoundExpr::Column { index: i, ty, name: format!("c{i}") },
            asc,
            nulls_first,
        }
    }

    fn sorted(rows: Vec<Vec<Value>>, schema: Schema, keys: Vec<SortKey>) -> Vec<Vec<Value>> {
        let ctx = QueryContext::new();
        let mut s = Sort::new(Box::new(Values::new(&rows, &schema)), &keys, &ctx);
        drain_op(&mut s)
    }

    /// The same input through the top-K path, which must agree with `sorted`
    /// on its first `k` rows.
    fn top_k(
        rows: Vec<Vec<Value>>,
        schema: Schema,
        keys: Vec<SortKey>,
        k: usize,
    ) -> Vec<Vec<Value>> {
        let ctx = QueryContext::new();
        let mut s = Sort::top_k(Box::new(Values::new(&rows, &schema)), &keys, k, &ctx);
        drain_op(&mut s)
    }

    fn drain_op(s: &mut dyn Operator) -> Vec<Vec<Value>> {
        let mut out = Vec::new();
        while let Some(b) = s.next().unwrap() {
            for i in 0..b.rows() {
                out.push((0..b.width()).map(|c| b.column(c).value(i)).collect());
            }
        }
        out
    }

    fn ints_schema() -> Schema {
        Schema::new(vec![Field::new("a", DataType::Int64)]).unwrap()
    }

    fn ints(vs: &[i64]) -> Vec<Vec<Value>> {
        vs.iter().map(|&i| vec![Value::Int(i)]).collect()
    }

    fn flat(rows: &[Vec<Value>]) -> Vec<Value> {
        rows.iter().map(|r| r[0].clone()).collect()
    }

    #[test]
    fn ascending_over_the_radix_path() {
        let got = sorted(
            ints(&[5, -2, 9, 0, -100]),
            ints_schema(),
            vec![key(0, DataType::Int64, true, true)],
        );
        assert_eq!(
            flat(&got),
            vec![Value::Int(-100), Value::Int(-2), Value::Int(0), Value::Int(5), Value::Int(9)]
        );
    }

    #[test]
    fn descending_over_the_radix_path() {
        let got = sorted(
            ints(&[5, -2, 9]),
            ints_schema(),
            vec![key(0, DataType::Int64, false, true)],
        );
        assert_eq!(flat(&got), vec![Value::Int(9), Value::Int(5), Value::Int(-2)]);
    }

    #[test]
    fn descending_stays_stable_on_ties() {
        // Two columns, sort on the first descending: the second must keep its
        // original relative order within each tie group.
        let schema = Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("seq", DataType::Int64),
        ])
        .unwrap();
        let rows: Vec<Vec<Value>> = (0..10i64)
            .map(|i| vec![Value::Int(i % 2), Value::Int(i)])
            .collect();
        let got = sorted(rows, schema, vec![key(0, DataType::Int64, false, true)]);
        let seq: Vec<i64> = got.iter().map(|r| r[1].as_i64().unwrap()).collect();
        assert_eq!(seq, vec![1, 3, 5, 7, 9, 0, 2, 4, 6, 8]);
    }

    #[test]
    fn radix_and_comparison_paths_agree() {
        use crate::common::splitmix64;
        let vs: Vec<i64> = (0..3_000u64).map(|i| splitmix64(i) as i64 / 7).collect();
        let asc = sorted(
            ints(&vs),
            ints_schema(),
            vec![key(0, DataType::Int64, true, true)],
        );
        let mut want = vs.clone();
        want.sort();
        assert_eq!(flat(&asc), want.into_iter().map(Value::Int).collect::<Vec<_>>());
    }

    #[test]
    fn floats_sort_by_value_including_negatives() {
        let schema = Schema::new(vec![Field::new("f", DataType::Float64)]).unwrap();
        let rows: Vec<Vec<Value>> = [1.5f64, -0.25, 1e300, -1e300, 0.0]
            .iter()
            .map(|&f| vec![Value::Float(f)])
            .collect();
        let got = sorted(rows, schema, vec![key(0, DataType::Float64, true, true)]);
        let fs: Vec<f64> = got.iter().map(|r| r[0].as_f64().unwrap()).collect();
        assert_eq!(fs, vec![-1e300, -0.25, 0.0, 1.5, 1e300]);
    }

    #[test]
    fn strings_take_the_comparison_path() {
        let schema = Schema::new(vec![Field::new("s", DataType::String)]).unwrap();
        let rows: Vec<Vec<Value>> = ["pear", "apple", "fig"]
            .iter()
            .map(|s| vec![Value::str(*s)])
            .collect();
        let got = sorted(rows, schema, vec![key(0, DataType::String, true, true)]);
        assert_eq!(
            flat(&got),
            vec![Value::str("apple"), Value::str("fig"), Value::str("pear")]
        );
    }

    #[test]
    fn nulls_first_and_last_are_absolute_not_direction_relative() {
        let schema =
            Schema::new(vec![Field::new("a", DataType::Nullable(Box::new(DataType::Int64)))])
                .unwrap();
        let rows = vec![
            vec![Value::Int(2)],
            vec![Value::Null],
            vec![Value::Int(1)],
        ];

        let f = sorted(rows.clone(), schema.clone(), vec![key(0, DataType::Int64, true, true)]);
        assert_eq!(flat(&f), vec![Value::Null, Value::Int(1), Value::Int(2)]);

        let l = sorted(rows.clone(), schema.clone(), vec![key(0, DataType::Int64, true, false)]);
        assert_eq!(flat(&l), vec![Value::Int(1), Value::Int(2), Value::Null]);

        // descending, NULLS FIRST: nulls still lead
        let df = sorted(rows.clone(), schema.clone(), vec![key(0, DataType::Int64, false, true)]);
        assert_eq!(flat(&df), vec![Value::Null, Value::Int(2), Value::Int(1)]);

        // descending, NULLS LAST
        let dl = sorted(rows, schema, vec![key(0, DataType::Int64, false, false)]);
        assert_eq!(flat(&dl), vec![Value::Int(2), Value::Int(1), Value::Null]);
    }

    #[test]
    fn multi_key_sorts_major_to_minor_with_mixed_directions() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Int64),
        ])
        .unwrap();
        let rows = vec![
            vec![Value::Int(1), Value::Int(1)],
            vec![Value::Int(2), Value::Int(5)],
            vec![Value::Int(1), Value::Int(9)],
            vec![Value::Int(2), Value::Int(2)],
        ];
        let got = sorted(
            rows,
            schema,
            vec![key(0, DataType::Int64, true, true), key(1, DataType::Int64, false, true)],
        );
        let pairs: Vec<(i64, i64)> = got
            .iter()
            .map(|r| (r[0].as_i64().unwrap(), r[1].as_i64().unwrap()))
            .collect();
        assert_eq!(pairs, vec![(1, 9), (1, 1), (2, 5), (2, 2)]);
    }

    #[test]
    fn sorting_an_expression_not_just_a_column() {
        use crate::sql::ast::BinaryOp;
        let schema = ints_schema();
        let rows = ints(&[1, -5, 3]);
        // ORDER BY a * -1 ASC: keys are -1, 5, -3, so the rows come out
        // largest-`a` first.
        let keys = vec![SortKey {
            expr: BoundExpr::Binary {
                left: Box::new(BoundExpr::Column {
                    index: 0,
                    ty: DataType::Int64,
                    name: "a".into(),
                }),
                op: BinaryOp::Multiply,
                right: Box::new(BoundExpr::lit(Value::Int(-1))),
                ty: DataType::Int64,
            },
            asc: true,
            nulls_first: true,
        }];
        let got = sorted(rows, schema, keys);
        assert_eq!(flat(&got), vec![Value::Int(3), Value::Int(1), Value::Int(-5)]);
    }

    /// A second key that is constant, purely to force the comparison path.
    fn const_key() -> SortKey {
        SortKey { expr: BoundExpr::lit(Value::Int(0)), asc: true, nulls_first: true }
    }

    #[test]
    fn probe_radix_path_is_stable_on_plus_and_minus_zero() {
        // `Value::cmp` collapses -0.0 == 0.0, so all four keys tie and a
        // stable sort must return the rows in input order. The radix path
        // sorts on `f64_to_lane`, which separates the two zeros.
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64),
            Field::new("x", DataType::Float64),
        ])
        .unwrap();
        let rows = vec![
            vec![Value::Int(1), Value::Float(0.0)],
            vec![Value::Int(2), Value::Float(-0.0)],
            vec![Value::Int(3), Value::Float(0.0)],
            vec![Value::Int(4), Value::Float(-0.0)],
        ];
        let radix = sorted(rows.clone(), schema.clone(), vec![key(1, DataType::Float64, true, true)]);
        let cmp = sorted(
            rows,
            schema,
            vec![key(1, DataType::Float64, true, true), const_key()],
        );
        let ids = |v: &[Vec<Value>]| -> Vec<i64> {
            v.iter().map(|r| r[0].as_i64().unwrap()).collect()
        };
        assert_eq!(ids(&cmp), vec![1, 2, 3, 4], "comparison path is stable");
        assert_eq!(
            ids(&radix),
            ids(&cmp),
            "the radix path returns a different order than the comparison path"
        );
    }

    #[test]
    fn probe_radix_path_puts_nan_last_like_value_ordering() {
        // `Value::cmp` sorts every NaN last regardless of sign; the radix lane
        // codec sends a negative NaN below -inf.
        let schema = Schema::new(vec![Field::new("x", DataType::Float64)]).unwrap();
        let rows: Vec<Vec<Value>> = [-f64::NAN, -1.0, f64::NAN, 1.0]
            .iter()
            .map(|&f| vec![Value::Float(f)])
            .collect();
        let got = sorted(rows, schema, vec![key(0, DataType::Float64, true, true)]);
        let fs: Vec<f64> = got.iter().map(|r| r[0].as_f64().unwrap()).collect();
        assert!(
            !fs[0].is_nan() && !fs[1].is_nan(),
            "a NaN sorted before the finite values: {fs:?}"
        );
    }

    #[test]
    fn empty_input_sorts_to_nothing() {
        let got = sorted(vec![], ints_schema(), vec![key(0, DataType::Int64, true, true)]);
        assert!(got.is_empty());
    }

    #[test]
    fn no_keys_preserves_input_order() {
        let got = sorted(ints(&[3, 1, 2]), ints_schema(), vec![]);
        assert_eq!(flat(&got), vec![Value::Int(3), Value::Int(1), Value::Int(2)]);
    }

    #[test]
    fn output_is_rebatched_at_block_size() {
        use crate::common::BLOCK_SIZE;
        let schema = ints_schema();
        let rows = ints(&(0..BLOCK_SIZE as i64 + 100).rev().collect::<Vec<_>>());
        let keys = vec![key(0, DataType::Int64, true, true)];
        let ctx = QueryContext::new();
        let mut s = Sort::new(Box::new(Values::new(&rows, &schema)), &keys, &ctx);
        let mut sizes = Vec::new();
        while let Some(b) = s.next().unwrap() {
            sizes.push(b.rows());
        }
        assert_eq!(sizes, vec![BLOCK_SIZE, 100]);
    }

    // ----------------------------------------------------------------- top-K

    #[test]
    fn top_k_agrees_with_a_full_sort_including_ties() {
        use crate::common::{splitmix64, BLOCK_SIZE};
        // Deliberately few distinct keys over several blocks, so almost every
        // row ties with something: that is where a top-K that merges
        // unstably, or that admits rows equal to the current worst, diverges
        // from a full sort. `seq` is the tie-breaking witness.
        let schema = Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("seq", DataType::Int64),
        ])
        .unwrap();
        let n = BLOCK_SIZE as i64 * 3 + 17;
        let rows: Vec<Vec<Value>> = (0..n)
            .map(|i| vec![Value::Int((splitmix64(i as u64) % 40) as i64), Value::Int(i)])
            .collect();
        for asc in [true, false] {
            let keys = vec![key(0, DataType::Int64, asc, true)];
            let full = sorted(rows.clone(), schema.clone(), keys.clone());
            for k in [1usize, 5, 100, 8192, 40_000] {
                let got = top_k(rows.clone(), schema.clone(), keys.clone(), k);
                let want = &full[..k.min(full.len())];
                assert_eq!(got.len(), want.len(), "k={k} asc={asc}");
                assert_eq!(got, want, "top-{k} (asc={asc}) disagrees with a full sort");
            }
        }
    }

    #[test]
    fn top_k_handles_nulls_strings_and_several_keys() {
        // The threshold test runs through `compare_keys`, so it has to honour
        // NULLS FIRST/LAST and per-key direction exactly like the full sort.
        let schema = Schema::new(vec![
            Field::new("s", DataType::Nullable(Box::new(DataType::String))),
            Field::new("n", DataType::Int64),
        ])
        .unwrap();
        let names = ["pear", "apple", "fig", "date"];
        let rows: Vec<Vec<Value>> = (0..1_000i64)
            .map(|i| {
                let s = if i % 7 == 0 { Value::Null } else { Value::str(names[i as usize % 4]) };
                vec![s, Value::Int(i % 13)]
            })
            .collect();
        for nulls_first in [true, false] {
            let keys = vec![
                key(0, DataType::String, true, nulls_first),
                key(1, DataType::Int64, false, nulls_first),
            ];
            let full = sorted(rows.clone(), schema.clone(), keys.clone());
            let got = top_k(rows.clone(), schema.clone(), keys, 9);
            assert_eq!(got, full[..9].to_vec(), "nulls_first={nulls_first}");
        }
    }

    #[test]
    fn top_k_keeps_a_bounded_buffer() {
        // The point of the fusion: a 100k-row input with LIMIT 3 must never
        // hold 100k rows. The bound is `trim_at + BLOCK_SIZE` rows, which for
        // a small k is two blocks -- ~590 KiB here once the sort scratch is
        // counted -- against 3.6 MiB for the full sort of the same input.
        use crate::common::BLOCK_SIZE;
        let schema = ints_schema();
        let n = 100_000i64;
        let rows = ints(&(0..n).rev().collect::<Vec<_>>());
        let keys = vec![key(0, DataType::Int64, true, true)];
        let ctx = QueryContext::with_budget(BLOCK_SIZE as i64 * 3 * 64);
        let mut s = Sort::top_k(Box::new(Values::new(&rows, &schema)), &keys, 3, &ctx);
        let got = drain_op(&mut s);
        assert_eq!(flat(&got), vec![Value::Int(0), Value::Int(1), Value::Int(2)]);
        drop(s);
        assert_eq!(ctx.mem.used(), 0);
        assert!(spilled_dirs().is_empty(), "a bounded top-K spilled");

        // ... whereas the same query without the fusion does not fit -- and
        // used to fail for it. Inverted: the full sort now spills instead, so
        // what "does not fit" buys is a slower answer, not no answer. The
        // fusion is still worth having; it is worth ~100k rows of temp file.
        let ctx2 = QueryContext::with_budget((BLOCK_SIZE as i64 + 64) * 64);
        let mut full = Sort::new(Box::new(Values::new(&rows, &schema)), &keys, &ctx2);
        let all = drain_op(&mut full);
        assert_eq!(all.len(), n as usize);
        assert_eq!(flat(&all[..3]), vec![Value::Int(0), Value::Int(1), Value::Int(2)]);
        assert!(!spilled_dirs().is_empty(), "a 512 KiB budget held a 100k-row sort?");
        drop(full);
        assert_eq!(ctx2.mem.used(), 0);
        assert_no_temp_files_left();
    }

    // --------------------------------------------------------------- spilling

    /// Spill directories created on this thread since the last check, and a
    /// reset. Thread-local, so a parallel test run cannot perturb it.
    fn spilled_dirs() -> Vec<std::path::PathBuf> {
        spill::SPILLED.with(|s| s.borrow().clone())
    }

    fn clear_spilled() {
        spill::SPILLED.with(|s| s.borrow_mut().clear());
    }

    /// Every spill directory this thread opened is gone. A spill that leaks
    /// files on the error path is a production incident, so this is asserted
    /// after the cancelled and over-budget cases too, not only the happy one.
    fn assert_no_temp_files_left() {
        for d in spilled_dirs() {
            assert!(!d.exists(), "spill directory {} outlived its query", d.display());
        }
        clear_spilled();
    }

    /// `n` rows over `d` distinct keys, so most rows tie -- the shape a merge
    /// that breaks ties by run rather than by input position gets wrong.
    fn tied_rows(n: i64, d: u64) -> (Schema, Vec<Vec<Value>>) {
        use crate::common::splitmix64;
        let schema = Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("seq", DataType::Int64),
        ])
        .unwrap();
        let rows = (0..n)
            .map(|i| vec![Value::Int((splitmix64(i as u64) % d) as i64), Value::Int(i)])
            .collect();
        (schema, rows)
    }

    #[test]
    fn a_spilled_sort_matches_the_in_memory_one_exactly() {
        // Not "the same multiset" and not "sorted": the same rows in the same
        // order, ties included. `seq` is the witness -- a merge that broke ties
        // by run would pass every other check here and still reorder every tie.
        clear_spilled();
        let n = BLOCK_SIZE as i64 * 12 + 37;
        let (schema, rows) = tied_rows(n, 24);
        for asc in [true, false] {
            for nulls_first in [true, false] {
                let keys = vec![key(0, DataType::Int64, asc, nulls_first)];
                let want = sorted(rows.clone(), schema.clone(), keys.clone());
                let ctx = QueryContext::with_budget(200 << 10);
                let mut s = Sort::new(Box::new(Values::new(&rows, &schema)), &keys, &ctx);
                let got = drain_op(&mut s);
                assert_eq!(got.len(), n as usize);
                assert_eq!(got, want, "spilled sort disagrees (asc={asc})");
                drop(s);
                assert_eq!(ctx.mem.used(), 0, "the merge kept its reservation");
            }
        }
        assert!(!spilled_dirs().is_empty(), "nothing spilled, so nothing was tested");
        assert_no_temp_files_left();
    }

    #[test]
    fn a_spilled_sort_handles_nulls_strings_and_several_keys() {
        // The comparison arm of the merge comparator: a lane cannot encode any
        // of these, and NULLS FIRST/LAST has to survive the round trip through
        // the spill file's null mask.
        clear_spilled();
        let schema = Schema::new(vec![
            Field::new("s", DataType::Nullable(Box::new(DataType::String))),
            Field::new("n", DataType::Int64),
        ])
        .unwrap();
        let names = ["pear", "apple", "fig", "date", ""];
        let rows: Vec<Vec<Value>> = (0..40_000i64)
            .map(|i| {
                let s = if i % 7 == 0 { Value::Null } else { Value::str(names[i as usize % 5]) };
                vec![s, Value::Int(i % 13)]
            })
            .collect();
        for nulls_first in [true, false] {
            let keys = vec![
                key(0, DataType::String, true, nulls_first),
                key(1, DataType::Int64, false, nulls_first),
            ];
            let want = sorted(rows.clone(), schema.clone(), keys.clone());
            let ctx = QueryContext::with_budget(1 << 20);
            let mut s = Sort::new(Box::new(Values::new(&rows, &schema)), &keys, &ctx);
            assert_eq!(drain_op(&mut s), want, "nulls_first={nulls_first}");
        }
        assert!(!spilled_dirs().is_empty(), "nothing spilled");
        assert_no_temp_files_left();
    }

    #[test]
    fn a_spilled_sort_carries_every_column_type_through_the_temp_file() {
        // The spill file is the only place in this operator where a block
        // stops being a block, so every physical kind has to survive it --
        // including a float's -0.0 and NaN, which a decimal round trip loses.
        clear_spilled();
        let schema = Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("u", DataType::UInt64),
            Field::new("f", DataType::Float64),
            Field::new("s", DataType::String),
            Field::new("n", DataType::Nullable(Box::new(DataType::Int64))),
        ])
        .unwrap();
        let odd = [0.0f64, -0.0, f64::NAN, f64::INFINITY, -1e300];
        let rows: Vec<Vec<Value>> = (0..30_000i64)
            .map(|i| {
                vec![
                    Value::Int(crate::common::splitmix64(i as u64) as i64 % 1000),
                    Value::UInt(i as u64),
                    Value::Float(odd[i as usize % odd.len()]),
                    Value::str(format!("row-{}", i % 97)),
                    if i % 5 == 0 { Value::Null } else { Value::Int(i) },
                ]
            })
            .collect();
        let keys = vec![key(0, DataType::Int64, true, true)];
        let want = sorted(rows.clone(), schema.clone(), keys.clone());
        let ctx = QueryContext::with_budget(2 << 20);
        let mut s = Sort::new(Box::new(Values::new(&rows, &schema)), &keys, &ctx);
        let got = drain_op(&mut s);
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            // `Value::Float(NaN) != Value::Float(NaN)` under `PartialEq`, so
            // compare the bits: that is the property the codec has to keep.
            for (a, b) in g.iter().zip(w) {
                match (a, b) {
                    (Value::Float(x), Value::Float(y)) => {
                        assert_eq!(x.to_bits(), y.to_bits(), "row {i}")
                    }
                    _ => assert_eq!(a, b, "row {i}"),
                }
            }
        }
        assert!(!spilled_dirs().is_empty(), "nothing spilled");
        assert_no_temp_files_left();
    }

    #[test]
    fn a_sort_that_spills_many_runs_merges_in_several_passes() {
        // A budget this small merges two runs at a time, so 300k rows means
        // ~18 runs and five merge passes. The point is that the extra passes
        // are reachable at all: a single-pass merge would either need every
        // run open at once or return a wrong answer.
        clear_spilled();
        let n = 300_000i64;
        let (schema, rows) = tied_rows(n, 1_000);
        let keys = vec![key(0, DataType::Int64, true, true)];
        let ctx = QueryContext::with_budget(160 << 10);
        let mut s = Sort::new(Box::new(Values::new(&rows, &schema)), &keys, &ctx);
        let got = drain_op(&mut s);
        assert_eq!(got.len(), n as usize);
        drop(s);
        let want = sorted(rows, schema, keys);
        assert_eq!(got, want);
        assert_eq!(ctx.mem.used(), 0);
        assert!(spilled_dirs().len() > 1, "only one directory, so no extra pass ran");
        assert_no_temp_files_left();
    }

    #[test]
    fn a_spilled_sort_with_no_keys_still_preserves_input_order() {
        // Degenerate but reachable: `merge_runs` has the same case. With no
        // keys every comparison ties, so the merge falls entirely through to
        // its run-index tie-break -- which is exactly why that tie-break has
        // to be "lower run first" and not something merely consistent.
        clear_spilled();
        let n = 60_000i64;
        let schema = ints_schema();
        let rows =
            ints(&(0..n).map(|i| crate::common::splitmix64(i as u64) as i64).collect::<Vec<_>>());
        let ctx = QueryContext::with_budget(400 << 10);
        let keys: Vec<SortKey> = vec![];
        let mut s = Sort::new(Box::new(Values::new(&rows, &schema)), &keys, &ctx);
        let got = drain_op(&mut s);
        drop(s);
        assert_eq!(flat(&got), rows.iter().map(|r| r[0].clone()).collect::<Vec<_>>());
        assert!(!spilled_dirs().is_empty(), "nothing spilled");
        assert_no_temp_files_left();
    }

    #[test]
    fn a_sort_whose_every_block_is_its_own_run_still_merges() {
        // The pathological shape: a budget under one block, so the buffer is
        // spilled on every block and the run count equals the block count.
        // `fanin` then has to bound the merge and `merge_pass` has to do the
        // rest, or the merge tries to open forty files against a 96 KiB
        // ceiling.
        clear_spilled();
        let n = BLOCK_SIZE as i64 * 20;
        let (schema, rows) = tied_rows(n, 64);
        let keys = vec![key(0, DataType::Int64, false, true)];
        let ctx = QueryContext::with_budget(96 << 10);
        let mut s = Sort::new(Box::new(Values::new(&rows, &schema)), &keys, &ctx);
        let got = drain_op(&mut s);
        drop(s);
        assert_eq!(got.len(), n as usize);
        assert_eq!(got, sorted(rows, schema, keys));
        assert_eq!(ctx.mem.used(), 0);
        assert_no_temp_files_left();
    }

    #[test]
    fn a_cancelled_spilling_sort_stops_and_takes_its_files_with_it() {
        clear_spilled();
        let (schema, rows) = tied_rows(200_000, 500);
        let keys = vec![key(0, DataType::Int64, true, true)];
        let ctx = QueryContext::with_budget(200 << 10);
        let mut s = Sort::new(Box::new(Values::new(&rows, &schema)), &keys, &ctx);
        // Cancelled after the first block, i.e. with runs already on disk.
        ctx.stop();
        let e = s.next().unwrap_err();
        assert!(e.to_string().contains("cancelled"), "{e}");
        drop(s);
        assert_eq!(ctx.mem.used(), 0);
        assert_no_temp_files_left();
    }

    #[test]
    fn a_spilled_sorts_batches_are_sized_by_the_budget_and_never_exceed_a_block() {
        // The merge holds two spilled blocks per open run, so a tight budget
        // buys itself a smaller batch rather than a second out-of-memory
        // error. What must not vary is the contract: every batch non-empty,
        // never above BLOCK_SIZE, and the concatenation is the whole relation.
        clear_spilled();
        let n = BLOCK_SIZE as i64 * 9 + 100;
        let (schema, rows) = tied_rows(n, 4_000);
        let keys = vec![key(0, DataType::Int64, true, true)];
        for budget in [200 << 10, 4 << 20] {
            let ctx = QueryContext::with_budget(budget);
            let mut s = Sort::new(Box::new(Values::new(&rows, &schema)), &keys, &ctx);
            let mut sizes = Vec::new();
            while let Some(b) = s.next().unwrap() {
                sizes.push(b.rows());
            }
            assert_eq!(sizes.iter().sum::<usize>(), n as usize, "budget={budget}");
            assert!(sizes.iter().all(|&r| r > 0 && r <= BLOCK_SIZE), "{sizes:?}");
            drop(s);
            assert_eq!(ctx.mem.used(), 0);
        }
        assert!(!spilled_dirs().is_empty(), "nothing spilled");
        assert_no_temp_files_left();
    }

    #[test]
    fn top_k_of_zero_and_of_more_than_the_input() {
        let schema = ints_schema();
        let rows = ints(&[3, 1, 2]);
        let keys = vec![key(0, DataType::Int64, true, true)];
        assert!(top_k(rows.clone(), schema.clone(), keys.clone(), 0).is_empty());
        let all = top_k(rows.clone(), schema.clone(), keys.clone(), 99);
        assert_eq!(flat(&all), vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        // No ORDER BY keys at all: input order, truncated.
        let none = top_k(rows, schema, vec![], 2);
        assert_eq!(flat(&none), vec![Value::Int(3), Value::Int(1)]);
    }

    #[test]
    fn a_cancelled_sort_stops_instead_of_materializing() {
        let schema = ints_schema();
        let rows = ints(&(0..50_000i64).rev().collect::<Vec<_>>());
        let keys = vec![key(0, DataType::Int64, true, true)];
        let ctx = QueryContext::new();
        ctx.stop();
        let mut s = Sort::new(Box::new(Values::new(&rows, &schema)), &keys, &ctx);
        assert!(s.next().is_err());
        let mut t = Sort::top_k(Box::new(Values::new(&rows, &schema)), &keys, 5, &ctx);
        assert!(t.next().is_err(), "the top-K path needs the checkpoint too");
    }

    // ------------------------------------------------------------ k-way merge

    /// Split `rows` into `n` contiguous runs, each sorted by `keys` -- exactly
    /// what an exchange worker hands back.
    fn runs(rows: &[Vec<Value>], schema: &Schema, keys: &[SortKey], n: usize) -> Vec<Block> {
        let ctx = QueryContext::new();
        let mut out = Vec::new();
        for i in 0..n {
            let (lo, hi) = (rows.len() * i / n, rows.len() * (i + 1) / n);
            let mut s = Sort::new(Box::new(Values::new(&rows[lo..hi], schema)), keys, &ctx);
            let mut run: Option<Block> = None;
            while let Some(b) = s.next().unwrap() {
                match &mut run {
                    None => run = Some(b),
                    Some(a) => a.extend(&b).unwrap(),
                }
            }
            out.push(run.unwrap_or_else(|| Block::empty(schema)));
        }
        out
    }

    fn merged(rows: &[Vec<Value>], schema: &Schema, keys: &[SortKey], n: usize, fetch: Option<usize>)
        -> Vec<Vec<Value>>
    {
        let ctx = QueryContext::new();
        let mut g = MemGuard::new(&ctx, "the sort buffer");
        let blocks = merge_runs(runs(rows, schema, keys, n), keys, fetch, &mut g).unwrap();
        blocks
            .iter()
            .flat_map(|b| (0..b.rows()).map(move |i| (0..b.width()).map(|c| b.column(c).value(i)).collect()))
            .collect()
    }

    #[test]
    fn merging_runs_reproduces_the_full_sort_exactly() {
        // Not "the same multiset": the same rows in the same order, ties
        // included. A merge that broke ties by run rather than by input
        // position would pass a multiset check and still reorder every tie.
        use crate::common::splitmix64;
        let schema = Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("seq", DataType::Int64),
        ])
        .unwrap();
        // Few distinct keys over many rows, so almost every row ties.
        let rows: Vec<Vec<Value>> = (0..5_000i64)
            .map(|i| vec![Value::Int((splitmix64(i as u64) % 20) as i64), Value::Int(i)])
            .collect();
        for asc in [true, false] {
            let keys = vec![key(0, DataType::Int64, asc, true)];
            let full = sorted(rows.clone(), schema.clone(), keys.clone());
            for n in [1usize, 2, 3, 7, 14, 64] {
                assert_eq!(merged(&rows, &schema, &keys, n, None), full, "n={n} asc={asc}");
                for k in [0usize, 1, 5, 1_000, 4_999, 5_000, 9_999] {
                    assert_eq!(
                        merged(&rows, &schema, &keys, n, Some(k)),
                        full[..k.min(full.len())].to_vec(),
                        "n={n} asc={asc} k={k}"
                    );
                }
            }
        }
    }

    #[test]
    fn merging_runs_handles_nulls_strings_and_several_keys() {
        // The comparison arm: a lane cannot encode any of these, so this is a
        // different code path from the one above.
        let schema = Schema::new(vec![
            Field::new("s", DataType::Nullable(Box::new(DataType::String))),
            Field::new("n", DataType::Int64),
        ])
        .unwrap();
        let names = ["pear", "apple", "fig", "date"];
        let rows: Vec<Vec<Value>> = (0..2_000i64)
            .map(|i| {
                let s = if i % 7 == 0 { Value::Null } else { Value::str(names[i as usize % 4]) };
                vec![s, Value::Int(i % 13)]
            })
            .collect();
        for nulls_first in [true, false] {
            let keys = vec![
                key(0, DataType::String, true, nulls_first),
                key(1, DataType::Int64, false, nulls_first),
            ];
            let full = sorted(rows.clone(), schema.clone(), keys.clone());
            for n in [2usize, 5, 14] {
                assert_eq!(merged(&rows, &schema, &keys, n, None), full, "n={n}");
                assert_eq!(merged(&rows, &schema, &keys, n, Some(9)), full[..9].to_vec());
            }
        }
    }

    #[test]
    fn merging_degenerate_run_sets() {
        let schema = ints_schema();
        let keys = vec![key(0, DataType::Int64, true, true)];
        let ctx = QueryContext::new();
        let mut g = MemGuard::new(&ctx, "the sort buffer");
        // No runs, all-empty runs, and a zero fetch all mean no output -- and
        // an empty result must be no blocks, not a zero-row block, or the
        // pipeline above sees a batch it has to special-case.
        assert!(merge_runs(vec![], &keys, None, &mut g).unwrap().is_empty());
        let empties = vec![Block::empty(&schema), Block::empty(&schema)];
        assert!(merge_runs(empties, &keys, None, &mut g).unwrap().is_empty());
        let rows = ints(&[3, 1, 2]);
        assert!(merge_runs(runs(&rows, &schema, &keys, 2), &keys, Some(0), &mut g).unwrap().is_empty());
        // No ORDER BY keys at all: the runs concatenate in worker order, which
        // is input order, and there is nothing to compare.
        let no_keys: Vec<SortKey> = vec![];
        let blocks = merge_runs(runs(&rows, &schema, &no_keys, 3), &no_keys, None, &mut g).unwrap();
        let got: Vec<i64> = blocks.iter().flat_map(|b| b.column(0).as_i64().unwrap().to_vec()).collect();
        assert_eq!(got, vec![3, 1, 2]);
    }

    #[test]
    fn merging_rebatches_at_block_size() {
        use crate::common::BLOCK_SIZE;
        let schema = ints_schema();
        let keys = vec![key(0, DataType::Int64, true, true)];
        let rows = ints(&(0..BLOCK_SIZE as i64 * 2 + 100).rev().collect::<Vec<_>>());
        let ctx = QueryContext::new();
        let mut g = MemGuard::new(&ctx, "the sort buffer");
        let blocks = merge_runs(runs(&rows, &schema, &keys, 5), &keys, None, &mut g).unwrap();
        let sizes: Vec<usize> = blocks.iter().map(|b| b.rows()).collect();
        assert_eq!(sizes, vec![BLOCK_SIZE, BLOCK_SIZE, 100]);
    }
}


