//! Hash join, covering `INNER`, `LEFT`, `RIGHT`, `FULL` and `CROSS`.
//!
//! Both sides are materialized, the **smaller** one becomes the build side,
//! and the larger probes it. Building on the smaller side is what bounds the
//! hash table by `min(|L|, |R|)` rather than by whichever relation the user
//! happened to write first, and it is free to decide here because both sides
//! are already in hand.
//!
//! ## One pipeline for five join types
//!
//! Rather than five code paths, the operator emits a stream of matched
//! `(left_row, right_row)` pairs and derives everything else from it:
//!
//! 1. **candidates** -- equi-matches from the hash index, or the full cross
//!    product when there is no `ON` clause;
//! 2. **residual** -- the non-equi remainder of the join condition, evaluated
//!    over a block of the candidate pairs and used to filter them. Crucially
//!    this happens *before* the matched sets are recorded, because a row whose
//!    only equi-match fails the residual is genuinely unmatched and an outer
//!    join must still NULL-pad it;
//! 3. **outer padding** -- unmatched left rows get `(row, NONE)`, unmatched
//!    right rows `(NONE, row)`, according to the join type.
//!
//! Output row order is probe-side order followed by the padding, and ties are
//! broken by build-side row order, so results are deterministic.
//!
//! ## Nothing is materialized that does not have to be
//!
//! The probe is a **cursor**, not a list. Each `next()` walks probe rows until
//! it has a block's worth of pairs, assembles it, and returns; the pair list
//! never exists in full. The padding phases work the same way, off two
//! matched-row bitsets (1 bit per input row) that accumulate across blocks --
//! which is why cutting a block mid-probe-row is safe: nothing is padded until
//! the probe has finished.
//!
//! What this replaces: a `Vec<(u32,u32)>` of every matched pair (8 B/pair), a
//! second `Vec<(Option<u32>,Option<u32>)>` of the same pairs plus the padding
//! (16 B/pair), one giant assembled output block, and then a full re-copy of
//! that block into batches. A 10M-row join result used to need ~240 MB of pair
//! lists and two copies of the output before the first row could be read; it
//! now needs one block of each.
//!
//! It also means a `LIMIT` above a join stops the probe. Measured over a 2M x
//! 100k inner join: the full `SELECT ts, name` costs 79.3ms, the same query
//! with `LIMIT 5` costs 11.4ms -- and the 11.4ms is entirely the two scans and
//! the index build, which no join can avoid.
//!
//! ## Gathering
//!
//! [`assemble`] decides *per block per side* whether that side is padded at
//! all. When it is not -- always true for an inner join, and true of the
//! matched blocks of an outer one -- the whole column is gathered with one
//! `Column::take`, a typed copy, instead of a `Value` round-trip per row
//! through a `ColumnBuilder`. Measured interleaved against forcing the builder
//! path, best-of-9: `SELECT ts, name` over a 2M x 100k join 113.9ms ->
//! 79.3ms (1.44x), `count()` over the same join 70.4ms -> 59.1ms (1.19x),
//! `LEFT JOIN` + `count()` 77.5ms -> 65.9ms (1.18x).
//!
//! ## NULL keys
//!
//! `NULL = NULL` is unknown, never true, so a key tuple containing a NULL
//! matches nothing. Such rows are skipped when building and when probing --
//! and because the matched sets start empty, they fall out naturally as
//! unmatched rows for an outer join, which is exactly right.
//!
//! ## Where the old row ceiling went
//!
//! `MAX_JOIN_ROWS` (10^8) used to reject a cross join before it allocated.
//! It is gone rather than re-derived: it was a guess about the *consumer's*
//! memory, not this operator's, and it was wrong in both directions -- it
//! fired at a point where the pair list alone was already 800 MB, and it
//! refused `... CROSS JOIN ... LIMIT 10`, which a streaming probe answers
//! instantly. What is left is honest: the build side and the index are charged
//! to [`super::MemTracker`], and a probe that really does run for an hour is
//! now interruptible by cancel or deadline, which is the actual fix.
//!
//! ## When the sides do not fit: the grace hash join
//!
//! Charging the build side to a budget bounds the damage but does not answer
//! the query, and "your fact table is bigger than 8 GiB" is not a plan. So a
//! side that will not fit is **partitioned by `hash(key)` to temp files**, both
//! sides on the same function, and the join is then run once per partition
//! *pair* by [`State::build`] -- the same constructor, the same probe, the same
//! padding, the same code. That reuse is the correctness argument:
//!
//! * equal keys hash equal (`hash_values` agrees with `Value`'s `Eq`, which is
//!   why `Date(5)` and `UInt(5)` still land in one partition), so **every match
//!   a row could have is inside its own partition**. A row unmatched in its
//!   partition is unmatched, full stop, and the outer padding a partition
//!   computes locally is the global answer. This is the step hand-rolled grace
//!   joins get wrong -- usually by padding a replicated side once per partition
//!   -- and it is why neither side is replicated here;
//! * the residual still runs *before* the matched sets are recorded, because it
//!   is literally the same [`State::next_block`] loop;
//! * a `NULL` in the key matches nothing, so those rows may go anywhere; they
//!   are dealt round-robin rather than to one bucket, which stops a
//!   mostly-`NULL` key from making partition 0 the whole side. They come back
//!   out as unmatched rows in whatever partition holds them, exactly as they do
//!   in memory.
//!
//! What changes is **output order**: rows come out partition by partition, not
//! in probe order. SQL does not order a join without `ORDER BY`, the hash
//! aggregate already reorders its groups when it spills, and the alternative --
//! buffering the whole result to re-sort it by probe row id -- is the very
//! materialization this operator exists to avoid.
//!
//! A partition pair that *still* will not fit is re-partitioned on fresh hash
//! bits (`level` seeds the mixer, so a second pass cannot reproduce the first
//! one's split) up to [`MAX_GRACE_LEVEL`]. Past that the input is one key with
//! a fan-out no split can reduce, and the honest outcome is the budget error.
//!
//! A **cross join never spills**: there is no key to partition on, so grace
//! does not apply. It also does not need to -- its output is the product, which
//! the streaming probe already hands out a block at a time -- but its two
//! materialized inputs are still bounded by the budget, and that is the one
//! shape left where a big input is an error rather than a slower query.
//!
//! The in-memory path pays a `usize` compare per block for the forced-spill
//! knob and nothing else: the switch is the `grow_to` this operator already
//! made, whose `Err` selects partitioning instead of returning. Measured with
//! the old and new `prepare` **alternating inside one process** (a temporary
//! switch, since removed) so both sides saw the same machine, best-of-11 and
//! median-of-11 per side, six rounds run with each side going first:
//! `SELECT count(*), sum(v)` over a 2M x 200k inner join, best-ratios 0.90,
//! 0.91, 1.11, 1.15, 1.01, 0.99 and median-ratios 1.20, 1.01, 1.08, 1.03,
//! 1.03, 0.97. No sign, and the same binary's own runs vary by 1.5x. Null.
//!
//! What spilling costs, on the same join, best-of-8 per configuration:
//!
//! ```text
//!   in memory                                    199.6 ms
//!   grace, one level  (32 partitions)            261.7 ms   1.31x
//!   grace, two levels                            458.4 ms   2.30x
//!   grace, three levels                          5 188 ms     26x
//! ```
//!
//! **One level is 1.3x**, measured five times at 1.14x to 1.38x -- and that is
//! remarkably cheap for a write and a read of both sides, worth knowing why:
//! the writes land in the page cache and never reach a platter inside one
//! query, and against that the partitioned join *wins back* much of what it
//! spends. 32 hash tables of 6 250 rows are 175 KiB each and live in L2, where
//! one table of 200 000 rows is 5.6 MiB and takes a cache miss per probe. That
//! is the radix-join effect, and here it nearly pays for the round trip.
//!
//! Each further level is another full read and write of everything, so the
//! curve is the one to keep in mind: a second level is reached when the input
//! is ~16x the budget, a third when it is ~500x. The 26x at three levels is
//! thousands of files' worth of syscalls and buffers; the two things that keep
//! it from being an order of magnitude worse are in the code, and both were
//! measured back to back on this join -- partitions buffer their rows before
//! framing them (see `pend_rows`, 1.5x at two levels) and a re-cut asks for the
//! number of buckets it actually needs (see [`Partitioner::new`], 2.7x at two
//! levels and 7.3x at three).
//!
//! Not done, and deliberately: when exactly one side fits, the classic hybrid
//! keeps that side resident and *streams* the other past it, writing nothing at
//! all. It is worth the whole 2.4x for a star schema. It needs the probe to
//! become a stream of blocks rather than one materialized block, and the
//! probe-side outer padding to move from "after everything" to "after each
//! block" -- a second probe implementation and a second outer-join argument, on
//! the path that currently gets both right. It belongs behind its own test, not
//! bolted onto this one.

use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::common::{BitSet, Error, Result, BLOCK_SIZE};
use crate::exec::expr;
use crate::planner::logical::BoundExpr;
use crate::sql::ast::JoinOp;
use crate::types::{Block, Column, ColumnBuilder, DataType, Schema, Value};

use super::sort::spill;
use super::{hash_values, MemGuard, Operator, QueryContext, ScanStats};

/// Row-id sentinel for "this side has no row", i.e. an outer-join pad. A block
/// can never hold `u32::MAX` rows, so no real id collides with it.
const NONE: u32 = u32::MAX;

pub struct Join<'a> {
    left: Box<dyn Operator + 'a>,
    right: Box<dyn Operator + 'a>,
    op: JoinOp,
    on: &'a [(usize, usize)],
    residual: Option<&'a BoundExpr>,
    schema: &'a Schema,
    ctx: &'a QueryContext,
    /// Built on the first `next()`, so `EXPLAIN` and the binder's throwaway
    /// pipelines never touch the inputs.
    state: Option<State>,
    /// Set instead of `state` when either side had to go to disk. `None` for
    /// every join that fits, which is the only case that costs anything.
    grace: Option<Grace>,
}

/// Which side is still being emitted.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Phase {
    Probe,
    PadLeft,
    PadRight,
    Done,
}

struct State {
    l: Block,
    r: Block,
    /// `None` for a cross join, which has no key to index.
    idx: Option<BuildIndex>,
    build_right: bool,
    bcols: Vec<usize>,
    pcols: Vec<usize>,
    phase: Phase,
    /// Probe-side cursor; also the padding cursor once the probe is done.
    p: usize,
    /// Inner cursor of the cross-product nested loop.
    q: usize,
    ml: BitSet,
    mr: BitSet,
    // Scratch reused for the life of the operator: the pair buffer, the key
    // probe tuple, and the gather index `assemble` hands to `Column::take`.
    pairs: Vec<(u32, u32)>,
    key: Vec<Value>,
    gather: Vec<u32>,
    guard: MemGuard,
    /// What the index alone costs, so the per-block charge can add the pair
    /// buffer to it rather than replacing it.
    idx_bytes: usize,
    /// Kept alive so the drained sides stay charged to the budget.
    _sides: [MemGuard; 2],
}

impl<'a> Join<'a> {
    pub fn new(
        left: Box<dyn Operator + 'a>,
        right: Box<dyn Operator + 'a>,
        op: JoinOp,
        on: &'a [(usize, usize)],
        residual: Option<&'a BoundExpr>,
        schema: &'a Schema,
        ctx: &'a QueryContext,
    ) -> Join<'a> {
        Join { left, right, op, on, residual, schema, ctx, state: None, grace: None }
    }

    /// Drain both sides, decide which one to index, and index it -- or, when
    /// either side will not fit, partition both to disk and set up [`Grace`].
    fn prepare(&mut self) -> Result<()> {
        // Separate guards: the drain charges a running total per side, and one
        // guard cannot hold two independent totals.
        let lg = MemGuard::new(self.ctx, "the join's left input");
        let rg = MemGuard::new(self.ctx, "the join's right input");
        // A cross join has no key to partition on; see the module docs.
        let can_spill = !self.on.is_empty();
        let forced = if can_spill { super::sort::forced_spill_rows() } else { 0 };
        let lcols: Vec<usize> = self.on.iter().map(|&(a, _)| a).collect();
        let rcols: Vec<usize> = self.on.iter().map(|&(_, b)| b).collect();
        let mut part: Option<Partitioner> = None;

        let l = drain_side(&mut self.left, self.ctx, lg, &lcols, 0, &mut part, forced, can_spill)?;
        let r = drain_side(&mut self.right, self.ctx, rg, &rcols, 1, &mut part, forced, can_spill)?;

        let part = match part {
            // One side may still be resident: the trigger fires on whichever
            // side hits the ceiling first, and the other was drained before
            // it. It goes to disk too, so the pair joins get the whole budget
            // rather than whatever is left over -- and so both sides are cut
            // on the same function, which is what the whole scheme rests on.
            Some(mut p) => {
                if let Some((b, g)) = l {
                    p.push(0, &b, &lcols, self.ctx)?;
                    drop((b, g));
                }
                if let Some((b, g)) = r {
                    p.push(1, &b, &rcols, self.ctx)?;
                    drop((b, g));
                }
                p
            }
            None => {
                let (l, lg) = l.expect("no partitioner means both sides are in hand");
                let (r, rg) = r.expect("no partitioner means both sides are in hand");
                // The index is charged *on top of* both sides, so two sides
                // that each fit can still be refused here -- which is where
                // the first version of this discovered it had not spilled at
                // all. Probed with a reservation that is released immediately,
                // because both sides are already in hand: falling back to
                // grace now costs one pass over memory, not a re-read.
                let mut probe = MemGuard::new(self.ctx, "the join hash index");
                let room = probe.grow_to(index_cost(l.rows().min(r.rows()))).is_ok();
                drop(probe);
                if room || !can_spill {
                    self.state =
                        Some(State::build(l, r, self.on, self.op, self.ctx, [lg, rg])?);
                    return Ok(());
                }
                let mut p = Partitioner::new(self.ctx, 0, l.rows().max(r.rows()), usize::MAX)?;
                p.push(0, &l, &lcols, self.ctx)?;
                p.push(1, &r, &rcols, self.ctx)?;
                drop((l, lg, r, rg));
                p
            }
        };
        let fallback = [self.left.schema().clone(), self.right.schema().clone()];
        let (pending, schemas) = part.finish(fallback)?;
        self.grace = Some(Grace { pending, cur: None, schemas, cols: [lcols, rcols], forced });
        Ok(())
    }
}

/// Drain one side into memory, or hand it (and everything after it) to `part`.
///
/// Returns `None` once anything has been partitioned; the guard is dropped on
/// that path rather than returned, which is what gives the reservation back
/// before the pair joins ask for it.
#[allow(clippy::too_many_arguments)]
fn drain_side(
    op: &mut Box<dyn Operator + '_>,
    ctx: &QueryContext,
    mut guard: MemGuard,
    cols: &[usize],
    side: usize,
    part: &mut Option<Partitioner>,
    forced: usize,
    can_spill: bool,
) -> Result<Option<(Block, MemGuard)>> {
    let mut acc: Option<Block> = None;
    loop {
        ctx.check()?;
        let Some(b) = op.next()? else { break };
        if b.rows() == 0 {
            continue;
        }
        // The other side already tipped over, so this one goes straight to
        // disk: there is no point buffering rows that would only be spilled.
        if let Some(p) = part.as_mut() {
            p.push(side, &b, cols, ctx)?;
            continue;
        }
        match &mut acc {
            None => acc = Some(b),
            Some(a) => a.extend(&b)?,
        }
        let a = acc.as_ref().expect("just filled");
        // Two tests, one branch each per block. `forced` is 0 unless the test
        // knob is set, and `grow_to` is the charge this loop already made --
        // its `Err` is the signal, not an error, because a side that does not
        // fit is a slower query rather than a refused one.
        let over = (forced != 0 && a.rows() >= forced) || guard.grow_to(a.bytes()).is_err();
        if over && can_spill {
            let p = part.insert(Partitioner::new(ctx, 0, a.rows(), usize::MAX)?);
            p.push(side, a, cols, ctx)?;
            acc = None;
            drop(guard);
            guard = MemGuard::new(ctx, "the join's spill buffer");
        } else if over {
            // A cross join cannot partition, so the charge stands and its
            // error is the answer.
            guard.grow_to(a.bytes())?;
        }
    }
    match acc {
        Some(a) => Ok(Some((a, guard))),
        // Nothing at all arrived, and nothing was spilled: an empty side is
        // still a side, so hand back the empty block the old drain would have.
        None if part.is_none() => Ok(Some((Block::empty(op.schema()), guard))),
        None => Ok(None),
    }
}

fn padding_wanted(op: JoinOp) -> (bool, bool) {
    match op {
        JoinOp::Left => (true, false),
        JoinOp::Right => (false, true),
        JoinOp::Full => (true, true),
        JoinOp::Inner | JoinOp::Cross => (false, false),
    }
}

impl Operator for Join<'_> {
    fn schema(&self) -> &Schema {
        self.schema
    }

    fn stats(&self) -> ScanStats {
        let mut s = self.left.stats();
        s.merge(&self.right.stats());
        s
    }

    fn next(&mut self) -> Result<Option<Block>> {
        if self.state.is_none() && self.grace.is_none() {
            self.prepare()?;
        }
        // Copy the shared references out first: they are `Copy`, so this ends
        // the borrow of `self` before `state` is borrowed mutably.
        let (ctx, op, residual, schema) = (self.ctx, self.op, self.residual, self.schema);
        if let Some(st) = self.state.as_mut() {
            return st.next_block(ctx, op, residual, schema);
        }
        let on = self.on;
        self.grace
            .as_mut()
            .expect("prepare sets exactly one of the two")
            .next_block(ctx, op, residual, schema, on)
    }
}

impl State {
    /// Index one side of a materialized pair and get ready to probe.
    ///
    /// The **only** way a `State` is made: the in-memory join builds one, and
    /// the grace driver builds one per partition pair. Build-side choice,
    /// output ordering, NULL-key handling and outer padding are therefore the
    /// same code in both, rather than two implementations that have to be kept
    /// in step by review.
    fn build(
        l: Block,
        r: Block,
        on: &[(usize, usize)],
        op: JoinOp,
        ctx: &QueryContext,
        sides: [MemGuard; 2],
    ) -> Result<State> {
        // Ties build on the right so the probe (and therefore the output)
        // stays in left-row order, which is what a reader expects to see.
        let build_right = r.rows() <= l.rows();
        let (bcols, pcols): (Vec<usize>, Vec<usize>) = if build_right {
            (on.iter().map(|&(_, b)| b).collect(), on.iter().map(|&(a, _)| a).collect())
        } else {
            (on.iter().map(|&(a, _)| a).collect(), on.iter().map(|&(_, b)| b).collect())
        };
        let (build, probe) = if build_right { (&r, &l) } else { (&l, &r) };
        check_cols(build, &bcols)?;
        check_cols(probe, &pcols)?;

        let mut guard = MemGuard::new(ctx, "the join hash index");
        let mut key = Vec::with_capacity(bcols.len());
        let mut idx_bytes = 0;
        let idx = if on.is_empty() {
            None
        } else {
            let ix = BuildIndex::build(build, &bcols, &mut key);
            idx_bytes = ix.bytes();
            guard.grow_to(idx_bytes)?;
            Some(ix)
        };

        let (want_left, want_right) = padding_wanted(op);
        Ok(State {
            ml: if want_left { BitSet::with_capacity_bits(l.rows()) } else { BitSet::new() },
            mr: if want_right { BitSet::with_capacity_bits(r.rows()) } else { BitSet::new() },
            l,
            r,
            idx,
            build_right,
            bcols,
            pcols,
            phase: Phase::Probe,
            p: 0,
            q: 0,
            pairs: Vec::with_capacity(BLOCK_SIZE),
            key,
            gather: Vec::with_capacity(BLOCK_SIZE),
            guard,
            idx_bytes,
            _sides: sides,
        })
    }

    fn next_block(
        &mut self,
        ctx: &QueryContext,
        op: JoinOp,
        residual: Option<&BoundExpr>,
        schema: &Schema,
    ) -> Result<Option<Block>> {
        let (want_left, want_right) = padding_wanted(op);
        loop {
            ctx.check()?;
            self.pairs.clear();
            match self.phase {
                Phase::Probe => {
                    if self.idx.is_some() {
                        self.probe_hash();
                    } else {
                        self.probe_cross();
                    }
                    if self.pairs.is_empty() {
                        self.p = 0;
                        self.phase = next_phase(want_left, want_right);
                        continue;
                    }
                    // The residual runs before "matched" is recorded: an
                    // equi-match it rejects leaves both rows unmatched.
                    if let Some(res) = residual {
                        let cand = assemble(&self.l, &self.r, &self.pairs, schema, &mut self.gather)?;
                        let sel = expr::eval_predicate(res, &cand)?;
                        if sel.len() < self.pairs.len() {
                            for (w, &i) in sel.iter().enumerate() {
                                self.pairs[w] = self.pairs[i as usize];
                            }
                            self.pairs.truncate(sel.len());
                        }
                        if self.pairs.is_empty() {
                            continue;
                        }
                    }
                    // Bitsets, not a pair list: padding is decided only after
                    // the whole probe, so marks may accumulate across blocks.
                    if want_left {
                        for &(a, _) in &self.pairs {
                            self.ml.set(a as usize);
                        }
                    }
                    if want_right {
                        for &(_, b) in &self.pairs {
                            self.mr.set(b as usize);
                        }
                    }
                    // The pair buffer is the one thing here that a single
                    // popular key can blow up: its whole chain is emitted
                    // before the block-size check, so `pairs` can reach the
                    // build side's fan-out for one key.
                    self.guard
                        .grow_to(self.idx_bytes + self.pairs.capacity() * size_of::<(u32, u32)>())?;
                }
                Phase::PadLeft => {
                    while self.p < self.l.rows() && self.pairs.len() < BLOCK_SIZE {
                        if !self.ml.get(self.p) {
                            self.pairs.push((self.p as u32, NONE));
                        }
                        self.p += 1;
                    }
                    if self.pairs.is_empty() {
                        self.p = 0;
                        self.phase = if want_right { Phase::PadRight } else { Phase::Done };
                        continue;
                    }
                }
                Phase::PadRight => {
                    while self.p < self.r.rows() && self.pairs.len() < BLOCK_SIZE {
                        if !self.mr.get(self.p) {
                            self.pairs.push((NONE, self.p as u32));
                        }
                        self.p += 1;
                    }
                    if self.pairs.is_empty() {
                        self.phase = Phase::Done;
                        continue;
                    }
                }
                Phase::Done => return Ok(None),
            }
            return assemble(&self.l, &self.r, &self.pairs, schema, &mut self.gather).map(Some);
        }
    }

    /// Walk probe rows until the pair buffer is full.
    ///
    /// A probe row's whole chain is emitted before the buffer is checked, so
    /// one very popular key can overshoot `BLOCK_SIZE`; splitting it would be
    /// correct too (the matched bitsets are cumulative), but not splitting
    /// keeps the fan-out of one row contiguous in the output.
    fn probe_hash(&mut self) {
        let State { l, r, idx, build_right, pcols, bcols, pairs, key, p, .. } = self;
        let ix = idx.as_ref().unwrap();
        let (build, probe) = if *build_right { (&*r, &*l) } else { (&*l, &*r) };
        while *p < probe.rows() && pairs.len() < BLOCK_SIZE {
            let row = *p;
            *p += 1;
            if !fill_key(probe, pcols, row, key) {
                continue; // a NULL in the key can never equal anything
            }
            let h = hash_values(key);
            let mut cur = ix.head(h, build, bcols, key);
            while cur != NONE {
                pairs.push(if *build_right {
                    (row as u32, cur)
                } else {
                    (cur, row as u32)
                });
                cur = ix.next[cur as usize];
            }
        }
    }

    /// The `ON`-less nested loop, as a resumable cursor rather than a
    /// materialized product.
    fn probe_cross(&mut self) {
        let (nl, nr) = (self.l.rows(), self.r.rows());
        while self.p < nl && self.pairs.len() < BLOCK_SIZE {
            while self.q < nr && self.pairs.len() < BLOCK_SIZE {
                self.pairs.push((self.p as u32, self.q as u32));
                self.q += 1;
            }
            if self.q < nr {
                return;
            }
            self.q = 0;
            self.p += 1;
        }
    }
}

// ============================================================ grace hash join

/// How many times a partition pair may be cut again before the operator admits
/// that splitting is not what is wrong.
///
/// At 32 ways a level, four levels is a million partitions. A pair that still
/// does not fit past that is one key whose fan-out no hash can divide, and
/// re-reading it a fifth time only delays the budget error it is going to get.
const MAX_GRACE_LEVEL: u32 = 4;

/// Upper bound on what indexing `n` build rows and buffering one block of
/// pairs costs.
///
/// [`BuildIndex`] holds `(2n).next_power_of_two()` slot `u32`s -- under `4n`
/// -- plus `n` chain `u32`s and `n` `u64` hashes: 28 B/row at the worst load
/// factor, rounded up to 32 with a constant for the tiny-`n` floor. It has to
/// be an *upper* bound, because it is what decides whether a partition is
/// joined or cut again, and an underestimate turns a spill back into the
/// budget error it exists to replace.
#[inline]
fn index_cost(build_rows: usize) -> usize {
    build_rows * 32 + BLOCK_SIZE * size_of::<(u32, u32)>() + 512
}

/// One side's partition files, still open.
struct SideFiles {
    writers: Vec<Option<spill::RunWriter>>,
    /// Rows destined for a partition but not yet framed; see [`Partitioner`].
    pend: Vec<Option<Block>>,
    /// In-memory footprint of what was written, per partition. Exact rather
    /// than estimated: it is `Block::bytes` of the very blocks that went out,
    /// which is what they will cost again on the way back in.
    bytes: Vec<usize>,
    rows: Vec<usize>,
    /// Taken from the first block written, and the schema every block of this
    /// side is read back against.
    schema: Option<Schema>,
}

impl SideFiles {
    fn new(n: usize) -> SideFiles {
        SideFiles {
            writers: (0..n).map(|_| None).collect(),
            pend: (0..n).map(|_| None).collect(),
            bytes: vec![0; n],
            rows: vec![0; n],
            schema: None,
        }
    }
}

/// Cuts both sides of one join on the same hash function.
///
/// One directory and one partition count for the pair -- two `Partitions` from
/// the hash aggregate would arm themselves independently and could disagree
/// about the number of buckets, which is the one thing that would silently
/// separate a key from its matches.
struct Partitioner {
    dir: spill::SpillDir,
    sides: [SideFiles; 2],
    mask: u64,
    /// Recursion depth, and the mixer seed: a second cut on the same bits
    /// would reproduce the first one and never terminate.
    level: u32,
    flush_at: usize,
    /// Rows a partition accumulates before it is framed into its file.
    ///
    /// Without it a partition's frames are `block_rows / n` long -- 256 rows at
    /// a 32-way cut -- and every re-cut divides that again, so a deep grace
    /// join ends up writing and parsing **eight-row frames**. Measured back to
    /// back (temporary env switch, since removed) on the 2M x 200k join in the
    /// module docs, forced to two levels: 2733 ms unbuffered against 1793 ms
    /// buffered and 1939 against 1374 in a second round -- 1.4-1.5x, all of it
    /// per-frame overhead.
    ///
    /// At *three* levels it is worth nothing (28.3 s either way), because there
    /// a partition holds fewer rows than a frame wants anyway and the file
    /// count is what hurts. That is the other half of the fix, and it lives in
    /// [`Partitioner::new`].
    pend_rows: usize,
    // Scratch reused for the life of the partitioner: the key tuple, the
    // per-row partition assignment, and the counting sort that groups rows by
    // partition. A `Vec` per bucket per block would allocate once per bucket
    // per 8192 rows.
    key: Vec<Value>,
    parts: Vec<u32>,
    counts: Vec<u32>,
    cursor: Vec<u32>,
    sel: Vec<u32>,
    /// Round-robin cursor for NULL keys; see the module docs.
    rr: u32,
}

/// One partition of the left side and its counterpart on the right.
struct Pair {
    l: Option<PathBuf>,
    r: Option<PathBuf>,
    lbytes: usize,
    rbytes: usize,
    lrows: usize,
    rrows: usize,
    level: u32,
    /// Shared: one directory holds every partition cut at one level and has to
    /// outlive the last of them, however early a `LIMIT` stops reading.
    _dir: Arc<spill::SpillDir>,
}

impl Pair {
    /// Unlink both files as soon as their rows are in hand, so peak disk
    /// tracks what is still owed rather than everything ever written.
    fn unlink(&self) {
        for p in [&self.l, &self.r].into_iter().flatten() {
            let _ = std::fs::remove_file(p);
        }
    }
}

impl Partitioner {
    /// `want_ways` is how many buckets the caller believes this cut needs; a
    /// re-cut knows, because it can see how far over the ceiling the pair is,
    /// and the first cut does not, so it asks for the maximum. Sizing it
    /// matters at depth: a pair only twice too big split 32 ways produces 32
    /// files holding a handful of rows each. Measured back to back on the join
    /// in the module docs, against the same build always asking for 32:
    /// **two levels 1793 -> 670 ms (2.7x), three levels 28.5 -> 3.9 s (7.3x)**,
    /// the third level dropping from 32 768 partitions to 2 048.
    fn new(
        ctx: &QueryContext,
        level: u32,
        hint_rows: usize,
        want_ways: usize,
    ) -> Result<Partitioner> {
        // A quarter of the budget for write-behind, split over *two* open
        // writers per partition rather than the aggregate's one, so the same
        // ceiling buys half as many buckets.
        //
        // The bucket count is what the write buffers can hold at their 4 KiB
        // floor, not a fraction of the budget: a tight budget wants *more*
        // buckets, because each one has to fit in what is left, and taking the
        // aggregate's "one bucket per 32 KiB of cap" here would hand a 256 KiB
        // budget two buckets and then re-cut them four times. Never more
        // buckets than there are rows to put in them -- a hundred one-row
        // files cost more in syscalls than they save in passes.
        const MIN_FLUSH: usize = 4 << 10;
        let cap = ((ctx.mem.limit().max(0) as usize) / 4).max(8 << 10);
        let want = (cap / (2 * MIN_FLUSH))
            .min(want_ways)
            .clamp(2, 32)
            .next_power_of_two()
            .min(32);
        let n = want.min(hint_rows.max(2).next_power_of_two());
        Ok(Partitioner {
            dir: spill::SpillDir::new()?,
            sides: [SideFiles::new(n), SideFiles::new(n)],
            mask: n as u64 - 1,
            level,
            flush_at: (cap / (2 * n)).clamp(MIN_FLUSH, 64 << 10),
            pend_rows: 0,
            key: Vec::new(),
            parts: Vec::new(),
            counts: Vec::new(),
            cursor: Vec::new(),
            sel: Vec::new(),
            rr: 0,
        })
    }

    /// Which partition a key with hash `h` belongs to.
    ///
    /// Re-mixed with the level rather than shifted by it: shifting runs out of
    /// hash bits after `64/log2(n)` levels and then stops partitioning at all,
    /// which turns a deep recursion into an infinite one.
    #[inline]
    fn part_of(&self, h: u64) -> u32 {
        let seed = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(self.level as u64 + 1);
        (crate::common::mum(h ^ seed, 0xD6E8_FEB8_6659_FD93) & self.mask) as u32
    }

    /// Cut one block of one side into partition files.
    fn push(&mut self, side: usize, b: &Block, cols: &[usize], ctx: &QueryContext) -> Result<()> {
        if b.rows() == 0 {
            return Ok(());
        }
        // A cancelled query that keeps writing gigabytes to disk is worse than
        // one that keeps burning CPU.
        ctx.check()?;
        // The in-memory path checks this when it picks a build side; the
        // spilling one has to check before it indexes a column that is not
        // there, and the message has to be the same one.
        check_cols(b, cols)?;
        let n = self.sides[side].bytes.len();

        self.parts.clear();
        self.parts.reserve(b.rows());
        for row in 0..b.rows() {
            let p = if fill_key(b, cols, row, &mut self.key) {
                self.part_of(hash_values(&self.key))
            } else {
                // NULL matches nothing, so any bucket is correct; dealing them
                // round-robin stops a mostly-NULL key from putting one side's
                // whole relation in bucket 0.
                self.rr = (self.rr + 1) & self.mask as u32;
                self.rr
            };
            self.parts.push(p);
        }

        // Counting sort into one flat buffer: rows of a bucket end up
        // contiguous in `sel`, in input order, so one `take` per bucket per
        // block is the whole gather.
        self.counts.clear();
        self.counts.resize(n + 1, 0);
        for &p in &self.parts {
            self.counts[p as usize + 1] += 1;
        }
        for i in 0..n {
            self.counts[i + 1] += self.counts[i];
        }
        self.sel.clear();
        self.sel.resize(self.parts.len(), 0);
        self.cursor.clear();
        self.cursor.extend_from_slice(&self.counts[..n]);
        for (r, &p) in self.parts.iter().enumerate() {
            let c = &mut self.cursor[p as usize];
            self.sel[*c as usize] = r as u32;
            *c += 1;
        }

        if self.sides[side].schema.is_none() {
            self.sides[side].schema = Some(spill::schema_of(b));
        }
        if self.pend_rows == 0 {
            // The whole pending set is `2 * n * pend_rows * row_bytes`, and it
            // is uncharged like the file buffers, so it is sized against the
            // *smaller* of a quarter of the budget and a flat 8 MiB -- a
            // quarter of the default 8 GiB budget would be two gigabytes of
            // write-behind for a join that had just said it was short of
            // memory. Floored at 256 rows because the point is to stop frames
            // from getting small, and capped at a block because past that the
            // reader gains nothing.
            let row_bytes = (b.bytes() / b.rows()).max(1);
            let share = ((ctx.mem.limit().max(0) as usize) / 4).clamp(8 << 10, 8 << 20);
            self.pend_rows = (share / (2 * n * row_bytes)).clamp(256, BLOCK_SIZE);
        }
        for p in 0..n {
            let (lo, hi) = (self.counts[p] as usize, self.counts[p + 1] as usize);
            if lo == hi {
                continue;
            }
            let sub = b.take(&self.sel[lo..hi]);
            self.sides[side].bytes[p] += sub.bytes();
            self.sides[side].rows[p] += sub.rows();
            // Taken out and put back rather than borrowed in place: `writer`
            // needs `&mut self` and the pending block lives inside `self`.
            let merged = match self.sides[side].pend[p].take() {
                None => sub,
                Some(mut a) => {
                    a.extend(&sub)?;
                    a
                }
            };
            if merged.rows() >= self.pend_rows {
                self.writer(side, p)?.push(&merged)?;
            } else {
                self.sides[side].pend[p] = Some(merged);
            }
        }
        Ok(())
    }

    /// The partition's writer, created on first use so a join whose overflow
    /// all lands in three buckets does not open sixty-four files.
    fn writer(&mut self, side: usize, p: usize) -> Result<&mut spill::RunWriter> {
        if self.sides[side].writers[p].is_none() {
            let w = self.dir.create_buffered(self.flush_at)?;
            self.sides[side].writers[p] = Some(w);
        }
        Ok(self.sides[side].writers[p].as_mut().expect("just created"))
    }

    /// Close every file and pair the partitions up.
    ///
    /// `fallback` supplies a side's shape when it wrote nothing at all, so an
    /// empty partition still assembles at the right width -- a side that
    /// silently lost its columns would drop them from every output row.
    fn finish(mut self, fallback: [Schema; 2]) -> Result<(Vec<Pair>, [Schema; 2])> {
        // Whatever never reached `pend_rows` still has to reach its file.
        for side in 0..2 {
            for p in 0..self.sides[side].pend.len() {
                if let Some(b) = self.sides[side].pend[p].take() {
                    self.writer(side, p)?.push(&b)?;
                }
            }
        }
        let Partitioner { dir, sides, level, .. } = self;
        let [ls, rs] = sides;
        let n = ls.bytes.len();
        let schemas = [
            ls.schema.clone().unwrap_or_else(|| fallback[0].clone()),
            rs.schema.clone().unwrap_or_else(|| fallback[1].clone()),
        ];
        let close = |ws: Vec<Option<spill::RunWriter>>| -> Result<Vec<Option<PathBuf>>> {
            ws.into_iter().map(|w| w.map(|w| w.finish()).transpose()).collect()
        };
        let (lp, rp) = (close(ls.writers)?, close(rs.writers)?);
        let dir = Arc::new(dir);
        let mut out = Vec::with_capacity(n);
        for (p, (l, r)) in lp.into_iter().zip(rp).enumerate() {
            out.push(Pair {
                l,
                r,
                lbytes: ls.bytes[p],
                rbytes: rs.bytes[p],
                lrows: ls.rows[p],
                rrows: rs.rows[p],
                level: level + 1,
                _dir: dir.clone(),
            });
        }
        Ok((out, schemas))
    }
}

/// Joins the partition pairs one at a time, re-cutting the ones that still do
/// not fit.
struct Grace {
    /// A stack, so a re-cut pair's children are consumed before its siblings
    /// and only one level's files are ever live at once.
    pending: Vec<Pair>,
    /// The pair being joined. Dropped -- and its budget released -- before the
    /// next one is read.
    cur: Option<State>,
    schemas: [Schema; 2],
    /// Join key columns per side, in `on` order, for the re-cut pass.
    cols: [Vec<usize>; 2],
    forced: usize,
}

impl Grace {
    fn next_block(
        &mut self,
        ctx: &QueryContext,
        op: JoinOp,
        residual: Option<&BoundExpr>,
        schema: &Schema,
        on: &[(usize, usize)],
    ) -> Result<Option<Block>> {
        loop {
            if let Some(st) = self.cur.as_mut() {
                match st.next_block(ctx, op, residual, schema)? {
                    Some(b) => return Ok(Some(b)),
                    None => self.cur = None,
                }
            }
            if !self.advance(ctx, op, on)? {
                return Ok(None);
            }
        }
    }

    /// Load the next pair worth joining, cutting it again if it is too big.
    fn advance(&mut self, ctx: &QueryContext, op: JoinOp, on: &[(usize, usize)]) -> Result<bool> {
        let (want_left, want_right) = padding_wanted(op);
        loop {
            let Some(p) = self.pending.pop() else { return Ok(false) };
            ctx.check()?;
            // A partition where one side is empty emits only that side's
            // padding, and often nothing at all -- an inner join over it is
            // provably empty. Skipping is not an optimization: an outer join
            // whose *whole* right side is empty has one such pair per bucket,
            // and reading them all back to produce nothing would be the
            // dominant cost of the query.
            let emits = match (p.lrows, p.rrows) {
                (0, 0) => false,
                (0, _) => want_right,
                (_, 0) => want_left,
                _ => true,
            };
            if !emits {
                p.unlink();
                continue;
            }
            if p.level < MAX_GRACE_LEVEL && !self.fits(&p, ctx) {
                self.split(p, ctx)?;
                continue;
            }
            let mut lg = MemGuard::new(ctx, "the join's left input");
            let mut rg = MemGuard::new(ctx, "the join's right input");
            let l = read_partition(p.l.as_deref(), &self.schemas[0], &mut lg)?;
            let r = read_partition(p.r.as_deref(), &self.schemas[1], &mut rg)?;
            p.unlink();
            self.cur = Some(State::build(l, r, on, op, ctx, [lg, rg])?);
            return Ok(true);
        }
    }

    /// Will joining this pair stay inside the budget?
    ///
    /// The two sides' footprints are exact -- they are `Block::bytes` of the
    /// very blocks that went out -- and [`index_cost`] bounds the rest. Half
    /// the ceiling rather than all of it, because the rest of the query is
    /// entitled to the other half.
    fn fits(&self, p: &Pair, ctx: &QueryContext) -> bool {
        if self.forced != 0 {
            return p.lrows.max(p.rrows) <= self.forced;
        }
        let need = p.lbytes + p.rbytes + index_cost(p.lrows.min(p.rrows));
        (need as i64).saturating_mul(2) <= ctx.mem.limit()
    }

    /// Re-cut one pair on fresh hash bits, streaming: the whole point is that
    /// the pair never has to be in memory at once, so it is read a block at a
    /// time straight into the next level's files.
    fn split(&mut self, p: Pair, ctx: &QueryContext) -> Result<()> {
        // How many times over the ceiling this pair is, which is how many
        // buckets it needs -- not the maximum, which would shred a pair that
        // is barely too big into files of a few rows each.
        let ways = if self.forced != 0 {
            p.lrows.max(p.rrows).div_ceil(self.forced.max(1))
        } else {
            let need = p.lbytes + p.rbytes + index_cost(p.lrows.min(p.rrows));
            let target = ((ctx.mem.limit().max(1) / 2) as usize).max(1);
            need.div_ceil(target)
        };
        let mut np = Partitioner::new(ctx, p.level, p.lrows.max(p.rrows), ways)?;
        for (side, path) in [&p.l, &p.r].into_iter().enumerate() {
            let Some(path) = path else { continue };
            let mut rd = spill::RunReader::open(path, self.schemas[side].clone())?;
            while let Some(b) = rd.next()? {
                np.push(side, &b, &self.cols[side], ctx)?;
            }
        }
        p.unlink();
        let (subs, _) = np.finish([self.schemas[0].clone(), self.schemas[1].clone()])?;
        self.pending.extend(subs);
        Ok(())
    }
}

/// Read one partition file back as a single block, charged as it grows.
fn read_partition(path: Option<&Path>, schema: &Schema, guard: &mut MemGuard) -> Result<Block> {
    let Some(path) = path else { return Ok(Block::empty(schema)) };
    let mut rd = spill::RunReader::open(path, schema.clone())?;
    let mut acc: Option<Block> = None;
    while let Some(b) = rd.next()? {
        match &mut acc {
            None => acc = Some(b),
            Some(a) => a.extend(&b)?,
        }
        guard.grow_to(acc.as_ref().map_or(0, |a| a.bytes()))?;
    }
    Ok(acc.unwrap_or_else(|| Block::empty(schema)))
}

fn next_phase(want_left: bool, want_right: bool) -> Phase {
    if want_left {
        Phase::PadLeft
    } else if want_right {
        Phase::PadRight
    } else {
        Phase::Done
    }
}

/// Open-addressed index over the build side's rows.
///
/// There is no key arena: the keys are already sitting in the build block, so
/// a slot only remembers a row id, and equal keys chain through `next`.
/// Probing compares the caller's borrowed scratch tuple against the block's
/// columns, so neither building nor probing allocates. The `FastMap<GroupKey,
/// Vec<u32>>` this replaces built a `Vec<Value>` per build row *and* per probe
/// row, plus a `Vec<u32>` per distinct key -- three allocation sources in the
/// hottest loop of the operator.
struct BuildIndex {
    /// `head + 1`, 0 for empty.
    slots: Vec<u32>,
    /// Next build row with an equal key, or [`NONE`].
    next: Vec<u32>,
    hashes: Vec<u64>,
}

impl BuildIndex {
    fn build(b: &Block, cols: &[usize], key: &mut Vec<Value>) -> BuildIndex {
        let n = b.rows();
        // Load factor under 1/2, as everywhere else in the engine: past that
        // linear probing degrades sharply and the memory is cheaper.
        let cap = (n * 2).max(64).next_power_of_two();
        let mask = cap - 1;
        let mut ix = BuildIndex {
            slots: vec![0; cap],
            next: vec![NONE; n],
            hashes: vec![0; n],
        };
        // Insert in reverse row order and prepend, so each chain comes out
        // ascending: output ties break by build-side row, and that is the only
        // way to get it without an O(chain) walk per insert.
        for i in (0..n).rev() {
            if !fill_key(b, cols, i, key) {
                continue;
            }
            let h = hash_values(key);
            ix.hashes[i] = h;
            let mut s = h as usize & mask;
            loop {
                let head = ix.slots[s];
                if head == 0 {
                    ix.slots[s] = i as u32 + 1;
                    break;
                }
                let j = head as usize - 1;
                if ix.hashes[j] == h && keys_equal(b, cols, j, key) {
                    ix.next[i] = j as u32;
                    ix.slots[s] = i as u32 + 1;
                    break;
                }
                s = (s + 1) & mask;
            }
        }
        ix
    }

    /// First build row whose key equals `key`, or [`NONE`].
    #[inline]
    fn head(&self, h: u64, b: &Block, cols: &[usize], key: &[Value]) -> u32 {
        let mask = self.slots.len() - 1;
        let mut s = h as usize & mask;
        loop {
            let head = self.slots[s];
            if head == 0 {
                return NONE;
            }
            let j = head as usize - 1;
            // Cached hash first: a mismatch rules the chain out without
            // decoding a single `Value`.
            if self.hashes[j] == h && keys_equal(b, cols, j, key) {
                return j as u32;
            }
            s = (s + 1) & mask;
        }
    }

    fn bytes(&self) -> usize {
        (self.slots.capacity() + self.next.capacity()) * size_of::<u32>()
            + self.hashes.capacity() * size_of::<u64>()
    }
}

fn check_cols(b: &Block, cols: &[usize]) -> Result<()> {
    for &c in cols {
        if c >= b.width() {
            return Err(Error::exec(format!(
                "join key column #{c} is out of range for a {}-column input",
                b.width()
            )));
        }
    }
    Ok(())
}

/// Fill `out` with a row's key tuple; `false` when any component is NULL.
#[inline]
fn fill_key(b: &Block, cols: &[usize], row: usize, out: &mut Vec<Value>) -> bool {
    out.clear();
    for &c in cols {
        let col = b.column(c);
        if col.is_null(row) {
            return false;
        }
        out.push(col.value(row));
    }
    true
}

#[inline]
fn keys_equal(b: &Block, cols: &[usize], row: usize, key: &[Value]) -> bool {
    cols.iter()
        .zip(key)
        .all(|(&c, k)| b.column(c).value(row) == *k)
}

/// Gather one output block: left columns then right columns, NULL-padding
/// wherever a side has no row.
///
/// Column types come from the join's own schema when it agrees physically with
/// the source, so an outer join labelled `Nullable(Int64)` in the plan stays
/// labelled that way. Where padding actually introduced a NULL the type is
/// widened regardless, because a column with a live mask must never claim to
/// be non-nullable.
fn assemble(
    l: &Block,
    r: &Block,
    pairs: &[(u32, u32)],
    schema: &Schema,
    gather: &mut Vec<u32>,
) -> Result<Block> {
    let mut cols: Vec<Column> = Vec::with_capacity(l.width() + r.width());
    for side in [false, true] {
        let (b, off) = if side { (r, l.width()) } else { (l, 0) };
        if b.width() == 0 {
            continue;
        }
        // Whether *this* side is padded is a property of the block, not of the
        // column, so it is decided once and the index vector is built once for
        // every column of the side.
        let padded = pairs.iter().any(|p| pick(p, side) == NONE);
        if !padded {
            gather.clear();
            gather.extend(pairs.iter().map(|p| pick(p, side)));
        }
        for (i, c) in b.columns.iter().enumerate() {
            let ty = want_ty(schema, off + i, c);
            cols.push(if padded {
                pad_take(c, pairs, side, ty)?
            } else {
                // The typed gather: one `Column::take` per column instead of a
                // `Value` round-trip per row through a builder.
                let mut out = c.take(gather);
                out.ty = if out.has_nulls() { ty.to_nullable() } else { ty };
                out
            });
        }
    }
    if cols.is_empty() {
        return Ok(Block::rows_only(pairs.len()));
    }
    Block::new(cols)
}

#[inline]
fn pick(p: &(u32, u32), right: bool) -> u32 {
    if right {
        p.1
    } else {
        p.0
    }
}

fn want_ty(schema: &Schema, i: usize, c: &Column) -> DataType {
    match schema.fields().get(i) {
        Some(f) if f.ty.physical() == c.ty.physical() => f.ty.clone(),
        _ => c.ty.clone(),
    }
}

/// The padding path: some rows on this side have no source row at all, so the
/// column has to be built value by value.
fn pad_take(c: &Column, pairs: &[(u32, u32)], right: bool, ty: DataType) -> Result<Column> {
    let mut b = ColumnBuilder::with_capacity(ty.to_nullable(), pairs.len());
    for p in pairs {
        match pick(p, right) {
            NONE => b.push_null(),
            r if c.is_null(r as usize) => b.push_null(),
            r => b.push_value(&c.value(r as usize))?,
        }
    }
    let mut out = b.finish();
    if !out.has_nulls() {
        out.ty = ty;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::Values;
    use crate::types::{Field, Value};

    fn s2(a: &str, b: &str) -> Schema {
        Schema::new(vec![
            Field::new(a, DataType::Int64),
            Field::new(b, DataType::Int64),
        ])
        .unwrap()
    }

    fn rows(pairs: &[(i64, i64)]) -> Vec<Vec<Value>> {
        pairs
            .iter()
            .map(|&(a, b)| vec![Value::Int(a), Value::Int(b)])
            .collect()
    }

    fn joined(
        lr: &[(i64, i64)],
        rr: &[(i64, i64)],
        op: JoinOp,
        on: &[(usize, usize)],
        residual: Option<&BoundExpr>,
    ) -> Vec<Vec<Value>> {
        let ls = s2("lk", "lv");
        let rs = s2("rk", "rv");
        let out = ls.concat(&rs);
        let (lrows, rrows) = (rows(lr), rows(rr));
        let ctx = QueryContext::new();
        let mut j = Join::new(
            Box::new(Values::new(&lrows, &ls)),
            Box::new(Values::new(&rrows, &rs)),
            op,
            on,
            residual,
            &out,
            &ctx,
        );
        let mut got = Vec::new();
        while let Some(b) = j.next().unwrap() {
            for i in 0..b.rows() {
                got.push((0..b.width()).map(|c| b.column(c).value(i)).collect::<Vec<_>>());
            }
        }
        got
    }

    fn v(xs: &[i64]) -> Vec<Value> {
        xs.iter().map(|&x| Value::Int(x)).collect()
    }
    fn vn(xs: &[Option<i64>]) -> Vec<Value> {
        xs.iter().map(|x| x.map_or(Value::Null, Value::Int)).collect()
    }

    const L: &[(i64, i64)] = &[(1, 10), (2, 20), (3, 30)];
    const R: &[(i64, i64)] = &[(2, 200), (3, 300), (4, 400)];

    #[test]
    fn inner_join_keeps_only_matches() {
        let got = joined(L, R, JoinOp::Inner, &[(0, 0)], None);
        assert_eq!(got, vec![v(&[2, 20, 2, 200]), v(&[3, 30, 3, 300])]);
    }

    #[test]
    fn left_join_pads_the_right_side() {
        let got = joined(L, R, JoinOp::Left, &[(0, 0)], None);
        assert_eq!(
            got,
            vec![
                v(&[2, 20, 2, 200]),
                v(&[3, 30, 3, 300]),
                vn(&[Some(1), Some(10), None, None]),
            ]
        );
    }

    #[test]
    fn right_join_pads_the_left_side() {
        let got = joined(L, R, JoinOp::Right, &[(0, 0)], None);
        assert_eq!(got.len(), 3);
        assert_eq!(got[2], vn(&[None, None, Some(4), Some(400)]));
    }

    #[test]
    fn full_join_pads_both() {
        let got = joined(L, R, JoinOp::Full, &[(0, 0)], None);
        assert_eq!(got.len(), 4);
        assert!(got.contains(&vn(&[Some(1), Some(10), None, None])));
        assert!(got.contains(&vn(&[None, None, Some(4), Some(400)])));
    }

    #[test]
    fn cross_join_is_the_full_product() {
        let got = joined(&L[..2], &R[..2], JoinOp::Cross, &[], None);
        assert_eq!(got.len(), 4);
        assert_eq!(got[0], v(&[1, 10, 2, 200]));
        assert_eq!(got[3], v(&[2, 20, 3, 300]));
    }

    #[test]
    fn duplicate_keys_produce_the_full_cartesian_per_key() {
        let l = [(1i64, 10i64), (1, 11)];
        let r = [(1i64, 100i64), (1, 101)];
        let got = joined(&l, &r, JoinOp::Inner, &[(0, 0)], None);
        assert_eq!(got.len(), 4);
    }

    #[test]
    fn composite_join_keys() {
        let l = [(1i64, 5i64), (1, 6)];
        let r = [(1i64, 5i64), (1, 7)];
        let got = joined(&l, &r, JoinOp::Inner, &[(0, 0), (1, 1)], None);
        assert_eq!(got, vec![v(&[1, 5, 1, 5])]);
    }

    #[test]
    fn null_keys_never_match() {
        let ls = s2("lk", "lv");
        let rs = s2("rk", "rv");
        let out = ls.concat(&rs);
        let lrows = vec![vec![Value::Null, Value::Int(1)]];
        let rrows = vec![vec![Value::Null, Value::Int(2)]];
        let ctx = QueryContext::new();
        let mut j = Join::new(
            Box::new(Values::new(&lrows, &ls)),
            Box::new(Values::new(&rrows, &rs)),
            JoinOp::Inner,
            &[(0, 0)],
            None,
            &out,
            &ctx,
        );
        assert!(j.next().unwrap().is_none(), "NULL = NULL is unknown, not true");
    }

    #[test]
    fn null_keys_still_appear_in_an_outer_join() {
        let ls = s2("lk", "lv");
        let rs = s2("rk", "rv");
        let out = ls.concat(&rs);
        let lrows = vec![vec![Value::Null, Value::Int(1)]];
        let rrows = vec![vec![Value::Int(9), Value::Int(2)]];
        let on = [(0usize, 0usize)];
        let ctx = QueryContext::new();
        let mut j = Join::new(
            Box::new(Values::new(&lrows, &ls)),
            Box::new(Values::new(&rrows, &rs)),
            JoinOp::Left,
            &on,
            None,
            &out,
            &ctx,
        );
        let b = j.next().unwrap().unwrap();
        assert_eq!(b.rows(), 1);
        assert!(b.column(0).is_null(0));
        assert!(b.column(2).is_null(0), "no match, so the right side is padded");
    }

    // ------------------------------------------------------------- residual

    fn residual_gt() -> BoundExpr {
        use crate::sql::ast::BinaryOp;
        // lv > rv, against the concatenated schema [lk, lv, rk, rv]
        BoundExpr::Binary {
            left: Box::new(BoundExpr::Column {
                index: 1,
                ty: DataType::Int64,
                name: "lv".into(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(BoundExpr::Column {
                index: 3,
                ty: DataType::Int64,
                name: "rv".into(),
            }),
            ty: DataType::Bool,
        }
    }

    #[test]
    fn residual_filters_equi_matches() {
        let l = [(1i64, 100i64), (2, 1)];
        let r = [(1i64, 5i64), (2, 5)];
        let res = residual_gt();
        let got = joined(&l, &r, JoinOp::Inner, &[(0, 0)], Some(&res));
        assert_eq!(got, vec![v(&[1, 100, 1, 5])]);
    }

    #[test]
    fn a_row_rejected_by_the_residual_is_unmatched_for_an_outer_join() {
        let l = [(2i64, 1i64)];
        let r = [(2i64, 5i64)];
        let res = residual_gt();
        let got = joined(&l, &r, JoinOp::Left, &[(0, 0)], Some(&res));
        assert_eq!(
            got,
            vec![vn(&[Some(2), Some(1), None, None])],
            "the equi-match failed the residual, so the left row is unmatched"
        );
    }

    #[test]
    fn cross_join_with_a_residual_is_a_general_theta_join() {
        let l = [(1i64, 100i64), (2, 1)];
        let r = [(9i64, 5i64)];
        let res = residual_gt();
        let got = joined(&l, &r, JoinOp::Cross, &[], Some(&res));
        assert_eq!(got, vec![v(&[1, 100, 9, 5])]);
    }

    // ------------------------------------------------------------ edge cases

    #[test]
    fn empty_sides() {
        assert!(joined(&[], R, JoinOp::Inner, &[(0, 0)], None).is_empty());
        assert!(joined(L, &[], JoinOp::Inner, &[(0, 0)], None).is_empty());
        // left join against nothing still emits every left row
        let got = joined(L, &[], JoinOp::Left, &[(0, 0)], None);
        assert_eq!(got.len(), 3);
        assert_eq!(got[0], vn(&[Some(1), Some(10), None, None]));
    }

    #[test]
    fn build_side_choice_does_not_change_the_answer() {
        // Make the right side much larger so the build flips to the left.
        let big: Vec<(i64, i64)> = (0..50).map(|i| (i % 5, i)).collect();
        let small = [(1i64, 7i64), (2, 8)];
        let a = joined(&small, &big, JoinOp::Inner, &[(0, 0)], None);
        let b = joined(&big, &small, JoinOp::Inner, &[(0, 0)], None);
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), 20, "keys 1 and 2 appear 10 times each in `big`");
    }

    #[test]
    fn an_absurd_cross_join_streams_instead_of_exhausting_memory() {
        // Inverted from `an_absurd_cross_join_errors_instead_of_exhausting_
        // memory`, which asserted that 10^12 output rows were refused up front
        // by MAX_JOIN_ROWS. The product is no longer materialized, so the
        // right behaviour is to hand back the first block immediately -- which
        // is what makes `CROSS JOIN ... LIMIT n` answerable at all. The
        // protection that replaced the constant is below it: the operator is
        // interruptible, and its inputs are charged to the budget.
        let s = Schema::new(vec![Field::new("a", DataType::Int64)]).unwrap();
        let out = s.concat(&s);
        let big: Vec<Vec<Value>> = (0..100_000i64).map(|i| vec![Value::Int(i)]).collect();
        let ctx = QueryContext::new();
        let mut j = Join::new(
            Box::new(Values::new(&big, &s)),
            Box::new(Values::new(&big, &s)),
            JoinOp::Cross,
            &[],
            None,
            &out,
            &ctx,
        );
        // 10^10 rows in total; the first block must not wait for them.
        let start = std::time::Instant::now();
        let b = j.next().unwrap().unwrap();
        assert_eq!(b.rows(), BLOCK_SIZE);
        assert!(start.elapsed() < std::time::Duration::from_secs(5), "the probe materialized");
        assert_eq!(b.column(0).value(0), Value::Int(0));
        assert_eq!(b.column(1).value(1), Value::Int(1));

        // ... and the runaway that used to be refused is now stoppable.
        ctx.stop();
        assert!(j.next().unwrap_err().to_string().contains("cancelled"));
    }

    #[test]
    fn a_cross_join_resumes_its_nested_loop_exactly_where_it_stopped() {
        // The cursor may cut a left row's inner span across blocks, so the
        // concatenation of every block must still be the plain product.
        let s = Schema::new(vec![Field::new("a", DataType::Int64)]).unwrap();
        let out = s.concat(&s);
        let l: Vec<Vec<Value>> = (0..5i64).map(|i| vec![Value::Int(i)]).collect();
        let r: Vec<Vec<Value>> = (0..3_000i64).map(|i| vec![Value::Int(i)]).collect();
        let ctx = QueryContext::new();
        let mut j = Join::new(
            Box::new(Values::new(&l, &s)),
            Box::new(Values::new(&r, &s)),
            JoinOp::Cross,
            &[],
            None,
            &out,
            &ctx,
        );
        let mut got = Vec::new();
        while let Some(b) = j.next().unwrap() {
            for i in 0..b.rows() {
                got.push((
                    b.column(0).value(i).as_i64().unwrap(),
                    b.column(1).value(i).as_i64().unwrap(),
                ));
            }
        }
        let want: Vec<(i64, i64)> = (0..5).flat_map(|a| (0..3_000).map(move |b| (a, b))).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn the_build_side_is_charged_to_the_budget_and_released() {
        let s = Schema::new(vec![Field::new("a", DataType::Int64)]).unwrap();
        let out = s.concat(&s);
        let big: Vec<Vec<Value>> = (0..100_000i64).map(|i| vec![Value::Int(i)]).collect();
        let on = [(0usize, 0usize)];

        let tight = QueryContext::with_budget(64 << 10);
        let mut j = Join::new(
            Box::new(Values::new(&big, &s)),
            Box::new(Values::new(&big, &s)),
            JoinOp::Inner,
            &on,
            None,
            &out,
            &tight,
        );
        let msg = j.next().unwrap_err().to_string();
        assert!(msg.contains("memory budget"), "{msg}");
        assert!(msg.contains("join"), "the message must name the join: {msg}");
        drop(j);
        assert_eq!(tight.mem.used(), 0);

        // The negative: with room, the same join runs and gives it all back.
        let ctx = QueryContext::new();
        let mut j = Join::new(
            Box::new(Values::new(&big, &s)),
            Box::new(Values::new(&big, &s)),
            JoinOp::Inner,
            &on,
            None,
            &out,
            &ctx,
        );
        let mut n = 0;
        while let Some(b) = j.next().unwrap() {
            n += b.rows();
        }
        assert_eq!(n, 100_000);
        assert!(ctx.mem.used() > 0, "nothing was charged");
        drop(j);
        assert_eq!(ctx.mem.used(), 0);
    }

    #[test]
    fn many_matches_per_key_are_emitted_in_build_row_order_across_blocks() {
        // One key with a fan-out far larger than a block: the chain has to
        // come out ascending, and the block boundary must not reorder it.
        let ls = s2("lk", "lv");
        let rs = s2("rk", "rv");
        let out = ls.concat(&rs);
        // Left is the smaller side, so it is indexed: its three equal keys are
        // one chain, and every probe row must see them as 0, 1, 2.
        let l: Vec<Vec<Value>> = (0..3i64).map(|i| vec![Value::Int(1), Value::Int(i)]).collect();
        let r: Vec<Vec<Value>> = (0..20_000i64)
            .map(|i| vec![Value::Int(1), Value::Int(i)])
            .collect();
        let ctx = QueryContext::new();
        let mut j = Join::new(
            Box::new(Values::new(&l, &ls)),
            Box::new(Values::new(&r, &rs)),
            JoinOp::Inner,
            &[(0, 0)],
            None,
            &out,
            &ctx,
        );
        let mut seq = Vec::new();
        while let Some(b) = j.next().unwrap() {
            for i in 0..b.rows() {
                seq.push((
                    b.column(1).value(i).as_i64().unwrap(),
                    b.column(3).value(i).as_i64().unwrap(),
                ));
            }
        }
        let want: Vec<(i64, i64)> =
            (0..20_000).flat_map(|p| (0..3).map(move |b| (b, p))).collect();
        assert_eq!(seq, want);
    }

    #[test]
    fn out_of_range_join_key_is_an_error() {
        let ls = s2("lk", "lv");
        let rs = s2("rk", "rv");
        let out = ls.concat(&rs);
        let (lrows, rrows) = (rows(L), rows(R));
        let on = [(9usize, 0usize)];
        let ctx = QueryContext::new();
        let mut j = Join::new(
            Box::new(Values::new(&lrows, &ls)),
            Box::new(Values::new(&rrows, &rs)),
            JoinOp::Inner,
            &on,
            None,
            &out,
            &ctx,
        );
        assert!(j.next().is_err());
    }

    #[test]
    fn probe_equal_date_and_uint_keys_must_join() {
        // `Value::Date(5) == Value::UInt(5)` (the same equality `WHERE d = 5`
        // uses and that `key_at` builds its GroupKey from), so an equi-join on
        // these two columns has to match.
        let ls = Schema::new(vec![
            Field::new("d", DataType::Date),
            Field::new("lv", DataType::Int64),
        ])
        .unwrap();
        let rs = Schema::new(vec![
            Field::new("n", DataType::UInt64),
            Field::new("rv", DataType::Int64),
        ])
        .unwrap();
        let out = ls.concat(&rs);
        let lrows = vec![vec![Value::Date(5), Value::Int(1)]];
        let rrows = vec![vec![Value::UInt(5), Value::Int(2)]];
        assert_eq!(lrows[0][0], rrows[0][0], "the keys compare equal");
        let on = [(0usize, 0usize)];

        let ctx = QueryContext::new();
        let mut j = Join::new(
            Box::new(Values::new(&lrows, &ls)),
            Box::new(Values::new(&rrows, &rs)),
            JoinOp::Inner,
            &on,
            None,
            &out,
            &ctx,
        );
        let mut n = 0;
        while let Some(b) = j.next().unwrap() {
            n += b.rows();
        }
        assert_eq!(n, 1, "equal keys produced no join row");

        // ... and a LEFT JOIN must not NULL-pad a row that has a match.
        let mut lj = Join::new(
            Box::new(Values::new(&lrows, &ls)),
            Box::new(Values::new(&rrows, &rs)),
            JoinOp::Left,
            &on,
            None,
            &out,
            &ctx,
        );
        let b = lj.next().unwrap().unwrap();
        assert!(
            !b.column(2).is_null(0),
            "LEFT JOIN NULL-padded a left row that does have a match"
        );
    }

    // ------------------------------------------------------- grace hash join

    fn spilled_dirs() -> Vec<std::path::PathBuf> {
        spill::SPILLED.with(|s| s.borrow().clone())
    }

    fn assert_no_temp_files_left() {
        for d in spilled_dirs() {
            assert!(!d.exists(), "spill directory {} outlived its query", d.display());
        }
        spill::SPILLED.with(|s| s.borrow_mut().clear());
    }

    /// Run one join under a budget, returning its rows *sorted*: a grace join
    /// answers partition by partition, so only the multiset is comparable.
    #[allow(clippy::too_many_arguments)]
    fn under(
        lr: &[Vec<Value>],
        rr: &[Vec<Value>],
        ls: &Schema,
        rs: &Schema,
        op: JoinOp,
        on: &[(usize, usize)],
        residual: Option<&BoundExpr>,
        budget: i64,
    ) -> Vec<Vec<Value>> {
        let out = ls.concat(rs);
        let ctx = QueryContext::with_budget(budget);
        let mut j = Join::new(
            Box::new(Values::new(lr, ls)),
            Box::new(Values::new(rr, rs)),
            op,
            on,
            residual,
            &out,
            &ctx,
        );
        let mut got = Vec::new();
        while let Some(b) = j.next().unwrap() {
            for i in 0..b.rows() {
                got.push((0..b.width()).map(|c| b.column(c).value(i)).collect::<Vec<_>>());
            }
        }
        drop(j);
        assert_eq!(ctx.mem.used(), 0, "the join kept its reservation");
        got.sort();
        got
    }

    /// Two sides with a key domain far smaller than the row count (so every
    /// bucket is populated), duplicates on both sides, keys present on only one
    /// side in both directions, and NULLs in the key.
    fn skewed(n: i64) -> (Schema, Schema, Vec<Vec<Value>>, Vec<Vec<Value>>) {
        let ls = Schema::new(vec![
            Field::new("lk", DataType::Nullable(Box::new(DataType::Int64))),
            Field::new("lv", DataType::Int64),
        ])
        .unwrap();
        let rs = Schema::new(vec![
            Field::new("rk", DataType::Nullable(Box::new(DataType::Int64))),
            Field::new("rv", DataType::Int64),
        ])
        .unwrap();
        let key = |i: i64, m: i64| {
            if i % 37 == 0 {
                Value::Null
            } else {
                Value::Int(crate::common::splitmix64(i as u64) as i64 % m)
            }
        };
        // Left keys reach 4000, right keys only 3000, so both directions have
        // keys the other side lacks -- which is the only way a FULL join's two
        // padding phases both get exercised.
        let l: Vec<Vec<Value>> = (0..n).map(|i| vec![key(i, 4000), Value::Int(i)]).collect();
        let r: Vec<Vec<Value>> =
            (0..n * 2 / 3).map(|i| vec![key(i + 7, 3000), Value::Int(i)]).collect();
        (ls, rs, l, r)
    }

    #[test]
    fn a_spilled_join_answers_exactly_what_the_in_memory_one_does() {
        // The whole claim, for every join type at once: a build side that does
        // not fit is a slower query, not an error, and the rows are the same
        // rows. FULL is the case grace joins break -- an unmatched row on
        // either side has to be padded exactly once, and a scheme that
        // replicated a side would pad it once per partition.
        spill::SPILLED.with(|s| s.borrow_mut().clear());
        let (ls, rs, l, r) = skewed(40_000);
        let on = [(0usize, 0usize)];
        for op in [JoinOp::Inner, JoinOp::Left, JoinOp::Right, JoinOp::Full] {
            let want = under(&l, &r, &ls, &rs, op, &on, None, 512 << 20);
            assert!(spilled_dirs().is_empty(), "the reference run spilled");
            let got = under(&l, &r, &ls, &rs, op, &on, None, 1 << 20);
            assert!(!spilled_dirs().is_empty(), "nothing spilled, so nothing was tested");
            assert_eq!(got.len(), want.len(), "{op:?}: wrong row count");
            assert_eq!(got, want, "{op:?}: a spilled join answered differently");
            assert_no_temp_files_left();
        }
    }

    #[test]
    fn a_spilled_outer_join_pads_a_row_once_and_not_once_per_partition() {
        // The sharpest form of the same trap, with a shape where every left
        // row is unmatched: the answer has exactly `n` rows, and a padding bug
        // multiplies it by the partition count rather than perturbing it.
        spill::SPILLED.with(|s| s.borrow_mut().clear());
        let s = Schema::new(vec![Field::new("k", DataType::Int64)]).unwrap();
        let l: Vec<Vec<Value>> = (0..30_000i64).map(|i| vec![Value::Int(i)]).collect();
        let r: Vec<Vec<Value>> = (0..30_000i64).map(|i| vec![Value::Int(-i - 1)]).collect();
        let on = [(0usize, 0usize)];
        let got = under(&l, &r, &s, &s, JoinOp::Full, &on, None, 1 << 20);
        assert!(!spilled_dirs().is_empty(), "nothing spilled");
        assert_eq!(got.len(), 60_000, "every row of both sides, padded once");
        assert_eq!(got.iter().filter(|row| row[0] == Value::Null).count(), 30_000);
        assert_eq!(got.iter().filter(|row| row[1] == Value::Null).count(), 30_000);
        assert_no_temp_files_left();
    }

    #[test]
    fn a_spilled_join_keeps_the_residual_ahead_of_the_matched_set() {
        // `lv > rv` rejects most equi-matches, and a rejected match must leave
        // *both* rows unmatched. Evaluating the residual after recording the
        // match is the classic way to get this wrong, and partitioning is where
        // a second, careless implementation would appear.
        spill::SPILLED.with(|s| s.borrow_mut().clear());
        let ls = s2("lk", "lv");
        let rs = s2("rk", "rv");
        let l: Vec<Vec<Value>> =
            (0..20_000i64).map(|i| vec![Value::Int(i % 500), Value::Int(i % 97)]).collect();
        let r: Vec<Vec<Value>> =
            (0..20_000i64).map(|i| vec![Value::Int(i % 500), Value::Int(i % 89)]).collect();
        let res = residual_gt();
        let on = [(0usize, 0usize)];
        for op in [JoinOp::Inner, JoinOp::Left, JoinOp::Full] {
            let want = under(&l, &r, &ls, &rs, op, &on, Some(&res), 512 << 20);
            let got = under(&l, &r, &ls, &rs, op, &on, Some(&res), 1 << 20);
            assert!(!spilled_dirs().is_empty(), "nothing spilled");
            assert_eq!(got, want, "{op:?}: the residual moved relative to the matched set");
            assert_no_temp_files_left();
        }
    }

    #[test]
    fn a_spilled_join_re_cuts_a_partition_that_still_does_not_fit() {
        // A budget far below one level's partitions, so `MAX_GRACE_LEVEL` has
        // to be reached through `split` rather than the answer being wrong. If
        // the level did not reseed the mixer, a re-cut would reproduce the
        // parent's split and this would recurse until the stack ran out.
        spill::SPILLED.with(|s| s.borrow_mut().clear());
        let (ls, rs, l, r) = skewed(60_000);
        let on = [(0usize, 0usize)];
        let want = under(&l, &r, &ls, &rs, JoinOp::Full, &on, None, 512 << 20);
        let got = under(&l, &r, &ls, &rs, JoinOp::Full, &on, None, 192 << 10);
        assert!(spilled_dirs().len() > 1, "only one level, so nothing was re-cut");
        assert_eq!(got, want);
        assert_no_temp_files_left();
    }

    #[test]
    fn a_spilled_join_puts_equal_keys_of_different_types_in_one_partition() {
        // `Value::Date(5) == Value::UInt(5)`, so the two rows have to meet --
        // which they only do because `hash_values` agrees with `Value`'s `Eq`.
        // A partition function that hashed the *representation* would separate
        // them and silently drop the match.
        spill::SPILLED.with(|s| s.borrow_mut().clear());
        let ls = Schema::new(vec![Field::new("d", DataType::Date)]).unwrap();
        let rs = Schema::new(vec![Field::new("n", DataType::UInt64)]).unwrap();
        let l: Vec<Vec<Value>> = (0..20_000u64).map(|i| vec![Value::Date(i as u32)]).collect();
        let r: Vec<Vec<Value>> = (0..20_000u64).map(|i| vec![Value::UInt(i)]).collect();
        let on = [(0usize, 0usize)];
        let got = under(&l, &r, &ls, &rs, JoinOp::Inner, &on, None, 512 << 10);
        assert!(!spilled_dirs().is_empty(), "nothing spilled");
        assert_eq!(got.len(), 20_000, "equal keys landed in different partitions");
        assert_no_temp_files_left();
    }

    #[test]
    fn a_spilled_join_that_is_abandoned_early_still_unlinks_its_files() {
        // A `LIMIT` above the join drops the operator with partitions still
        // pending; the directory has to go with it.
        spill::SPILLED.with(|s| s.borrow_mut().clear());
        let (ls, rs, l, r) = skewed(40_000);
        let out = ls.concat(&rs);
        let on = [(0usize, 0usize)];
        let ctx = QueryContext::with_budget(1 << 20);
        let mut j = Join::new(
            Box::new(Values::new(&l, &ls)),
            Box::new(Values::new(&r, &rs)),
            JoinOp::Full,
            &on,
            None,
            &out,
            &ctx,
        );
        assert!(j.next().unwrap().is_some());
        assert!(!spilled_dirs().is_empty(), "nothing spilled");
        drop(j);
        assert_eq!(ctx.mem.used(), 0);
        assert_no_temp_files_left();
    }

    #[test]
    fn a_cross_join_still_refuses_rather_than_spilling() {
        // There is no key to partition on, so the honest answer is the budget
        // error. Pinned because the grace path must not quietly claim to
        // handle a shape it cannot.
        let s = Schema::new(vec![Field::new("a", DataType::Int64)]).unwrap();
        let out = s.concat(&s);
        let big: Vec<Vec<Value>> = (0..100_000i64).map(|i| vec![Value::Int(i)]).collect();
        let ctx = QueryContext::with_budget(64 << 10);
        let mut j = Join::new(
            Box::new(Values::new(&big, &s)),
            Box::new(Values::new(&big, &s)),
            JoinOp::Cross,
            &[],
            None,
            &out,
            &ctx,
        );
        let msg = j.next().unwrap_err().to_string();
        assert!(msg.contains("memory budget"), "{msg}");
    }

    #[test]
    fn non_padded_columns_keep_their_declared_type() {
        let ls = s2("lk", "lv");
        let rs = s2("rk", "rv");
        let out = ls.concat(&rs);
        let (lrows, rrows) = (rows(L), rows(R));
        let on = [(0usize, 0usize)];
        let ctx = QueryContext::new();
        let mut j = Join::new(
            Box::new(Values::new(&lrows, &ls)),
            Box::new(Values::new(&rrows, &rs)),
            JoinOp::Inner,
            &on,
            None,
            &out,
            &ctx,
        );
        let b = j.next().unwrap().unwrap();
        assert_eq!(b.column(0).ty, DataType::Int64, "an inner join adds no nulls");
        assert!(!b.column(0).has_nulls());
    }
}
