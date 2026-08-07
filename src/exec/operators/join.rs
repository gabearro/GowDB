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

use std::mem::size_of;

use crate::common::{BitSet, Error, Result, BLOCK_SIZE};
use crate::exec::expr;
use crate::planner::logical::BoundExpr;
use crate::sql::ast::JoinOp;
use crate::types::{Block, Column, ColumnBuilder, DataType, Schema, Value};

use super::{drain, hash_values, MemGuard, Operator, QueryContext, ScanStats};

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
        Join { left, right, op, on, residual, schema, ctx, state: None }
    }

    /// Drain both sides, decide which one to index, and index it.
    fn prepare(&mut self) -> Result<()> {
        // Separate guards: `drain` charges a running total per side, and one
        // guard cannot hold two independent totals.
        let mut lg = MemGuard::new(self.ctx, "the join's left input");
        let mut rg = MemGuard::new(self.ctx, "the join's right input");
        let l = drain(&mut self.left, self.ctx, &mut lg)?;
        let r = drain(&mut self.right, self.ctx, &mut rg)?;

        // Ties build on the right so the probe (and therefore the output)
        // stays in left-row order, which is what a reader expects to see.
        let build_right = r.rows() <= l.rows();
        let (bcols, pcols): (Vec<usize>, Vec<usize>) = if build_right {
            (self.on.iter().map(|&(_, b)| b).collect(), self.on.iter().map(|&(a, _)| a).collect())
        } else {
            (self.on.iter().map(|&(a, _)| a).collect(), self.on.iter().map(|&(_, b)| b).collect())
        };
        let (build, probe) = if build_right { (&r, &l) } else { (&l, &r) };
        check_cols(build, &bcols)?;
        check_cols(probe, &pcols)?;

        let mut guard = MemGuard::new(self.ctx, "the join hash index");
        let mut key = Vec::with_capacity(bcols.len());
        let mut idx_bytes = 0;
        let idx = if self.on.is_empty() {
            None
        } else {
            let ix = BuildIndex::build(build, &bcols, &mut key);
            idx_bytes = ix.bytes();
            guard.grow_to(idx_bytes)?;
            Some(ix)
        };

        let (want_left, want_right) = padding_wanted(self.op);
        self.state = Some(State {
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
            _sides: [lg, rg],
        });
        Ok(())
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
        if self.state.is_none() {
            self.prepare()?;
        }
        // Copy the shared references out first: they are `Copy`, so this ends
        // the borrow of `self` before `state` is borrowed mutably.
        let (ctx, op, residual, schema) = (self.ctx, self.op, self.residual, self.schema);
        self.state.as_mut().unwrap().next_block(ctx, op, residual, schema)
    }
}

impl State {
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
