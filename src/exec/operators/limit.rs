//! `LIMIT n OFFSET k`, and ClickHouse's `LIMIT n BY (keys)`.
//!
//! Both are streaming and both stop pulling the moment they are satisfied,
//! which is what makes `SELECT ... LIMIT 10` over a billion-row table cost one
//! granule rather than a billion rows: the scan below is never asked for a
//! second batch.
//!
//! `LIMIT n BY` is the ClickHouse extension that keeps the first `n` rows for
//! each distinct key tuple -- "three sample URLs per domain" in one pass,
//! where standard SQL needs a window function. It cannot stop early (a new key
//! may appear in the last batch), so it is streaming but not short-circuiting.

use std::mem::size_of;

use crate::common::{FastMap, Result};
use crate::exec::expr;
use crate::planner::logical::BoundExpr;
use crate::types::{Block, Schema, Value};

use super::{GroupKey, MemGuard, Operator, QueryContext, ScanStats};

pub struct Limit<'a> {
    input: Box<dyn Operator + 'a>,
    ctx: &'a QueryContext,
    limit: Option<usize>,
    offset: usize,
    skipped: usize,
    emitted: usize,
    done: bool,
}

impl<'a> Limit<'a> {
    pub fn new(
        input: Box<dyn Operator + 'a>,
        limit: Option<usize>,
        offset: usize,
        ctx: &'a QueryContext,
    ) -> Limit<'a> {
        Limit { input, ctx, limit, offset, skipped: 0, emitted: 0, done: false }
    }
}

impl Operator for Limit<'_> {
    fn schema(&self) -> &Schema {
        self.input.schema()
    }

    fn stats(&self) -> ScanStats {
        self.input.stats()
    }

    fn next(&mut self) -> Result<Option<Block>> {
        if self.done {
            return Ok(None);
        }
        loop {
            if self.limit.is_some_and(|l| self.emitted >= l) {
                self.done = true;
                return Ok(None);
            }
            // A single `next()` can burn through many blocks while skipping a
            // large OFFSET, so the checkpoint belongs here rather than only in
            // the caller's loop.
            self.ctx.check()?;
            let Some(mut b) = self.input.next()? else {
                self.done = true;
                return Ok(None);
            };
            if self.skipped < self.offset {
                let skip = (self.offset - self.skipped).min(b.rows());
                self.skipped += skip;
                if skip == b.rows() {
                    continue;
                }
                b = b.slice(skip, b.rows());
            }
            if let Some(l) = self.limit {
                let room = l - self.emitted;
                if b.rows() > room {
                    b = b.slice(0, room);
                }
            }
            if b.rows() == 0 {
                continue;
            }
            self.emitted += b.rows();
            return Ok(Some(b));
        }
    }
}

/// `LIMIT n BY (keys)`: the first `n` rows per distinct key tuple.
pub struct LimitBy<'a> {
    input: Box<dyn Operator + 'a>,
    ctx: &'a QueryContext,
    limit: usize,
    keys: &'a [BoundExpr],
    seen: FastMap<GroupKey, usize>,
    /// The key tuple of the row in hand, refilled per row and never
    /// reallocated. It is a `GroupKey` rather than a `Vec<Value>` so the map
    /// can be probed with `&self.probe` directly: `HashMap` wants an owned key
    /// only to *insert*, and the old `entry(row_key(..))` built one per input
    /// row -- a heap allocation per row, to reach a counter that is usually
    /// already there. Measured interleaved against that form with an
    /// `AtomicBool`, alternating sides, best-of-11 over 2M rows:
    /// `LIMIT 3 BY k` (50k keys) 78.7 -> 37.6 ms (2.09x), `LIMIT 2 BY s`
    /// (4 keys) 74.6 -> 47.1 ms (1.58x), `LIMIT 1 BY id` (every row its own
    /// key, so every row still allocates) 228.1 -> 214.9 ms (1.06x).
    probe: GroupKey,
    /// Survivor row ids for the block in hand, likewise reused.
    sel: Vec<u32>,
    /// `seen` is the one thing here that grows with the data rather than with
    /// the block: one entry per distinct key tuple, held to end of stream.
    guard: MemGuard,
}

impl<'a> LimitBy<'a> {
    pub fn new(
        input: Box<dyn Operator + 'a>,
        limit: usize,
        keys: &'a [BoundExpr],
        ctx: &'a QueryContext,
    ) -> LimitBy<'a> {
        LimitBy {
            input,
            ctx,
            limit,
            keys,
            seen: FastMap::default(),
            probe: GroupKey(Vec::with_capacity(keys.len())),
            sel: Vec::new(),
            guard: MemGuard::new(ctx, "the LIMIT BY key table"),
        }
    }

    /// Charged once per block, never per key. The map's own table is ~8/7 of
    /// its live entries (it grows before the load factor reaches 1), and each
    /// `GroupKey` owns a heap tuple of `Value`s on top of that.
    fn seen_bytes(&self) -> usize {
        let per = size_of::<(GroupKey, usize)>()
            + 1
            + self.keys.len() * size_of::<Value>();
        self.seen.len() * per * 8 / 7
    }
}

impl Operator for LimitBy<'_> {
    fn schema(&self) -> &Schema {
        self.input.schema()
    }

    fn stats(&self) -> ScanStats {
        self.input.stats()
    }

    fn next(&mut self) -> Result<Option<Block>> {
        loop {
            self.ctx.check()?;
            let Some(b) = self.input.next()? else { break };
            if b.rows() == 0 {
                continue;
            }
            let key_cols = expr::eval_all(self.keys, &b)?;
            // Split out so the probe buffer and the map can be borrowed at
            // once; `entry()` is not usable here because it demands an owned
            // key up front, which is exactly the allocation being avoided.
            let (seen, probe, sel) = (&mut self.seen, &mut self.probe, &mut self.sel);
            sel.clear();
            for i in 0..b.rows() {
                probe.0.clear();
                probe.0.extend(key_cols.iter().map(|c| c.value(i)));
                match seen.get_mut(probe) {
                    Some(n) if *n >= self.limit => continue,
                    Some(n) => *n += 1,
                    // Only a genuinely new key pays for a key tuple, so the
                    // allocation count is the cardinality rather than the row
                    // count. A row that is its own key -- the shape that makes
                    // this operator unbounded -- is unchanged; everything else
                    // stops allocating entirely.
                    None => {
                        seen.insert(probe.clone(), 1);
                    }
                }
                sel.push(i as u32);
            }
            let held = self.seen_bytes();
            self.guard.grow_to(held)?;
            if self.sel.is_empty() {
                continue;
            }
            if self.sel.len() == b.rows() {
                return Ok(Some(b));
            }
            return Ok(Some(b.take(&self.sel)));
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::Values;
    use crate::types::{DataType, Field, Value};

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("v", DataType::Int64),
        ])
        .unwrap()
    }

    fn rows(n: i64) -> Vec<Vec<Value>> {
        (0..n).map(|i| vec![Value::Int(i % 3), Value::Int(i)]).collect()
    }

    fn collect(op: &mut dyn Operator, col: usize) -> Vec<i64> {
        let mut out = Vec::new();
        while let Some(b) = op.next().unwrap() {
            out.extend_from_slice(b.column(col).as_i64().unwrap());
        }
        out
    }

    #[test]
    fn limit_truncates() {
        let (s, r) = (schema(), rows(10));
        let ctx = QueryContext::new();
        let mut l = Limit::new(Box::new(Values::new(&r, &s)), Some(3), 0, &ctx);
        assert_eq!(collect(&mut l, 1), vec![0, 1, 2]);
    }

    #[test]
    fn offset_skips() {
        let (s, r) = (schema(), rows(10));
        let ctx = QueryContext::new();
        let mut l = Limit::new(Box::new(Values::new(&r, &s)), Some(3), 4, &ctx);
        assert_eq!(collect(&mut l, 1), vec![4, 5, 6]);
    }

    #[test]
    fn offset_without_limit_runs_to_the_end() {
        let (s, r) = (schema(), rows(6));
        let ctx = QueryContext::new();
        let mut l = Limit::new(Box::new(Values::new(&r, &s)), None, 4, &ctx);
        assert_eq!(collect(&mut l, 1), vec![4, 5]);
    }

    #[test]
    fn offset_past_the_end_yields_nothing() {
        let (s, r) = (schema(), rows(3));
        let ctx = QueryContext::new();
        let mut l = Limit::new(Box::new(Values::new(&r, &s)), Some(5), 99, &ctx);
        assert!(collect(&mut l, 1).is_empty());
    }

    #[test]
    fn limit_zero_yields_nothing() {
        let (s, r) = (schema(), rows(3));
        let ctx = QueryContext::new();
        let mut l = Limit::new(Box::new(Values::new(&r, &s)), Some(0), 0, &ctx);
        assert!(collect(&mut l, 1).is_empty());
    }

    #[test]
    fn limit_spanning_several_batches() {
        use crate::common::BLOCK_SIZE;
        let s = Schema::new(vec![Field::new("v", DataType::Int64)]).unwrap();
        let r: Vec<Vec<Value>> = (0..BLOCK_SIZE as i64 * 2)
            .map(|i| vec![Value::Int(i)])
            .collect();
        let ctx = QueryContext::new();
        let mut l = Limit::new(Box::new(Values::new(&r, &s)), Some(BLOCK_SIZE + 5), 0, &ctx);
        let got = collect(&mut l, 0);
        assert_eq!(got.len(), BLOCK_SIZE + 5);
        assert_eq!(got[BLOCK_SIZE + 4], BLOCK_SIZE as i64 + 4);
    }

    #[test]
    fn limit_stops_pulling_once_satisfied() {
        // A counting source proves the short-circuit rather than assuming it.
        struct Counting<'b> {
            inner: Values<'b>,
            pulls: std::rc::Rc<std::cell::Cell<usize>>,
        }
        impl Operator for Counting<'_> {
            fn schema(&self) -> &Schema {
                self.inner.schema()
            }
            fn next(&mut self) -> Result<Option<Block>> {
                self.pulls.set(self.pulls.get() + 1);
                self.inner.next()
            }
        }
        use crate::common::BLOCK_SIZE;
        let s = Schema::new(vec![Field::new("v", DataType::Int64)]).unwrap();
        let r: Vec<Vec<Value>> = (0..BLOCK_SIZE as i64 * 4)
            .map(|i| vec![Value::Int(i)])
            .collect();
        let pulls = std::rc::Rc::new(std::cell::Cell::new(0));
        let ctx = QueryContext::new();
        let mut l = Limit::new(
            Box::new(Counting { inner: Values::new(&r, &s), pulls: pulls.clone() }),
            Some(2),
            0,
            &ctx,
        );
        assert_eq!(collect(&mut l, 0), vec![0, 1]);
        assert_eq!(pulls.get(), 1, "only one batch should ever be pulled");
    }

    // ------------------------------------------------------------- LIMIT BY

    #[test]
    fn limit_by_keeps_n_per_key() {
        let (s, r) = (schema(), rows(12));
        let keys = vec![BoundExpr::Column { index: 0, ty: DataType::Int64, name: "k".into() }];
        let ctx = QueryContext::new();
        let mut l = LimitBy::new(Box::new(Values::new(&r, &s)), 2, &keys, &ctx);
        // keys cycle 0,1,2 -> first two of each are v = 0,1,2,3,4,5
        let mut got = collect(&mut l, 1);
        got.sort();
        assert_eq!(got, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn limit_by_with_no_keys_is_a_plain_limit() {
        let (s, r) = (schema(), rows(10));
        let keys: Vec<BoundExpr> = vec![];
        let ctx = QueryContext::new();
        let mut l = LimitBy::new(Box::new(Values::new(&r, &s)), 3, &keys, &ctx);
        assert_eq!(collect(&mut l, 1), vec![0, 1, 2]);
    }

    #[test]
    fn limit_by_counts_across_batches() {
        use crate::common::BLOCK_SIZE;
        let s = schema();
        // One key, more rows than a batch: the per-key counter must persist.
        let r: Vec<Vec<Value>> = (0..BLOCK_SIZE as i64 + 10)
            .map(|i| vec![Value::Int(0), Value::Int(i)])
            .collect();
        let keys = vec![BoundExpr::Column { index: 0, ty: DataType::Int64, name: "k".into() }];
        let ctx = QueryContext::new();
        let mut l = LimitBy::new(Box::new(Values::new(&r, &s)), 3, &keys, &ctx);
        assert_eq!(collect(&mut l, 1), vec![0, 1, 2]);
    }

    #[test]
    fn limit_by_charges_its_key_table_and_gives_it_back() {
        // One distinct key per row is the shape that makes `LIMIT n BY`
        // unbounded: the counter table is as large as the input.
        let s = schema();
        let r: Vec<Vec<Value>> = (0..60_000i64)
            .map(|i| vec![Value::Int(i), Value::Int(i)])
            .collect();
        let keys = vec![BoundExpr::Column { index: 0, ty: DataType::Int64, name: "k".into() }];

        let tight = QueryContext::with_budget(64 << 10);
        let mut l = LimitBy::new(Box::new(Values::new(&r, &s)), 1, &keys, &tight);
        let mut err = None;
        while err.is_none() {
            match l.next() {
                Ok(None) => break,
                Ok(Some(_)) => {}
                Err(e) => err = Some(e),
            }
        }
        let msg = err.expect("60k distinct keys fit in 64 KiB?").to_string();
        assert!(msg.contains("LIMIT BY key table"), "{msg}");
        drop(l);
        assert_eq!(tight.mem.used(), 0);

        let ctx = QueryContext::new();
        let mut l = LimitBy::new(Box::new(Values::new(&r, &s)), 1, &keys, &ctx);
        assert_eq!(collect(&mut l, 1).len(), 60_000);
        drop(l);
        assert_eq!(ctx.mem.used(), 0);
    }
}
