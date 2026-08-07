//! A table: a schema, a write buffer, and a list of immutable parts.
//!
//! ## The OLTP/OLAP split
//!
//! Writes land in the [`Delta`] at hash-map speed. Reads take one of two
//! paths, and this is the whole trick to being good at both workloads:
//!
//!   * **point lookups** (`get`/`multi_get`) consult the delta first, then
//!     probe parts newest-first. A bloom probe skips a foreign part in one
//!     cache line; inside a part, the router plus the learned-rank index find
//!     the row without decompressing anything.
//!   * **scans** flush the delta first, then read packed granules directly.
//!
//! Flushing before a scan is a deliberate choice. The alternative -- teaching
//! every scan operator to merge an unsorted hash map -- would put a branch and
//! a hash probe in the innermost loop of every aggregate, which is exactly the
//! loop that has to run at memory bandwidth. Paying one small part on the
//! first scan after a write burst is much cheaper, and compaction folds it
//! away. ([`Delta::freeze`] is the way out of even that; see its docs.)
//!
//! ## Snapshots
//!
//! The part list is `RwLock<Arc<PartSet>>` and every mutation is
//! copy-on-write: clone two vectors of pointers, edit the entries that
//! changed, bump the version, swap the `Arc`. A reader calls
//! [`Table::snapshot`] **once per query**, which is one uncontended lock
//! acquisition and one `Arc` clone -- about 20 ns against a scan that runs for
//! milliseconds. Doing it per part, or per granule, would be a catastrophic
//! regression, so the snapshot is threaded down into the scan loop as a
//! borrow and never re-derived.
//!
//! Mutating paths take `&mut self`, so they reach the set through
//! `RwLock::get_mut` and never lock at all: the lock exists for the future
//! concurrent reader, and costs the single-threaded writer nothing.
//!
//! Measured interleaved against the pre-snapshot tree (best of 12 per side,
//! 8 alternating rounds, 5M rows): parallel scan 1.05x, streaming scan 1.01x,
//! point lookup 0.99x, batched lookup 1.01x, single-row write 1.01x -- every
//! one inside this machine's ~5% noise floor. The versioning is free; the
//! win is on the delete path, and it is recorded next to [`Deletes`].
//!
//! ## Transactions
//!
//! A transaction is one more `Arc<PartSet>`: [`Table::begin_txn`] clones the
//! published pointer into `txn`, every mutation from then on edits *that* set
//! instead of the published one, and [`Table::commit_txn`] stores it back over
//! the published pointer. Because writers serialize -- the roadmap's decided
//! position, single-writer / multi-reader snapshot isolation, not MVCC -- the
//! published set cannot have moved in the meantime, so the transaction's
//! private view **is** the committed set at COMMIT and publishing it is a
//! pointer store. COMMIT is O(1), not O(parts) and certainly not O(rows).
//!
//! ROLLBACK is `txn = None` plus a delta clear. Nothing has to be undone
//! because nothing was done to the published set: parts are immutable, so the
//! overlay only ever *added* pointers and copy-on-write copies of delete masks,
//! and dropping the last `Arc` to them is the whole of the work. This is the
//! capability the keystone was for.
//!
//! The cost when no transaction is open is one `Option` discriminant test, on
//! paths that already run once per query ([`Table::snapshot`]) or once per
//! batch (`edit`/`peek`). Nothing per row, per block or per granule; the
//! `txn` field is an `Option<Arc<PartSet>>`, which is niche-packed into the
//! same eight bytes the `Arc` occupies.
//!
//! Measured interleaved against the pre-transaction shape (a temporary copy of
//! the old `locate` body in the tree, best-of-9 per side, alternating in one
//! loop, 200k probes into a 400k-row part): 0.985 / 1.001 / 1.013 -- the
//! `peek` on the 24 ns point-lookup path is the tightest place the branch
//! lands, and it does not show. Buffered writes and flushes measured *inside*
//! a transaction against the same table outside one: write 0.96-1.01x, flush
//! 0.99-1.01x. A transaction is free to be in.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::{Arc, RwLock};

use crate::common::pool;
use crate::common::{
    hash_key, Error, FastMap, Result, BLOCK_SIZE, FP_SEED, GRANULE_SIZE, G_SHIFT,
};
use crate::exec::expr;
use crate::planner::logical::{BoundExpr, ZoneFilter};
use crate::sort::{radix_sort_composite, radix_sort_soa};
use crate::types::{
    Block, Column, ColumnBuilder, ColumnData, DataType, PhysicalType, Schema, TableDef, Value,
};

use super::delta::{Delta, DeltaEntry, DeltaImage};
use super::granule::{Granule, Stats};
use super::part::{gather_rows, Deletes, Part, PartSet, Snapshot};

/// Inserts at least this large bypass the delta and become a part directly:
/// sorting and packing a big batch in one pass beats routing it through a hash
/// map row by row.
pub const BULK_INSERT_THRESHOLD: usize = 4 * GRANULE_SIZE;

/// Compact once the part count reaches this, to bound point-lookup fan-out.
pub const AUTO_COMPACT_PARTS: usize = 16;

/// Bytes a streaming scan tries to keep its decode buffer within, across all
/// projected columns. Sized for a conservative 32 KB L1 data cache.
const SCAN_L1_BUDGET: usize = 32 * 1024;

/// Granules a parallel scan worker claims per trip to the shared counter.
/// Large enough that the atomic disappears into the decode cost, small enough
/// that the last claim cannot leave a thread far behind the others.
const SCAN_CLAIM: usize = 8;

#[cfg(test)]
thread_local! {
    /// Countdown fault injection for the failure-atomicity tests: `n` means
    /// the `n`th build from now succeeds and the one after it fails.
    /// `usize::MAX` is disarmed.
    ///
    /// A test cannot otherwise provoke a build error: everything
    /// `Part::build_sel` can return `Err` for is an allocation or a codec
    /// failure deep inside the packer, and what these tests are about is the
    /// *table's* behaviour when one happens, not the packer's. A countdown
    /// rather than a flag because statement-level atomicity is about the
    /// *second* publish failing after the first succeeded, which a one-shot
    /// armed before the statement cannot express. Thread-local so concurrently
    /// running tests cannot arm each other's.
    static FAIL_BUILD: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

/// Arm with [`arm_build_failure`] and the next part build in `flush` or
/// `merge_parts` fails. `#[cfg(test)]`-gated, so the release build of the
/// flush path -- ~142 ns/row, run every 64k rows and on every SELECT --
/// contains no trace of it.
#[cfg(test)]
#[inline]
fn fail_build() -> Result<()> {
    FAIL_BUILD.with(|f| match f.get() {
        0 => {
            f.set(usize::MAX);
            Err(Error::storage("injected part-build failure"))
        }
        usize::MAX => Ok(()),
        n => {
            f.set(n - 1);
            Ok(())
        }
    })
}

#[cfg(not(test))]
#[inline(always)]
fn fail_build() -> Result<()> {
    Ok(())
}

#[cfg(test)]
fn arm_build_failure() {
    arm_build_failure_after(0);
}

/// The `after`th build from now succeeds; the next one fails.
#[cfg(test)]
pub(crate) fn arm_build_failure_after(after: usize) {
    FAIL_BUILD.with(|f| f.set(after));
}

/// Disarm, for a test whose injected failure may or may not have fired.
#[cfg(test)]
pub(crate) fn disarm_build_failure() {
    FAIL_BUILD.with(|f| f.set(usize::MAX));
}

#[cfg(test)]
thread_local! {
    /// Countdown fault injection for `ingest_block`'s tombstone loop: the
    /// `n`th key of the batch fails, with `n-1` tombstones already applied.
    ///
    /// The real failure is `Value::to_lane` on a key the coercion pass let
    /// through (a `Null` in a `UInt64` primary key reaches it as `Value::Null`,
    /// which has no lane), but provoking that *mid-batch* means smuggling one
    /// bad row into an otherwise valid 32k-row bulk insert, and the test is
    /// about the table's response, not about which values lack lanes.
    static FAIL_TOMBSTONE_IN: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

#[cfg(test)]
#[inline]
fn fail_tombstone() -> Result<()> {
    FAIL_TOMBSTONE_IN.with(|f| match f.get() {
        0 => {
            f.set(usize::MAX);
            Err(Error::storage("injected tombstone failure"))
        }
        usize::MAX => Ok(()),
        n => {
            f.set(n - 1);
            Ok(())
        }
    })
}

#[cfg(not(test))]
#[inline(always)]
fn fail_tombstone() -> Result<()> {
    Ok(())
}

#[cfg(test)]
fn arm_tombstone_failure(after: usize) {
    FAIL_TOMBSTONE_IN.with(|f| f.set(after));
}

/// Where a located row lives. `Copy` and pointer-free, so finding a row costs
/// no allocation at all.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RowLoc {
    /// Buffered in the write memtable, not yet in a part.
    Delta,
    Part { part: u32, pos: u32 },
}

pub struct Table {
    pub def: TableDef,
    /// Derived from `def` once at construction.
    ///
    /// `TableDef::pk_col` is not a field read: it walks the schema and runs
    /// several recursive type matches to decide whether the fast primary-key
    /// path applies. That is free once and costly per row, and the write path
    /// asks per row. `def` is only replaced wholesale (ALTER rebuilds the
    /// table), so these can never go stale.
    pk_col: Option<usize>,
    sort_col: Option<usize>,
    /// Physical kind of the primary-key column, for lane conversion.
    pk_phys: Option<PhysicalType>,
    delta: Delta,
    parts: RwLock<Arc<PartSet>>,
    /// The open transaction's private view: the committed parts as of
    /// `begin_txn`, plus everything this transaction has written since.
    /// `None` outside a transaction, which is the case every hot path is
    /// tuned for -- see the module docs.
    txn: Option<Arc<PartSet>>,
    delta_limit: usize,
    pub stats: Stats,
}

/// The part set every mutation edits, reached from a `&mut self` path.
///
/// `RwLock::get_mut` is a field access, not a lock: exclusive access to the
/// table already excludes every reader, so the writer pays nothing for the
/// lock that makes concurrent readers possible. `Arc::make_mut` then clones
/// the set only if a snapshot is still holding the old one -- in the
/// single-threaded case that is a refcount check and nothing else.
///
/// With a transaction open the target is its private view instead, and the
/// published set is not touched at all. That one redirection is the whole of
/// write-side transaction support: every caller below already funnels through
/// here, so there is exactly one place that has to know.
#[inline]
fn edit<'a>(
    parts: &'a mut RwLock<Arc<PartSet>>,
    txn: &'a mut Option<Arc<PartSet>>,
) -> &'a mut PartSet {
    // A poisoned lock still holds a perfectly good part set: the invariant
    // that could have been broken is a *mutation in flight*, and mutations
    // never hold the lock (see above). Refusing to read it would turn one
    // unrelated panic into a permanently unusable table.
    let arc = match txn {
        Some(view) => view,
        None => parts.get_mut().unwrap_or_else(|e| e.into_inner()),
    };
    Arc::make_mut(arc)
}

/// Read-only view of the part set from a `&mut self` path. Same reasoning as
/// [`edit`], minus the copy-on-write.
#[inline]
fn peek<'a>(parts: &'a mut RwLock<Arc<PartSet>>, txn: &'a Option<Arc<PartSet>>) -> &'a PartSet {
    match txn {
        Some(view) => view,
        None => parts.get_mut().unwrap_or_else(|e| e.into_inner()),
    }
}

/// Swap in a set that was built up privately, discarding the current one.
///
/// The counterpart to [`edit`] for a mutation with a *fallible* step in the
/// middle: `edit` hands out the live set (in the uncontended case literally
/// the published allocation), so an error partway through leaves half the
/// edit visible. Cloning first and swapping here makes the whole batch one
/// generation, which is the same guarantee `edit` gives -- just extended over
/// a sequence that can fail. Publication is a pointer store, so it cannot.
#[inline]
fn publish(parts: &mut RwLock<Arc<PartSet>>, txn: &mut Option<Arc<PartSet>>, set: PartSet) {
    match txn {
        Some(view) => *view = Arc::new(set),
        None => *parts.get_mut().unwrap_or_else(|e| e.into_inner()) = Arc::new(set),
    }
}

/// Can any row of this granule satisfy every zone filter?
///
/// The mutation-side mirror of `operators::scan::Scan::prunes`, and it has to
/// agree with it exactly: `zf.col` indexes the *projected* schema while
/// `Granule::columns` is indexed by table column, so the mapping runs through
/// `projection` in both places. Getting it backwards prunes on the wrong
/// column, and on this path that is not a slow answer but a wrong one -- rows
/// silently kept, or silently deleted. Bounds come from the frame-of-reference
/// header, so a pruned granule costs two `Value` comparisons and no decode.
#[inline]
fn zone_prunes(g: &Granule, projection: &[usize], zone: &[ZoneFilter]) -> bool {
    zone.iter().any(|zf| {
        projection
            .get(zf.col)
            .and_then(|&c| g.columns.get(c))
            .is_some_and(|pc| !zf.may_match(&pc.min_value(), &pc.max_value()))
    })
}

impl Table {
    pub fn new(def: TableDef, delta_limit: usize) -> Table {
        Table::from_part_set(def, PartSet::new(), delta_limit)
    }

    /// Rebuild a table from parts loaded off disk. The parts must already be
    /// sorted and indexed consistently with `def`.
    ///
    /// Adoption is where a decoded part's on-disk delete mask stops belonging
    /// to the part and starts belonging to the versioned set.
    pub fn from_parts(def: TableDef, parts: Vec<Part>, delta_limit: usize) -> Table {
        Table::from_part_set(def, PartSet::adopt(parts), delta_limit)
    }

    fn from_part_set(def: TableDef, set: PartSet, delta_limit: usize) -> Table {
        let keyed = def.has_fast_pk();
        let delta = Delta::new(keyed, &def.schema);
        let (pk_col, sort_col) = (def.pk_col(), def.sort_col());
        let pk_phys = pk_col.map(|c| def.schema.ty(c).base().physical());
        Table {
            def,
            pk_col,
            sort_col,
            pk_phys,
            delta,
            parts: RwLock::new(Arc::new(set)),
            txn: None,
            delta_limit,
            stats: Stats::default(),
        }
    }

    /// Cached `def.pk_col()`.
    #[inline(always)]
    pub fn pk_col(&self) -> Option<usize> {
        self.pk_col
    }

    /// Cached `def.sort_col()`.
    #[inline(always)]
    pub fn sort_col(&self) -> Option<usize> {
        self.sort_col
    }

    pub fn schema(&self) -> &Schema {
        &self.def.schema
    }

    /// Pin the current part set. One lock acquisition, one `Arc` clone.
    ///
    /// Everything a query reads must come from a single snapshot: the parts
    /// and the delete masks are published together, so mixing two of them is
    /// exactly the torn read this design exists to make impossible.
    ///
    /// Inside a transaction this is the transaction's own view, which is what
    /// read-your-own-writes *means*: the uncommitted parts are already layered
    /// over the pinned committed ones, so no operator learns anything about
    /// transactions and nothing checks an overlay per block or per granule.
    /// The lock is skipped entirely in that case -- the view is private.
    #[inline]
    pub fn snapshot(&self) -> Snapshot {
        if let Some(view) = &self.txn {
            return Snapshot::new(Arc::clone(view));
        }
        let g = self.parts.read().unwrap_or_else(|e| e.into_inner());
        Snapshot::new(Arc::clone(&g))
    }

    /// Pin the *committed* part set, ignoring any open transaction.
    ///
    /// What a reader that is not the writing session must see. A transaction's
    /// uncommitted parts never enter the published set, so this is exactly the
    /// pre-transaction view until COMMIT stores the overlay over it -- at
    /// which point it becomes the whole transaction at once, never a prefix.
    #[inline]
    pub fn committed_snapshot(&self) -> Snapshot {
        let g = self.parts.read().unwrap_or_else(|e| e.into_inner());
        Snapshot::new(Arc::clone(&g))
    }

    // ------------------------------------------------------- transactions

    /// Open a transaction: from here to `commit_txn`/`rollback_txn`, every
    /// mutation lands in a private overlay and the published set stands still.
    ///
    /// The write buffer is flushed first, so the transaction starts from a
    /// clean boundary and `rollback_txn` can discard the delta wholesale
    /// without eating writes that were already committed. Free where it
    /// matters: every caller that opens a transaction has just planned a query
    /// (which flushes) or is starting from an already-empty delta.
    ///
    /// Nested transactions are not a thing here -- the second `begin_txn` is
    /// the caller's bug, not a savepoint -- so it is refused rather than
    /// silently flattened.
    pub fn begin_txn(&mut self) -> Result<()> {
        if self.txn.is_some() {
            return Err(Error::storage(format!(
                "table `{}` is already inside a transaction",
                self.def.name
            )));
        }
        self.flush()?;
        let g = self.parts.get_mut().unwrap_or_else(|e| e.into_inner());
        self.txn = Some(Arc::clone(g));
        Ok(())
    }

    /// Publish everything the transaction built, as one new version.
    ///
    /// One pointer store, whatever the transaction did. Writers serialize, so
    /// the published set is still exactly what it was at `begin_txn` and the
    /// overlay already *is* the committed set -- there is nothing to merge and
    /// nothing to replay. Buffered rows are the caller's to flush first; doing
    /// it here would make the one infallible step in a COMMIT fallible.
    pub fn commit_txn(&mut self) {
        if let Some(view) = self.txn.take() {
            *self.parts.get_mut().unwrap_or_else(|e| e.into_inner()) = view;
        }
    }

    /// Drop the transaction's overlay and its buffered writes.
    ///
    /// Costs one `Arc` drop and one delta clear. Nothing is rewound because
    /// nothing was written: parts are immutable, so the overlay held new
    /// pointers and copy-on-write copies of the delete masks it touched, and
    /// the published set never saw either. This is the payoff the `Arc<PartSet>`
    /// keystone was for -- before it, a tombstone was a bit flipped inside a
    /// live part and there was no way back.
    pub fn rollback_txn(&mut self) {
        if self.txn.take().is_some() {
            // Everything in the delta arrived after `begin_txn` emptied it.
            self.delta.clear();
        }
    }

    #[inline]
    pub fn in_txn(&self) -> bool {
        self.txn.is_some()
    }

    /// Freeze the write buffer into an immutable, key-sorted image.
    ///
    /// The other half of a snapshot. Nothing reads it yet -- every SELECT
    /// still flushes first, and removing that means teaching the planner in
    /// `session.rs` to carry a `(Snapshot, Arc<DeltaImage>)` pair instead of
    /// calling `flush_all` -- but the storage side is complete: this and
    /// [`Table::snapshot`] together describe the table at one instant, with
    /// no I/O, no part build, and no mutation.
    pub fn freeze_delta(&self) -> Result<Arc<DeltaImage>> {
        self.delta.freeze()
    }

    /// Replace the whole part list, e.g. after loading a table directory.
    ///
    /// Deliberately targets the *published* set: this is recovery installing a
    /// table's on-disk contents, which happens before anything could have
    /// opened a transaction over it.
    pub fn set_parts(&mut self, parts: Vec<Part>) {
        let mut none = None;
        let set = edit(&mut self.parts, &mut none);
        *set = PartSet::adopt(parts);
        set.bump();
    }

    pub fn part_count(&self) -> usize {
        match &self.txn {
            Some(view) => view.len(),
            None => self.parts.read().unwrap_or_else(|e| e.into_inner()).len(),
        }
    }
    /// Monotonic version of the part set this session reads. Bumped by every
    /// mutation, so a cached plan can tell whether the storage under it moved.
    pub fn parts_version(&self) -> u64 {
        match &self.txn {
            Some(view) => view.version(),
            None => self.parts.read().unwrap_or_else(|e| e.into_inner()).version(),
        }
    }
    pub fn delta_len(&self) -> usize {
        self.delta.len()
    }
    pub fn has_pending_writes(&self) -> bool {
        !self.delta.is_empty()
    }

    // ------------------------------------------------------------ ingestion

    /// Insert a block whose columns are in schema order.
    pub fn insert(&mut self, block: Block) -> Result<usize> {
        let block = self.coerce_block(block)?;
        let n = block.rows();
        if n == 0 {
            return Ok(0);
        }
        if n >= BULK_INSERT_THRESHOLD {
            // Big batch: flush whatever is buffered so ordering stays
            // newest-part-wins, then pack this batch straight into a part.
            self.flush()?;
            self.ingest_block(block)?;
            self.maybe_auto_compact()?;
            return Ok(n);
        }
        // Small batch: buffer it. Cells are written straight into the delta's
        // arena -- building a `Vec<Value>` per row here was most of the cost of
        // a single-row insert.
        let pk = self.pk_col;
        let ncols = block.width();
        self.delta.reserve(n);
        let mut row: Vec<Value> = vec![Value::Null; ncols];
        for i in 0..n {
            for (c, cell) in row.iter_mut().enumerate() {
                *cell = block.column(c).value(i);
            }
            match pk {
                Some(c) => {
                    let lane =
                        row[c].to_lane_phys(self.pk_phys.unwrap(), self.def.schema.ty(c))?;
                    self.delta.put_keyed(lane, &row)?;
                }
                None => self.delta.append(&row)?,
            }
        }
        if self.delta.len() >= self.delta_limit {
            // `flush` bounds the part count itself.
            self.flush()?;
        }
        Ok(n)
    }

    /// Write a single row, row-oriented.
    ///
    /// The OLTP counterpart to [`Table::insert`]. Going through a `Block` for
    /// one row means allocating a `Vec` per column just to read it straight
    /// back out, which dominates the cost of a point write; this writes the
    /// values directly into the delta's arena and allocates nothing.
    ///
    /// Values are coerced to the declared column types on flush, by the same
    /// `ColumnBuilder` path every other writer uses.
    pub fn put_row(&mut self, values: &[Value]) -> Result<()> {
        let ncols = self.def.schema.len();
        if values.len() != ncols {
            return Err(Error::storage(format!(
                "table `{}` has {ncols} columns, got {}",
                self.def.name,
                values.len()
            )));
        }
        match self.pk_col {
            Some(pk) => {
                let lane = values[pk]
                    .to_lane_phys(self.pk_phys.unwrap(), self.def.schema.ty(pk))?;
                self.delta.put_keyed(lane, values)?;
            }
            None => self.delta.append(values)?,
        }
        if self.delta.len() >= self.delta_limit {
            self.flush()?;
        }
        Ok(())
    }

    /// Cast incoming columns to the declared types. The binder normally does
    /// this, but direct API users and the WAL replay path come through here.
    fn coerce_block(&self, block: Block) -> Result<Block> {
        if block.width() != self.def.schema.len() {
            return Err(Error::storage(format!(
                "table `{}` has {} columns, got {}",
                self.def.name,
                self.def.schema.len(),
                block.width()
            )));
        }
        // Three ways a column can need fixing up, and the third is the subtle
        // one: `Column::new` only *debug*-asserts that a column's buffer kind
        // matches its declared type, so a release-mode caller can hand us a
        // `Column { ty: DateTime, data: I64(..) }`. Declared type and declared
        // type would agree, and the mismatch would flow into storage, where
        // packing reads the buffer kind but decoding reads the declared type —
        // silently garbling every value. Compare against the *actual* buffer.
        let needs_cast = block.columns.iter().enumerate().any(|(i, c)| {
            let want = self.def.schema.ty(i);
            &c.ty != want
                || c.ty.physical() != want.physical()
                || c.data.physical() != want.physical()
        });
        if !needs_cast {
            return Ok(block);
        }
        let mut out = Vec::with_capacity(block.width());
        for (i, c) in block.columns.iter().enumerate() {
            let want = self.def.schema.ty(i).clone();
            if c.ty == want && c.data.physical() == want.physical() {
                out.push(c.clone());
                continue;
            }
            let mut b = ColumnBuilder::with_capacity(want.clone(), c.len());
            for r in 0..c.len() {
                let v = c.value(r);
                b.push_value(&v.cast_to(&want)?)?;
            }
            out.push(b.finish());
        }
        Block::new(out)
    }

    /// Sort, dedup, tombstone shadowed rows, and append a new part.
    fn ingest_block(&mut self, block: Block) -> Result<()> {
        if block.rows() == 0 {
            return Ok(());
        }
        let sort_col = self.sort_col;
        let pk_col = self.pk_col;

        // Sort by permutation rather than by materializing a sorted copy: the
        // packer reads rows individually anyway, so it can read them permuted.
        //
        // `key_order` answers both questions in one walk of the key column:
        // whether the sort can be skipped, and -- for the batch that skips it
        // -- whether it carries duplicate keys the dedup below must collapse.
        let order = if self.def.engine.is_sorted() && !self.def.order_by.is_empty() {
            // The duplicate answer is only actionable on a keyed table: a bare
            // sort key permits repeats, so asking would cost a compare per row
            // for something that is then discarded.
            key_order(&block, &self.def.order_by, pk_col.is_some())
        } else {
            KeyOrder::Sorted
        };
        let mut perm = match order {
            KeyOrder::Unsorted => Some(sort_permutation(&block, &self.def.order_by)?),
            _ => None,
        };
        match (pk_col, perm.as_mut()) {
            (Some(pk), Some(p)) => dedup_perm_last_by_key(&block, pk, p)?,
            // The batch arrived in key order, so the sort was skipped -- and
            // the dedup used to be skipped with it, because it only ever ran
            // over a permutation. That let a table with a *declared* PRIMARY
            // KEY keep two rows with the same key, purely because the rows
            // happened to arrive sorted and the batch happened to be big
            // enough for the bulk path. The MPH index built over that part is
            // only defined on distinct keys, so the index and the data then
            // disagreed: `WHERE pk = <const>` (which the physical planner
            // lowers to an index probe) and a scan of the same predicate
            // returned different rows.
            //
            // Collapsing rather than rejecting, for one decisive reason:
            // last-write-wins is already what this engine does for the same
            // keys arriving in two statements (the newer part tombstones the
            // older row) and for the buffered path (`Delta::put_keyed`
            // overwrites the slot). Rejecting within a batch would make
            // `INSERT INTO t VALUES (1,'a'),(1,'b')` succeed or fail depending
            // on the row count and on whether the rows happened to be sorted
            // -- the same statement, two answers, decided by batching. So the
            // duplicate is collapsed, keeping the last row, which is what
            // `duplicate_keys_within_one_insert_collapse` has always asserted
            // for the sorted-input case.
            (Some(pk), None) if order == KeyOrder::SortedWithDups => {
                perm = Some(keep_last_of_sorted_runs(&block, pk));
            }
            _ => {}
        }
        // Already in key order: read the block directly rather than through an
        // indirection that would change every memcpy into a gather.
        if perm.as_deref().is_some_and(|p| p.len() == block.rows() && is_identity(p)) {
            perm = None;
        }
        let sorted = block;

        // Pack first, tombstone second, publish last. Two steps here can fail
        // -- `Part::build_sel`, and `to_lane` once per key -- and a tombstone
        // is not undoable, so neither may touch anything a reader can see.
        //
        // Building first was already the case; the loop below was not. A
        // `to_lane` error on key 5000 of a batch used to leave keys 0..5000
        // tombstoned in the older parts with their replacement part thrown
        // away -- 5000 rows that no subsequent query could see, and no error
        // to explain them, because a tombstone hides a row rather than
        // reporting it. So the tombstones go into a private clone of the set
        // and the clone is published only once the whole batch has converted.
        // (`stats` and the lazily-built bloom filters do advance on the error
        // path. Both are pure caches; neither can change a query's answer.)
        let nrows = perm.as_ref().map_or(sorted.rows(), |p| p.len());
        fail_build()?;
        let new_part = Part::build_sel(&sorted, perm.as_deref(), sort_col, pk_col)?;

        // Hide any rows in older parts that these keys replace. Newest part
        // wins, so this must happen before the new part is pushed, and the
        // whole batch lands in one copy-on-write generation.
        let Table { parts, txn, stats, def, pk_phys, .. } = self;
        let shadowing = pk_col.filter(|_| !peek(parts, txn).is_empty());
        let Some(pk) = shadowing else {
            // Nothing to shadow: no fallible step is left, so edit the
            // published set in place and skip the clone entirely. This is the
            // whole first-part-of-an-empty-table case and the no-primary-key
            // engines, neither of which should pay for atomicity they cannot
            // violate.
            let set = edit(parts, txn);
            set.push(new_part);
            set.bump();
            return Ok(());
        };

        // The private replacement. Two vectors of pointers, plus -- because
        // the clone shares each `Arc<Deletes>` with the published set -- one
        // copy-on-write of each delete mask this batch touches. Both are costs
        // `edit` already pays the instant any reader holds a snapshot, which
        // in the HTAP case is most of the time, and a bulk ingest is >=4096
        // rows so they amortize over at least that many tombstones.
        //
        // Measured interleaved against the old in-place path (temporary
        // `AtomicBool` switch, best-of-9 per side, 5 alternating runs; 8
        // shadowing bulk ingests of 8192 keys each into a 131072-row part):
        // 1.067, 0.916, 1.057, 0.989, 1.013 -- mean 1.01, inside this
        // machine's noise. The atomicity is free.
        let mut next = peek(parts, txn).clone();
        for p in next.parts() {
            p.ensure_filter();
        }
        let keys = &sorted.columns[pk];
        let ty = def.schema.ty(pk);
        // Physical kind resolved once: `to_lane` walks the `Nullable`/
        // `LowCardinality` wrappers with a recursive match, and this loop runs
        // once per ingested row. Same reason the type is borrowed rather than
        // cloned -- a `DataType` clone can allocate.
        let phys = pk_phys.expect("a pk column implies a pk physical type");
        // Two loops rather than one carrying a `perm.is_some()` test: the
        // choice is constant across the batch and the test would be per row.
        match perm.as_deref() {
            Some(p) => {
                for &r in p {
                    fail_tombstone()?;
                    tombstone_key(&mut next, keys.value(r as usize).to_lane_phys(phys, ty)?, stats);
                }
            }
            None => {
                for r in 0..nrows {
                    fail_tombstone()?;
                    tombstone_key(&mut next, keys.value(r).to_lane_phys(phys, ty)?, stats);
                }
            }
        }
        next.push(new_part);
        next.bump();
        publish(parts, txn, next);
        Ok(())
    }

    /// Fold the delta into a new part.
    ///
    /// Failure-atomic by construction: everything that can fail happens before
    /// anything observable is touched. This is not defensive tidiness -- every
    /// SELECT flushes every table before planning, so a build error here is
    /// reachable from a pure read, and the old shape (drain, tombstone, *then*
    /// build) answered such an error by having already deleted both the
    /// buffered rows and the part rows they shadowed. An `Err` out of a read
    /// must leave the table exactly as it was.
    pub fn flush(&mut self) -> Result<()> {
        if self.delta.is_empty() {
            return Ok(());
        }

        // --- build phase: fallible, mutates nothing ---------------------
        //
        // `to_block` is `drain_to_block` minus the `clear` and minus the
        // touched-key `Vec`: same two passes over the arena (live_slots, then
        // one strided transpose per column), so moving the clear later costs
        // nothing. The touched keys are read straight out of the delta's index
        // below instead of being collected up front, which is one allocation
        // *less* per flush than before, and none at all when there is nothing
        // to tombstone.
        let block = self.delta.to_block(&self.def.schema)?;
        let new_part = if block.rows() > 0 {
            let sort_col = self.sort_col;
            let pk_col = self.pk_col;
            // No dedup here, unlike `ingest_block`: a keyed table's delta is
            // keyed too (`Delta::new(def.has_fast_pk(), ..)`), so `put_keyed`
            // has already collapsed repeats into one slot per key and
            // `to_block` cannot emit two rows sharing one. That is why only
            // the bulk path ever had the duplicate-key hole.
            let mut perm = if self.def.engine.is_sorted()
                && !self.def.order_by.is_empty()
                && key_order(&block, &self.def.order_by, false) == KeyOrder::Unsorted
            {
                Some(sort_permutation(&block, &self.def.order_by)?)
            } else {
                None
            };
            if perm.as_deref().is_some_and(is_identity) {
                perm = None;
            }
            fail_build()?;
            Some(Part::build_sel(&block, perm.as_deref(), sort_col, pk_col)?)
        } else {
            // Pure deletes: no rows to pack, but the tombstones below still
            // have to be applied.
            None
        };

        // --- publish phase: infallible from here ------------------------
        //
        // Tombstone every touched key -- including pure deletes, which
        // contribute no rows to `block` but must still hide the old row. The
        // whole batch lands in one copy-on-write generation: `edit` clones the
        // affected delete masks at most once each, no matter how many keys are
        // shadowed, and no snapshot can observe the batch half applied.
        //
        // Disjoint field borrows: the key iterator holds `delta`, the
        // tombstoning holds `parts` and `stats`.
        let Table { parts, txn, stats, delta, pk_col, .. } = self;
        let set = edit(parts, txn);
        if pk_col.is_some() && !set.is_empty() {
            for p in set.parts() {
                p.ensure_filter();
            }
            for lane in delta.touched_keys() {
                tombstone_key(set, lane, stats);
            }
        }
        delta.clear();
        if let Some(p) = new_part {
            set.push(p);
        }
        set.bump();

        // Bound the part count here rather than only in the write paths.
        // A scan flushes before reading, so an interleaved read/write workload
        // -- the hybrid case this engine exists for -- creates a part per
        // query. Left unchecked that grows without bound: every point lookup
        // then probes one more bloom filter, and every scan reads one more set
        // of undersized granules.
        //
        // Recursion is not a risk: `compact` calls `flush`, which returns
        // immediately because the delta is empty by this point.
        self.maybe_auto_compact()
    }


    // ------------------------------------------------------------ point ops

    /// Look up by primary key. Only meaningful when `def.has_fast_pk()`.
    pub fn get(&mut self, key: &Value) -> Result<Option<Vec<Value>>> {
        let Some(pk) = self.pk_col else {
            return Err(Error::storage(format!(
                "table `{}` has no single-column primary key to look up by",
                self.def.name
            )));
        };
        let lane = key.to_lane(self.def.schema.ty(pk))?;
        Ok(self.get_lane(lane))
    }

    /// Find a live row by primary-key lane **without materializing it**.
    ///
    /// This is the primitive; [`Table::get_lane`] is the convenience wrapper.
    /// The distinction matters on the hot path: returning `Vec<Value>` costs a
    /// heap allocation per lookup, which at ~100ns/lookup is a large fraction
    /// of the whole operation. Callers that only need existence, or only one
    /// column, should never pay for the row.
    #[inline]
    pub fn locate(&mut self, lane: u64) -> Option<RowLoc> {
        let Table { parts, txn, stats, delta, .. } = self;
        // `peek`, not `snapshot`: no lock and no `Arc` clone on a 59 ns path.
        // Exclusive access to the table is already exclusive access to the set.
        locate_in(peek(parts, txn), delta, stats, lane)
    }

    /// Read one column of a located row. No allocation for numeric columns.
    ///
    /// `snap` must be the same view `loc` was produced against -- a `RowLoc`
    /// is an index into a part set, and a set is only meaningful as a whole.
    #[inline]
    pub fn value_at(&self, snap: &Snapshot, lane: u64, loc: RowLoc, col: usize) -> Value {
        match loc {
            RowLoc::Delta => match self.delta.get(lane) {
                Some(DeltaEntry::Put(slot)) => self.delta.value_at(slot, col),
                _ => Value::Null,
            },
            RowLoc::Part { part, pos } => snap.part(part as usize).value_at(pos as usize, col),
        }
    }

    /// Materialize a located row into `out`, reusing its allocation.
    pub fn read_row_into(&self, set: &PartSet, lane: u64, loc: RowLoc, out: &mut Vec<Value>) {
        out.clear();
        match loc {
            RowLoc::Delta => {
                if let Some(DeltaEntry::Put(slot)) = self.delta.get(lane) {
                    out.extend((0..self.delta.ncols()).map(|c| self.delta.value_at(slot, c)));
                }
            }
            RowLoc::Part { part, pos } => {
                let p = set.part(part as usize);
                out.extend((0..self.def.schema.len()).map(|c| p.value_at(pos as usize, c)));
            }
        }
    }

    /// Look up and materialize a whole row. Allocates; prefer
    /// [`Table::locate`] plus [`Table::value_at`] on a hot path.
    ///
    /// Locate and read run against one part set, taken once: splitting them
    /// across two views would let compaction retire the part between them and
    /// turn a valid `RowLoc` into a stale index.
    pub fn get_lane(&mut self, lane: u64) -> Option<Vec<Value>> {
        let ncols = self.def.schema.len();
        let Table { parts, txn, stats, delta, def, .. } = self;
        let set: &PartSet = peek(parts, txn);
        let loc = locate_in(set, delta, stats, lane)?;
        let mut out = Vec::with_capacity(ncols);
        // Inlined `read_row_into`: `self` is destructured, so the method call
        // would need a reborrow the borrow checker cannot see through.
        match loc {
            RowLoc::Delta => {
                if let Some(DeltaEntry::Put(slot)) = delta.get(lane) {
                    out.extend((0..delta.ncols()).map(|c| delta.value_at(slot, c)));
                }
            }
            RowLoc::Part { part, pos } => {
                let p = set.part(part as usize);
                out.extend((0..def.schema.len()).map(|c| p.value_at(pos as usize, c)));
            }
        }
        Some(out)
    }

    /// Batched point lookups.
    ///
    /// Per part: phase 1 computes every candidate's packed-key location and
    /// software-prefetches it, phase 2 verifies and reads. Overlapping the
    /// cache misses buys memory-level parallelism a one-at-a-time `get` can
    /// never reach -- the misses issue concurrently instead of serially.
    pub fn multi_get(&mut self, lanes: &[u64], out: &mut [Option<Vec<Value>>]) {
        assert_eq!(lanes.len(), out.len());
        let mut locs = vec![None; lanes.len()];
        self.multi_locate(lanes, &mut locs);
        // The same set the locations were resolved against, and only one lock
        // acquisition for the whole batch.
        let snap = self.snapshot();
        for (i, loc) in locs.iter().enumerate() {
            out[i] = loc.map(|l| {
                let mut row = Vec::with_capacity(self.def.schema.len());
                self.read_row_into(snap.set(), lanes[i], l, &mut row);
                row
            });
        }
    }

    /// Batched lookup that returns *locations*, allocating nothing per hit.
    ///
    /// Per part: phase 1 computes every candidate's packed-key location and
    /// software-prefetches it, phase 2 records the hits. Overlapping the cache
    /// misses buys memory-level parallelism a one-at-a-time `locate` can never
    /// reach -- the misses issue concurrently instead of serially, which is
    /// worth roughly 2x on a table that does not fit in cache.
    pub fn multi_locate(&mut self, lanes: &[u64], out: &mut [Option<RowLoc>]) {
        assert_eq!(lanes.len(), out.len());
        out.iter_mut().for_each(|o| *o = None);

        let Table { parts, txn, stats, delta, .. } = self;
        let set: &PartSet = peek(parts, txn);

        let mut pending: Vec<u32> = (0..lanes.len() as u32).collect();
        if !delta.is_empty() {
            pending.retain(|&qi| match delta.get(lanes[qi as usize]) {
                Some(DeltaEntry::Put(_)) => {
                    out[qi as usize] = Some(RowLoc::Delta);
                    false
                }
                Some(DeltaEntry::Del) => false,
                None => true,
            });
        }

        let multi = set.len() > 1;
        let mut cands: Vec<(u32, u32)> = Vec::with_capacity(pending.len());
        for pi in (0..set.len()).rev() {
            if pending.is_empty() {
                break;
            }
            let p = set.part(pi);
            // Hoisted out of the probe loop: one null check per *part*, not
            // per candidate, and a clean part never touches a delete mask.
            let del = set.deletes(pi);
            cands.clear();
            let mut next: Vec<u32> = Vec::with_capacity(pending.len());
            // phase 1: locate + prefetch
            for &qi in &pending {
                let lane = lanes[qi as usize];
                let fph = hash_key(lane, FP_SEED);
                if multi && !p.may_contain(fph) {
                    stats.bloom_negative += 1;
                    next.push(qi);
                    continue;
                }
                match p.find_live(lane, fph, stats, del) {
                    Some(pos) => {
                        // Touch only the key column: the caller decides which
                        // columns it actually wants.
                        if let Some(pk) = p.pk_col {
                            p.granules[pos >> crate::common::G_SHIFT].columns[pk]
                                .prefetch(pos & (GRANULE_SIZE - 1));
                        }
                        cands.push((qi, pos as u32));
                    }
                    None => next.push(qi),
                }
            }
            // phase 2: record (prefetched lines now resident)
            for &(qi, pos) in &cands {
                out[qi as usize] = Some(RowLoc::Part { part: pi as u32, pos });
            }
            pending = next;
        }
    }

    /// Buffer a delete by primary key.
    pub fn delete_key(&mut self, key: &Value) -> Result<()> {
        let Some(pk) = self.pk_col else {
            return Err(Error::storage("table has no single-column primary key"));
        };
        let lane = key.to_lane(self.def.schema.ty(pk))?;
        self.delta.delete_keyed(lane);
        if self.delta.len() >= self.delta_limit {
            self.flush()?;
        }
        Ok(())
    }

    /// Mark a row deleted by absolute `(part, position)`. Used by
    /// `ALTER TABLE ... DELETE WHERE`, which locates rows by scanning.
    pub fn mark_deleted(&mut self, part: usize, pos: usize) -> bool {
        let set = edit(&mut self.parts, &mut self.txn);
        if part >= set.len() {
            return false;
        }
        let hidden = set.tombstone(part, pos);
        if hidden {
            set.bump();
        }
        hidden
    }

    // ------------------------------------------------------------ bulk delete

    /// Hide every live row matching `pred`, publishing **one** new part-set
    /// version for the whole statement.
    ///
    /// `projection` names table columns; `pred` and `zone` are expressed
    /// against *that* projection. Those are the same two index spaces a
    /// [`ScanNode`](crate::planner::logical::ScanNode) carries, and they are
    /// here for the same reason `MutationPlan` wraps a plan instead of being
    /// one: a mutation runs the predicate `optimizer::optimize` produced for
    /// it -- pushed-down PREWHERE, derived zone maps, folded constants -- so
    /// there is no second copy of the access-path rules to keep in step.
    /// `pred = None` is "every live row", which is what `DELETE FROM t` binds
    /// to once the folder has dropped its constantly-true predicate; that case
    /// decodes nothing at all.
    ///
    /// The cost that matters is the *publish*, and it is O(parts): parts are
    /// immutable and the set is copy-on-write, so hiding a million rows is one
    /// delete bitmap per touched part plus one pointer store -- not a million
    /// tombstones routed through the delta, and not a part rebuild. Measured
    /// against the per-key loop it replaced, in memory, on a 200k-row keyed
    /// table: 11-25x on the whole statement, 0.095 -> 0.008 us per affected
    /// row. A/B interleaved, best of 7 per side; the table is next to
    /// `session::Session::run_alter_delete`.
    pub fn delete_where(
        &mut self,
        projection: &[usize],
        pred: Option<&BoundExpr>,
        zone: &[ZoneFilter],
    ) -> Result<usize> {
        self.delete_where_keys(projection, pred, zone, None)
    }

    /// [`Table::delete_where`], additionally collecting the primary-key lane
    /// of every row it hid.
    ///
    /// A logging session needs them: [`crate::persist::Wal::append_delete`]
    /// names a row by key lane and there is no other delete record, so the
    /// lanes are what makes the sweep replayable. They have to be the rows
    /// *newly* hidden rather than
    /// the rows the predicate matched -- a row an earlier statement already
    /// tombstoned is not part of this statement's effect, and logging it again
    /// would make the count the caller reports disagree with the log. Reading
    /// one is a packed-lane load at a position the sweep already holds, which
    /// is why it happens here instead of in a second pass.
    pub fn delete_where_keys(
        &mut self,
        projection: &[usize],
        pred: Option<&BoundExpr>,
        zone: &[ZoneFilter],
        mut keys: Option<&mut Vec<u64>>,
    ) -> Result<usize> {
        // Buffered rows are not in any part, and a position is only meaningful
        // inside one. Free when the caller already planned a query.
        self.flush()?;

        // Cloned and published rather than edited in place. `read_columns` and
        // the predicate are both fallible, and `edit` hands out the live set --
        // so an error half way through would leave the tombstones written
        // before it visible, which is exactly the guarantee `publish` exists to
        // restore. The clone is two `Vec`s of pointers: O(parts).
        let mut set = peek(&mut self.parts, &self.txn).clone();
        let pk = keys.as_ref().and(self.pk_col);
        let mut sel: Vec<u32> = Vec::new();
        let mut hidden = 0usize;

        for pi in 0..set.len() {
            // One refcount bump per part detaches it from `set`'s borrow, so
            // the sweep can tombstone into the same set it is reading without
            // re-resolving the part per row.
            let p = Arc::clone(&set.parts()[pi]);
            for (gi, g) in p.granules.iter().enumerate() {
                if zone_prunes(g, projection, zone) {
                    continue;
                }
                let live = p.live_selection_into(gi, set.deletes(pi), &mut sel);
                let rows = live.map_or(g.len, |s| s.len());
                if rows == 0 {
                    continue;
                }
                let base = gi << G_SHIFT;
                let lanes = pk.map(|c| &g.columns[c]);
                // Borrows `set` and `keys` for the rest of the granule; the
                // predicate below borrows neither, so the two never collide.
                let mut hit = |off: usize| -> usize {
                    if !set.tombstone(pi, base + off) {
                        return 0;
                    }
                    if let (Some(k), Some(c)) = (keys.as_deref_mut(), lanes) {
                        k.push(c.lane(off));
                    }
                    1
                };

                let Some(e) = pred else {
                    // No predicate: every live row goes, and not one byte of
                    // column data is decoded to decide it.
                    match live {
                        None => (0..rows).for_each(|off| hidden += hit(off)),
                        Some(s) => s.iter().for_each(|&off| hidden += hit(off as usize)),
                    }
                    continue;
                };

                let blk = p.read_columns(gi, projection, live)?;
                let mask = expr::eval(e, &blk)?;
                let rows = rows.min(mask.len());
                // `expr::eval` + this walk rather than `expr::eval_predicate`:
                // the latter materialises a `Vec<u32>` of survivors *per
                // granule* -- one allocation per 1024 rows -- and then this
                // loop would walk it a second time. Matching on the column
                // kind once per granule keeps the dispatch out of the row loop
                // either way, which is the only thing the selection vector was
                // buying. SQL three-valued logic, as in `eval_predicate`: a
                // NULL fails a filter exactly as FALSE does, so a row that is
                // not provably TRUE is not deleted.
                macro_rules! sweep {
                    ($keep:expr) => {{
                        let keep = $keep;
                        match live {
                            None => {
                                for i in 0..rows {
                                    if keep(i) {
                                        hidden += hit(i);
                                    }
                                }
                            }
                            Some(s) => {
                                for i in 0..rows {
                                    if keep(i) {
                                        hidden += hit(s[i] as usize);
                                    }
                                }
                            }
                        }
                    }};
                }
                match (&mask.data, &mask.nulls) {
                    (ColumnData::U64(v), None) => sweep!(|i: usize| v[i] != 0),
                    (ColumnData::U64(v), Some(m)) => sweep!(|i: usize| !m.get(i) && v[i] != 0),
                    (ColumnData::I64(v), None) => sweep!(|i: usize| v[i] != 0),
                    (ColumnData::I64(v), Some(m)) => sweep!(|i: usize| !m.get(i) && v[i] != 0),
                    (ColumnData::F64(v), None) => sweep!(|i: usize| v[i] != 0.0),
                    (ColumnData::F64(v), Some(m)) => sweep!(|i: usize| !m.get(i) && v[i] != 0.0),
                    (ColumnData::Str(v), None) => sweep!(|i: usize| !v[i].is_empty()),
                    (ColumnData::Str(v), Some(m)) => {
                        sweep!(|i: usize| !m.get(i) && !v[i].is_empty())
                    }
                }
            }
        }

        // Publish unconditionally: a version bump costs nothing and a caller
        // that matched no row still deserves a set whose version says the
        // statement ran, so a cached plan cannot decide it did not.
        set.bump();
        publish(&mut self.parts, &mut self.txn, set);
        Ok(hidden)
    }

    /// Total live rows, counting buffered writes and their shadowing.
    pub fn row_count(&mut self) -> Result<usize> {
        self.flush()?;
        Ok(peek(&mut self.parts, &self.txn).live_rows())
    }

    // ----------------------------------------------------------- compaction

    /// Merge every part into one, dropping deleted rows.
    ///
    /// Parts are already sorted and live keys are disjoint across them (each
    /// ingest tombstones what it replaces), so this is a k-way heap merge over
    /// live-row cursors: `O(N log P)` with no re-sort.
    pub fn compact(&mut self) -> Result<()> {
        self.flush()?;
        let set = peek(&mut self.parts, &self.txn);
        if set.len() <= 1 && (set.is_empty() || set.deletes(0).is_none()) {
            return Ok(());
        }
        let all: Vec<usize> = (0..set.len()).collect();
        self.merge_parts(&all)
    }

    /// Merge the parts at `idxs` into one, dropping deleted rows.
    ///
    /// Any subset may be merged, not just a contiguous or newest-first one:
    /// each ingest tombstones the keys it replaces in older parts, so a live
    /// key exists in exactly one part and merging is order-independent.
    ///
    /// Parts are already sorted, so this is a k-way heap merge over live-row
    /// cursors: `O(N log P)` with no re-sort.
    fn merge_parts(&mut self, idxs: &[usize]) -> Result<()> {
        let set = peek(&mut self.parts, &self.txn);
        if idxs.len() < 2 && !idxs.iter().any(|&i| set.deletes(i).is_some()) {
            return Ok(());
        }
        let picked: Vec<&Part> = idxs.iter().map(|&i| set.part(i)).collect();
        let dels: Vec<Option<&Deletes>> = idxs.iter().map(|&i| set.deletes(i)).collect();

        let order: Vec<(u32, u32)> = if self.sort_col.is_some() {
            let mut heap: BinaryHeap<Reverse<(u64, u32, u32)>> = BinaryHeap::new();
            for (k, p) in picked.iter().enumerate() {
                if let Some(pos) = p.next_live(0, dels[k]) {
                    heap.push(Reverse((p.sort_lane_at(pos), k as u32, pos as u32)));
                }
            }
            let total: usize = idxs.iter().map(|&i| set.live_rows_of(i)).sum();
            let mut out = Vec::with_capacity(total);
            while let Some(Reverse((_, k, pos))) = heap.pop() {
                out.push((k, pos));
                let p = picked[k as usize];
                if let Some(np) = p.next_live(pos as usize + 1, dels[k as usize]) {
                    heap.push(Reverse((p.sort_lane_at(np), k, np as u32)));
                }
            }
            out
        } else {
            // Unsorted engine: concatenation preserves insert order. One
            // reused position buffer rather than a fresh `Vec` per part.
            let mut out = Vec::new();
            let mut pos = Vec::new();
            for (k, p) in picked.iter().enumerate() {
                pos.clear();
                p.live_positions_into(dels[k], &mut pos);
                out.extend(pos.iter().map(|&r| (k as u32, r as u32)));
            }
            out
        };

        let block = gather_rows(&picked, &order, &self.def.schema)?;
        drop(picked);
        drop(dels);

        // Build the replacement before unlinking the inputs. `Part::build` is
        // fallible and dropping a part is not undoable, so removing first
        // would turn a packing error into permanent loss of every row in the
        // merged set -- and auto-compaction runs off `flush`, which runs off
        // every SELECT, so that error is reachable from a read.
        fail_build()?;
        let merged = if block.rows() > 0 {
            Some(Part::build(&block, self.sort_col, self.pk_col)?)
        } else {
            None
        };

        // Publish: one new generation of the set, dropping the merged parts
        // (highest index first so the rest stay valid) and appending the
        // result. A reader that snapshotted before this keeps its `Arc`s, and
        // the part files the caller is about to unlink stay mapped under it.
        let set = edit(&mut self.parts, &mut self.txn);
        let mut sorted = idxs.to_vec();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        for i in sorted {
            set.remove(i);
        }
        if let Some(p) = merged {
            set.push(p);
        }
        set.bump();
        Ok(())
    }

    /// Keep the part count bounded without rewriting the whole table.
    ///
    /// A flat "merge everything once there are N parts" policy makes every
    /// Nth flush cost `O(total rows)`, and in an interleaved read/write
    /// workload a *query* triggers that flush -- so one unlucky query in
    /// sixteen stalls for the length of a full table rewrite. Instead this
    /// merges only the small parts, which is where the part count actually
    /// comes from: a scan flushing a handful of buffered rows.
    ///
    /// Parts are taken smallest-first while they stay small relative to the
    /// table, so a large compacted part is left alone and merge cost stays
    /// proportional to the churn rather than to the data.
    fn maybe_auto_compact(&mut self) -> Result<()> {
        let set = peek(&mut self.parts, &self.txn);
        if set.len() < AUTO_COMPACT_PARTS {
            return Ok(());
        }
        let total: usize = set.live_rows();
        let mut by_size: Vec<usize> = (0..set.len()).collect();
        by_size.sort_unstable_by_key(|&i| set.live_rows_of(i));

        // Take the smallest parts while their running total stays under a
        // quarter of the table. Always take at least half the parts, so the
        // count genuinely comes down even when the sizes are uniform.
        let budget = (total / 4).max(1);
        let min_take = AUTO_COMPACT_PARTS / 2;
        let mut take = Vec::new();
        let mut acc = 0usize;
        for &i in &by_size {
            let rows = set.live_rows_of(i);
            if take.len() >= min_take && acc + rows > budget {
                break;
            }
            acc += rows;
            take.push(i);
        }
        if take.len() < 2 {
            return Ok(());
        }
        self.merge_parts(&take)
    }

    // ---------------------------------------------------------------- scans

    /// Read every live row of a set of columns, in `BLOCK_SIZE` batches.
    ///
    /// This is the simple path; the executor's scan operator drives granules
    /// directly so it can apply zone-map pruning first.
    /// Stream every live row of `cols` through `f`, in batches.
    ///
    /// This is the scan the engine actually wants. `scan` materializes the
    /// whole result, which for a full-table aggregate means writing the entire
    /// column to the heap and reading it straight back -- twice the memory
    /// traffic of the packed data itself, and none of it cache-resident. Here
    /// one batch buffer is filled, handed to `f` while it is still hot in L2,
    /// and refilled in place, so a `SUM` over a billion rows allocates a fixed
    /// 64 KB regardless of table size.
    pub fn scan_each<F>(&mut self, cols: &[usize], f: F) -> Result<()>
    where
        F: FnMut(&Block) -> Result<()>,
    {
        self.flush()?;
        let snap = self.snapshot();
        self.scan_each_in(&snap, cols, f)
    }

    /// [`Table::scan_each`] over an already-pinned view.
    ///
    /// `&self`, and the snapshot is the caller's: this is the shape a
    /// concurrent reader wants, and the only thing still forcing the `&mut`
    /// wrapper above is the flush-before-read policy.
    pub fn scan_each_in<F>(&self, snap: &Snapshot, cols: &[usize], mut f: F) -> Result<()>
    where
        F: FnMut(&Block) -> Result<()>,
    {
        let batch_rows = scan_batch_rows(cols);
        let mut sc = ScanScratch::new(&self.def.schema, cols)?;
        let mut rows = 0usize;

        for pi in 0..snap.len() {
            let (p, del) = (snap.part(pi), snap.deletes(pi));
            for gi in 0..p.granule_count() {
                rows += sc.read_granule(p, gi, del, cols);
                sc.batch.set_rows(rows);
                if rows >= batch_rows {
                    f(&sc.batch)?;
                    sc.batch.clear();
                    rows = 0;
                }
            }
        }
        if rows > 0 {
            f(&sc.batch)?;
        }
        Ok(())
    }

    /// Fold every live row of `cols` **in parallel**, map-reduce style.
    ///
    /// Granules are independent -- separately packed, separately indexed -- so
    /// a scan is embarrassingly parallel and always has been; nothing but the
    /// API stopped it. Each worker folds into its own accumulator and the
    /// results merge at the end, which is the same shape
    /// [`crate::exec::functions::Accumulator::merge`] already exists for.
    ///
    /// Work is claimed from a shared counter in small batches rather than
    /// statically partitioned: granules cost wildly different amounts once
    /// zone maps and filters are involved, so a static split leaves threads
    /// idle at the end of every query.
    ///
    /// Memory stays bounded by the thread count, not the data: each worker
    /// keeps one L1-sized batch and one scratch buffer, and nothing
    /// accumulates across granules.
    pub fn scan_fold<T, I, F, M>(
        &mut self,
        cols: &[usize],
        init: I,
        fold: F,
        merge: M,
    ) -> Result<T>
    where
        T: Send,
        I: Fn() -> T + Sync,
        F: Fn(&mut T, &Block) -> Result<()> + Sync,
        M: Fn(T, T) -> T,
    {
        self.flush()?;
        let snap = self.snapshot();
        self.scan_fold_in(&snap, cols, init, fold, merge)
    }

    /// [`Table::scan_fold`] over an already-pinned view.
    ///
    /// The snapshot is taken once, by the caller, and borrowed by every
    /// worker. Nothing in the claim loop touches an `Arc` refcount or a lock:
    /// at 10 G rows/s a single atomic per granule would be visible.
    pub fn scan_fold_in<T, I, F, M>(
        &self,
        snap: &Snapshot,
        cols: &[usize],
        init: I,
        fold: F,
        merge: M,
    ) -> Result<T>
    where
        T: Send,
        I: Fn() -> T + Sync,
        F: Fn(&mut T, &Block) -> Result<()> + Sync,
        M: Fn(T, T) -> T,
    {
        // (part, granule) pairs, so work is claimable across part boundaries.
        //
        // Rejected, measured: replacing this with a per-part prefix sum
        // (`nparts + 1` entries instead of one per granule) and resolving a
        // claim back to `(part, granule)` arithmetically. It is O(nparts)
        // instead of O(ngranules) and allocates 78 KB less on a 10M-row
        // table, but the list costs 1.5 us to build for 4883 granules and
        // 2.9 us for 9766 -- 0.5-0.6% of the scans that use it, well under
        // this machine's ~5% noise floor -- against real complexity in the
        // hottest loop in the engine. Not worth it; do not retry without a
        // table two orders of magnitude larger.
        let work: Vec<(u32, u32)> = (0..snap.len())
            .flat_map(|pi| {
                (0..snap.part(pi).granule_count()).map(move |g| (pi as u32, g as u32))
            })
            .collect();

        // One participant per SCAN_CLAIM-sized chunk at most: waking a thread
        // to hand it eight granules and nothing else costs more than it saves.
        let threads = pool::global()
            .threads()
            .min(work.len().div_ceil(SCAN_CLAIM).max(1));
        if threads <= 1 {
            let mut acc = init();
            let mut sc = ScanScratch::new(&self.def.schema, cols)?;
            sc.fold_range(snap, cols, &work, &fold, &mut acc)?;
            return Ok(acc);
        }

        let schema = &self.def.schema;
        let cursor = std::sync::atomic::AtomicUsize::new(0);
        let (init, fold, work) = (&init, &fold, &work);

        // The pool rather than `thread::scope`. Each participant runs the same
        // claim loop either way -- the work stealing is in the shared cursor,
        // not in how the threads were obtained -- but spawning fourteen
        // threads costs the same whether the scan is 10M rows or 200k, and a
        // 200k-row scan is over before the spawns finish. Measured
        // interleaved: 1.05x on a 10M-row scan, 2.47x on a 200k-row one. The
        // second number is the HTAP case, and it is why this is not scoped
        // threads.
        let results: Vec<Result<T>> = pool::global().map(threads, |_| -> Result<T> {
            let mut acc = init();
            let mut sc = ScanScratch::new(schema, cols)?;
            loop {
                let at = cursor.fetch_add(SCAN_CLAIM, std::sync::atomic::Ordering::Relaxed);
                if at >= work.len() {
                    break;
                }
                let end = (at + SCAN_CLAIM).min(work.len());
                sc.fold_range(snap, cols, &work[at..end], fold, &mut acc)?;
            }
            Ok(acc)
        });

        let mut out: Option<T> = None;
        for r in results {
            let v = r?;
            out = Some(match out.take() {
                None => v,
                Some(prev) => merge(prev, v),
            });
        }
        Ok(out.unwrap_or_else(&init))
    }

    /// Read every live row of `cols` into blocks.
    ///
    /// Materializing the whole result costs memory proportional to the table;
    /// prefer [`Table::scan_each`] on anything large.
    pub fn scan(&mut self, cols: &[usize]) -> Result<Vec<Block>> {
        let mut out = Vec::new();
        let mut any = false;
        self.scan_each(cols, |b| {
            any = true;
            out.push(b.clone());
            Ok(())
        })?;
        if !any {
            // Preserve the column shape for an empty table.
            out.push(Block::new(
                cols.iter()
                    .map(|&c| {
                        let ty = self.def.schema.ty(c).clone();
                        Column::new(ty.clone(), ColumnData::for_physical(ty.physical()))
                    })
                    .collect(),
            )?);
        }
        Ok(out)
    }

    // -------------------------------------------------------- introspection

    pub fn data_bytes(&self) -> usize {
        self.snapshot().parts().iter().map(|p| p.data_bytes()).sum()
    }
    pub fn index_bytes(&self) -> usize {
        let snap = self.snapshot();
        snap.parts().iter().map(|p| p.index_bytes()).sum::<usize>() + snap.set().delete_bytes()
    }
    pub fn stored_rows(&self) -> usize {
        self.snapshot().parts().iter().map(|p| p.n_rows).sum()
    }

    /// Uncompressed size if every column were stored at its declared width.
    pub fn raw_bytes(&self) -> usize {
        let per_row: usize = self
            .def
            .schema
            .fields()
            .iter()
            .map(|f| declared_width(&f.ty))
            .sum();
        self.stored_rows() * per_row
    }

    /// Per-column compression, the number this engine exists to make small.
    pub fn compression_report(&self) -> CompressionReport {
        let snap = self.snapshot();
        let rows = self.stored_rows().max(1);
        let mut cols = Vec::new();
        for (ci, f) in self.def.schema.fields().iter().enumerate() {
            let bytes: usize = snap
                .parts()
                .iter()
                .flat_map(|p| p.granules.iter())
                .map(|g| g.columns[ci].data_bytes())
                .sum();
            cols.push(ColumnCompression {
                name: f.name.clone(),
                ty: f.ty.to_string(),
                declared_bits: declared_width(&f.ty) as f64 * 8.0,
                stored_bits: bytes as f64 * 8.0 / rows as f64,
                bytes,
            });
        }
        CompressionReport {
            rows: self.stored_rows(),
            raw_bytes: self.raw_bytes(),
            data_bytes: self.data_bytes(),
            index_bytes: self.index_bytes(),
            columns: cols,
        }
    }
}

/// Everything one scan participant reuses across every granule it touches.
///
/// Bundled rather than three loose locals because the third buffer is new:
/// `sel` replaces the `Vec<u32>` the old `live_selection` allocated *per
/// granule*, i.e. one allocation per 1024 rows on any table with a single
/// tombstone in it. Memory here is bounded by the thread count, not the data.
struct ScanScratch {
    batch: Block,
    /// Decode workspace for the packed-lane reader.
    scratch: Vec<u64>,
    /// Live-row selection of the granule being read.
    sel: Vec<u32>,
}

/// Rows a scan batch holds before it is handed on.
///
/// Sized so the whole batch -- every projected column -- stays in L1 while the
/// consumer walks it. That is the entire trick behind a scan running at memory
/// bandwidth: decode a slab, consume it while it is hot, refill in place. A
/// batch that spills to L2 costs measurably more than the extra per-batch call
/// overhead saves. Never below one granule (the decode unit) and never above
/// `BLOCK_SIZE` (the executor's unit).
#[inline]
fn scan_batch_rows(cols: &[usize]) -> usize {
    (SCAN_L1_BUDGET / (cols.len().max(1) * 8)).clamp(GRANULE_SIZE, BLOCK_SIZE)
}

impl ScanScratch {
    fn new(schema: &Schema, cols: &[usize]) -> Result<ScanScratch> {
        let rows = scan_batch_rows(cols);
        let batch = Block::new(
            cols.iter()
                .map(|&c| {
                    let ty = schema.ty(c).clone();
                    let mut col = Column::new(ty.clone(), ColumnData::for_physical(ty.physical()));
                    col.reserve(rows + GRANULE_SIZE);
                    col
                })
                .collect(),
        )?;
        Ok(ScanScratch {
            batch,
            scratch: Vec::with_capacity(GRANULE_SIZE),
            sel: Vec::new(),
        })
    }

    /// Append one granule's live rows to the batch; returns the rows added.
    #[inline]
    fn read_granule(
        &mut self,
        p: &Part,
        gi: usize,
        del: Option<&Deletes>,
        cols: &[usize],
    ) -> usize {
        let g = &p.granules[gi];
        if g.len == 0 {
            return 0;
        }
        // `None` here is both "no deletes anywhere in this part" and "none in
        // this granule", and neither costs a bitmap probe.
        let Some(sel) = p.live_selection_into(gi, del, &mut self.sel) else {
            for (k, &c) in cols.iter().enumerate() {
                g.columns[c].decode_append(0, g.len, &mut self.batch.columns[k], &mut self.scratch);
            }
            return g.len;
        };
        if sel.is_empty() {
            return 0;
        }
        for (k, &c) in cols.iter().enumerate() {
            g.columns[c].gather_append(sel, &mut self.batch.columns[k]);
        }
        sel.len()
    }

    /// Serial fold over an explicit work list: the single-threaded path, and
    /// the body each parallel worker runs on the range it claimed.
    fn fold_range<T, F>(
        &mut self,
        snap: &Snapshot,
        cols: &[usize],
        work: &[(u32, u32)],
        fold: &F,
        acc: &mut T,
    ) -> Result<()>
    where
        F: Fn(&mut T, &Block) -> Result<()>,
    {
        // The part and its delete mask are re-fetched only when the work list
        // crosses a part boundary, which for SCAN_CLAIM-sized claims is at
        // most once per claim.
        let Some(&(first, _)) = work.first() else { return Ok(()) };
        let mut cur = first;
        let mut p = snap.part(first as usize);
        let mut del = snap.deletes(first as usize);
        for &(pi, gi) in work {
            if pi != cur {
                cur = pi;
                p = snap.part(pi as usize);
                del = snap.deletes(pi as usize);
            }
            let rows = self.read_granule(p, gi as usize, del, cols);
            if rows > 0 {
                self.batch.set_rows(rows);
                fold(acc, &self.batch)?;
                self.batch.clear();
            }
        }
        Ok(())
    }
}

/// Hide the newest live row holding `lane`, if any part still has one.
///
/// Newest part wins, so this walks oldest-first and stops at the first live
/// hit -- a key lives in exactly one part by construction, and a row already
/// tombstoned means an older duplicate that a previous generation retired.
#[inline]
fn tombstone_key(set: &mut PartSet, lane: u64, stats: &mut Stats) {
    let fph = hash_key(lane, FP_SEED);
    for pi in 0..set.len() {
        if !set.part(pi).may_contain(fph) {
            stats.bloom_negative += 1;
            continue;
        }
        // Resolved into a local so the immutable borrow of `set` ends before
        // the mutable one begins.
        let hit = set.part(pi).find_live(lane, fph, stats, set.deletes(pi));
        if let Some(pos) = hit {
            set.tombstone(pi, pos);
            return;
        }
    }
}

/// Newest-first search for a live row, delta included.
#[inline]
fn locate_in(set: &PartSet, delta: &Delta, stats: &mut Stats, lane: u64) -> Option<RowLoc> {
    if !delta.is_empty() {
        match delta.get(lane) {
            Some(DeltaEntry::Put(_)) => return Some(RowLoc::Delta),
            Some(DeltaEntry::Del) => return None,
            None => {}
        }
    }
    let fph = hash_key(lane, FP_SEED);
    let multi = set.len() > 1;
    for pi in (0..set.len()).rev() {
        let p = set.part(pi);
        if multi && !p.may_contain(fph) {
            stats.bloom_negative += 1;
            continue;
        }
        if let Some(pos) = p.find_live(lane, fph, stats, set.deletes(pi)) {
            return Some(RowLoc::Part { part: pi as u32, pos: pos as u32 });
        }
    }
    None
}

fn declared_width(ty: &DataType) -> usize {
    match ty.base() {
        DataType::UInt8 | DataType::Int8 | DataType::Bool => 1,
        DataType::UInt16 | DataType::Int16 => 2,
        DataType::UInt32 | DataType::Int32 | DataType::Float32 | DataType::Date => 4,
        DataType::UInt64 | DataType::Int64 | DataType::Float64 | DataType::DateTime => 8,
        DataType::FixedString(n) => *n as usize,
        // A rough stand-in for "what a row-store would spend": pointer + a
        // short average payload.
        DataType::String => 24,
        _ => 8,
    }
}

#[derive(Debug, Clone)]
pub struct ColumnCompression {
    pub name: String,
    pub ty: String,
    pub declared_bits: f64,
    pub stored_bits: f64,
    pub bytes: usize,
}

#[derive(Debug, Clone)]
pub struct CompressionReport {
    pub rows: usize,
    pub raw_bytes: usize,
    pub data_bytes: usize,
    pub index_bytes: usize,
    pub columns: Vec<ColumnCompression>,
}

impl CompressionReport {
    pub fn ratio(&self) -> f64 {
        if self.data_bytes + self.index_bytes == 0 {
            return 0.0;
        }
        self.raw_bytes as f64 / (self.data_bytes + self.index_bytes) as f64
    }
}

impl std::fmt::Display for CompressionReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} rows", self.rows)?;
        for c in &self.columns {
            writeln!(
                f,
                "  {:<20} {:<24} {:>6.1} -> {:>6.2} bits/row",
                c.name, c.ty, c.declared_bits, c.stored_bits
            )?;
        }
        write!(
            f,
            "  {:.2} MB raw -> {:.2} MB packed (+{:.2} MB index) = {:.2}x smaller",
            self.raw_bytes as f64 / 1e6,
            self.data_bytes as f64 / 1e6,
            self.index_bytes as f64 / 1e6,
            self.ratio()
        )
    }
}

// ------------------------------------------------------------------ sorting

/// Row permutation that sorts `block` by `cols`, most significant first.
///
/// Picks the cheapest correct strategy: radix over order-preserving lanes when
/// every sort column is a non-nullable non-string, comparison sort otherwise.
pub fn sort_permutation(block: &Block, cols: &[usize]) -> Result<Vec<u32>> {
    let n = block.rows();
    if cols.is_empty() || n <= 1 {
        return Ok((0..n as u32).collect());
    }
    let radix_ok = cols.iter().all(|&c| {
        let col = block.column(c);
        !col.has_nulls() && col.ty.physical() != PhysicalType::Str
    });

    if radix_ok {
        if cols.len() == 1 {
            // Build the sort pairs straight off the column's own slice. An
            // earlier version called `lanes_of`, which *clones* the key column
            // first -- 80 MB of pure copy on a 10M-row sort, for data we only
            // ever read.
            use crate::common::{f64_to_lane, i64_to_lane};
            let col = block.column(cols[0]);
            // Parallel arrays, not `Vec<(u64, u32)>`: the pair pads to 16
            // bytes for 12 bytes of data, and the sort needs a scratch copy of
            // whatever shape it is given. The index array also *is* the
            // permutation, so there is no final rebuild either.
            let mut keys: Vec<u64> = match &col.data {
                ColumnData::U64(v) => v.clone(),
                ColumnData::I64(v) => v.iter().map(|&x| i64_to_lane(x)).collect(),
                ColumnData::F64(v) => v.iter().map(|&x| f64_to_lane(x)).collect(),
                ColumnData::Str(_) => {
                    return Err(Error::storage("string columns have no global sort lane"))
                }
            };
            let mut idx: Vec<u32> = (0..n as u32).collect();
            radix_sort_soa(&mut keys, &mut idx);
            drop(keys);
            return Ok(idx);
        }
        let mut all: Vec<Vec<u64>> = Vec::with_capacity(cols.len());
        for &c in cols {
            all.push(lanes_of(block, c)?);
        }
        let mut rows: Vec<(Vec<u64>, u32)> = (0..n)
            .map(|i| (all.iter().map(|v| v[i]).collect(), i as u32))
            .collect();
        radix_sort_composite(&mut rows, cols.len());
        return Ok(rows.into_iter().map(|(_, i)| i).collect());
    }

    // Comparison fallback: nullable or string keys. `Value`'s ordering already
    // puts NULL first, matching ascending-order semantics.
    let mut idx: Vec<u32> = (0..n as u32).collect();
    let keys: Vec<Vec<Value>> = (0..n)
        .map(|i| cols.iter().map(|&c| block.column(c).value(i)).collect())
        .collect();
    idx.sort_by(|&a, &b| keys[a as usize].cmp(&keys[b as usize]));
    Ok(idx)
}

fn lanes_of(block: &Block, c: usize) -> Result<Vec<u64>> {
    use crate::common::{f64_to_lane, i64_to_lane};
    let col = block.column(c);
    Ok(match &col.data {
        ColumnData::U64(v) => v.clone(),
        ColumnData::I64(v) => v.iter().map(|&x| i64_to_lane(x)).collect(),
        ColumnData::F64(v) => v.iter().map(|&x| f64_to_lane(x)).collect(),
        ColumnData::Str(_) => {
            return Err(Error::storage("string columns have no global sort lane"))
        }
    })
}

/// Keep only the last row for each key. Input must be sorted by that key.
///
/// A single INSERT may legitimately mention the same key twice; the MPH index
/// requires distinct keys, and last-write-wins is the semantics a keyed table
/// promises.
/// How `block` sits relative to the key `cols`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum KeyOrder {
    /// Already ordered, with no two adjacent rows sharing a leading key.
    Sorted,
    /// Already ordered, but the leading key repeats somewhere.
    SortedWithDups,
    Unsorted,
}

/// Classify `block` against the key `cols`, in one pass and no allocation.
///
/// Two answers rather than one, because the caller that needs both would
/// otherwise walk the key column twice. The first -- "is it already sorted" --
/// is what lets time-series ingest and anything replayed out of a sorted part
/// skip building the `(key, row)` pair array and its radix scratch buffer,
/// which on a ten-million-row load is hundreds of megabytes never allocated
/// rather than allocated and freed. The second exists because skipping the
/// sort used to skip the *dedup* with it (see `ingest_block`).
///
/// `want_dups` is not politeness, it is the measurement. Tracking duplicates
/// costs an extra compare per row, so the callers that cannot use the answer
/// must not pay it. It is a const generic below rather than a runtime flag, so
/// `false` compiles to exactly the loop that existed before this function did,
/// and the one branch that chooses between them is per *batch*.
///
/// Measured interleaved against the old `is_already_sorted` (temporary copy of
/// it in the tree, best-of-9 per side, alternating, 400k sorted `u64` keys):
/// `want_dups = false` 0.997-1.009x -- it is the same code -- and
/// `want_dups = true` 1.24-1.30x, i.e. +0.07 ns/row. The only caller that asks
/// for `true` is a bulk ingest into a keyed table, which measures 35.6 ns/row
/// end to end, so the duplicate check that closes task #28 costs 0.2% of it.
///
/// (Rejected, measured: keeping strictly-increasing as the loop's
/// fall-through and paying the second compare only on a tie. It reads like the
/// cheaper shape and is 2.13x slower -- the `continue` stops LLVM turning the
/// walk into a straight-line compare chain. Do not retry.)
///
/// `SortedWithDups` reports the leading key column only. That is exactly the
/// question `ingest_block` asks, because a primary key is always a single
/// column and always `order_by[0]` -- `TableDef::pk_col` returns `Some` only
/// when the key is `[c]` and `sort_col() == Some(c)`, and `sort_col` is
/// `order_by[0]`. Sorted by `order_by` therefore implies sorted by the primary
/// key, so its duplicates are adjacent and this finds all of them.
fn key_order(block: &Block, cols: &[usize], want_dups: bool) -> KeyOrder {
    let n = block.rows();
    if n <= 1 || cols.is_empty() {
        return KeyOrder::Sorted;
    }
    // Monomorphised per buffer kind so the inner loop is a flat slice walk on
    // a concrete type, not a `Value` comparison.
    //
    // `!(a <= b)` rather than `a > b`, which is not the same predicate for
    // floats: a NaN key answers false to both, and only the first spelling
    // reports it as unsorted. It has to, because the radix path orders floats
    // by `f64_to_lane`, which sorts NaN last -- this is what sends a block
    // containing one down that path instead of declaring it ordered.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn scan<T: PartialOrd, const DUPS: bool>(v: &[T]) -> KeyOrder {
        let mut dup = false;
        for w in v.windows(2) {
            if !(w[0] <= w[1]) {
                return KeyOrder::Unsorted;
            }
            if DUPS {
                dup |= w[0] == w[1];
            }
        }
        if dup { KeyOrder::SortedWithDups } else { KeyOrder::Sorted }
    }
    if cols.len() == 1 {
        let c = block.column(cols[0]);
        if c.has_nulls() {
            return KeyOrder::Unsorted;
        }
        return match &c.data {
            ColumnData::U64(v) => {
                if want_dups { scan::<_, true>(v) } else { scan::<_, false>(v) }
            }
            ColumnData::I64(v) => {
                if want_dups { scan::<_, true>(v) } else { scan::<_, false>(v) }
            }
            ColumnData::F64(v) => {
                if want_dups { scan::<_, true>(v) } else { scan::<_, false>(v) }
            }
            ColumnData::Str(v) => {
                if want_dups { scan::<_, true>(v) } else { scan::<_, false>(v) }
            }
        };
    }
    // Composite keys: compare row-wise. Still allocation-free, and the
    // leading-column answer falls straight out of the first inner step rather
    // than costing a second `Value` comparison -- if `cols[0]` compares `Less`
    // the loop breaks with `lead_eq` false, and any other outcome that does
    // not return has already established the tie.
    let mut dup = false;
    for i in 0..n - 1 {
        let mut lead_eq = false;
        for (j, &c) in cols.iter().enumerate() {
            let col = block.column(c);
            match col.value(i).cmp(&col.value(i + 1)) {
                std::cmp::Ordering::Less => break,
                std::cmp::Ordering::Greater => return KeyOrder::Unsorted,
                std::cmp::Ordering::Equal => lead_eq |= j == 0,
            }
        }
        dup |= lead_eq;
    }
    if dup { KeyOrder::SortedWithDups } else { KeyOrder::Sorted }
}

/// Positions of the last row of each run of equal keys, for a block that is
/// **already** in key order.
///
/// The counterpart to [`dedup_perm_last_by_key`] for the batch that skipped
/// the sort: same last-write-wins rule, one pass, and only reached when
/// [`key_order`] has already proved there is something to collapse -- so a
/// duplicate-free insert, which is nearly all of them, never allocates here.
///
/// A NaN key never arrives here: `key_order` answers `!(NaN <= NaN)` and sends
/// the block to the radix sort, where [`dedup_perm_last_by_key`] collapses it
/// by lane. This path only ever sees keys that compare.
fn keep_last_of_sorted_runs(block: &Block, key_col: usize) -> Vec<u32> {
    let n = block.rows();
    let mut out: Vec<u32> = Vec::with_capacity(n);
    // Same reason as `dedup_perm_last_by_key`: dispatch on the buffer kind
    // once, walk a concrete slice after that.
    fn run_ends<T: SameKey>(v: &[T], out: &mut Vec<u32>) {
        for i in 0..v.len() {
            if i + 1 == v.len() || !v[i].same(&v[i + 1]) {
                out.push(i as u32);
            }
        }
    }
    match &block.column(key_col).data {
        ColumnData::U64(v) => run_ends(v, &mut out),
        ColumnData::I64(v) => run_ends(v, &mut out),
        ColumnData::F64(v) => run_ends(v, &mut out),
        ColumnData::Str(v) => run_ends(v, &mut out),
    }
    out
}

/// True when a permutation reorders nothing.
///
/// Worth one linear pass to find out. Reading a granule through a permutation
/// turns a `memcpy` into per-element indexed loads, which is several times
/// slower and does not vectorize; when rows already arrive in key order -- the
/// normal case for time-series ingest, and for anything replayed out of a
/// sorted part -- skipping the indirection entirely is a large win.
#[inline]
fn is_identity(perm: &[u32]) -> bool {
    perm.iter().enumerate().all(|(i, &p)| p == i as u32)
}

/// Key equality as *storage* sees it: equality of the lane the value occupies,
/// not of the value.
///
/// They differ in exactly one place, and it matters. `f64_to_lane` folds every
/// NaN onto one lane (so that the radix order and `Value`'s order agree, and
/// so that the two zeros are one key), while `f64 == f64` reports two NaNs as
/// different values. A part must never hold two rows the MPH index can only
/// address as one, so a dedup that asked `==` would leave a Float64 primary
/// key with two NaN rows in exactly the state task #28 is about.
///
/// `-0.0 == 0.0` is already true in IEEE, so that half needs no special case.
trait SameKey {
    fn same(&self, other: &Self) -> bool;
}
impl SameKey for u64 {
    #[inline(always)]
    fn same(&self, o: &u64) -> bool {
        self == o
    }
}
impl SameKey for i64 {
    #[inline(always)]
    fn same(&self, o: &i64) -> bool {
        self == o
    }
}
impl SameKey for std::sync::Arc<str> {
    #[inline(always)]
    fn same(&self, o: &std::sync::Arc<str>) -> bool {
        self == o
    }
}
impl SameKey for f64 {
    #[inline(always)]
    fn same(&self, o: &f64) -> bool {
        self == o || (self.is_nan() && o.is_nan())
    }
}

/// Drop all but the last occurrence of each key, in permutation space.
///
/// Operating on the permutation rather than the block means a duplicate-free
/// insert -- the overwhelmingly common case -- touches no column data at all.
fn dedup_perm_last_by_key(block: &Block, key_col: usize, perm: &mut Vec<u32>) -> Result<()> {
    if perm.len() <= 1 {
        return Ok(());
    }
    // The buffer kind is matched once here, not once per row. It used to be
    // inside a closure the two loops below called per element, which is a
    // four-way dispatch in a loop whose whole job is one comparison. Measured
    // interleaved against that shape (temporary copy in the tree, best-of-9
    // per side, 400k duplicate-free keys): 0.565 vs 1.128 ns/row, **2.0x**.
    fn go<T: SameKey>(v: &[T], perm: &mut Vec<u32>) {
        let at = |i: u32| -> &T { &v[i as usize] };
        if !perm.windows(2).any(|w| at(w[0]).same(at(w[1]))) {
            return;
        }
        // The sort is stable, so within a run of equal keys the last entry is
        // the most recently written row -- the one a keyed table keeps.
        let mut out: Vec<u32> = Vec::with_capacity(perm.len());
        for i in 0..perm.len() {
            if i + 1 == perm.len() || !at(perm[i]).same(at(perm[i + 1])) {
                out.push(perm[i]);
            }
        }
        *perm = out;
    }
    match &block.column(key_col).data {
        ColumnData::U64(v) => go(v, perm),
        ColumnData::I64(v) => go(v, perm),
        ColumnData::F64(v) => go(v, perm),
        ColumnData::Str(v) => go(v, perm),
    }
    Ok(())
}

#[allow(dead_code)]
fn dedup_last_by_key(block: Block, key_col: usize) -> Result<Block> {
    let n = block.rows();
    if n <= 1 {
        return Ok(block);
    }
    // Scan for a duplicate before materializing anything. Duplicates within a
    // single insert are the exception, so the common path must not copy the
    // block -- an earlier version cloned it unconditionally, which on a
    // multi-million-row insert is hundreds of megabytes of pure waste.
    let has_dup = match &block.column(key_col).data {
        ColumnData::U64(v) => v.windows(2).any(|w| w[0] == w[1]),
        ColumnData::I64(v) => v.windows(2).any(|w| w[0] == w[1]),
        ColumnData::F64(v) => v.windows(2).any(|w| w[0] == w[1]),
        ColumnData::Str(v) => v.windows(2).any(|w| w[0] == w[1]),
    };
    if !has_dup {
        return Ok(block);
    }
    let lanes = lanes_of(&block, key_col)?;
    let mut keep: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        if i + 1 == n || lanes[i] != lanes[i + 1] {
            keep.push(i as u32);
        }
    }
    Ok(block.take(&keep))
}

/// Aggregate live-row counts across a set of keys, used by tests and by
/// `system` introspection.
pub fn count_distinct_lanes(lanes: &[u64]) -> usize {
    let mut seen: FastMap<u64, ()> = FastMap::default();
    for &l in lanes {
        seen.insert(l, ());
    }
    seen.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::splitmix64;
    use crate::types::{Column, Engine, Field, PhysicalType};
    use std::collections::BTreeMap;

    fn def(engine: Engine) -> TableDef {
        TableDef {
            name: "t".into(),
            schema: Schema::new(vec![
                Field::new("id", DataType::UInt64),
                Field::new("v", DataType::Int64),
                Field::new("cat", DataType::UInt32),
            ])
            .unwrap(),
            order_by: vec![0],
            primary_key: vec![0],
            partition_by: None,
            engine,
        }
    }

    fn table() -> Table {
        Table::new(def(Engine::MergeTree), 4096)
    }

    fn block(rows: &[(u64, i64, u32)]) -> Block {
        Block::new(vec![
            Column::u64s(DataType::UInt64, rows.iter().map(|r| r.0).collect()),
            Column::i64s(DataType::Int64, rows.iter().map(|r| r.1).collect()),
            Column::u64s(DataType::UInt32, rows.iter().map(|r| r.2 as u64).collect()),
        ])
        .unwrap()
    }

    #[test]
    fn a_column_whose_buffer_kind_disagrees_with_its_type_is_repaired() {
        // `Column::new` only debug-asserts this agreement, so a release-mode
        // caller can construct `Column { ty: DateTime, data: I64(..) }`.
        // Storage packs from the buffer kind and decodes from the declared
        // type, so letting that through would garble every value.
        let mut d = def(Engine::MergeTree);
        d.schema = Schema::new(vec![
            Field::new("ts", DataType::DateTime),
            Field::new("v", DataType::Int64),
        ])
        .unwrap();
        d.order_by = vec![0];
        d.primary_key = vec![0];
        let mut t = Table::new(d, 100_000);

        assert_eq!(DataType::DateTime.physical(), PhysicalType::U64);
        let wrong = Block::new(vec![
            // deliberately the wrong buffer kind for DateTime
            Column { ty: DataType::DateTime, data: ColumnData::I64(vec![1_700_000_000, 1_700_000_001]), nulls: None },
            Column::i64s(DataType::Int64, vec![10, 20]),
        ])
        .unwrap();
        t.insert(wrong).unwrap();
        t.flush().unwrap();

        let blocks = t.scan(&[0, 1]).unwrap();
        assert_eq!(blocks[0].column(0).value(0), Value::DateTime(1_700_000_000));
        assert_eq!(blocks[0].column(0).value(1), Value::DateTime(1_700_000_001));
        assert_eq!(blocks[0].column(1).as_i64().unwrap(), &[10, 20]);
    }

    #[test]
    fn insert_and_point_lookup() {
        let mut t = table();
        t.insert(block(&[(10, 100, 1), (20, -200, 2)])).unwrap();
        assert_eq!(
            t.get(&Value::UInt(10)).unwrap(),
            Some(vec![Value::UInt(10), Value::Int(100), Value::UInt(1)])
        );
        assert_eq!(t.get(&Value::UInt(99)).unwrap(), None);
        t.flush().unwrap();
        assert_eq!(
            t.get(&Value::UInt(20)).unwrap(),
            Some(vec![Value::UInt(20), Value::Int(-200), Value::UInt(2)])
        );
    }

    #[test]
    fn last_write_wins_across_flushes() {
        let mut t = table();
        t.insert(block(&[(1, 10, 0)])).unwrap();
        t.flush().unwrap();
        t.insert(block(&[(1, 20, 0)])).unwrap();
        t.flush().unwrap();
        assert_eq!(t.get(&Value::UInt(1)).unwrap().unwrap()[1], Value::Int(20));
        assert_eq!(t.row_count().unwrap(), 1, "the old row must be tombstoned");
    }

    #[test]
    fn duplicate_keys_within_one_insert_collapse() {
        let mut t = table();
        // Large enough to take the bulk path, with a repeated key.
        let mut rows: Vec<(u64, i64, u32)> =
            (0..BULK_INSERT_THRESHOLD as u64).map(|i| (i, i as i64, 0)).collect();
        rows.push((5, 999, 0));
        t.insert(block(&rows)).unwrap();
        assert_eq!(t.get(&Value::UInt(5)).unwrap().unwrap()[1], Value::Int(999));
        assert_eq!(t.row_count().unwrap(), BULK_INSERT_THRESHOLD);
    }

    /// Task #28. The batch above arrives *unsorted* (the repeat is appended at
    /// the end), so it was sorted and the dedup rode along on the permutation.
    /// A batch that is already in key order skipped the sort -- and with it
    /// the dedup -- so a table with a declared PRIMARY KEY kept both rows.
    ///
    /// That is not merely a duplicate: the part's MPH index is defined only on
    /// distinct keys, so the index and the data disagree, and the physical
    /// planner lowers `WHERE pk = <const>` to an index probe. Two access paths
    /// to the same predicate, two answers.
    #[test]
    fn duplicate_keys_collapse_even_when_the_batch_needs_no_sort() {
        const N: u64 = BULK_INSERT_THRESHOLD as u64;
        for dup_at in [0u64, 1, N / 2, N - 2] {
            let mut t = table();
            // Ascending, so `key_order` reports it sorted and the sort is
            // skipped -- and one key repeated in place, so the duplicates are
            // adjacent rather than appended.
            let mut rows: Vec<(u64, i64, u32)> = Vec::with_capacity(N as usize + 1);
            for i in 0..N {
                rows.push((i, i as i64, 0));
                if i == dup_at {
                    rows.push((i, i as i64 + 5_000, 0));
                }
            }
            assert!(
                rows.windows(2).all(|w| w[0].0 <= w[1].0),
                "the test needs an already-sorted batch"
            );
            t.insert(block(&rows)).unwrap();

            assert_eq!(
                t.row_count().unwrap(),
                N as usize,
                "dup_at={dup_at}: the duplicate key survived ingest"
            );
            assert_eq!(
                t.get(&Value::UInt(dup_at)).unwrap().unwrap()[1],
                Value::Int(dup_at as i64 + 5_000),
                "dup_at={dup_at}: last write must win, as it does across statements"
            );
            // The scan and the index must agree about how many rows hold the
            // key. This is the half a row count alone cannot catch.
            let snap = t.snapshot();
            let mut seen = 0usize;
            t.scan_each_in(&snap, &[0], |b| {
                seen += (0..b.rows())
                    .filter(|&r| b.column(0).value(r) == Value::UInt(dup_at))
                    .count();
                Ok(())
            })
            .unwrap();
            assert_eq!(seen, 1, "dup_at={dup_at}: a scan still sees two rows");
        }
    }

    /// Three of the same key in a row, and a run at each end of the batch:
    /// the boundaries are where a run-collapsing loop goes wrong.
    #[test]
    fn runs_of_duplicate_keys_collapse_to_the_last_row() {
        const N: usize = BULK_INSERT_THRESHOLD;
        let mut t = table();
        let mut rows: Vec<(u64, i64, u32)> = vec![(0, 1, 0), (0, 2, 0), (0, 3, 0)];
        rows.extend((1..N as u64 - 1).map(|i| (i, i as i64, 0)));
        let last = N as u64 - 1;
        rows.extend([(last, 10, 0), (last, 20, 0)]);
        t.insert(block(&rows)).unwrap();

        assert_eq!(t.row_count().unwrap(), N);
        assert_eq!(t.get(&Value::UInt(0)).unwrap().unwrap()[1], Value::Int(3));
        assert_eq!(t.get(&Value::UInt(last)).unwrap().unwrap()[1], Value::Int(20));
    }

    /// A table with no declared PRIMARY KEY has no uniqueness claim, so a
    /// sorted batch with repeated sort-key values must keep every row. The fix
    /// above must not quietly turn ORDER BY back into a unique key -- that was
    /// the row-eating bug the engine already backed out of once.
    #[test]
    fn a_sort_key_without_a_primary_key_still_keeps_duplicates() {
        let mut d = def(Engine::MergeTree);
        d.primary_key.clear();
        let mut t = Table::new(d, 100_000);
        assert_eq!(t.pk_col(), None);
        let rows: Vec<(u64, i64, u32)> = (0..BULK_INSERT_THRESHOLD as u64)
            .map(|i| (i / 4, i as i64, 0)) // every key repeated four times
            .collect();
        t.insert(block(&rows)).unwrap();
        assert_eq!(t.row_count().unwrap(), BULK_INSERT_THRESHOLD);
    }

    /// Every NaN occupies one storage lane (`f64_to_lane` folds them together
    /// so the radix order matches `Value`'s), so a part must not hold two NaN
    /// rows -- the MPH could only address them as one. `f64 == f64` says two
    /// NaNs differ, which is why the dedup asks `SameKey` instead.
    #[test]
    fn nan_and_negative_zero_are_one_key() {
        let mut d = def(Engine::MergeTree);
        d.schema = Schema::new(vec![
            Field::new("k", DataType::Float64),
            Field::new("v", DataType::Int64),
        ])
        .unwrap();
        d.order_by = vec![0];
        d.primary_key = vec![0];
        let mut t = Table::new(d, 100_000);
        assert!(t.def.has_fast_pk());

        let n = BULK_INSERT_THRESHOLD;
        let mut keys: Vec<f64> = vec![f64::NAN, f64::NAN, -0.0, 0.0];
        let mut vals: Vec<i64> = vec![1, 2, 3, 4];
        for i in 0..n {
            keys.push(i as f64 + 1.0);
            vals.push(i as i64 + 100);
        }
        let b = Block::new(vec![
            Column::f64s(DataType::Float64, keys),
            Column::i64s(DataType::Int64, vals),
        ])
        .unwrap();
        t.insert(b).unwrap();

        // Four input rows became two: one NaN key and one zero key.
        assert_eq!(t.row_count().unwrap(), n + 2);
        assert_eq!(t.get(&Value::Float(0.0)).unwrap().unwrap()[1], Value::Int(4));
        assert_eq!(t.get(&Value::Float(f64::NAN)).unwrap().unwrap()[1], Value::Int(2));
    }

    // ---- transactions -------------------------------------------------------

    #[test]
    fn a_transaction_reads_its_own_writes_and_publishes_nothing_until_commit() {
        let mut t = table();
        t.insert(block(&[(1, 10, 0), (2, 20, 0)])).unwrap();
        t.flush().unwrap();
        let outside = t.snapshot();

        t.begin_txn().unwrap();
        t.insert(block(&[(3, 30, 0)])).unwrap();
        t.insert(block(&[(1, 11, 0)])).unwrap(); // shadows a committed row
        t.delete_key(&Value::UInt(2)).unwrap();
        t.flush().unwrap();

        // Read-your-own-writes, through every path a query can take.
        assert_eq!(t.get(&Value::UInt(3)).unwrap().unwrap()[1], Value::Int(30));
        assert_eq!(t.get(&Value::UInt(1)).unwrap().unwrap()[1], Value::Int(11));
        assert_eq!(t.get(&Value::UInt(2)).unwrap(), None);
        assert_eq!(t.row_count().unwrap(), 2);

        // ...while nothing committed has moved.
        assert_eq!(outside.live_rows(), 2, "a pinned reader saw the transaction");
        assert_eq!(t.committed_snapshot().live_rows(), 2);

        t.commit_txn();
        assert!(!t.in_txn());
        assert_eq!(t.committed_snapshot().live_rows(), 2);
        assert_eq!(t.row_count().unwrap(), 2);
        assert_eq!(t.get(&Value::UInt(1)).unwrap().unwrap()[1], Value::Int(11));
        assert_eq!(t.get(&Value::UInt(3)).unwrap().unwrap()[1], Value::Int(30));
        assert_eq!(t.get(&Value::UInt(2)).unwrap(), None);
        assert_eq!(outside.live_rows(), 2, "and the pinned reader is still pinned");
    }

    /// The point of having done the keystone: undoing a transaction is
    /// dropping a pointer. Inserts, key-shadowing tombstones, a flushed part
    /// and a compaction all disappear together.
    #[test]
    fn rollback_leaves_no_trace() {
        let mut t = table();
        let base: Vec<(u64, i64, u32)> = (0..2_000).map(|i| (i, i as i64, 0)).collect();
        t.insert(block(&base)).unwrap();
        t.flush().unwrap();
        let before_parts = t.part_count();
        let before_version = t.parts_version();
        let pinned = t.snapshot();

        t.begin_txn().unwrap();
        // Enough shapes to cover every mutating path: a bulk ingest that
        // tombstones, a buffered write, a delete, and a compaction.
        let over: Vec<(u64, i64, u32)> =
            (0..BULK_INSERT_THRESHOLD as u64).map(|i| (i, -1, 0)).collect();
        t.insert(block(&over)).unwrap();
        t.insert(block(&[(9_999, 7, 0)])).unwrap();
        t.delete_key(&Value::UInt(5)).unwrap();
        t.flush().unwrap();
        t.compact().unwrap();
        assert_eq!(t.get(&Value::UInt(0)).unwrap().unwrap()[1], Value::Int(-1));

        t.rollback_txn();

        assert!(!t.in_txn());
        assert_eq!(t.delta_len(), 0, "buffered writes must go with the overlay");
        assert_eq!(t.part_count(), before_parts, "a part survived the rollback");
        assert_eq!(t.parts_version(), before_version, "the published set moved");
        assert_eq!(t.row_count().unwrap(), 2_000);
        assert_eq!(t.get(&Value::UInt(9_999)).unwrap(), None);
        assert_eq!(t.get(&Value::UInt(5)).unwrap().unwrap()[1], Value::Int(5));
        for k in 0..2_000u64 {
            assert_eq!(
                t.get(&Value::UInt(k)).unwrap().map(|r| r[1].clone()),
                Some(Value::Int(k as i64)),
                "key {k} was altered by a rolled-back transaction"
            );
        }
        assert!(std::ptr::eq(t.snapshot().part(0), pinned.part(0)), "same parts, not copies");
    }

    /// Writes made before BEGIN are committed data. Rolling back must not eat
    /// them just because they were still sitting in the delta -- which is why
    /// `begin_txn` flushes.
    #[test]
    fn rollback_keeps_writes_that_were_buffered_before_begin() {
        let mut t = table();
        t.insert(block(&[(1, 10, 0), (2, 20, 0)])).unwrap();
        assert_eq!(t.delta_len(), 2, "the test needs them still buffered");

        t.begin_txn().unwrap();
        assert_eq!(t.delta_len(), 0, "begin must flush to a clean boundary");
        t.insert(block(&[(3, 30, 0)])).unwrap();
        t.rollback_txn();

        assert_eq!(t.row_count().unwrap(), 2);
        assert_eq!(t.get(&Value::UInt(1)).unwrap().unwrap()[1], Value::Int(10));
        assert_eq!(t.get(&Value::UInt(2)).unwrap().unwrap()[1], Value::Int(20));
        assert_eq!(t.get(&Value::UInt(3)).unwrap(), None);
    }

    #[test]
    fn a_second_begin_is_refused_rather_than_flattened() {
        let mut t = table();
        t.begin_txn().unwrap();
        assert!(t.begin_txn().is_err(), "nesting is the caller's bug, not a savepoint");
        t.rollback_txn();
        t.begin_txn().unwrap();
        t.commit_txn();
        // Commit and rollback outside a transaction are no-ops, not panics.
        t.commit_txn();
        t.rollback_txn();
        assert!(!t.in_txn());
    }

    /// A committed transaction must be indistinguishable from the same writes
    /// made without one, right down to what a later reader can see.
    #[test]
    fn a_committed_transaction_matches_the_same_writes_without_one() {
        let write = |t: &mut Table| {
            t.insert(block(&[(1, 10, 0), (2, 20, 0)])).unwrap();
            t.flush().unwrap();
            t.insert(block(&[(2, 22, 0), (3, 30, 0)])).unwrap();
            t.delete_key(&Value::UInt(1)).unwrap();
            t.flush().unwrap();
        };
        let mut plain = table();
        write(&mut plain);

        let mut txn = table();
        txn.begin_txn().unwrap();
        write(&mut txn);
        txn.commit_txn();

        assert_eq!(plain.row_count().unwrap(), txn.row_count().unwrap());
        for k in 0..5u64 {
            assert_eq!(
                plain.get(&Value::UInt(k)).unwrap(),
                txn.get(&Value::UInt(k)).unwrap(),
                "key {k}"
            );
        }
    }

    #[test]
    fn delete_hides_the_row() {
        let mut t = table();
        t.insert(block(&[(1, 10, 0), (2, 20, 0)])).unwrap();
        t.flush().unwrap();
        t.delete_key(&Value::UInt(1)).unwrap();
        assert_eq!(t.get(&Value::UInt(1)).unwrap(), None);
        t.flush().unwrap();
        assert_eq!(t.get(&Value::UInt(1)).unwrap(), None);
        assert_eq!(t.row_count().unwrap(), 1);
    }

    #[test]
    fn a_failed_flush_destroys_nothing() {
        // Every SELECT flushes every table, so this error path is reachable
        // from a read. It used to drain the delta and tombstone the shadowed
        // part rows *before* packing, so a packing failure deleted key 3
        // outright and left key 2 tombstoned in the part with its replacement
        // already thrown away -- two rows silently gone, from a SELECT.
        let mut t = table();
        t.insert(block(&[(1, 10, 0), (2, 20, 0)])).unwrap();
        t.flush().unwrap();
        t.insert(block(&[(2, 22, 0), (3, 30, 0)])).unwrap();
        assert_eq!(t.delta_len(), 2, "test needs the rows still buffered");

        arm_build_failure();
        assert!(t.flush().is_err());

        assert_eq!(t.delta_len(), 2, "the delta must not have been drained");
        assert!(
            t.snapshot().deletes(0).is_none(),
            "no tombstone may be applied before the replacement part exists"
        );
        assert_eq!(t.get(&Value::UInt(1)).unwrap().unwrap()[1], Value::Int(10));
        assert_eq!(t.get(&Value::UInt(2)).unwrap().unwrap()[1], Value::Int(22));
        assert_eq!(t.get(&Value::UInt(3)).unwrap().unwrap()[1], Value::Int(30));

        // ...and the retry still does the whole job, tombstone included.
        t.flush().unwrap();
        assert_eq!(t.row_count().unwrap(), 3);
        assert_eq!(t.get(&Value::UInt(2)).unwrap().unwrap()[1], Value::Int(22));
    }

    #[test]
    fn a_failed_bulk_ingest_hides_no_rows() {
        // The failure mode pinned here is not a lost write, it is an
        // *invisible* row. A mid-batch `to_lane` error used to leave every key
        // the loop had already reached tombstoned in the older part, while the
        // replacement part meant to supersede them was dropped along with the
        // error. Nothing reports that: the rows simply stop existing, and a
        // count is no help because the tombstones are consistent with a
        // successful DELETE. So this asserts on the value every key reads back.
        const N: u64 = BULK_INSERT_THRESHOLD as u64;
        // Both spellings of the tombstone loop: `shuffled` forces a sort
        // permutation and so selects the gathered variant.
        for shuffled in [false, true] {
            let mut t = table();
            let base: Vec<(u64, i64, u32)> = (0..N).map(|k| (k, k as i64, 0)).collect();
            t.insert(block(&base)).unwrap();
            assert_eq!(t.part_count(), 1);

            let mut upd: Vec<(u64, i64, u32)> =
                (0..N).map(|k| (k, k as i64 + 1_000_000, 0)).collect();
            if shuffled {
                upd.reverse();
            }
            // Every key shadows one in the part above, so the loop runs N
            // times and this fails it half way through.
            arm_tombstone_failure(N as usize / 2);
            assert!(t.insert(block(&upd)).is_err(), "the injected failure must surface");

            assert_eq!(t.part_count(), 1, "the unpublishable part must not be published");
            assert!(
                t.snapshot().deletes(0).is_none(),
                "not one tombstone may survive a batch that did not complete"
            );
            assert_eq!(t.row_count().unwrap(), N as usize);
            for k in 0..N {
                assert_eq!(
                    t.get(&Value::UInt(k)).unwrap().map(|r| r[1].clone()),
                    Some(Value::Int(k as i64)),
                    "key {k} hidden or altered by a failed bulk ingest (shuffled={shuffled})"
                );
            }

            // ...and the retry still does the whole job, tombstones included.
            t.insert(block(&upd)).unwrap();
            assert_eq!(t.row_count().unwrap(), N as usize);
            for k in 0..N {
                let row = t
                    .get(&Value::UInt(k))
                    .unwrap()
                    .unwrap_or_else(|| panic!("key {k} lost on retry"));
                assert_eq!(row[1], Value::Int(k as i64 + 1_000_000), "stale key {k}");
            }
        }
    }

    #[test]
    fn a_failed_merge_keeps_its_input_parts() {
        let mut t = table();
        for chunk in 0..4u64 {
            let rows: Vec<(u64, i64, u32)> =
                (0..100).map(|i| (chunk * 100 + i, i as i64, 0)).collect();
            t.insert(block(&rows)).unwrap();
            t.flush().unwrap();
        }
        let before = t.part_count();
        assert!(before > 1);

        // The delta is empty, so `compact`'s leading flush returns before it
        // can consume the armed failure; the merge's build gets it.
        arm_build_failure();
        assert!(t.compact().is_err());

        assert_eq!(
            t.part_count(),
            before,
            "inputs must not be unlinked until the merged part exists"
        );
        assert_eq!(t.row_count().unwrap(), 400);
        for k in 0..400u64 {
            assert!(t.get(&Value::UInt(k)).unwrap().is_some(), "lost key {k}");
        }

        t.compact().unwrap();
        assert_eq!(t.part_count(), 1);
        assert_eq!(t.row_count().unwrap(), 400);
    }

    #[test]
    fn compaction_merges_and_drops_deleted() {
        let mut t = table();
        for chunk in 0..5u64 {
            let rows: Vec<(u64, i64, u32)> =
                (0..500).map(|i| (chunk * 500 + i, i as i64, 0)).collect();
            t.insert(block(&rows)).unwrap();
            t.flush().unwrap();
        }
        t.delete_key(&Value::UInt(7)).unwrap();
        t.flush().unwrap();
        assert!(t.part_count() > 1);
        t.compact().unwrap();
        assert_eq!(t.part_count(), 1);
        assert_eq!(t.row_count().unwrap(), 2499);
        assert_eq!(t.get(&Value::UInt(7)).unwrap(), None);
        assert_eq!(t.get(&Value::UInt(8)).unwrap().unwrap()[0], Value::UInt(8));
    }

    #[test]
    fn multi_get_matches_get() {
        let mut t = Table::new(def(Engine::MergeTree), 400);
        let mut model = BTreeMap::new();
        for i in 0..3000u64 {
            let k = splitmix64(i) % 10_000;
            let v = (i as i64 % 100) - 50;
            t.insert(block(&[(k, v, 1)])).unwrap();
            model.insert(k, v);
        }
        let probes: Vec<u64> = (0..2000u64).map(|i| splitmix64(i * 7) % 12_000).collect();
        let mut out = vec![None; probes.len()];
        t.multi_get(&probes, &mut out);
        for (i, &k) in probes.iter().enumerate() {
            let expect = t.get_lane(k);
            assert_eq!(out[i], expect, "key {k}");
            match model.get(&k) {
                Some(v) => assert_eq!(out[i].as_ref().unwrap()[1], Value::Int(*v)),
                None => assert!(out[i].is_none()),
            }
        }
    }

    #[test]
    fn fuzz_against_reference_model() {
        let mut t = Table::new(def(Engine::MergeTree), 700);
        let mut model: BTreeMap<u64, (i64, u32)> = BTreeMap::new();
        let mut seed = 0xDEAD_BEEFu64;
        let mut next = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            splitmix64(seed)
        };
        for step in 0..20_000u64 {
            let k = next() % 5_000;
            match next() % 10 {
                0..=5 => {
                    let (v, c) = ((next() % 1_000_000) as i64 - 500_000, (next() % 8) as u32);
                    t.insert(block(&[(k, v, c)])).unwrap();
                    model.insert(k, (v, c));
                }
                6..=7 => {
                    t.delete_key(&Value::UInt(k)).unwrap();
                    model.remove(&k);
                }
                8 => {
                    let got = t.get(&Value::UInt(k)).unwrap();
                    let want = model.get(&k).map(|&(v, c)| {
                        vec![Value::UInt(k), Value::Int(v), Value::UInt(c as u64)]
                    });
                    assert_eq!(got, want, "step {step} key {k}");
                }
                _ => {
                    if step % 4000 == 0 {
                        t.compact().unwrap();
                    } else if step % 900 == 0 {
                        t.flush().unwrap();
                    }
                }
            }
        }
        t.compact().unwrap();
        for k in 0..5_000u64 {
            let got = t.get(&Value::UInt(k)).unwrap();
            let want = model
                .get(&k)
                .map(|&(v, c)| vec![Value::UInt(k), Value::Int(v), Value::UInt(c as u64)]);
            assert_eq!(got, want, "final key {k}");
        }
        assert_eq!(t.row_count().unwrap(), model.len());
    }

    #[test]
    fn a_snapshot_does_not_move_when_a_write_shadows_it() {
        // The bug the keystone exists to remove. `mark_deleted` used to flip a
        // bit inside an already-published part, so a scan holding that part
        // would see the tombstone appear mid-walk: row present in granule 3,
        // absent in granule 7, of what was supposed to be one instant.
        let mut t = table();
        let rows: Vec<(u64, i64, u32)> = (0..3_000).map(|i| (i, i as i64, 0)).collect();
        t.insert(block(&rows)).unwrap();
        t.flush().unwrap();

        let pinned = t.snapshot();
        assert_eq!(pinned.live_rows(), 3_000);

        // Overwrite half the keys and delete a few: every one of these
        // tombstones a row in the part `pinned` is holding.
        let over: Vec<(u64, i64, u32)> = (0..1_500).map(|i| (i, -1, 0)).collect();
        t.insert(block(&over)).unwrap();
        t.delete_key(&Value::UInt(2_999)).unwrap();
        t.flush().unwrap();

        assert_eq!(pinned.live_rows(), 3_000, "the pinned view lost rows");
        assert!(t.parts_version() > pinned.version(), "the version must advance");
        // Read it: every original row is still there, at its original value.
        let mut seen = 0usize;
        t.scan_each_in(&pinned, &[0, 1], |b| {
            for r in 0..b.rows() {
                let Value::UInt(k) = b.column(0).value(r) else { panic!() };
                assert_eq!(b.column(1).value(r), Value::Int(k as i64), "key {k} moved");
                seen += 1;
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, 3_000);

        // ...and the live table has moved on exactly as it should.
        assert_eq!(t.row_count().unwrap(), 2_999);
        assert_eq!(t.get(&Value::UInt(0)).unwrap().unwrap()[1], Value::Int(-1));
    }

    #[test]
    fn a_compacted_away_part_stays_alive_under_a_snapshot() {
        // Compaction retires input parts and the caller unlinks their files.
        // A reader holding the old snapshot must still be able to decode them
        // -- the mmap is attached to the inode, not the directory entry.
        let mut t = table();
        for chunk in 0..4u64 {
            let rows: Vec<(u64, i64, u32)> =
                (0..500).map(|i| (chunk * 500 + i, i as i64, 0)).collect();
            t.insert(block(&rows)).unwrap();
            t.flush().unwrap();
        }
        let pinned = t.snapshot();
        let before: Vec<*const Part> = pinned.parts().iter().map(|p| &**p as *const _).collect();
        assert!(before.len() > 1);

        t.compact().unwrap();
        assert_eq!(t.part_count(), 1);

        // Same parts, same addresses, still readable.
        assert_eq!(pinned.len(), before.len());
        for (i, &p) in before.iter().enumerate() {
            assert!(std::ptr::eq(pinned.part(i), p));
        }
        let n = t
            .scan_fold_in(&pinned, &[0], || 0usize, |a, b| { *a += b.rows(); Ok(()) }, |a, b| a + b)
            .unwrap();
        assert_eq!(n, 2_000);
    }

    #[test]
    fn freeze_delta_sees_buffered_writes_without_flushing_them() {
        let mut t = table();
        t.insert(block(&[(5, 50, 0), (1, 10, 0)])).unwrap();
        assert_eq!(t.part_count(), 0, "the test needs the rows still buffered");

        let img = t.freeze_delta().unwrap();
        assert_eq!(img.keys(), &[1, 5], "the image must be key-sorted");
        assert_eq!(img.block().column(1).value(0), Value::Int(10));
        assert_eq!(t.delta_len(), 2, "freeze must not drain");
        assert_eq!(t.part_count(), 0, "freeze must not build a part");
    }

    #[test]
    fn parallel_scan_fold_matches_the_serial_scan() {
        let mut t = table();
        let rows: Vec<(u64, i64, u32)> = (0..50_000).map(|i| (i, i as i64 * 2, 0)).collect();
        t.insert(block(&rows)).unwrap();
        t.delete_key(&Value::UInt(3)).unwrap();
        t.delete_key(&Value::UInt(49_999)).unwrap();

        let expect: i64 = (0..50_000i64).map(|i| i * 2).sum::<i64>() - 6 - 99_998;

        // Every worker folds its own accumulator; the merge has to be exact.
        let (sum, count) = t
            .scan_fold(
                &[1],
                || (0i64, 0usize),
                |acc, b| {
                    acc.0 += b.column(0).as_i64()?.iter().sum::<i64>();
                    acc.1 += b.rows();
                    Ok(())
                },
                |a, b| (a.0 + b.0, a.1 + b.1),
            )
            .unwrap();
        assert_eq!(sum, expect, "parallel fold disagreed with the reference");
        assert_eq!(count, 49_998, "deleted rows leaked into a parallel scan");

        // ...and it agrees with the serial path row for row.
        let serial: i64 = t
            .scan(&[1])
            .unwrap()
            .iter()
            .map(|b| b.column(0).as_i64().unwrap().iter().sum::<i64>())
            .sum();
        assert_eq!(sum, serial);
    }

    #[test]
    fn parallel_scan_handles_empty_and_tiny_tables() {
        let mut t = table();
        let n = |t: &mut Table| {
            t.scan_fold(&[0], || 0usize, |a, b| { *a += b.rows(); Ok(()) }, |a, b| a + b)
                .unwrap()
        };
        assert_eq!(n(&mut t), 0, "empty table");
        t.insert(block(&[(1, 1, 0)])).unwrap();
        assert_eq!(n(&mut t), 1, "single row");
        let rows: Vec<(u64, i64, u32)> = (2..2000).map(|i| (i, i as i64, 0)).collect();
        t.insert(block(&rows)).unwrap();
        assert_eq!(n(&mut t), 1999);
    }

    #[test]
    fn parallel_scan_propagates_an_error_from_a_worker() {
        let mut t = table();
        let rows: Vec<(u64, i64, u32)> = (0..30_000).map(|i| (i, i as i64, 0)).collect();
        t.insert(block(&rows)).unwrap();
        let r = t.scan_fold(
            &[1],
            || 0i64,
            |_, _| Err(Error::exec("boom")),
            |a, _b| a,
        );
        assert!(r.is_err(), "a failing fold must not be swallowed");
    }

    #[test]
    fn scan_returns_every_live_row() {
        let mut t = table();
        let rows: Vec<(u64, i64, u32)> = (0..5000).map(|i| (i, i as i64 * 2, 0)).collect();
        t.insert(block(&rows)).unwrap();
        t.delete_key(&Value::UInt(3)).unwrap();
        let blocks = t.scan(&[0, 1]).unwrap();
        let total: usize = blocks.iter().map(|b| b.rows()).sum();
        assert_eq!(total, 4999);
        let mut sum = 0i64;
        for b in &blocks {
            sum += b.column(1).as_i64().unwrap().iter().sum::<i64>();
        }
        let expect: i64 = (0..5000i64).map(|i| i * 2).sum::<i64>() - 6;
        assert_eq!(sum, expect);
    }

    #[test]
    fn signed_primary_key_sorts_correctly() {
        let mut d = def(Engine::MergeTree);
        d.schema = Schema::new(vec![
            Field::new("k", DataType::Int64),
            Field::new("v", DataType::Int64),
        ])
        .unwrap();
        d.order_by = vec![0];
        d.primary_key = vec![0];
        let mut t = Table::new(d, 100_000);
        assert!(t.def.has_fast_pk(), "signed PKs take the fast path");

        let keys: Vec<i64> = vec![5, -3, 0, i64::MIN, i64::MAX, -1000, 1000];
        let b = Block::new(vec![
            Column::i64s(DataType::Int64, keys.clone()),
            Column::i64s(DataType::Int64, keys.iter().map(|k| k.wrapping_mul(2)).collect()),
        ])
        .unwrap();
        t.insert(b).unwrap();
        t.flush().unwrap();
        for &k in &keys {
            assert_eq!(
                t.get(&Value::Int(k)).unwrap(),
                Some(vec![Value::Int(k), Value::Int(k.wrapping_mul(2))]),
                "key {k}"
            );
        }
        // sorted order in storage
        let snap = t.snapshot();
        let p = snap.part(0);
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        for (i, &want) in sorted.iter().enumerate() {
            assert_eq!(p.value_at(i, 0), Value::Int(want));
        }
    }

    #[test]
    fn string_sort_key_works_without_the_lane_index() {
        let mut d = def(Engine::MergeTree);
        d.schema = Schema::new(vec![
            Field::new("name", DataType::String),
            Field::new("n", DataType::UInt64),
        ])
        .unwrap();
        d.order_by = vec![0];
        d.primary_key = vec![0];
        let mut t = Table::new(d, 100_000);
        assert!(!t.def.has_fast_pk());
        assert_eq!(t.def.sort_col(), None);

        let b = Block::new(vec![
            Column::strs(
                DataType::String,
                vec!["pear".into(), "apple".into(), "fig".into()],
            ),
            Column::u64s(DataType::UInt64, vec![1, 2, 3]),
        ])
        .unwrap();
        t.insert(b).unwrap();
        t.flush().unwrap();
        let blocks = t.scan(&[0, 1]).unwrap();
        let names: Vec<String> = blocks[0]
            .column(0)
            .as_str()
            .unwrap()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(names, vec!["apple", "fig", "pear"], "still sorted by value");
    }

    #[test]
    fn compression_report_shows_narrow_columns() {
        let mut t = table();
        let rows: Vec<(u64, i64, u32)> = (0..20_000)
            .map(|i| (1_000_000 + i, (i as i64 % 100) - 50, (i % 8) as u32))
            .collect();
        t.insert(block(&rows)).unwrap();
        t.compact().unwrap();
        let r = t.compression_report();
        assert_eq!(r.rows, 20_000);
        // cat has 8 distinct values -> 3 bits/row
        let cat = r.columns.iter().find(|c| c.name == "cat").unwrap();
        assert!(cat.stored_bits < 4.0, "cat took {} bits/row", cat.stored_bits);
        // clustered ids -> ~10 bits/row
        let id = r.columns.iter().find(|c| c.name == "id").unwrap();
        assert!(id.stored_bits < 12.0, "id took {} bits/row", id.stored_bits);
        assert!(r.ratio() > 3.0, "overall ratio only {:.2}x", r.ratio());
    }

    #[test]
    fn log_engine_appends_without_sorting() {
        let mut t = Table::new(def(Engine::Log), 100_000);
        assert!(!t.def.has_fast_pk());
        t.insert(block(&[(9, 1, 0), (3, 2, 0), (7, 3, 0)])).unwrap();
        t.flush().unwrap();
        let blocks = t.scan(&[0]).unwrap();
        assert_eq!(blocks[0].column(0).as_u64().unwrap(), &[9, 3, 7]);
    }

    #[test]
    fn sort_permutation_handles_composite_and_nullable_keys() {
        let b = Block::new(vec![
            Column::u64s(DataType::UInt64, vec![2, 1, 2, 1]),
            Column::i64s(DataType::Int64, vec![10, 30, 5, 20]),
        ])
        .unwrap();
        let perm = sort_permutation(&b, &[0, 1]).unwrap();
        let s = b.take(&perm);
        assert_eq!(s.column(0).as_u64().unwrap(), &[1, 1, 2, 2]);
        assert_eq!(s.column(1).as_i64().unwrap(), &[20, 30, 5, 10]);

        // nullable key falls back to comparison sort; NULL sorts first
        let mut nb = ColumnBuilder::new(DataType::Nullable(Box::new(DataType::Int64)));
        nb.push_value(&Value::Int(5)).unwrap();
        nb.push_null();
        nb.push_value(&Value::Int(1)).unwrap();
        let b = Block::new(vec![nb.finish()]).unwrap();
        let perm = sort_permutation(&b, &[0]).unwrap();
        assert_eq!(perm, vec![1, 2, 0]);
    }
    // ------------------------------------------------------------ bulk delete

    /// `col <op> literal` over the projected schema, which is the only shape
    /// `delete_where` ever sees: nothing in it knows the predicate came from
    /// the binder rather than from here.
    fn pred(col: usize, op: crate::sql::ast::BinaryOp, v: Value, ty: DataType) -> BoundExpr {
        BoundExpr::Binary {
            left: Box::new(BoundExpr::Column { index: col, ty: ty.clone(), name: "c".into() }),
            op,
            right: Box::new(BoundExpr::Literal { value: v, ty }),
            ty: DataType::Bool,
        }
    }

    fn live_ids(t: &mut Table) -> Vec<u64> {
        let mut out = Vec::new();
        t.scan_each(&[0], |b| {
            out.extend(b.column(0).as_u64().unwrap().iter().copied());
            Ok(())
        })
        .unwrap();
        out.sort_unstable();
        out
    }

    #[test]
    fn a_bulk_delete_hides_matching_rows_and_publishes_one_version() {
        use crate::sql::ast::BinaryOp;
        let mut t = table();
        let rows: Vec<(u64, i64, u32)> = (0..5_000u64).map(|i| (i, i as i64, 0)).collect();
        t.insert(block(&rows)).unwrap();
        t.flush().unwrap();
        let before = t.parts_version();

        // Every row below 3000, expressed against a one-column projection.
        let p = pred(0, BinaryOp::Lt, Value::UInt(3_000), DataType::UInt64);
        let n = t.delete_where(&[0], Some(&p), &[]).unwrap();
        assert_eq!(n, 3_000);
        assert_eq!(t.row_count().unwrap(), 2_000);
        assert_eq!(live_ids(&mut t).first().copied(), Some(3_000));
        // One version for the whole statement, whatever the row count: the
        // sweep publishes once, and `row_count`'s flush finds nothing to do.
        assert_eq!(t.parts_version(), before + 1);

        // Re-running it hides nothing more, and says so. Reporting rows that
        // were already tombstoned would make the count the caller shows
        // disagree with what the statement actually changed.
        assert_eq!(t.delete_where(&[0], Some(&p), &[]).unwrap(), 0);
        assert_eq!(t.row_count().unwrap(), 2_000);
    }

    /// No predicate is `DELETE FROM t` with no `WHERE`, which the folder
    /// reduces to exactly this. It must not decode a column to answer it.
    #[test]
    fn a_bulk_delete_with_no_predicate_empties_the_table() {
        let mut t = table();
        let rows: Vec<(u64, i64, u32)> = (0..4_000u64).map(|i| (i, i as i64, 1)).collect();
        t.insert(block(&rows)).unwrap();
        assert_eq!(t.delete_where(&[], None, &[]).unwrap(), 4_000);
        assert_eq!(t.row_count().unwrap(), 0);
        assert!(live_ids(&mut t).is_empty());
    }

    /// Rows still in the write buffer are not in any part, and a position is
    /// only meaningful inside one -- so the sweep has to fold them in first or
    /// a delete silently misses everything written since the last flush.
    #[test]
    fn a_bulk_delete_sees_rows_that_are_still_buffered() {
        use crate::sql::ast::BinaryOp;
        let mut t = table();
        t.insert(block(&[(1, 10, 0), (2, 20, 0), (3, 30, 0)])).unwrap();
        assert_eq!(t.delta_len(), 3, "the batch is small enough to buffer");
        let p = pred(0, BinaryOp::LtEq, Value::UInt(2), DataType::UInt64);
        assert_eq!(t.delete_where(&[0], Some(&p), &[]).unwrap(), 2);
        assert_eq!(live_ids(&mut t), vec![3]);
    }

    /// The zone filter's `col` indexes the *projected* schema while a granule's
    /// columns are indexed by table column. Projecting a column that is not #0
    /// is what tells the two apart: mapping them the wrong way round prunes on
    /// the key instead of the value and deletes the wrong rows entirely.
    #[test]
    fn zone_filters_prune_through_the_projection_not_around_it() {
        use crate::planner::logical::CmpOp;
        use crate::sql::ast::BinaryOp;
        let mut t = table();
        // `v` descends as `id` ascends, so pruning on the wrong one keeps the
        // wrong half rather than merely reading more granules.
        let rows: Vec<(u64, i64, u32)> =
            (0..4_000u64).map(|i| (i, 4_000 - i as i64, 0)).collect();
        t.insert(block(&rows)).unwrap();
        t.flush().unwrap();

        // Projection is [v]; the predicate and the zone filter both index it
        // as column 0, though `v` is table column 1.
        let p = pred(0, BinaryOp::Lt, Value::Int(100), DataType::Int64);
        let z = [ZoneFilter { col: 0, op: CmpOp::Lt, value: Value::Int(100) }];
        let n = t.delete_where(&[1], Some(&p), &z).unwrap();
        assert_eq!(n, 99, "v in 1..=99, i.e. the last 99 ids");
        assert_eq!(live_ids(&mut t).len(), 3_901);
        assert_eq!(live_ids(&mut t).last().copied(), Some(3_900));

        // And the pruning must not change the answer: same predicate, no zone
        // filters at all.
        let mut u = table();
        u.insert(block(&rows)).unwrap();
        u.flush().unwrap();
        assert_eq!(u.delete_where(&[1], Some(&p), &[]).unwrap(), 99);
        assert_eq!(live_ids(&mut u), live_ids(&mut t));
    }

    /// A NULL predicate is not TRUE, and a filter admits TRUE rows only. The
    /// three-valued rule has to hold here for the same reason it holds in
    /// `expr::eval_predicate`: a delete that treated UNKNOWN as "matches" would
    /// destroy rows the statement did not name.
    #[test]
    fn a_null_predicate_deletes_nothing() {
        use crate::sql::ast::BinaryOp;
        let mut d = def(Engine::MergeTree);
        d.schema = Schema::new(vec![
            Field::new("id", DataType::UInt64),
            Field::new("v", DataType::Nullable(Box::new(DataType::Int64))),
        ])
        .unwrap();
        let mut t = Table::new(d, 4096);
        let mut b = Block::new(vec![
            Column::u64s(DataType::UInt64, vec![1, 2, 3]),
            Column::i64s(DataType::Nullable(Box::new(DataType::Int64)), vec![0, 7, 0]),
        ])
        .unwrap();
        let mut nulls = crate::common::BitSet::new();
        nulls.set(0);
        nulls.set(2);
        b.columns[1].nulls = Some(nulls);
        t.insert(b).unwrap();

        let p = pred(0, BinaryOp::Gt, Value::Int(1), DataType::Int64);
        assert_eq!(t.delete_where(&[1], Some(&p), &[]).unwrap(), 1);
        assert_eq!(t.row_count().unwrap(), 2);
    }

    /// The transaction guarantee, at the storage layer: a sweep inside one
    /// writes into the private overlay, and ROLLBACK drops it whole. This is
    /// the capability the `Arc<PartSet>` keystone was for -- before it a
    /// tombstone was a bit flipped inside a live part, with no way back.
    #[test]
    fn a_bulk_delete_inside_a_rolled_back_transaction_leaves_every_row() {
        use crate::sql::ast::BinaryOp;
        let mut t = table();
        let rows: Vec<(u64, i64, u32)> = (0..3_000u64).map(|i| (i, i as i64, 0)).collect();
        t.insert(block(&rows)).unwrap();
        t.flush().unwrap();
        let committed = t.committed_snapshot();

        let p = pred(0, BinaryOp::Lt, Value::UInt(2_500), DataType::UInt64);
        t.begin_txn().unwrap();
        assert_eq!(t.delete_where(&[0], Some(&p), &[]).unwrap(), 2_500);
        assert_eq!(t.row_count().unwrap(), 500, "read-your-own-writes");
        // The published set never moved, so a reader outside the transaction
        // still sees all three thousand.
        assert_eq!(committed.live_rows(), 3_000);
        assert_eq!(t.committed_snapshot().live_rows(), 3_000);

        t.rollback_txn();
        assert_eq!(t.row_count().unwrap(), 3_000);
        assert_eq!(live_ids(&mut t).len(), 3_000);

        // ...and the same sweep committed really does publish.
        t.begin_txn().unwrap();
        assert_eq!(t.delete_where(&[0], Some(&p), &[]).unwrap(), 2_500);
        t.commit_txn();
        assert_eq!(t.row_count().unwrap(), 500);
    }

    /// The lanes a logging session needs. They must name the rows the sweep
    /// *newly* hid: replaying a delete for a row an earlier statement already
    /// tombstoned is harmless, but counting it is not.
    #[test]
    fn a_bulk_delete_reports_the_key_lanes_it_hid() {
        use crate::sql::ast::BinaryOp;
        let mut t = table();
        t.insert(block(&[(1, 1, 0), (2, 2, 0), (3, 3, 0), (4, 4, 0)])).unwrap();
        t.flush().unwrap();

        let mut keys = Vec::new();
        let p = pred(0, BinaryOp::LtEq, Value::UInt(2), DataType::UInt64);
        assert_eq!(t.delete_where_keys(&[0], Some(&p), &[], Some(&mut keys)).unwrap(), 2);
        keys.sort_unstable();
        assert_eq!(keys, vec![1, 2], "primary-key lanes, not positions");

        // Overlapping re-run: only the newly hidden row is reported.
        let mut keys = Vec::new();
        let p = pred(0, BinaryOp::LtEq, Value::UInt(3), DataType::UInt64);
        assert_eq!(t.delete_where_keys(&[0], Some(&p), &[], Some(&mut keys)).unwrap(), 1);
        assert_eq!(keys, vec![3]);
    }

}
