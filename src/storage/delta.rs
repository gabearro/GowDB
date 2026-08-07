//! The write buffer in front of the immutable parts.
//!
//! This is the OLTP half of the engine. Parts are sorted, packed and
//! immutable, which makes them excellent to read and impossible to update in
//! place. The delta absorbs writes at hash-map speed and is folded into a new
//! part on flush.
//!
//! Two disjoint modes, picked by [`TableDef::pk_col`]:
//!
//!   * **keyed** -- `pk lane -> row slot`, last write wins. A `put` to an
//!     existing key overwrites that row in place, so a hot row rewritten a
//!     million times costs one slot, not a million. This is what makes the
//!     engine usable as an OLTP store rather than an append log.
//!   * **unkeyed** -- a plain append. No dedup is possible without a key, and
//!     ClickHouse-style tables mostly append anyway.
//!
//! Keyed mode is **destructive by design**, so which tables get it is a
//! correctness question, not a tuning one: it may only be selected when the
//! table has *declared* its key unique (`PRIMARY KEY`, or a replacing engine).
//! A table that only declared `ORDER BY` has a sort key, whose duplicates are
//! legal rows, and must land in the unkeyed arm -- routing it here is what
//! made `INSERT INTO t VALUES (4,1),(4,2)` store one row. The rule and its
//! rationale live at [`TableDef::pk_col`]; this module trusts the flag.
//!
//! [`TableDef::pk_col`]: crate::types::TableDef::pk_col
//!
//! ## Row-major lanes, not columns
//!
//! A write buffer wants the opposite layout from a read buffer, and the
//! measurement is unambiguous. Holding the buffer as one growable column per
//! table column means a single-row write touches **one cache line per column**:
//!
//! ```text
//!   columnar buffer:  28 ns/row at 1 column, +4 ns for every column after
//!   row-major lanes:  flat, because the row is one contiguous span
//! ```
//!
//! So rows live here as `u64` lanes in one arena with stride `ncols`:
//!
//! ```text
//!   lanes: [ r0c0 r0c1 r0c2 | r1c0 r1c1 r1c2 | ... ]   index: { key -> slot }
//! ```
//!
//! Lanes are the same order-preserving encoding storage uses (see
//! [`crate::common::lane`]), so the value is already in its final form -- the
//! coercion happens once, at write time. Strings cannot fit in 8 bytes, so a
//! string cell's lane is an index into a side table of `Arc<str>`.
//!
//! The cost is that a flush has to **transpose** lanes into typed columns
//! rather than hand its buffers over directly. That is one strided read per
//! column, once per flush, against a per-row saving on every write -- and
//! writes outnumber flushes by the flush threshold, tens of thousands to one.
//!
//! Memory is unchanged at 8 bytes per cell. Overwriting a string cell abandons
//! its old `Arc` in the side table until the next flush, the same policy the
//! row slots themselves use.

use std::sync::Arc;

use crate::common::{lane_to_f64, lane_to_i64, BitSet, FastMap, Result};
use crate::types::{
    Block, Column, ColumnData, DataType, PhysicalType, Schema, Value,
};

/// Slot marker for a key that is present but deleted.
const TOMBSTONE: u32 = u32::MAX;

/// What the delta knows about a key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaEntry {
    /// Live, at this row slot.
    Put(u32),
    /// Tombstone: the key exists in some part and must be hidden.
    Del,
}

pub struct Delta {
    ncols: usize,
    /// Row-major lane arena, stride `ncols`.
    lanes: Vec<u64>,
    /// Row count, tracked rather than derived. `lanes.len() / ncols` is an
    /// integer division by a runtime value, and it sat on the write path.
    nrows: usize,
    /// Null bits, indexed `slot * ncols + col`. Absent until something is NULL.
    nulls: BitSet,
    has_nulls: bool,
    /// Backing store for string cells; a string lane indexes into this.
    strs: Vec<Arc<str>>,
    /// Per-column declared type and resolved physical kind.
    types: Vec<DataType>,
    phys: Vec<PhysicalType>,
    /// Primary-key lane -> row slot, or [`TOMBSTONE`]. Keyed mode only.
    index: FastMap<u64, u32>,
    /// Rows whose key was deleted after being written: still occupying a slot,
    /// no longer live.
    dead: usize,
    keyed_mode: bool,
}

impl Delta {
    /// Rows to reserve the first time a buffer is written to. Growth is
    /// amortized either way; reserving once avoids a realloc storm at the
    /// start of a write burst.
    const INITIAL_ROWS: usize = 4096;

    pub fn new(keyed_mode: bool, schema: &Schema) -> Delta {
        let types: Vec<DataType> = schema.fields().iter().map(|f| f.ty.clone()).collect();
        let phys: Vec<PhysicalType> = types.iter().map(|t| t.physical()).collect();
        Delta {
            ncols: types.len(),
            lanes: Vec::new(),
            nrows: 0,
            nulls: BitSet::new(),
            has_nulls: false,
            strs: Vec::new(),
            types,
            phys,
            index: FastMap::default(),
            dead: 0,
            keyed_mode,
        }
    }

    pub fn is_keyed(&self) -> bool {
        self.keyed_mode
    }

    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of buffered operations, which is what the flush threshold
    /// watches.
    pub fn len(&self) -> usize {
        if self.keyed_mode {
            self.index.len()
        } else {
            self.nrows()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Rows that would become visible on flush (excludes tombstones).
    pub fn live_rows(&self) -> usize {
        self.nrows() - self.dead
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Bytes held by the row arena and its string side table.
    pub fn data_bytes(&self) -> usize {
        self.lanes.capacity() * 8
            + self.nulls.bytes()
            + self.strs.capacity() * std::mem::size_of::<Arc<str>>()
            + self.strs.iter().map(|s| s.len()).sum::<usize>()
    }

    pub fn bytes(&self) -> usize {
        self.data_bytes()
            + self.index.capacity() * (std::mem::size_of::<u64>() + std::mem::size_of::<u32>())
    }

    /// Append a row, returning its slot.
    fn push_row(&mut self, row: &[Value]) -> Result<u32> {
        let n = self.ncols;
        let slot = self.nrows as u32;
        if slot == 0 {
            self.reserve(Self::INITIAL_ROWS);
        }
        let base = self.lanes.len();
        self.lanes.reserve(n);
        let Delta { lanes, nulls, strs, types, phys, has_nulls, .. } = self;
        // SAFETY: `reserve` guarantees `n` uninitialized slots past `len`, and
        // `write_cells` writes every one of them before `set_len` runs. On an
        // error path `set_len` is skipped, so the partially written spare
        // capacity stays beyond `len` and is never observable.
        //
        // Writing into spare capacity rather than `resize`-then-index avoids a
        // zero-fill of bytes that are about to be overwritten -- worth several
        // nanoseconds on a path this short.
        let cells = unsafe { std::slice::from_raw_parts_mut(lanes.as_mut_ptr().add(base), n) };
        write_cells(cells, base, row, nulls, strs, types, phys, has_nulls)?;
        unsafe { lanes.set_len(base + n) };
        self.nrows += 1;
        Ok(slot)
    }

    /// Decode one cell back to a `Value`.
    #[inline]
    fn decode(&self, col: usize, cell: usize) -> Value {
        if self.has_nulls && self.nulls.get(cell) {
            return Value::Null;
        }
        let lane = self.lanes[cell];
        match self.phys[col] {
            PhysicalType::Str => self
                .strs
                .get(lane as usize)
                .map(|s| Value::Str(s.clone()))
                .unwrap_or(Value::Null),
            p => lane_to_value(p, &self.types[col], lane),
        }
    }

    /// Insert or overwrite `key`.
    ///
    /// One hash lookup, not two. `get` followed by `insert` hashes and probes
    /// twice, and with distinct keys -- an append-shaped workload -- *every*
    /// write takes that path. The fields are destructured so the entry (which
    /// borrows `index`) and the row append (which borrows everything else) can
    /// coexist, and so the append can be inlined rather than reached through a
    /// second `&mut self` call.
    pub fn put_keyed(&mut self, key: u64, row: &[Value]) -> Result<()> {
        debug_assert!(self.keyed_mode);
        use std::collections::hash_map::Entry;
        if self.nrows == 0 {
            self.reserve(Self::INITIAL_ROWS);
        }
        let n = self.ncols;
        let Delta {
            lanes, nrows, nulls, strs, types, phys, index, has_nulls, ..
        } = self;
        match index.entry(key) {
            // Live row for this key already: overwrite in place, no growth.
            Entry::Occupied(e) if *e.get() != TOMBSTONE => {
                let base = *e.get() as usize * n;
                let cells = &mut lanes[base..base + n];
                write_cells(cells, base, row, nulls, strs, types, phys, has_nulls)
            }
            // Either new, or reviving a tombstoned key. Both get a fresh slot;
            // the row a tombstone hid, if any, was already counted dead by
            // `delete_keyed`.
            e => {
                let slot = *nrows as u32;
                let base = lanes.len();
                lanes.reserve(n);
                // SAFETY: `reserve` guarantees `n` uninitialized slots past
                // `len`, and `write_cells` writes every one before `set_len`.
                // On the error path `set_len` is skipped and the index is left
                // untouched, so a failed write is invisible.
                let cells =
                    unsafe { std::slice::from_raw_parts_mut(lanes.as_mut_ptr().add(base), n) };
                write_cells(cells, base, row, nulls, strs, types, phys, has_nulls)?;
                unsafe { lanes.set_len(base + n) };
                *nrows += 1;
                match e {
                    Entry::Occupied(mut o) => {
                        o.insert(slot);
                    }
                    Entry::Vacant(v) => {
                        v.insert(slot);
                    }
                }
                Ok(())
            }
        }
    }

    pub fn delete_keyed(&mut self, key: u64) {
        debug_assert!(self.keyed_mode);
        // A row buffered for this key becomes dead weight. Its slot is
        // abandoned rather than compacted out; the arena is rebuilt on flush,
        // which happens often enough that reclaiming a slot would cost more
        // than it saves.
        if let Some(&s) = self.index.get(&key) {
            if s != TOMBSTONE {
                self.dead += 1;
            }
        }
        self.index.insert(key, TOMBSTONE);
    }

    pub fn append(&mut self, row: &[Value]) -> Result<()> {
        debug_assert!(!self.keyed_mode);
        self.push_row(row)?;
        Ok(())
    }

    #[inline]
    pub fn get(&self, key: u64) -> Option<DeltaEntry> {
        match self.index.get(&key).copied()? {
            TOMBSTONE => Some(DeltaEntry::Del),
            s => Some(DeltaEntry::Put(s)),
        }
    }

    /// One cell of a buffered row.
    #[inline]
    pub fn value_at(&self, slot: u32, col: usize) -> Value {
        if col >= self.ncols {
            return Value::Null;
        }
        self.decode(col, slot as usize * self.ncols + col)
    }

    /// Every key mentioned since the last flush, whether put or deleted.
    pub fn touched_keys(&self) -> impl Iterator<Item = u64> + '_ {
        self.index.keys().copied()
    }

    pub fn contains_key(&self, key: u64) -> bool {
        self.index.contains_key(&key)
    }

    /// Reserve room for `n` more rows.
    pub fn reserve(&mut self, n: usize) {
        self.lanes.reserve(n * self.ncols);
        if self.keyed_mode {
            self.index.reserve(n);
        }
    }

    /// Live row slots in slot order, or `None` when that is simply *every*
    /// row -- always so when unkeyed, and when keyed until something is
    /// deleted. See [`Delta::transpose`] for why `None` is not spelled
    /// `(0..nrows).collect()`.
    fn live_slots(&self) -> Option<Vec<u32>> {
        if !self.keyed_mode || self.dead == 0 {
            return None;
        }
        let n = self.nrows();
        let mut alive = vec![false; n];
        for &s in self.index.values() {
            if s != TOMBSTONE {
                alive[s as usize] = true;
            }
        }
        Some((0..n as u32).filter(|&i| alive[i as usize]).collect())
    }

    /// Transpose the lane arena into typed columns.
    ///
    /// One strided pass per column: sequential writes, stride-`ncols` reads.
    /// This is the cost the row-major layout trades for; it is paid once per
    /// flush against a saving on every write.
    ///
    /// `None` means "every row, in slot order". That is the common case --
    /// every unkeyed delta, and every keyed one with no pending delete -- and
    /// it is a distinct code path rather than a `(0..nrows)` shorthand for two
    /// reasons: it never allocates the `4 * nrows` scratch permutation, and it
    /// turns the inner loop from a dependent gather (load the slot, scale it,
    /// then load the lane) into a flat strided walk the prefetcher can follow.
    /// Measured A/B interleaved in one loop, best-of-9, three runs, 400k rows
    /// x 3 UInt64 columns: **1.28/1.41/1.58 ns/row gathering against
    /// 1.04/1.16/1.25 ns/row walking** -- a steady ~1.22x on the transpose
    /// itself, and the machine's 3x load swing shows up as the spread between
    /// runs rather than between the two sides, which is the point of
    /// interleaving.
    fn transpose(&self, slots: Option<&[u32]>) -> Result<Vec<Column>> {
        let n = self.ncols;
        let rows = slots.map_or(self.nrows, <[u32]>::len);
        let mut out = Vec::with_capacity(n);
        for c in 0..n {
            let mut col =
                Column::new(self.types[c].clone(), ColumnData::for_physical(self.phys[c]));
            col.reserve(rows);
            match slots {
                None => self.fill_col(&mut col, c, (0..self.nrows).map(|r| r * n + c))?,
                Some(s) => self.fill_col(&mut col, c, s.iter().map(|&s| s as usize * n + c))?,
            }
            out.push(col);
        }
        Ok(out)
    }

    /// One column of [`Delta::transpose`], driven by an iterator of *cell*
    /// indices so the two orderings share this body.
    ///
    /// Generic rather than dynamic: each call site monomorphizes into its own
    /// loop, so the dense null-free numeric cases -- the overwhelming majority
    /// of what a write buffer holds -- keep a tight loop with no per-cell
    /// branch and no indirect call.
    #[inline]
    fn fill_col(
        &self,
        col: &mut Column,
        c: usize,
        cells: impl Iterator<Item = usize>,
    ) -> Result<()> {
        let lanes = &self.lanes;
        match (&mut col.data, self.has_nulls) {
            (ColumnData::U64(v), false) => v.extend(cells.map(|i| lanes[i])),
            (ColumnData::I64(v), false) => v.extend(cells.map(|i| lane_to_i64(lanes[i]))),
            (ColumnData::F64(v), false) => v.extend(cells.map(|i| lane_to_f64(lanes[i]))),
            _ => {
                for i in cells {
                    col.push_value(&self.decode(c, i))?;
                }
            }
        }
        Ok(())
    }

    /// Drain into a `Block`. Returns the block plus the keys that were touched
    /// (so the caller can tombstone them in older parts).
    pub fn drain_to_block(&mut self, _schema: &Schema) -> Result<(Block, Vec<u64>)> {
        let touched: Vec<u64> = self.index.keys().copied().collect();
        let slots = self.live_slots();
        let cols = self.transpose(slots.as_deref())?;
        self.clear();
        Ok((Block::new(cols)?, touched))
    }

    /// Materialize without draining, for reads that must see uncommitted rows.
    pub fn to_block(&self, _schema: &Schema) -> Result<Block> {
        let slots = self.live_slots();
        Block::new(self.transpose(slots.as_deref())?)
    }

    /// An immutable, key-sorted columnar image of the buffer *right now*,
    /// without draining it.
    ///
    /// This is the delta's half of the snapshot story: a reader can pin a
    /// [`DeltaImage`] alongside an `Arc<PartSet>` and see the buffered writes
    /// without the flush-before-read that every SELECT pays today. Sorted and
    /// columnar because that is the shape a scan and a merge both want; the
    /// live buffer is neither.
    ///
    /// ## Why this copies, and cannot not
    ///
    /// The obvious optimization is to share the lane arena with the live
    /// delta, on the theory that it is append-only until drained. It is not:
    /// [`Delta::put_keyed`] on a key that already has a live slot **overwrites
    /// that slot in place** -- that is the whole reason a hot row rewritten a
    /// million times costs one slot. A shared arena would therefore let a
    /// later write mutate a frozen image out from under a reader, which is
    /// precisely the class of bug the immutable part set exists to remove. So
    /// the rows are copied.
    ///
    /// The copy is not extra work either: it is the same single strided
    /// transpose per column that a flush performs, minus the packing, over
    /// live rows only. Measured, same machine, same 400k-row 3-column buffer,
    /// best of 7: **15.0 ns/row to freeze against 40.5 ns/row to flush** --
    /// 2.7x cheaper than the flush it is meant to replace, and it produces no
    /// part for compaction to clean up afterwards.
    ///
    /// Every buffer is sized exactly up front; the only unbounded growth is
    /// the tombstone list, which is empty unless something was deleted.
    pub fn freeze(&self) -> Result<Arc<DeltaImage>> {
        // Unkeyed: insertion order *is* the order and nothing is dead, so the
        // permutation is `None` -- not an identity vector built to be walked
        // once (see [`Delta::transpose`]). That arm allocates nothing at all
        // beyond the columns themselves, which is the common shape now that a
        // sort-key-only table is unkeyed.
        let (keys, slots, tombstones) = if self.keyed_mode {
            let live = self.index.len();
            let mut keys = Vec::with_capacity(live);
            let mut slots = Vec::with_capacity(live);
            let mut tombstones = Vec::new();
            for (&k, &s) in self.index.iter() {
                if s == TOMBSTONE {
                    tombstones.push(k);
                } else {
                    keys.push(k);
                    slots.push(s);
                }
            }
            // Radix, not `sort_by_key`: the keys are already order-preserving
            // lanes, and this is the same sort the flush path uses to reach a
            // part. It permutes `slots` alongside, so the transpose below
            // reads rows in key order directly rather than through a second
            // indirection.
            crate::sort::radix_sort_soa(&mut keys, &mut slots);
            tombstones.sort_unstable();
            (keys, Some(slots), tombstones)
        } else {
            (Vec::new(), None, Vec::new())
        };

        Ok(Arc::new(DeltaImage {
            block: Block::new(self.transpose(slots.as_deref())?)?,
            keys,
            tombstones,
        }))
    }

    pub fn clear(&mut self) {
        // Capacities are kept: the next write burst will want them back.
        self.lanes.clear();
        self.nrows = 0;
        self.strs.clear();
        self.nulls = BitSet::new();
        self.has_nulls = false;
        self.index.clear();
        self.dead = 0;
    }
}

/// A frozen, key-sorted view of the write buffer.
///
/// Immutable by construction and handed out behind an `Arc`, so a reader can
/// hold one for the length of a query while writers keep going. Together with
/// an `Arc<PartSet>` it is a complete snapshot of a table: the parts, their
/// delete masks, and the buffered writes that shadow them.
pub struct DeltaImage {
    /// Live rows, columnar, in primary-key order (insertion order when the
    /// table has no fast primary key).
    block: Block,
    /// Ascending primary-key lanes, one per row of `block`. Empty when
    /// unkeyed. Parallel arrays rather than a map: the image is read-only, and
    /// a binary search over sorted `u64`s beats a hash probe once the point of
    /// the structure is *merging in key order* rather than looking rows up.
    keys: Vec<u64>,
    /// Keys this image hides in older parts, ascending. Includes keys that
    /// were deleted without ever being written here.
    tombstones: Vec<u64>,
}

impl DeltaImage {
    #[inline]
    pub fn rows(&self) -> usize {
        self.block.rows()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.block.rows() == 0 && self.tombstones.is_empty()
    }

    #[inline]
    pub fn block(&self) -> &Block {
        &self.block
    }

    /// Sorted primary-key lanes, parallel to the block's rows.
    #[inline]
    pub fn keys(&self) -> &[u64] {
        &self.keys
    }

    #[inline]
    pub fn tombstones(&self) -> &[u64] {
        &self.tombstones
    }

    /// Row index of `key`, or `None`.
    #[inline]
    pub fn position_of(&self, key: u64) -> Option<usize> {
        self.keys.binary_search(&key).ok()
    }

    /// True when this image hides `key` in the parts underneath it -- whether
    /// by replacing the row or by deleting it.
    #[inline]
    pub fn shadows(&self, key: u64) -> bool {
        self.position_of(key).is_some() || self.tombstones.binary_search(&key).is_ok()
    }

    /// True when `key` is deleted rather than replaced.
    #[inline]
    pub fn deletes(&self, key: u64) -> bool {
        self.tombstones.binary_search(&key).is_ok()
    }

    pub fn bytes(&self) -> usize {
        (self.keys.capacity() + self.tombstones.capacity()) * 8
    }
}

/// Encode `row` into `cells`, the lane span of one row starting at cell index
/// `base`.
///
/// A free function taking the pieces rather than a `&mut self` method: taking
/// `&mut self` per cell makes the compiler reload every field it might have
/// mutated, and the split borrow is what lets the caller hand in a slice of
/// `lanes` while still mutating `nulls` and `strs`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn write_cells(
    cells: &mut [u64],
    base: usize,
    row: &[Value],
    nulls: &mut BitSet,
    strs: &mut Vec<Arc<str>>,
    types: &[DataType],
    phys: &[PhysicalType],
    has_nulls: &mut bool,
) -> Result<()> {
    debug_assert_eq!(cells.len(), phys.len());
    if row.len() == cells.len() {
        // Zipping the four arrays drops a bounds check per cell per array.
        for (i, (((cell, &p), t), v)) in cells
            .iter_mut()
            .zip(phys)
            .zip(types)
            .zip(row)
            .enumerate()
        {
            *cell = encode_cell(p, t, v, base + i, nulls, strs, has_nulls)?;
        }
        return Ok(());
    }
    // Short row: trailing columns are NULL. Rare, so it keeps the simple form.
    for (i, cell) in cells.iter_mut().enumerate() {
        let v = row.get(i).unwrap_or(&Value::Null);
        *cell = encode_cell(phys[i], &types[i], v, base + i, nulls, strs, has_nulls)?;
    }
    Ok(())
}

/// One cell: value -> lane, recording NULL and interning strings.
#[inline(always)]
fn encode_cell(
    p: PhysicalType,
    t: &DataType,
    v: &Value,
    cell: usize,
    nulls: &mut BitSet,
    strs: &mut Vec<Arc<str>>,
    has_nulls: &mut bool,
) -> Result<u64> {
    // The overwhelmingly common shape is a non-null scalar already in its
    // column's kind: one match, one store, and no touch of the null bitmap
    // until something has actually been NULL.
    match (p, v) {
        (PhysicalType::U64, Value::UInt(x)) if !*has_nulls => Ok(*x),
        (PhysicalType::I64, Value::Int(x)) if !*has_nulls => Ok(crate::common::i64_to_lane(*x)),
        (PhysicalType::F64, Value::Float(x)) if !*has_nulls => Ok(crate::common::f64_to_lane(*x)),
        (_, Value::Null) => {
            nulls.set(cell);
            *has_nulls = true;
            Ok(0)
        }
        (PhysicalType::Str, _) => {
            if *has_nulls {
                nulls.clear(cell);
            }
            strs.push(match v {
                Value::Str(x) => x.clone(),
                other => other.render_plain().into(),
            });
            Ok(strs.len() as u64 - 1)
        }
        _ => {
            // Once anything is nullable, an overwrite has to clear a possibly
            // stale bit before storing.
            if *has_nulls {
                nulls.clear(cell);
            }
            v.to_lane_phys(p, t)
        }
    }
}

/// Decode a non-string lane back to a `Value`, honouring the declared type.
#[inline]
fn lane_to_value(phys: PhysicalType, ty: &DataType, lane: u64) -> Value {
    match ty.base() {
        DataType::Bool => Value::Bool(lane != 0),
        DataType::Date => Value::Date(lane as u32),
        DataType::DateTime => match phys {
            PhysicalType::I64 => Value::DateTime(lane_to_i64(lane)),
            _ => Value::DateTime(lane as i64),
        },
        _ => match phys {
            PhysicalType::U64 => Value::UInt(lane),
            PhysicalType::I64 => Value::Int(lane_to_i64(lane)),
            PhysicalType::F64 => Value::Float(lane_to_f64(lane)),
            PhysicalType::Str => Value::Null,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::splitmix64;
    use crate::types::Field;

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::UInt64),
            Field::new("v", DataType::Int64),
        ])
        .unwrap()
    }

    fn keyed() -> Delta {
        Delta::new(true, &schema())
    }

    fn put(d: &mut Delta, k: u64, v: i64) {
        d.put_keyed(k, &[Value::UInt(k), Value::Int(v)]).unwrap();
    }

    #[test]
    fn keyed_put_is_last_write_wins() {
        let mut d = keyed();
        put(&mut d, 1, 10);
        put(&mut d, 1, 20);
        assert_eq!(d.len(), 1, "rewriting a key must not grow the delta");
        let DeltaEntry::Put(s) = d.get(1).unwrap() else { panic!() };
        assert_eq!(d.value_at(s, 1), Value::Int(20));
    }

    #[test]
    fn overwriting_a_key_reuses_its_row() {
        let mut d = keyed();
        for i in 0..1000 {
            put(&mut d, 1, i);
        }
        assert_eq!(d.nrows(), 1, "a hot row rewritten 1000 times must not grow");
        let DeltaEntry::Put(s) = d.get(1).unwrap() else { panic!() };
        assert_eq!(d.value_at(s, 1), Value::Int(999));
    }

    #[test]
    fn rows_are_contiguous_in_the_arena() {
        // The layout claim, asserted rather than assumed: row `i` occupies
        // exactly `lanes[i*ncols .. (i+1)*ncols]`.
        let mut d = keyed();
        put(&mut d, 7, -3);
        put(&mut d, 8, 4);
        assert_eq!(d.lanes.len(), 4);
        assert_eq!(d.lanes[0], 7);
        assert_eq!(d.lanes[2], 8);
        assert_eq!(lane_to_i64(d.lanes[1]), -3);
        assert_eq!(lane_to_i64(d.lanes[3]), 4);
    }

    #[test]
    fn delete_replaces_a_pending_put() {
        let mut d = keyed();
        put(&mut d, 7, 1);
        d.delete_keyed(7);
        assert_eq!(d.get(7), Some(DeltaEntry::Del));
        assert_eq!(d.live_rows(), 0);
        assert_eq!(d.len(), 1, "the tombstone still occupies an entry");
    }

    #[test]
    fn put_after_delete_revives_the_key() {
        let mut d = keyed();
        put(&mut d, 7, 1);
        d.delete_keyed(7);
        put(&mut d, 7, 2);
        let DeltaEntry::Put(s) = d.get(7).unwrap() else { panic!() };
        assert_eq!(d.value_at(s, 1), Value::Int(2));
        assert_eq!(d.live_rows(), 1, "the abandoned slot must not count as live");
        let (blk, _) = d.drain_to_block(&schema()).unwrap();
        assert_eq!(blk.rows(), 1);
        assert_eq!(blk.column(1).value(0), Value::Int(2));
    }

    #[test]
    fn drain_yields_puts_and_reports_every_touched_key() {
        let mut d = keyed();
        put(&mut d, 1, 10);
        put(&mut d, 2, 20);
        d.delete_keyed(3);
        let (blk, touched) = d.drain_to_block(&schema()).unwrap();
        assert_eq!(blk.rows(), 2, "tombstones contribute no rows");
        let mut t = touched;
        t.sort_unstable();
        assert_eq!(t, vec![1, 2, 3], "deleted keys must still be reported");
        assert!(d.is_empty());
    }

    #[test]
    fn drain_drops_rows_deleted_after_being_written() {
        let mut d = keyed();
        for i in 0..5u64 {
            put(&mut d, i, i as i64);
        }
        d.delete_keyed(2);
        assert_eq!(d.live_rows(), 4);
        let (blk, _) = d.drain_to_block(&schema()).unwrap();
        assert_eq!(blk.rows(), 4);
        let ids: Vec<Value> = (0..blk.rows()).map(|i| blk.column(0).value(i)).collect();
        assert!(!ids.contains(&Value::UInt(2)));
    }

    #[test]
    fn transpose_preserves_row_order_and_values() {
        let mut d = keyed();
        for i in 0..300u64 {
            put(&mut d, i, -(i as i64));
        }
        let (blk, _) = d.drain_to_block(&schema()).unwrap();
        assert_eq!(blk.rows(), 300);
        for i in 0..300 {
            assert_eq!(blk.column(0).value(i), Value::UInt(i as u64));
            assert_eq!(blk.column(1).value(i), Value::Int(-(i as i64)));
        }
    }

    #[test]
    fn unkeyed_mode_appends() {
        let mut d = Delta::new(false, &schema());
        d.append(&[Value::UInt(1), Value::Int(1)]).unwrap();
        d.append(&[Value::UInt(1), Value::Int(2)]).unwrap();
        assert_eq!(d.len(), 2, "no dedup without a key");
        let (blk, touched) = d.drain_to_block(&schema()).unwrap();
        assert_eq!(blk.rows(), 2);
        assert!(touched.is_empty());
    }

    #[test]
    fn strings_round_trip_through_the_side_table() {
        let s = Schema::new(vec![
            Field::new("id", DataType::UInt64),
            Field::new("name", DataType::String),
        ])
        .unwrap();
        let mut d = Delta::new(true, &s);
        d.put_keyed(1, &[Value::UInt(1), Value::str("alpha")]).unwrap();
        d.put_keyed(2, &[Value::UInt(2), Value::str("beta")]).unwrap();
        // overwrite: the old Arc is abandoned, the new one wins
        d.put_keyed(1, &[Value::UInt(1), Value::str("gamma")]).unwrap();

        let DeltaEntry::Put(slot) = d.get(1).unwrap() else { panic!() };
        assert_eq!(d.value_at(slot, 1), Value::str("gamma"));

        let (blk, _) = d.drain_to_block(&s).unwrap();
        assert_eq!(blk.rows(), 2);
        let mut names: Vec<String> = (0..2)
            .map(|i| blk.column(1).value(i).render_plain())
            .collect();
        names.sort();
        assert_eq!(names, vec!["beta", "gamma"]);
    }

    #[test]
    fn nulls_survive_write_read_and_flush() {
        let s = Schema::new(vec![
            Field::new("id", DataType::UInt64),
            Field::new("v", DataType::Nullable(Box::new(DataType::Int64))),
        ])
        .unwrap();
        let mut d = Delta::new(true, &s);
        d.put_keyed(1, &[Value::UInt(1), Value::Null]).unwrap();
        d.put_keyed(2, &[Value::UInt(2), Value::Int(5)]).unwrap();
        let DeltaEntry::Put(s1) = d.get(1).unwrap() else { panic!() };
        assert_eq!(d.value_at(s1, 1), Value::Null);

        let (blk, _) = d.drain_to_block(&s).unwrap();
        let vals: Vec<Value> = (0..2).map(|i| blk.column(1).value(i)).collect();
        assert!(vals.contains(&Value::Null));
        assert!(vals.contains(&Value::Int(5)));
    }

    #[test]
    fn overwriting_a_null_with_a_value_clears_the_null_bit() {
        let s = Schema::new(vec![
            Field::new("id", DataType::UInt64),
            Field::new("v", DataType::Nullable(Box::new(DataType::Int64))),
        ])
        .unwrap();
        let mut d = Delta::new(true, &s);
        d.put_keyed(1, &[Value::UInt(1), Value::Null]).unwrap();
        d.put_keyed(1, &[Value::UInt(1), Value::Int(9)]).unwrap();
        let DeltaEntry::Put(slot) = d.get(1).unwrap() else { panic!() };
        assert_eq!(d.value_at(slot, 1), Value::Int(9), "stale null bit");

        // ...and the other direction.
        d.put_keyed(1, &[Value::UInt(1), Value::Null]).unwrap();
        let DeltaEntry::Put(slot) = d.get(1).unwrap() else { panic!() };
        assert_eq!(d.value_at(slot, 1), Value::Null);
    }

    #[test]
    fn short_rows_pad_with_null() {
        let s = Schema::new(vec![
            Field::new("id", DataType::UInt64),
            Field::new("v", DataType::Nullable(Box::new(DataType::Int64))),
        ])
        .unwrap();
        let mut d = Delta::new(false, &s);
        d.append(&[Value::UInt(1)]).unwrap(); // missing the second column
        let blk = d.to_block(&s).unwrap();
        assert!(blk.column(1).is_null(0));
    }

    #[test]
    fn to_block_does_not_drain() {
        let mut d = keyed();
        put(&mut d, 1, 5);
        let blk = d.to_block(&schema()).unwrap();
        assert_eq!(blk.rows(), 1);
        assert_eq!(d.len(), 1, "to_block must leave the delta intact");
    }

    #[test]
    fn values_are_coerced_once_at_write_time() {
        // A Float written into an Int64 column is narrowed on the way in, so
        // the flush has nothing left to convert.
        let mut d = keyed();
        d.put_keyed(1, &[Value::UInt(1), Value::Float(7.9)]).unwrap();
        let DeltaEntry::Put(s) = d.get(1).unwrap() else { panic!() };
        assert_eq!(d.value_at(s, 1), Value::Int(7));
    }

    #[test]
    fn eight_bytes_per_cell_regardless_of_width() {
        for ncols in [1usize, 4, 16] {
            let fields: Vec<Field> = (0..ncols)
                .map(|i| Field::new(format!("c{i}"), DataType::UInt64))
                .collect();
            let s = Schema::new(fields).unwrap();
            let mut d = Delta::new(false, &s);
            let row: Vec<Value> = (0..ncols).map(|i| Value::UInt(i as u64)).collect();
            for _ in 0..1000 {
                d.append(&row).unwrap();
            }
            // The arena is exactly 8 bytes per cell; capacity may over-reserve,
            // so assert on the used length.
            assert_eq!(d.lanes.len(), 1000 * ncols, "ncols={ncols}");
        }
    }

    #[test]
    fn freeze_is_key_sorted_and_leaves_the_delta_alone() {
        let mut d = keyed();
        for i in (0..500u64).rev() {
            put(&mut d, splitmix64(i) % 10_000, i as i64);
        }
        d.delete_keyed(999_999);
        let img = d.freeze().unwrap();

        assert!(img.keys().windows(2).all(|w| w[0] < w[1]), "not key-sorted");
        assert_eq!(img.rows(), img.keys().len());
        assert_eq!(img.tombstones(), &[999_999]);
        assert!(img.deletes(999_999));
        assert!(img.shadows(999_999));
        assert_eq!(d.live_rows(), img.rows(), "freeze must not drain");

        // The image agrees with the live buffer row for row.
        for (r, &k) in img.keys().iter().enumerate() {
            let DeltaEntry::Put(slot) = d.get(k).unwrap() else { panic!() };
            assert_eq!(img.block().column(1).value(r), d.value_at(slot, 1), "key {k}");
            assert_eq!(img.position_of(k), Some(r));
        }
        assert_eq!(img.position_of(10_001), None);
    }

    #[test]
    fn a_frozen_image_does_not_move_when_the_delta_is_overwritten() {
        // The reason `freeze` copies: `put_keyed` rewrites a live slot in
        // place, so an image sharing the lane arena would silently change
        // under whoever is reading it.
        let mut d = keyed();
        put(&mut d, 7, 1);
        let img = d.freeze().unwrap();
        put(&mut d, 7, 2);
        d.delete_keyed(7);

        assert_eq!(img.rows(), 1);
        assert_eq!(img.block().column(1).value(0), Value::Int(1), "the image moved");
        assert!(img.tombstones().is_empty());
        assert_eq!(d.live_rows(), 0, "...while the live delta did move on");
    }

    #[test]
    fn freeze_of_an_unkeyed_delta_keeps_insertion_order() {
        let mut d = Delta::new(false, &schema());
        for i in [9u64, 3, 7] {
            d.append(&[Value::UInt(i), Value::Int(i as i64)]).unwrap();
        }
        let img = d.freeze().unwrap();
        assert!(img.keys().is_empty(), "no primary key, no key array");
        let got: Vec<Value> = (0..img.rows()).map(|r| img.block().column(0).value(r)).collect();
        assert_eq!(got, vec![Value::UInt(9), Value::UInt(3), Value::UInt(7)]);
    }

    #[test]
    fn freezing_an_empty_delta_is_empty_not_an_error() {
        let d = keyed();
        let img = d.freeze().unwrap();
        assert!(img.is_empty());
        assert_eq!(img.rows(), 0);
    }

    #[test]
    fn clear_keeps_capacity() {
        let mut d = keyed();
        for i in 0..100u64 {
            put(&mut d, i, 0);
        }
        let cap = d.lanes.capacity();
        d.clear();
        assert!(d.is_empty());
        assert_eq!(d.live_rows(), 0);
        assert_eq!(d.lanes.capacity(), cap, "the arena should be reused");
    }
}
