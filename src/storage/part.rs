//! A part: an immutable, sorted, compressed run of rows.
//!
//! This is ClickHouse's "data part" -- the thing an INSERT produces and a
//! merge consumes. Rows inside a part are sorted by the table's ORDER BY key,
//! which is what makes every access path here cheap:
//!
//!   * **routing**: an O(1) bucket table maps a key straight to a small span
//!     of granules, leaving a bounded binary search instead of a full one;
//!   * **zone maps**: `sort_min`/`sort_max` per granule prune ranges without
//!     reading data;
//!   * **bloom**: one split-block filter per part skips a whole foreign part
//!     in a single cache-line probe, built lazily only once a second part
//!     exists (a freshly compacted single-part table stores none at all);
//!   * **deletes**: a [`Deletes`] slot held *outside* the part, in the
//!     [`PartSet`], so a clean part costs one null check and zero bytes.
//!
//! ## Immutability is load-bearing, not aspirational
//!
//! A published `Part` is never written to again. It used to be *nearly* true:
//! `mark_deleted` flipped a bit in an older part on every write that shadowed
//! a key, so an in-flight scan could see a row present in granule 3 and absent
//! in granule 7 of what was supposed to be one snapshot. Single-threaded `&mut
//! self` hid that; a concurrent reader would have seen a torn read, not merely
//! a stale one.
//!
//! So delete state lives in [`PartSet::deletes`], versioned alongside the part
//! list and replaced copy-on-write. What remains inside `Part` is the
//! *construction-time* image -- see [`Part::deleted`] -- which exists only so
//! the decoder can hand a part's on-disk deletes to whoever adopts it, and is
//! taken away by [`PartSet::adopt`] the moment the part is published.
//!
//! The one derived value a part computes lazily is its bloom filter, and it is
//! behind a [`OnceLock`]: a pure function of bytes that never change, memoized
//! race-free, so "immutable" still holds in the sense that matters.
//!
//! ## Immutability is also what makes an incremental checkpoint possible
//!
//! Because a published part's bytes never change, "is the copy on disk still
//! correct?" is answerable without hashing or comparing anything: it is
//! correct exactly while the part is the same object and its delete mask has
//! not moved. [`PartSet::origins`] records the answer -- the sequence number
//! of the part file that already holds these bytes -- and
//! `persist::writer::write_table` skips every part that still has one. Without
//! it a checkpoint is O(entire database) and a read-only query rewrites the
//! whole table.
//!
//! ## ...and what makes a positional log record possible
//!
//! [`Part::pid`] is the part's durable identity: one varint inside its own
//! file, minted once and written back unchanged by every rewrite. It exists
//! because an unkeyed `DELETE` has no key lane to log and must therefore log a
//! *position*, and a position needs something that pins the rows it indexes.
//! The file number cannot: the first tombstone retracts it and the next
//! checkpoint reissues the part under a different name. The identity can,
//! because a rewrite changes only the mask -- `persist::writer::write_part_with`
//! copies granules verbatim, so rows and their offsets are the same bytes
//! before and after, and only a merge removes rows (producing a new part with
//! a new identity). 8 bytes per part, none per row.
//!
//! ## Unlink safety
//!
//! Compaction removes a part's file as soon as the merged replacement is
//! committed, but a `Snapshot` taken before that still holds `Arc<Part>`s
//! whose packed lanes point straight into the mapping of a now-unlinked inode.
//! That is deliberate and it is why the design can drop files eagerly:
//! `munmap`, not `unlink`, is what tears the pages down, because the mapping
//! holds a reference to the inode itself. Two tests pin exactly this --
//! `persist::mmap::mapping_survives_the_file_being_unlinked` and
//! `persist::reader::an_unlinked_part_is_still_readable_through_its_mapping`
//! -- and the `Arc<Part>` reference count is what keeps the mapping alive long
//! enough for it to matter. `table::a_compacted_away_part_stays_alive_under_a_
//! snapshot` is the storage-side half of the same claim.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use crate::common::{BitSet, Result, GRANULE_SIZE, G_SHIFT};
use crate::index::PartFilter;
use crate::types::{Block, Column, ColumnBuilder, Schema, Value};

use super::granule::{Granule, Stats};

pub struct Part {
    pub n_rows: usize,
    /// This part's durable identity: the name a log record uses to say "row
    /// `pos` of *that* part is hidden", and the reason such a record survives
    /// the checkpoint that rewrites the part under a new file name.
    ///
    /// Minted by [`PartSet::push`] from the set's own counter, written into
    /// the part file by `persist::writer::part_meta`, and read straight back
    /// -- so a part loaded from disk knows what it is called without a sidecar
    /// and without a directory listing. It never changes, because a rewrite
    /// changes only the delete mask: `write_part_with` copies granules
    /// verbatim, so row positions and row identity are the same bytes before
    /// and after. Only a merge removes rows, and a merge produces a *new*
    /// part with a new identity.
    ///
    /// Zero means "not minted yet", which is every part between `build` and
    /// `push`. 8 bytes per part, none per row.
    pub pid: u64,
    pub granules: Vec<Granule>,
    /// Sort-key lane of the first row of each granule: the sparse index.
    first_keys: Vec<u64>,
    part_max: u64,
    /// **Construction-time** delete image, and nothing else.
    ///
    /// `Part::build` leaves it empty; `Part::from_parts` fills it with what the
    /// decoder read off disk. [`PartSet::adopt`] then *takes* it (leaving this
    /// empty) and it becomes the set's initial [`Deletes`]. A part that is
    /// already inside a `PartSet` therefore always reads `deleted_count == 0`
    /// here, and the live answer only ever comes from [`PartSet::deletes`].
    ///
    /// It stays a field rather than a constructor argument alone because the
    /// persistence layer builds, mutates and serializes parts that have never
    /// been published -- see `persist::writer::part_bytes`.
    pub deleted: BitSet,
    pub deleted_count: usize,
    /// **Construction-time** file provenance, carried for exactly as long as
    /// `deleted` is: the sequence number of the part file this was decoded
    /// from, or [`NO_FILE`] for a part built in memory. [`PartSet::push`]
    /// takes it into [`PartSet::origins`], where it becomes versioned state
    /// like the delete mask, and it is meaningless here after that.
    origin: u64,
    /// O(1) granule router: `key -> ((key - base) >> shift)` buckets
    /// bracketing the sparse-index answer.
    route_base: u64,
    route_shift: u32,
    router: Vec<u32>,
    /// Memoized bloom. `OnceLock` rather than `Option` + `&mut self`: the
    /// filter is a pure function of the packed key column, so building it is
    /// not a mutation of the part in any sense a reader can observe, and this
    /// is what lets the whole tombstoning path run off `&Part`.
    filter: OnceLock<PartFilter>,
    pub sort_col: Option<usize>,
    pub pk_col: Option<usize>,
    pub ncols: usize,
}

impl Part {
    /// Build from a block already sorted by `sort_col`.
    ///
    /// Granule construction is embarrassingly parallel -- pack, fingerprints
    /// and MPH all depend only on their own rows -- so it fans out across
    /// every core with no shared mutable state and an order-preserving join.
    pub fn build(
        block: &Block,
        sort_col: Option<usize>,
        pk_col: Option<usize>,
    ) -> Result<Part> {
        Part::build_sel(block, None, sort_col, pk_col)
    }

    /// Build from `block`, reading rows through `perm` when one is given.
    ///
    /// See [`Granule::build_sel`]: this is how a sort reaches storage without
    /// first materializing a sorted copy of every column.
    pub fn build_sel(
        block: &Block,
        perm: Option<&[u32]>,
        sort_col: Option<usize>,
        pk_col: Option<usize>,
    ) -> Result<Part> {
        let n = perm.map_or(block.rows(), |p| p.len());
        let ncols = block.width();
        let nchunks = n.div_ceil(GRANULE_SIZE);
        let ranges: Vec<(usize, usize)> = (0..nchunks)
            .map(|i| (i * GRANULE_SIZE, ((i + 1) * GRANULE_SIZE).min(n)))
            .collect();

        let nthreads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
            .min(ranges.len().max(1));

        let granules: Vec<Granule> = if nthreads > 1 && ranges.len() >= 8 {
            let per = ranges.len().div_ceil(nthreads);
            std::thread::scope(|s| -> Result<Vec<Granule>> {
                let handles: Vec<_> = ranges
                    .chunks(per)
                    .map(|part| {
                        s.spawn(move || -> Result<Vec<Granule>> {
                            part.iter()
                                .map(|&(a, b)| Granule::build_sel(block, a, b, perm, sort_col, pk_col))
                                .collect()
                        })
                    })
                    .collect();
                let mut out = Vec::with_capacity(ranges.len());
                for h in handles {
                    out.extend(h.join().expect("granule build thread panicked")?);
                }
                Ok(out)
            })?
        } else {
            ranges
                .iter()
                .map(|&(a, b)| Granule::build_sel(block, a, b, perm, sort_col, pk_col))
                .collect::<Result<_>>()?
        };

        let first_keys: Vec<u64> = granules.iter().map(|g| g.sort_min).collect();
        let part_max = granules.last().map(|g| g.sort_max).unwrap_or(0);
        let (route_base, route_shift, router) = Self::build_router(&first_keys, part_max);

        Ok(Part {
            n_rows: n,
            pid: 0,
            granules,
            first_keys,
            part_max,
            deleted: BitSet::new(),
            deleted_count: 0,
            origin: NO_FILE,
            route_base,
            route_shift,
            router,
            filter: OnceLock::new(),
            sort_col,
            pk_col,
            ncols,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        n_rows: usize,
        granules: Vec<Granule>,
        deleted: BitSet,
        deleted_count: usize,
        sort_col: Option<usize>,
        pk_col: Option<usize>,
        ncols: usize,
    ) -> Part {
        let first_keys: Vec<u64> = granules.iter().map(|g| g.sort_min).collect();
        let part_max = granules.last().map(|g| g.sort_max).unwrap_or(0);
        let (route_base, route_shift, router) = Self::build_router(&first_keys, part_max);
        Part {
            n_rows,
            pid: 0,
            granules,
            first_keys,
            part_max,
            deleted,
            deleted_count,
            origin: NO_FILE,
            route_base,
            route_shift,
            router,
            filter: OnceLock::new(),
            sort_col,
            pk_col,
            ncols,
        }
    }

    /// Record which part file this was decoded from, before it is published.
    ///
    /// The loader's half of the incremental checkpoint: without it every part
    /// read back from disk looks new to the next `write_table`, and the first
    /// checkpoint after a restart rewrites the entire database once more.
    /// Only meaningful between decoding and [`PartSet::push`].
    ///
    /// It is also what marks the part [`HOMED`]: it was decoded from a file,
    /// so recovery will decode it from a file again.
    pub fn set_origin(&mut self, seq: u64) {
        self.origin = HOMED | seq;
    }

    fn build_router(first_keys: &[u64], part_max: u64) -> (u64, u32, Vec<u32>) {
        let ngran = first_keys.len();
        if ngran == 0 {
            return (0, 63, vec![0, 0]);
        }
        let base = first_keys[0];
        let range = part_max.saturating_sub(base);
        let nb = (ngran * 2).next_power_of_two().clamp(2, 1 << 20);
        let rb = 64 - range.leading_zeros();
        let shift = rb.saturating_sub(nb.trailing_zeros());
        let mut router = Vec::with_capacity(nb + 1);
        for b in 0..=nb as u128 {
            let start = base.saturating_add((b << shift).min(u64::MAX as u128) as u64);
            router.push(first_keys.partition_point(|&fk| fk <= start) as u32);
        }
        (base, shift, router)
    }

    /// Router bucket + bounded binary search = granule index for `lane`,
    /// or `None` if it precedes the part.
    #[inline(always)]
    pub fn locate_granule(&self, lane: u64) -> Option<usize> {
        if lane < self.route_base {
            return None;
        }
        let idx =
            (((lane - self.route_base) >> self.route_shift) as usize).min(self.router.len() - 2);
        let (lo, hi) = unsafe {
            (
                *self.router.get_unchecked(idx) as usize,
                *self.router.get_unchecked(idx + 1) as usize,
            )
        };
        let p = lo + self.first_keys[lo..hi].partition_point(|&fk| fk <= lane);
        if p == 0 {
            None
        } else {
            Some(p - 1)
        }
    }

    /// Row holding this primary-key lane, ignoring deletes, or `None`.
    ///
    /// The delete check is the caller's, because the caller is the only one
    /// holding a snapshot's [`Deletes`] -- and because a clean part (the
    /// overwhelming majority) then pays nothing at all for it.
    #[inline]
    pub fn find(&self, key: u64, fph: u64, stats: &mut Stats) -> Option<usize> {
        let pk = self.pk_col?;
        if key > self.part_max {
            return None;
        }
        let gi = self.locate_granule(key)?;
        let g = unsafe { self.granules.get_unchecked(gi) };
        if key > g.sort_max {
            stats.zone_pruned_point += 1;
            return None;
        }
        Some((gi << G_SHIFT) + g.find_key(pk, key, fph, stats)?)
    }

    /// Position of a *live* row with this primary-key lane, or `None`.
    #[inline]
    pub fn find_live(
        &self,
        key: u64,
        fph: u64,
        stats: &mut Stats,
        del: Option<&Deletes>,
    ) -> Option<usize> {
        let pos = self.find(key, fph, stats)?;
        // One predictable null check for a clean part; a bitmap probe only
        // where something has actually been deleted.
        match del {
            Some(d) if d.get(pos) => None,
            _ => Some(pos),
        }
    }

    /// Mark a row deleted **before the part is published**.
    ///
    /// The only legal callers are the ones that still own the part outright:
    /// the decoder assembling a part from disk, and the tests that build one.
    /// Once [`PartSet::adopt`] has taken the image this cannot be reached --
    /// a published part is behind an `Arc` and there is no `&mut` to be had.
    pub fn mark_deleted(&mut self, pos: usize) -> bool {
        if self.deleted.set(pos) {
            self.deleted_count += 1;
            true
        } else {
            false
        }
    }

    /// The part-level bloom, built on first use and kept forever.
    ///
    /// Decoding the packed key column costs a pass over the part, so this
    /// stays lazy: a single-part table never probes a bloom (`may_contain` is
    /// only consulted when there is a *foreign* part to skip) and so never
    /// pays for one.
    pub fn ensure_filter(&self) -> Option<&PartFilter> {
        let pk = self.pk_col?;
        Some(self.filter.get_or_init(|| {
            let mut f = PartFilter::new(self.n_rows.max(1));
            for g in &self.granules {
                let col = &g.columns[pk];
                for i in 0..g.len {
                    f.insert(col.lane(i));
                }
            }
            f
        }))
    }

    /// True when the part *may* hold this key. "No filter yet" means "maybe":
    /// this is the probe, never the builder, so it stays branch-cheap.
    #[inline(always)]
    pub fn may_contain(&self, fph: u64) -> bool {
        self.filter.get().is_none_or(|f| f.contains_hash(fph))
    }

    pub fn has_filter(&self) -> bool {
        self.filter.get().is_some()
    }

    pub fn granule_count(&self) -> usize {
        self.granules.len()
    }

    /// Live row indices within granule `gi`, written into `out`.
    ///
    /// `None` means "every row is live" -- the common case, and the one that
    /// must cost nothing: a clean part answers on the null check, and a clean
    /// granule inside a dirty part answers on one `u16` load rather than the
    /// `GRANULE_SIZE` bitmap probes the old shape spent proving it.
    ///
    /// `out` is the caller's reusable buffer. It used to be a fresh `Vec` per
    /// granule, i.e. one allocation per 1024 rows on a scan of a table with
    /// any deletes at all.
    pub fn live_selection_into<'s>(
        &self,
        gi: usize,
        del: Option<&Deletes>,
        out: &'s mut Vec<u32>,
    ) -> Option<&'s [u32]> {
        let d = del?;
        let ndel = d.granule_deleted(gi);
        if ndel == 0 {
            return None;
        }
        let g = &self.granules[gi];
        let base = gi << G_SHIFT;
        out.clear();
        out.reserve(g.len - (ndel as usize).min(g.len));
        out.extend((0..g.len as u32).filter(|&i| !d.get(base + i as usize)));
        Some(out)
    }

    /// Decode `cols` of granule `gi`. `sel` restricts to specific rows.
    pub fn read_columns(&self, gi: usize, cols: &[usize], sel: Option<&[u32]>) -> Result<Block> {
        let g = &self.granules[gi];
        let out: Vec<Column> = match sel {
            None => cols.iter().map(|&c| g.columns[c].decode(0, g.len)).collect(),
            Some(rows) => cols.iter().map(|&c| g.columns[c].gather(rows)).collect(),
        };
        if out.is_empty() {
            // A `count(*)`-shaped read needs the row count and nothing else.
            return Ok(Block::rows_only(sel.map_or(g.len, |s| s.len())));
        }
        Block::new(out)
    }

    /// Decode one cell. Used by compaction and by point lookups.
    #[inline]
    pub fn value_at(&self, pos: usize, col: usize) -> Value {
        let g = &self.granules[pos >> G_SHIFT];
        g.columns[col].value(pos & (GRANULE_SIZE - 1))
    }

    /// Sort-key lane at an absolute row position.
    #[inline]
    pub fn sort_lane_at(&self, pos: usize) -> u64 {
        match self.sort_col {
            Some(c) => {
                let g = &self.granules[pos >> G_SHIFT];
                g.columns[c].lane(pos & (GRANULE_SIZE - 1))
            }
            None => 0,
        }
    }

    /// Next live row at or after `pos`, or `None`.
    pub fn next_live(&self, mut pos: usize, del: Option<&Deletes>) -> Option<usize> {
        while pos < self.n_rows {
            // Positions are granule-major with a hole at the end of a partial
            // granule; skip straight past those.
            let gi = pos >> G_SHIFT;
            let off = pos & (GRANULE_SIZE - 1);
            if gi >= self.granules.len() {
                return None;
            }
            if off >= self.granules[gi].len {
                pos = (gi + 1) << G_SHIFT;
                continue;
            }
            match del {
                // A whole clean granule is answered without touching a bit.
                Some(d) if d.granule_deleted(gi) != 0 => {
                    if !d.get(pos) {
                        return Some(pos);
                    }
                }
                _ => return Some(pos),
            }
            pos += 1;
        }
        None
    }

    /// Absolute row positions in order, live only, appended to `out`.
    /// Positions are granule-major and skip the tail hole of a partial final
    /// granule.
    pub fn live_positions_into(&self, del: Option<&Deletes>, out: &mut Vec<usize>) {
        out.reserve(self.n_rows - del.map_or(0, |d| d.count()));
        for (gi, g) in self.granules.iter().enumerate() {
            let base = gi << G_SHIFT;
            match del {
                // The clean-granule fast path is the whole point: a merge of a
                // 10M-row part with a handful of tombstones walks a range, not
                // ten million bit tests.
                Some(d) if d.granule_deleted(gi) != 0 => {
                    out.extend((base..base + g.len).filter(|&p| !d.get(p)))
                }
                _ => out.extend(base..base + g.len),
            }
        }
    }

    /// Convenience wrapper over the part's own construction-time image. Only
    /// meaningful before publication -- see [`Part::deleted`].
    pub fn live_positions(&self) -> Vec<usize> {
        let d = self.born_deletes();
        let mut out = Vec::new();
        self.live_positions_into(d.as_ref(), &mut out);
        out
    }

    /// The construction-time delete image as a [`Deletes`], or `None` when the
    /// part was built clean (or has already been adopted).
    pub fn born_deletes(&self) -> Option<Deletes> {
        (self.deleted_count > 0).then(|| {
            Deletes::new(self.deleted.clone(), self.deleted_count, self.granules.len())
        })
    }

    /// Live rows according to the part's own construction-time image.
    ///
    /// Only meaningful before publication: a part inside a [`PartSet`] has no
    /// delete state of its own, so ask [`PartSet::live_rows_of`] instead.
    pub fn born_live_rows(&self) -> usize {
        self.n_rows - self.deleted_count
    }

    pub fn data_bytes(&self) -> usize {
        self.granules.iter().map(|g| g.data_bytes()).sum()
    }

    /// Index footprint of the part itself. Delete bitmaps are not part of it:
    /// they belong to the [`PartSet`], which reports them separately.
    pub fn index_bytes(&self) -> usize {
        self.first_keys.len() * 8
            + self.router.len() * 4
            + 48
            + self.deleted.bytes()
            + self.filter.get().map_or(0, |f| f.bytes())
            + self.granules.iter().map(|g| g.index_bytes()).sum::<usize>()
    }
}

// ------------------------------------------------------------- mask records

/// The positions one sweep hid, grouped by the durable identity of the part
/// they live in and delta-varint encoded as they arrive.
///
/// This is the write-ahead log's copy of a positional delete, built in the
/// sweep's inner loop so nothing is walked twice and nothing is collected per
/// row: `hide` appends between one and three bytes, and the log turns each run
/// into a `TAG_MASK_RUN` frame with one `extend_from_slice`. The obvious
/// alternative -- collect `Vec<u64>` of positions the way the keyed path
/// collects lanes -- costs eight bytes of heap per hidden row, which is 8 MB
/// on a million-row `DELETE` against a path that allocates nothing today.
///
/// It lives here rather than in `persist` because the producer is the storage
/// sweep and `storage` does not depend on `persist`. The encoding is LEB128,
/// decoded by `persist::format::Reader::varint`; `mask_runs_round_trip` pins
/// the two against each other.
///
/// Reused across statements: `clear` keeps both allocations, so a session
/// deleting in a loop allocates once.
#[derive(Default)]
pub struct MaskRuns {
    /// One entry per part touched: identity, how many positions, and where
    /// its deltas start in `bytes`.
    runs: Vec<(u64, u64, usize)>,
    bytes: Vec<u8>,
    /// Previous position within the current run. `u64::MAX` stands for the
    /// -1 that makes the first delta the position itself.
    prev: u64,
    /// Rows hidden in parts with no durable home, which no record can name.
    /// Non-zero is what makes the caller fall back to writing the table out.
    pub dark: usize,
}

impl MaskRuns {
    pub fn clear(&mut self) {
        self.runs.clear();
        self.bytes.clear();
        self.dark = 0;
    }

    /// Record that row `pos` of the part with identity `pid` was hidden.
    ///
    /// Positions must arrive ascending within a part, which the sweep
    /// guarantees: it walks granules in order and offsets in order inside
    /// each, and `base + off` is monotone across both.
    #[inline]
    pub fn hide(&mut self, pid: u64, pos: u64) {
        if self.runs.last().is_none_or(|&(p, ..)| p != pid) {
            self.runs.push((pid, 0, self.bytes.len()));
            self.prev = u64::MAX;
        }
        let k = self.runs.len() - 1;
        self.runs[k].1 += 1;
        // Wrapping, so the `prev = -1` case falls out of the same expression:
        // `pos - u64::MAX - 1 == pos`.
        let mut d = pos.wrapping_sub(self.prev).wrapping_sub(1);
        while d >= 0x80 {
            self.bytes.push((d as u8) | 0x80);
            d >>= 7;
        }
        self.bytes.push(d as u8);
        self.prev = pos;
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// `(pid, count, deltas)` for each part touched, in sweep order.
    pub fn runs(&self) -> impl Iterator<Item = (u64, u64, &[u8])> {
        self.runs.iter().enumerate().map(move |(i, &(pid, n, start))| {
            let end = self.runs.get(i + 1).map_or(self.bytes.len(), |&(_, _, s)| s);
            (pid, n, &self.bytes[start..end])
        })
    }
}

// ---------------------------------------------------------------- deletes

/// The delete mask of one part, immutable once published.
///
/// Two facts about the layout, both load-bearing:
///
///   * it is held as `Option<Arc<Deletes>>` per part, so a part that has never
///     had a row deleted costs 8 bytes and a null check -- no bitmap, no
///     counter, no cache line of the `Part` touched to learn that;
///   * `per_granule` turns "is this granule clean?" into one `u16` load. The
///     previous shape scanned up to `GRANULE_SIZE` bits per granule to answer
///     it, which on a 5M-row part with 1200 tombstones meant five million bit
///     probes to discover that 75% of the granules had nothing to skip.
///
/// A granule holds at most `GRANULE_SIZE` = 1024 rows, so `u16` is exactly
/// wide enough and half the size of the obvious `u32`.
///
/// Measured, interleaved against the previous shape, best of 12 per side over
/// 8 alternating rounds: a 5M-row parallel scan of a part carrying ~1200
/// tombstones went **6.92 -> 13.47 G rows/s (1.95x)**, and the same table
/// through the SQL executor's scan operator **1.21x**. A clean table moved by
/// less than the machine's noise floor (~5%), which is the other half of the
/// requirement -- the parts that have nothing deleted must not pay for this.
#[derive(Clone, Default)]
pub struct Deletes {
    bits: BitSet,
    count: usize,
    per_granule: Vec<u16>,
}

impl Deletes {
    /// Words of the delete bitmap that one granule owns. `GRANULE_SIZE` is a
    /// multiple of 64, so a granule's bits never straddle a word and the
    /// per-granule counts are exact `popcount`s over a fixed window.
    const GRANULE_WORDS: usize = GRANULE_SIZE / 64;

    /// Derive the per-granule counts from a raw bitmap in one pass:
    /// `O(n_rows/64)` words total, not per granule. Paid once when a part is
    /// loaded or adopted -- never on a read.
    pub fn new(bits: BitSet, count: usize, ngranules: usize) -> Deletes {
        let mut per_granule = vec![0u16; ngranules];
        let words = bits.words();
        for (gi, slot) in per_granule.iter_mut().enumerate() {
            let lo = gi * Self::GRANULE_WORDS;
            if lo >= words.len() {
                break;
            }
            let hi = (lo + Self::GRANULE_WORDS).min(words.len());
            *slot = words[lo..hi].iter().map(|w| w.count_ones()).sum::<u32>() as u16;
        }
        Deletes { bits, count, per_granule }
    }

    /// An all-live mask for a part of `ngranules` granules.
    pub fn empty(ngranules: usize) -> Deletes {
        Deletes { bits: BitSet::new(), count: 0, per_granule: vec![0; ngranules] }
    }

    #[inline(always)]
    pub fn get(&self, pos: usize) -> bool {
        self.bits.get(pos)
    }

    /// Deleted rows in granule `gi`; `0` means the granule is entirely live.
    #[inline(always)]
    pub fn granule_deleted(&self, gi: usize) -> u16 {
        // Out-of-range reads as clean: a part shorter than its mask is a
        // decoder problem, not a reason to branch here.
        self.per_granule.get(gi).copied().unwrap_or(0)
    }

    #[inline(always)]
    pub fn count(&self) -> usize {
        self.count
    }

    pub fn bits(&self) -> &BitSet {
        &self.bits
    }

    pub fn words(&self) -> &[u64] {
        self.bits.words()
    }

    /// Set one row. Returns true if it was previously live.
    ///
    /// Only reachable through `Arc::make_mut` on a set being edited, which is
    /// what makes copy-on-write cost one clone per *batch* of tombstones
    /// rather than one per tombstone.
    #[inline]
    pub fn set(&mut self, pos: usize) -> bool {
        if !self.bits.set(pos) {
            return false;
        }
        self.count += 1;
        let gi = pos >> G_SHIFT;
        if gi >= self.per_granule.len() {
            self.per_granule.resize(gi + 1, 0);
        }
        self.per_granule[gi] += 1;
        true
    }

    pub fn bytes(&self) -> usize {
        self.bits.bytes() + self.per_granule.len() * 2
    }
}

// --------------------------------------------------------------- part set

/// "This part is not in any file yet", as an origin. A real part file is
/// `part_000001.gpart` upwards -- `persist::store` allocates sequence numbers
/// from 1 -- so zero is free to mean "nowhere".
pub const NO_FILE: u64 = 0;

/// Top bit of an [`PartSet::origins`] entry: *these rows have been in a part
/// file at some point*, which is a strictly weaker and much longer-lived claim
/// than "file `seq` holds them right now".
///
/// The two claims answer two different questions and only one of them is
/// retracted by a tombstone. The checkpoint asks "may I reuse that file?" and
/// the answer becomes no the instant the mask moves. The log asks "will
/// recovery rebuild this part by *reading a file*, rather than by replaying
/// the inserts that built it?" -- and that stays yes forever, because the
/// checkpoint that rewrites the part writes the same rows in the same
/// positions under the same [`Part::pid`]. A record naming `(pid, pos)` is
/// therefore resolvable long after the file number it was minted under is
/// gone, which is exactly what a positional record has to survive.
///
/// One spare bit of a word that is already there: no second vector, no extra
/// byte resident or on disk.
const HOMED: u64 = 1 << 63;

/// An immutable, versioned list of parts and their delete masks.
///
/// This is the unit of atomicity for the whole storage layer. Every mutation
/// clones the three vectors -- pointers, not data -- edits the entries that
/// changed and bumps `version`; readers take one `Arc` and are then insulated
/// from every subsequent write. Nothing a reader can see ever changes under it.
pub struct PartSet {
    parts: Vec<Arc<Part>>,
    /// Parallel to `parts`. `None` = nothing deleted, the common case.
    deletes: Vec<Option<Arc<Deletes>>>,
    /// Parallel to `parts`: the sequence number of the part file that already
    /// holds *these rows with this delete mask*, or [`NO_FILE`].
    ///
    /// This is what makes a checkpoint incremental, and its correctness rests
    /// on two facts established above: a published part's bytes never change,
    /// and its delete mask is owned by this set. So an entry is valid exactly
    /// until the set replaces the part (never -- parts are only added and
    /// removed) or moves its mask ([`PartSet::tombstone`], which clears it).
    ///
    /// `AtomicU64` for interior mutability, not for concurrency: `write_table`
    /// learns the file a part landed in, and it holds the set through a
    /// `Snapshot` -- there is no `&mut` to be had, and adding one would mean
    /// threading `&mut Table` through the checkpoint just to write 8 bytes
    /// that no reader ever looks at. `Relaxed` is therefore honest: every
    /// store is made by the one thread that has exclusive write access to the
    /// table, and a handoff to another thread carries its own happens-before.
    ///
    /// Width: 8 bytes per *part*, not per row -- a part holds up to 2^40 rows.
    origins: Vec<AtomicU64>,
    /// The next [`Part::pid`] to hand out. Monotone within one lineage of the
    /// set: `push` never lowers it, `remove` never touches it, and the
    /// copy-on-write clone carries it forward -- so a merge cannot hand a
    /// fresh part the identity of one it just consumed.
    ///
    /// It rides on the set rather than on the `Table` so that every path that
    /// can add a part mints through one line. A transaction's private set
    /// carries a copy, and a ROLLBACK that discards it takes back the numbers
    /// with the parts they named -- sound because a part inside an uncommitted
    /// overlay has never been written to a file, so nothing can have cited it.
    next_pid: u64,
    version: u64,
}

/// Hand-written because `AtomicU64` is not `Clone`. Same cost as the derive
/// plus one `Vec` of 8 bytes per part; this runs once per copy-on-write
/// generation of the set (a flush, a batch of tombstones, a merge), never per
/// row and never per granule.
impl Clone for PartSet {
    fn clone(&self) -> PartSet {
        PartSet {
            parts: self.parts.clone(),
            deletes: self.deletes.clone(),
            origins: self
                .origins
                .iter()
                .map(|o| AtomicU64::new(o.load(Ordering::Relaxed)))
                .collect(),
            next_pid: self.next_pid,
            version: self.version,
        }
    }
}

impl Default for PartSet {
    fn default() -> PartSet {
        PartSet {
            parts: Vec::new(),
            deletes: Vec::new(),
            origins: Vec::new(),
            // Zero is "unminted"; identities start at one.
            next_pid: 1,
            version: 0,
        }
    }
}

impl PartSet {
    pub fn new() -> PartSet {
        PartSet::default()
    }

    /// Publish freshly built or freshly decoded parts.
    ///
    /// Takes each part's construction-time delete image as the set's initial
    /// mask, leaving the part clean. After this the part is immutable and its
    /// own `deleted`/`deleted_count` are meaningless -- which is the point.
    pub fn adopt(parts: Vec<Part>) -> PartSet {
        let mut set = PartSet {
            parts: Vec::with_capacity(parts.len()),
            deletes: Vec::with_capacity(parts.len()),
            origins: Vec::with_capacity(parts.len()),
            // Above every identity being adopted, so a part that arrives
            // without one (only a hand-built part can) cannot be minted onto
            // a number a later part in the same batch already owns.
            next_pid: 1 + parts.iter().map(|p| p.pid).max().unwrap_or(0),
            version: 0,
        };
        for p in parts {
            set.push(p);
        }
        set
    }

    /// Start handing out identities above `floor`.
    ///
    /// Called with the log's stream position for a table that has **no**
    /// parts, which is what a table is after `CREATE`, after `TRUNCATE` and
    /// after a `DROP` and re-`CREATE` under the same name. The log outlives
    /// all three by design (see the `wal` module docs), and its position never
    /// restarts, so seeding from it makes a new incarnation's identities
    /// strictly higher than every one the old incarnation ever minted. A
    /// citation left behind by the corpse then names a `pid` that does not
    /// exist and is refused, instead of resolving against a same-numbered part
    /// of the successor and hiding the wrong row.
    ///
    /// Only for the empty case: seeding a populated table would push its
    /// identities up by 64 MiB per checkpoint for no gain.
    pub fn seed_pids(&mut self, floor: u64) {
        if self.parts.is_empty() {
            self.next_pid = self.next_pid.max(floor.saturating_add(1));
        }
    }

    /// Append a part, minting its durable identity and adopting its
    /// construction-time deletes and file provenance.
    pub fn push(&mut self, mut part: Part) {
        // A part decoded from a file arrives with the identity it was written
        // under; one built in memory is minted here. Either way the counter
        // ends up above it, so the two sources cannot collide.
        if part.pid == 0 {
            part.pid = self.next_pid;
        }
        self.next_pid = self.next_pid.max(part.pid + 1);
        let del = (part.deleted_count > 0).then(|| {
            let bits = std::mem::take(&mut part.deleted);
            let n = std::mem::replace(&mut part.deleted_count, 0);
            Arc::new(Deletes::new(bits, n, part.granules.len()))
        });
        // Taken, like the delete image: from here the set owns the answer to
        // "where is this on disk", because from here the set owns the mask
        // that answer depends on.
        self.origins.push(AtomicU64::new(std::mem::replace(&mut part.origin, NO_FILE)));
        self.parts.push(Arc::new(part));
        self.deletes.push(del);
    }

    /// Drop the part at `i`, deletes and all.
    pub fn remove(&mut self, i: usize) -> Arc<Part> {
        self.deletes.remove(i);
        self.origins.remove(i);
        self.parts.remove(i)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.parts.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    #[inline(always)]
    pub fn parts(&self) -> &[Arc<Part>] {
        &self.parts
    }

    #[inline(always)]
    pub fn part(&self, i: usize) -> &Part {
        &self.parts[i]
    }

    #[inline(always)]
    pub fn deletes(&self, i: usize) -> Option<&Deletes> {
        self.deletes[i].as_deref()
    }

    #[inline(always)]
    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn bump(&mut self) {
        self.version += 1;
    }

    /// The part file that already holds part `i` exactly as this set sees it,
    /// or [`NO_FILE`].
    ///
    /// A checkpoint may only *trust* this after confirming the file is still
    /// there -- see `persist::writer::write_table`, which validates every
    /// answer against the directory listing before reusing a name. The
    /// consequence of a stale entry has to be a wasted rewrite, never a commit
    /// record naming a file that is not there.
    #[inline]
    pub fn origin(&self, i: usize) -> u64 {
        self.origins[i].load(Ordering::Relaxed) & !HOMED
    }

    /// The identity a log record may cite for part `i`, or `None` if recovery
    /// would not be able to resolve it.
    ///
    /// [`HOMED`] is the whole test: a part that has been in a file will be
    /// rebuilt from a file, positions and all, so `(pid, pos)` names the same
    /// row after a crash. A part built since the last checkpoint exists only
    /// as the log records that fed it, and how *those* records regroup into
    /// parts on replay depends on flush timing and compaction -- neither of
    /// which is in the log. Such a part is not citable and the caller falls
    /// back to writing the table out.
    #[inline]
    pub fn citable(&self, i: usize) -> Option<u64> {
        (self.origins[i].load(Ordering::Relaxed) & HOMED != 0).then(|| self.parts[i].pid)
    }

    /// Which part carries the durable identity `pid` **and came from a file**,
    /// if any.
    ///
    /// Deliberately [`PartSet::citable`] and not a bare `pid` compare: this is
    /// the read side of the record `citable` mints, and the two have to agree
    /// on which parts are addressable or the pair is unsound. A part built
    /// during a replay is not addressable for the reason `citable` gives --
    /// nothing in the log says where the flush boundaries fell, so its
    /// positions are an artefact of *this* replay rather than a fact about the
    /// stream.
    ///
    /// It is not a theoretical asymmetry. Identities are minted from a counter
    /// that a load seeds from the parts it found (`PartSet::adopt`), so a
    /// point-in-time restore starts numbering at the backup's highest identity
    /// -- exactly where the source database's post-backup parts started. A
    /// replay that flushed a part therefore handed it the very identity the
    /// staged log cites for a part the backup does not contain, and the record
    /// resolved against it: **measured, a `DELETE` of `id = 8` came back from a
    /// roll-forward having hidden `id = 1002` instead, exit 0 and no
    /// diagnostic**. With this test it resolves to nothing, which is the
    /// refusal the log's decoder already raises as corruption.
    ///
    /// Linear over the part list -- tens of entries, and this runs once per
    /// replayed record, not per row.
    pub fn index_of(&self, pid: u64) -> Option<usize> {
        (0..self.parts.len()).find(|&i| self.citable(i) == Some(pid))
    }

    /// Record that part `i`, with this set's current mask for it, is durably
    /// in file `seq`.
    ///
    /// `&self` deliberately: the caller reaches the set through a `Snapshot`.
    /// It must be called only *after* the file is fsynced and renamed into
    /// place, because that is the claim it makes.
    #[inline]
    pub fn set_origin(&self, i: usize, seq: u64) {
        self.origins[i].store(HOMED | seq, Ordering::Relaxed);
    }

    /// Live rows in part `i`.
    #[inline]
    pub fn live_rows_of(&self, i: usize) -> usize {
        self.parts[i].n_rows - self.deletes[i].as_ref().map_or(0, |d| d.count())
    }

    pub fn live_rows(&self) -> usize {
        (0..self.parts.len()).map(|i| self.live_rows_of(i)).sum()
    }

    pub fn delete_bytes(&self) -> usize {
        self.deletes.iter().flatten().map(|d| d.bytes()).sum()
    }

    /// Hide row `pos` of part `i`. Returns true if it was live.
    ///
    /// The `Arc::make_mut` is the copy-on-write hinge: it clones the mask the
    /// first time this set touches it while a snapshot still holds the old
    /// one, then mutates in place for every tombstone after that. A flush that
    /// shadows ten thousand keys pays one clone, not ten thousand.
    ///
    /// **A moved mask invalidates the file, but not the identity.** The delete
    /// bitmap is written inside the part file
    /// (`persist::writer::part_meta`), so a part whose rows are untouched but
    /// whose deletes moved has to be rewritten in full before that file can be
    /// reused -- hence the cleared sequence number. [`HOMED`] survives,
    /// because the rewrite reproduces the same rows at the same positions
    /// under the same [`Part::pid`]; a sidecar mask was considered for this
    /// and rejected, since it would need a second file to become durable
    /// atomically with the first, which is a new and weaker-by-default commit
    /// protocol. The write-ahead log is the commit protocol that already
    /// exists, so the mask goes there instead -- see `Session::apply_sweep`.
    ///
    /// Note the `d.set(pos)` return: re-deleting an already-dead row leaves
    /// the mask byte-identical, so it must *not* invalidate the file. That
    /// idempotence is also what makes a replayed mask record safe to apply
    /// twice, which a coarse checkpoint watermark guarantees will happen.
    pub fn tombstone(&mut self, i: usize, pos: usize) -> bool {
        let ngran = self.parts[i].granules.len();
        let slot = &mut self.deletes[i];
        let d = Arc::make_mut(slot.get_or_insert_with(|| Arc::new(Deletes::empty(ngran))));
        if !d.set(pos) {
            return false;
        }
        *self.origins[i].get_mut() &= HOMED;
        true
    }
}

/// A pinned, consistent view of a table's parts.
///
/// Holding one keeps every part in it alive -- including parts compaction has
/// already unlinked from disk (see the module docs on why that is safe) -- and
/// guarantees that the delete masks a scan reads are the ones that were
/// published together with those parts. Taking a snapshot is one `RwLock`
/// read plus one `Arc` clone, ~20 ns, and it happens **once per query**, never
/// per part and never per granule.
#[derive(Clone)]
pub struct Snapshot {
    set: Arc<PartSet>,
}

impl Snapshot {
    pub fn new(set: Arc<PartSet>) -> Snapshot {
        Snapshot { set }
    }

    #[inline(always)]
    pub fn set(&self) -> &PartSet {
        &self.set
    }

    #[inline(always)]
    pub fn parts(&self) -> &[Arc<Part>] {
        self.set.parts()
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.set.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    #[inline(always)]
    pub fn part(&self, i: usize) -> &Part {
        self.set.part(i)
    }

    #[inline(always)]
    pub fn deletes(&self, i: usize) -> Option<&Deletes> {
        self.set.deletes(i)
    }

    #[inline(always)]
    pub fn version(&self) -> u64 {
        self.set.version()
    }

    pub fn live_rows(&self) -> usize {
        self.set.live_rows()
    }
}

/// Gather rows named by `(part_index, position)` pairs into a `Block`.
///
/// Column-major: one pass per column over the whole merge order, so the output
/// buffers stay hot and each packed column is touched once. This is the shape
/// compaction and multi-part scans both want.
pub fn gather_rows(parts: &[&Part], order: &[(u32, u32)], schema: &Schema) -> Result<Block> {
    let mut cols = Vec::with_capacity(schema.len());
    for c in 0..schema.len() {
        let mut b = ColumnBuilder::with_capacity(schema.ty(c).clone(), order.len());
        for &(pi, pos) in order {
            b.push_value(&parts[pi as usize].value_at(pos as usize, c))?;
        }
        cols.push(b.finish());
    }
    Block::new(cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{hash_key, splitmix64, FP_SEED};
    use crate::types::DataType;

    fn sorted_block(n: u64, random: bool) -> Block {
        let mut keys: Vec<u64> = (0..n).map(|i| if random { splitmix64(i) } else { i }).collect();
        keys.sort_unstable();
        keys.dedup();
        let vals: Vec<i64> = keys.iter().map(|&k| (k % 500) as i64 - 250).collect();
        Block::new(vec![
            Column::u64s(DataType::UInt64, keys),
            Column::i64s(DataType::Int64, vals),
        ])
        .unwrap()
    }

    #[test]
    fn point_lookups_hit_every_key() {
        let b = sorted_block(20_000, true);
        let keys = b.column(0).as_u64().unwrap().to_vec();
        let p = Part::build(&b, Some(0), Some(0)).unwrap();
        let mut st = Stats::default();
        for (i, &k) in keys.iter().enumerate() {
            let pos = p.find_live(k, hash_key(k, FP_SEED), &mut st, None);
            assert!(pos.is_some(), "missing key {k}");
            assert_eq!(p.value_at(pos.unwrap(), 0), Value::UInt(k));
            assert_eq!(p.value_at(pos.unwrap(), 1), Value::Int((k % 500) as i64 - 250));
            let _ = i;
        }
    }

    #[test]
    fn foreign_keys_miss() {
        let b = sorted_block(10_000, true);
        let keys: std::collections::HashSet<u64> =
            b.column(0).as_u64().unwrap().iter().copied().collect();
        let p = Part::build(&b, Some(0), Some(0)).unwrap();
        let mut st = Stats::default();
        let mut probed = 0;
        for i in 0..5_000u64 {
            let k = splitmix64(9_000_000 + i);
            if !keys.contains(&k) {
                probed += 1;
                assert_eq!(p.find_live(k, hash_key(k, FP_SEED), &mut st, None), None);
            }
        }
        assert!(probed > 4_000);
    }

    #[test]
    fn router_locates_granules_correctly() {
        let b = sorted_block(50_000, false);
        let p = Part::build(&b, Some(0), Some(0)).unwrap();
        // sequential keys 0..50000, granule i covers [i*1024, ...)
        assert_eq!(p.locate_granule(0), Some(0));
        assert_eq!(p.locate_granule(1023), Some(0));
        assert_eq!(p.locate_granule(1024), Some(1));
        assert_eq!(p.locate_granule(49_999), Some(48));
        // a key past the end still routes to the last granule; zone maps reject it
        assert!(p.locate_granule(999_999).is_some());
    }

    #[test]
    fn deletes_hide_rows_and_are_counted() {
        let b = sorted_block(3000, false);
        let mut p = Part::build(&b, Some(0), Some(0)).unwrap();
        let mut st = Stats::default();
        let pos = p.find_live(5, hash_key(5, FP_SEED), &mut st, None).unwrap();
        assert!(p.mark_deleted(pos));
        assert!(!p.mark_deleted(pos), "second delete is a no-op");
        assert_eq!(p.deleted_count, 1);

        // Publishing moves the mask out of the part and into the set.
        let set = PartSet::adopt(vec![p]);
        let (p, d) = (set.part(0), set.deletes(0));
        assert_eq!(d.unwrap().count(), 1);
        assert_eq!(p.deleted_count, 0, "an adopted part keeps no delete state");
        assert_eq!(p.find_live(5, hash_key(5, FP_SEED), &mut st, d), None);
        assert_eq!(set.live_rows(), 2999);

        let mut buf = Vec::new();
        let sel = p.live_selection_into(0, d, &mut buf).unwrap();
        assert_eq!(sel.len(), 1023);
        assert!(!sel.contains(&5));
    }

    #[test]
    fn live_selection_is_none_when_clean() {
        let b = sorted_block(3000, false);
        let p = Part::build(&b, Some(0), Some(0)).unwrap();
        let mut buf = Vec::new();
        assert!(
            p.live_selection_into(0, None, &mut buf).is_none(),
            "a clean part must not walk a bitmap"
        );
        assert!(buf.is_empty(), "and must not fill a buffer either");
    }

    #[test]
    fn a_clean_granule_inside_a_dirty_part_costs_no_selection() {
        // 3000 rows -> granules 0,1,2. Delete one row in granule 1 only.
        let b = sorted_block(3000, false);
        let mut p = Part::build(&b, Some(0), Some(0)).unwrap();
        assert!(p.mark_deleted(GRANULE_SIZE + 3));
        let set = PartSet::adopt(vec![p]);
        let (p, d) = (set.part(0), set.deletes(0).unwrap());
        assert_eq!(d.granule_deleted(0), 0);
        assert_eq!(d.granule_deleted(1), 1);
        assert_eq!(d.granule_deleted(2), 0);
        let mut buf = Vec::new();
        assert!(p.live_selection_into(0, Some(d), &mut buf).is_none());
        assert!(p.live_selection_into(2, Some(d), &mut buf).is_none());
        assert_eq!(p.live_selection_into(1, Some(d), &mut buf).unwrap().len(), 1023);
    }

    #[test]
    fn tombstoning_a_set_leaves_an_outstanding_snapshot_untouched() {
        // The whole point of the keystone: a reader that took a snapshot
        // before a delete must not observe the delete, and must not observe a
        // half-applied batch of them either.
        let b = sorted_block(3000, false);
        let mut set = PartSet::adopt(vec![Part::build(&b, Some(0), Some(0)).unwrap()]);
        let before = Snapshot::new(Arc::new(set.clone()));

        set.bump();
        for pos in 0..100 {
            assert!(set.tombstone(0, pos));
        }
        let after = Snapshot::new(Arc::new(set));

        assert_eq!(before.live_rows(), 3000, "the pinned view moved");
        assert_eq!(after.live_rows(), 2900);
        assert!(before.deletes(0).is_none());
        assert_eq!(after.deletes(0).unwrap().count(), 100);
        assert!(after.version() > before.version());
        // ...and both views still point at the same immutable part.
        assert!(std::ptr::eq(before.part(0), after.part(0)));
    }

    /// The provenance a checkpoint reads: adopted from the part, carried
    /// through copy-on-write, dropped the instant a tombstone actually lands.
    #[test]
    fn a_parts_file_provenance_follows_its_delete_mask() {
        let b = sorted_block(3000, false);
        let mut p = Part::build(&b, Some(0), Some(0)).unwrap();
        p.set_origin(7);
        let mut set = PartSet::adopt(vec![p]);
        assert_eq!(set.origin(0), 7, "adoption must take the decoded file number");

        // A snapshot is a different set; both must still name file 7.
        let pinned = set.clone();
        assert_eq!(pinned.origin(0), 7, "a copy-on-write clone lost the provenance");

        // A tombstone that hides nothing leaves the mask byte-identical...
        set.tombstone(0, 5);
        assert_eq!(set.origin(0), NO_FILE, "a tombstone must invalidate the file");
        set.set_origin(0, 9);
        assert!(!set.tombstone(0, 5), "the row is already hidden");
        assert_eq!(set.origin(0), 9, "re-hiding a hidden row changed nothing on disk");

        // ...and the snapshot taken before any of it is untouched.
        assert_eq!(pinned.origin(0), 7);
    }

    /// The two claims one word carries, and the fact that only one of them is
    /// retracted by a tombstone. A part that has stopped naming a file is
    /// still citable, because the checkpoint that rewrites it reproduces the
    /// same rows at the same positions under the same identity -- which is the
    /// whole reason a positional log record works here.
    #[test]
    fn a_tombstone_retracts_the_file_but_not_the_identity() {
        let b = sorted_block(3000, false);
        let mut p = Part::build(&b, Some(0), Some(0)).unwrap();
        p.pid = 42;
        p.set_origin(7);
        let mut set = PartSet::adopt(vec![p]);
        assert_eq!(set.citable(0), Some(42));
        assert_eq!(set.origin(0), 7);

        set.tombstone(0, 5);
        assert_eq!(set.origin(0), NO_FILE, "the file no longer holds this mask");
        assert_eq!(set.citable(0), Some(42), "but the rows are still in a file somewhere");

        // A part built in memory has never been in a file, so nothing may name
        // it -- the residual the fold still covers.
        let fresh = Part::build(&sorted_block(10, false), Some(0), Some(0)).unwrap();
        set.push(fresh);
        assert_eq!(set.citable(1), None);
        assert_eq!(set.origin(1), NO_FILE);
    }

    /// Identities are minted once, never reissued, and never inherited: the
    /// counter only ever goes up, and a part that arrives already carrying one
    /// keeps it and pushes the counter above it.
    #[test]
    fn identities_are_minted_once_and_never_reissued() {
        let blk = || sorted_block(10, false);
        let mut set = PartSet::new();
        for _ in 0..3 {
            set.push(Part::build(&blk(), Some(0), Some(0)).unwrap());
        }
        assert_eq!((0..3).map(|i| set.part(i).pid).collect::<Vec<_>>(), vec![1, 2, 3]);
        // Resolution is by *citable* identity, so these three have to have
        // been in a file before they answer to theirs -- the rule
        // `only_a_part_that_has_been_in_a_file_answers_to_its_identity`
        // pins. Nothing about minting changes; the lookup gained a
        // precondition.
        for i in 0..3 {
            set.set_origin(i, 1 + i as u64);
        }
        assert_eq!(set.index_of(2), Some(1));
        assert_eq!(set.index_of(9), None);

        // Removing every part -- what a merge does -- must not free the
        // numbers they held.
        for i in (0..3).rev() {
            set.remove(i);
        }
        set.push(Part::build(&blk(), Some(0), Some(0)).unwrap());
        assert_eq!(set.part(0).pid, 4, "a merge must not reissue a consumed identity");

        // A decoded part keeps the identity in its bytes, and the counter
        // steps over it.
        let mut loaded = Part::build(&blk(), Some(0), Some(0)).unwrap();
        loaded.pid = 100;
        set.push(loaded);
        set.push(Part::build(&blk(), Some(0), Some(0)).unwrap());
        assert_eq!(set.part(2).pid, 101);

        // And the empty-table seed lifts it past a previous incarnation's.
        let mut fresh = PartSet::new();
        fresh.seed_pids(5_000);
        fresh.push(Part::build(&blk(), Some(0), Some(0)).unwrap());
        assert_eq!(fresh.part(0).pid, 5_001);
        fresh.seed_pids(9_000);
        fresh.push(Part::build(&blk(), Some(0), Some(0)).unwrap());
        assert_eq!(fresh.part(1).pid, 5_002, "a populated table is left alone");
    }

    /// The read side of a mask record has to agree with the write side about
    /// which parts are addressable, or the pair is unsound.
    ///
    /// This is not hypothetical. Identities are minted from a counter a load
    /// seeds from the parts it found, so a point-in-time restore starts
    /// numbering at the backup's highest identity -- precisely where the
    /// source database's post-backup parts started. A replay that flushed a
    /// part therefore handed it the very identity the staged log cites for a
    /// part the backup does not contain, and `index_of` resolved the record
    /// against it: a `DELETE FROM t WHERE id = 8`, rolled forward, hid
    /// `id = 1002` instead. Exit 0, no diagnostic.
    #[test]
    fn only_a_part_that_has_been_in_a_file_answers_to_its_identity() {
        let blk = || sorted_block(10, false);
        let mut set = PartSet::new();
        let mut replayed = Part::build(&blk(), Some(0), Some(0)).unwrap();
        replayed.pid = 7;
        set.push(replayed);
        assert_eq!(set.part(0).pid, 7);
        assert_eq!(
            set.index_of(7),
            None,
            "a part built in this replay has no durable positions to name"
        );

        // The same identity, once it really is in a file: the ordinary crash
        // recovery, which must keep working.
        set.set_origin(0, 3);
        assert_eq!(set.index_of(7), Some(0));

        // And a collision resolves to the one the log could have meant. The
        // second part is what a replay flush builds; it must not shadow the
        // first, whichever order they sit in.
        let mut ghost = Part::build(&blk(), Some(0), Some(0)).unwrap();
        ghost.pid = 7;
        set.push(ghost);
        assert_eq!(set.index_of(7), Some(0));
    }

    /// The gap encoding, against the numbers it is supposed to produce: a
    /// dense run is one byte a row, and the runs come back grouped by part in
    /// sweep order.
    #[test]
    fn mask_runs_group_by_part_and_encode_gaps() {
        let mut m = MaskRuns::default();
        for p in [0u64, 1, 2] {
            m.hide(4, p);
        }
        m.hide(6, 1_000);
        m.hide(6, 1_001);
        // Back to a part already seen: a new run, because the log frames one
        // record per contiguous group and replay resolves each independently.
        m.hide(4, 9);
        let got: Vec<(u64, u64, Vec<u8>)> =
            m.runs().map(|(p, n, d)| (p, n, d.to_vec())).collect();
        assert_eq!(
            got,
            vec![
                // 0, then +1, then +1 -- one byte each.
                (4, 3, vec![0, 0, 0]),
                // 1000 needs two bytes; the next is adjacent.
                (6, 2, vec![0xE8, 0x07, 0]),
                (4, 1, vec![9]),
            ]
        );
        assert!(!m.is_empty());
        m.clear();
        assert!(m.is_empty());
    }

    /// The three vectors are indexed by the same `i` everywhere, so a removal
    /// that shifted one of them would silently pair a part with another
    /// part's file -- the one way this mechanism could commit wrong bytes.
    #[test]
    fn removing_a_part_keeps_provenance_aligned() {
        let mut set = PartSet::new();
        for (i, n) in [900usize, 1100, 1300].into_iter().enumerate() {
            let mut p = Part::build(&sorted_block(n as u64, false), Some(0), Some(0)).unwrap();
            p.set_origin(i as u64 + 1);
            set.push(p);
        }
        set.remove(1);
        assert_eq!(set.len(), 2);
        assert_eq!((set.origin(0), set.origin(1)), (1, 3));
        assert_eq!((set.part(0).n_rows, set.part(1).n_rows), (900, 1300));
    }

    #[test]
    fn read_columns_projects_and_selects() {
        let b = sorted_block(2000, false);
        let p = Part::build(&b, Some(0), Some(0)).unwrap();
        let blk = p.read_columns(0, &[1], None).unwrap();
        assert_eq!(blk.width(), 1);
        assert_eq!(blk.rows(), 1024);

        let blk = p.read_columns(0, &[0, 1], Some(&[3, 7])).unwrap();
        assert_eq!(blk.rows(), 2);
        assert_eq!(blk.column(0).as_u64().unwrap(), &[3, 7]);

        // zero-column read still reports the row count, for count(*)
        let blk = p.read_columns(0, &[], None).unwrap();
        assert_eq!(blk.rows(), 1024);
        assert_eq!(blk.width(), 0);
    }

    #[test]
    fn bloom_filter_has_no_false_negatives() {
        let b = sorted_block(5000, true);
        let keys = b.column(0).as_u64().unwrap().to_vec();
        let p = Part::build(&b, Some(0), Some(0)).unwrap();
        assert!(p.may_contain(12345), "no filter => always maybe");
        assert!(!p.has_filter(), "probing must never build");
        p.ensure_filter();
        assert!(p.has_filter());
        for &k in &keys {
            assert!(p.may_contain(hash_key(k, FP_SEED)), "false negative on {k}");
        }
        // Memoized: a second call hands back the same filter, not a new one.
        let a = p.ensure_filter().unwrap() as *const _;
        assert!(std::ptr::eq(a, p.ensure_filter().unwrap()));
    }

    #[test]
    fn live_positions_skip_partial_granule_holes() {
        // 1500 rows => granule 0 full (1024), granule 1 holds 476.
        let b = sorted_block(1500, false);
        let p = Part::build(&b, Some(0), Some(0)).unwrap();
        let pos = p.live_positions();
        assert_eq!(pos.len(), 1500);
        assert_eq!(pos[1023], 1023);
        assert_eq!(pos[1024], GRANULE_SIZE, "second granule starts at a fresh base");
        assert_eq!(*pos.last().unwrap(), GRANULE_SIZE + 475);
    }

    #[test]
    fn next_live_walks_past_holes_and_deletes() {
        let b = sorted_block(1500, false);
        let mut p = Part::build(&b, Some(0), Some(0)).unwrap();
        p.mark_deleted(0);
        let set = PartSet::adopt(vec![p]);
        let (p, d) = (set.part(0), set.deletes(0));
        assert_eq!(p.next_live(0, d), Some(1));
        // position 1024 is the hole at the end of granule 0's slot range
        assert_eq!(p.next_live(1024, d), Some(GRANULE_SIZE));
        assert_eq!(p.next_live(GRANULE_SIZE + 476, d), None);
    }

    #[test]
    fn unsorted_part_without_pk_still_reads() {
        let b = Block::new(vec![Column::strs(
            DataType::String,
            vec!["c".into(), "a".into(), "b".into()],
        )])
        .unwrap();
        let p = Part::build(&b, None, None).unwrap();
        assert_eq!(p.n_rows, 3);
        let blk = p.read_columns(0, &[0], None).unwrap();
        let got: Vec<&str> = blk.column(0).as_str().unwrap().iter().map(|s| s.as_ref()).collect();
        assert_eq!(got, vec!["c", "a", "b"]);
    }
}
