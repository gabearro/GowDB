//! A granule: `GRANULE_SIZE` rows of every column, compressed independently,
//! plus an optional point-lookup index over the primary key.
//!
//! Granules are the unit of parallelism (built concurrently across cores), the
//! unit of pruning (zone maps are per granule per column), and the unit of
//! I/O. Making them independent is what lets part construction scale linearly
//! with cores and lets a range scan touch only the granules it needs.
//!
//! ## Learned ranks
//!
//! The point-lookup index does not store rows. Inside a granule the keys are
//! sorted, so a key's row is *predicted* by linear interpolation over
//! `[min, max]`; only the small prediction **error** is stored, fused with a
//! 6-bit fingerprint into one packed record per MPH slot:
//!
//! ```text
//!     rec = fp6(key) << ebits | (rank - predicted - err_bias)
//! ```
//!
//! Clustered keys predict near-exactly (0-1 error bits/row); uniform random
//! keys need ~7. One record load both filters foreign keys (1/64 false
//! positive) and yields the row, and verification against the packed key
//! column stays exact.

use crate::common::{fp6, hash_key, Result, FP_SEED};
use crate::encoding::{packed_lower_bound, PackedU64};
use crate::index::Mph;
use crate::types::Block;

use super::column::PackedColumn;

/// Point-lookup index over one granule's primary key column.
pub struct PkIndex {
    pub min: u64,
    pub max: u64,
    mph: Mph,
    /// `fp6(key) << ebits | (rank - predicted - err_bias)`, one per MPH slot.
    fpr: PackedU64,
    /// `((len-1) << 64) / (max-min)`, saturating: the interpolation slope.
    pmul: u64,
    err_bias: i32,
    ebits: u32,
    emask: u64,
}

impl PkIndex {
    fn build(keys: &[u64]) -> PkIndex {
        let len = keys.len();
        let mph = Mph::build(keys);
        let (min, max) = (keys[0], keys[len - 1]);
        let range = max - min;
        let pmul = if range == 0 {
            0
        } else {
            ((((len - 1) as u128) << 64) / range as u128).min(u64::MAX as u128) as u64
        };
        let predict = |k: u64| -> i64 { ((((k - min) as u128) * pmul as u128) >> 64) as i64 };
        let errs: Vec<i64> = keys
            .iter()
            .enumerate()
            .map(|(row, &k)| row as i64 - predict(k))
            .collect();
        let err_bias = *errs.iter().min().unwrap();
        let espan = (*errs.iter().max().unwrap() - err_bias) as u64;
        let ebits = 64 - espan.leading_zeros();
        let emask = if ebits == 0 { 0 } else { (1u64 << ebits) - 1 };
        let mut recs = vec![0u64; len];
        for (row, &k) in keys.iter().enumerate() {
            recs[mph.lookup(k)] =
                (fp6(hash_key(k, FP_SEED)) << ebits) | (errs[row] - err_bias) as u64;
        }
        PkIndex {
            min,
            max,
            mph,
            fpr: PackedU64::pack(&recs),
            pmul,
            err_bias: err_bias as i32,
            ebits,
            emask,
        }
    }

    /// Candidate row for `key`, or `None` if the fingerprint rules it out.
    ///
    /// The row is clamped into range: a fingerprint-colliding foreign key must
    /// never index out of bounds, and the caller verifies against the packed
    /// key anyway.
    #[inline(always)]
    pub fn candidate(&self, key: u64, fph: u64, len: usize) -> Option<usize> {
        let rec = self.fpr.get(self.mph.lookup(key));
        if rec >> self.ebits != fp6(fph) {
            return None;
        }
        let pred = (((key - self.min) as u128 * self.pmul as u128) >> 64) as i64;
        let row = (pred + self.err_bias as i64 + (rec & self.emask) as i64)
            .clamp(0, len as i64 - 1) as usize;
        Some(row)
    }

    pub fn to_parts(&self) -> PkIndexParts {
        let (gs, nb, n, sb, sw, sword) = self.mph.to_parts();
        PkIndexParts {
            min: self.min,
            max: self.max,
            pmul: self.pmul,
            err_bias: self.err_bias,
            ebits: self.ebits,
            mph_gs: gs,
            mph_nb: nb,
            mph_n: n,
            seed_base: sb,
            seed_width: sw,
            seed_words: sword,
            fpr_base: self.fpr.base(),
            fpr_width: self.fpr.width(),
            fpr_words: self.fpr.words().to_vec(),
        }
    }

    pub fn from_parts(p: PkIndexParts) -> Result<PkIndex> {
        let emask = if p.ebits == 0 { 0 } else { (1u64 << p.ebits) - 1 };
        Ok(PkIndex {
            min: p.min,
            max: p.max,
            mph: Mph::from_parts(p.mph_gs, p.mph_nb, p.mph_n, p.seed_base, p.seed_width, p.seed_words)?,
            fpr: PackedU64::from_parts(p.fpr_base, p.fpr_width, p.fpr_words),
            pmul: p.pmul,
            err_bias: p.err_bias,
            ebits: p.ebits,
            emask,
        })
    }

    pub fn bytes(&self) -> usize {
        self.mph.bytes() + self.fpr.bytes() + 32
    }
}

/// Flat form of a [`PkIndex`], for the on-disk writer.
pub struct PkIndexParts {
    pub min: u64,
    pub max: u64,
    pub pmul: u64,
    pub err_bias: i32,
    pub ebits: u32,
    pub mph_gs: u64,
    pub mph_nb: u32,
    pub mph_n: usize,
    pub seed_base: u64,
    pub seed_width: u32,
    pub seed_words: Vec<u64>,
    pub fpr_base: u64,
    pub fpr_width: u32,
    pub fpr_words: Vec<u64>,
}

pub struct Granule {
    pub len: usize,
    pub columns: Vec<PackedColumn>,
    /// Present only when the table has a single integer primary key **and**
    /// this granule's keys are unique. Duplicate keys fall back to
    /// interpolation search over the sorted packed key column, which is
    /// correct but ~3x slower.
    pub pk: Option<PkIndex>,
    /// Lane bounds of the sort column, cached for range pruning.
    pub sort_min: u64,
    pub sort_max: u64,
}

impl Granule {
    /// Build from `block[s..e]`. `sort_col` is the leading ORDER BY column;
    /// `pk_col` additionally requests the MPH point-lookup index.
    pub fn build(
        block: &Block,
        s: usize,
        e: usize,
        sort_col: Option<usize>,
        pk_col: Option<usize>,
    ) -> Result<Granule> {
        Granule::build_sel(block, s, e, None, sort_col, pk_col)
    }

    /// Build from `block[s..e]`, or from `perm[s..e]` when a permutation is
    /// supplied.
    ///
    /// The permutation exists so that sorting never materializes a sorted copy
    /// of the table. Packing already reads each granule's rows individually,
    /// so it can just as cheaply read them in permuted order -- which saves a
    /// full copy of every column on every flush, and on a bulk load saves peak
    /// memory proportional to the whole dataset.
    pub fn build_sel(
        block: &Block,
        s: usize,
        e: usize,
        perm: Option<&[u32]>,
        sort_col: Option<usize>,
        pk_col: Option<usize>,
    ) -> Result<Granule> {
        let len = e - s;
        let columns: Vec<PackedColumn> = match perm {
            None => block
                .columns
                .iter()
                .map(|c| PackedColumn::build(&c.slice(s, e)))
                .collect::<Result<_>>()?,
            Some(p) => block
                .columns
                .iter()
                .map(|c| PackedColumn::build(&c.take(&p[s..e])))
                .collect::<Result<_>>()?,
        };

        let (sort_min, sort_max) = match sort_col {
            Some(c) if len > 0 => {
                let pc = &columns[c];
                (pc.lane(0), pc.lane(len - 1))
            }
            _ => (0, u64::MAX),
        };

        let pk = match pk_col {
            Some(c) if len > 0 => {
                let pc = &columns[c];
                let keys: Vec<u64> = (0..len).map(|i| pc.lane(i)).collect();
                // The MPH requires distinct keys. Rows are already sorted by
                // this column, so a linear adjacency check is enough.
                let unique = keys.windows(2).all(|w| w[0] < w[1]);
                if unique {
                    Some(PkIndex::build(&keys))
                } else {
                    None
                }
            }
            _ => None,
        };

        Ok(Granule { len, columns, pk, sort_min, sort_max })
    }

    pub fn from_parts(
        len: usize,
        columns: Vec<PackedColumn>,
        pk: Option<PkIndex>,
        sort_min: u64,
        sort_max: u64,
    ) -> Granule {
        Granule { len, columns, pk, sort_min, sort_max }
    }

    /// Row index of `key` within this granule, or `None`.
    ///
    /// Two paths: the learned-rank index when available, otherwise
    /// interpolation search over the packed sorted key column. Both verify
    /// the candidate against the stored key, so neither can return a wrong row.
    #[inline]
    pub fn find_key(&self, pk_col: usize, key: u64, fph: u64, stats: &mut Stats) -> Option<usize> {
        let kc = &self.columns[pk_col];
        match &self.pk {
            Some(idx) => {
                stats.mph_probes += 1;
                let row = match idx.candidate(key, fph, self.len) {
                    Some(r) => r,
                    None => {
                        stats.fingerprint_negative += 1;
                        return None;
                    }
                };
                if kc.lane(row) != key {
                    stats.false_probe += 1;
                    return None;
                }
                Some(row)
            }
            None => {
                let row = packed_lower_bound(kc.lanes(), self.len, key);
                if row < self.len && kc.lane(row) == key {
                    Some(row)
                } else {
                    None
                }
            }
        }
    }

    /// First row whose sort lane is `>= lane`.
    #[inline]
    pub fn lower_bound(&self, sort_col: usize, lane: u64) -> usize {
        packed_lower_bound(self.columns[sort_col].lanes(), self.len, lane)
    }

    pub fn data_bytes(&self) -> usize {
        self.columns.iter().map(|c| c.data_bytes()).sum()
    }
    pub fn index_bytes(&self) -> usize {
        self.pk.as_ref().map_or(0, |p| p.bytes()) + 24
    }
}

/// Access-path counters. Cheap to maintain, and the only way to tell whether
/// pruning is actually firing rather than merely being implemented.
#[derive(Default, Debug, Clone, Copy)]
pub struct Stats {
    /// Filtered by the part-level bloom before any index work.
    pub bloom_negative: u64,
    /// Filtered by the 6-bit fingerprint in the fused rank record.
    pub fingerprint_negative: u64,
    /// Granules skipped by the primary-key zone map on a point lookup.
    pub zone_pruned_point: u64,
    /// Granules skipped by a zone map on a scan.
    pub zone_pruned_scan: u64,
    /// Granules actually read during a scan.
    pub granules_scanned: u64,
    pub mph_probes: u64,
    /// Fingerprint matched but the key did not: the true MPH false-positive.
    pub false_probe: u64,
    pub rows_scanned: u64,
}

impl Stats {
    pub fn merge(&mut self, o: &Stats) {
        self.bloom_negative += o.bloom_negative;
        self.fingerprint_negative += o.fingerprint_negative;
        self.zone_pruned_point += o.zone_pruned_point;
        self.zone_pruned_scan += o.zone_pruned_scan;
        self.granules_scanned += o.granules_scanned;
        self.mph_probes += o.mph_probes;
        self.false_probe += o.false_probe;
        self.rows_scanned += o.rows_scanned;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{splitmix64, GRANULE_SIZE};
    use crate::types::{Column, DataType};

    fn block_of(keys: &[u64]) -> Block {
        let vals: Vec<i64> = keys.iter().map(|&k| k as i64 ^ 7).collect();
        Block::new(vec![
            Column::u64s(DataType::UInt64, keys.to_vec()),
            Column::i64s(DataType::Int64, vals),
        ])
        .unwrap()
    }

    #[test]
    fn finds_every_key_and_rejects_foreign_ones() {
        let mut keys: Vec<u64> = (0..GRANULE_SIZE as u64).map(splitmix64).collect();
        keys.sort_unstable();
        let b = block_of(&keys);
        let g = Granule::build(&b, 0, keys.len(), Some(0), Some(0)).unwrap();
        assert!(g.pk.is_some(), "unique keys should get an MPH index");

        let mut st = Stats::default();
        for (row, &k) in keys.iter().enumerate() {
            assert_eq!(g.find_key(0, k, hash_key(k, FP_SEED), &mut st), Some(row));
        }
        for i in 0..3000u64 {
            let probe = splitmix64(1_000_000 + i);
            if !keys.contains(&probe) {
                assert_eq!(g.find_key(0, probe, hash_key(probe, FP_SEED), &mut st), None);
            }
        }
        assert!(st.fingerprint_negative > 0, "fingerprint should reject most probes");
    }

    #[test]
    fn duplicate_keys_fall_back_to_search() {
        let keys = vec![1u64, 2, 2, 3];
        let b = block_of(&keys);
        let g = Granule::build(&b, 0, 4, Some(0), Some(0)).unwrap();
        assert!(g.pk.is_none(), "duplicates must disable the MPH index");
        let mut st = Stats::default();
        // still finds the first occurrence
        assert_eq!(g.find_key(0, 2, hash_key(2, FP_SEED), &mut st), Some(1));
        assert_eq!(g.find_key(0, 9, hash_key(9, FP_SEED), &mut st), None);
    }

    #[test]
    fn adversarial_key_distributions_lookup_exactly() {
        // Worst cases for interpolation prediction: heavy clustering, outliers.
        let cases: Vec<Vec<u64>> = vec![
            (0..1024u64).collect(),
            (0..1023u64).chain([u64::MAX - 1]).collect(),
            [0u64].into_iter().chain((1..1024).map(|i| u64::MAX / 2 + i)).collect(),
            (0..512u64).chain((0..512).map(|i| u64::MAX - 512 + i)).collect(),
            (0..1024u64).map(|i| i * i * 31).collect(),
            vec![42],
            vec![5, u64::MAX],
        ];
        for keys in cases {
            let b = block_of(&keys);
            let g = Granule::build(&b, 0, keys.len(), Some(0), Some(0)).unwrap();
            let mut st = Stats::default();
            for (row, &k) in keys.iter().enumerate() {
                assert_eq!(
                    g.find_key(0, k, hash_key(k, FP_SEED), &mut st),
                    Some(row),
                    "key {k} in {:?}..",
                    &keys[..keys.len().min(3)]
                );
            }
            for i in 0..500u64 {
                let probe = splitmix64(i).wrapping_mul(2) | 1;
                if !keys.contains(&probe) {
                    assert_eq!(g.find_key(0, probe, hash_key(probe, FP_SEED), &mut st), None);
                }
            }
        }
    }

    #[test]
    fn sort_bounds_track_the_sort_column() {
        let keys: Vec<u64> = (100..200).collect();
        let b = block_of(&keys);
        let g = Granule::build(&b, 0, 100, Some(0), Some(0)).unwrap();
        assert_eq!(g.sort_min, 100);
        assert_eq!(g.sort_max, 199);
    }

    #[test]
    fn lower_bound_locates_range_starts() {
        let keys: Vec<u64> = (0..100u64).map(|i| i * 10).collect();
        let b = block_of(&keys);
        let g = Granule::build(&b, 0, 100, Some(0), Some(0)).unwrap();
        assert_eq!(g.lower_bound(0, 0), 0);
        assert_eq!(g.lower_bound(0, 55), 6); // first >= 55 is 60 at index 6
        assert_eq!(g.lower_bound(0, 990), 99);
        assert_eq!(g.lower_bound(0, 100_000), 100);
    }

    #[test]
    fn granule_without_pk_column_builds_fine() {
        let keys: Vec<u64> = (0..50).collect();
        let b = block_of(&keys);
        let g = Granule::build(&b, 0, 50, None, None).unwrap();
        assert!(g.pk.is_none());
        assert_eq!(g.len, 50);
        assert_eq!(g.columns.len(), 2);
    }

    #[test]
    fn pk_index_survives_a_parts_roundtrip() {
        let mut keys: Vec<u64> = (0..500u64).map(splitmix64).collect();
        keys.sort_unstable();
        let b = block_of(&keys);
        let g = Granule::build(&b, 0, keys.len(), Some(0), Some(0)).unwrap();
        let idx = g.pk.as_ref().unwrap();
        let back = PkIndex::from_parts(idx.to_parts()).unwrap();
        for (row, &k) in keys.iter().enumerate() {
            assert_eq!(
                back.candidate(k, hash_key(k, FP_SEED), keys.len()),
                Some(row)
            );
        }
    }
}
