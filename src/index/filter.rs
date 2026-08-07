//! Split-block bloom filter, one per part, 6 bits/key.
//!
//! A miss skips a whole foreign part with a *single cache-line probe* before
//! any MPH work. All eight probes land in one 64-byte block, so the cost is
//! one cache miss rather than eight. It reuses the fingerprint hash the caller
//! already computed for the granule lookup, so filtering adds zero hashing.
//!
//! Only consulted when multiple parts exist; after compaction the per-granule
//! fingerprints alone carry the filtering and a single-part table stores no
//! filter at all.

use crate::common::{fastrange, hash_key, FP_SEED};

pub const SEG_FILTER_BITS_PER_KEY: usize = 6;

const FILTER_SALTS: [u64; 8] = [
    0x47B6_137B_4497_4D91,
    0x8824_AD5B_A2B7_289D,
    0x7054_95C7_2DF1_424B,
    0x9EFC_4947_5C6B_FB31,
    0x5C6B_FB31_7054_95C7,
    0x2DF1_424B_9EFC_4947,
    0x4497_4D91_8824_AD5B,
    0xA2B7_289D_47B6_137B,
];

#[repr(align(64))]
#[derive(Clone, Copy)]
struct FilterBlock([u64; 8]);

#[derive(Clone)]
pub struct PartFilter {
    blocks: Vec<FilterBlock>,
}

impl PartFilter {
    pub fn new(n: usize) -> Self {
        let nblocks = ((n * SEG_FILTER_BITS_PER_KEY + 511) / 512).max(1);
        PartFilter { blocks: vec![FilterBlock([0; 8]); nblocks] }
    }

    #[inline]
    pub fn insert(&mut self, key: u64) {
        self.insert_hash(hash_key(key, FP_SEED));
    }

    #[inline]
    pub fn insert_hash(&mut self, h: u64) {
        let nb = self.blocks.len();
        let blk = &mut self.blocks[fastrange(h, nb)].0;
        for i in 0..8 {
            blk[i] |= 1u64 << (h.wrapping_mul(FILTER_SALTS[i]) >> 58);
        }
    }

    #[inline(always)]
    pub fn contains_hash(&self, h: u64) -> bool {
        let blk = unsafe { &self.blocks.get_unchecked(fastrange(h, self.blocks.len())).0 };
        // Branch-free AND-reduce: no early exit, no mispredicts.
        let mut ok = 1u64;
        for i in 0..8 {
            ok &= blk[i] >> (h.wrapping_mul(FILTER_SALTS[i]) >> 58);
        }
        ok & 1 == 1
    }

    #[inline]
    pub fn contains(&self, key: u64) -> bool {
        self.contains_hash(hash_key(key, FP_SEED))
    }

    /// Flat `u64` view for the on-disk writer.
    pub fn as_words(&self) -> &[u64] {
        // SAFETY: FilterBlock is repr(align(64)) over [u64; 8] -- exactly 8
        // u64 with no padding, so the block array is a contiguous u64 array.
        unsafe {
            std::slice::from_raw_parts(self.blocks.as_ptr() as *const u64, self.blocks.len() * 8)
        }
    }

    pub fn from_words(words: &[u64]) -> Self {
        let nblocks = words.len() / 8;
        let mut blocks = vec![FilterBlock([0; 8]); nblocks.max(1)];
        for (bi, chunk) in words.chunks_exact(8).enumerate() {
            blocks[bi].0.copy_from_slice(chunk);
        }
        PartFilter { blocks }
    }

    pub fn bytes(&self) -> usize {
        self.blocks.len() * 64
    }
}

impl std::fmt::Debug for PartFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PartFilter({} blocks, {} bytes)", self.blocks.len(), self.bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::splitmix64;

    #[test]
    fn no_false_negatives() {
        let mut f = PartFilter::new(10_000);
        let keys: Vec<u64> = (0..10_000u64).map(splitmix64).collect();
        for &k in &keys {
            f.insert(k);
        }
        for &k in &keys {
            assert!(f.contains(k), "false negative on {k}");
        }
    }

    #[test]
    fn false_positive_rate_is_sane() {
        let mut f = PartFilter::new(10_000);
        for i in 0..10_000u64 {
            f.insert(splitmix64(i));
        }
        let mut fp = 0;
        for i in 0..100_000u64 {
            if f.contains(splitmix64(1_000_000 + i)) {
                fp += 1;
            }
        }
        // 6 bits/key split-block lands around 5-10%; assert well clear of
        // "the filter does nothing".
        assert!(fp < 20_000, "false positive rate too high: {fp}/100000");
    }

    #[test]
    fn roundtrips_through_words() {
        let mut f = PartFilter::new(1000);
        for i in 0..1000u64 {
            f.insert(splitmix64(i));
        }
        let back = PartFilter::from_words(f.as_words());
        for i in 0..1000u64 {
            assert!(back.contains(splitmix64(i)));
        }
        assert_eq!(f.as_words(), back.as_words());
    }
}
