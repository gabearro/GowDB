//! CHD minimal perfect hash over a granule's primary-key column.
//!
//! `lookup(k)` returns a slot in `[0, n)` that is unique across the n keys the
//! table was built from -- no probing, no collision chain, one dependent load
//! for the displacement seed and one multiply. Foreign keys map somewhere in
//! range too, which is why every caller must verify against the stored key.
//!
//! The displacement seeds are themselves FOR bit-packed: they are small
//! integers, so the index costs a fraction of a byte per key.

use crate::common::{fastrange, hash_key, Result};
use crate::encoding::PackedU64;

const MPH_BUCKET_AVG: usize = 4;

pub struct Mph {
    seeds: PackedU64,
    nb: u32,
    gs: u64,
    n: usize,
}

impl Mph {
    pub fn build(keys: &[u64]) -> Self {
        let n = keys.len();
        if n == 0 {
            return Mph { seeds: PackedU64::pack(&[]), nb: 1, gs: 0, n: 0 };
        }
        let nb = n.div_ceil(MPH_BUCKET_AVG);
        'global: for attempt in 0..256u64 {
            let gs = attempt.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x51_7C_C1_B7;
            let mut count = vec![0u32; nb + 1];
            let bids: Vec<u32> = keys
                .iter()
                .map(|&k| fastrange(hash_key(k, gs), nb) as u32)
                .collect();
            for &b in &bids {
                count[b as usize + 1] += 1;
            }
            for i in 0..nb {
                count[i + 1] += count[i];
            }
            let mut flat = vec![0u64; n];
            let mut cursor = count.clone();
            for (i, &k) in keys.iter().enumerate() {
                let b = bids[i] as usize;
                flat[cursor[b] as usize] = k;
                cursor[b] += 1;
            }
            // Largest buckets first: they are the hardest to place, and doing
            // them while the occupancy map is sparse is what makes CHD
            // converge in near-linear time.
            let mut order: Vec<u32> = (0..nb as u32).collect();
            order.sort_unstable_by_key(|&b| {
                std::cmp::Reverse(count[b as usize + 1] - count[b as usize])
            });

            let mut occupied = vec![0u64; n.div_ceil(64)];
            let mut seeds = vec![0u32; nb];
            for &bi in &order {
                let (lo, hi) = (count[bi as usize] as usize, count[bi as usize + 1] as usize);
                if lo == hi {
                    continue;
                }
                let bucket = &flat[lo..hi];
                let mut done = false;
                'seed: for d in 1..=1_000_000u32 {
                    let s = gs ^ ((d as u64) << 32);
                    let mut placed = [0u32; 64];
                    let mut np = 0usize;
                    for &k in bucket {
                        let p = fastrange(hash_key(k, s), n) as u32;
                        if occupied[p as usize / 64] >> (p % 64) & 1 == 1
                            || placed[..np].contains(&p)
                        {
                            continue 'seed;
                        }
                        if np == placed.len() {
                            // Bucket larger than the scratch array: give up on
                            // this global seed rather than overflow.
                            continue 'global;
                        }
                        placed[np] = p;
                        np += 1;
                    }
                    for &p in &placed[..np] {
                        occupied[p as usize / 64] |= 1 << (p % 64);
                    }
                    seeds[bi as usize] = d;
                    done = true;
                    break;
                }
                if !done {
                    continue 'global;
                }
            }
            let seeds64: Vec<u64> = seeds.iter().map(|&d| d as u64).collect();
            return Mph { seeds: PackedU64::pack(&seeds64), nb: nb as u32, gs, n };
        }
        unreachable!("MPH construction failed after 256 global seeds");
    }

    #[inline(always)]
    pub fn lookup(&self, key: u64) -> usize {
        let d = self.seeds.get(fastrange(hash_key(key, self.gs), self.nb as usize));
        fastrange(hash_key(key, self.gs ^ (d << 32)), self.n)
    }

    /// Raw parts for the on-disk writer.
    pub fn to_parts(&self) -> (u64, u32, usize, u64, u32, Vec<u64>) {
        (
            self.gs,
            self.nb,
            self.n,
            self.seeds.base(),
            self.seeds.width(),
            self.seeds.words().to_vec(),
        )
    }

    pub fn from_parts(
        gs: u64,
        nb: u32,
        n: usize,
        seed_base: u64,
        seed_width: u32,
        seed_words: Vec<u64>,
    ) -> Result<Self> {
        use crate::common::Error;
        if nb == 0 {
            return Err(Error::corruption("mph bucket count is zero"));
        }
        Ok(Mph {
            seeds: PackedU64::from_parts(seed_base, seed_width, seed_words),
            nb,
            gs,
            n,
        })
    }

    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    pub fn bytes(&self) -> usize {
        self.seeds.bytes()
    }
}

impl std::fmt::Debug for Mph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mph({} keys, {} bytes)", self.n, self.bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{splitmix64, GRANULE_SIZE};

    #[test]
    fn is_minimal_and_perfect() {
        for &n in &[1usize, 2, 5, 100, 500, GRANULE_SIZE] {
            let mut keys: Vec<u64> = (0..n as u64).map(splitmix64).collect();
            keys.sort_unstable();
            keys.dedup();
            let mph = Mph::build(&keys);
            let mut seen = vec![false; keys.len()];
            for &k in &keys {
                let p = mph.lookup(k);
                assert!(p < keys.len());
                assert!(!seen[p], "collision at slot {p}");
                seen[p] = true;
            }
            assert!(seen.iter().all(|&s| s));
        }
    }

    #[test]
    fn handles_clustered_keys() {
        // Sequential keys hash-collide differently than random ones; make sure
        // displacement search still converges.
        let keys: Vec<u64> = (1_000_000..1_000_000 + GRANULE_SIZE as u64).collect();
        let mph = Mph::build(&keys);
        let mut seen = vec![false; keys.len()];
        for &k in &keys {
            let p = mph.lookup(k);
            assert!(!seen[p]);
            seen[p] = true;
        }
    }

    #[test]
    fn empty_is_safe() {
        let mph = Mph::build(&[]);
        assert!(mph.is_empty());
        assert_eq!(mph.len(), 0);
    }

    #[test]
    fn roundtrips_through_parts() {
        let keys: Vec<u64> = (0..777u64).map(splitmix64).collect();
        let mph = Mph::build(&keys);
        let (gs, nb, n, b, w, words) = mph.to_parts();
        let back = Mph::from_parts(gs, nb, n, b, w, words).unwrap();
        for &k in &keys {
            assert_eq!(mph.lookup(k), back.lookup(k));
        }
    }
}
