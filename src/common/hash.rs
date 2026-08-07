//! Hashing primitives shared by the minimal perfect hash, the bloom filters,
//! the hash-aggregation tables and the join build side.
//!
//! Everything here is a multiply-shift construction: no modulo, no branches,
//! and a single `mulx` on the critical path.

use std::hash::{BuildHasherDefault, Hasher};

/// Fold a 128-bit product back into 64 bits. The xor of the high and low
/// halves keeps entropy from both operands, which is what lets a single
/// `mum` behave like a full mixing round.
#[inline(always)]
pub fn mum(a: u64, b: u64) -> u64 {
    let r = (a as u128).wrapping_mul(b as u128);
    (r as u64) ^ ((r >> 64) as u64)
}

/// Seed enters as an odd multiplier: distinct seeds give decorrelated hash
/// families (required for CHD displacement search to converge).
#[inline(always)]
pub fn hash_key(key: u64, seed: u64) -> u64 {
    mum(
        mum(key ^ 0xA076_1D64_78BD_642F, seed | 1),
        seed ^ 0xE703_7ED1_A0B4_28DB,
    )
}

/// Lemire fastrange: hash -> [0, n) with one multiply, no div/mod.
#[inline(always)]
pub fn fastrange(h: u64, n: usize) -> usize {
    (((h as u128) * (n as u128)) >> 64) as usize
}

/// Per-slot fingerprint width. 6 bits => a 1/64 false-positive rate on the
/// fused fingerprint|rank record.
pub const FP_BITS: u32 = 6;
pub const FP_SEED: u64 = 0x243F_6A88_85A3_08D3;

/// Key fingerprint from the precomputed FP hash's top bits.
#[inline(always)]
pub fn fp6(fph: u64) -> u64 {
    fph >> (64 - FP_BITS)
}

#[inline(always)]
pub fn prefetch_read(p: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch(p as *const i8, _MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // `prfm pldl1keep` — the AArch64 equivalent of _MM_HINT_T0.
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = p;
}

#[inline(always)]
pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Hash a byte string. Used for `String` group keys and dictionary probes.
#[inline]
pub fn hash_bytes(bytes: &[u8], seed: u64) -> u64 {
    let mut h = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        h = mum(h ^ u64::from_le_bytes(c.try_into().unwrap()), 0xFF51_AFD7_ED55_8CCD);
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut tail = 0u64;
        for (i, &b) in rem.iter().enumerate() {
            tail |= (b as u64) << (i * 8);
        }
        h = mum(h ^ tail, 0xC4CE_B9FE_1A85_EC53);
    }
    mum(h ^ bytes.len() as u64, 0x9E37_79B9_7F4A_7C15)
}

/// Identity-ish hasher for `HashMap<u64, _>`: the keys we feed it are already
/// well-mixed, so one `mum` round is enough and saves SipHash entirely.
#[derive(Default, Clone, Copy)]
pub struct FastHasher(u64);

impl Hasher for FastHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline(always)]
    fn write_u64(&mut self, x: u64) {
        self.0 = mum(x ^ 0x2545_F491_4F6C_DD1D, 0x9E37_79B9_7F4A_7C15);
    }
    #[inline(always)]
    fn write_u32(&mut self, x: u32) {
        self.write_u64(x as u64);
    }
    #[inline(always)]
    fn write_usize(&mut self, x: usize) {
        self.write_u64(x as u64);
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0 = hash_bytes(bytes, self.0);
    }
}

pub type FastBuild = BuildHasherDefault<FastHasher>;
pub type FastMap<K, V> = std::collections::HashMap<K, V, FastBuild>;
pub type FastSet<K> = std::collections::HashSet<K, FastBuild>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fastrange_stays_in_bounds() {
        for n in [1usize, 2, 3, 7, 1024, 65_537] {
            for i in 0..2000u64 {
                assert!(fastrange(splitmix64(i), n) < n);
            }
        }
    }

    #[test]
    fn hash_key_decorrelates_across_seeds() {
        // Two seeds must not agree on the low bits for many keys, otherwise
        // CHD displacement search would stall.
        let mut agree = 0;
        for i in 0..10_000u64 {
            if hash_key(i, 1) & 0xFFFF == hash_key(i, 2) & 0xFFFF {
                agree += 1;
            }
        }
        assert!(agree < 20, "seeds correlated: {agree}");
    }

    #[test]
    fn hash_bytes_is_length_sensitive() {
        assert_ne!(hash_bytes(b"a\0", 0), hash_bytes(b"a", 0));
        assert_ne!(hash_bytes(b"", 0), hash_bytes(b"\0", 0));
        assert_eq!(hash_bytes(b"clickhouse", 7), hash_bytes(b"clickhouse", 7));
    }
}
