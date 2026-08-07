//! Frame-of-reference bit packing.
//!
//! `value[i] = base + payload[i]`, where the payload width is chosen per chunk
//! from the actual value range. The physical layout is picked per chunk for
//! decode speed:
//!
//!   * `width == 0`   -> constant column: no payload stored at all.
//!   * `width 1..=32` -> WORD-ALIGNED LANES: `floor(64/w)` values per word,
//!     never straddling a word boundary. Random access replaces the divide
//!     with a magic-reciprocal multiply; bulk decode reads each word once and
//!     extracts lanes in a branch-free shift loop LLVM unrolls/vectorizes.
//!     Costs up to `64 mod (64/w*w)` pad bits per word.
//!   * `width 33..=64` -> TIGHT STRADDLED: O(1) access via one shifted `u128`
//!     load. Wide payloads compress < 2x regardless and in practice are key
//!     columns -- point-verified and binary-searched, never bulk-decoded --
//!     so they keep maximum density while scan columns get the fast layout.
//!
//! The critical property for this engine is that **random access stays O(1)**:
//! MPH point lookups verify keys directly against compressed data, with no
//! decompression step and no row groups to inflate.

/// What keeps a `PackedU64`'s words alive.
///
/// A column is either built in memory or read straight out of a memory-mapped
/// part file. The mapped case is the point: it lets a table larger than RAM be
/// queried, because the OS pages granules in on demand and never materializes
/// the ones a query prunes.
///
/// Neither payload is ever read back out. Reads go through the raw `ptr`,
/// which is why the pointer exists at all -- these variants are here only to
/// run the right destructor at the right time, so `dead_code` sees fields
/// nothing touches and is, narrowly, correct.
#[allow(dead_code)]
enum Backing {
    Owned(Vec<u64>),
    /// Words live inside something else -- a mapping -- which this `Arc` keeps
    /// alive for exactly as long as the column that reads from it.
    Borrowed(std::sync::Arc<dyn Send + Sync>),
}

/// FOR-packed `u64` column chunk.
///
/// The word array is reached through a raw pointer rather than a slice or an
/// enum. Both alternatives cost a branch or a bounds check on the innermost
/// read in the engine -- `get` is the point-lookup path and the interpolation
/// search -- and the pointer is equally sound: it addresses memory owned by
/// `backing`, which lives exactly as long as `self`, and nothing here is ever
/// mutated after construction.
pub struct PackedU64 {
    pub(crate) base: u64,
    pub(crate) width: u32,
    /// Lanes per word; 0 => constant or straddled layout.
    pub(crate) per_word: u32,
    /// `ceil(2^32 / per_word)`: branch-free `i / per_word`.
    pub(crate) recip: u64,
    pub(crate) mask: u64,
    /// Start of the word array. Valid for `len` words for the life of `self`.
    ptr: *const u64,
    len: usize,
    /// +1 pad word so the straddled `get()` never branches on the last value.
    backing: Backing,
}

// SAFETY: `ptr` addresses immutable memory kept alive by `backing`, which is
// itself `Send + Sync` (a `Vec<u64>`, or an `Arc` over a read-only mapping).
// No interior mutability, no aliasing writes -- sharing a `PackedU64` across
// threads shares read-only bytes, which is what parallel scans need.
unsafe impl Send for PackedU64 {}
unsafe impl Sync for PackedU64 {}

impl Clone for PackedU64 {
    /// Always clones into owned words. A mapped column could share its `Arc`,
    /// but clones are rare (compaction, tests) and copying keeps the invariant
    /// "`ptr` points into *this* value's backing" trivially true.
    fn clone(&self) -> PackedU64 {
        PackedU64::from_parts(self.base, self.width, self.as_slice().to_vec())
    }
}

impl PartialEq for PackedU64 {
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base
            && self.width == other.width
            && self.as_slice() == other.as_slice()
    }
}
impl Eq for PackedU64 {}

impl PackedU64 {
    pub fn pack(vals: &[u64]) -> Self {
        let n = vals.len();
        if n == 0 {
            return PackedU64::own(0, 0, 0, 0, 0, vec![0; 2]);
        }
        let base = *vals.iter().min().unwrap();
        let range = *vals.iter().max().unwrap() - base;
        Self::pack_with_base(vals, base, range)
    }

    fn pack_with_base(vals: &[u64], base: u64, range: u64) -> Self {
        let n = vals.len();
        let width = (64 - range.leading_zeros()) as usize;
        let mask = if width == 0 {
            0
        } else if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        if width == 0 {
            return PackedU64::own(base, 0, 0, 0, mask, vec![0; 2]);
        }
        if width <= 32 {
            let per = 64 / width; // 2..=64 lanes
            let recip = ((1u64 << 32) + per as u64 - 1) / per as u64;
            let mut words = vec![0u64; n.div_ceil(per) + 1];
            for (i, &v) in vals.iter().enumerate() {
                words[i / per] |= (v - base) << ((i % per) * width);
            }
            return PackedU64::own(base, width as u32, per as u32, recip, mask, words);
        }
        // straddled
        let mut words = vec![0u64; ((n * width + 63) / 64 + 1).max(2)];
        let mut bit = 0usize;
        for &v in vals {
            let d = v - base;
            let (wi, off) = (bit >> 6, bit & 63);
            words[wi] |= d << off;
            if off + width > 64 {
                words[wi + 1] |= d >> (64 - off);
            }
            bit += width;
        }
        PackedU64::own(base, width as u32, 0, 0, mask, words)
    }

    /// Reconstruct from raw parts. Used by the on-disk reader, which stores
    /// exactly these fields and must not re-derive them.
    pub fn from_parts(base: u64, width: u32, words: Vec<u64>) -> Self {
        let mask = if width == 0 {
            0
        } else if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        if width == 0 || width > 32 {
            PackedU64::own(base, width, 0, 0, mask, words)
        } else {
            let per = 64 / width as usize;
            let recip = ((1u64 << 32) + per as u64 - 1) / per as u64;
            PackedU64::own(base, width, per as u32, recip, mask, words)
        }
    }

    /// Assemble around an owned word array.
    #[inline]
    fn own(base: u64, width: u32, per_word: u32, recip: u64, mask: u64, words: Vec<u64>) -> PackedU64 {
        // The pointer is taken before the `Vec` moves into the struct, which is
        // fine: moving a `Vec` moves its header, never its heap buffer.
        let (ptr, len) = (words.as_ptr(), words.len());
        PackedU64 { base, width, per_word, recip, mask, ptr, len, backing: Backing::Owned(words) }
    }

    /// Build a column whose words live inside `owner` -- a memory-mapped part.
    ///
    /// # Safety
    /// `words` must point at `len` readable, 8-byte-aligned `u64`s that stay
    /// valid and immutable for as long as `owner` is alive, and `owner` must be
    /// what keeps them alive.
    pub unsafe fn from_mapped(
        base: u64,
        width: u32,
        words: *const u64,
        len: usize,
        owner: std::sync::Arc<dyn Send + Sync>,
    ) -> PackedU64 {
        let mask = if width == 0 {
            0
        } else if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        let (per_word, recip) = if width == 0 || width > 32 {
            (0, 0)
        } else {
            let per = 64 / width as usize;
            (per as u32, ((1u64 << 32) + per as u64 - 1) / per as u64)
        };
        PackedU64 {
            base,
            width,
            per_word,
            recip,
            mask,
            ptr: words,
            len,
            backing: Backing::Borrowed(owner),
        }
    }

    /// The word array. One branch-free slice construction; the bounds were
    /// established once at construction.
    #[inline(always)]
    pub fn as_slice(&self) -> &[u64] {
        // SAFETY: see the type-level comment -- `ptr`/`len` describe memory
        // owned by `backing`, immutable for the life of `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// True when the words are read directly out of a mapping rather than the
    /// heap. Only introspection depends on this.
    pub fn is_mapped(&self) -> bool {
        matches!(self.backing, Backing::Borrowed(_))
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> u64 {
        let per = self.per_word as usize;
        if per != 0 {
            // exact for i < 2^26 given per <= 64; granules are <= 1024 rows
            let q = ((i as u64).wrapping_mul(self.recip) >> 32) as usize;
            let r = i - q * per;
            let w = unsafe { *self.ptr.add(q) };
            self.base.wrapping_add((w >> (r * self.width as usize)) & self.mask)
        } else if self.width == 0 {
            self.base
        } else {
            let bit = i * self.width as usize;
            let (wi, off) = (bit >> 6, bit & 63);
            // SAFETY: wi+1 < words.len() by the pad word
            let pair = unsafe {
                (*self.ptr.add(wi) as u128) | ((*self.ptr.add(wi + 1) as u128) << 64)
            };
            self.base.wrapping_add(((pair >> off) as u64) & self.mask)
        }
    }

    /// Bulk decode `[s, e)` into `out[..e-s]`. The aligned path reads each word
    /// once and extracts all lanes with constant-stride shifts.
    pub fn unpack_range(&self, s: usize, e: usize, out: &mut [u64]) {
        debug_assert!(s <= e && out.len() >= e - s);
        let base = self.base;
        let per = self.per_word as usize;
        if per != 0 {
            let w = self.width as usize;
            let mask = self.mask;
            let (mut i, mut k) = (s, 0usize);
            while i < e && i % per != 0 {
                out[k] = base.wrapping_add((self.as_slice()[i / per] >> ((i % per) * w)) & mask);
                i += 1;
                k += 1;
            }
            let nwords = (e - i) / per;
            if nwords > 0 {
                let fw = i / per;
                let dst = &mut out[k..k + nwords * per];
                // monomorphized per lane count: inner loop fully unrolled,
                // lane shifts become vpsrlvq-style SIMD with target-cpu=native
                match per {
                    2 => self.unpack_words::<2>(fw, nwords, dst),
                    3 => self.unpack_words::<3>(fw, nwords, dst),
                    4 => self.unpack_words::<4>(fw, nwords, dst),
                    5 => self.unpack_words::<5>(fw, nwords, dst),
                    6 => self.unpack_words::<6>(fw, nwords, dst),
                    7 => self.unpack_words::<7>(fw, nwords, dst),
                    8 => self.unpack_words::<8>(fw, nwords, dst),
                    9 => self.unpack_words::<9>(fw, nwords, dst),
                    10 => self.unpack_words::<10>(fw, nwords, dst),
                    12 => self.unpack_words::<12>(fw, nwords, dst),
                    16 => self.unpack_words::<16>(fw, nwords, dst),
                    21 => self.unpack_words::<21>(fw, nwords, dst),
                    32 => self.unpack_words::<32>(fw, nwords, dst),
                    64 => self.unpack_words::<64>(fw, nwords, dst),
                    _ => self.unpack_words_dyn(fw, nwords, per, dst),
                }
                i += nwords * per;
                k += nwords * per;
            }
            while i < e {
                out[k] = base.wrapping_add((self.as_slice()[i / per] >> ((i % per) * w)) & mask);
                i += 1;
                k += 1;
            }
        } else if self.width == 0 {
            out[..e - s].fill(base);
        } else {
            let w = self.width as usize;
            let mask = self.mask;
            let mut bit = s * w;
            for o in out[..e - s].iter_mut() {
                let (wi, off) = (bit >> 6, bit & 63);
                let pair = unsafe {
                    (*self.ptr.add(wi) as u128) | ((*self.ptr.add(wi + 1) as u128) << 64)
                };
                *o = base.wrapping_add(((pair >> off) as u64) & mask);
                bit += w;
            }
        }
    }

    #[inline(always)]
    fn unpack_words<const PER: usize>(&self, first_word: usize, nwords: usize, out: &mut [u64]) {
        let (base, w, mask) = (self.base, self.width as usize, self.mask);
        for wi in 0..nwords {
            let word = unsafe { *self.ptr.add(first_word + wi) };
            for l in 0..PER {
                unsafe {
                    *out.get_unchecked_mut(wi * PER + l) =
                        base.wrapping_add((word >> (l * w)) & mask);
                }
            }
        }
    }

    fn unpack_words_dyn(&self, first_word: usize, nwords: usize, per: usize, out: &mut [u64]) {
        let (base, w, mask) = (self.base, self.width as usize, self.mask);
        for wi in 0..nwords {
            let word = unsafe { *self.ptr.add(first_word + wi) };
            for l in 0..per {
                unsafe {
                    *out.get_unchecked_mut(wi * per + l) =
                        base.wrapping_add((word >> (l * w)) & mask);
                }
            }
        }
    }

    #[inline(always)]
    pub fn prefetch(&self, i: usize) {
        let per = self.per_word as usize;
        let wi = if per != 0 {
            ((i as u64).wrapping_mul(self.recip) >> 32) as usize
        } else {
            (i * self.width.max(1) as usize) >> 6
        };
        crate::common::prefetch_read(unsafe { self.ptr.add(wi.min(self.len - 1)) } as *const u8);
    }

    #[inline(always)]
    pub fn base(&self) -> u64 {
        self.base
    }
    #[inline(always)]
    pub fn width(&self) -> u32 {
        self.width
    }
    #[inline(always)]
    pub fn mask(&self) -> u64 {
        self.mask
    }
    /// Inclusive upper bound implied by the FOR metadata alone. Used as a
    /// free per-granule zone map: no value in the chunk can exceed this.
    #[inline(always)]
    pub fn value_ceiling(&self) -> u64 {
        self.base.wrapping_add(self.mask)
    }
    pub fn words(&self) -> &[u64] {
        self.as_slice()
    }

    pub fn bytes(&self) -> usize {
        // A mapped column costs no heap at all; the pages are the OS's.
        match self.backing {
            Backing::Owned(_) => self.len * 8 + 32,
            Backing::Borrowed(_) => 32,
        }
    }
}

impl std::fmt::Debug for PackedU64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PackedU64 {{ base: {}, width: {}, words: {}{} }}",
            self.base,
            self.width,
            self.len,
            if self.is_mapped() { ", mapped" } else { "" }
        )
    }
}

/// Interpolation-guided lower bound directly over packed sorted data.
///
/// Sorted keys are near-uniform inside a granule, so interpolation converges
/// in ~2 probes where binary search needs ~10 -- and each probe is an O(1)
/// packed `get`, so we never materialize the column.
pub fn packed_lower_bound(p: &PackedU64, len: usize, target: u64) -> usize {
    let (mut lo, mut hi) = (0usize, len);
    while lo < hi {
        let mid = if hi - lo > 8 {
            let (klo, khi) = (p.get(lo) as u128, p.get(hi - 1) as u128);
            if khi > klo {
                let t = (target as u128).clamp(klo, khi);
                lo + (((t - klo) * (hi - lo - 1) as u128) / (khi - klo)) as usize
            } else {
                lo + (hi - lo) / 2
            }
        } else {
            lo + (hi - lo) / 2
        };
        let mid = mid.clamp(lo, hi - 1);
        if p.get(mid) < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{splitmix64, zz_dec, zz_enc, GRANULE_SIZE};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            splitmix64(self.0)
        }
    }

    #[test]
    fn packed_roundtrip_all_widths() {
        let mut rng = Rng(3);
        for w in 0..=64u32 {
            let n = 300;
            let base = rng.next() >> 1;
            let vals: Vec<u64> = (0..n)
                .map(|_| {
                    let payload = if w == 0 {
                        0
                    } else if w == 64 {
                        rng.next()
                    } else {
                        rng.next() & ((1u64 << w) - 1)
                    };
                    if w == 64 { payload } else { base.wrapping_add(payload) }
                })
                .collect();
            let p = PackedU64::pack(&vals);
            for (i, &v) in vals.iter().enumerate() {
                assert_eq!(p.get(i), v, "w={w} i={i}");
            }
        }
        // zigzag negatives
        let vals: Vec<u64> = [-5i64, 0, 3, i64::MIN / 2, i64::MAX / 2]
            .iter()
            .map(|&v| zz_enc(v))
            .collect();
        let p = PackedU64::pack(&vals);
        for (i, &z) in vals.iter().enumerate() {
            assert_eq!(zz_dec(p.get(i)), zz_dec(z));
        }
    }

    #[test]
    fn magic_reciprocal_division_is_exact() {
        for per in 1u64..=64 {
            let recip = ((1u64 << 32) + per - 1) / per;
            for i in 0u64..8192 {
                assert_eq!((i.wrapping_mul(recip)) >> 32, i / per, "i={i} per={per}");
            }
        }
    }

    #[test]
    fn unpack_range_matches_get_all_layouts() {
        let mut rng = Rng(99);
        for w in 0..=64u32 {
            let n = 1024;
            let vals: Vec<u64> = (0..n)
                .map(|_| {
                    if w == 0 {
                        7777
                    } else if w == 64 {
                        rng.next()
                    } else {
                        1_000 + (rng.next() & ((1u64 << w) - 1))
                    }
                })
                .collect();
            let p = PackedU64::pack(&vals);
            for (i, &v) in vals.iter().enumerate() {
                assert_eq!(p.get(i), v, "get w={w} i={i}");
            }
            let mut buf = [0u64; GRANULE_SIZE];
            for &(s, e) in &[(0usize, n), (1, n - 1), (3, 700), (511, 513), (37, 38), (5, 5)] {
                p.unpack_range(s, e, &mut buf);
                for i in s..e {
                    assert_eq!(buf[i - s], vals[i], "unpack w={w} s={s} e={e} i={i}");
                }
            }
        }
    }

    #[test]
    fn from_parts_matches_pack() {
        let mut rng = Rng(1234);
        for w in [0u32, 1, 5, 17, 32, 33, 47, 64] {
            let vals: Vec<u64> = (0..500)
                .map(|_| if w == 0 { 9 } else if w == 64 { rng.next() } else { 9 + (rng.next() & ((1u64 << w) - 1)) })
                .collect();
            let p = PackedU64::pack(&vals);
            let q = PackedU64::from_parts(p.base(), p.width(), p.words().to_vec());
            for i in 0..vals.len() {
                assert_eq!(p.get(i), q.get(i), "w={w} i={i}");
            }
        }
    }

    #[test]
    fn packed_lower_bound_matches_raw() {
        let mut rng = Rng(11);
        for _ in 0..30 {
            let mut vals: Vec<u64> = (0..400).map(|_| rng.next() % 100_000).collect();
            vals.sort_unstable();
            let p = PackedU64::pack(&vals);
            for _ in 0..200 {
                let t = rng.next() % 120_000;
                assert_eq!(
                    packed_lower_bound(&p, vals.len(), t),
                    vals.partition_point(|&v| v < t)
                );
            }
        }
    }

    #[test]
    fn value_ceiling_bounds_every_element() {
        let mut rng = Rng(5150);
        for _ in 0..50 {
            let vals: Vec<u64> = (0..200).map(|_| rng.next() % 1_000_000).collect();
            let p = PackedU64::pack(&vals);
            for &v in &vals {
                assert!(v >= p.base() && v <= p.value_ceiling());
            }
        }
    }
}
