//! LSD radix sort for `(sort_key, row_id)` pairs.
//!
//! Parts are built by sorting rows on the table's ORDER BY key. Comparison
//! sorting costs `O(n log n)` comparisons; four 16-bit LSD passes cost
//! `O(4n)` with no branches at all, which is a large constant-factor win at
//! the millions-of-rows-per-flush scale we build parts at.
//!
//! We sort *permutations*, not rows: the payload is a `u32` row id, so each
//! pass moves 12 bytes regardless of how wide the table is, and the caller
//! gathers every column once at the end.

/// `(key, row_id)`. Keys are the packed `u64` lane of the sort column, which
/// is order-preserving for every type we support: unsigned integers directly,
/// signed via zigzag-then-flip, strings via order-preserving dictionary codes.
pub type KeyedRow = (u64, u32);

/// Sort by key ascending, stable with respect to the original row order.
pub fn radix_sort(rows: &mut Vec<KeyedRow>) {
    let n = rows.len();
    if n < 128 {
        rows.sort_by_key(|r| r.0);
        return;
    }
    let mut keys: Vec<u64> = rows.iter().map(|r| r.0).collect();
    let mut idx: Vec<u32> = rows.iter().map(|r| r.1).collect();
    radix_sort_soa(&mut keys, &mut idx);
    for (i, r) in rows.iter_mut().enumerate() {
        *r = (keys[i], idx[i]);
    }
}

/// Sort `idx` by `keys`, ascending and stable, as parallel arrays.
///
/// The array-of-structs form, `Vec<(u64, u32)>`, is 16 bytes per element:
/// `u64` forces 8-byte alignment, so the `u32` costs 8. Splitting the arrays
/// stores exactly 12 bytes per row, and since LSD radix needs a scratch buffer
/// of the same size, that is 24 bytes per row instead of 32 -- a quarter of
/// the sort's peak memory reclaimed, on the pass that dominates peak memory
/// during an unsorted bulk load.
///
/// Both arrays are walked sequentially in every pass, so nothing is lost in
/// locality by splitting them.
pub fn radix_sort_soa(keys: &mut Vec<u64>, idx: &mut Vec<u32>) {
    debug_assert_eq!(keys.len(), idx.len());
    let n = keys.len();
    if n < 128 {
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by_key(|&i| keys[i as usize]);
        let k: Vec<u64> = order.iter().map(|&i| keys[i as usize]).collect();
        let v: Vec<u32> = order.iter().map(|&i| idx[i as usize]).collect();
        *keys = k;
        *idx = v;
        return;
    }
    let mut kbuf: Vec<u64> = vec![0; n];
    let mut ibuf: Vec<u32> = vec![0; n];
    let mut counts = vec![0u32; 65_536];
    for pass in 0..4 {
        let shift = pass * 16;
        // Skip a pass entirely when every key shares these 16 bits -- common
        // for clustered keys, where the top two passes are usually no-ops.
        let first = (keys[0] >> shift) & 0xFFFF;
        if keys.iter().all(|k| (k >> shift) & 0xFFFF == first) {
            continue;
        }
        counts.iter_mut().for_each(|c| *c = 0);
        for k in keys.iter() {
            counts[((k >> shift) & 0xFFFF) as usize] += 1;
        }
        let mut sum = 0u32;
        for c in counts.iter_mut() {
            let t = *c;
            *c = sum;
            sum += t;
        }
        for i in 0..n {
            let d = ((keys[i] >> shift) & 0xFFFF) as usize;
            let at = counts[d] as usize;
            unsafe {
                *kbuf.get_unchecked_mut(at) = keys[i];
                *ibuf.get_unchecked_mut(at) = idx[i];
            }
            counts[d] += 1;
        }
        std::mem::swap(keys, &mut kbuf);
        std::mem::swap(idx, &mut ibuf);
    }
}

/// Sort by a composite key, most significant component first. Used for
/// multi-column ORDER BY. Runs LSD radix per component in reverse order,
/// relying on each pass being stable.
pub fn radix_sort_composite(rows: &mut Vec<(Vec<u64>, u32)>, ncols: usize) {
    if rows.len() < 2 || ncols == 0 {
        return;
    }
    // Least significant component first; stability preserves earlier passes.
    for c in (0..ncols).rev() {
        let mut keyed: Vec<KeyedRow> = rows
            .iter()
            .enumerate()
            .map(|(i, (k, _))| (k[c], i as u32))
            .collect();
        radix_sort(&mut keyed);
        let reordered: Vec<(Vec<u64>, u32)> = keyed
            .iter()
            .map(|&(_, i)| rows[i as usize].clone())
            .collect();
        *rows = reordered;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn soa_matches_the_pair_form_and_is_stable() {
        for &n in &[0usize, 1, 5, 127, 128, 5000, 100_000] {
            let mut keys: Vec<u64> = (0..n).map(|i| splitmix64(i as u64) % 1000).collect();
            let mut idx: Vec<u32> = (0..n as u32).collect();
            let mut expect: Vec<KeyedRow> =
                keys.iter().zip(&idx).map(|(&k, &i)| (k, i)).collect();
            expect.sort_by_key(|r| r.0);

            radix_sort_soa(&mut keys, &mut idx);
            let got: Vec<KeyedRow> = keys.iter().zip(&idx).map(|(&k, &i)| (k, i)).collect();
            assert_eq!(got, expect, "n={n}");
            // stability: equal keys keep ascending row ids
            for w in got.windows(2) {
                if w[0].0 == w[1].0 {
                    assert!(w[0].1 < w[1].1, "unstable at n={n}");
                }
            }
        }
    }

    #[test]
    fn soa_uses_less_memory_than_the_pair_form() {
        // The whole point of splitting: (u64, u32) pads to 16 bytes.
        assert_eq!(std::mem::size_of::<KeyedRow>(), 16);
        assert_eq!(
            std::mem::size_of::<u64>() + std::mem::size_of::<u32>(),
            12,
            "SoA stores 12 bytes per row where AoS stores 16"
        );
    }

    use super::*;
    use crate::common::splitmix64;

    #[test]
    fn matches_std_sort() {
        for &n in &[0usize, 1, 5, 127, 128, 5000, 100_000] {
            let mut rows: Vec<KeyedRow> =
                (0..n).map(|i| (splitmix64(i as u64), i as u32)).collect();
            let mut expect = rows.clone();
            expect.sort_by_key(|r| r.0);
            radix_sort(&mut rows);
            assert_eq!(rows, expect, "n={n}");
        }
    }

    #[test]
    fn is_stable_on_duplicate_keys() {
        let mut rows: Vec<KeyedRow> = (0..1000u32).map(|i| (i as u64 % 7, i)).collect();
        let mut expect = rows.clone();
        expect.sort_by_key(|r| r.0);
        radix_sort(&mut rows);
        assert_eq!(rows, expect);
        // within each key group the row ids must stay ascending
        for w in rows.windows(2) {
            if w[0].0 == w[1].0 {
                assert!(w[0].1 < w[1].1);
            }
        }
    }

    #[test]
    fn skips_uniform_high_passes() {
        // Clustered keys: top 32 bits identical. Result must still be sorted.
        let mut rows: Vec<KeyedRow> = (0..5000u32)
            .map(|i| (1_000_000_000u64 + (splitmix64(i as u64) & 0xFFFF), i))
            .collect();
        let mut expect = rows.clone();
        expect.sort_by_key(|r| r.0);
        radix_sort(&mut rows);
        assert_eq!(rows, expect);
    }

    #[test]
    fn composite_sorts_major_to_minor() {
        let mut rows: Vec<(Vec<u64>, u32)> = vec![
            (vec![2, 1], 0),
            (vec![1, 9], 1),
            (vec![2, 0], 2),
            (vec![1, 3], 3),
        ];
        radix_sort_composite(&mut rows, 2);
        let keys: Vec<Vec<u64>> = rows.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(keys, vec![vec![1, 3], vec![1, 9], vec![2, 0], vec![2, 1]]);
    }
}
