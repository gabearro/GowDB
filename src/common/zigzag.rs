//! Zigzag transforms. Signed columns are zigzag-encoded before
//! frame-of-reference packing so that a column straddling zero (or holding
//! small negatives) packs into a few bits instead of ~64.

#[inline(always)]
pub fn zz_enc(v: i64) -> u64 {
    ((v as u64) << 1) ^ ((v >> 63) as u64)
}

#[inline(always)]
pub fn zz_dec(z: u64) -> i64 {
    ((z >> 1) as i64) ^ -((z & 1) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_including_extremes() {
        for v in [0i64, 1, -1, 42, -42, i64::MAX, i64::MIN, i64::MAX - 1, i64::MIN + 1] {
            assert_eq!(zz_dec(zz_enc(v)), v, "v={v}");
        }
    }

    #[test]
    fn small_magnitudes_stay_small() {
        // The whole point: |v| small => encoded value small => few bits.
        for v in -1000i64..=1000 {
            assert!(zz_enc(v) <= 2001, "v={v}");
        }
    }
}
