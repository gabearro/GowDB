//! Order-preserving lane codec: logical value <-> the `u64` we bit-pack.
//!
//! Every column is stored as `u64` lanes. The mapping is chosen so that
//! **lane order equals logical order**, which is what makes the whole index
//! stack work on compressed data:
//!
//!   * the sparse index, the O(1) router and `locate_granule` compare lanes;
//!   * zone maps are `(base, max_lane)` straight out of the FOR metadata;
//!   * range predicates become integer comparisons on packed words;
//!   * interpolation search runs directly over packed lanes.
//!
//! ## Why not zigzag
//!
//! Zigzag (`(v << 1) ^ (v >> 63)`) is the usual choice for packing signed
//! integers, and it is what this engine used before it had a sort key. It is
//! *not* order preserving: `zz(0)=0, zz(-1)=1, zz(1)=2`, so sorted data stops
//! being sorted once packed.
//!
//! Sign-flip (`v as u64 ^ 1<<63`) is order preserving *and* compresses at
//! least as well, because FOR only cares about `max - min`:
//!
//! | values           | zigzag span | sign-flip span |
//! |------------------|-------------|----------------|
//! | `[-1000, 1000]`  | 2000        | 2000           |
//! | `[1000, 1100]`   | 200         | 100            |
//! | `[-100, -50]`    | 100         | 50             |
//!
//! Sign-flip wins whenever the range does not straddle zero, and ties when it
//! does. There is no case where zigzag is better, so there is no tradeoff to
//! make here.
//!
//! Floats use the standard total-order transform. NaN sorts at an extreme,
//! which is consistent with [`crate::types::Value`]'s ordering.

const SIGN: u64 = 1 << 63;

#[inline(always)]
pub fn i64_to_lane(v: i64) -> u64 {
    (v as u64) ^ SIGN
}

#[inline(always)]
pub fn lane_to_i64(l: u64) -> i64 {
    (l ^ SIGN) as i64
}

/// Canonical NaN lane: above every finite value and both infinities, matching
/// [`crate::types::Value`]'s "NaN sorts last".
const NAN_LANE: u64 = u64::MAX;

#[inline(always)]
pub fn f64_to_lane(x: f64) -> u64 {
    // The two canonicalizations below exist so that lane order is *exactly*
    // `Value`'s order. Without them the radix-sort path and the comparison
    // path disagree: raw bit patterns put negative NaN first (it looks like a
    // very negative number) and separate -0.0 from +0.0, while `Value` sorts
    // NaN last and treats the zeros as one value. Two sort paths that disagree
    // is a correctness bug, not a performance detail.
    if x.is_nan() {
        return NAN_LANE;
    }
    let x = if x == 0.0 { 0.0 } else { x };
    let b = x.to_bits();
    // Negative floats have descending bit patterns, so invert them entirely;
    // non-negative floats only need the sign bit set to sort above negatives.
    if b & SIGN != 0 {
        !b
    } else {
        b | SIGN
    }
}

#[inline(always)]
pub fn lane_to_f64(l: u64) -> f64 {
    if l & SIGN != 0 {
        f64::from_bits(l & !SIGN)
    } else {
        f64::from_bits(!l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i64_roundtrips_at_the_extremes() {
        for v in [0i64, 1, -1, i64::MAX, i64::MIN, i64::MAX - 1, i64::MIN + 1, 42, -42] {
            assert_eq!(lane_to_i64(i64_to_lane(v)), v, "v={v}");
        }
    }

    #[test]
    fn i64_lane_order_matches_value_order() {
        let vals = [i64::MIN, -1_000_000, -1, 0, 1, 1_000_000, i64::MAX];
        for w in vals.windows(2) {
            assert!(
                i64_to_lane(w[0]) < i64_to_lane(w[1]),
                "{} !< {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn i64_lane_span_never_exceeds_zigzag() {
        use crate::common::zz_enc;
        // FOR width is driven by max(lane) - min(lane) over the whole column,
        // so the span has to be measured across sampled values -- zigzag is
        // not monotonic, and its endpoints say nothing about its span.
        let span = |vals: &[i64], f: fn(i64) -> u64| -> u64 {
            let ls: Vec<u64> = vals.iter().map(|&v| f(v)).collect();
            ls.iter().max().unwrap() - ls.iter().min().unwrap()
        };
        let cases: Vec<Vec<i64>> = vec![
            (-1000..=1000).step_by(7).collect(),
            (1000..=1100).collect(),
            (-100..=-50).collect(),
            vec![0, 1],
            vec![i64::MIN, i64::MIN + 5, -1, 0],
        ];
        for vals in cases {
            let flip = span(&vals, i64_to_lane);
            let zig = span(&vals, zz_enc);
            assert!(
                flip <= zig,
                "sign-flip worse for {:?}..: {flip} > {zig}",
                &vals[..vals.len().min(3)]
            );
        }
        // And strictly better whenever the range does not straddle zero.
        let clustered: Vec<i64> = (1_000_000..1_000_100).collect();
        assert!(span(&clustered, i64_to_lane) < span(&clustered, zz_enc));
    }

    #[test]
    fn f64_roundtrips() {
        for x in [0.0f64, 1.0, -1.0, 1e300, -1e300, f64::MIN, f64::MAX,
                  f64::INFINITY, f64::NEG_INFINITY, 1.5, -0.25] {
            let back = lane_to_f64(f64_to_lane(x));
            assert_eq!(back.to_bits(), x.to_bits(), "x={x}");
        }
        // The two canonicalized cases round-trip by *value*, not by bits:
        // -0.0 stores as +0.0 and every NaN stores as one NaN. Both match how
        // `Value` compares them, which is what keeps sorting consistent.
        assert_eq!(lane_to_f64(f64_to_lane(-0.0)), 0.0);
        assert!(lane_to_f64(f64_to_lane(f64::NAN)).is_nan());
        assert!(lane_to_f64(f64_to_lane(-f64::NAN)).is_nan());
    }

    #[test]
    fn f64_lane_order_matches_value_ordering_exactly() {
        use crate::types::Value;
        // The radix-sort path orders by lane; the comparison path orders by
        // `Value`. They must agree on every pair, including the awkward ones.
        let xs = [
            f64::NEG_INFINITY, -1e300, -1.0, -0.0, 0.0, 1.0, 1e300,
            f64::INFINITY, f64::NAN,
        ];
        for &a in &xs {
            for &b in &xs {
                let by_lane = f64_to_lane(a).cmp(&f64_to_lane(b));
                let by_value = Value::Float(a).cmp(&Value::Float(b));
                assert_eq!(by_lane, by_value, "disagree on {a} vs {b}");
            }
        }
    }

    #[test]
    fn f64_lane_order_matches_value_order() {
        let vals = [
            f64::NEG_INFINITY, -1e300, -1.0, -0.25, -0.0, 0.0, 0.25, 1.0, 1e300,
            f64::INFINITY,
        ];
        for w in vals.windows(2) {
            let (a, b) = (f64_to_lane(w[0]), f64_to_lane(w[1]));
            // -0.0 and 0.0 are distinct bit patterns but equal values, so the
            // only requirement is non-decreasing.
            assert!(a <= b, "{} !<= {} ({a} vs {b})", w[0], w[1]);
        }
        // strict where the values are strictly ordered
        assert!(f64_to_lane(-1.0) < f64_to_lane(1.0));
        assert!(f64_to_lane(f64::NEG_INFINITY) < f64_to_lane(0.0));
    }
}
