//! Order-preserving per-granule string dictionary.
//!
//! Strings are stored once, sorted, in a single blob with a packed offset
//! array; each row keeps only a dictionary code. Sorting the dictionary is the
//! load-bearing decision:
//!
//!   * `code_a < code_b` **iff** `str_a < str_b`, so equality *and range*
//!     predicates, `min`/`max`, `ORDER BY` and zone-map pruning all run on
//!     packed integer codes without ever materializing a string;
//!   * codes are dense and small, so `PackedU64` squeezes a low-cardinality
//!     column to `ceil(log2(cardinality))` bits/row -- this is what
//!     ClickHouse spells `LowCardinality(String)`, except we apply it
//!     unconditionally and let the width collapse on its own when it pays.
//!
//! A granule holding 1024 rows drawn from 8 distinct strings costs 3 bits/row
//! plus the 8 strings, once.

use crate::common::{FastMap, Result};

#[derive(Clone, Default, PartialEq, Eq)]
pub struct StringDict {
    blob: Vec<u8>,
    /// `n_entries + 1` offsets into `blob`; entry `i` is `blob[off[i]..off[i+1]]`.
    offsets: Vec<u32>,
}

impl StringDict {
    pub fn empty() -> Self {
        StringDict { blob: Vec::new(), offsets: vec![0] }
    }

    /// Build from an already-sorted, already-deduplicated list.
    pub fn from_sorted(sorted_unique: &[&str]) -> Self {
        let total: usize = sorted_unique.iter().map(|s| s.len()).sum();
        let mut blob = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(sorted_unique.len() + 1);
        offsets.push(0u32);
        for s in sorted_unique {
            blob.extend_from_slice(s.as_bytes());
            offsets.push(blob.len() as u32);
        }
        StringDict { blob, offsets }
    }

    pub fn from_parts(blob: Vec<u8>, offsets: Vec<u32>) -> Result<Self> {
        let d = StringDict { blob, offsets };
        d.validate()?;
        Ok(d)
    }

    fn validate(&self) -> Result<()> {
        use crate::common::Error;
        if self.offsets.is_empty() {
            return Err(Error::corruption("string dictionary has no offset array"));
        }
        if *self.offsets.last().unwrap() as usize != self.blob.len() {
            return Err(Error::corruption("string dictionary offsets do not cover the blob"));
        }
        for w in self.offsets.windows(2) {
            if w[1] < w[0] {
                return Err(Error::corruption("string dictionary offsets not monotonic"));
            }
        }
        let text = std::str::from_utf8(&self.blob)?;
        // A whole-blob UTF-8 check is NOT enough. `get` slices the blob at
        // adjacent offsets and calls `from_utf8_unchecked`, so an offset
        // landing in the middle of a multi-byte codepoint would hand out a
        // `&str` over invalid UTF-8 -- undefined behaviour, reachable from a
        // corrupt file. Every boundary has to be a codepoint boundary, and a
        // valid blob says nothing about where the offsets fall inside it.
        for &off in &self.offsets {
            if !text.is_char_boundary(off as usize) {
                return Err(Error::corruption(format!(
                    "string dictionary offset {off} splits a UTF-8 codepoint"
                )));
            }
        }
        Ok(())
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Decode a dictionary code. Out-of-range codes yield `""` rather than
    /// panicking: a fingerprint-colliding probe can hand us a bogus code and
    /// the caller verifies afterwards.
    #[inline(always)]
    pub fn get(&self, code: u64) -> &str {
        let i = code as usize;
        if i + 1 >= self.offsets.len() {
            return "";
        }
        let (a, b) = (self.offsets[i] as usize, self.offsets[i + 1] as usize);
        // SAFETY: validated as UTF-8 at construction, offsets monotonic.
        unsafe { std::str::from_utf8_unchecked(&self.blob[a..b]) }
    }

    /// Exact code for `s`, or `None`. Binary search over the sorted blob.
    pub fn lookup(&self, s: &str) -> Option<u64> {
        let mut lo = 0usize;
        let mut hi = self.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.get(mid as u64).cmp(s) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid as u64),
            }
        }
        None
    }

    /// First code whose string is `>= s`. Feeds range predicates: a
    /// `col > 'foo'` filter becomes `code >= lower_bound("foo")` after an
    /// equality adjustment, evaluated entirely on packed integers.
    pub fn lower_bound(&self, s: &str) -> u64 {
        let mut lo = 0usize;
        let mut hi = self.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.get(mid as u64) < s {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo as u64
    }

    pub fn blob(&self) -> &[u8] {
        &self.blob
    }
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> + '_ {
        (0..self.len()).map(move |i| self.get(i as u64))
    }

    pub fn bytes(&self) -> usize {
        self.blob.len() + self.offsets.len() * 4
    }
}

impl std::fmt::Debug for StringDict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StringDict({} entries, {} bytes)", self.len(), self.bytes())
    }
}

/// Build a dictionary plus per-row codes from raw strings.
///
/// Returns codes in the *original row order*, so the caller can hand them
/// straight to `PackedU64::pack`.
pub fn encode<S: AsRef<str>>(values: &[S]) -> (StringDict, Vec<u64>) {
    // Collect distinct strings first; the dedup map is keyed by &str borrowed
    // from `values`, so no copies until we build the blob.
    let mut seen: FastMap<&str, ()> = FastMap::default();
    for v in values {
        seen.entry(v.as_ref()).or_insert(());
    }
    let mut uniq: Vec<&str> = seen.into_keys().collect();
    uniq.sort_unstable();

    let dict = StringDict::from_sorted(&uniq);
    // `uniq` is sorted, so its index *is* the code; a map avoids a binary
    // search per row.
    let index: FastMap<&str, u64> =
        uniq.iter().enumerate().map(|(i, &s)| (s, i as u64)).collect();
    let codes: Vec<u64> = values.iter().map(|v| index[v.as_ref()]).collect();
    (dict, codes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_roundtrips_and_preserves_order() {
        let vals = ["pear", "apple", "pear", "banana", "apple", "zucchini"];
        let (dict, codes) = encode(&vals);
        assert_eq!(dict.len(), 4);
        for (i, v) in vals.iter().enumerate() {
            assert_eq!(dict.get(codes[i]), *v, "row {i}");
        }
        // order preserving: sorting by code == sorting by string
        for i in 0..vals.len() {
            for j in 0..vals.len() {
                assert_eq!(
                    codes[i].cmp(&codes[j]),
                    vals[i].cmp(vals[j]),
                    "{} vs {}",
                    vals[i],
                    vals[j]
                );
            }
        }
    }

    #[test]
    fn lookup_and_lower_bound() {
        let (dict, _) = encode(&["b", "d", "a", "c"]);
        assert_eq!(dict.lookup("a"), Some(0));
        assert_eq!(dict.lookup("d"), Some(3));
        assert_eq!(dict.lookup("zzz"), None);
        assert_eq!(dict.lower_bound("a"), 0);
        assert_eq!(dict.lower_bound("bb"), 2); // between "b" and "c"
        assert_eq!(dict.lower_bound("zzz"), 4); // past the end
    }

    #[test]
    fn empty_and_single() {
        let empty: [&str; 0] = [];
        let (d, c) = encode(&empty);
        assert!(d.is_empty());
        assert!(c.is_empty());
        assert_eq!(d.get(0), "");

        let (d, c) = encode(&[""]);
        assert_eq!(d.len(), 1);
        assert_eq!(c, vec![0]);
        assert_eq!(d.get(0), "");
    }

    #[test]
    fn out_of_range_code_is_empty_not_panic() {
        let (d, _) = encode(&["x"]);
        assert_eq!(d.get(999), "");
    }

    #[test]
    fn from_parts_rejects_corruption() {
        let (d, _) = encode(&["a", "b"]);
        // offsets that do not cover the blob
        assert!(StringDict::from_parts(d.blob().to_vec(), vec![0, 1]).is_err());
        // non-monotonic
        assert!(StringDict::from_parts(d.blob().to_vec(), vec![0, 2, 1]).is_err());
        // valid roundtrip
        assert!(StringDict::from_parts(d.blob().to_vec(), d.offsets().to_vec()).is_ok());
    }

    #[test]
    fn from_parts_rejects_offsets_that_split_a_codepoint() {
        // "é" is two bytes; an offset of 1 lands inside it. `get` would then
        // build a &str over invalid UTF-8 via from_utf8_unchecked.
        let blob = "é".as_bytes().to_vec();
        assert_eq!(blob.len(), 2);
        let e = StringDict::from_parts(blob.clone(), vec![0, 1, 2]).unwrap_err();
        assert!(e.to_string().contains("splits a UTF-8 codepoint"), "{e}");
        // the same blob with honest boundaries is fine
        assert!(StringDict::from_parts(blob, vec![0, 2]).is_ok());
    }

    #[test]
    fn every_entry_of_a_validated_dict_is_valid_utf8() {
        // Belt and braces: round-trip a multibyte dictionary through
        // from_parts and read every entry back.
        let (d, _) = encode(&["日本語", "café", "a"]);
        let back = StringDict::from_parts(d.blob().to_vec(), d.offsets().to_vec()).unwrap();
        let got: Vec<&str> = back.iter().collect();
        assert_eq!(got, vec!["a", "café", "日本語"]);
    }

    #[test]
    fn utf8_multibyte_survives() {
        let vals = ["日本語", "café", "naïve", "日本語"];
        let (dict, codes) = encode(&vals);
        for (i, v) in vals.iter().enumerate() {
            assert_eq!(dict.get(codes[i]), *v);
        }
    }
}
