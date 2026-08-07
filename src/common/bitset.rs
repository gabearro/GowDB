//! Growable bitmap used for delete masks and SQL null masks.
//!
//! Allocation is lazy: a clean part (no deletes) and a NOT NULL column both
//! store zero bytes.

#[derive(Default, Clone, PartialEq, Eq)]
pub struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    pub fn new() -> Self {
        BitSet { words: Vec::new() }
    }

    /// A bitset preallocated to hold `n` bits, all clear.
    pub fn with_capacity_bits(n: usize) -> Self {
        BitSet { words: vec![0; n.div_ceil(64)] }
    }

    /// All bits in `[0, n)` set.
    pub fn all_set(n: usize) -> Self {
        let mut s = BitSet { words: vec![u64::MAX; n.div_ceil(64)] };
        let rem = n % 64;
        if rem != 0 {
            let last = s.words.len() - 1;
            s.words[last] = (1u64 << rem) - 1;
        }
        s
    }

    /// Returns true if the bit was newly set.
    #[inline(always)]
    pub fn set(&mut self, i: usize) -> bool {
        let w = i / 64;
        if self.words.len() <= w {
            self.words.resize(w + 1, 0);
        }
        let m = 1u64 << (i % 64);
        let fresh = self.words[w] & m == 0;
        self.words[w] |= m;
        fresh
    }

    /// Returns true if the bit was previously set.
    #[inline(always)]
    pub fn clear(&mut self, i: usize) -> bool {
        let w = i / 64;
        if self.words.len() <= w {
            return false;
        }
        let m = 1u64 << (i % 64);
        let was = self.words[w] & m != 0;
        self.words[w] &= !m;
        was
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> bool {
        let w = i / 64;
        w < self.words.len() && (self.words[w] >> (i % 64)) & 1 == 1
    }

    #[inline(always)]
    pub fn set_to(&mut self, i: usize, v: bool) {
        if v {
            self.set(i);
        } else {
            self.clear(i);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Number of set bits in `[0, n)`.
    pub fn count_ones_upto(&self, n: usize) -> usize {
        let full = (n / 64).min(self.words.len());
        let mut c: usize = self.words[..full].iter().map(|w| w.count_ones() as usize).sum();
        if full < self.words.len() && n % 64 != 0 {
            c += (self.words[full] & ((1u64 << (n % 64)) - 1)).count_ones() as usize;
        }
        c
    }

    /// Iterate set-bit indices below `n`, word at a time.
    pub fn iter_ones(&self, n: usize) -> impl Iterator<Item = usize> + '_ {
        let nwords = self.words.len();
        (0..nwords).flat_map(move |wi| {
            let mut w = self.words[wi];
            std::iter::from_fn(move || {
                if w == 0 {
                    return None;
                }
                let b = w.trailing_zeros() as usize;
                w &= w - 1;
                Some(wi * 64 + b)
            })
        }).take_while(move |&i| i < n)
    }

    pub fn union_with(&mut self, other: &BitSet) {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a |= *b;
        }
    }

    pub fn intersect_with(&mut self, other: &BitSet) {
        self.words.truncate(other.words.len());
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a &= *b;
        }
    }

    pub fn negate(&mut self, n: usize) {
        self.words.resize(n.div_ceil(64), 0);
        for w in self.words.iter_mut() {
            *w = !*w;
        }
        let rem = n % 64;
        if rem != 0 {
            let last = self.words.len() - 1;
            self.words[last] &= (1u64 << rem) - 1;
        }
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn from_words(words: Vec<u64>) -> Self {
        BitSet { words }
    }

    pub fn bytes(&self) -> usize {
        self.words.len() * 8
    }
}

impl std::fmt::Debug for BitSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BitSet({} words, {} set)", self.words.len(), self.count_ones())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_clear() {
        let mut b = BitSet::new();
        assert!(!b.get(500));
        assert!(b.set(500));
        assert!(!b.set(500)); // already set
        assert!(b.get(500));
        assert!(b.clear(500));
        assert!(!b.clear(500));
        assert!(!b.get(500));
    }

    #[test]
    fn all_set_respects_partial_last_word() {
        let b = BitSet::all_set(70);
        assert_eq!(b.count_ones(), 70);
        assert!(b.get(69));
        assert!(!b.get(70));
    }

    #[test]
    fn negate_masks_tail() {
        let mut b = BitSet::with_capacity_bits(10);
        b.set(3);
        b.negate(10);
        assert_eq!(b.count_ones(), 9);
        assert!(!b.get(3));
        assert!(b.get(9));
        assert!(!b.get(10));
    }

    #[test]
    fn iter_ones_matches_get() {
        let mut b = BitSet::new();
        for i in [0usize, 1, 63, 64, 65, 200] {
            b.set(i);
        }
        let got: Vec<usize> = b.iter_ones(256).collect();
        assert_eq!(got, vec![0, 1, 63, 64, 65, 200]);
    }

    #[test]
    fn count_ones_upto_is_prefix_exact() {
        let mut b = BitSet::new();
        for i in (0..300).step_by(3) {
            b.set(i);
        }
        for n in [0usize, 1, 64, 100, 299, 300] {
            let expect = (0..n).filter(|&i| b.get(i)).count();
            assert_eq!(b.count_ones_upto(n), expect, "n={n}");
        }
    }
}
