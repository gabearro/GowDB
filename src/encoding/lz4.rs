//! LZ4 block-format compression, hand-rolled.
//!
//! ## Why a byte codec at all, in an engine built around *not* decompressing
//!
//! Everything else in `encoding/` is a random-access codec: FOR bit packing
//! and dictionary codes are read in place, one shifted load per value, and a
//! point lookup never pays to inflate a block it does not need. That property
//! is the whole thesis of the engine and this module does **not** have it --
//! an LZ4 block is opaque until you have decoded all of it.
//!
//! So it earns its place only on bytes that are *already* going to be read
//! whole: part payloads streamed off disk, string dictionary blobs, WAL
//! records. There the bottleneck is bytes-through-the-device, and LZ4 is the
//! one point on the ratio/speed curve where the codec is not the bottleneck --
//! decode is a single forward pass of `memcpy`-shaped work, typically several
//! GB/s, i.e. faster than the disk it saves you reading from. Anything
//! heavier (entropy coding) inverts that and makes the CPU the limit.
//!
//! Bit-packed lanes are the interesting input: FOR packing has already removed
//! the *value* redundancy, but it leaves plenty of *byte* redundancy -- lanes
//! that repeat, high bytes that are all zero, runs of equal words -- and LZ4
//! picks that up for nearly free on top.
//!
//! ## Format (block, not frame)
//!
//! A block is a bare sequence of sequences. No magic, no checksum, no stored
//! decompressed size: the caller already knows how many bytes it wrote, and
//! the surrounding part format already checksums. One sequence is:
//!
//! ```text
//!   token   1 byte: high nibble = literal length, low nibble = matchlen - 4
//!   [ literal-length extras ]   only if the high nibble is 15
//!   literals                    `literal length` raw bytes
//!   offset  2 bytes LE          distance back into already-decoded output
//!   [ match-length extras ]     only if the low nibble is 15
//! ```
//!
//! Extras are 255-continuation: add each byte, stop on the first byte < 255.
//! The last sequence stops after its literals -- no offset, no match. The
//! reference encoder additionally guarantees that the last match ends at least
//! 12 bytes before the end and the trailing literal run is at least 5 bytes,
//! which is what lets an optimized decoder copy in 8- and 16-byte gulps
//! without overrunning; we honour both rules when encoding. We do *not*
//! enforce them when decoding, because they are encoder obligations and a
//! shorter well-formed block (`compress` of a 3-byte input, say) is still
//! unambiguous.
//!
//! ## Decoding is a hostile-input parser
//!
//! These bytes come off disk. They may be corrupt, and in a networked
//! deployment they may be chosen by someone. The decoder therefore validates
//! every length and every offset against **both** the remaining input and the
//! remaining output before it moves a byte, uses no `unsafe`, and cannot
//! panic, hang, or allocate an attacker-chosen amount of memory:
//!
//!   * a 255-continuation run is bounded by the input length *and* bailed out
//!     of as soon as the running total exceeds the output that is left;
//!   * an offset of 0, or one larger than the bytes produced so far, is
//!     corruption -- never a read before the start of the buffer;
//!   * [`decompress`] refuses an `expect_len` that the input provably cannot
//!     produce before it allocates. The bound is exactly 255x: per sequence,
//!     output is at most `(15 + 255a) + (19 + 255b)` for `3 + a + b + ll`
//!     input bytes, and `34 - 765 < 0`, so every sequence expands by strictly
//!     less than 255. A 4-byte block claiming 16 GiB is rejected for free.
//!
//! ## Encoding is deliberately the dumb one
//!
//! One hash table of positions, one pass, greedy, no lazy matching and no
//! chain of candidates: hash the 4 bytes at the cursor, look at the single
//! most recent position with that hash, take the match if it verifies. That is
//! the classic LZ4 operating point and it is the *reason* to use LZ4 -- a
//! smarter parser buys a few percent of ratio at several times the cost, at
//! which point you should have reached for a different codec entirely.
//!
//! The two cheap refinements the reference makes are kept because they cost
//! one comparison each: extend the match backwards over literals that have not
//! been emitted yet (shrinking the literal run for free), and re-seed the
//! table just behind the cursor after a match so the next sequence can still
//! see inside it.

use crate::common::{Error, Result};

/// Shortest match worth encoding. A sequence costs 3 bytes (token + offset),
/// so a 3-byte match would never pay for itself.
const MIN_MATCH: usize = 4;
/// A block must end with at least this many literal bytes.
const LAST_LITERALS: usize = 5;
/// No match may *start* within this many bytes of the end of the block.
const MF_LIMIT: usize = 12;
/// Below this, a conforming block cannot contain a match at all, so the best
/// we could do is literals -- which is always at least one byte bigger.
const MIN_INPUT: usize = MF_LIMIT + 1;
/// Offsets are 16-bit, so this is how far back a match may point.
const MAX_DISTANCE: usize = 65_535;
/// Nibble value that means "read continuation bytes".
const EXTRA: usize = 15;

/// Largest block we will compress. Matches the reference implementation, and
/// keeps positions inside the `u32` hash table with room for the +1 bias.
pub const MAX_INPUT_SIZE: usize = 0x7E00_0000;

/// Knuth's multiplicative constant, the standard LZ4 hash.
const HASH_MUL: u32 = 2_654_435_761;
/// 1<<16 entries (256 KiB) is the reference's default and the right size for
/// part-sized blocks; smaller inputs get a smaller table so that compressing a
/// 2 KiB column does not begin by zeroing a quarter of a megabyte.
const HASH_LOG_MAX: u32 = 16;
const HASH_LOG_MIN: u32 = 10;

/// How far behind the cursor to re-seed the table after emitting a match.
///
/// 2 is the reference's choice and the classic speed point: exactly one extra
/// insert per match. Raising it finds back-references whose period is shorter
/// than the match that hid them, which matters more here than it does for
/// general-purpose data -- a packed-lane column is *all* short periods.
/// Measured on this codec, A/B interleaved, compressed size and compressor
/// throughput at depth 6 against depth 2:
///
/// ```text
///   1 MiB of row-shaped blobs    -12.4% size    -1.8% speed
///   6-bit packed lane column     -15.1% size    -0.5% speed
///   mixed runs + noise           + 2.2% size    -7.1% speed
///   incompressible noise           0.0% size    -1.0% speed
/// ```
///
/// So it is a large win on a *synthetic* 6-bit lane column -- but raising this
/// to 6 and re-measuring on parts the engine actually writes changed the part
/// file by zero bytes (9.05 MiB either way, same 3909 of 5862 arrays left
/// uncompressed). The win does not reproduce on real lane data, whose repeat
/// period is longer than this lookback reaches. Left at the classic 2, which
/// is the documented LZ4 operating point.
const RESEED_DEPTH: usize = 2;

// The reference encoder widens its search stride after consecutive misses, to
// make incompressible input cheap to give up on. Tried here (`step = 1 +
// (misses >> 6)`, the reference constant) and measured interleaved on a
// 2M-row part write: **40% slower**, 25-29ms to 36-40ms, for 0.7% less on
// disk. It is slower for a specific reason worth recording, because the
// heuristic is otherwise obviously right: this encoder's give-up path is
// `dst.len() >= n`, which only triggers once enough *sequences* have been
// emitted. Skipping positions suppresses exactly the marginal matches that
// drive that counter, so the encoder stops bailing early and scans the whole
// array instead. The existing early exit is already the better give-up test;
// adding the stride defeats it. Not reinstated.

/// Upper bound on the compressed size of `n` bytes: the all-literal encoding,
/// which is one token per 255 literals plus slack for the leading token and
/// the length extras.
pub fn max_compressed_len(n: usize) -> usize {
    n.saturating_add(n / 255).saturating_add(16)
}

// ---------------------------------------------------------------------------
// compression
// ---------------------------------------------------------------------------

#[inline]
fn read_u32(src: &[u8], at: usize) -> u32 {
    // Callers keep `at + 4 <= src.len()`; this is our own data, not the
    // attacker's, so a bad index here is a bug and should be loud in debug.
    debug_assert!(at + 4 <= src.len());
    u32::from_le_bytes([src[at], src[at + 1], src[at + 2], src[at + 3]])
}

#[inline]
fn hash4(seq: u32, shift: u32) -> usize {
    (seq.wrapping_mul(HASH_MUL) >> shift) as usize
}

/// How many entries the position table gets for an `n`-byte input.
fn hash_log(n: usize) -> u32 {
    let bits = usize::BITS - n.leading_zeros();
    bits.clamp(HASH_LOG_MIN, HASH_LOG_MAX)
}

/// Number of continuation bytes needed to encode `len` in a nibble field.
#[inline]
fn extra_bytes(len: usize) -> usize {
    if len < EXTRA {
        0
    } else {
        (len - EXTRA) / 255 + 1
    }
}

/// Append a 255-continuation length. `rem` is the amount past the nibble.
#[inline]
fn push_extra(dst: &mut Vec<u8>, mut rem: usize) {
    while rem >= 255 {
        dst.push(255);
        rem -= 255;
    }
    dst.push(rem as u8);
}

/// Emit one full sequence: literals, then a back-reference.
fn emit_sequence(dst: &mut Vec<u8>, literals: &[u8], offset: u16, mlen: usize) {
    let ll = literals.len();
    let ml_code = mlen - MIN_MATCH;
    // Both nibbles are known before either extras field is written, which is
    // why the token can go out first.
    let token = ((ll.min(EXTRA) as u8) << 4) | ml_code.min(EXTRA) as u8;
    dst.push(token);
    if ll >= EXTRA {
        push_extra(dst, ll - EXTRA);
    }
    dst.extend_from_slice(literals);
    dst.extend_from_slice(&offset.to_le_bytes());
    if ml_code >= EXTRA {
        push_extra(dst, ml_code - EXTRA);
    }
}

/// Compress `src`. Returns `None` when the result would not be smaller --
/// including for inputs too short to hold a match at all, and for inputs so
/// large the block format cannot address them.
///
/// The caller is expected to store the original bytes on `None`, so we bail
/// out the moment the output has grown past the input rather than finishing a
/// block nobody will keep.
pub fn compress(src: &[u8]) -> Option<Vec<u8>> {
    let n = src.len();
    if n < MIN_INPUT || n > MAX_INPUT_SIZE {
        return None;
    }

    let log = hash_log(n);
    let shift = 32 - log;
    // 0 means "empty"; a position is stored biased by one.
    let mut table = vec![0u32; 1usize << log];
    let mut dst: Vec<u8> = Vec::with_capacity(n);

    // A match may not start at or after `search_limit`, and may not end after
    // `match_limit`. Both are safe subtractions because `n >= MIN_INPUT`.
    let search_limit = n - MF_LIMIT;
    let match_limit = n - LAST_LITERALS;

    let mut anchor = 0usize; // start of the literals not yet emitted
    let mut ip = 0usize;

    while ip < search_limit {
        let seq = read_u32(src, ip);
        let h = hash4(seq, shift);
        let entry = table[h];
        table[h] = (ip + 1) as u32;

        if entry == 0 {
            ip += 1;
            continue;
        }
        let cand = entry as usize - 1;
        // Reject a stale entry (out of window) or a hash collision. The 4-byte
        // verify is what lets the table be a single slot with no tags.
        if ip - cand > MAX_DISTANCE || read_u32(src, cand) != seq {
            ip += 1;
            continue;
        }

        // Forward extension. `ip + fwd < match_limit` keeps the trailing five
        // literals the format requires; `ip + MIN_MATCH` is always inside it
        // because `ip < n - 12`.
        let mut fwd = MIN_MATCH;
        while ip + fwd < match_limit && src[cand + fwd] == src[ip + fwd] {
            fwd += 1;
        }

        // Backward extension over literals we have not committed to yet. Each
        // byte moved here is a byte deleted from the literal run and added to
        // a match length that is already being encoded -- pure profit.
        let mut back = 0usize;
        while ip - back > anchor && cand > back && src[ip - back - 1] == src[cand - back - 1] {
            back += 1;
        }

        let start = ip - back;
        emit_sequence(&mut dst, &src[anchor..start], (ip - cand) as u16, fwd + back);
        ip += fwd;
        anchor = ip;

        // Already bigger than storing the bytes raw: stop, the caller will
        // keep the original.
        if dst.len() >= n {
            return None;
        }

        // Seed positions just behind the cursor. They sit inside the match we
        // just emitted, which is exactly the point: repeated structure whose
        // period is shorter than the match should still be findable.
        let seed_depth = (fwd + back).min(RESEED_DEPTH);
        for d in 2..=seed_depth {
            if ip >= d {
                let p = ip - d;
                table[hash4(read_u32(src, p), shift)] = (p + 1) as u32;
            }
        }
    }

    // Trailing literals, always at least LAST_LITERALS bytes by construction.
    let lit = n - anchor;
    if dst.len() + 1 + extra_bytes(lit) + lit >= n {
        return None; // would not be smaller; skip the final copy entirely
    }
    let token = (lit.min(EXTRA) as u8) << 4;
    dst.push(token);
    if lit >= EXTRA {
        push_extra(&mut dst, lit - EXTRA);
    }
    dst.extend_from_slice(&src[anchor..]);

    debug_assert!(dst.len() < n);
    Some(dst)
}

// ---------------------------------------------------------------------------
// decompression -- everything below runs on untrusted bytes
// ---------------------------------------------------------------------------

#[cold]
fn corrupt<T>(what: &str) -> Result<T> {
    Err(Error::corruption(format!("lz4: {what}")))
}

/// The most output any `src_len`-byte block can legally produce. See the
/// module docs for the derivation; the slack is pure paranoia.
#[inline]
pub fn expansion_ceiling(src_len: usize) -> usize {
    src_len.saturating_mul(255).saturating_add(16)
}

/// Read a 255-continuation length.
///
/// `budget` is the number of output bytes still available: no length can
/// exceed it, so the accumulator is checked against it on every byte. That
/// bounds both the value and -- together with the input bound -- the number of
/// iterations, so a run of `0xFF` bytes can neither overflow nor spin.
#[inline]
fn read_extra_len(src: &[u8], ip: &mut usize, budget: usize) -> Result<usize> {
    let mut total = 0usize;
    loop {
        let b = match src.get(*ip) {
            Some(&b) => b,
            None => return corrupt("truncated length"),
        };
        *ip += 1;
        total = match total.checked_add(b as usize) {
            Some(t) => t,
            None => return corrupt("length overflow"),
        };
        if total > budget {
            return corrupt("length exceeds remaining output");
        }
        if b != 255 {
            return Ok(total);
        }
    }
}

/// Decompress `src` into a caller-provided buffer, which must be exactly the
/// size of the original data. Allocates nothing.
pub fn decompress_into(src: &[u8], out: &mut [u8]) -> Result<()> {
    // An empty block is only a valid encoding of empty output. (`compress`
    // never produces one -- it is here so a caller that stored "zero bytes,
    // zero bytes" round-trips instead of erroring.)
    if src.is_empty() {
        return if out.is_empty() { Ok(()) } else { corrupt("empty input, non-empty output") };
    }

    let mut ip = 0usize; // read cursor in src
    let mut op = 0usize; // write cursor in out

    loop {
        // Reaching here with no input left means the previous sequence ended
        // in a match. A well-formed block always ends after a literal run.
        let token = match src.get(ip) {
            Some(&t) => t,
            None => return corrupt("stream ends after a match"),
        };
        ip += 1;

        // --- literals -----------------------------------------------------
        let mut ll = (token >> 4) as usize;
        if ll == EXTRA {
            ll += read_extra_len(src, &mut ip, out.len() - op)?;
        }
        if ll != 0 {
            let src_end = match ip.checked_add(ll) {
                Some(e) => e,
                None => return corrupt("literal length overflow"),
            };
            let lits = match src.get(ip..src_end) {
                Some(s) => s,
                None => return corrupt("literal run past end of input"),
            };
            let out_end = match op.checked_add(ll) {
                Some(e) => e,
                None => return corrupt("literal length overflow"),
            };
            let dst = match out.get_mut(op..out_end) {
                Some(d) => d,
                None => return corrupt("literal run past end of output"),
            };
            dst.copy_from_slice(lits);
            ip = src_end;
            op = out_end;
        }

        // The final sequence is literals only: no offset follows.
        if ip == src.len() {
            break;
        }

        // --- match --------------------------------------------------------
        let off_end = match ip.checked_add(2) {
            Some(e) => e,
            None => return corrupt("offset overflow"),
        };
        let off_bytes = match src.get(ip..off_end) {
            Some(b) => b,
            None => return corrupt("truncated match offset"),
        };
        let offset = u16::from_le_bytes([off_bytes[0], off_bytes[1]]) as usize;
        ip = off_end;
        if offset == 0 {
            return corrupt("zero match offset");
        }
        if offset > op {
            // Would read before the start of the output buffer.
            return corrupt("match offset points before start of output");
        }

        let mut ml = (token & 0x0F) as usize;
        if ml == EXTRA {
            ml += read_extra_len(src, &mut ip, out.len() - op)?;
        }
        ml += MIN_MATCH;

        let out_end = match op.checked_add(ml) {
            Some(e) => e,
            None => return corrupt("match length overflow"),
        };
        if out_end > out.len() {
            return corrupt("match runs past end of output");
        }
        let from = op - offset;

        if offset >= ml {
            // Source and destination are disjoint: one memmove.
            out.copy_within(from..from + ml, op);
        } else {
            // Overlapping. This is NOT a mistake to be optimized away: the
            // format uses it as run-length encoding (offset 1 replicates one
            // byte, offset 2 a pair, ...) and the bytes being read are
            // produced by this very loop. It must stay a forward, one-byte-at-
            // a-time copy or the output is wrong.
            for k in 0..ml {
                out[op + k] = out[from + k];
            }
        }
        op = out_end;
    }

    if op != out.len() {
        return corrupt("decoded length does not match expected length");
    }
    Ok(())
}

/// Decompress exactly `expect_len` bytes from `src`.
pub fn decompress(src: &[u8], expect_len: usize) -> Result<Vec<u8>> {
    // Validate the claim *before* honouring it with an allocation: a handful
    // of corrupt bytes must not be able to ask for gigabytes.
    if expect_len > expansion_ceiling(src.len()) {
        return corrupt("expected length exceeds what the input can produce");
    }
    let mut out = vec![0u8; expect_len];
    decompress_into(src, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::splitmix64;

    // -- helpers ----------------------------------------------------------

    fn rand_bytes(n: usize, seed: u64) -> Vec<u8> {
        (0..n).map(|i| (splitmix64(seed ^ i as u64) >> 13) as u8).collect()
    }

    /// An all-literal block, i.e. what a caller would have to write by hand
    /// for input `compress` refuses. Lets the round-trip helper exercise the
    /// decoder on inputs too small to compress.
    fn literal_block(data: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        let ll = data.len();
        v.push((ll.min(EXTRA) as u8) << 4);
        if ll >= EXTRA {
            push_extra(&mut v, ll - EXTRA);
        }
        v.extend_from_slice(data);
        v
    }

    /// Round-trip through the real codec when it accepts the input, through a
    /// hand-built literal block when it does not. Returns the encoded form.
    fn roundtrip(data: &[u8]) -> Vec<u8> {
        let enc = match compress(data) {
            Some(c) => {
                assert!(c.len() < data.len(), "compress kept a non-smaller block");
                assert!(c.len() <= max_compressed_len(data.len()));
                c
            }
            None => literal_block(data),
        };
        let dec = decompress(&enc, data.len()).expect("decompress failed");
        assert_eq!(dec, data, "round-trip mismatch (len {})", data.len());

        // decompress_into must agree with the allocating form.
        let mut buf = vec![0u8; data.len()];
        decompress_into(&enc, &mut buf).expect("decompress_into failed");
        assert_eq!(buf, data);
        enc
    }

    /// Hand-assembles blocks, including invalid ones, so decoder edge cases
    /// can be reached without going through our encoder.
    struct Builder {
        buf: Vec<u8>,
    }
    impl Builder {
        fn new() -> Self {
            Builder { buf: Vec::new() }
        }
        fn seq(mut self, lits: &[u8], offset: u16, mlen: usize) -> Self {
            emit_sequence(&mut self.buf, lits, offset, mlen);
            self
        }
        fn last(mut self, lits: &[u8]) -> Vec<u8> {
            let ll = lits.len();
            self.buf.push((ll.min(EXTRA) as u8) << 4);
            if ll >= EXTRA {
                push_extra(&mut self.buf, ll - EXTRA);
            }
            self.buf.extend_from_slice(lits);
            self.buf
        }
    }

    // -- round-trips ------------------------------------------------------

    #[test]
    fn roundtrip_empty() {
        assert_eq!(compress(&[]), None);
        roundtrip(&[]);
    }

    #[test]
    fn roundtrip_single_byte() {
        assert_eq!(compress(&[0x5a]), None);
        roundtrip(&[0x5a]);
    }

    #[test]
    fn roundtrip_every_short_length() {
        // Repetitive so anything long enough to compress, does.
        let base: Vec<u8> = (0..200u32).map(|i| b"abcd"[(i % 4) as usize]).collect();
        for len in 0..=200usize {
            roundtrip(&base[..len]);
        }
    }

    #[test]
    fn roundtrip_all_identical_run() {
        for len in [13usize, 64, 1000, 70_000] {
            let data = vec![0xABu8; len];
            let enc = roundtrip(&data);
            // A pure run is the best case for the format: ~1 sequence.
            assert!(enc.len() < len / 20 + 16, "run of {len} took {} bytes", enc.len());
        }
    }

    #[test]
    fn roundtrip_all_zero_run() {
        roundtrip(&vec![0u8; 1 << 16]);
    }

    #[test]
    fn roundtrip_random_bytes() {
        for seed in 0..16u64 {
            for len in [13usize, 100, 4096, 33_333] {
                roundtrip(&rand_bytes(len, seed));
            }
        }
    }

    #[test]
    fn roundtrip_repetitive_text() {
        let mut s = Vec::new();
        for i in 0..500 {
            s.extend_from_slice(b"the quick brown fox jumps over the lazy dog #");
            s.extend_from_slice(format!("{}\n", i % 17).as_bytes());
        }
        let enc = roundtrip(&s);
        assert!(enc.len() * 8 < s.len(), "text only shrank to {}/{}", enc.len(), s.len());
    }

    #[test]
    fn roundtrip_offset_one() {
        // A single repeated byte after a distinct prefix: the classic offset-1
        // run-length case, which forces the overlapping copy path.
        let mut d = b"prefix-".to_vec();
        d.extend(std::iter::repeat(b'q').take(5000));
        d.extend_from_slice(b"-suffix");
        roundtrip(&d);
    }

    #[test]
    fn roundtrip_offset_two_and_three() {
        for period in [2usize, 3, 4, 5, 7] {
            let pat: Vec<u8> = (0..period).map(|i| b'a' + i as u8).collect();
            let mut d = Vec::new();
            while d.len() < 4000 {
                d.extend_from_slice(&pat);
            }
            roundtrip(&d);
        }
    }

    #[test]
    fn emits_offset_65535_and_roundtrips() {
        // Layout: an 8-byte marker at 0, a zero run, the same marker at
        // exactly 65535, then a tail. The zero run all hashes to one slot, so
        // the marker's table entry survives and the match is found at the
        // maximum legal distance.
        let marker: [u8; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let mut d = Vec::new();
        d.extend_from_slice(&marker);
        d.resize(MAX_DISTANCE, 0);
        d.extend_from_slice(&marker);
        d.extend_from_slice(b"trailing-literals-xyz");
        assert_eq!(d[MAX_DISTANCE..MAX_DISTANCE + 8], marker);

        let enc = compress(&d).expect("should compress");
        // 0xFFFF little-endian must appear as the offset of the second marker.
        assert!(
            enc.windows(2).any(|w| w == [0xFF, 0xFF]),
            "expected a max-distance offset in {enc:?}"
        );
        assert_eq!(decompress(&enc, d.len()).unwrap(), d);
    }

    #[test]
    fn beyond_max_distance_still_roundtrips() {
        // Same shape, but one byte too far: the encoder must reject the
        // candidate rather than emit an offset it cannot represent.
        let marker: [u8; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let mut d = Vec::new();
        d.extend_from_slice(&marker);
        d.resize(MAX_DISTANCE + 1, 0);
        d.extend_from_slice(&marker);
        d.extend_from_slice(b"trailing-literals-xyz");
        roundtrip(&d);
    }

    #[test]
    fn compressor_never_emits_a_zero_offset() {
        // Walk the encoder's own output and check every offset field. A zero
        // offset is unrepresentable garbage that would trap the decoder.
        let data = {
            let mut d = Vec::new();
            for i in 0..2000u32 {
                d.extend_from_slice(&(i % 23).to_le_bytes());
            }
            d
        };
        let enc = compress(&data).unwrap();
        let mut ip = 0usize;
        while ip < enc.len() {
            let token = enc[ip];
            ip += 1;
            let mut ll = (token >> 4) as usize;
            if ll == EXTRA {
                ll += read_extra_len(&enc, &mut ip, usize::MAX).unwrap();
            }
            ip += ll;
            if ip == enc.len() {
                break;
            }
            let off = u16::from_le_bytes([enc[ip], enc[ip + 1]]);
            assert_ne!(off, 0, "zero offset emitted");
            ip += 2;
            if token & 0x0F == EXTRA as u8 {
                read_extra_len(&enc, &mut ip, usize::MAX).unwrap();
            }
        }
    }

    // -- realistic payloads ------------------------------------------------

    /// Pack `vals` at `width` bits per lane into `u64` words, the way the FOR
    /// codec lays out a column, and hand back the raw little-endian bytes.
    /// Rebuilt locally on purpose: this test must not couple to a module that
    /// is being edited elsewhere.
    fn bitpacked_lanes(vals: &[u64], width: u32) -> Vec<u8> {
        let per = 64 / width as usize;
        let mut words = vec![0u64; vals.len().div_ceil(per)];
        for (i, &v) in vals.iter().enumerate() {
            words[i / per] |= (v & ((1u64 << width) - 1)) << ((i % per) * width as usize);
        }
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    #[test]
    fn roundtrip_bitpacked_lane_array() {
        // A clustered low-cardinality column, which is what an ORDER BY key
        // actually looks like: FOR packing has already taken the values down
        // to 6 bits, but runs of equal values still make whole packed *words*
        // repeat, and that byte-level residue is exactly what LZ4 is here for.
        let vals: Vec<u64> = (0..8192u64).map(|i| (i / 100) % 40).collect();
        let bytes = bitpacked_lanes(&vals, 6);
        let enc = roundtrip(&bytes);
        assert!(enc.len() * 4 < bytes.len(), "packed lanes {} -> {}", bytes.len(), enc.len());

        // Wider lanes with a slowly-moving value: neighbouring words differ in
        // their low bits only, so matches are short but plentiful.
        let vals: Vec<u64> = (0..4096u64).map(|i| 1_000_000 + i / 8).collect();
        let bytes = bitpacked_lanes(&vals, 21);
        let enc = roundtrip(&bytes);
        assert!(enc.len() < bytes.len(), "monotone lanes did not shrink");

        // Genuinely high-entropy lanes: `compress` must decline, and the
        // round-trip must still hold through the literal path.
        let vals: Vec<u64> = (0..4096u64).map(splitmix64).collect();
        let bytes = bitpacked_lanes(&vals, 32);
        assert_eq!(compress(&bytes), None);
        roundtrip(&bytes);
    }

    #[test]
    fn roundtrip_string_dictionary_blob() {
        // The shape a StringDict serializes to: sorted unique strings
        // concatenated, then the u32 offset array. Built locally so this test
        // does not depend on another module's layout.
        let mut strings: Vec<String> = Vec::new();
        for i in 0..3000u32 {
            strings.push(format!("https://example.com/session/{}/page/{}", i % 400, i % 37));
        }
        strings.sort();
        strings.dedup();

        let mut blob = Vec::new();
        let mut offsets: Vec<u32> = vec![0];
        for s in &strings {
            blob.extend_from_slice(s.as_bytes());
            offsets.push(blob.len() as u32);
        }
        let mut payload = blob.clone();
        payload.extend(offsets.iter().flat_map(|o| o.to_le_bytes()));

        let enc = roundtrip(&payload);
        // Shared URL prefixes are the whole reason to compress a dictionary.
        assert!(enc.len() * 3 < payload.len(), "dict blob {} -> {}", payload.len(), enc.len());
    }

    #[test]
    fn roundtrip_one_mib_of_structure() {
        let mut d = Vec::with_capacity(1 << 20);
        let mut i = 0u64;
        while d.len() < (1 << 20) {
            d.extend_from_slice(&splitmix64(i / 64).to_le_bytes());
            d.extend_from_slice(b"|row|");
            i += 1;
        }
        roundtrip(&d);
    }

    // -- refusal to grow ---------------------------------------------------

    #[test]
    fn compress_returns_none_on_random() {
        for seed in 0..32u64 {
            let d = rand_bytes(20_000, 0xDEAD_0000 + seed);
            assert_eq!(compress(&d), None, "seed {seed} pretended to compress noise");
        }
    }

    #[test]
    fn compress_returns_none_on_tiny_inputs() {
        for len in 0..MIN_INPUT {
            assert_eq!(compress(&vec![7u8; len]), None, "len {len}");
        }
        // 13 is the first length that *may* hold a match, but the only legal
        // start position is 0, where the table is still empty -- so the first
        // length that actually compresses is 14.
        assert_eq!(compress(b"aaaaaaaaaaaaa").map(|c| c.len()), None);
        assert!(compress(b"aaaaaaaaaaaaaa").is_some());
    }

    #[test]
    fn max_compressed_len_bounds_every_output() {
        for len in [13usize, 100, 1000, 50_000] {
            for seed in 0..4u64 {
                let d = rand_bytes(len, seed);
                if let Some(c) = compress(&d) {
                    assert!(c.len() <= max_compressed_len(len));
                }
                assert!(literal_block(&d).len() <= max_compressed_len(len));
            }
        }
    }

    // -- length extension --------------------------------------------------

    #[test]
    fn long_literal_run_uses_extension_bytes() {
        // >15 literals then a match: forces the literal-length nibble to 15
        // plus one continuation byte.
        let mut d = rand_bytes(200, 99);
        let tail = d[0..64].to_vec();
        d.extend_from_slice(&tail);
        d.extend_from_slice(b"trailing!!");
        roundtrip(&d);
    }

    #[test]
    fn very_long_literal_run_spans_many_255s() {
        // 1500 incompressible bytes ahead of a big match: the literal length
        // needs six continuation bytes.
        let mut d = rand_bytes(1500, 7);
        d.extend(std::iter::repeat(b'=').take(4000));
        d.extend_from_slice(b"end-of-block");
        roundtrip(&d);
    }

    #[test]
    fn long_match_uses_extension_bytes() {
        let mut d = b"header-bytes-here".to_vec();
        d.extend(std::iter::repeat(b'z').take(30)); // match code 15+ but < 255
        d.extend_from_slice(b"tail-literals");
        roundtrip(&d);
    }

    #[test]
    fn very_long_match_spans_many_255s() {
        // A 100 KiB run is one match whose length needs ~390 continuation
        // bytes; that path must be exact in both directions.
        let mut d = b"lead-in-literals".to_vec();
        d.extend(std::iter::repeat(b'#').take(100_000));
        d.extend_from_slice(b"tail-literals");
        let enc = roundtrip(&d);
        assert!(enc.len() < 500);
    }

    #[test]
    fn hand_built_overlapping_copy_replicates_forward() {
        // offset 2, match length 10 over the literals "ab" must produce
        // "ababababab" -- a memmove would produce garbage here.
        let enc = Builder::new().seq(b"ab", 2, 10).last(b"!!!!!");
        let out = decompress(&enc, 2 + 10 + 5).unwrap();
        assert_eq!(&out, b"abababababab!!!!!");
    }

    #[test]
    fn hand_built_offset_one_is_run_length() {
        let enc = Builder::new().seq(b"Q", 1, 300).last(b"tail!");
        let out = decompress(&enc, 1 + 300 + 5).unwrap();
        assert_eq!(out.len(), 306);
        assert!(out[..301].iter().all(|&b| b == b'Q'));
        assert_eq!(&out[301..], b"tail!");
    }

    // -- hostile input -----------------------------------------------------

    #[test]
    fn truncation_sweep_never_panics() {
        let mut d = b"a-header-that-repeats ".to_vec();
        for i in 0..400u64 {
            d.extend_from_slice(b"a-header-that-repeats ");
            d.extend_from_slice(&splitmix64(i).to_le_bytes());
        }
        d.extend(std::iter::repeat(b'~').take(900));
        let enc = compress(&d).unwrap();
        let full = d.len();

        for cut in 0..enc.len() {
            let prefix = &enc[..cut];
            // Every strict prefix must be rejected, not decoded and not
            // panicked on. (A prefix could in principle be a valid encoding of
            // *fewer* bytes, which is why the expected length is pinned.)
            assert!(
                decompress(prefix, full).is_err(),
                "prefix of {cut} bytes decoded as if complete"
            );
            // ... and with a mismatched expectation too.
            let _ = decompress(prefix, full / 2);
            let _ = decompress(prefix, 0);
            let mut buf = vec![0u8; full];
            let _ = decompress_into(prefix, &mut buf);
        }
        assert_eq!(decompress(&enc, full).unwrap(), d);
    }

    #[test]
    fn corruption_sweep_never_panics() {
        let mut d = Vec::new();
        for i in 0..800u64 {
            d.extend_from_slice(b"row:");
            d.extend_from_slice(&(i % 50).to_le_bytes());
            d.extend_from_slice(b"|col:value|");
        }
        let enc = compress(&d).unwrap();
        let n = d.len();

        // Flip every bit position of every byte in a decent sample, plus a few
        // whole-byte substitutions that are likely to be adversarial (0x00,
        // 0xFF, 0xF0 tokens).
        for pos in 0..enc.len() {
            for bit in 0..8 {
                let mut c = enc.clone();
                c[pos] ^= 1 << bit;
                match decompress(&c, n) {
                    Ok(v) => assert_eq!(v.len(), n),
                    Err(e) => {
                        assert_eq!(e.code(), "CHECKSUM_MISMATCH");
                    }
                }
            }
            for byte in [0x00u8, 0x0F, 0xF0, 0xFF] {
                let mut c = enc.clone();
                c[pos] = byte;
                let _ = decompress(&c, n);
                let _ = decompress(&c, n * 2);
                let _ = decompress(&c, 1);
            }
        }
    }

    #[test]
    fn random_bytes_as_compressed_input_never_panic() {
        for seed in 0..300u64 {
            let len = (splitmix64(seed) % 200) as usize;
            let junk = rand_bytes(len, seed ^ 0xABCD);
            for expect in [0usize, 1, 7, 64, 1000, 100_000] {
                let _ = decompress(&junk, expect);
                let mut buf = vec![0u8; expect.min(4096)];
                let _ = decompress_into(&junk, &mut buf);
            }
        }
    }

    #[test]
    fn offset_before_start_of_output_rejected() {
        // First sequence, 4 literals, then a match 9 bytes back: there are
        // only 4 bytes of output, so the source is before the buffer.
        let enc = Builder::new().seq(b"abcd", 9, 6).last(b"xxxxx");
        let err = decompress(&enc, 15).unwrap_err();
        assert_eq!(err.code(), "CHECKSUM_MISMATCH");
        assert!(format!("{err}").contains("before start of output"), "{err}");
    }

    #[test]
    fn offset_one_past_the_output_rejected() {
        // Exactly one byte too far -- the off-by-one that a naive bounds check
        // gets wrong.
        let ok = Builder::new().seq(b"abcd", 4, 6).last(b"xxxxx");
        assert!(decompress(&ok, 15).is_ok());
        let bad = Builder::new().seq(b"abcd", 5, 6).last(b"xxxxx");
        assert!(decompress(&bad, 15).is_err());
    }

    #[test]
    fn zero_offset_rejected() {
        let enc = Builder::new().seq(b"abcd", 0, 6).last(b"xxxxx");
        let err = decompress(&enc, 15).unwrap_err();
        assert!(format!("{err}").contains("zero match offset"), "{err}");
    }

    #[test]
    fn match_running_past_end_of_output_rejected() {
        let enc = Builder::new().seq(b"abcd", 2, 1000).last(b"xxxxx");
        assert!(decompress(&enc, 15).is_err());
        // Even with a buffer big enough for the match, the total is wrong.
        let mut buf = vec![0u8; 4 + 1000];
        assert!(decompress_into(&enc, &mut buf).is_err());
    }

    #[test]
    fn literal_run_past_end_of_input_rejected() {
        // Token claims 12 literals, only 3 bytes follow.
        let enc = vec![0xC0u8, b'a', b'b', b'c'];
        let err = decompress(&enc, 12).unwrap_err();
        assert!(format!("{err}").contains("past end of input"), "{err}");
    }

    #[test]
    fn literal_run_past_end_of_output_rejected() {
        let enc = vec![0xA0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert!(decompress(&enc, 4).is_err());
    }

    #[test]
    fn truncated_length_extension_rejected() {
        // 0xF0: literal length 15 + extras, but the extras never terminate.
        for k in 0..40usize {
            let mut enc = vec![0xF0u8];
            enc.extend(std::iter::repeat(0xFFu8).take(k));
            assert!(decompress(&enc, 100_000).is_err(), "k = {k}");
        }
    }

    #[test]
    fn unterminated_255_run_terminates_quickly() {
        // 200k continuation bytes claiming a colossal length: must be rejected
        // by the budget check, not chased to the end and not hung on.
        let mut enc = vec![0xF0u8];
        enc.extend(std::iter::repeat(0xFFu8).take(200_000));
        assert!(decompress(&enc, 64).is_err());
        let mut buf = vec![0u8; 64];
        assert!(decompress_into(&enc, &mut buf).is_err());
    }

    #[test]
    fn stream_ending_after_a_match_rejected() {
        // A sequence with no trailing literal run at all.
        let mut enc = Vec::new();
        emit_sequence(&mut enc, b"abcd", 2, 6);
        assert!(decompress(&enc, 10).is_err());
    }

    #[test]
    fn wrong_expected_length_rejected_both_ways() {
        let d = b"repeat repeat repeat repeat repeat repeat repeat".to_vec();
        let enc = compress(&d).unwrap();
        assert!(decompress(&enc, d.len() - 1).is_err());
        assert!(decompress(&enc, d.len() + 1).is_err());
        assert_eq!(decompress(&enc, d.len()).unwrap(), d);
    }

    #[test]
    fn absurd_expected_length_rejected_before_allocating() {
        // Four bytes cannot produce a gigabyte; this must fail fast rather
        // than try to zero 1 GiB.
        let enc = vec![0x40u8, b'a', b'b', b'c', b'd'];
        let err = decompress(&enc, 1 << 30).unwrap_err();
        assert!(format!("{err}").contains("exceeds what the input can produce"), "{err}");
        // The legitimate ceiling is still accepted (and then fails honestly).
        assert!(decompress(&enc, expansion_ceiling(enc.len())).is_err());
    }

    #[test]
    fn expansion_ceiling_is_never_below_a_real_block() {
        // The bound must not reject anything a valid encoder can produce.
        for len in [14usize, 500, 100_000] {
            let d = vec![0x5Au8; len];
            let enc = compress(&d).unwrap();
            assert!(expansion_ceiling(enc.len()) >= len, "ceiling too tight at {len}");
        }
    }

    #[test]
    fn empty_input_only_decodes_to_empty() {
        assert_eq!(decompress(&[], 0).unwrap(), Vec::<u8>::new());
        assert!(decompress(&[], 1).is_err());
        let mut buf = [0u8; 4];
        assert!(decompress_into(&[], &mut buf).is_err());
        assert!(decompress_into(&[], &mut []).is_ok());
    }

    #[test]
    fn decompress_into_rejects_a_mis_sized_buffer() {
        let d = vec![b'k'; 4000];
        let enc = compress(&d).unwrap();
        let mut short = vec![0u8; 3999];
        assert!(decompress_into(&enc, &mut short).is_err());
        let mut long = vec![0u8; 4001];
        assert!(decompress_into(&enc, &mut long).is_err());
        let mut exact = vec![0u8; 4000];
        assert!(decompress_into(&enc, &mut exact).is_ok());
        assert_eq!(exact, d);
    }

    #[test]
    fn structured_fuzz_roundtrip() {
        // Randomly mix runs, repeats of earlier text and noise, so matches
        // land at every distance and both length-extension paths fire.
        for seed in 0..120u64 {
            let mut rng = seed.wrapping_mul(0x9E37_79B9);
            let mut next = || {
                rng = splitmix64(rng);
                rng
            };
            let mut d: Vec<u8> = Vec::new();
            let target = (next() % 40_000) as usize;
            while d.len() < target {
                match next() % 4 {
                    0 => {
                        let len = (next() % 300) as usize;
                        let b = next() as u8;
                        d.extend(std::iter::repeat(b).take(len));
                    }
                    1 => {
                        let len = (next() % 200) as usize;
                        for i in 0..len {
                            d.push((next() >> (i % 8)) as u8);
                        }
                    }
                    2 if !d.is_empty() => {
                        // Copy an earlier slice: a real back-reference.
                        let len = ((next() % 500) as usize).min(d.len());
                        let start = (next() as usize) % (d.len() - len + 1);
                        let piece = d[start..start + len].to_vec();
                        d.extend_from_slice(&piece);
                    }
                    _ => d.extend_from_slice(b"the same short phrase, again and again"),
                }
            }
            roundtrip(&d);
        }
    }

    // -- reference compatibility -------------------------------------------
    //
    // Blocks below were produced by the reference LZ4 implementation
    // (python-lz4's `lz4.block.compress(..., store_size=False)`, which wraps
    // upstream liblz4) and pasted in as literals. They pin our decoder to the
    // real format rather than to our own encoder's habits: note that
    // REF_LANES is the reference's *expanded* all-literal form, and REF_RUNS
    // exercises its long-match continuation bytes.

    // 270 -> 57 bytes
    const REF_TEXT: &[u8] = &[
        0xf0, 0x10, 0x74, 0x68, 0x65, 0x20, 0x71, 0x75, 0x69, 0x63, 0x6b, 0x20, 0x62, 0x72, 0x6f,
        0x77, 0x6e, 0x20, 0x66, 0x6f, 0x78, 0x20, 0x6a, 0x75, 0x6d, 0x70, 0x73, 0x20, 0x6f, 0x76,
        0x65, 0x72, 0x20, 0x1f, 0x00, 0x91, 0x6c, 0x61, 0x7a, 0x79, 0x20, 0x64, 0x6f, 0x67, 0x2e,
        0x0e, 0x00, 0x0f, 0x2d, 0x00, 0xc5, 0x50, 0x64, 0x6f, 0x67, 0x2e, 0x20,
    ];

    // 655 -> 38 bytes
    const REF_RUNS: &[u8] = &[
        0x1f, 0x41, 0x01, 0x00, 0xff, 0x19, 0x1c, 0x42, 0x01, 0x00, 0x00, 0x3d, 0x01, 0x0f, 0x08,
        0x00, 0xff, 0x2a, 0xf0, 0x03, 0x74, 0x61, 0x69, 0x6c, 0x2d, 0x6c, 0x69, 0x74, 0x65, 0x72,
        0x61, 0x6c, 0x73, 0x2d, 0x68, 0x65, 0x72, 0x65,
    ];

    // 636 -> 84 bytes
    const REF_MIXED: &[u8] = &[
        0xff, 0x32, 0xee, 0x12, 0xba, 0x0c, 0x9c, 0x4e, 0x7f, 0x90, 0xa9, 0x83, 0xc1, 0x01, 0xf4,
        0x6c, 0xdf, 0xaa, 0xbc, 0x1b, 0xaf, 0x18, 0xb4, 0x41, 0xb0, 0x0c, 0x40, 0x3d, 0xaa, 0x08,
        0xf8, 0x0e, 0x90, 0xff, 0x71, 0x10, 0xa5, 0x55, 0x25, 0x6d, 0xa3, 0x66, 0x0c, 0xde, 0x5b,
        0x7f, 0xd2, 0x02, 0x11, 0xe6, 0x3f, 0x15, 0x9a, 0x3a, 0xa2, 0x03, 0x9f, 0xf9, 0xd9, 0x79,
        0x42, 0x95, 0xcb, 0x55, 0x38, 0x23, 0x5a, 0x01, 0x00, 0xff, 0xe1, 0x0f, 0x34, 0x02, 0x2d,
        0x80, 0x74, 0x72, 0x61, 0x69, 0x6c, 0x69, 0x6e, 0x67,
    ];

    fn ref_text_plain() -> Vec<u8> {
        b"the quick brown fox jumps over the lazy dog. ".repeat(6)
    }

    fn ref_runs_plain() -> Vec<u8> {
        let mut d = vec![b'A'; 300];
        d.extend(std::iter::repeat(b'B').take(17));
        for _ in 0..20 {
            d.extend_from_slice(b"AAAABBBBAAAABBBB");
        }
        d.extend_from_slice(b"tail-literals-here");
        d
    }

    fn ref_mixed_plain() -> Vec<u8> {
        let noise: Vec<u8> = (0..64u64).map(|i| (splitmix64(i) >> 13) as u8).collect();
        let mut d = noise.clone();
        d.extend(std::iter::repeat(b'Z').take(500));
        d.extend_from_slice(&noise);
        d.extend_from_slice(b"trailing");
        d
    }

    #[test]
    fn decodes_reference_lz4_output() {
        for (enc, plain) in [
            (REF_TEXT, ref_text_plain()),
            (REF_RUNS, ref_runs_plain()),
            (REF_MIXED, ref_mixed_plain()),
        ] {
            let got = decompress(enc, plain.len()).expect("reference block failed to decode");
            assert_eq!(got, plain);
        }
    }

    #[test]
    fn decodes_reference_all_literal_block() {
        // 400 bytes of noise the reference could not compress: it emits a
        // single 0xF0 token with two continuation bytes (255 + 130 + 15).
        let plain: Vec<u8> = (0..400u64).map(|i| (splitmix64(i) & 0x7f) as u8).collect();
        let mut enc = vec![0xf0u8, 0xff, 0x82];
        enc.extend_from_slice(&plain);
        assert_eq!(decompress(&enc, 400).unwrap(), plain);
    }

    #[test]
    fn reference_blocks_survive_truncation_and_corruption() {
        for enc in [REF_TEXT, REF_RUNS, REF_MIXED] {
            for cut in 0..enc.len() {
                let _ = decompress(&enc[..cut], 1000);
            }
            for pos in 0..enc.len() {
                for bit in 0..8 {
                    let mut c = enc.to_vec();
                    c[pos] ^= 1 << bit;
                    let _ = decompress(&c, 655);
                }
            }
        }
    }
    // -- confirmed defect, see review -------------------------------------

    /// `compress` is fine; its **caller** is not. `persist::format`'s reader
    /// rejects any coded `u64` array whose decoded size exceeds 64x the frame
    /// it lives in, but this codec legitimately reaches ~255x. A one-granule
    /// column of two far-apart values compresses 154x, and the reader then
    /// refuses the block the writer just produced.
    ///
    /// Reachable end to end: a `MergeTree ... ORDER BY tuple()` table with a
    /// single `UInt64` column writes a part that can never be reopened.
    #[test]
    fn writer_reader_roundtrip_on_a_very_compressible_array() {
        use crate::persist::format::{Reader, Writer};

        let words: Vec<u64> = (0..crate::common::GRANULE_SIZE as u64)
            .map(|i| if i % 2 == 0 { 0 } else { 1u64 << 63 })
            .collect();

        let mut w = Writer::new();
        w.u64_words_coded(&words);
        let buf = w.finish();

        let mut r = Reader::new(&buf);
        let got = r
            .u64_words_coded()
            .expect("reader rejected the writer's own block");
        assert_eq!(got.into_vec(), words);
    }
}
