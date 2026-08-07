//! The byte-level substrate every on-disk structure is built from.
//!
//! This module knows nothing about granules, columns or schemas -- it is the
//! serialization plumbing underneath them. Three design decisions shape it:
//!
//! **Fixed-width little-endian for payload, varint for structure.** Bulk
//! payload (packed `u64` words, dictionary offsets) is written as raw LE words
//! so a future mmap reader can cast the region in place on any target we care
//! about. Structural numbers -- lengths, counts, per-granule row totals -- are
//! LEB128, because they are overwhelmingly small and a part header made of
//! `u64`s is mostly zero bytes.
//!
//! **Varints are canonical.** A value has exactly one legal encoding; overlong
//! forms (`80 00` for zero) are rejected rather than accepted-and-normalized.
//! Two writers emitting the same logical part must produce byte-identical
//! files, otherwise part checksums stop being usable as identity and merges
//! cannot dedupe.
//!
//! **Every read is hostile-input hardened.** Files are mutated by bit rot,
//! partial writes and outright malice. The reader therefore never indexes a
//! slice without a bounds check, never `unwrap`s on file-derived data, and --
//! the failure mode that actually kills processes -- never sizes an allocation
//! from a length prefix before checking that prefix against the bytes that
//! actually remain. A corrupt `varint` claiming 2^60 elements must cost one
//! comparison, not 8 exabytes of `Vec` capacity.
//!
//! The checksum is built from [`crate::common::hash`] rather than a second
//! hash construction (CRC tables, xxhash constants) so the crate keeps exactly
//! one mixing primitive to audit and tune. It is *not* a MAC: it detects
//! corruption, not a determined forger.

use crate::common::{hash_bytes, mum, zz_dec, zz_enc, Error, Result};

/// File magic. The trailing NUL keeps it 8 bytes so a header is one aligned
/// `u64` load, and makes `file`/`strings` output readable when debugging.
pub const MAGIC: [u8; 8] = *b"GRANULR\0";

/// Bumped whenever the layout of anything above this module changes
/// incompatibly. Readers accept `MIN_READ_VERSION..=FORMAT_VERSION`.
pub const FORMAT_VERSION: u32 = 2;

/// Oldest layout this build can still read.
///
/// v2 padded frame bodies and word arrays to 8 bytes so packed lanes can be
/// read directly out of a mapping. There is no way to reinterpret a v1 file's
/// unaligned words in place, and a reader that silently fell back to copying
/// every column would quietly give up the property the version exists to
/// provide -- so v1 is rejected rather than emulated. Rewrite such parts by
/// reading them with a v1 build and re-inserting.
pub const MIN_READ_VERSION: u32 = 2;

/// `MAGIC` + version word.
pub const HEADER_LEN: usize = MAGIC.len() + 4;

/// offset + version + checksum + trailing `MAGIC`.
pub const FOOTER_LEN: usize = 8 + 4 + 8 + MAGIC.len();

/// Domain-separates file checksums from the hash-table / bloom hash streams,
/// so a colliding pair found for one is not automatically one for the other.
const CHECKSUM_SEED: u64 = 0x9AE1_6A3B_2F90_404F;

/// Largest legal LEB128 encoding of a `u64`: 9 full groups plus one bit.
const VARINT_MAX_LEN: usize = 10;

// ---------------------------------------------------------------------------
// checksum
// ---------------------------------------------------------------------------

/// Fast non-cryptographic 64-bit checksum over a byte slice.
///
/// `hash_bytes` already folds the length in and finishes with a `mum`; the
/// extra round here is pure domain separation, so a value that happens to be
/// used as a group key never shares a digest with a section checksum.
#[inline]
pub fn checksum(bytes: &[u8]) -> u64 {
    mum(
        hash_bytes(bytes, CHECKSUM_SEED) ^ 0x2545_F491_4F6C_DD1D,
        0xD6E8_FEB8_6659_FD93,
    )
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Append-only byte builder.
///
/// Infallible by construction: it writes into a `Vec<u8>`, so there is no
/// error path to thread through the (long, mechanical) serialization code.
/// Actual I/O errors surface where the buffer is handed to the filesystem.
#[derive(Clone, Debug, Default)]
pub struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Writer { buf: Vec::new() }
    }

    pub fn with_capacity(n: usize) -> Self {
        Writer { buf: Vec::with_capacity(n) }
    }

    #[inline]
    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    #[inline]
    pub fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    pub fn i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Stored as raw IEEE-754 bits, not via a decimal round trip: that is the
    /// only encoding that preserves `-0.0`, both infinities and NaN payloads.
    #[inline]
    pub fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }

    /// LEB128. 1 byte for values < 128, 10 for the top of the range.
    #[inline]
    pub fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.buf.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.buf.push(v as u8);
    }

    /// Zigzag then LEB128, so small negatives cost one byte instead of ten.
    #[inline]
    pub fn svarint(&mut self, v: i64) {
        self.varint(zz_enc(v));
    }

    /// Length-prefixed blob.
    pub fn bytes(&mut self, b: &[u8]) {
        self.varint(b.len() as u64);
        self.buf.extend_from_slice(b);
    }

    /// UTF-8 is guaranteed by the `&str`, so the reader's validation is only
    /// ever exercised by corruption.
    pub fn str(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }

    /// Element count as a varint, then the words little-endian. Words stay
    /// fixed-width: these are bit-packed payloads with no small-value bias,
    /// and varints would both grow them and defeat a future zero-copy read.
    pub fn u64_slice(&mut self, v: &[u64]) {
        self.varint(v.len() as u64);
        self.buf.reserve(v.len() * 8);
        for &x in v {
            self.buf.extend_from_slice(&x.to_le_bytes());
        }
    }

    /// Words stored verbatim: tagged [`CODEC_NONE`] and padded to an 8-byte
    /// boundary, so a mapped reader can reinterpret them as `&[u64]` in place
    /// instead of copying them onto the heap.
    ///
    /// Use this for anything read at random -- the point-lookup index above
    /// all. Compressing an array costs it O(1) access, and an index that has
    /// to be decompressed to be probed is not an index.
    ///
    /// The padding only buys anything because [`write_framed`] aligns frame
    /// bodies too: alignment here is relative to the start of the buffer being
    /// written, and that only translates into an aligned *address* if every
    /// enclosing frame preserved it. The chain bottoms out at the mapping base,
    /// which the kernel returns page-aligned.
    pub fn u64_words(&mut self, v: &[u64]) {
        self.varint(v.len() as u64);
        self.u8(CODEC_NONE);
        self.align_to(8);
        self.buf.reserve(v.len() * 8);
        for &x in v {
            self.buf.extend_from_slice(&x.to_le_bytes());
        }
    }

    /// [`Writer::u64_words`], but LZ4'd when the block comes out at least
    /// [`CODEC_MIN_SAVING`]th smaller.
    ///
    /// Layout: `varint count | u8 codec | payload`, where an uncompressed
    /// payload is padded to 8 bytes and a compressed one is a length-prefixed
    /// block. The count is outside the codec branch so a reader can size the
    /// output buffer before it knows what it is decoding.
    pub fn u64_words_coded(&mut self, v: &[u64]) {
        self.varint(v.len() as u64);
        let raw: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        match crate::encoding::lz4::compress(&raw) {
            Some(c) if c.len() + raw.len() / CODEC_MIN_SAVING <= raw.len() => {
                self.u8(CODEC_LZ4);
                self.bytes(&c);
            }
            _ => {
                self.u8(CODEC_NONE);
                self.align_to(8);
                self.raw(&raw);
            }
        }
    }

    pub fn u32_slice(&mut self, v: &[u32]) {
        self.varint(v.len() as u64);
        self.buf.reserve(v.len() * 4);
        for &x in v {
            self.buf.extend_from_slice(&x.to_le_bytes());
        }
    }

    /// Raw bytes, no length prefix. For sections whose extent is known from
    /// elsewhere (framed bodies, magic).
    pub fn raw(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }

    #[inline]
    pub fn pos(&self) -> usize {
        self.buf.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Pad with zeros up to a multiple of `n`. Used before word arrays so an
    /// mmap-backed reader can hand out `&[u64]` without a copy.
    pub fn align_to(&mut self, n: usize) {
        debug_assert!(n.is_power_of_two(), "alignment {n} is not a power of two");
        if n <= 1 {
            return;
        }
        let r = self.buf.len() % n;
        if r != 0 {
            self.buf.resize(self.buf.len() + (n - r), 0);
        }
    }

    /// Overwrite a previously reserved 4-byte slot. Lets a writer emit a
    /// placeholder length, write the section, then backpatch the real value
    /// without buffering the section separately.
    pub fn patch_u32(&mut self, at: usize, v: u32) -> Result<()> {
        let end = at.checked_add(4).ok_or_else(|| Error::corruption("patch offset overflow"))?;
        let dst = self
            .buf
            .get_mut(at..end)
            .ok_or_else(|| Error::corruption(format!("patch at {at} is past end of buffer")))?;
        dst.copy_from_slice(&v.to_le_bytes());
        Ok(())
    }

    pub fn patch_u64(&mut self, at: usize, v: u64) -> Result<()> {
        let end = at.checked_add(8).ok_or_else(|| Error::corruption("patch offset overflow"))?;
        let dst = self
            .buf
            .get_mut(at..end)
            .ok_or_else(|| Error::corruption(format!("patch at {at} is past end of buffer")))?;
        dst.copy_from_slice(&v.to_le_bytes());
        Ok(())
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    pub fn finish(self) -> Vec<u8> {
        self.buf
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Zero-copy cursor over a byte buffer.
///
/// Borrowed slices handed out by [`Reader::bytes`] / [`Reader::str`] carry the
/// buffer's lifetime, not the cursor's, so a part reader can build a structure
/// of `&'a str` views over one mapped file without copying any of it.
#[derive(Clone, Debug)]
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    /// The single choke point for advancing the cursor: every other read goes
    /// through here, so bounds checking exists in exactly one place.
    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::corruption("read length overflows address space"))?;
        let s = self.buf.get(self.pos..end).ok_or_else(|| {
            Error::corruption(format!(
                "unexpected end of buffer: need {n} bytes at offset {}, {} remain",
                self.pos,
                self.remaining()
            ))
        })?;
        self.pos = end;
        Ok(s)
    }

    #[inline]
    pub fn u8(&mut self) -> Result<u8> {
        let b = self.take(1)?;
        b.first().copied().ok_or_else(|| Error::corruption("short read"))
    }

    #[inline]
    pub fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes(b.try_into().map_err(|_| Error::corruption("short u16"))?))
    }

    #[inline]
    pub fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes(b.try_into().map_err(|_| Error::corruption("short u32"))?))
    }

    #[inline]
    pub fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes(b.try_into().map_err(|_| Error::corruption("short u64"))?))
    }

    #[inline]
    pub fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }

    #[inline]
    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    /// LEB128 with two hard rules: no encoding longer than 10 bytes, and no
    /// non-canonical form. Both matter -- the first bounds the loop against a
    /// buffer of `0xFF`s, the second keeps serialization a bijection.
    pub fn varint(&mut self) -> Result<u64> {
        let mut out = 0u64;
        let mut shift = 0u32;
        loop {
            let b = self.u8()?;
            if shift == (VARINT_MAX_LEN as u32 - 1) * 7 {
                // 10th byte: only bit 63 is left to fill.
                return match b {
                    1 => Ok(out | (1u64 << 63)),
                    0 => Err(Error::corruption("non-canonical varint: overlong encoding")),
                    _ => Err(Error::corruption(format!("varint overflows u64 (tail byte {b:#04x})"))),
                };
            }
            out |= ((b & 0x7F) as u64) << shift;
            if b & 0x80 == 0 {
                // A zero continuation-free byte after at least one group means
                // the value was padded with redundant zero groups.
                if shift > 0 && b == 0 {
                    return Err(Error::corruption("non-canonical varint: trailing zero group"));
                }
                return Ok(out);
            }
            shift += 7;
        }
    }

    pub fn svarint(&mut self) -> Result<i64> {
        Ok(zz_dec(self.varint()?))
    }

    /// Length-prefixed blob. The length is checked against what is physically
    /// left in the buffer before the slice is formed, so a corrupt prefix
    /// cannot make us hand out (or reserve) memory we do not have.
    pub fn bytes(&mut self) -> Result<&'a [u8]> {
        let n = self.varint()?;
        if n > self.remaining() as u64 {
            return Err(Error::corruption(format!(
                "blob claims {n} bytes, only {} remain",
                self.remaining()
            )));
        }
        self.take(n as usize)
    }

    pub fn str(&mut self) -> Result<&'a str> {
        let b = self.bytes()?;
        Ok(std::str::from_utf8(b)?)
    }

    pub fn u64_slice(&mut self) -> Result<Vec<u64>> {
        let n = self.checked_count(8)?;
        let raw = self.take(n * 8)?;
        raw.chunks_exact(8)
            .map(|c| {
                c.try_into()
                    .map(u64::from_le_bytes)
                    .map_err(|_| Error::corruption("short u64 word"))
            })
            .collect()
    }

    /// Read an array written by [`Writer::u64_words`] or
    /// [`Writer::u64_words_coded`] -- the codec tag says which.
    ///
    /// The uncompressed case borrows; the compressed case allocates exactly
    /// the decoded size, which the header states before the codec branch so a
    /// corrupt block cannot steer the allocation.
    pub fn u64_words_coded(&mut self) -> Result<Words<'a>> {
        let n = self.varint()? as usize;
        let bytes = n
            .checked_mul(8)
            .ok_or_else(|| Error::corruption(format!("word count {n} overflows a byte length")))?;
        match self.u8()? {
            CODEC_NONE => {
                self.align_to(8)?;
                Ok(Words::Raw(self.take(bytes)?))
            }
            CODEC_LZ4 => {
                let block = self.bytes()?;
                // Bound the allocation before making it, using the codec's
                // own proven ceiling on how far a block of this size can
                // expand. A corrupt varint claiming 2^40 words has to cost a
                // comparison, not a terabyte of `Vec`.
                let ceiling = crate::encoding::lz4::expansion_ceiling(block.len());
                if bytes > ceiling {
                    return Err(Error::corruption(format!(
                        "compressed array claims {n} words ({bytes} bytes) from a \
                         {}-byte block, which can expand to at most {ceiling}",
                        block.len()
                    )));
                }
                let mut out = vec![0u64; n];
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), bytes)
                };
                crate::encoding::lz4::decompress_into(block, dst)?;
                if cfg!(target_endian = "big") {
                    for w in &mut out {
                        *w = w.swap_bytes();
                    }
                }
                Ok(Words::Owned(out))
            }
            other => Err(Error::corruption(format!("unknown word codec {other}"))),
        }
    }

    /// Total size of the buffer behind this cursor, for sanity-bounding
    /// allocations derived from length prefixes.
    #[inline]
    pub fn buf_len(&self) -> usize {
        self.buf.len()
    }

    pub fn u32_slice(&mut self) -> Result<Vec<u32>> {
        let n = self.checked_count(4)?;
        let raw = self.take(n * 4)?;
        raw.chunks_exact(4)
            .map(|c| {
                c.try_into()
                    .map(u32::from_le_bytes)
                    .map_err(|_| Error::corruption("short u32 word"))
            })
            .collect()
    }

    /// Read an element count and prove `count * elem_size` bytes are actually
    /// present. This is what stands between a flipped bit in a length prefix
    /// and a multi-terabyte `Vec::with_capacity`.
    fn checked_count(&mut self, elem_size: usize) -> Result<usize> {
        let n = self.varint()?;
        let need = n
            .checked_mul(elem_size as u64)
            .ok_or_else(|| Error::corruption(format!("element count {n} overflows a byte length")))?;
        if need > self.remaining() as u64 {
            return Err(Error::corruption(format!(
                "slice of {n} x {elem_size}B needs {need} bytes, only {} remain",
                self.remaining()
            )));
        }
        // `need <= remaining <= usize::MAX`, so the cast cannot truncate.
        Ok(n as usize)
    }

    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn seek(&mut self, pos: usize) -> Result<()> {
        if pos > self.buf.len() {
            return Err(Error::corruption(format!(
                "seek to {pos} past end of {}-byte buffer",
                self.buf.len()
            )));
        }
        self.pos = pos;
        Ok(())
    }

    pub fn skip(&mut self, n: usize) -> Result<()> {
        self.take(n).map(|_| ())
    }

    /// Mirror of [`Writer::align_to`]. Padding content is skipped, not
    /// verified: it is filler, and demanding zeros would freeze an
    /// implementation detail of today's writer into the format.
    pub fn align_to(&mut self, n: usize) -> Result<()> {
        debug_assert!(n.is_power_of_two(), "alignment {n} is not a power of two");
        if n <= 1 {
            return Ok(());
        }
        let r = self.pos % n;
        if r == 0 {
            return Ok(());
        }
        self.skip(n - r)
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// Everything from the cursor to the end, without advancing.
    #[inline]
    pub fn rest(&self) -> &'a [u8] {
        // `pos <= buf.len()` is an invariant of `take`/`seek`.
        self.buf.get(self.pos..).unwrap_or(&[])
    }
}

// ---------------------------------------------------------------------------
// word views
// ---------------------------------------------------------------------------

/// Words stored verbatim: 8-aligned, little-endian, mappable in place.
pub const CODEC_NONE: u8 = 0;

/// Words stored as one LZ4 block.
///
/// Compression is decided per array by whether it pays, not by configuration.
/// Bit-packing has already removed the leading zeros, so what LZ4 finds is
/// whatever *structure* survived that -- runs of equal values, repeating
/// stride patterns, dictionary codes with locality. On a dense high-entropy
/// column it finds nothing and the array stays raw.
pub const CODEC_LZ4: u8 = 1;

/// Reserved: Zstandard.
///
/// Not implemented, and deliberately not hand-rolled. LZ4 is a few hundred
/// lines because its format is literals and back-references; zstd adds FSE and
/// Huffman entropy coding with negotiated tables, and a from-scratch decoder
/// for it is a liability, not a feature. The tag is spoken for so that wiring
/// the `zstd` crate in later is two match arms and no format change -- if the
/// zero-dependency rule is ever worth trading for the extra ratio.
#[allow(dead_code)]
pub const CODEC_ZSTD: u8 = 2;

/// The saving that justifies giving up the mapping.
///
/// A compressed array cannot be read in place: it costs a decompression pass
/// and resident heap for the result, and it gives up O(1) random access into
/// the column, which is what makes point lookups cheap. An eighth off is the
/// point where the bytes saved are worth those three things; below it, storing
/// raw is simply better.
const CODEC_MIN_SAVING: usize = 8;

/// Where a word array's bytes came from.
pub enum Words<'a> {
    /// Still in the buffer, little-endian. May be castable in place; ask
    /// [`as_u64_slice`].
    Raw(&'a [u8]),
    /// Decompressed onto the heap.
    Owned(Vec<u64>),
}

impl Words<'_> {
    /// Number of `u64`s the array holds.
    pub fn len(&self) -> usize {
        match self {
            Words::Raw(b) => b.len() / 8,
            Words::Owned(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Owned words, copying and byte-swapping only if it has to.
    pub fn into_vec(self) -> Vec<u64> {
        match self {
            Words::Raw(b) => to_u64_vec(b),
            Words::Owned(v) => v,
        }
    }
}

/// Reinterpret `b` as `[u64]` in place, or `None` if this buffer cannot be
/// borrowed that way.
///
/// Two conditions. The bytes must be 8-aligned *as an address* -- the writer
/// arranges the offset, but a buffer that did not start aligned carries the
/// skew through, which is why this is a runtime check rather than a format
/// guarantee. And the target must be little-endian, since that is the byte
/// order on disk; a big-endian reader has to go through [`to_u64_vec`] and
/// swap. Callers must have a copying fallback for both.
#[inline]
pub fn as_u64_slice(b: &[u8]) -> Option<&[u64]> {
    if !cfg!(target_endian = "little") || b.as_ptr() as usize % align_of::<u64>() != 0 {
        return None;
    }
    // SAFETY: alignment is checked above, the length is exact, `u64` has no
    // invalid bit patterns, and the returned slice borrows `b` so the backing
    // buffer outlives it.
    Some(unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<u64>(), b.len() / 8) })
}

/// Copy `b` into owned words, decoding the on-disk little-endian order. The
/// fallback for whenever [`as_u64_slice`] declines.
pub fn to_u64_vec(b: &[u8]) -> Vec<u64> {
    b.chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().expect("chunks_exact yields 8 bytes")))
        .collect()
}

// ---------------------------------------------------------------------------
// framing
// ---------------------------------------------------------------------------

/// Write `body` as a self-delimiting, self-verifying section.
///
/// Layout: `varint len | u64 checksum | body`. The checksum sits *before* the
/// body so a reader that only wants to skip the section still touches a fixed
/// prefix, and so a torn write that loses the tail is caught by the length
/// check rather than by reading into the next section.
pub fn write_framed(w: &mut Writer, body: &[u8]) {
    w.varint(body.len() as u64);
    w.u64(checksum(body));
    w.raw(body);
}

/// [`write_framed`], plus zero padding that puts the body on an 8-byte
/// boundary.
///
/// Part files use this and nothing else does. The padding sits between the
/// header and the body -- not after it -- because that is what cancels the
/// variable-length `varint` in front, and an 8-aligned body offset is what
/// lets word arrays nested inside be read as `&[u64]` straight out of a
/// mapping. The write-ahead log deliberately keeps the unpadded frame: it
/// reads a run of zeros as bit rot, so padding there would manufacture the
/// exact pattern it is watching for.
pub fn write_framed_aligned(w: &mut Writer, body: &[u8]) {
    w.varint(body.len() as u64);
    w.u64(checksum(body));
    w.align_to(8);
    w.raw(body);
}

/// Read a section written by [`write_framed_aligned`].
///
/// The pad is not covered by the checksum, so it is checked directly: it must
/// be zero. Without that, the seven bytes in front of every body would be the
/// one place in a part file where a flipped bit goes unnoticed.
pub fn read_framed_aligned<'a>(r: &mut Reader<'a>) -> Result<&'a [u8]> {
    let len = r.varint()?;
    let want = r.u64()?;
    let pad = r.pos().next_multiple_of(8) - r.pos();
    if r.take(pad)?.iter().any(|&b| b != 0) {
        return Err(Error::corruption("frame alignment padding is not zero"));
    }
    finish_frame(r, len, want)
}

/// Read a section written by [`write_framed`], verifying its checksum.
pub fn read_framed<'a>(r: &mut Reader<'a>) -> Result<&'a [u8]> {
    let len = r.varint()?;
    let want = r.u64()?;
    finish_frame(r, len, want)
}

fn finish_frame<'a>(r: &mut Reader<'a>, len: u64, want: u64) -> Result<&'a [u8]> {
    if len > r.remaining() as u64 {
        return Err(Error::corruption(format!(
            "framed section claims {len} bytes, only {} remain",
            r.remaining()
        )));
    }
    let body = r.take(len as usize)?;
    let got = checksum(body);
    if got != want {
        return Err(Error::corruption(format!(
            "checksum mismatch over {len} bytes: stored {want:#018x}, computed {got:#018x}"
        )));
    }
    Ok(body)
}

// ---------------------------------------------------------------------------
// header / footer
// ---------------------------------------------------------------------------

/// `MAGIC | u32 version`.
pub fn write_header(w: &mut Writer) {
    w.raw(&MAGIC);
    w.u32(FORMAT_VERSION);
}

/// Verify magic and version, returning the file's format version so callers
/// can branch on older-but-supported layouts.
pub fn read_header(r: &mut Reader) -> Result<u32> {
    let m = r.take(MAGIC.len())?;
    if m != MAGIC {
        return Err(Error::corruption(format!("bad magic {m:02x?}, expected {MAGIC:02x?}")));
    }
    let v = r.u32()?;
    if v < MIN_READ_VERSION || v > FORMAT_VERSION {
        return Err(Error::corruption(format!(
            "unsupported format version {v}; this build reads \
             {MIN_READ_VERSION}..={FORMAT_VERSION}"
        )));
    }
    Ok(v)
}

/// `u64 meta_offset | u32 version | u64 checksum | MAGIC`.
///
/// A trailing fixed-size footer is what makes a part readable without scanning
/// it: open, read the last [`FOOTER_LEN`] bytes, jump straight to the metadata
/// block. The magic is repeated at the very end so a truncated file (the
/// classic crash-during-append) is diagnosed as truncation rather than as a
/// garbage offset.
pub fn write_footer(w: &mut Writer, meta_offset: u64) {
    let start = w.pos();
    w.u64(meta_offset);
    w.u32(FORMAT_VERSION);
    let ck = checksum(&w.as_slice()[start..]);
    w.u64(ck);
    w.raw(&MAGIC);
}

/// Parse the footer at the end of `buf`, returning the metadata offset.
pub fn read_footer(buf: &[u8]) -> Result<u64> {
    if buf.len() < FOOTER_LEN {
        return Err(Error::corruption(format!(
            "file of {} bytes is shorter than the {FOOTER_LEN}-byte footer",
            buf.len()
        )));
    }
    let mut r = Reader::new(buf);
    r.seek(buf.len() - FOOTER_LEN)?;
    let start = r.pos();
    let off = r.u64()?;
    let ver = r.u32()?;
    let want = r.u64()?;
    let m = r.take(MAGIC.len())?;
    if m != MAGIC {
        return Err(Error::corruption("missing trailing magic: file is truncated or not a part"));
    }
    let covered = buf
        .get(start..start + 12)
        .ok_or_else(|| Error::corruption("footer body out of range"))?;
    let got = checksum(covered);
    if got != want {
        return Err(Error::corruption(format!(
            "footer checksum mismatch: stored {want:#018x}, computed {got:#018x}"
        )));
    }
    if ver < MIN_READ_VERSION || ver > FORMAT_VERSION {
        return Err(Error::corruption(format!(
            "unsupported format version {ver} in footer; this build reads \
             {MIN_READ_VERSION}..={FORMAT_VERSION}"
        )));
    }
    if off > (buf.len() - FOOTER_LEN) as u64 {
        return Err(Error::corruption(format!(
            "metadata offset {off} points past the footer of a {}-byte file",
            buf.len()
        )));
    }
    Ok(off)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::splitmix64;

    fn is_corrupt(e: &Error) -> bool {
        matches!(e, Error::Corruption(_) | Error::Io(_))
    }

    /// Every primitive, in one buffer, so the round-trip and the truncation
    /// fuzz share a single description of "a valid document".
    fn build_doc() -> Vec<u8> {
        let mut w = Writer::with_capacity(256);
        write_header(&mut w);
        w.u8(0xAB);
        w.u16(0xBEEF);
        w.u32(0xDEAD_BEEF);
        w.u64(u64::MAX);
        w.i64(i64::MIN);
        w.f64(-0.0);
        w.varint(0);
        w.varint(300);
        w.varint(u64::MAX);
        w.svarint(-1);
        w.svarint(i64::MIN);
        w.bytes(b"");
        w.bytes(&[0xFF; 40]);
        w.str("granular \u{1F600}");
        w.align_to(8);
        w.u64_slice(&[1, 2, 3, u64::MAX]);
        w.u32_slice(&[7, 8]);
        write_framed(&mut w, b"framed body bytes");
        w.u32(0x5A5A_5A5A);
        w.finish()
    }

    /// Mirror of [`build_doc`]. Mismatches are returned as errors rather than
    /// asserted, so the same routine can be aimed at a deliberately corrupted
    /// buffer without the test harness panicking on a *legitimately* different
    /// value.
    fn parse_doc(buf: &[u8]) -> Result<()> {
        macro_rules! check {
            ($cond:expr, $what:expr) => {
                if !$cond {
                    return Err(Error::corruption(concat!("value mismatch: ", $what)));
                }
            };
        }
        let mut r = Reader::new(buf);
        check!(read_header(&mut r)? == FORMAT_VERSION, "version");
        check!(r.u8()? == 0xAB, "u8");
        check!(r.u16()? == 0xBEEF, "u16");
        check!(r.u32()? == 0xDEAD_BEEF, "u32");
        check!(r.u64()? == u64::MAX, "u64");
        check!(r.i64()? == i64::MIN, "i64");
        let z = r.f64()?;
        check!(z == 0.0 && z.is_sign_negative(), "f64 -0.0");
        check!(r.varint()? == 0, "varint 0");
        check!(r.varint()? == 300, "varint 300");
        check!(r.varint()? == u64::MAX, "varint max");
        check!(r.svarint()? == -1, "svarint -1");
        check!(r.svarint()? == i64::MIN, "svarint min");
        check!(r.bytes()?.is_empty(), "empty blob");
        check!(r.bytes()? == &[0xFF; 40][..], "blob");
        check!(r.str()? == "granular \u{1F600}", "str");
        r.align_to(8)?;
        check!(r.u64_slice()? == vec![1, 2, 3, u64::MAX], "u64 slice");
        check!(r.u32_slice()? == vec![7, 8], "u32 slice");
        check!(read_framed(&mut r)? == &b"framed body bytes"[..], "framed body");
        check!(r.u32()? == 0x5A5A_5A5A, "trailer");
        check!(r.remaining() == 0, "trailing bytes");
        Ok(())
    }

    // -- header / footer ---------------------------------------------------

    #[test]
    fn header_roundtrip() {
        let mut w = Writer::new();
        write_header(&mut w);
        let buf = w.finish();
        assert_eq!(buf.len(), HEADER_LEN);
        let mut r = Reader::new(&buf);
        assert_eq!(read_header(&mut r).unwrap(), FORMAT_VERSION);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn header_rejects_bad_magic() {
        let mut w = Writer::new();
        write_header(&mut w);
        let mut buf = w.finish();
        buf[3] ^= 0x20;
        assert!(is_corrupt(&read_header(&mut Reader::new(&buf)).unwrap_err()));
    }

    #[test]
    fn header_rejects_future_version() {
        let mut w = Writer::new();
        w.raw(&MAGIC);
        w.u32(FORMAT_VERSION + 7);
        let buf = w.finish();
        let e = read_header(&mut Reader::new(&buf)).unwrap_err();
        assert!(e.to_string().contains("unsupported format version"));
    }

    #[test]
    fn header_rejects_zero_version() {
        let mut w = Writer::new();
        w.raw(&MAGIC);
        w.u32(0);
        let buf = w.finish();
        assert!(is_corrupt(&read_header(&mut Reader::new(&buf)).unwrap_err()));
    }

    #[test]
    fn header_rejects_empty_and_short_buffers() {
        for n in 0..HEADER_LEN {
            let mut w = Writer::new();
            write_header(&mut w);
            let buf = w.finish();
            assert!(read_header(&mut Reader::new(&buf[..n])).is_err(), "n={n}");
        }
    }

    #[test]
    fn footer_roundtrip() {
        let mut w = Writer::new();
        write_header(&mut w);
        w.raw(&[0u8; 64]);
        let meta = w.pos() as u64;
        w.raw(b"metadata");
        write_footer(&mut w, meta);
        let buf = w.finish();
        assert_eq!(read_footer(&buf).unwrap(), meta);
    }

    #[test]
    fn footer_rejects_short_file() {
        for n in 0..FOOTER_LEN {
            let buf = vec![0u8; n];
            assert!(read_footer(&buf).is_err(), "n={n}");
        }
    }

    #[test]
    fn footer_detects_every_single_bit_flip() {
        let mut w = Writer::new();
        w.raw(b"body");
        write_footer(&mut w, 4);
        let buf = w.finish();
        for byte in 0..buf.len() {
            for bit in 0..8 {
                let mut c = buf.clone();
                c[byte] ^= 1 << bit;
                if byte < 4 {
                    continue; // outside the footer: not covered by design
                }
                assert!(read_footer(&c).is_err(), "byte={byte} bit={bit}");
            }
        }
    }

    #[test]
    fn footer_rejects_offset_past_end() {
        let mut w = Writer::new();
        w.raw(b"body");
        write_footer(&mut w, 4);
        let mut buf = w.finish();
        // Rewrite offset + version and repair the checksum: a *consistent*
        // footer with a nonsense offset must still be rejected.
        let start = buf.len() - FOOTER_LEN;
        buf[start..start + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        let ck = checksum(&buf[start..start + 12]);
        buf[start + 12..start + 20].copy_from_slice(&ck.to_le_bytes());
        let e = read_footer(&buf).unwrap_err();
        assert!(e.to_string().contains("past the footer"), "{e}");
    }

    // -- fixed width -------------------------------------------------------

    #[test]
    fn fixed_width_roundtrip_boundaries() {
        let mut w = Writer::new();
        for &v in &[0u8, 1, 0x7F, 0x80, u8::MAX] {
            w.u8(v);
        }
        for &v in &[0u16, 1, 0x00FF, 0xFF00, u16::MAX] {
            w.u16(v);
        }
        for &v in &[0u32, 1, 0xFFFF, u32::MAX] {
            w.u32(v);
        }
        for &v in &[0u64, 1, u64::from(u32::MAX) + 1, u64::MAX] {
            w.u64(v);
        }
        let buf = w.finish();
        let mut r = Reader::new(&buf);
        for &v in &[0u8, 1, 0x7F, 0x80, u8::MAX] {
            assert_eq!(r.u8().unwrap(), v);
        }
        for &v in &[0u16, 1, 0x00FF, 0xFF00, u16::MAX] {
            assert_eq!(r.u16().unwrap(), v);
        }
        for &v in &[0u32, 1, 0xFFFF, u32::MAX] {
            assert_eq!(r.u32().unwrap(), v);
        }
        for &v in &[0u64, 1, u64::from(u32::MAX) + 1, u64::MAX] {
            assert_eq!(r.u64().unwrap(), v);
        }
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn fixed_width_is_little_endian() {
        let mut w = Writer::new();
        w.u32(0x0403_0201);
        assert_eq!(w.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn i64_roundtrip_boundaries() {
        let vals = [0i64, 1, -1, 42, -42, i64::MAX, i64::MIN, i64::MIN + 1, i64::MAX - 1];
        let mut w = Writer::new();
        for &v in &vals {
            w.i64(v);
        }
        let buf = w.finish();
        let mut r = Reader::new(&buf);
        for &v in &vals {
            assert_eq!(r.i64().unwrap(), v);
        }
    }

    #[test]
    fn f64_roundtrip_including_specials() {
        let vals = [
            0.0f64,
            -0.0,
            1.0,
            -1.0,
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::EPSILON,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1e-308,
        ];
        let mut w = Writer::new();
        for &v in &vals {
            w.f64(v);
        }
        w.f64(f64::NAN);
        let buf = w.finish();
        let mut r = Reader::new(&buf);
        for &v in &vals {
            let got = r.f64().unwrap();
            assert_eq!(got.to_bits(), v.to_bits(), "v={v}");
        }
        assert!(r.f64().unwrap().is_nan());
    }

    #[test]
    fn negative_zero_survives_the_roundtrip() {
        let mut w = Writer::new();
        w.f64(-0.0);
        let buf = w.finish();
        let got = Reader::new(&buf).f64().unwrap();
        assert!(got == 0.0 && got.is_sign_negative());
    }

    // -- varints -----------------------------------------------------------

    #[test]
    fn varint_roundtrip_all_length_classes() {
        let mut vals = vec![0u64, 1, u64::MAX];
        // one value either side of every 7-bit boundary
        for k in 1..=9u32 {
            let edge = 1u64 << (7 * k);
            vals.push(edge - 1);
            vals.push(edge);
        }
        let mut rng = 0x1234_5678u64;
        for _ in 0..500 {
            rng = splitmix64(rng);
            vals.push(rng);
            vals.push(rng >> (rng % 63));
        }
        for &v in &vals {
            let mut w = Writer::new();
            w.varint(v);
            let buf = w.finish();
            assert!(buf.len() <= VARINT_MAX_LEN, "v={v} len={}", buf.len());
            let mut r = Reader::new(&buf);
            assert_eq!(r.varint().unwrap(), v, "v={v}");
            assert_eq!(r.remaining(), 0, "v={v} left {} bytes", r.remaining());
        }
    }

    #[test]
    fn varint_length_matches_magnitude() {
        for (v, want) in [
            (0u64, 1usize),
            (127, 1),
            (128, 2),
            (16_383, 2),
            (16_384, 3),
            (u64::MAX >> 1, 9),
            (u64::MAX, 10),
        ] {
            let mut w = Writer::new();
            w.varint(v);
            assert_eq!(w.pos(), want, "v={v}");
        }
    }

    #[test]
    fn varint_rejects_overflowing_encoding() {
        // 10 groups whose top byte contributes more than bit 63.
        let buf = [0xFFu8; 10];
        let e = Reader::new(&buf).varint().unwrap_err();
        assert!(e.to_string().contains("overflows u64"), "{e}");
        // ... and an 11-byte run must not be accepted either.
        let buf = [0xFFu8; 16];
        assert!(Reader::new(&buf).varint().is_err());
    }

    #[test]
    fn varint_rejects_non_canonical_encodings() {
        for buf in [
            vec![0x80u8, 0x00],             // zero, padded
            vec![0x81, 0x00],               // one, padded
            vec![0x80, 0x80, 0x80, 0x00],   // zero, padded harder
            vec![0xFF; 9].into_iter().chain([0x00]).collect(), // 10th group empty
        ] {
            let e = Reader::new(&buf).varint().unwrap_err();
            assert!(e.to_string().contains("non-canonical"), "buf={buf:02x?} err={e}");
        }
    }

    #[test]
    fn varint_truncation_errors_at_every_prefix() {
        let mut w = Writer::new();
        w.varint(u64::MAX);
        let buf = w.finish();
        for n in 0..buf.len() {
            assert!(Reader::new(&buf[..n]).varint().is_err(), "n={n}");
        }
    }

    #[test]
    fn svarint_roundtrip_boundaries() {
        let vals = [0i64, 1, -1, 63, -64, 8191, -8192, i64::MAX, i64::MIN, i64::MIN + 1];
        let mut w = Writer::new();
        for &v in &vals {
            w.svarint(v);
        }
        let buf = w.finish();
        let mut r = Reader::new(&buf);
        for &v in &vals {
            assert_eq!(r.svarint().unwrap(), v, "v={v}");
        }
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn svarint_keeps_small_negatives_cheap() {
        for v in -63i64..=63 {
            let mut w = Writer::new();
            w.svarint(v);
            assert_eq!(w.pos(), 1, "v={v}");
        }
    }

    // -- blobs and slices --------------------------------------------------

    #[test]
    fn bytes_roundtrip() {
        let cases: Vec<Vec<u8>> = vec![
            vec![],
            vec![0],
            b"hello".to_vec(),
            vec![0xFF; 127],
            vec![0xAA; 128],
            vec![0x5A; 100_000],
        ];
        let mut w = Writer::new();
        for c in &cases {
            w.bytes(c);
        }
        let buf = w.finish();
        let mut r = Reader::new(&buf);
        for c in &cases {
            assert_eq!(r.bytes().unwrap(), &c[..]);
        }
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn str_roundtrip_including_unicode_and_empty() {
        let cases = ["", "a", "hits.url", "\u{1F600}\u{4F60}\u{597D}", "line\nbreak\ttab"];
        let mut w = Writer::new();
        for c in &cases {
            w.str(c);
        }
        let buf = w.finish();
        let mut r = Reader::new(&buf);
        for c in &cases {
            assert_eq!(r.str().unwrap(), *c);
        }
    }

    #[test]
    fn str_rejects_invalid_utf8() {
        let mut w = Writer::new();
        w.bytes(&[0xC3, 0x28]); // truncated 2-byte sequence
        let buf = w.finish();
        let e = Reader::new(&buf).str().unwrap_err();
        assert!(matches!(e, Error::Corruption(_)), "{e}");
    }

    #[test]
    fn bytes_rejects_length_prefix_beyond_buffer() {
        // A hostile length must be rejected on a comparison, never by trying
        // to form (or allocate) the slice.
        let mut w = Writer::new();
        w.varint(u64::MAX);
        w.raw(b"short");
        let buf = w.finish();
        let e = Reader::new(&buf).bytes().unwrap_err();
        assert!(e.to_string().contains("only 5 remain"), "{e}");

        let mut w = Writer::new();
        w.varint(1 << 40);
        let buf = w.finish();
        assert!(Reader::new(&buf).bytes().is_err());
    }

    #[test]
    fn u64_slice_roundtrip() {
        for case in [vec![], vec![0u64], vec![u64::MAX; 3], (0..1000).collect::<Vec<u64>>()] {
            let mut w = Writer::new();
            w.u64_slice(&case);
            let buf = w.finish();
            let mut r = Reader::new(&buf);
            assert_eq!(r.u64_slice().unwrap(), case);
            assert_eq!(r.remaining(), 0);
        }
    }

    #[test]
    fn u32_slice_roundtrip() {
        for case in [vec![], vec![0u32], vec![u32::MAX; 5], (0..500).collect::<Vec<u32>>()] {
            let mut w = Writer::new();
            w.u32_slice(&case);
            let buf = w.finish();
            let mut r = Reader::new(&buf);
            assert_eq!(r.u32_slice().unwrap(), case);
            assert_eq!(r.remaining(), 0);
        }
    }

    #[test]
    fn slice_count_overflow_is_rejected_without_allocating() {
        // count * 8 overflows u64 outright.
        let mut w = Writer::new();
        w.varint(u64::MAX);
        w.raw(&[0u8; 32]);
        let buf = w.finish();
        let e = Reader::new(&buf).u64_slice().unwrap_err();
        assert!(e.to_string().contains("overflows"), "{e}");

        // count * 8 fits, but the bytes are not there: also a fast reject.
        let mut w = Writer::new();
        w.varint(1 << 40);
        w.raw(&[0u8; 32]);
        let buf = w.finish();
        let e = Reader::new(&buf).u64_slice().unwrap_err();
        assert!(e.to_string().contains("only 32 remain"), "{e}");

        let mut w = Writer::new();
        w.varint(u64::MAX / 2);
        w.raw(&[0u8; 32]);
        let buf = w.finish();
        assert!(Reader::new(&buf).u32_slice().is_err());
    }

    #[test]
    fn slice_truncated_by_one_word_errors() {
        let mut w = Writer::new();
        w.u64_slice(&[1, 2, 3]);
        let mut buf = w.finish();
        buf.truncate(buf.len() - 1);
        assert!(Reader::new(&buf).u64_slice().is_err());
    }

    // -- cursor mechanics --------------------------------------------------

    #[test]
    fn align_to_pads_and_skips_symmetrically() {
        for n in [1usize, 2, 4, 8, 16, 64] {
            for lead in 0..17usize {
                let mut w = Writer::new();
                w.raw(&vec![0xEE; lead]);
                w.align_to(n);
                assert_eq!(w.pos() % n, 0, "n={n} lead={lead}");
                assert!(w.pos() >= lead && w.pos() < lead + n);
                w.u64(0x1122_3344_5566_7788);
                let buf = w.finish();
                let mut r = Reader::new(&buf);
                r.skip(lead).unwrap();
                r.align_to(n).unwrap();
                assert_eq!(r.pos() % n, 0);
                assert_eq!(r.u64().unwrap(), 0x1122_3344_5566_7788);
            }
        }
    }

    #[test]
    fn align_to_is_a_noop_when_already_aligned() {
        let mut w = Writer::new();
        w.u64(1);
        w.align_to(8);
        assert_eq!(w.pos(), 8);
    }

    #[test]
    fn reader_align_past_end_errors() {
        let buf = [0u8; 3];
        let mut r = Reader::new(&buf);
        r.skip(3).unwrap();
        assert!(r.align_to(8).is_err());
        assert_eq!(r.pos(), 3, "a failed align must not move the cursor");
    }

    #[test]
    fn seek_pos_and_remaining_track_each_other() {
        let buf: Vec<u8> = (0..64u8).collect();
        let mut r = Reader::new(&buf);
        assert_eq!(r.remaining(), 64);
        assert_eq!(r.u32().unwrap(), 0x0302_0100);
        assert_eq!(r.pos(), 4);
        assert_eq!(r.remaining(), 60);
        r.seek(0).unwrap();
        assert_eq!(r.u32().unwrap(), 0x0302_0100, "seek must rewind");
        r.seek(64).unwrap();
        assert!(r.is_empty());
        assert!(r.seek(65).is_err());
        assert_eq!(r.pos(), 64, "a failed seek must not move the cursor");
    }

    #[test]
    fn take_at_the_very_end_is_ok_but_past_it_is_not() {
        let buf = [1u8, 2, 3, 4];
        let mut r = Reader::new(&buf);
        assert_eq!(r.take(4).unwrap(), &buf[..]);
        assert_eq!(r.take(0).unwrap(), b"");
        assert!(r.take(1).is_err());
        assert!(Reader::new(&buf).take(usize::MAX).is_err());
    }

    #[test]
    fn rest_returns_the_unread_tail() {
        let buf = [1u8, 2, 3, 4];
        let mut r = Reader::new(&buf);
        r.skip(2).unwrap();
        assert_eq!(r.rest(), &[3, 4]);
        assert_eq!(r.pos(), 2, "rest must not consume");
    }

    #[test]
    fn writer_capacity_and_patching() {
        let mut w = Writer::with_capacity(64);
        assert!(w.is_empty());
        let slot = w.pos();
        w.u32(0);
        w.u64(0);
        w.raw(b"payload");
        w.patch_u32(slot, 7).unwrap();
        w.patch_u64(slot + 4, u64::MAX).unwrap();
        assert!(w.patch_u32(w.pos() - 2, 1).is_err());
        assert!(w.patch_u64(usize::MAX, 1).is_err());
        let buf = w.finish();
        let mut r = Reader::new(&buf);
        assert_eq!(r.u32().unwrap(), 7);
        assert_eq!(r.u64().unwrap(), u64::MAX);
        assert_eq!(r.rest(), b"payload");
    }

    // -- checksum ----------------------------------------------------------

    #[test]
    fn checksum_is_deterministic_and_content_sensitive() {
        assert_eq!(checksum(b"granular"), checksum(b"granular"));
        assert_ne!(checksum(b"granular"), checksum(b"granulaR"));
        assert_ne!(checksum(b""), checksum(b"\0"));
        assert_ne!(checksum(b"a"), checksum(b"a\0"), "length must be mixed in");
    }

    #[test]
    fn checksum_detects_every_single_bit_flip() {
        let base: Vec<u8> = (0..256u32).map(|i| splitmix64(i as u64) as u8).collect();
        let want = checksum(&base);
        for byte in 0..base.len() {
            for bit in 0..8 {
                let mut c = base.clone();
                c[byte] ^= 1 << bit;
                assert_ne!(checksum(&c), want, "byte={byte} bit={bit}");
            }
        }
    }

    #[test]
    fn checksum_spreads_over_short_inputs() {
        // 4096 short buffers must not collide; a weak finalizer shows up here
        // as duplicates long before it shows up in a real part.
        let mut seen = std::collections::HashSet::new();
        for i in 0..4096u64 {
            assert!(seen.insert(checksum(&i.to_le_bytes())), "collision at {i}");
        }
    }

    // -- framing -----------------------------------------------------------

    #[test]
    fn framed_roundtrip() {
        for body in [vec![], vec![0u8], b"a section".to_vec(), vec![0x77; 70_000]] {
            let mut w = Writer::new();
            write_framed(&mut w, &body);
            w.u32(0xABCD_1234); // trailing field: framing must not overrun
            let buf = w.finish();
            let mut r = Reader::new(&buf);
            assert_eq!(read_framed(&mut r).unwrap(), &body[..]);
            assert_eq!(r.u32().unwrap(), 0xABCD_1234);
            assert_eq!(r.remaining(), 0);
        }
    }

    #[test]
    fn framed_detects_a_corrupted_body() {
        let mut w = Writer::new();
        write_framed(&mut w, b"the quick brown fox");
        let mut buf = w.finish();
        let last = buf.len() - 1;
        buf[last] ^= 0x01;
        let e = read_framed(&mut Reader::new(&buf)).unwrap_err();
        assert!(e.to_string().contains("checksum mismatch"), "{e}");
        assert_eq!(e.code(), "CHECKSUM_MISMATCH");
    }

    #[test]
    fn framed_detects_a_corrupted_checksum_field() {
        let mut w = Writer::new();
        write_framed(&mut w, b"body");
        let mut buf = w.finish();
        buf[2] ^= 0x80; // inside the stored checksum
        assert!(read_framed(&mut Reader::new(&buf)).is_err());
    }

    #[test]
    fn framed_detects_every_single_bit_flip() {
        let mut w = Writer::new();
        write_framed(&mut w, b"sixteen bytes ok");
        let buf = w.finish();
        for byte in 0..buf.len() {
            for bit in 0..8 {
                let mut c = buf.clone();
                c[byte] ^= 1 << bit;
                let mut r = Reader::new(&c);
                match read_framed(&mut r) {
                    Ok(b) => panic!("flip byte={byte} bit={bit} accepted body {b:02x?}"),
                    Err(e) => assert!(is_corrupt(&e), "wrong error kind: {e}"),
                }
            }
        }
    }

    #[test]
    fn framed_rejects_an_absurd_length() {
        let mut w = Writer::new();
        w.varint(u64::MAX); // length
        w.u64(0); // checksum
        w.raw(b"tiny");
        let buf = w.finish();
        let e = read_framed(&mut Reader::new(&buf)).unwrap_err();
        assert!(e.to_string().contains("only 4 remain"), "{e}");
    }

    #[test]
    fn framed_truncated_body_errors() {
        let mut w = Writer::new();
        write_framed(&mut w, &[0x42; 300]);
        let mut buf = w.finish();
        buf.truncate(buf.len() - 100);
        assert!(read_framed(&mut Reader::new(&buf)).is_err());
    }

    // -- whole-document fuzzing -------------------------------------------

    #[test]
    fn full_document_roundtrip() {
        let buf = build_doc();
        parse_doc(&buf).unwrap();
    }

    #[test]
    fn every_truncation_errors_and_never_panics() {
        let buf = build_doc();
        for n in 0..buf.len() {
            match parse_doc(&buf[..n]) {
                Ok(()) => panic!("prefix of {n}/{} bytes parsed as complete", buf.len()),
                Err(e) => assert!(is_corrupt(&e), "n={n} wrong error kind: {e}"),
            }
        }
    }

    #[test]
    fn every_single_byte_corruption_is_survivable() {
        // Unlike truncation, a flipped byte may legitimately still parse (a
        // payload byte inside a blob carries no redundancy). The contract is
        // only that we never panic and never hand back a bogus success from a
        // *framed* section, which the framing test covers.
        let buf = build_doc();
        for byte in 0..buf.len() {
            for bit in [0u32, 3, 7] {
                let mut c = buf.clone();
                c[byte] ^= 1 << bit;
                let _ = parse_doc(&c);
            }
        }
    }

    #[test]
    fn random_garbage_never_panics() {
        let mut seed = 0xDEAD_BEEFu64;
        for len in [0usize, 1, 2, 7, 8, 33, 64, 129, 512] {
            for _ in 0..64 {
                let mut buf = Vec::with_capacity(len);
                for _ in 0..len {
                    seed = splitmix64(seed);
                    buf.push(seed as u8);
                }
                let _ = parse_doc(&buf);
                let _ = read_footer(&buf);
                let mut r = Reader::new(&buf);
                let _ = read_header(&mut r);
                let _ = read_framed(&mut r);
                let _ = r.u64_slice();
                let _ = r.u32_slice();
                let _ = r.str();
                let _ = r.varint();
                let _ = r.svarint();
            }
        }
    }

    #[test]
    fn reads_are_position_exact() {
        // Every primitive must consume exactly what its writer produced;
        // otherwise fields silently shear into one another.
        let mut w = Writer::new();
        let mut ends = Vec::new();
        w.u8(1);
        ends.push(w.pos());
        w.u16(2);
        ends.push(w.pos());
        w.u32(3);
        ends.push(w.pos());
        w.u64(4);
        ends.push(w.pos());
        w.i64(-5);
        ends.push(w.pos());
        w.f64(6.5);
        ends.push(w.pos());
        w.varint(1 << 40);
        ends.push(w.pos());
        w.svarint(-70_000);
        ends.push(w.pos());
        w.bytes(b"xyz");
        ends.push(w.pos());
        w.str("s");
        ends.push(w.pos());
        w.u64_slice(&[9, 10]);
        ends.push(w.pos());
        w.u32_slice(&[11]);
        ends.push(w.pos());
        let buf = w.finish();

        let mut r = Reader::new(&buf);
        let mut i = 0;
        let check = |r: &Reader, i: &mut usize| {
            assert_eq!(r.pos(), ends[*i], "field {i}");
            *i += 1;
        };
        r.u8().unwrap();
        check(&r, &mut i);
        r.u16().unwrap();
        check(&r, &mut i);
        r.u32().unwrap();
        check(&r, &mut i);
        r.u64().unwrap();
        check(&r, &mut i);
        r.i64().unwrap();
        check(&r, &mut i);
        r.f64().unwrap();
        check(&r, &mut i);
        r.varint().unwrap();
        check(&r, &mut i);
        r.svarint().unwrap();
        check(&r, &mut i);
        r.bytes().unwrap();
        check(&r, &mut i);
        r.str().unwrap();
        check(&r, &mut i);
        r.u64_slice().unwrap();
        check(&r, &mut i);
        r.u32_slice().unwrap();
        check(&r, &mut i);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn nested_frames_roundtrip() {
        // The part writer builds sections out of sections; framing has to
        // survive being applied to its own output.
        let mut inner = Writer::new();
        inner.str("column: url");
        inner.u64_slice(&[1, 2, 3]);
        let inner = inner.finish();

        let mut mid = Writer::new();
        write_framed(&mut mid, &inner);
        write_framed(&mut mid, b"second section");
        let mid = mid.finish();

        let mut outer = Writer::new();
        write_header(&mut outer);
        write_framed(&mut outer, &mid);
        let buf = outer.finish();

        let mut r = Reader::new(&buf);
        read_header(&mut r).unwrap();
        let mid_body = read_framed(&mut r).unwrap();
        let mut mr = Reader::new(mid_body);
        let first = read_framed(&mut mr).unwrap();
        assert_eq!(read_framed(&mut mr).unwrap(), b"second section");
        let mut ir = Reader::new(first);
        assert_eq!(ir.str().unwrap(), "column: url");
        assert_eq!(ir.u64_slice().unwrap(), vec![1, 2, 3]);
    }
}
