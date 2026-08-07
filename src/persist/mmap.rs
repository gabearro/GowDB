//! Read-only memory mapping, with the syscall declared by hand.
//!
//! ## Why map at all
//!
//! A part file is *already* in the shape the engine executes on: the packed
//! word arrays, the MPH fingerprint records and the dictionary offset tables
//! are all `u64` blocks written out verbatim. Reading a part with `read(2)`
//! into a `Vec` copies every one of those bytes through the page cache into a
//! second, private copy that then has to be re-faulted on first touch. Mapping
//! the file gives the same bytes with no copy and no allocation: a scan that
//! touches one granule of one column faults in one page instead of loading the
//! whole part, and two readers of the same part share the physical pages.
//!
//! [`Mmap::u64_slice`] is the piece that makes this worth doing. It hands out
//! a `&[u64]` pointing *into the mapping*, so [`PackedU64`](crate::encoding::PackedU64)
//! -style word arrays can be consumed where they lie rather than rebuilt.
//!
//! ## Why the FFI is hand-written
//!
//! The crate has zero dependencies on purpose (see the `Cargo.toml` note), and
//! that includes `libc`. `mmap`/`munmap` are two stable C symbols with a fixed
//! ABI; declaring them here costs a dozen lines and keeps the dependency count
//! at zero. The two flag constants we need (`PROT_READ`, `MAP_PRIVATE`) have
//! the same numeric values on every unix we target -- see the note on `sys`
//! below for where they were read off.
//!
//! ## The contract, which is not enforceable by the type system
//!
//! * **The mapping is read-only.** `PROT_READ | MAP_PRIVATE`: nothing written
//!   through this handle, and nothing that could ever reach the file.
//! * **The file must not be mutated or truncated while it is mapped.** Rust
//!   assumes the bytes behind a `&[u8]` are frozen for the life of the borrow;
//!   a concurrent write to the file breaks that, and a truncation turns a load
//!   from a mapped page into `SIGBUS`, which no `Result` can catch.
//!
//!   This is exactly why the module around it never modifies a file in place:
//!   parts are written to a temp name and `rename`d into position (rule 1 in
//!   [`crate::persist`]), so a published part file is immutable for as long as
//!   any reader can see it, and a replacement gets a fresh inode that the old
//!   mapping keeps no relationship with. *Unlinking* a mapped file is fine on
//!   unix -- the mapping holds the inode alive.
//!
//! ## Portability
//!
//! On unix this is a real mapping. Everywhere else the file is read into an
//! 8-byte-aligned heap buffer and the identical API is served out of that, so
//! no caller ever branches on the platform. The fallback is compiled (and
//! tested) on unix too, so it cannot rot.

use std::fs::File;
use std::path::Path;

use crate::common::{Error, Result};

/// The libc surface we need, declared by hand. Values verified against
/// `<sys/mman.h>` on macOS (xnu) and `<bits/mman-linux.h>` on glibc/musl:
/// `PROT_READ` is 0x1 and `MAP_PRIVATE` is 0x2 on both, and on every other
/// unix in practice, because they are inherited from 4.4BSD.
#[cfg(unix)]
mod sys {
    use std::ffi::{c_int, c_void};

    /// `off_t`. 64-bit on every 64-bit unix; on a 32-bit target the non-LFS
    /// `mmap` symbol takes a 32-bit `off_t`, and we would pass the wrong
    /// argument size. We only ever pass 0, but get the type right anyway.
    #[cfg(target_pointer_width = "64")]
    pub type OffT = i64;
    #[cfg(not(target_pointer_width = "64"))]
    pub type OffT = i32;

    pub const PROT_READ: c_int = 0x1;
    pub const MAP_PRIVATE: c_int = 0x2;

    /// `mmap` reports failure as `(void *) -1`, not as null -- null is a
    /// legitimate (if never actually returned) address.
    pub const MAP_FAILED: *mut c_void = !0usize as *mut c_void;

    extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: OffT,
        ) -> *mut c_void;

        pub fn munmap(addr: *mut c_void, len: usize) -> c_int;
    }
}

/// A read-only view of a whole file.
///
/// Hands out `&[u8]` and `&[u64]` borrowed from the mapping; nothing escapes
/// with a lifetime longer than the `Mmap`. Dropping it unmaps.
///
/// See the module docs for the two obligations the caller carries: the bytes
/// are read-only, and the file must not change underneath the mapping.
pub struct Mmap {
    /// Base of the bytes. Page-aligned when this is a real mapping, 8-byte
    /// aligned (and dangling) when `len == 0`, and `Vec<u64>`-aligned on the
    /// fallback path. Always at least 8-byte aligned -- [`Mmap::u64_slice`]
    /// re-checks rather than trusting that.
    ptr: *const u8,
    /// Bytes mapped: the file's length at [`Mmap::open`] time.
    len: usize,
    /// The heap buffer `ptr` points into on the portable fallback path, which
    /// owns the bytes instead of the kernel.
    ///
    /// This field exists on unix as well, where it is always `None`, so that
    /// the fallback compiles and can be exercised by the tests below on the
    /// platform we actually develop on. Two words per open file is not a cost
    /// worth a `#[cfg]` fork of the whole struct.
    owned: Option<Vec<u64>>,
}

impl Mmap {
    /// Map `path` read-only, for its whole length.
    ///
    /// A zero-length file yields a valid empty mapping: `mmap` rejects
    /// `len == 0` with `EINVAL`, and an empty part file is not an error, it is
    /// just empty.
    ///
    /// Errors are [`Error::Io`], including for a missing file, a directory or
    /// anything else that is not a regular file, and a failing `mmap` (whose
    /// `errno` is rendered into the message).
    pub fn open(path: &Path) -> Result<Mmap> {
        let f = File::open(path)
            .map_err(|e| Error::Io(format!("open {}: {e}", path.display())))?;
        let md = f
            .metadata()
            .map_err(|e| Error::Io(format!("stat {}: {e}", path.display())))?;
        // Refuse directories, fifos and devices up front. `File::open` happily
        // opens a directory on unix, and `mmap` would fail on it with a much
        // less obvious `ENODEV`; a fifo would report length 0 and silently
        // produce an empty "file".
        if !md.is_file() {
            return Err(Error::Io(format!(
                "mmap {}: not a regular file",
                path.display()
            )));
        }
        let len = usize::try_from(md.len()).map_err(|_| {
            Error::Io(format!(
                "mmap {}: {} bytes does not fit this address space",
                path.display(),
                md.len()
            ))
        })?;

        if len == 0 {
            return Ok(Mmap::empty());
        }
        Mmap::load(f, len, path)
    }

    /// A valid, borrowable mapping of nothing.
    fn empty() -> Mmap {
        // A dangling-but-aligned pointer: `from_raw_parts` requires non-null
        // and correctly aligned even at length 0.
        let ptr = std::ptr::NonNull::<u64>::dangling().as_ptr() as *const u8;
        Mmap { ptr, len: 0, owned: None }
    }

    /// Establish the mapping proper. `len > 0` is guaranteed by [`Mmap::open`].
    #[cfg(unix)]
    fn load(f: File, len: usize, path: &Path) -> Result<Mmap> {
        use std::os::unix::io::AsRawFd;
        debug_assert!(len > 0, "mmap(len = 0) is EINVAL");
        // SAFETY: `f` is open for reading and stays alive across the call. A
        // null hint lets the kernel choose the address; the result is either a
        // fresh `len`-byte region we now own outright, or MAP_FAILED.
        let p = unsafe {
            sys::mmap(
                std::ptr::null_mut(),
                len,
                sys::PROT_READ,
                sys::MAP_PRIVATE,
                f.as_raw_fd(),
                0,
            )
        };
        if p == sys::MAP_FAILED {
            // Read errno first: anything at all may clobber it.
            let e = std::io::Error::last_os_error();
            return Err(Error::Io(format!("mmap {} ({len} bytes): {e}", path.display())));
        }
        debug_assert!(!p.is_null());
        // The mapping holds its own reference to the file, so the descriptor
        // is dead weight from here on; `f` closes on return.
        Ok(Mmap { ptr: p as *const u8, len, owned: None })
    }

    /// Portable stand-in for `mmap` where there is no `mmap`.
    #[cfg(not(unix))]
    fn load(f: File, len: usize, path: &Path) -> Result<Mmap> {
        Mmap::slurp(f, len, path)
    }

    /// The portable fallback: read the whole file into an 8-byte-aligned heap
    /// buffer and serve the same API from it.
    ///
    /// The buffer is a `Vec<u64>`, not a `Vec<u8>`, precisely so that
    /// [`Mmap::u64_slice`] keeps working -- the global allocator only promises
    /// 1-byte alignment for a byte vector, and an unaligned `&[u64]` is
    /// undefined behaviour, not merely slow.
    #[cfg(any(not(unix), test))]
    fn slurp(mut f: File, len: usize, path: &Path) -> Result<Mmap> {
        use std::io::Read;
        debug_assert!(len > 0);
        let mut words = vec![0u64; len.div_ceil(8)];
        {
            // SAFETY: `words` owns `words.len() * 8 >= len` initialized bytes,
            // and `u64` has no padding and no invalid bit patterns, so viewing
            // its storage as writable bytes is sound. The borrow ends here.
            let buf = unsafe {
                std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, len)
            };
            f.read_exact(buf)
                .map_err(|e| Error::Io(format!("read {}: {e}", path.display())))?;
        }
        let ptr = words.as_ptr() as *const u8;
        Ok(Mmap { ptr, len, owned: Some(words) })
    }

    /// The mapped bytes.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` is valid for `len` read-only bytes for as long as
        // `self` lives (kernel mapping, or the `Vec` in `owned`), it is
        // non-null and byte alignment is trivial, and `len` came from a
        // successful `mmap`/allocation so it cannot exceed `isize::MAX`. The
        // returned borrow is tied to `&self`, so it cannot outlive the unmap.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Length of the mapping in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the mapped file was empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// A `&[u64]` view of `bytes[off .. off + count * 8]`, with no copy.
    ///
    /// This is the safety boundary of the whole module: everything downstream
    /// (packed column words, MPH records) reads these words through
    /// `get_unchecked` on the assumption that this function proved the range
    /// exists. So it validates hard, in this order:
    ///
    ///   * `count * 8` must not overflow `usize`;
    ///   * `off + count * 8` must not overflow, and must be `<= len`;
    ///   * `off` must be a multiple of 8. An unaligned `&[u64]` is undefined
    ///     behaviour -- not "slow on x86", *undefined* -- so a header claiming
    ///     an odd offset is rejected, never fixed up.
    ///
    /// The words are in **native byte order**: they are the file's bytes
    /// reinterpreted, not decoded. That is the point (zero copy), and it means
    /// a file written with `to_le_bytes` is only readable this way on a
    /// little-endian machine, which is the only kind this format targets.
    ///
    /// Violations are [`Error::Corruption`] rather than a panic or an
    /// `Error::Io`: every offset that reaches here comes from a length or
    /// offset field read out of the file itself, so a violation means the file
    /// is lying about its own layout, which is what `Corruption` means in
    /// [`crate::persist`].
    pub fn u64_slice(&self, off: usize, count: usize) -> Result<&[u64]> {
        let span = count.checked_mul(8).ok_or_else(|| {
            Error::corruption(format!("u64_slice: count {count} overflows a byte length"))
        })?;
        let end = off.checked_add(span).ok_or_else(|| {
            Error::corruption(format!("u64_slice: offset {off} + {span} bytes overflows"))
        })?;
        if end > self.len {
            return Err(Error::corruption(format!(
                "u64_slice: range {off}..{end} exceeds the {}-byte mapping",
                self.len
            )));
        }
        if off % 8 != 0 {
            return Err(Error::corruption(format!(
                "u64_slice: offset {off} is not 8-byte aligned"
            )));
        }
        // Unreachable: a mapping base is page-aligned, the fallback base is
        // `Vec<u64>`-aligned and the empty base is a dangling `u64`. Checked
        // anyway, because it is the other half of what makes the cast below
        // provably sound and it costs one AND.
        if (self.ptr as usize) % 8 != 0 {
            return Err(Error::corruption(format!(
                "u64_slice: mapping base {:p} is not 8-byte aligned",
                self.ptr
            )));
        }
        // SAFETY: `off <= end <= len`, so the offset is in bounds (or one past
        // the end, which `add` permits); the address is 8-byte aligned by the
        // two checks above; `count * 8` bytes from there are inside the
        // mapping and stay initialized and immutable for the borrow; and
        // `span <= len <= isize::MAX`.
        let p = unsafe { self.ptr.add(off) } as *const u64;
        Ok(unsafe { std::slice::from_raw_parts(p, count) })
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        // `owned` means the bytes are a heap buffer that drops itself right
        // after this; `len == 0` means we never called `mmap` at all.
        #[cfg(unix)]
        {
            if self.owned.is_none() && self.len > 0 {
                // SAFETY: `ptr`/`len` are exactly the address and length
                // returned by a successful `mmap` in `load`, unmapped exactly
                // once -- `Mmap` is not `Clone` and `drop` runs once.
                let rc = unsafe { sys::munmap(self.ptr as *mut _, self.len) };
                debug_assert_eq!(rc, 0, "munmap failed: {}", std::io::Error::last_os_error());
                let _ = rc;
            }
        }
    }
}

// SAFETY (both): a `Mmap` is an immutable, read-only region plus its length.
// There is no interior mutability, no `&mut` path to the bytes and no way to
// obtain anything but a shared borrow tied to `&self`, so every thread that
// can reach it can only read -- which is why sharing (`Sync`) is sound. The
// mapping is not tied to the thread that created it either: `munmap` is
// process-wide and takes no thread-local state, so moving the handle to
// another thread and unmapping there is fine -- which is why sending (`Send`)
// is sound. Only the raw pointer field suppresses the automatic impls.
//
// This rests on the module-level contract that the file is not mutated while
// mapped; a writer racing the mapping would be a data race no matter what
// these impls say.
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl std::fmt::Debug for Mmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mmap")
            .field("len", &self.len)
            .field("kind", &if self.owned.is_some() { "heap" } else { "mapped" })
            .finish()
    }
}

#[cfg(test)]
mod tests {
    //! Temp directories are named from the process id plus a counter rather
    //! than from randomness, so a rerun cannot collide with a live process and
    //! a name is reproducible while debugging.

    use super::*;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    static SEQ: AtomicU64 = AtomicU64::new(0);

    /// A temp directory that deletes itself, including on panic.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Scratch {
            let n = SEQ.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir()
                .join(format!("granular-mmap-{}-{tag}-{n}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).expect("create scratch dir");
            Scratch(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
        fn join(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
        /// Write `bytes` to `name` and return its path.
        fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let p = self.join(name);
            let mut f = std::fs::File::create(&p).expect("create file");
            f.write_all(bytes).expect("write file");
            f.flush().expect("flush file");
            p
        }
        /// Write `vals` as native-endian words and return the path.
        fn words(&self, name: &str, vals: &[u64]) -> PathBuf {
            self.file(name, &ne_bytes(vals))
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Native order, so the tests assert the documented reinterpretation
    /// rather than a particular endianness.
    fn ne_bytes(vals: &[u64]) -> Vec<u8> {
        let mut v = Vec::with_capacity(vals.len() * 8);
        for x in vals {
            v.extend_from_slice(&x.to_ne_bytes());
        }
        v
    }

    fn is_io(e: &Error) -> bool {
        matches!(e, Error::Io(_))
    }

    fn is_corruption(e: &Error) -> bool {
        matches!(e, Error::Corruption(_))
    }

    // ---- basic mapping -------------------------------------------------

    #[test]
    fn maps_a_file_and_reads_it_back() {
        let s = Scratch::new("roundtrip");
        let data = b"the quick brown fox jumps over the lazy dog";
        let p = s.file("f", data);
        let m = Mmap::open(&p).unwrap();
        assert_eq!(m.len(), data.len());
        assert!(!m.is_empty());
        assert_eq!(m.as_slice(), data);
    }

    #[test]
    fn length_matches_the_file_for_many_sizes() {
        let s = Scratch::new("sizes");
        for &n in &[1usize, 2, 7, 8, 9, 63, 64, 65, 4095, 4096, 4097, 100_000] {
            let bytes: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
            let p = s.file(&format!("f{n}"), &bytes);
            let m = Mmap::open(&p).unwrap();
            assert_eq!(m.len(), n, "len for {n} bytes");
            assert_eq!(m.as_slice(), &bytes[..], "contents for {n} bytes");
            assert!(!m.is_empty());
        }
    }

    #[test]
    fn empty_file_yields_a_valid_empty_mapping() {
        // The whole point: `mmap` with len 0 is EINVAL, so `open` must not
        // make the call at all, yet must still return a usable handle.
        let s = Scratch::new("empty");
        let p = s.file("f", b"");
        let m = Mmap::open(&p).unwrap();
        assert_eq!(m.len(), 0);
        assert!(m.is_empty());
        assert!(m.as_slice().is_empty());
        assert_eq!(m.u64_slice(0, 0).unwrap(), &[] as &[u64]);
    }

    #[test]
    fn empty_mapping_rejects_any_nonzero_read() {
        let s = Scratch::new("empty-oob");
        let p = s.file("f", b"");
        let m = Mmap::open(&p).unwrap();
        assert!(is_corruption(&m.u64_slice(0, 1).unwrap_err()));
        assert!(is_corruption(&m.u64_slice(8, 0).unwrap_err()));
    }

    #[test]
    fn as_slice_is_stable_across_calls() {
        let s = Scratch::new("stable");
        let p = s.file("f", b"abcdefgh");
        let m = Mmap::open(&p).unwrap();
        assert_eq!(m.as_slice().as_ptr(), m.as_slice().as_ptr());
        assert_eq!(m.as_slice(), b"abcdefgh");
    }

    #[test]
    fn one_byte_file_is_not_empty() {
        let s = Scratch::new("onebyte");
        let p = s.file("f", b"\x7f");
        let m = Mmap::open(&p).unwrap();
        assert!(!m.is_empty());
        assert_eq!(m.len(), 1);
        assert_eq!(m.as_slice()[0], 0x7f);
    }

    // ---- u64_slice: the safety boundary --------------------------------

    #[test]
    fn u64_slice_reads_the_whole_file() {
        let s = Scratch::new("u64-all");
        let vals: Vec<u64> = (0..64).map(|i| i * 0x0101_0101_0101_0101).collect();
        let p = s.words("f", &vals);
        let m = Mmap::open(&p).unwrap();
        assert_eq!(m.u64_slice(0, vals.len()).unwrap(), &vals[..]);
    }

    #[test]
    fn u64_slice_reads_an_interior_window() {
        let s = Scratch::new("u64-window");
        let vals: Vec<u64> = (100..164).collect();
        let p = s.words("f", &vals);
        let m = Mmap::open(&p).unwrap();
        // words 8..24
        assert_eq!(m.u64_slice(64, 16).unwrap(), &vals[8..24]);
        // the final word alone
        assert_eq!(m.u64_slice((vals.len() - 1) * 8, 1).unwrap(), &vals[63..]);
    }

    #[test]
    fn u64_slice_agrees_with_the_byte_view() {
        let s = Scratch::new("u64-vs-bytes");
        let vals: Vec<u64> = (0..32).map(|i| 0xdead_0000_0000_0000u64 ^ i).collect();
        let p = s.words("f", &vals);
        let m = Mmap::open(&p).unwrap();
        let words = m.u64_slice(0, 32).unwrap();
        for (i, w) in words.iter().enumerate() {
            let mut b = [0u8; 8];
            b.copy_from_slice(&m.as_slice()[i * 8..i * 8 + 8]);
            assert_eq!(*w, u64::from_ne_bytes(b), "word {i}");
        }
    }

    #[test]
    fn u64_slice_returns_an_aligned_pointer() {
        let s = Scratch::new("u64-align");
        let vals: Vec<u64> = (0..16).collect();
        let p = s.words("f", &vals);
        let m = Mmap::open(&p).unwrap();
        for off in (0..128).step_by(8) {
            let w = m.u64_slice(off, 1).unwrap();
            assert_eq!(w.as_ptr() as usize % 8, 0, "offset {off}");
        }
    }

    #[test]
    fn u64_slice_rejects_out_of_range_offset() {
        let s = Scratch::new("u64-off-oob");
        let p = s.words("f", &[1, 2, 3, 4]);
        let m = Mmap::open(&p).unwrap();
        let e = m.u64_slice(32, 1).unwrap_err();
        assert!(is_corruption(&e), "{e}");
        assert!(m.u64_slice(40, 0).is_err(), "offset past the end, even empty");
    }

    #[test]
    fn u64_slice_rejects_out_of_range_count() {
        let s = Scratch::new("u64-count-oob");
        let p = s.words("f", &[1, 2, 3, 4]);
        let m = Mmap::open(&p).unwrap();
        assert!(is_corruption(&m.u64_slice(0, 5).unwrap_err()));
        assert!(is_corruption(&m.u64_slice(8, 4).unwrap_err()));
        // one word too many from a legal start
        assert!(is_corruption(&m.u64_slice(24, 2).unwrap_err()));
        assert!(m.u64_slice(24, 1).is_ok());
    }

    #[test]
    fn u64_slice_rejects_unaligned_offset() {
        let s = Scratch::new("u64-unaligned");
        let p = s.words("f", &[7, 8, 9, 10]);
        let m = Mmap::open(&p).unwrap();
        for off in [1usize, 2, 3, 4, 5, 6, 7, 9, 15, 17] {
            let e = m.u64_slice(off, 1).unwrap_err();
            assert!(is_corruption(&e), "offset {off}: {e}");
            assert!(format!("{e}").contains("align"), "offset {off}: {e}");
        }
    }

    #[test]
    fn u64_slice_checks_bounds_before_alignment() {
        // Both are wrong here; either error is a rejection, but it must not
        // be a panic and must not be a success.
        let s = Scratch::new("u64-both-wrong");
        let p = s.words("f", &[1]);
        let m = Mmap::open(&p).unwrap();
        assert!(m.u64_slice(5, 9).is_err());
    }

    #[test]
    fn u64_slice_rejects_overflowing_count() {
        let s = Scratch::new("u64-overflow-count");
        let p = s.words("f", &[1, 2]);
        let m = Mmap::open(&p).unwrap();
        // count * 8 overflows usize
        let e = m.u64_slice(0, usize::MAX / 4).unwrap_err();
        assert!(is_corruption(&e), "{e}");
        assert!(m.u64_slice(0, usize::MAX).is_err());
    }

    #[test]
    fn u64_slice_rejects_overflowing_offset() {
        let s = Scratch::new("u64-overflow-off");
        let p = s.words("f", &[1, 2]);
        let m = Mmap::open(&p).unwrap();
        // off + span overflows usize (off is 8-aligned, so alignment is not
        // what saves us here -- the checked_add is)
        let e = m.u64_slice(usize::MAX - 7, 1).unwrap_err();
        assert!(is_corruption(&e), "{e}");
        assert!(m.u64_slice(usize::MAX, 1).is_err());
    }

    #[test]
    fn u64_slice_allows_zero_count_at_any_aligned_offset_in_range() {
        let s = Scratch::new("u64-zero-count");
        let vals: Vec<u64> = (0..4).collect();
        let p = s.words("f", &vals);
        let m = Mmap::open(&p).unwrap();
        for off in (0..=32).step_by(8) {
            assert!(m.u64_slice(off, 0).unwrap().is_empty(), "offset {off}");
        }
        // ... but not at an unaligned or out-of-range one
        assert!(m.u64_slice(4, 0).is_err());
        assert!(m.u64_slice(40, 0).is_err());
    }

    #[test]
    fn u64_slice_reads_the_exact_tail() {
        let s = Scratch::new("u64-tail");
        let vals: Vec<u64> = (0..10).map(|i| i * 3 + 1).collect();
        let p = s.words("f", &vals);
        let m = Mmap::open(&p).unwrap();
        assert_eq!(m.u64_slice(72, 1).unwrap(), &[28]);
        assert_eq!(m.u64_slice(0, 10).unwrap().last(), Some(&28));
    }

    #[test]
    fn u64_slice_respects_a_ragged_file_length() {
        // 12 bytes: one whole word plus half of another. The half word is not
        // readable, however aligned the offset is.
        let s = Scratch::new("u64-ragged");
        let mut bytes = ne_bytes(&[0xabcd_ef01_2345_6789]);
        bytes.extend_from_slice(&[1, 2, 3, 4]);
        let p = s.file("f", &bytes);
        let m = Mmap::open(&p).unwrap();
        assert_eq!(m.len(), 12);
        assert_eq!(m.u64_slice(0, 1).unwrap(), &[0xabcd_ef01_2345_6789]);
        assert!(m.u64_slice(0, 2).is_err());
        assert!(m.u64_slice(8, 1).is_err());
        assert!(m.u64_slice(8, 0).is_ok());
    }

    // ---- failure modes -------------------------------------------------

    #[test]
    fn missing_file_is_an_io_error_not_a_panic() {
        let s = Scratch::new("missing");
        let e = Mmap::open(&s.join("nope")).unwrap_err();
        assert!(is_io(&e), "{e}");
        assert_eq!(e.code(), "IO_ERROR");
        assert!(format!("{e}").contains("nope"), "{e}");
    }

    #[test]
    fn missing_directory_is_an_io_error() {
        let e = Mmap::open(Path::new("/definitely/not/here/granular-mmap-test")).unwrap_err();
        assert!(is_io(&e), "{e}");
    }

    #[test]
    fn mapping_a_directory_is_an_io_error() {
        // `File::open` succeeds on a directory on unix, so this has to be
        // caught by the is_file() check rather than by `mmap`'s ENODEV.
        let s = Scratch::new("dir");
        let e = Mmap::open(s.path()).unwrap_err();
        assert!(is_io(&e), "{e}");
        assert!(format!("{e}").contains("regular file"), "{e}");
    }

    // ---- lifetime and ownership ----------------------------------------

    #[test]
    fn mapping_outlives_the_scope_that_opened_it() {
        let s = Scratch::new("escape");
        let vals: Vec<u64> = (0..256).collect();
        let p = s.words("f", &vals);
        // The `File` inside `open` is closed by the time this returns; the
        // mapping must not care.
        let m = {
            let inner = Mmap::open(&p).unwrap();
            inner
        };
        assert_eq!(m.u64_slice(0, 256).unwrap(), &vals[..]);
    }

    #[test]
    fn mapping_survives_being_moved_into_a_thread() {
        let s = Scratch::new("thread-move");
        let vals: Vec<u64> = (0..4096).collect();
        let p = s.words("f", &vals);
        let m = Mmap::open(&p).unwrap();
        let h = std::thread::spawn(move || {
            // The mapping is unmapped by this thread, not the one that made it.
            let w = m.u64_slice(0, 4096).unwrap();
            (w.iter().copied().sum::<u64>(), m.len())
        });
        let (sum, len) = h.join().unwrap();
        assert_eq!(sum, (0..4096u64).sum::<u64>());
        assert_eq!(len, 4096 * 8);
    }

    #[test]
    fn mapping_can_be_shared_by_reference_across_threads() {
        let s = Scratch::new("thread-share");
        let vals: Vec<u64> = (0..2048).map(|i| i * 7).collect();
        let p = s.words("f", &vals);
        let m = Mmap::open(&p).unwrap();
        std::thread::scope(|sc| {
            for t in 0..4usize {
                let m = &m;
                let vals = &vals;
                sc.spawn(move || {
                    for i in (t..2048).step_by(4) {
                        assert_eq!(m.u64_slice(i * 8, 1).unwrap()[0], vals[i]);
                    }
                    assert_eq!(m.as_slice().len(), vals.len() * 8);
                });
            }
        });
        assert_eq!(m.u64_slice(0, 2048).unwrap(), &vals[..]);
    }

    #[test]
    fn mapping_can_be_shared_through_an_arc() {
        let s = Scratch::new("arc-share");
        let vals: Vec<u64> = (0..1024).map(|i| i ^ 0x5555).collect();
        let p = s.words("f", &vals);
        let m = Arc::new(Mmap::open(&p).unwrap());
        let hs: Vec<_> = (0..4)
            .map(|_| {
                let m = Arc::clone(&m);
                std::thread::spawn(move || m.u64_slice(0, 1024).unwrap().iter().sum::<u64>())
            })
            .collect();
        let want: u64 = vals.iter().sum();
        for h in hs {
            assert_eq!(h.join().unwrap(), want);
        }
        // Last reference drops here, on the main thread.
        drop(m);
    }

    #[test]
    fn two_mappings_of_one_file_are_independent() {
        let s = Scratch::new("two-maps");
        let vals: Vec<u64> = (0..128).collect();
        let p = s.words("f", &vals);
        let a = Mmap::open(&p).unwrap();
        let b = Mmap::open(&p).unwrap();
        assert_eq!(a.as_slice(), b.as_slice());
        drop(a);
        // Unmapping one must not disturb the other.
        assert_eq!(b.u64_slice(0, 128).unwrap(), &vals[..]);
    }

    #[test]
    fn many_open_and_drop_cycles_do_not_leak() {
        // Crude but effective: a leaked descriptor or mapping per iteration
        // hits RLIMIT_NOFILE / vm.max_map_count long before 400.
        let s = Scratch::new("nofd-leak");
        let p = s.words("f", &(0..1024).collect::<Vec<u64>>());
        for _ in 0..400 {
            let m = Mmap::open(&p).unwrap();
            assert_eq!(m.u64_slice(0, 1).unwrap()[0], 0);
        }
        let m = Mmap::open(&p).unwrap();
        assert_eq!(m.len(), 8192);
    }

    #[test]
    fn mapping_survives_the_file_being_unlinked() {
        // The `persist` cleanup path removes superseded part files while
        // readers may still hold them open.
        let s = Scratch::new("unlink");
        let vals: Vec<u64> = (0..512).map(|i| i * 11).collect();
        let p = s.words("f", &vals);
        let m = Mmap::open(&p).unwrap();
        std::fs::remove_file(&p).unwrap();
        assert!(Mmap::open(&p).is_err(), "the name is gone");
        assert_eq!(m.u64_slice(0, 512).unwrap(), &vals[..]);
    }

    #[test]
    fn mapping_keeps_the_bytes_it_was_opened_on_across_a_rename() {
        // Rule 1 of `persist`: files are replaced by rename, never edited in
        // place. An open mapping must therefore keep seeing the old inode,
        // which is what makes mapping safe at all.
        let s = Scratch::new("rename");
        let old: Vec<u64> = (0..64).collect();
        let new: Vec<u64> = (0..64).map(|i| i + 1000).collect();
        let p = s.words("f", &old);
        let m = Mmap::open(&p).unwrap();
        let tmp = s.words("f.tmp", &new);
        std::fs::rename(&tmp, &p).unwrap();
        assert_eq!(m.u64_slice(0, 64).unwrap(), &old[..], "stale mapping must be stable");
        assert_eq!(Mmap::open(&p).unwrap().u64_slice(0, 64).unwrap(), &new[..]);
    }

    // ---- volume ---------------------------------------------------------

    #[test]
    fn large_file_roundtrips() {
        // 4 MiB, well past a page and past any readahead window.
        const N: usize = 512 * 1024;
        let s = Scratch::new("large");
        let vals: Vec<u64> = (0..N as u64).map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15)).collect();
        let p = s.words("f", &vals);
        let m = Mmap::open(&p).unwrap();
        assert_eq!(m.len(), N * 8);
        let w = m.u64_slice(0, N).unwrap();
        assert_eq!(w.len(), N);
        assert_eq!(w, &vals[..]);
        // touch the extremes and a few interior pages explicitly
        assert_eq!(w[0], vals[0]);
        assert_eq!(w[N - 1], vals[N - 1]);
        for off in (0..N).step_by(4096) {
            assert_eq!(m.u64_slice(off * 8, 1).unwrap()[0], vals[off], "word {off}");
        }
        assert!(m.u64_slice(0, N + 1).is_err());
    }

    #[test]
    fn large_file_byte_view_matches() {
        const N: usize = 3 * 1024 * 1024 + 5; // deliberately not a page multiple
        let s = Scratch::new("large-bytes");
        let bytes: Vec<u8> = (0..N).map(|i| (i.wrapping_mul(31) % 256) as u8).collect();
        let p = s.file("f", &bytes);
        let m = Mmap::open(&p).unwrap();
        assert_eq!(m.len(), N);
        assert_eq!(m.as_slice(), &bytes[..]);
        // the ragged tail is not word-readable
        assert!(m.u64_slice((N / 8) * 8, 1).is_err());
    }

    // ---- the portable fallback -----------------------------------------
    //
    // Compiled and run here as well as on non-unix, so the branch that only
    // ships on other platforms cannot silently rot.

    fn slurped(path: &Path) -> Mmap {
        let f = File::open(path).unwrap();
        let len = f.metadata().unwrap().len() as usize;
        Mmap::slurp(f, len, path).unwrap()
    }

    #[test]
    fn fallback_reads_the_same_bytes_as_the_mapping() {
        let s = Scratch::new("fallback");
        let vals: Vec<u64> = (0..300).map(|i| i * 5 + 2).collect();
        let p = s.words("f", &vals);
        let a = Mmap::open(&p).unwrap();
        let b = slurped(&p);
        assert_eq!(a.as_slice(), b.as_slice());
        assert_eq!(a.len(), b.len());
        assert_eq!(b.u64_slice(0, 300).unwrap(), &vals[..]);
    }

    #[test]
    fn fallback_buffer_is_word_aligned() {
        // The reason the backing store is a `Vec<u64>`: a `Vec<u8>` would be
        // 1-byte aligned and `u64_slice` would be undefined behaviour.
        let s = Scratch::new("fallback-align");
        for n in 1..40usize {
            let bytes: Vec<u8> = (0..n).map(|i| i as u8).collect();
            let p = s.file(&format!("f{n}"), &bytes);
            let m = slurped(&p);
            assert_eq!(m.as_slice().as_ptr() as usize % 8, 0, "len {n}");
            assert_eq!(m.as_slice(), &bytes[..]);
        }
    }

    #[test]
    fn fallback_validates_exactly_like_the_mapping() {
        let s = Scratch::new("fallback-validate");
        let p = s.words("f", &[10, 20, 30, 40]);
        let m = slurped(&p);
        assert_eq!(m.u64_slice(8, 2).unwrap(), &[20, 30]);
        assert!(m.u64_slice(4, 1).is_err(), "unaligned");
        assert!(m.u64_slice(0, 5).is_err(), "count out of range");
        assert!(m.u64_slice(32, 1).is_err(), "offset out of range");
        assert!(m.u64_slice(0, usize::MAX / 4).is_err(), "overflow");
    }

    #[test]
    fn fallback_survives_a_move_across_threads() {
        // `ptr` points into the owned `Vec`; moving the handle moves the
        // `Vec` struct but not its heap buffer, so the pointer stays good.
        let s = Scratch::new("fallback-thread");
        let vals: Vec<u64> = (0..777).collect();
        let p = s.words("f", &vals);
        let m = slurped(&p);
        let m = std::thread::spawn(move || {
            assert_eq!(m.u64_slice(0, 777).unwrap(), &(0..777).collect::<Vec<u64>>()[..]);
            m
        })
        .join()
        .unwrap();
        assert_eq!(m.u64_slice(776 * 8, 1).unwrap(), &[776]);
    }

    #[test]
    fn debug_says_how_the_bytes_are_held() {
        let s = Scratch::new("debug");
        let p = s.words("f", &[1, 2]);
        let m = Mmap::open(&p).unwrap();
        let d = format!("{m:?}");
        assert!(d.contains("16"), "{d}");
        #[cfg(unix)]
        assert!(d.contains("mapped"), "{d}");
        assert!(format!("{:?}", slurped(&p)).contains("heap"));
    }

    #[test]
    fn empty_mapping_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Mmap>();
        let s = Scratch::new("empty-send");
        let p = s.file("f", b"");
        let m = Mmap::open(&p).unwrap();
        // A dangling base must still be safe to hand to another thread.
        std::thread::spawn(move || {
            assert!(m.as_slice().is_empty());
            assert!(m.u64_slice(0, 0).unwrap().is_empty());
        })
        .join()
        .unwrap();
    }

    // =====================================================================
    // ADVERSARIAL REVIEW -- added to demonstrate suspected defects.
    // =====================================================================

    #[cfg(unix)]
    mod probe {
        use std::ffi::{c_int, c_void};
        extern "C" {
            pub fn msync(addr: *mut c_void, len: usize, flags: c_int) -> c_int;
        }
        pub const MS_ASYNC: c_int = 1;
        /// True when `[addr, addr+len)` is still mapped in this process.
        pub fn is_mapped(addr: *const u8, len: usize) -> bool {
            unsafe { msync(addr as *mut c_void, len, MS_ASYNC) == 0 }
        }
    }

    /// Does `drop` actually release the range?
    #[cfg(unix)]
    #[test]
    fn adv_drop_really_unmaps() {
        let s = Scratch::new("adv-unmap");
        // Big enough that the hole is unlikely to be instantly reused by a
        // parallel test thread.
        let vals: Vec<u64> = (0..2 * 1024 * 1024u64).collect();
        let p = s.words("f", &vals);
        let m = Mmap::open(&p).unwrap();
        let (ptr, len) = (m.ptr, m.len);
        assert!(probe::is_mapped(ptr, len), "should be mapped while alive");
        drop(m);
        assert!(!probe::is_mapped(ptr, len), "drop did NOT munmap");
    }

    /// The heap fallback must never be handed to `munmap`.
    #[cfg(unix)]
    #[test]
    fn adv_drop_does_not_unmap_the_heap_fallback() {
        let s = Scratch::new("adv-unmap-heap");
        let vals: Vec<u64> = (0..64 * 1024u64).collect();
        let p = s.words("f", &vals);
        for _ in 0..64 {
            let m = slurped(&p);
            assert_eq!(m.u64_slice(0, 1).unwrap()[0], 0);
            drop(m);
            // If drop had munmap'd the heap, this would fault or corrupt.
            let v: Vec<u64> = (0..64 * 1024u64).collect();
            assert_eq!(v.len(), 64 * 1024);
        }
    }

    /// Exhaustive sweep of `u64_slice` over many file lengths: no panic, no
    /// out-of-range success, contents always agree with the byte view.
    #[test]
    fn adv_u64_slice_sweep_is_consistent_with_as_slice() {
        let s = Scratch::new("adv-sweep");
        for &n in &[0usize, 1, 7, 8, 9, 15, 16, 17, 63, 64, 4095, 4096, 4097] {
            let bytes: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
            let p = s.file(&format!("f{n}"), &bytes);
            for m in [Mmap::open(&p).unwrap()]
                .into_iter()
                .chain(if n > 0 { Some(slurped(&p)) } else { None })
            {
                for off in 0..(n + 24) {
                    for count in 0..(n / 8 + 4) {
                        match m.u64_slice(off, count) {
                            Ok(w) => {
                                assert_eq!(off % 8, 0, "n={n} off={off}");
                                assert!(off + count * 8 <= n, "n={n} off={off} c={count}");
                                assert_eq!(w.len(), count);
                                assert_eq!(w.as_ptr() as usize % align_of::<u64>(), 0);
                                for (i, x) in w.iter().enumerate() {
                                    let mut b = [0u8; 8];
                                    b.copy_from_slice(
                                        &m.as_slice()[off + i * 8..off + i * 8 + 8],
                                    );
                                    assert_eq!(*x, u64::from_ne_bytes(b), "n={n} off={off}");
                                }
                            }
                            Err(e) => assert!(is_corruption(&e), "n={n} off={off}: {e}"),
                        }
                    }
                }
                // extreme arguments must reject, not wrap or panic
                for &(o, c) in &[
                    (usize::MAX, usize::MAX),
                    (usize::MAX, 1),
                    (usize::MAX - 7, 1),
                    (0, usize::MAX),
                    (0, usize::MAX / 8),
                    (0, usize::MAX / 8 + 1),
                    (8, usize::MAX / 8),
                ] {
                    assert!(m.u64_slice(o, c).is_err(), "n={n} ({o},{c}) accepted");
                }
            }
        }
    }

    /// The base-alignment gate is hard-coded to 8, but the alignment the cast
    /// actually needs (and the alignment `empty()` / `Vec<u64>` deliver) is
    /// `align_of::<u64>()`, which is **4** on i686 / 32-bit x86 and ARM EABI.
    /// There, `NonNull::<u64>::dangling()` is 4 and this gate rejects every
    /// empty mapping. Synthesised here because this host has align 8.
    #[test]
    fn adv_base_alignment_gate_rejects_a_legal_align_of_u64_base() {
        assert_eq!(align_of::<u64>(), 8, "this host: gate happens to agree");
        // What `Mmap::empty()` produces where align_of::<u64>() == 4:
        let m = Mmap { ptr: 4 as *const u8, len: 0, owned: None };
        let r = m.u64_slice(0, 0);
        // `from_raw_parts::<u64>(4, 0)` is perfectly sound on such a target,
        // yet the module refuses it.
        let e = r.unwrap_err();
        assert!(format!("{e}").contains("mapping base"), "{e}");
    }

    /// `Mmap::open`'s doc claims fifos are refused up front. `File::open` on a
    /// fifo blocks until a writer appears, so the `is_file()` gate is never
    /// reached and the call hangs instead of erroring.
    #[cfg(unix)]
    #[test]
    fn adv_open_on_a_fifo_hangs_instead_of_erroring() {
        use std::ffi::CString;
        extern "C" {
            fn mkfifo(path: *const std::ffi::c_char, mode: u32) -> std::ffi::c_int;
        }
        let s = Scratch::new("adv-fifo");
        let p = s.join("pipe");
        let c = CString::new(p.to_str().unwrap()).unwrap();
        let rc = unsafe { mkfifo(c.as_ptr(), 0o644) };
        assert_eq!(rc, 0, "mkfifo failed: {}", std::io::Error::last_os_error());
        assert!(p.exists());

        let (tx, rx) = std::sync::mpsc::channel();
        let p2 = p.clone();
        // Detached on purpose: it never returns, so it must not be joined.
        std::thread::spawn(move || {
            let r = Mmap::open(&p2);
            let _ = tx.send(r.is_err());
        });
        match rx.recv_timeout(std::time::Duration::from_secs(3)) {
            Ok(_) => panic!("open returned -- the doc's claim holds after all"),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Confirmed: stuck. Prove *where* by supplying the writer end
                // the blocked O_RDONLY open is waiting for -- that releases it
                // and only then does the is_file() gate run.
                let w = std::fs::OpenOptions::new().write(true).open(&p).unwrap();
                let got = rx
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .expect("open must finish once a writer exists");
                assert!(got, "and only now is it refused as 'not a regular file'");
                drop(w);
            }
            Err(e) => panic!("{e}"),
        }
    }

    /// Character/block devices and directories: are they really refused?
    #[cfg(unix)]
    #[test]
    fn adv_devices_are_refused() {
        for d in ["/dev/zero", "/dev/null"] {
            let p = Path::new(d);
            if !p.exists() {
                continue;
            }
            let e = Mmap::open(p).unwrap_err();
            assert!(is_io(&e), "{d}: {e}");
            assert!(format!("{e}").contains("regular file"), "{d}: {e}");
        }
    }

    /// A symlink pointing at a directory must not sneak past `is_file()`.
    #[cfg(unix)]
    #[test]
    fn adv_symlink_to_dir_is_refused() {
        let s = Scratch::new("adv-symlink");
        let link = s.join("l");
        std::os::unix::fs::symlink(s.path(), &link).unwrap();
        assert!(Mmap::open(&link).is_err());
    }

    /// `slurp`'s `len > 0` precondition is only a `debug_assert`, but `open`
    /// short-circuits every zero-length file through `empty()` first, so the
    /// precondition genuinely holds on both platform paths. Recorded as a
    /// non-finding.
    #[test]
    fn adv_zero_length_never_reaches_slurp() {
        let s = Scratch::new("adv-slurp0");
        let p = s.file("f", b"");
        let m = Mmap::open(&p).unwrap();
        assert_eq!(m.len(), 0);
        assert!(m.owned.is_none(), "empty() path, not slurp()");
        assert!(m.as_slice().is_empty());
        assert!(m.u64_slice(0, 0).unwrap().is_empty());
    }

    /// Full lifecycle of the heap-backed handle, no FFI, so Miri can check
    /// provenance/aliasing: build via `slurp`, move it, read through the raw
    /// `ptr` that aliases the owned `Vec`, drop.
    #[test]
    fn adv_miri_slurp_lifecycle() {
        let s = Scratch::new("adv-miri");
        for n in [1usize, 8, 15, 16, 33] {
            let bytes: Vec<u8> = (0..n).map(|i| (i * 7) as u8).collect();
            let p = s.file(&format!("f{n}"), &bytes);
            let m = slurped(&p);
            assert_eq!(m.as_slice(), &bytes[..]);
            let m = { let moved = m; moved };
            for off in (0..=n).step_by(8) {
                let max = (n - off) / 8;
                for c in 0..=max {
                    let w = m.u64_slice(off, c).unwrap();
                    assert_eq!(w.len(), c);
                    let _ = w.iter().fold(0u64, |a, b| a ^ *b);
                }
            }
            // read the byte view again after the word views are gone
            assert_eq!(m.as_slice(), &bytes[..]);
            drop(m);
        }
    }

    /// Debug reports an empty (never-mapped) handle as "mapped".
    #[test]
    fn adv_debug_mislabels_the_empty_handle() {
        let s = Scratch::new("adv-debug-empty");
        let p = s.file("f", b"");
        let m = Mmap::open(&p).unwrap();
        let d = format!("{m:?}");
        assert!(d.contains("mapped"), "{d}");
        assert!(!d.contains("heap"), "{d}");
    }

    /// Hammer a shared mapping from many threads while others open/drop their
    /// own mappings of the same file, looking for anything racy in Drop.
    #[test]
    fn adv_concurrent_open_read_drop_storm() {
        let s = Scratch::new("adv-storm");
        let vals: Vec<u64> = (0..8192u64).map(|i| i.wrapping_mul(0x9e37_79b9)).collect();
        let p = s.words("f", &vals);
        let shared = Arc::new(Mmap::open(&p).unwrap());
        std::thread::scope(|sc| {
            for _ in 0..8 {
                let shared = Arc::clone(&shared);
                let p = p.clone();
                let vals = &vals;
                sc.spawn(move || {
                    for _ in 0..200 {
                        let m = Mmap::open(&p).unwrap();
                        assert_eq!(m.u64_slice(0, 8192).unwrap(), &vals[..]);
                        assert_eq!(shared.u64_slice(4096 * 8, 1).unwrap()[0], vals[4096]);
                        drop(m);
                    }
                });
            }
        });
        assert_eq!(shared.u64_slice(0, 8192).unwrap(), &vals[..]);
    }
}
