//! The write-ahead log: the only thing standing between a committed write and
//! a power cut.
//!
//! A table's durable state is `parts + log`. Parts are rewritten by
//! checkpoints, which are expensive; the log absorbs writes in between at the
//! cost of one append and one `fsync`.
//!
//! ## File shape
//!
//! ```text
//!   header                        MAGIC + version, written once at creation
//!   varint len | u64 sum | body   one framed record, repeated
//! ```
//!
//! Records are framed individually rather than the file being framed as a
//! whole, because a log has no end: the checksum has to cover a unit that is
//! complete the instant it is written.
//!
//! ## Torn tails are normal, torn middles are not
//!
//! A crash during `write` leaves a partial record at the end of the file. That
//! is not corruption -- it is the expected shape of an interrupted append, and
//! the write it represents was never acknowledged. [`Wal::replay`] therefore
//! stops cleanly at the first record that cannot be complete and returns
//! everything before it.
//!
//! The distinction it draws is positional, and it is the whole trick: a
//! failure in a record that is followed by more bytes means the log damaged
//! itself *behind* a record it had already accepted, which no append can do --
//! that is bit rot, and it is reported. A failure in a record that runs to the
//! end of the file is a torn tail, and it is swallowed. (A short final write
//! that the filesystem padded rather than truncated lands in the second case
//! too, which is why a checksum failure at the very end is treated as a tear
//! rather than as rot. "The very end" stops at the last non-zero byte: a block
//! the filesystem allocated and never wrote back reads as a run of zeros, and
//! a hole is not a record -- see [`is_tail`].)
//!
//! ## Records that were logged before the write was known to succeed
//!
//! A log-before-apply caller fsyncs the record and *then* attempts the
//! mutation. When the mutation is rejected -- a type error in a widened block,
//! a constraint, an out-of-memory delta flush -- the statement returns an error
//! to the client while a durable record of it sits in the log. Replay would
//! resurrect a write that never happened, or choke on a record whose body the
//! mutation would have rejected. The record is durable; the write is not; and
//! nothing in a single-phase frame can tell the two apart.
//!
//! So a record has a second form. [`Wal::begin`] hands out a sequence number,
//! [`Wal::append_insert_staged`] and [`Wal::append_delete_staged`] log under
//! it, and only [`Wal::commit`] -- appended after the mutation has actually
//! succeeded -- makes those records visible to [`Wal::replay`]. A staged record
//! with no commit marker behind it is dropped, silently and by construction:
//! that is the definition of a write that was never acknowledged.
//!
//! The plain [`Wal::append_insert`]/[`Wal::append_delete`] forms are unchanged
//! and still committed the instant they are framed. They are the right choice
//! wherever the caller can order the work the other way round -- validate and
//! apply in memory, then log, then acknowledge -- which costs one `fsync`
//! instead of two and cannot leave an uncommitted record at all. Staging is for
//! the mutations that cannot be attempted before they are durable.
//!
//! Sequence numbers are per-log and resumed from the file on open, so a staged
//! group orphaned by a crash can never be released by a commit marker written
//! after the restart.
//!
//! ## LSNs are byte offsets, deliberately
//!
//! Every append returns the log-sequence number of the record it wrote, and
//! that number is the byte offset its frame starts at. Nothing is stored to
//! make this work: the offset is already unique, already monotonic, and
//! already the thing recovery navigates by.
//!
//! The alternative -- a counter carried in each record body -- was rejected
//! twice over. It costs a varint on every record, which on a `Delete` (a tag
//! and eight bytes) is a fifth of the frame. And it creates a *second* number
//! space that has to be kept consistent with the byte watermark a checkpoint
//! stores in the table's commit record, because that watermark is what
//! [`Wal::replay_from`] seeks to. Making the LSN and the watermark the same
//! number means "replay everything after LSN n" and "replay everything the
//! last checkpoint had not folded in" are the same call, and [`Wal::len`] is
//! both "how long is the log" and "what LSN does the next record get".
//!
//! The one thing a byte offset cannot do is stay monotonic across
//! [`Wal::truncate`], which a checkpoint calls once the records are inside
//! parts. It does not need to: the watermark is rewritten by the same
//! checkpoint, so both restart together and stay comparable.
//!
//! [`Wal::rewind_to`] is the other half. A transaction that rolls back leaves
//! staged records that replay would drop anyway -- but dropping them is not the
//! same as never having written them, and "ROLLBACK leaves no trace" is a claim
//! about the file too. Rewinding to the LSN the transaction started at makes
//! the log byte-identical to its pre-transaction state, which is sound exactly
//! because writers serialize: nothing else can have appended in between.
//!
//! ## What replay does not do
//!
//! Nothing here interprets records. `replay` hands back `Insert`/`Delete` in
//! log order and the caller applies them, because "apply" means different
//! things to a keyed table (idempotent, last-write-wins) and an append-only
//! one. Recovery skips the prefix a checkpoint already folded into parts using
//! the watermark in the table's commit record -- see [`super::store`].

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::common::{Error, Result};
use crate::types::{Block, Schema};

use super::format::{self, Reader, Writer};
use super::{reader, store};

const TAG_INSERT: u8 = 1;
const TAG_DELETE: u8 = 2;
/// Releases every staged record carrying the sequence number in its body.
const TAG_COMMIT: u8 = 3;

/// Set on an `INSERT`/`DELETE` tag to mark the record staged: durable, but not
/// part of the log's history until a [`TAG_COMMIT`] names its sequence number.
///
/// A flag bit rather than two more tags, so the payload encoding is shared
/// verbatim between the two forms and there is exactly one place that can get
/// it wrong. The high bit is free: tags are written as a single `u8`, never a
/// varint, so no existing value can collide with it.
const STAGED: u8 = 0x80;

/// One logged mutation.
#[derive(Clone, Debug, PartialEq)]
pub enum WalRecord {
    Insert(Block),
    /// Delete by primary-key *lane* (the order-preserving `u64` the key
    /// occupies in storage), not by `Value`: the lane is what the delta and
    /// the delete bitmaps are keyed by, and it needs no type context to log.
    Delete(u64),
}

/// What one frame turned out to hold.
enum Entry {
    /// A mutation. `Some(seq)` when it is staged and awaits a commit marker.
    Record(Option<u64>, WalRecord),
    Commit(u64),
}

pub struct Wal {
    file: File,
    path: PathBuf,
    len: u64,
    /// Next sequence number [`Wal::begin`] will hand out. Resumed past
    /// anything already in the file so that a commit marker written after a
    /// restart cannot release a group orphaned before it.
    next_seq: u64,
}

impl Wal {
    /// Open (or create) the log at `path`, verifying its header.
    ///
    /// A file shorter than a header cannot contain a record, so it is rewritten
    /// from scratch -- that is the "crashed between `creat` and the first
    /// write" case, and refusing to start over it would be a needless outage.
    pub fn open(path: &Path) -> Result<Wal> {
        let dir = parent_of(path);
        std::fs::create_dir_all(dir).map_err(|e| store::io_err("create directory", dir, e))?;
        let mut file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(path)
            .map_err(|e| store::io_err("open", path, e))?;
        let mut len = file
            .metadata()
            .map_err(|e| store::io_err("stat", path, e))?
            .len();

        let mut next_seq = 0u64;
        if len < format::HEADER_LEN as u64 {
            file.set_len(0).map_err(|e| store::io_err("truncate", path, e))?;
            write_header(&mut file, path)?;
            // The directory entry itself has to survive: a log whose file is
            // not durable is not a log.
            store::sync_dir(dir)?;
            len = format::HEADER_LEN as u64;
        } else {
            let buf = std::fs::read(path).map_err(|e| store::io_err("read", path, e))?;
            format::read_header(&mut Reader::new(&buf)).map_err(|e| store::prefix(path, e))?;

            // A crash can leave a half-written record at the tail. Replay
            // tolerates that and stops, but *appending* after it would not:
            // the new record would sit behind bytes that never parse, so the
            // next replay would stop before it and silently lose an
            // acknowledged write. Find the last intact boundary and discard
            // everything after it, so the log is append-clean again.
            let (good, seen) = Self::scan(&buf).map_err(|e| store::prefix(path, e))?;
            next_seq = seen;
            if good < len {
                file.set_len(good)
                    .map_err(|e| store::io_err("truncate the torn tail of", path, e))?;
                file.sync_all().map_err(|e| store::io_err("sync", path, e))?;
                store::sync_dir(dir)?;
                len = good;
            }
        }
        Ok(Wal { file, path: path.to_path_buf(), len, next_seq })
    }

    /// Log an insert. Not durable until [`Wal::sync`]. Returns its LSN.
    pub fn append_insert(&mut self, block: &Block) -> Result<u64> {
        self.put_insert(None, block)
    }

    /// Log a delete by primary-key lane. Not durable until [`Wal::sync`].
    /// Returns its LSN.
    pub fn append_delete(&mut self, key_lane: u64) -> Result<u64> {
        self.put_delete(None, key_lane)
    }

    /// Open a staging group. See the module docs: records logged under the
    /// returned sequence number stay invisible to [`Wal::replay`] until
    /// [`Wal::commit`] is given the same number.
    ///
    /// Cheap and infallible -- nothing is written. The group exists only in the
    /// records that carry its number.
    pub fn begin(&mut self) -> u64 {
        let seq = self.next_seq;
        // A log that appends 10^9 groups a second overflows this in 500 years.
        self.next_seq += 1;
        seq
    }

    /// Log an insert that replay must not apply until `seq` is committed.
    /// Returns its LSN.
    pub fn append_insert_staged(&mut self, seq: u64, block: &Block) -> Result<u64> {
        self.put_insert(Some(seq), block)
    }

    /// Log a delete that replay must not apply until `seq` is committed.
    /// Returns its LSN.
    pub fn append_delete_staged(&mut self, seq: u64, key_lane: u64) -> Result<u64> {
        self.put_delete(Some(seq), key_lane)
    }

    /// Release every record staged under `seq`. Not durable until
    /// [`Wal::sync`] -- and until it is, the write it covers must not be
    /// acknowledged, because that is the whole point of the marker.
    ///
    /// There is no matching `abort`: a group that is never committed is already
    /// dropped by replay, so writing a marker to say so would only add an
    /// `fsync` to the path that has just failed anyway. A caller that wants the
    /// abort erased rather than merely ignored rewinds instead --
    /// see [`Wal::rewind_to`].
    ///
    /// Returns the marker's LSN, which is the point at which the group became
    /// part of the log's history.
    pub fn commit(&mut self, seq: u64) -> Result<u64> {
        let mut body = Writer::with_capacity(16);
        body.u8(TAG_COMMIT);
        body.varint(seq);
        self.append(&body.finish())
    }

    /// Discard every record at or after `lsn`, durably.
    ///
    /// For ROLLBACK. `lsn` must be an LSN this log handed out (or [`Wal::len`]
    /// at some earlier moment), and every record after it must belong to the
    /// aborting transaction -- which holds because writers serialize, so
    /// nothing else can have appended since the transaction took that LSN.
    ///
    /// Replay would already drop those records: they are staged and no marker
    /// releases them. Rewinding is about the file rather than the recovery,
    /// and it is not cosmetic -- an abandoned group is bytes every subsequent
    /// `open` re-scans and every subsequent replay re-decodes, and a workload
    /// that rolls back often would otherwise grow a log made mostly of writes
    /// that never happened.
    ///
    /// A no-op when `lsn` is at or past the end, so a transaction that logged
    /// nothing costs nothing here.
    pub fn rewind_to(&mut self, lsn: u64) -> Result<()> {
        let floor = format::HEADER_LEN as u64;
        if lsn < floor {
            return Err(Error::storage(format!(
                "cannot rewind the log to {lsn}: the header ends at {floor}"
            )));
        }
        if lsn >= self.len {
            return Ok(());
        }
        self.file
            .set_len(lsn)
            .map_err(|e| store::io_err("rewind", &self.path, e))?;
        self.file
            .sync_all()
            .map_err(|e| store::io_err("fsync", &self.path, e))?;
        self.len = lsn;
        Ok(())
    }

    fn put_insert(&mut self, seq: Option<u64>, block: &Block) -> Result<u64> {
        // The same ceiling the reader applies to a record's row count. A log
        // record that cannot be replayed is worse than a refused insert: the
        // refusal is reported to the client, the unreadable record is
        // discovered at recovery.
        if block.rows() as u64 > reader::MAX_PART_ROWS {
            return Err(Error::storage(format!(
                "a log record of {} rows cannot be written: the format's limit is {} rows",
                block.rows(),
                reader::MAX_PART_ROWS
            )));
        }
        let mut body = Writer::with_capacity(block.bytes() + 64);
        put_tag(&mut body, TAG_INSERT, seq);
        super::writer::put_block(&mut body, block);
        self.append(&body.finish())
    }

    fn put_delete(&mut self, seq: Option<u64>, key_lane: u64) -> Result<u64> {
        let mut body = Writer::with_capacity(24);
        put_tag(&mut body, TAG_DELETE, seq);
        body.u64(key_lane);
        self.append(&body.finish())
    }

    /// Frame and append `body`, returning the record's LSN -- the offset it
    /// starts at, which is the log's length *before* the write.
    fn append(&mut self, body: &[u8]) -> Result<u64> {
        let lsn = self.len;
        let mut w = Writer::with_capacity(body.len() + 16);
        format::write_framed(&mut w, body);
        let bytes = w.finish();
        // One `write_all` per record: a record split across two syscalls could
        // be interleaved with another writer's, and framing cannot recover
        // from that the way it recovers from a short tail.
        self.file
            .write_all(&bytes)
            .map_err(|e| store::io_err("append to", &self.path, e))?;
        self.len += bytes.len() as u64;
        Ok(lsn)
    }

    /// Make every appended record durable.
    pub fn sync(&mut self) -> Result<()> {
        self.file
            .sync_all()
            .map_err(|e| store::io_err("fsync", &self.path, e))
    }

    /// Discard every record, durably, keeping the file (and its header) in
    /// place. Called by a checkpoint once the records are inside parts.
    pub fn truncate(&mut self) -> Result<()> {
        self.file
            .set_len(0)
            .map_err(|e| store::io_err("truncate", &self.path, e))?;
        write_header(&mut self.file, &self.path)?;
        self.len = format::HEADER_LEN as u64;
        Ok(())
    }

    /// Current size in bytes, i.e. the offset the next record will start at.
    /// This is the watermark a checkpoint records -- and, because an LSN *is*
    /// that offset, it is also the LSN the next append will return. The two
    /// readings are deliberately the same number; see the module docs.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len <= format::HEADER_LEN as u64
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Every record in the log, in order.
    pub fn replay(path: &Path, schema: &Schema) -> Result<Vec<WalRecord>> {
        Self::replay_from(path, schema, format::HEADER_LEN as u64)
    }

    /// Every record at or after `from`, each with the LSN it was written at.
    ///
    /// The same numbers [`Wal::append_insert`] and friends returned. That
    /// correspondence is what makes the checkpoint watermark and
    /// [`Wal::rewind_to`] sound -- both name a position in this space -- so it
    /// is worth being able to observe rather than merely assert.
    pub fn replay_with_lsn(
        path: &Path,
        schema: &Schema,
        from: u64,
    ) -> Result<Vec<(u64, WalRecord)>> {
        let buf = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(store::io_err("read", path, e)),
        };
        Self::replay_entries(&buf, schema, from).map_err(|e| store::prefix(path, e))
    }

    /// End offset of the last structurally intact record, and the first
    /// sequence number that is free to hand out.
    ///
    /// Framing only: a record whose frame is complete and whose checksum
    /// matches is a record we really wrote, whether or not its *body* still
    /// decodes against the current schema. Truncating on a body error would
    /// throw away durable data because of a schema mismatch, so body damage is
    /// left for `replay` to report.
    fn scan(buf: &[u8]) -> Result<(u64, u64)> {
        if buf.len() < format::HEADER_LEN {
            return Ok((format::HEADER_LEN as u64, 0));
        }
        let mut r = Reader::new(buf);
        format::read_header(&mut r)?;
        let mut good = r.pos() as u64;
        let mut next_seq = 0u64;
        while !r.is_empty() {
            let at = r.pos();
            match format::read_framed(&mut r) {
                Ok(body) => {
                    good = r.pos() as u64;
                    if let Some(s) = body_seq(body) {
                        next_seq = next_seq.max(s.saturating_add(1));
                    }
                }
                // A torn tail is the normal shape of a crash: stop here.
                Err(_) if is_tail(buf, at) => break,
                // Damage in the middle is not a tear. Refuse to silently
                // discard everything after it.
                Err(e) => return Err(record_err(at, 0, e)),
            }
        }
        Ok((good, next_seq))
    }

    /// Every record at or after byte offset `from`.
    ///
    /// `from` must be a record boundary recorded by a checkpoint; a bogus one
    /// lands mid-record and is caught by the frame checksum rather than
    /// silently yielding garbage.
    pub fn replay_from(path: &Path, schema: &Schema, from: u64) -> Result<Vec<WalRecord>> {
        let buf = match std::fs::read(path) {
            Ok(b) => b,
            // No log is the same as an empty log: nothing was ever written.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(store::io_err("read", path, e)),
        };
        Self::replay_bytes(&buf, schema, from).map_err(|e| store::prefix(path, e))
    }

    pub(crate) fn replay_bytes(
        buf: &[u8],
        schema: &Schema,
        from: u64,
    ) -> Result<Vec<WalRecord>> {
        // One extra pass moving the entries, against a recovery that then
        // *inserts every block into a table*. Carrying the LSN in the
        // primitive and dropping it here is the cheap direction: the other way
        // round would mean two nearly identical replay loops, and a second
        // implementation of the staged-record filter is exactly the thing that
        // would silently stop agreeing with the first.
        Ok(Self::replay_entries(buf, schema, from)?
            .into_iter()
            .map(|(_, rec)| rec)
            .collect())
    }

    /// The replay primitive: records with their LSNs.
    fn replay_entries(
        buf: &[u8],
        schema: &Schema,
        from: u64,
    ) -> Result<Vec<(u64, WalRecord)>> {
        if buf.len() < format::HEADER_LEN {
            // Torn before the first record could exist.
            return Ok(Vec::new());
        }
        let mut r = Reader::new(buf);
        format::read_header(&mut r)?;
        let start = (from.min(buf.len() as u64) as usize).max(format::HEADER_LEN);
        r.seek(start)?;

        let mut out = Vec::new();
        // Where each still-uncommitted record sits in `out`, with the sequence
        // number that would release it. This holds only genuinely staged
        // records -- at most one group per statement that failed since the last
        // checkpoint -- so the linear `retain` per commit marker is bounded by
        // the *failures*, not by the length of the log. A log with no staged
        // records (every log this engine has written so far) never touches it.
        let mut staged: Vec<(u64, usize)> = Vec::new();
        let mut seen = 0usize;
        while !r.is_empty() {
            let at = r.pos();
            // Only *framing* failures can be tears: an incomplete frame, or a
            // checksum that does not cover what is there. Once the checksum
            // passes, the bytes are provably the ones we wrote, so a body that
            // does not decode is damage no matter where it sits.
            let body = match format::read_framed(&mut r) {
                Ok(b) => b,
                Err(_) if is_tail(buf, at) => break,
                Err(e) => return Err(record_err(at, seen, e)),
            };
            match decode_entry(body, schema).map_err(|e| record_err(at, seen, e))? {
                Entry::Record(Some(seq), rec) => {
                    staged.push((seq, out.len()));
                    out.push((at as u64, rec));
                }
                Entry::Record(None, rec) => out.push((at as u64, rec)),
                Entry::Commit(seq) => staged.retain(|&(s, _)| s != seq),
            }
            seen += 1;
        }
        if staged.is_empty() {
            return Ok(out);
        }
        // What is left was logged for a mutation that never reported success.
        // Dropping it is not a repair -- it is the record meaning what it says.
        let mut orphan = vec![false; out.len()];
        for &(_, i) in &staged {
            orphan[i] = true;
        }
        Ok(out
            .into_iter()
            .zip(orphan)
            .filter(|&(_, dead)| !dead)
            .map(|(rec, _)| rec)
            .collect())
    }

    /// The LSN the next append will return. See [`Wal::len`] -- same number,
    /// spelled the way a transaction reads it.
    #[inline]
    pub fn lsn(&self) -> u64 {
        self.len
    }
}

/// Write a record's tag, and the sequence number a staged one carries in front
/// of its payload.
fn put_tag(w: &mut Writer, tag: u8, seq: Option<u64>) {
    match seq {
        Some(s) => {
            w.u8(tag | STAGED);
            w.varint(s);
        }
        None => w.u8(tag),
    }
}

/// The sequence number a frame body carries, for the open-time scan.
///
/// Deliberately lenient where `decode_entry` is not: `scan` has already proved
/// the frame checksum, and a body that will not parse against *some* schema is
/// a problem for `replay` to report -- refusing to open the log over it would
/// turn a schema mismatch into an outage.
fn body_seq(body: &[u8]) -> Option<u64> {
    let mut r = Reader::new(body);
    let tag = r.u8().ok()?;
    if tag == TAG_COMMIT || tag & STAGED != 0 {
        r.varint().ok()
    } else {
        None
    }
}

fn record_err(at: usize, index: usize, e: Error) -> Error {
    Error::corruption(format!("record {index} of the replay, at offset {at}: {e}"))
}

fn decode_entry(body: &[u8], schema: &Schema) -> Result<Entry> {
    let mut br = Reader::new(body);
    let tag = br.u8()?;
    let rec = if tag == TAG_COMMIT {
        Entry::Commit(br.varint()?)
    } else {
        // The sequence number sits between the tag and the payload, so the
        // staged and plain forms share one decoder for the part that matters.
        let seq = if tag & STAGED != 0 { Some(br.varint()?) } else { None };
        match tag & !STAGED {
            TAG_INSERT => Entry::Record(seq, WalRecord::Insert(reader::get_block(&mut br, schema)?)),
            TAG_DELETE => Entry::Record(seq, WalRecord::Delete(br.u64()?)),
            // The raw tag, not the masked one: `0x83` is not "tag 3".
            _ => return Err(Error::corruption(format!("unknown log record tag {tag}"))),
        }
    };
    if !br.is_empty() {
        return Err(Error::corruption(format!(
            "{} trailing bytes in a log record",
            br.remaining()
        )));
    }
    Ok(rec)
}

/// True when the record starting at `at` is the last thing in the file --
/// i.e. its frame does not end strictly before the end of the log's *data*, so
/// a failure to read it is an interrupted append rather than damage to an
/// accepted record.
///
/// The end of the data is the last non-zero byte, not EOF. A crash can leave
/// the tail of the log as a run of zeros rather than as a short file: the
/// filesystem allocated a block and never wrote it back, so what survives is a
/// hole. Those zeros are not a record and never were one -- a run of them
/// cannot be a frame we wrote, since every record body starts with a tag and
/// the checksum stored for an empty one is not zero -- so measuring against
/// EOF would turn the most ordinary crash shape into permanent bit rot, and
/// (through `last_intact_offset`) into a log that cannot be opened at all.
///
/// Zeros in the *middle* are still damage: a real record after them puts the
/// data end past them, so the frame they start ends strictly before it and is
/// reported, exactly like any other rot behind an accepted record.
fn is_tail(buf: &[u8], at: usize) -> bool {
    let data_end = buf.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1) as u64;
    let mut r = Reader::new(buf);
    if r.seek(at).is_err() {
        return true;
    }
    let Ok(len) = r.varint() else { return true };
    if r.u64().is_err() {
        return true;
    }
    match (r.pos() as u64).checked_add(len) {
        Some(end) => end >= data_end,
        None => true,
    }
}

fn write_header(file: &mut File, path: &Path) -> Result<()> {
    let mut w = Writer::with_capacity(format::HEADER_LEN);
    format::write_header(&mut w);
    file.write_all(w.as_slice())
        .map_err(|e| store::io_err("write the header of", path, e))?;
    file.sync_all().map_err(|e| store::io_err("fsync", path, e))
}

fn parent_of(path: &Path) -> &Path {
    match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persist::testkit::*;
    use crate::types::{Column, DataType, Value};

    fn rows(keys: &[u64]) -> Block {
        Block::new(vec![
            Column::u64s(DataType::UInt64, keys.to_vec()),
            Column::strs(
                DataType::String,
                keys.iter().map(|k| format!("h{k}").into()).collect(),
            ),
            Column::i64s(DataType::Int64, keys.iter().map(|&k| k as i64 * -3).collect()),
            Column::f64s(DataType::Float64, keys.iter().map(|&k| k as f64 / 4.0).collect()),
        ])
        .unwrap()
    }

    /// A log holding `n` alternating inserts and deletes, plus the record
    /// list it should replay to and the byte offset each record starts at.
    fn populated(s: &Scratch, n: usize) -> (PathBuf, Vec<WalRecord>, Vec<u64>) {
        let path = s.join("wal.log");
        let mut w = Wal::open(&path).unwrap();
        let mut want = Vec::new();
        let mut offsets = vec![w.len()];
        for i in 0..n as u64 {
            if i % 3 == 2 {
                w.append_delete(i * 7).unwrap();
                want.push(WalRecord::Delete(i * 7));
            } else {
                let b = rows(&[i * 10, i * 10 + 1]);
                w.append_insert(&b).unwrap();
                want.push(WalRecord::Insert(b));
            }
            offsets.push(w.len());
        }
        w.sync().unwrap();
        (path, want, offsets)
    }

    #[test]
    fn a_fresh_log_is_exactly_a_header() {
        let s = Scratch::new("wal-fresh");
        let path = s.join("wal.log");
        let w = Wal::open(&path).unwrap();
        assert!(w.is_empty());
        assert_eq!(w.len(), format::HEADER_LEN as u64);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), format::HEADER_LEN as u64);
        assert!(Wal::replay(&path, &schema()).unwrap().is_empty());
    }

    #[test]
    fn records_replay_in_order() {
        let s = Scratch::new("wal-order");
        let (path, want, _) = populated(&s, 25);
        assert_eq!(Wal::replay(&path, &schema()).unwrap(), want);
    }

    #[test]
    fn a_large_insert_roundtrips() {
        let s = Scratch::new("wal-big");
        let path = s.join("wal.log");
        let b = sample_block(5_000);
        let mut w = Wal::open(&path).unwrap();
        w.append_insert(&b).unwrap();
        w.sync().unwrap();
        let back = Wal::replay(&path, &schema()).unwrap();
        assert_eq!(back, vec![WalRecord::Insert(b)]);
    }

    #[test]
    fn nulls_and_strings_survive_the_log() {
        let s = Scratch::new("wal-nulls");
        let path = s.join("wal.log");
        let b = sample_block(200);
        assert!(b.column(2).has_nulls());
        let mut w = Wal::open(&path).unwrap();
        w.append_insert(&b).unwrap();
        w.sync().unwrap();
        let WalRecord::Insert(back) = Wal::replay(&path, &schema()).unwrap().remove(0) else {
            panic!("expected an insert")
        };
        assert!(back.column(2).is_null(0));
        assert_eq!(back.column(1).value(3), b.column(1).value(3));
        assert_eq!(back, b);
    }

    #[test]
    fn reopening_appends_after_the_existing_records() {
        let s = Scratch::new("wal-reopen");
        let (path, mut want, _) = populated(&s, 4);
        {
            let mut w = Wal::open(&path).unwrap();
            assert!(!w.is_empty());
            w.append_delete(999).unwrap();
            want.push(WalRecord::Delete(999));
            w.sync().unwrap();
        }
        assert_eq!(Wal::replay(&path, &schema()).unwrap(), want);
    }

    #[test]
    fn truncate_reclaims_the_log_but_keeps_it_usable() {
        let s = Scratch::new("wal-truncate");
        let (path, _, _) = populated(&s, 6);
        let mut w = Wal::open(&path).unwrap();
        w.truncate().unwrap();
        assert!(w.is_empty());
        assert_eq!(std::fs::metadata(&path).unwrap().len(), format::HEADER_LEN as u64);
        assert!(Wal::replay(&path, &schema()).unwrap().is_empty());

        w.append_delete(5).unwrap();
        w.sync().unwrap();
        assert_eq!(Wal::replay(&path, &schema()).unwrap(), vec![WalRecord::Delete(5)]);
    }

    #[test]
    fn replay_from_skips_a_checkpointed_prefix() {
        let s = Scratch::new("wal-from");
        let (path, want, offsets) = populated(&s, 9);
        for (i, &off) in offsets.iter().enumerate() {
            assert_eq!(
                Wal::replay_from(&path, &schema(), off).unwrap(),
                want[i.min(want.len())..],
                "from record {i}"
            );
        }
        // A watermark past the end yields nothing rather than an error.
        assert!(Wal::replay_from(&path, &schema(), 1 << 30).unwrap().is_empty());
    }

    /// The required torn-tail case: a crash mid-append.
    #[test]
    fn a_torn_tail_replays_the_intact_prefix() {
        let s = Scratch::new("wal-torn");
        let (path, want, offsets) = populated(&s, 10);
        let full = std::fs::read(&path).unwrap();
        let last = *offsets.last().unwrap() as usize;
        let cut = (offsets[offsets.len() - 2] as usize + last) / 2;
        assert!(cut > offsets[offsets.len() - 2] as usize && cut < last);
        std::fs::write(&path, &full[..cut]).unwrap();

        let back = Wal::replay(&path, &schema()).unwrap();
        assert_eq!(back, want[..want.len() - 1], "the torn record must be dropped, no error");
    }

    #[test]
    fn every_truncation_replays_a_clean_prefix() {
        let s = Scratch::new("wal-torn-sweep");
        let (path, want, offsets) = populated(&s, 12);
        let full = std::fs::read(&path).unwrap();
        let mut last_count = 0usize;
        for cut in 0..=full.len() {
            let got = Wal::replay_bytes(&full[..cut], &schema(), format::HEADER_LEN as u64)
                .unwrap_or_else(|e| panic!("prefix {cut} of {} errored: {e}", full.len()));
            assert_eq!(got, want[..got.len()], "prefix {cut} must be a prefix of the log");
            assert!(got.len() >= last_count, "prefix {cut} lost records");
            last_count = got.len();
            // A prefix that reaches a record boundary must contain exactly the
            // records that end at or before it.
            if let Some(k) = offsets.iter().position(|&o| o as usize == cut) {
                assert_eq!(got.len(), k, "prefix {cut} ends at record boundary {k}");
            }
        }
        assert_eq!(last_count, want.len());
    }

    #[test]
    fn a_truncated_header_is_an_empty_log_not_an_error() {
        let s = Scratch::new("wal-shorthdr");
        let path = s.join("wal.log");
        for n in 0..format::HEADER_LEN {
            std::fs::write(&path, vec![0u8; n]).unwrap();
            assert!(Wal::replay(&path, &schema()).unwrap().is_empty(), "n={n}");
            // ...and reopening rebuilds it rather than refusing to start.
            let w = Wal::open(&path).unwrap();
            assert!(w.is_empty());
        }
    }

    #[test]
    fn a_corrupt_record_in_the_middle_is_reported() {
        let s = Scratch::new("wal-midrot");
        let (path, _, offsets) = populated(&s, 5);
        let mut bytes = std::fs::read(&path).unwrap();
        // Last byte of the second record's body: fully inside the file, with
        // three more records after it.
        let victim = offsets[2] as usize - 1;
        bytes[victim] ^= 0x08;
        std::fs::write(&path, &bytes).unwrap();
        let e = Wal::replay(&path, &schema()).unwrap_err();
        assert_eq!(e.code(), "CHECKSUM_MISMATCH", "{e}");
        assert!(e.to_string().contains("record 1 of the replay"), "{e}");
    }

    #[test]
    fn damage_to_the_final_record_is_treated_as_a_tear() {
        let s = Scratch::new("wal-tailrot");
        let (path, want, _) = populated(&s, 5);
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x80;
        std::fs::write(&path, &bytes).unwrap();
        let back = Wal::replay(&path, &schema()).unwrap();
        assert_eq!(back, want[..want.len() - 1]);
    }

    #[test]
    fn a_bad_magic_is_corruption() {
        let s = Scratch::new("wal-magic");
        let (path, _, _) = populated(&s, 3);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[1] ^= 0x40;
        std::fs::write(&path, &bytes).unwrap();
        let e = Wal::replay(&path, &schema()).unwrap_err();
        assert!(e.to_string().contains("bad magic"), "{e}");
        assert!(Wal::open(&path).is_err(), "opening a foreign file must refuse");
    }

    #[test]
    fn a_newer_format_version_is_refused() {
        let s = Scratch::new("wal-version");
        let (path, _, _) = populated(&s, 3);
        let mut bytes = std::fs::read(&path).unwrap();
        let at = format::MAGIC.len();
        bytes[at..at + 4].copy_from_slice(&(format::FORMAT_VERSION + 1).to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let e = Wal::replay(&path, &schema()).unwrap_err();
        assert!(e.to_string().contains("unsupported format version"), "{e}");
        let e = Wal::open(&path).err().expect("a foreign header must be refused");
        assert!(e.to_string().contains("unsupported format version"), "{e}");
    }

    #[test]
    fn a_record_that_does_not_match_the_schema_is_refused() {
        use crate::types::{Field, Schema};
        let s = Scratch::new("wal-schema");
        let path = s.join("wal.log");
        let mut w = Wal::open(&path).unwrap();
        w.append_insert(&rows(&[1, 2])).unwrap();
        w.sync().unwrap();

        let narrow = Schema::new(vec![Field::new("id", DataType::UInt64)]).unwrap();
        let e = Wal::replay(&path, &narrow).unwrap_err();
        assert!(e.to_string().contains("the table has 1"), "{e}");

        let wrong = Schema::new(vec![
            Field::new("id", DataType::UInt64),
            Field::new("host", DataType::Int64), // was String
            Field::new("ms", DataType::Int64),
            Field::new("ratio", DataType::Float64),
        ])
        .unwrap();
        let e = Wal::replay(&path, &wrong).unwrap_err();
        assert!(e.to_string().contains("record column 1"), "{e}");
    }

    #[test]
    fn an_unknown_record_tag_is_corruption() {
        let s = Scratch::new("wal-tag");
        let path = s.join("wal.log");
        Wal::open(&path).unwrap();
        let mut w = Writer::new();
        format::write_framed(&mut w, &[9u8, 0, 0]); // unknown tag
        format::write_framed(&mut w, &[TAG_DELETE, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(w.as_slice()).unwrap();
        drop(f);
        let e = Wal::replay(&path, &schema()).unwrap_err();
        assert!(e.to_string().contains("unknown log record tag 9"), "{e}");
    }

    #[test]
    fn replaying_a_missing_log_yields_nothing() {
        let s = Scratch::new("wal-absent");
        assert!(Wal::replay(&s.join("nope.log"), &schema()).unwrap().is_empty());
    }

    #[test]
    fn sync_is_safe_to_repeat_and_survives_a_reopen() {
        let s = Scratch::new("wal-sync");
        let path = s.join("wal.log");
        let mut w = Wal::open(&path).unwrap();
        w.sync().unwrap();
        w.append_delete(1).unwrap();
        w.sync().unwrap();
        w.sync().unwrap();
        let on_disk = std::fs::metadata(&path).unwrap().len();
        assert_eq!(on_disk, w.len(), "a synced log must be exactly as long as we think");
        drop(w);
        assert_eq!(Wal::replay(&path, &schema()).unwrap(), vec![WalRecord::Delete(1)]);
    }

    #[test]
    fn delete_lanes_roundtrip_at_the_extremes() {
        let s = Scratch::new("wal-lanes");
        let path = s.join("wal.log");
        let lanes = [0u64, 1, u64::MAX, u64::MAX / 2, 1 << 63];
        let mut w = Wal::open(&path).unwrap();
        for &l in &lanes {
            w.append_delete(l).unwrap();
        }
        w.sync().unwrap();
        let got: Vec<u64> = Wal::replay(&path, &schema())
            .unwrap()
            .into_iter()
            .map(|r| match r {
                WalRecord::Delete(l) => l,
                _ => panic!("expected a delete"),
            })
            .collect();
        assert_eq!(got, lanes);
    }

    #[test]
    fn a_log_in_a_directory_that_does_not_exist_yet_is_created() {
        let s = Scratch::new("wal-mkdir");
        let path = s.join("db").join("t").join("wal.log");
        let mut w = Wal::open(&path).unwrap();
        w.append_delete(3).unwrap();
        w.sync().unwrap();
        assert_eq!(Wal::replay(&path, &schema()).unwrap(), vec![WalRecord::Delete(3)]);
    }

    #[test]
    fn records_are_framed_and_checksummed_individually() {
        // Two records must be independently verifiable: damaging the first
        // must not be reported as damage to the second.
        let s = Scratch::new("wal-frames");
        let (path, _, offsets) = populated(&s, 2);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(offsets.len(), 3);
        // Each record: varint length + 8-byte checksum + body.
        for w in offsets.windows(2) {
            let (a, b) = (w[0] as usize, w[1] as usize);
            let mut r = Reader::new(&bytes[a..b]);
            let len = r.varint().unwrap();
            let _sum = r.u64().unwrap();
            assert_eq!(r.pos() + len as usize, b - a, "record frame must be exact");
        }
    }

    // ---- adversarial review additions -----------------------------------

    /// A crash mid-append leaves a torn tail. `Wal::open` sizes itself from
    /// `metadata().len()` and appends *after* the garbage instead of
    /// truncating back to the last good record boundary, so the next record --
    /// which is acknowledged to the client after `sync()` -- sits behind a
    /// frame that can never be parsed.
    #[test]
    fn adversarial_reopen_after_a_torn_tail_destroys_the_log() {
        let s = Scratch::new("wal-adv-reopen-torn");
        let (path, want, offsets) = populated(&s, 4);
        let full = std::fs::read(&path).unwrap();

        // Crash halfway through appending the last record.
        let cut = (offsets[3] as usize + offsets[4] as usize) / 2;
        std::fs::write(&path, &full[..cut]).unwrap();
        // Before the restart, recovery is clean: 3 intact records.
        assert_eq!(Wal::replay(&path, &schema()).unwrap(), want[..3]);

        // Restart, log a new write, acknowledge it.
        let mut w = Wal::open(&path).unwrap();
        assert_eq!(
            w.len(),
            offsets[3],
            "open must rewind to the last intact record boundary"
        );
        w.append_delete(0xDEAD_BEEF).unwrap();
        w.sync().unwrap();

        // Crash again; recover.
        let after = Wal::replay(&path, &schema());
        match after {
            Err(e) => panic!("log became unreadable after a legal restart: {e}"),
            Ok(recs) => assert!(
                recs.contains(&WalRecord::Delete(0xDEAD_BEEF)),
                "the acknowledged post-restart record was silently dropped: {} records replayed",
                recs.len()
            ),
        }
    }

    /// The same shape, but arrived at through the public API only: append,
    /// sync, simulate the interrupted append by writing a partial frame with
    /// the raw file handle, restart, append again.
    #[test]
    fn adversarial_partial_frame_then_restart() {
        let s = Scratch::new("wal-adv-partial");
        let path = s.join("wal.log");
        {
            let mut w = Wal::open(&path).unwrap();
            w.append_delete(1).unwrap();
            w.sync().unwrap();
        }
        // An interrupted append: the frame header made it, the body did not.
        let mut torn = Writer::new();
        format::write_framed(&mut torn, &[TAG_DELETE, 0, 0, 0, 0, 0, 0, 0, 0]);
        let torn = torn.finish();
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&torn[..torn.len() - 4]).unwrap();
            f.sync_all().unwrap();
        }
        assert_eq!(
            Wal::replay(&path, &schema()).unwrap(),
            vec![WalRecord::Delete(1)],
            "the torn tail alone must replay the intact prefix"
        );

        // Restart and log another write.
        let mut w = Wal::open(&path).unwrap();
        w.append_delete(2).unwrap();
        w.sync().unwrap();

        let got = Wal::replay(&path, &schema());
        match got {
            Err(e) => panic!("log unreadable after restart: {e}"),
            Ok(r) => assert!(
                r.contains(&WalRecord::Delete(2)),
                "post-restart record lost, replayed {r:?}"
            ),
        }
    }

    /// The module docs claim "a short final write that the filesystem padded
    /// rather than truncated lands in the second case too". It only does when
    /// the padding is exactly as long as the frame; a block-sized zero tail
    /// (the classic ext4 delayed-allocation crash shape) parses as a
    /// zero-length frame that ends well before EOF, so `is_tail` says no and
    /// the whole log is rejected.
    #[test]
    fn adversarial_block_sized_zero_padding_is_reported_as_rot() {
        let s = Scratch::new("wal-adv-zeropad");
        let (path, want, _) = populated(&s, 4);
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&[0u8; 4096]).unwrap();
            f.sync_all().unwrap();
        }
        match Wal::replay(&path, &schema()) {
            Ok(got) => assert_eq!(got, want, "a zero-padded tail must replay the intact prefix"),
            Err(e) => panic!("a zero-padded tail was reported as corruption: {e}"),
        }
    }

    /// ...and because `Wal::open` now recovers the tail through the same
    /// predicate, a zero-padded log cannot even be opened: the table can never
    /// be checkpointed again, because `save_catalog` opens the log to truncate it.
    #[test]
    fn adversarial_zero_padded_log_cannot_be_reopened() {
        let s = Scratch::new("wal-adv-zeropad-open");
        let (path, _, _) = populated(&s, 4);
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&[0u8; 4096]).unwrap();
            f.sync_all().unwrap();
        }
        match Wal::open(&path) {
            Ok(w) => assert!(w.len() > 0),
            Err(e) => panic!("a zero-padded log became permanently unopenable: {e}"),
        }
    }

    /// The other half of the zero-padding contract: a hole is only a tear when
    /// it runs to the end. Zeros with an accepted record behind them are
    /// damage the log did to itself, and must still be reported.
    #[test]
    fn a_zero_run_in_the_middle_is_still_corruption() {
        let s = Scratch::new("wal-midzero");
        let (path, _, offsets) = populated(&s, 5);
        let full = std::fs::read(&path).unwrap();
        let at = offsets[2] as usize;
        let mut spliced = full[..at].to_vec();
        spliced.extend_from_slice(&[0u8; 512]);
        spliced.extend_from_slice(&full[at..]);
        std::fs::write(&path, &spliced).unwrap();
        let e = Wal::replay(&path, &schema()).unwrap_err();
        assert_eq!(e.code(), "CHECKSUM_MISMATCH", "{e}");
        assert!(e.to_string().contains("record 2 of the replay"), "{e}");
    }

    // ---- staged records --------------------------------------------------

    /// The bug: the record is fsynced *before* the mutation is attempted, so a
    /// statement that then fails leaves a durable record of a write that never
    /// happened. Replay would resurrect it.
    ///
    /// A staged record with no commit marker behind it is exactly that
    /// situation, spelled out in the log rather than inferred, and replay drops
    /// it.
    #[test]
    fn a_staged_record_that_was_never_committed_is_not_replayed() {
        let s = Scratch::new("wal-staged-drop");
        let path = s.join("wal.log");
        let mut w = Wal::open(&path).unwrap();

        w.append_delete(1).unwrap(); // an ordinary, already-committed write
        let seq = w.begin();
        w.append_insert_staged(seq, &rows(&[100, 101])).unwrap();
        w.sync().unwrap();
        // ...and here the mutation is rejected: no commit marker is written.

        assert_eq!(
            Wal::replay(&path, &schema()).unwrap(),
            vec![WalRecord::Delete(1)],
            "an uncommitted staged record must not be replayed"
        );
    }

    #[test]
    fn a_committed_staged_group_replays_in_log_order() {
        let s = Scratch::new("wal-staged-commit");
        let path = s.join("wal.log");
        let mut w = Wal::open(&path).unwrap();

        let b = rows(&[7, 8]);
        let a = w.begin();
        w.append_insert_staged(a, &b).unwrap();
        w.append_delete_staged(a, 42).unwrap();
        w.sync().unwrap();
        w.commit(a).unwrap();
        w.sync().unwrap();
        w.append_delete(9).unwrap();
        w.sync().unwrap();

        assert_eq!(
            Wal::replay(&path, &schema()).unwrap(),
            vec![WalRecord::Insert(b), WalRecord::Delete(42), WalRecord::Delete(9)],
            "a committed group must keep its position in the log, not move to the marker"
        );
    }

    /// The marker releases *its* group and nothing else. A later statement
    /// succeeding must not resurrect an earlier one that failed.
    #[test]
    fn a_commit_marker_releases_only_its_own_group() {
        let s = Scratch::new("wal-staged-scoped");
        let path = s.join("wal.log");
        let mut w = Wal::open(&path).unwrap();

        let failed = w.begin();
        w.append_delete_staged(failed, 0xBAD).unwrap();
        let ok = w.begin();
        w.append_delete_staged(ok, 0x600D).unwrap();
        w.commit(ok).unwrap();
        w.sync().unwrap();

        assert_eq!(
            Wal::replay(&path, &schema()).unwrap(),
            vec![WalRecord::Delete(0x600D)],
            "the failed group must stay dropped"
        );
    }

    /// Sequence numbers are resumed from the file, so a group orphaned by a
    /// crash cannot be released by a marker written after the restart -- which
    /// is exactly what would happen if `begin` restarted from zero.
    #[test]
    fn a_restart_cannot_release_a_group_orphaned_before_it() {
        let s = Scratch::new("wal-staged-restart");
        let path = s.join("wal.log");
        {
            let mut w = Wal::open(&path).unwrap();
            let seq = w.begin();
            assert_eq!(seq, 0, "a fresh log starts at zero");
            w.append_delete_staged(seq, 0xBAD).unwrap();
            w.sync().unwrap();
        }
        let mut w = Wal::open(&path).unwrap();
        let seq = w.begin();
        assert_ne!(seq, 0, "open must resume past the sequence numbers in the file");
        w.append_delete_staged(seq, 0x600D).unwrap();
        w.commit(seq).unwrap();
        w.sync().unwrap();

        assert_eq!(Wal::replay(&path, &schema()).unwrap(), vec![WalRecord::Delete(0x600D)]);
    }

    /// A commit marker is a record like any other, so a crash between the
    /// mutation and the marker's `fsync` is a torn tail -- and the write it
    /// covers was, correctly, never acknowledged.
    #[test]
    fn every_truncation_of_a_staged_log_replays_a_clean_prefix() {
        let s = Scratch::new("wal-staged-torn");
        let path = s.join("wal.log");
        let b = rows(&[3, 4]);
        {
            let mut w = Wal::open(&path).unwrap();
            w.append_delete(1).unwrap();
            let seq = w.begin();
            w.append_insert_staged(seq, &b).unwrap();
            w.commit(seq).unwrap();
            w.append_delete(2).unwrap();
            w.sync().unwrap();
        }
        let full = std::fs::read(&path).unwrap();
        let want = [WalRecord::Delete(1), WalRecord::Insert(b), WalRecord::Delete(2)];
        let mut last = 0usize;
        for cut in 0..=full.len() {
            let got = Wal::replay_bytes(&full[..cut], &schema(), format::HEADER_LEN as u64)
                .unwrap_or_else(|e| panic!("prefix {cut} errored: {e}"));
            // Every prefix is a prefix of the committed history: the staged
            // insert only appears once the marker behind it is whole.
            assert_eq!(got, want[..got.len()], "prefix {cut}");
            assert!(got.len() >= last, "prefix {cut} lost a record");
            last = got.len();
        }
        assert_eq!(last, want.len());
    }

    /// Reopening a log whose tail is a staged group must not truncate it away
    /// or refuse: staging is a body-level property, and `open` deals in frames.
    #[test]
    fn opening_a_log_that_ends_in_a_staged_group_is_clean() {
        let s = Scratch::new("wal-staged-open");
        let path = s.join("wal.log");
        let mut w = Wal::open(&path).unwrap();
        let seq = w.begin();
        w.append_delete_staged(seq, 5).unwrap();
        w.sync().unwrap();
        let len = w.len();
        drop(w);

        let mut w = Wal::open(&path).unwrap();
        assert_eq!(w.len(), len, "an intact staged record is not a torn tail");
        // It is still uncommitted, and still dropped -- but the group is now
        // closed to any future marker, so committing the *resumed* number does
        // not release it.
        let fresh = w.begin();
        w.commit(fresh).unwrap();
        w.sync().unwrap();
        assert!(Wal::replay(&path, &schema()).unwrap().is_empty());
    }

    /// The staged flag rides the high bit of the tag byte; a tag that sets it
    /// over an unknown value must still be reported with the byte that was
    /// actually there.
    #[test]
    fn an_unknown_staged_tag_names_the_raw_byte() {
        let s = Scratch::new("wal-staged-tag");
        let path = s.join("wal.log");
        Wal::open(&path).unwrap();
        let mut w = Writer::new();
        format::write_framed(&mut w, &[STAGED | 9, 0]);
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(w.as_slice()).unwrap();
        drop(f);
        let e = Wal::replay(&path, &schema()).unwrap_err();
        assert!(e.to_string().contains(&format!("tag {}", STAGED | 9)), "{e}");
    }

    // ---- LSNs --------------------------------------------------------------

    /// The whole contract in one test: the number an append hands back is the
    /// number replay reports, is the offset the frame starts at, and is what
    /// `len()` said a moment earlier. Those four have to be one number, because
    /// `replay_from` and the checkpoint watermark navigate by it.
    #[test]
    fn an_lsn_is_the_offset_replay_finds_the_record_at() {
        let s = Scratch::new("wal-lsn");
        let path = s.join("wal.log");
        let mut w = Wal::open(&path).unwrap();
        let mut want = Vec::new();
        for i in 0..12u64 {
            let before = w.lsn();
            assert_eq!(before, w.len(), "lsn() and len() are the same number");
            let lsn = if i % 3 == 2 {
                w.append_delete(i).unwrap()
            } else {
                w.append_insert(&rows(&[i * 10, i * 10 + 1])).unwrap()
            };
            assert_eq!(lsn, before, "an append returns the log's length before it");
            want.push(lsn);
        }
        w.sync().unwrap();

        let got = Wal::replay_with_lsn(&path, &schema(), format::HEADER_LEN as u64).unwrap();
        assert_eq!(got.len(), want.len());
        for (i, ((lsn, _), &expect)) in got.iter().zip(&want).enumerate() {
            assert_eq!(*lsn, expect, "record {i}");
            // ...and seeking to it replays exactly the suffix beginning there.
            assert_eq!(
                Wal::replay(&path, &schema()).unwrap().len() - i,
                Wal::replay_from(&path, &schema(), *lsn).unwrap().len(),
                "record {i}: an LSN is a valid watermark"
            );
        }
    }

    /// A committed staged group keeps the LSN of the record, not of the marker
    /// that released it. Recovery position and commit order are different
    /// questions.
    #[test]
    fn a_staged_record_keeps_its_own_lsn() {
        let s = Scratch::new("wal-lsn-staged");
        let path = s.join("wal.log");
        let mut w = Wal::open(&path).unwrap();
        let seq = w.begin();
        let at = w.append_delete_staged(seq, 77).unwrap();
        let marker = w.commit(seq).unwrap();
        w.sync().unwrap();
        assert!(marker > at);
        let got = Wal::replay_with_lsn(&path, &schema(), format::HEADER_LEN as u64).unwrap();
        assert_eq!(got, vec![(at, WalRecord::Delete(77))]);
    }

    // ---- rewind ------------------------------------------------------------

    /// ROLLBACK's half: rewinding to the LSN a transaction started at makes
    /// the file byte-identical to what it was, not merely semantically equal.
    #[test]
    fn rewinding_restores_the_log_byte_for_byte() {
        let s = Scratch::new("wal-rewind");
        let (path, want, _) = populated(&s, 6);
        let before = std::fs::read(&path).unwrap();

        let mut w = Wal::open(&path).unwrap();
        let mark = w.lsn();
        let seq = w.begin();
        w.append_insert_staged(seq, &rows(&[1, 2, 3])).unwrap();
        w.append_delete_staged(seq, 9).unwrap();
        w.sync().unwrap();
        assert!(w.len() > mark, "the aborting transaction really did write");

        w.rewind_to(mark).unwrap();
        assert_eq!(w.len(), mark);
        assert_eq!(std::fs::read(&path).unwrap(), before, "no trace on disk");
        assert_eq!(Wal::replay(&path, &schema()).unwrap(), want);

        // ...and the log is immediately usable again, with the next record
        // landing exactly where the rolled-back one did.
        assert_eq!(w.append_delete(5).unwrap(), mark);
        w.sync().unwrap();
        let mut expect = want.clone();
        expect.push(WalRecord::Delete(5));
        assert_eq!(Wal::replay(&path, &schema()).unwrap(), expect);
    }

    #[test]
    fn rewinding_forward_or_to_the_end_is_a_no_op() {
        let s = Scratch::new("wal-rewind-noop");
        let (path, want, _) = populated(&s, 3);
        let mut w = Wal::open(&path).unwrap();
        let len = w.len();
        // A transaction that logged nothing rewinds to where it already is.
        w.rewind_to(len).unwrap();
        w.rewind_to(len + 4096).unwrap();
        assert_eq!(w.len(), len);
        assert_eq!(Wal::replay(&path, &schema()).unwrap(), want);
    }

    /// The header is not a record and rewinding into it would leave a file
    /// that cannot be opened. Refused rather than clamped: a caller passing an
    /// LSN this small has a bug, and silently repairing it hides the bug.
    #[test]
    fn rewinding_into_the_header_is_refused() {
        let s = Scratch::new("wal-rewind-header");
        let (path, want, _) = populated(&s, 2);
        let mut w = Wal::open(&path).unwrap();
        assert!(w.rewind_to(0).is_err());
        assert!(w.rewind_to(format::HEADER_LEN as u64 - 1).is_err());
        assert_eq!(Wal::replay(&path, &schema()).unwrap(), want, "and nothing was lost");
    }

    #[test]
    fn the_lane_of_a_value_is_what_gets_logged() {
        // The log stores lanes, so a signed key logs its order-preserving
        // form and comes back identical.
        let s = Scratch::new("wal-signed");
        let path = s.join("wal.log");
        let lane = Value::Int(-42).to_lane(&DataType::Int64).unwrap();
        let mut w = Wal::open(&path).unwrap();
        w.append_delete(lane).unwrap();
        w.sync().unwrap();
        assert_eq!(Wal::replay(&path, &schema()).unwrap(), vec![WalRecord::Delete(lane)]);
    }
}
