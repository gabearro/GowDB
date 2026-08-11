//! The write-ahead log: the only thing standing between a committed write and
//! a power cut.
//!
//! A table's durable state is `parts + log`. Parts are rewritten by
//! checkpoints, which are expensive; the log absorbs writes in between at the
//! cost of one append and one `fsync`.
//!
//! ## The log is a directory of numbered segments
//!
//! ```text
//!   <root>/.wal/<db>/<table>/seg_<origin:020>.gwal
//! ```
//!
//! Each file is a 56-byte header and then framed records:
//!
//! ```text
//!    0  MAGIC[8]            written once, by store::atomic_write, at creation
//!    8  u32 FORMAT_VERSION
//!   12  u32 carry_len       bytes of carried TAG_DECIDE frames at the head
//!   16  u64 next_seq        Wal::begin's counter, carried across the roll
//!   24  u64 prev_seq        the previous segment's last recovery LSN
//!   32  u64 prev_ms         its wall clock
//!   40  u64 durable         updated in place by one pwrite per fsync
//!   48  u64 acked           the previous `durable`, whose fsync returned
//!   56  u64 check           over bytes 8..56 and the origin from the file name
//!   64  varint len | u64 sum | body   one framed record, repeated
//! ```
//!
//! Records are framed individually rather than the file being framed as a
//! whole, because a log has no end: the checksum has to cover a unit that is
//! complete the instant it is written. That frame is unchanged from the
//! single-file layout that came before -- the segmentation is about *which
//! file* bytes live in, not about how a record is written.
//!
//! The `origin` is deliberately **not** stored in the file. It is the name,
//! and `check` binds to it, so a segment that is renamed or copied over
//! another refuses instead of lying about where it sits in the stream.
//!
//! ### Archiving is not an operation
//!
//! A segment is *sealed* exactly when it is not the highest-origin file in the
//! directory. There is no seal record, no sidecar, no link and no copy: a
//! segment joins the archive by ceasing to be the newest, which happens the
//! instant its successor is published by one `rename`. That is the whole of
//! the fix for "exFAT has no hard links" -- the write-ahead log never touches
//! a link on any filesystem, because nothing is ever moved.
//!
//! It also deletes a crash window rather than handling one. There is no state
//! in which a segment has been sealed but its successor does not exist: if the
//! successor was not published, the old segment is still the highest-origin
//! file, hence still the active one, and appends resume into it.
//!
//! The directory sits at `<root>/.wal/...` rather than inside the table's own
//! directory, and that is load-bearing rather than tidy. `DROP TABLE` does
//! `remove_dir_all` on the table directory while a point-in-time recovery
//! rolls forward every table the *backup manifest* names -- so a log under the
//! table would turn "someone dropped the table, restore to just before" into
//! "restore the backup and lose everything after it", which is the commonest
//! reason this feature exists. Dot-prefixed, so [`store::is_safe_name`]
//! refuses `.wal` as a database name and the dropped-table collector cannot
//! reach it.
//!
//! ## Torn tails are normal, torn middles are not -- and the log records which
//!
//! A crash during `write` leaves a partial record at the end of a file. That
//! is not corruption; it is the expected shape of an interrupted append, and
//! the write it represents was never acknowledged. Damage *behind* a record
//! the log already accepted is bit rot, and it has to be reported, because
//! stopping there silently discards every acknowledged record after it.
//!
//! This distinction used to be positional: a frame that ran to the end of the
//! data was a tear, one that ended before it was rot. That is the bug this
//! layout exists to kill. The decision read the frame's own length field to
//! find its end -- and a corrupted length can always claim to reach the end,
//! so a torn *middle* read as a torn tail and replay stopped there, dropping
//! every record beyond it with exit 0 and no quarantine.
//!
//! So the inference is replaced by a recorded fact. [`Wal::sync`] stamps
//! `durable` -- the stream position through which this segment has been
//! acknowledged -- into the header with one 16-byte `pwrite`, immediately
//! before the `fsync` that was already going to happen. The classifier is then
//! one comparison, and it is the same one at every call site:
//!
//! ```text
//!   effective = min(durable, data_end)      data_end = last non-zero byte + 1
//!   a failure at p >= effective  ->  torn tail: truncate to p, silently
//!   a failure at p <  effective  ->  CORRUPTION: report, name the range
//!   the walk ending below `acked`  ->  CORRUPTION, whatever it looked like
//! ```
//!
//! The `min` is required and is not the old heuristic in disguise. The pwrite
//! and the appended bytes go into the same `fsync`, so when that `fsync`
//! *returns*, both are durable and `durable` is exact. When a crash lands
//! before it returns, the header page and the tail pages persist
//! independently, and the header page -- low offset, dirtied at every sync --
//! is the more likely of the two. So `durable` can over-claim, and only in one
//! shape: the appended bytes never reached the platter, which means the file
//! is short or the region reads as zeros. Clamping by `data_end` removes
//! exactly that shape. The zero rule survives, demoted: it no longer decides
//! tear-versus-rot, it only bounds a claim.
//!
//! But a clamp alone forgives *too much*: a hole that swallows the whole body
//! also ends the data early, so "everything was lost" would read as "nothing
//! was ever acknowledged" and replay would silently return no records at all.
//! That is why `acked` exists. It is the value `durable` held before the last
//! update -- and that update was followed by an `fsync` that returned, so
//! every byte below `acked` is provably on the platter. A segment whose
//! records stop below it has lost acknowledged data, whatever the tail looks
//! like, and says so. The forgiveness window is therefore exactly one group
//! commit wide, which is exactly the amount a single un-returned `fsync` can
//! be wrong by. It costs eight bytes in a header, written by the same
//! `pwrite`.
//!
//! One residual, stated rather than hidden: an interrupted append that leaves
//! **non-zero** garbage below `durable` -- a torn sector rather than a lost
//! block -- is reported as corruption. That is a false alarm, it is loud, and
//! it is the direction this subsystem is supposed to fail in.
//!
//! A **sealed** segment cannot have a torn tail at all: its `durable` was
//! stamped and fsynced before its successor was published, so the successor's
//! existence proves the whole file is durable. Every failure inside one is
//! corruption, full stop -- and a sealed segment that replays short of its
//! `durable` is corruption too, which is what catches a flipped length varint
//! in an archived segment whose file length never changed.
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
//! Sequence numbers are per log and carried across every roll in the
//! successor's header, so a staged group orphaned by a crash can never be
//! released by a commit marker written after the restart -- or after a
//! checkpoint. That header field replaced a `TAG_FENCE` record that only a
//! *carrying* truncation used to write, so the guarantee got stronger by
//! deleting code: it now holds unconditionally.
//!
//! ## Two ways to name a deleted row
//!
//! [`Wal::append_delete`] names one by its primary-key *lane* -- a value, so
//! it survives anything that moves the row, and a table with a single-column
//! key needs nothing else. The *default* MergeTree shape has no such lane:
//! `ORDER BY` alone is a sort key, not a unique one. Those tables delete by
//! scanning, so what they have to record is a **position**, and a position is
//! only meaningful relative to something that pins the rows it indexes.
//!
//! [`crate::storage::Part::pid`] is that thing, and it is why a positional
//! record is sound here where `(part index, row)` would not be. Recovery
//! rebuilds a *checkpointed* part by reading its file, which holds the same
//! rows at the same offsets under the same identity -- through the checkpoint
//! that rewrites the part under a new file name, and through a backup and
//! restore, because the identity is inside the part's own bytes rather than in
//! its directory entry. A part built *since* the last checkpoint has no file
//! and exists only as the `Insert` records that fed it, and how those regroup
//! on replay depends on flush timing and compaction; such a part is not
//! citable, and the sweep falls back to writing the table out for exactly
//! those rows. See `TAG_MASK_RUN` and `Session::apply_sweep`.
//!
//! ## LSNs are stream offsets, deliberately
//!
//! Every append returns the log-sequence number of the record it wrote: the
//! byte offset its frame starts at, counted over the *stream* rather than over
//! one file. For a record at file offset `o` in the segment named `O`, that is
//! `O + o - 56`. Segments abut exactly -- `origin(N+1) = origin(N) + len(N) -
//! 56` -- so the numbering is contiguous, and it never restarts for the life
//! of the data directory. At 1 GB/s of log, 2^64 bytes is 584 years.
//!
//! Nothing is stored to make this work: the offset is already unique, already
//! monotonic, and already the thing recovery navigates by. The alternative --
//! a counter carried in each record body -- was rejected twice over. It costs
//! a varint on every record, which on a `Delete` (a tag and eight bytes) is a
//! fifth of the frame. And it creates a *second* number space that has to be
//! kept consistent with the watermark a checkpoint stores in the table's
//! commit record, because that watermark is what [`Wal::replay_from`] seeks
//! to. Making the LSN and the watermark the same number means "replay
//! everything after LSN n" and "replay everything the last checkpoint had not
//! folded in" are the same call.
//!
//! The old single-file layout had a *second* number space anyway -- the
//! archive's own stream position, invented because a byte offset could not
//! survive `truncate`. Promoting that numbering to *being* the LSN collapses
//! the two into one. It also makes the checkpoint watermark exact by
//! construction rather than by arithmetic that has to be kept in step: a
//! checkpoint rolls the log and then records [`Wal::origin`], which *is* the
//! sealed segment's stream end.
//!
//! [`Wal::len`] is therefore "the stream position the next record will get",
//! not "how big is the active file". Anything that wants "how much has this
//! log grown since the last checkpoint" wants [`Wal::pending`].
//!
//! [`Wal::rewind_to`] is the other half. A transaction that rolls back leaves
//! staged records that replay would drop anyway -- but dropping them is not the
//! same as never having written them, and "ROLLBACK leaves no trace" is a claim
//! about the file too. Rewinding to the LSN the transaction started at makes
//! the log byte-identical to its pre-transaction state, which is sound exactly
//! because writers serialize: nothing else can have appended in between. It
//! also lowers `durable`, which is the one thing in this design that is not
//! monotone and the only place it needs to be: staged records *are* fsynced,
//! so leaving the stamp high would make the very next open report a healthy
//! truncated file as corruption.
//!
//! No transaction can span a roll, and that is now enforced rather than
//! inherited from five call sites: [`Wal::begin`] counts open groups and
//! [`Wal::roll`] refuses while any are outstanding. The floor `rewind_to`
//! checks is `origin + carry_len`, not `origin` -- the carried decision
//! markers live in that interval, and rewinding through them would make every
//! sibling's prepare read "no decision", which is abort, which silently drops
//! a committed transaction in *other* tables.
//!
//! ## A transaction that spans several tables
//!
//! Logs are per table, so an N-table transaction has N commit markers and N
//! `fsync`s, and a crash between two of them used to leave a *prefix* of the
//! transaction durable: tables A and B committed, C lost, and no error to
//! anyone. [`Wal::prepare`] and [`Wal::decide`] close that.
//!
//! The N-1 earlier participants log a **prepare** instead of a commit marker.
//! A prepare names the group it would release, the log that holds the
//! transaction's decision, and the sequence number of the decision inside it;
//! [`Wal::replay`] releases the group only if that decision is really there.
//! The last participant logs a **decision**, which is an ordinary commit
//! marker for its own group and the citable outcome for everyone else's. So
//! the whole transaction turns on one `fsync` -- the last one -- and a crash
//! before it drops every participant, including the ones already `fsync`ed.
//!
//! The prepares still cost N-1 barriers; what they no longer cost is N-1
//! *waits*. The protocol wants them durable before the decision is written and
//! in no order among themselves, so [`Session::commit_durable`](crate::session::Session) appends all of
//! them, then issues their barriers on the pool and joins before appending the
//! decision. The join is unconditional -- there is no path on which a failed
//! barrier is discovered after the decision has been written -- and a failure
//! aborts the COMMIT with every table still private. See "the barrier is a
//! device operation" below for the measurement and for the shortcut that is
//! refused.
//!
//! A citation names a *directory* (`../t2`), because a segment's file name
//! changes at every roll and the directory does not. Citations resolve against
//! the cited log's **active segment only**, which is exactly the set a roll
//! carries forward: [`Wal::may_be_cited`] answers false exactly when no
//! sibling log holds a record its table's parts do not already cover, which is
//! exactly when nothing can still cite us -- so the active segment holds the
//! complete citable set, always.
//!
//! That carry-forward is the same argument as before, and it stays on the roll
//! rather than moving onto retention. Making `prune` responsible for keeping a
//! cited segment would put a 2PC correctness obligation on a user-tunable byte
//! budget, so somebody setting `wal_archive_retention` aggressively could make
//! a committed transaction unresolvable through an unrelated knob.
//!
//! ## Ticks: when a record became a fact
//!
//! A byte offset says where a record is, not *when* it happened, and a
//! point-in-time recovery has to answer both. So [`Wal::sync`] -- the call
//! that turns appended bytes into acknowledged ones -- first appends a
//! [`TAG_TICK`] carrying two numbers: the wall clock, and a **recovery LSN**
//! taken from a process-wide counter.
//!
//! The placement is the whole design. A record's timestamp is the *first tick
//! at or after it*, which is exactly the instant it became durable; records
//! with no tick behind them were never acknowledged and a recovery must not
//! resurrect them. That makes every recovery cut land on a tick boundary, and
//! a cut on a tick boundary is a state the database really was in.
//!
//! The cost is one clock read and one 17-byte record per `fsync`, not per
//! record: a tick is emitted only when something has been appended since the
//! last one, so group commit -- several statements behind one `fsync` -- gets
//! one tick for the group, which is the correct granularity anyway. The
//! recovery LSN is *global* and the byte LSN is per log, deliberately: logs
//! are per table, so a byte offset cannot order a write to `a` against a write
//! to `b`, and "restore the database to this point" is a statement about every
//! table at once. It is resumed at [`Wal::open`] from the highest tick in the
//! active segment and, failing that, from `prev_seq` in its header -- so
//! unlike the old layout it cannot restart at 1 on a database that has been
//! checkpointed but never archived.
//!
//! ## What one append costs
//!
//! An autocommit `INSERT` is two `write_all`s and one `fsync`: the record,
//! then the tick that [`Wal::sync`] appends because something is dirty. The
//! stamp adds one 16-byte `pwrite` in front of the `fsync` -- per group
//! commit, not per record. Measured `F_FULLFSYNC` on the development machine
//! is 3.77 ms best; a 16-byte `pwrite` is ~2 us, so the stamp is +0.05% of the
//! call it rides on, and it buys the deletion of the entire silent-loss class.
//!
//! Nothing is allocated per record beyond the one buffer a record is built
//! into. That buffer is built *ending* at a fixed offset so the frame header
//! can be written in front of the body in place -- one allocation and zero
//! copies, where the previous shape allocated a second `Writer` and memcpy'd
//! the whole body into it, which on a granule-sized insert was a 128 kB malloc
//! and a 128 kB copy for nothing.
//!
//! ### The barrier is a device operation, and this is what follows from it
//!
//! A durability barrier here is `F_FULLFSYNC`, which flushes the *device's*
//! write cache. Measured on the development machine, best-of-21 with the order
//! alternated: `F_FULLFSYNC` 3.83-3.92 ms, plain `fsync(2)` 0.025-0.034 ms, a
//! directory `fsync` 4.0-4.8 ms. So the cost is per **call**, not per byte,
//! and the only two ways to spend less of it are to make fewer calls or to
//! make several calls share one flush. Both are used, and the second is the
//! one that is easy to get wrong:
//!
//!   * **Concurrent barriers share a flush; sequential ones cannot.** Measured
//!     on distinct files: three sequential 11.78 ms against 3.84 ms issued
//!     from three threads, eight sequential 31.78 against 12.08. That is what
//!     [`Session::commit_durable`](crate::session::Session) exploits -- the N-1 prepares of a
//!     multi-table transaction have to be durable before the decision is
//!     written but in no order relative to each other, so their barriers are
//!     issued together and joined. An N-table COMMIT costs two barriers of
//!     latency instead of N, and every prepare still gets a real
//!     `F_FULLFSYNC` that returned.
//!   * **What is refused, and why.** Downgrading the prepares to plain
//!     `fsync(2)` and leaning on the decision's full barrier to flush the
//!     device for all of them is ~2x faster again and is unsound.
//!     `F_FULLFSYNC` guarantees a post-condition of its *return*, not an
//!     ordering during its execution; a power cut inside that last flush can
//!     persist the decision while a participant's bytes are still in the
//!     device cache, and replay would then release every prepared group and
//!     find one participant's records missing. That is the half-committed
//!     transaction the two-phase protocol exists to prevent, with a smaller
//!     window. The same reasoning refuses `F_BARRIERFSYNC` (0.31-0.38 ms,
//!     tempting) for the decision: `kill -9` kills the process and leaves the
//!     page cache, which is exactly the property in question, so only a real
//!     power-cut rig could test it and this project does not have one.
//!
//! ### There is nothing to group-commit across writers
//!
//! Classical group commit batches the commits of *concurrent* writers into one
//! `fsync`. That mechanism has no state to work on here, and the reason is
//! structural rather than a matter of scale: [`Session`](crate::session::Session)`::check_txn` and
//! `check_owner` **refuse** a second connection's statement while a
//! transaction is open, and the `Writer` guard takes the lock exclusively for
//! a whole statement. There is no state in which two writers hold uncommitted
//! work that could share a barrier -- a concurrent writer is an error, not a
//! queue waiting to be batched. Do not build cross-connection group commit
//! without a server in front of it, and when there is one, note that what it
//! buys is bounded by the same measurement above: the second concurrent
//! barrier is nearly free, so the win is real but it is a *server* feature.
//!
//! Deferring the barrier instead -- acknowledging a COMMIT before it and
//! flushing on a timer, PostgreSQL's `synchronous_commit = off` -- is refused
//! on the same page. After the batching above, the residual is one barrier per
//! transaction, which is what a single-writer engine is supposed to pay; what
//! deferral buys is that last ~4 ms, and the price is the first line of this
//! file.
//!
//! ## Retention
//!
//! A byte budget ([`set_archive_retention`], reached from SQL by `SET
//! wal_archive_retention`) plus a count cap, because an archive that grows
//! without bound is how this feature becomes an outage, and a roll per fold
//! lets a delete-per-commit workload mint a segment per transaction. Pruning
//! drops whole segments, oldest first. There is no `HORIZON` file: the oldest
//! surviving segment's `prev_seq` *is* "the highest recovery LSN the archive
//! no longer holds", by construction -- which is strictly more correct than a
//! file, because the horizon then changes exactly when the segment disappears
//! and there is no window at all.
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
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::common::{mum, Error, Result};
use crate::storage::MaskRuns;
use crate::types::{Block, Schema};

use super::format::{self, Reader, Writer};
use super::{reader, store};

const TAG_INSERT: u8 = 1;
const TAG_DELETE: u8 = 2;
/// Releases every staged record carrying the sequence number in its body.
const TAG_COMMIT: u8 = 3;
/// Releases a staged group only if another log holds the decision it names.
/// Body: this log's sequence number, the decision's, then the path to the
/// directory of the log that holds it, relative to *this* log's directory.
const TAG_PREPARE: u8 = 4;
/// A [`TAG_COMMIT`] that a [`TAG_PREPARE`] in another log may cite. Identical
/// in effect where it sits; a distinct tag so [`Wal::roll`] can tell the
/// records it must carry forward from the ones it may drop, without keeping a
/// second index of which markers somebody else depends on.
const TAG_DECIDE: u8 = 5;
/// A run of key deletes: `varint count | count x u64 LE`. One frame per
/// [`DELETE_BATCH`] chunk instead of one per lane, which is where 55% of a
/// bulk `DELETE`'s log volume was going -- 900 000 bytes for 50 000 rows
/// against 400 097, in the same seven `write_all`s. The win that matters is
/// not append latency (an 8.3 ms statement loses 0.4 ms, under the noise
/// floor) but log volume, and therefore fold frequency at a fixed
/// `wal_fold_bytes`, and replay: 300 000 delete frames measured ~47 ms of a
/// 146 ms recovery and collapse to 37 run records.
const TAG_DELETE_RUN: u8 = 6;
/// Recovery LSN and wall clock, written by [`Wal::sync`] in front of the
/// `fsync` that makes the records behind it facts. Body: the LSN, then
/// milliseconds since the epoch. See the module docs.
const TAG_TICK: u8 = 7;
/// A run of *positions* hidden inside one part, named by that part's durable
/// identity: `varint pid | varint count | count x varint(pos - prev - 1)`.
///
/// The unkeyed counterpart to [`TAG_DELETE_RUN`], and the reason an unkeyed
/// `DELETE` no longer costs a table rewrite per statement. A key lane is a
/// *value* and survives anything that moves the row; a position is not, so it
/// needs something that pins the rows it indexes. [`crate::storage::Part::pid`]
/// is that thing: recovery rebuilds a checkpointed part by reading its file,
/// which holds the same rows at the same offsets under the same identity, so
/// `(pid, pos)` names the same row after a crash, after the checkpoint that
/// rewrote the part under a new file name, and after a restore. A part built
/// *since* the last checkpoint has no such file and is not citable; the sweep
/// falls back to writing the table out for those, and only those.
///
/// Delta-varints because positions leave the sweep ascending within a part
/// (granules in order, offsets in order inside each), which makes a dense
/// delete one byte per row where the keyed record spends eight. `prev` starts
/// at -1, so the first delta is the position itself.
const TAG_MASK_RUN: u8 = 9;

/// Set on an `INSERT`/`DELETE`/`DELETE_RUN` tag to mark the record staged:
/// durable, but not part of the log's history until a [`TAG_COMMIT`] -- or a
/// [`TAG_DECIDE`], or a [`TAG_PREPARE`] whose citation resolves -- names its
/// sequence number.
///
/// A flag bit rather than more tags, so the payload encoding is shared
/// verbatim between the two forms and there is exactly one place that can get
/// it wrong. The high bit is free: tags are written as a single `u8`, never a
/// varint, so no existing value can collide with it.
const STAGED: u8 = 0x80;

/// Lanes framed into one delete run before it is handed to `write`.
/// See [`Wal::put_deletes`] for why it is a chunk rather than the batch.
const DELETE_BATCH: usize = 8192;

/// Ceiling on a record body, checked on both sides of the frame.
///
/// [`reader::MAX_PART_ROWS`] is `1 << 40`, and a wide enough schema can put a
/// single block body past `u32` without ever reaching it. A refused insert
/// beats an unreplayable record: the refusal reaches the client, the record is
/// discovered at recovery.
const MAX_FRAME: u64 = 1 << 31;

// --------------------------------------------------------------- the archive

/// Where every table's log lives, under the data root.
///
/// Dot-prefixed for the same reason [`store::atomic_write`]'s temp files are:
/// [`store::is_safe_name`] refuses a leading dot, so no `CREATE DATABASE` can
/// ever collide with it and the dropped-table collector cannot reach it.
pub const WAL_DIR: &str = ".wal";

/// One segment of a table's log. Both live and archived: the newest is the
/// one being appended to and every other one is the archive.
const SEG_EXT: &str = "gwal";

/// Fixed header at the front of every segment. See the module docs.
pub const SEG_HEADER_LEN: u64 = 64;

/// Segments kept per table however small they are.
///
/// A roll per fold means an unkeyed-`DELETE`-per-`COMMIT` workload can produce
/// a segment per transaction, ~80 bytes each, and 64 MiB of budget would admit
/// ~800 000 of them -- a directory nobody can `ls`. The cap binds only when
/// segments are smaller than `retention / 256` (256 KiB at the default), i.e.
/// when folds are frequent, in which case the reach in *records* is still
/// large. Dropping for count raises the horizon exactly as dropping for bytes
/// does, so a recovery that needed them refuses, loudly.
const MAX_SEGMENTS: usize = 256;

/// Bytes of archived log kept per table before the oldest segments are
/// dropped.
///
/// 64 MiB is about 3.5 million logged deletes, or the log of a fairly busy
/// day for an embedded workload -- enough to cover the window between two
/// backups, small enough that nobody notices it under their data directory.
/// The alternative default, *off*, was rejected: point-in-time recovery you
/// have to have switched on before the incident is point-in-time recovery you
/// do not have when it happens.
pub const DEFAULT_ARCHIVE_BYTES: u64 = 64 << 20;

/// Default `wal_fold_bytes`: the per-table log size that triggers the only
/// automatic checkpoint this engine has.
///
/// 64 MiB, and the number is an argument rather than a round figure.
///
///   * A fold is [`crate::session::Session::fold_to_parts`]: a `flush` of the
///     buffered delta into a part, a `write_table` rewrite of the table's part
///     manifest, and a log roll. So its cost is O(*delta* + *parts*), not
///     O(*log*) and -- correcting what this comment used to claim -- not
///     O(*table rows*) either. It is still the wrong shape to pay often: the
///     manifest rewrite and the fsyncs are per fold, not per byte, so a small
///     threshold pays a fixed price over and over for a few megabytes of log.
///     That is the argument for a large number, and why "checkpoint every
///     commit" is not the answer.
///   * Measured, and it is not free: 4 x 2M-row `INSERT ... SELECT` in one
///     process ran **1.069x** slower at this default than at
///     `wal_fold_bytes = 0` (best-of-5, interleaved, one binary with the
///     feature toggled). It does not compound -- part count and on-disk size
///     came out identical either way. Sustained bulk writers who checkpoint on
///     their own schedule should set 0 and know why.
///   * It matches [`DEFAULT_ARCHIVE_BYTES`], so one fold publishes roughly one
///     archive segment's worth and retention keeps a few of them.
///   * At the ~73 bytes per row a narrow table logs, it caps replay at about
///     900k rows -- a bounded, sub-second recovery -- while a wide table hits
///     it sooner, which is the right direction: wide rows are what make replay
///     slow.
///
/// 0 disables it, which is exactly what the engine did before this existed and
/// is kept reachable for a workload that would rather checkpoint on its own
/// schedule.
pub const DEFAULT_FOLD_BYTES: u64 = 64 << 20;

/// The live retention budget. 0 keeps no sealed segment at all.
///
/// Process-global rather than per-session, which is the opposite of how
/// [`crate::settings`] holds everything else, and deliberately: retention
/// describes a *directory*, not a connection, and two sessions with different
/// retentions on one data directory would each be wrong about what the archive
/// contains. There is one writer per directory (the `LOCK` file sees to that),
/// so there is one answer.
static ARCHIVE_BYTES: AtomicU64 = AtomicU64::new(DEFAULT_ARCHIVE_BYTES);

/// The next recovery LSN. Never zero, so zero is free to mean "no tick".
static NEXT_COMMIT: AtomicU64 = AtomicU64::new(1);

pub fn set_archive_retention(bytes: u64) {
    ARCHIVE_BYTES.store(bytes, Ordering::Relaxed);
}

pub fn archive_retention() -> u64 {
    ARCHIVE_BYTES.load(Ordering::Relaxed)
}

/// The recovery LSN the next `fsync` will stamp.
///
/// Every record acknowledged so far carries a strictly smaller one, which is
/// what makes this the exact boundary between "already in the backup being
/// taken" and "must be replayed on top of it" -- see
/// [`crate::backup::write_archive`].
pub fn commit_seq() -> u64 {
    NEXT_COMMIT.load(Ordering::Acquire)
}

/// Resume the counter past `seq`, which some segment already used.
fn observe_commit(seq: u64) {
    if seq != 0 {
        NEXT_COMMIT.fetch_max(seq + 1, Ordering::AcqRel);
    }
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

/// The recovery LSNs and wall-clock times a segment spans. All zero means
/// "nothing was ever acknowledged from it".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Span {
    pub first_seq: u64,
    pub last_seq: u64,
    pub first_ms: u64,
    pub last_ms: u64,
}

impl Span {
    fn observe(&mut self, seq: u64, ms: u64) {
        if self.last_seq == 0 {
            self.first_seq = seq;
            self.first_ms = ms;
        }
        self.last_seq = seq;
        self.last_ms = ms;
    }

    pub fn is_empty(&self) -> bool {
        self.last_seq == 0
    }
}

/// Where a point-in-time recovery stops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// Everything the archive holds.
    Latest,
    /// Up to and including this recovery LSN.
    Lsn(u64),
    /// Up to and including this instant, in milliseconds since the epoch.
    Time(u64),
}

impl Target {
    /// Does a tick stamped `(seq, ms)` fall at or before the target?
    ///
    /// Both axes are monotone in the log, so the first tick that answers `no`
    /// ends the replay and nothing after it need be read.
    fn keeps(self, seq: u64, ms: u64) -> bool {
        match self {
            Target::Latest => true,
            Target::Lsn(n) => seq <= n,
            Target::Time(t) => ms <= t,
        }
    }
}

/// One logged mutation.
#[derive(Clone, Debug, PartialEq)]
pub enum WalRecord {
    Insert(Block),
    /// Delete by primary-key *lane* (the order-preserving `u64` the key
    /// occupies in storage), not by `Value`: the lane is what the delta and
    /// the delete bitmaps are keyed by, and it needs no type context to log.
    Delete(u64),
    /// Hide these row positions inside the part whose durable identity is the
    /// first field. The unkeyed counterpart of `Delete`; see `TAG_MASK_RUN`.
    ///
    /// One record per part rather than one per row -- the keyed path's
    /// `Vec<(lsn, WalRecord)>` costs forty bytes of recovery heap per lane,
    /// and positions are eight times denser in the log than lanes are, so
    /// expanding them the same way would multiply the peak by eight. A boxed
    /// slice keeps the enum the width it already was.
    Mask(u64, Box<[u64]>),
}

/// What one frame turned out to hold.
enum Entry<'a> {
    /// A mutation. `Some(seq)` when it is staged and awaits a commit marker.
    Record(Option<u64>, WalRecord),
    /// A run of key deletes, over the lane bytes still inside the frame --
    /// borrowed, not decoded, so a 65 kB run costs no allocation until the
    /// records are actually built.
    Deletes(Option<u64>, &'a [u8]),
    /// A run of hidden positions: the part identity, the count, and the
    /// delta-varint bytes, still borrowed out of the frame.
    Masks(Option<u64>, u64, u64, &'a [u8]),
    Commit(u64),
    /// Local group, the directory holding the decision, and the decision's
    /// number.
    Prepare(u64, &'a str, u64),
    /// A durability stamp. Bookkeeping to `replay` -- which only has to keep
    /// the recovery LSN counter from reissuing the number -- and the whole
    /// index to a point-in-time recovery, which reads it off the frame.
    Tick(u64),
}

// ---------------------------------------------------------------------------
// the segment header
// ---------------------------------------------------------------------------

/// A segment's fixed header, minus the parts that never vary.
///
/// Four things live here that a first cut put in a framed `TAG_HEAD` record at
/// the start of the body, and the difference is not cosmetic: a framed head
/// puts the body's start *above* the origin, which makes `stream_end >
/// covered_prefix` true by the head's own length for a table that was
/// checkpointed a microsecond ago -- turning [`Wal::may_be_cited`] and
/// [`archive_lags`] into constants. In the fixed header, `body_start = origin +
/// carry_len` with `carry_len = 0` in the overwhelming common case, so both
/// predicates are exact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Head {
    /// Bytes of carried [`TAG_DECIDE`] frames at the head of the body.
    carry_len: u32,
    /// [`Wal::begin`]'s counter, carried across the roll.
    next_seq: u64,
    /// The previous segment's last recovery LSN (0 for the first).
    prev_seq: u64,
    /// Its wall clock.
    prev_ms: u64,
    /// The stream position through which this segment is acknowledged.
    durable: u64,
    /// The value `durable` held before the last update. Its `fsync` returned,
    /// so every byte below it is on the platter and no amount of damage at the
    /// tail may be read as "it was never written".
    acked: u64,
}

/// Offset of the mutable `durable | check` pair. The only bytes `pwrite`
/// touches.
const STAMP_AT: u64 = 40;

impl Head {
    /// The header bytes of a segment whose name says `origin`.
    fn encode(&self, origin: u64) -> [u8; SEG_HEADER_LEN as usize] {
        let mut b = [0u8; SEG_HEADER_LEN as usize];
        b[..8].copy_from_slice(&format::MAGIC);
        b[8..12].copy_from_slice(&format::FORMAT_VERSION.to_le_bytes());
        b[12..16].copy_from_slice(&self.carry_len.to_le_bytes());
        b[16..24].copy_from_slice(&self.next_seq.to_le_bytes());
        b[24..32].copy_from_slice(&self.prev_seq.to_le_bytes());
        b[32..40].copy_from_slice(&self.prev_ms.to_le_bytes());
        b[40..48].copy_from_slice(&self.durable.to_le_bytes());
        b[48..56].copy_from_slice(&self.acked.to_le_bytes());
        let c = head_check(&b, origin);
        b[56..].copy_from_slice(&c.to_le_bytes());
        b
    }

    /// The 24 bytes `pwrite` puts back at [`STAMP_AT`].
    fn stamp(&self, origin: u64) -> [u8; 24] {
        self.encode(origin)[STAMP_AT as usize..].try_into().expect("24 bytes")
    }
}

/// The header's own check, over its fixed fields **and the origin the file
/// name declares**.
///
/// Binding to a number that is not in the file is the point: a segment that is
/// renamed, or copied over another, refuses instead of claiming a stream
/// position it does not occupy. It costs nothing -- the origin is already
/// parsed to find the file.
fn head_check(b: &[u8; SEG_HEADER_LEN as usize], origin: u64) -> u64 {
    let mut s = [0u8; 56];
    s[..48].copy_from_slice(&b[8..56]);
    s[48..].copy_from_slice(&origin.to_le_bytes());
    mum(format::checksum(&s), origin | 1)
}

/// Read a segment header, and say whether its stamp verifies.
///
/// A header whose `check` fails is not fatal: the fixed fields are still the
/// ones we wrote or the magic and version would have failed first, and the
/// honest response to an untrustworthy `durable` is to treat it as *infinite*,
/// so every failure in the file reports rather than being swallowed. Refusing
/// outright would turn a single flipped bit in a header into an unopenable
/// table when the records are all still there.
///
/// `buf` is the header and **not necessarily the segment**: four of the five
/// callers read exactly [`SEG_HEADER_LEN`] bytes, because the header is all
/// they want. So nothing here may be checked against `buf.len()` -- the bound
/// on `carry_len` belongs to [`Seg::load`], which is both the only caller
/// holding the whole file and the only code that turns `carry_len` into an
/// offset into it. Checking it here read every segment that carries a decision
/// as damaged, which is not a small mistake: `live_extent` answered `None`, so
/// the checkpoint recorded a watermark of 0 and every later open replayed the
/// whole stream again, duplicating committed rows without bound.
fn read_head(buf: &[u8], origin: u64) -> Result<(Head, bool)> {
    // Damage, always -- never a half-created file. A segment is published by
    // `store::atomic_write`: the whole header is written and `fsync`ed into a
    // temp file, and only then renamed into place, so at every instant a crash
    // can observe, a segment that exists is at least a full header. Rewriting
    // this file from scratch would therefore not be repairing a partial
    // creation, it would be discarding however many records the segment used to
    // hold -- and it did exactly that: an active segment cut below 64 bytes
    // replayed zero records and answered `count() = 0` with `Ok`.
    if (buf.len() as u64) < SEG_HEADER_LEN {
        return Err(Error::corruption(format!(
            "a log segment of {} bytes is shorter than its {SEG_HEADER_LEN}-byte header. A \
             segment is published whole by a write-fsync-rename, so no crash can leave one \
             short: this file has been damaged since, and whatever records it held are in it \
             and not anywhere else",
            buf.len()
        )));
    }
    let mut r = Reader::new(buf);
    format::read_header(&mut r)?;
    let head = Head {
        carry_len: r.u32()?,
        next_seq: r.u64()?,
        prev_seq: r.u64()?,
        prev_ms: r.u64()?,
        durable: r.u64()?,
        acked: r.u64()?,
    };
    let want = r.u64()?;
    let fixed: &[u8; SEG_HEADER_LEN as usize] =
        buf[..SEG_HEADER_LEN as usize].try_into().expect("checked above");
    Ok((head, head_check(fixed, origin) == want))
}

/// A segment is named for the stream position of its first body byte,
/// zero-padded so that lexical order is numeric order. That is what lets a
/// directory listing alone prove the archive has no hole -- see [`segments`] --
/// with no file opened.
fn seg_name(origin: u64) -> String {
    format!("seg_{origin:020}.{SEG_EXT}")
}

fn parse_seg_origin(name: &str) -> Option<u64> {
    let digits = name.strip_prefix("seg_")?.strip_suffix(SEG_EXT)?.strip_suffix('.')?;
    (digits.len() == 20 && digits.bytes().all(|b| b.is_ascii_digit()))
        .then(|| digits.parse().ok())
        .flatten()
}

/// Every segment origin in `dir`, ascending. The last is the active one.
fn seg_list(dir: &Path) -> Result<Vec<u64>> {
    let rd = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(store::io_err("read directory", dir, e)),
    };
    let mut out = Vec::new();
    for e in rd {
        let e = e.map_err(|e| store::io_err("read directory entry in", dir, e))?;
        if let Some(o) = e.file_name().to_str().and_then(parse_seg_origin) {
            out.push(o);
        }
    }
    out.sort_unstable();
    Ok(out)
}

/// One segment, loaded.
struct Seg {
    origin: u64,
    head: Head,
    /// The stamp verified, so `durable` may be believed.
    trusted: bool,
    /// Not the newest file in the directory, so it can have no torn tail.
    sealed: bool,
    buf: Vec<u8>,
}

impl Seg {
    fn load(dir: &Path, origin: u64, sealed: bool) -> Result<Seg> {
        let path = dir.join(seg_name(origin));
        let buf = std::fs::read(&path).map_err(|e| store::io_err("read", &path, e))?;
        let (head, trusted) = read_head(&buf, origin).map_err(|e| store::prefix(&path, e))?;
        // Here and nowhere else: this is the only place that holds the whole
        // segment, and `floor()` is about to turn `carry_len` into an offset
        // into `buf`. A header claiming more carried decisions than the file
        // holds would walk past them into records, or past the end.
        if head.carry_len as u64 > buf.len() as u64 - SEG_HEADER_LEN {
            return Err(store::prefix(
                &path,
                Error::corruption(format!(
                    "a segment of {} bytes declares {} bytes of carried decisions",
                    buf.len(),
                    head.carry_len
                )),
            ));
        }
        Ok(Seg { origin, head, trusted, sealed, buf })
    }

    fn path(&self, dir: &Path) -> PathBuf {
        dir.join(seg_name(self.origin))
    }

    /// The file offset up to which a failure is *rot* rather than a tear.
    ///
    /// See the module docs for why the clamp by the last non-zero byte is
    /// required and is not a heuristic: it removes exactly the shape in which
    /// `durable` can over-claim (appended bytes that never reached the
    /// platter) and nothing else. A sealed segment is not clamped at all --
    /// its successor's existence proves its `fsync` returned.
    fn effective(&self) -> u64 {
        if !self.trusted {
            // The stamp is unreadable, so nothing about this file may be
            // dismissed as an interrupted append.
            return u64::MAX;
        }
        let declared = self.at(self.head.durable);
        match self.sealed {
            true => declared,
            false => declared.min(data_end(&self.buf)),
        }
    }

    /// The file offset below which bytes are *provably* on the platter, so a
    /// walk that stops short of it has lost acknowledged data however the tail
    /// looks. A sealed segment's whole extent qualifies: its `fsync` returned
    /// before its successor was published.
    fn floor(&self) -> u64 {
        match (self.trusted, self.sealed) {
            (false, _) => SEG_HEADER_LEN,
            (true, true) => self.at(self.head.durable),
            (true, false) => self.at(self.head.acked),
        }
    }

    /// The file offset of a stream position in this segment.
    fn at(&self, lsn: u64) -> u64 {
        lsn.saturating_sub(self.origin) + SEG_HEADER_LEN
    }

    /// Whether a frame failure at file offset `at` is an interrupted append
    /// rather than damage behind a record the log already accepted.
    ///
    /// Two clauses, and the second is the one that makes this a *repair* of
    /// the old positional rule rather than a replacement for it.
    ///
    ///   * `at >= effective` -- the frame starts at or after the last byte
    ///     that could have been acknowledged. Nothing below it is at stake.
    ///   * the frame runs past the end of the data, **and** the extent it
    ///     claims is one the stamp says was acknowledged. This is the old
    ///     rule, and it is what keeps the commonest crash shape recoverable:
    ///     a block the filesystem allocated and never wrote back leaves zeros
    ///     part-way through a frame, and a short file leaves the same shape
    ///     with the zeros absent. Both must replay their prefix rather than
    ///     become an unopenable table.
    ///
    /// The bound is what closes the hole. A corrupted length can always claim
    /// to reach the end of the data -- that is the whole defect -- but after
    /// an `fsync` that returned, "the end of the data" and "the end of what
    /// the stamp acknowledges" are the same offset, so a claim that clears the
    /// first necessarily fails the second. The forged extent is only believed
    /// in the window between them, which is exactly the region no `fsync` has
    /// returned for.
    ///
    /// A sealed segment takes the first clause alone: its successor's
    /// existence proves its `fsync` returned, so it has no such window.
    ///
    /// So does a segment whose stamp does not verify, and that is not a detail.
    /// The second clause is the old positional rule *bounded by the stamp*, and
    /// with an untrustworthy bound it is just the old positional rule -- which a
    /// forged length can always satisfy, which is the defect this format version
    /// exists to remove. Reading `durable` here after `effective` has already
    /// decided it cannot be believed reinstated it: zero-filling a segment from
    /// byte 41 (inside `durable`, so the check fails and the surviving low byte
    /// leaves a small plausible watermark) replayed **0 of 6 acknowledged rows
    /// and reported success**. An unverifiable stamp forgives nothing.
    fn tear(&self, at: u64) -> bool {
        if at >= self.effective() {
            return true;
        }
        if self.sealed || !self.trusted {
            return false;
        }
        let mut r = Reader::new(&self.buf);
        if r.seek(at as usize).is_err() {
            return true;
        }
        let Ok(len) = r.varint() else { return true };
        if r.u64().is_err() {
            return true;
        }
        let Some(end) = (r.pos() as u64).checked_add(len) else { return true };
        end > data_end(&self.buf) && end <= self.at(self.head.durable)
    }

    fn stream_end(&self) -> u64 {
        self.origin + self.buf.len() as u64 - SEG_HEADER_LEN
    }
}

/// One past the last non-zero byte.
///
/// A crash can leave the tail of a segment as a run of zeros rather than as a
/// short file: the filesystem allocated a block and never wrote it back, so
/// what survives is a hole. Those zeros are not a record and never were one --
/// a run of them cannot be a frame we wrote, since every body starts with a
/// tag and the checksum stored for an empty body is not zero.
fn data_end(buf: &[u8]) -> u64 {
    buf.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1) as u64
}

/// Walk the frames of one segment from file offset `from`, classifying any
/// failure by the recorded watermark rather than by where it sits.
///
/// Returns the file offset the walk ended at: the end of the last intact
/// record, or the start of the torn tail. This is the *only* place in the
/// module that decides tear-versus-rot, and every reader goes through it.
fn walk_seg<'a>(
    seg: &'a Seg,
    from: u64,
    mut f: impl FnMut(u64, &'a [u8]) -> Result<()>,
) -> Result<u64> {
    let mut r = Reader::new(&seg.buf);
    let start = from.max(SEG_HEADER_LEN).min(seg.buf.len() as u64) as usize;
    r.seek(start)?;
    let mut end = start as u64;
    while !r.is_empty() {
        let at = r.pos() as u64;
        match format::read_framed(&mut r) {
            Ok(body) => {
                f(at, body)?;
                end = r.pos() as u64;
            }
            // An interrupted append, which was never acknowledged. Stop,
            // silently, exactly as before.
            Err(_) if seg.tear(at) => {
                end = at;
                break;
            }
            // Otherwise the log damaged itself behind a record it had already
            // accepted, which no append can do.
            Err(e) => return Err(rot(seg, at, seg.effective(), e)),
        }
    }
    // A segment that stops short of what its `fsync` already returned for has
    // lost acknowledged records, whatever the tail looks like -- which is the
    // half a `data_end` clamp on its own would forgive, right up to "the whole
    // body is a hole, so nothing was ever written".
    let floor = seg.floor();
    if end < floor {
        return Err(Error::corruption(format!(
            "the log segment replays to byte {end} but {floor} bytes of it were \
             acknowledged by an `fsync` that returned. {} bytes of committed records are \
             unaccounted for",
            floor - end
        )));
    }
    Ok(end)
}

fn hole(path: &Path, ends_at: u64, next: u64) -> Error {
    Error::corruption(format!(
        "the WAL has a hole: {} ends at stream position {ends_at}, but the next segment \
         starts at {next}. A recovery that spans the hole would silently skip whatever \
         those records held",
        path.display()
    ))
}

fn rot(seg: &Seg, at: u64, effective: u64, e: Error) -> Error {
    Error::corruption(format!(
        "damage at byte {at} of a {}sealed log segment ({e}); the segment declares \
         {} bytes acknowledged, so {} bytes of already-committed records sit behind \
         the damage. Stopping here would discard them silently",
        if seg.sealed { "" } else { "un" },
        effective,
        effective.saturating_sub(at)
    ))
}

/// What the open-time scan learned about the active segment.
#[derive(Clone, Copy, Default)]
struct Scanned {
    /// File offset one past the last structurally intact record.
    good: u64,
    /// First staging sequence number free to hand out.
    next_seq: u64,
    /// The segment holds a [`TAG_DECIDE`] another log may cite.
    decides: bool,
    /// The segment ends in records behind its last tick -- an append a crash
    /// caught before its `fsync`. Replay applies those records like any other
    /// (a plain append is part of the log's history the instant it is framed),
    /// so a recovery has to be able to place them in time too.
    dirty: bool,
    span: Span,
}

pub struct Wal {
    /// `<root>/.wal/<db>/<table>`.
    dir: PathBuf,
    /// The active segment.
    path: PathBuf,
    /// Append handle. `O_APPEND`, so a record can never land anywhere but the
    /// end however many writers share the descriptor.
    file: File,
    /// A second handle on the same inode, *not* `O_APPEND`, for the one
    /// `pwrite` that updates the header. POSIX says a positional write on an
    /// `O_APPEND` descriptor appends anyway, so the record handle cannot be
    /// reused for it.
    stamp: File,
    /// Stream position of this segment's first body byte -- the number in its
    /// name.
    origin: u64,
    /// One past the last byte, in stream positions: the LSN the next record
    /// will get.
    len: u64,
    /// The header exactly as it is on disk.
    ///
    /// A copy rather than a set of loose fields, because the stamp `pwrite`
    /// rewrites `durable` **and the check that covers the other fields**: a
    /// check computed over an in-memory `next_seq` that has advanced since the
    /// header was written would not match the bytes in the file, and the
    /// segment would read as untrusted from the next open onwards. The
    /// in-memory `next_seq` deliberately runs ahead; the header's is the one
    /// the last roll wrote, and that is the one a scan tops up.
    head: Head,
    /// The stream position the last checkpoint folded through, which is this
    /// segment's origin until the next roll. What [`Wal::pending`] measures
    /// against.
    folded: u64,
    /// Next sequence number [`Wal::begin`] will hand out.
    next_seq: u64,
    /// Groups begun and not yet committed, decided or rewound. [`Wal::roll`]
    /// refuses while this is non-zero, which turns "no transaction spans a
    /// roll" from five call-site facts into one assertion at the only place
    /// that could violate them.
    open_groups: u32,
    /// Whether the active segment holds a [`TAG_DECIDE`] another log may cite.
    decides: bool,
    /// The ticks this segment covers.
    span: Span,
    /// Records behind the last tick. What keeps [`Wal::sync`] from growing the
    /// log every time it is called on an idle table.
    dirty: bool,
    /// Sealed segment origins, ascending. Read once at open and maintained by
    /// [`Wal::roll`], so retention does no directory I/O at all.
    sealed: Vec<u64>,
    /// The buffer every record is framed in, reused across appends.
    ///
    /// The standing rule is that nothing is allocated per record on the append
    /// path, and a `Body` that owns its buffer breaks it once per record --
    /// which on a granule-sized insert is a 128 kB `malloc` and the matching
    /// `free`, i.e. an `mmap`/`munmap` pair and a fresh page fault on every
    /// byte written. Handing the same buffer out and taking it back afterwards
    /// costs a `Vec` move and makes the steady state zero.
    ///
    /// It is taken out ([`std::mem::take`]) rather than borrowed because
    /// [`Wal::append`] needs `&mut self` while the body is alive. The cost is
    /// that a body abandoned on an error path (a record over [`MAX_FRAME`])
    /// leaves an empty buffer behind and the next append re-allocates once;
    /// that is the price of a path that has already failed.
    ///
    /// One buffer per open log, sized to the largest record that log has
    /// written. Peak memory is unchanged -- that buffer existed anyway -- only
    /// its lifetime is.
    scratch: Writer,
    /// Test-only: make [`Wal::barrier`] report a device that refused to flush.
    ///
    /// There is no portable way to fail an `fsync` on demand from outside the
    /// process -- a file-size rlimit fails the *write*, which is a different
    /// path -- and the one thing two-phase commit turns on is what happens
    /// when a participant's barrier fails: the decision must not be written.
    /// A `bool` behind `cfg(test)` is the smallest thing that can assert it.
    #[cfg(test)]
    refuse_barrier: bool,
}

impl Wal {
    /// Open (or create) the log in directory `dir`.
    ///
    /// The active segment is the highest-origin file present; an empty
    /// directory gets `seg_0`. Both cases cost one `read_dir`, which is what
    /// the old layout already paid to resume the recovery LSN counter.
    pub fn open(dir: &Path) -> Result<Wal> {
        std::fs::create_dir_all(dir).map_err(|e| store::io_err("create directory", dir, e))?;
        let mut origins = seg_list(dir)?;
        if origins.is_empty() {
            store::atomic_write(&dir.join(seg_name(0)), &Head::default().encode(0))?;
            origins.push(0);
        }
        let origin = origins.pop().expect("just ensured non-empty");
        let seg = Seg::load(dir, origin, false)?;
        let path = seg.path(dir);
        let scanned = Self::scan(&seg).map_err(|e| store::prefix(&path, e))?;

        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .open(&path)
            .map_err(|e| store::io_err("open", &path, e))?;
        let stamp = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| store::io_err("open", &path, e))?;

        let mut w = Wal {
            dir: dir.to_path_buf(),
            path,
            file,
            stamp,
            origin,
            len: seg.stream_end(),
            head: seg.head,
            folded: origin,
            next_seq: seg.head.next_seq.max(scanned.next_seq),
            open_groups: 0,
            decides: scanned.decides,
            span: scanned.span,
            dirty: scanned.dirty,
            sealed: origins,
            scratch: Writer::new(),
            #[cfg(test)]
            refuse_barrier: false,
        };
        // A crash can leave a half-written record at the tail. Replay
        // tolerates that and stops, but *appending* after it would not: the
        // new record would sit behind bytes that never parse, so the next
        // replay would stop before it and silently lose an acknowledged write.
        // Discard everything after the last intact boundary, and bring the
        // stamp down with it so the truncated file does not read as short.
        if scanned.good < seg.buf.len() as u64 {
            let keep = origin + scanned.good - SEG_HEADER_LEN;
            w.file
                .set_len(scanned.good)
                .map_err(|e| store::io_err("truncate the torn tail of", &w.path, e))?;
            w.len = keep;
            let down = keep.min(w.head.durable);
            w.put_stamp(down)?;
            w.file.sync_all().map_err(|e| store::io_err("fsync", &w.path, e))?;
            store::sync_dir(dir)?;
        }
        // A recovery LSN is only unique if it is never handed out twice, and a
        // checkpoint rolls away the segment that would otherwise remember the
        // last one -- so a segment with no tick of its own resumes from the
        // `prev_seq` its header carries. No second file, no directory read:
        // the number is in the header this open already parsed.
        observe_commit(w.span.last_seq.max(w.head.prev_seq));
        Ok(w)
    }

    /// Log an insert. Not durable until [`Wal::sync`]. Returns its LSN.
    pub fn append_insert(&mut self, block: &Block) -> Result<u64> {
        self.put_insert(None, block)
    }

    /// Log a delete by primary-key lane. Not durable until [`Wal::sync`].
    /// Returns its LSN.
    pub fn append_delete(&mut self, key_lane: u64) -> Result<u64> {
        self.put_deletes(None, std::slice::from_ref(&key_lane))
    }

    /// Log one delete per lane, in one write. Returns the run's LSN.
    ///
    /// A *run*, not a record each: the tag and the staging sequence number are
    /// identical in every one, and so is the framing overhead, so a batch is
    /// `tag | count | lanes` and the per-lane cost drops from 19 bytes to 8.
    /// A bulk `DELETE` logs one of these per hidden row, and 50 000 rows go
    /// from 900 000 bytes to 400 097 in the same seven `write_all`s.
    pub fn append_deletes(&mut self, lanes: &[u64]) -> Result<u64> {
        self.put_deletes(None, lanes)
    }

    /// [`Wal::append_deletes`], staged under `seq`.
    pub fn append_deletes_staged(&mut self, seq: u64, lanes: &[u64]) -> Result<u64> {
        self.put_deletes(Some(seq), lanes)
    }

    /// Log the positions a sweep hid inside citable parts: one
    /// `TAG_MASK_RUN` per part. `seq` stages the group as everywhere else.
    ///
    /// The deltas arrive already encoded, in the buffer the sweep filled as it
    /// went, so nothing here walks a per-row collection: this is a tag, two
    /// varints and one `extend_from_slice` per *part*.
    pub fn append_masks(&mut self, seq: Option<u64>, masks: &MaskRuns) -> Result<u64> {
        let lsn = self.len;
        for (pid, n, deltas) in masks.runs() {
            let mut body = self.body(deltas.len() + 32);
            put_tag(&mut body.w, TAG_MASK_RUN, seq);
            body.w.varint(pid);
            body.w.varint(n as u64);
            body.w.raw(deltas);
            self.append(body)?;
        }
        Ok(lsn)
    }

    /// Open a staging group. See the module docs: records logged under the
    /// returned sequence number stay invisible to [`Wal::replay`] until
    /// [`Wal::commit`] is given the same number.
    ///
    /// Cheap and infallible -- nothing is written. The group exists only in the
    /// records that carry its number, and in the counter that stops a roll
    /// happening underneath it.
    pub fn begin(&mut self) -> u64 {
        let seq = self.next_seq;
        // A log that appends 10^9 groups a second overflows this in 500 years.
        self.next_seq += 1;
        self.open_groups += 1;
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
        self.put_deletes(Some(seq), std::slice::from_ref(&key_lane))
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
        let at = self.marker(TAG_COMMIT, seq);
        self.close_group();
        at
    }

    /// Release the group staged under `seq` **if** the log in directory
    /// `coordinator` commits `coord_seq`. The earlier participants of a
    /// multi-table transaction log this instead of [`Wal::commit`]; see the
    /// module docs.
    ///
    /// The transaction's fate is one record in one file, so a crash anywhere
    /// in the sequence -- including after this record is `fsync`ed -- resolves
    /// the same way for every participant. That is the whole guarantee, and it
    /// is why the *order* matters: every prepare must be durable before the
    /// decision is written, or the decision would commit a participant whose
    /// rows are not on disk yet.
    ///
    /// `coordinator` is stored relative to this log's own directory, not
    /// absolutely: a data directory that is copied, restored from a backup or
    /// simply moved must still resolve, and an absolute path baked into a log
    /// record is a path that stops being true the moment the tree is moved. It
    /// names the *directory*, because a segment's file name changes at every
    /// roll and the directory does not.
    pub fn prepare(&mut self, seq: u64, coordinator: &Path, coord_seq: u64) -> Result<u64> {
        let rel = relative_dir(&self.dir, coordinator)?;
        let mut body = self.body(rel.len() + 24);
        body.w.u8(TAG_PREPARE);
        body.w.varint(seq);
        body.w.varint(coord_seq);
        body.w.str(&rel);
        let at = self.append(body);
        self.close_group();
        at
    }

    /// The decision the prepares cite, and the last participant's own commit
    /// marker. Not durable until [`Wal::sync`] -- and *that* `fsync`, the last
    /// of the transaction, is the instant the whole thing commits.
    pub fn decide(&mut self, seq: u64) -> Result<u64> {
        let at = self.marker(TAG_DECIDE, seq)?;
        self.decides = true;
        self.close_group();
        Ok(at)
    }

    fn marker(&mut self, tag: u8, seq: u64) -> Result<u64> {
        let mut body = self.body(16);
        body.w.u8(tag);
        body.w.varint(seq);
        self.append(body)
    }

    /// One group is resolved. Saturating rather than asserting: a caller that
    /// commits the same group twice is a bug, but turning it into a panic
    /// inside the durability core is a worse one.
    fn close_group(&mut self) {
        self.open_groups = self.open_groups.saturating_sub(1);
    }

    /// Abandon a staging group without a marker and without a rewind.
    ///
    /// The one legitimate caller is the fold-on-commit path: a table whose
    /// COMMIT folds into parts never writes a marker, because the records the
    /// group staged are going into the parts instead and replay is supposed to
    /// drop them. Saying so explicitly is what keeps [`Wal::roll`]'s guard
    /// meaningful -- otherwise every unkeyed `DELETE ... COMMIT` would leave a
    /// phantom open group and the guard would refuse the fold that follows it.
    pub fn drop_group(&mut self) {
        self.close_group();
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
    /// The floor is `origin + carry_len`, not `origin`: the carried decision
    /// markers sit in that interval and a rewind through them would make every
    /// sibling's prepare read "no decision", which is abort, which silently
    /// drops a transaction that committed -- in the *other* tables. Not
    /// reachable from today's call sites, which is exactly why the check has
    /// to be right the first time. It is unreachable for a second reason too:
    /// no transaction can span a roll, because [`Wal::roll`] refuses while a
    /// group is open.
    ///
    /// A no-op when `lsn` is at or past the end, so a transaction that logged
    /// nothing costs nothing here.
    pub fn rewind_to(&mut self, lsn: u64) -> Result<()> {
        self.close_group();
        if lsn >= self.len {
            return Ok(());
        }
        let floor = self.origin + self.head.carry_len as u64;
        if lsn < floor {
            return Err(Error::storage(format!(
                "cannot rewind the log to {lsn}: the active segment's records start at \
                 {floor}, and the bytes below that are decision markers other tables' \
                 prepares still cite"
            )));
        }
        self.file
            .set_len(lsn - self.origin + SEG_HEADER_LEN)
            .map_err(|e| store::io_err("rewind", &self.path, e))?;
        self.len = lsn;
        // The stamp has to come down with the file: staged records *are*
        // fsynced, so `durable` can cover bytes this call is removing, and
        // leaving it high would make the very next open report a healthy
        // truncated file as corruption.
        self.put_stamp(self.head.durable.min(lsn))?;
        self.file
            .sync_all()
            .map_err(|e| store::io_err("fsync", &self.path, e))
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
        let mut body = self.body(block.bytes() + 64);
        put_tag(&mut body.w, TAG_INSERT, seq);
        super::writer::put_block(&mut body.w, block);
        if body.len() as u64 > MAX_FRAME {
            return Err(Error::storage(format!(
                "a log record of {} bytes cannot be written: the format's limit is \
                 {MAX_FRAME} bytes",
                body.len()
            )));
        }
        self.append(body)
    }

    /// Frame the lanes into runs and write them.
    ///
    /// Chunked, so the staging buffer is bounded by the constant rather than
    /// by the statement: a million-row `DELETE` would otherwise build an 8 MB
    /// `Vec` to hand to one `write`. A single lane keeps the plain
    /// [`TAG_DELETE`] record it always had, so a point delete is byte-for-byte
    /// what it was.
    fn put_deletes(&mut self, seq: Option<u64>, lanes: &[u64]) -> Result<u64> {
        let Some((&first, rest)) = lanes.split_first() else { return Ok(self.len) };
        if rest.is_empty() {
            let mut body = self.body(24);
            put_tag(&mut body.w, TAG_DELETE, seq);
            body.w.u64(first);
            return self.append(body);
        }
        let lsn = self.len;
        for chunk in lanes.chunks(DELETE_BATCH) {
            let mut body = self.body(chunk.len() * 8 + 24);
            put_tag(&mut body.w, TAG_DELETE_RUN, seq);
            body.w.varint(chunk.len() as u64);
            // `Writer::u64` is little-endian; this has to stay in step with
            // it, since `decode_entry` reads the two forms with one decoder.
            for &l in chunk {
                body.w.u64(l);
            }
            self.append(body)?;
        }
        Ok(lsn)
    }

    /// Lend the framing buffer out to build one record in.
    ///
    /// Pair every call with [`Wal::append`], which gives the buffer back.
    fn body(&mut self, cap: usize) -> Body {
        Body::new(std::mem::take(&mut self.scratch), cap)
    }

    /// Frame and append a body built by [`Body`], returning the record's LSN.
    ///
    /// The frame header is written *in front of* the body, inside the same
    /// allocation, so there is no second buffer and no copy of the payload.
    /// The buffer then goes back to [`Wal::scratch`] with its capacity intact,
    /// including when the write failed -- a failed append is exactly when the
    /// next one should not also have to allocate.
    fn append(&mut self, mut body: Body) -> Result<u64> {
        let r = self.append_framed(body.finish());
        self.scratch = body.w;
        r
    }

    /// Write already-framed bytes and return the LSN they start at.
    fn append_framed(&mut self, bytes: &[u8]) -> Result<u64> {
        let lsn = self.len;
        // One `write_all` per call, never one per record split across two: a
        // record split across two syscalls could be interleaved with another
        // writer's, and framing cannot recover from that the way it recovers
        // from a short tail.
        if let Err(e) = self.file.write_all(bytes) {
            // A `write_all` that fails part-way still wrote what it wrote --
            // ENOSPC on a 12 MiB volume left 53236 bytes behind from a call
            // that returned `Err`. The file is O_APPEND, so leaving `self.len`
            // at the pre-write value does not merely mis-report the length: a
            // later successful append lands at the real end of the file while
            // the LSN handed back names the torn offset, and `rewind_to` then
            // sees `lsn >= self.len` and declines to truncate. One `fstat` on
            // a path that has already failed restores the invariant the rest
            // of the type is written against, and the original error is what
            // the caller still gets.
            if let Ok(m) = self.file.metadata() {
                self.len = self.origin + m.len() - SEG_HEADER_LEN;
            }
            return Err(store::io_err("append to", &self.path, e));
        }
        self.len += bytes.len() as u64;
        self.dirty = true;
        Ok(lsn)
    }

    /// Make every appended record durable, and stamp it.
    ///
    /// The tick goes in *before* the `fsync`, so it is durable with exactly
    /// the records it covers -- a record whose tick did not make it was never
    /// acknowledged, and a point-in-time recovery must not resurrect it. The
    /// stamp goes in after the tick and before the `fsync` for the same
    /// reason: when the `fsync` returns, `durable` and the bytes it describes
    /// are durable together.
    ///
    /// A `sync` with nothing appended behind it writes nothing at all: an idle
    /// table that is `sync`ed in a loop grows no log and re-stamps no header.
    pub fn sync(&mut self) -> Result<()> {
        self.stage_sync()?;
        self.barrier()
    }

    /// Everything [`Wal::sync`] does except the barrier: the tick, and the
    /// stamp that says how far the barrier will make this segment durable.
    ///
    /// Split out for the one caller that has several logs to make durable at
    /// once. The bytes have to be written and the header stamped from the
    /// thread that owns the log -- both mutate it -- but the `fsync` that
    /// follows does not, and on this platform an `fsync` is a *device*
    /// operation whose cost several files can share. See
    /// [`Wal::barrier`].
    pub fn stage_sync(&mut self) -> Result<()> {
        if self.dirty {
            self.tick()?;
        }
        self.put_stamp(self.len)
    }

    /// The barrier alone: make everything [`Wal::stage_sync`] staged durable.
    ///
    /// `&self`, which is the whole point -- `File::sync_all` needs no mutable
    /// access, so N logs staged by one thread can be flushed concurrently by
    /// N. Nothing in the log's own state changes here: the stamp describing
    /// what this call makes durable was written before it, deliberately, so
    /// that when it returns the stamp and the bytes it describes are durable
    /// together.
    ///
    /// Calling this without a preceding `stage_sync` is not unsafe, merely
    /// useless: it would flush bytes the header does not yet claim.
    pub fn barrier(&self) -> Result<()> {
        #[cfg(test)]
        if self.refuse_barrier {
            return Err(store::io_err(
                "fsync",
                &self.path,
                std::io::Error::from_raw_os_error(19),
            ));
        }
        self.file.sync_all().map_err(|e| store::io_err("fsync", &self.path, e))
    }

    /// Test-only: from here on, [`Wal::barrier`] fails. See
    /// [`Wal::refuse_barrier`].
    #[cfg(test)]
    pub(crate) fn refuse_barriers(&mut self) {
        self.refuse_barrier = true;
    }

    /// Record that this segment is acknowledged through stream position `to`.
    ///
    /// One 16-byte positional write, and only when the number actually
    /// changed. It rides in front of an `fsync` the caller was already going
    /// to pay for, so it is not a durability barrier of its own.
    fn put_stamp(&mut self, to: u64) -> Result<()> {
        if to == self.head.durable {
            return Ok(());
        }
        // Every caller `fsync`s immediately after this, so the value being
        // replaced is one whose `fsync` returned and is provably on disk --
        // which is what makes it a floor. A rewind lowers both, because the
        // bytes it discards are going away.
        self.head.acked = match to > self.head.durable {
            true => self.head.durable,
            false => self.head.acked.min(to),
        };
        self.head.durable = to;
        pwrite(&self.stamp, STAMP_AT, &self.head.stamp(self.origin))
            .map_err(|e| store::io_err("stamp", &self.path, e))
    }

    fn tick(&mut self) -> Result<()> {
        let seq = NEXT_COMMIT.fetch_add(1, Ordering::AcqRel);
        // Non-decreasing per log even if the system clock steps backwards: a
        // recovery navigates this column with a binary decision per tick, and
        // a column that goes backwards would make the cut ambiguous. The
        // recovery LSN beside it is exact regardless.
        let ms = now_ms().max(self.span.last_ms).max(self.head.prev_ms);
        let mut body = self.body(24);
        body.w.u8(TAG_TICK);
        body.w.varint(seq);
        body.w.varint(ms);
        self.append(body)?;
        self.span.observe(seq, ms);
        self.dirty = false;
        Ok(())
    }

    /// Seal the active segment and start a fresh one. Called by a checkpoint
    /// once the records are inside parts.
    ///
    /// Nothing is moved, linked, copied or renamed to archive anything: the
    /// segment being sealed *is* its own archive file already, and it becomes
    /// part of the archive by the successor's arrival. That is the whole of
    /// the hard-link problem, solved by construction rather than by a
    /// fallback.
    ///
    /// Decision records are the one thing carried across. Another table's log
    /// may still hold a prepare citing one, and losing it would make that
    /// prepare unresolvable -- indistinguishable from a decision that was
    /// never written, which is an abort, which would silently drop a
    /// transaction that committed. They go into the successor's own
    /// `atomic_write`, in the same file and the same `fsync` as its header,
    /// because a crash between "successor exists" and "carried decisions
    /// durable" is exactly the state that loses them.
    ///
    /// A log that has never written a decision -- every log in a database that
    /// never runs a multi-table transaction -- takes none of this.
    pub fn roll(&mut self) -> Result<()> {
        if self.open_groups != 0 {
            return Err(Error::storage(format!(
                "cannot roll the log of {}: {} staging group(s) are still open, and a \
                 transaction that spanned the roll could not be rolled back",
                self.dir.display(),
                self.open_groups
            )));
        }
        let empty = self.is_empty();
        // Nothing appended, and nothing carried that could now be dropped.
        if empty && self.head.carry_len == 0 {
            return Ok(());
        }
        let carry = match self.decides && self.may_be_cited() {
            true => self.decisions()?,
            false => Vec::new(),
        };
        // The segment holds its carried decisions and nothing else, and they
        // are still citable: minting an identical successor at every
        // checkpoint would be pure churn. (Where they are *not* still citable
        // the roll goes ahead, because dropping them is the point.)
        if empty && !carry.is_empty() {
            return Ok(());
        }
        // Everything in the segment has to sit behind a tick, or a recovery
        // has no way to place it in time and would leave it out. Not a
        // formality: a plain append is part of the log's history the instant
        // it is framed, so a writer a crash caught between the append and its
        // `fsync` leaves records the *next* open replays like any other, while
        // the tick that would have stamped them was never written. This is the
        // stamp that says so.
        //
        // The `fsync` is conditional on the same flag, and that is not an
        // oversight: with no tick to add there are no bytes here that are not
        // already on the platter, because every acknowledged write already
        // fsynced this file. An `fsync` is ~3.8 ms and this is the ordinary
        // path.
        if self.dirty {
            self.tick()?;
            self.put_stamp(self.len)?;
            self.file.sync_all().map_err(|e| store::io_err("fsync", &self.path, e))?;
        }
        let origin = self.len;
        let mut w = Writer::with_capacity(SEG_HEADER_LEN as usize + 16 * carry.len());
        let mut carried = Writer::with_capacity(16 * carry.len());
        for &s in &carry {
            let mut body = Writer::with_capacity(11);
            body.u8(TAG_DECIDE);
            body.varint(s);
            format::write_framed(&mut carried, body.as_slice());
        }
        let carried = carried.finish();
        let head = Head {
            carry_len: carried.len() as u32,
            next_seq: self.next_seq,
            prev_seq: self.span.last_seq.max(self.head.prev_seq),
            prev_ms: self.span.last_ms.max(self.head.prev_ms),
            durable: origin + carried.len() as u64,
            acked: origin + carried.len() as u64,
        };
        w.raw(&head.encode(origin));
        w.raw(&carried);
        let path = self.dir.join(seg_name(origin));
        // Two full barriers for a 64-byte header, and left that way on
        // purpose. There is an argument that they are droppable -- if the
        // successor's directory entry is lost to a crash, replay opens the
        // sealed predecessor, whose `stream_end` is the checkpoint watermark,
        // and correctly finds nothing to replay -- but it turns on
        // `Wal::open`'s segment discovery agreeing, and on nothing ever having
        // been appended to the successor first. When a segment carries
        // decisions it is also the only thing keeping them alive. Rolls are
        // rare (one per checkpoint, and much rarer now that a citable unkeyed
        // sweep no longer folds), so this is the wrong place to spend the
        // audit. Recorded rather than done.
        store::atomic_write(&path, &w.finish())?;

        self.file = OpenOptions::new()
            .read(true)
            .append(true)
            .open(&path)
            .map_err(|e| store::io_err("open", &path, e))?;
        self.stamp = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| store::io_err("open", &path, e))?;
        self.sealed.push(self.origin);
        self.path = path;
        self.origin = origin;
        self.folded = origin;
        self.head = head;
        self.len = head.durable;
        self.decides = !carry.is_empty();
        self.dirty = false;
        self.span = Span::default();
        self.prune()
    }

    /// Drop whole sealed segments, oldest first, until the archive fits both
    /// the byte budget and [`MAX_SEGMENTS`].
    ///
    /// No `read_dir` and no `stat`: the listing is cached and a segment's size
    /// is the difference between its origin and its successor's, both of which
    /// are file names. The safety valve -- never drop a segment that ends past
    /// what the checkpoint has committed -- can never fire in a healthy
    /// database, because a roll is the only thing that seals and it happens
    /// inside the checkpoint that commits the parts. That is exactly why it
    /// should exist: it is the assertion that catches a future roll trigger
    /// added by someone who has not read this paragraph.
    fn prune(&mut self) -> Result<()> {
        let budget = archive_retention();
        let mut total = self.sealed_bytes();
        let mut dropped = 0usize;
        while dropped < self.sealed.len() {
            let over = total > budget || self.sealed.len() - dropped > MAX_SEGMENTS;
            if !over {
                break;
            }
            let o = self.sealed[dropped];
            let end = self.sealed.get(dropped + 1).copied().unwrap_or(self.origin);
            if end > self.folded {
                break;
            }
            let p = self.dir.join(seg_name(o));
            if std::fs::remove_file(&p).is_err() {
                break;
            }
            total -= SEG_HEADER_LEN + (end - o);
            dropped += 1;
        }
        if dropped > 0 {
            self.sealed.drain(..dropped);
            store::sync_dir(&self.dir)?;
        }
        Ok(())
    }

    /// Bytes the sealed segments occupy, from their names alone.
    fn sealed_bytes(&self) -> u64 {
        match self.sealed.first() {
            None => 0,
            Some(&first) => SEG_HEADER_LEN * self.sealed.len() as u64 + (self.origin - first),
        }
    }

    /// The sequence numbers the active segment has decided, ascending.
    fn decisions(&self) -> Result<Vec<u64>> {
        let mut out = Vec::new();
        walk_dir_newest(&self.dir, &mut out)?;
        Ok(out)
    }

    /// Whether some other table's log can still hold a prepare citing this
    /// one's decisions.
    ///
    /// The test is "does any other log hold a record its table's parts do not
    /// already cover", not "does any other log hold a prepare naming me". It
    /// is coarser on purpose: reading every sibling log to find out would cost
    /// the whole database's un-checkpointed bytes at every checkpoint, and the
    /// coarse answer is exact where it matters. A full checkpoint writes every
    /// table's parts -- and with them a watermark at the end of that table's
    /// log -- *before* it rolls any of them, so by the time this runs every
    /// sibling is covered and the decisions go. A single-table fold leaves the
    /// others uncovered, and those are exactly the ones that can still cite us.
    ///
    /// Exactness depends on `stream_end == origin + carry_len` for an idle
    /// log, which is what the fixed header buys: with the head fields in a
    /// framed record the comparison would be true by the head's own length for
    /// every table in the database and this would be a constant `true`.
    ///
    /// Conservative in every direction it cannot see: a directory it cannot
    /// read, a layout it does not recognise, or a data root with no `CATALOG`
    /// (so a log opened as a bare directory by a test can never send it
    /// walking the filesystem) all answer "yes, keep them".
    fn may_be_cited(&self) -> bool {
        let Some(root) = wal_root_of(&self.dir) else { return true };
        if !root.join(store::CATALOG_FILE).exists() {
            return true;
        }
        let Ok(dbs) = std::fs::read_dir(root.join(WAL_DIR)) else { return true };
        for db in dbs.flatten() {
            let Ok(tables) = std::fs::read_dir(db.path()) else { continue };
            for t in tables.flatten() {
                let dir = t.path();
                if dir == self.dir {
                    continue;
                }
                let Some(name) = db.file_name().to_str().map(str::to_string) else { continue };
                let Some(tn) = t.file_name().to_str().map(str::to_string) else { continue };
                let tdir = root.join(&name).join(&tn);
                let Some((end, floor)) = live_extent(&dir) else { continue };
                if end > floor.max(covered_prefix(&tdir)) {
                    return true;
                }
            }
        }
        false
    }

    /// The stream position the next record will get. **Not** the size of any
    /// file: see the module docs. [`Wal::pending`] is what a size threshold
    /// wants.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Bytes appended since the last checkpoint folded this log into parts.
    ///
    /// A field subtraction, no I/O. This is what `wal_fold_bytes` compares
    /// against: `len()` never restarts, so comparing a threshold with it would
    /// be permanently true after the first threshold's worth of log and would
    /// queue a fold on every statement forever.
    pub fn pending(&self) -> u64 {
        self.len - self.folded
    }

    /// Stream position of the active segment's first body byte -- and, after
    /// [`Wal::roll`], the exact stream end of the segment it just sealed. That
    /// identity is why a checkpoint can record this as its watermark instead
    /// of computing one.
    pub fn origin(&self) -> u64 {
        self.origin
    }

    pub fn is_empty(&self) -> bool {
        self.len == self.origin + self.head.carry_len as u64
    }

    /// The directory holding this log's segments.
    pub fn path(&self) -> &Path {
        &self.dir
    }

    /// Every record in the log, in order.
    pub fn replay(dir: &Path, schema: &Schema) -> Result<Vec<WalRecord>> {
        Self::replay_from(dir, schema, 0)
    }

    /// Every record at or after `from`, each with the LSN it was written at.
    ///
    /// The same numbers [`Wal::append_insert`] and friends returned. That
    /// correspondence is what makes the checkpoint watermark and
    /// [`Wal::rewind_to`] sound -- both name a position in this space -- so it
    /// is worth being able to observe rather than merely assert.
    pub fn replay_with_lsn(
        dir: &Path,
        schema: &Schema,
        from: u64,
    ) -> Result<Vec<(u64, WalRecord)>> {
        Self::replay_entries(dir, schema, from)
    }

    /// End offset of the last structurally intact record, the first sequence
    /// number that is free to hand out, and whether the segment holds a
    /// decision another log may cite.
    ///
    /// Framing only: a record whose frame is complete and whose checksum
    /// matches is a record we really wrote, whether or not its *body* still
    /// decodes against the current schema. Truncating on a body error would
    /// throw away durable data because of a schema mismatch, so body damage is
    /// left for `replay` to report.
    fn scan(seg: &Seg) -> Result<Scanned> {
        let mut out = Scanned { good: SEG_HEADER_LEN, ..Scanned::default() };
        out.good = walk_seg(seg, 0, |_, body| {
            out.next_seq = out.next_seq.max(body_next_seq(body));
            out.decides |= tagged(body, TAG_DECIDE).is_some();
            out.dirty = true;
            if let Some(&TAG_TICK) = body.first() {
                out.dirty = false;
                if let Some((seq, ms)) = tick_of(body) {
                    out.span.observe(seq, ms);
                }
            }
            Ok(())
        })?;
        Ok(out)
    }

    /// Every record at or after stream position `from`.
    ///
    /// `from` must be a record boundary recorded by a checkpoint; a bogus one
    /// lands mid-record and is caught by the frame checksum rather than
    /// silently yielding garbage.
    pub fn replay_from(dir: &Path, schema: &Schema, from: u64) -> Result<Vec<WalRecord>> {
        // One extra pass moving the entries, against a recovery that then
        // *inserts every block into a table*. Carrying the LSN in the
        // primitive and dropping it here is the cheap direction: the other way
        // round would mean two nearly identical replay loops, and a second
        // implementation of the staged-record filter is exactly the thing that
        // would silently stop agreeing with the first.
        Ok(Self::replay_entries(dir, schema, from)?.into_iter().map(|(_, r)| r).collect())
    }

    /// The replay primitive: records with their LSNs, across every segment the
    /// range touches.
    ///
    /// Staging groups and citations are threaded across segment boundaries in
    /// one pass, because a group can be staged in one segment and released in
    /// the next -- not by an ordinary transaction, which cannot span a roll,
    /// but by a crash that left one open across one.
    fn replay_entries(
        dir: &Path,
        schema: &Schema,
        from: u64,
    ) -> Result<Vec<(u64, WalRecord)>> {
        let origins = seg_list(dir)?;
        // The recovery LSN counter is process-global and a checkpoint rolls
        // away the ticks that would otherwise remember the last one, so a
        // process that only *reads* -- `BACKUP`, for one -- has to resume it
        // from the header before it does anything else. Getting this wrong is
        // not a small error: a backup would record a boundary that had already
        // been used, and the roll-forward after it would replay records the
        // backup already held. One 64-byte read, once per table per open.
        let end = match origins.last() {
            Some(&o) => {
                let path = dir.join(seg_name(o));
                let len = std::fs::metadata(&path).map_or(0, |m| m.len());
                let mut head = [0u8; SEG_HEADER_LEN as usize];
                if read_exact_at(&path, &mut head).is_some() {
                    if let Ok((h, _)) = read_head(&head, o) {
                        observe_commit(h.prev_seq);
                    }
                }
                o + len.saturating_sub(SEG_HEADER_LEN)
            }
            None => from,
        };
        let mut out: Vec<(u64, WalRecord)> = Vec::new();
        // Where each still-uncommitted record sits in `out`, with the sequence
        // number that would release it. This holds only genuinely staged
        // records -- at most one group per statement that failed since the last
        // checkpoint -- so the linear `retain` per commit marker is bounded by
        // the *failures*, not by the length of the log.
        let mut staged: Vec<(u64, usize)> = Vec::new();
        // The decisions of each cited log, read once. A transaction writes one
        // prepare per participant log, so this holds one entry per
        // *coordinator* this log ever prepared against.
        let mut cited: Vec<(String, Vec<u64>)> = Vec::new();
        let mut seen = 0usize;
        for (i, &origin) in origins.iter().enumerate() {
            let sealed = i + 1 < origins.len();
            // Wholly inside the prefix the checkpoint already folded.
            if sealed && origins[i + 1] <= from {
                continue;
            }
            let seg = Seg::load(dir, origin, sealed)?;
            let path = seg.path(dir);
            if sealed && seg.stream_end() != origins[i + 1] {
                return Err(hole(&path, seg.stream_end(), origins[i + 1]));
            }
            let start = from.saturating_sub(origin) + SEG_HEADER_LEN;
            let mut err = None;
            let walked = walk_seg(&seg, start, |at, body| {
                let lsn = origin + at - SEG_HEADER_LEN;
                match decode_entry(body, schema).map_err(|e| record_err(at, seen, e))? {
                    Entry::Record(seq, rec) => {
                        if let Some(s) = seq {
                            staged.push((s, out.len()));
                        }
                        out.push((lsn, rec));
                    }
                    Entry::Deletes(seq, lanes) => {
                        for l in lanes.chunks_exact(8) {
                            if let Some(s) = seq {
                                staged.push((s, out.len()));
                            }
                            let l = u64::from_le_bytes(l.try_into().expect("8 bytes"));
                            out.push((lsn, WalRecord::Delete(l)));
                        }
                    }
                    Entry::Masks(seq, pid, n, deltas) => {
                        if let Some(s) = seq {
                            staged.push((s, out.len()));
                        }
                        let pos =
                            mask_positions(n, deltas).map_err(|e| record_err(at, seen, e))?;
                        out.push((lsn, WalRecord::Mask(pid, pos)));
                    }
                    Entry::Commit(seq) => staged.retain(|&(s, _)| s != seq),
                    Entry::Prepare(seq, rel, coord_seq) => {
                        let k = match cited.iter().position(|(r, _)| r == rel) {
                            Some(k) => k,
                            None => {
                                let d =
                                    decisions_of(dir, rel).map_err(|e| record_err(at, seen, e))?;
                                cited.push((rel.to_string(), d));
                                cited.len() - 1
                            }
                        };
                        if cited[k].1.binary_search(&coord_seq).is_ok() {
                            staged.retain(|&(s, _)| s != seq);
                        }
                    }
                    // Recovery navigates by these; a replay only has to make
                    // sure the counter never hands the number out twice.
                    Entry::Tick(seq) => observe_commit(seq),
                }
                seen += 1;
                Ok(())
            });
            match walked {
                Ok(_) => {}
                Err(e) => {
                    err = Some(store::prefix(&path, e));
                }
            }
            if let Some(e) = err {
                return Err(e);
            }
        }
        // A watermark past the end of the stream used to be repaired in
        // silence -- the LSN restarted at every truncation, so it really could
        // be stale. It cannot be now: the stream never restarts, so this can
        // only mean segments are missing, and replaying nothing over a hole is
        // the exact failure this design exists to remove.
        if from > end {
            return Err(Error::corruption(format!(
                "{}: the checkpoint recorded log position {from}, but the segments present \
                 only reach {end}. {} bytes of committed records are missing, and replaying \
                 nothing over the hole would report success",
                dir.display(),
                from - end
            )));
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

/// A record body under construction, with room reserved in front of it for the
/// frame header so the header can be written *into the same allocation*.
///
/// `varint len | u64 sum | body` puts a variable-length prefix in front of a
/// body whose length is not known until it is built, which is why the previous
/// shape built the body in one `Writer` and then copied the whole thing into a
/// second one to frame it -- a malloc and a memcpy per record, 128 kB of each
/// on a granule-sized insert. Reserving the maximum header (a 10-byte varint
/// plus 8 bytes of checksum) and writing the real header so that it *ends* at
/// that offset costs one branch and no copy at all.
struct Body {
    w: Writer,
}

/// Reserved prefix: the longest `u64` varint plus the checksum.
const FRAME_MAX_HEAD: usize = 10 + 8;

impl Body {
    /// Take over a buffer and reserve the frame header in front of the body.
    ///
    /// The buffer comes from [`Wal::scratch`] via [`Wal::body`], which is what
    /// makes an append allocation-free once the log is warm; `Writer::new()`
    /// here would put the per-record `malloc` straight back.
    fn new(mut w: Writer, cap: usize) -> Body {
        w.clear();
        w.reserve(cap + FRAME_MAX_HEAD);
        w.raw(&[0u8; FRAME_MAX_HEAD]);
        Body { w }
    }

    fn len(&self) -> usize {
        self.w.as_slice().len() - FRAME_MAX_HEAD
    }

    /// The framed record, as a slice of the buffer the body was built in.
    ///
    /// The header is built on the stack, not in a second `Writer`: an
    /// eighteen-byte `Vec` is still a malloc, and this runs once per record.
    /// The LEB128 loop is therefore spelled out here rather than borrowed from
    /// [`Writer::varint`] -- routing that method through a shared stack encoder
    /// so both could use one copy measured 3.5x slower per varint (a
    /// `extend_from_slice` where a `push` used to be), and it is called for
    /// every length and count in every part the engine writes. Six lines here
    /// is the cheaper duplication, and every WAL test round-trips it.
    fn finish(&mut self) -> &[u8] {
        let sum = format::checksum(&self.w.as_slice()[FRAME_MAX_HEAD..]);
        let mut head = [0u8; FRAME_MAX_HEAD];
        let mut n = 0;
        let mut v = self.len() as u64;
        while v >= 0x80 {
            head[n] = (v as u8) | 0x80;
            n += 1;
            v >>= 7;
        }
        head[n] = v as u8;
        n += 1;
        head[n..n + 8].copy_from_slice(&sum.to_le_bytes());
        let at = FRAME_MAX_HEAD - (n + 8);
        let buf = self.w.as_mut_slice();
        buf[at..FRAME_MAX_HEAD].copy_from_slice(&head[..n + 8]);
        &buf[at..]
    }
}

/// One positional write. A second descriptor is required rather than
/// convenient: POSIX says a positional write on an `O_APPEND` descriptor
/// appends regardless of the offset.
fn pwrite(f: &File, at: u64, bytes: &[u8]) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::FileExt::write_all_at(f, bytes, at)
    }
    #[cfg(not(unix))]
    {
        use std::io::{Seek, SeekFrom};
        let mut f = f;
        f.seek(SeekFrom::Start(at))?;
        f.write_all(bytes)
    }
}

// ---------------------------------------------------------------------------
// the archive
// ---------------------------------------------------------------------------

/// Where `<db>.<table>`'s segments live under data root `root`.
pub fn wal_dir(root: &Path, db: &str, table: &str) -> PathBuf {
    root.join(WAL_DIR).join(db).join(table)
}

/// Kept under its old name for the callers that ask "where is this table's
/// archive": it is the same directory as its live log now.
pub fn archive_dir(root: &Path, db: &str, table: &str) -> PathBuf {
    wal_dir(root, db, table)
}

/// The data root a `<root>/.wal/<db>/<table>` directory belongs to.
fn wal_root_of(dir: &Path) -> Option<&Path> {
    let root = dir.parent()?.parent()?;
    (root.file_name()? == std::ffi::OsStr::new(WAL_DIR)).then(|| root.parent())?
}

/// [`live_extent`] for one table, by name.
pub fn live_extent_of(root: &Path, db: &str, table: &str) -> Option<(u64, u64)> {
    live_extent(&wal_dir(root, db, table))
}

/// `(stream end, first record position)` of a log directory's active segment,
/// from its name and its length -- one `stat` and one 56-byte read.
fn live_extent(dir: &Path) -> Option<(u64, u64)> {
    let origin = *seg_list(dir).ok()?.last()?;
    let path = dir.join(seg_name(origin));
    let len = std::fs::metadata(&path).ok()?.len();
    let mut head = [0u8; SEG_HEADER_LEN as usize];
    read_exact_at(&path, &mut head)?;
    let (h, _) = read_head(&head, origin).ok()?;
    let end = origin + len.saturating_sub(SEG_HEADER_LEN);
    // Clamped rather than believed. This is an advisory read -- one `stat` and
    // one 64-byte header, with no walk to prove anything -- so a `carry_len`
    // past the end of the file must not produce a floor above the stream end,
    // which would read as "this log holds nothing" and let a checkpoint
    // discard records. `Seg::load` refuses that segment outright the moment
    // anything actually reads it.
    Some((end, (origin + h.carry_len as u64).min(end)))
}

fn read_exact_at(path: &Path, buf: &mut [u8]) -> Option<()> {
    use std::io::Read;
    let mut f = File::open(path).ok()?;
    f.read_exact(buf).ok()
}

/// One archived segment.
#[derive(Clone, Debug)]
pub struct Segment {
    /// Stream position of its first record. Globally monotone, never
    /// restarted, and the number in the file's name.
    pub origin: u64,
    /// One past its last record's stream position -- which is its successor's
    /// origin, by construction rather than by arithmetic.
    pub end: u64,
    /// The recovery LSNs and wall-clock times it covers. `first_*` is the tick
    /// immediately *before* the segment (its own header's `prev_*`) and
    /// `last_*` is its last tick (its successor's `prev_*`), so both come from
    /// one 56-byte read at a fixed offset rather than from a scan.
    pub span: Span,
    pub path: PathBuf,
}

/// Every sealed segment of `<db>.<table>`, oldest first.
///
/// **Sealed only.** The newest file is the live log, and its tail may be an
/// interrupted append; the archive is exactly everything else.
///
/// A hole between two segments is impossible to express -- a segment's end
/// *is* its successor's origin -- so what is checked instead is the pair of
/// things that can actually go wrong: a segment whose file length disagrees
/// with the gap its name and its successor's name leave, and a segment whose
/// `durable` is short of its file length. Either means records are gone, and
/// replaying across that is the one failure a recovery feature must never
/// have.
pub fn segments(root: &Path, db: &str, table: &str) -> Result<Vec<Segment>> {
    let dir = wal_dir(root, db, table);
    let origins = seg_list(&dir)?;
    let mut out: Vec<Segment> = Vec::new();
    let mut prev: Option<(u64, PathBuf, Span, u64)> = None;
    for &origin in &origins {
        let path = dir.join(seg_name(origin));
        let len = std::fs::metadata(&path).map_err(|e| store::io_err("stat", &path, e))?.len();
        let mut head = [0u8; SEG_HEADER_LEN as usize];
        read_exact_at(&path, &mut head)
            .ok_or_else(|| Error::corruption(format!("{}: unreadable segment header", path.display())))?;
        let (h, trusted) = read_head(&head, origin).map_err(|e| store::prefix(&path, e))?;
        if let Some((o, p, mut span, plen)) = prev.take() {
            if o + plen - SEG_HEADER_LEN != origin {
                return Err(Error::corruption(format!(
                    "the WAL of `{db}.{table}` has a hole: {} is {plen} bytes, so it ends at \
                     stream position {}, but the next segment {} starts at {origin}. A \
                     recovery that spans the hole would silently skip whatever those \
                     records held",
                    p.display(),
                    o + plen - SEG_HEADER_LEN,
                    path.display()
                )));
            }
            span.last_seq = h.prev_seq;
            span.last_ms = h.prev_ms;
            out.push(Segment { origin: o, end: origin, span, path: p });
        }
        // A sealed segment's `durable` was stamped and fsynced before its
        // successor was published, so anything short of the file's length
        // means bytes are missing from the middle of the archive.
        if trusted && h.durable != origin + len - SEG_HEADER_LEN {
            let ahead = origin + len - SEG_HEADER_LEN;
            // Only for segments that will turn out to be sealed; the active
            // one is allowed to hold an unacknowledged tail.
            if origins.last() != Some(&origin) {
                return Err(Error::corruption(format!(
                    "archived segment {} is {len} bytes, so it runs to stream position \
                     {ahead}, but it declares only {} acknowledged. A recovery through it \
                     would stop early and report success",
                    path.display(),
                    h.durable
                )));
            }
        }
        prev = Some((
            origin,
            path,
            Span { first_seq: h.prev_seq, first_ms: h.prev_ms, ..Span::default() },
            len,
        ));
    }
    Ok(out)
}

/// The highest recovery LSN this table's archive no longer holds.
///
/// Derived rather than stored: `prune` drops oldest-first, so the oldest
/// surviving segment's `prev_seq` **is** the horizon by construction. Strictly
/// more correct than the `HORIZON` file it replaces, which had to be raised
/// before the unlink and so left a window in which it was too conservative.
pub fn archive_horizon(root: &Path, db: &str, table: &str) -> Result<u64> {
    let dir = wal_dir(root, db, table);
    let Some(&oldest) = seg_list(&dir)?.first() else { return Ok(0) };
    let path = dir.join(seg_name(oldest));
    let mut head = [0u8; SEG_HEADER_LEN as usize];
    let Some(()) = read_exact_at(&path, &mut head) else { return Ok(0) };
    Ok(read_head(&head, oldest).map(|(h, _)| h.prev_seq).unwrap_or(0))
}

/// Every `(db, table)` with a log under `root`.
pub fn archived_tables(root: &Path) -> Result<Vec<(String, String)>> {
    let base = root.join(WAL_DIR);
    let mut out = Vec::new();
    let Ok(dbs) = std::fs::read_dir(&base) else { return Ok(out) };
    for db in dbs.flatten() {
        let Some(dbn) = db.file_name().to_str().map(str::to_string) else { continue };
        let Ok(tables) = std::fs::read_dir(db.path()) else { continue };
        for t in tables.flatten() {
            if let Some(tn) = t.file_name().to_str() {
                out.push((dbn.clone(), tn.to_string()));
            }
        }
    }
    out.sort();
    Ok(out)
}

/// The table whose live segment still holds records the archive does not, if
/// there is one.
///
/// The archive is published by a checkpoint, so between checkpoints it is
/// behind the database. That difference is the whole of "this instant is not
/// recoverable yet": with every log rolled, the archive *is* the database's
/// history and any instant up to now is answerable; with one of them holding
/// records, an instant after the last archived tick would quietly resolve to
/// an earlier state.
///
/// Conservative in the direction that refuses: a writer appending while this
/// runs makes the live segment look non-empty, which is exactly the answer
/// that case deserves.
pub fn archive_lags(root: &Path) -> Result<Option<String>> {
    for (db, table) in archived_tables(root)? {
        if let Some((end, floor)) = live_extent(&wal_dir(root, &db, &table)) {
            if end > floor {
                return Ok(Some(format!("{db}.{table}")));
            }
        }
    }
    Ok(None)
}

/// The last tick the whole archive under `root` holds: the newest state any
/// recovery from it can reach.
pub fn archive_end(root: &Path) -> Result<Span> {
    let mut end = Span::default();
    for (db, table) in archived_tables(root)? {
        let segs = segments(root, &db, &table)?;
        for s in &segs {
            end.last_seq = end.last_seq.max(s.span.last_seq);
            end.last_ms = end.last_ms.max(s.span.last_ms);
        }
        // Where this table's archive *begins*: one past whatever retention has
        // dropped. Taken from [`archive_horizon`] rather than from the oldest
        // segment's own span, because the two disagree in exactly the case
        // that matters -- when retention has dropped every sealed segment the
        // list is empty and has nothing to say, while the horizon is still
        // exact, because it is read from the surviving segment's header.
        let first = archive_horizon(root, &db, &table)? + 1;
        if end.first_seq == 0 || first < end.first_seq {
            end.first_seq = first;
            end.first_ms = segs.first().map_or(0, |s| s.span.first_ms);
        }
    }
    Ok(end)
}

/// One table's archived stream, cut to `[from_seq, target]`.
#[derive(Debug)]
pub struct Recovered {
    /// A whole segment file -- header plus the framed records in range --
    /// ready to be written into a restored table's log directory and replayed
    /// by the ordinary loader. Byte ranges of the archived segments, copied
    /// verbatim: there is no second encoder here to drift from the one that
    /// wrote them.
    pub bytes: Vec<u8>,
    /// Mutations it holds. Bookkeeping records are not counted.
    pub records: u64,
    /// The ticks actually included.
    pub applied: Span,
}

/// The records of `<db>.<table>` that a backup taken at recovery LSN
/// `from_seq` does not already hold, up to `target`.
///
/// Both ends of the cut land on a tick, because a tick is the instant a group
/// of records became durable: everything behind one is a state the database
/// really was in, and everything in front of the next one is not yet
/// acknowledged. Replaying to the same target twice therefore produces the
/// same bytes -- the cut is a pure function of the archive.
pub fn recover(
    root: &Path,
    db: &str,
    table: &str,
    from_seq: u64,
    target: Target,
) -> Result<Recovered> {
    let dir = wal_dir(root, db, table);
    let segs = segments(root, db, table)?;
    let dropped = archive_horizon(root, db, table)?;
    if dropped >= from_seq && from_seq != 0 {
        return Err(Error::storage(format!(
            "the WAL archive of `{db}.{table}` no longer reaches back to the backup: \
             retention has dropped everything up to recovery LSN {dropped}, and the backup \
             needs {from_seq} onwards. Recovering from it would silently skip the records \
             in between"
        )));
    }
    let mut body: Vec<u8> = Vec::new();
    let mut rec = Recovered { bytes: Vec::new(), records: 0, applied: Span::default() };
    for s in &segs {
        // Every tick in it predates the backup, so all of it is already in the
        // parts the backup restored.
        if s.span.last_seq != 0 && s.span.last_seq < from_seq {
            continue;
        }
        let seg = Seg::load(&dir, s.origin, true)?;
        let cut = cut_segment(&seg, from_seq, target).map_err(|e| store::prefix(&s.path, e))?;
        body.extend_from_slice(&seg.buf[cut.from as usize..cut.to as usize]);
        rec.records += cut.records;
        for (seq, ms) in cut.ticks {
            rec.applied.observe(seq, ms);
        }
        if cut.stopped {
            break;
        }
    }
    // The restored stream starts at zero: there is nothing behind it, so the
    // segment the loader will find is both the whole archive and the active
    // log. `prev_ms` carries the clock so the restored database's first tick
    // cannot read earlier than the last one it replayed; `prev_seq` stays 0,
    // because nothing has been pruned from an archive that has never had a
    // segment retired.
    let head = Head {
        carry_len: 0,
        next_seq: 0,
        prev_seq: 0,
        prev_ms: rec.applied.last_ms,
        durable: body.len() as u64,
        acked: body.len() as u64,
    };
    let mut out = Vec::with_capacity(SEG_HEADER_LEN as usize + body.len());
    out.extend_from_slice(&head.encode(0));
    out.extend_from_slice(&body);
    rec.bytes = out;
    Ok(rec)
}

/// The name a recovered stream's one segment is written under.
pub fn recovered_seg_name() -> String {
    seg_name(0)
}

/// The stream position `<db>.<table>`'s log has reached, without opening it.
///
/// What `CREATE TABLE` stamps into a fresh `TABLE` file. The log directory
/// survives `DROP TABLE` -- that is what makes "restore to just before the
/// drop" work -- so a table recreated under the same name would otherwise
/// default to a watermark of 0 and replay the previous incarnation's whole
/// stream into a schema that need not match it. Recording where the stream has
/// already reached costs one `stat` at DDL time and makes that impossible.
pub fn stream_end(root: &Path, db: &str, table: &str) -> u64 {
    live_extent(&wal_dir(root, db, table)).map_or(0, |(end, _)| end)
}

/// Roll `<db>.<table>`'s log if it holds anything, and return the stream
/// position a checkpoint should record as covered.
///
/// The watermark and the fresh segment's origin are the same number *by
/// construction* here, rather than by arithmetic that has to be kept in step
/// with whatever the roll appended -- which is the whole reason the ordering
/// is "roll, then commit the parts" and not the reverse. A crash in between
/// leaves the old watermark and no new parts, so replay replays everything
/// from it, which is correct and is the one case that exercises multi-segment
/// replay.
///
/// A log with nothing in it is not opened at all: minting a segment for a
/// database that was only read is write amplification charged to a reader.
pub fn roll_for_checkpoint(dir: &Path) -> Result<u64> {
    match live_extent(dir) {
        None => Ok(0),
        Some((end, floor)) if end <= floor => Ok(end),
        Some(_) => {
            let mut w = Wal::open(dir)?;
            w.roll()?;
            Ok(w.origin())
        }
    }
}

/// Where one segment's kept bytes start and stop, as file offsets.
struct Cut {
    from: u64,
    to: u64,
    records: u64,
    /// The ticks inside `[from, to)`, in order.
    ticks: Vec<(u64, u64)>,
    /// A tick failed the target, so nothing after this segment is wanted.
    stopped: bool,
}

/// One pass over a segment, finding both cuts.
///
/// `from` is just past the last tick that predates the backup and `to` just
/// past the last one the target keeps -- so the copied range is exactly the
/// records that became durable inside the window, framing intact, with no
/// record re-encoded.
fn cut_segment(seg: &Seg, from_seq: u64, target: Target) -> Result<Cut> {
    let head = SEG_HEADER_LEN;
    let mut cut = Cut { from: head, to: head, records: 0, ticks: Vec::new(), stopped: false };
    let mut pending = 0u64;
    let mut stop = false;
    walk_seg(seg, 0, |at, body| {
        if stop {
            return Ok(());
        }
        match body.first() {
            Some(&TAG_TICK) => {
                let Some((seq, ms)) = tick_of(body) else { return Ok(()) };
                let after = at + frame_len(body);
                if seq < from_seq {
                    // Everything to here is inside the backup already.
                    cut.from = after;
                    cut.to = after;
                    cut.records = 0;
                    cut.ticks.clear();
                    pending = 0;
                } else if target.keeps(seq, ms) {
                    cut.to = after;
                    cut.records += std::mem::take(&mut pending);
                    cut.ticks.push((seq, ms));
                } else {
                    cut.stopped = true;
                    stop = true;
                }
            }
            Some(&t) if is_mutation(t) => pending += count_of(body),
            _ => {}
        }
        Ok(())
    })?;
    Ok(cut)
}

/// Bytes one frame occupies, given its body. The framing is
/// `varint len | u64 sum | body`, so this is a length computation and not a
/// second decoder.
fn frame_len(body: &[u8]) -> u64 {
    let mut w = Writer::with_capacity(10);
    w.varint(body.len() as u64);
    w.as_slice().len() as u64 + 8 + body.len() as u64
}

fn is_mutation(tag: u8) -> bool {
    matches!(tag & !STAGED, TAG_INSERT | TAG_DELETE | TAG_DELETE_RUN | TAG_MASK_RUN)
}

/// How many mutations one frame stands for. One, except for the two runs.
fn count_of(body: &[u8]) -> u64 {
    let mut r = Reader::new(body);
    let Ok(tag) = r.u8() else { return 1 };
    let bare = tag & !STAGED;
    if !matches!(bare, TAG_DELETE_RUN | TAG_MASK_RUN) {
        return 1;
    }
    // The staging sequence number, then -- for a mask run only -- the part
    // identity, both in front of the count.
    if (tag & STAGED != 0 && r.varint().is_err())
        || (bare == TAG_MASK_RUN && r.varint().is_err())
    {
        return 1;
    }
    r.varint().unwrap_or(1)
}

/// The `(recovery LSN, millis)` a [`TAG_TICK`] body carries.
fn tick_of(body: &[u8]) -> Option<(u64, u64)> {
    let mut r = Reader::new(body);
    (r.u8().ok()? == TAG_TICK).then(|| Some((r.varint().ok()?, r.varint().ok()?))).flatten()
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

/// The lower bound a frame body puts on [`Wal::begin`]'s next sequence
/// number, for the open-time scan.
///
/// Deliberately lenient where `decode_entry` is not: `scan` has already proved
/// the frame checksum, and a body that will not parse against *some* schema is
/// a problem for `replay` to report -- refusing to open the log over it would
/// turn a schema mismatch into an outage.
fn body_next_seq(body: &[u8]) -> u64 {
    let mut r = Reader::new(body);
    let Ok(tag) = r.u8() else { return 0 };
    match tag {
        TAG_COMMIT | TAG_DECIDE | TAG_PREPARE => {}
        t if t & STAGED != 0 => {}
        _ => return 0,
    }
    r.varint().map_or(0, |s| s.saturating_add(1))
}

/// The number a single-`varint` record of `tag` carries, or `None` when the
/// body is something else. Framing-level, like [`body_next_seq`].
fn tagged(body: &[u8], tag: u8) -> Option<u64> {
    let mut r = Reader::new(body);
    (r.u8().ok()? == tag).then(|| r.varint().ok()).flatten()
}

/// Collect the [`TAG_DECIDE`] numbers in a log directory's active segment.
///
/// Sealed segments are never consulted for a decision, and they need not be:
/// at every roll [`Wal::may_be_cited`] decides whether to carry the decisions
/// forward, and it answers false exactly when nothing can still cite them. So
/// the active segment holds the complete citable set, always.
fn walk_dir_newest(dir: &Path, out: &mut Vec<u64>) -> Result<()> {
    let Some(&origin) = seg_list(dir)?.last() else { return Ok(()) };
    let seg = Seg::load(dir, origin, false)?;
    let path = seg.path(dir);
    walk_seg(&seg, 0, |_, body| {
        if let Some(s) = tagged(body, TAG_DECIDE) {
            out.push(s);
        }
        Ok(())
    })
    .map_err(|e| store::prefix(&path, e))?;
    out.sort_unstable();
    out.dedup();
    Ok(())
}

/// The sequence numbers the log cited by `rel` has decided, ascending.
///
/// A missing directory is an empty set rather than an error, and so is an
/// unresolvable citation: both mean "no decision found", which means abort,
/// which is the answer that cannot invent rows. Damage *inside* the cited log
/// is reported, because a decision that might be there and might not is the
/// one thing this cannot guess at.
fn decisions_of(dir: &Path, rel: &str) -> Result<Vec<u64>> {
    let path = resolve_citation(dir, rel)?;
    if !path.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    walk_dir_newest(&path, &mut out)?;
    Ok(out)
}

/// The path from one log directory to another, as a citation.
///
/// Refused rather than approximated when the two cannot be compared -- one
/// absolute and one relative, or a component that is not UTF-8. A citation
/// that resolves to the wrong file is worse than a transaction that cannot
/// commit, because the wrong file may well contain a decision.
fn relative_dir(from: &Path, to: &Path) -> Result<String> {
    if from.is_absolute() != to.is_absolute() {
        return Err(Error::storage(format!(
            "cannot cite `{}` from `{}`: one path is absolute and the other is not",
            to.display(),
            from.display()
        )));
    }
    let mut a = from.components().peekable();
    let mut b = to.components().peekable();
    while a.peek().is_some() && a.peek() == b.peek() {
        a.next();
        b.next();
    }
    let mut out = String::new();
    for _ in a {
        out.push_str("../");
    }
    for c in b {
        let Component::Normal(s) = c else {
            return Err(Error::storage(format!("cannot cite `{}`", to.display())));
        };
        let Some(s) = s.to_str() else {
            return Err(Error::storage(format!("cannot cite `{}`", to.display())));
        };
        out.push_str(s);
        out.push('/');
    }
    out.pop();
    if out.is_empty() {
        return Err(Error::storage("a log cannot cite itself".to_string()));
    }
    Ok(out)
}

/// Turn a citation read out of a log back into a path, or refuse.
///
/// The citation is a hostile input like everything else read from disk, and it
/// is fed straight to `read_dir`. Resolved lexically rather than by the
/// filesystem, so a `..` cannot be pushed through a symlink into a tree this
/// database does not own, and bounded to the shape a citation can legitimately
/// have: relative, no root, ordinary names, and a result that is another
/// table's log directory under the *same* `.wal` root.
fn resolve_citation(dir: &Path, rel: &str) -> Result<PathBuf> {
    let bad = || Error::corruption(format!("`{rel}` is not a log citation"));
    if rel.is_empty() || rel.len() > 4096 {
        return Err(bad());
    }
    let mut out = dir.to_path_buf();
    for c in Path::new(rel).components() {
        match c {
            Component::ParentDir if out.pop() => {}
            Component::Normal(s) => match s.to_str() {
                Some(s) if store::is_safe_name(s) => out.push(s),
                _ => return Err(bad()),
            },
            _ => return Err(bad()),
        }
    }
    // `<wal root>/<db>/<table>`, and the same `<wal root>` this log sits
    // under. Anything else -- a table directory, a database directory, some
    // other tree entirely -- is refused rather than opened.
    let same = out.parent().and_then(Path::parent);
    if out == dir || same.is_none() || same != dir.parent().and_then(Path::parent) {
        return Err(bad());
    }
    Ok(out)
}

/// How much of the table directory `dir`'s log its committed parts already
/// cover. Zero -- "none of it" -- for anything that cannot be read, which is
/// the answer that keeps a decision rather than dropping it.
fn covered_prefix(dir: &Path) -> u64 {
    std::fs::read(dir.join(store::TABLE_FILE))
        .ok()
        .and_then(|b| reader::table_parts_from_bytes(&b).ok())
        .map_or(0, |(_, _, committed)| committed)
}

/// Undo [`crate::storage::MaskRuns`]'s delta encoding: `n` LEB128 gaps back
/// into ascending absolute positions.
///
/// Exactly `n` of them and not one byte left over. A short read or a trailing
/// byte means the frame is not the shape its own header claims, which is
/// corruption rather than something to salvage a prefix from -- a mask record
/// half applied hides a row nobody asked to hide.
fn mask_positions(n: u64, deltas: &[u8]) -> Result<Box<[u64]>> {
    let mut r = Reader::new(deltas);
    let mut out = Vec::with_capacity(n as usize);
    let mut prev = u64::MAX;
    for _ in 0..n {
        prev = prev.wrapping_add(r.varint()?).wrapping_add(1);
        out.push(prev);
    }
    if !r.is_empty() {
        return Err(Error::corruption(format!(
            "{} trailing bytes after the {n} positions of a mask run",
            r.remaining()
        )));
    }
    Ok(out.into_boxed_slice())
}

fn record_err(at: u64, index: usize, e: Error) -> Error {
    Error::corruption(format!("record {index} of the replay, at offset {at}: {e}"))
}

fn decode_entry<'a>(body: &'a [u8], schema: &Schema) -> Result<Entry<'a>> {
    let mut br = Reader::new(body);
    let tag = br.u8()?;
    // The markers are matched on the whole byte, so `STAGED` over one of them
    // falls through to the mutation arm and is reported as the tag it is.
    let rec = match tag {
        TAG_COMMIT | TAG_DECIDE => Entry::Commit(br.varint()?),
        TAG_PREPARE => {
            let (seq, coord_seq) = (br.varint()?, br.varint()?);
            Entry::Prepare(seq, br.str()?, coord_seq)
        }
        TAG_TICK => {
            let seq = br.varint()?;
            // Read to prove the frame is the shape it claims; the trailing
            // check below is what turns that into a rejection.
            br.varint()?;
            Entry::Tick(seq)
        }
        _ => {
            // The sequence number sits between the tag and the payload, so the
            // staged and plain forms share one decoder for the part that matters.
            let seq = if tag & STAGED != 0 { Some(br.varint()?) } else { None };
            match tag & !STAGED {
                TAG_INSERT => {
                    Entry::Record(seq, WalRecord::Insert(reader::get_block(&mut br, schema)?))
                }
                TAG_DELETE => Entry::Record(seq, WalRecord::Delete(br.u64()?)),
                TAG_DELETE_RUN => {
                    let n = br.varint()?;
                    let want = n.checked_mul(8).filter(|&w| w <= br.remaining() as u64);
                    let Some(want) = want else {
                        return Err(Error::corruption(format!(
                            "a delete run of {n} lanes needs more than the {} bytes in its \
                             frame",
                            br.remaining()
                        )));
                    };
                    Entry::Deletes(seq, br.take(want as usize)?)
                }
                TAG_MASK_RUN => {
                    let (pid, n) = (br.varint()?, br.varint()?);
                    // One delta is at least one byte, so a count larger than
                    // the frame's remainder is a corrupt frame and not a
                    // gigantic allocation.
                    if n > br.remaining() as u64 {
                        return Err(Error::corruption(format!(
                            "a mask run of {n} positions needs more than the {} bytes in \
                             its frame",
                            br.remaining()
                        )));
                    }
                    Entry::Masks(seq, pid, n, br.take(br.remaining())?)
                }
                // The raw tag, not the masked one: `0x83` is not "tag 3".
                _ => return Err(Error::corruption(format!("unknown log record tag {tag}"))),
            }
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

    /// The active segment's file. Every fixture that damages bytes by hand
    /// goes through this rather than a fixed name, because the name moves with
    /// every roll.
    fn active(dir: &Path) -> PathBuf {
        dir.join(seg_name(*seg_list(dir).unwrap().last().expect("a segment")))
    }

    /// A log holding `n` alternating inserts and deletes, plus the record list
    /// it should replay to and the LSN each record starts at.
    fn populated(s: &Scratch, n: usize) -> (PathBuf, Vec<WalRecord>, Vec<u64>) {
        let dir = s.join("wal");
        let mut w = Wal::open(&dir).unwrap();
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
        (dir, want, offsets)
    }

    /// Replay a hand-built image of the active segment, with the durability
    /// stamp lowered to `durable` so the image describes a log that really was
    /// acknowledged only that far.
    ///
    /// Damaging a segment without touching its stamp says "these bytes were
    /// acknowledged and are now wrong", which is corruption by definition.
    /// Simulating an *interrupted append* means saying so.
    fn restamp(path: &Path, bytes: &[u8], durable: u64) {
        let origin = parse_seg_origin(path.file_name().unwrap().to_str().unwrap()).unwrap();
        let mut img = bytes.to_vec();
        if img.len() >= SEG_HEADER_LEN as usize {
            let (mut h, _) = read_head(&img[..SEG_HEADER_LEN as usize], origin).unwrap();
            h.durable = durable;
            // Acknowledged *and* proven: an image built by hand is standing in
            // for a segment whose last `fsync` returned, so the floor moves
            // with the claim. Leaving `acked` behind would put every cut
            // inside the forgiveness window and the fixture would prove
            // nothing.
            h.acked = durable;
            img[..SEG_HEADER_LEN as usize].copy_from_slice(&h.encode(origin));
        }
        std::fs::write(path, &img).unwrap();
    }

    #[test]
    fn a_fresh_log_is_exactly_a_header() {
        let s = Scratch::new("wal-fresh");
        let dir = s.join("wal");
        let w = Wal::open(&dir).unwrap();
        assert!(w.is_empty());
        assert_eq!(w.len(), 0, "an empty stream is position zero, not a header length");
        assert_eq!(w.origin(), 0);
        assert_eq!(std::fs::metadata(active(&dir)).unwrap().len(), SEG_HEADER_LEN);
        assert!(Wal::replay(&dir, &schema()).unwrap().is_empty());
    }

    #[test]
    fn records_replay_in_order() {
        let s = Scratch::new("wal-order");
        let (dir, want, _) = populated(&s, 25);
        assert_eq!(Wal::replay(&dir, &schema()).unwrap(), want);
    }

    #[test]
    fn a_large_insert_roundtrips() {
        let s = Scratch::new("wal-big");
        let dir = s.join("wal");
        let b = sample_block(5_000);
        let mut w = Wal::open(&dir).unwrap();
        w.append_insert(&b).unwrap();
        w.sync().unwrap();
        let back = Wal::replay(&dir, &schema()).unwrap();
        assert_eq!(back, vec![WalRecord::Insert(b)]);
    }

    #[test]
    fn nulls_and_strings_survive_the_log() {
        let s = Scratch::new("wal-nulls");
        let dir = s.join("wal");
        let b = sample_block(200);
        assert!(b.column(2).has_nulls());
        let mut w = Wal::open(&dir).unwrap();
        w.append_insert(&b).unwrap();
        w.sync().unwrap();
        let WalRecord::Insert(back) = Wal::replay(&dir, &schema()).unwrap().remove(0) else {
            panic!("expected an insert")
        };
        assert!(back.column(2).is_null(0));
        assert_eq!(back.column(1).value(3), b.column(1).value(3));
        assert_eq!(back, b);
    }

    #[test]
    fn reopening_appends_after_the_existing_records() {
        let s = Scratch::new("wal-reopen");
        let (dir, mut want, _) = populated(&s, 4);
        {
            let mut w = Wal::open(&dir).unwrap();
            assert!(!w.is_empty());
            w.append_delete(999).unwrap();
            want.push(WalRecord::Delete(999));
            w.sync().unwrap();
        }
        assert_eq!(Wal::replay(&dir, &schema()).unwrap(), want);
    }

    /// A roll seals the segment where it stands and opens a fresh one whose
    /// origin is the sealed one's stream end. Nothing is copied, nothing is
    /// linked, and the sealed bytes are still on disk under their own name --
    /// which is the whole of "archiving is not an operation".
    #[test]
    fn a_roll_seals_in_place_and_starts_the_next_segment_where_it_ended() {
        let _x = exclusive();
        let s = Scratch::new("wal-roll");
        let (dir, _, _) = populated(&s, 6);
        let mut w = Wal::open(&dir).unwrap();
        let end = w.len();
        w.roll().unwrap();
        assert!(w.is_empty());
        assert_eq!(w.origin(), end, "the fresh segment starts where the sealed one ended");
        assert_eq!(w.len(), end, "and an LSN never restarts");
        assert_eq!(w.pending(), 0);
        assert_eq!(seg_list(&dir).unwrap().len(), 2, "the sealed segment stays where it was");
        assert!(Wal::replay_from(&dir, &schema(), end).unwrap().is_empty());

        let at = w.append_delete(5).unwrap();
        assert_eq!(at, end);
        w.sync().unwrap();
        assert_eq!(
            Wal::replay_from(&dir, &schema(), end).unwrap(),
            vec![WalRecord::Delete(5)]
        );
    }

    #[test]
    fn replay_from_skips_a_checkpointed_prefix() {
        let s = Scratch::new("wal-from");
        let (dir, want, offsets) = populated(&s, 9);
        for (i, &off) in offsets.iter().enumerate() {
            assert_eq!(
                Wal::replay_from(&dir, &schema(), off).unwrap(),
                want[i.min(want.len())..],
                "from record {i}"
            );
        }
        // A watermark past the end used to be clamped in silence. Under a
        // stream LSN it can only mean segments are gone, so it is reported.
        let e = Wal::replay_from(&dir, &schema(), 1 << 30).expect_err("a hole must report");
        assert!(e.to_string().contains("missing"), "{e}");
    }

    /// Replay spans segments, and the LSN space is continuous across them.
    /// The healthy database has one segment, so this is the least-exercised
    /// path in the module and the one a 56-byte slip would hide in.
    #[test]
    fn replay_crosses_segments_and_the_lsn_space_is_continuous() {
        let _x = exclusive();
        let s = Scratch::new("wal-multiseg");
        let dir = s.join("wal");
        let mut want: Vec<WalRecord> = Vec::new();
        let mut lsns: Vec<u64> = Vec::new();
        for gen in 0..4u64 {
            let mut w = Wal::open(&dir).unwrap();
            for k in 0..3u64 {
                lsns.push(w.append_delete(gen * 10 + k).unwrap());
                want.push(WalRecord::Delete(gen * 10 + k));
            }
            w.sync().unwrap();
            w.roll().unwrap();
        }
        assert_eq!(seg_list(&dir).unwrap().len(), 5);
        assert_eq!(Wal::replay(&dir, &schema()).unwrap(), want);
        // ...and every LSN is still a valid watermark, whichever segment it
        // fell in.
        let got = Wal::replay_with_lsn(&dir, &schema(), 0).unwrap();
        assert_eq!(got.iter().map(|&(l, _)| l).collect::<Vec<_>>(), lsns);
        for (i, &l) in lsns.iter().enumerate() {
            assert_eq!(
                Wal::replay_from(&dir, &schema(), l).unwrap(),
                want[i..],
                "watermark {l} (record {i})"
            );
        }
    }

    /// A segment missing from the middle of the chain is a hole, and the whole
    /// point of the numbering is that a directory listing proves it.
    #[test]
    fn a_missing_middle_segment_is_reported() {
        let _x = exclusive();
        let s = Scratch::new("wal-gap");
        let dir = s.join("wal");
        for gen in 0..3u64 {
            let mut w = Wal::open(&dir).unwrap();
            w.append_delete(gen).unwrap();
            w.sync().unwrap();
            w.roll().unwrap();
        }
        let origins = seg_list(&dir).unwrap();
        std::fs::remove_file(dir.join(seg_name(origins[1]))).unwrap();
        let e = Wal::replay(&dir, &schema()).expect_err("a hole must be reported");
        assert!(e.to_string().contains("hole"), "{e}");
    }

    /// **Inverted deliberately, and it was hiding a silent total loss.**
    ///
    /// This used to assert that a segment shorter than its 64-byte header is
    /// "an empty log, not an error": `Wal::open` rewrote it from scratch and
    /// `replay_entries` skipped it, both justified as the "crashed between
    /// `creat` and the first write" case. There is no such case. A segment is
    /// published by `store::atomic_write`, which writes the whole header into a
    /// temp file, `fsync`s it, renames it into place and `fsync`s the
    /// directory -- so at every instant a crash can observe, a segment that
    /// exists is at least a full header.
    ///
    /// What the two shortcuts actually did was discard a *damaged* segment's
    /// entire contents and report success. Reproduced end to end: six
    /// acknowledged rows, the active segment cut to 0, 8, 30 or 63 bytes,
    /// `SELECT count()` = 0, exit 0, nothing quarantined -- the exact forbidden
    /// outcome this format version was bumped to remove, reached without
    /// forging a single checksum.
    #[test]
    fn a_segment_shorter_than_its_header_is_damage_not_an_empty_log() {
        let s = Scratch::new("wal-shorthdr");
        let dir = s.join("wal");
        {
            let mut w = Wal::open(&dir).unwrap();
            w.append_delete(7).unwrap();
            w.sync().unwrap();
        }
        let p = active(&dir);
        let full = std::fs::read(&p).unwrap();
        assert_eq!(Wal::replay(&dir, &schema()).unwrap().len(), 1, "the fixture wrote nothing");
        for n in 0..SEG_HEADER_LEN as usize {
            std::fs::write(&p, &full[..n]).unwrap();
            assert!(
                Wal::open(&dir).is_err(),
                "a {n}-byte segment must be refused, not reset to an empty log"
            );
            Wal::replay(&dir, &schema())
                .expect_err(&format!("a {n}-byte segment must not replay as an empty log"));
            // …and the refusal must not repair itself away: an open that
            // rewrote the file would make the *second* attempt succeed with an
            // empty log, which is the same loss one call later.
            assert_eq!(
                std::fs::metadata(&p).unwrap().len(),
                n as u64,
                "the refused open rewrote the damaged segment (n={n})"
            );
        }
    }

    // ---- the classifier ----------------------------------------------------

    /// **The bug this format exists to kill.** A frame's length field is not
    /// covered by anything that can tell where the frame really ends, so a
    /// corrupted length can always claim to reach the end of the file. Under
    /// the old positional rule that made a torn *middle* read as a torn tail:
    /// replay stopped there, dropped every acknowledged record after it, and
    /// returned `Ok`.
    #[test]
    fn a_corrupted_length_in_the_middle_is_reported_not_swallowed() {
        let s = Scratch::new("wal-lenrot");
        let (dir, want, offsets) = populated(&s, 6);
        let p = active(&dir);
        let full = std::fs::read(&p).unwrap();
        // The length varint of the third record: entirely inside the file,
        // with three more records and a tick behind it.
        let victim = (offsets[2] - 0 + SEG_HEADER_LEN) as usize;
        for bit in 0..8 {
            let mut bytes = full.clone();
            bytes[victim] ^= 1 << bit;
            if bytes[victim] == full[victim] {
                continue;
            }
            std::fs::write(&p, &bytes).unwrap();
            match Wal::replay(&dir, &schema()) {
                Err(e) => {
                    assert_eq!(e.code(), "CHECKSUM_MISMATCH", "bit {bit}: {e}");
                    assert!(e.to_string().contains("silently"), "bit {bit}: {e}");
                }
                Ok(got) => assert_eq!(
                    got,
                    want,
                    "bit {bit} of the length field silently dropped {} records",
                    want.len() - got.len()
                ),
            }
        }
    }

    /// The exhaustive form, over all four damage quadrants.
    ///
    /// The contract has one exemption and it is the design's only residual: a
    /// segment's forgiveness window is exactly **one group commit** wide, the
    /// span between the last two `fsync`s, because at the moment of the last
    /// one "the stamp landed and the data did not" and "both landed and the
    /// data was then destroyed" are indistinguishable without a second
    /// `fsync` per commit. Everything below that -- every record whose
    /// `fsync` provably returned -- must come back or be reported. Never a
    /// silent short list.
    ///
    /// The garbage-fill quadrant is the one a truncation-only sweep never
    /// manufactures, and it is exactly where the old classifier lost data:
    /// truncation only ever produces tails.
    #[test]
    fn no_damage_produces_a_short_replay_that_reports_success() {
        let s = Scratch::new("wal-sweep");
        let (dir, mut want, _) = populated(&s, 5);
        // A second `fsync`, so the segment has an `acked` floor as well as a
        // `durable` claim: with only one sync ever performed nothing is
        // provably on the platter, and "the whole body is a hole" really is
        // indistinguishable from "the fsync never returned".
        {
            let mut w = Wal::open(&dir).unwrap();
            w.append_delete(0xFEED).unwrap();
            w.sync().unwrap();
            want.push(WalRecord::Delete(0xFEED));
        }
        let p = active(&dir);
        let full = std::fs::read(&p).unwrap();
        let body = SEG_HEADER_LEN as usize;
        // Everything the *second-to-last* `fsync` covered: the floor below
        // which nothing may be lost quietly.
        let acked = want.len() - 1;
        let check = |what: &str, at: usize| {
            match Wal::replay(&dir, &schema()) {
                Err(_) => {}
                Ok(got) => {
                    assert!(
                        got.len() >= acked,
                        "{what} at {at} replayed {} of {} records and reported success",
                        got.len(),
                        want.len()
                    );
                    assert_eq!(got, want[..got.len()], "{what} at {at} replayed a non-prefix");
                }
            };
        };
        for at in body..full.len() {
            for bit in 0..8 {
                let mut b = full.clone();
                b[at] ^= 1 << bit;
                std::fs::write(&p, &b).unwrap();
                check("a flipped bit", at);
            }
            let mut b = full.clone();
            b[at..].fill(0);
            std::fs::write(&p, &b).unwrap();
            check("a zero fill", at);

            let mut b = full.clone();
            b[at..].fill(0x5A);
            std::fs::write(&p, &b).unwrap();
            check("a garbage fill", at);

            std::fs::write(&p, &full[..at]).unwrap();
            check("a truncation", at);
        }
    }

    /// A genuine interrupted append: the bytes are above the stamp, because
    /// the `fsync` that would have raised it never returned. Swallowed,
    /// silently, exactly as before -- this is the case the whole positional
    /// heuristic existed to serve, and it is now served by a recorded fact.
    #[test]
    fn a_torn_tail_replays_the_intact_prefix() {
        let s = Scratch::new("wal-torn");
        let (dir, want, _) = populated(&s, 5);
        let p = active(&dir);
        // One more record with no `sync` behind it, then cut it in half.
        {
            let mut w = Wal::open(&dir).unwrap();
            w.append_delete(0xBAD).unwrap();
        }
        let full = std::fs::read(&p).unwrap();
        std::fs::write(&p, &full[..full.len() - 4]).unwrap();
        assert_eq!(
            Wal::replay(&dir, &schema()).unwrap(),
            want,
            "an unacknowledged partial record must be dropped, with no error"
        );
        // ...and it is the *stamp* doing the work, not the position: the same
        // bytes under a segment that says its `fsync` returned for all of them
        // are damage.
        restamp(&p, &full[..full.len() - 4], full.len() as u64 - SEG_HEADER_LEN);
        assert!(
            Wal::replay(&dir, &schema()).is_err(),
            "bytes the log said it had acknowledged must not be discarded quietly"
        );
    }

    #[test]
    fn damage_to_the_final_unacknowledged_record_is_treated_as_a_tear() {
        let s = Scratch::new("wal-tailrot");
        let (dir, want, _) = populated(&s, 5);
        let p = active(&dir);
        // An append a crash caught before its `fsync`: durable does not cover
        // it, so damage to it is an interrupted write.
        {
            let mut w = Wal::open(&dir).unwrap();
            w.append_delete(0xBAD).unwrap();
        }
        let mut bytes = std::fs::read(&p).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x80;
        std::fs::write(&p, &bytes).unwrap();
        let back = Wal::replay(&dir, &schema()).unwrap();
        assert_eq!(back, want, "the damaged final record must be dropped, no error");
    }

    /// The inverse of the test this replaces. Damage to a tick that was
    /// `fsync`ed is damage to *acknowledged* bytes, and it is reported now
    /// rather than swallowed as "the very end of the file". The old rule could
    /// not tell the two apart; the stamp can.
    #[test]
    fn damage_to_an_acknowledged_trailing_tick_is_reported() {
        let s = Scratch::new("wal-tickrot");
        let (dir, _, _) = populated(&s, 5);
        let p = active(&dir);
        let mut bytes = std::fs::read(&p).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x80;
        std::fs::write(&p, &bytes).unwrap();
        let e = Wal::replay(&dir, &schema()).expect_err("acknowledged damage must report");
        assert_eq!(e.code(), "CHECKSUM_MISMATCH", "{e}");
    }

    /// Zeros with acknowledged records behind them are damage the log did to
    /// itself and must still be reported -- the rule the zero clamp is
    /// deliberately narrow enough to preserve.
    #[test]
    fn a_zero_run_in_the_middle_is_still_corruption() {
        let s = Scratch::new("wal-midzero");
        let (dir, _, offsets) = populated(&s, 5);
        let p = active(&dir);
        let full = std::fs::read(&p).unwrap();
        let at = (offsets[2] + SEG_HEADER_LEN) as usize;
        let mut spliced = full[..at].to_vec();
        spliced.extend_from_slice(&[0u8; 512]);
        spliced.extend_from_slice(&full[at..]);
        std::fs::write(&p, &spliced).unwrap();
        let e = Wal::replay(&dir, &schema()).unwrap_err();
        assert_eq!(e.code(), "CHECKSUM_MISMATCH", "{e}");
    }

    /// A block the filesystem allocated and never wrote back reads as a run of
    /// zeros past the last acknowledged byte. That is an interrupted append,
    /// not rot, and the log must open and replay its prefix -- otherwise the
    /// commonest crash shape on ext4 becomes a permanently unopenable table.
    #[test]
    fn a_block_sized_zero_tail_is_a_tear_and_the_log_still_opens() {
        let s = Scratch::new("wal-zeropad");
        let (dir, want, _) = populated(&s, 4);
        {
            let mut f = OpenOptions::new().append(true).open(active(&dir)).unwrap();
            f.write_all(&[0u8; 4096]).unwrap();
            f.sync_all().unwrap();
        }
        assert_eq!(Wal::replay(&dir, &schema()).unwrap(), want);
        let w = Wal::open(&dir).unwrap();
        assert!(w.len() > 0, "a zero-padded log must not become unopenable");
    }

    /// Damage inside a *sealed* segment is corruption whatever it looks like:
    /// the segment's `fsync` returned before its successor was published, so
    /// its whole contents were acknowledged and nothing in it can be an
    /// interrupted append. This is the case a point-in-time recovery hits, and
    /// the one that used to return `Ok` with half the records.
    #[test]
    fn a_corrupted_length_inside_a_sealed_segment_is_reported() {
        let _x = exclusive();
        let s = Scratch::new("wal-sealedrot");
        let dir = s.join("wal");
        let mut w = Wal::open(&dir).unwrap();
        let mut victim = 0u64;
        for i in 0..6u64 {
            let at = w.append_delete(i).unwrap();
            if i == 2 {
                victim = at;
            }
        }
        w.sync().unwrap();
        w.roll().unwrap();
        w.append_delete(99).unwrap();
        w.sync().unwrap();
        drop(w);

        let sealed = dir.join(seg_name(0));
        let before = std::fs::metadata(&sealed).unwrap().len();
        let mut bytes = std::fs::read(&sealed).unwrap();
        // Flip the high bit of a length varint: the file's length does not
        // change, so nothing structural gives it away.
        bytes[(victim + SEG_HEADER_LEN) as usize] |= 0x80;
        std::fs::write(&sealed, &bytes).unwrap();
        assert_eq!(std::fs::metadata(&sealed).unwrap().len(), before);

        let e = Wal::replay(&dir, &schema()).expect_err("a sealed segment cannot have a tail");
        assert!(e.to_string().contains("sealed"), "{e}");
        assert!(e.to_string().contains("silently"), "{e}");
    }

    /// The same defence from the other side: a sealed segment that stops
    /// short of what its own header says it acknowledged.
    #[test]
    fn a_sealed_segment_that_replays_short_of_its_stamp_is_reported() {
        let _x = exclusive();
        let s = Scratch::new("wal-sealedshort");
        let dir = s.join("wal");
        let mut w = Wal::open(&dir).unwrap();
        for i in 0..4u64 {
            w.append_delete(i).unwrap();
        }
        w.sync().unwrap();
        w.roll().unwrap();
        drop(w);
        let sealed = dir.join(seg_name(0));
        let mut bytes = std::fs::read(&sealed).unwrap();
        // The file keeps its length, so the chain still meets and nothing
        // structural gives it away: only the segment's own stamp does.
        let n = bytes.len();
        bytes[n - 17..].fill(0);
        std::fs::write(&sealed, &bytes).unwrap();
        let e = Wal::replay(&dir, &schema()).expect_err("a short sealed segment must report");
        assert!(e.to_string().contains("sealed"), "{e}");
        assert!(e.to_string().contains("silently"), "{e}");
    }

    /// The stamp is bound to the origin the file *name* declares, so a segment
    /// cannot be renamed or copied into another position and be believed.
    #[test]
    fn a_segment_cannot_be_moved_to_another_stream_position() {
        let _x = exclusive();
        let s = Scratch::new("wal-transplant");
        let (dir, _, _) = populated(&s, 4);
        let bytes = std::fs::read(active(&dir)).unwrap();
        let (_, trusted) = read_head(&bytes, 0).unwrap();
        assert!(trusted, "the stamp must verify where it was written");
        let (_, moved) = read_head(&bytes, 4096).unwrap();
        assert!(!moved, "and must not verify anywhere else");
    }

    #[test]
    fn a_bad_magic_is_corruption() {
        let s = Scratch::new("wal-magic");
        let (dir, _, _) = populated(&s, 3);
        let p = active(&dir);
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[1] ^= 0x40;
        std::fs::write(&p, &bytes).unwrap();
        let e = Wal::replay(&dir, &schema()).unwrap_err();
        assert!(e.to_string().contains("bad magic"), "{e}");
        assert!(Wal::open(&dir).is_err(), "opening a foreign file must refuse");
    }

    /// Both directions of a version skew, and neither may read as damage: an
    /// operator told to go looking for corruption over a format change loses
    /// time, and `load_catalog` would file it in a quarantine list.
    #[test]
    fn a_format_version_skew_is_refused_as_a_version_problem() {
        let s = Scratch::new("wal-version");
        let (dir, _, _) = populated(&s, 3);
        let p = active(&dir);
        let full = std::fs::read(&p).unwrap();
        for (v, expect) in [
            (format::FORMAT_VERSION + 1, "Upgrade granular"),
            (format::FORMAT_VERSION - 1, "must be recreated"),
        ] {
            let mut bytes = full.clone();
            bytes[8..12].copy_from_slice(&v.to_le_bytes());
            std::fs::write(&p, &bytes).unwrap();
            for e in [
                Wal::replay(&dir, &schema()).expect_err("replay"),
                Wal::open(&dir).err().expect("open"),
            ] {
                assert_eq!(e.code(), "FORMAT_VERSION", "{e}");
                let m = e.to_string();
                assert!(m.contains(expect), "{m}");
                assert!(!m.contains("corrupt"), "a version skew must not read as damage: {m}");
            }
        }
    }

    #[test]
    fn a_record_that_does_not_match_the_schema_is_refused() {
        use crate::types::{Field, Schema};
        let s = Scratch::new("wal-schema");
        let dir = s.join("wal");
        let mut w = Wal::open(&dir).unwrap();
        w.append_insert(&rows(&[1, 2])).unwrap();
        w.sync().unwrap();

        let narrow = Schema::new(vec![Field::new("id", DataType::UInt64)]).unwrap();
        let e = Wal::replay(&dir, &narrow).unwrap_err();
        assert!(e.to_string().contains("the table has 1"), "{e}");

        let wrong = Schema::new(vec![
            Field::new("id", DataType::UInt64),
            Field::new("host", DataType::Int64), // was String
            Field::new("ms", DataType::Int64),
            Field::new("ratio", DataType::Float64),
        ])
        .unwrap();
        let e = Wal::replay(&dir, &wrong).unwrap_err();
        assert!(e.to_string().contains("record column 1"), "{e}");
    }

    /// A hand-written frame, appended and then stamped, so it is
    /// *acknowledged* garbage rather than a torn tail.
    fn append_raw(dir: &Path, body: &[u8]) {
        let p = active(dir);
        let mut w = Writer::new();
        format::write_framed(&mut w, body);
        let bytes = w.finish();
        let len = std::fs::metadata(&p).unwrap().len() + bytes.len() as u64;
        {
            let mut f = OpenOptions::new().append(true).open(&p).unwrap();
            f.write_all(&bytes).unwrap();
        }
        let img = std::fs::read(&p).unwrap();
        restamp(&p, &img, len - SEG_HEADER_LEN);
    }

    #[test]
    fn an_unknown_record_tag_is_corruption() {
        let s = Scratch::new("wal-tag");
        let dir = s.join("wal");
        Wal::open(&dir).unwrap();
        // 200 rather than 9: the tag space grew a `TAG_MASK_RUN` under it.
        append_raw(&dir, &[200u8, 0, 0]);
        append_raw(&dir, &[TAG_DELETE, 0, 0, 0, 0, 0, 0, 0, 0]);
        let e = Wal::replay(&dir, &schema()).unwrap_err();
        assert!(e.to_string().contains("unknown log record tag 200"), "{e}");
    }

    /// The mask record's two halves are written by different modules --
    /// `storage::MaskRuns` encodes the gaps, `persist::format::Reader` decodes
    /// them -- so the round trip is the only thing holding them together.
    /// Dense and scattered runs across two parts, in one log.
    #[test]
    fn a_mask_run_round_trips_through_the_log() {
        let s = Scratch::new("wal-mask-roundtrip");
        let dir = s.join("wal");
        let mut w = Wal::open(&dir).unwrap();
        let mut m = MaskRuns::default();
        // Dense from zero, then a scatter with gaps past a varint boundary.
        for p in [0u64, 1, 2, 3] {
            m.hide(7, p);
        }
        for p in [5u64, 300, 301, 100_000, 1 << 20] {
            m.hide(9, p);
        }
        w.append_masks(None, &m).unwrap();
        w.sync().unwrap();
        assert_eq!(
            Wal::replay(&dir, &schema()).unwrap(),
            vec![
                WalRecord::Mask(7, vec![0, 1, 2, 3].into_boxed_slice()),
                WalRecord::Mask(9, vec![5, 300, 301, 100_000, 1 << 20].into_boxed_slice()),
            ]
        );
    }

    /// A dense run is one byte per row, against the keyed record's eight.
    /// This is the whole reason the unkeyed path can afford to log positions.
    #[test]
    fn a_dense_mask_run_costs_a_byte_a_row() {
        let s = Scratch::new("wal-mask-density");
        let dir = s.join("wal");
        let mut w = Wal::open(&dir).unwrap();
        let before = w.len();
        let mut m = MaskRuns::default();
        for p in 0..1_000u64 {
            m.hide(1, p);
        }
        w.append_masks(None, &m).unwrap();
        let per_row = (w.len() - before) as f64 / 1_000.0;
        assert!(per_row < 1.1, "{per_row} bytes per hidden row");
    }

    /// A staged mask run with no commit marker behind it is dropped, exactly
    /// as a staged insert or delete is. The sweep runs inside a transaction,
    /// so this is the ordinary crash-in-the-middle case.
    #[test]
    fn a_staged_mask_run_needs_its_commit() {
        let s = Scratch::new("wal-mask-staged");
        let dir = s.join("wal");
        let mut w = Wal::open(&dir).unwrap();
        let mut m = MaskRuns::default();
        m.hide(3, 4);
        let seq = w.begin();
        w.append_masks(Some(seq), &m).unwrap();
        w.sync().unwrap();
        assert!(Wal::replay(&dir, &schema()).unwrap().is_empty());
        w.commit(seq).unwrap();
        w.sync().unwrap();
        assert_eq!(
            Wal::replay(&dir, &schema()).unwrap(),
            vec![WalRecord::Mask(3, vec![4].into_boxed_slice())]
        );
    }

    /// A count that does not match the bytes behind it is corruption, not a
    /// prefix to salvage: half a mask run hides rows nobody asked to hide.
    #[test]
    fn a_mask_run_whose_count_disagrees_with_its_body_is_corruption() {
        for body in [
            // Two positions claimed, one gap byte present.
            vec![TAG_MASK_RUN, 1, 2, 0],
            // One claimed, two present.
            vec![TAG_MASK_RUN, 1, 1, 0, 0],
            // A count larger than the frame could possibly hold.
            vec![TAG_MASK_RUN, 1, 200, 0],
        ] {
            let s = Scratch::new("wal-mask-damage");
            let dir = s.join("wal");
            Wal::open(&dir).unwrap();
            append_raw(&dir, &body);
            let e = Wal::replay(&dir, &schema()).unwrap_err();
            assert!(matches!(e, Error::Corruption(_)), "{body:?}: {e}");
        }
    }

    #[test]
    fn replaying_a_missing_log_yields_nothing() {
        let s = Scratch::new("wal-absent");
        assert!(Wal::replay(&s.join("nope"), &schema()).unwrap().is_empty());
    }

    #[test]
    fn sync_is_safe_to_repeat_and_survives_a_reopen() {
        let s = Scratch::new("wal-sync");
        let dir = s.join("wal");
        let mut w = Wal::open(&dir).unwrap();
        w.sync().unwrap();
        w.append_delete(1).unwrap();
        w.sync().unwrap();
        w.sync().unwrap();
        let on_disk = std::fs::metadata(active(&dir)).unwrap().len();
        assert_eq!(on_disk - SEG_HEADER_LEN, w.len(), "a synced log is exactly as long as we think");
        drop(w);
        assert_eq!(Wal::replay(&dir, &schema()).unwrap(), vec![WalRecord::Delete(1)]);
    }

    /// The stamp is the classifier's only input, so it has to be exact at
    /// every `fsync` on every path that syncs -- not merely non-decreasing.
    #[test]
    fn every_sync_leaves_the_stamp_at_the_stream_end() {
        let _x = exclusive();
        let s = Scratch::new("wal-stamp");
        let dir = s.join("wal");
        let mut w = Wal::open(&dir).unwrap();
        let stamped = |dir: &Path| {
            let o = *seg_list(dir).unwrap().last().unwrap();
            let b = std::fs::read(dir.join(seg_name(o))).unwrap();
            read_head(&b, o).unwrap().0.durable
        };
        w.append_delete(1).unwrap();
        w.sync().unwrap();
        assert_eq!(stamped(&dir), w.len(), "after sync");

        let seq = w.begin();
        w.append_delete_staged(seq, 2).unwrap();
        w.commit(seq).unwrap();
        w.sync().unwrap();
        assert_eq!(stamped(&dir), w.len(), "after a staged commit");

        w.append_delete(3).unwrap();
        w.roll().unwrap();
        assert_eq!(stamped(&dir), w.len(), "after a roll");
        assert_eq!(stamped(&dir), w.origin(), "the fresh segment is acknowledged to its origin");
    }

    #[test]
    fn delete_lanes_roundtrip_at_the_extremes() {
        let s = Scratch::new("wal-lanes");
        let dir = s.join("wal");
        let lanes = [0u64, 1, u64::MAX, u64::MAX / 2, 1 << 63];
        let mut w = Wal::open(&dir).unwrap();
        for &l in &lanes {
            w.append_delete(l).unwrap();
        }
        w.sync().unwrap();
        let one: Vec<u64> = Wal::replay(&dir, &schema())
            .unwrap()
            .into_iter()
            .map(|r| match r {
                WalRecord::Delete(l) => l,
                _ => panic!("expected a delete"),
            })
            .collect();
        assert_eq!(one, lanes);
        // ...and identically through the run encoding.
        let dir2 = s.join("wal2");
        let mut w = Wal::open(&dir2).unwrap();
        w.append_deletes(&lanes).unwrap();
        w.sync().unwrap();
        let run: Vec<u64> = Wal::replay(&dir2, &schema())
            .unwrap()
            .into_iter()
            .map(|r| match r {
                WalRecord::Delete(l) => l,
                _ => panic!("expected a delete"),
            })
            .collect();
        assert_eq!(run, lanes);
    }

    #[test]
    fn a_log_in_a_directory_that_does_not_exist_yet_is_created() {
        let s = Scratch::new("wal-mkdir");
        let dir = s.join("db").join("t");
        let mut w = Wal::open(&dir).unwrap();
        w.append_delete(3).unwrap();
        w.sync().unwrap();
        assert_eq!(Wal::replay(&dir, &schema()).unwrap(), vec![WalRecord::Delete(3)]);
    }

    #[test]
    fn records_are_framed_and_checksummed_individually() {
        // Two records must be independently verifiable: damaging the first
        // must not be reported as damage to the second.
        let s = Scratch::new("wal-frames");
        let (dir, _, offsets) = populated(&s, 2);
        let bytes = std::fs::read(active(&dir)).unwrap();
        assert_eq!(offsets.len(), 3);
        // Each record: varint length + 8-byte checksum + body.
        for w in offsets.windows(2) {
            let (a, b) = ((w[0] + SEG_HEADER_LEN) as usize, (w[1] + SEG_HEADER_LEN) as usize);
            let mut r = Reader::new(&bytes[a..b]);
            let len = r.varint().unwrap();
            let _sum = r.u64().unwrap();
            assert_eq!(r.pos() + len as usize, b - a, "record frame must be exact");
        }
    }

    /// A crash mid-append leaves a torn tail. `Wal::open` must rewind to the
    /// last intact boundary before appending, or the next record -- which is
    /// acknowledged to the client after `sync()` -- would sit behind a frame
    /// that can never be parsed.
    #[test]
    fn reopening_after_a_torn_tail_rewinds_before_appending() {
        let s = Scratch::new("wal-adv-reopen-torn");
        let (dir, want, offsets) = populated(&s, 4);
        let p = active(&dir);
        let full = std::fs::read(&p).unwrap();

        // Crash halfway through appending the last record, with the stamp
        // where the last completed `sync` left it.
        let cut = ((offsets[3] + offsets[4]) / 2 + SEG_HEADER_LEN) as usize;
        restamp(&p, &full[..cut], offsets[3]);
        assert_eq!(Wal::replay(&dir, &schema()).unwrap(), want[..3]);

        let mut w = Wal::open(&dir).unwrap();
        assert_eq!(w.len(), offsets[3], "open must rewind to the last intact boundary");
        w.append_delete(0xDEAD_BEEF).unwrap();
        w.sync().unwrap();

        match Wal::replay(&dir, &schema()) {
            Err(e) => panic!("log became unreadable after a legal restart: {e}"),
            Ok(recs) => assert!(
                recs.contains(&WalRecord::Delete(0xDEAD_BEEF)),
                "the acknowledged post-restart record was silently dropped: {} replayed",
                recs.len()
            ),
        }
    }

    /// The same shape through the public API only: append, sync, simulate the
    /// interrupted append by writing a partial frame with the raw handle (so
    /// it lands *above* the stamp, which is what makes it a tear), restart,
    /// append again.
    #[test]
    fn a_partial_frame_then_a_restart_keeps_the_log_usable() {
        let s = Scratch::new("wal-adv-partial");
        let dir = s.join("wal");
        {
            let mut w = Wal::open(&dir).unwrap();
            w.append_delete(1).unwrap();
            w.sync().unwrap();
        }
        let mut torn = Writer::new();
        format::write_framed(&mut torn, &[TAG_DELETE, 0, 0, 0, 0, 0, 0, 0, 0]);
        let torn = torn.finish();
        {
            let mut f = OpenOptions::new().append(true).open(active(&dir)).unwrap();
            f.write_all(&torn[..torn.len() - 4]).unwrap();
            f.sync_all().unwrap();
        }
        assert_eq!(
            Wal::replay(&dir, &schema()).unwrap(),
            vec![WalRecord::Delete(1)],
            "the torn tail alone must replay the intact prefix"
        );

        let mut w = Wal::open(&dir).unwrap();
        w.append_delete(2).unwrap();
        w.sync().unwrap();
        match Wal::replay(&dir, &schema()) {
            Err(e) => panic!("log unreadable after restart: {e}"),
            Ok(r) => assert!(
                r.contains(&WalRecord::Delete(2)),
                "post-restart record lost, replayed {r:?}"
            ),
        }
    }

    // ---- staged records --------------------------------------------------

    /// The bug: the record is fsynced *before* the mutation is attempted, so a
    /// statement that then fails leaves a durable record of a write that never
    /// happened. Replay would resurrect it.
    #[test]
    fn a_staged_record_that_was_never_committed_is_not_replayed() {
        let s = Scratch::new("wal-staged-drop");
        let dir = s.join("wal");
        let mut w = Wal::open(&dir).unwrap();

        w.append_delete(1).unwrap(); // an ordinary, already-committed write
        let seq = w.begin();
        w.append_insert_staged(seq, &rows(&[100, 101])).unwrap();
        w.sync().unwrap();
        // ...and here the mutation is rejected: no commit marker is written.

        assert_eq!(
            Wal::replay(&dir, &schema()).unwrap(),
            vec![WalRecord::Delete(1)],
            "an uncommitted staged record must not be replayed"
        );
    }

    #[test]
    fn a_committed_staged_group_replays_in_log_order() {
        let s = Scratch::new("wal-staged-commit");
        let dir = s.join("wal");
        let mut w = Wal::open(&dir).unwrap();

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
            Wal::replay(&dir, &schema()).unwrap(),
            vec![WalRecord::Insert(b), WalRecord::Delete(42), WalRecord::Delete(9)],
            "a committed group must keep its position in the log, not move to the marker"
        );
    }

    /// The marker releases *its* group and nothing else. A later statement
    /// succeeding must not resurrect an earlier one that failed.
    #[test]
    fn a_commit_marker_releases_only_its_own_group() {
        let s = Scratch::new("wal-staged-scoped");
        let dir = s.join("wal");
        let mut w = Wal::open(&dir).unwrap();

        let failed = w.begin();
        w.append_delete_staged(failed, 0xBAD).unwrap();
        let ok = w.begin();
        w.append_delete_staged(ok, 0x600D).unwrap();
        w.commit(ok).unwrap();
        w.sync().unwrap();

        assert_eq!(
            Wal::replay(&dir, &schema()).unwrap(),
            vec![WalRecord::Delete(0x600D)],
            "the failed group must stay dropped"
        );
    }

    /// Sequence numbers are resumed from the segment header, so a group
    /// orphaned by a crash cannot be released by a marker written after the
    /// restart -- which is exactly what would happen if `begin` restarted.
    #[test]
    fn a_restart_cannot_release_a_group_orphaned_before_it() {
        let s = Scratch::new("wal-staged-restart");
        let dir = s.join("wal");
        {
            let mut w = Wal::open(&dir).unwrap();
            let seq = w.begin();
            assert_eq!(seq, 0, "a fresh log starts at zero");
            w.append_delete_staged(seq, 0xBAD).unwrap();
            w.sync().unwrap();
        }
        let mut w = Wal::open(&dir).unwrap();
        let seq = w.begin();
        assert_ne!(seq, 0, "open must resume past the sequence numbers in the file");
        w.append_delete_staged(seq, 0x600D).unwrap();
        w.commit(seq).unwrap();
        w.sync().unwrap();

        assert_eq!(Wal::replay(&dir, &schema()).unwrap(), vec![WalRecord::Delete(0x600D)]);
    }

    /// ...and across a *roll*, which is the case the header field made
    /// unconditional. The old layout only fenced the counter when the
    /// truncation happened to be carrying decisions.
    #[test]
    fn a_roll_cannot_release_a_group_orphaned_before_it() {
        let _x = exclusive();
        let s = Scratch::new("wal-staged-roll");
        let dir = s.join("wal");
        let orphan = {
            let mut w = Wal::open(&dir).unwrap();
            let seq = w.begin();
            w.append_delete_staged(seq, 0xBAD).unwrap();
            w.sync().unwrap();
            seq
        };
        // The crash left the group open; a fresh handle inherits the counter
        // but not the group, so the roll is allowed.
        let mut w = Wal::open(&dir).unwrap();
        w.roll().unwrap();
        assert!(w.begin() > orphan, "the counter must not restart across a roll");
        assert!(Wal::replay(&dir, &schema()).unwrap().is_empty());
    }

    /// Reopening a log whose tail is a staged group must not truncate it away
    /// or refuse: staging is a body-level property, and `open` deals in frames.
    #[test]
    fn opening_a_log_that_ends_in_a_staged_group_is_clean() {
        let s = Scratch::new("wal-staged-open");
        let dir = s.join("wal");
        let mut w = Wal::open(&dir).unwrap();
        let seq = w.begin();
        w.append_delete_staged(seq, 5).unwrap();
        w.sync().unwrap();
        let len = w.len();
        drop(w);

        let mut w = Wal::open(&dir).unwrap();
        assert_eq!(w.len(), len, "an intact staged record is not a torn tail");
        let fresh = w.begin();
        w.commit(fresh).unwrap();
        w.sync().unwrap();
        assert!(Wal::replay(&dir, &schema()).unwrap().is_empty());
    }

    /// The staged flag rides the high bit of the tag byte; a tag that sets it
    /// over an unknown value must still be reported with the byte that was
    /// actually there.
    #[test]
    fn an_unknown_staged_tag_names_the_raw_byte() {
        let s = Scratch::new("wal-staged-tag");
        let dir = s.join("wal");
        Wal::open(&dir).unwrap();
        append_raw(&dir, &[STAGED | 72, 0]);
        let e = Wal::replay(&dir, &schema()).unwrap_err();
        assert!(e.to_string().contains(&format!("tag {}", STAGED | 72)), "{e}");
    }

    // ---- LSNs --------------------------------------------------------------

    /// The whole contract in one test: the number an append hands back is the
    /// number replay reports, is the stream position the frame starts at, and
    /// is what `len()` said a moment earlier.
    #[test]
    fn an_lsn_is_the_offset_replay_finds_the_record_at() {
        let s = Scratch::new("wal-lsn");
        let dir = s.join("wal");
        let mut w = Wal::open(&dir).unwrap();
        let mut want = Vec::new();
        for i in 0..12u64 {
            let before = w.lsn();
            assert_eq!(before, w.len(), "lsn() and len() are the same number");
            let lsn = if i % 3 == 2 {
                w.append_delete(i).unwrap()
            } else {
                w.append_insert(&rows(&[i * 10, i * 10 + 1])).unwrap()
            };
            assert_eq!(lsn, before, "an append returns the stream position before it");
            want.push(lsn);
        }
        w.sync().unwrap();

        let got = Wal::replay_with_lsn(&dir, &schema(), 0).unwrap();
        assert_eq!(got.len(), want.len());
        for (i, ((lsn, _), &expect)) in got.iter().zip(&want).enumerate() {
            assert_eq!(*lsn, expect, "record {i}");
            assert_eq!(
                Wal::replay(&dir, &schema()).unwrap().len() - i,
                Wal::replay_from(&dir, &schema(), *lsn).unwrap().len(),
                "record {i}: an LSN is a valid watermark"
            );
        }
    }

    /// A committed staged group keeps the LSN of the record, not of the marker
    /// that released it.
    #[test]
    fn a_staged_record_keeps_its_own_lsn() {
        let s = Scratch::new("wal-lsn-staged");
        let dir = s.join("wal");
        let mut w = Wal::open(&dir).unwrap();
        let seq = w.begin();
        let at = w.append_delete_staged(seq, 77).unwrap();
        let marker = w.commit(seq).unwrap();
        w.sync().unwrap();
        assert!(marker > at);
        let got = Wal::replay_with_lsn(&dir, &schema(), 0).unwrap();
        assert_eq!(got, vec![(at, WalRecord::Delete(77))]);
    }

    // ---- rewind ------------------------------------------------------------

    /// ROLLBACK's half: rewinding to the LSN a transaction started at makes
    /// the directory byte-identical to what it was, not merely semantically
    /// equal.
    #[test]
    fn rewinding_restores_the_log_byte_for_byte() {
        let s = Scratch::new("wal-rewind");
        let (dir, want, _) = populated(&s, 6);
        let before = digest(&dir);

        let mut w = Wal::open(&dir).unwrap();
        let mark = w.lsn();
        let seq = w.begin();
        w.append_insert_staged(seq, &rows(&[1, 2, 3])).unwrap();
        w.append_delete_staged(seq, 9).unwrap();
        w.sync().unwrap();
        assert!(w.len() > mark, "the aborting transaction really did write");

        w.rewind_to(mark).unwrap();
        assert_eq!(w.len(), mark);
        assert_eq!(digest(&dir), before, "no trace on disk");
        assert_eq!(Wal::replay(&dir, &schema()).unwrap(), want);

        assert_eq!(w.append_delete(5).unwrap(), mark);
        w.sync().unwrap();
        let mut expect = want.clone();
        expect.push(WalRecord::Delete(5));
        assert_eq!(Wal::replay(&dir, &schema()).unwrap(), expect);
    }

    /// Every file in `dir` with its *record* bytes. A rewind is a claim about
    /// the whole log directory rather than about one file -- but not about the
    /// segment header, which carries a durability watermark that legitimately
    /// moves when a transaction syncs and is then discarded. The claim is that
    /// no record and no byte of length survives.
    fn digest(dir: &Path) -> Vec<(String, Vec<u8>)> {
        let mut out: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| {
                let b = std::fs::read(e.path()).unwrap();
                let head = (SEG_HEADER_LEN as usize).min(b.len());
                (e.file_name().to_string_lossy().into_owned(), b[head..].to_vec())
            })
            .collect();
        out.sort();
        out
    }

    #[test]
    fn rewinding_forward_or_to_the_end_is_a_no_op() {
        let s = Scratch::new("wal-rewind-noop");
        let (dir, want, _) = populated(&s, 3);
        let mut w = Wal::open(&dir).unwrap();
        let len = w.len();
        w.rewind_to(len).unwrap();
        w.rewind_to(len + 4096).unwrap();
        assert_eq!(w.len(), len);
        assert_eq!(Wal::replay(&dir, &schema()).unwrap(), want);
    }

    /// The floor is the segment's *body start*, not its origin. Carried
    /// decision markers live below it, and a rewind through them would make
    /// every sibling's prepare read "no decision" -- abort -- and silently
    /// drop a committed transaction in the other tables.
    #[test]
    fn rewinding_below_the_segments_first_record_is_refused() {
        let _x = exclusive();
        let s = Scratch::new("wal-rewind-floor");
        let (dir, want, _) = populated(&s, 2);
        let mut w = Wal::open(&dir).unwrap();
        w.roll().unwrap();
        let origin = w.origin();
        assert!(w.rewind_to(origin - 1).is_err(), "below the active segment");
        assert!(w.rewind_to(0).is_err(), "into a sealed segment");
        assert_eq!(Wal::replay(&dir, &schema()).unwrap(), want, "and nothing was lost");
    }

    /// No transaction may span a roll, and it is enforced where it could be
    /// broken rather than suffered where it would surface -- which is a
    /// `ROLLBACK` returning an error that `Session::commit`'s failure arm
    /// swallows whole.
    #[test]
    fn a_roll_is_refused_while_a_staging_group_is_open() {
        let _x = exclusive();
        let s = Scratch::new("wal-roll-guard");
        let dir = s.join("wal");
        let mut w = Wal::open(&dir).unwrap();
        w.append_delete(1).unwrap();
        w.sync().unwrap();
        let seq = w.begin();
        w.append_delete_staged(seq, 2).unwrap();
        let e = w.roll().expect_err("a roll under an open group must refuse");
        assert!(e.to_string().contains("staging group"), "{e}");
        // ...and once the group is resolved, it goes through.
        w.commit(seq).unwrap();
        w.sync().unwrap();
        w.roll().unwrap();
    }

    #[test]
    fn the_lane_of_a_value_is_what_gets_logged() {
        let s = Scratch::new("wal-signed");
        let dir = s.join("wal");
        let lane = Value::Int(-42).to_lane(&DataType::Int64).unwrap();
        let mut w = Wal::open(&dir).unwrap();
        w.append_delete(lane).unwrap();
        w.sync().unwrap();
        assert_eq!(Wal::replay(&dir, &schema()).unwrap(), vec![WalRecord::Delete(lane)]);
    }

    // ---- two-phase commit -------------------------------------------------

    /// A log directory in the layout `may_be_cited` and the citations
    /// understand: `<root>/.wal/<db>/<table>`, under a root with a `CATALOG`.
    fn table_wal(s: &Scratch, name: &str) -> PathBuf {
        std::fs::write(s.join(store::CATALOG_FILE), b"not read here").unwrap();
        std::fs::create_dir_all(s.path().join("db").join(name)).unwrap();
        let d = wal_dir(s.path(), "db", name);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Declare `name`'s parts to cover `covered` of its stream, which is what
    /// a checkpoint records and what `may_be_cited` reads.
    fn mark_covered(s: &Scratch, name: &str, covered: u64) {
        let dir = s.path().join("db").join(name);
        let doc = super::super::writer::table_doc(&table_def(name), &[], covered);
        std::fs::write(dir.join(store::TABLE_FILE), doc).unwrap();
    }

    /// Two staged groups, one per log, committed with `b` prepared against
    /// `a`'s decision.
    fn two_phase(s: &Scratch) -> (PathBuf, PathBuf) {
        let (da, db) = (table_wal(s, "a"), table_wal(s, "b"));
        let (mut a, mut b) = (Wal::open(&da).unwrap(), Wal::open(&db).unwrap());
        let (sa, sb) = (a.begin(), b.begin());
        a.append_insert_staged(sa, &rows(&[10, 11])).unwrap();
        b.append_delete_staged(sb, 77).unwrap();
        // Participants first, each durable before the decision exists; the
        // coordinator's marker last, and it is the commit point.
        b.prepare(sb, &da, sa).unwrap();
        b.sync().unwrap();
        a.decide(sa).unwrap();
        a.sync().unwrap();
        (da, db)
    }

    fn replayed(dir: &Path) -> usize {
        Wal::replay(dir, &schema()).unwrap().len()
    }

    /// The transaction turns on one record in one file, so a participant that
    /// is already `fsync`ed still drops.
    #[test]
    fn a_prepared_group_waits_for_the_decision_it_cites() {
        let s = Scratch::new("wal-2pc");
        let (da, db) = (table_wal(&s, "a"), table_wal(&s, "b"));
        let (mut a, mut b) = (Wal::open(&da).unwrap(), Wal::open(&db).unwrap());
        let (sa, sb) = (a.begin(), b.begin());
        a.append_delete_staged(sa, 1).unwrap();
        b.append_delete_staged(sb, 2).unwrap();
        b.prepare(sb, &da, sa).unwrap();
        b.sync().unwrap();

        // A crash here has fsynced the participant and not the coordinator.
        assert_eq!(replayed(&db), 0, "a prepare with no decision must not commit");
        assert_eq!(replayed(&da), 0);

        a.decide(sa).unwrap();
        a.sync().unwrap();
        assert_eq!(Wal::replay(&db, &schema()).unwrap(), vec![WalRecord::Delete(2)]);
        assert_eq!(
            Wal::replay(&da, &schema()).unwrap(),
            vec![WalRecord::Delete(1)],
            "the decision is also the coordinator's own commit marker"
        );
    }

    /// Every crash point in the commit sequence, byte by byte, over both logs
    /// at once -- and again with a roll forced between the two participants,
    /// which is the state a checkpoint racing a multi-table commit would leave.
    #[test]
    fn every_prefix_of_a_two_phase_commit_is_all_or_nothing() {
        let _x = exclusive();
        for roll_between in [false, true] {
            let s = Scratch::new(&format!("wal-2pc-sweep-{roll_between}"));
            let (da, db) = (table_wal(&s, "a"), table_wal(&s, "b"));
            let (la, lb);
            {
                let (mut a, mut b) = (Wal::open(&da).unwrap(), Wal::open(&db).unwrap());
                if roll_between {
                    a.append_delete(0).unwrap();
                    a.sync().unwrap();
                    a.roll().unwrap();
                }
                let (sa, sb) = (a.begin(), b.begin());
                a.append_insert_staged(sa, &rows(&[10, 11])).unwrap();
                b.append_delete_staged(sb, 77).unwrap();
                la = a.len();
                lb = b.len();
                b.prepare(sb, &da, sa).unwrap();
                b.sync().unwrap();
                a.decide(sa).unwrap();
                a.sync().unwrap();
            }
            let (pa, pb) = (active(&da), active(&db));
            let (oa, ob) = (
                *seg_list(&da).unwrap().last().unwrap(),
                *seg_list(&db).unwrap().last().unwrap(),
            );
            let (fa, fb) = (std::fs::read(&pa).unwrap(), std::fs::read(&pb).unwrap());
            // Only what *this transaction* added: with a roll forced first,
            // the sealed segment still holds the record that preceded it.
            let after = |d: &Path, from: u64| Wal::replay_from(d, &schema(), from).unwrap().len();
            let mut committed = 0usize;
            // Phase 1: the participant's prepare is landing; the coordinator
            // is still at its pre-commit length. Both images are stamped to
            // their cut, which is what an interrupted append really looks like.
            for cut in (lb - ob + SEG_HEADER_LEN) as usize..=fb.len() {
                restamp(&pb, &fb[..cut], lb);
                restamp(&pa, &fa[..(la - oa + SEG_HEADER_LEN) as usize], la);
                assert_eq!(after(&db, ob), 0, "b at {cut} committed without a decision");
                assert_eq!(after(&da, oa), 0, "a at {cut} committed early");
            }
            // Phase 2: the participant is whole and the decision is landing.
            std::fs::write(&pb, &fb).unwrap();
            for cut in (la - oa + SEG_HEADER_LEN) as usize..=fa.len() {
                restamp(&pa, &fa[..cut], la);
                let (ra, rb) = (after(&da, oa), after(&db, ob));
                assert_eq!(
                    ra > 0,
                    rb > 0,
                    "a cut to {cut} of {}: a={ra}, b={rb} -- not a prefix of the transaction",
                    fa.len()
                );
                committed += usize::from(ra > 0);
            }
            assert!(committed > 0, "no cut committed: the fixture never wrote a decision");
        }
    }

    /// A checkpoint rolls a log. If it rolled the decisions away with it, a
    /// prepare somewhere else would become unresolvable -- which reads as
    /// "never committed" and silently drops a transaction that did.
    #[test]
    fn rolling_the_coordinator_keeps_the_decisions_a_prepare_still_cites() {
        let _x = exclusive();
        let s = Scratch::new("wal-2pc-carry");
        let (da, db) = two_phase(&s);
        assert_eq!(replayed(&db), 1);

        // `b` has records its parts do not cover, so its prepare is live.
        let after = |d: &Path| {
            let w = Wal::open(d).unwrap();
            Wal::replay_from(d, &schema(), w.origin()).unwrap().len()
        };
        Wal::open(&da).unwrap().roll().unwrap();
        assert_eq!(after(&da), 0, "the coordinator's own records are in parts now");
        assert_eq!(replayed(&db), 1, "the participant must still resolve to commit");
        // ...and again, across a reopen, since the carried decision has to
        // survive the scan that reads it back.
        Wal::open(&da).unwrap().roll().unwrap();
        assert_eq!(replayed(&db), 1);
    }

    /// A segment that carries decisions is an ordinary segment, and every path
    /// that reads only its 64-byte header has to say so.
    ///
    /// Four of the five callers of [`read_head`] hand it exactly the header,
    /// so a `carry_len` bound computed against *that* buffer read every
    /// carry-carrying segment as damaged -- and each caller then failed in its
    /// own quiet way. The worst was `live_extent`: it answers `None`, which
    /// [`roll_for_checkpoint`] turns into a watermark of **0**, so the next
    /// open replays the whole stream again and the one after that replays it
    /// once more. Rows the database had acknowledged multiplied on every
    /// restart, and a `DELETE`d row came back.
    ///
    /// So this asserts the header-only readers directly, on the one state that
    /// produces a non-zero `carry_len`: a coordinator rolled while a sibling's
    /// prepare still cites its decision.
    #[test]
    fn a_segment_that_carries_decisions_is_read_as_healthy_by_every_path() {
        let _x = exclusive();
        let s = Scratch::new("wal-2pc-carry-header");
        let (da, _db) = two_phase(&s);
        let (origin, carry) = {
            let mut a = Wal::open(&da).unwrap();
            a.roll().unwrap();
            (a.origin(), a.head.carry_len as u64)
        };
        assert!(carry > 0, "the fixture carried nothing, so this proves nothing");

        // The advisory extent, which is what a checkpoint asks. The floor is
        // past the carried decisions and the log holds nothing beyond them.
        let (end, floor) = live_extent(&da).expect("a carrying segment still has an extent");
        assert_eq!((end, floor), (origin + carry, origin + carry));
        // ...so the checkpoint records the stream end, not zero.
        assert_eq!(roll_for_checkpoint(&da).unwrap(), origin + carry);

        // The archive listing, the horizon, and the replay path that resumes
        // the recovery-LSN counter -- all of them read the same header.
        let segs = segments(s.path(), "db", "a").expect("a carrying segment is not a hole");
        assert_eq!(segs.last().map(|g| g.end), Some(origin));
        assert_eq!(archive_horizon(s.path(), "db", "a").unwrap(), 0, "nothing has been pruned");
        assert!(Wal::replay_from(&da, &schema(), origin).unwrap().is_empty());

        // And the bound itself still exists, where the whole file is in hand:
        // a header that claims more carried decisions than the segment holds
        // would walk `floor()` past them into records.
        let path = da.join(seg_name(origin));
        let mut img = std::fs::read(&path).unwrap();
        let mut head = Head { carry_len: carry as u32 + 1, ..Default::default() };
        head.durable = origin + carry;
        head.acked = head.durable;
        img[..SEG_HEADER_LEN as usize].copy_from_slice(&head.encode(origin));
        std::fs::write(&path, &img).unwrap();
        let e = match Seg::load(&da, origin, false) {
            Err(e) => e,
            Ok(_) => panic!("an overrunning carry must be refused"),
        };
        assert!(e.to_string().contains("carried decisions"), "{e}");
    }

    /// The other half: once every other log is inside its table's parts,
    /// nothing can cite a decision and it goes.
    #[test]
    fn rolling_drops_decisions_once_every_other_log_is_covered() {
        let _x = exclusive();
        let s = Scratch::new("wal-2pc-drop");
        let (da, db) = two_phase(&s);
        mark_covered(&s, "b", Wal::open(&db).unwrap().len());

        let mut a = Wal::open(&da).unwrap();
        a.roll().unwrap();
        assert!(a.is_empty(), "a log nothing can cite must roll to an empty segment");
        assert_eq!(a.head.carry_len, 0);
    }

    /// An idle sibling must answer "not citable" *exactly*, or decisions
    /// accumulate for ever and a full data-root walk runs at every checkpoint
    /// without ever being able to answer no. This is what puts the head fields
    /// in a fixed header instead of a framed record.
    #[test]
    fn a_checkpointed_sibling_does_not_look_live() {
        let _x = exclusive();
        let s = Scratch::new("wal-2pc-idle");
        let (da, db) = two_phase(&s);
        // A full checkpoint: roll every log and record the watermark each one
        // produced, which is exactly what `save_catalog` does.
        for (name, d) in [("a", &da), ("b", &db)] {
            let mut w = Wal::open(d).unwrap();
            w.roll().unwrap();
            mark_covered(&s, name, w.origin());
        }
        assert!(!Wal::open(&da).unwrap().may_be_cited(), "an idle sibling is not live");
        assert_eq!(archive_lags(s.path()).unwrap(), None, "and the archive does not lag");
    }

    /// A citation is read out of a file and handed to `read_dir`, so it is a
    /// hostile input like every other field on disk.
    #[test]
    fn a_citation_that_is_not_a_sibling_log_is_refused() {
        let s = Scratch::new("wal-2pc-path");
        let db = table_wal(&s, "b");
        for rel in [
            "/etc/passwd",
            "../../../../../../../../../../etc",
            "..",
            "",
            "../../db",
            "../c",
        ] {
            let _ = std::fs::remove_dir_all(&db);
            let mut w = Wal::open(&db).unwrap();
            let seq = w.begin();
            w.append_delete_staged(seq, 5).unwrap();
            let mut body = w.body(64);
            body.w.u8(TAG_PREPARE);
            body.w.varint(seq);
            body.w.varint(0);
            body.w.str(rel);
            w.append(body).unwrap();
            w.sync().unwrap();
            let got = Wal::replay(&db, &schema());
            if rel == "../c" {
                // A citation that stays legal but names a log that is not
                // there is not damage: it is a decision that was never
                // written, which is an abort.
                assert!(got.unwrap().is_empty(), "{rel} must not commit");
            } else {
                let e = got.expect_err("a bogus citation must be refused");
                assert!(e.to_string().contains("citation"), "{rel}: {e}");
            }
        }
    }

    #[test]
    fn a_citation_names_the_coordinator_relative_to_the_citing_log() {
        let s = Scratch::new("wal-2pc-rel");
        let (da, db) = (table_wal(&s, "a"), table_wal(&s, "b"));
        assert_eq!(relative_dir(&db, &da).unwrap(), "../a");
        assert_eq!(resolve_citation(&db, "../a").unwrap(), da);
        // A transaction may span databases, so a citation may have to climb
        // two levels rather than one.
        let other = wal_dir(s.path(), "db2", "a");
        assert_eq!(relative_dir(&db, &other).unwrap(), "../../db2/a");
        assert_eq!(resolve_citation(&db, "../../db2/a").unwrap(), other);
        // Two paths that cannot be compared are refused rather than guessed
        // at, and so is a target that is not below anything.
        assert!(relative_dir(&da, Path::new("db/a")).is_err());
        assert!(relative_dir(Path::new("/x/a"), Path::new("/x/a")).is_err());
    }

    /// A moved -- or restored -- data directory still resolves, which is the
    /// reason the citation is relative and not absolute.
    #[test]
    fn a_two_phase_commit_survives_the_directory_being_moved() {
        let s = Scratch::new("wal-2pc-move");
        let (_, db) = two_phase(&s);
        assert_eq!(replayed(&db), 1);
        let moved = s.join("copy");
        copy_dir(&s.path().join(WAL_DIR), &moved.join(WAL_DIR));
        std::fs::create_dir_all(&moved).unwrap();
        std::fs::copy(s.join(store::CATALOG_FILE), moved.join(store::CATALOG_FILE)).unwrap();
        assert_eq!(replayed(&wal_dir(&moved, "db", "b")), 1);
    }

    fn copy_dir(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for e in std::fs::read_dir(from).unwrap().flatten() {
            let (src, dst) = (e.path(), to.join(e.file_name()));
            if e.file_type().unwrap().is_dir() {
                copy_dir(&src, &dst);
            } else {
                std::fs::copy(&src, &dst).unwrap();
            }
        }
    }

    // ---- ticks and the archive ---------------------------------------------

    /// Retention and the recovery-LSN counter are process-global by design --
    /// they describe a data directory, and there is one writer per directory.
    /// `cargo test` puts several of those directories in one process, so the
    /// tests that lean on either one take this first.
    static ARCHIVE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn exclusive() -> std::sync::MutexGuard<'static, ()> {
        ARCHIVE.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A tick per `fsync`, not per record, and none at all when there is
    /// nothing behind it: an idle table `sync`ed in a loop must not grow a log.
    #[test]
    fn a_tick_stamps_a_group_not_a_record() {
        let s = Scratch::new("wal-tick");
        let dir = s.join("wal");
        let mut w = Wal::open(&dir).unwrap();
        for i in 0..5u64 {
            w.append_delete(i).unwrap();
        }
        w.sync().unwrap();
        let after = w.len();
        w.sync().unwrap();
        w.sync().unwrap();
        assert_eq!(w.len(), after, "a sync with nothing behind it must write nothing");

        let ticks = ticks_in(&dir);
        assert_eq!(ticks.len(), 1, "one fsync, one tick: {ticks:?}");
        assert!(ticks[0].0 > 0 && ticks[0].1 > 0);
        assert_eq!(Wal::replay(&dir, &schema()).unwrap().len(), 5);

        w.append_delete(99).unwrap();
        w.sync().unwrap();
        let ticks = ticks_in(&dir);
        assert_eq!(ticks.len(), 2);
        assert!(ticks[1].0 > ticks[0].0, "recovery LSNs are strictly increasing: {ticks:?}");
        assert!(ticks[1].1 >= ticks[0].1, "and the clock column never goes backwards");
    }

    fn ticks_in(dir: &Path) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        let origins = seg_list(dir).unwrap();
        for (i, &o) in origins.iter().enumerate() {
            let seg = Seg::load(dir, o, i + 1 < origins.len()).unwrap();
            walk_seg(&seg, 0, |_, b| {
                if let Some(t) = tick_of(b) {
                    out.push(t);
                }
                Ok(())
            })
            .unwrap();
        }
        out
    }

    /// A checkpoint retires the log into the archive, and the segments chain
    /// by stream position with no gap.
    #[test]
    fn rolling_archives_the_log_and_chains_the_stream() {
        let _x = exclusive();
        let s = Scratch::new("wal-archive");
        let d = table_wal(&s, "t");
        let mut want = Vec::new();
        for gen in 0..4u64 {
            let mut w = Wal::open(&d).unwrap();
            let origin = w.origin();
            w.append_delete(gen).unwrap();
            w.sync().unwrap();
            want.push(WalRecord::Delete(gen));
            w.roll().unwrap();
            assert!(w.origin() > origin, "generation {gen}");
            assert!(w.is_empty());
        }
        let segs = segments(s.path(), "db", "t").unwrap();
        assert_eq!(segs.len(), 4);
        assert_eq!(segs[0].origin, 0);
        for (i, w) in segs.windows(2).enumerate() {
            assert_eq!(w[0].end, w[1].origin, "segments {i} and {} must meet", i + 1);
            assert!(w[0].span.last_seq <= w[1].span.first_seq, "and their stamps must too");
        }
        // The whole stream replays to exactly what was written, in order.
        let rec = recover(s.path(), "db", "t", 0, Target::Latest).unwrap();
        assert_eq!(rec.records, 4);
        assert_eq!(replay_image(&s, &rec.bytes), want);
    }

    /// Replay a `Recovered` image the way `restore_until` will: write it into
    /// a fresh log directory and let the ordinary loader read it.
    fn replay_image(s: &Scratch, bytes: &[u8]) -> Vec<WalRecord> {
        let d = s.join(&format!("restored-{}", bytes.len()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(recovered_seg_name()), bytes).unwrap();
        Wal::replay(&d, &schema()).unwrap()
    }

    /// The cut is a pure function of the archive, on both axes, and it always
    /// lands on a tick -- so every recovered state is one the database really
    /// was in.
    #[test]
    fn a_recovery_cut_is_exact_on_both_axes() {
        let _x = exclusive();
        let s = Scratch::new("wal-cut");
        let d = table_wal(&s, "t");
        let mut stamps = Vec::new();
        for gen in 0..5u64 {
            let mut w = Wal::open(&d).unwrap();
            w.append_delete(gen).unwrap();
            w.sync().unwrap();
            stamps.push(w.span.last_seq);
            w.roll().unwrap();
        }
        for (i, &seq) in stamps.iter().enumerate() {
            let rec = recover(s.path(), "db", "t", 0, Target::Lsn(seq)).unwrap();
            assert_eq!(rec.records as usize, i + 1, "up to LSN {seq}");
            assert_eq!(rec.applied.last_seq, seq);
            // Twice is the same bytes: nothing here reads a clock.
            let again = recover(s.path(), "db", "t", 0, Target::Lsn(seq)).unwrap();
            assert_eq!(rec.bytes, again.bytes);
            // ...and starting *after* a stamp drops exactly what it covered.
            let from = recover(s.path(), "db", "t", seq + 1, Target::Latest).unwrap();
            assert_eq!(from.records as usize, stamps.len() - i - 1, "from LSN {}", seq + 1);
        }
        // A target before anything archived yields an empty log, not an error.
        let none = recover(s.path(), "db", "t", 0, Target::Lsn(0)).unwrap();
        assert_eq!(none.records, 0);
        assert_eq!(none.bytes.len(), SEG_HEADER_LEN as usize);
    }

    /// Retention is the difference between a feature and an outage. What it
    /// drops has to be recorded, or the recovery that needed it would replay a
    /// shorter history and say nothing. The record is the oldest surviving
    /// segment's own header -- derived, so there is no window in which it
    /// disagrees with the directory.
    #[test]
    fn retention_drops_the_oldest_and_the_horizon_follows_the_files() {
        let _x = exclusive();
        let s = Scratch::new("wal-retain");
        let d = table_wal(&s, "t");
        let keep = archive_retention();
        set_archive_retention(1);
        let mut first = 0u64;
        for gen in 0..6u64 {
            let mut w = Wal::open(&d).unwrap();
            w.append_insert(&rows(&[gen * 10, gen * 10 + 1])).unwrap();
            w.sync().unwrap();
            if gen == 0 {
                first = w.span.last_seq;
            }
            w.roll().unwrap();
        }
        set_archive_retention(keep);

        let segs = segments(s.path(), "db", "t").unwrap();
        assert!(segs.len() < 6, "a 1-byte budget must drop almost everything: {}", segs.len());
        assert!(segs.is_empty() || segs[0].origin > 0, "the survivors start after the hole");

        let h = archive_horizon(s.path(), "db", "t").unwrap();
        assert!(h >= first, "the horizon must record what went: {h} < {first}");
        let e = recover(s.path(), "db", "t", first, Target::Latest)
            .expect_err("a recovery that needs a dropped segment must refuse");
        assert!(e.to_string().contains("retention has dropped"), "{e}");
        // ...while one that starts after the horizon is still served.
        assert!(recover(s.path(), "db", "t", h + 1, Target::Latest).is_ok());
    }

    /// The count cap, which the byte budget cannot express: a roll per fold
    /// lets a delete-per-commit workload mint a segment per transaction, and
    /// 64 MiB of budget would admit ~800 000 of them.
    #[test]
    fn retention_caps_the_segment_count_as_well_as_the_bytes() {
        let _x = exclusive();
        let s = Scratch::new("wal-maxseg");
        let d = table_wal(&s, "t");
        let mut w = Wal::open(&d).unwrap();
        for gen in 0..(MAX_SEGMENTS as u64 + 20) {
            w.append_delete(gen).unwrap();
            w.sync().unwrap();
            w.roll().unwrap();
        }
        let n = seg_list(&d).unwrap().len();
        assert!(n <= MAX_SEGMENTS + 1, "{n} segments survived a {MAX_SEGMENTS} cap");
        assert!(n > MAX_SEGMENTS / 2, "the cap must not empty the archive: {n}");
    }

    /// A plain append is history the instant it is framed, so a crash before
    /// its `fsync` leaves records that replay applies and no tick covers. The
    /// checkpoint that retires them stamps them, or a recovery could not place
    /// them and would leave them out -- silently.
    #[test]
    fn records_a_crash_left_unstamped_are_stamped_when_they_are_archived() {
        let _x = exclusive();
        let s = Scratch::new("wal-unstamped");
        let d = table_wal(&s, "t");
        {
            let mut w = Wal::open(&d).unwrap();
            w.append_delete(7).unwrap(); // ...and no `sync`: the process dies here
        }
        assert_eq!(Wal::replay(&d, &schema()).unwrap(), vec![WalRecord::Delete(7)]);

        // A read-only session's exit checkpoint: it retires the log without
        // ever having logged anything of its own.
        Wal::open(&d).unwrap().roll().unwrap();
        let segs = segments(s.path(), "db", "t").unwrap();
        assert_eq!(segs.len(), 1);
        assert!(!segs[0].span.is_empty(), "the segment must carry a tick to be placeable");
        let rec = recover(s.path(), "db", "t", 0, Target::Latest).unwrap();
        assert_eq!(
            replay_image(&s, &rec.bytes),
            vec![WalRecord::Delete(7)],
            "an unstamped record must not be dropped by the recovery"
        );
    }

    /// A hole is the failure a recovery must never paper over.
    #[test]
    fn a_missing_segment_is_a_hole_with_a_named_range() {
        let _x = exclusive();
        let s = Scratch::new("wal-hole");
        let d = table_wal(&s, "t");
        for gen in 0..3u64 {
            let mut w = Wal::open(&d).unwrap();
            w.append_delete(gen).unwrap();
            w.sync().unwrap();
            w.roll().unwrap();
        }
        let segs = segments(s.path(), "db", "t").unwrap();
        std::fs::remove_file(d.join(seg_name(segs[1].origin))).unwrap();
        let e = segments(s.path(), "db", "t").expect_err("a hole must be reported");
        assert!(e.to_string().contains("hole"), "{e}");
        assert!(e.to_string().contains(&segs[1].origin.to_string()), "{e}");
        assert!(recover(s.path(), "db", "t", 0, Target::Latest).is_err());
    }

    /// The other end of the same contract: a sealed segment that is shorter
    /// than the stream position its successor's name implies would stop a
    /// recovery early and report success.
    #[test]
    fn a_segment_shorter_than_the_chain_says_is_reported() {
        let _x = exclusive();
        let s = Scratch::new("wal-short");
        let d = table_wal(&s, "t");
        let mut w = Wal::open(&d).unwrap();
        w.append_insert(&rows(&[1, 2, 3])).unwrap();
        w.sync().unwrap();
        w.roll().unwrap();
        drop(w);
        let seg = segments(s.path(), "db", "t").unwrap().remove(0);
        let bytes = std::fs::read(&seg.path).unwrap();
        std::fs::write(&seg.path, &bytes[..bytes.len() - 4]).unwrap();
        let e = segments(s.path(), "db", "t").expect_err("a short segment must be reported");
        assert!(e.to_string().contains("report success"), "{e}");
    }

    /// The recovery LSN is what a backup's boundary is expressed in, so it can
    /// never be handed out twice -- including across a restart that finds an
    /// empty log because a checkpoint rolled it. The header's `prev_seq` is
    /// what makes that free; the old layout had to read the newest seal, and
    /// silently restarted at 1 on a database that had never archived.
    #[test]
    fn the_recovery_lsn_resumes_across_a_restart() {
        let _x = exclusive();
        let s = Scratch::new("wal-resume");
        let d = s.join("wal");
        let mut w = Wal::open(&d).unwrap();
        w.append_delete(1).unwrap();
        w.sync().unwrap();
        let used = w.span.last_seq;
        w.roll().unwrap();
        drop(w);

        NEXT_COMMIT.store(1, Ordering::Release);
        let mut w = Wal::open(&d).unwrap();
        assert!(commit_seq() > used, "a rolled log must resume from its header");
        w.append_delete(2).unwrap();
        w.sync().unwrap();
        assert!(w.span.last_seq > used, "and the next tick must be past it: {}", w.span.last_seq);
    }

    /// The delete run replaces one frame per lane. The assertion that used to
    /// pin "byte-identical to one append each" is inverted: the batch is now
    /// *one record* that replays to the same `WalRecord`s in the same order,
    /// with the same LSN reported for the first.
    #[test]
    fn a_batch_of_deletes_is_one_run_that_replays_the_same_records() {
        let s = Scratch::new("wal-delete-batch");
        let lanes: Vec<u64> = (0..64u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
        let want: Vec<WalRecord> = lanes.iter().map(|&l| WalRecord::Delete(l)).collect();
        for seq in [None, Some(7u64)] {
            let tag = if seq.is_some() { "staged" } else { "plain" };
            let one = s.join(&format!("one-{tag}"));
            let many = s.join(&format!("many-{tag}"));
            let mut a = Wal::open(&one).unwrap();
            let mut b = Wal::open(&many).unwrap();
            let mut first_a = None;
            for &l in &lanes {
                let at = match seq {
                    Some(q) => a.append_delete_staged(q, l).unwrap(),
                    None => a.append_delete(l).unwrap(),
                };
                first_a.get_or_insert(at);
            }
            let first_b = match seq {
                Some(q) => b.append_deletes_staged(q, &lanes).unwrap(),
                None => b.append_deletes(&lanes).unwrap(),
            };
            assert_eq!(first_a.unwrap(), first_b, "the run reports the first record's LSN");
            assert!(b.len() * 2 < a.len(), "a run must be far smaller: {} vs {}", b.len(), a.len());
            if let Some(q) = seq {
                a.commit(q).unwrap();
                b.commit(q).unwrap();
            }
            a.sync().unwrap();
            b.sync().unwrap();
            assert_eq!(Wal::replay(&one, &schema()).unwrap(), want, "{tag}: one per lane");
            assert_eq!(Wal::replay(&many, &schema()).unwrap(), want, "{tag}: one run");
        }
        // Empty is a no-op that still reports where the next record will land.
        let dir = s.join("empty");
        let mut w = Wal::open(&dir).unwrap();
        let at = w.lsn();
        assert_eq!(w.append_deletes(&[]).unwrap(), at);
        assert_eq!(w.len(), at);
    }

    /// A run whose declared count runs past its own frame must be refused
    /// rather than read into the next record.
    #[test]
    fn a_delete_run_that_overruns_its_frame_is_refused() {
        let s = Scratch::new("wal-run-overrun");
        let dir = s.join("wal");
        Wal::open(&dir).unwrap();
        append_raw(&dir, &[TAG_DELETE_RUN, 200, 1, 2, 3, 4, 5, 6, 7, 8]);
        let e = Wal::replay(&dir, &schema()).unwrap_err();
        assert!(e.to_string().contains("delete run"), "{e}");
    }

    /// 50 000 lanes, the shape a bulk `DELETE` on the default table shape
    /// produces. The win is log volume, so the byte count is asserted directly
    /// rather than inferred from a timing.
    #[test]
    fn a_bulk_delete_run_is_less_than_half_the_bytes() {
        let s = Scratch::new("wal-run-bytes");
        let dir = s.join("wal");
        let lanes: Vec<u64> = (0..50_000u64).collect();
        let mut w = Wal::open(&dir).unwrap();
        w.append_deletes(&lanes).unwrap();
        let n = w.len();
        assert_eq!(n, 400_097, "the run encoding's exact size");
        assert!(n * 2 < 50_000 * 19, "against {} bytes one frame per lane", 50_000 * 19);
    }
}
