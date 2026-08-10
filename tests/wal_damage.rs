//! Damage to a log segment, swept exhaustively, end to end through `Session`.
//!
//! The rule the segmented format exists to enforce has exactly two legal
//! outcomes and one forbidden one:
//!
//!   * **report** -- the open or the query fails, naming the file. Damage
//!     behind a record whose `fsync` returned is lost acknowledged data and it
//!     is an error, never a shorter answer;
//!   * **swallow** -- the damage is above what the segment says was
//!     acknowledged, so it is an interrupted append that was never
//!     acknowledged to anybody, and replay stops at it silently;
//!   * **and never** -- a short row count with `Ok`, which is the defect the
//!     format version was bumped to remove. A caller cannot tell that outcome
//!     from a database that was simply never written to, and the next exit
//!     checkpoint folds the survivors into parts and destroys the evidence.
//!
//! `src/persist/wal.rs` sweeps the same four damage quadrants against the `Wal`
//! type directly. This file is the other end of the same claim: through
//! `Session::open` and a real `SELECT`, which is where a user meets it, and
//! over a **sealed** segment as well as the active one. The distinction
//! matters and is not cosmetic -- a sealed segment is one that stopped being
//! the newest file in its directory, its `durable` covers its whole length by
//! construction, and it therefore has no forgiveness window at all. Every
//! byte of it is acknowledged, so every failure inside it must be reported.
//! The active segment has a window exactly one group commit wide, and the
//! sweep checks that the window is that and no wider.
//!
//! ## Cost
//!
//! Each damage image is a fresh process-equivalent: `Session::open` runs
//! recovery, which is one `read_dir`, a scan of the active segment and, on the
//! healthy path, two `fsync`s. So the sweeps are shaped to keep the fixture
//! small (six rows, ~300 bytes of log) and still cover every byte: 300 bytes x
//! (8 flips + 3 fills) is ~3300 opens per sweep. `GRANULAR_WAL_DAMAGE=<n>`
//! multiplies the fixture size for a longer run.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use granular::persist::wal::SEG_HEADER_LEN;
use granular::{Session, Value};

// ------------------------------------------------------------------ fixture

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "granular-wd-{}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed),
            tag
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Scratch(d)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const DDL: &str = "CREATE TABLE t (id UInt64, s String) ENGINE = MergeTree ORDER BY id";

fn rows() -> u64 {
    std::env::var("GRANULAR_WAL_DAMAGE").ok().and_then(|v| v.parse().ok()).unwrap_or(1) * 6
}

fn wal_dir(root: &Path) -> PathBuf {
    root.join(".wal").join("default").join("t")
}

/// Every segment of the table's log, oldest first.
fn segs(root: &Path) -> Vec<PathBuf> {
    let d = wal_dir(root);
    let mut names: Vec<String> = std::fs::read_dir(&d)
        .expect("log directory")
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".gwal"))
        .collect();
    names.sort();
    names.into_iter().map(|n| d.join(n)).collect()
}

/// A directory whose table's records live in the **active** segment: written
/// and never checkpointed, so nothing has been folded into parts and the log
/// is the only copy. `Session` has no `Drop` checkpoint, which is what makes
/// this reachable at all.
fn active_fixture(s: &Scratch, n: u64) -> PathBuf {
    let mut db = Session::open(s.path()).unwrap();
    db.execute(DDL).unwrap();
    for i in 0..n {
        db.execute(&format!("INSERT INTO t VALUES ({i},'row-{i}')")).unwrap();
    }
    drop(db);
    let v = segs(s.path());
    assert_eq!(v.len(), 1, "the fixture should hold exactly one segment, got {}", v.len());
    v.into_iter().next().unwrap()
}

/// A directory with a **sealed** segment that is still load-bearing: the rows
/// live inside it, nothing has been folded into parts, and replay has to cross
/// it to answer at all.
///
/// A sealed segment whose records are already in parts would make the sweep
/// vacuous -- the parts would answer correctly however badly the segment were
/// damaged, and every silently-dropped record would be masked. The state that
/// makes it load-bearing is not contrived: it is the documented crash window
/// between `Wal::roll` and `write_table`, where the successor has been
/// published and the watermark still names the sealed segment's origin. So the
/// fixture reaches it the same way a crash does -- roll the log and stop --
/// rather than by rearranging files the engine wrote.
fn sealed_fixture(s: &Scratch, n: u64) -> (PathBuf, PathBuf) {
    let mut db = Session::open(s.path()).unwrap();
    db.execute(DDL).unwrap();
    for i in 0..n {
        db.execute(&format!("INSERT INTO t VALUES ({i},'row-{i}')")).unwrap();
    }
    drop(db);
    granular::persist::Wal::open(&wal_dir(s.path())).unwrap().roll().unwrap();
    let v = segs(s.path());
    assert_eq!(v.len(), 2, "the roll should have sealed one segment and started one");
    let mut it = v.into_iter();
    let sealed = it.next().unwrap();
    (sealed, it.next().unwrap())
}

// ------------------------------------------------------------------ oracles

/// What a reopen makes of the directory: the ids the table answers with, or
/// `None` for a refusal (at the open or at the query -- either is a report).
fn reopen(root: &Path) -> Option<Vec<u64>> {
    let mut db = Session::open(root).ok()?;
    let rs = db.query("SELECT id FROM t ORDER BY id").ok()?;
    Some(
        rs.to_values()
            .iter()
            .map(|r| match r[0] {
                Value::UInt(n) => n,
                ref o => panic!("id came back as {o}"),
            })
            .collect(),
    )
}

/// Counters for one sweep, so a passing run reports how much of each outcome
/// it actually produced rather than just "ok".
#[derive(Default)]
struct Tally {
    images: u64,
    reported: u64,
    swallowed: u64,
    intact: u64,
}

impl Tally {
    fn record(&mut self, ids: Option<&[u64]>, n: u64) {
        self.images += 1;
        match ids {
            None => self.reported += 1,
            Some(v) if v.len() as u64 == n => self.intact += 1,
            Some(_) => self.swallowed += 1,
        }
    }
    fn say(&self, what: &str) {
        eprintln!(
            "  {what}: {} images -> {} reported, {} swallowed as a tear, {} intact",
            self.images, self.reported, self.swallowed, self.intact
        );
    }
}

/// Apply `f` to a copy of `full`, write it over `seg`, reopen, and check the
/// answer against the contract. `floor` is the number of leading rows that
/// must never go missing quietly.
fn check(
    root: &Path,
    seg: &Path,
    full: &[u8],
    n: u64,
    floor: u64,
    what: &str,
    at: usize,
    tally: &mut Tally,
    f: impl FnOnce(&mut Vec<u8>),
) {
    let mut b = full.to_vec();
    f(&mut b);
    if b == full {
        return;
    }
    std::fs::write(seg, &b).unwrap();
    let ids = reopen(root);
    if let Some(v) = &ids {
        for (i, &id) in v.iter().enumerate() {
            assert_eq!(
                id, i as u64,
                "{what} at byte {at} of {}: the answer is not a prefix ({v:?})",
                seg.display()
            );
        }
        assert!(
            v.len() as u64 >= floor,
            "{what} at byte {at} of {}: replayed {} of {n} rows and reported SUCCESS. \
             {floor} of them sit below what the segment says its `fsync` returned for, \
             so this is silently dropped acknowledged data",
            seg.display(),
            v.len()
        );
        assert!(
            v.len() as u64 <= n,
            "{what} at byte {at}: {} rows came back but only {n} were ever written",
            v.len()
        );
    }
    tally.record(ids.as_deref(), n);
}

// -------------------------------------------------------------------- sweeps

/// Every byte of a **sealed** segment, in all four damage quadrants.
///
/// A sealed segment has no forgiveness window: it stopped being the newest
/// file in the directory, which can only happen after the `fsync` that
/// published its successor, so every byte of it is acknowledged. `floor` is
/// therefore the whole row count -- any answer at all must be the complete
/// one, and anything less has to be a refusal.
#[test]
fn every_byte_of_a_sealed_segment_is_reported_or_intact() {
    let n = rows();
    let s = Scratch::new("sealed");
    let (sealed, _active) = sealed_fixture(&s, n);
    let full = std::fs::read(&sealed).unwrap();
    assert_eq!(
        reopen(s.path()).as_deref(),
        Some(&(0..n).collect::<Vec<_>>()[..]),
        "the undamaged fixture must replay every row out of the sealed segment"
    );

    let mut t = Tally::default();
    for at in 0..full.len() {
        for bit in 0..8 {
            check(s.path(), &sealed, &full, n, n, "a flipped bit", at, &mut t, |b| {
                b[at] ^= 1 << bit
            });
        }
        check(s.path(), &sealed, &full, n, n, "a zero fill", at, &mut t, |b| b[at..].fill(0));
        check(s.path(), &sealed, &full, n, n, "a garbage fill", at, &mut t, |b| b[at..].fill(0x5A));
        check(s.path(), &sealed, &full, n, n, "a truncation", at, &mut t, |b| b.truncate(at));
    }
    assert!(t.reported > 0, "no damage to a sealed segment was reported -- the sweep is vacuous");
    assert_eq!(
        t.swallowed, 0,
        "{} images of a SEALED segment replayed short and reported success. A sealed \
         segment's `durable` covers its whole length, so it has no torn tail and nothing \
         inside it may be dismissed as an interrupted append",
        t.swallowed
    );
    t.say("sealed segment");
}

/// Every byte of the **active** segment, in all four quadrants.
///
/// The active segment does have a forgiveness window, and the claim is that it
/// is exactly one group commit wide. The fixture syncs after every insert, so
/// the second-to-last `fsync` covers rows `0..n-1`: those are provably on the
/// platter and may never go missing without an error. The last row may, since
/// a crash between its `fsync` starting and returning is a real crash shape.
#[test]
fn every_byte_of_the_active_segment_stays_inside_one_group_commit() {
    let n = rows();
    let s = Scratch::new("active");
    let seg = active_fixture(&s, n);
    let full = std::fs::read(&seg).unwrap();
    assert_eq!(
        reopen(s.path()).as_deref(),
        Some(&(0..n).collect::<Vec<_>>()[..]),
        "the undamaged fixture must replay every row"
    );

    let mut t = Tally::default();
    for at in 0..full.len() {
        // Below the header there is no forgiveness at all: the header is
        // published whole by a write-fsync-rename, so no crash can leave one
        // short and any damage to it is damage to a segment that already held
        // every row. Above it, the window is one group commit -- the last row.
        let floor = if at < SEG_HEADER_LEN as usize { n } else { n - 1 };
        if at >= SEG_HEADER_LEN as usize {
            for bit in 0..8 {
                check(s.path(), &seg, &full, n, floor, "a flipped bit", at, &mut t, |b| {
                    b[at] ^= 1 << bit
                });
            }
        }
        check(s.path(), &seg, &full, n, floor, "a zero fill", at, &mut t, |b| b[at..].fill(0));
        check(s.path(), &seg, &full, n, floor, "a garbage fill", at, &mut t, |b| {
            b[at..].fill(0x5A)
        });
        check(s.path(), &seg, &full, n, floor, "a truncation", at, &mut t, |b| b.truncate(at));
    }
    assert!(t.reported > 0, "no damage to the active segment was reported");
    t.say("active segment");
}

/// A segment cut below its own header, which is the shape the sweep above
/// found and which used to lose **every** acknowledged record in silence.
///
/// `Wal::open` rewrote a short segment from scratch and `replay_entries`
/// skipped one, both justified as "a crash between `creat` and the first
/// write". No such crash exists: a segment is published by
/// `store::atomic_write`, which writes the whole 64-byte header into a temp
/// file, `fsync`s it, and only then renames it into place, so at every instant
/// a crash can observe, a segment that exists is at least a full header. The
/// two shortcuts were therefore never repairing a partial creation -- they were
/// discarding a damaged segment's entire contents and answering `Ok`.
/// Reproduced at cuts of 0, 8, 30 and 63 bytes: `count()` came back 0 of 6,
/// exit 0, nothing quarantined.
#[test]
fn a_segment_cut_below_its_header_is_reported_and_not_rewritten() {
    let n = rows();
    for cut in [0usize, 1, 8, 30, 63] {
        let s = Scratch::new("short");
        let seg = active_fixture(&s, n);
        let full = std::fs::read(&seg).unwrap();
        std::fs::write(&seg, &full[..cut]).unwrap();
        assert_eq!(
            reopen(s.path()),
            None,
            "a {cut}-byte log segment was accepted: the {n} acknowledged rows it held were \
             dropped and the answer reported success"
        );
        // …and the refusal has to survive a second open, rather than the first
        // one repairing the file into an empty log that the second then
        // happily believes.
        assert_eq!(reopen(s.path()), None, "the {cut}-byte segment was repaired away on reopen");
        assert_eq!(
            std::fs::metadata(&seg).map(|m| m.len()).unwrap_or(u64::MAX),
            cut as u64,
            "the refused open rewrote the damaged segment"
        );
    }
}

/// The segment header, which is the nearest thing this format has to a footer:
/// a fixed 64-byte record with its own check word, holding the two durability
/// points and the chain fields.
///
/// Damage anywhere in it must be reported or leave the answer whole -- never
/// turned into a shorter one. Note that a failed check is deliberately *not*
/// fatal: the fixed fields are still the ones we wrote or the magic and
/// version would have failed first, and the honest response to an
/// untrustworthy stamp is to distrust it, which makes every failure in the
/// file report. Refusing outright would turn one flipped bit into an
/// unopenable table whose records are all still there. So a large majority of
/// header flips leave the answer intact, and that is correct; what matters is
/// that none of them make it shorter.
#[test]
fn damage_to_a_segment_header_is_reported_or_leaves_the_answer_whole() {
    let n = rows();
    let s = Scratch::new("header");
    let seg = active_fixture(&s, n);
    let full = std::fs::read(&seg).unwrap();

    let mut t = Tally::default();
    for at in 0..SEG_HEADER_LEN as usize {
        for bit in 0..8 {
            check(s.path(), &seg, &full, n, n, "a flipped header bit", at, &mut t, |b| {
                b[at] ^= 1 << bit
            });
        }
    }
    assert!(t.reported > 0, "damage to the segment header was never reported");
    assert_eq!(t.swallowed, 0, "a damaged segment header produced a short answer with Ok");
    t.say("segment header");
}

/// The stamp is load-bearing and its check is what makes it so.
///
/// A torn tail -- a partial record with no `fsync` behind it -- is swallowed,
/// because the stamp says those bytes were never acknowledged. Flip one bit of
/// the check word that covers the stamp and the *same bytes* must be reported
/// instead: an untrustworthy stamp forgives nothing. That is the difference
/// between a recorded fact and a positional guess, asserted directly.
///
/// The same mechanism is what stops a header being transplanted: the check
/// covers the segment's **origin**, which lives only in the file name, so a
/// header lifted from a sibling segment fails it and the recipient stops
/// forgiving anything.
#[test]
fn an_untrustworthy_stamp_forgives_nothing() {
    let n = rows();
    let s = Scratch::new("stamp");
    let seg = active_fixture(&s, n);
    // One more record with no `sync` behind it, cut in half: a genuine
    // interrupted append, above the stamp, which must be swallowed.
    {
        let mut db = Session::open(s.path()).unwrap();
        db.execute(&format!("INSERT INTO t VALUES ({n},'torn')")).unwrap();
        drop(db);
    }
    let full = std::fs::read(&seg).unwrap();
    let torn = &full[..full.len() - 4];
    std::fs::write(&seg, torn).unwrap();
    let ids = reopen(s.path());
    assert!(
        matches!(&ids, Some(v) if v.len() as u64 >= n),
        "a genuine torn tail must replay the acknowledged prefix, got {ids:?}"
    );

    // Now the same tail under a header whose check does not verify.
    let mut broken = torn.to_vec();
    broken[SEG_HEADER_LEN as usize - 1] ^= 0x80;
    std::fs::write(&seg, &broken).unwrap();
    assert_eq!(
        reopen(s.path()),
        None,
        "the same torn tail was still swallowed under a stamp whose check fails -- an \
         unverifiable stamp must forgive nothing, or the fix is a positional guess again"
    );

    // A header lifted from a sibling segment fails the same check, because it
    // covers the origin and the origin is the file name.
    let s2 = Scratch::new("transplant");
    let (sealed, active) = sealed_fixture(&s2, n);
    assert!(reopen(s2.path()).is_some(), "the two-segment fixture must open before it is broken");
    let donor = std::fs::read(&active).unwrap();
    let mut victim = std::fs::read(&sealed).unwrap();
    let head = SEG_HEADER_LEN as usize;
    assert_ne!(&donor[..head], &victim[..head], "the two headers are already identical");
    victim[..head].copy_from_slice(&donor[..head]);
    // Damage a record in the middle, in the region the donor's much lower
    // `durable` would have dismissed as never acknowledged.
    let mid = head + (victim.len() - head) / 2;
    victim[mid] ^= 0xFF;
    std::fs::write(&sealed, &victim).unwrap();
    assert_eq!(
        reopen(s2.path()),
        None,
        "a segment wearing another segment's header dismissed damage as an interrupted \
         append -- the check must bind to the origin, which lives only in the file name"
    );
}

/// The length varint specifically, in a **sealed** segment, at every frame and
/// every bit.
///
/// This is the shipped defect in the one place the old classifier could not
/// even in principle get right. The rule that replaced it -- a sealed segment
/// has no tail -- has to hold at the first byte of every frame, including the
/// last one, which is where a positional heuristic is at its most convincing.
#[test]
fn a_length_varint_in_a_sealed_segment_is_never_a_tear() {
    let n = rows();
    let s = Scratch::new("sealed-len");
    let (sealed, _) = sealed_fixture(&s, n);
    let full = std::fs::read(&sealed).unwrap();
    let starts = frame_starts(&full);
    assert!(starts.len() > 4, "the sealed fixture holds only {} frames", starts.len());

    let mut t = Tally::default();
    for &at in &starts {
        for bit in 0..8 {
            check(s.path(), &sealed, &full, n, n, "a flipped length bit", at, &mut t, |b| {
                b[at] ^= 1 << bit
            });
        }
    }
    assert!(t.reported > 0, "no forged length in a sealed segment was reported");
    assert_eq!(t.swallowed, 0, "a forged length in a sealed segment silently dropped records");
    t.say("sealed length varints");
}

/// The same at the **last** frame of the active segment, which is the one
/// place a forged length is allowed to be forgiven -- and only there, and only
/// up to the stamp.
///
/// The fixture syncs after every insert, so every frame but the trailing tick
/// sits below `durable`. A forged length at any of them must be reported; the
/// contract for the tail is checked by the exhaustive sweep above.
#[test]
fn a_length_varint_below_the_stamp_is_never_a_tear() {
    let n = rows();
    let s = Scratch::new("active-len");
    let seg = active_fixture(&s, n);
    let full = std::fs::read(&seg).unwrap();
    let starts = frame_starts(&full);
    assert!(starts.len() > 4, "the active fixture holds only {} frames", starts.len());

    let mut t = Tally::default();
    // Every frame but the last: the last one is the trailing tick, which the
    // final `fsync` covers but which carries no rows, so damaging it costs
    // nothing a row count can see.
    for &at in &starts[..starts.len() - 1] {
        for bit in 0..8 {
            check(s.path(), &seg, &full, n, n, "a flipped length bit", at, &mut t, |b| {
                b[at] ^= 1 << bit
            });
        }
    }
    assert!(t.reported > 0, "no forged length below the stamp was reported");
    assert_eq!(
        t.swallowed, 0,
        "a forged length below the durability stamp was treated as a torn tail and \
         silently dropped the records behind it -- this is the original defect"
    );
    t.say("active length varints");
}

/// Damage inside an archived segment, read by a **point-in-time recovery**.
///
/// This is the instance nothing else covers, and it is the one that costs the
/// most when it is wrong: `RESTORE ... UNTIL LATEST` unpacks a backup and rolls
/// the archive forward over it, and the archive is exactly the sealed segments
/// this file damages. A recovery that stops early does not merely answer short
/// -- it writes a database, in a new directory, that the operator will keep and
/// the original of which they may then delete.
///
/// Every restore here goes through the shipped binary, because a recovery that
/// is only correct in-process is not a recovery. The bit is flipped in a length
/// varint in the middle of an archived segment, which leaves the **file length
/// unchanged**: that is what makes it the hard case, since a length cross-check
/// alone would pass it.
#[test]
fn damage_in_an_archived_segment_refuses_the_recovery_it_would_shorten() {
    let s = Scratch::new("pitr");
    let db = s.path().join("db");
    let arc = s.path().join("base.gbak");
    let bin = env!("CARGO_BIN_EXE_granular");

    let run = |dir: &Path, sql: &str| -> (bool, String) {
        let out = std::process::Command::new(bin)
            .args(["--data", dir.to_str().unwrap(), "-q", sql])
            .args(["--format", "tsv", "--no-header"])
            .output()
            .expect("spawn granular");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned()
                + &String::from_utf8_lossy(&out.stderr),
        )
    };

    assert!(run(&db, DDL).0);
    assert!(run(&db, "INSERT INTO t VALUES (0,'a'),(1,'b')").0);
    assert!(run(&db, &format!("BACKUP TO '{}'", arc.display())).0);
    // Six more rows, each its own invocation so each lands behind its own tick
    // and the archive has interior boundaries to be damaged between.
    for i in 2..8u64 {
        assert!(run(&db, &format!("INSERT INTO t VALUES ({i},'r{i}')")).0);
    }
    let want: Vec<String> = (0..8u64).map(|i| i.to_string()).collect();

    // The archived segments: everything but the newest.
    let mut arch = segs(&db);
    arch.pop().expect("an active segment");
    assert!(!arch.is_empty(), "the fixture archived nothing -- there is nothing to damage");
    assert!(arch.len() >= 4, "the fixture archived only {} segments", arch.len());

    // The healthy recovery first, or the rest of this proves nothing.
    let good = s.path().join("out-good");
    let (ok, out) = run(&db, &format!("RESTORE FROM '{}' TO '{}' UNTIL LATEST", arc.display(), good.display()));
    assert!(ok, "the undamaged recovery failed: {out}");
    let (ok, got) = run(&good, "SELECT id FROM t ORDER BY id");
    assert!(ok, "the recovered database would not open: {got}");
    assert_eq!(
        got.split_ascii_whitespace().collect::<Vec<_>>(),
        want,
        "the undamaged recovery did not restore every row"
    );

    // Every archived segment, every frame start, every bit -- the length field
    // is where the old classifier could not be right even in principle, and a
    // recovery walks all of these segments, not just the last.
    let (mut images, mut refused) = (0u32, 0u32);
    for (v, victim) in arch.iter().enumerate() {
        let full = std::fs::read(victim).unwrap();
        let starts = frame_starts(&full);
        assert!(starts.len() >= 2, "archived segment {v} holds only {} frames", starts.len());
        for (k, &at) in starts.iter().enumerate() {
            for bit in 0..8 {
                let mut b = full.clone();
                b[at] ^= 1 << bit;
                if b == full {
                    continue;
                }
                assert_eq!(b.len(), full.len(), "the image changed length");
                std::fs::write(victim, &b).unwrap();
                let out_dir = s.path().join(format!("out-{v}-{k}-{bit}"));
                let (ok, _) = run(
                    &db,
                    &format!(
                        "RESTORE FROM '{}' TO '{}' UNTIL LATEST",
                        arc.display(),
                        out_dir.display()
                    ),
                );
                images += 1;
                if ok {
                    // It said the recovery succeeded, so it has to have
                    // recovered everything: this damage is inside an archived
                    // segment, whose whole extent was acknowledged before its
                    // successor existed.
                    let (ok, got) = run(&out_dir, "SELECT id FROM t ORDER BY id");
                    let got: Vec<&str> = got.split_ascii_whitespace().collect();
                    assert!(
                        ok && got == want,
                        "a flipped bit {bit} in the length field at byte {at} of archived \
                         segment {v} produced a recovery that reported SUCCESS and restored \
                         {got:?} instead of {want:?}. The operator now holds a database that \
                         is missing committed rows and has been told it is complete"
                    );
                } else {
                    refused += 1;
                }
                std::fs::write(victim, &full).unwrap();
            }
        }
    }
    assert!(refused > 0, "no damaged archive was ever refused -- the sweep is vacuous");
    eprintln!(
        "  archived-segment recovery: {} segments, {images} images -> {refused} refused",
        arch.len()
    );
}

/// Frame starts in a segment, by walking `varint len | u64 sum | body` by hand.
/// Deliberately not using the reader: a test that finds the frames with the
/// same code under test would agree with it about where they are not.
fn frame_starts(full: &[u8]) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut p = SEG_HEADER_LEN as usize;
    while p < full.len() {
        let at = p;
        let (mut len, mut shift) = (0u64, 0u32);
        loop {
            if p >= full.len() || shift > 63 {
                return starts;
            }
            let b = full[p];
            p += 1;
            len |= u64::from(b & 0x7f) << shift;
            if b < 0x80 {
                break;
            }
            shift += 7;
        }
        match p.checked_add(8 + len as usize) {
            Some(next) if next <= full.len() => p = next,
            _ => return starts,
        }
        starts.push(at);
    }
    starts
}
