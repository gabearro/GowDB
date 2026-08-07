//! Durable, crash-safe on-disk storage.
//!
//! ```text
//!   <root>/
//!     CATALOG                     databases -> table definitions
//!     <db>/<table>/
//!       TABLE                     this table's definition + the parts that
//!                                 are currently live + the WAL prefix they
//!                                 already cover
//!       part_000001.gpart         one immutable part per file
//!       part_000002.gpart
//!       wal.log                   framed log of writes not yet in a part
//! ```
//!
//! ## Three rules the whole module is built around
//!
//! **1. Nothing is ever modified in place.** Every file is written to a temp
//! name in its final directory, `fsync`ed, then `rename`d over the target, and
//! the *directory* is `fsync`ed after. POSIX `rename` is atomic, so a reader
//! (or a crash) sees either the whole old file or the whole new one, never a
//! prefix. Skipping either fsync silently gives up the guarantee: without the
//! first, the rename can be durable while the bytes it points at are not;
//! without the second, the rename itself can be lost. See [`store::atomic_write`].
//!
//! **2. There is exactly one commit point per table.** Parts are written under
//! fresh, never-reused sequence numbers, so publishing them cannot damage what
//! is already on disk; the `TABLE` file, written last and atomically, is what
//! makes them live. A crash before that leaves orphan part files that no
//! reader can see and the next write collects. This is why `read_table` never
//! scans the directory for parts -- the directory listing is not the truth,
//! the committed `TABLE` file is.
//!
//! **3. Everything read from disk is a hostile input.** Every file is
//! checksummed by [`format::write_framed`], every count is validated against
//! the bytes actually present, and every field that feeds an `unsafe`
//! `get_unchecked` downstream (packed word arrays, MPH slot counts) is proved
//! large enough *before* the structure is built. Corruption produces
//! `Error::Corruption`; it never panics and it never yields a
//! structurally-valid-but-wrong `Part`.
//!
//! ## What a part file stores, and what it deliberately does not
//!
//! A part file holds everything needed to reconstruct a [`Part`](crate::storage::Part)
//! *without recomputing anything super-linear*: the packed words, the string
//! dictionaries, the null and delete bitmaps, and -- the expensive one -- the
//! per-granule CHD minimal perfect hash and its fused fingerprint/rank
//! records. Rebuilding an MPH costs a displacement search per granule; loading
//! one costs a `memcpy`. Persisting it is the entire reason this format is not
//! just "write the rows back out".
//!
//! What it does *not* store is anything cheap and derivable: the sparse index,
//! the O(1) granule router and the part-level bloom filter are all rebuilt by
//! [`Part::from_parts`](crate::storage::Part::from_parts) or on first use,
//! because they are linear passes over data we are already touching.
//!
//! ## Checkpoint protocol
//!
//! [`save_catalog`] flushes every write buffer into parts, commits each table,
//! then truncates that table's WAL and records the new (empty) log position.
//! The `TABLE` file carries the byte offset of the log prefix its parts
//! already cover, so recovery replays *only* the suffix. That offset is what
//! keeps the two-step "commit parts, then discard the log" safe in both crash
//! windows: crashing before the truncation replays a covered prefix (which the
//! offset skips), and crashing after it leaves a log shorter than the recorded
//! offset (which [`load_catalog`] detects and repairs).

pub mod format;
pub mod mmap;
pub mod reader;
pub mod store;
pub mod wal;
pub mod writer;

pub use reader::{part_from_bytes, part_from_mmap, read_part, read_table};
pub use store::{load_catalog, save_catalog};
pub use wal::{Wal, WalRecord};
pub use writer::{write_part, write_table};

#[cfg(test)]
pub(crate) mod testkit {
    //! Shared fixtures. Temp directories are named from the process id plus a
    //! counter rather than from randomness, so a rerun cannot collide with a
    //! live process and a name is reproducible while debugging.

    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::common::{splitmix64, G_SHIFT};
    use crate::storage::{Part, Table};
    use crate::types::{
        Block, Column, ColumnBuilder, DataType, Engine, Field, Schema, TableDef, Value,
    };

    static SEQ: AtomicU64 = AtomicU64::new(0);

    /// A temp directory that deletes itself, including on panic.
    pub struct Scratch(PathBuf);

    impl Scratch {
        pub fn new(tag: &str) -> Scratch {
            let n = SEQ.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir()
                .join(format!("granular-persist-{}-{tag}-{n}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).expect("create scratch dir");
            Scratch(p)
        }
        pub fn path(&self) -> &Path {
            &self.0
        }
        pub fn join(&self, s: &str) -> PathBuf {
            self.0.join(s)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    pub fn schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::UInt64),
            Field::new("host", DataType::String),
            Field::new("ms", DataType::Nullable(Box::new(DataType::Int64))),
            Field::new("ratio", DataType::Float64),
        ])
        .unwrap()
    }

    pub fn table_def(name: &str) -> TableDef {
        TableDef {
            name: name.into(),
            schema: schema(),
            order_by: vec![0],
            primary_key: vec![0],
            partition_by: None,
            engine: Engine::MergeTree,
        }
    }

    /// `n` rows sorted by a unique pseudo-random key, with a low-cardinality
    /// string column, a nullable column that is null every 7th row, and a
    /// float column. Exercises all four physical kinds plus a null mask.
    pub fn sample_block(n: usize) -> Block {
        let mut keys: Vec<u64> = (0..n as u64 * 3)
            .map(|i| splitmix64(i) % 4_000_000)
            .collect();
        keys.sort_unstable();
        keys.dedup();
        assert!(keys.len() >= n, "not enough distinct keys for {n} rows");
        keys.truncate(n);
        let hosts: Vec<std::sync::Arc<str>> = keys
            .iter()
            .map(|k| format!("host-{:02}", k % 37).into())
            .collect();
        let mut ms = ColumnBuilder::new(DataType::Nullable(Box::new(DataType::Int64)));
        for (i, &k) in keys.iter().enumerate() {
            if i % 7 == 0 {
                ms.push_null();
            } else {
                ms.push_value(&Value::Int((k % 10_000) as i64 - 5_000)).unwrap();
            }
        }
        let ratio: Vec<f64> = keys.iter().map(|&k| (k % 1000) as f64 / 8.0 - 60.0).collect();
        Block::new(vec![
            Column::u64s(DataType::UInt64, keys),
            Column::strs(DataType::String, hosts),
            ms.finish(),
            Column::f64s(DataType::Float64, ratio),
        ])
        .unwrap()
    }

    /// A part with `n` rows and every 13th row of every granule deleted.
    pub fn sample_part(n: usize) -> Part {
        let b = sample_block(n);
        let mut p = Part::build(&b, Some(0), Some(0)).unwrap();
        for gi in 0..p.granule_count() {
            let len = p.granules[gi].len;
            for i in (0..len).step_by(13) {
                p.mark_deleted((gi << G_SHIFT) + i);
            }
        }
        p
    }

    /// A table holding one part per entry in `chunks`, with disjoint key
    /// ranges so nothing tombstones anything else.
    pub fn sample_table(name: &str, chunks: &[usize]) -> Table {
        let mut t = Table::new(table_def(name), 1 << 20);
        let mut lo = 0u64;
        for &n in chunks {
            let b = sample_block(n);
            let keys: Vec<u64> = b.column(0).as_u64().unwrap().iter().map(|&k| k + lo).collect();
            lo += 10_000_000;
            let mut cols = b.columns.clone();
            cols[0] = Column::u64s(DataType::UInt64, keys);
            t.insert(Block::new(cols).unwrap()).unwrap();
            t.flush().unwrap();
        }
        t
    }

    /// Every `(position, row)` of a part, for exact comparison across a
    /// round trip.
    pub fn dump(p: &Part) -> Vec<(usize, Vec<Value>)> {
        let mut out = Vec::new();
        for (gi, g) in p.granules.iter().enumerate() {
            let base = gi << G_SHIFT;
            for i in 0..g.len {
                let pos = base + i;
                out.push((pos, (0..p.ncols).map(|c| p.value_at(pos, c)).collect()));
            }
        }
        out
    }

    pub fn deleted_positions(p: &Part) -> Vec<usize> {
        let mut v = Vec::new();
        for (gi, g) in p.granules.iter().enumerate() {
            let base = gi << G_SHIFT;
            for i in 0..g.len {
                if p.deleted.get(base + i) {
                    v.push(base + i);
                }
            }
        }
        v
    }
}
