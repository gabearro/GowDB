//! # granular
//!
//! A hybrid OLAP + OLTP storage engine with a ClickHouse-flavoured SQL front
//! end, built for maximum throughput per byte stored.
//!
//! ## Layering
//!
//! ```text
//!   sql/       lexer -> AST                (text in)
//!   planner/   bind -> logical -> optimize -> physical
//!   exec/      vectorized operators over Blocks
//!   storage/   tables -> parts -> granules -> packed columns
//!   persist/   on-disk part format, WAL, catalog
//!   encoding/  FOR bit packing, string dictionaries
//!   index/     minimal perfect hash, split-block bloom
//!   common/    hashing, bitsets, zigzag, errors
//! ```
//!
//! ## The one idea everything else follows from
//!
//! Data is stored **frame-of-reference bit-packed per column per granule**,
//! and every access path is designed to read it *without decompressing*:
//!
//!   * point lookups verify keys directly against packed words (one shifted
//!     `u128` load), so a `get` costs the same on compressed data as on raw;
//!   * range scans binary-search packed keys by interpolation;
//!   * predicate evaluation on strings runs on order-preserving dictionary
//!     codes, so `WHERE name > 'm'` never materializes a string;
//!   * zone maps fall out of the FOR metadata for free -- `base` and
//!     `base + mask` bound the granule with no extra bytes stored.
//!
//! ## Quick start
//!
//! ```no_run
//! use granular::Session;
//!
//! let mut db = Session::in_memory();
//! db.execute("CREATE TABLE hits (id UInt64, url String, ms UInt32) ENGINE = MergeTree ORDER BY id")?;
//! db.execute("INSERT INTO hits VALUES (1, '/', 12), (2, '/about', 40)")?;
//! let rs = db.query("SELECT url, sum(ms) FROM hits GROUP BY url ORDER BY url")?;
//! println!("{rs}");
//! # Ok::<(), granular::Error>(())
//! ```
//!
//! ## Concurrency
//!
//! A [`Session`] is one connection: it owns the catalog and its write API
//! takes `&mut self`. To read from several threads at once, move it behind a
//! [`Db`] and take a [`Reader`] per thread -- `Send + Sync + Clone`, with no
//! lifetime, so a pool can hold them.
//!
//! ```no_run
//! use granular::Session;
//!
//! let db = Session::in_memory().into_shared();
//! db.execute("CREATE TABLE t (id UInt64) ENGINE = MergeTree ORDER BY id")?;
//! let reader = db.reader();
//! std::thread::scope(|scope| {
//!     for _ in 0..4 {
//!         let r = reader.clone();
//!         scope.spawn(move || r.query("SELECT count() FROM t").unwrap());
//!     }
//! });
//! # Ok::<(), granular::Error>(())
//! ```
//!
//! Isolation is snapshot: a query pins one part set per table and a writer
//! publishes a new one, so a reader never sees half a commit. Writers
//! serialize with each other and with readers only for the length of one
//! statement. Large results stream through [`Session::read_stream`] or
//! [`Reader::cursor`] rather than materializing.

pub mod common;
pub mod encoding;
pub mod exec;
pub mod index;
pub mod io;
pub mod persist;
pub mod planner;
pub mod session;
pub mod settings;
pub mod catalog;
pub mod sort;
pub mod sql;
pub mod storage;
pub mod system;
pub mod types;

/// Backup, restore and verify. The source lives beside the rest of the on-disk
/// format, in `persist/`, because that is what it is a part of; it is declared
/// here only so that `persist/mod.rs` needs no edit. Move the declaration to
/// `pub mod backup;` there whenever that file is next touched -- the path
/// attribute is the only thing holding the module one level too high.
#[path = "persist/backup.rs"]
pub mod backup;

pub use common::{Error, Result};
pub use session::{Cursor, Db, Reader, ResultSet, Session, StreamItem, Writer};
pub use types::{Block, Column, DataType, Field, Schema, Value};
