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

pub mod common;
pub mod encoding;
pub mod exec;
pub mod index;
pub mod persist;
pub mod planner;
pub mod session;
pub mod catalog;
pub mod sort;
pub mod sql;
pub mod storage;
pub mod types;

pub use common::{Error, Result};
pub use session::{ResultSet, Session};
pub use types::{Block, Column, DataType, Field, Schema, Value};
