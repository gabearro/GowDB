//! Primitives with no dependencies on anything else in the crate.

pub mod bitset;
pub mod error;
pub mod hash;
pub mod lane;
pub mod pool;
pub mod zigzag;

pub use bitset::BitSet;
pub use error::{Error, Result};
pub use lane::{f64_to_lane, i64_to_lane, lane_to_f64, lane_to_i64};
pub use hash::{
    fastrange, fp6, hash_bytes, hash_key, mum, prefetch_read, splitmix64, FastBuild, FastHasher,
    FastMap, FastSet, FP_BITS, FP_SEED,
};
pub use zigzag::{zz_dec, zz_enc};

/// Rows per granule. Must stay a power of two: the row-position encoding is
/// `granule_index << G_SHIFT | offset`, and the packed-lane math assumes it.
pub const GRANULE_SIZE: usize = 1024;
pub const G_SHIFT: u32 = GRANULE_SIZE.trailing_zeros();

/// Rows per vectorized execution block. Sized so a handful of `u64` columns
/// stay resident in L2 across an operator pipeline.
pub const BLOCK_SIZE: usize = 8192;
