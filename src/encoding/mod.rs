//! Column codecs. Everything here turns a raw column into packed words and
//! back, and every codec keeps O(1) random access so point lookups never pay
//! a decompression cost.

pub mod bitpack;
pub mod dict;
pub mod lz4;

pub use bitpack::{packed_lower_bound, PackedU64};
pub use dict::StringDict;
