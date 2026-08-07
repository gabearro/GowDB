//! The storage engine: tables -> parts -> granules -> packed columns.
//!
//! ```text
//!   Table          schema + write buffer + a list of parts
//!    |- Delta      hash-map write buffer (OLTP)
//!    `- Part       immutable, sorted, bloom-filtered run
//!        `- Granule   1024 rows, independently packed and indexed
//!            `- PackedColumn   FOR bit-packed lanes (+ dictionary, + nulls)
//! ```

pub mod column;
pub mod delta;
pub mod granule;
pub mod part;
pub mod table;

pub use column::PackedColumn;
pub use delta::{Delta, DeltaEntry};
pub use granule::{Granule, PkIndex, Stats};
pub use part::Part;
pub use table::{
    sort_permutation, ColumnCompression, CompressionReport, RowLoc, Table, AUTO_COMPACT_PARTS,
    BULK_INSERT_THRESHOLD,
};
