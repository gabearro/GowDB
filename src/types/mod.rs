//! The type system and the data containers built on it.
//!
//! Two containers, used at different scales:
//!   * [`Value`] -- one scalar. Literals, group keys, result cells.
//!   * [`Block`] -- a vectorized batch of columns. Everything on the hot path.

pub mod block;
pub mod datatype;
pub mod schema;
pub mod value;

pub use block::{Block, Column, ColumnBuilder, ColumnData};
pub use datatype::{DataType, PhysicalType};
pub use schema::{Engine, Field, Schema, TableDef};
pub use value::{
    parse_civil_days,
    civil_from_days, days_from_civil, fmt_date, fmt_datetime, parse_date, parse_datetime, Value,
};
