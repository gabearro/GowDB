//! Function registries.
//!
//! Two flavours, both looked up by lowercase name:
//!   * [`ScalarFn`] -- row-wise, vectorized: takes `Column`s, returns a
//!     `Column` of the same length.
//!   * [`AggFn`] -- builds an [`Accumulator`] that folds many rows into one
//!     `Value`.
//!
//! Both carry a `ret` callback so the binder can type-check a call without
//! executing it.

pub mod agg;
pub mod scalar;

use crate::common::Result;
use crate::types::{Column, DataType, Value};
use std::any::Any;

/// A vectorized scalar function.
pub struct ScalarFn {
    pub name: &'static str,
    /// `(min, max)` argument count; `max == usize::MAX` means variadic.
    pub arity: (usize, usize),
    /// Return type given argument types. Also the arity/type validator.
    pub ret: fn(&[DataType]) -> Result<DataType>,
    /// Evaluate over a batch. Every argument column has exactly `rows` rows
    /// (constants have already been expanded by the caller).
    pub eval: fn(args: &[Column], rows: usize) -> Result<Column>,
}

impl ScalarFn {
    pub fn check_arity(&self, n: usize) -> Result<()> {
        let (lo, hi) = self.arity;
        if n < lo || n > hi {
            let want = if hi == usize::MAX {
                format!("at least {lo}")
            } else if lo == hi {
                format!("exactly {lo}")
            } else {
                format!("{lo} to {hi}")
            };
            return Err(crate::common::Error::bind(format!(
                "function {} takes {want} arguments, got {n}",
                self.name
            )));
        }
        Ok(())
    }
}

/// Folds rows into a single value. One instance per group.
pub trait Accumulator: Any + Send {
    /// Fold the rows named by `sel` (indices into `args`' columns).
    fn update(&mut self, args: &[Column], sel: &[u32]) -> Result<()>;
    /// Combine a partial aggregate computed elsewhere (parallel scan merge).
    fn merge(&mut self, other: &dyn Accumulator) -> Result<()>;
    /// Final value. Must be callable more than once.
    ///
    /// Fallible because the fold is deliberately wider than the declared return
    /// type -- `sum` totals in `i128`, `avg` divides at a *promoted* decimal
    /// scale -- so the narrowing at the end can genuinely not fit. The
    /// engine-wide policy for that is **error; never saturate, never wrap**.
    ///
    /// Saturating is the worst of the three and is what this used to do, for
    /// the sole reason that `finish` returned `Value` and had no way to say no.
    /// A saturated total is a *number*: it compares equal to itself, it sorts,
    /// it renders, and nothing downstream can tell it from the true one.
    /// Concretely, over a `Decimal64(2)` column holding 10^12, `avg` answered
    /// 999999999999.999999 while `max` over the same column answered
    /// 1000000000000.00; and `sum(x)/count(*)`, which goes through
    /// `scalar::dec_divide` and *does* raise, contradicted `avg(x)` on the same
    /// rows. Raising is what makes those agree.
    fn finish(&self) -> Result<Value>;
    fn as_any(&self) -> &dyn Any;
    /// A fresh accumulator of the same kind and configuration.
    fn boxed_clone(&self) -> Box<dyn Accumulator>;
}

/// An aggregate function definition.
pub struct AggFn {
    pub name: &'static str,
    pub arity: (usize, usize),
    /// Return type given argument types and parametric arguments
    /// (`quantile(0.9)(x)` passes `[0.9]` as params).
    pub ret: fn(&[DataType], &[Value]) -> Result<DataType>,
    pub new: fn(&[DataType], &[Value]) -> Result<Box<dyn Accumulator>>,
    /// True when `DISTINCT` is meaningful (`count`, `sum`, `avg`, `uniq`).
    pub supports_distinct: bool,
}

impl AggFn {
    pub fn check_arity(&self, n: usize) -> Result<()> {
        let (lo, hi) = self.arity;
        if n < lo || n > hi {
            return Err(crate::common::Error::bind(format!(
                "aggregate {} takes {lo}..{hi} arguments, got {n}",
                self.name
            )));
        }
        Ok(())
    }
}

/// Look up a scalar function by name (case-insensitive).
pub fn scalar(name: &str) -> Option<&'static ScalarFn> {
    scalar::lookup(&name.to_ascii_lowercase())
}

/// Look up an aggregate by name (case-insensitive). Also resolves the
/// ClickHouse `-If` combinator suffix, e.g. `sumIf`, `countIf`, `avgIf`.
pub fn aggregate(name: &str) -> Option<&'static AggFn> {
    agg::lookup(&name.to_ascii_lowercase())
}

pub fn is_aggregate(name: &str) -> bool {
    aggregate(name).is_some()
}
