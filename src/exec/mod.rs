//! Vectorized execution.

pub mod expr;
pub mod functions;
pub mod operators;

// `exchange` is an operator and its file lives with the operators. It is
// declared here rather than as `pub mod exchange;` in `operators/mod.rs`
// because that file is owned by another change in flight and this one must not
// touch it. Nothing inside the module depends on where it is mounted -- it
// names everything through `crate::exec::operators` rather than `super` -- so
// moving the declaration one file over is a two-line change with no other
// consequence, and it should be made.
#[path = "operators/exchange.rs"]
pub mod exchange;

/// Run a plan with the parallel exchange in it. See [`exchange`].
///
/// A drop-in replacement for [`operators::execute_ctx`]: same signature, same
/// answers, same `ScanStats`. It is a separate entry point rather than a flag
/// because `operators::build_physical` drops a `PhysicalPlan::Exchange` on the
/// floor -- it has no operator that can honour one -- so this is the only way
/// to reach a parallel pipeline.
///
/// `execute_parallel` is the one to call: it takes the [`QueryContext`] that
/// carries the query's memory budget, deadline and cancel flag.
/// `execute_parallel_stats` is the convenience for a caller that has none yet
/// and makes a fresh default one per query.
///
/// [`QueryContext`]: operators::QueryContext
pub use exchange::{execute_ctx as execute_parallel, execute_with_stats as execute_parallel_stats};
