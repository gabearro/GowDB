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
/// because the decision belongs to the caller that knows whether this query is
/// worth a fleet -- and because, until `build_physical` itself consults
/// `exchange::try_build`, this is the only way to reach a parallel pipeline.
pub use exchange::{execute_ctx as execute_parallel, execute_with_stats as execute_parallel_stats};
