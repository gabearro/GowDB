//! Query planning: AST -> bound logical plan -> optimized logical plan ->
//! physical plan.
//!
//! The first three stages answer "what rows does this query want"; the last
//! answers "how will the engine get them". They are separate types rather than
//! flags on one tree because the decisions differ in kind: a logical rewrite
//! must preserve the answer for *any* storage, while a physical choice --
//! index versus scan, top-K versus full sort -- is only valid against the
//! storage that is actually there. See [`physical`].

pub mod binder;
pub mod logical;
pub mod optimizer;
pub mod physical;

pub use logical::{BoundAgg, BoundExpr, CmpOp, LogicalPlan, ScanNode, SortKey, ZoneFilter};
pub use physical::{IndexPath, PhysicalPlan};

/// The physical plan for an optimized logical plan, rendered as an indented
/// tree.
///
/// The entry point `EXPLAIN PIPELINE` should call: it is the only rendering
/// that reflects the operators that will actually run, because the access-path
/// decision is not visible anywhere in the logical plan.
pub fn explain_physical(
    plan: &LogicalPlan,
    catalog: &crate::catalog::Catalog,
) -> crate::common::Result<String> {
    Ok(physical::lower(plan, catalog)?.explain())
}
