//! `UNION ALL` (concatenation) and `UNION` (concatenation plus dedup).
//!
//! `UNION DISTINCT` is not a separate implementation: it is
//! [`super::distinct::Distinct`] stacked on top of the concatenation, which is
//! exactly what it means. Composing operators instead of special-casing keeps
//! the dedup logic in one place and means `UNION` automatically inherits
//! `DISTINCT`'s streaming behaviour.
//!
//! Branches may disagree about column types where the plan's schema says
//! `Int64` but one branch produced `UInt64` (a literal `VALUES` leg, say). The
//! result set renders each block through its own column types, so a mismatch
//! would show up as inconsistent formatting between rows. Blocks are therefore
//! coerced to the union schema's physical shape on the way out -- and only
//! when they actually differ, so the common case costs one pointer comparison
//! per column.

use crate::catalog::Catalog;
use crate::common::Result;
use crate::planner::logical::LogicalPlan;
use crate::types::{Block, Column, ColumnBuilder, Schema};

use super::{build, distinct::Distinct, Operator, QueryContext, ScanStats};

pub struct Union<'a> {
    inputs: Vec<Box<dyn Operator + 'a>>,
    schema: &'a Schema,
    cur: usize,
}

/// Build a `UNION` pipeline, wrapping in a `Distinct` unless `ALL` was given.
pub fn build_union<'a>(
    inputs: &'a [LogicalPlan],
    all: bool,
    schema: &'a Schema,
    catalog: &'a Catalog,
    ctx: &'a QueryContext,
) -> Result<Box<dyn Operator + 'a>> {
    // The context has to reach the branches: a `UNION` of two aggregates is
    // exactly the shape whose memory a budget needs to see.
    let ops: Vec<Box<dyn Operator + 'a>> = inputs
        .iter()
        .map(|p| build(p, catalog, ctx))
        .collect::<Result<_>>()?;
    let u = Union { inputs: ops, schema, cur: 0 };
    Ok(if all { Box::new(u) } else { Box::new(Distinct::new(Box::new(u))) })
}

impl Operator for Union<'_> {
    fn schema(&self) -> &Schema {
        self.schema
    }

    fn stats(&self) -> ScanStats {
        let mut s = ScanStats::default();
        for i in &self.inputs {
            s.merge(&i.stats());
        }
        s
    }

    fn next(&mut self) -> Result<Option<Block>> {
        while self.cur < self.inputs.len() {
            match self.inputs[self.cur].next()? {
                Some(b) if b.rows() > 0 => return Ok(Some(coerce(b, self.schema)?)),
                Some(_) => continue,
                None => self.cur += 1,
            }
        }
        Ok(None)
    }
}

/// Retype a branch's columns to the union schema when the physical
/// representation differs. A no-op in the usual case.
fn coerce(b: Block, schema: &Schema) -> Result<Block> {
    if b.width() != schema.len() {
        return Ok(b);
    }
    let needs = b
        .columns
        .iter()
        .enumerate()
        .any(|(i, c)| c.ty.physical() != schema.ty(i).physical());
    if !needs {
        return Ok(b);
    }
    let rows = b.rows();
    let mut out: Vec<Column> = Vec::with_capacity(b.width());
    for (i, c) in b.columns.iter().enumerate() {
        let want = schema.ty(i);
        if c.ty.physical() == want.physical() {
            out.push(c.clone());
            continue;
        }
        let ty = if c.has_nulls() { want.to_nullable() } else { want.clone() };
        let mut nb = ColumnBuilder::with_capacity(ty, rows);
        for r in 0..rows {
            if c.is_null(r) {
                nb.push_null();
            } else {
                nb.push_value(&c.value(r))?;
            }
        }
        out.push(nb.finish());
    }
    Block::new(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataType, Field, Value};

    fn schema() -> Schema {
        Schema::new(vec![Field::new("a", DataType::Int64)]).unwrap()
    }

    fn values_plan(vs: &[i64], s: &Schema) -> LogicalPlan {
        LogicalPlan::Values {
            rows: vs.iter().map(|&i| vec![Value::Int(i)]).collect(),
            schema: s.clone(),
        }
    }

    fn run(plans: Vec<LogicalPlan>, all: bool, s: &Schema) -> Vec<i64> {
        let cat = Catalog::in_memory();
        let ctx = QueryContext::new();
        let mut op = build_union(&plans, all, s, &cat, &ctx).unwrap();
        let mut out = Vec::new();
        while let Some(b) = op.next().unwrap() {
            out.extend_from_slice(b.column(0).as_i64().unwrap());
        }
        out
    }

    #[test]
    fn union_all_concatenates_in_order() {
        let s = schema();
        let plans = vec![values_plan(&[1, 2], &s), values_plan(&[2, 3], &s)];
        assert_eq!(run(plans, true, &s), vec![1, 2, 2, 3]);
    }

    #[test]
    fn union_without_all_deduplicates() {
        let s = schema();
        let plans = vec![values_plan(&[1, 2], &s), values_plan(&[2, 3], &s)];
        assert_eq!(run(plans, false, &s), vec![1, 2, 3]);
    }

    #[test]
    fn a_single_branch_is_passed_through() {
        let s = schema();
        assert_eq!(run(vec![values_plan(&[5, 5], &s)], true, &s), vec![5, 5]);
    }

    #[test]
    fn empty_branches_are_skipped_not_treated_as_end_of_stream() {
        let s = schema();
        let plans = vec![
            values_plan(&[], &s),
            values_plan(&[7], &s),
            values_plan(&[], &s),
            values_plan(&[8], &s),
        ];
        assert_eq!(run(plans, true, &s), vec![7, 8]);
    }

    #[test]
    fn branches_are_coerced_to_the_union_schema() {
        // Branch 2 produces UInt64 where the union schema says Int64.
        let s = schema();
        let plans = vec![
            values_plan(&[1], &s),
            LogicalPlan::Values {
                rows: vec![vec![Value::UInt(2)]],
                schema: Schema::new(vec![Field::new("a", DataType::UInt64)]).unwrap(),
            },
        ];
        let cat = Catalog::in_memory();
        let ctx = QueryContext::new();
        let mut op = build_union(&plans, true, &s, &cat, &ctx).unwrap();
        let mut tys = Vec::new();
        while let Some(b) = op.next().unwrap() {
            tys.push(b.column(0).ty.clone());
        }
        assert_eq!(tys, vec![DataType::Int64, DataType::Int64]);
    }

    #[test]
    fn no_branches_at_all() {
        let s = schema();
        assert!(run(vec![], true, &s).is_empty());
    }
}
