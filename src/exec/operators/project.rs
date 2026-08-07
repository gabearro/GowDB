//! `SELECT <exprs>`: evaluate a list of expressions per batch.
//!
//! Thin by design. All the work lives in [`crate::exec::expr`], and a
//! projection that is only a column reordering costs one `Vec` clone per
//! column -- the optimizer's projection-pruning pass is what makes sure the
//! scan never decoded the columns a projection would have dropped, so there is
//! nothing left to save here.
//!
//! The zero-expression case is real: `SELECT count(*)` leaves a projection
//! with no columns but a row count that still has to survive, which is exactly
//! what [`Block::rows_only`] is for.

use crate::common::Result;
use crate::exec::expr;
use crate::planner::logical::BoundExpr;
use crate::types::{Block, Schema};

use super::{Operator, ScanStats};

pub struct Project<'a> {
    input: Box<dyn Operator + 'a>,
    exprs: &'a [BoundExpr],
    schema: &'a Schema,
}

impl<'a> Project<'a> {
    pub fn new(
        input: Box<dyn Operator + 'a>,
        exprs: &'a [BoundExpr],
        schema: &'a Schema,
    ) -> Project<'a> {
        Project { input, exprs, schema }
    }
}

impl Operator for Project<'_> {
    fn schema(&self) -> &Schema {
        self.schema
    }

    fn stats(&self) -> ScanStats {
        self.input.stats()
    }

    fn next(&mut self) -> Result<Option<Block>> {
        while let Some(b) = self.input.next()? {
            if b.rows() == 0 {
                continue;
            }
            if self.exprs.is_empty() {
                return Ok(Some(Block::rows_only(b.rows())));
            }
            let cols = expr::eval_all(self.exprs, &b)?;
            return Ok(Some(Block::new(cols)?));
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::Values;
    use crate::sql::ast::BinaryOp;
    use crate::types::{DataType, Field, Value};

    fn src_schema() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Int64),
        ])
        .unwrap()
    }

    fn rows() -> Vec<Vec<Value>> {
        vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(2), Value::Int(20)],
        ]
    }

    fn col(i: usize) -> BoundExpr {
        BoundExpr::Column { index: i, ty: DataType::Int64, name: format!("c{i}") }
    }

    #[test]
    fn evaluates_expressions_over_each_batch() {
        let s = src_schema();
        let r = rows();
        let out_schema = Schema::new(vec![Field::new("sum", DataType::Int64)]).unwrap();
        let exprs = vec![BoundExpr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Plus,
            right: Box::new(col(1)),
            ty: DataType::Int64,
        }];
        let mut p = Project::new(Box::new(Values::new(&r, &s)), &exprs, &out_schema);
        let b = p.next().unwrap().unwrap();
        assert_eq!(b.column(0).as_i64().unwrap(), &[11, 22]);
        assert!(p.next().unwrap().is_none());
    }

    #[test]
    fn reorders_and_duplicates_columns() {
        let s = src_schema();
        let r = rows();
        let out_schema = Schema::new_unchecked(vec![
            Field::new("b", DataType::Int64),
            Field::new("a", DataType::Int64),
            Field::new("b2", DataType::Int64),
        ]);
        let exprs = vec![col(1), col(0), col(1)];
        let mut p = Project::new(Box::new(Values::new(&r, &s)), &exprs, &out_schema);
        let b = p.next().unwrap().unwrap();
        assert_eq!(b.width(), 3);
        assert_eq!(b.column(0).as_i64().unwrap(), &[10, 20]);
        assert_eq!(b.column(1).as_i64().unwrap(), &[1, 2]);
        assert_eq!(b.column(2).as_i64().unwrap(), &[10, 20]);
    }

    #[test]
    fn empty_projection_keeps_the_row_count() {
        let s = src_schema();
        let r = rows();
        let out_schema = Schema::empty();
        let exprs: Vec<BoundExpr> = vec![];
        let mut p = Project::new(Box::new(Values::new(&r, &s)), &exprs, &out_schema);
        let b = p.next().unwrap().unwrap();
        assert_eq!(b.rows(), 2);
        assert_eq!(b.width(), 0);
    }

    #[test]
    fn projection_schema_is_its_own_not_the_inputs() {
        let s = src_schema();
        let r = rows();
        let out_schema = Schema::new(vec![Field::new("only", DataType::Int64)]).unwrap();
        let exprs = vec![col(0)];
        let p = Project::new(Box::new(Values::new(&r, &s)), &exprs, &out_schema);
        assert_eq!(p.schema().name(0), "only");
    }
}
