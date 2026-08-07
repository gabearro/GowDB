//! `WHERE`, for predicates the scan could not absorb.
//!
//! The optimizer pushes everything it legally can into [`super::scan`]'s
//! PREWHERE, so what survives here is what could not be pushed: predicates
//! over a join output, over an aggregate result (`HAVING`), or over a
//! projection's computed columns.
//!
//! Empty results are swallowed rather than forwarded. A filter that rejects a
//! whole batch must not hand an empty block upstream, because `None` is the
//! only end-of-stream signal and a caller that stopped at the first empty
//! block would truncate the query.

use crate::common::Result;
use crate::exec::expr;
use crate::planner::logical::BoundExpr;
use crate::types::{Block, Schema};

use super::{Operator, ScanStats};

pub struct Filter<'a> {
    input: Box<dyn Operator + 'a>,
    predicate: &'a BoundExpr,
}

impl<'a> Filter<'a> {
    pub fn new(input: Box<dyn Operator + 'a>, predicate: &'a BoundExpr) -> Filter<'a> {
        Filter { input, predicate }
    }
}

impl Operator for Filter<'_> {
    fn schema(&self) -> &Schema {
        self.input.schema()
    }

    fn stats(&self) -> ScanStats {
        self.input.stats()
    }

    fn next(&mut self) -> Result<Option<Block>> {
        while let Some(b) = self.input.next()? {
            let sel = expr::eval_predicate(self.predicate, &b)?;
            if sel.is_empty() {
                continue;
            }
            // Nothing was rejected: hand the block through untouched rather
            // than paying for a full gather.
            if sel.len() == b.rows() {
                return Ok(Some(b));
            }
            return Ok(Some(b.take(&sel)));
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

    fn schema() -> Schema {
        Schema::new(vec![Field::new("a", DataType::Int64)]).unwrap()
    }

    fn rows(vs: &[Option<i64>]) -> Vec<Vec<Value>> {
        vs.iter()
            .map(|v| vec![v.map_or(Value::Null, Value::Int)])
            .collect()
    }

    fn gt(n: i64) -> BoundExpr {
        BoundExpr::Binary {
            left: Box::new(BoundExpr::Column {
                index: 0,
                ty: DataType::Int64,
                name: "a".into(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(BoundExpr::lit(Value::Int(n))),
            ty: DataType::Bool,
        }
    }

    fn run(data: &[Option<i64>], pred: &BoundExpr) -> Vec<i64> {
        let s = schema();
        let src = rows(data);
        let mut f = Filter::new(Box::new(Values::new(&src, &s)), pred);
        let mut out = Vec::new();
        while let Some(b) = f.next().unwrap() {
            for i in 0..b.rows() {
                out.push(b.column(0).as_i64().unwrap()[i]);
            }
        }
        out
    }

    #[test]
    fn keeps_only_matching_rows() {
        let d = [Some(1), Some(5), Some(9)];
        assert_eq!(run(&d, &gt(4)), vec![5, 9]);
    }

    #[test]
    fn null_fails_the_filter() {
        let d = [Some(1), None, Some(9)];
        assert_eq!(run(&d, &gt(0)), vec![1, 9], "NULL > 0 is NULL, not true");
    }

    #[test]
    fn rejecting_everything_ends_the_stream_cleanly() {
        let d = [Some(1), Some(2)];
        assert!(run(&d, &gt(100)).is_empty());
    }

    #[test]
    fn schema_is_the_inputs() {
        let s = schema();
        let src = rows(&[Some(1)]);
        let p = gt(0);
        let f = Filter::new(Box::new(Values::new(&src, &s)), &p);
        assert_eq!(f.schema().len(), 1);
        assert_eq!(f.schema().name(0), "a");
    }
}
