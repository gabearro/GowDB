//! Literal row sources: `VALUES`, and the zero-row `Empty`.
//!
//! `VALUES` is the one place in the engine where data arrives as
//! [`Value`]s rather than columns, so this operator is a transposer: it walks
//! the row list building one [`ColumnBuilder`] per output column, coercing each
//! literal to the schema's declared type on the way in. Batching at
//! [`BLOCK_SIZE`] matters because `INSERT ... VALUES` of a large literal
//! payload flows through here.
//!
//! `Empty` exists so the optimizer can replace a provably-empty subtree
//! (`WHERE 1 = 0`, an empty `IN` list) with something that still carries a
//! schema. Downstream operators need the column types even when there are no
//! rows -- a join against an empty side still has to emit NULL-padded columns
//! of the right shape.

use crate::common::{Error, Result, BLOCK_SIZE};
use crate::types::{Block, ColumnBuilder, Schema, Value};

use super::Operator;

pub struct Values<'a> {
    rows: &'a [Vec<Value>],
    schema: &'a Schema,
    pos: usize,
}

impl<'a> Values<'a> {
    pub fn new(rows: &'a [Vec<Value>], schema: &'a Schema) -> Values<'a> {
        Values { rows, schema, pos: 0 }
    }
}

impl Operator for Values<'_> {
    fn schema(&self) -> &Schema {
        self.schema
    }

    fn next(&mut self) -> Result<Option<Block>> {
        if self.pos >= self.rows.len() {
            return Ok(None);
        }
        let end = (self.pos + BLOCK_SIZE).min(self.rows.len());
        let n = end - self.pos;
        let width = self.schema.len();
        let mut builders: Vec<ColumnBuilder> = (0..width)
            .map(|c| ColumnBuilder::with_capacity(self.schema.ty(c).clone(), n))
            .collect();
        for row in &self.rows[self.pos..end] {
            if row.len() != width {
                return Err(Error::exec(format!(
                    "VALUES row has {} columns, the schema declares {width}",
                    row.len()
                )));
            }
            for (b, v) in builders.iter_mut().zip(row) {
                b.push_value(v)?;
            }
        }
        self.pos = end;
        if width == 0 {
            return Ok(Some(Block::rows_only(n)));
        }
        Ok(Some(Block::new(builders.into_iter().map(|b| b.finish()).collect())?))
    }
}

/// Zero rows, known schema.
pub struct Empty<'a> {
    schema: &'a Schema,
}

impl<'a> Empty<'a> {
    pub fn new(schema: &'a Schema) -> Empty<'a> {
        Empty { schema }
    }
}

impl Operator for Empty<'_> {
    fn schema(&self) -> &Schema {
        self.schema
    }
    fn next(&mut self) -> Result<Option<Block>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataType, Field};

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Int64),
            Field::new("s", DataType::String),
        ])
        .unwrap()
    }

    #[test]
    fn transposes_rows_into_columns() {
        let s = schema();
        let rows = vec![
            vec![Value::Int(1), Value::str("x")],
            vec![Value::Int(2), Value::str("y")],
        ];
        let mut v = Values::new(&rows, &s);
        let b = v.next().unwrap().unwrap();
        assert_eq!(b.rows(), 2);
        assert_eq!(b.column(0).as_i64().unwrap(), &[1, 2]);
        assert_eq!(b.column(1).value(1), Value::str("y"));
        assert!(v.next().unwrap().is_none());
    }

    #[test]
    fn coerces_literals_to_the_declared_type() {
        let s = Schema::new(vec![Field::new("f", DataType::Float64)]).unwrap();
        let rows = vec![vec![Value::Int(3)]];
        let b = Values::new(&rows, &s).next().unwrap().unwrap();
        assert_eq!(b.column(0).as_f64().unwrap(), &[3.0]);
    }

    #[test]
    fn nulls_survive() {
        let s = Schema::new(vec![Field::new("a", DataType::Int64)]).unwrap();
        let rows = vec![vec![Value::Int(1)], vec![Value::Null]];
        let b = Values::new(&rows, &s).next().unwrap().unwrap();
        assert!(!b.column(0).is_null(0));
        assert!(b.column(0).is_null(1));
    }

    #[test]
    fn batches_at_block_size() {
        let s = Schema::new(vec![Field::new("a", DataType::Int64)]).unwrap();
        let rows: Vec<Vec<Value>> = (0..BLOCK_SIZE as i64 + 3)
            .map(|i| vec![Value::Int(i)])
            .collect();
        let mut v = Values::new(&rows, &s);
        assert_eq!(v.next().unwrap().unwrap().rows(), BLOCK_SIZE);
        assert_eq!(v.next().unwrap().unwrap().rows(), 3);
        assert!(v.next().unwrap().is_none());
    }

    #[test]
    fn ragged_rows_are_rejected() {
        let s = schema();
        let rows = vec![vec![Value::Int(1)]];
        assert!(Values::new(&rows, &s).next().is_err());
    }

    #[test]
    fn no_rows_at_all() {
        let s = schema();
        let rows: Vec<Vec<Value>> = vec![];
        assert!(Values::new(&rows, &s).next().unwrap().is_none());
    }

    #[test]
    fn empty_operator_yields_nothing_but_keeps_its_schema() {
        let s = schema();
        let mut e = Empty::new(&s);
        assert_eq!(e.schema().len(), 2);
        assert!(e.next().unwrap().is_none());
    }
}
