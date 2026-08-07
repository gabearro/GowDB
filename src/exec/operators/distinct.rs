//! `SELECT DISTINCT`, and the dedup half of `UNION` (without `ALL`).
//!
//! Hash-based and **streaming**: rows come out in first-seen order as soon as
//! they arrive, rather than after a blocking sort. That matters because
//! `DISTINCT` sitting under a `LIMIT` is common, and a sort-based
//! implementation would have to consume the entire input before emitting the
//! first of ten rows.
//!
//! The cost is a live [`GroupKey`] per distinct row for the duration of the
//! query. That is the same footprint a hash aggregate pays, and it is the
//! right trade for the same reason: real `DISTINCT` cardinalities are small
//! relative to the row counts they are distilled from.

use crate::common::{FastSet, Result};
use crate::types::{Block, Schema};

use super::{row_key, GroupKey, Operator, ScanStats};

pub struct Distinct<'a> {
    input: Box<dyn Operator + 'a>,
    seen: FastSet<GroupKey>,
}

impl<'a> Distinct<'a> {
    pub fn new(input: Box<dyn Operator + 'a>) -> Distinct<'a> {
        Distinct { input, seen: FastSet::default() }
    }
}

impl Operator for Distinct<'_> {
    fn schema(&self) -> &Schema {
        self.input.schema()
    }

    fn stats(&self) -> ScanStats {
        self.input.stats()
    }

    fn next(&mut self) -> Result<Option<Block>> {
        while let Some(b) = self.input.next()? {
            if b.rows() == 0 {
                continue;
            }
            let mut sel = Vec::with_capacity(b.rows());
            for i in 0..b.rows() {
                if self.seen.insert(row_key(&b.columns, i)) {
                    sel.push(i as u32);
                }
            }
            if sel.is_empty() {
                continue;
            }
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
    use crate::types::{DataType, Field, Value};

    fn schema1() -> Schema {
        Schema::new(vec![Field::new("a", DataType::Int64)]).unwrap()
    }
    fn schema2() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::String),
        ])
        .unwrap()
    }

    fn drain1(op: &mut dyn Operator) -> Vec<Value> {
        let mut out = Vec::new();
        while let Some(b) = op.next().unwrap() {
            for i in 0..b.rows() {
                out.push(b.column(0).value(i));
            }
        }
        out
    }

    #[test]
    fn removes_duplicates_preserving_first_seen_order() {
        let s = schema1();
        let r: Vec<Vec<Value>> = [3i64, 1, 3, 2, 1, 3]
            .iter()
            .map(|&i| vec![Value::Int(i)])
            .collect();
        let mut d = Distinct::new(Box::new(Values::new(&r, &s)));
        assert_eq!(drain1(&mut d), vec![Value::Int(3), Value::Int(1), Value::Int(2)]);
    }

    #[test]
    fn distinguishes_on_the_whole_tuple() {
        let s = schema2();
        let r = vec![
            vec![Value::Int(1), Value::str("x")],
            vec![Value::Int(1), Value::str("y")],
            vec![Value::Int(1), Value::str("x")],
        ];
        let mut d = Distinct::new(Box::new(Values::new(&r, &s)));
        let mut n = 0;
        while let Some(b) = d.next().unwrap() {
            n += b.rows();
        }
        assert_eq!(n, 2);
    }

    #[test]
    fn null_is_a_value_and_dedups_against_itself() {
        let s = schema1();
        let r = vec![
            vec![Value::Null],
            vec![Value::Int(1)],
            vec![Value::Null],
        ];
        let mut d = Distinct::new(Box::new(Values::new(&r, &s)));
        assert_eq!(drain1(&mut d), vec![Value::Null, Value::Int(1)]);
    }

    /// Feeds a fixed list of already-built blocks, so a test can hand
    /// `Distinct` two batches whose column *types* differ -- exactly what
    /// `union::coerce` produces for a `UNION DISTINCT` of a `Date` branch and
    /// a `UInt64` branch (it leaves both alone because they are physically
    /// identical).
    struct Blocks {
        schema: Schema,
        blocks: Vec<Block>,
        pos: usize,
    }
    impl Operator for Blocks {
        fn schema(&self) -> &Schema {
            &self.schema
        }
        fn next(&mut self) -> Result<Option<Block>> {
            if self.pos >= self.blocks.len() {
                return Ok(None);
            }
            self.pos += 1;
            Ok(Some(self.blocks[self.pos - 1].clone()))
        }
    }

    #[test]
    fn probe_dedups_values_that_compare_equal_across_representations() {
        use crate::types::Column;
        let s = Schema::new(vec![Field::new("a", DataType::Date)]).unwrap();
        let dates = Block::new(vec![Column::u64s(DataType::Date, vec![5])]).unwrap();
        let uints = Block::new(vec![Column::u64s(DataType::UInt64, vec![5])]).unwrap();
        assert_eq!(
            dates.column(0).value(0),
            uints.column(0).value(0),
            "the two rows compare equal"
        );
        let src = Blocks { schema: s, blocks: vec![dates, uints], pos: 0 };
        let mut d = Distinct::new(Box::new(src));
        let mut n = 0;
        while let Some(b) = d.next().unwrap() {
            n += b.rows();
        }
        assert_eq!(n, 1, "DISTINCT kept two rows that compare equal");
    }

    #[test]
    fn dedups_across_batch_boundaries() {
        use crate::common::BLOCK_SIZE;
        let s = schema1();
        // Two full batches of the same three values.
        let r: Vec<Vec<Value>> = (0..BLOCK_SIZE as i64 * 2)
            .map(|i| vec![Value::Int(i % 3)])
            .collect();
        let mut d = Distinct::new(Box::new(Values::new(&r, &s)));
        assert_eq!(drain1(&mut d).len(), 3);
    }

    #[test]
    fn empty_input_is_fine() {
        let s = schema1();
        let r: Vec<Vec<Value>> = vec![];
        let mut d = Distinct::new(Box::new(Values::new(&r, &s)));
        assert!(d.next().unwrap().is_none());
    }
}
