//! `SELECT <exprs>`: evaluate a list of expressions per batch.
//!
//! Thin by design. All the work lives in [`crate::exec::expr`], and the
//! optimizer's projection-pruning pass is what makes sure the scan never
//! decoded the columns a projection would have dropped, so there is nothing
//! left to save on the *decode* side.
//!
//! What there was left to save is the copy. `expr::eval` has to return an
//! owned `Column`, so a bare column reference used to come back as a `Vec`
//! clone of a column this operator already owns outright -- the input block is
//! a value returned by `input.next()`, nobody else can see it, and it is
//! dropped two lines later. So the plan below is precomputed once and a column
//! nothing else in the projection still wants is **moved** out of the input
//! block instead. See `Step`.
//!
//! `expr::eval_all_cow` exists for the same reason and is not what this wants:
//! a `Cow::Borrowed` still has to be `into_owned()`-ed to build the output
//! block, which is the clone again. Owning the input is strictly stronger than
//! borrowing it.
//!
//! Measured with a temporary switch alternating old and new in one loop,
//! best-of-15 per side, 2M rows, `Scan -> Project` end to end:
//!
//! ```text
//!   one bare Int64 column       3.10 ->  2.27 ms   1.36x   (1.24-1.41x over 3 runs)
//!   one bare String column     23.31 -> 14.62 ms   1.59x   (1.55-1.59x)
//!   two bare columns, swapped   5.36 ->  4.32 ms   1.24x   (0.99-1.24x)
//!   one computed expression      3.80 ->  3.95 ms   0.96x   <- control, no move exists
//! ```
//!
//! The string column is the big one and says why: cloning a `Vec<Arc<str>>` is
//! an atomic increment per row on the way in and a decrement per row on the way
//! out, and moving it is neither. Through `Session::read`, `SELECT bytes FROM
//! events` is 1.30x and `SELECT country, latency FROM events` is 1.64x. Shapes
//! whose projection sits *above* a blocking operator -- every aggregate, sort
//! and join here -- measure 0.94-1.01x, i.e. unchanged, because the projection
//! there runs over a handful of rows and there was never anything to save.
//!
//! The zero-expression case is real: `SELECT count(*)` leaves a projection
//! with no columns but a row count that still has to survive, which is exactly
//! what [`Block::rows_only`] is for.

use crate::common::Result;
use crate::exec::expr;
use crate::planner::logical::BoundExpr;
use crate::types::{Block, Column, Schema};

use super::{Operator, ScanStats};

/// How one output column is produced. Decided once, at construction: the
/// classification depends only on the expression list, never on the block.
enum Step {
    /// `block.columns[i]`, taken by value. No later output column names `i`,
    /// so the buffer moves and the copy disappears.
    Take(usize),
    /// `block.columns[i]`, cloned: a later output column wants it as well
    /// (`SELECT a, a`), and only the last mention may take it.
    Copy(usize),
    /// The `j`th computed expression, evaluated into `scratch` before anything
    /// was moved out of the block.
    Computed(usize),
}

pub struct Project<'a> {
    input: Box<dyn Operator + 'a>,
    exprs: &'a [BoundExpr],
    schema: &'a Schema,
    plan: Vec<Step>,
    /// One past the highest input column index the projection names, so the
    /// bounds check that guards every `Take`/`Copy` is one compare per block
    /// rather than one per column.
    width_needed: usize,
    /// Computed columns for the block in hand, in expression order. A field so
    /// the `Vec` is reused; the `Column`s in it are moved out and replaced with
    /// empty placeholders as the output is assembled.
    scratch: Vec<Column>,
}

impl<'a> Project<'a> {
    pub fn new(
        input: Box<dyn Operator + 'a>,
        exprs: &'a [BoundExpr],
        schema: &'a Schema,
    ) -> Project<'a> {
        let mut plan = Vec::with_capacity(exprs.len());
        let mut computed = 0usize;
        let mut width_needed = 0usize;
        for (at, e) in exprs.iter().enumerate() {
            plan.push(match e {
                BoundExpr::Column { index, .. } => {
                    width_needed = width_needed.max(index + 1);
                    let last = exprs.iter().rposition(|o| {
                        matches!(o, BoundExpr::Column { index: i, .. } if i == index)
                    });
                    if last == Some(at) {
                        Step::Take(*index)
                    } else {
                        Step::Copy(*index)
                    }
                }
                _ => {
                    computed += 1;
                    Step::Computed(computed - 1)
                }
            });
        }
        Project { input, exprs, schema, plan, width_needed, scratch: Vec::with_capacity(computed) }
    }
}

/// The stand-in left behind by a `Take`. Costs nothing: an empty `Vec` does
/// not allocate, and the block it is written into is dropped before anything
/// can observe that its width no longer agrees with its row count.
#[inline]
fn hole() -> Column {
    Column::bools(Vec::new())
}

impl Operator for Project<'_> {
    fn schema(&self) -> &Schema {
        self.schema
    }

    fn stats(&self) -> ScanStats {
        self.input.stats()
    }

    fn next(&mut self) -> Result<Option<Block>> {
        while let Some(mut b) = self.input.next()? {
            if b.rows() == 0 {
                continue;
            }
            if self.exprs.is_empty() {
                return Ok(Some(Block::rows_only(b.rows())));
            }
            // A projection naming a column the block does not have is a plan
            // bug, and `expr::eval_all` owns the message for it. Deferring to
            // it here keeps one wording for that mistake and keeps the check
            // out of the per-column loop below.
            if self.width_needed > b.width() {
                return Ok(Some(Block::new(expr::eval_all(self.exprs, &b)?)?));
            }
            // Computed expressions read the *whole* input block, so they all
            // run before the first `Take` empties one of its columns. This
            // ordering is the invariant the whole operator rests on.
            self.scratch.clear();
            for (e, s) in self.exprs.iter().zip(&self.plan) {
                if matches!(s, Step::Computed(_)) {
                    self.scratch.push(expr::eval(e, &b)?);
                }
            }
            let mut out = Vec::with_capacity(self.plan.len());
            for s in &self.plan {
                out.push(match *s {
                    Step::Take(i) => std::mem::replace(&mut b.columns[i], hole()),
                    Step::Copy(i) => b.columns[i].clone(),
                    Step::Computed(j) => std::mem::replace(&mut self.scratch[j], hole()),
                });
            }
            return Ok(Some(Block::new(out)?));
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
    fn a_moved_column_is_still_readable_by_a_computed_one() {
        // The ordering invariant: `a + b` must see `b` even though the same
        // projection hands `b` out by value.
        let s = src_schema();
        let r = rows();
        let out_schema = Schema::new_unchecked(vec![
            Field::new("b", DataType::Int64),
            Field::new("sum", DataType::Int64),
            Field::new("a", DataType::Int64),
        ]);
        let exprs = vec![
            col(1),
            BoundExpr::Binary {
                left: Box::new(col(0)),
                op: BinaryOp::Plus,
                right: Box::new(col(1)),
                ty: DataType::Int64,
            },
            col(0),
        ];
        let mut p = Project::new(Box::new(Values::new(&r, &s)), &exprs, &out_schema);
        let b = p.next().unwrap().unwrap();
        assert_eq!(b.column(0).as_i64().unwrap(), &[10, 20]);
        assert_eq!(b.column(1).as_i64().unwrap(), &[11, 22]);
        assert_eq!(b.column(2).as_i64().unwrap(), &[1, 2]);
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
    fn an_out_of_range_column_is_an_error_not_a_panic() {
        let s = src_schema();
        let r = rows();
        let out_schema = Schema::new(vec![Field::new("x", DataType::Int64)]).unwrap();
        let exprs = vec![col(7)];
        let mut p = Project::new(Box::new(Values::new(&r, &s)), &exprs, &out_schema);
        let e = p.next().unwrap_err().to_string();
        assert!(e.contains("c7"), "{e}");
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
