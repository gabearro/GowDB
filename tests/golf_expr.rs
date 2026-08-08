//! Every per-block specialization in [`granular::exec::expr`] and its scalar
//! registry, checked against the general path it replaced.
//!
//! The evaluator now decides things once per block that it used to decide once
//! per row: which lane kind a column is, whether it has a null mask at all,
//! whether a comparison's literal is NaN, which side of `i64::MAX` an unsigned
//! literal falls. Every one of those is a chance to have picked a *different*
//! answer in a corner -- a NaN, a `-0.0`, a lane at `i64::MAX`, an empty string,
//! a NULL in the middle of an `IN` list -- and unit tests spot-check corners one
//! at a time.
//!
//! So this file does not assert answers. It generates NULL-bearing,
//! boundary-valued columns, runs each expression twice -- once with the
//! specializations on and once with them off -- and demands the two `Column`s
//! be equal, lane for lane and null bit for null bit. The general path is the
//! oracle; anything the fast path changed shows up as a diff no matter which
//! corner it hides in.
//!
//! `Column`'s `PartialEq` compares the declared type as well as the data, so a
//! fast path that produced the right numbers under a `Nullable(Bool)` where the
//! general path said `Bool` fails here too.

use granular::exec::expr;
use granular::planner::logical::BoundExpr as B;
use granular::sql::ast::{BinaryOp, UnaryOp};
use granular::types::{Block, Column, ColumnBuilder, ColumnData, DataType, Value};
use std::sync::Arc;

// --------------------------------------------------------------- generation

fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Boundary values first, then pseudo-random ones. Both matter: the boundaries
/// are where a widening or a saturating cast changes its mind, and the random
/// tail is what makes a sorted-probe or a binary search actually get exercised.
const I64S: &[i64] = &[
    0,
    1,
    -1,
    2,
    -2,
    7,
    -7,
    127,
    128,
    255,
    256,
    -128,
    -129,
    32767,
    -32768,
    2_147_483_647,
    -2_147_483_648,
    4_294_967_295,
    9_007_199_254_740_992, // 2^53, where f64 stops being exact on integers
    9_007_199_254_740_993,
    i64::MAX,
    i64::MIN,
    i64::MAX - 1,
    i64::MIN + 1,
];

const U64S: &[u64] = &[
    0,
    1,
    2,
    255,
    256,
    65535,
    4_294_967_295,
    9_007_199_254_740_992,
    9_007_199_254_740_993,
    i64::MAX as u64 - 1,
    i64::MAX as u64,
    i64::MAX as u64 + 1, // the lane that makes `as_i64` give up
    u64::MAX - 1,
    u64::MAX,
];

const F64S: &[f64] = &[
    0.0,
    -0.0, // must compare Equal to 0.0, and must not sort apart from it
    1.0,
    -1.0,
    0.5,
    -0.5,
    1e300,
    -1e300,
    f64::MIN_POSITIVE,
    -f64::MIN_POSITIVE,
    f64::EPSILON,
    9_007_199_254_740_993.0,
    9.223_372_036_854_776e18, // 2^63 exactly, the i64::MAX rounding boundary
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    -f64::NAN,
];

const STRS: &[&str] = &["", "a", "A", "ab", "aa", "b", "US", "us", "DE", "zzz", "\u{e9}", "a\0b"];

/// `n` rows of `vals`, cycled, with every `nth` row NULL. `nth == 0` means no
/// mask at all -- which is a *different* column from one whose mask exists but
/// happens to be empty, and the specializations branch on exactly that.
fn build_col(ty: DataType, n: usize, nth: usize, mut at: impl FnMut(usize) -> Value) -> Column {
    let want_nulls = nth != 0;
    let mut b = ColumnBuilder::new(if want_nulls { ty.to_nullable() } else { ty });
    for i in 0..n {
        if want_nulls && i % nth == 0 {
            b.push_null();
        } else {
            b.push_value(&at(i)).unwrap();
        }
    }
    b.finish()
}

fn ints(n: usize, nth: usize, seed: u64) -> Column {
    build_col(DataType::Int64, n, nth, |i| {
        Value::Int(if i < I64S.len() {
            I64S[i]
        } else {
            I64S[(splitmix(i as u64 + seed) % I64S.len() as u64) as usize]
        })
    })
}

fn uints(n: usize, nth: usize, seed: u64) -> Column {
    build_col(DataType::UInt64, n, nth, |i| {
        Value::UInt(if i < U64S.len() {
            U64S[i]
        } else {
            U64S[(splitmix(i as u64 + seed) % U64S.len() as u64) as usize]
        })
    })
}

fn floats(n: usize, nth: usize, seed: u64) -> Column {
    build_col(DataType::Float64, n, nth, |i| {
        Value::Float(if i < F64S.len() {
            F64S[i]
        } else {
            F64S[(splitmix(i as u64 + seed) % F64S.len() as u64) as usize]
        })
    })
}

fn strs(n: usize, nth: usize, seed: u64) -> Column {
    build_col(DataType::String, n, nth, |i| {
        Value::str(if i < STRS.len() {
            STRS[i]
        } else {
            STRS[(splitmix(i as u64 + seed) % STRS.len() as u64) as usize]
        })
    })
}

/// A `Decimal64(s)` column from raw lanes, so the scale-mismatch fallbacks get
/// exercised rather than only the equal-scale fast path.
fn decs(s: u8, n: usize, nth: usize, seed: u64) -> Column {
    build_col(DataType::Decimal64(s), n, nth, |i| {
        Value::Decimal(
            if i < I64S.len() {
                I64S[i]
            } else {
                I64S[(splitmix(i as u64 + seed) % I64S.len() as u64) as usize]
            },
            s,
        )
    })
}

fn bools(n: usize, nth: usize, seed: u64) -> Column {
    build_col(DataType::Bool, n, nth, |i| Value::Bool(splitmix(i as u64 + seed) & 1 == 0))
}

const N: usize = 61; // prime, so no null stride divides it evenly

/// Every column shape, at every null density that matters: none at all, one in
/// three, and every row.
fn columns() -> Vec<Column> {
    let mut out = Vec::new();
    for &nth in &[0usize, 3, 1] {
        out.push(ints(N, nth, 1));
        out.push(uints(N, nth, 2));
        out.push(floats(N, nth, 3));
        out.push(strs(N, nth, 4));
        out.push(decs(2, N, nth, 5));
        out.push(decs(4, N, nth, 6));
        out.push(bools(N, nth, 7));
        // Non-negative only: a `DateTime` lane is physically unsigned here, so
        // a negative one is refused at push time rather than being a corner
        // this test could reach.
        out.push(build_col(DataType::DateTime, N, nth, |i| {
            Value::DateTime(I64S[i % I64S.len()].saturating_abs())
        }));
        out.push(build_col(DataType::Date, N, nth, |i| {
            Value::Date((splitmix(i as u64) % 80_000) as u32)
        }));
    }
    out
}

/// Literals spanning every family and every boundary a specialization keys off.
fn literals() -> Vec<Value> {
    let mut out = vec![Value::Null];
    for &x in &[0i64, 1, -1, 7, 255, i64::MAX, i64::MIN, 9_007_199_254_740_993] {
        out.push(Value::Int(x));
    }
    for &x in &[0u64, 1, 7, i64::MAX as u64, i64::MAX as u64 + 1, u64::MAX] {
        out.push(Value::UInt(x));
    }
    for &x in &[0.0f64, -0.0, 1.0, 0.5, -1.5, f64::NAN, f64::INFINITY, 9.223_372_036_854_776e18] {
        out.push(Value::Float(x));
    }
    for s in ["", "a", "US", "zzz"] {
        out.push(Value::str(s));
    }
    out.push(Value::Bool(true));
    out.push(Value::Decimal(150, 2));
    out.push(Value::Decimal(15_000, 4));
    out.push(Value::Date(19_000));
    out.push(Value::DateTime(1_700_000_000));
    out
}

const OPS: [BinaryOp; 6] = [
    BinaryOp::Eq,
    BinaryOp::NotEq,
    BinaryOp::Lt,
    BinaryOp::LtEq,
    BinaryOp::Gt,
    BinaryOp::GtEq,
];

// -------------------------------------------------------------- the harness

/// Bit-for-bit column equality.
///
/// `Column`'s own `PartialEq` cannot be used as the oracle here: it compares
/// `f64` lanes with `==`, and `NaN != NaN` would fail every float expression
/// whose two paths agree perfectly. Comparing the bits is both the fix and the
/// stricter check -- the two paths run the same instructions on the same
/// operands, so a differing NaN payload or a `-0.0` that turned into `0.0`
/// would be a real difference, not a false alarm.
fn col_eq(a: &Column, b: &Column) -> bool {
    if a.ty != b.ty || a.nulls != b.nulls {
        return false;
    }
    match (&a.data, &b.data) {
        (ColumnData::F64(x), ColumnData::F64(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
        }
        (x, y) => x == y,
    }
}

/// Evaluate `e` both ways and demand the same answer.
///
/// Errors count: a fast path that accepts an expression the general path
/// rejects (or the other way round) is just as wrong as one that returns
/// different numbers, so the `Err` case is compared by message.
#[track_caller]
fn same(what: &str, e: &B, b: &Block) {
    let slow = expr::eval_general(e, b);
    let fast = expr::eval(e, b);
    match (&slow, &fast) {
        (Ok(x), Ok(y)) => {
            assert!(col_eq(x, y), "{what}: fast path disagrees\n  {e}\n  {x:?}\n  {y:?}")
        }
        (Err(x), Err(y)) => {
            assert_eq!(x.to_string(), y.to_string(), "{what}: different errors\n  {e}")
        }
        _ => panic!("{what}: one path errored and the other did not\n  {e}\n  slow={slow:?}\n  fast={fast:?}"),
    }
    // The predicate entry point has its own fused loop, so it needs its own
    // comparison rather than being implied by `eval`.
    let slow = expr::eval_predicate_general(e, b);
    let fast = expr::eval_predicate(e, b);
    match (&slow, &fast) {
        (Ok(x), Ok(y)) => assert_eq!(x, y, "{what}: predicate disagrees\n  {e}"),
        (Err(_), Err(_)) => {}
        _ => panic!("{what}: predicate errored on one path only\n  {e}"),
    }
}

fn cref(i: usize, c: &Column) -> B {
    B::Column { index: i, ty: c.ty.clone(), name: format!("c{i}") }
}

fn cmp(l: B, op: BinaryOp, r: B) -> B {
    B::Binary { left: Box::new(l), op, right: Box::new(r), ty: DataType::Bool }
}

// ------------------------------------------------------------------- tests

/// Every column shape against every literal, both operand orders, all six
/// operators. This is the specialization with the most branches -- unmaterialized
/// literals, mixed signedness collapsing to a constant answer, NaN settled per
/// block, decimal scales sent to the general path -- so it gets the widest
/// sweep.
#[test]
fn comparison_against_a_literal_matches_the_general_path() {
    let lits = literals();
    for c in columns() {
        let blk = Block::new(vec![c.clone()]).unwrap();
        let col = cref(0, &c);
        for v in &lits {
            for op in OPS {
                same("col op lit", &cmp(col.clone(), op, B::lit(v.clone())), &blk);
                // The literal on the left is the same comparison mirrored, and
                // the mirror is a separate code path.
                same("lit op col", &cmp(B::lit(v.clone()), op, col.clone()), &blk);
            }
        }
    }
}

#[test]
fn comparison_between_two_columns_matches_the_general_path() {
    let cols = columns();
    for a in &cols {
        for b in &cols {
            let blk = Block::new(vec![a.clone(), b.clone()]).unwrap();
            let (x, y) = (cref(0, a), cref(1, b));
            for op in OPS {
                same("col op col", &cmp(x.clone(), op, y.clone()), &blk);
            }
        }
    }
}

/// Comparison results feeding `AND`/`OR` is what a `WHERE` clause is, and the
/// three-valued fold has a no-mask fast path that has to agree with the general
/// one on every one of the nine TRUE/FALSE/NULL combinations.
#[test]
fn three_valued_logic_matches_the_general_path() {
    for a in columns() {
        for b in columns() {
            if a.len() != b.len() {
                continue;
            }
            let blk = Block::new(vec![a.clone(), b.clone()]).unwrap();
            let l = cmp(cref(0, &a), BinaryOp::Gt, B::lit(Value::Int(0)));
            let r = cmp(cref(1, &b), BinaryOp::Lt, B::lit(Value::Int(100)));
            for op in [BinaryOp::And, BinaryOp::Or] {
                let e = B::Binary {
                    left: Box::new(l.clone()),
                    op,
                    right: Box::new(r.clone()),
                    ty: DataType::Bool,
                };
                same("cmp AND/OR cmp", &e, &blk);
                // ...and the raw columns, which reach the fold as whatever
                // truthiness their lane kind has rather than as 0/1.
                let raw = B::Binary {
                    left: Box::new(cref(0, &a)),
                    op,
                    right: Box::new(cref(1, &b)),
                    ty: DataType::Bool,
                };
                same("col AND/OR col", &raw, &blk);
            }
            same(
                "NOT col",
                &B::Unary { op: UnaryOp::Not, expr: Box::new(cref(0, &a)), ty: DataType::Bool },
                &blk,
            );
        }
    }
}

/// Three-argument `AND`, because the variadic fold is a different loop from the
/// two-argument one.
#[test]
fn variadic_logic_matches_the_general_path() {
    let cols = columns();
    for w in cols.chunks(3) {
        if w.len() < 3 {
            continue;
        }
        let blk = Block::new(w.to_vec()).unwrap();
        for name in ["and", "or"] {
            let f = granular::exec::functions::scalar(name).unwrap();
            let e = B::Scalar {
                func: f,
                args: (0..3).map(|i| cref(i, &w[i])).collect(),
                ty: DataType::Bool,
            };
            same(name, &e, &blk);
        }
    }
}

/// The lane readers behind `+`, `-` and `*` now borrow instead of copying, and
/// borrowing the wrong buffer would read a decimal's unit count as a number.
#[test]
fn arithmetic_matches_the_general_path() {
    let cols = columns();
    for a in &cols {
        for b in &cols {
            let blk = Block::new(vec![a.clone(), b.clone()]).unwrap();
            for op in [
                BinaryOp::Plus,
                BinaryOp::Minus,
                BinaryOp::Multiply,
                BinaryOp::Divide,
                BinaryOp::IntDiv,
                BinaryOp::Modulo,
                BinaryOp::Concat,
            ] {
                let e = B::Binary {
                    left: Box::new(cref(0, a)),
                    op,
                    right: Box::new(cref(1, b)),
                    ty: DataType::Int64,
                };
                same("arith", &e, &blk);
            }
        }
    }
}

/// `IN` projects the probe list into the column's own lane domain, which is
/// only exact when every probe is an integer of the same family. The lists here
/// deliberately mix families, straddle `i64::MAX`, and carry NULLs.
#[test]
fn in_list_matches_the_general_path() {
    let lists: Vec<Vec<Value>> = vec![
        vec![],
        vec![Value::Null],
        vec![Value::Int(0)],
        vec![Value::Int(1), Value::Int(-1), Value::Int(i64::MAX)],
        vec![Value::Int(1), Value::Null, Value::Int(7)],
        vec![Value::UInt(0), Value::UInt(i64::MAX as u64), Value::UInt(i64::MAX as u64 + 1)],
        vec![Value::UInt(u64::MAX)],
        // mixed families: a float or a decimal probe must send the whole list
        // back to the general path rather than be rounded into a lane
        vec![Value::Int(1), Value::Float(1.0)],
        vec![Value::Float(0.5)],
        vec![Value::Decimal(150, 2), Value::Int(2)],
        vec![Value::str("a"), Value::str("US"), Value::str("")],
        vec![Value::str("a"), Value::Int(1), Value::Null],
        vec![Value::Bool(true), Value::Int(0)],
        vec![Value::Date(19_000), Value::DateTime(1_700_000_000)],
        // long enough to take the sorted/binary-searched branch
        (0..40i64).map(Value::Int).collect(),
        (0..40i64).map(|k| Value::UInt(k as u64)).collect(),
        (0..40).map(|k| Value::str(STRS[k % STRS.len()])).collect(),
    ];
    for c in columns() {
        let blk = Block::new(vec![c.clone()]).unwrap();
        for list in &lists {
            for negated in [false, true] {
                let e = B::InList {
                    expr: Box::new(cref(0, &c)),
                    list: list.clone(),
                    negated,
                };
                same("IN", &e, &blk);
            }
        }
    }
}

/// `LIKE` no longer splats its pattern into a column. The subject may be a
/// non-string column, which is rendered first, and that rendering is shared
/// with the old path -- so a difference here would mean the constant-pattern
/// entry point disagrees with the two-column registry one.
#[test]
fn like_matches_the_general_path() {
    let pats = ["", "%", "%%", "_", "a", "a%", "%a", "%a%", "a_", "US", "us", "\\%", "%_%", "1%"];
    for c in columns() {
        let blk = Block::new(vec![c.clone()]).unwrap();
        for p in pats {
            for negated in [false, true] {
                for ci in [false, true] {
                    let e = B::Like {
                        expr: Box::new(cref(0, &c)),
                        pattern: p.into(),
                        negated,
                        case_insensitive: ci,
                    };
                    same("LIKE", &e, &blk);
                }
            }
        }
    }
}

/// `CASE` decides the lane kind and the null mask once per block now; the arms
/// are borrowed rather than copied. Conditions that are NULL, non-Bool, or
/// never true all have to keep falling through the same way.
#[test]
fn case_matches_the_general_path() {
    let cols = columns();
    for w in cols.chunks(3) {
        if w.len() < 3 {
            continue;
        }
        let blk = Block::new(w.to_vec()).unwrap();
        for else_result in [None, Some(Box::new(cref(2, &w[2])))] {
            let e = B::Case {
                when_then: vec![
                    (cref(0, &w[0]), B::lit(Value::Int(100))),
                    (
                        cmp(cref(1, &w[1]), BinaryOp::Gt, B::lit(Value::Int(0))),
                        cref(1, &w[1]),
                    ),
                ],
                else_result: else_result.clone(),
                ty: DataType::Int64.to_nullable(),
            };
            same("CASE", &e, &blk);
        }
    }
}

/// A predicate is a comparison plus a selection vector, and the fused loop
/// builds the vector without the `Bool` column in between. Nested and negated
/// shapes go through `eval` first, so they exercise the seam between the two.
#[test]
fn predicates_match_the_general_path() {
    for c in columns() {
        let blk = Block::new(vec![c.clone()]).unwrap();
        let col = cref(0, &c);
        for v in [Value::Int(0), Value::Int(i64::MAX), Value::Float(f64::NAN), Value::str("a")] {
            for op in OPS {
                same("pred", &cmp(col.clone(), op, B::lit(v.clone())), &blk);
                let neg = B::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(cmp(col.clone(), op, B::lit(v.clone()))),
                    ty: DataType::Bool,
                };
                same("NOT pred", &neg, &blk);
            }
        }
        same("IS NULL", &B::IsNull { expr: Box::new(col.clone()), negated: false }, &blk);
        same("IS NOT NULL", &B::IsNull { expr: Box::new(col), negated: true }, &blk);
    }
}

/// A comparison over computed operands: neither side is a bare column, so the
/// literal specialization does not apply and the two sides arrive as owned
/// columns whose types the arithmetic chose.
#[test]
fn nested_expressions_match_the_general_path() {
    let cols = columns();
    for a in &cols {
        for b in &cols {
            let blk = Block::new(vec![a.clone(), b.clone()]).unwrap();
            let sum = B::Binary {
                left: Box::new(cref(0, a)),
                op: BinaryOp::Plus,
                right: Box::new(cref(1, b)),
                ty: DataType::Int64,
            };
            for op in OPS {
                same("(a+b) op b", &cmp(sum.clone(), op, cref(1, b)), &blk);
                same("(a+b) op lit", &cmp(sum.clone(), op, B::lit(Value::Int(3))), &blk);
            }
        }
    }
}

/// A zero-row block is the shape that catches an off-by-one in a `reserve` or a
/// `set_len`, and a one-row one catches a fast path that only ever ran on a
/// full vector.
#[test]
fn degenerate_block_sizes_match_the_general_path() {
    for n in [0usize, 1, 2, 7, 64, 65] {
        for nth in [0usize, 1, 2] {
            let cs =
                vec![ints(n, nth, 11), uints(n, nth, 12), floats(n, nth, 13), strs(n, nth, 14)];
            for c in &cs {
                let blk = Block::new(vec![c.clone()]).unwrap();
                for op in OPS {
                    same("tiny", &cmp(cref(0, c), op, B::lit(Value::Int(1))), &blk);
                }
                same(
                    "tiny IN",
                    &B::InList {
                        expr: Box::new(cref(0, c)),
                        list: vec![Value::Int(1), Value::Null],
                        negated: false,
                    },
                    &blk,
                );
            }
        }
    }
}

/// The selection vector itself, not just the rows it names. The branchless
/// build writes every index and only advances past the accepted ones, so a
/// wrong cursor shows up as a wrong length or a stale index rather than as a
/// wrong comparison -- which is why this asserts the vector against a
/// hand-rolled filter as well as against the general path.
#[test]
fn selection_vectors_are_exactly_the_true_rows() {
    for c in columns() {
        let blk = Block::new(vec![c.clone()]).unwrap();
        for v in [Value::Int(0), Value::Int(7), Value::str("a"), Value::Float(0.5)] {
            for op in OPS {
                let e = cmp(cref(0, &c), op, B::lit(v.clone()));
                let Ok(sel) = expr::eval_predicate(&e, &blk) else { continue };
                let bits = expr::eval_general(&e, &blk).unwrap();
                let want: Vec<u32> = (0..blk.rows() as u32)
                    .filter(|&i| {
                        !bits.is_null(i as usize) && bits.value(i as usize) == Value::Bool(true)
                    })
                    .collect();
                assert_eq!(sel, want, "selection vector for {e}");
                assert!(sel.windows(2).all(|w| w[0] < w[1]), "not ascending: {sel:?}");
            }
        }
    }
}

/// The one property that has to hold for the six-way dispatch to be sound: for
/// a total order exactly one of `<`, `=`, `>` is true of any pair, so `<=` is
/// `!>` and `>=` is `!<` and `!=` is `!=`. If the three primitives ever stop
/// partitioning -- a NaN rule that says neither Less nor Greater nor Equal --
/// the derived operators go wrong silently.
#[test]
fn the_six_operators_partition_every_pair() {
    for c in columns() {
        let blk = Block::new(vec![c.clone()]).unwrap();
        for v in literals() {
            let col = cref(0, &c);
            let of = |op| expr::eval(&cmp(col.clone(), op, B::lit(v.clone())), &blk);
            let (Ok(lt), Ok(eq), Ok(gt)) =
                (of(BinaryOp::Lt), of(BinaryOp::Eq), of(BinaryOp::Gt))
            else {
                continue;
            };
            let (le, ge, ne) = (
                of(BinaryOp::LtEq).unwrap(),
                of(BinaryOp::GtEq).unwrap(),
                of(BinaryOp::NotEq).unwrap(),
            );
            for i in 0..blk.rows() {
                if lt.is_null(i) {
                    continue;
                }
                let (l, e, g) = (
                    lt.value(i) == Value::Bool(true),
                    eq.value(i) == Value::Bool(true),
                    gt.value(i) == Value::Bool(true),
                );
                assert_eq!(l as u8 + e as u8 + g as u8, 1, "row {i} of {c:?} vs {v} is not trichotomous");
                assert_eq!(le.value(i) == Value::Bool(true), !g, "<= is not !>");
                assert_eq!(ge.value(i) == Value::Bool(true), !l, ">= is not !<");
                assert_eq!(ne.value(i) == Value::Bool(true), !e, "!= is not !=");
            }
        }
    }
}

/// The scalar registry, one argument at a time, across every column shape. Not
/// a specialization check so much as a net: `and`/`or`/`not` and the lane
/// readers are shared by dozens of entries, and a borrow that reads the wrong
/// buffer would show up here even where no test names it.
#[test]
fn scalar_registry_matches_the_general_path() {
    let unary = [
        "abs", "negate", "not", "isNull", "isNotNull", "toString", "toFloat64", "toInt64",
        "toUInt64", "length", "upper", "lower", "trim", "reverse", "sqrt", "exp", "floor",
        "ceil", "round", "toYear", "toMonth", "empty",
    ];
    for c in columns() {
        let blk = Block::new(vec![c.clone()]).unwrap();
        for name in unary {
            let Some(f) = granular::exec::functions::scalar(name) else { continue };
            let e = B::Scalar {
                func: f,
                args: vec![cref(0, &c)],
                ty: DataType::Int64.to_nullable(),
            };
            same(name, &e, &blk);
        }
    }
}

/// Two-argument entries, where the lane readers have to agree with each other
/// about which representation the pair promoted to.
#[test]
fn binary_scalar_registry_matches_the_general_path() {
    let names = ["plus", "minus", "multiply", "divide", "intDiv", "modulo", "concat", "if",
                 "ifNull", "coalesce", "greatest", "least", "position", "nullIf"];
    let cols = columns();
    for a in &cols {
        for b in &cols {
            let blk = Block::new(vec![a.clone(), b.clone()]).unwrap();
            for name in names {
                let Some(f) = granular::exec::functions::scalar(name) else { continue };
                let e = B::Scalar {
                    func: f,
                    args: vec![cref(0, a), cref(1, b)],
                    ty: DataType::Int64.to_nullable(),
                };
                same(name, &e, &blk);
            }
        }
    }
}

/// A `Arc<str>` column whose entries are all the *same* allocation is what a
/// dictionary decode hands back, and pointer-shared strings are the case a
/// length-or-`memcmp` shortcut could get wrong.
#[test]
fn shared_string_allocations_compare_the_same() {
    let one: Arc<str> = Arc::from("US");
    let two: Arc<str> = Arc::from("US");
    let col = Column::strs(
        DataType::String,
        (0..N).map(|i| if i % 2 == 0 { one.clone() } else { two.clone() }).collect(),
    );
    let blk = Block::new(vec![col.clone()]).unwrap();
    for op in OPS {
        same("shared arcs", &cmp(cref(0, &col), op, B::lit(Value::str("US"))), &blk);
        same("shared arcs", &cmp(cref(0, &col), op, B::lit(Value::str("DE"))), &blk);
    }
}

// ------------------------------------------------- oracles the switch cannot
//
// Three specializations do not live behind `eval_general`, because they are
// inside the scalar registry (whose `eval` signature is frozen) or replaced
// their predecessor outright. Comparing them against themselves would prove
// nothing, so these check them against a reference written here from the
// documented rules instead.

/// Per-row three-valued logic, straight from the truth table in the scalar
/// registry's module docs. The `and`/`or` fold now skips the whole NULL dance
/// when no argument carries a mask, which is right only if "no mask" really
/// does imply "no NULL can reach the output".
#[test]
fn and_or_match_the_truth_table_row_by_row() {
    fn truthy(c: &Column, i: usize) -> Option<bool> {
        if c.is_null(i) {
            return None;
        }
        Some(c.value(i).truthy())
    }
    let cols = columns();
    for a in &cols {
        for b in &cols {
            let blk = Block::new(vec![a.clone(), b.clone()]).unwrap();
            for (name, dominant) in [("and", false), ("or", true)] {
                let f = granular::exec::functions::scalar(name).unwrap();
                let e = B::Scalar {
                    func: f,
                    args: vec![cref(0, a), cref(1, b)],
                    ty: DataType::Bool,
                };
                let got = expr::eval(&e, &blk).unwrap();
                for i in 0..blk.rows() {
                    let (x, y) = (truthy(a, i), truthy(b, i));
                    // A dominant operand settles the row whatever the other is;
                    // otherwise an unknown makes the answer unknown.
                    let want = if x == Some(dominant) || y == Some(dominant) {
                        Some(dominant)
                    } else if x.is_none() || y.is_none() {
                        None
                    } else {
                        Some(!dominant)
                    };
                    let have = if got.is_null(i) {
                        None
                    } else {
                        Some(got.value(i) == Value::Bool(true))
                    };
                    assert_eq!(have, want, "{name} row {i}: {x:?} {name} {y:?}");
                }
            }
        }
    }
}

/// `CASE` picks the first arm whose condition is TRUE, and a NULL condition is
/// not TRUE. The per-block hoist rewrote that loop, so this checks the picking
/// itself rather than one path against another.
#[test]
fn case_picks_the_first_true_arm_row_by_row() {
    let cols = columns();
    for w in cols.chunks(2) {
        if w.len() < 2 {
            continue;
        }
        let blk = Block::new(w.to_vec()).unwrap();
        let e = B::Case {
            when_then: vec![
                (cref(0, &w[0]), B::lit(Value::Int(10))),
                (cref(1, &w[1]), B::lit(Value::Int(20))),
            ],
            else_result: Some(Box::new(B::lit(Value::Int(30)))),
            ty: DataType::Int64,
        };
        let got = expr::eval(&e, &blk).unwrap();
        for i in 0..blk.rows() {
            let fires = |c: &Column| !c.is_null(i) && c.value(i).truthy();
            let want = if fires(&w[0]) {
                10
            } else if fires(&w[1]) {
                20
            } else {
                30
            };
            assert_eq!(got.value(i), Value::Int(want), "CASE row {i}");
        }
    }
}

/// `IN` against a reference written from the NULL rules in the module docs: a
/// hit is TRUE, a miss against a list holding NULL is unknown, a NULL probe is
/// unknown, and only a miss against an entirely known list is FALSE.
#[test]
fn in_list_matches_the_null_rules_row_by_row() {
    let lists: Vec<Vec<Value>> = vec![
        vec![Value::Int(0), Value::Int(1), Value::Int(i64::MAX)],
        vec![Value::Int(0), Value::Null],
        vec![Value::UInt(i64::MAX as u64 + 1)],
        vec![Value::str("a"), Value::str("")],
        (0..40i64).map(Value::Int).collect(),
    ];
    for c in columns() {
        let blk = Block::new(vec![c.clone()]).unwrap();
        for list in &lists {
            let has_null = list.iter().any(|v| v.is_null());
            for negated in [false, true] {
                let e =
                    B::InList { expr: Box::new(cref(0, &c)), list: list.clone(), negated };
                let got = expr::eval(&e, &blk).unwrap();
                for i in 0..blk.rows() {
                    let want = if c.is_null(i) {
                        None
                    } else {
                        let v = c.value(i);
                        if list.iter().any(|x| !x.is_null() && *x == v) {
                            Some(!negated)
                        } else if has_null {
                            None
                        } else {
                            Some(negated)
                        }
                    };
                    let have = if got.is_null(i) {
                        None
                    } else {
                        Some(got.value(i) == Value::Bool(true))
                    };
                    assert_eq!(have, want, "IN row {i} of {:?} in {list:?}", c.ty);
                }
            }
        }
    }
}

/// `col LIKE 'p'` no longer goes through the registry's two-column entry, so
/// this builds that call explicitly -- pattern splatted into a column, exactly
/// as it used to be -- and demands the same answer.
#[test]
fn like_agrees_with_the_two_column_registry_entry() {
    let pats = ["", "%", "_", "a%", "%a%", "US", "\\%", "%_%"];
    for c in columns() {
        if !c.ty.is_string() {
            continue;
        }
        let blk = Block::new(vec![c.clone()]).unwrap();
        for p in pats {
            for (neg, ci, name) in [
                (false, false, "like"),
                (true, false, "notLike"),
                (false, true, "ilike"),
                (true, true, "notILike"),
            ] {
                let sugar = B::Like {
                    expr: Box::new(cref(0, &c)),
                    pattern: p.into(),
                    negated: neg,
                    case_insensitive: ci,
                };
                let splat = Column::strs(
                    DataType::String,
                    vec![Arc::from(p); blk.rows()],
                );
                let two = Block::new(vec![c.clone(), splat]).unwrap();
                let f = granular::exec::functions::scalar(name).unwrap();
                let explicit = B::Scalar {
                    func: f,
                    args: vec![
                        cref(0, &c),
                        B::Column { index: 1, ty: DataType::String, name: "p".into() },
                    ],
                    ty: DataType::Bool,
                };
                assert!(
                    col_eq(
                        &expr::eval(&sugar, &blk).unwrap(),
                        &expr::eval(&explicit, &two).unwrap()
                    ),
                    "{name} '{p}' over {:?}",
                    c.ty
                );
            }
        }
    }
}
