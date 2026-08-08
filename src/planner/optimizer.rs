//! Logical plan rewrites.
//!
//! Five passes, in the order that makes each one see the most opportunity:
//!
//! 1. **constant folding** -- collapses literal arithmetic so later passes see
//!    `x > 100` rather than `x > 50 + 50`, and turns `WHERE 1 = 1` into
//!    nothing at all;
//! 2. **predicate shaping** -- rewrites a conjunct into the form the passes
//!    below it can read: `NOT` pushed inward, an OR-chain of equalities folded
//!    into an `IN`, an arithmetic identity dropped off a column;
//! 3. **predicate pushdown** -- sinks each conjunct as close to the scan as it
//!    can legally go, through joins, aggregates, unions and set operators as
//!    well as the trivial cases. This is the single highest-value rewrite in a
//!    columnar engine, because a predicate that reaches the scan filters rows
//!    *before* the other columns are decoded -- and because below a join it
//!    decides the size of the hash table rather than the size of the answer;
//! 4. **zone-filter extraction** -- turns `col <op> literal` conjuncts into
//!    granule-level pruning tests. This is what makes a selective query touch
//!    a handful of granules instead of the whole table;
//! 5. **projection pruning** -- narrows each scan to the columns actually
//!    read, so unreferenced columns are never touched at all.
//!
//! Passes 2 and 3 are one story told twice: pass 2 exists because pushdown,
//! the zone maps and the index path all recognize predicates by *shape*, and a
//! predicate written in a shape none of them recognize is a predicate that
//! reaches none of them. `NOT (k > 5)` and `k = 5 OR k = 6` are each worth
//! ~20x on this machine purely for being spelled the other way (see
//! `normalize_predicates`).
//!
//! Every pass is a pure `LogicalPlan -> LogicalPlan` function, so they compose
//! and can be tested in isolation.
//!
//! What the two new passes cost a query no rule fires on, since planning runs
//! once per query and the rules do not: `EXPLAIN` over six shapes chosen so
//! nothing moves (a point lookup, a four-conjunct WHERE, a HAVING over an
//! aggregate result, and outer joins with unsinkable conjuncts), best-of-8 x
//! 3000, three rounds, against a HEAD build. 4.9-5.7 us/plan becomes
//! 5.2-5.7 us on the cheapest and 8.5-10.8 becomes 10.0-11.4 on the rest --
//! one to two microseconds of extra tree walking, per query, against wins
//! measured in milliseconds. Cross-process rather than interleaved, because
//! the switch that made the interleaved measurements has been deleted; only a
//! difference this size or larger is meaningful.

use crate::common::{Error, Result};
use crate::sql::ast::{BinaryOp, JoinOp, UnaryOp};
use crate::types::{DataType, Schema, Value};

use super::logical::{BoundExpr, CmpOp, LogicalPlan, ZoneFilter};

/// How deep a plan or expression may nest before the optimizer refuses it.
///
/// Every pass here walks the tree recursively, and so does the physical
/// planner underneath -- so a plan too deep for *this* file is a plan that
/// would have crashed the executor a moment later. Refusing it at the top of
/// the optimizer is the last place the query can still fail cleanly.
///
/// The binder caps its own recursion at the same 200, and it builds these
/// trees, so nothing that binds can trip this. It is a backstop for plans
/// assembled by hand and for future passes that *grow* a plan (a join
/// reordering, a subquery decorrelation) rather than only shrinking it.
///
/// The depth is a by-value `usize` rather than a counter in a guard object:
/// these are free functions with no shared state, so there is nothing for an
/// early `?` return to leak.
const MAX_PLAN_DEPTH: usize = 200;

#[cold]
fn too_deep(what: &str) -> Error {
    Error::unsupported(format!(
        "{what} nests more than {MAX_PLAN_DEPTH} levels deep; every optimizer pass and the \
         physical planner recurse once per level and would run out of stack"
    ))
}

pub fn optimize(plan: LogicalPlan) -> Result<LogicalPlan> {
    let plan = fold_constants(plan)?;
    let plan = normalize_predicates(plan)?;
    let plan = push_down_filters(plan)?;
    let plan = extract_zone_filters(plan)?;
    prune_projections(plan)
}

// -------------------------------------------------------------- 1. folding

/// Evaluate an expression that mentions no columns.
///
/// Deliberately does *not* fold scalar function calls: `now()` and `rand()`
/// are in the registry, and folding them at plan time would be wrong. Only
/// operators with pure, total semantics on literals are folded.
pub fn const_eval(e: &BoundExpr) -> Option<Value> {
    const_eval_at(e, 0)
}

/// SQL truth of a value that may be NULL: `None` is UNKNOWN.
///
/// Nothing in this file may call [`Value::truthy`] directly on a value that
/// could be NULL. `truthy` folds NULL into `false`, and that single conflation
/// is what made `(NULL < 5) AND (1 = 1)` fold to `false` while the vectorized
/// path (`logic_var!` in exec/functions/scalar.rs) answered NULL for the same
/// expression against a column -- one expression, two answers, depending only
/// on whether the planner could see through it.
///
/// The two call sites that legitimately collapse UNKNOWN to "not taken" -- a
/// CASE arm falling through, and a filter conjunct admitting no row -- say so
/// with an explicit `== Some(true)`, so the collapse is visible where it
/// happens rather than hidden inside a helper.
#[inline]
fn truth(v: &Value) -> Option<bool> {
    if v.is_null() {
        None
    } else {
        Some(v.truthy())
    }
}

/// Past [`MAX_PLAN_DEPTH`] this answers "not a constant" instead of erroring:
/// the caller's contract is already "fold this if you can", and every caller
/// is correct when the answer is `None`. The expression itself is refused a
/// moment later, by [`fold_expr`], which is the pass that owns the tree.
fn const_eval_at(e: &BoundExpr, depth: usize) -> Option<Value> {
    if depth > MAX_PLAN_DEPTH {
        return None;
    }
    let depth = depth + 1;
    let const_eval = |e: &BoundExpr| const_eval_at(e, depth);
    match e {
        BoundExpr::Literal { value, .. } => Some(value.clone()),
        BoundExpr::Cast { expr, ty } => const_eval(expr)?.cast_to(ty).ok(),
        BoundExpr::Unary { op, expr, .. } => {
            let v = const_eval(expr)?;
            match op {
                UnaryOp::Not => Some(match truth(&v) {
                    Some(b) => Value::Bool(!b),
                    None => Value::Null,
                }),
                UnaryOp::Neg => match v {
                    Value::Int(i) => i.checked_neg().map(Value::Int),
                    Value::Float(f) => Some(Value::Float(-f)),
                    Value::UInt(u) if u <= i64::MAX as u64 => Some(Value::Int(-(u as i64))),
                    // `-NULL` is NULL, not "unfoldable". `e_negate` carries the
                    // input's null mask straight through, so declining here
                    // left `SELECT -CAST(NULL AS Nullable(Int64))` as a live
                    // node in the plan for a value the planner already knew.
                    Value::Null => Some(Value::Null),
                    _ => None,
                },
            }
        }
        BoundExpr::Binary { left, op, right, .. } => {
            // Short-circuit first: `false AND <anything>` folds even when the
            // other side is not constant. Three-valued, and identical in shape
            // to the `dominant`/`any_null` loop in `logic_var!`
            // (exec/functions/scalar.rs) -- a dominant operand (false for AND,
            // true for OR) decides the row outright, otherwise an UNKNOWN
            // operand poisons it, otherwise the non-dominant value stands.
            //
            // The outer `Option` is "constant at plan time?" and the inner one
            // is SQL's UNKNOWN; collapsing them into one `Option<Value>` and
            // asking `truthy()` is exactly the bug this shape prevents.
            if let Some(dominant) = match op {
                BinaryOp::And => Some(false),
                BinaryOp::Or => Some(true),
                _ => None,
            } {
                let l = const_eval(left).as_ref().map(truth);
                let r = const_eval(right).as_ref().map(truth);
                return Some(match (l, r) {
                    (Some(Some(b)), _) | (_, Some(Some(b))) if b == dominant => {
                        Value::Bool(dominant)
                    }
                    // Nothing dominates, so both sides must be known before
                    // this folds at all -- and a NULL on either side is now
                    // load-bearing rather than a silent `false`.
                    (Some(Some(_)), Some(Some(_))) => Value::Bool(!dominant),
                    (Some(_), Some(_)) => Value::Null,
                    _ => return None,
                });
            }
            let (a, b) = (const_eval(left)?, const_eval(right)?);
            if a.is_null() || b.is_null() {
                return Some(Value::Null);
            }
            if op.is_comparison() {
                let ord = a.cmp(&b);
                use std::cmp::Ordering::*;
                return Some(Value::Bool(match op {
                    BinaryOp::Eq => ord == Equal,
                    BinaryOp::NotEq => ord != Equal,
                    BinaryOp::Lt => ord == Less,
                    BinaryOp::LtEq => ord != Greater,
                    BinaryOp::Gt => ord == Greater,
                    BinaryOp::GtEq => ord != Less,
                    _ => unreachable!(),
                }));
            }
            if matches!(op, BinaryOp::Concat) {
                return Some(Value::str(format!(
                    "{}{}",
                    a.render_plain(),
                    b.render_plain()
                )));
            }
            // Every arithmetic operator below reads both operands through
            // `as_i64`/`as_f64`, and a decimal's lane is a *unit count*: the
            // integer paths would add a `Decimal64(2)` to a `Decimal64(4)`
            // lane-for-lane and hand back a constant 100x wrong, and the float
            // fallback would round `0.1 + 0.2` straight back into the answer
            // this type exists to avoid. Even matching scales fold wrong, since
            // the result comes back `Int(375)` rather than `Decimal(375, 2)`.
            //
            // One guard rather than one per branch: comparison and `||` are
            // already returned above (both are exact on decimals -- `Value::cmp`
            // widens to a common scale in i128, and `render_plain` prints the
            // point), so everything still here is arithmetic. Declining costs
            // one block of runtime `dec_arith`, which is exact.
            if a.decimal_parts().is_some() || b.decimal_parts().is_some() {
                return None;
            }
            // `DIV` and `%` are integer operations *whatever* they are handed.
            // The runtime's `int_binop!` (exec/functions/scalar.rs) pushes both
            // operands through `to_i64_vec` and types the result
            // `Nullable(Int64)`, so this has to truncate too rather than sort
            // itself by operand type below: folded as float division,
            // `(4 / 7) DIV -17.5` was -0.0326 here against the evaluator's 0,
            // and `7 DIV 0.5` was 14 against its NULL. Found by
            // `folding_never_changes_the_answer` at seed 31.
            //
            // `wrapping_*` rather than `checked_*` because that is what the
            // runtime does, right down to `i64::MIN / -1`; a zero divisor
            // nulls the row (module docs, exec/functions/scalar.rs).
            if matches!(op, BinaryOp::IntDiv | BinaryOp::Modulo) {
                let (x, y) = (a.as_i64()?, b.as_i64()?);
                return Some(if y == 0 {
                    Value::Null
                } else if matches!(op, BinaryOp::IntDiv) {
                    Value::Int(x.wrapping_div(y))
                } else {
                    Value::Int(x.wrapping_rem(y))
                });
            }
            // The rest. Integer paths use checked math so a folded overflow
            // becomes "don't fold" rather than a wrong constant.
            match (a.as_i64(), b.as_i64()) {
                (Some(x), Some(y))
                    if !matches!(a, Value::Float(_)) && !matches!(b, Value::Float(_)) =>
                {
                    let r = match op {
                        BinaryOp::Plus => x.checked_add(y),
                        BinaryOp::Minus => x.checked_sub(y),
                        BinaryOp::Multiply => x.checked_mul(y),
                        BinaryOp::Divide => {
                            let (fx, fy) = (x as f64, y as f64);
                            return Some(if fy == 0.0 {
                                Value::Null
                            } else {
                                Value::Float(fx / fy)
                            });
                        }
                        _ => None,
                    };
                    r.map(Value::Int)
                }
                _ => {
                    let (x, y) = (a.as_f64()?, b.as_f64()?);
                    Some(match op {
                        BinaryOp::Plus => Value::Float(x + y),
                        BinaryOp::Minus => Value::Float(x - y),
                        BinaryOp::Multiply => Value::Float(x * y),
                        BinaryOp::Divide => {
                            if y == 0.0 {
                                Value::Null
                            } else {
                                Value::Float(x / y)
                            }
                        }
                        _ => return None,
                    })
                }
            }
        }
        BoundExpr::InList { expr, list, negated } => {
            let v = const_eval(expr)?;
            // The same three rules `eval_in_list` (exec/expr.rs) applies per
            // row: a NULL probe is UNKNOWN, a hit is decided, and a *miss*
            // against a list that contains NULL is UNKNOWN -- the value might
            // have been the unknown entry. Only a miss against an entirely
            // known list is `false`.
            if v.is_null() {
                return Some(Value::Null);
            }
            // `v` is known non-NULL, so plain `contains` can only match a
            // non-NULL entry; the scan for NULLs is deferred to the miss path,
            // which is the rarer one and the only one that needs it.
            Some(if list.contains(&v) {
                Value::Bool(!*negated)
            } else if list.iter().any(Value::is_null) {
                Value::Null
            } else {
                Value::Bool(*negated)
            })
        }
        BoundExpr::IsNull { expr, negated } => {
            let v = const_eval(expr)?;
            Some(Value::Bool(v.is_null() != *negated))
        }
        BoundExpr::Case { when_then, else_result, .. } => {
            for (w, t) in when_then {
                // An UNKNOWN condition falls through to the next arm rather
                // than taking it -- deliberately the same collapse
                // `truthy_mask` (exec/expr.rs) makes, and the reason
                // `CASE WHEN NULL THEN 1 ELSE 2 END` is 2 and not NULL.
                if truth(&const_eval(w)?) == Some(true) {
                    return const_eval(t);
                }
            }
            match else_result {
                Some(e) => const_eval(e),
                None => Some(Value::Null),
            }
        }
        BoundExpr::Column { .. } | BoundExpr::Scalar { .. } | BoundExpr::Like { .. } => None,
    }
}

fn fold_expr(e: BoundExpr) -> Result<BoundExpr> {
    fold_expr_at(e, 0)
}

/// A/B measured interleaved, best-of-40, `--release`, three separate runs:
/// folding a 150-deep `((a + 1) + 1) ...` chain went 110us -> 7.4us (14.9x,
/// and 13.4x / 14.2x on the repeat runs), and a realistic four-conjunct WHERE
/// 757ns -> 265ns (2.9x). Of that, the `could_fold` gate below carries the
/// asymptotic win and reusing the child's box carries a further 1.7-2.2x on
/// both shapes. Clone-only cost was measured and subtracted from every side.
fn fold_expr_at(e: BoundExpr, depth: usize) -> Result<BoundExpr> {
    if depth > MAX_PLAN_DEPTH {
        return Err(too_deep("expression"));
    }
    let fold = |e: BoundExpr| fold_expr_at(e, depth + 1);
    // Fold *through* the child's existing allocation. `Box::new(fold(*e)?)`
    // frees one box and mallocs another for every interior node in the tree,
    // for a rewrite that usually returns the node unchanged; the swap costs a
    // two-word store instead. The placeholder is two unit variants, so
    // constructing it is free and it is never observed -- on the `?` path the
    // box holding it is dropped immediately.
    let boxed = |mut e: Box<BoundExpr>| -> Result<Box<BoundExpr>> {
        let placeholder = BoundExpr::Literal { value: Value::Null, ty: DataType::Bool };
        *e = fold(std::mem::replace(&mut *e, placeholder))?;
        Ok(e)
    };
    // Fold children first so `1 + (2 * 3)` collapses fully in one pass.
    let e = match e {
        BoundExpr::Unary { op, expr, ty } => {
            BoundExpr::Unary { op, expr: boxed(expr)?, ty }
        }
        BoundExpr::Binary { left, op, right, ty } => BoundExpr::Binary {
            left: boxed(left)?,
            op,
            right: boxed(right)?,
            ty,
        },
        BoundExpr::Scalar { func, args, ty } => BoundExpr::Scalar {
            func,
            args: args.into_iter().map(fold).collect::<Result<Vec<_>>>()?,
            ty,
        },
        BoundExpr::Cast { expr, ty } => BoundExpr::Cast { expr: boxed(expr)?, ty },
        BoundExpr::Case { when_then, else_result, ty } => BoundExpr::Case {
            when_then: when_then
                .into_iter()
                .map(|(w, t)| Ok((fold(w)?, fold(t)?)))
                .collect::<Result<Vec<_>>>()?,
            else_result: else_result.map(boxed).transpose()?,
            ty,
        },
        BoundExpr::InList { expr, list, negated } => {
            BoundExpr::InList { expr: boxed(expr)?, list, negated }
        }
        BoundExpr::Like { expr, pattern, negated, case_insensitive } => BoundExpr::Like {
            expr: boxed(expr)?,
            pattern,
            negated,
            case_insensitive,
        },
        BoundExpr::IsNull { expr, negated } => {
            BoundExpr::IsNull { expr: boxed(expr)?, negated }
        }
        other => other,
    };
    if !could_fold(&e) {
        return Ok(e);
    }
    Ok(match const_eval(&e) {
        Some(v) => {
            let ty = e.ty();
            BoundExpr::Literal { value: v, ty }
        }
        None => e,
    })
}

/// Can `const_eval` possibly succeed on a node whose children are *already
/// folded*?
///
/// This is what keeps the pass linear. `const_eval` re-walks the whole subtree
/// under each node, so calling it unconditionally at every level costs
/// O(nodes x depth): a left-deep `((a + 1) + 1) + 1 ...` chain re-descends its
/// entire left spine once per level, only to hit the same non-constant leaf
/// every time. Bottom-up folding has already turned every constant subtree
/// into a `Literal`, so "is my child constant?" is a tag check, and a node
/// that fails it cannot fold no matter how much of the subtree we walk.
///
/// Answering `true` too often is merely slow; answering `false` when
/// `const_eval` would have said `Some` would silently stop folding. So each
/// arm mirrors the precondition of the matching `const_eval` arm exactly:
/// every arm there starts with `const_eval(child)?`, except AND/OR, which
/// needs only *one* constant side, and CASE, which gives up at its first
/// non-constant WHEN.
fn could_fold(e: &BoundExpr) -> bool {
    let lit = |e: &BoundExpr| matches!(e, BoundExpr::Literal { .. });
    match e {
        // Already a literal: `const_eval` would clone the value out and build
        // an identical node. For a `Value::Str` that clone is an `Arc` bump
        // and a `Value` move for nothing.
        BoundExpr::Literal { .. } => false,
        BoundExpr::Binary { left, op, right, .. } if op.is_logical() => lit(left) || lit(right),
        BoundExpr::Binary { left, right, .. } => lit(left) && lit(right),
        BoundExpr::Unary { expr, .. }
        | BoundExpr::Cast { expr, .. }
        | BoundExpr::InList { expr, .. }
        | BoundExpr::IsNull { expr, .. } => lit(expr),
        BoundExpr::Case { when_then, .. } => when_then.first().is_none_or(|(w, _)| lit(w)),
        // `Scalar` is never folded (`now()`, `rand()`), `Like` has no folding
        // arm, and a `Column` is the definition of non-constant.
        BoundExpr::Column { .. } | BoundExpr::Scalar { .. } | BoundExpr::Like { .. } => false,
    }
}

fn fold_all(exprs: Vec<BoundExpr>) -> Result<Vec<BoundExpr>> {
    exprs.into_iter().map(fold_expr).collect()
}

fn fold_constants(plan: LogicalPlan) -> Result<LogicalPlan> {
    map_plan(plan, &mut |p| {
        Ok(match p {
            LogicalPlan::Filter { input, predicate } => {
                LogicalPlan::Filter { input, predicate: fold_expr(predicate)? }
            }
            LogicalPlan::Project { input, exprs, schema } => LogicalPlan::Project {
                input,
                exprs: fold_all(exprs)?,
                schema,
            },
            LogicalPlan::Scan(mut s) => {
                s.filters = fold_all(s.filters)?;
                LogicalPlan::Scan(s)
            }
            other => other,
        })
    }, 0)
}

// ------------------------------------------------------ 2. predicate shaping

/// Rewrite every predicate into the shape the passes below can read.
///
/// Nothing here changes *which rows* a predicate admits. What it changes is
/// whether the rest of the planner can see that: pushdown splits on `AND`, the
/// zone maps read `col <op> literal`, and the index path reads `pk = c` and
/// `pk IN (...)`. A predicate spelled any other way reaches none of them and
/// is evaluated per row over the whole table.
///
/// Measured on this machine, 300k rows, `SELECT count() FROM a WHERE <p>`,
/// A/B interleaved best-of-9 through a temporary switch, three runs:
///
/// ```text
///   NOT (k > 299990)          0.38 ms -> 0.017 ms   21-25x  -> k <= 299990
///   NOT (k > hi OR k < lo)    0.23 ms -> 0.021 ms   10-12x  -> two conjuncts
///   k = 1 OR k = 2 OR k = 3   0.25 ms -> 0.010 ms   23-29x  -> k IN (..), and
///                                                              an index probe
///   s + 0 = 150000            0.21 ms -> 0.056 ms   3.3-3.9x
/// ```
///
/// `CAST(5 AS UInt64)` needs nothing here: constant folding above already
/// collapses a cast over a literal, and the measurement agrees (0.95-0.99x,
/// i.e. noise, against the bare literal). Recorded so nobody adds a rule for
/// it.
///
/// Only predicate positions are rewritten -- a `Filter`, a scan's PREWHERE and
/// a join residual. A projection is a value context, where `NOT NOT x` and
/// `x + 0` are not the same expression as `x` for every `x`, and there is
/// nothing downstream that would read the rewritten shape anyway.
fn normalize_predicates(plan: LogicalPlan) -> Result<LogicalPlan> {
    map_plan(
        plan,
        &mut |p| {
            Ok(match p {
                LogicalPlan::Filter { input, predicate } => {
                    LogicalPlan::Filter { input, predicate: norm(predicate)? }
                }
                LogicalPlan::Scan(mut s) => {
                    s.filters = s.filters.into_iter().map(norm).collect::<Result<_>>()?;
                    LogicalPlan::Scan(s)
                }
                LogicalPlan::Join { left, right, op, on, residual, schema } => LogicalPlan::Join {
                    left,
                    right,
                    op,
                    on,
                    residual: residual.map(norm).transpose()?,
                    schema,
                },
                other => other,
            })
        },
        0,
    )
}

fn norm(e: BoundExpr) -> Result<BoundExpr> {
    norm_at(e, 0)
}

/// Walks only the boolean skeleton -- `AND`, `OR`, `NOT` and the comparison
/// leaves. A predicate buried inside a `CASE` arm or a function argument is
/// left alone: no pass below reads into one, so rewriting it would be work
/// nobody collects on.
fn norm_at(e: BoundExpr, depth: usize) -> Result<BoundExpr> {
    if depth > MAX_PLAN_DEPTH {
        return Err(too_deep("expression"));
    }
    let d = depth + 1;
    Ok(match e {
        BoundExpr::Unary { op: UnaryOp::Not, expr, ty } => {
            let inner = norm_at(*expr, d)?;
            match negate(&inner) {
                // Re-normalized because the rewrite exposes more of it: the
                // `AND` a De Morgan step just produced has two fresh operands
                // that may each fold into an `IN`.
                Some(n) => norm_at(n, d)?,
                None => BoundExpr::Unary { op: UnaryOp::Not, expr: Box::new(inner), ty },
            }
        }
        BoundExpr::Binary { left, op, right, ty } if op.is_logical() => {
            let e = BoundExpr::Binary {
                left: Box::new(norm_at(*left, d)?),
                op,
                right: Box::new(norm_at(*right, d)?),
                ty,
            };
            if op == BinaryOp::Or {
                or_to_in(e)
            } else {
                e
            }
        }
        BoundExpr::Binary { left, op, right, ty } if op.is_comparison() => BoundExpr::Binary {
            left: Box::new(strip_identity(*left)),
            op,
            right: Box::new(strip_identity(*right)),
            ty,
        },
        other => other,
    })
}

/// `NOT e` written without the `NOT`, or `None` when there is no such form.
///
/// Every arm is an identity of **Kleene** three-valued logic, not two-valued
/// logic, which is the only reason this is safe: `NOT (k > 5)` is `k <= 5`
/// *including* when `k` is NULL, because both are UNKNOWN there rather than
/// one being true and the other false. De Morgan holds in Kleene logic too
/// (`NOT (U OR F)` and `NOT U AND NOT F` are both UNKNOWN), and that is the
/// arm worth having: `NOT (a OR b)` becomes two conjuncts, which pushdown can
/// then move to two different places.
///
/// `NOT (x LIKE p)` -> `x NOT LIKE p` and `NOT (x IN l)` -> `x NOT IN l` are
/// exact rather than approximate: `like_const` (exec/functions/scalar.rs) XORs
/// the match with `negated` and carries the subject's null mask through
/// untouched, and `eval_in_list` answers UNKNOWN for a NULL probe whichever
/// way `negated` is set.
fn negate(e: &BoundExpr) -> Option<BoundExpr> {
    Some(match e {
        // `NOT NOT x` is `x` only when `x` is already boolean. `NOT NOT 5` is
        // TRUE and 5 is not, and this file has to hand the same value to the
        // vectorized evaluator that the evaluator would have computed itself.
        BoundExpr::Unary { op: UnaryOp::Not, expr, .. }
            if expr.ty().base() == &DataType::Bool =>
        {
            (**expr).clone()
        }
        BoundExpr::Binary { left, op, right, ty } if op.is_logical() => BoundExpr::Binary {
            left: Box::new(negate(left)?),
            op: if *op == BinaryOp::And { BinaryOp::Or } else { BinaryOp::And },
            right: Box::new(negate(right)?),
            ty: ty.clone(),
        },
        BoundExpr::Binary { left, op, right, ty } if op.is_comparison() => BoundExpr::Binary {
            left: left.clone(),
            op: negate_cmp(*op),
            right: right.clone(),
            ty: ty.clone(),
        },
        BoundExpr::InList { expr, list, negated } => BoundExpr::InList {
            expr: expr.clone(),
            list: list.clone(),
            negated: !negated,
        },
        BoundExpr::Like { expr, pattern, negated, case_insensitive } => BoundExpr::Like {
            expr: expr.clone(),
            pattern: pattern.clone(),
            negated: !negated,
            case_insensitive: *case_insensitive,
        },
        BoundExpr::IsNull { expr, negated } => {
            BoundExpr::IsNull { expr: expr.clone(), negated: !negated }
        }
        _ => return None,
    })
}

fn negate_cmp(op: BinaryOp) -> BinaryOp {
    match op {
        BinaryOp::Eq => BinaryOp::NotEq,
        BinaryOp::NotEq => BinaryOp::Eq,
        BinaryOp::Lt => BinaryOp::GtEq,
        BinaryOp::LtEq => BinaryOp::Gt,
        BinaryOp::Gt => BinaryOp::LtEq,
        BinaryOp::GtEq => BinaryOp::Lt,
        other => other,
    }
}

/// `k = 1 OR k = 2 OR k = 3` -> `k IN (1, 2, 3)`.
///
/// The `IN` is what the index path and the zone maps read; a chain of ORs
/// reaches neither, because both recognize one conjunct at a time and an OR is
/// one conjunct that mentions three values.
///
/// Truth-table identical in three-valued logic, which is why the NULL cases
/// are refused rather than handled: `k = NULL` is UNKNOWN for every `k` and a
/// NULL *in* the list makes a miss UNKNOWN, so both forms still agree -- but
/// `as_zone_filter` and `key_set` then each have to reason about it for a
/// shape no query writes.
fn or_to_in(e: BoundExpr) -> BoundExpr {
    let mut col: Option<(usize, DataType, String)> = None;
    let mut list: Vec<Value> = Vec::new();
    if !collect_in(&e, &mut col, &mut list) {
        return e;
    }
    match col {
        // One disjunct is not an OR-chain; two is the smallest one that had to
        // be written as an OR.
        Some((index, ty, name)) if list.len() > 1 => BoundExpr::InList {
            expr: Box::new(BoundExpr::Column { index, ty, name }),
            list,
            negated: false,
        },
        _ => e,
    }
}

/// Every disjunct is `col = literal` or `col IN (...)` on the *same* column.
///
/// Fails as soon as one is not, so a mixed `k = 1 OR v > 2` costs one walk and
/// no rewrite. Accepting an `IN` as a disjunct is what makes the chain
/// associativity-blind: `norm_at` folds bottom-up, so by the time the outer OR
/// of `k = 1 OR k = 2 OR k = 3` is reached its left operand is already
/// `k IN (1, 2)`.
fn collect_in(
    e: &BoundExpr,
    col: &mut Option<(usize, DataType, String)>,
    list: &mut Vec<Value>,
) -> bool {
    let mut same = |c: &BoundExpr| match (c, &*col) {
        (BoundExpr::Column { index, .. }, Some((i, ..))) => index == i,
        (BoundExpr::Column { index, ty, name }, None) => {
            *col = Some((*index, ty.clone(), name.clone()));
            true
        }
        _ => false,
    };
    match e {
        BoundExpr::Binary { left, op: BinaryOp::Or, right, .. } => {
            collect_in(left, col, list) && collect_in(right, col, list)
        }
        BoundExpr::Binary { left, op: BinaryOp::Eq, right, .. } => {
            let (c, v) = match (left.as_literal(), right.as_literal()) {
                (None, Some(v)) => (&**left, v),
                (Some(v), None) => (&**right, v),
                _ => return false,
            };
            if v.is_null() || !same(c) {
                return false;
            }
            list.push(v.clone());
            true
        }
        BoundExpr::InList { expr, list: l, negated: false } => {
            if l.iter().any(Value::is_null) || !same(expr) {
                return false;
            }
            list.extend(l.iter().cloned());
            true
        }
        _ => false,
    }
}

/// `x + 0`, `0 + x`, `x - 0`, `x * 1`, `1 * x` and a cast to the type `x`
/// already has, all of which leave a bare column where the zone maps and the
/// index want one.
///
/// **The type guard is the whole rule.** `k + 0` on a `UInt64` column binds to
/// `Int64` -- `promote` (types/datatype.rs) picks the signed type as soon as
/// either side is signed, and the literal `0` is signed -- so on this machine
/// `SELECT k + 0 FROM a` renders `18446744073709551615` as `-1`. Dropping the
/// `+ 0` would not merely re-plan that query, it would change its answer.
/// Requiring the node's own declared type to equal its operand's is what makes
/// the rewrite an identity instead of a silent widening.
///
/// Applied to the top of a comparison operand only. `(k + 0) * 2 = 4` would
/// simplify to `k * 2 = 4`, which is still not a shape any pass below reads,
/// so descending would be work with no collector.
fn strip_identity(e: BoundExpr) -> BoundExpr {
    // An *integer* literal, deliberately: `x + 0.0` is not the identity on
    // `-0.0`, and `x * Decimal(1, 0)` rescales.
    let unit = |e: &BoundExpr, want: u64| {
        matches!(e.as_literal(), Some(Value::Int(v)) if *v == want as i64)
            || matches!(e.as_literal(), Some(Value::UInt(v)) if *v == want)
    };
    match e {
        BoundExpr::Cast { expr, ty } if expr.ty() == ty => strip_identity(*expr),
        BoundExpr::Binary { left, op, right, ty } => {
            // `0 - x` is negation, not identity, so `Minus` only sheds a
            // right-hand zero.
            let shed_right = matches!(op, BinaryOp::Plus | BinaryOp::Minus) && unit(&right, 0)
                || op == BinaryOp::Multiply && unit(&right, 1);
            let shed_left = op == BinaryOp::Plus && unit(&left, 0)
                || op == BinaryOp::Multiply && unit(&left, 1);
            if shed_right && left.ty() == ty {
                return strip_identity(*left);
            }
            if shed_left && right.ty() == ty {
                return strip_identity(*right);
            }
            BoundExpr::Binary { left, op, right, ty }
        }
        other => other,
    }
}

// -------------------------------------------------------- 3. filter pushdown

fn push_down_filters(plan: LogicalPlan) -> Result<LogicalPlan> {
    sink_filter(plan, 0)
}

fn sink_filter(plan: LogicalPlan, depth: usize) -> Result<LogicalPlan> {
    if depth > MAX_PLAN_DEPTH {
        return Err(too_deep("plan"));
    }
    let depth = depth + 1;
    Ok(match plan {
        LogicalPlan::Filter { input, predicate } => {
            let input = sink_filter(*input, depth)?;
            let conjuncts = predicate.split_conjuncts();

            // One `const_eval` walk per conjunct decides both fates, where the
            // old shape walked every conjunct twice and built two Vecs:
            //
            //   * constantly FALSE kills the subtree -- and so does constantly
            //     UNKNOWN, because a filter admits TRUE rows only, so a
            //     conjunct that is never TRUE can never let a row through.
            //     This is the one place in this file where collapsing UNKNOWN
            //     into "not taken" is the answer rather than a shortcut, which
            //     is why it is spelled `!= Some(true)` rather than `!truthy()`;
            //   * constantly TRUE is noise and is dropped.
            let mut kept: Vec<BoundExpr> = Vec::with_capacity(conjuncts.len());
            for c in conjuncts {
                match const_eval(&c) {
                    Some(v) if truth(&v) != Some(true) => {
                        return Ok(LogicalPlan::Empty { schema: input.schema().clone() });
                    }
                    Some(_) => {}
                    None => kept.push(c),
                }
            }
            let conjuncts = kept;
            if conjuncts.is_empty() {
                return Ok(input);
            }

            match input {
                // Straight into the scan: this is the PREWHERE we want.
                LogicalPlan::Scan(mut s) => {
                    s.filters.extend(conjuncts);
                    LogicalPlan::Scan(s)
                }
                // Filtering before sorting is strictly cheaper and always
                // equivalent -- sorting does not change which rows exist.
                LogicalPlan::Sort { input: si, keys } => LogicalPlan::Sort {
                    input: Box::new(sink_filter(
                        LogicalPlan::Filter {
                            input: si,
                            predicate: BoundExpr::join_conjuncts(conjuncts).unwrap(),
                        },
                        depth,
                    )?),
                    keys,
                },
                // Through a projection, but only for conjuncts whose columns
                // are all bare pass-throughs: otherwise the predicate would
                // have to be rewritten in terms of expressions it cannot see.
                LogicalPlan::Project { input: pi, exprs, schema } => {
                    let passthrough: Vec<Option<usize>> =
                        exprs.iter().map(|e| e.as_column()).collect();
                    let (pushable, keep): (Vec<BoundExpr>, Vec<BoundExpr>) =
                        conjuncts.into_iter().partition(|c| {
                            c.referenced_columns()
                                .iter()
                                .all(|&i| passthrough.get(i).copied().flatten().is_some())
                        });
                    let mut new_input = *pi;
                    if !pushable.is_empty() {
                        let mapped: Vec<BoundExpr> = pushable
                            .into_iter()
                            .map(|mut c| {
                                c.remap_columns(&|i| passthrough.get(i).copied().flatten())
                                    .expect("partition guaranteed every column maps");
                                c
                            })
                            .collect();
                        new_input = sink_filter(
                            LogicalPlan::Filter {
                                input: Box::new(new_input),
                                predicate: BoundExpr::join_conjuncts(mapped).unwrap(),
                            },
                            depth,
                        )?;
                    }
                    let proj =
                        LogicalPlan::Project { input: Box::new(new_input), exprs, schema };
                    above(proj, keep)
                }

                // ------------------------------------------------ through join
                //
                // The legality table, because someone will need it. "sinks"
                // means the conjunct moves into that input; "stays" means it
                // remains a `Filter` over the join.
                //
                // ```text
                //   join    left-only conjunct   right-only conjunct   spanning
                //   INNER   sinks                sinks                 join cond
                //   CROSS   sinks                sinks                 join cond
                //   LEFT    sinks                STAYS                 stays
                //   RIGHT   STAYS                sinks                 stays
                //   FULL    STAYS                STAYS                 stays
                // ```
                //
                // The two STAYS on the outer rows are the classic bug. A LEFT
                // JOIN emits a NULL-padded row for every left row with no
                // match; a right-side conjunct evaluated *above* the join sees
                // those NULLs and answers UNKNOWN, so the row is dropped, but
                // the same conjunct evaluated *inside* the right input never
                // sees them at all and the row survives NULL-padded. Sinking
                // it silently converts the join to an INNER join. FULL has
                // that hazard on both sides at once.
                //
                // What each conjunct costs where is the reason to bother: a
                // predicate above the join filters the *answer*, one below it
                // decides how many rows the hash table is built over. A/B
                // interleaved best-of-9 through a temporary switch, three runs
                // each, `SELECT count() FROM a JOIN b ON a.k = b.k WHERE
                // a.k = <mid>`:
                //
                // ```text
                //     rows        pushdown off   on         ratio
                //   300k x 300k    13-15 ms      0.022 ms   537-637x
                //     1M x 1M      61-154 ms     0.045 ms   1028-2436x
                // ```
                //
                // The ratio grows with the row count because what is removed
                // is a hash build over a whole input and what remains is one
                // index probe.
                LogicalPlan::Join { left, right, op, mut on, residual, schema } => {
                    // The operator emits `left.columns ++ right.columns`
                    // (`assemble`, exec/operators/join.rs), so the left width
                    // is where one side's indices stop and the other's begin.
                    // If the declared schema disagrees with that sum the split
                    // point is a guess, and a guess here rewrites answers.
                    //
                    // Not a theoretical guard. A *nested* `USING` join is
                    // narrower than the concatenation: `merge_using_key`
                    // (planner/binder.rs) shadows the two per-side copies of
                    // the key and pushes one merged entry, so the enclosing
                    // join's schema is one column shorter than its inputs put
                    // together. `(p JOIN q USING (id)) JOIN s USING (id)`
                    // reaches here with `flat` false and nothing moves, which
                    // is the right answer at the wrong speed rather than the
                    // other way round.
                    let ln = left.schema().len();
                    let flat = ln + right.schema().len() == schema.len();
                    let (to_left, to_right) = match op {
                        JoinOp::Inner | JoinOp::Cross => (true, true),
                        JoinOp::Left => (true, false),
                        JoinOp::Right => (false, true),
                        JoinOp::Full => (false, false),
                    };
                    let (mut lp, mut rp, mut keep) = (Vec::new(), Vec::new(), Vec::new());
                    for c in conjuncts {
                        let cols = c.referenced_columns();
                        let side = if !flat || !deterministic(&c) {
                            None
                        } else if cols.iter().all(|&i| i < ln) {
                            Some(false)
                        } else if cols.iter().all(|&i| i >= ln) {
                            Some(true)
                        } else {
                            None
                        };
                        match side {
                            Some(false) if to_left => lp.push(c),
                            Some(true) if to_right => {
                                let mut c = c;
                                c.remap_columns(&|i| Some(i - ln))
                                    .expect("every column of a right-only conjunct is >= ln");
                                rp.push(c);
                            }
                            _ => keep.push(c),
                        }
                    }

                    // A spanning `l.x = r.y` in the WHERE is the ON clause
                    // written somewhere else, for an inner join: both reject a
                    // NULL on either side, and an inner join applies its
                    // condition to exactly the rows a filter above it would
                    // have seen. Moving it is transformative rather than
                    // tidy -- `FROM a, b WHERE a.k = b.k` is otherwise a full
                    // cross product materialized one block at a time before
                    // anything looks at it, and `CROSS JOIN` with a condition
                    // *is* `INNER JOIN` (`padding_wanted` agrees they are the
                    // same join; the operator picks hash over nested-loop on
                    // `on.is_empty()`, not on the op).
                    //
                    // Only the equi-column shape moves. Folding the rest into
                    // `residual` was tried and is not obviously a win: the
                    // residual runs inside `assemble` over the same rows the
                    // filter above would have seen, so it saves one block
                    // round-trip and nothing else.
                    if matches!(op, JoinOp::Inner | JoinOp::Cross) && flat {
                        let mut stays = Vec::with_capacity(keep.len());
                        for c in keep {
                            if let BoundExpr::Binary { left: a, op: BinaryOp::Eq, right: b, .. } =
                                &c
                            {
                                match (a.as_column(), b.as_column()) {
                                    (Some(x), Some(y)) if x < ln && y >= ln => {
                                        on.push((x, y - ln));
                                        continue;
                                    }
                                    (Some(x), Some(y)) if y < ln && x >= ln => {
                                        on.push((y, x - ln));
                                        continue;
                                    }
                                    _ => {}
                                }
                            }
                            stays.push(c);
                        }
                        keep = stays;
                    }

                    // TRANSITIVE INFERENCE. `on` says the two key columns are
                    // equal in every row the join *matches*, and equal by SQL
                    // equality -- a NULL key matches nothing, in this operator
                    // as in the standard ("NULL keys", exec/operators/join.rs)
                    // -- so both are non-NULL and interchangeable in any
                    // predicate that reads nothing else. `a.k = 150000` on one
                    // side therefore implies `b.k = 150000` on the other, and
                    // pushing both is the difference between pruning one input
                    // and pruning two. On the query above, with one-sided
                    // pushdown already on: 3.2-3.4 ms against 0.015 ms at
                    // 300k x 300k (213-251x), and 10.3-10.7 ms against
                    // 0.018 ms at 1M x 1M. One-sided pushdown alone is 4.2x,
                    // so nearly all of the 537x above is this.
                    //
                    // The legality table for the *inferred* copy is the mirror
                    // image of the one for an original conjunct, and reads the
                    // other way round on the outer joins:
                    //
                    // ```text
                    //   join    inferred into LEFT input   into RIGHT input
                    //   INNER   legal                      legal
                    //   CROSS   legal                      legal
                    //   LEFT    NO                         legal
                    //   RIGHT   legal                      NO
                    //   FULL    NO                         NO
                    // ```
                    //
                    // An inferred predicate only ever removes rows that could
                    // not have matched anything, so the one question is whether
                    // this join emits a row for an input row that matches
                    // nothing. A LEFT JOIN does for the left input, which is
                    // why nothing may be inferred *into* it -- and does not for
                    // the right, which is why a right-side conjunct that may
                    // not be written by hand (it would drop the NULL-padded
                    // rows) may still be inferred.
                    //
                    // Known boundary: only conjuncts crossing *this* join seed
                    // it. A predicate already sitting inside a subquery on the
                    // join key --
                    // `(SELECT k FROM a WHERE k = c) a JOIN b ON a.k = b.k` --
                    // is not mirrored, and measures 2.5 ms at 300k x 300k
                    // against 0.012 ms for the same query written without the
                    // subquery. Closing it needs a walk of the child subtree
                    // for single-column filters, mapped up through whatever
                    // projections lie between; `equal_columns` below is half of
                    // that machinery. Not built, because the shape only arises
                    // when somebody hand-pushes a predicate this pass would
                    // have pushed for them.
                    let (ls, rs) = (left.schema(), right.schema());
                    let (into_l, into_r) = match op {
                        JoinOp::Inner | JoinOp::Cross => (true, true),
                        JoinOp::Left => (false, true),
                        JoinOp::Right => (true, false),
                        JoinOp::Full => (false, false),
                    };
                    // Both directions are read off the *pre-extension* lists,
                    // so an inferred conjunct is never itself re-inferred back
                    // across the join it came from.
                    let add_r =
                        into_r.then(|| mirrored(&lp, &on, ls, rs, &left, false, &rp));
                    if into_l {
                        let add_l = mirrored(&rp, &on, rs, ls, &right, true, &lp);
                        lp.extend(add_l);
                    }
                    rp.extend(add_r.into_iter().flatten());

                    let op = match op {
                        JoinOp::Cross if !on.is_empty() => JoinOp::Inner,
                        other => other,
                    };

                    let join = LogicalPlan::Join {
                        left: Box::new(below(*left, lp, depth)?),
                        right: Box::new(below(*right, rp, depth)?),
                        op,
                        on,
                        residual,
                        schema,
                    };
                    above(join, keep)
                }

                // ------------------------------------------- through aggregate
                //
                // A conjunct over GROUP BY keys alone selects whole groups, so
                // running it on the input rows instead removes exactly the
                // groups it would have removed and leaves every surviving
                // group's aggregates untouched. One over an aggregate *result*
                // cannot move -- that is what HAVING is, and the binder
                // already put it here rather than in the WHERE.
                //
                // So the work left is real: the binder lowers
                // `GROUP BY k HAVING k = 5` to a `Filter` above the
                // `Aggregate`, and nothing moved it down. 300k rows, A/B
                // interleaved best-of-9, three runs: 15.6-16.6 ms ->
                // 0.036-0.042 ms, 370-430x. A *computed* group key
                // (`GROUP BY k + 1 HAVING g = c`) is 55x rather than 400x --
                // the substituted expression is not a bare column, so it
                // reaches the PREWHERE but not the zone maps.
                //
                // The output schema is `[group..., agg...]`, so a conjunct
                // qualifies iff every column it reads is below `group.len()`.
                // Column `i` is then replaced by `group[i]` -- the whole
                // expression, not a renumbering, because `GROUP BY toYear(ts)`
                // names a column that only exists above this node.
                LogicalPlan::Aggregate { input, group, aggs, schema } => {
                    let (down, keep): (Vec<BoundExpr>, Vec<BoundExpr>) =
                        conjuncts.into_iter().partition(|c| {
                            deterministic(c)
                                && c.referenced_columns().iter().all(|&i| {
                                    group.get(i).is_some_and(deterministic)
                                })
                        });
                    let mut inner = *input;
                    if !down.is_empty() {
                        let mapped: Vec<BoundExpr> = down
                            .into_iter()
                            .map(|mut c| {
                                subst_columns(&mut c, &|i| group[i].clone());
                                c
                            })
                            .collect();
                        inner = below(inner, mapped, depth)?;
                    }
                    let agg =
                        LogicalPlan::Aggregate { input: Box::new(inner), group, aggs, schema };
                    above(agg, keep)
                }

                // ----------------------------------------------- through union
                //
                // Filtering a concatenation is filtering each branch. Every
                // branch has the union's arity by construction, so a conjunct
                // means the same thing in each one and needs no rewriting --
                // only a clone, which is planning-time memory for a per-row
                // saving. Distinct-union is unaffected: dedup then filter and
                // filter then dedup are the same set.
                //
                // 300k + 300k rows, `WHERE k = 150000` outside the union:
                // 0.75-0.83 ms -> 0.015 ms, 51-52x. Outside a DISTINCT over
                // the same data it is 17.5-19.2 ms -> 0.035 ms, 495-540x --
                // the dedup is what made that one expensive.
                LogicalPlan::Union { inputs, all, schema } => {
                    let (down, keep) = movable(conjuncts);
                    let mut branches = Vec::with_capacity(inputs.len());
                    for b in inputs {
                        branches.push(below(b, down.clone(), depth)?);
                    }
                    above(LogicalPlan::Union { inputs: branches, all, schema }, keep)
                }

                // Through DISTINCT unchanged: filtering before dedup and after
                // it select the same set, and doing it first is strictly less
                // to dedup.
                LogicalPlan::Distinct { input } => {
                    let (down, keep) = movable(conjuncts);
                    let d = LogicalPlan::Distinct { input: Box::new(below(*input, down, depth)?) };
                    above(d, keep)
                }

                // `Limit`, `LimitBy` and `Window` end the descent, and all
                // three for the same reason: they choose rows by *position*
                // among their input, so removing input rows changes which ones
                // they choose. `LIMIT 5` after a filter is not `LIMIT 5`
                // before it.
                other => LogicalPlan::Filter {
                    input: Box::new(other),
                    predicate: BoundExpr::join_conjuncts(conjuncts).unwrap(),
                },
            }
        }
        other => map_children_res(other, |c| sink_filter(c, depth))?,
    })
}

/// `preds` as a `Filter` under `child`, sunk one level further, or `child`
/// itself when there is nothing left to filter with.
fn below(child: LogicalPlan, preds: Vec<BoundExpr>, depth: usize) -> Result<LogicalPlan> {
    match BoundExpr::join_conjuncts(preds) {
        Some(p) => sink_filter(LogicalPlan::Filter { input: Box::new(child), predicate: p }, depth),
        None => Ok(child),
    }
}

/// The conjuncts that could not move, put back over the node they stopped at.
fn above(child: LogicalPlan, preds: Vec<BoundExpr>) -> LogicalPlan {
    match BoundExpr::join_conjuncts(preds) {
        Some(p) => LogicalPlan::Filter { input: Box::new(child), predicate: p },
        None => child,
    }
}

/// Split conjuncts into (may move, must stay) on determinism alone, for the
/// operators where that is the only question.
fn movable(cs: Vec<BoundExpr>) -> (Vec<BoundExpr>, Vec<BoundExpr>) {
    cs.into_iter().partition(deterministic)
}

/// Functions whose value is not decided by their arguments.
///
/// Sinking a conjunct changes how many rows it runs over and in what order --
/// below a union it sees one branch at a time, below a join it sees a whole
/// input rather than the matches -- so a predicate mentioning any of these
/// would return a different number of rows in its new home. `rand()` is a
/// counter-based splitmix and `now()` is read once per block, so neither is
/// even stable within one query.
///
/// [`ScalarFn`](crate::exec::functions::ScalarFn) carries no purity flag, so
/// this is a list of names and a new impure function has to be added to it.
/// `const_eval` takes the blunter version of the same precaution and folds no
/// scalar call at all.
const IMPURE: [&str; 4] = ["now", "today", "rand", "rand64"];

fn deterministic(e: &BoundExpr) -> bool {
    let mut ok = true;
    e.visit(&mut |n| {
        if let BoundExpr::Scalar { func, .. } = n {
            ok &= !IMPURE.contains(&func.name);
        }
    });
    ok
}

/// Replace every `Column` with whatever `f` says stands in its place.
///
/// Wider than [`BoundExpr::remap_columns`], which can only renumber: a
/// conjunct sinking below an aggregate substitutes a whole GROUP BY expression
/// for the output column that names it, and one mirrored across an equi-join
/// pair has to restate the column's type and name as well as its index.
///
/// Unguarded recursion, like [`BoundExpr::visit`] and `remap_columns` next to
/// it: the binder caps expression depth at the same 200 these passes do, and
/// every tree reaching here was built by the binder.
fn subst_columns(e: &mut BoundExpr, f: &dyn Fn(usize) -> BoundExpr) {
    match e {
        BoundExpr::Column { index, .. } => *e = f(*index),
        BoundExpr::Unary { expr, .. }
        | BoundExpr::Cast { expr, .. }
        | BoundExpr::InList { expr, .. }
        | BoundExpr::Like { expr, .. }
        | BoundExpr::IsNull { expr, .. } => subst_columns(expr, f),
        BoundExpr::Binary { left, right, .. } => {
            subst_columns(left, f);
            subst_columns(right, f);
        }
        BoundExpr::Scalar { args, .. } => args.iter_mut().for_each(|a| subst_columns(a, f)),
        BoundExpr::Case { when_then, else_result, .. } => {
            for (w, t) in when_then.iter_mut() {
                subst_columns(w, f);
                subst_columns(t, f);
            }
            if let Some(x) = else_result {
                subst_columns(x, f);
            }
        }
        BoundExpr::Literal { .. } => {}
    }
}

/// Every conjunct of `src` restated against the other side of an equi-join
/// pair. See the call site for why this is legal.
///
/// Three guards, all narrow on purpose:
///
///   * the conjunct must read *exactly one* column, and some column equal to it
///     must be a key of an `on` pair. `a.k = 150000` qualifies, `a.k = a.v`
///     does not -- `a.v` has no counterpart to restate it against;
///   * the two key columns must share a base type. `on` compares them through
///     the join's own key extraction, but the mirrored *predicate* would be
///     compared under whatever rule its literal and the new column imply, and
///     `k = -1` does not mean the same thing to a `UInt64` lane as to an
///     `Int64` one;
///   * `have` is the target side's existing conjuncts. An inferred duplicate of
///     one the query already wrote costs a per-row evaluation forever and buys
///     nothing.
///
/// "some column equal to it" is what makes a star schema work. In
/// `x JOIN y ON x.k = y.k JOIN z ON y.k = z.k`, the conjunct sinking into the
/// outer join's left input is written on `x.k`, but the outer `on` pair names
/// `y.k`; without the closure over [`equal_columns`] the third table is the
/// only one that still gets scanned, and the query stays two orders of
/// magnitude off: at 300k, one hop is 27-29 ms -> 3.95 ms (8.8x) and the
/// closure is 27-29 ms -> 0.040-0.051 ms (614-672x).
fn mirrored(
    src: &[BoundExpr],
    pairs: &[(usize, usize)],
    from: &Schema,
    to: &Schema,
    from_plan: &LogicalPlan,
    from_right: bool,
    have: &[BoundExpr],
) -> Vec<BoundExpr> {
    let mut out: Vec<BoundExpr> = Vec::new();
    if pairs.is_empty() || src.is_empty() {
        return out;
    }
    let mut same = Vec::new();
    equal_columns(from_plan, &mut same);
    for c in src {
        let cols = c.referenced_columns();
        let [k] = cols[..] else { continue };
        // The equivalence class of `k`, grown to a fixpoint. Both lists are a
        // handful of entries -- one per equi-join pair in the subtree -- so the
        // quadratic sweep is cheaper than any index over them, and it runs once
        // per conjunct at plan time.
        let mut class = vec![k];
        let mut i = 0;
        while i < class.len() {
            let c = class[i];
            for &(a, b) in &same {
                for (x, y) in [(a, b), (b, a)] {
                    if x == c && !class.contains(&y) {
                        class.push(y);
                    }
                }
            }
            i += 1;
        }
        let Some(&(l, r)) = pairs
            .iter()
            .find(|(l, r)| class.contains(if from_right { r } else { l }))
        else {
            continue;
        };
        let (s, t) = if from_right { (r, l) } else { (l, r) };
        if from.ty(s).base() != to.ty(t).base() {
            continue;
        }
        let mut m = c.clone();
        subst_columns(&mut m, &|_| BoundExpr::Column {
            index: t,
            ty: to.ty(t).clone(),
            name: to.name(t).to_string(),
        });
        let text = m.to_string();
        if !have.iter().chain(out.iter()).any(|e| e.to_string() == text) {
            out.push(m);
        }
    }
    out
}

/// Pairs of output columns this subtree guarantees are equal, and non-NULL, in
/// **every** row it emits.
///
/// Only an inner join's own condition qualifies. An outer join's `on` pair says
/// nothing about its output: the NULL-padded rows are exactly the ones where
/// the two keys are not equal, which is the whole point of the padding. What
/// does survive an outer join is whatever its never-padded side already
/// guaranteed, so the recursion follows the same side the padding does not.
///
/// `Filter`, `Sort`, `Limit`, `LimitBy` and `Distinct` only drop or reorder
/// rows, so a guarantee over their input is a guarantee over their output.
/// Everything else -- projections that compute, aggregates, unions, windows --
/// answers nothing rather than a rule with a caveat, because a wrong pair here
/// mirrors a predicate onto a column that does not satisfy it.
fn equal_columns(p: &LogicalPlan, out: &mut Vec<(usize, usize)>) {
    match p {
        LogicalPlan::Join { left, right, op, on, .. } => {
            let ln = left.schema().len();
            if matches!(op, JoinOp::Inner | JoinOp::Cross) {
                out.extend(on.iter().map(|&(l, r)| (l, ln + r)));
            }
            if matches!(op, JoinOp::Inner | JoinOp::Cross | JoinOp::Left) {
                equal_columns(left, out);
            }
            if matches!(op, JoinOp::Inner | JoinOp::Cross | JoinOp::Right) {
                let at = out.len();
                equal_columns(right, out);
                for e in out[at..].iter_mut() {
                    *e = (e.0 + ln, e.1 + ln);
                }
            }
        }
        LogicalPlan::Filter { input, .. }
        | LogicalPlan::Sort { input, .. }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::LimitBy { input, .. }
        | LogicalPlan::Distinct { input } => equal_columns(input, out),
        _ => {}
    }
}

// ---------------------------------------------------- 4. zone-map extraction

/// Turn `col <op> literal` (either operand order) into a granule pruning test.
fn as_zone_filter(e: &BoundExpr) -> Option<ZoneFilter> {
    match e {
        BoundExpr::Binary { left, op, right, .. } => {
            let cmp = CmpOp::from_binary(*op)?;
            if let (Some(col), Some(v)) = (left.as_column(), right.as_literal()) {
                if v.is_null() {
                    return None;
                }
                return Some(ZoneFilter { col, op: cmp, value: v.clone() });
            }
            if let (Some(v), Some(col)) = (left.as_literal(), right.as_column()) {
                if v.is_null() {
                    return None;
                }
                // `5 < x` is `x > 5`.
                return Some(ZoneFilter { col, op: cmp.flip(), value: v.clone() });
            }
            None
        }
        // `x IN (a, b, c)` bounds x by [min, max] of the list. Weaker than
        // exact membership, but it prunes, and the filter still runs per row.
        BoundExpr::InList { expr, list, negated: false } => {
            let col = expr.as_column()?;
            let lo = list.iter().filter(|v| !v.is_null()).min()?;
            Some(ZoneFilter { col, op: CmpOp::GtEq, value: lo.clone() })
        }
        _ => None,
    }
}

/// `x IN (...)` yields a second, upper-bound filter.
fn as_zone_filter_upper(e: &BoundExpr) -> Option<ZoneFilter> {
    match e {
        BoundExpr::InList { expr, list, negated: false } => {
            let col = expr.as_column()?;
            let hi = list.iter().filter(|v| !v.is_null()).max()?;
            Some(ZoneFilter { col, op: CmpOp::LtEq, value: hi.clone() })
        }
        _ => None,
    }
}

fn extract_zone_filters(plan: LogicalPlan) -> Result<LogicalPlan> {
    map_plan(plan, &mut |p| {
        Ok(match p {
            LogicalPlan::Scan(mut s) => {
                let mut zf = Vec::new();
                for f in &s.filters {
                    if let Some(z) = as_zone_filter(f) {
                        zf.push(z);
                    }
                    if let Some(z) = as_zone_filter_upper(f) {
                        zf.push(z);
                    }
                }
                s.zone_filters = zf;
                LogicalPlan::Scan(s)
            }
            other => other,
        })
    }, 0)
}

// ------------------------------------------------------ 5. projection pruning

/// Narrow each scan to the columns something above it actually reads.
///
/// The binder chooses a scan's projection as "every column mentioned anywhere
/// in the query", a safe over-approximation. Every column dropped here is a
/// column never decoded.
fn prune_projections(plan: LogicalPlan) -> Result<LogicalPlan> {
    prune_at(plan, 0)
}

fn prune_at(plan: LogicalPlan, depth: usize) -> Result<LogicalPlan> {
    if depth > MAX_PLAN_DEPTH {
        return Err(too_deep("plan"));
    }
    let depth = depth + 1;
    // Only the always-safe case: a scan directly under a projection. Deeper
    // pruning needs a full column-liveness analysis, which is not worth the
    // failure modes until profiles say so.
    Ok(match plan {
        LogicalPlan::Project { input, exprs, schema } => match *input {
            LogicalPlan::Scan(s) => {
                let mut needed: Vec<usize> = Vec::new();
                for e in exprs.iter().chain(s.filters.iter()) {
                    for c in e.referenced_columns() {
                        if !needed.contains(&c) {
                            needed.push(c);
                        }
                    }
                }
                needed.sort_unstable();
                if needed.len() == s.schema.len() {
                    return Ok(LogicalPlan::Project {
                        input: Box::new(LogicalPlan::Scan(s)),
                        exprs,
                        schema,
                    });
                }
                let mut s = s;
                let remap: Vec<Option<usize>> = (0..s.schema.len())
                    .map(|i| needed.iter().position(|&n| n == i))
                    .collect();
                s.projection = needed.iter().map(|&i| s.projection[i]).collect();
                s.schema = s.schema.project(&needed);
                for f in s.filters.iter_mut() {
                    f.remap_columns(&|i| remap[i])?;
                }
                for z in s.zone_filters.iter_mut() {
                    z.col = remap[z.col].expect("filter columns are always retained");
                }
                let exprs: Vec<BoundExpr> = exprs
                    .into_iter()
                    .map(|mut e| {
                        e.remap_columns(&|i| remap[i])?;
                        Ok(e)
                    })
                    .collect::<Result<_>>()?;
                LogicalPlan::Project { input: Box::new(LogicalPlan::Scan(s)), exprs, schema }
            }
            other => LogicalPlan::Project {
                input: Box::new(prune_at(other, depth)?),
                exprs,
                schema,
            },
        },
        other => map_children_res(other, |c| prune_at(c, depth))?,
    })
}

// ------------------------------------------------------------------ helpers

/// Apply `f` to every node, children first.
fn map_plan(
    plan: LogicalPlan,
    f: &mut dyn FnMut(LogicalPlan) -> Result<LogicalPlan>,
    depth: usize,
) -> Result<LogicalPlan> {
    if depth > MAX_PLAN_DEPTH {
        return Err(too_deep("plan"));
    }
    let plan = map_children_res(plan, |c| map_plan(c, f, depth + 1))?;
    f(plan)
}

fn map_children_res(
    plan: LogicalPlan,
    mut f: impl FnMut(LogicalPlan) -> Result<LogicalPlan>,
) -> Result<LogicalPlan> {
    Ok(match plan {
        LogicalPlan::Filter { input, predicate } => {
            LogicalPlan::Filter { input: Box::new(f(*input)?), predicate }
        }
        LogicalPlan::Project { input, exprs, schema } => {
            LogicalPlan::Project { input: Box::new(f(*input)?), exprs, schema }
        }
        LogicalPlan::Aggregate { input, group, aggs, schema } => {
            LogicalPlan::Aggregate { input: Box::new(f(*input)?), group, aggs, schema }
        }
        LogicalPlan::Sort { input, keys } => LogicalPlan::Sort { input: Box::new(f(*input)?), keys },
        LogicalPlan::Limit { input, limit, offset } => {
            LogicalPlan::Limit { input: Box::new(f(*input)?), limit, offset }
        }
        LogicalPlan::LimitBy { input, limit, keys } => {
            LogicalPlan::LimitBy { input: Box::new(f(*input)?), limit, keys }
        }
        LogicalPlan::Distinct { input } => LogicalPlan::Distinct { input: Box::new(f(*input)?) },
        // Recursed into but never rewritten: everything below a window still
        // wants constant folding, predicate pushdown and projection pruning.
        // `push_filter` is what refuses to move a predicate *across* it, which
        // is a different question and is answered by its `other` arm.
        LogicalPlan::Window { input, node } => {
            LogicalPlan::Window { input: Box::new(f(*input)?), node }
        }
        LogicalPlan::Join { left, right, op, on, residual, schema } => LogicalPlan::Join {
            left: Box::new(f(*left)?),
            right: Box::new(f(*right)?),
            op,
            on,
            residual,
            schema,
        },
        LogicalPlan::Union { inputs, all, schema } => {
            let mut v = Vec::with_capacity(inputs.len());
            for i in inputs {
                v.push(f(i)?);
            }
            LogicalPlan::Union { inputs: v, all, schema }
        }
        leaf => leaf,
    })
}

#[cfg(test)]
mod tests {
    use super::super::logical::ScanNode;
    use super::*;
    use crate::types::{Field, Schema};

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Int64),
            Field::new("c", DataType::String),
        ])
        .unwrap()
    }

    fn scan() -> LogicalPlan {
        LogicalPlan::Scan(Box::new(ScanNode {
            table: "default.t".into(),
            projection: vec![0, 1, 2],
            schema: schema(),
            filters: vec![],
            zone_filters: vec![],
        }))
    }

    fn col(i: usize) -> BoundExpr {
        BoundExpr::Column {
            index: i,
            ty: schema().ty(i).clone(),
            name: schema().name(i).to_string(),
        }
    }

    fn bin(l: BoundExpr, op: BinaryOp, r: BoundExpr) -> BoundExpr {
        let ty = if op.is_comparison() || op.is_logical() {
            DataType::Bool
        } else {
            DataType::Int64
        };
        BoundExpr::Binary { left: Box::new(l), op, right: Box::new(r), ty }
    }

    #[test]
    fn folds_nested_arithmetic() {
        let e = bin(
            BoundExpr::lit(Value::Int(1)),
            BinaryOp::Plus,
            bin(
                BoundExpr::lit(Value::Int(2)),
                BinaryOp::Multiply,
                BoundExpr::lit(Value::Int(3)),
            ),
        );
        assert_eq!(const_eval(&e), Some(Value::Int(7)));
    }

    #[test]
    fn folds_comparisons_and_logic() {
        let e = bin(BoundExpr::lit(Value::Int(5)), BinaryOp::Gt, BoundExpr::lit(Value::Int(3)));
        assert_eq!(const_eval(&e), Some(Value::Bool(true)));

        // short circuit: `false AND <column>` is false without evaluating it
        let e = bin(BoundExpr::lit(Value::Bool(false)), BinaryOp::And, col(0));
        assert_eq!(const_eval(&e), Some(Value::Bool(false)));
        let e = bin(BoundExpr::lit(Value::Bool(true)), BinaryOp::Or, col(0));
        assert_eq!(const_eval(&e), Some(Value::Bool(true)));
        // no short circuit available
        let e = bin(BoundExpr::lit(Value::Bool(true)), BinaryOp::And, col(0));
        assert_eq!(const_eval(&e), None);
    }

    #[test]
    fn does_not_fold_on_overflow_or_impurity() {
        let e = bin(
            BoundExpr::lit(Value::Int(i64::MAX)),
            BinaryOp::Plus,
            BoundExpr::lit(Value::Int(1)),
        );
        assert_eq!(const_eval(&e), None, "overflow must not fold to a wrong value");
        assert_eq!(const_eval(&col(0)), None);
    }

    #[test]
    fn null_propagates_through_folding() {
        let e = bin(BoundExpr::lit(Value::Null), BinaryOp::Plus, BoundExpr::lit(Value::Int(1)));
        assert_eq!(const_eval(&e), Some(Value::Null));
        let e =
            bin(BoundExpr::lit(Value::Int(1)), BinaryOp::Divide, BoundExpr::lit(Value::Int(0)));
        assert_eq!(const_eval(&e), Some(Value::Null));
    }

    /// `Value::as_i64` hands back a decimal's *unit count*, so the integer fold
    /// added `Decimal64(2)` to `Decimal64(4)` lane for lane -- `1.50 + 2.2500`
    /// came out 22650 through the binary -- and the float fallback would have
    /// rounded `0.1 + 0.2` straight back into the answer this type exists to
    /// avoid. Declining costs one block of the exact runtime `dec_arith`.
    #[test]
    fn arithmetic_on_decimals_is_never_folded() {
        let d = |u, s| BoundExpr::lit(Value::Decimal(u, s));
        for op in [
            BinaryOp::Plus,
            BinaryOp::Minus,
            BinaryOp::Multiply,
            BinaryOp::Divide,
            BinaryOp::IntDiv,
            BinaryOp::Modulo,
        ] {
            // Mixed scales, matching scales, and a decimal against a plain
            // integer: all three fold to a lie, and matching scales is the trap
            // -- `150 + 225` is the right *units* and the wrong *type*.
            for e in [
                bin(d(150, 2), op, d(22_500, 4)),
                bin(d(150, 2), op, d(225, 2)),
                bin(d(150, 2), op, BoundExpr::lit(Value::Int(2))),
                bin(BoundExpr::lit(Value::Int(2)), op, d(150, 2)),
            ] {
                assert_eq!(const_eval(&e), None, "`{e}` must not fold");
            }
        }
        // But everything exact on a decimal must still fold, or the guard has
        // cost real work: `Value::cmp` widens both sides to a common scale in
        // i128, and `render_plain` prints the point.
        let lt = bin(d(150, 2), BinaryOp::Lt, d(22_500, 4));
        assert_eq!(const_eval(&lt), Some(Value::Bool(true)));
        assert_eq!(const_eval(&bin(d(150, 2), BinaryOp::Eq, d(15_000, 4))), Some(Value::Bool(true)));
        let cat = bin(d(150, 2), BinaryOp::Concat, BoundExpr::lit(Value::str("!")));
        assert_eq!(const_eval(&cat), Some(Value::str("1.50!")));
        // And the folder still agrees with the evaluator everywhere it answers.
        for e in [lt, cat] {
            assert_eq!(const_eval(&e).unwrap(), vectorized(&e).unwrap(), "`{e}`");
        }
    }

    #[test]
    fn filter_sinks_into_the_scan() {
        let p = LogicalPlan::Filter {
            input: Box::new(scan()),
            predicate: bin(col(0), BinaryOp::Gt, BoundExpr::lit(Value::Int(10))),
        };
        let out = optimize(p).unwrap();
        match &out {
            LogicalPlan::Scan(s) => {
                assert_eq!(s.filters.len(), 1);
                assert_eq!(s.zone_filters.len(), 1);
                assert_eq!(s.zone_filters[0].op, CmpOp::Gt);
                assert_eq!(s.zone_filters[0].value, Value::Int(10));
            }
            other => panic!("expected Scan, got {}", other.explain()),
        }
    }

    #[test]
    fn conjuncts_split_and_all_reach_the_scan() {
        let pred = bin(
            bin(col(0), BinaryOp::Gt, BoundExpr::lit(Value::Int(10))),
            BinaryOp::And,
            bin(col(1), BinaryOp::LtEq, BoundExpr::lit(Value::Int(99))),
        );
        let out =
            optimize(LogicalPlan::Filter { input: Box::new(scan()), predicate: pred }).unwrap();
        match &out {
            LogicalPlan::Scan(s) => {
                assert_eq!(s.filters.len(), 2);
                assert_eq!(s.zone_filters.len(), 2);
            }
            other => panic!("got {}", other.explain()),
        }
    }

    #[test]
    fn literal_on_the_left_flips_the_operator() {
        // `10 < a` must become the zone filter `a > 10`, not `a < 10`.
        let pred = bin(BoundExpr::lit(Value::Int(10)), BinaryOp::Lt, col(0));
        let out =
            optimize(LogicalPlan::Filter { input: Box::new(scan()), predicate: pred }).unwrap();
        match &out {
            LogicalPlan::Scan(s) => {
                assert_eq!(s.zone_filters[0].op, CmpOp::Gt);
                assert_eq!(s.zone_filters[0].value, Value::Int(10));
            }
            other => panic!("got {}", other.explain()),
        }
    }

    #[test]
    fn always_true_filter_disappears_and_false_kills_the_plan() {
        let t = bin(BoundExpr::lit(Value::Int(1)), BinaryOp::Eq, BoundExpr::lit(Value::Int(1)));
        let out = optimize(LogicalPlan::Filter { input: Box::new(scan()), predicate: t }).unwrap();
        assert!(matches!(out, LogicalPlan::Scan(ref s) if s.filters.is_empty()));

        let f = bin(BoundExpr::lit(Value::Int(1)), BinaryOp::Eq, BoundExpr::lit(Value::Int(2)));
        let out = optimize(LogicalPlan::Filter { input: Box::new(scan()), predicate: f }).unwrap();
        assert!(matches!(out, LogicalPlan::Empty { .. }));
    }

    #[test]
    fn filter_sinks_below_sort() {
        let p = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Sort {
                input: Box::new(scan()),
                keys: vec![super::super::logical::SortKey {
                    expr: col(0),
                    asc: true,
                    nulls_first: true,
                }],
            }),
            predicate: bin(col(0), BinaryOp::Gt, BoundExpr::lit(Value::Int(1))),
        };
        let out = optimize(p).unwrap();
        match &out {
            LogicalPlan::Sort { input, .. } => {
                assert!(matches!(**input, LogicalPlan::Scan(ref s) if s.filters.len() == 1));
            }
            other => panic!("got {}", other.explain()),
        }
    }

    #[test]
    fn projection_pruning_narrows_the_scan() {
        // SELECT a FROM t  -> the scan should stop reading b and c.
        let p = LogicalPlan::Project {
            input: Box::new(scan()),
            exprs: vec![col(0)],
            schema: schema().project(&[0]),
        };
        let out = optimize(p).unwrap();
        match &out {
            LogicalPlan::Project { input, exprs, .. } => match &**input {
                LogicalPlan::Scan(s) => {
                    assert_eq!(s.projection, vec![0]);
                    assert_eq!(s.schema.len(), 1);
                    assert_eq!(exprs[0].as_column(), Some(0));
                }
                other => panic!("got {}", other.explain()),
            },
            other => panic!("got {}", other.explain()),
        }
    }

    #[test]
    fn pruning_keeps_columns_a_filter_needs() {
        // SELECT a FROM t WHERE b > 1 -- b must survive for the filter, and
        // the filter's column index must be remapped to its new position.
        let p = LogicalPlan::Project {
            input: Box::new(LogicalPlan::Filter {
                input: Box::new(scan()),
                predicate: bin(col(1), BinaryOp::Gt, BoundExpr::lit(Value::Int(1))),
            }),
            exprs: vec![col(0)],
            schema: schema().project(&[0]),
        };
        let out = optimize(p).unwrap();
        match &out {
            LogicalPlan::Project { input, exprs, .. } => match &**input {
                LogicalPlan::Scan(s) => {
                    assert_eq!(s.projection, vec![0, 1], "c dropped, a and b kept");
                    assert_eq!(exprs[0].as_column(), Some(0));
                    assert_eq!(s.filters[0].referenced_columns(), vec![1]);
                    assert_eq!(s.zone_filters[0].col, 1);
                }
                other => panic!("got {}", other.explain()),
            },
            other => panic!("got {}", other.explain()),
        }
    }

    #[test]
    fn in_list_yields_a_bounding_range() {
        let pred = BoundExpr::InList {
            expr: Box::new(col(0)),
            list: vec![Value::Int(5), Value::Int(1), Value::Int(9)],
            negated: false,
        };
        let out =
            optimize(LogicalPlan::Filter { input: Box::new(scan()), predicate: pred }).unwrap();
        match &out {
            LogicalPlan::Scan(s) => {
                assert_eq!(s.zone_filters.len(), 2);
                let lo = s.zone_filters.iter().find(|z| z.op == CmpOp::GtEq).unwrap();
                let hi = s.zone_filters.iter().find(|z| z.op == CmpOp::LtEq).unwrap();
                assert_eq!(lo.value, Value::Int(1));
                assert_eq!(hi.value, Value::Int(9));
            }
            other => panic!("got {}", other.explain()),
        }
    }

    // ------------------------------------------------------ recursion depth

    /// A plan deeper than the guard is refused here rather than by the
    /// physical planner's stack. The shape is not exotic: every level is one
    /// `Limit`, and `optimize` walks it four times over.
    #[test]
    fn a_plan_deeper_than_the_guard_is_an_error_not_a_crash() {
        let mut p = scan();
        for _ in 0..MAX_PLAN_DEPTH + 5 {
            p = LogicalPlan::Limit { input: Box::new(p), limit: Some(1), offset: 0 };
        }
        let e = match optimize(p) {
            Err(e) => e.to_string(),
            Ok(p) => panic!("expected a depth error, got {}", p.explain()),
        };
        assert!(e.contains("nests more than"), "{e}");

        // ...and one level under the limit still optimizes normally.
        let mut p = scan();
        for _ in 0..MAX_PLAN_DEPTH - 2 {
            p = LogicalPlan::Limit { input: Box::new(p), limit: Some(1), offset: 0 };
        }
        assert!(optimize(p).is_ok());
    }

    /// Same for an expression: `fold_expr` recurses per node, and so does
    /// `const_eval` under it.
    #[test]
    fn an_expression_deeper_than_the_guard_is_an_error_not_a_crash() {
        let mut e = col(0);
        for _ in 0..MAX_PLAN_DEPTH + 5 {
            e = BoundExpr::Unary {
                op: UnaryOp::Not,
                expr: Box::new(e),
                ty: DataType::Bool,
            };
        }
        let err = match optimize(LogicalPlan::Filter { input: Box::new(scan()), predicate: e }) {
            Err(err) => err.to_string(),
            Ok(p) => panic!("expected a depth error, got {}", p.explain()),
        };
        assert!(err.contains("nests more than"), "{err}");
    }

    /// `const_eval` degrades to "not constant" rather than erroring, because
    /// that is the answer every one of its callers is already correct for.
    #[test]
    fn const_eval_gives_up_below_the_guard_instead_of_reporting() {
        let mut e = BoundExpr::lit(Value::Bool(true));
        for _ in 0..MAX_PLAN_DEPTH + 5 {
            e = BoundExpr::Unary {
                op: UnaryOp::Not,
                expr: Box::new(e),
                ty: DataType::Bool,
            };
        }
        assert_eq!(const_eval(&e), None);
        // Shallow enough, and it folds as it always did.
        let shallow = BoundExpr::Unary {
            op: UnaryOp::Not,
            expr: Box::new(BoundExpr::lit(Value::Bool(true))),
            ty: DataType::Bool,
        };
        assert_eq!(const_eval(&shallow), Some(Value::Bool(false)));
    }

    // ------------------------------------------------- three-valued logic
    //
    // The folder and the vectorized evaluator are two independent
    // implementations of the same expression semantics, and the *only* thing
    // that decides which one a query gets is whether the planner could see a
    // constant. So "they agree" is not a nice-to-have, it is the entire
    // contract of this pass, and the property test below is what pins it.

    /// xorshift64*, so a failure prints a seed that reproduces it exactly.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// NULL-heavy on purpose: a pool where NULL is rare would need thousands
    /// of cases to reach `NULL AND true`, which is the shape under test.
    /// Bools, ints, floats and strings all appear so that the cross-family
    /// comparison rules are covered by the same invariant.
    ///
    /// `numeric` narrows the pool to what an arithmetic operand may be. The
    /// generator has to respect that, because the *binder* does: `SELECT 'q' +
    /// NULL` never reaches this pass at all, it is refused with "plus:
    /// argument 1 must be numeric". Generating it anyway would have this test
    /// report a divergence -- the folder answers NULL from its
    /// either-side-is-NULL rule while the evaluator raises on the string
    /// column -- for a tree the engine cannot produce. Bools are excluded for
    /// the same reason in reverse: `(1=2) - (1=1)` *does* bind, and diverges
    /// for an unrelated reason already on file as BUG 4 in tests/differential.rs.
    fn gen_value(r: &mut Rng, numeric: bool) -> Value {
        match r.below(10) {
            0 | 1 | 2 => Value::Null,
            3 if !numeric => Value::Bool(true),
            4 if !numeric => Value::Bool(false),
            5 => Value::Int(0),
            6 => Value::Int(1),
            7 => Value::Int(-3),
            8 => Value::Float(2.5),
            9 if !numeric => Value::str("q"),
            _ => Value::Int(7),
        }
    }

    /// One CASE arm: `family` is 0 numeric, 1 boolean, 2 string. NULL belongs
    /// to every family, which is what makes CASE the cheapest way to inject an
    /// UNKNOWN into a tree that is otherwise all known values.
    fn gen_arm(r: &mut Rng, family: u64) -> Value {
        match (family, r.below(6)) {
            (_, 0 | 1) => Value::Null,
            (1, n) => Value::Bool(n % 2 == 0),
            (2, _) => Value::str("q"),
            (_, 2) => Value::Int(0),
            (_, 3) => Value::Int(-3),
            (_, 4) => Value::Float(2.5),
            _ => Value::Int(7),
        }
    }

    /// `numeric` means "this node sits where only a number may sit". It is not
    /// hereditary: a comparison or an `IN` is boolean-valued whatever its
    /// operands are, so their children are generated unrestricted, and that is
    /// where the cross-family and NULL-bearing shapes come from.
    fn gen_expr(r: &mut Rng, depth: u32, numeric: bool) -> BoundExpr {
        if depth == 0 {
            return BoundExpr::lit(gen_value(r, numeric));
        }
        let d = depth - 1;
        // Boolean-valued nodes cannot appear where a number is wanted, so in
        // numeric position the draw is folded onto the arms that can.
        let pick = match r.below(14) {
            k if numeric && !matches!(k, 0 | 1 | 5 | 9 | 10 | 13) => 9,
            k => k,
        };
        let sub = |r: &mut Rng| Box::new(gen_expr(r, d, false));
        let num = |r: &mut Rng| Box::new(gen_expr(r, d, true));
        match pick {
            0 | 1 => BoundExpr::lit(gen_value(r, numeric)),
            2 => BoundExpr::Binary {
                left: sub(r),
                op: BinaryOp::And,
                right: sub(r),
                ty: DataType::Bool.to_nullable(),
            },
            3 => BoundExpr::Binary {
                left: sub(r),
                op: BinaryOp::Or,
                right: sub(r),
                ty: DataType::Bool.to_nullable(),
            },
            4 => BoundExpr::Unary {
                op: UnaryOp::Not,
                expr: sub(r),
                ty: DataType::Bool.to_nullable(),
            },
            5 => BoundExpr::Unary {
                op: UnaryOp::Neg,
                expr: num(r),
                ty: DataType::Int64.to_nullable(),
            },
            6 | 7 | 8 => {
                let op = *[
                    BinaryOp::Eq,
                    BinaryOp::NotEq,
                    BinaryOp::Lt,
                    BinaryOp::LtEq,
                    BinaryOp::Gt,
                    BinaryOp::GtEq,
                ]
                .get(r.below(6) as usize)
                .unwrap();
                BoundExpr::Binary {
                    left: sub(r),
                    op,
                    right: sub(r),
                    ty: DataType::Bool.to_nullable(),
                }
            }
            9 | 10 => {
                let op = *[
                    BinaryOp::Plus,
                    BinaryOp::Minus,
                    BinaryOp::Multiply,
                    BinaryOp::Divide,
                    BinaryOp::IntDiv,
                    BinaryOp::Modulo,
                ]
                .get(r.below(6) as usize)
                .unwrap();
                BoundExpr::Binary {
                    left: num(r),
                    op,
                    right: num(r),
                    ty: DataType::Int64.to_nullable(),
                }
            }
            11 => {
                let n = 1 + r.below(3) as usize;
                BoundExpr::InList {
                    expr: sub(r),
                    list: (0..n).map(|_| gen_value(r, false)).collect(),
                    negated: r.below(2) == 0,
                }
            }
            12 => BoundExpr::IsNull { expr: sub(r), negated: r.below(2) == 0 },
            // Every arm of one CASE is drawn from a single type family, and the
            // node's declared type is the family's -- both because that is what
            // the binder computes and because the binder outright *refuses*
            // anything else ("no common type for Int64 and String"). Two
            // separate false reports came from getting this wrong: declaring
            // `Int64` over a `THEN 2.5` truncated the arm to 2 in the
            // vectorized path, and a `THEN 0 ELSE 'q'` gathered the taken arm
            // through `ty.base().physical()` into `Str("0")`. `eval_case`
            // reading its result type is not a bug; a generator that lies
            // about it is.
            _ => {
                let family = if numeric { 0 } else { r.below(3) };
                let arms = 1 + r.below(2) as usize;
                let mut results: Vec<Value> = Vec::new();
                let when_then: Vec<(BoundExpr, BoundExpr)> = (0..arms)
                    .map(|_| {
                        let w = gen_expr(r, d, false);
                        let t = gen_arm(r, family);
                        results.push(t.clone());
                        (w, BoundExpr::lit(t))
                    })
                    .collect();
                let else_result = (r.below(2) == 0).then(|| {
                    let v = gen_arm(r, family);
                    results.push(v.clone());
                    Box::new(BoundExpr::lit(v))
                });
                // Int and Float in one family widen to Float, exactly as
                // `CASE WHEN .. THEN 5 ELSE 2.5 END` binds.
                let ty = match family {
                    1 => DataType::Bool,
                    2 => DataType::String,
                    _ if results.iter().any(|v| matches!(v, Value::Float(_))) => DataType::Float64,
                    _ => DataType::Int64,
                };
                BoundExpr::Case { when_then, else_result, ty: ty.to_nullable() }
            }
        }
    }

    /// What the *vectorized* evaluator -- the path a query takes whenever a
    /// column is involved -- says about a one-row block.
    fn vectorized(e: &BoundExpr) -> Result<Value> {
        let c = crate::exec::expr::eval(e, &crate::types::Block::rows_only(1))?;
        Ok(c.value(0))
    }

    /// **The invariant this whole module exists to keep.**
    ///
    /// For any expression the folder is willing to collapse, the constant it
    /// produces must be the value the vectorized evaluator would have computed
    /// for the very same tree. Anything else means one query plan answers a
    /// question differently from another plan for the same question, decided
    /// by nothing more than whether a column happened to be mentioned.
    ///
    /// The comparison is `Value`'s own equality, which is by *value* and not
    /// by variant (`Int(1) == Bool(true) == Float(1.0)`), because the two
    /// paths legitimately choose different physical representations for the
    /// same answer -- but they agree on NULL, since NULL compares equal to
    /// nothing but NULL.
    ///
    /// Only one direction is asserted: folding may *decline* (that is merely a
    /// missed optimization), but it may never answer differently. It also may
    /// not answer at all where the evaluator raises -- a query that succeeds
    /// only because the planner folded it is the same defect wearing a hat.
    ///
    /// `GRANULAR_FOLD_CASES` raises the case count for a soak, the same knob
    /// `GRANULAR_DIFF_CASES` is for tests/differential.rs. The default is sized
    /// to stay inside a `cargo test` run; 2 000 000 has been run clean.
    #[test]
    fn folding_never_changes_the_answer() {
        let cases: u64 = std::env::var("GRANULAR_FOLD_CASES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20_000);
        let mut checked = 0usize;
        let mut nulls = 0usize;
        for seed in 1..=cases {
            let mut r = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
            // Depth 1 is where a single operator meets a single literal, and
            // depth 4 is where a fold has to survive being nested under three
            // other ones; cycling covers both without a second loop.
            let e = gen_expr(&mut r, 1 + (seed % 4) as u32, false);
            let Some(folded) = const_eval(&e) else { continue };
            checked += 1;
            nulls += folded.is_null() as usize;
            match vectorized(&e) {
                Ok(v) => assert!(
                    v == folded && v.is_null() == folded.is_null(),
                    "seed {seed}: `{e}` folds to {folded:?} but evaluates to {v:?}"
                ),
                Err(err) => panic!("seed {seed}: `{e}` folds to {folded:?}, but evaluating \
                     the unfolded tree fails: {err}"),
            }
        }
        // Guard against the generator quietly drifting into shapes that never
        // fold, or that fold but never produce UNKNOWN -- either would leave
        // the assertion above passing vacuously on the case it was written for.
        let cases = cases as usize;
        assert!(checked * 3 > cases, "only {checked} of {cases} cases folded at all");
        assert!(nulls * 20 > cases, "only {nulls} folded to NULL; the 3VL paths are undertested");
    }

    /// The same invariant stated as the six expressions that used to break it,
    /// so a regression names itself instead of printing a seed.
    ///
    /// Every one of these folded to a `Bool` before: `truthy()` maps NULL to
    /// `false`, and `InList` did not look at NULL at all. `NOT BETWEEN` is the
    /// reason this was a wrong-*rows* bug and not just a wrong-cell one --
    /// `BETWEEN` desugars to `>= AND <=`, so a NULL subject folded the negation
    /// to `true` and the filter admitted every row it should have excluded.
    #[test]
    fn three_valued_logic_survives_folding() {
        let n = || BoundExpr::lit(Value::Null);
        let lt5 = || bin(n(), BinaryOp::Lt, BoundExpr::lit(Value::Int(5)));
        let t = || bin(BoundExpr::lit(Value::Int(1)), BinaryOp::Eq, BoundExpr::lit(Value::Int(1)));
        let f = || bin(BoundExpr::lit(Value::Int(1)), BinaryOp::Eq, BoundExpr::lit(Value::Int(2)));

        for (what, e) in [
            ("(NULL < 5) AND true", bin(lt5(), BinaryOp::And, t())),
            ("(NULL < 5) OR false", bin(lt5(), BinaryOp::Or, f())),
            ("NULL AND NULL", bin(n(), BinaryOp::And, n())),
            ("NULL OR NULL", bin(n(), BinaryOp::Or, n())),
            (
                "NULL BETWEEN 1 AND 5",
                bin(
                    bin(n(), BinaryOp::GtEq, BoundExpr::lit(Value::Int(1))),
                    BinaryOp::And,
                    bin(n(), BinaryOp::LtEq, BoundExpr::lit(Value::Int(5))),
                ),
            ),
            (
                "NULL IN (1, 2)",
                BoundExpr::InList {
                    expr: Box::new(n()),
                    list: vec![Value::Int(1), Value::Int(2)],
                    negated: false,
                },
            ),
            (
                "5 IN (2, NULL)",
                BoundExpr::InList {
                    expr: Box::new(BoundExpr::lit(Value::Int(5))),
                    list: vec![Value::Int(2), Value::Null],
                    negated: false,
                },
            ),
            (
                "5 NOT IN (2, NULL)",
                BoundExpr::InList {
                    expr: Box::new(BoundExpr::lit(Value::Int(5))),
                    list: vec![Value::Int(2), Value::Null],
                    negated: true,
                },
            ),
        ] {
            assert_eq!(const_eval(&e), Some(Value::Null), "{what} must fold to UNKNOWN");
            // ...and `NOT UNKNOWN` is UNKNOWN, which is the step that used to
            // turn a false into a true and let rows through.
            let neg = BoundExpr::Unary {
                op: UnaryOp::Not,
                expr: Box::new(e),
                ty: DataType::Bool.to_nullable(),
            };
            assert_eq!(const_eval(&neg), Some(Value::Null), "NOT ({what}) must fold to UNKNOWN");
        }

        // A dominant operand still decides the whole expression, including
        // through an operand that is not constant at all -- losing that would
        // trade one bug for a lost optimization.
        assert_eq!(
            const_eval(&bin(BoundExpr::lit(Value::Bool(false)), BinaryOp::And, n())),
            Some(Value::Bool(false))
        );
        assert_eq!(
            const_eval(&bin(n(), BinaryOp::Or, BoundExpr::lit(Value::Bool(true)))),
            Some(Value::Bool(true))
        );
        assert_eq!(
            const_eval(&bin(BoundExpr::lit(Value::Bool(false)), BinaryOp::And, col(0))),
            Some(Value::Bool(false))
        );
        // Membership that is *decided* still folds to a plain boolean, NULLs
        // in the list notwithstanding: a hit is a hit.
        assert_eq!(
            const_eval(&BoundExpr::InList {
                expr: Box::new(BoundExpr::lit(Value::Int(2))),
                list: vec![Value::Int(2), Value::Null],
                negated: false,
            }),
            Some(Value::Bool(true))
        );
    }

    /// `WHERE <constantly UNKNOWN>` admits nothing, exactly like `WHERE false`.
    /// Before the 3VL fix this plan was reached by folding the predicate to
    /// `false`; now the predicate folds to NULL, and the emptiness has to come
    /// from the filter rule instead.
    #[test]
    fn a_constantly_unknown_filter_kills_the_plan() {
        let pred = bin(
            bin(BoundExpr::lit(Value::Null), BinaryOp::Lt, BoundExpr::lit(Value::Int(5))),
            BinaryOp::And,
            bin(col(0), BinaryOp::Gt, BoundExpr::lit(Value::Int(1))),
        );
        let out =
            optimize(LogicalPlan::Filter { input: Box::new(scan()), predicate: pred }).unwrap();
        assert!(matches!(out, LogicalPlan::Empty { .. }), "got {}", out.explain());
    }

    // ------------------------------------------------- predicate shaping
    //
    // The end-to-end proof that these keep the answer is tests/plan_pushdown.rs,
    // which runs every rewritten predicate and its hand-written equivalent over
    // a NULL-heavy fixture. What is pinned here is the *shape*: a rule that
    // stops firing is invisible to an answer test.

    fn shaped(e: BoundExpr) -> String {
        norm(e).unwrap().to_string()
    }

    fn not(e: BoundExpr) -> BoundExpr {
        BoundExpr::Unary { op: UnaryOp::Not, expr: Box::new(e), ty: DataType::Bool }
    }

    fn lit(v: i64) -> BoundExpr {
        BoundExpr::lit(Value::Int(v))
    }

    #[test]
    fn not_is_pushed_inward_until_it_disappears() {
        // The whole point of the De Morgan arm: one conjunct becomes two, and
        // pushdown moves conjuncts.
        let both = bin(bin(col(0), BinaryOp::Gt, lit(5)), BinaryOp::Or,
                       bin(col(1), BinaryOp::Lt, lit(1)));
        assert_eq!(shaped(not(both)), "((a#0 <= 5) AND (b#1 >= 1))");
        assert_eq!(norm(not(bin(col(0), BinaryOp::Gt, lit(5)))).unwrap()
                       .split_conjuncts().len(), 1);
        assert_eq!(shaped(not(bin(col(0), BinaryOp::Eq, lit(5)))), "(a#0 != 5)");
        assert_eq!(
            shaped(not(BoundExpr::IsNull { expr: Box::new(col(0)), negated: false })),
            "a#0 IS NOT NULL"
        );
        assert_eq!(
            shaped(not(BoundExpr::InList {
                expr: Box::new(col(0)),
                list: vec![Value::Int(1)],
                negated: false,
            })),
            "a#0 NOT IN (1)"
        );
        // `NOT NOT x` collapses for a boolean `x` and must not for anything
        // else: `NOT NOT 5` is TRUE, and 5 is not.
        let b = bin(col(0), BinaryOp::Gt, lit(5));
        assert_eq!(shaped(not(not(b))), "(a#0 > 5)");
        assert_eq!(shaped(not(not(col(0)))), "NOT (NOT (a#0))");
    }

    #[test]
    fn an_or_chain_on_one_column_becomes_an_in_list() {
        let eq = |v: i64| bin(col(0), BinaryOp::Eq, lit(v));
        let chain = bin(bin(eq(1), BinaryOp::Or, eq(2)), BinaryOp::Or, eq(3));
        assert_eq!(shaped(chain), "a#0 IN (1, 2, 3)");
        // Right-associated, and with the literal on the left, and merging into
        // an `IN` that is already there.
        let flipped = bin(eq(1), BinaryOp::Or, bin(lit(2), BinaryOp::Eq, col(0)));
        assert_eq!(shaped(flipped), "a#0 IN (1, 2)");
        let merged = bin(
            BoundExpr::InList {
                expr: Box::new(col(0)),
                list: vec![Value::Int(1), Value::Int(2)],
                negated: false,
            },
            BinaryOp::Or,
            eq(3),
        );
        assert_eq!(shaped(merged), "a#0 IN (1, 2, 3)");

        // ...and the three shapes that must stay an OR. Two columns is not a
        // chain on one; a NULL probe would need `as_zone_filter` and `key_set`
        // to grow a rule for a shape no query writes; and a disjunct that is
        // not an equality at all cannot join the list.
        let two_cols = bin(eq(1), BinaryOp::Or, bin(col(1), BinaryOp::Eq, lit(2)));
        assert_eq!(shaped(two_cols), "((a#0 = 1) OR (b#1 = 2))");
        let nul = bin(eq(1), BinaryOp::Or, bin(col(0), BinaryOp::Eq, BoundExpr::lit(Value::Null)));
        assert_eq!(shaped(nul), "((a#0 = 1) OR (a#0 = NULL))");
        let ranged = bin(eq(1), BinaryOp::Or, bin(col(0), BinaryOp::Gt, lit(9)));
        assert_eq!(shaped(ranged), "((a#0 = 1) OR (a#0 > 9))");
    }

    #[test]
    fn an_arithmetic_identity_is_shed_only_when_the_type_survives() {
        // `a` is Int64 and so is the literal, so `a + 0` really is `a`.
        let p = bin(bin(col(0), BinaryOp::Plus, lit(0)), BinaryOp::Gt, lit(5));
        assert_eq!(shaped(p), "(a#0 > 5)");
        for (l, op, r) in [
            (lit(0), BinaryOp::Plus, col(0)),
            (col(0), BinaryOp::Minus, lit(0)),
            (col(0), BinaryOp::Multiply, lit(1)),
            (lit(1), BinaryOp::Multiply, col(0)),
        ] {
            let p = bin(bin(l, op, r), BinaryOp::Gt, lit(5));
            assert_eq!(shaped(p), "(a#0 > 5)", "{op:?}");
        }
        // Nested, and on the literal side too.
        let inner = bin(bin(col(0), BinaryOp::Plus, lit(0)), BinaryOp::Plus, lit(0));
        assert_eq!(shaped(bin(inner, BinaryOp::Gt, lit(5))), "(a#0 > 5)");

        // `0 - a` is negation, not identity.
        let neg = bin(bin(lit(0), BinaryOp::Minus, col(0)), BinaryOp::Gt, lit(5));
        assert_eq!(shaped(neg), "((0 - a#0) > 5)");
        // A widening `+ 0` keeps its node. Spelled by hand because the binder
        // is what normally computes the mismatch: `UInt64 + Int64` promotes to
        // `Int64`, and `18446744073709551615 + 0` is -1 there.
        let widened = BoundExpr::Binary {
            left: Box::new(BoundExpr::Column {
                index: 0,
                ty: DataType::UInt64,
                name: "a".into(),
            }),
            op: BinaryOp::Plus,
            right: Box::new(lit(0)),
            ty: DataType::Int64,
        };
        assert_eq!(shaped(bin(widened, BinaryOp::Gt, lit(5))), "((a#0 + 0) > 5)");
    }

    #[test]
    fn only_a_purely_boolean_skeleton_is_reshaped() {
        // A predicate inside a CASE arm is a value context and is left alone --
        // nothing below reads into one, so rewriting it collects nothing.
        let case = BoundExpr::Case {
            when_then: vec![(not(bin(col(0), BinaryOp::Gt, lit(5))), lit(1))],
            else_result: None,
            ty: DataType::Int64,
        };
        assert_eq!(shaped(case.clone()), case.to_string());
    }

    // ------------------------------------------------------ pushdown legality

    /// The join legality table, asserted rather than only commented. The
    /// end-to-end version, over NULLs on both sides of all five join types, is
    /// `every_join_type_and_predicate_answers_what_a_nested_loop_says` in
    /// tests/plan_pushdown.rs; this is the plan shape it rests on.
    #[test]
    fn a_conjunct_sinks_into_exactly_the_sides_the_table_allows() {
        use crate::types::Field;
        let side = || {
            LogicalPlan::Scan(Box::new(ScanNode {
                table: "default.t".into(),
                projection: vec![0],
                schema: Schema::new(vec![Field::new("a", DataType::Int64)]).unwrap(),
                filters: vec![],
                zone_filters: vec![],
            }))
        };
        let joined = Schema::new(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Int64),
        ])
        .unwrap();
        // `left#0 = 5` and `right#1 = 5`, over `t JOIN t ON l#0 = r#0`.
        for (op, want_left, want_right) in [
            (JoinOp::Inner, "both", "both"),
            (JoinOp::Left, "both", "none"),
            (JoinOp::Right, "none", "both"),
            (JoinOp::Full, "none", "none"),
        ] {
            for (at, want) in [(0usize, want_left), (1, want_right)] {
                let pred = BoundExpr::Binary {
                    left: Box::new(BoundExpr::Column {
                        index: at,
                        ty: DataType::Int64,
                        name: "a".into(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(lit(5)),
                    ty: DataType::Bool,
                };
                let out = optimize(LogicalPlan::Filter {
                    input: Box::new(LogicalPlan::Join {
                        left: Box::new(side()),
                        right: Box::new(side()),
                        op,
                        on: vec![(0, 0)],
                        residual: None,
                        schema: joined.clone(),
                    }),
                    predicate: pred,
                })
                .unwrap();
                let e = out.explain();
                let got = match (e.contains("Filter"), e.matches("prewhere").count()) {
                    (true, 0) => "none",
                    (false, 2) => "both",
                    _ => "other",
                };
                assert_eq!(got, want, "{op:?}, conjunct on column {at}:\n{e}");
            }
        }
    }

    /// The inference is what makes "both": one written conjunct, two pruned
    /// inputs. Without the `on` pair there is nothing to mirror it across.
    #[test]
    fn without_an_equi_pair_there_is_nothing_to_infer() {
        use crate::types::Field;
        let side = || {
            LogicalPlan::Scan(Box::new(ScanNode {
                table: "default.t".into(),
                projection: vec![0],
                schema: Schema::new(vec![Field::new("a", DataType::Int64)]).unwrap(),
                filters: vec![],
                zone_filters: vec![],
            }))
        };
        let joined = Schema::new(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::Int64),
        ])
        .unwrap();
        let out = optimize(LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Join {
                left: Box::new(side()),
                right: Box::new(side()),
                op: JoinOp::Cross,
                on: vec![],
                residual: None,
                schema: joined,
            }),
            predicate: bin(col(0), BinaryOp::Eq, lit(5)),
        })
        .unwrap();
        let e = out.explain();
        assert_eq!(e.matches("prewhere").count(), 1, "only the side it was written on:\n{e}");
    }

    /// `equal_columns` is the difference between a two-table join and a star
    /// schema. It must report only what survives to the *output*: an outer
    /// join's own condition does not, because the NULL-padded rows are exactly
    /// the ones where it fails.
    #[test]
    fn only_an_inner_joins_condition_is_an_output_equality() {
        use crate::types::Field;
        let leaf = |n: usize| {
            LogicalPlan::Scan(Box::new(ScanNode {
                table: "default.t".into(),
                projection: (0..n).collect(),
                schema: Schema::new(
                    (0..n).map(|i| Field::new(format!("c{i}"), DataType::Int64)).collect(),
                )
                .unwrap(),
                filters: vec![],
                zone_filters: vec![],
            }))
        };
        let sch = |n: usize| {
            Schema::new((0..n).map(|i| Field::new(format!("c{i}"), DataType::Int64)).collect())
                .unwrap()
        };
        let join = |l: LogicalPlan, r: LogicalPlan, op, on: Vec<(usize, usize)>, w| {
            LogicalPlan::Join {
                left: Box::new(l),
                right: Box::new(r),
                op,
                on,
                residual: None,
                schema: sch(w),
            }
        };
        let mut v = Vec::new();
        equal_columns(&join(leaf(1), leaf(1), JoinOp::Inner, vec![(0, 0)], 2), &mut v);
        assert_eq!(v, vec![(0, 1)]);

        for op in [JoinOp::Left, JoinOp::Right, JoinOp::Full] {
            let mut v = Vec::new();
            equal_columns(&join(leaf(1), leaf(1), op, vec![(0, 0)], 2), &mut v);
            assert!(v.is_empty(), "{op:?} pads, so its keys are not equal in the output");
        }

        // Nested: the inner join's pair has to come back shifted by the outer
        // left width, and only from a side the outer join does not pad.
        let inner = join(leaf(1), leaf(1), JoinOp::Inner, vec![(0, 0)], 2);
        let mut v = Vec::new();
        equal_columns(&join(leaf(1), inner, JoinOp::Inner, vec![(0, 0)], 3), &mut v);
        assert_eq!(v, vec![(0, 1), (1, 2)], "own pair, then the child's shifted by 1");

        let inner = join(leaf(1), leaf(1), JoinOp::Inner, vec![(0, 0)], 2);
        let mut v = Vec::new();
        equal_columns(&join(leaf(1), inner, JoinOp::Left, vec![(0, 0)], 3), &mut v);
        assert!(v.is_empty(), "the child is on the padded side, so nothing survives");
    }

    #[test]
    fn an_impure_predicate_never_moves() {
        let rand = BoundExpr::Scalar {
            func: crate::exec::functions::scalar::lookup("rand").unwrap(),
            args: vec![],
            ty: DataType::UInt64,
        };
        assert!(!deterministic(&bin(col(0), BinaryOp::Eq, rand)));
        assert!(deterministic(&bin(col(0), BinaryOp::Eq, lit(5))));
        let lower = BoundExpr::Scalar {
            func: crate::exec::functions::scalar::lookup("lower").unwrap(),
            args: vec![col(2)],
            ty: DataType::String,
        };
        assert!(deterministic(&lower), "an ordinary function is not impure");
    }
}
