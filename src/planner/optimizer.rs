//! Logical plan rewrites.
//!
//! Four passes, in the order that makes each one see the most opportunity:
//!
//! 1. **constant folding** -- collapses literal arithmetic so later passes see
//!    `x > 100` rather than `x > 50 + 50`, and turns `WHERE 1 = 1` into
//!    nothing at all;
//! 2. **predicate pushdown** -- sinks each conjunct as close to the scan as it
//!    can legally go. This is the single highest-value rewrite in a columnar
//!    engine, because a predicate that reaches the scan filters rows *before*
//!    the other columns are decoded;
//! 3. **zone-filter extraction** -- turns `col <op> literal` conjuncts into
//!    granule-level pruning tests. This is what makes a selective query touch
//!    a handful of granules instead of the whole table;
//! 4. **projection pruning** -- narrows each scan to the columns actually
//!    read, so unreferenced columns are never touched at all.
//!
//! Every pass is a pure `LogicalPlan -> LogicalPlan` function, so they compose
//! and can be tested in isolation.

use crate::common::{Error, Result};
use crate::sql::ast::{BinaryOp, UnaryOp};
use crate::types::{DataType, Value};

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

// -------------------------------------------------------- 2. filter pushdown

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
                    match BoundExpr::join_conjuncts(keep) {
                        Some(p) => LogicalPlan::Filter { input: Box::new(proj), predicate: p },
                        None => proj,
                    }
                }
                other => LogicalPlan::Filter {
                    input: Box::new(other),
                    predicate: BoundExpr::join_conjuncts(conjuncts).unwrap(),
                },
            }
        }
        other => map_children_res(other, |c| sink_filter(c, depth))?,
    })
}

// ---------------------------------------------------- 3. zone-map extraction

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

// ------------------------------------------------------ 4. projection pruning

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
}
