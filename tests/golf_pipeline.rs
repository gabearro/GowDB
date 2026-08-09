//! The pipeline must answer the same thing however it is driven.
//!
//! This file exists because of a specific class of mistake: the optimizations
//! in `exec::operators` are invisible from the outside *until they are wrong*.
//! `Project` now hands a bare column reference out of the input block **by
//! value** instead of cloning it, which is only safe because every computed
//! expression in the same projection is evaluated first; get that ordering
//! backwards and `SELECT b, a + b` silently returns garbage for the sum. So
//! the assertions here are values, never timings, and they are chosen to walk
//! both sides of every branch that operator takes:
//!
//!   * `Take`     -- a bare column nothing else in the projection wants
//!   * `Copy`     -- the same column named twice, where only the last may move
//!   * `Computed` -- an expression, which must still see the whole input block
//!   * the empty projection, which carries a row count and no columns
//!   * the out-of-range fallback, which must be an error and not a panic
//!
//! and because a fast path that quietly swallows the general case is the
//! dangerous outcome, every query is also run through *both* builders --
//! `operators::build_physical`, which drops the exchange and runs one thread,
//! and `exchange::build`, which fans the same plan across the pool -- and the
//! two results are compared cell for cell.

use granular::catalog::Catalog;
use granular::exec::operators::{self, Operator, QueryContext};
use granular::planner::logical::{BoundExpr, LogicalPlan, ScanNode, SortKey};
use granular::planner::{binder::Binder, optimizer, physical};
use granular::sql::ast::{BinaryOp, ObjectName, Statement};
use granular::types::{Block, Column, ColumnBuilder, DataType, Field, Schema, Value};
use granular::Session;

/// Enough rows that the exchange actually goes parallel and a scan produces
/// many blocks, but small enough to stay a test.
const N: usize = 300_000;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const COUNTRIES: [&str; 8] = ["US", "DE", "FR", "JP", "BR", "IN", "GB", "CA"];

fn country_of(i: usize) -> &'static str {
    COUNTRIES[(splitmix64(i as u64) >> 20) as usize % COUNTRIES.len()]
}
fn latency_of(i: usize) -> u64 {
    splitmix64(i as u64) % 900 + 10
}
/// Every 97th row is NULL, so a null bitmap has to survive being moved out of
/// a block along with the values it describes.
fn bytes_of(i: usize) -> Option<i64> {
    (i % 97 != 0).then(|| (splitmix64(i as u64) % 65536) as i64)
}

/// `events` (five columns, one string, one nullable) plus a `dim` to join.
fn db() -> Session {
    let mut db = Session::in_memory();
    db.execute(
        "CREATE TABLE events (
            ts       DateTime,
            user_id  UInt32,
            country  String,
            latency  UInt32,
            bytes    Nullable(Int64)
        ) ENGINE = MergeTree ORDER BY ts",
    )
    .unwrap();
    let mut bytes = ColumnBuilder::with_capacity(DataType::Nullable(Box::new(DataType::Int64)), N);
    for i in 0..N {
        match bytes_of(i) {
            None => bytes.push_null(),
            Some(v) => bytes.push_value(&Value::Int(v)).unwrap(),
        }
    }
    let blk = Block::new(vec![
        Column::u64s(
            DataType::DateTime,
            (0..N).map(|i| 1_700_000_000 + i as u64).collect(),
        ),
        Column::u64s(DataType::UInt32, (0..N).map(|i| splitmix64(i as u64) % 1000).collect()),
        Column::strs(DataType::String, (0..N).map(|i| country_of(i).into()).collect()),
        Column::u64s(DataType::UInt32, (0..N).map(latency_of).collect()),
        bytes.finish(),
    ])
    .unwrap();
    db.catalog.table_mut(&ObjectName::bare("events")).unwrap().insert(blk).unwrap();

    db.execute("CREATE TABLE dim (k UInt32, tag String) ENGINE = MergeTree ORDER BY k").unwrap();
    let ks: Vec<u64> = (10..910).step_by(3).collect();
    let dim = Block::new(vec![
        Column::u64s(DataType::UInt32, ks.clone()),
        Column::strs(DataType::String, ks.iter().map(|k| format!("t{k}").into()).collect()),
    ])
    .unwrap();
    db.catalog.table_mut(&ObjectName::bare("dim")).unwrap().insert(dim).unwrap();
    db.catalog.flush_all().unwrap();
    db
}

fn plan_of(cat: &Catalog, sql: &str) -> LogicalPlan {
    let Statement::Query(q) = granular::sql::parser::parse_one(sql).unwrap() else {
        panic!("`{sql}` is not a query")
    };
    optimizer::optimize(Binder::new(cat).bind_query(&q).unwrap()).unwrap()
}

/// One cell, rendered *with its variant*. Comparing `Value`s directly would
/// not catch a path that returned `UInt(7)` where the other returned `Int(7)`,
/// because `Value`'s `Eq` equates numerics across representations -- and a
/// projection that moved a column instead of cloning it is exactly the change
/// that could alter which variant comes back.
fn cell(v: &Value) -> String {
    match v {
        Value::Null => "NULL".to_string(),
        v => format!("{:?}|{}", std::mem::discriminant(v), v.render_plain()),
    }
}

fn cells(op: &mut Box<dyn operators::Operator + '_>) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    while let Some(b) = op.next().unwrap() {
        for r in 0..b.rows() {
            out.push((0..b.width()).map(|c| cell(&b.column(c).value(r))).collect());
        }
    }
    out
}

/// The scalar in the single cell of a single-row answer.
fn scalar(rows: &[Vec<String>], sql: &str) -> String {
    assert_eq!(rows.len(), 1, "`{sql}` returned {} rows, wanted 1", rows.len());
    rows[0][0].split('|').nth(1).unwrap().to_string()
}

fn field(rows: &[Vec<String>], r: usize, c: usize) -> &str {
    rows[r][c].split('|').nth(1).unwrap()
}

/// Run `sql` through both builders and return the (identical) rows.
///
/// `operators::build_physical` drops the `Exchange` node the planner emitted
/// and builds the subtree under it, which is precisely the serial pipeline;
/// `exchange::build` honours it. Same plan, same expressions, two drivers.
fn both_ways(cat: &Catalog, sql: &str) -> Vec<Vec<String>> {
    let plan = plan_of(cat, sql);
    let ctx = QueryContext::new();
    let mut serial =
        operators::build_physical(physical::lower(&plan, cat).unwrap(), cat, &ctx).unwrap();
    let a = cells(&mut serial);
    drop(serial);
    let mut par =
        granular::exec::exchange::build(physical::lower(&plan, cat).unwrap(), cat, &ctx).unwrap();
    let b = cells(&mut par);
    drop(par);
    assert_eq!(a, b, "serial and parallel pipelines disagree on `{sql}`");
    assert_eq!(ctx.mem.used(), 0, "`{sql}` kept part of its reservation");
    a
}

// -------------------------------------------------------- projection shapes

#[test]
fn projection_shapes_agree_with_hand_computed_answers() {
    let db = db();
    let cat = &db.catalog;

    // `Take` only: two bare columns, reordered relative to the table.
    let got = both_ways(cat, "SELECT country, ts FROM events ORDER BY ts LIMIT 4");
    assert_eq!(got.len(), 4);
    for (i, row) in got.iter().enumerate() {
        assert_eq!(row[0].split('|').nth(1).unwrap(), country_of(i));
        assert_eq!(
            row[1],
            cell(&Value::DateTime(1_700_000_000 + i as i64)),
            "a moved DateTime column must keep its variant, not decay to UInt"
        );
    }

    // `Copy` + `Take`: the same column three times. Only the last mention may
    // move, and all three have to hold the same values.
    let got = both_ways(cat, "SELECT latency, latency, latency FROM events ORDER BY ts LIMIT 5");
    for (i, row) in got.iter().enumerate() {
        assert_eq!(row[0], row[1], "a duplicated column disagreed with itself");
        assert_eq!(row[1], row[2]);
        assert_eq!(field(&got, i, 0), latency_of(i).to_string());
    }

    // `Computed` with `Take` on *both* sides of it, over the very columns the
    // computation reads. This is the ordering invariant: if a move ran before
    // the evaluation, `latency * 2` would see an emptied column.
    let got = both_ways(
        cat,
        "SELECT latency, latency * 2, user_id, latency + user_id, user_id \
         FROM events ORDER BY ts LIMIT 6",
    );
    assert_eq!(got.len(), 6);
    for r in 0..got.len() {
        let n = |c: usize| -> i64 { field(&got, r, c).parse().unwrap() };
        assert_eq!(n(1), n(0) * 2, "a computed column did not see its input");
        assert_eq!(n(3), n(0) + n(2));
        assert_eq!(got[r][2], got[r][4], "the duplicated bare column disagreed");
        assert_eq!(n(0), latency_of(r) as i64);
    }

    // The empty projection: `count(*)` keeps a row count and no columns.
    let c = both_ways(cat, "SELECT count() FROM events");
    assert_eq!(scalar(&c, "count"), N.to_string());

    // NULLs ride along with the column they belong to, in both copies.
    let got = both_ways(cat, "SELECT bytes, bytes FROM events ORDER BY ts LIMIT 3");
    for (i, row) in got.iter().enumerate() {
        assert_eq!(row[0], row[1], "row {i}: the copy disagreed with the moved column");
        match bytes_of(i) {
            None => assert_eq!(row[0], "NULL"),
            Some(v) => assert_eq!(row[0], cell(&Value::Int(v))),
        }
    }
    assert_eq!(got[0][0], "NULL", "row 0 is one of the seeded NULLs");
    assert_ne!(got[1][0], "NULL", "and row 1 is not, or this proves nothing");
}

#[test]
fn a_spread_of_query_shapes_returns_the_right_values() {
    let db = db();
    let cat = &db.catalog;

    // Expectations derived from the generator, not from the engine.
    let mut per_country = [0u64; 8];
    let mut sum_bytes = 0i64;
    let (mut n_nonnull, mut n_slow) = (0u64, 0u64);
    for i in 0..N {
        per_country[COUNTRIES.iter().position(|c| *c == country_of(i)).unwrap()] += 1;
        if let Some(v) = bytes_of(i) {
            sum_bytes += v;
            n_nonnull += 1;
        }
        if latency_of(i) > 500 {
            n_slow += 1;
        }
    }

    let one = |sql: &str| scalar(&both_ways(cat, sql), sql);
    assert_eq!(one("SELECT sum(bytes) FROM events"), sum_bytes.to_string());
    assert_eq!(one("SELECT count(bytes) FROM events"), n_nonnull.to_string());
    assert_eq!(one("SELECT count() FROM events WHERE latency > 500"), n_slow.to_string());
    // An identity over a projected column: exercises the whole expression path
    // and can only hold if every row survived the projection intact.
    assert_eq!(one("SELECT sum(latency * 2) - 2 * sum(latency) FROM events"), "0");

    // GROUP BY, ordered so the check does not depend on group order.
    let got = both_ways(cat, "SELECT country, count() FROM events GROUP BY country ORDER BY country");
    let mut want: Vec<(&str, u64)> =
        COUNTRIES.iter().enumerate().map(|(i, c)| (*c, per_country[i])).collect();
    want.sort();
    assert_eq!(got.len(), 8);
    for (r, (c, n)) in want.iter().enumerate() {
        assert_eq!(field(&got, r, 0), *c);
        assert_eq!(field(&got, r, 1), n.to_string());
    }

    // filter -> project -> sort -> limit: the four-operator shape.
    let got = both_ways(
        cat,
        "SELECT latency, country FROM events WHERE latency > 890 \
         ORDER BY latency DESC, country ASC, ts ASC LIMIT 7",
    );
    assert_eq!(got.len(), 7);
    let lat: Vec<i64> = (0..got.len()).map(|r| field(&got, r, 0).parse().unwrap()).collect();
    assert!(lat.windows(2).all(|w| w[0] >= w[1]), "not descending: {lat:?}");
    assert!(lat.iter().all(|&l| l > 890));

    // A join, with a grouped projection over its output.
    let want_join = (0..N).filter(|&i| (latency_of(i) - 10) % 3 == 0).count() as u64;
    assert_eq!(
        one("SELECT count() FROM events e JOIN dim d ON e.latency = d.k"),
        want_join.to_string(),
        "the join lost or duplicated rows"
    );
    let joined = both_ways(
        cat,
        "SELECT d.tag, count() FROM events e JOIN dim d ON e.latency = d.k \
         GROUP BY d.tag ORDER BY d.tag LIMIT 5",
    );
    assert_eq!(joined.len(), 5);

    // DISTINCT, the streaming operator that keys on whole rows.
    assert_eq!(both_ways(cat, "SELECT DISTINCT country FROM events ORDER BY country").len(), 8);

    // A full projection of every column, so a block is moved out wholesale.
    let all = both_ways(cat, "SELECT ts, user_id, country, latency, bytes FROM events ORDER BY ts LIMIT 2");
    assert_eq!(all[0].len(), 5);
    assert_eq!(all[1][2].split('|').nth(1).unwrap(), country_of(1));
    assert_eq!(all[0][4], "NULL");
}

/// `LIMIT` with nothing to order by is the one plan shape that reaches
/// `operators::drain`, whose byte accounting changed. Both the rows it returns
/// and the reservation it hands back are asserted.
#[test]
fn a_keyless_limit_returns_input_order_and_gives_its_budget_back() {
    let db = db();
    let cat = &db.catalog;
    let full = cat.table_by_path("default.events").unwrap().schema().clone();
    let node = ScanNode {
        table: "default.events".into(),
        schema: full.project(&[2]),
        projection: vec![2],
        filters: vec![],
        zone_filters: vec![],
    };
    let no_keys: Vec<SortKey> = vec![];

    let ctx = QueryContext::new();
    let mut op: Box<dyn operators::Operator> = Box::new(operators::sort::Sort::top_k(
        Box::new(operators::scan::Scan::new(&node, cat, &ctx).unwrap()),
        &no_keys,
        5,
        &ctx,
    ));
    let got = cells(&mut op);
    drop(op);
    assert_eq!(got.len(), 5, "a keyless top-k must still stop at k");
    for (i, row) in got.iter().enumerate() {
        assert_eq!(
            row[0].split('|').nth(1).unwrap(),
            country_of(i),
            "keyless order must be input order"
        );
    }
    assert_eq!(ctx.mem.used(), 0, "the drain kept its reservation");

    // The same shape under a budget far too small for the input. The summed
    // accounting must still refuse it -- a cheaper estimate that stopped
    // noticing would turn a bounded query into an unbounded one -- and must
    // hand back everything it charged either way.
    let tight = QueryContext::with_budget(64 << 10);
    let mut op: Box<dyn operators::Operator> = Box::new(operators::sort::Sort::top_k(
        Box::new(operators::scan::Scan::new(&node, cat, &tight).unwrap()),
        &no_keys,
        5,
        &tight,
    ));
    let e = op.next().unwrap_err().to_string();
    drop(op);
    assert!(e.contains("memory budget"), "{e}");
    assert_eq!(tight.mem.used(), 0, "the failed query kept its reservation");
}

/// The slow paths must stay reachable. Each of these bypasses the move.
#[test]
fn the_general_projection_paths_still_run() {
    let s = Schema::new(vec![
        Field::new("a", DataType::Int64),
        Field::new("b", DataType::Int64),
    ])
    .unwrap();
    let rows = vec![
        vec![Value::Int(3), Value::Int(4)],
        vec![Value::Int(5), Value::Int(6)],
    ];
    let col = |i: usize| BoundExpr::Column { index: i, ty: DataType::Int64, name: format!("c{i}") };
    let out2 = Schema::new_unchecked(vec![
        Field::new("x", DataType::Int64),
        Field::new("y", DataType::Int64),
    ]);
    let pctx = QueryContext::new();
    macro_rules! project {
        ($exprs:expr, $sch:expr) => {
            operators::project::Project::new(
                Box::new(operators::values::Values::new(&rows, &s)),
                $exprs,
                $sch,
                &pctx,
            )
        };
    }

    // Literal-only projection: no bare column at all, so nothing can move.
    let exprs = vec![BoundExpr::lit(Value::Int(9)), BoundExpr::lit(Value::str("hi"))];
    let sch = Schema::new_unchecked(vec![
        Field::new("n", DataType::Int64),
        Field::new("s", DataType::String),
    ]);
    let b = project!(&exprs, &sch).next().unwrap().unwrap();
    assert_eq!(b.rows(), 2);
    assert_eq!(b.column(0).value(1), Value::Int(9));
    assert_eq!(b.column(1).value(0).render_plain(), "hi");

    // A projection naming a column the block does not have is an error that
    // names the column: not a panic, and not a silently short block.
    let exprs = vec![col(0), col(9)];
    let e = project!(&exprs, &out2).next().unwrap_err().to_string();
    assert!(e.contains("c9"), "{e}");

    // Computed-first, bare-second, over the same column: the evaluation must
    // not see a moved-out buffer.
    let exprs = vec![
        BoundExpr::Binary {
            left: Box::new(col(0)),
            op: BinaryOp::Plus,
            right: Box::new(col(1)),
            ty: DataType::Int64,
        },
        col(1),
    ];
    let b = project!(&exprs, &out2).next().unwrap().unwrap();
    assert_eq!(b.column(0).as_i64().unwrap(), &[7, 11]);
    assert_eq!(b.column(1).as_i64().unwrap(), &[4, 6]);

    // Every output naming the *same* column: one `Take` and n-1 `Copy`s, and
    // the block still has to come out the right width.
    let exprs = vec![col(1), col(1), col(1)];
    let sch3 = Schema::new_unchecked(vec![
        Field::new("p", DataType::Int64),
        Field::new("q", DataType::Int64),
        Field::new("r", DataType::Int64),
    ]);
    let b = project!(&exprs, &sch3).next().unwrap().unwrap();
    assert_eq!(b.width(), 3);
    for c in 0..3 {
        assert_eq!(b.column(c).as_i64().unwrap(), &[4, 6], "column {c}");
    }
}
