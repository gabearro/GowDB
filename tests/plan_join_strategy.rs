//! The join has two algorithms now, so the thing worth testing is that they
//! are the *same join*.
//!
//! Every case here runs one join twice -- once as a hash join, once with the
//! primary-key access path attached -- and compares the rows. That is the only
//! assertion that catches the failure mode an index join actually has: not a
//! crash, but a quietly smaller answer, because a key nobody probed for is a
//! row nobody padded.
//!
//! Each case also asserts *which* strategy ran, through [`Join::strategy`] and
//! through the scan counters. A strategy that silently never fires would
//! otherwise pass every row comparison in this file by running the hash join
//! twice, which is exactly how a planner rule comes to be dead code.
//!
//! The plans are built here rather than through SQL because the operator's
//! contract is with the planner, not with the parser: [`choose`] names a
//! strategy from side sizes and index availability, and this file is the proof
//! that the operator honours both answers. It also makes the shapes that matter
//! -- an empty side, a NULL key, a key with no lane -- three lines each instead
//! of a `CREATE TABLE`.

use granular::catalog::Catalog;
use granular::exec::operators::join::{choose, Join, JoinIndexSide, JoinStrategy, SideFacts};
use granular::exec::operators::scan::Scan;
use granular::exec::operators::values::Values;
use granular::exec::operators::{Operator, QueryContext, ScanStats};
use granular::planner::logical::ScanNode;
use granular::sql::ast::{JoinOp, ObjectName};
use granular::types::{Block, Column, DataType, Field, Schema, Value};
use granular::{Result, Session};

// ------------------------------------------------------------------ fixtures

/// Filler rows the fixtures add beyond the keys a test names.
///
/// Not decoration: the crossover is 128 rows of keyed table per probe, so on a
/// three-row table the operator is *right* to read the whole thing and every
/// positive case here would fall back. `FILL` puts the table on the side of
/// the crossover the tests are about. The filler keys sit far above anything
/// probed, so they only ever show up as unmatched rows of the keyed side --
/// which is exactly what the `FULL` join case needs to count.
const FILL: u64 = 4096;

/// A keyed table `t(k UInt64, w Int64)` holding `keys` plus `fill` filler rows,
/// the scan node a planner would have lowered `SELECT k, w FROM t` to, and the
/// table's row count.
fn keyed(name: &str, keys: &[u64], fill: u64) -> Result<(Session, ScanNode, usize)> {
    let mut db = Session::in_memory();
    db.execute(&format!(
        "CREATE TABLE {name} (k UInt64, w Int64) ENGINE = MergeTree ORDER BY k PRIMARY KEY k"
    ))?;
    let all: Vec<u64> = keys.iter().copied().chain(1_000_000..1_000_000 + fill).collect();
    if !all.is_empty() {
        let blk = Block::new(vec![
            Column::u64s(DataType::UInt64, all.clone()),
            Column::i64s(DataType::Int64, all.iter().map(|&k| k as i64 * 10).collect()),
        ])?;
        db.catalog.table_mut(&ObjectName::bare(name))?.insert(blk)?;
    }
    // Parts *are* the table by the time a read runs -- `Session` flushes before
    // every statement -- and the index path reads parts only. Skipping this
    // would leave rows in the delta that the fetch cannot see and the hash join
    // can, which would look like an index-join bug and be a test bug.
    db.catalog.flush_all()?;
    let schema = Schema::new(vec![
        Field::new("k", DataType::UInt64),
        Field::new("w", DataType::Int64),
    ])?;
    let node = ScanNode {
        table: name.into(),
        projection: vec![0, 1],
        schema,
        filters: vec![],
        zone_filters: vec![],
    };
    Ok((db, node, all.len()))
}

/// The un-keyed side: `(j, u)` rows fed straight in.
fn probe_schema() -> Schema {
    Schema::new(vec![
        Field::new("j", DataType::Nullable(Box::new(DataType::UInt64))),
        Field::new("u", DataType::Int64),
    ])
    .unwrap()
}

fn prows(keys: &[Option<u64>]) -> Vec<Vec<Value>> {
    keys.iter()
        .enumerate()
        .map(|(i, k)| vec![k.map_or(Value::Null, Value::UInt), Value::Int(i as i64)])
        .collect()
}

/// Run one join. `index` attaches the primary-key path; `probe_right` puts the
/// un-keyed rows on the right, which is what a `RIGHT JOIN` needs.
fn run(
    cat: &Catalog,
    node: &ScanNode,
    rows: &[Vec<Value>],
    op: JoinOp,
    on: &[(usize, usize)],
    probe_right: bool,
    index: bool,
) -> Result<(Vec<Vec<Value>>, Option<JoinStrategy>, ScanStats)> {
    let ps = probe_schema();
    let vals = || -> Box<dyn Operator + '_> { Box::new(Values::new(rows, &ps)) };
    let ctx = QueryContext::new();
    let scan = || -> Result<Box<dyn Operator + '_>> { Ok(Box::new(Scan::new(node, cat, &ctx)?)) };
    let out = if probe_right {
        node.schema.concat(&ps)
    } else {
        ps.concat(&node.schema)
    };
    let (l, r) = if probe_right { (scan()?, vals()) } else { (vals(), scan()?) };
    let mut j = Join::new(l, r, op, on, None, &out, &ctx);
    if index {
        j = j.with_index(JoinIndexSide { right: !probe_right, node, catalog: cat });
    }
    let mut got = Vec::new();
    while let Some(b) = j.next()? {
        for i in 0..b.rows() {
            got.push((0..b.width()).map(|c| b.column(c).value(i)).collect::<Vec<_>>());
        }
    }
    // Neither strategy promises an order -- the hash join emits in probe order
    // and the index join in fetch order -- so only the multiset is comparable.
    got.sort();
    Ok((got, j.strategy(), j.stats()))
}

/// Both strategies, same join. Returns the rows they agreed on.
#[allow(clippy::too_many_arguments)]
fn both(
    db: &Session,
    node: &ScanNode,
    rows: &[Vec<Value>],
    op: JoinOp,
    on: &[(usize, usize)],
    probe_right: bool,
    want: JoinStrategy,
    what: &str,
) -> Result<Vec<Vec<Value>>> {
    let (hash, hs, hstats) = run(&db.catalog, node, rows, op, on, probe_right, false)?;
    let (idx, is, istats) = run(&db.catalog, node, rows, op, on, probe_right, true)?;
    assert_eq!(hs, Some(JoinStrategy::Hash), "{what}: the reference run was not a hash join");
    assert_eq!(is, Some(want), "{what}: wrong strategy");
    assert_eq!(idx, hash, "{what}: the two strategies disagree");
    if want != JoinStrategy::Hash {
        // The counters are the second, independent witness: a `strategy()` that
        // lied would still have to read the whole keyed side to lie, and this
        // is what says it did not.
        assert!(
            istats.granules_read <= hstats.granules_read,
            "{what}: the index join read {} granules, the hash join {}",
            istats.granules_read,
            hstats.granules_read
        );
    }
    Ok(hash)
}

const ON: &[(usize, usize)] = &[(0, 0)];
/// `ON` for the probe-on-the-right layout: left is the keyed table.
const ON_R: &[(usize, usize)] = &[(0, 0)];

// ------------------------------------------------------- the strategies agree

#[test]
fn inner_and_left_joins_agree_under_both_strategies() -> Result<()> {
    // Keys on both sides the other lacks, a duplicate probe key, a NULL key,
    // and a key present only in the table -- every row class an outer join has
    // to decide about, in one probe side.
    let (db, node, _) = keyed("t", &[1, 2, 3, 4], FILL)?;
    let rows = prows(&[Some(2), Some(3), Some(3), None, Some(99)]);
    let want = JoinStrategy::IndexNestedLoop { index_right: true };

    let inner = both(&db, &node, &rows, JoinOp::Inner, ON, false, want, "inner")?;
    assert_eq!(inner.len(), 3, "2, 3 and the duplicate 3");

    let left = both(&db, &node, &rows, JoinOp::Left, ON, false, want, "left")?;
    assert_eq!(left.len(), 5, "every probe row exactly once");
    // The NULL key and the missing key are padded, not dropped.
    assert_eq!(left.iter().filter(|r| r[2] == Value::Null).count(), 2);
    Ok(())
}

#[test]
fn a_right_join_fetches_the_left_side() -> Result<()> {
    // The mirror image: the preserved side is the un-keyed one, so the *left*
    // is the side that may be fetched. A strategy that only ever indexed the
    // right would fall back here and this asserts it does not.
    let (db, node, _) = keyed("t", &[1, 2, 3, 4], FILL)?;
    let rows = prows(&[Some(2), Some(3), Some(3), None, Some(99)]);
    let want = JoinStrategy::IndexNestedLoop { index_right: false };
    let got = both(&db, &node, &rows, JoinOp::Right, ON_R, true, want, "right")?;
    assert_eq!(got.len(), 5, "every right row exactly once");
    assert_eq!(got.iter().filter(|r| r[0] == Value::Null).count(), 2);
    Ok(())
}

#[test]
fn duplicate_keys_on_the_probe_side_join_once_per_pair() -> Result<()> {
    // A `MergeTree` primary key is unique, so the fan-out an index join has to
    // get right is on the *probe* side: three rows with key 2 must produce
    // three output rows, not one. Deduplicating the key list is what makes
    // this a real risk -- three probe rows become one lane.
    let (db, node, _) = keyed("t", &[1, 2, 3], FILL)?;
    let rows = prows(&[Some(2), Some(2), Some(2)]);
    let want = JoinStrategy::IndexNestedLoop { index_right: true };
    let got = both(&db, &node, &rows, JoinOp::Inner, ON, false, want, "fanout")?;
    assert_eq!(got.len(), 3);
    Ok(())
}

#[test]
fn an_empty_side_is_still_a_side() -> Result<()> {
    let (db, node, _) = keyed("t", &[1, 2, 3], FILL)?;
    let want = JoinStrategy::IndexNestedLoop { index_right: true };
    let empty: Vec<Vec<Value>> = Vec::new();
    assert!(both(&db, &node, &empty, JoinOp::Inner, ON, false, want, "empty probe")?.is_empty());
    assert!(both(&db, &node, &empty, JoinOp::Left, ON, false, want, "empty left")?.is_empty());

    // ... and an empty *table*: a left join over it pads the probe row, and the
    // fetch has to come back with a correctly shaped nothing rather than a
    // zero-width block that would shift the padding. One probe row, because
    // one probe beats any scan and is the only thing worth doing to a table
    // whose row count is zero.
    let (db, node, n) = keyed("u", &[], 0)?;
    assert_eq!(n, 0);
    let rows = prows(&[Some(1)]);
    let got = both(&db, &node, &rows, JoinOp::Left, ON, false, want, "empty table")?;
    assert_eq!(got.len(), 1);
    assert!(got.iter().all(|r| r[2] == Value::Null && r[3] == Value::Null));
    Ok(())
}

#[test]
fn only_null_keys_fetch_nothing_and_pad_everything() -> Result<()> {
    let (db, node, _) = keyed("t", &[1, 2, 3], FILL)?;
    let rows = prows(&[None, None]);
    let want = JoinStrategy::IndexNestedLoop { index_right: true };
    assert!(both(&db, &node, &rows, JoinOp::Inner, ON, false, want, "all null")?.is_empty());
    let got = both(&db, &node, &rows, JoinOp::Left, ON, false, want, "all null left")?;
    assert_eq!(got.len(), 2, "NULL = NULL is unknown, so both rows are unmatched");
    Ok(())
}

#[test]
fn the_index_join_never_materializes_the_keyed_side() -> Result<()> {
    // The memory claim, measured rather than asserted in prose. Both sides of
    // a hash join are drained into one block each and charged to the budget;
    // an index join charges the probe side and the handful of rows it fetched.
    // Sampled after the first `next()`, which is when `prepare` has run and
    // both sides are in hand.
    //
    // A big keyed side on purpose: *both* strategies charge a 64 KiB pair
    // buffer (`Vec::with_capacity(BLOCK_SIZE)` of `(u32, u32)`), which is a
    // floor under the ratio and swamps it on a small table. The difference
    // between the two numbers is the keyed side, and that is what has to
    // disappear.
    let (db, node, _) = keyed("t", &[1, 2, 3], 65_536)?;
    let ps = probe_schema();
    let rows = prows(&[Some(2)]);
    let out = ps.concat(&node.schema);
    let charged = |index: bool| -> Result<i64> {
        let ctx = QueryContext::new();
        let mut j = Join::new(
            Box::new(Values::new(&rows, &ps)),
            Box::new(Scan::new(&node, &db.catalog, &ctx)?),
            JoinOp::Inner,
            ON,
            None,
            &out,
            &ctx,
        );
        if index {
            j = j.with_index(JoinIndexSide { right: true, node: &node, catalog: &db.catalog });
        }
        j.next()?;
        let used = ctx.mem.used();
        drop(j);
        assert_eq!(ctx.mem.used(), 0, "the join kept its reservation");
        Ok(used)
    };
    let (hash, idx) = (charged(false)?, charged(true)?);
    assert!(
        idx * 8 < hash,
        "the index join charged {idx} B against the hash join's {hash} B"
    );
    Ok(())
}

// ------------------------------------------------------------ negative cases

#[test]
fn a_full_join_never_fetches_either_side() -> Result<()> {
    // Both sides are preserved, so a row of the keyed side that nobody probed
    // for still has to be emitted -- and a fetch by key is exactly the thing
    // that cannot produce it. The rows are asserted as well as the strategy:
    // the failure this guards against is 4 rows instead of 6.
    let (db, node, n) = keyed("t", &[1, 2, 3, 4], FILL)?;
    let rows = prows(&[Some(2), Some(99)]);
    let got = both(&db, &node, &rows, JoinOp::Full, ON, false, JoinStrategy::Hash, "full")?;
    assert_eq!(got.len(), n + 1, "every keyed row once, plus the unmatched 99");
    Ok(())
}

#[test]
fn a_cross_join_never_fetches_either_side() -> Result<()> {
    let (db, node, n) = keyed("t", &[1, 2, 3], FILL)?;
    let rows = prows(&[Some(1), Some(2)]);
    let got = both(&db, &node, &rows, JoinOp::Cross, &[], false, JoinStrategy::Hash, "cross")?;
    assert_eq!(got.len(), n * 2, "the full product");
    Ok(())
}

#[test]
fn a_left_join_refuses_to_fetch_the_side_it_must_preserve() -> Result<()> {
    // The index offered for the *left* of a `LEFT JOIN`: fetching it would
    // drop every left row whose key the right side does not mention, which is
    // the whole point of the join type. `probe_right` puts the un-keyed rows on
    // the right, so `with_index` names the left.
    let (db, node, n) = keyed("t", &[1, 2, 3, 4], FILL)?;
    let rows = prows(&[Some(2)]);
    let got = both(&db, &node, &rows, JoinOp::Left, ON_R, true, JoinStrategy::Hash, "left-keyed")?;
    assert_eq!(got.len(), n, "every row of the keyed left side");
    Ok(())
}

#[test]
fn a_probe_side_larger_than_the_crossover_falls_back() -> Result<()> {
    // The negative case the crossover exists for: 4096 probes against a 3-row
    // table is 4096 index lookups to avoid reading three rows. The planner's
    // `max_rows` is an upper bound and can be loose, so the operator counts the
    // rows it actually drained and changes its mind -- and the rows it already
    // pulled have to reach the hash join, or the answer is short.
    let (db, node, _) = keyed("t", &[1, 2, 3], 0)?;
    let rows = prows(&(0..4096u64).map(|i| Some(i % 3 + 1)).collect::<Vec<_>>());
    let got = both(&db, &node, &rows, JoinOp::Inner, ON, false, JoinStrategy::Hash, "too many")?;
    assert_eq!(got.len(), 4096, "the rows drained before the fallback were not handed back");
    Ok(())
}

#[test]
fn an_unkeyed_table_falls_back() -> Result<()> {
    // `ORDER BY` without `PRIMARY KEY` is a sorted table with no MPH to probe.
    let mut db = Session::in_memory();
    db.execute("CREATE TABLE u (k UInt64, w Int64) ENGINE = MergeTree ORDER BY tuple()")?;
    let blk = Block::new(vec![
        Column::u64s(DataType::UInt64, vec![1, 2, 3]),
        Column::i64s(DataType::Int64, vec![10, 20, 30]),
    ])?;
    db.catalog.table_mut(&ObjectName::bare("u"))?.insert(blk)?;
    db.catalog.flush_all()?;
    let schema = Schema::new(vec![
        Field::new("k", DataType::UInt64),
        Field::new("w", DataType::Int64),
    ])?;
    let node = ScanNode {
        table: "u".into(),
        projection: vec![0, 1],
        schema,
        filters: vec![],
        zone_filters: vec![],
    };
    let rows = prows(&[Some(2)]);
    let got = both(&db, &node, &rows, JoinOp::Inner, ON, false, JoinStrategy::Hash, "unkeyed")?;
    assert_eq!(got.len(), 1);
    Ok(())
}

#[test]
fn a_key_that_has_no_lane_falls_back_rather_than_dropping_rows() -> Result<()> {
    // `Int(-1)` names no lane of a `UInt64` key. Skipping it would be right
    // here and wrong for a `Float64` key holding NaN, which `Value`'s equality
    // does match -- so the operator refuses to tell the two apart and falls
    // back. The assertion is that the *answer* is unchanged either way -- and
    // the table is filled so that the size gate passes and the lane gate is
    // what actually fires.
    let (db, node, _) = keyed("t", &[1, 2, 3], FILL)?;
    let ps = Schema::new(vec![
        Field::new("j", DataType::Int64),
        Field::new("u", DataType::Int64),
    ])?;
    let rows = vec![
        vec![Value::Int(2), Value::Int(0)],
        vec![Value::Int(-1), Value::Int(1)],
    ];
    let out = ps.concat(&node.schema);
    let ctx = QueryContext::new();
    let mut j = Join::new(
        Box::new(Values::new(&rows, &ps)),
        Box::new(Scan::new(&node, &db.catalog, &ctx)?),
        JoinOp::Left,
        ON,
        None,
        &out,
        &ctx,
    )
    .with_index(JoinIndexSide { right: true, node: &node, catalog: &db.catalog });
    let mut n = 0;
    while let Some(b) = j.next()? {
        n += b.rows();
    }
    assert_eq!(n, 2, "a left join emits both rows whichever strategy ran");
    assert_eq!(j.strategy(), Some(JoinStrategy::Hash), "-1 has no UInt64 lane");
    Ok(())
}

#[test]
fn a_composite_key_falls_back() -> Result<()> {
    // Two equi-keys, one single-column index. `choose` refuses before the
    // operator looks at a table, and the join still has to answer.
    let (db, node, _) = keyed("t", &[1, 2, 3], FILL)?;
    let rows = prows(&[Some(2), Some(3)]);
    // `j = k AND u = w`: row (2, 0) has u=0 and w=20, so nothing matches.
    let on: &[(usize, usize)] = &[(0, 0), (1, 1)];
    let got = both(&db, &node, &rows, JoinOp::Inner, on, false, JoinStrategy::Hash, "composite")?;
    assert!(got.is_empty());
    assert_eq!(choose(JoinOp::Inner, on, small(), keyed_facts()), JoinStrategy::Hash);
    Ok(())
}

// --------------------------------------------------------- the pure function

fn small() -> SideFacts {
    SideFacts { max_rows: Some(1), keyed: false }
}
fn keyed_facts() -> SideFacts {
    SideFacts { max_rows: Some(1_000_000), keyed: true }
}

#[test]
fn choose_names_a_strategy_from_facts_alone() {
    let big = SideFacts { max_rows: Some(1_000_000), keyed: false };
    let unknown = SideFacts { max_rows: None, keyed: false };
    let inl = |index_right| JoinStrategy::IndexNestedLoop { index_right };

    assert_eq!(choose(JoinOp::Inner, ON, small(), keyed_facts()), inl(true));
    assert_eq!(choose(JoinOp::Inner, ON, keyed_facts(), small()), inl(false));
    assert_eq!(choose(JoinOp::Left, ON, small(), keyed_facts()), inl(true));
    assert_eq!(choose(JoinOp::Right, ON, keyed_facts(), small()), inl(false));

    // The preserved side is never the fetched one.
    assert_eq!(choose(JoinOp::Left, ON, keyed_facts(), small()), JoinStrategy::Hash);
    assert_eq!(choose(JoinOp::Right, ON, small(), keyed_facts()), JoinStrategy::Hash);
    assert_eq!(choose(JoinOp::Full, ON, small(), keyed_facts()), JoinStrategy::Hash);
    assert_eq!(choose(JoinOp::Cross, &[], small(), keyed_facts()), JoinStrategy::Hash);

    // A probe side nobody can bound is not a small one.
    assert_eq!(choose(JoinOp::Inner, ON, unknown, keyed_facts()), JoinStrategy::Hash);
    assert_eq!(choose(JoinOp::Inner, ON, big, keyed_facts()), JoinStrategy::Hash);
    // Neither side keyed.
    assert_eq!(choose(JoinOp::Inner, ON, small(), big), JoinStrategy::Hash);

    // The crossover itself: 16 rows of keyed table per probe, and one probe
    // always wins whatever the ratio says.
    let n = |n| SideFacts { max_rows: Some(n), keyed: false };
    let m = |m| SideFacts { max_rows: Some(m), keyed: true };
    assert_eq!(choose(JoinOp::Inner, ON, n(1000), m(16_000)), inl(true));
    assert_eq!(choose(JoinOp::Inner, ON, n(1001), m(16_000)), JoinStrategy::Hash);
    assert_eq!(choose(JoinOp::Inner, ON, n(1), m(3)), inl(true), "one probe beats any scan");
    // ... and it cannot overflow into firing on an absurd probe side.
    assert_eq!(choose(JoinOp::Inner, ON, n(usize::MAX), m(16_000)), JoinStrategy::Hash);
}
