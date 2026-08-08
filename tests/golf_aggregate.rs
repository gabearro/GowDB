//! `GROUP BY` fast paths against the general path, on the same rows.
//!
//! `exec::operators::aggregate` resolves a single integer key straight off the
//! block's lane and memoizes a single string key by the address of its decoded
//! `Arc`. Neither is allowed to be *nearly* right: a fast path that puts one
//! key in two groups, or two keys in one, is a wrong answer that arrives
//! sooner. So every case here runs the query twice -- once as shipped, once
//! with [`GENERAL_KEYS`] forcing the per-row `Value` path -- and asserts the
//! two result sets are identical value for value, in the same order.
//!
//! What that covers: cardinalities from one group to a million, integer and
//! string and dictionary-encoded and nullable and composite keys, and the
//! aggregates whose answers depend on *which* rows reached them (`any`,
//! `groupArray`, `argMin`) as well as the ones that only count.
//!
//! A profiling driver rides along, off unless `GOLF_PROF` is set:
//!
//! ```text
//! GOLF_PROF=1 cargo test --release --test golf_aggregate -- --nocapture prof
//! ```

use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::time::Instant;

use granular::exec::operators::aggregate::GENERAL_KEYS;
use granular::types::{Block, Column, ColumnBuilder, DataType, Value};
use granular::Session;

/// `GENERAL_KEYS` is process-global and the test harness is threaded, so the
/// two halves of every comparison have to be taken under one lock.
static SWITCH: Mutex<()> = Mutex::new(());

fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Every result row, in the order the query produced it.
fn rows_of(db: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let rs = db.query(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in rs.blocks {
        for i in 0..b.rows() {
            out.push((0..b.width()).map(|c| b.column(c).value(i)).collect());
        }
    }
    out
}

/// Run `sql` both ways and return the (identical) answer.
fn both_ways(db: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let _lk = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
    let fast = rows_of(db, sql);
    GENERAL_KEYS.store(true, Ordering::Relaxed);
    let general = rows_of(db, sql);
    GENERAL_KEYS.store(false, Ordering::Relaxed);
    assert_eq!(
        fast.len(),
        general.len(),
        "fast and general paths disagree on row count for `{sql}`"
    );
    for (i, (f, g)) in fast.iter().zip(&general).enumerate() {
        assert_eq!(f, g, "row {i} of `{sql}` differs between the fast and general paths");
    }
    fast
}

// --------------------------------------------------------------- the fixture

/// `n` rows of `(i Int64, u UInt64, sml UInt32, d Date, t DateTime, s String,
/// lc LowCardinality(String), nn Nullable(Int64), v Int64)`.
///
/// `groups` sets the key cardinality: `i`, `u`, `sml` and `s` all carry the
/// same `groups` distinct values, so one fixture serves every key type at
/// every cardinality.
fn fixture(n: usize, groups: u64) -> Session {
    let mut db = Session::in_memory();
    db.execute(
        "CREATE TABLE t (
            i    Int64,
            u    UInt64,
            sml  UInt32,
            d    Date,
            t    DateTime,
            s    String,
            lc   LowCardinality(String),
            nn   Nullable(Int64),
            v    Int64
         ) ENGINE = MergeTree ORDER BY v",
    )
    .unwrap();

    let mut i = Vec::with_capacity(n);
    let mut u = Vec::with_capacity(n);
    let mut sml = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    let mut ts = Vec::with_capacity(n);
    let mut s: Vec<std::sync::Arc<str>> = Vec::with_capacity(n);
    let mut lc: Vec<std::sync::Arc<str>> = Vec::with_capacity(n);
    let mut nb = ColumnBuilder::with_capacity(DataType::Int64.to_nullable(), n);
    let mut v = Vec::with_capacity(n);
    // One `Arc` per distinct value, cloned per row: what a granule decode
    // produces, and what the address memo is built to exploit.
    let pool: Vec<std::sync::Arc<str>> =
        (0..groups).map(|k| std::sync::Arc::from(format!("k{k}").as_str())).collect();
    for r in 0..n {
        let h = splitmix(r as u64);
        let k = h % groups;
        // Negative keys too: `Value::Int`'s lane is a sign-extended `i64`, and
        // a fast path that dropped the sign would only show up here.
        i.push(if k % 3 == 0 { -(k as i64) } else { k as i64 });
        u.push(k);
        sml.push(k % 65_536);
        d.push(k % 30_000);
        ts.push(1_700_000_000u64 + k);
        s.push(pool[k as usize].clone());
        lc.push(pool[k as usize].clone());
        if r % 7 == 0 {
            nb.push_null();
        } else {
            nb.push_value(&Value::Int(k as i64)).unwrap();
        }
        v.push((h >> 20) as i64 % 1000);
    }
    let blk = Block::new(vec![
        Column::i64s(DataType::Int64, i),
        Column::u64s(DataType::UInt64, u),
        Column::u64s(DataType::UInt32, sml),
        Column::u64s(DataType::Date, d),
        Column::u64s(DataType::DateTime, ts),
        Column::strs(DataType::String, s),
        Column::strs(DataType::String, lc),
        nb.finish(),
        Column::i64s(DataType::Int64, v),
    ])
    .unwrap();
    {
        use granular::sql::ast::ObjectName;
        db.catalog.table_mut(&ObjectName::bare("t")).unwrap().insert(blk).unwrap();
        db.catalog.flush_all().unwrap();
    }
    db
}

/// Every single-column key the fixture offers.
const KEYS: &[&str] = &["i", "u", "sml", "d", "t", "s", "lc", "nn"];

// ------------------------------------------------------------ the assertions

#[test]
fn every_key_type_groups_the_same_both_ways() {
    for &(n, groups) in &[(1_000usize, 1u64), (20_000, 8), (20_000, 1_000)] {
        let mut db = fixture(n, groups);
        for k in KEYS {
            let got = both_ways(
                &mut db,
                &format!("SELECT {k}, count(), sum(v), min(v), max(v) FROM t GROUP BY {k} ORDER BY {k}"),
            );
            // A grouping that silently collapsed would still agree with
            // itself, so pin the shape as well as the agreement.
            let want = if *k == "nn" { groups as usize + 1 } else { groups as usize };
            assert_eq!(got.len(), want.min(n), "GROUP BY {k} produced {} groups", got.len());
        }
    }
}

#[test]
fn a_hundred_thousand_groups_agree() {
    let mut db = fixture(400_000, 100_000);
    for k in ["i", "u", "s"] {
        let got = both_ways(
            &mut db,
            &format!("SELECT {k}, count() FROM t GROUP BY {k} ORDER BY {k} LIMIT 5"),
        );
        assert_eq!(got.len(), 5);
    }
    // 400k draws over 100k keys leaves ~1.8% of them undrawn; what matters is
    // that the grouping did not collapse, not the exact coupon-collector tail.
    let n = both_ways(&mut db, "SELECT count() FROM (SELECT u FROM t GROUP BY u)");
    assert!(matches!(n[0][0], Value::UInt(k) if k > 95_000), "{:?}", n[0][0]);
}

#[test]
fn a_million_groups_agree() {
    // More groups than rows per block by two orders of magnitude, so the table
    // outgrows every cache and the probe is doing what it does at scale.
    let mut db = fixture(1_000_000, 1_000_000);
    let got = both_ways(&mut db, "SELECT u, count() FROM t GROUP BY u ORDER BY u LIMIT 4");
    assert_eq!(got.len(), 4);
    let n = both_ways(&mut db, "SELECT count() FROM (SELECT u, count() FROM t GROUP BY u)");
    // ~632k distinct keys out of 1M draws with replacement.
    assert!(matches!(n[0][0], Value::UInt(k) if k > 500_000), "{:?}", n[0][0]);
}

#[test]
fn order_sensitive_aggregates_see_the_same_rows_in_the_same_order() {
    // `any`, `anyLast`, `argMin` and `groupArray` are defined against feed
    // order, so they are the aggregates a fast path that reordered or
    // re-bucketed rows would break -- and the ones that would still look right
    // under `count()`.
    let mut db = fixture(20_000, 64);
    for k in KEYS {
        both_ways(
            &mut db,
            &format!(
                "SELECT {k}, any(v), anyLast(v), argMin(v, v), argMax(v, v), \
                        groupArray(v), uniq(v), uniqExact(v), \
                        quantile(0.9)(v), quantileExact(0.5)(v), avg(v) \
                 FROM t GROUP BY {k} ORDER BY {k}"
            ),
        );
    }
}

#[test]
fn distinct_and_if_variants_agree() {
    let mut db = fixture(20_000, 32);
    for k in ["i", "u", "s", "nn"] {
        both_ways(
            &mut db,
            &format!(
                "SELECT {k}, count(DISTINCT v), sum(DISTINCT v), avg(DISTINCT v), \
                        uniqExact(DISTINCT v), \
                        countIf(v > 500), sumIf(v, v > 500), avgIf(v, v > 500), \
                        maxIf(v, v > 500), groupArrayIf(v, v > 900) \
                 FROM t GROUP BY {k} ORDER BY {k}"
            ),
        );
    }
}

#[test]
fn composite_and_expression_keys_agree() {
    let mut db = fixture(20_000, 100);
    for g in [
        "i, s",
        "s, i",
        "u, sml, d",
        "nn, s",
        "i + 1",
        "abs(i)",
        "toString(u)",
        "i, i",
    ] {
        both_ways(
            &mut db,
            &format!("SELECT {g}, count(), sum(v) FROM t GROUP BY {g} ORDER BY {g}"),
        );
    }
}

#[test]
fn keys_at_the_edges_of_the_lane_agree() {
    // `Value::hash` puts a `UInt` past `i64::MAX` on its *float* branch, so
    // those keys must not take the lane path -- and must still group with the
    // ones that do. Negative `Int64`s, `u64::MAX` and zero all in one table.
    let mut db = Session::in_memory();
    db.execute("CREATE TABLE e (u UInt64, i Int64, v Int64) ENGINE = MergeTree ORDER BY v").unwrap();
    let edges: Vec<u64> = vec![
        0,
        1,
        i64::MAX as u64,
        i64::MAX as u64 + 1,
        u64::MAX,
        u64::MAX - 1,
        1 << 63,
        (1 << 63) + 7,
    ];
    let n = 4_000;
    let mut u = Vec::with_capacity(n);
    let mut i = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);
    for r in 0..n {
        let e = edges[r % edges.len()];
        u.push(e);
        i.push(e as i64);
        v.push((r % 97) as i64);
    }
    let blk = Block::new(vec![
        Column::u64s(DataType::UInt64, u),
        Column::i64s(DataType::Int64, i),
        Column::i64s(DataType::Int64, v),
    ])
    .unwrap();
    {
        use granular::sql::ast::ObjectName;
        db.catalog.table_mut(&ObjectName::bare("e")).unwrap().insert(blk).unwrap();
        db.catalog.flush_all().unwrap();
    }
    // Not `edges.len()`: `Value`'s own equality folds the two lanes past
    // `2^63` that share an `f64`, and folding them is what the general path
    // does -- the point here is that the lane path folds them the same way.
    let got = both_ways(&mut db, "SELECT u, count(), sum(v) FROM e GROUP BY u ORDER BY u");
    assert!(got.len() >= 5, "{} groups over {} edge keys", got.len(), edges.len());
    let got = both_ways(&mut db, "SELECT i, count(), sum(v) FROM e GROUP BY i ORDER BY i");
    assert_eq!(got.len(), 7, "the signed reading of the same lanes has 7 distinct values");
}

#[test]
fn a_string_key_that_never_repeats_still_agrees() {
    // The address memo gives up on a column whose blocks hold more distinct
    // strings than it can. That is a per-query latch, so the block *after* it
    // fires has to keep answering correctly.
    let n = 60_000;
    let mut db = Session::in_memory();
    db.execute("CREATE TABLE u (s String, v Int64) ENGINE = MergeTree ORDER BY v").unwrap();
    let s: Vec<std::sync::Arc<str>> =
        (0..n).map(|r| std::sync::Arc::from(format!("u{r:07}").as_str())).collect();
    let v: Vec<i64> = (0..n).map(|r| (r % 13) as i64).collect();
    let blk = Block::new(vec![
        Column::strs(DataType::String, s),
        Column::i64s(DataType::Int64, v),
    ])
    .unwrap();
    {
        use granular::sql::ast::ObjectName;
        db.catalog.table_mut(&ObjectName::bare("u")).unwrap().insert(blk).unwrap();
        db.catalog.flush_all().unwrap();
    }
    let got = both_ways(&mut db, "SELECT count() FROM (SELECT s FROM u GROUP BY s)");
    assert_eq!(got[0][0], Value::UInt(n as u64));
    both_ways(&mut db, "SELECT s, count(), sum(v) FROM u GROUP BY s ORDER BY s LIMIT 20");
}

#[test]
fn an_empty_and_a_one_row_relation_agree() {
    let mut db = fixture(1, 1);
    both_ways(&mut db, "SELECT i, count() FROM t GROUP BY i");
    both_ways(&mut db, "SELECT count(), sum(v) FROM t");
    both_ways(&mut db, "SELECT i, count() FROM t WHERE v < -1 GROUP BY i");
    let got = both_ways(&mut db, "SELECT count() FROM t WHERE v < -1");
    assert_eq!(got[0][0], Value::UInt(0));
}

#[test]
fn a_spilling_group_by_agrees_with_the_general_path() {
    // Under a budget the table cannot meet, the fast path has to hand the
    // *frozen* loop the same groups -- which is the one place a `find` runs
    // without an insert behind it.
    let mut db = fixture(200_000, 50_000);
    db.execute("SET max_memory_usage = 33554432").unwrap();
    both_ways(&mut db, "SELECT u, count(), sum(v) FROM t GROUP BY u ORDER BY u LIMIT 50");
    both_ways(&mut db, "SELECT s, count(), sum(v) FROM t GROUP BY s ORDER BY s LIMIT 50");
}

// ------------------------------------------------------------------ profiling

/// Interleaved A/B of the shipped path against `GENERAL_KEYS`, plus a long
/// enough loop to hang a sampler off. Off unless `GOLF_PROF` is set.
#[test]
fn prof() {
    let Ok(spec) = std::env::var("GOLF_PROF") else { return };
    let rounds: usize = std::env::var("GOLF_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(7);
    let n: usize = std::env::var("GOLF_ROWS").ok().and_then(|s| s.parse().ok()).unwrap_or(2_000_000);
    let groups: u64 =
        std::env::var("GOLF_GROUPS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let mut db = fixture(n, groups);
    let sqls: Vec<String> = spec
        .split(';')
        .filter(|s| !s.trim().is_empty() && *s != "1")
        .map(|s| s.to_string())
        .collect();
    let sqls = if sqls.is_empty() {
        vec![
            "SELECT s, count(), avg(v) FROM t GROUP BY s".to_string(),
            "SELECT i, count() FROM t GROUP BY i".to_string(),
        ]
    } else {
        sqls
    };
    // `GENERAL_KEYS` is the switch that survives; the ones that measured the
    // argument borrowing and the probe prefetch were temporary and were
    // deleted once their numbers were in comments next to the code they
    // justify. Re-measuring either means adding a switch back here.
    let knob: &'static std::sync::atomic::AtomicBool = &GENERAL_KEYS;
    let _lk = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
    for sql in &sqls {
        let _ = db.query(sql).unwrap();
        let (mut fast, mut gen) = (f64::MAX, f64::MAX);
        for _ in 0..rounds {
            for side in [false, true] {
                knob.store(side, Ordering::Relaxed);
                let t = Instant::now();
                let rs = db.query(sql).unwrap();
                let dt = t.elapsed().as_secs_f64();
                std::hint::black_box(rs.blocks.len());
                let slot = if side { &mut gen } else { &mut fast };
                *slot = slot.min(dt);
            }
        }
        knob.store(false, Ordering::Relaxed);
        println!(
            "{n} rows / {groups} groups  new {:8.3} ms  old {:8.3} ms  {:.3}x  {sql}",
            fast * 1e3,
            gen * 1e3,
            gen / fast
        );
    }
}
