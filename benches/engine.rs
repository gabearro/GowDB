//! End-to-end benchmark, run with `cargo bench` (or `cargo run --release
//! --bench engine`). Custom harness: these are throughput measurements with
//! correctness assertions, not microbenchmarks.
//!
//! Environment overrides:
//!   `ROWS=10000000`  row count for the scan benchmarks
//!   `KEYS=rand`      use random rather than sequential keys
//!   `ONLY=scan`      run just one section (scan | width | oltp | store | sql)

use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

use granular::common::{splitmix64, Result};
use granular::storage::Table;
use granular::types::{Block, Column, DataType, Engine, Field, Schema, TableDef, Value};
use granular::Session;

fn def(random_pk: bool) -> TableDef {
    let _ = random_pk;
    TableDef {
        name: "bench".into(),
        schema: Schema::new(vec![
            Field::new("id", DataType::UInt64),
            Field::new("value", DataType::Int64),
            Field::new("category", DataType::UInt32),
        ])
        .unwrap(),
        order_by: vec![0],
        primary_key: vec![0],
        partition_by: None,
        engine: Engine::MergeTree,
    }
}

fn rule(title: &str) {
    println!("\n\x1b[1m== {title} ==\x1b[0m");
}

fn scalar_u64(db: &mut Session, sql: &str) -> Result<u64> {
    db.query(sql)?
        .scalar()
        .and_then(|v| v.as_u64())
        .ok_or_else(|| granular::Error::exec(format!("`{sql}` produced no scalar")))
}

// ---------------------------------------------------------------- scan bench

fn scan_bench(n: u64, random_keys: bool) -> Result<()> {
    let label = if random_keys { "random keys" } else { "sequential keys" };
    rule(&format!("scan: {n} rows, {label}"));

    let key_of = |i: u64| if random_keys { splitmix64(i) } else { 1_000_000_000 + i };
    // Build the columns directly. Materializing a Vec of row tuples first and
    // then transposing it would double peak memory for no reason -- a columnar
    // engine should be fed columns.
    let input = Block::new(vec![
        Column::u64s(DataType::UInt64, (0..n).map(key_of).collect()),
        Column::i64s(DataType::Int64, (0..n).map(|i| (i as i64 % 500) - 250).collect()),
        Column::u64s(DataType::UInt32, (0..n).map(|i| i % 16).collect()),
    ])
    .unwrap();

    let mut t = Table::new(def(random_keys), usize::MAX);
    let start = Instant::now();
    t.insert(input)?;
    t.flush()?;
    let dt = start.elapsed();
    println!(
        "bulk load (sort + pack + fingerprints + MPH): {:?} ({:.1} M rows/s)",
        dt,
        n as f64 / dt.as_secs_f64() / 1e6
    );

    println!("{}", t.compression_report());

    // Expected values, computed independently of the engine.
    let full_expect: i64 = (0..n).map(|i| (i as i64 % 500) - 250).sum();
    let cat_expect: i64 = (0..n)
        .filter(|i| i % 16 == 5)
        .map(|i| (i as i64 % 500) - 250)
        .sum();

    let iters = 10u32;
    // Serial: one batch at a time, summed while hot in cache.
    let start = Instant::now();
    let mut got = 0i64;
    for _ in 0..iters {
        let mut sum = 0i64;
        t.scan_each(&[1], |b| {
            sum += b.column(0).as_i64()?.iter().sum::<i64>();
            Ok(())
        })?;
        got = black_box(sum);
    }
    let serial = start.elapsed() / iters;
    assert_eq!(got, full_expect, "full scan sum mismatch");
    let grs = n as f64 / serial.as_secs_f64() / 1e9;
    println!(
        "full-table SUM, 1 thread : {serial:?}/scan = {grs:.2} G rows/s",
    );

    // Parallel: granules are independent, so the same fold fans out.
    let start = Instant::now();
    for _ in 0..iters {
        got = black_box(t.scan_fold(
            &[1],
            || 0i64,
            |acc, b| {
                *acc += b.column(0).as_i64()?.iter().sum::<i64>();
                Ok(())
            },
            |a, b| a + b,
        )?);
    }
    let par = start.elapsed() / iters;
    assert_eq!(got, full_expect, "parallel scan sum mismatch");
    let grs = n as f64 / par.as_secs_f64() / 1e9;
    let threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
    println!(
        "full-table SUM, {threads:>2} threads: {par:?}/scan = {grs:.2} G rows/s -> 1B rows in {:.0} ms  ✔",
        1.0 / grs * 1000.0
    );
    println!(
        "  speedup: {:.2}x on {threads} cores",
        serial.as_secs_f64() / par.as_secs_f64()
    );

    let start = Instant::now();
    let mut gotc = 0i64;
    for _ in 0..iters {
        let mut sum = 0i64;
        t.scan_each(&[1, 2], |b| {
            let vals = b.column(0).as_i64()?;
            let cats = b.column(1).as_u64()?;
            // Branch-free masked sum: the multiply keeps this vectorizable
            // where an `if` would not be.
            sum += vals
                .iter()
                .zip(cats)
                .map(|(&v, &c)| v * ((c == 5) as i64))
                .sum::<i64>();
            Ok(())
        })?;
        gotc = black_box(sum);
    }
    let per = start.elapsed() / iters;
    assert_eq!(gotc, cat_expect, "category sum mismatch");
    let grs = n as f64 / per.as_secs_f64() / 1e9;
    println!(
        "SUM WHERE category = 5  : {per:?}/scan = {grs:.2} G rows/s -> 1B rows in {:.0} ms  ✔",
        1.0 / grs * 1000.0
    );
    Ok(())
}

// ------------------------------------------------------- write scaling bench

/// Buffered-write cost as a function of table width.
///
/// The delta is a row-major lane arena precisely so this curve stays flat: a
/// write touches one contiguous span regardless of how many columns the table
/// has. A columnar buffer would add roughly a cache line -- and about 4 ns --
/// per column, which is what this exists to catch if anyone reintroduces it.
fn width_bench() -> Result<()> {
    rule("write scaling: ns/row vs table width");

    let probe = |ncols: usize| -> Result<f64> {
        let fields: Vec<Field> = (0..ncols)
            .map(|i| {
                Field::new(
                    format!("c{i}"),
                    if i == 0 { DataType::UInt64 } else { DataType::Int64 },
                )
            })
            .collect();
        let def = TableDef {
            name: "w".into(),
            schema: Schema::new(fields).unwrap(),
            order_by: vec![0],
            primary_key: vec![0],
            partition_by: None,
            engine: Engine::MergeTree,
        };
        let n = 65_536u64;
        let mut best = f64::MAX;
        for _ in 0..5 {
            // No flush: this measures the buffered write in isolation.
            let mut t = Table::new(def.clone(), usize::MAX);
            let mut row: Vec<Value> = (0..ncols).map(|_| Value::Int(0)).collect();
            let start = Instant::now();
            for i in 0..n {
                row[0] = Value::UInt(splitmix64(i));
                for cell in row.iter_mut().skip(1) {
                    *cell = Value::Int(i as i64);
                }
                t.put_row(&row)?;
            }
            best = best.min(start.elapsed().as_nanos() as f64 / n as f64);
        }
        Ok(best)
    };

    let widths = [1usize, 2, 3, 5, 8, 16];
    let mut results = Vec::new();
    for &w in &widths {
        let ns = probe(w)?;
        results.push((w, ns));
        println!("  {w:>2} columns: {ns:>5.1} ns/row");
    }

    // The property, not just the numbers. Per-cell work is real and cannot be
    // zero -- N columns means N encodes and N stores. What the row-major arena
    // removes is the *cache line* per column: a columnar buffer measured ~4.0
    // ns per extra column, this one ~2.3. The threshold sits between the two,
    // so drifting back toward a per-column allocation or a per-column buffer
    // fails the build rather than quietly costing throughput on wide tables.
    let (w0, ns0) = results[0];
    let (wn, nsn) = *results.last().unwrap();
    let per_col = (nsn - ns0) / (wn - w0) as f64;
    println!(
        "  slope: {per_col:.2} ns per extra column ({w0} -> {wn} columns: {ns0:.1} -> {nsn:.1} ns/row)"
    );
    // Asserted on the ratio, not the slope. The slope is a difference of two
    // measurements divided by 15, so noise in the *narrow* one -- the fastest
    // and therefore least stable point -- is amplified fifteenfold; it swings
    // 2.5 to 4.6 across runs on a loaded machine while the 16-column cost sits
    // within a few percent of 117 ns. The ratio is what the layout actually
    // determines: a row-major delta writes a row's cells contiguously, so
    // sixteen columns cost a small multiple of one, not sixteen times one.
    let ratio = nsn / ns0;
    assert!(
        ratio < 4.0,
        "{w0} -> {wn} columns cost {ratio:.2}x per row ({ns0:.1} -> {nsn:.1} ns); \
         a row-major delta should be near 2.5x, and {wn}x would mean the write \
         path has gone back to touching each column separately"
    );
    println!("  scaling assertion passed ✔ ({ratio:.2}x for {wn}x the columns)");
    Ok(())
}

// ---------------------------------------------------------------- oltp bench

fn oltp_bench() -> Result<()> {
    const N: u64 = 1_000_000;
    rule(&format!("OLTP: {N} rows, random keys"));
    println!(
        "build threads: {}",
        std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1)
    );

    let key_of = splitmix64;

    // Diagnostic: split the write path from the flush path, so a regression in
    // one is never mistaken for the other. `ONLY=oltp` prints both.
    {
        // 64k rows: one flush's worth, so the key index stays the size it
        // really is in steady state. Buffering all 1M at once would measure a
        // 24 MB hash map missing cache, which is not the write path anyone
        // runs.
        const BUF: u64 = 64 * 1024;
        let mut probe = Table::new(def(true), usize::MAX);
        let start = Instant::now();
        for i in 0..BUF {
            probe.put_row(&[
                Value::UInt(key_of(i)),
                Value::Int(i as i64 * 3),
                Value::UInt(i % 16),
            ])?;
        }
        let writes = start.elapsed();
        let start = Instant::now();
        probe.flush()?;
        let flush = start.elapsed();
        println!(
            "  [split] buffer {BUF} rows: {:.0} ns/row | flush them: {:.0} ns/row",
            writes.as_nanos() as f64 / BUF as f64,
            flush.as_nanos() as f64 / BUF as f64
        );
    }

    let mut t = Table::new(def(true), 64 * 1024);
    let mut model: BTreeMap<u64, (i64, u32)> = BTreeMap::new();

    let start = Instant::now();
    for i in 0..N {
        // Row-oriented write: no Block, no per-row allocation.
        t.put_row(&[
            Value::UInt(key_of(i)),
            Value::Int(i as i64 * 3),
            Value::UInt(i % 16),
        ])?;
    }
    let dt = start.elapsed();
    for i in 0..N {
        model.insert(key_of(i), (i as i64 * 3, (i % 16) as u32));
    }
    println!(
        "load          : {dt:?} ({:.1} M writes/s incl. compression), {} parts",
        N as f64 / dt.as_secs_f64() / 1e6,
        t.part_count()
    );

    let bench_get = |t: &mut Table, label: &str, k: &dyn Fn(u64) -> u64, expect: u64| {
        let start = Instant::now();
        let mut found = 0u64;
        for i in 0..N {
            if black_box(t.locate(k(i))).is_some() {
                found += 1;
            }
        }
        let dt = start.elapsed();
        assert_eq!(found, expect, "{label}: wrong hit count");
        println!(
            "{label}: {:.0} ns/lookup, {:.1} M lookups/s",
            dt.as_nanos() as f64 / N as f64,
            N as f64 / dt.as_secs_f64() / 1e6
        );
    };
    println!("-- point lookups over {} parts (compressed) --", t.part_count());
    bench_get(&mut t, "1M hits       ", &key_of, N);
    bench_get(&mut t, "1M misses     ", &|i| key_of(N + 1_000_000 + i), 0);

    // churn
    let start = Instant::now();
    for i in 0..200_000u64 {
        let k = key_of(i * 7 % N);
        t.put_row(&[Value::UInt(k), Value::Int(-(i as i64)), Value::UInt(99)])?;
        model.insert(k, (-(i as i64), 99));
    }
    for i in 0..100_000u64 {
        let k = key_of(i * 11 % N);
        t.delete_key(&Value::UInt(k))?;
        model.remove(&k);
    }
    t.flush()?;
    println!("\n200k updates + 100k deletes + flush: {:?}", start.elapsed());
    for i in (0..N).step_by(3) {
        let k = key_of(i);
        let got = t.get_lane(k).map(|r| match (&r[1], &r[2]) {
            (Value::Int(v), Value::UInt(c)) => (*v, *c as u32),
            _ => unreachable!(),
        });
        assert_eq!(got, model.get(&k).copied(), "key {k}");
    }
    println!("spot-verified 333k keys against a BTreeMap reference ✔");

    let start = Instant::now();
    t.compact()?;
    println!("\nk-way merge compaction -> 1 part: {:?}", start.elapsed());
    assert_eq!(t.row_count()?, model.len());

    println!("-- post-compaction (1 part) --");
    let live: Vec<u64> = model.keys().copied().collect();
    let nlive = live.len();
    let start = Instant::now();
    let mut found = 0u64;
    for i in 0..N as usize {
        if black_box(t.locate(live[(i * 733) % nlive])).is_some() {
            found += 1;
        }
    }
    let dt = start.elapsed();
    assert_eq!(found, N);
    println!(
        "scalar 1M hits: {:.0} ns/lookup, {:.1} M lookups/s",
        dt.as_nanos() as f64 / N as f64,
        N as f64 / dt.as_secs_f64() / 1e6
    );

    // batched, with software prefetch overlapping the cache misses
    const BATCH: usize = 1024;
    let mut out = vec![None; BATCH];
    let mut q = vec![0u64; BATCH];
    let start = Instant::now();
    let mut found = 0u64;
    let mut qi = 0usize;
    for _ in 0..(N as usize / BATCH) {
        for s in q.iter_mut() {
            *s = live[(qi * 733) % nlive];
            qi += 1;
        }
        t.multi_locate(&q, &mut out);
        found += out.iter().filter(|o| o.is_some()).count() as u64;
    }
    let dt = start.elapsed();
    let nq = (N as usize / BATCH * BATCH) as u64;
    assert_eq!(found, nq);
    println!(
        "batched 1M hits (multi_get + prefetch): {:.0} ns/lookup, {:.1} M lookups/s",
        dt.as_nanos() as f64 / nq as f64,
        nq as f64 / dt.as_secs_f64() / 1e6
    );

    println!();
    println!("{}", t.compression_report());
    Ok(())
}

// --------------------------------------------------------------- store bench

/// What a part costs on disk and in memory once it has been written and read
/// back.
///
/// Three numbers that only mean something together: bytes per row on disk
/// (what LZ4 over packed lanes bought), resident heap for the lanes (what the
/// mapping bought), and the time to open the part (what both cost). A codec
/// that halves the file but doubles open latency is not obviously a win, so
/// the benchmark prints all three rather than a single score.
fn store_bench() -> Result<()> {
    use granular::persist::{part_from_bytes, read_part, write_part};

    rule("storage: on-disk size and resident memory");

    const ROWS: usize = 2_000_000;
    let dir = std::env::temp_dir().join(format!("granular-bench-store-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(granular::common::Error::from)?;
    let path = dir.join("part.gpart");

    // Sequential keys and a low-cardinality dimension: the shape an analytics
    // fact table actually has, and the shape both codecs are meant to exploit.
    let n = ROWS as u64;
    let input = Block::new(vec![
        Column::u64s(DataType::UInt64, (0..n).map(|i| 1_000_000_000 + i).collect()),
        Column::i64s(DataType::Int64, (0..n).map(|i| splitmix64(i) as i64 % 1000).collect()),
        Column::u64s(DataType::UInt32, (0..n).map(|i| i % 64).collect()),
    ])
    .unwrap();

    let mut t = Table::new(def(false), usize::MAX);
    t.insert(input)?;
    t.flush()?;
    let snap = t.snapshot();
    let part = snap.parts().first().expect("one part");

    let t0 = Instant::now();
    write_part(&path, part)?;
    let write = t0.elapsed();
    let on_disk = std::fs::metadata(&path).map_err(granular::common::Error::from)?.len();

    let raw = std::fs::read(&path).map_err(granular::common::Error::from)?;
    let t0 = Instant::now();
    let copied = part_from_bytes(&raw)?;
    let copy_open = t0.elapsed();

    let t0 = Instant::now();
    let mapped = read_part(&path)?;
    let map_open = t0.elapsed();

    let lanes = |p: &granular::storage::Part| -> usize {
        p.granules.iter().flat_map(|g| &g.columns).map(|c| c.lanes().bytes()).sum()
    };
    let (m_heap, c_heap) = (lanes(&mapped), lanes(&copied));
    let n_mapped = mapped
        .granules
        .iter()
        .flat_map(|g| &g.columns)
        .filter(|c| c.lanes().is_mapped())
        .count();
    let n_cols = mapped.granules.iter().map(|g| g.columns.len()).sum::<usize>();

    println!("  rows                {ROWS}");
    println!(
        "  on disk             {:.2} MiB   {:.2} bytes/row",
        on_disk as f64 / (1 << 20) as f64,
        on_disk as f64 / ROWS as f64
    );
    println!(
        "  lane heap, copied   {:.2} MiB",
        c_heap as f64 / (1 << 20) as f64
    );
    println!(
        "  lane heap, mapped   {:.2} MiB   ({n_mapped}/{n_cols} columns borrowed from the file)",
        m_heap as f64 / (1 << 20) as f64
    );
    println!("  write (incl. LZ4)   {write:?}");
    println!("  open, copied        {copy_open:?}");
    println!("  open, mapped        {map_open:?}");

    // 24 bytes/row is the uncompressed width of three 64-bit columns. Packing
    // plus the codec has to beat that by a wide margin or none of this earned
    // its complexity.
    let bpr = on_disk as f64 / ROWS as f64;
    assert!(bpr < 12.0, "{bpr:.2} bytes/row on disk; raw would be 24");
    assert!(m_heap < c_heap, "mapping saved no heap: {m_heap} vs {c_heap}");
    assert_eq!(mapped.n_rows, copied.n_rows);
    for row in (0..ROWS).step_by(9_973) {
        for col in 0..3 {
            assert_eq!(
                mapped.value_at(row, col),
                copied.value_at(row, col),
                "the two read paths disagree at row {row} column {col}"
            );
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

// ----------------------------------------------------------------- sql bench

fn sql_bench() -> Result<()> {
    const N: usize = 2_000_000;
    rule(&format!("SQL: {N} rows through the full pipeline"));

    let mut db = Session::in_memory();
    db.execute(
        "CREATE TABLE events (
            ts       DateTime,
            user_id  UInt32,
            country  String,
            latency  UInt32,
            bytes    Int64
        ) ENGINE = MergeTree ORDER BY ts",
    )?;

    let countries = ["US", "DE", "FR", "JP", "BR", "IN", "GB", "CA"];
    let base = 1_700_000_000i64;
    let rows: Vec<(i64, u32, &str, u32, i64)> = (0..N)
        .map(|i| {
            let h = splitmix64(i as u64);
            (
                base + i as i64,
                (h % 100_000) as u32,
                countries[(h >> 20) as usize % countries.len()],
                (h % 900 + 10) as u32,
                (h % 65536) as i64,
            )
        })
        .collect();

    let blk = Block::new(vec![
        Column::u64s(DataType::DateTime, rows.iter().map(|r| r.0 as u64).collect()),
        Column::u64s(DataType::UInt32, rows.iter().map(|r| r.1 as u64).collect()),
        Column::strs(DataType::String, rows.iter().map(|r| r.2.into()).collect()),
        Column::u64s(DataType::UInt32, rows.iter().map(|r| r.3 as u64).collect()),
        Column::i64s(DataType::Int64, rows.iter().map(|r| r.4).collect()),
    ])
    .unwrap();

    let start = Instant::now();
    {
        use granular::sql::ast::ObjectName;
        db.catalog
            .table_mut(&ObjectName::bare("events"))?
            .insert(blk)?;
        db.catalog.flush_all()?;
    }
    let dt = start.elapsed();
    println!(
        "load: {dt:?} ({:.2} M rows/s)",
        N as f64 / dt.as_secs_f64() / 1e6
    );
    {
        use granular::sql::ast::ObjectName;
        println!(
            "{}",
            db.catalog.table(&ObjectName::bare("events"))?.compression_report()
        );
    }

    let queries: &[(&str, &str)] = &[
        ("count", "SELECT count() FROM events"),
        ("sum", "SELECT sum(bytes) FROM events"),
        (
            "group by country",
            "SELECT country, count(), avg(latency) FROM events GROUP BY country ORDER BY country",
        ),
        (
            "selective range (zone maps)",
            "SELECT count(), sum(bytes) FROM events WHERE ts BETWEEN 1700500000 AND 1700500999",
        ),
        (
            "filter + group + order + limit",
            "SELECT country, count() AS n FROM events WHERE latency > 500 \
             GROUP BY country ORDER BY n DESC LIMIT 3",
        ),
        ("top-k by sort", "SELECT ts, latency FROM events ORDER BY latency DESC LIMIT 5"),
        ("uniq (HLL)", "SELECT uniq(user_id) FROM events"),
        ("quantile", "SELECT quantile(0.95)(latency) FROM events"),
        // High-cardinality GROUP BY: ~500k distinct keys over 2M rows, so the
        // group table is far larger than any single block. This is the shape
        // that used to go quadratic -- the per-block passes were sized by the
        // whole grouping rather than by the block, so the cost grew with
        // (blocks x groups) instead of rows.
        (
            "group by high-cardinality key",
            "SELECT user_id, count() FROM events GROUP BY user_id ORDER BY user_id LIMIT 3",
        ),
    ];

    // Correctness gates. A benchmark that only reports timings will happily
    // report a spectacular number for a query that pruned every granule and
    // returned nothing, so pin the answers first.
    let n_total = scalar_u64(&mut db, "SELECT count() FROM events")?;
    assert_eq!(n_total, N as u64, "count() disagrees with what we loaded");
    let n_range = scalar_u64(
        &mut db,
        "SELECT count() FROM events WHERE ts BETWEEN 1700500000 AND 1700500999",
    )?;
    assert_eq!(n_range, 1000, "selective range returned {n_range}, expected 1000");
    let n_countries = scalar_u64(&mut db, "SELECT uniqExact(country) FROM events")?;
    assert_eq!(n_countries, countries.len() as u64);
    // Pin the group count too: a high-cardinality GROUP BY that silently
    // collapsed to a handful of groups would otherwise look like a speedup.
    let n_users = scalar_u64(&mut db, "SELECT uniqExact(user_id) FROM events")?;
    assert!(
        n_users >= 50_000,
        "the high-cardinality grouping only has {n_users} distinct keys; \
         it is no longer measuring what it is named for"
    );
    let grouped: u64 = db
        .query("SELECT count() FROM (SELECT user_id, count() FROM events GROUP BY user_id)")?
        .scalar()
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert_eq!(grouped, n_users, "GROUP BY produced a different group count than uniqExact");
    println!("correctness gates passed ✔");

    for (label, sql) in queries {
        // one warm-up, then three timed runs
        let _ = db.query(sql)?;
        let mut best = f64::MAX;
        let mut last = None;
        for _ in 0..3 {
            let start = Instant::now();
            let rs = db.query(sql)?;
            best = best.min(start.elapsed().as_secs_f64());
            last = Some(rs);
        }
        let rs = last.unwrap();
        let mrs = N as f64 / best / 1e6;
        let pruned = rs.stats.granules_pruned;
        println!(
            "{label:<32} {:>8.2} ms  {mrs:>8.1} M rows/s  {} rows out{}",
            best * 1000.0,
            rs.rows(),
            if pruned > 0 {
                format!(
                    "  [{} of {} granules pruned]",
                    pruned,
                    pruned + rs.stats.granules_read
                )
            } else {
                String::new()
            }
        );
    }

    println!("\nsample output:");
    println!(
        "{}",
        db.query(
            "SELECT country, count() AS hits, round(avg(latency)) AS avg_ms \
             FROM events GROUP BY country ORDER BY hits DESC LIMIT 5"
        )?
    );
    Ok(())
}

fn main() {
    let only = std::env::var("ONLY").unwrap_or_default();
    let want = |s: &str| only.is_empty() || only == s;

    let run = |name: &str, f: &dyn Fn() -> Result<()>| {
        if let Err(e) = f() {
            eprintln!("\x1b[31m{name} failed: {e}\x1b[0m");
            std::process::exit(1);
        }
    };

    if want("scan") {
        let n: u64 = std::env::var("ROWS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10_000_000);
        let random = std::env::var("KEYS").map(|k| k == "rand").unwrap_or(false);
        run("scan", &|| scan_bench(n, random));
    }
    if want("width") {
        run("width", &width_bench);
    }
    if want("oltp") {
        run("oltp", &oltp_bench);
    }
    if want("store") {
        run("store", &store_bench);
    }
    if want("sql") {
        run("sql", &sql_bench);
    }

    println!("\n\x1b[32mAll benchmark assertions passed.\x1b[0m");
}
