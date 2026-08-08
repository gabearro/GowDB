# granular

A hybrid **OLAP + OLTP** database engine in Rust, with a ClickHouse-flavoured
SQL front end. Zero runtime dependencies — every codec, index, hash and file
format is hand-rolled so the storage footprint and the instruction count on the
hot path stay under our control.

```sql
CREATE TABLE events (
    ts      DateTime,
    user_id UInt32,
    country String,
    latency UInt32
) ENGINE = MergeTree ORDER BY ts;

INSERT INTO events VALUES (1700000000, 42, 'US', 120);

SELECT country, count(), quantile(0.95)(latency)
FROM events
WHERE ts BETWEEN 1700000000 AND 1700003600
GROUP BY country
ORDER BY 2 DESC
LIMIT 10;
```

---

## The one idea everything else follows from

Data is stored **frame-of-reference bit-packed, per column, per granule**, and
every access path is designed to read it **without decompressing**.

That single constraint is what makes one engine good at both workloads:

| | how it stays fast on packed data |
|---|---|
| point lookup | verifies the key directly against packed words — one shifted `u128` load |
| range scan | interpolation-searches packed keys; never materializes the key column |
| string predicate | runs on order-preserving dictionary codes; never materializes a string |
| zone maps | fall out of the FOR metadata for free — `base` and `base + mask` bound the granule at zero extra bytes |

A conventional compressed column store has to inflate a block before it can
answer anything, which is why they are good at scans and bad at lookups. Here,
compression is not on the critical path of either.

### What that buys, measured

`cargo bench` prints these with correctness assertions attached — a benchmark
that only reports timings will happily report a spectacular number for a query
that pruned every granule and returned nothing.

Per-column compression on 10M rows:

| column | declared | stored |
|---|---|---|
| monotonic `id` (1024 rows/granule) | 64 bits | 11 bits |
| `value`, small range | 64 bits | 9.5 bits |
| `category`, 16 distinct values | 32 bits | 4.3 bits |
| `country`, 8 distinct strings | 192 bits | 3.8 bits |
| genuinely random `u64` | 64 bits | 55 bits |

The last row matters: frame-of-reference **never inflates**, and it never costs
a decompression step, so it is safe to apply unconditionally. Whole tables land
around 4.4x smaller (clustered) to 1.4x (random keys).

### Scaling out

Three things stack on top of the packed representation, all measured on the
same 14-core machine with `cargo bench`.

**Parallel scans.** `Table::scan_fold` is a work-stealing map-reduce over
granules: workers claim eight at a time from one atomic cursor, each with its
own L1-sized batch buffer, and the per-worker accumulators merge at the end.
On 10M rows a full `SUM` goes from 1.28 to 6.19 G rows/s — 4.8x on 14 cores,
or a billion rows in 162 ms.

It runs on a persistent thread pool rather than `std::thread::scope`, and that
choice is measured rather than assumed. Spawning fourteen threads costs the
same whether the scan is large or small, so on a 10M-row scan the two are
within noise (1.05x, best of six interleaved runs) — but on a 200k-row scan,
the kind an HTAP workload issues constantly, the pool is **2.47x faster** and
won every round.

**Mapped parts.** Parts are `mmap`ed, and packed lanes are read *in place* out
of the mapping instead of being copied to the heap. This is what the v2 format
pays its alignment padding for: frame bodies and word arrays are padded to 8
bytes so the chain from the page-aligned mapping base down to an individual
lane array preserves alignment, and the reader can reinterpret those bytes as
`&[u64]` with a runtime alignment check and a copying fallback.

**LZ4 over packed lanes.** A hand-rolled LZ4 block codec (no dependency),
applied per word array and kept only when it saves at least an eighth.
Bit-packing has already removed the leading zeros, so what LZ4 finds is
whatever structure survived that; on a dense high-entropy column it finds
nothing and the array stays raw and mappable. Zstandard is not implemented and
deliberately not hand-rolled — its entropy stage is a liability to write from
scratch — but the format reserves a codec tag, so wiring the `zstd` crate in
is two match arms and no format change, if the zero-dependency rule is ever
worth trading away.

The two interact, and the tradeoff is deliberate. A compressed array cannot be
read in place — it costs a decompression pass, resident heap, and O(1) random
access into the column. So the point-lookup index is pinned uncompressed: an
index that must be decompressed to be probed is not an index. Point lookups
are unchanged at 79 ns.

A 2M-row, 3-column part:

| | |
|---|---|
| on disk | 4.75 bytes/row (raw would be 24) |
| lane heap, copied | 7.36 MiB |
| lane heap, mapped | **1.73 MiB** — 4.3x less |
| columns read straight from the file | 3909 of 5862 (67%) |
| write, including compression | 75 ms |
| open | ~8 ms either way |

The third of columns that compressed well are the ones that had to be decoded
onto the heap; the rest cost 32 bytes each and live in reclaimable page cache
rather than resident memory. Open time is a wash — the codec spends roughly
what the mapping saves — which is the honest summary: LZ4 buys disk, the
mapping buys RAM, and neither is free.

**Not yet parallel: the SQL executor.** `scan_fold` is a storage-layer
primitive. SQL still runs through a serial pull-based operator tree, so
`SELECT count()` measures 79 M rows/s where the storage layer does 6,000.
Closing that needs an exchange operator with per-thread pipelines and a merge
per stateful operator, which is its own piece of work.

### Versus the pre-restructure engine

The single-file predecessor in [`.attic/`](.attic/) was hardcoded to one fixed
`(u64, i64, u32)` schema. This one is fully generic — arbitrary typed columns,
nulls, strings, SQL — which is exactly the kind of change that usually costs a
large constant factor. It mostly does not, and where it does, here is the
number rather than a shrug.

Both built with the same flags, run interleaved on the same machine, best of
three (the interleaving matters: a loaded laptop varies by 3x, and only an A/B
run in the same conditions is meaningful):

**10M rows, keys arriving in order** — the time-series case:

| | original | this engine | |
|---|---|---|---|
| bulk load | 611 ms | **276 ms** | 2.2x faster |
| full-table `SUM` | **5.83 ms** (1.72 G rows/s) | 6.18 ms (1.62 G rows/s) | 6% slower |
| `SUM WHERE category = 5` | 11.79 ms | **9.59 ms** | 1.23x faster |
| peak RSS | 519 MB | **293 MB** | 43% less |

**10M rows, random key order** — the sort actually runs:

| | original | this engine | |
|---|---|---|---|
| bulk load | **609 ms** | 703 ms | 15% slower |
| full-table `SUM` | **5.94 ms** | 6.47 ms | 9% slower |
| `SUM WHERE category = 5` | 13.6 ms | **11.8 ms** | 1.15x faster |
| peak RSS | 579 MB | 583 MB | parity |

**1M random keys, OLTP mix:**

| | original | this engine | |
|---|---|---|---|
| point lookup, 15 parts | 147 ns | **119 ns** | 1.24x faster |
| point lookup, compacted | 103 ns | **68 ns** | 1.51x faster |
| batched `multi_locate` | **48 ns** | 54 ns | ~parity |
| k-way merge compaction | 80 ms | 80 ms | parity |
| single-row writes | **143 ms** (7.0 M/s) | 152 ms (6.6 M/s) | 6% slower |

Where it wins is where the work was moved off the hot path: reads never
allocate, scans decode straight into an L1-sized buffer, the write buffer holds
rows as pre-encoded lanes so a flush transposes rather than converts, and ingest
skips the sort entirely when data already arrives in key order.

### Write scaling

Buffered writes were the last place this engine trailed. Splitting the write
path from the flush path located it exactly:

| phase, 64k rows | original | before | after |
|---|---|---|---|
| buffer the rows | 28 ns/row | 46 ns/row | 36 ns/row |
| flush them into a part | 68 ns/row | **67 ns/row** | 67 ns/row |

Flush was already at parity — sorting, packing, fingerprints and MPH
construction cost the same. The whole difference was buffering, and measuring
it against table width showed why: a buffer held as one growable column per
table column touches **one cache line per column** on a single-row write.

The delta is now a row-major arena of `u64` lanes with stride `ncols`, so a
write touches one contiguous span whatever the width. `cargo bench` measures
the curve and **asserts on its slope**, so a regression to per-column layout
fails the build:

| columns | 1 | 2 | 3 | 5 | 8 | 16 |
|---|---|---|---|---|---|---|
| columnar buffer | 28 | 32 | 36 | 42 | 57 | ~85 |
| **row-major lanes** | 31 | 34 | 36 | 39 | **45** | **65** |
| | | | | | 1.27x | 1.3x |

Per-column cost fell from ~4.0 ns to ~2.3 ns — the remainder is irreducible
per-cell work (N columns means N encodes and N stores), not layout. The
crossover is around three columns: narrow tables pay ~3 ns for the arena,
everything wider gets steadily faster, and real schemas are wider. End to end
on the three-column OLTP load the gap closed from 13% to **6%**.

Getting there took four attempts, and three of them were wrong — worth
recording, because each looked obviously right beforehand:

* the first row-major version was **slower than columnar at every width**
  (35 ns at one column, slope 6.2) — `resize`-then-index bounds-checks every
  cell and a `&mut self` call per cell forces field reloads;
* `nrows()` derived the row count as `lanes.len() / ncols`, putting an integer
  division on the write path;
* the rewrite silently dropped the single-hash-lookup entry API, and with
  distinct keys *every* write took the two-lookup path.

Two cheaper hypotheses were tried and rejected outright by measurement:
materializing the sorted block for flush-sized batches (70 vs 66 ns/row,
worse), and removing per-column bounds checks in the columnar version
(46 -> 45 ns/row, noise).

**SQL pipeline**, 2M rows, end to end through parse → bind → optimize →
execute:

| query | throughput |
|---|---|
| `count()` | 484 M rows/s |
| `sum(bytes)` | 279 M rows/s |
| `uniq(user_id)` (HyperLogLog) | 214 M rows/s |
| `quantile(0.95)(latency)` | 76 M rows/s |
| `ORDER BY latency DESC LIMIT 5` | 65 M rows/s |
| `GROUP BY country` (2 aggregates) | 44 M rows/s |
| `WHERE ... GROUP BY ... ORDER BY ... LIMIT` | 44 M rows/s |
| selective range over a sorted key | 0.09 ms — 1952 of 1954 granules pruned |

Ingest through the SQL layer runs at 18 M rows/s.

Hash aggregation used to be the outlier at 19 M rows/s, for two reasons that
are worth naming because they are easy to reintroduce: a `HashMap` keyed by an
owned tuple has to be *probed* with an owned tuple, costing a heap allocation
per input row; and bucketing rows per group as `vec![Vec::new(); ngroups]`
allocates a vector per group **per block**. The group table is now
open-addressed over a row-major key arena, probed against a borrowed slice, and
bucketing is a counting sort into two reused buffers. Single-string keys get
their own path that hashes a borrowed `&str` rather than cloning an `Arc` per
row.

---

## Architecture

### Storage layout

```text
  Table          schema + write buffer + a list of parts
   ├─ Delta      row-major lane arena + key index (the OLTP half)
   └─ Part       immutable, sorted, bloom-filtered run
       └─ Granule   1024 rows, independently packed and indexed
           └─ PackedColumn   FOR-packed lanes + optional dictionary + optional nulls
```

### The index stack, cheapest test first

A point lookup passes through four filters before it touches data:

1. **part-level split-block bloom** — 6 bits/key, all eight probes land in one
   64-byte block, so a foreign part is skipped for the cost of a single cache
   miss. Built lazily, only once a second part exists; a freshly compacted
   table stores none at all.
2. **O(1) router** — `(key - base) >> shift` indexes a bucket table that
   brackets the sparse index, leaving a bounded binary search instead of a full
   one.
3. **granule zone map** — `sort_min`/`sort_max`, free from the FOR metadata.
4. **learned rank + fingerprint** — see below.

#### Learned ranks

The point-lookup index does not store rows. Keys inside a granule are sorted,
so a key's row is *predicted* by linear interpolation over `[min, max]`, and
only the small prediction **error** is stored — fused with a 6-bit fingerprint
into one packed record per minimal-perfect-hash slot:

```text
rec = fp6(key) << ebits | (rank - predicted - err_bias)
```

Clustered keys predict near-exactly (0–1 error bits/row); uniform random keys
need about 7. One record load both rejects foreign keys (1/64 false-positive
rate) *and* yields the row. Verification against the packed key column is exact,
so a false positive costs one extra load and never a wrong answer.

#### Why sign-flip instead of zigzag

Every column is stored as `u64` lanes, and the mapping is **order-preserving**
so that lane comparisons are value comparisons. For signed integers that means
`v as u64 ^ (1<<63)`, not zigzag. Zigzag is not monotonic (`zz(0)=0, zz(-1)=1,
zz(1)=2`), so sorted data stops being sorted once packed. Sign-flip is
order-preserving *and* compresses at least as well, because FOR only cares about
`max - min`:

| values | zigzag span | sign-flip span |
|---|---|---|
| `[-1000, 1000]` | 2000 | 2000 |
| `[1000, 1100]` | 200 | **100** |
| `[-100, -50]` | 100 | **50** |

There is no case where zigzag wins, so there is no tradeoff to make.

#### Strings

Every string column is dictionary-encoded per granule, with the dictionary
**sorted**. That single decision gives:

* `code_a < code_b` **iff** `str_a < str_b`, so equality, range predicates,
  `min`/`max`, `ORDER BY` and zone-map pruning all run on packed integers;
* dense small codes, so `PackedU64` squeezes the column to
  `ceil(log2(cardinality))` bits/row.

This is what ClickHouse spells `LowCardinality(String)`, except it is applied
unconditionally and the width collapses on its own when it pays. 1024 rows drawn
from 8 distinct strings cost 3 bits/row plus the 8 strings, once.

### Staying hybrid under interleaving

Being fast at each workload separately is not the same as being both on one
table at once, and the failure mode is structural. A scan flushes the write
buffer before reading, so alternating writes and queries creates **a part per
query**. Left alone that grows without bound: every point lookup probes one
more bloom filter, every scan reads one more set of undersized granules, and
the engine quietly stops being either thing. Measured, before it was fixed:

| queries | 0 | 15 | 30 | 45 | 60 |
|---|---|---|---|---|---|
| parts (unbounded) | 1 | 16 | 31 | 46 | 60 |
| parts (now) | 1 | 9 | 10 | 7 | 11 |

Two things hold it together. Auto-compaction runs inside `flush`, not only in
the write paths, so a scan-triggered flush is bounded like any other. And
compaction merges **only the small parts** rather than rewriting the table:
a flat "merge everything at N parts" policy makes every Nth query pay
`O(total rows)`, which showed up as scan times climbing 0.4 -> 2.6 ms as the
table grew. Merging smallest-first while the running total stays under a
quarter of the table keeps that cost proportional to the churn instead.

Merging an arbitrary subset is sound because each ingest tombstones the keys it
replaces in older parts, so a live key exists in exactly one part and merge
order does not matter. `tests/htap.rs` pins all of this: writes visible to the
next query, bounded parts under interleaving, point lookups still resolving
through whatever parts remain, and last-write-wins surviving partial merges.

### The OLTP/OLAP split

Writes land in a hash-map `Delta` at memory speed, with last-write-wins per
primary key — a hot row rewritten a million times costs one entry, not a
million. Reads take one of two paths:

* **point lookups** consult the delta, then probe parts newest-first;
* **scans flush the delta first**, then read packed granules directly.

Flushing before a scan is deliberate. The alternative — teaching every scan
operator to merge an unsorted hash map — would put a branch and a hash probe in
the innermost loop of every aggregate, which is exactly the loop that has to run
at memory bandwidth. Paying one small part on the first scan after a write burst
is much cheaper, and compaction folds it away.

### Compaction

Parts are sorted and live keys are disjoint across them (each ingest tombstones
what it replaces), so compaction is a k-way heap merge over live-row cursors:
`O(N log P)` with no re-sort.

### Durability

A table's durable state is `parts + log`.

* **Parts** are written by `checkpoint()`: to a temp file, `fsync`'d, then
  `rename`d into place, with the parent directory `fsync`'d afterwards so the
  rename itself cannot be lost. Every part and the catalog carry a checksum, so
  a corrupt file produces an error rather than silently wrong data.
* **The write-ahead log** covers everything since the last checkpoint. Each
  `INSERT`/`DELETE`/`UPDATE` appends a framed, checksummed record and `fsync`s
  it *before* the write is acknowledged, so dropping the process without
  checkpointing loses nothing. `Session::open` replays each log from the
  watermark its last checkpoint recorded. DDL checkpoints immediately, so a
  table can never have a log the catalog does not know about.
* **Torn tails are normal, torn middles are not.** A crash mid-append leaves a
  partial record — or, on filesystems that allocate blocks eagerly, a run of
  zeros. Both are treated as an interrupted write that was never acknowledged:
  replay stops cleanly and `open` truncates back to the last intact record so
  appends resume correctly. Damage *behind* a record that the log already
  accepted is something no append can do, so that is reported as corruption.

`Session::set_wal_enabled(false)` turns the log off for bulk loading, where an
`fsync` per statement dominates and you would re-run the load after a crash
anyway.

---

## Building and running

> **Note on this checkout:** the directory name contains a colon
> (`OLAP:OLTP database`), which breaks every `cargo` command on macOS —
> cargo puts `<target>/debug/deps` on the colon-separated
> `DYLD_FALLBACK_LIBRARY_PATH`. Either rename the directory (recommended) or
> redirect the target dir:
>
> ```bash
> export CARGO_TARGET_DIR=/tmp/granular-target
> ```

```bash
cargo test              # correctness
cargo bench             # throughput, with assertions
cargo run --release     # SQL shell
```

### The shell

```text
granular                          in-memory REPL
granular --data ./db              persistent REPL
granular -q "SELECT 1"            one shot
granular --data ./db -f setup.sql run a script
echo "SELECT 1" | granular        piped input
```

REPL dot-commands: `.tables`, `.schema TABLE`, `.stats TABLE` (compression and
index footprint), `.help`, `.quit`.

### As a library

```rust
use granular::Session;

let mut db = Session::in_memory();
db.execute("CREATE TABLE t (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id")?;
db.execute("INSERT INTO t VALUES (1, 10), (2, 20)")?;
let rs = db.query("SELECT sum(v) FROM t")?;
println!("{rs}");
# Ok::<(), granular::Error>(())
```

`Session::open(dir)` gives the same API backed by disk; call `checkpoint()` to
persist.

---

## Benchmark knobs

```bash
ROWS=50000000 cargo bench      # scan benchmark row count
KEYS=rand     cargo bench      # random instead of sequential keys
ONLY=sql      cargo bench      # just one section: scan | oltp | sql
```

---

## SQL coverage

ClickHouse-*flavoured*, not ClickHouse-complete. What follows is measured, not
aspirational — every "yes" below has a test in `tests/sql.rs`.

### Supported

| area | coverage |
|---|---|
| **DDL** | `CREATE TABLE [IF NOT EXISTS] … ENGINE = … ORDER BY … [PRIMARY KEY] [PARTITION BY]`, `CREATE TABLE … AS SELECT`, `CREATE DATABASE`, `DROP TABLE/DATABASE [IF EXISTS]`, `TRUNCATE`, `ALTER TABLE … ADD/DROP COLUMN`, `USE` |
| **DML** | `INSERT … VALUES` (multi-row, explicit column list), `INSERT … SELECT`, `ALTER TABLE … DELETE WHERE`, `ALTER TABLE … UPDATE … WHERE`, `OPTIMIZE TABLE [FINAL]` |
| **SELECT** | `DISTINCT`, expression projections with `AS`, `*` and `t.*`, `PREWHERE`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY` (multi-key, `ASC`/`DESC`, `NULLS FIRST/LAST`), `LIMIT`/`OFFSET`, ClickHouse's reversed `LIMIT off, n`, `LIMIT n BY (…)` |
| **FROM** | base tables, subqueries, `WITH … AS (…)` CTEs, `FINAL` |
| **JOIN** | `INNER`, `LEFT`, `RIGHT`, `FULL`, `CROSS`, comma joins, `ON` and `USING (…)`; hash join, with NULL padding on outer sides |
| **Set ops** | `UNION ALL`, `UNION DISTINCT` |
| **Subqueries** | uncorrelated scalar `(SELECT …)`, `x [NOT] IN (SELECT …)`, `[NOT] EXISTS (…)` — evaluated once before planning and folded to literals |
| **Types** | `UInt8/16/32/64`, `Int8/16/32/64`, `Float32/64`, `Bool`, `String`, `FixedString(n)`, `Date`, `DateTime`, `Nullable(T)`, `LowCardinality(T)` |
| **Expressions** | full precedence table, `IS [NOT] NULL`, `[NOT] IN (list)`, `[NOT] BETWEEN`, `[NOT] LIKE`/`ILIKE`, `CASE`, `CAST(x AS T)` and `x::T`, `INTERVAL n UNIT`, tuples |
| **Functions** | 118 scalar (math, strings, dates, nulls, conditionals, hashing) |
| **Aggregates** | `count`, `sum`, `avg`, `min`, `max`, `any`, `anyLast`, `argMin`, `argMax`, `uniq` (HyperLogLog), `uniqExact`, `quantile`/`quantileExact`/`median`, `varPop`, `varSamp`, `stddevPop`, `stddevSamp`, `groupArray`, plus every `-If` combinator (`sumIf`, `countIf`, …) and `count(DISTINCT …)` |
| **Introspection** | `SHOW TABLES`, `SHOW DATABASES`, `SHOW CREATE TABLE`, `DESCRIBE`, `EXPLAIN [PLAN\|AST]` |

### Not supported

Each of these fails with a specific `NOT_IMPLEMENTED` message naming the
feature — none of them silently do something else.

| feature | note |
|---|---|
| `EXCEPT` / `INTERSECT` | parsed, then rejected; only `UNION` is implemented |
| correlated subqueries | rejected explicitly (uncorrelated ones work) |
| window functions | not parsed |
| arrays | no `Array(T)` type. `groupArray` returns a joined `String`, `splitByChar` returns the first field |
| `Decimal` | rejected at type-parse time |
| `GROUP BY … WITH TOTALS/ROLLUP/CUBE` | `WITH TOTALS` parses and is rejected; `ROLLUP`/`CUBE` are not parsed |
| regex (`match`, `extract`) | would need a regex engine; the crate has no dependencies |
| `SummingMergeTree` | rejected — see the deviations below |
| materialized views, `ARRAY JOIN`, `SAMPLE`, `TTL`, table functions, distributed/replicated engines | not implemented |

### Deliberate deviations from ClickHouse

These are behaviour differences you would notice, so they are called out rather
than buried:

1. **`MergeTree` is replacing.** Inserting a row whose primary key already
   exists *replaces* it; ClickHouse's `MergeTree` would keep both. This is what
   makes the engine usable for OLTP — point updates and deletes fall out of it —
   and it is why `ReplacingMergeTree` is a synonym here.
   `SummingMergeTree` is rejected outright rather than silently behaving as a
   replacing engine, which would return wrong sums.
2. **Division by zero yields `NULL`**, in both the executor and the constant
   folder. ClickHouse raises for `/` and returns 0 for `intDiv`.
3. **`Date` starts at the epoch.** It is an unsigned day count (as in
   ClickHouse), so pre-1970 dates are not representable; truncating a pre-epoch
   `DateTime` to a `Date` clamps to 1970-01-01. `DateTime` itself is signed and
   does reach back before 1970.
4. **`cityHash64` is not CityHash** — it is this engine's internal mixer, so
   values will not match a real ClickHouse.
5. **`LowCardinality(T)` is a no-op hint.** Every string column is already
   dictionary-encoded per granule, so it and plain `String` have identical
   storage.
6. **`now()` / `today()` are evaluated once per block**, and `rand()` is a
   deterministic splitmix over a process-global counter.
7. **`FINAL` is accepted and is a genuine no-op.** Since `MergeTree` here is
   already replacing, there are never un-collapsed duplicate keys for `FINAL`
   to merge — it is redundant rather than ignored.

## Layout

```text
src/
  common/     hashing, bitsets, lane codec, errors, constants
  encoding/   bitpack.rs (FOR), dict.rs (order-preserving string dictionary)
  index/      mph.rs (CHD minimal perfect hash), filter.rs (split-block bloom)
  types/      datatype.rs, value.rs, schema.rs, block.rs (vectorized batches)
  sort.rs     LSD radix sort over order-preserving lanes
  storage/    column.rs, granule.rs, part.rs, delta.rs, table.rs
  persist/    format.rs, writer.rs, reader.rs, wal.rs, store.rs
  sql/        lexer.rs, ast.rs, parser.rs
  planner/    logical.rs, binder.rs, optimizer.rs
  exec/       expr.rs, functions/, operators/
  catalog.rs  databases and tables
  session.rs  the public API
  main.rs     the shell
benches/engine.rs
tests/
```

## License

Dual-licensed under [Apache License 2.0](LICENSE-APACHE) or
[MIT](LICENSE-MIT), at your option — the Rust ecosystem convention.

Apache-2.0 is there for the explicit patent grant, which is worth more than
usual for this project: the engine hand-rolls its own frame-of-reference bit
packing, CHD minimal perfect hashing, split-block bloom filters, learned rank
indexes and an LZ4 codec. MIT is there because it is shorter and imposes no
`NOTICE` obligation. Take whichever suits you; you need not say which.
