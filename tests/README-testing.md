# Testing granular

Most of the test suite checks the engine against **its own author's** reading of
SQL. One file does not.

| file                    | what it is                                             |
|-------------------------|--------------------------------------------------------|
| `tests/sql.rs`          | end-to-end acceptance tests: text in, values out        |
| `tests/htap.rs`         | mixed read/write workloads over the delta and parts     |
| `tests/persistence.rs`  | crash/reopen, WAL replay, catalog durability            |
| `tests/robustness.rs`   | malformed input, truncation, resource limits            |
| `tests/differential.rs` | **randomized differential testing against `sqlite3`**   |

Everything below is about the last one. It is the only file in the repository
whose notion of "correct" comes from outside the repository.

Every cargo command in this repository has to redirect the target directory,
because the checkout path contains a colon and cargo's dyld path cannot cope:

```sh
cd "/Users/gabriel/RustroverProjects/OLAP:OLTP database"
CARGO_TARGET_DIR=/tmp/gr-target cargo test
```

---

## Why `tests/differential.rs` exists

The other files were written by the same person who wrote the engine. That
makes them a good regression net and a poor correctness argument: where the
implementation misunderstands SQL, the test written from the same understanding
agrees with it, and a coverage report will happily call the misunderstood lines
`FULLY COVERED`. The bug that motivated this harness was found that way — a
five-minute manual diff against `sqlite3` showed that
`SELECT * FROM a JOIN b USING (id)` emitted the join column twice and
`SELECT id FROM a JOIN b USING (id)` did not bind at all, inside lines the
coverage report was perfectly happy with.

`tests/differential.rs` replaces that circularity with an **independent oracle**.
It generates a random schema, random rows and a random query, runs the same
semantics through granular and through the `sqlite3` CLI, and compares. Neither
engine gets a vote on what is correct. They only get to disagree.

### Running it

```sh
# the default: 400 cases, fixed seed, ~2s
CARGO_TARGET_DIR=/tmp/gr-target cargo test --test differential

# a soak
GRANULAR_DIFF_CASES=50000 GRANULAR_DIFF_SEED=$RANDOM \
  CARGO_TARGET_DIR=/tmp/gr-target cargo test --test differential differential_against_sqlite -- --nocapture
```

| variable                   | effect                                                       |
|----------------------------|--------------------------------------------------------------|
| `GRANULAR_DIFF_CASES`      | number of cases (default 400)                                |
| `GRANULAR_DIFF_SEED`       | starting seed, **decimal**; case *n* uses `seed + n`          |
| `GRANULAR_DIFF_VERBOSE`    | print progress every 50 cases                                |
| `GRANULAR_DIFF_NO_SHRINK`  | print the case exactly as generated, unshrunk                |
| `GRANULAR_DIFF_NO_BATCH`   | one `sqlite3` process per case (4x slower; see below)        |
| `GRANULAR_DIFF_SQLITE`     | use this `sqlite3` binary instead of probing                 |

**Without `sqlite3` the tests skip, loudly, and never fail.** CI on a box with no
`sqlite3` stays green and says so on stderr. That path is not taken on faith —
point the override at nothing and watch it happen:

```sh
GRANULAR_DIFF_SQLITE=/definitely/not/here \
  CARGO_TARGET_DIR=/tmp/gr-target cargo test --test differential -- --nocapture
# 14 passed, with "SKIP: differential tests need the `sqlite3` CLI ..." on stderr
```

Every run prints what it actually generated, so a clean run is evidence rather
than a claim:

```
differential: 400 cases against sqlite3, 0 rejected by both engines, 0 mismatches
  coverage: 203 joins (107 USING), 49 SELECT *, 149 aggregate, 50 DISTINCT,
            82 LIMIT, 33 UNION, 88 with scalar calls
  storage:  130181 rows loaded, widest table 8209, 48 tables past GRANULE_SIZE,
            7 past BLOCK_SIZE, 203 multi-INSERT, 112 OPTIMIZEd
```

If any of those ever reads 0, the generator has silently stopped covering
something and "0 mismatches" means nothing. Add a counter whenever you add a
construct — every line of that report exists because the alternative is trusting
a number with no denominator.

### What it generates

* **Schemas** over the dialect intersection: `Int64`, `Float64`, `String` and
  their `Nullable` variants. Both tables carry `id` and `k` so `USING` joins are
  always well-formed; the rest are per-table prefixed so an unqualified name is
  never ambiguous.
* **Sort keys** in all three shapes granular distinguishes: `ORDER BY id`
  (single integer, so *fast-PK*), `ORDER BY (id, k)` and `ORDER BY tuple()`.
* **Rows** including NULLs, negatives, zero, `-0.0`, empty string, embedded
  quotes, LIKE metacharacters as data, duplicates, and values that collide under
  the sort key.
* **Volume**: most cases are under eight rows so failures shrink fast, but ~8%
  cross `GRANULE_SIZE` (1024) and ~3% cross `BLOCK_SIZE` (8192). Nothing about a
  zone-map skip, a granule-boundary search or a short final block is reachable
  with eight rows.
* **Physical layout**: rows are split across several `INSERT`s about half the
  time, so the delta and one or more sealed parts are both live at query time,
  and ~20% of tables get `OPTIMIZE TABLE ... FINAL`. sqlite cannot tell the
  difference, which is exactly what makes it a useful oracle for it.
* **Queries**: projections (arithmetic, `CASE`, `CAST`, shared scalar
  functions), `WHERE` with `AND`/`OR`/`NOT`/`IS NULL`/comparisons/`BETWEEN`/
  `IN`/`LIKE`, `GROUP BY`, `HAVING`, `ORDER BY` with explicit
  `NULLS FIRST`/`NULLS LAST`, `LIMIT`/`OFFSET`, `DISTINCT`, `UNION`/`UNION ALL`,
  all five join types with both `ON` and `USING`, and `count`/`sum`/`avg`/
  `min`/`max` — the first three also with `DISTINCT`. (`min(DISTINCT x)` and
  `max(DISTINCT x)` are rejected by granular, correctly: `DISTINCT` cannot
  change their answer.)

### How the comparison is made fair

* **Types are recovered, not guessed.** sqlite3 runs in `.mode quote`, not
  `.mode tabs` + `.nullvalue NULL`. TSV cannot distinguish `NULL` from the
  string `'NULL'`, nor the integer `12` from the text `'12'`, and both
  distinctions matter — the second is how a wrong *result type* would slip past.
  Quote mode gives bare `NULL`, bare digits for INTEGER, a `.`/`e` for REAL, and
  `'…'` with `''` escaping for TEXT.
* **Numbers compare numerically across Int and Real**, because the type of a
  numeric aggregate is not part of the intersection (SQLite's `sum` over an
  INTEGER column is INTEGER). `-0.0 == 0.0` falls out of that and is correct.
* **Row order.** A query without `ORDER BY` has no defined order in *either*
  engine, so those results are compared as multisets. When there is an
  `ORDER BY`, the generator always emits a **total** order — a random prefix
  followed by every remaining output ordinal — so `LIMIT`/`OFFSET` slice a
  sequence both engines must agree on, and the comparison is positional.
* **Floats** compare with a relative tolerance of `1e-12`. That is tight on
  purpose; see the note on summation below.
* **Both engines refusing** a query counts as agreement. Chasing those would
  drown the run in parser-message noise.

### Float summation: the divergence that turned out not to exist

The brief for this harness predicted that `sum` over reals would legitimately
differ — SQLite sums naively, granular uses Neumaier compensation — and asked
for an honest tolerance rather than a pretence.

Measured, it is not true of any current SQLite: **3.44 (Nov 2023) adopted
Kahan–Babuška–Neumaier in `sum`, `avg` and `total`**. On the 3.54 shipped with
this machine, `1e16 + 1×100 − 1e16` gives *both* engines the exact answer, 100.
`float_summation_agrees_because_both_engines_compensate` asserts that from both
directions, and states what to do if it is ever run against a pre-3.44 sqlite3
(the divergence is real there, granular is the accurate side, and the assertion
says so instead of widening the tolerance).

Because the harness must stay correct against an older oracle too, the generator
still draws reals from a pool whose sums are exact in binary64 — quarters and
small powers of ten. That is what lets the tolerance stay at `1e-12` and still
be honest, rather than absorbing real bugs in slack.

### Shrinking

On a mismatch the harness minimizes the schema, the rows and the query, then
prints a standalone script for each dialect. Reductions: drop `LIMIT`/`OFFSET`,
`ORDER BY`, `HAVING`, `DISTINCT`, the set-operation branch, `WHERE` (or one side
of a top-level `AND`/`OR`); replace an expression with a subterm; drop a select
item; simplify the join; halve the row set, then drop rows one at a time;
collapse multi-`INSERT` and `OPTIMIZE`; simplify the sort key; drop an
unreferenced column. Each candidate is kept only if it still disagrees, to a
fixpoint, under a budget of 400 candidate evaluations.

The output pastes straight into both shells:

```
seed: 1592644099 (GRANULAR_DIFF_SEED=1592644099 GRANULAR_DIFF_CASES=1)

--- granular (paste into: granular -q '...') ---
CREATE TABLE t0 (id Int64, k Int64) ENGINE = MergeTree ORDER BY id;
INSERT INTO t0 VALUES (0, 2), (0, 1);
SELECT x0.k FROM t0 AS x0 GROUP BY x0.k;
...
--- row count: granular 1, sqlite 2 ---
```

### Two ways this harness lied to itself

Both are worth knowing about, because both produced confident-looking output
that meant nothing, and both are the kind of mistake any differential harness
makes.

**A known divergence leaked back in through a function.** The generator emits
only lowercase text so SQLite's case-insensitive `LIKE` is unobservable — and
then `upper()` was added to the scalar-function menu, which manufactures exactly
the uppercase the invariant depended on not existing. A 60 000-case soak
reported `upper('ab') LIKE '%a%'` as an engine bug; it is not. The invariant to
hold when extending `gen_call` is **nothing case-shifted may reach a `LIKE`**,
and it is enforced by building `LIKE`'s operand separately rather than from the
general text-expression generator.

**A known bug leaked back in through a fallback.** After BUG 5 was found, the
generator started anchoring every predicate atom on a column so the planner
could not constant-fold it. But when the schema happened to have no column of
the chosen type, the code fell back to a *literal* operand — and quietly emitted
`WHERE 2.25 NOT IN (NULL)`. That is twelve mismatches in one soak, every one of
them BUG 5 arriving through a hole in the guard that was supposed to exclude it.
The fix is to fall back to `Int` (every schema has `id` and `k`) instead of to a
literal. **A guard with a fallback is only as good as the fallback.**

### Shrinker drift

A shrinker that minimizes toward "any divergence" can drift to a *different*
bug than the one it started from. That happened here once — a real
`count()`-over-a-filter case shrank into `WHERE 'z'`, which diverges for an
unrelated reason. The fix was to forbid replacing a comparison with one of its
operands (that turns a predicate into a value and hands each engine its own
truthiness rules). `GRANULAR_DIFF_NO_SHRINK=1` shows the unshrunk case when
drift is suspected again.

### Cost

Process spawn, not SQL, dominates. Cases are therefore sent to `sqlite3` in
batches of up to 32 (or 3000 rows, whichever comes first), delimited by a
sentinel row; `.bail on` means a batch that hits an error is re-run one process
per case so the error can be attributed. Measured A/B interleaved, best-of-3,
1200 cases at seed 4242: **1.68s batched vs 6.82s unbatched**, a 4x saving.

Two things follow from that design, and both are tested rather than assumed:

* `batched_and_unbatched_sqlite_agree` runs the same 24 cases both ways and
  requires identical results. An off-by-one in sentinel attribution would not
  crash — it would compare granular's case *n* against sqlite's case *n+1* — so
  it needs a test, not a comment.
* `GRANULAR_DIFF_NO_BATCH=1` keeps the slow path reachable for debugging.

Rendering allocations were the second cost: emitting an `INSERT` through
`Vec<String>` + `join` allocated two `String`s per *cell* (60k for one 8200-row
table, redone per dialect per shrink candidate). Writing into one reused buffer
took **15% off the whole run** at 4000 cases (4.58s vs 5.39s, same protocol) —
which is a lot, given the run is mostly subprocesses and query execution.
`sqlite_path()` is also probed once via `OnceLock` instead of spawning
`sqlite3 -version` on every call.

### Proving the harness itself works

A harness that finds nothing is more likely broken than the engine is perfect.
Three tests exist purely to make "0 mismatches" mean something:

* `comparator_catches_injected_wrong_answers` — feeds the comparator a dropped
  row, a perturbed float, `NULL`-rendered-as-0, `''`-vs-`NULL`, `1`-vs-`'1'`, a
  missing column and a reordered result, and asserts each is caught. It also
  asserts the two things that must **not** be flagged: `1` vs `1.0`, and a
  reordering of a query that has no `ORDER BY`.
* `harness_detects_a_corrupted_engine_answer` — runs a real query through both
  engines end to end, confirms they agree, then corrupts granular's answer and
  confirms the same code path objects.
* `sqlite_driver_recovers_types_exactly` — pins the quote-mode parser against
  every ambiguous shape, including the literal text `'NULL'`.

### Soak results

Run at the time this harness was written, after each finding below was excluded
or pinned. Different seeds, so these are independent samples, not repeats.

| seed        | cases  | rows loaded | mismatches |
|-------------|--------|-------------|------------|
| 987654321   | 30 000 |  9 815 248  | 0          |
| 31337000    | 50 000 | 17 386 725  | 0          |
| 778899      | 30 000 | 10 282 095  | 0          |
| 555000111   | 40 000 | 13 687 297  | 0          |
| 90210777    | 60 000 | 20 503 237  | 0          |
| 1234500     | 60 000 | 20 121 103  | 0          |

Roughly 330 000 cases and 110 million rows in total, against five distinct bugs.

Four of the five were found within the **first 200 cases** of the first run that
could reach them — BUG 1 accounted for all twelve mismatches in the very first
36 cases the harness ever ran. BUG 5 is the exception and the instructive one:
it needed 60 000 cases, because reaching it takes a `CASE` with no `ELSE` inside
a `BETWEEN` inside a `WHERE`, which only the widened predicate grammar produces.

So: findings come fast and then stop. When a soak comes back clean, the next
move is to **widen the grammar, not to raise the case count** — every increment
of grammar in this file paid off within a run or two, and no increment of case
count ever did on its own.

---

## Known divergences (pinned by tests)

These are **dialect differences, not bugs**. Most are excluded from the
generator so they cannot mask real findings, and each is pinned by a test that
fails the day the difference stops existing — so the exclusion gets re-argued
or deleted instead of rotting.

"Most", not "all": #11 is absorbed by the *comparator* instead, and that is the
better pattern where it applies. Excluding a construct costs the coverage of
everything it would have reached; teaching the comparison what the two engines
actually promise costs nothing and keeps generating the construct.

| # | construct | granular | sqlite3 |
|---|-----------|----------|---------|
| ~~1~~ | ~~`1 IN (2, NULL)`~~ | **withdrawn — it is BUG 5** | |
| 2 | `7 / 2` | `3.5` (always-float) | `3` (integer division) |
| 3 | `'ABC' LIKE 'a%'` | `false` (case-sensitive) | `1` (ASCII case-insensitive) |
| 4 | `CAST(1.0 AS <text>)` | `'1'` | `'1.0'` |
| 5 | `sum` over reals | compensated | compensated since 3.44 — **no divergence** |
| 6 | `CAST((1=1) AS <text>)` | `'true'` (real Bool type) | `'1'` (no Bool type) |
| 7 | `INTERSECT ALL` / `EXCEPT ALL` | works | not parsed (3.54) — **direction inverted**; the plain forms are now generated |
| 8 | `round(2.5)` | `2` (half to even) | `3.0` (half away from zero) |
| 9 | `concat(NULL,'b')` | `NULL` (propagates) | `'b'` (skips NULLs) |
| 10 | `GROUPS` frames, `RANGE <n> PRECEDING`, `EXCLUDE`, `FILTER` | parse error | supported |
| 11 | `1.0 / 3.0` | `0.333333` (`Decimal64(6)`) | `0.333333333333333` (binary64) — **not generator-excluded**, see below |

Row 11 covers products as well as quotients: multiplication caps at
`MUL_MAX_SCALE` (6, never below the wider operand's own scale) and rounds past
it. The cap was added because its absence was a genuine bug, not a dialect
difference — see below.

The generator sidesteps #3 by emitting only lowercase text and lowercase
patterns, which makes the difference *unobservable* rather than papered over.

That invariant is easier to break than it looks. A 60 000-case soak reported

```sql
SELECT CASE WHEN upper(b1) LIKE '%a%' THEN … END FROM t1;   -- b1 = 'ab'
-- granular: NULL      sqlite: 4
```

which is not an engine bug at all: `upper()` manufactures the uppercase the
generator was carefully not producing, and `LIKE` is the one operator that
folds case. `replace(s,'a','X')` had the same hole. The fix is not to drop
`upper` — it is fine everywhere else, because every other text operator
compares bytes — but to build `LIKE`'s operand from a plain column or literal
instead of from the general text-expression generator. The rule to keep in mind
when extending `gen_call`: **nothing case-shifted may reach a `LIKE`**.

---

Entry 11 is the counter-example to the whole "exclude it from the generator"
reflex, and it is worth reading before adding a twelfth row to this table.

Decimal division answers at `max(scale(lhs), DIV_MIN_SCALE)`, where
`DIV_MIN_SCALE` is 6 — so `(2.25 / 8.0) / 4.0` is `0.070313` against an exact
`0.0703125`, because the true quotient needs scale 7 and the type carries 6.
That is not an engine bug: no fixed scale is exact for every quotient (`1/3`
settles it), and the exactness the type buys on `+`, `-` and `*` is the entire
reason bare decimal literals stopped being floats.

Chasing that seed further turned up something that was *not* a divergence at
all. Scales add under multiplication, and nothing capped them, so two scale-6
quotients made a scale-12 product with six integer digits of headroom left:

```sql
SELECT (4000.0 / 2.0) * (4000.0 / 2.0);
-- granular: ERROR  multiply: result does not fit Decimal64(12)
-- sqlite:   4000000.0
```

Four million, refused. `2000.0 * 2000.0` was fine — it was the scale-6 division
feeding the multiply that poisoned it, which is why the two decimal features
had to be looked at together. Fixed by capping the product scale at
`MUL_MAX_SCALE` and rounding into it rather than refusing; the 20 000-case
property test in `scalar.rs` now asserts the product is within half a unit of
exact at whatever scale it came back at, which is strictly stronger than the
plain equality it asserted before, since that is what the claim degenerates to
wherever the cap does not bite.

The trap is that **divergence #2 already claimed to have handled this** — "the
generator only ever divides by a non-zero *real* literal, where both agree".
That claim was false. It survived 25 000 cases because the literal pool happens
to produce quotients that terminate by scale 6, and it broke at 100 000 the
first time two divisions chained. A restriction that is *almost* true is worse
than no restriction, because the green runs are read as evidence.

So the fix went into `cells_equal`, not the generator: an exact decimal is
compared **at its own scale**, both sides held to within half a unit there. A
quotient wrong by a unit still fails; only sqlite's extra digits are forgiven.
`decimal_division_is_fixed_scale_and_the_comparator_still_sees_errors` pins
both halves — that the divergence is real, *and* that the slack granted for it
does not extend to a decimal that is genuinely wrong, or to two plain floats.

The same reasoning had already been written down one section up, in the note
about entry 1: an exclusion is a claim about the engine, and claims need
evidence. `Value::Decimal`'s arm in `cell_of_value` even carried a comment
saying it existed "to keep the match total rather than to carry traffic" — true
when it was written, false the day decimal literals became exact, and nothing
re-read it for two waves.

---

Entry 1 is struck through on purpose, and it is the most useful line in the
table. It was filed as a dialect difference on the strength of a single probe
with literal operands (`SELECT 1 IN (2, NULL)`). Running the same expression
against a *column* showed granular answering `NULL`, in exact agreement with
SQLite — so the engine has two answers for one expression and the "dialect
difference" was a bug all along. **A one-line probe is not enough evidence to
exclude something from a differential harness**; every exclusion here should be
re-checked against a non-constant operand before it is believed.

---

## Bugs this harness found

All five were found by the randomized run, minimized by the shrinker, and
confirmed by hand. None are fixed here — the engine files belong to other
tasks — so each is pinned by a test that asserts the **current, wrong**
behaviour and tells you what to do when it changes.

### BUG 1 — duplicate sort keys are silently dropped (data loss)

```sql
CREATE TABLE t (id Int64, v Int64) ENGINE = MergeTree ORDER BY id;
INSERT INTO t VALUES (4, 1), (4, 2);   -- "2 rows affected"
SELECT count(*) FROM t;                -- granular: 1     sqlite: 2
```

`ORDER BY id` on a single integer column makes the table *fast-PK*
(`Schema::has_fast_pk`, `src/types/schema.rs`), which routes writes into the
**keyed** delta, where `put_keyed` overwrites the slot an existing key already
owns (`src/storage/delta.rs`). But ClickHouse's `ORDER BY` is a *sort* key, not
a unique key: duplicates are legal and every row must survive. The `INSERT`
still reports the full count, so nothing in the system says data was lost.

`ORDER BY (id, k)` and `ORDER BY tuple()` are unaffected, which is what lets the
generator keep colliding sort keys on those two shapes.
Pinned: `duplicate_sort_keys_are_silently_dropped`.

### BUG 2 — `sum` over zero rows depends on its argument's nullability

```sql
CREATE TABLE t (id Int64, nn Float64, nl Nullable(Float64)) ENGINE = MergeTree ORDER BY id;
SELECT sum(nn), sum(nl) FROM t;   -- granular: 0, NULL     sqlite: NULL, NULL
```

SQLite and the SQL standard say NULL for both. ClickHouse says 0 for both.
granular says *both*, decided by a static property of the argument — so
whichever compatibility target it means to follow, one of the two answers is
wrong. `avg`, `min`, `max` and `count` over the same empty input all agree with
SQLite, which is what makes `sum` the outlier rather than a house style. It also
fires through `HAVING`, where it changes the row count.
Pinned: `sum_over_zero_rows_depends_on_argument_nullability`; forgiven in the
comparator by the narrow `is_known_empty_sum` filter.

### BUG 3 — a non-boolean `WHERE` operand follows neither reference

```sql
SELECT count(*) FROM t WHERE 'z';   -- granular: every row    sqlite: 0 rows
SELECT count(*) FROM t WHERE '';    -- granular: 0 rows       sqlite: 0 rows
```

SQLite coerces text to a number, so `'z'` is 0 and filters everything out.
ClickHouse rejects a `String` filter outright. granular does a third thing —
Python-style truthiness, where a non-empty string is true.
Pinned: `non_boolean_where_operand_follows_neither_reference`.

### BUG 4 — arithmetic on two booleans is typed as a boolean

```sql
SELECT (1=1) + (2=2);   -- should be 2; granular returns `true`
SELECT (1=2) - (1=1);   -- should be -1; granular fails: "-1 is not a Bool"
```

`DataType::promote` (`src/types/datatype.rs`) opens with
`_ if ba == bb => ba.clone()`, so `promote(Bool, Bool)` is `Bool`; the arm four
lines below that correctly widens `Bool` against an integer never gets a chance.
The result column is a `Bool` and the arithmetic answer is forced into it — so
the *same expression shape* silently truncates in one direction (2 → `true`) and
aborts the query in the other (−1 → error). Mixed `Bool`/`Int64` operands take
the correct arm and are fine.
Pinned: `arithmetic_on_two_booleans_is_typed_as_a_boolean`.

### BUG 5 — constant folding drops three-valued logic

```sql
SELECT (CAST(NULL AS Nullable(Int64)) < 5) AND (1 = 1);  -- granular: false   sqlite: NULL
SELECT CAST(NULL AS Nullable(Int64)) BETWEEN 1 AND 5;    -- granular: false   sqlite: NULL
SELECT 5 IN (2, NULL);                                   -- granular: false   sqlite: NULL
```

`const_eval_at` in `src/planner/optimizer.rs` folds constant subexpressions, and
two of its arms lose UNKNOWN:

* `AND`/`OR` decide via `Value::truthy()`, which maps NULL to *false*. The
  `Binary` arm immediately below returns `Value::Null` as soon as either side is
  NULL, so this is a local omission in the short-circuit branch, not a house
  convention.
* `InList` has no NULL handling at all — `let found = list.contains(&v)` — so
  both a NULL operand and a miss against a list containing NULL fold to `false`.

`BETWEEN` is collateral damage: it desugars to `>= AND <=`.

What makes this a bug and not a dialect position is that granular's own
**vectorized** path answers `NULL` for every one of these and matches SQLite
exactly. The same expression gets two different answers depending on whether the
planner happened to fold it, so whichever answer is intended, one is wrong. In a
`WHERE`, `NOT BETWEEN` folding to `true` on a NULL operand admits rows that must
be excluded.
Pinned: `constant_folding_drops_three_valued_logic`. Because of it, the
generator anchors every predicate atom on a column, so the planner cannot fold
a predicate away — which is also what let NULLs back into generated `IN` lists.

---

## Not yet covered

Honest gaps, in rough order of expected yield:

* **Self-joins** (`t0 AS a JOIN t0 AS b`). The generator maps slot 0 to `t0` and
  slot 1 to `t1` unconditionally; a self-join needs a slot→table indirection in
  `E::Col`, `refs_col` and `shift_cols`.
* **Joins at scale.** Tables over `JOINABLE_ROWS` (64) are only ever scanned,
  because a many-to-many join of two thousand-row tables is a cartesian
  blow-up. A hash join across a granule boundary is therefore untested here —
  it needs a generator that picks join keys with a bounded fan-out.
* **Subqueries** — derived tables in `FROM`, `IN (SELECT …)`, correlated
  predicates.
* **`DELETE` / `ALTER`** interleaved with reads; only `INSERT` and `OPTIMIZE`
  are generated.
* **Persistence.** Every case runs in `Session::in_memory()`, so nothing here
  crosses a reopen. `tests/persistence.rs` covers that, without an oracle.
* **Date and DateTime**, deliberately: the two dialects' date handling is barely
  related and would generate noise instead of evidence.
* **Concurrency.** One session, one thread per case.
* **Constant-folded predicates.** Deliberately excluded until BUG 5 is fixed —
  every generated atom is anchored on a column so the planner cannot fold it.
  When the fold is repaired, remove the anchor in `gen_atom_pred` and the whole
  folding path comes back under test, which is where BUG 5 lived.
