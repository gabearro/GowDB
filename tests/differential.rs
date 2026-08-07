//! Randomized differential testing of `granular` against the `sqlite3` CLI.
//!
//! # Why this file exists
//!
//! Every other test in this repository encodes *the implementer's* reading of
//! SQL. When the implementer and the test author are the same person, a shared
//! misunderstanding is invisible: the test agrees with the bug. This harness
//! replaces that circularity with an independent oracle. It generates a random
//! schema, random rows and a random query, runs the *same* semantics through
//! both engines, and compares answers. Neither engine gets a vote on what is
//! correct; they only get to disagree, and a disagreement is a defect in one of
//! them.
//!
//! # What it is allowed to compare
//!
//! Only the **dialect intersection**. `granular` speaks a ClickHouse flavour
//! and SQLite speaks its own; the generator is deliberately restricted so that
//! a mismatch means a bug rather than a documented dialect difference. The
//! restrictions and the reasoning behind each one are in `KNOWN DIVERGENCES`
//! below and in `tests/README-testing.md`; every one of them is pinned by
//! `known_divergences_still_reproduce`, which fails the moment the engine
//! stops diverging so the exclusion gets deleted instead of rotting.
//!
//! # Mutations
//!
//! A fifth of cases run one or two `DELETE`/`UPDATE` statements between the
//! load and the query. Both are ANSI and granular accepts them verbatim, so the
//! *same* statement text goes to both engines and the ordinary query comparison
//! observes the state it left behind -- no new rendering, no new comparator, and
//! the whole query generator serving as the read-back. Two restrictions are in
//! force and both are bugs rather than dialect differences: mutations only run
//! on keyed tables (BUG 7) and a statement's predicate never reads a column it
//! assigns (BUG 8). Each is pinned by its own test, so the restriction has to be
//! deleted when the bug is.
//!
//! # Determinism
//!
//! One seed reproduces one case exactly: schema, rows and query all come out of
//! `splitmix64` (the engine's own mixer -- no dependency, and reusing it means
//! the harness has nothing of its own to get wrong). A failure prints a
//! standalone script for each dialect that can be pasted straight into
//! `sqlite3` and `granular -q`, after shrinking the schema, the rows and the
//! query down to the smallest input that still disagrees.
//!
//! # Running it
//!
//! `cargo test --test differential` runs `DEFAULT_CASES`. The knobs, all
//! optional:
//!
//! ```text
//!   GRANULAR_DIFF_CASES=50000     how many cases (default DEFAULT_CASES)
//!   GRANULAR_DIFF_SEED=12345      starting seed, decimal; case n uses seed+n
//!   GRANULAR_DIFF_VERBOSE=1       progress every 50 cases
//!   GRANULAR_DIFF_NO_SHRINK=1     report the case exactly as generated
//!   GRANULAR_DIFF_NO_BATCH=1      one sqlite3 process per case (4x slower)
//!   GRANULAR_DIFF_SQLITE=/path    use this sqlite3 (point it at a nonexistent
//!                                 path to exercise the skip-without-sqlite3
//!                                 branch on a machine that has sqlite3)
//! ```
//!
//! See `tests/README-testing.md` for the long form, including the four bugs
//! this harness found and the dialect differences it deliberately avoids.

use std::any::Any;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::process::{Command, Stdio};
use std::sync::Once;

use granular::common::splitmix64;
use granular::types::Value;
use granular::Session;

/// Kept small enough that `cargo test` stays interactive. The dominant cost is
/// one `sqlite3` process per *batch*, not per case -- see `sqlite_batch`.
const DEFAULT_CASES: usize = 400;

/// Cases per `sqlite3` invocation. Process spawn dominates: 1200 cases take
/// 1.68s batched and 6.82s one-process-per-case (A/B interleaved, best-of-3).
const BATCH: usize = 32;

/// Row budget per batch, so one big case does not turn a batch into a
/// multi-megabyte script.
const BATCH_ROWS: usize = 3000;

/// Emitted between cases in a batch so the reader can attribute rows. It is a
/// single-column row of a string no generator can produce (the text pool is
/// lowercase letters, `%`, `_` and `'`).
const SENTINEL: &str = "<<granular-diff-boundary>>";

// =========================================================================
// KNOWN DIVERGENCES  -- genuine dialect differences, not bugs.
//
// Each one is excluded from the generator *and* pinned by a test, so it cannot
// silently stop being true.
//
//  1. WITHDRAWN.  `1 IN (2, NULL)` returning false was filed here as
//     ClickHouse compatibility, on the strength of one probe with literal
//     operands. It is not: granular's *vectorized* IN returns NULL and agrees
//     with SQLite exactly; only the constant-folding path answers false. Same
//     engine, two answers, so it is BUG 5, not a dialect. The generator now
//     puts NULL in IN lists again -- it just anchors the operand on a column so
//     the planner cannot fold the predicate away.
//
//  2. Integer division.  `7 / 2` is 3 in SQLite (integer division) and 3.5 in
//     ClickHouse/granular (always-float division). The generator only ever
//     divides by a non-zero *real* literal, where both agree.
//
//  3. LIKE case folding.  SQLite's LIKE is ASCII-case-insensitive by default;
//     granular's is case-sensitive (ClickHouse has ILIKE for the other one).
//     The generator emits only lowercase text and lowercase patterns, which
//     makes the difference unobservable rather than papered over.
//
//  4. `CAST(<real> AS TEXT)`.  SQLite renders 1.0 as '1.0'; granular renders it
//     as '1'. A rendering convention, not an evaluation difference. Real->Text
//     casts are not generated.
//
//  5. Float summation -- the divergence that turned out not to exist.
//     The brief predicted that granular's Neumaier compensation would have to
//     be reconciled against SQLite's naive summation. Measured: SQLite adopted
//     Kahan-Babuska-Neumaier in `sum`/`avg`/`total` in 3.44, so on 3.54 the
//     two agree bit-for-bit even on 1e16 + 1*100 - 1e16. See
//     `float_summation_agrees_because_both_engines_compensate`, which asserts
//     both sides and states what to do on an older sqlite3. The generator
//     still draws reals whose sums are exact in binary64 (quarters and small
//     powers of ten) so the harness stays correct against a pre-3.44 oracle,
//     and `FLOAT_REL_TOL` stays tight rather than absorbing real bugs.
//
//  6. `CAST(<bool> AS TEXT)`.  granular has a real `Bool` type and renders
//     'true'; SQLite has none and renders '1'. Only the rendering differs --
//     Bool arithmetic itself agrees now that BUG 4 is fixed, mixed with an
//     integer or with another Bool. Bool->Text casts are not generated.
//
//  7. `EXCEPT` / `INTERSECT`.  granular parses them and says, clearly, that
//     only UNION is implemented. A capability gap, not a wrong answer; the
//     set-operation menu is UNION and UNION ALL.
//
//  8. `round`.  granular rounds half to even (ClickHouse's rule): `round(2.5)`
//     is 2. SQLite rounds half away from zero: 3.0. Not in `gen_call`.
//
//  9. `concat`.  SQLite's `concat()` skips NULL arguments, so
//     `concat(NULL,'b')` is 'b'; granular propagates and returns NULL. Not in
//     `gen_call`.
// =========================================================================
// BUGS THIS HARNESS FOUND -- real defects, each pinned by a test that asserts
// the current *wrong* behaviour so it fails the day the engine is fixed. The
// engine files belong to other tasks, so none are repaired here.
//
//   BUG 1  duplicate sort keys are silently dropped (data loss)
//          -> duplicate_sort_keys_are_silently_dropped
//   BUG 2  FIXED. `sum` over zero rows answered 0 or NULL by argument
//          nullability; it is NULL either way now, and the test is inverted
//          -> sum_over_zero_rows_is_null_regardless_of_nullability
//   BUG 3  a non-boolean WHERE operand follows neither reference
//          -> non_boolean_where_operand_follows_neither_reference
//   BUG 4  FIXED. arithmetic on two booleans was typed as a boolean; it
//          widens to Int64 now, and the test is inverted
//          -> arithmetic_on_two_booleans_widens_to_an_integer
//   BUG 5  FIXED. constant folding dropped three-valued logic (AND/OR and
//          IN); the folder now agrees with the vectorized evaluator, and the
//          test is inverted
//          -> constant_folding_keeps_three_valued_logic
//   BUG 6  subtraction of two *unsigned* operands wraps in the vectorized
//          path and errors in the constant-folding path -- one expression,
//          two behaviours. Found when BUG 4's fix let the generator emit
//          `length(a) - length(b)` (`length` returns UInt64)
//          -> unsigned_subtraction_wraps_one_way_and_errors_the_other
//   BUG 7  UPDATE on a table with no single-column primary key *appends* the
//          rewritten rows and leaves the originals in place, so a three-row
//          table becomes four rows with two versions of the same row live at
//          once. No error. Found by hand while adding mutation generation
//          (below), which is why the generator only mutates keyed tables
//          -> unkeyed_update_duplicates_the_row_instead_of_replacing_it
//   BUG 8  UPDATE whose predicate reads a column the same statement assigns
//          is a silent no-op: `UPDATE t SET id = id + 10 WHERE id = 1`
//          changes nothing. `session::run_alter_update` renders the mutation
//          as `SELECT id + 10 AS id ... WHERE id = 1`, and this dialect lets
//          WHERE see select-list aliases -- so the predicate binds to the
//          *assignment*, not the column, and matches no row. Binding the
//          predicate against the table rather than a synthesized projection
//          (`Binder::bind_update`) is the fix
//          -> update_whose_predicate_reads_an_assigned_column_does_nothing
//
// Full write-ups with reproducers are in tests/README-testing.md.
// =========================================================================

/// Relative tolerance for float comparison. Deliberately tight: the generator
/// keeps every real exactly representable and every sum exact, so anything
/// looser would be hiding a real disagreement rather than absorbing noise. The
/// slack that remains is for the one genuinely inexact operation in the
/// grammar, `avg` (a single division, so <=1ulp on each side).
const FLOAT_REL_TOL: f64 = 1e-12;

// ------------------------------------------------------------------ rng

/// `splitmix64` over a counter. Same stream the engine's own hash tests use,
/// so there is no second-rate PRNG in the repo to audit.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(1);
        splitmix64(self.0)
    }
    /// Uniform in `0..n`. `n` must be non-zero.
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    /// `n` in `lo..=hi`.
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }
    /// True with probability `p/100`.
    fn pct(&mut self, p: u64) -> bool {
        self.next() % 100 < p
    }
    /// Callers write `*rng.pick(&[...])` on `&'static str` menus. clippy calls
    /// that an explicit auto-deref; it is wrong -- dropping the `*` makes
    /// inference pick `[str]`, which is unsized, and the file stops compiling.
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

// ------------------------------------------------------------------ values

#[derive(Clone, Debug)]
enum Cell {
    Null,
    Int(i64),
    Real(f64),
    Text(String),
}

impl Cell {
    /// SQL literal text, appended in place. Identical in both dialects, which
    /// is the point: the two scripts differ only where they must (DDL, CAST
    /// type names).
    ///
    /// Writes into the caller's buffer rather than returning a `String`: an
    /// 8200-row table is ~30k literals, each script is rendered for two
    /// dialects, and the shrinker re-renders both hundreds of times. Returning
    /// an owned `String` per cell was the harness's single largest allocation
    /// source.
    fn write_sql(&self, out: &mut String) {
        match self {
            Cell::Null => out.push_str("NULL"),
            Cell::Int(i) => {
                let _ = write!(out, "{i}");
            }
            // `{:?}` on f64 is the shortest round-tripping form and always
            // carries a `.` or an `e`, so the literal can never be re-lexed as
            // an integer -- which would silently flip SQLite into integer
            // division and manufacture a fake divergence.
            Cell::Real(f) => {
                let _ = write!(out, "{f:?}");
            }
            Cell::Text(s) => {
                out.push('\'');
                for c in s.chars() {
                    if c == '\'' {
                        out.push('\'');
                    }
                    out.push(c);
                }
                out.push('\'');
            }
        }
    }

    /// Sort rank that unifies Int and Real, so a row containing `1` sorts to
    /// the same place as the same row containing `1.0`. Required for the
    /// multiset comparison of unordered queries to be type-blind in exactly the
    /// way `cells_equal` is.
    fn rank(&self) -> u8 {
        match self {
            Cell::Null => 0,
            Cell::Int(_) | Cell::Real(_) => 1,
            Cell::Text(_) => 2,
        }
    }

    fn num(&self) -> f64 {
        match self {
            Cell::Int(i) => *i as f64,
            Cell::Real(f) => *f,
            _ => 0.0,
        }
    }

    /// Sort key for the numeric case. `total_cmp` orders `-0.0` *before* `0.0`,
    /// which is exactly wrong here: the two engines render the same IEEE zero
    /// differently (granular keeps the sign from `0 * -0.5`, SQLite's quote
    /// mode prints `0.0`), so a raw `total_cmp` sorts the two sides into
    /// different orders and reports a mismatch that is entirely the harness's
    /// own doing. Seen once, on `SELECT CAST(k AS Float64) * b1, ...`.
    fn sort_num(&self) -> f64 {
        let f = self.num();
        if f == 0.0 {
            0.0
        } else {
            f
        }
    }

    fn text(&self) -> &str {
        match self {
            Cell::Text(s) => s,
            _ => "",
        }
    }
}

/// Semantic equality across the two engines' type systems.
///
/// Int and Real compare numerically because SQLite's `sum` over an INTEGER
/// column returns INTEGER while granular may return either, and the *type* of a
/// numeric aggregate is not part of the intersection. `-0.0 == 0.0` falls out
/// of that and is correct: SQLite prints `0.0`, granular prints `-0`, and they
/// are the same number.
fn cells_equal(a: &Cell, b: &Cell) -> bool {
    match (a, b) {
        (Cell::Null, Cell::Null) => true,
        (Cell::Text(x), Cell::Text(y)) => x == y,
        (Cell::Int(x), Cell::Int(y)) => x == y,
        (Cell::Int(_) | Cell::Real(_), Cell::Int(_) | Cell::Real(_)) => {
            let (x, y) = (a.num(), b.num());
            if x.is_nan() || y.is_nan() {
                return x.is_nan() && y.is_nan();
            }
            if x == y {
                return true;
            }
            (x - y).abs() <= FLOAT_REL_TOL * x.abs().max(y.abs()).max(1.0)
        }
        _ => false,
    }
}

fn rows_equal(a: &[Cell], b: &[Cell]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| cells_equal(x, y))
}

/// Canonical order used to compare unordered results as multisets. Only needs
/// to be *a* total-ish order that both sides agree on; ties among
/// `cells_equal` values are harmless because such rows compare equal anyway.
fn canonical_sort(rows: &mut [Vec<Cell>]) {
    rows.sort_by(|x, y| {
        for (a, b) in x.iter().zip(y) {
            let o = a
                .rank()
                .cmp(&b.rank())
                .then_with(|| a.sort_num().total_cmp(&b.sort_num()))
                .then_with(|| a.text().cmp(b.text()));
            if o != std::cmp::Ordering::Equal {
                return o;
            }
        }
        x.len().cmp(&y.len())
    });
}

fn fmt_cell(c: &Cell) -> String {
    match c {
        Cell::Null => "NULL".into(),
        Cell::Int(i) => i.to_string(),
        Cell::Real(f) => format!("{f:?}"),
        Cell::Text(s) => format!("{s:?}"),
    }
}

fn fmt_rows(rows: &[Vec<Cell>]) -> String {
    let mut s = String::new();
    for r in rows.iter().take(40) {
        let cells: Vec<String> = r.iter().map(fmt_cell).collect();
        let _ = writeln!(s, "    {}", cells.join(" | "));
    }
    if rows.len() > 40 {
        let _ = writeln!(s, "    ... {} more rows", rows.len() - 40);
    }
    s
}

// ------------------------------------------------------------------ schema

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    Int,
    Real,
    Text,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dialect {
    Granular,
    Sqlite,
}

impl Ty {
    fn column(self, d: Dialect, nullable: bool) -> String {
        match d {
            Dialect::Granular => {
                let base = match self {
                    Ty::Int => "Int64",
                    Ty::Real => "Float64",
                    Ty::Text => "String",
                };
                if nullable {
                    format!("Nullable({base})")
                } else {
                    base.to_string()
                }
            }
            Dialect::Sqlite => {
                let base = match self {
                    Ty::Int => "INTEGER",
                    Ty::Real => "REAL",
                    Ty::Text => "TEXT",
                };
                if nullable {
                    base.to_string()
                } else {
                    format!("{base} NOT NULL")
                }
            }
        }
    }

    fn cast_name(self, d: Dialect) -> &'static str {
        match (self, d) {
            (Ty::Int, Dialect::Granular) => "Int64",
            (Ty::Real, Dialect::Granular) => "Float64",
            (Ty::Text, Dialect::Granular) => "String",
            (Ty::Int, Dialect::Sqlite) => "INTEGER",
            (Ty::Real, Dialect::Sqlite) => "REAL",
            (Ty::Text, Dialect::Sqlite) => "TEXT",
        }
    }
}

#[derive(Clone, Debug)]
struct ColDef {
    name: String,
    ty: Ty,
    nullable: bool,
}

/// The shape of granular's `ORDER BY` clause, which sqlite has no equivalent
/// of. It is not cosmetic: a *single integer* sort column makes the table
/// "fast-PK" (`Schema::has_fast_pk`) and routes writes through the keyed delta,
/// where an existing key is overwritten. See `BUG 1` in
/// `duplicate_sort_keys_are_silently_dropped`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SortKey {
    /// `ORDER BY id` -- keyed delta, so the generator must keep `id` unique.
    Id,
    /// `ORDER BY (id, k)` -- composite, so not fast-PK, so duplicates survive.
    IdK,
    /// `ORDER BY tuple()` -- unkeyed append.
    Tuple,
}

impl SortKey {
    fn clause(self) -> &'static str {
        match self {
            SortKey::Id => "ORDER BY id",
            SortKey::IdK => "ORDER BY (id, k)",
            SortKey::Tuple => "ORDER BY tuple()",
        }
    }
}

#[derive(Clone, Debug)]
struct TableDef {
    name: String,
    cols: Vec<ColDef>,
    rows: Vec<Vec<Cell>>,
    sort: SortKey,
    /// Rows per `INSERT`. One statement is the easy path; several make the
    /// engine hold live data in the delta *and* in one or more sealed parts at
    /// once, which is where the scan has to merge two different physical
    /// representations of the same table. sqlite cannot tell the difference,
    /// which is exactly what makes it a useful oracle for it.
    chunk: usize,
    /// Emit `OPTIMIZE TABLE ... FINAL` after loading, forcing the parts to
    /// merge before the query runs.
    optimize: bool,
    /// Declare `PRIMARY KEY id` and draw `id` without replacement.
    ///
    /// Only mutation cases set this, and they have to: `DELETE`/`UPDATE`
    /// currently refuse a table with no single-column primary key, and a
    /// primary key in this engine is a *unique* key -- so colliding ids, which
    /// every other case deliberately generates, would make granular collapse
    /// rows sqlite keeps and produce a divergence that is about the key, not
    /// about the mutation. Widening mutation coverage to unkeyed tables is
    /// gated on BUG 7.
    keyed: bool,
}

// ------------------------------------------------------------------ expressions

#[derive(Clone, Debug)]
enum E {
    /// `x{slot}.{name}` -- always qualified, so a join can never make it
    /// ambiguous.
    Col(usize, usize),
    /// An *unqualified* name, plus the type it resolves to (the generator needs
    /// the type to keep comparisons same-category). Only produced for `USING`
    /// columns, where the bare name is the coalesced join column -- exactly the
    /// construct the first hand-run diff against sqlite3 caught granular
    /// getting wrong.
    #[allow(dead_code)]
    Bare(String, Ty),
    Lit(Cell),
    Bin(&'static str, Box<E>, Box<E>),
    Not(Box<E>),
    /// `expr IS [NOT] NULL`
    IsNull(Box<E>, bool),
    /// `expr [NOT] BETWEEN lo AND hi`
    Between(Box<E>, Box<E>, Box<E>, bool),
    /// `expr [NOT] IN (list)`
    In(Box<E>, Vec<E>, bool),
    /// `expr [NOT] LIKE 'pattern'`
    Like(Box<E>, String, bool),
    /// `CASE WHEN .. THEN .. [ELSE ..] END`
    Case(Vec<(E, E)>, Option<Box<E>>),
    Cast(Box<E>, Ty),
    /// A scalar function spelled identically in both dialects. Only the subset
    /// in `gen_call` is generated -- see the comment there for the ones that
    /// look shared but are not.
    Call(&'static str, Vec<E>),
    /// `count(*)` is `Agg("count", None, false)`; the flag is `DISTINCT`.
    Agg(&'static str, Option<Box<E>>, bool),
}

impl E {
    fn b(self) -> Box<E> {
        Box::new(self)
    }

    fn render(&self, d: Dialect, tables: &[TableDef], out: &mut String) {
        match self {
            E::Col(slot, c) => {
                let _ = write!(out, "x{slot}.{}", tables[*slot].cols[*c].name);
            }
            E::Bare(name, _) => out.push_str(name),
            E::Lit(c) => c.write_sql(out),
            E::Bin(op, l, r) => {
                out.push('(');
                l.render(d, tables, out);
                let _ = write!(out, " {op} ");
                r.render(d, tables, out);
                out.push(')');
            }
            E::Not(e) => {
                out.push_str("(NOT ");
                e.render(d, tables, out);
                out.push(')');
            }
            E::IsNull(e, neg) => {
                out.push('(');
                e.render(d, tables, out);
                out.push_str(if *neg { " IS NOT NULL)" } else { " IS NULL)" });
            }
            E::Between(e, lo, hi, neg) => {
                out.push('(');
                e.render(d, tables, out);
                out.push_str(if *neg { " NOT BETWEEN " } else { " BETWEEN " });
                lo.render(d, tables, out);
                out.push_str(" AND ");
                hi.render(d, tables, out);
                out.push(')');
            }
            E::In(e, list, neg) => {
                out.push('(');
                e.render(d, tables, out);
                out.push_str(if *neg { " NOT IN (" } else { " IN (" });
                for (i, it) in list.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    it.render(d, tables, out);
                }
                out.push_str("))");
            }
            E::Like(e, pat, neg) => {
                out.push('(');
                e.render(d, tables, out);
                out.push_str(if *neg { " NOT LIKE '" } else { " LIKE '" });
                out.push_str(&pat.replace('\'', "''"));
                out.push_str("')");
            }
            E::Case(arms, els) => {
                out.push_str("CASE");
                for (w, t) in arms {
                    out.push_str(" WHEN ");
                    w.render(d, tables, out);
                    out.push_str(" THEN ");
                    t.render(d, tables, out);
                }
                if let Some(e) = els {
                    out.push_str(" ELSE ");
                    e.render(d, tables, out);
                }
                out.push_str(" END");
            }
            E::Cast(e, ty) => {
                out.push_str("CAST(");
                e.render(d, tables, out);
                let _ = write!(out, " AS {})", ty.cast_name(d));
            }
            E::Call(name, args) => {
                let _ = write!(out, "{name}(");
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    a.render(d, tables, out);
                }
                out.push(')');
            }
            E::Agg(name, arg, distinct) => {
                let _ = write!(out, "{name}(");
                if *distinct {
                    out.push_str("DISTINCT ");
                }
                match arg {
                    Some(a) => a.render(d, tables, out),
                    None => out.push('*'),
                }
                out.push(')');
            }
        }
    }

    fn refs_col(&self, slot: usize, col: usize) -> bool {
        let mut hit = false;
        self.walk(&mut |e| {
            if let E::Col(s, c) = e {
                hit |= *s == slot && *c == col;
            }
        });
        hit
    }

    fn refs_slot(&self, slot: usize) -> bool {
        let mut hit = false;
        self.walk(&mut |e| {
            if let E::Col(s, _) = e {
                hit |= *s == slot;
            }
        });
        hit
    }

    fn walk(&self, f: &mut impl FnMut(&E)) {
        f(self);
        match self {
            E::Col(..) | E::Bare(..) | E::Lit(_) => {}
            E::Bin(_, a, b) => {
                a.walk(f);
                b.walk(f);
            }
            E::Not(a) | E::IsNull(a, _) | E::Cast(a, _) | E::Like(a, _, _) => a.walk(f),
            E::Between(a, b, c, _) => {
                a.walk(f);
                b.walk(f);
                c.walk(f);
            }
            E::In(a, list, _) => {
                a.walk(f);
                for x in list {
                    x.walk(f);
                }
            }
            E::Call(_, args) => {
                for x in args {
                    x.walk(f);
                }
            }
            E::Case(arms, els) => {
                for (w, t) in arms {
                    w.walk(f);
                    t.walk(f);
                }
                if let Some(e) = els {
                    e.walk(f);
                }
            }
            E::Agg(_, Some(a), _) => a.walk(f),
            E::Agg(_, None, _) => {}
        }
    }

    /// Rewrite column indices after a column has been dropped from `slot`.
    fn shift_cols(&mut self, slot: usize, removed: usize) {
        match self {
            E::Col(s, c) => {
                if *s == slot && *c > removed {
                    *c -= 1;
                }
            }
            E::Bare(..) | E::Lit(_) => {}
            E::Bin(_, a, b) => {
                a.shift_cols(slot, removed);
                b.shift_cols(slot, removed);
            }
            E::Not(a) | E::IsNull(a, _) | E::Cast(a, _) | E::Like(a, _, _) => {
                a.shift_cols(slot, removed)
            }
            E::Between(a, b, c, _) => {
                a.shift_cols(slot, removed);
                b.shift_cols(slot, removed);
                c.shift_cols(slot, removed);
            }
            E::In(a, list, _) => {
                a.shift_cols(slot, removed);
                for x in list {
                    x.shift_cols(slot, removed);
                }
            }
            E::Call(_, args) => {
                for x in args {
                    x.shift_cols(slot, removed);
                }
            }
            E::Case(arms, els) => {
                for (w, t) in arms {
                    w.shift_cols(slot, removed);
                    t.shift_cols(slot, removed);
                }
                if let Some(e) = els {
                    e.shift_cols(slot, removed);
                }
            }
            E::Agg(_, Some(a), _) => a.shift_cols(slot, removed),
            E::Agg(_, None, _) => {}
        }
    }
}

// ------------------------------------------------------------------ query

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum JoinOp {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

impl JoinOp {
    fn sql(self) -> &'static str {
        match self {
            JoinOp::Inner => "INNER JOIN",
            JoinOp::Left => "LEFT OUTER JOIN",
            JoinOp::Right => "RIGHT OUTER JOIN",
            JoinOp::Full => "FULL OUTER JOIN",
            JoinOp::Cross => "CROSS JOIN",
        }
    }
}

#[derive(Clone, Debug)]
enum JoinCon {
    On(E),
    Using(Vec<String>),
    None,
}

#[derive(Clone, Debug)]
enum From {
    One(usize),
    Join(JoinOp, JoinCon),
}

/// A generated `SELECT`.
///
/// Row order is only defined when `order` is non-empty, and the generator only
/// ever produces a *total* order (a random prefix followed by every remaining
/// output ordinal), so `LIMIT`/`OFFSET` slice a well-defined sequence in both
/// engines. When `order` is empty the comparison is a multiset comparison --
/// neither engine promises an order there, and pretending otherwise would
/// manufacture failures.
#[derive(Clone, Debug)]
struct Query {
    from: From,
    /// `SELECT *`; `items` is then empty and the arity is whatever the engine
    /// decides -- which is itself under test.
    star: bool,
    distinct: bool,
    items: Vec<E>,
    filter: Option<E>,
    group_by: Vec<usize>,
    having: Option<E>,
    /// `UNION [ALL] / EXCEPT / INTERSECT` with a second query of identical
    /// arity and column types. The tail carries its own FROM/WHERE but never
    /// its own ORDER BY or LIMIT -- both dialects attach those to the compound,
    /// not to the branch.
    set_tail: Option<(&'static str, Box<Query>)>,
    /// `(1-based ordinal, ascending, nulls first)`
    order: Vec<(usize, bool, bool)>,
    limit: Option<usize>,
    offset: usize,
}

impl Query {
    /// Every expression the query owns, including the ones in a set-operation
    /// branch. Forgetting the branch here is not a cosmetic bug: the
    /// column-dropping reduction decides a column is unreferenced from this
    /// walk, and a missed reference means the shrinker renders a column index
    /// that no longer exists. That is exactly how it first failed.
    fn for_each_expr(&self, f: &mut impl FnMut(&E)) {
        for e in &self.items {
            f(e);
        }
        for e in [&self.filter, &self.having].into_iter().flatten() {
            f(e);
        }
        if let From::Join(_, JoinCon::On(e)) = &self.from {
            f(e);
        }
        if let Some((_, tail)) = &self.set_tail {
            tail.for_each_expr(f);
        }
    }

    fn for_each_expr_mut(&mut self, f: &mut impl FnMut(&mut E)) {
        for e in &mut self.items {
            f(e);
        }
        for e in [&mut self.filter, &mut self.having].into_iter().flatten() {
            f(e);
        }
        if let From::Join(_, JoinCon::On(e)) = &mut self.from {
            f(e);
        }
        if let Some((_, tail)) = &mut self.set_tail {
            tail.for_each_expr_mut(f);
        }
    }

    /// True when any branch selects `*`, in which case dropping a column
    /// changes the answer even though no expression names it.
    fn any_star(&self) -> bool {
        self.star || self.set_tail.as_ref().is_some_and(|(_, t)| t.any_star())
    }
}

/// A `DELETE` or `UPDATE` run between the load and the query.
///
/// Spelled identically in both dialects -- `DELETE FROM t WHERE p` and
/// `UPDATE t SET c = e WHERE p` are ANSI, and granular accepts them as well as
/// its ClickHouse `ALTER TABLE t DELETE|UPDATE` synonym -- so the harness needs
/// no dialect-specific rendering and no new comparison machinery: it runs the
/// same statement through both engines and lets the existing `SELECT`
/// comparison observe the resulting table state. That makes the query
/// generator, in full, the read-back for every mutation.
#[derive(Clone, Debug)]
struct Mutation {
    /// Table slot the statement targets.
    slot: usize,
    /// `(column index, new value)`. Empty means this is a `DELETE`.
    set: Vec<(usize, E)>,
    /// `None` emits no `WHERE` at all, which must mean "every row".
    pred: Option<E>,
    /// Compact afterwards (granular only). A delete is a bitmap write, so this
    /// is the step that has to actually drop the rows from the parts -- the
    /// one place a mutation and the merge path meet.
    then_optimize: bool,
}

impl Mutation {
    fn is_delete(&self) -> bool {
        self.set.is_empty()
    }

    fn render(&self, d: Dialect, tables: &[TableDef]) -> String {
        let t = &tables[self.slot];
        let mut s = String::with_capacity(96);
        if self.is_delete() {
            let _ = write!(s, "DELETE FROM {}", t.name);
        } else {
            let _ = write!(s, "UPDATE {} SET ", t.name);
            for (i, (c, e)) in self.set.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                let _ = write!(s, "{} = ", t.cols[*c].name);
                e.render(d, tables, &mut s);
            }
        }
        if let Some(p) = &self.pred {
            s.push_str(" WHERE ");
            p.render(d, tables, &mut s);
        }
        s
    }
}

#[derive(Clone, Debug)]
struct Case {
    seed: u64,
    tables: Vec<TableDef>,
    /// Applied in order, after every table is loaded and before the query.
    mutations: Vec<Mutation>,
    query: Query,
}

impl Case {
    /// DDL + DML, one statement per element, no trailing `;`.
    fn setup(&self, d: Dialect) -> Vec<String> {
        let mut out = Vec::with_capacity(self.tables.len() * 2);
        for t in &self.tables {
            let cols: Vec<String> = t
                .cols
                .iter()
                .map(|c| format!("{} {}", c.name, c.ty.column(d, c.nullable)))
                .collect();
            out.push(match d {
                // granular's MergeTree needs a sort key; `id` and `k` are the
                // only columns present in every generated schema.
                Dialect::Granular => format!(
                    "CREATE TABLE {} ({}) ENGINE = MergeTree {}{}",
                    t.name,
                    cols.join(", "),
                    if t.keyed { "PRIMARY KEY id " } else { "" },
                    t.sort.clause()
                ),
                Dialect::Sqlite => format!("CREATE TABLE {} ({})", t.name, cols.join(", ")),
            });
            // One buffer for the whole statement, reused across chunks. The
            // obvious `Vec<String>` + `join` version allocated two Strings per
            // *cell*; on an 8200-row table that is 60k allocations for one
            // INSERT, re-done for both dialects on every shrink candidate.
            // Measured A/B interleaved, best-of-3, 4000 cases at seed
            // 987654321 (which contains several 8k-row tables):
            // 4.58s reused-buffer vs 5.39s per-cell-String, i.e. 15% off the
            // *whole* run -- and the run is dominated by sqlite3 processes and
            // by granular actually executing, not by rendering.
            let mut stmt = String::with_capacity(t.rows.len().min(t.chunk) * 8 * t.cols.len() + 32);
            for chunk in t.rows.chunks(t.chunk.max(1)) {
                stmt.clear();
                let _ = write!(stmt, "INSERT INTO {} VALUES ", t.name);
                for (i, r) in chunk.iter().enumerate() {
                    stmt.push_str(if i == 0 { "(" } else { ", (" });
                    for (j, c) in r.iter().enumerate() {
                        if j > 0 {
                            stmt.push_str(", ");
                        }
                        c.write_sql(&mut stmt);
                    }
                    stmt.push(')');
                }
                out.push(stmt.clone());
            }
            // sqlite has nothing to merge, so the statement only goes to
            // granular -- it is the closest thing to "now read it back off
            // disk instead of out of the write buffer" the harness can ask for.
            if t.optimize && d == Dialect::Granular && !t.rows.is_empty() {
                out.push(format!("OPTIMIZE TABLE {} FINAL", t.name));
            }
        }
        // Mutations run after every table is loaded, so a statement is never
        // racing an INSERT that would have changed which rows it matched.
        for m in &self.mutations {
            out.push(m.render(d, &self.tables));
            if m.then_optimize && d == Dialect::Granular {
                out.push(format!("OPTIMIZE TABLE {} FINAL", self.tables[m.slot].name));
            }
        }
        out
    }

    fn select(&self, d: Dialect) -> String {
        let q = &self.query;
        let mut s = String::with_capacity(256);
        self.render_core(q, d, &mut s);
        if let Some((op, tail)) = &q.set_tail {
            let _ = write!(s, " {op} ");
            self.render_core(tail, d, &mut s);
        }
        if !q.order.is_empty() {
            s.push_str(" ORDER BY ");
            for (i, (ord, asc, nf)) in q.order.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                // Ordinals rather than aliases: both dialects accept them, they
                // stay valid through every shrink step that rewrites the select
                // list, and they are the only form that works on the far side
                // of a UNION.
                let _ = write!(
                    s,
                    "{ord} {} NULLS {}",
                    if *asc { "ASC" } else { "DESC" },
                    if *nf { "FIRST" } else { "LAST" }
                );
            }
        }
        if let Some(n) = q.limit {
            let _ = write!(s, " LIMIT {n}");
            if q.offset > 0 {
                let _ = write!(s, " OFFSET {}", q.offset);
            }
        }
        s
    }

    /// Everything from `SELECT` through `HAVING`: one branch of a compound
    /// query, or the whole thing when there is no set operation.
    fn render_core(&self, q: &Query, d: Dialect, s: &mut String) {
        let t = &self.tables;
        s.push_str("SELECT ");
        if q.distinct {
            s.push_str("DISTINCT ");
        }
        if q.star {
            s.push('*');
        } else {
            for (i, e) in q.items.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                e.render(d, t, s);
            }
        }
        match &q.from {
            From::One(slot) => {
                let _ = write!(s, " FROM {} AS x{slot}", t[*slot].name);
            }
            From::Join(op, con) => {
                let _ = write!(s, " FROM {} AS x0 {} {} AS x1", t[0].name, op.sql(), t[1].name);
                match con {
                    JoinCon::On(e) => {
                        s.push_str(" ON ");
                        e.render(d, t, s);
                    }
                    JoinCon::Using(cols) => {
                        let _ = write!(s, " USING ({})", cols.join(", "));
                    }
                    JoinCon::None => {}
                }
            }
        }
        if let Some(w) = &q.filter {
            s.push_str(" WHERE ");
            w.render(d, t, s);
        }
        if !q.group_by.is_empty() {
            s.push_str(" GROUP BY ");
            for (i, gi) in q.group_by.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                q.items[*gi].render(d, t, s);
            }
        }
        if let Some(h) = &q.having {
            s.push_str(" HAVING ");
            h.render(d, t, s);
        }
    }

    /// A standalone, paste-able script for one dialect.
    fn script(&self, d: Dialect) -> String {
        let mut s = String::new();
        for stmt in self.setup(d) {
            let _ = writeln!(s, "{stmt};");
        }
        let _ = writeln!(s, "{};", self.select(d));
        s
    }
}

// ------------------------------------------------------------------ generator

const INTS: [i64; 12] = [0, 1, -1, 2, 3, -3, 7, 10, -10, 42, -9999, 9999];
/// Quarters and small powers of ten: every one is exact in binary64 and so is
/// every sum of them, which is what lets `FLOAT_REL_TOL` stay at 1e-12 instead
/// of hiding a real disagreement behind slack. See KNOWN DIVERGENCES #5.
const REALS: [f64; 10] = [0.0, -0.0, 0.5, -0.5, 1.0, 2.25, -3.75, 100.0, -0.25, 1024.0];
/// Lowercase only, so SQLite's case-insensitive LIKE cannot be observed
/// (KNOWN DIVERGENCES #3). `'` and the LIKE metacharacters are in the pool
/// because quoting and metacharacter-as-data are where string handling breaks.
const TEXTS: [&str; 10] = ["", "a", "b", "ab", "ba", "abc", "z", "a%", "a_", "'"];
const PATTERNS: [&str; 10] = ["a", "a%", "%a", "%a%", "_", "a_", "%", "", "ab%", "%b_"];

fn gen_value(rng: &mut Rng, ty: Ty, nullable: bool) -> Cell {
    if nullable && rng.pct(25) {
        return Cell::Null;
    }
    match ty {
        Ty::Int => Cell::Int(*rng.pick(&INTS)),
        Ty::Real => Cell::Real(*rng.pick(&REALS)),
        Ty::Text => Cell::Text((*rng.pick(&TEXTS)).to_string()),
    }
}

/// Above this, a case stops being joinable: a cross or many-to-many join of two
/// thousand-row tables is a million-row result, which tells us nothing a
/// thousand-row scan does not and costs a hundred times as much.
const JOINABLE_ROWS: usize = 64;

fn gen_tables(rng: &mut Rng, keyed: bool) -> Vec<TableDef> {
    let mut tables = Vec::with_capacity(2);
    // At most one table per case may be big, so the sqlite script stays inside
    // a megabyte and a mismatch stays shrinkable.
    let big_slot = rng.below(2);
    for (slot, (t, prefix)) in ["t0", "t1"].iter().zip(["a", "b"]).enumerate() {
        // `id` and `k` exist in both tables with the same type so USING joins
        // are always well-formed; everything else is prefixed per table so an
        // unqualified reference to it can never be ambiguous.
        let k_nullable = rng.pct(50);
        let mut cols = vec![
            ColDef { name: "id".into(), ty: Ty::Int, nullable: false },
            ColDef { name: "k".into(), ty: Ty::Int, nullable: k_nullable },
        ];
        let extra = rng.range(1, 3);
        for i in 0..extra {
            let ty = *rng.pick(&[Ty::Int, Ty::Real, Ty::Text]);
            cols.push(ColDef {
                name: format!("{prefix}{i}"),
                ty,
                nullable: rng.pct(60),
            });
        }
        // `ORDER BY (id, k)` cannot have a nullable `k` component, and the
        // engine would rather reject the DDL than sort NULLs into a key.
        // A keyed table must sort by its key, so the choice collapses.
        let sort = if keyed {
            SortKey::Id
        } else {
            match rng.below(3) {
                0 if !k_nullable => SortKey::IdK,
                1 => SortKey::Tuple,
                _ => SortKey::Id,
            }
        };
        // Most cases stay tiny so a failure shrinks fast and reads clearly. A
        // slice of them go past GRANULE_SIZE (1024) and BLOCK_SIZE (8192),
        // because those are the two numbers this engine's every fast path is
        // built around -- a zone-map skip, a granule-boundary binary search or
        // a partial final block cannot be reached at all with eight rows, and
        // sqlite is entirely indifferent to them.
        let n = if slot == big_slot {
            match rng.below(100) {
                0..=88 => rng.range(0, 8),
                // Straddles GRANULE_SIZE = 1024.
                89..=96 => rng.range(1000, 1100),
                // Straddles BLOCK_SIZE = 8192, so the scan emits a full block
                // plus a short tail.
                _ => rng.range(8180, 8210),
            }
        } else {
            rng.range(0, 8)
        };
        // Key domains widen with the table: five distinct ids across eight
        // thousand rows would make every aggregate one enormous group and every
        // join a cartesian product, testing nothing but memory.
        let id_domain = if n > JOINABLE_ROWS { n / 3 + 1 } else { 5 };
        let k_domain = if n > JOINABLE_ROWS { 97 } else { 4 };
        // A keyed table's `id` is drawn without replacement, and shuffled so it
        // is not also already in sort order -- ingest ordering is its own code
        // path and a pre-sorted key would never leave it.
        let mut unique: Vec<i64> = if keyed { (0..n as i64).collect() } else { Vec::new() };
        for i in (1..unique.len()).rev() {
            unique.swap(i, rng.below(i + 1));
        }
        let mut rows = Vec::with_capacity(n);
        for _r_i in 0..n {
            let mut r = Vec::with_capacity(cols.len());
            // Colliding sort keys are exactly where granule boundaries and
            // many-to-many hash-join buckets get exercised, so `id` is always
            // drawn from a small domain. This used to exempt `ORDER BY id`,
            // because that shape silently dropped duplicates (BUG 1) and the
            // generator would have reported the same loss on every case. Now
            // that a sort key no longer deduplicates, every shape collides --
            // which is what makes this harness actually stress the fix rather
            // than route around it.
            r.push(Cell::Int(match unique.pop() {
                Some(id) => id,
                None => rng.below(id_domain) as i64,
            }));
            r.push(if cols[1].nullable && rng.pct(25) {
                Cell::Null
            } else {
                Cell::Int(rng.below(k_domain) as i64)
            });
            for c in &cols[2..] {
                r.push(gen_value(rng, c.ty, c.nullable));
            }
            rows.push(r);
        }
        // Split the load across statements often enough that the delta and one
        // or more sealed parts are both live at query time.
        let chunk = if n > 1 && rng.pct(45) {
            rng.range(1, n)
        } else {
            n.max(1)
        };
        tables.push(TableDef {
            name: (*t).to_string(),
            cols,
            rows,
            sort,
            chunk,
            optimize: rng.pct(20),
            keyed,
        });
    }
    tables
}

/// The mutation statements a case runs before its query, or none.
///
/// One or two, on one table, because the point is the *state they leave
/// behind*: the query generator is the read-back, and it already covers every
/// shape worth reading a mutated table with.
fn gen_mutations(rng: &mut Rng, tables: &[TableDef]) -> Vec<Mutation> {
    let n = if rng.pct(25) { 2 } else { 1 };
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let slot = rng.below(tables.len());
        let t = &tables[slot];
        // Which columns this statement assigns. Never `id`: this engine keys
        // the delete bitmap by it, so moving a row's key is a delete plus an
        // insert under a new key -- a case worth testing and not one to fold
        // into the general generator while BUG 8 makes it a silent no-op.
        let mut set = Vec::new();
        if !rng.pct(40) {
            let ncols = t.cols.len();
            for c in 1..ncols {
                // Two assignments at most, so the statement stays readable in a
                // reproducer.
                if set.len() < 2 && rng.pct(45) {
                    set.push(c);
                }
            }
            if set.is_empty() {
                set.push(1); // `k`, always present
            }
        }
        // The predicate reads EVERY column, including the ones this statement
        // assigns. That overlap used to be excluded because it was BUG 8 --
        // the predicate bound against a synthesized select list, so an
        // assignment shadowed the stored value and the statement matched
        // nothing. Now that a mutation binds against the table's own scope, the
        // overlap is the interesting case rather than the excluded one.
        let visible: Vec<(String, Ty)> =
            t.cols.iter().map(|c| (c.name.clone(), c.ty)).collect();
        let sc = Scope { cols: Vec::new(), bare: visible };
        let pred = if rng.pct(12) && !set.is_empty() {
            // No WHERE at all. Only for UPDATE: an unconditional DELETE empties
            // the table, and a case whose query then reads nothing tests the
            // comparator rather than the engine.
            None
        } else {
            Some(gen_pred(rng, &sc, 1))
        };
        let assign: Vec<(usize, E)> = set
            .into_iter()
            .map(|c| {
                let ty = t.cols[c].ty;
                // A non-nullable column must not receive a NULL, and any
                // expression over a nullable operand can produce one -- so the
                // expression forms are reserved for columns that can hold the
                // answer. sqlite would accept the NULL and granular would
                // reject it, which is a schema difference, not a bug.
                let e = if t.cols[c].nullable {
                    gen_scalar(rng, &sc, ty, 1)
                } else {
                    E::Lit(gen_value(rng, ty, false))
                };
                (c, e)
            })
            .collect();
        out.push(Mutation { slot, set: assign, pred, then_optimize: rng.pct(30) });
    }
    out
}

/// Columns visible to the query, as `(slot, index, type)`.
struct Scope {
    cols: Vec<(usize, usize, Ty)>,
    /// Unqualified names introduced by `USING`.
    bare: Vec<(String, Ty)>,
}

impl Scope {
    fn of(tables: &[TableDef], from: &From) -> Scope {
        let mut cols = Vec::new();
        let mut bare = Vec::new();
        let slots: &[usize] = match from {
            From::One(s) => std::slice::from_ref(s),
            From::Join(..) => &[0, 1],
        };
        for &s in slots {
            for (i, c) in tables[s].cols.iter().enumerate() {
                cols.push((s, i, c.ty));
            }
        }
        if let From::Join(_, JoinCon::Using(names)) = from {
            for n in names {
                let ty = tables[0].cols.iter().find(|c| &c.name == n).unwrap().ty;
                bare.push((n.clone(), ty));
            }
        }
        Scope { cols, bare }
    }

    /// True when some visible column has this type. Bare names count: a
    /// mutation's scope is bare-only, and reporting "no Int column" there would
    /// send `gen_atom_pred` down its all-literal fallback.
    fn has(&self, ty: Ty) -> bool {
        self.cols.iter().any(|(_, _, t)| *t == ty)
            || self.bare.iter().any(|(_, t)| *t == ty)
    }

    /// A random visible column of `ty`, or `None` if the schema has none.
    ///
    /// Count-then-index rather than collect-then-pick: this is the single most
    /// called function in the generator (every leaf of every expression of
    /// every one of tens of thousands of cases), and the obvious version built
    /// two throwaway `Vec`s per call.
    fn any_of(&self, rng: &mut Rng, ty: Ty) -> Option<E> {
        // Prefer the bare (coalesced) form sometimes -- it is the construct the
        // hand-run diff caught granular failing to bind at all.
        // Always, not sometimes, when there is nothing else: a mutation's scope
        // has no qualified columns at all (there is no FROM to qualify against),
        // so falling through would make every leaf a literal.
        let nbare = self.bare.iter().filter(|(_, t)| *t == ty).count();
        if nbare > 0 && (self.cols.is_empty() || rng.pct(35)) {
            let k = rng.below(nbare);
            let (n, t) = self.bare.iter().filter(|(_, t)| *t == ty).nth(k).unwrap();
            return Some(E::Bare(n.clone(), *t));
        }
        let ncols = self.cols.iter().filter(|(_, _, t)| *t == ty).count();
        if ncols == 0 {
            return None;
        }
        let k = rng.below(ncols);
        let (s, i, _) = self.cols.iter().filter(|(_, _, t)| *t == ty).nth(k).unwrap();
        Some(E::Col(*s, *i))
    }
}

fn gen_scalar(rng: &mut Rng, sc: &Scope, ty: Ty, depth: u32) -> E {
    if depth == 0 || rng.pct(35) {
        return match sc.any_of(rng, ty) {
            Some(c) if rng.pct(75) => c,
            _ => E::Lit(gen_value(rng, ty, false)),
        };
    }
    if rng.pct(22) {
        if let Some(c) = gen_call(rng, sc, ty, depth - 1) {
            return c;
        }
    }
    match ty {
        Ty::Int => match rng.below(6) {
            // The right operand is *signed*-valued (`gen_int_nonbool` emits no
            // comparison and no call, so never `Bool` and never `length`'s
            // `UInt64`). At most one operand can therefore be unsigned, which
            // is what keeps BUG 6 -- unsigned subtraction wrapping -- out of
            // the grammar. Mixed unsigned/signed promotes to `Int64` and both
            // engines agree.
            0 => E::Bin(
                *rng.pick(&["+", "-", "*"]),
                gen_scalar(rng, sc, Ty::Int, depth - 1).b(),
                gen_int_nonbool(rng, sc, depth - 1).b(),
            ),
            // Non-zero literal divisor: `x % 0` is NULL in SQLite and the
            // generator has no business probing granular's answer for it here.
            1 => E::Bin(
                "%",
                gen_scalar(rng, sc, Ty::Int, depth - 1).b(),
                E::Lit(Cell::Int(*rng.pick(&[2i64, 3, 5, -3]))).b(),
            ),
            2 => E::Cast(gen_scalar(rng, sc, Ty::Real, depth - 1).b(), Ty::Int),
            3 => gen_case(rng, sc, Ty::Int, depth - 1),
            // Arithmetic on two *booleans*, explicitly. This was BUG 4:
            // `promote(Bool, Bool)` was `Bool`, which truncated `(1=1)+(2=2)`
            // to `true` and killed `(1=2)-(1=1)` at execution with "-1 is not
            // a Bool". Fixed -- the pair widens to `Int64` -- so the shape
            // belongs in the grammar, and it gets its own arm rather than
            // being left to chance because two `gen_scalar(Ty::Int)` draws
            // both landing on a predicate is rare.
            4 => E::Bin(
                *rng.pick(&["+", "-", "*"]),
                gen_pred(rng, sc, depth - 1).b(),
                gen_pred(rng, sc, depth - 1).b(),
            ),
            _ => gen_pred(rng, sc, depth - 1),
        },
        Ty::Real => match rng.below(4) {
            0 => E::Bin(
                *rng.pick(&["+", "-", "*"]),
                gen_scalar(rng, sc, Ty::Real, depth - 1).b(),
                gen_scalar(rng, sc, Ty::Real, depth - 1).b(),
            ),
            // Only ever divided by a non-zero *real* literal: integer division
            // is a dialect difference, not a bug (KNOWN DIVERGENCES #2).
            1 => {
                let num = if rng.pct(50) { Ty::Int } else { Ty::Real };
                E::Bin(
                    "/",
                    gen_scalar(rng, sc, num, depth - 1).b(),
                    E::Lit(Cell::Real(*rng.pick(&[2.0f64, 4.0, -0.5, 8.0]))).b(),
                )
            }
            2 => E::Cast(gen_scalar(rng, sc, Ty::Int, depth - 1).b(), Ty::Real),
            _ => gen_case(rng, sc, Ty::Real, depth - 1),
        },
        Ty::Text => match rng.below(3) {
            0 => gen_case(rng, sc, Ty::Text, depth - 1),
            // Int->Text agrees; Real->Text does not (KNOWN DIVERGENCES #4), and
            // neither does Bool->Text: granular has a real Bool type and
            // renders 'true', SQLite has none and renders '1'. `gen_scalar`'s
            // Int arm can hand back a *comparison*, so the argument has to come
            // from the boolean-free generator or that difference leaks in.
            1 => E::Cast(gen_int_nonbool(rng, sc, depth - 1).b(), Ty::Text),
            _ => match sc.any_of(rng, Ty::Text) {
                Some(c) => c,
                None => E::Lit(gen_value(rng, Ty::Text, false)),
            },
        },
    }
}

/// Scalar functions spelled and behaving identically in both dialects.
///
/// What is deliberately NOT here, and why -- each was probed by hand against
/// both engines before being excluded:
///
///   * `round`  -- granular rounds half to even (ClickHouse's rule):
///     `round(2.5)` is 2 and `round(-2.5)` is -2. SQLite rounds half away from
///     zero: 3.0 and -3.0.
///   * `concat` -- SQLite's `concat()` *skips* NULL arguments, so
///     `concat(NULL,'b')` is `'b'`; granular propagates and returns NULL.
///   * `sqrt`/`ln`/`log` -- granular returns NaN for a negative argument,
///     SQLite returns NULL. Both are defensible; they are not the same.
///   * `least`/`greatest`, `position`, `char_length` -- no SQLite spelling.
fn gen_call(rng: &mut Rng, sc: &Scope, ty: Ty, depth: u32) -> Option<E> {
    let arg = |rng: &mut Rng, t: Ty| gen_scalar(rng, sc, t, depth);
    Some(match ty {
        Ty::Text => match rng.below(6) {
            0 => E::Call("lower", vec![arg(rng, Ty::Text)]),
            1 => E::Call("upper", vec![arg(rng, Ty::Text)]),
            2 => E::Call("trim", vec![arg(rng, Ty::Text)]),
            3 => E::Call(
                "replace",
                // A non-empty needle: SQLite's `replace(x, '', y)` returns `x`
                // unchanged and granular's behaviour for it is not pinned by
                // anything, so it is not a fair comparison.
                vec![
                    arg(rng, Ty::Text),
                    E::Lit(Cell::Text((*rng.pick(&["a", "b", "ab"])).to_string())),
                    E::Lit(Cell::Text((*rng.pick(&["X", "", "yy"])).to_string())),
                ],
            ),
            4 => E::Call(
                "substring",
                vec![
                    arg(rng, Ty::Text),
                    // 1-based, positive only: a negative start counts from the
                    // end in SQLite and is not defined the same way elsewhere.
                    E::Lit(Cell::Int(rng.range(1, 3) as i64)),
                    E::Lit(Cell::Int(rng.range(0, 3) as i64)),
                ],
            ),
            _ => E::Call(
                *rng.pick(&["coalesce", "ifnull", "nullif"]),
                vec![arg(rng, Ty::Text), E::Lit(gen_value(rng, Ty::Text, false))],
            ),
        },
        Ty::Int => match rng.below(4) {
            0 => E::Call("abs", vec![arg(rng, Ty::Int)]),
            1 => E::Call("sign", vec![arg(rng, Ty::Int)]),
            2 => E::Call("length", vec![arg(rng, Ty::Text)]),
            _ => E::Call(
                *rng.pick(&["coalesce", "ifnull", "nullif"]),
                vec![arg(rng, Ty::Int), E::Lit(gen_value(rng, Ty::Int, false))],
            ),
        },
        Ty::Real => match rng.below(4) {
            0 => E::Call("abs", vec![arg(rng, Ty::Real)]),
            1 => E::Call("floor", vec![arg(rng, Ty::Real)]),
            2 => E::Call("ceil", vec![arg(rng, Ty::Real)]),
            _ => E::Call(
                *rng.pick(&["coalesce", "ifnull", "nullif"]),
                vec![arg(rng, Ty::Real), E::Lit(gen_value(rng, Ty::Real, false))],
            ),
        },
    })
}

/// Integer-valued and never boolean-valued -- `gen_scalar(Ty::Int, ..)` may
/// return a comparison, which is an integer in SQLite but a `Bool` in granular.
/// That only matters where the *type* leaks into the answer, i.e. under a text
/// CAST, so this exists only for that call site.
fn gen_int_nonbool(rng: &mut Rng, sc: &Scope, depth: u32) -> E {
    if depth == 0 || rng.pct(50) {
        return match sc.any_of(rng, Ty::Int) {
            Some(c) if rng.pct(75) => c,
            _ => E::Lit(gen_value(rng, Ty::Int, false)),
        };
    }
    match rng.below(3) {
        0 => E::Bin(
            *rng.pick(&["+", "-", "*"]),
            gen_int_nonbool(rng, sc, depth - 1).b(),
            gen_int_nonbool(rng, sc, depth - 1).b(),
        ),
        1 => E::Bin(
            "%",
            gen_int_nonbool(rng, sc, depth - 1).b(),
            E::Lit(Cell::Int(*rng.pick(&[2i64, 3, 5, -3]))).b(),
        ),
        _ => E::Cast(gen_scalar(rng, sc, Ty::Real, depth - 1).b(), Ty::Int),
    }
}

fn gen_case(rng: &mut Rng, sc: &Scope, ty: Ty, depth: u32) -> E {
    let arms = rng.range(1, 2);
    let mut v = Vec::with_capacity(arms);
    for _ in 0..arms {
        v.push((gen_pred(rng, sc, depth), gen_scalar(rng, sc, ty, depth)));
    }
    // A CASE with no ELSE yields NULL on fall-through in both dialects, which
    // is worth generating: it is the cheapest way to inject NULLs into an
    // otherwise non-null expression tree.
    let els = if rng.pct(70) {
        Some(gen_scalar(rng, sc, ty, depth).b())
    } else {
        None
    };
    E::Case(v, els)
}

/// A boolean-valued expression. Comparisons are always same-category (numeric
/// with numeric, text with text): SQLite's type affinity gives cross-category
/// comparisons rules granular has never claimed to implement, so mixing them
/// would generate noise instead of evidence.
fn gen_pred(rng: &mut Rng, sc: &Scope, depth: u32) -> E {
    if depth == 0 {
        return gen_atom_pred(rng, sc);
    }
    match rng.below(8) {
        0 => E::Bin("AND", gen_pred(rng, sc, depth - 1).b(), gen_pred(rng, sc, depth - 1).b()),
        1 => E::Bin("OR", gen_pred(rng, sc, depth - 1).b(), gen_pred(rng, sc, depth - 1).b()),
        2 => E::Not(gen_pred(rng, sc, depth - 1).b()),
        _ => gen_atom_pred(rng, sc),
    }
}

fn gen_atom_pred(rng: &mut Rng, sc: &Scope) -> E {
    let mut ty = *rng.pick(&[Ty::Int, Ty::Int, Ty::Real, Ty::Text]);
    // If the schema happens to have no column of that type, fall back to Int
    // rather than to a *literal* operand. Every generated schema has `id` and
    // `k`, so Int is always available. This is not cosmetic: an all-literal
    // predicate is constant-folded, and the folding path does not implement
    // three-valued logic (BUG 5), so the literal fallback quietly emitted
    // `WHERE 2.25 NOT IN (NULL)` -- twelve mismatches in one soak, all of them
    // the same already-known bug arriving through a hole in this function.
    if !sc.has(ty) {
        ty = Ty::Int;
    }
    // A quarter of the time the left side is a *computed* expression rather
    // than a bare column. That is the difference between exercising the zone-map
    // and index fast paths (which only fire on a plain column) and exercising
    // the general evaluator, and a predicate generator that never leaves the
    // fast path tests half the engine. Integers go through the boolean-free
    // generator: comparing a `Bool` against an integer literal is a
    // cross-category comparison, not a wider one.
    let widened = rng.pct(25);
    let column = sc.any_of(rng, ty);
    let lhs = match (widened, column) {
        (true, _) => {
            let e = match ty {
                Ty::Int => gen_int_nonbool(rng, sc, 1),
                other => gen_scalar(rng, sc, other, 1),
            };
            // Every atom must mention a column. An all-literal predicate is
            // constant-folded away by the planner, and granular's folding path
            // does not implement three-valued logic (BUG 5) -- so a generator
            // that emits constant predicates is not testing the engine's
            // predicate evaluation at all, it is testing a known-broken
            // shortcut. Anchoring on a column keeps every atom on the real
            // vectorized path.
            if mentions_column(&e) {
                e
            } else {
                match sc.any_of(rng, ty) {
                    Some(c) => c,
                    None => e,
                }
            }
        }
        (false, Some(c)) => c,
        // Unreachable given the `sc.has(ty)` fallback above -- kept as a
        // belt-and-braces arm rather than an `unwrap`, because the failure mode
        // if the invariant ever breaks is a constant predicate and a phantom
        // BUG 5 report, not a panic that would point at the real cause.
        (false, None) => E::Lit(gen_value(rng, ty, false)),
    };
    match rng.below(10) {
        0 => E::IsNull(lhs.b(), rng.pct(50)),
        // Text BETWEEN is in the intersection: both engines order strings by
        // bytes, and the pool is ASCII.
        1 => E::Between(
            lhs.b(),
            E::Lit(gen_value(rng, ty, false)).b(),
            E::Lit(gen_value(rng, ty, false)).b(),
            rng.pct(30),
        ),
        2 => {
            // NULL *is* allowed in the list -- but only because the operand is
            // a column, so the planner cannot fold the whole thing. granular's
            // vectorized IN implements three-valued logic correctly and agrees
            // with SQLite here; only its constant-folding path does not
            // (BUG 5).
            let n = rng.range(1, 3);
            let list = (0..n)
                .map(|_| {
                    let nullable = rng.pct(20);
                    E::Lit(gen_value(rng, ty, nullable))
                })
                .collect();
            E::In(lhs.b(), list, rng.pct(30))
        }
        // LIKE's operand is deliberately *not* the widened `lhs`. The generator
        // keeps text lowercase so SQLite's case-insensitive LIKE is
        // unobservable (KNOWN DIVERGENCES #3) -- but `upper(s)` and
        // `replace(s,'a','X')` manufacture uppercase, and then the difference
        // is observable again. A 60000-case soak found exactly that:
        // `upper('ab') LIKE '%a%'` is false in granular and true in SQLite.
        // The invariant is "nothing case-shifted ever reaches a LIKE", and the
        // only way to hold it is to build the operand here.
        3 if ty == Ty::Text => {
            let operand = match sc.any_of(rng, Ty::Text) {
                Some(c) => c,
                None => E::Lit(gen_value(rng, Ty::Text, false)),
            };
            E::Like(operand.b(), (*rng.pick(&PATTERNS)).to_string(), rng.pct(30))
        }
        _ => {
            let op = *rng.pick(&["=", "<>", "<", "<=", ">", ">="]);
            let rhs = match sc.any_of(rng, ty) {
                Some(c) if rng.pct(40) => c,
                _ => E::Lit(gen_value(rng, ty, false)),
            };
            E::Bin(op, lhs.b(), rhs.b())
        }
    }
}

/// True when the expression reads a column, i.e. the planner cannot fold it to
/// a literal. See the note in `gen_atom_pred`.
fn mentions_column(e: &E) -> bool {
    let mut hit = false;
    e.walk(&mut |x| hit |= matches!(x, E::Col(..) | E::Bare(..)));
    hit
}

fn gen_from(rng: &mut Rng, tables: &[TableDef]) -> From {
    // A join between two big tables is a cartesian blow-up that measures
    // nothing; big tables get scanned, not joined.
    if rng.pct(40) || tables.iter().any(|t| t.rows.len() > JOINABLE_ROWS) {
        return From::One(rng.below(2));
    }
    let op = *rng.pick(&[JoinOp::Inner, JoinOp::Left, JoinOp::Right, JoinOp::Full, JoinOp::Cross]);
    let con = if op == JoinOp::Cross {
        JoinCon::None
    } else if rng.pct(50) {
        let names: Vec<String> = if rng.pct(50) {
            vec!["id".into()]
        } else if rng.pct(50) {
            vec!["k".into()]
        } else {
            vec!["id".into(), "k".into()]
        };
        JoinCon::Using(names)
    } else {
        let l = if rng.pct(50) { 0 } else { 1 };
        JoinCon::On(E::Bin(
            *rng.pick(&["=", "<", ">="]),
            E::Col(0, l).b(),
            E::Col(1, rng.below(2)).b(),
        ))
    };
    From::Join(op, con)
}

fn gen_query(rng: &mut Rng, tables: &[TableDef]) -> Query {
    let from = gen_from(rng, tables);
    let sc = Scope::of(tables, &from);
    let filter = if rng.pct(60) {
        Some(gen_pred(rng, &sc, 2))
    } else {
        None
    };

    // `SELECT *`: arity and column order are themselves under test, so there is
    // no ordinal list to order by and the comparison is a multiset comparison.
    if rng.pct(12) {
        return Query {
            from,
            star: true,
            distinct: rng.pct(20),
            items: Vec::new(),
            filter,
            group_by: Vec::new(),
            having: None,
            set_tail: None,
            order: Vec::new(),
            limit: None,
            offset: 0,
        };
    }

    let aggregate = rng.pct(40);
    let mut items = Vec::new();
    let mut group_by = Vec::new();
    let mut having = None;

    if aggregate {
        // Select list is exactly (group keys, aggregates). SQLite tolerates a
        // bare non-grouped column; ClickHouse does not, and the standard does
        // not, so it is out of the intersection.
        let nkeys = rng.range(0, 2);
        for _ in 0..nkeys {
            let ty = *rng.pick(&[Ty::Int, Ty::Real, Ty::Text]);
            if let Some(c) = sc.any_of(rng, ty) {
                group_by.push(items.len());
                items.push(c);
            }
        }
        let naggs = rng.range(1, 3);
        for _ in 0..naggs {
            items.push(gen_agg(rng, &sc).0);
        }
        if rng.pct(30) {
            // The comparand's type has to follow the aggregate's: `min(<text>)`
            // against an integer is a cross-category comparison, which is out
            // of the intersection (SQLite has affinity rules for it, granular
            // rejects it outright) and would show up as harness noise.
            let (a, ty) = gen_agg(rng, &sc);
            // A *global* `sum` in HAVING used to be excluded: it can see zero
            // rows, where BUG 2 answered 0 instead of NULL, and that flipped
            // the predicate and changed the *row count* rather than a cell.
            // BUG 2 is fixed (empty `sum` is NULL either way), so the shape is
            // back -- and it is the one that exercises the fix hardest.
            having = Some(E::Bin(
                *rng.pick(&["=", "<>", "<", ">=", ">"]),
                a.b(),
                E::Lit(gen_value(rng, ty, false)).b(),
            ));
        }
    } else {
        let n = rng.range(1, 4);
        for _ in 0..n {
            let ty = *rng.pick(&[Ty::Int, Ty::Int, Ty::Real, Ty::Text]);
            items.push(gen_scalar(rng, &sc, ty, 2));
        }
    }

    let order = if rng.pct(70) {
        total_order(rng, items.len())
    } else {
        Vec::new()
    };
    let (limit, offset) = if !order.is_empty() && rng.pct(30) {
        (Some(rng.range(0, 4)), rng.range(0, 2))
    } else {
        (None, 0)
    };

    // DISTINCT is only offered on non-aggregate projections: `SELECT DISTINCT
    // g, count(*) ... GROUP BY g` is legal but redundant in both engines, so it
    // buys no coverage.
    let distinct = !aggregate && rng.pct(20);
    let mut q =
        Query { from, star: false, distinct, items, filter, group_by, having, set_tail: None, order, limit, offset };

    // The second branch of a set operation is the first branch with a
    // different filter. Cloning guarantees identical arity and column types,
    // which is what both dialects require, and a *different* filter is what
    // makes UNION's dedup and EXCEPT/INTERSECT's matching do real work --
    // NULL-vs-NULL equality in a set operation is its own semantics, distinct
    // from `=`, and both engines have to get it right.
    // A *global* `sum` behind a set operation used to be excluded: it can see
    // zero rows, BUG 2 answered 0 there instead of NULL, and unlike a plain
    // wrong cell that row then took part in UNION's dedup and sorted to a
    // different position than sqlite's NULL row, shifting everything after it.
    // BUG 2 is fixed, so the combination is generated again.
    if rng.pct(12) {
        let mut tail = q.clone();
        tail.order.clear();
        tail.limit = None;
        tail.offset = 0;
        tail.filter = if rng.pct(70) {
            Some(gen_pred(rng, &sc, 1))
        } else {
            None
        };
        // EXCEPT and INTERSECT parse but are not implemented in granular -- it
        // says so, clearly, at bind time. That is a capability gap, not a wrong
        // answer, so the generator stays on the two set operations both engines
        // actually evaluate. Pinned by `known_divergences_still_reproduce`.
        let op = *rng.pick(&["UNION ALL", "UNION"]);
        q.set_tail = Some((op, Box::new(tail)));
    }
    q
}

/// Returns the aggregate and the category of its result, which the caller needs
/// to build a same-category comparand for `HAVING`.
fn gen_agg(rng: &mut Rng, sc: &Scope) -> (E, Ty) {
    if rng.pct(25) {
        return (E::Agg("count", None, false), Ty::Int);
    }
    let name = *rng.pick(&["count", "sum", "avg", "min", "max"]);
    // sum/avg need a number; min/max/count take anything.
    let ty = if name == "sum" || name == "avg" {
        *rng.pick(&[Ty::Int, Ty::Real])
    } else {
        *rng.pick(&[Ty::Int, Ty::Real, Ty::Text])
    };
    let arg = match sc.any_of(rng, ty) {
        Some(c) => c,
        None => E::Lit(gen_value(rng, ty, false)),
    };
    let result = match name {
        "count" => Ty::Int,
        "avg" => Ty::Real,
        _ => ty,
    };
    // `min`/`max` reject DISTINCT in granular -- with good reason, it is a
    // no-op for them -- so the flag is only offered to the three aggregates
    // where it changes the answer.
    let distinct = matches!(name, "count" | "sum" | "avg") && rng.pct(25);
    (E::Agg(name, Some(arg.b()), distinct), result)
}

/// A random ordering prefix followed by every remaining output ordinal, so the
/// full row is a tiebreaker chain. Rows that still tie are byte-identical, so
/// `LIMIT` slices a sequence both engines must agree on.
fn total_order(rng: &mut Rng, n: usize) -> Vec<(usize, bool, bool)> {
    let mut ords: Vec<usize> = (1..=n).collect();
    // Fisher-Yates over a prefix only; the tail stays in natural order so the
    // shrinker's output is readable.
    let prefix = rng.range(0, n);
    for i in 0..prefix {
        let j = i + rng.below(n - i);
        ords.swap(i, j);
    }
    ords.into_iter()
        .enumerate()
        .map(|(i, o)| {
            if i < prefix {
                (o, rng.pct(50), rng.pct(50))
            } else {
                (o, true, true)
            }
        })
        .collect()
}

fn gen_case_at(seed: u64) -> Case {
    let mut rng = Rng::new(seed);
    // A fifth of cases mutate. Not more: a mutation case has to give up the
    // colliding `id` domain (see `TableDef::keyed`), and colliding sort keys
    // are what BUG 1 lives in, so trading too much of that away to test DELETE
    // would be a net loss of coverage.
    let mutating = rng.pct(20);
    let tables = gen_tables(&mut rng, mutating);
    let mutations = if mutating { gen_mutations(&mut rng, &tables) } else { Vec::new() };
    let query = gen_query(&mut rng, &tables);
    Case { seed, tables, mutations, query }
}

// ------------------------------------------------------------------ granular driver

thread_local! {
    /// Set only while a thread is inside `catch_unwind` around the engine.
    static QUIET: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A panic inside the engine is a finding, not a reason to lose the whole run,
/// so it is caught and reported like any other divergence. The default hook
/// would bury the reproducer under one backtrace per case, so it is suppressed
/// -- but *only* for the thread that is currently inside the engine. The naive
/// version of this (a hook that swallows everything) also swallows the
/// harness's own assertion messages, which cost a full debugging cycle here:
/// the first real run reported "12 mismatches" and printed not one of them.
fn silence_panics() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if QUIET.with(|q| q.get()) {
                return;
            }
            prev(info);
        }));
    });
}

fn panic_text(e: Box<dyn Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic>".to_string()
    }
}

fn cell_of_value(v: &Value) -> Cell {
    match v {
        Value::Null => Cell::Null,
        // SQLite has no boolean type; a comparison there yields 1/0. Unifying
        // here is a rendering decision, not a semantic one.
        Value::Bool(b) => Cell::Int(*b as i64),
        Value::UInt(u) => Cell::Int(*u as i64),
        Value::Int(i) => Cell::Int(*i),
        Value::Float(f) => Cell::Real(*f),
        Value::Str(s) => Cell::Text(s.to_string()),
        Value::Date(d) => Cell::Int(*d as i64),
        Value::DateTime(t) => Cell::Int(*t),
        // SQLite has no exact decimal type, so a decimal can only ever be
        // diffed against a REAL and the comparison is necessarily approximate.
        // The generator does not emit `Decimal` columns for exactly that reason
        // (see `Ty`), so this arm exists to keep the match total rather than to
        // carry traffic; `Decimal64` is covered by the property tests in
        // src/types/value.rs and src/exec/functions/scalar.rs instead.
        Value::Decimal(..) => Cell::Real(v.as_f64().unwrap_or(f64::NAN)),
    }
}

type Outcome = Result<Vec<Vec<Cell>>, String>;

fn run_granular(case: &Case) -> Outcome {
    silence_panics();
    QUIET.with(|q| q.set(true));
    let r = catch_unwind(AssertUnwindSafe(|| -> Outcome {
        let mut s = Session::in_memory();
        for stmt in case.setup(Dialect::Granular) {
            s.execute(&stmt).map_err(|e| format!("{e}"))?;
        }
        let rs = s.query(&case.select(Dialect::Granular)).map_err(|e| format!("{e}"))?;
        Ok(rs
            .to_values()
            .iter()
            .map(|r| r.iter().map(cell_of_value).collect())
            .collect())
    }));
    QUIET.with(|q| q.set(false));
    match r {
        Ok(v) => v,
        Err(p) => Err(format!("PANIC: {}", panic_text(p))),
    }
}

// ------------------------------------------------------------------ sqlite driver

/// Probed once. It used to probe on every call, which meant an extra process
/// spawn per batch -- a third of the harness's process budget spent asking a
/// question whose answer cannot change mid-run.
fn sqlite_path() -> Option<&'static str> {
    static FOUND: std::sync::OnceLock<Option<&'static str>> = std::sync::OnceLock::new();
    *FOUND.get_or_init(|| {
        let runs = |p: &str| {
            Command::new(p)
                .arg("-version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        };
        // An explicit override is authoritative: pointing it at a path that
        // does not exist is how the skip-without-sqlite3 path gets tested on a
        // machine that *does* have sqlite3, which is every machine that would
        // notice the skip path being broken.
        if let Ok(p) = std::env::var("GRANULAR_DIFF_SQLITE") {
            return runs(&p).then(|| &*Box::leak(p.into_boxed_str()));
        }
        // `/usr/bin/sqlite3` on macOS; fall back to PATH so a Linux CI box works.
        ["/usr/bin/sqlite3", "sqlite3"].into_iter().find(|p| runs(p))
    })
}

/// Run a script through `sqlite3 :memory:`, returning `(stdout, stderr)`.
fn sqlite_raw(script: &str) -> Result<(String, String), String> {
    let bin = sqlite_path().ok_or("sqlite3 not found")?;
    let mut child = Command::new(bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn sqlite3: {e}"))?;
    // The write happens on a helper thread: a large result set can fill the
    // stdout pipe while we are still writing stdin, and the two-pipe deadlock
    // that causes is exactly the kind of flake a test harness must not have.
    let mut stdin = child.stdin.take().expect("piped");
    let owned = script.to_string();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(owned.as_bytes());
    });
    let out = child.wait_with_output().map_err(|e| format!("wait sqlite3: {e}"))?;
    let _ = writer.join();
    Ok((
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    ))
}

const PREAMBLE: &str = "\
.bail on
.mode quote
.separator \"\\t\"
.headers off
";

/// Why `.mode quote` and not the more obvious `.mode tabs` + `.nullvalue NULL`:
/// TSV cannot tell `NULL` from the three-character string `'NULL'`, nor the
/// integer `12` from the text `'12'`. Both distinctions matter here -- the
/// second one is how a wrong result type would slip past unnoticed. Quote mode
/// is unambiguous: bare `NULL`, bare digits for INTEGER, a `.`/`e` for REAL,
/// and `'...'` with `''` escaping for TEXT.
fn parse_quote_row(line: &str) -> Vec<Cell> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i <= b.len() {
        if i == b.len() {
            out.push(Cell::Null);
            break;
        }
        if b[i] == b'\'' {
            let mut s = String::new();
            i += 1;
            loop {
                if i >= b.len() {
                    break;
                }
                if b[i] == b'\'' {
                    if i + 1 < b.len() && b[i + 1] == b'\'' {
                        s.push('\'');
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                s.push(b[i] as char);
                i += 1;
            }
            out.push(Cell::Text(s));
        } else {
            let start = i;
            while i < b.len() && b[i] != b'\t' {
                i += 1;
            }
            let tok = &line[start..i];
            out.push(parse_scalar(tok));
        }
        if i < b.len() && b[i] == b'\t' {
            i += 1;
            if i == b.len() {
                out.push(Cell::Null);
                break;
            }
        } else {
            break;
        }
    }
    out
}

fn parse_scalar(tok: &str) -> Cell {
    if tok == "NULL" {
        return Cell::Null;
    }
    if !tok.contains('.') && !tok.contains('e') && !tok.contains('E') {
        if let Ok(i) = tok.parse::<i64>() {
            return Cell::Int(i);
        }
    }
    match tok {
        "Inf" => return Cell::Real(f64::INFINITY),
        "-Inf" => return Cell::Real(f64::NEG_INFINITY),
        _ => {}
    }
    match tok.parse::<f64>() {
        Ok(f) => Cell::Real(f),
        // Anything else is a shape the harness does not model (a BLOB, say);
        // surfacing it as text makes it a loud mismatch rather than a silent
        // coercion.
        Err(_) => Cell::Text(tok.to_string()),
    }
}

/// Run one case on its own. Slow (one process) but unambiguous: `.bail on`
/// means a non-empty stderr is that case's error.
fn sqlite_one(case: &Case) -> Outcome {
    let mut script = String::from(PREAMBLE);
    script.push_str(&case.script(Dialect::Sqlite));
    let (out, err) = sqlite_raw(&script)?;
    if !err.trim().is_empty() {
        return Err(err.trim().to_string());
    }
    Ok(out.lines().map(parse_quote_row).collect())
}

/// Run a batch in one process. `.bail on` aborts the whole script at the first
/// error, so the caller must fall back to `sqlite_one` for the batch when that
/// happens -- errors are rare enough (the generator stays inside the
/// intersection) that the amortized process count is still ~1/BATCH.
fn sqlite_batch(cases: &[Case]) -> Option<Vec<Vec<Vec<Cell>>>> {
    let mut script = String::from(PREAMBLE);
    for c in cases {
        for t in &c.tables {
            let _ = writeln!(script, "DROP TABLE IF EXISTS {};", t.name);
        }
        script.push_str(&c.script(Dialect::Sqlite));
        let _ = writeln!(script, "SELECT '{SENTINEL}';");
    }
    let (out, err) = sqlite_raw(&script).ok()?;
    if !err.trim().is_empty() {
        return None;
    }
    let marker = format!("'{}'", SENTINEL.replace('\'', "''"));
    let mut all = Vec::with_capacity(cases.len());
    let mut cur = Vec::new();
    for line in out.lines() {
        if line == marker {
            all.push(std::mem::take(&mut cur));
            continue;
        }
        cur.push(parse_quote_row(line));
    }
    if all.len() != cases.len() {
        return None;
    }
    Some(all)
}

// ------------------------------------------------------------------ comparison

#[derive(Debug)]
enum Diff {
    /// One engine answered and the other refused.
    OnlyOneRan { granular: Outcome, sqlite: Outcome },
    RowCount { g: usize, s: usize },
    Row { at: usize, g: Vec<Cell>, s: Vec<Cell> },
}

impl Cell {
    fn is_null(&self) -> bool {
        matches!(self, Cell::Null)
    }
    fn is_zero(&self) -> bool {
        matches!(self, Cell::Int(0)) || matches!(self, Cell::Real(f) if *f == 0.0)
    }
}

fn compare(case: &Case, g: Outcome, s: Outcome) -> Option<Diff> {
    match (&g, &s) {
        // Both refused: they agree that the query is not meaningful. That is
        // not evidence of anything, and chasing it would drown the run in
        // parser-message noise.
        (Err(_), Err(_)) => None,
        (Err(_), Ok(_)) | (Ok(_), Err(_)) => Some(Diff::OnlyOneRan { granular: g, sqlite: s }),
        (Ok(gr), Ok(sr)) => {
            let (mut a, mut b) = (gr.clone(), sr.clone());
            if case.query.order.is_empty() {
                // No ORDER BY means no defined order in *either* engine.
                canonical_sort(&mut a);
                canonical_sort(&mut b);
            }
            if a.len() != b.len() {
                return Some(Diff::RowCount { g: a.len(), s: b.len() });
            }
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                if !rows_equal(x, y) {
                    return Some(Diff::Row { at: i, g: x.clone(), s: y.clone() });
                }
            }
            None
        }
    }
}

fn diverges(case: &Case) -> bool {
    compare(case, run_granular(case), sqlite_one(case)).is_some()
}

// ------------------------------------------------------------------ shrinking

/// Greedy delta-debugging to a fixpoint: every candidate is a strictly smaller
/// case, and it is kept only if it still disagrees. Bounded because each
/// candidate costs a `sqlite3` process.
fn shrink(mut case: Case, budget: &mut u32) -> Case {
    let mut progress = true;
    while progress && *budget > 0 {
        progress = false;
        for cand in reductions(&case) {
            if *budget == 0 {
                break;
            }
            *budget -= 1;
            if diverges(&cand) {
                case = cand;
                progress = true;
                break;
            }
        }
    }
    case
}

fn reductions(case: &Case) -> Vec<Case> {
    let mut out = Vec::new();
    let q = &case.query;

    // Mutations first: dropping one is the largest single simplification a
    // mutation case admits, and if the divergence survives it the mutation was
    // never the point. Then narrow what survives -- an `UPDATE` reduces to a
    // one-column `SET`, and any statement reduces to an unconditional one.
    for i in 0..case.mutations.len() {
        let mut c = case.clone();
        c.mutations.remove(i);
        out.push(c);
    }
    for i in 0..case.mutations.len() {
        if case.mutations[i].then_optimize {
            let mut c = case.clone();
            c.mutations[i].then_optimize = false;
            out.push(c);
        }
        if case.mutations[i].set.len() > 1 {
            let mut c = case.clone();
            c.mutations[i].set.truncate(1);
            out.push(c);
        }
        if case.mutations[i].pred.is_some() && !case.mutations[i].set.is_empty() {
            let mut c = case.clone();
            c.mutations[i].pred = None;
            out.push(c);
        }
    }

    if q.limit.is_some() {
        let mut c = case.clone();
        c.query.limit = None;
        c.query.offset = 0;
        out.push(c);
    }
    if q.offset > 0 {
        let mut c = case.clone();
        c.query.offset = 0;
        out.push(c);
    }
    if !q.order.is_empty() && q.limit.is_none() {
        // Dropping ORDER BY switches the comparison to multiset; if the
        // disagreement survives that, it was never about ordering.
        let mut c = case.clone();
        c.query.order.clear();
        out.push(c);
    }
    if q.having.is_some() {
        let mut c = case.clone();
        c.query.having = None;
        out.push(c);
    }
    if q.distinct {
        let mut c = case.clone();
        c.query.distinct = false;
        out.push(c);
    }
    // Drop the set operation, or keep only its second branch.
    if let Some((_, tail)) = &q.set_tail {
        let mut c = case.clone();
        c.query.set_tail = None;
        out.push(c);
        let mut c = case.clone();
        let (order, limit, offset) = (q.order.clone(), q.limit, q.offset);
        c.query = (**tail).clone();
        c.query.order = order;
        c.query.limit = limit;
        c.query.offset = offset;
        out.push(c);
    }
    if q.filter.is_some() {
        let mut c = case.clone();
        c.query.filter = None;
        out.push(c);
    }
    // Peel one side off a top-level AND/OR in the WHERE clause.
    if let Some(E::Bin(op, l, r)) = &q.filter {
        if *op == "AND" || *op == "OR" {
            for side in [l, r] {
                let mut c = case.clone();
                c.query.filter = Some((**side).clone());
                out.push(c);
            }
        }
    }
    // Simplify one expression in place.
    for (i, e) in q.items.iter().enumerate() {
        for s in simplifications(e) {
            let mut c = case.clone();
            c.query.items[i] = s;
            out.push(c);
        }
    }
    if let Some(f) = &q.filter {
        for s in simplifications(f) {
            let mut c = case.clone();
            c.query.filter = Some(s);
            out.push(c);
        }
    }
    // Drop a select item. Group keys stay: removing one changes the query's
    // meaning rather than shrinking it. Skipped while a set operation is
    // present, because the two branches must keep identical arity and the
    // "drop the set tail" reduction above gets there first anyway.
    if q.items.len() > 1 && !q.star && q.set_tail.is_none() {
        for i in 0..q.items.len() {
            if q.group_by.contains(&i) {
                continue;
            }
            let mut c = case.clone();
            c.query.items.remove(i);
            c.query.group_by = q.group_by.iter().map(|&g| if g > i { g - 1 } else { g }).collect();
            let n = c.query.items.len();
            c.query.order.retain(|(o, _, _)| *o != i + 1);
            for (o, _, _) in c.query.order.iter_mut() {
                if *o > i + 1 {
                    *o -= 1;
                }
            }
            if c.query.order.len() > n {
                continue;
            }
            out.push(c);
        }
    }
    // Simplify the FROM clause. Also deferred behind the set-tail reduction:
    // the tail's own FROM would still name the slot this drops.
    if let (From::Join(op, con), None) = (&q.from, &q.set_tail) {
        if *op != JoinOp::Inner && !matches!(con, JoinCon::None) {
            let mut c = case.clone();
            c.query.from = From::Join(JoinOp::Inner, con.clone());
            out.push(c);
        }
        if let JoinCon::Using(names) = con {
            if names.len() > 1 {
                for i in 0..names.len() {
                    let mut n = names.clone();
                    n.remove(i);
                    let mut c = case.clone();
                    c.query.from = From::Join(*op, JoinCon::Using(n));
                    out.push(c);
                }
            }
        }
        for slot in [0usize, 1] {
            let other = 1 - slot;
            if q.star || !matches!(con, JoinCon::None | JoinCon::Using(_)) {
                continue;
            }
            // Bare (USING) names lose their meaning without the join, and a
            // reference to the dropped side stops resolving.
            let mut blocked = false;
            q.for_each_expr(&mut |e| {
                e.walk(&mut |x| blocked |= matches!(x, E::Bare(..)));
                blocked |= e.refs_slot(other);
            });
            if blocked {
                continue;
            }
            let mut c = case.clone();
            c.query.from = From::One(slot);
            out.push(c);
        }
    }
    // Simplify the sort key. `ORDER BY id` is the shape a reader expects, so
    // reduce to it whenever the rows still allow it -- a reproducer that says
    // `ORDER BY (id, k)` invites the reader to suspect the composite key when
    // the composite key is not the point.
    for t in 0..case.tables.len() {
        if case.tables[t].sort == SortKey::Id {
            continue;
        }
        let ids: Vec<i64> = case.tables[t]
            .rows
            .iter()
            .filter_map(|r| match r[0] {
                Cell::Int(i) => Some(i),
                _ => None,
            })
            .collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != ids.len() {
            continue; // duplicates would hit BUG 1 and change the answer
        }
        let mut c = case.clone();
        c.tables[t].sort = SortKey::Id;
        out.push(c);
    }
    // Halve the row set before picking rows off one at a time. Without this a
    // 8200-row case burns the whole shrink budget removing eight thousand rows
    // individually and never reaches the query; with it, seven halvings get to
    // a readable size first.
    for t in 0..case.tables.len() {
        let n = case.tables[t].rows.len();
        if n < 4 {
            continue;
        }
        for keep in [(0, n / 2), (n / 2, n)] {
            let mut c = case.clone();
            c.tables[t].rows = case.tables[t].rows[keep.0..keep.1].to_vec();
            c.tables[t].chunk = c.tables[t].chunk.min(c.tables[t].rows.len().max(1));
            out.push(c);
        }
    }
    // Then one row at a time, but only once the table is small enough that the
    // candidate list (one full `Case` clone each) is cheap.
    for t in 0..case.tables.len() {
        if case.tables[t].rows.len() > 32 {
            continue;
        }
        for r in 0..case.tables[t].rows.len() {
            let mut c = case.clone();
            c.tables[t].rows.remove(r);
            c.tables[t].chunk = c.tables[t].chunk.min(c.tables[t].rows.len().max(1));
            out.push(c);
        }
    }
    // Collapse the load back to a single INSERT and drop OPTIMIZE, so the
    // reproducer only mentions them when they matter.
    for t in 0..case.tables.len() {
        if case.tables[t].chunk < case.tables[t].rows.len() {
            let mut c = case.clone();
            c.tables[t].chunk = usize::MAX;
            out.push(c);
        }
        if case.tables[t].optimize {
            let mut c = case.clone();
            c.tables[t].optimize = false;
            out.push(c);
        }
    }
    // Drop an unreferenced column. `id` and `k` stay: they are the sort key,
    // and `k` is also the only column a USING join can name besides `id`.
    for t in 0..case.tables.len() {
        // A mutation names columns by bare name and holds assignment targets by
        // index, so both would be invalidated by a shift. Mutation cases have
        // at most four columns anyway; they shrink through the earlier steps.
        if case.mutations.iter().any(|m| m.slot == t) {
            continue;
        }
        for col in (2..case.tables[t].cols.len()).rev() {
            if case.query.any_star() {
                continue;
            }
            let mut referenced = false;
            case.query.for_each_expr(&mut |e| referenced |= e.refs_col(t, col));
            if referenced {
                continue;
            }
            let mut c = case.clone();
            c.tables[t].cols.remove(col);
            for row in &mut c.tables[t].rows {
                row.remove(col);
            }
            c.query.for_each_expr_mut(&mut |e| e.shift_cols(t, col));
            out.push(c);
        }
    }
    out
}

fn is_comparison(op: &str) -> bool {
    matches!(op, "=" | "<>" | "<" | "<=" | ">" | ">=")
}

/// One-step simplifications of an expression: replace it with a child, then
/// recursively simplify each child in place. Depth is <=4 by construction so
/// the candidate set stays in the low tens.
fn simplifications(e: &E) -> Vec<E> {
    let mut out = Vec::new();
    match e {
        E::Col(..) | E::Bare(..) | E::Lit(_) => {}
        E::Bin(op, a, b) => {
            // Replacing a *comparison* with one of its operands turns a
            // predicate into a value, and both engines will then apply their
            // own truthiness rules to it -- rules that are outside the dialect
            // intersection (SQLite coerces `'z'` to 0/false, granular treats
            // non-empty as true, ClickHouse rejects it). A shrink that does
            // that stops reproducing the original bug and starts reproducing a
            // different one, which is worse than not shrinking: it was
            // observed doing exactly that here, turning a real
            // `count()`-over-filtered-rows case into `WHERE 'z'`. Connectives
            // are safe because their operands are predicates too.
            if !is_comparison(op) {
                out.push((**a).clone());
                out.push((**b).clone());
            }
            for s in simplifications(a) {
                out.push(E::Bin(op, s.b(), b.clone()));
            }
            for s in simplifications(b) {
                out.push(E::Bin(op, a.clone(), s.b()));
            }
        }
        E::Not(a) => {
            out.push((**a).clone());
            for s in simplifications(a) {
                out.push(E::Not(s.b()));
            }
        }
        E::IsNull(a, n) => {
            for s in simplifications(a) {
                out.push(E::IsNull(s.b(), *n));
            }
        }
        E::Cast(a, t) => {
            for s in simplifications(a) {
                out.push(E::Cast(s.b(), *t));
            }
        }
        E::Like(a, p, n) => {
            for s in simplifications(a) {
                out.push(E::Like(s.b(), p.clone(), *n));
            }
        }
        E::Between(a, lo, hi, n) => {
            for s in simplifications(a) {
                out.push(E::Between(s.b(), lo.clone(), hi.clone(), *n));
            }
        }
        E::In(a, list, n) => {
            if list.len() > 1 {
                for i in 0..list.len() {
                    let mut l = list.clone();
                    l.remove(i);
                    out.push(E::In(a.clone(), l, *n));
                }
            }
            for s in simplifications(a) {
                out.push(E::In(s.b(), list.clone(), *n));
            }
        }
        E::Case(arms, els) => {
            if let Some(x) = els {
                out.push((**x).clone());
            }
            if let Some((_, t)) = arms.first() {
                out.push(t.clone());
            }
            if arms.len() > 1 {
                for i in 0..arms.len() {
                    let mut a = arms.clone();
                    a.remove(i);
                    out.push(E::Case(a, els.clone()));
                }
            }
            if els.is_some() {
                out.push(E::Case(arms.clone(), None));
            }
        }
        E::Call(name, args) => {
            for (i, a) in args.iter().enumerate() {
                for s in simplifications(a) {
                    let mut v = args.clone();
                    v[i] = s;
                    out.push(E::Call(name, v));
                }
            }
        }
        E::Agg(name, Some(a), distinct) => {
            if *distinct {
                out.push(E::Agg(name, Some(a.clone()), false));
            }
            for s in simplifications(a) {
                out.push(E::Agg(name, Some(s.b()), *distinct));
            }
        }
        E::Agg(_, None, _) => {}
    }
    out
}

// ------------------------------------------------------------------ reporting

/// A reproducer that still needs an 8000-row table after shrinking is a real
/// reproducer, but twelve of them inline in one assertion message is not
/// readable. Keep the head and tail -- the DDL and the query, which is what a
/// reader looks at first -- and say what was cut.
fn elide(script: &str) -> String {
    const KEEP: usize = 40;
    let lines: Vec<&str> = script.lines().collect();
    if lines.len() <= KEEP * 2 {
        return script.to_string();
    }
    let mut s = lines[..KEEP].join("\n");
    let _ = write!(
        s,
        "\n-- ... {} statements elided; rerun the seed above for the full script ...\n",
        lines.len() - KEEP * 2
    );
    s.push_str(&lines[lines.len() - KEEP..].join("\n"));
    s.push('\n');
    s
}

fn report(case: &Case, diff: &Diff) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "\n================ DIFFERENTIAL MISMATCH ================");
    // The scripts below are the *shrunk* case. The seed regenerates the
    // original one, which is what you want when the shrinker is suspected of
    // having drifted to a different bug.
    let _ = writeln!(
        s,
        "seed: {} (unshrunk: GRANULAR_DIFF_SEED={} GRANULAR_DIFF_CASES=1 GRANULAR_DIFF_NO_SHRINK=1)",
        case.seed, case.seed
    );
    let _ = writeln!(
        s,
        "\n--- granular (paste into: granular -q '...') ---\n{}",
        elide(&case.script(Dialect::Granular))
    );
    let _ = writeln!(
        s,
        "--- sqlite3 (paste into: sqlite3) ---\n{}",
        elide(&case.script(Dialect::Sqlite))
    );
    match diff {
        Diff::OnlyOneRan { granular, sqlite } => {
            let _ = writeln!(s, "--- one engine refused the query ---");
            match granular {
                Ok(r) => {
                    let _ = writeln!(s, "granular: {} rows\n{}", r.len(), fmt_rows(r));
                }
                Err(e) => {
                    let _ = writeln!(s, "granular: ERROR {e}");
                }
            }
            match sqlite {
                Ok(r) => {
                    let _ = writeln!(s, "sqlite:   {} rows\n{}", r.len(), fmt_rows(r));
                }
                Err(e) => {
                    let _ = writeln!(s, "sqlite:   ERROR {e}");
                }
            }
        }
        Diff::RowCount { g, s: sr } => {
            let _ = writeln!(s, "--- row count: granular {g}, sqlite {sr} ---");
            if let Ok(r) = run_granular(case) {
                let _ = writeln!(s, "granular rows:\n{}", fmt_rows(&r));
            }
            if let Ok(r) = sqlite_one(case) {
                let _ = writeln!(s, "sqlite rows:\n{}", fmt_rows(&r));
            }
        }
        Diff::Row { at, g, s: sr } => {
            let _ = writeln!(s, "--- first differing row (index {at}) ---");
            let _ = writeln!(s, "granular: {}", g.iter().map(fmt_cell).collect::<Vec<_>>().join(" | "));
            let _ = writeln!(s, "sqlite:   {}", sr.iter().map(fmt_cell).collect::<Vec<_>>().join(" | "));
            if case.query.order.is_empty() {
                let _ = writeln!(s, "(compared as multisets: query has no ORDER BY)");
            }
        }
    }
    let _ = writeln!(s, "=======================================================");
    s
}

// ------------------------------------------------------------------ the tests

/// What the run actually generated. Printed with the result so a silent loss of
/// coverage cannot masquerade as a clean run.
#[derive(Default)]
struct Coverage {
    joins: usize,
    using: usize,
    star: usize,
    aggregate: usize,
    distinct: usize,
    limit: usize,
    set_ops: usize,
    calls: usize,
    multi_insert: usize,
    optimized: usize,
    over_granule: usize,
    over_block: usize,
    rows_loaded: usize,
    max_rows: usize,
    deletes: usize,
    updates: usize,
}

impl Coverage {
    fn observe(&mut self, c: &Case) {
        match &c.query.from {
            From::Join(_, con) => {
                self.joins += 1;
                if matches!(con, JoinCon::Using(_)) {
                    self.using += 1;
                }
            }
            From::One(_) => {}
        }
        self.star += c.query.star as usize;
        self.aggregate += (!c.query.group_by.is_empty()
            || c.query.items.iter().any(|e| matches!(e, E::Agg(..))))
            as usize;
        self.distinct += c.query.distinct as usize;
        self.limit += c.query.limit.is_some() as usize;
        self.set_ops += c.query.set_tail.is_some() as usize;
        let mut called = false;
        c.query.for_each_expr(&mut |e| e.walk(&mut |x| called |= matches!(x, E::Call(..))));
        self.calls += called as usize;
        for m in &c.mutations {
            if m.is_delete() {
                self.deletes += 1;
            } else {
                self.updates += 1;
            }
        }
        for t in &c.tables {
            let n = t.rows.len();
            self.rows_loaded += n;
            self.max_rows = self.max_rows.max(n);
            self.over_granule += (n > 1024) as usize;
            self.over_block += (n > 8192) as usize;
            self.multi_insert += (n > t.chunk) as usize;
            self.optimized += (t.optimize && n > 0) as usize;
        }
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn skip_without_sqlite() -> bool {
    if sqlite_path().is_none() {
        eprintln!(
            "SKIP: differential tests need the `sqlite3` CLI on PATH (or at \
             /usr/bin/sqlite3). Not found -- the engine is untested against an \
             external oracle in this run."
        );
        return true;
    }
    false
}

#[test]
fn differential_against_sqlite() {
    if skip_without_sqlite() {
        return;
    }
    let cases = env_usize("GRANULAR_DIFF_CASES", DEFAULT_CASES);
    // A fixed default seed keeps CI reproducible; a soak run overrides it.
    let seed0 = env_usize("GRANULAR_DIFF_SEED", 0x5EED_D1FF) as u64;
    let verbose = std::env::var("GRANULAR_DIFF_VERBOSE").is_ok();

    let mut checked = 0usize;
    let mut both_refused = 0usize;
    let mut failures: Vec<String> = Vec::new();
    // "0 mismatches" is only worth reading next to what was actually
    // generated. A run that quietly stopped emitting joins would otherwise look
    // exactly like a run that found nothing.
    let mut cov = Coverage::default();

    let mut i = 0usize;
    while i < cases {
        // Batches are bounded by *rows*, not just by case count: one 8000-row
        // case is worth more script bytes than thirty small ones, and a 20MB
        // script through a pipe is a different kind of test than the one we
        // meant to write.
        let mut batch: Vec<Case> = Vec::with_capacity(BATCH);
        let mut rows_queued = 0usize;
        while i + batch.len() < cases && batch.len() < BATCH && rows_queued < BATCH_ROWS {
            let c = gen_case_at(seed0.wrapping_add((i + batch.len()) as u64));
            rows_queued += c.tables.iter().map(|t| t.rows.len()).sum::<usize>();
            batch.push(c);
        }
        let n = batch.len();
        // Batching is worth 4x: measured A/B interleaved, best-of-3, 1200 cases
        // at seed 4242 -- 1.68s batched vs 6.82s one process per case. Process
        // spawn, not SQL, is the harness's dominant cost.
        // `GRANULAR_DIFF_NO_BATCH` keeps the slow path reachable, because the
        // batched reader attributes rows by sentinel and a bug there would be
        // invisible: the answers would just belong to the wrong cases.
        let sqlite_out = if std::env::var("GRANULAR_DIFF_NO_BATCH").is_ok() {
            None
        } else {
            sqlite_batch(&batch)
        };
        for (j, case) in batch.iter().enumerate() {
            cov.observe(case);
            let g = run_granular(case);
            let s: Outcome = match &sqlite_out {
                Some(all) => Ok(all[j].clone()),
                // The batch bailed: something in it errored on the sqlite side,
                // so this batch has to be re-run one process per case to find
                // out which.
                None => sqlite_one(case),
            };
            if matches!((&g, &s), (Err(_), Err(_))) {
                both_refused += 1;
            }
            checked += 1;
            if let Some(diff) = compare(case, g, s) {
                // Shrinking is on by default because an unshrunk reproducer is
                // usually unreadable, but it can drift to a *different*
                // divergence than the one it started from; `GRANULAR_DIFF_NO_SHRINK`
                // shows the case exactly as generated when that is suspected.
                let mut budget = if std::env::var("GRANULAR_DIFF_NO_SHRINK").is_ok() {
                    0
                } else {
                    400
                };
                let small = shrink(case.clone(), &mut budget);
                let d2 = compare(&small, run_granular(&small), sqlite_one(&small))
                    .unwrap_or(diff);
                failures.push(report(&small, &d2));
                if failures.len() >= 12 {
                    i = cases;
                    break;
                }
            }
            if verbose && checked % 50 == 0 {
                eprintln!("  {checked}/{cases} checked, {} mismatches", failures.len());
            }
        }
        i += n;
    }

    eprintln!(
        "differential: {checked} cases against sqlite3, {both_refused} rejected by both engines, \
         {} mismatches\n  coverage: {} joins ({} USING), {} SELECT *, {} aggregate, {} DISTINCT, \
         {} LIMIT, {} UNION, {} with scalar calls\n  mutation: {} DELETE, {} UPDATE\
         \n  storage:  {} rows loaded, widest table {}, \
         {} tables past GRANULE_SIZE, {} past BLOCK_SIZE, {} multi-INSERT, {} OPTIMIZEd",
        failures.len(),
        cov.joins,
        cov.using,
        cov.star,
        cov.aggregate,
        cov.distinct,
        cov.limit,
        cov.set_ops,
        cov.calls,
        cov.deletes,
        cov.updates,
        cov.rows_loaded,
        cov.max_rows,
        cov.over_granule,
        cov.over_block,
        cov.multi_insert,
        cov.optimized,
    );
    assert!(
        failures.is_empty(),
        "{} differential mismatch(es) against sqlite3:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The harness is only worth its runtime if it can actually see a wrong answer.
/// This injects one -- a row deleted, a value perturbed, a NULL turned into a
/// zero, a row order reversed -- and asserts the comparator catches each.
/// Without this, "no mismatches" is indistinguishable from "no comparison".
#[test]
fn comparator_catches_injected_wrong_answers() {
    let case = gen_case_at(1);
    let truth = vec![
        vec![Cell::Int(1), Cell::Text("a".into())],
        vec![Cell::Int(2), Cell::Null],
        vec![Cell::Real(1.5), Cell::Text("".into())],
    ];

    let same = |a: &Vec<Vec<Cell>>| compare(&case, Ok(a.clone()), Ok(truth.clone())).is_none();
    assert!(same(&truth), "identical results must compare equal");

    // A dropped row.
    let mut dropped = truth.clone();
    dropped.pop();
    assert!(!same(&dropped), "a missing row must be caught");

    // A perturbed number, well outside FLOAT_REL_TOL.
    let mut wrong = truth.clone();
    wrong[2][0] = Cell::Real(1.5000001);
    assert!(!same(&wrong), "a perturbed float must be caught");

    // NULL rendered as zero -- the classic differential-test false negative.
    let mut nulled = truth.clone();
    nulled[1][1] = Cell::Int(0);
    assert!(!same(&nulled), "NULL vs 0 must be caught");

    // Empty string vs NULL, which TSV output could not have distinguished.
    let mut empty = truth.clone();
    empty[2][1] = Cell::Null;
    assert!(!same(&empty), "'' vs NULL must be caught");

    // Integer vs the text that renders identically.
    let mut texty = truth.clone();
    texty[0][0] = Cell::Text("1".into());
    assert!(!same(&texty), "1 vs '1' must be caught");

    // Wrong arity.
    let mut narrow = truth.clone();
    narrow[0].pop();
    assert!(!same(&narrow), "a missing column must be caught");

    // Order-sensitivity: with ORDER BY, a reversal is a failure; without it,
    // it must NOT be, because neither engine promises an order.
    let mut ordered = case.clone();
    ordered.query.order = vec![(1, true, true)];
    let mut unordered = case.clone();
    unordered.query.order.clear();
    let rev: Vec<Vec<Cell>> = truth.iter().rev().cloned().collect();
    assert!(
        compare(&ordered, Ok(rev.clone()), Ok(truth.clone())).is_some(),
        "with ORDER BY, a reordered result must be caught"
    );
    assert!(
        compare(&unordered, Ok(rev), Ok(truth.clone())).is_none(),
        "without ORDER BY, row order is not defined in either engine"
    );

    // Tolerance is not a licence: values that really are equal must pass.
    let mut retyped = truth.clone();
    retyped[0][0] = Cell::Real(1.0);
    assert!(same(&retyped), "1 and 1.0 are the same number in both engines");

    // The reproducer must survive a big case: `elide` keeps the DDL and the
    // query (what a reader looks at first) and cuts the middle, rather than
    // pasting eight thousand INSERTs into an assertion message.
    let mut big = gen_case_at(2);
    big.tables[0].rows = (0..5000)
        .map(|i| {
            let mut r = vec![Cell::Int(i), Cell::Int(i % 7)];
            r.extend(big.tables[0].cols[2..].iter().map(|_| Cell::Null));
            r
        })
        .collect();
    big.tables[0].chunk = 1;
    let text = report(&big, &Diff::RowCount { g: 1, s: 2 });
    assert!(text.contains("statements elided"), "elide did not fire:\n{text}");
    assert!(text.contains("CREATE TABLE t0"), "elide cut the DDL");
    assert!(
        text.lines().count() < 400,
        "reproducer is {} lines; it should be elided",
        text.lines().count()
    );
}

/// End-to-end proof that the injected-error test above is testing the *real*
/// path: hand granular a query whose answer we know and sqlite the same one,
/// then corrupt the granular side and confirm the full pipeline flags it.
#[test]
fn harness_detects_a_corrupted_engine_answer() {
    if skip_without_sqlite() {
        return;
    }
    let case = Case {
        seed: 0,
        tables: vec![TableDef {
            name: "t0".into(),
            cols: vec![
                ColDef { name: "id".into(), ty: Ty::Int, nullable: false },
                ColDef { name: "k".into(), ty: Ty::Int, nullable: true },
            ],
            rows: vec![
                vec![Cell::Int(1), Cell::Int(10)],
                vec![Cell::Int(2), Cell::Null],
                vec![Cell::Int(3), Cell::Int(30)],
            ],
            sort: SortKey::Id,
            chunk: usize::MAX,
            optimize: false,
            keyed: false,
        }],
        mutations: Vec::new(),
        query: Query {
            from: From::One(0),
            star: false,
            distinct: false,
            items: vec![E::Col(0, 0), E::Col(0, 1)],
            filter: None,
            group_by: Vec::new(),
            having: None,
            set_tail: None,
            order: vec![(1, true, true), (2, true, true)],
            limit: None,
            offset: 0,
        },
    };
    let g = run_granular(&case).expect("granular must run this");
    let s = sqlite_one(&case).expect("sqlite must run this");
    assert_eq!(g.len(), 3, "granular returned {g:?}");
    assert!(
        compare(&case, Ok(g.clone()), Ok(s.clone())).is_none(),
        "engines must agree on a trivial query:\n{}",
        report(&case, &compare(&case, Ok(g.clone()), Ok(s.clone())).unwrap())
    );
    // Now break the engine's answer and confirm the same code path objects.
    let mut corrupt = g;
    corrupt[1][1] = Cell::Int(0);
    assert!(
        compare(&case, Ok(corrupt), Ok(s)).is_some(),
        "the harness failed to notice a corrupted answer -- it proves nothing"
    );
}

/// Sanity check on the sqlite driver's type recovery. If this is wrong, every
/// other comparison in the file is meaningless.
#[test]
fn sqlite_driver_recovers_types_exactly() {
    if skip_without_sqlite() {
        return;
    }
    let script = format!(
        "{PREAMBLE}SELECT 1, 1.0, 'ab', NULL, '', 'it''s', 'NULL', '12', -0.0, 0.1;"
    );
    let (out, err) = sqlite_raw(&script).expect("sqlite3 runs");
    assert!(err.trim().is_empty(), "sqlite stderr: {err}");
    let row = parse_quote_row(out.lines().next().expect("one row"));
    let want = [
        Cell::Int(1),
        Cell::Real(1.0),
        Cell::Text("ab".into()),
        Cell::Null,
        Cell::Text("".into()),
        Cell::Text("it's".into()),
        // The reason `.mode tabs` + `.nullvalue NULL` is not used: this cell
        // and the NULL two columns back would be the same four characters.
        Cell::Text("NULL".into()),
        Cell::Text("12".into()),
        Cell::Real(-0.0),
        Cell::Real(0.1),
    ];
    assert_eq!(row.len(), want.len(), "parsed {row:?}");
    for (i, (got, expect)) in row.iter().zip(&want).enumerate() {
        assert!(
            matches!(
                (got, expect),
                (Cell::Null, Cell::Null)
                    | (Cell::Int(_), Cell::Int(_))
                    | (Cell::Real(_), Cell::Real(_))
                    | (Cell::Text(_), Cell::Text(_))
            ) && cells_equal(got, expect),
            "column {i}: got {got:?}, want {expect:?}"
        );
    }
    // And a NULL in the last position, which the line-based parser has to
    // recover from a trailing separator.
    let (out, _) = sqlite_raw(&format!("{PREAMBLE}SELECT 1, NULL;")).unwrap();
    let row = parse_quote_row(out.lines().next().unwrap());
    assert_eq!(row.len(), 2);
    assert!(matches!(row[1], Cell::Null));
}

/// The batched sqlite reader attributes rows to cases by a sentinel row. If
/// that attribution were off by one the harness would not crash -- it would
/// compare granular's answer for case *n* against sqlite's answer for case
/// *n+1* and report a flood of nonsense, or worse, silently agree. So: run the
/// same cases both ways and require identical results.
#[test]
fn batched_and_unbatched_sqlite_agree() {
    if skip_without_sqlite() {
        return;
    }
    let cases: Vec<Case> = (0..24u64).map(gen_case_at).collect();
    let batched = sqlite_batch(&cases).expect("no generated case should error in sqlite");
    assert_eq!(batched.len(), cases.len());
    for (i, case) in cases.iter().enumerate() {
        let solo = sqlite_one(case).unwrap_or_else(|e| panic!("case {i}: {e}"));
        assert_eq!(
            solo.len(),
            batched[i].len(),
            "case {i}: batched gave {} rows, solo gave {} -- sentinel attribution is off\n{}",
            batched[i].len(),
            solo.len(),
            case.script(Dialect::Sqlite)
        );
        for (r, (a, b)) in solo.iter().zip(&batched[i]).enumerate() {
            assert!(
                rows_equal(a, b),
                "case {i} row {r}: solo {a:?} vs batched {b:?}"
            );
        }
    }
}

/// KNOWN DIVERGENCES, pinned.
///
/// Each entry is a dialect difference the generator deliberately avoids. This
/// test asserts the difference *still exists*: the day granular changes its
/// mind, this fails and forces the exclusion to be re-argued or deleted, which
/// is the only way an allowlist stays honest.
#[test]
fn known_divergences_still_reproduce() {
    if skip_without_sqlite() {
        return;
    }
    let probe = |g_sql: &str, s_sql: &str| -> (String, String) {
        let mut sess = Session::in_memory();
        let gv = match sess.query(g_sql) {
            Ok(rs) => rs.to_values().first().map(|r| fmt_cell(&cell_of_value(&r[0]))).unwrap_or_default(),
            Err(e) => format!("ERROR: {e}"),
        };
        let (out, err) = sqlite_raw(&format!("{PREAMBLE}{s_sql};")).expect("sqlite3 runs");
        let sv = if err.trim().is_empty() {
            out.lines()
                .next()
                .map(|l| fmt_cell(&parse_quote_row(l)[0]))
                .unwrap_or_default()
        } else {
            format!("ERROR: {}", err.trim())
        };
        (gv, sv)
    };

    // #1 is withdrawn -- it turned out to be BUG 5, now fixed. See
    // `constant_folding_keeps_three_valued_logic`.

    // #2 -- integer division.
    let (g, s) = probe("SELECT 7 / 2", "SELECT 7 / 2");
    assert_ne!(g, s, "integer division now agrees ({g} vs {s}); drop KNOWN DIVERGENCE #2");

    // #3 -- LIKE case folding.
    let (g, s) = probe("SELECT 'ABC' LIKE 'a%'", "SELECT 'ABC' LIKE 'a%'");
    assert_ne!(g, s, "LIKE case folding now agrees ({g} vs {s}); drop KNOWN DIVERGENCE #3");

    // #4 -- CAST(<real> AS text) rendering.
    let (g, s) = probe("SELECT CAST(1.0 AS String)", "SELECT CAST(1.0 AS TEXT)");
    assert_ne!(g, s, "real->text rendering now agrees ({g} vs {s}); drop KNOWN DIVERGENCE #4");

    // #6 -- Bool is a real type in granular and not one in SQLite, so it
    // renders differently under a text cast. Only the *rendering* differs;
    // Bool arithmetic agrees in both operand shapes (see BUG 4, now fixed).
    let (g, s) = probe("SELECT CAST((1=1) AS String)", "SELECT CAST((1=1) AS TEXT)");
    assert_ne!(g, s, "bool->text rendering now agrees ({g} vs {s}); drop KNOWN DIVERGENCE #6");

    // #7 -- EXCEPT and INTERSECT: a stated capability gap, not a wrong answer.
    // When they land, put them back in `gen_query`'s set-operation menu.
    let mut sess = Session::in_memory();
    for op in ["EXCEPT", "INTERSECT"] {
        let e = sess
            .query(&format!("SELECT 1 {op} SELECT 1"))
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            e.contains("not implemented") || e.contains("only UNION"),
            "{op} now runs (or fails differently: {e:?}) -- add it back to \
             gen_query's set-operation menu"
        );
    }

    // #8 -- granular rounds half to even, SQLite half away from zero.
    let (g, s) = probe("SELECT round(2.5)", "SELECT round(2.5)");
    assert_ne!(g, s, "round() now agrees ({g} vs {s}); it can go into gen_call");

    // #9 -- SQLite's concat() drops NULL arguments; granular propagates.
    let (g, s) = probe("SELECT concat(CAST(NULL AS String), 'b')", "SELECT concat(NULL, 'b')");
    assert_ne!(g, s, "concat() NULL handling now agrees ({g} vs {s})");
}

/// KNOWN DIVERGENCE #5, measured rather than assumed.
///
/// The brief for this harness predicted that float `sum` would legitimately
/// diverge: SQLite sums naively, granular uses Neumaier compensation. **That
/// prediction is false on any SQLite from 3.44 (Nov 2023) onward** -- SQLite
/// adopted Kahan-Babuska-Neumaier in `sum`, `avg` and `total` in that release.
/// Measured here on 3.54: the adversarial input below gives *both* engines the
/// exact answer, so no tolerance slack is needed for summation at all and
/// `FLOAT_REL_TOL` can stay at 1e-12.
///
/// The test still pins the assumption from both ends. If it is ever run
/// against a pre-3.44 sqlite3 the divergence reappears; the assertion below
/// spells out which engine is right in that case (granular, and by exactly the
/// compensated residual) rather than silently widening the tolerance.
#[test]
fn float_summation_agrees_because_both_engines_compensate() {
    if skip_without_sqlite() {
        return;
    }
    // 1e16 + 1 + ... + 1 - 1e16. Naive summation loses every 1 (each is below
    // the ulp of 1e16, which is 2); compensated summation keeps all of them.
    const N: usize = 100;
    let mut vals = vec![1e16f64];
    vals.extend(std::iter::repeat_n(1.0, N));
    vals.push(-1e16);
    let exact = N as f64;

    let lits: Vec<String> = vals.iter().map(|v| format!("({v:?})")).collect();
    let mut sess = Session::in_memory();
    sess.execute("CREATE TABLE f (id Int64, v Float64) ENGINE = MergeTree ORDER BY id").unwrap();
    let tuples: Vec<String> = vals.iter().enumerate().map(|(i, v)| format!("({i}, {v:?})")).collect();
    sess.execute(&format!("INSERT INTO f VALUES {}", tuples.join(", "))).unwrap();
    let g = match sess.query("SELECT sum(v) FROM f").unwrap().scalar().unwrap() {
        Value::Float(f) => f,
        other => panic!("sum(v) was {other:?}"),
    };

    let script = format!(
        "{PREAMBLE}CREATE TABLE f(v REAL);\nINSERT INTO f VALUES {};\nSELECT sum(v) FROM f;",
        lits.join(", ")
    );
    let (out, err) = sqlite_raw(&script).expect("sqlite3 runs");
    assert!(err.trim().is_empty(), "{err}");
    let s = match parse_quote_row(out.lines().next().unwrap())[0] {
        Cell::Real(f) => f,
        Cell::Int(i) => i as f64,
        ref other => panic!("sqlite sum was {other:?}"),
    };

    assert_eq!(g, exact, "granular's compensated sum should be exact");
    if s == exact {
        // The expected case on sqlite3 >= 3.44: both compensate, so the tight
        // FLOAT_REL_TOL is justified and the generator's REALS pool is a
        // belt-and-braces measure rather than a load-bearing one.
        return;
    }
    // Pre-3.44 sqlite3: the divergence is real, and granular is the accurate
    // side. Assert that rather than treating the mismatch as a granular bug.
    assert!(
        (g - exact).abs() < (s - exact).abs(),
        "sqlite ({s}) and granular ({g}) disagree on a compensated sum and \
         granular is NOT the more accurate one -- that is a granular bug, not \
         a dialect difference"
    );
    eprintln!(
        "note: this sqlite3 sums naively (got {s}, exact {exact}); float \
         aggregates over adversarial magnitudes are outside the intersection \
         on this machine"
    );
}

/// BUG 1, found by this harness on its first run and pinned here.
///
/// `ENGINE = MergeTree ORDER BY id` on a single integer column makes the table
/// "fast-PK" (`Schema::has_fast_pk` in src/types/schema.rs, consumed by
/// `Table::new` in src/storage/table.rs), which routes writes into the *keyed*
/// delta. The keyed delta is last-write-wins: `put_keyed` overwrites the slot
/// an existing key already owns. But `ORDER BY` in ClickHouse's MergeTree is a
/// sort key, **not** a unique key -- duplicate values are legal and every row
/// must survive. granular silently drops all but the last, and the INSERT still
/// reports the full row count, so nothing in the system says data was lost.
///
/// `ORDER BY (id, k)` and `ORDER BY tuple()` are unaffected, which is what lets
/// the generator keep testing colliding sort keys on those two shapes.
///
/// FIXED, and this test is the inversion. `TableDef::pk_col` now returns a key
/// only when the table *declared* one -- an explicit `PRIMARY KEY`, or
/// `ENGINE = ReplacingMergeTree` -- so `ORDER BY` alone leaves the table
/// unkeyed and every duplicate survives. Both opt-ins are asserted below,
/// because a fix that kept the rows by disabling the keyed delta outright
/// would pass the first assertion and silently cost the OLTP path.
#[test]
fn a_sort_key_alone_does_not_deduplicate() {
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE t (id Int64, v Int64) ENGINE = MergeTree ORDER BY id").unwrap();
    s.execute("INSERT INTO t VALUES (4, 1), (4, 2)").unwrap();
    let n = s.query("SELECT count(*) FROM t").unwrap().scalar().unwrap();
    assert_eq!(cell_of_value(&n).num(), 2.0, "a sort key must not deduplicate");

    // ...and uniqueness is still available when it is asked for, by either
    // route, still last-write-wins.
    for ddl in [
        "CREATE TABLE u (id Int64, v Int64) ENGINE = MergeTree ORDER BY id PRIMARY KEY id",
        "CREATE TABLE u (id Int64, v Int64) ENGINE = ReplacingMergeTree ORDER BY id",
    ] {
        let mut s = Session::in_memory();
        s.execute(ddl).unwrap();
        s.execute("INSERT INTO u VALUES (4, 1), (4, 2)").unwrap();
        let n = s.query("SELECT count(*) FROM u").unwrap().scalar().unwrap();
        assert_eq!(cell_of_value(&n).num(), 1.0, "declared key stopped upserting: {ddl}");
        let v = s.query("SELECT v FROM u").unwrap().scalar().unwrap();
        assert_eq!(cell_of_value(&v).num(), 2.0, "not last-write-wins: {ddl}");
    }

    // The composite and empty sort keys keep every row, including a row that
    // duplicates another in full.
    for clause in ["ORDER BY (id, v)", "ORDER BY tuple()"] {
        let mut s = Session::in_memory();
        s.execute(&format!("CREATE TABLE t (id Int64, v Int64) ENGINE = MergeTree {clause}"))
            .unwrap();
        s.execute("INSERT INTO t VALUES (4, 1), (4, 1), (4, 2)").unwrap();
        let n = s.query("SELECT count(*) FROM t").unwrap().scalar().unwrap();
        assert_eq!(cell_of_value(&n).num(), 3.0, "{clause} lost rows too");
    }
}

/// BUG 2, found by this harness and now FIXED -- this test is the inversion.
///
/// `sum` over zero rows used to answer 0 or NULL depending on whether its
/// argument was declared `Nullable`. SQLite and the SQL standard say NULL for
/// both; ClickHouse says 0 for both. No dialect says "it depends", so one of
/// the two answers was wrong whichever compatibility target you read it
/// against, and the disagreement was internal to granular.
///
/// Resolved toward NULL (`SumAcc::finish` in src/exec/functions/agg.rs), which
/// is both the SQLite answer this harness diffs against and what `avg`, `min`
/// and `max` next door already did over the same input -- so `sum` was the
/// outlier, and picking 0 would have meant changing three other aggregates.
/// `sum`'s return type is now unconditionally `Nullable`.
#[test]
fn sum_over_zero_rows_is_null_regardless_of_nullability() {
    let mut s = Session::in_memory();
    s.execute(
        "CREATE TABLE t (id Int64, nn Float64, nl Nullable(Float64)) \
         ENGINE = MergeTree ORDER BY id",
    )
    .unwrap();
    let r = s.query("SELECT sum(nn), sum(nl) FROM t").unwrap().to_values();
    assert_eq!(r.len(), 1);
    let (nn, nl) = (cell_of_value(&r[0][0]), cell_of_value(&r[0][1]));
    assert!(nn.is_null() && nl.is_null(), "got {nn:?}, {nl:?}");
    // Same answer via a filter that removes every row, i.e. it is about the
    // empty input rather than about the table being empty.
    s.execute("INSERT INTO t VALUES (1, 5.0, 5.0)").unwrap();
    let r = s.query("SELECT sum(nn), sum(nl) FROM t WHERE id > 99").unwrap().to_values();
    assert!(cell_of_value(&r[0][0]).is_null() && cell_of_value(&r[0][1]).is_null());
    // ...and the neighbouring aggregates agree, which is the consistency that
    // was missing.
    let r = s.query("SELECT avg(nn), min(nn), max(nn), count(nn) FROM t WHERE id > 99")
        .unwrap()
        .to_values();
    let got: Vec<Cell> = r[0].iter().map(cell_of_value).collect();
    assert!(got[0].is_null() && got[1].is_null() && got[2].is_null() && got[3].is_zero(),
        "avg/min/max/count over an empty input should be NULL, NULL, NULL, 0; got {got:?}");
    // A non-empty input is unaffected -- the fix is about the empty set only.
    let r = s.query("SELECT sum(nn), sum(nl) FROM t").unwrap().to_values();
    let got: Vec<Cell> = r[0].iter().map(cell_of_value).collect();
    assert!(matches!(got[..], [Cell::Real(a), Cell::Real(b)] if a == 5.0 && b == 5.0), "{got:?}");
    // HAVING is the shape where the divergence changed the row *count* rather
    // than a cell: a NULL predicate keeps no rows, where 0 > -1 kept one.
    let n = s.query("SELECT count(*) FROM t WHERE id > 99 HAVING sum(nn) > -1")
        .unwrap()
        .to_values();
    assert!(n.is_empty(), "empty global group must not survive HAVING: {n:?}");
}

/// BUG 3, found by this harness (via the shrinker, then confirmed by hand).
///
/// A non-boolean `WHERE` operand. SQLite coerces text to a number, so `'z'`
/// is 0 and filters everything out; ClickHouse rejects a String filter
/// outright. granular does a third thing -- Python-style truthiness, where a
/// non-empty string is true -- so `WHERE 'z'` keeps every row. It is not
/// reachable from the generator's grammar (predicates are always built as
/// predicates), which is why this is a pinned note rather than a filter.
#[test]
fn non_boolean_where_operand_follows_neither_reference() {
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE t (id Int64) ENGINE = MergeTree ORDER BY id").unwrap();
    s.execute("INSERT INTO t VALUES (1), (2)").unwrap();
    let mut kept = |sql: &str| -> f64 {
        cell_of_value(&s.query(sql).unwrap().scalar().unwrap()).num()
    };
    assert_eq!(kept("SELECT count(*) FROM t WHERE 'z'"), 2.0, "BUG 3 changed");
    assert_eq!(kept("SELECT count(*) FROM t WHERE ''"), 0.0);
    // SQLite's answer for both is 0 (text coerces to the number 0), and
    // ClickHouse's answer for both is an error.
}

/// BUG 4, found by this harness and now FIXED -- this test is the inversion.
///
/// `DataType::promote` (src/types/datatype.rs) opened with
/// `_ if ba == bb => ba.clone()`, so `promote(Bool, Bool)` was `Bool` and the
/// arm below it that correctly widens `Bool` against an integer never got a
/// chance. The result column was a `Bool` and the arithmetic answer was forced
/// back into it: `(1=1)+(2=2)` rendered as `true` (2 coerced into a boolean
/// lane) and `(1=2)-(1=1)` killed the query with "-1 is not a Bool". One
/// expression shape, truncating one way and aborting the other.
///
/// The Bool/Bool pair is now matched ahead of the equal-types shortcut and
/// widens to `Int64`, the same type the mixed Bool/Int arm already produced.
/// Both shapes agree with SQLite, and the generator emits them again.
#[test]
fn arithmetic_on_two_booleans_widens_to_an_integer() {
    let mut s = Session::in_memory();
    let one = |s: &mut Session, sql: &str| -> Result<Cell, String> {
        s.query(sql)
            .map_err(|e| e.to_string())
            .map(|rs| cell_of_value(&rs.scalar().unwrap()))
    };
    assert!(matches!(one(&mut s, "SELECT (1=1) + (2=2)"), Ok(Cell::Int(2))));
    assert!(matches!(one(&mut s, "SELECT (1=2) - (1=1)"), Ok(Cell::Int(-1))));
    assert!(matches!(one(&mut s, "SELECT (1=1) * (1=1)"), Ok(Cell::Int(1))));
    // Mixing in an integer still takes the arm it always did.
    assert!(matches!(one(&mut s, "SELECT (1=1) + 1"), Ok(Cell::Int(2))));
    assert!(matches!(one(&mut s, "SELECT (1=1) * 2"), Ok(Cell::Int(2))));
    assert!(matches!(one(&mut s, "SELECT 0 - (1=1)"), Ok(Cell::Int(-1))));
    // Against a column, i.e. off the constant-folding path, so both the folded
    // and the vectorized evaluator are covered.
    s.execute("CREATE TABLE t (id Int64) ENGINE = MergeTree ORDER BY id").unwrap();
    s.execute("INSERT INTO t VALUES (1), (2)").unwrap();
    let r = s.query("SELECT (id=1) + (id=2), (id=1) - (id=2) FROM t ORDER BY id")
        .unwrap()
        .to_values();
    let got: Vec<Cell> = r.iter().flat_map(|row| row.iter().map(cell_of_value)).collect();
    assert!(
        matches!(got[..], [Cell::Int(1), Cell::Int(1), Cell::Int(1), Cell::Int(-1)]),
        "{got:?}"
    );
}

/// BUG 6, found by this harness the moment BUG 4's fix widened the grammar:
/// with both arithmetic operands drawn from the general integer generator,
/// `length(x0.a1) - length('ab')` became reachable, and `length` returns
/// `UInt64`.
///
/// `DataType::promote`'s equal-types shortcut hands two unsigned operands an
/// unsigned result, and the two evaluators then disagree about what to do with
/// a negative answer:
///
///   * the **vectorized** path wraps (`wrapping_sub` on `u64`), so
///     `length(col) - length('ab')` for a 1-character value is 2^64-1;
///   * the **constant-folding** path computes -1 as an `i64` and then fails
///     `Column::constant`, killing the query with "-1 is not a UInt64".
///
/// One expression, two behaviours -- the same signature BUG 4 had, which is
/// what makes it a bug rather than the unsigned-wrap dialect note it would
/// otherwise be (ClickHouse wraps too; SQLite has no unsigned type at all).
/// The repair is not in the promotion table: making unsigned pairs promote to
/// `Int64` would cost `UInt64` columns their top bit. It is either `length`
/// returning `Int64` (src/exec/functions/scalar.rs, `r_length`) or the two
/// evaluators being made to agree on the overflow.
///
/// Until then the generator keeps at most one unsigned operand per arithmetic
/// node -- see the `Ty::Int` arm 0 of `gen_scalar`.
#[test]
fn unsigned_subtraction_wraps_one_way_and_errors_the_other() {
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE t (id Int64, a String) ENGINE = MergeTree ORDER BY id").unwrap();
    s.execute("INSERT INTO t VALUES (1, 'z')").unwrap();
    // Vectorized: wraps.
    let got = cell_of_value(
        &s.query("SELECT length(a) - length('ab') FROM t").unwrap().scalar().unwrap(),
    );
    assert!(
        matches!(got, Cell::Real(f) if f == u64::MAX as f64) || matches!(got, Cell::Int(-1)),
        "BUG 6 changed: vectorized unsigned subtraction is now {got:?}"
    );
    let wrapped = !matches!(got, Cell::Int(-1));
    // Constant-folded: the same expression refuses to run at all.
    let folded = s.query("SELECT length('z') - length('ab')");
    assert!(
        !wrapped || folded.as_ref().err().map(|e| e.to_string()).unwrap_or_default()
            .contains("is not a UInt64"),
        "BUG 6 changed: folded unsigned subtraction now gives {:?}",
        folded.map(|r| cell_of_value(&r.scalar().unwrap()))
    );
    // A signed operand on either side takes the correct promotion arm, which is
    // why the generator's arithmetic keeps one side signed.
    let ok = cell_of_value(&s.query("SELECT length(a) - 2 FROM t").unwrap().scalar().unwrap());
    assert!(matches!(ok, Cell::Int(-1)), "{ok:?}");
}

/// BUG 5 -- **FIXED**. Found by this harness at case ~1287334 of a 60 000-case
/// soak, minimized by hand from the shrinker's output, and inverted here when
/// the fix landed.
///
/// `const_eval_at` in src/planner/optimizer.rs folds constant subexpressions.
/// Two of its arms used to drop three-valued logic:
///
///   * `AND`/`OR` decided via `Value::truthy()`, which maps NULL to *false*, so
///     `(NULL < 5) AND (1 = 1)` folded to `false` where SQL says UNKNOWN. The
///     `Binary` arm right below it always got this right -- it returns
///     `Value::Null` the moment either side is NULL -- so the omission was
///     local to the short-circuit branch, not a house convention.
///   * `InList` had no NULL handling at all (`let found = list.contains(&v)`),
///     so both `NULL IN (1,2)` and `5 IN (2, NULL)` folded to `false`.
///
/// `BETWEEN` was collateral: it desugars to `>= AND <=` and inherited the
/// first, which made `NULL NOT BETWEEN 1 AND 5` fold to `true` and admit rows
/// from a `WHERE` that had to exclude them -- a wrong-*rows* bug.
///
/// What made it unambiguous rather than a dialect argument: granular's own
/// **vectorized** path answered NULL for every one of these and agreed with
/// SQLite exactly, so the same expression got two different answers depending
/// only on whether the planner could fold it.
///
/// This is also why KNOWN DIVERGENCE #1 was withdrawn. It was filed as
/// ClickHouse compatibility on the strength of a single all-literal probe;
/// running the same expression against a column disproved that.
///
/// The invariant is now pinned upstream by
/// `planner::optimizer::tests::folding_never_changes_the_answer`, a property
/// test that diffs the folder against the vectorized evaluator over generated
/// NULL-bearing expressions. This test survives as the named-shape witness.
#[test]
fn constant_folding_keeps_three_valued_logic() {
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE t (id Int64, k Nullable(Int64)) ENGINE = MergeTree ORDER BY id")
        .unwrap();
    // Row 1 has a NULL key; row 2 has a key that *misses* an IN list which
    // itself contains NULL -- the second shape SQL says is UNKNOWN.
    s.execute("INSERT INTO t VALUES (1, NULL), (2, 5)").unwrap();

    let mut one = |sql: &str| cell_of_value(&s.query(sql).unwrap().scalar().unwrap());
    const N: &str = "CAST(NULL AS Nullable(Int64))";

    // Folded, and now UNKNOWN -- the same answer the column forms give below.
    for expr in [
        format!("({N} < 5) AND (1 = 1)"),
        format!("({N} < 5) OR (1 = 2)"),
        format!("{N} BETWEEN 1 AND 5"),
        format!("{N} NOT BETWEEN 1 AND 5"),
        format!("{N} IN (1, 2)"),
        "5 IN (2, NULL)".to_string(),
        "5 NOT IN (2, NULL)".to_string(),
    ] {
        let got = one(&format!("SELECT {expr}"));
        assert!(
            got.is_null(),
            "BUG 5 regressed: folded `{expr}` is {got:?}, expected NULL"
        );
    }

    // A decided membership still folds to a plain boolean even though the list
    // carries a NULL, and a dominant operand still short-circuits: the fix must
    // not have bought 3VL by giving up folding.
    assert!(matches!(one("SELECT 2 IN (2, NULL)"), Cell::Int(1)));
    assert!(matches!(one("SELECT (1 = 2) AND (NULL < 5)"), Cell::Int(0)));
    assert!(matches!(one("SELECT (1 = 1) OR (NULL < 5)"), Cell::Int(1)));

    // Not folded (a column is involved): correct, and equal to SQLite's answer.
    // `id = 1` is the NULL-operand shape; `id = 2` is the miss-against-a-list-
    // containing-NULL shape, which is the exact expression `5 IN (2, NULL)`
    // above and which granular gets right here and wrong there.
    for (expr, id) in [
        ("(k < 5) AND (1 = 1)", 1),
        ("(k < 5) OR (1 = 2)", 1),
        ("k BETWEEN 1 AND 5", 1),
        ("k NOT BETWEEN 1 AND 5", 1),
        ("k IN (1, 2)", 1),
        ("k IN (2, NULL)", 2),
    ] {
        let got = one(&format!("SELECT {expr} FROM t WHERE id = {id}"));
        assert!(
            got.is_null(),
            "the vectorized path is supposed to be the correct one; `{expr}` \
             at id={id} gave {got:?}"
        );
    }

    // The folding arms that always handled NULL, kept as the control group:
    // the diagnosis above was about two specific arms, not about folding in
    // general, and these must not have changed.
    for expr in [
        format!("{N} < 5"),
        format!("NOT ({N} < 5)"),
        format!("{N} + 1"),
        "CAST(NULL AS Nullable(String)) LIKE 'a%'".to_string(),
    ] {
        let got = one(&format!("SELECT {expr}"));
        assert!(got.is_null(), "folded `{expr}` gave {got:?}, expected NULL");
    }
    drop(one);

    // `WHERE <constantly UNKNOWN>` admits nothing, exactly like `WHERE false`.
    // Before the fix this plan was reached by folding the predicate to `false`;
    // the predicate folds to NULL now, so the emptiness has to come from the
    // filter rule in `sink_filter` instead.
    for pred in [format!("{N} IN (1, 2)"), format!("({N} < 5) AND (1 = 1)")] {
        let rs = s.query(&format!("SELECT id FROM t WHERE {pred}")).unwrap();
        assert_eq!(rs.rows(), 0, "`WHERE {pred}` returned rows");
    }
}

/// The generator must be a pure function of its seed, or a printed reproducer
/// is a lie.
#[test]
fn generator_is_reproducible_from_its_seed() {
    for seed in [1u64, 7, 99, 100_003] {
        let a = gen_case_at(seed);
        let b = gen_case_at(seed);
        assert_eq!(a.script(Dialect::Granular), b.script(Dialect::Granular));
        assert_eq!(a.script(Dialect::Sqlite), b.script(Dialect::Sqlite));
    }
    // ...and distinct seeds must actually produce distinct cases, or the run
    // is 400 copies of one test.
    let mut seen: HashMap<String, u64> = HashMap::new();
    for seed in 0..200u64 {
        let c = gen_case_at(seed);
        seen.insert(c.script(Dialect::Granular), seed);
    }
    assert!(seen.len() > 190, "only {} distinct cases in 200 seeds", seen.len());
}

/// Both renderings must be well-formed for their dialect. A generator that
/// emits garbage would show up as a flood of `OnlyOneRan` diffs; this catches
/// it as a crisp failure instead.
#[test]
fn generated_sql_is_accepted_by_at_least_one_engine() {
    if skip_without_sqlite() {
        return;
    }
    let mut sqlite_rejects = 0;
    let n = 120;
    for seed in 0..n {
        let c = gen_case_at(seed);
        if sqlite_one(&c).is_err() {
            sqlite_rejects += 1;
        }
    }
    // SQLite is the reference dialect here; if it cannot parse what we emit,
    // the grammar has drifted out of the intersection.
    assert!(
        sqlite_rejects * 20 < n,
        "sqlite rejected {sqlite_rejects}/{n} generated queries -- the grammar \
         left the dialect intersection"
    );
}

/// BUG 7, FIXED, and this test is the inversion.
///
/// `UPDATE` on a table with no single-column primary key used to compute the
/// new row images and *append* them. On a keyed table the insert tombstones the
/// original by key, which is what makes the mutation a replacement; with no key
/// there was nothing to tombstone against, so both versions stayed live and the
/// table grew. No error, and `SELECT count()` was the only evidence -- silent
/// corruption rather than a wrong answer.
///
/// The repair was the delete half the path never had: hide the matching rows
/// (one delete-bitmap write per part, which needs no key -- a row's identity is
/// its position) and *then* append. Both halves are asserted here, because an
/// UPDATE that deleted without appending would also leave a plausible count.
#[test]
fn an_unkeyed_update_replaces_rather_than_duplicating() {
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE t (id UInt64, v UInt64) ENGINE = MergeTree ORDER BY id").unwrap();
    s.execute("INSERT INTO t VALUES (1,10),(2,20),(3,30)").unwrap();
    s.execute("UPDATE t SET v = 99 WHERE id = 2").unwrap();
    let n = s.query("SELECT count() FROM t").unwrap().scalar().unwrap().as_u64();
    assert_eq!(n, Some(3), "an unkeyed UPDATE must replace, not duplicate");
    // ...and it must be the NEW image that survives, not the old one.
    let vs: Vec<i64> = s
        .query("SELECT v FROM t ORDER BY id")
        .unwrap()
        .to_values()
        .iter()
        .filter_map(|r| r[0].as_i64())
        .collect();
    assert_eq!(vs, vec![10, 99, 30], "the update wrote the wrong row image");
    // Exactly one row for the updated key: the old image must be tombstoned,
    // not merely outnumbered.
    let both = s.query("SELECT v FROM t WHERE id = 2").unwrap().to_values();
    assert_eq!(both.len(), 1, "the pre-update image is still live: {both:?}");
}

/// BUG 8, FIXED, and this test is the inversion.
///
/// `UPDATE t SET id = id + 10 WHERE id = 1` used to change nothing at all: no
/// row rewritten, none removed, no error. The mutation was rendered as a query
/// -- `SELECT id + 10 AS id, v AS v FROM t WHERE id = 1` -- and this dialect
/// lets `WHERE` see select-list aliases, so `id` in the predicate bound to the
/// *assignment*, giving `WHERE id + 10 = 1`, which matches no row. Any `UPDATE`
/// whose predicate read a column the same statement assigned was affected;
/// changing the key was just the clearest case.
///
/// The fix was not to change alias visibility -- that rule is deliberate and
/// shared with the whole SELECT path -- but to stop expressing a mutation as a
/// synthesized SELECT. `Binder::bind_update` binds the predicate against the
/// table's own scope, which has no select list in it to shadow anything.
#[test]
fn an_update_predicate_binds_against_the_table_not_the_assignments() {
    let mut s = Session::in_memory();
    s.execute("CREATE TABLE t (id Int64, v Int64) ENGINE = MergeTree PRIMARY KEY id ORDER BY id")
        .unwrap();
    s.execute("INSERT INTO t VALUES (1,10),(2,20)").unwrap();
    s.execute("UPDATE t SET id = id + 10 WHERE id = 1").unwrap();
    let got = s.query("SELECT id, v FROM t ORDER BY id").unwrap().to_values();
    let ids: Vec<i64> = got.iter().filter_map(|r| r[0].as_i64()).collect();
    assert_eq!(
        ids,
        vec![2, 11],
        "an UPDATE whose predicate reads a column it also assigns must match \
         against the stored value, not against the assignment"
    );
    // The same shape without the overlap still works -- the fix did not buy the
    // overlapping case at the cost of the ordinary one. Row 1 is now keyed 11.
    s.execute("UPDATE t SET v = v + 1 WHERE id = 11").unwrap();
    let v = s.query("SELECT v FROM t WHERE id = 11").unwrap().scalar().unwrap().as_i64();
    assert_eq!(v, Some(11), "a non-overlapping UPDATE regressed");
}

/// The two spellings, end to end through the public API, against sqlite3 for
/// the shapes the generator does not reach: an unconditional `DELETE`, and a
/// `DELETE`/`UPDATE` written the ClickHouse way. Both must leave the table in
/// the state sqlite leaves it in, and the ANSI and `ALTER` forms must leave it
/// in the *same* state as each other -- that is what "one implementation, two
/// spellings" has to mean in practice.
#[test]
fn both_mutation_spellings_agree_with_sqlite_and_with_each_other() {
    if skip_without_sqlite() {
        return;
    }
    let table = TableDef {
        name: "t0".into(),
        cols: vec![
            ColDef { name: "id".into(), ty: Ty::Int, nullable: false },
            ColDef { name: "k".into(), ty: Ty::Int, nullable: true },
            ColDef { name: "a0".into(), ty: Ty::Text, nullable: true },
        ],
        rows: (0..7)
            .map(|i| {
                vec![
                    Cell::Int(i),
                    if i % 3 == 0 { Cell::Null } else { Cell::Int(i * 2) },
                    Cell::Text(format!("s{i}")),
                ]
            })
            .collect(),
        sort: SortKey::Id,
        chunk: 3,
        optimize: false,
        keyed: true,
    };
    let query = Query {
        from: From::One(0),
        star: false,
        distinct: false,
        items: vec![E::Col(0, 0), E::Col(0, 1), E::Col(0, 2)],
        filter: None,
        group_by: Vec::new(),
        having: None,
        set_tail: None,
        order: vec![(1, true, true)],
        limit: None,
        offset: 0,
    };
    // `pred: None` on a DELETE is the shape `gen_mutations` refuses to emit,
    // because it empties the table and leaves the query nothing to read.
    let pred = || Some(E::Bin("<", E::Bare("k".into(), Ty::Int).b(), E::Lit(Cell::Int(8)).b()));
    let cases: Vec<(&str, Vec<Mutation>)> = vec![
        ("delete all", vec![Mutation { slot: 0, set: vec![], pred: None, then_optimize: true }]),
        ("delete some", vec![Mutation { slot: 0, set: vec![], pred: pred(), then_optimize: false }]),
        (
            "update then delete",
            vec![
                Mutation {
                    slot: 0,
                    set: vec![(2, E::Lit(Cell::Text("z".into())))],
                    pred: pred(),
                    then_optimize: false,
                },
                Mutation { slot: 0, set: vec![], pred: None, then_optimize: false },
            ],
        ),
    ];
    for (what, mutations) in cases {
        let case = Case {
            seed: 0,
            tables: vec![table.clone()],
            mutations,
            query: query.clone(),
        };
        let g = run_granular(&case).unwrap_or_else(|e| panic!("{what}: granular refused: {e}"));
        let s = sqlite_one(&case).unwrap_or_else(|e| panic!("{what}: sqlite refused: {e}"));
        assert!(
            compare(&case, Ok(g.clone()), Ok(s)).is_none(),
            "{what}:\n{}",
            case.script(Dialect::Granular)
        );

        // Same statements, ClickHouse spelling, same answer.
        let mut alter = Session::in_memory();
        for stmt in case.setup(Dialect::Granular) {
            let ch = if let Some(rest) = stmt.strip_prefix("DELETE FROM t0") {
                format!("ALTER TABLE t0 DELETE{}", if rest.is_empty() { " WHERE 1 = 1" } else { rest })
            } else if let Some(rest) = stmt.strip_prefix("UPDATE t0 SET ") {
                format!("ALTER TABLE t0 UPDATE {rest}")
            } else {
                stmt
            };
            alter.execute(&ch).unwrap_or_else(|e| panic!("{what}: `{ch}`: {e}"));
        }
        let got: Vec<Vec<Cell>> = alter
            .query(&case.select(Dialect::Granular))
            .unwrap()
            .to_values()
            .iter()
            .map(|r| r.iter().map(cell_of_value).collect())
            .collect();
        assert_eq!(fmt_rows(&got), fmt_rows(&g), "{what}: the two spellings diverged");
    }
}
