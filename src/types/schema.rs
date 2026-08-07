//! Table and result-set schemas.

use super::datatype::DataType;
use super::value::{parse_date, parse_datetime, Value};
use crate::common::{Error, FastMap, Result};
use std::fmt;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Field {
    pub name: String,
    pub ty: DataType,
    /// `DEFAULT <literal>`, already evaluated and cast to `ty`.
    ///
    /// Private, and that is the whole point. This used to be an
    /// `Option<String>` holding the default's SQL text, which nothing on the
    /// insert path knew how to evaluate: every omitted column was filled with
    /// the type's zero while `SHOW CREATE TABLE` cheerfully echoed the
    /// `DEFAULT` back, so the data was wrong at ingest and unrecoverable
    /// afterwards. Storing a `Value` that can only be produced by
    /// [`Field::with_default`] makes "declared but never evaluated"
    /// unrepresentable, and leaves the fill path a scalar clone rather than a
    /// parse (the insert path is ~60ns/row; a re-parse there is not affordable).
    default: Option<Value>,
}

impl Field {
    pub fn new(name: impl Into<String>, ty: DataType) -> Self {
        Field { name: name.into(), ty, default: None }
    }

    /// Attach `DEFAULT <lit>` from its SQL text, evaluating it **once**, here.
    ///
    /// Both failure modes are DDL errors by design: `DEFAULT 'abc'` on an
    /// `Int64` is rejected now rather than at the first insert, and a
    /// non-constant default (`DEFAULT now()`) is reported as unsupported
    /// rather than accepted and silently ignored.
    pub fn with_default(self, lit: &str) -> Result<Self> {
        let v = parse_sql_literal(lit).ok_or_else(|| {
            Error::unsupported(format!(
                "DEFAULT for column `{}` must be a constant literal, got `{lit}`",
                self.name
            ))
        })?;
        self.with_default_value(v)
    }

    /// [`Field::with_default`] for a caller that already holds the literal as a
    /// [`Value`] (the parser's `Expr::Literal`), skipping the render/re-parse.
    pub fn with_default_value(mut self, v: Value) -> Result<Self> {
        let value = coerce_default(&v, &self.ty).map_err(|e| {
            Error::bind(format!("bad DEFAULT {v} for column `{}` {}: {e}", self.name, self.ty))
        })?;
        self.default = Some(value);
        Ok(self)
    }

    /// The evaluated default, ready to hand to `Column::constant`.
    #[inline]
    pub fn default_value(&self) -> Option<&Value> {
        self.default.as_ref()
    }

    /// The default rendered back to SQL text: what `SHOW CREATE TABLE` prints
    /// and what the catalog stores. Round-trips through [`Field::with_default`].
    ///
    /// Temporal values get quoted, which `Value`'s own `Display` does not do —
    /// a bare `DEFAULT 2024-01-01` is three integers and a subtraction.
    pub fn default_sql(&self) -> Option<String> {
        self.default.as_ref().map(|v| match v {
            Value::Date(_) | Value::DateTime(_) => format!("'{}'", v.render_plain()),
            other => other.to_string(),
        })
    }

    /// What an INSERT that omits this column stores: the `DEFAULT` if there is
    /// one, else NULL for a nullable column, else the type's zero.
    ///
    /// Costs one `Value` clone (an `Arc` bump for strings) per omitted column
    /// per *block*, not per row — callers expand it with `Column::constant`.
    pub fn fill_value(&self) -> Value {
        match &self.default {
            Some(v) => v.clone(),
            None if self.ty.is_nullable() => Value::Null,
            None => self.ty.zero_value(),
        }
    }
}

/// Parse the SQL text of a *literal*.
///
/// Deliberately not an expression parser: this layer has no binder, and
/// anything it cannot fold to a constant here must be refused at DDL time
/// rather than stored unevaluated. `None` means "not a literal".
///
/// The accepted forms are exactly what `Expr`'s `Display` emits for constant
/// expressions plus what a user writes, so `DEFAULT` text survives the
/// parse -> catalog -> reload round trip: `NULL`, `TRUE`/`FALSE`, `'quoted'`
/// with `''` escapes, and signed integers/floats (including `inf`/`nan`, which
/// `Value`'s float rendering can produce).
fn parse_sql_literal(text: &str) -> Option<Value> {
    let t = text.trim();
    if t.eq_ignore_ascii_case("null") {
        return Some(Value::Null);
    }
    if t.eq_ignore_ascii_case("true") {
        return Some(Value::Bool(true));
    }
    if t.eq_ignore_ascii_case("false") {
        return Some(Value::Bool(false));
    }
    if let Some(rest) = t.strip_prefix('\'') {
        let body = rest.strip_suffix('\'')?;
        let mut out = String::with_capacity(body.len());
        let mut it = body.chars();
        while let Some(c) = it.next() {
            if c == '\'' {
                // A quote inside the body is only legal doubled; a lone one
                // means the literal ended early and this is an expression
                // (`'a' || 'b'`), which we must not accept as a constant.
                if it.next() != Some('\'') {
                    return None;
                }
                out.push('\'');
            } else {
                out.push(c);
            }
        }
        return Some(Value::str(out));
    }
    // Integers keep their exact width: routing them through f64 would round
    // a u64 default past 2^53. Unsigned first so large u64s stay unsigned.
    if !t.contains(['.', 'e', 'E']) {
        if let Ok(u) = t.parse::<u64>() {
            return Some(Value::UInt(u));
        }
        if let Ok(i) = t.parse::<i64>() {
            return Some(Value::Int(i));
        }
    }
    t.parse::<f64>().ok().map(Value::Float)
}

/// Cast a literal to the column type with the same rules the INSERT path uses,
/// so `DEFAULT '2024-01-01'` on a `Date` means the date, not a failed integer
/// parse. Kept next to the constructor because the check is the DDL gate:
/// anything that would fail per-row at insert must fail here instead.
fn coerce_default(v: &Value, ty: &DataType) -> Result<Value> {
    if v.is_null() {
        return if ty.is_nullable() {
            Ok(Value::Null)
        } else {
            Err(Error::bind(format!("cannot store NULL in non-nullable {ty}")))
        };
    }
    match (v, ty.base()) {
        (Value::Str(s), DataType::Date) => Ok(Value::Date(parse_date(s)?)),
        (Value::Str(s), DataType::DateTime) => Ok(Value::DateTime(parse_datetime(s)?)),
        _ => v.cast_to(ty),
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
pub struct Schema {
    fields: Vec<Field>,
    index: FastMap<String, usize>,
}

impl Schema {
    pub fn new(fields: Vec<Field>) -> Result<Self> {
        let mut index = FastMap::default();
        for (i, f) in fields.iter().enumerate() {
            if index.insert(f.name.clone(), i).is_some() {
                return Err(Error::bind(format!("duplicate column `{}`", f.name)));
            }
        }
        Ok(Schema { fields, index })
    }

    /// Build without the duplicate check, for internally-generated schemas
    /// (projections may legitimately repeat an expression's name).
    pub fn new_unchecked(fields: Vec<Field>) -> Self {
        let mut index = FastMap::default();
        for (i, f) in fields.iter().enumerate() {
            index.entry(f.name.clone()).or_insert(i);
        }
        Schema { fields, index }
    }

    pub fn empty() -> Self {
        Schema::default()
    }

    pub fn fields(&self) -> &[Field] {
        &self.fields
    }
    pub fn len(&self) -> usize {
        self.fields.len()
    }
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
    pub fn field(&self, i: usize) -> &Field {
        &self.fields[i]
    }
    pub fn name(&self, i: usize) -> &str {
        &self.fields[i].name
    }
    pub fn ty(&self, i: usize) -> &DataType {
        &self.fields[i].ty
    }

    /// Case-sensitive first (ClickHouse semantics), then a case-insensitive
    /// fallback so `SELECT ID` finds `id` instead of erroring unhelpfully.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        if let Some(&i) = self.index.get(name) {
            return Some(i);
        }
        let mut found = None;
        for (i, f) in self.fields.iter().enumerate() {
            if f.name.eq_ignore_ascii_case(name) {
                if found.is_some() {
                    return None; // ambiguous
                }
                found = Some(i);
            }
        }
        found
    }

    pub fn require(&self, name: &str) -> Result<usize> {
        self.index_of(name).ok_or_else(|| {
            let names: Vec<&str> = self.fields.iter().map(|f| f.name.as_str()).collect();
            Error::bind(format!(
                "unknown column `{name}`; available: {}",
                names.join(", ")
            ))
        })
    }

    pub fn push(&mut self, f: Field) {
        self.index.entry(f.name.clone()).or_insert(self.fields.len());
        self.fields.push(f);
    }

    /// Schema of a subset of columns, in the given order.
    pub fn project(&self, cols: &[usize]) -> Schema {
        Schema::new_unchecked(cols.iter().map(|&i| self.fields[i].clone()).collect())
    }

    /// Concatenate two schemas, for join outputs.
    pub fn concat(&self, other: &Schema) -> Schema {
        let mut f = self.fields.clone();
        f.extend(other.fields.iter().cloned());
        Schema::new_unchecked(f)
    }
}

impl fmt::Debug for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Schema[")?;
        for (i, fl) in self.fields.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{} {}", fl.name, fl.ty)?;
        }
        write!(f, "]")
    }
}

/// How a MergeTree-style table is laid out on disk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableDef {
    pub name: String,
    pub schema: Schema,
    /// Column indices forming the sort key. Rows within a part are stored in
    /// this order, which is what makes zone maps and range pruning work.
    pub order_by: Vec<usize>,
    /// Column indices forming the primary key, or **empty when the table
    /// declared none**.
    ///
    /// Empty is not the same as `order_by`, and conflating the two is what
    /// made plain `INSERT`s lose rows: this key is a *uniqueness* claim (it
    /// drives the keyed delta, the MPH index and last-write-wins), whereas
    /// `order_by` is only a sort order and says nothing about duplicates. See
    /// [`TableDef::pk_col`] for the rule that reads it.
    pub primary_key: Vec<usize>,
    /// Optional partition-key column index (coarse pruning above zone maps).
    pub partition_by: Option<usize>,
    pub engine: Engine,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Engine {
    /// Sorted and indexed. `ORDER BY` is a **sort key**: rows sharing one are
    /// duplicates, they are all kept, and `INSERT` is an append.
    ///
    /// This used to be a "deliberate divergence" from ClickHouse in which
    /// `MergeTree` replaced on the ORDER BY key. The divergence was a
    /// row-eating bug wearing a design rationale: `INSERT INTO t VALUES
    /// (4,1),(4,2)` on `ORDER BY id` reported two rows affected and stored
    /// one, with no error and no way to get the row back. A sort key is not a
    /// uniqueness constraint in any SQL dialect, and nothing in the DDL let
    /// the user say they wanted the other thing.
    ///
    /// Replacing is still available and still the OLTP story — it just has to
    /// be asked for, by `PRIMARY KEY` or by
    /// [`ReplacingMergeTree`](Engine::ReplacingMergeTree). See
    /// [`TableDef::pk_col`].
    #[default]
    MergeTree,
    /// `MergeTree` plus **replacing on the sort key**: an insert of an
    /// existing key tombstones the old row.
    ///
    /// No longer a synonym for [`MergeTree`](Engine::MergeTree) — it is the
    /// engine-level way to opt a table into the keyed delta and the 59 ns
    /// point-lookup path without naming a `PRIMARY KEY`, and it is the
    /// one-word migration for anyone who was relying on the old
    /// `MergeTree`-replaces-silently behaviour.
    ///
    /// Collapsing happens at write and at merge time rather than only under
    /// `FINAL`, which is stricter than ClickHouse and is what the MPH index
    /// over a part requires.
    ReplacingMergeTree,
    /// Accepted by the parser and the catalog for round-tripping, but
    /// **rejected at CREATE TABLE**: summing at merge time is not implemented,
    /// and silently replacing instead would return wrong sums.
    SummingMergeTree,
    /// No ordering, no index; append-only scratch. Duplicate keys are kept.
    Log,
    /// In-memory only, dropped on restart.
    Memory,
}

impl Engine {
    pub fn parse(s: &str) -> Result<Engine> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "mergetree" => Engine::MergeTree,
            "replacingmergetree" => Engine::ReplacingMergeTree,
            "summingmergetree" => Engine::SummingMergeTree,
            "log" | "tinylog" | "stripelog" => Engine::Log,
            "memory" => Engine::Memory,
            other => return Err(Error::unsupported(format!("engine `{other}`"))),
        })
    }
    pub fn name(&self) -> &'static str {
        match self {
            Engine::MergeTree => "MergeTree",
            Engine::ReplacingMergeTree => "ReplacingMergeTree",
            Engine::SummingMergeTree => "SummingMergeTree",
            Engine::Log => "Log",
            Engine::Memory => "Memory",
        }
    }
    /// Whether the engine keeps rows sorted and indexed.
    pub fn is_sorted(&self) -> bool {
        matches!(
            self,
            Engine::MergeTree | Engine::ReplacingMergeTree | Engine::SummingMergeTree
        )
    }
    pub fn is_persistent(&self) -> bool {
        !matches!(self, Engine::Memory)
    }
}

impl TableDef {
    /// The leading ORDER BY column, if the engine keeps rows sorted and that
    /// column's storage lane is order-preserving.
    ///
    /// String columns are excluded: their lanes are *per-granule* dictionary
    /// codes, so they order correctly within a granule but carry no meaning
    /// across granules. Sorting still works for them (the comparison happens
    /// on the values), but the sparse index, the router and range pruning by
    /// lane do not, so we report `None` and fall back to scanning.
    pub fn sort_col(&self) -> Option<usize> {
        if !self.engine.is_sorted() {
            return None;
        }
        let c = *self.order_by.first()?;
        let t = self.schema.ty(c);
        if t.is_nullable() || t.is_string() {
            return None;
        }
        Some(c)
    }

    /// The column the *unique-key* machinery keys on, if the table has one.
    ///
    /// This one predicate decides everything downstream: whether writes go to
    /// the keyed delta (last-write-wins, one slot per key) or are appended,
    /// whether a part carries an MPH index, whether duplicates are collapsed
    /// at merge, and whether `ALTER ... UPDATE/DELETE` are available.
    ///
    /// ## Why a declaration is required, and ORDER BY is not one
    ///
    /// This used to be `primary_key.len() == 1 && ...`, with `primary_key`
    /// defaulted to `order_by` at CREATE TABLE. So `ORDER BY id` alone made
    /// the table keyed, and `INSERT INTO t VALUES (4,1),(4,2)` reported two
    /// rows and stored one — the second `put_keyed` overwrote the slot the
    /// first owned. Silent, unrecoverable row loss on the most ordinary
    /// statement there is; the differential harness produced it in 12 of its
    /// first 36 cases.
    ///
    /// The three candidate fixes, and why this one:
    ///
    ///   * *Make the keyed delta multi-valued* (key -> list of slots). It
    ///     fixes the delta and nothing else: the MPH index over a part is a
    ///     **minimal perfect hash**, which is only defined on distinct keys,
    ///     and the merge path collapses runs of equal keys as well. Duplicates
    ///     and the point-lookup index are mutually exclusive by construction,
    ///     one level below this decision, so there is no version of this that
    ///     keeps both. It would also cost the row-major lane arena its
    ///     flatness, which is a measured 28 ns/row win.
    ///   * *Key off the engine only* (`ReplacingMergeTree` replaces,
    ///     `MergeTree` appends). Faithful to ClickHouse, but it leaves no way
    ///     to say "unique key" in a `MergeTree` table, and every table built
    ///     through the storage API rather than through DDL loses the OLTP path
    ///     with no opt-in short of changing engine.
    ///   * **This: uniqueness is a declaration.** An explicit `PRIMARY KEY`
    ///     asserts it per-table; `ReplacingMergeTree` asserts it per-engine.
    ///     `ORDER BY` asserts only an order, which is what it means in SQL and
    ///     in ClickHouse, so a table that declares neither keeps its
    ///     duplicates.
    ///
    /// The cost is real and is the price of not losing rows: an
    /// `ORDER BY`-only table gets no point-lookup index, and `ALTER TABLE ...
    /// UPDATE/DELETE` refuse it with "requires a single-column primary key".
    /// Both are loud, and either declaration turns them back on.
    ///
    /// The rest of the predicate is unchanged and still load-bearing. The key
    /// must be a single column, because the lane index holds one `u64` per
    /// row; non-nullable and non-string, because a lane must be
    /// order-preserving across granules (see [`TableDef::sort_col`]); and it
    /// must *lead* `order_by`, because `find_live` routes by the **sort** lane
    /// and then probes the **primary key** index — if the two disagree the
    /// router sends lookups to the wrong granule.
    pub fn pk_col(&self) -> Option<usize> {
        // Empty `primary_key` means "none declared", not "same as ORDER BY".
        let key: &[usize] = if !self.primary_key.is_empty() {
            &self.primary_key
        } else if self.engine == Engine::ReplacingMergeTree {
            &self.order_by
        } else {
            return None;
        };
        // `sort_col` covers engine-is-sorted, non-nullable and non-string, so
        // one comparison finishes the job -- and this is the whole body of
        // `has_fast_pk`, which used to run the same checks a second time.
        match key {
            &[c] if self.sort_col() == Some(c) => Some(c),
            _ => None,
        }
    }

    /// True when [`TableDef::pk_col`] found a key: the table is unique-keyed
    /// and gets the MPH + learned-rank point-lookup path.
    #[inline]
    pub fn has_fast_pk(&self) -> bool {
        self.pk_col().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::UInt64),
            Field::new("name", DataType::String),
            Field::new("amount", DataType::Int64),
        ])
        .unwrap()
    }

    #[test]
    fn lookup_exact_then_case_insensitive() {
        let s = schema();
        assert_eq!(s.index_of("id"), Some(0));
        assert_eq!(s.index_of("ID"), Some(0));
        assert_eq!(s.index_of("missing"), None);
        assert!(s.require("missing").unwrap_err().to_string().contains("id, name, amount"));
    }

    #[test]
    fn duplicate_columns_rejected() {
        let r = Schema::new(vec![
            Field::new("x", DataType::UInt64),
            Field::new("x", DataType::UInt64),
        ]);
        assert!(r.is_err());
    }

    #[test]
    fn ambiguous_case_insensitive_match_is_none() {
        let s = Schema::new(vec![
            Field::new("x", DataType::UInt64),
            Field::new("X", DataType::UInt64),
        ])
        .unwrap();
        assert_eq!(s.index_of("x"), Some(0)); // exact wins
        assert_eq!(s.index_of("xX"), None);
    }

    #[test]
    fn project_and_concat() {
        let s = schema();
        let p = s.project(&[2, 0]);
        assert_eq!(p.name(0), "amount");
        assert_eq!(p.name(1), "id");
        assert_eq!(s.concat(&p).len(), 5);
    }

    /// The bug this pins: `ORDER BY id` alone used to make the table
    /// unique-keyed, so `INSERT INTO t VALUES (4,1),(4,2)` kept one row.
    /// A sort key is not a uniqueness claim; only a declaration is.
    #[test]
    fn a_sort_key_alone_is_not_a_unique_key() {
        let mut def = TableDef {
            name: "t".into(),
            schema: schema(),
            order_by: vec![0],
            primary_key: Vec::new(), // ORDER BY id, and nothing else
            partition_by: None,
            engine: Engine::MergeTree,
        };
        assert_eq!(def.pk_col(), None, "ORDER BY must not imply uniqueness");
        assert_eq!(def.sort_col(), Some(0), "...but it is still the sort order");

        // Declaring it per-table turns the keyed path back on.
        def.primary_key = vec![0];
        assert_eq!(def.pk_col(), Some(0));

        // ...and so does declaring it per-engine, with no PRIMARY KEY at all.
        def.primary_key = Vec::new();
        def.engine = Engine::ReplacingMergeTree;
        assert_eq!(def.pk_col(), Some(0));

        // The engine opt-in obeys the same shape rules as an explicit key: a
        // composite ORDER BY has no single lane to index by.
        def.order_by = vec![0, 2];
        assert_eq!(def.pk_col(), None);
        assert_eq!(def.sort_col(), Some(0), "still sorted, just not keyed");
    }

    #[test]
    fn fast_pk_detection() {
        let mut def = TableDef {
            name: "t".into(),
            schema: schema(),
            order_by: vec![0],
            primary_key: vec![0],
            partition_by: None,
            engine: Engine::MergeTree,
        };
        assert!(def.has_fast_pk());
        assert_eq!(def.pk_col(), Some(0));
        assert_eq!(def.sort_col(), Some(0));

        // Signed columns are fine: their lanes are sign-flipped, not zigzag,
        // so they stay order-preserving.
        def.order_by = vec![2];
        def.primary_key = vec![2];
        assert!(def.has_fast_pk());

        // A String sort key sorts correctly but gets no lane-based index.
        def.order_by = vec![1];
        def.primary_key = vec![1];
        assert_eq!(def.sort_col(), None);
        assert!(!def.has_fast_pk());

        // PK must lead the sort order, or the router points at the wrong granule.
        def.order_by = vec![0];
        def.primary_key = vec![2];
        assert!(!def.has_fast_pk());

        def.primary_key = vec![0, 2]; // composite
        assert!(!def.has_fast_pk());

        def.primary_key = vec![0];
        def.engine = Engine::Log;
        assert!(!def.has_fast_pk());
        assert_eq!(def.sort_col(), None);
    }

    // ------------------------------------------------------------- defaults
    // The bug these pin: `DEFAULT` used to be stored as SQL text that nothing
    // ever evaluated, so `INSERT INTO t (id) VALUES (1)` wrote the type's zero
    // into every defaulted column while SHOW CREATE TABLE claimed otherwise.

    fn defaulted(ty: DataType, lit: &str) -> Field {
        Field::new("c", ty).with_default(lit).unwrap()
    }

    #[test]
    fn default_is_evaluated_at_ddl_not_at_insert() {
        let f = defaulted(DataType::String, "'hello'");
        assert_eq!(f.default_value(), Some(&Value::str("hello")));
        assert_eq!(f.fill_value(), Value::str("hello"));

        let f = defaulted(DataType::Int64, "42");
        assert_eq!(f.default_value(), Some(&Value::Int(42)));
        assert_eq!(f.fill_value(), Value::Int(42));

        // No default: nullable fills NULL, everything else the type's zero.
        let f = Field::new("c", DataType::Int64);
        assert_eq!(f.default_value(), None);
        assert_eq!(f.fill_value(), Value::Int(0));
        assert!(Field::new("c", DataType::Int64.to_nullable()).fill_value().is_null());
        assert_eq!(Field::new("c", DataType::String).fill_value(), Value::str(""));
    }

    #[test]
    fn default_is_cast_to_the_column_type() {
        // The stored value is already in the column's own representation, so
        // the fill needs no coercion per block. `Value`'s Eq collapses numeric
        // representations, hence the `matches!` on the variant: a `Date`
        // column holding `UInt(19723)` would lane-encode wrong.
        assert!(matches!(
            defaulted(DataType::UInt8, "7").default_value(),
            Some(Value::UInt(7))
        ));
        assert!(matches!(
            defaulted(DataType::Float32, "1.5").default_value(),
            Some(Value::Float(f)) if *f == 1.5
        ));
        assert!(matches!(
            defaulted(DataType::Bool, "true").default_value(),
            Some(Value::Bool(true))
        ));
        // A wide integer literal narrows to the declared type, not to f64.
        assert!(matches!(
            defaulted(DataType::Int16, "7").default_value(),
            Some(Value::Int(7))
        ));
        // Strings adopt the column type, matching what VALUES does.
        assert!(matches!(
            defaulted(DataType::Date, "'2024-01-01'").default_value(),
            Some(Value::Date(19_723))
        ));
        assert!(matches!(
            defaulted(DataType::DateTime, "'1970-01-01 00:00:10'").default_value(),
            Some(Value::DateTime(10))
        ));
        // Nullable columns take a literal or NULL.
        let n = DataType::Int64.to_nullable();
        assert_eq!(defaulted(n.clone(), "-1").default_value(), Some(&Value::Int(-1)));
        assert_eq!(defaulted(n, "NULL").default_value(), Some(&Value::Null));
    }

    #[test]
    fn bad_default_is_rejected_by_ddl() {
        let err = |ty: DataType, lit: &str| Field::new("c", ty).with_default(lit).unwrap_err();
        // Wrong type: the whole reason this check moved to CREATE TABLE.
        assert!(err(DataType::Int64, "'abc'").to_string().contains("DEFAULT"));
        assert!(err(DataType::Float64, "'abc'").to_string().contains("DEFAULT"));
        // Out of range for the declared width.
        assert!(err(DataType::UInt8, "300").to_string().contains("DEFAULT"));
        assert!(err(DataType::UInt8, "-1").to_string().contains("DEFAULT"));
        // NULL needs a Nullable column.
        assert!(err(DataType::Int64, "NULL").to_string().contains("DEFAULT"));
        // Not a date.
        assert!(err(DataType::Date, "'not-a-date'").to_string().contains("DEFAULT"));
        // Non-constant: accepted-and-ignored is what produced silently wrong
        // rows, so it is an error until there is something to evaluate it.
        for lit in ["now()", "a + 1", "'a' || 'b'", "", "rand"] {
            let e = err(DataType::String, lit);
            assert!(
                e.to_string().contains("constant literal"),
                "`{lit}` should be refused as non-constant, got {e}"
            );
        }
    }

    #[test]
    fn default_round_trips_through_sql_text() {
        // The catalog stores `default_sql()` and reloads it through
        // `with_default`, so the pair must be exact for every literal kind.
        for (ty, lit) in [
            (DataType::String, "'hello'"),
            (DataType::String, "'it''s'"),
            (DataType::String, "''"),
            (DataType::UInt64, "18446744073709551615"),
            (DataType::Int64, "-9223372036854775808"),
            (DataType::Float64, "1.5"),
            (DataType::Float64, "-0.25"),
            (DataType::Bool, "true"),
            (DataType::Date, "'2024-02-29'"),
            (DataType::DateTime, "'2024-01-15 13:45:30'"),
            (DataType::Nullable(Box::new(DataType::Int64)), "NULL"),
            (DataType::Nullable(Box::new(DataType::Float32)), "1.5"),
        ] {
            let f = defaulted(ty.clone(), lit);
            let text = f.default_sql().expect("a default was set");
            let back = defaulted(ty.clone(), &text);
            assert_eq!(back.default_value(), f.default_value(), "{ty} DEFAULT {lit}");
            assert_eq!(back.default_sql(), Some(text.clone()), "{ty} DEFAULT {lit} is not a fixpoint");
        }
        assert_eq!(Field::new("c", DataType::UInt8).default_sql(), None);
    }

    #[test]
    fn u64_default_keeps_full_width() {
        // Parsing every integer through f64 would round this to 2^64.
        let f = defaulted(DataType::UInt64, "18446744073709551615");
        assert_eq!(f.default_value(), Some(&Value::UInt(u64::MAX)));
    }

    #[test]
    fn engine_parse() {
        assert_eq!(Engine::parse("MergeTree").unwrap(), Engine::MergeTree);
        assert_eq!(Engine::parse("memory").unwrap(), Engine::Memory);
        assert!(Engine::parse("Kafka").is_err());
        assert!(Engine::MergeTree.is_sorted());
        assert!(!Engine::Memory.is_persistent());
    }
}
