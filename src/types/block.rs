//! Vectorized batches: the unit every execution operator consumes and
//! produces.
//!
//! A `Block` is a set of equal-length columns plus a row count. Operators pull
//! blocks (`BLOCK_SIZE` rows by default) rather than tuples, so a filter is a
//! tight loop over a `&[i64]` and an aggregate is a strided reduction. This is
//! the same shape ClickHouse uses, and it is why `SUM` runs at memory
//! bandwidth instead of at interpreter speed.
//!
//! Null representation: a set bit in `nulls` means the row **is NULL**. A
//! column with no nulls stores `None` and costs zero bytes, matching the
//! delete-mask convention in storage.

use super::datatype::{DataType, PhysicalType};
use super::schema::Schema;
use super::value::Value;
use crate::common::{lane_to_f64, lane_to_i64, BitSet, Error, Result};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub enum ColumnData {
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    Str(Vec<Arc<str>>),
}

impl ColumnData {
    pub fn len(&self) -> usize {
        match self {
            ColumnData::U64(v) => v.len(),
            ColumnData::I64(v) => v.len(),
            ColumnData::F64(v) => v.len(),
            ColumnData::Str(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn physical(&self) -> PhysicalType {
        match self {
            ColumnData::U64(_) => PhysicalType::U64,
            ColumnData::I64(_) => PhysicalType::I64,
            ColumnData::F64(_) => PhysicalType::F64,
            ColumnData::Str(_) => PhysicalType::Str,
        }
    }
    /// An empty buffer of the same physical kind.
    pub fn empty_like(&self) -> ColumnData {
        match self {
            ColumnData::U64(_) => ColumnData::U64(Vec::new()),
            ColumnData::I64(_) => ColumnData::I64(Vec::new()),
            ColumnData::F64(_) => ColumnData::F64(Vec::new()),
            ColumnData::Str(_) => ColumnData::Str(Vec::new()),
        }
    }
    pub fn for_physical(p: PhysicalType) -> ColumnData {
        match p {
            PhysicalType::U64 => ColumnData::U64(Vec::new()),
            PhysicalType::I64 => ColumnData::I64(Vec::new()),
            PhysicalType::F64 => ColumnData::F64(Vec::new()),
            PhysicalType::Str => ColumnData::Str(Vec::new()),
        }
    }
    pub fn truncate(&mut self, n: usize) {
        match self {
            ColumnData::U64(v) => v.truncate(n),
            ColumnData::I64(v) => v.truncate(n),
            ColumnData::F64(v) => v.truncate(n),
            ColumnData::Str(v) => v.truncate(n),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Column {
    pub ty: DataType,
    pub data: ColumnData,
    /// Set bit == NULL. `None` means "no nulls", and costs nothing.
    pub nulls: Option<BitSet>,
}

impl Column {
    pub fn new(ty: DataType, data: ColumnData) -> Column {
        debug_assert_eq!(
            ty.physical(),
            data.physical(),
            "column data kind does not match declared type {ty}"
        );
        Column { ty, data, nulls: None }
    }

    pub fn with_nulls(ty: DataType, data: ColumnData, nulls: BitSet) -> Column {
        let nulls = if nulls.is_empty() { None } else { Some(nulls) };
        Column { ty, data, nulls }
    }

    pub fn u64s(ty: DataType, v: Vec<u64>) -> Column {
        Column::new(ty, ColumnData::U64(v))
    }
    pub fn i64s(ty: DataType, v: Vec<i64>) -> Column {
        Column::new(ty, ColumnData::I64(v))
    }
    pub fn f64s(ty: DataType, v: Vec<f64>) -> Column {
        Column::new(ty, ColumnData::F64(v))
    }
    pub fn strs(ty: DataType, v: Vec<Arc<str>>) -> Column {
        Column::new(ty, ColumnData::Str(v))
    }
    pub fn bools(v: Vec<u64>) -> Column {
        Column::new(DataType::Bool, ColumnData::U64(v))
    }

    /// A column of `n` copies of `v`. Used for constant folding.
    pub fn constant(ty: &DataType, v: &Value, n: usize) -> Result<Column> {
        if v.is_null() {
            let mut b = ColumnBuilder::new(ty.to_nullable());
            for _ in 0..n {
                b.push_null();
            }
            return Ok(b.finish());
        }
        Ok(match ty.base().physical() {
            PhysicalType::U64 => Column::u64s(
                ty.clone(),
                vec![v.as_u64().ok_or_else(|| Error::exec(format!("{v} is not a {ty}")))?; n],
            ),
            PhysicalType::I64 => Column::i64s(
                ty.clone(),
                vec![v.as_i64().ok_or_else(|| Error::exec(format!("{v} is not a {ty}")))?; n],
            ),
            PhysicalType::F64 => Column::f64s(
                ty.clone(),
                vec![v.as_f64().ok_or_else(|| Error::exec(format!("{v} is not a {ty}")))?; n],
            ),
            PhysicalType::Str => {
                let s: Arc<str> = v.render_plain().into();
                Column::strs(ty.clone(), vec![s; n])
            }
        })
    }

    /// Append one value, coercing it to this column's declared type.
    ///
    /// The same coercion [`ColumnBuilder`] performs, but on a live column, so
    /// a growable typed buffer can be written to and read back without going
    /// through a builder. The write memtable is built on this.
    pub fn push_value(&mut self, v: &Value) -> Result<()> {
        // Exact-type fast path: one match over both the buffer kind and the
        // value, instead of a null check, a match on the buffer, and a match
        // inside `as_u64`/`as_i64`/`as_f64` -- three matches on the same value
        // where one will do. This is the innermost step of every write.
        match (&mut self.data, v) {
            (ColumnData::U64(b), Value::UInt(x)) => {
                b.push(*x);
                return Ok(());
            }
            (ColumnData::I64(b), Value::Int(x)) => {
                b.push(*x);
                return Ok(());
            }
            (ColumnData::F64(b), Value::Float(x)) => {
                b.push(*x);
                return Ok(());
            }
            (ColumnData::Str(b), Value::Str(x)) => {
                b.push(x.clone());
                return Ok(());
            }
            _ => {}
        }
        let at = self.len();
        if v.is_null() {
            self.nulls.get_or_insert_with(BitSet::new).set(at);
            match &mut self.data {
                ColumnData::U64(b) => b.push(0),
                ColumnData::I64(b) => b.push(0),
                ColumnData::F64(b) => b.push(0.0),
                ColumnData::Str(b) => b.push("".into()),
            }
            return Ok(());
        }
        match &mut self.data {
            ColumnData::U64(b) => b.push(
                v.as_u64()
                    .ok_or_else(|| Error::exec(format!("{v} is not a {}", self.ty)))?,
            ),
            ColumnData::I64(b) => b.push(
                v.as_i64()
                    .ok_or_else(|| Error::exec(format!("{v} is not a {}", self.ty)))?,
            ),
            ColumnData::F64(b) => b.push(
                v.as_f64()
                    .ok_or_else(|| Error::exec(format!("{v} is not a {}", self.ty)))?,
            ),
            ColumnData::Str(b) => b.push(match v {
                Value::Str(s) => s.clone(),
                other => other.render_plain().into(),
            }),
        }
        Ok(())
    }

    /// Overwrite row `i`. Used when a keyed write replaces a buffered row.
    pub fn set_value(&mut self, i: usize, v: &Value) -> Result<()> {
        // Same exact-type fast path as `push_value`; an overwrite is the other
        // half of a keyed write.
        if self.nulls.is_none() {
            match (&mut self.data, v) {
                (ColumnData::U64(b), Value::UInt(x)) => {
                    b[i] = *x;
                    return Ok(());
                }
                (ColumnData::I64(b), Value::Int(x)) => {
                    b[i] = *x;
                    return Ok(());
                }
                (ColumnData::F64(b), Value::Float(x)) => {
                    b[i] = *x;
                    return Ok(());
                }
                (ColumnData::Str(b), Value::Str(x)) => {
                    b[i] = x.clone();
                    return Ok(());
                }
                _ => {}
            }
        }
        if v.is_null() {
            self.nulls.get_or_insert_with(BitSet::new).set(i);
            return Ok(());
        }
        if let Some(n) = self.nulls.as_mut() {
            n.clear(i);
        }
        match &mut self.data {
            ColumnData::U64(b) => {
                b[i] = v
                    .as_u64()
                    .ok_or_else(|| Error::exec(format!("{v} is not a {}", self.ty)))?
            }
            ColumnData::I64(b) => {
                b[i] = v
                    .as_i64()
                    .ok_or_else(|| Error::exec(format!("{v} is not a {}", self.ty)))?
            }
            ColumnData::F64(b) => {
                b[i] = v
                    .as_f64()
                    .ok_or_else(|| Error::exec(format!("{v} is not a {}", self.ty)))?
            }
            ColumnData::Str(b) => {
                b[i] = match v {
                    Value::Str(s) => s.clone(),
                    other => other.render_plain().into(),
                }
            }
        }
        Ok(())
    }

    /// Drop every row but keep the allocation, so a streaming scan can refill
    /// the same buffer instead of allocating a block per batch.
    pub fn clear(&mut self) {
        self.data.truncate(0);
        self.nulls = None;
    }

    /// Reserve room for `n` more rows. Scans call this once per output block
    /// so appending a granule never reallocates mid-decode.
    pub fn reserve(&mut self, n: usize) {
        match &mut self.data {
            ColumnData::U64(v) => v.reserve(n),
            ColumnData::I64(v) => v.reserve(n),
            ColumnData::F64(v) => v.reserve(n),
            ColumnData::Str(v) => v.reserve(n),
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline(always)]
    pub fn is_null(&self, i: usize) -> bool {
        self.nulls.as_ref().is_some_and(|n| n.get(i))
    }

    pub fn has_nulls(&self) -> bool {
        self.nulls.is_some()
    }

    pub fn null_count(&self) -> usize {
        self.nulls.as_ref().map_or(0, |n| n.count_ones_upto(self.len()))
    }

    /// Materialize one cell. Slow path -- for result rendering and group keys,
    /// never for bulk arithmetic.
    pub fn value(&self, i: usize) -> Value {
        if self.is_null(i) {
            return Value::Null;
        }
        match (&self.data, self.ty.base()) {
            (ColumnData::U64(v), DataType::Bool) => Value::Bool(v[i] != 0),
            (ColumnData::U64(v), DataType::Date) => Value::Date(v[i] as u32),
            (ColumnData::U64(v), DataType::DateTime) => Value::DateTime(v[i] as i64),
            (ColumnData::U64(v), _) => Value::UInt(v[i]),
            (ColumnData::I64(v), DataType::DateTime) => Value::DateTime(v[i]),
            // A decimal lane is a unit count; the scale lives only in the type,
            // so this is where it is put back on.
            //
            // Free: both arms are reached through the same jump table on the
            // `DataType` discriminant, so `Int64` gains no branch. A/B
            // interleaved, best-of-80 x 11 runs, `--release`, 65_536 values --
            // `Int64` 3.94 ns/value before against 3.99 after (1.5% apart, on a
            // machine whose repeats of identical code span 258-348us), and a
            // `Decimal64(2)` column over lanes identical to an `Int64` one
            // reads at 3.68 ns/value against its 3.99.
            (ColumnData::I64(v), DataType::Decimal64(s)) => Value::Decimal(v[i], *s),
            (ColumnData::I64(v), _) => Value::Int(v[i]),
            (ColumnData::F64(v), _) => Value::Float(v[i]),
            (ColumnData::Str(v), _) => Value::Str(v[i].clone()),
        }
    }

    pub fn as_u64(&self) -> Result<&[u64]> {
        match &self.data {
            ColumnData::U64(v) => Ok(v),
            other => Err(Error::exec(format!(
                "expected u64 column, found {:?}",
                other.physical()
            ))),
        }
    }
    pub fn as_i64(&self) -> Result<&[i64]> {
        match &self.data {
            ColumnData::I64(v) => Ok(v),
            other => Err(Error::exec(format!(
                "expected i64 column, found {:?}",
                other.physical()
            ))),
        }
    }
    pub fn as_f64(&self) -> Result<&[f64]> {
        match &self.data {
            ColumnData::F64(v) => Ok(v),
            other => Err(Error::exec(format!(
                "expected f64 column, found {:?}",
                other.physical()
            ))),
        }
    }
    pub fn as_str(&self) -> Result<&[Arc<str>]> {
        match &self.data {
            ColumnData::Str(v) => Ok(v),
            other => Err(Error::exec(format!(
                "expected string column, found {:?}",
                other.physical()
            ))),
        }
    }

    /// Read every row as `f64`, for arithmetic that has been promoted to
    /// float. Nulls become 0.0; callers must consult the null mask.
    pub fn to_f64_vec(&self) -> Result<Vec<f64>> {
        Ok(match &self.data {
            ColumnData::U64(v) => v.iter().map(|&x| x as f64).collect(),
            ColumnData::I64(v) => v.iter().map(|&x| x as f64).collect(),
            ColumnData::F64(v) => v.clone(),
            ColumnData::Str(_) => {
                return Err(Error::exec("cannot use a String column as a number"))
            }
        })
    }

    /// Read every row as `i64`. Unsigned values above `i64::MAX` wrap; this is
    /// only used after the planner has proven the operands fit.
    pub fn to_i64_vec(&self) -> Result<Vec<i64>> {
        Ok(match &self.data {
            ColumnData::U64(v) => v.iter().map(|&x| x as i64).collect(),
            ColumnData::I64(v) => v.clone(),
            ColumnData::F64(v) => v.iter().map(|&x| x as i64).collect(),
            ColumnData::Str(_) => {
                return Err(Error::exec("cannot use a String column as an integer"))
            }
        })
    }

    /// Gather rows by index. The universal reshaping primitive: filters,
    /// sorts, joins and limits all reduce to this.
    pub fn take(&self, idx: &[u32]) -> Column {
        let data = match &self.data {
            ColumnData::U64(v) => {
                ColumnData::U64(idx.iter().map(|&i| v[i as usize]).collect())
            }
            ColumnData::I64(v) => {
                ColumnData::I64(idx.iter().map(|&i| v[i as usize]).collect())
            }
            ColumnData::F64(v) => {
                ColumnData::F64(idx.iter().map(|&i| v[i as usize]).collect())
            }
            ColumnData::Str(v) => {
                ColumnData::Str(idx.iter().map(|&i| v[i as usize].clone()).collect())
            }
        };
        let nulls = self.nulls.as_ref().map(|n| {
            let mut out = BitSet::new();
            for (o, &i) in idx.iter().enumerate() {
                if n.get(i as usize) {
                    out.set(o);
                }
            }
            out
        });
        Column {
            ty: self.ty.clone(),
            data,
            nulls: nulls.filter(|n| !n.is_empty()),
        }
    }

    pub fn slice(&self, s: usize, e: usize) -> Column {
        let data = match &self.data {
            ColumnData::U64(v) => ColumnData::U64(v[s..e].to_vec()),
            ColumnData::I64(v) => ColumnData::I64(v[s..e].to_vec()),
            ColumnData::F64(v) => ColumnData::F64(v[s..e].to_vec()),
            ColumnData::Str(v) => ColumnData::Str(v[s..e].to_vec()),
        };
        let nulls = self.nulls.as_ref().map(|n| {
            let mut out = BitSet::new();
            for i in s..e {
                if n.get(i) {
                    out.set(i - s);
                }
            }
            out
        });
        Column {
            ty: self.ty.clone(),
            data,
            nulls: nulls.filter(|n| !n.is_empty()),
        }
    }

    /// Append `other` in place. Types must match.
    pub fn extend(&mut self, other: &Column) -> Result<()> {
        let base = self.len();
        match (&mut self.data, &other.data) {
            (ColumnData::U64(a), ColumnData::U64(b)) => a.extend_from_slice(b),
            (ColumnData::I64(a), ColumnData::I64(b)) => a.extend_from_slice(b),
            (ColumnData::F64(a), ColumnData::F64(b)) => a.extend_from_slice(b),
            (ColumnData::Str(a), ColumnData::Str(b)) => a.extend_from_slice(b),
            _ => {
                return Err(Error::exec(format!(
                    "cannot concatenate {} with {}",
                    self.ty, other.ty
                )))
            }
        }
        if other.nulls.is_some() {
            let n = self.nulls.get_or_insert_with(BitSet::new);
            for i in 0..other.len() {
                if other.is_null(i) {
                    n.set(base + i);
                }
            }
        }
        Ok(())
    }

    /// Decode a packed storage lane into this column's representation.
    #[inline(always)]
    pub fn push_lane(&mut self, lane: u64) {
        match &mut self.data {
            ColumnData::U64(v) => v.push(lane),
            ColumnData::I64(v) => v.push(lane_to_i64(lane)),
            ColumnData::F64(v) => v.push(lane_to_f64(lane)),
            ColumnData::Str(_) => unreachable!("string lanes need a dictionary"),
        }
    }

    pub fn bytes(&self) -> usize {
        let d = match &self.data {
            ColumnData::U64(v) => v.len() * 8,
            ColumnData::I64(v) => v.len() * 8,
            ColumnData::F64(v) => v.len() * 8,
            ColumnData::Str(v) => v.iter().map(|s| s.len() + 16).sum(),
        };
        d + self.nulls.as_ref().map_or(0, |n| n.bytes())
    }
}

/// Incremental column construction from `Value`s.
pub struct ColumnBuilder {
    ty: DataType,
    data: ColumnData,
    nulls: BitSet,
    len: usize,
}

impl ColumnBuilder {
    pub fn new(ty: DataType) -> Self {
        let data = ColumnData::for_physical(ty.physical());
        ColumnBuilder { ty, data, nulls: BitSet::new(), len: 0 }
    }

    pub fn with_capacity(ty: DataType, cap: usize) -> Self {
        let data = match ty.physical() {
            PhysicalType::U64 => ColumnData::U64(Vec::with_capacity(cap)),
            PhysicalType::I64 => ColumnData::I64(Vec::with_capacity(cap)),
            PhysicalType::F64 => ColumnData::F64(Vec::with_capacity(cap)),
            PhysicalType::Str => ColumnData::Str(Vec::with_capacity(cap)),
        };
        ColumnBuilder { ty, data, nulls: BitSet::new(), len: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn push_null(&mut self) {
        self.nulls.set(self.len);
        match &mut self.data {
            ColumnData::U64(v) => v.push(0),
            ColumnData::I64(v) => v.push(0),
            ColumnData::F64(v) => v.push(0.0),
            ColumnData::Str(v) => v.push("".into()),
        }
        self.len += 1;
    }

    /// Push a value, coercing it to the builder's declared type.
    pub fn push_value(&mut self, v: &Value) -> Result<()> {
        if v.is_null() {
            self.push_null();
            return Ok(());
        }
        match &mut self.data {
            ColumnData::U64(buf) => buf.push(
                v.as_u64()
                    .ok_or_else(|| Error::exec(format!("{v} is not a {}", self.ty)))?,
            ),
            ColumnData::I64(buf) => buf.push(
                v.as_i64()
                    .ok_or_else(|| Error::exec(format!("{v} is not a {}", self.ty)))?,
            ),
            ColumnData::F64(buf) => buf.push(
                v.as_f64()
                    .ok_or_else(|| Error::exec(format!("{v} is not a {}", self.ty)))?,
            ),
            ColumnData::Str(buf) => buf.push(match v {
                Value::Str(s) => s.clone(),
                other => other.render_plain().into(),
            }),
        }
        self.len += 1;
        Ok(())
    }

    pub fn push_u64(&mut self, x: u64) {
        match &mut self.data {
            ColumnData::U64(v) => v.push(x),
            _ => unreachable!("push_u64 on non-u64 builder"),
        }
        self.len += 1;
    }
    pub fn push_i64(&mut self, x: i64) {
        match &mut self.data {
            ColumnData::I64(v) => v.push(x),
            _ => unreachable!("push_i64 on non-i64 builder"),
        }
        self.len += 1;
    }
    pub fn push_f64(&mut self, x: f64) {
        match &mut self.data {
            ColumnData::F64(v) => v.push(x),
            _ => unreachable!("push_f64 on non-f64 builder"),
        }
        self.len += 1;
    }
    pub fn push_str(&mut self, s: Arc<str>) {
        match &mut self.data {
            ColumnData::Str(v) => v.push(s),
            _ => unreachable!("push_str on non-string builder"),
        }
        self.len += 1;
    }

    pub fn finish(self) -> Column {
        Column {
            ty: self.ty,
            data: self.data,
            nulls: if self.nulls.is_empty() { None } else { Some(self.nulls) },
        }
    }
}

/// A batch of equal-length columns.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub columns: Vec<Column>,
    rows: usize,
}

impl Block {
    pub fn new(columns: Vec<Column>) -> Result<Block> {
        let rows = columns.first().map_or(0, |c| c.len());
        for (i, c) in columns.iter().enumerate() {
            if c.len() != rows {
                return Err(Error::exec(format!(
                    "ragged block: column {i} has {} rows, expected {rows}",
                    c.len()
                )));
            }
        }
        Ok(Block { columns, rows })
    }

    /// A block with the given row count and no columns. Used by
    /// `SELECT count(*)` where no column data is needed at all.
    pub fn rows_only(rows: usize) -> Block {
        Block { columns: Vec::new(), rows }
    }

    /// Declare the row count after appending directly into `columns`.
    ///
    /// Streaming scans fill the column buffers in place rather than building a
    /// new block per batch, so the count has to be published separately. It is
    /// passed in rather than inferred because a zero-column block -- what
    /// `SELECT count(*)` scans -- still has a row count.
    pub fn set_rows(&mut self, n: usize) {
        debug_assert!(
            self.columns.iter().all(|c| c.len() == n),
            "set_rows({n}) disagrees with the column lengths"
        );
        self.rows = n;
    }

    /// Drop every row but keep the allocations, so a streaming scan can refill
    /// the same buffers instead of allocating a block per batch.
    pub fn clear(&mut self) {
        for c in self.columns.iter_mut() {
            c.clear();
        }
        self.rows = 0;
    }

    /// An empty block matching `schema`'s types.
    pub fn empty(schema: &Schema) -> Block {
        Block {
            columns: schema
                .fields()
                .iter()
                .map(|f| Column::new(f.ty.clone(), ColumnData::for_physical(f.ty.physical())))
                .collect(),
            rows: 0,
        }
    }

    #[inline(always)]
    pub fn rows(&self) -> usize {
        self.rows
    }
    #[inline(always)]
    pub fn width(&self) -> usize {
        self.columns.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }
    #[inline(always)]
    pub fn column(&self, i: usize) -> &Column {
        &self.columns[i]
    }

    pub fn schema(&self, names: &[String]) -> Schema {
        Schema::new_unchecked(
            self.columns
                .iter()
                .zip(names)
                .map(|(c, n)| super::schema::Field::new(n.clone(), c.ty.clone()))
                .collect(),
        )
    }

    pub fn take(&self, idx: &[u32]) -> Block {
        Block {
            columns: self.columns.iter().map(|c| c.take(idx)).collect(),
            rows: idx.len(),
        }
    }

    pub fn slice(&self, s: usize, e: usize) -> Block {
        let e = e.min(self.rows);
        let s = s.min(e);
        Block {
            columns: self.columns.iter().map(|c| c.slice(s, e)).collect(),
            rows: e - s,
        }
    }

    /// Keep rows whose mask bit is set.
    pub fn filter_mask(&self, mask: &BitSet) -> Block {
        let idx: Vec<u32> = (0..self.rows as u32)
            .filter(|&i| mask.get(i as usize))
            .collect();
        self.take(&idx)
    }

    pub fn extend(&mut self, other: &Block) -> Result<()> {
        if self.columns.is_empty() && self.rows == 0 {
            *self = other.clone();
            return Ok(());
        }
        if self.width() != other.width() {
            return Err(Error::exec(format!(
                "cannot concatenate blocks of width {} and {}",
                self.width(),
                other.width()
            )));
        }
        for (a, b) in self.columns.iter_mut().zip(&other.columns) {
            a.extend(b)?;
        }
        self.rows += other.rows;
        Ok(())
    }

    /// Append a `Vec<Column>` as new columns, e.g. joining the probe side.
    pub fn append_columns(&mut self, cols: Vec<Column>) -> Result<()> {
        for c in &cols {
            if c.len() != self.rows {
                return Err(Error::exec(format!(
                    "appended column has {} rows, block has {}",
                    c.len(),
                    self.rows
                )));
            }
        }
        self.columns.extend(cols);
        Ok(())
    }

    pub fn bytes(&self) -> usize {
        self.columns.iter().map(|c| c.bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col() -> Column {
        Column::i64s(DataType::Int64, vec![10, 20, 30, 40])
    }

    #[test]
    fn take_reorders_and_filters() {
        let c = col();
        let t = c.take(&[3, 0, 0]);
        assert_eq!(t.as_i64().unwrap(), &[40, 10, 10]);
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn take_carries_nulls() {
        let mut b = ColumnBuilder::new(DataType::Nullable(Box::new(DataType::Int64)));
        b.push_value(&Value::Int(1)).unwrap();
        b.push_null();
        b.push_value(&Value::Int(3)).unwrap();
        let c = b.finish();
        assert!(c.is_null(1));
        assert_eq!(c.null_count(), 1);

        let t = c.take(&[2, 1]);
        assert!(!t.is_null(0));
        assert!(t.is_null(1));
        assert_eq!(t.value(0), Value::Int(3));
        assert_eq!(t.value(1), Value::Null);
    }

    #[test]
    fn take_with_no_surviving_nulls_drops_the_mask() {
        let mut b = ColumnBuilder::new(DataType::Nullable(Box::new(DataType::Int64)));
        b.push_value(&Value::Int(1)).unwrap();
        b.push_null();
        let c = b.finish();
        let t = c.take(&[0]);
        assert!(!t.has_nulls(), "mask should be dropped when nothing is null");
    }

    #[test]
    fn slice_bounds_and_nulls() {
        let mut b = ColumnBuilder::new(DataType::Nullable(Box::new(DataType::Int64)));
        for i in 0..5 {
            if i == 2 {
                b.push_null();
            } else {
                b.push_value(&Value::Int(i)).unwrap();
            }
        }
        let c = b.finish();
        let s = c.slice(1, 4);
        assert_eq!(s.len(), 3);
        assert!(s.is_null(1));
        assert_eq!(s.value(0), Value::Int(1));
    }

    #[test]
    fn value_respects_logical_type() {
        let d = Column::u64s(DataType::Date, vec![0, 19_723]);
        assert_eq!(d.value(0), Value::Date(0));
        assert_eq!(d.value(1).to_string(), "2024-01-01");

        let b = Column::u64s(DataType::Bool, vec![0, 1]);
        assert_eq!(b.value(0), Value::Bool(false));
        assert_eq!(b.value(1), Value::Bool(true));

        // A decimal lane is a unit count. `Value`'s `Eq` is blind to the
        // variant, so `assert_eq!(.., Value::Int(381))` would pass here for the
        // right answer *and* the wrong one; match the variant and the scale.
        let p = Column::i64s(DataType::Decimal64(2), vec![381, -119]);
        assert!(matches!(p.value(0), Value::Decimal(381, 2)), "{:?}", p.value(0));
        assert_eq!(p.value(0).to_string(), "3.81");
        assert_eq!(p.value(1).to_string(), "-1.19");
        // Nullable wraps the scale, and `base()` has to see through it.
        let n = Column::i64s(DataType::Decimal64(4).to_nullable(), vec![15_000]);
        assert_eq!(n.value(0).to_string(), "1.5000");
    }

    #[test]
    fn block_rejects_ragged_columns() {
        let a = Column::i64s(DataType::Int64, vec![1, 2]);
        let b = Column::i64s(DataType::Int64, vec![1]);
        assert!(Block::new(vec![a, b]).is_err());
    }

    #[test]
    fn block_take_and_extend() {
        let mut blk = Block::new(vec![col(), Column::u64s(DataType::UInt64, vec![1, 2, 3, 4])])
            .unwrap();
        assert_eq!(blk.rows(), 4);
        let t = blk.take(&[1, 2]);
        assert_eq!(t.rows(), 2);
        assert_eq!(t.column(0).as_i64().unwrap(), &[20, 30]);

        blk.extend(&t).unwrap();
        assert_eq!(blk.rows(), 6);
        assert_eq!(blk.column(0).as_i64().unwrap(), &[10, 20, 30, 40, 20, 30]);
    }

    #[test]
    fn filter_mask_keeps_set_bits() {
        let blk = Block::new(vec![col()]).unwrap();
        let mut m = BitSet::new();
        m.set(0);
        m.set(3);
        let f = blk.filter_mask(&m);
        assert_eq!(f.column(0).as_i64().unwrap(), &[10, 40]);
    }

    #[test]
    fn constant_column() {
        let c = Column::constant(&DataType::Int64, &Value::Int(7), 3).unwrap();
        assert_eq!(c.as_i64().unwrap(), &[7, 7, 7]);
        let n = Column::constant(&DataType::Int64, &Value::Null, 2).unwrap();
        assert!(n.is_null(0) && n.is_null(1));
    }

    #[test]
    fn builder_coerces_and_tracks_nulls() {
        let mut b = ColumnBuilder::new(DataType::Float64);
        b.push_value(&Value::Int(3)).unwrap();
        b.push_value(&Value::Float(1.5)).unwrap();
        let c = b.finish();
        assert_eq!(c.as_f64().unwrap(), &[3.0, 1.5]);
        assert!(!c.has_nulls());

        let mut b = ColumnBuilder::new(DataType::String);
        b.push_value(&Value::Int(42)).unwrap();
        assert_eq!(b.finish().as_str().unwrap()[0].as_ref(), "42");
    }

    #[test]
    fn push_lane_decodes_storage_representation() {
        use crate::common::{f64_to_lane, i64_to_lane};
        let mut c = Column::i64s(DataType::Int64, vec![]);
        c.push_lane(i64_to_lane(-5));
        c.push_lane(i64_to_lane(9));
        assert_eq!(c.as_i64().unwrap(), &[-5, 9]);

        let mut f = Column::f64s(DataType::Float64, vec![]);
        f.push_lane(f64_to_lane(1.5));
        assert_eq!(f.as_f64().unwrap(), &[1.5]);
    }

    #[test]
    fn rows_only_block_supports_count_star() {
        let b = Block::rows_only(1000);
        assert_eq!(b.rows(), 1000);
        assert_eq!(b.width(), 0);
    }
}
