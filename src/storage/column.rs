//! A single column inside a single granule, compressed.
//!
//! This is where the engine's storage cost is actually decided. Every column
//! is frame-of-reference bit-packed against its own per-granule minimum, so
//! the width collapses to whatever the *local* value range needs rather than
//! the declared type width:
//!
//! | column                          | declared | stored          |
//! |---------------------------------|----------|-----------------|
//! | monotonic id, 1024 rows/granule | 64 bits  | ~10 bits        |
//! | `status` from 8 distinct values | 32 bits  | 3 bits          |
//! | timestamps within one hour      | 32 bits  | 12 bits         |
//! | genuinely random u64            | 64 bits  | 64 bits         |
//!
//! The last row matters: FOR never *inflates*, and it never costs a
//! decompression step, so it is safe to apply unconditionally.
//!
//! Zone maps are nearly free here. `PackedU64::base()` is the exact minimum by
//! construction, so a granule only pays 8 extra bytes to also record its exact
//! maximum -- 0.06 bits/row at 1024 rows/granule.

use crate::common::{f64_to_lane, i64_to_lane, lane_to_f64, lane_to_i64, BitSet, Result};
use crate::encoding::{dict, PackedU64, StringDict};
use crate::types::{Column, ColumnData, DataType, PhysicalType, Value};
use std::sync::Arc;

pub struct PackedColumn {
    pub ty: DataType,
    lanes: PackedU64,
    /// Present only for string columns: order-preserving, so `lanes` can be
    /// compared and range-searched without decoding.
    dict: Option<StringDict>,
    nulls: Option<BitSet>,
    /// Exact maximum lane. The minimum is `lanes.base()`, exact by definition.
    max_lane: u64,
    len: usize,
}

impl PackedColumn {
    /// Compress `col`, which must hold exactly the granule's rows.
    pub fn build(col: &Column) -> Result<PackedColumn> {
        let len = col.len();
        let (lanes_raw, dict) = match &col.data {
            ColumnData::U64(v) => (v.clone(), None),
            ColumnData::I64(v) => (v.iter().map(|&x| i64_to_lane(x)).collect(), None),
            ColumnData::F64(v) => (v.iter().map(|&x| f64_to_lane(x)).collect(), None),
            ColumnData::Str(v) => {
                let strs: Vec<&str> = v.iter().map(|s| s.as_ref()).collect();
                let (d, codes) = dict::encode(&strs);
                (codes, Some(d))
            }
        };
        let max_lane = lanes_raw.iter().copied().max().unwrap_or(0);
        Ok(PackedColumn {
            ty: col.ty.clone(),
            lanes: PackedU64::pack(&lanes_raw),
            dict,
            nulls: col.nulls.clone(),
            max_lane,
            len,
        })
    }

    /// Reassemble from on-disk parts without re-deriving anything.
    pub fn from_parts(
        ty: DataType,
        lanes: PackedU64,
        dict: Option<StringDict>,
        nulls: Option<BitSet>,
        max_lane: u64,
        len: usize,
    ) -> PackedColumn {
        PackedColumn { ty, lanes, dict, nulls, max_lane, len }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline(always)]
    pub fn lane(&self, i: usize) -> u64 {
        self.lanes.get(i)
    }
    #[inline(always)]
    pub fn is_null(&self, i: usize) -> bool {
        self.nulls.as_ref().is_some_and(|n| n.get(i))
    }
    pub fn lanes(&self) -> &PackedU64 {
        &self.lanes
    }
    pub fn dict(&self) -> Option<&StringDict> {
        self.dict.as_ref()
    }
    pub fn nulls(&self) -> Option<&BitSet> {
        self.nulls.as_ref()
    }
    #[inline(always)]
    pub fn min_lane(&self) -> u64 {
        self.lanes.base()
    }
    #[inline(always)]
    pub fn max_lane(&self) -> u64 {
        self.max_lane
    }
    #[inline(always)]
    pub fn prefetch(&self, i: usize) {
        self.lanes.prefetch(i);
    }

    /// Decode one cell.
    pub fn value(&self, i: usize) -> Value {
        if self.is_null(i) {
            return Value::Null;
        }
        let lane = self.lanes.get(i);
        self.lane_to_value(lane)
    }

    fn lane_to_value(&self, lane: u64) -> Value {
        match self.ty.base() {
            DataType::Bool => Value::Bool(lane != 0),
            DataType::Date => Value::Date(lane as u32),
            DataType::DateTime => match self.ty.physical() {
                PhysicalType::I64 => Value::DateTime(lane_to_i64(lane)),
                _ => Value::DateTime(lane as i64),
            },
            // Zone maps are compared against filter literals under `Value`'s
            // ordering, so a decimal min/max must carry its scale or pruning
            // compares a unit count against a number and skips live granules.
            DataType::Decimal64(s) => Value::Decimal(lane_to_i64(lane), *s),
            _ => match self.ty.physical() {
                PhysicalType::U64 => Value::UInt(lane),
                PhysicalType::I64 => Value::Int(lane_to_i64(lane)),
                PhysicalType::F64 => Value::Float(lane_to_f64(lane)),
                PhysicalType::Str => Value::Str(
                    self.dict
                        .as_ref()
                        .map(|d| d.get(lane))
                        .unwrap_or("")
                        .into(),
                ),
            },
        }
    }

    /// The smallest value in this granule, for zone-map pruning. `Value::Null`
    /// when every row is NULL.
    pub fn min_value(&self) -> Value {
        self.min_lane_live().map_or(Value::Null, |l| self.lane_to_value(l))
    }

    /// The largest value in this granule.
    pub fn max_value(&self) -> Value {
        self.max_lane_live().map_or(Value::Null, |l| self.lane_to_value(l))
    }

    /// Zone-map bounds must ignore NULL rows, whose lanes are stored as 0 and
    /// would otherwise drag the minimum down to 0 and defeat pruning. With no
    /// nulls this is the free FOR metadata; with nulls we pay one pass.
    ///
    /// `None` is "no live row", which covers the empty granule and the all-NULL
    /// one at once, so the separate `all_null()` pre-check this replaced is
    /// gone. **That was a null result, and is recorded so nobody retries it:**
    /// dropping the extra pass looks like halving the work (a nullable column
    /// used to pay a `count_ones_upto` *and* a scan, for each of two bounds)
    /// and measures as nothing. A/B interleaved, best-of-200 x 4 runs,
    /// `--release`, `min_value()+max_value()` over a 1024-row granule:
    /// 3.16us -> 3.16us at 10% nulls, 1.72us -> 1.72us at 90%, 9ns -> 9ns with
    /// no nulls. All 1.00x. `count_ones_upto` over 1024 bits is 16 popcounts
    /// against a 1024-iteration `BitSet::get` + `PackedU64::get` walk; it was
    /// never where the time went. Kept only because it is ten lines shorter.
    ///
    /// The time is in that walk, and the fix is not here: `min_value` and
    /// `max_value` scan it separately, so pruning a nullable column costs two
    /// walks where one fused pass would do. Halving it means a call site that
    /// asks for both bounds at once -- see `prunes` in exec/operators/scan.rs
    /// and exec/operators/exchange.rs, which is where the pair is consumed.
    fn min_lane_live(&self) -> Option<u64> {
        match &self.nulls {
            None => (self.len != 0).then(|| self.lanes.base()),
            Some(n) => (0..self.len).filter(|&i| !n.get(i)).map(|i| self.lanes.get(i)).min(),
        }
    }

    fn max_lane_live(&self) -> Option<u64> {
        match &self.nulls {
            None => (self.len != 0).then_some(self.max_lane),
            Some(n) => (0..self.len).filter(|&i| !n.get(i)).map(|i| self.lanes.get(i)).max(),
        }
    }

    /// Append `[s, e)` to the end of `out`, allocating nothing per granule.
    ///
    /// This is the hot path of every scan, and the three costs it exists to
    /// avoid are all invisible in a naive implementation:
    ///
    ///   1. **the intermediate `Vec`** -- decoding into a fresh buffer and then
    ///      copying it into the output block doubles the memory traffic and
    ///      allocates once per granule per column;
    ///   2. **the zero-fill** -- `vec![0u64; n]` memsets a buffer that
    ///      `unpack_range` immediately overwrites in full;
    ///   3. **the second pass for signed and float columns** -- lane decoding
    ///      is a bijection on the same 8 bytes, so it can be applied *in place*
    ///      over the bytes just written rather than into another buffer.
    ///
    /// Together those are the difference between a scan running at memory
    /// bandwidth and one running at allocator speed.
    pub fn decode_append(&self, s: usize, e: usize, out: &mut Column, scratch: &mut Vec<u64>) {
        debug_assert!(s <= e && e <= self.len);
        debug_assert_eq!(out.ty.physical(), self.ty.physical());
        let n = e - s;
        let base_rows = out.len();

        match &mut out.data {
            ColumnData::U64(v) => {
                v.reserve(n);
                // SAFETY: `reserve` guarantees `n` uninitialized slots past
                // `len`, and `unpack_range` writes every one of `out[..n]` on
                // all three packed layouts (lane-aligned, constant, straddled),
                // so no element is left uninitialized before `set_len`.
                unsafe {
                    let dst = std::slice::from_raw_parts_mut(v.as_mut_ptr().add(v.len()), n);
                    self.lanes.unpack_range(s, e, dst);
                    v.set_len(base_rows + n);
                }
            }
            ColumnData::I64(v) => {
                v.reserve(n);
                // SAFETY: as above. `i64` and `u64` share size and alignment,
                // so the freshly written lanes can be reinterpreted in place;
                // `lane_to_i64` is a sign-bit flip, which is exactly what the
                // xor below performs on that view.
                unsafe {
                    let dst =
                        std::slice::from_raw_parts_mut(v.as_mut_ptr().add(v.len()) as *mut u64, n);
                    self.lanes.unpack_range(s, e, dst);
                    for x in dst.iter_mut() {
                        *x ^= 1 << 63;
                    }
                    v.set_len(base_rows + n);
                }
            }
            ColumnData::F64(v) => {
                v.reserve(n);
                // SAFETY: as above; `f64` and `u64` share size and alignment.
                unsafe {
                    let dst =
                        std::slice::from_raw_parts_mut(v.as_mut_ptr().add(v.len()) as *mut u64, n);
                    self.lanes.unpack_range(s, e, dst);
                    for x in dst.iter_mut() {
                        *x = lane_to_f64(*x).to_bits();
                    }
                    v.set_len(base_rows + n);
                }
            }
            ColumnData::Str(v) => {
                scratch.clear();
                scratch.resize(n, 0);
                self.lanes.unpack_range(s, e, scratch);
                let empty = StringDict::empty();
                let d = self.dict.as_ref().unwrap_or(&empty);
                // One `Arc` per *distinct* value, then refcount bumps per row.
                let table: Vec<Arc<str>> =
                    (0..d.len()).map(|i| Arc::from(d.get(i as u64))).collect();
                let blank: Arc<str> = Arc::from("");
                v.reserve(n);
                v.extend(scratch.iter().map(|&c| {
                    table.get(c as usize).cloned().unwrap_or_else(|| blank.clone())
                }));
            }
        }

        if let Some(nm) = &self.nulls {
            let out_nulls = out.nulls.get_or_insert_with(BitSet::new);
            for i in s..e {
                if nm.get(i) {
                    out_nulls.set(base_rows + i - s);
                }
            }
        }
    }

    /// Append only the rows named by `rows` (indices into this granule).
    pub fn gather_append(&self, rows: &[u32], out: &mut Column) {
        let base_rows = out.len();
        match &mut out.data {
            ColumnData::U64(v) => {
                v.extend(rows.iter().map(|&i| self.lanes.get(i as usize)))
            }
            ColumnData::I64(v) => {
                v.extend(rows.iter().map(|&i| lane_to_i64(self.lanes.get(i as usize))))
            }
            ColumnData::F64(v) => {
                v.extend(rows.iter().map(|&i| lane_to_f64(self.lanes.get(i as usize))))
            }
            ColumnData::Str(v) => {
                let empty = StringDict::empty();
                let d = self.dict.as_ref().unwrap_or(&empty);
                let table: Vec<Arc<str>> =
                    (0..d.len()).map(|i| Arc::from(d.get(i as u64))).collect();
                let blank: Arc<str> = Arc::from("");
                v.extend(rows.iter().map(|&i| {
                    let c = self.lanes.get(i as usize) as usize;
                    table.get(c).cloned().unwrap_or_else(|| blank.clone())
                }));
            }
        }
        if let Some(nm) = &self.nulls {
            let out_nulls = out.nulls.get_or_insert_with(BitSet::new);
            for (o, &i) in rows.iter().enumerate() {
                if nm.get(i as usize) {
                    out_nulls.set(base_rows + o);
                }
            }
        }
    }

    /// Bulk-decode `[s, e)` into a fresh `Column`.
    ///
    /// String decoding builds the `Arc<str>` table once per call (one
    /// allocation per *distinct* value) and then clones refcounts per row,
    /// rather than allocating per row.
    pub fn decode(&self, s: usize, e: usize) -> Column {
        debug_assert!(s <= e && e <= self.len);
        let n = e - s;
        let data = match self.ty.physical() {
            PhysicalType::U64 => {
                let mut v = vec![0u64; n];
                self.lanes.unpack_range(s, e, &mut v);
                ColumnData::U64(v)
            }
            PhysicalType::I64 => {
                let mut v = vec![0u64; n];
                self.lanes.unpack_range(s, e, &mut v);
                ColumnData::I64(v.into_iter().map(lane_to_i64).collect())
            }
            PhysicalType::F64 => {
                let mut v = vec![0u64; n];
                self.lanes.unpack_range(s, e, &mut v);
                ColumnData::F64(v.into_iter().map(lane_to_f64).collect())
            }
            PhysicalType::Str => {
                let mut codes = vec![0u64; n];
                self.lanes.unpack_range(s, e, &mut codes);
                let empty = StringDict::empty();
                let d = self.dict.as_ref().unwrap_or(&empty);
                let table: Vec<Arc<str>> = (0..d.len()).map(|i| Arc::from(d.get(i as u64))).collect();
                let blank: Arc<str> = Arc::from("");
                ColumnData::Str(
                    codes
                        .into_iter()
                        .map(|c| {
                            table.get(c as usize).cloned().unwrap_or_else(|| blank.clone())
                        })
                        .collect(),
                )
            }
        };
        let nulls = self.nulls.as_ref().map(|nm| {
            let mut out = BitSet::new();
            for i in s..e {
                if nm.get(i) {
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

    /// Decode only the rows named by `rows` (absolute indices into the
    /// granule). Used after a predicate has already selected rows, so we never
    /// materialize columns we are about to throw away.
    pub fn gather(&self, rows: &[u32]) -> Column {
        let data = match self.ty.physical() {
            PhysicalType::U64 => {
                ColumnData::U64(rows.iter().map(|&i| self.lanes.get(i as usize)).collect())
            }
            PhysicalType::I64 => ColumnData::I64(
                rows.iter().map(|&i| lane_to_i64(self.lanes.get(i as usize))).collect(),
            ),
            PhysicalType::F64 => ColumnData::F64(
                rows.iter()
                    .map(|&i| lane_to_f64(self.lanes.get(i as usize)))
                    .collect(),
            ),
            PhysicalType::Str => {
                let empty = StringDict::empty();
                let d = self.dict.as_ref().unwrap_or(&empty);
                let table: Vec<Arc<str>> = (0..d.len()).map(|i| Arc::from(d.get(i as u64))).collect();
                let blank: Arc<str> = Arc::from("");
                ColumnData::Str(
                    rows.iter()
                        .map(|&i| {
                            let c = self.lanes.get(i as usize) as usize;
                            table.get(c).cloned().unwrap_or_else(|| blank.clone())
                        })
                        .collect(),
                )
            }
        };
        let nulls = self.nulls.as_ref().map(|nm| {
            let mut out = BitSet::new();
            for (o, &i) in rows.iter().enumerate() {
                if nm.get(i as usize) {
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

    /// Translate a literal into the lane space of *this* granule, so a
    /// predicate can run entirely on packed integers.
    ///
    /// For strings this consults the per-granule dictionary, which is why the
    /// result is only meaningful for this granule. `Ok(None)` means "no lane
    /// in this granule can equal that literal" -- itself a pruning signal.
    pub fn lane_for(&self, v: &Value) -> Result<Option<u64>> {
        if v.is_null() {
            return Ok(None);
        }
        Ok(match self.ty.physical() {
            PhysicalType::Str => {
                let s = match v {
                    Value::Str(s) => s.to_string(),
                    other => other.render_plain(),
                };
                match self.dict.as_ref().and_then(|d| d.lookup(&s)) {
                    Some(c) => Some(c),
                    None => None,
                }
            }
            _ => Some(v.to_lane(&self.ty)?),
        })
    }

    pub fn data_bytes(&self) -> usize {
        self.lanes.bytes()
            + self.dict.as_ref().map_or(0, |d| d.bytes())
            + self.nulls.as_ref().map_or(0, |n| n.bytes())
    }
}

impl std::fmt::Debug for PackedColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PackedColumn({}, {} rows, {} bits/row{})",
            self.ty,
            self.len,
            if self.len == 0 { 0 } else { self.data_bytes() * 8 / self.len },
            if self.dict.is_some() { ", dict" } else { "" }
        )
    }
}

/// Split a `Block` covering `[s, e)` into packed columns.
pub fn build_granule_columns(block: &Block, s: usize, e: usize) -> Result<Vec<PackedColumn>> {
    block
        .columns
        .iter()
        .map(|c| PackedColumn::build(&c.slice(s, e)))
        .collect()
}

use crate::types::Block;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ColumnBuilder;

    fn packed(col: Column) -> PackedColumn {
        PackedColumn::build(&col).unwrap()
    }

    #[test]
    fn roundtrips_every_physical_kind() {
        let u = packed(Column::u64s(DataType::UInt64, vec![5, 9, 5, 100]));
        assert_eq!(u.decode(0, 4).as_u64().unwrap(), &[5, 9, 5, 100]);

        let i = packed(Column::i64s(DataType::Int64, vec![-5, 0, 7]));
        assert_eq!(i.decode(0, 3).as_i64().unwrap(), &[-5, 0, 7]);

        let f = packed(Column::f64s(DataType::Float64, vec![1.5, -0.25, 1e300]));
        assert_eq!(f.decode(0, 3).as_f64().unwrap(), &[1.5, -0.25, 1e300]);

        let s = packed(Column::strs(
            DataType::String,
            vec!["b".into(), "a".into(), "b".into()],
        ));
        let d = s.decode(0, 3);
        let got: Vec<&str> = d.as_str().unwrap().iter().map(|x| x.as_ref()).collect();
        assert_eq!(got, vec!["b", "a", "b"]);
    }

    #[test]
    fn low_cardinality_strings_pack_to_few_bits() {
        // 1024 rows drawn from 4 distinct strings must cost ~2 bits/row of
        // codes plus the dictionary, not 1024 string headers.
        let vals: Vec<Arc<str>> = (0..1024)
            .map(|i| Arc::from(["alpha", "beta", "gamma", "delta"][i % 4]))
            .collect();
        let p = packed(Column::strs(DataType::String, vals));
        // 1024 rows * 2 bits = 256 bytes of codes, + tiny dict + 32B header
        assert!(p.data_bytes() < 400, "string column too large: {}", p.data_bytes());
    }

    #[test]
    fn clustered_integers_pack_narrow() {
        let vals: Vec<u64> = (1_000_000_000..1_000_000_000 + 1024).collect();
        let p = packed(Column::u64s(DataType::UInt64, vals));
        // range is 1023 -> 10 bits/row -> ~1280 bytes, vs 8192 raw
        assert!(p.data_bytes() < 1600, "got {}", p.data_bytes());
    }

    #[test]
    fn zone_map_bounds_are_exact() {
        let p = packed(Column::i64s(DataType::Int64, vec![7, -3, 22, 0]));
        assert_eq!(p.min_value(), Value::Int(-3));
        assert_eq!(p.max_value(), Value::Int(22));

        let s = packed(Column::strs(
            DataType::String,
            vec!["pear".into(), "apple".into(), "zed".into()],
        ));
        assert_eq!(s.min_value(), Value::str("apple"));
        assert_eq!(s.max_value(), Value::str("zed"));
    }

    /// A decimal lane is a unit count, so a bound handed back as `Int(units)`
    /// is compared against the filter's *number*: `min < 0.05` reads as
    /// `1 < 0.05` and prunes a granule holding four matching rows. Silently
    /// missing rows, which is worse than a visibly wrong one -- and unlike
    /// every other decimal defect, invisible in the output.
    #[test]
    fn decimal_zone_map_bounds_carry_their_scale() {
        use crate::planner::logical::{CmpOp, ZoneFilter};
        // A `price Decimal64(2)` granule spanning $0.01 .. $10.24.
        let p = packed(Column::i64s(DataType::Decimal64(2), (1..=1024).collect()));
        // `Value`'s `Eq` is blind to the variant (`Int(1) == Decimal(1, 2)` is
        // *false* only because 1 != 0.01; `Decimal(1, 2) == Decimal(10, 3)` is
        // true), so the scale itself has to be matched, not compared.
        assert!(matches!(p.min_value(), Value::Decimal(1, 2)), "{:?}", p.min_value());
        assert!(matches!(p.max_value(), Value::Decimal(1024, 2)), "{:?}", p.max_value());

        let (min, max) = (p.min_value(), p.max_value());
        let zf = |op| ZoneFilter { col: 0, op, value: Value::Decimal(5, 2) };
        // $0.05 sits inside [$0.01, $10.24]: every one of these must read it.
        for op in [CmpOp::Lt, CmpOp::LtEq, CmpOp::Gt, CmpOp::GtEq, CmpOp::Eq] {
            assert!(zf(op).may_match(&min, &max), "{op:?} pruned a live granule");
        }
        // ... and pruning still works, or the fix would have bought nothing.
        let far = ZoneFilter { col: 0, op: CmpOp::Gt, value: Value::Decimal(5000, 2) };
        assert!(!far.may_match(&min, &max), "$50.00 is above every row here");
    }

    /// The same defect end to end, which is the only place it is visible.
    /// `price` descends with `id`, so the rows under $0.05 sit in the *last*
    /// granule and every earlier one is legitimately pruned. Measured through
    /// `Session` with the scale dropped: 0 rows out of 4.
    #[test]
    fn a_decimal_filter_reaches_rows_in_a_later_granule() {
        let n = 3 * crate::common::GRANULE_SIZE as i64;
        let mut s = crate::Session::in_memory();
        s.execute("CREATE TABLE z (id UInt64, price Decimal64(2)) ENGINE = MergeTree ORDER BY id")
            .unwrap();
        let t = s.catalog.table_by_path_mut("default.z").unwrap();
        t.insert(
            Block::new(vec![
                Column::u64s(DataType::UInt64, (0..n as u64).collect()),
                Column::i64s(DataType::Decimal64(2), (1..=n).rev().collect()),
            ])
            .unwrap(),
        )
        .unwrap();
        t.flush().unwrap();
        // Compared against a `Decimal64(2)` literal so that only the zone map is
        // under test: the row-level comparison then runs lane against lane.
        let r = s
            .query("SELECT id FROM z WHERE price < CAST('0.05' AS Decimal64(2)) ORDER BY id")
            .unwrap();
        assert_eq!(r.rows(), 4, "a granule holding live rows was pruned");
        let ids: Vec<Value> = (0..4).map(|i| r.blocks[0].column(0).value(i)).collect();
        assert_eq!(ids, [n - 4, n - 3, n - 2, n - 1].map(|i| Value::UInt(i as u64)));
    }

    #[test]
    fn zone_map_ignores_null_rows() {
        let mut b = ColumnBuilder::new(DataType::Nullable(Box::new(DataType::Int64)));
        b.push_null();
        b.push_value(&Value::Int(50)).unwrap();
        b.push_value(&Value::Int(70)).unwrap();
        let p = packed(b.finish());
        // The NULL row stores lane 0; a naive min would report 0.
        assert_eq!(p.min_value(), Value::Int(50));
        assert_eq!(p.max_value(), Value::Int(70));
    }

    #[test]
    fn all_null_column_has_null_bounds() {
        let mut b = ColumnBuilder::new(DataType::Nullable(Box::new(DataType::Int64)));
        b.push_null();
        b.push_null();
        let p = packed(b.finish());
        assert_eq!(p.min_value(), Value::Null);
        assert_eq!(p.max_value(), Value::Null);
    }

    #[test]
    fn decode_preserves_nulls() {
        let mut b = ColumnBuilder::new(DataType::Nullable(Box::new(DataType::Int64)));
        b.push_value(&Value::Int(1)).unwrap();
        b.push_null();
        b.push_value(&Value::Int(3)).unwrap();
        let p = packed(b.finish());
        let d = p.decode(0, 3);
        assert!(!d.is_null(0));
        assert!(d.is_null(1));
        assert_eq!(p.value(1), Value::Null);
        assert_eq!(p.value(2), Value::Int(3));
    }

    #[test]
    fn decode_subrange_and_gather_agree() {
        let vals: Vec<u64> = (0..100).map(|i| i * 3).collect();
        let p = packed(Column::u64s(DataType::UInt64, vals.clone()));
        assert_eq!(p.decode(10, 20).as_u64().unwrap(), &vals[10..20]);
        let g = p.gather(&[99, 0, 50]);
        assert_eq!(g.as_u64().unwrap(), &[297, 0, 150]);
    }

    #[test]
    fn lane_for_resolves_through_the_dictionary() {
        let s = packed(Column::strs(
            DataType::String,
            vec!["b".into(), "a".into()],
        ));
        assert_eq!(s.lane_for(&Value::str("a")).unwrap(), Some(0));
        assert_eq!(s.lane_for(&Value::str("b")).unwrap(), Some(1));
        // absent from this granule's dictionary => provably no match here
        assert_eq!(s.lane_for(&Value::str("zzz")).unwrap(), None);

        let i = packed(Column::i64s(DataType::Int64, vec![-5, 5]));
        assert_eq!(i.lane_for(&Value::Int(-5)).unwrap(), Some(i64_to_lane(-5)));
    }

    #[test]
    fn bool_and_date_decode_to_logical_values() {
        let b = packed(Column::u64s(DataType::Bool, vec![0, 1]));
        assert_eq!(b.value(0), Value::Bool(false));
        assert_eq!(b.value(1), Value::Bool(true));

        let d = packed(Column::u64s(DataType::Date, vec![19_723]));
        assert_eq!(d.value(0), Value::Date(19_723));
        assert_eq!(d.value(0).to_string(), "2024-01-01");
    }

    #[test]
    fn empty_column_is_safe() {
        let p = packed(Column::u64s(DataType::UInt64, vec![]));
        assert!(p.is_empty());
        assert_eq!(p.min_value(), Value::Null);
        assert_eq!(p.decode(0, 0).len(), 0);
    }
}
