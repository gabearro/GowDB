# .attic

The pre-restructure sources, kept for reference. Nothing here is built.

* `main.rs.v2-original` — the original single-file engine (v0.2.0, 1848 lines).
  Every algorithm in it now lives in `src/`:

  | original                        | now                                        |
  |---------------------------------|--------------------------------------------|
  | `PackedU64`, `packed_lower_bound` | `src/encoding/bitpack.rs`                |
  | `Mph` (CHD)                     | `src/index/mph.rs`                         |
  | `SegFilter`                     | `src/index/filter.rs`                      |
  | `BitSet`                        | `src/common/bitset.rs`                     |
  | `mum`/`hash_key`/`fastrange`/`fp6` | `src/common/hash.rs`                    |
  | `radix_sort_rows`               | `src/sort.rs`                              |
  | `Granule` + learned ranks       | `src/storage/granule.rs`                   |
  | `Segment`                       | `src/storage/part.rs`                      |
  | `Db` + delta memtable           | `src/storage/table.rs`, `src/storage/delta.rs` |
  | the `main()` benchmark          | `benches/engine.rs`                        |
  | its `#[cfg(test)]` modules      | split across the modules above             |

  Two things changed rather than moved: the fixed `(u64, i64, u32)` row became
  an arbitrary typed schema, and signed/float columns switched from zigzag to
  an order-preserving lane encoding (see `src/common/lane.rs` for why).

* `config.toml.original` — was at the repo root, where cargo does not read it.
  It now lives at `.cargo/config.toml`.
