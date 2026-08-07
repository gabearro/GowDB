# tests/corpus

Accumulated fuzzing corpus for `tests/robustness.rs`. Committed on purpose:
the fuzzer replays every file here before it generates anything new, so a
crash found on a lucky seed at 3am stays found, and coverage grows across runs
instead of restarting cold on each one.

| directory | replayed through | contents |
|-----------|------------------|----------|
| `sql/`    | `granular::sql::parse` | UTF-8 SQL text |
| `part/`   | `persist::part_from_bytes` | damaged `.gpart` images |
| `doc/`    | `persist::reader::{catalog_from_bytes, table_parts_from_bytes}` | damaged `CATALOG` / `TABLE` documents |
| `block/`  | `persist::reader::block_from_bytes` | damaged WAL record bodies |

## What a file name means

A file is named for the **behaviour** its bytes produced, not for the bytes:
`<16 hex digits>` is a hash of the error code plus the error message with every
run of digits collapsed to `#`. With no sanitizer coverage available (zero
dependencies, stable toolchain), the diagnostic is the closest observable
proxy for "which rejection site did this input reach", and every `Err` in
`persist::reader` carries a distinct string.

That makes the file name the novelty check: a rerun that rediscovers the same
rejection site with different bytes writes nothing, so the directory converges
instead of churning, and a diff here means the *set of reachable outcomes*
changed — which is exactly the diff worth reviewing.

## Bounds

Enforced by `Corpus::offer` in `tests/robustness.rs`: at most 64 files per
directory, at most 8 KiB per file. Large multi-granule part images are
deliberately over the line — they are regenerated from the fixture on every run
anyway, and persisting them would put a megabyte in the repository to re-test
what the fixture already re-tests.

## Editing by hand

Adding a file is fine and is the right way to pin a specific input: any file
that fits the bounds is replayed. The name only has to be unique; it does not
have to match the hash of anything. Deleting a file loses coverage silently,
so prefer leaving it.

Set `GRANULAR_FUZZ_NO_WRITE=1` to replay without writing (read-only checkouts,
CI jobs that should not produce a dirty tree).
