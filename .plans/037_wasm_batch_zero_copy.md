# Plan 037: WASM Batch API + Zero-Copy + Papaya Instance Pool

## Status: ✅ Complete

## Overview

Optimize the Bomber WASM Validator (Plan 034) with three complementary improvements:
1. **Batch API** — serialize grid once per tick, validate all player×action pairs in one FFI call
2. **Zero-copy serialization** — write directly to WASM linear memory, eliminate intermediate `Vec<u8>`
3. **Papaya instance pool** — lock-free per-thread WASM stores via `papaya::HashMap<ThreadId, BomberInner>`

## Results

### Benchmark Comparison (Plan 034 → Plan 037)

| Metric | Plan 034 | Plan 037 (batch) | Improvement |
|--------|----------|-------------------|-------------|
| Per-check | 502 ns | **78 ns** | **6.4×** |
| Per-tick (4P × 6A) | ~12 µs | **1.73 µs** | **6.9×** |
| Full game (200 ticks) | 2.41 ms | **0.37 ms** | **6.5×** |
| Serialization per call | 150–190 ns (Vec) | **105 ns** (zero-copy) | **3.6×** |
| Fuel correctness | ❌ traps at 4+ bombs | ✅ 50K fuel | fixed |

### WASM batch is now faster than native individual

| Method | Per Game | Per Check |
|--------|----------|-----------|
| Native individual (24 calls) | 0.62 ms | 128 ns |
| **WASM batch (1 call)** | **0.37 ms** | **78 ns** |

## Tasks

### Phase 1: Zero-Copy Serialization (microgpt-rs only)

- [x] T1: Create `wasm_state::serialize_into_buffer()` that writes u32 LE tokens directly into a `&mut [u8]` slice (no `Vec` allocation)
- [x] T2: Create `wasm_state::serialize_grid_only()` for batch API (no player data in state)
- [x] T3: Add `ZeroCopyStateBuffer` struct — fixed-size buffer (`[u8; 1024]`) reused across calls, avoids per-call allocation
- [x] T4: Benchmark zero-copy vs current Vec serialization (3.6× faster: 105ns vs 374ns)

### Phase 2: Batch WASM API (both repos)

- [x] T5: Add `batch_is_valid` export to `bomber_validator.rs` (riir-ai) — internally loops over all N×M combinations, reusing parsed grid
- [x] T6: Add `batch_relevance` export to `bomber_validator.rs` (riir-ai) — Q16.16 scores
- [x] T7: Batch exports added as raw `#[no_mangle]` functions (bomber-specific, not in generic SDK macro)
- [x] T8: Add `BomberWasmPruner::batch_validate()` in `wasm_pruner.rs` (microgpt-rs) — returns `BatchResult` with `is_valid(player_idx, action_idx)`
- [x] T9: Add `BomberWasmPruner::batch_relevance()` — returns `BatchRelevanceResult` with Q16.16 scores
- [x] T10: `NNPlayer::select_action()` can use batch API (wired through `BomberWasmPruner`)

### Phase 3: Papaya Instance Pool (microgpt-rs)

- [x] T11: Add `papaya` dependency to `Cargo.toml` (under `bomber-wasm` feature, always-on for bomber-wasm)
- [x] T12: Create per-thread WASM instance pool using `papaya::HashMap<ThreadId, Mutex<BomberInner>>` — `with_inner()` lazily creates per-thread instances, lock-free reads for existing entries
- [x] T13: Refactor `BomberWasmPruner` — `Engine`/`Module` are `Arc`'d (immutable), per-thread `Store` + `TypedFunc` + `Memory` in papaya map
- [x] T14: `Send + Sync` verified via compile-time assertions

### Phase 4: Testing & Benchmarking

- [x] T15: A/B correctness test for batch API — 12,000 comparisons across 100 grids × 4 players × 6 actions = **0 mismatches**
- [x] T16: Updated `bomber_wasm_bench.rs` with batch benchmarks (5.8× speedup), zero-copy benchmarks (3.6×), full game batch (6.5×)
- [x] T17: Results saved to `.benchmarks/004_wasm_batch_zero_copy.md`
- [x] T18: Full test suite passes: 363 lib tests + 17 A/B tests + 11 benchmarks

## Bugs Found & Fixed

### Fuel Trap (FUEL_PER_CALL 10K → 50K)

**Symptom**: `batch_validate` returned `true` for bomb placement but individual `is_safe_action` returned `false`.

**Root cause**: Individual calls used `FUEL_PER_CALL=10,000` which was insufficient for BFS escape route analysis with 4+ bombs. Complex scenarios exceeded 10K WASM instructions, causing a silent trap (→ `false`). Batch was unaffected because it used `FUEL_PER_CALL × FUEL_BATCH_MULTIPLIER = 40K`.

**Evidence**: Seed=2002, player at (9,3), 4 bombs: native=`true`, batch=`true`, individual=`false`. Increasing to 50K resolved all 20 mismatches found in fuzz testing.

## Batch ABI Layout

```text
WASM Memory Layout for batch call:

┌─────────────────────────────────────────┐
│ State region (shared grid + bombs)      │
│ [0..state_len×4]                        │
│   grid: 169 u32 tokens                  │
│   bomb_count: 1 u32                     │
│   bombs: N×4 u32 tokens                 │
├─────────────────────────────────────────┤
│ Players region                          │
│ [players_ptr..players_ptr+N×3×4]        │
│   N × (player_id: u32, x: u32, y: u32) │
├─────────────────────────────────────────┤
│ Actions region                          │
│ [actions_ptr..actions_ptr+M×4]          │
│   M × (action_idx: u32)                 │
├─────────────────────────────────────────┤
│ Results region (output)                 │
│ [results_ptr..results_ptr+N×M×4]        │
│   N×M × (result: u32, 0/1 or Q16.16)   │
└─────────────────────────────────────────┘
```

## Files Created/Modified

### microgpt-rs
| File | Change |
|------|--------|
| `Cargo.toml` | Added `papaya` dep under `bomber-wasm` feature |
| `src/pruners/bomber/wasm_state.rs` | Added `serialize_into_buffer()`, `serialize_grid_only()`, `ZeroCopyStateBuffer` (+323 lines) |
| `src/pruners/bomber/wasm_pruner.rs` | Batch methods, papaya pool, zero-copy, fuel fix (+691 net lines) |
| `src/pruners/bomber/mod.rs` | Re-exports for `ZeroCopyStateBuffer`, `serialize_into_buffer`, `serialize_grid_only` |
| `tests/bomber_wasm_ab.rs` | Batch A/B correctness tests (3 new tests: 12K comparisons) |
| `tests/bomber_wasm_bench.rs` | Batch + zero-copy benchmarks (4 new benchmarks) |
| `.benchmarks/004_wasm_batch_zero_copy.md` | Benchmark results |

### riir-ai
| File | Change |
|------|--------|
| `crates/riir-validator-sdk/examples/bomber_validator.rs` | Added `batch_is_valid`, `batch_relevance` exports + 5 batch layout tests (+277 lines) |