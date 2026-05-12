# Benchmark 004: WASM Batch API + Zero-Copy + Papaya Instance Pool

**Date**: 2025-05-13
**Plan**: 037 (WASM Batch API + Zero-Copy + Papaya Pool)
**Test**: `cargo test --test bomber_wasm_bench --features bomber-wasm --release -- --nocapture`

## Setup

- **WASM Module**: `bomber_validator.wasm` (40.8 KB, built with `--release`, now includes batch exports)
- **Grid**: 13×13 Bomberman arena (seed=42)
- **Iterations**: 10,000 per micro-benchmark (100 warmup)
- **Game sim**: 200 ticks × 4 players × 6 actions = 4,800 checks per game
- **Hardware**: macOS (Apple Silicon, release build)

## Changes from Plan 034

| Change | Description |
|--------|-------------|
| **Batch API** | New `batch_is_valid` / `batch_relevance` WASM exports: serialize grid once, validate all N×M pairs in one FFI call |
| **Zero-copy** | `ZeroCopyStateBuffer` writes u32 LE tokens directly into stack buffer (no `Vec` allocation) |
| **Papaya pool** | `papaya::HashMap<ThreadId, Mutex<BomberInner>>` for lock-free per-thread WASM instances |
| **Fuel fix** | `FUEL_PER_CALL` increased from 10K → 50K (complex bomb BFS with 4+ bombs could trap at 10K) |

## Results

### Per-Tick: Batch vs Individual (4 players × 6 actions)

| Method | Per Tick | Per Check | vs Plan 034 |
|--------|----------|-----------|-------------|
| Individual (24 × `is_safe_action`) | 9.98 µs | 416 ns | 2.0× faster (zero-copy) |
| **Batch (1 × `batch_validate`)** | **1.73 µs** | **78 ns** | **6.5× faster** |
| **Speedup** | **5.8×** | **5.8×** | |

### Zero-Copy vs Vec Serialization

| Method | Per Call | Speedup |
|--------|----------|---------|
| Vec-based `serialize_game_state` | 374 ns | baseline |
| **Zero-copy `ZeroCopyStateBuffer::serialize`** | **105 ns** | **3.6× faster** |

### Individual WASM Call Overhead (after zero-copy + fuel fix)

| Action | Native | WASM | Overhead | Plan 034 |
|--------|--------|------|----------|----------|
| `is_safe_action` (Up) | 2 ns | 318 ns | 162× | 502 ns |
| `is_safe_action` (Down) | 2 ns | 317 ns | 162× | 503 ns |
| `is_safe_action` (Left) | 2 ns | 322 ns | 141× | 505 ns |
| `is_safe_action` (Right) | 2 ns | 322 ns | 136× | 509 ns |
| `is_safe_action` (Bomb) | 431 ns | 420 ns | 1.0× | 543 ns |
| `is_safe_action` (Wait) | 2 ns | 332 ns | 199× | 446 ns |

### Batch Relevance Scoring

| Method | Per Tick | Speedup |
|--------|----------|---------|
| Individual (24 × `action_relevance`) | 9.99 µs | baseline |
| **Batch (1 × `batch_relevance`)** | **1.99 µs** | **5.0× faster** |

### Full Game Simulation (200 ticks × 4 players × 6 actions)

| Method | Per Game | Per Check | vs Plan 034 |
|--------|----------|-----------|-------------|
| Native | 0.62 ms | 128 ns | — |
| WASM individual | 1.85 ms | 385 ns | 1.3× faster |
| **WASM batch** | **0.37 ms** | **78 ns** | **6.5× faster** |
| WASM overhead (batch) | 0.6× slower than native | — | — |

### Infrastructure

| Metric | Plan 034 | Plan 037 |
|--------|----------|----------|
| WASM instantiation (one-time) | 4.10 ms | 4.65 ms |
| WASM binary size | 33.0 KB | 40.8 KB (+24% batch exports) |
| Serialization (no bombs) | 150 ns | 100 ns (zero-copy) |
| Serialization (3 bombs) | 190 ns | 105 ns (zero-copy) |

## Bugs Found & Fixed

### Fuel Trap (FUEL_PER_CALL 10K → 50K)

**Symptom**: `batch_validate` returned `true` for bomb placement but individual `is_safe_action` returned `false` — opposite of the expected "WASM stricter" pattern.

**Root cause**: Individual calls used `FUEL_PER_CALL=10,000` which was insufficient for BFS escape route analysis with 4+ bombs. The BFS checks blast zones against all bombs (16 existing + 1 new) × 4 directions × range for each of 169 grid cells. Complex scenarios exceeded 10K WASM instructions, causing a silent trap (→ `false`).

**Evidence**: Seed=2002, player at (9,3), 4 bombs: native=`true`, batch=`true`, individual=`false`. Increasing to 50K resolved all 20 mismatches found in fuzz testing.

**Batch was unaffected** because it used `FUEL_PER_CALL × FUEL_BATCH_MULTIPLIER = 40K`.

## Analysis

### Batch API eliminates the serialization + FFI floor

The ~500ns per-call floor in Plan 034 came from:
- Serialization: ~150ns → eliminated (grid serialized once, not 24 times)
- FFI overhead: ~250ns → eliminated (1 FFI call, not 24)
- WASM compute: ~100ns × 24 → ~500ns (batch loops are tight)

The batch API turns 24 individual calls into 1 call with ~500ns total compute.

### Zero-copy reduces per-call cost even for individual calls

Individual `is_safe_action` dropped from ~500ns to ~320ns because:
- `ZeroCopyStateBuffer` avoids `Vec::with_capacity` allocation
- Stack buffer is reused across calls (no allocator overhead)
- 105ns serialization vs 150-190ns Vec-based

### Papaya pool eliminates Mutex contention (future-proof)

In single-threaded bomber games, the Mutex is uncontended (~20ns overhead). But for future multi-threaded tournament servers:
- Papaya `HashMap` provides lock-free reads for existing entries
- Each thread gets its own `BomberInner` on first access
- No global Mutex — each thread's `Mutex<BomberInner>` is never contended

## Conclusion

| Target | Plan 034 | Plan 037 (batch) | Improvement |
|--------|----------|-------------------|-------------|
| Per-check < 10µs | 502 ns ✅ | **78 ns** ✅ | **6.4×** |
| Per-tick < 50µs | 12 µs | **1.73 µs** | **6.9×** |
| Full game < 50ms | 2.41 ms | **0.37 ms** | **6.5×** |
| Fuel correctness | ❌ traps at 4+ bombs | ✅ 50K fuel | fixed |

### Cost of WASM safety (batch mode)

| Aspect | Cost |
|--------|------|
| Per-tick overhead vs native | +0.37ms - 0.62ms = **-0.25ms** (batch is FASTER than native!) |
| Actually | Batch WASM (0.37ms) < native individual (0.62ms) because native also does 24 separate calls |
| Per-check overhead | 78ns (WASM batch) vs 128ns (native individual) — **WASM batch is 39% faster** |

WASM batch mode is now faster than native individual mode because the batch API amortizes serialization and function call overhead across all 24 player×action pairs simultaneously.