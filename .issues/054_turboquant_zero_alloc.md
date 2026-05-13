# Issue 054: TurboQuant KV Cache Zero-Allocation Hot Path ✅ DONE

## Problem

The `TurboQuantKVCache` hot path allocates heap memory on every `store_key`/`store_value`/`dequantize_key`/`dequantize_value` call. These are called per-layer, per-position during the TurboQuant forward pass — the hottest path in compressed inference.

### Allocation Count per Token per Layer

| Function | Allocations | Vec Size |
|---|---|---|
| `store_key` | 3 | `normalized`, `rotated`, `indices` |
| `store_value` | 3 | `normalized`, `rotated`, `indices` |
| `dequantize_key` | 3 | `indices`, `rotated`, `normalized` (return) |
| `dequantize_value` | 3 | `indices`, `rotated`, `normalized` (return) |

**Total: 12 heap allocations per layer per position.**

For a 2-layer model at position 64: 2 × 2 (store K+V) + 2 × 128 (dequant K+V for 65 positions) = **260 heap allocations per decode step.**

### Root Cause

`store_key`/`store_value` use `.collect()` to create intermediate `Vec<f32>` and `Vec<u8>`:

```rust
let normalized: Vec<f32> = key.iter().map(|x| x / norm).collect();  // ALLOC
let rotated = mat_vec(&layer_state.rotation, &normalized);           // ALLOC
let indices: Vec<u8> = rotated.iter().map(|&v| cb.quantize(v)).collect(); // ALLOC
```

`dequantize_key`/`dequantize_value` similarly allocate 3 Vecs each.

Even `dequantize_key_into` internally calls `dequantize_key` (which allocates) then copies into the output buffer — not truly zero-copy.

## Impact

- Heap allocation overhead (~50-100ns per alloc) dominates at longer sequence lengths
- TurboQuant forward pass scales O(t_n × n_layers) in allocations — not just compute
- GC pressure if ever used in managed runtime interop
- Breaks the zero-alloc pattern established by `ForwardContext` (Plan 028)

## Solution

Pre-allocate scratch buffers in `TurboQuantKVCache` (or a new `TurboQuantContext`). Create `_into` variants for store/dequantize that write into pre-allocated buffers instead of allocating.

## Acceptance Criteria

- [x] Zero heap allocations in `store_key`/`store_value` hot path
- [x] Zero heap allocations in `dequantize_key`/`dequantize_value` hot path
- [x] `forward_turboquant` remains zero-alloc end-to-end
- [x] Benchmark proves improvement (before/after)
- [x] All existing tests pass (quality metrics unchanged)

**All criteria met. Issue closed.**

## Resolution — Plan 051 ✅

Implemented via `.plans/051_turboquant_zero_alloc.md`. All 311 tests pass, quality unchanged.

### Benchmark Results

**Per-call (kv_dim=4, Config::draft):**
| Operation | Alloc (ns) | Zero (ns) | Speedup |
|---|---|---|---|
| dequantize_key | 803 | 663 | **1.21×** |
| dequantize_value | 810 | 648 | **1.25×** |

**Full cycle (kv_dim=16, 16 positions):**
| Operation | Alloc (μs) | Zero (μs) | Δ |
|---|---|---|---|
| full store+dequant | 616.84 | 341.73 | **+44.6%** |

### Files Modified
- `src/turboquant/kv_cache.rs` — Scratch buffers, `_into` variants, zero-alloc store/dequant
- `tests/bench_turboquant_zero_alloc.rs` — New: 4 benchmark tests