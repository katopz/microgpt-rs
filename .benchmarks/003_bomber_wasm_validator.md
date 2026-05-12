# Benchmark 003: Bomber WASM Validator — Native vs WASM Performance

**Date**: 2025-05-12
**Plan**: 034 (Bomber WASM Validator)
**Test**: `cargo test --test bomber_wasm_bench --features bomber-wasm --release -- --nocapture`

## Setup

- **WASM Module**: `bomber_validator.wasm` (33.0 KB, built with `--release`)
- **Grid**: 13×13 Bomberman arena (seed=42)
- **Iterations**: 10,000 per micro-benchmark (100 warmup)
- **Game sim**: 200 ticks × 4 players × 6 actions = 4,800 checks per game
- **Hardware**: macOS (Apple Silicon, release build)

## Results

### Per-Call Overhead

| Metric | Native Rust | WASM (wasmtime) | Overhead | Target | Status |
|--------|-------------|------------------|----------|--------|--------|
| `is_safe_action` (Up, no bombs) | 2 ns | 502 ns | 251× | < 10µs | ✅ |
| `is_safe_action` (Down, no bombs) | 2 ns | 503 ns | 251× | < 10µs | ✅ |
| `is_safe_action` (Left, no bombs) | 2 ns | 505 ns | 224× | < 10µs | ✅ |
| `is_safe_action` (Right, no bombs) | 2 ns | 509 ns | 226× | < 10µs | ✅ |
| `is_safe_action` (Bomb, no bombs) | 492 ns | 543 ns | 1.1× | < 10µs | ✅ |
| `is_safe_action` (Wait, no bombs) | 2 ns | 446 ns | 255× | < 10µs | ✅ |
| `is_safe_action` (Up, 3 bombs) | 6 ns | 470 ns | 77× | < 10µs | ✅ |
| `action_relevance` (Up, 3 bombs) | — | 550 ns | — | < 20µs | ✅ |
| `action_relevance` (Bomb, 3 bombs) | — | 370 ns | — | < 20µs | ✅ |

### Full Game Simulation

| Metric | Native | WASM | Overhead |
|--------|--------|------|----------|
| Per game (200 ticks × 4 players × 6 actions) | 0.68 ms | 2.41 ms | 3.6× |
| Per check (avg across all actions) | 141 ns | 502 ns | 3.6× |

### Infrastructure

| Metric | Value |
|--------|-------|
| WASM instantiation (one-time) | 4.10 ms |
| Serialization (no bombs, 13×13) | 0.15 µs |
| Serialization (3 bombs, 13×13) | 0.19 µs |
| WASM binary size | 33.0 KB |

## Analysis

### Movement overhead (251×) is expected

Native movement checks are trivial — one bounds check + one array lookup (~2ns). WASM adds serialization (0.15µs) + memory copy + FFI call + fuel accounting. The fixed ~500ns floor is the WASM calling overhead, not algorithmic cost.

### Bomb action is nearly parity (1.1×)

Both native and WASM run the same BFS-based `has_escape_route`. The algorithmic work (169-cell BFS) dominates the fixed overhead, making WASM competitive. This is the realistic scenario — bomb placement is the complex safety check.

### Full game: only 3.6× slower

A full 200-tick game with 4 players and 6 action checks per tick takes only 2.41ms via WASM. The 50ms target is met with 20× headroom.

### Relevance scoring is fast

Q16.16 fixed-point scoring runs in 370–550ns, well under the 20µs target. The WASM module computes proximity, wall density, and center bias without heap allocations.

## Conclusion

| Target | Result | Margin |
|--------|--------|--------|
| `is_safe_action` < 10µs | 0.37–0.55µs | 18–27× headroom |
| `relevance` < 20µs | 0.37–0.55µs | 36–54× headroom |
| Full game < 50ms | 2.41ms | 20× headroom |

WASM validation overhead is acceptable for real-time Bomberman gameplay. The ~500ns fixed cost per call is negligible compared to game tick budgets (~5ms per tick at 200Hz).