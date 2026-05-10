# Plan 034: Bomber WASM Validator — Extract Safety Rules into Sandboxable Module

**Branch:** `develop/feature/034_bomber_wasm_validator`
**Depends on:** Plan 033 (Bomberman Arena), Plan 032 (HL Infrastructure), Plan 021 (ScreeningPruner)
**Status:** In Progress

---

## Goal

Extract the bomber AI safety rules (`is_safe_action`, `in_blast_zone`, `has_escape_route`, `escape_distance`) from native Rust in `players.rs` into a standalone WASM module. This proves the **validator-as-WASM** architecture: the same game, same arena, but P3 (Validator) loads `bomber_validator.wasm` instead of running native Rust.

**Why:** The HL thesis requires that validators evolve between tournament rounds. WASM sandboxing enables hot-swapping validators without recompiling the host — an agent can write new `validator.rs`, compile to `.wasm`, and the arena reloads it on the next round. This is the foundation for autonomous validator evolution.

---

## Three-Phase Roadmap

### Phase 1: LoRA Model for Bomberman (Deferred)

Train a LoRA adapter on game traces (board state → action probabilities).

- Requires: training corpus from arena runs, state encoder (13×13 grid → fixed-dim), riir-burner pipeline
- Blocked by: no bomber-specific training corpus exists yet
- Would make P2 a real neural-net player instead of heuristic `GreedyPlayer`

### Phase 2: WASM Bomber Validator (This Plan) ✅

Extract safety rules into `bomber_validator.wasm`. P3 loads WASM at runtime.

- Ready: wasmtime + validator-sdk infrastructure exists in riir-ai
- Proves: WASM validator works for game AI, hot-swap works between rounds
- Self-contained: no LoRA needed, validator logic already exists in Rust

### Phase 3: Full HL Stack Wiring (Deferred)

Wire `BanditPruner<HotSwapPruner<BomberWasmPruner>>` for P4.

- Requires: Phase 1 + Phase 2 complete
- Would make P4 use LoRA proposals → WASM validation → Bandit adaptation
- The absorb-compress cycle would promote low-Q actions into WASM hard blocks

---

## Architecture

### Current (Plan 033)

```
ValidatorPlayer (P3)
├── known_bombs, known_powerups
├── select_action():
│   ├── is_safe_action()     ← native Rust, inline
│   ├── in_blast_zone()      ← native Rust, inline
│   ├── has_escape_route()   ← native Rust, inline
│   ├── escape_distance()    ← native Rust, inline
│   └── score_action()       ← native Rust, inline
└── No hot-swap, no sandbox
```

### Target (Plan 034)

```
ValidatorPlayer (P3)
├── known_bombs, known_powerups
├── wasm: BomberWasmPruner   ← loads bomber_validator.wasm
├── select_action():
│   ├── wasm.is_safe_action()    ← sandboxed WASM call
│   ├── wasm.relevance()         ← sandboxed WASM call
│   └── fallback: native Rust    ← if WASM fails to load
└── HotSwapPruner: reload between rounds (blake3 hash check)

bomber_validator.wasm
├── is_safe_action(action, state_buf) -> u32
├── relevance(action, state_buf) -> u32 (Q16.16)
├── name() -> "bomber_validator"
└── version() -> (major, minor, patch)
```

### WASM ABI — Bomber Game State

The existing `Validator` trait uses `(depth, token_idx, parent_tokens)` for DDTree token pruning. Bomber needs game-state-aware validation. We define a **bomber-specific ABI** that passes compact game state via linear memory:

```
┌─ Game State Buffer (passed via pointer + length) ─────────────┐
│                                                                 │
│ [0..168]     grid: 13×13 cells, 1 byte each                    │
│              0=Floor, 1=FixedWall, 2=DestructibleWall,          │
│              3=PowerUpHidden, 4+=unused                         │
│                                                                 │
│ [169]        player_x: u8                                       │
│ [170]        player_y: u8                                       │
│ [171]        player_id: u8 (for future opponent tracking)       │
│ [172]        bomb_count: u8 (N, max 16)                         │
│ [173..173+N*4]  bombs: N × (x: u8, y: u8, range: u8, fuse: u8) │
│                                                                 │
│ [end-2]      powerup_count: u8 (M, max 16)                      │
│ [end-2+1..end-2+M*2] powerups: M × (x: u8, y: u8)              │
│                                                                 │
│ Total: ~250 bytes (fits in WASM scratch buffer)                 │
└─────────────────────────────────────────────────────────────────┘

WASM Exports:
  is_safe_action(action_idx: u8, state_ptr: u32, state_len: u32) -> u32
    action_idx: 0=Up, 1=Down, 2=Left, 3=Right, 4=Bomb, 5=Wait
    returns: 1=safe, 0=unsafe

  relevance(action_idx: u8, state_ptr: u32, state_len: u32) -> u32
    returns: Q16.16 fixed-point score (0x00010000 = 1.0)

  name() -> u32     // pointer to null-terminated string
  version() -> u32  // (major << 16) | (minor << 8) | patch
```

### File Structure

```
microgpt-rs/
├── src/pruners/bomber/
│   ├── mod.rs               # Add BomberWasmPruner re-export
│   ├── players.rs           # ValidatorPlayer uses wasm field
│   ├── wasm_pruner.rs       # NEW: BomberWasmPruner (wasmtime loader)
│   └── wasm_state.rs        # NEW: serialize game state for WASM
│
├── validators/
│   └── bomber/              # NEW: compiles to bomber_validator.wasm
│       ├── Cargo.toml       # no_std, wasm32-unknown-unknown target
│       └── src/
│           ├── lib.rs        # export_validator! boilerplate
│           ├── grid.rs       # ArenaGrid logic (no_std compatible)
│           ├── blast.rs      # in_blast_zone, is_in_single_blast
│           ├── escape.rs     # has_escape_route, escape_distance BFS
│           └── safety.rs     # is_safe_action, should_place_bomb
│
├── Cargo.toml               # Add wasmtime dep, bomber-wasm feature
└── Makefile.toml             # (optional) cargo-make for WASM build
```

---

## Tasks

- [ ] **Task 1: Bomber Validator Crate** (`validators/bomber/`)
  - Create `no_std` crate targeting `wasm32-unknown-unknown`
  - Implement game state parsing from binary buffer
  - Port `in_blast_zone` / `is_in_single_blast` to `no_std` (no Vec, use fixed arrays)
  - Port `has_escape_route` / `escape_distance` BFS to `no_std` (fixed-size queue)
  - Port `is_safe_action` / `should_place_bomb`
  - Export via `#[no_mangle] extern "C"` ABI
  - Unit tests run natively (`#[cfg(test)]` with std)

- [ ] **Task 2: WASM Build Pipeline**
  - Add build script or justfile target: `cargo build -p bomber_validator --target wasm32-unknown-unknown --release`
  - Output: `target/wasm32-unknown-unknown/release/bomber_validator.wasm`
  - Optional: `wasm-opt -Oz` for size optimization
  - CI: add wasm32 target check

- [ ] **Task 3: Game State Serialization** (`src/pruners/bomber/wasm_state.rs`)
  - `serialize_game_state(grid, pos, bombs, powerups) -> Vec<u8>`
  - Follow the binary layout defined in the ABI section
  - Unit tests verifying roundtrip with mock states

- [ ] **Task 4: BomberWasmPruner Loader** (`src/pruners/bomber/wasm_pruner.rs`)
  - Load `.wasm` file with wasmtime
  - Call `is_safe_action(action, state_ptr, state_len) -> u32`
  - Call `relevance(action, state_ptr, state_len) -> u32` (Q16.16 decode)
  - Fuel limit: 1000 units per call (prevent infinite loops)
  - Memory limit: 64 pages (4MB)
  - Graceful fallback: if WASM fails, use native Rust safety rules
  - Implement `ScreeningPruner` trait for DDTree compatibility

- [ ] **Task 5: Wire into ValidatorPlayer**
  - Add `wasm: Option<BomberWasmPruner>` field to `ValidatorPlayer`
  - `ValidatorPlayer::new_with_wasm(id, wasm_path)` constructor
  - `select_action()`: call `wasm.is_safe_action()` instead of native
  - `select_action()`: call `wasm.relevance()` for scoring
  - Feature gate: `#[cfg(feature = "bomber-wasm")]` for WASM path
  - Arena example: add `--wasm` flag to load `.wasm` for P3

- [ ] **Task 6: HotSwapPruner Integration**
  - Wrap `BomberWasmPruner` in `HotSwapPruner<BomberWasmPruner>`
  - Between tournament rounds, call `hot_swap.reload()`
  - If `.wasm` file changed on disk (blake3 hash), reload without restart
  - This proves: agent can evolve validator between rounds

- [ ] **Task 7: Arena Integration & A/B Test**
  - Run 100-round tournament with WASM P3 vs native P3
  - Verify identical scores (WASM should match native bit-for-bit for same inputs)
  - Add `bomber_04_wasm_proof.rs` example: 100 rounds, compare WASM vs native
  - Benchmark: WASM call overhead vs native (target <5µs per call)

- [ ] **Task 8: Tests & Docs**
  - Unit tests: game state serialization roundtrip
  - Integration tests: WASM validator matches native for 100+ board states
  - Edge cases: empty grid, full of walls, player on bomb, chain explosions
  - Update `.docs/10_bomber_arena.md` with WASM architecture
  - Update `README.md` with WASM validator section

---

## Cargo.toml Changes

```toml
# microgpt-rs/Cargo.toml
[features]
bomber = ["bevy_ecs", "bandit"]
bomber-wasm = ["bomber", "wasmtime"]  # WASM validator support

[dependencies]
wasmtime = { version = "28", optional = true }

# validators/bomber/Cargo.toml
[package]
name = "bomber-validator"
edition = "2021"

[lib]
crate-type = ["cdylib"]  # Produces .wasm when targeting wasm32-unknown-unknown

[dependencies]
# no_std — no external dependencies
```

---

## Expected Results

### Correctness

WASM validator produces **identical** results to native Rust for the same inputs. This is guaranteed by:
- Same algorithm (ported line-by-line)
- No floating point (all integer math, BFS uses fixed arrays)
- Deterministic (no randomness, no I/O in WASM)

### Performance

| Metric | Native Rust | WASM (wasmtime) | Target |
|--------|-------------|------------------|--------|
| `is_safe_action` | ~200ns | ~2-5µs | <10µs |
| `relevance` (full score) | ~500ns | ~5-10µs | <20µs |
| Full game (200 ticks × 6 calls) | ~0.6ms | ~6-12ms | <50ms |

WASM overhead is acceptable — even at 10× slower, the arena runs in <50ms per game (vs 5.6ms native). The bottleneck is WASM instantiation and memory copy, not computation.

### Hot-Swap

Between rounds, if `bomber_validator.wasm` changes on disk:
1. `HotSwapPruner::reload()` detects blake3 hash change
2. Loads new WASM, replaces inner pruner
3. Next round uses new validator rules
4. Zero downtime — no restart, no recompile of host

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| WASM overhead too high for 200-tick games | Benchmark early (Task 7); fuel limit prevents runaway |
| `no_std` BFS with fixed arrays — stack overflow | Use heapless::Vec or fixed [u8; 256] queue, max 169 cells |
| Game state serialization bugs | Roundtrip tests, fuzz with random grids |
| wasmtime compile time adds to CI | Feature-gate behind `bomber-wasm`, optional dep |
| WASM validator diverges from native | A/B test in Task 7, share test suite |

---

## References

- Plan 033: Bomberman Arena (existing heuristic players)
- Plan 032: HL Infrastructure (HotSwapPruner, TrialLog, AbsorbCompress)
- Plan 021: ScreeningPruner (trait definition)
- `riir-ai/crates/riir-wasm/` — existing WasmPruner implementation (wasmtime-based)
- `riir-ai/crates/riir-validator-sdk/` — Validator trait and export_validator! macro
- `src/pruners/bomber/players.rs` — current native safety rules (lines 125-446)