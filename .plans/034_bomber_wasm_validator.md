# Plan 034: Bomber WASM Validator — Moved to riir-ai

**Branch:** `develop/feature/034_bomber_wasm_validator`
**Status:** Moved to `riir-ai/.plans/034_bomber_wasm_validator.md`

> This plan has been moved to the private `riir-ai` repo. The validator source code is a commercial secret (Secret A2) and lives alongside `riir-validator-sdk`. The WASM loader (`BomberWasmPruner`) remains in `microgpt-rs`.
>
> **See:** `riir-ai/.plans/034_bomber_wasm_validator.md` for the refined architecture.

---

## What Stayed in microgpt-rs

- `src/pruners/bomber/wasm_pruner.rs` — `BomberWasmPruner` (wasmtime loader)
- `src/pruners/bomber/wasm_state.rs` — game state serialization for WASM ABI
- `src/pruners/bomber/players.rs` — `NNPlayer` wiring (loads `.bin` + `.wasm` at runtime)
- `examples/bomber_01_arena.rs` — tournament (loads artifacts)
- `examples/bomber_04_nn.rs` — NNPlayer demo (needs secrets to run)

## What Moved to riir-ai

- `crates/riir-validator-sdk/examples/bomber_validator.rs` — safety rules → `bomber_validator.wasm`
- `crates/riir-gpu/src/game/` — replay parsing + game policy config
- `crates/riir-gpu/examples/train_bomber.rs` — trains `game_lora.bin`
- `crates/riir-examples/examples/bomber_demo.rs` — cross-cutting demo