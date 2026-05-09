# Handover 012: WASM Validator Pipeline

## What Happen

Implemented **all phases** of Plan 015 — WASM Validator Pipeline. The system allows Curators to write domain-specific validators in Rust, compile them to `.wasm`, validate them locally with a CLI tool, and have the microgpt-rs DDTree load and execute them as `ConstraintPruner` instances via Wasmtime. Full documentation includes README, `.docs/09_wasm_validator.md` ABI spec, and updated `.research/05_Artifact_Definition.md`.

**Working end-to-end flow:**
1. Platform generates `Validator` trait in riir-validator-sdk
2. `cargo build --example bracket_validator --target wasm32-unknown-unknown --release`
3. microgpt-rs `WasmPruner::load_from_file("bracket_validator.wasm")` loads it
4. `build_dd_tree_pruned(marginals, config, &wasm_pruner, false)` prunes invalid branches
5. DDTree skips branches that fail the WASM validator's rules

## Where is the Plan/Code/Test

**Plan:** `microgpt-rs/.plans/015_wasm_validator_pipeline.md`

**Code — microgpt-rs (WasmPruner Runtime):**
| File | Purpose |
|------|---------|
| `src/wasm/mod.rs` | Re-exports behind `#[cfg(feature = "wasm")]` |
| `src/wasm/abi.rs` | WASM ABI constants, memory layout, read/write helpers |
| `src/wasm/state.rs` | `ValidatorState` for wasmtime Store |
| `src/wasm/wasm_pruner.rs` | `WasmPruner` implementing `ConstraintPruner` via Wasmtime |
| `src/lib.rs` | `pub mod wasm` behind feature flag |
| `Cargo.toml` | `wasm = ["wasmtime", "wat"]` feature |

**Code — riir-validator-sdk (Curator SDK):**
| File | Purpose |
|------|---------|
| `src/lib.rs` | Re-exports, documentation |
| `src/validator.rs` | `Validator` trait (name, version, is_valid, validate_string) |
| `src/exports.rs` | `export_validator!` macro generating `#[unsafe(no_mangle)] extern "C"` ABI |
| `src/memory.rs` | WASM linear memory helpers (read_parent_tokens, read_string, write_name) |
| `examples/bracket_validator.rs` | Bracket balancing validator (14 tests) |
| `examples/keyword_validator.rs` | Rust keyword placement validator (21 tests) |

**Code — riir-validator-sdk (Validator Check CLI — Phase 3):**
| File | Purpose |
|------|---------|
| `src/bin/riir-validator-check.rs` | Pre-upload validator: exports, smoke test, latency, memory, WASI check |
| `Cargo.toml` | `cli = ["wasmtime"]` feature + `[[bin]]` target |

**Code — microgpt-rs (Benchmarks — Phase 4.5-4.6):**
| File | Purpose |
|------|---------|
| `src/benchmark.rs` | `bench_wasm_vs_no_pruner` — DDTree build time comparison |

**Documentation (Phase 5):**
| File | Purpose |
|------|---------|
| `README.md` | Feature flags table + WASM Validator Pipeline section + project structure |
| `.docs/09_wasm_validator.md` | WASM ABI spec, security model, performance targets |
| `.research/05_Artifact_Definition.md` | WasmPruner row in existing implementations table |

**Tests:**
- microgpt-rs: 45 WASM unit tests (`cargo test --features wasm -- wasm::`)
- microgpt-rs: 22 WASM integration tests (`cargo test --features wasm -- wasm_integration`)
- riir-validator-sdk: 17 lib tests, 35 example tests
- **Total: 102 tests passing** with `--features wasm`

## Reflection Struggling/Solved

**Solved:**
1. **wasmtime 28 API differences** — `get_typed_func` requires `&mut store` not `&store`; `add_fuel` doesn't exist, use `set_fuel` instead
2. **Borrow checker with TypedFunc** — `TypedFunc::call` takes `&mut impl AsContextMut` while `TypedFunc` borrows from the same struct. Solved by moving logic into `WasmInner` methods that own `&mut self`, allowing field-level borrows
3. **edition 2024 `#[no_mangle]`** — Requires `#[unsafe(no_mangle)]` syntax in edition 2024
4. **SIGSEGV in SDK tests** — WASM memory helpers (`write_name`, `read_parent_tokens`, etc.) write to fixed addresses (0x0100, 0x0200) that are only valid inside a WASM linear memory. Removed tests that access WASM memory in native; these are tested via WasmPruner integration instead
5. **`ConstraintPruner: Send + Sync`** — WasmPruner wraps wasmtime state in `std::sync::Mutex<WasmInner>` to satisfy thread safety. Mutex is uncontended in practice (single-threaded DDTree building)

**Architecture decisions:**
- WasmPruner uses `Mutex` not `RefCell` because `ConstraintPruner` requires `Send + Sync`
- Fuel is set via `store.set_fuel(FUEL_PER_CALL)` before each call (replaces, not adds)
- Trap during `is_valid` returns `false` (reject on error — safe default)
- SDK uses `LazyLock` for static validator initialization (thread-safe, works in both WASM and native test targets)

## Remain Work

**All Plan 015 phases are complete.** No remaining tasks.

**Future considerations:**
- `type_validator.rs` example (basic Rust type syntax `:`, `->`, `<`, `>`)
- Wasmtime version: plan specified v28, currently using v28. Latest is v44 — consider upgrading
- Memory helpers are `unsafe` — could add safer wrappers with bounds checking
- CI pipeline for wasm32-unknown-unknown target (manual for now)
- Private `riir-forge` marketplace hosting (separate plan)

## Issues Ref

- Plan: `.plans/015_wasm_validator_pipeline.md`
- Related: `.plans/007_constraint_validator.md` (SynPruner, PartialParser — native validators)

## How to Dev/Test

```sh
# Phase 1: WasmPruner unit tests (microgpt-rs)
cd microgpt-rs
cargo test --features wasm -- wasm::

# Phase 2: SDK build + test (riir-validator-sdk)
cd riir-ai
cargo test
cargo build --example bracket_validator --target wasm32-unknown-unknown --release
cargo build --example keyword_validator --target wasm32-unknown-unknown --release

# Phase 3: Validator Check CLI (validate .wasm before upload)
cd riir-ai
cargo build --example bracket_validator --target wasm32-unknown-unknown --release
cargo run --features cli -- ../riir-ai/target/wasm32-unknown-unknown/release/examples/bracket_validator.wasm

# Phase 4: Integration tests (loads .wasm from SDK build)
cd riir-ai && cargo build --example bracket_validator --target wasm32-unknown-unknown --release
cd riir-ai && cargo build --example keyword_validator --target wasm32-unknown-unknown --release
cd ../microgpt-rs
cargo test --features wasm -- wasm_integration

# Phase 4.5-4.6: Benchmarks (WasmPruner vs NoPruner DDTree build)
cd microgpt-rs && cargo run --features wasm

# Full test suite with WASM
cd microgpt-rs && cargo test --features wasm

# Clippy
cd microgpt-rs && cargo clippy --features wasm --fix --allow-dirty
cd riir-ai && cargo clippy --features cli --fix --allow-dirty

# Without WASM feature (should still work — no regressions)
cd microgpt-rs && cargo test
```

**Example: Writing a new validator**
```rust
// in riir-validator-sdk/examples/my_validator.rs
use riir_validator_sdk::Validator;

struct MyValidator;
impl Default for MyValidator { fn default() -> Self { Self } }

impl Validator for MyValidator {
    fn name(&self) -> &str { "my_validator" }
    fn version(&self) -> (u8, u8, u8) { (1, 0, 0) }
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        token_idx > 0 // reject padding token
    }
}

riir_validator_sdk::export_validator!(MyValidator);
```
```sh
cargo build --example my_validator --target wasm32-unknown-unknown --release