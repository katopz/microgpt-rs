# Plan 015: WASM Validator Pipeline — WasmPruner Runtime + SDK + Curator Workflow

## Objective

Build the production pipeline for `validator.wasm` — the Deterministic Validator artifact that Curators write, compile, and upload to the marketplace. This plan covers three pieces:

1. **`WasmPruner`** — a `ConstraintPruner` impl in microgpt-rs that loads and executes `.wasm` files via Wasmtime
2. **`riir-validator-sdk`** — a separate MIT-licensed crate that Curators use to write domain validators
3. **Curator workflow** — the build→validate→upload pipeline (marketplace hosting in private `riir-forge` repo)

## The Problem

Current validators (`SynPruner`, `SudokuPruner`) are hardcoded Rust compiled into the binary. The Curator Marketplace requires Curators to upload specialized `domain_validator.wasm` files (e.g., `django_validator.wasm`, `numpy_validator.wasm`) that the engine loads at runtime.

The engine needs a sandboxed runtime that:
- Loads untrusted `.wasm` files safely
- Calls `is_valid(depth, token_idx, parent_tokens)` per DDTree node (~100 calls/step)
- Has ≤5% overhead vs native `ConstraintPruner` impl
- Works across all platforms (macOS, Linux, Windows, WASM/browser)

## Runtime Verdict: Wasmtime

Based on [WASM runtime comparison 2026](https://reintech.io/blog/wasmtime-vs-wasmer-vs-wasmedge-wasm-runtime-comparison-2026):

| Criteria | Wasmtime | Wasmer | WasmEdge |
|----------|----------|--------|----------|
| Rust-first API | ✅ Native | Good | Good |
| Plugin systems | ✅ Best-in-class | Good | Limited |
| Sandboxing | ✅ Capability-based (WASI) | Good | Good |
| Overhead vs native | 5-10% | 2-21% (backend-dependent) | 8% |
| Memory footprint | 15MB | 12MB | 8MB |
| Instantiation | 2-5ms | <1ms (Singlepass) | 1.5ms |
| WASI support | ✅ Complete | Complete | Complete + extensions |
| Production maturity | ✅ High | High | Medium-High |
| Component Model | ✅ Full support | In progress | Partial |
| Ecosystem | Bytecode Alliance (Mozilla, Fastly, Intel, Microsoft) | WAPM packages | CNCF/Kubernetes |

**Decision: Wasmtime.** Reasons:
1. **Rust-native.** Same language as microgpt-rs. No FFI bridge needed.
2. **Plugin system design.** Built for exactly our use case — loading untrusted modules into a host application.
3. **Sandboxing.** Capability-based security via WASI. Curator `.wasm` files can't access filesystem, network, or env vars unless explicitly granted.
4. **Bytecode Alliance backing.** Mozilla, Fastly, Intel, Microsoft. Not going away.
5. **Component Model.** Full support — future-proof for typed interfaces.

Wasmer's multi-backend flexibility is unnecessary (we don't need LLVM optimization for rule-checking functions). WasmEdge's cloud-native focus doesn't match our plugin system use case.

## Security & Repo Split

Per `.research/03_Commercial_Open_Source_Strategy_Verdict.md`:

| Component | Repo | License | Rationale |
|-----------|------|---------|-----------|
| `WasmPruner` runtime | `microgpt-rs` | MIT | Plumbing — loads .wasm into DDTree. Useless without validators. |
| `riir-validator-sdk` | New repo `riir-validator-sdk` | MIT | Curator-facing SDK. Must be OSS for adoption. Thin wrapper, no secrets. |
| Marketplace hosting | Private repo `riir-forge` | Proprietary | Curator upload, quality gate, validation, hosting. Protects Curator IP. |
| Semantic validator | Private repo `aegis-validator` | Proprietary | Secret C — sandboxed cargo check loop. Not part of this plan. |

**Key insight:** The SDK and runtime are "plumbing" — technically impressive but useless without Curator domain knowledge. Making them MIT maximizes Curator adoption (top-of-funnel). The moat is the hosted marketplace and the Curator IP, not the loading mechanism.

## Architecture

### The WASM ABI (Contract Between SDK and Runtime)

Curators implement a simple C-compatible interface:

```
Exports:
  is_valid(depth: u32, token_idx: u32, parent_tokens_ptr: u32, parent_tokens_len: u32) -> u32
  validate_string(code_ptr: u32, code_len: u32) -> u32
  name() -> u32          // returns pointer to validator name string
  version() -> u32       // returns validator version as packed u32

Imports (provided by host):
  abort(message_ptr: u32, message_len: u32)  // trap with error message
```

Return values: `1` = valid, `0` = invalid.

The SDK wraps this in a safe Rust trait so Curators never touch raw pointers:

```rust
// In riir-validator-sdk
pub trait Validator: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> (u8, u8, u8);
    
    /// Check if token_idx at given depth is valid given parent tokens.
    /// Called per DDTree node (~100 calls per step).
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;
    
    /// Optional: validate a decoded string (for code-level validation).
    /// Called after DDTree build for top-K candidate paths.
    fn validate_string(&self, code: &str) -> bool {
        true // default: accept everything
    }
}
```

### WasmPruner in microgpt-rs

```rust
// In microgpt-rs, behind feature flag "wasm"
pub struct WasmPruner {
    store: wasmtime::Store<ValidatorState>,
    is_valid_fn: TypedFunc<(u32, u32, u32, u32), u32>,
    validate_string_fn: TypedFunc<(u32, u32), u32>,
    memory: Memory,
}

impl ConstraintPruner for WasmPruner {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // Write parent_tokens to WASM linear memory
        // Call is_valid_fn
        // Return result == 1
    }
}
```

### Curator Workflow

```
1. Create project:
   cargo new my-django-validator
   cd my-django-validator

2. Add SDK dependency:
   cargo add riir-validator-sdk

3. Implement Validator trait:
   use riir_validator_sdk::Validator;
   
   struct DjangoValidator;
   impl Validator for DjangoValidator { ... }

4. Build for WASM:
   cargo build --target wasm32-unknown-unknown --release

5. Validate locally:
   riir-validator-check target/wasm32-unknown-unknown/release/my_django_validator.wasm

6. Upload to marketplace (via CLI or web):
   riir-validator-upload my_django_validator.wasm --domain django --provenance "github.com/django/django@5.0"
```

### Data Flow

```
Curator writes validator.rs
         │
         ▼
cargo build --target wasm32-unknown-unknown
         │
         ▼
domain_validator.wasm ──► riir-validator-check (local validation)
         │
         ▼ (upload)
riir-forge marketplace (private: quality gate, hosting)
         │
         ▼ (download per request)
microgpt-rs WasmPruner ──► wasmtime::Instance
         │
         ▼
build_dd_tree_pruned(&marginals, &config, &wasm_pruner)
         │
         ▼
DDTree with domain-specific pruning
```

## Dependency Additions (`Cargo.toml`)

```toml
[dependencies]
# ... existing ...
wasmtime = { version = "28", optional = true }
wat = { version = "1", optional = true }          # WAT text format parser (dev/debug)

[features]
default = []
# ... existing ...
wasm = ["wasmtime", "wat"]                         # WASM validator runtime
full = ["leviathan", "sudoku", "validator", "rest", "gpu", "wasm"]
```

## Phase Breakdown

### Phase 1: WasmPruner Core (microgpt-rs)

The runtime that loads `.wasm` files and implements `ConstraintPruner`.

Files to create:
- `src/wasm/mod.rs` — re-exports
- `src/wasm/wasm_pruner.rs` — `WasmPruner` struct implementing `ConstraintPruner`
- `src/wasm/abi.rs` — WASM ABI constants, memory layout helpers
- `src/wasm/state.rs` — `ValidatorState` for wasmtime Store

Key implementation:
```
WasmPruner::load(path) -> Result<WasmPruner>
  - Engine::default()
  - Module::from_file()
  - Linker::new() (no WASI by default — sandboxed)
  - Store::new()
  - Instance::new()
  - Extract exported functions + memory

WasmPruner::is_valid(depth, token_idx, parent_tokens)
  - Write parent_tokens to WASM linear memory (pre-allocated region)
  - Call is_valid_fn(depth, token_idx, ptr, len)
  - Return result == 1
```

Performance target: ≤5% overhead vs native `SynPruner` in DDTree build.

### Phase 2: SDK Crate (new repo `riir-validator-sdk`)

A thin MIT-licensed crate that Curators depend on.

Structure:
```
riir-validator-sdk/
  src/
    lib.rs          — Validator trait, re-exports
    validator.rs    — Validator trait definition
    macros.rs       — #[validator] proc-macro (optional, Phase 3)
    exports.rs      — #[no_mangle] extern "C" functions (glue code)
    memory.rs       — WASM linear memory helpers for SDK users
  tests/
    basic.rs        — test that a simple validator compiles
  examples/
    bracket_validator.rs  — example: bracket balancing validator
    keyword_validator.rs  — example: keyword acceptance validator
  Cargo.toml
```

The `exports.rs` generates the WASM ABI boilerplate:
```rust
// Curator writes:
use riir_validator_sdk::Validator;

struct MyValidator;
impl Validator for MyValidator {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // domain-specific rules
    }
}

riir_validator_sdk::export_validator!(MyValidator);
```

The `export_validator!` macro generates:
```rust
#[no_mangle]
pub extern "C" fn is_valid(depth: u32, token_idx: u32, ptr: u32, len: u32) -> u32 {
    // Read parent_tokens from WASM memory
    // Call MyValidator::is_valid
    // Return 1 or 0
}
```

### Phase 3: Validator Check Tool (CLI)

A CLI tool that validates a `.wasm` file before upload:
- Checks required exports exist (`is_valid`, `name`, `version`)
- Runs smoke tests (calls `is_valid` with known inputs)
- Measures latency (must be <50μs per call)
- Checks memory usage (must be <1MB)
- Verifies no WASI imports (fully sandboxed)

This lives in `riir-validator-sdk` as a binary target:
```
cargo install riir-validator-sdk --features cli
riir-validator-check path/to/validator.wasm
```

### Phase 4: Example Curator Validators

Built-in validators that demonstrate the SDK:

| Validator | What it validates | File |
|-----------|-------------------|------|
| `bracket_validator` | Bracket balancing (like PartialParser) | `riir-validator-sdk/examples/` |
| `keyword_validator` | Rust keyword placement rules | `riir-validator-sdk/examples/` |
| `type_validator` | Basic Rust type syntax (`:`, `->`, `<`, `>`) | `riir-validator-sdk/examples/` |

These also serve as integration tests for the WasmPruner runtime.

### Phase 5: Integration Tests

In microgpt-rs `tests/integration.rs`:

```rust
#[cfg(feature = "wasm")]
mod wasm_pruner {
    // Load example validator.wasm
    // Build DDTree with WasmPruner
    // Verify pruning matches expected behavior
    // Benchmark: WasmPruner vs SynPruner overhead
}
```

## WASM ABI Specification

### Memory Layout

```
WASM Linear Memory:
  ┌─────────────────────────────────────────────┐
  │ 0x000000 - 0x0000FF  │ Validator State      │ (reserved, 256 bytes)
  │ 0x000100 - 0x0001FF  │ Validator Name       │ (max 256 bytes, null-terminated)
  │ 0x000200 - 0x001FFF  │ Scratch Buffer       │ (7.5 KB for parent_tokens + strings)
  │ 0x002000+            │ Validator Heap       │ (validator's own allocations)
  └─────────────────────────────────────────────┘
```

### Export Functions

| Export | Signature | Description |
|--------|-----------|-------------|
| `is_valid` | `(u32, u32, u32, u32) -> u32` | Token-level validation. Args: depth, token_idx, parent_tokens_ptr, parent_tokens_len. Returns 1=valid, 0=invalid. |
| `validate_string` | `(u32, u32) -> u32` | String-level validation. Args: code_ptr, code_len. Returns 1=valid, 0=invalid. |
| `name` | `() -> u32` | Returns pointer to null-terminated name string (max 256 bytes). |
| `version` | `() -> u32` | Returns version as `(major << 16) \| (minor << 8) \| patch`. |

### Import Functions (Host → Guest)

| Import | Signature | Description |
|--------|-----------|-------------|
| `abort` | `(u32, u32)` | Trap with error. Args: message_ptr, message_len. Only used for debug. |

### Constraints

- **No WASI imports.** Validators are fully sandboxed. No filesystem, no network, no env vars, no clock.
- **No floating-point.** Validators use integer logic only. Deterministic across platforms.
- **Max memory: 64 pages (4MB).** Validators must be lightweight.
- **Max execution time: 100μs per `is_valid` call.** Enforced by wasmtime fuel mechanism.

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| `WasmPruner::load()` | <10ms | Module instantiation |
| `is_valid()` per call | <5μs | DDTree hot path |
| DDTree build overhead | ≤5% vs native | Full benchmark comparison |
| Memory per instance | <5MB | wasmtime Store footprint |
| Concurrent validators | 10+ | Multiple WasmPruner instances |

## Tasks

### Phase 1: WasmPruner Core
- [x] 1.1 Add `wasm` feature to `Cargo.toml` with `wasmtime` + `wat` deps
- [x] 1.2 Create `src/wasm/mod.rs` with re-exports (behind `#[cfg(feature = "wasm")]`)
- [x] 1.3 Create `src/wasm/abi.rs` with ABI constants and memory layout
- [x] 1.4 Create `src/wasm/state.rs` with `ValidatorState` struct
- [x] 1.5 Create `src/wasm/wasm_pruner.rs` with `WasmPruner` implementing `ConstraintPruner`
- [x] 1.6 Add `pub mod wasm;` to `src/lib.rs` (behind `#[cfg(feature = "wasm")]`)
- [x] 1.7 Add fuel-based execution limit (100μs per call budget)
- [x] 1.8 Add tests: load, is_valid, invalid_wasm, missing_exports

### Phase 2: SDK Crate (`riir-validator-sdk`)
- [x] 2.1 Create new repo `riir-validator-sdk` (MIT license)
- [x] 2.2 Define `Validator` trait in `src/validator.rs`
- [x] 2.3 Implement `export_validator!` macro in `src/exports.rs`
- [x] 2.4 Implement WASM memory helpers in `src/memory.rs`
- [x] 2.5 Create `bracket_validator.rs` example
- [x] 2.6 Create `keyword_validator.rs` example
- [x] 2.7 Verify `cargo build --target wasm32-unknown-unknown` works for examples
- [x] 2.8 Add CI: build + test on wasm32-unknown-unknown target

### Phase 3: Validator Check CLI
- [x] 3.1 Add `cli` feature to `riir-validator-sdk` Cargo.toml
- [x] 3.2 Implement `src/bin/riir-validator-check.rs`
- [x] 3.3 Check: required exports exist
- [x] 3.4 Check: smoke test with known inputs
- [x] 3.5 Check: latency measurement (<50μs per call)
- [x] 3.6 Check: memory usage (<1MB)
- [x] 3.7 Check: no WASI imports (sandboxed)
- [x] 3.8 Output: pass/fail report with details

### Phase 4: Integration
- [x] 4.1 Build example validators from SDK as `.wasm` files
- [x] 4.2 Add WASM integration tests in `tests/integration.rs` (behind `wasm` feature)
- [x] 4.3 Test: WasmPruner loads example `.wasm` files
- [x] 4.4 Test: DDTree build with WasmPruner produces correct results
- [x] 4.5 Benchmark: WasmPruner vs SynPruner overhead
- [x] 4.6 Benchmark: WasmPruner vs NoPruner tree quality

### Phase 5: Documentation
- [x] 5.1 Update README with `wasm` feature flag description
- [x] 5.2 Add "Curator Guide" section: how to write, build, and upload validators
- [x] 5.3 Add WASM ABI spec to `.docs/`
- [x] 5.4 Update `.research/05_Artifact_Definition.md` with WASM production details

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Wasmtime overhead too high for DDTree hot path | Can't use for token-level pruning | Measure early (Phase 1). Fallback: use WasmPruner for Tier 1 only (post-DDTree path validation), keep native for hot path. |
| Curator validators crash/loop | DoS attack on engine | Fuel-based execution limit. WasmPruner catches traps and returns `false` (reject). |
| WASM memory corruption | Incorrect validation results | Validators run in isolated instances. No shared memory between validators. |
| SDK not ergonomic enough | Curators don't adopt | Phase 2 example validators serve as templates. `export_validator!` macro hides ABI details. |
| `syn` can't compile to WASM | Can't use SynPruner as WASM validator | Expected. WASM validators are Tier 0 (rule-based). Tier 1 (AST parsing) stays native. |
| Wasmtime version conflicts | Build failures | Pin wasmtime version in Cargo.toml. SDK uses same version. |

## Relationship to Other Artifacts

### `.bin` (Neural Adapter) — Handled by anyrag

Per `anyrag/.plans/003_self_improving_cycle.md`:
- `anyrag` records `TranslationEpisode`s (source, generated Rust, compilation result)
- `SelfImprovingCycle` orchestrates the 32-day training pipeline
- `KnowledgeExporter::export_for_lora()` produces JSONL training data
- microgpt-rs Plan 008 consumes JSONL → trains `lora.bin` via wgpu

The `.wasm` (this plan) and `.bin` (anyrag plan 003) form the two Curator artifacts:
- `.wasm` = Deterministic Validator (rules engine, OSS tooling)
- `.bin` = Neural Adapter (trained weights, SaaS hosting)

### `.wasm` Hosting — Handled by `riir-forge` (private)

Curator upload, quality gate, and hosting happen in the private `riir-forge` repo:
- Curator uploads `domain_validator.wasm` + `domain_lora.bin` + provenance
- Platform runs `riir-validator-check` as quality gate
- Platform hosts `.wasm` files securely (Curator IP protection)
- Platform injects `.wasm` per-translation request based on buyer's dependency analysis

This plan only covers the OSS plumbing (runtime + SDK). The private marketplace infrastructure is a separate plan in `riir-forge`.

## Feature Flags

```toml
[features]
default = []
leviathan = []
sudoku = []
validator = ["syn", "proc-macro2"]
rest = ["reqwest", "tokio"]
gpu = ["wgpu", "bytemuck", "pollster", "safetensors"]
wasm = ["wasmtime", "wat"]                          # NEW: WASM validator runtime
full = ["leviathan", "sudoku", "validator", "rest", "gpu", "wasm"]
```

## Files to Create/Modify

| File | Action | Phase |
|------|--------|-------|
| `Cargo.toml` | Add `wasm` feature + deps | 1 |
| `src/lib.rs` | Add `pub mod wasm;` | 1 |
| `src/wasm/mod.rs` | New | 1 |
| `src/wasm/abi.rs` | New | 1 |
| `src/wasm/state.rs` | New | 1 |
| `src/wasm/wasm_pruner.rs` | New | 1 |
| `tests/wasm_validator.rs` | New (behind `wasm` feature) | 4 |
| `README.md` | Add `wasm` feature flag | 5 |

## References

- `.research/03_Commercial_Open_Source_Strategy_Verdict.md` — Moat definitions, repo split
- `.research/04_LoRA_Architecture_Verdict.md` — Artifact terminology
- `.research/05_Artifact_Definition.md` — Deterministic Validator vs Neural Adapter
- `.plans/007_constraint_validator.md` — SynPruner, PartialParser (native validators)
- `.plans/008_wgpu_lora_training.md` — GPU LoRA training (produces `.bin`)
- `anyrag/.plans/003_self_improving_cycle.md` — Self-improving cycle, JSONL export, episode recording
- [Wasmtime vs Wasmer vs WasmEdge (2026)](https://reintech.io/blog/wasmtime-vs-wasmer-vs-wasmedge-wasm-runtime-comparison-2026) — Runtime comparison
- [Wasmtime documentation](https://docs.wasmtime.dev/) — API reference