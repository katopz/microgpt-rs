# microgpt-rs: WASM Validator Pipeline

## Architecture

The WASM Validator Pipeline allows Curators to write domain-specific constraint validators
that execute in a sandboxed WebAssembly runtime during speculative decoding.

### Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `WasmPruner` | `src/wasm/wasm_pruner.rs` | Loads .wasm, implements `ConstraintPruner` |
| `abi.rs` | `src/wasm/abi.rs` | Memory layout constants, read/write helpers |
| `state.rs` | `src/wasm/state.rs` | `ValidatorState` for wasmtime Store metadata |
| `riir-validator-sdk` | External repo | SDK for writing validators |

### WASM ABI Specification

#### Memory Layout

```
WASM Linear Memory:
  ┌─────────────────────────────────────────────┐
  │ 0x000000 - 0x0000FF  │ Validator State      │ (256 bytes, reserved)
  │ 0x000100 - 0x0001FF  │ Validator Name       │ (max 256 bytes, null-terminated)
  │ 0x000200 - 0x001FFF  │ Scratch Buffer       │ (7.5 KB for parent_tokens + strings)
  │ 0x002000+            │ Validator Heap       │ (validator's own allocations)
  └─────────────────────────────────────────────┘
```

#### Export Functions

| Export | Signature | Description |
|--------|-----------|-------------|
| `memory` | Linear memory | Required for data passing |
| `is_valid` | `(u32, u32, u32, u32) -> u32` | Token validation. Args: depth, token_idx, ptr, len. Returns 1/0. |
| `validate_string` | `(u32, u32) -> u32` | String validation. Args: ptr, len. Returns 1/0. |
| `name` | `() -> u32` | Pointer to null-terminated name (max 255 bytes). |
| `version` | `() -> u32` | Packed `(major << 16) \| (minor << 8) \| patch`. |

#### Constraints

- **No WASI imports** — fully sandboxed. No filesystem, network, env, clock.
- **No floating-point** — integer logic only, deterministic across platforms.
- **Max memory: 64 pages (4MB)** — validators must be lightweight.
- **Max execution: 100,000 fuel per call** — enforced by wasmtime (~100μs budget).

### Security Model

- Each `WasmPruner` instance has its own wasmtime `Store`
- Validators cannot access host memory outside the WASM linear memory
- Fuel-based execution prevents infinite loops
- Traps (crashes) return `false` (safe reject)
- `Mutex<WasmInner>` wraps mutable wasmtime state for `Send + Sync`

### Performance Targets

| Metric | Target | How Measured |
|--------|--------|-------------|
| `WasmPruner::load()` | <10ms | Module instantiation |
| `is_valid()` per call | <5μs | DDTree hot path |
| DDTree build overhead | ≤5% vs native | Full benchmark comparison |
| Memory per instance | <5MB | wasmtime Store footprint |

### Writing a Validator

See the [riir-validator-sdk](https://github.com/katopz/riir-validator-sdk) repository for:
- `Validator` trait definition
- `export_validator!` macro
- Memory helpers (`read_parent_tokens`, `read_string`, `write_name`)
- Example validators (`bracket_validator`, `keyword_validator`)