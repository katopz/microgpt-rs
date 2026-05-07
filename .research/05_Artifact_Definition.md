# Artifact Definition: Deterministic Validator vs Neural Adapter

**Date:** 2025-06
**Status:** Canonical Terminology
**Context:** microgpt-rs + anyrag neuro-symbolic architecture

---

## Terminology Correction

The term "Computable LoRA" (cLoRA) is an academic metaphor that should be avoided in engineering implementation. It conflates deterministic code execution with neural weight adaptation.

In this neuro-symbolic architecture (microgpt-rs + anyrag), there are exactly **two distinct artifacts**. They operate in different phases and are made of fundamentally different materials.

---

## Artifact 1: The Deterministic Validator

**File:** `validator.wasm` / `rules.rs`

**Previously referred to as:** cLoRA, Computable LoRA, In-Model Computer.

| Property | Value |
|----------|-------|
| **What it is** | A compiled binary (WebAssembly) or a hardcoded Rust state-machine (e.g., `syn` parser, `SudokuPruner`, `SynPruner`) |
| **Material** | Executable logic / Code |
| **When it is used** | At inference time (runtime) |
| **Role** | The Referee / Training Wheels |
| **How it works** | Intercepts the speculative draft tree (DDTree) and instantly assigns probability 0.0 to any tokens that violate strict rules (e.g., Rust syntax) |
| **Why it's NOT a LoRA** | No neural weights, no matrix multiplication, cannot "learn". Only executes strict if/else logic and state transitions |

### Existing Implementations in microgpt-rs

| Implementation | Location | What it validates |
|---------------|----------|-------------------|
| `ConstraintPruner` trait | `src/speculative/types.rs` | The trait interface — pluggable into DDTree hot path |
| `NoPruner` | `src/speculative/types.rs` | Passthrough (no validation) |
| `SudokuPruner` | `src/speculative/sudoku_pruner.rs` | Path-aware row/col/box constraint checking |
| `SynPruner` | `src/validator/syn_pruner.rs` | Two-tier Rust syntax validation (bracket DFA + `syn` parse) |
| `PartialParser` | `src/validator/partial_parser.rs` | Tier 0: O(n) bracket balancing DFA |

All implement the same `ConstraintPruner` trait:

```rust
pub trait ConstraintPruner: Send + Sync {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;
}
```

### Curator Deliverable (Marketplace)

Curators upload specialized `domain_validator.wasm` files that encode domain-specific rules:

- `django_validator.wasm` — knows Django ORM patterns, rejects invalid model definitions
- `numpy_validator.wasm` — knows ndarray API, rejects wrong dtype conversions
- `async_validator.wasm` — knows async/await rules, catches missing `.await`

**License:** MIT (open source). Basic validators ship with the engine. Domain validators are Curator artifacts on the marketplace.

---

## Artifact 2: The Neural Adapter

**File:** `lora.bin` / `lora.safetensors`

**Previously referred to as:** Traditional LoRA, Muscle Memory.

| Property | Value |
|----------|-------|
| **What it is** | A file containing low-rank weight matrices (floating-point numbers: f16, bf16) |
| **Material** | Neural weights / Math |
| **When it is used** | After training (deployment) |
| **Role** | The Intelligence / Muscle Memory |
| **How it works** | Modifies the baseline probabilities of the LLM via Low-Rank Adaptation: `W' = W + AB` where A and B are low-rank matrices |
| **Why it IS a LoRA** | It mathematically alters the target model's output distribution via standard Low-Rank Adaptation |

### Not Yet Implemented

The Neural Adapter does not exist in microgpt-rs yet. This is the gap identified in `.research/03`.

### Curator Deliverable (Marketplace)

Curators upload specialized `domain_lora.bin` files:

- `reqwest_lora.bin` — makes the LLM naturally output idiomatic HTTP client code
- `serde_lora.bin` — makes the LLM naturally output correct serialization/deserialization
- `tokio_lora.bin` — makes the LLM naturally output correct async runtime code

**License:** Proprietary (SaaS). Hosted on the platform, never distributed. This is the fuel for the engine.

---

## The Symbiotic Relationship

The two artifacts form the core of the self-improving RAG loop:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  1. Validator (.wasm) forces the "dumb" base LLM       │
│     to produce valid code by aggressively pruning       │
│     its mistakes during speculative decoding.           │
│                                                         │
│  2. Valid code outputs are saved into anyrag (Turso).   │
│                                                         │
│  3. Once enough valid outputs accumulate, they are      │
│     used to train the Adapter (.bin).                   │
│                                                         │
│  4. The Adapter (.bin) is attached to the LLM,          │
│     making it permanently smarter.                      │
│                                                         │
│  5. Because the LLM is now smarter, the Validator       │
│     (.wasm) rarely has to intervene.                    │
│                                                         │
│  Result: Validator auto-generates the training data     │
│          required to build the Adapter.                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### The Flywheel

```
Validator prunes → Valid code saved → Train Adapter → LLM gets smarter
       ↑                                                    │
       └──────── Validator intervenes less ─────────────────┘
```

---

## Summary Table

| | Deterministic Validator | Neural Adapter |
|---|---|---|
| **File** | `.wasm` / `.rs` | `.bin` / `.safetensors` |
| **Material** | Code (executable logic) | Weights (floating-point math) |
| **Phase** | Runtime (inference) | Deployment (after training) |
| **Role** | Referee (prunes invalid) | Intelligence (produces valid) |
| **Learns?** | No — hardcoded rules | Yes — trained on data |
| **LoRA?** | No — not a weight adapter | Yes — standard Low-Rank Adaptation |
| **Current state** | ✅ Working (`SynPruner`, `SudokuPruner`) | ❌ Not implemented yet |
| **License** | MIT (open) | Proprietary (SaaS) |

---

## Action Items

- [ ] Rename `ComputableLora` struct to `SymbolicIntercept` or remove it (functionality already in `ConstraintPruner` implementations)
- [ ] Implement actual LoRA weight loading (rank-decomposed matrices A, B)
- [ ] Design `.safetensors` schema for Neural Adapter files
- [ ] Design `.wasm` interface for Curator-uploaded Deterministic Validators
- [ ] Update README to use "Deterministic Validator" and "Neural Adapter" terminology