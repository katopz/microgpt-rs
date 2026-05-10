# Plan 026: PPoT — Logit-Parameterized CPU Resampling

**Branch:** `develop/feature/026_ppot_logit_resampling`
**Depends on:** Plan 021 (ScreeningPruner), Plan 013 (Zero-Alloc)
**Research:** `.research/11_PPoT_Probabilistic_Programs_of_Thought.md`

---

## Overview

Distill "Probabilistic Programs of Thought" (arXiv:2604.17290) into microgpt-rs. After DFlash produces marginals, identify high-entropy positions and resample variant programs using **only CPU** — no additional GPU forward passes. Feed resampled paths through existing `ConstraintPruner` / `ScreeningPruner` for verification.

The primary integration point is **post-DDTree rescue**: when speculative decoding rejects all tree paths, try PPoT resampling on the highest-scoring rejected path before falling back to greedy. This is the highest-ROI entry point because it activates only on failure (zero overhead on success path), marginals are already computed, and any valid path found is a pure win.

**Expected result:** 2-5% improvement in speculative decoding acceptance rate at near-zero compute cost.

---

## Tasks

- [x] **Task 1: TokenRule enum and support sets** (`src/speculative/ppot/types.rs`)
  - Define `TokenRule` enum: `Digit`, `Compare`, `Arithmetic`, `Augment`, `All`
  - Each variant maps to a `fn support(&self, vocab_size: usize) -> Vec<usize>` returning token IDs in its support
  - `TokenRule::All` returns `0..vocab_size` (unrestricted resampling)
  - Support sets are computed once from tokenizer vocabulary, cached in `PpotConfig`
  - Unit tests for each rule's support set completeness

- [x] **Task 2: Per-position entropy calculation** (`src/speculative/ppot/entropy.rs`)
  - `fn token_entropy(probs: &[f32]) -> f32` — Shannon entropy `H = -Σ p*log(p)`
  - `fn identify_high_entropy_positions(marginals: &[&[f32]], threshold: f32) -> Vec<usize>` — returns positions where `H(i) > threshold`
  - `fn identify_positions_by_rule(marginals: &[&[f32]], rule: TokenRule, threshold: f32) -> Vec<usize>` — filters by both entropy and rule support
  - Entropy threshold defaults to `0.5` (tunable via config)
  - Zero-alloc variant: `identify_positions_into(marginals, threshold, &mut Vec<usize>)`
  - Unit tests: zero entropy for deterministic distribution, high entropy for uniform

- [x] **Task 3: PPoT resample core** (`src/speculative/ppot/resample.rs`)
  - `fn ppot_resample(marginals: &[&[f32]], positions: &[usize], rng: &mut Rng) -> Vec<usize>` — resample only specified positions, keep rest from base path
  - `fn ppot_resample_with_support(marginals: &[&[f32]], positions: &[usize], support: &[Vec<usize>], rng: &mut Rng) -> Vec<usize>` — resample within rule-specific support
  - `fn ppot_resample_different_value(marginals: &[&[f32]], positions: &[usize], original: &[usize], rng: &mut Rng) -> Vec<usize>` — conditioned on not reproducing original
  - Zero-alloc variant using `SpeculativeContext` scratch buffers
  - Different-value constraint via existing `sample_residual_distribution_into` with delta `q`
  - Unit tests for each variant

- [x] **Task 4: PPoT module structure** (`src/speculative/ppot/mod.rs`)
  - Public API: `ppot_rescue()`, `ppot_augment_tree()`
  - Re-export `TokenRule`, entropy functions, resample functions
  - `PpotConfig` struct: `entropy_threshold: f32`, `num_samples: usize`, `rule: TokenRule`, `different_constraint: bool`
  - Wire into `src/speculative/mod.rs` with `#[cfg(feature = "ppot")]` feature gate

- [x] **Task 5: Post-DDTree rescue integration** (`src/speculative/step.rs`)
  - Add `ppot_rescue()` function called after DDTree verification fails
  - Pipeline: extract marginals → identify high-entropy positions → resample m paths → screen each through `ScreeningPruner` → return first valid
  - Falls back to greedy only if PPoT rescue also fails
  - Feature-gated behind `ppot` feature flag (opt-in, no default overhead)
  - Integration test: rescue finds valid path when DDTree rejects all

- [x] **Task 6: Config extensions** (`src/types.rs`)
  - `pub ppot_entropy_threshold: f32` — default `0.5`
  - `pub ppot_num_samples: usize` — default `10`
  - `pub ppot_rule: String` — default `"all"`, one of `digit|compare|arithmetic|augment|all`
  - `pub ppot_enabled: bool` — default `false` (must opt-in)
  - Parse from existing config file format, backward compatible (missing fields use defaults)

- [x] **Task 7: Benchmarks** (`src/benchmark.rs`)
  - Benchmark: entropy calculation overhead (should be <1% of DFlash time)
  - Benchmark: PPoT resample throughput (samples/ms on CPU)
  - Benchmark: end-to-end speculative decoding with PPoT rescue vs without
  - Before/after acceptance rate comparison
  - Add to benchmark output with `ppot` feature flag

- [x] **Task 8: Update README and module docs**
  - Add `PPoT: Logit-Parameterized CPU Resampling (Plan 026)` section to architecture
  - Update Project Structure with `src/speculative/ppot/` directory
  - Update feature flags section with `ppot`
  - Reference `.research/11_PPoT_Probabilistic_Programs_of_Thought.md`

---

## File Change Summary

| File | Change |
|------|--------|
| `microgpt-rs/src/speculative/ppot/mod.rs` | New: module root, public API, `PpotConfig` |
| `microgpt-rs/src/speculative/ppot/types.rs` | New: `TokenRule` enum with support sets |
| `microgpt-rs/src/speculative/ppot/entropy.rs` | New: entropy calculation, position identification |
| `microgpt-rs/src/speculative/ppot/resample.rs` | New: CPU resampling core |
| `microgpt-rs/src/speculative/mod.rs` | Add `pub mod ppot` (feature-gated) |
| `microgpt-rs/src/speculative/step.rs` | Add `ppot_rescue()` integration |
| `microgpt-rs/src/types.rs` | Add `PpotConfig` fields to `Config` |
| `microgpt-rs/src/benchmark.rs` | Add PPoT benchmarks |
| `microgpt-rs/Cargo.toml` | Add `[features] ppot = []` |
| `microgpt-rs/README.md` | Add PPoT architecture section |

---

## Architecture

```
DFlash → DDTree → Verify
                ↓ (all rejected)
          ┌─────────────────────────────────┐
          │     PPoT Rescue (CPU only)       │
          │                                 │
          │  1. Read marginals_flat          │
          │  2. Calculate per-position H(i)  │
          │  3. Identify |L| high-H positions│
          │  4. For m samples:               │
          │     a. Resample positions in L   │
          │     b. Screen via ScreeningPruner│
          │     c. If valid → return path    │
          │  5. All invalid → greedy fallback│
          └─────────────────────────────────┘
```

### Entropy Calculation
```
H(i) = -Σ_{x ∈ vocab} P(x) × ln(P(x))

Threshold H > 0.5 → position is "uncertain" → candidate RV
```

### Resample with Different-Value Constraint
```
P_sample(x | x ≠ x_original) = normalize(max(0, P(x) - δ(x, x_original)))

Uses existing sample_residual_distribution_into() with q = delta at original token
```

### TokenRule Support (from PPoT Appendix C)
| Rule | Tokens | Use Case |
|---|---|---|
| `Digit` | `0-9` (token IDs for digit strings) | Math word problems |
| `Compare` | `==, >, <, !=, <=, >=` | Conditional logic |
| `Arithmetic` | `+, -, *, /, //, **` | Expression correction |
| `Augment` | `+=, -=, *=, /=, //=` | Assignment correction |
| `All` | Full vocabulary | General purpose |

---

## Backward Compatibility

- Feature-gated behind `ppot` feature flag — **zero overhead when disabled** (default)
- `Config` fields have sensible defaults — existing config files work unchanged
- All existing `ConstraintPruner` / `ScreeningPruner` implementations work as-is
- DDTree, DFlash, verifier pipeline untouched — PPoT is a post-processing addon
- No WASM ABI changes — `riir-validator-sdk` not affected
- No `anyrag` changes needed — PPoT operates on already-computed marginals

---

## Performance Targets

| Metric | Target | Rationale |
|---|---|---|
| Entropy calculation overhead | < 0.1% of DFlash time | O(vocab × lookahead) on CPU |
| PPoT resample throughput | > 10,000 samples/ms | Simple categorical sampling |
| Rescue activation rate | ~20-30% of decoding steps | Where DDTree fully rejects |
| Acceptance rate improvement | +2-5% over baseline | From PPoT paper GSM8k results |
| Wall-clock overhead | < 1% total | Paper shows m=0 and m=20 curves indistinguishable |

---

## Out of Scope

- Full PPoT as primary sampling strategy (defer until rescue proves insufficient)
- Tokenizer-aware regex position identification (requires tokenizer module integration)
- Subset resampling for structured output (CRUXEval-style, future plan)
- Token type learning / automatic rule discovery
- Autoregressive position cascade (resample succeeding tokens after RV change)
- `anyrag` integration for document-level PPoT
- `riir-validator-sdk` WASM-side PPoT (stays host-side)

---

## References

- "Probabilistic Programs of Thought" (arXiv:2604.17290)
- PPoT Reference Implementation: `raw/PPoT/ppot/`
- Research: `.research/11_PPoT_Probabilistic_Programs_of_Thought.md`
- Screening Pruner: `.plans/021_screening_pruner.md`
- Zero-Alloc: `.plans/013_zero_alloc_rayon.md`
