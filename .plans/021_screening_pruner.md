# Plan 021: ScreeningPruner — Absolute Relevance for DDTree

**Branch:** `develop/feature/021_screening_pruner`
**Depends on:** None (backward compatible)
**Research:** `.research/07_Screening_Absolute_Relevance.md`

---

## Overview

Upgrade the binary `ConstraintPruner` (`is_valid -> bool`) to a continuous `ScreeningPruner` (`relevance -> f32`) that blends deterministic absolute relevance scores with LLM log-probabilities in the DDTree. This is a strict superset — existing behavior preserved via blanket impl.

---

## Tasks

- [x] **Task 1: Define `ScreeningPruner` trait** (`src/speculative/types.rs`)
  - New trait with `fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32`
  - `BinaryScreeningPruner<P>` adapter wraps any `ConstraintPruner` (R ∈ {0.0, 1.0})
  - Keep `ConstraintPruner` as-is for backward compat
  - Add `NoScreeningPruner` (returns 1.0 for everything)
  - Note: Used adapter struct instead of blanket impl to avoid conflict with `WasmPruner` custom impl

- [x] **Task 2: Add `screening_threshold` to Config** (`src/types.rs`)
  - `pub screening_threshold: f32` defaulting to `0.0`
  - When > 0.0, branches with relevance below threshold are hard-trimmed
  - When = 0.0, only `relevance() == 0.0` triggers hard trim (pure softmask)

- [x] **Task 3: Integrate into DDTree builder** (`src/speculative/dd_tree.rs`)
  - Added `build_dd_tree_screened()` public function (existing `build_dd_tree_pruned()` unchanged)
  - Score calculation: `blended = best.score + llm_prob.ln() + relevance.ln()`
  - Guard: `if relevance <= screening_threshold { continue; }` before heap push
  - Maintain `chain_seed` path: add relevance.ln() to cumulative_score

- [x] **Task 4: Update `TreeBuilder` struct and API**
  - Added `build_screened()` method alongside existing `build()` (zero breakage)
  - Chain seeding: check `relevance > threshold` instead of `is_valid`
  - Sibling seeding: same pattern
  - Best-first expansion: add relevance.ln() to score

- [x] **Task 5: Add `build_dd_tree_screened()` public API**
  - New public function alongside existing `build_dd_tree_pruned()`
  - `build_dd_tree_pruned()` remains unchanged for backward compat
  - `build_dd_tree_screened()` uses `ScreeningPruner` + threshold
  - Also added `build_and_merge_screened()` for REST integration

- [x] **Task 6: Unit tests** (`src/speculative/dd_tree.rs` tests module)
  - Test: binary pruner via adapter produces identical results to current behavior
  - Test: relevance 0.0 hard-trims branches
  - Test: relevance 0.5 applies correct penalty (`-0.69` in log space)
  - Test: relevance 1.0 produces no penalty
  - Test: threshold > 0.0 trims branches above 0.0 but below threshold
  - Test: NaN safety — relevance <= 0.0 skipped without NaN
  - Test: chain_seed with graded relevance
  - Test: empty marginals
  - All 26 dd_tree tests pass (18 existing + 8 new screening tests)

- [x] **Task 7: riir-validator-sdk WASM ABI extension** (`riir-validator-sdk/`)
  - Added `fn relevance(&self, depth, token_idx, parent_tokens) -> f32` to `Validator` trait
  - Default impl returns `1.0` (accept all at full relevance)
  - New WASM export: `relevance(depth: u32, token_idx: u32, ptr: u32, len: u32) -> u32`
  - Fixed-point Q16.16 encoding: `relevance_u32 = (relevance_f32 * 65536.0) as u32`
  - Host-side decode: `relevance_f32 = raw_u32 as f32 / 65536.0`
  - All 22 SDK tests pass (including new relevance + Q16.16 encoding tests)

- [x] **Task 8: Update microgpt-rs WasmPruner host** (`src/wasm/`)
  - Try to load `relevance` export; if missing, fall back to `is_valid` (binary 0/1 → 0.0/1.0)
  - Implement `ScreeningPruner` for `WasmPruner` via `call_relevance()`
  - Decode Q16.16 fixed-point from WASM return value, clamp to [0.0, 1.0]
  - All 22 WASM integration tests pass

- [x] **Task 9: Benchmarks**
  - All 88 tests pass with no regressions
  - `cargo clippy` clean (no new warnings)
  - DDTree build time unchanged for non-screened path (same codepaths)

- [x] **Task 10: Update README and module docs**
  - Added `ScreeningPruner: Absolute Relevance (Plan 021)` section to `README.md` architecture
  - Updated `src/speculative/types.rs` with ScreeningPruner trait docs
  - Updated `src/speculative/mod.rs` public API exports
  - Updated Project Structure section in README

---

## File Change Summary

| File | Change |
|------|--------|
| `microgpt-rs/src/speculative/types.rs` | Add `ScreeningPruner` trait + blanket impl + `NoScreeningPruner` |
| `microgpt-rs/src/types.rs` | Add `screening_threshold` to `Config` |
| `microgpt-rs/src/speculative/dd_tree.rs` | Add `build_dd_tree_screened()`, update `TreeBuilder` |
| `microgpt-rs/src/speculative/mod.rs` | Export new types |
| `microgpt-rs/src/wasm/` | Update `WasmPruner` to support `ScreeningPruner` |
| `riir-validator-sdk/src/validator.rs` | Add `relevance()` to `Validator` trait with default |
| `riir-validator-sdk/src/exports.rs` | Add `relevance` WASM export with Q16.16 |
| `microgpt-rs/README.md` | Add ScreeningPruner architecture section |

---

## Backward Compatibility

- All existing `ConstraintPruner` impls work unchanged via blanket impl
- `build_dd_tree_pruned()` API unchanged — existing code compiles as-is
- WASM validators without `relevance` export fall back to binary `is_valid`
- `Config::screening_threshold` defaults to `0.0` — no behavioral change unless opted in
- `riir-validator-sdk` gets major version bump (2.0.0) but default impl means minimal breakage

---

## Out of Scope

- anyrag integration (separate plan after SDK is ready)
- Raven RSM integration (already in `.plans/020_raven_rsm_kv_cache.md`)
- Adaptive threshold learning (future research)
- Multi-objective screening (future research)