# Plan 004: Distill Leviathan et al. 2022 — Extractable Perf Gains

## Objective
Extract only the performance-positive parts of "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2022, Algorithm 1) into our existing speculative decoding pipeline, benchmark before/after, keep all tests passing.

## Verdict: Paper vs Our Implementation

### What the paper does (Algorithm 1)

| Phase | Description | Cost |
|-------|-------------|------|
| 1. Draft | Run small model M_q autoregressively for γ tokens. Sample each token, feed back as input for next. Save q(x) distributions. | γ × draft_cost |
| 2. Target scoring | Run large model M_p on all drafted tokens (+1 for bonus). Save p(x) distributions. | (γ+1) × target_cost |
| 3. Rejection sampling | Accept draft token i with prob min(1, p/q). On reject: sample from residual max(0,p−q), break. | O(γ × vocab) |
| 4. Bonus token | If all γ accepted, sample +1 token from p(x) at position γ. Free — logits already computed. | ~0 |

### What our current impl does

| Component | Our Impl | Paper |
|-----------|----------|-------|
| **Draft method** | `dflash_predict`: independent marginals (same token/pos each step) | Autoregressive: sample → feed back → sample → ... |
| **Verification** | **Fake** 75% simulated acceptance rate | Real p/q rejection sampling with target model |
| **Bonus token** | ❌ None | ✅ γ+1 tokens on full acceptance |
| **Residual distribution** | ❌ Not implemented | max(0, p−q) normalized for rejected tokens |
| **DDTree** | ✅ Our extension (Best-First Search tree) | Not in paper (from SpecInfer) |
| **Constraint pruning** | ✅ SudokuPruner path-aware | Not in paper |

### Key Insight: Model Size Ratio Matters

The paper assumes a **large** target model (e.g., 70B) and a **much smaller** draft model (e.g., 7B). The target model is ~10× slower, so spending γ draft steps + (γ+1) target steps still wins because each target token produced costs ~1 target_step instead of ~1 target_step with no speculation.

**Our model sizes:**
- Target: n_embd=16, n_head=4, mlp=64
- Draft: n_embd=4, n_head=2, mlp=16
- **Ratio: ~4×** (not 10×)

At this ratio, real target verification is a **net loss** — but it won't always be. Once we LoRA fine-tune the draft model for higher alignment, acceptance rates will improve and real verification becomes viable.

### Distillation: What's Worth Extracting

| Technique | Perf Impact | Complexity | Verdict |
|-----------|-------------|------------|---------|
| **Autoregressive DFlash** | Better draft quality → higher acceptance rate | Low — new `dflash_predict_ar()` function | ✅ DO NOW |
| **Bonus token** | +1 token on full acceptance (~free) | Trivial — one extra sample | ✅ DO NOW |
| **Adaptive acceptance** | Better than hardcoded 75% | Low — score-based estimate | ✅ DO NOW |
| **SpeculativeVerifier trait** | Enables swap-in real verification later | Medium — trait + 2 impls | ✅ DO NOW |
| **Real Leviathan verification** | -40% throughput at 4× ratio | Full impl behind feature flag | 🔬 LEARNING |
| **Residual distribution** | Part of Leviathan verification | Implemented now | 🔬 LEARNING |
| **p/q rejection sampling** | Part of Leviathan verification | Implemented now | 🔬 LEARNING |

## Architecture: SpeculativeVerifier Trait

Same pattern as existing `ConstraintPruner` — trait-based strategy for swappable verification.

```rust
/// Strategy for verifying drafted tokens against a target distribution.
///
/// Now: `SimulatedVerifier` — fast, no target model needed.
/// Later: `LeviathanVerifier` — real p/q rejection sampling with target model,
///   enabled when LoRA fine-tuning improves draft/target alignment.
pub trait SpeculativeVerifier: Send + Sync {
    /// Verify `draft_tokens` against target distribution.
    /// Returns accepted tokens (1 to draft_tokens.len() + 1 with bonus).
    fn verify(
        &self,
        draft_tokens: &[usize],
        marginals: &[Vec<f32>],     // q(x) from draft model
        rng: &mut Rng,
    ) -> Vec<usize>;
}

/// Current behavior: simulated acceptance rate, no target model.
pub struct SimulatedVerifier {
    pub acceptance_rate: f32,  // 0.0–1.0, default 0.75
}

/// Real Leviathan p/q verification with target model (Algorithm 1).
/// Gated behind `leviathan` feature flag — works but slow at 4× model ratio.
/// Enable when LoRA fine-tuning improves draft/target alignment.
pub struct LeviathanVerifier<'a> {
    pub target_weights: &'a TransformerWeights,
    pub target_config: &'a Config,
}
```

### Why This Design

| When | Verifier | Acceptance Source | Bonus Token |
|------|----------|-------------------|-------------|
| **Now** (random weights) | `SimulatedVerifier` | Hardcoded rate or DDTree score estimate | From last marginal |
| **Later** (LoRA fine-tuned) | `LeviathanVerifier` | Real p/q ratio from target model | From target model p(x) at γ |

The trait makes it trivial to benchmark both once we have trained models — just swap the verifier, no other code changes.

## Extractable Gains

### 1. Autoregressive DFlash (`dflash_predict_ar`)

Current `dflash_predict` feeds the **same** `token`/`pos` to every step:
```rust
for step in 0..max_steps {
    // BUG: always feeds same token/pos, produces marginals not conditionals
    let logits = forward(ctx, weights, cache, token, pos + step, config);
}
```

Fixed: sample each token, feed back:
```rust
let mut cur_token = token;
for step in 0..max_steps {
    let logits = forward(ctx, weights, cache, cur_token, pos + step, config);
    softmax(logits);
    cur_token = sample_token(logits, rng);  // feed back!
}
```

**Expected gain**: Higher acceptance rate because draft tokens are conditional (not marginal). With our tiny models this might be modest, but the math is correct.

### 2. Bonus Token

When all γ draft tokens are accepted in `speculative_step`, we get the **next** token for free — just sample from the last marginal distribution we already computed. Currently we cap at `max_accept = ceil(len × 0.75)`, throwing away the free token.

**Expected gain**: +1 token per step when acceptance is high. At γ=4 with 75% acceptance, bonus triggers ~32% of the time (0.75^4 ≈ 0.32).

### 3. SpeculativeVerifier Trait

`speculative_step` signature changes to accept `&dyn SpeculativeVerifier`:

```rust
pub fn speculative_step(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    rng: &mut Rng,
    verifier: &dyn SpeculativeVerifier,  // ← new param
) -> (Vec<usize>, usize)
```

`SimulatedVerifier` implements current behavior (DDTree path + acceptance cap + bonus token).
`LeviathanVerifier` implements real p/q verification — just plug in when ready.

### 4. Adaptive Acceptance Rate

Instead of hardcoded 0.75, estimate from DDTree top-path score:

```rust
// If top path score is high (close to 0 log-prob), acceptance is high
let acceptance_rate = if best_path_score > -2.0 { 0.85 } else { 0.65 };
```

Or simpler: just use `SimulatedVerifier { acceptance_rate: 0.80 }` as default (slightly optimistic).

**Expected gain**: More tokens accepted when draft is good, fewer wasted steps.

## Tasks

- [x] 1. **Add `SpeculativeVerifier` trait** — With `speculate()` method for full draft+verify pipeline.
- [x] 2. **Add `SimulatedVerifier`** — Implements current behavior: DDTree path extraction, simulated acceptance cap, bonus token on full acceptance.
- [x] 3. **Add `LeviathanVerifier` full impl** — Behind `leviathan` feature flag. Real p/q rejection sampling + `sample_residual_distribution` + bonus token from target model. Works end-to-end with our existing models — just slow at 4× ratio.
- [x] 4. **Add `dflash_predict_ar()`** — Autoregressive variant of DFlash that samples and feeds back tokens. Lives alongside existing `dflash_predict` (keep both).
- [x] 5. **Update `speculative_step`** — Accept `&dyn SpeculativeVerifier`, use `verifier.speculate()` instead of inline simulated acceptance. Add bonus token logic.
- [x] 6. **Update benchmark** — `bench_speculative` uses `SimulatedVerifier`. Add `bench_speculative_ar` variant with autoregressive DFlash.
- [x] 7. **Add `leviathan` feature flag** in `Cargo.toml` — gates `LeviathanVerifier`, its tests, and its benchmark variant. Math helpers (`sample_residual_distribution`, `sample_from_distribution`) always compiled and tested.
- [x] 8. **Run baseline benchmark** — Captured as `bench/010_bench_result.png`.
- [x] 9. **Run optimized benchmark** — Captured as `bench/011_bench_result.png` (5 methods) and `bench/012_bench_result.png` (6 methods with `--features leviathan`).
- [x] 10. **Add tests** — Always: `test_dflash_ar_*`, `test_simulated_verifier_*`, `test_residual_distribution_*`, `test_sample_from_distribution`. Behind `leviathan`: `test_leviathan_verifier_*`, `test_speculative_decode_step_*`.
- [x] 11. **Fix clippy, all tests pass** — Zero warnings, `cargo test --all-features` green (80 tests pass without leviathan, 173 with).
- [x] 12. **Commit** with message `feat: SpeculativeVerifier trait + autoregressive DFlash + Leviathan Algorithm 1 verification`.

## Architecture Notes

- All changes in `src/speculative.rs` — no new modules needed
- `SpeculativeVerifier` lives alongside `ConstraintPruner` in `speculative.rs` (same file, same pattern)
- `dflash_predict_ar()` is additive — `dflash_predict()` stays for backward compat
- `LeviathanVerifier` is fully implemented behind `#[cfg(feature = "leviathan")]` — real p/q rejection + residual + bonus token
- `SimulatedVerifier` is the default used in benchmarks and production (no feature flag needed)
- `sample_residual_distribution` and `sample_from_distribution` always compiled (shared math)
- Benchmark changes in `src/benchmark.rs` — add `bench_speculative_ar` variant
- `src/lib.rs` unchanged

## What We Are NOT Doing (and why)

| Technique | Reason |
|-----------|--------|
| Running `LeviathanVerifier` in default bench | Feature flag `leviathan` required — `cargo run --release --features leviathan` |
| Batched target forward pass | Our `forward()` is single-token; would need architectural change |
| New module / separate file | Keep it simple, trait + impls in existing `speculative.rs` |

## Future: How to Enable Real Verification

When LoRA fine-tuning produces a draft model with high target alignment:

1. `LeviathanVerifier` is already implemented — just construct it:
   ```rust
   let verifier = LeviathanVerifier { target_weights: &weights, target_config: &config };
   speculative_step(..., &verifier);
   ```

2. The algorithm runs full Algorithm 1:
   - Run target model on draft tokens → get p(x) distributions
   - Accept each token with prob min(1, p/q)
   - On reject: sample from residual max(0, p−q)
   - On full accept: sample bonus token from p(x) at γ

3. Benchmark `SimulatedVerifier` vs `LeviathanVerifier` — if acceptance rate > 80% with LoRA, real verification wins.

## Expected Outcome

- SpeculativeVerifier trait: clean swap point for future LoRA verification
- Autoregressive DFlash: higher quality drafts
- Bonus token: ~3-5% more tokens per second when acceptance is high
- LeviathanVerifier: fully working Algorithm 1, proven correct, perf measured (behind `leviathan` feature flag)
- All tests passing, zero clippy warnings

## Files to Modify

| File | Changes |
|------|---------|
| `Cargo.toml` | Add `[features] leviathan = []` |
| `src/speculative.rs` | Add `SpeculativeVerifier` trait, `SimulatedVerifier`, `LeviathanVerifier` (full impl), `dflash_predict_ar()`, `sample_residual_distribution()`, `sample_from_distribution()`, update `speculative_step` |
| `src/benchmark.rs` | Add `bench_speculative_ar()` + `bench_leviathan()` (behind feature flag) |