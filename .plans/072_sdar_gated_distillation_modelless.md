# Plan 072: SDAR Gated Distillation — Modelless Path

**Branch:** `develop/feature/072_sdar_gated_distillation_modelless`
**Depends on:** Plan 049 (G-Zero Phase 1), Plan 032 (HL Infrastructure), Plan 030 (Bandit)
**Research:** `.research/38_SDAR_Self_Distilled_Agentic_RL.md`
**Model-Based Twin:** `riir-ai/.plans/073_sdar_gated_distillation_model_based.md`
**Source:** https://github.com/ZJU-REAL/SDAR (audited: algorithm, ablations)
**Goal:** Adapt SDAR's token-level sigmoid gating pattern to our modelless distillation stack. Apply asymmetric trust (endorse positive gaps, attenuate negative) to bandit updates and absorb-compress promotions. No gradients — pure modelless signal gating.

**Key Insight:** SDAR proves uniform distillation collapses in multi-turn settings. The fix is a sigmoid gate: `gt = σ(β·Δt)` that modulates signal intensity per token. We adapt this pattern to our modelless stack: gate bandit reward signals by teacher-student quality gap, gate absorb-compress promotions by positive-gap-only criteria.

**Honest Scope:** This is the **modelless** (free, no-gradient) path. We don't compute teacher forward passes. Instead, we use the **asymmetric trust principle** from SDAR:
- Positive gaps (skill/validation endorses token) → strong update signal
- Negative gaps (rejection could be noise) → soft/attenuated update signal
- Sigmoid provides smooth, bounded modulation (no gradient explosion)

**Why modelless first:** Validates the gating pattern cheaply. If sigmoid-gated bandit updates improve convergence on game benchmarks, the pattern is worth porting to the gradient-based path (Plan 073).

---

## Tasks

### Phase 0: Benchmark Baseline (MUST DO FIRST)

- [ ] **T1: Create benchmark test** — `tests/bench_sdar_gated_modelless.rs`
  - Baseline: existing `DeltaGatedAbsorbCompress` + `BanditPruner` with UCB1 (scalar δ)
  - Compare: `SdarGatedAbsorbCompress` + `SdarBanditPruner` with sigmoid-gated δ
  - Metrics: DDTree nodes, accept rate, bandit regret convergence, win rate (1000 episodes)
  - Domains: Bomber arena (single quality axis), Go 9×9 (positional multi-axis)
  - Hyperparameters from paper: β=5.0 (sigmoid sharpness), λ=0.01 (auxiliary weight analog)
  - **Gate:** Must show measurable improvement on at least one metric before Phase 2

### Phase 1: Sigmoid Gate Primitive

The core building block — a reusable sigmoid gate function.

**From SDAR paper (Section 2.3):**
- Gate: `gt = σ(β · x)` where `σ(z) = 1 / (1 + exp(-z))`
- Properties: smooth, differentiable, bounded ∈ (0,1), monotonic
- β=5.0 is empirically optimal (β=0 = no gate, β→∞ = binary gate)

- [ ] **T2: Implement `sdar_gate` function** — `src/pruners/sdar_gate.rs`
  ```rust
  //! SDAR-inspired sigmoid gating for modelless distillation signals.
  //!
  //! Adapts the asymmetric trust principle from SDAR (arXiv:2605.15155):
  //! - Positive input (endorsement) → gate opens → strong signal
  //! - Negative input (rejection) → gate closes → attenuated signal
  //! - β controls sharpness (5.0 = paper-validated optimum)
  //!
  //! Unlike SDAR's gradient-based loss, this operates on pre-computed
  //! scalar signals (δ, relevance scores, bandit rewards).

  /// Default sigmoid sharpness from SDAR paper (β=5.0).
  pub const SDAR_BETA: f32 = 5.0;

  /// SDAR sigmoid gate: σ(β · x).
  ///
  /// Returns value in (0, 1). Positive x → gate opens, negative x → gate closes.
  #[inline]
  pub fn sdar_gate(x: f32, beta: f32) -> f32 {
      let z = beta * x;
      if z >= 0.0 {
          1.0 / (1.0 + (-z).exp()) // numerically stable for z >= 0
      } else {
          let ez = z.exp();
          ez / (1.0 + ez) // numerically stable for z < 0
      }
  }

  /// Convenience: gate with default β=5.0.
  #[inline]
  pub fn sdar_gate_default(x: f32) -> f32 {
      sdar_gate(x, SDAR_BETA)
  }

  /// Gate a scalar signal: apply asymmetric trust.
  ///
  /// signal * σ(β · gap) where gap is the trust indicator.
  /// Positive gap → signal passes through. Negative gap → signal attenuated.
  #[inline]
  pub fn sdar_modulate(signal: f32, gap: f32, beta: f32) -> f32 {
      signal * sdar_gate(gap, beta)
  }
  ```

  Module structure:
  ```
  src/pruners/
      sdar_gate.rs      ← NEW (this file)
      mod.rs             ← add `pub mod sdar_gate;`
      bandit.rs          ← existing
      g_zero/            ← existing
  ```

### Phase 2: Gated Bandit Update

Apply SDAR gate to bandit arm update magnitude.

**Motivation:** SDAR gates distillation loss by teacher-student gap. Analogously, we gate bandit Q-value updates by reward quality gap. When reward signal is noisy (negative gap), attenuate the update. When reward signal is trustworthy (positive gap), pass it through.

- [ ] **T3: Add `SdarBanditPruner` wrapper** — `src/pruners/bandit.rs`
  - Wraps existing `BanditPruner<P>` with sigmoid-gated reward updates
  - `update(arm, reward)`: compute `gap = reward - q_values[arm]`, gate = `σ(β·gap)`, update with `gated_reward = reward * gate`
  - Property: positive reward surprise → full update, negative reward surprise → attenuated update
  - This is the modelless analog of SDAR's token-level gating

- [ ] **T4: Unit tests for `SdarBanditPruner`** — `src/pruners/bandit.rs` (test module)
  - Test: gate opens for positive gap (reward > Q-value)
  - Test: gate closes for negative gap (reward < Q-value)
  - Test: convergence still reaches optimal arm (no regression vs ungated)
  - Test: β=0 degrades to uniform (no gate), β→∞ degrades to binary

### Phase 3: Gated Absorb-Compress

Apply SDAR gate to absorb-compress promotion decisions.

**Motivation:** Our existing `AbsorbCompress` uses a benefit-ratio threshold (hard binary gate). SDAR's sigmoid provides soft gating — partial credit for borderline cases.

- [ ] **T5: Add `SdarGatedAbsorbCompress` variant** — `src/pruners/g_zero/absorb.rs` (or new file)
  - Replace hard benefit-ratio threshold with sigmoid gate
  - `gap = benefit_ratio - 1.0` (positive = beneficial, negative = harmful)
  - `gate = σ(β · gap)`, promotion probability ∝ gate
  - Property: borderline promotions get partial probability instead of all-or-nothing
  - This is the modelless analog of SDAR's smooth modulation on borderline tokens

- [ ] **T6: Unit tests for `SdarGatedAbsorbCompress`**
  - Test: high benefit ratio → gate ≈ 1.0 (promote)
  - Test: zero benefit ratio → gate ≈ 0.5 (neutral)
  - Test: negative benefit ratio → gate ≈ 0.0 (block)
  - Test: β sensitivity matches paper ablation (β=5 optimal)

### Phase 4: Integration + Benchmark

- [ ] **T7: Run benchmarks from T1** — compare ungated vs gated
  - Domains: Bomber GvG (1000 rounds), Go 9×9 self-play (200 games)
  - Metrics: win rate, regret curve, DDTree accept rate
  - Record results in benchmark file

- [ ] **T8: Feature gate** — `sdar_gate` feature in `Cargo.toml`
  ```toml
  [features]
  sdar_gate = []  # off by default, opt-in
  ```
  - All new types behind `#[cfg(feature = "sdar_gate")]`

- [ ] **T9: Update README.md** — add SDAR gating section
  - Brief description, link to research doc, benchmark results

---

## Success Criteria

1. `SdarBanditPruner` converges to optimal arm (no regression vs ungated)
2. `SdarGatedAbsorbCompress` produces measurable improvement on at least one domain metric
3. All tests pass: `cargo test --features sdar_gate`
4. No regressions on existing benchmarks (run without `sdar_gate`)

## Failure Mode

If gated variants show no improvement over ungated baselines:
- Document the negative result in `.research/38_SDAR_Self_Distilled_Agentic_RL.md`
- Keep `sdar_gate` module as infrastructure for model-based Plan 073
- The gating pattern may still help at gradient level even if modelless signal gating doesn't

## Hyperparameter Guide

| Parameter | Default | Range | Effect |
|---|---|---|---|
| β (sharpness) | 5.0 | [1.0, 10.0] | 1=soft, 5=optimal, 10=binary |
| gap offset | 0.0 | varies | Shift gate center if reward distribution is biased |

## Timeline

- Phase 0 (T1): 1 day
- Phase 1 (T2): 0.5 day
- Phase 2 (T3-T4): 1 day
- Phase 3 (T5-T6): 1 day
- Phase 4 (T7-T9): 1 day
- **Total: ~4.5 days**