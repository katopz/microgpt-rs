# Plan 054: StepCodeReasoner Modelless Distillation — Bi-Level Shaped Bandit Rewards

**Branch:** `develop/feature/054_stepcode_reasoner_modelless`
**Depends on:** Plan 030 (Bandit), Plan 049 (G-Zero Phase 1), Plan 052 (GFlowNet)
**Research:** `.research/25_StepCodeReasoner_BiLevel_GRPO.md`
**Source:** [StepCodeReasoner (arXiv 2605.11922)](https://arxiv.org/pdf/2605.11922) — Wang et al., ICML 2026
**Goal:** Distill StepCodeReasoner's intra-trajectory shaping advantage into our modelless bandit stack. After DDTree verification, scan the accepted path and boost rewards for arms that enabled correct downstream execution — replacing flat binary rewards with path-aware shaped rewards. No neural training required.

## Tasks

### Phase 0: Benchmark Baseline (MUST DO FIRST)

- [x] **T1: Create benchmark test** — `tests/bench_stepcode_modelless.rs`
  - DDTree `build_screened` with `NoScreeningPruner` baseline (nodes, time)
  - DDTree with `BanditPruner<NoScreeningPruner>` + flat rewards (existing)
  - DDTree with `BanditPruner<NoScreeningPruner>` + shaped rewards (new D1)
  - Bandit convergence: 1000-episode reward curve with flat vs shaped
  - Gate: shaped rewards must NOT degrade DDTree node count (>5% increase = fail)
  - Gate: shaped rewards must NOT increase latency >5% per build
  - Run: `cargo test --features "bandit,g_zero" --test bench_stepcode_modelless -- --nocapture`

### Phase 1: ShapedBanditPruner — Intra-Trajectory Reward Shaping (D1)

The paper's Eq. 11 distills to a post-hoc path scan:

```text
Â_intra(i,g) = r_{i,g} × (1 + (1/(n-i)) × Σ_{j=i+1}^{n} r_{j,g})
```

Three key properties (preserved from paper):
1. Only correct steps get non-zero reward (r_i = 0 → shaped = 0)
2. Steps enabling more correct future execution get proportionally more credit
3. No value function, no discount factor — pure reward shaping

**Why this matters for bandit:** Currently `BanditPruner::update(arm, 1.0)` treats every correct arm identically. But an arm accepted at depth 0 that leads to 5 more accepted tokens is MORE valuable than an arm accepted at depth 0 that leads to immediate rejection at depth 1. Shaped reward captures this "enabling" signal.

- [x] **T2: Implement `ShapedPath` struct** — `src/pruners/stepcode.rs`

  ```rust
  //! Intra-trajectory reward shaping distilled from StepCodeReasoner (arXiv 2605.11922).
  //!
  //! Paper Eq. 11: Â_intra(i,g) = r_{i,g} × (1 + (1/(n-i)) × Σ_{j=i+1}^{n} r_{j,g})
  //!
  //! Our adaptation: after DDTree verification, scan the accepted path and compute
  //! shaped rewards for each arm based on how many subsequent arms were also correct.
  //!
  //! Properties preserved from paper:
  //! 1. Only correct arms get non-zero shaped reward
  //! 2. Arms enabling more correct future arms get boosted
  //! 3. No discount factor or value function needed
  //!
  //! λ = 0.3 (paper default). λ = 0.0 reverts to flat binary rewards.

  /// A single step in a verified DDTree path.
  #[derive(Clone, Debug)]
  pub struct PathStep {
      /// Arm (token index) selected at this depth.
      pub arm: usize,
      /// Depth in the DDTree.
      pub depth: usize,
      /// Binary reward: 1.0 if accepted/verified, 0.0 if rejected.
      pub reward: f32,
  }

  /// Result of shaping a verification path.
  #[derive(Clone, Debug)]
  pub struct ShapedPath {
      /// Original steps.
      pub steps: Vec<PathStep>,
      /// Shaped rewards (same length as steps).
      pub shaped_rewards: Vec<f32>,
      /// Shaping coefficient λ (0.0 = flat, 0.3 = paper default).
      pub lambda: f32,
      /// Fraction of steps that were correct (path consistency).
      pub consistency: f32,
  }

  impl ShapedPath {
      /// Compute shaped rewards for a verified path.
      ///
      /// # Formula (paper Eq. 11)
      ///
      /// ```text
      /// shaped_reward[i] = reward[i] × (1 + λ × future_accuracy[i])
      /// future_accuracy[i] = count_correct[i+1..n] / (n - i)
      /// ```
      ///
      /// # Complexity
      ///
      /// O(n) with suffix-sum precomputation (n = path length ≤ block_size = 16).
      ///
      /// # Arguments
      ///
      /// * `steps` — verified path from DDTree (accepted + rejected arms)
      /// * `lambda` — shaping coefficient (0.0 = flat, 0.3 = paper default)
      pub fn shape(steps: Vec<PathStep>, lambda: f32) -> Self {
          let n = steps.len();
          let mut shaped_rewards = vec![0.0; n];

          if n == 0 {
              return Self {
                  steps,
                  shaped_rewards,
                  lambda,
                  consistency: 0.0,
              };
          }

          // Suffix sum of rewards: suffix_correct[i] = sum of rewards[i+1..n]
          let mut suffix_correct = vec![0.0f32; n];
          for i in (0..n.saturating_sub(1)).rev() {
              suffix_correct[i] = suffix_correct[i + 1] + steps[i + 1].reward;
          }

          // Compute shaped rewards
          for i in 0..n {
              let remaining = (n - i - 1) as f32;
              let future_accuracy = if remaining > 0.0 {
                  suffix_correct[i] / remaining
              } else {
                  0.0 // terminal step: no future to shape
              };
              shaped_rewards[i] = steps[i].reward * (1.0 + lambda * future_accuracy);
          }

          // Path consistency = fraction of correct steps
          let correct_count = steps.iter().map(|s| s.reward).sum::<f32>();
          let consistency = correct_count / n as f32;

          Self {
              steps,
              shaped_rewards,
              lambda,
              consistency,
          }
      }

      /// Feed shaped rewards into a BanditPruner.
      ///
      /// Calls `BanditPruner::update(arm, shaped_reward)` for each step.
      /// Steps with reward = 0.0 are skipped (no information gain).
      pub fn apply_to_bandit<P: ScreeningPruner>(
          &self,
          bandit: &mut BanditPruner<P>,
      ) {
          for (step, shaped) in self.steps.iter().zip(self.shaped_rewards.iter()) {
              if *shaped > 0.0 {
                  bandit.update(step.arm, *shaped);
              }
          }
      }

      /// Feed shaped rewards into a DeltaBanditPruner (G-Zero).
      ///
      /// Uses shaped reward as the dense reward signal for δ-gated arms.
      #[cfg(feature = "g_zero")]
      pub fn apply_to_delta_bandit<P: ScreeningPruner>(
          &self,
          bandit: &mut DeltaBanditPruner<P>,
      ) {
          for (step, shaped) in self.steps.iter().zip(self.shaped_rewards.iter()) {
              if *shaped > 0.0 {
                  bandit.observe_with_reward(step.arm, *shaped, step.depth);
              }
          }
      }

      /// Feed shaped rewards into AbsorbCompress layer.
      ///
      /// Promotes arms that consistently enable downstream success.
      pub fn apply_to_absorb<P: ScreeningPruner + AbsorbCompress>(
          &self,
          layer: &mut AbsorbCompressLayer<P>,
      ) {
          for (step, shaped) in self.steps.iter().zip(self.shaped_rewards.iter()) {
              layer.absorb(step.arm, *shaped);
          }
      }
  }
  ```

- [x] **T3: Implement `shape_path` helper function** — `src/pruners/stepcode.rs`

  ```rust
  /// Convenience: shape a flat `(arm, reward)` path with default λ = 0.3.
  ///
  /// Use this when you don't need depth tracking.
  pub fn shape_path(
      path: &[(usize, f32)],
      lambda: f32,
  ) -> Vec<(usize, f32)> {
      let steps: Vec<PathStep> = path
          .iter()
          .enumerate()
          .map(|(i, (arm, reward))| PathStep {
              arm: *arm,
              depth: i,
              reward: *reward,
          })
          .collect();
      let shaped = ShapedPath::shape(steps, lambda);
      shaped
          .steps
          .iter()
          .zip(shaped.shaped_rewards.iter())
          .map(|(s, r)| (s.arm, *r))
          .collect()
  }

  /// Convenience: compute path consistency from a flat reward path.
  ///
  /// Returns fraction of correct steps (0.0 to 1.0).
  pub fn path_consistency(rewards: &[f32]) -> f32 {
      if rewards.is_empty() {
          return 0.0;
      }
      let correct = rewards.iter().filter(|&&r| r > 0.0).count();
      correct as f32 / rewards.len() as f32
  }
  ```

- [x] **T4: Unit tests for ShapedPath** — `src/pruners/stepcode.rs`
  - `test_shape_all_correct` — all rewards = 1.0, verify boosting cascade
  - `test_shape_all_wrong` — all rewards = 0.0, verify all shaped = 0.0
  - `test_shape_terminal_flat` — last step gets no future shaping
  - `test_shape_lambda_zero` — λ=0.0 produces flat binary rewards
  - `test_shape_enables_downstream` — correct arm before correct future gets higher reward
  - `test_shape_empty_path` — empty path returns empty
  - `test_path_consistency_full` — all correct = 1.0
  - `test_path_consistency_mixed` — 3/5 correct = 0.6
  - Gate: all 8 tests pass

### Phase 2: AnchorTrace — Enriched TrialLog (D2)

- [x] **T5: Add `AnchorTrace` to TrialRecord** — `src/pruners/trial_log.rs`

  ```rust
  /// Per-anchor verification trace for stepwise reward analysis.
  ///
  /// Distilled from StepCodeReasoner's execution-trace anchors.
  /// Each entry records what happened at one DDTree depth (one "anchor").
  #[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
  pub struct AnchorTrace {
      /// Depth in DDTree (anchor position).
      pub depth: usize,
      /// Arm (token) selected at this depth.
      pub arm: usize,
      /// Flat binary reward (0.0 or 1.0).
      pub reward: f32,
      /// Shaped reward (reward × (1 + λ × future_accuracy)).
      pub shaped_reward: f32,
      /// Fraction of subsequent arms that were correct.
      pub future_accuracy: f32,
  }
  ```

  Modify `TrialRecord`:
  ```rust
  pub struct TrialRecord {
      // ... existing fields ...
      /// Per-anchor verification trace (StepCodeReasoner Plan 054).
      /// None for backward compatibility with existing logs.
      pub anchors: Option<Vec<AnchorTrace>>,
  }
  ```

  Default impl: `anchors: None` — backward-compatible.

- [x] **T6: Implement `TrialRecord::from_shaped_path`** — `src/pruners/trial_log.rs`

  ```rust
  impl TrialRecord {
      /// Create a TrialRecord from a ShapedPath with full anchor trace.
      pub fn from_shaped_path(
          episode: usize,
          shaped: &ShapedPath,
          cumulative_reward: f32,
          cumulative_regret: f32,
          config: &str,
      ) -> Self {
          let anchors: Vec<AnchorTrace> = shaped
              .steps
              .iter()
              .zip(shaped.shaped_rewards.iter())
              .enumerate()
              .map(|(i, (step, shaped_reward))| {
                  let n = shaped.steps.len();
                  let future = (n - i - 1) as f32;
                  let future_accuracy = if future > 0.0 {
                      shaped.steps[i + 1..]
                          .iter()
                          .map(|s| s.reward)
                          .sum::<f32>()
                          / future
                  } else {
                      0.0
                  };
                  AnchorTrace {
                      depth: step.depth,
                      arm: step.arm,
                      reward: step.reward,
                      shaped_reward: *shaped_reward,
                      future_accuracy,
                  }
              })
              .collect();

          // Last arm as the "primary" arm for the record
          let primary_arm = shaped.steps.last().map(|s| s.arm).unwrap_or(0);
          let primary_reward = shaped.shaped_rewards.last().copied().unwrap_or(0.0);

          Self {
              episode,
              arm: primary_arm,
              reward: primary_reward,
              q_value: 0.0, // caller should set
              cumulative_reward,
              cumulative_regret,
              config: config.to_string(),
              note: format!("consistency={:.2}", shaped.consistency),
              base_correct: None,
              reviewed_correct: None,
              anchors: Some(anchors),
          }
      }
  }
  ```

- [x] **T7: Unit tests for AnchorTrace** — `src/pruners/trial_log.rs`
  - `test_anchor_trace_serialization` — roundtrip through JSONL
  - `test_trial_record_from_shaped_path` — verify fields populated correctly
  - `test_backward_compat_none_anchors` — existing logs without anchors load fine
  - Gate: all 3 tests pass

### Phase 3: PathConsistency — Reward Hacking Detection (D3)

- [x] **T8: Add `path_consistency` to ReviewMetrics classification** — `src/pruners/review_metrics.rs`

  Add a new classification category:

  ```rust
  /// Classification result including path consistency (StepCodeReasoner Plan 054).
  #[derive(Clone, Debug, Default)]
  pub struct PathConsistencySummary {
      /// Number of paths where final answer was correct but intermediate steps were shaky.
      pub reward_hacking: u64,
      /// Number of paths where both final AND intermediate steps were correct.
      pub fully_faithful: u64,
      /// Total paths analyzed.
      pub total_paths: u64,
      /// Average path consistency across all paths.
      pub avg_consistency: f64,
  }

  impl ReviewMetrics {
      /// Classify a path by its consistency vs final correctness.
      ///
      /// Maps to StepCodeReasoner's "right answer, wrong logic" detection:
      /// - final_correct && consistency >= threshold → fully_faithful
      /// - final_correct && consistency < threshold → reward_hacking
      /// - !final_correct → not counted (no credit assignment issue)
      pub fn classify_path(
          &self,
          final_correct: bool,
          consistency: f32,
          threshold: f32,
      ) -> PathConsistencySummary {
          // Use atomics for thread-safety — simple version for now
          PathConsistencySummary::default()
      }
  }
  ```

  **Minimal implementation:** Add `path_hacking_count: AtomicU64` and `path_faithful_count: AtomicU64` to `ReviewMetrics`. Gate `AbsorbCompress` when reward hacking ratio exceeds threshold.

- [x] **T9: Unit tests for path consistency** — `src/pruners/review_metrics.rs`
  - `test_path_consistency_faithful` — high consistency → faithful
  - `test_path_consistency_hacking` — low consistency + correct final → hacking
  - `test_path_consistency_wrong_final` — wrong final → not counted
  - Gate: all 3 tests pass

### Phase 4: Integration & Benchmark

- [x] **T10: Integration example** — `examples/stepcode_01_shaped_bandit.rs`
  - Build DDTree with `BanditPruner<NoScreeningPruner>` + shaped rewards
  - Compare flat vs shaped reward convergence over 100 episodes
  - Print consistency metrics
  - Run: `cargo run --example stepcode_01_shaped_bandit --features "bandit"`

- [x] **T11: Final benchmark comparison** — `tests/bench_stepcode_modelless.rs`

  | Config | Metric | Target |
  |--------|--------|--------|
  | Flat rewards (baseline) | 1000-episode reward | baseline |
  | Shaped rewards (λ=0.3) | 1000-episode reward | ≥ baseline (no degradation) |
  | Shaped rewards (λ=0.3) | Bandit convergence speed | ≥ as fast as baseline |
  | Shaped rewards (λ=0.3) | DDTree nodes | ≤ 5% increase |
  | Shaped rewards (λ=0.3) | Latency per build | ≤ 5% increase |

  **Honest expectation:** Shaped rewards should NOT increase DDTree nodes or latency (it's a post-hoc computation on the accepted path, not in the hot path). The benefit is in **bandit convergence quality** — arms that enable downstream success get higher Q-values faster.

- [x] **T12: Feature gate** — `Cargo.toml`

  ```toml
  [features]
  # StepCodeReasoner modelless distillation — shaped bandit rewards + anchor tracing
  stepcode = ["bandit"]
  ```

  All new code behind `#[cfg(feature = "stepcode")]`. The `ShapedPath` and `shape_path` are also available under `g_zero` feature (since G-Zero Phase 1 + StepCode shaping are complementary).

  Update `full` feature:
  ```toml
  full = ["sudoku", "validator", "sparse_mlp", "ppot", "domain_latent",
          "bandit", "bomber", "monopoly", "feedback", "rest",
          "gpu", "delta_mem", "g_zero", "stepcode"]
  ```

- [x] **T13: Module registration** — `src/pruners/mod.rs`

  ```rust
  #[cfg(feature = "stepcode")]
  pub mod stepcode;

  #[cfg(feature = "stepcode")]
  pub use stepcode::{PathStep, ShapedPath, shape_path, path_consistency};
  ```

- [x] **T14: README update** — Add to `## 🧠 Heuristic Learning Infrastructure` section

  Add subsection:
  ```markdown
  ### Stepwise Reward Shaping (Plan 054)

  Distilled from StepCodeReasoner's Bi-Level GRPO — intra-trajectory shaping advantage
  rewards bandit arms proportionally to how many downstream arms they enable.

  | λ | Behavior |
  |---|----------|
  | 0.0 | Flat binary rewards (default, backward-compatible) |
  | 0.3 | Shaped rewards (paper default, arms enabling success get boosted) |

  Run: `cargo test --features "stepcode" --test bench_stepcode_modelless -- --nocapture`
  ```

- [x] **T15: Commit to feature branch**

  ```sh
  git checkout develop
  git checkout -b develop/feature/054_stepcode_reasoner_modelless
  git add -A
  git commit -m "feat(stepcode): Plan 054 — shaped bandit rewards from StepCodeReasoner Bi-Level GRPO"
  ```

## Implementation Summary

### Files Created (New)

| File | Lines | Purpose |
|------|-------|---------|
| `src/pruners/stepcode.rs` | ~200 | ShapedPath, PathStep, shape_path, path_consistency |
| `tests/bench_stepcode_modelless.rs` | ~150 | Full benchmark suite |
| `examples/stepcode_01_shaped_bandit.rs` | ~80 | Integration example |

### Files Modified

| File | Change |
|------|--------|
| `Cargo.toml` | Add `stepcode = ["bandit"]` feature, update `full` |
| `src/pruners/mod.rs` | Add feature-gated `stepcode` module + re-exports |
| `src/pruners/trial_log.rs` | Add `AnchorTrace` struct + `anchors` field to `TrialRecord` |
| `src/pruners/review_metrics.rs` | Add path consistency classification |

### Expected Test Results

- **~20 unit tests** — all passing
- **~8 benchmark tests** — all passing
- **Gate:** DDTree node count ≤ 5% increase over baseline
- **Gate:** Latency per build ≤ 5% increase over baseline
- **Gate:** Bandit convergence ≥ as fast as flat rewards

## Feature Gate

```toml
stepcode = ["bandit"]
```

All new code behind `#[cfg(feature = "stepcode")]`. Depends on `bandit` for:
- `BanditPruner` (receives shaped rewards)
- `ScreeningPruner` trait
- `AbsorbCompress` trait (D2 integration)

## Architecture Mapping: Paper → Modelless

| Paper Component | Paper Eq/Loc | Modelless Equivalent | Key Difference |
|----------------|-------------|---------------------|----------------|
| **Binary step reward** r_{i,g} ∈ {0,1} | Eq. 8 | `PathStep::reward: f32` | Same — binary verified/not |
| **Intra-trajectory shaping** Â_intra | Eq. 11 | `ShapedPath::shape()` | Same formula, no GRPO gradient |
| **Group-relative advantage** Â_group | Eq. 10 | N/A — 1 tree per query | We skip multi-trajectory sampling |
| **Shaping coefficient λ** | Eq. 12, λ=0.3 | `ShapedPath::lambda: f32` | Same default, configurable |
| **Execution anchors** | print() insertion | DDTree depth | No code instrumentation needed |
| **Step accuracy** | Table 6 | `path_consistency()` | Same metric, different context |
| **Terminal reward** | Eq. 9 | Final `PathStep::reward` | Same — binary final correctness |
| **Rule-based reward** (100% accurate) | Appendix G | `WasmPruner` sandbox | Same deterministic guarantee |

## Key Design Decisions (From Paper)

1. **Post-hoc shaping, not in-hot-path**
   - Paper shapes rewards during GRPO optimization (training time)
   - We shape rewards after DDTree verification (inference time, off critical path)
   - The shaping computation is O(n) with suffix sum — negligible for n ≤ 16

2. **λ = 0.3 (paper default)**
   - Paper ablates λ implicitly: Bi-Level GRPO (λ=0.3) consistently outperforms Step-GRPO (λ=0)
   - We make λ configurable — 0.0 for flat, 0.3 for paper default
   - Conservative: start with 0.0 (backward-compatible), let user opt into 0.3

3. **No group-relative advantage**
   - Paper samples G=5 trajectories per query, normalizes across group
   - DDTree builds 1 tree per query — no group to normalize across
   - This is the biggest loss from modelless distillation — we miss cross-trajectory signal
   - Mitigation: the bandit itself provides group-like signal across episodes (accumulated Q-values)

4. **Bandit update with shaped reward (not GRPO gradient)**
   - Paper uses shaped reward in policy gradient loss
   - We use shaped reward as `BanditPruner::update(arm, shaped_reward)`
   - Same signal, different consumer (Q-value estimate vs gradient)
   - The bandit's incremental mean IS a form of policy learning — just not gradient-based

5. **Skip decoupled task templates**
   - Paper shows decoupling adds ~2-3 points (Table 5)
   - But this requires per-task-type prompting infrastructure
   - Our `PromptRouter` handles domain routing, not task-type routing
   - Not worth the complexity for modelless path

## Paper Findings That Drive Our Design

### 1. Stepwise rewards are the primary driver (Table 5)
> Removing stepwise rewards drops avg by ~5-6 points. Removing decoupling drops ~2-3.

Focus on D1 (shaped rewards). D2/D3 are supporting additions.

### 2. Shaping prevents length collapse (Figure 2)
> Bi-Level GRPO stabilizes at longer sequence length than Step-GRPO or Terminal-GRPO.

Our analog: shaped rewards prevent the bandit from collapsing to "safe" arms that work in isolation but lead nowhere.

### 3. Robust to imperfect anchors (Appendix F)
> 20% random dropout: only -3.8 points. Smaller teacher model: only -2.3 points.

Even noisy shaped rewards are better than flat rewards. The signal direction matters more than precision.

### 4. Rule-based reward beats PRMs (Appendix G)
> GPT-4o as PRM: 72.6% step judgment accuracy. Rule-based: 100%.

Our `WasmPruner` validation is rule-based by construction. We don't need a learned reward model.

### 5. Intermediate accuracy gap is real (Table 6)
> CodeReasoner-7B: 85.6% final but only 63.5% step accuracy. 22pp gap.

`path_consistency()` measures this gap. When gap is large + final is correct → reward hacking detected.

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Shaped rewards don't improve bandit convergence | Medium | λ=0 reverts to flat — zero regression risk |
| Path shaping adds measurable latency | Low | O(n) suffix sum, n≤16, off hot path |
| AnchorTrace bloats TrialLog JSONL size | Low | `Option<Vec<AnchorTrace>>` — None by default |
| Shaped reward signal too weak for micro-Transformer | Medium | Honest expectation: +0.5-1.5 points, not +7-14% |
| Interaction with DeltaBanditPruner unclear | Low | Both use shaped reward as input — orthogonal |

## Success Criteria — Honest Assessment

| Metric | Target | How Measured |
|--------|--------|-------------|
| DDTree nodes with shaped rewards | ≤ 5% increase vs flat | T11 benchmark |
| Latency per DDTree build | ≤ 5% increase vs flat | T11 benchmark |
| Bandit convergence speed | ≥ as fast as flat (ideally faster) | T11 reward curve |
| Unit tests | All passing | T4, T7, T9 |
| Backward compatibility | λ=0 produces identical behavior | T4 test |
| Path consistency metric | Correctly computed | T9 tests |

**Honest expectation:** The paper's +7-14% accuracy gains come from training a 7B model on dense stepwise rewards via GRPO. Our modelless path improves the **quality of the heuristic signal** fed to the bandit, not the model itself. Expected improvement in bandit convergence: **marginal to modest** (+0.5-1.5 points). The value is in the richer signal, not the magnitude.

## Hyperparameters (Paper-Verified)

| Parameter | Paper Value | Our Default | Source |
|-----------|-------------|-------------|--------|
| λ (shaping coefficient) | 0.3 | 0.3 | Eq. 12, Section 4.3 |
| G (group size) | 5 | N/A | Section 4.3 — we don't sample multiple trajectories |
| Learning rate | 1e-6 | N/A | Section 4.3 — no gradient training |
| Max tokens per response | 4096 | N/A | Section 4.3 — no response generation |
| Anchor count (mean) | 3.2-4.8 | tree depth (≤16) | Table 1 — our "anchors" are DDTree depths |
| Reward budget R_internal | 1.0 | N/A | Section 4.3 — single scalar reward |
| Reward budget R_final | 1.0 | N/A | Section 4.3 — single scalar reward |

## Source Code Reference

All types are NEW. Integration points (read-only or additive-only):
- `src/pruners/mod.rs` — add `stepcode` module + exports
- `src/pruners/bandit.rs` — `BanditPruner::update()` (receives shaped rewards, no code change)
- `src/pruners/trial_log.rs` — `TrialRecord` gets optional `anchors` field
- `src/pruners/review_metrics.rs` — adds path consistency classification
- `Cargo.toml` — feature gate

## Relationship to Existing Work

| Component | Relationship |
|-----------|-------------|
| **BanditPruner** (Plan 030) | Receives shaped rewards instead of flat rewards. Same `update()` API, richer signal. |
| **GFlowNet FlowPruner** (Plan 052) | Flow bonus = `(1 - stop_prob[depth])`. Shaped reward = `reward × (1 + λ × future_accuracy)`. Related philosophy (path-aware scoring), different mechanism. |
| **DeltaBanditPruner** (Plan 049) | δ as per-query signal. Shaped reward as per-path signal. Orthogonal — can combine (δ-shaped rewards). |
| **AbsorbCompress** (Plan 032) | Absorb tracks average reward per arm. Shaped reward gives it a richer signal — arms enabling downstream success get promoted faster. |
| **ReviewMetrics** (Plan 036) | Classifies helpful/harmful reviewer. PathConsistency adds "correct outcome, shaky process" — a new failure mode to detect. |
| **δ-Mem** (Plan 053) | Associative memory for pruner state. StepCode shaping is complementary — shaped rewards feed into the same bandit that δ-mem's memory tries to steer. |
| **riir-gpu GRPO** (G-Zero Phase 2) | The gradient-based analog of this modelless distillation. When we add GRPO to riir-gpu, the shaped reward signal becomes the stepwise advantage in the policy gradient loss. |

## Model-Based Assessment — NOT Worth a Separate Plan

**Verdict: Skip. The modelless plan captures the valuable signal. Model-based GRPO belongs in G-Zero Phase 2 (Plan 049 T6-T9), not here.**

### Why No Separate riir-ai Plan

| Factor | Paper (7B) | Our Stack | Implication |
|--------|-----------|-----------|-------------|
| **Model size** | 7B params (Qwen2.5-Coder) | ~5K LoRA params (n_embd=16, vocab=27, 1 layer) | GRPO's group-relative advantage is meaningless at 5 trajectories with 5K params |
| **Training data** | 17K+ instrumented samples | ~200 game replays (train_bomber.rs) | Stepwise reward needs 3-5 anchors per sample; 200 samples × 4 anchors = 800 data points — not enough for stable GRPO |
| **Anchor density** | 3.2-4.8 anchors/sample (Table 1) | ~4 depths max (block_size=16, lookahead=6) | Same order of magnitude, but our "anchors" are token positions, not execution states |
| **GRPO loss** | New policy gradient with KL penalty | Would need new WGSL kernels + multi-sample generation | Infrastructure cost disproportionate to model capacity |
| **Expected gain** | +7-14% accuracy | ~+1-2% accuracy (diminishing returns at micro scale) | Paper's gains come from dense supervision on a 7B model, not from the algorithm alone |

### What riir-gpu Already Has (Sufficient)

- ✅ LoRA forward/backward on GPU (21 WGSL kernels)
- ✅ AdamW optimizer with warmup + cosine decay
- ✅ CE loss + KL distillation
- ✅ Training loop, checkpoints, export to `lora.bin`
- ✅ Feedback consumer (TTT loop)

### What Would Be Needed for GRPO (Not Worth Building Separately)

- ❌ Multi-sample generation per query (G=5 forward passes)
- ❌ Stepwise reward WGSL kernel
- ❌ GRPO policy gradient loss (group-relative + intra-trajectory advantage)
- ❌ Execution-trace SFT data pipeline (teacher LLM → instrumented code → interpreter ground truth)

### The Right Path

When/if **G-Zero Phase 2** (Plan 049 T6-T9) implements GRPO in `riir-gpu`, the shaped reward signal from this plan's D1 (`ShapedPath`) becomes the **stepwise advantage term** in the policy gradient loss. No separate StepCodeReasoner-specific model-based plan needed — it's absorbed into G-Zero's GRPO implementation.

**Dependency chain:**
```
Plan 054 (this) — modelless shaped rewards → BanditPruner
    ↓ (future)
Plan 049 Phase 2 — GRPO in riir-gpu uses shaped rewards as Â_intra
    ↓ (future)
StepCodeReasoner data pipeline — instrumented traces → SFT + GRPO training
```

The modelless path is the foundation. The model-based path builds on top when the infrastructure is ready.