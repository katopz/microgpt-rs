# Plan 071: ROPD Rubric Modelless Distillation

**Branch:** `develop/feature/071_ropd_rubric_modelless`
**Depends on:** Plan 049 (G-Zero Phase 1), Plan 032 (HL Infrastructure)
**Research:** `.research/36_ROPD_Rubric_OnPolicy_Distillation.md`
**Model-Based Twin:** `riir-ai/.plans/072_ropd_rubric_model_based.md`
**Source:** `.raw/ROPD_official/` (audited: `algo/`, `prompts/`)
**Goal:** Distill ROPD's rubric-based scoring into our modelless stack. Replace scalar HintDelta with structured RubricVector — multi-criteria reward without LLM judges. Template rubrics + WASM validators provide per-criterion scoring at inference speed (~µs).

**Key Insight:** ROPD's rubric = (criterion, weight) pairs scored by binary pass/fail. Our `Validator` trait already does binary + graded validation. The gap: our reward is scalar δ, ROPD's is a weighted vector. This plan vectorizes the reward signal while keeping everything modelless.

**Honest Caveat (from Plan 053 lesson):** δ-Mem's vector corrections showed no DDTree gain — "the correction surface is too simple." Rubric vectors may face the same issue in game domains (Bomber, FFT, Go) where quality is well-captured by scalar reward. Rubrics help most in domains with **multiple independent quality axes** (code gen: correctness + style + security). Benchmarks will determine if this ships behind default features or stays opt-in.

---

## Tasks

### Phase 0: Benchmark Baseline (MUST DO FIRST)

- [ ] **T1: Create benchmark test** — `tests/bench_ropd_rubric_modelless.rs`
  - Baseline: `DeltaGatedAbsorbCompress` + `DeltaBanditPruner` with scalar δ (existing)
  - Compare: `RubricGatedAbsorbCompress` + `RubricBanditPruner` with RubricVector
  - Metrics: DDTree nodes, latency, reward convergence (1000 episodes)
  - Domains: Bomber arena (single quality axis), FFT arena (role-based multi-axis)
  - **Gate:** Must show measurable improvement on at least one metric before proceeding past Phase 2

### Phase 1: Core Types — RubricVector + RubricTemplate

The foundation: structured rubric representation that replaces scalar δ.

**From ROPD codebase:**
- `ropd.rubric.v1` schema: `{criterion_id, category, criterion, points}`
- Scoring: `s_i = Σ(w_k * v_{i,k}) / (Σ w_k + ε)` where `v_{i,k} ∈ {0,1}`

**Our adaptation:** Template-based criteria (no LLM), WASM-verifiable (no judge API).

- [ ] **T2: Implement `RubricTemplate` enum** — `src/pruners/ropd_rubric/template.rs`
  ```rust
  /// Fixed rubric criteria per domain — no LLM generation needed.
  /// Each variant maps to a deterministic WASM-checkable criterion.
  #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
  pub enum RubricCriterion {
      /// Did it answer the question / complete the task?
      TaskFulfillment,
      /// Is the output in valid format / type?
      OutputStructure,
      /// Does it satisfy constraints (budget, bounds, limits)?
      ConstraintSatisfaction,
      /// Are all required components present?
      Completeness,
      /// Is the answer correct (verifiable domains only)?
      Correctness,
  }

  /// Domain-specific rubric template: which criteria apply + their weights.
  /// Mirrors ROPD's weight semantics: 5=decisive, 4=strong, 2=supporting, 1=routine.
  #[derive(Debug, Clone)]
  pub struct RubricTemplate {
      pub criteria: Vec<(RubricCriterion, f32)>,  // (criterion, weight)
  }
  ```

  Pre-built templates per domain:
  - `RubricTemplate::bomber()` — survival + safety + efficiency (3 criteria, single-axis quality)
  - `RubricTemplate::fft_tactics()` — role_fulfillment + team_coordination + survival (3 criteria, multi-axis)
  - `RubricTemplate::generic()` — task + structure + constraints (3 criteria, baseline)

- [ ] **T3: Implement `RubricVector`** — `src/pruners/ropd_rubric/types.rs`
  ```rust
  /// Structured multi-criteria score — ROPD's reward without LLM.
  /// Replaces scalar HintDelta with per-criterion pass/fail vector.
  #[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
  pub struct RubricVector {
      /// Per-criterion scores [0.0, 1.0]
      pub scores: Vec<f32>,
      /// Per-criterion weights (importance)
      pub weights: Vec<f32>,
      /// Template that generated this rubric
      pub template_id: usize,
  }

  impl RubricVector {
      /// ROPD formula: weighted_score = Σ(w_k * v_k) / Σ(w_k)
      pub fn weighted_score(&self) -> f32;

      /// Which criteria have gaps vs a reference — for targeted absorb.
      /// Returns (criterion_index, gap_magnitude) sorted by weight × gap.
      pub fn gap_criteria(&self, reference: &RubricVector) -> Vec<(usize, f32)>;

      /// Compress to scalar for compatibility with existing bandit.
      /// Equivalent to HintDelta.value for drop-in replacement.
      pub fn to_scalar_delta(&self, reference: &RubricVector) -> f32;
  }
  ```

  **Design principle (SRP):** `RubricVector` is pure data — no domain logic. Domain-specific scoring stays in the template/validator.

- [ ] **T4: Implement `RubricScorer` trait** — `src/pruners/ropd_rubric/scorer.rs`
  ```rust
  /// Scores a response against a rubric template — no LLM.
  /// Implementations use WASM validators, pattern matching, or game-state queries.
  pub trait RubricScorer: Send + Sync {
      /// Score a response against all criteria in the template.
      /// Returns RubricVector with per-criterion pass/fail + weights.
      fn score(&self, response: &str, template: &RubricTemplate) -> RubricVector;
  }
  ```

  Two implementations:
  - `PatternScorer` — regex/pattern-based criterion checks (cheap, always available)
  - `ValidatorScorer` — wraps existing `Validator` trait as rubric scorer

### Phase 2: RubricGatedAbsorbCompress

Replace `DeltaGatedAbsorbCompress`'s scalar δ gate with rubric vector gate.

**From ROPD:** Rubric scores reveal *which criteria* the student fails.
**From our HL:** Absorb-compress promotes observed patterns to hard constraints.
**Synthesis:** Only absorb when rubric reveals a gap in a high-weight criterion.

- [ ] **T5: Implement `RubricGatedAbsorbCompress<P>`** — `src/pruners/ropd_rubric/rubric_absorb.rs`
  ```rust
  /// Absorb-compress gated by rubric vector instead of scalar δ.
  ///
  /// Key difference from DeltaGatedAbsorbCompress:
  ///   - Delta: gate on scalar δ > threshold (blind — why did it trigger?)
  ///   - Rubric: gate on specific criterion gap (targeted — "constraint #2 failed")
  ///
  /// This enables per-criterion absorb targeting:
  ///   - High-weight criterion gap → promote to hard constraint
  ///   - Low-weight criterion gap → ignore (not worth promoting)
  pub struct RubricGatedAbsorbCompress<P: ScreeningPruner> {
      inner: AbsorbCompressLayer<P>,
      /// Per-arm rubric history (replaces delta_history)
      rubric_history: Vec<RubricVector>,
      /// Per-arm reference rubric (golden/hint-assisted baseline)
      reference_rubrics: Vec<RubricVector>,
      config: RubricGatedConfig,
  }

  pub struct RubricGatedConfig {
      /// Minimum weighted gap to trigger absorb
      pub gap_threshold: f32,       // default: 0.3
      /// Only absorb gaps in criteria with weight ≥ this
      pub min_weight_for_absorb: f32,  // default: 2.0
  }
  ```

  Implements `ScreeningPruner` (delegates to inner) + `AbsorbCompress` (gated by rubric gap).

- [ ] **T6: Unit tests for RubricGatedAbsorbCompress**
  - Test: absorb triggers when high-weight criterion has gap
  - Test: absorb skipped when only low-weight criteria have gaps
  - Test: gap_criteria returns sorted by weight × gap magnitude
  - Test: no reference rubric → skip absorb (same as no δ evidence)
  - Test: compatibility with existing `AbsorbCompressLayer` inner

### Phase 3: RubricBanditPruner

Replace `DeltaBanditPruner`'s scalar δ reward with rubric-weighted score.

**From ROPD:** reward = `(student_score - teacher_score) / reward_scale`
**Our adaptation:** reward = `(student.weighted_score() - reference.weighted_score()) / max_score`

- [ ] **T7: Implement `RubricBanditPruner<P>`** — `src/pruners/ropd_rubric/rubric_bandit.rs`
  ```rust
  /// Bandit pruner using rubric-weighted scores as reward.
  ///
  /// Drop-in replacement for DeltaBanditPruner:
  ///   - Delta: reward = δ.value (scalar, intrinsic)
  ///   - Rubric: reward = weighted_gap (vector → scalar, criterion-aware)
  ///
  /// Optional: per-criterion sub-bandits for fine-grained arm selection.
  pub struct RubricBanditPruner<P: ScreeningPruner> {
      inner: BanditPruner<P>,
      num_arms: usize,
      /// Per-arm rubric scores (for logging/debugging)
      rubric_history: Vec<RubricVector>,
      config: RubricBanditConfig,
  }

  pub struct RubricBanditConfig {
      /// Use per-criterion sub-bandits instead of scalar reward
      pub per_criterion_bandits: bool,  // default: false (start simple)
  }
  ```

  `observe_rubric(arm, student_rubric, reference_rubric)` → compute reward → `inner.update(arm, reward)`.

- [ ] **T8: Unit tests for RubricBanditPruner**
  - Test: reward = positive when student outperforms reference
  - Test: reward = negative when student underperforms
  - Test: reward = 0 when scores match
  - Test: bandit converges toward arms with smaller rubric gaps
  - Test: per_criterion_bandits=false (default) matches scalar behavior

### Phase 4: Integration + Arena Benchmarks

Wire rubric components into existing game arenas for real-world validation.

- [ ] **T9: Implement `RubricPlayer` for Bomber arena** — `src/pruners/bomber/rubric_player.rs`
  - Same structure as `GZeroPlayer` but with rubric components
  - Template: `RubricTemplate::bomber()` (survival + safety + efficiency)
  - Scorer: game-state-based (alive? in blast zone? used bombs efficiently?)
  - Compare: `GZeroPlayer` (δ) vs `RubricPlayer` (rubric) vs `GreedyPlayer`

- [ ] **T10: Implement `RubricFFTPlayer` for FFT arena** — `src/pruners/fft/rubric_player.rs`
  - Template: `RubricTemplate::fft_tactics()` (role_fulfillment + team_coordination + survival)
  - Multi-axis domain where rubrics should help most
  - Compare: `GZeroFFTPlayer` (δ) vs `RubricFFTPlayer` (rubric) vs `TFTPlayer`

- [ ] **T11: Run benchmarks + record results**
  - `bench_ropd_rubric_modelless` — full comparison
  - Record in `.benchmarks/007_ropd_rubric_modelless.md`
  - **Decision gate:**
    - If rubric > δ on multi-axis domain (FFT) → ship behind feature gate
    - If rubric ≈ δ on single-axis domain (Bomber) → expected, document
    - If rubric < δ everywhere → stop, document why, keep code behind feature gate

### Phase 5: Feature Gate + Docs

- [ ] **T12: Feature gate** — `ropd_rubric = ["bandit"]` in `Cargo.toml`
  - All new code behind `#[cfg(feature = "ropd_rubric")]`
  - Off by default (same as `delta_mem`)
  - Feature implies `bandit` (reuses `AbsorbCompressLayer`, `BanditPruner`)

- [ ] **T13: Update module structure** — `src/pruners/ropd_rubric/mod.rs`
  ```
  src/pruners/ropd_rubric/
      mod.rs              — re-exports
      template.rs         — RubricCriterion, RubricTemplate
      types.rs            — RubricVector
      scorer.rs           — RubricScorer trait + PatternScorer + ValidatorScorer
      rubric_absorb.rs    — RubricGatedAbsorbCompress<P>
      rubric_bandit.rs    — RubricBanditPruner<P>
  ```

- [ ] **T14: Update documentation**
  - `.docs/09_heuristic-learning.md` — add ROPD Modelless section
  - `README.md` — add ROPD Rubric Modelless entry
  - `.docs/01_overview.md` — update module structure
  - Cross-reference `riir-ai/.plans/072_ropd_rubric_model_based.md`

---

## Benchmark Plan

### Baseline (before)

| Player | Arena | Survival | Score | Latency P50 |
|--------|-------|----------|-------|-------------|
| GZero (δ) | Bomber | 64.1% | 1.8 | 0.5µs |
| GZero (δ) | FFT | 15.8% | 0.16 | 572.8µs |

### Target (after)

| Player | Arena | Survival | Score | Latency P50 | Hypothesis |
|--------|-------|----------|-------|-------------|------------|
| Rubric | Bomber | ~64% | ~1.8 | ~0.5µs | ≈ No change (single-axis) |
| Rubric | FFT | >15.8% | >0.16 | ~600µs | ↑ Multi-axis helps |

**Success criteria:** Measurable improvement on FFT (multi-axis domain) without regression on Bomber (single-axis).

**Failure criteria (from Plan 053 lesson):** If vector corrections add overhead without quality gain (like δ-Mem's 2500% latency for 0% quality), we stop at Phase 2 and document.

---

## Design Decisions

### DRY: Reuse, Don't Reimplement

| Component | Reuses From |
|-----------|-------------|
| `AbsorbCompressLayer<P>` inner | Plan 032 (unchanged) |
| `BanditPruner<P>` inner | Plan 030 (unchanged) |
| `ScreeningPruner` trait | Existing (unchanged) |
| `RubricTemplate` | Pattern from `QueryTemplate` (Plan 049) |
| `RubricVector` | Pattern from `HintDelta` (Plan 049) |

### SOLID Principles

| Principle | Application |
|-----------|-------------|
| **SRP** | `RubricVector` = data, `RubricScorer` = scoring, `RubricGatedAbsorbCompress` = gating |
| **OCP** | New `RubricScorer` impls without modifying existing code |
| **LSP** | `RubricGatedAbsorbCompress<P>` implements same `ScreeningPruner` + `AbsorbCompress` |
| **ISP** | Separate traits for scoring (`RubricScorer`) vs gating (`AbsorbCompress`) |
| **DIP** | Depend on `ScreeningPruner` trait, not concrete pruner |

### Feature Gate Strategy

```toml
[features]
default = []
ropd_rubric = ["bandit"]  # off by default, opt-in
```

Rationale: Same pattern as `delta_mem` and `g_zero`. Rubric modelless is experimental until benchmarks prove value. If benchmarks show clear gain, promote to `default` in future plan.

---

## Relationship to Model-Based Plan (riir-ai Plan 072)

| Aspect | This Plan (071, modelless) | Plan 072 (model-based) |
|--------|---------------------------|----------------------|
| Rubric source | Template (fixed) | LLM-generated (adaptive) |
| Verifier | WASM / pattern matching | LLM API call |
| Teacher | Replays / hint-assisted | External API / offline index |
| Reward cost | ~µs | ~$0.01-0.10 per group |
| Training | None (bandit/absorb only) | GRPO/DPO weight updates |
| Shared types | `RubricVector`, `RubricTemplate`, `RubricCriterion` | Same types, more fields |

**Shared types** should be defined here (modelless) and re-exported by riir-ai's `ropd` module. The model-based plan adds LLM client wrappers and GRPO integration on top of the same `RubricVector` core.

---

## References

- `.research/36_ROPD_Rubric_OnPolicy_Distillation.md` — full codebase audit
- `.raw/ROPD_official/` — ROPD source (prompts, algo, training)
- Plan 049: G-Zero Self-Play (our intrinsic δ signal)
- Plan 053: δ-Mem Modelless (vector correction precedent — no DDTree gain)
- Plan 052: GFlowNet Modelless (flow-based modelless)
- `riir-ai/.plans/072_ropd_rubric_model_based.md` — model-based twin plan