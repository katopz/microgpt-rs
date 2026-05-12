# Plan 040: Cross-Training Techniques — Research → Game Domain (wgpu)

**Branch:** `develop/feature/040_cross_training_game_domain`
**Depends on:** Plan 038 (Domain Latent), Plan 039 (Game Replay Training Data)
**Research:** `.research/04`, `.research/07`, `.research/14`, `.research/15`, `.research/16`
**Target:** `riir-gpu` (wgpu game training) + `microgpt-rs` (feature flags)

---

## Overview

Distill training techniques from `.research/` into the game training pipeline.
Focus exclusively on **game domain** (`riir-gpu`) where we have full control.

**Language domain (`riir-burner`) is DEFERRED** — both Rust (burn) and Python (unsloth-mlx)
backends are unstable future work. `unsloth-mlx` is an external dependency we cannot modify.
`riir-burner` tasks will be planned when the pipeline matures.

### Scope

| Pipeline | Repo | Status | This Plan |
|----------|------|--------|-----------|
| Game training | `riir-gpu` | ✅ Stable, modifiable | **Focus here** |
| Language training | `riir-burner` | ❌ Unstable (both RS+PY) | ❌ Deferred |
| Language inference | `microgpt-rs` | ✅ Works | Feature flag only |

### Why Game Domain Only

1. **riir-gpu is ours** — wgpu compute shaders, full source control
2. **Small scale works** — 6-action vocab, ~6K params, trains in seconds
3. **Game replay data pipeline exists** — Plan 039 provides JSONL corpus
4. **DomainLatent already works** — Plan 038 Task 5a is ✅ in riir-gpu
5. **LoRA training verified** — `train_bomber.rs` exports real adapters

### What Was Deferred from Research

| Research | Technique | Applicable Now | Deferred |
|----------|-----------|----------------|----------|
| 04 LoRA Architecture | Draft-target distillation | ✅ Game domain | Language domain |
| 07 Screening | Gradient-guided target selection | ✅ Game domain | Language domain |
| 14 Learning Beyond Gradients | Absorb + Compress | ✅ Game domain | Coding agent loop |
| 15 Reinforced Agent | Helpfulness-Harmfulness metrics | ✅ Game domain | — |
| 16 AutoTTS | β parameterization | ✅ Game domain | — |
| 09 EMO | Document-level routing | ❌ Needs multi-domain | Language domain |
| 04 LoRA Architecture | Multi-LoRA stacking | ❌ Needs stable adapters | Language domain |
| 04 LoRA Architecture | Reader/Writer LoRA pairs | ❌ Needs Plan 025 runtime | Language domain |
| 18 Free Transformer | Domain latent for burn | ❌ riir-burner unstable | Language domain |

---

## Architecture

### Game Training Pipeline (Modified)

```
game_replay.jsonl (Plan 039)
        │
        ▼
    DataLoader (riir-gpu)
        │
        ▼
┌───────────────────────────────┐
│  Forward (wgpu compute shader)│
│  + LoRA A/B                   │
│  + DomainLatent (Plan 038)    │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  Loss (cross-entropy)         │
│  + ReviewMetrics ← NEW        │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  AdamW (wgpu compute shader)  │
│  LR via BetaConfig ← NEW      │
│  + Screening probe ← NEW      │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  Post-training analysis       │
│  Absorb + Compress ← NEW      │
│  Draft distillation ← NEW     │
│  Export:                      │
│    game_lora.bin              │
│    game_draft_lora.bin ← NEW  │
│    game.dlat                  │
│    training_report.json ← NEW │
└───────────────────────────────┘
```

### microgpt-rs Feature Flag

```toml
# Cargo.toml — separate game from language concerns
[features]
default = []
# Game domain: small models, 6-action vocab, wgpu-trained LoRA
game_domain = ["domain_latent"]
# Language domain: LLM models, BPE vocab, burn/unsloth-trained LoRA (future)
language_domain = ["domain_latent", "lora"]
# Full: everything
full = ["game_domain", "language_domain", "speculative", "sparse_mlp", "ppot"]
```

### New Types for riir-gpu

```rust
/// Single-scalar training configuration (Research 16: β parameterization).
/// All hyperparameters are deterministic monotonic functions of β ∈ [0.0, 1.0].
/// Low β = fast/cheap training. High β = thorough/expensive training.
pub struct BetaConfig {
    pub beta: f32,
}

impl BetaConfig {
    /// Learning rate: 1e-4 + 9e-4 × β (higher β → higher LR → faster convergence)
    pub fn learning_rate(&self) -> f32 {
        1e-4 + 9e-4 * self.beta
    }

    /// LoRA rank: 4 + 28 × β as usize (higher β → more capacity)
    pub fn lora_rank(&self) -> usize {
        (4.0 + 28.0 * self.beta) as usize
    }

    /// Warmup steps: 50 + 450 × β as usize
    pub fn warmup_steps(&self) -> usize {
        (50.0 + 450.0 * self.beta) as usize
    }

    /// Weight decay: 0.01 - 0.009 × β (higher β → less regularization)
    pub fn weight_decay(&self) -> f32 {
        0.01 - 0.009 * self.beta
    }

    /// Epochs: 1 + 9 × β as usize (higher β → more passes)
    pub fn epochs(&self) -> usize {
        (1.0 + 9.0 * self.beta) as usize
    }

    /// Draft LoRA rank: 2 + 6 × β as usize (smaller than target)
    pub fn draft_rank(&self) -> usize {
        (2.0 + 6.0 * self.beta) as usize
    }
}

/// Training quality metrics (Research 15: Helpfulness-Harmfulness).
/// Tracks whether training interventions are net-positive.
#[derive(Debug, Default)]
pub struct ReviewMetrics {
    /// Epochs where loss decreased (helpful).
    pub helpful_epochs: usize,
    /// Epochs where loss increased (harmful).
    pub harmful_epochs: usize,
    /// Epochs where loss stayed within tolerance (neutral).
    pub neutral_epochs: usize,
    /// Best loss achieved.
    pub best_loss: f32,
    /// Loss at which helpful/harmful is judged (first epoch's loss).
    pub baseline_loss: f32,
}

impl ReviewMetrics {
    /// Benefit-to-risk ratio = helpful / max(harmful, 1).
    /// Ratio > 2.0 = training is clearly working.
    /// Ratio < 1.0 = training is net-negative — stop and investigate.
    pub fn benefit_ratio(&self) -> f64 {
        self.helpful_epochs as f64 / self.harmful_epochs.max(1) as f64
    }

    /// Record an epoch's loss change.
    pub fn record_epoch(&mut self, prev_loss: f32, curr_loss: f32, tolerance: f32) {
        if self.baseline_loss == 0.0 {
            self.baseline_loss = prev_loss;
        }
        let delta = prev_loss - curr_loss;
        if delta > tolerance {
            self.helpful_epochs += 1;
        } else if delta < -tolerance {
            self.harmful_epochs += 1;
        } else {
            self.neutral_epochs += 1;
        }
        if curr_loss < self.best_loss || self.best_loss == 0.0 {
            self.best_loss = curr_loss;
        }
    }
}

/// Game-specific training metrics.
#[derive(Debug, Default)]
pub struct GameMetrics {
    /// Per-action accuracy on validation set.
    pub action_accuracy: [f32; 6], // Up/Down/Left/Right/Bomb/Wait
    /// Win rate on validation games (before training).
    pub baseline_win_rate: f32,
    /// Win rate on validation games (after training).
    pub final_win_rate: f32,
    /// Action distribution entropy (higher = more diverse strategies).
    pub action_entropy: f32,
    /// Number of validation games evaluated.
    pub validation_games: usize,
}

impl GameMetrics {
    /// Did training improve win rate?
    pub fn helpful(&self) -> bool {
        self.final_win_rate > self.baseline_win_rate
    }

    /// Win rate delta.
    pub fn win_delta(&self) -> f32 {
        self.final_win_rate - self.baseline_win_rate
    }
}

/// Absorb + Compress result (Research 14: Learning Beyond Gradients).
/// After training, stable patterns can be "compressed" into freeze recommendations.
#[derive(Debug)]
pub struct CompressReport {
    /// LoRA targets where training converged (< 0.01 loss change in last 3 epochs).
    pub stable_targets: Vec<String>, // target names
    /// LoRA targets still learning (active gradient).
    pub active_targets: Vec<String>,
    /// Domain latent norm (if trained).
    pub domain_latent_norm: Option<f32>,
    /// Recommendation: which targets to freeze for next training run.
    pub freeze_recommendation: Vec<String>,
}

/// Draft distillation report (Research 04: Draft-Target Alignment).
#[derive(Debug)]
pub struct DistillReport {
    /// KL divergence between draft and target distributions.
    pub kl_divergence: f32,
    /// Draft LoRA rank (smaller than target).
    pub draft_rank: usize,
    /// Target LoRA rank.
    pub target_rank: usize,
    /// Number of samples used for distillation.
    pub distill_samples: usize,
}

/// Complete training report with all metrics.
#[derive(Debug)]
pub struct GameTrainingReport {
    /// Standard training report (loss history, steps, epochs).
    pub base: TrainingReport,
    /// Review metrics (helpful/harmful epochs).
    pub review: ReviewMetrics,
    /// Game-specific metrics (win rate, action accuracy).
    pub game: GameMetrics,
    /// Compress report (post-training analysis).
    pub compress: Option<CompressReport>,
    /// Distill report (if draft was distilled).
    pub distill: Option<DistillReport>,
    /// β used for this training run.
    pub beta: f32,
}
```

---

## Tasks

- [ ] **Task 1: BetaConfig for game training** (`riir-gpu`)
  - Create `training_config.rs` in `riir-gpu` with `BetaConfig` struct
  - Derive all hyperparams from single β: `learning_rate`, `lora_rank`, `warmup_steps`, `weight_decay`, `epochs`, `draft_rank`
  - Default β = 0.3 (fast iteration), production β = 0.7 (thorough)
  - Update `TrainingConfig` to accept `Option<f32>` for β — when set, overrides individual fields
  - Update `train_bomber.rs` to support `--beta` CLI flag
  - Monotonicity constraint: all budget params NON-DECREASING in β
  - Tests: monotonicity of all derived functions, boundary values (β=0, β=1), β overrides explicit params

- [ ] **Task 2: ReviewMetrics for training quality** (`riir-gpu`)
  - Add `ReviewMetrics` to `training_config.rs`
  - Record per-epoch loss changes: helpful (loss↓), harmful (loss↑), neutral (loss→)
  - Compute `benefit_ratio()` at end of training
  - Log warning if `benefit_ratio < 1.0` ("training is net-negative, check LR/data")
  - Integrate into `Trainer::train()` — track metrics alongside loss history
  - Include in `TrainingReport::Display` output
  - Tests: benefit_ratio computation, epoch recording, boundary cases (all helpful, all harmful)

- [ ] **Task 3: GameMetrics for game-specific quality** (`riir-gpu`)
  - Add `GameMetrics` struct to track per-action accuracy and win rate
  - After training, run validation games with trained LoRA to compute:
    - Per-action accuracy (correct action vs model's top-1 prediction)
    - Win rate before vs after training
    - Action distribution entropy (are strategies diverse or degenerate?)
  - Add `validate_game()` function that runs N validation episodes and records metrics
  - Tests: metrics computed correctly, helpful/harmful detection, action accuracy

- [ ] **Task 4: Absorb + Compress post-training analysis** (`riir-gpu`)
  - After training completes, analyze per-target LoRA gradient norms from last N epochs
  - Targets with gradient norm < threshold for last 3 epochs → "stable"
  - Stable targets are candidates for freezing in next training run
  - Output `CompressReport` as part of training report
  - This is Research 14's "compress" operation: stable patterns → freeze recommendation
  - We do NOT auto-freeze — report only, human decides
  - Tests: compress report generation, stability detection, freeze recommendation logic

- [ ] **Task 5: Draft-target distillation** (`riir-gpu`)
  - Implements Research 04 P3 for game domain
  - After training target LoRA, distill a smaller draft LoRA:
    - Forward pass with target LoRA on training data → get action probabilities
    - Train draft LoRA (rank 2) to match target's distribution (KL divergence loss)
    - Draft is smaller → faster inference in microgpt-rs speculative decoding
  - Add `distill_draft()` function to `Trainer`
  - Export both: `game_lora.bin` (target) + `game_draft_lora.bin` (draft)
  - Output `DistillReport` with KL divergence, ranks, sample count
  - This is the game-domain analog of Leviathan verification (Plan 004)
  - Tests: draft distribution approximates target (KL < threshold), draft is smaller rank

- [ ] **Task 6: Screening-guided LoRA target probe** (`riir-gpu`)
  - Before full training, do a short probe (10 steps) with LoRA on all targets
  - Measure per-target gradient magnitude during probe
  - Rank targets by gradient magnitude (high gradient = high learning signal)
  - Select top-K targets for full training (skip low-signal targets)
  - Research 07 applied to training: relevance = gradient signal → prune → focus compute
  - Config: `--lora-top-k 3` (only train top 3 targets)
  - Default: all targets (backward compatible)
  - Tests: probe produces gradient ranking, top-K selection is correct, full training still works

- [ ] **Task 7: GameTrainingReport and JSON export** (`riir-gpu`)
  - Consolidate `TrainingReport`, `ReviewMetrics`, `GameMetrics`, `CompressReport`, `DistillReport` into `GameTrainingReport`
  - Implement `Serialize` for JSON export to `output/training_report.json`
  - This enables cross-run comparison and tracking training quality over time
  - Tests: JSON roundtrip, all fields present, Display output formatted

- [ ] **Task 8: `game_domain` feature flag in microgpt-rs** (`microgpt-rs`)
  - Add `game_domain` feature flag to `Cargo.toml`
  - Feature enables: `domain_latent` (already exists), game-specific LoRA loading, draft LoRA loading
  - `language_domain` feature flag added as placeholder (no code yet, just the flag)
  - `game_domain` implies `domain_latent` feature
  - `full` feature includes both `game_domain` and `language_domain`
  - Ensure existing tests pass with and without `game_domain`
  - Tests: feature flag compiles, correct features implied, no regression

- [ ] **Task 9: Benchmarks**
  - Benchmark: `BetaConfig` training (β=0.3) vs manual config (same derived params) — should be identical
  - Benchmark: Draft distillation quality (KL divergence over epochs)
  - Benchmark: Screening-guided top-3 vs all targets (training time + final loss)
  - Benchmark: Training with and without domain latent
  - All benchmarks use `std::time::Instant` like existing benchmarks
  - Output to `bench/040_*` files

---

## File Change Summary

| File | Change | Target |
|------|--------|--------|
| `riir-gpu/src/training_config.rs` | New: `BetaConfig`, `ReviewMetrics`, `GameMetrics`, `CompressReport`, `DistillReport`, `GameTrainingReport` | riir-gpu |
| `riir-gpu/src/lib.rs` | Export new types | riir-gpu |
| `riir-gpu/src/training_loop.rs` | Integrate `BetaConfig`, `ReviewMetrics`, `GameMetrics`, update `TrainingReport` | riir-gpu |
| `riir-gpu/src/lora.rs` | Add gradient norm tracking for screening probe | riir-gpu |
| `riir-gpu/examples/train_bomber.rs` | `--beta` flag, `--lora-top-k` flag, distillation, validation, JSON report | riir-gpu |
| `microgpt-rs/Cargo.toml` | Add `game_domain` and `language_domain` feature flags | microgpt-rs |

---

## Design Decisions

### 1. Game Domain Only

`riir-burner` (both backends) is unstable future work:
- **Python (unsloth-mlx)**: external dependency, cannot modify
- **Rust (burn)**: experimental, needs Metal backend build, not production-ready
- When `riir-burner` stabilizes, language-domain tasks get their own plan

### 2. β Parameterization (Research 16)

One scalar controls all training hyperparameters. Prevents inconsistent combinations.

| Parameter | β=0.0 (fast) | β=0.3 (default) | β=0.7 (production) | β=1.0 (thorough) |
|-----------|-------------|-----------------|-------------------|-----------------|
| learning_rate | 1e-4 | 3.7e-4 | 7.3e-4 | 1e-3 |
| lora_rank | 4 | 12 | 24 | 32 |
| warmup_steps | 50 | 185 | 365 | 500 |
| weight_decay | 0.01 | 0.007 | 0.004 | 0.001 |
| epochs | 1 | 3 | 7 | 10 |
| draft_rank | 2 | 3 | 6 | 8 |

### 3. ReviewMetrics over Raw Loss (Research 15)

Loss alone doesn't tell you if training is helping or hurting. `benefit_ratio < 1.0` means
something is wrong (bad data, bad LR, bad architecture). The paper's insight: measure before trusting.

### 4. Absorb + Compress is Report-Only (Research 14)

We do NOT auto-freeze targets. `CompressReport` recommends which targets are stable.
The human decides whether to freeze for the next run. Avoids premature convergence.

### 5. Draft Distillation Follows Target (Research 04)

Train target LoRA first on high-quality data. Then distill draft from target's distribution.
Draft is smaller (rank 2) for faster speculative decoding. This is the game-domain path
to making Plan 004 (Leviathan verification) viable.

### 6. Feature Flags Separate Domains

`game_domain` = small models, 6-action vocab, wgpu-trained LoRA, deterministic validators.
`language_domain` = LLM models, BPE vocab, burn/unsloth-trained LoRA (future).
Different scales, different pipelines, different feature flags.

---

## Priority Order

| Priority | Task | Why | Effort |
|----------|------|-----|--------|
| **P0** | Task 1: BetaConfig | Reduces config complexity, enables easy sweep | Small |
| **P0** | Task 2: ReviewMetrics | Training quality visibility, prevents wasted runs | Small |
| **P1** | Task 3: GameMetrics | Game-specific training quality | Small |
| **P1** | Task 8: Feature flags | Separate game from language concerns | Small |
| **P2** | Task 5: Draft distillation | Enables Plan 004 at runtime | Medium |
| **P2** | Task 4: Absorb+Compress | Post-training analysis | Small |
| **P3** | Task 6: Screening probe | Reduces training time by skipping low-signal targets | Medium |
| **P3** | Task 7: JSON report | Cross-run comparison | Small |
| **P4** | Task 9: Benchmarks | Validate all techniques | Small |

---

## Deferred to Future Plans

| Technique | Why Deferred | When |
|-----------|-------------|------|
| DomainLatent for burn (038 T5b) | riir-burner unstable | When burn backend matures |
| Multi-LoRA stacking (04) | Needs stable adapter format | When both backends stabilize |
| Reader/Writer LoRA pairs (025) | Needs runtime support in microgpt-rs | When Plan 025 is implemented |
| EMO data routing (09) | Needs multi-domain training data | When game has multiple sub-domains |
| S-LoRA multi-tenant (04) | Needs GPU infra + real model | Far future |
| Gemma 4 LoRA training (011) | riir-burner Rust backend unstable | When burn backend matures |
| Coding agent in loop (14) | Research-only concept | Never (heuristic learning is the game path) |

---

## Connection to Existing Plans

| Plan | Relationship |
|------|-------------|
| **038** Domain Latent | riir-gpu domain_latent already ✅, burn side deferred |
| **039** Game Replay | Task 5 (distillation) consumes replay data |
| **004** Leviathan | Task 5 produces draft LoRA that makes verification viable |
| **030** Bandit | Future: bandit could dynamically adjust β during training |
| **025** Bidirectional Prefill | Reader/Writer pairs deferred to language domain |
| **010** Burn LoRA | Foundation for language domain — deferred until stable |
| **011** Gemma 4 LoRA | All techniques apply when riir-burner matures |

---

## Expected Outcomes

1. **β parameterization** — `--beta 0.5` replaces 6 hyperparameter flags
2. **Training quality metrics** — `benefit_ratio > 2.0` confirms training is net-positive
3. **Game quality metrics** — win rate + per-action accuracy track game skill
4. **Compress report** — identifies stable LoRA targets for freezing
5. **Draft distillation** — small game LoRA for speculative decoding
6. **Feature flags** — clean separation of game vs language domain
7. **JSON training reports** — cross-run comparison and tracking

---

## Research Citations

```bibtex
@misc{weng2026learning_beyond_gradients,
  title = {Learning Beyond Gradients},
  author = {Weng, Jiayi},
  year = {2026},
  howpublished = {\url{https://trinkle23897.github.io/learning-beyond-gradients/}}
}

@article{zheng2026autotts,
  title  = {LLMs Improving LLMs: Agentic Discovery for Test-Time Scaling},
  author = {Zheng, Tong and others},
  journal = {arXiv preprint},
  year    = {2026}
}

@misc{ta2026reinforced_agent,
  title = {Reinforced Agent: Inference-Time Feedback for Tool-Calling Agents},
  author = {Ta, Anh and Zhu, Junjie and Shayandeh, Shahin},
  year = {2026},
  eprint = {2604.27233}
}

% LoRA Architecture Verdict (internal research 04)
% Screening Absolute Relevance (internal research 07)