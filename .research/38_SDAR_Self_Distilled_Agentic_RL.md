# Research 38: SDAR — Self-Distilled Agentic Reinforcement Learning

> **Paper:** [Self-Distilled Agentic Reinforcement Learning](https://arxiv.org/abs/2605.15155) — Lu et al., 2026 (28 pages)
> **Code:** https://github.com/ZJU-REAL/SDAR
> **Date:** 2025-06-15
> **Related Plans:** Plan 072 (microgpt-rs, modelless SDAR gate), Plan 073 (riir-ai, model-based SDAR loss)
> **Supersedes:** None — extends Plan 071/072 (ROPD) with token-level gated distillation

## Executive Summary

SDAR solves the instability of combining GRPO with On-Policy Self-Distillation (OPSD) for multi-turn agents. The key innovation: a **sigmoid-gated token-level distillation loss** that trusts positive teacher endorsements and softly attenuates negative teacher rejections. This auxiliary loss runs alongside GRPO without touching the RL advantage signal.

**Why we care:** Our system already has GRPO (`loss_grpo.rs`), KL distillation (`distill.rs`), bandit skill retrieval (`bandit.rs`), and the `GZeroLoop` orchestration. SDAR fills a critical gap — our distillation is **uniform** KL per token, which the paper proves **collapses** in multi-turn settings. The sigmoid gate makes OPSD stable and consistently beneficial.

**Key results (Qwen2.5-7B):**
- ALFWorld: GRPO 81.2 → SDAR 85.9 (+4.7)
- WebShop-Acc: GRPO 72.6 → SDAR 82.8 (+10.2)
- Search-QA: GRPO 42.0 → SDAR 49.0 (+7.0)
- Naive GRPO+OPSD: **collapses** on Qwen3-1.7B (32.0 vs GRPO's 46.1)
- Even **random** skill retrieval beats GRPO baseline (+1.9 on ALFWorld)

---

## Paper Core

### Problem

Two observations make multi-turn OPSD problematic:

1. **Multi-turn OPSD Instability** — Once student drifts from teacher-supported trajectory, token-level supervision becomes unreliable. Per-turn KL surges, performance degrades catastrophically (Figure 2 Left).

2. **Asymmetric Trust** — The teacher is the same policy + privileged context (retrieved skills). Negative teacher rejections may indicate skill quality issues, not token errors. Over 50% of tokens have negative teacher-student gap (Figure 3 Left).

### Solution: SDAR

SDAR treats OPSD as a **gated auxiliary objective** while keeping RL as the primary backbone:

1. **Teacher branch**: Same policy πθ + privileged context c+ (retrieved skill)
2. **Gap signal**: `Δt = sg(log πθ(yt|s+_t) - log πθ(yt|st))` — **detached**
3. **Sigmoid gate**: `gt = σ(β · Δt)` ∈ (0,1)
4. **Token loss**: `ℓt = gt · (log πθ(yt|s+_t) - log πθ(yt|st))`
5. **Total loss**: `L = L_GRPO + λ · L_SDAR`

### Key Formulas

**Gap gating (best strategy):**
```
gt = σ(β · Δt)    where Δt = log πT(yt|s+_t) - log πθ(yt|st)
```

**SDAR loss (masked token average):**
```
L_SDAR = Agg(gt · Δt) = (Σ mt · gt · Δt) / (Σ mt)
```

**Total objective:**
```
L(θ) = L_GRPO(θ) + λ_SDAR · L_SDAR(θ)
```

**Hyperparameters (paper-validated):**
- `λ = 0.01` (distillation coefficient — too large overwhelms RL signal)
- `β = 5.0` (sigmoid sharpness — too small = uniform, too large = binary)
- Group size K=8, clip ε=0.2, lr=1e-6

### Skill Retrieval Strategies

Paper tests four retrieval quality tiers:
1. **UCB** — multi-armed bandit over skill library
2. **Keyword Matching** — direct category lookup
3. **Full** — always retrieve relevant skill
4. **Random** — zero task awareness

**Critical finding:** Even random retrieval beats GRPO (+1.9 ALFWorld). Higher quality amplifies gains. Gating filters noise from bad skills.

---

## Detailed Analysis

### Token-Level Gating Strategies (Ablation)

Three gating strategies compared (Figure 6):

| Strategy | Formula | Asymptote | Notes |
|---|---|---|---|
| **Gap** (best) | `gt = σ(β·Δt)` | ~0.84 | Direct teacher-student disagreement signal |
| Entropy | `gt = σ(β·ht)` | ~0.76 | Indirect proxy, activates on already-handled tokens |
| Soft-OR | `gt = σ(β·[1-(1-ht)(1-Δt)])` | ~0.80 | Dilutes selectivity, triggers when either is moderate |

**Gap gating wins** because it precisely measures where teacher disagrees. Entropy erroneously activates on uncertain but well-handled tokens.

### Sharpness β (Figure 7)

| β | Result | Issue |
|---|---|---|
| 0 (no gate) | Collapses | Inherits multi-turn OPSD instability |
| 1 | Suboptimal | Too soft, insufficient selectivity |
| **5 (optimal)** | Best | Balances modulation and selectivity |
| 10 | Suboptimal | Too sharp, binarizes gate, loses partial credit |

### Distillation Coefficient λ (Figure 8)

| λ | Result | Issue |
|---|---|---|
| 0.001 | Insufficient | Too weak to meaningfully aid RL |
| **0.01 (optimal)** | Best | Steady complementary signal |
| 0.1 | Collapses | Distillation dominates, teacher is on average less confident |

### Distillation Objective (Figure 9)

| Objective | Result | Why |
|---|---|---|
| **Reverse KL** (best) | Best | Mode-seeking, down-weights low-teacher-prob tokens |
| JSD | Middle | Symmetric compromise inherits mode-covering tendency |
| Forward KL | Worst | Mode-covering, spreads mass across unreliable guidance |

**Key insight for us:** Our `kl_divergence(p, q)` computes KL(P‖Q). For distillation, we want **reverse KL** KL(π_student ‖ π_teacher) — same direction as SDAR. This is already the correct direction for our `distill_draft()`.

### Robustness to Retrieval Quality (Table 2)

| Retrieval | ALFWorld | WebShop-Score | WebShop-Acc |
|---|---|---|---|
| w/o OPSD (GRPO baseline) | 81.2 | 80.9 | 72.6 |
| Random | 83.1 (+1.9) | 82.5 (+1.6) | 73.6 (+1.0) |
| Full | 83.2 (+2.0) | 87.2 (+6.3) | 78.1 (+5.5) |
| KM | 85.9 (+4.7) | 89.4 (+8.5) | 82.8 (+10.2) |
| UCB | 86.8 (+5.6) | 87.5 (+6.6) | 81.2 (+8.6) |

**Implication:** The gating mechanism itself is the primary value driver, not retrieval fidelity. Our bandit-based retrieval (UCB1) is already at the high end.

---

## Cross-Reference: What We Already Have

| SDAR Component | Our Code | Status |
|---|---|---|
| GRPO clipped surrogate | `riir-gpu/src/loss_grpo.rs` — `grpo_loss()` | ✅ Production |
| Group advantage z-score | `riir-gpu/src/loss_grpo.rs` — `group_advantage()` | ✅ Production |
| KL divergence (reverse direction) | `riir-gpu/src/distill.rs` — `kl_divergence()` | ✅ Production |
| LoRA-only training | `riir-gpu` full stack — wgpu | ✅ Production |
| DPO loss | `riir-gpu/src/loss_dpo.rs` — `GpuDpoLoss` | ✅ Production |
| Multi-arm bandit (UCB) | `microgpt-rs/src/pruners/bandit.rs` | ✅ Production |
| Self-play loop | `riir-gpu/src/gzero_loop.rs` | ✅ Production |
| Sigmoid activation | Various — standard math | ✅ Available |
| **Token-level gap gating** | **MISSING** | ❌ Gap |
| **Detached gate + auxiliary loss** | **MISSING** | ❌ Gap |
| **Teacher forward pass w/ privileged ctx** | **MISSING** | ❌ Gap |

## What's New for Us

### 1. Token-Level Gated Distillation Loss (CRITICAL)

Our `distill_draft()` uses uniform KL per row — no token selectivity. SDAR proves this collapses in multi-turn settings. The fix:

```
gt = σ(β · (log πT(yt|s+_t) - log πθ(yt|st)))    # detached
ℓt = gt · (log πT(yt|s+_t) - log πθ(yt|st))        # gate modulates loss
L_SDAR = Agg(ℓt)                                     # masked token average
```

This is ~50 lines of new code in a new `loss_sdar.rs` module.

### 2. Asymmetric Trust Design Pattern

SDAR formalizes: **trust positive teacher endorsements strongly, attenuate negative rejections conservatively.** This pattern applies beyond distillation:

- Bandit arm scoring: weight positive reward signals more than negative
- AbsorbCompress: gate promotions on positive δ only, not negative
- DPO pair selection: prefer pairs where teacher clearly endorses chosen

### 3. Self-Paced Token Curriculum

The sigmoid gate naturally creates a curriculum:
- Early training: most tokens have negative gap → gate < 0.5 → suppresses distillation
- Late training: more tokens enter positive-gap regime → gate > 0.5 → activates distillation
- No hand-crafted schedule needed

This complements our existing `AbsorbCompress` benefit-ratio gate — same principle at different granularity.

### 4. Reverse KL Validated for Partial Teachers

Our `kl_divergence()` already uses the correct direction. SDAR's ablation confirms this is the right choice when the teacher is "partially weak" (skill-conditioned, not a true oracle).

---

## What's NOT Applicable

| SDAR Aspect | Why Not For Us |
|---|---|
| LLM agent benchmarks | We train game-playing models, not web agents |
| 8×H800 full-model training | LoRA-only on consumer GPU |
| Python training framework | Pure Rust/wgpu stack |
| Qwen model family | Model-agnostic via LoRA |
| Prompt templates | Game domain has different action spaces |

---

## Honest Assessment

### Strengths for Our System
1. **Solves a real problem** — our uniform distillation would likely collapse in multi-turn game play (Bomber GvG, Go self-play)
2. **Small code change** — sigmoid gate + auxiliary loss, ~100 lines total
3. **Validated hyperparameters** — β=5.0, λ=0.01, no guesswork
4. **Robust to noisy skills** — our bandit retrieval has quality variance, gate handles it

### Risks
1. **Domain gap** — paper tests web/text agents, we do games. Token distributions differ.
2. **LoRA-only** — paper trains full weights. Our gradient landscape is different (lower rank).
3. **No game-domain validation** — need our own benchmarks (Bomber, Go, FFT).

### Priority
**HIGH** — fills a proven gap in our distillation stack. Without gating, our multi-turn distillation is likely unstable. The fix is small, well-validated, and fits our existing architecture cleanly.

---

## References

- SDAR paper: https://arxiv.org/abs/2605.15155
- SDAR code: https://github.com/ZJU-REAL/SDAR
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- Related: ROPD (our Research 36), G-Zero (our Research 21), Bandit (our Research 37)