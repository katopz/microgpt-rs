# Research 36: ROPD — Rubric-based On-policy Distillation

> **Paper:** [Rubric-based On-policy Distillation](https://arxiv.org/abs/2605.07396) — Fang et al., 2026 (27 pages)
> **Source:** `.raw/ROPD_official/` — full codebase + paper audited
> **Date:** 2025-06-01
> **Related Plans:** Plan 071 (microgpt-rs, modelless), Plan 072 (riir-ai, model-based)

## Executive Summary

ROPD reframes knowledge distillation from **token-level imitation** (requires teacher logits) to **rubric-based semantic scoring** (requires only teacher text). Instead of matching token distributions, it induces structured scoring criteria from teacher-student contrasts, then uses weighted pass-rates as GRPO rewards.

**Why we care:** Our system already has GRPO, DPO, bandit-based modelless distillation, and the `Validator` trait. ROPD's contribution is the **Rubricator → Verifier → Reward** pipeline, which produces interpretable, multi-criteria reward signals. This can enhance both our modelless path (template rubrics + WASM verifiers) and model-based path (LLM rubricator + GRPO).

**Key result:** ROPD outperforms logit-based OPD despite using strictly less teacher information. Rubric reward AUC = 0.90 vs teacher logit AUC = 0.35 (near random). Up to **10× sample efficiency** gain. Student can even **transcend** the teacher (AIME25 thinking: 68.75 > teacher 67.08).

---

## Paper Core

### Problem

Traditional on-policy distillation (OPD) requires teacher logits (white-box). This locks you out of:
- Proprietary API-based teachers (GPT-4, Claude, Gemini)
- Cross-architecture distillation (different tokenizers, vocabularies)
- Heterogeneous model pairs

### Solution: Rubric-based OPD

For each prompt x:

1. Student generates K on-policy rollouts: `Y^S_x = {y^s_1, ..., y^s_K}`
2. Teacher provides M reference responses: `Y^T_x = {y^t_1, ..., y^t_M}`
3. **Rubricator** induces prompt-specific rubrics by contrasting teacher vs student
4. **Verifier** scores each rollout against rubric (binary pass/fail per criterion)
5. Weighted rubric score = reward for GRPO optimization

### Key Formulas

**Rubric induction:**
```
C_x = {c_k}_{k=1}^{K}  where each c_k = (criterion_text, weight_w_k)
```

**Rubric scoring (per rollout i) — Eq. 4:**
```
s_i = Σ(w_k * v_{i,k}) / (Σ w_k + ε)    where v_{i,k} ∈ {0,1}
```

**GRPO advantage — Eq. 6:**
```
A_i = (r_i - mean(r_j)) / (std(r_j) + ε)
```

**Reward (from code):**
```
reward = (student_score - teacher_score) / reward_scale
```

ROPD uses **relative** reward (student vs teacher on same rubric), not absolute. The rubric is shared across all rollouts in the same group — consistent with GRPO's group-relative advantage.

---

## Main Results (Paper Tables 1–3)

### Black-Box Setting (GPT-5.2 teacher, Qwen3-4B student)

| Method | AIME24 | AIME25 | HMMT25 Nov | GPQA-D | HealthBench | IFEval |
|--------|--------|--------|------------|--------|-------------|--------|
| Student base | 24.17 | 20.83 | 7.08 | 35.66 | 83.32 | 85.21 |
| SFT | 26.69 | 22.50 | 8.33 | — | — | — |
| T-Judge | 62.50 | 56.64 | 38.75 | 36.29 | 84.52 | 84.40 |
| OVD | 61.56 | 55.71 | 37.92 | 35.74 | 83.68 | 84.23 |
| GAD | 27.52 | 23.34 | 14.11 | 36.02 | 83.57 | 85.12 |
| **ROPD** | **65.02** | **58.75** | **41.67** | **36.50** | **84.92** | **85.28** |

ROPD ranks first across all 14 benchmark configurations.

### White-Box Setting (Qwen3-30B-A3B teacher, Qwen3-4B student)

| Method | Access | AIME24 | AIME25 | HMMT25 Feb | HMMT25 Nov | Avg |
|--------|--------|--------|--------|------------|------------|-----|
| LOPD | logit | 47.92 | 38.75 | 20.42 | 24.17 | 32.82 |
| ExOPD | logit | 50.66 | 41.25 | 22.42 | 26.68 | 35.25 |
| **ROPD** | **text** | **63.33** | **55.93** | **25.40** | **38.80** | **45.87** |

ROPD closes **74.1%** of student-teacher gap vs LOPD's 42.1% — a 1.8× improvement with strictly less information.

### Cross-Architecture (Gemma3-4B student, GPT-5.2 teacher)

| Method | AIME24 | AIME25 | HMMT Feb | HMMT Nov | Avg |
|--------|--------|--------|----------|----------|-----|
| Gemma3 base | 6.67 | 12.92 | 1.67 | 6.25 | 6.88 |
| OVD | 7.38 | 13.00 | 2.05 | 6.36 | 7.20 |
| **ROPD** | **10.00** | **13.72** | **2.92** | **6.88** | **8.38** |

Works even when student is very weak (+50% relative improvement on AIME24 from 6.67 base).

---

## Mechanism Analysis (Section 4 — Critical for Our System)

### 4.1 The Informativeness Paradox

**Why do restricted rubric signals surpass dense logit supervision?**

The paper analyzes 3,120 AIME24 rollouts across 13 checkpoints. Key finding:

| Signal | AUC (correctness alignment) | Verdict |
|--------|----------------------------|---------|
| Rubric reward | **0.90** | Excellent |
| Teacher logit | **0.35** | Near random! |
| Top-24 overlap | ~0.50 | Random |

Teacher logit is a **misaligned proxy** — it rewards fluent but logically flawed paths over correct but stylistically novel ones. ROPD's rubric decomposes quality into discrete, verifiable criteria, providing **outcome-oriented** feedback.

**Implication for us:** Our `HintDelta` is a log-prob-based signal (like teacher logit, but intrinsic). It may suffer from the same misalignment — measuring distributional shift, not correctness. Rubric-style structured scoring could be more aligned with actual quality, even if less "dense."

### 4.2 Springboard, Not Mirror — The Phase Shift

Training trajectories reveal a fascinating pattern:

1. **Early training:** ROPD's token overlap with teacher surges FASTER than LOPD — rubrics effectively codify the teacher's basic formatting and linguistic norms
2. **Mid training:** Sharp divergence — ROPD's accuracy and rubric rewards scale synchronously while its logit similarity **declines**
3. **Late training:** LOPD suffers post-saturation degradation; ROPD remains robust

> "ROPD uses the teacher as a springboard, not a mirror. Once the student masters the teacher's reasoning language, it transcends the teacher's specific token distribution to seek higher-order correctness."

**Student transcends teacher:** AIME25 (thinking) — ROPD 68.75 > GPT-5.2 teacher 67.08.

### 4.3 Inter-Dimensional Interference

| Metric | ROPD | LOPD |
|--------|------|------|
| Improvement rate | 50.0% | 29.3% |
| Regression rate | **6.2%** | **15.9%** |
| Net improvement | +48 cells | +17 cells |

LOPD's monolithic scalar signal causes **inter-dimensional interference** — improving one facet (e.g., format) erodes another (e.g., logical coherence). ROPD's per-rubric rewards prevent this by enabling **directional advancement** — the optimizer can penalize specific failures without eroding mastered milestones.

**Implication for modelless:** Our `DeltaGatedAbsorbCompress` uses scalar δ. If δ improves on one dimension but regresses another, we can't detect it. Rubric vectors could enable **per-criterion absorb** — fix specific failures without disturbing working behavior.

### 4.4 Case Study: 3.6× Wider Margin

| | Rubric Score | Scalar Judge Score |
|---|---|---|
| Rollout A (correct reasoning) | 0.77 | 0.70 |
| Rollout C (fabricated derivation) | 0.23 | 0.55 |
| **Margin** | **Δ = 0.54** | **Δ = 0.15** |

Scalar judge is swayed by Rollout C's superficial fluency. Rubric's decoupled dimensions (factorization C3, coherence C4, factual accuracy C5) prevent fabricated derivations from hiding behind well-structured prose.

---

## Ablation Study (Table 6 — Design Priorities)

| Design Choice | AIME24 Pass@1 | Delta |
|---|---|---|
| Full ROPD | **65.02** | baseline |
| w/o blind scoring (verifier sees identity) | 61.75 | −3.25 |
| w/o sharing (per-student rubrics) | 61.25 | −3.75 |
| w/o multi-teacher (m=1) | **47.08** | **−17.94** |

**Critical findings for our plans:**

1. **Multi-teacher is the PRIMARY driver.** Dropping m=4 → m=1 causes catastrophic collapse. A single teacher answer over-anchors the rubric to a specific solution trajectory — criteria collapse into "path-matching" instead of "correctness-checking." **Our modelless path must use multiple references** (replay golden + hint-assisted variants).

2. **Cross-rollout sharing** (+3.75 pts). One shared rubric per prompt > per-pair rubrics. The global view surfaces systematic gaps invisible to isolated pairs.

3. **Blind scoring** (+3.25 pts). But critically: "evaluating students in a vacuum causes the Verifier to collapse toward mean scores regardless of task complexity." **Teacher must remain in the blind pool** for difficulty calibration. Without it, the reward distribution becomes flat.

### Efficiency (Figure 3)

| Metric | ROPD | LOPD |
|--------|------|------|
| Samples to match LOPD best | 1.6k | 15.4k |
| Sample efficiency | **9.6×** | baseline |
| Wall-clock to same threshold | 5.5h | 34.4h |
| Compute efficiency | **6.3×** | baseline |
| Post-saturation degradation | None | Present |

Despite higher per-step overhead (2 LLM calls for rubricator + verifier), ROPD's superior sample efficiency more than compensates.

---

## Prompt Templates (Paper Appendix D)

### Schema Version Discrepancy

The paper's English prompts and the codebase's Chinese prompts use **different schemas**:

| Component | Paper (English) | Codebase (Chinese) |
|-----------|----------------|-------------------|
| Rubricator | `black_opd.rubric.v1` | `ropd.rubric.v1` |
| Verifier | `black_opd.verifier.v1` (single-response) | `ropd.batch_verifier.v2` (batch) |

The codebase evolved past the paper — batch verification is more efficient but adds complexity. **Plan 072 should start with paper's single-response verifier** (simpler), add batch later.

### Rubricator Prompt (English — key structure)

```
[Input]
  Question, m Teacher Responses, n Student Responses

[Core Objective]
  Generate ONE shared rubric with K criteria that:
  - Captures quality dimensions where teachers show strong performance
  - Targets dimensions where students exhibit systematic weaknesses
  - Is applicable to any single response independently

[Three Required Categories]
  1. Task Completion — final answer, format, task contract
  2. Observable Quality — correct steps, valid manipulations, no hallucinations
  3. General Reasoning — coherence, derivation flow, self-checking

[Weight Semantics]
  5 = decisive bottleneck, 4 = strong discriminative, 2 = supporting, 1 = routine
  (3 = rare intermediate, avoid)
  K ∈ [4, 12], dynamically chosen per prompt

[Anti-Bias Rules]
  - Do NOT reward copying teacher wording/style/method
  - Do NOT assume any single teacher is fully correct
  - Do NOT define criteria requiring teacher comparison at verification time

[Multi-Teacher Design Rules]
  - When teachers agree on a quality dimension → higher weight
  - When teachers disagree → accept ANY valid approach
  - Rubrics must NOT collapse into "be more like Teacher #3"

[Output]
  {schema_version, rubrics: [{criterion_id, category, criterion, weight}],
   K, max_weighted_sum, estimated_student_pass_rate}
```

### Verifier Prompt (English — single-response)

```
[Input]
  Question, single Response, Rubric set

[Rules]
  - Binary judgement per criterion: true/false, no partial credit
  - If criterion has multiple conditions → ALL must be met
  - Different but valid method that exhibits same merit → true
  - No extra standards beyond what criterion text requires

[Output]
  {schema_version, judgements: [bool], weighted_score, pass_rate}
```

---

## Codebase Audit (Architecture)

```
algo/ropd/client.py          — Rubricator + Verifier API clients
algo/ropd/prompts.py          — Prompt template rendering
algo/ropd/reward_manager.py   — RopdRewardManager (821 lines, core orchestrator)
algo/ropd_pipeline.py         — BlackOPDPipeline (rollout grouping, pair evaluation)
algo/ropd_scheduler.py        — BoundedRequestScheduler (priority queue: teacher→rubric→verify)
algo/ropd_teacher_index.py    — OfflineTeacherIndex (JSONL cache, SHA-256 fingerprinting)
prompts/rubricator.txt         — Rubric induction prompt (~200 lines, Chinese)
prompts/verifier.txt           — Rubric verification prompt (~100 lines, Chinese)
```

### RopdRewardManager Flow (reward_manager.py)

```
__call__(data) → reward_tensor
  ├── _build_groups(data)                          # Group rollouts by prompt
  ├── _evaluate_initial_groups(groups)
  │     └── _evaluate_group(group)
  │           ├── _generate_teacher_answer()        # Get teacher response(s)
  │           ├── _build_shuffled_answer_items()    # Shuffle teacher + student for blind scoring
  │           ├── rubric_client.generate()          # Induce rubric from contrast
  │           ├── _score_answers_with_step_retry()  # Score all answers against rubric
  │           └── _restore_scores_by_source()       # Map back to teacher/student
  └── _build_reward_control(group_records)          # Final reward tensor
```

Key design decisions from code:

1. **Shuffled blind scoring** — teacher and student answers are shuffled before verification to prevent positional bias. `_build_shuffled_answer_items()` assigns random keys, scores are restored via `_restore_scores_by_source()`.

2. **Relative reward** — `reward = (student_score - teacher_score) / reward_scale`. Not absolute score. This means the rubric can be strict without breaking the reward signal.

3. **Offline teacher index** — teacher responses are pre-computed and cached in JSONL, keyed by `(uid, prompt_hash)`. Decouples teacher inference from training loop. Uses SHA-256 fingerprinting for teacher config integrity. **We should use blake3** (project convention).

4. **Bounded scheduler** — priority queue with backpressure. Stages: teacher (0) > rubricator (1) > verifier (2). Prevents API overload.

5. **Step retry with rubric caching** — if verification fails, the same rubric is reused (cached by uid + pair_index + teacher + student hashes).

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | GRPO (AdamW, β1=0.9, β2=0.95) |
| Learning rate | 1e-6, cosine schedule, 100 warmup |
| Batch size | 32 |
| Student rollouts per prompt (n) | 8 |
| Teacher references (m) | **4** (critical — ablation shows m=1 costs 17.9 pts) |
| Rubric items (K) | 4–12 (dynamically chosen) |
| Rubricator temperature | 0.7 |
| Verifier temperature | 0.0 (deterministic) |
| Weight decay | 0.1, gradient clipping 1.0 |
| Precision | bf16 |
| Hardware | 8×A100-80GB |
| Training data | DAPO-Math-17K, RaR-Science-20K, RaR-Medical-20K |
| Eval | temp 1.0, top-p 0.95, k=16 samples, max 32,768 tokens |

---

## Mapping to Our System

### Component Alignment

| ROPD Component | Our Equivalent | Gap? |
|---|---|---|
| GRPO optimizer | `GrpoConfig`, `group_advantage()`, `grpo_loss()` | ✅ Complete |
| DPO loss | `GpuDpoLoss`, `LengthNormalizedDpo` | ✅ Complete |
| Rollout grouping | `GZeroLoop` round orchestration | ✅ Complete |
| Proposer | `TemplateProposerAdapter` / `NeuralProposer` | ✅ Complete |
| Rubricator (LLM) | — | ❌ Missing |
| Verifier (LLM) | — | ❌ Missing |
| Teacher index | — | ❌ Missing |
| Bounded scheduler | — | ❌ Missing (Tokio channels replace) |
| Delta filtering | `DeltaFilter` (6-stage) | ✅ Complete |
| Constraint validation | `Validator` trait (WASM) | ⚠️ Partial (binary, not rubric) |
| Bandit reward | `DeltaBanditPruner` | ⚠️ Scalar δ, not vector |
| Absorb-compress | `DeltaGatedAbsorbCompress` | ⚠️ Scalar gate, not rubric gate |
| Feedback loop | `send_feedback()` | ✅ Complete |

### Two Integration Paths

#### Path A: Modelless (microgpt-rs, Plan 071)

Replace LLM rubricator/verifier with **template rubrics + WASM validators**:

```
ROPD model-based:   LLM Rubricator → LLM Verifier → reward
Our modelless:       RubricTemplate → WASM Validator → RubricVector reward
```

Key insight: Our `HintDelta` is already a scalar rubric (measures teacher-student gap).
ROPD's contribution is making it a **vector** (per-criterion scores).

Components:
- `RubricTemplate` — fixed criteria per domain (extend `QueryTemplate`)
- `RubricVector` — multi-criteria score (replaces scalar δ)
- `RubricGatedAbsorbCompress<P>` — vector-gated absorb (replaces `DeltaGatedAbsorbCompress<P>`)
- `RubricBanditPruner<P>` — per-criterion bandit arms (replaces `DeltaBanditPruner<P>`)

**Multi-reference is critical** (ablation: m=1 → −17.9 pts). Modelless path must use multiple references:
- Replay golden (from `RegressionSuite`)
- Hint-assisted responses (from existing hint mechanism)
- Alternative winning paths (from `ReplayBackwardWalker`, Plan 052 D4)

#### Path B: Model-Based (riir-ai, Plan 072)

Add LLM rubricator/verifier as reward source for existing GRPO:

```
ROPD:  Prompt → Student rollouts → Teacher refs → Rubricator → Verifier → GRPO
Our:   Same flow, using existing GZeroLoop + new RubricReward
```

Components:
- `RubricReward` — new reward source (implements `RewardSource` trait)
- `RubricatorClient` — API client wrapping rubricator prompt
- `VerifierClient` — API client wrapping verifier prompt (start single-response per paper)
- `OfflineTeacherIndex` — JSONL cache with blake3 fingerprinting
- Integration into `GZeroLoop` as alternative to HintDelta

### Key Differences: ROPD vs Our G-Zero

| Aspect | G-Zero | ROPD |
|--------|--------|------|
| Reward source | Intrinsic (model's own log-probs) | External (teacher text + LLM judge) |
| Signal type | Scalar δ | Vector (per-criterion scores) |
| Teacher needed | No (self-play) | Yes (m=4 reference responses) |
| Cost per reward | ~0 (just forward pass) | ~$0.01–0.10 (2 LLM API calls) |
| Interpretability | Low (scalar, hard to debug) | High (named criteria + weights) |
| Correctness alignment | Unknown (log-prob based, may share logit's 0.35 AUC problem) | Proven (0.90 AUC) |
| Inter-dimensional interference | Possible (scalar signal) | Prevented (per-criterion) |
| Domains | Open-ended (no ground truth) | Any (teacher provides reference) |
| Student transcendence | Not demonstrated | Yes (68.75 > teacher 67.08) |

### What ROPD Gets Right (for us)

1. **Vector rewards** — our scalar δ can't distinguish "correct but incomplete" from "wrong but well-structured". 3.6× wider discrimination margin.
2. **Relative scoring** — `(student - teacher) / scale` is robust to rubric difficulty variation.
3. **Shuffled blind scoring** — prevents positional bias in LLM judges.
4. **Offline teacher cache** — decouples expensive teacher inference from training.
5. **Anti-bias prompt engineering** — the rubricator prompt's prohibition list is production-grade.
6. **Multi-teacher coverage** — m=4 prevents rubric collapse to single-teacher path-matching.
7. **Inter-dimensional isolation** — 6.2% regression rate vs LOPD's 15.9%.

### What ROPD Misses (that we have)

1. **Modelless path** — ROPD requires LLM calls for rubricator + verifier. Our template rubrics + WASM validators can run at inference speed (~µs).
2. **Self-play** — ROPD needs an external teacher. G-Zero's HintDelta is intrinsic.
3. **Bandit infrastructure** — ROPD doesn't adaptively learn which criteria matter. Our bandit does.
4. **Absorb-compress** — ROPD doesn't compress rubrics into hard constraints over time.

### What We Should Worry About

1. **Logit misalignment may apply to HintDelta too.** HintDelta measures log-prob shift — if teacher logit AUC is 0.35, our δ may also be poorly aligned with correctness. This is an open question for benchmarking.
2. **Plan 053 lesson:** δ-Mem's vector corrections showed no DDTree gain — "correction surface too simple." Rubric vectors may face the same issue in game domains where quality is well-captured by scalar reward.
3. **Template rubrics lack adaptivity.** ROPD's rubrics are prompt-specific, dynamically generated. Our modelless templates are fixed per domain. The multi-teacher insight (diverse references prevent collapse) suggests we need diverse templates too.

---

## Concrete Algorithms for Integration

### Algorithm 1: RubricVector (Modelless)

```text
Input: student_response, M references (golden replay + hint-assisted + alternatives)
Output: RubricVector { scores: Vec<f32>, weights: Vec<f32> }

For each criterion c_k in domain's rubric template:
  v_k = WASM_validator_k.is_valid(response)  // 0.0 or 1.0
  w_k = criterion_weight_k                    // from template config

Return RubricVector { scores: [v_1..v_K], weights: [w_1..w_K] }
weighted_score = Σ(w_k * v_k) / Σ(w_k)
```

**Multi-reference requirement (from ablation):** Must score M ≥ 2 references alongside student to prevent rubric collapse. References come from `RegressionSuite` golden + hint-assisted paths.

### Algorithm 2: Rubric-Gated Absorb (Modelless)

```text
Input: arm, RubricVector(student), RubricVector(reference)
Output: absorb decision + gap targeting

gap = reference.weighted_score() - student.weighted_score()
if gap > threshold:
  gap_criteria = reference.gap_criteria(student)  // sorted by weight × gap
  inner.absorb(arm, gap)                           // promote to hard constraint
  target_criteria = gap_criteria[0]                // highest-impact criterion
  // Per-criterion absorb prevents inter-dimensional interference
  // (ROPD's 6.2% regression vs LOPD's 15.9%)
```

### Algorithm 3: ROPD Reward for GRPO (Model-Based)

```text
Input: group of n student rollouts, m teacher references
Output: reward per rollout

1. rubric = rubricator(prompt, teacher_refs, student_samples)
   // Shared rubric across all rollouts (+3.75 pts from sharing ablation)
2. For each rollout i in [1..n]:
     scores_i = verifier(rubric, rollout_i)
     s_i = Σ(w_k * v_{i,k}) / Σ(w_k)
3. Blind score teacher refs alongside students
   // Teacher in blind pool for difficulty calibration (+3.25 pts)
   // Shuffle to prevent positional bias
   s_teacher = mean(teacher_scores)
4. For each rollout i:
     reward_i = (s_i - s_teacher) / max(rubric.max_weighted_sum, 1)
5. advantages = group_advantage(rewards, n)
   // (r_i - μ) / σ within group — no value model needed
6. Return advantages → grpo_loss(advantages, clip_epsilon=0.2)
```

---

## Limitations (from Paper Section 6)

1. **Only tested on formal reasoning** (math, science, medical). Subjective/creative tasks not validated. IFEval results suggest instruction-following is preserved, but not rigorously tested.
2. **Depends on Rubricator/Verifier instruction-following.** Preliminary results show robustness to model replacement (likely because "verifying a solution's integrity is inherently simpler than its derivation"), but needs broader validation.
3. **Not tested for very weak students + very strong teachers** gap (Gemma3 experiments are encouraging but limited).

---

## Risk Assessment

| Risk | Modelless | Model-Based |
|------|-----------|-------------|
| Latency overhead | Low (WASM ~µs) | High (2 LLM API calls per group) |
| Quality of rubrics | Medium (fixed templates) | High (LLM-generated, adaptive) |
| Teacher dependency | None (uses replays/hints) | Full (need teacher API, m≥4 critical) |
| δ-Mem lesson | Plan 053: vector corrections = no DDTree gain | N/A (GRPO path is different) |
| HintDelta misalignment | Our δ may share logit's 0.35 AUC problem — needs benchmark | ROPD's rubric AUC = 0.90, proven |
| Inter-dimensional interference | Scalar δ may cause 15.9% regression like LOPD | Per-criterion scoring prevents |
| Sample efficiency | Unknown — needs benchmark | Paper: 10× vs logit-based |
| Multi-reference requirement | Must provide M ≥ 2 references | m=4 from paper config |

---

## References

- [ROPD: Rubric-based On-policy Distillation](https://arxiv.org/abs/2605.07396) — Fang et al., 2026
- `.raw/ROPD_official/` — Full audited codebase
- Plan 049: G-Zero Self-Play Distillation (our intrinsic reward)
- Plan 052: GFlowNet Modelless Distillation (our flow-based modelless)
- Plan 053: δ-Mem Modelless Distillation (our associative memory — no DDTree gain)
- Plan 059: HLA Distillation Validation (our model-based training loop)
- Research 33: AutoGo Distillation Strategy