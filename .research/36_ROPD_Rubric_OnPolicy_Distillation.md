# Research 36: ROPD — Rubric-based On-policy Distillation

> **Paper:** Rubric-based On-policy Distillation (Fang et al., 2026)
> **Source:** `.raw/ROPD_official/` — full codebase audited: `algo/`, `prompts/`, `verl/`, `training/`
> **Date:** 2025-06-01
> **Related Plans:** Plan 071 (microgpt-rs, modelless), Plan 072 (riir-ai, model-based)

## Executive Summary

ROPD reframes knowledge distillation from **token-level imitation** (requires teacher logits) to **rubric-based semantic scoring** (requires only teacher text). Instead of matching token distributions, it induces structured scoring criteria from teacher-student contrasts, then uses weighted pass-rates as GRPO rewards.

**Why we care:** Our system already has GRPO, DPO, bandit-based modelless distillation, and the `Validator` trait. ROPD's contribution is the **Rubricator → Verifier → Reward** pipeline, which produces interpretable, multi-criteria reward signals. This can enhance both our modelless path (template rubrics + WASM verifiers) and model-based path (LLM rubricator + GRPO).

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

**Rubric scoring (per rollout i):**
```
s_i = Σ(w_k * v_{i,k}) / (Σ w_k + ε)    where v_{i,k} ∈ {0,1}
```

**Reward (from code, not paper):**
```
reward = (student_score - teacher_score) / reward_scale
```

This is important — ROPD uses **relative** reward (student vs teacher on same rubric), not absolute.

### Why Rubrics Beat Logits

| Property | Logit-based OPD | Rubric-based OPD |
|----------|----------------|-----------------|
| Teacher access | White-box only | Black-box OK |
| Cross-architecture | Needs aligned tokenizers | No tokenizer needed |
| Sample efficiency | Token-level noise | Semantic filtering (up to 10x) |
| Interpretability | Opaque logits | Named criteria + scores |
| Reward decomposability | Single KL divergence | Per-criterion pass/fail |

---

## Codebase Audit

### Architecture (from `.raw/ROPD_official/`)

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

1. **Shuffled scoring** — teacher and student answers are shuffled before verification to prevent positional bias. `_build_shuffled_answer_items()` assigns random keys, scores are restored via `_restore_scores_by_source()`.

2. **Relative reward** — `reward = (student_score - teacher_score) / reward_scale`. Not absolute score. This means the rubric can be strict without breaking the reward signal.

3. **Offline teacher index** — teacher responses are pre-computed and cached in JSONL, keyed by `(uid, prompt_hash)`. Decouples teacher inference from training loop. Uses SHA-256 fingerprinting for teacher config integrity.

4. **Bounded scheduler** — priority queue with backpressure. Stages: teacher (0) > rubricator (1) > verifier (2). Prevents API overload.

5. **Step retry with rubric caching** — if verification fails, the same rubric is reused (cached by uid + pair_index + teacher + student hashes).

### Rubricator Prompt Analysis (prompts/rubricator.txt)

The rubricator prompt is **heavily engineered** (~200 lines). Key design patterns:

1. **Three category scenarios** (not 1:1 with rubric items):
   - Task Fulfillment and Requirement Compliance
   - Observable Response Quality
   - General Reasoning Quality

2. **Weight semantics** (not uniform):
   - 5 pts: decisive bottleneck (core task failure)
   - 4 pts: strong discriminative bottleneck
   - 2 pts: supporting merit
   - 1 pt: low-risk routine requirement
   - 3 pts: rare intermediate weight (avoid)

3. **Anti-bias rules** (explicit prohibitions):
   - No "uses same method as reference"
   - No "matches reference final answer"
   - No "similar style/wording to reference"
   - No encoding potentially wrong intermediate conclusions
   - No rewarding length/confidence/style

4. **Output format**: JSON with `schema_version: "ropd.rubric.v1"`, 4-12 criteria, each with `criterion_id`, `category`, `criterion` (text), `points`.

### Verifier Prompt Analysis (prompts/verifier.txt)

Simpler (~100 lines). Key patterns:

1. **Binary judgement** per criterion: true/false, no partial credit
2. **Independent scoring**: each answer scored independently, no cross-comparison
3. **Batch scoring**: multiple answers in one call (efficiency)
4. **Output format**: JSON with `schema_version: "ropd.batch_verifier.v2"`, per-answer `judgement` array + `final_score`

### Training Configuration (from README + train.sh)

| Parameter | Value |
|-----------|-------|
| Optimizer | GRPO |
| Learning rate | 1e-6 |
| Batch size | 32 |
| Student rollouts per prompt | 8 |
| Teacher references | 4 |
| Rubric items | 4-12 |
| Training data | DAPO-Math-17K, RaR-Science-20K, RaR-Medical-20K |
| Eval temperature | 1.0, top-p 0.95 |
| Eval samples | 16 per problem |
| Max gen length | 32,768 tokens |

---

## Mapping to Our System

### Component Alignment

| ROPD Component | Our Equivalent | Gap? |
|---------------|----------------|------|
| GRPO optimizer | `GrpoConfig`, `group_advantage()`, `grpo_loss()` | ✅ Complete |
| DPO loss | `GpuDpoLoss`, `LengthNormalizedDpo` | ✅ Complete |
| Rollout grouping | `GZeroLoop` round orchestration | ✅ Complete |
| Proposer | `TemplateProposerAdapter` / `NeuralProposer` | ✅ Complete |
| Rubricator (LLM) | — | ❌ Missing |
| Verifier (LLM) | — | ❌ Missing |
| Teacher index | — | ❌ Missing |
| Bounded scheduler | — | ❌ Missing (but Tokio channels replace) |
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

#### Path B: Model-Based (riir-ai, Plan 072)

Add LLM rubricator/verifier as reward source for existing GRPO:

```
ROPD:  Prompt → Student rollouts → Teacher refs → Rubricator → Verifier → GRPO
Our:   Same flow, using existing GZeroLoop + new RubricReward
```

Components:
- `RubricReward` — new reward source in `loss_grpo.rs`
- `RubricatorClient` — API client wrapping rubricator.txt prompt
- `VerifierClient` — API client wrapping verifier.txt prompt
- `OfflineTeacherIndex` — JSONL cache with blake3 fingerprinting
- Integration into `GZeroLoop` as alternative to HintDelta

### Key Differences: ROPD vs Our G-Zero

| Aspect | G-Zero | ROPD |
|--------|--------|------|
| Reward source | Intrinsic (model's own log-probs) | External (teacher text + LLM judge) |
| Signal type | Scalar δ | Vector (per-criterion scores) |
| Teacher needed | No (self-play) | Yes (reference responses) |
| Cost per reward | ~0 (just forward pass) | ~$0.01-0.10 (2 LLM API calls) |
| Interpretability | Low (scalar, hard to debug) | High (named criteria, weights) |
| Domains | Open-ended (no ground truth) | Any (teacher provides reference) |
| Cross-architecture | N/A (same model) | Core feature (different architectures) |

### What ROPD Gets Right (for us)

1. **Vector rewards** — our scalar δ can't distinguish "correct but incomplete" from "wrong but well-structured". Rubric vectors can.
2. **Relative scoring** — `(student - teacher) / scale` is robust to rubric difficulty variation.
3. **Shuffled blind scoring** — prevents positional bias in LLM judges.
4. **Offline teacher cache** — decouples expensive teacher inference from training.
5. **Anti-bias prompt engineering** — the rubricator prompt's prohibition list is production-grade.

### What ROPD Misses (that we have)

1. **Modelless path** — ROPD requires LLM calls for rubricator + verifier. Our template rubrics + WASM validators can run at inference speed (~µs).
2. **Self-play** — ROPD needs an external teacher. G-Zero's HintDelta is intrinsic.
3. **Bandit infrastructure** — ROPD doesn't adaptively learn which criteria matter. Our bandit does.
4. **Absorb-compress** — ROPD doesn't compress rubrics into hard constraints over time.

---

## Concrete Algorithms for Integration

### Algorithm 1: RubricVector (Modelless)

```text
Input: student_response, reference (golden replay or hint-assisted)
Output: RubricVector { scores: Vec<f32>, weights: Vec<f32> }

For each criterion c_k in domain's rubric template:
  v_k = WASM_validator_k.is_valid(response)  // 0.0 or 1.0
  w_k = criterion_weight_k                    // from template config

Return RubricVector { scores: [v_1..v_K], weights: [w_1..w_K] }
weighted_score = Σ(w_k * v_k) / Σ(w_k)
```

### Algorithm 2: Rubric-Gated Absorb (Modelless)

```text
Input: arm, RubricVector(student), RubricVector(reference)
Output: absorb decision + gap targeting

gap = reference.weighted_score() - student.weighted_score()
if gap > threshold:
  gap_criteria = reference.gap_criteria(student)  // which criteria differ
  inner.absorb(arm, gap)                           // promote to hard constraint
  target_criteria = gap_criteria with highest weight
  // Next episode: bias exploration toward fixing target_criteria
```

### Algorithm 3: ROPD Reward for GRPO (Model-Based)

```text
Input: group of K student rollouts, M teacher references
Output: reward per rollout

1. rubric = rubricator_client.generate(prompt, teacher_refs, student_samples)
2. For each rollout i in [1..K]:
     scores_i = verifier_client.score(rubric, rollout_i)
     s_i = Σ(w_k * v_{i,k}) / Σ(w_k)
3. teacher_scores = verifier_client.score(rubric, teacher_refs)
   s_teacher = mean(teacher_scores)
4. For each rollout i:
     reward_i = (s_i - s_teacher) / max(rubric.maximum_score, 1)
5. Return rewards → group_advantage(rewards, K) → grpo_loss(...)
```

---

## Risk Assessment

| Risk | Modelless | Model-Based |
|------|-----------|-------------|
| Latency overhead | Low (WASM validators are ~µs) | High (2 LLM API calls per group) |
| Quality of rubrics | Medium (template = fixed) | High (LLM-generated, adaptive) |
| Teacher dependency | None (uses replays/hints) | Full (need teacher API) |
| δ-Mem lesson | Plan 053 showed no DDTree gain from vector corrections — rubric vectors may face same issue | N/A (GRPO path is different) |
| Sample efficiency | Unknown — needs benchmark | Paper claims 10x vs logit-based |
| Cross-domain transfer | Templates are domain-specific | LLM rubrics generalize better |

---

## References

- [ROPD: Rubric-based On-policy Distillation](https://arxiv.org/abs/ropd2026) — Fang et al., 2026
- `.raw/ROPD_official/` — Full audited codebase
- Plan 049: G-Zero Self-Play Distillation (our intrinsic reward)
- Plan 052: GFlowNet Modelless Distillation (our flow-based modelless)
- Plan 053: δ-Mem Modelless Distillation (our associative memory — no DDTree gain)
- Plan 059: HLA Distillation Validation (our model-based training loop)
- Research 33: AutoGo Distillation Strategy
```
microgpt-rs/.research/36_ROPD_Rubric_OnPolicy_Distillation.md