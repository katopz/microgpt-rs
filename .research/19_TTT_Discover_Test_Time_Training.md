# Research: TTT-Discover — Test-Time Training for Discovery (19)

> Source: [Learning to Discover at Test Time](https://test-time-training.github.io/discover.pdf) by Mert Yuksekgonul*, Daniel Koceja*, Xinhao Li*, Federico Bianchi* et al. (Stanford · NVIDIA · Astera Institute · UC San Diego · Together AI)
> Date: 2026, distilled 2025-07
> Raw code: `.raw/discover/`

## Summary

TTT-Discover performs **reinforcement learning at test time** on a single problem, so the LLM continues to train with experience specific to that problem. Unlike standard RL (maximize average reward), the discovery goal is to find **one exceptional solution** that beats the state-of-the-art. This reframing yields two key components:

1. **Entropic objective** — favors max-reward trajectories over average-reward ones via exponential tilting `w_β(a) = exp(β·R(a)) / E[exp(β·R)]`, with adaptive β per state via KL budget constraint.
2. **PUCT-based state reuse** — selects which past solution to refine next, using max-child reward (not mean) plus exploration bonus.

Results: new SOTA across mathematics (Erdős' minimum overlap), GPU kernel engineering (TriMul, 2× faster than best human), algorithm design (1st place AtCoder AHC039/058), and biology (single-cell denoising). Cost: ~$500 per problem using open models (gpt-oss-120b).

---

## Core Concepts

### Discovery Problem Formulation

A discovery problem defines:
- **State** `s`: a candidate solution (code, construction, kernel)
- **Action** `a`: thinking tokens + code that transitions to a new state `s' = Parse(a)`
- **Reward** `R(s)`: continuous score (inverse runtime, bound tightness, benchmark score)
- **Discovery**: finding any `s` where `R(s) > R(s_sota)`

This is **not** standard RL:
- Goal is **one** great solution, not good average performance
- Policy is a **means to an end**, not the artifact itself
- No deployment phase — the policy can overfit to this single problem

### Entropic Objective

Instead of expected reward `E[R]`, optimize:

```
J_β(θ) = E_s [ log E_a~πθ [ exp(β(s) · R(s,a)) ] ]
```

The gradient gives importance-weighted policy gradient:
```
w_β(a) = exp(β·R(a)) / E[exp(β·R)]
A_β(a) = w_β(a) - 1   (baseline = 1 since E[w] = 1)
```

As `β → ∞`, this tends to max — exactly what discovery needs. But too-large β early causes instability; too-small later makes advantages vanish.

**Adaptive β**: Set `β(s)` by enforcing `KL(q_β || uniform) = γ` (fixed at `ln(2)`) via bisection search. This auto-tunes per state:
- States with consistent small improvements → larger β (more aggressive)
- States with occasional huge improvements → smaller β (prevent outlier domination)

### PUCT State Reuse

Maintain a buffer `H` of past solutions with rewards. Select next starting point via:

```
score(s) = Q(s) + c · scale · P(s) · √(1+T) / (1+n(s))
```

Key differences from AlphaZero PUCT:
- `Q(s)` = **max** child reward (not mean) — optimistic about best outcome
- `P(s)` = rank-based prior over archived states (not learned policy)
- Visitation counts backprop to all ancestors
- Block entire lineage from current batch for diversity

Buffer management: keep top-1000 states by reward, always retain initial seeds, keep top-2 children per expanded parent.

### Implementation Architecture

```
discover.py
  DiscoverConfig → discover()
    → DatasetBuilder (builds rollout batches)
    → rl/train.py Config → main()
      → do_sync_training (50 epochs)
        → do_group_rollout_and_filter_constant_reward
        → compute_advantages (entropic_adaptive_beta)
        → assemble_training_data
        → optim step (LoRA rank 32, Adam lr=4e-5)
```

Hyperparameters (fixed across all domains):
- Batch: 512 rollouts (8 groups × 64)
- LoRA rank: 32
- KL penalty: 0.01–0.1
- Temperature: 1.0
- Reasoning: high

### Environment Abstraction

```python
class Environment:
    reward_function: BaseRewardEvaluator
    state_type: State
    def get_question(self) -> str: ...

class BaseRewardEvaluator:
    def get_reward(self, code: str, state: State) -> dict:
        # Returns: reward, correctness, raw_score, msg, result_construction, stdout
```

`sandbox_reward_evaluator.py` runs generated code in sandboxes with timeouts (typically 1000s for math, 10s for kernels).

---

## What Maps to Our System

### What Actually Applies

#### 1. Entropic Advantage Estimation (Medium Value, Conceptual)

The entropic objective's insight — **favor the max over the mean** — maps to how we think about DDTree search. Our `extract_best_path_into` already selects the single best path (max, not mean). The entropic weighting concept could inform how we score and rank candidate paths.

However, DDTree operates at token level with a fixed policy (no training during inference). The entropic objective is about **training** the policy, not about search within a fixed policy. So this maps conceptually, not directly.

**Where it could apply**: If we ever do test-time LoRA updates (riir-burner produces LoRA adapters), the entropic objective would be the right one for discovery-style tasks.

#### 2. Solution Buffer + Reuse (Medium Value, Future-Facing)

TTT-Discover maintains a buffer of past solutions with rewards and reuses the best ones. Our system doesn't maintain a solution buffer — each inference is independent. But:

- **riir-burner** produces LoRA adapters from training data — this IS a form of "solution reuse" (past solutions distilled into weights)
- **anyrag** catalog-driven domain shaping IS a form of "state reuse" (past queries configure future ones)
- The **bandit** in Plan 030 learns across episodes — similar to PUCT's visitation counting

**Where it could apply**: A "solution cache" at the anyrag level — store past (query, solution, reward) tuples and use PUCT-like scoring to decide when to reuse vs. regenerate.

#### 3. Adaptive β via KL Budget (Low-Medium Value, Pattern)

The idea of setting a hyperparameter by constraining KL divergence is a clean pattern. We already have `InferenceBudget::from_beta()` in anyrag (from AutoTTS research 16). The TTT-Discover adaptive β is different — it's per-state during RL training, not per-domain at config time.

**Where it could apply**: If we add per-request budget adaptation (not just per-domain), the KL-constrained β search could set `tree_budget` dynamically based on observed reward distribution within a session.

#### 4. Environment Abstraction (High Value, Already Partially Exists)

TTT-Discover's `Environment` + `BaseRewardEvaluator` pattern is essentially what we have:
- `Environment` ↔ anyrag's catalog + domain classifier
- `BaseRewardEvaluator` ↔ `WasmPruner` (ScreeningPruner trait)
- `State` ↔ the current inference context
- `Reward` ↔ `relevance()` score from ScreeningPruner

The key insight from their code: **sandboxed code execution as reward**. Our WasmPruner already does this — it runs WASM in a sandbox to evaluate relevance. The pattern is validated.

#### 5. riir-burner as Test-Time Training Infrastructure (High Value, Direct Fit)

This is the **most direct mapping**. TTT-Discover:
1. Takes a problem description
2. Generates candidate solutions
3. Evaluates them (sandbox reward)
4. Updates the policy via LoRA (rank 32, Adam lr=4e-5)
5. Repeats for 50 steps

Our riir-burner pipeline:
1. Takes a corpus (JSONL)
2. Trains LoRA adapter (rank 32, via unsloth/burn)
3. Packs to binary for riir-ai to load

**Gap**: riir-burner trains offline (curated corpus), not online (generated solutions with reward feedback). But the infrastructure is nearly identical. The bridge would be:
- anyrag generates solutions → evaluates with WasmPruner → exports high-reward JSONL → riir-burner trains LoRA
- This is the **E2E game training pipeline** (Plan 041)

### What Does NOT Map

| TTT-Discover Concept | Why It Doesn't Apply |
|---|---|
| **Full LLM LoRA at test time** | We don't fine-tune the base model per-query. Our LoRA adapters are trained offline by riir-burner. Test-time training of a 120B model costs ~$500/problem — not viable for production inference |
| **512 rollouts per step** | We do single-pass inference. Generating 512 candidates per query is research-grade, not production |
| **Code generation as solution** | Our system generates tokens (code completion, reasoning), not standalone programs. TTT-Discover generates Python/C++ programs that are executed and scored |
| **Majority voting / Best-of-N** | We use `extract_best_path_into` (single best path through DDTree). We don't aggregate across multiple independent solutions |
| **PUCT over solution buffer** | We don't maintain a persistent buffer across queries. Each request is stateless |
| **Thinking tokens + code** | Their actions contain both reasoning (thinking tokens) and executable code. Our tokens are the output, not executable |
| **$500/problem cost** | This is research infrastructure. Production inference must be sub-second, sub-cent |

---

## Comparison: TTT-Discover vs AutoTTS (Research 16) vs Our System

| Aspect | AutoTTS (R16) | TTT-Discover (R19) | Our System |
|---|---|---|---|
| **What adapts** | Search controller (code) | LLM policy (LoRA weights) | Config (domain → budget) |
| **Adaptation mechanism** | Agent rewrites Python | RL gradient updates | TOML config + router |
| **Cost of adaptation** | $39.9, 160 min | ~$500, hours | $0 (config lookup) |
| **Granularity** | Reasoning chain | Full program | Token sequence |
| **Replay/cache** | Frozen traces (0 LLM calls) | Solution buffer | None (stateless) |
| **β parameterization** | Single scalar → all knobs | Adaptive per-state KL | Per-domain config ✅ |
| **Objective** | Accuracy - γ·Cost | max reward (discovery) | Best path score |
| **Already built** | Partial (InferenceBudget) | Not applicable | ✅ DomainConfig |

**Key distinction**: AutoTTS is about **search strategy** (how to allocate compute). TTT-Discover is about **model improvement** (how to make the model better for this problem). We already have the search strategy (DDTree). We don't do model improvement at test time, and shouldn't for production.

---

## Application to Our System

### Direct Mappings

| Paper Concept | Our Equivalent | Status |
|---|---|---|
| **Environment + Reward** | `ScreeningPruner::relevance()` + WasmPruner | ✅ Exists |
| **Solution buffer** | anyrag catalog (past queries shape future) | ✅ Partial |
| **LoRA training** | riir-burner pipeline | ✅ Exists (offline) |
| **β parameterization** | `InferenceBudget::from_beta()` in anyrag | ✅ Implemented (R16) |
| **Best-of-N selection** | `extract_best_path_into()` | ✅ Exists |
| **Per-domain budget** | DomainConfig.inference | ✅ Implemented |
| **Sandbox execution** | WasmPruner (WASM sandbox) | ✅ Exists |
| **Test-time LoRA updates** | — | ❌ Not applicable |
| **PUCT state reuse** | — | ❌ Not applicable |
| **Entropic advantage** | — | ❌ Not applicable (no RL training) |
| **512 rollouts/step** | — | ❌ Not applicable |

### What to Build (Gap Analysis)

#### Priority 1: E2E Feedback Loop (Plan 041 Bridge)

The most valuable insight from TTT-Discover is the **closed loop**: generate → evaluate → train → repeat.

Our architecture already has the pieces:
```
anyrag (generate + classify + evaluate)
  → WasmPruner (reward = relevance score)
  → export high-reward results as JSONL
  → riir-burner (train LoRA)
  → riir-ai (deploy adapter)
```

This is exactly Plan 041 (e2e_game_training_pipeline). TTT-Discover validates this architecture works — they use the same loop at research scale. We need it at production scale (automated, not manual).

**Specific action**: Ensure Plan 041 defines the reward signal format that flows from WasmPruner back to riir-burner training data.

#### Priority 2: Solution Cache in anyrag (New Concept)

TTT-Discover's buffer `H` stores past (state, reward) pairs and reuses the best ones. In production, this becomes:

```rust
// Conceptual: in anyrag
struct SolutionCache {
    entries: Vec<(query_hash: u64, solution: String, reward: f32, timestamp: i64)>,
}

impl SolutionCache {
    /// PUCT-inspired selection: high reward + under-explored
    fn select_for_reuse(&self, query: &str) -> Option<&CachedSolution> {
        // Rank by reward, weighted by recency and diversity
    }
}
```

This is **not** the same as KV cache. This is caching complete solutions for similar queries. The PUCT scoring gives a principled way to decide when to reuse (high reward, diverse) vs. regenerate.

**Estimated scope**: ~200 lines in anyrag, behind a feature flag.

#### Priority 3: Per-Request Budget Adaptation (Future)

Currently, `InferenceBudget` is per-domain (from config). TTT-Discover adapts per-state during training. A middle ground:

```
Domain config provides base budget (β=0.5)
  → Router classifies query → gets base budget
  → If query is similar to past low-reward queries → bump β
  → If query is similar to past high-reward queries → keep β
```

This requires the solution cache (Priority 2) and connects to the bandit (Plan 030).

---

## Key Takeaways

1. **The loop is the product.** TTT-Discover's power comes from the closed loop (generate → evaluate → train). Our E2E pipeline (Plan 041) IS this loop. The paper validates the architecture.

2. **Discovery ≠ production.** TTT-Discover spends $500 and hours per problem. We need sub-second, sub-cent responses. The ideas transfer as **architecture patterns**, not as runtime behavior.

3. **Max over mean is already our philosophy.** `extract_best_path_into` selects the single best path. DDTree search is best-first. We already optimize for the best outcome, not average.

4. **WasmPruner IS the reward evaluator.** The paper's `BaseRewardEvaluator` running in sandboxes is exactly our `WasmPruner` running WASM. This pattern is validated across both TTT-Discover and our production system.

5. **LoRA training is the bridge.** riir-burner trains LoRA adapters offline. If we close the loop (Plan 041), we get a weaker but production-viable version of test-time training: the model improves across sessions, not within a single session.

6. **Don't build PUCT at runtime.** PUCT is for research-scale discovery (50 steps × 512 rollouts). For production, domain config + bandit learning is the right abstraction.

7. **The environment abstraction is clean.** Their `Environment` + `BaseRewardEvaluator` is worth studying for how we structure anyrag's domain system. The separation of "question generation" from "reward evaluation" maps to our separation of "classification" from "screening."

8. **Complementary to Heuristic Learning (R14).** Research 14 showed coding agents can evolve programmatic heuristics (code edits, no gradients) that rival Deep RL. TTT-Discover shows RL gradient updates on LoRA weights can discover SOTA solutions. These are two ends of a spectrum:
   - **R14 (Heuristic Learning)**: Code → Agent edits → Better code. Fast iteration, no GPU training. Our Bomberman HL Arena (Plan 033) uses this.
   - **R19 (TTT-Discover)**: Code → RL gradient → Better weights → Better code. Slow iteration, requires GPU training. Our riir-burner pipeline is the infrastructure.
   - **Our sweet spot**: Use R14-style heuristic learning for game logic (Bomberman, Monopoly) where the policy is code. Use R19-style LoRA training for language tasks (code completion, reasoning) where the policy is neural weights. Both share the same loop: generate → evaluate → improve.

---

## Citation

```bibtex
@article{yuksekgonul2026tttdiscover,
  title   = {Learning to Discover at Test Time},
  author  = {Yuksekgonul, Mert and Koceja, Daniel and Li, Xinhao and
             Bianchi, Federico and McCaleb, Jed and Wang, Xiaolong and
             Kautz, Jan and Choi, Yejin and Zou, James and Guestrin, Carlos and Sun, Yu},
  journal = {arXiv preprint arXiv:2601.16175},
  year    = {2026}
}