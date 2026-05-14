# MicroGPT-RS

Speculative Decoding with DFlash & DDTree — a high-performance Rust implementation of a micro-Transformer with built-in benchmarking and visualization.

Inspired by [microgpt-c](https://github.com/nicholasgasior/microgpt-c), [talos-vs-macbook](https://github.com/alexcb123/talos-vs-macbook), and [Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub/).

## 🚀 Key Features

- **Real Transformer Inference** — Full GPT forward pass with RMSNorm, multi-head causal attention, ReLU MLP, KV cache, and temperature sampling.
- **Zero-Alloc Forward Pass** — Pre-allocated `ForwardContext` buffers eliminate heap allocations per inference step.
- **DDTree (Dynamic Draft Tree)** — Best-First Search using a `BinaryHeap` to build a candidate token tree from marginal log-probabilities.
- **ConstraintPruner** — Pluggable trait for neuro-symbolic intercept: deterministic rules engine prunes invalid branches before target verification.
- **ScreeningPruner** — Upgraded binary pruning to graded relevance (`R ∈ [0.0, 1.0]`) with blended score formula.
- **SpeculativeVerifier** — Swappable verification via trait: `SimulatedVerifier` (fast) or `LeviathanVerifier` (real p/q rejection sampling).
- **Raven RSM** — O(1) KV cache replacement with sparse Top-K routing. Unselected slots completely frozen.
- **Percepta** — O(log N) 2D convex hull attention with ternary search. Proves LLMs can execute programs internally.
- **Sparse MLP** — Unstructured sparsity acceleration, skipping dead neurons in ReLU activations.
- **BPE Tokenizer** — Train/encode/decode with Config::bpe() preset for code generation.
- **Multi-Armed Bandit** — Adaptive `ScreeningPruner` with UCB1, ε-greedy, Thompson Sampling strategies.
- **Heuristic Learning** — TrialLog, AbsorbCompress, HotSwapPruner, RegressionSuite, ReviewMetrics for policy evolution.
- **Bomberman Arena** — 4-player HL proof: adaptive intelligence (+177) > greedy (+131) > static rules (-30) > random (-55).
- **Monopoly FSM Arena** — 4-player turn-based FSM: sequential phase AI (PreTurn→Rolling→Resolving→Strategic→EndTurn) with bandit strategy adaptation across 1000 games.
- **Bandit + WASM Pruners** — `BanditPruner` wraps any `ScreeningPruner` with exploration. `WasmPruner` loads sandboxed `.wasm` validators.
- **TurboQuant KV Cache** — 5-8× KV cache compression via random rotation + Lloyd-Max quantization (2-4 bit). 3-bit: 0.99 attention correlation, 0.98 cosine similarity.
- **PFlash Block-Sparse Prefill** — Block-sparse speculative prefill with sink/window/alpha selection rules. Up to 21× sequence reduction with 100% NIAH needle retrieval.
- **G-Zero Self-Play** — Verifier-free Hint-δ intrinsic reward makes modelless HL smarter (δ-gated AbsorbCompress + δ-reward BanditPruner), then optionally adds model-based self-play (GRPO Proposer + length-normalized DPO Generator). No external LLM judge needed.

📖 **Deep dives:** See [`.docs/`](.docs/) for architecture, speculative decoding, performance, sudoku, validator, HL, bomber arena, and monopoly FSM details.

## 🏗️ Architecture

Matching the talos-vs-macbook reference model:

| Parameter | Value |
|-----------|-------|
| `vocab_size` | 27 (a–z + BOS) |
| `block_size` | 16 |
| `n_embd` | 16 |
| `n_head` | 4 |
| `mlp_hidden` | 64 (4×) |
| `n_layer` | 1 |
| `temperature` | 0.5 |

### Core Pipeline

```
LLM drafts logits → ConstraintPruner filters invalid → DDTree builds valid-only tree → Target verifies
```

### Key Traits

```rust
pub trait ConstraintPruner: Send + Sync {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;
}

pub trait ScreeningPruner: Send + Sync {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32;
}

pub trait SpeculativeVerifier: Send + Sync {
    fn speculate(&mut self, draft_weights, draft_config, token, pos, rng) -> Vec<usize>;
}
```

### Routing & Conditioning

- **Prompt Router** — `KeywordRouter` scores prompt against domain keywords, `ExpertRegistry` selects `ScreeningPruner` + LoRA.
- **Embedding Router** — Three-tier fallback: embedding search → domain classify → keyword (local).
- **Bidirectional Prefill** — Prompt tokens attend to ALL other prompt tokens (no causal mask during prefill).
- **Modality LoRA Switching** — `reader_lora` active during prefill, `writer_lora` active during decode. Reference swap, zero data movement.
- **PPoT** — Logit-parameterized CPU resampling on failure. Zero overhead on success path.

📖 See [`.docs/02_architecture.md`](.docs/02_architecture.md) for full details.

### Early Exit & Dynamic Budget (Plan 026)

- **`Config::with_overrides()`** — Apply per-domain inference budget from TOML. `None` fields unchanged, `Some` fields override.
- **`early_exit_patience`** / **`early_exit_gap`** — Confidence-gap early exit in DDTree Phase C. When the best path dominates for `patience` consecutive iterations with a score gap > `gap`, expansion stops early.
- **`InferenceOverrides`** DTO — Plain struct (no serde) for dependency-free budget injection.
- **Default**: `early_exit_patience = 0`, `early_exit_gap = 0.0` — zero behavioral change.

## 🧠 Deterministic Validator

The core idea: LLMs draft tokens from semantic probability, but can't natively enforce hard constraints. A deterministic rules engine sits between draft and verification:

```
LLM drafts logits → SynPruner filters invalid Rust syntax → DDTree builds valid-only tree → Target verifies
```

**Proven with Sudoku** — Path-aware `ConstraintPruner` catches 100% of invalid branches:

```
Unpruned:    100 nodes,  46 accumulated-valid (46.0%)
Static-Only: 100 nodes,  84 accumulated-valid (84.0%)
Path-Aware:  100 nodes, 100 accumulated-valid (100.0%)
```

**Arto Inkala "World's Hardest Sudoku"**: 49,559 steps, 7 hull vertices, 7,079.9× compression.

📖 See [`.docs/05_sudoku.md`](.docs/05_sudoku.md) and [`.docs/06_validator.md`](.docs/06_validator.md).

## 📊 Benchmark Results

Run on Apple Silicon (single-threaded, `--release`, 50k iterations, **zero-alloc hot paths**).

**Models:** Target (embd=16, heads=4, mlp=64) · Draft (embd=4, heads=2, mlp=16) · Run `047`

```
Method                         Throughput         μs/step  Avg Accept Len
───────────────────────────────────────────────────────────────────────────────
Transformer AR                    900,464 tok/s       1.11            1.00
DFlash                           4,231,267 tok/s       1.89            8.00
DDTree Build                      430,911 trees/s      2.32            —
Speculative (Simulated)          1,143,669 tok/s       4.37            5.00
Speculative (AR Draft)           1,643,545 tok/s       4.26            7.00
Leviathan (Algorithm 1)           114,387 tok/s      10.31            1.18
Leviathan (w/ rollback)           206,605 tok/s       5.69            1.18
Spec (conditioned)               1,157,438 tok/s       5.83            6.74
Prefill (no compress)           19,425,142 tok/s       3.29           64.00
Prefill (compressed)             1,962,114 tok/s       3.57            7.00
DDTree (chain-seed)                447,251 trees/s      2.24           16.00
DDTree (screened R=1.0)            338,390 trees/s      2.96           16.00
forward_raven (16 slots)         1,617,183 trees/s      0.62            —
raven_recall (1000 noise)        9,252,063 tok/s       0.11           63.21
───────────────────────────────────────────────────────────────────────────────
📈 Best speedup: 1.82x (Speculative AR Draft vs AR)
```

📖 See [`.docs/04_performance.md`](.docs/04_performance.md) for per-benchmark explanations, zero-alloc improvements, and screening overhead analysis.

## 🦅 Raven RSM: O(1) Routing Slot Memory

Fixed-size slot memory with sparse Top-K routing. Unselected slots are **completely frozen** — 10K noise updates leave passkey slots untouched. 2.98× faster than flat attention at pos=8.

| Property | Evidence |
|----------|----------|
| Frozen slots work | 10,000 noise updates, slot 12 identical to 6 decimals |
| O(1) stays flat | Raven stays 1.0× while flat grows 1.1× from pos 16→240 |
| 2.98× faster | 62,653 tok/s (Raven) vs 21,019 tok/s (flat) |

📖 See [`.docs/08_lucebox_techniques.md`](.docs/08_lucebox_techniques.md).

## ⚡ Sparse MLP

CPU sparse vector × dense matrix multiply. Skips dead neurons from ReLU activations (~50% zero by definition, up to 99% with L1 regularization).

```
Dense W2:   output[r] = Σ_{c=0}^{cols-1} W[r,c] × hidden[c]    → always cols multiplications
Sparse W2:  output[r] = Σ_{c ∈ alive} W[r,c] × hidden[c]        → only alive multiplications
```

The Trinity: **Raven** (O(1) memory) + **Screening** (O(1) judgment) + **Sparse MLP** (O(alive) FLOPs).

## 🔬 Percepta: O(log N) 2D Convex Hull Attention

When keys form a convex hull, finding the maximum attention score becomes ternary search → **O(log N)**.

**Proved:** All 4 arithmetic ops (+, −, ×, ÷), power, combined expressions, backtracking search (4×4 Sudoku, 8-Queens, 9×9 Arto Inkala) — all computed via attention-based state retrieval.

**960 arithmetic operations** verified: all a+b, a×b, a−b, a÷b for a,b ∈ 0..=10.

## 🗜️ TurboQuant: Near-Optimal KV Cache Compression

Compresses KV cache from f32 (32 bits) to 2-4 bits per coordinate using random rotation + Lloyd-Max scalar quantization. Based on [TurboQuant (Zandieh et al., 2025)](https://arxiv.org/pdf/2504.19874).

| Metric | Flat f32 | TQ 3-bit | TQ 4-bit |
|--------|----------|----------|----------|
| Bytes/token | 128 | 24 (**5.3×**) | 24 (**5.3×**) |
| 32K ctx memory | 1073.7 MB | 151.0 MB (**7.1×**) | 151.0 MB (**7.1×**) |
| Key cosine sim | 1.0000 | 0.9825 | 0.9958 |
| Attention correlation | 1.0000 | 0.9907 | 0.9978 |
| Output cosine sim | 1.0000 | 0.9989 | 0.9975 |

Architecture: random orthogonal rotation → Beta-distributed coordinates → Lloyd-Max codebook → bit-packed storage. Unbiased attention scores by construction (E[estimated] = true).

**Zero-alloc hot path (Plan 051):** Pre-allocated scratch buffers eliminate all heap allocations from `store_key`/`store_value`/`dequantize_key_into`/`dequantize_value_into`. Full store+dequant cycle **44.6% faster**, per-call dequantize **17-20% faster** at production kv_dim.

📁 `src/turboquant/` — `codebook.rs`, `rotation.rs`, `kv_cache.rs`, `forward.rs`, `types.rs`

## ⚡ PFlash: Block-Sparse Speculative Prefill

Compresses long prompts before target prefill using block-level importance scoring with selection rules (sink + window + last_n_full + alpha threshold). Ported from [lucebox-hub/pflash](https://github.com/Luce-Org/lucebox-hub/) C++/CUDA implementation.

| Metric | Before | After | Gain |
|--------|--------|-------|------|
| 4K ctx tokens | 4096 | 192 | **21.3×** |
| NIAH retrieval | 100% | **100%** (20/20) | preserved |
| block_select throughput | — | ~30M blocks/s | — |
| 128K ctx block_select | — | 140µs | — |

C++ reference: 128K → 2.6K tokens (50× seq reduction), TTFT ~257s → ~24.8s (**10.4×** speedup).

Composable with TurboQuant: TQ compresses the *precision* dimension (fewer bits), PFlash compresses the *sequence* dimension (fewer tokens). Combined: **6.7× total resource reduction**.

📁 `src/speculative/prefill.rs` — `block_select`, `block_select_grid`, `compress_prompt_blocks`, `BlockAttentionScorer`

## 🎰 Multi-Armed Bandit

`ScreeningPruner::relevance()` IS a reward signal. DDTree's best-first search IS exploration. The bandit adds **policy update across episodes**.

| Strategy | Selection | Regret Bound |
|----------|-----------|--------------|
| `Ucb1` | `Q(a) + sqrt(2·ln(N)/n(a))` | O(log N) |
| `EpsilonGreedy` | Explore w/ prob ε | O(√N) with decay |
| `ThompsonSampling` | Sample from Beta(α, β) | O(log N) asymptotic |

**Constrained bandit** — domain `ScreeningPruner` masks invalid arms. `relevance(arm) = 0.0` → bandit score overridden → arm never pulled, even with highest reward.

## 🧠 Heuristic Learning Infrastructure

HL = software systems evolve through **code updates** not weight updates. A coding agent reads feedback and directly edits policies, validators, tests.

```
Episode N:   BanditPruner selects arm → environment runs → reward → TrialLog.append()
Episode N+k: AbsorbCompress promotes stable low-Q arms to hard blocks
Round N+m:   Agent writes new validator.rs → compile .wasm → HotSwapPruner.reload() → RegressionSuite
```

📖 See [`.docs/09_heuristic-learning.md`](.docs/09_heuristic-learning.md).

### Inference-Time Review Metrics

Based on arXiv:2604.27233 — tracks whether reviewer intervention is net-positive via **Helpfulness/Harmfulness** metrics and a **benefit-to-risk ratio** (paper found 3.1:1 for o3-mini). Gates `AbsorbCompress` when ratio drops below threshold.

| Ratio | Interpretation |
|:-----:|:---------------|
| > 3.0 | Excellent reviewer (paper quality) |
| 2.0–3.0 | Acceptable (default threshold) |
| < 1.0 | Net-negative — stop reviewing |

Run: `cargo run --example review_01_metrics --features bandit`

## 🎮 Bomberman HL Arena — ✅ HL Thesis Proven

4-player Bomberman arena with `bevy_ecs` standalone. **Result: HL (+177) > Greedy (+131) > Validator (-30) > Random (-55)**.

| Player | Tech | Score | Wins |
|--------|------|-------|------|
| **HL** 🐵 | Opponent tracking + strategy + bandit | **+177** | **8** |
| Greedy 🐱 | Heuristic + 20% safe exploration | +131 | 5 |
| Validator 🐶 | Static safety rules | -30 | 1 |
| Random 🐰 | Blast-zone avoidance only | -55 | 9 |

📖 See [`.docs/10_bomber_arena.md`](.docs/10_bomber_arena.md).

## 🎲 Monopoly FSM Arena

4-player Monopoly with `bevy_ecs` standalone. Turn-based event-driven FSM with 8 phases, 40-square board, and 4 AI tiers.

| Player | Tech | Strategy |
|--------|------|----------|
| **HL** 🧠 | Bandit + opponent modeling + phase adaptation | Adaptive (Development preferred, Q=0.71) |
| Greedy 💰 | Heuristic scoring + set-completing trades | Aggressive acquisition + building |
| Validator 🛡️ | Safety rules ($200 reserve, no opponent monopolies) | Strategic buys + efficient building |
| Random 🎲 | Square-parity pseudo-random | Baseline |

**1000-game proof:** HL 56.5% win rate, 93.7% survival, +41.3pp over Validator. ✅ HL Thesis PROVEN (threshold: ≥5pp). Bandit explores all 5 strategies. Performance: 84.5 games/sec, 41µs/turn (24.4× under target).

4 examples (headless arena, TUI replay, 1000-game proof, benchmark).

📖 See [`.docs/11_monopoly_fsm.md`](.docs/11_monopoly_fsm.md).

## ⚔️ FFT Tactics Arena — TFT Party AI

Final Fantasy Tactics-inspired 4v4 ATB (Active Time Battle) arena with status effects, 6 classes, and 5 AI strategies. **TFT (Tit-for-Tat) dominates with 99% win rate** — game theory's optimal strategy applied to MMORPG party combat.

| Player | Tech | Win% | Survival | Kills/rnd |
|--------|------|------|----------|-----------|
| **TFT** 🦊 | Provocation FSM + role-based response | **99.0** | **95.7%** | **1.10** |
| HL 🐵 | Bandit Q-learning over 9 action types | 91.5 | 85.9% | 0.88 |
| Greedy 🐱 | Weakest-target + heal + potion | 56.1 | 35.7% | 0.83 |
| GZero 🤖 | Template hints + δ bandit + heuristics | 15.8 | 61.9% | 0.16 |
| Validator 🐶 | Safety-first + debuff cure + retreat | — | — | — |

**TFT game theory:** Nice (role default) → Retaliatory (on provoke from `GameEvent::DamageDealt`) → Forgiving (10% generous TFT + 5-tick timer). Each class retaliates differently: Knight intercepts, WhiteMage heals first then attacks, BlackMage bursts.

**GvG Round-Robin** (250 rounds × 6 matchups): TFT 92.5% > HL 73.0% > Greedy 61.6%. Nash analysis confirms TFT is a dominant strategy.

3 examples (arena, GvG tournament, A/B benchmark).
📖 See [`.docs/09_heuristic-learning.md`](.docs/09_heuristic-learning.md) for full benchmark results.

## 🔄 Self-Improving Loop (Plan 048)

The system closes the feedback → retrain → hot-swap cycle for continuous improvement:

```text
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌───────────┐
│  Inference   │────▸│  anyrag Cache     │────▸│  LoRA Retrain │────▸│  Hot-Swap  │
│  + Feedback  │     │  episodic memory  │     │  (wgpu GPU)   │     │  zero-downtime │
└─────────────┘     └──────────────────┘     └──────────────┘     └───────────┘
```

- **FeedbackConsumer** polls anyrag episodic cache for new feedback samples
- **Retrain** triggers LoRA fine-tuning on accumulated samples via wgpu GPU pipeline
- **Hot-Swap** signals inference layer to swap adapters without downtime
- Feature-gated: `cargo build -p riir-gpu --features feedback-consumer`

See [riir-ai `.docs/13_research_audit_results.md`](../riir-ai/.docs/13_research_audit_results.md) for the full research audit.

## 🎯 G-Zero: Verifier-Free Self-Play (Plan 049)

Distilled from [G-Zero: Self-Play for Open-Ended Generation from Zero Data](https://arxiv.org/pdf/2605.09959) (Huang et al., 2026). Makes our existing **modelless HL smarter** with the Hint-δ signal, then optionally adds gradient-based self-play on top.

### Core Innovation: Hint-δ

An intrinsic reward measuring how much a hint shifts the Generator's predictive distribution — **no external verifier or LLM judge needed**:

```text
δ(q, h, a_hard) = (1/T) Σ [log πG(at | q, h, a<t) − log πG(at | q, a<t)]
```

δ is large only when the query is challenging AND the hint carries information the Generator lacks. Two objectives in one scalar — and it's architecture-agnostic.

### Two Phases: Modelless First, Model-Based Second

| Phase | Mechanism | Updates | Cost | Strength |
|-------|-----------|---------|------|----------|
| **Phase 1 (Modelless)** | δ → `AbsorbCompress` + `BanditPruner` | Heuristics/rules | Low | Safe, fast, proven HL loop |
| **Phase 2 (Model-Based)** | δ → GRPO + DPO | LoRA weights | High | Stronger for open-ended domains |

Phase 1 makes the existing modelless path **smarter** — δ is a denser, more informative reward than raw environment feedback. Phase 2 adds neural self-play only when needed.

### Phase 1: Smarter Modelless (T1–T5)

```text
TemplateProposer ──(query, hint)──▸ Generator (frozen, inference only)
       │                                    │
       │                             log-probs with/without hint
       │                                    │
       │                               HintDelta
       │                                    │
       │                    ┌───────────────┴──────────────┐
       │                    ▼                              ▼
       │          DeltaGatedAbsorbCompress      DeltaBanditPruner
       │          (promote high-δ arms          (δ as dense reward
       │           to hard constraints)          for arm selection)
       │                    │                              │
       │                    └──────────┬───────────────────┘
       │                               ▼
       │                     TrialLog (JSONL)
       │                               │
       └─── next episode ◂─────────────┘
```

**No gradient updates.** The model generates log-probs for inference only. All learning happens through heuristic promotion and bandit Q-values, same as existing HL — but with a better reward signal.

| New Component | What | Why Smarter |
|---------------|------|-------------|
| `HintDelta` | Log-prob shift computation | Shared foundation for both phases |
| `DeltaGatedAbsorbCompress` | Absorb only when δ reveals blind spot | Promotes heuristics the model doesn't already know |
| `DeltaBanditPruner` | δ as dense reward for arm selection | No need to wait for episode completion |
| `TemplateProposer` | Rule-based query-hint generation | 0 GPU cost, targets blind spots from bandit history |

### Phase 2: Model-Based Self-Play (T6–T9)

Builds on Phase 1's δ computation — adds gradient-based training via GRPO (Proposer) and length-normalized DPO (Generator):

```text
Phase 2a — Proposer Training (GRPO):
  NeuralProposer πP generates {(qi, hi)} → Generator answers unassisted
  → δ reward + length/BLEU penalties → GRPO gradient update

Phase 2b — Generator Training (Length-Normalized DPO):
  Frozen πP generates query-hints → Generator answers with/without hint
  → lower-half δ filter → DPO update (hint-assisted=chosen, unassisted=rejected)
  → HotSwapPruner reloads adapter (zero-downtime)
```

### Three Training Paths

```text
SelfImprovingCycle {
  Collecting → ReadyToSynthesize → ...
    ├── Path A (existing):  Export JSONL → riir-burner LoRA SFT          (modelless HL)
    ├── Path B (Phase 1):   δ → DeltaGatedAbsorbCompress + DeltaBanditPruner (smarter modelless)
    └── Path C (Phase 2):   Proposer↔Generator self-play → DPO LoRA      (model-based G-Zero)
}
```

Path A → B is **incremental** (same architecture, better signal). Path B → C is **opt-in** (add gradient training when modelless plateaus). All three feed into `HotSwapPruner`.

### Key Design Decisions (from paper)

| Decision | Rationale |
|----------|-----------|
| **Modelless first** | δ is architecture-agnostic — use it without DPO/GRPO before adding complexity |
| Lower-half δ filter `[0, 50th %ile]` | Low-δ = hard-to-distinguish pairs = fine-grained DPO signal; high-δ = answer leakage |
| Length-normalized DPO | Neutralizes vanilla DPO's length bias via per-token mean log-ratio |
| Length penalty `λ·max(0, |h|-200)/100` | Prevents verbose hint reward hacking |
| BLEU duplication penalty `|Ci|/|B|` | Prevents Proposer collapse into repetitive pairs |

### Critical Finding

>70% of DPO training pool is **non-verifiable tasks** (advice, writing, explanation), yet reasoning **transfers** to verifiable math domains. Structural depth is internalized, not memorized.

| Model | Chat (AlpLC) | IFEval-pS | AIME25 | Average |
|-------|-------------|-----------|--------|---------|
| Qwen3-8B base → G-Zero R2 | 8.47 | 43.81 | **12.40** | **35.43** (+1.48) |
| Llama-3.1-8B → G-Zero R2 | **27.86** | 59.52 | 0.63 | **43.90** (+1.13) |

📖 See [`.plans/049_g_zero_self_play.md`](.plans/049_g_zero_self_play.md) for full implementation plan, types, hyperparameters, and risk assessment.

## 🌊 GFlowNet Modelless Distillation (Plan 052)

Distills the GFlowNet shortest-path theorem — **minimize flow = shortest paths** — into the existing ScreeningPruner + BanditPruner + DDTree stack **without any neural network training**.

**Core insight:** The paper proves that minimizing expected trajectory length `E[nτ]` forces the backward policy `P_B` to assign zero probability to all non-shortest paths. Our stack already computes forward marginals (LoRA logits = P_F), backward relevance (WASM validator = P_B), and flow proxy (BanditPruner Q-values = F(s)). We harmonize these signals.

### Four Additive Distillations

| Distillation | Component | What It Does |
|-------------|-----------|-------------|
| **D1: FlowPruner** | `FlowPruner<P: ScreeningPruner>` | Wraps any screener, adds `λ × (1 - stop_prob[depth])` flow bonus |
| **D2: Balanced DDTree** | `build_dd_tree_balanced()` | Scores beams with `ln(P_llm) + w × ln(R) + λ × flow_bonus` |
| **D3: Flow-weighted bandit** | `observe_delta_with_flow()` | Adds `λ_length / prefix_len` trajectory length bonus to δ reward |
| **D4: Backward replay** | `ReplayBackwardWalker` | Walks winning replays backward, finds safe alternatives = P_B data |

### Benchmark Results (NoScreeningPruner baseline)

| Metric | Result |
|--------|--------|
| FlowPruner node delta | **+0.0%** ✅ |
| Balanced DDTree backward compat | **Identical to `build_screened`** ✅ |
| Flow-weighted bandit reward delta | **+0.0%** ✅ |
| Backward replay alternatives | **4.0 avg/tick** (target: ≥2) ✅ |

Run: `cargo test --features "bandit,g_zero,bomber" --test bench_gflownet_modelless -- --nocapture`

📖 See [`.plans/052_gflownet_modelless_distillation.md`](.plans/052_gflownet_modelless_distillation.md) for full plan, [`.research/23_GFlowNet_Shortest_Paths.md`](.research/23_GFlowNet_Shortest_Paths.md) for paper analysis.

## 🏭 Productions

MicroGPT-RS is the **core inference library** — pure algorithms, zero side effects. It powers a broader production ecosystem:

### E2E Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────────┐
│  RAG Engine  │    │  Training    │    │  Service Layer                   │
│  ingest,     │───▸│  Pipeline    │───▸│  ┌──────────────────────────┐   │
│  curate,     │JSON│  LoRA train  │.bin│  │  Transpiler Service      │   │
│  export      │    │  + pack      │    │  │  (uses microgpt-rs lib)  │   │
└──────────────┘    └──────────────┘    │  └────────────┬─────────────┘   │
                                        │               │                  │
                                        │  ┌────────────▼─────────────┐   │
                                        │  │  WASM Validator SDK      │   │
                                        │  │  builds .wasm validators │   │
                                        │  └──────────────────────────┘   │
                                        │                                  │
                                        │  ┌──────────────────────────┐   │
                                        │  │  Domain Router           │   │
                                        │  │  keyword + embedding     │   │
                                        │  └──────────────────────────┘   │
                                        │                                  │
                                        │  ┌──────────────────────────┐   │
                                        │  │  GPU Training            │   │
                                        │  │  wgpu LoRA forward/bwd   │   │
                                        │  └──────────────────────────┘   │
                                        │                                  │
                                        │  ┌──────────────────────────┐   │
                                        │  │  REST Client             │   │
                                        │  │  vector search + tokens  │   │
                                        │  └──────────────────────────┘   │
                                        └──────────────────────────────────┘
```

### How It Flows

1. **RAG Engine** (anyrag) — Self-improving knowledge base with plugin-based ingestion (`Ingestor` trait), episodic memory, catalog-driven domain shaping, slot management, inference budget API (β parameterization), Turso/SQLite storage, REST API + CLI, and Cloud Run deployment. Curates quality training data and exports JSONL. Episodic memory accumulates edge cases per-translation, feeding back into the curation loop.

2. **Training Pipeline** (riir-burner) — LoRA fine-tuning for Gemma 4 E4B on Rust code corpus. Takes curated JSONL, trains LoRA adapters (Python→Rust pairs), produces compact `adapter.bin` with BLAKE3 checksum. Rust handles pack/verify; Python (unsloth/MLX) handles training. CLI subcommands: `pack`, `verify`, `train`, `pipeline`. Shell scripts: `lora.sh`, `pack.sh`.

3. **Service Layer** (riir-ai, private) — Monorepo housing:
   - **WASM Validator SDK** (riir-validator-sdk) — WASM Validator trait + `export_validator!` macro + streaming events ABI. Compiles to sandboxed `.wasm` modules that plug into microgpt-rs's `WasmPruner`.
   - **WASM Runtime** — Host-side `WasmPruner` implementing `ConstraintPruner` + `ScreeningPruner`. Loads `.wasm`, calls `is_valid`/`relevance` in sandboxed wasmtime.
   - **Prompt Router + Expert Registry** — `KeywordRouter` (V1) + `EmbeddingRouter` (V2, 3-tier fallback via RAG) + `ExpertRegistry` mapping domains to pruner + LoRA pairs. Config-driven via `domains.toml` with domain inference budget (β). Routing strategies: keyword, embedding, combined.
   - **GPU Training** — ✅ Production-ready `wgpu` compute pipeline with 21 WGSL kernels. Forward, backward (LoRA grads only), AdamW optimizer, cross-entropy loss, PFlash block-sparse prefill (4 kernels), TurboQuant attention scoring, TTT feedback consumer. Targets WebGPU, Metal, Vulkan, DX12. LoRA export/load.
   - **REST Client** — HTTP client for vector search against the RAG Engine. Retrieves historically successful token continuations merged into DDTree branches.
   - **Transpiler** (riir-transpiler) — Python→Rust transpilation service loading `.wasm` validators + `.bin` LoRA adapter. Exercises the full pipeline: BPE tokenize → WASM validate → DDTree prune → compiler feedback.

### Architecture Split

| Layer | Repo | What | Status | License |
|-------|------|------|--------|---------|
| **Engine** | microgpt-rs | DDTree, zero-alloc, ConstraintPruner, ScreeningPruner | ✅ Working | MIT |
| **Validator** | microgpt-rs | SynPruner + PartialParser + CompilerFeedback | ✅ Working | MIT |
| **RAG Engine** | anyrag | Plugin ingestion (`Ingestor` trait), episodic memory, slot management, catalog-driven domain shaping, inference budget API (β), Turso/SQLite storage | ✅ Working | MIT |
| **Training Pipeline** | riir-burner | LoRA fine-tuning (Gemma 4 E4B), adapter packing (BLAKE3), corpus dedup, pack/verify/train/pipeline CLI | ✅ Working | MIT |
| **WASM SDK** | riir-ai | Validator trait + export macro + streaming events ABI + CLI checker | ✅ Working | Private |
| **WASM Runtime** | riir-ai | WasmPruner + wasmtime sandbox | ✅ Working | Private |
| **Router** | riir-ai | Keyword + Embedding routing (3-tier fallback), ExpertRegistry, domain inference budget (β) | ✅ Working | Private |
| **GPU Training** | riir-ai | ✅ Production-ready wgpu pipeline (21 WGSL kernels): forward/backward, PFlash, TurboQuant, feedback consumer, LoRA export | ✅ Working | Private |
| **REST Client** | riir-ai | Vector search, tokenization, agent hints | ✅ Working | Private |
| **Transpiler** | riir-ai | Python→Rust transpilation, compiler feedback loop | ✅ Working | Private |

### Key Insight

The engine (microgpt-rs) is MIT and fully functional. But without trained LoRA adapters from riir-burner (the "fuel") and domain-specific WASM validators from riir-ai, it produces syntactically-valid-but-semantically-generic output. The private riir-ai monorepo holds the trained weights, validator SDK, and orchestration — the intelligence layer that makes the engine production-grade for specific domains like Python→Rust transpilation. anyrag's episodic memory accumulates edge cases per-translation, creating a data flywheel that improves accuracy over time.

## 🛠️ Getting Started

### Prerequisites

- Rust 1.85+ (edition 2024, 1.93+ recommended)

### Build & Run

```sh
# Build with optimizations
cargo build --release

# Run benchmark + generate plot (16 benchmarks)
cargo run --release

# Run with Sudoku constraint pruner
cargo run --release --features sudoku

# Run everything
cargo run --release --all-features

# Run all tests (674 total)
cargo test --quiet --workspace --all-features

# Run Sudoku solver example
cargo run --example sudoku_01_9x9 --features sudoku

# Run speculative decoding comparison
cargo run --example sudoku_02_speculative --features sudoku

# Run TUI visualization
cargo run --example sudoku_03_tui --features sudoku

# Lint
cargo clippy --all-targets --all-features --quiet
```

### Feature Flags

| Flag | Description |
|------|-------------|
| `sudoku` | SudokuPruner constraint pruning + examples |
| `validator` | SynPruner + partial parser (BPE tokenizer, `syn` AST) |
| `sparse_mlp` | TwELL-inspired sparse MLP matmul (Plan 022) |
| `ppot` | PPoT logit-parameterized CPU resampling + adaptive rescue (Plan 026) |
| `domain_latent` | Mid-layer domain conditioning (Plan 038) |
| `bandit` | Multi-armed bandit + HL infrastructure (TrialLog, AbsorbCompress, HotSwapPruner) |
| `bomber` | Bomberman HL arena (bevy_ecs + bandit, Plan 033) |
| `bomber-wasm` | WASM bomber validator loader (bomber + wasmtime + papaya, Plan 034) |
| `monopoly` | Monopoly FSM arena (bevy_ecs + bandit, Plan 035) |
| `feedback` | E2E feedback loop — sends inference results to REST endpoint (Plan 042, requires consumer in riir-gpu) |
| `rest` | REST bridge test + merge stub (Plan 009, client lives in riir-ai/riir-rest) |
| `embedding_router` | Semantic embedding routing (Plan 024, not yet started) |
| `gpu` | Placeholder — GPU training lives in riir-ai/riir-gpu |
| `game_domain` | Alias for `domain_latent` — game-specific Config presets (Plan 040) |
| `language_domain` | Language domain: BPE vocab, LLM models (Plan 040, future) |
| `g_zero` | G-Zero self-play + FFT arena + Bomber arena + TFT party AI (Plans 049–055) |
| `full` | Enable all features |

> **Note:** `LeviathanVerifier` is always compiled (no feature gate) — it's part of `verifier.rs` and `benchmark.rs`. `Transformer AR`, `DFlash`, `Raven`, `TurboQuant`, and `PFlash` are also always available — they're zero-cost until their caches are instantiated.

## 📁 Project Structure

```
src/
  lib.rs            Module index
  main.rs           Entry point (proof → bench → Percepta bench → plot)
  types.rs          Config, Rng, math kernels, LoraAdapter, LoraPair
  transformer.rs    Weights, KVCache (flat/paged/raven), ForwardContext, forward/generate
  speculative/      SOLID decomposition:
    types.rs        TreeNode, ConstraintPruner, ScreeningPruner, SpeculativeContext
    dd_tree.rs      DDTree build (best-first + chain-seed + screened)
    dflash.rs       DFlash predict (marginal, AR, parallel, conditioned)
    verifier.rs     SpeculativeVerifier, SimulatedVerifier, LeviathanVerifier
    step.rs         High-level step functions (speculative, rollback, conditioned)
    prefill.rs      Speculative prefill scoring + prompt compression
    sampling.rs     Temperature, top-k, top-p sampling strategies
    ppot/           PPoT CPU resampling:
      mod.rs         Module root
      entropy.rs     Entropy-based sampling
      resample.rs    Resampling strategies
      knowledge.rs   Knowledge distillation
      rank.rs        Rank-based selection
      types.rs       PPoT types
  pruners/          Pruner & HL infrastructure:
    bandit.rs       BanditPruner, BanditSession, BanditEnv, strategies
    trial_log.rs    TrialLog JSONL persistence
    absorb_compress.rs  Q-value → hard block promotion
    hot_swap.rs     Runtime pruner reload via blake3
    regression.rs   Golden trace replay
    review_metrics.rs   Helpfulness/Harmfulness metrics + benefit-risk ratio
    sudoku_pruner.rs    Path-aware Sudoku constraint pruning
    tactical_pruner.rs  Tactical pathfinding pruner
    dungeon_pruner.rs   Dungeon map pruner
    dungeon_pathfinder.rs  Dungeon pathfinder
    map_generator.rs    Procedural map generation
    pathfinder.rs      A* pathfinding
    g_zero/          G-Zero self-play distillation:
      mod.rs           Module root
      delta_absorb.rs  Delta absorb logic
      delta_bandit.rs  Delta bandit strategies
      template_proposer.rs  Template proposing
      types.rs         G-Zero types
    bomber/          Bomberman HL arena (bevy_ecs):
      mod.rs           Module root
      arena.rs         Arena setup
      players.rs       Player entities
      replay.rs        Replay system
      systems.rs       ECS systems
      wasm_pruner.rs   WASM pruner
      wasm_state.rs    WASM state
      tft_player.rs    TftPlayer — game theory Tit-for-Tat bomber (Issue 056)
    fft/             FFT Tactics Arena (ATB battle engine):
      mod.rs           Module root
      types.rs         Class, Team, ActionType, Stats, Unit, Action, GameEvent, TFT types
      battle.rs        BattleState, ATB resolution, resolve_action
      players.rs       FftPlayer trait + Greedy, Validator, HL implementations
      status.rs        Status effects (Poison, Sleep, Haste, Slow, etc.)
      g_zero_player.rs GZeroFFTPlayer — template hints + δ bandit (Plan 053)
      tft_player.rs    TftFFTPlayer — Tit-for-Tat party AI (Plan 055)
    monopoly/        Monopoly FSM arena (bevy_ecs):
      mod.rs           Module root
      board.rs         Board definition
      players.rs       Player entities
      systems.rs       ECS systems
  tokenizer/        BPE tokenizer (encode/decode/train):
    mod.rs           Module root
    bpe.rs           BPE algorithm
    types.rs         Tokenizer types
  validator/        SynPruner + PartialParser + CompilerFeedback:
    mod.rs           Module root
    partial_parser.rs  Partial JSON/code parsing
    syn_pruner.rs    Syntax-aware pruning
    types.rs         Validator types
  percepta.rs       O(log N) convex hull attention, Sudoku solvers, StreamingSolver
  turboquant/      TurboQuant KV cache compression:
    mod.rs          Module root (re-exports)
    types.rs        TurboQuantCodebook, TurboQuantLayer, TurboQuantKVCacheConfig
    codebook.rs     Lloyd-Max codebook (compute_codebook, quantize, dequantize)
    rotation.rs     QR-based orthogonal rotation + QJL projection
    kv_cache.rs     TurboQuantKVCache (store_key, store_value, dequantize, bit-pack)
    forward.rs      attention_turboquant, dequantize_keys_flat/values_flat, cosine_similarity
  alloc.rs          Debug-only tracking allocator (feature-gated debug_assertions)
  feedback.rs       TTT feedback (feature-gated feedback)
  benchmark.rs      BenchResult, run_all, save_results_csv
  plot.rs           PNG horizontal bar chart
examples/           39 examples (sudoku, validator, bandit, bomber, monopoly, tactical, dungeon, raven, prefill)
tests/              88+ integration tests + 9 benchmark suites (TurboQuant, PFlash NIAH)
bench/              Auto-numbered PNG + CSV benchmark output
```

## 🔧 Production Lessons from NVIDIA Dynamo

Lessons from [NVIDIA Dynamo's agentic inference](https://developer.nvidia.com/blog/streaming-tokens-and-tools-multi-turn-agentic-harness-support-in-nvidia-dynamo/) applied to our stack:

| Lesson | Our Implementation |
|--------|-------------------|
| Prompt stability for KV cache reuse | `PagedKVCache` prefix reuse; prefix stability benchmark |
| Streaming tool dispatch | `DraftEvent` enum fires at structural completion |
| Interleaved reasoning preserved | `extract_parent_tokens()` maintains ordered sequences |
| Single parser ownership | `ConstraintPruner` owns structural, `ScreeningPruner` owns semantic |
| Catalog metadata shapes behavior | `TruncationPolicy` + `ReasoningRetention` per domain |
| Per-request agent hints | `AgentHints` with latency_sensitivity, priority, speculative_prefill |
| `/v1/tokenize` for context accounting | BPE-based tokenize/detokenize endpoint types |

## 📜 References

- [microgpt-c](https://github.com/nicholasgasior/microgpt-c) — Original C implementation
- [talos-vs-macbook](https://github.com/alexcb123/talos-vs-macbook) — Reference model
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192) — Leviathan et al., 2022
- [DFlash: Block-Diffusion Speculative Decoding](https://arxiv.org/abs/2602.06036) — Wang et al., 2026
- [DDTree: Block Diffusion Draft Trees](https://arxiv.org/abs/2604.12989) — Ringel & Romano, 2026
- [Cross-Family Speculative Prefill](https://arxiv.org/abs/2603.02631) — Liu et al., ICLR 2026
- [ZAYA1-VL-8B Technical Report](https://arxiv.org/abs/2504.02268) — Bidirectional prefix attention, token-specific LoRAs
- [Raven: Sparse Memory Routing](https://github.com/goombalab/raven) — Afzal et al., 2025
- [Percepta: Can LLMs Be Computers?](https://www.percepta.ai/blog/can-llms-be-computers) — O(log N) hull attention
- [Sparser, Faster, Lighter Transformers](https://arxiv.org/abs/2603.23198) — Sakana AI, 2025
- [EMO: Mixture of Experts](https://arxiv.org/abs/2406.08732) — Document-level routing
- [Probabilistic Programs of Thought](https://arxiv.org/abs/2604.17290) — Logit-parameterized CPU resampling
- [Reinforced Agent: Inference-Time Feedback](https://arxiv.org/abs/2604.27233) — Review metrics, benefit-risk ratio
- [Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub/) — Per-chip LLM inference
- [TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate](https://arxiv.org/pdf/2504.19874) — Zandieh et al., 2025
- [Luce PFlash: Speculative Prefill Compression for Long-Context Spec Decode](https://github.com/Luce-Org/lucebox-hub/) — lucebox-hub, 2026
- [Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/) — Heuristic Learning paradigm
- [G-Zero: Self-Play for Open-Ended Generation from Zero Data](https://arxiv.org/pdf/2605.09959) — Huang et al., 2026 — Verifier-free co-evolutionary self-play via Hint-δ, GRPO Proposer, length-normalized DPO Generator