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

1. **RAG Engine** — A self-improving knowledge base ingests documents, curates quality training data, and exports JSONL. Episodic memory accumulates edge cases per-translation, feeding back into the curation loop.

2. **Training Pipeline** — Takes curated JSONL and trains LoRA adapters (Python→Rust pairs for the RIIR use case). Produces compact `adapter.bin` with blake3 checksum. Uses unsloth/MLX for Gemma 4 training; Rust handles pack/verify.

3. **Service Layer** — A private monorepo housing:
   - **WASM Validator SDK** — Internal crate for writing domain-specific constraint validators. The `Validator` trait + `export_validator!` macro compile to sandboxed `.wasm` modules that plug into microgpt-rs's `WasmPruner`.
   - **WASM Runtime** — Host-side `WasmPruner` implementing `ConstraintPruner` + `ScreeningPruner`. Loads `.wasm`, calls `is_valid`/`relevance` in sandboxed wasmtime.
   - **Domain Router** — `KeywordRouter` (V1) + `EmbeddingRouter` (V2 via RAG) + `ExpertRegistry` mapping domains to pruner + LoRA pairs. Config-driven via `domains.toml`.
   - **GPU Training** — `wgpu` compute pipeline with 16 WGSL kernels. Forward, backward (LoRA grads only), AdamW optimizer, cross-entropy loss. Targets WebGPU, Metal, Vulkan, DX12.
   - **REST Client** — HTTP client for vector search against the RAG engine. Retrieves historically successful token continuations merged into DDTree branches.
   - **Transpiler** — Python→Rust transpilation service loading `.wasm` validators + `.bin` LoRA adapter. Exercises the full pipeline: BPE tokenize → WASM validate → DDTree prune → compiler feedback.

### Architecture Split

| Layer | What | Status | License |
|-------|------|--------|---------|
| **Engine** | microgpt-rs (DDTree, zero-alloc, ConstraintPruner, ScreeningPruner) | ✅ Working | MIT |
| **Validator** | SynPruner + PartialParser + CompilerFeedback | ✅ Working | MIT |
| **WASM SDK** | Validator trait + export macro + CLI checker | ✅ Working | Private |
| **WASM Runtime** | WasmPruner + wasmtime sandbox | ✅ Working | Private |
| **Router** | Keyword + Embedding routing, ExpertRegistry | ✅ Working | Private |
| **GPU Training** | wgpu forward/backward/optimizer | ✅ Working | Private |
| **REST Client** | Vector search, tokenization, agent hints | ✅ Working | Private |
| **RAG Engine** | Ingestion, curation, episodic memory | ✅ Working | MIT |
| **Training Pipeline** | LoRA fine-tuning, adapter packing | ✅ Working | MIT |

### Key Insight

The engine is MIT and fully functional. But without trained LoRA adapters (the "fuel") and domain-specific WASM validators, it produces syntactically-valid-but-semantically-generic output. The private service layer holds the trained weights, validator SDK, and orchestration — the intelligence layer that makes the engine production-grade for specific domains like Python→Rust transpilation.

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
| `sparse_mlp` | TwELL-inspired sparse MLP matmul |
| `ppot` | PPoT logit-parameterized CPU resampling + adaptive rescue |
| `bandit` | Multi-armed bandit + HL infrastructure (TrialLog, AbsorbCompress, HotSwapPruner) |
| `bomber` | Bomberman HL arena (bevy_ecs + bandit) |
| `bomber-wasm` | WASM bomber validator loader (bomber + wasmtime) |
| `monopoly` | Monopoly FSM arena (bevy_ecs + bandit) |
| `rest` | REST client for RAG-augmented speculative decoding |
| `embedding_router` | Semantic embedding retrieval from anyrag |
| `gpu` | GPU compute via wgpu, safetensors model loading (planned) |
| `leviathan` | LeviathanVerifier real p/q rejection sampling |
| `full` | Enable all features |

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
    ppot/           PPoT CPU resampling (entropy, resample, knowledge, rank)
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
    bomber/          Bomberman HL arena (bevy_ecs)
    monopoly/        Monopoly FSM arena (bevy_ecs)
  tokenizer/        BPE tokenizer (encode/decode/train)
  validator/        SynPruner + PartialParser + CompilerFeedback
  percepta.rs       O(log N) convex hull attention, Sudoku solvers, StreamingSolver
  benchmark.rs      BenchResult, run_all, save_results_csv
  plot.rs           PNG horizontal bar chart
examples/           36 examples (sudoku, validator, bandit, bomber, monopoly, tactical, dungeon, raven, prefill)
tests/              88 integration tests + 7 benchmark suites
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
- [Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/) — Heuristic Learning paradigm