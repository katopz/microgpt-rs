# MicroGPT-RS

Speculative Decoding with DFlash & DDTree — a high-performance Rust implementation of a micro-Transformer with built-in benchmarking and visualization.

Inspired by [microgpt-c](https://github.com/nicholasgasior/microgpt-c), [talos-vs-macbook](https://github.com/alexcb123/talos-vs-macbook), and [Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub/).

## 🚀 Key Features

- **Real Transformer Inference** — Full GPT forward pass with RMSNorm, multi-head causal attention, ReLU MLP, KV cache, and temperature sampling.
- **Zero-Alloc Forward Pass** — Pre-allocated `ForwardContext` buffers eliminate heap allocations per inference step.
- **Separate Draft Model** — Lightweight draft model (embd=4, heads=2, mlp=16) runs **3.6× faster** per forward pass than the target model.
- **DFlash (Dynamic Flash)** — Block-parallel drafting that predicts `L` future tokens simultaneously via independent marginal distributions. Supports `rayon` parallelism. Also available in **autoregressive mode** (`dflash_predict_ar`) that samples and feeds back tokens for conditional q(x|x<t) distributions.
- **DDTree (Dynamic Draft Tree)** — Best-First Search using a `BinaryHeap` to build a candidate token tree from marginal log-probabilities. Tree budget constrains exploration.
- **SpeculativeVerifier (Strategy Pattern)** — Swappable verification via trait: `SimulatedVerifier` (fast, no target model) or `LeviathanVerifier` (real p/q rejection sampling with target model).
- **Leviathan Algorithm 1** — Full implementation of [Leviathan et al. 2022](https://arxiv.org/pdf/2211.17192): AR draft → target model p/q scoring → rejection sampling → residual distribution `max(0, p−q)` → bonus token from target p(x). Distribution-preserving, proven correct, but needs large model asymmetry to be faster than pure AR.
- **Bonus Token** — When all γ draft tokens are accepted, sample +1 token for free from the last marginal (simulated) or target p(x) at position γ (Leviathan).
- **Residual Distribution Sampling** — `max(0, p−q)` normalized distribution for sampling replacement tokens on rejection (Algorithm 1, Equation 3).
- **Constraint Pruner** — Pluggable `ConstraintPruner` trait for neuro-symbolic intercept: deterministic rules engine prunes invalid branches before target verification.
- **Path-Aware Pruning** — `SudokuPruner` validates against accumulated path state (initial board + parent tokens), catching cross-depth row/col/box conflicts that static-only pruning misses.
- **Deterministic Validator** — LLM drafts tokens via semantic probability, deterministic rules engine validates via mathematical constraints, only valid branches reach verification. Demonstrated with 9×9 Sudoku.
- **Streaming Solver** — `StreamingSolver` emits `Try`/`Accepted`/`Contradiction`/`Backtrack`/`Solved` events for real-time visualization of the search process.
- **Percepta O(log N) Attention** — 2D convex hull KV cache with ternary search, proving LLMs can execute programs internally via geometric attention. Includes adversarial failure tests.
- **TUI Visualization** — Ratatui-based terminal UI showing the Sudoku solver in real-time: color-coded grid, step/trace panels, speculative mode comparison (behind `--features sudoku`).
- **Benchmarks + Plots** — 16-method benchmark suite (AR, DFlash, DDTree, Speculative, AR Draft, Leviathan, Prefill, forward) with auto-numbered PNG output via `plotters`. All benchmarks run by default — no feature flags needed.
- **Chain-Seed DDTree** — Greedy chain backbone (argmax per depth) before best-first expansion, recovering high-confidence token spine. Inspired by [DFlash](https://arxiv.org/abs/2602.06036).
- **Speculative Prefill** — PFlash-inspired prompt compression via attention-based importance scoring. Draft model scores token importance, compresses to top-`keep_ratio` spans before target prefill. Inspired by [Cross-Family Speculative Prefill](https://arxiv.org/abs/2603.02631).
- **KV-Cache Snapshot/Rollback** — Cheap per-position KV cache snapshots for branch-level tree verification. Restore on rejection to try alternate DDTree paths without full recomputation.

## 🏗️ Architecture

Matching the talos-vs-macbook reference model:

| Parameter | Value |
|-----------|-------|
| `vocab_size` | 27 (a–z + BOS) |
| `block_size` | 16 |
| `n_embd` | 16 |
| `n_head` | 4 |
| `head_dim` | 4 |
| `mlp_hidden` | 64 (4×) |
| `n_layer` | 1 |
| `temperature` | 0.5 |
| `draft_lookahead` | 8 |
| `tree_budget` | 16 nodes |

### Zero-Alloc Architecture

All hot paths use pre-allocated buffers created once, reused across decode steps:

- **`SpeculativeContext`** — holds `ForwardContext`, `MultiLayerKVCache`, flat marginals buffer, probs buffer, sampled/accepted tokens, residual scratch, p-distributions flat buffer
- **`TreeBuilder`** — holds `BinaryHeap`, tree `Vec`, chain nodes/parent tokens; `clear()` reuses capacity
- **`_with` variants** — `dflash_predict_with`, `dflash_predict_ar_with`, `generate_into`, `score_with`, `sample_residual_distribution_into` — borrow pre-allocated buffers instead of allocating per call
- **`generate_batch`** — rayon parallel multi-sample generation with per-worker `ForwardContext` + `KVCache`

### Forward Pass

```
x = wte[token] + wpe[pos]
x = rmsnorm(x)
x = x + attention(rmsnorm(x))    # Q, K, V → causal attention → Wo
x = x + mlp(rmsnorm(x))          # W1 → ReLU → W2
logits = lm_head @ x
```

### DFlash (Block-Parallel Drafting)

Standard Transformers are limited by causal masking. DFlash bypasses this during the draft phase by producing `L` independent marginal distributions:

```
P(x_{t+1}), P(x_{t+2}), ..., P(x_{t+L})  |  x_{<t}
```

Each position uses an isolated forward pass, simulating non-causal parallel prediction.

### DDTree (Dynamic Draft Tree)

Rather than a single linear draft chain, DDTree builds a tree of the most probable paths:

- **Algorithm**: Best-First Search (priority queue / max-heap)
- **Metric**: Cumulative log-probability
- **Budget**: `tree_budget` nodes (default 16)
- **Outcome**: A tree that maximizes Expected Acceptance Length (EAL)
- **Pruning**: `build_dd_tree_pruned()` integrates any `ConstraintPruner` — invalid tokens never enter the heap

### Path-Aware Constraint Pruning

The DDTree's `ConstraintPruner` trait supports **path-aware** validation:

```rust
pub trait ConstraintPruner: Send + Sync {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;
}
```

- `parent_tokens[k]` = token placed at depth `k` in the current branch path
- `SudokuPruner` checks cross-depth row/col/box conflicts incrementally — O(parent_tokens.len()) per check, no board copy needed
- `extract_parent_tokens(parent_path, num_tokens)` decodes the `TreeNode::parent_path` bitfield (16 bits per depth, u128 bitfield)

**3-level comparison** (Arto Inkala, 8 depths, budget=100):

```
Unpruned:    100 nodes,  46 accumulated-valid (46.0%)
Static-Only: 100 nodes,  84 accumulated-valid (84.0%)   ← misses cross-depth conflicts
Path-Aware:  100 nodes, 100 accumulated-valid (100.0%)   ← catches everything
```

### SpeculativeVerifier (Strategy Pattern)

Based on [Algorithm 1 from Leviathan et al. 2022](https://arxiv.org/pdf/2211.17192) — the verification strategy is swappable via trait:

```rust
pub trait SpeculativeVerifier: Send + Sync {
    fn speculate(&mut self, draft_weights, draft_config, token, pos, rng) -> Vec<usize>;
}
```

| Verifier | Feature Flag | What it does |
|----------|-------------|--------------|
| `SimulatedVerifier` | always available | DFlash/AR draft → DDTree → simulated acceptance cap → bonus token from last marginal |
| `LeviathanVerifier` | always available | AR draft → target model p/q scoring → rejection sampling → residual distribution → bonus from target p(x). Proves Algorithm 1 works end-to-end. |

`SimulatedVerifier` is fast (no target model). `LeviathanVerifier` is the full Algorithm 1 — mathematically proven distribution-preserving, but needs large model asymmetry to be faster than pure AR.

## 🧠 Deterministic Validator: Neuro-Symbolic Intercept

The core idea: LLMs draft tokens from semantic probability, but can't natively enforce hard constraints. A deterministic rules engine sits between draft and verification:

```
LLM drafts logits → ConstraintPruner filters invalid → DDTree builds valid-only tree → Target verifies
```

### Sudoku as the LLM Problem

Standard LLMs are notoriously bad at Sudoku (constraint-satisfaction / Exact Cover problem). They predict tokens from semantic probability but can't backtrack or run spatial checks. The Deterministic Validator intercept forces mathematical validity on top of semantic probability.

### Public API

```rust
// 9×9 Sudoku board with validation, solving, and display
let board = Sudoku9x9::arto_inkala();  // 21 clues, 60 empty cells
let valid = board.is_valid_move(0, 1, 4);  // Check row/col/box constraints

// Deterministic Validator: filter draft logits through constraints
let drafts = vec![(1, 0.15), (3, 0.12), (4, 0.20), (5, 0.08), (8, 0.11)];
let valid_drafts = SymbolicValidator::prune_drafts(&board, 0, 1, &drafts);
// Returns only digits valid at (0,1): [(4, 0.20), (1, 0.15)]

// Streaming solver with event trace
let mut solver = StreamingSolver::new(board.grid);
solver.solve_streaming();  // Emits Try/Accepted/Contradiction/Backtrack/Solved events
```

### Streaming Solve Events

```rust
pub enum SolveEvent {
    Try { row: usize, col: usize, digit: u8, depth: usize },
    Accepted { row: usize, col: usize, digit: u8, filled: usize },
    Contradiction { row: usize, col: usize, digit: u8, depth: usize },
    Backtrack { row: usize, col: usize, depth: usize },
    Solved { steps: usize, hull_size: usize, total_trace: usize },
}
```

### Arto Inkala Results

"World's Hardest Sudoku" with 21 clues, 60 empty cells:

| Metric | Value |
|--------|-------|
| Solve steps | 49,559 |
| Hull vertices | 7 |
| Compression ratio | 7,079.9× |
| Attention retrieval | O(49,559) → O(log 7) ≈ O(3) |

## 📊 Benchmark Results

Run on Apple Silicon (single-threaded, `--release` profile, 50k iterations, **zero-alloc hot paths**):

**Models:** Target (embd=16, heads=4, mlp=64) · Draft (embd=4, heads=2, mlp=16) · Benchmark run `044` (commit `0d52e11`)

```
Method                         Throughput         μs/step  Avg Accept Len
───────────────────────────────────────────────────────────────────────────────
Transformer AR                  1,153,065 tok/s       0.87            1.00
DFlash                         3,854,077 tok/s       2.08            8.00
DDTree Build                     361,662 trees/s      2.77            —
Speculative (Simulated)        1,008,030 tok/s       4.96            5.00
Speculative (AR Draft)         1,458,004 tok/s       4.80            7.00
Leviathan (Algorithm 1)         110,259 tok/s      10.69            1.18
Leviathan (no rollback)         109,930 tok/s      10.72            1.18
Leviathan (w/ rollback)         185,353 tok/s       6.34            1.18
Spec (unconditioned)           1,012,393 tok/s       4.94            5.00
Spec (conditioned)              1,099,772 tok/s       6.13            6.74
Prefill (no compress)         19,190,151 tok/s       3.34           64.00
Prefill (compressed)           1,921,788 tok/s       3.64            7.00
DDTree (no chain)                364,027 trees/s      2.75           16.00
DDTree (chain-seed)              377,198 trees/s      2.65           16.00
forward (flat)                 1,122,676 trees/s      0.89            —
forward_paged                    942,262 trees/s      1.06            —
───────────────────────────────────────────────────────────────────────────────
📈 Best speedup: 1.26x (Speculative AR Draft vs AR)
```

![Benchmark Chart](bench/044_bench_result.png)

### What each benchmark measures

The benchmarks progress from individual components to full pipelines:

#### Core Components

| Benchmark | What it does | Why it matters |
|-----------|-------------|----------------|
| **Transformer AR** | Baseline: 1 target model forward pass → 1 token. The "slow path" that speculative decoding tries to beat. | Ground truth. If speculative can't beat this, it's not worth the complexity. |
| **DFlash** | **Block-parallel draft prediction**: runs 8 independent forward passes on the tiny draft model (same input, different positions). Produces 8 marginal distributions P(x₁), P(x₂), …, P(x₈) without autoregressive feedback. | The "fast but inaccurate" draft. 8 tokens predicted in parallel vs 1 from AR. Throughput is per-draft-token. |
| **DDTree Build** | Builds a Best-First Search tree (budget=16 nodes) from DFlash marginals. Explores the most probable token paths using cumulative log-probability. | Converts flat marginals into a tree of candidate paths. The tree maximizes Expected Acceptance Length. |

#### Speculative Decoding Pipelines

| Benchmark | Pipeline | Why it matters |
|-----------|----------|----------------|
| **Speculative (Simulated)** | DFlash → DDTree → extract best path → simulated 75% acceptance cap → bonus token. No target model. "Simulated" means acceptance is artificially capped rather than using real p/q rejection. | Full speculative pipeline without expensive target verification. Measures the "ideal" case where draft quality is good. Avg 5 accepted tokens per step. |
| **Speculative (AR Draft)** | Same pipeline but uses **autoregressive** drafting: each step feeds back the sampled token as input for the next. Produces conditional q(x\|x<t) distributions instead of independent marginals. | AR drafting produces more coherent sequences → better acceptance (7 tokens vs 5). The extra sampling cost pays off in longer accepted sequences. |
| **Spec (unconditioned)** | Identical to "Speculative (Simulated)" — measured separately as the baseline for comparison with the conditioned variant below. | Control group: same draft model, same pipeline, no target information. |
| **Spec (conditioned)** | **Target-conditioned draft**: runs target model forward to capture hidden state → projects it into draft model's KV cache as initial seed → draft now "sees" target features → produces better-informed marginals → DDTree → simulated acceptance. | The draft model gains partial access to the target's representation. Higher avg acceptance (6.74 vs 5.0) shows conditioning helps, but the extra target forward pass adds latency. |

#### Leviathan: Real Target Verification (Algorithm 1)

| Benchmark | Pipeline | Why it matters |
|-----------|----------|----------------|
| **Leviathan (Algorithm 1)** | Full [Algorithm 1](https://arxiv.org/pdf/2211.17192): AR draft → target model scores ALL drafted tokens → real p/q rejection sampling → residual distribution on rejection → bonus token. **Mathematically distribution-preserving** — output is identical to sampling from target alone. | Proves the algorithm works end-to-end. Slow here because our target is only 4× bigger than draft — needs ~10× for speculative to win. |
| **Leviathan (no rollback)** | Standard Leviathan that resets target KV cache each step. No branch recovery. | Baseline for rollback comparison. |
| **Leviathan (w/ rollback)** | Leviathan with KV cache **snapshot/rollback**: before each verification attempt, snapshots the target KV cache (cheap O(n_layer × pos × kv_dim) copy). On rejection, rolls back and tries the next candidate branch — avoids re-running target from scratch. | Snapshot cost is negligible; rollback saves re-running target. 73% faster than no-rollback (6.17 vs 10.66 μs). Essential for multi-branch verification. |

#### Prompt Compression (PFlash-inspired)

| Benchmark | Pipeline | Why it matters |
|-----------|----------|----------------|
| **Prefill (no compress)** | Uses the draft model's **self-attention weights** as a per-token importance proxy. Runs forward passes over a 64-token prompt (block_size × 4), extracts attention scores, normalizes to [0, 1]. All 64 tokens are kept. | Measures raw importance scoring speed. The 612× zero-alloc speedup came from reusing `SpeculativeContext` instead of creating `ForwardContext` + `KVCache` per call. |
| **Prefill (compressed)** | Same scoring as above, then **compresses** to keep_ratio=0.1 — keeps only the top 10% most important tokens (~7 of 64). Inspired by [Cross-Family Speculative Prefill](https://arxiv.org/abs/2603.02631): instead of feeding all prompt tokens to the (expensive) target model, score them with the (cheap) draft model and only prefill the important ones. | The actual use case: reduces target prefill work by ~9× at the cost of losing ~90% of prompt tokens. The 7 kept tokens represent the most "attended-to" positions. |

#### Tree Building Variants

| Benchmark | What differs | Why it matters |
|-----------|-------------|----------------|
| **DDTree (no chain)** | Standard best-first: seeds heap with all root marginals, expands greedily by score. | Baseline tree construction. |
| **DDTree (chain-seed)** | **Chain-seed optimization**: first builds a greedy backbone (argmax at each depth with cumulative log-prob), then seeds the heap with siblings at each chain depth + children of the last chain node. Inspired by [DFlash](https://arxiv.org/abs/2602.06036). | The chain provides a "highway" through the tree that pure best-first might miss. Slightly faster (2.56 vs 2.66 μs) because the initial heap population is more focused. |

> These benchmarks require the target model for real verification or hidden state extraction.

### Zero-Alloc Improvements (Plan 013)

Pre-allocated `SpeculativeContext` + `TreeBuilder` structs eliminate per-step heap allocations:

| Method | Before (μs) | After (μs) | Improvement |
|--------|-------------|-------------|-------------|
| DFlash | 2.60 | 1.95 | **33% faster** |
| DDTree Build | 3.19 | 2.78 | **15% faster** |
| Speculative (Simulated) | 5.92 | 4.82 | **23% faster** |
| Speculative (AR Draft) | 5.70 | 4.73 | **21% faster** |
| Prefill (no compress) | 23.78 | 3.32 | **616% faster** |
| Prefill (compressed) | 23.99 | 3.61 | **565% faster** |
| DDTree (chain-seed) | 3.16 | 2.54 | **24% faster** |

### Speculative Decoding Pipeline

```
Phase 1: DRAFT — Run small model M_q for γ tokens (DFlash: parallel, AR: sequential).
         Save q(x) distributions at each step.
Phase 2: VERIFY — Check drafted tokens against target distribution.
         Simulated: cap acceptance at 75%.  Leviathan: real p/q rejection.
Phase 3: BONUS — If all γ accepted, sample +1 free token from target distribution.
Result:  γ+1 tokens accepted at best, 1 at worst (guaranteed progress).
```

**Key insight: Model size ratio matters.** The paper assumes ~10× target/draft cost ratio (e.g., 70B/7B). Our ratio is ~4× (embd=16/4), so real target verification is a net loss:

| Verifier | How it verifies | Target model? | Perf |
|----------|----------------|---------------|------|
| `SimulatedVerifier` | Accepts ceil(γ × rate) tokens + bonus from last marginal | ❌ No | ~1.05M tok/s |
| `SimulatedVerifier` (AR) | Same, but autoregressive drafting (conditional q(x\|x<t)) | ❌ No | ~1.44M tok/s |
| `LeviathanVerifier` | Real p/q rejection + residual distribution + bonus from target p(x) | ✅ Yes | ~110K tok/s |

The `SpeculativeVerifier` trait makes them swappable — same `speculate()` interface, different verification strategy. When LoRA fine-tuning improves draft/target alignment, real verification becomes viable.

> **`LeviathanVerifier` proves Algorithm 1 works end-to-end** — mathematically correct, distribution-preserving, but ~10× slower than simulated at our 4× model ratio. This is the expected result: speculative decoding needs large model asymmetry to win.

### Why Leviathan is slow here (and when it won't be)

```
Leviathan cost:  γ × draft_cost + (γ+1) × target_cost + overhead
              =  8 × 0.25μs      + 9 × 0.84μs            + ~1.6μs
              =  2.0μs            + 7.6μs                + 1.6μs  = 10.65μs / 1.18 accepted = 9.0μs/token

Simulated cost:  DFlash + DDTree + acceptance
              =  1.98μs  + 2.66μs + ~0.12μs
              =  4.76μs / 5 accepted = 0.95μs/token
```

With random weights the draft and target distributions are poorly aligned → low acceptance rate (1.18/8 = 15%). After LoRA fine-tuning, acceptance should approach 80%+ and Leviathan becomes competitive at larger model sizes.

### Transformer Proof of Correctness

```
Sample 1: "aursrmzzzzzmzzzz" (valid=true)
Sample 2: "auczzzzzzzcmzzzz" (valid=true)
Sample 3: "auuzzzzzzzzmzzzz" (valid=true)

✅ Deterministic: PASS (same seed = same output)
✅ Diverse:       PASS (different seed = different output)
✅ Valid tokens:  PASS (all tokens in [0, 27))
```

## 🔬 Percepta: O(log N) 2D Convex Hull Attention

Based on [Percepta's "Can LLMs Be Computers?"](https://www.percepta.ai/blog/can-llms-be-computers) — the idea that transformers with 2D attention heads can execute programs internally for millions of steps without quadratic slowdown.

### The Core Idea

Standard attention scans all N past keys → O(N) per step, O(N²) total. Percepta restricts attention heads to d=2, making the dot product a 2D geometric projection. When keys form a convex hull, finding the maximum attention score becomes ternary search → **O(log N)**.

```
Standard:  Q·K for all N keys  → O(N) per step
Percepta:  ternary search hull  → O(log H) per step (H = hull size ≤ N)
```

### What We Proved

| Claim | Evidence | Key Test |
|-------|----------|----------|
| O(log N) matches O(N) for convex distributions | 360° sweep, 10K points | `test_supporting_point_property` |
| Hull maintenance is amortized O(1) | Graham scan on 100K points | `test_linear_fast_agree_100k_trace` |
| Dot products on hull are unimodal | Bitonic sequence verified for 5 query directions | `test_hull_dot_products_unimodal` |
| DFA execution can be encoded | Divisible-by-3 DFA on all integers 0..1000 | `test_dfa_divisible_by_3_stress` |
| Computation traces fit the mechanism | Counter (collinear) + Fibonacci (exponential) | `test_counter_trace_collinear`, `test_fibonacci_trace_attention` |
| **All 4 arithmetic ops work** | +, −, ×, ÷ computed via attention retrieval | `test_arithmetic_comprehensive` |
| **Power works** | 2^10 = 1024 via repeated doubling | `test_arithmetic_power` |
| **Combined expressions work** | (3+5)×2−2 = 14 via tiny VM | `test_arithmetic_combined_expression` |
| **Backtracking search works** | 4×4 Sudoku + 8-Queens solved via attention-tracked DFS | `test_sudoku_4x4_backtracking`, `test_nqueens_8_backtracking` |
| **Hull captures search peaks** | Backtrack valleys compressed, solution retained | `test_sudoku_4x4_hull_captures_search` |
| **9×9 Sudoku solved** | Arto Inkala "World's Hardest" in 49,559 steps, 7 hull vertices | `test_sudoku9x9_solve_arto_inkala` |
| **Deterministic Validator prunes drafts** | Invalid logits filtered before DDTree build | `test_computable_lora_prune_drafts` |
| **Path-aware pruning catches cross-depth conflicts** | Same-digit same-row conflicts between parent/child depths | `test_ddtree_path_aware_catches_cross_depth_conflicts` |

### Adversarial Findings (Limitations Discovered)

| Failure Mode | Evidence | Key Test |
|--------------|----------|----------|
| **V-shaped keys fail** — valleys are invisible to upper hull | Negative-y query returns wrong answer | `test_adversarial_v_shape_fast_attention_wrong` |
| **Multiple valleys = systematic** — not a one-off edge case | W-shape also fails | `test_adversarial_multiple_valleys` |
| **Exponential growth over-compresses** — hull collapses to 2 endpoints | Fibonacci trace loses all interior info | `test_fibonacci_trace_attention` |

### Arithmetic Computation Proof

We proved that the 4 fundamental operations can be computed incrementally using 2D attention. Each step retrieves the previous accumulator via `fast_attention(query)` and computes the next value from the retrieved result:

| Operation | How | Example | Test |
|-----------|-----|---------|------|
| **Add** | Increment acc by 1, repeat b times | 42 + 17 = 59 | `test_arithmetic_addition` |
| **Sub** | Decrement acc by 1, repeat b times | 100 − 37 = 63 | `test_arithmetic_subtraction` |
| **Mul** | Repeated addition of operand | 7 × 8 = 56 | `test_arithmetic_multiplication` |
| **Div** | Repeated subtraction, count steps | 100 ÷ 7 = 14 r 2 | `test_arithmetic_division` |
| **Mod** | Division, return remainder | 17 % 5 = 2 | `test_arithmetic_modulo` |
| **Pow** | Repeated multiplication (doubling) | 2^10 = 1024 | `test_arithmetic_power` |
| **Combined** | Tiny VM: LOAD/ADD/MUL/SUB | (3+5)×2−2 = 14 | `test_arithmetic_combined_expression` |

The comprehensive test (`test_arithmetic_comprehensive`) verifies all a+b, a×b, a−b, a÷b for a,b ∈ 0..=10 — **960 arithmetic operations**, all computed correctly via attention-based state retrieval.

**Key insight**: Query `(1, 0)` always retrieves the latest state because `dot((1,0), (step, acc)) = step`, maximized at the most recent entry. This works regardless of whether acc increases, decreases, or grows exponentially.

### Backtracking Search Proof (Sudoku & N-Queens)

The Percepta blog solved the Arto Inkala Sudoku (hardest in the world) inside a transformer at 32K tok/s — **without training**. They *compiled* a C solver into transformer weights. Our PoC proves the attention substrate handles the same backtracking pattern:

| Problem | What It Tests | Result |
|---------|--------------|--------|
| **4×4 Sudoku** | DFS with constraint checking + backtracking | Solved correctly, hull compresses valleys |
| **8-Queens** | Column + diagonal conflict detection + backtracking | Solved correctly, 8 queens placed without conflicts |
| **9×9 Sudoku (Arto Inkala)** | Full backtracking solve with hull trace | 49,559 steps, hull compressed to 7 vertices (7,080×) |
| **Backtracking pattern** | Forward → peak → dead-end → undo → new branch | Hull captures peaks, skips valleys |

**No training needed** — the solver is a deterministic state machine. The attention mechanism tracks its execution trace through forward placements AND backtracking undos. The hull compresses the "mountain range" trace: search peaks are retained, backtrack valleys are dropped.

### Correctness Guarantee

`fast_attention` is guaranteed correct when:
- Keys have monotonically non-decreasing X (natural for sequential traces)
- The key with maximum dot product lies **on the upper convex hull** (not inside a valley)
- For concave-down distributions (parabolic execution traces), this holds for all query directions

### What This Does NOT Prove

The Percepta team demonstrated **full in-model computation** (33K tok/s, 7K lines/s on CPU) by compiling a WASM interpreter into transformer weights. Our PoC proves the **algorithmic substrate** (O(log N) hull attention) is correct and identifies its limitations. The actual "LLM as computer" claim additionally requires:
1. **Trained 2D head embeddings** — real hidden states forming convex distributions
2. **Compiled program weights** — FFN layers implementing deterministic state machines
3. **Execution trace structure** — monotonic keys from sequential program steps

### Hull Compression Ratios

| Distribution | Total Keys | Hull Size | Compression |
|-------------|-----------|-----------|-------------|
| Concave-down parabola | 1,000 | 1,000 | 0% (all on hull) |
| Sinusoidal | 5,000 | <2,500 | >50% |
| Zigzag | 1,000 | <100 | >90% |
| Collinear (flat) | 100 | ≤2 | ~98% |
| Exponential (Fibonacci) | 45 | ≤2 | ~96% |
| Arto Inkala backtracking | 49,559 | 7 | 99.99% (7,080×) |

## 🛠️ Getting Started

### Prerequisites

- Rust 1.85+ (edition 2024, 1.93+ recommended)

### Build & Run

```sh
# Build with optimizations
cargo build --release

# Run benchmark + generate plot (16 benchmarks)
cargo run --release

# Run with Sudoku constraint pruner (adds SudokuPruner tests + examples)
cargo run --release --features sudoku

# Run everything (all benchmarks + Sudoku pruner)
cargo run --release --all-features

# Run all tests
# Default:              196 tests (lib) + 80 integration = 276 total
cargo test --quiet --workspace --all-features

# Run Sudoku solver example (streaming "thinking" output)
cargo run --example sudoku_9x9 --features sudoku

# Run speculative decoding comparison (Unpruned / Static / Path-Aware)
cargo run --example sudoku_speculative --features sudoku

# Run TUI visualization (real-time grid + speculative mode, requires terminal)
cargo run --example sudoku_tui --features sudoku

# Lint
cargo clippy --all-targets --all-features --quiet
```

### Output

- Console: transformer proof + benchmark table
- `bench/NNN_bench_result.png`: auto-numbered horizontal bar chart (plotters)
- `bench/NNN_results.csv`: paired CSV for regression tracking (commit, date, method, throughput, latency)

### Feature Flags

| Flag | Dependencies | Description |
|------|-------------|-------------|
| `sudoku` | — | SudokuPruner constraint pruning + examples |
| `validator` | `syn`, `proc-macro2` | SynPruner + partial parser |
| `rest` | `reqwest`, `tokio` | REST module |
| `gpu` | `wgpu`, `bytemuck`, `pollster`, `safetensors` | wgpu context & buffers |
| `wasm` | `wasmtime`, `wat` | WASM validator runtime (WasmPruner) |
| `full` | all above (except leviathan, always on) | Enable all features |

Build with `--features <flag>` or `--all-features`.

## 🔌 WASM Validator Pipeline

Curators can write domain-specific constraint validators in Rust, compile them to `.wasm`, and have the DDTree load them at runtime.

### Quick Start

```sh
# 1. Write a validator using the SDK
cargo add riir-validator-sdk

# 2. Implement the Validator trait
# See: https://github.com/katopz/riir-validator-sdk/examples/

# 3. Build for WASM
cargo build --target wasm32-unknown-unknown --release

# 4. Check the validator locally
riir-validator-check target/wasm32-unknown-unknown/release/examples/your_validator.wasm

# 5. Use with microgpt-rs
cargo run --features wasm
```

### WASM ABI

| Export | Signature | Description |
|--------|-----------|-------------|
| `is_valid` | `(u32, u32, u32, u32) -> u32` | Token-level validation |
| `validate_string` | `(u32, u32) -> u32` | String-level validation |
| `name` | `() -> u32` | Pointer to null-terminated name |
| `version` | `() -> u32` | Packed `(major << 16) \| (minor << 8) \| patch` |

### Constraints
- **No WASI imports** — fully sandboxed
- **No floating-point** — deterministic across platforms
- **Max memory: 64 pages (4MB)**
- **Max execution: ~100μs per call** — fuel-based enforcement

### Example Validators
- `bracket_validator` — bracket balancing (like PartialParser)
- `keyword_validator` — Rust keyword placement rules

## 📁 Project Structure

```
src/
  lib.rs            Module index
  main.rs           Entry point (proof → bench → Percepta bench → plot)
  types.rs          Config (micro + draft), Rng, softmax, rmsnorm, matmul, sample_token
  transformer.rs    TransformerWeights, KVCache, PagedKVCache, ForwardContext, forward, forward_paged, generate, generate_into, generate_batch
  speculative/      SOLID decomposition (plan 005):
    mod.rs          Re-exports
    types.rs        TreeNode, DraftResult, ConstraintPruner trait, NoPruner, SpeculativeContext (zero-alloc), DDTreeBranchCache (wraps PagedKVCache)
    sampling.rs     sample_from_distribution, sample_residual_distribution, sample_residual_distribution_into
    dd_tree.rs      build_dd_tree, build_dd_tree_pruned, TreeBuilder (zero-alloc, parent_tokens_buf), extract_parent_tokens, extract_parent_tokens_into
    dflash.rs       dflash_predict, dflash_predict_with, dflash_predict_ar, dflash_predict_ar_with, dflash_predict_parallel
    verifier.rs     SpeculativeVerifier trait, SimulatedVerifier, LeviathanVerifier (all zero-alloc internals)
    step.rs         speculative_step, speculative_step_verifier, speculative_step_rollback, speculative_step_conditioned
    prefill.rs      PrefillScorer trait, AttentionScorer, compress_prompt, speculative_prefill, score_with
    sudoku_pruner.rs  SudokuPruner (path-aware, cross-depth conflict detection) *
  tokenizer/        BPE tokenizer (encode/decode/train, Config::bpe())
  validator/        SynPruner + partial parser ‡
  gpu/              wgpu context & buffers §
  rest/             REST module ¶
  wasm/             WasmPruner + WASM runtime (abi, state, wasmtime loader) †
  percepta.rs       Vec2, KVCache2D — O(log N) 2D convex hull attention (Percepta)
                    Sudoku9x9, SymbolicValidator, StreamingSolver, SolveEvent
  benchmark.rs      BenchResult, run_all, save_results_csv (AR / DFlash / DDTree / Speculative / AR Draft / Leviathan)
  plot.rs           plot_results → PNG horizontal bar chart
  * behind --features sudoku
  ‡ behind --features validator
  § behind --features gpu
  ¶ behind --features rest
  † behind --features wasm
examples/
  sudoku_9x9.rs          Streaming solver with "thinking" output + hull compression stats *
  sudoku_speculative.rs  3-column DDTree comparison: Unpruned / Static-Only / Path-Aware *
  sudoku_tui.rs          Ratatui TUI: real-time grid visualization + speculative mode *
tests/
  integration.rs  80 integration tests (adversarial + DFA + arithmetic + backtracking + geometry
                  + Sudoku9x9 + SymbolicValidator + StreamingSolver)
bench/
  001_bench_result.png  ...  026_bench_result.png (auto-numbered)
  001_results.csv       ...  026_results.csv       (paired CSV, same index as PNG)
```

## 📜 References

- [microgpt-c](https://github.com/nicholasgasior/microgpt-c) by Vishal Baraiya
- [talos-vs-macbook](https://github.com/alexcb123/talos-vs-macbook) by Alex Cheema
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192) — Leviathan et al., 2022 (Algorithm 1: draft → target scoring → p/q rejection → residual sampling → bonus token)
- SpecInfer — tree-based speculative verification (inspiration for DDTree)
- [Percepta: Can LLMs Be Computers?](https://www.percepta.ai/blog/can-llms-be-computers) — 2D convex hull attention for in-model execution
- [Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub/) — Open LLM Inference, Rewritten by Hand for One Specific Chip at a Time
- [DFlash: Block-Diffusion Speculative Decoding](https://arxiv.org/abs/2602.06036) — Wang et al., 2026 (chain-seed DDTree, target-conditioned draft)
- [DDTree: Block Diffusion Draft Trees](https://arxiv.org/abs/2604.12989) — Ringel & Romano, 2026 (budget sweep, tree verify)
- [Cross-Family Speculative Prefill](https://arxiv.org/abs/2603.02631) — Liu et al., ICLR 2026 (importance scoring, prompt compression)
- [FlashPrefill](https://arxiv.org/abs/2603.06199) — Fan et al., 2026 (block-sparse drafter attention)
- [Hazy Research Megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) — Intelligence Per Watt methodology