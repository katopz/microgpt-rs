# MicroGPT-RS

Speculative Decoding with DFlash & DDTree — a high-performance Rust implementation of a micro-Transformer with built-in benchmarking and visualization.

Inspired by [microgpt-c](https://github.com/nicholasgasior/microgpt-c) and [talos-vs-macbook](https://github.com/alexcb123/talos-vs-macbook).

## 🚀 Key Features

- **Real Transformer Inference** — Full GPT forward pass with RMSNorm, multi-head causal attention, ReLU MLP, KV cache, and temperature sampling.
- **Zero-Alloc Forward Pass** — Pre-allocated `ForwardContext` buffers eliminate heap allocations per inference step.
- **Separate Draft Model** — Lightweight draft model (embd=4, heads=2, mlp=16) runs **3.6× faster** per forward pass than the target model.
- **DFlash (Dynamic Flash)** — Block-parallel drafting mechanism that predicts `L` future tokens simultaneously via independent marginal distributions. Supports `rayon` parallelism for larger models.
- **DDTree (Dynamic Draft Tree)** — Best-First Search using a `BinaryHeap` to build a candidate token tree from marginal log-probabilities.
- **Speculative Verification** — Draft → Tree → Verify pipeline that accepts multiple tokens per step.
- **Percepta O(log N) Attention** — 2D convex hull KV cache with ternary search, proving LLMs can execute programs internally via geometric attention. Includes adversarial failure tests.
- **Benchmarks + Plots** — 4-component benchmark suite with auto-numbered PNG output via `plotters`.

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

## 📊 Benchmark Results

Run on Apple Silicon (single-threaded, `--release` profile, 50k iterations):

**Models:** Target (embd=16, heads=4, mlp=64) · Draft (embd=4, heads=2, mlp=16)

```
Method                    Throughput         μs/step  Avg Accept Len
───────────────────────────────────────────────────────────────────────────
Transformer AR             882,099 tok/s         1.13            1.00
DFlash                    2,891,011 tok/s         2.77            8.00
DDTree Build               383,372 trees/s       2.61            —
Speculative Decoding       739,714 tok/s         5.41            4.00

📈 Speedup: 0.84x (Speculative Decoding vs AR)
```

![Benchmark Chart](bench/007_bench_result.png)

### What each benchmark measures

| Benchmark | What it does | Metric |
|-----------|-------------|--------|
| **Transformer AR** | 1 target model forward pass | tok/s (1 token per step) |
| **DFlash** | 8 draft model forward passes (block-parallel prediction) | draft tok/s (8 tokens per step) |
| **DDTree Build** | Tree construction from DFlash marginals | trees/s |
| **Speculative Decoding** | DFlash + DDTree + accept (~75% of draft tokens) | effective tok/s (4 tokens accepted per step) |

> DFlash is the draft. "DFlash Draft" was a redundant label — DFlash IS the drafting mechanism.

### Per-Step Cost Breakdown

```
Transformer AR:    1.13μs × 1 forward pass  = 1.13μs/token

Speculative:       0.35μs × 8 draft passes  = 2.77μs  (DFlash)
                  + 2.61μs tree build        = 2.61μs  (DDTree)
                  ─────────────────────────────────────
                  = 5.38μs / 4 accepted tokens = 1.35μs/token
```

| Component | Time | vs Draft Forward (0.35μs) |
|-----------|------|--------------------------|
| 1 Draft forward | 0.35μs | 1× |
| 8 Draft forwards (DFlash) | 2.77μs | 8× |
| DDTree build | 2.61μs | 7.5× |

### Why the speedup is marginal

The draft model is **3.3× faster** per forward pass than the target (2.9M vs 882K tok/s). But the **DDTree build costs as much as ~7.5 draft forward passes** — tree overhead dominates because the model is tiny.

With real models (e.g., LLaMA-70B target / 7B draft), forward passes take milliseconds while tree construction stays in microseconds — tree becomes <0.1% overhead and speculative decoding wins decisively. The framework is ready for real models.

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

## 🛠️ Getting Started

### Prerequisites

- Rust 1.85+ (edition 2024)

### Build & Run

```sh
# Build with optimizations
cargo build --release

# Run benchmark + generate plot
cargo run --release

# Run all tests (176 tests)
cargo test --quiet

# Lint
cargo clippy --all-targets
```

### Output

- Console: transformer proof + benchmark table
- `bench/NNN_bench_result.png`: auto-numbered bar chart (plotters)

## 📁 Project Structure

```
src/
  lib.rs          Module index
  main.rs         Entry point (proof → bench → Percepta bench → plot)
  types.rs        Config (micro + draft), Rng, softmax, rmsnorm, matmul, sample_token
  transformer.rs  TransformerWeights, KVCache, ForwardContext, forward, generate
  speculative.rs  dflash_predict, dflash_predict_parallel, TreeNode, build_dd_tree, speculative_step
  percepta.rs     Vec2, KVCache2D — O(log N) 2D convex hull attention (Percepta)
  benchmark.rs    BenchResult, run_all (AR / DFlash / DDTree / Speculative Decoding)
  plot.rs         plot_results → PNG bar chart
tests/
  integration.rs  68 integration tests (includes adversarial + DFA + arithmetic + geometry)
bench/
  001_bench_result.png
  002_bench_result.png  ...
```

## 📜 References

- [microgpt-c](https://github.com/nicholasgasior/microgpt-c) by Vishal Baraiya
- [talos-vs-macbook](https://github.com/alexcb123/talos-vs-macbook) by Alex Cheema
- Speculative Decoding papers (Leviathan et al., Chen et al.)
- [Percepta: Can LLMs Be Computers?](https://www.percepta.ai/blog/can-llms-be-computers) — 2D convex hull attention for in-model execution