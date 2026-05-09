# Research: Sakana TwELL — Unstructured Sparsity for MLP Acceleration

**Date:** 2025-06
**Status:** Research → Verdict
**Context:** microgpt-rs inference engine (CPU + GPU paths)
**Paper:** "Sparser, Faster, Lighter Transformer Language Models" (arXiv:2603.23198) — Sakana AI & NVIDIA

---

## TL;DR

MLP layers account for ~80% of LLM FLOPs. With ReLU activation (which microgpt-rs already uses), up to 99% of hidden neurons become exactly `0.0` after training with L1 regularization. Standard dense matmul wastes cycles computing `0 × weight`. The TwELL (Tile-wise ELLPACK) algorithm dynamically compresses non-zero activations and skips dead neurons entirely.

Applied to microgpt-rs: replace the `w2 @ hidden` matmul in the MLP with a sparse variant that only processes the non-zero (alive) neurons. The `active_neurons` index buffer lives in `ForwardContext` for zero-alloc execution.

---

## The Problem: Dense Matmul on Sparse Data

### Current MLP Flow (from `transformer.rs`)

```rust
// Phase 1: w1 projection + ReLU (always dense — input is not sparse)
types::matmul_relu(&mut ctx.hidden, &layer_weights.mlp_w1, &ctx.x, mlp_hidden, n);

// Phase 2: w2 projection (CURRENTLY DENSE — but hidden is 95-99% zeros!)
matmul(&mut ctx.x, &layer_weights.mlp_w2, &ctx.hidden, n, mlp_hidden);
```

Phase 2 is the bottleneck:
- `mlp_hidden = 4 × n_embd` (standard 4x expansion)
- For `n_embd=4096`, `mlp_hidden=16384`
- If 99% of `hidden` is 0.0, we're doing 16,384 multiplications per row when only ~164 matter

### The Paper's Finding

The Sakana/NVIDIA paper proves:
1. ReLU naturally produces 90%+ sparsity in trained LLMs
2. Adding light L1 regularization pushes sparsity to 99%+ without hurting perplexity
3. The TwELL format (compressed sparse row with tile alignment) enables efficient GPU execution
4. On CPU, simple index packing (our approach) achieves equivalent speedup

---

## The Mathematical Core

### Dense Matmul (current)
```
output[r] = Σ_{c=0}^{cols-1} weight[r*cols + c] × hidden[c]
```
Computes `cols` multiplications per row regardless of how many `hidden[c]` are zero.

### Sparse Matmul (proposed)
```
Phase 1 (Pack): Scan hidden[c] for c where hidden[c] > threshold
Phase 2 (Multiply):
  output[r] = Σ_{c ∈ alive} weight[r*cols + c] × hidden[c]
```
Only computes `|alive|` multiplications per row.

### Speedup Estimate

| Sparsity | Alive % | Theoretical Speedup | Practical Speedup |
|----------|---------|--------------------|--------------------|
| 90% | 10% | 10× | 5-8× (packing overhead) |
| 95% | 5% | 20× | 10-15× |
| 99% | 1% | 100× | 20-50× |

Practical speedup is lower due to packing phase, cache effects, and branch prediction overhead. The packing cost is `O(mlp_hidden)` scan; the savings are `O(rows × mlp_hidden × (1 - sparsity))`.

### Break-even Analysis

Packing cost: `mlp_hidden` comparisons + `alive_count` stores
Savings per row: `mlp_hidden - alive_count` skipped multiplications
Total savings: `n_embd × (mlp_hidden - alive_count)` multiplications

Break-even sparsity where packing cost < savings:
- At `n_embd=64, mlp_hidden=256`: break-even at ~20% sparsity (80% alive)
- At `n_embd=4096, mlp_hidden=16384`: break-even at ~1% sparsity (99% alive)

For our current configs, sparse wins even at moderate sparsity.

---

## Integration Architecture

### Where It Plugs In

The sparse matmul replaces **only** the second MLP matmul (`w2 @ hidden`):

```
forward() / forward_paged() / forward_raven()
  │
  ├── Attention (unchanged)
  │
  └── MLP
       ├── matmul_relu(hidden, w1, x)  ← Dense (input not sparse)
       └── sparse_matmul(x, w2, hidden) ← Sparse (hidden is ReLU output)
```

All three forward functions (`forward`, `forward_paged`, `forward_raven`) have identical MLP blocks, so the sparse optimization applies uniformly.

### CPU Path: Sparse Index Packing

```rust
/// Sparse matrix-vector multiply for ReLU-activated inputs.
/// Only processes columns where input[c] > threshold.
#[inline(always)]
pub fn sparse_matmul(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
    // Pre-allocated buffers from ForwardContext
    active_indices: &mut [usize],
    active_values: &mut [f32],
) -> usize {
    // Phase 1: Pack alive neurons
    let mut alive = 0;
    for c in 0..cols {
        if input[c] > 0.0 {
            active_indices[alive] = c;
            active_values[alive] = input[c];
            alive += 1;
        }
    }

    // Phase 2: Sparse multiply
    for r in 0..rows {
        let row_off = r * cols;
        let mut sum = 0.0f32;
        for i in 0..alive {
            let c = unsafe { *active_indices.get_unchecked(i) };
            let val = unsafe { *active_values.get_unchecked(i) };
            sum += unsafe { *weight.get_unchecked(row_off + c) } * val;
        }
        unsafe { *output.get_unchecked_mut(r) = sum; }
    }

    alive // Return count for diagnostics
}
```

### GPU Path: Keep Dense

GPU matmul is memory-bound, not compute-bound. Sparse gather patterns cause warp divergence and uncoalesced memory access. On GPU:
- Keep using `dispatch_lora_merge` + `dispatch_relu` for MLP w1
- Keep using dense `dispatch_lora_merge` for MLP w2
- If GPU sparse is needed in future, use structured sparsity (N:M pattern) not unstructured

### Feature Gate Design

```toml
[features]
default = []
sparse_mlp = []  # Enable sparse MLP matmul on CPU path
```

When `sparse_mlp` is enabled:
- `ForwardContext` gets `active_indices: Vec<usize>` and `active_values: Vec<f32>` buffers
- `forward()` uses `sparse_matmul` for the w2 matmul
- GPU path ignores this feature entirely

When `sparse_mlp` is disabled (default):
- Everything works exactly as before
- Zero overhead, zero risk

### Fallback Strategy

The sparse path should include a **runtime sparsity check**:

```rust
if alive_count < (cols as f32 * AUTO_SPARSE_THRESHOLD) as usize {
    sparse_matmul(...)  // Sparse wins when < threshold% alive
} else {
    matmul(...)         // Dense wins when too many alive neurons
}
```

This way, even with `sparse_mlp` enabled, models without L1 training (low sparsity) automatically fall back to dense without performance regression.

---

## Why Only Phase 2 (w2)?

The MLP has two matmuls:

| Matmul | Shape | Input | Sparse? | Optimization |
|--------|-------|-------|---------|--------------|
| `w1 @ x` | [mlp_hidden, n_embd] | Post-RMSNorm activation | No (not ReLU'd) | Keep dense |
| `w2 @ hidden` | [n_embd, mlp_hidden] | Post-ReLU hidden | Yes! (95-99% zeros) | Sparse |

`w1 @ x` operates on the residual-stream activation, which is dense (RMSNorm doesn't zero things out). Only `w2 @ hidden` benefits from sparsity because `hidden = ReLU(w1 @ x)` is guaranteed to have many exact zeros.

---

## The Trinity: How This Completes the Architecture

```
                    Raven (Fixed Memory Slots)
                    O(1) KV routing, sparse write
                           │
                           ▼
              ┌─────────────────────────┐
              │   microgpt-rs Engine    │
              │                         │
              │   Attention: Raven RSM  │
              │   Branching: Screening  │
              │   MLP: TwELL Sparse     │
              └─────────────────────────┘
                           ▲
                           │
              ┌────────────┴────────────┐
              │                         │
     Screening (Absolute        TwELL (Unstructured
     Relevance Pruning)         Sparsity MLP)
     DDTree branch control      MLP compute acceleration
```

- **Raven**: O(1) memory — the *memory* problem
- **Screening**: Absolute relevance — the *judgment* problem
- **TwELL**: Sparse MLP — the *compute* problem

Together: memory-bounded, judgment-guided, compute-optimized inference.

---

## Risks & Caveats

1. **Sparsity depends on training**: Without L1 regularization, ReLU sparsity may be only 70-80%, reducing speedup. Runtime fallback to dense handles this.
2. **Packing overhead**: The scan phase adds O(mlp_hidden) work. For small models (micro config: mlp_hidden=64), the overhead may exceed savings. Only enable for models where mlp_hidden >= threshold (e.g., 256).
3. **No GPU benefit**: GPU sparse matmul requires structured sparsity (2:4, 4:8 patterns). Unstructured sparsity causes warp divergence. CPU-only optimization.
4. **Benchmark or it didn't happen**: Must benchmark with real model weights, not synthetic data. The PoC shows the principle; real speedup depends on actual sparsity patterns.
5. **Cache effects**: Sparse access patterns into the weight matrix are less cache-friendly than dense sequential access. At very low sparsity (<5% alive), the working set fits in cache anyway. At medium sparsity (20-50%), cache misses may reduce gains.

---

## Verdict: Adopt (Feature-Gated)

The sparse MLP optimization is:
- **Mathematically sound** — exploits real ReLU sparsity
- **Low risk** — feature-gated, runtime fallback, zero impact when disabled
- **High impact** — targets the heaviest matmul in the forward pass
- **Compatible** — works alongside Raven, Screening, GPU path
- **Honest** — speedup claims must be validated by benchmarks, not assumed

Implementation strategy:
1. Add `sparse_matmul` to `types.rs` alongside existing `matmul` / `matmul_relu`
2. Add `active_indices` / `active_values` buffers to `ForwardContext`
3. Feature-gate with `sparse_mlp`
4. Add runtime auto-detection (sparsity threshold)
5. Benchmark before claiming victory

---

## References

- "Sparser, Faster, Lighter Transformer Language Models" (arXiv:2603.23198) — Sakana AI & NVIDIA
- microgpt-rs MLP: `src/transformer.rs` lines 362-377 (forward), 487-506 (forward_paged), 1070-1089 (forward_raven)
- microgpt-rs matmul: `src/types.rs` — `matmul()`, `matmul_relu()`
- microgpt-rs GPU MLP: `src/gpu/forward.rs` — `dispatch_layer()` lines 555-585
- Raven RSM: `.research/06_Raven_Routing_Slot_Memories.md`
- Screening Pruner: `.research/07_Screening_Absolute_Relevance.md`
