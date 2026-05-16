# Plan 062: FFO Distillation — First-Order Hypergradient for Modelless Stack

**Branch:** `develop/feature/062_ffo_distillation`
**Depends on:** Plan 053 (δ-Mem Modelless Distillation), Plan 052 (GFlowNet Modelless Distillation)
**Research:** `.research/30_FFO_First_Order_Differentiable_Optimization.md`
**Code:** `.raw/FFOLayer/` (local source audit)
**Goal:** Distill the most valuable techniques from FFOLayer (arXiv:2512.02494) into the microgpt-rs modelless distillation stack. Benchmark-first: measure before, implement, measure after, revert if no gain.

## Tasks

- [ ] T0: Plan creation
- [ ] T1: Benchmark baseline (current ScreeningPruner + BanditPruner + DDTree)
- [ ] T2: `DualMaskedPruner<P>` — active constraint masking using bandit Q-values as dual proxy
- [ ] T3: Benchmark T2 vs baseline
- [ ] T4: `SchurHLAKernel` — Cholesky-accelerated AHLA update (optional, gated on T2 results)
- [ ] T5: Benchmark T4 vs baseline
- [ ] T6: Run clippy, fix warnings, commit

## Architecture

### What This Actually Is

Distilling two techniques from the FFOLayer paper into our Rust inference stack:

1. **Active constraint masking** (P1 from Research 30): The paper uses dual values (Lagrange multipliers) to identify which inequality constraints are active at the optimum, then sparsifies the KKT system. We adapt this by using **bandit Q-values** as a proxy for dual values — high Q-value arms are "active constraints" worth evaluating, low Q-value arms can be masked out (skipped).

2. **KKT Schur complement** (P2 from Research 30): The paper solves the KKT system via Cholesky factorization + Schur complement instead of forming the full Hessian. We adapt this for the **AHLA attention kernel** where the asymmetric update involves solving a least-squares system that has KKT structure.

### What This Is NOT

- NOT a PyTorch differentiable optimization layer (we're pure Rust inference)
- NOT integrating CVXPY or any external solver
- NOT replacing DDTree search with optimization (DDTree already re-solves)
- NOT replacing AdamW in riir-gpu (that's Plan 067 in riir-ai)

### Design: DualMaskedPruner

```rust
// src/pruners/dual_masked.rs

/// Wraps any ScreeningPruner, gating evaluation by bandit Q-value thresholds.
///
/// Distilled from FFOLayer's active-set Lagrangian masking (ffocp_eq.py L1248-1269):
///   mask = (slack <= tol) & (dual >= cutoff)
///   if |mask| > cap: keep top-k by dual value
///
/// We replace:
///   dual values → bandit Q-values (accumulated reward signal)
///   slack threshold → relevance baseline (configurable)
///   cap → max_active_prunes (prevents evaluation explosion)
pub struct DualMaskedPruner<P: ScreeningPruner> {
    inner: P,
    dual_cutoff: f32,        // Q-value threshold for "active" constraints
    max_active: usize,       // cap on active constraints per evaluation
    baseline_relevance: f32, // minimum relevance to consider
}

impl<P: ScreeningPruner> ScreeningPruner for DualMaskedPruner<P> {
    fn relevance(&self, token: usize, context: &[f32], logits: &[f32]) -> f32 {
        // Delegate to inner pruner but with early exit if below baseline
        let r = self.inner.relevance(token, context, logits);
        if r < self.baseline_relevance { return 0.0; }
        r
    }
}
```

### Integration with BanditPruner

The DualMaskedPruner wraps ScreeningPruner, while BanditPruner wraps DualMaskedPruner:

```text
Before: BanditPruner → ScreeningPruner → DDTree
After:  BanditPruner → DualMaskedPruner → ScreeningPruner → DDTree
```

The bandit's Q-values flow down to DualMaskedPruner which uses them as "dual proxies" for the active-set masking.

### File Changes

| File | Change |
|------|--------|
| `src/pruners/dual_masked.rs` | New file — DualMaskedPruner wrapper |
| `src/pruners/mod.rs` | Export new module |
| `src/speculative/types.rs` | Add DualMaskedPruner to SpeculativeContext (optional) |
| `tests/` | New test for dual masking behavior |

### Benchmark Design

#### Baseline (T1)
- Config: micro (27 vocab, 16 embd, 4 heads, hd=4)
- Metrics: DDTree nodes explored, latency per build, acceptance rate
- 1000 speculative steps with SimulatedVerifier

#### T3 (DualMaskedPruner)
- Same config + metrics
- Compare: nodes explored (should decrease), latency (should decrease), acceptance rate (should not degrade >5%)

#### Success Criteria
| Metric | Target |
|--------|--------|
| DDTree node delta | ≤5% fewer nodes explored |
| Latency delta | ≤3% improvement OR no regression |
| Acceptance rate delta | ≤5% regression |
| Memory overhead | ≤1KB per DualMaskedPruner instance |

### Risk Assessment

**Low risk:** DualMaskedPruner is a thin wrapper that falls back to inner pruner. If no gain, we simply don't use it.

**Medium risk:** SchurHLAKernel modifies the AHLA hot path. Gated on T2 results — only proceed if DualMaskedPruner shows clear signal that constraint masking helps.

## References

- Research 30: `.research/30_FFO_First_Order_Differentiable_Optimization.md`
- Paper: arXiv:2512.02494 — "A Fully First-Order Layer for Differentiable Optimization"
- Code: `.raw/FFOLayer/src/ffolayer/` — ffocp_eq.py, ffoqp_eq.py, utils.py
- Plan 052: GFlowNet Modelless Distillation (flow-based pruning precedent)
- Plan 053: δ-Mem Modelless Distillation (modelless distillation precedent)