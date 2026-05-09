# Research: EMO — Emergent Modularity via Document-Level Routing Constraints

**Date:** 2025-06
**Status:** Research → Verdict
**Context:** microgpt-rs + anyrag + riir-validator-sdk neuro-symbolic architecture
**Paper:** "EMO: Pretraining Mixture of Experts for Emergent Modularity" (2025)

---

## TL;DR

Standard Mixture-of-Experts (MoE) models like Mixtral route every token independently across all experts. The result: experts learn low-level syntax (one handles commas, another handles "the") instead of high-level semantic domains (one for code, another for math). You can't extract a subset of experts because they're all entangled.

EMO fixes this with a **document-level routing constraint**: before routing any token, first select a pool of experts for the entire document. Every token in that document must route *only within that pool*. This forces experts to specialize in high-level domains because they never see tokens outside their pool during training.

The result: you can extract 8 coding experts from a 128-expert model, throw away the other 120, and retain 99% performance. This is the "modular MoE" that the marketplace model needs.

---

## The Problem: Token-Level MoE Creates Syntax Experts

### How Standard MoE Routing Works

```
Token 1 (comma)     → Top-2 of 128 experts → [Expert_3, Expert_47]
Token 2 ("function") → Top-2 of 128 experts → [Expert_12, Expert_89]
Token 3 ("the")     → Top-2 of 128 experts → [Expert_3, Expert_91]
```

Every token picks freely. Expert_3 sees both commas AND the word "the" — it learns syntax, not semantics. If you extract Expert_12 (which sometimes sees "function"), it breaks without Expert_3 (which handles the surrounding syntax).

### Why This Breaks Modularity

```
Standard MoE: 128 experts, each handles ~20% of tokens
Expert_3: commas, "the", periods, "and"  (syntax glue)
Expert_12: "function", "class", imports   (code keywords)
Expert_89: variable names, strings        (identifiers)

→ You CANNOT extract "the coding experts" because every expert
   handles a mix of syntax and semantics.
```

---

## The Solution: Document-Level Pool Constraint

### EMO's Two-Phase Routing

```
Phase 1 (Document Pool Selection):
  "This document is about Rust code" → Pool D = [Expert_4, Expert_12, Expert_19,
                                                  Expert_42, Expert_55, Expert_88,
                                                  Expert_91, Expert_102]
  (Select 8-32 experts for the ENTIRE document)

Phase 2 (Token Routing within Pool):
  Token 1 → Top-2 of Pool D only → [Expert_4, Expert_42]
  Token 2 → Top-2 of Pool D only → [Expert_12, Expert_91]
  Token 3 → Top-2 of Pool D only → [Expert_4, Expert_55]
  (Every token is CONSTRAINED to Pool D)
```

### Why This Creates Semantic Experts

Because Expert_3 (the comma expert) is never selected for code documents, it either:
- Dies (gets no gradient updates from code docs), or
- Specializes in non-code domains (prose, math, etc.)

Meanwhile, Expert_12 only ever sees code tokens. It learns "code semantics" not "code syntax" because the syntax tokens are handled by other experts in the same pool.

### The Math (Simplified)

Standard MoE routing for token `t`:
```
P(expert_i | t) = softmax(W_router · h_t)  [over all 128 experts]
```

EMO routing for token `t` in document with pool `D`:
```
P(expert_i | t, D) = softmax(W_router · h_t)  [only for i ∈ D]
                    = 0                         [for i ∉ D]
```

The constraint is applied during **training**, not just inference. This is the key — you can't retrofit EMO onto a pretrained MoE without retraining.

---

## What We Can Actually Use

### What EMO Proves (Conceptually Valid)

1. **Document-level routing creates modular experts.** If you constrain routing to a pool for an entire document/task, the experts specialize in domains. This is mathematically proven.

2. **Expert subsets run standalone.** EMO shows you can extract 8/128 experts and retain 99% domain performance. This validates the "only load what you need" pattern.

3. **Modularity enables marketplace economics.** If experts are truly domain-specialized, they can be independently created, priced, and distributed.

### What We Cannot Use (Practically)

| Barrier | Why |
|---------|-----|
| **No MoE architecture** | `microgpt-rs` uses dense MLP layers (`mlp_w1`, `mlp_w2`). No router, no expert FFNs, no `W_router`. Adding MoE requires replacing the entire forward pass. |
| **Models too small** | Current configs: `n_embd=16-64`, `mlp_hidden=16-256`. MoE with 128 experts would mean 128 separate MLPs, each needing enough capacity to be useful. At our scale, one dense MLP is more efficient than 8 tiny experts. |
| **EMO is a training-time constraint** | The document-level pool must be applied during pretraining. We don't have a training pipeline for MoE models. This is not an inference-only optimization. |
| **No pretrained EMO model** | There is no open-weight EMO model we can download and use. The paper proves the concept but doesn't ship a usable checkpoint. |

### What We Actually Distilled (Plan 023)

Instead of neural MoE, we apply EMO's **concept** — document-level domain selection — to our existing architecture:

```
EMO:     Prompt → Document Pool Selection → Constrain neural routing
Ours:    Prompt → Domain Classification     → Select pruner + LoRA adapter
```

| EMO Concept | Our Distillation | Where |
|-------------|-----------------|-------|
| Document pool selection | `PromptRouter` trait — classify prompt once per request | `microgpt-rs` Plan 023 |
| Constrained routing to pool | `ExpertRegistry` — lock pruner+LoRA for entire generation | `microgpt-rs` Plan 023 |
| Modular expert subsets | `ExpertBundle` — `Box<dyn ScreeningPruner>` + `Option<PathBuf>` for LoRA | `microgpt-rs` Plan 023 |
| Domain-specialized experts | Curator `.wasm` validators + `.bin` LoRA adapters | Marketplace |
| Expert marketplace | Curators upload domain bundles, users load only what they need | Strategy doc |

The key difference: EMO's "experts" are neural network weight slices. Our "experts" are deterministic pruners + optional LoRA adapters. Same modularity concept, different mechanism — and ours works at our model scale.

---

## What This Does NOT Prove

1. **Does not prove we should add MoE.** At our model sizes, MoE overhead exceeds benefit. Dense + LoRA is more efficient.

2. **Does not prove batch-level routing equals EMO routing.** Our routing is manual (keyword/config-driven). EMO routing is learned during pretraining. Our experts don't "specialize" — they're explicitly assigned.

3. **Does not prove the marketplace will work.** EMO proves neural experts can be modular. Our marketplace is about deterministic validators + LoRA adapters, which are modular by construction (they're separate files). The marketplace viability is a business question, not a technical one.

---

## Integration Points

| Component | Current | With Plan 023 | Future (EMO-scale) |
|-----------|---------|---------------|-------------------|
| Routing | Manual (developer picks pruner) | `KeywordRouter` auto-selects | Embedding-based via anyrag (Plan 005) |
| Experts | Hardcoded pruners | `ExpertRegistry` from `domains.toml` | Curator marketplace uploads |
| Expert loading | Compile-time (feature flags) | Runtime (WASM cache) | Runtime (neural expert swap) |
| LoRA selection | Not implemented | `Option<PathBuf>` in registry | Actual adapter loading |
| MoE architecture | None | None | Only if models scale to 1B+ params |

---

## Honest Caveats

1. **We are not implementing EMO.** We are borrowing the conceptual pattern (document-level domain selection → constrained expert subset) and applying it to our deterministic + LoRA architecture. This is honest and practical. Calling it "EMO" would be misleading.

2. **Batch-level routing is weaker than EMO routing.** EMO forces every token through domain-specific neural pathways. Our batch-level routing forces every token through the same *pruner*, but the neural network weights are the same (unless LoRA is loaded). The constraint is at the validation level, not the computation level.

3. **Future MoE integration is possible but far off.** If microgpt-rs eventually scales to models where MoE makes sense (1B+ params, 128+ experts), the `PromptRouter` + `ExpertRegistry` architecture from Plan 023 naturally extends — the registry would hold neural expert slices instead of WASM pruners. The trait interface doesn't change.

4. **The "99% performance with 8/128 experts" claim requires EMO-trained models.** You can't get this with standard MoE models. If you extract 8 experts from Mixtral, performance tanks. EMO's training constraint is what makes extraction work.

---

## Verdict

**Conceptually valid, practically adapted.** EMO proves that document-level routing constraints create modular, extractable experts. We cannot use the neural mechanism (no MoE arch, models too small, no training pipeline), but we **can** use the pattern: classify the task once, select the right tools, lock them for the entire generation.

This is exactly what Plan 023 implements — `PromptRouter` classifies the domain, `ExpertRegistry` loads the right pruner + LoRA, and DDTree runs with the selected tools for the entire request. Same modularity, different mechanism, honest about what it is.

The paper validates the marketplace concept (modular experts are possible) but does not provide the mechanism we use (deterministic pruners + LoRA adapters instead of neural expert slices).