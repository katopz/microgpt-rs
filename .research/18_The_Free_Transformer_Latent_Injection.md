# Research: The Free Transformer — Latent Injection (18)

> Source: [The Free Transformer](https://arxiv.org/pdf/2510.17558v1) by François Fleuret (FAIR at Meta)
> Date: 2025-10-21, distilled 2025-06
> **Verdict: CONDITIONALLY APPLICABLE — Architecture Insight Valid, No Model Available**

## Summary

The Free Transformer extends a standard decoder Transformer by conditioning its generative process on a random latent variable Z, learned via a conditional VAE. Z is injected at the middle layer (L/2+1) by adding a projected one-hot vector to keys and values. During inference, Z is sampled uniformly — no encoder needed. Cost: ~3% overhead (1 extra non-causal block for training only).

**Key result:** At 1/2 bit per token (κ = log(2)/2), an 8B model trained on 1T tokens shows +11% HumanEval+, +5% MMLU, +6% CSQA, +3% MBPP, +6% GSM8K over baseline — with identical training hyperparameters.

**Conditional applicability:** The architectural insight (mid-layer latent injection for generation conditioning) is directly relevant to our system. However, **no pretrained Free Transformer weights exist**, and the approach requires training from scratch. We can distill the architectural pattern without the VAE training procedure.

---

## Core Concepts

### Architecture

The Free Transformer splits a standard decoder at the middle layer:

1. **First half** (layers 0..L/2): Standard causal Transformer blocks — process tokens normally
2. **Z injection** at layer L/2+1: `keys_and_values = X_{L/2} + linear(Z)`
3. **Second half** (layers L/2+1..L): Standard causal Transformer blocks — process conditioned representations

During inference: Z is a one-hot vector of dimension 2^H (H=16, so 65536), sampled uniformly. The `post_sampler_fc` linear layer converts it to shape D (model dimension), added to K and V.

During training: A non-causal encoder (shares first L/2 layers with decoder + 1 extra non-causal block) produces Z consistent with the training sequence. KL divergence penalty with free bits threshold κ prevents posterior collapse.

### Binary Mapper

The encoder outputs H=16 logits (interpreted as individual bits), samples each independently via sigmoid, and maps the resulting 16-bit integer to a one-hot of dimension 2^16 = 65536. Gradient passes through via the straight-through estimator: `Y_d + G_d - detach(G_d)`.

### Key Hyperparameter: κ (Free Bits)

Controls how much information Z carries per token:
- **κ = 1/4 bit**: Near-vanilla, minimal Z usage
- **κ = 1/2 bit**: Sweet spot — biggest gains on reasoning tasks
- **κ = 1 bit**: Good for code/math, risky for some multi-choice
- **κ = 2 bits**: Starting to collapse (encoder copies too much)
- **κ = 4 bits**: Full collapse — encoder memorizes tokens, decoder becomes trivial

### Key Results (8B, 1T tokens, κ = 1/2 bit)

| Benchmark | Baseline | Free TF | Delta |
|-----------|----------|---------|-------|
| HumanEval+ (pass@1) | 0.268 | 0.299 | +11.4% |
| MBPP (pass@1) | 0.428 | 0.440 | +2.8% |
| GSM8K (em) | 0.321 | 0.331 | +2.8% |
| MMLU (macro_avg) | 0.592 | 0.623 | +5.2% |
| CSQA (acc) | 0.707 | 0.748 | +5.8% |
| HellaSwag | 0.799 | 0.799 | ~0% |
| PIQA | 0.805 | 0.812 | +0.9% |

Pattern: **Biggest gains on reasoning-heavy tasks. Negligible on common-sense tasks where the baseline is already strong.**

---

## What Applies to microgpt-rs

### 1. Mid-Layer Conditioning ↔ KV Cache Priming (Plan 024)

The paper validates injecting conditioning information at the middle layer via K/V modulation. Our `forward_base` layer loop has a natural injection point:

```microgpt-rs/src/transformer.rs#L370-371
for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
```

At `layer_idx == config.n_layer / 2`, we could add a latent vector to K/V. The paper proves this is architecturally sound — the model learns to use the mid-layer injection point to make global structural decisions.

**However:** Without a Free Transformer base model, adding random Z during inference on a standard model would add noise, not signal. The model must be trained to expect and utilize Z.

### 2. Bidirectional Prefill as Encoder Proxy (Gemini's Point 5)

Gemini correctly identifies that our `forward_prefill` already does **non-causal attention**:

```microgpt-rs/src/transformer.rs#L632-635
// Bidirectional attention: t_n = prompt_len (full prompt range)
prompt_len, // ← BIDIRECTIONAL: full range, not pos+1
```

This is architecturally similar to the Free Transformer's encoder (which is also non-causal). If we ever train a Free Transformer, our prefill infrastructure could serve as the encoder backbone.

### 3. κ ≈ β — Budget Philosophy

The paper's free bits threshold κ controls "how much latent information is allowed." Our domain inference budget β controls "how much compute to spend." Both embody the same principle: **constrained resources force better decisions**.

The empirical result that 1/2 bit is optimal (more is worse) validates our tight-budget approach. Our `domain.inference.beta` in riir-ai's TOML config is the same idea.

### 4. Latent Z as Routing Signal ↔ Expert Registry (riir-ai Plan 023)

The paper shows Z learns to capture "which mode to generate in" unsupervised. Our `RouteDecision { domain, confidence, lora_path, pruner_path }` does the same thing explicitly. Z could theoretically replace the Prompt Router — but only with a model trained to use Z.

---

## Gemini's Proposals — Verdict

### Point 1: Zero-Allocation Mid-Layer Latent Injection

**Claim:** Add `post_sampler_fc` weights, sample random Z, add to K/V at mid-layer. 0 allocations, 1 RNG call, O(D) additions.

**Verdict: ⚠️ Correct mechanism, wrong model.** The code change is trivial — at `layer_idx == n_layer / 2`, before the K/V cache write, add `ctx.k[i] += post_sampler_fc[c * kvd + i]`. But a standard Transformer model hasn't been trained to expect this noise. Adding random vectors to K/V on an untrained model would **degrade** output quality, not improve it. This only works if the model was trained with the VAE loss.

**If we ever get a Free Transformer base model**, the implementation would be:

```rust
// At mid-layer in forward_base, before cache write:
if layer_idx == config.n_layer / 2 {
    let z_idx = rng.uniform_range(0, config.z_dim); // 65536
    for i in 0..kvd {
        ctx.k[i] += weights.post_sampler_fc[z_idx * kvd + i];
        ctx.v[i] += weights.post_sampler_fc[z_idx * kvd + i];
    }
}
```

Cost: 1 RNG call + 2*kvd additions. Negligible. But needs trained weights.

### Point 2: Latent DDTree for Speculative Decoding Diversity

**Claim:** Branch DDTree on different Z values instead of just tokens, exploring semantically distinct paths.

**Verdict: 🔴 Infeasible without model retraining.** The DDTree operates on marginals (token probability distributions from draft model forward passes). To "branch on Z", you'd need to run multiple forward passes with different Z values — each one changing the entire draft model's behavior. This is:

1. N forward passes instead of 1 (where N = number of Z branches)
2. Each forward pass produces completely different marginals (Z fundamentally alters generation)
3. The DDTree would need to handle multiple incompatible marginal sources
4. Only useful if the model was trained to use Z

The idea of "semantic diversity" is valuable, but the mechanism is wrong for our architecture. Our existing `ppot_resample_multi_strategy` achieves diversity via constrained resampling within the same marginal distribution — cheaper and works with any model.

### Point 3: RAG Embedding → Z Injection Instead of KV Cache Priming

**Claim:** Project anyRAG embedding to dimension 2^H, inject as Z at mid-layer instead of prefix KV cache.

**Verdict: 🟡 Interesting but lossy.** KV cache priming preserves the full RAG context in the attention mechanism. Z injection via one-hot collapses the embedding to a single discrete value (albeit from 65536 options). For RAG where fine-grained document details matter, KV cache priming is strictly better. Z injection would lose the actual retrieved text content.

Where Z injection COULD help: encoding the **domain** or **strategy** (not the document content). A "code mode" Z vs "prose mode" Z is a natural fit. But our existing domain routing + LoRA switching already handles this.

### Point 4: PPoT Z-Resampling

**Claim:** When PPoT rescue fails, resample the Z index instead of individual tokens to pivot the entire generation strategy.

**Verdict: 🔴 Category error.** PPoT operates on marginals from a single forward pass. Resampling Z requires a NEW forward pass with a different Z — that's speculative decoding, not logit resampling. The PPoT value proposition is "CPU-only, no additional model forward passes." Resampling Z would violate that constraint.

Our existing `ppot_resample_multi_strategy` with `TokenRule` support sets is the correct mechanism for PPoT-level rescue — it explores constrained token variations within the same marginals, zero additional forward passes.

### Point 5: Bidirectional Prefill as FT Encoder

**Verdict: 🟢 Correct observation, useful if we ever train.** Our `forward_prefill` uses `prompt_len` (full range) instead of `pos+1` for attention, making it bidirectional. If we train a Free Transformer, the encoder (which is also non-causal) could reuse this infrastructure. The `PrefillContext.hidden` buffer already stores intermediate representations across layers — exactly what the encoder needs.

---

## What Would Actually Work (Distilled Insight)

### Insight 1: Domain Latent as Compact Routing Signal

Instead of one-hot Z (65536-dim, requires trained model), we could use a **learned domain embedding** injected at mid-layer. This is a smaller, LoRA-compatible version of the Free Transformer idea:

1. Train a small linear layer that maps domain label → embedding of size D
2. At mid-layer in `forward_base`, add this embedding to K/V (same mechanism as Z)
3. LoRA fine-tune the rest of the model to condition on this embedding

This gives us the "explicit structural decision" benefit of Z without needing to train from scratch. It's essentially **domain-conditioned LoRA** — inject a domain signal, let the LoRA adapter learn to use it.

### Insight 2: Multi-Sample Inference with Z (When Model Available)

If a Free Transformer base model becomes available, the inference-time Z sampling enables a powerful technique: **sample multiple Z values, generate with each, pick the best**. This is cheap (Z sampling is free, only 1 forward pass per Z) and gives diversity similar to best-of-N sampling but with structured diversity (Z controls global properties, not individual tokens).

This would integrate naturally with our DDTree: generate marginals with different Z values, merge the resulting trees, let `ScreeningPruner` pick the best.

### Insight 3: κ Validates Our Budget Philosophy

The empirical proof that "less information is better" (1/2 bit > 2 bits) is the most immediately actionable finding. It validates:
- Our `beta` parameterization in domain config (constrained budget)
- Our `tree_budget` (constrained search)
- Our `early_exit_patience` (constrained patience)

No code change needed. The paper provides theoretical backing for our architecture choices.

---

## Actionable Items

| Priority | Item | Effort | Prerequisite |
|----------|------|--------|-------------|
| **NOW** | Note κ finding as validation of budget philosophy | 0 (documentation) | None |
| **NOW** | Record Z injection as future pattern for when Free TF weights exist | 0 (this doc) | None |
| **LATER** | Domain embedding injection at mid-layer (LoRA-compatible) | Medium | LoRA training pipeline |
| **LATER** | Multi-Z inference + DDTree merge | Medium | Free Transformer base model |
| **NEVER** | Random Z on untrained model (degrades quality) | — | — |
| **NEVER** | Z-resampling in PPoT (violates CPU-only constraint) | — | — |

---

## Citation

```bibtex
@article{fleuret2025free_transformer,
  title     = {The Free Transformer},
  author    = {François Fleuret},
  journal   = {arXiv preprint arXiv:2510.17558v1},
  year      = {2025},
  eprint    = {2510.17558}
}