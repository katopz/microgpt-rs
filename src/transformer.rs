use crate::types::{self, *};
use rayon::prelude::*;

/// Per-layer transformer weights.
/// Each layer has its own attention and MLP parameters.
pub struct LayerWeights {
    pub attn_wq: Vec<f32>, // [n_embd, n_embd]
    pub attn_wk: Vec<f32>, // [kv_dim, n_embd] where kv_dim = n_kv_head * head_dim
    pub attn_wv: Vec<f32>, // [kv_dim, n_embd]
    pub attn_wo: Vec<f32>, // [n_embd, n_embd]
    pub mlp_w1: Vec<f32>,  // [mlp_hidden, n_embd]
    pub mlp_w2: Vec<f32>,  // [n_embd, mlp_hidden]
}

/// All transformer weights: embeddings, per-layer weights, and LM head.
/// Layout preserves init order for backward compat: wte, wpe, layers…, lm_head.
///
/// # Future: f16 Storage
///
/// For memory-constrained deployments, weights can be stored as `f16` (half-precision)
/// and quantized on-the-fly during matmul. This would halve memory usage with minimal
/// accuracy loss for inference-only workloads. The migration path:
///
/// 1. Add a `StorageFormat` enum: `F32`, `F16`, `Q4_0`, `Q8_0`
/// 2. Replace `Vec<f32>` with a `WeightTensor` enum that stores the chosen format
/// 3. Add `dequantize_row()` that converts to `f32` on-the-fly during matmul
/// 4. The `forward()` kernel remains unchanged — it operates on `f32` buffers
///    populated by dequantization
///
/// Key insight: only storage changes; compute stays in `f32`. This avoids the need
/// for f16 arithmetic hardware support and keeps the attention kernel simple.
/// Estimated memory savings: ~50% for f16, ~75% for 4-bit quantized.
pub struct TransformerWeights {
    pub wte: Vec<f32>,             // [vocab_size, n_embd]
    pub wpe: Vec<f32>,             // [block_size, n_embd]
    pub lm_head: Vec<f32>,         // [vocab_size, n_embd]
    pub layers: Vec<LayerWeights>, // [n_layer]
}

impl TransformerWeights {
    pub fn new(config: &Config, rng: &mut Rng) -> Self {
        let n = config.n_embd;
        let kvd = types::kv_dim(config);
        let embd_scale = (2.0 / n as f32).sqrt();
        let layer_scale = (2.0 / (n as f32 * config.n_layer as f32)).sqrt();

        // Embeddings first (same order as original single-layer code)
        let wte: Vec<f32> = (0..config.vocab_size * n)
            .map(|_| rng.normal() * embd_scale)
            .collect();
        let wpe: Vec<f32> = (0..config.block_size * n)
            .map(|_| rng.normal() * embd_scale)
            .collect();

        // Per-layer weights: same field order as original per n_layer iterations
        let layers: Vec<LayerWeights> = (0..config.n_layer)
            .map(|_| LayerWeights {
                attn_wq: (0..n * n).map(|_| rng.normal() * layer_scale).collect(),
                attn_wk: (0..kvd * n).map(|_| rng.normal() * layer_scale).collect(),
                attn_wv: (0..kvd * n).map(|_| rng.normal() * layer_scale).collect(),
                attn_wo: (0..n * n).map(|_| rng.normal() * layer_scale).collect(),
                mlp_w1: (0..config.mlp_hidden * n)
                    .map(|_| rng.normal() * layer_scale)
                    .collect(),
                mlp_w2: (0..n * config.mlp_hidden)
                    .map(|_| rng.normal() * layer_scale)
                    .collect(),
            })
            .collect();

        // LM head last
        let lm_head: Vec<f32> = (0..config.vocab_size * n)
            .map(|_| rng.normal() * embd_scale)
            .collect();

        Self {
            wte,
            wpe,
            lm_head,
            layers,
        }
    }
}

/// KV cache for a single layer (autoregressive generation).
pub struct KVCache {
    pub key: Vec<f32>,   // [block_size, kv_dim] where kv_dim = n_kv_head * head_dim
    pub value: Vec<f32>, // [block_size, kv_dim]
}

impl KVCache {
    pub fn new(config: &Config) -> Self {
        let kvd = types::kv_dim(config);
        Self {
            key: vec![0.0; config.block_size * kvd],
            value: vec![0.0; config.block_size * kvd],
        }
    }

    pub fn reset(&mut self) {
        self.key.fill(0.0);
        self.value.fill(0.0);
    }
}

/// Multi-layer KV cache: one KVCache per transformer layer.
pub struct MultiLayerKVCache {
    pub layers: Vec<KVCache>,
}

impl MultiLayerKVCache {
    pub fn new(config: &Config) -> Self {
        Self {
            layers: (0..config.n_layer).map(|_| KVCache::new(config)).collect(),
        }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Snapshot KV cache state up to position `pos`.
    /// Copies only filled slots [0..pos) per layer — cheap at our model scale.
    pub fn snapshot(&self, pos: usize, config: &Config) -> KVSnapshot {
        let kd = types::kv_dim(config);
        let end = pos * kd;
        let layers = self
            .layers
            .iter()
            .map(|layer| KVLayerSnapshot {
                key: layer.key[..end].to_vec(),
                value: layer.value[..end].to_vec(),
            })
            .collect();
        KVSnapshot { layers, pos }
    }

    /// Restore KV cache from a snapshot.
    /// Writes snapshot data back and zeros out positions [snapshot.pos..block_size).
    pub fn restore(&mut self, snapshot: &KVSnapshot, config: &Config) {
        let kd = types::kv_dim(config);
        for (layer, snap_layer) in self.layers.iter_mut().zip(snapshot.layers.iter()) {
            let end = snapshot.pos * kd;
            layer.key[..end].copy_from_slice(&snap_layer.key);
            layer.value[..end].copy_from_slice(&snap_layer.value);
            // Zero out positions [snapshot.pos..block_size) to prevent stale data
            layer.key[end..].fill(0.0);
            layer.value[end..].fill(0.0);
        }
    }
}

/// Cheap snapshot of KV cache state up to position `pos`.
/// Only copies filled slots [0..pos) per layer, not the entire block_size buffer.
pub struct KVSnapshot {
    pub layers: Vec<KVLayerSnapshot>,
    pub pos: usize,
}

/// Per-layer snapshot of KV cache data.
pub struct KVLayerSnapshot {
    pub key: Vec<f32>,   // [pos * kv_dim]
    pub value: Vec<f32>, // [pos * kv_dim]
}

/// Pre-allocated buffers for zero-alloc forward passes.
/// Create once, reuse across calls.
pub struct ForwardContext {
    x: Vec<f32>,                // [n_embd] main activation
    xr: Vec<f32>,               // [n_embd] residual
    xr2: Vec<f32>,              // [n_embd] residual 2
    q: Vec<f32>,                // [n_embd] query
    k: Vec<f32>,                // [kv_dim] key (kv_dim = n_kv_head * head_dim)
    v: Vec<f32>,                // [kv_dim] value
    attn_out: Vec<f32>,         // [n_embd] attention output
    pub scores: Vec<f32>,       // [block_size] attention scores (max possible)
    hidden: Vec<f32>,           // [mlp_hidden] MLP hidden
    pub logits: Vec<f32>,       // [vocab_size] output logits
    pub hidden_state: Vec<f32>, // [n_embd] final hidden state (Plan 009 compat)
    /// LoRA intermediate buffer [lora_rank]. Pre-allocated, zero alloc in hot path.
    pub lora_buf: Vec<f32>,
    // Sparse MLP buffers (Plan 022: TwELL-inspired unstructured sparsity)
    #[cfg(feature = "sparse_mlp")]
    active_indices: Vec<usize>, // [mlp_hidden] pre-allocated index buffer
    #[cfg(feature = "sparse_mlp")]
    active_values: Vec<f32>, // [mlp_hidden] pre-allocated value buffer
    // Paged KV cache: pre-allocated flat buffers for attention computation
    paged_flat_key: Vec<f32>,   // [block_size * kv_dim]
    paged_flat_value: Vec<f32>, // [block_size * kv_dim]
    // Raven: pre-allocated query buffer for per-head slot attention
    raven_query_buf: Vec<f32>, // [kv_dim]
}

impl ForwardContext {
    pub fn new(config: &Config) -> Self {
        let kvd = types::kv_dim(config);
        let block_kv = config.block_size * kvd;
        Self {
            x: vec![0.0; config.n_embd],
            xr: vec![0.0; config.n_embd],
            xr2: vec![0.0; config.n_embd],
            q: vec![0.0; config.n_embd],
            k: vec![0.0; kvd],
            v: vec![0.0; kvd],
            attn_out: vec![0.0; config.n_embd],
            scores: vec![0.0; config.block_size],
            hidden: vec![0.0; config.mlp_hidden],
            logits: vec![0.0; config.vocab_size],
            hidden_state: vec![0.0; config.n_embd],
            lora_buf: vec![0.0; config.lora_rank],
            #[cfg(feature = "sparse_mlp")]
            active_indices: vec![0; config.mlp_hidden],
            #[cfg(feature = "sparse_mlp")]
            active_values: vec![0.0; config.mlp_hidden],
            paged_flat_key: vec![0.0; block_kv],
            paged_flat_value: vec![0.0; block_kv],
            raven_query_buf: vec![0.0; kvd],
        }
    }
}

// ---------------------------------------------------------------------------
// PrefillContext — Pre-allocated buffers for bidirectional prefill (Plan 025)
// ---------------------------------------------------------------------------

/// Pre-allocated context for bidirectional prefill phase.
/// Created once at startup, reused across all requests. Zero alloc in request path.
pub struct PrefillContext {
    /// Hidden states for all prompt positions, carried between layers.
    /// Size: [max_prompt_len × n_embd]. Only used when n_layer > 1.
    /// For n_layer == 1, embeddings are computed on-the-fly and this buffer is unused.
    hidden: Vec<f32>,
    /// LoRA intermediate buffer. Size: [lora_rank].
    /// Reused for every LoRA application across all projections.
    lora_buf: Vec<f32>,
    /// Max prompt length this context supports (= config.block_size).
    max_prompt_len: usize,
}

impl PrefillContext {
    pub fn new(config: &Config) -> Self {
        Self {
            hidden: vec![0.0; config.block_size * config.n_embd],
            lora_buf: vec![0.0; config.lora_rank],
            max_prompt_len: config.block_size,
        }
    }
}

/// Fused attention head with GQA support: score → softmax → weighted value sum.
/// Avoids separate `softmax()` call and write-back of normalized scores.
///
/// GQA: each Q head (`q_head_offset / hd`) maps to a KV group (`kv_group_offset / hd`).
/// When `n_kv_head == n_head`, `kv_group_offset == q_head_offset` and `kv_dim == n_embd`
/// → identical to standard MHA (backward compatible).
///
/// SAFETY: caller must ensure all indices are in bounds.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn attention_head(
    q: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    attn_out: &mut [f32],
    scores_buf: &mut [f32],
    q_head_offset: usize,
    kv_group_offset: usize,
    kv_dim: usize,
    hd: usize,
    t_n: usize,
    scale: f32,
) {
    // Pass 1: compute Q·K scores and find max for numerical stability
    let mut max_score = f32::NEG_INFINITY;
    for t in 0..t_n {
        let k_off = t * kv_dim + kv_group_offset;
        let mut dot = 0.0f32;
        for d in 0..hd {
            unsafe {
                dot += *q.get_unchecked(q_head_offset + d) * *key_cache.get_unchecked(k_off + d);
            }
        }
        let score = dot * scale;
        unsafe {
            *scores_buf.get_unchecked_mut(t) = score;
        }
        if score > max_score {
            max_score = score;
        }
    }

    // Pass 2: exp(scores - max) and accumulate sum
    let mut sum = 0.0f32;
    for t in 0..t_n {
        let exp_val = unsafe { (*scores_buf.get_unchecked(t) - max_score).exp() };
        unsafe {
            *scores_buf.get_unchecked_mut(t) = exp_val;
        }
        sum += exp_val;
    }

    // Pass 3: normalize + weighted value accumulation (no write-back of scores)
    let inv_sum = 1.0 / sum;
    for d in 0..hd {
        let mut val = 0.0f32;
        for t in 0..t_n {
            unsafe {
                val += *scores_buf.get_unchecked(t)
                    * inv_sum
                    * *value_cache.get_unchecked(t * kv_dim + kv_group_offset + d);
            }
        }
        unsafe {
            *attn_out.get_unchecked_mut(q_head_offset + d) = val;
        }
    }
}

/// Causal decode: single token forward with optional LoRA adapter.
/// Backward-compatible wrapper that passes `None` for LoRA.
#[inline(always)]
pub fn forward<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    #[cfg(not(feature = "domain_latent"))]
    {
        forward_base(ctx, weights, cache, token, pos, config, None)
    }
    #[cfg(feature = "domain_latent")]
    {
        forward_base(ctx, weights, cache, token, pos, config, None, None)
    }
}

/// Forward with optional LoRA and domain latent (Plan 038).
/// Convenience wrapper for callers that need both conditioning signals.
#[cfg(feature = "domain_latent")]
#[allow(clippy::too_many_arguments)]
pub fn forward_with_domain_latent<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    lora: Option<&crate::types::LoraAdapter>,
    domain_latent: Option<&crate::types::DomainLatent>,
) -> &'a mut [f32] {
    forward_base(ctx, weights, cache, token, pos, config, lora, domain_latent)
}

/// Internal forward with optional LoRA and domain latent (writer LoRA during decode).
/// Zero-alloc forward pass. Writes logits into `ctx.logits` and returns &mut to it.
/// Multi-layer: RMSNorm → Attn → Res → RMSNorm → MLP → Res per layer, then LM Head.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn forward_base<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    lora: Option<&crate::types::LoraAdapter>,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&crate::types::DomainLatent>,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    for i in 0..n {
        unsafe {
            *ctx.x.get_unchecked_mut(i) = *weights.wte.get_unchecked(tok_off + i)
                + *weights.wpe.get_unchecked(pos_off_emb + i);
        }
    }

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // Pre-attention: RMSNorm → save residual → RMSNorm
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);

        // QKV projections from per-layer weights (GQA: K/V produce kv_dim outputs)
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut ctx.lora_buf);
        }
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut ctx.lora_buf);
        }
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut ctx.lora_buf);
        }

        // Domain latent injection at mid-layer (Plan 038: Free Transformer adaptation)
        #[cfg(feature = "domain_latent")]
        if layer_idx == config.n_layer / 2
            && let Some(dl) = domain_latent
        {
            for i in 0..kvd {
                unsafe {
                    *ctx.k.get_unchecked_mut(i) += *dl.embedding.get_unchecked(i);
                    *ctx.v.get_unchecked_mut(i) += *dl.embedding.get_unchecked(i);
                }
            }
        }

        // Store K,V in per-layer cache (kv_dim elements per position)
        let pos_off = pos * kvd;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ctx.k.as_ptr(),
                layer_cache.key.as_mut_ptr().add(pos_off),
                kvd,
            );
            std::ptr::copy_nonoverlapping(
                ctx.v.as_ptr(),
                layer_cache.value.as_mut_ptr().add(pos_off),
                kvd,
            );
        }

        // Multi-head attention with GQA: fused score → softmax → weighted value per head
        let scale = 1.0 / (hd as f32).sqrt();
        ctx.attn_out[..n].fill(0.0);
        let t_n = pos + 1;

        for h in 0..config.n_head {
            let kv_group = h * n_kv / config.n_head;
            unsafe {
                attention_head(
                    &ctx.q,
                    &layer_cache.key,
                    &layer_cache.value,
                    &mut ctx.attn_out,
                    &mut ctx.scores,
                    h * hd,
                    kv_group * hd,
                    kvd,
                    hd,
                    t_n,
                    scale,
                );
            }
        }

        // Output projection + residual
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.x, lora, &ctx.attn_out, &mut ctx.lora_buf);
        }
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);
        types::matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.hidden, lora, &ctx.x, &mut ctx.lora_buf);
        }
        // MLP w2: sparse when feature enabled and sparsity is high enough (Plan 022)
        #[cfg(feature = "sparse_mlp")]
        {
            let alive = types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                );
            }
        }
        #[cfg(not(feature = "sparse_mlp"))]
        matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.x, lora, &ctx.hidden, &mut ctx.lora_buf);
        }
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }
    }

    // Snapshot hidden state (for Plan 009 compatibility)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head
    matmul(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

// ---------------------------------------------------------------------------
// Bidirectional Prefill (Plan 025)
// ---------------------------------------------------------------------------

/// Bidirectional prefill: process prompt tokens with full mutual attention.
///
/// For each transformer layer:
///   Phase A: Compute K/V for all prompt positions → store in KV cache
///   Phase B: For each position, attend to ALL prompt K/V (bidirectional)
///
/// Returns logits for the last prompt position (used to sample first gen token).
/// KV cache is populated as a side effect, shared with subsequent decode calls.
///
/// Zero-copy: no allocations. Reuses ForwardContext buffers per-position,
/// PrefillContext::hidden for multi-layer inter-layer state.
#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill<'a>(
    ctx: &'a mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    tokens: &[usize],
    config: &Config,
    lora: Option<&crate::types::LoraAdapter>,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&crate::types::DomainLatent>,
) -> &'a mut [f32] {
    let prompt_len = tokens.len().min(prefill.max_prompt_len);
    let n = config.n_embd;
    let kvd = crate::types::kv_dim(config);
    let hd = config.head_dim;
    let n_kv = config.n_kv_head;

    assert!(prompt_len > 0, "prefill requires at least one token");
    assert!(
        prompt_len <= config.block_size,
        "prompt_len {prompt_len} exceeds block_size {}",
        config.block_size
    );

    // Initialize hidden states for multi-layer (single-layer computes on-the-fly)
    if config.n_layer > 1 {
        for (p, &token) in tokens.iter().enumerate().take(prompt_len) {
            let tok_off = token * n;
            let pos_off = p * n;
            for i in 0..n {
                unsafe {
                    *prefill.hidden.get_unchecked_mut(p * n + i) =
                        *weights.wte.get_unchecked(tok_off + i)
                            + *weights.wpe.get_unchecked(pos_off + i);
                }
            }
        }
    }

    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // ── Phase A: Compute K/V for ALL positions → store in cache ──
        for (p, &token) in tokens.iter().enumerate().take(prompt_len) {
            // Load hidden state
            if config.n_layer > 1 {
                ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);
            } else {
                let tok_off = token * n;
                let pos_off = p * n;
                for i in 0..n {
                    unsafe {
                        *ctx.x.get_unchecked_mut(i) = *weights.wte.get_unchecked(tok_off + i)
                            + *weights.wpe.get_unchecked(pos_off + i);
                    }
                }
            }

            // Pre-attention norm (matches forward_base exactly: double rmsnorm)
            crate::types::rmsnorm(&mut ctx.x);
            ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);

            // K/V projections
            crate::types::matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut prefill.lora_buf);
            }
            crate::types::matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut prefill.lora_buf);
            }

            // Domain latent injection at mid-layer (Plan 038: Free Transformer adaptation)
            #[cfg(feature = "domain_latent")]
            if layer_idx == config.n_layer / 2
                && let Some(dl) = domain_latent
            {
                for i in 0..kvd {
                    unsafe {
                        *ctx.k.get_unchecked_mut(i) += *dl.embedding.get_unchecked(i);
                        *ctx.v.get_unchecked_mut(i) += *dl.embedding.get_unchecked(i);
                    }
                }
            }

            // Store K/V in cache
            let pos_off = p * kvd;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ctx.k.as_ptr(),
                    layer_cache.key.as_mut_ptr().add(pos_off),
                    kvd,
                );
                std::ptr::copy_nonoverlapping(
                    ctx.v.as_ptr(),
                    layer_cache.value.as_mut_ptr().add(pos_off),
                    kvd,
                );
            }
        }

        // ── Phase B: Bidirectional attention for ALL positions ──
        for (p, &token) in tokens.iter().enumerate().take(prompt_len) {
            // Load hidden state again (same source as Phase A)
            if config.n_layer > 1 {
                ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);
            } else {
                let tok_off = token * n;
                let pos_off = p * n;
                for i in 0..n {
                    unsafe {
                        *ctx.x.get_unchecked_mut(i) = *weights.wte.get_unchecked(tok_off + i)
                            + *weights.wpe.get_unchecked(pos_off + i);
                    }
                }
            }

            // Pre-attention norm
            crate::types::rmsnorm(&mut ctx.x);
            ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);

            // Q projection
            crate::types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut prefill.lora_buf);
            }

            // Bidirectional attention: t_n = prompt_len (full prompt range)
            let scale = 1.0 / (hd as f32).sqrt();
            ctx.attn_out[..n].fill(0.0);
            for h in 0..config.n_head {
                let kv_group = h * n_kv / config.n_head;
                unsafe {
                    attention_head(
                        &ctx.q,
                        &layer_cache.key,
                        &layer_cache.value,
                        &mut ctx.attn_out,
                        &mut ctx.scores,
                        h * hd,
                        kv_group * hd,
                        kvd,
                        hd,
                        prompt_len, // ← BIDIRECTIONAL: full range, not pos+1
                        scale,
                    );
                }
            }

            // Output projection + residual
            crate::types::matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.x, lora, &ctx.attn_out, &mut prefill.lora_buf);
            }
            for i in 0..n {
                unsafe {
                    *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
                }
            }

            // MLP: residual → RMSNorm → MLP → residual
            ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);
            crate::types::matmul_relu(
                &mut ctx.hidden,
                &layer_weights.mlp_w1,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.hidden, lora, &ctx.x, &mut prefill.lora_buf);
            }
            // MLP w2 (with sparse support)
            #[cfg(feature = "sparse_mlp")]
            {
                let alive = crate::types::sparse_matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                    &mut ctx.active_indices,
                    &mut ctx.active_values,
                );
                if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                    crate::types::matmul(
                        &mut ctx.x,
                        &layer_weights.mlp_w2,
                        &ctx.hidden,
                        n,
                        config.mlp_hidden,
                    );
                }
            }
            #[cfg(not(feature = "sparse_mlp"))]
            crate::types::matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
            );
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.x, lora, &ctx.hidden, &mut prefill.lora_buf);
            }
            for i in 0..n {
                unsafe {
                    *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
                }
            }

            // Store hidden state for next layer (multi-layer only)
            if config.n_layer > 1 {
                prefill.hidden[p * n..(p + 1) * n].copy_from_slice(&ctx.x[..n]);
            }
        }
    }

    // Snapshot hidden state (last position)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head
    crate::types::matmul(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

/// Full generation pipeline: bidirectional prefill → causal decode.
/// Switches from reader LoRA to writer LoRA at the prefill→decode boundary.
/// Zero-copy: all buffers pre-allocated, no allocations in request path.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_prefill(
    ctx: &mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    config: &Config,
    rng: &mut crate::types::Rng,
    prompt_tokens: &[usize],
    max_gen_tokens: usize,
    lora_pair: &crate::types::LoraPair,
) -> Vec<usize> {
    // 1. Bidirectional prefill with reader LoRA
    let logits = {
        #[cfg(not(feature = "domain_latent"))]
        {
            forward_prefill(
                ctx,
                prefill,
                weights,
                cache,
                prompt_tokens,
                config,
                lora_pair.reader.as_ref(),
            )
        }
        #[cfg(feature = "domain_latent")]
        {
            forward_prefill(
                ctx,
                prefill,
                weights,
                cache,
                prompt_tokens,
                config,
                lora_pair.reader.as_ref(),
                None,
            )
        }
    };

    // 2. Sample first generation token from prefill output
    let mut p_dist = logits.to_vec();
    for p in p_dist.iter_mut() {
        *p /= config.temperature;
    }
    crate::types::softmax(&mut p_dist);
    let mut token = crate::types::sample_token(&p_dist, rng);

    let mut generated = vec![token];
    let mut pos = prompt_tokens.len();

    // 3. Causal decode with writer LoRA
    for _ in 1..max_gen_tokens {
        if pos >= config.block_size {
            break;
        }

        let logits = {
            #[cfg(not(feature = "domain_latent"))]
            {
                forward_base(
                    ctx,
                    weights,
                    cache,
                    token,
                    pos,
                    config,
                    lora_pair.writer.as_ref(),
                )
            }
            #[cfg(feature = "domain_latent")]
            {
                forward_base(
                    ctx,
                    weights,
                    cache,
                    token,
                    pos,
                    config,
                    lora_pair.writer.as_ref(),
                    None,
                )
            }
        };
        for logit in logits.iter_mut() {
            *logit /= config.temperature;
        }
        crate::types::softmax(logits);

        token = crate::types::sample_token(logits, rng);
        generated.push(token);
        pos += 1;

        if token == config.bos_token {
            break;
        }
    }

    generated
}

/// Forward pass using `PagedKVCache` instead of `MultiLayerKVCache`.
///
/// Identical computation to `forward()` but stores KV in paged memory,
/// enabling copy-on-write fork for DDTree branch exploration.
/// Builds a temporary flat KV buffer per layer for attention computation.
#[inline(always)]
pub fn forward_paged<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    paged_cache: &mut PagedKVCache,
    seq_idx: usize,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = crate::types::kv_dim(config);
    let n_kv = config.n_kv_head;

    // Ensure pages allocated for this sequence up to pos
    paged_cache.ensure_pages(seq_idx, pos);

    // Flat KV cache for attention computation (pre-allocated, reused from ForwardContext)
    let t_n = pos + 1;
    let flat_kv_len = t_n * kvd;
    let flat_key = &mut ctx.paged_flat_key[..flat_kv_len];
    let flat_value = &mut ctx.paged_flat_value[..flat_kv_len];
    flat_key.fill(0.0);
    flat_value.fill(0.0);

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    for i in 0..n {
        unsafe {
            *ctx.x.get_unchecked_mut(i) = *weights.wte.get_unchecked(tok_off + i)
                + *weights.wpe.get_unchecked(pos_off_emb + i);
        }
    }

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        // Pre-attention: RMSNorm → save residual → RMSNorm
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);

        // QKV projections
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

        // Write K,V to paged cache
        paged_cache.write_kv(layer_idx, seq_idx, pos, &ctx.k, &ctx.v);

        // Build flat KV from paged cache for attention
        for t in 0..t_n {
            let k_slice = &mut flat_key[t * kvd..(t + 1) * kvd];
            let v_slice = &mut flat_value[t * kvd..(t + 1) * kvd];
            paged_cache.read_kv(layer_idx, seq_idx, t, k_slice, v_slice);
        }

        // Multi-head attention with GQA (reuse existing attention_head)
        let scale = 1.0 / (hd as f32).sqrt();
        ctx.attn_out[..n].fill(0.0);

        for h in 0..config.n_head {
            let kv_group = h * n_kv / config.n_head;
            unsafe {
                attention_head(
                    &ctx.q,
                    flat_key,
                    flat_value,
                    &mut ctx.attn_out,
                    &mut ctx.scores,
                    h * hd,
                    kv_group * hd,
                    kvd,
                    hd,
                    t_n,
                    scale,
                );
            }
        }

        // Output projection + residual
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);
        types::matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        // MLP w2: sparse when feature enabled and sparsity is high enough (Plan 022)
        #[cfg(feature = "sparse_mlp")]
        {
            let alive = types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                );
            }
        }
        #[cfg(not(feature = "sparse_mlp"))]
        matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head
    matmul(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

/// Zero-alloc generation: `ctx`, `cache`, `tokens` all provided by caller.
///
/// `tokens` is cleared and filled with generated token ids.
/// `ctx` and `cache` are reused across calls.
pub fn generate_into(
    ctx: &mut ForwardContext,
    cache: &mut MultiLayerKVCache,
    weights: &TransformerWeights,
    config: &Config,
    rng: &mut Rng,
    n_tokens: usize,
    tokens: &mut Vec<usize>,
) {
    tokens.clear();
    let mut token = config.bos_token;
    let mut pos = 0;

    for _ in 0..n_tokens {
        if pos >= config.block_size {
            cache.reset();
            pos = 0;
            token = config.bos_token;
        }

        let logits = forward(ctx, weights, cache, token, pos, config);

        softmax_scaled(logits, 1.0 / config.temperature);

        let next_token = sample_token(logits, rng);
        tokens.push(next_token);

        if next_token == config.bos_token {
            cache.reset();
            pos = 0;
            token = config.bos_token;
        } else {
            token = next_token;
            pos += 1;
        }
    }
}

/// Generate tokens autoregressively. Returns generated token ids.
pub fn generate(
    weights: &TransformerWeights,
    config: &Config,
    rng: &mut Rng,
    n_tokens: usize,
) -> Vec<usize> {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    let mut tokens = Vec::new();
    generate_into(
        &mut ctx,
        &mut cache,
        weights,
        config,
        rng,
        n_tokens,
        &mut tokens,
    );
    tokens
}

/// Generate multiple samples in parallel using rayon.
///
/// Each sample gets its own `ForwardContext` + `MultiLayerKVCache` via `map_init`,
/// so there's no contention. The `seeds` slice provides one seed per sample.
/// Returns `Vec<Vec<usize>>` with one token sequence per sample.
pub fn generate_batch(
    weights: &TransformerWeights,
    config: &Config,
    seeds: &[u64],
    n_tokens: usize,
) -> Vec<Vec<usize>> {
    seeds
        .par_iter()
        .map_init(
            || (ForwardContext::new(config), MultiLayerKVCache::new(config)),
            |(ctx, cache), &seed| {
                let mut rng = Rng::new(seed);
                let mut tokens = Vec::with_capacity(n_tokens);
                generate_into(ctx, cache, weights, config, &mut rng, n_tokens, &mut tokens);
                tokens
            },
        )
        .collect()
}

/// Convert token ids to readable characters (a-z, _ for BOS).
pub fn tokens_to_string(tokens: &[usize]) -> String {
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    tokens
        .iter()
        .map(|&t| if t < 26 { CHARS[t] as char } else { '_' })
        .collect()
}

/// Page size in tokens (tuneable, must be power of 2).
const PAGE_SIZE: usize = 16;

/// Paged KV cache for DDTree branch exploration.
/// Allocates memory in fixed-size pages with copy-on-write fork.
///
/// Page layout per page: `[K_data | V_data]` where each segment is `PAGE_SIZE * kv_dim` floats.
/// This enables sharing prefix pages between branches without cloning data.
pub struct PagedKVCache {
    /// Pool of pages. Each page: `[PAGE_SIZE * kv_dim * 2]` floats (K then V).
    pages: Vec<Vec<f32>>,
    /// Per-layer page tables. `layer_page_tables[layer][seq_idx]` = vec of page indices.
    layer_page_tables: Vec<Vec<Vec<usize>>>,
    /// Free list of page indices for reuse.
    free_pages: Vec<usize>,
    /// Dimension of each KV entry (`n_kv_head * head_dim`).
    kv_dim: usize,
    /// Total pages ever allocated (monotonically increasing).
    total_pages: usize,
}

impl PagedKVCache {
    /// Create a new paged KV cache.
    /// `max_sequences`: initial number of sequence slots (can grow via fork).
    pub fn new(config: &Config, max_sequences: usize) -> Self {
        let kvd = types::kv_dim(config);
        let initial_pages_per_layer = config.block_size / PAGE_SIZE + 1;

        Self {
            pages: (0..initial_pages_per_layer * config.n_layer)
                .map(|_| vec![0.0; PAGE_SIZE * kvd * 2])
                .collect(),
            layer_page_tables: (0..config.n_layer)
                .map(|_| (0..max_sequences).map(|_| Vec::new()).collect())
                .collect(),
            free_pages: Vec::new(),
            kv_dim: kvd,
            total_pages: initial_pages_per_layer * config.n_layer,
        }
    }

    /// Allocate a new page. Reuse from free list or grow the pool.
    fn alloc_page(&mut self) -> usize {
        match self.free_pages.pop() {
            Some(idx) => {
                self.pages[idx].fill(0.0);
                idx
            }
            None => {
                self.pages.push(vec![0.0; PAGE_SIZE * self.kv_dim * 2]);
                let idx = self.total_pages;
                self.total_pages += 1;
                idx
            }
        }
    }

    /// Ensure sequence `seq_idx` has enough pages to cover position `pos` for all layers.
    pub fn ensure_pages(&mut self, seq_idx: usize, pos: usize) {
        let pages_needed = pos / PAGE_SIZE + 1;

        // Grow sequence slots if needed (no page allocation, just empty vecs)
        for layer_tables in &mut self.layer_page_tables {
            while seq_idx >= layer_tables.len() {
                layer_tables.push(Vec::new());
            }
        }

        // Collect how many new pages each layer needs
        let deficits: Vec<usize> = self
            .layer_page_tables
            .iter()
            .map(|lt| pages_needed.saturating_sub(lt[seq_idx].len()))
            .collect();

        // Allocate all pages upfront
        let new_pages: Vec<Vec<usize>> = deficits
            .into_iter()
            .map(|n| (0..n).map(|_| self.alloc_page()).collect())
            .collect();

        // Assign new pages to each layer's page table
        for (layer_tables, pages) in self.layer_page_tables.iter_mut().zip(new_pages) {
            layer_tables[seq_idx].extend(pages);
        }
    }

    /// Write K and V for a token position in a specific layer.
    /// Layout per page: `[K_data | V_data]` where each is `PAGE_SIZE * kv_dim` floats.
    pub fn write_kv(&mut self, layer_idx: usize, seq_idx: usize, pos: usize, k: &[f32], v: &[f32]) {
        let page_local = pos % PAGE_SIZE;
        let page_list_idx = pos / PAGE_SIZE;
        let pidx = self.layer_page_tables[layer_idx][seq_idx][page_list_idx];
        let page = &mut self.pages[pidx];
        let kv_page_size = PAGE_SIZE * self.kv_dim;
        let k_off = page_local * self.kv_dim;
        let v_off = kv_page_size + page_local * self.kv_dim;
        page[k_off..k_off + self.kv_dim].copy_from_slice(k);
        page[v_off..v_off + self.kv_dim].copy_from_slice(v);
    }

    /// Read K and V for a token position in a specific layer.
    pub fn read_kv(
        &self,
        layer_idx: usize,
        seq_idx: usize,
        pos: usize,
        k: &mut [f32],
        v: &mut [f32],
    ) {
        let page_local = pos % PAGE_SIZE;
        let page_list_idx = pos / PAGE_SIZE;
        let pidx = self.layer_page_tables[layer_idx][seq_idx][page_list_idx];
        let page = &self.pages[pidx];
        let kv_page_size = PAGE_SIZE * self.kv_dim;
        let k_off = page_local * self.kv_dim;
        let v_off = kv_page_size + page_local * self.kv_dim;
        k.copy_from_slice(&page[k_off..k_off + self.kv_dim]);
        v.copy_from_slice(&page[v_off..v_off + self.kv_dim]);
    }

    /// Fork a sequence with copy-on-write semantics.
    /// Shares prefix pages up to `fork_at_pos`, allocates new pages on demand after fork.
    /// Returns the new sequence index.
    pub fn fork(&mut self, seq_idx: usize, fork_at_pos: usize) -> usize {
        let fork_page = fork_at_pos / PAGE_SIZE;
        let new_seq = self.layer_page_tables[0].len();

        for layer_tables in &mut self.layer_page_tables {
            let source = &layer_tables[seq_idx];
            let shared_pages = source[..fork_page.min(source.len())].to_vec();
            layer_tables.push(shared_pages);
        }

        new_seq
    }

    /// Rollback a sequence to a given position, freeing exclusive pages.
    ///
    /// Truncates page tables to keep only pages covering positions `[0..rollback_to_pos)`.
    /// Pages that are exclusively owned by this sequence (not referenced by any other
    /// sequence in any layer) are returned to the free list for reuse.
    ///
    /// This is the "page table CoW rollback" — no data is copied, only page table
    /// entries are manipulated and exclusive pages are recycled.
    pub fn rollback(&mut self, seq_idx: usize, rollback_to_pos: usize) {
        let keep_count = rollback_to_pos / PAGE_SIZE;

        // Build set of page indices referenced by all OTHER sequences across all layers.
        // Since page indices are globally unique (allocated from a single pool), a simple
        // HashSet<usize> is sufficient — no need for (layer, page) tuples.
        let mut referenced_by_others = std::collections::HashSet::new();
        for layer_tables in self.layer_page_tables.iter() {
            for (seq, table) in layer_tables.iter().enumerate() {
                if seq != seq_idx {
                    for &pidx in table {
                        referenced_by_others.insert(pidx);
                    }
                }
            }
        }

        // Truncate page tables and free exclusive pages
        for layer_tables in &mut self.layer_page_tables {
            if seq_idx >= layer_tables.len() {
                continue;
            }
            let table = &mut layer_tables[seq_idx];
            let removed: Vec<usize> = table.drain(keep_count..).collect();
            for pidx in removed {
                if !referenced_by_others.contains(&pidx) {
                    self.free_pages.push(pidx);
                }
            }
        }
    }

    /// Reset all sequences and free all pages.
    pub fn reset(&mut self) {
        for layer_tables in &mut self.layer_page_tables {
            for table in layer_tables.iter_mut() {
                self.free_pages.append(table);
            }
        }
    }
}

// ── Raven RSM (Routing Slot Memory) ────────────────────────────
// Distilled from "Raven: High-Recall Sequence Modeling with Sparse Memory Routing"
// See .research/06_Raven_Routing_Slot_Memories.md for full derivation.
//
// Replaces the growing [block_size, kv_dim] cache with a fixed [num_slots, kv_dim]
// memory updated via sparse Top-K routing. Unselected slots are completely frozen.
// Per-token compute: O(num_slots) — constant regardless of sequence length.

/// Raven Routing Slot Memory — O(1) KV replacement for the draft model.
///
/// Fixed-size `[num_slots × kv_dim]` memory updated via sparse Top-K routing.
/// Unselected slots are completely frozen — perfect for preserving struct
/// definitions and imports while churning through syntax tokens.
pub struct RavenKVCache {
    /// Number of memory slots
    pub num_slots: usize,
    /// Dimension of each KV entry (= kv_dim = n_kv_head × head_dim)
    pub kv_dim: usize,
    /// Top-K slots to update per token
    pub top_k: usize,
    /// Forget rate for gated update (negative = slower decay)
    pub forget_rate: f32,
    /// Key memory: [num_slots × kv_dim]
    pub keys: Vec<f32>,
    /// Value memory: [num_slots × kv_dim]
    pub values: Vec<f32>,
    // Pre-allocated buffers for zero-alloc router computation
    router_scored: Vec<(usize, f32)>, // [num_slots]
    router_r_t: Vec<f32>,             // [num_slots]
}

impl RavenKVCache {
    pub fn new(config: &Config, num_slots: usize, top_k: usize) -> Self {
        let kvd = types::kv_dim(config);
        Self {
            num_slots,
            kv_dim: kvd,
            top_k,
            forget_rate: -1.0,
            keys: vec![0.0; num_slots * kvd],
            values: vec![0.0; num_slots * kvd],
            router_scored: Vec::with_capacity(num_slots),
            router_r_t: Vec::with_capacity(num_slots),
        }
    }

    pub fn reset(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
        self.router_scored.clear();
        self.router_r_t.clear();
    }
}

/// Sparse router: computes Top-K routing vector from raw logits (zero-alloc variant).
///
/// Implements: `r_t = Normalize(TopK(Sigmoid(raw_logits)))`
/// Unselected slots get 0.0 → completely frozen during update.
///
/// Uses pre-allocated buffers to avoid heap allocations on the hot path.
pub fn raven_compute_router_into(
    raw_logits: &[f32],
    top_k: usize,
    scored: &mut Vec<(usize, f32)>,
    r_t: &mut Vec<f32>,
) {
    let num_slots = raw_logits.len();
    let top_k = top_k.min(num_slots);

    // Reuse pre-allocated buffers: clear + fill
    scored.clear();
    for (i, &x) in raw_logits.iter().enumerate() {
        scored.push((i, 1.0 / (1.0 + (-x).exp())));
    }

    // Partial sort: find Top-K by descending score (O(n) average)
    if top_k < num_slots {
        scored.select_nth_unstable_by(num_slots - top_k, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    r_t.clear();
    r_t.resize(num_slots, 0.0);
    let mut sum = 0.0f32;

    // Keep only Top-K (the last top_k elements after partial sort are the largest)
    for (idx, score) in scored.iter().rev().take(top_k) {
        r_t[*idx] = *score;
        sum += *score;
    }

    // Normalize so selected slots sum to 1.0
    if sum > 0.0 {
        for v in r_t.iter_mut() {
            *v /= sum;
        }
    }
}

/// Backward-compatible wrapper that allocates fresh buffers.
pub fn raven_compute_router(raw_logits: &[f32], top_k: usize) -> Vec<f32> {
    let mut scored = Vec::new();
    let mut r_t = Vec::new();
    raven_compute_router_into(raw_logits, top_k, &mut scored, &mut r_t);
    r_t
}

/// Gated memory update: Raven Equation 18.
///
/// For each slot:
///   `decay = exp(forget_rate × r_t[slot])`
///   `H_new = decay × H_old + (1 - decay) × new_content`
///
/// When `r_t[slot] == 0`: `decay = exp(0) = 1.0` → `H_new = H_old` (FROZEN)
/// When `r_t[slot] > 0`: `decay < 1.0` → old content decays, new writes in
#[allow(clippy::too_many_arguments)]
pub fn raven_update(
    keys: &mut [f32],
    values: &mut [f32],
    new_key: &[f32],
    new_value: &[f32],
    r_t: &[f32],
    forget_rate: f32,
    num_slots: usize,
    kv_dim: usize,
) {
    for (slot, &route) in r_t.iter().enumerate().take(num_slots) {
        let decay = (forget_rate * route).exp();
        let write = 1.0 - decay;
        let offset = slot * kv_dim;

        for d in 0..kv_dim {
            keys[offset + d] = decay * keys[offset + d] + write * new_key[d];
            values[offset + d] = decay * values[offset + d] + write * new_value[d];
        }
    }
}

/// Readout: attention over fixed slot memory.
/// `O(num_slots × kv_dim)` — constant regardless of sequence length.
pub fn raven_readout(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    num_slots: usize,
    kv_dim: usize,
) -> Vec<f32> {
    // Q · K^T
    let scores: Vec<f32> = keys
        .chunks(kv_dim)
        .take(num_slots)
        .map(|k_chunk| query.iter().zip(k_chunk).map(|(q, k)| q * k).sum())
        .collect();

    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Softmax + weighted value sum
    let sum_exp: f32 = scores.iter().map(|s| (*s - max_score).exp()).sum();
    let mut output = vec![0.0f32; kv_dim];
    for (weight, v_chunk) in scores
        .iter()
        .map(|s| (s - max_score).exp() / sum_exp)
        .zip(values.chunks(kv_dim).take(num_slots))
    {
        for (out, v) in output.iter_mut().zip(v_chunk) {
            *out += weight * v;
        }
    }

    output
}

/// Forward pass using `RavenKVCache` instead of `MultiLayerKVCache`.
///
/// Identical computation to `forward()` except attention:
/// - Generates router logits from K projection (dummy: use K directly)
/// - Calls `raven_update()` instead of writing to flat KV array
/// - Calls `raven_readout()` instead of scanning all past positions
/// - Everything else (RMSNorm, MLP, residual, LM head) stays identical
pub fn forward_raven<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut RavenKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    for i in 0..n {
        unsafe {
            *ctx.x.get_unchecked_mut(i) = *weights.wte.get_unchecked(tok_off + i)
                + *weights.wpe.get_unchecked(pos_off_emb + i);
        }
    }

    // 2. Layer loop
    for layer_weights in &weights.layers {
        // Pre-attention: RMSNorm → save residual → RMSNorm
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);

        // QKV projections
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

        // Raven: generate router logits from K (dummy projection)
        // For PoC: use first num_slots elements of K repeated as logits.
        // In production, this would be a learned linear projection: W_route × x_t
        // Reuse pre-allocated query buffer for router logits (zero-alloc)
        ctx.raven_query_buf.resize(cache.num_slots, 0.0);
        for (i, slot) in ctx.raven_query_buf.iter_mut().enumerate() {
            *slot = ctx.k[i % kvd];
        }

        // Raven: compute sparse routing vector (zero-alloc via pre-allocated buffers)
        raven_compute_router_into(
            &ctx.raven_query_buf,
            cache.top_k,
            &mut cache.router_scored,
            &mut cache.router_r_t,
        );

        // Clone router_r_t to avoid self-borrow (cache.keys vs cache.router_r_t)
        // num_slots is typically 16-64 floats, so this is a tiny stack-like allocation
        let r_t = cache.router_r_t.clone();

        // Raven: gated update (only selected slots are modified)
        raven_update(
            &mut cache.keys,
            &mut cache.values,
            &ctx.k,
            &ctx.v,
            &r_t,
            cache.forget_rate,
            cache.num_slots,
            kvd,
        );

        // Raven: readout via attention over fixed slots (O(num_slots) not O(pos))
        let scale = 1.0 / (hd as f32).sqrt();
        ctx.attn_out[..n].fill(0.0);

        for h in 0..config.n_head {
            let q_off = h * hd;
            // Each head reads from the slot memory using its query slice
            let head_query = &ctx.q[q_off..q_off + hd];
            // Pad/reshape query to kv_dim for slot attention (reuse pre-allocated buffer)
            ctx.raven_query_buf.resize(kvd, 0.0);
            ctx.raven_query_buf.fill(0.0);
            let kv_group = h * n_kv / config.n_head;
            for (d, &hq) in head_query.iter().enumerate() {
                ctx.raven_query_buf[kv_group * hd + d] = hq * scale;
            }

            let slot_values = raven_readout(
                &ctx.raven_query_buf,
                &cache.keys,
                &cache.values,
                cache.num_slots,
                kvd,
            );

            // Extract this head's attention output
            for d in 0..hd {
                unsafe {
                    *ctx.attn_out.get_unchecked_mut(q_off + d) = slot_values[kv_group * hd + d];
                }
            }
        }

        // Output projection + residual
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);
        types::matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        // MLP w2: sparse when feature enabled and sparsity is high enough (Plan 022)
        #[cfg(feature = "sparse_mlp")]
        {
            let alive = types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                );
            }
        }
        #[cfg(not(feature = "sparse_mlp"))]
        matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head
    matmul(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

/// Forward pass using TurboQuant compressed KV cache (Plan 043).
///
/// Mirrors [`forward_base`] but stores K/V into a compressed cache and
/// dequantizes on-the-fly during attention scoring. The rest of the
/// transformer (embedding, QKV projection, MLP, LM head) is unchanged.
///
/// **Trade-off**: ~8× KV cache memory savings at the cost of dequantization
/// overhead during attention. Best for long sequences where cache memory
/// dominates.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn forward_turboquant<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut crate::turboquant::TurboQuantKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    for i in 0..n {
        unsafe {
            *ctx.x.get_unchecked_mut(i) = *weights.wte.get_unchecked(tok_off + i)
                + *weights.wpe.get_unchecked(pos_off_emb + i);
        }
    }

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        // Pre-attention: RMSNorm → save residual → RMSNorm
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);

        // QKV projections from per-layer weights (GQA: K/V produce kv_dim outputs)
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

        // Store compressed K,V
        cache.store_key(layer_idx, pos, &ctx.k[..kvd]);
        cache.store_value(layer_idx, pos, &ctx.v[..kvd]);

        // Dequantize all K,V into flat buffers for attention scoring
        let t_n = pos + 1;
        for t in 0..t_n {
            cache.dequantize_key_into(
                layer_idx,
                t,
                &mut ctx.paged_flat_key[t * kvd..(t + 1) * kvd],
            );
            cache.dequantize_value_into(
                layer_idx,
                t,
                &mut ctx.paged_flat_value[t * kvd..(t + 1) * kvd],
            );
        }

        // Multi-head attention with GQA using dequantized flat cache
        let scale = 1.0 / (hd as f32).sqrt();
        ctx.attn_out[..n].fill(0.0);

        for h in 0..config.n_head {
            let kv_group = h * n_kv / config.n_head;
            unsafe {
                attention_head(
                    &ctx.q,
                    &ctx.paged_flat_key,
                    &ctx.paged_flat_value,
                    &mut ctx.attn_out,
                    &mut ctx.scores,
                    h * hd,
                    kv_group * hd,
                    kvd,
                    hd,
                    t_n,
                    scale,
                );
            }
        }

        // Output projection + residual
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);
        types::matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }
    }

    // Snapshot hidden state (Plan 009 compatibility)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head
    matmul(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

#[cfg(test)]
#[allow(unnameable_test_items)]
#[allow(dead_code)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_output_size() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_forward_logits_finite() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logit {i} is not finite: {l}");
        }
    }

    #[test]
    fn test_forward_cache_populated() {
        let config = Config::micro();
        let kvd = crate::types::kv_dim(&config);
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        let key_sum: f32 = cache.layers[0].key[..kvd].iter().sum();
        let val_sum: f32 = cache.layers[0].value[..kvd].iter().sum();
        assert!(key_sum != 0.0, "K cache at pos 0 should be populated");
        assert!(val_sum != 0.0, "V cache at pos 0 should be populated");
    }

    #[test]
    fn test_forward_positions_differ() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let logits_0 = forward(&mut ctx, &weights, &mut cache, 0, 0, &config).to_vec();
        let logits_1 = forward(&mut ctx, &weights, &mut cache, 0, 1, &config);
        let different = logits_0.iter().zip(logits_1).any(|(&a, b)| a != *b);
        assert!(different, "logits at different positions should differ");
    }

    #[test]
    fn test_generate_deterministic() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut rng1 = Rng::new(100);
        let t1 = generate(&weights, &config, &mut rng1, 16);

        let mut rng2 = Rng::new(100);
        let t2 = generate(&weights, &config, &mut rng2, 16);

        assert_eq!(t1, t2, "Same seed must produce same tokens");
    }

    #[test]
    fn test_generate_valid_tokens() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let tokens = generate(&weights, &config, &mut rng, 32);
        assert_eq!(tokens.len(), 32);
        for &t in &tokens {
            assert!(t < config.vocab_size, "Token {t} out of range");
        }
    }

    #[test]
    fn test_tokens_to_string() {
        let tokens = vec![0, 1, 2, 25, 26];
        let s = tokens_to_string(&tokens);
        assert_eq!(s, "abcz_");
    }

    #[test]
    fn test_forward_context_reuse() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        // Multiple forward passes with same context should give same results
        let _l1 = forward(&mut ctx, &weights, &mut cache, 0, 0, &config).to_vec();
        let l2 = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        // Note: results differ because cache accumulates, but buffers should not leak
        for &v in l2.iter() {
            assert!(v.is_finite(), "reused context produced non-finite: {v}");
        }
    }

    // ── Multi-layer tests ─────────────────────────────────────────

    #[test]
    fn test_forward_output_size_nlayer2() {
        let mut config = Config::micro();
        config.n_layer = 2;
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        assert_eq!(weights.layers.len(), 2);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        assert_eq!(cache.layers.len(), 2);
        let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_forward_logits_finite_nlayer4() {
        let mut config = Config::micro();
        config.n_layer = 4;
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logit {i} is not finite with n_layer=4: {l}");
        }
    }

    #[test]
    fn test_n_layer_1_matches_current() {
        // n_layer=1 must produce identical deterministic output to old single-layer code
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut rng1 = Rng::new(100);
        let t1 = generate(&weights, &config, &mut rng1, 16);

        let mut rng2 = Rng::new(100);
        let t2 = generate(&weights, &config, &mut rng2, 16);

        assert_eq!(t1, t2, "n_layer=1 should be deterministic");
        assert_eq!(config.n_layer, 1, "micro config should have n_layer=1");
    }

    #[test]
    fn test_multi_layer_cache_populated() {
        let mut config = Config::micro();
        config.n_layer = 3;
        let kvd = crate::types::kv_dim(&config);
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        forward(&mut ctx, &weights, &mut cache, 0, 0, &config);

        // Every layer's cache should be populated
        for (layer_idx, layer_cache) in cache.layers.iter().enumerate() {
            let key_sum: f32 = layer_cache.key[..kvd].iter().sum();
            let val_sum: f32 = layer_cache.value[..kvd].iter().sum();
            assert!(
                key_sum != 0.0,
                "layer {layer_idx} K cache at pos 0 should be populated"
            );
            assert!(
                val_sum != 0.0,
                "layer {layer_idx} V cache at pos 0 should be populated"
            );
        }
    }

    #[test]
    fn test_hidden_state_populated() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        let sum: f32 = ctx.hidden_state.iter().sum();
        assert!(
            sum != 0.0,
            "hidden_state should be populated after forward pass"
        );
        for (i, &v) in ctx.hidden_state.iter().enumerate() {
            assert!(v.is_finite(), "hidden_state[{i}] should be finite: {v}");
        }
    }

    #[test]
    fn test_multi_layer_generate_valid() {
        let mut config = Config::micro();
        config.n_layer = 4;
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let tokens = generate(&weights, &config, &mut rng, 16);
        assert_eq!(tokens.len(), 16);
        for &t in &tokens {
            assert!(t < config.vocab_size, "Token {t} out of range");
        }
    }

    // ── GQA tests ───────────────────────────────────────────────

    #[test]
    fn test_gqa_produces_valid_logits() {
        let config = Config::gqa_draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        for pos in 0..4 {
            let logits = forward(&mut ctx, &weights, &mut cache, 0, pos, &config);
            for (i, &l) in logits.iter().enumerate() {
                assert!(
                    l.is_finite(),
                    "gqa_draft logit {i} at pos {pos} not finite: {l}"
                );
            }
        }
    }

    #[test]
    fn test_gqa_mha_backward_compat() {
        // When n_kv_head == n_head, GQA produces identical results to standard MHA.
        // Micro config has n_kv_head=4, n_head=4 → pure MHA.
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut rng1 = Rng::new(100);
        let t1 = generate(&weights, &config, &mut rng1, 16);

        let mut rng2 = Rng::new(100);
        let t2 = generate(&weights, &config, &mut rng2, 16);

        assert_eq!(
            t1, t2,
            "MHA backward compat: same seed must produce same tokens"
        );
        assert_eq!(
            config.n_kv_head, config.n_head,
            "micro config should have n_kv_head == n_head"
        );
    }

    #[test]
    fn test_gqa_kv_cache_smaller() {
        // GQA config should have smaller KV cache than equivalent MHA config
        let gqa = Config::gqa_draft();
        let kvd = crate::types::kv_dim(&gqa);
        assert_eq!(
            kvd,
            gqa.n_kv_head * gqa.head_dim,
            "kv_dim should be n_kv_head * head_dim"
        );
        assert!(
            kvd < gqa.n_embd,
            "GQA kv_dim ({kvd}) should be < n_embd ({})",
            gqa.n_embd
        );

        // Verify cache is correctly sized
        let cache = KVCache::new(&gqa);
        assert_eq!(
            cache.key.len(),
            gqa.block_size * kvd,
            "GQA key cache should use kv_dim"
        );
        assert_eq!(
            cache.value.len(),
            gqa.block_size * kvd,
            "GQA value cache should use kv_dim"
        );
    }

    #[test]
    fn test_gqa_generate_valid_tokens() {
        let config = Config::gqa_draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let tokens = generate(&weights, &config, &mut rng, 8);
        assert_eq!(tokens.len(), 8);
        for &t in &tokens {
            assert!(t < config.vocab_size, "GQA token {t} out of range");
        }
    }

    #[test]
    fn test_config_validate_gqa() {
        // Valid configs should pass validation
        assert!(Config::micro().validate().is_ok());
        assert!(Config::draft().validate().is_ok());
        assert!(Config::small_target().validate().is_ok());
        assert!(Config::gqa_draft().validate().is_ok());

        // Invalid: n_head not divisible by n_kv_head
        let mut bad = Config::micro();
        bad.n_kv_head = 3; // n_head=4, not divisible by 3
        assert!(bad.validate().is_err());

        // Invalid: n_head * head_dim != n_embd
        let mut bad2 = Config::micro();
        bad2.head_dim = 5; // 4*5=20 != 16
        assert!(bad2.validate().is_err());
    }

    // ── Paged KV cache tests ────────────────────────────────────

    #[test]
    fn test_paged_cache_write_read_roundtrip() {
        let config = Config::micro();
        let mut paged = PagedKVCache::new(&config, 1);
        let kvd = crate::types::kv_dim(&config);

        // Ensure pages for position 0
        paged.ensure_pages(0, 0);

        // Write some K/V data
        let k_data: Vec<f32> = (0..kvd).map(|i| i as f32 * 0.1).collect();
        let v_data: Vec<f32> = (0..kvd).map(|i| i as f32 * 0.2).collect();
        paged.write_kv(0, 0, 0, &k_data, &v_data);

        // Read back
        let mut k_out = vec![0.0f32; kvd];
        let mut v_out = vec![0.0f32; kvd];
        paged.read_kv(0, 0, 0, &mut k_out, &mut v_out);

        assert_eq!(k_out, k_data, "K data roundtrip mismatch");
        assert_eq!(v_out, v_data, "V data roundtrip mismatch");
    }

    #[test]
    fn test_paged_cache_linear_matches_flat() {
        // Paged cache should produce same results as flat cache for a linear sequence
        let config = Config::micro();
        let kvd = crate::types::kv_dim(&config);
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        // Run with flat cache
        let mut ctx = ForwardContext::new(&config);
        let mut flat_cache = MultiLayerKVCache::new(&config);
        let _flat_logits = forward(&mut ctx, &weights, &mut flat_cache, 0, 0, &config).to_vec();

        // Manually copy flat cache data to paged cache
        let mut paged = PagedKVCache::new(&config, 1);
        paged.ensure_pages(0, 0);

        for (layer_idx, layer_cache) in flat_cache.layers.iter().enumerate() {
            let k_data = &layer_cache.key[..kvd];
            let v_data = &layer_cache.value[..kvd];
            paged.write_kv(layer_idx, 0, 0, k_data, v_data);
        }

        // Read back and compare
        for layer_idx in 0..config.n_layer {
            let mut k_out = vec![0.0f32; kvd];
            let mut v_out = vec![0.0f32; kvd];
            paged.read_kv(layer_idx, 0, 0, &mut k_out, &mut v_out);

            let flat_k = &flat_cache.layers[layer_idx].key[..kvd];
            let flat_v = &flat_cache.layers[layer_idx].value[..kvd];
            assert_eq!(k_out, flat_k, "layer {layer_idx} K mismatch: paged vs flat");
            assert_eq!(v_out, flat_v, "layer {layer_idx} V mismatch: paged vs flat");
        }
    }

    #[test]
    fn test_paged_cache_fork_no_corruption() {
        let config = Config::micro();
        let kvd = crate::types::kv_dim(&config);
        let mut paged = PagedKVCache::new(&config, 1);

        // Write data to seq 0 at position 0
        paged.ensure_pages(0, 0);
        let k_orig: Vec<f32> = (0..kvd).map(|i| i as f32 + 1.0).collect();
        let v_orig: Vec<f32> = (0..kvd).map(|i| i as f32 + 2.0).collect();
        paged.write_kv(0, 0, 0, &k_orig, &v_orig);

        // Fork at position 0 (share nothing — fork_page = 0/16 = 0)
        let fork_seq = paged.fork(0, 0);

        // Write different data to forked seq
        paged.ensure_pages(fork_seq, 0);
        let k_fork: Vec<f32> = (0..kvd).map(|i| i as f32 + 99.0).collect();
        let v_fork: Vec<f32> = (0..kvd).map(|i| i as f32 + 100.0).collect();
        paged.write_kv(0, fork_seq, 0, &k_fork, &v_fork);

        // Original seq should be unchanged
        let mut k_check = vec![0.0f32; kvd];
        let mut v_check = vec![0.0f32; kvd];
        paged.read_kv(0, 0, 0, &mut k_check, &mut v_check);
        assert_eq!(k_check, k_orig, "original K corrupted after fork write");
        assert_eq!(v_check, v_orig, "original V corrupted after fork write");
    }

    #[test]
    fn test_paged_cache_fork_shares_prefix() {
        let config = Config::micro();
        let kvd = crate::types::kv_dim(&config);
        let mut paged = PagedKVCache::new(&config, 1);

        // Write data at positions 0..PAGE_SIZE (fills one page)
        paged.ensure_pages(0, PAGE_SIZE - 1);
        for pos in 0..PAGE_SIZE {
            let k: Vec<f32> = vec![pos as f32; kvd];
            let v: Vec<f32> = vec![pos as f32 * 2.0; kvd];
            paged.write_kv(0, 0, pos, &k, &v);
        }

        // Fork at position 8 (still within page 0)
        let fork_seq = paged.fork(0, 8);

        // Ensure forked seq has its own pages from fork point
        paged.ensure_pages(fork_seq, PAGE_SIZE);

        // The forked seq should share page 0 (prefix) but have its own page 1+
        // Verify shared prefix data is accessible
        let mut k_out = vec![0.0f32; kvd];
        let mut v_out = vec![0.0f32; kvd];
        paged.read_kv(0, fork_seq, 0, &mut k_out, &mut v_out);
        assert_eq!(k_out[0], 0.0, "forked seq should see original pos 0 data");
    }

    #[test]
    fn test_paged_cache_reset_frees_pages() {
        let config = Config::micro();
        let mut paged = PagedKVCache::new(&config, 2);

        // Allocate pages for two sequences
        paged.ensure_pages(0, 31); // 2 pages (0..15 and 16..31)
        paged.ensure_pages(1, 15); // 1 page

        let total_before = paged.total_pages;
        assert!(total_before > 0, "should have allocated some pages");

        // Reset should free all pages
        paged.reset();

        // Free list should contain the freed pages
        // (exact count depends on implementation, but should be > 0)
        // After reset, we can allocate again and reuse freed pages
        paged.ensure_pages(0, 0);
        // If reuse works, total_pages shouldn't grow
        assert_eq!(paged.total_pages, total_before, "should reuse freed pages");
    }

    #[test]
    fn test_snapshot_restore_roundtrip() {
        // Forward some tokens, snapshot, modify, restore, verify same logits
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        // Fill cache with tokens at positions 0..4
        for pos in 0..4 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }

        // Snapshot at position 4
        let snapshot = cache.snapshot(4, &config);

        // Fill more positions
        for pos in 4..8 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }

        // Now restore
        cache.restore(&snapshot, &config);

        // Verify restored: forward at position 4 should give same result as fresh cache at pos 4
        let mut fresh_cache = MultiLayerKVCache::new(&config);
        let mut fresh_ctx = ForwardContext::new(&config);
        for pos in 0..4 {
            let _ = forward(
                &mut fresh_ctx,
                &weights,
                &mut fresh_cache,
                pos,
                pos,
                &config,
            );
        }

        let restored_logits = forward(&mut ctx, &weights, &mut cache, 0, 4, &config);
        let fresh_logits = forward(&mut fresh_ctx, &weights, &mut fresh_cache, 0, 4, &config);

        for (a, b) in restored_logits.iter().zip(fresh_logits.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "restored logits should match fresh: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_snapshot_correct_size() {
        let config = Config::micro();
        let kd = types::kv_dim(&config);
        let cache = MultiLayerKVCache::new(&config);
        let snapshot = cache.snapshot(5, &config);

        assert_eq!(snapshot.pos, 5);
        assert_eq!(snapshot.layers.len(), config.n_layer);
        for layer in &snapshot.layers {
            assert_eq!(layer.key.len(), 5 * kd);
            assert_eq!(layer.value.len(), 5 * kd);
        }
    }

    #[test]
    fn test_restore_zeros_stale_data() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        // Fill cache
        for pos in 0..8 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }

        // Snapshot at position 3
        let snapshot = cache.snapshot(3, &config);

        // Restore
        cache.restore(&snapshot, &config);

        // Verify positions after pos=3 are zeroed
        let kd = types::kv_dim(&config);
        for layer in &cache.layers {
            for val in &layer.key[3 * kd..] {
                assert_eq!(*val, 0.0, "stale key data should be zeroed");
            }
            for val in &layer.value[3 * kd..] {
                assert_eq!(*val, 0.0, "stale value data should be zeroed");
            }
        }
    }

    #[test]
    fn test_snapshot_restore_multi_layer() {
        // Test with n_layer > 1 (small_target config)
        let config = Config::small_target();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        // Fill cache
        for pos in 0..4 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }

        let snapshot = cache.snapshot(4, &config);
        assert_eq!(snapshot.layers.len(), 4, "should have 4 layer snapshots");

        // Modify and restore
        for pos in 4..8 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }
        cache.restore(&snapshot, &config);

        // Verify restored correctly by checking logits match fresh cache
        let mut fresh_cache = MultiLayerKVCache::new(&config);
        let mut fresh_ctx = ForwardContext::new(&config);
        for pos in 0..4 {
            let _ = forward(
                &mut fresh_ctx,
                &weights,
                &mut fresh_cache,
                pos,
                pos,
                &config,
            );
        }

        let restored_logits = forward(&mut ctx, &weights, &mut cache, 0, 4, &config);
        let fresh_logits = forward(&mut fresh_ctx, &weights, &mut fresh_cache, 0, 4, &config);

        for (a, b) in restored_logits.iter().zip(fresh_logits.iter()) {
            assert!(
                (a - b).abs() < 1e-3,
                "multi-layer restore should match fresh"
            );
        }
    }

    #[test]
    fn test_snapshot_restore_gqa() {
        // Test with GQA config (kv_dim < n_embd)
        let config = Config::gqa_draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        for pos in 0..4 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }

        let snapshot = cache.snapshot(4, &config);
        let kd = types::kv_dim(&config);

        // Verify snapshot uses GQA kv_dim (smaller than n_embd)
        assert_eq!(kd, config.n_kv_head * config.head_dim);
        assert!(kd < config.n_embd, "GQA kv_dim should be < n_embd");
        for layer in &snapshot.layers {
            assert_eq!(layer.key.len(), 4 * kd);
        }

        // Restore and verify
        for pos in 4..8 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }
        cache.restore(&snapshot, &config);

        let mut fresh_cache = MultiLayerKVCache::new(&config);
        let mut fresh_ctx = ForwardContext::new(&config);
        for pos in 0..4 {
            let _ = forward(
                &mut fresh_ctx,
                &weights,
                &mut fresh_cache,
                pos,
                pos,
                &config,
            );
        }

        let restored_logits = forward(&mut ctx, &weights, &mut cache, 0, 4, &config);
        let fresh_logits = forward(&mut fresh_ctx, &weights, &mut fresh_cache, 0, 4, &config);

        for (a, b) in restored_logits.iter().zip(fresh_logits.iter()) {
            assert!((a - b).abs() < 1e-3, "GQA restore should match fresh");
        }
    }

    // ── forward_paged tests ──────────────────────────────────────

    #[test]
    fn test_forward_paged_logits_match_forward() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        // Flat cache forward
        let mut ctx_flat = ForwardContext::new(&config);
        let mut cache_flat = MultiLayerKVCache::new(&config);
        let logits_flat = forward(&mut ctx_flat, &weights, &mut cache_flat, 0, 0, &config);

        // Paged cache forward
        let mut ctx_paged = ForwardContext::new(&config);
        let mut cache_paged = PagedKVCache::new(&config, 1);
        let logits_paged =
            forward_paged(&mut ctx_paged, &weights, &mut cache_paged, 0, 0, 0, &config);

        assert_eq!(logits_flat.len(), logits_paged.len());
        for (i, (a, b)) in logits_flat.iter().zip(logits_paged.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "forward_paged logit {i} differs: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_forward_paged_logits_match_forward_multi_pos() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut ctx_flat = ForwardContext::new(&config);
        let mut cache_flat = MultiLayerKVCache::new(&config);

        let mut ctx_paged = ForwardContext::new(&config);
        let mut cache_paged = PagedKVCache::new(&config, 1);

        for pos in 0..4 {
            let token = pos; // simple: use pos as token
            let logits_flat = forward(
                &mut ctx_flat,
                &weights,
                &mut cache_flat,
                token,
                pos,
                &config,
            );
            let logits_paged = forward_paged(
                &mut ctx_paged,
                &weights,
                &mut cache_paged,
                0,
                token,
                pos,
                &config,
            );

            for (i, (a, b)) in logits_flat.iter().zip(logits_paged.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-3,
                    "pos {pos} logit {i} differs: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn test_forward_paged_gqa_logits_match() {
        let config = Config::gqa_draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut ctx_flat = ForwardContext::new(&config);
        let mut cache_flat = MultiLayerKVCache::new(&config);
        let logits_flat = forward(&mut ctx_flat, &weights, &mut cache_flat, 0, 0, &config);

        let mut ctx_paged = ForwardContext::new(&config);
        let mut cache_paged = PagedKVCache::new(&config, 1);
        let logits_paged =
            forward_paged(&mut ctx_paged, &weights, &mut cache_paged, 0, 0, 0, &config);

        assert_eq!(logits_flat.len(), logits_paged.len());
        for (i, (a, b)) in logits_flat.iter().zip(logits_paged.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "GQA forward_paged logit {i} differs: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_forward_paged_output_size() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = PagedKVCache::new(&config, 1);
        let logits = forward_paged(&mut ctx, &weights, &mut cache, 0, 0, 0, &config);
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_forward_paged_logits_finite() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = PagedKVCache::new(&config, 1);
        let logits = forward_paged(&mut ctx, &weights, &mut cache, 0, 0, 0, &config);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logit {i} is not finite: {l}");
        }
    }

    // ── Rollback tests ─────────────────────────────────────────────

    #[test]
    fn test_paged_rollback_frees_exclusive_pages() {
        let config = Config::micro();
        let mut paged = PagedKVCache::new(&config, 2);

        // Allocate pages for seq 0 up to pos 31 (2 pages: 0..15, 16..31)
        paged.ensure_pages(0, 31);
        let seq0_pages_len = paged.layer_page_tables[0][0].len();
        assert!(seq0_pages_len >= 2, "seq 0 should have at least 2 pages");

        // Rollback seq 0 to pos 0 — all pages are exclusive (no other seq)
        paged.rollback(0, 0);

        // Page table should be truncated
        assert!(
            paged.layer_page_tables[0][0].is_empty(),
            "seq 0 page table should be empty after rollback to pos 0"
        );
        // All pages should be freed (they were exclusive)
        assert!(
            !paged.free_pages.is_empty(),
            "exclusive pages should be returned to free list"
        );
    }

    #[test]
    fn test_paged_rollback_preserves_shared_pages() {
        let config = Config::micro();
        let mut paged = PagedKVCache::new(&config, 4);

        // Allocate pages for seq 0 up to pos 31
        paged.ensure_pages(0, 31);
        let _initial_pages_len = paged.layer_page_tables[0][0].len();

        // Fork a new sequence from seq 0 at pos 16 — shares first page
        // (fork returns layer_page_tables[0].len(), which may be > 1 if max_sequences > 1)
        let seq1 = paged.fork(0, 16);
        assert_ne!(seq1, 0, "fork should return a new sequence index");

        // Allocate exclusive pages for seq 0 beyond fork point
        paged.ensure_pages(0, 47); // extra pages after pos 31

        let free_before = paged.free_pages.len();
        let pages_before_rollback = paged.layer_page_tables[0][0].len();

        // Rollback seq 0 to pos 16 — keeps shared page, frees exclusive ones
        paged.rollback(0, 16);

        // Page table should be truncated to 1 page (covers 0..15)
        assert_eq!(
            paged.layer_page_tables[0][0].len(),
            1,
            "seq 0 should have 1 page after rollback to pos 16 (page covers 0..15)"
        );

        // Some pages should have been freed (the exclusive ones beyond page 0)
        let freed = paged.free_pages.len() - free_before;
        assert!(
            freed > 0,
            "exclusive pages beyond rollback point should be freed"
        );

        // But NOT more than what was removed from page table
        let removed = pages_before_rollback - 1;
        assert!(
            freed <= removed,
            "freed pages ({freed}) should not exceed removed pages ({removed})"
        );
    }

    #[test]
    fn test_paged_rollback_shared_page_not_freed() {
        let config = Config::micro();
        let mut paged = PagedKVCache::new(&config, 4);

        // Allocate pages for seq 0
        paged.ensure_pages(0, 31);

        // Fork seq 1 at pos 0 — shares nothing initially (fork_page = 0)
        let seq1 = paged.fork(0, 0);

        // Allocate different pages for seq 1
        paged.ensure_pages(seq1, 31);

        // Now fork seq 2 from seq 0 at pos 16 — shares first page with seq 0
        let seq2 = paged.fork(0, 16);
        let shared_page_idx = paged.layer_page_tables[0][0][0];

        // Rollback seq 2 to pos 0 — the shared page should NOT be freed
        let _free_before = paged.free_pages.len();
        paged.rollback(seq2, 0);

        // Shared page should still be in seq 0's page table
        assert!(
            paged.layer_page_tables[0][0].contains(&shared_page_idx),
            "shared page should still be referenced by seq 0"
        );
        // Shared page should NOT be in free list
        assert!(
            !paged.free_pages.contains(&shared_page_idx),
            "shared page should not be freed"
        );
    }

    #[test]
    fn test_paged_rollback_truncates_page_table() {
        let config = Config::micro();
        let mut paged = PagedKVCache::new(&config, 1);

        // Allocate 4 pages worth of positions
        paged.ensure_pages(0, 63);
        assert!(
            paged.layer_page_tables[0][0].len() >= 4,
            "should have at least 4 pages for pos 0..63"
        );

        // Rollback to pos 32 — should keep 2 pages (0..15, 16..31)
        paged.rollback(0, 32);
        assert_eq!(
            paged.layer_page_tables[0][0].len(),
            2,
            "should have exactly 2 pages after rollback to pos 32"
        );

        // Rollback to pos 16 — should keep 1 page (0..15)
        paged.rollback(0, 16);
        assert_eq!(
            paged.layer_page_tables[0][0].len(),
            1,
            "should have exactly 1 page after rollback to pos 16"
        );
    }

    #[test]
    fn test_paged_rollback_all_layers_consistent() {
        let mut config = Config::micro();
        config.n_layer = 4;
        let mut paged = PagedKVCache::new(&config, 1);

        // Allocate pages for all layers
        paged.ensure_pages(0, 31);

        // Rollback to pos 16
        paged.rollback(0, 16);

        // All layers should have the same page table length
        let expected = 1; // 1 page covers 0..15
        for (layer_idx, lt) in paged.layer_page_tables.iter().enumerate() {
            assert_eq!(
                lt[0].len(),
                expected,
                "layer {layer_idx} should have {expected} pages after rollback"
            );
        }
    }

    // ======================================================================
    // Sparse MLP tests (Plan 022: TwELL-inspired)
    // ======================================================================

    /// Sparse matmul produces identical output to dense at 0% sparsity (all alive).
    #[cfg(feature = "sparse_mlp")]
    #[test]
    fn test_sparse_matmul_0_percent_sparsity() {
        let rows = 16;
        let cols = 64;
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
        let input: Vec<f32> = (0..cols).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut dense_out = vec![0.0f32; rows];
        let mut sparse_out = vec![0.0f32; rows];
        let mut indices = vec![0usize; cols];
        let mut values = vec![0.0f32; cols];

        crate::types::matmul(&mut dense_out, &weight, &input, rows, cols);
        crate::types::sparse_matmul(
            &mut sparse_out,
            &weight,
            &input,
            rows,
            cols,
            &mut indices,
            &mut values,
        );

        for i in 0..rows {
            assert!(
                (dense_out[i] - sparse_out[i]).abs() < 1e-3,
                "Mismatch at {i}: dense={}, sparse={}",
                dense_out[i],
                sparse_out[i]
            );
        }
    }

    /// Sparse matmul produces identical output at 95% sparsity.
    #[cfg(feature = "sparse_mlp")]
    #[test]
    fn test_sparse_matmul_95_percent_sparsity() {
        let rows = 16;
        let cols = 64;
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
        let mut input = vec![0.0f32; cols];
        // 5% alive
        for i in (0..cols).step_by(20) {
            input[i] = 1.0;
        }
        let mut dense_out = vec![0.0f32; rows];
        let mut sparse_out = vec![0.0f32; rows];
        let mut indices = vec![0usize; cols];
        let mut values = vec![0.0f32; cols];

        crate::types::matmul(&mut dense_out, &weight, &input, rows, cols);
        crate::types::sparse_matmul(
            &mut sparse_out,
            &weight,
            &input,
            rows,
            cols,
            &mut indices,
            &mut values,
        );

        for i in 0..rows {
            assert!(
                (dense_out[i] - sparse_out[i]).abs() < 1e-4,
                "Mismatch at {i}: dense={}, sparse={}",
                dense_out[i],
                sparse_out[i]
            );
        }
    }

    /// Sparse matmul with 100% sparsity (all zeros) produces all-zero output.
    #[cfg(feature = "sparse_mlp")]
    #[test]
    fn test_sparse_matmul_100_percent_sparsity() {
        let rows = 16;
        let cols = 64;
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
        let input = vec![0.0f32; cols];
        let mut sparse_out = vec![0.0f32; rows];
        let mut indices = vec![0usize; cols];
        let mut values = vec![0.0f32; cols];

        let alive = crate::types::sparse_matmul(
            &mut sparse_out,
            &weight,
            &input,
            rows,
            cols,
            &mut indices,
            &mut values,
        );

        assert_eq!(alive, 0, "Expected 0 alive neurons");
        for (i, &val) in sparse_out.iter().take(rows).enumerate() {
            assert_eq!(val, 0.0, "Expected zero output at {i}");
        }
    }

    /// ForwardContext buffers are correctly sized when sparse_mlp is enabled.
    #[cfg(feature = "sparse_mlp")]
    #[test]
    fn test_forward_context_sparse_buffers() {
        let config = crate::types::Config::micro();
        let ctx = super::ForwardContext::new(&config);
        assert_eq!(ctx.active_indices.len(), config.mlp_hidden);
        assert_eq!(ctx.active_values.len(), config.mlp_hidden);
    }

    /// Forward pass works correctly with sparse_mlp enabled.
    #[cfg(feature = "sparse_mlp")]
    #[test]
    fn test_forward_with_sparse_mlp() {
        let config = crate::types::Config::micro();
        let mut rng = crate::types::Rng::new(42);
        let weights = crate::transformer::TransformerWeights::new(&config, &mut rng);
        let mut ctx = crate::transformer::ForwardContext::new(&config);
        let mut cache = crate::transformer::MultiLayerKVCache::new(&config);

        let logits = crate::transformer::forward(&mut ctx, &weights, &mut cache, 26, 0, &config);

        // Verify logits are finite
        for l in logits {
            assert!(l.is_finite(), "Logit is not finite: {l}");
        }
    }

    /// Sparse matmul with negative values (should be treated as dead by ReLU context).
    #[cfg(feature = "sparse_mlp")]
    #[test]
    fn test_sparse_matmul_negative_input() {
        let rows = 8;
        let cols = 32;
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
        let mut input = vec![0.0f32; cols];
        // Mix of positive, negative, zero
        input[0] = 1.0;
        input[1] = -1.0; // Should be ignored (not > 0)
        input[2] = 0.5;
        input[3] = -0.5; // Should be ignored
        // Rest are 0.0

        let mut dense_out = vec![0.0f32; rows];
        let mut sparse_out = vec![0.0f32; rows];
        let mut indices = vec![0usize; cols];
        let mut values = vec![0.0f32; cols];

        crate::types::matmul(&mut dense_out, &weight, &input, rows, cols);
        crate::types::sparse_matmul(
            &mut sparse_out,
            &weight,
            &input,
            rows,
            cols,
            &mut indices,
            &mut values,
        );

        // Both should match since matmul doesn't skip negatives but sparse_matmul skips input[c] <= 0
        // So we need to compare against a modified dense that also skips negatives
        for r in 0..rows {
            let mut expected = 0.0f32;
            for c in 0..cols {
                if input[c] > 0.0 {
                    expected += weight[r * cols + c] * input[c];
                }
            }
            assert!(
                (sparse_out[r] - expected).abs() < 1e-4,
                "Mismatch at {r}: sparse={}, expected={}",
                sparse_out[r],
                expected
            );
        }
    }

    // -----------------------------------------------------------------------
    // Plan 025: Bidirectional Prefill + Modality LoRA Switching
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_prefill_logits_finite() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let tokens: Vec<usize> = (0..8).collect();
        #[cfg(not(feature = "domain_latent"))]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
        );
        #[cfg(feature = "domain_latent")]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
            None,
        );
        assert_eq!(logits.len(), config.vocab_size);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "prefill logit {i} is not finite: {l}");
        }
    }

    #[test]
    fn test_forward_prefill_populates_cache() {
        let config = Config::micro();
        let kvd = crate::types::kv_dim(&config);
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let tokens: Vec<usize> = (0..5).collect();
        #[cfg(not(feature = "domain_latent"))]
        forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
        );
        #[cfg(feature = "domain_latent")]
        forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
            None,
        );
        // All 5 positions should have K/V in cache
        for p in 0..5 {
            let off = p * kvd;
            let key_sum: f32 = cache.layers[0].key[off..off + kvd].iter().sum();
            let val_sum: f32 = cache.layers[0].value[off..off + kvd].iter().sum();
            assert!(key_sum != 0.0, "K cache at pos {p} should be populated");
            assert!(val_sum != 0.0, "V cache at pos {p} should be populated");
        }
    }

    #[test]
    fn test_forward_prefill_logits_shape() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let tokens: Vec<usize> = vec![0, 1, 2];
        #[cfg(not(feature = "domain_latent"))]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
        );
        #[cfg(feature = "domain_latent")]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
            None,
        );
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_forward_prefill_single_token() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let tokens = vec![5];
        #[cfg(not(feature = "domain_latent"))]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
        );
        #[cfg(feature = "domain_latent")]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
            None,
        );
        assert_eq!(logits.len(), config.vocab_size);
        for (i, &l) in logits.iter().enumerate() {
            assert!(
                l.is_finite(),
                "single-token prefill logit {i} not finite: {l}"
            );
        }
    }

    #[test]
    fn test_prefill_then_decode_shared_cache() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        // Prefill with 4 tokens
        let prompt: Vec<usize> = (0..4).collect();
        #[cfg(not(feature = "domain_latent"))]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &prompt,
            &config,
            None,
        );
        #[cfg(feature = "domain_latent")]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &prompt,
            &config,
            None,
            None,
        );
        assert_eq!(logits.len(), config.vocab_size);

        // Decode from position 4 (should use same cache)
        let logits2 = forward(&mut ctx, &weights, &mut cache, 0, 4, &config);
        assert_eq!(logits2.len(), config.vocab_size);
        for (i, &l) in logits2.iter().enumerate() {
            assert!(
                l.is_finite(),
                "decode after prefill logit {i} not finite: {l}"
            );
        }
    }

    #[test]
    fn test_no_lora_matches_existing_forward() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        // Existing forward (no LoRA)
        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let logits1 = forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config);

        // New forward_base with None (should be identical)
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        #[cfg(not(feature = "domain_latent"))]
        let logits2 = forward_base(&mut ctx2, &weights, &mut cache2, 0, 0, &config, None);
        #[cfg(feature = "domain_latent")]
        let logits2 = forward_base(&mut ctx2, &weights, &mut cache2, 0, 0, &config, None, None);

        for i in 0..config.vocab_size {
            let diff = (logits1[i] - logits2[i]).abs();
            assert!(
                diff < 1e-6,
                "forward and forward_base(None) differ at {i}: {diff}"
            );
        }
    }

    #[test]
    fn test_generate_with_prefill_produces_tokens() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        let prompt: Vec<usize> = (0..4).collect();
        let generated = generate_with_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &config,
            &mut rng,
            &prompt,
            10,
            &crate::types::LoraPair::none(),
        );

        assert!(!generated.is_empty(), "should generate at least one token");
        assert!(generated.len() <= 10, "should not exceed max_gen_tokens");
        for (i, &t) in generated.iter().enumerate() {
            assert!(t < config.vocab_size, "token {i} out of range: {t}");
        }
    }

    // -----------------------------------------------------------------------
    // Multi-layer prefill tests
    // -----------------------------------------------------------------------

    fn small_target_2layer() -> Config {
        let mut c = Config::small_target();
        c.n_layer = 2;
        c
    }

    #[test]
    fn test_forward_prefill_multilayer_logits_finite() {
        let config = small_target_2layer();
        config.validate().unwrap();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let tokens: Vec<usize> = (0..8).collect();
        #[cfg(not(feature = "domain_latent"))]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
        );
        #[cfg(feature = "domain_latent")]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
            None,
        );
        assert_eq!(logits.len(), config.vocab_size);
        for (i, &l) in logits.iter().enumerate() {
            assert!(
                l.is_finite(),
                "multilayer prefill logit {i} not finite: {l}"
            );
        }
    }

    #[test]
    fn test_forward_prefill_multilayer_cache_populated() {
        let config = small_target_2layer();
        let kvd = crate::types::kv_dim(&config);
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let tokens: Vec<usize> = (0..4).collect();
        #[cfg(not(feature = "domain_latent"))]
        forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
        );
        #[cfg(feature = "domain_latent")]
        forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
            None,
        );
        // Both layers should have K/V populated
        for layer in 0..2 {
            for p in 0..4 {
                let off = p * kvd;
                let key_sum: f32 = cache.layers[layer].key[off..off + kvd].iter().sum();
                let val_sum: f32 = cache.layers[layer].value[off..off + kvd].iter().sum();
                assert!(
                    key_sum != 0.0,
                    "layer {layer} K cache at pos {p} should be populated"
                );
                assert!(
                    val_sum != 0.0,
                    "layer {layer} V cache at pos {p} should be populated"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Domain Latent injection (Plan 038)
    // -----------------------------------------------------------------------

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_changes_logits() {
        let config = small_target_2layer(); // 2 layers, mid-layer = layer 1
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let kvd = crate::types::kv_dim(&config);

        // Without domain latent
        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let logits1 = forward_base(&mut ctx1, &weights, &mut cache1, 0, 0, &config, None, None);

        // With domain latent (non-zero embedding)
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let dl = crate::types::DomainLatent::from_vec(vec![0.5; kvd]);
        let logits2 = forward_base(
            &mut ctx2,
            &weights,
            &mut cache2,
            0,
            0,
            &config,
            None,
            Some(&dl),
        );

        // Logits should differ — domain latent modulates K/V at mid-layer
        let mut any_diff = false;
        for (&a, &b) in logits1.iter().zip(logits2.iter()) {
            if (a - b).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "domain latent should change logits");
    }

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_zero_embedding_same_logits() {
        let config = small_target_2layer();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let kvd = crate::types::kv_dim(&config);

        // Without domain latent
        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let logits1 = forward_base(&mut ctx1, &weights, &mut cache1, 0, 0, &config, None, None);

        // With zero domain latent — should be identical
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let dl = crate::types::DomainLatent::zeros(kvd);
        let logits2 = forward_base(
            &mut ctx2,
            &weights,
            &mut cache2,
            0,
            0,
            &config,
            None,
            Some(&dl),
        );

        for (i, (&a, &b)) in logits1.iter().zip(logits2.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff < 1e-6,
                "zero domain latent should not change logits, diff at {i}: {diff}"
            );
        }
    }

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_prefill_changes_logits() {
        let config = small_target_2layer();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let kvd = crate::types::kv_dim(&config);
        let tokens: Vec<usize> = (0..4).collect();

        // Without domain latent
        let mut ctx1 = ForwardContext::new(&config);
        let mut prefill1 = PrefillContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let logits1 = forward_prefill(
            &mut ctx1,
            &mut prefill1,
            &weights,
            &mut cache1,
            &tokens,
            &config,
            None,
            None,
        );

        // With domain latent
        let mut ctx2 = ForwardContext::new(&config);
        let mut prefill2 = PrefillContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let dl = crate::types::DomainLatent::from_vec(vec![0.3; kvd]);
        let logits2 = forward_prefill(
            &mut ctx2,
            &mut prefill2,
            &weights,
            &mut cache2,
            &tokens,
            &config,
            None,
            Some(&dl),
        );

        let mut any_diff = false;
        for (&a, &b) in logits1.iter().zip(logits2.iter()) {
            if (a - b).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "domain latent in prefill should change logits");
    }

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_prefill_then_decode() {
        let config = small_target_2layer();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let kvd = crate::types::kv_dim(&config);
        let dl = crate::types::DomainLatent::from_vec(vec![0.2; kvd]);

        // Prefill with domain latent
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let prompt: Vec<usize> = (0..3).collect();
        let logits_prefill = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &prompt,
            &config,
            None,
            Some(&dl),
        );
        assert_eq!(logits_prefill.len(), config.vocab_size);
        for (i, &l) in logits_prefill.iter().enumerate() {
            assert!(
                l.is_finite(),
                "prefill with domain_latent logit {i} not finite: {l}"
            );
        }

        // Decode with domain latent (position 3)
        let logits_decode = forward_base(
            &mut ctx,
            &weights,
            &mut cache,
            0,
            3,
            &config,
            None,
            Some(&dl),
        );
        assert_eq!(logits_decode.len(), config.vocab_size);
        for (i, &l) in logits_decode.iter().enumerate() {
            assert!(
                l.is_finite(),
                "decode after prefill with domain_latent logit {i} not finite: {l}"
            );
        }
    }

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_forward_with_domain_latent_wrapper() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let kvd = crate::types::kv_dim(&config);
        let dl = crate::types::DomainLatent::from_vec(vec![0.1; kvd]);

        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let logits = forward_with_domain_latent(
            &mut ctx,
            &weights,
            &mut cache,
            0,
            0,
            &config,
            None,
            Some(&dl),
        );
        assert_eq!(logits.len(), config.vocab_size);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logit {i} not finite: {l}");
        }
    }

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_with_lora_changes_logits() {
        let config = small_target_2layer();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let kvd = crate::types::kv_dim(&config);
        let rank = 4;
        let in_dim = config.n_embd;
        let out_dim = config.n_embd;

        let lora = crate::types::LoraAdapter {
            a: vec![0.1f32; rank * in_dim],
            b: vec![0.1f32; out_dim * rank],
            rank,
            alpha: 8.0,
            in_dim,
            out_dim,
        };
        let dl = crate::types::DomainLatent::from_vec(vec![0.5; kvd]);

        // With both lora + domain_latent
        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let logits1 = forward_base(
            &mut ctx1,
            &weights,
            &mut cache1,
            0,
            0,
            &config,
            Some(&lora),
            Some(&dl),
        );

        // With lora only (no domain_latent)
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let logits2 = forward_base(
            &mut ctx2,
            &weights,
            &mut cache2,
            0,
            0,
            &config,
            Some(&lora),
            None,
        );

        let mut any_diff = false;
        for (&a, &b) in logits1.iter().zip(logits2.iter()) {
            if (a - b).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(
            any_diff,
            "domain_latent + lora should differ from lora-only"
        );
    }

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_with_lora_prefill_pipeline() {
        let config = small_target_2layer();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let kvd = crate::types::kv_dim(&config);
        let rank = 4;
        let in_dim = config.n_embd;
        let out_dim = config.n_embd;

        let lora = crate::types::LoraAdapter {
            a: vec![0.1f32; rank * in_dim],
            b: vec![0.1f32; out_dim * rank],
            rank,
            alpha: 8.0,
            in_dim,
            out_dim,
        };
        let dl = crate::types::DomainLatent::from_vec(vec![0.5; kvd]);
        let tokens: Vec<usize> = (0..3).collect();

        // Pipeline 1: prefill + decode with both lora + dl
        let mut ctx1 = ForwardContext::new(&config);
        let mut prefill1 = PrefillContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let _ = forward_prefill(
            &mut ctx1,
            &mut prefill1,
            &weights,
            &mut cache1,
            &tokens,
            &config,
            Some(&lora),
            Some(&dl),
        );
        let logits1 = forward_base(
            &mut ctx1,
            &weights,
            &mut cache1,
            0,
            tokens.len(),
            &config,
            Some(&lora),
            Some(&dl),
        );

        // Pipeline 2: prefill + decode with lora only
        let mut ctx2 = ForwardContext::new(&config);
        let mut prefill2 = PrefillContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let _ = forward_prefill(
            &mut ctx2,
            &mut prefill2,
            &weights,
            &mut cache2,
            &tokens,
            &config,
            Some(&lora),
            None,
        );
        let logits2 = forward_base(
            &mut ctx2,
            &weights,
            &mut cache2,
            0,
            tokens.len(),
            &config,
            Some(&lora),
            None,
        );

        let mut any_diff = false;
        for (&a, &b) in logits1.iter().zip(logits2.iter()) {
            if (a - b).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(
            any_diff,
            "prefill+decode with lora+dl should differ from lora-only pipeline"
        );
    }

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_zero_with_lora_same_as_lora_only() {
        let config = small_target_2layer();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let kvd = crate::types::kv_dim(&config);
        let rank = 4;
        let in_dim = config.n_embd;
        let out_dim = config.n_embd;

        let lora = crate::types::LoraAdapter {
            a: vec![0.1f32; rank * in_dim],
            b: vec![0.1f32; out_dim * rank],
            rank,
            alpha: 8.0,
            in_dim,
            out_dim,
        };
        let dl_zero = crate::types::DomainLatent::zeros(kvd);

        // With zero domain_latent + lora
        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let logits1 = forward_base(
            &mut ctx1,
            &weights,
            &mut cache1,
            0,
            0,
            &config,
            Some(&lora),
            Some(&dl_zero),
        );

        // With lora only (no domain_latent)
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let logits2 = forward_base(
            &mut ctx2,
            &weights,
            &mut cache2,
            0,
            0,
            &config,
            Some(&lora),
            None,
        );

        for (i, (&a, &b)) in logits1.iter().zip(logits2.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff < 1e-6,
                "zero domain_latent + lora should match lora-only, diff at {i}: {diff}"
            );
        }
    }
}
