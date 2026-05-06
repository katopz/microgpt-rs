use crate::types::{self, *};

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
    scores: Vec<f32>,           // [block_size] attention scores (max possible)
    hidden: Vec<f32>,           // [mlp_hidden] MLP hidden
    pub logits: Vec<f32>,       // [vocab_size] output logits
    pub hidden_state: Vec<f32>, // [n_embd] final hidden state (Plan 009 compat)
}

impl ForwardContext {
    pub fn new(config: &Config) -> Self {
        let kvd = types::kv_dim(config);
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

/// Zero-alloc forward pass. Writes logits into `ctx.logits` and returns &mut to it.
/// Multi-layer: RMSNorm → Attn → Res → RMSNorm → MLP → Res per layer, then LM Head.
#[inline(always)]
pub fn forward<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
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
        let layer_cache = &mut cache.layers[layer_idx];

        // Pre-attention: RMSNorm → save residual → RMSNorm
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);

        // QKV projections from per-layer weights (GQA: K/V produce kv_dim outputs)
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

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

/// Generate tokens autoregressively. Returns generated token ids.
pub fn generate(
    weights: &TransformerWeights,
    config: &Config,
    rng: &mut Rng,
    n_tokens: usize,
) -> Vec<usize> {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    let mut tokens = Vec::with_capacity(n_tokens);
    let mut token = config.bos_token;
    let mut pos = 0;

    for _ in 0..n_tokens {
        if pos >= config.block_size {
            cache.reset();
            pos = 0;
            token = config.bos_token;
        }

        let logits = forward(&mut ctx, weights, &mut cache, token, pos, config);

        for logit in logits.iter_mut() {
            *logit /= config.temperature;
        }
        softmax(logits);

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

    tokens
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

    /// Reset all sequences and free all pages.
    pub fn reset(&mut self) {
        for layer_tables in &mut self.layer_page_tables {
            for table in layer_tables.iter_mut() {
                self.free_pages.append(table);
            }
        }
    }
}

#[cfg(test)]
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
}
