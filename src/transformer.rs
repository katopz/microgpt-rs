use crate::types::*;

/// All transformer weights stored as flat f32 vectors.
/// Layout matches talos-vs-macbook bench_c.c.
pub struct TransformerWeights {
    pub wte: Vec<f32>,     // [vocab_size, n_embd]
    pub wpe: Vec<f32>,     // [block_size, n_embd]
    pub attn_wq: Vec<f32>, // [n_embd, n_embd]
    pub attn_wk: Vec<f32>, // [n_embd, n_embd]
    pub attn_wv: Vec<f32>, // [n_embd, n_embd]
    pub attn_wo: Vec<f32>, // [n_embd, n_embd]
    pub mlp_w1: Vec<f32>,  // [mlp_hidden, n_embd]
    pub mlp_w2: Vec<f32>,  // [n_embd, mlp_hidden]
    pub lm_head: Vec<f32>, // [vocab_size, n_embd]
}

impl TransformerWeights {
    pub fn new(config: &Config, rng: &mut Rng) -> Self {
        let n = config.n_embd;
        let scale = (2.0 / n as f32).sqrt(); // He init

        let mut init =
            |len: usize| -> Vec<f32> { (0..len).map(|_| rng.normal() * scale).collect() };

        Self {
            wte: init(config.vocab_size * n),
            wpe: init(config.block_size * n),
            attn_wq: init(n * n),
            attn_wk: init(n * n),
            attn_wv: init(n * n),
            attn_wo: init(n * n),
            mlp_w1: init(config.mlp_hidden * n),
            mlp_w2: init(n * config.mlp_hidden),
            lm_head: init(config.vocab_size * n),
        }
    }
}

/// KV cache for autoregressive generation.
pub struct KVCache {
    pub key: Vec<f32>,   // [block_size, n_embd]
    pub value: Vec<f32>, // [block_size, n_embd]
}

impl KVCache {
    pub fn new(config: &Config) -> Self {
        let n = config.n_embd;
        Self {
            key: vec![0.0; config.block_size * n],
            value: vec![0.0; config.block_size * n],
        }
    }

    pub fn reset(&mut self) {
        self.key.fill(0.0);
        self.value.fill(0.0);
    }
}

/// Pre-allocated buffers for zero-alloc forward passes.
/// Create once, reuse across calls.
pub struct ForwardContext {
    x: Vec<f32>,          // [n_embd] main activation
    xr: Vec<f32>,         // [n_embd] residual
    xr2: Vec<f32>,        // [n_embd] residual 2
    q: Vec<f32>,          // [n_embd] query
    k: Vec<f32>,          // [n_embd] key
    v: Vec<f32>,          // [n_embd] value
    attn_out: Vec<f32>,   // [n_embd] attention output
    scores: Vec<f32>,     // [block_size] attention scores (max possible)
    hidden: Vec<f32>,     // [mlp_hidden] MLP hidden
    pub logits: Vec<f32>, // [vocab_size] output logits
}

impl ForwardContext {
    pub fn new(config: &Config) -> Self {
        Self {
            x: vec![0.0; config.n_embd],
            xr: vec![0.0; config.n_embd],
            xr2: vec![0.0; config.n_embd],
            q: vec![0.0; config.n_embd],
            k: vec![0.0; config.n_embd],
            v: vec![0.0; config.n_embd],
            attn_out: vec![0.0; config.n_embd],
            scores: vec![0.0; config.block_size],
            hidden: vec![0.0; config.mlp_hidden],
            logits: vec![0.0; config.vocab_size],
        }
    }
}

/// Zero-alloc forward pass. Writes logits into `ctx.logits` and returns &mut to it.
/// Matches bench_c.c: RMSNorm → Attn → Res → RMSNorm → MLP → Res → LM Head.
pub fn forward<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut KVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;

    // 1. Embedding: x = wte[token] + wpe[pos]
    for (i, xi) in ctx.x.iter_mut().enumerate().take(n) {
        *xi = weights.wte[token * n + i] + weights.wpe[pos * n + i];
    }

    // 2. RMSNorm on embedding
    rmsnorm(&mut ctx.x);

    // 3. Save residual → pre-attention RMSNorm
    ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
    rmsnorm(&mut ctx.x);

    // 4. Q, K, V projections
    matmul(&mut ctx.q, &weights.attn_wq, &ctx.x, n, n);
    matmul(&mut ctx.k, &weights.attn_wk, &ctx.x, n, n);
    matmul(&mut ctx.v, &weights.attn_wv, &ctx.x, n, n);

    // Store K, V in cache
    let pos_off = pos * n;
    cache.key[pos_off..pos_off + n].copy_from_slice(&ctx.k[..n]);
    cache.value[pos_off..pos_off + n].copy_from_slice(&ctx.v[..n]);

    // 5. Multi-head attention with causal mask
    let scale = 1.0 / (hd as f32).sqrt();
    ctx.attn_out[..n].fill(0.0);

    for h in 0..config.n_head {
        let h_off = h * hd;
        let t_n = pos + 1;

        // Attention scores (use pre-allocated buffer)
        let scores = &mut ctx.scores[..t_n];
        for (t, score) in scores.iter_mut().enumerate() {
            let mut dot = 0.0f32;
            let k_off = t * n + h_off;
            for d in 0..hd {
                dot += ctx.q[h_off + d] * cache.key[k_off + d];
            }
            *score = dot * scale;
        }
        softmax(scores);

        // Weighted sum of values
        for d in 0..hd {
            let mut val = 0.0f32;
            for (t, &s) in scores.iter().enumerate() {
                val += s * cache.value[t * n + h_off + d];
            }
            ctx.attn_out[h_off + d] = val;
        }
    }

    // 6. Output projection + residual
    // Reuse ctx.x as scratch for matmul output
    matmul(&mut ctx.x, &weights.attn_wo, &ctx.attn_out, n, n);
    for (xi, &ri) in ctx.x.iter_mut().zip(&ctx.xr).take(n) {
        *xi += ri;
    }

    // 7. Save residual → pre-MLP RMSNorm
    ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
    rmsnorm(&mut ctx.x);

    // 8. MLP: W1 → ReLU → W2
    matmul(
        &mut ctx.hidden,
        &weights.mlp_w1,
        &ctx.x,
        config.mlp_hidden,
        n,
    );
    for val in ctx.hidden.iter_mut().take(config.mlp_hidden) {
        *val = val.max(0.0); // ReLU
    }
    matmul(
        &mut ctx.x,
        &weights.mlp_w2,
        &ctx.hidden,
        n,
        config.mlp_hidden,
    );

    // 9. Residual
    for (xi, &ri) in ctx.x.iter_mut().zip(&ctx.xr2).take(n) {
        *xi += ri;
    }

    // 10. LM Head
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
    let mut cache = KVCache::new(config);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_output_size() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = KVCache::new(&config);
        let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_forward_logits_finite() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = KVCache::new(&config);
        let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logit {i} is not finite: {l}");
        }
    }

    #[test]
    fn test_forward_cache_populated() {
        let config = Config::micro();
        let n = config.n_embd;
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = KVCache::new(&config);
        forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        let key_sum: f32 = cache.key[..n].iter().sum();
        let val_sum: f32 = cache.value[..n].iter().sum();
        assert!(key_sum != 0.0, "K cache at pos 0 should be populated");
        assert!(val_sum != 0.0, "V cache at pos 0 should be populated");
    }

    #[test]
    fn test_forward_positions_differ() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = KVCache::new(&config);
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
        let mut cache = KVCache::new(&config);

        // Multiple forward passes with same context should give same results
        let _l1 = forward(&mut ctx, &weights, &mut cache, 0, 0, &config).to_vec();
        let l2 = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        // Note: results differ because cache accumulates, but buffers should not leak
        for &v in l2.iter() {
            assert!(v.is_finite(), "reused context produced non-finite: {v}");
        }
    }
}
