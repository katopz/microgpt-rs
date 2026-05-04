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
        Self {
            key: vec![0.0; config.block_size * config.n_embd],
            value: vec![0.0; config.block_size * config.n_embd],
        }
    }

    pub fn reset(&mut self) {
        self.key.fill(0.0);
        self.value.fill(0.0);
    }
}

/// Single forward pass. Returns raw logits [vocab_size].
/// Matches bench_c.c architecture: RMSNorm → Attn → Res → RMSNorm → MLP → Res → LM Head.
pub fn forward(
    weights: &TransformerWeights,
    cache: &mut KVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> Vec<f32> {
    let n = config.n_embd;
    let hd = config.head_dim;

    // 1. Embedding: x = wte[token] + wpe[pos]
    let mut x = vec![0.0; n];
    for (i, xi) in x.iter_mut().enumerate() {
        *xi = weights.wte[token * n + i] + weights.wpe[pos * n + i];
    }

    // 2. RMSNorm on embedding (matches C reference)
    rmsnorm(&mut x);

    // 3. Save residual, pre-attention RMSNorm
    let xr = x.clone();
    rmsnorm(&mut x);

    // 4. Q, K, V projections
    let mut q = vec![0.0; n];
    let mut k = vec![0.0; n];
    let mut v = vec![0.0; n];
    matmul(&mut q, &weights.attn_wq, &x, n, n);
    matmul(&mut k, &weights.attn_wk, &x, n, n);
    matmul(&mut v, &weights.attn_wv, &x, n, n);

    // Store K, V in cache
    let pos_off = pos * n;
    cache.key[pos_off..pos_off + n].copy_from_slice(&k);
    cache.value[pos_off..pos_off + n].copy_from_slice(&v);

    // 5. Multi-head attention with causal mask
    let scale = 1.0 / (hd as f32).sqrt();
    let mut attn_out = vec![0.0; n];

    for h in 0..config.n_head {
        let h_off = h * hd;
        let t_n = pos + 1;

        // Attention scores
        let mut scores = vec![0.0; t_n];
        for (t, score) in scores.iter_mut().enumerate() {
            let mut dot = 0.0f32;
            let k_off = t * n + h_off;
            for d in 0..hd {
                dot += q[h_off + d] * cache.key[k_off + d];
            }
            *score = dot * scale;
        }
        softmax(&mut scores);

        // Weighted sum of values
        for d in 0..hd {
            let mut val = 0.0f32;
            for (t, &s) in scores.iter().enumerate() {
                val += s * cache.value[t * n + h_off + d];
            }
            attn_out[h_off + d] = val;
        }
    }

    // 6. Output projection + residual
    matmul(&mut x, &weights.attn_wo, &attn_out, n, n);
    for (xi, &ri) in x.iter_mut().zip(&xr) {
        *xi += ri;
    }

    // 7. Pre-MLP: save residual + RMSNorm
    let xr2 = x.clone();
    rmsnorm(&mut x);

    // 8. MLP: W1 → ReLU → W2
    let mut hidden = vec![0.0; config.mlp_hidden];
    matmul(&mut hidden, &weights.mlp_w1, &x, config.mlp_hidden, n);
    for val in hidden.iter_mut() {
        *val = val.max(0.0); // ReLU
    }
    matmul(&mut x, &weights.mlp_w2, &hidden, n, config.mlp_hidden);

    // 9. Residual
    for (xi, &ri) in x.iter_mut().zip(&xr2) {
        *xi += ri;
    }

    // 10. LM Head
    let mut logits = vec![0.0; config.vocab_size];
    matmul(&mut logits, &weights.lm_head, &x, config.vocab_size, n);

    logits
}

/// Generate tokens autoregressively. Returns generated token ids.
pub fn generate(
    weights: &TransformerWeights,
    config: &Config,
    rng: &mut Rng,
    n_tokens: usize,
) -> Vec<usize> {
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

        let logits = forward(weights, &mut cache, token, pos, config);

        let mut probs = logits;
        for logit in probs.iter_mut() {
            *logit /= config.temperature;
        }
        softmax(&mut probs);

        let next_token = sample_token(&probs, rng);
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
        let mut cache = KVCache::new(&config);
        let logits = forward(&weights, &mut cache, 0, 0, &config);
        assert_eq!(logits.len(), config.vocab_size);
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
}
