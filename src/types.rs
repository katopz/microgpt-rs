// Shared configuration, RNG, and math utilities.

#[derive(Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub mlp_hidden: usize,
    pub n_layer: usize,
    pub n_kv_head: usize,
    pub bos_token: usize,
    pub temperature: f32,
    pub draft_lookahead: usize,
    pub tree_budget: usize,
    pub parallel_threshold: usize,
    // LoRA fields (Plan 008)
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub lora_dropout: f32,
    pub lora_targets: Vec<String>,
    // Screening Pruner (Plan 021)
    pub screening_threshold: f32,
}

impl Config {
    /// Micro GPT config matching talos-vs-macbook reference:
    /// vocab=27, block=16, n_layer=1, n_head=4, n_embd=16, head_dim=4,
    /// RMSNorm (no learnable gain), ReLU MLP (4x), no biases, untied lm_head.
    pub fn micro() -> Self {
        Self {
            vocab_size: 27,
            block_size: 16,
            n_embd: 16,
            n_head: 4,
            head_dim: 4,
            mlp_hidden: 64,
            n_layer: 1,
            n_kv_head: 4,
            bos_token: 26,
            temperature: 0.5,
            draft_lookahead: 8,
            tree_budget: 16,
            parallel_threshold: 128,
            lora_rank: 4,
            lora_alpha: 8.0,
            lora_dropout: 0.0,
            lora_targets: Vec::new(),
            screening_threshold: 0.0,
        }
    }

    /// Micro config with LoRA defaults (Plan 008).
    pub fn micro_lora() -> Self {
        let mut c = Self::micro();
        c.lora_rank = 4;
        c.lora_alpha = 8.0;
        c.lora_dropout = 0.0;
        c.lora_targets = vec![
            "q".into(),
            "k".into(),
            "v".into(),
            "o".into(),
            "mlp1".into(),
            "mlp2".into(),
        ];
        c
    }

    /// Lightweight draft model for speculative decoding (~4× smaller than target).
    /// Same vocab/block to share embeddings, but embd=4, heads=2, mlp=16.
    pub fn draft() -> Self {
        Self {
            vocab_size: 27,
            block_size: 16,
            n_embd: 4,
            n_head: 2,
            head_dim: 2,
            mlp_hidden: 16,
            n_layer: 1,
            n_kv_head: 2,
            bos_token: 26,
            temperature: 0.5,
            draft_lookahead: 8,
            tree_budget: 16,
            parallel_threshold: 128,
            lora_rank: 4,
            lora_alpha: 8.0,
            lora_dropout: 0.0,
            lora_targets: Vec::new(),
            screening_threshold: 0.0,
        }
    }

    /// Small target model for multi-layer testing.
    /// vocab=4096, block=256, n_layer=4, n_head=4, n_embd=64, head_dim=16,
    /// MLP hidden=256.
    pub fn small_target() -> Self {
        Self {
            vocab_size: 4096,
            block_size: 256,
            n_embd: 64,
            n_head: 4,
            head_dim: 16,
            mlp_hidden: 256,
            n_layer: 4,
            n_kv_head: 4,
            bos_token: 0,
            temperature: 0.8,
            draft_lookahead: 5,
            tree_budget: 32,
            parallel_threshold: 128,
            lora_rank: 4,
            lora_alpha: 8.0,
            lora_dropout: 0.0,
            lora_targets: Vec::new(),
            screening_threshold: 0.0,
        }
    }

    /// GQA draft config: 8 Q heads, 2 KV heads (4:1 ratio, 4× KV cache reduction).
    pub fn gqa_draft() -> Self {
        Self {
            vocab_size: 4096,
            block_size: 256,
            n_embd: 64,
            n_head: 8,
            head_dim: 8,
            mlp_hidden: 256,
            n_layer: 4,
            n_kv_head: 2,
            bos_token: 0,
            temperature: 0.8,
            draft_lookahead: 5,
            tree_budget: 32,
            parallel_threshold: 128,
            lora_rank: 4,
            lora_alpha: 8.0,
            lora_dropout: 0.0,
            lora_targets: Vec::new(),
            screening_threshold: 0.0,
        }
    }

    /// BPE tokenizer config for Rust source code.
    /// vocab=4096, block=256, n_layer=1, n_head=4, n_embd=32, head_dim=8,
    /// MLP hidden=128.
    pub fn bpe() -> Self {
        Self {
            vocab_size: 4096,
            block_size: 256,
            n_embd: 32,
            n_head: 4,
            head_dim: 8,
            mlp_hidden: 128,
            n_layer: 1,
            n_kv_head: 4,
            bos_token: 1,
            temperature: 0.8,
            draft_lookahead: 8,
            tree_budget: 32,
            parallel_threshold: 128,
            lora_rank: 4,
            lora_alpha: 8.0,
            lora_dropout: 0.0,
            lora_targets: Vec::new(),
            screening_threshold: 0.0,
        }
    }

    /// BPE draft model (smaller for speculative decoding).
    /// Same vocab/block as bpe(), but embd=16, heads=2, mlp=64.
    pub fn bpe_draft() -> Self {
        Self {
            vocab_size: 4096,
            block_size: 256,
            n_embd: 16,
            n_head: 2,
            head_dim: 8,
            mlp_hidden: 64,
            n_layer: 1,
            n_kv_head: 2,
            bos_token: 1,
            temperature: 0.8,
            draft_lookahead: 8,
            tree_budget: 32,
            parallel_threshold: 128,
            lora_rank: 4,
            lora_alpha: 8.0,
            lora_dropout: 0.0,
            lora_targets: Vec::new(),
            screening_threshold: 0.0,
        }
    }

    /// Validate config consistency. Returns Err with message on invalid config.
    pub fn validate(&self) -> Result<(), String> {
        if !self.n_head.is_multiple_of(self.n_kv_head) {
            return Err(format!(
                "n_head ({}) must be divisible by n_kv_head ({})",
                self.n_head, self.n_kv_head
            ));
        }
        if self.n_head * self.head_dim != self.n_embd {
            return Err(format!(
                "n_head ({}) * head_dim ({}) must equal n_embd ({})",
                self.n_head, self.head_dim, self.n_embd
            ));
        }
        if self.n_kv_head * self.head_dim > self.n_embd {
            return Err(format!(
                "n_kv_head ({}) * head_dim ({}) must not exceed n_embd ({})",
                self.n_kv_head, self.head_dim, self.n_embd
            ));
        }
        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::micro()
    }
}

/// KV dimension: total float count per token in KV cache.
#[inline(always)]
pub fn kv_dim(config: &Config) -> usize {
    config.n_kv_head * config.head_dim
}

/// XorShift64 PRNG — deterministic per seed.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    #[allow(clippy::should_implement_trait)]
    #[inline(always)]
    pub fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Uniform [0, 1).
    #[inline(always)]
    pub fn uniform(&mut self) -> f32 {
        (self.next() >> 11) as f32 * (1.0 / 9007199254740992.0)
    }

    /// Standard normal via Box-Muller transform.
    #[inline]
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-10);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// In-place softmax. Handles empty slices gracefully.
/// Two-pass: find max (for numerical stability), then exp+sum+normalize fused.
#[inline(always)]
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Pass 1: find max for numerical stability
    let max_val = {
        let mut max = x[0];
        let len = x.len();
        for i in 1..len {
            let v = unsafe { *x.get_unchecked(i) };
            if v > max {
                max = v;
            }
        }
        max
    };

    // Pass 2: exp, sum, and normalize in one loop
    let mut sum = 0.0f32;
    for val in x.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }

    let inv_sum = 1.0 / sum;
    for val in x.iter_mut() {
        *val *= inv_sum;
    }
}

/// In-place RMSNorm (no learnable gain).
/// Two-pass: compute mean-square, then scale.
#[inline(always)]
pub fn rmsnorm(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Pass 1: sum of squares
    let mut sum_sq = 0.0f32;
    for &v in x.iter() {
        sum_sq += v * v;
    }

    // Pass 2: scale
    let inv_rms = 1.0 / (sum_sq / x.len() as f32 + 1e-5).sqrt();
    for val in x.iter_mut() {
        *val *= inv_rms;
    }
}

/// Matrix-vector multiply: output = weight @ input.
/// Weight layout: [rows, cols] row-major.
#[inline(always)]
pub fn matmul(output: &mut [f32], weight: &[f32], input: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row_off = r * cols;
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum +=
                unsafe { *weight.get_unchecked(row_off + c) } * unsafe { *input.get_unchecked(c) };
        }
        unsafe {
            *output.get_unchecked_mut(r) = sum;
        }
    }
}

/// Fused matrix-vector multiply + ReLU: output = max(0, weight @ input).
/// Saves one full buffer scan vs separate matmul + ReLU.
/// Used for MLP hidden layer where activation immediately follows projection.
#[inline(always)]
pub fn matmul_relu(output: &mut [f32], weight: &[f32], input: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row_off = r * cols;
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum +=
                unsafe { *weight.get_unchecked(row_off + c) } * unsafe { *input.get_unchecked(c) };
        }
        // Fused ReLU: clamp to non-negative
        unsafe {
            *output.get_unchecked_mut(r) = sum.max(0.0);
        }
    }
}

/// Sample a token index from a probability distribution using cumulative scan.
#[inline(always)]
pub fn sample_token(probs: &[f32], rng: &mut Rng) -> usize {
    let r = rng.uniform();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len() - 1
}
