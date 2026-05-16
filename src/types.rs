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
    // Sparse MLP (Plan 022)
    pub sparse_threshold: f32,
    // Early exit (Plan 026: AutoTTS)
    pub early_exit_patience: usize,
    pub early_exit_gap: f32,
    // MTP Drafter thresholds (Plan 055: Gemma 4 MTP)
    pub mtp_activation_threshold: usize,
    pub mtp_cluster_vocab_threshold: usize,
    pub mtp_shared_kv_prompt_threshold: usize,
    pub mtp_cluster_size: usize,
    // HLA Attention (Plan 057: Higher-order Linear Attention)
    pub hla_mode: HlaMode,
    pub hla_normalize: bool,
    pub hla_decay: f32,
}

/// Attention mode for HLA (Higher-order Linear Attention).
///
/// - `Standard`: SDPA with KV cache (default, backward-compatible).
/// - `Hla`: Symmetric second-order linear attention — O(1) per-token memory.
/// - `Ahla`: Asymmetric second-order linear attention — lower state cost than symmetric.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HlaMode {
    #[default]
    Standard,
    /// Symmetric second-order: SK, CQV, mQ accumulators.
    Hla,
    /// Asymmetric second-order: PKV, mK accumulators.
    Ahla,
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
            sparse_threshold: 0.8,
            early_exit_patience: 0,
            early_exit_gap: 0.0,
            mtp_activation_threshold: usize::MAX,
            mtp_cluster_vocab_threshold: usize::MAX,
            mtp_shared_kv_prompt_threshold: usize::MAX,
            mtp_cluster_size: 512,
            hla_mode: HlaMode::Standard,
            hla_normalize: false,
            hla_decay: 1.0,
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

    /// Game config for Bomberman LoRA training (Plan 041).
    /// Tiny Transformer for board state → action prediction.
    /// 10-token vocab: 4 board cells (0-3) + 6 actions (4-9).
    /// 170-token sequences: 169 board cells + 1 action.
    /// ~18K params total, ~1.5K LoRA params (rank=4).
    pub fn game() -> Self {
        Self {
            vocab_size: 10,
            block_size: 170,
            n_embd: 32,
            n_head: 4,
            head_dim: 8,
            mlp_hidden: 128,
            n_layer: 1,
            n_kv_head: 4,
            bos_token: 0,
            temperature: 1.0,
            draft_lookahead: 0,
            tree_budget: 0,
            parallel_threshold: 128,
            lora_rank: 4,
            lora_alpha: 8.0,
            lora_dropout: 0.0,
            lora_targets: vec![
                "q".into(),
                "k".into(),
                "v".into(),
                "o".into(),
                "mlp1".into(),
                "mlp2".into(),
            ],
            screening_threshold: 0.0,
            sparse_threshold: 0.8,
            early_exit_patience: 0,
            early_exit_gap: 0.0,
            mtp_activation_threshold: usize::MAX,
            mtp_cluster_vocab_threshold: usize::MAX,
            mtp_shared_kv_prompt_threshold: usize::MAX,
            mtp_cluster_size: 512,
            hla_mode: HlaMode::Standard,
            hla_normalize: false,
            hla_decay: 1.0,
        }
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
            sparse_threshold: 0.8,
            early_exit_patience: 0,
            early_exit_gap: 0.0,
            mtp_activation_threshold: usize::MAX,
            mtp_cluster_vocab_threshold: usize::MAX,
            mtp_shared_kv_prompt_threshold: usize::MAX,
            mtp_cluster_size: 512,
            hla_mode: HlaMode::Standard,
            hla_normalize: false,
            hla_decay: 1.0,
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
            sparse_threshold: 0.8,
            early_exit_patience: 0,
            early_exit_gap: 0.0,
            mtp_activation_threshold: 64,
            mtp_cluster_vocab_threshold: usize::MAX,
            mtp_shared_kv_prompt_threshold: 128,
            mtp_cluster_size: 512,
            hla_mode: HlaMode::Standard,
            hla_normalize: false,
            hla_decay: 1.0,
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
            sparse_threshold: 0.8,
            early_exit_patience: 0,
            early_exit_gap: 0.0,
            mtp_activation_threshold: 64,
            mtp_cluster_vocab_threshold: usize::MAX,
            mtp_shared_kv_prompt_threshold: 128,
            mtp_cluster_size: 512,
            hla_mode: HlaMode::Standard,
            hla_normalize: false,
            hla_decay: 1.0,
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
            sparse_threshold: 0.8,
            early_exit_patience: 0,
            early_exit_gap: 0.0,
            mtp_activation_threshold: 32,
            mtp_cluster_vocab_threshold: 4096,
            mtp_shared_kv_prompt_threshold: 64,
            mtp_cluster_size: 512,
            hla_mode: HlaMode::Standard,
            hla_normalize: false,
            hla_decay: 1.0,
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
            sparse_threshold: 0.8,
            early_exit_patience: 0,
            early_exit_gap: 0.0,
            mtp_activation_threshold: 16,
            mtp_cluster_vocab_threshold: 4096,
            mtp_shared_kv_prompt_threshold: 64,
            mtp_cluster_size: 512,
            hla_mode: HlaMode::Standard,
            hla_normalize: false,
            hla_decay: 1.0,
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
        // MTP thresholds must be consistent
        if self.mtp_cluster_size == 0 {
            return Err("mtp_cluster_size must be > 0".into());
        }
        Ok(())
    }

    /// Apply per-domain inference overrides, returning a new Config.
    ///
    /// `None` fields are left unchanged; `Some` fields replace the current value.
    /// Used by the router to inject domain-specific budgets from TOML config.
    pub fn with_overrides(&self, overrides: &InferenceOverrides) -> Self {
        let mut c = self.clone();
        if let Some(v) = overrides.tree_budget {
            c.tree_budget = v;
        }
        if let Some(v) = overrides.draft_lookahead {
            c.draft_lookahead = v;
        }
        if let Some(v) = overrides.parallel_threshold {
            c.parallel_threshold = v;
        }
        if let Some(v) = overrides.screening_threshold {
            c.screening_threshold = v;
        }
        if let Some(v) = overrides.temperature {
            c.temperature = v;
        }
        if let Some(v) = overrides.sparse_threshold {
            c.sparse_threshold = v;
        }
        if let Some(v) = overrides.early_exit_patience {
            c.early_exit_patience = v;
        }
        if let Some(v) = overrides.early_exit_gap {
            c.early_exit_gap = v;
        }
        if let Some(v) = overrides.mtp_activation_threshold {
            c.mtp_activation_threshold = v;
        }
        if let Some(v) = overrides.mtp_cluster_vocab_threshold {
            c.mtp_cluster_vocab_threshold = v;
        }
        if let Some(v) = overrides.mtp_shared_kv_prompt_threshold {
            c.mtp_shared_kv_prompt_threshold = v;
        }
        if let Some(v) = overrides.mtp_cluster_size {
            c.mtp_cluster_size = v;
        }
        c
    }
}

/// Override DTO for applying per-domain inference budget to a [`Config`].
///
/// All fields are `Option` — `None` means "keep Config's current value".
/// This is a plain struct (no serde) to keep `microgpt-rs` dependency-free
/// from router/TOML types. Conversion from the router's `InferenceBudget`
/// happens at the router boundary.
///
/// See Plan 026 (AutoTTS Dynamic Inference Budget).
#[derive(Debug, Clone, Default)]
pub struct InferenceOverrides {
    pub tree_budget: Option<usize>,
    pub draft_lookahead: Option<usize>,
    pub parallel_threshold: Option<usize>,
    pub screening_threshold: Option<f32>,
    pub temperature: Option<f32>,
    pub sparse_threshold: Option<f32>,
    pub early_exit_patience: Option<usize>,
    pub early_exit_gap: Option<f32>,
    // MTP Drafter overrides (Plan 055: Gemma 4 MTP)
    pub mtp_activation_threshold: Option<usize>,
    pub mtp_cluster_vocab_threshold: Option<usize>,
    pub mtp_shared_kv_prompt_threshold: Option<usize>,
    pub mtp_cluster_size: Option<usize>,
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
/// Three-pass: find max → exp+sum → normalize.
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

    // Pass 2: exp(x - max) + accumulate sum
    let mut sum = 0.0f32;
    for val in x.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }

    // Pass 3: normalize
    let inv_sum = 1.0 / sum;
    for val in x.iter_mut() {
        *val *= inv_sum;
    }
}

/// In-place softmax with temperature scaling: `softmax(x / temperature)`.
///
/// Fuses the temperature division into the exp computation, saving one full pass
/// vs separate `for p /= temp; softmax(x)`.
///
/// `inv_temp` should be `1.0 / temperature` — compute once, pass to every call.
#[inline(always)]
pub fn softmax_scaled(x: &mut [f32], inv_temp: f32) {
    if x.is_empty() {
        return;
    }

    // Pass 1: find max for numerical stability (on raw values, before temp scaling)
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

    // Pass 2: exp((x - max) * inv_temp) + accumulate sum
    let mut sum = 0.0f32;
    for val in x.iter_mut() {
        *val = ((*val - max_val) * inv_temp).exp();
        sum += *val;
    }

    // Pass 3: normalize
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

    // Pass 1: sum of squares (SIMD-accelerated)
    let sum_sq = crate::simd::simd_dot_f32(x, x, x.len());

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
    crate::simd::simd_matmul_rows(output, weight, input, rows, cols);
}

/// Fused matrix-vector multiply + ReLU: output = max(0, weight @ input).
/// Saves one full buffer scan vs separate matmul + ReLU.
/// Used for MLP hidden layer where activation immediately follows projection.
#[inline(always)]
pub fn matmul_relu(output: &mut [f32], weight: &[f32], input: &[f32], rows: usize, cols: usize) {
    crate::simd::simd_matmul_relu_rows(output, weight, input, rows, cols);
}

/// Sparse matrix-vector multiply for ReLU-activated inputs (TwELL-inspired).
///
/// Only processes columns where `input[c] > 0.0`, skipping dead neurons entirely.
/// Exploits the natural sparsity of ReLU activations in MLP layers where 95-99%
/// of hidden neurons are exactly zero after training with L1 regularization.
///
/// Distilled from "Sparser, Faster, Lighter Transformer Language Models"
/// (arXiv:2603.23198) by Sakana AI & NVIDIA.
///
/// Two-phase execution:
/// 1. Dynamic Packing: scan input, store non-zero indices & values into pre-allocated buffers
/// 2. Sparse Multiply: only iterate weights at alive column indices
///
/// Returns the number of alive (non-zero) neurons for diagnostics/threshold checks.
/// Buffers `active_indices` and `active_values` must be pre-allocated to at least `cols` capacity.
#[cfg(feature = "sparse_mlp")]
#[inline(always)]
pub fn sparse_matmul(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
    active_indices: &mut [usize],
    active_values: &mut [f32],
) -> usize {
    // Phase 1: Pack alive neurons (software TwELL formulation)
    let mut alive = 0;
    for c in 0..cols {
        if unsafe { *input.get_unchecked(c) } > 0.0 {
            unsafe {
                *active_indices.get_unchecked_mut(alive) = c;
                *active_values.get_unchecked_mut(alive) = *input.get_unchecked(c);
            }
            alive += 1;
        }
    }

    // Phase 2: Sparse multiply — SIMD-accelerated (Plan 060 T5)
    // NEON gathers 4 elements/iter, AVX2 gathers 8 elements/iter via hardware gather.
    // Scalar fallback for alive ≤ 4 (gather overhead exceeds benefit).
    crate::simd::simd_sparse_matmul_rows(
        output,
        weight,
        active_indices,
        active_values,
        rows,
        cols,
        alive,
    );

    alive
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

// ---------------------------------------------------------------------------
// LoRA Adapter — CPU inference path (Plan 025)
// ---------------------------------------------------------------------------

/// CPU-side LoRA adapter for inference.
/// Loads from the same binary format as `GpuLoraAdapter` (Plan 008):
/// `[LORA(4) | version(4) | blake3(32) | payload...]`
/// where payload = `[n_adapters(4) | rank(4) | alpha(4) | adapter_data...]`
/// and adapter_data = `[in_dim(4) | out_dim(4) | a_f32s | b_f32s]`
///
/// Zero-copy: loaded once per domain, reference-passed during inference.
pub struct LoraAdapter {
    /// Down-projection: [rank × in_dim]
    pub a: Vec<f32>,
    /// Up-projection: [out_dim × rank]
    pub b: Vec<f32>,
    /// LoRA rank.
    pub rank: usize,
    /// Scaling factor (alpha / rank).
    pub alpha: f32,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
}

impl LoraAdapter {
    /// Load a single-adapter LoRA file from the Plan 008 binary format.
    /// For multi-adapter files (multiple targets like Q, K, V), loads the first adapter.
    /// Returns the adapter with its rank, alpha, and weight matrices.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        const LORA_MAGIC: &[u8; 4] = b"LORA";
        const LORA_VERSION: u32 = 1;

        let file_data =
            std::fs::read(path).map_err(|e| format!("Failed to read lora file: {e}"))?;

        if file_data.len() < 44 {
            return Err("File too small for lora header".into());
        }

        if &file_data[0..4] != LORA_MAGIC {
            return Err("Invalid lora magic bytes".into());
        }

        let version = u32::from_le_bytes(
            file_data[4..8]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("Version parse: {e}"))?,
        );
        if version != LORA_VERSION {
            return Err(format!("Unsupported lora version: {version}"));
        }

        let stored_checksum = &file_data[8..40];
        let payload = &file_data[40..];

        let computed = blake3::hash(payload);
        if computed.as_bytes() != stored_checksum {
            return Err("LoRA file checksum mismatch".into());
        }

        let mut offset = 0usize;
        let n_adapters = read_u32_le(payload, &mut offset)?;
        let rank = read_u32_le(payload, &mut offset)? as usize;
        let alpha = read_f32_le(payload, &mut offset)?;

        if n_adapters == 0 {
            return Err("No adapters in lora file".into());
        }

        // Load first adapter
        let in_dim = read_u32_le(payload, &mut offset)? as usize;
        let out_dim = read_u32_le(payload, &mut offset)? as usize;

        let a_count = rank * in_dim;
        let b_count = out_dim * rank;
        let a_bytes = a_count * std::mem::size_of::<f32>();
        let b_bytes = b_count * std::mem::size_of::<f32>();

        if offset + a_bytes + b_bytes > payload.len() {
            return Err("Truncated adapter data".into());
        }

        let a: Vec<f32> = payload[offset..offset + a_bytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
            .collect();
        offset += a_bytes;

        let b: Vec<f32> = payload[offset..offset + b_bytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
            .collect();

        Ok(Self {
            a,
            b,
            rank,
            alpha,
            in_dim,
            out_dim,
        })
    }

    /// Load LoRA adapters from a compact binary format.
    ///
    /// Format:
    /// ```text
    /// [MAGIC: "LORA" 4B]
    /// [VERSION: 1B]
    /// [RANK: 2B LE]
    /// [N_LAYERS: 2B LE]
    /// [N_TARGETS: 2B LE]
    /// [TARGET_IDS: N_TARGETS × 2B LE]  (0=q_proj, 1=k_proj, 2=v_proj, 3=o_proj,
    ///                                    4=gate_proj, 5=up_proj, 6=down_proj)
    /// [LAYER_DATA: for each (layer, target):
    ///   [A_ROWS: 2B][A_COLS: 2B][A_DATA: A_ROWS×A_COLS × 4B f32]
    ///   [B_ROWS: 2B][B_COLS: 2B][B_DATA: B_ROWS×B_COLS × 4B f32]
    /// ]
    /// [BLAKE3_HASH: 32B]  — covers everything before it
    /// ```
    ///
    /// Alpha defaults to `rank * 2`.
    pub fn load_from_bin(path: &std::path::Path) -> Result<Vec<Self>, String> {
        const LORA_MAGIC: &[u8; 4] = b"LORA";
        const LORA_VERSION: u8 = 1;

        let file_data =
            std::fs::read(path).map_err(|e| format!("Failed to read lora bin file: {e}"))?;

        // Minimum: magic(4) + version(1) + rank(2) + n_layers(2) + n_targets(2) + hash(32) = 43
        if file_data.len() < 43 {
            return Err("File too small for lora bin header".into());
        }

        // Validate blake3 checksum — last 32 bytes cover everything before them
        let data_len = file_data.len() - 32;
        let stored_checksum = &file_data[data_len..];
        let computed = blake3::hash(&file_data[..data_len]);
        if computed.as_bytes() != stored_checksum {
            return Err("LoRA bin file checksum mismatch".into());
        }

        let mut offset = 0usize;

        // Magic
        if &file_data[offset..offset + 4] != LORA_MAGIC {
            return Err("Invalid lora bin magic bytes".into());
        }
        offset += 4;

        // Version
        let version = file_data[offset];
        if version != LORA_VERSION {
            return Err(format!("Unsupported lora bin version: {version}"));
        }
        offset += 1;

        // Rank
        let rank = read_u16_le(&file_data, &mut offset)? as usize;

        // N_LAYERS
        let n_layers = read_u16_le(&file_data, &mut offset)? as usize;

        // N_TARGETS
        let n_targets = read_u16_le(&file_data, &mut offset)? as usize;

        if n_layers == 0 || n_targets == 0 {
            return Err("No layers or targets in lora bin file".into());
        }

        // TARGET_IDS
        let mut target_ids = Vec::with_capacity(n_targets);
        for _ in 0..n_targets {
            let tid = read_u16_le(&file_data, &mut offset)?;
            match tid {
                0..=6 => target_ids.push(tid),
                _ => return Err(format!("Invalid target ID: {tid}")),
            }
        }

        // LAYER_DATA
        let alpha = (rank * 2) as f32;
        let mut adapters = Vec::with_capacity(n_layers * n_targets);

        for _layer in 0..n_layers {
            for &_target_id in &target_ids {
                // A matrix: [rank × in_dim]
                let a_rows = read_u16_le(&file_data, &mut offset)? as usize;
                let a_cols = read_u16_le(&file_data, &mut offset)? as usize;
                let a_count = a_rows * a_cols;
                let a_bytes = a_count * std::mem::size_of::<f32>();

                if offset + a_bytes > data_len {
                    return Err("Truncated A matrix data".into());
                }

                let a: Vec<f32> = file_data[offset..offset + a_bytes]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
                    .collect();
                offset += a_bytes;

                // B matrix: [out_dim × rank]
                let b_rows = read_u16_le(&file_data, &mut offset)? as usize;
                let b_cols = read_u16_le(&file_data, &mut offset)? as usize;
                let b_count = b_rows * b_cols;
                let b_bytes = b_count * std::mem::size_of::<f32>();

                if offset + b_bytes > data_len {
                    return Err("Truncated B matrix data".into());
                }

                let b: Vec<f32> = file_data[offset..offset + b_bytes]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
                    .collect();
                offset += b_bytes;

                let in_dim = a_cols;
                let out_dim = b_rows;

                adapters.push(Self {
                    a,
                    b,
                    rank,
                    alpha,
                    in_dim,
                    out_dim,
                });
            }
        }

        if offset != data_len {
            return Err(format!(
                "Unexpected trailing data: read {offset}, expected {data_len}"
            ));
        }

        if adapters.is_empty() {
            return Err("No adapters loaded from lora bin file".into());
        }

        Ok(adapters)
    }
}

/// Apply LoRA delta in-place: `output += (alpha/rank) × B @ (A @ input)`
///
/// `lora_buf` is a pre-allocated `[rank]` intermediate buffer — zero alloc in hot path.
/// The B×hidden multiplication and scaling are fused directly into the output accumulation,
/// avoiding a separate delta buffer.
#[inline(always)]
pub fn lora_apply(output: &mut [f32], lora: &LoraAdapter, input: &[f32], lora_buf: &mut [f32]) {
    let scale = lora.alpha / lora.rank as f32;

    // 1. hidden = A @ input  (rank × in_dim) @ [in_dim] → [rank]
    matmul(lora_buf, &lora.a, input, lora.rank, lora.in_dim);

    // 2. output += scale × (B @ hidden)  — fused, no intermediate delta buffer
    for r in 0..lora.out_dim {
        let row_off = r * lora.rank;
        let mut sum = 0.0f32;
        for k in 0..lora.rank {
            unsafe {
                sum += *lora.b.get_unchecked(row_off + k) * *lora_buf.get_unchecked(k);
            }
        }
        unsafe {
            *output.get_unchecked_mut(r) += scale * sum;
        }
    }
}

/// A loaded LoRA pair for modality-specific inference (Plan 025).
/// Reader is active during bidirectional prefill, writer during causal decode.
/// Switching is a reference swap — zero data movement.
pub struct LoraPair {
    /// LoRA active during bidirectional prefill (e.g., Python Reader).
    pub reader: Option<LoraAdapter>,
    /// LoRA active during causal decode (e.g., Rust Writer).
    pub writer: Option<LoraAdapter>,
}

impl LoraPair {
    /// Empty pair — no LoRA applied.
    pub fn none() -> Self {
        Self {
            reader: None,
            writer: None,
        }
    }
}

/// Domain latent embedding for mid-layer conditioning (Plan 038).
///
/// Injected at layer `n_layer / 2` by adding to K and V projections before cache write.
/// Inspired by the Free Transformer's mid-layer latent injection, adapted for
/// supervised domain conditioning via LoRA fine-tuning.
///
/// Shape: `[kv_dim]` — one embedding per domain, matching K/V dimension for GQA.
///
/// # Binary format
///
/// ```text
/// [MAGIC: "DLAT" 4B][VERSION: 1B][KV_DIM: 4B LE][EMBEDDING: kv_dim × f32 LE][BLAKE3: 32B]
/// ```
///
/// BLAKE3 checksum covers everything before it (magic through embedding).
#[cfg(feature = "domain_latent")]
#[derive(Debug)]
pub struct DomainLatent {
    /// Domain embedding vector, shape `[kv_dim]`.
    pub embedding: Vec<f32>,
}

#[cfg(feature = "domain_latent")]
impl DomainLatent {
    const MAGIC: &[u8; 4] = b"DLAT";
    const VERSION: u8 = 1;

    /// Load domain latent from binary file.
    ///
    /// Format: `[MAGIC 4B][VERSION 1B][KV_DIM 4B LE][EMBEDDING kv_dim×f32 LE][BLAKE3 32B]`
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let data =
            std::fs::read(path).map_err(|e| format!("Failed to read domain_latent file: {e}"))?;

        // Minimum: magic(4) + version(1) + kv_dim(4) + hash(32) = 41
        if data.len() < 41 {
            return Err("File too small for domain_latent header".into());
        }

        // Validate BLAKE3 checksum — last 32 bytes cover everything before them
        let payload_end = data.len() - 32;
        let stored_checksum = &data[payload_end..];
        let computed = blake3::hash(&data[..payload_end]);
        if computed.as_bytes() != stored_checksum {
            return Err("Domain latent file checksum mismatch".into());
        }

        let mut offset = 0usize;

        // Magic
        if &data[offset..offset + 4] != Self::MAGIC {
            return Err("Invalid domain_latent magic bytes".into());
        }
        offset += 4;

        // Version
        let version = data[offset];
        if version != Self::VERSION {
            return Err(format!("Unsupported domain_latent version: {version}"));
        }
        offset += 1;

        // KV_DIM
        let kv_dim = u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("kv_dim parse: {e}"))?,
        ) as usize;
        offset += 4;

        // Embedding data
        let embed_bytes = kv_dim * std::mem::size_of::<f32>();
        if offset + embed_bytes > payload_end {
            return Err(format!(
                "Truncated embedding data: expected {embed_bytes} bytes at offset {offset}, payload ends at {payload_end}"
            ));
        }

        let embedding: Vec<f32> = data[offset..offset + embed_bytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
            .collect();

        if embedding.len() != kv_dim {
            return Err(format!(
                "Embedding length mismatch: got {}, expected {kv_dim}",
                embedding.len()
            ));
        }

        Ok(Self { embedding })
    }

    /// Save domain latent to binary file (for tests and training export).
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let kv_dim = self.embedding.len();
        let embed_bytes = kv_dim * std::mem::size_of::<f32>();
        let payload_len = 4 + 1 + 4 + embed_bytes;
        let mut buf = Vec::with_capacity(payload_len + 32);

        buf.extend_from_slice(Self::MAGIC);
        buf.push(Self::VERSION);
        buf.extend_from_slice(&(kv_dim as u32).to_le_bytes());
        for &val in &self.embedding {
            buf.extend_from_slice(&val.to_le_bytes());
        }

        let hash = blake3::hash(&buf);
        buf.extend_from_slice(hash.as_bytes());

        std::fs::write(path, &buf)
            .map_err(|e| format!("Failed to write domain_latent file: {e}"))?;

        Ok(())
    }

    /// Create a zero-initialized domain latent of the given kv_dim.
    pub fn zeros(kv_dim: usize) -> Self {
        Self {
            embedding: vec![0.0; kv_dim],
        }
    }

    /// Create a domain latent from a raw embedding vector.
    pub fn from_vec(embedding: Vec<f32>) -> Self {
        Self { embedding }
    }
}

fn read_u32_le(data: &[u8], offset: &mut usize) -> Result<u32, String> {
    if *offset + 4 > data.len() {
        return Err("Unexpected end of data reading u32".into());
    }
    let val = u32::from_le_bytes(
        data[*offset..*offset + 4]
            .try_into()
            .map_err(|e: std::array::TryFromSliceError| format!("u32 parse: {e}"))?,
    );
    *offset += 4;
    Ok(val)
}

fn read_f32_le(data: &[u8], offset: &mut usize) -> Result<f32, String> {
    if *offset + 4 > data.len() {
        return Err("Unexpected end of data reading f32".into());
    }
    let val = f32::from_le_bytes(
        data[*offset..*offset + 4]
            .try_into()
            .map_err(|e: std::array::TryFromSliceError| format!("f32 parse: {e}"))?,
    );
    *offset += 4;
    Ok(val)
}

fn read_u16_le(data: &[u8], offset: &mut usize) -> Result<u16, String> {
    if *offset + 2 > data.len() {
        return Err("Unexpected end of data reading u16".into());
    }
    let val = u16::from_le_bytes(
        data[*offset..*offset + 2]
            .try_into()
            .map_err(|e: std::array::TryFromSliceError| format!("u16 parse: {e}"))?,
    );
    *offset += 2;
    Ok(val)
}

/// Output of a single inference pass, with reward signal for feedback loop.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceResult {
    /// Domain that handled this inference.
    pub domain: String,
    /// Best-path reward (max relevance score from WasmPruner).
    pub reward: f32,
    /// Number of nodes explored in DDTree.
    pub tree_budget_used: usize,
    /// Inference budget level (0=cheap, 1=moderate, 2=expensive).
    pub budget_level: u8,
    /// Input prompt hash (for dedup, not stored).
    pub prompt_hash: u64,
    /// Generated output text.
    pub output: String,
    /// Timestamp (Uuid v7 prefix).
    pub timestamp: i64,
    /// Was this result screened out (reward below threshold)?
    pub screened: bool,
}

#[cfg(test)]
mod tests_types {
    use super::*;

    #[test]
    fn test_with_overrides_none_unchanged() {
        let config = Config::draft();
        let overrides = InferenceOverrides::default();
        let result = config.with_overrides(&overrides);
        assert_eq!(result.tree_budget, config.tree_budget);
        assert_eq!(result.draft_lookahead, config.draft_lookahead);
        assert_eq!(result.temperature, config.temperature);
        assert_eq!(result.early_exit_patience, config.early_exit_patience);
        assert_eq!(result.early_exit_gap, config.early_exit_gap);
    }

    #[test]
    fn test_with_overrides_some_applied() {
        let config = Config::draft();
        let overrides = InferenceOverrides {
            tree_budget: Some(5000),
            draft_lookahead: Some(12),
            temperature: None,
            early_exit_patience: Some(5),
            early_exit_gap: Some(2.0),
            ..Default::default()
        };
        let result = config.with_overrides(&overrides);
        assert_eq!(result.tree_budget, 5000);
        assert_eq!(result.draft_lookahead, 12);
        assert_eq!(result.temperature, config.temperature); // None → unchanged
        assert_eq!(result.early_exit_patience, 5);
        assert_eq!(result.early_exit_gap, 2.0);
    }

    #[test]
    fn test_with_overrides_all_fields() {
        let config = Config::micro();
        let overrides = InferenceOverrides {
            tree_budget: Some(9999),
            draft_lookahead: Some(15),
            parallel_threshold: Some(256),
            screening_threshold: Some(0.3),
            temperature: Some(0.1),
            sparse_threshold: Some(0.5),
            early_exit_patience: Some(10),
            early_exit_gap: Some(3.0),
            mtp_activation_threshold: None,
            mtp_cluster_vocab_threshold: None,
            mtp_shared_kv_prompt_threshold: None,
            mtp_cluster_size: None,
        };
        let result = config.with_overrides(&overrides);
        assert_eq!(result.tree_budget, 9999);
        assert_eq!(result.draft_lookahead, 15);
        assert_eq!(result.parallel_threshold, 256);
        assert_eq!(result.screening_threshold, 0.3);
        assert_eq!(result.temperature, 0.1);
        assert_eq!(result.sparse_threshold, 0.5);
        assert_eq!(result.early_exit_patience, 10);
        assert_eq!(result.early_exit_gap, 3.0);
    }

    #[test]
    fn test_early_exit_defaults_disabled() {
        let config = Config::draft();
        assert_eq!(config.early_exit_patience, 0);
        assert_eq!(config.early_exit_gap, 0.0);
    }

    // ── DomainLatent (Plan 038) ────────────────────────────────────

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_save_load_roundtrip() {
        let original = DomainLatent::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let dir = std::env::temp_dir().join(format!(
            "microgpt_test_dlat_roundtrip_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.dlat");

        original.save(&path).unwrap();
        let loaded = DomainLatent::load(&path).unwrap();

        assert_eq!(loaded.embedding.len(), 4);
        for (i, (&o, &l)) in original.embedding.iter().zip(&loaded.embedding).enumerate() {
            assert!(
                (o - l).abs() < 1e-6,
                "embedding[{i}] mismatch: saved {o}, loaded {l}"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_zeros() {
        let dl = DomainLatent::zeros(8);
        assert_eq!(dl.embedding.len(), 8);
        assert!(dl.embedding.iter().all(|&v| v == 0.0));
    }

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_invalid_magic() {
        let dir = std::env::temp_dir().join(format!(
            "microgpt_test_dlat_badmagic_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bad_magic.dlat");

        // Write file with wrong magic
        let mut buf = b"XLAT".to_vec(); // wrong magic
        buf.push(1); // version
        buf.extend_from_slice(&4u32.to_le_bytes()); // kv_dim
        buf.extend_from_slice(&[0u8; 16]); // 4 x f32 zeros
        let hash = blake3::hash(&buf);
        buf.extend_from_slice(hash.as_bytes());
        std::fs::write(&path, &buf).unwrap();

        let result = DomainLatent::load(&path);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("magic"),
            "expected magic error"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_checksum_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "microgpt_test_dlat_checksum_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bad_checksum.dlat");

        let dl = DomainLatent::from_vec(vec![1.0, 2.0, 3.0]);
        dl.save(&path).unwrap();

        // Corrupt one byte in the file
        let mut data = std::fs::read(&path).unwrap();
        let corrupt_idx = 10.min(data.len() - 33); // corrupt a payload byte
        data[corrupt_idx] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();

        let result = DomainLatent::load(&path);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("checksum"),
            "expected checksum error"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_file_too_small() {
        let dir = std::env::temp_dir().join(format!(
            "microgpt_test_dlat_toosmall_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("too_small.dlat");

        std::fs::write(&path, b"DLAT").unwrap(); // only 4 bytes, need >= 41
        let result = DomainLatent::load(&path);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("too small"),
            "expected too small error"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_config_game() {
        let c = Config::game();
        assert_eq!(c.vocab_size, 10); // 4 board cells + 6 actions
        assert_eq!(c.block_size, 170); // 169 board + 1 action
        assert_eq!(c.n_layer, 1);
        assert_eq!(c.n_embd, 32);
        assert_eq!(c.n_head, 4);
        assert_eq!(c.head_dim, 8);
        assert_eq!(c.mlp_hidden, 128);
        assert!(c.block_size >= 170, "block_size must fit game sequences");
        assert_eq!(c.lora_rank, 4);
        assert!(!c.lora_targets.is_empty(), "game config needs LoRA targets");
    }
}
