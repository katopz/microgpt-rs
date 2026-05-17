//! Vanilla Transformer with ReGLU FFN and CHT Hull KV Cache.
//!
//! Pure Rust inference engine for Percepta's transformer-vm.
//! Implements autoregressive token generation using O(log N) hard attention
//! via the Convex Hull Trick (CHT).
//!
//! # Architecture
//!
//! Each generation step:
//! 1. Embedding lookup + position encoding (slots 0, 1, 2)
//! 2. For each layer: Attention (CHT hull cache) + ReGLU FFN
//! 3. Unembedding (argmax) → next token
//!
//! # Key Features
//!
//! - O(log N) per step via CHT hull cache (vs O(N) standard attention)
//! - ReGLU FFN: `relu(gate) * value` activation
//! - 2D keys/values per head for geometric attention
//!
//! Ported from `transformer.py` + `transformer.cpp` in Percepta's transformer-vm
//! (Apache-2.0 © Percepta).

use std::collections::HashMap;

use super::hull::HardAttentionHead;
use super::types::TieBreak;
use super::weights::TransformerWeights;

// ── Constants ──────────────────────────────────────────────────

/// Inverse of ln(2), used in position encoding: `1/ln(2) - 1/ln(pos+2)`.
const INV_LOG2: f64 = 1.0 / std::f64::consts::LN_2;

// ── Config ─────────────────────────────────────────────────────

/// Transformer runtime configuration.
///
/// Model dimensions must match the corresponding fields in [`TransformerWeights`].
/// Validated on construction in [`VanillaTransformer::new`].
#[derive(Clone, Debug)]
pub struct TransformerConfig {
    /// Model dimension (typically 36, must equal `weights.d_model`).
    pub d_model: usize,
    /// Number of attention heads per layer (typically 18 = d_model / 2).
    pub n_heads: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// FFN hidden dimension.
    pub d_ffn: usize,
    /// Stop token name (e.g., `"halt"`).
    pub stop_token: String,
    /// Maximum tokens to generate (safety limit).
    pub max_gen: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            d_model: 36,
            n_heads: 18,
            n_layers: 7,
            d_ffn: 36,
            stop_token: "halt".to_string(),
            max_gen: 5000,
        }
    }
}

// ── Vocabulary ─────────────────────────────────────────────────

/// Token vocabulary mapping for encoding/decoding between string tokens and IDs.
#[derive(Clone, Debug)]
pub struct TransformerVocab {
    /// Token name by ID.
    id_to_token: Vec<String>,
    /// Token ID by name.
    token_to_id: HashMap<String, usize>,
    /// Stop token ID (cached for fast comparison in hot loop).
    stop_token_id: usize,
}

impl TransformerVocab {
    /// Build vocabulary from a list of token names.
    ///
    /// The stop token name is looked up in the token list;
    /// defaults to ID 0 if not found.
    pub fn new(tokens: Vec<String>, stop_token: &str) -> Self {
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, tok) in tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), i);
        }
        let stop_token_id = token_to_id.get(stop_token).copied().unwrap_or(0);
        Self {
            id_to_token: tokens,
            token_to_id,
            stop_token_id,
        }
    }

    /// Get token name by ID. Returns `"<???>"` for invalid IDs.
    pub fn token_name(&self, id: usize) -> &str {
        match self.id_to_token.get(id) {
            Some(s) => s.as_str(),
            None => "<???>",
        }
    }

    /// Get token ID by name.
    pub fn token_id(&self, name: &str) -> Option<usize> {
        self.token_to_id.get(name).copied()
    }

    /// Get the stop token ID.
    pub fn stop_token_id(&self) -> usize {
        self.stop_token_id
    }

    /// Get the number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    /// Check if the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }
}

// ── Result ─────────────────────────────────────────────────────

/// Result of autoregressive generation.
#[derive(Clone, Debug)]
pub struct GenerationResult {
    /// Generated token names (including prefix).
    pub tokens: Vec<String>,
    /// Execution trace values decoded from `out(XY)` tokens.
    pub trace: Vec<f64>,
}

// ── VanillaTransformer ─────────────────────────────────────────

/// Vanilla transformer with ReGLU FFN and CHT hull KV cache.
///
/// Implements autoregressive generation for Percepta's transformer-vm,
/// where each "token" represents one step of WASM execution.
///
/// # Example
///
/// ```ignore
/// use percepta::transformer::{VanillaTransformer, TransformerConfig, TransformerVocab};
///
/// let weights = TransformerWeights { /* ... */ };
/// let config = TransformerConfig::default();
/// let vocab = TransformerVocab::new(token_names, "halt");
/// let model = VanillaTransformer::new(weights, config, vocab);
/// let result = model.generate(&prefix, 1000);
/// ```
pub struct VanillaTransformer {
    weights: TransformerWeights,
    config: TransformerConfig,
    vocab: TransformerVocab,
}

impl VanillaTransformer {
    /// Create a new transformer from weights, config, and vocabulary.
    ///
    /// # Panics
    ///
    /// Panics if config dimensions don't match weight dimensions.
    pub fn new(
        weights: TransformerWeights,
        config: TransformerConfig,
        vocab: TransformerVocab,
    ) -> Self {
        assert_eq!(
            config.d_model, weights.d_model,
            "config.d_model ({}) must match weights.d_model ({})",
            config.d_model, weights.d_model
        );
        assert_eq!(
            config.n_heads, weights.n_heads,
            "config.n_heads ({}) must match weights.n_heads ({})",
            config.n_heads, weights.n_heads
        );
        assert_eq!(
            config.n_layers, weights.n_layers,
            "config.n_layers ({}) must match weights.n_layers ({})",
            config.n_layers, weights.n_layers
        );
        assert_eq!(
            config.d_ffn, weights.d_ffn,
            "config.d_ffn ({}) must match weights.d_ffn ({})",
            config.d_ffn, weights.d_ffn
        );
        Self {
            weights,
            config,
            vocab,
        }
    }

    // ── Public API ─────────────────────────────────────────────

    /// Generate tokens autoregressively starting from a prefix.
    ///
    /// Processes the prefix through the model to build the KV cache,
    /// then generates new tokens until the stop token or `max_tokens` is reached.
    ///
    /// Returns the full token sequence (prefix + generated) and decoded trace.
    pub fn generate(&self, prefix: &[String], max_tokens: usize) -> GenerationResult {
        let d = self.config.d_model;
        let n_layers = self.config.n_layers;
        let n_heads = self.config.n_heads;
        let total_heads = n_layers * n_heads;

        // Convert prefix tokens to IDs (skip unknowns)
        let mut token_ids: Vec<usize> = prefix
            .iter()
            .filter_map(|tok| self.vocab.token_id(tok))
            .collect();

        // Empty prefix → nothing to generate
        if token_ids.is_empty() {
            return GenerationResult {
                tokens: vec![],
                trace: vec![],
            };
        }

        // Initialize hull caches: one per (layer, head)
        let mut caches: Vec<HardAttentionHead> =
            (0..total_heads).map(|_| HardAttentionHead::new()).collect();

        // Residual buffer (reused across positions)
        let mut residual = vec![0.0f64; d];

        let plen = token_ids.len();
        let max_gen = max_tokens.min(self.config.max_gen);

        // Process all positions (prefix + generated)
        for pos in 0..(plen + max_gen) {
            // Embedding lookup
            residual.copy_from_slice(&self.weights.embedding[token_ids[pos]]);

            // Position encoding
            add_position_encoding(&mut residual, pos);

            // Forward pass through all layers + prediction
            let predicted = self.forward_step(&mut residual, pos, &mut caches);

            // At the boundary, append predicted token
            if pos + 1 == token_ids.len() {
                token_ids.push(predicted);
                if predicted == self.vocab.stop_token_id {
                    break;
                }
            }
        }

        // Build result
        let tokens: Vec<String> = token_ids
            .iter()
            .map(|&id| self.vocab.token_name(id).to_string())
            .collect();

        let trace = decode_trace(&tokens);

        GenerationResult { tokens, trace }
    }

    /// Get a reference to the transformer weights.
    pub fn weights(&self) -> &TransformerWeights {
        &self.weights
    }

    /// Get a reference to the transformer config.
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Get a reference to the vocabulary.
    pub fn vocab(&self) -> &TransformerVocab {
        &self.vocab
    }

    // ── Core Computation ───────────────────────────────────────

    /// Single forward pass: layers → unembedding → argmax.
    ///
    /// Processes the residual through all transformer layers (attention + FFN),
    /// then predicts the next token via argmax over the unembedding matrix.
    ///
    /// `residual` is modified in place (layer outputs accumulated via residual connections).
    /// `pos` is the current absolute position (used for sequence numbering in KV cache).
    /// `caches` holds all hull heads indexed as `[layer * n_heads + head]`.
    fn forward_step(
        &self,
        residual: &mut [f64],
        pos: usize,
        caches: &mut [HardAttentionHead],
    ) -> usize {
        let d = self.config.d_model;
        let n_layers = self.config.n_layers;
        let n_heads = self.config.n_heads;
        let d_ffn = self.config.d_ffn;

        // Scratch buffers (allocated once per forward step, reused across layers)
        let mut qkv = vec![0.0f64; 3 * d];
        let mut head_out = vec![0.0f64; d];
        let mut sublayer_out = vec![0.0f64; d];
        let mut ff = vec![0.0f64; 2 * d_ffn];
        let mut gated = vec![0.0f64; d_ffn];

        for layer_idx in 0..n_layers {
            let seq = (pos * n_layers + layer_idx) as i64;
            let head_start = layer_idx * n_heads;

            // Attention sublayer
            self.apply_attention(
                layer_idx,
                residual,
                seq,
                &mut caches[head_start..head_start + n_heads],
                &mut qkv,
                &mut head_out,
                &mut sublayer_out,
            );

            // Residual connection
            for i in 0..d {
                residual[i] += sublayer_out[i];
            }

            // FFN sublayer
            self.apply_ffn(layer_idx, residual, &mut ff, &mut gated, &mut sublayer_out);

            // Residual connection
            for i in 0..d {
                residual[i] += sublayer_out[i];
            }
        }

        self.predict_token(residual)
    }

    /// Apply attention sublayer for one layer (all heads).
    ///
    /// Computes QKV projection, inserts key-value pairs into hull caches,
    /// queries for maximum attention values, and applies output projection.
    ///
    /// The result is written to `out` (caller handles residual addition).
    #[allow(clippy::too_many_arguments)]
    fn apply_attention(
        &self,
        layer_idx: usize,
        residual: &[f64],
        seq: i64,
        heads: &mut [HardAttentionHead],
        qkv: &mut [f64],
        head_out: &mut [f64],
        out: &mut [f64],
    ) {
        let d = self.config.d_model;
        let n_heads = self.config.n_heads;
        let layer = &self.weights.layers[layer_idx];

        // QKV projection: [3*d, d] @ [d] → [3*d]
        matvec(&layer.attention.in_proj, residual, qkv);

        let (q_slice, rest) = qkv.split_at(d);
        let (k_slice, v_slice) = rest.split_at(d);

        // Zero head output buffer
        head_out.fill(0.0);

        // Process each attention head (2D key/value/query per head)
        for h in 0..n_heads {
            let key = [k_slice[h * 2], k_slice[h * 2 + 1]];
            let val = [v_slice[h * 2], v_slice[h * 2 + 1]];
            let query = [q_slice[h * 2], q_slice[h * 2 + 1]];

            let tie_break = self.get_tie_break(layer_idx, h);

            // Insert key-value pair into hull cache
            heads[h].insert(key, val, seq);

            // Query hull cache for maximum attention value
            if let Some(result) = heads[h].query(query, tie_break) {
                head_out[h * 2] = result[0];
                head_out[h * 2 + 1] = result[1];
            }
        }

        // Output projection: [d, d] @ [d] → [d]
        matvec(&layer.attention.out_proj, head_out, out);
    }

    /// Apply FFN sublayer for one layer with ReGLU activation.
    ///
    /// Computes: `ff_out(relu(ff_in_gate(x)) * ff_in_value(x))`
    ///
    /// The result is written to `out` (caller handles residual addition).
    fn apply_ffn(
        &self,
        layer_idx: usize,
        residual: &[f64],
        ff: &mut [f64],
        gated: &mut [f64],
        out: &mut [f64],
    ) {
        let d_ffn = self.config.d_ffn;
        let layer = &self.weights.layers[layer_idx];

        // FFN input: [2*d_ffn, d] @ [d] → [2*d_ffn]
        matvec(&layer.ffn.ff_in, residual, ff);

        // ReGLU: relu(gate) * value
        let (gate, value) = ff.split_at(d_ffn);
        for i in 0..d_ffn {
            gated[i] = relu(gate[i]) * value[i];
        }

        // FFN output: [d_model, d_ffn] @ [d_ffn] → [d_model]
        matvec(&layer.ffn.ff_out, gated, out);
    }

    /// Predict the next token via argmax over unembedding scores.
    ///
    /// Computes `argmax(unembedding @ residual)` — the token whose
    /// unembedding row has the highest dot product with the residual.
    fn predict_token(&self, residual: &[f64]) -> usize {
        let vocab_size = self.weights.vocab_size;
        let mut best_id = 0;
        let mut best_score = f64::NEG_INFINITY;

        for tok_id in 0..vocab_size {
            let score = dot_product(&self.weights.unembedding[tok_id], residual);
            if score > best_score {
                best_score = score;
                best_id = tok_id;
            }
        }

        best_id
    }

    /// Get tie-break mode for a specific (layer, head) pair.
    ///
    /// Returns [`TieBreak::Latest`] if the head's tiebreak flag is set,
    /// [`TieBreak::Average`] otherwise (including when metadata is missing).
    fn get_tie_break(&self, layer_idx: usize, head: usize) -> TieBreak {
        match self.weights.head_tiebreak.get(layer_idx) {
            Some(row) => match row.get(head) {
                Some(true) => TieBreak::Latest,
                _ => TieBreak::Average,
            },
            None => TieBreak::Average,
        }
    }
}

// ── Position Encoding ──────────────────────────────────────────

/// Add deterministic position features to the residual stream.
///
/// Modifies slots 0, 1, 2 of the residual vector:
/// - `slot[0] += pos`
/// - `slot[1] += 1/ln(2) - 1/ln(pos + 2)`
/// - `slot[2] += pos²`
///
/// This provides the model with absolute position information through
/// three features: linear position, logarithmic inverse position, and
/// quadratic position.
#[inline]
fn add_position_encoding(x: &mut [f64], pos: usize) {
    let pos_f = pos as f64;
    x[0] += pos_f;
    x[1] += INV_LOG2 - 1.0 / (pos_f + 2.0).ln();
    x[2] += pos_f * pos_f;
}

// ── Activation ─────────────────────────────────────────────────

/// ReLU activation: `max(0, x)`.
#[inline]
fn relu(x: f64) -> f64 {
    match x > 0.0 {
        true => x,
        false => 0.0,
    }
}

// ── Linear Algebra ─────────────────────────────────────────────

/// Matrix-vector multiply: `y = W @ x`.
///
/// `W` is `[rows][cols]` (row-major `Vec<Vec<f64>>`),
/// `x` is `[cols]`, `y` is `[rows]`.
#[inline]
fn matvec(w: &[Vec<f64>], x: &[f64], y: &mut [f64]) {
    for (i, row) in w.iter().enumerate() {
        y[i] = dot_product(row, x);
    }
}

/// Dot product of two f64 slices.
#[inline]
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

// ── Token Encoding / Decoding ──────────────────────────────────

/// Decode execution trace from output tokens.
///
/// Extracts byte values from tokens matching the `out(XY)` or `out(C)` pattern,
/// producing a flat list of byte values as f64.
pub fn decode_trace(tokens: &[String]) -> Vec<f64> {
    tokens
        .iter()
        .filter_map(|token| parse_output_byte(token).map(|b| b as f64))
        .collect()
}

/// Encode a byte value as an output token string.
///
/// - Printable ASCII (0x20..=0x7E): `out(C)` where C is the character
/// - All other values: `out(XY)` where XY is lowercase hex
///
/// # Example
///
/// ```ignore
/// use percepta::transformer::encode_output_byte;
/// assert_eq!(encode_output_byte(65), "out(A)");
/// assert_eq!(encode_output_byte(0xFF), "out(ff)");
/// assert_eq!(encode_output_byte(0x0A), "out(0a)");
/// ```
pub fn encode_output_byte(byte: u8) -> String {
    match (0x20..=0x7E).contains(&byte) {
        true => format!("out({})", byte as char),
        false => format!("out({byte:02x})"),
    }
}

/// Parse a byte value from an output token.
///
/// Supports two formats:
/// - `out(C)` (length 6): single ASCII character → its byte value
/// - `out(XY)` (length 7+): lowercase hex string → parsed byte value
///
/// Returns `None` for tokens that don't match the output format.
pub fn parse_output_byte(token: &str) -> Option<u8> {
    match (token.starts_with("out("), token.ends_with(')')) {
        (true, true) if token.len() == 6 => {
            // Single char format: out(A) → ASCII value of 'A'
            Some(token.as_bytes()[4])
        }
        (true, true) if token.len() > 6 => {
            // Hex format: out(4a) → 0x4a
            let content = &token[4..token.len() - 1];
            u8::from_str_radix(content, 16).ok()
        }
        _ => None,
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::weights::{AttentionWeights, FfnWeights, LayerWeights};
    use super::*;

    // ── Helper: build minimal zero weights for unit tests ──────

    fn make_test_weights(
        d: usize,
        n_heads: usize,
        n_layers: usize,
        d_ffn: usize,
        vocab: usize,
    ) -> TransformerWeights {
        TransformerWeights {
            embedding: vec![vec![0.0; d]; vocab],
            unembedding: vec![vec![0.0; d]; vocab],
            layers: (0..n_layers)
                .map(|_| LayerWeights {
                    attention: AttentionWeights {
                        in_proj: vec![vec![0.0; d]; 3 * d],
                        out_proj: vec![vec![0.0; d]; d],
                    },
                    ffn: FfnWeights {
                        ff_in: vec![vec![0.0; d]; 2 * d_ffn],
                        ff_out: vec![vec![0.0; d_ffn]; d],
                    },
                })
                .collect(),
            head_tiebreak: vec![vec![false; n_heads]; n_layers],
            attn_erase: vec![vec![]; n_layers],
            ffn_erase: vec![vec![]; n_layers],
            d_model: d,
            n_heads,
            d_ffn,
            n_layers,
            vocab_size: vocab,
        }
    }

    // ── Position encoding ─────────────────────────────────────

    #[test]
    fn test_position_encoding_pos_zero() {
        let mut x = vec![0.0; 36];
        add_position_encoding(&mut x, 0);
        // pos=0: slot[0] += 0, slot[1] += 1/ln2 - 1/ln2 ≈ 0, slot[2] += 0
        assert!((x[0]).abs() < 1e-12, "slot[0] should be 0");
        assert!((x[1]).abs() < 1e-12, "slot[1] should be ≈0");
        assert!((x[2]).abs() < 1e-12, "slot[2] should be 0");
    }

    #[test]
    fn test_position_encoding_pos_one() {
        let mut x = vec![0.0; 36];
        add_position_encoding(&mut x, 1);
        assert!((x[0] - 1.0).abs() < 1e-12, "slot[0] should be 1");
        let expected_inv = INV_LOG2 - 1.0 / 3.0_f64.ln();
        assert!(
            (x[1] - expected_inv).abs() < 1e-12,
            "slot[1] should be {expected_inv}"
        );
        assert!((x[2] - 1.0).abs() < 1e-12, "slot[2] should be 1");
    }

    #[test]
    fn test_position_encoding_pos_ten() {
        let mut x = vec![1.0; 36];
        add_position_encoding(&mut x, 10);
        assert!((x[0] - 11.0).abs() < 1e-12, "slot[0] should be 11");
        let expected_inv = INV_LOG2 - 1.0 / 12.0_f64.ln();
        assert!(
            (x[1] - 1.0 - expected_inv).abs() < 1e-12,
            "slot[1] mismatch"
        );
        assert!((x[2] - 101.0).abs() < 1e-12, "slot[2] should be 101");
    }

    // ── Activation ─────────────────────────────────────────────

    #[test]
    fn test_relu() {
        assert_eq!(relu(3.0), 3.0);
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(-2.0), 0.0);
        assert_eq!(relu(0.001), 0.001);
        assert_eq!(relu(-0.001), 0.0);
    }

    // ── Linear algebra ─────────────────────────────────────────

    #[test]
    fn test_matvec_identity() {
        let n = 3;
        let w: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let x = vec![2.0, 3.0, 4.0];
        let mut y = vec![0.0; n];
        matvec(&w, &x, &mut y);
        assert!((y[0] - 2.0).abs() < 1e-12);
        assert!((y[1] - 3.0).abs() < 1e-12);
        assert!((y[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_matvec_general() {
        let w = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let x = vec![2.0, 3.0];
        let mut y = vec![0.0; 2];
        matvec(&w, &x, &mut y);
        assert!((y[0] - 8.0).abs() < 1e-12, "1*2 + 2*3 = 8");
        assert!((y[1] - 18.0).abs() < 1e-12, "3*2 + 4*3 = 18");
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-12, "4+10+18=32");
    }

    #[test]
    fn test_dot_product_zero() {
        let a = [0.0; 4];
        let b = [1.0, 2.0, 3.0, 4.0];
        assert!((dot_product(&a, &b)).abs() < 1e-12);
    }

    // ── Token encoding / decoding ──────────────────────────────

    #[test]
    fn test_parse_output_byte_ascii() {
        assert_eq!(parse_output_byte("out(A)"), Some(65));
        assert_eq!(parse_output_byte("out(a)"), Some(97));
        assert_eq!(parse_output_byte("out(0)"), Some(0x30)); // ASCII '0'
        assert_eq!(parse_output_byte("out( )"), Some(0x20)); // ASCII space
    }

    #[test]
    fn test_parse_output_byte_hex() {
        assert_eq!(parse_output_byte("out(4a)"), Some(0x4a));
        assert_eq!(parse_output_byte("out(ff)"), Some(0xff));
        assert_eq!(parse_output_byte("out(00)"), Some(0x00));
        assert_eq!(parse_output_byte("out(0a)"), Some(0x0a));
    }

    #[test]
    fn test_parse_output_byte_invalid() {
        assert_eq!(parse_output_byte("halt"), None);
        assert_eq!(parse_output_byte("out("), None);
        assert_eq!(parse_output_byte("out()"), None);
        assert_eq!(parse_output_byte("push"), None);
        assert_eq!(parse_output_byte(""), None);
    }

    #[test]
    fn test_encode_output_byte_printable() {
        assert_eq!(encode_output_byte(65), "out(A)");
        assert_eq!(encode_output_byte(97), "out(a)");
        assert_eq!(encode_output_byte(48), "out(0)");
        assert_eq!(encode_output_byte(32), "out( )");
    }

    #[test]
    fn test_encode_output_byte_non_printable() {
        assert_eq!(encode_output_byte(0xFF), "out(ff)");
        assert_eq!(encode_output_byte(0x0A), "out(0a)");
        assert_eq!(encode_output_byte(0x00), "out(00)");
        assert_eq!(encode_output_byte(0x7F), "out(7f)");
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        for byte in 0u8..=255 {
            let encoded = encode_output_byte(byte);
            let decoded = parse_output_byte(&encoded);
            assert_eq!(
                decoded,
                Some(byte),
                "roundtrip failed for byte {byte}: encoded={encoded}"
            );
        }
    }

    #[test]
    fn test_decode_trace() {
        let tokens = vec![
            "push".to_string(),
            "out(48)".to_string(), // 'H' = 0x48
            "out(65)".to_string(), // 'e' = 0x65
            "halt".to_string(),
        ];
        let trace = decode_trace(&tokens);
        assert_eq!(trace, vec![0x48 as f64, 0x65 as f64]);
    }

    #[test]
    fn test_decode_trace_empty() {
        let tokens: Vec<String> = vec!["push".to_string(), "halt".to_string()];
        let trace = decode_trace(&tokens);
        assert!(trace.is_empty());
    }

    // ── Vocabulary ─────────────────────────────────────────────

    #[test]
    fn test_vocab_basic() {
        let tokens = vec!["halt".to_string(), "push".to_string(), "add".to_string()];
        let vocab = TransformerVocab::new(tokens, "halt");
        assert_eq!(vocab.stop_token_id(), 0);
        assert_eq!(vocab.token_id("push"), Some(1));
        assert_eq!(vocab.token_id("add"), Some(2));
        assert_eq!(vocab.token_id("unknown"), None);
        assert_eq!(vocab.token_name(0), "halt");
        assert_eq!(vocab.token_name(2), "add");
        assert_eq!(vocab.token_name(99), "<???>");
        assert_eq!(vocab.len(), 3);
        assert!(!vocab.is_empty());
    }

    #[test]
    fn test_vocab_missing_stop_token() {
        let tokens = vec!["a".to_string(), "b".to_string()];
        let vocab = TransformerVocab::new(tokens, "halt");
        // Stop token not found → defaults to 0
        assert_eq!(vocab.stop_token_id(), 0);
    }

    // ── Config ─────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let config = TransformerConfig::default();
        assert_eq!(config.d_model, 36);
        assert_eq!(config.n_heads, 18);
        assert_eq!(config.n_layers, 7);
        assert_eq!(config.d_ffn, 36);
        assert_eq!(config.stop_token, "halt");
        assert_eq!(config.max_gen, 5000);
    }

    // ── ReGLU ──────────────────────────────────────────────────

    #[test]
    fn test_reglu_activation() {
        // Gate: [2.0, -1.0, 0.5]
        // Value: [3.0, 4.0, 5.0]
        // ReGLU: relu(gate) * value = [6.0, 0.0, 2.5]
        let gate = [2.0, -1.0, 0.5];
        let value = [3.0, 4.0, 5.0];
        let expected = [6.0, 0.0, 2.5];
        for i in 0..3 {
            let result = relu(gate[i]) * value[i];
            assert!(
                (result - expected[i]).abs() < 1e-12,
                "ReGLU[{i}]: expected {}, got {result}",
                expected[i]
            );
        }
    }

    // ── Transformer integration ────────────────────────────────

    #[test]
    fn test_generate_empty_prefix() {
        let weights = make_test_weights(4, 2, 1, 4, 3);
        let config = TransformerConfig {
            d_model: 4,
            n_heads: 2,
            n_layers: 1,
            d_ffn: 4,
            stop_token: "halt".to_string(),
            max_gen: 10,
        };
        let vocab = TransformerVocab::new(
            vec!["halt".to_string(), "a".to_string(), "b".to_string()],
            "halt",
        );
        let model = VanillaTransformer::new(weights, config, vocab);
        let result = model.generate(&[], 10);
        assert!(result.tokens.is_empty());
        assert!(result.trace.is_empty());
    }

    #[test]
    fn test_generate_with_zero_weights() {
        // All weights are zero → all logits are zero → argmax picks token 0 (halt)
        let d = 4;
        let n_heads = 2;
        let n_layers = 1;
        let d_ffn = 4;
        let vocab_size = 3;
        let weights = make_test_weights(d, n_heads, n_layers, d_ffn, vocab_size);
        let config = TransformerConfig {
            d_model: d,
            n_heads,
            n_layers,
            d_ffn,
            stop_token: "halt".to_string(),
            max_gen: 10,
        };
        let vocab = TransformerVocab::new(
            vec!["halt".to_string(), "a".to_string(), "b".to_string()],
            "halt",
        );
        let model = VanillaTransformer::new(weights, config, vocab);
        let prefix = vec!["a".to_string()];
        let result = model.generate(&prefix, 10);
        // With all-zero weights, first generated token should be ID 0 (halt)
        assert!(
            result.tokens.len() >= 2,
            "should have prefix + at least halt"
        );
        assert_eq!(result.tokens.last().unwrap(), "halt");
    }

    #[test]
    fn test_generate_unknown_tokens_skipped() {
        let d = 4;
        let weights = make_test_weights(d, 2, 1, d, 2);
        let config = TransformerConfig {
            d_model: d,
            n_heads: 2,
            n_layers: 1,
            d_ffn: d,
            stop_token: "halt".to_string(),
            max_gen: 5,
        };
        let vocab = TransformerVocab::new(vec!["halt".to_string(), "a".to_string()], "halt");
        let model = VanillaTransformer::new(weights, config, vocab);
        // "unknown" is not in vocab → skipped → empty prefix → empty result
        let result = model.generate(&["unknown".to_string()], 5);
        assert!(result.tokens.is_empty());
    }

    #[test]
    fn test_dimension_mismatch_panics() {
        let weights = make_test_weights(4, 2, 1, 4, 3);
        let wrong_config = TransformerConfig {
            d_model: 99, // mismatch
            n_heads: 2,
            n_layers: 1,
            d_ffn: 4,
            stop_token: "halt".to_string(),
            max_gen: 10,
        };
        let vocab = TransformerVocab::new(vec!["halt".to_string()], "halt");
        let result = std::panic::catch_unwind(|| {
            VanillaTransformer::new(weights, wrong_config, vocab);
        });
        assert!(result.is_err(), "should panic on dimension mismatch");
    }

    #[test]
    fn test_predict_token_argmax() {
        let d = 4;
        let vocab_size = 3;
        let mut weights = make_test_weights(d, 2, 1, d, vocab_size);
        // Set unembedding so token 2 has the highest score for residual [1,0,0,0]
        weights.unembedding[0] = vec![0.1, 0.0, 0.0, 0.0];
        weights.unembedding[1] = vec![0.5, 0.0, 0.0, 0.0];
        weights.unembedding[2] = vec![1.0, 0.0, 0.0, 0.0]; // highest dot with [1,0,0,0]
        let config = TransformerConfig {
            d_model: d,
            n_heads: 2,
            n_layers: 1,
            d_ffn: d,
            stop_token: "c".to_string(),
            max_gen: 10,
        };
        let vocab = TransformerVocab::new(
            vec!["halt".to_string(), "a".to_string(), "c".to_string()],
            "c",
        );
        let model = VanillaTransformer::new(weights, config, vocab);
        let residual = vec![1.0, 0.0, 0.0, 0.0];
        let predicted = model.predict_token(&residual);
        assert_eq!(predicted, 2, "should predict token 2 (highest score)");
    }

    #[test]
    fn test_get_tie_break_default() {
        let weights = make_test_weights(4, 2, 1, 4, 3);
        let config = TransformerConfig {
            d_model: 4,
            n_heads: 2,
            n_layers: 1,
            d_ffn: 4,
            stop_token: "halt".to_string(),
            max_gen: 10,
        };
        let vocab = TransformerVocab::new(vec!["halt".to_string()], "halt");
        let model = VanillaTransformer::new(weights, config, vocab);
        // Default: no tiebreak metadata → Average
        assert_eq!(model.get_tie_break(0, 0), TieBreak::Average);
        assert_eq!(model.get_tie_break(0, 1), TieBreak::Average);
    }

    #[test]
    fn test_get_tie_break_with_metadata() {
        let mut weights = make_test_weights(4, 2, 1, 4, 3);
        weights.head_tiebreak = vec![vec![false, true]]; // head 1 = Latest
        let config = TransformerConfig {
            d_model: 4,
            n_heads: 2,
            n_layers: 1,
            d_ffn: 4,
            stop_token: "halt".to_string(),
            max_gen: 10,
        };
        let vocab = TransformerVocab::new(vec!["halt".to_string()], "halt");
        let model = VanillaTransformer::new(weights, config, vocab);
        assert_eq!(model.get_tie_break(0, 0), TieBreak::Average);
        assert_eq!(model.get_tie_break(0, 1), TieBreak::Latest);
    }

    #[test]
    fn test_accessors() {
        let weights = make_test_weights(4, 2, 1, 4, 3);
        let config = TransformerConfig {
            d_model: 4,
            n_heads: 2,
            n_layers: 1,
            d_ffn: 4,
            stop_token: "halt".to_string(),
            max_gen: 10,
        };
        let vocab = TransformerVocab::new(vec!["halt".to_string()], "halt");
        let model = VanillaTransformer::new(weights, config, vocab);
        assert_eq!(model.weights().d_model, 4);
        assert_eq!(model.config().n_heads, 2);
        assert_eq!(model.vocab().len(), 1);
    }
}
