//! Compressed KV cache using TurboQuant quantization.
//!
//! Stores K and V tensors in bit-packed format:
//! - 2-bit: 4 values per u8
//! - 3-bit: stored as 4-bit (2 per u8) for simplicity
//! - 4-bit: 2 values per u8

use super::codebook::compute_codebook;
use super::rotation::{generate_qjl_matrix, generate_rotation_matrix};
use super::types::{TurboQuantKVCacheConfig, TurboQuantLayer};
use crate::simd::simd_scale_inplace;
use crate::types;

/// Compressed KV cache using TurboQuant quantization.
///
/// Each KV vector is: normalized → rotated → quantized → bit-packed.
/// Reconstruction: unpack → dequantize → inverse rotate → rescale.
pub struct TurboQuantKVCache {
    /// Per-layer quantization state (rotation matrix + codebooks).
    pub layers: Vec<TurboQuantLayer>,
    /// Bit-packed key indices: layers × positions × packed_coords.
    key_indices: Vec<Vec<Vec<u8>>>,
    /// Per-position key norms (for reconstruction).
    key_norms: Vec<Vec<f32>>,
    /// Bit-packed value indices.
    val_indices: Vec<Vec<Vec<u8>>>,
    /// Per-position value norms.
    val_norms: Vec<Vec<f32>>,
    /// Current position.
    pos: usize,
    /// Number of layers.
    n_layers: usize,
    /// KV dimension (n_kv_head * head_dim).
    kv_dim: usize,
    /// Key bits per coordinate.
    key_bits: u8,
    /// Value bits per coordinate.
    val_bits: u8,
    /// Maximum sequence length.
    max_seq_len: usize,
    // ── Scratch buffers for zero-alloc hot path (Plan 051) ──
    /// Normalized input: [kv_dim]. Reused across store/dequantize calls.
    scratch_normalized: Vec<f32>,
    /// Rotation output: [kv_dim]. Reused across store/dequantize calls.
    scratch_rotated: Vec<f32>,
    /// Quantized/unpacked indices: [kv_dim]. Reused across store/dequantize calls.
    scratch_indices: Vec<u8>,
}

impl TurboQuantKVCache {
    /// Create a new compressed KV cache from config.
    pub fn new(config: &types::Config, key_bits: u8, val_bits: u8) -> Self {
        let n_layers = config.n_layer;
        let kv_dim = types::kv_dim(config);
        let max_seq_len = config.block_size;

        // Shared rotation/codebook per layer (same dim → same codebook)
        let key_codebook = compute_codebook(kv_dim, key_bits);
        let val_codebook = compute_codebook(kv_dim, val_bits);
        let rotation = generate_rotation_matrix(kv_dim, 42);
        let qjl_matrix = generate_qjl_matrix(kv_dim, 43);

        let layer = TurboQuantLayer {
            rotation,
            qjl_matrix,
            key_codebook: key_codebook.clone(),
            val_codebook: val_codebook.clone(),
        };
        let layers = vec![layer; n_layers];

        let packed_key_len = packed_len(kv_dim, key_bits);
        let packed_val_len = packed_len(kv_dim, val_bits);

        Self {
            layers,
            key_indices: vec![vec![vec![0u8; packed_key_len]; max_seq_len]; n_layers],
            key_norms: vec![vec![0.0f32; max_seq_len]; n_layers],
            val_indices: vec![vec![vec![0u8; packed_val_len]; max_seq_len]; n_layers],
            val_norms: vec![vec![0.0f32; max_seq_len]; n_layers],
            pos: 0,
            n_layers,
            kv_dim,
            key_bits,
            val_bits,
            max_seq_len,
            scratch_normalized: vec![0.0f32; kv_dim],
            scratch_rotated: vec![0.0f32; kv_dim],
            scratch_indices: vec![0u8; kv_dim],
        }
    }

    /// Create from explicit config struct.
    pub fn with_config(tq_config: &TurboQuantKVCacheConfig) -> Self {
        let key_codebook = compute_codebook(tq_config.kv_dim, tq_config.key_bits);
        let val_codebook = compute_codebook(tq_config.kv_dim, tq_config.val_bits);
        let rotation = generate_rotation_matrix(tq_config.kv_dim, tq_config.seed);
        let qjl_matrix = generate_qjl_matrix(tq_config.kv_dim, tq_config.seed.wrapping_add(1));

        let layer = TurboQuantLayer {
            rotation,
            qjl_matrix,
            key_codebook: key_codebook.clone(),
            val_codebook: val_codebook.clone(),
        };
        let layers = vec![layer; tq_config.n_layers];

        let packed_key_len = packed_len(tq_config.kv_dim, tq_config.key_bits);
        let packed_val_len = packed_len(tq_config.kv_dim, tq_config.val_bits);

        Self {
            layers,
            key_indices: vec![
                vec![vec![0u8; packed_key_len]; tq_config.kv_dim];
                tq_config.n_layers
            ],
            key_norms: vec![vec![0.0f32; tq_config.kv_dim]; tq_config.n_layers],
            val_indices: vec![
                vec![vec![0u8; packed_val_len]; tq_config.kv_dim];
                tq_config.n_layers
            ],
            val_norms: vec![vec![0.0f32; tq_config.kv_dim]; tq_config.n_layers],
            pos: 0,
            n_layers: tq_config.n_layers,
            kv_dim: tq_config.kv_dim,
            key_bits: tq_config.key_bits,
            val_bits: tq_config.val_bits,
            max_seq_len: tq_config.kv_dim,
            scratch_normalized: vec![0.0f32; tq_config.kv_dim],
            scratch_rotated: vec![0.0f32; tq_config.kv_dim],
            scratch_indices: vec![0u8; tq_config.kv_dim],
        }
    }

    /// Quantize and store a key vector at given layer and position.
    pub fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        debug_assert_eq!(key.len(), self.kv_dim);
        let layer_state = &self.layers[layer];

        // Compute norm
        let norm: f32 = key.iter().map(|x| x * x).sum::<f32>().sqrt();
        self.key_norms[layer][pos] = norm;

        if norm < 1e-8 {
            return;
        }

        // Normalize into scratch buffer (copy + SIMD scale)
        let inv_norm = 1.0 / norm;
        self.scratch_normalized[..key.len()].copy_from_slice(key);
        simd_scale_inplace(&mut self.scratch_normalized, inv_norm);

        // Rotate in-place: R * normalized → scratch_rotated
        mat_vec_into(
            &layer_state.rotation,
            &self.scratch_normalized,
            &mut self.scratch_rotated,
        );

        // Quantize in-place into scratch_indices
        let cb = &layer_state.key_codebook;
        for (i, &v) in self.scratch_rotated.iter().enumerate() {
            unsafe {
                *self.scratch_indices.get_unchecked_mut(i) = cb.quantize(v);
            }
        }

        // Pack into existing buffer (zero-alloc)
        pack_indices_into(
            &self.scratch_indices,
            self.key_bits,
            &mut self.key_indices[layer][pos],
        );
    }

    /// Quantize and store a value vector at given layer and position.
    pub fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        debug_assert_eq!(value.len(), self.kv_dim);
        let layer_state = &self.layers[layer];

        let norm: f32 = value.iter().map(|x| x * x).sum::<f32>().sqrt();
        self.val_norms[layer][pos] = norm;

        if norm < 1e-8 {
            return;
        }

        // Normalize into scratch buffer (copy + SIMD scale)
        let inv_norm = 1.0 / norm;
        self.scratch_normalized[..value.len()].copy_from_slice(value);
        simd_scale_inplace(&mut self.scratch_normalized, inv_norm);

        // Rotate in-place: R * normalized → scratch_rotated
        mat_vec_into(
            &layer_state.rotation,
            &self.scratch_normalized,
            &mut self.scratch_rotated,
        );

        // Quantize in-place into scratch_indices
        let cb = &layer_state.val_codebook;
        for (i, &v) in self.scratch_rotated.iter().enumerate() {
            unsafe {
                *self.scratch_indices.get_unchecked_mut(i) = cb.quantize(v);
            }
        }

        // Pack into existing buffer (zero-alloc)
        pack_indices_into(
            &self.scratch_indices,
            self.val_bits,
            &mut self.val_indices[layer][pos],
        );
    }

    /// Dequantize key at position. Returns reconstructed key vector.
    pub fn dequantize_key(&self, layer: usize, pos: usize) -> Vec<f32> {
        let layer_state = &self.layers[layer];
        let norm = self.key_norms[layer][pos];
        if norm < 1e-8 {
            return vec![0.0; self.kv_dim];
        }

        let indices = unpack_indices(&self.key_indices[layer][pos], self.key_bits, self.kv_dim);
        let rotated: Vec<f32> = indices
            .iter()
            .map(|&i| layer_state.key_codebook.dequantize(i))
            .collect();

        // Inverse rotation: R^T * rotated (orthogonal → transpose = inverse)
        let normalized = mat_vec_t(&layer_state.rotation, &rotated);
        normalized.iter().map(|x| x * norm).collect()
    }

    /// Dequantize value at position. Returns reconstructed value vector.
    pub fn dequantize_value(&self, layer: usize, pos: usize) -> Vec<f32> {
        let layer_state = &self.layers[layer];
        let norm = self.val_norms[layer][pos];
        if norm < 1e-8 {
            return vec![0.0; self.kv_dim];
        }

        let indices = unpack_indices(&self.val_indices[layer][pos], self.val_bits, self.kv_dim);
        let rotated: Vec<f32> = indices
            .iter()
            .map(|&i| layer_state.val_codebook.dequantize(i))
            .collect();
        let normalized = mat_vec_t(&layer_state.rotation, &rotated);
        normalized.iter().map(|x| x * norm).collect()
    }

    /// Dequantize key into pre-allocated buffer. Zero-alloc hot path (Plan 051).
    ///
    /// Uses internal scratch buffers — requires `&mut self`.
    /// Reconstruction: unpack → dequantize → inverse rotate → scale by norm.
    pub fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.kv_dim);
        let layer_state = &self.layers[layer];
        let norm = self.key_norms[layer][pos];

        if norm < 1e-8 {
            out.fill(0.0);
            return;
        }

        // Unpack in-place into scratch_indices
        unpack_indices_into(
            &self.key_indices[layer][pos],
            self.key_bits,
            self.kv_dim,
            &mut self.scratch_indices,
        );

        // Dequantize in-place into scratch_rotated
        let cb = &layer_state.key_codebook;
        for (i, &idx) in self.scratch_indices.iter().enumerate() {
            unsafe {
                *self.scratch_rotated.get_unchecked_mut(i) = cb.dequantize(idx);
            }
        }

        // Inverse rotate in-place: R^T * rotated → scratch_normalized
        mat_vec_t_into(
            &layer_state.rotation,
            &self.scratch_rotated,
            &mut self.scratch_normalized,
        );

        // Scale by norm → output (copy + SIMD scale)
        out.copy_from_slice(&self.scratch_normalized);
        simd_scale_inplace(out, norm);
    }

    /// Dequantize value into pre-allocated buffer. Zero-alloc hot path (Plan 051).
    ///
    /// Uses internal scratch buffers — requires `&mut self`.
    /// Reconstruction: unpack → dequantize → inverse rotate → scale by norm.
    pub fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.kv_dim);
        let layer_state = &self.layers[layer];
        let norm = self.val_norms[layer][pos];

        if norm < 1e-8 {
            out.fill(0.0);
            return;
        }

        // Unpack in-place into scratch_indices
        unpack_indices_into(
            &self.val_indices[layer][pos],
            self.val_bits,
            self.kv_dim,
            &mut self.scratch_indices,
        );

        // Dequantize in-place into scratch_rotated
        let cb = &layer_state.val_codebook;
        for (i, &idx) in self.scratch_indices.iter().enumerate() {
            unsafe {
                *self.scratch_rotated.get_unchecked_mut(i) = cb.dequantize(idx);
            }
        }

        // Inverse rotate in-place: R^T * rotated → scratch_normalized
        mat_vec_t_into(
            &layer_state.rotation,
            &self.scratch_rotated,
            &mut self.scratch_normalized,
        );

        // Scale by norm → output (copy + SIMD scale)
        out.copy_from_slice(&self.scratch_normalized);
        simd_scale_inplace(out, norm);
    }

    /// Reset cache for new sequence.
    pub fn reset(&mut self) {
        for layer in 0..self.n_layers {
            for pos in 0..self.max_seq_len {
                self.key_indices[layer][pos].fill(0);
                self.key_norms[layer][pos] = 0.0;
                self.val_indices[layer][pos].fill(0);
                self.val_norms[layer][pos] = 0.0;
            }
        }
        self.pos = 0;
    }

    /// Bytes stored per token (K + V, all layers).
    pub fn bytes_per_token(&self) -> usize {
        let packed_key = packed_len(self.kv_dim, self.key_bits);
        let packed_val = packed_len(self.kv_dim, self.val_bits);
        let per_layer = packed_key + packed_val + 8; // +8 for two f32 norms
        per_layer * self.n_layers
    }

    /// Compression ratio vs f32 KV cache.
    pub fn compression_ratio(&self) -> f64 {
        let flat_bytes = self.kv_dim * 4 * 2 * self.n_layers; // f32, K+V
        flat_bytes as f64 / self.bytes_per_token() as f64
    }

    /// Get current position.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Set current position (for manual position management).
    pub fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// Get KV dimension.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }
}

// ── Matrix operations ────────────────────────────────────────

/// Matrix-vector multiply: result = M * v (M is dim×dim row-major).
/// Backward-compat wrapper kept for tests. Hot path uses [`mat_vec_into`].
#[allow(dead_code)]
fn mat_vec(m: &[f32], v: &[f32]) -> Vec<f32> {
    let dim = v.len();
    let mut result = vec![0.0f32; dim];
    mat_vec_into(m, v, &mut result);
    result
}

/// In-place matrix-vector multiply: out = M * v (zero-alloc, Plan 051).
fn mat_vec_into(m: &[f32], v: &[f32], out: &mut [f32]) {
    let dim = v.len();
    debug_assert_eq!(out.len(), dim);
    crate::simd::simd_matmul_rows(out, m, v, dim, dim);
}

/// Transpose matrix-vector multiply: result = M^T * v (M is dim×dim row-major).
fn mat_vec_t(m: &[f32], v: &[f32]) -> Vec<f32> {
    let dim = v.len();
    let mut result = vec![0.0f32; dim];
    mat_vec_t_into(m, v, &mut result);
    result
}

/// In-place transpose matrix-vector multiply: out = M^T * v (zero-alloc, Plan 051).
///
/// TODO: Replace with SIMD transpose matmul once available in `crate::simd`.
fn mat_vec_t_into(m: &[f32], v: &[f32], out: &mut [f32]) {
    let dim = v.len();
    debug_assert_eq!(out.len(), dim);
    for (i, out_val) in out.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for j in 0..dim {
            unsafe {
                sum += *m.get_unchecked(j * dim + i) * *v.get_unchecked(j);
            }
        }
        *out_val = sum;
    }
}

// ── Bit packing ──────────────────────────────────────────────

/// Packed byte length for n values at given bits per value.
fn packed_len(n: usize, bits: u8) -> usize {
    match bits {
        2 => n.div_ceil(4),
        3 | 4 => n.div_ceil(2), // 3-bit stored as 4-bit (2 per u8)
        8 => n,
        _ => (n * bits as usize).div_ceil(8),
    }
}

/// Pack quantized indices into bytes.
/// Backward-compat wrapper kept for tests. Hot path uses [`pack_indices_into`].
#[allow(dead_code)]
fn pack_indices(indices: &[u8], bits: u8) -> Vec<u8> {
    match bits {
        2 => {
            let n = indices.len().div_ceil(4);
            let mut packed = vec![0u8; n];
            for (i, &idx) in indices.iter().enumerate() {
                let byte = i / 4;
                let shift = (i % 4) * 2;
                packed[byte] |= (idx & 0x3) << shift;
            }
            packed
        }
        3 | 4 => {
            // Store 3-bit as 4-bit (2 values per u8)
            let n = indices.len().div_ceil(2);
            let mut packed = vec![0u8; n];
            for (i, &idx) in indices.iter().enumerate() {
                let byte = i / 2;
                let shift = (i % 2) * 4;
                packed[byte] |= (idx & 0xF) << shift;
            }
            packed
        }
        8 => indices.to_vec(),
        _ => {
            // Generic bit packing
            let total_bits = indices.len() * bits as usize;
            let n = total_bits.div_ceil(8);
            let mut packed = vec![0u8; n];
            let mut bit_pos = 0usize;
            for &idx in indices {
                let byte_pos = bit_pos / 8;
                let shift = bit_pos % 8;
                packed[byte_pos] |= idx << shift;
                if shift + bits as usize > 8 {
                    packed[byte_pos + 1] |= idx >> (8 - shift);
                }
                bit_pos += bits as usize;
            }
            packed
        }
    }
}

/// Pack indices into pre-allocated buffer (zero-alloc, Plan 051).
fn pack_indices_into(indices: &[u8], bits: u8, out: &mut [u8]) {
    match bits {
        2 => {
            out.fill(0);
            for (i, &idx) in indices.iter().enumerate() {
                let byte = i / 4;
                let shift = (i % 4) * 2;
                unsafe {
                    *out.get_unchecked_mut(byte) |= (idx & 0x3) << shift;
                }
            }
        }
        3 | 4 => {
            out.fill(0);
            for (i, &idx) in indices.iter().enumerate() {
                let byte = i / 2;
                let shift = (i % 2) * 4;
                unsafe {
                    *out.get_unchecked_mut(byte) |= (idx & 0xF) << shift;
                }
            }
        }
        8 => {
            let copy_len = out.len().min(indices.len());
            out[..copy_len].copy_from_slice(&indices[..copy_len]);
        }
        _ => {
            out.fill(0);
            let mut bit_pos = 0usize;
            for &idx in indices {
                let byte_pos = bit_pos / 8;
                let shift = bit_pos % 8;
                unsafe {
                    *out.get_unchecked_mut(byte_pos) |= idx << shift;
                    if shift + bits as usize > 8 {
                        *out.get_unchecked_mut(byte_pos + 1) |= idx >> (8 - shift);
                    }
                }
                bit_pos += bits as usize;
            }
        }
    }
}

/// Unpack bytes back to indices.
fn unpack_indices(packed: &[u8], bits: u8, n: usize) -> Vec<u8> {
    match bits {
        2 => {
            let mut indices = vec![0u8; n];
            for (i, out) in indices.iter_mut().enumerate() {
                let byte = i / 4;
                let shift = (i % 4) * 2;
                if byte < packed.len() {
                    *out = (packed[byte] >> shift) & 0x3;
                }
            }
            indices
        }
        3 | 4 => {
            let mut indices = vec![0u8; n];
            for (i, out) in indices.iter_mut().enumerate() {
                let byte = i / 2;
                let shift = (i % 2) * 4;
                if byte < packed.len() {
                    *out = (packed[byte] >> shift) & 0xF;
                }
            }
            indices
        }
        8 => {
            let mut indices = vec![0u8; n];
            let copy_len = n.min(packed.len());
            indices[..copy_len].copy_from_slice(&packed[..copy_len]);
            indices
        }
        _ => {
            // Generic bit unpacking
            let mut indices = vec![0u8; n];
            let mask = (1u8 << bits) - 1;
            let mut bit_pos = 0usize;
            for idx in indices.iter_mut().take(n) {
                let byte_pos = bit_pos / 8;
                let shift = bit_pos % 8;
                if byte_pos < packed.len() {
                    *idx = (packed[byte_pos] >> shift) & mask;
                    if shift + bits as usize > 8 && byte_pos + 1 < packed.len() {
                        *idx |= (packed[byte_pos + 1] << (8 - shift)) & mask;
                    }
                }
                bit_pos += bits as usize;
            }
            indices
        }
    }
}

/// Unpack indices into pre-allocated buffer (zero-alloc, Plan 051).
fn unpack_indices_into(packed: &[u8], bits: u8, n: usize, out: &mut [u8]) {
    debug_assert!(out.len() >= n);
    match bits {
        2 => {
            for i in 0..n {
                let byte = i / 4;
                let shift = (i % 4) * 2;
                if byte < packed.len() {
                    unsafe {
                        *out.get_unchecked_mut(i) = (*packed.get_unchecked(byte) >> shift) & 0x3;
                    }
                } else {
                    unsafe {
                        *out.get_unchecked_mut(i) = 0;
                    }
                }
            }
        }
        3 | 4 => {
            for i in 0..n {
                let byte = i / 2;
                let shift = (i % 2) * 4;
                if byte < packed.len() {
                    unsafe {
                        *out.get_unchecked_mut(i) = (*packed.get_unchecked(byte) >> shift) & 0xF;
                    }
                } else {
                    unsafe {
                        *out.get_unchecked_mut(i) = 0;
                    }
                }
            }
        }
        8 => {
            let copy_len = n.min(packed.len());
            out[..copy_len].copy_from_slice(&packed[..copy_len]);
            out[copy_len..n].fill(0);
        }
        _ => {
            let mask = (1u8 << bits) - 1;
            let mut bit_pos = 0usize;
            for i in 0..n {
                let byte_pos = bit_pos / 8;
                let shift = bit_pos % 8;
                if byte_pos < packed.len() {
                    unsafe {
                        *out.get_unchecked_mut(i) =
                            (*packed.get_unchecked(byte_pos) >> shift) & mask;
                        if shift + bits as usize > 8 && byte_pos + 1 < packed.len() {
                            *out.get_unchecked_mut(i) |=
                                (*packed.get_unchecked(byte_pos + 1) << (8 - shift)) & mask;
                        }
                    }
                } else {
                    unsafe {
                        *out.get_unchecked_mut(i) = 0;
                    }
                }
                bit_pos += bits as usize;
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Config;

    fn test_config() -> Config {
        Config::micro()
    }

    #[test]
    fn test_kv_cache_roundtrip() {
        let config = test_config();
        let mut cache = TurboQuantKVCache::new(&config, 4, 4);

        let key: Vec<f32> = (0..cache.kv_dim)
            .map(|i| (i as f32 * 0.1 - 1.0).sin())
            .collect();
        let val: Vec<f32> = (0..cache.kv_dim).map(|i| (i as f32 * 0.2).cos()).collect();

        cache.store_key(0, 0, &key);
        cache.store_value(0, 0, &val);

        let recon_key = cache.dequantize_key(0, 0);
        let recon_val = cache.dequantize_value(0, 0);

        // Cosine similarity should be high
        let cos_key = cosine_sim(&key, &recon_key);
        let cos_val = cosine_sim(&val, &recon_val);

        assert!(cos_key > 0.90, "Key cos_sim = {cos_key}, expected > 0.90");
        assert!(cos_val > 0.85, "Val cos_sim = {cos_val}, expected > 0.85");
    }

    #[test]
    fn test_kv_cache_roundtrip_multi_pos() {
        let config = test_config();
        let mut cache = TurboQuantKVCache::new(&config, 4, 4);

        for pos in 0..5 {
            let key: Vec<f32> = (0..cache.kv_dim)
                .map(|i| ((i + pos * 7) as f32 * 0.1).sin())
                .collect();
            let val: Vec<f32> = (0..cache.kv_dim)
                .map(|i| ((i + pos * 11) as f32 * 0.15).cos())
                .collect();
            cache.store_key(0, pos, &key);
            cache.store_value(0, pos, &val);

            let recon_key = cache.dequantize_key(0, pos);
            let cos = cosine_sim(&key, &recon_key);
            assert!(cos > 0.90, "pos={pos} key cos_sim={cos}, expected > 0.90");
        }
    }

    #[test]
    fn test_compression_ratio_4bit() {
        let config = test_config();
        let cache = TurboQuantKVCache::new(&config, 4, 4);
        let ratio = cache.compression_ratio();
        // 4-bit should give ~8x compression (32/4 = 8x, minus norm overhead)
        assert!(ratio > 4.0, "Compression ratio {ratio}, expected > 4.0");
    }

    #[test]
    fn test_compression_ratio_2bit() {
        let config = test_config();
        let cache = TurboQuantKVCache::new(&config, 2, 2);
        let ratio = cache.compression_ratio();
        // 2-bit should give ~16x compression (32/2 = 16x, minus norm overhead)
        assert!(ratio > 6.0, "Compression ratio {ratio}, expected > 6.0");
    }

    #[test]
    fn test_reset_clears() {
        let config = test_config();
        let mut cache = TurboQuantKVCache::new(&config, 4, 4);
        let key = vec![1.0f32; cache.kv_dim];
        cache.store_key(0, 0, &key);
        assert!(cache.key_norms[0][0] > 0.0);

        cache.reset();
        let recon = cache.dequantize_key(0, 0);
        assert!(
            recon.iter().all(|&x| x.abs() < 1e-6),
            "After reset, all values should be ~0"
        );
    }

    #[test]
    fn test_zero_vector_handling() {
        let config = test_config();
        let mut cache = TurboQuantKVCache::new(&config, 4, 4);
        let zeros = vec![0.0f32; cache.kv_dim];
        cache.store_key(0, 0, &zeros);

        let recon = cache.dequantize_key(0, 0);
        assert!(recon.iter().all(|&x| x.abs() < 1e-6));
    }

    #[test]
    fn test_bytes_per_token() {
        let config = test_config();
        let cache = TurboQuantKVCache::new(&config, 4, 4);
        let bpt = cache.bytes_per_token();
        // Should be significantly less than f32 flat: kv_dim * 4 * 2 per layer
        let flat_per_layer = cache.kv_dim * 4 * 2;
        assert!(
            bpt < flat_per_layer * config.n_layer,
            "bytes_per_token {bpt} should be < flat {flat_per_layer} * {n_layers}",
            n_layers = config.n_layer
        );
    }

    #[test]
    fn test_pack_unpack_roundtrip_2bit() {
        let indices: Vec<u8> = vec![0, 1, 2, 3, 0, 2, 1, 3];
        let packed = pack_indices(&indices, 2);
        let unpacked = unpack_indices(&packed, 2, indices.len());
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_pack_unpack_roundtrip_4bit() {
        let indices: Vec<u8> = vec![0, 5, 10, 15, 3, 7, 11, 12];
        let packed = pack_indices(&indices, 4);
        let unpacked = unpack_indices(&packed, 4, indices.len());
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_pack_unpack_roundtrip_3bit() {
        // 3-bit stored as 4-bit
        let indices: Vec<u8> = vec![0, 3, 5, 7, 1, 2, 6, 4];
        let packed = pack_indices(&indices, 3);
        let unpacked = unpack_indices(&packed, 3, indices.len());
        // Values should be preserved (3-bit values fit in 4-bit)
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_dequantize_into() {
        let config = test_config();
        let mut cache = TurboQuantKVCache::new(&config, 4, 4);

        let key: Vec<f32> = (0..cache.kv_dim).map(|i| (i as f32 * 0.3).sin()).collect();
        cache.store_key(0, 0, &key);

        // Zero-alloc _into path (Plan 051): uses scratch buffers, requires &mut self
        let mut buf = vec![0.0f32; cache.kv_dim];
        cache.dequantize_key_into(0, 0, &mut buf);

        // Compare with allocating path (still &self)
        let direct = cache.dequantize_key(0, 0);
        assert_eq!(buf, direct, "zero-alloc _into must match allocating path");
    }

    #[test]
    fn test_multi_layer_independence() {
        // Use 2-layer config (micro() has n_layer=1, insufficient for multi-layer test)
        let config = types::Config {
            n_layer: 2,
            ..test_config()
        };
        let mut cache = TurboQuantKVCache::new(&config, 4, 4);

        let key0: Vec<f32> = (0..cache.kv_dim).map(|i| (i as f32).sin()).collect();
        let key1: Vec<f32> = (0..cache.kv_dim).map(|i| (i as f32).cos()).collect();

        cache.store_key(0, 0, &key0);
        cache.store_key(1, 0, &key1);

        let recon0 = cache.dequantize_key(0, 0);
        let recon1 = cache.dequantize_key(1, 0);

        // Different layers should reconstruct independently
        let cos0 = cosine_sim(&key0, &recon0);
        let cos1 = cosine_sim(&key1, &recon1);
        assert!(cos0 > 0.90, "Layer 0 cos_sim={cos0}");
        assert!(cos1 > 0.90, "Layer 1 cos_sim={cos1}");
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-8 || nb < 1e-8 {
            return 0.0;
        }
        dot / (na * nb)
    }
}
