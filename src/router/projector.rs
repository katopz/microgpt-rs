//! Embedding projector — dimension projection strategies for KV cache priming.
//!
//! Projects an embedding vector from the retrieval model's dimension (e.g., 768)
//! to the draft model's hidden dimension (`n_embd`, e.g., 64). The projected
//! vector is injected as conditioning context via `dflash_predict_conditioned_with`.
//!
//! # Strategies
//!
//! - [`TruncatePadProjector`] — truncate or zero-pad (default, zero-cost)
//! - [`LinearProjector`] — placeholder for learned linear projection (future)
//!
//! # Why Truncate?
//!
//! The first N dimensions of an embedding often carry the most information
//! (PCA-like). A 768-dim embedding truncated to 16 dims is lossy but preserves
//! the principal components. If this proves insufficient, `LinearProjector` can
//! be trained later using paired (embedding, hidden_state) data.

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Projects an embedding vector to the draft model's hidden dimension.
///
/// Implementations must be `Send + Sync` so the projector can be shared
/// across threads (stored in the router behind an `Arc` or `Box`).
pub trait EmbeddingProjector: Send + Sync {
    /// Project `embedding` to exactly `target_dim` dimensions.
    ///
    /// The returned vector length MUST equal `target_dim`.
    fn project(&self, embedding: &[f32], target_dim: usize) -> Vec<f32>;
}

// ---------------------------------------------------------------------------
// TruncatePadProjector
// ---------------------------------------------------------------------------

/// Strategy 1: Truncate or zero-pad. Zero-cost, no training needed.
///
/// - If `embedding.len() > target_dim`: take the first `target_dim` elements.
/// - If `embedding.len() < target_dim`: zero-pad to `target_dim`.
/// - If `embedding.len() == target_dim`: identity (clone).
/// - If `embedding` is empty: returns all-zeros of length `target_dim`.
pub struct TruncatePadProjector;

impl EmbeddingProjector for TruncatePadProjector {
    fn project(&self, embedding: &[f32], target_dim: usize) -> Vec<f32> {
        match embedding.len().cmp(&target_dim) {
            std::cmp::Ordering::Greater => {
                // Truncate: take first target_dim elements
                embedding[..target_dim].to_vec()
            }
            std::cmp::Ordering::Equal => {
                // Identity: clone
                embedding.to_vec()
            }
            std::cmp::Ordering::Less => {
                // Pad: copy what we have, fill rest with zeros
                let mut result = vec![0.0f32; target_dim];
                result[..embedding.len()].copy_from_slice(embedding);
                result
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LinearProjector (placeholder for future)
// ---------------------------------------------------------------------------

/// Strategy 2: Learned linear projection (future, requires training).
///
/// Computes `output = W * embedding + b` where:
/// - `W`: `[target_dim, embedding_dim]` weight matrix
/// - `b`: `[target_dim]` bias vector
///
/// **NOT implemented in this plan** — placeholder for future extension.
/// Requires paired (embedding, hidden_state) training data from the target model.
pub struct LinearProjector {
    // weights: Vec<f32>,  // [target_dim, embedding_dim]
    // bias: Vec<f32>,     // [target_dim]
}

impl EmbeddingProjector for LinearProjector {
    fn project(&self, embedding: &[f32], target_dim: usize) -> Vec<f32> {
        // Fallback to truncate-pad until weights are trained
        TruncatePadProjector.project(embedding, target_dim)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_768_to_64() {
        let embedding: Vec<f32> = (0..768).map(|i| i as f32 * 0.001).collect();
        let result = TruncatePadProjector.project(&embedding, 64);

        assert_eq!(result.len(), 64);
        // First 64 elements preserved
        for i in 0..64 {
            assert!((result[i] - i as f32 * 0.001).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_pad_32_to_64() {
        let embedding: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
        let result = TruncatePadProjector.project(&embedding, 64);

        assert_eq!(result.len(), 64);
        // First 32 elements preserved
        for i in 0..32 {
            assert!((result[i] - i as f32 * 0.01).abs() < f32::EPSILON);
        }
        // Remaining 32 elements are zero
        for i in 32..64 {
            assert!((result[i]).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_identity_64_to_64() {
        let embedding: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let result = TruncatePadProjector.project(&embedding, 64);

        assert_eq!(result.len(), 64);
        assert_eq!(result, embedding);
    }

    #[test]
    fn test_empty_embedding_produces_zeros() {
        let result = TruncatePadProjector.project(&[], 64);

        assert_eq!(result.len(), 64);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_empty_embedding_zero_target_dim() {
        let result = TruncatePadProjector.project(&[1.0, 2.0, 3.0], 0);

        assert!(result.is_empty());
    }

    #[test]
    fn test_truncate_preserves_sign() {
        let embedding = vec![-1.5, 2.3, -0.7, 0.0, 4.1, -3.2, 1.8, -0.3];
        let result = TruncatePadProjector.project(&embedding, 4);

        assert_eq!(result, vec![-1.5, 2.3, -0.7, 0.0]);
    }

    #[test]
    fn test_pad_preserves_sign() {
        let embedding = vec![-1.5, 2.3];
        let result = TruncatePadProjector.project(&embedding, 5);

        assert_eq!(result, vec![-1.5, 2.3, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_linear_projector_falls_back_to_truncate_pad() {
        let embedding = vec![1.0, 2.0, 3.0];
        let result = LinearProjector {}.project(&embedding, 2);

        assert_eq!(result, vec![1.0, 2.0]);
    }
}
