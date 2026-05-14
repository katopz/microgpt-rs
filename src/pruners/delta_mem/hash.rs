//! Hashes context features into a compact r-dimensional vector.
//!
//! Replaces the paper's learned projections (Wmq, Wmk, Wmv).
//! Verified from `delta_impl.py` L805-814 (_normalize_memory_projection):
//!   key = L2_norm(tanh(W_mk · x))   (unit sphere, prevents explosion)
//!   val = W_mv · x                  (no normalization on values)
//!
//! We replace learned W with random LSH projection + same normalization.

/// Feature hasher using random projection.
///
/// Same normalization as paper: tanh → L2 normalize for keys/queries,
/// raw projection for values.
pub struct FeatureHasher {
    /// Memory rank.
    rank: usize,
    /// Random projection matrix [rank × feature_dim].
    projection: Vec<f32>,
    /// Seed for deterministic hashing.
    seed: u64,
}

impl FeatureHasher {
    /// Create a new feature hasher with random projection.
    ///
    /// Uses Kaiming-like initialization scaled by sqrt(2/rank).
    pub fn new(rank: usize, feature_dim: usize, seed: u64) -> Self {
        let mut projection = Vec::with_capacity(rank * feature_dim);
        let mut rng = fastrand::Rng::with_seed(seed);
        let scale = (2.0 / rank as f32).sqrt();

        for _ in 0..(rank * feature_dim) {
            projection.push(rng.f32() * 2.0 * scale - scale);
        }

        Self {
            rank,
            projection,
            seed,
        }
    }

    /// Hash to L2-normalized key/query vector.
    /// `L2_norm(tanh(projection · features))` — same as paper Eq 4.
    pub fn hash_key(&self, features: &[f32]) -> Vec<f32> {
        let projected = self.project(features);
        // tanh activation (same as paper)
        let tanhed: Vec<f32> = projected.iter().map(|&x| x.tanh()).collect();
        // L2 normalize (prevents state explosion — verified from source)
        let norm: f32 = tanhed.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        tanhed.iter().map(|&x| x / norm).collect()
    }

    /// Hash to raw value vector (no normalization, same as paper).
    /// `projection · features`
    pub fn hash_value(&self, features: &[f32]) -> Vec<f32> {
        self.project(features)
    }

    /// Matrix-vector multiply: projection · features
    fn project(&self, features: &[f32]) -> Vec<f32> {
        let dim = features.len();
        let mut result = vec![0.0; self.rank];
        for i in 0..self.rank {
            let mut sum = 0.0f32;
            for j in 0..dim {
                if j * self.rank + i < self.projection.len() {
                    // Column-major access for projection[i, j]
                    sum += self.projection[j * self.rank + i] * features[j];
                }
            }
            result[i] = sum;
        }
        result
    }

    /// Get the rank dimension.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the seed used for initialization.
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl Clone for FeatureHasher {
    fn clone(&self) -> Self {
        Self {
            rank: self.rank,
            projection: self.projection.clone(),
            seed: self.seed,
        }
    }
}

/// Extract features from DDTree context for memory hashing.
#[derive(Clone, Debug)]
pub struct ContextFeatures {
    /// Domain hash (from PromptRouter domain string)
    pub domain: u64,
    /// Current depth in DDTree (normalized to [0, 1])
    pub depth_normalized: f32,
    /// Token entropy at current position (from marginals)
    pub token_entropy: f32,
    /// Parent path length (normalized)
    pub path_length_normalized: f32,
    /// Screening relevance score at parent
    pub parent_relevance: f32,
}

impl ContextFeatures {
    /// Convert to feature vector for hashing.
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            (self.domain & 0xFF) as f32 / 255.0, // Low byte of domain hash
            ((self.domain >> 8) & 0xFF) as f32 / 255.0,
            ((self.domain >> 16) & 0xFF) as f32 / 255.0,
            ((self.domain >> 24) & 0xFF) as f32 / 255.0,
            self.depth_normalized,
            self.token_entropy,
            self.path_length_normalized,
            self.parent_relevance,
        ]
    }

    /// Extract from DDTree context during build.
    pub fn from_tree_context(depth: usize, _token_idx: usize, parent_tokens: &[usize]) -> Self {
        let max_depth = 32.0f32; // Typical DDTree max depth
        let max_path = 256.0f32; // Typical max path length

        Self {
            domain: 0, // Set by caller based on PromptRouter
            depth_normalized: (depth as f32 / max_depth).min(1.0),
            token_entropy: 0.0, // Would be filled from model marginals
            path_length_normalized: (parent_tokens.len() as f32 / max_path).min(1.0),
            parent_relevance: 1.0, // Default: no parent info
        }
    }
}

/// Extract features from generation outcome for memory values.
#[derive(Clone, Debug)]
pub struct OutcomeFeatures {
    /// Hint-δ value (from DeltaBanditPruner)
    pub delta: f32,
    /// Solution quality (path length / budget ratio)
    pub quality: f32,
    /// Whether DDTree found a valid solution (0.0 or 1.0)
    pub success: f32,
}

impl OutcomeFeatures {
    /// Convert to feature vector for memory value.
    pub fn to_vec(&self) -> Vec<f32> {
        vec![self.delta, self.quality, self.success]
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_key_is_normalized() {
        let hasher = FeatureHasher::new(8, 5, 42);
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let key = hasher.hash_key(&features);

        // L2 norm should be ~1.0
        let norm: f32 = key.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Key should be L2-normalized, got norm={norm}"
        );
    }

    #[test]
    fn test_hash_value_is_raw() {
        let hasher = FeatureHasher::new(8, 5, 42);
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let val = hasher.hash_value(&features);

        // Values should NOT be normalized (can have any magnitude)
        assert_eq!(val.len(), 8);
        // Values should generally be non-zero for non-zero input
        assert!(val.iter().any(|&x| x.abs() > 0.0));
    }

    #[test]
    fn test_deterministic_same_seed() {
        let h1 = FeatureHasher::new(8, 5, 42);
        let h2 = FeatureHasher::new(8, 5, 42);
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let k1 = h1.hash_key(&features);
        let k2 = h2.hash_key(&features);

        for (a, b) in k1.iter().zip(k2.iter()) {
            assert!((a - b).abs() < 1e-6, "Same seed should produce same hash");
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let h1 = FeatureHasher::new(8, 5, 42);
        let h2 = FeatureHasher::new(8, 5, 99);
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let k1 = h1.hash_key(&features);
        let k2 = h2.hash_key(&features);

        // Very unlikely all dimensions are identical
        let identical = k1.iter().zip(k2.iter()).all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(
            !identical,
            "Different seeds should produce different hashes"
        );
    }

    #[test]
    fn test_context_features_to_vec() {
        let ctx = ContextFeatures {
            domain: 0x01020304,
            depth_normalized: 0.5,
            token_entropy: 0.3,
            path_length_normalized: 0.2,
            parent_relevance: 0.8,
        };
        let vec = ctx.to_vec();
        assert_eq!(vec.len(), 8);
        assert!((vec[0] - 4.0 / 255.0).abs() < 1e-6); // low byte
        assert!((vec[4] - 0.5).abs() < 1e-6); // depth_normalized
    }

    #[test]
    fn test_context_features_from_tree() {
        let ctx = ContextFeatures::from_tree_context(16, 3, &[1, 2, 3]);
        assert!((ctx.depth_normalized - 0.5).abs() < 0.1);
        assert!((ctx.path_length_normalized - 3.0 / 256.0).abs() < 0.01);
    }

    #[test]
    fn test_outcome_features_to_vec() {
        let outcome = OutcomeFeatures {
            delta: 0.5,
            quality: 0.8,
            success: 1.0,
        };
        let vec = outcome.to_vec();
        assert_eq!(vec.len(), 3);
        assert!((vec[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_key_tanh_bounds() {
        let hasher = FeatureHasher::new(8, 5, 42);
        // Very large features
        let features = vec![100.0, -100.0, 100.0, -100.0, 100.0];
        let key = hasher.hash_key(&features);

        // After tanh, all values should be in [-1, 1]
        for &k in &key {
            assert!(k.abs() <= 1.0 + 1e-6, "tanh output should be in [-1, 1]");
        }
    }
}
