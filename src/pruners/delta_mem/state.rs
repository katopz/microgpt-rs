//! Compact associative memory updated by delta-rule learning.
//!
//! Distilled from δ-mem (arXiv 2605.12357), verified against source:
//!   `delta_impl.py` L1917-1929 (_memory_affine_scan_torch)
//!
//! Core formula (per-row, coupled gates):
//!   read_t  = S · q_t
//!   pred_t  = S · k_t
//!   S'[i,:] = (1-β[i]) · S[i,:] - β[i] · pred_t[i] · k_t + β[i] · v_t[i] · k_t
//!
//! With normalize_qk=True (default), keys and queries are L2-normalized
//! after tanh, keeping them on the unit sphere to prevent state explosion.

use serde::{Deserialize, Serialize};

/// Configuration for delta memory state.
#[derive(Clone, Debug)]
pub struct DeltaMemoryConfig {
    /// Memory rank r (paper default: 8). State is r×r = 64 floats.
    pub rank: usize,
    /// Initial β (write gate). sigmoid(-1.5) ≈ 0.182 (paper default).
    pub beta_init: f32,
    /// Whether to couple λ = 1 - β (paper default: true).
    pub couple_gates: bool,
}

impl Default for DeltaMemoryConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            beta_init: 0.182, // sigmoid(-1.5)
            couple_gates: true,
        }
    }
}

/// Compact r×r associative memory updated by delta-rule learning.
///
/// Memory layout: `state[row * rank + col]` = S[row, col].
/// Total size: rank² floats (256 bytes at rank=8).
pub struct DeltaMemoryState {
    /// Associative memory matrix [rank × rank], row-major.
    state: Vec<f32>,
    /// Config.
    config: DeltaMemoryConfig,
    /// Per-dimension write gate β [rank].
    beta: Vec<f32>,
    /// Number of updates (for adaptive gate scheduling).
    update_count: usize,
    /// Recent prediction errors for gate adaptation [rank × window].
    error_history: Vec<f32>,
    /// Error history window size.
    error_window: usize,
}

impl DeltaMemoryState {
    pub fn new(config: DeltaMemoryConfig) -> Self {
        let rank = config.rank;
        let beta_init = config.beta_init;
        Self {
            state: vec![0.0; rank * rank],
            config,
            beta: vec![beta_init; rank],
            update_count: 0,
            error_history: Vec::new(),
            error_window: 64,
        }
    }

    /// Read: r_t = S_{t-1} · q_t
    ///
    /// O(r²) — constant regardless of history length.
    /// Verified: `delta_impl.py` L1921 `read_t = torch.einsum("bij,bj->bi", current_state, q_t)`
    pub fn read(&self, query: &[f32]) -> Vec<f32> {
        let rank = self.config.rank;
        assert_eq!(query.len(), rank, "query dimension must match rank");
        let mut result = vec![0.0; rank];
        for i in 0..rank {
            let mut sum = 0.0f32;
            for j in 0..rank {
                sum += self.state[i * rank + j] * query[j];
            }
            result[i] = sum;
        }
        result
    }

    /// Write: delta-rule update (coupled gates).
    ///
    /// Per-row update, verified from `_memory_affine_scan_torch` L1923-1929:
    ///   pred_t = S · k_t              (prediction)
    ///   S'[i,:] = (1-β[i])·S[i,:] - β[i]·pred_t[i]·k + β[i]·v[i]·k
    ///
    /// Key/value MUST be L2-normalized before calling (see FeatureHasher).
    pub fn write(&mut self, key: &[f32], value: &[f32]) {
        let rank = self.config.rank;
        assert_eq!(key.len(), rank, "key dimension must match rank");
        assert_eq!(value.len(), rank, "value dimension must match rank");

        // pred_t = S · k_t (prediction: what current state says about this key)
        let predictions = self.read(key);

        // Per-row delta-rule update
        for i in 0..rank {
            let beta_i = self.beta[i];
            let lambda_i = if self.config.couple_gates {
                1.0 - beta_i
            } else {
                1.0
            };
            let pred_i = predictions[i];
            let val_i = value[i];

            for j in 0..rank {
                let old = self.state[i * rank + j];
                // S'[i,j] = λ·S[i,j] - β·pred_i·k_j + β·v_i·k_j
                self.state[i * rank + j] =
                    lambda_i * old - beta_i * pred_i * key[j] + beta_i * val_i * key[j];
            }

            // Track prediction error for gate adaptation
            let error = (val_i - pred_i).abs();
            self.push_error(error);
        }

        self.update_count += 1;
    }

    /// Segment-State Write: average features over a segment, write once.
    ///
    /// Verified from `_memory_affine_scan_torch` with `message_mean` granularity:
    ///   average all k_t and v_t over the segment, then single write.
    pub fn write_segment(&mut self, keys: &[Vec<f32>], values: &[Vec<f32>]) {
        if keys.is_empty() {
            return;
        }
        let rank = self.config.rank;

        let mut avg_key = vec![0.0f32; rank];
        let mut avg_val = vec![0.0f32; rank];
        let n = keys.len() as f32;

        for k in keys {
            for (j, kj) in k.iter().enumerate() {
                avg_key[j] += kj / n;
            }
        }
        for v in values {
            for (j, vj) in v.iter().enumerate() {
                avg_val[j] += vj / n;
            }
        }

        self.write(&avg_key, &avg_val);
    }

    /// Adaptive gate: adjust β based on recent prediction error variance.
    ///
    /// Paper uses learned sigmoid(W_β · x + bias). We use δ variance:
    ///   high variance → larger β (write more aggressively)
    ///   low variance  → smaller β (conserve stable state)
    pub fn adapt_gates(&mut self, recent_errors: &[f32]) {
        if recent_errors.is_empty() {
            return;
        }
        let mean: f32 = recent_errors.iter().sum::<f32>() / recent_errors.len() as f32;
        let variance: f32 = recent_errors
            .iter()
            .map(|e| (e - mean).powi(2))
            .sum::<f32>()
            / recent_errors.len() as f32;

        // Map variance to β adjustment: high variance → increase β
        let beta_adjustment = (variance * 0.1).min(0.05); // Cap adjustment
        for beta in self.beta.iter_mut() {
            *beta = (*beta + beta_adjustment).clamp(0.01, 0.95);
        }
    }

    /// Reset state to zeros.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.beta.fill(self.config.beta_init);
        self.update_count = 0;
        self.error_history.clear();
    }

    /// Snapshot state for serialization.
    pub fn snapshot(&self) -> DeltaMemorySnapshot {
        DeltaMemorySnapshot {
            state: self.state.clone(),
            rank: self.config.rank,
            beta: self.beta.clone(),
            update_count: self.update_count,
        }
    }

    /// Restore from snapshot.
    pub fn restore(&mut self, snapshot: &DeltaMemorySnapshot) {
        assert_eq!(snapshot.rank, self.config.rank, "rank mismatch on restore");
        self.state.copy_from_slice(&snapshot.state);
        self.beta.copy_from_slice(&snapshot.beta);
        self.update_count = snapshot.update_count;
    }

    /// Push error to history ring buffer
    fn push_error(&mut self, error: f32) {
        if self.error_history.len() >= self.error_window * self.config.rank {
            // Drain oldest batch
            self.error_history.drain(..self.config.rank);
        }
        self.error_history.push(error);
    }

    /// Get current error history (for external gate adaptation)
    pub fn error_history(&self) -> &[f32] {
        &self.error_history
    }

    /// Get current beta values
    pub fn beta(&self) -> &[f32] {
        &self.beta
    }

    /// Get config reference
    pub fn config(&self) -> &DeltaMemoryConfig {
        &self.config
    }

    /// Number of updates performed
    pub fn update_count(&self) -> usize {
        self.update_count
    }

    /// State norm (for diagnostics / explosion check)
    pub fn state_norm(&self) -> f32 {
        self.state.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

/// Serializable snapshot of memory state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeltaMemorySnapshot {
    pub state: Vec<f32>,
    pub rank: usize,
    pub beta: Vec<f32>,
    pub update_count: usize,
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_state_is_zero() {
        let state = DeltaMemoryState::new(DeltaMemoryConfig::default());
        assert_eq!(state.state.len(), 64); // 8×8
        assert!(state.state.iter().all(|&x| x == 0.0));
        assert!((state.state_norm()).abs() < 1e-6);
    }

    #[test]
    fn test_read_zero_state_returns_zero() {
        let state = DeltaMemoryState::new(DeltaMemoryConfig::default());
        let query = vec![1.0; 8];
        let result = state.read(&query);
        assert!(result.iter().all(|&x| x.abs() < 1e-6));
    }

    #[test]
    fn test_write_then_read() {
        let mut state = DeltaMemoryState::new(DeltaMemoryConfig::default());
        let key = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let value = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        state.write(&key, &value);

        // Reading with the same key should return something related to value
        let readout = state.read(&key);
        // After 1 write: β·v_i·k_j is the dominant term for the first write
        assert!(
            readout[1].abs() > 0.0,
            "Should have non-zero readout after write"
        );
    }

    #[test]
    fn test_write_segment() {
        let mut state = DeltaMemoryState::new(DeltaMemoryConfig::default());
        let keys = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let values = vec![
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];

        state.write_segment(&keys, &values);

        // Should have written with averaged key and value
        assert!(state.update_count() == 1);
    }

    #[test]
    fn test_snapshot_restore_roundtrip() {
        let mut state = DeltaMemoryState::new(DeltaMemoryConfig::default());
        let key = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let value = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        state.write(&key, &value);

        let snap = state.snapshot();
        let mut state2 = DeltaMemoryState::new(DeltaMemoryConfig::default());
        state2.restore(&snap);

        let r1 = state.read(&key);
        let r2 = state2.read(&key);
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let mut state = DeltaMemoryState::new(DeltaMemoryConfig::default());
        let key = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let value = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        state.write(&key, &value);
        assert!(state.update_count() > 0);

        state.reset();
        assert!(state.state.iter().all(|&x| x == 0.0));
        assert_eq!(state.update_count(), 0);
    }

    #[test]
    fn test_coupled_gates_beta_lambda() {
        let config = DeltaMemoryConfig {
            rank: 4,
            beta_init: 0.2,
            couple_gates: true,
        };
        let mut state = DeltaMemoryState::new(config);
        let key = vec![1.0, 0.0, 0.0, 0.0];
        let value = vec![0.0, 1.0, 0.0, 0.0];

        state.write(&key, &value);

        // With coupled gates (λ = 1 - β = 0.8):
        // S'[0,0] = 0.8 * 0 - 0.2 * 0 * 1 + 0.2 * 0 * 1 = 0
        // S'[0,1] = 0.8 * 0 - 0.2 * 0 * 0 + 0.2 * 0 * 0 = 0
        // S'[1,0] = 0.8 * 0 - 0.2 * pred[1] * 1 + 0.2 * 1 * 1
        //   where pred[1] = S[1,:] · k = 0 (initial state is zero)
        //   = 0 + 0 + 0.2 = 0.2
        assert!((state.state[1 * 4 + 0] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_adapt_gates_increases_beta_with_high_variance() {
        let mut state = DeltaMemoryState::new(DeltaMemoryConfig {
            rank: 4,
            beta_init: 0.1,
            couple_gates: true,
        });
        let initial_beta = state.beta()[0];

        // High variance errors
        state.adapt_gates(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

        let new_beta = state.beta()[0];
        assert!(new_beta > initial_beta, "High variance should increase β");
    }

    #[test]
    fn test_state_norm_bounded_after_many_writes() {
        let mut state = DeltaMemoryState::new(DeltaMemoryConfig::default());

        // Write normalized keys/values repeatedly
        for i in 0..200 {
            let phase = i as f32 * 0.1;
            let key: Vec<f32> = (0..8).map(|j| (phase + j as f32 * 0.5).sin()).collect();
            let norm: f32 = key.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            let key_norm: Vec<f32> = key.iter().map(|x| x / norm).collect();
            let value: Vec<f32> = (0..8).map(|j| (phase + j as f32 * 0.3).cos()).collect();

            state.write(&key_norm, &value);
        }

        // State norm should be bounded (no explosion with normalized keys)
        let norm = state.state_norm();
        assert!(norm < 100.0, "State norm should be bounded, got {norm}");
    }

    #[test]
    fn test_custom_rank() {
        let config = DeltaMemoryConfig {
            rank: 4,
            ..Default::default()
        };
        let state = DeltaMemoryState::new(config);
        assert_eq!(state.state.len(), 16); // 4×4
    }

    #[test]
    fn test_snapshot_serialization() {
        let mut state = DeltaMemoryState::new(DeltaMemoryConfig {
            rank: 4,
            ..Default::default()
        });
        let key = vec![1.0, 0.0, 0.0, 0.0];
        let value = vec![0.0, 1.0, 0.0, 0.0];
        state.write(&key, &value);

        let snap = state.snapshot();
        let json = serde_json::to_string(&snap).unwrap();
        let restored: DeltaMemorySnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.rank, 4);
        assert_eq!(restored.state.len(), 16);
        assert_eq!(restored.update_count, 1);
    }
}
