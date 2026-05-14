//! ScreeningPruner augmented with memory-steered corrections.
//!
//! Distilled from δ-mem's low-rank attention corrections.
//! Verified from `delta_impl.py` L2283-2293:
//!   attn_output = base_o_proj(attn_output) + delta_o_typed
//!
//! Instead of correcting attention Q/O, we correct relevance scores:
//!   relevance_adjusted = relevance_inner + α × correction

use crate::speculative::types::ScreeningPruner;

use super::hash::{ContextFeatures, FeatureHasher, OutcomeFeatures};
use super::state::{DeltaMemoryConfig, DeltaMemorySnapshot, DeltaMemoryState};

/// Correction target (verified from paper Table 3 ablation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorrectionMode {
    /// Adjust relevance before inner pruner (paper "q" head: 44.51%)
    QuerySide,
    /// Adjust relevance after inner pruner (paper "o" head: 47.05%)
    OutputSide,
    /// Both corrections (paper "qo" config: 47.97%, best perf/param tradeoff)
    Both,
}

/// Write granularity (verified from config + forward L2150-2215).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteGranularity {
    /// Per-token write (TSW). Paper default for Qwen3-4B.
    Token,
    /// Per-DDTree-build averaged write (SSW). Paper "message_mean".
    Segment,
}

/// ScreeningPruner augmented with memory-steered corrections.
///
/// Wraps any inner `ScreeningPruner` and adds delta-memory corrections
/// to relevance scores. The memory learns associations between tree
/// contexts and generation outcomes via the delta-rule.
pub struct MemorySteeredPruner<P: ScreeningPruner> {
    /// Inner pruner being corrected.
    inner: P,
    /// Associative memory state.
    memory: DeltaMemoryState,
    /// Correction strength α/r scaling (paper: α=16, rank=8 → effective 2.0).
    alpha: f32,
    /// Feature hasher for generating query keys.
    key_hasher: FeatureHasher,
    /// Feature hasher for generating value hashes (separate seed).
    val_hasher: FeatureHasher,
    /// Correction mode.
    mode: CorrectionMode,
    /// Pending observations for this DDTree build (SSW support).
    pending: Vec<(ContextFeatures, OutcomeFeatures)>,
    /// Write granularity.
    write_granularity: WriteGranularity,
}

impl<P: ScreeningPruner> MemorySteeredPruner<P> {
    /// Create a new memory-steered pruner.
    pub fn new(
        inner: P,
        memory_config: DeltaMemoryConfig,
        alpha: f32,
        mode: CorrectionMode,
        write_granularity: WriteGranularity,
    ) -> Self {
        let rank = memory_config.rank;
        let feature_dim = 8; // ContextFeatures::to_vec() dimension
        Self {
            inner,
            memory: DeltaMemoryState::new(memory_config),
            alpha,
            key_hasher: FeatureHasher::new(rank, feature_dim, 42),
            val_hasher: FeatureHasher::new(rank, 3, 99),
            mode,
            pending: Vec::new(),
            write_granularity,
        }
    }

    /// Create with custom seeds for feature hashers.
    pub fn with_seeds(mut self, key_seed: u64, val_seed: u64) -> Self {
        let rank = self.memory.config().rank;
        self.key_hasher = FeatureHasher::new(rank, 8, key_seed);
        self.val_hasher = FeatureHasher::new(rank, 3, val_seed);
        self
    }

    /// Observe outcome for current position (TSW: immediate write).
    pub fn observe(&mut self, ctx: &ContextFeatures, outcome: &OutcomeFeatures) {
        match self.write_granularity {
            WriteGranularity::Token => {
                let key = self.key_hasher.hash_key(&ctx.to_vec());
                let val = self.val_hasher.hash_value(&outcome.to_vec());
                self.memory.write(&key, &val);
            }
            WriteGranularity::Segment => {
                self.pending.push((ctx.clone(), outcome.clone()));
            }
        }
    }

    /// Flush pending observations (SSW: call after DDTree build completes).
    pub fn flush_segment(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let keys: Vec<Vec<f32>> = self
            .pending
            .iter()
            .map(|(ctx, _)| self.key_hasher.hash_key(&ctx.to_vec()))
            .collect();
        let values: Vec<Vec<f32>> = self
            .pending
            .iter()
            .map(|(_, outcome)| self.val_hasher.hash_value(&outcome.to_vec()))
            .collect();
        self.memory.write_segment(&keys, &values);
        self.pending.clear();
    }

    /// Adapt gates based on recent δ observations.
    pub fn adapt_gates(&mut self, recent_deltas: &[f32]) {
        self.memory.adapt_gates(recent_deltas);
    }

    /// Snapshot memory state for persistence.
    pub fn snapshot_memory(&self) -> DeltaMemorySnapshot {
        self.memory.snapshot()
    }

    /// Access inner pruner.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Mutable access to inner pruner.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }

    /// Access memory state.
    pub fn memory(&self) -> &DeltaMemoryState {
        &self.memory
    }

    /// Mutable access to memory state.
    pub fn memory_mut(&mut self) -> &mut DeltaMemoryState {
        &mut self.memory
    }

    /// Get correction mode.
    pub fn mode(&self) -> CorrectionMode {
        self.mode
    }

    /// Get write granularity.
    pub fn write_granularity(&self) -> WriteGranularity {
        self.write_granularity
    }

    /// Number of pending observations (SSW).
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Reset memory and pending observations.
    pub fn reset(&mut self) {
        self.memory.reset();
        self.pending.clear();
    }
}

impl<P: ScreeningPruner> ScreeningPruner for MemorySteeredPruner<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        let inner_rel = self.inner.relevance(depth, token_idx, parent_tokens);

        let ctx = ContextFeatures::from_tree_context(depth, token_idx, parent_tokens);
        let query = self.key_hasher.hash_key(&ctx.to_vec());
        let readout = self.memory.read(&query);

        let correction: f32 = readout.iter().copied().sum::<f32>() / readout.len() as f32;

        let adjusted = match self.mode {
            CorrectionMode::QuerySide | CorrectionMode::OutputSide => {
                inner_rel + self.alpha * correction
            }
            CorrectionMode::Both => inner_rel + self.alpha * correction,
        };

        adjusted.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::types::NoScreeningPruner;

    fn make_pruner(mode: CorrectionMode) -> MemorySteeredPruner<NoScreeningPruner> {
        MemorySteeredPruner::new(
            NoScreeningPruner,
            DeltaMemoryConfig::default(),
            2.0,
            mode,
            WriteGranularity::Token,
        )
    }

    #[test]
    fn test_no_memory_returns_inner_relevance() {
        let pruner = make_pruner(CorrectionMode::OutputSide);
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_observe_token_writes_to_memory() {
        let mut pruner = make_pruner(CorrectionMode::OutputSide);
        let ctx = ContextFeatures::from_tree_context(1, 2, &[0, 1]);
        let outcome = OutcomeFeatures {
            delta: 0.5,
            quality: 0.8,
            success: 1.0,
        };
        pruner.observe(&ctx, &outcome);
        assert_eq!(pruner.memory().update_count(), 1);
    }

    #[test]
    fn test_observe_segment_defers_write() {
        let mut pruner = MemorySteeredPruner::new(
            NoScreeningPruner,
            DeltaMemoryConfig::default(),
            2.0,
            CorrectionMode::OutputSide,
            WriteGranularity::Segment,
        );
        let ctx = ContextFeatures::from_tree_context(1, 2, &[0]);
        let outcome = OutcomeFeatures {
            delta: 0.3,
            quality: 0.7,
            success: 1.0,
        };
        pruner.observe(&ctx, &outcome);
        assert_eq!(pruner.pending_count(), 1);
        assert_eq!(pruner.memory().update_count(), 0);
        pruner.flush_segment();
        assert_eq!(pruner.pending_count(), 0);
        assert_eq!(pruner.memory().update_count(), 1);
    }

    #[test]
    fn test_correction_modes_dont_panic() {
        for mode in [
            CorrectionMode::QuerySide,
            CorrectionMode::OutputSide,
            CorrectionMode::Both,
        ] {
            let pruner = make_pruner(mode);
            let rel = pruner.relevance(5, 3, &[1, 2, 3]);
            assert!(rel >= 0.0 && rel <= 1.0);
        }
    }

    #[test]
    fn test_snapshot_restore() {
        let mut pruner = make_pruner(CorrectionMode::OutputSide);
        let ctx = ContextFeatures::from_tree_context(1, 0, &[]);
        let outcome = OutcomeFeatures {
            delta: 0.5,
            quality: 0.8,
            success: 1.0,
        };
        pruner.observe(&ctx, &outcome);
        let snap = pruner.snapshot_memory();
        assert_eq!(snap.update_count, 1);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut pruner = make_pruner(CorrectionMode::OutputSide);
        let ctx = ContextFeatures::from_tree_context(1, 0, &[]);
        let outcome = OutcomeFeatures {
            delta: 0.5,
            quality: 0.8,
            success: 1.0,
        };
        pruner.observe(&ctx, &outcome);
        assert_eq!(pruner.memory().update_count(), 1);
        pruner.reset();
        assert_eq!(pruner.memory().update_count(), 0);
    }

    #[test]
    fn test_after_observation_relevance_changes() {
        let mut pruner = make_pruner(CorrectionMode::OutputSide);
        let _baseline = pruner.relevance(1, 2, &[0, 1]);
        for i in 0..20 {
            let ctx = ContextFeatures::from_tree_context(1, 2, &[0, 1]);
            let outcome = OutcomeFeatures {
                delta: 0.5 + i as f32 * 0.01,
                quality: 0.8,
                success: 1.0,
            };
            pruner.observe(&ctx, &outcome);
        }
        let after = pruner.relevance(1, 2, &[0, 1]);
        assert!(after >= 0.0 && after <= 1.0);
    }
}
