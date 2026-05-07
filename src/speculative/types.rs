use crate::transformer::{ForwardContext, MultiLayerKVCache};
use crate::types::Config;
use std::cmp::Ordering;

// ── Constraint Pruner: Neuro-Symbolic Intercept ──────────────────

/// Trait for pruning drafted tokens against deterministic constraints.
///
/// The Computable LoRA concept: before the target model verifies drafted
/// branches, a rules engine prunes invalid ones. This prevents the DDTree
/// from wasting budget on branches that can never be accepted.
///
/// Without pruner: DDTree explores ALL high-probability tokens.
/// With pruner:    DDTree explores only VALID high-probability tokens.
pub trait ConstraintPruner: Send + Sync {
    /// Check if `token_idx` at the given `depth` is valid, given the
    /// tokens placed at earlier depths in this path.
    ///
    /// `parent_token[i]` = token placed at depth `i` in the current path.
    /// At depth 0, `parent_tokens` is empty.
    ///
    /// Returns `false` to prune (reject) this branch.
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;
}

/// No-op pruner: allows all tokens (original DDTree behavior).
pub struct NoPruner;

impl ConstraintPruner for NoPruner {
    fn is_valid(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> bool {
        true
    }
}

// ── DDTree Node ────────────────────────────────────────────────

/// DDTree node for Best-First Search.
#[derive(Copy, Clone, PartialEq)]
pub struct TreeNode {
    pub score: f32,
    pub depth: usize,
    pub token_idx: usize,
    pub parent_path: u64,
}

impl Eq for TreeNode {}

impl PartialOrd for TreeNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TreeNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

// ── Draft Result ────────────────────────────────────────────────

/// Result of autoregressive drafting: marginals + sampled tokens.
pub struct DraftResult {
    pub marginals: Vec<Vec<f32>>,
    pub sampled_tokens: Vec<usize>,
}

// ── Pre-allocated Speculative Context ──────────────────────────

/// Pre-allocated buffers for zero-alloc speculative decoding.
///
/// Create once with `SpeculativeContext::new(config)`, reuse across calls.
/// Call `reset()` between decode steps to clear transient state.
///
/// All hot-path operations borrow from this struct instead of allocating:
/// - `dflash_predict_with` reuses `ctx`, `cache`, `marginals_flat`, `probs_buf`
/// - `build_dd_tree` reuses `TreeBuilder` heap/tree buffers
/// - `sample_residual_distribution_into` reuses `residual_buf`
/// - Leviathan rejection sampling reuses `p_distributions_flat`
pub struct SpeculativeContext {
    /// Pre-allocated forward pass buffers (embedding, attention, MLP, logits).
    pub ctx: ForwardContext,
    /// Pre-allocated KV cache for draft model.
    pub cache: MultiLayerKVCache,
    /// Flat marginals buffer: `[draft_lookahead * vocab_size]`.
    /// Each step's marginal occupies `[step * vocab_size..(step+1) * vocab_size]`.
    pub marginals_flat: Vec<f32>,
    /// Temp probs buffer: `[vocab_size]` for logits→softmax in-place.
    pub probs_buf: Vec<f32>,
    /// Pre-allocated sampled tokens: `[draft_lookahead]`.
    pub sampled_tokens: Vec<usize>,
    /// Pre-allocated accepted tokens: `[draft_lookahead + 1]`.
    pub accepted_buf: Vec<usize>,
    /// Pre-allocated path buffer: `[draft_lookahead + 1]`.
    pub path_buf: Vec<usize>,
    /// Residual distribution scratch: `[vocab_size]` for `sample_residual_distribution_into`.
    pub residual_buf: Vec<f32>,
    /// Flat p-distributions buffer for Leviathan: `[(draft_lookahead + 1) * vocab_size]`.
    pub p_distributions_flat: Vec<f32>,
    /// Number of steps populated in last operation (for slicing).
    pub steps_populated: usize,
}

impl SpeculativeContext {
    /// Allocate all buffers from config dimensions.
    pub fn new(config: &Config) -> Self {
        let vocab_size = config.vocab_size;
        let draft_lookahead = config.draft_lookahead;

        Self {
            ctx: ForwardContext::new(config),
            cache: MultiLayerKVCache::new(config),
            marginals_flat: vec![0.0f32; draft_lookahead * vocab_size],
            probs_buf: vec![0.0f32; vocab_size],
            sampled_tokens: vec![0usize; draft_lookahead],
            accepted_buf: vec![0usize; draft_lookahead + 1],
            path_buf: vec![0usize; draft_lookahead + 1],
            residual_buf: vec![0.0f32; vocab_size],
            p_distributions_flat: vec![0.0f32; (draft_lookahead + 1) * vocab_size],
            steps_populated: 0,
        }
    }

    /// Reset transient state between decode steps.
    /// Clears lengths to zero; buffers retain capacity for reuse.
    pub fn reset(&mut self) {
        self.cache.reset();
        self.steps_populated = 0;
    }

    /// Get marginal slice for a specific step.
    /// Returns empty slice if step is out of range.
    pub fn marginal_slice(&self, step: usize, vocab_size: usize) -> &[f32] {
        let start = step * vocab_size;
        let end = start + vocab_size;
        if end <= self.marginals_flat.len() && step < self.steps_populated {
            &self.marginals_flat[start..end]
        } else {
            &[]
        }
    }

    /// Get mutable marginal slice for a specific step.
    pub fn marginal_slice_mut(&mut self, step: usize, vocab_size: usize) -> &mut [f32] {
        let start = step * vocab_size;
        let end = start + vocab_size;
        if end <= self.marginals_flat.len() {
            &mut self.marginals_flat[start..end]
        } else {
            &mut []
        }
    }

    /// Get populated marginals as slice-of-slices (borrowed view).
    /// Returns a Vec of borrowed slices for compatibility with existing APIs.
    pub fn marginals_view(&self, vocab_size: usize) -> Vec<&[f32]> {
        (0..self.steps_populated)
            .map(|step| self.marginal_slice(step, vocab_size))
            .collect()
    }

    /// Get populated sampled tokens.
    pub fn sampled_tokens(&self) -> &[usize] {
        &self.sampled_tokens[..self.steps_populated]
    }

    /// Get p-distribution slice for Leviathan step.
    pub fn p_dist_slice(&self, step: usize, vocab_size: usize) -> &[f32] {
        let start = step * vocab_size;
        let end = start + vocab_size;
        if end <= self.p_distributions_flat.len() {
            &self.p_distributions_flat[start..end]
        } else {
            &[]
        }
    }

    /// Get mutable p-distribution slice for Leviathan step.
    pub fn p_dist_slice_mut(&mut self, step: usize, vocab_size: usize) -> &mut [f32] {
        let start = step * vocab_size;
        let end = start + vocab_size;
        if end <= self.p_distributions_flat.len() {
            &mut self.p_distributions_flat[start..end]
        } else {
            &mut []
        }
    }
}
