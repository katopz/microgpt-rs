use crate::transformer::{
    ForwardContext, MultiLayerKVCache, PagedKVCache, TransformerWeights, forward_paged,
};
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
    pub parent_path: u128,
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

// ── DDTree Branch Cache ────────────────────────────────────────

/// Paged KV cache wrapper for DDTree branch exploration.
/// Forks share prefix pages, only new pages allocate after fork point.
///
/// This enables best-first search in DDTree where multiple token branches
/// are explored from a shared prefix — copy-on-write avoids duplicating
/// the entire KV cache for each branch.
pub struct DDTreeBranchCache {
    paged: PagedKVCache,
    branch_count: usize,
    max_branches: usize,
}

impl DDTreeBranchCache {
    /// Create a new branch cache.
    /// `max_branches` determines how many concurrent sequences the paged cache supports.
    pub fn new(config: &Config, max_branches: usize) -> Self {
        Self {
            paged: PagedKVCache::new(config, max_branches),
            branch_count: 1, // sequence 0 = trunk
            max_branches,
        }
    }

    /// Fork from an existing sequence at the given position.
    /// Returns the new sequence index.
    /// Shared prefix pages are NOT copied — copy-on-write semantics.
    pub fn fork_branch(&mut self, from: usize, at_pos: usize) -> usize {
        if self.branch_count >= self.max_branches {
            return from; // budget exhausted, reuse source
        }
        let new_seq = self.paged.fork(from, at_pos);
        self.branch_count += 1;
        new_seq
    }

    /// Forward pass on a specific branch sequence.
    /// Returns logits slice via `forward_paged`.
    pub fn forward_branch<'a>(
        &mut self,
        ctx: &'a mut ForwardContext,
        weights: &TransformerWeights,
        seq_idx: usize,
        token: usize,
        pos: usize,
        config: &Config,
    ) -> &'a mut [f32] {
        forward_paged(ctx, weights, &mut self.paged, seq_idx, token, pos, config)
    }

    /// Rollback a branch to a given position, freeing exclusive pages.
    ///
    /// Keeps pages covering positions `[0..at_pos)` and truncates the rest.
    /// Pages exclusively owned by this branch (not shared with other branches)
    /// are returned to the free list. Shared prefix pages are preserved.
    pub fn rollback_branch(&mut self, seq_idx: usize, at_pos: usize) {
        self.paged.rollback(seq_idx, at_pos);
    }

    /// Fully discard a branch, freeing all its exclusive pages.
    ///
    /// Rolls back to position 0 and decrements `branch_count` if the branch
    /// is not the trunk (seq 0). The trunk cannot be discarded — use `reset()`
    /// to clear everything.
    pub fn discard_branch(&mut self, seq_idx: usize) {
        self.paged.rollback(seq_idx, 0);
        if seq_idx > 0 && self.branch_count > 1 {
            self.branch_count -= 1;
        }
    }

    /// Reset all branches, freeing pages back to pool.
    pub fn reset(&mut self) {
        self.paged.reset();
        self.branch_count = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TransformerWeights;
    use crate::types::Rng;

    #[test]
    fn test_branch_cache_fork_branch() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 8);

        // Run trunk forward at pos 0..3
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        for pos in 0..4 {
            let _ = cache.forward_branch(&mut ctx, &weights, 0, pos, pos, &config);
        }

        // Fork at pos 2 — should return a new sequence index (not source)
        let branch = cache.fork_branch(0, 2);
        assert_ne!(
            branch, 0,
            "first fork should return different seq_idx than source"
        );

        // Fork again — should return another new unique sequence index
        let branch2 = cache.fork_branch(0, 2);
        assert_ne!(
            branch2, 0,
            "second fork should return different seq_idx than source"
        );
        assert_ne!(branch2, branch, "each fork should return unique seq_idx");
    }

    #[test]
    fn test_branch_cache_fork_branch_budget_exhausted() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 2); // max 2 branches

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        // Trunk at pos 0
        let _ = cache.forward_branch(&mut ctx, &weights, 0, 0, 0, &config);

        // First fork succeeds — returns a new seq_idx (not source)
        let b1 = cache.fork_branch(0, 0);
        assert_ne!(b1, 0, "fork should return new seq_idx");

        // Budget exhausted (branch_count == max_branches), returns source
        let b2 = cache.fork_branch(0, 0);
        assert_eq!(b2, 0, "budget exhausted should return source seq_idx");
    }

    #[test]
    fn test_branch_cache_forward_branch_logits() {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut cache = DDTreeBranchCache::new(&config, 4);
        let mut ctx = ForwardContext::new(&config);

        let logits = cache.forward_branch(&mut ctx, &weights, 0, 0, 0, &config);
        assert_eq!(logits.len(), config.vocab_size);

        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logit {i} not finite: {l}");
        }
    }

    #[test]
    fn test_branch_cache_fork_shared_prefix_forward() {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut cache = DDTreeBranchCache::new(&config, 4);
        let mut ctx = ForwardContext::new(&config);

        // Trunk: pos 0, 1, 2
        for pos in 0..3 {
            let _ = cache.forward_branch(&mut ctx, &weights, 0, pos, pos, &config);
        }

        // Fork at pos 2 — returns a new seq_idx (not source)
        let branch = cache.fork_branch(0, 2);
        assert_ne!(branch, 0, "fork should return new seq_idx");

        // Continue branch from pos 2 with different token
        let logits = cache.forward_branch(&mut ctx, &weights, branch, 5, 2, &config);
        assert_eq!(logits.len(), config.vocab_size);

        // All logits should be finite
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "branch logit {i} not finite: {l}");
        }
    }

    #[test]
    fn test_branch_cache_reset() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 8);

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        let _ = cache.forward_branch(&mut ctx, &weights, 0, 0, 0, &config);
        let _ = cache.fork_branch(0, 0);

        assert_eq!(cache.branch_count, 2);

        cache.reset();

        assert_eq!(
            cache.branch_count, 1,
            "reset should restore branch_count to 1"
        );
    }

    #[test]
    fn test_branch_cache_rollback_branch_allows_forward_after() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 4);

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        // Trunk: pos 0, 1, 2, 3
        for pos in 0..4 {
            let _ = cache.forward_branch(&mut ctx, &weights, 0, pos, pos, &config);
        }

        // Rollback trunk to pos 2
        cache.rollback_branch(0, 2);

        // Forward should still work after rollback — pages for pos 0..2 are intact
        let logits = cache.forward_branch(&mut ctx, &weights, 0, 5, 2, &config);
        assert_eq!(logits.len(), config.vocab_size);
        for (i, &l) in logits.iter().enumerate() {
            assert!(
                l.is_finite(),
                "logit {i} after rollback should be finite: {l}"
            );
        }
    }

    #[test]
    fn test_branch_cache_discard_branch_decrements_count() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 8);

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        // Trunk at pos 0
        let _ = cache.forward_branch(&mut ctx, &weights, 0, 0, 0, &config);

        // Fork a branch
        let branch = cache.fork_branch(0, 0);
        assert_ne!(branch, 0, "fork should return new seq_idx");
        assert_eq!(cache.branch_count, 2);

        // Forward on branch
        let _ = cache.forward_branch(&mut ctx, &weights, branch, 5, 1, &config);

        // Discard branch
        cache.discard_branch(branch);
        assert_eq!(
            cache.branch_count, 1,
            "discard should decrement branch_count"
        );
    }

    #[test]
    fn test_branch_cache_discard_trunk_does_not_decrement() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 4);

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        let _ = cache.forward_branch(&mut ctx, &weights, 0, 0, 0, &config);

        // Discard trunk (seq 0) should not decrement branch_count
        cache.discard_branch(0);
        assert_eq!(
            cache.branch_count, 1,
            "discarding trunk should not decrement branch_count"
        );
    }

    #[test]
    fn test_branch_cache_rollback_shared_pages_preserved() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 4);

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        // Trunk: pos 0, 1, 2
        for pos in 0..3 {
            let _ = cache.forward_branch(&mut ctx, &weights, 0, pos, pos, &config);
        }

        // Capture trunk state at pos 2 before branching
        let _trunk_logits = cache
            .forward_branch(&mut ctx, &weights, 0, 0, 3, &config)
            .to_vec();

        // Fork at pos 1
        let branch = cache.fork_branch(0, 1);
        assert_ne!(branch, 0, "fork should return new seq_idx");

        // Forward on branch to allocate exclusive pages
        let _ = cache.forward_branch(&mut ctx, &weights, branch, 5, 1, &config);
        let _ = cache.forward_branch(&mut ctx, &weights, branch, 7, 2, &config);

        // Rollback branch to pos 1 — shared prefix pages should be preserved
        cache.rollback_branch(branch, 1);

        // Trunk should still work: shared pages (pos 0) were not freed
        let trunk_logits_after = cache
            .forward_branch(&mut ctx, &weights, 0, 1, 3, &config)
            .to_vec();
        for (i, &l) in trunk_logits_after.iter().enumerate() {
            assert!(
                l.is_finite(),
                "trunk logit {i} should be finite after branch rollback: {l}"
            );
        }

        // Forking again from trunk should succeed — shared pages are intact
        let branch2 = cache.fork_branch(0, 1);
        assert_ne!(branch2, 0, "fork after rollback should return new seq_idx");
        let logits2 = cache.forward_branch(&mut ctx, &weights, branch2, 3, 1, &config);
        assert_eq!(logits2.len(), config.vocab_size);
        for (i, &l) in logits2.iter().enumerate() {
            assert!(l.is_finite(), "branch2 logit {i} should be finite: {l}");
        }
    }
}
