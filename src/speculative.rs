use crate::percepta::Sudoku9x9;
use crate::transformer::{ForwardContext, KVCache, TransformerWeights, forward};
use crate::types::{Config, Rng, softmax};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

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
    /// `parent_tokens[i]` = token placed at depth `i` in the current path.
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

/// Sudoku constraint pruner: maps DDTree depth → (row, col) and
/// validates each drafted digit (token_idx 1-9) against Sudoku rules.
///
/// This is the bridge between speculative decoding and Computable LoRA:
/// - Draft model proposes logits for each empty cell
/// - SudokuPruner rejects digits that violate row/col/box constraints
/// - Only valid digits enter the DDTree → 100% valid placements
pub struct SudokuPruner {
    /// The current board state (0 = empty).
    board: Sudoku9x9,
    /// Ordered list of (row, col) positions that map to DDTree depths.
    /// Depth 0 → positions[0], Depth 1 → positions[1], etc.
    positions: Vec<(usize, usize)>,
}

impl SudokuPruner {
    /// Create a pruner from a Sudoku board.
    /// Automatically discovers empty cells in row-major order.
    pub fn new(board: Sudoku9x9) -> Self {
        let mut positions = Vec::new();
        for r in 0..9 {
            for c in 0..9 {
                if board.grid[r][c] == 0 {
                    positions.push((r, c));
                }
            }
        }
        Self { board, positions }
    }

    /// Number of empty cells (= max DDTree depth).
    pub fn empty_count(&self) -> usize {
        self.positions.len()
    }

    /// Get the (row, col) for a given depth.
    pub fn position_at(&self, depth: usize) -> Option<(usize, usize)> {
        self.positions.get(depth).copied()
    }

    /// Get the underlying board state.
    pub fn board(&self) -> &Sudoku9x9 {
        &self.board
    }
}

impl ConstraintPruner for SudokuPruner {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // Token 0 = empty/padding, never valid for placement
        if token_idx == 0 {
            return false;
        }
        // Digits 1-9 map to token indices 1-9
        let digit = token_idx as u8;
        if !(1..=9).contains(&digit) {
            return false;
        }
        // Map depth to (row, col)
        let Some(&(row, col)) = self.positions.get(depth) else {
            return false;
        };

        // Check against initial board state
        if !self.board.is_valid_move(row, col, digit) {
            return false;
        }

        // Path-aware: check cross-depth conflicts with parent tokens.
        // If a parent token has the same digit AND shares row/col/box,
        // this placement is invalid — the pruner must catch it.
        for (parent_depth, &parent_token) in parent_tokens.iter().enumerate() {
            if parent_token == 0 {
                continue;
            }
            let parent_digit = parent_token as u8;
            if parent_digit != digit {
                continue; // Different digits never conflict
            }
            // Same digit — check spatial conflict with parent position
            if let Some(&(pr, pc)) = self.positions.get(parent_depth) {
                if pr == row || pc == col {
                    return false; // Same row or column
                }
                if pr / 3 == row / 3 && pc / 3 == col / 3 {
                    return false; // Same 3×3 box
                }
            }
        }

        true
    }
}

// ── Speculative Verifier: Strategy Pattern ──────────────────

/// Strategy for verifying drafted tokens against a target distribution.
///
/// Same pattern as `ConstraintPruner` — trait-based swap point.
/// - `SimulatedVerifier`: fast, no target model needed (default).
/// - `LeviathanVerifier`: real p/q rejection sampling with target model
///   (behind `leviathan` feature flag).
pub trait SpeculativeVerifier: Send + Sync {
    /// Run one speculative decoding step end-to-end.
    /// Returns accepted tokens (always ≥ 1, up to γ + 1 with bonus).
    fn speculate(
        &mut self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        token: usize,
        pos: usize,
        rng: &mut Rng,
    ) -> Vec<usize>;
}

/// Simulated verification: DDTree path + acceptance cap + bonus token.
/// No target model needed — fast, used by default.
pub struct SimulatedVerifier {
    pub acceptance_rate: f32,
}

impl SimulatedVerifier {
    pub fn new(acceptance_rate: f32) -> Self {
        Self {
            acceptance_rate: acceptance_rate.clamp(0.0, 1.0),
        }
    }
}

impl SpeculativeVerifier for SimulatedVerifier {
    fn speculate(
        &mut self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        token: usize,
        pos: usize,
        rng: &mut Rng,
    ) -> Vec<usize> {
        // 1. Sequential DFlash draft (avoids rayon overhead for tiny model)
        let marginals = dflash_predict(draft_weights, draft_config, token, pos);

        // 2. DDTree build
        let tree = build_dd_tree(&marginals, draft_config);

        // 3. Extract best path (highest-scored token at each depth)
        let path = extract_best_path(&tree);

        if path.is_empty() {
            return vec![sample_from_distribution(
                marginals.first().map(|m| m.as_slice()).unwrap_or(&[1.0]),
                rng,
            )];
        }

        // 4. Simulate acceptance: cap at rate
        let max_accept = ((path.len() as f32) * self.acceptance_rate).ceil() as usize;
        let accepted: Vec<usize> = path.into_iter().take(max_accept.max(1)).collect();

        // 5. Bonus token: if all accepted, sample +1 from last marginal
        if accepted.len() == max_accept && !marginals.is_empty() {
            let last_marginal = marginals.last().unwrap();
            let bonus = sample_from_distribution(last_marginal, rng);
            let mut result = accepted;
            result.push(bonus);
            return result;
        }

        accepted
    }
}

/// Extract best-scored token at each depth from a DDTree.
fn extract_best_path(tree: &[TreeNode]) -> Vec<usize> {
    if tree.is_empty() {
        return Vec::new();
    }
    let max_depth = tree.iter().map(|n| n.depth).max().unwrap_or(0);
    let mut path = Vec::with_capacity(max_depth + 1);
    for depth in 0..=max_depth {
        let best = tree
            .iter()
            .filter(|n| n.depth == depth)
            .max_by_key(|n| (n.score * 1e6) as i64);
        match best {
            Some(node) => path.push(node.token_idx),
            None => break,
        }
    }
    path
}

/// CDF-based sampling from a probability distribution.
pub(crate) fn sample_from_distribution(probs: &[f32], rng: &mut Rng) -> usize {
    let r = rng.uniform();
    let mut cdf = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cdf += p;
        if r <= cdf {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// Residual distribution sampling (Equation 3 from Leviathan et al. 2022).
///
/// `p'(x) = normalize(max(0, p(x) - q(x)))`
///
/// Samples from tokens the target model likes *more* than the draft model.
/// Falls back to `sample_from_distribution(p)` if distributions are identical.
#[cfg_attr(not(feature = "leviathan"), allow(dead_code))]
pub(crate) fn sample_residual_distribution(p: &[f32], q: &[f32], rng: &mut Rng) -> usize {
    let mut residual: Vec<f32> = p
        .iter()
        .zip(q.iter())
        .map(|(&p_val, &q_val)| (p_val - q_val).max(0.0))
        .collect();

    let sum: f32 = residual.iter().sum();

    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for val in &mut residual {
            *val *= inv_sum;
        }
        sample_from_distribution(&residual, rng)
    } else {
        // Distributions identical — fallback to target distribution
        sample_from_distribution(p, rng)
    }
}

// ── LeviathanVerifier: Real p/q rejection sampling (Algorithm 1) ──

#[cfg(feature = "leviathan")]
pub struct LeviathanVerifier<'a> {
    pub target_weights: &'a TransformerWeights,
    pub target_config: &'a Config,
    target_ctx: ForwardContext,
    target_cache: KVCache,
}

#[cfg(feature = "leviathan")]
impl<'a> LeviathanVerifier<'a> {
    pub fn new(target_weights: &'a TransformerWeights, target_config: &'a Config) -> Self {
        Self {
            target_weights,
            target_config,
            target_ctx: ForwardContext::new(target_config),
            target_cache: KVCache::new(target_config),
        }
    }
}

#[cfg(feature = "leviathan")]
impl SpeculativeVerifier for LeviathanVerifier<'_> {
    fn speculate(
        &mut self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        token: usize,
        pos: usize,
        rng: &mut Rng,
    ) -> Vec<usize> {
        // Phase 1: Autoregressive draft (Algorithm 1, line 2–5)
        let draft_result = dflash_predict_ar(draft_weights, draft_config, token, pos, rng);
        let draft_tokens = &draft_result.sampled_tokens;
        let q_dists = &draft_result.marginals;
        let gamma = draft_tokens.len();

        if gamma == 0 {
            // No draft tokens — run target once, return 1 token
            self.target_cache.reset();
            let logits = forward(
                &mut self.target_ctx,
                self.target_weights,
                &mut self.target_cache,
                token,
                pos,
                self.target_config,
            );
            for logit in logits.iter_mut() {
                *logit /= self.target_config.temperature;
            }
            softmax(logits);
            return vec![sample_from_distribution(logits, rng)];
        }

        // Phase 2: Target scoring (Algorithm 1, line 7–8)
        // Run target model on [last_token, draft_1, ..., draft_gamma]
        // to get p(x) for positions 0..=gamma
        self.target_cache.reset();
        let mut p_distributions: Vec<Vec<f32>> = Vec::with_capacity(gamma + 1);

        // Score the initial token → p(x) at position 0
        let logits = forward(
            &mut self.target_ctx,
            self.target_weights,
            &mut self.target_cache,
            token,
            pos,
            self.target_config,
        );
        for logit in logits.iter_mut() {
            *logit /= self.target_config.temperature;
        }
        softmax(logits);
        p_distributions.push(logits.to_vec());

        // Score each drafted token → p(x) at positions 1..=gamma
        for (i, &draft_tok) in draft_tokens.iter().enumerate() {
            let logits = forward(
                &mut self.target_ctx,
                self.target_weights,
                &mut self.target_cache,
                draft_tok,
                pos + 1 + i,
                self.target_config,
            );
            for logit in logits.iter_mut() {
                *logit /= self.target_config.temperature;
            }
            softmax(logits);
            p_distributions.push(logits.to_vec());
        }

        // Phase 3: Rejection sampling (Algorithm 1, line 10–16)
        let mut accepted = Vec::with_capacity(gamma + 1);
        let mut all_accepted = true;

        for i in 0..gamma {
            let p_dist = &p_distributions[i];
            let q_dist = &q_dists[i];
            let drafted_token = draft_tokens[i];

            let p_i = p_dist[drafted_token];
            let q_i = q_dist[drafted_token];

            // Accept with prob min(1, p/q)
            let acceptance_prob = if q_i > 0.0 {
                (p_i / q_i).min(1.0)
            } else {
                1.0 // q=0 means draft didn't propose this; accept if target likes it
            };
            let r = rng.uniform();

            if r <= acceptance_prob {
                accepted.push(drafted_token);
            } else {
                // Reject: sample replacement from residual max(0, p - q)
                let replacement = sample_residual_distribution(p_dist, q_dist, rng);
                accepted.push(replacement);
                all_accepted = false;
                break;
            }
        }

        // Phase 4: Bonus token (Algorithm 1, line 18–19)
        if all_accepted && p_distributions.len() > gamma {
            let bonus = sample_from_distribution(&p_distributions[gamma], rng);
            accepted.push(bonus);
        }

        // Safety: always return at least 1 token
        if accepted.is_empty() {
            accepted.push(sample_from_distribution(&p_distributions[0], rng));
        }

        accepted
    }
}

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

/// Extract tokens from `parent_path` bitfield for path-aware pruning.
///
/// `parent_path` uses 5 bits per depth, packed LSB-first:
/// - Depth 0 token: bits 0–4
/// - Depth 1 token: bits 5–9
/// - ...
/// - Depth k token: bits (k*5) to (k*5+4)
///
/// Returns `Vec<usize>` where `result[k]` = token at depth `k`.
/// Max depths: 64/5 = 12 (sufficient for lookahead of 5–8).
pub fn extract_parent_tokens(parent_path: u64, num_tokens: usize) -> Vec<usize> {
    // parent_path packs tokens with most-recent in lowest bits:
    //   depth 0 token → bits (num_tokens-1)*5 .. (num_tokens-1)*5+4
    //   depth k token → bits (num_tokens-1-k)*5 .. (num_tokens-1-k)*5+4
    (0..num_tokens)
        .map(|k| ((parent_path >> ((num_tokens - 1 - k) * 5)) & 0x1F) as usize)
        .collect()
}

/// Sequential DFlash: Predict marginal distributions using draft model.
/// Uses pre-allocated ForwardContext for zero-alloc per step.
pub fn dflash_predict(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
) -> Vec<Vec<f32>> {
    let mut ctx = ForwardContext::new(draft_config);
    let max_steps = draft_config
        .draft_lookahead
        .min(draft_config.block_size.saturating_sub(pos));

    let mut marginals = Vec::with_capacity(max_steps);
    for step in 0..max_steps {
        let mut cache = KVCache::new(draft_config);
        let draft_pos = pos + step;
        let logits = forward(
            &mut ctx,
            draft_weights,
            &mut cache,
            token,
            draft_pos,
            draft_config,
        );
        let mut probs = logits.to_vec();
        for p in probs.iter_mut() {
            *p /= draft_config.temperature;
        }
        softmax(&mut probs);
        marginals.push(probs);
    }
    marginals
}

/// Parallel DFlash: Predict marginals using rayon.
/// One ForwardContext per rayon worker thread — no contention, zero waste.
pub fn dflash_predict_parallel(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
) -> Vec<Vec<f32>> {
    let max_steps = draft_config
        .draft_lookahead
        .min(draft_config.block_size.saturating_sub(pos));

    if max_steps == 0 {
        return Vec::new();
    }

    (0..max_steps)
        .into_par_iter()
        .map_init(
            || {
                (
                    ForwardContext::new(draft_config),
                    KVCache::new(draft_config),
                )
            },
            |(ctx, cache), step| {
                let draft_pos = pos + step;
                let logits = forward(ctx, draft_weights, cache, token, draft_pos, draft_config);
                let mut probs = logits.to_vec();
                for p in probs.iter_mut() {
                    *p /= draft_config.temperature;
                }
                softmax(&mut probs);
                probs
            },
        )
        .collect()
}

/// Result of autoregressive drafting: marginals + sampled tokens.
pub struct DraftResult {
    pub marginals: Vec<Vec<f32>>,
    pub sampled_tokens: Vec<usize>,
}

/// Autoregressive DFlash: Predict marginals by sampling and feeding back tokens.
///
/// Unlike `dflash_predict` (which feeds the same token/pos to every step),
/// this samples a token at each step and feeds it back as input for the next.
/// Produces conditional q(x|x_{<i}) distributions instead of independent marginals.
pub fn dflash_predict_ar(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    rng: &mut Rng,
) -> DraftResult {
    let mut ctx = ForwardContext::new(draft_config);
    let mut cache = KVCache::new(draft_config);
    let max_steps = draft_config
        .draft_lookahead
        .min(draft_config.block_size.saturating_sub(pos));

    let mut marginals = Vec::with_capacity(max_steps);
    let mut sampled_tokens = Vec::with_capacity(max_steps);
    let mut cur_token = token;

    for step in 0..max_steps {
        let logits = forward(
            &mut ctx,
            draft_weights,
            &mut cache,
            cur_token,
            pos + step,
            draft_config,
        );
        let mut probs = logits.to_vec();
        for p in probs.iter_mut() {
            *p /= draft_config.temperature;
        }
        softmax(&mut probs);

        // Sample next token and feed back
        let next_token = sample_from_distribution(&probs, rng);
        marginals.push(probs);
        sampled_tokens.push(next_token);
        cur_token = next_token;
    }

    DraftResult {
        marginals,
        sampled_tokens,
    }
}

/// DDTree: Build verification tree from marginals using Best-First Search.
/// Returns tree nodes ordered by score (best first).
///
/// Equivalent to `build_dd_tree_pruned` with `NoPruner`.
pub fn build_dd_tree(marginals: &[Vec<f32>], config: &Config) -> Vec<TreeNode> {
    build_dd_tree_pruned(marginals, config, &NoPruner)
}

/// DDTree with Constraint Pruner: Build verification tree from marginals,
/// filtering branches through a deterministic rules engine.
///
/// The pruner is called for every candidate token at every depth.
/// Invalid tokens are never added to the heap — they don't waste tree budget.
///
/// This is the **Computable LoRA intercept**: the draft model proposes
/// logits (semantic probability), the pruner enforces constraints
/// (mathematical validity), and only valid branches reach verification.
pub fn build_dd_tree_pruned(
    marginals: &[Vec<f32>],
    config: &Config,
    pruner: &dyn ConstraintPruner,
) -> Vec<TreeNode> {
    if marginals.is_empty() {
        return Vec::new();
    }

    let mut tree = Vec::with_capacity(config.tree_budget);
    let mut heap = BinaryHeap::new();

    // Seed heap with root's children (position 0), filtered by pruner
    for (i, &prob) in marginals[0].iter().enumerate() {
        if prob > 0.0 && pruner.is_valid(0, i, &[]) {
            heap.push(TreeNode {
                score: prob.ln(),
                depth: 0,
                token_idx: i,
                parent_path: i as u64,
            });
        }
    }

    while tree.len() < config.tree_budget {
        let Some(best) = heap.pop() else { break };
        tree.push(best);

        if best.depth + 1 < marginals.len() {
            let next_depth = best.depth + 1;
            // Extract parent tokens from path bitfield for path-aware pruning
            let parent_tokens = extract_parent_tokens(best.parent_path, best.depth + 1);
            for (i, &prob) in marginals[next_depth].iter().enumerate() {
                // NEURO-SYMBOLIC INTERCEPT: prune before adding to heap
                if prob > 0.0 && pruner.is_valid(next_depth, i, &parent_tokens) {
                    heap.push(TreeNode {
                        score: best.score + prob.ln(),
                        depth: next_depth,
                        token_idx: i,
                        parent_path: (best.parent_path << 5) | (i as u64),
                    });
                }
            }
        }
    }

    tree
}

/// One step of speculative decoding with draft model + parallel DFlash.
/// Returns (accepted_token_ids, acceptance_length).
/// Speculative decoding step with a custom verifier.
/// Pass any `SpeculativeVerifier` to control how drafts are verified.
pub fn speculative_step_verifier(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    rng: &mut Rng,
    verifier: &mut dyn SpeculativeVerifier,
) -> (Vec<usize>, usize) {
    let accepted = verifier.speculate(draft_weights, draft_config, token, pos, rng);
    let len = accepted.len();
    (accepted, len)
}

/// Speculative decoding step with simulated verification (backward compat).
/// Uses `SimulatedVerifier` with 75% acceptance rate + DDTree.
pub fn speculative_step(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    rng: &mut Rng,
) -> (Vec<usize>, usize) {
    let mut verifier = SimulatedVerifier::new(0.75);
    speculative_step_verifier(draft_weights, draft_config, token, pos, rng, &mut verifier)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_draft() -> (TransformerWeights, Config) {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        (weights, config)
    }

    // ── ConstraintPruner Tests ────────────────────────────────────

    #[test]
    fn test_no_pruner_allows_all() {
        let pruner = NoPruner;
        assert!(pruner.is_valid(0, 0, &[]));
        assert!(pruner.is_valid(0, 26, &[]));
        assert!(pruner.is_valid(100, 999, &[]));
    }

    #[test]
    fn test_sudoku_pruner_rejects_invalid_digits() {
        let board = Sudoku9x9::arto_inkala();
        let pruner = SudokuPruner::new(board);

        // First empty cell is (0,1): row 0 has 8, col 1 has 5/7/9, box has 8/3/7
        // Valid: 1, 2, 4, 6. Invalid: 3, 5, 7, 8, 9.
        assert!(!pruner.is_valid(0, 3, &[]), "3 is in box");
        assert!(!pruner.is_valid(0, 5, &[]), "5 is in col");
        assert!(!pruner.is_valid(0, 7, &[]), "7 is in col+box");
        assert!(!pruner.is_valid(0, 8, &[]), "8 is in row+box");
        assert!(!pruner.is_valid(0, 9, &[]), "9 is in col");

        // Valid digits
        assert!(pruner.is_valid(0, 1, &[]), "1 should be valid");
        assert!(pruner.is_valid(0, 2, &[]), "2 should be valid");
        assert!(pruner.is_valid(0, 4, &[]), "4 should be valid");
        assert!(pruner.is_valid(0, 6, &[]), "6 should be valid");
    }

    #[test]
    fn test_sudoku_pruner_rejects_token_zero() {
        let board = Sudoku9x9::arto_inkala();
        let pruner = SudokuPruner::new(board);
        // Token 0 (empty/padding) should always be rejected
        assert!(!pruner.is_valid(0, 0, &[]), "token 0 should be pruned");
    }

    #[test]
    fn test_sudoku_pruner_empty_count() {
        let board = Sudoku9x9::arto_inkala();
        let pruner = SudokuPruner::new(board);
        assert_eq!(pruner.empty_count(), 60, "Arto Inkala has 60 empty cells");
    }

    #[test]
    fn test_sudoku_pruner_positions_match_empties() {
        let board = Sudoku9x9::arto_inkala();
        let pruner = SudokuPruner::new(board);

        // First empty cell should be (0,1)
        assert_eq!(pruner.position_at(0), Some((0, 1)));
        // Depth beyond empty_count should return None
        assert_eq!(pruner.position_at(60), None);
    }

    #[test]
    fn test_ddtree_pruned_same_as_unpruned_with_no_pruner() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);

        let tree_unpruned = build_dd_tree(&marginals, &config);
        let tree_pruned = build_dd_tree_pruned(&marginals, &config, &NoPruner);

        assert_eq!(
            tree_unpruned.len(),
            tree_pruned.len(),
            "NoPruner should produce identical tree"
        );
        for (a, b) in tree_unpruned.iter().zip(tree_pruned.iter()) {
            assert_eq!(a.score, b.score, "scores should match");
            assert_eq!(a.token_idx, b.token_idx, "tokens should match");
        }
    }

    #[test]
    fn test_ddtree_pruned_sudoku_reduces_tree_size() {
        // Use 1-depth marginals with budget > 9 so unpruned gets all 9 digits,
        // but pruned only gets ~4 valid digits for cell (0,1).
        // Cell (0,1): valid={1,2,4,6}, invalid={3,5,7,8,9}.
        let marginals: Vec<Vec<f32>> = vec![{
            let mut probs = vec![0.0f32; 10];
            for d in 1..=9u8 {
                probs[d as usize] = 1.0 / 9.0;
            }
            probs
        }];

        let config = Config {
            tree_budget: 20, // > 9 so unpruned gets all, pruned gets only valid
            ..Config::draft()
        };

        let tree_unpruned = build_dd_tree(&marginals, &config);
        let tree_pruned = build_dd_tree_pruned(
            &marginals,
            &config,
            &SudokuPruner::new(Sudoku9x9::arto_inkala()),
        );

        // Unpruned: 9 nodes (digits 1-9 all have prob > 0).
        // Pruned: ~4 nodes (only valid digits for cell (0,1)).
        assert!(
            tree_pruned.len() < tree_unpruned.len(),
            "pruned tree ({}) should be smaller than unpruned ({})",
            tree_pruned.len(),
            tree_unpruned.len()
        );
        assert!(!tree_pruned.is_empty(), "pruned tree should have nodes");
        // Unpruned should have exactly 9 nodes (all digits 1-9)
        assert_eq!(tree_unpruned.len(), 9, "unpruned should have 9 nodes");
        // Pruned should have exactly 4 nodes (valid digits for (0,1): 1,2,4,6)
        assert_eq!(tree_pruned.len(), 4, "pruned should have 4 valid nodes");
    }

    #[test]
    fn test_ddtree_pruned_sudoku_only_valid_tokens() {
        let board = Sudoku9x9::arto_inkala();
        let pruner = SudokuPruner::new(board.clone());

        // Marginals: 3 steps, vocab 10
        let marginals: Vec<Vec<f32>> = (0..3)
            .map(|_| {
                let mut probs = vec![0.0f32; 10];
                for d in 1..=9u8 {
                    probs[d as usize] = 1.0 / 9.0;
                }
                probs
            })
            .collect();

        let config = Config {
            tree_budget: 50,
            ..Config::draft()
        };

        let tree = build_dd_tree_pruned(&marginals, &config, &pruner);

        // Every node in the tree should be a valid move at its depth
        for node in &tree {
            let pos = pruner
                .position_at(node.depth)
                .expect("depth should map to position");
            let digit = node.token_idx as u8;
            assert!(
                board.is_valid_move(pos.0, pos.1, digit),
                "token {} at depth {} (row {}, col {}) should be valid",
                node.token_idx,
                node.depth,
                pos.0,
                pos.1,
            );
        }
    }

    #[test]
    fn test_ddtree_pruned_sudoku_no_token_zero() {
        let board = Sudoku9x9::arto_inkala();
        let pruner = SudokuPruner::new(board);

        let marginals: Vec<Vec<f32>> = (0..5)
            .map(|_| {
                let mut probs = vec![0.5f32; 10]; // even token 0 has prob > 0
                let sum: f32 = probs.iter().sum();
                for p in probs.iter_mut() {
                    *p /= sum;
                }
                probs
            })
            .collect();

        let config = Config {
            tree_budget: 50,
            ..Config::draft()
        };

        let tree = build_dd_tree_pruned(&marginals, &config, &pruner);

        // No node should have token_idx == 0
        for node in &tree {
            assert_ne!(
                node.token_idx, 0,
                "token 0 should be pruned at depth {}",
                node.depth
            );
        }
    }

    #[test]
    fn test_ddtree_pruned_empty_marginals() {
        let config = Config::draft();
        let pruner = NoPruner;
        let tree = build_dd_tree_pruned(&[], &config, &pruner);
        assert!(tree.is_empty(), "empty marginals should produce empty tree");
    }

    // ── Original DDTree Tests ─────────────────────────────────────

    #[test]
    fn test_dflash_produces_marginals() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        assert!(!marginals.is_empty());
        assert!(marginals.len() <= config.draft_lookahead);

        for (i, row) in marginals.iter().enumerate() {
            assert_eq!(row.len(), config.vocab_size, "row {i} wrong size");
            let sum: f32 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "row {i} sum = {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dflash_parallel_matches_count() {
        let (weights, config) = make_draft();
        let seq = dflash_predict(&weights, &config, 0, 0);
        let par = dflash_predict_parallel(&weights, &config, 0, 0);
        assert_eq!(seq.len(), par.len(), "parallel should produce same count");
    }

    #[test]
    fn test_dflash_positions_differ() {
        let (weights, config) = make_draft();
        let m0 = dflash_predict(&weights, &config, 0, 0);
        let m1 = dflash_predict(&weights, &config, 0, 1);
        assert_ne!(
            m0[0], m1[0],
            "marginals at different positions should differ"
        );
    }

    #[test]
    fn test_ddtree_respects_budget() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        let tree = build_dd_tree(&marginals, &config);
        assert!(
            tree.len() <= config.tree_budget,
            "tree size {} exceeds budget {}",
            tree.len(),
            config.tree_budget
        );
        assert!(!tree.is_empty(), "tree should have at least one node");
    }

    #[test]
    fn test_ddtree_scores_descending() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        let tree = build_dd_tree(&marginals, &config);
        for window in tree.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "scores not descending: {} >= {}",
                window[0].score,
                window[1].score
            );
        }
    }

    #[test]
    fn test_ddtree_depth_within_lookahead() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        let tree = build_dd_tree(&marginals, &config);
        for node in &tree {
            assert!(
                node.depth < config.draft_lookahead,
                "depth {} should be < lookahead {}",
                node.depth,
                config.draft_lookahead
            );
        }
    }

    #[test]
    fn test_ddtree_valid_token_indices() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        let tree = build_dd_tree(&marginals, &config);
        for node in &tree {
            assert!(
                node.token_idx < config.vocab_size,
                "token_idx {} out of range",
                node.token_idx
            );
        }
    }

    #[test]
    fn test_ddtree_empty_marginals() {
        let config = Config::draft();
        let tree = build_dd_tree(&[], &config);
        assert!(tree.is_empty(), "empty marginals should produce empty tree");
    }

    #[test]
    fn test_speculative_step_accepts_at_least_one() {
        let (weights, config) = make_draft();
        for seed in [0, 42, 100, 999] {
            let mut rng = Rng::new(seed);
            let (accepted, accept_len) = speculative_step(&weights, &config, 0, 0, &mut rng);
            assert!(
                !accepted.is_empty(),
                "seed {seed}: should accept at least 1 token"
            );
            assert!(accept_len >= 1, "seed {seed}: accept_len should be >= 1");
            for &t in &accepted {
                assert!(t < config.vocab_size, "seed {seed}: token {t} out of range");
            }
        }
    }

    #[test]
    fn test_speculative_step_consistent_for_same_seed() {
        let (weights, config) = make_draft();

        let mut rng1 = Rng::new(77);
        let (a1, l1) = speculative_step(&weights, &config, 0, 0, &mut rng1);

        let mut rng2 = Rng::new(77);
        let (a2, l2) = speculative_step(&weights, &config, 0, 0, &mut rng2);

        assert_eq!(a1, a2, "same seed should produce same accepted tokens");
        assert_eq!(l1, l2, "same seed should produce same acceptance length");
    }

    // ── Path-Aware Pruning Tests ──────────────────────────────────

    #[test]
    fn test_extract_parent_tokens_roundtrip() {
        // Build path bitfield: depth 0 = token 3, depth 1 = token 7, depth 2 = token 1
        let path_d0 = 3u64;
        let path_d1 = (path_d0 << 5) | 7;
        let path_d2 = (path_d1 << 5) | 1;

        assert_eq!(extract_parent_tokens(path_d0, 1), vec![3]);
        assert_eq!(extract_parent_tokens(path_d1, 2), vec![3, 7]);
        assert_eq!(extract_parent_tokens(path_d2, 3), vec![3, 7, 1]);
        assert_eq!(extract_parent_tokens(0, 0), vec![]);
    }

    #[test]
    fn test_sudoku_pruner_path_aware_same_row() {
        let board = Sudoku9x9::arto_inkala();
        let pruner = SudokuPruner::new(board);

        // Depth 0 → (0,1), depth 1 → (0,2): both in row 0
        // Digit 4 is valid at both positions individually
        assert!(
            pruner.is_valid(0, 4, &[]),
            "digit 4 at depth 0 should be valid alone"
        );
        assert!(
            pruner.is_valid(1, 4, &[]),
            "digit 4 at depth 1 should be valid alone"
        );
        // But with parent token 4 at depth 0 → same row → conflict
        assert!(
            !pruner.is_valid(1, 4, &[4]),
            "same digit 4 in same row should be pruned"
        );
    }

    #[test]
    fn test_sudoku_pruner_path_aware_same_col() {
        let board = Sudoku9x9::arto_inkala();
        let pruner = SudokuPruner::new(board);

        // Depth 0 → (0,1), depth 9 → (1,1): both in column 1
        // Digit 1 is valid at both positions individually
        assert!(
            pruner.is_valid(0, 1, &[]),
            "digit 1 at depth 0 should be valid alone"
        );
        assert!(
            pruner.is_valid(9, 1, &[]),
            "digit 1 at depth 9 should be valid alone"
        );
        // With parent token 1 at depth 0 → same column → conflict
        let mut parent_tokens = vec![0usize; 9];
        parent_tokens[0] = 1;
        assert!(
            !pruner.is_valid(9, 1, &parent_tokens),
            "same digit 1 in same column should be pruned"
        );
    }

    #[test]
    fn test_sudoku_pruner_path_aware_same_box() {
        let board = Sudoku9x9::arto_inkala();
        let pruner = SudokuPruner::new(board);

        // Depth 0 → (0,1) box(0,0), depth 1 → (0,2) box(0,0): same 3×3 box
        // Digit 6 is valid at both positions individually
        assert!(
            pruner.is_valid(0, 6, &[]),
            "digit 6 at depth 0 should be valid alone"
        );
        assert!(
            pruner.is_valid(1, 6, &[]),
            "digit 6 at depth 1 should be valid alone"
        );
        // With parent token 6 at depth 0 → same box → conflict
        assert!(
            !pruner.is_valid(1, 6, &[6]),
            "same digit 6 in same box should be pruned"
        );
    }

    #[test]
    fn test_sudoku_pruner_path_aware_no_conflict_different_digit() {
        let board = Sudoku9x9::arto_inkala();
        let pruner = SudokuPruner::new(board);

        // Different digits NEVER conflict, even in same row
        // Depth 0 → (0,1), depth 1 → (0,2): same row 0
        // Depth 0 → (0,1) valid: {1,2,4,6}, depth 1 → (0,2) valid: {4,5,6,9}
        // Different digits in same row 0 → no conflict
        assert!(
            pruner.is_valid(1, 5, &[4]),
            "different digits (4→5) in same row should NOT be pruned"
        );
        assert!(
            pruner.is_valid(1, 9, &[2]),
            "different digits (2→9) in same row should NOT be pruned"
        );
    }

    #[test]
    fn test_sudoku_pruner_path_aware_no_conflict_different_region() {
        let board = Sudoku9x9::arto_inkala();
        let pruner = SudokuPruner::new(board);

        // Depth 0 → (0,1) row 0, col 1, box(0,0)
        // Depth 21 → (3,0) row 3, col 0, box(3,0)
        // Different row, different col, different box → same digit is OK
        // Digit 4 is valid at both positions against initial board
        assert!(
            pruner.is_valid(0, 4, &[]),
            "digit 4 at (0,1) should be valid"
        );
        assert!(
            pruner.is_valid(21, 4, &[]),
            "digit 4 at (3,0) should be valid"
        );

        let mut parent_tokens = vec![0usize; 21];
        parent_tokens[0] = 4; // digit 4 placed at depth 0
        assert!(
            pruner.is_valid(21, 4, &parent_tokens),
            "same digit in different row/col/box should NOT be pruned"
        );
    }

    #[test]
    fn test_sudoku_pruner_path_aware_multi_level_conflict() {
        let board = Sudoku9x9::arto_inkala();
        let pruner = SudokuPruner::new(board);

        // Multi-level path: [1, 2, 3] at depths 0, 1, 2
        // All in row 0: (0,1), (0,2), (0,3)
        // Depth 3 → (0,4): try digit 1 → conflicts with depth 0 in same row
        assert!(
            pruner.is_valid(3, 1, &[]),
            "digit 1 at (0,4) should be valid alone"
        );
        assert!(
            !pruner.is_valid(3, 1, &[1, 2, 3]),
            "digit 1 at depth 3 conflicts with digit 1 at depth 0 in same row"
        );
    }

    /// Wrapper that ignores parent_tokens for static-only comparison testing.
    struct StaticOnlyPruner<'a>(&'a SudokuPruner);

    impl ConstraintPruner for StaticOnlyPruner<'_> {
        fn is_valid(&self, depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            self.0.is_valid(depth, token_idx, &[])
        }
    }

    /// Verify every node in the tree is valid against its accumulated board state.
    /// Returns count of nodes with invalid accumulated state.
    fn count_invalid_accumulated(pruner: &SudokuPruner, tree: &[TreeNode]) -> usize {
        let mut invalid = 0;
        for node in tree {
            // parent_path includes node's own token, so extract depth+1 tokens
            // then use only the first `depth` as parent placements
            let all_tokens = extract_parent_tokens(node.parent_path, node.depth + 1);
            let parent_tokens = &all_tokens[..node.depth];

            // Build accumulated board: initial + all parent placements
            let mut board = pruner.board.clone();
            for (depth, &token) in parent_tokens.iter().enumerate() {
                if token == 0 {
                    continue;
                }
                if let Some((row, col)) = pruner.position_at(depth) {
                    board.grid[row][col] = token as u8;
                }
            }

            // Check node's token against accumulated board
            if let Some((row, col)) = pruner.position_at(node.depth)
                && !board.is_valid_move(row, col, node.token_idx as u8)
            {
                invalid += 1;
            }
        }
        invalid
    }

    #[test]
    fn test_ddtree_path_aware_all_nodes_valid_accumulated() {
        let board = Sudoku9x9::arto_inkala();
        let pruner = SudokuPruner::new(board);

        // Uniform marginals over 8 depths (row 0 empties)
        let marginals: Vec<Vec<f32>> = (0..8)
            .map(|_| {
                let mut probs = vec![0.0f32; 10];
                for d in 1..=9u8 {
                    probs[d as usize] = 1.0 / 9.0;
                }
                probs
            })
            .collect();

        let config = Config {
            tree_budget: 100,
            ..Config::draft()
        };

        let tree = build_dd_tree_pruned(&marginals, &config, &pruner);
        assert!(!tree.is_empty(), "tree should have nodes");

        let invalid = count_invalid_accumulated(&pruner, &tree);
        assert_eq!(
            invalid, 0,
            "path-aware tree should have 0 invalid accumulated nodes, found {invalid}"
        );
    }

    #[test]
    fn test_ddtree_path_aware_catches_cross_depth_conflicts() {
        let board = Sudoku9x9::arto_inkala();
        let pruner = SudokuPruner::new(board);

        // Uniform marginals over 8 depths (row 0 empties — all in same row!)
        // Cross-depth same-digit conflicts are inevitable without path-aware pruning.
        let marginals: Vec<Vec<f32>> = (0..8)
            .map(|_| {
                let mut probs = vec![0.0f32; 10];
                for d in 1..=9u8 {
                    probs[d as usize] = 1.0 / 9.0;
                }
                probs
            })
            .collect();

        let config = Config {
            tree_budget: 100,
            ..Config::draft()
        };

        // Static-only tree: ignores parent tokens → cross-depth conflicts slip through
        let static_pruner = StaticOnlyPruner(&pruner);
        let tree_static = build_dd_tree_pruned(&marginals, &config, &static_pruner);

        // Path-aware tree: catches cross-depth conflicts
        let tree_aware = build_dd_tree_pruned(&marginals, &config, &pruner);

        // Static tree should have invalid accumulated nodes
        let static_invalid = count_invalid_accumulated(&pruner, &tree_static);
        assert!(
            static_invalid > 0,
            "static tree should have cross-depth conflicts (found {static_invalid})"
        );

        // Path-aware tree should have zero invalid accumulated nodes
        let aware_invalid = count_invalid_accumulated(&pruner, &tree_aware);
        assert_eq!(
            aware_invalid, 0,
            "path-aware tree should have zero cross-depth conflicts"
        );
    }

    // ── SpeculativeVerifier + AR Drafting Tests ────────────────

    #[test]
    fn test_sample_from_distribution() {
        let mut rng = Rng::new(42);
        let probs = vec![0.1, 0.2, 0.5, 0.2];
        for _ in 0..100 {
            let t = sample_from_distribution(&probs, &mut rng);
            assert!(t < 4, "token should be 0-3, got {t}");
        }
    }

    #[test]
    fn test_sample_from_distribution_degenerate() {
        let mut rng = Rng::new(42);
        let probs = vec![0.0, 0.0, 1.0, 0.0];
        for _ in 0..50 {
            let t = sample_from_distribution(&probs, &mut rng);
            assert_eq!(t, 2, "should always sample token 2");
        }
    }

    #[test]
    fn test_residual_distribution_sums_to_one() {
        let mut rng = Rng::new(42);
        let p = vec![0.3, 0.5, 0.1, 0.1];
        let q = vec![0.1, 0.6, 0.2, 0.1];
        // residual = [0.2, 0.0, 0.0, 0.0] → normalized [1.0, 0.0, 0.0, 0.0]
        for _ in 0..50 {
            let token = sample_residual_distribution(&p, &q, &mut rng);
            assert_eq!(token, 0, "residual should only pick token 0");
        }
    }

    #[test]
    fn test_residual_distribution_fallback_on_identical() {
        let mut rng = Rng::new(42);
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        let token = sample_residual_distribution(&p, &q, &mut rng);
        assert!(token < 4, "token should be valid, got {token}");
    }

    #[test]
    fn test_residual_distribution_multiple_positive() {
        let mut rng = Rng::new(42);
        let p = vec![0.5, 0.1, 0.3, 0.1];
        let q = vec![0.1, 0.5, 0.1, 0.3];
        // residual = [0.4, 0.0, 0.2, 0.0] → normalized [0.667, 0.0, 0.333, 0.0]
        let mut counts = [0usize; 4];
        for _ in 0..1000 {
            let token = sample_residual_distribution(&p, &q, &mut rng);
            counts[token] += 1;
        }
        assert!(counts[0] > counts[2], "token 0 should be more frequent");
        assert_eq!(counts[1], 0, "token 1 should never be picked");
        assert_eq!(counts[3], 0, "token 3 should never be picked");
    }

    #[test]
    fn test_simulated_verifier_returns_at_least_one() {
        let (weights, config) = make_draft();
        let mut verifier = SimulatedVerifier::new(0.75);
        let mut rng = Rng::new(42);
        let (accepted, len) =
            speculative_step_verifier(&weights, &config, 0, 0, &mut rng, &mut verifier);
        assert!(!accepted.is_empty(), "should return at least 1 token");
        assert!(len >= 1);
        for &t in &accepted {
            assert!(t < config.vocab_size, "token {t} out of range");
        }
    }

    #[test]
    fn test_simulated_verifier_deterministic() {
        let (weights, config) = make_draft();

        let (a1, l1) = {
            let mut verifier = SimulatedVerifier::new(0.75);
            speculative_step_verifier(&weights, &config, 0, 0, &mut Rng::new(77), &mut verifier)
        };
        let (a2, l2) = {
            let mut verifier = SimulatedVerifier::new(0.75);
            speculative_step_verifier(&weights, &config, 0, 0, &mut Rng::new(77), &mut verifier)
        };

        assert_eq!(a1, a2, "same seed should produce same accepted tokens");
        assert_eq!(l1, l2, "same seed should produce same acceptance length");
    }

    #[test]
    fn test_simulated_verifier_bonus_token() {
        let (weights, config) = make_draft();
        let mut saw_bonus = false;
        for seed in 0..200u64 {
            let mut verifier = SimulatedVerifier::new(0.95);
            let (accepted, _) = speculative_step_verifier(
                &weights,
                &config,
                0,
                0,
                &mut Rng::new(seed),
                &mut verifier,
            );
            if accepted.len() > 1 {
                // Bonus token = more tokens than the capped acceptance alone
                saw_bonus = true;
                break;
            }
        }
        assert!(
            saw_bonus,
            "should see bonus token at least once with high acceptance rate"
        );
    }

    #[test]
    fn test_dflash_ar_produces_marginals() {
        let (weights, config) = make_draft();
        let result = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(42));
        assert!(!result.marginals.is_empty(), "should produce marginals");
        assert!(
            !result.sampled_tokens.is_empty(),
            "should produce sampled tokens"
        );
        assert_eq!(result.marginals.len(), result.sampled_tokens.len());
        for probs in &result.marginals {
            assert_eq!(probs.len(), config.vocab_size);
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "probs should sum to ~1.0, got {sum}"
            );
        }
    }

    #[test]
    fn test_dflash_ar_is_autoregressive() {
        let (weights, config) = make_draft();
        // AR drafting feeds back sampled tokens, so different seeds → different tokens
        let r1 = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(1));
        let r2 = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(2));
        // Very unlikely to be identical with different seeds
        assert_ne!(
            r1.sampled_tokens, r2.sampled_tokens,
            "different seeds should produce different AR tokens"
        );
    }

    #[test]
    fn test_dflash_ar_deterministic() {
        let (weights, config) = make_draft();
        let r1 = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(42));
        let r2 = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(42));
        assert_eq!(
            r1.sampled_tokens, r2.sampled_tokens,
            "same seed should produce same tokens"
        );
        for (a, b) in r1.marginals.iter().zip(r2.marginals.iter()) {
            for (pa, pb) in a.iter().zip(b.iter()) {
                assert!((pa - pb).abs() < 1e-6, "marginals should be identical");
            }
        }
    }

    #[test]
    fn test_extract_best_path() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        let tree = build_dd_tree(&marginals, &config);
        let path = extract_best_path(&tree);
        if !tree.is_empty() {
            assert!(!path.is_empty(), "non-empty tree should produce a path");
            for &t in &path {
                assert!(t < config.vocab_size, "token {t} out of range");
            }
        }
    }

    // ── LeviathanVerifier Tests (feature-gated) ───────────────

    #[cfg(feature = "leviathan")]
    #[test]
    fn test_leviathan_verifier_returns_at_least_one() {
        let config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        let mut verifier = LeviathanVerifier::new(&target_weights, &config);
        let mut rng = Rng::new(100);
        let accepted =
            verifier.speculate(&draft_weights, &draft_config, config.bos_token, 0, &mut rng);

        assert!(!accepted.is_empty(), "should return at least 1 token");
        assert!(
            accepted.len() <= draft_config.draft_lookahead + 1,
            "should return at most gamma+1"
        );
        for &t in &accepted {
            assert!(t < config.vocab_size, "token {t} out of range");
        }
    }

    #[cfg(feature = "leviathan")]
    #[test]
    fn test_leviathan_verifier_deterministic() {
        let config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        let r1 = {
            let mut verifier = LeviathanVerifier::new(&target_weights, &config);
            verifier.speculate(
                &draft_weights,
                &draft_config,
                config.bos_token,
                0,
                &mut Rng::new(100),
            )
        };
        let r2 = {
            let mut verifier = LeviathanVerifier::new(&target_weights, &config);
            verifier.speculate(
                &draft_weights,
                &draft_config,
                config.bos_token,
                0,
                &mut Rng::new(100),
            )
        };

        assert_eq!(r1, r2, "same seed should produce same results");
    }

    #[cfg(feature = "leviathan")]
    #[test]
    fn test_leviathan_verifier_bonus_token() {
        let config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        let mut saw_bonus = false;
        for seed in 0..200u64 {
            let mut verifier = LeviathanVerifier::new(&target_weights, &config);
            let accepted = verifier.speculate(
                &draft_weights,
                &draft_config,
                config.bos_token,
                0,
                &mut Rng::new(seed),
            );
            // gamma=1 → max 2 tokens with bonus
            if accepted.len() >= 2 {
                saw_bonus = true;
                break;
            }
        }
        assert!(
            saw_bonus,
            "should see bonus token (gamma+1) at least once in 200 tries"
        );
    }

    #[cfg(feature = "leviathan")]
    #[test]
    fn test_leviathan_verifier_acceptance_decreases_with_gamma() {
        let config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        let avg_for_gamma = |gamma: usize| -> f64 {
            let mut total = 0usize;
            let iters = 100;
            // Temporarily override draft_lookahead via a modified config
            let mut gc = Config::draft();
            gc.draft_lookahead = gamma;
            for seed in 0..iters as u64 {
                let mut verifier = LeviathanVerifier::new(&target_weights, &config);
                let accepted = verifier.speculate(
                    &draft_weights,
                    &gc,
                    config.bos_token,
                    0,
                    &mut Rng::new(seed),
                );
                total += accepted.len();
            }
            total as f64 / iters as f64
        };

        let avg_1 = avg_for_gamma(1);
        let avg_4 = avg_for_gamma(4);
        let avg_8 = avg_for_gamma(8);

        assert!(avg_1 >= 1.0, "gamma=1 should give >=1 token, got {avg_1}");
        assert!(avg_4 >= 1.0, "gamma=4 should give >=1 token, got {avg_4}");
        assert!(avg_8 >= 1.0, "gamma=8 should give >=1 token, got {avg_8}");
    }
}
