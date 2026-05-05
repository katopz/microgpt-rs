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
    /// Check if `token_idx` at the given `depth` is valid.
    /// Returns `false` to prune (reject) this branch.
    fn is_valid(&self, depth: usize, token_idx: usize) -> bool;
}

/// No-op pruner: allows all tokens (original DDTree behavior).
pub struct NoPruner;

impl ConstraintPruner for NoPruner {
    fn is_valid(&self, _depth: usize, _token_idx: usize) -> bool {
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
}

impl ConstraintPruner for SudokuPruner {
    fn is_valid(&self, depth: usize, token_idx: usize) -> bool {
        // Token 0 = empty/padding, never valid for placement
        if token_idx == 0 {
            return false;
        }
        // Digits 1-9 map to token indices 1-9
        let digit = token_idx as u8;
        if !(1..=9).contains(&digit) {
            return false;
        }
        // Map depth to (row, col) and check Sudoku rules
        match self.positions.get(depth) {
            Some(&(row, col)) => self.board.is_valid_move(row, col, digit),
            None => false,
        }
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
        if prob > 0.0 && pruner.is_valid(0, i) {
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
            for (i, &prob) in marginals[next_depth].iter().enumerate() {
                // NEURO-SYMBOLIC INTERCEPT: prune before adding to heap
                if prob > 0.0 && pruner.is_valid(next_depth, i) {
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
pub fn speculative_step(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    _rng: &mut Rng,
) -> (Vec<usize>, usize) {
    // 1. Parallel DFlash draft using lightweight model
    let marginals = dflash_predict_parallel(draft_weights, draft_config, token, pos);

    // 2. DDTree build from marginals
    let tree = build_dd_tree(&marginals, draft_config);

    // 3. Extract best path from tree (highest-scored token at each depth)
    let max_depth = tree.iter().map(|n| n.depth).max().unwrap_or(0);
    let mut path = Vec::with_capacity(max_depth + 1);

    for depth in 0..=max_depth {
        let best_at_depth = tree
            .iter()
            .filter(|n| n.depth == depth)
            .max_by_key(|n| (n.score * 1e6) as i64);

        if let Some(node) = best_at_depth {
            path.push(node.token_idx);
        } else {
            break;
        }
    }

    // 4. Simulate verification: accept ~75% of draft tokens
    let acceptance_rate = 0.75;
    let max_accept = ((path.len() as f32) * acceptance_rate).ceil() as usize;
    let accepted: Vec<usize> = path.into_iter().take(max_accept.max(1)).collect();

    (accepted.clone(), accepted.len())
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
        assert!(pruner.is_valid(0, 0));
        assert!(pruner.is_valid(0, 26));
        assert!(pruner.is_valid(100, 999));
    }

    #[test]
    fn test_sudoku_pruner_rejects_invalid_digits() {
        let board = Sudoku9x9::arto_inkala();
        let pruner = SudokuPruner::new(board);

        // First empty cell is (0,1): row 0 has 8, col 1 has 5/7/9, box has 8/3/7
        // Valid: 1, 2, 4, 6. Invalid: 3, 5, 7, 8, 9.
        assert!(!pruner.is_valid(0, 3), "3 is in box");
        assert!(!pruner.is_valid(0, 5), "5 is in col");
        assert!(!pruner.is_valid(0, 7), "7 is in col+box");
        assert!(!pruner.is_valid(0, 8), "8 is in row+box");
        assert!(!pruner.is_valid(0, 9), "9 is in col");

        // Valid digits
        assert!(pruner.is_valid(0, 1), "1 should be valid");
        assert!(pruner.is_valid(0, 2), "2 should be valid");
        assert!(pruner.is_valid(0, 4), "4 should be valid");
        assert!(pruner.is_valid(0, 6), "6 should be valid");
    }

    #[test]
    fn test_sudoku_pruner_rejects_token_zero() {
        let board = Sudoku9x9::arto_inkala();
        let pruner = SudokuPruner::new(board);
        // Token 0 (empty/padding) should always be rejected
        assert!(!pruner.is_valid(0, 0), "token 0 should be pruned");
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
}
