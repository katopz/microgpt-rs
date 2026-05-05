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
}
