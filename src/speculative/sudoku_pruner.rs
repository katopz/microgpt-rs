//! Sudoku constraint pruner — maps DDTree depth to (row, col) and validates
//! drafted digits against Sudoku rules. Gated behind `sudoku` feature.

use crate::percepta::Sudoku9x9;
use crate::speculative::types::ConstraintPruner;

/// Sudoku constraint pruner: maps DDTree depth → (row, col) and
/// validates each drafted digit (token_idx 1-9) against Sudoku rules.
///
/// This is the bridge between speculative decoding and Computable LoRA:
/// - Draft model proposes logits for each empty cell
/// - SudokuPruner rejects digits that violate row/col/box constraints
/// - Only valid digits enter the DDTree → 100% valid placements
#[cfg(feature = "sudoku")]
pub struct SudokuPruner {
    /// The current board state (0 = empty).
    board: Sudoku9x9,
    /// Ordered list of (row, col) positions that map to DDTree depths.
    /// Depth 0 → positions[0], Depth 1 → positions[1], etc.
    positions: Vec<(usize, usize)>,
}

#[cfg(feature = "sudoku")]
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

#[cfg(feature = "sudoku")]
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

#[cfg(all(test, feature = "sudoku"))]
mod tests {
    use super::*;

    fn make_board() -> Sudoku9x9 {
        Sudoku9x9::arto_inkala()
    }

    #[test]
    fn test_sudoku_pruner_rejects_invalid_digits() {
        let board = make_board();
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
        let board = make_board();
        let pruner = SudokuPruner::new(board);
        assert!(!pruner.is_valid(0, 0, &[]), "token 0 should be pruned");
    }

    #[test]
    fn test_sudoku_pruner_empty_count() {
        let board = make_board();
        let pruner = SudokuPruner::new(board);
        assert_eq!(pruner.empty_count(), 60, "Arto Inkala has 60 empty cells");
    }

    #[test]
    fn test_sudoku_pruner_positions_match_empties() {
        let board = make_board();
        let pruner = SudokuPruner::new(board);

        // First empty cell should be (0,1)
        assert_eq!(pruner.position_at(0), Some((0, 1)));
        // Depth beyond empty_count should return None
        assert_eq!(pruner.position_at(60), None);
    }

    #[test]
    fn test_sudoku_pruner_path_aware_same_row() {
        let board = make_board();
        let pruner = SudokuPruner::new(board);

        // Depth 0 → (0,1), depth 1 → (0,2): both in row 0
        assert!(
            pruner.is_valid(0, 4, &[]),
            "digit 4 at depth 0 should be valid alone"
        );
        assert!(
            pruner.is_valid(1, 4, &[]),
            "digit 4 at depth 1 should be valid alone"
        );
        assert!(
            !pruner.is_valid(1, 4, &[4]),
            "same digit 4 in same row should be pruned"
        );
    }

    #[test]
    fn test_sudoku_pruner_path_aware_same_col() {
        let board = make_board();
        let pruner = SudokuPruner::new(board);

        // Depth 0 → (0,1), depth 9 → (1,1): both in column 1
        assert!(
            pruner.is_valid(0, 1, &[]),
            "digit 1 at depth 0 should be valid alone"
        );
        assert!(
            pruner.is_valid(9, 1, &[]),
            "digit 1 at depth 9 should be valid alone"
        );
        let mut parent_tokens = vec![0usize; 9];
        parent_tokens[0] = 1;
        assert!(
            !pruner.is_valid(9, 1, &parent_tokens),
            "same digit 1 in same column should be pruned"
        );
    }

    #[test]
    fn test_sudoku_pruner_path_aware_same_box() {
        let board = make_board();
        let pruner = SudokuPruner::new(board);

        // Depth 0 → (0,1) box(0,0), depth 1 → (0,2) box(0,0): same 3×3 box
        assert!(
            pruner.is_valid(0, 6, &[]),
            "digit 6 at depth 0 should be valid alone"
        );
        assert!(
            pruner.is_valid(1, 6, &[]),
            "digit 6 at depth 1 should be valid alone"
        );
        assert!(
            !pruner.is_valid(1, 6, &[6]),
            "same digit 6 in same box should be pruned"
        );
    }

    #[test]
    fn test_sudoku_pruner_path_aware_no_conflict_different_digit() {
        let board = make_board();
        let pruner = SudokuPruner::new(board);

        // Different digits NEVER conflict, even in same row
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
        let board = make_board();
        let pruner = SudokuPruner::new(board);

        // Depth 0 → (0,1) row 0, col 1, box(0,0)
        // Depth 21 → (3,0) row 3, col 0, box(3,0)
        assert!(
            pruner.is_valid(0, 4, &[]),
            "digit 4 at (0,1) should be valid"
        );
        assert!(
            pruner.is_valid(21, 4, &[]),
            "digit 4 at (3,0) should be valid"
        );

        let mut parent_tokens = vec![0usize; 21];
        parent_tokens[0] = 4;
        assert!(
            pruner.is_valid(21, 4, &parent_tokens),
            "same digit in different row/col/box should NOT be pruned"
        );
    }

    #[test]
    fn test_sudoku_pruner_path_aware_multi_level_conflict() {
        let board = make_board();
        let pruner = SudokuPruner::new(board);

        // Multi-level path: [1, 2, 3] at depths 0, 1, 2
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

    #[test]
    fn test_ddtree_pruned_sudoku_reduces_tree_size() {
        use crate::speculative::dd_tree::{build_dd_tree, build_dd_tree_pruned};
        use crate::types::Config;

        let marginals: Vec<Vec<f32>> = vec![{
            let mut probs = vec![0.0f32; 10];
            for d in 1..=9u8 {
                probs[d as usize] = 1.0 / 9.0;
            }
            probs
        }];

        let config = Config {
            tree_budget: 20,
            ..Config::draft()
        };

        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let tree_unpruned = build_dd_tree(&mv, &config);
        let tree_pruned =
            build_dd_tree_pruned(&mv, &config, &SudokuPruner::new(make_board()), false);

        assert!(
            tree_pruned.len() < tree_unpruned.len(),
            "pruned tree ({}) should be smaller than unpruned ({})",
            tree_pruned.len(),
            tree_unpruned.len()
        );
        assert!(!tree_pruned.is_empty(), "pruned tree should have nodes");
        assert_eq!(tree_unpruned.len(), 9, "unpruned should have 9 nodes");
        assert_eq!(tree_pruned.len(), 4, "pruned should have 4 valid nodes");
    }

    #[test]
    fn test_ddtree_pruned_sudoku_only_valid_tokens() {
        use crate::speculative::dd_tree::build_dd_tree_pruned;
        use crate::types::Config;

        let board = make_board();
        let pruner = SudokuPruner::new(board.clone());

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
            tree_budget: 100,
            ..Config::draft()
        };

        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let tree = build_dd_tree_pruned(&mv, &config, &pruner, false);

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
        use crate::speculative::dd_tree::build_dd_tree_pruned;
        use crate::types::Config;

        let board = make_board();
        let pruner = SudokuPruner::new(board);

        let marginals: Vec<Vec<f32>> = (0..5)
            .map(|_| {
                let mut probs = vec![0.5f32; 10];
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

        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let tree = build_dd_tree_pruned(&mv, &config, &pruner, false);

        for node in &tree {
            assert_ne!(
                node.token_idx, 0,
                "token 0 should be pruned at depth {}",
                node.depth
            );
        }
    }

    /// Wrapper that ignores parent_tokens for static-only comparison testing.
    struct StaticOnlyPruner<'a>(&'a SudokuPruner);

    impl ConstraintPruner for StaticOnlyPruner<'_> {
        fn is_valid(&self, depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            self.0.is_valid(depth, token_idx, &[])
        }
    }

    /// Verify every node in the tree is valid against its accumulated board state.
    fn count_invalid_accumulated(
        pruner: &SudokuPruner,
        tree: &[crate::speculative::types::TreeNode],
    ) -> usize {
        use crate::speculative::dd_tree::extract_parent_tokens;

        let mut invalid = 0;
        for node in tree {
            let all_tokens = extract_parent_tokens(node.parent_path, node.depth + 1);
            let parent_tokens = &all_tokens[..node.depth];

            let mut board = pruner.board.clone();
            for (depth, &token) in parent_tokens.iter().enumerate() {
                if token == 0 {
                    continue;
                }
                if let Some((row, col)) = pruner.position_at(depth) {
                    board.grid[row][col] = token as u8;
                }
            }

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
        use crate::speculative::dd_tree::build_dd_tree_pruned;
        use crate::types::Config;

        let board = make_board();
        let pruner = SudokuPruner::new(board);

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
            tree_budget: 50,
            ..Config::draft()
        };

        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let tree = build_dd_tree_pruned(&mv, &config, &pruner, false);
        assert!(!tree.is_empty(), "tree should have nodes");

        let invalid = count_invalid_accumulated(&pruner, &tree);
        assert_eq!(
            invalid, 0,
            "path-aware tree should have 0 invalid accumulated nodes, found {invalid}"
        );
    }

    #[test]
    fn test_ddtree_path_aware_catches_cross_depth_conflicts() {
        use crate::speculative::dd_tree::build_dd_tree_pruned;
        use crate::types::Config;

        let board = make_board();
        let pruner = SudokuPruner::new(board);

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

        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let static_pruner = StaticOnlyPruner(&pruner);
        let tree_static = build_dd_tree_pruned(&mv, &config, &static_pruner, false);
        let tree_aware = build_dd_tree_pruned(&mv, &config, &pruner, false);

        let static_invalid = count_invalid_accumulated(&pruner, &tree_static);
        assert!(
            static_invalid > 0,
            "static tree should have cross-depth conflicts (found {static_invalid})"
        );

        let aware_invalid = count_invalid_accumulated(&pruner, &tree_aware);
        assert_eq!(
            aware_invalid, 0,
            "path-aware tree should have zero cross-depth conflicts"
        );
    }
}
