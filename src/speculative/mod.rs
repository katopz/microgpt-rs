pub mod dd_tree;
pub mod dflash;
pub mod sampling;
pub mod step;
pub mod types;
pub mod verifier;

#[cfg(feature = "sudoku")]
pub mod sudoku_pruner;

// Re-exports — preserves existing import paths like `speculative::build_dd_tree`
pub use dd_tree::{build_dd_tree, build_dd_tree_pruned, extract_best_path, extract_parent_tokens};
pub use dflash::{dflash_predict, dflash_predict_ar, dflash_predict_parallel};
pub use sampling::{sample_from_distribution, sample_residual_distribution};
pub use step::{speculative_step, speculative_step_verifier};
pub use types::{ConstraintPruner, DraftResult, NoPruner, TreeNode};
pub use verifier::{SimulatedVerifier, SpeculativeVerifier};

#[cfg(feature = "leviathan")]
pub use verifier::LeviathanVerifier;

#[cfg(feature = "sudoku")]
pub use sudoku_pruner::SudokuPruner;
