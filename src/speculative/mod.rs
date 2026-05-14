pub mod dd_tree;
pub mod dflash;
pub mod prefill;
pub mod sampling;
pub mod step;
pub mod types;
pub mod verifier;

#[cfg(feature = "ppot")]
pub mod ppot;

#[cfg(feature = "bandit")]
pub mod flow_pruner;

// Re-exports — preserves existing import paths like `speculative::build_dd_tree`
pub use dd_tree::{
    TreeBuilder, build_dd_tree, build_dd_tree_balanced, build_dd_tree_pruned,
    build_dd_tree_screened, build_inference_result, extract_best_path, extract_best_path_into,
    extract_parent_tokens, merge_retrieved_branches,
};
pub use dflash::{
    dflash_predict, dflash_predict_ar, dflash_predict_ar_with, dflash_predict_conditioned,
    dflash_predict_conditioned_with, dflash_predict_parallel, dflash_predict_with,
};
pub use prefill::{
    AttentionScorer, BlockAttentionScorer, PrefillScorer, RandomScorer, UniformScorer,
    block_select, block_select_grid, compress_prompt, compress_prompt_blocks, should_compress,
    speculative_prefill, speculative_prefill_adaptive, speculative_prefill_block,
};
pub use sampling::{
    sample_from_distribution, sample_residual_distribution, sample_residual_distribution_into,
};
pub use step::{speculative_step, speculative_step_verifier};
pub use types::{
    BinaryScreeningPruner, BlockScores, ConstraintPruner, DDTreeBranchCache, DraftEvent,
    DraftResult, FlashPrefillConfig, NoPruner, NoScreeningPruner, PrefillMode, RejectionReason,
    ScreeningPruner, SpeculativeContext, TreeNode,
};
pub use verifier::{SimulatedVerifier, SpeculativeVerifier};

pub use verifier::LeviathanVerifier;

pub use step::{
    speculative_step_conditioned, speculative_step_conditioned_with, speculative_step_rollback,
    speculative_step_rollback_with,
};

#[cfg(feature = "bandit")]
pub use flow_pruner::FlowPruner;

#[cfg(feature = "sudoku")]
pub use crate::pruners::SudokuPruner;

// ── PPoT Re-exports (Plan 026 + 027) ──────────────────────────
#[cfg(feature = "ppot")]
pub use ppot::{
    ErrorKind, PpotConfig, RejectionInsight, SessionKnowledge, TokenRule,
    identify_high_entropy_positions, identify_high_entropy_positions_into,
    identify_positions_adaptive, identify_positions_adaptive_into, identify_positions_by_rule,
    identify_positions_by_rule_into, ppot_resample, ppot_resample_different_value,
    ppot_resample_multi_strategy, ppot_resample_with_support, ppot_rescue, ppot_rescue_adaptive,
    rank_by_consistency, rank_by_consistency_weighted, select_best_variant, token_entropy,
};
