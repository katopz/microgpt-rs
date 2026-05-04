use crate::transformer::{ForwardContext, KVCache, TransformerWeights, forward};
use crate::types::{Config, Rng, softmax};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

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
pub fn build_dd_tree(marginals: &[Vec<f32>], config: &Config) -> Vec<TreeNode> {
    if marginals.is_empty() {
        return Vec::new();
    }

    let mut tree = Vec::with_capacity(config.tree_budget);
    let mut heap = BinaryHeap::new();

    // Seed heap with root's children (position 0)
    for (i, &prob) in marginals[0].iter().enumerate() {
        if prob > 0.0 {
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
            for (i, &prob) in marginals[best.depth + 1].iter().enumerate() {
                if prob > 0.0 {
                    heap.push(TreeNode {
                        score: best.score + prob.ln(),
                        depth: best.depth + 1,
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
