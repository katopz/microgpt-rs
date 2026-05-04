use crate::transformer::{KVCache, TransformerWeights, forward};
use crate::types::{Config, Rng, softmax};
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

/// DFlash: Predict marginal distributions for L lookahead positions.
/// Each position uses an independent forward pass (simulating block-parallel prediction).
pub fn dflash_predict(
    weights: &TransformerWeights,
    token: usize,
    pos: usize,
    config: &Config,
) -> Vec<Vec<f32>> {
    let mut marginals = Vec::with_capacity(config.draft_lookahead);

    for step in 0..config.draft_lookahead {
        let draft_pos = pos + step;
        if draft_pos >= config.block_size {
            break;
        }
        let mut draft_cache = KVCache::new(config);
        let logits = forward(weights, &mut draft_cache, token, draft_pos, config);
        let mut probs = logits;
        for logit in probs.iter_mut() {
            *logit /= config.temperature;
        }
        softmax(&mut probs);
        marginals.push(probs);
    }

    marginals
}

/// DDTree: Build verification tree from marginals using Best-First Search.
/// Returns tree nodes ordered by score (best first).
pub fn build_dd_tree(marginals: &[Vec<f32>], config: &Config) -> Vec<TreeNode> {
    if marginals.is_empty() {
        return Vec::new();
    }

    let mut tree = Vec::new();
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

/// One step of speculative decoding.
/// Returns (accepted_token_ids, acceptance_length).
pub fn speculative_step(
    weights: &TransformerWeights,
    token: usize,
    pos: usize,
    config: &Config,
    _rng: &mut Rng,
) -> (Vec<usize>, usize) {
    // 1. DFlash draft: predict L marginal distributions
    let marginals = dflash_predict(weights, token, pos, config);

    // 2. DDTree build: construct candidate tree
    let tree = build_dd_tree(&marginals, config);

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

    // 4. Simulate verification: accept ~70% of draft tokens
    let acceptance_rate = 0.7;
    let max_accept = ((path.len() as f32) * acceptance_rate).ceil() as usize;
    let accepted: Vec<usize> = path.into_iter().take(max_accept.max(1)).collect();

    (accepted.clone(), accepted.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dflash_produces_marginals() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let marginals = dflash_predict(&weights, 0, 0, &config);
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
    fn test_ddtree_respects_budget() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let marginals = dflash_predict(&weights, 0, 0, &config);
        let tree = build_dd_tree(&marginals, &config);

        assert!(
            tree.len() <= config.tree_budget,
            "tree size {} exceeds budget {}",
            tree.len(),
            config.tree_budget
        );
    }

    #[test]
    fn test_ddtree_scores_descending() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let marginals = dflash_predict(&weights, 0, 0, &config);
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
    fn test_speculative_step() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let (accepted, accept_len) = speculative_step(&weights, 0, 0, &config, &mut rng);

        assert!(!accepted.is_empty(), "should accept at least 1 token");
        assert!(accept_len >= 1);
        for &t in &accepted {
            assert!(t < config.vocab_size, "token {t} out of range");
        }
    }
}
