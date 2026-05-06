use std::collections::BinaryHeap;

use super::types::{ConstraintPruner, NoPruner, TreeNode};

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

/// DDTree: Build verification tree from marginals using Best-First Search.
/// Returns tree nodes ordered by score (best first).
///
/// Equivalent to `build_dd_tree_pruned` with `NoPruner`.
pub fn build_dd_tree(marginals: &[Vec<f32>], config: &crate::types::Config) -> Vec<TreeNode> {
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
    config: &crate::types::Config,
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

/// Extract best-scored token at each depth from a DDTree.
pub fn extract_best_path(tree: &[TreeNode]) -> Vec<usize> {
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

/// Inject retrieved token sequences into the DDTree as candidate branches.
///
/// Each retrieved sequence becomes a path with blended score.
/// Score blending: `(1-w) * log(draft_prob) + w * log(similarity)`
///
/// This is a pure computation function — no feature gating needed.
/// The REST feature provides the data; this function processes it.
pub fn merge_retrieved_branches(
    tree: &mut Vec<TreeNode>,
    marginals: &[Vec<f32>],
    config: &crate::types::Config,
    token_sequences: &[Vec<usize>],
    scores: &[f32],
    rest_weight: f32,
) {
    if token_sequences.is_empty() || rest_weight <= 0.0 {
        return;
    }

    let inv_weight = 1.0 - rest_weight;

    for (seq_idx, seq) in token_sequences.iter().enumerate() {
        let similarity = scores.get(seq_idx).copied().unwrap_or(0.0);
        if similarity <= 0.0 {
            continue;
        }

        for (depth, &token_idx) in seq.iter().enumerate() {
            if depth >= marginals.len() {
                break;
            }
            if token_idx >= config.vocab_size {
                break;
            }

            let base_prob = marginals[depth].get(token_idx).copied().unwrap_or(0.0);
            if base_prob <= 0.0 {
                continue;
            }

            let blended = (base_prob.ln() * inv_weight) + (similarity.ln() * rest_weight);

            // Reconstruct parent_path from sequence prefix up to current depth
            let parent_path = seq[..=depth].iter().enumerate().fold(0u64, |acc, (d, &t)| {
                if d == 0 {
                    t as u64
                } else {
                    (acc << 5) | (t as u64)
                }
            });

            tree.push(TreeNode {
                score: blended,
                depth,
                token_idx,
                parent_path,
            });
        }
    }

    // Re-sort by score descending
    tree.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    tree.truncate(config.tree_budget);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::dflash::dflash_predict;
    use crate::transformer::TransformerWeights;
    use crate::types::{Config, Rng};

    fn make_draft() -> (TransformerWeights, Config) {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        (weights, config)
    }

    #[test]
    fn test_extract_parent_tokens_roundtrip() {
        let path_d0 = 3u64;
        let path_d1 = (path_d0 << 5) | 7;
        let path_d2 = (path_d1 << 5) | 1;

        assert_eq!(extract_parent_tokens(path_d0, 1), vec![3]);
        assert_eq!(extract_parent_tokens(path_d1, 2), vec![3, 7]);
        assert_eq!(extract_parent_tokens(path_d2, 3), vec![3, 7, 1]);
        let empty: Vec<usize> = extract_parent_tokens(0, 0);
        assert!(empty.is_empty());
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
    fn test_ddtree_pruned_empty_marginals() {
        let config = Config::draft();
        let pruner = NoPruner;
        let tree = build_dd_tree_pruned(&[], &config, &pruner);
        assert!(tree.is_empty(), "empty marginals should produce empty tree");
    }

    // ── merge_retrieved_branches Tests ─────────────────────────

    #[test]
    fn test_merge_empty_retrieval_noop() {
        let config = Config::draft();
        let marginals = vec![vec![0.5; config.vocab_size]];
        let mut tree = vec![TreeNode {
            score: 1.0,
            depth: 0,
            token_idx: 0,
            parent_path: 0,
        }];
        let original_len = tree.len();

        merge_retrieved_branches(&mut tree, &marginals, &config, &[], &[], 0.5);

        assert_eq!(
            tree.len(),
            original_len,
            "empty retrieval should not change tree"
        );
    }

    #[test]
    fn test_merge_preserves_budget() {
        let config = Config::draft();
        let marginals = vec![vec![0.1; config.vocab_size]; 4];
        let mut tree = build_dd_tree(&marginals, &config);

        // Create many sequences that would exceed budget
        let sequences: Vec<Vec<usize>> = (0..100)
            .map(|i| vec![i % config.vocab_size, (i + 1) % config.vocab_size])
            .collect();
        let scores: Vec<f32> = (0..100).map(|_| 0.9).collect();

        merge_retrieved_branches(&mut tree, &marginals, &config, &sequences, &scores, 0.3);

        assert!(
            tree.len() <= config.tree_budget,
            "tree should not exceed budget, got {}",
            tree.len()
        );
    }

    #[test]
    fn test_merge_sorts_by_score() {
        let config = Config::draft();
        let marginals = vec![vec![0.1; config.vocab_size]; 2];
        let mut tree = Vec::new();

        let sequences = vec![vec![0, 1], vec![2, 3]];
        let scores = vec![0.5, 0.9];

        merge_retrieved_branches(&mut tree, &marginals, &config, &sequences, &scores, 0.5);

        for window in tree.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "tree should be sorted by score descending"
            );
        }
    }

    #[test]
    fn test_merge_with_empty_tree_adds_nodes() {
        let config = Config::draft();
        // Marginals with non-zero prob at specific tokens
        let mut m0 = vec![0.0; config.vocab_size];
        m0[5] = 0.8;
        let mut m1 = vec![0.0; config.vocab_size];
        m1[10] = 0.6;
        let marginals = vec![m0, m1];
        let mut tree = Vec::new();

        let sequences = vec![vec![5, 10]];
        let scores = vec![0.7];

        merge_retrieved_branches(&mut tree, &marginals, &config, &sequences, &scores, 0.3);

        assert_eq!(tree.len(), 2, "should add 2 nodes for 2-depth sequence");
        assert_eq!(tree[0].token_idx, 5, "first node should be token 5");
    }

    #[test]
    fn test_merge_zero_weight_is_noop() {
        let config = Config::draft();
        let marginals = vec![vec![0.5; config.vocab_size]];
        let mut tree = Vec::new();

        let sequences = vec![vec![0]];
        let scores = vec![0.9];

        merge_retrieved_branches(&mut tree, &marginals, &config, &sequences, &scores, 0.0);

        assert!(tree.is_empty(), "zero rest_weight should be no-op");
    }
}
