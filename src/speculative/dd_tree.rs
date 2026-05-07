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
/// Equivalent to `build_dd_tree_pruned` with `NoPruner` and `chain_seed=false`.
pub fn build_dd_tree(marginals: &[Vec<f32>], config: &crate::types::Config) -> Vec<TreeNode> {
    build_dd_tree_pruned(marginals, config, &NoPruner, false)
}

/// DDTree with Constraint Pruner: Build verification tree from marginals,
/// filtering branches through a deterministic rules engine.
///
/// When `chain_seed=true`, builds a greedy chain backbone first (argmax at
/// each depth with cumulative log-prob scores), then seeds the best-first
/// heap with siblings at each chain depth and children of the last chain
/// node. Standard best-first expansion fills the remaining budget.
///
/// When `chain_seed=false`, uses the original best-first algorithm.
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
    chain_seed: bool,
) -> Vec<TreeNode> {
    if marginals.is_empty() {
        return Vec::new();
    }

    let mut tree = Vec::with_capacity(config.tree_budget);
    let mut heap = BinaryHeap::new();

    if chain_seed {
        // ── Phase A: Build greedy chain backbone ──────────────
        let mut chain_nodes: Vec<TreeNode> = Vec::new();
        let mut cumulative_score: f32 = 0.0;
        let mut parent_path: u64 = 0;
        let mut chain_parent_tokens: Vec<usize> = Vec::new();

        for (depth, marginal) in marginals.iter().enumerate() {
            if tree.len() >= config.tree_budget {
                break;
            }

            // Find argmax token at this depth
            let best_token = marginal
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i);

            let Some(token_idx) = best_token else {
                break;
            };
            let prob = marginal[token_idx];

            // Chain breaks if argmax has zero prob or is pruned
            if prob <= 0.0 || !pruner.is_valid(depth, token_idx, &chain_parent_tokens) {
                break;
            }

            cumulative_score += prob.ln();
            let node_path = if depth == 0 {
                token_idx as u64
            } else {
                (parent_path << 5) | (token_idx as u64)
            };

            let node = TreeNode {
                score: cumulative_score,
                depth,
                token_idx,
                parent_path: node_path,
            };

            tree.push(node);
            chain_nodes.push(node);
            parent_path = node_path;
            chain_parent_tokens.push(token_idx);
        }

        // ── Phase B: Seed heap with siblings + last chain children ──
        if chain_nodes.is_empty() {
            // No chain built — fall back to original root seeding
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
        } else {
            // Seed siblings at each chain depth
            for chain_node in &chain_nodes {
                let depth = chain_node.depth;
                let parent_chain_score = if depth == 0 {
                    0.0f32
                } else {
                    chain_nodes[depth - 1].score
                };

                // Parent tokens for pruning: chain tokens at depths 0..depth-1
                let sibling_parent_tokens =
                    extract_parent_tokens(chain_node.parent_path >> 5, depth);

                for (i, &prob) in marginals[depth].iter().enumerate() {
                    if i == chain_node.token_idx {
                        continue;
                    }
                    if prob > 0.0 && pruner.is_valid(depth, i, &sibling_parent_tokens) {
                        let sibling_path = if depth == 0 {
                            i as u64
                        } else {
                            let ancestor_path = chain_node.parent_path >> 5;
                            (ancestor_path << 5) | (i as u64)
                        };

                        heap.push(TreeNode {
                            score: parent_chain_score + prob.ln(),
                            depth,
                            token_idx: i,
                            parent_path: sibling_path,
                        });
                    }
                }
            }

            // Seed children of the last chain node
            let last = chain_nodes.last().unwrap();
            if last.depth + 1 < marginals.len() {
                let next_depth = last.depth + 1;
                let parent_tokens = extract_parent_tokens(last.parent_path, last.depth + 1);
                for (i, &prob) in marginals[next_depth].iter().enumerate() {
                    if prob > 0.0 && pruner.is_valid(next_depth, i, &parent_tokens) {
                        heap.push(TreeNode {
                            score: last.score + prob.ln(),
                            depth: next_depth,
                            token_idx: i,
                            parent_path: (last.parent_path << 5) | (i as u64),
                        });
                    }
                }
            }
        }
    } else {
        // Original behavior: seed heap with root's children, filtered by pruner
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
    }

    // ── Phase C: Standard best-first expansion ────────────────
    while tree.len() < config.tree_budget {
        let Some(best) = heap.pop() else {
            break;
        };
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

    // ── Original DDTree Tests ─────────────────────────────────

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
        let tree_pruned = build_dd_tree_pruned(&marginals, &config, &NoPruner, false);

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
        let tree = build_dd_tree_pruned(&[], &config, &pruner, false);
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

    // ── Chain-Seed DDTree Tests ───────────────────────────────

    /// Create marginals with known argmax at each depth for deterministic testing.
    fn make_chain_marginals(config: &Config) -> Vec<Vec<f32>> {
        let mut m0 = vec![0.01; config.vocab_size];
        m0[5] = 0.9;
        let mut m1 = vec![0.01; config.vocab_size];
        m1[10] = 0.85;
        let mut m2 = vec![0.01; config.vocab_size];
        m2[3] = 0.8;
        vec![m0, m1, m2]
    }

    #[test]
    fn test_chain_seed_produces_chain_path() {
        let config = Config::draft();
        let marginals = make_chain_marginals(&config);

        let tree = build_dd_tree_pruned(&marginals, &config, &NoPruner, true);

        // Chain nodes are the first 3 entries (depths 0, 1, 2)
        assert!(
            tree.len() >= 3,
            "tree should have at least 3 chain nodes, got {}",
            tree.len()
        );

        // Verify chain nodes form contiguous path with argmax tokens
        assert_eq!(tree[0].depth, 0, "first chain node at depth 0");
        assert_eq!(tree[0].token_idx, 5, "chain node depth 0 = argmax token 5");

        assert_eq!(tree[1].depth, 1, "second chain node at depth 1");
        assert_eq!(
            tree[1].token_idx, 10,
            "chain node depth 1 = argmax token 10"
        );

        assert_eq!(tree[2].depth, 2, "third chain node at depth 2");
        assert_eq!(tree[2].token_idx, 3, "chain node depth 2 = argmax token 3");

        // Verify chain node parent_paths form contiguous path
        assert_eq!(tree[0].parent_path, 5, "depth 0 parent_path = token 5");
        assert_eq!(
            tree[1].parent_path,
            (5u64 << 5) | 10,
            "depth 1 parent_path = [5, 10]"
        );
        assert_eq!(
            tree[2].parent_path,
            ((5u64 << 5) | 10) << 5 | 3,
            "depth 2 parent_path = [5, 10, 3]"
        );

        // Verify cumulative scores
        let expected_d0 = marginals[0][5].ln();
        let expected_d1 = expected_d0 + marginals[1][10].ln();
        let expected_d2 = expected_d1 + marginals[2][3].ln();

        assert!(
            (tree[0].score - expected_d0).abs() < 1e-5,
            "depth 0 score: expected {expected_d0}, got {}",
            tree[0].score
        );
        assert!(
            (tree[1].score - expected_d1).abs() < 1e-5,
            "depth 1 score: expected {expected_d1}, got {}",
            tree[1].score
        );
        assert!(
            (tree[2].score - expected_d2).abs() < 1e-5,
            "depth 2 score: expected {expected_d2}, got {}",
            tree[2].score
        );
    }

    #[test]
    fn test_chain_seed_false_matches_original() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);

        // build_dd_tree calls build_dd_tree_pruned with chain_seed=false
        let tree_via_wrapper = build_dd_tree(&marginals, &config);
        let tree_via_pruned = build_dd_tree_pruned(&marginals, &config, &NoPruner, false);

        assert_eq!(
            tree_via_wrapper.len(),
            tree_via_pruned.len(),
            "both should produce same number of nodes"
        );
        for (a, b) in tree_via_wrapper.iter().zip(tree_via_pruned.iter()) {
            assert_eq!(a.score, b.score, "scores should match");
            assert_eq!(a.token_idx, b.token_idx, "tokens should match");
            assert_eq!(a.depth, b.depth, "depths should match");
            assert_eq!(a.parent_path, b.parent_path, "parent_paths should match");
        }
    }

    #[test]
    fn test_chain_seed_respects_budget() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);

        let tree = build_dd_tree_pruned(&marginals, &config, &NoPruner, true);

        assert!(
            tree.len() <= config.tree_budget,
            "chain-seed tree size {} exceeds budget {}",
            tree.len(),
            config.tree_budget
        );
        assert!(!tree.is_empty(), "tree should have at least one node");
    }

    /// Pruner that blocks a specific token at a specific depth.
    struct BlockTokenPruner {
        depth: usize,
        token: usize,
    }

    impl ConstraintPruner for BlockTokenPruner {
        fn is_valid(&self, depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            !(depth == self.depth && token_idx == self.token)
        }
    }

    #[test]
    fn test_chain_seed_with_pruner() {
        let config = Config::draft();
        let marginals = make_chain_marginals(&config);

        // Block token 10 at depth 1 (the argmax) — chain should break there
        let pruner = BlockTokenPruner {
            depth: 1,
            token: 10,
        };
        let tree = build_dd_tree_pruned(&marginals, &config, &pruner, true);

        // Chain should have only depth 0 (broke at depth 1)
        assert!(
            !tree.is_empty(),
            "tree should have at least the depth 0 chain node"
        );
        assert_eq!(
            tree[0].token_idx, 5,
            "depth 0 chain node should be argmax token 5"
        );
        assert_eq!(tree[0].depth, 0);

        // No node at depth 1 should have token 10 (blocked)
        for node in &tree {
            if node.depth == 1 {
                assert_ne!(
                    node.token_idx, 10,
                    "blocked token 10 should not appear at depth 1"
                );
            }
        }

        // Verify tree still contains valid nodes (siblings and best-first)
        assert!(
            tree.len() > 1,
            "tree should have more than just the chain node (siblings/best-first)"
        );
    }

    #[test]
    fn test_chain_seed_empty_marginals() {
        let config = Config::draft();
        let tree = build_dd_tree_pruned(&[], &config, &NoPruner, true);
        assert!(
            tree.is_empty(),
            "empty marginals should produce empty tree with chain_seed=true"
        );
    }
}
