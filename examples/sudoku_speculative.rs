//! Sudoku Speculative Decoding: DDTree + Computable LoRA Pruning
//!
//! Demonstrates the neuro-symbolic intercept in action:
//! - Draft model proposes logits (simulated uniform for Sudoku digits)
//! - ConstraintPruner filters invalid digits before DDTree build
//! - Compare tree size and valid-branch ratio: with vs without pruning
//!
//! Run: cargo run --example sudoku_speculative

use microgpt_rs::percepta::Sudoku9x9;
use microgpt_rs::speculative::{
    ConstraintPruner, SudokuPruner, build_dd_tree, build_dd_tree_pruned,
};
use microgpt_rs::types::Config;

fn main() {
    println!("🧠 Sudoku Speculative Decoding: DDTree + Computable LoRA");
    println!("{}", "═".repeat(60));

    let board = Sudoku9x9::arto_inkala();
    let clues = board.clue_count();
    let empty = 81 - clues;

    println!("\n📝 Arto Inkala Puzzle — {clues} clues, {empty} empty cells\n");
    print!("{}", board.display());

    let pruner = SudokuPruner::new(board.clone());

    // ── 1. Simulate draft model marginals ──────────────────────────
    // In a real system, the draft model would produce these.
    // For Sudoku, we simulate uniform probability over digits 1-9.
    // (vocab_size = 10: index 0 = padding, 1-9 = digits)
    let lookahead = 5usize.min(pruner.empty_count());
    let marginals: Vec<Vec<f32>> = (0..lookahead)
        .map(|depth| {
            let mut probs = vec![0.0f32; 10];
            let valid_count = count_valid_at_depth(&pruner, depth);
            // Uniform over valid digits for a fair simulation
            for d in 1..=9u8 {
                if pruner.is_valid(depth, d as usize) {
                    probs[d as usize] = 1.0 / valid_count as f32;
                }
            }
            probs
        })
        .collect();

    println!("📊 Draft Model Marginals (uniform over valid digits)");
    println!("{}", "─".repeat(60));
    for (depth, probs) in marginals.iter().enumerate() {
        let pos = pruner.position_at(depth).unwrap_or((0, 0));
        let valid_digits: Vec<u8> = (1..=9)
            .filter(|&d| pruner.is_valid(depth, d as usize))
            .collect();
        let total_prob: f32 = probs.iter().sum();
        println!(
            "  Depth {depth}: ({},{}) valid={:?} sum={total_prob:.3}",
            pos.0 + 1,
            pos.1 + 1,
            valid_digits,
        );
    }

    let config = Config {
        tree_budget: 100,
        ..Config::draft()
    };

    // ── 2. Build DDTree WITHOUT pruning ────────────────────────────
    // Use raw marginals (no constraint filtering)
    let raw_marginals: Vec<Vec<f32>> = (0..lookahead)
        .map(|_| {
            let mut probs = vec![0.0f32; 10];
            for d in 1..=9u8 {
                probs[d as usize] = 1.0 / 9.0;
            }
            probs
        })
        .collect();

    let tree_unpruned = build_dd_tree(&raw_marginals, &config);

    // ── 3. Build DDTree WITH Computable LoRA pruning ───────────────
    let tree_pruned = build_dd_tree_pruned(&raw_marginals, &config, &pruner);

    // ── 4. Compare results ─────────────────────────────────────────
    println!("\n📈 DDTree Comparison: Without vs With Pruning");
    println!("{}", "─".repeat(60));

    let unpruned_valid = count_valid_branches(&tree_unpruned, &pruner);
    let pruned_valid = count_valid_branches(&tree_pruned, &pruner);

    let unpruned_valid_pct = if tree_unpruned.is_empty() {
        0.0
    } else {
        unpruned_valid as f64 / tree_unpruned.len() as f64 * 100.0
    };
    let pruned_valid_pct = if tree_pruned.is_empty() {
        0.0
    } else {
        pruned_valid as f64 / tree_pruned.len() as f64 * 100.0
    };

    println!("  {:<25} {:>12} {:>12}", "Metric", "Unpruned", "Pruned");
    println!("{}", "─".repeat(50));
    println!(
        "  {:<25} {:>12} {:>12}",
        "Tree nodes",
        tree_unpruned.len(),
        tree_pruned.len()
    );
    println!(
        "  {:<25} {:>12} {:>12}",
        "Valid branches", unpruned_valid, pruned_valid
    );
    println!(
        "  {:<25} {:>11.1}% {:>11.1}%",
        "Valid ratio", unpruned_valid_pct, pruned_valid_pct
    );
    println!(
        "  {:<25} {:>12} {:>12}",
        "Invalid branches",
        tree_unpruned.len() - unpruned_valid,
        tree_pruned.len() - pruned_valid
    );

    // ── 5. Show token distribution at each depth ───────────────────
    println!("\n🔍 Token Distribution by Depth");
    println!("{}", "─".repeat(60));

    let max_depth_unpruned = tree_unpruned.iter().map(|n| n.depth).max().unwrap_or(0);
    let max_depth_pruned = tree_pruned.iter().map(|n| n.depth).max().unwrap_or(0);
    let max_depth = max_depth_unpruned.max(max_depth_pruned);

    println!(
        "  {:<6} {:<14} {:<14} {:<10} Position",
        "Depth", "Unpruned", "Pruned", "Pruned?"
    );
    println!("{}", "─".repeat(60));

    for depth in 0..=max_depth {
        let mut unpruned_set: Vec<u8> = tree_unpruned
            .iter()
            .filter(|n| n.depth == depth)
            .map(|n| n.token_idx as u8)
            .collect();
        unpruned_set.sort();
        unpruned_set.dedup();

        let mut pruned_set: Vec<u8> = tree_pruned
            .iter()
            .filter(|n| n.depth == depth)
            .map(|n| n.token_idx as u8)
            .collect();
        pruned_set.sort();
        pruned_set.dedup();

        let was_pruned = unpruned_set.len() > pruned_set.len();
        let removed: Vec<u8> = unpruned_set
            .iter()
            .filter(|d| !pruned_set.contains(d))
            .copied()
            .collect();
        let pos = pruner
            .position_at(depth)
            .map(|(r, c)| format!("({},{})", r + 1, c + 1))
            .unwrap_or_else(|| "—".to_string());

        println!(
            "  {depth:<6} {:<14} {:<14} {:<10} {pos}",
            format!("{:?}", unpruned_set),
            format!("{:?}", pruned_set),
            if was_pruned {
                format!("✂️ -{:?}", removed)
            } else {
                "=".to_string()
            },
        );
    }

    // ── 6. Summary ─────────────────────────────────────────────────
    println!("\n✨ Summary");
    println!("{}", "─".repeat(60));

    let reduction =
        (tree_unpruned.len() - tree_pruned.len()) as f64 / tree_unpruned.len() as f64 * 100.0;

    println!(
        "  Pruning removed {} invalid branches ({reduction:.1}% reduction)",
        tree_unpruned.len() - tree_pruned.len()
    );
    println!(
        "  Pruned tree: {}% valid branches (vs {}% unpruned)",
        pruned_valid_pct, unpruned_valid_pct
    );
    println!("  Verification budget saved: tree only explores valid paths");

    if pruned_valid == tree_pruned.len() && !tree_pruned.is_empty() {
        println!("\n  ✅ Computable LoRA guarantees 100% valid placements!");
    }

    println!(
        "\n  Next step: target model verifies only {} branches instead of {}",
        tree_pruned.len(),
        tree_unpruned.len()
    );
}

fn count_valid_at_depth(pruner: &SudokuPruner, depth: usize) -> usize {
    (1..=9)
        .filter(|&d| pruner.is_valid(depth, d as usize))
        .count()
}

fn count_valid_branches(
    tree: &[microgpt_rs::speculative::TreeNode],
    pruner: &SudokuPruner,
) -> usize {
    tree.iter()
        .filter(|node| pruner.is_valid(node.depth, node.token_idx))
        .count()
}
