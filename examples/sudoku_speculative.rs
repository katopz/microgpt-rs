//! Sudoku Speculative Decoding: DDTree + Symbolic Validator Pruning
//!
//! Demonstrates the neuro-symbolic intercept with 3-level comparison:
//! - **Unpruned**: Draft model proposes all high-probability tokens
//! - **Static-Only**: Prunes against initial board, ignores cross-depth conflicts
//! - **Path-Aware**: Prunes against initial board AND parent tokens in same path
//!
//! Shows that path-aware pruning catches cross-depth row/col/box conflicts
//! that static-only pruning misses.
//!
//! Run: cargo run --example sudoku_speculative

use microgpt_rs::percepta::Sudoku9x9;
use microgpt_rs::pruners::SudokuPruner;
use microgpt_rs::speculative::{
    ConstraintPruner, TreeNode, build_dd_tree, build_dd_tree_pruned, extract_parent_tokens,
};
use microgpt_rs::types::Config;

// ── Static-Only Pruner: ignores parent_tokens (depth-0-only validation) ──

/// Wraps `SudokuPruner` but ignores parent path context.
/// Only validates against the initial board state — the "before" state
/// of Plan 002. Used to demonstrate the gap that path-aware pruning fills.
struct StaticOnlyPruner<'a>(&'a SudokuPruner);

impl ConstraintPruner for StaticOnlyPruner<'_> {
    fn is_valid(&self, depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        self.0.is_valid(depth, token_idx, &[])
    }
}

fn main() {
    println!("🧠 Sudoku Speculative Decoding: DDTree + Symbolic Validator");
    println!("{}", "═".repeat(60));

    let board = Sudoku9x9::arto_inkala();
    let clues = board.clue_count();
    let empty = 81 - clues;

    println!("\n📝 Arto Inkala Puzzle — {clues} clues, {empty} empty cells\n");
    print!("{}", board.display());

    let pruner = SudokuPruner::new(board.clone());

    // ── 1. Simulate draft model marginals ──────────────────────────
    // Uniform probability over digits 1-9 (vocab_size = 10, index 0 = padding).
    // Use 8 depths to stress row 0 — all empties in same row cause cross-depth conflicts.
    let lookahead = 8usize.min(pruner.empty_count());

    let raw_marginals: Vec<Vec<f32>> = (0..lookahead)
        .map(|_| {
            let mut probs = vec![0.0f32; 10];
            for d in 1..=9u8 {
                probs[d as usize] = 1.0 / 9.0;
            }
            probs
        })
        .collect();

    println!("📊 Draft Model Marginals (uniform over digits 1-9, {lookahead} depths)");
    println!("{}", "─".repeat(60));
    for depth in 0..lookahead {
        let pos = pruner.position_at(depth).unwrap_or((0, 0));
        let valid_digits: Vec<u8> = (1..=9)
            .filter(|&d| pruner.is_valid(depth, d as usize, &[]))
            .collect();
        println!(
            "  Depth {depth}: ({},{}) static-valid={:?}",
            pos.0 + 1,
            pos.1 + 1,
            valid_digits,
        );
    }

    let config = Config {
        tree_budget: 100,
        ..Config::draft()
    };

    // ── 2. Build 3 DDTree variants ─────────────────────────────────
    let mv: Vec<&[f32]> = raw_marginals.iter().map(|s| s.as_slice()).collect();
    let tree_unpruned = build_dd_tree(&mv, &config);

    let static_pruner = StaticOnlyPruner(&pruner);
    let tree_static = build_dd_tree_pruned(&mv, &config, &static_pruner, false);

    let tree_aware = build_dd_tree_pruned(&mv, &config, &pruner, false);

    // ── 3. Count validity for each tree ────────────────────────────
    // Static validity: valid against initial board only
    let unpruned_static_valid = count_static_valid(&tree_unpruned, &pruner);
    let static_static_valid = count_static_valid(&tree_static, &pruner);
    let aware_static_valid = count_static_valid(&tree_aware, &pruner);

    // Accumulated validity: valid against initial board + all parent tokens in path
    let unpruned_accum_valid = count_accumulated_valid(&tree_unpruned, &pruner);
    let static_accum_valid = count_accumulated_valid(&tree_static, &pruner);
    let aware_accum_valid = count_accumulated_valid(&tree_aware, &pruner);

    // ── 4. Three-column comparison ─────────────────────────────────
    println!("\n📈 DDTree Comparison: Unpruned vs Static-Only vs Path-Aware");
    println!("{}", "─".repeat(70));
    println!(
        "  {:<24} {:>12} {:>12} {:>12}",
        "Metric", "Unpruned", "Static-Only", "Path-Aware"
    );
    println!("{}", "─".repeat(62));

    let pct = |v: usize, total: usize| -> String {
        if total == 0 {
            return "  —".to_string();
        }
        format!("{:.1}%", v as f64 / total as f64 * 100.0)
    };

    println!(
        "  {:<24} {:>12} {:>12} {:>12}",
        "Tree nodes",
        tree_unpruned.len(),
        tree_static.len(),
        tree_aware.len(),
    );
    println!(
        "  {:<24} {:>12} {:>12} {:>12}",
        "Static valid", unpruned_static_valid, static_static_valid, aware_static_valid,
    );
    println!(
        "  {:<24} {:>12} {:>12} {:>12}",
        "Static valid %",
        pct(unpruned_static_valid, tree_unpruned.len()),
        pct(static_static_valid, tree_static.len()),
        pct(aware_static_valid, tree_aware.len()),
    );
    println!(
        "  {:<24} {:>12} {:>12} {:>12}",
        "Accumulated valid", unpruned_accum_valid, static_accum_valid, aware_accum_valid,
    );
    println!(
        "  {:<24} {:>12} {:>12} {:>12}",
        "Accumulated valid %",
        pct(unpruned_accum_valid, tree_unpruned.len()),
        pct(static_accum_valid, tree_static.len()),
        pct(aware_accum_valid, tree_aware.len()),
    );

    // ── 5. Show token distribution by depth ────────────────────────
    println!("\n🔍 Token Distribution by Depth");
    println!("{}", "─".repeat(70));

    let max_depth = tree_unpruned
        .iter()
        .chain(tree_static.iter())
        .chain(tree_aware.iter())
        .map(|n| n.depth)
        .max()
        .unwrap_or(0);

    println!(
        "  {:<6} {:<14} {:<14} {:<14} Position",
        "Depth", "Unpruned", "Static-Only", "Path-Aware"
    );
    println!("{}", "─".repeat(70));

    for depth in 0..=max_depth {
        let unpruned_set = token_set_at_depth(&tree_unpruned, depth);
        let static_set = token_set_at_depth(&tree_static, depth);
        let aware_set = token_set_at_depth(&tree_aware, depth);

        let pos = pruner
            .position_at(depth)
            .map(|(r, c)| format!("({},{})", r + 1, c + 1))
            .unwrap_or_else(|| "—".to_string());

        println!(
            "  {depth:<6} {:<14} {:<14} {:<14} {pos}",
            format!("{:?}", unpruned_set),
            format!("{:?}", static_set),
            format!("{:?}", aware_set),
        );
    }

    // ── 6. Cross-depth conflict examples ───────────────────────────
    println!("\n🔗 Cross-Depth Conflict Detection");
    println!("{}", "─".repeat(60));

    // Find an example: static-only missed it, path-aware caught it
    let static_conflicts = tree_static.len() - static_accum_valid;
    let aware_conflicts = tree_aware.len() - aware_accum_valid;
    let caught_by_path = static_conflicts.saturating_sub(aware_conflicts);

    println!("  Static-only tree has {static_conflicts} nodes with cross-depth conflicts");
    println!("  Path-aware tree has {aware_conflicts} nodes with cross-depth conflicts");
    println!("  Path-aware pruning caught {caught_by_path} additional cross-depth violations");

    if let Some(example) = find_cross_depth_conflict(&tree_static, &pruner) {
        let (depth, token, parent_depth, parent_token, pos, parent_pos) = example;
        println!(
            "\n  Example: depth {depth} places digit {token} at ({},{}), \
             but depth {parent_depth} already placed digit {parent_token} at ({},{})",
            pos.0 + 1,
            pos.1 + 1,
            parent_pos.0 + 1,
            parent_pos.1 + 1,
        );
        let same_row = pos.0 == parent_pos.0;
        let same_col = pos.1 == parent_pos.1;
        let same_box = pos.0 / 3 == parent_pos.0 / 3 && pos.1 / 3 == parent_pos.1 / 3;
        if same_row {
            println!("  → Same row {}!", pos.0 + 1);
        }
        if same_col {
            println!("  → Same column {}!", pos.1 + 1);
        }
        if same_box && !same_row && !same_col {
            println!("  → Same 3×3 box!");
        }
    }

    // ── 7. Summary ─────────────────────────────────────────────────
    println!("\n✨ Summary");
    println!("{}", "─".repeat(60));

    println!(
        "  Unpruned:    {} tree nodes, {:>5} accumulated-valid ({})",
        tree_unpruned.len(),
        unpruned_accum_valid,
        pct(unpruned_accum_valid, tree_unpruned.len()),
    );
    println!(
        "  Static-Only: {} tree nodes, {:>5} accumulated-valid ({})",
        tree_static.len(),
        static_accum_valid,
        pct(static_accum_valid, tree_static.len()),
    );
    println!(
        "  Path-Aware:  {} tree nodes, {:>5} accumulated-valid ({})",
        tree_aware.len(),
        aware_accum_valid,
        pct(aware_accum_valid, tree_aware.len()),
    );

    if aware_accum_valid == tree_aware.len() && !tree_aware.is_empty() {
        println!("\n  ✅ Path-Aware pruning guarantees 100% accumulated validity!");
    }

    if caught_by_path > 0 {
        println!(
            "  🔗 Path awareness caught {caught_by_path} cross-depth conflicts \
             that static-only missed"
        );
    }

    println!(
        "\n  Target model verifies only {} branches (path-aware) instead of {} (unpruned)",
        tree_aware.len(),
        tree_unpruned.len(),
    );
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Count nodes valid against initial board state (static check).
fn count_static_valid(tree: &[TreeNode], pruner: &SudokuPruner) -> usize {
    tree.iter()
        .filter(|node| pruner.is_valid(node.depth, node.token_idx, &[]))
        .count()
}

/// Count nodes valid against accumulated board state (initial + parent placements).
fn count_accumulated_valid(tree: &[TreeNode], pruner: &SudokuPruner) -> usize {
    let mut valid = 0;
    for node in tree {
        // parent_path includes node's own token, extract depth+1 then take first depth
        let all_tokens = extract_parent_tokens(node.parent_path, node.depth + 1);
        let parent_tokens = &all_tokens[..node.depth];

        // Build accumulated board: initial + all parent placements
        let mut board = pruner.board().clone();
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
            && board.is_valid_move(row, col, node.token_idx as u8)
        {
            valid += 1;
        }
    }
    valid
}

/// Get sorted, deduplicated token indices at a given depth.
fn token_set_at_depth(tree: &[TreeNode], depth: usize) -> Vec<u8> {
    let mut tokens: Vec<u8> = tree
        .iter()
        .filter(|n| n.depth == depth)
        .map(|n| n.token_idx as u8)
        .collect();
    tokens.sort();
    tokens.dedup();
    tokens
}

/// Conflict details: (depth, token, conflict_depth, conflict_token, pos, conflict_pos).
type ConflictDetails = (usize, usize, usize, usize, (usize, usize), (usize, usize));

/// Find first cross-depth conflict in a tree (for demonstration).
fn find_cross_depth_conflict(tree: &[TreeNode], pruner: &SudokuPruner) -> Option<ConflictDetails> {
    for node in tree {
        let all_tokens = extract_parent_tokens(node.parent_path, node.depth + 1);
        let parent_tokens = &all_tokens[..node.depth];

        let mut board = pruner.board().clone();
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
            // Find which parent caused the conflict
            let digit = node.token_idx as u8;
            for (pd, &pt) in parent_tokens.iter().enumerate() {
                if pt == 0 || pt as u8 != digit {
                    continue;
                }
                if let Some(ppos) = pruner.position_at(pd) {
                    let same_row = ppos.0 == row;
                    let same_col = ppos.1 == col;
                    let same_box = ppos.0 / 3 == row / 3 && ppos.1 / 3 == col / 3;
                    if same_row || same_col || same_box {
                        return Some((node.depth, node.token_idx, pd, pt, (row, col), ppos));
                    }
                }
            }
        }
    }
    None
}
