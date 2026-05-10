//! 9×9 Sudoku Example: Streaming "Thinking" Output
//!
//! Demonstrates the Symbolic Validator concept:
//! - Deterministic rules engine prunes LLM hallucinations
//! - O(log N) attention retrieves execution state via convex hull
//! - Streaming output shows step-by-step constraint satisfaction
//!
//! Run: cargo run --example sudoku_01_9x9

use microgpt_rs::percepta::{StreamingSolver, Sudoku9x9, SymbolicValidator, Vec2};

fn main() {
    println!("🧠 Symbolic Validator: 9×9 Sudoku Solver");
    println!("{}", "═".repeat(60));

    // ── 1. Load the Arto Inkala puzzle ────────────────────────────
    let puzzle = Sudoku9x9::arto_inkala();
    let clues = puzzle.clue_count();

    println!("\n📝 Puzzle loaded — {clues} clues given.\n");
    print!("{}", puzzle.display());

    // ── 2. Symbolic Validator intercept demo ─────────────────────────
    println!("\n⚡ Symbolic Validator Intercept Demo");
    println!("{}", "─".repeat(60));

    // Simulate: fast draft model proposes logits for cell (0,1)
    // The LLM "guesses" based on semantic probability
    let llm_drafts: Vec<(u8, f32)> = vec![
        (3, -0.1), // LLM is confident about 3
        (5, -0.5), // Thinks 5 is possible
        (8, -0.8), // Maybe 8?
        (2, -1.2), // Wild guess
        (7, -1.5), // Another guess
    ];

    let row = 0;
    let col = 1;

    println!("  LLM proposes for ({}, {}):", row + 1, col + 1);
    for (digit, prob) in &llm_drafts {
        println!("    Digit {digit}: log_prob={prob:.2}");
    }

    // Symbolic Validator prunes invalid moves deterministically
    let valid = SymbolicValidator::prune_drafts(&puzzle, row, col, &llm_drafts);

    println!("\n  After Symbolic Validator pruning:");
    for (digit, prob) in &valid {
        println!("    ✅ Digit {digit}: log_prob={prob:.2}");
    }

    // Show which were pruned
    let valid_digits: Vec<u8> = valid.iter().map(|(d, _)| *d).collect();
    let pruned: Vec<(u8, f32)> = llm_drafts
        .iter()
        .filter(|(d, _)| !valid_digits.contains(d))
        .copied()
        .collect();

    if !pruned.is_empty() {
        println!("\n  Pruned by rules engine:");
        for (digit, prob) in &pruned {
            let reason = violation_reason(&puzzle, row, col, *digit);
            println!("    ❌ Digit {digit}: log_prob={prob:.2} — {reason}");
        }
    }

    // ── 3. Solve with streaming "thinking" output ─────────────────
    println!("\n\n🔍 Depth-First Exploration");
    println!("{}", "─".repeat(60));

    let mut solver = StreamingSolver::new(Sudoku9x9::arto_inkala().grid);
    let solved = solver.solve_streaming();

    // Print the streaming events
    print!("{}", solver.format_events());

    if !solved {
        println!("\n  ❌ No solution found.");
        return;
    }

    // ── 4. Show solved board ──────────────────────────────────────
    println!("\n📋 Solved Board");
    println!("{}", "─".repeat(60));
    print!("{}", solver.state.display());

    // ── 5. O(log N) attention verification ────────────────────────
    println!("\n🧮 O(log N) Attention Verification");
    println!("{}", "─".repeat(60));

    let query = Vec2::new(1.0, 0.0);
    let (lin_score, lin_val) = solver.cache.linear_attention(&query);
    let (fast_score, fast_val) = solver.cache.fast_attention(&query);

    let score_match = (lin_score - fast_score).abs() < 1e-3;
    let val_match = lin_val == fast_val;

    println!("  Linear scan:  score={lin_score:.3}, value={lin_val}");
    println!("  Fast (hull):  score={fast_score:.3}, value={fast_val}");
    println!(
        "  Match: {}",
        if score_match && val_match {
            "✅"
        } else {
            "❌"
        }
    );

    // ── 6. Hull compression stats ─────────────────────────────────
    let total = solver.cache.len();
    let hull = solver.cache.hull_len();
    let compression = total as f64 / hull as f64;

    println!("\n📊 Trace Statistics");
    println!("{}", "─".repeat(60));
    println!("  Total trace entries: {total}");
    println!("  Hull vertices:       {hull}");
    println!("  Compression ratio:   {compression:.1}x");
    println!(
        "  Attention speedup:   O({total}) → O(log {hull}) ≈ O({})",
        (hull as f64).log2().ceil() as usize
    );
    println!("\n✨ Done.");
}

/// Explain why a digit violates Sudoku constraints.
fn violation_reason(state: &Sudoku9x9, row: usize, col: usize, digit: u8) -> &'static str {
    // Check row
    for c in 0..9 {
        if state.grid[row][c] == digit {
            return "already in row";
        }
    }
    // Check column
    for r in 0..9 {
        if state.grid[r][col] == digit {
            return "already in column";
        }
    }
    // Check box
    let box_r = (row / 3) * 3;
    let box_c = (col / 3) * 3;
    for r in 0..3 {
        for c in 0..3 {
            if state.grid[box_r + r][box_c + c] == digit {
                return "already in 3×3 box";
            }
        }
    }
    "unknown violation"
}
