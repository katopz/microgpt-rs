//! Blue Bear Tactical Puzzle — Solver & Benchmark
//!
//! Demonstrates using Speculative Decoding (DDTree) with ConstraintPruner
//! as a heavily constrained state-space solver.
//!
//! The DDTree with uniform marginals operates as BFS/Best-First search.
//! The ConstraintPruner eliminates impossible moves (walls, locked treasures,
//! dead monsters), keeping the branching factor small.
//!
//! Run: `cargo run --example blue_bear`

use microgpt_rs::pruners::tactical_pruner::{GameState, TacticalPruner};
use microgpt_rs::speculative::{build_dd_tree_pruned, extract_parent_tokens};
use microgpt_rs::types::Config;

// ── Emoji Map ──────────────────────────────────────────────────

const BEAR: &str = "🐻";
const MONSTER_LIVE: &str = "👹";
const MONSTER_DEAD: &str = "💀";
const TREASURE: &str = "💎";
const GOAL: &str = "🚪";
const GOAL_OPEN: &str = "🏆";
const WALL: &str = "🧱";
const FLOOR: &str = "⬜";
const ITEM: &str = "🔑";

fn cell_emoji(pruner: &TacticalPruner, state: &GameState, r: usize, c: usize) -> String {
    if state.r == r && state.c == c {
        return BEAR.into();
    }
    for (i, &(mr, mc)) in pruner.monsters.iter().enumerate() {
        if (mr, mc) == (r, c) && (state.killed_monsters & (1 << i)) == 0 {
            return MONSTER_LIVE.into();
        }
    }
    for (i, &(mr, mc)) in pruner.monsters.iter().enumerate() {
        if (mr, mc) == (r, c) && (state.dropped_items & (1 << i)) != 0 {
            return ITEM.into();
        }
    }
    for (i, &(tr, tc)) in pruner.treasures.iter().enumerate() {
        if (tr, tc) == (r, c) && (state.collected_treasures & (1 << i)) == 0 {
            return TREASURE.into();
        }
    }
    if pruner.goal == (r, c) {
        let all = (1 << pruner.treasures.len()) - 1;
        return if state.collected_treasures == all {
            GOAL_OPEN.into()
        } else {
            GOAL.into()
        };
    }
    if pruner.grid[r][c] == '#' {
        return WALL.into();
    }
    for (i, &(mr, mc)) in pruner.monsters.iter().enumerate() {
        if (mr, mc) == (r, c) && (state.killed_monsters & (1 << i)) != 0 {
            return MONSTER_DEAD.into();
        }
    }
    FLOOR.into()
}

fn print_grid(pruner: &TacticalPruner, state: &GameState) {
    for r in 0..pruner.grid.len() {
        for c in 0..pruner.grid[r].len() {
            print!("{} ", cell_emoji(pruner, state, r, c));
        }
        println!();
    }
}

fn action_name(action: usize) -> &'static str {
    match action {
        0 => "↑ Up",
        1 => "↓ Down",
        2 => "← Left",
        3 => "→ Right",
        4 => "⚔ Attack",
        _ => "???",
    }
}

fn main() {
    // Map: B=Start, X=Monster+Treasure, T=Treasure, #=Wall, M=Monster, G=Goal
    //
    //   B X T
    //   # M G
    //
    // Solution: → ⚔ ↓ ⚔ ↑ → ↓ (7 steps)
    // Note: DDTree packs 16 bits/token into u128 → max lookahead = 8.
    let map = "\
        B X T\n\
        # M G";

    let pruner = TacticalPruner::new(map);

    let mut config = Config::draft();
    config.vocab_size = 5; // Up, Down, Left, Right, Attack
    config.draft_lookahead = 8; // u128/16 = 8 tokens max
    config.tree_budget = 10000;

    // Uniform marginals: DDTree operates as BFS / Best-First search
    let marginals = vec![vec![0.2f32; 5]; config.draft_lookahead];
    let refs: Vec<&[f32]> = marginals.iter().map(|v| v.as_slice()).collect();

    println!("🐻 Blue Bear Tactical Solver");
    println!("Map:\n{map}\n");

    let start = std::time::Instant::now();
    let tree = build_dd_tree_pruned(&refs, &config, &pruner, false);
    let elapsed = start.elapsed();

    println!("Tree built: {} nodes in {:.2?}", tree.len(), elapsed);

    // Structural assertions
    assert!(!tree.is_empty(), "Tree should contain nodes after pruning");
    assert!(
        tree.len() < 1000,
        "Pruned tree should be small, got {} nodes",
        tree.len()
    );

    // Find first path that reaches the goal
    let mut solution = None;
    for node in &tree {
        let path = extract_parent_tokens(node.parent_path, node.depth + 1);
        if let Some(state) = pruner.replay_state(&path)
            && (state.r, state.c) == pruner.goal
        {
            solution = Some(path);
            break;
        }
    }

    let path = solution.expect("Puzzle should be solvable within lookahead");
    assert_eq!(path.len(), 7, "Expected 7-step solution for BXT/SMG map");

    println!("🎉 Found solution in {} steps!\n", path.len());

    // Verify expected solution: → ⚔ ↓ ⚔ ↑ → ↓
    let expected = [3, 4, 1, 4, 0, 3, 1]; // Right, Attack, Down, Attack, Up, Right, Down
    assert_eq!(
        path, expected,
        "Solution should match expected action sequence"
    );

    let mut state = pruner.initial_state();

    // Assert initial state
    assert_eq!((state.r, state.c), (0, 0), "Bear starts at (0,0)");
    assert_eq!(state.inventory, 0, "Start with empty inventory");
    assert_eq!(state.killed_monsters, 0, "Start with no kills");
    assert_eq!(state.collected_treasures, 0, "Start with no treasures");

    println!("START:");
    print_grid(&pruner, &state);
    println!();

    for (i, &action) in path.iter().enumerate() {
        state = pruner.apply_action(&state, action).unwrap();
        println!(
            "Step {}: {} | inv={} | killed={:02b} | collected={:02b}",
            i + 1,
            action_name(action),
            state.inventory,
            state.killed_monsters,
            state.collected_treasures,
        );
        print_grid(&pruner, &state);
        println!();
    }

    // Verify final state
    assert_eq!((state.r, state.c), pruner.goal, "Bear must be at goal");
    let all_treasures = (1 << pruner.treasures.len()) - 1;
    assert_eq!(
        state.collected_treasures, all_treasures,
        "All treasures must be collected"
    );
    assert_eq!(
        state.killed_monsters, all_treasures,
        "All monsters must be killed"
    );
    assert_eq!(state.inventory, 0, "Inventory should be empty at goal");
    println!("✅ Solution verified: bear at goal, all treasures collected, all monsters killed.");
}
