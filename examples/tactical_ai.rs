//! Hierarchical Tactical AI — DDTree (Strategic) + A* (Tactical)
//!
//! Demonstrates a two-level AI architecture for grid-based tactical puzzles:
//! - **Strategic Layer**: DDTree chooses target visit order (tokens = target indices)
//! - **Tactical Layer**: A* computes paths between targets
//! - **Execution Layer**: Combines paths into full step-by-step solution
//!
//! This scales to real game maps (16×16+) because DDTree only sees
//! strategic decisions (7 targets), not every movement step (100+).
//!
//! Run: `cargo run --example tactical_ai`

use std::collections::HashSet;

use microgpt_rs::pruners::pathfinder::{Target, enumerate_targets, find_path};
use microgpt_rs::pruners::tactical_pruner::{GameState, TacticalPruner};
use microgpt_rs::speculative::types::ConstraintPruner;
use microgpt_rs::speculative::{build_dd_tree_pruned, extract_parent_tokens};
use microgpt_rs::types::Config;

// ── Emoji ──────────────────────────────────────────────────────

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

// ── Strategic Pruner ───────────────────────────────────────────
/// Wraps TacticalPruner at the strategic level.
/// Token indices map to targets: 0..M = monsters, M..M+T = treasures, last = goal.
/// The pruner validates strategic constraints (inventory, goal-lock, reachability).
struct StrategicPruner<'a> {
    tactical: &'a TacticalPruner,
    targets: Vec<Target>,
}

impl<'a> StrategicPruner<'a> {
    fn new(tactical: &'a TacticalPruner) -> Self {
        let targets = enumerate_targets(tactical.monsters.len(), tactical.treasures.len());
        Self { tactical, targets }
    }

    /// Build the A* blocked set: live monsters + goal (if treasures remain).
    fn blocked_set(&self, state: &GameState) -> HashSet<(usize, usize)> {
        let mut blocked = HashSet::new();
        for (i, &pos) in self.tactical.monsters.iter().enumerate() {
            if (state.killed_monsters & (1 << i)) == 0 {
                blocked.insert(pos);
            }
        }
        // Goal is locked until all treasures collected — block it from A*
        let all_treasures = (1 << self.tactical.treasures.len()) - 1;
        if state.collected_treasures != all_treasures {
            blocked.insert(self.tactical.goal);
        }
        blocked
    }

    /// A* blocked set adjusted for a specific target.
    /// Unblocks the target monster (so A* can reach it) or the goal (for Goal target).
    fn blocked_for_target(&self, state: &GameState, target: &Target) -> HashSet<(usize, usize)> {
        let mut blocked = self.blocked_set(state);
        if let Target::Monster(i) = target {
            blocked.remove(&self.tactical.monsters[*i]);
        }
        if let Target::Goal = target {
            blocked.remove(&self.tactical.goal);
        }
        blocked
    }

    /// Replay parent_tokens as target visit sequence, returning final GameState.
    fn replay_targets(
        &self,
        parent_tokens: &[usize],
        start_state: &GameState,
    ) -> Option<GameState> {
        let mut state = start_state.clone();

        for &token_idx in parent_tokens {
            let target = self.targets.get(token_idx)?;
            let target_pos = target.pos(
                &self.tactical.monsters,
                &self.tactical.treasures,
                self.tactical.goal,
            );

            let blocked = self.blocked_for_target(&state, target);

            // A* path from current position to target
            let path = find_path(
                &self.tactical.grid,
                (state.r, state.c),
                target_pos,
                &blocked,
            )?;

            // Execute movement steps
            for &action in &path {
                state = self.tactical.apply_action(&state, action)?;
            }

            // Execute target action (attack monster)
            if let Target::Monster(_) = target {
                state = self.tactical.apply_action(&state, 4)?; // Attack
            }
        }

        Some(state)
    }
}

impl ConstraintPruner for StrategicPruner<'_> {
    fn is_valid(&self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        let Some(target) = self.targets.get(token_idx) else {
            return false;
        };

        // Check target not already visited
        if parent_tokens.contains(&token_idx) {
            return false;
        }

        // Simulate state after visiting parent targets
        let start_state = self.tactical.initial_state();
        let Some(state) = self.replay_targets(parent_tokens, &start_state) else {
            return false;
        };

        let blocked = self.blocked_for_target(&state, target);

        match target {
            Target::Monster(i) => {
                if (state.killed_monsters & (1 << i)) != 0 {
                    return false;
                }
                let pos = self.tactical.monsters[*i];
                find_path(&self.tactical.grid, (state.r, state.c), pos, &blocked).is_some()
            }
            Target::Treasure(j) => {
                if (state.collected_treasures & (1 << j)) != 0 {
                    return false;
                }
                // Must have item to unlock treasure
                if state.inventory == 0 {
                    return false;
                }
                let pos = self.tactical.treasures[*j];
                // No live monster on same tile
                for (i, &m_pos) in self.tactical.monsters.iter().enumerate() {
                    if m_pos == pos && (state.killed_monsters & (1 << i)) == 0 {
                        return false;
                    }
                }
                find_path(&self.tactical.grid, (state.r, state.c), pos, &blocked).is_some()
            }
            Target::Goal => {
                let all_treasures = (1 << self.tactical.treasures.len()) - 1;
                if state.collected_treasures != all_treasures {
                    return false;
                }
                find_path(
                    &self.tactical.grid,
                    (state.r, state.c),
                    self.tactical.goal,
                    &blocked,
                )
                .is_some()
            }
        }
    }
}

// ── Solve & Execute ────────────────────────────────────────────

fn solve_hierarchical(pruner: &TacticalPruner) -> Option<Vec<usize>> {
    let state = pruner.initial_state();
    let strategic = StrategicPruner::new(pruner);
    let num_targets = strategic.targets.len();

    let mut config = Config::draft();
    config.vocab_size = num_targets;
    config.draft_lookahead = num_targets;
    config.tree_budget = 10000;

    // Uniform marginals (BFS) — let the pruner do all the work
    let marginals = vec![vec![1.0f32 / num_targets as f32; num_targets]; config.draft_lookahead];
    let refs: Vec<&[f32]> = marginals.iter().map(|v| v.as_slice()).collect();

    let tree = build_dd_tree_pruned(&refs, &config, &strategic, false);

    // Find target sequence that reaches the goal
    for node in &tree {
        let target_seq = extract_parent_tokens(node.parent_path, node.depth + 1);
        let Some(final_state) = strategic.replay_targets(&target_seq, &state) else {
            continue;
        };
        if (final_state.r, final_state.c) == pruner.goal {
            return expand_targets_to_actions(pruner, &target_seq);
        }
    }

    None
}

fn expand_targets_to_actions(pruner: &TacticalPruner, target_seq: &[usize]) -> Option<Vec<usize>> {
    let targets = enumerate_targets(pruner.monsters.len(), pruner.treasures.len());
    let mut state = pruner.initial_state();
    let mut all_actions = Vec::new();

    let strategic = StrategicPruner::new(pruner);

    for &token_idx in target_seq {
        let target = &targets[token_idx];
        let target_pos = target.pos(&pruner.monsters, &pruner.treasures, pruner.goal);

        let blocked = strategic.blocked_for_target(&state, target);

        // A* path to target
        let path = find_path(&pruner.grid, (state.r, state.c), target_pos, &blocked)?;

        // Execute movement
        for &action in &path {
            state = pruner.apply_action(&state, action)?;
            all_actions.push(action);
        }

        // Execute target action (attack monster)
        if let Target::Monster(_) = target {
            state = pruner.apply_action(&state, 4)?; // Attack
            all_actions.push(4);
        }
    }

    Some(all_actions)
}

// ── 17×16 Dungeon Map ──────────────────────────────────────────
//
//  ################
//  #B.....#.......#    Monsters: 3 at (4,2) (10,8) (15,5)
//  #.####.#.####..#    Treasures: 3 at (3,12) (10,1) (15,13)
//  #....#.#.#..T..#    Goal: G at (8,14)
//  #.M..#.#.#.###.#    Start: B at (1,1)
//  ####.#.#.#.....#
//  #....#...#.....#    Strategic tokens: 3M + 3T + 1G = 7
//  #.########.###.#    DDTree lookahead = 7 → fits u128/16 ✓
//  #.#.......#...G#
//  #.#.###.###.##.#
//  #T..#.#.M.#..#.#
//  ###.#.#.#..##..#
//  #...#.#.#.##...#
//  #.###.#.#....#.#
//  #.....#.####.#.#
//  #....M.....#.T.#
//  ################

const MAP: &str = "\
# # # # # # # # # # # # # # # #
# B . . . . . # . . . . . . . #
# . # # # # . # . # # # # . . #
# . . . . # . # . # . . T . . #
# . M . . # . # . # . # # # . #
# # # # . # . # . # . . . . . #
# . . . . # . . . # . . . . . #
# . # # # # # # # # . # # # . #
# . # . . . . . . . # . . . G #
# . # . # # # . # # # . # # . #
# T . . # . # . M . # . . # . #
# # # . # . # . # . . # # . . #
# . . . # . # . # . # # . . . #
# . # # # . # . # . . . . # . #
# . . . . . # . # # # # . # . #
# . . . . M . . . . . . # T . #
# # # # # # # # # # # # # # # #";

fn main() {
    let pruner = TacticalPruner::new(MAP);

    println!("🐻 Hierarchical Tactical AI — 16×16 Dungeon");
    println!(
        "Monsters: {} Treasures: {} Goal: ({},{})",
        pruner.monsters.len(),
        pruner.treasures.len(),
        pruner.goal.0,
        pruner.goal.1,
    );
    println!();

    // Show initial state
    let initial = pruner.initial_state();
    print_grid(&pruner, &initial);
    println!();

    let start = std::time::Instant::now();
    let solution = solve_hierarchical(&pruner);
    let elapsed = start.elapsed();

    match solution {
        Some(actions) => {
            println!(
                "🎉 Hierarchical solution found in {} steps ({:.2?})",
                actions.len(),
                elapsed
            );

            // Verify by replaying through TacticalPruner
            let mut state = pruner.initial_state();
            for (i, &action) in actions.iter().enumerate() {
                state = pruner.apply_action(&state, action).unwrap();
                if i < 5 || i >= actions.len() - 3 {
                    println!(
                        "Step {:>3}: {} | pos=({},{}) inv={} killed={:03b} collected={:03b}",
                        i + 1,
                        TacticalPruner::action_name(action),
                        state.r,
                        state.c,
                        state.inventory,
                        state.killed_monsters,
                        state.collected_treasures,
                    );
                } else if i == 5 {
                    println!("  ... ({} more steps) ...", actions.len() - 8);
                }
            }

            println!();
            print_grid(&pruner, &state);
            println!();

            // Assertions
            assert_eq!((state.r, state.c), pruner.goal, "Bear must be at goal");
            let all_treasures = (1 << pruner.treasures.len()) - 1;
            assert_eq!(
                state.collected_treasures, all_treasures,
                "All treasures collected"
            );
            assert_eq!(
                state.killed_monsters,
                (1 << pruner.monsters.len()) - 1,
                "All monsters killed"
            );
            println!("✅ Solution verified: at goal, all treasures, all monsters killed.");
        }
        None => {
            println!("❌ No solution found.");
        }
    }
}
