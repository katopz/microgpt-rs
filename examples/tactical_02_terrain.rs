//! Terrain Cost Example — DDTree + A* with Terrain-Weighted Pathfinding
//!
//! Demonstrates how the hierarchical AI considers terrain costs when pathfinding:
//! - **Desert Crossing**: Sand (~) shortcut through wall gap vs grass detour
//! - **River Crossing**: Water (w) expensive direct route vs bridge (.) circuitous
//! - **Mixed Terrain**: 8×8 maze with sand + water + grass cost trade-offs
//!
//! A* uses terrain-weighted costs: grass(.)=1, sand(~)=2, water(w)=3.
//! The AI prefers cheaper terrain routes when available, and accepts
//! expensive terrain only when it provides a meaningful shortcut.
//!
//! Run: `cargo run --example tactical_terrain`

use std::collections::HashSet;

use microgpt_rs::pruners::pathfinder::{Target, enumerate_targets, find_path};
use microgpt_rs::pruners::tactical_pruner::{GameState, TacticalPruner};
use microgpt_rs::speculative::types::ConstraintPruner;
use microgpt_rs::speculative::{build_dd_tree_pruned, extract_parent_tokens};
use microgpt_rs::types::Config;

// ── Terrain Emoji ──────────────────────────────────────────────

const BEAR: &str = "🐻";
const MONSTER_LIVE: &str = "👹";
const MONSTER_DEAD: &str = "💀";
const TREASURE: &str = "💎";
const GOAL: &str = "🚪";
const GOAL_OPEN: &str = "🏆";
const WALL: &str = "🧱";
const GRASS: &str = "⬜";
const SAND: &str = "🟨";
const WATER: &str = "🟦";
const ITEM: &str = "🔑";

fn terrain_emoji(ch: char) -> &'static str {
    match ch {
        '#' => WALL,
        '~' => SAND,
        'w' => WATER,
        _ => GRASS,
    }
}

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
    for (i, &(mr, mc)) in pruner.monsters.iter().enumerate() {
        if (mr, mc) == (r, c) && (state.killed_monsters & (1 << i)) != 0 {
            return MONSTER_DEAD.into();
        }
    }
    terrain_emoji(pruner.grid[r][c]).into()
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

struct StrategicPruner<'a> {
    tactical: &'a TacticalPruner,
    targets: Vec<Target>,
}

impl<'a> StrategicPruner<'a> {
    fn new(tactical: &'a TacticalPruner) -> Self {
        let targets = enumerate_targets(tactical.monsters.len(), tactical.treasures.len());
        Self { tactical, targets }
    }

    fn blocked_set(&self, state: &GameState) -> HashSet<(usize, usize)> {
        let mut blocked = HashSet::new();
        for (i, &pos) in self.tactical.monsters.iter().enumerate() {
            if (state.killed_monsters & (1 << i)) == 0 {
                blocked.insert(pos);
            }
        }
        let all_treasures = (1 << self.tactical.treasures.len()) - 1;
        if state.collected_treasures != all_treasures {
            blocked.insert(self.tactical.goal);
        }
        blocked
    }

    fn blocked_for_target(&self, state: &GameState, target: &Target) -> HashSet<(usize, usize)> {
        let mut blocked = self.blocked_set(state);
        match target {
            Target::Monster(i) => {
                blocked.remove(&self.tactical.monsters[*i]);
            }
            Target::Goal => {
                blocked.remove(&self.tactical.goal);
            }
            Target::Treasure(_) => {}
        }
        blocked
    }

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
            let path = find_path(
                &self.tactical.grid,
                (state.r, state.c),
                target_pos,
                &blocked,
            )?;
            for &action in &path {
                state = self.tactical.apply_action(&state, action)?;
            }
            if let Target::Monster(_) = target {
                state = self.tactical.apply_action(&state, 4)?;
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
        if parent_tokens.contains(&token_idx) {
            return false;
        }
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
                if state.inventory == 0 {
                    return false;
                }
                let pos = self.tactical.treasures[*j];
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

    let marginals = vec![vec![1.0f32 / num_targets as f32; num_targets]; config.draft_lookahead];
    let refs: Vec<&[f32]> = marginals.iter().map(|v| v.as_slice()).collect();

    let tree = build_dd_tree_pruned(&refs, &config, &strategic, false);

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
        let path = find_path(&pruner.grid, (state.r, state.c), target_pos, &blocked)?;
        for &action in &path {
            state = pruner.apply_action(&state, action)?;
            all_actions.push(action);
        }
        if let Target::Monster(_) = target {
            state = pruner.apply_action(&state, 4)?;
            all_actions.push(4);
        }
    }

    Some(all_actions)
}

// ── Map Definitions ────────────────────────────────────────────

/// Desert Crossing (5×7): Sand (~) shortcut through wall gap.
/// The wall column at col 3 forces traversal through the sand tile.
/// The player must cross sand to reach treasure and goal.
const MAP_DESERT: &str = "\
. . . . . . .
B . . # . . .
. . . # . . .
. M . ~ . T G
. . . # . . .";

/// River Crossing (5×7): Water (w) direct but costly, bridge (.) circuitous but cheap.
/// Water tiles cost 3x more than grass. A* prefers the grass bridge route.
const MAP_RIVER: &str = "\
. . . . . . .
B . . . . . .
w w w . w w w
. . . M . T .
. . . . . . G";

/// Mixed Terrain (8×8): Complex maze with sand, water, walls, and grass.
/// Multiple terrain types create interesting cost trade-offs.
/// Wall at col 4 forces sand traversal; water tiles add optional hazards.
const MAP_MIXED: &str = "\
. ~ . . # . . .
. ~ . . # . w .
B . . M # . w .
. . ~ ~ . . . .
. . . . . T . .
. # # . . . . .
. . . . . . . .
. . . . . . . G";

// ── Solve Helper ───────────────────────────────────────────────

struct MapResult {
    actions: Vec<usize>,
    final_state: GameState,
}

fn solve_map(name: &str, map_str: &str) -> Option<MapResult> {
    let pruner = TacticalPruner::new(map_str);

    println!(
        "═══ {} ({}×{}) ═══",
        name,
        pruner.grid.len(),
        pruner.grid[0].len()
    );
    println!(
        "Monsters: {} Treasures: {} Goal: ({},{})",
        pruner.monsters.len(),
        pruner.treasures.len(),
        pruner.goal.0,
        pruner.goal.1,
    );

    // Count terrain types
    let mut sand_count = 0usize;
    let mut water_count = 0usize;
    let mut grass_count = 0usize;
    let mut wall_count = 0usize;
    for row in &pruner.grid {
        for &ch in row {
            match ch {
                '~' => sand_count += 1,
                'w' => water_count += 1,
                '#' => wall_count += 1,
                _ => grass_count += 1,
            }
        }
    }
    println!(
        "Terrain: {grass_count} grass, {sand_count} sand, {water_count} water, {wall_count} walls",
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
            // Replay to get final state
            let mut state = pruner.initial_state();
            for &action in &actions {
                state = pruner.apply_action(&state, action).unwrap();
            }

            let movement_steps = actions.iter().filter(|&&a| a < 4).count();
            let attack_steps = actions.iter().filter(|&&a| a == 4).count();

            println!("🎉 Solution found!");
            println!(
                "   Steps: {} ({} movement + {} attack)",
                actions.len(),
                movement_steps,
                attack_steps,
            );
            println!("   Total cost (terrain-weighted): {}", state.total_cost);
            println!("   Time: {:.2?}", elapsed);

            // Print action sequence with arrows
            print!("   Path: ");
            for &action in &actions {
                let symbol = match action {
                    0 => "↑",
                    1 => "↓",
                    2 => "←",
                    3 => "→",
                    4 => "⚔",
                    _ => "?",
                };
                print!("{symbol}");
            }
            println!();
            println!();

            // Show final state
            print_grid(&pruner, &state);
            println!();

            // Assertions
            assert_eq!((state.r, state.c), pruner.goal, "Must be at goal");
            let all_treasures = (1 << pruner.treasures.len()) - 1;
            assert_eq!(
                state.collected_treasures, all_treasures,
                "All treasures collected"
            );
            println!("✅ Verified: at goal, all treasures collected.");

            Some(MapResult {
                actions,
                final_state: state,
            })
        }
        None => {
            println!("❌ No solution found.");
            None
        }
    }
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       Terrain Cost Example — DDTree + A* Pathfinding           ║");
    println!("║       Grass(.)=1  Sand(~)=2  Water(w)=3  Wall(#)=blocked      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Map 1: Desert Crossing ─────────────────────────────────
    let desert = solve_map("Desert Crossing", MAP_DESERT);
    assert!(desert.is_some(), "Desert map must be solvable");

    let desert_ref = desert.as_ref().unwrap();
    let desert_moves = desert_ref.actions.iter().filter(|&&a| a < 4).count() as u32;
    let desert_surcharge = desert_ref.final_state.total_cost - desert_moves;
    assert!(
        desert_ref.final_state.total_cost > desert_moves,
        "Desert: terrain cost must exceed step count (sand adds cost)"
    );
    println!(
        "   📊 Desert: {desert_moves} steps, cost {} — sand surcharge: {desert_surcharge}",
        desert_ref.final_state.total_cost,
    );
    println!();

    // ── Map 2: River Crossing ──────────────────────────────────
    let river = solve_map("River Crossing", MAP_RIVER);
    assert!(river.is_some(), "River map must be solvable");

    let river_ref = river.as_ref().unwrap();
    let river_moves = river_ref.actions.iter().filter(|&&a| a < 4).count() as u32;
    println!(
        "   📊 River: {river_moves} steps, cost {} — bridge route avoids water",
        river_ref.final_state.total_cost,
    );
    println!();

    // ── Map 3: Mixed Terrain ───────────────────────────────────
    let mixed = solve_map("Mixed Terrain", MAP_MIXED);
    assert!(mixed.is_some(), "Mixed terrain map must be solvable");

    let mixed_ref = mixed.as_ref().unwrap();
    let mixed_moves = mixed_ref.actions.iter().filter(|&&a| a < 4).count() as u32;
    let mixed_surcharge = mixed_ref.final_state.total_cost - mixed_moves;
    assert!(
        mixed_ref.final_state.total_cost > mixed_moves,
        "Mixed: terrain cost must exceed step count"
    );
    println!(
        "   📊 Mixed: {mixed_moves} steps, cost {} — terrain surcharge: {mixed_surcharge}",
        mixed_ref.final_state.total_cost,
    );
    println!();

    // ── Summary Table ──────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                          Summary                                ║");
    println!("╠═══════════════╦══════════╦═══════════╦═══════════╦══════════════╣");
    println!("║ Map           ║ Steps    ║ Cost      ║ Surcharge ║ Route        ║");
    println!("╠═══════════════╬══════════╬═══════════╬═══════════╬══════════════╣");

    let results: [(&str, u32, u32, &str); 3] = [
        (
            "Desert",
            desert_moves,
            desert_ref.final_state.total_cost,
            "sand shortcut",
        ),
        (
            "River",
            river_moves,
            river_ref.final_state.total_cost,
            "bridge bypass",
        ),
        (
            "Mixed",
            mixed_moves,
            mixed_ref.final_state.total_cost,
            "sand+grass",
        ),
    ];

    for (name, steps, cost, route) in results {
        let surcharge = cost - steps;
        println!(
            "║ {:<13} ║ {:>8} ║ {:>9} ║ {:>+9} ║ {:<12} ║",
            name, steps, cost, surcharge, route,
        );
    }

    println!("╚═══════════════╩══════════╩═══════════╩═══════════╩══════════════╝");
    println!();
    println!("💡 Key observations:");
    println!("   • Desert: Sand (~) costs 2x — wall gap forces sand traversal");
    println!("   • River:  Water (w) costs 3x — A* takes longer grass bridge route");
    println!("   • Mixed:  Sand unavoidable — A* minimizes expensive terrain steps");
    println!();
    println!("✅ All terrain maps solved. A* prefers cheaper terrain routes.");
}
