//! Multi-Floor Dungeon — DDTree (Strategic) + Multi-Floor A* (Tactical)
//!
//! Demonstrates multi-floor dungeon exploration with:
//! - **Dungeon 1** (B1→B2): 2-floor, 8×8, 4 monsters, 2 treasures
//! - **Dungeon 2** (F1→F2→F3): 3-floor, 6×6, 2 monsters, 2 treasures
//!
//! Architecture mirrors tactical_ai.rs but adapted for multi-floor:
//! - Strategic Layer: DDTree chooses target visit order (tokens = targets)
//! - Tactical Layer: find_path_multifloor computes cross-floor A* paths
//! - Execution Layer: DungeonPruner.apply_action validates each step
//!
//! Run: `cargo run --example dungeon_02_multifloor`

use std::collections::HashMap;

use microgpt_rs::pruners::dungeon_pathfinder::{
    DungeonAction, MultiFloorBlocked, MultiFloorTarget, enumerate_multifloor_targets,
    find_path_multifloor,
};
use microgpt_rs::pruners::dungeon_pruner::{
    DungeonMap, DungeonPruner, DungeonState, StairConnection,
};
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
const FLOOR_TILE: &str = "⬜";
const ITEM: &str = "🔑";
const STAIRS: &str = "🪜";

fn is_stair_on_floor(map: &DungeonMap, floor: usize, r: usize, c: usize) -> bool {
    map.stairs.iter().any(|s| {
        (s.from.0 == floor && s.from.1 == r && s.from.2 == c)
            || (s.to.0 == floor && s.to.1 == r && s.to.2 == c)
    })
}

fn cell_emoji_dungeon(
    pruner: &DungeonPruner,
    state: &DungeonState,
    floor: usize,
    r: usize,
    c: usize,
) -> String {
    if state.floor == floor && state.r == r && state.c == c {
        return BEAR.into();
    }
    for (i, &(f, mr, mc)) in pruner.map.monsters.iter().enumerate() {
        if (f, mr, mc) == (floor, r, c) && (state.killed_monsters & (1 << i)) == 0 {
            return MONSTER_LIVE.into();
        }
    }
    for (i, &(f, mr, mc)) in pruner.map.monsters.iter().enumerate() {
        if (f, mr, mc) == (floor, r, c) && (state.dropped_items & (1 << i)) != 0 {
            return ITEM.into();
        }
    }
    for (i, &(f, tr, tc)) in pruner.map.treasures.iter().enumerate() {
        if (f, tr, tc) == (floor, r, c) && (state.collected_treasures & (1 << i)) == 0 {
            return TREASURE.into();
        }
    }
    if pruner.map.goal == (floor, r, c) {
        let all = (1 << pruner.map.treasures.len()) - 1;
        return if state.collected_treasures == all {
            GOAL_OPEN.into()
        } else {
            GOAL.into()
        };
    }
    if is_stair_on_floor(&pruner.map, floor, r, c) {
        return STAIRS.into();
    }
    if pruner.map.floors[floor][r][c] == '#' {
        return WALL.into();
    }
    for (i, &(f, mr, mc)) in pruner.map.monsters.iter().enumerate() {
        if (f, mr, mc) == (floor, r, c) && (state.killed_monsters & (1 << i)) != 0 {
            return MONSTER_DEAD.into();
        }
    }
    FLOOR_TILE.into()
}

fn print_floor(pruner: &DungeonPruner, state: &DungeonState, floor: usize) {
    let grid = &pruner.map.floors[floor];
    for (r, row) in grid.iter().enumerate() {
        for c in 0..row.len() {
            print!("{} ", cell_emoji_dungeon(pruner, state, floor, r, c));
        }
        println!();
    }
}

fn print_all_floors(pruner: &DungeonPruner, state: &DungeonState) {
    for floor in 0..pruner.map.floors.len() {
        println!("  ── Floor {floor} ──");
        print_floor(pruner, state, floor);
    }
}

// ── Action Conversion ─────────────────────────────────────────

/// Convert `DungeonAction` from pathfinder to `usize` for `DungeonPruner.apply_action`.
fn dungeon_action_to_usize(action: &DungeonAction) -> usize {
    match action {
        DungeonAction::Move(n) => *n,
        DungeonAction::Attack => 4,
        DungeonAction::UseStairs(_) => 5,
    }
}

// ── Multi-Floor Strategic Pruner ──────────────────────────────

/// Wraps DungeonPruner at the strategic level.
/// Token indices map to multi-floor targets:
///   0..M = monsters, M..M+T = treasures, last = goal.
struct MultiFloorStrategicPruner<'a> {
    pruner: &'a DungeonPruner,
    targets: Vec<MultiFloorTarget>,
}

impl<'a> MultiFloorStrategicPruner<'a> {
    fn new(pruner: &'a DungeonPruner) -> Self {
        let targets =
            enumerate_multifloor_targets(pruner.map.monsters.len(), pruner.map.treasures.len());
        Self { pruner, targets }
    }

    /// Build per-floor blocked set adjusted for a specific target and state.
    fn blocked_for_target(
        &self,
        state: &DungeonState,
        target: &MultiFloorTarget,
    ) -> MultiFloorBlocked {
        let mut blocked: MultiFloorBlocked = HashMap::new();

        // Block goal unless all treasures collected or targeting goal
        let all_treasures = (1 << self.pruner.map.treasures.len()) - 1;
        if state.collected_treasures != all_treasures {
            match target {
                MultiFloorTarget::Goal => {}
                _ => {
                    let (floor, r, c) = self.pruner.map.goal;
                    blocked.entry(floor).or_default().insert((r, c));
                }
            }
        }

        // Block uncollected treasures if no item (can't walk onto locked treasure)
        if state.inventory == 0 {
            for (i, &(f, r, c)) in self.pruner.map.treasures.iter().enumerate() {
                if (state.collected_treasures & (1 << i)) == 0 {
                    match target {
                        MultiFloorTarget::Treasure(j) if *j == i => continue,
                        _ => {
                            blocked.entry(f).or_default().insert((r, c));
                        }
                    }
                }
            }
        }

        blocked
    }

    /// Replay parent_tokens as target visit sequence, returning final DungeonState.
    fn replay_targets(
        &self,
        parent_tokens: &[usize],
        start_state: &DungeonState,
    ) -> Option<DungeonState> {
        let mut state = start_state.clone();

        for &token_idx in parent_tokens {
            let target = self.targets.get(token_idx)?;
            let target_pos = target.pos(
                &self.pruner.map.monsters,
                &self.pruner.map.treasures,
                self.pruner.map.goal,
            );

            let blocked = self.blocked_for_target(&state, target);
            let path = find_path_multifloor(
                &self.pruner.map,
                (state.floor, state.r, state.c),
                target_pos,
                &blocked,
            )?;

            for action in &path {
                let action_usize = dungeon_action_to_usize(action);
                state = self.pruner.apply_action(&state, action_usize)?;
            }

            if let MultiFloorTarget::Monster(_) = target {
                state = self.pruner.apply_action(&state, 4)?;
            }
        }

        Some(state)
    }
}

impl ConstraintPruner for MultiFloorStrategicPruner<'_> {
    fn is_valid(&self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        let Some(target) = self.targets.get(token_idx) else {
            return false;
        };

        if parent_tokens.contains(&token_idx) {
            return false;
        }

        let start_state = self.pruner.initial_state();
        let Some(state) = self.replay_targets(parent_tokens, &start_state) else {
            return false;
        };

        let blocked = self.blocked_for_target(&state, target);

        match target {
            MultiFloorTarget::Monster(i) => {
                if (state.killed_monsters & (1 << i)) != 0 {
                    return false;
                }
                let pos = self.pruner.map.monsters[*i];
                find_path_multifloor(
                    &self.pruner.map,
                    (state.floor, state.r, state.c),
                    pos,
                    &blocked,
                )
                .is_some()
            }
            MultiFloorTarget::Treasure(j) => {
                if (state.collected_treasures & (1 << j)) != 0 {
                    return false;
                }
                if state.inventory == 0 {
                    return false;
                }
                let pos = self.pruner.map.treasures[*j];
                for (i, &m_pos) in self.pruner.map.monsters.iter().enumerate() {
                    if m_pos == pos && (state.killed_monsters & (1 << i)) == 0 {
                        return false;
                    }
                }
                find_path_multifloor(
                    &self.pruner.map,
                    (state.floor, state.r, state.c),
                    pos,
                    &blocked,
                )
                .is_some()
            }
            MultiFloorTarget::Goal => {
                let all_treasures = (1 << self.pruner.map.treasures.len()) - 1;
                if state.collected_treasures != all_treasures {
                    return false;
                }
                find_path_multifloor(
                    &self.pruner.map,
                    (state.floor, state.r, state.c),
                    self.pruner.map.goal,
                    &blocked,
                )
                .is_some()
            }
        }
    }
}

// ── Solve & Execute ────────────────────────────────────────────

fn solve_multifloor(pruner: &DungeonPruner) -> Option<Vec<usize>> {
    let strategic = MultiFloorStrategicPruner::new(pruner);
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
        let start_state = pruner.initial_state();
        let Some(final_state) = strategic.replay_targets(&target_seq, &start_state) else {
            continue;
        };
        if (final_state.floor, final_state.r, final_state.c) == pruner.map.goal {
            return expand_targets_to_dungeon_actions(pruner, &target_seq);
        }
    }

    None
}

fn expand_targets_to_dungeon_actions(
    pruner: &DungeonPruner,
    target_seq: &[usize],
) -> Option<Vec<usize>> {
    let strategic = MultiFloorStrategicPruner::new(pruner);
    let mut state = pruner.initial_state();
    let mut all_actions = Vec::new();

    for &token_idx in target_seq {
        let target = &strategic.targets[token_idx];
        let target_pos = target.pos(&pruner.map.monsters, &pruner.map.treasures, pruner.map.goal);

        let blocked = strategic.blocked_for_target(&state, target);
        let path = find_path_multifloor(
            &pruner.map,
            (state.floor, state.r, state.c),
            target_pos,
            &blocked,
        )?;

        for action in &path {
            let action_usize = dungeon_action_to_usize(action);
            state = pruner.apply_action(&state, action_usize)?;
            all_actions.push(action_usize);
        }

        if let MultiFloorTarget::Monster(_) = target {
            state = pruner.apply_action(&state, 4)?;
            all_actions.push(4);
        }
    }

    Some(all_actions)
}

// ── Dungeon 1: Two-Floor (B1 → B2) ───────────────────────────
//
//  Floor 0 (B1): 8×8 — start, 2 monsters, 1 treasure, stairs down
//  ################
//  #B.....#.......#   Start: B at (1,1)
//  #......#.......#   Monsters: M0 at (3,2), M1 at (5,5)
//  #.M............#   Treasure: T0 at (6,2)
//  #..............#   Stairs: 🪜 at (7,6) → Floor 1 (1,1)
//  #.....M........#
//  #.T............#
//  #..............#
//  ################
//
//  Floor 1 (B2): 8×8 — 2 monsters, 1 treasure, goal
//  ################
//  #..............#   Monsters: M2 at (2,2), M3 at (5,5)
//  #.M............#   Treasure: T1 at (6,5)
//  #.....#........#   Goal: G at (7,6)
//  #..............#
//  #.....M........#
//  #.....T........#
//  #.....G........#
//  ################

const DUNGEON1_FLOOR0: &str = "\
# # # # # # # #
# B . . . . . #
# . . . . . . #
# . M . . . . #
# . . . . . . #
# . . . . M . #
# . T . . . . #
# . . . . . . #";

const DUNGEON1_FLOOR1: &str = "\
# # # # # # # #
# . . . . . . #
# . M . . . . #
# . . . . # . #
# . . . . . . #
# . . . . M . #
# . . . . T . #
# . . . . . G #";

fn dungeon1_stairs() -> Vec<StairConnection> {
    vec![StairConnection {
        from: (0, 7, 6),
        to: (1, 1, 1),
    }]
}

// ── Dungeon 2: Three-Floor (F1 → F2 → F3) ────────────────────
//
//  Floor 0 (F1): 6×6 — start, stairs down
//  ######
//  #B...#   Start: B at (1,1)
//  #....#   Stairs: 🪜 at (5,4) → Floor 1 (1,1)
//  #....#
//  #....#
//  #....#
//
//  Floor 1 (F2): 6×6 — monster, treasure, stairs down
//  ######
//  #..M.#   Monster: M0 at (1,3)
//  #....#   Treasure: T0 at (4,2)
//  #....#   Stairs: 🪜 at (5,4) → Floor 2 (1,1)
//  #.T..#
//  #....#
//
//  Floor 2 (F3): 6×6 — monster, treasure, goal
//  ######
//  #....#   Monster: M1 at (2,2)
//  #.M..#   Treasure: T1 at (4,3)
//  #....#   Goal: G at (5,4)
//  #..T.#
//  #...G#

const DUNGEON2_FLOOR0: &str = "\
# # # # # #
# B . . . #
# . . . . #
# . . . . #
# . . . . #
# . . . . #";

const DUNGEON2_FLOOR1: &str = "\
# # # # # #
# . . M . #
# . . . . #
# . . . . #
# . T . . #
# . . . . #";

const DUNGEON2_FLOOR2: &str = "\
# # # # # #
# . . . . #
# . M . . #
# . . . . #
# . . T . #
# . . . G #";

fn dungeon2_stairs() -> Vec<StairConnection> {
    vec![
        StairConnection {
            from: (0, 5, 4),
            to: (1, 1, 1),
        },
        StairConnection {
            from: (1, 5, 4),
            to: (2, 1, 1),
        },
    ]
}

// ── Solution Display ──────────────────────────────────────────

fn print_solution(pruner: &DungeonPruner, actions: &[usize]) {
    let mut state = pruner.initial_state();
    let mut prev_floor = state.floor;
    let mut floor_steps: usize = 0;
    let mut floor_cost_start: u32 = 0;
    let num_actions = actions.len();

    for (i, &action) in actions.iter().enumerate() {
        let prev_cost = state.total_cost;
        state = pruner.apply_action(&state, action).unwrap();

        if state.floor != prev_floor {
            let floor_cost = prev_cost - floor_cost_start;
            println!("  ── Floor {prev_floor}: {floor_steps} actions, cost +{floor_cost} ──");
            prev_floor = state.floor;
            floor_steps = 0;
            floor_cost_start = prev_cost;
        }

        floor_steps += 1;

        let show = i < 5 || i >= num_actions - 3 || action == 4 || action == 5;
        if show {
            println!(
                "  Step {:>3}: {:<12} | {}",
                i + 1,
                DungeonPruner::action_name(action),
                state.summary(),
            );
        } else if i == 5 {
            println!("  ... ({} more steps) ...", num_actions.saturating_sub(8));
        }
    }

    let floor_cost = state.total_cost - floor_cost_start;
    println!("  ── Floor {prev_floor}: {floor_steps} actions, cost +{floor_cost} ──");
    println!(
        "  Total: {} actions, cost {}",
        num_actions, state.total_cost
    );
}

// ── Main ──────────────────────────────────────────────────────

fn main() {
    // ═══════════════════════════════════════════════════════════
    //  Dungeon 1: Two-Floor (B1 → B2)
    // ═══════════════════════════════════════════════════════════
    println!("🏰 Dungeon 1: Two-Floor (B1 → B2)");
    println!();

    let map1 = DungeonMap::new(&[DUNGEON1_FLOOR0, DUNGEON1_FLOOR1], dungeon1_stairs());
    let pruner1 = DungeonPruner::new(map1);

    println!(
        "  Floors: {} Monsters: {} Treasures: {} Goal: {:?}",
        pruner1.map.floors.len(),
        pruner1.map.monsters.len(),
        pruner1.map.treasures.len(),
        pruner1.map.goal,
    );
    println!(
        "  Start: {:?} Stairs: {}",
        pruner1.map.start,
        pruner1.map.stairs.len(),
    );
    println!();

    let initial1 = pruner1.initial_state();
    print_all_floors(&pruner1, &initial1);
    println!();

    let start = std::time::Instant::now();
    let solution1 = solve_multifloor(&pruner1);
    let elapsed1 = start.elapsed();

    match solution1 {
        Some(actions) => {
            println!(
                "🎉 Dungeon 1 solved in {} steps ({:.2?})",
                actions.len(),
                elapsed1,
            );
            println!();
            print_solution(&pruner1, &actions);
            println!();

            let state = pruner1.replay_state(&actions).unwrap();
            print_all_floors(&pruner1, &state);
            println!();

            // Assertions
            assert_eq!(
                (state.floor, state.r, state.c),
                pruner1.map.goal,
                "Player must be at goal",
            );
            let all_treasures = (1 << pruner1.map.treasures.len()) - 1;
            assert_eq!(
                state.collected_treasures, all_treasures,
                "All treasures collected",
            );
            println!("✅ Dungeon 1 verified: at goal, all treasures collected.");
        }
        None => {
            println!("❌ Dungeon 1: No solution found.");
        }
    }

    println!();

    // ═══════════════════════════════════════════════════════════
    //  Dungeon 2: Three-Floor (F1 → F2 → F3)
    // ═══════════════════════════════════════════════════════════
    println!("🏰 Dungeon 2: Three-Floor (F1 → F2 → F3)");
    println!();

    let map2 = DungeonMap::new(
        &[DUNGEON2_FLOOR0, DUNGEON2_FLOOR1, DUNGEON2_FLOOR2],
        dungeon2_stairs(),
    );
    let pruner2 = DungeonPruner::new(map2);

    println!(
        "  Floors: {} Monsters: {} Treasures: {} Goal: {:?}",
        pruner2.map.floors.len(),
        pruner2.map.monsters.len(),
        pruner2.map.treasures.len(),
        pruner2.map.goal,
    );
    println!(
        "  Start: {:?} Stairs: {}",
        pruner2.map.start,
        pruner2.map.stairs.len(),
    );
    println!();

    let initial2 = pruner2.initial_state();
    print_all_floors(&pruner2, &initial2);
    println!();

    let start = std::time::Instant::now();
    let solution2 = solve_multifloor(&pruner2);
    let elapsed2 = start.elapsed();

    match solution2 {
        Some(actions) => {
            println!(
                "🎉 Dungeon 2 solved in {} steps ({:.2?})",
                actions.len(),
                elapsed2,
            );
            println!();
            print_solution(&pruner2, &actions);
            println!();

            let state = pruner2.replay_state(&actions).unwrap();
            print_all_floors(&pruner2, &state);
            println!();

            assert_eq!(
                (state.floor, state.r, state.c),
                pruner2.map.goal,
                "Player must be at goal",
            );
            let all_treasures = (1 << pruner2.map.treasures.len()) - 1;
            assert_eq!(
                state.collected_treasures, all_treasures,
                "All treasures collected",
            );
            println!("✅ Dungeon 2 verified: at goal, all treasures collected.");
        }
        None => {
            println!("❌ Dungeon 2: No solution found.");
        }
    }
}
