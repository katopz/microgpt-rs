//! Procedural Map Generation & Solving
//!
//! Generates random maps with MapGenerator and solves them with the strategic solver.
//! Demonstrates end-to-end procedural generation → validation → solving pipeline.
//!
//! Run: `cargo run --example tactical_procedural`

use std::collections::HashSet;

use microgpt_rs::pruners::map_generator::{GeneratedDungeon, MapGenerator};
use microgpt_rs::pruners::pathfinder::{Target, enumerate_targets, find_path};
use microgpt_rs::pruners::tactical_pruner::{GameState, TacticalPruner};
use microgpt_rs::speculative::types::ConstraintPruner;
use microgpt_rs::speculative::{build_dd_tree_pruned, extract_parent_tokens};
use microgpt_rs::types::Config;

// ── Strategic Pruner (local copy from tactical_bench.rs) ───────

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

// ── Solve Result ───────────────────────────────────────────────

struct SolveResult {
    solvable: bool,
    steps: Option<usize>,
    elapsed_us: u128,
    cost: Option<u32>,
    nodes: usize,
}

// ── Solver ─────────────────────────────────────────────────────

fn solve_strategic(pruner: &TacticalPruner) -> SolveResult {
    let state = pruner.initial_state();
    let strategic = StrategicPruner::new(pruner);
    let num_targets = strategic.targets.len();

    let mut config = Config::draft();
    config.vocab_size = num_targets;
    config.draft_lookahead = num_targets;
    config.tree_budget = 10000;

    let marginals = vec![vec![1.0f32 / num_targets as f32; num_targets]; config.draft_lookahead];
    let refs: Vec<&[f32]> = marginals.iter().map(|v| v.as_slice()).collect();

    let start = std::time::Instant::now();
    let tree = build_dd_tree_pruned(&refs, &config, &strategic, false);
    let elapsed = start.elapsed();

    let mut steps = None;
    let mut cost = None;
    for node in &tree {
        let target_seq = extract_parent_tokens(node.parent_path, node.depth + 1);
        if let Some(final_state) = strategic.replay_targets(&target_seq, &state)
            && (final_state.r, final_state.c) == pruner.goal
        {
            let all_treasures = (1 << pruner.treasures.len()) - 1;
            if final_state.collected_treasures == all_treasures
                && final_state.killed_monsters == all_treasures
            {
                cost = Some(final_state.total_cost);
                if let Some(actions) = expand_targets_to_actions(pruner, &target_seq) {
                    steps = Some(actions.len());
                }
                break;
            }
        }
    }

    SolveResult {
        solvable: steps.is_some(),
        steps,
        elapsed_us: elapsed.as_micros(),
        cost,
        nodes: tree.len(),
    }
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

// ── Multi-Floor Stats ──────────────────────────────────────────

struct DungeonStats {
    total_monsters: usize,
    total_treasures: usize,
    total_walls: usize,
    total_tiles: usize,
    floors: Vec<FloorStats>,
}

struct FloorStats {
    floor_idx: usize,
    monsters: usize,
    treasures: usize,
    walls: usize,
    size: (usize, usize),
}

impl DungeonStats {
    fn from_dungeon(dungeon: &GeneratedDungeon) -> Self {
        let mut total_monsters = 0;
        let mut total_treasures = 0;
        let mut total_walls = 0;
        let mut total_tiles = 0;
        let mut floors = Vec::new();

        for (idx, floor) in dungeon.map.floors.iter().enumerate() {
            let walls = floor
                .grid
                .iter()
                .flat_map(|row| row.iter())
                .filter(|&&c| c == '#')
                .count();
            let tiles = floor.grid.iter().map(|row| row.len()).sum::<usize>();

            total_monsters += floor.monsters.len();
            total_treasures += floor.treasures.len();
            total_walls += walls;
            total_tiles += tiles;

            floors.push(FloorStats {
                floor_idx: idx,
                monsters: floor.monsters.len(),
                treasures: floor.treasures.len(),
                walls,
                size: (floor.grid.len(), floor.grid[0].len()),
            });
        }

        DungeonStats {
            total_monsters,
            total_treasures,
            total_walls,
            total_tiles,
            floors,
        }
    }
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     Procedural Map Generation & Solving Pipeline        ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // ── Part 1: Single-Floor Generation + Solving ─────────────
    println!("━━━ Part 1: Single-Floor Maps (Seeds 1..=10) ━━━━━━━━━━━━");
    println!(
        "{:<6} {:<10} {:<8} {:<8} {:<12} {:<8} {:<8}",
        "Seed", "Solvable", "Steps", "Cost", "Time", "Nodes", "Status"
    );
    println!("{}", "─".repeat(68));

    let mut single_results = Vec::new();
    let mut example_map_str = String::new();
    let mut example_seed = 0u64;

    for seed in 1..=10u64 {
        let mut generator = MapGenerator::new(seed)
            .with_monsters(2)
            .with_treasures(2)
            .with_wall_density(0.15);

        let map = match generator.generate_single_floor() {
            Some(m) => m,
            None => {
                println!(
                    "{:<6} {:<10} {:<8} {:<8} {:<12} {:<8} ❌ gen failed",
                    seed, "—", "—", "—", "—", "—"
                );
                single_results.push(SolveResult {
                    solvable: false,
                    steps: None,
                    elapsed_us: 0,
                    cost: None,
                    nodes: 0,
                });
                continue;
            }
        };

        let map_str = map.to_map_string();
        let pruner = TacticalPruner::new(&map_str);
        let result = solve_strategic(&pruner);

        let status = match result.solvable {
            true => "✅",
            false => "❌",
        };
        let steps_str = match result.steps {
            Some(s) => format!("{s}"),
            None => "—".into(),
        };
        let cost_str = match result.cost {
            Some(c) => format!("{c}"),
            None => "—".into(),
        };
        let elapsed = if result.elapsed_us > 1000 {
            format!("{:.2}ms", result.elapsed_us as f64 / 1000.0)
        } else {
            format!("{}µs", result.elapsed_us)
        };

        println!(
            "{:<6} {:<10} {:<8} {:<8} {:<12} {:<8} {}",
            seed, result.solvable, steps_str, cost_str, elapsed, result.nodes, status
        );

        // Save first solvable map as example
        if result.solvable && example_map_str.is_empty() {
            example_map_str = map_str.clone();
            example_seed = seed;
        }

        single_results.push(result);
    }

    println!();

    // ── Part 2: Multi-Floor Dungeons ──────────────────────────
    println!("━━━ Part 2: Multi-Floor Dungeons (Seeds 101..=105) ━━━━━━");
    let mut dungeon_successes = 0usize;
    let mut dungeon_stats = Vec::new();

    for seed in 101..=105u64 {
        let num_floors = match seed % 2 {
            0 => 2,
            _ => 3,
        };

        let mut generator = MapGenerator::new(seed)
            .with_width(6)
            .with_height(6)
            .with_monsters(1)
            .with_treasures(1)
            .with_wall_density(0.15);

        match generator.generate_multi_floor(num_floors) {
            Some(dungeon) => {
                dungeon_successes += 1;
                let stats = DungeonStats::from_dungeon(&dungeon);
                let wall_pct = (stats.total_walls as f64 / stats.total_tiles as f64) * 100.0;
                println!(
                    "  Seed {seed}: {num_floors} floors, {} monsters, {} treasures, {:.1}% walls ✅",
                    stats.total_monsters, stats.total_treasures, wall_pct
                );

                for floor in &stats.floors {
                    println!(
                        "    Floor {}: {}×{}, {}M {}T {}#",
                        floor.floor_idx,
                        floor.size.0,
                        floor.size.1,
                        floor.monsters,
                        floor.treasures,
                        floor.walls
                    );
                }

                dungeon_stats.push(stats);
            }
            None => {
                println!("  Seed {seed}: {num_floors} floors — generation failed ❌");
            }
        }
    }

    println!();

    // ── Part 3: Example Map ASCII Art ─────────────────────────
    if !example_map_str.is_empty() {
        println!("━━━ Example Generated Map (Seed {example_seed}) ━━━━━━━━━━━━");
        for line in example_map_str.lines() {
            println!("  {line}");
        }
        println!();
        println!("  Legend: B=Start  M=Monster  T=Treasure  G=Goal  #=Wall  .=Floor");
    } else {
        println!("━━━ No solvable map found for display ━━━━━━━━━━━━━━━━━━━━");
    }

    println!();

    // ── Part 4: Summary Statistics ────────────────────────────
    println!("━━━ Summary Statistics ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let total = single_results.len();
    let solved = single_results.iter().filter(|r| r.solvable).count();
    let solvability = (solved as f64 / total as f64) * 100.0;

    let avg_steps = {
        let solved_steps: Vec<usize> = single_results.iter().filter_map(|r| r.steps).collect();
        match solved_steps.is_empty() {
            true => 0.0,
            false => solved_steps.iter().sum::<usize>() as f64 / solved_steps.len() as f64,
        }
    };

    let avg_time_us = {
        let solved_times: Vec<u128> = single_results
            .iter()
            .filter(|r| r.solvable)
            .map(|r| r.elapsed_us)
            .collect();
        match solved_times.is_empty() {
            true => 0.0,
            false => solved_times.iter().sum::<u128>() as f64 / solved_times.len() as f64,
        }
    };

    let avg_time_display = if avg_time_us > 1000.0 {
        format!("{:.2}ms", avg_time_us / 1000.0)
    } else {
        format!("{:.0}µs", avg_time_us)
    };

    let avg_cost = {
        let costs: Vec<u32> = single_results.iter().filter_map(|r| r.cost).collect();
        match costs.is_empty() {
            true => 0.0,
            false => costs.iter().sum::<u32>() as f64 / costs.len() as f64,
        }
    };

    println!("  Single-Floor Maps Generated : {total}");
    println!("  Solvable                    : {solved}/{total} ({solvability:.0}%)");
    println!("  Average Steps (solved)      : {avg_steps:.1}");
    println!("  Average Cost (solved)       : {avg_cost:.1}");
    println!("  Average Solve Time          : {avg_time_display}");
    println!("  Multi-Floor Dungeons OK     : {dungeon_successes}/5");
    println!();

    // ── Assertions ────────────────────────────────────────────
    println!("━━━ Assertions ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let solvability_ok = solvability >= 50.0;
    println!(
        "  Solvability ≥ 50%           : {} ({solvability:.0}%)",
        match solvability_ok {
            true => "✅ PASS",
            false => "❌ FAIL",
        }
    );

    let dungeon_ok = dungeon_successes >= 3;
    println!(
        "  Multi-floor ≥ 3/5           : {} ({dungeon_successes}/5)",
        match dungeon_ok {
            true => "✅ PASS",
            false => "❌ FAIL",
        }
    );

    assert!(
        solvability_ok,
        "At least 50% of single-floor maps should be solvable, got {solvability:.0}%"
    );
    assert!(
        dungeon_ok,
        "At least 3/5 multi-floor dungeons should generate, got {dungeon_successes}/5"
    );

    println!();
    println!("All assertions passed! ✅");
}
