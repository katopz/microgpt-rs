//! Tactical AI — Parallel Batch Solving with Rayon
//!
//! Demonstrates parallel speedup by solving N procedurally generated maps
//! both sequentially and in parallel using rayon::par_iter.
//!
//! Run: `cargo run --example tactical_parallel`

use std::collections::HashSet;

use microgpt_rs::pruners::map_generator::MapGenerator;
use microgpt_rs::pruners::pathfinder::{Target, enumerate_targets, find_path};
use microgpt_rs::pruners::tactical_pruner::{GameState, TacticalPruner};
use microgpt_rs::speculative::types::ConstraintPruner;
use microgpt_rs::speculative::{build_dd_tree_pruned, extract_parent_tokens};
use microgpt_rs::types::Config;
use rayon::prelude::*;

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

// ── Standalone Solver ──────────────────────────────────────────

/// Solves a single map string using the strategic approach.
///
/// Returns `Some((actions, elapsed_us))` if solved, `None` otherwise.
fn solve_one(map_str: &str) -> Option<(Vec<usize>, u128)> {
    let pruner = TacticalPruner::new(map_str);
    let state = pruner.initial_state();
    let strategic = StrategicPruner::new(&pruner);
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

    for node in &tree {
        let target_seq = extract_parent_tokens(node.parent_path, node.depth + 1);
        if let Some(final_state) = strategic.replay_targets(&target_seq, &state)
            && (final_state.r, final_state.c) == pruner.goal
        {
            let all_treasures = (1 << pruner.treasures.len()) - 1;
            if final_state.collected_treasures == all_treasures
                && final_state.killed_monsters == all_treasures
            {
                let actions = expand_targets_to_actions(&pruner, &target_seq)?;
                return Some((actions, elapsed.as_micros()));
            }
        }
    }

    None
}

/// Expands a target sequence into concrete action steps.
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

// ── Map Generation ─────────────────────────────────────────────

/// Generates N procedural maps using MapGenerator with seeds 1..=n.
fn generate_maps(n: u64) -> Vec<(u64, String)> {
    let mut maps = Vec::with_capacity(n as usize);
    for seed in 1..=n {
        let mut generator = MapGenerator::new(seed)
            .with_width(10)
            .with_height(10)
            .with_monsters(3)
            .with_treasures(2)
            .with_wall_density(0.15);
        match generator.generate_single_floor() {
            Some(map) => maps.push((seed, map.to_map_string())),
            None => println!("   ⚠ Seed {seed} generated unsolvable map, skipping"),
        }
    }
    maps
}

// ── Formatting Helpers ─────────────────────────────────────────

fn format_duration(us: u128) -> String {
    match us {
        0..=999 => format!("{us}µs"),
        1_000..=999_999 => format!("{:.2}ms", us as f64 / 1000.0),
        _ => format!("{:.2}s", us as f64 / 1_000_000.0),
    }
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          Tactical AI — Parallel Batch Solving (Rayon)          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let num_maps: u64 = 15;

    // ── 1. Generate procedural maps ────────────────────────────
    println!("🗺  Generating {num_maps} procedural maps (seeds 1..={num_maps})...");
    let maps = generate_maps(num_maps);
    let actual_count = maps.len();
    println!("   ✓ {actual_count} solvable maps generated");
    println!();

    if maps.is_empty() {
        println!("❌ No solvable maps generated. Try different parameters.");
        return;
    }

    // ── 2. Sequential solve ────────────────────────────────────
    println!("🔄 Solving sequentially...");
    let seq_start = std::time::Instant::now();
    let sequential_results: Vec<(Option<Vec<usize>>, u128)> = maps
        .iter()
        .map(|(_, map_str)| match solve_one(map_str) {
            Some((actions, elapsed)) => (Some(actions), elapsed),
            None => (None, 0),
        })
        .collect();
    let seq_total = seq_start.elapsed().as_micros();
    println!("   ✓ Sequential done in {}", format_duration(seq_total));
    println!();

    // ── 3. Parallel solve ──────────────────────────────────────
    println!("⚡ Solving in parallel with rayon...");
    let par_start = std::time::Instant::now();
    let parallel_results: Vec<(Option<Vec<usize>>, u128)> = maps
        .par_iter()
        .map(|(_, map_str)| match solve_one(map_str) {
            Some((actions, elapsed)) => (Some(actions), elapsed),
            None => (None, 0),
        })
        .collect();
    let par_total = par_start.elapsed().as_micros();
    println!("   ✓ Parallel done in {}", format_duration(par_total));
    println!();

    // ── 4. Verify results match ────────────────────────────────
    let mut all_match = true;
    for (i, ((seq_res, _), (par_res, _))) in sequential_results
        .iter()
        .zip(parallel_results.iter())
        .enumerate()
    {
        let seq_solved = seq_res.is_some();
        let par_solved = par_res.is_some();
        if seq_solved != par_solved {
            println!(
                "   ❌ Map {} (seed {}): sequential={}, parallel={}",
                i + 1,
                maps[i].0,
                seq_solved,
                par_solved,
            );
            all_match = false;
        }
    }
    if all_match {
        println!("✅ All sequential and parallel results match");
    }
    println!();

    // ── 5. Print results table ─────────────────────────────────
    println!("┌──────┬──────┬────────────┬────────────┬────────┬───────┐");
    println!("│ Map  │ Seed │ Sequential │  Parallel  │ Solved │ Steps │");
    println!("├──────┼──────┼────────────┼────────────┼────────┼───────┤");

    for (i, ((seq_res, seq_time), (_par_res, par_time))) in sequential_results
        .iter()
        .zip(parallel_results.iter())
        .enumerate()
    {
        let seed = maps[i].0;
        let solved = match seq_res {
            Some(_) => "  ✅  ",
            None => "  ❌  ",
        };
        let steps = match seq_res {
            Some(actions) => format!("{:>5}", actions.len()),
            None => "   —".into(),
        };
        let seq_str = format_duration(*seq_time);
        let par_str = format_duration(*par_time);

        println!(
            "│ {:>4} │ {:>4} │ {:>10} │ {:>10} │ {} │ {} │",
            i + 1,
            seed,
            seq_str,
            par_str,
            solved,
            steps,
        );
    }

    println!("└──────┴──────┴────────────┴────────────┴────────┴───────┘");
    println!();

    // ── 6. Summary ─────────────────────────────────────────────
    let seq_solved_count = sequential_results
        .iter()
        .filter(|(res, _)| res.is_some())
        .count();
    let par_solved_count = parallel_results
        .iter()
        .filter(|(res, _)| res.is_some())
        .count();

    let speedup = match par_total {
        0 => 0.0,
        _ => seq_total as f64 / par_total as f64,
    };

    let solvability = (seq_solved_count as f64 / actual_count as f64) * 100.0;

    println!("📊 Summary:");
    println!("   • Sequential total time: {}", format_duration(seq_total));
    println!("   • Parallel total time:   {}", format_duration(par_total));
    println!("   • Speedup factor:        {speedup:.2}x");
    println!("   • Solvability rate:      {seq_solved_count}/{actual_count} ({solvability:.0}%)");
    println!();

    // ── 7. Assertions ──────────────────────────────────────────
    assert!(all_match, "Sequential and parallel results must match");

    assert!(
        seq_solved_count > 0,
        "At least some maps must be solvable (got {seq_solved_count}/{actual_count})",
    );

    assert_eq!(
        seq_solved_count, par_solved_count,
        "Sequential and parallel solve counts must match",
    );

    println!("✅ All assertions passed.");
}
