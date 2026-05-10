//! Tactical AI Benchmark — Strategic vs Brute-Force DDTree
//!
//! Compares two solving approaches on multiple maps:
//! - **Brute-force**: DDTree on micro-actions (vocab=5: Up/Down/Left/Right/Attack)
//! - **Strategic**: DDTree on target tokens (vocab=N targets) + A* path expansion
//!
//! Key insight: Brute-force scales as O(5^steps) — infeasible beyond ~8 steps.
//! Strategic scales as O(N! targets) + O(map_size) per A* — works for 100+ steps.
//!
//! Run: `cargo run --example tactical_05_bench`

use std::collections::HashSet;

use microgpt_rs::pruners::pathfinder::{Target, enumerate_targets, find_path};
use microgpt_rs::pruners::tactical_pruner::{GameState, TacticalPruner};
use microgpt_rs::speculative::types::ConstraintPruner;
use microgpt_rs::speculative::{build_dd_tree_pruned, extract_parent_tokens};
use microgpt_rs::types::Config;

// ── Strategic Pruner (same as tactical_ai.rs) ──────────────────

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
        if let Target::Monster(i) = target {
            blocked.remove(&self.tactical.monsters[*i]);
        }
        if let Target::Goal = target {
            blocked.remove(&self.tactical.goal);
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

// ── Benchmark Results ──────────────────────────────────────────

struct BenchResult {
    name: String,
    approach: String,
    map_size: String,
    targets: usize,
    steps: Option<usize>,
    nodes: usize,
    elapsed_us: u128,
    solved: bool,
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.solved { "✅" } else { "❌" };
        let steps_str = match self.steps {
            Some(s) => format!("{s}"),
            None => "—".into(),
        };
        let elapsed = if self.elapsed_us > 1000 {
            format!("{:.2}ms", self.elapsed_us as f64 / 1000.0)
        } else {
            format!("{}µs", self.elapsed_us)
        };
        write!(
            f,
            "{:<20} {:<14} {:<8} {:<8} {:<8} {:<10} {}",
            self.name, self.approach, self.map_size, self.targets, steps_str, elapsed, status
        )
    }
}

// ── Solvers ────────────────────────────────────────────────────

fn solve_bruteforce(pruner: &TacticalPruner, lookahead: usize, budget: usize) -> BenchResult {
    let mut config = Config::draft();
    config.vocab_size = 5;
    config.draft_lookahead = lookahead;
    config.tree_budget = budget;

    let marginals = vec![vec![0.2f32; 5]; config.draft_lookahead];
    let refs: Vec<&[f32]> = marginals.iter().map(|v| v.as_slice()).collect();

    let start = std::time::Instant::now();
    let tree = build_dd_tree_pruned(&refs, &config, pruner, false);
    let elapsed = start.elapsed();

    let mut steps = None;
    for node in &tree {
        let path = extract_parent_tokens(node.parent_path, node.depth + 1);
        if let Some(state) = pruner.replay_state(&path)
            && (state.r, state.c) == pruner.goal
        {
            let all_treasures = (1 << pruner.treasures.len()) - 1;
            if state.collected_treasures == all_treasures && state.killed_monsters == all_treasures
            {
                steps = Some(path.len());
                break;
            }
        }
    }

    BenchResult {
        name: String::new(),
        approach: "Brute-force".into(),
        map_size: format!("{}×{}", pruner.grid.len(), pruner.grid[0].len()),
        targets: pruner.monsters.len() + pruner.treasures.len() + 1,
        steps,
        nodes: tree.len(),
        elapsed_us: elapsed.as_micros(),
        solved: steps.is_some(),
    }
}

fn solve_strategic(pruner: &TacticalPruner) -> BenchResult {
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
    for node in &tree {
        let target_seq = extract_parent_tokens(node.parent_path, node.depth + 1);
        if let Some(final_state) = strategic.replay_targets(&target_seq, &state)
            && (final_state.r, final_state.c) == pruner.goal
        {
            let all_treasures = (1 << pruner.treasures.len()) - 1;
            if final_state.collected_treasures == all_treasures
                && final_state.killed_monsters == all_treasures
            {
                // Expand to action steps
                if let Some(actions) = expand_targets_to_actions(pruner, &target_seq) {
                    steps = Some(actions.len());
                }
                break;
            }
        }
    }

    BenchResult {
        name: String::new(),
        approach: "Strategic".into(),
        map_size: format!("{}×{}", pruner.grid.len(), pruner.grid[0].len()),
        targets: num_targets,
        steps,
        nodes: tree.len(),
        elapsed_us: elapsed.as_micros(),
        solved: steps.is_some(),
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

// ── Maps ───────────────────────────────────────────────────────

/// Small 2×3 map — both brute-force and strategic work.
/// Solution: → ⚔ ↓ ⚔ ↑ → ↓ (7 steps)
const MAP_SMALL: &str = "\
B X T
# M G";

/// Original 17×16 dungeon from tactical_ai.rs
const MAP_ORIGINAL: &str = "\
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

/// Alternative map 1: "Open Arena" — fewer walls, wider spaces.
/// 3 monsters, 3 treasures, goal. More direct paths.
const MAP_ARENA: &str = "\
# # # # # # # # # # # # # # # #
# B . . . . . . . . . . . . . #
# . . . . . . . . . . . . . . #
# . . M . . . . . . . . T . . #
# . . . . . . . . . . . . . . #
# . . . . . . . . . . . . . . #
# . . . . . . M . . . . . . . #
# . . . . . . . . . . . . . . #
# . . . . . . . . . . . . . . #
# . . . . . . . . . . . . . . #
# . . T . . . . . M . . . . . #
# . . . . . . . . . . . . . . #
# . . . . . . . . . . . . . . #
# . . . . . . . . . . . T . . #
# . . . . . . . . . . . . . G #
# # # # # # # # # # # # # # # #";

/// Alternative map 2: "Corridor Maze" — horizontal walls with gaps form corridors.
/// 3 monsters, 3 treasures, goal. Must navigate through gap openings.
/// Gaps at columns 3, 7, 11 in wall rows (3, 7, 12) create a grid of corridors.
const MAP_CORRIDOR: &str = "\
# # # # # # # # # # # # # # # #
# B . . . . . . . . . . . . . #
# . . . . . . . . . . . . . . #
# # # . # # # . # # # . # # # #
# . . . . . . . . . . . . . . #
# . . . . . . . . . . . . . . #
# . . M . . . . . . . T . . . #
# # # . # # # . # # # . # # # #
# . . . . . . . . . . . . . . #
# . . . . . . . . . . . . . . #
# . T . . . . M . . . . . M . #
# . . . . . . . . . . . . . . #
# # # . # # # . # # # . # # # #
# . . . . . . . . . . . . . . #
# . . . . . . . . T . . . . G #
# # # # # # # # # # # # # # # #";

// ── Main ───────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          Tactical AI Benchmark: Strategic vs Brute-Force       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let mut results: Vec<BenchResult> = Vec::new();

    // ── 1. Small map: both approaches ──────────────────────────
    println!("📋 Map 1: Small (2×3) — Both approaches feasible");
    let pruner_small = TacticalPruner::new(MAP_SMALL);
    println!(
        "   Monsters: {} Treasures: {} Goal: ({},{})",
        pruner_small.monsters.len(),
        pruner_small.treasures.len(),
        pruner_small.goal.0,
        pruner_small.goal.1,
    );

    // Brute-force on small map
    let mut bf_small = solve_bruteforce(&pruner_small, 8, 10000);
    bf_small.name = "Small".into();
    println!(
        "   Brute-force: {} nodes, {:.2?}, solved={}",
        bf_small.nodes,
        std::time::Duration::from_micros(bf_small.elapsed_us as u64),
        bf_small.solved
    );
    results.push(bf_small);

    // Strategic on small map
    let mut st_small = solve_strategic(&pruner_small);
    st_small.name = "Small".into();
    println!(
        "   Strategic:   {} nodes, {:.2?}, solved={}",
        st_small.nodes,
        std::time::Duration::from_micros(st_small.elapsed_us as u64),
        st_small.solved
    );
    results.push(st_small);

    println!();

    // ── 2. Original 16×16 dungeon ─────────────────────────────
    println!("📋 Map 2: Original 17×16 Dungeon");
    let pruner_original = TacticalPruner::new(MAP_ORIGINAL);
    println!(
        "   Monsters: {} Treasures: {} Goal: ({},{})",
        pruner_original.monsters.len(),
        pruner_original.treasures.len(),
        pruner_original.goal.0,
        pruner_original.goal.1,
    );

    let mut st_original = solve_strategic(&pruner_original);
    st_original.name = "Original".into();
    println!(
        "   Strategic: {} nodes, {:.2?}, {} steps, solved={}",
        st_original.nodes,
        std::time::Duration::from_micros(st_original.elapsed_us as u64),
        st_original.steps.map_or("—".into(), |s| format!("{s}")),
        st_original.solved
    );
    results.push(st_original);

    // Brute-force is infeasible on 16×16 — estimate state space
    // 5^125 = astronomical. Even 5^8 = 390,625 (DDTree max lookahead = 8)
    // The solution requires ~125 steps, far beyond brute-force reach.
    let bf_original = BenchResult {
        name: "Original".into(),
        approach: "Brute-force".into(),
        map_size: format!(
            "{}×{}",
            pruner_original.grid.len(),
            pruner_original.grid[0].len()
        ),
        targets: pruner_original.monsters.len() + pruner_original.treasures.len() + 1,
        steps: None,
        nodes: 0,
        elapsed_us: 0,
        solved: false,
    };
    println!("   Brute-force: INFEASIBLE (5^125 state space, lookahead max=8)");
    results.push(bf_original);

    println!();

    // ── 3. Arena map ──────────────────────────────────────────
    println!("📋 Map 3: Open Arena (16×16)");
    let pruner_arena = TacticalPruner::new(MAP_ARENA);
    println!(
        "   Monsters: {} Treasures: {} Goal: ({},{})",
        pruner_arena.monsters.len(),
        pruner_arena.treasures.len(),
        pruner_arena.goal.0,
        pruner_arena.goal.1,
    );

    let mut st_arena = solve_strategic(&pruner_arena);
    st_arena.name = "Arena".into();
    println!(
        "   Strategic: {} nodes, {:.2?}, {} steps, solved={}",
        st_arena.nodes,
        std::time::Duration::from_micros(st_arena.elapsed_us as u64),
        st_arena.steps.map_or("—".into(), |s| format!("{s}")),
        st_arena.solved
    );
    let arena_solved = st_arena.solved;
    results.push(st_arena);

    // Verify solution correctness if solved
    if arena_solved {
        verify_solution(&pruner_arena, "Arena");
    }

    println!();

    // ── 4. Corridor maze map ──────────────────────────────────
    println!("📋 Map 4: Corridor Maze (16×16)");
    let pruner_corridor = TacticalPruner::new(MAP_CORRIDOR);
    println!(
        "   Monsters: {} Treasures: {} Goal: ({},{})",
        pruner_corridor.monsters.len(),
        pruner_corridor.treasures.len(),
        pruner_corridor.goal.0,
        pruner_corridor.goal.1,
    );

    let mut st_corridor = solve_strategic(&pruner_corridor);
    st_corridor.name = "Corridor".into();
    println!(
        "   Strategic: {} nodes, {:.2?}, {} steps, solved={}",
        st_corridor.nodes,
        std::time::Duration::from_micros(st_corridor.elapsed_us as u64),
        st_corridor.steps.map_or("—".into(), |s| format!("{s}")),
        st_corridor.solved
    );
    let corridor_solved = st_corridor.solved;
    results.push(st_corridor);

    // Verify solution correctness if solved
    if corridor_solved {
        verify_solution(&pruner_corridor, "Corridor");
    }

    println!();

    // ── Summary Table ─────────────────────────────────────────
    println!("┌────────────────────┬──────────────┬────────┬────────┬────────┬──────────┬──────┐");
    println!("│ Map                │ Approach     │ Size   │ Targets│ Steps  │ Time     │ OK?  │");
    println!("├────────────────────┼──────────────┼────────┼────────┼────────┼──────────┼──────┤");

    for r in &results {
        let steps_str = r.steps.map_or("  —   ".into(), |s| format!("{s:>6}",));
        let elapsed = if r.elapsed_us == 0 {
            "  N/A   ".into()
        } else if r.elapsed_us > 1000 {
            format!("{:>5.1}ms", r.elapsed_us as f64 / 1000.0)
        } else {
            format!("{:>5}µs", r.elapsed_us)
        };
        let status = if r.solved { "  ✅  " } else { "  ❌  " };
        println!(
            "│ {:<18} │ {:<12} │ {:<6} │ {:<6} │ {} │ {} │ {} │",
            r.name, r.approach, r.map_size, r.targets, steps_str, elapsed, status
        );
    }

    println!("└────────────────────┴──────────────┴────────┴────────┴────────┴──────────┴──────┘");

    println!();
    println!("📊 Scaling Analysis:");
    println!("   • Brute-force DDTree: vocab=5, max lookahead=8 (u128/16 constraint)");
    println!("     → State space: 5^8 = 390,625 nodes (max)");
    println!("     → Only works for puzzles solvable in ≤8 steps");
    println!("   • Strategic DDTree: vocab=N targets, lookahead=N");
    println!("     → State space: N! permutations (7! = 5,040 for 7 targets)");
    println!("     → A* expands each target into actual movement steps");
    println!("     → Works for puzzles with 100+ steps, any map size");

    // Assertions — small map must be solvable by both
    assert!(
        results
            .iter()
            .find(|r| r.name == "Small" && r.approach == "Brute-force")
            .unwrap()
            .solved,
        "Small map must be solvable by brute-force"
    );
    assert!(
        results
            .iter()
            .find(|r| r.name == "Small" && r.approach == "Strategic")
            .unwrap()
            .solved,
        "Small map must be solvable by strategic"
    );
    assert!(
        results
            .iter()
            .find(|r| r.name == "Original" && r.approach == "Strategic")
            .unwrap()
            .solved,
        "Original 16×16 map must be solvable by strategic"
    );
    assert!(
        results
            .iter()
            .find(|r| r.name == "Arena" && r.approach == "Strategic")
            .unwrap()
            .solved,
        "Arena map must be solvable by strategic"
    );
    assert!(
        results
            .iter()
            .find(|r| r.name == "Corridor" && r.approach == "Strategic")
            .unwrap()
            .solved,
        "Corridor map must be solvable by strategic"
    );

    println!();
    println!("✅ All assertions passed. All maps verified solvable.");
}

/// Verify a strategic solution by replaying actions through TacticalPruner.
fn verify_solution(pruner: &TacticalPruner, name: &str) {
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
        if let Some(final_state) = strategic.replay_targets(&target_seq, &state)
            && (final_state.r, final_state.c) == pruner.goal
        {
            let all_treasures = (1 << pruner.treasures.len()) - 1;
            if final_state.collected_treasures == all_treasures
                && final_state.killed_monsters == all_treasures
                && let Some(actions) = expand_targets_to_actions(pruner, &target_seq)
            {
                // Replay actions through TacticalPruner for full verification
                let mut verify_state = pruner.initial_state();
                for &action in &actions {
                    verify_state = pruner.apply_action(&verify_state, action).unwrap();
                }
                assert_eq!(
                    (verify_state.r, verify_state.c),
                    pruner.goal,
                    "{name}: Bear must be at goal"
                );
                assert_eq!(
                    verify_state.collected_treasures, all_treasures,
                    "{name}: All treasures collected"
                );
                assert_eq!(
                    verify_state.killed_monsters, all_treasures,
                    "{name}: All monsters killed"
                );
                println!(
                    "   ✅ {name}: Solution verified — {} actions, all goals met",
                    actions.len()
                );
                return;
            }
        }
    }
    println!("   ❌ {name}: No valid solution found!");
}
