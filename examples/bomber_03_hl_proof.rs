//! Bomberman HL Proof — 1000-Round Tournament (Plan 033, Task 8)
//!
//! Runs 1000-round tournament with 4 players at different HL tech levels.
//! After every 100 rounds, P4 runs absorb-compress cycle.
//! Prints comparison table: survival rate, kill count, avg score, powerup efficiency.
//!
//! Expected: P4 (🐵 HL) > P3 (🐶 Validator) > P2 (🐱 Greedy) > P1 (🐰 Random)
//!
//! Run: `cargo run --example bomber_03_hl_proof --features bomber`

use std::collections::HashMap;

use fastrand::Rng;

use microgpt_rs::pruners::bomber::{
    BomberAction, BomberPlayer, GameEvent, GreedyPlayer, GridPos, HLPlayer, RandomPlayer,
    ValidatorPlayer, init_world, run_tick, spawn_players,
};

// ── Config ─────────────────────────────────────────────────────

const ROUNDS: usize = 1000;
const TICK_LIMIT: u32 = 200;
const COMPRESS_INTERVAL: usize = 100;
const TOP_TRACES: usize = 10;

// ── Stats ──────────────────────────────────────────────────────

#[derive(Clone, Default)]
struct PlayerStats {
    survival_count: u32,
    kill_count: u32,
    death_count: u32,
    _suicide_count: u32,
    powerup_count: u32,
    total_score: i64,
    rounds_played: u32,
}

impl PlayerStats {
    fn survival_rate(&self) -> f32 {
        if self.rounds_played == 0 {
            return 0.0;
        }
        self.survival_count as f32 / self.rounds_played as f32
    }

    fn avg_score(&self) -> f32 {
        if self.rounds_played == 0 {
            return 0.0;
        }
        self.total_score as f32 / self.rounds_played as f32
    }

    fn avg_kills(&self) -> f32 {
        if self.rounds_played == 0 {
            return 0.0;
        }
        self.kill_count as f32 / self.rounds_played as f32
    }

    fn powerup_efficiency(&self) -> f32 {
        if self.rounds_played == 0 {
            return 0.0;
        }
        self.powerup_count as f32 / self.rounds_played as f32
    }
}

// ── Trace Record ───────────────────────────────────────────────

struct RoundTrace {
    round: usize,
    scores: [i32; 4],
    _survivors: Vec<u8>,
    ticks: u32,
    p4_actions: Vec<BomberAction>,
    p4_survived: bool,
}

// ── Round Result ───────────────────────────────────────────────

struct RoundResult {
    scores: [i32; 4],
    survivors: Vec<u8>,
    deaths: Vec<u8>,
    kills: Vec<(u8, u8)>,
    powerups: Vec<(u8, u32)>,
    ticks: u32,
    p4_actions: Vec<BomberAction>,
}

fn run_round(seed: u64, players: &mut [Box<dyn BomberPlayer>], rng: &mut Rng) -> RoundResult {
    let mut world = init_world(seed);
    let entities = spawn_players(&mut world);

    for p in players.iter_mut() {
        p.reset();
    }

    let mut all_events: Vec<GameEvent> = Vec::new();
    let mut p4_actions: Vec<BomberAction> = Vec::new();

    for _tick in 0..TICK_LIMIT {
        // Drain previous events
        {
            use bevy_ecs::event::Events;
            let mut ev = world.resource_mut::<Events<GameEvent>>();
            all_events.extend(ev.drain().collect::<Vec<GameEvent>>());
        }

        // Each player selects an action
        let mut actions = [None; 4];
        for (i, player) in players.iter_mut().enumerate() {
            let pos = world
                .get::<GridPos>(entities[i])
                .copied()
                .unwrap_or_default();
            let alive = world
                .get::<microgpt_rs::pruners::bomber::Alive>(entities[i])
                .is_some();
            if alive {
                let grid = world
                    .resource::<microgpt_rs::pruners::bomber::ArenaGrid>()
                    .clone();
                let action = player.select_action(&grid, pos, &all_events, rng);
                if i == 3 {
                    p4_actions.push(action);
                }
                actions[i] = Some(action);
            }
        }

        let ongoing = run_tick(&mut world, actions);
        if !ongoing {
            break;
        }
    }

    // Drain remaining events
    {
        use bevy_ecs::event::Events;
        let mut ev = world.resource_mut::<Events<GameEvent>>();
        all_events.extend(ev.drain().collect::<Vec<GameEvent>>());
    }

    // Compute scores from events
    let mut scores = [0i32; 4];
    let mut deaths = Vec::new();
    let mut kills = Vec::new();
    let mut powerups = Vec::new();
    let mut survivors = Vec::new();

    for event in &all_events {
        match event {
            GameEvent::PlayerKilled { victim, killer } => {
                deaths.push(*victim);
                scores[*victim as usize] -= 3;
                match killer {
                    Some(k) if *k != *victim => {
                        kills.push((*k, *victim));
                        scores[*k as usize] += 3;
                    }
                    _ => {
                        // Suicide (killer == victim or killer unknown)
                        scores[*victim as usize] -= 2;
                    }
                }
            }
            GameEvent::PowerUpCollected { player, .. } => {
                scores[*player as usize] += 1;
                powerups.push((*player, 1));
            }
            GameEvent::RoundEnd { survivors: s } => {
                survivors = s.clone();
            }
            _ => {}
        }
    }

    // Winner / timeout bonus
    if survivors.len() == 1 {
        scores[survivors[0] as usize] += 5;
    } else if survivors.len() > 1 {
        for &s in &survivors {
            scores[s as usize] += 3;
        }
    }

    let ticks = world
        .resource::<microgpt_rs::pruners::bomber::TickCounter>()
        .tick;

    RoundResult {
        scores,
        survivors,
        deaths,
        kills,
        powerups,
        ticks,
        p4_actions,
    }
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    let mut rng = Rng::with_seed(42);
    let mut players: Vec<Box<dyn BomberPlayer>> = vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(ValidatorPlayer::new(2)),
        Box::new(HLPlayer::new(3)),
    ];

    println!("╔═══ Bomberman HL Proof — {ROUNDS}-Round Tournament ═══════════════════╗");
    println!("║  P1 🐰 Random  |  P2 🐱 Greedy  |  P3 🐶 Validator  |  P4 🐵 HL  ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
    println!();

    let mut stats: Vec<PlayerStats> = vec![PlayerStats::default(); 4];
    let mut traces: Vec<RoundTrace> = Vec::new();
    let mut _p4_survived_count: u32 = 0;
    let mut _p3_survived_count: u32 = 0;

    for round in 0..ROUNDS {
        let seed = 42 + round as u64;
        let result = run_round(seed, &mut players, &mut rng);

        // Update stats
        for (i, s) in result.scores.iter().enumerate() {
            stats[i].total_score += *s as i64;
            stats[i].rounds_played += 1;
        }

        for &victim in &result.deaths {
            stats[victim as usize].death_count += 1;
        }

        for (killer, _victim) in &result.kills {
            stats[*killer as usize].kill_count += 1;
        }

        for (player, count) in &result.powerups {
            stats[*player as usize].powerup_count += count;
        }

        for &s in &result.survivors {
            stats[s as usize].survival_count += 1;
        }

        // Track P4 vs P3 survival
        if result.survivors.contains(&3) {
            _p4_survived_count += 1;
        }
        if result.survivors.contains(&2) {
            _p3_survived_count += 1;
        }

        // Update HL player with outcome
        let p4_survived = result.survivors.contains(&3);
        let p4_killed = result.kills.iter().any(|(k, _)| *k == 3);
        let p4_pu_count = result.powerups.iter().filter(|(p, _)| *p == 3).count();
        if let Some(hl) = players[3].as_any_mut().downcast_mut::<HLPlayer>() {
            hl.update_outcome(p4_survived, p4_killed, p4_pu_count as u32);
        }

        // Store trace
        let trace = RoundTrace {
            round,
            scores: result.scores,
            _survivors: result.survivors.clone(),
            ticks: result.ticks,
            p4_actions: result.p4_actions,
            p4_survived,
        };
        traces.push(trace);

        // Absorb-compress cycle every 100 rounds
        if (round + 1) % COMPRESS_INTERVAL == 0 {
            if let Some(hl) = players[3].as_any_mut().downcast_mut::<HLPlayer>() {
                let compressed = hl.compress_cycle();
                if !compressed.is_empty() {
                    println!(
                        "  [Round {}/{}] P4 compressed {} arms: {:?}",
                        round + 1,
                        ROUNDS,
                        compressed.len(),
                        compressed,
                    );
                }
            }
        }

        // Progress every 200 rounds
        if (round + 1) % 200 == 0 {
            let emoji = ["🐰", "🐱", "🐶", "🐵"];
            let names = ["Random", "Greedy", "Validator", "HL"];
            println!("  [Round {}/{}]", round + 1, ROUNDS);
            for i in 0..4 {
                println!(
                    "    {} {:<10} survival={:.1}%  avg_score={:+.1}  kills={:.1}/round",
                    emoji[i],
                    names[i],
                    stats[i].survival_rate() * 100.0,
                    stats[i].avg_score(),
                    stats[i].avg_kills(),
                );
            }
            println!();
        }
    }

    // ── Final Results ──────────────────────────────────────────────

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  FINAL RESULTS ({ROUNDS} rounds)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let emoji = ["🐰", "🐱", "🐶", "🐵"];
    let names = ["Random", "Greedy", "Validator", "HL"];
    let tech = ["(baseline)", "(heuristic)", "(+validator)", "(+bandit+AC)"];

    println!(
        "  {:<4} {:<10} {:<14} {:>10} {:>10} {:>12} {:>10} {:>10}",
        "", "Player", "Tech", "Survival%", "AvgScore", "Kills/Round", "Deaths", "PU/Round"
    );
    println!("  {}", "─".repeat(80));

    // Sort by survival rate descending
    let mut ranking: Vec<usize> = (0..4).collect();
    ranking.sort_by(|&a, &b| {
        stats[b]
            .survival_rate()
            .partial_cmp(&stats[a].survival_rate())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (rank, &idx) in ranking.iter().enumerate() {
        println!(
            "  #{:<3} {} {:<10} {:<14} {:>9.1}% {:>+9.1} {:>11.2} {:>10} {:>9.2}",
            rank + 1,
            emoji[idx],
            names[idx],
            tech[idx],
            stats[idx].survival_rate() * 100.0,
            stats[idx].avg_score(),
            stats[idx].avg_kills(),
            stats[idx].death_count,
            stats[idx].powerup_efficiency(),
        );
    }

    // ── The Key Proof: P3 vs P4 ───────────────────────────────────

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  THE KEY PROOF: P3 (🐶) vs P4 (🐵)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!(
        "  P3 survival: {:.1}%  ({}/{})",
        stats[2].survival_rate() * 100.0,
        stats[2].survival_count,
        stats[2].rounds_played
    );
    println!(
        "  P4 survival: {:.1}%  ({}/{})",
        stats[3].survival_rate() * 100.0,
        stats[3].survival_count,
        stats[3].rounds_played
    );

    let delta = (stats[3].survival_rate() - stats[2].survival_rate()) * 100.0;
    let proof = delta > 5.0;
    println!("  Delta: {delta:+.1}pp");
    println!();

    if proof {
        println!("  ✅ PROVEN: P4 (Full HL) outperforms P3 (Static Validator)");
        println!("     The bandit's ability to adapt relevance based on observed");
        println!("     outcomes makes the validator more valuable than static rules.");
    } else if delta > 0.0 {
        println!("  ⚠️  MARGINAL: P4 slightly outperforms P3 ({delta:+.1}pp)");
        println!("     Consider increasing rounds or tuning HL parameters.");
    } else {
        println!("  ❌ NOT PROVEN: P4 did not outperform P3");
        println!("     The bandit may need more episodes or different strategies.");
    }

    // ── Golden Traces ──────────────────────────────────────────────

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  GOLDEN TRACES (Top {TOP_TRACES} P4 episodes)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Filter and sort P4's best episodes (survived, highest score, fewest ticks)
    let mut p4_best: Vec<&RoundTrace> = traces.iter().filter(|t| t.p4_survived).collect();
    p4_best.sort_by(|a, b| {
        b.scores[3]
            .cmp(&a.scores[3])
            .then_with(|| a.ticks.cmp(&b.ticks))
    });

    for (i, trace) in p4_best.iter().take(TOP_TRACES).enumerate() {
        let action_summary = summarize_actions(&trace.p4_actions);
        println!(
            "  #{} Round {:>4}  Score={:+4}  Ticks={:>3}  Actions=[{}]",
            i + 1,
            trace.round + 1,
            trace.scores[3],
            trace.ticks,
            action_summary,
        );
    }

    if p4_best.is_empty() {
        println!("  (no P4 survival episodes recorded)");
    }

    // ── Compression Summary ────────────────────────────────────────

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  COMPRESSION EVIDENCE");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    if let Some(hl) = players[3].as_any().downcast_ref::<HLPlayer>() {
        let report = hl.compress_report();
        println!("  {report}");
    } else {
        println!("  (P4 not available for report)");
    }

    println!();
}

// ── Helpers ─────────────────────────────────────────────────────

fn summarize_actions(actions: &[BomberAction]) -> String {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for a in actions {
        *counts.entry(format!("{a}")).or_default() += 1;
    }
    let mut pairs: Vec<_> = counts.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1));
    pairs
        .iter()
        .map(|(k, v)| format!("{k}×{v}"))
        .collect::<Vec<_>>()
        .join(" ")
}
