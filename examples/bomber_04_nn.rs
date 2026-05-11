//! Bomberman HL Arena — NNPlayer Demo (Plan 034, Task 8)
//!
//! Demonstrates NNPlayer (P2.5) with WASM validator safety checks.
//! Loads `bomber_validator.wasm` at runtime and runs a tournament
//! comparing WASM-validated vs native safety rules.
//!
//! # Run
//!
//! ```sh
//! # With WASM validator (requires bomber_validator.wasm artifact):
//! cargo run --example bomber_04_nn --features bomber-wasm -- /path/to/bomber_validator.wasm
//!
//! # Without WASM (native fallback):
//! cargo run --example bomber_04_nn --features bomber-wasm
//! ```
//!
//! # Secrets
//!
//! This example requires `bomber_validator.wasm` to use WASM validation.
//! Build it from `riir-ai`:
//! ```sh
//! cd riir-ai && cargo build --example bomber_validator --target wasm32-unknown-unknown --release
//! # Output: target/wasm32-unknown-unknown/release/examples/bomber_validator.wasm
//! ```

use std::env;
use std::time::Instant;

use fastrand::Rng;

#[cfg(feature = "bomber-wasm")]
use microgpt_rs::pruners::bomber::{
    BomberPlayer, GameEvent, GreedyPlayer, HLPlayer, NNPlayer, RandomPlayer, ValidatorPlayer,
    init_world, run_tick, spawn_players,
};

#[cfg(not(feature = "bomber-wasm"))]
fn main() {
    eprintln!("Error: This example requires the 'bomber-wasm' feature.");
    eprintln!("Run with: cargo run --example bomber_04_nn --features bomber-wasm");
    std::process::exit(1);
}

#[cfg(feature = "bomber-wasm")]
fn main() {
    let args: Vec<String> = env::args().collect();
    let wasm_path = args.get(1).map(|s| s.as_str());

    let mut rng = Rng::with_seed(42);

    // ── WASM Loading ───────────────────────────────────────────

    // Create NNPlayer — silently falls back to native if WASM fails
    let nn_player: Box<dyn BomberPlayer> = match wasm_path {
        Some(path) => {
            let start = Instant::now();
            let player = NNPlayer::new_with_wasm(2, path);
            let elapsed = start.elapsed();
            let loaded = player.name() == "NN-WASM";
            if loaded {
                println!(
                    "✅ WASM validator loaded: {path} ({:.2}ms)",
                    elapsed.as_secs_f64() * 1000.0
                );
            } else {
                eprintln!("⚠️  WASM load failed: {path}");
                eprintln!("   Falling back to native safety rules.");
            }
            Box::new(player)
        }
        None => {
            println!("ℹ️  No WASM path provided — using native safety rules.");
            Box::new(NNPlayer::new_native(2))
        }
    };
    let wasm_loaded = nn_player.name() == "NN-WASM";

    // ── Player Setup ───────────────────────────────────────────

    const ROUNDS: usize = 20;
    const TICK_LIMIT: u32 = 200;

    let mut players: Vec<Box<dyn BomberPlayer>> = vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        nn_player,
        Box::new(HLPlayer::new(3)),
    ];

    println!();
    println!("╔═══ Bomberman NNPlayer Arena ═════════════════════════════════╗");
    println!(
        "║  P1 🐰 Random  |  P2 🐱 Greedy  |  P3 🤖 NN-{}  |  P4 🐵 HL  ║",
        if wasm_loaded { "WASM" } else { "Native" }
    );
    println!("╚═════════════════════════════════════════════════════════════╝");
    println!("  Rounds: {ROUNDS}  |  Tick limit: {TICK_LIMIT}");
    println!();

    // ── Tournament ──────────────────────────────────────────────

    let mut scores = [0i32; 4];
    let mut wins = [0u32; 4];
    let mut deaths = [0u32; 4];
    let mut total_ticks = 0u32;

    for round in 0..ROUNDS {
        let seed = 42 + round as u64;
        let round_start = Instant::now();
        let result = run_round(seed, &mut players, &mut rng, TICK_LIMIT);
        let round_elapsed = round_start.elapsed();

        // Update stats
        for (i, s) in result.scores.iter().enumerate() {
            scores[i] += s;
        }
        if let Some(winner) = result.winner {
            wins[winner as usize] += 1;
        }
        for &victim in &result.deaths {
            deaths[victim as usize] += 1;
        }
        total_ticks += result.ticks;

        // Update HL player with outcome
        let survived = !result.deaths.contains(&3);
        let killed = result.kills.iter().any(|(killer, _)| *killer == 3);
        let powerups = result.powerups.iter().filter(|(p, _)| *p == 3).count();
        if let Some(hl) = players[3].as_any_mut().downcast_mut::<HLPlayer>() {
            hl.update_outcome(survived, killed, powerups as u32);
        }

        // Print round result
        let emoji = ["🐰", "🐱", "🤖", "🐵"];
        let winner_str = match result.winner {
            Some(w) => format!("{} P{}", emoji[w as usize], w),
            None => "Draw".to_string(),
        };
        println!(
            "Round {:>2}: Winner={:<12} Scores=[{}] Ticks={:>3} ({:.1}ms)",
            round + 1,
            winner_str,
            result
                .scores
                .iter()
                .enumerate()
                .map(|(i, s)| format!("{}:{:+}", emoji[i], s))
                .collect::<Vec<_>>()
                .join(" "),
            result.ticks,
            round_elapsed.as_secs_f64() * 1000.0,
        );
    }

    // ── Final Standings ─────────────────────────────────────────

    println!();
    println!("═══ Final Standings ({ROUNDS} rounds) ═══");
    let emoji = ["🐰", "🐱", "🤖", "🐵"];
    let names = [
        "Random",
        "Greedy",
        if wasm_loaded { "NN-WASM" } else { "NN-Native" },
        "HL",
    ];
    let mut ranking: Vec<(usize, i32)> = scores.iter().copied().enumerate().collect();
    ranking.sort_by(|a, b| b.1.cmp(&a.1));

    for (rank, (idx, score)) in ranking.iter().enumerate() {
        println!(
            "  #{} {} {:<10} Score={:+5}  Wins={}  Deaths={}  AvgTicks={:.0}",
            rank + 1,
            emoji[*idx],
            names[*idx],
            score,
            wins[*idx],
            deaths[*idx],
            total_ticks as f64 / ROUNDS as f64,
        );
    }

    println!();
    println!("═══ WASM Stats ═══");
    println!(
        "  Validator: {}",
        if wasm_loaded {
            "Loaded"
        } else {
            "Native fallback"
        }
    );
    println!(
        "  Avg ticks/round: {:.1}",
        total_ticks as f64 / ROUNDS as f64
    );
    println!("  Avg time/round: N/A (see individual rounds above)");

    // ── A/B Comparison (if WASM loaded) ─────────────────────────

    if wasm_loaded {
        println!();
        println!("═══ A/B Safety Comparison ═══");
        println!("  Running 5 rounds with native ValidatorPlayer (P3) for comparison...");

        // Reset HL player for fresh comparison
        let mut native_players: Vec<Box<dyn BomberPlayer>> = vec![
            Box::new(RandomPlayer::new(0)),
            Box::new(GreedyPlayer::new(1)),
            Box::new(ValidatorPlayer::new(2)),
            Box::new(HLPlayer::new(3)),
        ];

        let mut native_scores = [0i32; 4];
        let mut native_wins = [0u32; 4];

        for round in 0..5 {
            let seed = 1000 + round as u64;
            let result = run_round(seed, &mut native_players, &mut rng, TICK_LIMIT);

            for (i, s) in result.scores.iter().enumerate() {
                native_scores[i] += s;
            }
            if let Some(winner) = result.winner {
                native_wins[winner as usize] += 1;
            }

            // Update HL
            let survived = !result.deaths.contains(&3);
            let killed = result.kills.iter().any(|(killer, _)| *killer == 3);
            let powerups = result.powerups.iter().filter(|(p, _)| *p == 3).count();
            if let Some(hl) = native_players[3].as_any_mut().downcast_mut::<HLPlayer>() {
                hl.update_outcome(survived, killed, powerups as u32);
            }
        }

        println!(
            "  Native P3 (ValidatorPlayer): Score={:+5} Wins={}",
            native_scores[2], native_wins[2]
        );
        println!("  WASM   P3 (NNPlayer):        Score=N/A (see tournament above)");
        println!();
        println!("  Note: Run bomber_01_arena with same seeds for full A/B comparison.");
    }
}

// ── Round Runner ────────────────────────────────────────────────

#[cfg(feature = "bomber-wasm")]
struct RoundResult {
    scores: [i32; 4],
    winner: Option<u8>,
    deaths: Vec<u8>,
    kills: Vec<(u8, u8)>,
    powerups: Vec<(u8, u32)>,
    ticks: u32,
}

#[cfg(feature = "bomber-wasm")]
fn run_round(
    seed: u64,
    players: &mut [Box<dyn BomberPlayer>],
    rng: &mut Rng,
    tick_limit: u32,
) -> RoundResult {
    let mut world = init_world(seed);
    let entities = spawn_players(&mut world);

    // Reset players
    for p in players.iter_mut() {
        p.reset();
    }

    let mut round_events: Vec<GameEvent> = Vec::new();

    // Run tick loop
    for _tick in 0..tick_limit {
        // Drain events from previous tick
        let tick_events: Vec<GameEvent> = {
            let mut event_reader = world.resource_mut::<bevy_ecs::event::Events<GameEvent>>();
            event_reader.drain().collect()
        };
        round_events.extend(tick_events.iter().cloned());

        // Each player selects an action
        let mut actions = [None; 4];
        for (i, player) in players.iter_mut().enumerate() {
            let pos = world
                .get::<microgpt_rs::pruners::bomber::GridPos>(entities[i])
                .copied()
                .unwrap_or_default();
            let alive = world
                .get::<microgpt_rs::pruners::bomber::Alive>(entities[i])
                .is_some();
            if alive {
                actions[i] = Some(
                    player.select_action(
                        &world
                            .resource::<microgpt_rs::pruners::bomber::ArenaGrid>()
                            .clone(),
                        pos,
                        &tick_events,
                        rng,
                    ),
                );
            }
        }

        let ongoing = run_tick(&mut world, actions);
        if !ongoing {
            break;
        }
    }

    // Drain remaining events
    {
        let mut event_reader = world.resource_mut::<bevy_ecs::event::Events<GameEvent>>();
        round_events.extend(event_reader.drain().collect::<Vec<GameEvent>>());
    }

    // Compute scores from events
    let mut scores = [0i32; 4];
    let mut deaths = Vec::new();
    let mut kills = Vec::new();
    let mut powerups = Vec::new();
    let mut survivors = Vec::new();

    for event in &round_events {
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

    // Winner bonus
    let winner = if survivors.len() == 1 {
        scores[survivors[0] as usize] += 5;
        Some(survivors[0])
    } else if survivors.len() > 1 {
        for &s in &survivors {
            scores[s as usize] += 3;
        }
        None
    } else {
        None
    };

    let ticks = world
        .resource::<microgpt_rs::pruners::bomber::TickCounter>()
        .tick;

    RoundResult {
        scores,
        winner,
        deaths,
        kills,
        powerups,
        ticks,
    }
}
