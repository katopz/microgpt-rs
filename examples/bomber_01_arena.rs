//! Bomberman HL Arena — Headless Tournament Runner (Plan 033, Task 6)
//!
//! Runs N rounds of 4-player Bomberman with progressively more HL technology.
//! Output: per-round results and cumulative standings.
//!
//! Run: `cargo run --example bomber_01_arena --features bomber`

use fastrand::Rng;

use microgpt_rs::pruners::bomber::{
    BomberAction, BomberPlayer, GameEvent, GreedyPlayer, HLPlayer, RandomPlayer, ValidatorPlayer,
    init_world, run_tick, spawn_players,
};

// ── Config ─────────────────────────────────────────────────────

const ROUNDS: usize = 10;
const TICK_LIMIT: u32 = 200;

// ── Main ───────────────────────────────────────────────────────

fn main() {
    let mut rng = Rng::with_seed(42);
    let mut players: Vec<Box<dyn BomberPlayer>> = vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(ValidatorPlayer::new(2)),
        Box::new(HLPlayer::new(3)),
    ];

    println!("╔═══ Bomberman HL Arena ═══════════════════════════════════╗");
    println!("║  P1 🐰 Random  |  P2 🐱 Greedy  |  P3 🐶 Validator  |  P4 🐵 HL  ║");
    println!("╚═════════════════════════════════════════════════════════╝");
    println!();

    let mut scores = [0i32; 4];
    let mut wins = [0u32; 4];
    let mut deaths = [0u32; 4];

    for round in 0..ROUNDS {
        let seed = 42 + round as u64;
        let result = run_round(seed, &mut players, &mut rng);

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

        // Update HL player with outcome
        let survived = !result.deaths.contains(&3);
        let killed = result.kills.iter().any(|(killer, _)| *killer == 3);
        let powerups = result.powerups.iter().filter(|(p, _)| *p == 3).count();
        if let Some(hl) = players[3].as_any_mut().downcast_mut::<HLPlayer>() {
            hl.update_outcome(survived, killed, powerups as u32);
        }

        // Print round result
        let emoji = ["🐰", "🐱", "🐶", "🐵"];
        let winner_str = match result.winner {
            Some(w) => format!("{} P{}", emoji[w as usize], w),
            None => "Draw".to_string(),
        };
        println!(
            "Round {:>3}: Winner={:<12} Scores=[{}] Ticks={}",
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
        );
    }

    // Final standings
    println!();
    println!("═══ Final Standings ({ROUNDS} rounds) ═══");
    let emoji = ["🐰", "🐱", "🐶", "🐵"];
    let names = ["Random", "Greedy", "Validator", "HL"];
    let mut ranking: Vec<(usize, i32)> = scores.iter().copied().enumerate().collect();
    ranking.sort_by(|a, b| b.1.cmp(&a.1));

    for (rank, (idx, score)) in ranking.iter().enumerate() {
        println!(
            "  #{} {} {:<10} Score={:+5}  Wins={}  Deaths={}",
            rank + 1,
            emoji[*idx],
            names[*idx],
            score,
            wins[*idx],
            deaths[*idx],
        );
    }
}

// ── Round Runner ────────────────────────────────────────────────

struct RoundResult {
    scores: [i32; 4],
    winner: Option<u8>,
    deaths: Vec<u8>,
    kills: Vec<(u8, u8)>,
    powerups: Vec<(u8, u32)>,
    ticks: u32,
}

fn run_round(seed: u64, players: &mut [Box<dyn BomberPlayer>], rng: &mut Rng) -> RoundResult {
    let mut world = init_world(seed);
    let entities = spawn_players(&mut world);

    // Reset players
    for p in players.iter_mut() {
        p.reset();
    }

    let mut round_events: Vec<GameEvent> = Vec::new();

    // Run tick loop
    for _tick in 0..TICK_LIMIT {
        // Collect events from previous tick
        {
            let mut event_reader = world.resource_mut::<bevy_ecs::event::Events<GameEvent>>();
            round_events.extend(event_reader.drain().collect::<Vec<GameEvent>>());
        }

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
                        &round_events,
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
                        // Suicide (killer == victim or killer unknown)
                        scores[*victim as usize] -= 2;
                    }
                }
            }
            GameEvent::PowerUpCollected { player, kind: _ } => {
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
        // Timeout: survivors get +3 each
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
