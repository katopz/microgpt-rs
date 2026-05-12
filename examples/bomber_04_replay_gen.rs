//! Dedicated replay generator for Bomberman training data.
//! Runs 1000 rounds with 4 players, filters to dump only P3 (Validator) and P4 (HL) winning episodes.
//! Output: output/replays/bomber_replay_{timestamp}.jsonl
//!
//! Run: `cargo run --example bomber_04_replay_gen --features bomber`

use std::path::PathBuf;

use fastrand::Rng;

use microgpt_rs::pruners::bomber::replay::{
    ReplaySample, ReplayWriter, serialize_board, serialize_bombs, serialize_powerups,
};
use microgpt_rs::pruners::bomber::{
    BomberPlayer, GameEvent, GreedyPlayer, HLPlayer, RandomPlayer, ValidatorPlayer, init_world,
    run_tick, spawn_players,
};

// ── Config ─────────────────────────────────────────────────────

const ROUNDS: usize = 1000;
const TICK_LIMIT: u32 = 200;
const QUALITY_THRESHOLD: f32 = 0.5;
const ACTION_NAMES: [&str; 6] = ["Up", "Down", "Left", "Right", "Bomb", "Wait"];

// ── Pending Sample ─────────────────────────────────────────────

struct PendingSample {
    board: Vec<u8>,
    player_pos: [u8; 2],
    player_id: u8,
    bombs: Vec<[u8; 4]>,
    powerups: Vec<[u8; 2]>,
    action: u8,
    tick: u32,
    player_type: String,
}

// ── Round Result ───────────────────────────────────────────────

struct RoundResult {
    _scores: [i32; 4],
    survivors: Vec<u8>,
    _deaths: Vec<u8>,
    kills: Vec<(u8, u8)>,
    powerups: Vec<(u8, u32)>,
    _ticks: u32,
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    // Parse optional CLI arg for output dir (default: output/replays)
    let args: Vec<String> = std::env::args().collect();
    let output_dir: PathBuf = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from("output/replays")
    };
    std::fs::create_dir_all(&output_dir).ok();

    let mut rng = Rng::with_seed(42);
    let mut players: Vec<Box<dyn BomberPlayer>> = vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(ValidatorPlayer::new(2)),
        Box::new(HLPlayer::new(3)),
    ];

    // Create a single ReplayWriter for the session
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let filename = format!("bomber_replay_{timestamp}.jsonl");
    let path = output_dir.join(&filename);
    let mut writer = ReplayWriter::create(&path, 0).expect("failed to create replay writer");

    println!("╔═══ Bomberman Replay Generator ════════════════════════════╗");
    println!("║  P1 Random | P2 Greedy | P3 Validator | P4 HL           ║");
    println!("║  Dumping P3+P4 samples with quality > {QUALITY_THRESHOLD}              ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    let mut action_counts = [0usize; 6];
    let mut total_quality = 0.0f64;
    let mut total_samples = 0u64;

    for round in 0..ROUNDS {
        let seed = 42 + round as u64;
        let (result, pending) = run_round(seed, &mut players, &mut rng);

        // Update HL player with outcome
        let p4_survived = result.survivors.contains(&3);
        let p4_killed = result.kills.iter().any(|(k, _)| *k == 3);
        let p4_pu_count = result.powerups.iter().filter(|(p, _)| *p == 3).count();
        if let Some(hl) = players[3].as_any_mut().downcast_mut::<HLPlayer>() {
            hl.update_outcome(p4_survived, p4_killed, p4_pu_count as u32);
        }

        // Backfill quality and write filtered samples
        for ps in pending {
            let survived = result.survivors.contains(&ps.player_id);
            let winner = result.survivors.len() == 1 && result.survivors[0] == ps.player_id;
            let pu_count = result
                .powerups
                .iter()
                .filter(|(p, _)| *p == ps.player_id)
                .count() as u32;
            let kill_count = result
                .kills
                .iter()
                .filter(|(k, _)| *k == ps.player_id)
                .count() as u32;
            let quality = ReplaySample::quality(survived, winner, pu_count, kill_count);

            if quality > QUALITY_THRESHOLD {
                let sample = ReplaySample {
                    board: ps.board,
                    player_pos: ps.player_pos,
                    player_id: ps.player_id,
                    bombs: ps.bombs,
                    powerups: ps.powerups,
                    action: ps.action,
                    quality,
                    tick: ps.tick,
                    round: round as u32,
                    player_type: ps.player_type,
                };
                writer.write_sample(&sample).ok();
                action_counts[ps.action as usize] += 1;
                total_quality += quality as f64;
                total_samples += 1;
            }
        }

        // Progress every 200 rounds
        if (round + 1) % 200 == 0 {
            println!("  [Round {}/{}] samples={total_samples}", round + 1, ROUNDS,);
        }
    }

    writer.flush().ok();

    // ── Stats ──────────────────────────────────────────────────

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  REPLAY GENERATION COMPLETE ({ROUNDS} rounds)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Total samples written: {total_samples}");

    println!("  Action distribution:");
    for (i, name) in ACTION_NAMES.iter().enumerate() {
        let count = action_counts[i];
        if count > 0 {
            let pct = (count as f64 / total_samples as f64) * 100.0;
            println!("    {name:<8} {count:>6}  ({pct:.1}%)");
        }
    }

    let avg_quality = if total_samples > 0 {
        total_quality / total_samples as f64
    } else {
        0.0
    };
    println!("  Average quality: {avg_quality:.3}");
    println!("  Output: {}", path.display());
    println!();
}

// ── Round Runner ───────────────────────────────────────────────

fn run_round(
    seed: u64,
    players: &mut [Box<dyn BomberPlayer>],
    rng: &mut Rng,
) -> (RoundResult, Vec<PendingSample>) {
    let mut world = init_world(seed);
    let entities = spawn_players(&mut world);

    for p in players.iter_mut() {
        p.reset();
    }

    let mut all_events: Vec<GameEvent> = Vec::new();
    let mut pending_samples: Vec<PendingSample> = Vec::new();

    for _tick in 0..TICK_LIMIT {
        // Drain events from previous tick
        let tick_events: Vec<GameEvent> = {
            use bevy_ecs::event::Events;
            let mut ev = world.resource_mut::<Events<GameEvent>>();
            ev.drain().collect()
        };
        all_events.extend(tick_events.iter().cloned());

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
                let grid = world
                    .resource::<microgpt_rs::pruners::bomber::ArenaGrid>()
                    .clone();
                let action = player.select_action(&grid, pos, &tick_events, rng);
                actions[i] = Some(action);

                // Capture P3 (Validator, index 2) and P4 (HL, index 3) only
                if i == 2 || i == 3 {
                    let grid_ref = world.resource::<microgpt_rs::pruners::bomber::ArenaGrid>();
                    let board = serialize_board(grid_ref);
                    let bombs = serialize_bombs(&mut world);
                    let powerups = serialize_powerups(&mut world);
                    let tick = world
                        .resource::<microgpt_rs::pruners::bomber::TickCounter>()
                        .tick;

                    pending_samples.push(PendingSample {
                        board,
                        player_pos: [pos.x as u8, pos.y as u8],
                        player_id: i as u8,
                        bombs,
                        powerups,
                        action: action.as_usize() as u8,
                        tick,
                        player_type: if i == 2 { "Validator" } else { "HL" }.to_string(),
                    });
                }
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

    let result = RoundResult {
        _scores: scores,
        survivors,
        _deaths: deaths,
        kills,
        powerups,
        _ticks: ticks,
    };

    (result, pending_samples)
}
