//! Monopoly HL Proof — 1000-Game Tournament
//!
//! Runs 1000 Monopoly games with 4 AI players at different HL tech levels.
//! Tracks survival rates, win rates, and bandit learning Q-values.
//!
//! Expected ranking: P4 (🧠 HL) > P3 (🛡️ Validator) > P2 (💰 Greedy) > P1 (🎲 Random)
//!
//! Run: `cargo run --example monopoly_03_hl_proof --features monopoly`

use fastrand::Rng;
use microgpt_rs::pruners::monopoly::{
    GameEvent, GreedyPlayer, HLPlayer, MonopolyPlayer, RandomPlayer, Strategy, ValidatorPlayer,
    run_game,
};

// ── Config ─────────────────────────────────────────────────────

const GAMES: usize = 1000;
const MAX_TURNS: u32 = 300;
const SEED: u64 = 42;

// ── Player Metadata ────────────────────────────────────────────

const EMOJI: [&str; 4] = ["🎲", "💰", "🛡️", "🧠"];
const NAMES: [&str; 4] = ["Random", "Greedy", "Validator", "HL"];

// ── Stats ──────────────────────────────────────────────────────

struct PlayerStats {
    wins: u32,
    bankruptcies: u32,
    survival_count: u32,
}

impl PlayerStats {
    const fn new() -> Self {
        Self {
            wins: 0,
            bankruptcies: 0,
            survival_count: 0,
        }
    }

    fn survival_rate(&self, games: usize) -> f64 {
        if games == 0 {
            return 0.0;
        }
        self.survival_count as f64 / games as f64
    }

    fn win_rate(&self, games: usize) -> f64 {
        if games == 0 {
            return 0.0;
        }
        self.wins as f64 / games as f64
    }
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    let mut rng = Rng::with_seed(SEED);
    let mut players: [Box<dyn MonopolyPlayer>; 4] = [
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(ValidatorPlayer::new(2)),
        Box::new(HLPlayer::new(3)),
    ];

    // ── Header ─────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Monopoly HL Proof — {GAMES} Games (seed={SEED}, max_turns={MAX_TURNS})");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut stats = [
        PlayerStats::new(),
        PlayerStats::new(),
        PlayerStats::new(),
        PlayerStats::new(),
    ];
    let mut total_turns: u64 = 0;
    let mut total_bankruptcies: u32 = 0;

    // ── Game Loop ──────────────────────────────────────────────

    for game in 0..GAMES {
        let game_seed = SEED + game as u64;
        let result = run_game(game_seed, &mut players, &mut rng, MAX_TURNS);

        total_turns += result.total_turns as u64;

        // Detect bankruptcies and survivors from events
        let mut bankrupt_ids = [false; 4];
        for event in &result.events {
            if let GameEvent::PlayerBankrupt { player, .. } = event {
                bankrupt_ids[*player as usize] = true;
                stats[*player as usize].bankruptcies += 1;
                total_bankruptcies += 1;
            }
        }

        // Record survival and wins
        for pid in 0u8..4 {
            if !bankrupt_ids[pid as usize] {
                stats[pid as usize].survival_count += 1;
            }
            if result.winner == pid {
                stats[pid as usize].wins += 1;
            }
        }

        // ── HL Bandit Update ───────────────────────────────────
        // Use the actual strategy the HL player used this game
        // (selected via start_game() -> reset() before the game)
        let hl_survived = !bankrupt_ids[3];
        let hl_won = result.winner == 3;

        let reward = match (hl_survived, hl_won) {
            (_, true) => 1.0,
            (true, false) => 0.3,
            (false, false) => 0.0,
        };

        if let Some(hl) = players[3].as_any_mut().downcast_mut::<HLPlayer>() {
            // Reward the strategy that was actually active during this game
            let strategy = Strategy::all()[hl.current_strategy];
            hl.update_outcome(strategy, reward);

            // Compress every 200 games
            if (game + 1) % 200 == 0 {
                let compressed = hl.compress_cycle();
                if !compressed.is_empty() {
                    let names = HLPlayer::strategy_names();
                    let which: Vec<&str> = compressed.iter().map(|&i| names[i]).collect();
                    eprintln!("  [HL] Compressed: {}", which.join(", "));
                }
            }
        }

        // ── Progress ───────────────────────────────────────────
        if (game + 1) % 250 == 0 {
            let mark = if game + 1 == GAMES { "✓" } else { "..." };
            println!("Progress: {}/{} {mark}", game + 1, GAMES);
        }
    }

    println!();

    // ── Survival Rate Table ────────────────────────────────────

    let mut ranking: Vec<usize> = (0..4).collect();
    ranking.sort_by(|&a, &b| {
        stats[b]
            .survival_rate(GAMES)
            .partial_cmp(&stats[a].survival_rate(GAMES))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("─── Survival Rate ─────────────────────────────────────────────");
    for (rank, &idx) in ranking.iter().enumerate() {
        let rate = stats[idx].survival_rate(GAMES) * 100.0;
        println!(
            "  #{:<3} {} {:<12}{rate:.1}%",
            rank + 1,
            EMOJI[idx],
            NAMES[idx],
        );
    }
    println!();

    // ── Win Rate Table ─────────────────────────────────────────

    ranking.sort_by(|&a, &b| {
        stats[b]
            .win_rate(GAMES)
            .partial_cmp(&stats[a].win_rate(GAMES))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("─── Win Rate ──────────────────────────────────────────────────");
    for (rank, &idx) in ranking.iter().enumerate() {
        let rate = stats[idx].win_rate(GAMES) * 100.0;
        let wins = stats[idx].wins;
        println!(
            "  #{:<3} {} {:<12}{wins} wins ({rate:.1}%)",
            rank + 1,
            EMOJI[idx],
            NAMES[idx],
        );
    }
    println!();

    // ── Key Proof: HL vs Validator ─────────────────────────────

    let hl_survival = stats[3].survival_rate(GAMES) * 100.0;
    let val_survival = stats[2].survival_rate(GAMES) * 100.0;
    let delta = hl_survival - val_survival;

    println!("─── Key Proof ─────────────────────────────────────────────────");
    println!(
        "  HL survival ({hl_survival:.1}%) - Validator survival ({val_survival:.1}%) = {delta:+.1}pp",
    );
    match delta {
        d if d >= 5.0 => println!("  ✅ PROVEN (threshold: ≥5pp)"),
        d if d > 0.0 => println!("  ⚠️  MARGINAL ({d:+.1}pp < 5pp threshold)"),
        _ => println!("  ❌ NOT PROVEN"),
    }
    println!();

    // ── HL Bandit Q-Values ─────────────────────────────────────

    println!("─── HL Bandit Q-Values ────────────────────────────────────────");
    if let Some(hl) = players[3].as_any().downcast_ref::<HLPlayer>() {
        let q = hl.strategy_q();
        let visits = hl.strategy_visits();
        let names = HLPlayer::strategy_names();

        let mut best_idx = 0;
        let mut best_q = f32::NEG_INFINITY;
        for (i, name) in names.iter().enumerate() {
            println!("  {name:<14}{:.2} ({} visits)", q[i], visits[i]);
            if q[i] > best_q {
                best_q = q[i];
                best_idx = i;
            }
        }
        println!(
            "  → Preferred strategy: {} (Q={:.2})",
            names[best_idx], best_q
        );
    } else {
        println!("  (HL player not available for Q-value report)");
    }
    println!();

    // ── Game Statistics ────────────────────────────────────────

    let avg_turns = total_turns as f64 / GAMES as f64;
    println!("─── Game Statistics ────────────────────────────────────────────");
    println!("  Avg turns/game: {avg_turns:.1}");
    println!("  Total bankruptcies: {total_bankruptcies}");
}
