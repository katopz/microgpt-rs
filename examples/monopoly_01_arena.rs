//! Monopoly FSM Arena — Headless Tournament Runner
//!
//! Runs N games of 4-player Monopoly with AI strategies.
//! Output: per-game progress, final standings, and HL thesis check.
//!
//! Run: `cargo run --example monopoly_01_arena --features monopoly`

use fastrand::Rng;

use microgpt_rs::pruners::monopoly::{
    GameEvent, GreedyPlayer, HLPlayer, MonopolyPlayer, RandomPlayer, Strategy, ValidatorPlayer,
    run_game,
};

// ── Config ─────────────────────────────────────────────────────

const GAMES: usize = 100;
const MAX_TURNS: u32 = 300;

// ── Main ───────────────────────────────────────────────────────

fn main() {
    let mut rng = Rng::with_seed(42);

    println!("╔═══ Monopoly FSM Arena ═══════════════════════════════════╗");
    println!("║  P1 🎲 Random  |  P2 💰 Greedy  |  P3 🛡️ Validator  |  P4 🧠 HL  ║");
    println!("╚═════════════════════════════════════════════════════════╝");
    println!();

    // Players persist across games — HL bandit Q-values accumulate
    let mut players: [Box<dyn MonopolyPlayer>; 4] = [
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(ValidatorPlayer::new(2)),
        Box::new(HLPlayer::new(3)),
    ];

    let mut wins = [0u32; 4];
    let mut bankruptcies = [0u32; 4];
    let mut total_net_worth = [0i64; 4];
    let mut total_turns = 0u32;

    for game in 0..GAMES {
        let seed = 42 + game as u64;

        let result = run_game(seed, &mut players, &mut rng, MAX_TURNS);

        wins[result.winner as usize] += 1;
        total_turns += result.total_turns;

        // Count bankruptcies from events
        for event in &result.events {
            if let GameEvent::PlayerBankrupt { player, .. } = event {
                bankruptcies[*player as usize] += 1;
            }
        }

        // Accumulate net worth proxy from events
        for event in &result.events {
            match event {
                GameEvent::SalaryCollected { player, amount } => {
                    total_net_worth[*player as usize] += *amount as i64;
                }
                GameEvent::PropertyBought { player, price, .. } => {
                    total_net_worth[*player as usize] += *price as i64;
                }
                GameEvent::RentPaid { payer, amount, .. } => {
                    total_net_worth[*payer as usize] -= *amount as i64;
                }
                _ => {}
            }
        }

        // Update HL player bandit Q-values with game outcome
        let survived = !result
            .events
            .iter()
            .any(|e| matches!(e, GameEvent::PlayerBankrupt { player, .. } if *player == 3));
        let won = result.winner == 3;
        if let Some(hl) = players[3].as_any_mut().downcast_mut::<HLPlayer>() {
            let strategy = Strategy::all()[hl.current_strategy];
            let reward = match (survived, won) {
                (_, true) => 1.0,
                (true, false) => 0.3,
                (false, false) => -1.0,
            };
            hl.update_outcome(strategy, reward);

            // Compress every 50 games
            if (game + 1) % 50 == 0 {
                let compressed = hl.compress_cycle();
                if !compressed.is_empty() {
                    let names = HLPlayer::strategy_names();
                    let which: Vec<&str> = compressed.iter().map(|&i| names[i]).collect();
                    eprintln!("  [HL] Compressed strategies: {}", which.join(", "));
                }
            }
        }

        // Progress indicator
        if (game + 1) % 25 == 0 {
            println!("  ... completed {}/{} games", game + 1, GAMES);
        }
    }

    // Print final standings
    println!();
    println!("═══ Final Standings ({GAMES} Games) ═══");
    println!(
        "  {:<4} {:<12} {:>6} {:>12} {:>10}",
        "#", "Player", "Wins", "Bankruptcies", "Win %"
    );

    let mut ranking: Vec<(usize, u32)> = wins.iter().enumerate().map(|(i, &w)| (i, w)).collect();
    ranking.sort_by(|a, b| b.1.cmp(&a.1));

    let names = ["🎲 Random", "💰 Greedy", "🛡️ Validator", "🧠 HL"];
    for (rank, &(idx, w)) in ranking.iter().enumerate() {
        let win_pct = w as f64 / GAMES as f64 * 100.0;
        println!(
            "  #{:<3} {:<12} {:>6} {:>12} {:>9.1}%",
            rank + 1,
            names[idx],
            w,
            bankruptcies[idx],
            win_pct
        );
    }

    println!();
    println!("  Avg turns/game: {:.1}", total_turns as f64 / GAMES as f64);
    println!();

    // HL thesis check
    let hl_wins = wins[3];
    let validator_wins = wins[2];
    let diff = hl_wins as i32 - validator_wins as i32;
    let pp = diff as f64 / GAMES as f64 * 100.0;

    if hl_wins > validator_wins && pp >= 5.0 {
        println!(
            "  ✅ HL thesis PROVEN: HL ({hl_wins}W) > Validator ({validator_wins}W) by {pp:.1}pp (threshold: 5pp)"
        );
    } else {
        println!(
            "  ❌ HL thesis NOT proven: HL ({hl_wins}W) vs Validator ({validator_wins}W), diff={pp:.1}pp (threshold: 5pp)"
        );
    }
    println!();

    // HL bandit Q-values report
    if let Some(hl) = players[3].as_any().downcast_ref::<HLPlayer>() {
        println!("─── HL Bandit Q-Values ────────────────────────────────────────");
        let q = hl.strategy_q();
        let visits = hl.strategy_visits();
        let strategy_names = HLPlayer::strategy_names();
        for (i, name) in strategy_names.iter().enumerate() {
            println!("  {name:<14}{:.2} ({} visits)", q[i], visits[i]);
        }
    }
}
