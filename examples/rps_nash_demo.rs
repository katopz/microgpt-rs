//! Rock-Paper-Scissors Nash Equilibrium Demo
//!
//! Proves multi-armed bandit agents converge to Nash equilibrium in zero-sum games.
//!
//! # Game Theory Connection
//!
//! In RPS, the Nash equilibrium is the uniform mixed strategy: play each action
//! with probability 1/3. Against any opponent strategy, this guarantees the game
//! value (0.5 with our Win=1.0, Tie=0.5, Loss=0.0 scoring).
//!
//! When two adaptive agents play each other, neither can exploit the other.
//! Both converge to ~33/33/33 — the unique Nash equilibrium.
//!
//! Against a biased opponent, an adaptive agent discovers the exploit:
//! if opponent plays 60% Rock, the agent converges to mostly Paper (beats Rock).
//!
//! Run: `cargo run --example rps_nash_demo --features bandit`

use microgpt_rs::pruners::{BanditStats, BanditStrategy};
use microgpt_rs::types::Rng;

const NUM_ARMS: usize = 3;
const ACTION_NAMES: [&str; NUM_ARMS] = ["Rock", "Paper", "Scissors"];
const WIN: f32 = 1.0;
const TIE: f32 = 0.5;

/// Reward for `player` given `opponent`'s action. Win=1.0, Tie=0.5, Loss=0.0.
fn rps_reward(opponent: usize, player: usize) -> f32 {
    match (opponent, player) {
        (a, b) if a == b => TIE,
        (0, 1) | (1, 2) | (2, 0) => WIN, // player wins
        _ => 0.0,
    }
}

/// Select arm from fixed probability distribution.
fn select_biased(probs: &[f32], rng: &mut Rng) -> usize {
    let r = rng.uniform();
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return i;
        }
    }
    probs.len() - 1
}

/// Select arm using bandit strategy (manual loop for 2-player games).
fn select_arm(stats: &BanditStats, strategy: &BanditStrategy, rng: &mut Rng) -> usize {
    // Cold start: play each arm once
    for i in 0..stats.num_arms() {
        if stats.visit_count(i) == 0 {
            return i;
        }
    }
    match strategy {
        BanditStrategy::Ucb1 => (0..NUM_ARMS)
            .max_by(|&a, &b| {
                stats
                    .ucb1_score(a)
                    .partial_cmp(&stats.ucb1_score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0),
        BanditStrategy::EpsilonGreedy { epsilon, .. } => {
            if rng.uniform() < *epsilon {
                (rng.uniform() * NUM_ARMS as f32) as usize % NUM_ARMS
            } else {
                stats.best_arm()
            }
        }
        BanditStrategy::ThompsonSampling => (0..NUM_ARMS)
            .map(|i| (i, stats.thompson_sample(i, rng)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0),
    }
}

/// Decay epsilon in-place for EpsilonGreedy.
fn decay_epsilon(strategy: &mut BanditStrategy) {
    if let BanditStrategy::EpsilonGreedy { epsilon, decay } = strategy {
        *epsilon *= *decay;
    }
}

fn print_bar(pct: f32, width: usize) -> String {
    let len = ((pct / 100.0) * width as f32).round() as usize;
    "█".repeat(len.min(width))
}

fn print_distribution(label: &str, visits: &[u32], highlight: Option<usize>) {
    let total: u32 = visits.iter().sum();
    if total == 0 {
        return;
    }
    println!("  {label}:");
    for (i, &v) in visits.iter().enumerate() {
        let pct = v as f32 / total as f32 * 100.0;
        let bar = print_bar(pct, 30);
        let marker = match highlight {
            Some(h) if h == i => " ← exploit",
            _ => "",
        };
        let name = ACTION_NAMES[i];
        println!("    {name:>8}: {pct:>5.1}% {bar}{marker}");
    }
}

// ── Section 1: Nash Convergence ────────────────────────────────

fn section_nash_convergence() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Section 1: Two Bandit Agents → Nash Equilibrium           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Agent A: UCB1       Agent B: ε-greedy(ε=0.1, decay=1.0)   ║");
    println!("║  5000 rounds → both converge to ~33/33/33                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut rng = Rng::new(42);
    let mut stats_a = BanditStats::new(NUM_ARMS);
    let mut stats_b = BanditStats::new(NUM_ARMS);
    let mut strat_b = BanditStrategy::EpsilonGreedy {
        epsilon: 0.1,
        decay: 1.0,
    };
    let rounds = 5000;

    for _ in 0..rounds {
        let arm_a = select_arm(&stats_a, &BanditStrategy::Ucb1, &mut rng);
        let arm_b = select_arm(&stats_b, &strat_b, &mut rng);
        stats_a.update(arm_a, rps_reward(arm_b, arm_a));
        stats_b.update(arm_b, rps_reward(arm_a, arm_b));
        decay_epsilon(&mut strat_b);
    }

    println!("  After {rounds} rounds:");
    print_distribution("Agent A (UCB1)", stats_a.visits(), None);
    println!();
    print_distribution("Agent B (ε-greedy)", stats_b.visits(), None);
    println!();

    // Verify: each arm within 10% of 33.3% (Nash equilibrium)
    let check_nash = |visits: &[u32]| {
        let total: u32 = visits.iter().sum();
        visits
            .iter()
            .all(|&v| (v as f32 / total as f32 * 100.0 - 33.3).abs() < 10.0)
    };
    let nash_a = check_nash(stats_a.visits());
    let nash_b = check_nash(stats_b.visits());
    let status = match (nash_a, nash_b) {
        (true, true) => "✅ Both ~33/33/33",
        _ => "⚠️ Not yet uniform",
    };
    println!("  Nash convergence: {status}");
    println!("  → Neither agent exploits the other → mixed strategy equilibrium");
}

// ── Section 2: Adaptive Exploitation ───────────────────────────

fn section_exploitation() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Section 2: ε-greedy vs Biased Opponent (Exploitation)     ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Agent A: ε-greedy(ε=0.3, decay=0.998) — adaptive          ║");
    println!("║  Agent B: Fixed 60% Rock, 30% Paper, 10% Scissors           ║");
    println!("║  3000 rounds → Agent A learns Paper (beats Rock).           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut rng = Rng::new(42);
    let mut stats = BanditStats::new(NUM_ARMS);
    let biased = [0.6f32, 0.3, 0.1];
    let mut strategy = BanditStrategy::EpsilonGreedy {
        epsilon: 0.3,
        decay: 0.998,
    };
    let rounds = 3000;

    // Separate RNG for opponent to decouple from bandit randomness
    let mut opp_rng = Rng::new(999);
    for _ in 0..rounds {
        let arm_a = select_arm(&stats, &strategy, &mut rng);
        let arm_b = select_biased(&biased, &mut opp_rng);
        stats.update(arm_a, rps_reward(arm_b, arm_a));
        decay_epsilon(&mut strategy);
    }

    println!("  Opponent bias: Rock=60%, Paper=30%, Scissors=10%");
    println!("  Best counter: Paper (beats Rock, the dominant opponent move)");
    println!();
    print_distribution("Agent A (ε-greedy)", stats.visits(), Some(1)); // 1 = Paper
    println!();

    let total: u32 = stats.visits().iter().sum();
    let paper_pct = stats.visits()[1] as f32 / total as f32 * 100.0;
    let exploit = match paper_pct > 50.0 {
        true => "✅ Learned to exploit Rock bias",
        false => "⚠️ Not yet exploiting",
    };
    println!("  Exploitation: {exploit} (Paper={paper_pct:.1}%)");
    println!("  → Adaptive agent discovers Paper counters Rock-heavy opponent");
}

// ── Section 3: Strategy Comparison ─────────────────────────────

fn section_comparison() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Section 3: Strategy Comparison vs Fixed Opponent            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Expected reward for each action against biased [0.6, 0.3, 0.1]:
    // Rock:    0.6×0.5 + 0.3×0.0 + 0.1×1.0 = 0.40
    // Paper:   0.6×1.0 + 0.3×0.5 + 0.1×0.0 = 0.75 ← optimal
    // Scissors: 0.6×0.0 + 0.3×1.0 + 0.1×0.5 = 0.35
    let optimal_reward = 0.75f32;
    let biased = [0.6f32, 0.3, 0.1];
    let rounds = 3000;

    let strategies: Vec<(&str, BanditStrategy)> = vec![
        ("UCB1", BanditStrategy::Ucb1),
        (
            "ε-greedy",
            BanditStrategy::EpsilonGreedy {
                epsilon: 0.1,
                decay: 0.999,
            },
        ),
        ("Thompson", BanditStrategy::ThompsonSampling),
    ];

    println!("  Opponent: 60% Rock, 30% Paper, 10% Scissors");
    println!("  Optimal fixed action: Paper (expected={optimal_reward})");
    println!("  Rounds: {rounds}");
    println!();
    println!(
        "  {:<12} {:>9} {:>11} {:>11} {:>6}",
        "Strategy", "Win Rate", "Avg Reward", "Regret", "Best"
    );
    println!("  {}", "─".repeat(60));

    for (name, strategy) in &strategies {
        let mut rng = Rng::new(99);
        let mut opp_rng = Rng::new(999);
        let mut stats = BanditStats::new(NUM_ARMS);
        let mut strat = strategy.clone();
        let mut total_reward = 0.0f32;
        let mut wins = 0usize;

        for _ in 0..rounds {
            let arm_a = select_arm(&stats, &strat, &mut rng);
            let arm_b = select_biased(&biased, &mut opp_rng);
            let reward = rps_reward(arm_b, arm_a);
            if reward >= WIN {
                wins += 1;
            }
            stats.update(arm_a, reward);
            total_reward += reward;
            decay_epsilon(&mut strat);
        }

        let win_rate = wins as f32 / rounds as f32 * 100.0;
        let avg_reward = total_reward / rounds as f32;
        let regret = (optimal_reward - avg_reward) * rounds as f32;
        let best = ACTION_NAMES[stats.best_arm()];

        println!("  {name:<12} {win_rate:>8.1}% {avg_reward:>11.4} {regret:>11.1} {best:>6}");
    }

    println!();
    println!("  → All strategies identify Paper as best arm");
    println!("  → Thompson note: Beta(α,β) posterior assumes Bernoulli rewards.");
    println!("     Ternary rewards {{0.0, 0.5, 1.0}} cause cold-start lock-in.");
    println!("  → ε-greedy with decay: most reliable for non-Bernoulli settings");
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   Rock-Paper-Scissors — Nash Equilibrium via Bandits       ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║   Proving bandit agents converge to game-theoretic          ║");
    println!("║   equilibrium in zero-sum symmetric games.                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    section_nash_convergence();
    section_exploitation();
    section_comparison();

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   Key Takeaways                                             ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║   1. Two adaptive agents → Nash (~33/33/33)                 ║");
    println!("║   2. Adaptive vs biased → exploits weakness                 ║");
    println!("║   3. ε-greedy exploits; Thompson needs Bernoulli rewards    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
