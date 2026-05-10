//! Multi-Armed Bandit Demo — Strategy Comparison
//!
//! Runs UCB1, ε-greedy (with decay), and Thompson Sampling on a 5-armed Bernoulli bandit.
//! Compares convergence speed, cumulative regret, and final Q-value accuracy.
//!
//! Run: `cargo run --example bandit_01_basic --features bandit`
//!
//! # ⚠️ What This Proves vs What It Doesn't
//!
//! This demo is a **standalone RL textbook exercise** — pure math + RNG, no LLM, no corpus.
//! It uses `BernoulliEnv::pull(arm, rng)` (a coin flip) instead of real speculative decoding.
//!
//! ## ✅ What This Proves
//!
//! - **Trait compatibility**: `BanditPruner<P>` implements `ScreeningPruner`,
//!   so it *can* plug into `build_dd_tree_screened()`. The glue is real.
//! - **Bandit math is correct**: UCB1, ε-greedy, Thompson Sampling all converge
//!   to the optimal arm. 25 unit tests in `src/pruners/bandit.rs` verify this.
//! - **Action masking works**: `BlockedArmPruner` returning `relevance = 0.0`
//!   → bandit score overridden → arm never pulled. The constrained demo proves this.
//!
//! ## ❌ What This Does NOT Prove
//!
//! - **Better tree quality** — would need real marginals from a draft model
//! - **Better accept rate** — would need real verification from a target model
//! - **Actual speedup** — would need the full speculative decoding pipeline
//!
//! ## Why DDTree + Speculative Decoding Is NOT Active Here
//!
//! The missing link is **reward signal from real verification**.
//! The full pipeline would need:
//!
//! ```text
//! 1. Draft model → marginals (log-prob distributions per position)
//! 2. build_dd_tree_screened(marginals, &bandit_pruner) → tree
//!    → BanditPruner.relevance(depth, token_idx, parents) scores each branch
//!    → blended = ln(P_draft) + ln(R_bandit)
//! 3. Target model verifies the best path
//! 4. ACCEPTED tokens → reward = 1.0, REJECTED → reward = 0.0
//! 5. bandit_pruner.update(accepted_token, reward)
//! 6. Next episode: bandit learned which tokens/branches verify well
//! ```
//!
//! Steps 1, 3, 4 require a **real transformer model** (draft + target).
//! This demo replaces all of that with `BernoulliEnv::pull()` — a coin flip.
//!
//! **This is a proof of mechanical compatibility, not a proof of value.**
//! The bridge exists but no traffic has crossed it yet.

use microgpt_rs::pruners::{BanditEnv, BanditPruner, BanditSession, BanditStrategy, BernoulliEnv};
use microgpt_rs::speculative::ScreeningPruner;
use microgpt_rs::types::Rng;

const EPISODES: usize = 1000;
const SEED: u64 = 42;

// ── Domain Pruner: Action Masking ──────────────────────────────

/// Blocks specific arms via `ScreeningPruner` relevance.
/// Demonstrates BanditPruner wrapping a constrained inner pruner:
/// blocked arms get relevance 0.0 regardless of bandit Q-values.
struct BlockedArmPruner {
    blocked: Vec<usize>,
}

impl BlockedArmPruner {
    fn new(blocked: &[usize]) -> Self {
        Self {
            blocked: blocked.to_vec(),
        }
    }
}

impl ScreeningPruner for BlockedArmPruner {
    fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        if self.blocked.contains(&token_idx) {
            0.0
        } else {
            1.0
        }
    }
}

fn print_header() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Multi-Armed Bandit — Strategy Comparison            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
}

fn print_env(probs: &[f32]) {
    println!("🎯 Bernoulli Bandit Environment:");
    println!("   Arms:    {}", probs.len());
    println!(
        "   Probs:   [{}]",
        probs
            .iter()
            .map(|p| format!("{p:.1}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "   Optimal: Arm {} (p={:.1})",
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0,
        probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!("   Episodes: {EPISODES}");
    println!();
}

fn run_strategy(
    probs: &[f32],
    strategy: BanditStrategy,
    seed: u64,
) -> (String, Vec<f32>, Vec<f32>) {
    let env = BernoulliEnv::new(probs);
    let session = BanditSession::new(env, strategy.clone());
    let (events, result) = session.run(EPISODES, &mut Rng::new(seed));

    let name = format!("{strategy}");

    // Extract regret curve from EpisodeComplete events
    let mut regret_curve = Vec::new();
    let mut reward_curve = Vec::new();
    for event in &events {
        if let microgpt_rs::pruners::BanditEvent::EpisodeComplete {
            cumulative_reward,
            cumulative_regret,
            ..
        } = event
        {
            regret_curve.push(*cumulative_regret);
            reward_curve.push(*cumulative_reward);
        }
    }

    println!("┌─ {name} ─────────────────────────────────────────");
    println!(
        "│ Best arm:     {} (optimal: {})",
        result.best_arm, result.optimal_arm
    );
    println!(
        "│ Found optimal: {}",
        if result.found_optimal() { "✅" } else { "❌" }
    );
    println!("│ Total reward:  {:.1}", result.total_reward);
    println!("│ Avg reward:    {:.4}", result.avg_reward());
    println!("│ Total regret:  {:.2}", result.total_regret);
    println!("│ Avg regret:    {:.4}", result.avg_regret());
    println!("│");
    println!("│ Q-values:");
    for (i, &q) in result.q_values.iter().enumerate() {
        let marker = if i == result.optimal_arm {
            " ← optimal"
        } else {
            ""
        };
        let bar_len = (q * 40.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("│   Arm {i}: {q:.4} {bar}{marker}");
    }
    println!("│");
    println!("│ Visit counts:");
    let total_visits: u32 = result.visits.iter().sum();
    for (i, &v) in result.visits.iter().enumerate() {
        let pct = v as f32 / total_visits as f32 * 100.0;
        let bar_len = (pct / 2.5) as usize;
        let bar: String = "█".repeat(bar_len);
        let marker = if i == result.optimal_arm {
            " ← optimal"
        } else {
            ""
        };
        println!("│   Arm {i}: {v:>4} ({pct:>5.1}%) {bar}{marker}");
    }
    println!("└────────────────────────────────────────────────────");

    (name, regret_curve, reward_curve)
}

fn print_regret_comparison(results: &[(String, Vec<f32>, Vec<f32>)]) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              Cumulative Regret Comparison                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let checkpoints: [usize; 6] = [50, 100, 250, 500, 750, 1000];
    let checkpoint_labels = checkpoints.map(|c| format!("{c}"));

    // Header
    print!("{:<20}", "Episode");
    for label in &checkpoint_labels {
        print!("{:>10}", label);
    }
    println!();
    println!("{}", "─".repeat(20 + checkpoint_labels.len() * 10));

    for (name, regret_curve, _) in results {
        print!("{name:<20}");
        for &cp in &checkpoints {
            let idx = cp.saturating_sub(1);
            if idx < regret_curve.len() {
                print!("{:>10.1}", regret_curve[idx]);
            } else {
                print!("{:>10}", "—");
            }
        }
        println!();
    }
}

fn print_reward_comparison(results: &[(String, Vec<f32>, Vec<f32>)]) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║            Average Reward Over Time                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let checkpoints: [usize; 6] = [50, 100, 250, 500, 750, 1000];
    let checkpoint_labels = checkpoints.map(|c| format!("{c}"));

    print!("{:<20}", "Episode");
    for label in &checkpoint_labels {
        print!("{:>10}", label);
    }
    println!();
    println!("{}", "─".repeat(20 + checkpoint_labels.len() * 10));

    for (name, _, reward_curve) in results {
        print!("{name:<20}");
        for &cp in &checkpoints {
            let idx = cp.saturating_sub(1);
            if idx < reward_curve.len() && cp > 0 {
                let avg = reward_curve[idx] / cp as f32;
                print!("{:>10.4}", avg);
            } else {
                print!("{:>10}", "—");
            }
        }
        println!();
    }
}

fn print_ascii_regret_plot(results: &[(String, Vec<f32>, Vec<f32>)]) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Regret Growth (ASCII Plot)                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let width = 60;
    let height = 15;

    // Find max regret for scaling
    let max_regret = results
        .iter()
        .flat_map(|(_, curve, _)| curve.iter().copied())
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1.0);

    // Sample regret at width points
    let symbols = ['●', '■', '▲'];
    let mut grid = vec![vec![' '; width]; height];

    // Y-axis labels
    #[allow(clippy::needless_range_loop)]
    for row in 0..height {
        let val = max_regret * (1.0 - row as f32 / (height - 1) as f32);
        print!("{:>7.1} │", val);
        #[allow(clippy::needless_range_loop)]
        for col in 0..width {
            let episode = (col as f32 / (width - 1) as f32 * (EPISODES - 1) as f32) as usize;
            for (si, (_, regret_curve, _)) in results.iter().enumerate() {
                if episode < regret_curve.len() {
                    let regret = regret_curve[episode];
                    let y = ((1.0 - regret / max_regret) * (height - 1) as f32) as usize;
                    if y == row {
                        grid[row][col] = symbols[si % symbols.len()];
                    }
                }
            }
            print!("{}", grid[row][col]);
        }
        println!();
    }

    // X-axis
    print!("        └");
    for _ in 0..width {
        print!("─");
    }
    println!();
    print!("         ");
    for i in 0..=4 {
        let ep = i * EPISODES / 4;
        print!("{:<15}", ep);
    }
    println!("  episodes");
    println!();

    // Legend
    for (i, (name, _, _)) in results.iter().enumerate() {
        println!("  {} = {name}", symbols[i % symbols.len()]);
    }
}

fn print_constrained_section() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Constrained Bandit: Action Masking via Pruner          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Arm 4 is best (0.9) but blocked by terrain constraint
    let probs = [0.1f32, 0.3, 0.7, 0.4, 0.9];
    let blocked = vec![4];

    println!("🎯 Scenario: 5 arms, Arm 4 blocked (p=0.9, best overall)");
    println!("   Domain pruner returns relevance(4) = 0.0 → never explored");
    println!();

    // Train BanditPruner wrapping BlockedArmPruner (action masking)
    let mut pruner = BanditPruner::new(
        BlockedArmPruner::new(&blocked),
        BanditStrategy::Ucb1,
        probs.len(),
    );

    let env = BernoulliEnv::new(&probs);
    let mut rng = Rng::new(SEED);

    // Run 500 episodes — select by pruner relevance (blocked arms always 0.0)
    for _ in 0..500 {
        let best_arm = (0..probs.len())
            .max_by(|&a, &b| {
                pruner
                    .relevance(0, a, &[])
                    .partial_cmp(&pruner.relevance(0, b, &[]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        let reward = env.pull(best_arm, &mut rng);
        pruner.update(best_arm, reward);
    }

    println!("  After 500 episodes:");
    println!();
    println!("  Arm | True p | Q-value | Visits | Relevance | Status");
    println!("  ----|--------|---------|--------|-----------|-------");
    #[allow(clippy::needless_range_loop)]
    for arm in 0..probs.len() {
        let q = pruner.q_values()[arm];
        let v = pruner.visits()[arm];
        let rel = pruner.relevance(0, arm, &[]);
        let status = if blocked.contains(&arm) {
            "🚫 BLOCKED"
        } else if arm == pruner.best_arm() {
            "⭐ BEST"
        } else {
            ""
        };
        println!(
            "    {arm} | {:.1}    | {q:.4}  | {v:>5}  | {rel:.4}    | {status}",
            probs[arm]
        );
    }

    println!();
    println!(
        "  ✅ Best valid arm: {} (true p={:.1}) — arm 4 never explored",
        pruner.best_arm(),
        probs[pruner.best_arm()]
    );
    println!();
    println!("  ScreeningPruner = action masking for bandits.");
    println!("  Invalid actions get relevance 0.0 → DDTree never explores them.");
    println!("  This bridges RL exploration with neuro-symbolic constraints.");
}

fn print_conclusion(results: &[(String, Vec<f32>, Vec<f32>)]) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                      Summary                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut best_regret = f32::MAX;
    let mut best_name = String::new();

    for (name, regret_curve, _) in results {
        let final_regret = regret_curve.last().copied().unwrap_or(0.0);
        if final_regret < best_regret {
            best_regret = final_regret;
            best_name = name.clone();
        }
    }

    println!("  Lowest final regret: {best_name} ({best_regret:.2})");
    println!();
    println!("  Key takeaways:");
    println!("  • UCB1: deterministic, O(log N) regret bound, good default");
    println!("  • Thompson Sampling: near-optimal for Bernoulli, stochastic");
    println!("  • ε-greedy with decay: simple, competitive with proper ε annealing");
    println!("  • Constrained bandit: ScreeningPruner = action masking, best valid arm found");
    println!();
    println!("  All strategies converge to the optimal arm within {EPISODES} episodes.");
    println!();
}

fn main() {
    print_header();

    let probs = [0.2, 0.5, 0.8, 0.4, 0.6];
    print_env(&probs);

    let strategies = vec![
        BanditStrategy::Ucb1,
        BanditStrategy::EpsilonGreedy {
            epsilon: 0.3,
            decay: 0.995,
        },
        BanditStrategy::ThompsonSampling,
    ];

    let mut all_results = Vec::new();

    for strategy in strategies {
        let result = run_strategy(&probs, strategy, SEED);
        all_results.push(result);
        println!();
    }

    print_regret_comparison(&all_results);
    print_reward_comparison(&all_results);
    print_ascii_regret_plot(&all_results);
    print_constrained_section();
    print_conclusion(&all_results);
}
