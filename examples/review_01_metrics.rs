//! Demo: Inference-Time Review Metrics with BanditSession.
//!
//! Demonstrates Plan 036 — tracking how often the bandit reviewer
//! *fixes* a wrong random pick (helpful) vs *breaks* a correct
//! random pick (harmful), computing the benefit-to-risk ratio.
//!
//! Based on arXiv:2604.27233 — "Reinforced Agent: Inference-Time Feedback
//! for Tool-Calling Agents". The paper found a 3.1:1 benefit-to-risk ratio
//! for reasoning reviewers.
//!
//! # Usage
//!
//! ```sh
//! cargo run --example review_01_metrics --features bandit
//! ```

use std::sync::Arc;

use microgpt_rs::pruners::{
    BanditEnv, BanditSession, BanditStrategy, BernoulliEnv, ReviewMetrics, ReviewStrategy,
};
use microgpt_rs::types::Rng;

fn main() {
    println!("=== Inference-Time Review Metrics Demo ===\n");
    println!("arXiv:2604.27233 — Reinforced Agent Distillation\n");

    // ── Environment Setup ──────────────────────────────────────
    // 5 arms with varying success rates. Arm 2 is optimal (0.8).
    let probs = [0.2, 0.4, 0.8, 0.3, 0.6];
    let env = BernoulliEnv::new(&probs);

    println!("Environment: BernoulliEnv with 5 arms");
    println!("  Arm probabilities: {probs:?}");
    println!(
        "  Optimal arm: {} (p={:.1})\n",
        env.optimal_arm(),
        probs[env.optimal_arm()]
    );

    // ── Review Strategy ────────────────────────────────────────
    let strategy_review = ReviewStrategy::ProgressiveFeedback { max_loops: 2 };
    println!("Review strategy: {strategy_review} (paper's best performer)\n");

    // ── Run with ReviewMetrics ─────────────────────────────────
    let episodes = 1000;
    let seed: u64 = 42;

    println!("Running {episodes} episodes with each strategy...\n");

    // Shared metrics — same Arc injected into session
    let metrics_ucb1 = Arc::new(ReviewMetrics::new());
    let metrics_thompson = Arc::new(ReviewMetrics::new());
    let metrics_epsilon = Arc::new(ReviewMetrics::new());

    // UCB1
    let session_ucb1 = BanditSession::new(BernoulliEnv::new(&probs), BanditStrategy::Ucb1)
        .with_metrics(Arc::clone(&metrics_ucb1));
    let (_events, result_ucb1) = session_ucb1.run(episodes, &mut Rng::new(seed));

    // Thompson Sampling
    let session_thompson =
        BanditSession::new(BernoulliEnv::new(&probs), BanditStrategy::ThompsonSampling)
            .with_metrics(Arc::clone(&metrics_thompson));
    let (_events, result_thompson) = session_thompson.run(episodes, &mut Rng::new(seed));

    // ε-greedy with decay
    let session_epsilon = BanditSession::new(
        BernoulliEnv::new(&probs),
        BanditStrategy::EpsilonGreedy {
            epsilon: 0.3,
            decay: 0.995,
        },
    )
    .with_metrics(Arc::clone(&metrics_epsilon));
    let (_events, result_epsilon) = session_epsilon.run(episodes, &mut Rng::new(seed));

    // ── Results Table ──────────────────────────────────────────
    let optimal = env.optimal_arm();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Strategy   │ Reward │ Regret │ Best │ Optimal │ Found?       │");
    println!("├─────────────────────────────────────────────────────────────────┤");
    print_row("UCB1", &result_ucb1, optimal);
    print_row("Thompson", &result_thompson, optimal);
    print_row("ε-greedy", &result_epsilon, optimal);
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    // ── Review Metrics ─────────────────────────────────────────
    println!("=== Review Metrics: Did the bandit reviewer help or hurt? ===\n");

    print_metrics_detail("UCB1", &metrics_ucb1, &result_ucb1);
    print_metrics_detail("Thompson", &metrics_thompson, &result_thompson);
    print_metrics_detail("ε-greedy", &metrics_epsilon, &result_epsilon);

    // ── Benefit-Ratio Gate Simulation ──────────────────────────
    println!("=== AbsorbCompress Benefit-Ratio Gate ===\n");

    let threshold = 2.0;
    println!("Minimum benefit-ratio threshold: {threshold:.1}:1");
    println!("(Paper found 3.1:1 for o3-mini; we use 2.0 as conservative default)\n");

    for (name, metrics) in [
        ("UCB1", &metrics_ucb1),
        ("Thompson", &metrics_thompson),
        ("ε-greedy", &metrics_epsilon),
    ] {
        let ratio = metrics.benefit_ratio();
        let gated = if ratio < threshold {
            "BLOCKED"
        } else {
            "ALLOWED"
        };
        let ratio_str = if ratio.is_infinite() {
            "∞".to_string()
        } else {
            format!("{ratio:.2}")
        };
        println!("  {name:<10}: ratio={ratio_str:>5}:1 → compression {gated}");
    }

    println!();

    // ── Q-value Convergence ────────────────────────────────────
    println!("=== Q-value Convergence (final estimates) ===\n");

    print_q_values("UCB1", &result_ucb1, &probs);
    print_q_values("Thompson", &result_thompson, &probs);
    print_q_values("ε-greedy", &result_epsilon, &probs);

    // ── Summary ────────────────────────────────────────────────
    println!("=== Key Takeaways ===\n");

    let summary = metrics_ucb1.summary();
    println!(
        "  • UCB1 reviewer fixed {:.1}% of random errors (helpful)",
        summary.helpfulness
    );
    println!(
        "  • UCB1 reviewer broke {:.1}% of correct picks (harmful)",
        summary.harmfulness
    );
    println!(
        "  • Benefit-to-risk ratio: {}",
        if summary.benefit_ratio.is_infinite() {
            "∞:1 (perfect — never broke a correct pick)".to_string()
        } else {
            format!("{:.1}:1", summary.benefit_ratio)
        }
    );
    println!(
        "  • AbsorbCompress would {} compress at threshold {threshold:.1}:1",
        if metrics_ucb1.benefit_ratio() >= threshold {
            "PROCEED to"
        } else {
            "NOT"
        }
    );
    println!();
}

// ── Helpers ─────────────────────────────────────────────────────

fn print_row(name: &str, result: &microgpt_rs::pruners::BanditResult, optimal: usize) {
    let found = if result.best_arm == optimal {
        "✓ YES  "
    } else {
        "✗ NO   "
    };
    println!(
        "│ {:<10} │ {:6.1} │ {:6.1} │   {:2} │   {:2}    │ {}     │",
        name, result.total_reward, result.total_regret, result.best_arm, optimal, found
    );
}

fn print_metrics_detail(
    name: &str,
    metrics: &ReviewMetrics,
    result: &microgpt_rs::pruners::BanditResult,
) {
    let summary = metrics.summary();
    let ratio_str = if summary.benefit_ratio.is_infinite() {
        "∞".to_string()
    } else {
        format!("{:.2}", summary.benefit_ratio)
    };

    println!("  {name}:");
    println!("    {metrics}");
    println!(
        "    Helpful={}/{} ({} fixed random errors)",
        summary.helpful,
        summary.helpful + summary.both_wrong,
        summary.helpful
    );
    println!(
        "    Harmful={}/{} ({} broke correct picks)",
        summary.harmful,
        summary.harmful + summary.both_correct,
        summary.harmful
    );
    println!(
        "    Benefit ratio = {ratio_str}:1 (bandit found arm {})",
        result.best_arm
    );
    println!();
}

fn print_q_values(name: &str, result: &microgpt_rs::pruners::BanditResult, true_probs: &[f32]) {
    println!("  {name}:");
    for (arm, (q, visits)) in result.q_values.iter().zip(result.visits.iter()).enumerate() {
        let true_p = true_probs.get(arm).copied().unwrap_or(0.0);
        let err = (q - true_p).abs();
        let bar_len = (*q * 20.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!(
            "    arm {}: Q={:.3} (true={:.1}, err={:.3}) visits={:4} {bar}",
            arm, q, true_p, err, visits
        );
    }
    println!();
}
