//! Stepwise Reward Shaping — StepCodeReasoner Modelless Distillation (Plan 054).
//!
//! Demonstrates how intra-trajectory reward shaping boosts arms that enable
//! downstream success, compared to flat binary rewards.
//!
//! Run: `cargo run --example stepcode_01_shaped_bandit --features "stepcode"`

use std::time::Instant;

#[cfg(feature = "stepcode")]
fn main() {
    use microgpt_rs::pruners::{
        BanditEnv, BanditPruner, BanditSession, BanditStats, BanditStrategy, BernoulliEnv,
        PathStep, ShapedPath, path_consistency, shape_path,
    };
    use microgpt_rs::speculative::types::NoScreeningPruner;
    use microgpt_rs::types::Rng;

    let episodes = 100;
    let probs = [0.1f32, 0.3, 0.5, 0.7, 0.9];
    let lambda = 0.3;
    let path_len = 5;

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Stepwise Reward Shaping — StepCodeReasoner Plan 054         ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Environment: {} arms, probs = {:?}", probs.len(), probs);
    println!("Episodes:    {episodes}");
    println!("Path length: {path_len} steps per episode");
    println!("Lambda (λ):  {lambda}");
    println!();

    // ── Section 1: Sample Path Shaping Demo ────────────────────────
    println!("━━━ Section 1: Sample Path Shaping (λ={lambda}) ━━━━━━━━━━━━━");
    println!();

    // Build a sample verification path: 5 steps, some correct, some wrong
    let sample_steps: Vec<PathStep> = vec![
        PathStep {
            arm: 2,
            depth: 0,
            reward: 1.0,
        }, // correct
        PathStep {
            arm: 2,
            depth: 1,
            reward: 1.0,
        }, // correct
        PathStep {
            arm: 2,
            depth: 2,
            reward: 0.0,
        }, // wrong
        PathStep {
            arm: 2,
            depth: 3,
            reward: 1.0,
        }, // correct
        PathStep {
            arm: 2,
            depth: 4,
            reward: 1.0,
        }, // correct (terminal)
    ];

    println!("  Input path (arm=2, 5 steps): rewards = [1.0, 1.0, 0.0, 1.0, 1.0]");
    println!();

    let shaped = ShapedPath::shape(sample_steps.clone(), lambda);

    // Compute future_accuracy for display
    let n = sample_steps.len();
    let mut suffix_correct = vec![0.0f32; n];
    for i in (0..n.saturating_sub(1)).rev() {
        suffix_correct[i] = suffix_correct[i + 1] + sample_steps[i + 1].reward;
    }

    println!("  Step │ Depth │ Reward │ Future Acc │ Shaped Reward │ Note");
    println!("  ─────┼───────┼────────┼────────────┼───────────────┼──────────────────");
    for (i, (step, shaped_r)) in shaped
        .steps
        .iter()
        .zip(shaped.shaped_rewards.iter())
        .enumerate()
    {
        let remaining = (n - i - 1) as f32;
        let future_acc = if remaining > 0.0 {
            suffix_correct[i] / remaining
        } else {
            0.0
        };
        let note = if *shaped_r == 0.0 {
            "zero → stays zero"
        } else if future_acc == 0.0 {
            "terminal, no future"
        } else {
            "boosted by future"
        };
        println!(
            "    {i}  │   {depth:<3} │  {reward:.1}  │   {future_acc:.4}  │     {shaped_r:.4}  │ {note}",
            depth = step.depth,
            reward = step.reward,
        );
    }
    println!();
    println!(
        "  Path consistency: {:.2} (4/5 correct)",
        shaped.consistency
    );
    println!();

    // Also demonstrate shape_path convenience
    let flat_path: Vec<(usize, f32)> = vec![(0, 1.0), (1, 1.0), (2, 0.0), (3, 1.0), (4, 1.0)];
    let shaped_flat = shape_path(&flat_path, lambda);
    println!("  shape_path convenience: {:?}", shaped_flat);
    println!();

    // ── Section 2: Flat Rewards Baseline ──────────────────────────
    println!("━━━ Section 2: Flat Rewards Baseline (100 episodes) ━━━━━━━━━");
    println!();

    let env = BernoulliEnv::new(&probs);
    let session = BanditSession::new(env, BanditStrategy::Ucb1);
    let start = Instant::now();
    let (_, result_flat) = session.run(episodes, &mut Rng::new(42));
    let flat_time = start.elapsed();

    println!("  Total reward:     {:.2}", result_flat.total_reward);
    println!("  Total regret:     {:.2}", result_flat.total_regret);
    println!("  Avg reward:       {:.4}", result_flat.avg_reward());
    println!(
        "  Found optimal:    {} (arm {})",
        result_flat.found_optimal(),
        result_flat.optimal_arm
    );
    println!("  Best arm:         {}", result_flat.best_arm);
    println!("  Q-values:         {:?}", result_flat.q_values);
    println!("  Visits:           {:?}", result_flat.visits);
    println!("  Time:             {flat_time:?}");
    println!();

    // ── Section 3: Shaped Rewards ─────────────────────────────────
    println!("━━━ Section 3: Shaped Rewards (λ={lambda}, 100 episodes) ━━━━");
    println!();

    // Demonstrate BanditPruner<NoScreeningPruner> creation (integration point)
    let _bandit_pruner: BanditPruner<NoScreeningPruner> =
        BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, probs.len());
    println!(
        "  BanditPruner<NoScreeningPruner> created ({} arms, UCB1)",
        probs.len()
    );
    println!();

    // Manual shaped simulation using BanditStats
    let env_shaped = BernoulliEnv::new(&probs);
    let mut stats = BanditStats::new(probs.len());
    let mut rng = Rng::new(42);
    let mut shaped_total_reward = 0.0f32;
    let mut shaped_total_regret = 0.0f32;
    let optimal_arm = env_shaped.optimal_arm();
    let optimal_reward = env_shaped.optimal_reward();

    let start = Instant::now();
    for _episode in 0..episodes {
        let arm = select_ucb1(&stats, probs.len());

        // Simulate multi-step path
        let mut path = Vec::with_capacity(path_len);
        for step in 0..path_len {
            let reward = env_shaped.pull(arm, &mut rng);
            path.push(PathStep {
                arm,
                depth: step,
                reward,
            });
        }

        // Shape rewards
        let shaped = ShapedPath::shape(path, lambda);

        // Feed shaped rewards back
        for (step, shaped_r) in shaped.steps.iter().zip(shaped.shaped_rewards.iter()) {
            if *shaped_r > 0.0 {
                stats.update(step.arm, *shaped_r);
                shaped_total_reward += *shaped_r;
            }
        }

        shaped_total_regret += optimal_reward - env_shaped.expected_reward(arm);
    }
    let shaped_time = start.elapsed();

    let shaped_best_arm = stats.best_arm();
    let shaped_found_optimal = shaped_best_arm == optimal_arm;

    println!("  Total shaped reward: {shaped_total_reward:.2}");
    println!("  Total regret:        {shaped_total_regret:.2}");
    println!(
        "  Avg reward:          {:.4}",
        shaped_total_reward / episodes as f32
    );
    println!("  Found optimal:       {shaped_found_optimal} (arm {optimal_arm})");
    println!("  Best arm:            {shaped_best_arm}");
    println!("  Q-values:            {:?}", stats.q_values());
    println!("  Visits:              {:?}", stats.visits());
    println!("  Time:                {shaped_time:?}");
    println!();

    // ── Section 4: Comparison ─────────────────────────────────────
    println!("━━━ Section 4: Flat vs Shaped Comparison ━━━━━━━━━━━━━━━━━━━━");
    println!();

    let regret_delta = if result_flat.total_regret.abs() > f32::EPSILON {
        (shaped_total_regret - result_flat.total_regret) / result_flat.total_regret * 100.0
    } else {
        0.0
    };

    println!("  Metric              │ Flat       │ Shaped     │ Delta");
    println!("  ────────────────────┼────────────┼────────────┼───────────");
    println!(
        "  Total reward        │ {:<10.2} │ {:<10.2} │ (shaped inflated by λ)",
        result_flat.total_reward, shaped_total_reward
    );
    println!(
        "  Total regret        │ {:<10.2} │ {:<10.2} │ {:+.1}%",
        result_flat.total_regret, shaped_total_regret, regret_delta
    );
    println!(
        "  Found optimal       │ {:<10} │ {:<10} │",
        result_flat.found_optimal(),
        shaped_found_optimal
    );
    println!(
        "  Best arm            │ {:<10} │ {:<10} │",
        result_flat.best_arm, shaped_best_arm
    );
    println!();
    println!("  Shaped rewards boost arms that enable downstream success.");
    println!("  λ=0.3 gives ~15-30% bonus for early correct steps.");
    println!("  λ=0.0 reverts to flat binary (backward compatible).");
    println!();

    // ── Section 5: Path Consistency Metrics ───────────────────────
    println!("━━━ Section 5: Path Consistency Metrics ━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let sample_paths: &[(&[f32], &str)] = &[
        (&[1.0, 1.0, 1.0, 1.0, 1.0], "all correct"),
        (&[0.0, 0.0, 0.0, 0.0, 0.0], "all wrong"),
        (&[1.0, 0.0, 1.0, 0.0, 1.0], "alternating (3/5)"),
        (&[1.0, 1.0, 1.0, 0.0, 0.0], "early correct (3/5)"),
        (&[0.0, 0.0, 1.0, 1.0, 1.0], "late correct (3/5)"),
        (&[1.0], "single correct"),
        (&[], "empty path"),
    ];

    println!("  Path rewards                │ Consistency │ Description");
    println!("  ────────────────────────────┼─────────────┼────────────────");
    for (rewards, desc) in sample_paths {
        let consistency = path_consistency(rewards);
        let rewards_str = if rewards.is_empty() {
            "[]".to_string()
        } else {
            format!("{:?}", rewards)
        };
        println!("  {rewards_str:<28} │ {consistency:>11.2} │ {desc}");
    }
    println!();

    println!("{}", "═".repeat(64));
}

#[cfg(not(feature = "stepcode"))]
fn main() {
    eprintln!(
        "Enable stepcode feature: cargo run --example stepcode_01_shaped_bandit --features stepcode"
    );
}

#[cfg(feature = "stepcode")]
fn select_ucb1(stats: &microgpt_rs::pruners::BanditStats, num_arms: usize) -> usize {
    for i in 0..num_arms {
        if stats.visit_count(i) == 0 {
            return i;
        }
    }
    (0..num_arms)
        .max_by(|&a, &b| {
            stats
                .ucb1_score(a)
                .partial_cmp(&stats.ucb1_score(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0)
}
