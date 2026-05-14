//! Benchmark for Plan 054: StepCodeReasoner Modelless Distillation.
//!
//! Run: `cargo test --features "bandit,stepcode" --test bench_stepcode_modelless -- --nocapture`

#[cfg(feature = "stepcode")]
use std::time::Instant;

#[cfg(feature = "stepcode")]
use microgpt_rs::pruners::{
    BanditEnv, BanditSession, BanditStats, BanditStrategy, BernoulliEnv, PathStep, ShapedPath,
    path_consistency, shape_path,
};

#[cfg(feature = "stepcode")]
use microgpt_rs::types::Rng;

// ── Bench 1: Shape Path Overhead ────────────────────────────────

#[cfg(feature = "stepcode")]
#[test]
fn bench_shape_path_overhead() {
    let path_len = 16; // typical block_size
    let iters = 100_000;
    let warmup = 1_000;
    let lambda = 0.3;

    println!("\n🧪 Bench 1: ShapedPath::shape() Overhead ({iters} iters, path_len={path_len})");
    println!("{}", "═".repeat(70));

    // Create a sample path with mixed correct/incorrect steps
    let steps: Vec<PathStep> = (0..path_len)
        .map(|i| PathStep {
            arm: i,
            depth: i,
            reward: if i % 3 == 0 { 0.0 } else { 1.0 },
        })
        .collect();

    // Warmup
    for _ in 0..warmup {
        let _ = ShapedPath::shape(steps.clone(), lambda);
    }

    // Baseline: flat path (clone only, no shaping computation)
    let start = Instant::now();
    for _ in 0..iters {
        let _ = steps.clone();
    }
    let clone_time = start.elapsed();

    // Shaped path (clone + shaping)
    let start = Instant::now();
    for _ in 0..iters {
        let _ = ShapedPath::shape(steps.clone(), lambda);
    }
    let shaped_time = start.elapsed();

    // Net shaping overhead = (clone+shape) - clone
    let shaping_ns = shaped_time.as_nanos() as f64 - clone_time.as_nanos() as f64;
    let per_call_ns = shaping_ns / iters as f64;

    println!("   Clone only:             {clone_time:>8?}");
    println!("   Clone + shape:          {shaped_time:>8?}");
    println!("   Net shaping overhead:   {per_call_ns:.0} ns/call");
    println!(
        "   Shaping total:          {:.2} ms",
        shaping_ns / 1_000_000.0
    );

    // Gate: net shaping overhead must be < 2µs per call (O(n) suffix sum)
    let gate_ok = per_call_ns < 2000.0;
    println!(
        "   Gate (<2µs overhead):   {}",
        if gate_ok { "✅ PASS" } else { "❌ FAIL" }
    );

    // Also benchmark shape_path() convenience function
    let flat_path: Vec<(usize, f32)> = (0..path_len)
        .map(|i| (i, if i % 3 == 0 { 0.0 } else { 1.0 }))
        .collect();

    // Warmup
    for _ in 0..warmup {
        let _ = shape_path(&flat_path, lambda);
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _ = shape_path(&flat_path, lambda);
    }
    let convenience_time = start.elapsed();
    let convenience_per_ns = convenience_time.as_nanos() as f64 / iters as f64;

    println!();
    println!("   shape_path() convenience fn:");
    println!("     Time:                 {convenience_time:>8?}");
    println!("     Per call:             {convenience_per_ns:.0} ns");
}

// ── Bench 2: Flat vs Shaped Convergence ─────────────────────────

#[cfg(feature = "stepcode")]
#[test]
fn bench_bandit_flat_vs_shaped_convergence() {
    let episodes = 1000;
    let probs = [0.1f32, 0.3, 0.5, 0.7, 0.9];
    let path_len = 5; // steps per episode for shaped simulation
    let lambda = 0.3;

    println!("\n🧪 Bench 2: Bandit Convergence — Flat vs Shaped ({episodes} episodes)");
    println!("{}", "═".repeat(70));

    // ── Flat Baseline ────────────────────────────────────────────
    let env_flat = BernoulliEnv::new(&probs);
    let session_flat = BanditSession::new(env_flat, BanditStrategy::Ucb1);
    let start = Instant::now();
    let (_, result_flat) = session_flat.run(episodes, &mut Rng::new(42));
    let flat_time = start.elapsed();

    println!("   Flat Rewards (UCB1):");
    println!("     Total reward:         {:.2}", result_flat.total_reward);
    println!("     Total regret:         {:.2}", result_flat.total_regret);
    println!("     Avg reward:           {:.4}", result_flat.avg_reward());
    println!("     Avg regret:           {:.4}", result_flat.avg_regret());
    println!("     Found optimal:       {}", result_flat.found_optimal());
    println!(
        "     Best arm:            {} (optimal: {})",
        result_flat.best_arm, result_flat.optimal_arm
    );
    println!("     Q-values:            {:?}", result_flat.q_values);
    println!("     Visits:              {:?}", result_flat.visits);
    println!("     Time:                {flat_time:>8?}");

    // ── Shaped Rewards ───────────────────────────────────────────
    // Simulate multi-step paths with shaped reward feedback.
    // Each episode = one arm selected → path_len pulls → ShapedPath → feed back.
    let env_shaped = BernoulliEnv::new(&probs);
    let mut stats = BanditStats::new(probs.len());
    let mut rng = Rng::new(42);
    let mut shaped_total_reward = 0.0f32;
    let mut shaped_total_regret = 0.0f32;
    let optimal_arm = env_shaped.optimal_arm();
    let optimal_reward = env_shaped.optimal_reward();

    let start = Instant::now();
    for _episode in 0..episodes {
        // Select arm using UCB1
        let arm = select_ucb1_arm(&stats, probs.len());

        // Simulate a multi-step path by pulling the same arm multiple times
        let mut path_steps = Vec::with_capacity(path_len);
        for step in 0..path_len {
            let reward = env_shaped.pull(arm, &mut rng);
            path_steps.push(PathStep {
                arm,
                depth: step,
                reward,
            });
        }

        // Compute shaped rewards
        let shaped = ShapedPath::shape(path_steps, lambda);

        // Feed shaped rewards back to bandit stats
        for (step, shaped_reward) in shaped.steps.iter().zip(shaped.shaped_rewards.iter()) {
            if *shaped_reward > 0.0 {
                stats.update(step.arm, *shaped_reward);
                shaped_total_reward += *shaped_reward;
            }
        }

        // Track regret based on expected reward of the chosen arm
        shaped_total_regret += optimal_reward - env_shaped.expected_reward(arm);
    }
    let shaped_time = start.elapsed();

    let shaped_avg_reward = shaped_total_reward / episodes as f32;
    let shaped_avg_regret = shaped_total_regret / episodes as f32;
    let shaped_best_arm = stats.best_arm();
    let shaped_found_optimal = shaped_best_arm == optimal_arm;

    println!();
    println!("   Shaped Rewards (λ={lambda}, path_len={path_len}):");
    println!("     Total reward:         {shaped_total_reward:.2}");
    println!("     Total regret:         {shaped_total_regret:.2}");
    println!("     Avg reward:           {shaped_avg_reward:.4}");
    println!("     Avg regret:           {shaped_avg_regret:.4}");
    println!("     Found optimal:       {shaped_found_optimal}");
    println!("     Best arm:            {shaped_best_arm} (optimal: {optimal_arm})");
    println!("     Q-values:            {:?}", stats.q_values());
    println!("     Visits:              {:?}", stats.visits());
    println!("     Time:                {shaped_time:>8?}");

    // ── Comparison ───────────────────────────────────────────────
    println!();
    println!("   ── Comparison ──────────────────────────────────────");

    // Shaped rewards are inflated by (1 + λ × future_accuracy), so we compare
    // convergence quality (found optimal, regret ratio) not raw reward totals.
    let regret_delta = if result_flat.total_regret.abs() > f32::EPSILON {
        (shaped_total_regret - result_flat.total_regret) / result_flat.total_regret * 100.0
    } else {
        0.0
    };

    println!("     Regret delta:        {regret_delta:+.1}%");
    println!("     Flat found optimal:  {}", result_flat.found_optimal());
    println!("     Shaped found optimal:{shaped_found_optimal}");

    // Gate: shaped rewards must NOT degrade by >5%
    // Measured by regret increase (positive delta = worse).
    // If shaped found optimal, that's sufficient regardless of regret delta.
    let gate_ok = regret_delta <= 5.0 || shaped_found_optimal;
    println!(
        "     Gate (regret Δ ≤ 5% or found optimal): {}",
        if gate_ok { "✅ PASS" } else { "❌ FAIL" }
    );

    // Also run Thompson Sampling comparison
    println!();
    println!("   ── Thompson Sampling Comparison ────────────────────");

    let env_ts_flat = BernoulliEnv::new(&probs);
    let session_ts = BanditSession::new(env_ts_flat, BanditStrategy::ThompsonSampling);
    let start = Instant::now();
    let (_, result_ts) = session_ts.run(episodes, &mut Rng::new(42));
    let ts_flat_time = start.elapsed();

    println!(
        "   Thompson Flat: reward={:.2}, regret={:.2}, optimal={}",
        result_ts.total_reward,
        result_ts.total_regret,
        result_ts.found_optimal()
    );
    println!("     Time:                {ts_flat_time:>8?}");

    // Shaped Thompson: manual loop with shaped rewards
    let env_ts_shaped = BernoulliEnv::new(&probs);
    let mut ts_stats = BanditStats::new(probs.len());
    let mut ts_rng = Rng::new(42);
    let mut ts_shaped_reward = 0.0f32;
    let mut ts_shaped_regret = 0.0f32;

    let start = Instant::now();
    for _episode in 0..episodes {
        let arm = select_thompson_arm(&ts_stats, probs.len(), &mut ts_rng);

        let mut path_steps = Vec::with_capacity(path_len);
        for step in 0..path_len {
            let reward = env_ts_shaped.pull(arm, &mut ts_rng);
            path_steps.push(PathStep {
                arm,
                depth: step,
                reward,
            });
        }

        let shaped = ShapedPath::shape(path_steps, lambda);
        for (step, shaped_reward) in shaped.steps.iter().zip(shaped.shaped_rewards.iter()) {
            if *shaped_reward > 0.0 {
                ts_stats.update(step.arm, *shaped_reward);
                ts_shaped_reward += *shaped_reward;
            }
        }

        ts_shaped_regret += optimal_reward - env_ts_shaped.expected_reward(arm);
    }
    let ts_shaped_time = start.elapsed();

    println!(
        "   Thompson Shaped: reward={ts_shaped_reward:.2}, regret={ts_shaped_regret:.2}, optimal={}",
        ts_stats.best_arm() == optimal_arm
    );
    println!("     Time:                {ts_shaped_time:>8?}");
}

/// Select arm using UCB1 strategy from BanditStats.
#[cfg(feature = "stepcode")]
fn select_ucb1_arm(stats: &BanditStats, num_arms: usize) -> usize {
    // Cold start: play each arm once
    for i in 0..num_arms {
        if stats.visit_count(i) == 0 {
            return i;
        }
    }
    // UCB1 selection
    (0..num_arms)
        .max_by(|&a, &b| {
            stats
                .ucb1_score(a)
                .partial_cmp(&stats.ucb1_score(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0)
}

/// Select arm using Thompson Sampling from BanditStats.
#[cfg(feature = "stepcode")]
fn select_thompson_arm(stats: &BanditStats, num_arms: usize, rng: &mut Rng) -> usize {
    // Cold start: play each arm once
    for i in 0..num_arms {
        if stats.visit_count(i) == 0 {
            return i;
        }
    }
    // Thompson sampling
    (0..num_arms)
        .map(|i| (i, stats.thompson_sample(i, rng)))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ── Bench 3: Path Consistency Computation ───────────────────────

#[cfg(feature = "stepcode")]
#[test]
fn bench_path_consistency_computation() {
    let iters = 100_000;
    let warmup = 1_000;

    println!("\n🧪 Bench 3: path_consistency() Overhead ({iters} iters)");
    println!("{}", "═".repeat(70));

    let rewards_full: Vec<f32> = vec![1.0; 16];
    let rewards_mixed: Vec<f32> = (0..16)
        .map(|i| if i % 3 == 0 { 0.0 } else { 1.0 })
        .collect();
    let rewards_sparse: Vec<f32> = (0..16).map(|i| if i == 7 { 1.0 } else { 0.0 }).collect();

    // Warmup
    for _ in 0..warmup {
        let _ = path_consistency(&rewards_full);
        let _ = path_consistency(&rewards_mixed);
        let _ = path_consistency(&rewards_sparse);
    }

    // Benchmark each pattern
    let start = Instant::now();
    for _ in 0..iters {
        let _ = path_consistency(&rewards_full);
    }
    let time_full = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        let _ = path_consistency(&rewards_mixed);
    }
    let time_mixed = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        let _ = path_consistency(&rewards_sparse);
    }
    let time_sparse = start.elapsed();

    let avg_ns = (time_full.as_nanos() + time_mixed.as_nanos() + time_sparse.as_nanos()) as f64
        / 3.0
        / iters as f64;

    println!("   All correct (16/16):    {time_full:>8?}");
    println!("   Mixed (11/16):          {time_mixed:>8?}");
    println!("   Sparse (1/16):          {time_sparse:>8?}");
    println!("   Avg per call:           {avg_ns:.0} ns");

    // Verify correctness
    let c_full = path_consistency(&rewards_full);
    let c_mixed = path_consistency(&rewards_mixed);
    let c_sparse = path_consistency(&rewards_sparse);

    println!();
    println!("   Consistency values:");
    println!("     All correct:          {c_full:.4} (expected: 1.0000)");
    println!(
        "     Mixed:                {c_mixed:.4} (expected: {:.4})",
        10.0 / 16.0
    );
    println!(
        "     Sparse:               {c_sparse:.4} (expected: {:.4})",
        1.0 / 16.0
    );

    // Gate: correctness
    let gate_ok = (c_full - 1.0).abs() < 1e-6
        && (c_mixed - 10.0 / 16.0).abs() < 1e-6
        && (c_sparse - 1.0 / 16.0).abs() < 1e-6;
    println!(
        "     Gate (correctness):   {}",
        if gate_ok { "✅ PASS" } else { "❌ FAIL" }
    );
}

// ── Bench 4: Shaped Reward Values ───────────────────────────────

#[cfg(feature = "stepcode")]
#[test]
fn bench_shaped_reward_values() {
    println!("\n🧪 Bench 4: Shaped Reward Mathematical Correctness");
    println!("{}", "═".repeat(70));

    let lambda = 0.3;

    // Case 1: All correct → consistency = 1.0, rewards monotonically decreasing
    let steps_all: Vec<PathStep> = (0..5)
        .map(|i| PathStep {
            arm: i,
            depth: i,
            reward: 1.0,
        })
        .collect();
    let shaped_all = ShapedPath::shape(steps_all, lambda);

    println!("   Case 1: All correct (5 steps, λ={lambda})");
    println!(
        "     Consistency:          {:.4} (expected: 1.0000)",
        shaped_all.consistency
    );
    println!("     Shaped rewards:       {:?}", shaped_all.shaped_rewards);
    // Terminal: 1.0 × (1 + 0.3 × 0) = 1.0
    // First: 1.0 × (1 + 0.3 × 4/4) = 1.3
    println!("     Expected terminal:    1.0000");
    println!("     Expected first:       1.3000");

    let all_correct_ok = (shaped_all.consistency - 1.0).abs() < 1e-6
        && (shaped_all.shaped_rewards.last().unwrap() - 1.0).abs() < 1e-6
        && (shaped_all.shaped_rewards.first().unwrap() - 1.3).abs() < 1e-5;

    // Rewards should be monotonically decreasing (earlier steps enable more future)
    let monotonic_ok = shaped_all.shaped_rewards.windows(2).all(|w| w[0] >= w[1]);

    println!(
        "     Gate (consistency=1, terminal=1, first=1.3): {}",
        if all_correct_ok {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );
    println!(
        "     Gate (monotonic decreasing): {}",
        if monotonic_ok { "✅ PASS" } else { "❌ FAIL" }
    );

    // Case 2: All wrong → all shaped = 0.0
    let steps_wrong: Vec<PathStep> = (0..5)
        .map(|i| PathStep {
            arm: i,
            depth: i,
            reward: 0.0,
        })
        .collect();
    let shaped_wrong = ShapedPath::shape(steps_wrong, lambda);

    println!();
    println!("   Case 2: All wrong (5 steps)");
    println!(
        "     Consistency:          {:.4} (expected: 0.0000)",
        shaped_wrong.consistency
    );
    println!(
        "     Shaped rewards:       {:?}",
        shaped_wrong.shaped_rewards
    );

    let all_wrong_ok = shaped_wrong.shaped_rewards.iter().all(|&r| r == 0.0)
        && (shaped_wrong.consistency - 0.0).abs() < 1e-6;
    println!(
        "     Gate (all zero):      {}",
        if all_wrong_ok { "✅ PASS" } else { "❌ FAIL" }
    );

    // Case 3: λ = 0 → flat binary rewards
    let steps_flat: Vec<PathStep> = (0..3)
        .map(|i| PathStep {
            arm: i,
            depth: i,
            reward: 1.0,
        })
        .collect();
    let shaped_flat = ShapedPath::shape(steps_flat, 0.0);

    println!();
    println!("   Case 3: λ=0 (flat binary, 3 steps)");
    println!(
        "     Shaped rewards:       {:?}",
        shaped_flat.shaped_rewards
    );

    let flat_ok = shaped_flat
        .shaped_rewards
        .iter()
        .all(|&r| (r - 1.0).abs() < 1e-6);
    println!(
        "     Gate (all = 1.0):     {}",
        if flat_ok { "✅ PASS" } else { "❌ FAIL" }
    );

    // Case 4: Enables downstream — arm 0 correct, arm 1 correct, arm 2 wrong
    let steps_enable = vec![
        PathStep {
            arm: 0,
            depth: 0,
            reward: 1.0,
        },
        PathStep {
            arm: 1,
            depth: 1,
            reward: 1.0,
        },
        PathStep {
            arm: 2,
            depth: 2,
            reward: 0.0,
        },
    ];
    let shaped_enable = ShapedPath::shape(steps_enable, 0.3);

    println!();
    println!("   Case 4: Enables downstream (arm0=✓, arm1=✓, arm2=✗)");
    println!(
        "     Shaped rewards:       {:?}",
        shaped_enable.shaped_rewards
    );
    // Arm 0: 1.0 × (1 + 0.3 × 1/2) = 1.0 × 1.15 = 1.15
    // Arm 1: 1.0 × (1 + 0.3 × 0/1) = 1.0
    // Arm 2: 0.0 × anything = 0.0
    println!("     Expected:             [1.15, 1.0, 0.0]");

    let enable_ok = (shaped_enable.shaped_rewards[0] - 1.15).abs() < 1e-4
        && (shaped_enable.shaped_rewards[1] - 1.0).abs() < 1e-5
        && (shaped_enable.shaped_rewards[2] - 0.0).abs() < 1e-6
        && shaped_enable.shaped_rewards[0] > shaped_enable.shaped_rewards[1];
    println!(
        "     Gate (arm0 > arm1):   {}",
        if enable_ok { "✅ PASS" } else { "❌ FAIL" }
    );

    // Case 5: Empty path
    let shaped_empty = ShapedPath::shape(vec![], lambda);
    println!();
    println!("   Case 5: Empty path");
    println!(
        "     Shaped rewards:       {:?}",
        shaped_empty.shaped_rewards
    );
    println!("     Consistency:          {:.4}", shaped_empty.consistency);

    let empty_ok =
        shaped_empty.shaped_rewards.is_empty() && (shaped_empty.consistency - 0.0).abs() < 1e-6;
    println!(
        "     Gate:                 {}",
        if empty_ok { "✅ PASS" } else { "❌ FAIL" }
    );

    // Case 6: Single step (terminal only)
    let steps_single = vec![PathStep {
        arm: 0,
        depth: 0,
        reward: 1.0,
    }];
    let shaped_single = ShapedPath::shape(steps_single, 0.5);
    println!();
    println!("   Case 6: Single step (terminal only, λ=0.5)");
    println!(
        "     Shaped rewards:       {:?}",
        shaped_single.shaped_rewards
    );
    // Terminal step gets no future: 1.0 × (1 + 0.5 × 0) = 1.0
    let single_ok = (shaped_single.shaped_rewards[0] - 1.0).abs() < 1e-6;
    println!(
        "     Gate (shaped = 1.0):  {}",
        if single_ok { "✅ PASS" } else { "❌ FAIL" }
    );

    // Overall gate
    println!();
    let all_ok = all_correct_ok
        && monotonic_ok
        && all_wrong_ok
        && flat_ok
        && enable_ok
        && empty_ok
        && single_ok;
    println!(
        "   Overall Gate (all pass): {}",
        if all_ok { "✅ PASS" } else { "❌ FAIL" }
    );
}

// ── Summary ─────────────────────────────────────────────────────

#[cfg(feature = "stepcode")]
#[test]
fn bench_stepcode_modelless_summary() {
    println!("\n📋 Plan 054: StepCodeReasoner Modelless Distillation — Benchmark Summary");
    println!("{}", "═".repeat(70));
    println!("   Bench 1: ShapedPath::shape() overhead   — bench_shape_path_overhead");
    println!(
        "   Bench 2: Flat vs Shaped convergence      — bench_bandit_flat_vs_shaped_convergence"
    );
    println!("   Bench 3: path_consistency() overhead      — bench_path_consistency_computation");
    println!("   Bench 4: Shaped reward correctness        — bench_shaped_reward_values");
    println!();
    println!(
        "   Run: cargo test --features \"bandit,stepcode\" --test bench_stepcode_modelless -- --nocapture"
    );
    println!("{}", "═".repeat(70));
}
