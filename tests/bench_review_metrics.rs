//! Benchmarks for Plan 036: Inference-Time Review Metrics.
//!
//! Measures overhead of review metrics tracking:
//! - `ReviewMetrics::record()` throughput (atomic increments)
//! - `BanditSession::run()` with and without metrics
//! - `ppot_rescue_reviewed()` vs `ppot_rescue()` loop overhead
//!
//! ```sh
//! cargo test --test bench_review_metrics --features bandit,ppot -- --nocapture
//! ```

use std::sync::Arc;

#[cfg(feature = "bandit")]
use microgpt_rs::pruners::{BanditSession, BanditStrategy, BernoulliEnv, ReviewMetrics};
#[cfg(feature = "bandit")]
use microgpt_rs::types::Rng;

// ── Helpers ──────────────────────────────────────────────────────

/// Simple wall-clock timer for microbenchmarks.
struct Timer {
    start: std::time::Instant,
}

impl Timer {
    fn new() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }

    /// Elapsed time in microseconds.
    fn elapsed_us(&self) -> f64 {
        self.start.elapsed().as_nanos() as f64 / 1000.0
    }
}

// ── Benchmark: ReviewMetrics::record() throughput ────────────────

#[cfg(feature = "bandit")]
#[test]
fn bench_record_throughput() {
    let metrics = ReviewMetrics::new();
    let iterations = 1_000_000;

    // Warm up
    for _ in 0..1000 {
        metrics.record(true, true);
    }
    metrics.reset();

    let timer = Timer::new();
    for i in 0..iterations {
        // Cycle through all 4 classifications
        match i % 4 {
            0 => metrics.record(false, true),  // helpful
            1 => metrics.record(true, false),  // harmful
            2 => metrics.record(true, true),   // both_correct
            _ => metrics.record(false, false), // both_wrong
        }
    }
    let elapsed = timer.elapsed_us();

    let per_call_ns = (elapsed * 1000.0) / iterations as f64;
    let throughput = iterations as f64 / (elapsed / 1_000_000.0);

    println!("\n=== ReviewMetrics::record() throughput ===");
    println!("  {iterations} calls in {elapsed:.1}µs");
    println!("  {per_call_ns:.2} ns/call");
    println!("  {throughput:.0} ops/sec");
    println!("  Final: {metrics}");

    assert_eq!(metrics.total(), iterations);
    assert_eq!(metrics.helpful_count(), iterations / 4);
    assert_eq!(metrics.harmful_count(), iterations / 4);
    assert_eq!(metrics.both_correct_count(), iterations / 4);
    assert_eq!(metrics.both_wrong_count(), iterations / 4);

    // Target: <1ns per call (single atomic increment path)
    // On most hardware this should be 0.5-2ns
    println!(
        "  Target: <5ns/call (relaxed atomic) — actual: {per_call_ns:.2}ns — {}",
        if per_call_ns < 5.0 { "PASS" } else { "SLOW" }
    );
}

// ── Benchmark: ReviewMetrics::summary() throughput ───────────────

#[cfg(feature = "bandit")]
#[test]
fn bench_summary_throughput() {
    let metrics = ReviewMetrics::new();

    // Populate with some data
    for i in 0..1000 {
        match i % 4 {
            0 => metrics.record(false, true),
            1 => metrics.record(true, false),
            2 => metrics.record(true, true),
            _ => metrics.record(false, false),
        }
    }

    let iterations = 100_000;
    let timer = Timer::new();
    for _ in 0..iterations {
        let _ = metrics.summary();
    }
    let elapsed = timer.elapsed_us();

    let per_call_ns = (elapsed * 1000.0) / iterations as f64;

    println!("\n=== ReviewMetrics::summary() throughput ===");
    println!("  {iterations} calls in {elapsed:.1}µs");
    println!("  {per_call_ns:.2} ns/call");
    println!("  (4 atomic loads + arithmetic)");
}

// ── Benchmark: BanditSession with vs without metrics ─────────────
//
// NOTE: The "overhead" includes simulation logic (modulo + comparison)
// per episode, NOT just metrics.record(). The record() call itself is
// ~0.8ns (see bench_record_throughput). The overhead here comes from
// the `base_correct` simulation: `episode % num_arms == optimal_arm`.
// In production, `base_correct` comes from actual pruner comparison.

#[cfg(feature = "bandit")]
#[test]
fn bench_bandit_with_without_metrics() {
    let probs = [0.2, 0.4, 0.8, 0.3, 0.6];
    let episodes = 10_000;
    let seed: u64 = 42;

    // Without metrics
    let timer = Timer::new();
    let session_plain = BanditSession::new(BernoulliEnv::new(&probs), BanditStrategy::Ucb1);
    let (_events, result_plain) = session_plain.run(episodes, &mut Rng::new(seed));
    let elapsed_plain = timer.elapsed_us();

    // With metrics (includes simulation overhead: modulo + comparison + record)
    let metrics = Arc::new(ReviewMetrics::new());
    let timer = Timer::new();
    let session_metrics = BanditSession::new(BernoulliEnv::new(&probs), BanditStrategy::Ucb1)
        .with_metrics(Arc::clone(&metrics));
    let (_events, result_metrics) = session_metrics.run(episodes, &mut Rng::new(seed));
    let elapsed_metrics = timer.elapsed_us();

    // Record-only micro-bench: isolate metrics.record() overhead
    // (same number of calls, pre-computed values, no bandit session)
    let metrics_micro = Arc::new(ReviewMetrics::new());
    let timer = Timer::new();
    let optimal_arm = 2usize;
    let num_arms = 5usize;
    for episode in 0..episodes {
        let arm = episode % num_arms; // deterministic like UCB1 cold-start
        let reviewed_correct = arm == optimal_arm;
        let base_correct = episode % num_arms == optimal_arm;
        metrics_micro.record(base_correct, reviewed_correct);
    }
    let elapsed_record_only = timer.elapsed_us();

    let overhead_us = elapsed_metrics - elapsed_plain;
    let overhead_pct = (overhead_us / elapsed_plain) * 100.0;
    let per_episode_ns = (overhead_us * 1000.0) / episodes as f64;
    let record_per_episode_ns = (elapsed_record_only * 1000.0) / episodes as f64;

    println!("\n=== BanditSession with/without ReviewMetrics ===");
    println!("  {episodes} episodes per run");
    println!("  Without metrics:    {elapsed_plain:.1}µs");
    println!("  With metrics:       {elapsed_metrics:.1}µs");
    println!("  Record-only (bare): {elapsed_record_only:.1}µs");
    println!("  Session overhead:   {overhead_us:.1}µs ({overhead_pct:.1}%)");
    println!("  Per episode:        {per_episode_ns:.1}ns (session + simulation)");
    println!("  Per episode:        {record_per_episode_ns:.1}ns (record-only, no session)");
    println!("  Metrics: {metrics}");

    // Both should find the same optimal arm (same seed, same strategy)
    assert_eq!(result_plain.best_arm, result_metrics.best_arm);

    println!(
        "  record() overhead: {record_per_episode_ns:.1}ns/ep — {}",
        if record_per_episode_ns < 10.0 {
            "PASS"
        } else {
            "SLOW"
        }
    );
    println!("  Note: session overhead includes modulo simulation, not just record()");
}

// ── Benchmark: BanditSession with metrics across strategies ──────

#[cfg(feature = "bandit")]
#[test]
fn bench_bandit_metrics_all_strategies() {
    let probs = [0.2, 0.4, 0.8, 0.3, 0.6];
    let episodes = 10_000;
    let seed: u64 = 42;

    println!("\n=== BanditSession ReviewMetrics across strategies ===");
    println!("  {episodes} episodes each, seed={seed}\n");

    let strategies = vec![
        ("UCB1", BanditStrategy::Ucb1),
        (
            "ε-greedy(0.3)",
            BanditStrategy::EpsilonGreedy {
                epsilon: 0.3,
                decay: 0.995,
            },
        ),
        ("Thompson", BanditStrategy::ThompsonSampling),
    ];

    for (name, strategy) in strategies {
        let metrics = Arc::new(ReviewMetrics::new());
        let timer = Timer::new();
        let session = BanditSession::new(BernoulliEnv::new(&probs), strategy)
            .with_metrics(Arc::clone(&metrics));
        let (_events, result) = session.run(episodes, &mut Rng::new(seed));
        let elapsed = timer.elapsed_us();

        let ratio = metrics.benefit_ratio();
        let ratio_str = if ratio.is_infinite() {
            "∞".to_string()
        } else {
            format!("{ratio:.2}")
        };

        println!(
            "  {name:15}: {elapsed:8.1}µs | reward={:6.1} | arm={} | ratio={ratio_str}:1 | {metrics}",
            result.total_reward, result.best_arm
        );
    }
}

// ── Benchmark: Benefit ratio computation ─────────────────────────

#[cfg(feature = "bandit")]
#[test]
fn bench_benefit_ratio() {
    let metrics = ReviewMetrics::new();

    // Populate with realistic distribution
    for _ in 0..368 {
        metrics.record(false, true); // helpful
    }
    for _ in 0..117 {
        metrics.record(true, false); // harmful
    }
    for _ in 0..400 {
        metrics.record(true, true); // both_correct
    }
    for _ in 0..115 {
        metrics.record(false, false); // both_wrong
    }

    let iterations = 100_000;
    let timer = Timer::new();
    for _ in 0..iterations {
        let _ = metrics.benefit_ratio();
    }
    let elapsed = timer.elapsed_us();

    let per_call_ns = (elapsed * 1000.0) / iterations as f64;

    println!("\n=== ReviewMetrics::benefit_ratio() throughput ===");
    println!("  {iterations} calls in {elapsed:.1}µs");
    println!("  {per_call_ns:.2} ns/call");
    println!("  Ratio: {:.2}:1", metrics.benefit_ratio());
}

// ── Benchmark: ppot_rescue_reviewed vs ppot_rescue ───────────────

#[cfg(all(feature = "bandit", feature = "ppot"))]
#[test]
fn bench_rescue_reviewed_overhead() {
    use microgpt_rs::speculative::ppot::{PpotConfig, ppot_rescue, ppot_rescue_reviewed};
    use microgpt_rs::speculative::types::NoScreeningPruner;

    let vocab_size = 27;
    let seq_len = 8;
    let seed: u64 = 42;

    // Build simple marginals (concentrated on a few tokens)
    let marginal_slices: Vec<Vec<f32>> = (0..seq_len)
        .map(|pos| {
            let mut m = vec![0.01; vocab_size];
            // Concentrate probability on token (pos % vocab_size)
            m[pos % vocab_size] = 0.5;
            m[(pos + 1) % vocab_size] = 0.3;
            // Normalize
            let sum: f32 = m.iter().sum();
            m.iter_mut().for_each(|v| *v /= sum);
            m
        })
        .collect();
    let marginals: Vec<&[f32]> = marginal_slices.iter().map(|m| m.as_slice()).collect();

    let base_path: Vec<usize> = (0..seq_len).map(|i| i % vocab_size).collect();
    let pruner = NoScreeningPruner;
    let mut scratch = vec![0.0; vocab_size];

    // Baseline: ppot_rescue
    let iterations = 1000;
    let timer = Timer::new();
    for _ in 0..iterations {
        let _ = ppot_rescue(
            &marginals,
            &base_path,
            &pruner,
            &PpotConfig::default(),
            &mut scratch,
            &mut Rng::new(seed),
        );
    }
    let elapsed_rescue = timer.elapsed_us();

    // Reviewed: ppot_rescue_reviewed with max_loops=2
    let config_reviewed = PpotConfig::with_review_loop(2);

    let metrics = Arc::new(ReviewMetrics::new());
    // Pre-populate metrics so benefit_ratio > 0
    for _ in 0..10 {
        metrics.record(false, true);
    }

    let timer = Timer::new();
    for _ in 0..iterations {
        let _ = ppot_rescue_reviewed(
            &marginals,
            &base_path,
            &pruner,
            &config_reviewed,
            Some(&metrics),
            &mut scratch,
            &mut Rng::new(seed),
        );
    }
    let elapsed_reviewed = timer.elapsed_us();

    let overhead_us = elapsed_reviewed - elapsed_rescue;
    let overhead_pct = if elapsed_rescue > 0.0 {
        (overhead_us / elapsed_rescue) * 100.0
    } else {
        0.0
    };

    println!("\n=== ppot_rescue_reviewed vs ppot_rescue ===");
    println!("  {iterations} iterations each");
    println!("  ppot_rescue:          {elapsed_rescue:.1}µs");
    println!("  ppot_rescue_reviewed: {elapsed_reviewed:.1}µs (max_loops=2)");
    println!("  Overhead:             {overhead_us:.1}µs ({overhead_pct:.1}%)");
    println!(
        "  Expected: overhead proportional to max_loops ({:.1}x)",
        elapsed_reviewed / elapsed_rescue.max(0.001)
    );
}

// ── Benchmark: Thread contention simulation ──────────────────────

#[cfg(feature = "bandit")]
#[test]
fn bench_record_threaded() {
    use std::sync::Barrier;
    use std::thread;

    let metrics = Arc::new(ReviewMetrics::new());
    let num_threads = 4;
    let records_per_thread = 250_000;
    let barrier = Arc::new(Barrier::new(num_threads));

    let timer = Timer::new();

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let metrics = Arc::clone(&metrics);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for i in 0..records_per_thread {
                    let base = (thread_id + i) % 2 == 0;
                    let reviewed = i % 3 != 0;
                    metrics.record(base, reviewed);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let elapsed = timer.elapsed_us();
    let total_records = num_threads * records_per_thread;
    let per_call_ns = (elapsed * 1000.0) / total_records as f64;

    println!("\n=== ReviewMetrics::record() threaded contention ===");
    println!("  {num_threads} threads × {records_per_thread} records = {total_records} total");
    println!("  {elapsed:.1}µs total");
    println!("  {per_call_ns:.2} ns/call (with contention)");
    println!("  Final: {metrics}");

    assert_eq!(metrics.total(), total_records as u64);
}
