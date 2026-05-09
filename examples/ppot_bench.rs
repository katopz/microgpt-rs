//! PPoT benchmark — run with: `cargo run --example ppot_bench --features ppot`
//!
//! Standalone binary for PPoT (Probabilistic Programs of Thought) benchmarks.
//! Separated from the main binary to avoid icache regression (72KB binary bloat
//! causing 7-15% regression in unrelated DDTree benchmarks).

use microgpt_rs::benchmark::{BenchCategory, BenchResult};
use microgpt_rs::speculative::ppot::{
    PpotConfig, SessionKnowledge, TokenRule, identify_high_entropy_positions, ppot_resample,
    ppot_resample_different_value, ppot_resample_with_support, ppot_rescue, ppot_rescue_adaptive,
    token_entropy,
};
use microgpt_rs::speculative::{
    NoScreeningPruner, SpeculativeContext, dflash_predict_with, sample_from_distribution,
};
use microgpt_rs::transformer::TransformerWeights;
use microgpt_rs::types::{Config, Rng};
use std::time::Instant;

fn main() {
    let draft_config = Config::draft();
    let mut draft_rng = Rng::new(99);
    let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

    let warmup = 1000;
    let iters = 50000;

    println!("\n📊 PPoT Benchmarks ({iters} iterations, {warmup} warmup)");
    println!("{}", "═".repeat(60));
    println!(
        "   Draft model: embd={}, heads={}, mlp={}",
        draft_config.n_embd, draft_config.n_head, draft_config.mlp_hidden
    );

    let mut results = Vec::new();

    // Entropy
    let entropy_br = bench_ppot_entropy(&draft_weights, &draft_config, warmup, iters);
    results.push(entropy_br);

    // Resample (3 variants)
    let (resample_basic_br, resample_diff_br, resample_support_br) =
        bench_ppot_resample(&draft_weights, &draft_config, warmup, iters);
    results.push(resample_basic_br);
    results.push(resample_diff_br);
    results.push(resample_support_br);

    // Rescue (3 strategies)
    let (rescue_greedy_br, rescue_ppot_br, rescue_adaptive_br) =
        bench_ppot_rescue(&draft_weights, &draft_config, warmup, iters);
    results.push(rescue_greedy_br);
    results.push(rescue_ppot_br);
    results.push(rescue_adaptive_br);

    // Print results
    println!("\n📊 PPoT Benchmark Results");
    println!("{}", "─".repeat(75));
    println!(
        "  {:<30} {:>15} {:>15} {:>15}",
        "Method", "Throughput", "μs/step", "Avg Accept Len"
    );
    println!("{}", "─".repeat(75));

    for r in &results {
        let unit = if r.avg_acceptance_len > 0.0 {
            "tok/s"
        } else {
            "trees/s"
        };
        println!(
            "  {:<30} {:>12.0} {:>3} {:>12.2} {:>15.2}",
            r.label, r.throughput, unit, r.time_per_step_us, r.avg_acceptance_len,
        );
    }

    println!("{}", "─".repeat(75));
    println!("\n✨ Done.");
}

/// Benchmark: Shannon entropy calculation overhead.
///
/// Measures `token_entropy()` cost on marginal distributions from DFlash.
/// Target: <1% of DFlash time (entropy is O(vocab_size) per position).
fn bench_ppot_entropy(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let mut sctx = SpeculativeContext::new(draft_config);

    // Produce marginals once via DFlash
    sctx.reset();
    let steps = dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
    let vocab_size = draft_config.vocab_size;
    let marginals: Vec<&[f32]> = (0..steps)
        .map(|step| sctx.marginal_slice(step, vocab_size))
        .collect();

    // Warmup
    for _ in 0..warmup {
        for m in &marginals {
            std::hint::black_box(token_entropy(m));
        }
    }

    let mut total_entropy_calls = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        for m in &marginals {
            std::hint::black_box(token_entropy(m));
            total_entropy_calls += 1;
        }
    }
    let elapsed = start.elapsed();

    let calls_per_sec = total_entropy_calls as f64 / elapsed.as_secs_f64();

    BenchResult {
        label: "PPoT Entropy (H calc)".into(),
        throughput: calls_per_sec,
        time_per_step_us: elapsed.as_micros() as f64 / total_entropy_calls as f64,
        avg_acceptance_len: steps as f64,
        color: (148, 103, 189),
        category: BenchCategory::Infrastructure,
    }
}

/// Benchmark: PPoT resample throughput (samples/ms on CPU).
///
/// Compares three resampling modes:
/// - Basic: unrestricted resample from full vocabulary
/// - Different-value: conditioned on not reproducing original token
/// - Support-constrained: Digit rule support set
///
/// All modes are CPU-only (no GPU forward passes).
fn bench_ppot_resample(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> (BenchResult, BenchResult, BenchResult) {
    let mut rng = Rng::new(99);
    let mut sctx = SpeculativeContext::new(draft_config);

    // Produce marginals and identify high-entropy positions
    sctx.reset();
    let steps = dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
    let vocab_size = draft_config.vocab_size;
    let marginals: Vec<&[f32]> = (0..steps)
        .map(|step| sctx.marginal_slice(step, vocab_size))
        .collect();

    let threshold = 0.5f32;
    let positions = identify_high_entropy_positions(&marginals, threshold);
    let positions = if positions.is_empty() {
        vec![0usize.min(steps.saturating_sub(1))]
    } else {
        positions
    };

    // Base path: argmax from marginals
    let base_path: Vec<usize> = marginals
        .iter()
        .map(|m| {
            m.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect();

    let num_samples = 10usize;
    let mut scratch = vec![0.0f32; vocab_size];

    // Warmup all three modes
    for _ in 0..warmup {
        for _ in 0..num_samples {
            std::hint::black_box(ppot_resample(&base_path, &marginals, &positions, &mut rng));
        }
        for _ in 0..num_samples {
            std::hint::black_box(ppot_resample_different_value(
                &base_path,
                &marginals,
                &positions,
                &mut scratch,
                &mut rng,
            ));
        }
        let digit_support = TokenRule::Digit.support(vocab_size);
        let mut temp_scratch = vec![0.0f32; vocab_size];
        for _ in 0..num_samples {
            std::hint::black_box(ppot_resample_with_support(
                &base_path,
                &marginals,
                &positions,
                &digit_support,
                &mut temp_scratch,
                &mut rng,
            ));
        }
    }

    // Benchmark: Basic resample
    let start = Instant::now();
    let mut total_samples_basic = 0usize;
    for _ in 0..iters {
        for _ in 0..num_samples {
            std::hint::black_box(ppot_resample(&base_path, &marginals, &positions, &mut rng));
            total_samples_basic += 1;
        }
    }
    let elapsed_basic = start.elapsed();

    // Benchmark: Different-value resample
    let start = Instant::now();
    let mut total_samples_diff = 0usize;
    for _ in 0..iters {
        for _ in 0..num_samples {
            std::hint::black_box(ppot_resample_different_value(
                &base_path,
                &marginals,
                &positions,
                &mut scratch,
                &mut rng,
            ));
            total_samples_diff += 1;
        }
    }
    let elapsed_diff = start.elapsed();

    // Benchmark: Support-constrained resample (Digit rule)
    let digit_support = TokenRule::Digit.support(vocab_size);
    let mut temp_scratch = vec![0.0f32; vocab_size];
    let start = Instant::now();
    let mut total_samples_support = 0usize;
    for _ in 0..iters {
        for _ in 0..num_samples {
            std::hint::black_box(ppot_resample_with_support(
                &base_path,
                &marginals,
                &positions,
                &digit_support,
                &mut temp_scratch,
                &mut rng,
            ));
            total_samples_support += 1;
        }
    }
    let elapsed_support = start.elapsed();

    let samples_per_sec = |total: usize, elapsed: std::time::Duration| -> f64 {
        total as f64 / elapsed.as_secs_f64()
    };

    (
        BenchResult {
            label: "PPoT Resample (basic)".into(),
            throughput: samples_per_sec(total_samples_basic, elapsed_basic),
            time_per_step_us: elapsed_basic.as_micros() as f64 / total_samples_basic as f64,
            avg_acceptance_len: num_samples as f64,
            color: (188, 143, 143),
            category: BenchCategory::Infrastructure,
        },
        BenchResult {
            label: "PPoT Resample (diff-value)".into(),
            throughput: samples_per_sec(total_samples_diff, elapsed_diff),
            time_per_step_us: elapsed_diff.as_micros() as f64 / total_samples_diff as f64,
            avg_acceptance_len: num_samples as f64,
            color: (205, 133, 63),
            category: BenchCategory::Infrastructure,
        },
        BenchResult {
            label: "PPoT Resample (digit)".into(),
            throughput: samples_per_sec(total_samples_support, elapsed_support),
            time_per_step_us: elapsed_support.as_micros() as f64 / total_samples_support as f64,
            avg_acceptance_len: num_samples as f64,
            color: (210, 180, 140),
            category: BenchCategory::Infrastructure,
        },
    )
}

/// Benchmark: End-to-end PPoT rescue comparison.
///
/// Compares three rescue strategies when speculative decoding rejects all paths:
/// 1. **Greedy fallback** — no rescue, just take argmax (baseline)
/// 2. **PPoT rescue** — Plan 026 random resampling with entropy selection
/// 3. **PPoT adaptive rescue** — Plan 027 adaptive with knowledge + strategy cycling
///
/// Simulates rejection by using a 0% acceptance verifier, forcing rescue on every step.
/// Measures wall-clock overhead and rescue success rate (valid paths found).
fn bench_ppot_rescue(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> (BenchResult, BenchResult, BenchResult) {
    let mut rng = Rng::new(99);
    let mut sctx = SpeculativeContext::new(draft_config);

    let vocab_size = draft_config.vocab_size;
    let ppot_config = PpotConfig::for_char_level().with_cached_support(vocab_size);
    let num_samples = ppot_config.num_samples;

    // Pre-generate marginals for reproducibility
    sctx.reset();
    let steps = dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
    let marginals: Vec<&[f32]> = (0..steps)
        .map(|step| sctx.marginal_slice(step, vocab_size))
        .collect();

    // Base path: argmax from marginals
    let base_path: Vec<usize> = marginals
        .iter()
        .map(|m| {
            m.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect();

    let mut scratch = vec![0.0f32; vocab_size];

    // ── 1. Greedy fallback (no rescue) ──
    // Warmup
    for _ in 0..warmup {
        sctx.reset();
        let _ = dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
    }

    let start = Instant::now();
    for _ in 0..iters {
        sctx.reset();
        let steps_now = dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
        // Greedy: argmax from first marginal
        let first_marginal = sctx.marginal_slice(0, vocab_size);
        if !first_marginal.is_empty() {
            let _greedy_token = sample_from_distribution(first_marginal, &mut rng);
        }
        std::hint::black_box(steps_now);
    }
    let elapsed_greedy = start.elapsed();

    // ── 2. PPoT rescue (Plan 026: random) ──
    // Warmup
    for _ in 0..warmup {
        sctx.reset();
        let _ = dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
        let mv: Vec<&[f32]> = (0..sctx.steps_populated)
            .map(|step| sctx.marginal_slice(step, vocab_size))
            .collect();
        let _ = ppot_rescue(
            &mv,
            &base_path,
            &NoScreeningPruner,
            &ppot_config,
            &mut scratch,
            &mut rng,
        );
    }

    let mut total_rescued_ppot = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        sctx.reset();
        let _ = dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
        let mv: Vec<&[f32]> = (0..sctx.steps_populated)
            .map(|step| sctx.marginal_slice(step, vocab_size))
            .collect();
        let result = ppot_rescue(
            &mv,
            &base_path,
            &NoScreeningPruner,
            &ppot_config,
            &mut scratch,
            &mut rng,
        );
        if result.is_some() {
            total_rescued_ppot += 1;
        }
    }
    let elapsed_ppot = start.elapsed();

    // ── 3. PPoT adaptive rescue (Plan 027: knowledge + strategy cycling) ──
    let mut knowledge = SessionKnowledge::from_config(&ppot_config);

    // Warmup
    for _ in 0..warmup {
        sctx.reset();
        let _ = dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
        let mv: Vec<&[f32]> = (0..sctx.steps_populated)
            .map(|step| sctx.marginal_slice(step, vocab_size))
            .collect();
        let _ = ppot_rescue_adaptive(
            &mv,
            &base_path,
            &NoScreeningPruner,
            &ppot_config,
            &mut knowledge,
            &mut scratch,
            &mut rng,
        );
    }

    // Reset knowledge after warmup
    knowledge = SessionKnowledge::from_config(&ppot_config);

    let mut total_rescued_adaptive = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        sctx.reset();
        let _ = dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
        let mv: Vec<&[f32]> = (0..sctx.steps_populated)
            .map(|step| sctx.marginal_slice(step, vocab_size))
            .collect();
        let result = ppot_rescue_adaptive(
            &mv,
            &base_path,
            &NoScreeningPruner,
            &ppot_config,
            &mut knowledge,
            &mut scratch,
            &mut rng,
        );
        if result.is_some() {
            total_rescued_adaptive += 1;
        }
    }
    let elapsed_adaptive = start.elapsed();

    let iters_f = iters as f64;

    (
        BenchResult {
            label: "PPoT Greedy Fallback".into(),
            throughput: iters_f / elapsed_greedy.as_secs_f64(),
            time_per_step_us: elapsed_greedy.as_micros() as f64 / iters_f,
            avg_acceptance_len: 1.0, // greedy always returns 1 token
            color: (169, 169, 169),
            category: BenchCategory::Infrastructure,
        },
        BenchResult {
            label: "PPoT Rescue (Plan 026)".into(),
            throughput: iters_f / elapsed_ppot.as_secs_f64(),
            time_per_step_us: elapsed_ppot.as_micros() as f64 / iters_f,
            avg_acceptance_len: total_rescued_ppot as f64 / iters_f * num_samples as f64,
            color: (70, 130, 180),
            category: BenchCategory::Infrastructure,
        },
        BenchResult {
            label: "PPoT Adaptive (Plan 027)".into(),
            throughput: iters_f / elapsed_adaptive.as_secs_f64(),
            time_per_step_us: elapsed_adaptive.as_micros() as f64 / iters_f,
            avg_acceptance_len: total_rescued_adaptive as f64 / iters_f * num_samples as f64,
            color: (100, 149, 237),
            category: BenchCategory::Infrastructure,
        },
    )
}
