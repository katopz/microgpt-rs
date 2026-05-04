use crate::speculative::{build_dd_tree, dflash_predict, speculative_step};
use crate::transformer::{KVCache, TransformerWeights, forward};
use crate::types::{Config, softmax};
use std::time::Instant;

/// Single benchmark result.
pub struct BenchResult {
    pub label: String,
    pub throughput: f64,
    pub time_per_step_us: f64,
    pub avg_acceptance_len: f64,
    pub color: (u8, u8, u8),
}

/// Run all 4 benchmarks and return results.
pub fn run_all(config: &Config) -> Vec<BenchResult> {
    let mut rng = crate::types::Rng::new(42);
    let weights = TransformerWeights::new(config, &mut rng);

    let warmup = 1000;
    let iters = 50000;

    println!("\n📊 Running benchmarks ({iters} iterations, {warmup} warmup)...\n");

    let ar = bench_ar(&weights, config, warmup, iters);
    let dflash = bench_dflash(&weights, config, warmup, iters);
    let ddtree = bench_ddtree(&weights, config, warmup, iters);
    let spec = bench_speculative(&weights, config, warmup, iters);

    vec![ar, dflash, ddtree, spec]
}

fn bench_ar(
    weights: &TransformerWeights,
    config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let mut cache = KVCache::new(config);

    // Warmup
    for _ in 0..warmup {
        cache.reset();
        let logits = forward(weights, &mut cache, 0, 0, config);
        let mut probs = logits;
        for p in probs.iter_mut() {
            *p /= config.temperature;
        }
        softmax(&mut probs);
    }

    // Timed run
    let start = Instant::now();
    for _ in 0..iters {
        cache.reset();
        let logits = forward(weights, &mut cache, 0, 0, config);
        let mut probs = logits;
        for p in probs.iter_mut() {
            *p /= config.temperature;
        }
        softmax(&mut probs);
    }
    let elapsed = start.elapsed();

    let tps = iters as f64 / elapsed.as_secs_f64();
    BenchResult {
        label: "Transformer AR".into(),
        throughput: tps,
        time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
        avg_acceptance_len: 1.0,
        color: (70, 130, 180),
    }
}

fn bench_dflash(
    weights: &TransformerWeights,
    config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    // Warmup
    for _ in 0..warmup {
        let _ = dflash_predict(weights, 0, 0, config);
    }

    // Timed run
    let mut total_draft_tokens = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let marginals = dflash_predict(weights, 0, 0, config);
        total_draft_tokens += marginals.len();
    }
    let elapsed = start.elapsed();

    let tps = total_draft_tokens as f64 / elapsed.as_secs_f64();
    BenchResult {
        label: "DFlash Draft".into(),
        throughput: tps,
        time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
        avg_acceptance_len: config.draft_lookahead as f64,
        color: (255, 99, 71),
    }
}

fn bench_ddtree(
    weights: &TransformerWeights,
    config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let marginals = dflash_predict(weights, 0, 0, config);

    // Warmup
    for _ in 0..warmup {
        let _ = build_dd_tree(&marginals, config);
    }

    // Timed run
    let start = Instant::now();
    for _ in 0..iters {
        let _ = build_dd_tree(&marginals, config);
    }
    let elapsed = start.elapsed();

    let ops = iters as f64 / elapsed.as_secs_f64();
    BenchResult {
        label: "DDTree Build".into(),
        throughput: ops,
        time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
        avg_acceptance_len: 0.0,
        color: (50, 205, 50),
    }
}

fn bench_speculative(
    weights: &TransformerWeights,
    config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let mut rng = crate::types::Rng::new(99);

    // Warmup
    for _ in 0..warmup {
        let _ = speculative_step(weights, 0, 0, config, &mut rng);
    }

    // Timed run
    let mut total_accepted = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let (accepted, _) = speculative_step(weights, 0, 0, config, &mut rng);
        total_accepted += accepted.len();
    }
    let elapsed = start.elapsed();

    let tps = total_accepted as f64 / elapsed.as_secs_f64();
    let avg_accept = total_accepted as f64 / iters as f64;
    BenchResult {
        label: "DFlash+DDTree".into(),
        throughput: tps,
        time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
        avg_acceptance_len: avg_accept,
        color: (255, 165, 0),
    }
}
