use crate::speculative::{build_dd_tree, dflash_predict};
use crate::transformer::{ForwardContext, KVCache, TransformerWeights, forward};
use crate::types::{Config, Rng, softmax};
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
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(config, &mut rng);

    // Separate draft model (~4x smaller) for speculative decoding
    let draft_config = Config::draft();
    let mut draft_rng = Rng::new(99);
    let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

    let warmup = 1000;
    let iters = 50000;

    println!("\n📊 Running benchmarks ({iters} iterations, {warmup} warmup)...");
    println!(
        "   Target model: embd={}, heads={}, mlp={}",
        config.n_embd, config.n_head, config.mlp_hidden
    );
    println!(
        "   Draft  model: embd={}, heads={}, mlp={}",
        draft_config.n_embd, draft_config.n_head, draft_config.mlp_hidden
    );

    let ar = bench_ar(&weights, config, warmup, iters);
    let dflash = bench_dflash(&draft_weights, &draft_config, warmup, iters);
    let ddtree = bench_ddtree(&draft_weights, &draft_config, warmup, iters);
    let spec = bench_speculative(&draft_weights, &draft_config, warmup, iters);

    vec![ar, dflash, ddtree, spec]
}

fn bench_ar(
    weights: &TransformerWeights,
    config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let mut ctx = ForwardContext::new(config);
    let mut cache = KVCache::new(config);

    // Warmup
    for _ in 0..warmup {
        cache.reset();
        let logits = forward(&mut ctx, weights, &mut cache, 0, 0, config);
        for logit in logits.iter_mut() {
            *logit /= config.temperature;
        }
        softmax(logits);
    }

    // Timed run
    let start = Instant::now();
    for _ in 0..iters {
        cache.reset();
        let logits = forward(&mut ctx, weights, &mut cache, 0, 0, config);
        for logit in logits.iter_mut() {
            *logit /= config.temperature;
        }
        softmax(logits);
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
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    // Warmup
    for _ in 0..warmup {
        let _ = dflash_predict(draft_weights, draft_config, 0, 0);
    }

    // Timed run
    let mut total_draft_tokens = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let marginals = dflash_predict(draft_weights, draft_config, 0, 0);
        total_draft_tokens += marginals.len();
    }
    let elapsed = start.elapsed();

    let tps = total_draft_tokens as f64 / elapsed.as_secs_f64();
    BenchResult {
        label: "DFlash".into(),
        throughput: tps,
        time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
        avg_acceptance_len: draft_config.draft_lookahead as f64,
        color: (255, 99, 71),
    }
}

fn bench_ddtree(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let marginals = dflash_predict(draft_weights, draft_config, 0, 0);

    // Warmup
    for _ in 0..warmup {
        let _ = build_dd_tree(&marginals, draft_config);
    }

    // Timed run
    let start = Instant::now();
    for _ in 0..iters {
        let _ = build_dd_tree(&marginals, draft_config);
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
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let mut rng = Rng::new(99);

    // Warmup
    for _ in 0..warmup {
        let _ = run_speculative_step(draft_weights, draft_config, &mut rng);
    }

    // Timed run
    let mut total_accepted = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let accepted = run_speculative_step(draft_weights, draft_config, &mut rng);
        total_accepted += accepted.len();
    }
    let elapsed = start.elapsed();

    let tps = total_accepted as f64 / elapsed.as_secs_f64();
    let avg_accept = total_accepted as f64 / iters as f64;
    BenchResult {
        label: "Speculative Decoding".into(),
        throughput: tps,
        time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
        avg_acceptance_len: avg_accept,
        color: (255, 165, 0),
    }
}

/// Sequential speculative step: DFlash draft → DDTree build → accept path.
/// Avoids rayon overhead for tiny draft model.
fn run_speculative_step(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    _rng: &mut Rng,
) -> Vec<usize> {
    let marginals = dflash_predict(draft_weights, draft_config, 0, 0);
    let tree = build_dd_tree(&marginals, draft_config);

    // Extract best path (highest-scored token at each depth)
    let max_depth = tree.iter().map(|n| n.depth).max().unwrap_or(0);
    let mut path = Vec::with_capacity(max_depth + 1);
    for depth in 0..=max_depth {
        let best = tree
            .iter()
            .filter(|n| n.depth == depth)
            .max_by_key(|n| (n.score * 1e6) as i64);
        if let Some(node) = best {
            path.push(node.token_idx);
        } else {
            break;
        }
    }

    // Accept ~75% of draft tokens (simulated verification)
    let max_accept = ((path.len() as f32) * 0.75).ceil() as usize;
    path.into_iter().take(max_accept.max(1)).collect()
}
