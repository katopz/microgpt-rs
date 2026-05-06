use crate::speculative::{
    SimulatedVerifier, build_dd_tree, dflash_predict, dflash_predict_ar, sample_from_distribution,
    speculative_step_verifier,
};
use crate::transformer::{ForwardContext, KVCache, TransformerWeights, forward};
use crate::types::{Config, Rng, softmax};
use std::time::Instant;

#[cfg(feature = "leviathan")]
use crate::speculative::LeviathanVerifier;

/// Single benchmark result.
pub struct BenchResult {
    pub label: String,
    pub throughput: f64,
    pub time_per_step_us: f64,
    pub avg_acceptance_len: f64,
    pub color: (u8, u8, u8),
}

/// Run all benchmarks and return results.
pub fn run_all(config: &Config) -> Vec<BenchResult> {
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(config, &mut rng);

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
    let spec_ar = bench_speculative_ar(&draft_weights, &draft_config, warmup, iters);

    #[allow(unused_mut)]
    let mut results = vec![ar, dflash, ddtree, spec, spec_ar];

    #[cfg(feature = "leviathan")]
    {
        let leviathan = bench_leviathan(
            &draft_weights,
            &draft_config,
            &weights,
            config,
            warmup,
            iters,
        );
        results.push(leviathan);
    }

    results
}

fn bench_ar(
    weights: &TransformerWeights,
    config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let mut ctx = ForwardContext::new(config);
    let mut cache = KVCache::new(config);

    for _ in 0..warmup {
        cache.reset();
        let logits = forward(&mut ctx, weights, &mut cache, 0, 0, config);
        for logit in logits.iter_mut() {
            *logit /= config.temperature;
        }
        softmax(logits);
    }

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
    for _ in 0..warmup {
        let _ = dflash_predict(draft_weights, draft_config, 0, 0);
    }

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

    for _ in 0..warmup {
        let _ = build_dd_tree(&marginals, draft_config);
    }

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

/// Speculative decoding with SimulatedVerifier (DFlash + DDTree + simulated 75% acceptance).
fn bench_speculative(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let mut rng = Rng::new(99);
    let mut verifier = SimulatedVerifier::new(0.75);

    for _ in 0..warmup {
        let _ =
            speculative_step_verifier(draft_weights, draft_config, 0, 0, &mut rng, &mut verifier);
    }

    let mut total_accepted = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let (accepted, _) =
            speculative_step_verifier(draft_weights, draft_config, 0, 0, &mut rng, &mut verifier);
        total_accepted += accepted.len();
    }
    let elapsed = start.elapsed();

    let tps = total_accepted as f64 / elapsed.as_secs_f64();
    let avg_accept = total_accepted as f64 / iters as f64;
    BenchResult {
        label: "Speculative (Simulated)".into(),
        throughput: tps,
        time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
        avg_acceptance_len: avg_accept,
        color: (255, 165, 0),
    }
}

/// Speculative decoding with AR drafting + DDTree + simulated acceptance.
/// Measures pure AR drafting benefit without target model verification cost.
fn bench_speculative_ar(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let mut rng = Rng::new(99);

    for _ in 0..warmup {
        let _ = run_speculative_ar_step(draft_weights, draft_config, &mut rng);
    }

    let mut total_accepted = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let accepted = run_speculative_ar_step(draft_weights, draft_config, &mut rng);
        total_accepted += accepted.len();
    }
    let elapsed = start.elapsed();

    let tps = total_accepted as f64 / elapsed.as_secs_f64();
    let avg_accept = total_accepted as f64 / iters as f64;
    BenchResult {
        label: "Speculative (AR Draft)".into(),
        throughput: tps,
        time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
        avg_acceptance_len: avg_accept,
        color: (255, 200, 0),
    }
}

/// AR draft + DDTree + simulated acceptance + bonus token.
fn run_speculative_ar_step(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    rng: &mut Rng,
) -> Vec<usize> {
    let draft_result = dflash_predict_ar(draft_weights, draft_config, 0, 0, rng);
    let tree = build_dd_tree(&draft_result.marginals, draft_config);

    // Extract best path (highest-scored token at each depth)
    let max_depth = tree.iter().map(|n| n.depth).max().unwrap_or(0);
    let mut path = Vec::with_capacity(max_depth + 1);
    for depth in 0..=max_depth {
        let best = tree
            .iter()
            .filter(|n| n.depth == depth)
            .max_by_key(|n| (n.score * 1e6) as i64);
        match best {
            Some(node) => path.push(node.token_idx),
            None => break,
        }
    }

    if path.is_empty() {
        return vec![sample_from_distribution(
            draft_result
                .marginals
                .first()
                .map(|m| m.as_slice())
                .unwrap_or(&[1.0]),
            rng,
        )];
    }

    // Simulated acceptance: 75% cap
    let acceptance_rate = 0.75;
    let max_accept = ((path.len() as f32) * acceptance_rate).ceil() as usize;
    let accepted: Vec<usize> = path.into_iter().take(max_accept.max(1)).collect();

    // Bonus token: if all accepted, sample +1 from last marginal
    if accepted.len() == max_accept && !draft_result.marginals.is_empty() {
        let last_marginal = draft_result.marginals.last().unwrap();
        let bonus = sample_from_distribution(last_marginal, rng);
        let mut result = accepted;
        result.push(bonus);
        return result;
    }

    accepted
}

/// Leviathan Algorithm 1: AR draft + real target p/q verification.
/// Requires `--features leviathan` to run.
#[cfg(feature = "leviathan")]
fn bench_leviathan(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let mut rng = Rng::new(99);
    let mut verifier = LeviathanVerifier::new(target_weights, target_config);

    for _ in 0..warmup {
        let _ =
            speculative_step_verifier(draft_weights, draft_config, 0, 0, &mut rng, &mut verifier);
    }

    let mut total_accepted = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let (accepted, _) =
            speculative_step_verifier(draft_weights, draft_config, 0, 0, &mut rng, &mut verifier);
        total_accepted += accepted.len();
    }
    let elapsed = start.elapsed();

    let tps = total_accepted as f64 / elapsed.as_secs_f64();
    let avg_accept = total_accepted as f64 / iters as f64;
    BenchResult {
        label: "Leviathan (Algorithm 1)".into(),
        throughput: tps,
        time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
        avg_acceptance_len: avg_accept,
        color: (148, 0, 211),
    }
}
