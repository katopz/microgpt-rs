use crate::speculative::{
    AttentionScorer, NoPruner, SimulatedVerifier, SpeculativeContext, TreeBuilder, compress_prompt,
    dflash_predict_ar_with, dflash_predict_with, extract_best_path_into, sample_from_distribution,
    speculative_step_verifier,
};
use crate::transformer::{
    ForwardContext, MultiLayerKVCache, PagedKVCache, TransformerWeights, forward, forward_paged,
    generate_into, tokens_to_string,
};
use crate::types::{Config, Rng, softmax};
use rayon::prelude::*;
use std::io::Write;
use std::time::Instant;

use crate::speculative::{
    LeviathanVerifier, speculative_step_conditioned_with, speculative_step_rollback_with,
};

/// Single benchmark result.
pub struct BenchResult {
    pub label: String,
    pub throughput: f64,
    pub time_per_step_us: f64,
    pub avg_acceptance_len: f64,
    pub color: (u8, u8, u8),
}

/// Save benchmark results to numbered CSV (e.g. `bench/024_results.csv`).
///
/// CSV columns: `commit,date,method,throughput,us_per_step,avg_accept_len`
/// One file per run, numbered to match the PNG chart. Always writes header + data.
pub fn save_results_csv(results: &[BenchResult], path: &str) -> std::io::Result<()> {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());

    let date = chrono_like_now();

    let mut file = std::fs::File::create(path)?;

    writeln!(
        file,
        "commit,date,method,throughput,us_per_step,avg_accept_len"
    )?;

    for r in results {
        writeln!(
            file,
            "{},{},{},{:.0},{:.2},{:.2}",
            commit, date, r.label, r.throughput, r.time_per_step_us, r.avg_acceptance_len,
        )?;
    }

    Ok(())
}

/// Simple timestamp without chrono dependency: `YYYY-MM-DDTHH:MM:SS`
fn chrono_like_now() -> String {
    let output = std::process::Command::new("date")
        .arg("+%Y-%m-%dT%H:%M:%S")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    output.unwrap_or_else(|| "unknown".into())
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

        // Snapshot/rollback overhead
        let (no_rollback, with_rollback) = bench_snapshot_rollback(
            &draft_weights,
            &draft_config,
            &weights,
            config,
            warmup,
            iters,
        );
        results.push(no_rollback);
        results.push(with_rollback);

        // Conditioned vs unconditioned draft
        let (uncond_br, cond_br) = bench_conditioned_vs_unconditioned(
            &draft_weights,
            &draft_config,
            &weights,
            config,
            warmup,
            iters,
        );
        results.push(uncond_br);
        results.push(cond_br);
    }

    // Prefill compression comparison
    let (nocompress_br, compress_br) =
        bench_prefill_compression(&draft_weights, &draft_config, warmup, iters);
    results.push(nocompress_br);
    results.push(compress_br);

    // Chain-seed comparison
    let (no_chain, chain) = bench_ddtree_chain_seed(&draft_weights, &draft_config, warmup, iters);
    results.push(no_chain);
    results.push(chain);

    // Paged vs flat cache comparison
    let (flat_br, paged_br) = bench_paged_vs_flat_cache(config);
    results.push(flat_br);
    results.push(paged_br);

    // WasmPruner vs NoPruner DDTree build comparison
    #[cfg(feature = "wasm")]
    results.extend(bench_wasm_vs_no_pruner(
        &draft_weights,
        &draft_config,
        warmup,
        iters,
    ));

    results
}

fn bench_ar(
    weights: &TransformerWeights,
    config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);

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
    let mut sctx = SpeculativeContext::new(draft_config);

    for _ in 0..warmup {
        sctx.reset();
        dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
    }

    let mut total_draft_tokens = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        sctx.reset();
        let steps = dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
        total_draft_tokens += steps;
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
    let mut sctx = SpeculativeContext::new(draft_config);
    sctx.reset();
    dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
    let marginals_view: Vec<&[f32]> = (0..sctx.steps_populated)
        .map(|step| sctx.marginal_slice(step, draft_config.vocab_size))
        .collect();

    let mut tree_builder = TreeBuilder::new(draft_config);

    for _ in 0..warmup {
        let _ = tree_builder.build(&marginals_view, draft_config, &NoPruner, false);
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _ = tree_builder.build(&marginals_view, draft_config, &NoPruner, false);
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

#[cfg(feature = "wasm")]
fn bench_wasm_vs_no_pruner(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> Vec<BenchResult> {
    use crate::wasm::WasmPruner;

    let wasm_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../riir-validator-sdk/target/wasm32-unknown-unknown/release/examples/bracket_validator.wasm");

    let wasm_pruner = match WasmPruner::load_from_file(wasm_path.to_str().unwrap_or("")) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("⚠️  Skipping WasmPruner benchmark: {e}");
            return Vec::new();
        }
    };

    let mut sctx = SpeculativeContext::new(draft_config);
    sctx.reset();
    dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
    let marginals_view: Vec<&[f32]> = (0..sctx.steps_populated)
        .map(|step| sctx.marginal_slice(step, draft_config.vocab_size))
        .collect();

    let mut tree_builder = TreeBuilder::new(draft_config);

    for _ in 0..warmup {
        let _ = tree_builder.build(&marginals_view, draft_config, &NoPruner, false);
    }
    for _ in 0..warmup {
        let _ = tree_builder.build(&marginals_view, draft_config, &wasm_pruner, false);
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _ = tree_builder.build(&marginals_view, draft_config, &NoPruner, false);
    }
    let no_pruner_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        let _ = tree_builder.build(&marginals_view, draft_config, &wasm_pruner, false);
    }
    let wasm_elapsed = start.elapsed();

    let wasm_name = wasm_pruner.name();
    vec![
        BenchResult {
            label: "DDTree+NoPruner".into(),
            throughput: iters as f64 / no_pruner_elapsed.as_secs_f64(),
            time_per_step_us: no_pruner_elapsed.as_micros() as f64 / iters as f64,
            avg_acceptance_len: 0.0,
            color: (100, 200, 100),
        },
        BenchResult {
            label: format!("DDTree+WasmPruner({wasm_name})"),
            throughput: iters as f64 / wasm_elapsed.as_secs_f64(),
            time_per_step_us: wasm_elapsed.as_micros() as f64 / iters as f64,
            avg_acceptance_len: 0.0,
            color: (200, 100, 200),
        },
    ]
}

fn bench_speculative(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let mut rng = Rng::new(99);
    let mut verifier = SimulatedVerifier::new(0.75, draft_config);

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
    let mut sctx = SpeculativeContext::new(draft_config);
    let mut tree_builder = TreeBuilder::new(draft_config);

    for _ in 0..warmup {
        let _ = run_speculative_ar_step(
            &mut sctx,
            &mut tree_builder,
            draft_weights,
            draft_config,
            &mut rng,
        );
    }

    let mut total_accepted = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let accepted = run_speculative_ar_step(
            &mut sctx,
            &mut tree_builder,
            draft_weights,
            draft_config,
            &mut rng,
        );
        total_accepted += accepted;
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
    sctx: &mut SpeculativeContext,
    tree_builder: &mut TreeBuilder,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    rng: &mut Rng,
) -> usize {
    // 1. Zero-alloc AR draft
    sctx.reset();
    let steps = dflash_predict_ar_with(sctx, draft_weights, draft_config, 0, 0, rng);
    let vocab_size = draft_config.vocab_size;

    // 2. Build tree from marginals
    let marginals_view: Vec<&[f32]> = (0..steps)
        .map(|step| sctx.marginal_slice(step, vocab_size))
        .collect();
    let tree = tree_builder.build(&marginals_view, draft_config, &NoPruner, false);

    // 3. Extract best path
    extract_best_path_into(tree, &mut sctx.path_buf);

    if sctx.path_buf.is_empty() {
        let first_marginal = sctx.marginal_slice(0, vocab_size);
        return sample_from_distribution(
            if first_marginal.is_empty() {
                &[1.0]
            } else {
                first_marginal
            },
            rng,
        );
    }

    // 4. Simulated acceptance: 75% cap
    let acceptance_rate = 0.75;
    let max_accept = ((sctx.path_buf.len() as f32) * acceptance_rate).ceil() as usize;
    sctx.accepted_buf.clear();
    sctx.accepted_buf
        .extend(sctx.path_buf.iter().take(max_accept.max(1)).copied());

    // 5. Bonus token
    if sctx.accepted_buf.len() == max_accept && steps > 0 {
        let last_step = steps - 1;
        let last_marginal = sctx.marginal_slice(last_step, vocab_size);
        let bonus = sample_from_distribution(
            if last_marginal.is_empty() {
                &[1.0]
            } else {
                last_marginal
            },
            rng,
        );
        sctx.accepted_buf.push(bonus);
    }

    sctx.accepted_buf.len()
}

/// Benchmark: Chain-seed DDTree vs regular DDTree.
/// Compares chain_seed=true vs chain_seed=false acceptance length.
pub fn bench_ddtree_chain_seed(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> (BenchResult, BenchResult) {
    let mut sctx = SpeculativeContext::new(draft_config);
    sctx.reset();
    dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
    let mv: Vec<&[f32]> = (0..sctx.steps_populated)
        .map(|step| sctx.marginal_slice(step, draft_config.vocab_size))
        .collect();

    let mut tree_builder = TreeBuilder::new(draft_config);

    // Warmup
    for _ in 0..warmup {
        let _ = tree_builder.build(&mv, draft_config, &NoPruner, false);
        let _ = tree_builder.build(&mv, draft_config, &NoPruner, true);
    }

    // Benchmark no-chain
    let start = Instant::now();
    let mut total_nodes_no_chain = 0usize;
    for _ in 0..iters {
        let tree = tree_builder.build(&mv, draft_config, &NoPruner, false);
        total_nodes_no_chain += tree.len();
    }
    let elapsed_no_chain = start.elapsed();

    // Benchmark chain-seed
    let start = Instant::now();
    let mut total_nodes_chain = 0usize;
    for _ in 0..iters {
        let tree = tree_builder.build(&mv, draft_config, &NoPruner, true);
        total_nodes_chain += tree.len();
    }
    let elapsed_chain = start.elapsed();

    let ops_no_chain = iters as f64 / elapsed_no_chain.as_secs_f64();
    let ops_chain = iters as f64 / elapsed_chain.as_secs_f64();

    (
        BenchResult {
            label: "DDTree (no chain)".into(),
            throughput: ops_no_chain,
            time_per_step_us: elapsed_no_chain.as_micros() as f64 / iters as f64,
            avg_acceptance_len: total_nodes_no_chain as f64 / iters as f64,
            color: (50, 205, 50),
        },
        BenchResult {
            label: "DDTree (chain-seed)".into(),
            throughput: ops_chain,
            time_per_step_us: elapsed_chain.as_micros() as f64 / iters as f64,
            avg_acceptance_len: total_nodes_chain as f64 / iters as f64,
            color: (0, 200, 100),
        },
    )
}

/// Benchmark: DDTree budget sweep across multiple budgets.
/// Returns results for each budget with and without chain-seed.
pub fn bench_ddtree_budget_sweep(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    budgets: &[usize],
    warmup: usize,
    iters: usize,
) -> Vec<BenchResult> {
    let mut sctx = SpeculativeContext::new(draft_config);
    sctx.reset();
    dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
    let mv: Vec<&[f32]> = (0..sctx.steps_populated)
        .map(|step| sctx.marginal_slice(step, draft_config.vocab_size))
        .collect();

    let mut tree_builder = TreeBuilder::new(draft_config);
    let mut results = Vec::with_capacity(budgets.len() * 2);

    for &budget in budgets {
        let mut sweep_config = draft_config.clone();
        sweep_config.tree_budget = budget;

        // Warmup
        for _ in 0..warmup {
            let _ = tree_builder.build(&mv, &sweep_config, &NoPruner, false);
            let _ = tree_builder.build(&mv, &sweep_config, &NoPruner, true);
        }

        // Benchmark no-chain
        let start = Instant::now();
        let mut total_nodes = 0usize;
        for _ in 0..iters {
            let tree = tree_builder.build(&mv, &sweep_config, &NoPruner, false);
            total_nodes += tree.len();
        }
        let elapsed = start.elapsed();

        results.push(BenchResult {
            label: format!("DDTree budget={budget}"),
            throughput: iters as f64 / elapsed.as_secs_f64(),
            time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
            avg_acceptance_len: total_nodes as f64 / iters as f64,
            color: (100, 149, 237),
        });

        // Benchmark chain-seed
        let start = Instant::now();
        let mut total_nodes = 0usize;
        for _ in 0..iters {
            let tree = tree_builder.build(&mv, &sweep_config, &NoPruner, true);
            total_nodes += tree.len();
        }
        let elapsed = start.elapsed();

        results.push(BenchResult {
            label: format!("DDTree chain budget={budget}"),
            throughput: iters as f64 / elapsed.as_secs_f64(),
            time_per_step_us: elapsed.as_micros() as f64 / iters as f64,
            avg_acceptance_len: total_nodes as f64 / iters as f64,
            color: (60, 179, 113),
        });
    }

    results
}

/// Leviathan Algorithm 1: AR draft + real target p/q verification.
fn bench_leviathan(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let mut rng = Rng::new(99);
    let mut verifier = LeviathanVerifier::new(target_weights, target_config, draft_config);

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

/// Task 3.8: Benchmark snapshot/rollback overhead vs no-rollback speculative step.
/// Measures the cost of KV-Cache snapshot + restore per speculative step.
/// Snapshot cost: O(n_layer × pos × kv_dim) — cheap at our model scale.
fn bench_snapshot_rollback(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    warmup: usize,
    iters: usize,
) -> (BenchResult, BenchResult) {
    // ── No-rollback: standard Leviathan (resets cache each step) ──
    let mut rng_no_rb = Rng::new(99);
    let mut verifier = LeviathanVerifier::new(target_weights, target_config, draft_config);

    for _ in 0..warmup {
        let _ = speculative_step_verifier(
            draft_weights,
            draft_config,
            0,
            0,
            &mut rng_no_rb,
            &mut verifier,
        );
    }

    let mut total_no_rb = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let (accepted, _) = speculative_step_verifier(
            draft_weights,
            draft_config,
            0,
            0,
            &mut rng_no_rb,
            &mut verifier,
        );
        total_no_rb += accepted.len();
    }
    let elapsed_no_rb = start.elapsed();

    let no_rollback = BenchResult {
        label: "Leviathan (no rollback)".into(),
        throughput: total_no_rb as f64 / elapsed_no_rb.as_secs_f64(),
        time_per_step_us: elapsed_no_rb.as_micros() as f64 / iters as f64,
        avg_acceptance_len: total_no_rb as f64 / iters as f64,
        color: (180, 100, 220),
    };

    // ── With rollback: speculative_step_rollback_with (zero-alloc, snapshots + restores) ──
    let mut rng_rb = Rng::new(99);
    let mut target_ctx = ForwardContext::new(target_config);
    let mut target_cache = MultiLayerKVCache::new(target_config);
    let mut draft_sctx = SpeculativeContext::new(draft_config);
    let mut tree_builder = TreeBuilder::new(draft_config);
    let mut probs_buf = vec![0.0f32; target_config.vocab_size];
    let mut residual_buf = vec![0.0f32; target_config.vocab_size];

    for _ in 0..warmup {
        target_cache.reset();
        draft_sctx.reset();
        let _ = speculative_step_rollback_with(
            &mut draft_sctx,
            &mut tree_builder,
            draft_weights,
            draft_config,
            target_weights,
            target_config,
            &mut target_ctx,
            &mut target_cache,
            &mut probs_buf,
            &mut residual_buf,
            0,
            0,
            &mut rng_rb,
        );
    }

    let mut total_rb = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        target_cache.reset();
        draft_sctx.reset();
        let (accepted, _) = speculative_step_rollback_with(
            &mut draft_sctx,
            &mut tree_builder,
            draft_weights,
            draft_config,
            target_weights,
            target_config,
            &mut target_ctx,
            &mut target_cache,
            &mut probs_buf,
            &mut residual_buf,
            0,
            0,
            &mut rng_rb,
        );
        total_rb += accepted.len();
    }
    let elapsed_rb = start.elapsed();

    let with_rollback = BenchResult {
        label: "Leviathan (w/ rollback)".into(),
        throughput: total_rb as f64 / elapsed_rb.as_secs_f64(),
        time_per_step_us: elapsed_rb.as_micros() as f64 / iters as f64,
        avg_acceptance_len: total_rb as f64 / iters as f64,
        color: (220, 120, 180),
    };

    (no_rollback, with_rollback)
}

/// Task 5.4: Benchmark conditioned vs unconditioned draft marginals.
/// Measures acceptance length with target-conditioned draft (DFlash-inspired)
/// vs standard unconditioned draft, using simulated acceptance.
/// Also reports approximate KL divergence from the target distribution.
fn bench_conditioned_vs_unconditioned(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    warmup: usize,
    iters: usize,
) -> (BenchResult, BenchResult) {
    // ── Unconditioned: SimulatedVerifier (standard DFlash) ──
    let mut rng_uncond = Rng::new(99);
    let mut verifier = SimulatedVerifier::new(0.75, draft_config);

    for _ in 0..warmup {
        let _ = speculative_step_verifier(
            draft_weights,
            draft_config,
            0,
            0,
            &mut rng_uncond,
            &mut verifier,
        );
    }

    let mut total_uncond = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let (accepted, _) = speculative_step_verifier(
            draft_weights,
            draft_config,
            0,
            0,
            &mut rng_uncond,
            &mut verifier,
        );
        total_uncond += accepted.len();
    }
    let elapsed_uncond = start.elapsed();

    let uncond = BenchResult {
        label: "Spec (unconditioned)".into(),
        throughput: total_uncond as f64 / elapsed_uncond.as_secs_f64(),
        time_per_step_us: elapsed_uncond.as_micros() as f64 / iters as f64,
        avg_acceptance_len: total_uncond as f64 / iters as f64,
        color: (255, 140, 0),
    };

    // ── Conditioned: target hidden state seeds draft KV cache (zero-alloc) ──
    let mut rng_cond = Rng::new(99);
    let mut target_ctx = ForwardContext::new(target_config);
    let mut target_cache = MultiLayerKVCache::new(target_config);
    let mut draft_sctx = SpeculativeContext::new(draft_config);
    let mut tree_builder = TreeBuilder::new(draft_config);
    let mut probs_buf = vec![0.0f32; target_config.vocab_size];

    for _ in 0..warmup {
        target_cache.reset();
        draft_sctx.reset();
        let _ = speculative_step_conditioned_with(
            &mut draft_sctx,
            &mut tree_builder,
            draft_weights,
            draft_config,
            target_weights,
            target_config,
            &mut target_ctx,
            &mut target_cache,
            &mut probs_buf,
            0,
            0,
            &mut rng_cond,
        );
    }

    let mut total_cond = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        target_cache.reset();
        draft_sctx.reset();
        let (accepted, _) = speculative_step_conditioned_with(
            &mut draft_sctx,
            &mut tree_builder,
            draft_weights,
            draft_config,
            target_weights,
            target_config,
            &mut target_ctx,
            &mut target_cache,
            &mut probs_buf,
            0,
            0,
            &mut rng_cond,
        );
        total_cond += accepted.len();
    }
    let elapsed_cond = start.elapsed();

    let cond = BenchResult {
        label: "Spec (conditioned)".into(),
        throughput: total_cond as f64 / elapsed_cond.as_secs_f64(),
        time_per_step_us: elapsed_cond.as_micros() as f64 / iters as f64,
        avg_acceptance_len: total_cond as f64 / iters as f64,
        color: (0, 180, 180),
    };

    (uncond, cond)
}

fn bench_prefill_compression(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> (BenchResult, BenchResult) {
    let prompt_len = draft_config.block_size * 4;
    let prompt_tokens: Vec<usize> = (0..prompt_len)
        .map(|i| i % draft_config.vocab_size)
        .collect();

    let scorer = AttentionScorer;
    let mut scores_buf = vec![0.0f32; prompt_len];
    let mut sctx = SpeculativeContext::new(draft_config);

    // ── No compression (keep_ratio=1.0) ──
    for _ in 0..warmup {
        scorer.score_with(
            &mut sctx,
            draft_weights,
            draft_config,
            &prompt_tokens,
            &mut scores_buf,
        );
        let _ = compress_prompt(&scores_buf, 1.0, 0, 0);
    }

    let mut total_nocompress = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        scorer.score_with(
            &mut sctx,
            draft_weights,
            draft_config,
            &prompt_tokens,
            &mut scores_buf,
        );
        let selected = compress_prompt(&scores_buf, 1.0, 0, 0);
        total_nocompress += selected.len();
    }
    let elapsed_nocompress = start.elapsed();

    let nocompress = BenchResult {
        label: "Prefill (no compress)".into(),
        throughput: total_nocompress as f64 / elapsed_nocompress.as_secs_f64(),
        time_per_step_us: elapsed_nocompress.as_micros() as f64 / iters as f64,
        avg_acceptance_len: total_nocompress as f64 / iters as f64,
        color: (180, 180, 180),
    };

    // ── Compressed (keep_ratio=0.1) ──
    for _ in 0..warmup {
        scorer.score_with(
            &mut sctx,
            draft_weights,
            draft_config,
            &prompt_tokens,
            &mut scores_buf,
        );
        let _ = compress_prompt(&scores_buf, 0.1, 0, 0);
    }

    let mut total_compress = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        scorer.score_with(
            &mut sctx,
            draft_weights,
            draft_config,
            &prompt_tokens,
            &mut scores_buf,
        );
        let selected = compress_prompt(&scores_buf, 0.1, 0, 0);
        total_compress += selected.len();
    }
    let elapsed_compress = start.elapsed();

    let compress = BenchResult {
        label: "Prefill (compressed)".into(),
        throughput: total_compress as f64 / elapsed_compress.as_secs_f64(),
        time_per_step_us: elapsed_compress.as_micros() as f64 / iters as f64,
        avg_acceptance_len: total_compress as f64 / iters as f64,
        color: (0, 200, 100),
    };

    (nocompress, compress)
}

/// Benchmark: paged KV cache vs flat KV cache forward pass throughput.
///
/// Measures `forward()` (flat MultiLayerKVCache) vs `forward_paged()` (PagedKVCache)
/// over multiple positions, reporting tokens/sec and µs/step for each.
pub fn bench_paged_vs_flat_cache(config: &Config) -> (BenchResult, BenchResult) {
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(config, &mut rng);
    let iters = 200;

    // Warm up both paths
    {
        let mut ctx = ForwardContext::new(config);
        let mut cache = MultiLayerKVCache::new(config);
        let _ = forward(&mut ctx, &weights, &mut cache, 0, 0, config);
    }
    {
        let mut ctx = ForwardContext::new(config);
        let mut cache = PagedKVCache::new(config, 1);
        let _ = forward_paged(&mut ctx, &weights, &mut cache, 0, 0, 0, config);
    }

    // Benchmark flat cache
    let mut ctx_flat = ForwardContext::new(config);
    let mut cache_flat = MultiLayerKVCache::new(config);
    let start_flat = Instant::now();
    for _ in 0..iters {
        cache_flat.reset();
        let max_pos = config.block_size.min(8);
        for pos in 0..max_pos {
            let _ = forward(&mut ctx_flat, &weights, &mut cache_flat, pos, pos, config);
        }
    }
    let elapsed_flat = start_flat.elapsed();

    // Benchmark paged cache
    let mut ctx_paged = ForwardContext::new(config);
    let mut cache_paged = PagedKVCache::new(config, 1);
    let start_paged = Instant::now();
    for _ in 0..iters {
        cache_paged.reset();
        let max_pos = config.block_size.min(8);
        for pos in 0..max_pos {
            let _ = forward_paged(
                &mut ctx_paged,
                &weights,
                &mut cache_paged,
                0,
                pos,
                pos,
                config,
            );
        }
    }
    let elapsed_paged = start_paged.elapsed();

    let steps_per_iter = config.block_size.min(8) as f64;

    let flat_result = BenchResult {
        label: "forward (flat)".into(),
        throughput: iters as f64 * steps_per_iter / elapsed_flat.as_secs_f64(),
        time_per_step_us: elapsed_flat.as_micros() as f64 / (iters as f64 * steps_per_iter),
        avg_acceptance_len: 0.0,
        color: (100, 149, 237),
    };

    let paged_result = BenchResult {
        label: "forward_paged".into(),
        throughput: iters as f64 * steps_per_iter / elapsed_paged.as_secs_f64(),
        time_per_step_us: elapsed_paged.as_micros() as f64 / (iters as f64 * steps_per_iter),
        avg_acceptance_len: 0.0,
        color: (255, 165, 0),
    };

    (flat_result, paged_result)
}

/// Run all core benchmarks in parallel using rayon's `par_iter`.
///
/// Same core benchmarks as `run_all()` but runs them concurrently via
/// rayon. Feature-gated and setup-heavy benchmarks are appended sequentially.
pub fn run_all_parallel(config: &Config) -> Vec<BenchResult> {
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(config, &mut rng);

    let draft_config = Config::draft();
    let mut draft_rng = Rng::new(99);
    let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

    let warmup = 1000;
    let iters = 50000;

    println!("\n📊 Running benchmarks in parallel ({iters} iterations, {warmup} warmup)...");

    #[derive(Clone, Copy)]
    enum BenchKind {
        Ar,
        DFlash,
        DdTree,
        Speculative,
        SpeculativeAr,
    }

    let core_kinds = [
        BenchKind::Ar,
        BenchKind::DFlash,
        BenchKind::DdTree,
        BenchKind::Speculative,
        BenchKind::SpeculativeAr,
    ];

    let mut results: Vec<BenchResult> = core_kinds
        .par_iter()
        .map(|&kind| match kind {
            BenchKind::Ar => bench_ar(&weights, config, warmup, iters),
            BenchKind::DFlash => bench_dflash(&draft_weights, &draft_config, warmup, iters),
            BenchKind::DdTree => bench_ddtree(&draft_weights, &draft_config, warmup, iters),
            BenchKind::Speculative => {
                bench_speculative(&draft_weights, &draft_config, warmup, iters)
            }
            BenchKind::SpeculativeAr => {
                bench_speculative_ar(&draft_weights, &draft_config, warmup, iters)
            }
        })
        .collect();

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

        let (no_rollback, with_rollback) = bench_snapshot_rollback(
            &draft_weights,
            &draft_config,
            &weights,
            config,
            warmup,
            iters,
        );
        results.push(no_rollback);
        results.push(with_rollback);

        let (uncond_br, cond_br) = bench_conditioned_vs_unconditioned(
            &draft_weights,
            &draft_config,
            &weights,
            config,
            warmup,
            iters,
        );
        results.push(uncond_br);
        results.push(cond_br);
    }

    let (nocompress_br, compress_br) =
        bench_prefill_compression(&draft_weights, &draft_config, warmup, iters);
    results.push(nocompress_br);
    results.push(compress_br);

    let (no_chain, chain) = bench_ddtree_chain_seed(&draft_weights, &draft_config, warmup, iters);
    results.push(no_chain);
    results.push(chain);

    let (flat_br, paged_br) = bench_paged_vs_flat_cache(config);
    results.push(flat_br);
    results.push(paged_br);

    // WasmPruner vs NoPruner DDTree build comparison
    #[cfg(feature = "wasm")]
    results.extend(bench_wasm_vs_no_pruner(
        &draft_weights,
        &draft_config,
        warmup,
        iters,
    ));

    results
}

/// Generate multiple text samples in parallel using rayon's `par_iter`.
///
/// Each sample gets its own `ForwardContext` + `MultiLayerKVCache` via
/// `map_init`, so there's no contention. Prints each sample's output
/// in order after all complete.
pub fn generate_batch(count: usize, max_tokens: usize) {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    println!("\n📝 Generating {count} samples ({max_tokens} tokens each) in parallel...");

    let seeds: Vec<u64> = (0..count).map(|i| 42 + i as u64).collect();

    let mut samples: Vec<(usize, Vec<usize>)> = seeds
        .par_iter()
        .enumerate()
        .map_init(
            || {
                (
                    ForwardContext::new(&config),
                    MultiLayerKVCache::new(&config),
                )
            },
            |(ctx, cache), (idx, &seed)| {
                let mut sample_rng = Rng::new(seed);
                let mut tokens = Vec::with_capacity(max_tokens);
                generate_into(
                    ctx,
                    cache,
                    &weights,
                    &config,
                    &mut sample_rng,
                    max_tokens,
                    &mut tokens,
                );
                (idx, tokens)
            },
        )
        .collect();

    samples.sort_by_key(|(idx, _)| *idx);
    for (idx, tokens) in &samples {
        let text = tokens_to_string(tokens);
        println!("  Sample {}: \"{text}\"", idx + 1);
    }
}
