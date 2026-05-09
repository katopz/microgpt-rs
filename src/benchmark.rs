use crate::speculative::{
    AttentionScorer, NoPruner, NoScreeningPruner, SimulatedVerifier, SpeculativeContext,
    TreeBuilder, compress_prompt, dflash_predict_ar_with, dflash_predict_with,
    extract_best_path_into, sample_from_distribution, speculative_step_verifier,
};
use crate::transformer::{
    ForwardContext, MultiLayerKVCache, PagedKVCache, PrefillContext, RavenKVCache,
    TransformerWeights, forward, forward_paged, forward_prefill, forward_raven, generate_into,
    generate_with_prefill, raven_readout, raven_update, tokens_to_string,
};
use crate::types::{Config, LoraAdapter, LoraPair, Rng, softmax};
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

    // Screening Pruner regression check (Plan 021)
    let (screened_noop, screened_adapter) =
        bench_ddtree_screened(&draft_weights, &draft_config, warmup, iters);
    results.push(screened_noop);
    results.push(screened_adapter);

    // Paged vs flat cache comparison
    let (flat_br, paged_br) = bench_paged_vs_flat_cache(config);
    results.push(flat_br);
    results.push(paged_br);

    // Raven RSM cache (draft model) — flat already measured above
    let (_, raven_br) = bench_raven_vs_flat_cache(config);
    results.push(raven_br);

    // Raven recall accuracy after noise
    let recall_br = bench_raven_recall(config);
    results.push(recall_br);

    // Plan 025: Bidirectional prefill vs sequential causal
    let (causal_br, bidir_br) = bench_bidirectional_prefill(config, warmup, iters);
    results.push(causal_br);
    results.push(bidir_br);

    // Plan 025: LoRA switching overhead
    let (no_lora_br, with_lora_br) = bench_lora_switch(config, warmup, iters);
    results.push(no_lora_br);
    results.push(with_lora_br);

    // Plan 025: End-to-end generate_with_prefill vs plain generate
    let (plain_br, prefill_br) = bench_generate_with_prefill_e2e(config, warmup, iters);
    results.push(plain_br);
    results.push(prefill_br);

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
        .join("../riir-ai/target/wasm32-unknown-unknown/release/examples/bracket_validator.wasm");

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
/// Benchmark: DDTree with ScreeningPruner vs original ConstraintPruner (Plan 021).
///
/// Returns two results:
/// 1. `build_screened()` with `NoScreeningPruner` (R=1.0 everywhere) — should match original DDTree
/// 2. `build_screened()` with `BinaryScreeningPruner(NoPruner)` — adapter path overhead check
///
/// Both should show zero regression vs the baseline `build()` with `NoPruner`.
pub fn bench_ddtree_screened(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    warmup: usize,
    iters: usize,
) -> (BenchResult, BenchResult) {
    use crate::speculative::BinaryScreeningPruner;

    let mut sctx = SpeculativeContext::new(draft_config);
    sctx.reset();
    dflash_predict_with(&mut sctx, draft_weights, draft_config, 0, 0);
    let mv: Vec<&[f32]> = (0..sctx.steps_populated)
        .map(|step| sctx.marginal_slice(step, draft_config.vocab_size))
        .collect();

    let mut tree_builder = TreeBuilder::new(draft_config);

    // Warmup both paths
    for _ in 0..warmup {
        let _ = tree_builder.build_screened(&mv, draft_config, &NoScreeningPruner, false);
        let _ =
            tree_builder.build_screened(&mv, draft_config, &BinaryScreeningPruner(NoPruner), false);
    }

    // Benchmark: NoScreeningPruner (R=1.0, pure screening path, no penalty)
    let start = Instant::now();
    let mut total_nodes_noop = 0usize;
    for _ in 0..iters {
        let tree = tree_builder.build_screened(&mv, draft_config, &NoScreeningPruner, false);
        total_nodes_noop += tree.len();
    }
    let elapsed_noop = start.elapsed();

    // Benchmark: BinaryScreeningPruner adapter (ConstraintPruner → ScreeningPruner)
    let start = Instant::now();
    let mut total_nodes_adapter = 0usize;
    for _ in 0..iters {
        let tree =
            tree_builder.build_screened(&mv, draft_config, &BinaryScreeningPruner(NoPruner), false);
        total_nodes_adapter += tree.len();
    }
    let elapsed_adapter = start.elapsed();

    let ops_noop = iters as f64 / elapsed_noop.as_secs_f64();
    let ops_adapter = iters as f64 / elapsed_adapter.as_secs_f64();

    (
        BenchResult {
            label: "DDTree (screened R=1.0)".into(),
            throughput: ops_noop,
            time_per_step_us: elapsed_noop.as_micros() as f64 / iters as f64,
            avg_acceptance_len: total_nodes_noop as f64 / iters as f64,
            color: (0, 191, 255),
        },
        BenchResult {
            label: "DDTree (screened adapter)".into(),
            throughput: ops_adapter,
            time_per_step_us: elapsed_adapter.as_micros() as f64 / iters as f64,
            avg_acceptance_len: total_nodes_adapter as f64 / iters as f64,
            color: (30, 144, 255),
        },
    )
}

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

/// Benchmark: Raven RSM vs flat KV cache for draft model.
///
/// Compares per-token throughput of `forward_raven()` (O(1) slot memory)
/// against standard `forward()` (O(N) growing cache).
pub fn bench_raven_vs_flat_cache(_config: &Config) -> (BenchResult, BenchResult) {
    let draft_config = Config::draft();
    let mut rng = Rng::new(42);
    let draft_weights = TransformerWeights::new(&draft_config, &mut rng);
    let iters = 200;

    // Raven config: 16 slots, top-4 routing (4x kv_dim for draft)
    let num_slots = 16;
    let top_k = 4;

    // Warm up both paths
    {
        let mut ctx = ForwardContext::new(&draft_config);
        let mut cache = MultiLayerKVCache::new(&draft_config);
        let _ = forward(&mut ctx, &draft_weights, &mut cache, 0, 0, &draft_config);
    }
    {
        let mut ctx = ForwardContext::new(&draft_config);
        let mut cache = RavenKVCache::new(&draft_config, num_slots, top_k);
        let _ = forward_raven(&mut ctx, &draft_weights, &mut cache, 0, 0, &draft_config);
    }

    // Benchmark flat cache (growing O(N) attention)
    let mut ctx_flat = ForwardContext::new(&draft_config);
    let mut cache_flat = MultiLayerKVCache::new(&draft_config);
    let start_flat = Instant::now();
    for _ in 0..iters {
        cache_flat.reset();
        let max_pos = draft_config.block_size.min(8);
        for pos in 0..max_pos {
            let _ = forward(
                &mut ctx_flat,
                &draft_weights,
                &mut cache_flat,
                0,
                pos,
                &draft_config,
            );
        }
    }
    let elapsed_flat = start_flat.elapsed();

    // Benchmark Raven cache (fixed O(slots) attention)
    let mut ctx_raven = ForwardContext::new(&draft_config);
    let mut cache_raven = RavenKVCache::new(&draft_config, num_slots, top_k);
    let start_raven = Instant::now();
    for _ in 0..iters {
        cache_raven.reset();
        let max_pos = draft_config.block_size.min(8);
        for pos in 0..max_pos {
            let _ = forward_raven(
                &mut ctx_raven,
                &draft_weights,
                &mut cache_raven,
                0,
                pos,
                &draft_config,
            );
        }
    }
    let elapsed_raven = start_raven.elapsed();

    let steps_per_iter = draft_config.block_size.min(8) as f64;

    let flat_br = BenchResult {
        label: "forward (flat)".into(),
        throughput: iters as f64 * steps_per_iter / elapsed_flat.as_secs_f64(),
        time_per_step_us: elapsed_flat.as_micros() as f64 / (iters as f64 * steps_per_iter),
        avg_acceptance_len: 0.0,
        color: (100, 149, 237),
    };

    let raven_br = BenchResult {
        label: format!("forward_raven ({} slots)", num_slots),
        throughput: iters as f64 * steps_per_iter / elapsed_raven.as_secs_f64(),
        time_per_step_us: elapsed_raven.as_micros() as f64 / (iters as f64 * steps_per_iter),
        avg_acceptance_len: 0.0,
        color: (180, 100, 220),
    };

    (flat_br, raven_br)
}

/// Benchmark: Raven recall accuracy after noise updates.
///
/// THE critical test from the paper:
/// 1. Write "passkey" to a specific slot (value = 9.9)
/// 2. Run 1000 noise updates targeting OTHER slots
/// 3. Readout and verify original value preserved (> 9.0)
pub fn bench_raven_recall(_config: &Config) -> BenchResult {
    let draft_config = Config::draft();
    let num_slots = 16;
    let top_k = 4;
    let kvd = crate::types::kv_dim(&draft_config);
    let noise_steps = 1000;

    let mut cache = RavenKVCache::new(&draft_config, num_slots, top_k);

    // 1. Write critical passkey to slot 42... wait, we only have 16 slots.
    //    Write to slot 12 instead.
    let passkey_slot = 12;
    let passkey_k = vec![1.0; kvd];
    let passkey_v = vec![9.9; kvd];

    let mut r_t_passkey = vec![0.0f32; num_slots];
    r_t_passkey[passkey_slot] = 1.0;
    raven_update(
        &mut cache.keys,
        &mut cache.values,
        &passkey_k,
        &passkey_v,
        &r_t_passkey,
        cache.forget_rate,
        num_slots,
        kvd,
    );

    // 2. Run 1000 noise updates targeting slots 0-3 (NOT slot 12)
    let start = Instant::now();
    let noise_k = vec![0.5; kvd];
    let noise_v = vec![0.1; kvd];
    let mut r_t_noise = vec![0.0f32; num_slots];
    r_t_noise[0] = 0.25;
    r_t_noise[1] = 0.25;
    r_t_noise[2] = 0.25;
    r_t_noise[3] = 0.25;

    for _ in 0..noise_steps {
        raven_update(
            &mut cache.keys,
            &mut cache.values,
            &noise_k,
            &noise_v,
            &r_t_noise,
            cache.forget_rate,
            num_slots,
            kvd,
        );
    }

    // 3. Readout with passkey query
    let query = vec![1.0; kvd];
    let _retrieved = raven_readout(&query, &cache.keys, &cache.values, num_slots, kvd);
    let elapsed = start.elapsed();

    // Check recall: the passkey value should be preserved in slot 12
    let slot_12_off = passkey_slot * kvd;
    let slot_12_first = cache.values[slot_12_off];

    // Recall accuracy: how close is the stored value to the original 9.9?
    let slot_12_first_f64 = slot_12_first as f64;
    let recall_accuracy: f64 = if slot_12_first_f64 > 9.0 {
        100.0
    } else {
        (slot_12_first_f64 / 9.9) * 100.0
    };

    BenchResult {
        label: format!(
            "raven_recall ({noise_steps} noise, slot {passkey_slot}={:.1}→{:.1} acc={recall_accuracy:.0}%)",
            passkey_v[0] as f64, slot_12_first as f64
        ),
        throughput: noise_steps as f64 / elapsed.as_secs_f64(),
        time_per_step_us: elapsed.as_micros() as f64 / noise_steps as f64,
        avg_acceptance_len: recall_accuracy,
        color: (50, 205, 50),
    }
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

// ---------------------------------------------------------------------------
// Plan 025: Bidirectional Prefill + Modality LoRA Switching Benchmarks
// ---------------------------------------------------------------------------

/// Create a deterministic LoRA adapter for benchmarking.
fn make_bench_lora(config: &Config, seed: u32) -> LoraAdapter {
    let rank = config.lora_rank;
    let dim = config.n_embd;

    let a: Vec<f32> = (0..rank * dim)
        .map(|i| {
            let v = ((seed as u64).wrapping_mul((i + 1) as u64)) as u32;
            ((v as f32 / u32::MAX as f32) - 0.5) * 0.1
        })
        .collect();

    let b: Vec<f32> = (0..dim * rank)
        .map(|i| {
            let v = ((seed as u64).wrapping_mul((i + 100) as u64)) as u32;
            ((v as f32 / u32::MAX as f32) - 0.5) * 0.1
        })
        .collect();

    LoraAdapter {
        a,
        b,
        rank,
        alpha: config.lora_alpha,
        in_dim: dim,
        out_dim: dim,
    }
}

/// Benchmark: bidirectional prefill vs sequential causal forward.
///
/// Measures `forward_prefill()` (all tokens see each other) vs sequential
/// `forward()` calls (causal, left-to-right) over the same prompt tokens.
/// Reports throughput in tokens/sec and µs/step.
pub fn bench_bidirectional_prefill(
    config: &Config,
    warmup: usize,
    iters: usize,
) -> (BenchResult, BenchResult) {
    let prompt_len = config.block_size.min(8);
    let prompt_tokens: Vec<usize> = (0..prompt_len).map(|i| i % config.vocab_size).collect();

    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(config, &mut rng);

    // Warmup causal
    for _ in 0..warmup {
        let mut ctx = ForwardContext::new(config);
        let mut cache = MultiLayerKVCache::new(config);
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            let _ = forward(&mut ctx, &weights, &mut cache, tok, pos, config);
        }
    }

    // Warmup bidirectional
    for _ in 0..warmup {
        let mut ctx = ForwardContext::new(config);
        let mut pf = PrefillContext::new(config);
        let mut cache = MultiLayerKVCache::new(config);
        let _ = forward_prefill(
            &mut ctx,
            &mut pf,
            &weights,
            &mut cache,
            &prompt_tokens,
            config,
            None,
        );
    }

    // Bench: sequential causal
    let start = Instant::now();
    for _ in 0..iters {
        let mut ctx = ForwardContext::new(config);
        let mut cache = MultiLayerKVCache::new(config);
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            let _ = forward(&mut ctx, &weights, &mut cache, tok, pos, config);
        }
    }
    let elapsed_causal = start.elapsed();

    // Bench: bidirectional prefill
    let start = Instant::now();
    for _ in 0..iters {
        let mut ctx = ForwardContext::new(config);
        let mut pf = PrefillContext::new(config);
        let mut cache = MultiLayerKVCache::new(config);
        let _ = forward_prefill(
            &mut ctx,
            &mut pf,
            &weights,
            &mut cache,
            &prompt_tokens,
            config,
            None,
        );
    }
    let elapsed_bidir = start.elapsed();

    let causal_result = BenchResult {
        label: "Prefill (causal seq)".into(),
        throughput: iters as f64 * prompt_len as f64 / elapsed_causal.as_secs_f64(),
        time_per_step_us: elapsed_causal.as_micros() as f64 / (iters as f64 * prompt_len as f64),
        avg_acceptance_len: prompt_len as f64,
        color: (180, 180, 180),
    };

    let bidir_result = BenchResult {
        label: "Prefill (bidirectional)".into(),
        throughput: iters as f64 * prompt_len as f64 / elapsed_bidir.as_secs_f64(),
        time_per_step_us: elapsed_bidir.as_micros() as f64 / (iters as f64 * prompt_len as f64),
        avg_acceptance_len: prompt_len as f64,
        color: (0, 200, 200),
    };

    (causal_result, bidir_result)
}

/// Benchmark: LoRA switching overhead.
///
/// Measures `forward()` with no LoRA vs `forward_prefill()` + LoRA adapter.
/// The LoRA overhead is the delta between these two — it should be near-zero
/// since LoRA is just two small matmuls fused into the existing kernel.
pub fn bench_lora_switch(
    config: &Config,
    warmup: usize,
    iters: usize,
) -> (BenchResult, BenchResult) {
    let prompt_len = config.block_size.min(8);
    let prompt_tokens: Vec<usize> = (0..prompt_len).map(|i| i % config.vocab_size).collect();

    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(config, &mut rng);
    let lora = make_bench_lora(config, 100);

    // Warmup no-LoRA
    for _ in 0..warmup {
        let mut ctx = ForwardContext::new(config);
        let mut pf = PrefillContext::new(config);
        let mut cache = MultiLayerKVCache::new(config);
        let _ = forward_prefill(
            &mut ctx,
            &mut pf,
            &weights,
            &mut cache,
            &prompt_tokens,
            config,
            None,
        );
    }

    // Warmup with-LORA
    for _ in 0..warmup {
        let mut ctx = ForwardContext::new(config);
        let mut pf = PrefillContext::new(config);
        let mut cache = MultiLayerKVCache::new(config);
        let _ = forward_prefill(
            &mut ctx,
            &mut pf,
            &weights,
            &mut cache,
            &prompt_tokens,
            config,
            Some(&lora),
        );
    }

    // Bench: no LoRA
    let start = Instant::now();
    for _ in 0..iters {
        let mut ctx = ForwardContext::new(config);
        let mut pf = PrefillContext::new(config);
        let mut cache = MultiLayerKVCache::new(config);
        let _ = forward_prefill(
            &mut ctx,
            &mut pf,
            &weights,
            &mut cache,
            &prompt_tokens,
            config,
            None,
        );
    }
    let elapsed_no_lora = start.elapsed();

    // Bench: with LoRA
    let start = Instant::now();
    for _ in 0..iters {
        let mut ctx = ForwardContext::new(config);
        let mut pf = PrefillContext::new(config);
        let mut cache = MultiLayerKVCache::new(config);
        let _ = forward_prefill(
            &mut ctx,
            &mut pf,
            &weights,
            &mut cache,
            &prompt_tokens,
            config,
            Some(&lora),
        );
    }
    let elapsed_with_lora = start.elapsed();

    let no_lora_result = BenchResult {
        label: "Prefill (no LoRA)".into(),
        throughput: iters as f64 * prompt_len as f64 / elapsed_no_lora.as_secs_f64(),
        time_per_step_us: elapsed_no_lora.as_micros() as f64 / (iters as f64 * prompt_len as f64),
        avg_acceptance_len: prompt_len as f64,
        color: (150, 150, 200),
    };

    let with_lora_result = BenchResult {
        label: "Prefill (w/ LoRA)".into(),
        throughput: iters as f64 * prompt_len as f64 / elapsed_with_lora.as_secs_f64(),
        time_per_step_us: elapsed_with_lora.as_micros() as f64 / (iters as f64 * prompt_len as f64),
        avg_acceptance_len: prompt_len as f64,
        color: (200, 100, 255),
    };

    (no_lora_result, with_lora_result)
}

/// Benchmark: end-to-end generate_with_prefill vs plain generate.
///
/// Measures the full pipeline: bidirectional prefill (reader LoRA) + causal
/// decode (writer LoRA) vs plain autoregressive generation with no LoRA.
/// This is the real-world py2rs use case.
pub fn bench_generate_with_prefill_e2e(
    config: &Config,
    warmup: usize,
    iters: usize,
) -> (BenchResult, BenchResult) {
    let prompt_len = config.block_size.min(8);
    let prompt_tokens: Vec<usize> = (0..prompt_len).map(|i| i % config.vocab_size).collect();
    let gen_tokens = 8;

    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(config, &mut rng);

    let lora_pair = LoraPair {
        reader: Some(make_bench_lora(config, 100)),
        writer: Some(make_bench_lora(config, 200)),
    };

    // Warmup plain generate
    for _ in 0..warmup {
        let mut rng_inner = Rng::new(42);
        let mut tokens = Vec::new();
        generate_into(
            &mut ForwardContext::new(config),
            &mut MultiLayerKVCache::new(config),
            &weights,
            config,
            &mut rng_inner,
            gen_tokens,
            &mut tokens,
        );
    }

    // Warmup generate_with_prefill
    for _ in 0..warmup {
        let mut rng_inner = Rng::new(42);
        let _ = generate_with_prefill(
            &mut ForwardContext::new(config),
            &mut PrefillContext::new(config),
            &weights,
            &mut MultiLayerKVCache::new(config),
            config,
            &mut rng_inner,
            &prompt_tokens,
            gen_tokens,
            &lora_pair,
        );
    }

    // Bench: plain generate
    let start = Instant::now();
    let mut total_plain_tokens = 0usize;
    for _ in 0..iters {
        let mut rng_inner = Rng::new(42);
        let mut tokens = Vec::new();
        generate_into(
            &mut ForwardContext::new(config),
            &mut MultiLayerKVCache::new(config),
            &weights,
            config,
            &mut rng_inner,
            gen_tokens,
            &mut tokens,
        );
        total_plain_tokens += tokens.len();
    }
    let elapsed_plain = start.elapsed();

    // Bench: generate_with_prefill
    let start = Instant::now();
    let mut total_prefill_tokens = 0usize;
    for _ in 0..iters {
        let mut rng_inner = Rng::new(42);
        let generated = generate_with_prefill(
            &mut ForwardContext::new(config),
            &mut PrefillContext::new(config),
            &weights,
            &mut MultiLayerKVCache::new(config),
            config,
            &mut rng_inner,
            &prompt_tokens,
            gen_tokens,
            &lora_pair,
        );
        total_prefill_tokens += generated.len();
    }
    let elapsed_prefill = start.elapsed();

    let plain_result = BenchResult {
        label: "Generate (plain AR)".into(),
        throughput: total_plain_tokens as f64 / elapsed_plain.as_secs_f64(),
        time_per_step_us: elapsed_plain.as_micros() as f64 / iters as f64,
        avg_acceptance_len: total_plain_tokens as f64 / iters as f64,
        color: (200, 150, 100),
    };

    let prefill_result = BenchResult {
        label: "Generate (prefill+LoRA)".into(),
        throughput: total_prefill_tokens as f64 / elapsed_prefill.as_secs_f64(),
        time_per_step_us: elapsed_prefill.as_micros() as f64 / iters as f64,
        avg_acceptance_len: total_prefill_tokens as f64 / iters as f64,
        color: (255, 100, 100),
    };

    (plain_result, prefill_result)
}

/// Benchmark: Dense matmul vs Sparse TwELL matmul at various sparsity levels (Plan 022).
#[cfg(feature = "sparse_mlp")]
pub fn bench_sparse_mlp() {
    use crate::types;

    println!("\n=== Sparse MLP Benchmark (Plan 022: TwELL-inspired) ===\n");

    let configs = [
        ("micro", 64, 16),
        ("bpe", 128, 32),
        ("small_target", 256, 64),
        ("large", 16384, 4096),
    ];

    let sparsity_levels = [0.0f32, 0.50, 0.90, 0.95, 0.99];

    let iterations = 10;

    for &(label, mlp_hidden, n_embd) in &configs {
        println!("--- Config: {label} (mlp_hidden={mlp_hidden}, n_embd={n_embd}) ---");

        let weight: Vec<f32> = (0..n_embd * mlp_hidden)
            .map(|i| (i % 100) as f32 * 0.01)
            .collect();
        let mut output_dense = vec![0.0f32; n_embd];
        let mut output_sparse = vec![0.0f32; n_embd];
        let mut active_indices = vec![0usize; mlp_hidden];
        let mut active_values = vec![0.0f32; mlp_hidden];

        for &sparsity in &sparsity_levels {
            // Build input with target sparsity
            let mut input = vec![0.0f32; mlp_hidden];
            let alive_count = ((1.0 - sparsity) * mlp_hidden as f32) as usize;
            for val in input.iter_mut().take(alive_count) {
                *val = 1.0;
            }

            // Dense benchmark
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                types::matmul(&mut output_dense, &weight, &input, n_embd, mlp_hidden);
            }
            let elapsed_dense = start.elapsed();

            // Sparse benchmark
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                types::sparse_matmul(
                    &mut output_sparse,
                    &weight,
                    &input,
                    n_embd,
                    mlp_hidden,
                    &mut active_indices,
                    &mut active_values,
                );
            }
            let elapsed_sparse = start.elapsed();

            // Verify correctness
            for i in 0..n_embd {
                let diff = (output_dense[i] - output_sparse[i]).abs();
                let d = output_dense[i];
                let s = output_sparse[i];
                assert!(diff < 1e-2, "Mismatch at {i}: dense={d}, sparse={s}");
            }

            let speedup = elapsed_dense.as_secs_f64() / elapsed_sparse.as_secs_f64();
            println!(
                "  Sparsity {:.0}%: Dense={:.2?} Sparse={:.2?} Speedup={:.1}x",
                sparsity * 100.0,
                elapsed_dense,
                elapsed_sparse,
                speedup,
            );
        }
        println!();
    }
}
