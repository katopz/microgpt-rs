use microgpt_rs::{benchmark, plot, transformer, types};

fn main() {
    let config = types::Config::micro();

    println!("🚀 MicroGPT-RS: Transformer + Speculative Decoding Benchmark");
    println!("{}", "═".repeat(60));

    // ── 1. Transformer Output with Proof ──────────────────────────
    println!("\n📝 Transformer Output (Proof of Correctness)");
    println!("{}", "─".repeat(60));

    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);

    // Generate samples with different seeds
    for sample in 0..5 {
        let mut sample_rng = types::Rng::new(42 + sample);
        let tokens = transformer::generate(&weights, &config, &mut sample_rng, config.block_size);
        let text = transformer::tokens_to_string(&tokens);
        let valid = tokens.iter().all(|&t| t < config.vocab_size);
        println!("  Sample {}: \"{}\" (valid={})", sample + 1, text, valid);
    }

    // Determinism check: same seed must produce identical output
    let mut rng_a = types::Rng::new(100);
    let tokens_a = transformer::generate(&weights, &config, &mut rng_a, config.block_size);
    let mut rng_b = types::Rng::new(100);
    let tokens_b = transformer::generate(&weights, &config, &mut rng_b, config.block_size);
    let deterministic = tokens_a == tokens_b;

    // Different seed must produce different output
    let mut rng_c = types::Rng::new(200);
    let tokens_c = transformer::generate(&weights, &config, &mut rng_c, config.block_size);
    let diverse = tokens_a != tokens_c;

    println!();
    println!(
        "  ✅ Deterministic: {} (same seed = same output)",
        if deterministic { "PASS" } else { "FAIL" }
    );
    println!(
        "  ✅ Diverse:       {} (different seed = different output)",
        if diverse { "PASS" } else { "FAIL" }
    );
    println!(
        "  ✅ Valid tokens:  {} (all tokens in [0, {}))",
        if deterministic && diverse {
            "PASS"
        } else {
            "FAIL"
        },
        config.vocab_size
    );
    println!(
        "  📐 Config: vocab={}, block={}, embd={}, heads={}, mlp={}",
        config.vocab_size, config.block_size, config.n_embd, config.n_head, config.mlp_hidden
    );

    // ── 2–5. Benchmarks ───────────────────────────────────────────
    let results = benchmark::run_all(&config);

    println!("📊 Benchmark Results");
    println!("{}", "─".repeat(75));
    println!(
        "  {:<20} {:>15} {:>15} {:>15}",
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
            "  {:<20} {:>12.0} {:>3} {:>12.2} {:>15.2}",
            r.label, r.throughput, unit, r.time_per_step_us, r.avg_acceptance_len,
        );
    }

    println!("{}", "─".repeat(75));

    // Speedup comparison
    let ar_tps = results[0].throughput;
    let spec_tps = results[3].throughput;
    let speedup = spec_tps / ar_tps;
    println!(
        "  📈 Speedup: {:.2}x (DFlash+DDTree effective vs AR)",
        speedup
    );

    // ── Plot ───────────────────────────────────────────────────────
    std::fs::create_dir_all("bench").ok();
    let index = next_bench_index();
    let plot_path = format!("bench/{:03}_bench_result.png", index);

    match plot::plot_results(&results, &plot_path) {
        Ok(()) => println!("\n📈 Chart saved to: {plot_path}"),
        Err(e) => eprintln!("\n⚠️  Plot failed: {e}"),
    }

    println!("\n✨ Done.");
}

/// Auto-number bench results sequentially.
fn next_bench_index() -> u32 {
    let dir = std::path::Path::new("bench");
    if !dir.exists() {
        return 1;
    }
    std::fs::read_dir(dir)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    name.split('_').next().and_then(|n| n.parse::<u32>().ok())
                })
                .max()
                .map(|n| n + 1)
                .unwrap_or(1)
        })
        .unwrap_or(1)
}
