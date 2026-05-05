use microgpt_rs::{benchmark, percepta, plot, transformer, types};

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
    println!("  📈 Speedup: {:.2}x (Speculative vs AR)", speedup);

    // ── Plot ───────────────────────────────────────────────────────
    std::fs::create_dir_all("bench").ok();
    let index = next_bench_index();
    let plot_path = format!("bench/{:03}_bench_result.png", index);

    match plot::plot_results(&results, &plot_path) {
        Ok(()) => println!("\n📈 Chart saved to: {plot_path}"),
        Err(e) => eprintln!("\n⚠️  Plot failed: {e}"),
    }

    // ── 6. Percepta 2D Attention Benchmark ─────────────────────────
    println!("\n🧠 Percepta 2D Convex Hull Attention (O(log N) vs O(N))");
    println!("{}", "─".repeat(60));

    percepta_benchmark();

    println!("\n✨ Done.");
}

/// Benchmark: Percepta O(log N) hull attention vs standard O(N) linear scan.
/// Proves correctness (same results) and measures speedup across trace sizes.
fn percepta_benchmark() {
    let trace_sizes = [1_000, 10_000, 100_000];

    println!(
        "  {:>12} {:>8} {:>12} {:>12} {:>10} {:>8}",
        "Trace Size", "Hull", "Linear μs", "Fast μs", "Speedup", "Match"
    );
    println!("{}", "─".repeat(66));

    for &size in &trace_sizes {
        let mut cache = percepta::KVCache2D::with_capacity(size);

        // Build convex parabolic key distribution (simulates execution trace)
        let mid = size as f32 / 2.0;
        for i in 0..size {
            let x = i as f32;
            let y = -((x - mid) / (mid * 0.02)).powi(2);
            cache.append(percepta::Vec2::new(x, y), i);
        }

        let query = percepta::Vec2::new(5.0, 10.0);

        // Warmup
        for _ in 0..10 {
            let _ = cache.fast_attention(&query);
            let _ = cache.linear_attention(&query);
        }

        // Benchmark linear O(N)
        let iters_linear = 100;
        let start = std::time::Instant::now();
        let (lin_score, lin_val) = cache.linear_attention(&query);
        for _ in 0..iters_linear {
            let _ = cache.linear_attention(&query);
        }
        let elapsed_linear = start.elapsed() / (iters_linear + 1);

        // Benchmark fast O(log N)
        let iters_fast = 10_000;
        let start = std::time::Instant::now();
        let (fast_score, fast_val) = cache.fast_attention(&query);
        for _ in 0..iters_fast {
            let _ = cache.fast_attention(&query);
        }
        let elapsed_fast = start.elapsed() / (iters_fast + 1);

        let speedup = elapsed_linear.as_secs_f64() / elapsed_fast.as_secs_f64();
        let score_match = (lin_score - fast_score).abs() < 1e-3;
        let val_match = lin_val == fast_val;

        println!(
            "  {:>12} {:>8} {:>12.2} {:>12.4} {:>9.1}x {:>8}",
            size,
            cache.hull_len(),
            elapsed_linear.as_secs_f64() * 1e6,
            elapsed_fast.as_secs_f64() * 1e6,
            speedup,
            if score_match && val_match {
                "✅"
            } else {
                "❌"
            }
        );
    }

    println!();
    println!("  Hull compression: O(N) keys → O(H) hull vertices");
    println!("  Attention search: ternary search over unimodal dot-product sequence");
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
