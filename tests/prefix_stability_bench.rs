//! Prefix stability benchmark (Plan 029, Dynamo Lesson 1).
//!
//! Measures impact of stable vs unstable prefix on speculative pipeline.
//! NVIDIA Dynamo found ~5× TTFT penalty from varying prefix at position zero.
//!
//! Run: cargo test --features ppot prefix_stability_bench -- --nocapture

#[cfg(feature = "ppot")]
#[test]
fn prefix_stability_bench() {
    use microgpt_rs::speculative::{SpeculativeContext, dflash_predict_with};
    use microgpt_rs::transformer::TransformerWeights;
    use microgpt_rs::types::{Config, Rng};
    use std::time::Instant;

    /// Stable prefix: fixed system prompt tokens prepended every step.
    const STABLE_PREFIX: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

    /// Measure time for forward pass with a given prefix pattern.
    fn measure_forward_with_prefix(
        config: &Config,
        weights: &TransformerWeights,
        prefix: &[usize],
        continuation_tokens: usize,
        warmup: usize,
        iters: usize,
    ) -> (f64, f64) {
        let vocab_size = config.vocab_size;
        let mut sctx = SpeculativeContext::new(config);

        // Warmup
        for _ in 0..warmup {
            sctx.reset();
            let mut pos = 0;
            for &tok in prefix {
                let _ = dflash_predict_with(&mut sctx, weights, config, tok, pos);
                pos += 1;
            }
            for i in 0..continuation_tokens {
                let tok = i % vocab_size;
                let _ = dflash_predict_with(&mut sctx, weights, config, tok, pos);
                pos += 1;
            }
            std::hint::black_box(&sctx);
        }

        // Measured iterations
        let mut total_us = 0.0f64;
        let mut total_tokens = 0usize;

        for _ in 0..iters {
            sctx.reset();
            let mut pos = 0;

            let start = Instant::now();
            for &tok in prefix {
                let steps = dflash_predict_with(&mut sctx, weights, config, tok, pos);
                total_tokens += steps;
                pos += 1;
            }
            for i in 0..continuation_tokens {
                let tok = i % vocab_size;
                let steps = dflash_predict_with(&mut sctx, weights, config, tok, pos);
                total_tokens += steps;
                pos += 1;
            }
            total_us += start.elapsed().as_micros() as f64;
            std::hint::black_box(&sctx);
        }

        let avg_us_per_step = total_us / iters as f64;
        let tokens_per_sec = total_tokens as f64 / (total_us / 1_000_000.0);
        (avg_us_per_step, tokens_per_sec)
    }

    let config = Config::draft();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let vocab_size = config.vocab_size;
    let warmup = 50;
    let iters = 500;
    let continuation_tokens = 16;

    println!("\n🔬 Prefix Stability Benchmark (Plan 029, Dynamo Lesson 1)");
    println!("{}", "═".repeat(70));
    println!(
        "  Config: {} layers, {} heads, {} embedding dim, vocab {}",
        config.n_layer, config.n_head, config.n_embd, vocab_size
    );
    println!(
        "  Prefix: {} tokens, Continuation: {} tokens",
        STABLE_PREFIX.len(),
        continuation_tokens
    );
    println!("  Warmup: {warmup}, Iterations: {iters}");
    println!("{}", "─".repeat(70));

    // 1. Stable prefix — same prefix every iteration
    let (stable_us, stable_tps) = measure_forward_with_prefix(
        &config,
        &weights,
        &STABLE_PREFIX,
        continuation_tokens,
        warmup,
        iters,
    );

    println!("  Stable prefix:   {stable_us:8.1} μs/run, {stable_tps:8.0} tok/s");

    // 2. Unstable prefix — different prefix every iteration
    //    Simulates per-request metadata at position zero (e.g., billing header, session ID)
    let mut unstable_us_total = 0.0f64;
    let mut unstable_tokens_total = 0usize;

    // Warmup
    for seed in 0..warmup {
        let mut sctx = SpeculativeContext::new(&config);
        sctx.reset();
        let mut pos = 0;
        // Varying prefix: seed-dependent tokens at position zero
        let unstable_prefix: Vec<usize> = (0..STABLE_PREFIX.len())
            .map(|i| (seed + i) % vocab_size)
            .collect();
        for &tok in &unstable_prefix {
            let _ = dflash_predict_with(&mut sctx, &weights, &config, tok, pos);
            pos += 1;
        }
        for i in 0..continuation_tokens {
            let tok = i % vocab_size;
            let _ = dflash_predict_with(&mut sctx, &weights, &config, tok, pos);
            pos += 1;
        }
        std::hint::black_box(&sctx);
    }

    // Measured
    for seed in 0..iters {
        let mut sctx = SpeculativeContext::new(&config);
        sctx.reset();
        let mut pos = 0;
        let unstable_prefix: Vec<usize> = (0..STABLE_PREFIX.len())
            .map(|i| (seed + i) % vocab_size)
            .collect();

        let start = Instant::now();
        for &tok in &unstable_prefix {
            let steps = dflash_predict_with(&mut sctx, &weights, &config, tok, pos);
            unstable_tokens_total += steps;
            pos += 1;
        }
        for i in 0..continuation_tokens {
            let tok = i % vocab_size;
            let steps = dflash_predict_with(&mut sctx, &weights, &config, tok, pos);
            unstable_tokens_total += steps;
            pos += 1;
        }
        unstable_us_total += start.elapsed().as_micros() as f64;
        std::hint::black_box(&sctx);
    }

    let unstable_us = unstable_us_total / iters as f64;
    let unstable_tps = unstable_tokens_total as f64 / (unstable_us_total / 1_000_000.0);

    println!("  Unstable prefix: {unstable_us:8.1} μs/run, {unstable_tps:8.0} tok/s");

    // 3. Compare
    let ratio = if unstable_us > 0.0 {
        stable_us / unstable_us
    } else {
        1.0
    };
    println!("{}", "─".repeat(70));
    println!("  Ratio (stable/unstable): {ratio:.3}×");
    println!("  Note: On CPU, prefix stability has minimal impact (no HW KV cache).");
    println!("  On GPU with PagedKVCache prefix caching, unstable prefix causes ~5× TTFT penalty.");
    println!("        (NVIDIA Dynamo: 911ms vs 168ms on 52K prompt, B200)");

    // Sanity check: both should produce reasonable throughput
    assert!(stable_tps > 0.0, "stable throughput should be positive");
    assert!(unstable_tps > 0.0, "unstable throughput should be positive");

    // Suppress unused variable warning for rng (used by TransformerWeights::new)
    let _ = rng;
}
