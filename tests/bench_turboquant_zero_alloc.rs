//! TurboQuant Zero-Allocation Hot Path Benchmark (Plan 051).
//!
//! Compares the allocating vs zero-alloc paths for:
//! - store_key / store_value (both now zero-alloc internally)
//! - dequantize_key (allocating Vec return) vs dequantize_key_into (zero-alloc scratch buffers)
//! - dequantize_value (allocating) vs dequantize_value_into (zero-alloc)
//! - Full forward_turboquant vs baseline forward
//!
//! Run: cargo test bench_turboquant_zero_alloc -- --nocapture

use std::hint::black_box;
use std::time::Instant;

use microgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward, forward_turboquant,
};
use microgpt_rs::turboquant::TurboQuantKVCache;
use microgpt_rs::types::{Config, Rng, kv_dim};

/// Generate synthetic KV vector for position `pos`.
fn synthetic_kv(kv_dim: usize, pos: usize) -> Vec<f32> {
    (0..kv_dim)
        .map(|i| ((i + pos * 7) as f32 * 0.1).sin() + ((i + pos * 3) as f32 * 0.07).cos())
        .collect()
}

#[test]
fn bench_turboquant_zero_alloc_store_dequant() {
    let config = Config::micro();
    let kv_dim = kv_dim(&config);
    let n_positions = config.block_size;

    let warmup = 100;
    let iters = 10_000;

    println!("\n🧪 TurboQuant Zero-Alloc Benchmark (Plan 051)");
    println!("   kv_dim={kv_dim}, n_positions={n_positions}, warmup={warmup}, iters={iters}");
    println!("{}", "═".repeat(70));

    // Synthetic KV data (shared across all benchmarks)
    let keys: Vec<Vec<f32>> = (0..n_positions).map(|p| synthetic_kv(kv_dim, p)).collect();
    let vals: Vec<Vec<f32>> = (0..n_positions)
        .map(|p| synthetic_kv(kv_dim, p + 100))
        .collect();

    // ── Component 1: store_key (zero-alloc since Plan 051) ──

    let mut cache = TurboQuantKVCache::new(&config, 3, 3);
    for _ in 0..warmup {
        cache.reset();
        for (pos, key) in keys.iter().enumerate() {
            cache.store_key(0, pos, key);
        }
    }
    let start = Instant::now();
    for _ in 0..iters {
        cache.reset();
        for (pos, key) in keys.iter().enumerate() {
            cache.store_key(0, pos, key);
        }
    }
    let store_key_time = start.elapsed();
    let store_key_us = store_key_time.as_nanos() as f64 / iters as f64 / 1000.0;

    // ── Component 2: store_value (zero-alloc since Plan 051) ──

    let mut cache = TurboQuantKVCache::new(&config, 3, 3);
    for _ in 0..warmup {
        cache.reset();
        for (pos, val) in vals.iter().enumerate() {
            cache.store_value(0, pos, val);
        }
    }
    let start = Instant::now();
    for _ in 0..iters {
        cache.reset();
        for (pos, val) in vals.iter().enumerate() {
            cache.store_value(0, pos, val);
        }
    }
    let store_value_time = start.elapsed();
    let store_value_us = store_value_time.as_nanos() as f64 / iters as f64 / 1000.0;

    // ── Component 3a: dequantize_key (ALLOCATING path — backward compat) ──

    let mut cache = TurboQuantKVCache::new(&config, 3, 3);
    cache.reset();
    for (pos, key) in keys.iter().enumerate() {
        cache.store_key(0, pos, key);
    }
    // Warmup
    for _ in 0..warmup {
        for pos in 0..n_positions {
            black_box(cache.dequantize_key(0, pos));
        }
    }
    let start = Instant::now();
    for _ in 0..iters {
        for pos in 0..n_positions {
            black_box(cache.dequantize_key(0, pos));
        }
    }
    let dequant_key_alloc_time = start.elapsed();
    let dequant_key_alloc_us = dequant_key_alloc_time.as_nanos() as f64 / iters as f64 / 1000.0;

    // ── Component 3b: dequantize_key_into (ZERO-ALLOC path — Plan 051) ──

    let mut cache = TurboQuantKVCache::new(&config, 3, 3);
    cache.reset();
    for (pos, key) in keys.iter().enumerate() {
        cache.store_key(0, pos, key);
    }
    let mut buf = vec![0.0f32; kv_dim];
    // Warmup
    for _ in 0..warmup {
        for pos in 0..n_positions {
            cache.dequantize_key_into(0, pos, &mut buf);
            black_box(&buf);
        }
    }
    let start = Instant::now();
    for _ in 0..iters {
        for pos in 0..n_positions {
            cache.dequantize_key_into(0, pos, &mut buf);
            black_box(&buf);
        }
    }
    let dequant_key_zero_time = start.elapsed();
    let dequant_key_zero_us = dequant_key_zero_time.as_nanos() as f64 / iters as f64 / 1000.0;

    // ── Component 4a: dequantize_value (ALLOCATING path) ──

    let mut cache = TurboQuantKVCache::new(&config, 3, 3);
    cache.reset();
    for (pos, val) in vals.iter().enumerate() {
        cache.store_value(0, pos, val);
    }
    for _ in 0..warmup {
        for pos in 0..n_positions {
            black_box(cache.dequantize_value(0, pos));
        }
    }
    let start = Instant::now();
    for _ in 0..iters {
        for pos in 0..n_positions {
            black_box(cache.dequantize_value(0, pos));
        }
    }
    let dequant_val_alloc_time = start.elapsed();
    let dequant_val_alloc_us = dequant_val_alloc_time.as_nanos() as f64 / iters as f64 / 1000.0;

    // ── Component 4b: dequantize_value_into (ZERO-ALLOC path — Plan 051) ──

    let mut cache = TurboQuantKVCache::new(&config, 3, 3);
    cache.reset();
    for (pos, val) in vals.iter().enumerate() {
        cache.store_value(0, pos, val);
    }
    let mut buf = vec![0.0f32; kv_dim];
    for _ in 0..warmup {
        for pos in 0..n_positions {
            cache.dequantize_value_into(0, pos, &mut buf);
            black_box(&buf);
        }
    }
    let start = Instant::now();
    for _ in 0..iters {
        for pos in 0..n_positions {
            cache.dequantize_value_into(0, pos, &mut buf);
            black_box(&buf);
        }
    }
    let dequant_val_zero_time = start.elapsed();
    let dequant_val_zero_us = dequant_val_zero_time.as_nanos() as f64 / iters as f64 / 1000.0;

    // ── Component 5: Full store + dequantize cycle (zero-alloc end-to-end) ──

    let mut cache = TurboQuantKVCache::new(&config, 3, 3);
    let mut key_buf = vec![0.0f32; kv_dim];
    let mut val_buf = vec![0.0f32; kv_dim];

    for _ in 0..warmup {
        cache.reset();
        for pos in 0..n_positions {
            cache.store_key(0, pos, &keys[pos]);
            cache.store_value(0, pos, &vals[pos]);
        }
        for pos in 0..n_positions {
            cache.dequantize_key_into(0, pos, &mut key_buf);
            cache.dequantize_value_into(0, pos, &mut val_buf);
            black_box(&key_buf);
            black_box(&val_buf);
        }
    }
    let start = Instant::now();
    for _ in 0..iters {
        cache.reset();
        for pos in 0..n_positions {
            cache.store_key(0, pos, &keys[pos]);
            cache.store_value(0, pos, &vals[pos]);
        }
        for pos in 0..n_positions {
            cache.dequantize_key_into(0, pos, &mut key_buf);
            cache.dequantize_value_into(0, pos, &mut val_buf);
            black_box(&key_buf);
            black_box(&val_buf);
        }
    }
    let full_zero_time = start.elapsed();
    let full_zero_us = full_zero_time.as_nanos() as f64 / iters as f64 / 1000.0;

    // ── Component 6: Full store + dequantize cycle (allocating path, for comparison) ──

    let mut cache = TurboQuantKVCache::new(&config, 3, 3);

    for _ in 0..warmup {
        cache.reset();
        for pos in 0..n_positions {
            cache.store_key(0, pos, &keys[pos]);
            cache.store_value(0, pos, &vals[pos]);
        }
        for pos in 0..n_positions {
            black_box(cache.dequantize_key(0, pos));
            black_box(cache.dequantize_value(0, pos));
        }
    }
    let start = Instant::now();
    for _ in 0..iters {
        cache.reset();
        for pos in 0..n_positions {
            cache.store_key(0, pos, &keys[pos]);
            cache.store_value(0, pos, &vals[pos]);
        }
        for pos in 0..n_positions {
            black_box(cache.dequantize_key(0, pos));
            black_box(cache.dequantize_value(0, pos));
        }
    }
    let full_alloc_time = start.elapsed();
    let full_alloc_us = full_alloc_time.as_nanos() as f64 / iters as f64 / 1000.0;

    // ── Results ──

    let key_delta_pct = (dequant_key_alloc_us - dequant_key_zero_us) / dequant_key_alloc_us * 100.0;
    let val_delta_pct = (dequant_val_alloc_us - dequant_val_zero_us) / dequant_val_alloc_us * 100.0;
    let full_delta_pct = (full_alloc_us - full_zero_us) / full_alloc_us * 100.0;

    println!();
    println!("┌────────────────────────────────┬────────────┬────────────┬──────────┐");
    println!("│ Operation                       │ Alloc (μs) │ Zero (μs)  │    Δ %%  │");
    println!("├────────────────────────────────┼────────────┼────────────┼──────────┤");
    println!(
        "│ store_key ({n_positions} pos)         │ {store_key_us:>10.2} │ (zero)     │ baseline │"
    );
    println!(
        "│ store_value ({n_positions} pos)       │ {store_value_us:>10.2} │ (zero)     │ baseline │"
    );
    println!(
        "│ dequant_key ({n_positions} pos)       │ {dequant_key_alloc_us:>10.2} │ {dequant_key_zero_us:>10.2} │ {key_delta_pct:>+7.1}% │"
    );
    println!(
        "│ dequant_value ({n_positions} pos)     │ {dequant_val_alloc_us:>10.2} │ {dequant_val_zero_us:>10.2} │ {val_delta_pct:>+7.1}% │"
    );
    println!(
        "│ full store+dequant ({n_positions} pos) │ {full_alloc_us:>10.2} │ {full_zero_us:>10.2} │ {full_delta_pct:>+7.1}% │"
    );
    println!("└────────────────────────────────┴────────────┴────────────┴──────────┘");
    println!();

    // ── Quality gate: verify zero-alloc path produces identical results ──

    let mut cache = TurboQuantKVCache::new(&config, 3, 3);
    cache.reset();
    for pos in 0..n_positions {
        cache.store_key(0, pos, &keys[pos]);
        cache.store_value(0, pos, &vals[pos]);
    }

    let mut zero_buf = vec![0.0f32; kv_dim];
    for pos in 0..n_positions {
        let alloc_result = cache.dequantize_key(0, pos);
        cache.dequantize_key_into(0, pos, &mut zero_buf);

        let max_diff = alloc_result
            .iter()
            .zip(zero_buf.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "dequantize_key mismatch at pos {pos}: max_diff={max_diff}"
        );

        let alloc_result = cache.dequantize_value(0, pos);
        cache.dequantize_value_into(0, pos, &mut zero_buf);

        let max_diff = alloc_result
            .iter()
            .zip(zero_buf.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "dequantize_value mismatch at pos {pos}: max_diff={max_diff}"
        );
    }

    println!("✅ Quality gate: zero-alloc path matches allocating path (max_diff < 1e-6)");
    println!();
}

#[test]
fn bench_turboquant_zero_alloc_forward() {
    let config = Config::micro();
    let kvd = kv_dim(&config);
    let n_positions = 16; // decode positions

    let warmup = 100;
    let iters = 10_000;

    println!("🧪 TurboQuant forward_turboquant Zero-Alloc Benchmark (Plan 051)");
    println!("   kv_dim={kvd}, n_positions={n_positions}, warmup={warmup}, iters={iters}");
    println!("{}", "═".repeat(70));

    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // ── Baseline: forward (flat f32 KV cache) ──

    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    for _ in 0..warmup {
        cache.reset();
        for pos in 0..n_positions {
            let token = rng.next() as usize % config.vocab_size;
            black_box(forward(&mut ctx, &weights, &mut cache, token, pos, &config));
        }
    }

    let start = Instant::now();
    for _ in 0..iters {
        cache.reset();
        for pos in 0..n_positions {
            let token = rng.next() as usize % config.vocab_size;
            black_box(forward(&mut ctx, &weights, &mut cache, token, pos, &config));
        }
    }
    let flat_time = start.elapsed();
    let flat_us = flat_time.as_nanos() as f64 / iters as f64 / 1000.0;

    // ── Optimized: forward_turboquant (zero-alloc KV cache) ──

    let mut tq_cache = TurboQuantKVCache::new(&config, 3, 3);

    for _ in 0..warmup {
        tq_cache.reset();
        for pos in 0..n_positions {
            let token = rng.next() as usize % config.vocab_size;
            black_box(forward_turboquant(
                &mut ctx,
                &weights,
                &mut tq_cache,
                token,
                pos,
                &config,
            ));
        }
    }

    let start = Instant::now();
    for _ in 0..iters {
        tq_cache.reset();
        for pos in 0..n_positions {
            let token = rng.next() as usize % config.vocab_size;
            black_box(forward_turboquant(
                &mut ctx,
                &weights,
                &mut tq_cache,
                token,
                pos,
                &config,
            ));
        }
    }
    let tq_time = start.elapsed();
    let tq_us = tq_time.as_nanos() as f64 / iters as f64 / 1000.0;

    // ── Per-token breakdown at position n_positions/2 (steady state) ──

    let half_pos = n_positions / 2;

    // Flat forward at steady state
    for _ in 0..warmup {
        let token = rng.next() as usize % config.vocab_size;
        black_box(forward(
            &mut ctx, &weights, &mut cache, token, half_pos, &config,
        ));
    }
    let start = Instant::now();
    for _ in 0..iters {
        let token = rng.next() as usize % config.vocab_size;
        black_box(forward(
            &mut ctx, &weights, &mut cache, token, half_pos, &config,
        ));
    }
    let flat_steady_us = start.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;

    // TQ forward at steady state
    for _ in 0..warmup {
        let token = rng.next() as usize % config.vocab_size;
        black_box(forward_turboquant(
            &mut ctx,
            &weights,
            &mut tq_cache,
            token,
            half_pos,
            &config,
        ));
    }
    let start = Instant::now();
    for _ in 0..iters {
        let token = rng.next() as usize % config.vocab_size;
        black_box(forward_turboquant(
            &mut ctx,
            &weights,
            &mut tq_cache,
            token,
            half_pos,
            &config,
        ));
    }
    let tq_steady_us = start.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;

    // ── Results ──

    println!();
    println!("┌─────────────────────────────────┬────────────┬────────────┐");
    println!("│ Forward Path                     │  Time (μs) │ vs Flat    │");
    println!("├─────────────────────────────────┼────────────┼────────────┤");
    println!("│ flat f32 KV ({n_positions} pos decode)  │ {flat_us:>10.2} │ baseline   │");
    let flat_tq_ratio = flat_us / tq_us;
    println!("│ TQ-3bit KV ({n_positions} pos decode)   │ {tq_us:>10.2} │ {flat_tq_ratio:>9.2}× │");
    println!("├─────────────────────────────────┼────────────┼────────────┤");
    println!("│ flat f32 KV (steady pos={half_pos})      │ {flat_steady_us:>10.2} │ baseline   │");
    let steady_ratio = flat_steady_us / tq_steady_us;
    println!(
        "│ TQ-3bit KV (steady pos={half_pos})       │ {tq_steady_us:>10.2} │ {steady_ratio:>9.2}× │"
    );
    println!("└─────────────────────────────────┴────────────┴────────────┘");
    println!();

    // Quality: TQ forward should produce finite logits
    tq_cache.reset();
    let token = 0;
    let logits = forward_turboquant(&mut ctx, &weights, &mut tq_cache, token, 0, &config);
    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "TQ logit[{i}] = {l} is not finite");
    }
    println!("✅ Quality gate: TQ forward logits are finite");
    println!();
}

#[test]
fn bench_turboquant_zero_alloc_per_token() {
    let config = Config::micro();
    let kvd = kv_dim(&config);

    let warmup = 1000;
    let iters = 50_000;

    println!("🧪 TurboQuant Per-Token Cost Breakdown (Plan 051)");
    println!("   kv_dim={kvd}, warmup={warmup}, iters={iters}");
    println!("{}", "═".repeat(70));

    let key: Vec<f32> = (0..kvd).map(|i| (i as f32 * 0.1).sin()).collect();
    let val: Vec<f32> = (0..kvd).map(|i| (i as f32 * 0.2).cos()).collect();

    // ── store_key per call ──

    let mut cache = TurboQuantKVCache::new(&config, 3, 3);
    for _ in 0..warmup {
        cache.store_key(0, 0, &key);
    }
    let start = Instant::now();
    for _ in 0..iters {
        cache.store_key(0, 0, &key);
    }
    let store_key_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    // ── store_value per call ──

    for _ in 0..warmup {
        cache.store_value(0, 0, &val);
    }
    let start = Instant::now();
    for _ in 0..iters {
        cache.store_value(0, 0, &val);
    }
    let store_val_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    // ── dequantize_key (allocating) per call ──

    cache.store_key(0, 0, &key);
    for _ in 0..warmup {
        black_box(cache.dequantize_key(0, 0));
    }
    let start = Instant::now();
    for _ in 0..iters {
        black_box(cache.dequantize_key(0, 0));
    }
    let dequant_key_alloc_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    // ── dequantize_key_into (zero-alloc) per call ──

    let mut buf = vec![0.0f32; kvd];
    for _ in 0..warmup {
        cache.dequantize_key_into(0, 0, &mut buf);
        black_box(&buf);
    }
    let start = Instant::now();
    for _ in 0..iters {
        cache.dequantize_key_into(0, 0, &mut buf);
        black_box(&buf);
    }
    let dequant_key_zero_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    // ── dequantize_value (allocating) per call ──

    cache.store_value(0, 0, &val);
    for _ in 0..warmup {
        black_box(cache.dequantize_value(0, 0));
    }
    let start = Instant::now();
    for _ in 0..iters {
        black_box(cache.dequantize_value(0, 0));
    }
    let dequant_val_alloc_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    // ── dequantize_value_into (zero-alloc) per call ──

    for _ in 0..warmup {
        cache.dequantize_value_into(0, 0, &mut buf);
        black_box(&buf);
    }
    let start = Instant::now();
    for _ in 0..iters {
        cache.dequantize_value_into(0, 0, &mut buf);
        black_box(&buf);
    }
    let dequant_val_zero_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    // ── Results ──

    let key_speedup = dequant_key_alloc_ns / dequant_key_zero_ns;
    let val_speedup = dequant_val_alloc_ns / dequant_val_zero_ns;
    let key_delta_pct = (dequant_key_alloc_ns - dequant_key_zero_ns) / dequant_key_alloc_ns * 100.0;
    let val_delta_pct = (dequant_val_alloc_ns - dequant_val_zero_ns) / dequant_val_alloc_ns * 100.0;

    println!();
    println!("┌───────────────────────────────┬───────────┬───────────┬─────────┬──────────┐");
    println!("│ Operation                      │ Alloc (ns)│ Zero (ns) │ Speedup │    Δ %%   │");
    println!("├───────────────────────────────┼───────────┼───────────┼─────────┼──────────┤");
    println!(
        "│ store_key                     │ {store_key_ns:>9.0} │ (zero)   │    —    │ baseline │"
    );
    println!(
        "│ store_value                   │ {store_val_ns:>9.0} │ (zero)   │    —    │ baseline │"
    );
    println!(
        "│ dequantize_key                │ {dequant_key_alloc_ns:>9.0} │ {dequant_key_zero_ns:>9.0} │ {key_speedup:>6.2}× │ {key_delta_pct:>+7.1}% │"
    );
    println!(
        "│ dequantize_value              │ {dequant_val_alloc_ns:>9.0} │ {dequant_val_zero_ns:>9.0} │ {val_speedup:>6.2}× │ {val_delta_pct:>+7.1}% │"
    );
    println!("└───────────────────────────────┴───────────┴───────────┴─────────┴──────────┘");
    println!();

    // ── Note on kv_dim=16 results ──
    // At kv_dim=16 (Config::micro), the rotation matmul (16×16) dominates ~5000ns compute.
    // Vec allocation for 16 f32s (~64 bytes) is ~30-50ns — only ~1% of total cost.
    // Zero-alloc savings are invisible at this micro-scale.
    // See bench_turboquant_zero_alloc_large_kv for realistic kv_dim=48+ results.
    println!("ℹ️  Note: kv_dim=16 is too small for allocation savings to show per-call.");
    println!("   Allocation ~30-50ns is <1% of ~5000ns compute. See large_kv test below.");
    println!();
}

#[test]
fn bench_turboquant_zero_alloc_large_kv() {
    // Use Config::draft() for more realistic kv_dim=48 where allocation overhead is visible.
    // At kv_dim=48: 3 Vec<f32>[48] + 1 Vec<u8>[48] per call = ~1.5KB allocated.
    // The allocation + dealloc overhead (~100-200ns) becomes a meaningful fraction of compute.
    let config = Config::draft();
    let kvd = kv_dim(&config);

    let warmup = 1000;
    let iters = 50_000;

    println!("🧪 TurboQuant Per-Token Cost Breakdown — Large kv_dim (Plan 051)");
    println!("   kv_dim={kvd} (Config::draft), warmup={warmup}, iters={iters}");
    println!("{}", "═".repeat(70));

    let key: Vec<f32> = (0..kvd).map(|i| (i as f32 * 0.1).sin()).collect();
    let val: Vec<f32> = (0..kvd).map(|i| (i as f32 * 0.2).cos()).collect();

    let mut cache = TurboQuantKVCache::new(&config, 3, 3);

    // ── store_key per call (zero-alloc since Plan 051) ──

    for _ in 0..warmup {
        cache.store_key(0, 0, &key);
    }
    let start = Instant::now();
    for _ in 0..iters {
        cache.store_key(0, 0, &key);
    }
    let store_key_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    // ── store_value per call (zero-alloc since Plan 051) ──

    for _ in 0..warmup {
        cache.store_value(0, 0, &val);
    }
    let start = Instant::now();
    for _ in 0..iters {
        cache.store_value(0, 0, &val);
    }
    let store_val_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    // ── dequantize_key (allocating: 3 Vec per call) ──

    cache.store_key(0, 0, &key);
    for _ in 0..warmup {
        black_box(cache.dequantize_key(0, 0));
    }
    let start = Instant::now();
    for _ in 0..iters {
        black_box(cache.dequantize_key(0, 0));
    }
    let dequant_key_alloc_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    // ── dequantize_key_into (zero-alloc: scratch buffers) ──

    let mut buf = vec![0.0f32; kvd];
    for _ in 0..warmup {
        cache.dequantize_key_into(0, 0, &mut buf);
        black_box(&buf);
    }
    let start = Instant::now();
    for _ in 0..iters {
        cache.dequantize_key_into(0, 0, &mut buf);
        black_box(&buf);
    }
    let dequant_key_zero_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    // ── dequantize_value (allocating) ──

    cache.store_value(0, 0, &val);
    for _ in 0..warmup {
        black_box(cache.dequantize_value(0, 0));
    }
    let start = Instant::now();
    for _ in 0..iters {
        black_box(cache.dequantize_value(0, 0));
    }
    let dequant_val_alloc_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    // ── dequantize_value_into (zero-alloc) ──

    for _ in 0..warmup {
        cache.dequantize_value_into(0, 0, &mut buf);
        black_box(&buf);
    }
    let start = Instant::now();
    for _ in 0..iters {
        cache.dequantize_value_into(0, 0, &mut buf);
        black_box(&buf);
    }
    let dequant_val_zero_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    // ── Results ──

    let key_speedup = dequant_key_alloc_ns / dequant_key_zero_ns;
    let val_speedup = dequant_val_alloc_ns / dequant_val_zero_ns;
    let key_delta_pct = (dequant_key_alloc_ns - dequant_key_zero_ns) / dequant_key_alloc_ns * 100.0;
    let val_delta_pct = (dequant_val_alloc_ns - dequant_val_zero_ns) / dequant_val_alloc_ns * 100.0;

    println!();
    println!("┌───────────────────────────────┬───────────┬───────────┬─────────┬──────────┐");
    println!("│ Operation                      │ Alloc (ns)│ Zero (ns) │ Speedup │    Δ %   │");
    println!("├───────────────────────────────┼───────────┼───────────┼─────────┼──────────┤");
    println!(
        "│ store_key (zero-alloc)        │ {store_key_ns:>9.0} │ (zero)   │    —    │ baseline │"
    );
    println!(
        "│ store_value (zero-alloc)      │ {store_val_ns:>9.0} │ (zero)   │    —    │ baseline │"
    );
    println!(
        "│ dequantize_key                │ {dequant_key_alloc_ns:>9.0} │ {dequant_key_zero_ns:>9.0} │ {key_speedup:>6.2}× │ {key_delta_pct:>+7.1}% │"
    );
    println!(
        "│ dequantize_value              │ {dequant_val_alloc_ns:>9.0} │ {dequant_val_zero_ns:>9.0} │ {val_speedup:>6.2}× │ {val_delta_pct:>+7.1}% │"
    );
    println!("└───────────────────────────────┴───────────┴───────────┴─────────┴──────────┘");
    println!();

    // Quality gate: zero-alloc must produce identical results
    let alloc_result = cache.dequantize_key(0, 0);
    cache.dequantize_key_into(0, 0, &mut buf);
    let max_diff = alloc_result
        .iter()
        .zip(buf.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-6,
        "dequantize_key mismatch at kv_dim={kvd}: max_diff={max_diff}"
    );
    println!("✅ Quality gate: zero-alloc matches allocating path (max_diff < 1e-6)");

    // At kv_dim=48, allocation savings should be measurable.
    // Each allocating dequantize creates 3 Vecs (~576 bytes each).
    // Zero-alloc avoids ~1.7KB allocation + deallocation per call.
    // Assert: zero-alloc path should not be slower than allocating path.
    assert!(
        dequant_key_zero_ns <= dequant_key_alloc_ns * 1.05,
        "dequantize_key_into ({dequant_key_zero_ns:.0}ns) should not be >5% slower than allocating ({dequant_key_alloc_ns:.0}ns)"
    );
    assert!(
        dequant_val_zero_ns <= dequant_val_alloc_ns * 1.05,
        "dequantize_value_into ({dequant_val_zero_ns:.0}ns) should not be >5% slower than allocating ({dequant_val_alloc_ns:.0}ns)"
    );

    if dequant_key_zero_ns < dequant_key_alloc_ns {
        println!(
            "✅ dequantize_key_into is {:.1}% faster ({:.0}ns saved per call)",
            key_delta_pct,
            dequant_key_alloc_ns - dequant_key_zero_ns
        );
    }
    if dequant_val_zero_ns < dequant_val_alloc_ns {
        println!(
            "✅ dequantize_value_into is {:.1}% faster ({:.0}ns saved per call)",
            val_delta_pct,
            dequant_val_alloc_ns - dequant_val_zero_ns
        );
    }
    println!();
}
