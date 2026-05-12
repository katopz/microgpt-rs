//! TurboQuant KV Cache Compression benchmarks.
//! Plan 043 Task 1 (baseline) + Task 7 (quality validation).
//!
//! These benchmarks test:
//! 1. Compression ratio: bytes_per_token comparison (flat f32 vs TQ 2/3/4-bit)
//! 2. Round-trip quality: cos_sim(original, quantize→dequantize) for keys and values
//! 3. Attention fidelity: correlation between flat and TQ attention scores
//!
//! Run with: cargo test bench_turboquant -- --nocapture

use std::hint::black_box;
use std::time::Instant;

use microgpt_rs::transformer::TransformerWeights;
use microgpt_rs::turboquant::TurboQuantKVCache;
use microgpt_rs::turboquant::forward::{
    attention_turboquant, cosine_similarity, dequantize_keys_flat, dequantize_values_flat,
};
use microgpt_rs::types::{Config, Rng, kv_dim};

/// Generate a synthetic key/value vector for position `pos`.
fn synthetic_kv(kv_dim: usize, pos: usize) -> Vec<f32> {
    (0..kv_dim)
        .map(|i| ((i + pos * 7) as f32 * 0.1).sin() + ((i + pos * 3) as f32 * 0.07).cos())
        .collect()
}

/// Pearson correlation coefficient between two slices.
fn pearson_correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len()) as f32;
    if n < 2.0 {
        return 0.0;
    }
    let mean_a: f32 = a.iter().sum::<f32>() / n;
    let mean_b: f32 = b.iter().sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;
    for i in 0..n as usize {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if var_a < 1e-12 || var_b < 1e-12 {
        return 0.0;
    }
    cov / (var_a * var_b).sqrt()
}

#[test]
fn bench_turboquant_compression_ratio() {
    let config = Config::micro();
    let kv_dim = kv_dim(&config);
    let n_positions = config.block_size;

    // Flat f32 bytes per token (K + V, all layers)
    let flat_bytes_per_token = kv_dim * 4 * 2 * config.n_layer; // f32, K+V, per layer

    println!(
        "\n🧪 TurboQuant Compression Ratio (kv_dim={kv_dim}, n_layer={n_layer})",
        n_layer = config.n_layer
    );
    println!("{}", "═".repeat(60));
    println!("Flat f32 bytes/token: {flat_bytes_per_token}");
    println!();

    for bits in [2u8, 3, 4] {
        let cache = TurboQuantKVCache::new(&config, bits, bits);
        let bpt = cache.bytes_per_token();
        let ratio = cache.compression_ratio();
        let pct = bpt as f64 / flat_bytes_per_token as f64 * 100.0;

        println!(
            "  {bits}-bit: {bpt:>4} bytes/token ({pct:>5.1}% of flat, {ratio:.1}× compression)"
        );

        // Validate compression targets
        match bits {
            2 => assert!(ratio > 6.0, "2-bit ratio {ratio} should be > 6.0"),
            3 => assert!(ratio > 4.0, "3-bit ratio {ratio} should be > 4.0"),
            4 => assert!(ratio > 3.0, "4-bit ratio {ratio} should be > 3.0"),
            _ => {}
        }
    }

    // Throughput: time to store + dequantize all positions
    let iterations = 100u64;
    for bits in [2u8, 3, 4] {
        let mut cache = TurboQuantKVCache::new(&config, bits, bits);
        let keys: Vec<Vec<f32>> = (0..n_positions).map(|p| synthetic_kv(kv_dim, p)).collect();
        let vals: Vec<Vec<f32>> = (0..n_positions)
            .map(|p| synthetic_kv(kv_dim, p + 100))
            .collect();

        let start = Instant::now();
        for _ in 0..iterations {
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
        let elapsed = start.elapsed();
        let per_seq = elapsed / iterations as u32;

        println!(
            "  {bits}-bit store+dequant {n_positions} pos: {per_seq:?} ({its} iters)",
            its = iterations
        );
    }
}

#[test]
fn bench_turboquant_roundtrip_quality() {
    let config = Config::micro();
    let kv_dim = kv_dim(&config);
    let n_positions = config.block_size;

    println!("\n🧪 TurboQuant Round-trip Quality (kv_dim={kv_dim}, {n_positions} positions)");
    println!("{}", "═".repeat(60));

    // Generate synthetic KV entries
    let keys: Vec<Vec<f32>> = (0..n_positions).map(|p| synthetic_kv(kv_dim, p)).collect();
    let vals: Vec<Vec<f32>> = (0..n_positions)
        .map(|p| synthetic_kv(kv_dim, p + 100))
        .collect();

    for bits in [2u8, 3, 4] {
        let mut cache = TurboQuantKVCache::new(&config, bits, bits);

        let mut key_cos_sum = 0.0f32;
        let mut val_cos_sum = 0.0f32;
        let mut min_key_cos = f32::MAX;
        let mut min_val_cos = f32::MAX;

        for pos in 0..n_positions {
            cache.store_key(0, pos, &keys[pos]);
            cache.store_value(0, pos, &vals[pos]);

            let recon_key = cache.dequantize_key(0, pos);
            let recon_val = cache.dequantize_value(0, pos);

            let key_cos = cosine_similarity(&keys[pos], &recon_key);
            let val_cos = cosine_similarity(&vals[pos], &recon_val);

            key_cos_sum += key_cos;
            val_cos_sum += val_cos;
            min_key_cos = min_key_cos.min(key_cos);
            min_val_cos = min_val_cos.min(val_cos);
        }

        let avg_key_cos = key_cos_sum / n_positions as f32;
        let avg_val_cos = val_cos_sum / n_positions as f32;

        println!("  {bits}-bit keys:   avg cos_sim={avg_key_cos:.4}, min={min_key_cos:.4}");
        println!("  {bits}-bit values: avg cos_sim={avg_val_cos:.4}, min={min_val_cos:.4}");

        // Quality targets
        match bits {
            4 => {
                assert!(
                    avg_key_cos > 0.95,
                    "4-bit key cos_sim {avg_key_cos} should be > 0.95"
                );
                assert!(
                    avg_val_cos > 0.90,
                    "4-bit val cos_sim {avg_val_cos} should be > 0.90"
                );
            }
            3 => {
                assert!(
                    avg_key_cos > 0.90,
                    "3-bit key cos_sim {avg_key_cos} should be > 0.90"
                );
                assert!(
                    avg_val_cos > 0.85,
                    "3-bit val cos_sim {avg_val_cos} should be > 0.85"
                );
            }
            2 => {
                assert!(
                    avg_key_cos > 0.80,
                    "2-bit key cos_sim {avg_key_cos} should be > 0.80"
                );
                assert!(
                    avg_val_cos > 0.75,
                    "2-bit val cos_sim {avg_val_cos} should be > 0.75"
                );
            }
            _ => {}
        }
        println!();
    }
}

#[test]
fn bench_turboquant_attention_fidelity() {
    let config = Config::micro();
    let kv_dim = kv_dim(&config);
    let head_dim = config.head_dim;
    let n_head = config.n_head;
    let n_embd = config.n_embd;
    let n_positions = config.block_size;

    println!("\n🧪 TurboQuant Attention Fidelity (head_dim={head_dim}, n_head={n_head})");
    println!("{}", "═".repeat(60));

    let mut rng = Rng::new(42);
    let _weights = TransformerWeights::new(&config, &mut rng);

    // Generate random query
    let query: Vec<f32> = (0..n_embd).map(|_| rng.normal()).collect();

    // Generate synthetic KV entries
    let keys: Vec<Vec<f32>> = (0..n_positions).map(|p| synthetic_kv(kv_dim, p)).collect();
    let vals: Vec<Vec<f32>> = (0..n_positions)
        .map(|p| synthetic_kv(kv_dim, p + 100))
        .collect();

    // Build flat KV cache for reference scores
    let mut flat_keys = vec![0.0f32; n_positions * kv_dim];
    let mut flat_values = vec![0.0f32; n_positions * kv_dim];
    for pos in 0..n_positions {
        flat_keys[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&keys[pos]);
        flat_values[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&vals[pos]);
    }

    // Compute flat reference attention scores for head 0
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut flat_scores = vec![0.0f32; n_positions];
    for t in 0..n_positions {
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += query[d] * flat_keys[t * kv_dim + d];
        }
        flat_scores[t] = dot * scale;
    }

    for bits in [2u8, 3, 4] {
        let mut cache = TurboQuantKVCache::new(&config, bits, bits);
        for pos in 0..n_positions {
            cache.store_key(0, pos, &keys[pos]);
            cache.store_value(0, pos, &vals[pos]);
        }

        // Dequantize keys for TQ attention
        let tq_keys = dequantize_keys_flat(&cache, 0, n_positions - 1, kv_dim);
        let tq_values = dequantize_values_flat(&cache, 0, n_positions - 1, kv_dim);

        // Compute TQ attention scores for head 0
        let mut tq_scores = vec![0.0f32; n_positions];
        for t in 0..n_positions {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += query[d] * tq_keys[t * kv_dim + d];
            }
            tq_scores[t] = dot * scale;
        }

        // Correlation between flat and TQ attention scores
        let correlation = pearson_correlation(&flat_scores, &tq_scores);

        // Cosine similarity between score vectors
        let score_cos = cosine_similarity(&flat_scores, &tq_scores);

        // Full attention forward with TQ
        let mut attn_out_flat = vec![0.0f32; n_embd];
        let mut attn_out_tq = vec![0.0f32; n_embd];
        let mut scores_buf = vec![0.0f32; config.block_size];

        attention_turboquant(
            &query,
            &flat_keys,
            &flat_values,
            &mut attn_out_flat,
            &mut scores_buf,
            0,
            0,
            kv_dim,
            head_dim,
            n_positions - 1,
            scale,
        );

        attention_turboquant(
            &query,
            &tq_keys,
            &tq_values,
            &mut attn_out_tq,
            &mut scores_buf,
            0,
            0,
            kv_dim,
            head_dim,
            n_positions - 1,
            scale,
        );

        let output_cos = cosine_similarity(&attn_out_flat, &attn_out_tq);

        println!("  {bits}-bit:");
        println!("    Score correlation:  {correlation:.4}");
        println!("    Score cos_sim:      {score_cos:.4}");
        println!("    Output cos_sim:     {output_cos:.4}");

        // Fidelity targets
        match bits {
            4 => {
                assert!(
                    correlation > 0.98,
                    "4-bit correlation {correlation} should be > 0.98"
                );
                assert!(
                    output_cos > 0.95,
                    "4-bit output cos_sim {output_cos} should be > 0.95"
                );
            }
            3 => {
                assert!(
                    correlation > 0.95,
                    "3-bit correlation {correlation} should be > 0.95"
                );
                assert!(
                    output_cos > 0.90,
                    "3-bit output cos_sim {output_cos} should be > 0.90"
                );
            }
            2 => {
                assert!(
                    correlation > 0.85,
                    "2-bit correlation {correlation} should be > 0.85"
                );
                assert!(
                    output_cos > 0.80,
                    "2-bit output cos_sim {output_cos} should be > 0.80"
                );
            }
            _ => {}
        }
        println!();
    }
}
