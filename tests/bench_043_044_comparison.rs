//! Plan 043 + 044 Before/After Comparison Benchmarks
//!
//! Comprehensive benchmark comparing baseline vs optimized implementations:
//! - Plan 043 (TurboQuant): flat f32 KV cache vs bit-packed quantized cache
//! - Plan 044 (PFlash): full prefill vs block-sparse speculative prefill
//!
//! Run with: cargo test bench_043_044 -- --nocapture

use std::hint::black_box;
use std::time::Instant;

use microgpt_rs::speculative::types::FlashPrefillConfig;
use microgpt_rs::speculative::{block_select, block_select_grid, compress_prompt_blocks};
use microgpt_rs::transformer::TransformerWeights;
use microgpt_rs::turboquant::TurboQuantKVCache;
use microgpt_rs::turboquant::forward::{
    attention_turboquant, cosine_similarity, dequantize_keys_flat, dequantize_values_flat,
};
use microgpt_rs::types::{Config, Rng, kv_dim};

/// Generate sparse importance scores: mostly hay (0.01) with a few needle peaks (1.0).
/// Simulates real attention patterns where most tokens are unimportant.
fn sparse_scores(len: usize, seed: u64, needle_density: f64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = state as f64 / u64::MAX as f64;
            if r < needle_density { 1.0f32 } else { 0.01f32 }
        })
        .collect()
}

// ── Helpers ──────────────────────────────────────────────────

/// Synthetic KV vector for position `pos`.
fn synthetic_kv(kv_dim: usize, pos: usize) -> Vec<f32> {
    (0..kv_dim)
        .map(|i| ((i + pos * 7) as f32 * 0.1).sin() + ((i + pos * 3) as f32 * 0.07).cos())
        .collect()
}

/// Deterministic pseudo-random scores from seed.
fn deterministic_scores(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state as f32 / u64::MAX as f32).min(1.0)
        })
        .collect()
}

/// Pearson correlation between two slices.
fn pearson_correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len()) as f32;
    if n < 2.0 {
        return 0.0;
    }
    let mean_a: f32 = a.iter().sum::<f32>() / n;
    let mean_b: f32 = b.iter().sum::<f32>() / n;
    let (cov, var_a, var_b) =
        a.iter()
            .zip(b)
            .fold((0.0f32, 0.0f32, 0.0f32), |(c, va, vb), (ai, bi)| {
                let da = ai - mean_a;
                let db = bi - mean_b;
                (c + da * db, va + da * da, vb + db * db)
            });
    if var_a < 1e-12 || var_b < 1e-12 {
        return 0.0;
    }
    cov / (var_a * var_b).sqrt()
}

/// Write a CSV row.
fn csv_row(fields: &[&str]) -> String {
    fields
        .iter()
        .map(|f| {
            if f.contains(',') || f.contains('"') {
                format!("\"{}\"", f.replace('"', "\"\""))
            } else {
                f.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

// ══════════════════════════════════════════════════════════════
// Plan 043: TurboQuant KV Cache Compression — Before vs After
// ══════════════════════════════════════════════════════════════

#[test]
fn bench_043_turboquant_before_after() {
    let config = Config::micro();
    let kv_dim = kv_dim(&config);
    let head_dim = config.head_dim;
    let n_embd = config.n_embd;
    let n_layer = config.n_layer;
    let n_positions = config.block_size;
    let mut rng = Rng::new(42);
    let _weights = TransformerWeights::new(&config, &mut rng);

    // Flat f32 bytes per token (K + V, all layers)
    let flat_bytes_per_token = kv_dim * 4 * 2 * n_layer;

    // Generate synthetic KV data once
    let keys: Vec<Vec<f32>> = (0..n_positions).map(|p| synthetic_kv(kv_dim, p)).collect();
    let vals: Vec<Vec<f32>> = (0..n_positions)
        .map(|p| synthetic_kv(kv_dim, p + 100))
        .collect();

    // Generate random query for attention fidelity
    let query: Vec<f32> = (0..n_embd).map(|_| rng.normal()).collect();
    let scale = 1.0 / (head_dim as f32).sqrt();

    println!("\n{}", "═".repeat(72));
    println!("  Plan 043: TurboQuant KV Cache Compression — Before vs After");
    println!("{}", "═".repeat(72));
    println!(
        "  Config: kv_dim={kv_dim}, head_dim={head_dim}, n_layer={n_layer}, seq_len={n_positions}"
    );
    println!();

    // ── Memory Comparison ──────────────────────────────────────
    println!("  ┌─ Memory ──────────────────────────────────────────────────┐");
    println!("  │ Flat f32 bytes/token:   {flat_bytes_per_token:>8}                       │");
    let mut csv_rows: Vec<String> = Vec::new();
    csv_rows.push(csv_row(&[
        "benchmark",
        "metric",
        "before",
        "after",
        "unit",
        "gain",
        "quality",
    ]));

    for bits in [2u8, 3, 4] {
        let cache = TurboQuantKVCache::new(&config, bits, bits);
        let bpt = cache.bytes_per_token();
        let ratio = cache.compression_ratio();
        let pct = bpt as f64 / flat_bytes_per_token as f64 * 100.0;
        println!(
            "  │ {bits}-bit bytes/token:      {bpt:>8}  ({pct:>5.1}% of flat, {ratio:.1}×)    │"
        );

        csv_rows.push(csv_row(&[
            &format!("TQ-{bits}bit"),
            "bytes_per_token",
            &flat_bytes_per_token.to_string(),
            &bpt.to_string(),
            "bytes",
            &format!("{ratio:.1}x"),
            "-",
        ]));
    }
    println!("  └───────────────────────────────────────────────────────────┘");
    println!();

    // ── Throughput: Store + Dequantize ─────────────────────────
    println!("  ┌─ Store+Dequantize Throughput ({n_positions} positions) ──────────────┐");
    println!("  │ Note: TQ has higher compute cost per token (rotation + quant),    │");
    println!("  │ but saves memory bandwidth. Net win at long contexts where       │");
    println!("  │ memory BW is the bottleneck, not compute.                        │");

    let iters = 100u64;

    // Baseline: flat f32 copy (store + read back)
    let start = Instant::now();
    for _ in 0..iters {
        let mut flat_keys = vec![0.0f32; n_positions * kv_dim];
        let mut flat_vals = vec![0.0f32; n_positions * kv_dim];
        for pos in 0..n_positions {
            flat_keys[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&keys[pos]);
            flat_vals[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&vals[pos]);
        }
        black_box(&flat_keys);
        black_box(&flat_vals);
    }
    let flat_elapsed = start.elapsed();
    let flat_per_seq = flat_elapsed / iters as u32;
    let flat_tok_per_sec = (n_positions as f64 * iters as f64) / flat_elapsed.as_secs_f64();

    println!("  │ Flat f32:      {flat_per_seq:>10?}  ({flat_tok_per_sec:>10.0} tok/s)       │");

    csv_rows.push(csv_row(&[
        "TQ-store",
        "throughput",
        &format!("{flat_tok_per_sec:.0}"),
        "", // filled per-bit below
        "tok/s",
        "",
        "",
    ]));

    for bits in [2u8, 3, 4] {
        let mut cache = TurboQuantKVCache::new(&config, bits, bits);

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
        let elapsed = start.elapsed();
        let per_seq = elapsed / iters as u32;
        let tok_per_sec = (n_positions as f64 * iters as f64) / elapsed.as_secs_f64();
        let overhead = elapsed.as_secs_f64() / flat_elapsed.as_secs_f64();

        println!(
            "  │ TQ {bits}-bit:      {per_seq:>10?}  ({tok_per_sec:>10.0} tok/s, {overhead:.1}× compute) │"
        );

        csv_rows.push(csv_row(&[
            &format!("TQ-{bits}bit"),
            "store+dequant",
            &format!("{flat_tok_per_sec:.0}"),
            &format!("{tok_per_sec:.0}"),
            "tok/s",
            &format!("{overhead:.1}x compute"),
            "memory wins at scale",
        ]));
    }
    println!("  └───────────────────────────────────────────────────────────┘");
    println!();

    // ── Quality: Round-trip Cosine Similarity ──────────────────
    println!("  ┌─ Round-trip Quality (cosine similarity) ──────────────────┐");
    println!("  │ Flat f32 is baseline (1.0000 by definition)               │");
    println!("  │                                                           │");

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

        println!(
            "  │ {bits}-bit keys:   avg={avg_key_cos:.4}, min={min_key_cos:.4}                  │"
        );
        println!(
            "  │ {bits}-bit values: avg={avg_val_cos:.4}, min={min_val_cos:.4}                  │"
        );

        csv_rows.push(csv_row(&[
            &format!("TQ-{bits}bit"),
            "key_cos_sim",
            "1.0000",
            &format!("{avg_key_cos:.4}"),
            "cos_sim",
            &format!("{:.4}", 1.0 - avg_key_cos),
            &format!("min={min_key_cos:.4}"),
        ]));
        csv_rows.push(csv_row(&[
            &format!("TQ-{bits}bit"),
            "val_cos_sim",
            "1.0000",
            &format!("{avg_val_cos:.4}"),
            "cos_sim",
            &format!("{:.4}", 1.0 - avg_val_cos),
            &format!("min={min_val_cos:.4}"),
        ]));
    }
    println!("  └───────────────────────────────────────────────────────────┘");
    println!();

    // ── Attention Fidelity ─────────────────────────────────────
    println!("  ┌─ Attention Score Fidelity ────────────────────────────────┐");

    // Compute flat reference attention scores for head 0
    let mut flat_keys = vec![0.0f32; n_positions * kv_dim];
    let mut flat_values = vec![0.0f32; n_positions * kv_dim];
    for pos in 0..n_positions {
        flat_keys[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&keys[pos]);
        flat_values[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&vals[pos]);
    }

    let mut flat_scores = vec![0.0f32; n_positions];
    for t in 0..n_positions {
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += query[d] * flat_keys[t * kv_dim + d];
        }
        flat_scores[t] = dot * scale;
    }

    // Flat attention forward
    let mut attn_out_flat = vec![0.0f32; n_embd];
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

    for bits in [2u8, 3, 4] {
        let mut cache = TurboQuantKVCache::new(&config, bits, bits);
        for pos in 0..n_positions {
            cache.store_key(0, pos, &keys[pos]);
            cache.store_value(0, pos, &vals[pos]);
        }

        let tq_keys = dequantize_keys_flat(&cache, 0, n_positions - 1, kv_dim);
        let tq_values = dequantize_values_flat(&cache, 0, n_positions - 1, kv_dim);

        let mut tq_scores = vec![0.0f32; n_positions];
        for t in 0..n_positions {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += query[d] * tq_keys[t * kv_dim + d];
            }
            tq_scores[t] = dot * scale;
        }

        let correlation = pearson_correlation(&flat_scores, &tq_scores);
        let score_cos = cosine_similarity(&flat_scores, &tq_scores);

        let mut attn_out_tq = vec![0.0f32; n_embd];
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

        println!(
            "  │ {bits}-bit: corr={correlation:.4}, score_cos={score_cos:.4}, out_cos={output_cos:.4} │"
        );

        csv_rows.push(csv_row(&[
            &format!("TQ-{bits}bit"),
            "attention_correlation",
            "1.0000",
            &format!("{correlation:.4}"),
            "pearson_r",
            &format!("{:.4}", 1.0 - correlation),
            &format!("output_cos={output_cos:.4}"),
        ]));
    }
    println!("  └───────────────────────────────────────────────────────────┘");
    println!();

    // ── Memory at Scale ────────────────────────────────────────
    println!("  ┌─ Memory at Scale (hypothetical 32K context, 32 layers, hd=128) ──┐");
    let scale_tokens = 32_768usize;
    let scale_kv_dim = 128usize;
    let scale_layers = 32usize;
    let flat_total = scale_tokens * scale_kv_dim * 4 * 2 * scale_layers;
    println!(
        "  │ Flat f32:  {flat_total:>12} bytes ({:>6.1} MB)                          │",
        flat_total as f64 / 1e6
    );

    for bits in [2u8, 3, 4] {
        let packed_per_vec = match bits {
            2 => scale_kv_dim / 4,
            3 | 4 => scale_kv_dim / 2,
            _ => scale_kv_dim,
        };
        let tq_total = scale_tokens * (packed_per_vec + 8) * 2 * scale_layers;
        let ratio = flat_total as f64 / tq_total as f64;
        println!(
            "  │ TQ {bits}-bit: {tq_total:>12} bytes ({:>6.1} MB)  → {ratio:.1}× compression     │",
            tq_total as f64 / 1e6
        );

        csv_rows.push(csv_row(&[
            &format!("TQ-{bits}bit"),
            "32K_ctx_memory",
            &format!("{:.1}", flat_total as f64 / 1e6),
            &format!("{:.1}", tq_total as f64 / 1e6),
            "MB",
            &format!("{ratio:.1}x"),
            "-",
        ]));
    }
    println!("  └───────────────────────────────────────────────────────────┘");

    // ══════════════════════════════════════════════════════════════
    // Plan 044: PFlash Block-Sparse Speculative Prefill
    // ══════════════════════════════════════════════════════════════

    println!("\n{}", "═".repeat(72));
    println!("  Plan 044: PFlash Block-Sparse Speculative Prefill — Before vs After");
    println!("{}", "═".repeat(72));
    println!();

    // ── PFlash: Compression with realistic sparse scores ───────
    // The default FlashPrefillConfig has last_n_full=1, which means
    // q_block >= num_blocks-1 is always true → all blocks kept.
    // Real PFlash uses alpha-only selection for long prompts (128K+).
    // Here we test with sparse scores (95% hay, 5% needles) to show
    // the compression potential when blocks can be differentiated.

    let prompt_lengths_pf: &[usize] = &[512, 1024, 2048, 4096];
    let alphas: &[f32] = &[0.05, 0.12, 0.25, 0.50];

    // Config that actually demonstrates compression: disable last_n_full
    let mut cfg_compress = FlashPrefillConfig {
        attention_sink: 1, // keep first block
        window: 1,         // keep last block
        last_n_full: 0,    // allow middle blocks to be dropped
        ..Default::default()
    };

    println!("  ┌─ Compression by Prompt Length × Alpha (sparse scores) ───┐");
    println!("  │ Before: all tokens kept (100%)                            │");
    println!("  │ Scores: 5% needles (score=1.0), 95% hay (score=0.01)     │");
    println!("  │ Rules: sink=1, window=1, last_n_full=0 (drop middle ok)  │");
    println!("  │                                                           │");

    let mut pf_best_ratio = 1.0f64;
    let mut pf_long_scores: Vec<f32> = Vec::new();
    let pf_long_len = 2048usize;

    for &prompt_len in prompt_lengths_pf {
        let num_blocks = prompt_len / 32;
        let scores = sparse_scores(prompt_len, 99, 0.05);

        if prompt_len == pf_long_len {
            pf_long_scores = scores.clone();
        }

        let mut row = format!("  │ len={prompt_len:>5} ({num_blocks:>3} blocks): ");
        let mut first = true;

        for &alpha in alphas {
            cfg_compress.alpha = alpha;

            let selected = compress_prompt_blocks(&scores, &cfg_compress, 2, 2);
            let ratio = selected.len() as f64 / prompt_len as f64;
            let reduction = if ratio > 0.001 { 1.0 / ratio } else { 999.0 };

            if !first {
                row.push_str(" | ");
            }
            row.push_str(&format!("α={alpha:.2}→{:.0}%", ratio * 100.0));
            first = false;

            if ratio < pf_best_ratio && prompt_len == pf_long_len {
                pf_best_ratio = ratio;
            }

            csv_rows.push(csv_row(&[
                &format!("PF-{prompt_len}-alpha{alpha:.2}"),
                "token_reduction",
                &prompt_len.to_string(),
                &selected.len().to_string(),
                "tokens",
                &format!("{reduction:.1}x"),
                &format!("{:.1}%", ratio * 100.0),
            ]));
        }
        while row.len() < 63 {
            row.push(' ');
        }
        row.push('│');
        println!("{row}");
    }
    println!("  └───────────────────────────────────────────────────────────┘");
    println!();

    // ── Also show default config (all kept) for comparison ─────
    println!("  ┌─ Default Config (last_n_full=1): All Blocks Kept ────────┐");
    println!("  │ With default config, q_block >= num_blocks-1 is always   │");
    println!("  │ true, so all blocks are kept. This is by design: the     │");
    println!("  │ last_n_full rule ensures quality when prompt is short.   │");
    println!("  │ Real PFlash at 128K+ tokens shows meaningful compression.│");
    let cfg_default = FlashPrefillConfig::default();
    for &prompt_len in &[512, 2048] {
        let scores = sparse_scores(prompt_len, 99, 0.05);
        let selected = compress_prompt_blocks(&scores, &cfg_default, 2, 2);
        let ratio = selected.len() as f64 / prompt_len as f64;
        println!(
            "  │   len={prompt_len}: kept={}/{} ({:.0}%) — last_n_full keeps all  │",
            selected.len(),
            prompt_len,
            ratio * 100.0
        );
    }
    println!("  └───────────────────────────────────────────────────────────┘");
    println!();

    // ── Compression Latency by Prompt Length ───────────────────
    println!("  ┌─ Compression Latency by Prompt Length (α=0.12, sparse) ──┐");

    for &prompt_len in prompt_lengths_pf {
        let scores = sparse_scores(prompt_len, 99, 0.05);
        cfg_compress.alpha = 0.12;
        let iters_pf = 1_000;

        let start = Instant::now();
        let mut selected = Vec::new();
        for _ in 0..iters_pf {
            selected = compress_prompt_blocks(&scores, &cfg_compress, 2, 2);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / iters_pf as u32;

        let ratio = selected.len() as f64 / prompt_len as f64;
        let reduction = if ratio > 0.001 { 1.0 / ratio } else { 999.0 };

        println!(
            "  │ len={prompt_len:>5}: {per_call:>8?}  kept={kept:>5} ({reduction:>5.1}× reduction)   │",
            kept = selected.len()
        );

        csv_rows.push(csv_row(&[
            &format!("PF-latency-{prompt_len}"),
            "compression_time",
            "N/A",
            &format!("{:.1}", per_call.as_secs_f64() * 1e6),
            "µs",
            "-",
            &format!("{reduction:.1}x reduction"),
        ]));
    }
    println!("  └───────────────────────────────────────────────────────────┘");
    println!();

    // ── block_select Throughput ────────────────────────────────
    println!("  ┌─ block_select Throughput ─────────────────────────────────┐");

    let cfg = FlashPrefillConfig::default();
    let prompt_lengths: &[usize] = &[64, 128, 256, 512, 1024, 2048];

    for &len in prompt_lengths {
        let num_blocks = len / cfg.block_size;
        let scores = deterministic_scores(num_blocks, 42);
        let iters_bs = 10_000;

        let start = Instant::now();
        for _ in 0..iters_bs {
            black_box(block_select(&scores, &cfg));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / iters_bs as u32;
        let blocks_per_sec = num_blocks as f64 * iters_bs as f64 / elapsed.as_secs_f64();

        println!(
            "  │ len={len:>5} ({num_blocks:>4} blocks): {per_call:>6?} ({blocks_per_sec:>10.0} blocks/s) │"
        );

        csv_rows.push(csv_row(&[
            &format!("PF-block_select-{len}"),
            "throughput",
            "N/A",
            &format!("{blocks_per_sec:.0}"),
            "blocks/s",
            "-",
            &format!("{num_blocks} blocks"),
        ]));
    }
    println!("  └───────────────────────────────────────────────────────────┘");
    println!();

    // ── block_select_grid Throughput ───────────────────────────
    println!("  ┌─ block_select_grid Throughput (multi-head) ──────────────┐");

    let grid_configs: &[(usize, usize, usize)] = &[(4, 4, 4), (8, 16, 4), (16, 32, 8), (32, 64, 8)];

    for &(m, n, h) in grid_configs {
        let grid: Vec<f32> = deterministic_scores(m * n * h, 77);
        let iters_grid = 5_000;

        let start = Instant::now();
        for _ in 0..iters_grid {
            black_box(block_select_grid(&grid, m, n, h, &cfg));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / iters_grid as u32;
        let grid_sz = m * n * h;

        println!("  │ {m:>3}×{n:>3}×{h:>2} ({grid_sz:>6} entries): {per_call:>6?}              │");

        csv_rows.push(csv_row(&[
            &format!("PF-grid-{m}x{n}x{h}"),
            "throughput",
            "N/A",
            &format!(
                "{:.0}",
                grid_sz as f64 * iters_grid as f64 / elapsed.as_secs_f64()
            ),
            "entries/s",
            "-",
            &format!("{grid_sz} entries"),
        ]));
    }
    println!("  └───────────────────────────────────────────────────────────┘");
    println!();

    // ── NIAH Needle Retrieval ────────────────────────────────────
    println!("  ┌─ NIAH Needle Retrieval ───────────────────────────────────┐");

    let niah_lengths: &[usize] = &[128, 256, 512, 1024];
    let niah_alphas: &[f32] = &[0.05, 0.12, 0.25, 0.50];

    // Test both configs: default (all kept) and compress (can drop)
    println!("  │ Config A (default, last_n_full=1): always keeps all      │");
    println!("  │ Config B (compress, last_n_full=0): can drop middle blocks│");
    println!("  │                                                           │");

    // Config B for NIAH
    let mut cfg_niah_b = FlashPrefillConfig {
        attention_sink: 1,
        window: 1,
        last_n_full: 0,
        ..Default::default()
    };

    let mut total_cases_a = 0u32;
    let mut passed_cases_a = 0u32;
    let mut total_cases_b = 0u32;
    let mut passed_cases_b = 0u32;

    for &prompt_len in niah_lengths {
        for &alpha in niah_alphas {
            let hay_per_side = (prompt_len - 2) / 2;
            let needle_pos = hay_per_side;
            let secret_pos = hay_per_side + 1;
            let actual_len = hay_per_side * 2 + 2;

            // Crafted scores: needle stands out
            let mut niah_scores = vec![0.01f32; actual_len];
            niah_scores[needle_pos] = 1.0;
            niah_scores[secret_pos] = 0.95;

            // Config A: default
            let cfg_a = FlashPrefillConfig {
                alpha,
                ..Default::default()
            };
            let sel_a = compress_prompt_blocks(&niah_scores, &cfg_a, 2, 2);
            let survives_a = sel_a.contains(&needle_pos) && sel_a.contains(&secret_pos);
            total_cases_a += 1;
            if survives_a {
                passed_cases_a += 1;
            }

            // Config B: compress
            cfg_niah_b.alpha = alpha;
            let sel_b = compress_prompt_blocks(&niah_scores, &cfg_niah_b, 2, 2);
            let survives_b = sel_b.contains(&needle_pos) && sel_b.contains(&secret_pos);
            total_cases_b += 1;
            if survives_b {
                passed_cases_b += 1;
            }
        }
    }

    let rate_a = passed_cases_a as f64 / total_cases_a as f64 * 100.0;
    let rate_b = passed_cases_b as f64 / total_cases_b as f64 * 100.0;

    println!(
        "  │ Config A retrieval: {rate_a:>5.1}% ({passed_cases_a}/{total_cases_a} cases)              │"
    );
    println!(
        "  │ Config B retrieval: {rate_b:>5.1}% ({passed_cases_b}/{total_cases_b} cases)              │"
    );
    println!("  │                                                           │");
    println!("  │ Note: retrieval tested with crafted importance scores.    │");
    println!("  │ Config A: needle always survives (last_n_full keeps all). │");
    println!("  │ Config B: needle survives when block scores are high.     │");

    csv_rows.push(csv_row(&[
        "PFlash-default",
        "niah_retrieval",
        "100.0",
        &format!("{rate_a:.1}"),
        "%",
        &format!("{:.1}%", 100.0 - rate_a),
        &format!("{passed_cases_a}/{total_cases_a}"),
    ]));
    csv_rows.push(csv_row(&[
        "PFlash-compress",
        "niah_retrieval",
        "100.0",
        &format!("{rate_b:.1}"),
        "%",
        &format!("{:.1}%", 100.0 - rate_b),
        &format!("{passed_cases_b}/{total_cases_b}"),
    ]));
    println!("  └───────────────────────────────────────────────────────────┘");
    println!();

    // ── Combined TTFT Estimate ─────────────────────────────────
    println!("  ┌─ Combined Effect Estimate (2048-token prompt, TTFT) ─────┐");
    println!("  │                                                           │");
    println!("  │ Plan 043 (TQ):  KV cache 5-8× smaller → less memory BW   │");
    println!("  │ Plan 044 (PF):  Prefill 2-10× fewer tokens → less compute│");
    println!("  │ Combined:       Both reductions multiply                  │");

    let combined_alphas: &[f32] = &[0.05, 0.12, 0.25, 0.50];
    for &alpha in combined_alphas {
        cfg_compress.alpha = alpha;
        let sel = compress_prompt_blocks(&pf_long_scores, &cfg_compress, 2, 2);
        let pf_ratio = sel.len() as f64 / pf_long_len as f64;

        for bits in [3u8, 4] {
            let cache = TurboQuantKVCache::new(&config, bits, bits);
            let tq_ratio = 1.0 / cache.compression_ratio();
            let combined = pf_ratio * tq_ratio;
            let combined_x = 1.0 / combined;
            println!(
                "  │   TQ {bits}-bit + PF α={alpha:.2}: seq={:>5.1}%, mem={:.1}% → {:.1}% ({combined_x:>5.1}×) │",
                pf_ratio * 100.0,
                tq_ratio * 100.0,
                combined * 100.0,
            );

            csv_rows.push(csv_row(&[
                &format!("Combined-TQ{bits}-PF{alpha:.2}"),
                "combined_resource",
                "100.0",
                &format!("{:.1}", combined * 100.0),
                "%",
                &format!("{combined_x:.1}x"),
                &format!("seq={:.1}% mem={:.1}%", pf_ratio * 100.0, tq_ratio * 100.0),
            ]));
        }
    }
    println!("  └───────────────────────────────────────────────────────────┘");

    // ── Assertions ─────────────────────────────────────────────
    // TurboQuant: 3-bit should be > 4× compression, > 0.85 cosine
    let cache_3bit = TurboQuantKVCache::new(&config, 3, 3);
    assert!(
        cache_3bit.compression_ratio() > 4.0,
        "3-bit compression ratio should be > 4.0"
    );

    let mut cache_3bit_mut = TurboQuantKVCache::new(&config, 3, 3);
    let test_key = synthetic_kv(kv_dim, 0);
    cache_3bit_mut.store_key(0, 0, &test_key);
    let recon = cache_3bit_mut.dequantize_key(0, 0);
    let cos = cosine_similarity(&test_key, &recon);
    assert!(cos > 0.85, "3-bit key cos_sim {cos} should be > 0.85");

    // PFlash: retrieval should be >= 50%
    assert!(
        rate_b >= 50.0,
        "NIAH retrieval {rate_b:.0}% should be >= 50%"
    );

    println!("\n  ✅ All assertions passed.");
}
