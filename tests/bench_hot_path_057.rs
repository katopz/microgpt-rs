//! Hot-path component benchmarks for Issue 057.
//!
//! Measures before/after for:
//! 1. rmsnorm (scalar vs SIMD) on standalone buffer
//! 2. softmax + softmax_scaled on standalone buffer
//! 3. forward decode at pos=0, pos=64, pos=127 (exercises attention_head)
//!
//! Run with: cargo test bench_hot_path_057 -- --nocapture

use std::hint::black_box;
use std::time::Instant;

use microgpt_rs::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use microgpt_rs::types::{Config, Rng, rmsnorm, softmax, softmax_scaled};

const WARMUP: usize = 100;
const ITERS: usize = 10000;

#[test]
fn bench_hot_path_057() {
    let config = Config::micro();

    // Game config for forward tests (block_size=170 supports pos=127)
    let game_config = Config::game();
    let mut game_rng = Rng::new(42);
    let game_weights = TransformerWeights::new(&game_config, &mut game_rng);
    let mut game_ctx = ForwardContext::new(&game_config);
    let mut game_cache = MultiLayerKVCache::new(&game_config);

    let n = config.n_embd;
    let vocab = config.vocab_size;

    // ── Standalone buffers for component benchmarks (avoids private field access) ──
    let mut rmsnorm_buf = vec![1.0f32; n];
    let mut softmax_buf = vec![0.5f32; vocab];

    // ── Component A: rmsnorm (called 2× per layer × n_layer per token) ──
    for _ in 0..WARMUP {
        rmsnorm_buf.fill(1.0);
        rmsnorm(&mut rmsnorm_buf);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        rmsnorm_buf.fill(1.0);
        rmsnorm(&mut rmsnorm_buf);
    }
    let t_rmsnorm = start.elapsed();
    let rmsnorm_us = t_rmsnorm.as_micros() as f64 / ITERS as f64;

    // ── Component B: softmax (called once per token decode) ──
    for _ in 0..WARMUP {
        softmax_buf.fill(0.5);
        softmax(&mut softmax_buf);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        softmax_buf.fill(0.5);
        softmax(&mut softmax_buf);
    }
    let t_softmax = start.elapsed();
    let softmax_us = t_softmax.as_micros() as f64 / ITERS as f64;

    // ── Component C: softmax_scaled ──
    let inv_temp = 1.0 / config.temperature;
    for _ in 0..WARMUP {
        softmax_buf.fill(0.5);
        softmax_scaled(&mut softmax_buf, inv_temp);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        softmax_buf.fill(0.5);
        softmax_scaled(&mut softmax_buf, inv_temp);
    }
    let t_softmax_scaled = start.elapsed();
    let softmax_scaled_us = t_softmax_scaled.as_micros() as f64 / ITERS as f64;

    // ── Component D: forward decode at pos=0 (short attention, 1 position) ──
    // Uses game_config (block_size=256) so all positions fit within bounds
    game_cache.reset();
    for _ in 0..WARMUP {
        black_box(forward(
            &mut game_ctx,
            &game_weights,
            &mut game_cache,
            0,
            0,
            &game_config,
        ));
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        game_cache.reset();
        black_box(forward(
            &mut game_ctx,
            &game_weights,
            &mut game_cache,
            0,
            0,
            &game_config,
        ));
    }
    let t_fwd_0 = start.elapsed();
    let fwd_0_us = t_fwd_0.as_micros() as f64 / ITERS as f64;

    // ── Component E: forward decode at pos=64 (longer attention, 65 positions) ──
    game_cache.reset();
    for pos in 0..64 {
        forward(
            &mut game_ctx,
            &game_weights,
            &mut game_cache,
            0,
            pos,
            &game_config,
        );
    }
    for _ in 0..WARMUP {
        black_box(forward(
            &mut game_ctx,
            &game_weights,
            &mut game_cache,
            0,
            64,
            &game_config,
        ));
    }
    let start = Instant::now();
    for _ in 0..1000 {
        black_box(forward(
            &mut game_ctx,
            &game_weights,
            &mut game_cache,
            0,
            64,
            &game_config,
        ));
    }
    let t_fwd_64 = start.elapsed();
    let fwd_64_us = t_fwd_64.as_micros() as f64 / 1000.0;

    // ── Component F: forward decode at pos=127 (128 positions) ──
    game_cache.reset();
    for pos in 0..127 {
        forward(
            &mut game_ctx,
            &game_weights,
            &mut game_cache,
            0,
            pos,
            &game_config,
        );
    }
    for _ in 0..WARMUP {
        black_box(forward(
            &mut game_ctx,
            &game_weights,
            &mut game_cache,
            0,
            127,
            &game_config,
        ));
    }
    let start = Instant::now();
    for _ in 0..500 {
        black_box(forward(
            &mut game_ctx,
            &game_weights,
            &mut game_cache,
            0,
            127,
            &game_config,
        ));
    }
    let t_fwd_127 = start.elapsed();
    let fwd_127_us = t_fwd_127.as_micros() as f64 / 500.0;

    // ── Summary ──
    let rmsnorm_calls = 2 * config.n_layer;
    let rmsnorm_total = rmsnorm_us * rmsnorm_calls as f64;

    println!();
    println!("═══ Issue 057 Hot-Path Component Benchmark ═══");
    println!(
        "  rmsnorm/softmax: Config::micro() vocab={}, embd={}, heads={}, mlp={}, n_layer={}",
        config.vocab_size, config.n_embd, config.n_head, config.mlp_hidden, config.n_layer
    );
    println!(
        "  forward:        Config::game()  vocab={}, embd={}, heads={}, mlp={}, n_layer={}, block={}",
        game_config.vocab_size,
        game_config.n_embd,
        game_config.n_head,
        game_config.mlp_hidden,
        game_config.n_layer,
        game_config.block_size
    );
    println!("  SIMD level: {:?}", microgpt_rs::simd::simd_level());
    println!();
    println!("  Component                  | μs/call   | calls/token | μs/token");
    println!("  ---------------------------|-----------|-------------|----------");
    println!(
        "  rmsnorm (micro,embd=16)   | {rmsnorm_us:>9.3} | {rmsnorm_calls:>11} | {rmsnorm_total:>8.3}"
    );
    println!(
        "  softmax (micro,vocab=27)  | {softmax_us:>9.3} | {softmax_calls:>11} | {softmax_total:>8.3}",
        softmax_us = softmax_us,
        softmax_calls = 1,
        softmax_total = softmax_us
    );
    println!(
        "  softmax_scaled            | {softmax_scaled_us:>9.3} | {ssc:>11} | {sst:>8.3}",
        softmax_scaled_us = softmax_scaled_us,
        ssc = 1,
        sst = softmax_scaled_us
    );
    println!(
        "  forward game(pos=0,t_n=1) | {fwd_0_us:>9.3} | {fc:>11} | {fwd_0_us:>8.3}",
        fc = 1
    );
    println!(
        "  forward game(pos=64,t=65) | {fwd_64_us:>9.3} | {fc:>11} | {fwd_64_us:>8.3}",
        fc = 1
    );
    println!(
        "  forward game(pos=127,t=128)| {fwd_127_us:>8.3} | {fc:>11} | {fwd_127_us:>8.3}",
        fc = 1
    );
    println!();
    println!(
        "  Attention scaling: pos=0→64 is {:.1}×, pos=0→127 is {:.1}×",
        fwd_64_us / fwd_0_us,
        fwd_127_us / fwd_0_us
    );
    println!(
        "  rmsnorm fraction of game forward(pos=0): {:.1}%",
        rmsnorm_total / fwd_0_us * 100.0
    );
    println!();
}
