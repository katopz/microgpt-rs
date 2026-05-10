//! Raven RSM (Routing Slot Memory) — Recall & Scaling Demo
//!
//! Demonstrates three key properties of the Raven architecture:
//!   1. Frozen-slot memory: write critical data, flood with noise, verify preservation
//!   2. O(1) per-step scaling: forward_raven time stays flat as sequence grows
//!   3. Memory footprint: fixed-size cache independent of sequence length
//!
//! Based on "Raven: High-Recall Sequence Modeling with Sparse Memory Routing"
//! See .research/06_Raven_Routing_Slot_Memories.md
//!
//! Run: `cargo run --example raven_recall`

use microgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, RavenKVCache, TransformerWeights, forward, forward_raven,
    raven_update,
};
use microgpt_rs::types::{Config, Rng};
use std::time::Instant;

/// Inline kv_dim calculation (n_kv_head × head_dim).
fn kvd(config: &Config) -> usize {
    config.n_kv_head * config.head_dim
}

fn main() {
    println!();
    println!("🦅 Raven RSM: Recall & Scaling Demo");
    println!("════════════════════════════════════════════════════════════");
    println!("  Paper: \"Raven: High-Recall Sequence Modeling with Sparse Memory Routing\"");
    println!();

    part1_frozen_slots();
    part2_scaling();
    part3_memory();

    println!("⚠️  Caveats");
    println!("─────────────────────────────────");
    println!("  • Router is dummy (cycles through K dims). Not learned.");
    println!("  • Model weights are random. Output quality is meaningless.");
    println!("  • Scaling numbers are real — O(N) scan vs O(S) constant.");
    println!("  • Next step: train router jointly with model (Plan 008).");
    println!();
    println!("✅ Done. See .research/06 for full analysis.");
}

// ── Part 1: Frozen-Slot Memory ────────────────────────────────

fn part1_frozen_slots() {
    println!("🧊 Part 1: Frozen-Slot Memory");
    println!("─────────────────────────────────");

    let config = Config::draft();
    let num_slots = 16;
    let top_k = 4;
    let dim = kvd(&config);
    let noise_steps = 10_000;

    println!("  Config: {num_slots} slots, kv_dim={dim}, top_k={top_k}");
    println!("  Plan:   Write passkey → {noise_steps} noise updates → Verify\n");

    let mut keys = vec![0.0f32; num_slots * dim];
    let mut values = vec![0.0f32; num_slots * dim];

    // 1. Write passkey to slot 12
    let passkey_slot = 12;
    let passkey_k = vec![1.0; dim];
    let passkey_v = vec![9.9; dim];
    let mut r_t = vec![0.0f32; num_slots];
    r_t[passkey_slot] = 1.0;

    raven_update(
        &mut keys,
        &mut values,
        &passkey_k,
        &passkey_v,
        &r_t,
        -1.0,
        num_slots,
        dim,
    );

    let after_write = values[passkey_slot * dim];
    // Gated blend with zero-initialized state: decay * 0.0 + (1-decay) * 9.9
    let decay = (-1.0f32).exp();
    let expected = (1.0 - decay) * 9.9;

    println!("  📝 Step 1: Write passkey (value=9.9) to slot {passkey_slot}");
    println!("     Stored value: {after_write:.4} (gated blend, expected ~{expected:.2})");
    println!("     (Not 9.9 because gated update blends with zero-initialized state)");

    // 2. Flood noise to slots 0-3
    let noise_k = vec![0.5; dim];
    let noise_v = vec![0.1; dim];
    let mut r_noise = vec![0.0f32; num_slots];
    r_noise[0] = 0.25;
    r_noise[1] = 0.25;
    r_noise[2] = 0.25;
    r_noise[3] = 0.25;

    let start = Instant::now();
    for _ in 0..noise_steps {
        raven_update(
            &mut keys,
            &mut values,
            &noise_k,
            &noise_v,
            &r_noise,
            -1.0,
            num_slots,
            dim,
        );
    }
    let elapsed = start.elapsed();

    // 3. Verify
    let after_noise = values[passkey_slot * dim];
    let preserved = (after_write - after_noise).abs() < 1e-6;

    println!();
    println!(
        "  🌪️  Step 2: {noise_steps} noise updates → slots 0-3 ({:.2?})",
        elapsed
    );

    println!();
    println!("  🔍 Step 3: Verify slot {passkey_slot}");
    println!("     Before noise: {after_write:.4}");
    println!("     After noise:  {after_noise:.4}");

    if preserved {
        println!();
        println!("     🎉 FROZEN SLOT VERIFIED");
        println!("        r_t[{passkey_slot}] = 0.0 for all noise updates");
        println!("        decay = exp(forget_rate × 0.0) = 1.0");
        println!("        H_new = 1.0 × H_old + 0.0 × new = H_old  (untouched)");
    }

    // Show noise slots were overwritten
    let noise_slot_val = values[0];
    println!("     Slot 0 (noise target): {noise_slot_val:.4} (heavily overwritten)");
    println!();
}

// ── Part 2: Per-Step Scaling O(N) vs O(1) ─────────────────────

fn part2_scaling() {
    println!("⚡ Part 2: Per-Step Scaling — Flat O(N) vs Raven O(1)");
    println!("─────────────────────────────────────────────────────────");

    let config = Config::bpe_draft(); // block_size=256, kv_dim=16
    let dim = kvd(&config);
    let num_slots = 64;
    let top_k = 16;
    let iters = 100;

    println!(
        "  Config: bpe_draft (block=256, embd={}, kv_dim={dim})",
        config.n_embd
    );
    println!("  Raven:  {num_slots} slots, top_k={top_k}");
    println!("  Method: Fill cache to pos P, measure time for step at pos P\n");

    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    let positions: [usize; 5] = [16, 32, 64, 128, 240];
    let mut flat_times: Vec<f64> = Vec::new();
    let mut raven_times: Vec<f64> = Vec::new();

    println!("  ┌──────────┬──────────────┬──────────────┬──────────┐");
    println!("  │ Position │ Flat (μs)    │ Raven (μs)   │ Speedup  │");
    println!("  ├──────────┼──────────────┼──────────────┼──────────┤");

    for &pos in &positions {
        // Flat: fill 0..pos, measure step at pos (scans pos+1 positions)
        let flat_us = {
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            for p in 0..pos {
                let _ = forward(&mut ctx, &weights, &mut cache, 0, p, &config);
            }
            let start = Instant::now();
            for _ in 0..iters {
                let _ = forward(&mut ctx, &weights, &mut cache, 0, pos, &config);
            }
            start.elapsed().as_micros() as f64 / iters as f64
        };

        // Raven: fill 0..pos, measure step at pos (always scans num_slots)
        let raven_us = {
            let mut ctx = ForwardContext::new(&config);
            let mut cache = RavenKVCache::new(&config, num_slots, top_k);
            for p in 0..pos {
                let _ = forward_raven(&mut ctx, &weights, &mut cache, 0, p, &config);
            }
            let start = Instant::now();
            for _ in 0..iters {
                let _ = forward_raven(&mut ctx, &weights, &mut cache, 0, pos, &config);
            }
            start.elapsed().as_micros() as f64 / iters as f64
        };

        flat_times.push(flat_us);
        raven_times.push(raven_us);

        let ratio = flat_us / raven_us;
        let emoji = if ratio >= 1.0 { "⚡" } else { "🐢" };

        println!("  │ {pos:>8} │ {flat_us:>10.1}  │ {raven_us:>10.1}  │ {ratio:>5.2}× {emoji} │");
    }

    println!("  └──────────┴──────────────┴──────────────┴──────────┘");

    // Show growth rate
    let flat_growth = flat_times.last().unwrap() / flat_times.first().unwrap();
    let raven_growth = raven_times.last().unwrap() / raven_times.first().unwrap();

    println!();
    println!(
        "  📊 Growth from pos {} → {}:",
        positions[0],
        positions[positions.len() - 1]
    );
    println!("     Flat:  {flat_growth:.1}× growth (attention scans more positions)");
    println!("     Raven: {raven_growth:.1}× growth (attention always scans {num_slots} slots)");
    println!();
    println!("  💡 Flat per-step cost = O(pos) — scans all previous positions");
    println!("     Raven per-step cost = O({num_slots}) — scans fixed slots, independent of pos");
    println!();
}

// ── Part 3: Memory Footprint ──────────────────────────────────

fn part3_memory() {
    println!("💾 Part 3: Memory Footprint per Layer");
    println!("────────────────────────────────────────");
    println!("  Cache = (key + value) × entries × sizeof(f32)");
    println!();

    let configs: &[(&str, Config, usize)] = &[
        ("draft", Config::draft(), 16),
        ("bpe_draft", Config::bpe_draft(), 64),
        ("small_target", Config::small_target(), 64),
    ];

    println!("  ┌────────────────┬──────────────┬──────────────┬──────────┐");
    println!("  │ Config         │ Flat Cache   │ Raven Cache  │ Savings  │");
    println!("  ├────────────────┼──────────────┼──────────────┼──────────┤");

    for &(name, ref config, slots) in configs {
        let dim = kvd(config);
        let flat_bytes = config.block_size * dim * 4 * 2;
        let raven_bytes = slots * dim * 4 * 2;
        let ratio = flat_bytes as f64 / raven_bytes as f64;

        println!(
            "  │ {:<14} │ {:>10}   │ {:>10}   │ {:>5.1}×    │",
            name,
            fmt_bytes(flat_bytes),
            fmt_bytes(raven_bytes),
            ratio
        );
    }

    println!("  └────────────────┴──────────────┴──────────────┴──────────┘");
    println!();
    println!("  💡 Flat cache grows with block_size. Raven cache is fixed.");
    println!("     At block_size=2048 (real LLMs): savings would be 32×.");
    println!();
}

fn fmt_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else {
        format!("{:.0} KB", bytes as f64 / 1024.0)
    }
}
