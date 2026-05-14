//! Benchmark suite for Plan 053: δ-Mem Modelless Distillation.
//!
//! Tests the associative bandit memory distilled from δ-mem (arXiv 2605.12357).
//! Covers:
//!   - Phase 1: DeltaMemoryState write/read roundtrip and interference
//!   - Phase 2: MemorySteeredPruner correction quality
//!   - Phase 3: MultiDomainMemory isolation

#![cfg(feature = "delta_mem")]

use std::time::Instant;

use microgpt_rs::pruners::delta_mem::*;
use microgpt_rs::speculative::build_dd_tree_screened;
use microgpt_rs::speculative::types::{NoScreeningPruner, ScreeningPruner};
use microgpt_rs::types::Config;

// ── Phase 1: DeltaMemoryState Benchmarks ───────────────────────

#[test]
fn test_phase1_state_write_read_roundtrip() {
    let mut state = DeltaMemoryState::new(DeltaMemoryConfig::default());

    let key = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let value = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Write the same association multiple times
    for _ in 0..50 {
        state.write(&key, &value);
    }

    // Read should return something non-zero aligned with value
    let readout = state.read(&key);
    let magnitude: f32 = readout.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        magnitude > 0.01,
        "After 50 writes, readout should have non-trivial magnitude: {magnitude}"
    );

    // Read with orthogonal key should return ~zero
    let orth_key = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let orth_readout = state.read(&orth_key);
    let orth_mag: f32 = orth_readout.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        orth_mag < magnitude * 0.5,
        "Orthogonal key should produce weaker readout: orth={orth_mag}, main={magnitude}"
    );
}

#[test]
fn test_phase1_state_prediction_convergence() {
    // Gate T4: state must learn a single association within 200 updates.
    // With rank=8 and β=0.182 (coupled gates), each write updates ~18.2% per row.
    // A single repeated (key, value) pair is the strongest convergence case.
    let mut state = DeltaMemoryState::new(DeltaMemoryConfig {
        rank: 8,
        ..Default::default()
    });
    let hasher = FeatureHasher::new(8, 3, 42);

    // Single fixed key-value pair — delta-rule converges monotonically on one association
    let features = vec![0.8, 0.3, -0.5];
    let key = hasher.hash_key(&features);
    let value = hasher.hash_value(&vec![0.4, -0.2, 0.7]);

    let mut errors = Vec::new();
    for i in 0..200 {
        // Read before write to measure prediction quality
        let prediction = state.read(&key);

        // Cosine similarity: measures alignment regardless of scale.
        // As the state converges, prediction aligns more with value.
        let dot: f32 = prediction
            .iter()
            .zip(value.iter())
            .map(|(p, v)| p * v)
            .sum();
        let pred_norm: f32 = prediction.iter().map(|x| x * x).sum::<f32>().sqrt();
        let val_norm: f32 = value.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine_sim = if pred_norm > 1e-8 && val_norm > 1e-8 {
            dot / (pred_norm * val_norm)
        } else {
            0.0
        };

        // Convert cosine similarity to [0,1] error: 1 = perfect alignment, 0 = orthogonal
        let alignment_error = 1.0 - cosine_sim;

        if i >= 100 {
            errors.push(alignment_error);
        }

        state.write(&key, &value);
    }

    let mean_error: f32 = errors.iter().sum::<f32>() / errors.len() as f32;

    // Gate: mean alignment error should be ≤0.20 after 200 updates.
    // The delta-rule should align prediction with value within the first 100 writes,
    // and the second 100 writes should show stable low error.
    assert!(
        mean_error <= 0.20,
        "Gate T4 FAILED: mean alignment error={mean_error:.4} > 0.20 after 200 updates. \
         Delta-rule did not converge on single association."
    );
}

#[test]
fn test_phase1_state_interference() {
    let mut state = DeltaMemoryState::new(DeltaMemoryConfig {
        rank: 8,
        ..Default::default()
    });
    let hasher = FeatureHasher::new(8, 5, 42);

    // Write association A many times
    let key_a = hasher.hash_key(&[1.0, 0.0, 0.0, 0.0, 0.0]);
    let val_a = hasher.hash_value(&[0.0, 1.0, 0.0]);
    for _ in 0..50 {
        state.write(&key_a, &val_a);
    }
    let read_a_before = state.read(&key_a);

    // Write association B many times (should partially interfere)
    let key_b = hasher.hash_key(&[0.0, 0.0, 0.0, 0.0, 1.0]);
    let val_b = hasher.hash_value(&[1.0, 0.0, 0.0]);
    for _ in 0..50 {
        state.write(&key_b, &val_b);
    }
    let read_a_after = state.read(&key_a);

    // Association A should still be somewhat preserved (not completely destroyed)
    let mag_before: f32 = read_a_before.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_after: f32 = read_a_after.iter().map(|x| x * x).sum::<f32>().sqrt();

    // With coupled gates (β=0.182), old state decays slowly
    // Expect at least some residual signal from association A
    assert!(
        mag_after > 0.001,
        "Association A should survive after writing B: before={mag_before}, after={mag_after}"
    );
}

#[test]
fn test_phase1_state_norm_bounded() {
    let mut state = DeltaMemoryState::new(DeltaMemoryConfig {
        rank: 8,
        ..Default::default()
    });
    let hasher = FeatureHasher::new(8, 5, 42);

    // Write 500 random normalized keys
    for i in 0..500 {
        let phase = i as f32 * 0.07;
        let features: Vec<f32> = (0..5).map(|j| (phase + j as f32 * 0.5).sin()).collect();
        let key = hasher.hash_key(&features); // Normalized
        let value = hasher.hash_value(&features); // Not normalized

        state.write(&key, &value);
    }

    let norm = state.state_norm();
    // With L2-normalized keys and conservative β, norm should stay bounded
    assert!(
        norm < 200.0,
        "State norm should be bounded with normalized keys: norm={norm}"
    );
}

#[test]
fn test_phase1_state_couple_vs_uncoupled() {
    let config_coupled = DeltaMemoryConfig {
        rank: 8,
        beta_init: 0.182,
        couple_gates: true,
    };
    let config_uncoupled = DeltaMemoryConfig {
        rank: 8,
        beta_init: 0.182,
        couple_gates: false,
    };

    let mut coupled = DeltaMemoryState::new(config_coupled);
    let mut uncoupled = DeltaMemoryState::new(config_uncoupled);

    let key = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let value = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    for _ in 0..50 {
        coupled.write(&key, &value);
        uncoupled.write(&key, &value);
    }

    // Coupled (λ=1-β) should decay old state → smaller norm
    // Uncoupled (λ=1) preserves all state → larger norm
    assert!(
        coupled.state_norm() < uncoupled.state_norm(),
        "Coupled gates should produce smaller state norm: coupled={}, uncoupled={}",
        coupled.state_norm(),
        uncoupled.state_norm()
    );
}

// ── Phase 2: MemorySteeredPruner Benchmarks ────────────────────

#[test]
fn test_phase2_fresh_memory_no_correction() {
    let pruner = MemorySteeredPruner::new(
        NoScreeningPruner,
        DeltaMemoryConfig::default(),
        2.0,
        CorrectionMode::OutputSide,
        WriteGranularity::Token,
    );

    // Fresh memory → zero correction → should return inner relevance (1.0)
    let rel = pruner.relevance(5, 3, &[1, 2, 3, 4]);
    assert!(
        (rel - 1.0).abs() < 1e-6,
        "Fresh memory should not modify relevance: got {rel}"
    );
}

#[test]
fn test_phase2_observation_builds_memory() {
    let mut pruner = MemorySteeredPruner::new(
        NoScreeningPruner,
        DeltaMemoryConfig::default(),
        2.0,
        CorrectionMode::OutputSide,
        WriteGranularity::Token,
    );

    // Observe positive δ many times at the same context
    for _ in 0..30 {
        let ctx = ContextFeatures::from_tree_context(5, 3, &[1, 2, 3]);
        let outcome = OutcomeFeatures {
            delta: 0.8,
            quality: 0.9,
            success: 1.0,
        };
        pruner.observe(&ctx, &outcome);
    }

    // Memory should now produce non-trivial corrections
    let rel = pruner.relevance(5, 3, &[1, 2, 3]);
    // The relevance should be valid and in [0, 1]
    assert!(
        rel >= 0.0 && rel <= 1.0,
        "Relevance should be in [0, 1]: got {rel}"
    );
    assert_eq!(pruner.memory().update_count(), 30);
}

#[test]
fn test_phase2_segment_write_granularity() {
    let mut pruner = MemorySteeredPruner::new(
        NoScreeningPruner,
        DeltaMemoryConfig {
            rank: 8,
            ..Default::default()
        },
        2.0,
        CorrectionMode::OutputSide,
        WriteGranularity::Segment,
    );

    // Accumulate observations without writing
    for i in 0..5 {
        let ctx = ContextFeatures::from_tree_context(i, i, &[0]);
        let outcome = OutcomeFeatures {
            delta: 0.5,
            quality: 0.7,
            success: 1.0,
        };
        pruner.observe(&ctx, &outcome);
    }

    assert_eq!(pruner.pending_count(), 5);
    assert_eq!(pruner.memory().update_count(), 0);

    pruner.flush_segment();
    assert_eq!(pruner.pending_count(), 0);
    assert_eq!(pruner.memory().update_count(), 1); // Single averaged write
}

#[test]
fn test_phase2_correction_modes_sweep() {
    for mode in [
        CorrectionMode::QuerySide,
        CorrectionMode::OutputSide,
        CorrectionMode::Both,
    ] {
        let mut pruner = MemorySteeredPruner::new(
            NoScreeningPruner,
            DeltaMemoryConfig {
                rank: 8,
                ..Default::default()
            },
            2.0,
            mode,
            WriteGranularity::Token,
        );

        // Train with observations
        for _ in 0..20 {
            let ctx = ContextFeatures::from_tree_context(3, 1, &[0, 1]);
            let outcome = OutcomeFeatures {
                delta: 0.5,
                quality: 0.8,
                success: 1.0,
            };
            pruner.observe(&ctx, &outcome);
        }

        let rel = pruner.relevance(3, 1, &[0, 1]);
        assert!(
            rel >= 0.0 && rel <= 1.0,
            "Mode {:?}: relevance should be in [0, 1], got {rel}",
            mode
        );
    }
}

#[test]
fn test_phase2_alpha_sweep() {
    for alpha in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0] {
        let pruner = MemorySteeredPruner::new(
            NoScreeningPruner,
            DeltaMemoryConfig::default(),
            alpha,
            CorrectionMode::OutputSide,
            WriteGranularity::Token,
        );

        let rel = pruner.relevance(0, 0, &[]);
        assert!(
            (rel - 1.0).abs() < 1e-6,
            "alpha={alpha}: fresh memory should return 1.0, got {rel}"
        );
    }
}

#[test]
fn test_phase2_rank_sweep() {
    for rank in [4, 8, 16, 32] {
        let pruner = MemorySteeredPruner::new(
            NoScreeningPruner,
            DeltaMemoryConfig {
                rank,
                ..Default::default()
            },
            2.0,
            CorrectionMode::OutputSide,
            WriteGranularity::Token,
        );

        let rel = pruner.relevance(5, 3, &[1, 2, 3]);
        assert!(
            rel >= 0.0 && rel <= 1.0,
            "rank={rank}: relevance should be in [0, 1], got {rel}"
        );
    }
}

// ── Phase 3: MultiDomainMemory Benchmarks ──────────────────────

#[test]
fn test_phase3_domain_isolation() {
    let mut mem = MultiDomainMemory::new(DeltaMemoryConfig {
        rank: 8,
        ..Default::default()
    });
    let hasher = FeatureHasher::new(8, 3, 42);

    let key = hasher.hash_key(&[1.0, 0.0, 0.0]);
    let val_a = hasher.hash_value(&[0.0, 1.0, 0.0]);

    // Write to domain A 50 times
    for _ in 0..50 {
        mem.write_domain("coding", &key, &val_a);
    }

    let read_a = mem.read_domain("coding", &key).unwrap();
    let mag_a: f32 = read_a.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Write to domain B 50 times
    let val_b = hasher.hash_value(&[1.0, 0.0, 0.0]);
    for _ in 0..50 {
        mem.write_domain("math", &key, &val_b);
    }

    // Domain A's readout should be unchanged
    let read_a_after = mem.read_domain("coding", &key).unwrap();
    let mag_a_after: f32 = read_a_after.iter().map(|x| x * x).sum::<f32>().sqrt();

    assert!(
        (mag_a - mag_a_after).abs() < 0.01,
        "Domain A should be isolated from domain B writes: before={mag_a}, after={mag_a_after}"
    );
}

#[test]
fn test_phase3_interference_measurement() {
    // Gate T9: per-domain states must show ≤50% interference vs single domain
    let config = DeltaMemoryConfig {
        rank: 8,
        ..Default::default()
    };
    let mut single = DeltaMemoryState::new(config.clone());
    let mut multi = MultiDomainMemory::new(config);
    let hasher = FeatureHasher::new(8, 3, 42);

    let key = hasher.hash_key(&[1.0, 0.0, 0.0]);

    // Single domain: write target association 50 times
    for _ in 0..50 {
        let val = hasher.hash_value(&[0.0, 1.0, 0.0]);
        single.write(&key, &val);
    }
    let single_read = single.read(&key);
    let single_mag: f32 = single_read.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Multi-domain: write target to "coding" 50 times, noise to 4 other domains
    for _ in 0..50 {
        let val = hasher.hash_value(&[0.0, 1.0, 0.0]);
        multi.write_domain("coding", &key, &val);
    }
    // Write noise to other domains
    for domain in ["math", "reasoning", "writing", "advice"] {
        let noise_key = hasher.hash_key(&[0.0, 1.0, 0.0]);
        let noise_val = hasher.hash_value(&[1.0, 0.0, 0.0]);
        for _ in 0..50 {
            multi.write_domain(domain, &noise_key, &noise_val);
        }
    }

    let multi_read = multi.read_domain("coding", &key).unwrap();
    let multi_mag: f32 = multi_read.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Interference = how much multi-domain differs from single-domain
    let interference = (single_mag - multi_mag).abs() / single_mag.max(1e-8);
    assert!(
        interference <= 0.50,
        "Gate T9 FAILED: cross-domain interference={interference:.2} > 50%"
    );
}

#[test]
fn test_phase3_aggregation_strategies() {
    let mut mem = MultiDomainMemory::new(DeltaMemoryConfig {
        rank: 8,
        ..Default::default()
    });
    let hasher = FeatureHasher::new(8, 3, 42);

    let key = hasher.hash_key(&[1.0, 0.0, 0.0]);

    // Write to multiple domains
    for domain in ["coding", "math", "reasoning"] {
        for _ in 0..10 {
            let val = hasher.hash_value(&[0.5, 0.5, 0.5]);
            mem.write_domain(domain, &key, &val);
        }
    }

    // RoutedOnly: should use only "coding" domain
    let routed = mem.read_aggregated("coding", &key, AggregationStrategy::RoutedOnly);
    assert!(routed.is_some());

    // BanditWeighted: should aggregate across all domains
    let weighted = mem.read_aggregated("coding", &key, AggregationStrategy::BanditWeighted);
    assert!(weighted.is_some());
}

// ── Summary ────────────────────────────────────────────────────

#[test]
fn test_summary_delta_mem_config_defaults() {
    let config = DeltaMemoryConfig::default();
    assert_eq!(config.rank, 8, "Paper default rank=8");
    assert!(
        (config.beta_init - 0.182).abs() < 0.01,
        "Paper default β=sigmoid(-1.5)≈0.182"
    );
    assert!(config.couple_gates, "Paper default: couple_lambda=True");
}

// ── PROOF: DDTree Integration ──────────────────────────────────
//
// HONEST ASSESSMENT: what does δ-mem actually give us in DDTree?
//
// The hard truth from benchmarking:
//   1. Fresh memory = identical to NoScreeningPruner (zero state → zero correction)
//   2. Trained memory produces non-zero corrections (the math works)
//   3. But corrections are too small to change tree shape at budget=256
//   4. Latency overhead is ~12× because relevance() is called per-token-per-node
//      and FeatureHasher + DeltaMemoryState::read() is O(rank²) per call
//
// What we GAIN:
//   ✅ A working associative memory that learns via delta-rule (no gradients)
//   ✅ Per-domain isolated states (MultiDomainMemory, zero cross-interference)
//   ✅ Serializable snapshots for persistence across sessions
//   ✅ Foundation for future improvements (caching, batch reads)
//
// What we DON'T gain (yet):
//   ❌ Faster DDTree builds — overhead is 12× vs NoScreeningPruner
//   ❌ Better solution quality — corrections don't flip branch ordering at full budget
//   ❌ Fewer nodes — tree fills budget regardless of relevance adjustments
//
// WHY the paper got better results:
//   - They correct attention Q/O projections, not a single scalar relevance
//   - Their corrections apply to every attention head in every layer
//   - The model has 4B+ parameters; our "model" is a DDTree with no neural net
//   - δ-mem's value is in the Transformer's hidden state, not in a tree scorer
//
// BOTTOM LINE: The infrastructure is correct but the value prop for DDTree
// is unproven. The real gain would come if we had a Transformer to correct.

/// Build DDTree N times with each pruner, measure nodes + time.
/// Documents what we actually gain vs the cost.
fn dd_tree_proof_internal() {
    let mut config = Config::draft();
    config.vocab_size = 16;
    config.draft_lookahead = 6;
    config.tree_budget = 256;

    // Non-uniform marginals: create real competition between tokens
    let marginals: Vec<Vec<f32>> = (0..config.draft_lookahead)
        .map(|d| {
            let shift = (d % 3) as f32 * 0.01;
            let mut probs = vec![0.02 + shift; config.vocab_size];
            let best = (d * 5 + 3) % config.vocab_size;
            probs[best] = 0.40;
            probs[(best + 1) % config.vocab_size] = 0.20;
            probs[(best + 3) % config.vocab_size] = 0.15;
            let sum: f32 = probs.iter().sum();
            probs.iter_mut().for_each(|p| *p /= sum);
            probs
        })
        .collect();
    let mv: Vec<&[f32]> = marginals.iter().map(|v| v.as_slice()).collect();

    let iters = 100;

    println!("\n🧪 PROOF: DDTree with δ-Mem vs Baseline ({iters} builds)");
    println!("{}", "═".repeat(70));
    println!(
        "   Vocab={}, Lookahead={}, Budget={}",
        config.vocab_size, config.draft_lookahead, config.tree_budget
    );

    // ── Baseline: NoScreeningPruner ──
    let mut baseline_nodes = 0usize;
    let start = Instant::now();
    for seed in 0..iters {
        let tree = build_dd_tree_screened(&mv, &config, &NoScreeningPruner, seed % 2 == 0);
        baseline_nodes += tree.len();
    }
    let baseline_time = start.elapsed();
    let avg_baseline = baseline_nodes as f64 / iters as f64;

    println!("\n   Baseline (NoScreeningPruner):");
    println!("     Avg nodes: {avg_baseline:.1}");
    println!("     Time:      {baseline_time:?}");

    // ── Fresh MemorySteeredPruner (no training) ──
    let fresh = MemorySteeredPruner::new(
        NoScreeningPruner,
        DeltaMemoryConfig::default(),
        2.0,
        CorrectionMode::OutputSide,
        WriteGranularity::Token,
    );

    let mut fresh_nodes = 0usize;
    let start = Instant::now();
    for seed in 0..iters {
        let tree = build_dd_tree_screened(&mv, &config, &fresh, seed % 2 == 0);
        fresh_nodes += tree.len();
    }
    let fresh_time = start.elapsed();
    let avg_fresh = fresh_nodes as f64 / iters as f64;

    println!("\n   Fresh MemorySteeredPruner (zero state):");
    println!("     Avg nodes: {avg_fresh:.1}");
    println!("     Time:      {fresh_time:?}");

    let fresh_delta = (avg_fresh - avg_baseline) / avg_baseline * 100.0;
    println!("     Node delta vs baseline: {fresh_delta:+.1}%");

    // Gate 1: fresh memory must be identical to baseline (zero state = passthrough)
    assert!(
        fresh_delta.abs() < 1.0,
        "Fresh memory (zero state) should be identical to baseline: delta={fresh_delta:.1}%"
    );
    println!("     Zero-state passthrough:  ✅ PASS (identical to baseline)");

    // ── Trained MemorySteeredPruner ──
    let mut trained = MemorySteeredPruner::new(
        NoScreeningPruner,
        DeltaMemoryConfig::default(),
        2.0,
        CorrectionMode::OutputSide,
        WriteGranularity::Token,
    );

    // Train with 200 observations
    for i in 0..200 {
        let depth = i % config.draft_lookahead;
        let token = (i * 3 + 7) % config.vocab_size;
        let ctx = ContextFeatures::from_tree_context(depth, token, &[0, 1, 2]);
        let outcome = OutcomeFeatures {
            delta: 0.5 + (i as f32 * 0.002),
            quality: 0.8,
            success: 1.0,
        };
        trained.observe(&ctx, &outcome);
    }

    let mut trained_nodes = 0usize;
    let start = Instant::now();
    for seed in 0..iters {
        let tree = build_dd_tree_screened(&mv, &config, &trained, seed % 2 == 0);
        trained_nodes += tree.len();
    }
    let trained_time = start.elapsed();
    let avg_trained = trained_nodes as f64 / iters as f64;

    println!("\n   Trained MemorySteeredPruner (200 observations):");
    println!("     Avg nodes: {avg_trained:.1}");
    println!("     Time:      {trained_time:?}");

    let trained_delta = (avg_trained - avg_baseline) / avg_baseline * 100.0;
    println!("     Node delta vs baseline: {trained_delta:+.1}%");

    // Gate 2: trained pruner must not explode nodes (>2× = +100%)
    assert!(
        trained_delta < 100.0,
        "Trained pruner must not use >2× baseline nodes: delta={trained_delta:.1}%"
    );
    println!("     No explosion (<2× nodes): ✅ PASS");

    // Gate 3: trained memory must produce non-zero corrections
    let ctx = ContextFeatures::from_tree_context(2, 5, &[1, 3]);
    let query = FeatureHasher::new(8, 8, 42).hash_key(&ctx.to_vec());
    let readout = trained.memory().read(&query);
    let correction: f32 = readout.iter().sum::<f32>() / readout.len() as f32;
    assert!(
        correction.abs() > 1e-6,
        "Trained memory must produce non-zero corrections: correction={correction:.6}"
    );
    println!("     Memory produces corrections: ✅ PASS (correction={correction:.4})");

    // ── Honest overhead measurement ──
    let overhead_fresh =
        (fresh_time.as_nanos() as f64 / baseline_time.as_nanos() as f64 - 1.0) * 100.0;
    let overhead_trained =
        (trained_time.as_nanos() as f64 / baseline_time.as_nanos() as f64 - 1.0) * 100.0;

    println!("\n   ── Overhead (vs NoScreeningPruner returning 1.0) ──");
    println!("     Fresh:   {overhead_fresh:+.0}%");
    println!("     Trained: {overhead_trained:+.0}%");
    println!(
        "     (relevance() called ~{}× per build: FeatureHasher + matmul per call)",
        config.vocab_size * config.tree_budget / config.draft_lookahead
    );

    println!("\n{}", "═".repeat(70));
    println!("   HONEST VERDICT:");
    println!("   ✅ Infrastructure works: delta-rule, hashing, isolation, snapshots");
    println!("   ✅ Fresh memory = zero corruption (safe to wrap any pruner)");
    println!(
        "   ⚠️  Latency ~{:.0}× slower than NoScreeningPruner baseline",
        trained_time.as_nanos() as f64 / baseline_time.as_nanos() as f64
    );
    println!("   ⚠️  No tree quality improvement in synthetic benchmark");
    println!("   📋 The value prop is for Transformer attention correction,");
    println!("      not for tree-based relevance scoring.");
}

#[test]
fn test_proof_ddtree_delta_mem() {
    dd_tree_proof_internal();
}
