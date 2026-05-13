//! PPoT profiling test — run with: cargo test --features ppot ppot_bench -- --nocapture
//!
//! Breaks down Plan 027 adaptive rescue overhead into per-component timings
//! to identify optimization targets.

#[cfg(feature = "ppot")]
#[test]
fn ppot_profile_components() {
    use microgpt_rs::speculative::ppot::{
        PpotConfig, RejectionInsight, SessionKnowledge, TokenRule, identify_high_entropy_positions,
        identify_positions_adaptive, ppot_resample, ppot_resample_different_value,
        ppot_resample_multi_strategy, ppot_rescue, ppot_rescue_adaptive, rank_by_consistency,
        token_entropy,
    };
    use microgpt_rs::speculative::{
        NoScreeningPruner, SpeculativeContext, dflash_predict_with, sample_from_distribution,
    };
    use microgpt_rs::transformer::TransformerWeights;
    use microgpt_rs::types::{Config, Rng};
    use std::time::Instant;

    let draft_config = Config::draft();
    let mut draft_rng = Rng::new(99);
    let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);
    let vocab_size = draft_config.vocab_size;
    let warmup = 100;
    let iters = 10000;

    println!(
        "\n🧪 PPoT Component Profile ({} iters, {} warmup)",
        iters, warmup
    );
    println!("{}", "═".repeat(70));

    // ── Setup: produce marginals + base path ──────────────────────
    let mut sctx = SpeculativeContext::new(&draft_config);
    sctx.reset();
    let steps = dflash_predict_with(&mut sctx, &draft_weights, &draft_config, 0, 0);
    let marginals: Vec<&[f32]> = (0..steps)
        .map(|step| sctx.marginal_slice(step, vocab_size))
        .collect();
    let base_path: Vec<usize> = marginals
        .iter()
        .map(|m| {
            m.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect();
    let positions_entropy = identify_high_entropy_positions(&marginals, 0.5);
    let positions = if positions_entropy.is_empty() {
        vec![0]
    } else {
        positions_entropy
    };

    println!(
        "  Setup: {} steps, {} high-H positions, vocab_size={}",
        steps,
        positions.len(),
        vocab_size
    );
    println!("{}", "─".repeat(70));

    // ── 1. Entropy calculation ────────────────────────────────────
    let mut rng = Rng::new(99);
    for _ in 0..warmup {
        for m in &marginals {
            std::hint::black_box(token_entropy(m));
        }
    }
    let mut total_calls = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        for m in &marginals {
            std::hint::black_box(token_entropy(m));
            total_calls += 1;
        }
    }
    let t_entropy = start.elapsed();
    println!(
        "  1. Entropy H(i):          {:>8.2} μs/rescue  ({:.3} μs/call, {} steps)",
        t_entropy.as_micros() as f64 / iters as f64,
        t_entropy.as_micros() as f64 / total_calls as f64,
        steps,
    );

    // ── 2. Identify positions (entropy-only) ──────────────────────
    for _ in 0..warmup {
        std::hint::black_box(identify_high_entropy_positions(&marginals, 0.5));
    }
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(identify_high_entropy_positions(&marginals, 0.5));
    }
    let t_identify_entropy = start.elapsed();
    println!(
        "  2a. Identify positions (entropy):  {:>8.2} μs/rescue",
        t_identify_entropy.as_micros() as f64 / iters as f64,
    );

    // ── 3. Identify positions (adaptive with knowledge) ───────────
    let mut knowledge = SessionKnowledge::from_config(&PpotConfig::for_char_level());
    // Pre-fill knowledge to simulate warm state
    for _ in 0..50 {
        let variant = ppot_resample(&base_path, &marginals, &positions, &mut rng);
        for &pos in &positions {
            if pos < variant.len() {
                knowledge.record(RejectionInsight {
                    position: pos,
                    rule: TokenRule::Digit,
                    original_token: base_path[pos],
                    attempted_token: variant[pos],
                    error_kind: None,
                    entropy: 0.5,
                    accepted: rng.next().is_multiple_of(3),
                });
            }
        }
    }

    for _ in 0..warmup {
        std::hint::black_box(identify_positions_adaptive(
            &marginals,
            0.5,
            Some(&knowledge),
        ));
    }
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(identify_positions_adaptive(
            &marginals,
            0.5,
            Some(&knowledge),
        ));
    }
    let t_identify_adaptive = start.elapsed();
    println!(
        "  2b. Identify positions (adaptive): {:>8.2} μs/rescue  (Δ{:+.1}% vs entropy)",
        t_identify_adaptive.as_micros() as f64 / iters as f64,
        (t_identify_adaptive.as_secs_f64() / t_identify_entropy.as_secs_f64() - 1.0) * 100.0,
    );

    // ── 4. Knowledge queries ──────────────────────────────────────
    // position_affinity
    let start = Instant::now();
    for _ in 0..iters {
        for &pos in &positions {
            std::hint::black_box(knowledge.position_affinity(pos));
        }
    }
    let t_affinity = start.elapsed();

    // should_skip_position
    let start = Instant::now();
    for _ in 0..iters {
        for &pos in &positions {
            std::hint::black_box(knowledge.should_skip_position(pos));
        }
    }
    let t_skip = start.elapsed();

    // preferred_rules
    let start = Instant::now();
    for _ in 0..iters {
        for &pos in &positions {
            std::hint::black_box(knowledge.preferred_rules(pos));
        }
    }
    let t_preferred = start.elapsed();

    // adaptive_threshold
    let ppot_config = PpotConfig::for_char_level().with_cached_support(vocab_size);
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(knowledge.adaptive_threshold(&ppot_config));
    }
    let t_threshold = start.elapsed();

    println!(
        "  3. Knowledge queries (per rescue, {} positions):",
        positions.len()
    );
    println!(
        "     position_affinity:     {:>8.2} μs",
        t_affinity.as_micros() as f64 / iters as f64
    );
    println!(
        "     should_skip_position:  {:>8.2} μs",
        t_skip.as_micros() as f64 / iters as f64
    );
    println!(
        "     preferred_rules:       {:>8.2} μs",
        t_preferred.as_micros() as f64 / iters as f64
    );
    println!(
        "     adaptive_threshold:    {:>8.2} μs",
        t_threshold.as_micros() as f64 / iters as f64
    );

    // ── 5. Resample variants ──────────────────────────────────────
    let mut rng = Rng::new(99);
    let mut scratch = vec![0.0f32; vocab_size];

    // Basic
    for _ in 0..warmup {
        for _ in 0..10 {
            ppot_resample(&base_path, &marginals, &positions, &mut rng);
        }
    }
    let mut total = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        for _ in 0..10 {
            ppot_resample(&base_path, &marginals, &positions, &mut rng);
            total += 1;
        }
    }
    let t_resample_basic = start.elapsed();
    println!(
        "  4a. Resample basic (×10):         {:>8.2} μs  ({:.3} μs/sample)",
        t_resample_basic.as_micros() as f64 / iters as f64,
        t_resample_basic.as_micros() as f64 / total as f64,
    );

    // Different-value
    let mut rng = Rng::new(99);
    for _ in 0..warmup {
        for _ in 0..10 {
            ppot_resample_different_value(
                &base_path,
                &marginals,
                &positions,
                &mut scratch,
                &mut rng,
            );
        }
    }
    total = 0;
    let start = Instant::now();
    for _ in 0..iters {
        for _ in 0..10 {
            ppot_resample_different_value(
                &base_path,
                &marginals,
                &positions,
                &mut scratch,
                &mut rng,
            );
            total += 1;
        }
    }
    let t_resample_diff = start.elapsed();
    println!(
        "  4b. Resample diff-value (×10):    {:>8.2} μs  ({:.3} μs/sample)",
        t_resample_diff.as_micros() as f64 / iters as f64,
        t_resample_diff.as_micros() as f64 / total as f64,
    );

    // Multi-strategy (Plan 027's actual variant generator)
    let mut rng = Rng::new(99);
    for _ in 0..warmup {
        ppot_resample_multi_strategy(
            &base_path,
            &marginals,
            &positions,
            10,
            &[],
            &ppot_config,
            &mut scratch,
            &mut rng,
        );
    }
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(ppot_resample_multi_strategy(
            &base_path,
            &marginals,
            &positions,
            10,
            &[],
            &ppot_config,
            &mut scratch,
            &mut rng,
        ));
    }
    let t_multi_strategy = start.elapsed();
    println!(
        "  4c. Multi-strategy (×10):         {:>8.2} μs",
        t_multi_strategy.as_micros() as f64 / iters as f64,
    );

    // ── 6. Knowledge recording ────────────────────────────────────
    let mut rng = Rng::new(99);
    // Simulate the recording loop: 10 samples × N positions
    for _ in 0..warmup {
        for _ in 0..10 {
            let variant = ppot_resample(&base_path, &marginals, &positions, &mut rng);
            for &pos in &positions {
                if pos < variant.len() {
                    knowledge.record(RejectionInsight {
                        position: pos,
                        rule: TokenRule::All,
                        original_token: base_path[pos],
                        attempted_token: variant[pos],
                        error_kind: None,
                        entropy: 0.5,
                        accepted: true,
                    });
                }
            }
        }
    }
    knowledge = SessionKnowledge::from_config(&ppot_config);
    let start = Instant::now();
    for _ in 0..iters {
        for sample_idx in 0..10usize {
            let variant = ppot_resample(&base_path, &marginals, &positions, &mut rng);
            let rule = TokenRule::STRATEGIES[sample_idx % 5];
            for &pos in &positions {
                if pos < variant.len() {
                    knowledge.record(RejectionInsight {
                        position: pos,
                        rule,
                        original_token: base_path[pos],
                        attempted_token: variant[pos],
                        error_kind: None,
                        entropy: 0.5,
                        accepted: sample_idx % 3 == 0,
                    });
                }
            }
        }
    }
    let t_record = start.elapsed();
    println!(
        "  5. Knowledge record (×10×{}pos):  {:>8.2} μs",
        positions.len(),
        t_record.as_micros() as f64 / iters as f64,
    );

    // ── 7. Self-consistency ranking ───────────────────────────────
    let mut rng = Rng::new(99);
    let variants: Vec<Vec<usize>> = (0..10)
        .map(|_| ppot_resample(&base_path, &marginals, &positions, &mut rng))
        .collect();
    for _ in 0..warmup {
        std::hint::black_box(rank_by_consistency(&variants));
    }
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(rank_by_consistency(&variants));
    }
    let t_rank = start.elapsed();
    println!(
        "  6. Rank by consistency (10 vars): {:>8.2} μs",
        t_rank.as_micros() as f64 / iters as f64,
    );

    // ── 8. Full end-to-end comparison ─────────────────────────────
    println!("{}", "─".repeat(70));
    println!("  Full rescue comparison ({} iters):", iters);

    // Greedy baseline
    let mut rng = Rng::new(99);
    for _ in 0..warmup {
        sctx.reset();
        dflash_predict_with(&mut sctx, &draft_weights, &draft_config, 0, 0);
    }
    let start = Instant::now();
    for _ in 0..iters {
        sctx.reset();
        let steps_now = dflash_predict_with(&mut sctx, &draft_weights, &draft_config, 0, 0);
        let m = sctx.marginal_slice(0, vocab_size);
        if !m.is_empty() {
            sample_from_distribution(m, &mut rng);
        }
        std::hint::black_box(steps_now);
    }
    let t_greedy = start.elapsed();
    println!(
        "    Greedy:               {:>8.1} μs/step",
        t_greedy.as_micros() as f64 / iters as f64,
    );

    // PPoT rescue (Plan 026)
    let mut rng = Rng::new(99);
    let mut scratch = vec![0.0f32; vocab_size];
    for _ in 0..warmup {
        sctx.reset();
        dflash_predict_with(&mut sctx, &draft_weights, &draft_config, 0, 0);
        let mv: Vec<&[f32]> = (0..sctx.steps_populated)
            .map(|s| sctx.marginal_slice(s, vocab_size))
            .collect();
        let bp: Vec<usize> = mv
            .iter()
            .map(|m| {
                m.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect();
        ppot_rescue(
            &mv,
            &bp,
            &NoScreeningPruner,
            &ppot_config,
            &mut scratch,
            &mut rng,
        );
    }
    let mut rescued_026 = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        sctx.reset();
        dflash_predict_with(&mut sctx, &draft_weights, &draft_config, 0, 0);
        let mv: Vec<&[f32]> = (0..sctx.steps_populated)
            .map(|s| sctx.marginal_slice(s, vocab_size))
            .collect();
        let bp: Vec<usize> = mv
            .iter()
            .map(|m| {
                m.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect();
        if ppot_rescue(
            &mv,
            &bp,
            &NoScreeningPruner,
            &ppot_config,
            &mut scratch,
            &mut rng,
        )
        .is_some()
        {
            rescued_026 += 1;
        }
    }
    let t_026 = start.elapsed();
    println!(
        "    PPoT 026 (random):    {:>8.1} μs/step  (Δ{:+.1}%, rescued {}/{})",
        t_026.as_micros() as f64 / iters as f64,
        (t_026.as_secs_f64() / t_greedy.as_secs_f64() - 1.0) * 100.0,
        rescued_026,
        iters,
    );

    // PPoT adaptive rescue (Plan 027)
    let mut rng = Rng::new(99);
    let mut knowledge = SessionKnowledge::from_config(&ppot_config);
    for _ in 0..warmup {
        sctx.reset();
        dflash_predict_with(&mut sctx, &draft_weights, &draft_config, 0, 0);
        let mv: Vec<&[f32]> = (0..sctx.steps_populated)
            .map(|s| sctx.marginal_slice(s, vocab_size))
            .collect();
        let bp: Vec<usize> = mv
            .iter()
            .map(|m| {
                m.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect();
        ppot_rescue_adaptive(
            &mv,
            &bp,
            &NoScreeningPruner,
            &ppot_config,
            &mut knowledge,
            &mut scratch,
            &mut rng,
        );
    }
    knowledge = SessionKnowledge::from_config(&ppot_config);
    let mut rescued_027 = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        sctx.reset();
        dflash_predict_with(&mut sctx, &draft_weights, &draft_config, 0, 0);
        let mv: Vec<&[f32]> = (0..sctx.steps_populated)
            .map(|s| sctx.marginal_slice(s, vocab_size))
            .collect();
        let bp: Vec<usize> = mv
            .iter()
            .map(|m| {
                m.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect();
        if ppot_rescue_adaptive(
            &mv,
            &bp,
            &NoScreeningPruner,
            &ppot_config,
            &mut knowledge,
            &mut scratch,
            &mut rng,
        )
        .is_some()
        {
            rescued_027 += 1;
        }
    }
    let t_027 = start.elapsed();
    println!(
        "    PPoT 027 (adaptive):  {:>8.1} μs/step  (Δ{:+.1}%, rescued {}/{})",
        t_027.as_micros() as f64 / iters as f64,
        (t_027.as_secs_f64() / t_greedy.as_secs_f64() - 1.0) * 100.0,
        rescued_027,
        iters,
    );

    // ── Summary ───────────────────────────────────────────────────
    let overhead_027 = t_027 - t_greedy;
    println!("{}", "─".repeat(70));
    println!(
        "  Plan 027 overhead breakdown (total Δ{:.1} μs):",
        overhead_027.as_micros() as f64 / iters as f64,
    );
    println!(
        "    Multi-strategy gen:     ~{:.1} μs  (4c)",
        t_multi_strategy.as_micros() as f64 / iters as f64,
    );
    println!(
        "    Knowledge record:       ~{:.1} μs  (5, includes resample cost)",
        t_record.as_micros() as f64 / iters as f64,
    );
    println!(
        "    Identify adaptive:      ~{:.1} μs  (2b)",
        t_identify_adaptive.as_micros() as f64 / iters as f64,
    );
    println!(
        "    Knowledge queries:      ~{:.1} μs  (3 affinity+skip+preferred+threshold)",
        (t_affinity + t_skip + t_preferred + t_threshold).as_micros() as f64 / iters as f64,
    );
    println!(
        "    Rank consistency:       ~{:.1} μs  (6)",
        t_rank.as_micros() as f64 / iters as f64,
    );
    println!(
        "    Knowledge accumulated:  {} insights",
        knowledge.insight_count(),
    );
    println!("{}", "═".repeat(70));
}
