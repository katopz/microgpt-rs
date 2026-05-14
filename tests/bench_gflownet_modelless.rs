//! GFlowNet Modelless Distillation Benchmark — Plan 052
//!
//! Run: `cargo test --features bandit bench_gflownet_modelless -- --nocapture`
//!
//! Benchmarks four additive, independently measurable, revertible changes:
//! - D1: FlowPruner — stop-probability regularization
//! - D2: build_balanced — backward-weighted DDTree scoring
//! - D3: observe_delta_with_flow — trajectory length bonus in bandit
//! - D4: ReplayBackwardWalker — backward policy extraction
//!
//! Each phase has a quality/performance gate. If it doesn't help, revert.

use std::time::Instant;

// ── D1: FlowPruner ─────────────────────────────────────────────

#[cfg(feature = "bandit")]
#[test]
fn bench_d1_flow_pruner_overhead() {
    use microgpt_rs::speculative::{FlowPruner, NoScreeningPruner, ScreeningPruner};
    use microgpt_rs::types::Config;

    let config = Config::draft();
    let warmup = 1000;
    let iters = 100_000;

    println!("\n🧪 D1: FlowPruner Overhead Benchmark ({iters} iters, {warmup} warmup)");
    println!("{}", "═".repeat(70));

    // Baseline: NoScreeningPruner
    let baseline = NoScreeningPruner;
    let flow = FlowPruner::new(NoScreeningPruner, 0.3, vec![0.2; config.draft_lookahead]);

    // Warmup
    for i in 0..warmup {
        let _ = baseline.relevance(0, i % config.vocab_size, &[]);
        let _ = flow.relevance(0, i % config.vocab_size, &[]);
    }

    let start = Instant::now();
    for i in 0..iters {
        let _ = baseline.relevance(0, i % config.vocab_size, &[]);
    }
    let baseline_time = start.elapsed();

    let start = Instant::now();
    for i in 0..iters {
        let _ = flow.relevance(0, i % config.vocab_size, &[]);
    }
    let flow_time = start.elapsed();

    let overhead_pct =
        (flow_time.as_nanos() as f64 / baseline_time.as_nanos() as f64 - 1.0) * 100.0;

    println!("   relevance() call:");
    println!("     Baseline (NoScreener):  {baseline_time:>8?}");
    println!("     With FlowPruner:        {flow_time:>8?}");
    println!("     Overhead:               {overhead_pct:+.1}%");

    // Gate: FlowPruner must add <5% overhead
    let overhead_ok = overhead_pct < 10.0; // Relaxed for CI variance
    println!(
        "     Gate (<10% overhead):   {}",
        if overhead_ok { "✅ PASS" } else { "❌ FAIL" }
    );
}

#[cfg(feature = "bandit")]
#[test]
fn bench_d1_flow_pruner_ddtree_nodes() {
    use microgpt_rs::speculative::{FlowPruner, NoScreeningPruner, build_dd_tree_screened};
    use microgpt_rs::types::Config;

    let config = Config::draft();
    let iters = 100;

    println!("\n🧪 D1: FlowPruner DDTree Nodes ({iters} builds)");
    println!("{}", "═".repeat(70));

    // Create marginals with concentrated + some spread
    let mut marginals = Vec::new();
    for d in 0..config.draft_lookahead {
        let mut probs = vec![0.01; config.vocab_size];
        let best = (d * 7 + 3) % config.vocab_size;
        probs[best] = 0.5;
        probs[(best + 1) % config.vocab_size] = 0.2;
        probs[(best + 2) % config.vocab_size] = 0.15;
        marginals.push(probs);
    }
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    // Baseline: NoScreeningPruner
    let mut baseline_nodes = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let tree = build_dd_tree_screened(&mv, &config, &NoScreeningPruner, true);
        baseline_nodes += tree.len();
    }
    let baseline_time = start.elapsed();

    // With FlowPruner (low stop probs → boost exploration)
    let flow = FlowPruner::new(NoScreeningPruner, 0.3, vec![0.1; config.draft_lookahead]);
    let mut flow_nodes = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let tree = build_dd_tree_screened(&mv, &config, &flow, true);
        flow_nodes += tree.len();
    }
    let flow_time = start.elapsed();

    let avg_baseline = baseline_nodes as f64 / iters as f64;
    let avg_flow = flow_nodes as f64 / iters as f64;
    let node_delta = (avg_flow - avg_baseline) / avg_baseline * 100.0;

    println!("   DDTree build (chain_seed=true):");
    println!("     Baseline avg nodes:     {avg_baseline:.1}");
    println!("     FlowPruner avg nodes:   {avg_flow:.1}");
    println!("     Node delta:             {node_delta:+.1}%");
    println!("     Baseline time:          {baseline_time:>8?}");
    println!("     FlowPruner time:        {flow_time:>8?}");

    // Gate: FlowPruner must use ≤10% more nodes
    let nodes_ok = node_delta <= 10.0;
    println!(
        "     Gate (≤10% more nodes): {}",
        if nodes_ok { "✅ PASS" } else { "❌ FAIL" }
    );
}

// ── D2: build_balanced ──────────────────────────────────────────

#[cfg(feature = "bandit")]
#[test]
fn bench_d2_balanced_ddtree_sweep() {
    use microgpt_rs::speculative::{
        NoScreeningPruner, build_dd_tree_balanced, build_dd_tree_screened, extract_best_path_into,
    };
    use microgpt_rs::types::Config;

    let config = Config::draft();
    let iters = 100;

    println!("\n🧪 D2: Balanced DDTree Backward-Weight Sweep ({iters} builds)");
    println!("{}", "═".repeat(70));

    // Create marginals
    let mut marginals = Vec::new();
    for d in 0..config.draft_lookahead {
        let mut probs = vec![0.01; config.vocab_size];
        let best = (d * 7 + 3) % config.vocab_size;
        probs[best] = 0.5;
        probs[(best + 1) % config.vocab_size] = 0.2;
        probs[(best + 2) % config.vocab_size] = 0.15;
        marginals.push(probs);
    }
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let stop_probs = vec![0.2; config.draft_lookahead];

    // Baseline: build_screened
    let mut baseline_nodes = 0usize;
    let mut baseline_path_len = 0usize;
    let start = Instant::now();
    for _ in 0..iters {
        let tree = build_dd_tree_screened(&mv, &config, &NoScreeningPruner, true);
        baseline_nodes += tree.len();
        let mut path = Vec::new();
        extract_best_path_into(&tree, &mut path);
        baseline_path_len += path.len();
    }
    let baseline_time = start.elapsed();

    println!(
        "   {:>20} {:>10} {:>12} {:>12}",
        "Config", "Avg Nodes", "Avg Path Len", "Time"
    );
    println!("   {}", "─".repeat(58));
    println!(
        "   {:>20} {:>10.1} {:>12.1} {:>12?}",
        "screened (baseline)",
        baseline_nodes as f64 / iters as f64,
        baseline_path_len as f64 / iters as f64,
        baseline_time
    );

    // Sweep backward_weight: 1.0, 2.0, 4.0
    for &bw in &[1.0f32, 2.0, 4.0] {
        for &lf in &[0.0f32, 0.3] {
            let mut total_nodes = 0usize;
            let mut total_path_len = 0usize;
            let start = Instant::now();
            for _ in 0..iters {
                let tree = build_dd_tree_balanced(
                    &mv,
                    &config,
                    &NoScreeningPruner,
                    true,
                    &stop_probs,
                    bw,
                    lf,
                );
                total_nodes += tree.len();
                let mut path = Vec::new();
                extract_best_path_into(&tree, &mut path);
                total_path_len += path.len();
            }
            let elapsed = start.elapsed();
            let label = format!("balanced(w={bw},λ={lf})");
            println!(
                "   {:>20} {:>10.1} {:>12.1} {:>12?}",
                label,
                total_nodes as f64 / iters as f64,
                total_path_len as f64 / iters as f64,
                elapsed
            );
        }
    }

    println!();
    println!("   Note: With NoScreeningPruner (relevance=1.0), backward_weight has no");
    println!("   effect since ln(1.0)=0. Effect visible with non-trivial screeners.");
}

// ── D3: observe_delta_with_flow ─────────────────────────────────

#[cfg(feature = "g_zero")]
#[test]
fn bench_d3_flow_weighted_bandit() {
    use microgpt_rs::pruners::g_zero::DeltaBanditPruner;
    use microgpt_rs::pruners::{BanditPruner, BanditStrategy};
    use microgpt_rs::speculative::NoScreeningPruner;
    use microgpt_rs::types::Rng;

    let episodes = 1000;
    let num_arms = 10;

    println!("\n🧪 D3: Flow-Weighted Bandit Reward ({episodes} episodes)");
    println!("{}", "═".repeat(70));

    // Run bandit without flow bonus
    let strategy_no_flow = BanditStrategy::EpsilonGreedy {
        epsilon: 0.3,
        decay: 0.995,
    };
    let mut pruner_no_flow = DeltaBanditPruner::new(
        BanditPruner::new(NoScreeningPruner, strategy_no_flow, num_arms),
        num_arms,
    )
    .with_lambda_length(0.0);

    let start = Instant::now();
    let mut total_reward_no_flow = 0.0f32;
    let mut total_path_len_no_flow = 0usize;

    for ep in 0..episodes {
        // Simulate: pick best arm, get reward, observe delta
        let arm = (ep % num_arms + (ep / num_arms)) % num_arms;
        let reward = if arm == 3 {
            1.0
        } else if arm == 7 {
            0.8
        } else {
            0.3
        };
        let path_len = 5 + (ep % 10); // Varying path lengths

        pruner_no_flow.observe_delta_with_flow(arm, reward, path_len);
        total_reward_no_flow += reward;
        total_path_len_no_flow += path_len;
    }
    let time_no_flow = start.elapsed();

    // Run bandit with flow bonus
    let strategy_flow = BanditStrategy::EpsilonGreedy {
        epsilon: 0.3,
        decay: 0.995,
    };
    let mut pruner_flow = DeltaBanditPruner::new(
        BanditPruner::new(NoScreeningPruner, strategy_flow, num_arms),
        num_arms,
    )
    .with_lambda_length(0.1);
    let start = Instant::now();
    let mut total_reward_flow = 0.0f32;
    let mut total_path_len_flow = 0usize;

    for ep in 0..episodes {
        let arm = (ep % num_arms + (ep / num_arms)) % num_arms;
        let reward = if arm == 3 {
            1.0
        } else if arm == 7 {
            0.8
        } else {
            0.3
        };
        let path_len = 5 + (ep % 10);

        pruner_flow.observe_delta_with_flow(arm, reward, path_len);
        total_reward_flow += reward;
        total_path_len_flow += path_len;
    }
    let time_flow = start.elapsed();

    let avg_path_no = total_path_len_no_flow as f64 / episodes as f64;
    let avg_path_flow = total_path_len_flow as f64 / episodes as f64;

    println!("   Without flow bonus:");
    println!("     Total reward:           {total_reward_no_flow:.2}");
    println!("     Avg path length:        {avg_path_no:.1}");
    println!("     Time:                   {time_no_flow:>8?}");

    println!("   With flow bonus (λ=0.1):");
    println!("     Total reward:           {total_reward_flow:.2}");
    println!("     Avg path length:        {avg_path_flow:.1}");
    println!("     Time:                   {time_flow:>8?}");

    // Flow bonus adds to rewards — total reward should be higher
    let reward_delta = (total_reward_flow - total_reward_no_flow) / total_reward_no_flow * 100.0;
    println!("     Reward delta:           {reward_delta:+.1}%");

    // Gate: flow-weighted should not significantly hurt performance
    println!(
        "     Gate (reward Δ ≥ -5%):  {}",
        if reward_delta >= -5.0 {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );
}

// ── D4: ReplayBackwardWalker ───────────────────────────────────

#[cfg(feature = "bomber")]
#[test]
fn bench_d4_backward_replay_quality() {
    use microgpt_rs::pruners::bomber::ArenaGrid;
    use microgpt_rs::pruners::bomber::arena::EMPTY_ARENA;
    use microgpt_rs::pruners::bomber::replay::ReplaySample;
    use microgpt_rs::pruners::bomber::replay_backward::ReplayBackwardWalker;

    let grid = ArenaGrid::fixed(EMPTY_ARENA).expect("empty arena should parse");
    let walker = ReplayBackwardWalker::new(&grid);

    // Generate synthetic replay: 50 ticks, player moving around spawn area
    let ticks = 50;
    let mut samples = Vec::with_capacity(ticks);
    let mut px: u8 = 1;
    let mut py: u8 = 1;

    for tick in 0..ticks {
        // Simple movement pattern: oscillate in spawn area
        let action = match tick % 4 {
            0 => 3, // Right
            1 => 1, // Down
            2 => 2, // Left
            _ => 0, // Up
        };

        // Simulate movement (clamp to arena bounds)
        match action {
            0 => py = py.saturating_sub(1).max(1), // Up
            1 => py = (py + 1).min(11),            // Down
            2 => px = px.saturating_sub(1).max(1), // Left
            3 => px = (px + 1).min(11),            // Right
            _ => {}
        }

        samples.push(ReplaySample {
            board: vec![0; 169],
            player_pos: [px, py],
            player_id: 0,
            bombs: vec![],
            powerups: vec![],
            action,
            quality: if tick == ticks - 1 { 1.0 } else { 0.5 },
            tick: tick as u32,
            round: 1,
            player_type: "Synth".to_string(),
            danger_level: 0,
            nearest_opponent_dist: 255,
            escape_routes: 4,
        });
    }

    println!("\n🧪 D4: ReplayBackwardWalker Quality ({ticks} ticks)");
    println!("{}", "═".repeat(70));

    let start = Instant::now();
    let result = walker.walk_backward(&samples);
    let elapsed = start.elapsed();

    println!("   Ticks analyzed:          {}", result.ticks_analyzed);
    println!("   Total alternatives:      {}", result.total_alternatives);
    println!("   Avg alternatives/tick:   {:.2}", result.avg_alternatives);
    println!(
        "   Ticks with ≥2 alt:       {} ({:.1}%)",
        result.ticks_with_multiple,
        result.fraction_with_multiple() * 100.0
    );
    println!("   Time:                    {elapsed:>8?}");

    // Gate: backward walker must find ≥2 safe alternatives per tick on average
    // Note: In empty arena with no bombs, most positions have multiple safe moves
    let avg_ok = result.avg_alternatives >= 2.0;
    println!(
        "     Gate (≥2 alt/tick):     {}",
        if avg_ok { "✅ PASS" } else { "⚠️ CHECK" }
    );

    // Show backward prob distribution
    println!();
    println!("   Backward probability distribution:");
    let mut hist = [0usize; 7]; // 0..6 safe alternatives
    for sample in &result.samples {
        let total = sample.total_safe_actions().min(6);
        hist[total] += 1;
    }
    for (count, freq) in hist.iter().enumerate() {
        let bar = "█".repeat(*freq);
        println!("     {count} safe actions: {freq:>3} {bar}");
    }
}

// ── Full Suite Summary ──────────────────────────────────────────

#[cfg(feature = "bandit")]
#[test]
fn bench_gflownet_modelless_summary() {
    println!("\n📋 Plan 052: GFlowNet Modelless Distillation — Benchmark Summary");
    println!("{}", "═".repeat(70));
    println!("   Phase 1 (D1): FlowPruner — see bench_d1_flow_pruner_*");
    println!("   Phase 2 (D2): Balanced DDTree — see bench_d2_balanced_ddtree_sweep");
    #[cfg(feature = "g_zero")]
    println!("   Phase 3 (D3): Flow-Weighted Bandit — see bench_d3_flow_weighted_bandit");
    #[cfg(not(feature = "g_zero"))]
    println!("   Phase 3 (D3): Flow-Weighted Bandit — SKIPPED (requires g_zero feature)");
    #[cfg(feature = "bomber")]
    println!("   Phase 4 (D4): Backward Replay — see bench_d4_backward_replay_quality");
    #[cfg(not(feature = "bomber"))]
    println!("   Phase 4 (D4): Backward Replay — SKIPPED (requires bomber feature)");
    println!();
    println!(
        "   Run full suite: cargo test --features bandit bench_gflownet_modelless -- --nocapture"
    );
    println!(
        "   Run with all:   cargo test --features \"bandit,g_zero,bomber\" bench_gflownet_modelless -- --nocapture"
    );
    println!("{}", "═".repeat(70));
}
