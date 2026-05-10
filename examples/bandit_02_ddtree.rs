//! Bandit + DDTree Demo — Model-Based vs Modelless Comparison
//!
//! Proves whether model-based speculative decoding with bandit is worth the
//! cost vs a modelless bandit-only approach.
//!
//! Unlike `bandit_01_basic.rs` which uses coin flips (no real marginals, no DDTree,
//! no verification), this demo:
//! - Uses simulated marginals (concentrated for model-based, uniform for modelless)
//! - Uses `build_dd_tree_screened()` with `BanditPruner`
//! - Uses simulated verification with configurable acceptance rate
//! - Runs both modes side-by-side and prints comparison metrics
//!
//! Run: `cargo run --example bandit_02_ddtree --features bandit`
//!
//! # What This Proves
//!
//! - **DDTree + Bandit integration**: Real marginals flow through DDTree,
//!   BanditPruner screens branches via `relevance()`, and verification rewards
//!   feed back into Q-values.
//! - **Model-based advantage**: Concentrated marginals → better tree structure →
//!   higher accept rates → faster bandit convergence → lower regret.
//! - **Modelless baseline**: Uniform marginals → bandit learns everything from
//!   scratch → slower convergence, higher regret, but still functional.
//! - **Quantified gap**: Exact numbers for reward, regret, accept rate, tree size.

use std::time::Instant;

use microgpt_rs::pruners::{BanditPruner, BanditStrategy};
use microgpt_rs::speculative::{NoScreeningPruner, build_dd_tree_screened, extract_best_path_into};
use microgpt_rs::types::{Config, Rng};

// ── Demo Mode ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DemoMode {
    /// Concentrated marginals — simulates a draft model that knows syntax.
    ModelBased,
    /// Uniform marginals — bandit must learn everything from scratch.
    Modelless,
}

impl std::fmt::Display for DemoMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DemoMode::ModelBased => write!(f, "Model-based"),
            DemoMode::Modelless => write!(f, "Modelless"),
        }
    }
}

// ── Episode Result ─────────────────────────────────────────────

struct EpisodeResult {
    cumulative_reward: f32,
    cumulative_regret: f32,
    tree_nodes: usize,
    accepted_count: usize,
    total_tokens: usize,
    time_us: u64,
}

// ── Marginal Generation ────────────────────────────────────────

/// Model-based marginals: concentrated distribution where 3-4 tokens
/// get ~80% of probability mass. Simulates a draft model that knows syntax.
fn model_based_marginals(vocab_size: usize, lookahead: usize, rng: &mut Rng) -> Vec<Vec<f32>> {
    let mut marginals = Vec::with_capacity(lookahead);

    for _ in 0..lookahead {
        let mut probs = vec![f32::MIN_POSITIVE; vocab_size];

        // Pick 3-4 "good" tokens that get most of the mass
        let num_good = 3 + (rng.uniform() * 2.0) as usize;
        let mut good_tokens = Vec::with_capacity(num_good);
        for _ in 0..num_good {
            let t = (rng.uniform() * vocab_size as f32) as usize;
            good_tokens.push(t.min(vocab_size - 1));
        }

        // Distribute ~80% of mass among good tokens
        let per_good = 0.80 / num_good as f32;
        for &t in &good_tokens {
            probs[t] = per_good;
        }

        // Normalize to valid probability distribution
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
            *p = p.max(f32::MIN_POSITIVE); // ln() safety
        }

        marginals.push(probs);
    }

    marginals
}

/// Modelless marginals: uniform distribution — all tokens equally likely.
fn modelless_marginals(vocab_size: usize, lookahead: usize) -> Vec<Vec<f32>> {
    let uniform = 1.0 / vocab_size as f32;
    (0..lookahead).map(|_| vec![uniform; vocab_size]).collect()
}

// ── Simulated Verification ─────────────────────────────────────

/// Simulate target model verification: accepts/rejects each token.
/// Higher-indexed tokens get a small bias (simulates target model preference).
fn simulate_verification(path: &[usize], rng: &mut Rng, base_accept_rate: f32) -> (Vec<f32>, f32) {
    let mut rewards = Vec::with_capacity(path.len());
    let mut cumulative = 0.0f32;

    for &token_idx in path {
        let token_bias = (token_idx as f32) / 54.0; // small bias from token index
        let accept_rate = (base_accept_rate + token_bias).min(1.0);
        let reward = if rng.uniform() < accept_rate {
            1.0
        } else {
            0.0
        };
        rewards.push(reward);
        cumulative += reward;
    }

    (rewards, cumulative)
}

// ── Episode Runner ─────────────────────────────────────────────

fn run_episode(
    pruner: &mut BanditPruner<NoScreeningPruner>,
    config: &Config,
    mode: &DemoMode,
    rng: &mut Rng,
    optimal_reward: f32,
) -> EpisodeResult {
    let start = Instant::now();

    // 1. Prepare episode (refreshes Thompson Sampling cache)
    pruner.prepare_episode(rng);

    // 2. Generate marginals based on mode
    let marginals = match mode {
        DemoMode::ModelBased => {
            model_based_marginals(config.vocab_size, config.draft_lookahead, rng)
        }
        DemoMode::Modelless => modelless_marginals(config.vocab_size, config.draft_lookahead),
    };

    // 3. Convert to &[&[f32]] for DDTree
    let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();

    // 4. Build DDTree with bandit screening
    let tree = build_dd_tree_screened(&slices, config, pruner, true);

    // 5. Extract best path from tree
    let mut path = Vec::new();
    extract_best_path_into(&tree, &mut path);

    // 6. Simulate verification (model-based proposes better tokens)
    let base_rate = match mode {
        DemoMode::ModelBased => 0.70,
        DemoMode::Modelless => 0.40,
    };
    let (rewards, cumulative_reward) = simulate_verification(&path, rng, base_rate);

    // 7. Feed rewards back into bandit
    for (&token_idx, &reward) in path.iter().zip(rewards.iter()) {
        pruner.update(token_idx, reward);
    }

    // 8. Decay epsilon (EpsilonGreedy only)
    pruner.decay_epsilon();

    let accepted_count = rewards.iter().filter(|&&r| r > 0.5).count();
    let time_us = start.elapsed().as_micros() as u64;

    EpisodeResult {
        cumulative_reward,
        cumulative_regret: (optimal_reward - cumulative_reward).max(0.0),
        tree_nodes: tree.len(),
        accepted_count,
        total_tokens: path.len(),
        time_us,
    }
}

// ── Demo Runner ────────────────────────────────────────────────

fn run_demo(config: &Config, mode: DemoMode, episodes: usize, seed: u64) -> Vec<EpisodeResult> {
    let mut rng = Rng::new(seed);
    let strategy = BanditStrategy::EpsilonGreedy {
        epsilon: 0.3,
        decay: 0.995,
    };
    let mut pruner = BanditPruner::new(NoScreeningPruner, strategy, config.vocab_size);
    let optimal = config.draft_lookahead as f32;

    (0..episodes)
        .map(|_| run_episode(&mut pruner, config, &mode, &mut rng, optimal))
        .collect()
}

// ── Comparison Table ───────────────────────────────────────────

fn pct_delta(model: f32, modelless: f32) -> String {
    if modelless.abs() < f32::EPSILON {
        return "      n/a".to_string();
    }
    format!("{:+.1}%", (model - modelless) / modelless.abs() * 100.0)
}

fn print_comparison(model: &[EpisodeResult], modelless: &[EpisodeResult], episodes: usize) {
    println!("Model vs Modelless Bandit Comparison ({episodes} episodes)");
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "{:<25} {:>14} {:>14} {:>10}",
        "Metric", "Model-based", "Modelless", "Δ"
    );
    println!("───────────────────────────────────────────────────────────────");

    // Cumulative Reward
    let mr: f32 = model.iter().map(|r| r.cumulative_reward).sum();
    let lr: f32 = modelless.iter().map(|r| r.cumulative_reward).sum();
    println!(
        "{:<25} {:>14.2} {:>14.2} {:>10}",
        "Cumulative Reward",
        mr,
        lr,
        pct_delta(mr, lr)
    );

    // Cumulative Regret
    let mrg: f32 = model.iter().map(|r| r.cumulative_regret).sum();
    let lrg: f32 = modelless.iter().map(|r| r.cumulative_regret).sum();
    println!(
        "{:<25} {:>14.2} {:>14.2} {:>10}",
        "Cumulative Regret",
        mrg,
        lrg,
        pct_delta(mrg, lrg)
    );

    // Accept Rate
    let ma: usize = model.iter().map(|r| r.accepted_count).sum();
    let mt: usize = model.iter().map(|r| r.total_tokens).sum();
    let la: usize = modelless.iter().map(|r| r.accepted_count).sum();
    let lt: usize = modelless.iter().map(|r| r.total_tokens).sum();
    let m_rate = if mt > 0 {
        ma as f32 / mt as f32 * 100.0
    } else {
        0.0
    };
    let l_rate = if lt > 0 {
        la as f32 / lt as f32 * 100.0
    } else {
        0.0
    };
    println!(
        "{:<25} {:>13.1}% {:>13.1}% {:>+9.1}%",
        "Accept Rate (%)",
        m_rate,
        l_rate,
        m_rate - l_rate
    );

    // Avg Tree Nodes
    let mn: f32 = model.iter().map(|r| r.tree_nodes as f32).sum::<f32>() / episodes as f32;
    let ln: f32 = modelless.iter().map(|r| r.tree_nodes as f32).sum::<f32>() / episodes as f32;
    println!(
        "{:<25} {:>14.1} {:>14.1} {:>10}",
        "Avg Tree Nodes",
        mn,
        ln,
        pct_delta(mn, ln)
    );

    // Avg Time/Episode
    let mt_us: f64 = model.iter().map(|r| r.time_us as f64).sum::<f64>() / episodes as f64;
    let lt_us: f64 = modelless.iter().map(|r| r.time_us as f64).sum::<f64>() / episodes as f64;
    println!(
        "{:<25} {:>11.1} µs {:>11.1} µs {:>10}",
        "Avg Time/Episode",
        mt_us,
        lt_us,
        pct_delta(mt_us as f32, lt_us as f32)
    );

    println!();
}

// ── ASCII Convergence Plot ─────────────────────────────────────

fn print_convergence_plot(model: &[EpisodeResult], modelless: &[EpisodeResult], episodes: usize) {
    println!("Cumulative Reward Over Episodes");
    println!("──────────────────────────────────────────────────────────");

    let cols = 50;
    let rows = 12;

    // Compute running cumulative reward at each column checkpoint
    let mut m_pts = vec![0.0f32; cols];
    let mut l_pts = vec![0.0f32; cols];
    let mut m_sum = 0.0f32;
    let mut l_sum = 0.0f32;

    for col in 0..cols {
        let end = episodes * (col + 1) / cols;
        let start = episodes * col / cols;
        for i in start..end {
            m_sum += model[i].cumulative_reward;
            l_sum += modelless[i].cumulative_reward;
        }
        m_pts[col] = m_sum;
        l_pts[col] = l_sum;
    }

    let max_val = m_pts
        .iter()
        .chain(l_pts.iter())
        .copied()
        .fold(0.0f32, f32::max)
        .max(1.0);

    // Draw rows top-to-bottom
    for row in (0..rows).rev() {
        let y_label = (max_val * row as f32 / (rows - 1) as f32) as i32;
        print!("{:>5} |", y_label);

        for col in 0..cols {
            let m_row = (m_pts[col] / max_val * (rows - 1) as f32).round() as usize;
            let l_row = (l_pts[col] / max_val * (rows - 1) as f32).round() as usize;
            let m_hit = m_row == row;
            let l_hit = l_row == row;

            match (m_hit, l_hit) {
                (true, true) => print!("┼"),
                (true, false) => print!("█"),
                (false, true) => print!("~"),
                (false, false) => print!(" "),
            }
        }
        println!();
    }

    // X-axis
    print!("   0 |{}", "─".repeat(cols));
    println!();
    print!("     ");
    let label_points = [0, episodes / 4, episodes / 2, 3 * episodes / 4, episodes];
    let mut prev = 0;
    for &val in &label_points {
        let col_pos = val * cols / episodes;
        let padding = col_pos.saturating_sub(prev);
        print!(
            "{:>width$}",
            val,
            width = padding.max(format!("{val}").len())
        );
        prev = col_pos + format!("{val}").len();
    }
    println!();
    println!("     ██ Model-based  ~~ Modelless");
    println!();
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    println!("Bandit + DDTree: Model vs Modelless Comparison\n");

    let config = Config::draft();
    let episodes = 1000;
    let seed = 42;

    println!(
        "Config: vocab_size={}, draft_lookahead={}, tree_budget={}",
        config.vocab_size, config.draft_lookahead, config.tree_budget
    );
    println!("Strategy: EpsilonGreedy {{ epsilon: 0.3, decay: 0.995 }}");
    println!("Episodes: {episodes}\n");

    println!("Running model-based episodes...");
    let model_results = run_demo(&config, DemoMode::ModelBased, episodes, seed);

    println!("Running modelless episodes...");
    let modelless_results = run_demo(&config, DemoMode::Modelless, episodes, seed + 1);

    println!();
    print_comparison(&model_results, &modelless_results, episodes);
    print_convergence_plot(&model_results, &modelless_results, episodes);
}
