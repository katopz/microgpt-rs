//! Slot Machine Bandit Demo — Rules-Based Speculative Decoding
//!
//! Proves DDTree + BanditPruner produces **actual value** with rules-based
//! marginals, verification, and reward — no real transformer needed.
//!
//! Unlike `bandit_demo.rs` (coin flips, no DDTree, disclaimer required) and
//! `bandit_ddtree_demo.rs` (random marginals, random verification), this demo:
//! - Uses **structured reel weights** as marginals (non-uniform per position)
//! - Uses **payline rules** for verification (deterministic combo checking)
//! - Uses **payout tables** for reward (graded 0.0–1.0, not binary)
//! - Bandit **learns which symbols lead to paying combos**
//!
//! No disclaimer needed — the full loop is closed:
//! `Marginals → DDTree → Verification → Reward → Bandit learns → Repeat`
//!
//! Run: `cargo run --example slot_bandit_demo --features bandit`

use microgpt_rs::pruners::{BanditPruner, BanditStrategy};
use microgpt_rs::speculative::{ScreeningPruner, build_dd_tree_screened, extract_best_path_into};
use microgpt_rs::types::{Config, Rng};

// ── Constants ──────────────────────────────────────────────────

const EPISODES: usize = 500;
const SEED: u64 = 42;
const NUM_REELS: usize = 3;
const NUM_SYMBOLS: usize = 6;

// ── Symbol ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Symbol {
    Cherry,
    Lemon,
    Orange,
    Bell,
    Diamond,
    Seven,
}

impl Symbol {
    fn from_usize(n: usize) -> Self {
        match n % NUM_SYMBOLS {
            0 => Self::Cherry,
            1 => Self::Lemon,
            2 => Self::Orange,
            3 => Self::Bell,
            4 => Self::Diamond,
            _ => Self::Seven,
        }
    }

    fn emoji(self) -> &'static str {
        match self {
            Self::Cherry => "🍒",
            Self::Lemon => "🍋",
            Self::Orange => "🍊",
            Self::Bell => "🔔",
            Self::Diamond => "💎",
            Self::Seven => "7️⃣",
        }
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.emoji())
    }
}

// ── Combo ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Combo {
    Jackpot,
    BigWin,
    Nice,
    Win,
    Pair,
    Miss,
}

impl Combo {
    fn reward(self) -> f32 {
        match self {
            Self::Jackpot => 1.0,
            Self::BigWin => 0.8,
            Self::Nice => 0.6,
            Self::Win => 0.5,
            Self::Pair => 0.2,
            Self::Miss => 0.0,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Jackpot => "JACKPOT 7️⃣7️⃣7️⃣",
            Self::BigWin => "BIG WIN 💎💎💎",
            Self::Nice => "NICE 🔔🔔🔔",
            Self::Win => "TRIPLE 🍒🍒🍒",
            Self::Pair => "PAIR",
            Self::Miss => "MISS",
        }
    }
}

// ── Slot Reels ─────────────────────────────────────────────────

struct SlotReels {
    weights: [[f32; NUM_SYMBOLS]; NUM_REELS],
}

impl SlotReels {
    fn new() -> Self {
        Self {
            weights: [
                [0.30, 0.25, 0.20, 0.15, 0.07, 0.03], // Reel 0: Cherry-heavy
                [0.25, 0.20, 0.20, 0.15, 0.10, 0.10], // Reel 1: Balanced
                [0.20, 0.20, 0.20, 0.15, 0.15, 0.10], // Reel 2: Diamond/Seven heavier
            ],
        }
    }

    fn marginals(&self) -> Vec<Vec<f32>> {
        self.weights.iter().map(|reel| reel.to_vec()).collect()
    }

    fn spin_random(&self, rng: &mut Rng) -> Vec<usize> {
        (0..NUM_REELS)
            .map(|reel| {
                let r = rng.uniform();
                let mut cumulative = 0.0f32;
                for (symbol, &weight) in self.weights[reel].iter().enumerate() {
                    cumulative += weight;
                    if r < cumulative {
                        return symbol;
                    }
                }
                NUM_SYMBOLS - 1
            })
            .collect()
    }
}

// ── Payline Rules ──────────────────────────────────────────────

struct PaylineRules;

impl PaylineRules {
    fn evaluate(path: &[usize]) -> (Combo, f32) {
        if path.len() < NUM_REELS {
            return (Combo::Miss, 0.0);
        }

        let (s0, s1, s2) = (path[0], path[1], path[2]);

        // Triple: all three match
        if s0 == s1 && s1 == s2 {
            let combo = match s0 {
                5 => Combo::Jackpot,
                4 => Combo::BigWin,
                3 => Combo::Nice,
                _ => Combo::Win,
            };
            return (combo, combo.reward());
        }

        // Pair: any two match
        if s0 == s1 || s1 == s2 || s0 == s2 {
            return (Combo::Pair, Combo::Pair.reward());
        }

        (Combo::Miss, 0.0)
    }
}

// ── Slot Screening Pruner ─────────────────────────────────────

/// Payline-aware domain knowledge: knows which symbol patterns form paying combos.
///
/// This is the "draft model's syntactic knowledge" equivalent — it provides
/// relevance hints about which symbol completions are promising.
/// The BanditPruner layer learns on top of this which combos actually pay.
struct SlotScreeningPruner;

impl ScreeningPruner for SlotScreeningPruner {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        match depth {
            0 => match token_idx {
                5 => 0.95, // Seven — highest payout potential
                4 => 0.90, // Diamond
                3 => 0.85, // Bell
                _ => 0.70, // Cherry, Lemon, Orange
            },
            1 => {
                let matches_parent = parent_tokens.first() == Some(&token_idx);
                if matches_parent {
                    match token_idx {
                        5 => 1.00, // Seven triple pursuit
                        4 => 0.95, // Diamond triple pursuit
                        3 => 0.90, // Bell triple pursuit
                        _ => 0.80, // Other triple pursuit
                    }
                } else {
                    0.50 // Different symbol — pair still possible
                }
            }
            2 => {
                let m0 = parent_tokens.first() == Some(&token_idx);
                let m1 = parent_tokens.get(1) == Some(&token_idx);
                match (m0, m1) {
                    (true, true) => 1.00,          // Triple completion
                    (true, _) | (_, true) => 0.70, // Pair
                    _ => 0.30,                     // No match possible
                }
            }
            _ => 0.50,
        }
    }
}

// ── Episode Result ─────────────────────────────────────────────

#[derive(Debug, Clone)]
struct EpisodeResult {
    reward: f32,
    combo: Combo,
    path: Vec<usize>,
    tree_nodes: usize,
}

// ── Episode Runners ────────────────────────────────────────────

fn run_bandit_episode(
    pruner: &mut BanditPruner<SlotScreeningPruner>,
    config: &Config,
    reels: &SlotReels,
    rng: &mut Rng,
) -> EpisodeResult {
    pruner.prepare_episode(rng);

    let marginals = reels.marginals();
    let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();

    let tree = build_dd_tree_screened(&slices, config, pruner, true);

    let mut path = Vec::new();
    extract_best_path_into(&tree, &mut path);

    let (combo, reward) = PaylineRules::evaluate(&path);

    for &symbol in &path {
        pruner.update(symbol, reward);
    }

    pruner.decay_epsilon();

    EpisodeResult {
        reward,
        combo,
        path,
        tree_nodes: tree.len(),
    }
}

fn run_random_episode(reels: &SlotReels, rng: &mut Rng) -> EpisodeResult {
    let path = reels.spin_random(rng);
    let (combo, reward) = PaylineRules::evaluate(&path);
    EpisodeResult {
        reward,
        combo,
        path,
        tree_nodes: 0,
    }
}

// ── Strategy Runner ────────────────────────────────────────────

struct StrategyResult {
    name: String,
    results: Vec<EpisodeResult>,
    final_q_values: Vec<f32>,
}

fn run_strategy(
    config: &Config,
    reels: &SlotReels,
    strategy: BanditStrategy,
    episodes: usize,
    seed: u64,
    name: &str,
) -> StrategyResult {
    let mut rng = Rng::new(seed);
    let mut pruner = BanditPruner::new(SlotScreeningPruner, strategy, NUM_SYMBOLS);

    let results: Vec<EpisodeResult> = (0..episodes)
        .map(|_| run_bandit_episode(&mut pruner, config, reels, &mut rng))
        .collect();

    let final_q = pruner.q_values().to_vec();

    StrategyResult {
        name: name.to_string(),
        results,
        final_q_values: final_q,
    }
}

fn run_random_baseline(reels: &SlotReels, episodes: usize, seed: u64) -> StrategyResult {
    let mut rng = Rng::new(seed);
    let results: Vec<EpisodeResult> = (0..episodes)
        .map(|_| run_random_episode(reels, &mut rng))
        .collect();

    StrategyResult {
        name: "Random".to_string(),
        results,
        final_q_values: vec![0.0; NUM_SYMBOLS],
    }
}

// ── Printing ───────────────────────────────────────────────────

fn print_header() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   Slot Machine Bandit — Rules-Based Speculative Decoding    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Full loop closed — no real transformer needed:");
    println!("  Reel weights → DDTree → Payline rules → Reward → Bandit learns");
    println!();
}

fn print_reel_weights(reels: &SlotReels) {
    println!("🎰 Reel Weights (Marginals)");
    println!("─────────────────────────────────────────────────────");
    print!("{:<10}", "Symbol");
    for reel in 0..NUM_REELS {
        print!("  Reel {reel}     ");
    }
    println!();
    println!("─────────────────────────────────────────────────────");

    for symbol in 0..NUM_SYMBOLS {
        let s = Symbol::from_usize(symbol);
        print!("{s:<10}");
        for reel in 0..NUM_REELS {
            print!("  {:>6.1}%     ", reels.weights[reel][symbol] * 100.0);
        }
        println!();
    }
    println!();
}

fn print_paytable() {
    println!("💰 Paytable (Verification Rules)");
    println!("─────────────────────────────────────────────────────");
    println!("  {:<20} {:>10} {:>10}", "Combo", "Symbols", "Reward");
    println!("─────────────────────────────────────────────────────");
    println!("  {:<20} {:>10} {:>10.1}", "JACKPOT", "7️⃣7️⃣7️⃣", 1.0);
    println!("  {:<20} {:>10} {:>10.1}", "BIG WIN", "💎💎💎", 0.8);
    println!("  {:<20} {:>10} {:>10.1}", "NICE", "🔔🔔🔔", 0.6);
    println!("  {:<20} {:>10} {:>10.1}", "TRIPLE", "🍒🍒🍒 etc", 0.5);
    println!("  {:<20} {:>10} {:>10.1}", "PAIR", "xx* / x*x", 0.2);
    println!("  {:<20} {:>10} {:>10.1}", "MISS", "anything", 0.0);
    println!();
}

fn print_comparison(strategies: &[StrategyResult], episodes: usize) {
    println!("📊 Strategy Comparison ({episodes} episodes)");
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "{:<12} {:>11} {:>11} {:>14} {:>10} {:>8} {:>10}",
        "Strategy", "Total Rwd", "Avg Rwd", "Best Combo", "Triples", "Pairs", "Avg Tree"
    );
    println!("───────────────────────────────────────────────────────────────────────────");

    for strategy in strategies {
        let total: f32 = strategy.results.iter().map(|r| r.reward).sum();
        let avg = total / episodes as f32;
        let best = strategy
            .results
            .iter()
            .map(|r| r.combo)
            .min()
            .unwrap_or(Combo::Miss);
        let triples = strategy
            .results
            .iter()
            .filter(|r| {
                matches!(
                    r.combo,
                    Combo::Jackpot | Combo::BigWin | Combo::Nice | Combo::Win
                )
            })
            .count();
        let pairs = strategy
            .results
            .iter()
            .filter(|r| matches!(r.combo, Combo::Pair))
            .count();
        let best_short = best.label().split_whitespace().next().unwrap_or("?");
        let avg_tree: f32 = strategy
            .results
            .iter()
            .map(|r| r.tree_nodes as f32)
            .sum::<f32>()
            / strategy.results.len().max(1) as f32;

        println!(
            "{:<12} {:>11.2} {:>11.4} {:>14} {:>10} {:>8} {:>10.1}",
            strategy.name, total, avg, best_short, triples, pairs, avg_tree,
        );
    }
    println!();
}

fn print_q_values(strategies: &[StrategyResult]) {
    let bandit_strategies: Vec<_> = strategies.iter().filter(|s| s.name != "Random").collect();

    if bandit_strategies.is_empty() {
        return;
    }

    println!("🎯 Bandit Q-Values (What the bandit learned)");
    println!("═══════════════════════════════════════════════════════════════");

    print!("{:<12}", "Symbol");
    for strategy in &bandit_strategies {
        print!("  {:>12}", strategy.name);
    }
    println!();
    println!("───────────────────────────────────────────────────────────────");

    for symbol in 0..NUM_SYMBOLS {
        let s = Symbol::from_usize(symbol);
        print!("{s:<12}");
        for strategy in &bandit_strategies {
            let q = strategy.final_q_values.get(symbol).copied().unwrap_or(0.0);
            print!("  {:>12.4}", q);
        }
        println!();
    }
    println!();
}

fn print_best_combos(strategies: &[StrategyResult]) {
    println!("🏆 Best Combos Found");
    println!("─────────────────────────────────────────────────────");

    for strategy in strategies {
        let best = strategy.results.iter().min_by_key(|r| r.combo).unwrap();

        let path_str: String = best
            .path
            .iter()
            .map(|&s| Symbol::from_usize(s).emoji())
            .collect();

        println!(
            "  {:<12} {} ({}, reward={:.1})",
            strategy.name,
            path_str,
            best.combo.label(),
            best.reward,
        );
    }
    println!();
}

fn print_convergence(strategies: &[StrategyResult], episodes: usize) {
    let window = 50;
    let num_windows = episodes / window;

    println!("📈 Convergence (Avg Reward Per {window}-Episode Window)");
    println!("───────────────────────────────────────────────────────────────");

    print!("{:>15}", "Episodes");
    for strategy in strategies {
        print!("  {:>12}", strategy.name);
    }
    println!();
    println!("───────────────────────────────────────────────────────────────");

    for w in 0..num_windows {
        let start = w * window;
        let end = start + window;
        print!("{start:>5}-{end:<9}");

        for strategy in strategies {
            let avg: f32 = strategy.results[start..end]
                .iter()
                .map(|r| r.reward)
                .sum::<f32>()
                / window as f32;
            print!("  {:>12.4}", avg);
        }
        println!();
    }
    println!();
}

fn print_verdict(strategies: &[StrategyResult]) {
    println!("✅ Verdict");
    println!("═══════════════════════════════════════════════════════════════");

    let random_total: f32 = strategies
        .iter()
        .find(|s| s.name == "Random")
        .map(|s| s.results.iter().map(|r| r.reward).sum())
        .unwrap_or(0.0);

    let mut all_outperform = true;

    for strategy in strategies.iter().filter(|s| s.name != "Random") {
        let total: f32 = strategy.results.iter().map(|r| r.reward).sum();
        let delta = if random_total > 0.0 {
            (total - random_total) / random_total * 100.0
        } else {
            0.0
        };

        let verdict = if total > random_total * 1.1 {
            "✓ OUTPERFORMS"
        } else if total > random_total {
            "≈ SLIGHTLY BETTER"
        } else {
            all_outperform = false;
            "✗ NO IMPROVEMENT"
        };

        println!(
            "  {:<12} vs Random: {:.2} vs {:.2} ({delta:+.1}%) → {verdict}",
            strategy.name, total, random_total,
        );
    }

    println!();
    if all_outperform {
        println!("  No disclaimer needed — the loop is closed.");
        println!("  Marginals → DDTree → Verification → Reward → Bandit learns.");
    } else {
        println!("  Results may vary with seed — try different seeds for comparison.");
    }
    println!();
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    print_header();

    let reels = SlotReels::new();
    print_reel_weights(&reels);
    print_paytable();

    let mut config = Config::draft();
    config.vocab_size = NUM_SYMBOLS;
    config.draft_lookahead = NUM_REELS;
    config.tree_budget = 50;
    config.screening_threshold = 0.0;

    println!(
        "Config: vocab={}, lookahead={}, budget={}",
        config.vocab_size, config.draft_lookahead, config.tree_budget
    );
    println!("Episodes: {EPISODES}\n");

    println!("Running UCB1...");
    let ucb1 = run_strategy(
        &config,
        &reels,
        BanditStrategy::Ucb1,
        EPISODES,
        SEED,
        "UCB1",
    );

    println!("Running ε-greedy...");
    let egreedy = run_strategy(
        &config,
        &reels,
        BanditStrategy::EpsilonGreedy {
            epsilon: 0.3,
            decay: 0.995,
        },
        EPISODES,
        SEED + 1,
        "ε-greedy",
    );

    println!("Running Thompson...");
    let thompson = run_strategy(
        &config,
        &reels,
        BanditStrategy::ThompsonSampling,
        EPISODES,
        SEED + 2,
        "Thompson",
    );

    println!("Running random baseline...");
    let random = run_random_baseline(&reels, EPISODES, SEED + 3);

    let strategies = vec![ucb1, egreedy, thompson, random];

    println!();
    print_comparison(&strategies, EPISODES);
    print_q_values(&strategies);
    print_best_combos(&strategies);
    print_convergence(&strategies, EPISODES);
    print_verdict(&strategies);
}
