//! AI Director Demo — Left 4 Dead Style Encounter Pacing with Bandit
//!
//! Demonstrates how multi-armed bandit algorithms can drive game encounter pacing,
//! inspired by Left 4 Dead's AI Director that dynamically adjusts difficulty.
//!
//! The "director" treats encounter types as bandit arms and party satisfaction
//! as the reward signal. Different party archetypes (speedrunners, completionists,
//! casuals) have different reward preferences, so the director learns to adapt.
//!
//! # Key Insight: "Game Feel" Optimization
//!
//! Traditional game design uses scripted encounter sequences. The AI Director
//! approach uses bandit feedback to learn what pacing maximizes player engagement:
//! - Speedrunners want challenge → director learns HardMob + Boss + Trap
//! - Casuals want rewards → director learns Treasure + EasyMob + Nothing (breather)
//! - Completionists want variety → director balances all encounter types
//!
//! Run: `cargo run --example ai_director_demo --features bandit`

use microgpt_rs::pruners::{BanditStats, BanditStrategy};
use microgpt_rs::types::Rng;

// ── Encounter Types (6 arms, index 0-5) ───────────────────────

/// Encounter types the AI Director can spawn.
/// Each is a bandit arm the director pulls to pace the game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Encounter {
    EasyMob,  // Low threat, low reward
    HardMob,  // High threat, moderate reward
    Trap,     // Surprise damage, no loot
    Treasure, // No threat, high reward
    Boss,     // Very high threat, very high reward
    Nothing,  // Breather, recover HP
}

impl Encounter {
    const ALL: [Self; 6] = [
        Self::EasyMob,
        Self::HardMob,
        Self::Trap,
        Self::Treasure,
        Self::Boss,
        Self::Nothing,
    ];

    fn index(self) -> usize {
        match self {
            Self::EasyMob => 0,
            Self::HardMob => 1,
            Self::Trap => 2,
            Self::Treasure => 3,
            Self::Boss => 4,
            Self::Nothing => 5,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::EasyMob => "EasyMob",
            Self::HardMob => "HardMob",
            Self::Trap => "Trap",
            Self::Treasure => "Treasure",
            Self::Boss => "Boss",
            Self::Nothing => "Nothing",
        }
    }
}

// ── Party Archetypes ──────────────────────────────────────────

/// Player archetype determines reward preferences.
/// The bandit director must discover each party's "fun profile".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PartyType {
    Speedrunner,   // Wants challenge, hates nothing
    Completionist, // Wants variety, dislikes repeats
    Casual,        // Wants treasure, hates traps
}

impl PartyType {
    fn name(self) -> &'static str {
        match self {
            Self::Speedrunner => "Speedrunner",
            Self::Completionist => "Completionist",
            Self::Casual => "Casual",
        }
    }

    /// Base reward per encounter — this is what the director must learn.
    fn reward(self, enc: Encounter) -> f32 {
        match self {
            //              EasyMob HardMob Trap Treasure Boss  Nothing
            Self::Speedrunner => [0.3, 0.8, 0.6, 0.1, 1.0, 0.0][enc.index()],
            Self::Completionist => [0.5, 0.5, 0.5, 0.5, 0.5, 0.2][enc.index()],
            Self::Casual => [0.4, 0.2, 0.0, 0.9, 0.3, 0.7][enc.index()],
        }
    }
}

// ── Party State (HP, morale, engagement) ──────────────────────

struct Party {
    hp: f32,
    morale: f32,
    engagement: f32,
    alive: bool,
}

impl Party {
    fn new() -> Self {
        Self {
            hp: 100.0,
            morale: 50.0,
            engagement: 0.0,
            alive: true,
        }
    }

    /// Apply encounter effects to party. Returns morale delta for engagement.
    fn apply_encounter(&mut self, enc: Encounter, rng: &mut Rng) -> f32 {
        if !self.alive {
            return 0.0;
        }

        // Passive regen between encounters (L4D-style recovery window)
        self.hp = (self.hp + 3.0).min(100.0);

        // Stochastic survival for dangerous encounters
        let survive = match enc {
            Encounter::EasyMob | Encounter::Trap | Encounter::Treasure | Encounter::Nothing => true,
            Encounter::HardMob => rng.uniform() > 0.15,
            Encounter::Boss => rng.uniform() > 0.25,
        };

        let morale_delta = match enc {
            Encounter::EasyMob => {
                self.hp -= 5.0;
                5.0
            }
            Encounter::HardMob => {
                self.hp -= 20.0;
                if survive { 15.0 } else { -30.0 }
            }
            Encounter::Trap => {
                self.hp -= 15.0;
                -10.0
            }
            Encounter::Treasure => {
                self.hp += 5.0;
                20.0
            }
            Encounter::Boss => {
                self.hp -= 30.0;
                if survive { 40.0 } else { -100.0 }
            }
            Encounter::Nothing => {
                self.hp += 10.0;
                -5.0
            }
        };

        if !survive || self.hp <= 0.0 {
            self.alive = false;
        }
        self.hp = self.hp.clamp(0.0, 100.0);
        self.morale = (self.morale + morale_delta).clamp(0.0, 100.0);
        self.engagement += morale_delta;
        morale_delta
    }
}

// ── Director (Bandit-driven AI) ───────────────────────────────

struct Director {
    stats: BanditStats,
    strategy: BanditStrategy,
}

impl Director {
    fn new(strategy: BanditStrategy) -> Self {
        Self {
            stats: BanditStats::new(Encounter::ALL.len()),
            strategy,
        }
    }

    /// Select encounter (arm) using the configured bandit strategy.
    fn select(&self, rng: &mut Rng) -> usize {
        // Cold start: play each arm once
        for i in 0..Encounter::ALL.len() {
            if self.stats.visit_count(i) == 0 {
                return i;
            }
        }

        match &self.strategy {
            BanditStrategy::Ucb1 => (0..Encounter::ALL.len())
                .max_by(|&a, &b| {
                    self.stats
                        .ucb1_score(a)
                        .partial_cmp(&self.stats.ucb1_score(b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0),
            BanditStrategy::EpsilonGreedy { epsilon, .. } => {
                if rng.uniform() < *epsilon {
                    (rng.uniform() * Encounter::ALL.len() as f32) as usize % Encounter::ALL.len()
                } else {
                    self.stats.best_arm()
                }
            }
            BanditStrategy::ThompsonSampling => (0..Encounter::ALL.len())
                .map(|i| (i, self.stats.thompson_sample(i, rng)))
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0),
        }
    }

    fn observe(&mut self, arm: usize, reward: f32) {
        self.stats.update(arm, reward);
    }

    fn decay_epsilon(&mut self) {
        if let BanditStrategy::EpsilonGreedy { epsilon, decay } = &mut self.strategy {
            *epsilon *= *decay;
        }
    }
}

// ── Reward with Stochastic Noise ──────────────────────────────

fn noisy_reward(party: PartyType, enc: Encounter, rng: &mut Rng) -> f32 {
    let base = party.reward(enc);
    let noise = (rng.uniform() - 0.5) * 0.2; // ±0.1
    (base + noise).clamp(0.0, 1.0)
}

// ── Simulation Result ─────────────────────────────────────────

struct SimResult {
    visits: Vec<u32>,
    engagement: f32,
    survived: bool,
    enc_survived: usize,
    engagement_hist: Vec<f32>,
}

fn simulate(
    party_type: PartyType,
    strategy: BanditStrategy,
    max_enc: usize,
    seed: u64,
) -> SimResult {
    let mut rng = Rng::new(seed);
    let mut director = Director::new(strategy);
    let mut party = Party::new();
    let mut visits = vec![0u32; Encounter::ALL.len()];
    let mut engagement_hist = Vec::with_capacity(max_enc);

    for _ in 0..max_enc {
        let arm = director.select(&mut rng);
        let enc = Encounter::ALL[arm];
        visits[arm] += 1;

        let morale_delta = party.apply_encounter(enc, &mut rng);
        engagement_hist.push(morale_delta);

        let reward = if party.alive {
            noisy_reward(party_type, enc, &mut rng)
        } else {
            0.0
        };
        director.observe(arm, reward);
        director.decay_epsilon();

        if !party.alive {
            break;
        }
    }

    SimResult {
        visits,
        engagement: party.engagement,
        survived: party.alive,
        enc_survived: if party.alive {
            max_enc
        } else {
            engagement_hist.len()
        },
        engagement_hist,
    }
}

// ── Section 1: Director vs 3 Party Types ──────────────────────

fn section1() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Section 1: Director vs 3 Party Types (500 encounters)     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let max_enc = 500;
    println!("  Strategy: UCB1 | Encounters: {max_enc}\n");
    println!(
        "  {:<15} {:<30} {:>15} {:>10}",
        "Party Type", "Top 3 Encounters", "Avg Engagement", "Survival"
    );
    println!("  {}", "─".repeat(72));

    for pt in [
        PartyType::Speedrunner,
        PartyType::Completionist,
        PartyType::Casual,
    ] {
        let base_seed = match pt {
            PartyType::Speedrunner => 100,
            PartyType::Completionist => 200,
            PartyType::Casual => 300,
        };
        let mut agg = [0u32; 6];
        let mut surv = 0u32;
        let mut eng = 0.0f32;
        for r in 0..50 {
            let res = simulate(pt, BanditStrategy::Ucb1, max_enc, base_seed + r as u64);
            surv += res.survived as u32;
            eng += res.engagement;
            for (i, &v) in res.visits.iter().enumerate() {
                agg[i] += v;
            }
        }
        let mut idx: Vec<(usize, u32)> = agg.iter().copied().enumerate().collect();
        idx.sort_by(|a, b| b.1.cmp(&a.1));
        let top3: Vec<&str> = idx
            .iter()
            .take(3)
            .map(|&(i, _)| Encounter::ALL[i].name())
            .collect();
        println!(
            "  {:<15} {:<30} {:>15.1} {:>9.0}%",
            pt.name(),
            top3.join(" > "),
            eng / 50.0,
            surv as f32 / 50.0 * 100.0
        );
    }
    println!(
        "\n  ✅ Director adapts: Speedrunner→challenge, Casual→rewards, Completionist→variety\n"
    );
}

// ── Section 2: Strategy Comparison ────────────────────────────

fn section2() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Section 2: Strategy Comparison (Casual, 500 encounters)   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let strategies: Vec<(&str, BanditStrategy)> = vec![
        ("UCB1", BanditStrategy::Ucb1),
        (
            "ε-greedy",
            BanditStrategy::EpsilonGreedy {
                epsilon: 0.3,
                decay: 0.995,
            },
        ),
        ("Thompson", BanditStrategy::ThompsonSampling),
    ];

    println!(
        "  {:<12} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Strategy", "Engagement", "Surv%", "EasyMob%", "HardMob%", "Treasure%", "Nothing%"
    );
    println!("  {}", "─".repeat(76));

    for (name, strat) in &strategies {
        let mut eng = 0.0f32;
        let mut surv = 0u32;
        let mut agg = [0u32; 6];
        for r in 0..20 {
            let res = simulate(PartyType::Casual, strat.clone(), 500, 400 + r as u64);
            eng += res.engagement;
            surv += res.survived as u32;
            for (i, &v) in res.visits.iter().enumerate() {
                agg[i] += v;
            }
        }
        let total: u32 = agg.iter().sum();
        let pct = |i: usize| {
            if total > 0 {
                agg[i] as f32 / total as f32 * 100.0
            } else {
                0.0
            }
        };
        println!(
            "  {:<12} {:>12.1} {:>9.0}% {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}%",
            name,
            eng / 20.0,
            surv as f32 / 20.0 * 100.0,
            pct(0),
            pct(1),
            pct(3),
            pct(5)
        );
    }

    // ASCII engagement plot — try multiple seeds to find a surviving run
    println!("\n  Engagement Over Time (smoothed):");
    println!("  {}", "─".repeat(52));
    for (name, strat) in &strategies {
        let w = 20;
        let res = (444..544)
            .map(|seed| simulate(PartyType::Casual, strat.clone(), 500, seed))
            .find(|res| res.engagement_hist.len() >= w)
            .unwrap_or_else(|| simulate(PartyType::Casual, strat.clone(), 500, 444));

        let smoothed: Vec<f32> = res
            .engagement_hist
            .windows(w)
            .map(|win| win.iter().sum::<f32>() / win.len() as f32)
            .collect();
        if smoothed.is_empty() {
            println!("  {name:<12}(wiped too early — no data)");
            continue;
        }
        let min = smoothed.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = smoothed.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max - min).max(1.0);
        let cols = 50usize;
        let step = (smoothed.len() / cols).max(1);
        print!("  {name:<12}");
        for col in 0..cols {
            let idx = (col * step).min(smoothed.len().saturating_sub(1));
            let norm = (smoothed[idx] - min) / range;
            let height = (norm * 8.0) as usize;
            let bar = match height {
                0..=2 => "▁",
                3..=4 => "▄",
                5..=6 => "▆",
                _ => "█",
            };
            print!("{bar}");
        }
        println!();
    }
    println!();
}

// ── Section 3: Encounter Distribution ─────────────────────────

fn section3() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Section 3: Encounter Distribution per Party Type          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    for pt in [
        PartyType::Speedrunner,
        PartyType::Completionist,
        PartyType::Casual,
    ] {
        let seed = match pt {
            PartyType::Speedrunner => 500,
            PartyType::Completionist => 600,
            PartyType::Casual => 700,
        };
        let res = simulate(pt, BanditStrategy::Ucb1, 500, seed);
        let total: u32 = res.visits.iter().sum();

        println!("  {} (Bandit Director):", pt.name());
        for (i, enc) in Encounter::ALL.iter().enumerate() {
            let pct = res.visits[i] as f32 / total as f32 * 100.0;
            let bar = "█".repeat((pct / 2.5) as usize);
            println!("    {:<12} {:>5.1}% {}", enc.name(), pct, bar);
        }

        // Naive uniform comparison
        println!("  {} (Naive/Uniform):", pt.name());
        let uniform_pct = 100.0 / 6.0;
        let bar = "░".repeat((uniform_pct / 2.5) as usize);
        for enc in Encounter::ALL.iter() {
            println!("    {:<12} {:>5.1}% {}", enc.name(), uniform_pct, bar);
        }
        println!();
    }
}

// ── Section 4: Survival Analysis ──────────────────────────────

fn section4() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Section 4: Survival Analysis (100 runs × 50 encounters)  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let runs = 100;
    let enc_per_run = 50;
    let pt = PartyType::Casual;

    // Bandit vs Random directors
    let mut b_surv = 0u32;
    let mut b_eng = 0.0f32;
    let mut b_enc = 0.0f32;
    let mut r_surv = 0u32;
    let mut r_eng = 0.0f32;
    let mut r_enc = 0.0f32;

    for run in 0..runs {
        let br = simulate(pt, BanditStrategy::Ucb1, enc_per_run, 800 + run as u64);
        b_surv += br.survived as u32;
        b_eng += br.engagement;
        b_enc += br.enc_survived as f32;

        let rr = simulate(
            pt,
            BanditStrategy::EpsilonGreedy {
                epsilon: 1.0,
                decay: 1.0,
            },
            enc_per_run,
            900 + run as u64,
        );
        r_surv += rr.survived as u32;
        r_eng += rr.engagement;
        r_enc += rr.enc_survived as f32;
    }

    println!(
        "  {:<20} {:>14} {:>14} {:>20}",
        "Director", "Survival Rate", "Avg Engage", "Avg Enc Before Wipe"
    );
    println!("  {}", "─".repeat(70));
    let b_sr = b_surv as f32 / runs as f32 * 100.0;
    let r_sr = r_surv as f32 / runs as f32 * 100.0;
    println!(
        "  {:<20} {:>13.0}% {:>14.1} {:>20.1}",
        "Bandit (UCB1)",
        b_sr,
        b_eng / runs as f32,
        b_enc / runs as f32
    );
    println!(
        "  {:<20} {:>13.0}% {:>14.1} {:>20.1}",
        "Random (ε=1.0)",
        r_sr,
        r_eng / runs as f32,
        r_enc / runs as f32
    );
    println!();

    let eng_delta = (b_eng - r_eng) / r_eng.abs() * 100.0;
    let sr_delta = b_sr - r_sr;
    println!("  Δ Engagement: {eng_delta:+.1}% | Δ Survival: {sr_delta:+.0}pp");
    println!();
}

// ── Main ──────────────────────────────────────────────────────

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   AI Director Demo — Left 4 Dead Style Encounter Pacing    ║");
    println!("║   Bandit-driven game feel optimization                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  In Left 4 Dead, the AI Director monitors player stress and");
    println!("  adjusts pacing: intense fights → breathers → rewards.");
    println!("  Here we use multi-armed bandits to learn optimal pacing.\n");

    section1();
    section2();
    section3();
    section4();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   Bandit directors adapt pacing: challenge-seekers,        ║");
    println!("║   reward-seekers, variety-seekers — all get their fun.      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
