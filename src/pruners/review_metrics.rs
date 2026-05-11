//! Inference-time review metrics for Reinforced Agent Distillation.
//!
//! Tracks helpful/harmful classifications of reviewer intervention,
//! enabling data-driven decisions about when reviewer feedback is net-positive.
//!
//! Based on arXiv:2604.27233 — "Reinforced Agent: Inference-Time Feedback
//! for Tool-Calling Agents". The paper found a 3.1:1 benefit-to-risk ratio
//! for reasoning reviewers, with +5.5% irrelevance detection and +7.1%
//! multi-turn task improvement.
//!
//! # Classification Matrix
//!
//! | Base Correct | Reviewed Correct | Classification    |
//! |:------------:|:----------------:|:------------------|
//! | false        | true             | Helpful           |
//! | true         | false            | Harmful           |
//! | true         | true             | Both Correct      |
//! | false        | false            | Both Wrong        |
//!
//! # Thread Safety
//!
//! All counters use `AtomicU64` with `Ordering::Relaxed` — these are
//! statistics, not synchronization primitives. Zero lock contention.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

// ── Review Summary ──────────────────────────────────────────────

/// Computed review metrics snapshot for display/logging.
#[derive(Clone, Debug, Default)]
pub struct ReviewSummary {
    /// % of cases where base was WRONG and reviewer FIXED it.
    pub helpfulness: f64,
    /// % of cases where base was RIGHT and reviewer BROKE it.
    pub harmfulness: f64,
    /// Helpfulness ÷ Harmfulness (benefit-to-risk ratio).
    /// `f64::INFINITY` when harmfulness is zero.
    pub benefit_ratio: f64,
    /// Total observations classified.
    pub total: u64,
    /// Raw counter: base wrong → reviewed correct.
    pub helpful: u64,
    /// Raw counter: base correct → reviewed wrong.
    pub harmful: u64,
    /// Raw counter: both agreed correct.
    pub both_correct: u64,
    /// Raw counter: both agreed wrong.
    pub both_wrong: u64,
}

// ── Review Metrics ──────────────────────────────────────────────

/// Thread-safe review metrics using atomic counters.
///
/// Classifies each (base_correct, reviewed_correct) pair and increments
/// the appropriate counter. All operations are lock-free.
///
/// # Example
///
/// ```rust,ignore
/// let metrics = ReviewMetrics::new();
///
/// // Base was wrong, reviewer fixed it → helpful
/// metrics.record(false, true);
///
/// // Base was right, reviewer broke it → harmful
/// metrics.record(true, false);
///
/// // Both correct
/// metrics.record(true, true);
///
/// let summary = metrics.summary();
/// println!("{metrics}"); // "helpful=33.3% harmful=33.3% ratio=1.0:1 n=3"
/// ```
pub struct ReviewMetrics {
    helpful: AtomicU64,
    harmful: AtomicU64,
    both_correct: AtomicU64,
    both_wrong: AtomicU64,
}

impl ReviewMetrics {
    /// Create zero-initialized metrics.
    pub fn new() -> Self {
        Self {
            helpful: AtomicU64::new(0),
            harmful: AtomicU64::new(0),
            both_correct: AtomicU64::new(0),
            both_wrong: AtomicU64::new(0),
        }
    }

    /// Classify and record a (base, reviewed) outcome.
    ///
    /// Thread-safe: single atomic increment, no lock.
    pub fn record(&self, base_correct: bool, reviewed_correct: bool) {
        match (base_correct, reviewed_correct) {
            (false, true) => self.helpful.fetch_add(1, Ordering::Relaxed),
            (true, false) => self.harmful.fetch_add(1, Ordering::Relaxed),
            (true, true) => self.both_correct.fetch_add(1, Ordering::Relaxed),
            (false, false) => self.both_wrong.fetch_add(1, Ordering::Relaxed),
        };
    }

    /// % of cases where base was WRONG and reviewer FIXED it.
    ///
    /// `helpful / (helpful + both_wrong)` as percentage.
    /// Returns 0.0 when denominator is zero.
    pub fn helpfulness(&self) -> f64 {
        let h = self.helpful.load(Ordering::Relaxed) as f64;
        let w = self.both_wrong.load(Ordering::Relaxed) as f64;
        let denom = h + w;
        if denom == 0.0 { 0.0 } else { h / denom * 100.0 }
    }

    /// % of cases where base was RIGHT and reviewer BROKE it.
    ///
    /// `harmful / (harmful + both_correct)` as percentage.
    /// Returns 0.0 when denominator is zero.
    pub fn harmfulness(&self) -> f64 {
        let h = self.harmful.load(Ordering::Relaxed) as f64;
        let c = self.both_correct.load(Ordering::Relaxed) as f64;
        let denom = h + c;
        if denom == 0.0 { 0.0 } else { h / denom * 100.0 }
    }

    /// Benefit-to-risk ratio: `helpfulness / harmfulness`.
    ///
    /// Returns `f64::INFINITY` when harmfulness is zero
    /// (reviewer never broke a correct base).
    pub fn benefit_ratio(&self) -> f64 {
        let helpfulness = self.helpfulness();
        let harmfulness = self.harmfulness();
        if harmfulness == 0.0 {
            if helpfulness > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            helpfulness / harmfulness
        }
    }

    /// Total number of classified observations.
    pub fn total(&self) -> u64 {
        self.helpful.load(Ordering::Relaxed)
            + self.harmful.load(Ordering::Relaxed)
            + self.both_correct.load(Ordering::Relaxed)
            + self.both_wrong.load(Ordering::Relaxed)
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.helpful.store(0, Ordering::Relaxed);
        self.harmful.store(0, Ordering::Relaxed);
        self.both_correct.store(0, Ordering::Relaxed);
        self.both_wrong.store(0, Ordering::Relaxed);
    }

    /// Compute a snapshot of all metrics for display/logging.
    pub fn summary(&self) -> ReviewSummary {
        let helpful = self.helpful.load(Ordering::Relaxed);
        let harmful = self.harmful.load(Ordering::Relaxed);
        let both_correct = self.both_correct.load(Ordering::Relaxed);
        let both_wrong = self.both_wrong.load(Ordering::Relaxed);
        ReviewSummary {
            helpfulness: self.helpfulness(),
            harmfulness: self.harmfulness(),
            benefit_ratio: self.benefit_ratio(),
            total: self.total(),
            helpful,
            harmful,
            both_correct,
            both_wrong,
        }
    }

    /// Raw helpful count (base wrong, reviewer fixed).
    pub fn helpful_count(&self) -> u64 {
        self.helpful.load(Ordering::Relaxed)
    }

    /// Raw harmful count (base correct, reviewer broke).
    pub fn harmful_count(&self) -> u64 {
        self.harmful.load(Ordering::Relaxed)
    }

    /// Raw both-correct count.
    pub fn both_correct_count(&self) -> u64 {
        self.both_correct.load(Ordering::Relaxed)
    }

    /// Raw both-wrong count.
    pub fn both_wrong_count(&self) -> u64 {
        self.both_wrong.load(Ordering::Relaxed)
    }
}

impl Default for ReviewMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ReviewMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ratio = self.benefit_ratio();
        let ratio_str = if ratio.is_infinite() {
            "∞".to_string()
        } else {
            format!("{ratio:.1}")
        };
        write!(
            f,
            "helpful={:.1}% harmful={:.1}% ratio={}:1 n={}",
            self.helpfulness(),
            self.harmfulness(),
            ratio_str,
            self.total()
        )
    }
}

impl fmt::Display for ReviewSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ratio_str = if self.benefit_ratio.is_infinite() {
            "∞".to_string()
        } else {
            format!("{:.1}", self.benefit_ratio)
        };
        write!(
            f,
            "helpful={:.1}% harmful={:.1}% ratio={}:1 n={}",
            self.helpfulness, self.harmfulness, ratio_str, self.total
        )
    }
}

// ── Review Strategy ─────────────────────────────────────────────

/// Structured review loop strategy mirroring arXiv:2604.27233's mechanisms.
///
/// The paper evaluates three inference-time feedback strategies:
/// - **Progressive Feedback** (rN): iteratively review and inject rejection feedback.
///   Best performer in the paper.
/// - **Best-of-N Selection** (sN): generate N candidates, reviewer picks best.
///   Maps to DDTree with budget N.
/// - **Best-of-N Grading** (gN): score each candidate 0.0–1.0, pick highest.
///   Maps to DDTree with ScreeningPruner relevance scoring.
///
/// Paper notation: `r2` = ProgressiveFeedback with max_loops=2, `s5` = BestOfNSelection
/// with candidates=5, `g5` = BestOfNGrading with candidates=5.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReviewStrategy {
    /// Progressive feedback: iteratively review and inject rejection feedback.
    /// Paper's rN — up to N review loops.
    ProgressiveFeedback { max_loops: usize },
    /// Best-of-N selection: generate N candidates, reviewer picks best.
    /// Maps to DDTree with budget N.
    BestOfNSelection { candidates: usize },
    /// Best-of-N grading: score each candidate 0.0–1.0, pick highest.
    /// Maps to DDTree with ScreeningPruner relevance scoring.
    BestOfNGrading { candidates: usize },
}

impl Default for ReviewStrategy {
    /// Default: `ProgressiveFeedback { max_loops: 2 }` — paper's best performer.
    fn default() -> Self {
        Self::ProgressiveFeedback { max_loops: 2 }
    }
}

impl fmt::Display for ReviewStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProgressiveFeedback { max_loops } => write!(f, "r{max_loops}"),
            Self::BestOfNSelection { candidates } => write!(f, "s{candidates}"),
            Self::BestOfNGrading { candidates } => write!(f, "g{candidates}"),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_classifications() {
        let metrics = ReviewMetrics::new();

        // Helpful: base wrong, reviewer fixed
        metrics.record(false, true);
        assert_eq!(metrics.helpful_count(), 1);

        // Harmful: base correct, reviewer broke
        metrics.record(true, false);
        assert_eq!(metrics.harmful_count(), 1);

        // Both correct
        metrics.record(true, true);
        assert_eq!(metrics.both_correct_count(), 1);

        // Both wrong
        metrics.record(false, false);
        assert_eq!(metrics.both_wrong_count(), 1);

        assert_eq!(metrics.total(), 4);
    }

    #[test]
    fn test_helpfulness_calculation() {
        let metrics = ReviewMetrics::new();

        // 3 helpful out of (3 helpful + 1 both_wrong) = 75%
        metrics.record(false, true); // helpful
        metrics.record(false, true); // helpful
        metrics.record(false, true); // helpful
        metrics.record(false, false); // both_wrong

        assert_eq!(metrics.helpful_count(), 3);
        assert_eq!(metrics.both_wrong_count(), 1);
        let h = metrics.helpfulness();
        assert!((h - 75.0).abs() < 0.01, "expected 75.0%, got {h}%");
    }

    #[test]
    fn test_harmfulness_calculation() {
        let metrics = ReviewMetrics::new();

        // 1 harmful out of (1 harmful + 3 both_correct) = 25%
        metrics.record(true, false); // harmful
        metrics.record(true, true); // both_correct
        metrics.record(true, true); // both_correct
        metrics.record(true, true); // both_correct

        let h = metrics.harmfulness();
        assert!((h - 25.0).abs() < 0.01, "expected 25.0%, got {h}%");
    }

    #[test]
    fn test_benefit_ratio() {
        let metrics = ReviewMetrics::new();

        // helpfulness=75%, harmfulness=25% → ratio=3.0
        metrics.record(false, true); // helpful
        metrics.record(false, true); // helpful
        metrics.record(false, true); // helpful
        metrics.record(false, false); // both_wrong

        metrics.record(true, false); // harmful
        metrics.record(true, true); // both_correct
        metrics.record(true, true); // both_correct
        metrics.record(true, true); // both_correct

        let ratio = metrics.benefit_ratio();
        assert!((ratio - 3.0).abs() < 0.01, "expected 3.0, got {ratio}");
    }

    #[test]
    fn test_zero_harmful_infinity_ratio() {
        let metrics = ReviewMetrics::new();

        // Only helpful records, no harmful → ratio = ∞
        metrics.record(false, true); // helpful
        metrics.record(false, false); // both_wrong

        // No harmful records: base_correct=true never produced reviewed_correct=false
        let ratio = metrics.benefit_ratio();
        assert!(ratio.is_infinite(), "expected infinity, got {ratio}");
    }

    #[test]
    fn test_empty_metrics_zero() {
        let metrics = ReviewMetrics::new();

        assert_eq!(metrics.total(), 0);
        assert_eq!(metrics.helpfulness(), 0.0);
        assert_eq!(metrics.harmfulness(), 0.0);
        assert_eq!(metrics.benefit_ratio(), 0.0);
    }

    #[test]
    fn test_reset() {
        let metrics = ReviewMetrics::new();
        metrics.record(false, true);
        metrics.record(true, false);
        assert_eq!(metrics.total(), 2);

        metrics.reset();
        assert_eq!(metrics.total(), 0);
        assert_eq!(metrics.helpful_count(), 0);
        assert_eq!(metrics.harmful_count(), 0);
    }

    #[test]
    fn test_display_format() {
        let metrics = ReviewMetrics::new();

        // Create 3 helpful, 1 harmful, 5 both_correct, 1 both_wrong → 10 total
        for _ in 0..3 {
            metrics.record(false, true); // helpful
        }
        metrics.record(true, false); // harmful
        for _ in 0..5 {
            metrics.record(true, true); // both_correct
        }
        metrics.record(false, false); // both_wrong

        let display = format!("{metrics}");
        assert!(
            display.contains("helpful="),
            "display should contain 'helpful=': {display}"
        );
        assert!(
            display.contains("harmful="),
            "display should contain 'harmful=': {display}"
        );
        assert!(
            display.contains("ratio="),
            "display should contain 'ratio=': {display}"
        );
        assert!(
            display.contains("n=10"),
            "display should contain 'n=10': {display}"
        );
    }

    #[test]
    fn test_display_infinity_ratio() {
        let metrics = ReviewMetrics::new();
        metrics.record(false, true);
        let display = format!("{metrics}");
        assert!(
            display.contains("ratio=∞"),
            "display should show ∞ for infinite ratio: {display}"
        );
    }

    #[test]
    fn test_summary_matches_metrics() {
        let metrics = ReviewMetrics::new();
        metrics.record(false, true);
        metrics.record(true, false);
        metrics.record(true, true);
        metrics.record(false, false);

        let summary = metrics.summary();
        assert_eq!(summary.helpful, 1);
        assert_eq!(summary.harmful, 1);
        assert_eq!(summary.both_correct, 1);
        assert_eq!(summary.both_wrong, 1);
        assert_eq!(summary.total, 4);
        assert!((summary.helpfulness - metrics.helpfulness()).abs() < 0.01);
        assert!((summary.harmfulness - metrics.harmfulness()).abs() < 0.01);
    }

    // ── ReviewStrategy Tests ────────────────────────────────────

    #[test]
    fn test_strategy_default() {
        match ReviewStrategy::default() {
            ReviewStrategy::ProgressiveFeedback { max_loops } => {
                assert_eq!(max_loops, 2);
            }
            _ => panic!("default should be ProgressiveFeedback"),
        }
    }

    #[test]
    fn test_strategy_display_progressive() {
        let strategy = ReviewStrategy::ProgressiveFeedback { max_loops: 3 };
        assert_eq!(format!("{strategy}"), "r3");
    }

    #[test]
    fn test_strategy_display_selection() {
        let strategy = ReviewStrategy::BestOfNSelection { candidates: 5 };
        assert_eq!(format!("{strategy}"), "s5");
    }

    #[test]
    fn test_strategy_display_grading() {
        let strategy = ReviewStrategy::BestOfNGrading { candidates: 10 };
        assert_eq!(format!("{strategy}"), "g10");
    }

    #[test]
    fn test_strategy_equality() {
        let a = ReviewStrategy::ProgressiveFeedback { max_loops: 2 };
        let b = ReviewStrategy::ProgressiveFeedback { max_loops: 2 };
        let c = ReviewStrategy::ProgressiveFeedback { max_loops: 3 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_summary_display() {
        let summary = ReviewSummary {
            helpfulness: 36.8,
            harmfulness: 11.7,
            benefit_ratio: 3.1,
            total: 1000,
            helpful: 368,
            harmful: 117,
            both_correct: 400,
            both_wrong: 115,
        };
        let display = format!("{summary}");
        assert!(
            display.contains("helpful=36.8%"),
            "expected helpful=36.8% in: {display}"
        );
        assert!(
            display.contains("harmful=11.7%"),
            "expected harmful=11.7% in: {display}"
        );
        assert!(
            display.contains("ratio=3.1:1"),
            "expected ratio=3.1:1 in: {display}"
        );
        assert!(display.contains("n=1000"), "expected n=1000 in: {display}");
    }
}
