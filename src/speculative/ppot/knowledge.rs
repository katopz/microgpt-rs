//! PPoT knowledge: Rejection memory for adaptive rescue.
//!
//! Distilled from "Test-time Recursive Thinking" (arXiv:2602.03094).
//! When PPoT resamples token variants and the `ConstraintPruner` rejects them,
//! structured "don't" insights are recorded that bias future resampling within
//! the same generation session.
//!
//! TRT proves three things applied at token level:
//! 1. **"Don'ts" beat "dos"** — recording failure patterns outperforms successes
//! 2. **Knowledge is compact** — stays under 1.5% of context after 64 rounds
//! 3. **Depth beats breadth** — iterative refinement with accumulated knowledge
//!    outperforms parallel random sampling
//!
//! Ring buffer sizing: 64 × ~48 bytes = 3KB per session.

use super::types::{PpotConfig, TokenRule};

// ── Rejection Insight ──────────────────────────────────────────

/// A single resampling attempt outcome, recorded for adaptive learning.
///
/// Each insight captures what was tried (position, rule, token), the context
/// (entropy level), and the result (accepted or rejected). Over a generation
/// session, patterns emerge: certain positions consistently fail with certain
/// rules, while others succeed.
///
/// The ring buffer in [`SessionKnowledge`] keeps the most recent insights,
/// evicting oldest entries when full. This naturally forgets stale patterns.
#[derive(Clone, Debug)]
pub struct RejectionInsight {
    /// Position in the token sequence that was resampled.
    pub position: usize,
    /// Original token at this position before resampling.
    pub original_token: usize,
    /// Token attempted during resampling.
    pub attempted_token: usize,
    /// Shannon entropy at this position (uncertainty measure).
    pub entropy: f32,
    /// Token rule used for this resampling attempt.
    pub rule: TokenRule,
    /// Error category if rejected (placeholder for future constraint diagnostics).
    pub error_kind: Option<ErrorKind>,
    /// Whether the pruner accepted this variant.
    pub accepted: bool,
}

/// Categorization of rejection reasons (future use for structured "don'ts").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ErrorKind {
    /// Token violates a hard constraint (e.g., invalid digit in math expression).
    ConstraintViolation,
    /// Token has low relevance score from screening pruner.
    LowRelevance,
    /// Token produces an invalid parse tree.
    ParseError,
    /// Unknown or uncategorized rejection.
    Unknown,
}

/// Maximum number of tracked positions for precomputed stats.
/// Covers speculative lookahead (positions 0–15).
const MAX_POSITIONS: usize = 16;

/// Precomputed per-position statistics for O(1) lookups.
///
/// Updated on every [`record`](SessionKnowledge::record) call. When insights
/// are evicted from the ring buffer, the corresponding counters are decremented.
#[derive(Clone, Debug, Default)]
struct PositionStats {
    /// Number of accepted insights at this position.
    accepted: usize,
    /// Number of rejected insights at this position.
    rejected: usize,
    /// Per-rule success counts at this position, indexed by `TokenRule::index()`.
    rule_success: [usize; 5],
    /// Per-rule failure counts at this position, indexed by `TokenRule::index()`.
    rule_fail: [usize; 5],
}

// ── Session Knowledge ──────────────────────────────────────────

/// Bounded ring buffer of rejection insights for adaptive PPoT rescue.
///
/// Accumulates "don't" knowledge across resampling attempts within a single
/// generation session. Insights feed back into position selection and strategy
/// choice for subsequent rescue attempts.
///
/// # Ring Buffer
///
/// Insights are stored in a fixed-capacity ring buffer. When full, the oldest
/// insight is evicted. Default capacity: 64 insights (~3KB).
///
/// # Cold Start
///
/// When empty (no insights), all query methods return neutral defaults:
/// - `success_rate()` → 0.5 (50%, no bias)
/// - `position_affinity()` → 0.0 (neutral)
/// - `should_skip_position()` → false (don't skip)
/// - `preferred_rules()` → empty (no preference)
///
/// This ensures the system gracefully degrades to Plan 026 baseline behavior.
///
/// # Thread Safety
///
/// `SessionKnowledge` is NOT thread-safe. It's designed for single-threaded
/// use within a speculative decode step. If parallel rescue is needed,
/// wrap in `Mutex` or use per-thread instances.
pub struct SessionKnowledge {
    /// Ring buffer of recorded insights.
    insights: Vec<RejectionInsight>,
    /// Maximum number of insights to retain.
    max_insights: usize,
    /// Write cursor (wraps around).
    write_pos: usize,
    /// Number of insights currently in buffer.
    count: usize,
    /// Per-rule success counts, indexed by `TokenRule::index()`.
    success_count_by_rule: [usize; 5],
    /// Per-rule failure counts, indexed by `TokenRule::index()`.
    fail_count_by_rule: [usize; 5],
    /// Last rescue result: was it a success?
    last_rescue_success: Option<bool>,
    /// Precomputed per-position statistics for O(1) lookups.
    position_stats: [PositionStats; MAX_POSITIONS],
}

impl SessionKnowledge {
    /// Create a new knowledge store with the given capacity.
    pub fn new(max_insights: usize) -> Self {
        Self {
            insights: Vec::with_capacity(max_insights),
            max_insights,
            write_pos: 0,
            count: 0,
            success_count_by_rule: [0; 5],
            fail_count_by_rule: [0; 5],
            last_rescue_success: None,
            position_stats: Default::default(),
        }
    }

    /// Create with default capacity (64 insights).
    pub fn with_default_capacity() -> Self {
        Self::new(64)
    }

    /// Create from PpotConfig.
    pub fn from_config(config: &PpotConfig) -> Self {
        Self::new(config.max_insights)
    }

    /// Record a rejection insight.
    ///
    /// Appends to the ring buffer, evicting the oldest entry if full.
    /// Updates per-rule success/failure counters.
    pub fn record(&mut self, insight: RejectionInsight) {
        let rule_idx = insight.rule.index();
        let accepted = insight.accepted;
        let position = insight.position;

        // If buffer is full, evict oldest and adjust counters
        if self.count >= self.max_insights {
            if let Some(evicted) = self.insights.get(self.write_pos) {
                let evicted_rule_idx = evicted.rule.index();
                let evicted_position = evicted.position;
                if evicted.accepted {
                    self.success_count_by_rule[evicted_rule_idx] =
                        self.success_count_by_rule[evicted_rule_idx].saturating_sub(1);
                } else {
                    self.fail_count_by_rule[evicted_rule_idx] =
                        self.fail_count_by_rule[evicted_rule_idx].saturating_sub(1);
                }
                // Evict from position stats
                if evicted_position < MAX_POSITIONS {
                    let ps = &mut self.position_stats[evicted_position];
                    if evicted.accepted {
                        ps.accepted = ps.accepted.saturating_sub(1);
                        ps.rule_success[evicted_rule_idx] =
                            ps.rule_success[evicted_rule_idx].saturating_sub(1);
                    } else {
                        ps.rejected = ps.rejected.saturating_sub(1);
                        ps.rule_fail[evicted_rule_idx] =
                            ps.rule_fail[evicted_rule_idx].saturating_sub(1);
                    }
                }
            }
            // Overwrite at write position
            if self.write_pos < self.insights.len() {
                self.insights[self.write_pos] = insight;
            }
        } else {
            // Buffer not yet full, just push
            self.insights.push(insight);
        }

        // Update counters for new insight (use captured `accepted`, not self.insights.last())
        if accepted {
            self.success_count_by_rule[rule_idx] += 1;
        } else {
            self.fail_count_by_rule[rule_idx] += 1;
        }

        // Update position stats for new insight
        if position < MAX_POSITIONS {
            let ps = &mut self.position_stats[position];
            if accepted {
                ps.accepted += 1;
                ps.rule_success[rule_idx] += 1;
            } else {
                ps.rejected += 1;
                ps.rule_fail[rule_idx] += 1;
            }
        }

        // Advance write cursor
        self.write_pos = (self.write_pos + 1) % self.max_insights.max(1);
        if self.count < self.max_insights {
            self.count += 1;
        }
    }

    /// Record a batch of rejection insights in one call.
    ///
    /// Thin wrapper that calls [`record`](Self::record) for each insight.
    /// Useful when collecting insights into a buffer before recording
    /// (avoids interleaving knowledge queries with writes).
    #[inline]
    pub fn record_batch(&mut self, insights: impl Iterator<Item = RejectionInsight>) {
        for insight in insights {
            self.record(insight);
        }
    }

    /// Record whether the last rescue attempt succeeded or failed.
    ///
    /// Used by adaptive threshold adjustment (TRT: models switch strategy
    /// more after failure — 82% — than success — 74%).
    pub fn record_rescue_result(&mut self, success: bool) {
        self.last_rescue_success = Some(success);
    }

    /// Whether any insights have been recorded.
    pub fn has_insights(&self) -> bool {
        self.count > 0
    }

    /// Number of insights currently in the ring buffer.
    pub fn insight_count(&self) -> usize {
        self.count
    }

    /// Per-rule success rate: `successes / (successes + failures)`.
    ///
    /// Returns 0.5 (neutral) when no insights exist for the rule.
    pub fn success_rate(&self, rule: TokenRule) -> f32 {
        let idx = rule.index();
        let successes = self.success_count_by_rule[idx];
        let failures = self.fail_count_by_rule[idx];
        let total = successes + failures;
        if total == 0 {
            0.5 // neutral: no bias
        } else {
            successes as f32 / total as f32
        }
    }

    /// How often resampling this position succeeds.
    ///
    /// Returns the fraction of accepted insights at this position.
    /// Returns 0.0 (neutral, no priority) when no insights exist for the position.
    pub fn position_affinity(&self, position: usize) -> f32 {
        if position < MAX_POSITIONS {
            let stats = &self.position_stats[position];
            let total = stats.accepted + stats.rejected;
            if total == 0 {
                0.0 // no data: neutral priority
            } else {
                stats.accepted as f32 / total as f32
            }
        } else {
            // Fallback for out-of-range positions
            let mut accepted = 0usize;
            let mut total = 0usize;
            for insight in &self.insights {
                if insight.position == position {
                    total += 1;
                    if insight.accepted {
                        accepted += 1;
                    }
                }
            }
            if total == 0 {
                0.0
            } else {
                accepted as f32 / total as f32
            }
        }
    }

    /// Whether this position should be skipped in resampling.
    ///
    /// Returns `true` if there are >= `min_failures` consecutive failures at
    /// this position with NO successes. Default threshold: 3 failures.
    ///
    /// TRT finding: positions that consistently fail should be deprioritized
    /// in favor of positions with historical success.
    pub fn should_skip_position(&self, position: usize) -> bool {
        self.should_skip_position_with_threshold(position, 3)
    }

    /// Skip check with configurable failure threshold.
    pub fn should_skip_position_with_threshold(
        &self,
        position: usize,
        min_failures: usize,
    ) -> bool {
        if position < MAX_POSITIONS {
            // When no accepted insights exist, all failures are consecutive.
            // When any accepted insight exists, we never skip regardless.
            let stats = &self.position_stats[position];
            stats.accepted == 0 && stats.rejected >= min_failures
        } else {
            // Fallback for out-of-range positions
            let mut consecutive_fails = 0usize;
            let mut has_success = false;
            for insight in &self.insights {
                if insight.position == position {
                    if insight.accepted {
                        has_success = true;
                        consecutive_fails = 0;
                    } else {
                        consecutive_fails += 1;
                    }
                }
            }
            !has_success && consecutive_fails >= min_failures
        }
    }

    /// Preferred rules for a position, sorted by success rate descending.
    ///
    /// Returns rules that have been tried at this position with positive results,
    /// ordered from most to least successful. Returns `[None; 5]` if no insights
    /// exist for the position.
    ///
    /// Stack-allocated (no heap) — max 5 rules (one per `TokenRule` variant).
    pub fn preferred_rules(&self, position: usize) -> [Option<TokenRule>; 5] {
        let mut rule_stats: [(TokenRule, f32); 5] = [
            (TokenRule::Digit, 0.0),
            (TokenRule::Compare, 0.0),
            (TokenRule::Arithmetic, 0.0),
            (TokenRule::Augment, 0.0),
            (TokenRule::All, 0.0),
        ];

        if position < MAX_POSITIONS {
            // O(1) from precomputed per-position stats
            let stats = &self.position_stats[position];
            for (i, entry) in rule_stats.iter_mut().enumerate() {
                entry.1 = stats.rule_success[i] as f32 - stats.rule_fail[i] as f32 * 0.1;
            }
        } else {
            // Fallback for out-of-range positions: O(n) scan
            for insight in &self.insights {
                if insight.position == position {
                    let idx = insight.rule.index();
                    if insight.accepted {
                        rule_stats[idx].1 += 1.0;
                    } else {
                        rule_stats[idx].1 -= 0.1;
                    }
                }
            }
        }

        // Sort by score descending
        rule_stats.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Fill fixed-size array with rules having positive score
        let mut result = [None; 5];
        let mut out_idx = 0;
        for (rule, score) in &rule_stats {
            if *score > 0.0 && out_idx < 5 {
                result[out_idx] = Some(*rule);
                out_idx += 1;
            }
        }
        result
    }

    /// Compute adaptive entropy threshold based on last rescue result.
    ///
    /// TRT finding: models switch strategy more after failure (82%) than
    /// success (74%). We capture this by:
    /// - **Lowering** threshold after failure → explore more positions
    /// - **Raising** threshold after success → focus on fewer, higher-quality positions
    ///
    /// Falls back to config's base threshold on cold start (no history).
    pub fn adaptive_threshold(&self, config: &PpotConfig) -> f32 {
        match self.last_rescue_success {
            None => config.entropy_threshold,
            Some(true) => {
                let raised = config.entropy_threshold + config.threshold_raise_on_success;
                config.clamp_threshold(raised)
            }
            Some(false) => {
                let lowered = config.entropy_threshold - config.threshold_lower_on_fail;
                config.clamp_threshold(lowered)
            }
        }
    }

    /// Reset all knowledge (e.g., between generation sessions).
    pub fn reset(&mut self) {
        self.insights.clear();
        self.write_pos = 0;
        self.count = 0;
        self.success_count_by_rule = [0; 5];
        self.fail_count_by_rule = [0; 5];
        self.last_rescue_success = None;
        self.position_stats = Default::default();
    }

    /// Get an iterator over current insights (most recent first).
    pub fn insights(&self) -> impl Iterator<Item = &RejectionInsight> {
        // Read from most recent backwards
        let count = self.count;
        let max = self.max_insights;
        let start = if count < max { 0 } else { self.write_pos };

        self.insights
            .iter()
            .cycle()
            .skip(start)
            .take(count.min(self.insights.len()))
    }

    /// Approximate memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        // RejectionInsight with optimal field packing:
        // position(8) + original_token(8) + attempted_token(8) + entropy(4)
        // + rule(1) + error_kind(1) + accepted(1) + padding(1) ≈ ~32 bytes
        self.count * 32 + std::mem::size_of::<Self>()
    }
}

impl Default for SessionKnowledge {
    fn default() -> Self {
        Self::with_default_capacity()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_insight(position: usize, rule: TokenRule, accepted: bool) -> RejectionInsight {
        RejectionInsight {
            position,
            rule,
            original_token: 0,
            attempted_token: if accepted { 1 } else { 2 },
            error_kind: None,
            entropy: 0.5,
            accepted,
        }
    }

    #[test]
    fn test_knowledge_empty_state() {
        let k = SessionKnowledge::new(64);
        assert!(!k.has_insights());
        assert_eq!(k.insight_count(), 0);
        assert!(!k.should_skip_position(0));
        assert_eq!(k.success_rate(TokenRule::Digit), 0.5);
        assert_eq!(k.position_affinity(0), 0.0);
        assert!(k.preferred_rules(0).iter().all(|r| r.is_none()));
    }

    #[test]
    fn test_knowledge_record_and_query() {
        let mut k = SessionKnowledge::new(64);

        // Record 3 successes and 1 failure for Digit at position 0
        k.record(make_insight(0, TokenRule::Digit, true));
        k.record(make_insight(0, TokenRule::Digit, true));
        k.record(make_insight(0, TokenRule::Digit, true));
        k.record(make_insight(0, TokenRule::Digit, false));

        assert!(k.has_insights());
        assert_eq!(k.insight_count(), 4);
        assert!((k.success_rate(TokenRule::Digit) - 0.75).abs() < 0.01);
        assert!((k.position_affinity(0) - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_knowledge_ring_buffer_eviction() {
        let mut k = SessionKnowledge::new(4); // tiny buffer

        // Fill buffer
        k.record(make_insight(0, TokenRule::Digit, true));
        k.record(make_insight(1, TokenRule::Arithmetic, false));
        k.record(make_insight(2, TokenRule::Compare, true));
        k.record(make_insight(3, TokenRule::Augment, false));

        assert_eq!(k.insight_count(), 4);

        // Add one more → evicts oldest (pos 0, Digit, success)
        k.record(make_insight(4, TokenRule::All, true));

        assert_eq!(k.insight_count(), 4);

        // Digit should have lost one success
        assert_eq!(k.success_count_by_rule[TokenRule::Digit.index()], 0);
        assert_eq!(k.fail_count_by_rule[TokenRule::Digit.index()], 0);

        // All should have the new success
        assert_eq!(k.success_count_by_rule[TokenRule::All.index()], 1);
    }

    #[test]
    fn test_knowledge_should_skip_position() {
        let mut k = SessionKnowledge::new(64);

        // Position 0: 3 failures, no success → should skip
        k.record(make_insight(0, TokenRule::Digit, false));
        k.record(make_insight(0, TokenRule::Arithmetic, false));
        k.record(make_insight(0, TokenRule::Compare, false));

        assert!(k.should_skip_position(0));
        assert!(!k.should_skip_position(1)); // no data → don't skip

        // Position 1: 2 failures then 1 success → should NOT skip
        k.record(make_insight(1, TokenRule::Digit, false));
        k.record(make_insight(1, TokenRule::Arithmetic, false));
        k.record(make_insight(1, TokenRule::Compare, true));

        assert!(!k.should_skip_position(1));
    }

    #[test]
    fn test_knowledge_preferred_rules() {
        let mut k = SessionKnowledge::new(64);

        // Position 0: Digit succeeds 2x, Arithmetic fails 2x
        k.record(make_insight(0, TokenRule::Digit, true));
        k.record(make_insight(0, TokenRule::Digit, true));
        k.record(make_insight(0, TokenRule::Arithmetic, false));
        k.record(make_insight(0, TokenRule::Arithmetic, false));

        let preferred = k.preferred_rules(0);
        let has_preferred = preferred.iter().any(|r| r.is_some());
        assert!(has_preferred, "should have at least one preferred rule");
        assert_eq!(
            preferred[0],
            Some(TokenRule::Digit),
            "Digit should be preferred"
        );
    }

    #[test]
    fn test_knowledge_adaptive_threshold_cold_start() {
        let k = SessionKnowledge::new(64);
        let config = PpotConfig::default();

        // No history → use config threshold
        let threshold = k.adaptive_threshold(&config);
        assert!((threshold - config.entropy_threshold).abs() < 0.001);
    }

    #[test]
    fn test_knowledge_adaptive_threshold_after_success() {
        let mut k = SessionKnowledge::new(64);
        let config = PpotConfig::default();

        k.record_rescue_result(true);
        let threshold = k.adaptive_threshold(&config);
        let expected = config.entropy_threshold + config.threshold_raise_on_success;
        assert!(
            (threshold - expected).abs() < 0.001,
            "threshold should be raised after success"
        );
    }

    #[test]
    fn test_knowledge_adaptive_threshold_after_failure() {
        let mut k = SessionKnowledge::new(64);
        let config = PpotConfig::default();

        k.record_rescue_result(false);
        let threshold = k.adaptive_threshold(&config);
        let expected = config.entropy_threshold - config.threshold_lower_on_fail;
        assert!(
            (threshold - expected).abs() < 0.001,
            "threshold should be lowered after failure"
        );
    }

    #[test]
    fn test_knowledge_adaptive_threshold_clamped() {
        let mut k = SessionKnowledge::new(64);
        let mut config = PpotConfig::default();
        config.entropy_threshold = 0.95;
        config.threshold_raise_on_success = 0.2;
        config.entropy_threshold_max = 1.0;

        k.record_rescue_result(true);
        let threshold = k.adaptive_threshold(&config);
        assert!(
            (threshold - 1.0).abs() < 0.001,
            "threshold should be clamped to max"
        );
    }

    #[test]
    fn test_knowledge_reset() {
        let mut k = SessionKnowledge::new(64);
        k.record(make_insight(0, TokenRule::Digit, true));
        k.record_rescue_result(false);

        assert!(k.has_insights());

        k.reset();

        assert!(!k.has_insights());
        assert_eq!(k.insight_count(), 0);
        assert!(k.last_rescue_success.is_none());
        assert_eq!(k.success_count_by_rule, [0; 5]);
        assert_eq!(k.fail_count_by_rule, [0; 5]);
    }

    #[test]
    fn test_knowledge_memory_usage() {
        let mut k = SessionKnowledge::new(64);
        // [PositionStats; 16] index adds ~1.5KB to struct size
        assert!(k.memory_usage() < 3072, "empty knowledge should be < 3KB");

        for i in 0..64 {
            k.record(make_insight(i, TokenRule::Digit, i % 2 == 0));
        }

        assert!(
            k.memory_usage() < 8192,
            "full knowledge should be < 8KB, got {}",
            k.memory_usage()
        );
    }

    #[test]
    fn test_knowledge_per_rule_independent() {
        let mut k = SessionKnowledge::new(64);

        // Digit: 100% success
        k.record(make_insight(0, TokenRule::Digit, true));
        k.record(make_insight(0, TokenRule::Digit, true));

        // Arithmetic: 0% success
        k.record(make_insight(1, TokenRule::Arithmetic, false));
        k.record(make_insight(1, TokenRule::Arithmetic, false));

        assert!((k.success_rate(TokenRule::Digit) - 1.0).abs() < 0.01);
        assert!((k.success_rate(TokenRule::Arithmetic) - 0.0).abs() < 0.01);
        assert_eq!(k.success_rate(TokenRule::Compare), 0.5); // no data → neutral
    }

    #[test]
    fn test_knowledge_should_skip_with_threshold() {
        let mut k = SessionKnowledge::new(64);

        // Only 2 failures at position 0
        k.record(make_insight(0, TokenRule::Digit, false));
        k.record(make_insight(0, TokenRule::Digit, false));

        // Default threshold (3) → should NOT skip
        assert!(!k.should_skip_position(0));

        // Custom threshold (2) → should skip
        assert!(k.should_skip_position_with_threshold(0, 2));
    }

    #[test]
    fn test_knowledge_ring_buffer_reuses_evicted_slots() {
        let mut k = SessionKnowledge::new(3);

        // Fill: [A, B, C]
        k.record(make_insight(0, TokenRule::Digit, true)); // A
        k.record(make_insight(1, TokenRule::Arithmetic, false)); // B
        k.record(make_insight(2, TokenRule::Compare, true)); // C

        // Add D → evicts A: [D, B, C]
        k.record(make_insight(3, TokenRule::Augment, false)); // D

        // Add E → evicts B: [D, E, C]
        k.record(make_insight(4, TokenRule::All, true)); // E

        // Digit rule: evicted A was a success
        assert_eq!(k.success_count_by_rule[TokenRule::Digit.index()], 0);
        assert_eq!(k.fail_count_by_rule[TokenRule::Digit.index()], 0);

        // Arithmetic rule: evicted B was a failure
        assert_eq!(k.success_count_by_rule[TokenRule::Arithmetic.index()], 0);
        assert_eq!(k.fail_count_by_rule[TokenRule::Arithmetic.index()], 0);

        // Compare rule: C was a success (still in buffer)
        assert_eq!(k.success_count_by_rule[TokenRule::Compare.index()], 1);

        // Augment rule: D was a failure (still in buffer)
        assert_eq!(k.fail_count_by_rule[TokenRule::Augment.index()], 1);

        // All rule: E was a success (still in buffer)
        assert_eq!(k.success_count_by_rule[TokenRule::All.index()], 1);

        assert_eq!(k.insight_count(), 3);
    }
}
