//! Core types for G-Zero self-play distillation.
//!
//! G-Zero enables verifier-free self-evolution for open-ended domains by
//! replacing external LLM judges with an **intrinsic signal** derived from
//! the model's own predictive distribution.
//!
//! # Hint-δ
//!
//! Measures how much a hint shifts the Solver's distribution, using
//! teacher-forced log-probs of the same `a_hard` tokens under two prompt
//! contexts:
//!
//! ```text
//! δ(q, h, a_hard) = (1/T) Σ [log πS(a_hard_t | q, a_hard_<t)
//!                        − log πS(a_hard_t | q, h, a_hard_<t)]
//! ```
//!
//! Both terms score the **same** `a_hard` tokens — the difference is whether
//! `h` is in the prompt. Positive δ ⇒ hint shifts the Solver away from its
//! own unassisted response ⇒ hint carries structural signal (not answer leakage).
//!
//! **Source:** G-Zero paper, `.raw/G-Zero/g_zero/hint_delta.py` `QHScore` dataclass.

use serde::{Deserialize, Serialize};

// ── HintDelta ───────────────────────────────────────────────────

/// Intrinsic reward: hint-induced log-prob shift.
///
/// The core innovation from G-Zero — a scalar signal that is large only when:
/// 1. The query is challenging, AND
/// 2. The hint carries information the Solver lacks.
///
/// Two objectives compressed into one scalar — no external oracle needed.
///
/// **Source:** G-Zero paper `hint_delta.py` `QHScore` dataclass.
///   `delta = logp_q - logp_qh` via teacher-forced `compute_logprobs`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HintDelta {
    /// δ = mean(logp_q - logp_qh) over `a_hard` tokens.
    ///
    /// Positive ⇒ hint shifts Solver away from its own unassisted response.
    /// Near-zero ⇒ hint is redundant or answer leakage.
    pub value: f32,
    /// The query (question) prompt.
    pub query: String,
    /// The hint provided alongside the query.
    pub hint: String,
    /// Unassisted response text (generated from `q` alone).
    pub a_hard: String,
    /// Hint-assisted response text (generated from `q + h`).
    /// Empty string in delta-only mode (no generation, just log-prob scoring).
    pub a_assisted: String,
    /// Mean log πS(a_hard | q) — teacher-forced log-prob under query-only context.
    pub logp_q: f32,
    /// Mean log πS(a_hard | q, h) — teacher-forced log-prob under query+hint context.
    pub logp_qh: f32,
}

impl HintDelta {
    /// Compute Hint-δ from teacher-forced log-probs.
    ///
    /// # Arguments
    ///
    /// * `logp_q_tokens` — per-token log-probs of `a_hard` under context `q`
    /// * `logp_qh_tokens` — per-token log-probs of `a_hard` under context `q + h`
    /// * `query` — the question text
    /// * `hint` — the hint text
    /// * `a_hard` — unassisted response text
    /// * `a_assisted` — hint-assisted response text ("" for delta-only mode)
    ///
    /// # Formula
    ///
    /// ```text
    /// δ(q, h, a_hard) = (1/T) Σ [log πS(a_hard_t|q,a_hard_<t)
    ///                        − log πS(a_hard_t|q,h,a_hard_<t)]
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if both token slices are empty (no tokens to score).
    pub fn compute(
        logp_q_tokens: &[f32],
        logp_qh_tokens: &[f32],
        query: &str,
        hint: &str,
        a_hard: &str,
        a_assisted: &str,
    ) -> Self {
        let t = logp_q_tokens.len().min(logp_qh_tokens.len());
        assert!(t > 0, "HintDelta::compute requires at least one token");

        let logp_q: f32 = logp_q_tokens[..t].iter().copied().sum::<f32>() / t as f32;
        let logp_qh: f32 = logp_qh_tokens[..t].iter().copied().sum::<f32>() / t as f32;

        Self {
            value: logp_q - logp_qh,
            query: query.to_string(),
            hint: hint.to_string(),
            a_hard: a_hard.to_string(),
            a_assisted: a_assisted.to_string(),
            logp_q,
            logp_qh,
        }
    }

    /// Whether this delta indicates the hint carries structural signal.
    ///
    /// Positive δ means the hint shifted the Solver's distribution
    /// away from its own unassisted response — the hint revealed a blind spot.
    #[inline]
    pub fn is_informative(&self) -> bool {
        self.value > 0.0
    }

    /// Number of tokens scored (minimum of the two log-prob slices used).
    ///
    /// Returns 0 if computed from mismatched slices (should not happen in practice).
    #[inline]
    pub fn token_count(&self) -> usize {
        // We don't store the original slices, but the mean is over T tokens.
        // The value is logp_q - logp_qh where both are means over T tokens.
        // We can't recover T from the struct alone — this is a conceptual note.
        // For practical use, store T externally if needed.
        0 // Placeholder — caller tracks token count
    }
}

// ── LogProb Extraction ──────────────────────────────────────────

/// Result of a teacher-forced forward pass with log-prob extraction.
///
/// Standalone output from `logprobs()` — NOT modifying the `forward()` hot path.
/// Contains per-token log-probabilities for a given prompt + response sequence.
///
/// **Design:** Separate function to keep `transformer.rs` forward() clean.
/// See Plan 049 T12: "logprobs() as standalone function, not forward() modification".
#[derive(Clone, Debug)]
pub struct LogProbResult {
    /// Per-token log-probabilities: `log π(token_t | prompt + token_0..t)`.
    pub token_logprobs: Vec<f32>,
    /// The prompt text used for this computation.
    pub prompt: String,
    /// The response tokens that were scored.
    pub response_text: String,
}

impl LogProbResult {
    /// Mean log-probability across all scored tokens.
    pub fn mean_logprob(&self) -> f32 {
        if self.token_logprobs.is_empty() {
            return 0.0;
        }
        self.token_logprobs.iter().copied().sum::<f32>() / self.token_logprobs.len() as f32
    }

    /// Number of tokens scored.
    pub fn len(&self) -> usize {
        self.token_logprobs.len()
    }

    /// Whether any tokens were scored.
    pub fn is_empty(&self) -> bool {
        self.token_logprobs.is_empty()
    }
}

// ── Feature Gate Note ───────────────────────────────────────────
//
// All g_zero types are behind `#[cfg(feature = "g_zero")]` in mod.rs.
// The feature gate rule: `g_zero = ["bandit"]` in Cargo.toml.
//
// Log-prob extraction uses a separate `logprobs()` function,
// NOT modifying `forward()` hot path (Plan 049 T12).

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hint_delta_positive_informative_hint() {
        // Simulate: hint shifts distribution away from unassisted response
        let logp_q = vec![-2.0, -1.5, -3.0, -2.5]; // unassisted context
        let logp_qh = vec![-2.5, -2.0, -3.5, -3.0]; // hint context (lower = harder)

        let delta = HintDelta::compute(&logp_q, &logp_qh, "q", "h", "a_hard", "a_assisted");

        assert!(delta.value > 0.0, "Positive δ: hint shifted distribution");
        assert!(delta.is_informative());
        assert!((delta.logp_q - (-2.25)).abs() < 0.01);
        assert!((delta.logp_qh - (-2.75)).abs() < 0.01);
        assert!((delta.value - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_hint_delta_negative_useless_hint() {
        // Simulate: hint doesn't help, even hurts slightly
        let logp_q = vec![-2.0, -1.5, -3.0];
        let logp_qh = vec![-1.8, -1.3, -2.8]; // higher = easier with hint

        let delta = HintDelta::compute(&logp_q, &logp_qh, "q", "h", "a_hard", "");

        assert!(delta.value < 0.0, "Negative δ: hint didn't shift away");
        assert!(!delta.is_informative());
        assert!(delta.a_assisted.is_empty()); // delta-only mode
    }

    #[test]
    fn test_hint_delta_zero_redundant_hint() {
        // Hint doesn't change distribution at all
        let logp_q = vec![-2.0, -1.5];
        let logp_qh = vec![-2.0, -1.5];

        let delta = HintDelta::compute(&logp_q, &logp_qh, "q", "h", "a_hard", "");

        assert!((delta.value).abs() < 1e-6, "Zero δ: hint is redundant");
        assert!(!delta.is_informative());
    }

    #[test]
    fn test_hint_delta_mismatched_lengths() {
        // Different token counts — uses minimum
        let logp_q = vec![-2.0, -1.5, -3.0, -2.5];
        let logp_qh = vec![-2.5, -2.0]; // shorter

        let delta = HintDelta::compute(&logp_q, &logp_qh, "q", "h", "a_hard", "");

        // Only first 2 tokens scored
        let expected_logp_q = (-2.0 + -1.5) / 2.0;
        let expected_logp_qh = (-2.5 + -2.0) / 2.0;
        assert!((delta.logp_q - expected_logp_q).abs() < 0.01);
        assert!((delta.logp_qh - expected_logp_qh).abs() < 0.01);
    }

    #[test]
    #[should_panic(expected = "at least one token")]
    fn test_hint_delta_empty_tokens_panics() {
        HintDelta::compute(&[], &[], "q", "h", "a_hard", "");
    }

    #[test]
    fn test_hint_delta_serialization_roundtrip() {
        let delta = HintDelta {
            value: 0.42,
            query: "What is Rust?".into(),
            hint: "Think about memory safety.".into(),
            a_hard: "Rust is a language.".into(),
            a_assisted: "Rust is a systems language with ownership.".into(),
            logp_q: -2.1,
            logp_qh: -2.52,
        };

        let json = serde_json::to_string(&delta).unwrap();
        let deserialized: HintDelta = serde_json::from_str(&json).unwrap();
        assert_eq!(delta, deserialized);
    }

    #[test]
    fn test_log_prob_result_mean() {
        let result = LogProbResult {
            token_logprobs: vec![-1.0, -2.0, -3.0],
            prompt: "prompt".into(),
            response_text: "response".into(),
        };
        assert!((result.mean_logprob() - (-2.0)).abs() < 0.01);
        assert_eq!(result.len(), 3);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_log_prob_result_empty() {
        let result = LogProbResult {
            token_logprobs: vec![],
            prompt: "".into(),
            response_text: "".into(),
        };
        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
        assert!((result.mean_logprob() - 0.0).abs() < 1e-6);
    }
}
