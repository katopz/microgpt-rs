//! PPoT: Probabilistic Programs of Thought — Logit-Parameterized CPU Resampling.
//!
//! Distilled from "Probabilistic Programs of Thought" (arXiv:2604.17290) and
//! "Test-time Recursive Thinking" (arXiv:2602.03094).
//!
//! After DFlash produces marginals, this module identifies high-entropy positions
//! and resamples variant programs using **only CPU** — no additional GPU forward
//! passes. Resampled paths are verified through existing `ScreeningPruner`.
//!
//! # Architecture
//!
//! ```text
//! DFlash → DDTree → Verify
//!                 ↓ (all rejected)
//!           ┌─────────────────────────────────┐
//!           │     PPoT Rescue (CPU only)       │
//!           │                                 │
//!           │  1. Read marginals              │
//!           │  2. Calculate per-position H(i) │
//!           │  3. Identify high-H positions   │
//!           │  4. For m samples:              │
//!           │     a. Resample positions       │
//!           │     b. Screen via Pruner        │
//!           │     c. If valid → return path   │
//!           │  5. All invalid → greedy fallback│
//!           └─────────────────────────────────┘
//! ```
//!
//! # Two Layers
//!
//! - **Plan 026 (Baseline):** Random resampling with entropy-based position selection.
//!   Use [`ppot_rescue()`] for simple rescue.
//!
//! - **Plan 027 (Adaptive):** TRT-inspired rejection memory, strategy cycling,
//!   self-consistency ranking. Use [`ppot_rescue_adaptive()`] for adaptive rescue.
//!
//! # Feature Gate
//!
//! All PPoT code is behind the `ppot` feature flag. When disabled, zero overhead
//! is incurred on the speculative decoding path.

pub mod entropy;
pub mod knowledge;
pub mod rank;
pub mod resample;
pub mod types;

// ── Re-exports: Plan 026 Core ──────────────────────────────────

pub use entropy::{
    identify_high_entropy_positions, identify_high_entropy_positions_into,
    identify_positions_by_rule, identify_positions_by_rule_into, token_entropy,
};

pub use resample::{
    ppot_resample, ppot_resample_different_value, ppot_resample_with_support, ppot_rescue,
};

pub use types::{PpotConfig, TokenRule};

// ── Re-exports: Plan 027 Adaptive ──────────────────────────────

pub use entropy::{identify_positions_adaptive, identify_positions_adaptive_into};

pub use knowledge::{ErrorKind, RejectionInsight, SessionKnowledge};

pub use rank::{rank_by_consistency, rank_by_consistency_weighted, select_best_variant};

pub use resample::{ppot_resample_multi_strategy, ppot_rescue_adaptive};

// ── Re-exports: Plan 036 Review Loop ───────────────────────────

#[cfg(feature = "bandit")]
pub use resample::ppot_rescue_reviewed;
