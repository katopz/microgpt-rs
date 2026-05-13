//! G-Zero self-play distillation — verifier-free self-evolution for open-ended domains.
//!
//! G-Zero replaces external LLM judges with an **intrinsic signal** (Hint-δ) derived
//! from the model's own predictive distribution. Two paths:
//!
//! - **Modelless** (Phase 1): δ → AbsorbCompress + BanditPruner — no gradient updates
//! - **Model-based** (Phase 2): δ → DPO/GRPO weight updates (future, in riir-gpu)
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "g_zero")]`.
//! Feature: `g_zero = ["bandit"]` in `Cargo.toml`.
//!
//! # Components
//!
//! - [`HintDelta`] — core intrinsic reward signal (log-prob shift)
//! - [`DeltaGatedAbsorbCompress`] — absorb only when δ reveals blind spot
//! - [`DeltaBanditPruner`] — δ as dense reward for bandit arm selection
//! - [`TemplateProposer`] — rule-based query-hint generation
//!
//! **Source:** [G-Zero: Self-Play for Open-Ended Generation from Zero Data](https://arxiv.org/pdf/2605.09959) — Huang et al., 2026

pub mod delta_absorb;
pub mod delta_bandit;
pub mod template_proposer;
pub mod types;

pub use delta_absorb::{DeltaGatedAbsorbCompress, DeltaGatedConfig};
pub use delta_bandit::DeltaBanditPruner;
pub use template_proposer::{GeneratedPair, QueryTemplate, TemplateProposer};
pub use types::{HintDelta, LogProbResult};
