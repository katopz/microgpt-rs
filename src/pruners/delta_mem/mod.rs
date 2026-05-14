//! δ-mem modelless distillation: associative bandit memory.
//!
//! Distilled from δ-mem (arXiv 2605.12357), verified against source:
//!   `delta_impl.py` L1895-1938 (_memory_affine_scan_torch)
//!
//! # Modelless Adaptation
//!
//! Paper uses learned projections (W_mq, W_mk, W_mv, W∆q, W∆o).
//! We replace them with feature hashing (FeatureHasher).
//! The delta-rule update is identical — prediction error drives learning.
//!
//! # Source Code Mapping
//!
//! | Paper Component    | Source Location                    | Our Equivalent              |
//! |--------------------|------------------------------------|-----------------------------|
//! | OSAM state S       | DeltaMemAttention.delta_state      | DeltaMemoryState.state      |
//! | Read S·q           | L1921: einsum("bij,bj->bi")       | DeltaMemoryState::read()    |
//! | Write S'=(1-β)S-β·pred⊗k+β·v⊗k | L1923-1929      | DeltaMemoryState::write()   |
//! | Gate β=sigmoid(W·x+b) | L917-925 with couple_lambda    | Heuristic from δ statistics  |
//! | normalize_qk       | L805-814: L2_norm(tanh(...))     | FeatureHasher::hash_key()   |
//! | delta_o correction | L2283: attn_output + delta_o      | MemorySteeredPruner          |
//! | MSW (4 heads)      | L795-803: reshape + scan          | MultiDomainMemory            |
//! | SSW (message_mean) | L2150-2215: avg then single write | write_segment()              |

pub mod hash;
pub mod multi;
pub mod multi_pruner;
pub mod pruner;
pub mod state;

pub use hash::{ContextFeatures, FeatureHasher, OutcomeFeatures};
pub use multi::{AggregationStrategy, MultiDomainMemory};
pub use multi_pruner::MultiDomainMemoryPruner;
pub use pruner::{CorrectionMode, MemorySteeredPruner, WriteGranularity};
pub use state::{DeltaMemoryConfig, DeltaMemorySnapshot, DeltaMemoryState};
