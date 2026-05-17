//! Go game integration — AutoGo API bridge, GameState, and tournament infrastructure.
//!
//! Plan 065: AutoGo Distillation
//!
//! ## Modules
//!
//! - [`types`] — `GoAction`, `GoCell` enums
//! - [`state`] — `GoState` board with full Go logic + `GameState` trait impl + `GoHeuristic`
//! - [`autogo_client`] — REST API client for AutoGo's `play.py` server
//! - [`replay`] — Game recording and deterministic playback
//! - [`players`] — `GoPlayer` trait + 6 AI player implementations (Plan 065 Phase 2)

pub mod autogo_client;
pub mod players;
pub mod replay;
pub mod state;
pub mod types;

// ── Re-exports ─────────────────────────────────────────────────

// Types
pub use types::{GoAction, GoCell};

// State
pub use state::{DEFAULT_KOMI, GoHeuristic, GoState};

// Replay
pub use replay::{GoReplay, MoveRecord, ReplayError};

// API Client
pub use autogo_client::{AutoGoClient, AutoGoError, AutoGoGameState};

// Players
pub use players::{
    GoGZeroPlayer, GoGreedyPlayer, GoHLPlayer, GoMctsPlayer, GoMoveCategory, GoPlayer,
    GoRandomPlayer, GoTemplate, GoValidatorPlayer,
};
