//! Domain-specific constraint pruners for the DDTree search engine.

pub mod dungeon_pathfinder;
pub mod dungeon_pruner;
pub mod map_generator;
pub mod pathfinder;
pub mod tactical_pruner;

pub use dungeon_pathfinder::{
    DungeonAction, MultiFloorBlocked, MultiFloorTarget, enumerate_multifloor_targets,
    find_path_multifloor, find_path_on_floor,
};
pub use dungeon_pruner::{DungeonMap, DungeonPruner, DungeonState, FloorGrid, StairConnection};
pub use map_generator::{GeneratedDungeon, GeneratedMap, MapGenerator};
pub use pathfinder::{Target, enumerate_targets, find_distance, find_path, reachable_positions};
pub use tactical_pruner::{GameState, TacticalPruner};

#[cfg(feature = "bomber")]
pub mod bomber;

pub mod game_state; // Always compiled — GameState trait has no bevy_ecs dependency (G1 fix, Plan 065)

#[cfg(feature = "sudoku")]
pub mod sudoku_pruner;

#[cfg(feature = "sudoku")]
pub use sudoku_pruner::SudokuPruner;

#[cfg(feature = "bandit")]
pub mod absorb_compress;

#[cfg(feature = "bandit")]
pub mod bandit;

#[cfg(feature = "bandit")]
pub mod hot_swap;

#[cfg(feature = "bandit")]
pub mod regression;

#[cfg(feature = "bandit")]
pub mod review_metrics;

#[cfg(feature = "bandit")]
pub mod trial_log;

#[cfg(feature = "bandit")]
pub use absorb_compress::{AbsorbCompress, AbsorbCompressLayer, CompressConfig};

#[cfg(feature = "bandit")]
pub use bandit::{
    BanditEnv, BanditEvent, BanditPruner, BanditResult, BanditSession, BanditStats, BanditStrategy,
    BernoulliEnv, GaussianEnv, SharedBanditStats,
};

#[cfg(feature = "bandit")]
pub use review_metrics::{EntropyAnomalySummary, ReviewMetrics, ReviewStrategy, ReviewSummary};

#[cfg(feature = "bandit")]
pub use hot_swap::HotSwapPruner;

#[cfg(feature = "bandit")]
pub use regression::{GoldenTrace, RegressionResult, RegressionSuite, ReplayReward};

#[cfg(feature = "bandit")]
pub use trial_log::{SharedTrialLog, TrialLog, TrialRecord, TrialSummary};

#[cfg(feature = "g_zero")]
pub mod g_zero;

#[cfg(feature = "g_zero")]
pub use g_zero::{
    BomberTemplate, BomberTemplateProposer, DeltaBanditPruner, DeltaGatedAbsorbCompress,
    DeltaGatedConfig, GeneratedPair, HintDelta, LogProbResult, QueryTemplate, TemplateProposer,
};

#[cfg(feature = "stepcode")]
pub mod stepcode;

#[cfg(feature = "stepcode")]
pub use stepcode::{PathStep, ShapedPath, path_consistency, shape_path};

#[cfg(feature = "delta_mem")]
pub mod delta_mem;

#[cfg(feature = "delta_mem")]
pub use delta_mem::{
    AggregationStrategy, ContextFeatures, CorrectionMode, DeltaMemoryConfig, DeltaMemorySnapshot,
    DeltaMemoryState, FeatureHasher, MemorySteeredPruner, MultiDomainMemory,
    MultiDomainMemoryPruner, OutcomeFeatures, WriteGranularity,
};

#[cfg(all(feature = "g_zero", feature = "fft"))]
pub use g_zero::{FFTTemplate, FFTTemplateProposer};

#[cfg(all(feature = "fft", feature = "g_zero"))]
pub use fft::GZeroFFTPlayer;

#[cfg(feature = "bomber")]
pub use bomber::{
    ArenaGrid, BomberAction, BomberPlayer, GridPos, ScoreBoard, TickCounter, run_tick,
    spawn_players,
};

pub use game_state::{ActionSpaceLog, StateHeuristic, mcts_search};

#[cfg(all(feature = "game_state", feature = "bomber"))]
pub use game_state::{BombSnapshot, BomberHeuristic, BomberState, PlayerSnapshot};

#[cfg(feature = "monopoly")]
pub mod monopoly;

#[cfg(feature = "monopoly")]
pub use monopoly::{
    Board, BoardSquare, CardDeck, CardEffect, DecisionContext, GameConfig, GamePhase, JailDecision,
    JailReason, MonopolyPlayer, Owned, Player, Property, PropertyGroup, ReleaseMethod, Statistics,
    Strategy, TaxKind, TradeOffer, TradeResponse, TurnPhase, build_board, shuffle_decks,
    square_kind, square_name,
};

#[cfg(feature = "go")]
pub mod go;

#[cfg(feature = "go")]
pub use go::{
    AutoGoClient, AutoGoError, AutoGoGameState, DEFAULT_KOMI, GoAction, GoCell, GoGZeroPlayer,
    GoGreedyPlayer, GoHLPlayer, GoHeuristic, GoMctsPlayer, GoMoveCategory, GoPlayer,
    GoRandomPlayer, GoReplay, GoState, GoTemplate, GoValidatorPlayer, MoveRecord, ReplayError,
};

#[cfg(feature = "fft")]
pub mod fft;

#[cfg(feature = "fft")]
pub use fft::{
    Action, ActionType, ActiveEffect, BattleState, Class, FftPlayer, GameEvent, GreedyFFTPlayer,
    HLFFTPlayer, Pos, Stats, StatusEffect, Team, Unit, ValidatorFFTPlayer, resolve_action,
};
