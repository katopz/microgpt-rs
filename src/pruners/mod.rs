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
pub mod trial_log;

#[cfg(feature = "bandit")]
pub use absorb_compress::{AbsorbCompress, AbsorbCompressLayer, CompressConfig};

#[cfg(feature = "bandit")]
pub use bandit::{
    BanditEnv, BanditEvent, BanditPruner, BanditResult, BanditSession, BanditStats, BanditStrategy,
    BernoulliEnv, GaussianEnv,
};

#[cfg(feature = "bandit")]
pub use hot_swap::HotSwapPruner;

#[cfg(feature = "bandit")]
pub use regression::{GoldenTrace, RegressionResult, RegressionSuite, ReplayReward};

#[cfg(feature = "bandit")]
pub use trial_log::{TrialLog, TrialRecord, TrialSummary};

#[cfg(feature = "bomber")]
pub use bomber::{
    ArenaGrid, BomberAction, BomberPlayer, GameEvent, GreedyPlayer, GridPos, HLPlayer,
    PlayerEntities, RandomPlayer, ScoreBoard, TickCounter, ValidatorPlayer, init_world, run_tick,
    spawn_players,
};
