//! Bomberman HL Arena — 4-Player Heuristic Learning Proof (Plan 033)
//!
//! bevy_ecs standalone + ratatui TUI arena where 4 AI players compete,
//! each using progressively more HL technology.

pub mod arena;
pub mod players;
pub mod systems;

pub use arena::ArenaGrid;
pub use players::{BomberPlayer, GreedyPlayer, HLPlayer, RandomPlayer, ValidatorPlayer};
pub use systems::*;

use std::fmt;

use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};

// ── Constants ──────────────────────────────────────────────────

pub const ARENA_W: usize = 13;
pub const ARENA_H: usize = 13;
pub const BOMB_FUSE_TICKS: u32 = 4;
pub const DEFAULT_BLAST_RANGE: u32 = 2;
pub const DEFAULT_MAX_BOMBS: u8 = 1;
pub const DEFAULT_SPEED: u8 = 1;
pub const TICK_LIMIT: u32 = 200;
pub const DESTRUCTIBLE_FILL: f32 = 0.40;
pub const SPAWN_POSITIONS: [(i32, i32); 4] = [(1, 1), (11, 1), (1, 11), (11, 11)];

// ── Action ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BomberAction {
    Up,
    Down,
    Left,
    Right,
    Bomb,
    Wait,
}

impl BomberAction {
    pub fn as_usize(&self) -> usize {
        match self {
            Self::Up => 0,
            Self::Down => 1,
            Self::Left => 2,
            Self::Right => 3,
            Self::Bomb => 4,
            Self::Wait => 5,
        }
    }

    pub fn all() -> [BomberAction; 6] {
        [
            Self::Up,
            Self::Down,
            Self::Left,
            Self::Right,
            Self::Bomb,
            Self::Wait,
        ]
    }
}

impl fmt::Display for BomberAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Up => "↑",
            Self::Down => "↓",
            Self::Left => "←",
            Self::Right => "→",
            Self::Bomb => "💣",
            Self::Wait => "⏸",
        };
        write!(f, "{s}")
    }
}

impl From<usize> for BomberAction {
    fn from(val: usize) -> Self {
        match val {
            0 => Self::Up,
            1 => Self::Down,
            2 => Self::Left,
            3 => Self::Right,
            4 => Self::Bomb,
            _ => Self::Wait,
        }
    }
}

// ── Power-Up ───────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PowerUpKind {
    BombUp,
    FireUp,
    SpeedUp,
}

impl fmt::Display for PowerUpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::BombUp => "💥",
            Self::FireUp => "🔥",
            Self::SpeedUp => "👟",
        };
        write!(f, "{s}")
    }
}

// ── Grid Cell ──────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    Floor,
    FixedWall,
    DestructibleWall,
    PowerUpHidden(PowerUpKind),
}

// ── Components ─────────────────────────────────────────────────

#[derive(Component)]
pub struct Player {
    pub id: u8,
}

#[derive(Component)]
pub struct Bomb;

#[derive(Component)]
pub struct PowerUp {
    pub kind: PowerUpKind,
}

#[derive(Component)]
pub struct Blast;

#[derive(Component)]
pub struct DestructibleWall;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct GridPos {
    pub x: i32,
    pub y: i32,
}

#[derive(Component)]
pub struct BombFuse {
    pub owner: Entity,
    pub ticks_remaining: u32,
}

#[derive(Component)]
pub struct BombRange {
    pub cells: u32,
}

#[derive(Component)]
pub struct BombCount {
    pub max: u8,
    pub active: u8,
}

#[derive(Component)]
pub struct Speed {
    pub cells_per_tick: u8,
}

#[derive(Component, Default)]
pub struct Alive;

// ── Resources ──────────────────────────────────────────────────

#[derive(Resource)]
pub struct GameRng {
    pub seed: u64,
}

#[derive(Resource, Default)]
pub struct TickCounter {
    pub tick: u32,
}

impl TickCounter {
    pub fn inc(&mut self) -> u32 {
        self.tick += 1;
        self.tick
    }
}

#[derive(Resource, Default)]
pub struct ScoreBoard {
    pub scores: [i32; 4],
}

impl ScoreBoard {
    pub fn add(&mut self, player: u8, points: i32) {
        if let 0..=3 = player { self.scores[player as usize] += points }
    }
}

#[derive(Resource)]
pub struct PlayerEntities {
    pub entities: [Entity; 4],
}

// ── Events ─────────────────────────────────────────────────────

#[derive(Event, Clone, Debug)]
pub enum GameEvent {
    PlayerMoved {
        player: u8,
        from: (i32, i32),
        to: (i32, i32),
    },
    BombPlaced {
        player: u8,
        pos: (i32, i32),
    },
    BombExploded {
        pos: (i32, i32),
        range: u32,
    },
    PlayerKilled {
        victim: u8,
        killer: Option<u8>,
    },
    PowerUpCollected {
        player: u8,
        kind: PowerUpKind,
    },
    WallDestroyed {
        pos: (i32, i32),
    },
    RoundEnd {
        survivors: Vec<u8>,
    },
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_roundtrip() {
        for (i, expected) in BomberAction::all().into_iter().enumerate() {
            let action = BomberAction::from(i);
            assert_eq!(action, expected);
            assert_eq!(action.as_usize(), i);
        }
    }

    #[test]
    fn action_from_overflow_clamps_to_wait() {
        assert_eq!(BomberAction::from(99), BomberAction::Wait);
    }

    #[test]
    fn scoreboard_add_valid() {
        let mut sb = ScoreBoard::default();
        sb.add(0, 10);
        sb.add(2, -5);
        sb.add(3, 7);
        assert_eq!(sb.scores, [10, 0, -5, 7]);
    }

    #[test]
    fn scoreboard_add_ignores_invalid_player() {
        let mut sb = ScoreBoard::default();
        sb.add(4, 100);
        assert_eq!(sb.scores, [0, 0, 0, 0]);
    }

    #[test]
    fn tick_counter_inc() {
        let mut tc = TickCounter::default();
        assert_eq!(tc.inc(), 1);
        assert_eq!(tc.inc(), 2);
        assert_eq!(tc.tick, 2);
    }

    #[test]
    fn powerup_display() {
        assert_eq!(format!("{}", PowerUpKind::BombUp), "💥");
        assert_eq!(format!("{}", PowerUpKind::FireUp), "🔥");
        assert_eq!(format!("{}", PowerUpKind::SpeedUp), "👟");
    }

    #[test]
    fn action_display() {
        assert_eq!(format!("{}", BomberAction::Up), "↑");
        assert_eq!(format!("{}", BomberAction::Bomb), "💣");
        assert_eq!(format!("{}", BomberAction::Wait), "⏸");
    }
}
