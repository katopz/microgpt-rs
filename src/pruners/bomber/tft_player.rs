//! Tit-for-Tat bomber player — game theory composite (Issue 056).
//!
//! Combines Nice (Greedy scorer) + Retaliatory (HL attack) + Forgiving (auto-reset).
//! 2-state FSM: Nice (default) ↔ Retaliatory (when provoked by nearby hostile bomb).
//!
//! # Game Theory Principles (Axelrod's Tournament)
//!
//! 1. **Nice** — Never provoke first; default to Greedy's resource collection heuristic.
//! 2. **Retaliatory** — When opponent bomb blast threatens us, switch to HL attack tactics.
//! 3. **Forgiving** — After `retaliation_duration` ticks, auto-reset to Nice.
//! 4. **Clear** — Simple 2-state FSM; opponents can learn to cooperate.
//! 5. **Generous** — 10% chance to forgive a provocation without retaliating.

use std::any::Any;

use fastrand::Rng;

use super::players::{
    BomberPlayer, count_escape_routes, in_blast_zone, intercept_score, is_safe_action, move_target,
    predict_direction, score_action, trap_score,
};
use super::{
    ArenaGrid, BOMB_FUSE_TICKS, BomberAction, Cell, DEFAULT_BLAST_RANGE, GameEvent, GridPos,
};

// ── Constants ──────────────────────────────────────────────────

const ACTION_COUNT: usize = 6;

const ALL_ACTIONS: [BomberAction; ACTION_COUNT] = [
    BomberAction::Up,
    BomberAction::Down,
    BomberAction::Left,
    BomberAction::Right,
    BomberAction::Bomb,
    BomberAction::Wait,
];

/// Tracked bomb: (position, blast_range, fuse_ticks_remaining).
type KnownBomb = ((i32, i32), u32, u32);

/// Tracked opponent: (player_id, current_pos, prev_pos).
type KnownOpponent = (u8, (i32, i32), Option<(i32, i32)>);

// ── FSM State ──────────────────────────────────────────────────

/// Tit-for-Tat mode: Nice (cooperative) or Retaliatory (attacking).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TftMode {
    /// Default: use Greedy's score_action heuristic for resource collection.
    Nice,
    /// Provoked: use HL-style hunt/intercept/trap tactics. Auto-reverts after ticks_left reaches 0.
    Retaliatory { ticks_left: u8 },
}

impl std::fmt::Display for TftMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TftMode::Nice => write!(f, "Nice"),
            TftMode::Retaliatory { ticks_left } => write!(f, "Retaliatory({ticks_left})"),
        }
    }
}

/// Simple round-level outcome tracking for diagnostics.
#[derive(Default, Clone, Copy)]
pub struct TftRoundStats {
    pub ticks_nice: u32,
    pub ticks_retaliatory: u32,
    pub provocations: u32,
    pub forgivenesses: u32,
}

impl std::fmt::Display for TftRoundStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total = self.ticks_nice + self.ticks_retaliatory;
        let nice_pct = if total > 0 {
            self.ticks_nice as f32 / total as f32 * 100.0
        } else {
            0.0
        };
        write!(
            f,
            "Nice={nice_pct:.0}% Provocations={} Forgivenesses={}",
            self.provocations, self.forgivenesses
        )
    }
}

// ── Player ─────────────────────────────────────────────────────

/// Tit-for-Tat bomber player.
///
/// Game theory composite that adapts between cooperation and retaliation:
/// - **Nice mode**: Uses Greedy's `score_action` heuristic for efficient resource collection.
/// - **Retaliatory mode**: Uses HL's proven attack tactics (hunt, intercept, trap, chokepoint).
/// - **Forgiving**: Auto-switches back to Nice after N ticks (configurable).
/// - **Generous TFT**: 10% chance to forgive a provocation without retaliating.
pub struct TftPlayer {
    _id: u8,
    known_bombs: Vec<KnownBomb>,
    known_powerups: Vec<(i32, i32)>,
    known_opponents: Vec<KnownOpponent>,
    mode: TftMode,
    provocation_radius: i32,
    retaliation_duration: u8,
    forgiveness_chance: f32,
    last_dir: Option<BomberAction>,
    round_stats: TftRoundStats,
}

impl TftPlayer {
    pub fn new(id: u8) -> Self {
        Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            known_opponents: Vec::new(),
            mode: TftMode::Nice,
            provocation_radius: 4,
            retaliation_duration: 10,
            forgiveness_chance: 0.10,
            last_dir: None,
            round_stats: TftRoundStats::default(),
        }
    }

    /// Create TftPlayer with custom parameters.
    pub fn with_params(
        id: u8,
        provocation_radius: i32,
        retaliation_duration: u8,
        forgiveness_chance: f32,
    ) -> Self {
        Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            known_opponents: Vec::new(),
            mode: TftMode::Nice,
            provocation_radius,
            retaliation_duration,
            forgiveness_chance,
            last_dir: None,
            round_stats: TftRoundStats::default(),
        }
    }

    /// Current FSM mode (for diagnostics).
    pub fn mode(&self) -> TftMode {
        self.mode
    }

    /// Round-level stats (for diagnostics).
    pub fn round_stats(&self) -> &TftRoundStats {
        &self.round_stats
    }

    /// Check if provoked: we're actually in a blast zone AND an opponent is close.
    ///
    /// Conservative: only retaliates when genuinely threatened (in blast zone),
    /// not just because bombs exist nearby. Uses wall-aware blast check.
    fn is_provoked(
        pos: GridPos,
        grid: &ArenaGrid,
        bombs: &[KnownBomb],
        opponents: &[KnownOpponent],
        radius: i32,
    ) -> bool {
        // Must be in actual danger (wall-aware blast zone check)
        if !in_blast_zone(pos, grid, bombs) {
            return false;
        }

        // AND an opponent is close enough to be the aggressor
        opponents
            .iter()
            .any(|(_, (ox, oy), _)| (pos.x - ox).abs() + (pos.y - oy).abs() <= radius)
    }

    /// Update known bomb list from events.
    fn update_bombs(bombs: &mut Vec<KnownBomb>, events: &[GameEvent]) {
        for bomb in bombs.iter_mut() {
            bomb.2 = bomb.2.saturating_sub(1);
        }
        for event in events {
            match event {
                GameEvent::BombPlaced { pos, .. } => {
                    if !bombs.iter().any(|(p, _, _)| *p == *pos) {
                        bombs.push((*pos, DEFAULT_BLAST_RANGE, BOMB_FUSE_TICKS));
                    }
                }
                GameEvent::BombExploded { pos, .. } => {
                    bombs.retain(|(p, _, _)| *p != *pos);
                }
                _ => {}
            }
        }
    }

    /// Update known power-up list from events.
    fn update_powerups(powerups: &mut Vec<(i32, i32)>, events: &[GameEvent]) {
        for event in events {
            match event {
                GameEvent::PowerUpRevealed { pos, .. } => {
                    if !powerups.contains(pos) {
                        powerups.push(*pos);
                    }
                }
                GameEvent::PowerUpCollected { pos, .. } => {
                    powerups.retain(|p| p != pos);
                }
                _ => {}
            }
        }
    }

    /// Track opponent positions from events.
    fn update_opponents(opponents: &mut Vec<KnownOpponent>, events: &[GameEvent], my_id: u8) {
        for event in events {
            match event {
                GameEvent::PlayerMoved { player, to, .. } => {
                    if *player == my_id {
                        continue;
                    }
                    if let Some(entry) = opponents.iter_mut().find(|(p, _, _)| *p == *player) {
                        entry.2 = Some(entry.1);
                        entry.1 = *to;
                    } else {
                        opponents.push((*player, *to, None));
                    }
                }
                GameEvent::BombPlaced { player, pos } => {
                    if *player == my_id {
                        continue;
                    }
                    if let Some(entry) = opponents.iter_mut().find(|(p, _, _)| *p == *player) {
                        entry.2 = Some(entry.1);
                        entry.1 = *pos;
                    } else {
                        opponents.push((*player, *pos, None));
                    }
                }
                GameEvent::PlayerKilled { victim, .. } => {
                    opponents.retain(|(p, _, _)| *p != *victim);
                }
                _ => {}
            }
        }
    }

    /// Compute retaliation strategy bonus (HL-style tactics).
    fn retaliation_bonus(
        action: &BomberAction,
        grid: &ArenaGrid,
        pos: GridPos,
        nearest_opponent: Option<(i32, i32)>,
        predicted_opponent: Option<(i32, i32)>,
    ) -> f32 {
        let Some((ox, oy)) = nearest_opponent else {
            return 0.0;
        };

        let mut bonus = 0.0f32;
        match action {
            BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right => {
                let target = move_target(action, pos);
                let current_dist = (pos.x - ox).abs() + (pos.y - oy).abs();
                let target_dist = (target.x - ox).abs() + (target.y - oy).abs();

                // Hunt: move toward opponent
                if target_dist < current_dist {
                    bonus += 1.5;
                }

                // Intercept: move toward predicted position
                bonus += intercept_score((target.x, target.y), (ox, oy), predicted_opponent);

                // Chokepoint: prefer moving where opponent has fewer escapes
                if target_dist <= 3 {
                    let routes = count_escape_routes((target.x, target.y), grid);
                    if routes <= 1 {
                        bonus += 1.0;
                    }
                }
            }
            BomberAction::Bomb => {
                // Strategic value: wall count bonus
                let wall_count = [(0i32, -1), (0, 1), (-1, 0), (1, 0)]
                    .iter()
                    .filter(|&&(dx, dy)| {
                        matches!(
                            grid.get(pos.x + dx, pos.y + dy),
                            Cell::DestructibleWall | Cell::PowerUpHidden(_)
                        )
                    })
                    .count();
                bonus += wall_count as f32 * 0.5;

                // Attack: trap scoring when opponent is nearby
                bonus += trap_score((pos.x, pos.y), (ox, oy), grid, DEFAULT_BLAST_RANGE);
            }
            BomberAction::Wait => {}
        }
        bonus
    }
}

// ── BomberPlayer Trait ─────────────────────────────────────────

impl BomberPlayer for TftPlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        rng: &mut Rng,
    ) -> BomberAction {
        // 1. Update game state
        Self::update_bombs(&mut self.known_bombs, events);
        Self::update_powerups(&mut self.known_powerups, events);
        Self::update_opponents(&mut self.known_opponents, events, self._id);

        // Find nearest opponent and predicted trajectory
        let nearest_info = self
            .known_opponents
            .iter()
            .filter(|(_, op, _)| grid.is_walkable(op.0, op.1))
            .min_by_key(|(_, op, _)| (pos.x - op.0).abs() + (pos.y - op.1).abs());

        let nearest_opponent = nearest_info.map(|(_, op, _)| *op);
        let predicted_opponent =
            nearest_info.and_then(|(_, op, prev)| predict_direction(*op, *prev));

        // 2. Update mode FSM
        let provoked = Self::is_provoked(
            pos,
            grid,
            &self.known_bombs,
            &self.known_opponents,
            self.provocation_radius,
        );

        self.mode = match self.mode {
            // Forgiving: auto-reset to Nice when retaliation expires
            TftMode::Retaliatory { ticks_left: 0 } => {
                self.round_stats.forgivenesses += 1;
                TftMode::Nice
            }
            // Countdown retaliation ticks
            TftMode::Retaliatory { ticks_left } => TftMode::Retaliatory {
                ticks_left: ticks_left.saturating_sub(1),
            },
            // Nice mode: check for provocation
            TftMode::Nice if provoked => {
                self.round_stats.provocations += 1;
                // Generous TFT: forgive without retaliating
                if rng.f32() < self.forgiveness_chance {
                    self.round_stats.forgivenesses += 1;
                    TftMode::Nice
                } else {
                    TftMode::Retaliatory {
                        ticks_left: self.retaliation_duration,
                    }
                }
            }
            TftMode::Nice => TftMode::Nice,
        };

        // Track stats
        match self.mode {
            TftMode::Nice => self.round_stats.ticks_nice += 1,
            TftMode::Retaliatory { .. } => self.round_stats.ticks_retaliatory += 1,
        }

        // 3. Score actions based on current mode
        let bomb_positions: std::collections::HashSet<(i32, i32)> =
            self.known_bombs.iter().map(|(p, _, _)| *p).collect();

        let mut scores: [(BomberAction, f32); ACTION_COUNT] = ALL_ACTIONS.map(|a| (a, 0.0));

        for (i, action) in ALL_ACTIONS.iter().enumerate() {
            let h = score_action(
                action,
                grid,
                pos,
                &self.known_bombs,
                &self.known_powerups,
                self.last_dir,
            );

            // Domain hard block (unwalkable, unsafe bomb)
            if h == f32::NEG_INFINITY {
                scores[i] = (*action, h);
                continue;
            }

            // Safety validation — hard-block unsafe Bomb/Wait
            let is_move = matches!(
                action,
                BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
            );
            if !is_move && !is_safe_action(action, grid, pos, &self.known_bombs) {
                scores[i] = (*action, f32::NEG_INFINITY);
                continue;
            }

            // Strategy bonus: only in Retaliatory mode AND not in immediate danger.
            // When in blast zone, even Retaliatory TFT must flee first — hunt later.
            let in_danger = in_blast_zone(pos, grid, &self.known_bombs);
            let strategy_bonus = match (self.mode, in_danger) {
                (TftMode::Retaliatory { .. }, false) => {
                    Self::retaliation_bonus(action, grid, pos, nearest_opponent, predicted_opponent)
                }
                _ => 0.0,
            };

            scores[i] = (*action, h + strategy_bonus);
        }

        // 4. ε-greedy: 10% random safe exploration
        if rng.f32() < 0.10 {
            let safe_explore: Vec<usize> = (0..ACTION_COUNT)
                .filter(|&i| {
                    if scores[i].1 <= f32::NEG_INFINITY {
                        return false;
                    }
                    let action = ALL_ACTIONS[i];
                    match action {
                        BomberAction::Up
                        | BomberAction::Down
                        | BomberAction::Left
                        | BomberAction::Right => {
                            let target = move_target(&action, pos);
                            grid.is_walkable(target.x, target.y)
                                && !bomb_positions.contains(&(target.x, target.y))
                                && !in_blast_zone(target, grid, &self.known_bombs)
                        }
                        _ => false, // Don't randomly explore Bomb/Wait
                    }
                })
                .collect();
            if !safe_explore.is_empty() {
                let pick = safe_explore[rng.usize(0..safe_explore.len())];
                let action = scores[pick].0;
                self.last_dir = match action {
                    BomberAction::Up
                    | BomberAction::Down
                    | BomberAction::Left
                    | BomberAction::Right => Some(action),
                    _ => self.last_dir,
                };
                return action;
            }
        }

        // 5. Pick best action
        let best = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(a, _)| *a)
            .unwrap_or(BomberAction::Wait);

        // Track own bomb placement
        if best == BomberAction::Bomb {
            self.known_bombs
                .push(((pos.x, pos.y), DEFAULT_BLAST_RANGE, BOMB_FUSE_TICKS));
        }

        // Track direction
        if matches!(
            best,
            BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
        ) {
            self.last_dir = Some(best);
        }

        best
    }

    fn name(&self) -> &str {
        "TFT"
    }

    fn emoji(&self) -> &str {
        "🦊"
    }

    fn reset(&mut self) {
        self.known_bombs.clear();
        self.known_powerups.clear();
        self.known_opponents.clear();
        self.mode = TftMode::Nice;
        self.last_dir = None;
        self.round_stats = TftRoundStats::default();
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_grid() -> ArenaGrid {
        ArenaGrid::generate(42)
    }

    #[test]
    fn test_tft_player_new() {
        let player = TftPlayer::new(0);
        assert_eq!(player.name(), "TFT");
        assert_eq!(player.emoji(), "🦊");
        assert_eq!(player.mode, TftMode::Nice);
        assert_eq!(player.provocation_radius, 4);
        assert_eq!(player.retaliation_duration, 10);
    }

    #[test]
    fn test_tft_player_with_params() {
        let player = TftPlayer::with_params(1, 6, 15, 0.05);
        assert_eq!(player.provocation_radius, 6);
        assert_eq!(player.retaliation_duration, 15);
        assert!((player.forgiveness_chance - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tft_player_reset() {
        let mut player = TftPlayer::new(0);
        player.mode = TftMode::Retaliatory { ticks_left: 5 };
        player.known_bombs.push(((1, 1), 2, 4));
        player.round_stats.ticks_nice = 10;
        player.round_stats.ticks_retaliatory = 3;
        player.round_stats.provocations = 1;
        player.reset();
        assert!(player.known_bombs.is_empty());
        assert_eq!(player.mode, TftMode::Nice);
        assert_eq!(player.round_stats.ticks_nice, 0);
        assert_eq!(player.round_stats.ticks_retaliatory, 0);
    }

    #[test]
    fn test_tft_mode_default_nice() {
        let player = TftPlayer::new(0);
        assert_eq!(player.mode, TftMode::Nice);
    }

    #[test]
    fn test_tft_mode_display() {
        assert_eq!(format!("{}", TftMode::Nice), "Nice");
        assert_eq!(
            format!("{}", TftMode::Retaliatory { ticks_left: 7 }),
            "Retaliatory(7)"
        );
    }

    #[test]
    fn test_is_provoked_no_bombs() {
        let grid = empty_grid();
        let pos = GridPos { x: 5, y: 5 };
        let bombs: Vec<KnownBomb> = vec![];
        let opponents: Vec<KnownOpponent> = vec![(1, (6, 5), None)];
        assert!(!TftPlayer::is_provoked(pos, &grid, &bombs, &opponents, 4));
    }

    #[test]
    fn test_is_provoked_no_opponents() {
        let grid = empty_grid();
        let pos = GridPos { x: 5, y: 5 };
        let bombs: Vec<KnownBomb> = vec![((5, 3), 2, 4)];
        let opponents: Vec<KnownOpponent> = vec![];
        assert!(!TftPlayer::is_provoked(pos, &grid, &bombs, &opponents, 4));
    }

    #[test]
    fn test_is_provoked_both_present() {
        let grid = empty_grid();
        // Spawn position (1,1) is guaranteed walkable; bomb ON player = always in blast zone
        let pos = GridPos { x: 1, y: 1 };
        let bombs: Vec<KnownBomb> = vec![((1, 1), 2, 4)];
        let opponents: Vec<KnownOpponent> = vec![(1, (2, 1), None)];
        assert!(TftPlayer::is_provoked(pos, &grid, &bombs, &opponents, 4));
    }

    #[test]
    fn test_is_provoked_opponent_far_away() {
        let grid = empty_grid();
        let pos = GridPos { x: 1, y: 1 };
        // Bomb on player (in blast zone) but opponent too far away
        let bombs: Vec<KnownBomb> = vec![((1, 1), 2, 4)];
        let opponents: Vec<KnownOpponent> = vec![(1, (11, 11), None)];
        assert!(!TftPlayer::is_provoked(pos, &grid, &bombs, &opponents, 4));
    }

    #[test]
    fn test_is_provoked_not_in_blast_zone() {
        let grid = empty_grid();
        // Spawn corners (1,1) and (11,11) are far apart with guaranteed walls between
        let pos = GridPos { x: 1, y: 1 };
        let bombs: Vec<KnownBomb> = vec![((11, 11), 2, 4)];
        let opponents: Vec<KnownOpponent> = vec![(1, (2, 1), None)];
        assert!(!TftPlayer::is_provoked(pos, &grid, &bombs, &opponents, 4));
    }

    #[test]
    fn test_tft_select_action_returns_valid() {
        let mut player = TftPlayer::new(0);
        let grid = empty_grid();
        let pos = GridPos { x: 1, y: 1 };
        let mut rng = Rng::with_seed(42);

        let action = player.select_action(&grid, pos, &[], &mut rng);
        assert!(matches!(
            action,
            BomberAction::Up
                | BomberAction::Down
                | BomberAction::Left
                | BomberAction::Right
                | BomberAction::Bomb
                | BomberAction::Wait
        ));
    }

    #[test]
    fn test_tft_trait_methods() {
        let player = TftPlayer::new(2);
        assert_eq!(player.name(), "TFT");
        assert_eq!(player.emoji(), "🦊");
        let _any_ref = player.as_any();
    }

    #[test]
    fn test_round_stats_display() {
        let stats = TftRoundStats {
            ticks_nice: 80,
            ticks_retaliatory: 20,
            provocations: 3,
            forgivenesses: 1,
        };
        let display = format!("{stats}");
        assert!(display.contains("Nice=80%"));
        assert!(display.contains("Provocations=3"));
    }
}
