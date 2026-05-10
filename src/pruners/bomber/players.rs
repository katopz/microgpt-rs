//! AI player trait and implementations for Bomberman HL Arena.
//!
//! Four player types representing increasing HL technology levels:
//! - P1 (Random): no model, no learning — pure baseline
//! - P2 (Greedy): heuristic action selection simulating LoRA marginals
//! - P3 (Validator): heuristic + hard safety rules simulating WASM validator
//! - P4 (Full HL): bandit-adapted selection with absorb-compress

use std::any::Any;

use fastrand::Rng;

use super::{ArenaGrid, BomberAction, GameEvent, GridPos};

// ── Constants ──────────────────────────────────────────────────

const ACTION_COUNT: usize = 6;
const DEFAULT_BLAST_RANGE: u32 = 2;

const ALL_ACTIONS: [BomberAction; ACTION_COUNT] = [
    BomberAction::Up,
    BomberAction::Down,
    BomberAction::Left,
    BomberAction::Right,
    BomberAction::Bomb,
    BomberAction::Wait,
];

// ── Trait ──────────────────────────────────────────────────────

/// AI player trait for Bomberman arena.
///
/// Each implementation represents a different HL technology level:
/// - P1 (Random): no model, no learning
/// - P2 (Model): LoRA-based action selection
/// - P3 (Validated): LoRA + WASM validator
/// - P4 (Full HL): LoRA + WASM + Bandit + TrialLog + AbsorbCompress
pub trait BomberPlayer {
    /// Select an action given the current game state.
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        rng: &mut Rng,
    ) -> BomberAction;

    /// Player display name.
    fn name(&self) -> &str;

    /// Emoji for TUI rendering.
    fn emoji(&self) -> &str;

    /// Reset internal state for a new round.
    fn reset(&mut self);

    /// Downcast support for HL player updates.
    fn as_any(&self) -> &dyn Any;

    /// Downcast support for HL player updates (mutable).
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// ── Shared Helpers ─────────────────────────────────────────────

/// Compute target position after applying a move action.
fn move_target(action: &BomberAction, pos: GridPos) -> GridPos {
    match action {
        BomberAction::Up => GridPos {
            x: pos.x,
            y: pos.y - 1,
        },
        BomberAction::Down => GridPos {
            x: pos.x,
            y: pos.y + 1,
        },
        BomberAction::Left => GridPos {
            x: pos.x - 1,
            y: pos.y,
        },
        BomberAction::Right => GridPos {
            x: pos.x + 1,
            y: pos.y,
        },
        BomberAction::Bomb | BomberAction::Wait => pos,
    }
}

/// Convert action to index 0..6.
fn action_index(action: &BomberAction) -> usize {
    match action {
        BomberAction::Up => 0,
        BomberAction::Down => 1,
        BomberAction::Left => 2,
        BomberAction::Right => 3,
        BomberAction::Bomb => 4,
        BomberAction::Wait => 5,
    }
}

/// Convert index 0..6 to action.
fn index_to_action(idx: usize) -> BomberAction {
    match idx {
        0 => BomberAction::Up,
        1 => BomberAction::Down,
        2 => BomberAction::Left,
        3 => BomberAction::Right,
        4 => BomberAction::Bomb,
        _ => BomberAction::Wait,
    }
}

/// Manhattan distance between two grid positions.
#[allow(dead_code)]
fn manhattan(a: GridPos, b: GridPos) -> i32 {
    (a.x - b.x).abs() + (a.y - b.y).abs()
}

/// Check if position is in the blast zone of any known bomb.
fn in_blast_zone(pos: GridPos, bombs: &[((i32, i32), u32)]) -> bool {
    for &(bomb_pos, range) in bombs {
        let dx = (pos.x - bomb_pos.0).abs();
        let dy = (pos.y - bomb_pos.1).abs();
        // Same row or same column and within range
        if (dx == 0 && dy <= range as i32) || (dy == 0 && dx <= range as i32) {
            return true;
        }
    }
    false
}

/// Update known bomb list from events.
fn update_bombs(bombs: &mut Vec<((i32, i32), u32)>, events: &[GameEvent]) {
    for event in events {
        match event {
            GameEvent::BombPlaced { pos, .. } => {
                if !bombs.iter().any(|(p, _)| *p == *pos) {
                    bombs.push((*pos, DEFAULT_BLAST_RANGE));
                }
            }
            GameEvent::BombExploded { pos, .. } => {
                bombs.retain(|(p, _)| *p != *pos);
            }
            _ => {}
        }
    }
}

/// Check if player has an escape route after placing a bomb at `bomb_pos`.
/// BFS from `player_pos` — must reach a cell outside the blast zone within `blast_range + 1` steps.
fn has_escape_route(
    grid: &ArenaGrid,
    player_pos: GridPos,
    bomb_pos: (i32, i32),
    blast_range: u32,
) -> bool {
    use std::collections::{HashSet, VecDeque};

    let max_steps = blast_range as i32 + 1;
    let mut visited: HashSet<(i32, i32)> = HashSet::new();
    let mut queue: VecDeque<((i32, i32), i32)> = VecDeque::new();

    // Don't start ON the bomb — that's instant death
    if player_pos.x == bomb_pos.0 && player_pos.y == bomb_pos.1 {
        return false;
    }

    queue.push_back(((player_pos.x, player_pos.y), 0));
    visited.insert((player_pos.x, player_pos.y));

    while let Some(((cx, cy), steps)) = queue.pop_front() {
        if steps > max_steps {
            continue;
        }

        // Is this cell safe (outside blast zone)?
        let dx = (cx - bomb_pos.0).abs();
        let dy = (cy - bomb_pos.1).abs();
        let in_blast =
            (dx == 0 && dy <= blast_range as i32) || (dy == 0 && dx <= blast_range as i32);
        if !in_blast {
            return true;
        }

        // Expand neighbors
        for (nx, ny) in [(cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)] {
            if visited.insert((nx, ny)) && grid.is_walkable(nx, ny) {
                queue.push_back(((nx, ny), steps + 1));
            }
        }
    }

    false
}

/// Check if an action is safe given the current state.
fn is_safe_action(
    action: &BomberAction,
    grid: &ArenaGrid,
    pos: GridPos,
    bombs: &[((i32, i32), u32)],
) -> bool {
    match action {
        BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right => {
            let target = move_target(action, pos);
            if !grid.is_walkable(target.x, target.y) {
                return false;
            }
            // Don't walk into blast zone
            let mut future_bombs = bombs.to_vec();
            update_bombs(&mut future_bombs, &[]);
            !in_blast_zone(target, &future_bombs)
        }
        BomberAction::Bomb => {
            // Player stands ON the bomb but moves away next tick — check escape
            // from each adjacent cell (mirrors should_place_bomb logic).
            [(0i32, -1), (0, 1), (-1, 0), (1, 0)]
                .iter()
                .any(|&(dx, dy)| {
                    let nx = pos.x + dx;
                    let ny = pos.y + dy;
                    grid.is_walkable(nx, ny)
                        && has_escape_route(
                            grid,
                            GridPos { x: nx, y: ny },
                            (pos.x, pos.y),
                            DEFAULT_BLAST_RANGE,
                        )
                })
        }
        BomberAction::Wait => {
            // Waiting is only safe if not in blast zone
            !in_blast_zone(pos, bombs)
        }
    }
}

/// Heuristic score for an action (used by Greedy, Validator, HL players).
fn heuristic_score(
    action: &BomberAction,
    grid: &ArenaGrid,
    pos: GridPos,
    bombs: &[((i32, i32), u32)],
    last_dir: Option<BomberAction>,
) -> f32 {
    let target = move_target(action, pos);

    match action {
        BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right => {
            // Walking into wall — invalid
            if !grid.is_walkable(target.x, target.y) {
                return -1.0;
            }

            // Walking into blast zone — very bad
            if in_blast_zone(target, bombs) {
                return -0.8;
            }

            let mut score = 0.1f32;

            // Penalize reversing — prefer direction persistence
            if let Some(last) = last_dir
                && matches!(
                    (*action, last),
                    (BomberAction::Up, BomberAction::Down)
                        | (BomberAction::Down, BomberAction::Up)
                        | (BomberAction::Left, BomberAction::Right)
                        | (BomberAction::Right, BomberAction::Left)
                )
            {
                score -= 0.3;
            }

            // Moving away from blast zone when in danger
            if in_blast_zone(pos, bombs) && !in_blast_zone(target, bombs) {
                score += 0.8;
            }

            // Moving toward powerup
            if matches!(grid.get(target.x, target.y), super::Cell::PowerUpHidden(_)) {
                score += 0.6;
            }

            // Moving toward center (explore heuristic)
            let center = 6i32;
            let dist_before = (pos.x - center).abs() + (pos.y - center).abs();
            let dist_after = (target.x - center).abs() + (target.y - center).abs();
            if dist_after < dist_before {
                score += 0.2;
            }

            score
        }
        BomberAction::Bomb => {
            // Need escape route
            if !has_escape_route(grid, pos, (pos.x, pos.y), DEFAULT_BLAST_RANGE) {
                return -0.9;
            }

            // Placing bomb near destructible walls is good
            let mut score = 0.2f32;
            for (dx, dy) in [(0i32, -1), (0, 1), (-1, 0), (1, 0)] {
                let cx = pos.x + dx;
                let cy = pos.y + dy;
                match grid.get(cx, cy) {
                    super::Cell::DestructibleWall => score += 0.15,
                    super::Cell::PowerUpHidden(_) => score += 0.2,
                    _ => {}
                }
            }

            score
        }
        BomberAction::Wait => {
            // Waiting is generally bad (opportunity cost)
            if in_blast_zone(pos, bombs) {
                -0.5 // Waiting in blast zone is terrible
            } else {
                -0.1
            }
        }
    }
}

// ── AI State Machine ────────────────────────────────────────────

/// AI behavior state for purposeful movement.
#[derive(Clone, Copy, Debug, PartialEq)]
enum AiState {
    /// Walking in a direction. Pick new direction when blocked.
    Explore { dir: BomberAction },
    /// Actively pathfinding toward a destructible wall to bomb it.
    Hunt { target: (i32, i32) },
    /// Escaping blast zone toward nearest safe cell.
    Flee,
}

impl Default for AiState {
    fn default() -> Self {
        AiState::Explore {
            dir: BomberAction::Wait,
        }
    }
}

/// Check if `action` would reverse `prev` direction.
fn is_reverse(action: BomberAction, prev: Option<BomberAction>) -> bool {
    matches!(
        (action, prev),
        (BomberAction::Up, Some(BomberAction::Down))
            | (BomberAction::Down, Some(BomberAction::Up))
            | (BomberAction::Left, Some(BomberAction::Right))
            | (BomberAction::Right, Some(BomberAction::Left))
    )
}

/// Pick a new exploration direction. Avoids blast zones and reversing when possible.
fn pick_explore_dir(
    grid: &ArenaGrid,
    pos: GridPos,
    prev_dir: Option<BomberAction>,
    bombs: &[((i32, i32), u32)],
    rng: &mut Rng,
) -> BomberAction {
    use BomberAction::{Down, Left, Right, Up};
    let dirs = [Up, Down, Left, Right];

    // Prefer: walkable AND not in blast zone
    let safe: Vec<BomberAction> = dirs
        .iter()
        .filter(|&&d| {
            let t = move_target(&d, pos);
            grid.is_walkable(t.x, t.y) && !in_blast_zone(t, bombs)
        })
        .copied()
        .collect();

    if !safe.is_empty() {
        let preferred: Vec<BomberAction> = safe
            .iter()
            .filter(|&&d| !is_reverse(d, prev_dir))
            .copied()
            .collect();
        let candidates = if preferred.is_empty() {
            &safe
        } else {
            &preferred
        };
        return candidates[rng.usize(0..candidates.len())];
    }

    // No safe direction — fall back to any walkable
    let valid: Vec<BomberAction> = dirs
        .iter()
        .filter(|&&d| {
            let t = move_target(&d, pos);
            grid.is_walkable(t.x, t.y)
        })
        .copied()
        .collect();

    if valid.is_empty() {
        return BomberAction::Wait;
    }
    valid[rng.usize(0..valid.len())]
}

/// BFS: find nearest cell outside any bomb blast zone.
fn find_safe_cell(
    grid: &ArenaGrid,
    pos: GridPos,
    bombs: &[((i32, i32), u32)],
) -> Option<(i32, i32)> {
    use std::collections::{HashSet, VecDeque};

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    queue.push_back((pos.x, pos.y));
    visited.insert((pos.x, pos.y));

    while let Some((cx, cy)) = queue.pop_front() {
        if !in_blast_zone(GridPos { x: cx, y: cy }, bombs) {
            return Some((cx, cy));
        }
        for (nx, ny) in [(cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)] {
            if visited.insert((nx, ny)) && grid.is_walkable(nx, ny) {
                queue.push_back((nx, ny));
            }
        }
    }

    None
}

/// BFS: find first step from `from` toward `to`.
fn next_step_toward(grid: &ArenaGrid, from: GridPos, to: (i32, i32)) -> Option<BomberAction> {
    if from.x == to.0 && from.y == to.1 {
        return None;
    }

    use std::collections::{HashMap, VecDeque};

    let mut queue: VecDeque<(i32, i32)> = VecDeque::new();
    let mut parent: HashMap<(i32, i32), (i32, i32)> = HashMap::new();

    queue.push_back((from.x, from.y));
    parent.insert((from.x, from.y), (from.x, from.y));

    while let Some((cx, cy)) = queue.pop_front() {
        if cx == to.0 && cy == to.1 {
            // Trace back to first step
            let mut step = (cx, cy);
            while parent[&step] != (from.x, from.y) {
                step = parent[&step];
            }
            let dx = step.0 - from.x;
            let dy = step.1 - from.y;
            return match (dx, dy) {
                (0, -1) => Some(BomberAction::Up),
                (0, 1) => Some(BomberAction::Down),
                (-1, 0) => Some(BomberAction::Left),
                (1, 0) => Some(BomberAction::Right),
                _ => None,
            };
        }

        for (nx, ny) in [(cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)] {
            if !parent.contains_key(&(nx, ny)) && grid.is_walkable(nx, ny) {
                parent.insert((nx, ny), (cx, cy));
                queue.push_back((nx, ny));
            }
        }
    }

    None
}

/// Check if player should place a bomb at current position.
///
/// The player stands ON the bomb but moves away next tick, so escape is
/// checked from adjacent cells — not from the bomb position itself.
fn should_place_bomb(grid: &ArenaGrid, pos: GridPos, bombs: &[((i32, i32), u32)]) -> bool {
    // Don't place if there's already a bomb here
    if bombs.iter().any(|(p, _)| p.0 == pos.x && p.1 == pos.y) {
        return false;
    }

    // Count adjacent destructible walls
    let wall_count = [(0i32, -1), (0, 1), (-1, 0), (1, 0)]
        .iter()
        .filter(|&&(dx, dy)| {
            matches!(
                grid.get(pos.x + dx, pos.y + dy),
                super::Cell::DestructibleWall | super::Cell::PowerUpHidden(_)
            )
        })
        .count();

    if wall_count == 0 {
        return false;
    }

    // Player will move to an adjacent cell next tick (1 step used).
    // From that cell, has_escape_route checks if safety is reachable within
    // max_steps (3) — total 4 steps matches BOMB_FUSE_TICKS.
    let neighbors = [(0i32, -1), (0, 1), (-1, 0), (1, 0)];
    neighbors.iter().any(|&(dx, dy)| {
        let nx = pos.x + dx;
        let ny = pos.y + dy;
        grid.is_walkable(nx, ny)
            && has_escape_route(
                grid,
                GridPos { x: nx, y: ny },
                (pos.x, pos.y),
                DEFAULT_BLAST_RANGE,
            )
    })
}

/// BFS: find nearest destructible wall reachable from `pos`.
/// Prefers `PowerUpHidden` walls over plain `DestructibleWall`.
fn find_nearest_wall(grid: &ArenaGrid, pos: GridPos) -> Option<(i32, i32)> {
    use std::collections::{HashSet, VecDeque};

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    queue.push_back((pos.x, pos.y));
    visited.insert((pos.x, pos.y));

    while let Some((cx, cy)) = queue.pop_front() {
        // Check neighbors for destructible walls (prefer powerups)
        let mut powerup_wall = None;
        let mut normal_wall = None;

        for (dx, dy) in [(0i32, -1), (0, 1), (-1, 0), (1, 0)] {
            match grid.get(cx + dx, cy + dy) {
                super::Cell::PowerUpHidden(_) if powerup_wall.is_none() => {
                    powerup_wall = Some((cx + dx, cy + dy));
                }
                super::Cell::DestructibleWall if normal_wall.is_none() => {
                    normal_wall = Some((cx + dx, cy + dy));
                }
                _ => {}
            }
        }

        if let Some(w) = powerup_wall {
            return Some(w);
        }
        if let Some(w) = normal_wall {
            return Some(w);
        }

        for (nx, ny) in [(cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)] {
            if visited.insert((nx, ny)) && grid.is_walkable(nx, ny) {
                queue.push_back((nx, ny));
            }
        }
    }

    None
}

/// BFS: find first step toward any walkable cell adjacent to `wall`.
fn step_toward_wall(grid: &ArenaGrid, from: GridPos, wall: (i32, i32)) -> Option<BomberAction> {
    let dx = (from.x - wall.0).abs();
    let dy = (from.y - wall.1).abs();
    if dx + dy <= 1 {
        return None; // Already adjacent
    }

    use std::collections::{HashMap, VecDeque};

    let mut queue: VecDeque<(i32, i32)> = VecDeque::new();
    let mut parent: HashMap<(i32, i32), (i32, i32)> = HashMap::new();

    queue.push_back((from.x, from.y));
    parent.insert((from.x, from.y), (from.x, from.y));

    while let Some((cx, cy)) = queue.pop_front() {
        // Is this cell adjacent to the target wall?
        let cdx = (cx - wall.0).abs();
        let cdy = (cy - wall.1).abs();
        if cdx + cdy == 1 {
            // Trace back to first step
            let mut step = (cx, cy);
            while parent[&step] != (from.x, from.y) {
                step = parent[&step];
            }
            let sdx = step.0 - from.x;
            let sdy = step.1 - from.y;
            return match (sdx, sdy) {
                (0, -1) => Some(BomberAction::Up),
                (0, 1) => Some(BomberAction::Down),
                (-1, 0) => Some(BomberAction::Left),
                (1, 0) => Some(BomberAction::Right),
                _ => None,
            };
        }

        for (nx, ny) in [(cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)] {
            if !parent.contains_key(&(nx, ny)) && grid.is_walkable(nx, ny) {
                parent.insert((nx, ny), (cx, cy));
                queue.push_back((nx, ny));
            }
        }
    }

    None
}

// ── P1: Random ─────────────────────────────────────────────────

/// P1: Modelless baseline — uniform random action selection.
///
/// No learning. No memory. No model. Pure baseline.
/// Avoids walking into walls (up to 3 re-rolls, then Wait).
pub struct RandomPlayer {
    _id: u8,
}

impl RandomPlayer {
    pub fn new(id: u8) -> Self {
        Self { _id: id }
    }
}

impl BomberPlayer for RandomPlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        _events: &[GameEvent],
        rng: &mut Rng,
    ) -> BomberAction {
        // Try random actions, avoid walls (3 attempts)
        for _ in 0..3 {
            let idx = rng.usize(0..ACTION_COUNT);
            let action = index_to_action(idx);
            let target = move_target(&action, pos);
            if action == BomberAction::Bomb || action == BomberAction::Wait {
                return action;
            }
            if grid.is_walkable(target.x, target.y) {
                return action;
            }
        }
        BomberAction::Wait
    }

    fn name(&self) -> &str {
        "Random"
    }

    fn emoji(&self) -> &str {
        "🐰"
    }

    fn reset(&mut self) {}

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── P2: Greedy (Model proxy) ──────────────────────────────────

/// P2: Model-based player — state machine for purposeful movement.
///
/// Uses a state machine (Explore/Flee) for committed, non-oscillating movement:
/// - Explore: walk in a direction until blocked, bomb near destructible walls
/// - Flee: BFS pathfind to nearest safe cell when in blast zone
///
/// 20% random exploration to avoid predictability.
pub struct GreedyPlayer {
    _id: u8,
    state: AiState,
    known_bombs: Vec<((i32, i32), u32)>,
}

impl GreedyPlayer {
    pub fn new(id: u8) -> Self {
        Self {
            _id: id,
            state: AiState::default(),
            known_bombs: Vec::new(),
        }
    }
}

impl BomberPlayer for GreedyPlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        rng: &mut Rng,
    ) -> BomberAction {
        update_bombs(&mut self.known_bombs, events);

        // ── Evaluate: transition state if needed ──
        let in_danger = in_blast_zone(pos, &self.known_bombs);

        if in_danger {
            self.state = AiState::Flee;
        } else {
            match self.state {
                AiState::Flee => {
                    // Safe now → hunt a wall or explore
                    if let Some(wall) = find_nearest_wall(grid, pos) {
                        self.state = AiState::Hunt { target: wall };
                    } else {
                        self.state = AiState::Explore {
                            dir: pick_explore_dir(grid, pos, None, &self.known_bombs, rng),
                        };
                    }
                }
                AiState::Hunt { target } => {
                    // Target destroyed? Find new one
                    if !matches!(
                        grid.get(target.0, target.1),
                        super::Cell::DestructibleWall | super::Cell::PowerUpHidden(_)
                    ) {
                        if let Some(wall) = find_nearest_wall(grid, pos) {
                            self.state = AiState::Hunt { target: wall };
                        } else {
                            self.state = AiState::Explore {
                                dir: pick_explore_dir(grid, pos, None, &self.known_bombs, rng),
                            };
                        }
                    }
                }
                AiState::Explore { .. } => {}
            }
        }

        // 20% random exploration (only when safe and not hunting)
        if !in_danger && rng.f32() < 0.2 && !matches!(self.state, AiState::Hunt { .. }) {
            let idx = rng.usize(0..ACTION_COUNT);
            let action = index_to_action(idx);
            let target = move_target(&action, pos);
            if action == BomberAction::Bomb
                || action == BomberAction::Wait
                || grid.is_walkable(target.x, target.y)
            {
                if matches!(
                    action,
                    BomberAction::Up
                        | BomberAction::Down
                        | BomberAction::Left
                        | BomberAction::Right
                ) {
                    self.state = AiState::Explore { dir: action };
                }
                return action;
            }
        }

        // ── Execute current state ──
        match self.state {
            AiState::Flee => {
                if let Some(safe) = find_safe_cell(grid, pos, &self.known_bombs)
                    && let Some(step) = next_step_toward(grid, pos, safe)
                {
                    return step;
                }
                BomberAction::Wait
            }
            AiState::Hunt { target } => {
                // Adjacent to target? Bomb it
                let adj = (pos.x - target.0).abs() + (pos.y - target.1).abs();
                if adj == 1 && should_place_bomb(grid, pos, &self.known_bombs) {
                    self.known_bombs.push(((pos.x, pos.y), DEFAULT_BLAST_RANGE));
                    self.state = AiState::Flee;
                    return BomberAction::Bomb;
                }
                // Walk toward target
                match step_toward_wall(grid, pos, target) {
                    Some(step) if !in_blast_zone(move_target(&step, pos), &self.known_bombs) => {
                        step
                    }
                    _ => {
                        // Unreachable or unsafe → explore
                        let new_dir = pick_explore_dir(grid, pos, None, &self.known_bombs, rng);
                        self.state = AiState::Explore { dir: new_dir };
                        new_dir
                    }
                }
            }
            AiState::Explore { dir } => {
                // Bomb opportunity — always bomb when conditions are met
                if should_place_bomb(grid, pos, &self.known_bombs) {
                    self.known_bombs.push(((pos.x, pos.y), DEFAULT_BLAST_RANGE));
                    self.state = AiState::Flee;
                    return BomberAction::Bomb;
                }

                // Continue direction if walkable AND safe from blasts
                let target = move_target(&dir, pos);
                if matches!(
                    dir,
                    BomberAction::Up
                        | BomberAction::Down
                        | BomberAction::Left
                        | BomberAction::Right
                ) && grid.is_walkable(target.x, target.y)
                    && !in_blast_zone(target, &self.known_bombs)
                {
                    dir
                } else {
                    // Blocked → try hunt, else new direction
                    if let Some(wall) = find_nearest_wall(grid, pos) {
                        self.state = AiState::Hunt { target: wall };
                        return step_toward_wall(grid, pos, wall).unwrap_or(BomberAction::Wait);
                    }
                    let new_dir = pick_explore_dir(grid, pos, Some(dir), &self.known_bombs, rng);
                    self.state = AiState::Explore { dir: new_dir };
                    new_dir
                }
            }
        }
    }

    fn name(&self) -> &str {
        "Greedy"
    }

    fn emoji(&self) -> &str {
        "🐱"
    }

    fn reset(&mut self) {
        self.state = AiState::default();
        self.known_bombs.clear();
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── P3: Validator ──────────────────────────────────────────────

/// P3: Model + Validator — state machine with safety validation.
///
/// Same state machine as P2 (Explore/Flee) but adds a safety validation layer:
/// - State machine picks action, then validates against hard safety rules
/// - Falls back to heuristic scoring if state machine action is unsafe
/// - Never walk into active blast zones, walls, or place bomb without escape route
pub struct ValidatorPlayer {
    _id: u8,
    known_bombs: Vec<((i32, i32), u32)>,
    state: AiState,
}

impl ValidatorPlayer {
    pub fn new(id: u8) -> Self {
        Self {
            _id: id,
            known_bombs: Vec::new(),
            state: AiState::default(),
        }
    }
}

impl BomberPlayer for ValidatorPlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        rng: &mut Rng,
    ) -> BomberAction {
        update_bombs(&mut self.known_bombs, events);

        // ── Evaluate: transition state if needed ──
        let in_danger = in_blast_zone(pos, &self.known_bombs);

        if in_danger {
            self.state = AiState::Flee;
        } else {
            match self.state {
                AiState::Flee => {
                    if let Some(wall) = find_nearest_wall(grid, pos) {
                        self.state = AiState::Hunt { target: wall };
                    } else {
                        self.state = AiState::Explore {
                            dir: pick_explore_dir(grid, pos, None, &self.known_bombs, rng),
                        };
                    }
                }
                AiState::Hunt { target } => {
                    if !matches!(
                        grid.get(target.0, target.1),
                        super::Cell::DestructibleWall | super::Cell::PowerUpHidden(_)
                    ) {
                        if let Some(wall) = find_nearest_wall(grid, pos) {
                            self.state = AiState::Hunt { target: wall };
                        } else {
                            self.state = AiState::Explore {
                                dir: pick_explore_dir(grid, pos, None, &self.known_bombs, rng),
                            };
                        }
                    }
                }
                AiState::Explore { .. } => {}
            }
        }

        // ── Execute: state machine picks action ──
        let action = match self.state {
            AiState::Flee => {
                if let Some(safe) = find_safe_cell(grid, pos, &self.known_bombs)
                    && let Some(step) = next_step_toward(grid, pos, safe)
                {
                    step
                } else {
                    BomberAction::Wait
                }
            }
            AiState::Hunt { target } => {
                let adj = (pos.x - target.0).abs() + (pos.y - target.1).abs();
                if adj == 1 && should_place_bomb(grid, pos, &self.known_bombs) {
                    BomberAction::Bomb
                } else {
                    match step_toward_wall(grid, pos, target) {
                        Some(step)
                            if !in_blast_zone(move_target(&step, pos), &self.known_bombs) =>
                        {
                            step
                        }
                        _ => {
                            let new_dir = pick_explore_dir(grid, pos, None, &self.known_bombs, rng);
                            self.state = AiState::Explore { dir: new_dir };
                            new_dir
                        }
                    }
                }
            }
            AiState::Explore { dir } => {
                // Bomb opportunity (safety validated below)
                if should_place_bomb(grid, pos, &self.known_bombs) {
                    BomberAction::Bomb
                } else {
                    // Continue direction if walkable AND safe from blasts
                    let target = move_target(&dir, pos);
                    if matches!(
                        dir,
                        BomberAction::Up
                            | BomberAction::Down
                            | BomberAction::Left
                            | BomberAction::Right
                    ) && grid.is_walkable(target.x, target.y)
                        && !in_blast_zone(target, &self.known_bombs)
                    {
                        dir
                    } else {
                        // Blocked → try hunt, else new direction
                        if let Some(wall) = find_nearest_wall(grid, pos) {
                            self.state = AiState::Hunt { target: wall };
                            step_toward_wall(grid, pos, wall).unwrap_or(BomberAction::Wait)
                        } else {
                            let new_dir =
                                pick_explore_dir(grid, pos, Some(dir), &self.known_bombs, rng);
                            self.state = AiState::Explore { dir: new_dir };
                            new_dir
                        }
                    }
                }
            }
        };

        // ── Safety validation ──
        if is_safe_action(&action, grid, pos, &self.known_bombs) {
            match action {
                BomberAction::Up
                | BomberAction::Down
                | BomberAction::Left
                | BomberAction::Right => {
                    // Preserve Hunt state during pathfinding; update dir for Explore
                    if !matches!(self.state, AiState::Hunt { .. }) {
                        self.state = AiState::Explore { dir: action };
                    }
                }
                BomberAction::Bomb => {
                    self.known_bombs.push(((pos.x, pos.y), DEFAULT_BLAST_RANGE));
                    self.state = AiState::Flee;
                }
                BomberAction::Wait => {}
            }
            return action;
        }

        // Fallback: pick best safe action via heuristic
        let mut safe_actions: Vec<(BomberAction, f32)> = Vec::new();
        for a in &ALL_ACTIONS {
            if is_safe_action(a, grid, pos, &self.known_bombs) {
                let score = heuristic_score(a, grid, pos, &self.known_bombs, None);
                safe_actions.push((*a, score));
            }
        }

        if !safe_actions.is_empty() {
            safe_actions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let chosen = safe_actions[0].0;
            if matches!(
                chosen,
                BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
            ) {
                self.state = AiState::Explore { dir: chosen };
            }
            return chosen;
        }

        BomberAction::Wait
    }

    fn name(&self) -> &str {
        "Validator"
    }

    fn emoji(&self) -> &str {
        "🐶"
    }

    fn reset(&mut self) {
        self.known_bombs.clear();
        self.state = AiState::default();
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── P4: Full HL ────────────────────────────────────────────────

/// P4: Full HL — bandit-adapted action selection with absorb-compress.
///
/// Same base as P3 but uses a simple bandit over the 6 actions to adapt
/// relevance scores based on observed outcomes. Compresses stable low-Q
/// arms into hard blocks over time.
pub struct HLPlayer {
    _id: u8,
    known_bombs: Vec<((i32, i32), u32)>,
    q_values: [f32; ACTION_COUNT],
    visits: [u32; ACTION_COUNT],
    total_pulls: u32,
    compressed: [bool; ACTION_COUNT],
    round_actions: Vec<BomberAction>,
    last_dir: Option<BomberAction>,
}

impl HLPlayer {
    pub fn new(id: u8) -> Self {
        Self {
            _id: id,
            known_bombs: Vec::new(),
            q_values: [0.0; ACTION_COUNT],
            visits: [0; ACTION_COUNT],
            total_pulls: 0,
            compressed: [false; ACTION_COUNT],
            round_actions: Vec::new(),
            last_dir: None,
        }
    }

    /// Update bandit Q-values based on round outcome.
    ///
    /// Distributes reward across ALL actions taken this round (not just the last).
    /// This prevents misattribution where only the final action gets blamed for death.
    pub fn update_outcome(
        &mut self,
        survived: bool,
        killed_opponent: bool,
        collected_powerups: u32,
    ) {
        if self.round_actions.is_empty() {
            return;
        }

        // Base reward shaping
        let base_reward = if survived { 1.0 } else { -1.0 }
            + if killed_opponent { 0.5 } else { 0.0 }
            + collected_powerups as f32 * 0.2;

        // Count action frequency for proportional update
        let mut action_counts = [0u32; ACTION_COUNT];
        for action in &self.round_actions {
            action_counts[action_index(action)] += 1;
        }

        // Update Q-values for each unique action taken this round
        for (idx, &count) in action_counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            // Weight reward by how often this action was taken
            let proportion = count as f32 / self.round_actions.len() as f32;
            let reward = base_reward * proportion;

            self.visits[idx] += 1;
            self.total_pulls += 1;
            let n = self.visits[idx] as f32;
            self.q_values[idx] += (reward - self.q_values[idx]) / n;
        }
    }

    /// Run absorb-compress cycle. Returns newly compressed arm indices.
    pub fn compress_cycle(&mut self) -> Vec<usize> {
        let min_visits = 20u32;
        let threshold = 0.1f32;
        let mut newly_compressed = Vec::new();

        for i in 0..ACTION_COUNT {
            if self.compressed[i] {
                continue;
            }
            if self.visits[i] >= min_visits && self.q_values[i] < threshold {
                self.compressed[i] = true;
                newly_compressed.push(i);
            }
        }

        newly_compressed
    }

    /// Generate a compression report string.
    pub fn compress_report(&self) -> String {
        let compressed_count = self.compressed.iter().filter(|&&c| c).count();
        let compressed_names: Vec<String> = self
            .compressed
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c)
            .map(|(i, _)| format!("{}({:.2})", index_to_action(i), self.q_values[i]))
            .collect();

        format!(
            "Pulls={} Compressed={}/{} [{}] Q=[{}]",
            self.total_pulls,
            compressed_count,
            ACTION_COUNT,
            compressed_names.join(","),
            self.q_values
                .iter()
                .enumerate()
                .map(|(i, q)| format!("{}:{:.2}", index_to_action(i), q))
                .collect::<Vec<_>>()
                .join(" "),
        )
    }
}

impl BomberPlayer for HLPlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        rng: &mut Rng,
    ) -> BomberAction {
        update_bombs(&mut self.known_bombs, events);

        // Compute blended scores: 60% heuristic + 40% bandit Q-value
        let mut scores: [(BomberAction, f32); ACTION_COUNT] = ALL_ACTIONS.map(|a| (a, 0.0));

        for (i, action) in ALL_ACTIONS.iter().enumerate() {
            // Skip compressed (hard-blocked) arms
            if self.compressed[i] {
                scores[i] = (*action, f32::NEG_INFINITY);
                continue;
            }

            let h = heuristic_score(action, grid, pos, &self.known_bombs, self.last_dir);

            // Domain hard block (walking into wall) overrides everything
            if h <= -1.0 {
                scores[i] = (*action, h);
                continue;
            }

            // Safety validation — penalize unsafe actions
            let safe = is_safe_action(action, grid, pos, &self.known_bombs);
            let safety_bonus = if safe { 0.0 } else { -0.5 };

            // Bandit Q-value component (default 0.0 for unvisited arms)
            let bandit_q = if self.visits[i] > 0 {
                self.q_values[i]
            } else {
                0.0
            };

            // Blend: 60% heuristic + 40% bandit + safety
            let blended = h * 0.6 + bandit_q * 0.4 + safety_bonus;
            scores[i] = (*action, blended);
        }

        // ε-greedy: 10% explore, 90% exploit
        if rng.f32() < 0.1 {
            // Pick a random non-compressed action
            let valid: Vec<usize> = (0..ACTION_COUNT)
                .filter(|&i| !self.compressed[i] && scores[i].1 > f32::NEG_INFINITY)
                .collect();
            if !valid.is_empty() {
                let pick = valid[rng.usize(0..valid.len())];
                let action = scores[pick].0;
                self.round_actions.push(action);
                if matches!(
                    action,
                    BomberAction::Up
                        | BomberAction::Down
                        | BomberAction::Left
                        | BomberAction::Right
                ) {
                    self.last_dir = Some(action);
                }
                return action;
            }
        }

        // Pick best action
        let best = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(a, _)| *a)
            .unwrap_or(BomberAction::Wait);

        self.round_actions.push(best);
        if matches!(
            best,
            BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
        ) {
            self.last_dir = Some(best);
        }
        best
    }

    fn name(&self) -> &str {
        "HL"
    }

    fn emoji(&self) -> &str {
        "🐵"
    }

    fn reset(&mut self) {
        self.known_bombs.clear();
        self.round_actions.clear();
        self.last_dir = None;
        // NOTE: Q-values, visits, compressed persist across rounds (bandit memory)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── Factory ────────────────────────────────────────────────────

/// Create the 4 player instances for a tournament.
pub fn create_players() -> Vec<Box<dyn BomberPlayer>> {
    vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(ValidatorPlayer::new(2)),
        Box::new(HLPlayer::new(3)),
    ]
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_grid() -> ArenaGrid {
        ArenaGrid::generate(42)
    }

    #[test]
    fn test_random_player_valid_actions() {
        let mut player = RandomPlayer::new(0);
        let grid = empty_grid();
        let mut rng = Rng::with_seed(42);
        let pos = GridPos { x: 1, y: 1 }; // Spawn position — always walkable

        for _ in 0..50 {
            let action = player.select_action(&grid, pos, &[], &mut rng);
            // Should never walk into a wall
            if action != BomberAction::Bomb && action != BomberAction::Wait {
                let target = move_target(&action, pos);
                assert!(
                    grid.is_walkable(target.x, target.y),
                    "RandomPlayer walked into wall at ({},{})",
                    target.x,
                    target.y,
                );
            }
        }
    }

    #[test]
    fn test_greedy_player_prefers_safety() {
        let mut player = GreedyPlayer::new(1);
        let grid = empty_grid();
        let mut rng = Rng::with_seed(42);
        let pos = GridPos { x: 3, y: 3 };

        // Without bombs, should prefer valid moves
        let action = player.select_action(&grid, pos, &[], &mut rng);
        if action != BomberAction::Bomb && action != BomberAction::Wait {
            let target = move_target(&action, pos);
            assert!(grid.is_walkable(target.x, target.y));
        }
    }

    #[test]
    fn test_validator_player_rejects_unsafe() {
        let mut player = ValidatorPlayer::new(2);
        let grid = empty_grid();
        let mut rng = Rng::with_seed(42);
        let pos = GridPos { x: 3, y: 3 };

        // With a bomb aimed at us, should avoid blast zone
        let events = vec![GameEvent::BombPlaced {
            player: 0,
            pos: (3, 1),
        }];
        player.known_bombs = vec![((3, 1), 2)];

        let action = player.select_action(&grid, pos, &events, &mut rng);
        // Should not move into blast zone (3,1 has range 2, so (3,3) is in blast)
        // The player at (3,3) is in blast zone — should try to escape
        if action != BomberAction::Bomb && action != BomberAction::Wait {
            let target = move_target(&action, pos);
            // Moving out of blast zone is preferred
            assert!(
                target.x != 3 || target.y < 1 || target.y > 3,
                "Validator should escape blast zone, moved to ({},{})",
                target.x,
                target.y,
            );
        }
    }

    #[test]
    fn test_hl_player_adapts() {
        let mut player = HLPlayer::new(3);
        let _grid = empty_grid();
        let _rng = Rng::with_seed(42);
        let _pos = GridPos { x: 3, y: 3 };

        // Simulate several rounds with good outcomes for Up
        for _ in 0..25 {
            player.round_actions.clear();
            // Push Up as the only action for this round
            player.round_actions.push(BomberAction::Up);
            player.update_outcome(true, false, 0);
        }

        // Q-value for Up should be positive
        let up_idx = action_index(&BomberAction::Up);
        assert!(
            player.q_values[up_idx] > 0.0,
            "HL should learn Up is good, Q={}",
            player.q_values[up_idx],
        );
    }
}
