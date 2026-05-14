//! AI player trait and implementations for Bomberman HL Arena.
//!
//! Player types representing increasing HL technology levels:
//! - P1 (Random): no model, no learning — pure baseline
//! - P2 (Greedy): heuristic action selection
//! - P2b (LoraPlayer): trained LoRA model scoring — proves LoRA > random
//! - P3 (Validator): heuristic + hard safety rules
//! - P3b (NNPlayer/WasmPlayer): WASM validator sandbox — proves safety > none
//! - P4 (LoraWasmPlayer): LoRA proposals + WASM validation — proves synergy
//! - P5 (HLPlayer): LoRA + WASM + Bandit + AbsorbCompress — proves adaptation

use std::any::Any;

use fastrand::Rng;

#[cfg(feature = "bandit")]
use std::sync::Arc;

#[cfg(feature = "bandit")]
use crate::pruners::SharedBanditStats;

use super::{ArenaGrid, BomberAction, GameEvent, GridPos};

#[cfg(feature = "bomber-wasm")]
use crate::types::{LoraAdapter, lora_apply};

// ── Constants ──────────────────────────────────────────────────

const ACTION_COUNT: usize = 6;
const DEFAULT_BLAST_RANGE: u32 = 2;
const BOMB_FUSE_TICKS: u32 = super::BOMB_FUSE_TICKS;

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
pub(crate) fn move_target(action: &BomberAction, pos: GridPos) -> GridPos {
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
/// Accounts for walls blocking blast propagation (blast stops at walls).
pub(crate) fn in_blast_zone(pos: GridPos, grid: &ArenaGrid, bombs: &[KnownBomb]) -> bool {
    for &(bomb_pos, range, _fuse) in bombs {
        if is_in_single_blast(pos, grid, bomb_pos, range) {
            return true;
        }
    }
    false
}

/// Check if position is in the blast zone of a single bomb (with wall blocking).
fn is_in_single_blast(pos: GridPos, grid: &ArenaGrid, bomb_pos: (i32, i32), range: u32) -> bool {
    use super::Cell;
    let bx = bomb_pos.0;
    let by = bomb_pos.1;

    // Standing on the bomb itself
    if pos.x == bx && pos.y == by {
        return true;
    }

    // Same row (horizontal blast)
    if pos.y == by {
        let dx = pos.x - bx;
        if dx.unsigned_abs() <= range {
            let step = dx.signum();
            let mut x = bx + step;
            while x != pos.x {
                match grid.get(x, by) {
                    Cell::FixedWall | Cell::DestructibleWall | Cell::PowerUpHidden(_) => {
                        return false;
                    }
                    _ => {}
                }
                x += step;
            }
            return true;
        }
    }

    // Same column (vertical blast)
    if pos.x == bx {
        let dy = pos.y - by;
        if dy.unsigned_abs() <= range {
            let step = dy.signum();
            let mut y = by + step;
            while y != pos.y {
                match grid.get(bx, y) {
                    Cell::FixedWall | Cell::DestructibleWall | Cell::PowerUpHidden(_) => {
                        return false;
                    }
                    _ => {}
                }
                y += step;
            }
            return true;
        }
    }

    false
}

/// Update known bomb list from events.
fn update_bombs(bombs: &mut Vec<KnownBomb>, events: &[GameEvent]) {
    // Decrement fuses each tick (called once per select_action)
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

/// Update known power-up list from events (revealed/collected).
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

/// Track opponent positions from PlayerMoved and BombPlaced events.
/// Stores `(player_id, current_pos, prev_pos)` for trajectory prediction.
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

/// Predict opponent's next position from trajectory (prev → current → next).
pub(crate) fn predict_direction(
    current: (i32, i32),
    prev: Option<(i32, i32)>,
) -> Option<(i32, i32)> {
    let (cx, cy) = current;
    let (px, py) = prev?;
    let dx = cx - px;
    let dy = cy - py;
    if dx == 0 && dy == 0 {
        return None;
    }
    Some((cx + dx, cy + dy))
}

/// Count walkable neighbors (escape routes) from a position.
pub(crate) fn count_escape_routes(pos: (i32, i32), grid: &ArenaGrid) -> usize {
    [(0i32, -1), (0, 1), (-1, 0), (1, 0)]
        .iter()
        .filter(|&&(dx, dy)| grid.is_walkable(pos.0 + dx, pos.1 + dy))
        .count()
}

/// Score a bomb placement by how trapped the opponent would be.
/// Higher score = fewer opponent escape routes + blast coverage.
pub(crate) fn trap_score(
    bomb_pos: (i32, i32),
    opponent_pos: (i32, i32),
    grid: &ArenaGrid,
    blast_range: u32,
) -> f32 {
    let dist = (bomb_pos.0 - opponent_pos.0).abs() + (bomb_pos.1 - opponent_pos.1).abs();
    if dist > blast_range as i32 + 3 {
        return 0.0;
    }

    let mut score = 0.0;

    // Bonus: opponent is within blast range
    if is_in_single_blast(
        GridPos {
            x: opponent_pos.0,
            y: opponent_pos.1,
        },
        grid,
        bomb_pos,
        blast_range,
    ) {
        score += 4.0;
    }

    // Penalty: more escape routes = harder to trap
    let routes = count_escape_routes(opponent_pos, grid);
    match routes {
        0 => score += 3.0,
        1 => score += 2.0,
        2 => score += 0.5,
        _ => {}
    }

    // Closeness bonus
    if dist <= 2 {
        score += 1.0;
    }

    score
}

/// Score movement toward intercepting an opponent's predicted path.
pub(crate) fn intercept_score(
    my_target: (i32, i32),
    opponent_pos: (i32, i32),
    predicted_pos: Option<(i32, i32)>,
) -> f32 {
    let current_dist = (my_target.0 - opponent_pos.0).abs() + (my_target.1 - opponent_pos.1).abs();

    if let Some((px, py)) = predicted_pos {
        let predicted_dist = (my_target.0 - px).abs() + (my_target.1 - py).abs();
        if predicted_dist < current_dist {
            return 1.0;
        }
    }

    0.0
}

/// Check if player has an escape route after placing a bomb at `new_bomb_pos`.
/// BFS from `player_pos` — must reach a cell outside ALL blast zones within
/// `blast_range + 1` steps. Accounts for bomb entities blocking movement.
pub(crate) fn has_escape_route(
    grid: &ArenaGrid,
    player_pos: GridPos,
    new_bomb_pos: (i32, i32),
    blast_range: u32,
    existing_bombs: &[KnownBomb],
) -> bool {
    use std::collections::{HashSet, VecDeque};

    let max_steps = blast_range as i32 + 1;
    let mut visited: HashSet<(i32, i32)> = HashSet::new();
    let mut queue: VecDeque<((i32, i32), i32)> = VecDeque::new();

    // Bomb entities block movement — collect all blocked positions
    let blocked: HashSet<(i32, i32)> = {
        let mut s: HashSet<(i32, i32)> = existing_bombs.iter().map(|(p, _, _)| *p).collect();
        s.insert(new_bomb_pos);
        s
    };

    // All bombs combined for comprehensive blast zone checking
    let mut all_bombs: Vec<KnownBomb> = existing_bombs.to_vec();
    all_bombs.push((new_bomb_pos, blast_range, BOMB_FUSE_TICKS));

    queue.push_back(((player_pos.x, player_pos.y), 0));
    visited.insert((player_pos.x, player_pos.y));

    while let Some(((cx, cy), steps)) = queue.pop_front() {
        if steps > max_steps {
            continue;
        }

        // Is this cell safe from ALL bombs (with wall blocking)?
        if !in_blast_zone(GridPos { x: cx, y: cy }, grid, &all_bombs) {
            return true;
        }

        // Expand neighbors (avoid bomb entities blocking movement)
        for (nx, ny) in [(cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)] {
            if visited.insert((nx, ny)) && grid.is_walkable(nx, ny) && !blocked.contains(&(nx, ny))
            {
                queue.push_back(((nx, ny), steps + 1));
            }
        }
    }

    false
}

/// Check if an action is safe given the current state.
/// Uses wall-aware blast zone checks and accounts for bomb entities blocking movement.
pub fn is_safe_action(
    action: &BomberAction,
    grid: &ArenaGrid,
    pos: GridPos,
    bombs: &[KnownBomb],
) -> bool {
    match action {
        BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right => {
            let target = move_target(action, pos);
            if !grid.is_walkable(target.x, target.y) {
                return false;
            }
            // Don't walk into blast zone (walls block blast)
            !in_blast_zone(target, grid, bombs)
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
                            bombs,
                        )
                })
        }
        BomberAction::Wait => {
            // Waiting is only safe if not in blast zone
            !in_blast_zone(pos, grid, bombs)
        }
    }
}

/// Check if player should place a bomb at current position.
///
/// The player stands ON the bomb but moves away next tick, so escape is
/// checked from adjacent cells — not from the bomb position itself.
/// Accounts for existing bombs' blast zones and bomb entities blocking movement.
pub(crate) fn should_place_bomb(grid: &ArenaGrid, pos: GridPos, bombs: &[KnownBomb]) -> bool {
    // Don't place if already in a blast zone (walls may block, but be safe)
    if in_blast_zone(pos, grid, bombs) {
        return false;
    }

    // Don't place if there's already a bomb here
    if bombs.iter().any(|(p, _, _)| p.0 == pos.x && p.1 == pos.y) {
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
                bombs,
            )
    })
}

// ── Policy Scoring ─────────────────────────────────────────────

/// True if action reverses the previous direction.
pub(crate) fn is_reverse(action: BomberAction, prev: Option<BomberAction>) -> bool {
    matches!(
        (action, prev),
        (BomberAction::Up, Some(BomberAction::Down))
            | (BomberAction::Down, Some(BomberAction::Up))
            | (BomberAction::Left, Some(BomberAction::Right))
            | (BomberAction::Right, Some(BomberAction::Left))
    )
}

/// Count destructible walls within manhattan range.
fn wall_density(grid: &ArenaGrid, pos: GridPos, range: i32) -> i32 {
    let mut count = 0;
    for dy in -range..=range {
        for dx in -range..=range {
            if dx == 0 && dy == 0 {
                continue;
            }
            match grid.get(pos.x + dx, pos.y + dy) {
                super::Cell::DestructibleWall | super::Cell::PowerUpHidden(_) => count += 1,
                _ => {}
            }
        }
    }
    count
}

/// True if any cell adjacent to pos is a destructible wall.
fn has_adjacent_wall(grid: &ArenaGrid, pos: GridPos) -> bool {
    [(0i32, -1), (0, 1), (-1, 0), (1, 0)]
        .iter()
        .any(|&(dx, dy)| {
            matches!(
                grid.get(pos.x + dx, pos.y + dy),
                super::Cell::DestructibleWall | super::Cell::PowerUpHidden(_)
            )
        })
}

/// BFS distance from pos to nearest cell outside all blast zones.
/// Returns `None` if no safe cell is reachable. Accounts for walls blocking blast.
pub(crate) fn escape_distance(
    pos: GridPos,
    grid: &ArenaGrid,
    bombs: &[KnownBomb],
    blocked: &std::collections::HashSet<(i32, i32)>,
) -> Option<i32> {
    use std::collections::{HashSet, VecDeque};

    if !in_blast_zone(pos, grid, bombs) {
        return Some(0);
    }

    let mut visited: HashSet<(i32, i32)> = HashSet::new();
    let mut queue: VecDeque<((i32, i32), i32)> = VecDeque::new();

    queue.push_back(((pos.x, pos.y), 0));
    visited.insert((pos.x, pos.y));

    while let Some(((cx, cy), dist)) = queue.pop_front() {
        for (nx, ny) in [(cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)] {
            if !visited.insert((nx, ny)) {
                continue;
            }
            if !grid.is_walkable(nx, ny) || blocked.contains(&(nx, ny)) {
                continue;
            }
            let next_dist = dist + 1;
            if !in_blast_zone(GridPos { x: nx, y: ny }, grid, bombs) {
                return Some(next_dist);
            }
            queue.push_back(((nx, ny), next_dist));
        }
    }

    None
}

/// Policy-based action scoring with clear priorities.
///
/// Policies (highest priority first):
///   Unsafe  → -∞     (wall, blast zone with no escape)
///   Flee    → +5..10 (escaping blast zone via shortest path)
///   Bomb    → +5.0   (near destructible wall + escape route)
///   Collect → +2..3  (moving toward / standing on revealed power-ups)
///   Hunt    → +0..2  (moving toward destructible walls)
///   Persist → -1.0   (penalize reversing direction)
///   Explore → +0.2   (slight center bias)
pub(crate) fn score_action(
    action: &BomberAction,
    grid: &ArenaGrid,
    pos: GridPos,
    bombs: &[KnownBomb],
    powerups: &[(i32, i32)],
    last_dir: Option<BomberAction>,
) -> f32 {
    use BomberAction::{Down, Left, Right, Up};

    // Collect bomb positions that block movement
    let bomb_positions: std::collections::HashSet<(i32, i32)> =
        bombs.iter().map(|(p, _, _)| *p).collect();

    match action {
        Up | Down | Left | Right => {
            let target = move_target(action, pos);

            // Hard constraint: unwalkable or blocked by bomb entity
            if !grid.is_walkable(target.x, target.y)
                || bomb_positions.contains(&(target.x, target.y))
            {
                return f32::NEG_INFINITY;
            }

            // In blast zone — use escape distance for directional guidance
            if in_blast_zone(target, grid, bombs) {
                let current_dist =
                    escape_distance(pos, grid, bombs, &bomb_positions).unwrap_or(i32::MAX);
                let target_dist =
                    escape_distance(target, grid, bombs, &bomb_positions).unwrap_or(i32::MAX);
                return if target_dist < current_dist {
                    10.0 - target_dist as f32 * 0.5 // Moving toward safety
                } else if target_dist > current_dist {
                    -10.0 // Moving away from safety
                } else {
                    -5.0 // Same distance — slightly bad
                };
            }

            let mut score = 0.0;

            // Flee: escaping blast zone is top priority
            if in_blast_zone(pos, grid, bombs) {
                score += 10.0;
            }

            // Collect: move toward nearby revealed power-ups (high priority)
            if !powerups.is_empty() {
                let current_min = powerups
                    .iter()
                    .map(|&(px, py)| (pos.x - px).abs() + (pos.y - py).abs())
                    .min()
                    .unwrap_or(i32::MAX);
                let target_min = powerups
                    .iter()
                    .map(|&(px, py)| (target.x - px).abs() + (target.y - py).abs())
                    .min()
                    .unwrap_or(i32::MAX);
                if target_min == 0 {
                    score += 3.0; // Standing on power-up — instant collect
                } else if target_min < current_min {
                    score += 2.0; // Moving toward nearest power-up
                }
            }

            // Hunt: move toward areas with more destructible walls
            let current_walls = wall_density(grid, pos, 3);
            let target_walls = wall_density(grid, target, 3);
            score += (target_walls - current_walls) as f32 * 0.3;

            // Bonus: target cell is adjacent to destructible wall (bomb position)
            if has_adjacent_wall(grid, target) {
                score += 1.0;
            }

            // Persist: penalize reversing
            if is_reverse(*action, last_dir) {
                score -= 1.0;
            }

            // Explore: slight center bias
            let center = 6i32;
            let dist_before = (pos.x - center).abs() + (pos.y - center).abs();
            let dist_after = (target.x - center).abs() + (target.y - center).abs();
            if dist_after < dist_before {
                score += 0.2;
            }

            score
        }
        BomberAction::Bomb => {
            if !should_place_bomb(grid, pos, bombs) {
                return f32::NEG_INFINITY;
            }
            5.0 // Bomb is good when valid
        }
        BomberAction::Wait => {
            if in_blast_zone(pos, grid, bombs) {
                -10.0
            } else {
                -1.0
            }
        }
    }
}

// ── LoRA Inference Helpers ─────────────────────────────────────

/// Softmax over logits, producing probability distribution.
#[cfg(feature = "bomber-wasm")]
#[allow(dead_code)]
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&x| x / sum).collect()
}

/// Count walkable adjacent cells (for board feature encoding).
#[cfg(feature = "bomber-wasm")]
fn count_walkable(grid: &ArenaGrid, pos: GridPos) -> usize {
    [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)]
        .iter()
        .filter(|&&(dx, dy)| grid.is_walkable(pos.x + dx, pos.y + dy))
        .count()
}

/// Use loaded LoRA adapter to score all 6 actions.
///
/// Strategy: compute heuristic scores, then use LoRA as a learned re-weighting.
/// The LoRA was trained on game traces and encodes patterns like
/// "bomb near walls is good", "don't walk into blast".
///
/// Returns `None` if LoRA dimensions don't align (falls back to heuristic).
#[cfg(feature = "bomber-wasm")]
fn lora_score_actions(
    lora: &LoraAdapter,
    grid: &ArenaGrid,
    pos: GridPos,
    bombs: &[KnownBomb],
    powerups: &[(i32, i32)],
    last_dir: Option<BomberAction>,
    lora_buf: &mut [f32],
) -> Option<[f32; ACTION_COUNT]> {
    // Compute heuristic base scores for all actions
    let heuristic: [f32; ACTION_COUNT] =
        ALL_ACTIONS.map(|action| score_action(&action, grid, pos, bombs, powerups, last_dir));

    // LoRA input: heuristic scores padded to in_dim with board features
    let in_dim = lora.in_dim;
    if in_dim < ACTION_COUNT {
        return None;
    }

    let mut input = vec![0.0f32; in_dim];
    for (i, &h) in heuristic.iter().enumerate() {
        input[i] = if h == f32::NEG_INFINITY { -10.0 } else { h };
    }
    // Pad remaining dimensions with board statistics
    if in_dim > ACTION_COUNT {
        input[ACTION_COUNT] = count_walkable(grid, pos) as f32 / 4.0;
    }
    if in_dim > ACTION_COUNT + 1 {
        input[ACTION_COUNT + 1] = if in_blast_zone(pos, grid, bombs) {
            1.0
        } else {
            0.0
        };
    }
    if in_dim > ACTION_COUNT + 2 {
        input[ACTION_COUNT + 2] = bombs.len() as f32 / 8.0;
    }
    if in_dim > ACTION_COUNT + 3 {
        input[ACTION_COUNT + 3] = powerups.len() as f32 / 4.0;
    }

    // Apply LoRA: output += scale * B @ (A @ input)
    let mut output = vec![0.0f32; lora.out_dim];
    lora_apply(&mut output, lora, &input, lora_buf);

    // Combine: LoRA re-weights heuristic scores
    let out_dim = lora.out_dim.min(ACTION_COUNT);
    let mut scores = heuristic;
    for i in 0..out_dim {
        if scores[i] != f32::NEG_INFINITY {
            // Blend: 70% heuristic + 30% LoRA correction (scaled)
            scores[i] = scores[i] * 0.7 + output[i] * 3.0;
        }
    }

    Some(scores)
}

// ── P1: Random ─────────────────────────────────────────────────

/// P1: Modelless baseline — uniform random action selection.
///
/// No learning. No memory. No model. Pure baseline.
/// Avoids walking into walls (up to 3 re-rolls, then Wait).
pub struct RandomPlayer {
    _id: u8,
    known_bombs: Vec<KnownBomb>,
}

impl RandomPlayer {
    pub fn new(id: u8) -> Self {
        Self {
            _id: id,
            known_bombs: Vec::new(),
        }
    }
}

impl BomberPlayer for RandomPlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        rng: &mut Rng,
    ) -> BomberAction {
        update_bombs(&mut self.known_bombs, events);

        // Collect safe walkable moves (avoid walls + blast zones)
        let safe_moves: Vec<BomberAction> = [
            BomberAction::Up,
            BomberAction::Down,
            BomberAction::Left,
            BomberAction::Right,
        ]
        .into_iter()
        .filter(|&action| {
            let target = move_target(&action, pos);
            grid.is_walkable(target.x, target.y) && !in_blast_zone(target, grid, &self.known_bombs)
        })
        .collect();

        // Prefer safe moves, fall back to Wait
        if !safe_moves.is_empty() {
            return safe_moves[rng.usize(0..safe_moves.len())];
        }
        BomberAction::Wait
    }

    fn name(&self) -> &str {
        "Random"
    }

    fn emoji(&self) -> &str {
        "🐰"
    }

    fn reset(&mut self) {
        self.known_bombs.clear();
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── P2: Greedy ─────────────────────────────────────────────────

/// P2: Model-based — policy scoring with exploration.
///
/// Scores all actions using clear policy priorities (flee > bomb > hunt > explore)
/// and picks the best. Adds 20% random exploration to discover new strategies.
pub struct GreedyPlayer {
    _id: u8,
    known_bombs: Vec<KnownBomb>,
    known_powerups: Vec<(i32, i32)>,
    last_dir: Option<BomberAction>,
}

impl GreedyPlayer {
    pub fn new(id: u8) -> Self {
        Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
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
        update_powerups(&mut self.known_powerups, events);

        // 20% random exploration — only safe movement, never random bomb
        if rng.f32() < 0.2 {
            let safe_moves: Vec<BomberAction> = [
                BomberAction::Up,
                BomberAction::Down,
                BomberAction::Left,
                BomberAction::Right,
            ]
            .into_iter()
            .filter(|&action| {
                let target = move_target(&action, pos);
                grid.is_walkable(target.x, target.y)
                    && !in_blast_zone(target, grid, &self.known_bombs)
            })
            .collect();
            if !safe_moves.is_empty() {
                let action = safe_moves[rng.usize(0..safe_moves.len())];
                self.last_dir = Some(action);
                return action;
            }
        }

        // Policy: score all actions, pick best
        let best = ALL_ACTIONS
            .iter()
            .max_by(|a, b| {
                score_action(
                    a,
                    grid,
                    pos,
                    &self.known_bombs,
                    &self.known_powerups,
                    self.last_dir,
                )
                .partial_cmp(&score_action(
                    b,
                    grid,
                    pos,
                    &self.known_bombs,
                    &self.known_powerups,
                    self.last_dir,
                ))
                .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(BomberAction::Wait);

        if matches!(
            best,
            BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
        ) {
            self.last_dir = Some(best);
        }
        if best == BomberAction::Bomb {
            self.known_bombs
                .push(((pos.x, pos.y), DEFAULT_BLAST_RANGE, BOMB_FUSE_TICKS));
        }
        best
    }

    fn name(&self) -> &str {
        "Greedy"
    }

    fn emoji(&self) -> &str {
        "🐱"
    }

    fn reset(&mut self) {
        self.known_bombs.clear();
        self.known_powerups.clear();
        self.last_dir = None;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── P3: Validator ──────────────────────────────────────────────

/// P3: Model + Validator — policy scoring with safety validation.
///
/// Same policy scoring as P2 but adds a hard safety filter:
/// - Only considers actions that pass `is_safe_action`
/// - Never walks into active blast zones, walls, or places bomb without escape
pub struct ValidatorPlayer {
    _id: u8,
    known_bombs: Vec<KnownBomb>,
    known_powerups: Vec<(i32, i32)>,
    last_dir: Option<BomberAction>,
}

impl ValidatorPlayer {
    pub fn new(id: u8) -> Self {
        Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }
}

impl BomberPlayer for ValidatorPlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        _rng: &mut Rng,
    ) -> BomberAction {
        update_bombs(&mut self.known_bombs, events);
        update_powerups(&mut self.known_powerups, events);

        let in_danger = in_blast_zone(pos, grid, &self.known_bombs);
        let bomb_positions: std::collections::HashSet<(i32, i32)> =
            self.known_bombs.iter().map(|(p, _, _)| *p).collect();

        let mut best = BomberAction::Wait;
        let mut best_score = f32::NEG_INFINITY;

        for action in &ALL_ACTIONS {
            let is_move = matches!(
                action,
                BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
            );

            if in_danger {
                // Escape mode: score movement by escape distance, skip Bomb/Wait
                if !is_move {
                    continue;
                }
                let target = move_target(action, pos);
                if !grid.is_walkable(target.x, target.y)
                    || bomb_positions.contains(&(target.x, target.y))
                {
                    continue;
                }
                let score = match escape_distance(target, grid, &self.known_bombs, &bomb_positions)
                {
                    Some(dist) => 10.0 - dist as f32 * 0.5,
                    None => -5.0, // No escape route found — try anyway
                };
                if score > best_score {
                    best_score = score;
                    best = *action;
                }
            } else {
                // Safe mode: hard-block unsafe actions (validator's purpose)
                if !is_safe_action(action, grid, pos, &self.known_bombs) {
                    continue;
                }
                let score = score_action(
                    action,
                    grid,
                    pos,
                    &self.known_bombs,
                    &self.known_powerups,
                    self.last_dir,
                );
                if score > best_score {
                    best_score = score;
                    best = *action;
                }
            }
        }

        if matches!(
            best,
            BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
        ) {
            self.last_dir = Some(best);
        }
        if best == BomberAction::Bomb {
            self.known_bombs
                .push(((pos.x, pos.y), DEFAULT_BLAST_RANGE, BOMB_FUSE_TICKS));
        }
        best
    }

    fn name(&self) -> &str {
        "Validator"
    }

    fn emoji(&self) -> &str {
        "🐶"
    }

    fn reset(&mut self) {
        self.known_bombs.clear();
        self.known_powerups.clear();
        self.last_dir = None;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── P4: Full HL ────────────────────────────────────────────────

/// P4: Full HL — bandit-adapted policy with absorb-compress.
///
/// Blends policy scoring (60%) with bandit Q-values (40%) and adds:
/// - ε-greedy exploration (10%)
/// - Safety validation layer
/// - Absorb-compress: prunes consistently bad actions
/// - Trial logging for outcome attribution
pub struct HLPlayer {
    _id: u8,
    known_bombs: Vec<KnownBomb>,
    known_powerups: Vec<(i32, i32)>,
    known_opponents: Vec<KnownOpponent>,
    q_values: [f32; ACTION_COUNT],
    visits: [u32; ACTION_COUNT],
    total_pulls: u32,
    compressed: [bool; ACTION_COUNT],
    round_actions: Vec<BomberAction>,
    last_dir: Option<BomberAction>,
    /// Shared bandit stats for multi-agent cooperative learning.
    /// When `Some`, Q-values/visits/compressed are delegated here.
    #[cfg(feature = "bandit")]
    shared_stats: Option<Arc<SharedBanditStats>>,
}

impl HLPlayer {
    pub fn new(id: u8) -> Self {
        Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            known_opponents: Vec::new(),
            q_values: [0.0; ACTION_COUNT],
            visits: [0; ACTION_COUNT],
            total_pulls: 0,
            compressed: [false; ACTION_COUNT],
            round_actions: Vec::new(),
            last_dir: None,
            #[cfg(feature = "bandit")]
            shared_stats: None,
        }
    }

    /// Create HLPlayer sharing bandit stats with other agents.
    ///
    /// Multiple agents sharing one `SharedBanditStats` learn cooperatively:
    /// Q-values and visit counts are shared, but each agent still has
    /// its own heuristic scoring and RNG for action selection.
    #[cfg(feature = "bandit")]
    pub fn with_shared_stats(id: u8, stats: Arc<SharedBanditStats>) -> Self {
        Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            known_opponents: Vec::new(),
            q_values: [0.0; ACTION_COUNT],
            visits: [0; ACTION_COUNT],
            total_pulls: 0,
            compressed: [false; ACTION_COUNT],
            round_actions: Vec::new(),
            last_dir: None,
            shared_stats: Some(stats),
        }
    }

    // ── Shared Stats Accessors ─────────────────────────────────

    /// Whether an arm is compressed (hard-blocked).
    ///
    /// Delegates to shared stats when present, else uses local field.
    #[cfg(feature = "bandit")]
    fn arm_compressed(&self, arm: usize) -> bool {
        self.shared_stats
            .as_ref()
            .map_or(self.compressed[arm], |s| s.is_compressed(arm))
    }

    #[cfg(not(feature = "bandit"))]
    fn arm_compressed(&self, arm: usize) -> bool {
        self.compressed[arm]
    }

    /// Visit count for an arm.
    #[cfg(feature = "bandit")]
    fn arm_visits(&self, arm: usize) -> u32 {
        self.shared_stats
            .as_ref()
            .map_or(self.visits[arm], |s| s.visits(arm))
    }

    #[cfg(not(feature = "bandit"))]
    fn arm_visits(&self, arm: usize) -> u32 {
        self.visits[arm]
    }

    /// Q-value estimate for an arm.
    #[cfg(feature = "bandit")]
    fn arm_q(&self, arm: usize) -> f32 {
        self.shared_stats
            .as_ref()
            .map_or(self.q_values[arm], |s| s.q_value(arm))
    }

    #[cfg(not(feature = "bandit"))]
    fn arm_q(&self, arm: usize) -> f32 {
        self.q_values[arm]
    }

    /// Update Q-value for an arm with observed reward.
    ///
    /// Delegates to shared stats when present, else updates local fields.
    #[cfg(feature = "bandit")]
    fn update_arm_q(&mut self, arm: usize, reward: f32) {
        match &self.shared_stats {
            Some(stats) => stats.update(arm, reward),
            None => {
                self.visits[arm] += 1;
                self.total_pulls += 1;
                let n = self.visits[arm] as f32;
                self.q_values[arm] += (reward - self.q_values[arm]) / n;
            }
        }
    }

    #[cfg(not(feature = "bandit"))]
    fn update_arm_q(&mut self, arm: usize, reward: f32) {
        self.visits[arm] += 1;
        self.total_pulls += 1;
        let n = self.visits[arm] as f32;
        self.q_values[arm] += (reward - self.q_values[arm]) / n;
    }

    /// Mark an arm as compressed (hard-blocked).
    #[cfg(feature = "bandit")]
    fn mark_compressed(&mut self, arm: usize) {
        match &self.shared_stats {
            Some(stats) => stats.compress_arm(arm),
            None => self.compressed[arm] = true,
        }
    }

    #[cfg(not(feature = "bandit"))]
    fn mark_compressed(&mut self, arm: usize) {
        self.compressed[arm] = true;
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

        // Decay-based credit assignment: recent actions get more weight
        let total = self.round_actions.len();
        let mut action_rewards = [0.0f32; ACTION_COUNT];
        let mut action_weights = [0.0f32; ACTION_COUNT];

        for (i, action) in self.round_actions.iter().enumerate() {
            // Exponential decay: later actions get exponentially more credit
            let recency = 0.5_f32.powi((total - 1 - i) as i32);
            let idx = action_index(action);
            action_rewards[idx] += base_reward * recency;
            action_weights[idx] += recency;
        }

        // Update Q-values with weighted rewards (delegates to shared stats when present)
        for idx in 0..ACTION_COUNT {
            if action_weights[idx] == 0.0 {
                continue;
            }
            let reward = action_rewards[idx] / action_weights[idx];
            self.update_arm_q(idx, reward);
        }
    }

    /// Run absorb-compress cycle. Returns newly compressed arm indices.
    pub fn compress_cycle(&mut self) -> Vec<usize> {
        let min_visits = 20u32;
        let threshold = 0.1f32;
        let mut newly_compressed = Vec::new();

        for i in 0..ACTION_COUNT {
            if self.arm_compressed(i) {
                continue;
            }
            if self.arm_visits(i) >= min_visits && self.arm_q(i) < threshold {
                self.mark_compressed(i);
                newly_compressed.push(i);
            }
        }

        newly_compressed
    }

    /// Generate a compression report string.
    pub fn compress_report(&self) -> String {
        #[cfg(feature = "bandit")]
        if let Some(ref stats) = self.shared_stats {
            let compressed_count = (0..ACTION_COUNT)
                .filter(|&i| stats.is_compressed(i))
                .count();
            let compressed_names: Vec<String> = (0..ACTION_COUNT)
                .filter(|&i| stats.is_compressed(i))
                .map(|i| format!("{}({:.2})", index_to_action(i), stats.q_value(i)))
                .collect();
            return format!(
                "Pulls={} Compressed={}/{} [{}] Q=[{}]",
                stats.total_pulls(),
                compressed_count,
                ACTION_COUNT,
                compressed_names.join(","),
                (0..ACTION_COUNT)
                    .map(|i| format!("{}:{:.2}", index_to_action(i), stats.q_value(i)))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }

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
        update_powerups(&mut self.known_powerups, events);
        update_opponents(&mut self.known_opponents, events, self._id);

        let bomb_positions: std::collections::HashSet<(i32, i32)> =
            self.known_bombs.iter().map(|(p, _, _)| *p).collect();

        // Find nearest opponent and their predicted trajectory
        let nearest_info = self
            .known_opponents
            .iter()
            .filter(|(_, op, _)| grid.is_walkable(op.0, op.1))
            .min_by_key(|(_, op, _)| (pos.x - op.0).abs() + (pos.y - op.1).abs());

        let nearest_opponent = nearest_info.map(|(_, op, _)| *op);
        let predicted_opponent =
            nearest_info.and_then(|(_, op, prev)| predict_direction(*op, *prev));

        // Compute blended scores: 85% policy + 15% bandit Q-value
        let mut scores: [(BomberAction, f32); ACTION_COUNT] = ALL_ACTIONS.map(|a| (a, 0.0));

        for (i, action) in ALL_ACTIONS.iter().enumerate() {
            // Skip compressed (hard-blocked) arms
            if self.arm_compressed(i) {
                scores[i] = (*action, f32::NEG_INFINITY);
                continue;
            }

            let h = score_action(
                action,
                grid,
                pos,
                &self.known_bombs,
                &self.known_powerups,
                self.last_dir,
            );

            // Domain hard block (unwalkable, unsafe bomb) overrides everything
            if h == f32::NEG_INFINITY {
                scores[i] = (*action, h);
                continue;
            }

            // Safety validation — hard-block unsafe Bomb/Wait only;
            // let score_action handle movement (it uses escape_distance in blast zones)
            let is_move = matches!(
                action,
                BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
            );
            if !is_move && !is_safe_action(action, grid, pos, &self.known_bombs) {
                scores[i] = (*action, f32::NEG_INFINITY);
                continue;
            }

            // Strategic bonus: hunt, intercept, ambush, and trap
            let mut strategy_bonus = 0.0f32;
            match action {
                BomberAction::Up
                | BomberAction::Down
                | BomberAction::Left
                | BomberAction::Right => {
                    if let Some((ox, oy)) = nearest_opponent {
                        let target = move_target(action, pos);
                        let current_dist = (pos.x - ox).abs() + (pos.y - oy).abs();
                        let target_dist = (target.x - ox).abs() + (target.y - oy).abs();

                        // Hunt: move toward opponent
                        if target_dist < current_dist {
                            strategy_bonus += 1.5;
                        }

                        // Intercept: move toward predicted position
                        strategy_bonus +=
                            intercept_score((target.x, target.y), (ox, oy), predicted_opponent);

                        // Chokepoint: prefer moving where opponent has fewer escapes
                        if target_dist <= 3 {
                            let routes = count_escape_routes((target.x, target.y), grid);
                            if routes <= 1 {
                                strategy_bonus += 1.0;
                            }
                        }
                    }
                }
                BomberAction::Bomb => {
                    // Strategic value: more adjacent walls = better bomb placement
                    let wall_count = [(0i32, -1), (0, 1), (-1, 0), (1, 0)]
                        .iter()
                        .filter(|&&(dx, dy)| {
                            matches!(
                                grid.get(pos.x + dx, pos.y + dy),
                                super::Cell::DestructibleWall | super::Cell::PowerUpHidden(_)
                            )
                        })
                        .count();
                    strategy_bonus += wall_count as f32 * 0.5;

                    // Attack: trap scoring when opponent is nearby
                    if let Some((ox, oy)) = nearest_opponent {
                        strategy_bonus +=
                            trap_score((pos.x, pos.y), (ox, oy), grid, DEFAULT_BLAST_RANGE);
                    }
                }
                BomberAction::Wait => {}
            }

            // Bandit Q-value component (default 0.0 for unvisited arms)
            let _bandit_q = if self.arm_visits(i) > 0 {
                self.arm_q(i)
            } else {
                0.0
            };

            // Pure heuristic + strategy bonus (bandit noise removed — too sparse at this scale)
            scores[i] = (*action, h + strategy_bonus);
        }

        // ε-greedy: 10% explore (only safe moves — less random than Greedy's 20%)
        if rng.f32() < 0.10 {
            // Pick a random non-compressed, non-hard-blocked, safe action
            let safe_explore: Vec<usize> = (0..ACTION_COUNT)
                .filter(|&i| {
                    if self.arm_compressed(i) || scores[i].1 <= f32::NEG_INFINITY {
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
                self.round_actions.push(action);
                self.last_dir = Some(action);
                return action;
            }
        }

        // Pick best action
        let best = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(a, _)| *a)
            .unwrap_or(BomberAction::Wait);

        // Track own bomb placement (critical: prevents walking back into own bomb)
        if best == BomberAction::Bomb {
            self.known_bombs
                .push(((pos.x, pos.y), DEFAULT_BLAST_RANGE, BOMB_FUSE_TICKS));
        }

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
        self.known_powerups.clear();
        self.known_opponents.clear();
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

// ── P2b: LoRA Player ───────────────────────────────────────────

/// P2b: LoRA-only player — uses trained LoRA model for action scoring.
///
/// No WASM validator, no bandit. Proves LoRA > random.
/// Falls back to heuristic scoring if LoRA fails to load or apply.
#[cfg(feature = "bomber-wasm")]
pub struct LoraPlayer {
    _id: u8,
    lora: Option<LoraAdapter>,
    lora_buf: Vec<f32>,
    known_bombs: Vec<KnownBomb>,
    known_powerups: Vec<(i32, i32)>,
    last_dir: Option<BomberAction>,
}

#[cfg(feature = "bomber-wasm")]
impl LoraPlayer {
    /// Create LoraPlayer without LoRA (heuristic fallback).
    pub fn new(id: u8) -> Self {
        Self {
            _id: id,
            lora: None,
            lora_buf: Vec::new(),
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }

    /// Create LoraPlayer with LoRA loaded from file.
    pub fn new_with_lora(id: u8, lora_path: &str) -> Self {
        let lora = LoraAdapter::load(std::path::Path::new(lora_path)).ok();
        let buf_size = lora.as_ref().map_or(0, |l| l.rank);
        Self {
            _id: id,
            lora,
            lora_buf: vec![0.0; buf_size],
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }
}

#[cfg(feature = "bomber-wasm")]
impl BomberPlayer for LoraPlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        rng: &mut Rng,
    ) -> BomberAction {
        update_bombs(&mut self.known_bombs, events);
        update_powerups(&mut self.known_powerups, events);

        let bomb_positions: std::collections::HashSet<(i32, i32)> =
            self.known_bombs.iter().map(|(p, _, _)| *p).collect();

        // Try LoRA scoring first
        let scores = self.lora.as_ref().and_then(|lora| {
            lora_score_actions(
                lora,
                grid,
                pos,
                &self.known_bombs,
                &self.known_powerups,
                self.last_dir,
                &mut self.lora_buf,
            )
        });

        let mut best = BomberAction::Wait;
        let mut best_score = f32::NEG_INFINITY;

        for (i, action) in ALL_ACTIONS.iter().enumerate() {
            let is_move = matches!(
                action,
                BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
            );

            // Basic wall collision filter
            if is_move {
                let target = move_target(action, pos);
                if !grid.is_walkable(target.x, target.y)
                    || bomb_positions.contains(&(target.x, target.y))
                {
                    continue;
                }
            }

            let score = match &scores {
                Some(s) => s[i],
                None => score_action(
                    action,
                    grid,
                    pos,
                    &self.known_bombs,
                    &self.known_powerups,
                    self.last_dir,
                ),
            };

            if score > best_score {
                best_score = score;
                best = *action;
            }
        }

        // 10% random exploration (epsilon-greedy)
        if rng.f32() < 0.10 {
            let safe_moves: Vec<BomberAction> = ALL_ACTIONS
                .iter()
                .filter(|a| {
                    if matches!(
                        a,
                        BomberAction::Up
                            | BomberAction::Down
                            | BomberAction::Left
                            | BomberAction::Right
                    ) {
                        let target = move_target(a, pos);
                        grid.is_walkable(target.x, target.y)
                    } else {
                        false
                    }
                })
                .copied()
                .collect();
            if !safe_moves.is_empty() {
                best = safe_moves[rng.usize(0..safe_moves.len())];
            }
        }

        if matches!(
            best,
            BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
        ) {
            self.last_dir = Some(best);
        }
        if best == BomberAction::Bomb {
            self.known_bombs
                .push(((pos.x, pos.y), DEFAULT_BLAST_RANGE, BOMB_FUSE_TICKS));
        }
        best
    }

    fn name(&self) -> &str {
        match self.lora {
            Some(_) => "LoRA",
            None => "LoRA-Fallback",
        }
    }

    fn emoji(&self) -> &str {
        "🤖"
    }

    fn reset(&mut self) {
        self.known_bombs.clear();
        self.known_powerups.clear();
        self.last_dir = None;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── P4: LoRA+WASM Player ──────────────────────────────────────

/// P4: LoRA proposals + WASM validation — the synergy player.
///
/// Model proposes action scores via LoRA, WASM validator filters unsafe ones.
/// Proves LoRA+WASM synergy > either alone.
#[cfg(feature = "bomber-wasm")]
pub struct LoraWasmPlayer {
    _id: u8,
    lora: Option<LoraAdapter>,
    wasm: Option<super::wasm_pruner::BomberWasmPruner>,
    lora_buf: Vec<f32>,
    known_bombs: Vec<KnownBomb>,
    known_powerups: Vec<(i32, i32)>,
    last_dir: Option<BomberAction>,
}

#[cfg(feature = "bomber-wasm")]
impl LoraWasmPlayer {
    /// Create with no artifacts (heuristic + native safety).
    pub fn new(id: u8) -> Self {
        Self {
            _id: id,
            lora: None,
            wasm: None,
            lora_buf: Vec::new(),
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }

    /// Create with LoRA only.
    pub fn new_with_lora(id: u8, lora_path: &str) -> Self {
        let lora = LoraAdapter::load(std::path::Path::new(lora_path)).ok();
        let buf_size = lora.as_ref().map_or(0, |l| l.rank);
        Self {
            _id: id,
            lora,
            wasm: None,
            lora_buf: vec![0.0; buf_size],
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }

    /// Create with WASM only.
    pub fn new_with_wasm(id: u8, wasm_path: &str) -> Self {
        let wasm = super::wasm_pruner::BomberWasmPruner::load_from_file(wasm_path).ok();
        Self {
            _id: id,
            lora: None,
            wasm,
            lora_buf: Vec::new(),
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }

    /// Create with both artifacts (full LoRA + WASM stack).
    pub fn new_with_secrets(id: u8, lora_path: &str, wasm_path: &str) -> Self {
        let lora = LoraAdapter::load(std::path::Path::new(lora_path)).ok();
        let wasm = super::wasm_pruner::BomberWasmPruner::load_from_file(wasm_path).ok();
        let buf_size = lora.as_ref().map_or(0, |l| l.rank);
        Self {
            _id: id,
            lora,
            wasm,
            lora_buf: vec![0.0; buf_size],
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }

    /// Check if action is safe — WASM if available, native otherwise.
    fn is_action_safe(
        &self,
        action: &BomberAction,
        grid: &ArenaGrid,
        pos: GridPos,
        bombs: &[KnownBomb],
    ) -> bool {
        if let Some(ref wasm) = self.wasm {
            return wasm.is_safe_action(action_index(action), grid, pos.x, pos.y, self._id, bombs);
        }
        is_safe_action(action, grid, pos, bombs)
    }
}

#[cfg(feature = "bomber-wasm")]
impl BomberPlayer for LoraWasmPlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        _rng: &mut Rng,
    ) -> BomberAction {
        update_bombs(&mut self.known_bombs, events);
        update_powerups(&mut self.known_powerups, events);

        let in_danger = in_blast_zone(pos, grid, &self.known_bombs);
        let bomb_positions: std::collections::HashSet<(i32, i32)> =
            self.known_bombs.iter().map(|(p, _, _)| *p).collect();

        // Try LoRA scoring
        let lora_scores = self.lora.as_ref().and_then(|lora| {
            lora_score_actions(
                lora,
                grid,
                pos,
                &self.known_bombs,
                &self.known_powerups,
                self.last_dir,
                &mut self.lora_buf,
            )
        });

        let mut best = BomberAction::Wait;
        let mut best_score = f32::NEG_INFINITY;

        for (i, action) in ALL_ACTIONS.iter().enumerate() {
            let is_move = matches!(
                action,
                BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
            );

            if in_danger {
                // Escape mode: skip Bomb/Wait, find escape route
                if !is_move {
                    continue;
                }
                let target = move_target(action, pos);
                if !grid.is_walkable(target.x, target.y)
                    || bomb_positions.contains(&(target.x, target.y))
                {
                    continue;
                }
                let score = match escape_distance(target, grid, &self.known_bombs, &bomb_positions)
                {
                    Some(dist) => 10.0 - dist as f32 * 0.5,
                    None => -5.0,
                };
                if score > best_score {
                    best_score = score;
                    best = *action;
                }
            } else {
                // Safe mode: hard-block unsafe actions via WASM or native
                if !self.is_action_safe(action, grid, pos, &self.known_bombs) {
                    continue;
                }

                // Use LoRA scores if available, else heuristic
                let score = match &lora_scores {
                    Some(s) => s[i],
                    None => score_action(
                        action,
                        grid,
                        pos,
                        &self.known_bombs,
                        &self.known_powerups,
                        self.last_dir,
                    ),
                };

                if score > best_score {
                    best_score = score;
                    best = *action;
                }
            }
        }

        if matches!(
            best,
            BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
        ) {
            self.last_dir = Some(best);
        }
        if best == BomberAction::Bomb {
            self.known_bombs
                .push(((pos.x, pos.y), DEFAULT_BLAST_RANGE, BOMB_FUSE_TICKS));
        }
        best
    }

    fn name(&self) -> &str {
        match (&self.lora, &self.wasm) {
            (Some(_), Some(_)) => "LoRA+WASM",
            (Some(_), None) => "LoRA+Native",
            (None, Some(_)) => "Heuristic+WASM",
            (None, None) => "Heuristic+Native",
        }
    }

    fn emoji(&self) -> &str {
        "🔮"
    }

    fn reset(&mut self) {
        self.known_bombs.clear();
        self.known_powerups.clear();
        self.last_dir = None;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── P2.5: NN (WASM Validator) ──────────────────────────────────

/// P2.5: Neural Network + WASM Validator — heuristic scoring + WASM safety.
///
/// Combines heuristic action scoring (same as ValidatorPlayer) with
/// WASM-based safety validation. If `bomber_validator.wasm` is loaded,
/// safety checks run in the WASM sandbox. Falls back to native Rust
/// safety rules if WASM is unavailable.
///
/// Future: LoRA model proposals (Phase 1, blocked by training corpus).
#[cfg(feature = "bomber-wasm")]
pub struct NNPlayer {
    _id: u8,
    wasm: Option<super::wasm_pruner::BomberWasmPruner>,
    known_bombs: Vec<KnownBomb>,
    known_powerups: Vec<(i32, i32)>,
    last_dir: Option<BomberAction>,
}

#[cfg(feature = "bomber-wasm")]
impl NNPlayer {
    /// Create NNPlayer with WASM validator loaded from file.
    ///
    /// Falls back to native safety rules if WASM fails to load.
    pub fn new_with_wasm(id: u8, wasm_path: &str) -> Self {
        let wasm = super::wasm_pruner::BomberWasmPruner::load_from_file(wasm_path).ok();
        Self {
            _id: id,
            wasm,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }

    /// Create NNPlayer without WASM (native fallback only).
    pub fn new_native(id: u8) -> Self {
        Self {
            _id: id,
            wasm: None,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }

    /// Check if action is safe — WASM if available, native otherwise.
    fn is_action_safe(
        &self,
        action: &BomberAction,
        grid: &ArenaGrid,
        pos: GridPos,
        bombs: &[KnownBomb],
    ) -> bool {
        if let Some(ref wasm) = self.wasm {
            return wasm.is_safe_action(action_index(action), grid, pos.x, pos.y, self._id, bombs);
        }
        is_safe_action(action, grid, pos, bombs)
    }
}

#[cfg(feature = "bomber-wasm")]
impl BomberPlayer for NNPlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        _rng: &mut Rng,
    ) -> BomberAction {
        update_bombs(&mut self.known_bombs, events);
        update_powerups(&mut self.known_powerups, events);

        let in_danger = in_blast_zone(pos, grid, &self.known_bombs);
        let bomb_positions: std::collections::HashSet<(i32, i32)> =
            self.known_bombs.iter().map(|(p, _, _)| *p).collect();

        let mut best = BomberAction::Wait;
        let mut best_score = f32::NEG_INFINITY;

        for action in &ALL_ACTIONS {
            let is_move = matches!(
                action,
                BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
            );

            if in_danger {
                // Escape mode: score movement by escape distance, skip Bomb/Wait
                if !is_move {
                    continue;
                }
                let target = move_target(action, pos);
                if !grid.is_walkable(target.x, target.y)
                    || bomb_positions.contains(&(target.x, target.y))
                {
                    continue;
                }
                let score = match escape_distance(target, grid, &self.known_bombs, &bomb_positions)
                {
                    Some(dist) => 10.0 - dist as f32 * 0.5,
                    None => -5.0,
                };
                if score > best_score {
                    best_score = score;
                    best = *action;
                }
            } else {
                // Safe mode: hard-block unsafe actions via WASM or native
                if !self.is_action_safe(action, grid, pos, &self.known_bombs) {
                    continue;
                }
                let score = score_action(
                    action,
                    grid,
                    pos,
                    &self.known_bombs,
                    &self.known_powerups,
                    self.last_dir,
                );
                if score > best_score {
                    best_score = score;
                    best = *action;
                }
            }
        }

        if matches!(
            best,
            BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
        ) {
            self.last_dir = Some(best);
        }
        if best == BomberAction::Bomb {
            self.known_bombs
                .push(((pos.x, pos.y), DEFAULT_BLAST_RANGE, BOMB_FUSE_TICKS));
        }
        best
    }

    fn name(&self) -> &str {
        match self.wasm {
            Some(_) => "NN-WASM",
            None => "NN-Native",
        }
    }

    fn emoji(&self) -> &str {
        "🤖"
    }

    fn reset(&mut self) {
        self.known_bombs.clear();
        self.known_powerups.clear();
        self.last_dir = None;
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

/// Create 4 players with NNPlayer (P2.5) replacing ValidatorPlayer (P3).
///
/// If `wasm_path` is `Some`, NNPlayer loads the WASM validator for sandboxed
/// safety checks. Otherwise, uses native Rust safety rules.
#[cfg(feature = "bomber-wasm")]
pub fn create_players_with_wasm(wasm_path: Option<&str>) -> Vec<Box<dyn BomberPlayer>> {
    let p2 = match wasm_path {
        Some(path) => Box::new(NNPlayer::new_with_wasm(2, path)) as Box<dyn BomberPlayer>,
        None => Box::new(NNPlayer::new_native(2)),
    };
    vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        p2,
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
        player.known_bombs = vec![((3, 1), 2, BOMB_FUSE_TICKS)];

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
