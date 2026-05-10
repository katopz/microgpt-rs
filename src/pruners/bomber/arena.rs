use bevy_ecs::prelude::Resource;

use super::{ARENA_H, ARENA_W, Cell, DESTRUCTIBLE_FILL, PowerUpKind, SPAWN_POSITIONS};

/// The 13×13 Bomberman arena grid.
///
/// Grid coordinates: `cells[y][x]` where (0,0) is top-left.
/// Standard Bomberman layout: fixed walls at even row/col intersections,
/// destructible walls randomly placed, 3×3 corners kept clear for spawns.
#[derive(Clone, Debug, Resource)]
pub struct ArenaGrid {
    /// Grid cells: `cells[y][x]`
    pub cells: Vec<Vec<Cell>>,
    pub width: usize,
    pub height: usize,
}

impl ArenaGrid {
    /// Generate a 13×13 arena grid from the given seed.
    #[allow(clippy::needless_range_loop)]
    pub fn generate(seed: u64) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut cells = vec![vec![Cell::Floor; ARENA_W]; ARENA_H];

        // Border walls
        for y in 0..ARENA_H {
            for x in 0..ARENA_W {
                if x == 0 || x == ARENA_W - 1 || y == 0 || y == ARENA_H - 1 {
                    cells[y][x] = Cell::FixedWall;
                }
            }
        }

        // Interior pillars at even x, even y (0-indexed)
        for y in 2..ARENA_H - 1 {
            for x in 2..ARENA_W - 1 {
                if x % 2 == 0 && y % 2 == 0 {
                    cells[y][x] = Cell::FixedWall;
                }
            }
        }

        // Destructible walls + hidden power-ups (~40% fill, exclude spawn zones)
        for y in 1..ARENA_H - 1 {
            for x in 1..ARENA_W - 1 {
                if cells[y][x] != Cell::Floor || Self::is_in_spawn_zone(x, y) {
                    continue;
                }
                if rng.f32() < DESTRUCTIBLE_FILL {
                    cells[y][x] = Self::random_destructible(&mut rng);
                }
            }
        }

        Self {
            cells,
            width: ARENA_W,
            height: ARENA_H,
        }
    }

    /// Pick `DestructibleWall` or `PowerUpHidden` (20% power-up chance).
    fn random_destructible(rng: &mut fastrand::Rng) -> Cell {
        match rng.f32() < 0.2 {
            true => {
                let kind = match rng.u8(0..3) {
                    0 => PowerUpKind::BombUp,
                    1 => PowerUpKind::FireUp,
                    _ => PowerUpKind::SpeedUp,
                };
                Cell::PowerUpHidden(kind)
            }
            false => Cell::DestructibleWall,
        }
    }

    /// Check if (x, y) is within any player's 3×3 spawn safe zone.
    fn is_in_spawn_zone(x: usize, y: usize) -> bool {
        SPAWN_POSITIONS.iter().any(|&(sx, sy)| {
            (x as i32 - sx).unsigned_abs() <= 1 && (y as i32 - sy).unsigned_abs() <= 1
        })
    }

    /// Safe cell access. Returns `FixedWall` for out-of-bounds.
    pub fn get(&self, x: i32, y: i32) -> Cell {
        match self.is_in_bounds(x, y) {
            true => self.cells[y as usize][x as usize],
            false => Cell::FixedWall,
        }
    }

    /// Set cell at (x, y). No-op for out-of-bounds.
    pub fn set(&mut self, x: i32, y: i32, cell: Cell) {
        if !self.is_in_bounds(x, y) {
            return;
        }
        self.cells[y as usize][x as usize] = cell;
    }

    /// True if the cell is walkable (`Floor` or `PowerUpHidden`).
    pub fn is_walkable(&self, x: i32, y: i32) -> bool {
        matches!(self.get(x, y), Cell::Floor | Cell::PowerUpHidden(_))
    }

    /// True if (x, y) is within grid bounds.
    pub fn is_in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && (x as usize) < self.width && y >= 0 && (y as usize) < self.height
    }

    /// Returns the spawn position for the given player (0..3).
    pub fn spawn_pos(&self, player_id: u8) -> (i32, i32) {
        SPAWN_POSITIONS[player_id as usize]
    }

    /// Clear spawn zones of destructible walls and power-ups for safe respawning.
    pub fn clear_for_respawn(&mut self) {
        for &(sx, sy) in &SPAWN_POSITIONS {
            for dy in -1_i32..=1 {
                for dx in -1_i32..=1 {
                    let (x, y) = (sx + dx, sy + dy);
                    match self.get(x, y) {
                        Cell::DestructibleWall | Cell::PowerUpHidden(_) => {
                            self.set(x, y, Cell::Floor);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_dimensions() {
        let grid = ArenaGrid::generate(42);
        assert_eq!(grid.width, 13);
        assert_eq!(grid.height, 13);
        assert_eq!(grid.cells.len(), 13);
        assert!(grid.cells.iter().all(|row| row.len() == 13));
    }

    #[test]
    fn test_border_walls() {
        let grid = ArenaGrid::generate(42);
        for y in 0..13 {
            for x in 0..13 {
                if x == 0 || x == 12 || y == 0 || y == 12 {
                    assert_eq!(grid.cells[y][x], Cell::FixedWall, "border ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn test_fixed_walls_pattern() {
        let grid = ArenaGrid::generate(42);
        for y in 2..11 {
            for x in 2..11 {
                if x % 2 == 0 && y % 2 == 0 {
                    assert_eq!(grid.cells[y][x], Cell::FixedWall, "pillar ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn test_corners_clear() {
        let grid = ArenaGrid::generate(123);
        for &(sx, sy) in &SPAWN_POSITIONS {
            for dy in -1_i32..=1 {
                for dx in -1_i32..=1 {
                    let (x, y) = (sx + dx, sy + dy);
                    if x < 1 || x > 11 || y < 1 || y > 11 {
                        continue;
                    }
                    match grid.cells[y as usize][x as usize] {
                        Cell::Floor | Cell::FixedWall => {}
                        other => panic!("spawn ({x},{y}) has {other:?}"),
                    }
                }
            }
        }
    }

    #[test]
    fn test_seed_reproducibility() {
        let a = ArenaGrid::generate(999);
        let b = ArenaGrid::generate(999);
        assert_eq!(a.cells, b.cells);
    }
}
