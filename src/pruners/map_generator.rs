//! Procedural Map Generator for testing tactical puzzles.
//!
//! Generates solvable single-floor and multi-floor maps using seeded
//! randomness for reproducibility. Uses random-walk wall placement
//! and BFS connectivity verification.
//!
//! # Usage
//!
//! ```ignore
//! use crate::pruners::map_generator::MapGenerator;
//!
//! let mut gen = MapGenerator::new(42)
//!     .with_width(10)
//!     .with_height(10)
//!     .with_monsters(3)
//!     .with_treasures(2)
//!     .with_wall_density(0.15)
//!     .with_terrain();
//!
//! let map = gen.generate_single_floor().expect("solvable map");
//! let map_str = map.to_map_string();
//! let pruner = TacticalPruner::new(&map_str);
//! ```

use crate::pruners::pathfinder::{is_passable, reachable_positions};
use std::collections::HashSet;

// ── Public Types ───────────────────────────────────────────────

/// Result of procedural generation for a single floor.
#[derive(Clone, Debug)]
pub struct GeneratedMap {
    pub grid: Vec<Vec<char>>,
    pub start: (usize, usize),
    pub goal: (usize, usize),
    pub monsters: Vec<(usize, usize)>,
    pub treasures: Vec<(usize, usize)>,
}

/// Stairs connecting two floors in a multi-floor dungeon.
#[derive(Clone, Debug)]
pub struct Stairs {
    pub pos: (usize, usize),
    pub from_floor: usize,
    pub to_floor: usize,
}

/// Multi-floor dungeon representation.
#[derive(Clone, Debug)]
pub struct DungeonMap {
    pub floors: Vec<GeneratedMap>,
    pub stairs: Vec<Stairs>,
}

/// Result of multi-floor procedural generation.
#[derive(Clone, Debug)]
pub struct GeneratedDungeon {
    pub map: DungeonMap,
}

/// Configuration for map generation.
///
/// Use builder methods to customize, then call `generate_single_floor`
/// or `generate_multi_floor`.
#[derive(Clone, Debug)]
pub struct MapGenerator {
    pub width: usize,
    pub height: usize,
    pub num_monsters: usize,
    pub num_treasures: usize,
    pub wall_density: f32,
    pub terrain_mix: bool,
    pub seed: u64,
}

// ── MapGenerator Implementation ────────────────────────────────

impl Default for MapGenerator {
    fn default() -> Self {
        Self::new(42)
    }
}

impl MapGenerator {
    /// Creates a new generator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            width: 8,
            height: 8,
            num_monsters: 2,
            num_treasures: 1,
            wall_density: 0.15,
            terrain_mix: false,
            seed,
        }
    }

    /// Sets the map width (minimum 5).
    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Sets the map height (minimum 5).
    pub fn with_height(mut self, height: usize) -> Self {
        self.height = height;
        self
    }

    /// Sets the number of monsters to place.
    pub fn with_monsters(mut self, count: usize) -> Self {
        self.num_monsters = count;
        self
    }

    /// Sets the number of treasures to place.
    pub fn with_treasures(mut self, count: usize) -> Self {
        self.num_treasures = count;
        self
    }

    /// Sets wall density (clamped to 0.0–0.3).
    pub fn with_wall_density(mut self, density: f32) -> Self {
        self.wall_density = density;
        self
    }

    /// Enables terrain mixing (sand `~` and water `w` patches).
    pub fn with_terrain(mut self) -> Self {
        self.terrain_mix = true;
        self
    }

    /// Generates a single-floor map.
    ///
    /// Returns `None` if the generated map is unsolvable.
    /// The seed is incremented after each call for variety.
    pub fn generate_single_floor(&mut self) -> Option<GeneratedMap> {
        let mut rng = fastrand::Rng::with_seed(self.seed);

        let width = self.width.max(5);
        let height = self.height.max(5);
        let wall_density = self.wall_density.clamp(0.0, 0.3);

        // 1. Create all-floor grid
        let mut grid = vec![vec!['.'; width]; height];

        // 2. Place walls via random walk from center
        place_walls(&mut grid, wall_density, &mut rng);

        // 3. Place start at random edge, goal at opposite edge
        let start = pick_edge_pos(height, width, &mut rng, true);
        let goal = pick_edge_pos(height, width, &mut rng, false);

        // Ensure start and goal tiles are floor
        grid[start.0][start.1] = '.';
        grid[goal.0][goal.1] = '.';

        // 4. Ensure connectivity (carve through walls if needed)
        if !self.ensure_connectivity(&mut grid, start, goal) {
            self.seed = self.seed.wrapping_add(1);
            return None;
        }

        // 5. Find reachable positions from start
        let blocked = HashSet::new();
        let reachable = reachable_positions(&grid, start, &blocked);

        if !reachable.contains(&goal) {
            self.seed = self.seed.wrapping_add(1);
            return None;
        }

        // 6. Place monsters and treasures on reachable floor tiles
        let mut available: Vec<(usize, usize)> = reachable
            .into_iter()
            .filter(|&p| p != start && p != goal)
            .collect();

        // Sort for deterministic ordering (HashSet iteration is non-deterministic)
        available.sort_unstable();

        // Fisher-Yates shuffle for random placement
        for i in (1..available.len()).rev() {
            let j = rng.usize(0..=i);
            available.swap(i, j);
        }

        let num_monsters = self.num_monsters.min(available.len());
        let mut monsters = Vec::with_capacity(num_monsters);
        for _ in 0..num_monsters {
            if let Some(pos) = available.pop() {
                monsters.push(pos);
            }
        }

        let num_treasures = self.num_treasures.min(available.len());
        let mut treasures = Vec::with_capacity(num_treasures);
        for _ in 0..num_treasures {
            if let Some(pos) = available.pop() {
                treasures.push(pos);
            }
        }

        // 7. Verify all targets reachable from start
        if !verify_all_reachable(&grid, start, &monsters, &treasures, goal) {
            self.seed = self.seed.wrapping_add(1);
            return None;
        }

        let mut result = GeneratedMap {
            grid,
            start,
            goal,
            monsters,
            treasures,
        };

        // 8. Optionally add terrain patches
        if self.terrain_mix {
            self.add_terrain_patches(&mut result.grid);
        }

        // Increment seed for next call
        self.seed = self.seed.wrapping_add(1);

        Some(result)
    }

    /// Generates a multi-floor dungeon.
    ///
    /// Each floor is generated independently with offset seeds.
    /// Stairs connect adjacent floors at matching positions.
    /// Start is on floor 0, goal is on the top floor.
    pub fn generate_multi_floor(&mut self, num_floors: usize) -> Option<GeneratedDungeon> {
        if num_floors == 0 {
            return None;
        }

        let original_seed = self.seed;
        let mut floors = Vec::with_capacity(num_floors);

        // Generate each floor with a unique seed offset
        for floor_idx in 0..num_floors {
            self.seed = original_seed.wrapping_add((floor_idx as u64) * 7919);
            let floor = self.generate_single_floor()?;
            floors.push(floor);
        }

        // Place stairs connecting adjacent floors at matching positions
        let mut stairs = Vec::new();
        for floor_idx in 0..num_floors.saturating_sub(1) {
            let floor = &floors[floor_idx];
            let next_floor = &floors[floor_idx + 1];
            let mut stair_rng =
                fastrand::Rng::with_seed(original_seed.wrapping_add((floor_idx as u64) * 104729));

            let stair_pos = find_stair_position(floor, next_floor, &mut stair_rng);

            if let Some(pos) = stair_pos {
                stairs.push(Stairs {
                    pos,
                    from_floor: floor_idx,
                    to_floor: floor_idx + 1,
                });
            } else {
                self.seed = original_seed.wrapping_add(num_floors as u64);
                return None;
            }
        }

        self.seed = original_seed.wrapping_add(num_floors as u64);

        Some(GeneratedDungeon {
            map: DungeonMap { floors, stairs },
        })
    }

    /// Ensures `start` can reach `goal` via BFS.
    ///
    /// If not reachable, carves a direct path through walls.
    /// Returns `true` if connectivity is established.
    pub fn ensure_connectivity(
        &self,
        grid: &mut [Vec<char>],
        start: (usize, usize),
        goal: (usize, usize),
    ) -> bool {
        let blocked = HashSet::new();
        let reachable = reachable_positions(grid, start, &blocked);

        if reachable.contains(&goal) {
            return true;
        }

        // Carve a direct path from start toward goal
        let mut r = start.0;
        let mut c = start.1;
        let max_steps = grid.len() + grid[0].len() + 10;
        let mut steps = 0;

        while (r, c) != goal && steps < max_steps {
            // Move toward goal, preferring row then column
            if r < goal.0 {
                r += 1;
            } else if r > goal.0 {
                r = r.saturating_sub(1);
            } else if c < goal.1 {
                c += 1;
            } else if c > goal.1 {
                c = c.saturating_sub(1);
            }

            if grid[r][c] == '#' {
                grid[r][c] = '.';
            }
            steps += 1;
        }

        // Verify connectivity after carving
        let reachable = reachable_positions(grid, start, &blocked);
        reachable.contains(&goal)
    }

    /// Adds sand (`~`) and water (`w`) terrain patches to the grid.
    ///
    /// Places 2–4 clusters of 3–5 terrain tiles each.
    /// Only replaces floor (`.`) tiles, preserving walls.
    pub fn add_terrain_patches(&self, grid: &mut [Vec<char>]) {
        let mut rng = fastrand::Rng::with_seed(self.seed.wrapping_add(9973));
        let height = grid.len();
        let width = grid[0].len();

        if height < 4 || width < 4 {
            return;
        }

        let num_patches = rng.usize(2..=4);

        for _ in 0..num_patches {
            let terrain_char = if rng.bool() { '~' } else { 'w' };
            let mut r = rng.usize(1..height.saturating_sub(1));
            let mut c = rng.usize(1..width.saturating_sub(1));
            let patch_size = rng.usize(3..=5);

            for _ in 0..patch_size {
                if r > 0 && r < height - 1 && c > 0 && c < width - 1 && grid[r][c] == '.' {
                    grid[r][c] = terrain_char;
                }

                // Random walk within patch
                match rng.usize(0..4) {
                    0 if r > 1 => r -= 1,
                    1 if r < height - 2 => r += 1,
                    2 if c > 1 => c -= 1,
                    3 if c < width - 2 => c += 1,
                    _ => {}
                }
            }
        }
    }
}

// ── GeneratedMap Implementation ────────────────────────────────

impl GeneratedMap {
    /// Produces the map string format expected by `TacticalPruner`.
    ///
    /// Symbols: `B`=start, `M`=monster, `T`=treasure, `X`=monster+treasure,
    /// `G`=goal, `#`=wall, `.`=floor, `~`=sand, `w`=water.
    ///
    /// Tiles are space-separated, rows are newline-separated.
    pub fn to_map_string(&self) -> String {
        let mut lines = Vec::with_capacity(self.grid.len());

        for r in 0..self.grid.len() {
            let mut parts = Vec::with_capacity(self.grid[r].len());

            for c in 0..self.grid[r].len() {
                let ch = self.tile_char(r, c);
                parts.push(ch);
            }

            let line: Vec<String> = parts.iter().map(|c| format!("{c}")).collect();
            lines.push(line.join(" "));
        }

        lines.join("\n")
    }

    /// Returns the display character for a tile at `(r, c)`.
    fn tile_char(&self, r: usize, c: usize) -> char {
        let pos = (r, c);

        if pos == self.start {
            return 'B';
        }
        if pos == self.goal {
            return 'G';
        }

        let has_monster = self.monsters.contains(&pos);
        let has_treasure = self.treasures.contains(&pos);

        match (has_monster, has_treasure) {
            (true, true) => 'X',
            (true, false) => 'M',
            (false, true) => 'T',
            (false, false) => self.grid[r][c],
        }
    }
}

// ── Private Helpers ────────────────────────────────────────────

/// Places walls using random walk from center area.
///
/// Each walk starts from the grid center and places walls along a
/// random path. This produces organic wall clusters while minimizing
/// isolated sections.
fn place_walls(grid: &mut [Vec<char>], wall_density: f32, rng: &mut fastrand::Rng) {
    let height = grid.len();
    let width = grid[0].len();
    let center_r = height / 2;
    let center_c = width / 2;

    let inner_tiles = width.saturating_sub(2) * height.saturating_sub(2);
    let target_walls = (inner_tiles as f32 * wall_density) as usize;

    if target_walls == 0 {
        return;
    }

    let num_walks = (target_walls / 3).clamp(1, 20);
    let steps_per_walk = (target_walls / num_walks).max(2);

    let mut walls_placed = 0;

    for _ in 0..num_walks {
        let mut r = center_r;
        let mut c = center_c;

        for _ in 0..steps_per_walk {
            if walls_placed >= target_walls {
                return;
            }

            // Place wall on inner tiles only (preserve edges for start/goal)
            if r > 0 && r < height - 1 && c > 0 && c < width - 1 && grid[r][c] != '#' {
                grid[r][c] = '#';
                walls_placed += 1;
            }

            // Random walk direction
            match rng.usize(0..4) {
                0 if r > 1 => r -= 1,
                1 if r < height - 2 => r += 1,
                2 if c > 1 => c -= 1,
                3 if c < width - 2 => c += 1,
                _ => {}
            }
        }
    }
}

/// Picks a random position on the specified edge.
///
/// `top_or_left`: if true, picks from top or left edge; otherwise bottom or right.
fn pick_edge_pos(
    height: usize,
    width: usize,
    rng: &mut fastrand::Rng,
    top_or_left: bool,
) -> (usize, usize) {
    if top_or_left {
        if rng.bool() && width > 0 {
            (0, rng.usize(0..width))
        } else if height > 0 {
            (rng.usize(0..height), 0)
        } else {
            (0, 0)
        }
    } else if rng.bool() && width > 0 {
        (height.saturating_sub(1), rng.usize(0..width))
    } else if height > 0 {
        (rng.usize(0..height), width.saturating_sub(1))
    } else {
        (0, 0)
    }
}

/// Verifies all targets are reachable from `start` via BFS.
fn verify_all_reachable(
    grid: &[Vec<char>],
    start: (usize, usize),
    monsters: &[(usize, usize)],
    treasures: &[(usize, usize)],
    goal: (usize, usize),
) -> bool {
    let blocked = HashSet::new();
    let reachable = reachable_positions(grid, start, &blocked);

    if !reachable.contains(&goal) {
        return false;
    }

    for &pos in monsters {
        if !reachable.contains(&pos) {
            return false;
        }
    }

    for &pos in treasures {
        if !reachable.contains(&pos) {
            return false;
        }
    }

    true
}

/// Finds a valid stair position that is passable on both adjacent floors.
fn find_stair_position(
    floor: &GeneratedMap,
    next_floor: &GeneratedMap,
    rng: &mut fastrand::Rng,
) -> Option<(usize, usize)> {
    let height = floor.grid.len();
    let width = floor.grid[0].len();

    for _ in 0..100 {
        let r = rng.usize(1..height.saturating_sub(1).max(2));
        let c = rng.usize(1..width.saturating_sub(1).max(2));

        if is_passable(&floor.grid, r, c)
            && is_passable(&next_floor.grid, r, c)
            && (r, c) != floor.start
            && (r, c) != floor.goal
        {
            return Some((r, c));
        }
    }

    None
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::tactical_pruner::TacticalPruner;

    #[test]
    fn test_generate_single_floor_basic() {
        let mut generator = MapGenerator::new(42);
        let map = generator.generate_single_floor();

        assert!(map.is_some(), "Should generate a valid map");
        let map = map.unwrap();

        assert_eq!(map.grid.len(), 8);
        assert_eq!(map.grid[0].len(), 8);
        assert_ne!(map.start, map.goal);
    }

    #[test]
    fn test_generate_single_floor_solvability() {
        let mut generator = MapGenerator::new(123);
        let map = generator
            .generate_single_floor()
            .expect("Should generate map");

        let map_str = map.to_map_string();
        let pruner = TacticalPruner::new(&map_str);
        let state = pruner.initial_state();

        assert_eq!((state.r, state.c), map.start);
    }

    #[test]
    fn test_reproducibility() {
        let mut gen1 = MapGenerator::new(999);
        let mut gen2 = MapGenerator::new(999);

        let map1 = gen1.generate_single_floor().unwrap();
        let map2 = gen2.generate_single_floor().unwrap();

        assert_eq!(map1.grid, map2.grid);
        assert_eq!(map1.start, map2.start);
        assert_eq!(map1.goal, map2.goal);
        assert_eq!(map1.monsters, map2.monsters);
        assert_eq!(map1.treasures, map2.treasures);
    }

    #[test]
    fn test_different_seeds_different_maps() {
        let mut gen1 = MapGenerator::new(100);
        let mut gen2 = MapGenerator::new(200);

        let map1 = gen1.generate_single_floor().unwrap();
        let map2 = gen2.generate_single_floor().unwrap();

        // Very unlikely to be identical with different seeds
        assert_ne!(map1.grid, map2.grid);
    }

    #[test]
    fn test_generate_multi_floor() {
        let mut generator = MapGenerator::new(42).with_width(6).with_height(6);
        let dungeon = generator.generate_multi_floor(3);

        assert!(dungeon.is_some(), "Should generate a multi-floor dungeon");
        let dungeon = dungeon.unwrap();

        assert_eq!(dungeon.map.floors.len(), 3);
        assert_eq!(dungeon.map.stairs.len(), 2);
    }

    #[test]
    fn test_multi_floor_stairs_connectivity() {
        let mut generator = MapGenerator::new(77).with_width(6).with_height(6);
        let dungeon = generator.generate_multi_floor(2).unwrap();

        for stairs in &dungeon.map.stairs {
            let from_floor = &dungeon.map.floors[stairs.from_floor];
            let to_floor = &dungeon.map.floors[stairs.to_floor];

            assert!(is_passable(&from_floor.grid, stairs.pos.0, stairs.pos.1));
            assert!(is_passable(&to_floor.grid, stairs.pos.0, stairs.pos.1));
        }
    }

    #[test]
    fn test_to_map_string_format() {
        let mut generator = MapGenerator::new(42)
            .with_width(5)
            .with_height(5)
            .with_monsters(1)
            .with_treasures(1);

        let map = generator.generate_single_floor().unwrap();
        let map_str = map.to_map_string();

        assert!(map_str.contains('B'), "Should contain start marker");
        assert!(map_str.contains('G'), "Should contain goal marker");

        // Should be parseable by TacticalPruner
        let pruner = TacticalPruner::new(&map_str);
        let _ = pruner.initial_state();
    }

    #[test]
    fn test_ensure_connectivity() {
        let mut grid = vec![
            vec!['.', '.', '#', '.', '.'],
            vec!['.', '.', '#', '.', '.'],
            vec!['.', '.', '#', '.', '.'],
            vec!['.', '.', '.', '.', '.'],
        ];

        let start = (0, 0);
        let goal = (0, 4);
        let generator = MapGenerator::new(1);

        let connected = generator.ensure_connectivity(&mut grid, start, goal);
        assert!(connected, "Should be able to carve a path");

        let blocked = HashSet::new();
        let reachable = reachable_positions(&grid, start, &blocked);
        assert!(reachable.contains(&goal));
    }

    #[test]
    fn test_add_terrain_patches() {
        let mut grid = vec![vec!['.'; 10]; 10];
        let generator = MapGenerator::new(42);
        generator.add_terrain_patches(&mut grid);

        let has_terrain = grid
            .iter()
            .flat_map(|row| row.iter())
            .any(|&c| c == '~' || c == 'w');
        assert!(has_terrain, "Should have some terrain patches");
    }

    #[test]
    fn test_wall_density_respected() {
        let mut generator = MapGenerator::new(42)
            .with_width(20)
            .with_height(20)
            .with_wall_density(0.1)
            .with_monsters(0)
            .with_treasures(0);

        let map = generator.generate_single_floor().unwrap();

        let total = map.grid.len() * map.grid[0].len();
        let walls: usize = map
            .grid
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&c| c == '#')
            .count();

        let actual_density = walls as f32 / total as f32;
        assert!(
            actual_density < 0.25,
            "Wall density {actual_density} should be reasonable"
        );
    }

    #[test]
    fn test_zero_floors_returns_none() {
        let mut generator = MapGenerator::new(42);
        assert!(generator.generate_multi_floor(0).is_none());
    }

    #[test]
    fn test_builder_pattern() {
        let generator = MapGenerator::new(42)
            .with_width(10)
            .with_height(12)
            .with_monsters(3)
            .with_treasures(2)
            .with_wall_density(0.2)
            .with_terrain();

        assert_eq!(generator.width, 10);
        assert_eq!(generator.height, 12);
        assert_eq!(generator.num_monsters, 3);
        assert_eq!(generator.num_treasures, 2);
        assert!((generator.wall_density - 0.2).abs() < f32::EPSILON);
        assert!(generator.terrain_mix);
    }

    #[test]
    fn test_all_targets_reachable() {
        let mut generator = MapGenerator::new(55)
            .with_width(8)
            .with_height(8)
            .with_monsters(3)
            .with_treasures(2);

        let map = generator.generate_single_floor().unwrap();
        let blocked = HashSet::new();
        let reachable = reachable_positions(&map.grid, map.start, &blocked);

        assert!(reachable.contains(&map.goal), "Goal must be reachable");
        for &m in &map.monsters {
            assert!(reachable.contains(&m), "Monster at {m:?} must be reachable");
        }
        for &t in &map.treasures {
            assert!(
                reachable.contains(&t),
                "Treasure at {t:?} must be reachable"
            );
        }
    }

    #[test]
    fn test_default_generator() {
        let generator = MapGenerator::default();
        assert_eq!(generator.width, 8);
        assert_eq!(generator.height, 8);
        assert_eq!(generator.seed, 42);
    }

    #[test]
    fn test_multi_floor_dungeon_start_goal() {
        let mut generator = MapGenerator::new(42).with_width(6).with_height(6);
        let dungeon = generator.generate_multi_floor(3).unwrap();

        // Floor 0 should have a valid start
        let floor0 = &dungeon.map.floors[0];
        assert!(is_passable(&floor0.grid, floor0.start.0, floor0.start.1));

        // Top floor should have a valid goal
        let top = &dungeon.map.floors[2];
        assert!(is_passable(&top.grid, top.goal.0, top.goal.1));
    }
}
