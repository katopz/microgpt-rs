//! Multi-floor Dungeon Pruner
//!
//! Extends the single-floor TacticalPruner pattern to multi-floor dungeons
//! with inter-floor stair connections.
//!
//! Actions:
//! - 0 = Up
//! - 1 = Down
//! - 2 = Left
//! - 3 = Right
//! - 4 = Attack
//! - 5 = Use Stairs
//!
//! Follows the same deterministic rules as TacticalPruner:
//! - Wall collisions and per-floor grid bounds
//! - Monsters that can be killed and drop items
//! - Locked treasures requiring inventory items
//! - Goal/exit locked until all treasures collected
//! - Inventory system (max 2 items)
//! - Stair connections are bidirectional

/// Maximum inventory capacity.
const MAX_INVENTORY: u8 = 2;

/// A single floor's grid.
pub type FloorGrid = Vec<Vec<char>>;

/// Connection between two floors via stairs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StairConnection {
    pub from: (usize, usize, usize), // (floor, r, c) - stairs down position
    pub to: (usize, usize, usize),   // (floor, r, c) - stairs up destination
}

/// Multi-floor dungeon map.
#[derive(Clone, Debug)]
pub struct DungeonMap {
    pub floors: Vec<FloorGrid>,
    pub stairs: Vec<StairConnection>,
    pub start: (usize, usize, usize),          // (floor, r, c)
    pub goal: (usize, usize, usize),           // (floor, r, c)
    pub monsters: Vec<(usize, usize, usize)>,  // (floor, r, c)
    pub treasures: Vec<(usize, usize, usize)>, // (floor, r, c)
}

impl DungeonMap {
    /// Parse multiple floor map strings into a dungeon.
    ///
    /// Map symbols:
    /// - `B` or `S` = Start (player) — only one across all floors
    /// - `M` = Monster
    /// - `T` = Treasure (locked, needs item)
    /// - `G` = Goal/Exit
    /// - `X` = Monster + Treasure on same tile
    /// - `#` = Wall
    /// - `.` = Floor
    /// - `~` = Sand (cost 2)
    /// - `w` = Water (cost 3)
    pub fn new(floor_maps: &[&str], stairs: Vec<StairConnection>) -> Self {
        let mut floors = Vec::new();
        let mut start = (0usize, 0usize, 0usize);
        let mut goal = (0usize, 0usize, 0usize);
        let mut monsters = Vec::new();
        let mut treasures = Vec::new();

        for (floor_idx, map_str) in floor_maps.iter().enumerate() {
            let mut grid = Vec::new();

            for (r, line) in map_str.lines().enumerate() {
                let mut row = Vec::new();
                for (c, ch) in line.split_whitespace().enumerate() {
                    let char_val = ch.chars().next().unwrap();
                    match char_val {
                        'B' | 'S' => {
                            start = (floor_idx, r, c);
                            row.push('.');
                        }
                        'M' => {
                            monsters.push((floor_idx, r, c));
                            row.push('.');
                        }
                        'T' => {
                            treasures.push((floor_idx, r, c));
                            row.push('.');
                        }
                        'G' => {
                            goal = (floor_idx, r, c);
                            row.push('.');
                        }
                        'X' => {
                            monsters.push((floor_idx, r, c));
                            treasures.push((floor_idx, r, c));
                            row.push('.');
                        }
                        _ => row.push(char_val),
                    }
                }
                grid.push(row);
            }
            floors.push(grid);
        }

        Self {
            floors,
            stairs,
            start,
            goal,
            monsters,
            treasures,
        }
    }

    /// The starting state before any actions are taken.
    pub fn initial_state(&self) -> DungeonState {
        DungeonState {
            floor: self.start.0,
            r: self.start.1,
            c: self.start.2,
            inventory: 0,
            killed_monsters: 0,
            collected_treasures: 0,
            dropped_items: 0,
            total_cost: 0,
        }
    }
}

/// State for multi-floor dungeon.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DungeonState {
    pub floor: usize,
    pub r: usize,
    pub c: usize,
    pub inventory: u8,
    pub killed_monsters: u32,
    pub collected_treasures: u32,
    pub dropped_items: u32,
    pub total_cost: u32,
}

impl DungeonState {
    /// Returns a compact summary string of the state for display.
    pub fn summary(&self) -> String {
        let floor = self.floor;
        let r = self.r;
        let c = self.c;
        let inventory = self.inventory;
        let total_cost = self.total_cost;
        let killed_monsters = self.killed_monsters;
        let collected_treasures = self.collected_treasures;
        let dropped_items = self.dropped_items;
        format!(
            "floor={floor} pos=({r}, {c}) inv={inventory} cost={total_cost} killed={killed_monsters:b} treasures={collected_treasures:b} dropped={dropped_items:b}"
        )
    }
}

/// Multi-floor dungeon pruner implementing deterministic rules.
///
/// Enforces physical and combat rules for multi-floor dungeons with:
/// - Per-floor grid movement with wall collisions and bounds checking
/// - Monsters that can be killed and drop items
/// - Locked treasures requiring inventory items
/// - Goal/exit locked until all treasures collected
/// - Inter-floor stair connections (bidirectional)
/// - Inventory system (max 2 items)
pub struct DungeonPruner {
    pub map: DungeonMap,
}

impl DungeonPruner {
    /// Create a new DungeonPruner from a DungeonMap.
    pub fn new(map: DungeonMap) -> Self {
        Self { map }
    }

    /// Returns the terrain cost of stepping onto a tile on the given floor.
    ///
    /// Terrain costs:
    /// - `.` = Floor (cost 1)
    /// - `~` = Sand (cost 2)
    /// - `w` = Water (cost 3)
    pub fn terrain_cost(&self, floor: usize, r: usize, c: usize) -> u32 {
        match self.map.floors[floor][r][c] {
            '~' => 2,
            'w' => 3,
            _ => 1,
        }
    }

    /// Applies a single action to the state.
    ///
    /// Actions:
    /// - 0 = Up
    /// - 1 = Down
    /// - 2 = Left
    /// - 3 = Right
    /// - 4 = Attack
    /// - 5 = Use Stairs
    ///
    /// Returns `None` if the action is impossible.
    pub fn apply_action(&self, state: &DungeonState, action: usize) -> Option<DungeonState> {
        let mut next = state.clone();

        match action {
            0..=3 => self.apply_move(&mut next, action)?,
            4 => self.apply_attack(&mut next)?,
            5 => self.apply_stairs(&mut next)?,
            _ => return None,
        }

        Some(next)
    }

    /// Apply movement action (0=Up, 1=Down, 2=Left, 3=Right).
    fn apply_move(&self, next: &mut DungeonState, action: usize) -> Option<()> {
        let (dr, dc): (isize, isize) = match action {
            0 => (-1, 0),
            1 => (1, 0),
            2 => (0, -1),
            3 => (0, 1),
            _ => return None,
        };

        let nr = next.r as isize + dr;
        let nc = next.c as isize + dc;

        // Grid bounds check
        let grid = &self.map.floors[next.floor];
        if nr < 0 || nc < 0 || nr >= grid.len() as isize {
            return None;
        }
        let nr = nr as usize;
        let nc = nc as usize;
        if nc >= grid[nr].len() {
            return None;
        }

        // Wall collision
        if grid[nr][nc] == '#' {
            return None;
        }

        // Goal validation (locked until all treasures collected)
        if (next.floor, nr, nc) == self.map.goal {
            let all_treasures = (1 << self.map.treasures.len()) - 1;
            if next.collected_treasures != all_treasures {
                return None;
            }
        }

        // Check for a LIVE monster at the target tile
        let live_monster_here = self
            .map
            .monsters
            .iter()
            .enumerate()
            .any(|(i, &(f, mr, mc))| {
                (f, mr, mc) == (next.floor, nr, nc) && (next.killed_monsters & (1 << i)) == 0
            });

        // Treasure collection (locked without item)
        if !live_monster_here {
            for (i, &(f, tr, tc)) in self.map.treasures.iter().enumerate() {
                if (f, tr, tc) == (next.floor, nr, nc) && (next.collected_treasures & (1 << i)) == 0
                {
                    if next.inventory > 0 {
                        next.inventory -= 1;
                        next.collected_treasures |= 1 << i;
                    } else {
                        return None; // Cannot walk onto locked treasure without item
                    }
                }
            }
        }

        // Update coordinates
        next.r = nr;
        next.c = nc;

        // Accumulate movement cost (terrain-dependent)
        next.total_cost += self.terrain_cost(next.floor, nr, nc);

        // Auto-pickup dropped items at new tile
        for (i, &(f, mr, mc)) in self.map.monsters.iter().enumerate() {
            if (f, mr, mc) == (next.floor, nr, nc)
                && (next.dropped_items & (1 << i)) != 0
                && next.inventory < MAX_INVENTORY
            {
                next.inventory += 1;
                next.dropped_items &= !(1 << i);
            }
        }

        Some(())
    }

    /// Apply attack action — must be standing on a live monster's tile.
    fn apply_attack(&self, next: &mut DungeonState) -> Option<()> {
        let m_idx = self
            .map
            .monsters
            .iter()
            .position(|&(f, mr, mc)| (f, mr, mc) == (next.floor, next.r, next.c));

        match m_idx {
            Some(idx) if (next.killed_monsters & (1 << idx)) == 0 => {
                next.killed_monsters |= 1 << idx;
                next.dropped_items |= 1 << idx;

                // Auto-pickup if inventory allows
                if next.inventory < MAX_INVENTORY {
                    next.inventory += 1;
                    next.dropped_items &= !(1 << idx);
                }

                // Check for treasure underneath the killed monster
                for (i, &(f, tr, tc)) in self.map.treasures.iter().enumerate() {
                    if (f, tr, tc) == (next.floor, next.r, next.c)
                        && (next.collected_treasures & (1 << i)) == 0
                        && next.inventory > 0
                    {
                        next.inventory -= 1;
                        next.collected_treasures |= 1 << i;
                    }
                }

                Some(())
            }
            _ => None, // No live monster here to attack
        }
    }

    /// Apply use stairs action — must be standing on a stair connection.
    ///
    /// Stairs are bidirectional: standing on either `from` or `to` transitions
    /// to the opposite end.
    fn apply_stairs(&self, next: &mut DungeonState) -> Option<()> {
        let current_pos = (next.floor, next.r, next.c);

        // Find a stair connection matching current position
        let stair = self
            .map
            .stairs
            .iter()
            .find(|s| s.from == current_pos || s.to == current_pos)?;

        // Determine destination (opposite end of the connection)
        let destination = if stair.from == current_pos {
            stair.to
        } else {
            stair.from
        };

        // Validate destination is within bounds and passable
        let (dest_floor, dest_r, dest_c) = destination;
        if dest_floor >= self.map.floors.len() {
            return None;
        }

        let grid = &self.map.floors[dest_floor];
        if dest_r >= grid.len() || dest_c >= grid[dest_r].len() {
            return None;
        }

        if grid[dest_r][dest_c] == '#' {
            return None;
        }

        // Transition to new floor
        next.floor = dest_floor;
        next.r = dest_r;
        next.c = dest_c;
        next.total_cost += 1; // Base stair traversal cost

        Some(())
    }

    /// Replays a sequence of actions from the starting position.
    pub fn replay_state(&self, actions: &[usize]) -> Option<DungeonState> {
        let mut state = self.map.initial_state();
        for &action in actions {
            state = self.apply_action(&state, action)?;
        }
        Some(state)
    }

    /// Returns the starting state.
    pub fn initial_state(&self) -> DungeonState {
        self.map.initial_state()
    }

    /// Returns the action name for display.
    pub fn action_name(action: usize) -> &'static str {
        match action {
            0 => "Up",
            1 => "Down",
            2 => "Left",
            3 => "Right",
            4 => "Attack",
            5 => "Use Stairs",
            _ => "?",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_floor_maps() -> Vec<&'static str> {
        vec![
            // Floor 0: start at top-left, monster at (1,1)
            ". . .\n\
             . M .\n\
             . . .",
            // Floor 1: treasure at (0,2), goal at (2,2)
            ". . T\n\
             . . .\n\
             . . G",
        ]
    }

    fn two_floor_stairs() -> Vec<StairConnection> {
        vec![StairConnection {
            from: (0, 2, 2), // Floor 0, bottom-right
            to: (1, 0, 0),   // Floor 1, top-left
        }]
    }

    #[test]
    fn test_dungeon_map_parses_floors() {
        let map = DungeonMap::new(&two_floor_maps(), two_floor_stairs());
        assert_eq!(map.floors.len(), 2);
        assert_eq!(map.start, (0, 0, 0));
        assert_eq!(map.goal, (1, 2, 2));
        assert_eq!(map.monsters, vec![(0, 1, 1)]);
        assert_eq!(map.treasures, vec![(1, 0, 2)]);
    }

    #[test]
    fn test_initial_state() {
        let map = DungeonMap::new(&two_floor_maps(), two_floor_stairs());
        let pruner = DungeonPruner::new(map);
        let state = pruner.initial_state();
        assert_eq!(state.floor, 0);
        assert_eq!(state.r, 0);
        assert_eq!(state.c, 0);
        assert_eq!(state.inventory, 0);
        assert_eq!(state.total_cost, 0);
    }

    #[test]
    fn test_move_within_floor() {
        let map = DungeonMap::new(&two_floor_maps(), two_floor_stairs());
        let pruner = DungeonPruner::new(map);
        let state = pruner.initial_state();

        // Move right
        let next = pruner.apply_action(&state, 3).unwrap();
        assert_eq!((next.floor, next.r, next.c), (0, 0, 1));
        assert_eq!(next.total_cost, 1);
    }

    #[test]
    fn test_move_into_wall_blocked() {
        let maps = vec![
            ". # .\n\
             . . .\n\
             . . .",
        ];
        let map = DungeonMap::new(&maps, vec![]);
        let pruner = DungeonPruner::new(map);
        let state = pruner.initial_state();

        // Move right into wall
        assert!(pruner.apply_action(&state, 3).is_none());
    }

    #[test]
    fn test_move_out_of_bounds_blocked() {
        let maps = vec![". . .\n. . ."];
        let map = DungeonMap::new(&maps, vec![]);
        let pruner = DungeonPruner::new(map);
        let state = pruner.initial_state();

        // Move up from top row
        assert!(pruner.apply_action(&state, 0).is_none());
    }

    #[test]
    fn test_attack_monster() {
        let map = DungeonMap::new(&two_floor_maps(), two_floor_stairs());
        let pruner = DungeonPruner::new(map);
        let state = pruner.initial_state();

        // Move to monster position: right, down
        let s1 = pruner.apply_action(&state, 3).unwrap(); // (0,0,1)
        let s2 = pruner.apply_action(&s1, 1).unwrap(); // (0,1,1) — monster tile

        // Attack
        let s3 = pruner.apply_action(&s2, 4).unwrap();
        assert_eq!(s3.killed_monsters, 1); // first monster killed
        assert_eq!(s3.inventory, 1); // auto-pickup
    }

    #[test]
    fn test_attack_no_monster_fails() {
        let map = DungeonMap::new(&two_floor_maps(), two_floor_stairs());
        let pruner = DungeonPruner::new(map);
        let state = pruner.initial_state();

        // Attack at start position (no monster)
        assert!(pruner.apply_action(&state, 4).is_none());
    }

    #[test]
    fn test_use_stairs() {
        let map = DungeonMap::new(&two_floor_maps(), two_floor_stairs());
        let pruner = DungeonPruner::new(map);
        let state = pruner.initial_state();

        // Navigate to stairs at (0, 2, 2)
        let s1 = pruner.apply_action(&state, 3).unwrap(); // (0,0,1)
        let s2 = pruner.apply_action(&s1, 3).unwrap(); // (0,0,2)
        let s3 = pruner.apply_action(&s2, 1).unwrap(); // (0,1,2)
        let s4 = pruner.apply_action(&s3, 1).unwrap(); // (0,2,2) — stairs!

        // Use stairs
        let s5 = pruner.apply_action(&s4, 5).unwrap();
        assert_eq!((s5.floor, s5.r, s5.c), (1, 0, 0));
    }

    #[test]
    fn test_stairs_not_at_position_fails() {
        let map = DungeonMap::new(&two_floor_maps(), two_floor_stairs());
        let pruner = DungeonPruner::new(map);
        let state = pruner.initial_state();

        // Try stairs at start (no stairs here)
        assert!(pruner.apply_action(&state, 5).is_none());
    }

    #[test]
    fn test_stairs_bidirectional() {
        let map = DungeonMap::new(&two_floor_maps(), two_floor_stairs());
        let pruner = DungeonPruner::new(map);

        // Start on floor 1 at stairs destination (1, 0, 0)
        let state = DungeonState {
            floor: 1,
            r: 0,
            c: 0,
            inventory: 0,
            killed_monsters: 0,
            collected_treasures: 0,
            dropped_items: 0,
            total_cost: 0,
        };

        // Use stairs back to floor 0
        let next = pruner.apply_action(&state, 5).unwrap();
        assert_eq!((next.floor, next.r, next.c), (0, 2, 2));
    }

    #[test]
    fn test_goal_locked_without_treasures() {
        let map = DungeonMap::new(&two_floor_maps(), two_floor_stairs());
        let pruner = DungeonPruner::new(map);

        // Place player adjacent to goal on floor 1
        let state = DungeonState {
            floor: 1,
            r: 2,
            c: 1,
            inventory: 0,
            killed_monsters: 0,
            collected_treasures: 0,
            dropped_items: 0,
            total_cost: 0,
        };

        // Try to step onto goal — locked
        assert!(pruner.apply_action(&state, 3).is_none());
    }

    #[test]
    fn test_terrain_cost_sand() {
        let maps = vec![". ~ .\n. . ."];
        let map = DungeonMap::new(&maps, vec![]);
        let pruner = DungeonPruner::new(map);
        let state = pruner.initial_state();

        // Step onto sand
        let next = pruner.apply_action(&state, 3).unwrap();
        assert_eq!(next.total_cost, 2);
    }

    #[test]
    fn test_terrain_cost_water() {
        let maps = vec![". w .\n. . ."];
        let map = DungeonMap::new(&maps, vec![]);
        let pruner = DungeonPruner::new(map);
        let state = pruner.initial_state();

        // Step onto water
        let next = pruner.apply_action(&state, 3).unwrap();
        assert_eq!(next.total_cost, 3);
    }

    #[test]
    fn test_replay_state() {
        let map = DungeonMap::new(&two_floor_maps(), two_floor_stairs());
        let pruner = DungeonPruner::new(map);

        let actions = vec![3, 1, 4]; // Right, Down, Attack (kill monster at (0,1,1))
        let result = pruner.replay_state(&actions).unwrap();
        assert_eq!((result.floor, result.r, result.c), (0, 1, 1));
        assert_eq!(result.killed_monsters, 1);
    }

    #[test]
    fn test_replay_invalid_action_returns_none() {
        let map = DungeonMap::new(&two_floor_maps(), two_floor_stairs());
        let pruner = DungeonPruner::new(map);

        // Attack at start (no monster) — invalid
        let actions = vec![4];
        assert!(pruner.replay_state(&actions).is_none());
    }

    #[test]
    fn test_action_names() {
        assert_eq!(DungeonPruner::action_name(0), "Up");
        assert_eq!(DungeonPruner::action_name(1), "Down");
        assert_eq!(DungeonPruner::action_name(2), "Left");
        assert_eq!(DungeonPruner::action_name(3), "Right");
        assert_eq!(DungeonPruner::action_name(4), "Attack");
        assert_eq!(DungeonPruner::action_name(5), "Use Stairs");
        assert_eq!(DungeonPruner::action_name(6), "?");
    }

    #[test]
    fn test_invalid_action_returns_none() {
        let map = DungeonMap::new(&two_floor_maps(), two_floor_stairs());
        let pruner = DungeonPruner::new(map);
        let state = pruner.initial_state();

        assert!(pruner.apply_action(&state, 6).is_none());
        assert!(pruner.apply_action(&state, 99).is_none());
    }

    #[test]
    fn test_state_summary() {
        let state = DungeonState {
            floor: 1,
            r: 2,
            c: 3,
            inventory: 1,
            killed_monsters: 0b01,
            collected_treasures: 0b10,
            dropped_items: 0b00,
            total_cost: 5,
        };
        let summary = state.summary();
        assert!(summary.contains("floor=1"));
        assert!(summary.contains("pos=(2, 3)"));
        assert!(summary.contains("inv=1"));
        assert!(summary.contains("cost=5"));
    }

    #[test]
    fn test_monster_and_treasure_same_tile() {
        let maps = vec![
            ". . .\n\
             . X .\n\
             . . .",
        ];
        let map = DungeonMap::new(&maps, vec![]);
        assert_eq!(map.monsters, vec![(0, 1, 1)]);
        assert_eq!(map.treasures, vec![(0, 1, 1)]);
    }

    #[test]
    fn test_treasure_locked_without_item() {
        let maps = vec![
            ". . .\n\
             . M T\n\
             . . .",
        ];
        let map = DungeonMap::new(&maps, vec![]);
        let pruner = DungeonPruner::new(map);
        let state = pruner.initial_state();

        // Move to monster and kill it
        let s1 = pruner.apply_action(&state, 3).unwrap(); // (0,0,1)
        let s2 = pruner.apply_action(&s1, 1).unwrap(); // (0,1,1) — monster
        let s3 = pruner.apply_action(&s2, 4).unwrap(); // attack
        assert_eq!(s3.inventory, 1);

        // Try to move onto treasure with item
        let s4 = pruner.apply_action(&s3, 3).unwrap(); // (0,1,2) — treasure
        assert_eq!(s4.collected_treasures, 1);
        assert_eq!(s4.inventory, 0); // Used item
    }

    #[test]
    fn test_treasure_blocked_without_item() {
        let maps = vec![
            ". T .\n\
             . . .\n\
             . . .",
        ];
        let map = DungeonMap::new(&maps, vec![]);
        let pruner = DungeonPruner::new(map);
        let state = pruner.initial_state();

        // Try to move onto treasure without item — blocked
        assert!(pruner.apply_action(&state, 3).is_none());
    }

    #[test]
    fn test_full_dungeon_run() {
        // Two-floor dungeon: monster on floor 0, treasure on floor 1
        let map = DungeonMap::new(&two_floor_maps(), two_floor_stairs());
        let pruner = DungeonPruner::new(map);
        let mut state = pruner.initial_state();

        // Floor 0: navigate to monster at (0,1,1)
        state = pruner.apply_action(&state, 3).unwrap(); // (0,0,1)
        state = pruner.apply_action(&state, 1).unwrap(); // (0,1,1)

        // Attack monster
        state = pruner.apply_action(&state, 4).unwrap();
        assert_eq!(state.inventory, 1);

        // Navigate to stairs at (0,2,2)
        state = pruner.apply_action(&state, 3).unwrap(); // (0,1,2)
        state = pruner.apply_action(&state, 1).unwrap(); // (0,2,2)

        // Use stairs to floor 1
        state = pruner.apply_action(&state, 5).unwrap();
        assert_eq!(state.floor, 1);
        assert_eq!((state.r, state.c), (0, 0));

        // Navigate to treasure at (1,0,2)
        state = pruner.apply_action(&state, 3).unwrap(); // (1,0,1)
        state = pruner.apply_action(&state, 3).unwrap(); // (1,0,2) — treasure!
        assert_eq!(state.collected_treasures, 1);
        assert_eq!(state.inventory, 0);

        // Navigate to goal at (1,2,2) — all treasures collected
        state = pruner.apply_action(&state, 1).unwrap(); // (1,1,2)
        state = pruner.apply_action(&state, 1).unwrap(); // (1,2,2) — goal!
        assert_eq!((state.floor, state.r, state.c), (1, 2, 2));
    }

    #[test]
    fn test_auto_pickup_dropped_item() {
        let maps = vec![
            ". . .\n\
             M M .\n\
             . . .",
        ];
        let map = DungeonMap::new(&maps, vec![]);
        let pruner = DungeonPruner::new(map);

        // Start at (0,0), inventory max = 2
        let state = DungeonState {
            floor: 0,
            r: 1,
            c: 0,
            inventory: 2,          // Full inventory
            killed_monsters: 0b11, // Both killed
            collected_treasures: 0,
            dropped_items: 0b11, // Both dropped items on floor
            total_cost: 0,
        };

        // Can't pick up — inventory full
        // Move away then back
        let s1 = pruner.apply_action(&state, 0).unwrap(); // (0,0,0)
        assert_eq!(s1.inventory, 2); // Still full

        // Use an item (simulate by lowering inventory)
        let mut s2 = s1.clone();
        s2.inventory = 1;

        // Move back to monster with dropped item
        let s3 = pruner.apply_action(&s2, 1).unwrap(); // (0,1,0) — dropped item
        assert_eq!(s3.inventory, 2); // Picked up
        assert_eq!(s3.dropped_items, 0b10); // Only first item picked up
    }

    #[test]
    fn test_three_floor_dungeon() {
        let maps = vec![
            // Floor 0: start
            ". . .\n\
             . M .\n\
             . . .",
            // Floor 1: mid
            ". . .\n\
             . T .\n\
             . . .",
            // Floor 2: goal
            ". . .\n\
             . . .\n\
             . . G",
        ];
        let stairs = vec![
            StairConnection {
                from: (0, 2, 2),
                to: (1, 0, 0),
            },
            StairConnection {
                from: (1, 2, 2),
                to: (2, 0, 0),
            },
        ];
        let map = DungeonMap::new(&maps, stairs);
        let pruner = DungeonPruner::new(map);

        assert_eq!(pruner.map.floors.len(), 3);
        assert_eq!(pruner.map.start, (0, 0, 0));
        assert_eq!(pruner.map.goal, (2, 2, 2));
        assert_eq!(pruner.map.monsters, vec![(0, 1, 1)]);
        assert_eq!(pruner.map.treasures, vec![(1, 1, 1)]);

        let state = pruner.initial_state();
        assert_eq!(state.floor, 0);
    }
}
