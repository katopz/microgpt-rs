//! Grid-based Tactical Puzzle — Constraint Pruner
//!
//! A generic deterministic rules engine for grid-based puzzle games with:
//! - Wall collisions and grid bounds
//! - Monsters that can be killed and drop items
//! - Locked treasures requiring inventory items
//! - Goal/exit locked until all treasures collected
//! - Inventory system (configurable max)
//!
//! Used with DDTree as a pure A* / Best-First search engine.
//! The pruner eliminates invalid state transitions, keeping the branching
//! factor small enough for exhaustive search within the DDTree budget.

use crate::speculative::types::ConstraintPruner;

/// Represents the deterministic state of a grid-based tactical puzzle.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GameState {
    pub r: usize,
    pub c: usize,
    pub inventory: u8,            // Limited inventory (max 2 items)
    pub killed_monsters: u32,     // Bitmask of killed monsters
    pub collected_treasures: u32, // Bitmask of collected treasures
    pub dropped_items: u32,       // Bitmask of items currently on the floor
}

impl GameState {
    /// Returns a compact summary string of the state for display.
    pub fn summary(&self) -> String {
        format!(
            "pos=({}, {}) inv={} killed={:02b} treasures={:02b} dropped={:02b}",
            self.r,
            self.c,
            self.inventory,
            self.killed_monsters,
            self.collected_treasures,
            self.dropped_items
        )
    }
}

/// Generic grid-based tactical puzzle constraint pruner.
///
/// Enforces physical and combat rules for games with grid movement,
/// monsters, locked treasures, inventory items, and goal conditions.
pub struct TacticalPruner {
    pub grid: Vec<Vec<char>>,
    pub start_r: usize,
    pub start_c: usize,
    pub monsters: Vec<(usize, usize)>,
    pub treasures: Vec<(usize, usize)>,
    pub goal: (usize, usize),
}

impl TacticalPruner {
    /// Parse a grid string into the internal representation.
    ///
    /// Map symbols:
    /// - `B` or `S` = Start (player)
    /// - `M` = Monster
    /// - `T` = Treasure (locked, needs item)
    /// - `G` = Goal/Exit
    /// - `X` = Monster + Treasure on same tile
    /// - `#` = Wall
    /// - `.` = Floor
    pub fn new(map_str: &str) -> Self {
        let mut grid = Vec::new();
        let mut start_r = 0;
        let mut start_c = 0;
        let mut monsters = Vec::new();
        let mut treasures = Vec::new();
        let mut goal = (0, 0);

        for (r, line) in map_str.lines().enumerate() {
            let mut row = Vec::new();
            for (c, ch) in line.split_whitespace().enumerate() {
                let char_val = ch.chars().next().unwrap();
                match char_val {
                    'B' | 'S' => {
                        start_r = r;
                        start_c = c;
                        row.push('.');
                    }
                    'M' => {
                        monsters.push((r, c));
                        row.push('.');
                    }
                    'T' => {
                        treasures.push((r, c));
                        row.push('.');
                    }
                    'G' => {
                        goal = (r, c);
                        row.push('.');
                    }
                    'X' => {
                        // Monster and Treasure on same tile
                        monsters.push((r, c));
                        treasures.push((r, c));
                        row.push('.');
                    }
                    _ => row.push(char_val),
                }
            }
            grid.push(row);
        }

        Self {
            grid,
            start_r,
            start_c,
            monsters,
            treasures,
            goal,
        }
    }

    /// The starting state before any actions are taken.
    pub fn initial_state(&self) -> GameState {
        GameState {
            r: self.start_r,
            c: self.start_c,
            inventory: 0,
            killed_monsters: 0,
            collected_treasures: 0,
            dropped_items: 0,
        }
    }

    /// Applies a single action (0:Up, 1:Down, 2:Left, 3:Right, 4:Attack) to the state.
    /// Returns `None` if the move is physically or logically impossible.
    pub fn apply_action(&self, state: &GameState, action: usize) -> Option<GameState> {
        let mut next = state.clone();

        match action {
            0..=3 => {
                // MOVE ACTION
                let dr = if action == 0 {
                    -1
                } else if action == 1 {
                    1
                } else {
                    0
                };
                let dc = if action == 2 {
                    -1
                } else if action == 3 {
                    1
                } else {
                    0
                };

                let nr = next.r as isize + dr;
                let nc = next.c as isize + dc;

                // 1. Grid bounds
                if nr < 0
                    || nc < 0
                    || nr >= self.grid.len() as isize
                    || nc >= self.grid[0].len() as isize
                {
                    return None;
                }

                let nr = nr as usize;
                let nc = nc as usize;

                // 2. Wall collisions
                if self.grid[nr][nc] == '#' {
                    return None;
                }

                // 3. Goal validation (exit locked until all treasures collected)
                if (nr, nc) == self.goal {
                    let all_treasures = (1 << self.treasures.len()) - 1;
                    if next.collected_treasures != all_treasures {
                        return None;
                    }
                }

                // Check for a LIVE monster at the target tile
                let mut live_monster_here = false;
                for (i, &m_pos) in self.monsters.iter().enumerate() {
                    if m_pos == (nr, nc) && (next.killed_monsters & (1 << i)) == 0 {
                        live_monster_here = true;
                    }
                }

                // 4. Treasure collection (locked without item)
                if !live_monster_here {
                    for (i, &t_pos) in self.treasures.iter().enumerate() {
                        if t_pos == (nr, nc) && (next.collected_treasures & (1 << i)) == 0 {
                            if next.inventory > 0 {
                                next.inventory -= 1;
                                next.collected_treasures |= 1 << i;
                            } else {
                                // Cannot walk onto locked treasure without item
                                return None;
                            }
                        }
                    }
                }

                // Update coordinates
                next.r = nr;
                next.c = nc;

                // 5. Auto-pickup dropped items at new tile
                for (i, &m_pos) in self.monsters.iter().enumerate() {
                    if m_pos == (nr, nc)
                        && (next.dropped_items & (1 << i)) != 0
                        && next.inventory < 2
                    {
                        next.inventory += 1;
                        next.dropped_items &= !(1 << i);
                    }
                }
            }
            4 => {
                // ATTACK ACTION — must be on a live monster's tile
                let m_idx = self.monsters.iter().position(|&p| p == (next.r, next.c));

                if let Some(idx) = m_idx
                    && (next.killed_monsters & (1 << idx)) == 0
                {
                    next.killed_monsters |= 1 << idx;
                    next.dropped_items |= 1 << idx;

                    // Auto-pickup if inventory allows
                    if next.inventory < 2 {
                        next.inventory += 1;
                        next.dropped_items &= !(1 << idx);
                    }

                    // Check for treasure underneath the killed monster
                    for (i, &t_pos) in self.treasures.iter().enumerate() {
                        if t_pos == (next.r, next.c)
                            && (next.collected_treasures & (1 << i)) == 0
                            && next.inventory > 0
                        {
                            next.inventory -= 1;
                            next.collected_treasures |= 1 << i;
                        }
                    }
                } else {
                    return None; // No live monster here to attack
                }
            }
            _ => return None,
        }

        Some(next)
    }

    /// Replays a sequence of actions from the starting position.
    pub fn replay_state(&self, actions: &[usize]) -> Option<GameState> {
        let mut state = self.initial_state();
        for &action in actions {
            state = self.apply_action(&state, action)?;
        }
        Some(state)
    }

    /// Returns the action name for display.
    pub fn action_name(action: usize) -> &'static str {
        match action {
            0 => "Up",
            1 => "Down",
            2 => "Left",
            3 => "Right",
            4 => "Attack",
            _ => "?",
        }
    }
}

impl ConstraintPruner for TacticalPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        match self.replay_state(parent_tokens) {
            Some(state) => self.apply_action(&state, token_idx).is_some(),
            None => false,
        }
    }
}
