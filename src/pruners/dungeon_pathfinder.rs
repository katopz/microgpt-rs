//! Cross-Floor A* Pathfinder for Multi-Layer Dungeons
//!
//! Provides hierarchical pathfinding that operates across multiple dungeon floors
//! connected by stairways. Uses a two-level approach:
//! 1. Floor graph BFS to find the shortest sequence of floor transitions
//! 2. Floor-local A* (delegated to `pathfinder::find_path`) for each floor segment
//!
//! Used by the dungeon pruner as the tactical pathfinding layer for
//! multi-floor dungeon exploration.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::pruners::dungeon_pruner::DungeonMap;
use crate::pruners::pathfinder::{find_distance, find_path, is_passable};

// ── Type Aliases ──────────────────────────────────────────────

/// Stair traversal result: (stair_index, entrance_pos, exit_pos).
type StairTraversal = (usize, (usize, usize), (usize, usize));

/// Best stair candidate: (distance, stair_index, entrance_pos, exit_pos).
type StairCandidate = (u32, usize, (usize, usize), (usize, usize));

// ── Dungeon Action ────────────────────────────────────────────

/// An action in the multi-floor dungeon context.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DungeonAction {
    /// A* path step (action 0-3: Up/Down/Left/Right).
    Move(usize),
    /// Attack monster on current tile.
    Attack,
    /// Use stairs, index into `DungeonMap::stairs`.
    UseStairs(usize),
}

impl DungeonAction {
    /// Returns the action name for display.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Move(a) => match a {
                0 => "↑ Up",
                1 => "↓ Down",
                2 => "← Left",
                3 => "→ Right",
                _ => "???",
            },
            Self::Attack => "⚔ Attack",
            Self::UseStairs(_) => "🪜 Stairs",
        }
    }

    /// Returns a detailed description including stair index if applicable.
    pub fn detail(&self) -> String {
        match self {
            Self::UseStairs(idx) => format!("🪜 Stairs #{idx}"),
            _ => self.name().to_string(),
        }
    }
}

// ── Types ─────────────────────────────────────────────────────

/// Blocked positions per floor.
pub type MultiFloorBlocked = HashMap<usize, HashSet<(usize, usize)>>;

// ── Multi-Floor Target ────────────────────────────────────────

/// A strategic target in the multi-floor context.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MultiFloorTarget {
    /// Kill the monster at index `i`.
    Monster(usize),
    /// Collect the treasure at index `j`.
    Treasure(usize),
    /// Reach the goal/exit.
    Goal,
}

impl MultiFloorTarget {
    /// Returns the 3D position `(floor, r, c)` of this target.
    pub fn pos(
        &self,
        monsters: &[(usize, usize, usize)],
        treasures: &[(usize, usize, usize)],
        goal: (usize, usize, usize),
    ) -> (usize, usize, usize) {
        match self {
            Self::Monster(i) => monsters[*i],
            Self::Treasure(j) => treasures[*j],
            Self::Goal => goal,
        }
    }
}

/// Enumerates all strategic targets from the dungeon map data.
///
/// Order: monsters first, then treasures, then goal.
pub fn enumerate_multifloor_targets(
    num_monsters: usize,
    num_treasures: usize,
) -> Vec<MultiFloorTarget> {
    let mut targets = Vec::with_capacity(num_monsters + num_treasures + 1);
    for i in 0..num_monsters {
        targets.push(MultiFloorTarget::Monster(i));
    }
    for j in 0..num_treasures {
        targets.push(MultiFloorTarget::Treasure(j));
    }
    targets.push(MultiFloorTarget::Goal);
    targets
}

// ── Floor-Local Pathfinding ───────────────────────────────────

/// Find path on a single floor (delegates to existing `find_path`).
///
/// Returns action indices (0=Up, 1=Down, 2=Left, 3=Right), or `None` if no
/// path exists. Returns `None` if `floor` is out of bounds.
pub fn find_path_on_floor(
    dungeon: &DungeonMap,
    floor: usize,
    from: (usize, usize),
    to: (usize, usize),
    blocked: &HashSet<(usize, usize)>,
) -> Option<Vec<usize>> {
    let grid = dungeon.floors.get(floor)?;
    find_path(grid, from, to, blocked)
}

// ── Cross-Floor Pathfinding ───────────────────────────────────

/// Find a path from one position to another, potentially crossing floors.
///
/// Uses a hierarchical approach:
/// 1. If same floor → delegate to `find_path_on_floor`
/// 2. If different floors → BFS on floor graph to find shortest stair
///    sequence, then A* for each floor segment
///
/// Returns a sequence of `DungeonAction`s, or `None` if no path exists.
pub fn find_path_multifloor(
    dungeon: &DungeonMap,
    from: (usize, usize, usize),
    to: (usize, usize, usize),
    blocked: &MultiFloorBlocked,
) -> Option<Vec<DungeonAction>> {
    let (from_floor, from_pos) = (from.0, (from.1, from.2));
    let (to_floor, to_pos) = (to.0, (to.1, to.2));

    // Same floor — simple delegation
    if from_floor == to_floor {
        let floor_blocked = blocked.get(&from_floor).cloned().unwrap_or_default();
        let path = find_path_on_floor(dungeon, from_floor, from_pos, to_pos, &floor_blocked)?;
        return Some(path.into_iter().map(DungeonAction::Move).collect());
    }

    // Different floors — find shortest floor sequence via BFS
    let floor_sequence = bfs_floor_sequence(dungeon, from_floor, to_floor)?;

    // Chain paths through stairs for each floor transition
    let mut actions = Vec::new();
    let mut current_pos = from_pos;
    let mut current_floor = from_floor;

    for &next_floor in &floor_sequence {
        let (stair_idx, stair_from, stair_to) =
            find_best_stair(dungeon, current_floor, next_floor, current_pos, blocked)?;

        // Path from current position to stairs entrance on current floor
        let floor_blocked = blocked.get(&current_floor).cloned().unwrap_or_default();
        let path_to_stairs = find_path_on_floor(
            dungeon,
            current_floor,
            current_pos,
            stair_from,
            &floor_blocked,
        )?;

        // Add movement actions to reach stairs
        for action in path_to_stairs {
            actions.push(DungeonAction::Move(action));
        }

        // Add stair transition action
        actions.push(DungeonAction::UseStairs(stair_idx));

        // Update position to stair destination on next floor
        current_pos = stair_to;
        current_floor = next_floor;
    }

    // Final segment: from last stair destination to target on destination floor
    let floor_blocked = blocked.get(&current_floor).cloned().unwrap_or_default();
    let final_path =
        find_path_on_floor(dungeon, current_floor, current_pos, to_pos, &floor_blocked)?;

    for action in final_path {
        actions.push(DungeonAction::Move(action));
    }

    Some(actions)
}

// ── Floor Graph BFS ───────────────────────────────────────────

/// BFS on the floor graph to find the shortest sequence of floor transitions.
///
/// Returns the sequence of floors to visit (excluding `from_floor`, including
/// `to_floor`), or `None` if no floor path exists.
fn bfs_floor_sequence(
    dungeon: &DungeonMap,
    from_floor: usize,
    to_floor: usize,
) -> Option<Vec<usize>> {
    if from_floor == to_floor {
        return Some(Vec::new());
    }

    // Build adjacency: which floors are reachable from each floor via stairs
    let mut adjacent: HashMap<usize, Vec<usize>> = HashMap::new();
    for stair in &dungeon.stairs {
        let a = stair.from.0;
        let b = stair.to.0;
        adjacent.entry(a).or_default().push(b);
        adjacent.entry(b).or_default().push(a);
    }

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    queue.push_back((from_floor, Vec::new()));
    visited.insert(from_floor);

    while let Some((floor, path)) = queue.pop_front() {
        if floor == to_floor {
            return Some(path);
        }

        let neighbors = adjacent.get(&floor).cloned().unwrap_or_default();
        for next_floor in neighbors {
            if visited.contains(&next_floor) {
                continue;
            }
            visited.insert(next_floor);
            let mut new_path = path.clone();
            new_path.push(next_floor);
            queue.push_back((next_floor, new_path));
        }
    }

    None
}

// ── Best Stair Selection ──────────────────────────────────────

/// Find the best stair connection between two adjacent floors.
///
/// "Best" means the stair entrance closest (by A* distance) to `current_pos`
/// on `from_floor`, considering blocked positions.
///
/// Returns `(stair_index, entrance_position, exit_position)`, or `None` if no
/// usable stair exists.
fn find_best_stair(
    dungeon: &DungeonMap,
    from_floor: usize,
    to_floor: usize,
    current_pos: (usize, usize),
    blocked: &MultiFloorBlocked,
) -> Option<StairTraversal> {
    let floor_blocked = blocked.get(&from_floor).cloned().unwrap_or_default();
    let grid = dungeon.floors.get(from_floor)?;

    let mut best: Option<StairCandidate> = None;

    for (idx, stair) in dungeon.stairs.iter().enumerate() {
        // Check both directions: forward and reverse
        let (entrance_3d, exit_3d) = match (
            stair.from.0 == from_floor && stair.to.0 == to_floor,
            stair.to.0 == from_floor && stair.from.0 == to_floor,
        ) {
            (true, _) => (stair.from, stair.to),
            (false, true) => (stair.to, stair.from),
            _ => continue,
        };

        let entrance_pos = (entrance_3d.1, entrance_3d.2);
        let exit_pos = (exit_3d.1, exit_3d.2);

        // Skip impassable or blocked stair entrances
        if !is_passable(grid, entrance_pos.0, entrance_pos.1) {
            continue;
        }
        if floor_blocked.contains(&entrance_pos) {
            continue;
        }

        // Compute distance to stair entrance; skip if unreachable
        let dist = match find_distance(grid, current_pos, entrance_pos, &floor_blocked) {
            Some(d) => d,
            None => continue,
        };

        // Keep the closest stair entrance
        match best {
            None => best = Some((dist, idx, entrance_pos, exit_pos)),
            Some((best_d, _, _, _)) if dist < best_d => {
                best = Some((dist, idx, entrance_pos, exit_pos));
            }
            _ => {}
        }
    }

    best.map(|(_, idx, entrance, exit)| (idx, entrance, exit))
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::dungeon_pruner::StairConnection;

    fn parse_grid(s: &str) -> Vec<Vec<char>> {
        s.lines()
            .map(|line| {
                line.split_whitespace()
                    .map(|c| c.chars().next().unwrap())
                    .collect()
            })
            .collect()
    }

    fn make_two_floor_dungeon() -> DungeonMap {
        let floor0 = parse_grid(". . .\n. . .\n. . .");
        let floor1 = parse_grid(". . .\n. . .\n. . .");

        DungeonMap {
            floors: vec![floor0, floor1],
            stairs: vec![StairConnection {
                from: (0, 2, 2),
                to: (1, 0, 0),
            }],
            start: (0, 0, 0),
            goal: (1, 2, 2),
            monsters: vec![],
            treasures: vec![],
        }
    }

    fn make_three_floor_dungeon() -> DungeonMap {
        let floor0 = parse_grid(". . .\n. . .\n. . .");
        let floor1 = parse_grid(". . .\n. . .\n. . .");
        let floor2 = parse_grid(". . .\n. . .\n. . .");

        DungeonMap {
            floors: vec![floor0, floor1, floor2],
            stairs: vec![
                StairConnection {
                    from: (0, 2, 2),
                    to: (1, 0, 0),
                },
                StairConnection {
                    from: (1, 2, 2),
                    to: (2, 0, 0),
                },
            ],
            start: (0, 0, 0),
            goal: (2, 2, 2),
            monsters: vec![(1, 1, 1)],
            treasures: vec![(1, 0, 2)],
        }
    }

    #[test]
    fn test_enumerate_multifloor_targets() {
        let targets = enumerate_multifloor_targets(2, 3);
        assert_eq!(targets.len(), 6);
        assert_eq!(targets[0], MultiFloorTarget::Monster(0));
        assert_eq!(targets[1], MultiFloorTarget::Monster(1));
        assert_eq!(targets[2], MultiFloorTarget::Treasure(0));
        assert_eq!(targets[3], MultiFloorTarget::Treasure(1));
        assert_eq!(targets[4], MultiFloorTarget::Treasure(2));
        assert_eq!(targets[5], MultiFloorTarget::Goal);
    }

    #[test]
    fn test_same_floor_path() {
        let dungeon = make_two_floor_dungeon();
        let blocked = MultiFloorBlocked::new();

        let path = find_path_multifloor(&dungeon, (0, 0, 0), (0, 0, 2), &blocked);
        assert!(path.is_some());
        let actions = path.unwrap();
        assert!(actions.iter().all(|a| matches!(a, DungeonAction::Move(_))));
    }

    #[test]
    fn test_cross_floor_path() {
        let dungeon = make_two_floor_dungeon();
        let blocked = MultiFloorBlocked::new();

        let path = find_path_multifloor(&dungeon, (0, 0, 0), (1, 2, 2), &blocked);
        assert!(path.is_some());

        let actions = path.unwrap();
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, DungeonAction::UseStairs(_)))
        );
        assert!(matches!(actions.last(), Some(DungeonAction::Move(_))));
    }

    #[test]
    fn test_three_floor_path() {
        let dungeon = make_three_floor_dungeon();
        let blocked = MultiFloorBlocked::new();

        let path = find_path_multifloor(&dungeon, (0, 0, 0), (2, 2, 2), &blocked);
        assert!(path.is_some());

        let actions = path.unwrap();
        let stair_count = actions
            .iter()
            .filter(|a| matches!(a, DungeonAction::UseStairs(_)))
            .count();
        assert_eq!(stair_count, 2, "Should use stairs twice for 3-floor path");
    }

    #[test]
    fn test_bfs_floor_sequence_direct() {
        let dungeon = make_two_floor_dungeon();
        let sequence = bfs_floor_sequence(&dungeon, 0, 1);
        assert_eq!(sequence, Some(vec![1]));
    }

    #[test]
    fn test_bfs_floor_sequence_same_floor() {
        let dungeon = make_two_floor_dungeon();
        let sequence = bfs_floor_sequence(&dungeon, 0, 0);
        assert_eq!(sequence, Some(vec![]));
    }

    #[test]
    fn test_bfs_floor_sequence_three_floors() {
        let dungeon = make_three_floor_dungeon();
        let sequence = bfs_floor_sequence(&dungeon, 0, 2);
        assert_eq!(sequence, Some(vec![1, 2]));
    }

    #[test]
    fn test_bfs_floor_sequence_unreachable() {
        let dungeon = DungeonMap {
            floors: vec![parse_grid("."), parse_grid("."), parse_grid(".")],
            stairs: vec![StairConnection {
                from: (0, 0, 0),
                to: (1, 0, 0),
            }],
            start: (0, 0, 0),
            goal: (2, 0, 0),
            monsters: vec![],
            treasures: vec![],
        };

        let sequence = bfs_floor_sequence(&dungeon, 0, 2);
        assert!(sequence.is_none(), "Floor 2 should be unreachable");
    }

    #[test]
    fn test_dungeon_action_name() {
        assert_eq!(DungeonAction::Move(0).name(), "↑ Up");
        assert_eq!(DungeonAction::Move(1).name(), "↓ Down");
        assert_eq!(DungeonAction::Move(2).name(), "← Left");
        assert_eq!(DungeonAction::Move(3).name(), "→ Right");
        assert_eq!(DungeonAction::Attack.name(), "⚔ Attack");
        assert_eq!(DungeonAction::UseStairs(0).name(), "🪜 Stairs");
    }

    #[test]
    fn test_dungeon_action_detail() {
        assert_eq!(DungeonAction::UseStairs(3).detail(), "🪜 Stairs #3");
        assert_eq!(DungeonAction::Attack.detail(), "⚔ Attack");
    }

    #[test]
    fn test_multi_floor_target_pos() {
        let monsters = vec![(0, 1, 1), (1, 2, 2)];
        let treasures = vec![(0, 3, 3)];
        let goal = (2, 0, 0);

        assert_eq!(
            MultiFloorTarget::Monster(0).pos(&monsters, &treasures, goal),
            (0, 1, 1)
        );
        assert_eq!(
            MultiFloorTarget::Monster(1).pos(&monsters, &treasures, goal),
            (1, 2, 2)
        );
        assert_eq!(
            MultiFloorTarget::Treasure(0).pos(&monsters, &treasures, goal),
            (0, 3, 3)
        );
        assert_eq!(
            MultiFloorTarget::Goal.pos(&monsters, &treasures, goal),
            (2, 0, 0)
        );
    }

    #[test]
    fn test_blocked_stair_uses_alternative() {
        let floor0 = parse_grid(". . .\n. . .\n. . .");
        let floor1 = parse_grid(". . .\n. . .\n. . .");

        let dungeon = DungeonMap {
            floors: vec![floor0, floor1],
            stairs: vec![
                StairConnection {
                    from: (0, 2, 2),
                    to: (1, 0, 0),
                },
                StairConnection {
                    from: (0, 0, 2),
                    to: (1, 2, 0),
                },
            ],
            start: (0, 0, 0),
            goal: (1, 2, 2),
            monsters: vec![],
            treasures: vec![],
        };

        // Block the closer stair at (2, 2)
        let mut blocked = MultiFloorBlocked::new();
        let mut floor_blocked = HashSet::new();
        floor_blocked.insert((2, 2));
        blocked.insert(0, floor_blocked);

        let path = find_path_multifloor(&dungeon, (0, 0, 0), (1, 2, 2), &blocked);
        assert!(path.is_some(), "Should find path via alternative stair");

        let actions = path.unwrap();
        // Should use stair index 1 (the alternative at (0, 2))
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, DungeonAction::UseStairs(1)))
        );
    }

    #[test]
    fn test_find_path_on_floor() {
        let dungeon = make_two_floor_dungeon();
        let blocked = HashSet::new();

        let path = find_path_on_floor(&dungeon, 0, (0, 0), (2, 2), &blocked);
        assert!(path.is_some());
        assert!(!path.unwrap().is_empty());
    }

    #[test]
    fn test_find_path_on_floor_invalid_floor() {
        let dungeon = make_two_floor_dungeon();
        let blocked = HashSet::new();

        let path = find_path_on_floor(&dungeon, 5, (0, 0), (2, 2), &blocked);
        assert!(path.is_none(), "Invalid floor should return None");
    }

    #[test]
    fn test_same_position_returns_empty() {
        let dungeon = make_two_floor_dungeon();
        let blocked = MultiFloorBlocked::new();

        let path = find_path_multifloor(&dungeon, (0, 1, 1), (0, 1, 1), &blocked);
        assert!(path.is_some());
        assert!(path.unwrap().is_empty());
    }
}
