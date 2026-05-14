//! ReplayBackwardWalker — GFlowNet-inspired backward policy extraction.
//!
//! The GFlowNet paper constructs the backward policy by reversing graph edges.
//! We walk winning game replays backward through the action validator to learn
//! which actions are on shortest paths.
//!
//! At each tick, we test alternative actions via `is_safe_action()` and record
//! which alternatives were also safe. This gives us:
//! - (state, chosen_action, safe_alternatives) → backward policy data
//! - Safe alternatives = "what else could have worked" = P_B(s'|s)
//!
//! # Usage
//!
//! ```rust,ignore
//! let samples: Vec<ReplaySample> = load_replay("replay.jsonl");
//! let walker = ReplayBackwardWalker::new(&grid);
//! let backward_data = walker.walk_backward(&samples);
//! ```

use super::players::is_safe_action;
use super::replay::ReplaySample;
use super::{ArenaGrid, BomberAction, GridPos};

/// Bomb tuple: `((x, y), blast_range, fuse_ticks_remaining)`.
///
/// Mirrors the private `KnownBomb` type in `players.rs`.
type BombRef = ((i32, i32), u32, u32);

// ── BackwardSample ──────────────────────────────────────────────

/// A single backward policy sample from replay analysis.
///
/// Records the state, the chosen action, and which alternative actions
/// were also safe at that tick. This gives us the backward policy P_B(s'|s).
#[derive(Clone, Debug)]
pub struct BackwardSample {
    /// Tick number in the replay.
    pub tick: u32,
    /// Round number in the replay.
    pub round: u32,
    /// Action that was actually taken (as index).
    pub chosen_action: usize,
    /// Safe alternative actions (excluding chosen).
    pub safe_alternatives: Vec<usize>,
    /// Number of safe alternatives found.
    pub num_alternatives: usize,
    /// Quality of the replay sample.
    pub quality: f32,
}

impl BackwardSample {
    /// Number of safe actions including the chosen one.
    pub fn total_safe_actions(&self) -> usize {
        self.num_alternatives + 1 // alternatives + chosen
    }

    /// Backward probability: uniform over safe actions.
    /// P_B(chosen | state) = 1 / |safe_actions|
    pub fn backward_prob(&self) -> f32 {
        let total = self.total_safe_actions();
        if total == 0 {
            return 0.0;
        }
        1.0 / total as f32
    }
}

// ── BackwardWalkResult ──────────────────────────────────────────

/// Result of backward walking a replay.
#[derive(Clone, Debug)]
pub struct BackwardWalkResult {
    /// Total ticks analyzed.
    pub ticks_analyzed: usize,
    /// Backward samples (one per tick with safe alternatives).
    pub samples: Vec<BackwardSample>,
    /// Total safe alternatives found across all ticks.
    pub total_alternatives: usize,
    /// Average alternatives per tick.
    pub avg_alternatives: f32,
    /// Ticks with ≥2 safe alternatives.
    pub ticks_with_multiple: usize,
}

impl BackwardWalkResult {
    /// Average safe alternatives per tick.
    pub fn avg_alternatives_per_tick(&self) -> f32 {
        if self.ticks_analyzed == 0 {
            return 0.0;
        }
        self.total_alternatives as f32 / self.ticks_analyzed as f32
    }

    /// Fraction of ticks with ≥2 safe alternatives.
    pub fn fraction_with_multiple(&self) -> f32 {
        if self.ticks_analyzed == 0 {
            return 0.0;
        }
        self.ticks_with_multiple as f32 / self.ticks_analyzed as f32
    }
}

// ── ReplayBackwardWalker ────────────────────────────────────────

/// Walks winning game replays backward to extract backward policy data.
///
/// At each tick (from last to first), tests all alternative actions via
/// `is_safe_action()` to find which ones were also safe. This gives us:
/// - The backward policy P_B(s'|s) as uniform over safe actions
/// - Trajectory structure: how many alternative paths existed
///
/// # Design
///
/// - **Offline processing**: Processes replay JSONL, doesn't run during game ticks
/// - **Zero runtime overhead**: Analysis is done post-game
/// - **Composable**: Output feeds into bandit rewards or training data
pub struct ReplayBackwardWalker<'a> {
    /// Reference to the arena grid for walkability checks.
    grid: &'a ArenaGrid,
}

impl<'a> ReplayBackwardWalker<'a> {
    /// Create a new backward walker with the given arena grid.
    pub fn new(grid: &'a ArenaGrid) -> Self {
        Self { grid }
    }

    /// Walk a replay backward from final tick to first tick.
    ///
    /// At each tick, tests alternative actions via `is_safe_action()`.
    /// Returns backward policy samples for the entire replay.
    ///
    /// # Arguments
    ///
    /// * `samples` — Replay samples sorted by tick (ascending)
    ///
    /// # Returns
    ///
    /// `BackwardWalkResult` with per-tick backward policy data.
    pub fn walk_backward(&self, samples: &[ReplaySample]) -> BackwardWalkResult {
        let mut result = BackwardWalkResult {
            ticks_analyzed: 0,
            samples: Vec::with_capacity(samples.len()),
            total_alternatives: 0,
            avg_alternatives: 0.0,
            ticks_with_multiple: 0,
        };

        if samples.is_empty() {
            return result;
        }

        // Walk backward: last tick to first tick
        for sample in samples.iter().rev() {
            let pos = GridPos {
                x: sample.player_pos[0] as i32,
                y: sample.player_pos[1] as i32,
            };

            let _chosen = BomberAction::from(sample.action as usize);

            // Convert replay bombs `[(x, y, range, fuse)]` → `[((x, y), range, fuse)]`
            let bombs: Vec<BombRef> = sample
                .bombs
                .iter()
                .map(|b| ((b[0] as i32, b[1] as i32), b[2] as u32, b[3] as u32))
                .collect();

            // Test all actions except the chosen one
            let mut safe_alternatives = Vec::new();
            for action in BomberAction::all() {
                let action_idx = action.as_usize();
                if action_idx == sample.action as usize {
                    continue; // Skip the chosen action
                }

                if is_safe_action(&action, self.grid, pos, &bombs) {
                    safe_alternatives.push(action_idx);
                }
            }

            let num_alternatives = safe_alternatives.len();
            if num_alternatives >= 2 {
                result.ticks_with_multiple += 1;
            }

            result.total_alternatives += num_alternatives;
            result.ticks_analyzed += 1;

            result.samples.push(BackwardSample {
                tick: sample.tick,
                round: sample.round,
                chosen_action: sample.action as usize,
                safe_alternatives,
                num_alternatives,
                quality: sample.quality,
            });
        }

        result.avg_alternatives = result.avg_alternatives_per_tick();
        result
    }

    /// Walk multiple replays and aggregate results.
    ///
    /// Useful for batch processing replay files.
    pub fn walk_backward_multi(&self, replays: &[Vec<ReplaySample>]) -> BackwardWalkResult {
        let mut combined = BackwardWalkResult {
            ticks_analyzed: 0,
            samples: Vec::new(),
            total_alternatives: 0,
            avg_alternatives: 0.0,
            ticks_with_multiple: 0,
        };

        for replay in replays {
            let result = self.walk_backward(replay);
            combined.ticks_analyzed += result.ticks_analyzed;
            combined.samples.extend(result.samples);
            combined.total_alternatives += result.total_alternatives;
            combined.ticks_with_multiple += result.ticks_with_multiple;
        }

        combined.avg_alternatives = combined.avg_alternatives_per_tick();
        combined
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::arena::EMPTY_ARENA;
    use super::*;

    /// Create an empty arena grid using the preset template.
    fn make_grid() -> ArenaGrid {
        ArenaGrid::fixed(EMPTY_ARENA).expect("EMPTY_ARENA should parse")
    }

    /// Build a minimal `ReplaySample` for testing.
    fn make_sample(tick: u32, px: u8, py: u8, action: u8, quality: f32) -> ReplaySample {
        ReplaySample {
            board: vec![0u8; 169],
            player_pos: [px, py],
            player_id: 0,
            bombs: vec![],
            powerups: vec![],
            action,
            quality,
            tick,
            round: 1,
            player_type: "Test".to_string(),
            danger_level: 0,
            nearest_opponent_dist: 255,
            escape_routes: 4,
        }
    }

    #[test]
    fn test_backward_sample_backward_prob() {
        let sample = BackwardSample {
            tick: 1,
            round: 1,
            chosen_action: 0,
            safe_alternatives: vec![1, 2],
            num_alternatives: 2,
            quality: 1.0,
        };
        // 3 safe actions total (chosen + 2 alternatives)
        assert!((sample.backward_prob() - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(sample.total_safe_actions(), 3);
    }

    #[test]
    fn test_backward_sample_no_alternatives() {
        let sample = BackwardSample {
            tick: 1,
            round: 1,
            chosen_action: 0,
            safe_alternatives: vec![],
            num_alternatives: 0,
            quality: 1.0,
        };
        assert!((sample.backward_prob() - 1.0).abs() < 1e-6);
        assert_eq!(sample.total_safe_actions(), 1);
    }

    #[test]
    fn test_walk_backward_empty() {
        let grid = make_grid();
        let walker = ReplayBackwardWalker::new(&grid);
        let result = walker.walk_backward(&[]);

        assert_eq!(result.ticks_analyzed, 0);
        assert!(result.samples.is_empty());
        assert_eq!(result.total_alternatives, 0);
    }

    #[test]
    fn test_walk_backward_finds_alternatives() {
        let grid = make_grid();
        let walker = ReplayBackwardWalker::new(&grid);

        // Player at (1,1) — spawn corner, open floor.
        // Action Wait (5) was chosen — other movement actions may be safe.
        let samples = vec![make_sample(0, 1, 1, 5, 1.0)];

        let result = walker.walk_backward(&samples);

        assert_eq!(result.ticks_analyzed, 1);
        assert_eq!(result.samples.len(), 1);

        let sample = &result.samples[0];
        assert_eq!(sample.chosen_action, 5); // Wait
        assert!(
            sample.num_alternatives > 0,
            "Should find safe alternatives at (1,1)"
        );
    }

    #[test]
    fn test_walk_backward_multi() {
        let grid = make_grid();
        let walker = ReplayBackwardWalker::new(&grid);

        let replay1 = vec![make_sample(0, 1, 1, 5, 1.0)];
        let replay2 = vec![make_sample(0, 3, 3, 5, 1.0)];

        let result = walker.walk_backward_multi(&[replay1, replay2]);

        assert_eq!(result.ticks_analyzed, 2);
        assert_eq!(result.samples.len(), 2);
    }

    #[test]
    fn test_walk_backward_order_is_reversed() {
        let grid = make_grid();
        let walker = ReplayBackwardWalker::new(&grid);

        let samples = vec![
            make_sample(0, 1, 1, 5, 0.5),
            make_sample(1, 1, 2, 5, 0.7),
            make_sample(2, 1, 3, 5, 1.0),
        ];

        let result = walker.walk_backward(&samples);

        // Walked backward: tick 2 first, then 1, then 0
        assert_eq!(result.samples[0].tick, 2);
        assert_eq!(result.samples[1].tick, 1);
        assert_eq!(result.samples[2].tick, 0);
    }

    #[test]
    fn test_walk_result_metrics() {
        let grid = make_grid();
        let walker = ReplayBackwardWalker::new(&grid);

        let samples = vec![make_sample(0, 1, 1, 5, 1.0)];
        let result = walker.walk_backward(&samples);

        assert!(result.avg_alternatives >= 0.0);
        assert!(result.fraction_with_multiple() >= 0.0);
    }

    #[test]
    fn test_walk_backward_with_bombs() {
        let grid = make_grid();
        let walker = ReplayBackwardWalker::new(&grid);

        // Player at (3,3) with a bomb at (5,3) range=2 fuse=3
        let mut sample = make_sample(0, 3, 3, 5, 1.0);
        sample.bombs = vec![[5, 3, 2, 3]];

        let result = walker.walk_backward(&[sample]);

        assert_eq!(result.ticks_analyzed, 1);
        // Right action (3) should be unsafe — walks toward bomb
        let right_safe = result.samples[0].safe_alternatives.contains(&3);
        // The bomb at (5,3) with range 2 threatens (3,3) to (5,3) on that row
        // Walking right toward (4,3) would be in blast zone
        assert!(!right_safe, "Right should be unsafe near bomb at (5,3)");
    }
}
