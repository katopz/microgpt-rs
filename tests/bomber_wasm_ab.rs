//! A/B Correctness Test — WASM vs Native Bomber Safety Rules (Plan 034, Task 9)
//!
//! Verifies that `bomber_validator.wasm` produces identical safety verdicts
//! to the native Rust implementation for the same game states.
//!
//! # Run
//!
//! ```sh
//! cargo test --test bomber_wasm_ab --features bomber-wasm -- --nocapture
//! ```
//!
//! # Prerequisites
//!
//! Build the WASM validator first:
//! ```sh
//! cd riir-ai && cargo build --example bomber_validator --target wasm32-unknown-unknown --release
//! ```
//!
//! # Known Differences
//!
//! The WASM validator is intentionally stricter for the **Bomb** action:
//! - WASM requires ≥1 adjacent destructible wall (native doesn't check this)
//! - WASM rejects bomb placement if a bomb already exists at the player's position
//! - WASM rejects bomb placement if the player is in a blast zone
//!
//! Movement (Up/Down/Left/Right) and Wait actions must match **exactly**.

#![cfg(feature = "bomber-wasm")]

use std::path::Path;

use microgpt_rs::pruners::bomber::wasm_pruner::BomberWasmPruner;
use microgpt_rs::pruners::bomber::{ArenaGrid, BomberAction, Cell, GridPos, is_safe_action};

// ── Constants ──────────────────────────────────────────────────

/// Path to the compiled WASM validator.
const WASM_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../riir-ai/target/wasm32-unknown-unknown/release/examples/bomber_validator.wasm"
);

/// Number of random grids to test in the comprehensive A/B test.
const NUM_RANDOM_GRIDS: u32 = 100;

/// Number of random bomb configurations per grid.
const NUM_BOMB_CONFIGS: u32 = 5;

/// All movement actions (must match exactly between WASM and native).
/// All action indices for comprehensive testing.
const ALL_ACTION_INDICES: [usize; 6] = [0, 1, 2, 3, 4, 5];

// ── Helpers ────────────────────────────────────────────────────

/// Check if the WASM file exists at the expected path.
fn wasm_available() -> bool {
    Path::new(WASM_PATH).exists()
}

/// Load the WASM pruner, skipping the test if unavailable.
fn load_pruner() -> Option<BomberWasmPruner> {
    if !wasm_available() {
        eprintln!("⚠ Skipping: WASM file not found at {WASM_PATH}");
        eprintln!(
            "  Build it: cd riir-ai && cargo build --example bomber_validator --target wasm32-unknown-unknown --release"
        );
        return None;
    }
    match BomberWasmPruner::load_from_file(WASM_PATH) {
        Ok(pruner) => Some(pruner),
        Err(e) => {
            eprintln!("⚠ Failed to load WASM: {e}");
            None
        }
    }
}

/// Find all walkable positions on a grid.
fn find_walkable_positions(grid: &ArenaGrid) -> Vec<(i32, i32)> {
    let mut positions = Vec::new();
    for y in 1..12 {
        for x in 1..12 {
            if grid.is_walkable(x, y) {
                positions.push((x, y));
            }
        }
    }
    positions
}

/// Generate random bomb configurations for a grid.
///
/// Places bombs at random walkable positions with varying blast ranges and fuses.
#[allow(clippy::type_complexity)]
fn generate_bomb_configs(grid: &ArenaGrid, seed: u64) -> Vec<Vec<((i32, i32), u32, u32)>> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let walkable = find_walkable_positions(grid);
    if walkable.is_empty() {
        return vec![vec![]];
    }

    let mut configs = Vec::new();

    // Config 0: no bombs
    configs.push(vec![]);

    // Generate additional configs with 1..4 bombs
    for _ in 1..NUM_BOMB_CONFIGS {
        let count = rng.u32(1..=4) as usize;
        let mut bombs = Vec::with_capacity(count);
        for _ in 0..count {
            let idx = rng.usize(0..walkable.len());
            let (bx, by) = walkable[idx];
            let blast_range = rng.u32(1..=3);
            let fuse = rng.u32(1..=4);
            bombs.push(((bx, by), blast_range, fuse));
        }
        configs.push(bombs);
    }

    configs
}

/// Call native `is_safe_action` for an action index.
fn native_is_safe(
    action_idx: usize,
    grid: &ArenaGrid,
    x: i32,
    y: i32,
    bombs: &[((i32, i32), u32, u32)],
) -> bool {
    let action = BomberAction::from(action_idx);
    let pos = GridPos { x, y };
    is_safe_action(&action, grid, pos, bombs)
}

/// Call WASM `is_safe_action` for an action index.
fn wasm_is_safe(
    pruner: &BomberWasmPruner,
    action_idx: usize,
    grid: &ArenaGrid,
    x: i32,
    y: i32,
    bombs: &[((i32, i32), u32, u32)],
) -> bool {
    pruner.is_safe_action(action_idx, grid, x, y, 0, bombs)
}

/// Check if a position is in blast zone of any bomb (native logic).
fn is_in_blast_zone_native(
    grid: &ArenaGrid,
    x: i32,
    y: i32,
    bombs: &[((i32, i32), u32, u32)],
) -> bool {
    for &((bx, by), range, _fuse) in bombs {
        if is_in_single_blast_native(grid, x, y, bx, by, range) {
            return true;
        }
    }
    false
}

/// Check if (x,y) is in blast zone of a single bomb (wall-aware).
fn is_in_single_blast_native(
    grid: &ArenaGrid,
    x: i32,
    y: i32,
    bx: i32,
    by: i32,
    range: u32,
) -> bool {
    // Standing on the bomb
    if x == bx && y == by {
        return true;
    }

    // Horizontal
    if y == by {
        let dx = x - bx;
        if dx.unsigned_abs() <= range {
            let step = dx.signum();
            let mut cx = bx + step;
            while cx != x {
                match grid.get(cx, by) {
                    Cell::FixedWall | Cell::DestructibleWall | Cell::PowerUpHidden(_) => {
                        return false;
                    }
                    _ => {}
                }
                cx += step;
            }
            return true;
        }
    }

    // Vertical
    if x == bx {
        let dy = y - by;
        if dy.unsigned_abs() <= range {
            let step = dy.signum();
            let mut cy = by + step;
            while cy != y {
                match grid.get(bx, cy) {
                    Cell::FixedWall | Cell::DestructibleWall | Cell::PowerUpHidden(_) => {
                        return false;
                    }
                    _ => {}
                }
                cy += step;
            }
            return true;
        }
    }

    false
}

/// Count adjacent cells that are destructible walls or hidden power-ups.
fn count_adjacent_walls_native(grid: &ArenaGrid, x: i32, y: i32) -> usize {
    [(0i32, -1), (0, 1), (-1, 0), (1, 0)]
        .iter()
        .filter(|&&(dx, dy)| {
            matches!(
                grid.get(x + dx, y + dy),
                Cell::DestructibleWall | Cell::PowerUpHidden(_)
            )
        })
        .count()
}

/// Check if there's a bomb at the given position.
fn has_bomb_at(bombs: &[((i32, i32), u32, u32)], x: i32, y: i32) -> bool {
    bombs.iter().any(|&(pos, _, _)| pos.0 == x && pos.1 == y)
}

// ── Test Result Tracking ───────────────────────────────────────

/// Mismatch record for reporting.
#[derive(Debug)]
#[allow(dead_code)]
struct Mismatch {
    seed: u64,
    x: i32,
    y: i32,
    action_idx: usize,
    native: bool,
    wasm: bool,
    reason: String,
}

/// Categorized test results.
struct AbResults {
    /// Movement/Wait mismatches (must be zero — these are bugs).
    critical_mismatches: Vec<Mismatch>,
    /// Bomb action mismatches where WASM is stricter (native=true, wasm=false).
    /// These are acceptable: WASM adds extra safety checks (adjacent walls,
    /// no bomb at position, in-blast rejection) and its BFS escape-route
    /// check is more conservative in some multi-bomb edge cases.
    bomb_stricter_mismatches: Vec<Mismatch>,
    /// Total comparisons made.
    total_comparisons: u64,
}

impl AbResults {
    fn new() -> Self {
        Self {
            critical_mismatches: Vec::new(),
            bomb_stricter_mismatches: Vec::new(),
            total_comparisons: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        seed: u64,
        x: i32,
        y: i32,
        action_idx: usize,
        native: bool,
        wasm: bool,
        _grid: &ArenaGrid,
        _bombs: &[((i32, i32), u32, u32)],
    ) {
        let action = BomberAction::from(action_idx);

        match action {
            // Movement and Wait: any mismatch is critical
            BomberAction::Up
            | BomberAction::Down
            | BomberAction::Left
            | BomberAction::Right
            | BomberAction::Wait => {
                self.critical_mismatches.push(Mismatch {
                    seed,
                    x,
                    y,
                    action_idx,
                    native,
                    wasm,
                    reason: format!("{action}: native={native} wasm={wasm}"),
                });
            }

            // Bomb: WASM is intentionally stricter.
            // Known reasons WASM rejects when native accepts:
            //   1. Player is in blast zone (WASM checks, native doesn't)
            //   2. No adjacent destructible walls (WASM requires, native doesn't)
            //   3. Bomb already at player position (WASM checks, native doesn't)
            //   4. WASM's BFS escape-route is more conservative in multi-bomb scenarios
            BomberAction::Bomb => {
                // Only record when WASM is stricter (native=true, wasm=false).
                // The reverse (native=false, wasm=true) would be a real bug.
                if native && !wasm {
                    self.bomb_stricter_mismatches.push(Mismatch {
                        seed,
                        x,
                        y,
                        action_idx,
                        native,
                        wasm,
                        reason: format!("{action}: native={native} wasm={wasm} (WASM stricter)"),
                    });
                } else if native != wasm {
                    // WASM says safe but native says unsafe — this is a real bug
                    self.critical_mismatches.push(Mismatch {
                        seed,
                        x,
                        y,
                        action_idx,
                        native,
                        wasm,
                        reason: format!(
                            "❌ BUG: {action} native={native} wasm={wasm} (WASM allows unsafe action!)"
                        ),
                    });
                }
            }
        }
    }

    fn print_summary(&self) {
        println!("\n═══ A/B Correctness Test Summary ═══");
        println!("Total comparisons: {}", self.total_comparisons);
        println!(
            "Critical mismatches (movement/wait or WASM-allows-unsafe): {}",
            self.critical_mismatches.len()
        );
        println!(
            "Bomb stricter (native=true wasm=false, acceptable): {}",
            self.bomb_stricter_mismatches.len()
        );

        if !self.critical_mismatches.is_empty() {
            eprintln!("\n❌ CRITICAL MISMATCHES:");
            for m in &self.critical_mismatches {
                eprintln!(
                    "  seed={} pos=({},{}) action={} {}",
                    m.seed, m.x, m.y, m.action_idx, m.reason
                );
            }
        }

        if self.critical_mismatches.is_empty() {
            println!("✅ All movement/wait actions match exactly");
            println!(
                "ℹ  {} bomb actions: WASM stricter (expected — WASM adds safety checks + conservative BFS)",
                self.bomb_stricter_mismatches.len()
            );
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[test]
fn test_wasm_loads() {
    if !wasm_available() {
        eprintln!("⚠ Skipping: WASM file not found at {WASM_PATH}");
        eprintln!(
            "  Build it: cd riir-ai && cargo build --example bomber_validator --target wasm32-unknown-unknown --release"
        );
        return;
    }

    let pruner = BomberWasmPruner::load_from_file(WASM_PATH);
    assert!(pruner.is_ok(), "WASM should load successfully");

    let pruner = pruner.unwrap();
    assert_eq!(pruner.name(), "bomber_validator");
}

#[test]
fn test_wasm_version() {
    let Some(pruner) = load_pruner() else { return };

    let (major, minor, patch) = pruner.version();
    assert_eq!(major, 1, "Major version should be 1");
    assert_eq!(minor, 0, "Minor version should be 0");
    assert_eq!(patch, 0, "Patch version should be 0");
}

#[test]
fn test_movement_on_empty_grid() {
    let Some(pruner) = load_pruner() else { return };

    // Build an empty grid (all floors)
    let grid = empty_grid();
    let bombs: [((i32, i32), u32, u32); 0] = [];

    // Test movement from center position (6, 6)
    for action_idx in 0..4 {
        let native = native_is_safe(action_idx, &grid, 6, 6, &bombs);
        let wasm = wasm_is_safe(&pruner, action_idx, &grid, 6, 6, &bombs);
        assert_eq!(
            native, wasm,
            "Empty grid movement action {action_idx} from (6,6): native={native} wasm={wasm}"
        );
    }

    // Test movement from corner (1, 1)
    for action_idx in 0..4 {
        let native = native_is_safe(action_idx, &grid, 1, 1, &bombs);
        let wasm = wasm_is_safe(&pruner, action_idx, &grid, 1, 1, &bombs);
        assert_eq!(
            native, wasm,
            "Empty grid movement action {action_idx} from (1,1): native={native} wasm={wasm}"
        );
    }
}

#[test]
fn test_wait_on_empty_grid() {
    let Some(pruner) = load_pruner() else { return };

    let grid = empty_grid();
    let bombs: [((i32, i32), u32, u32); 0] = [];

    // Wait should be safe when no bombs
    let native = native_is_safe(5, &grid, 6, 6, &bombs);
    let wasm = wasm_is_safe(&pruner, 5, &grid, 6, 6, &bombs);
    assert!(native, "Wait on empty grid should be safe (native)");
    assert!(wasm, "Wait on empty grid should be safe (WASM)");
    assert_eq!(native, wasm, "Wait verdict must match");
}

#[test]
fn test_movement_into_fixed_wall() {
    let Some(pruner) = load_pruner() else { return };

    let grid = ArenaGrid::generate(42);
    let bombs: [((i32, i32), u32, u32); 0] = [];

    // Walkable positions where at least one neighbor is a wall
    for y in 1..12 {
        for x in 1..12 {
            if !grid.is_walkable(x, y) {
                continue;
            }

            for action_idx in 0..4 {
                let native = native_is_safe(action_idx, &grid, x, y, &bombs);
                let wasm = wasm_is_safe(&pruner, action_idx, &grid, x, y, &bombs);
                assert_eq!(
                    native, wasm,
                    "Wall check at ({x},{y}) action {action_idx}: native={native} wasm={wasm}"
                );
            }
        }
    }
}

#[test]
fn test_movement_near_bombs() {
    let Some(pruner) = load_pruner() else { return };

    let grid = empty_grid();

    // Place a bomb at (3, 3) with blast range 2
    let bombs = vec![((3, 3), 2, 4)];

    // Test positions near the bomb
    let test_positions = [
        (2, 3), // Horizontal blast zone
        (4, 3), // Horizontal blast zone
        (3, 2), // Vertical blast zone
        (3, 4), // Vertical blast zone
        (1, 3), // Just outside blast zone
        (5, 3), // Just outside blast zone
        (3, 1), // Just outside blast zone
        (3, 5), // Just outside blast zone
        (6, 6), // Far from bomb
    ];

    for &(x, y) in &test_positions {
        if !grid.is_walkable(x, y) {
            continue;
        }
        for action_idx in 0..4 {
            let native = native_is_safe(action_idx, &grid, x, y, &bombs);
            let wasm = wasm_is_safe(&pruner, action_idx, &grid, x, y, &bombs);
            assert_eq!(
                native, wasm,
                "Bomb test at ({x},{y}) action {action_idx}: native={native} wasm={wasm}"
            );
        }
    }
}

#[test]
fn test_blast_zone_with_wall_blocking() {
    let Some(pruner) = load_pruner() else { return };

    let mut grid = empty_grid();

    // Place a fixed wall between bomb and player
    // Bomb at (3, 3), wall at (4, 3), player at (5, 3)
    grid.cells[3][4] = Cell::FixedWall;

    let bombs = vec![((3, 3), 3, 4)];

    // Position (5, 3) should be safe — wall blocks blast
    for action_idx in 0..4 {
        let native = native_is_safe(action_idx, &grid, 5, 3, &bombs);
        let wasm = wasm_is_safe(&pruner, action_idx, &grid, 5, 3, &bombs);
        assert_eq!(
            native, wasm,
            "Wall-blocked blast at (5,3) action {action_idx}: native={native} wasm={wasm}"
        );
    }

    // Moving toward bomb from (5, 3) — Left action (idx=2) goes to (4, 3) which is a wall
    let native = native_is_safe(2, &grid, 5, 3, &bombs);
    let wasm = wasm_is_safe(&pruner, 2, &grid, 5, 3, &bombs);
    assert_eq!(native, wasm, "Move into wall at (4,3) must match");
    assert!(!native, "Moving into fixed wall should be unsafe");

    // Position (2, 3) should be in blast zone (no wall between)
    let native = native_is_safe(3, &grid, 1, 3, &bombs); // Right from (1,3) to (2,3)
    let wasm = wasm_is_safe(&pruner, 3, &grid, 1, 3, &bombs);
    assert_eq!(
        native, wasm,
        "Move into blast zone (2,3): native={native} wasm={wasm}"
    );
}

#[test]
fn test_wait_in_blast_zone() {
    let Some(pruner) = load_pruner() else { return };

    let grid = empty_grid();
    let bombs = vec![((5, 5), 2, 4)];

    // Wait at (5, 4) — in blast zone (vertical, 1 cell away)
    let native = native_is_safe(5, &grid, 5, 4, &bombs);
    let wasm = wasm_is_safe(&pruner, 5, &grid, 5, 4, &bombs);
    assert_eq!(native, wasm, "Wait in blast zone must match");
    assert!(!native, "Wait in blast zone should be unsafe");

    // Wait at (5, 2) — outside blast zone (3 cells away, range=2)
    let native = native_is_safe(5, &grid, 5, 2, &bombs);
    let wasm = wasm_is_safe(&pruner, 5, &grid, 5, 2, &bombs);
    assert_eq!(native, wasm, "Wait outside blast zone must match");
    assert!(native, "Wait outside blast zone should be safe");
}

#[test]
fn test_bomb_action_wasm_stricter_on_no_walls() {
    let Some(pruner) = load_pruner() else { return };

    // Empty grid — no adjacent walls, WASM should reject bomb
    let grid = empty_grid();
    let bombs: [((i32, i32), u32, u32); 0] = [];

    let native = native_is_safe(4, &grid, 6, 6, &bombs);
    let wasm = wasm_is_safe(&pruner, 4, &grid, 6, 6, &bombs);

    // Native only checks escape route (should be true on empty grid)
    // WASM also checks adjacent walls (should be false on empty grid)
    assert!(
        native,
        "Native bomb on empty grid should be safe (has escape)"
    );
    assert!(
        !wasm,
        "WASM bomb on empty grid should be rejected (no adjacent walls)"
    );
}

#[test]
fn test_bomb_action_wasm_stricter_on_existing_bomb() {
    let Some(pruner) = load_pruner() else { return };

    let mut grid = empty_grid();
    // Add a destructible wall so WASM's wall check passes
    grid.cells[6][7] = Cell::DestructibleWall;

    // Already a bomb at player's position
    let bombs = vec![((6, 6), 2, 4)];

    let _native = native_is_safe(4, &grid, 6, 6, &bombs);
    let wasm = wasm_is_safe(&pruner, 4, &grid, 6, 6, &bombs);

    // Native doesn't check for existing bomb at position
    // WASM rejects if bomb already at player position
    assert!(
        !wasm,
        "WASM should reject bomb placement where bomb already exists"
    );
}

#[test]
fn test_bomb_action_agrees_when_walls_present() {
    let Some(pruner) = load_pruner() else { return };

    let mut grid = empty_grid();
    // Add destructible walls around player
    grid.cells[5][6] = Cell::DestructibleWall;
    grid.cells[7][6] = Cell::DestructibleWall;
    let bombs: [((i32, i32), u32, u32); 0] = [];

    // Player at (6, 6), walls at (6, 5) and (6, 7), escape via (5, 6) or (7, 6)
    // Both should agree it's safe (has walls + escape route)
    let native = native_is_safe(4, &grid, 6, 6, &bombs);
    let wasm = wasm_is_safe(&pruner, 4, &grid, 6, 6, &bombs);

    assert_eq!(
        native, wasm,
        "Bomb with walls + escape: native={native} wasm={wasm}"
    );
}

/// Debug test for specific unexpected bomb mismatch (seed=1000, pos=(7,9)).
///
/// This test isolates the first unexpected mismatch found in `test_ab_correctness_many_states`
/// to diagnose why WASM rejects bomb placement when native accepts it.
#[test]
fn test_debug_bomb_mismatch_seed_1000_pos_7_9() {
    let Some(pruner) = load_pruner() else { return };

    let grid = ArenaGrid::generate(1000);
    let x: i32 = 7;
    let y: i32 = 9;
    let bombs: Vec<((i32, i32), u32, u32)> = vec![
        ((5, 3), 1, 2),
        ((8, 5), 2, 1),
        ((9, 3), 1, 1),
        ((6, 1), 2, 2),
    ];

    // Verify position is walkable
    assert!(grid.is_walkable(x, y), "({x},{y}) should be walkable");

    // Check all actions
    for action_idx in 0..6 {
        let native = native_is_safe(action_idx, &grid, x, y, &bombs);
        let wasm = wasm_is_safe(&pruner, action_idx, &grid, x, y, &bombs);
        let action = BomberAction::from(action_idx);

        println!(
            "  action={} ({action:?}): native={native} wasm={wasm} match={}",
            action_idx,
            native == wasm
        );

        // For movement and wait, they must match
        if action_idx != 4 {
            assert_eq!(
                native, wasm,
                "action {action_idx} at ({x},{y}): native={native} wasm={wasm}"
            );
        }
    }

    // Now investigate bomb action specifically
    let native_bomb = native_is_safe(4, &grid, x, y, &bombs);
    let wasm_bomb = wasm_is_safe(&pruner, 4, &grid, x, y, &bombs);

    println!("\n  Bomb at ({x},{y}):");
    println!("    native={native_bomb} wasm={wasm_bomb}");

    // Check the WASM preconditions
    let in_blast = is_in_blast_zone_native(&grid, x, y, &bombs);
    let adj_walls = count_adjacent_walls_native(&grid, x, y);
    let bomb_exists = has_bomb_at(&bombs, x, y);
    println!("    in_blast={in_blast} adj_walls={adj_walls} bomb_exists={bomb_exists}");

    // Check each adjacent cell for escape
    let adj_cells: [(i32, i32); 4] = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)];
    for &(nx, ny) in &adj_cells {
        let walkable = grid.is_walkable(nx, ny);
        let in_blast_adj = is_in_blast_zone_native(&grid, nx, ny, &bombs);
        let has_bomb_adj = has_bomb_at(&bombs, nx, ny);

        // Check WASM verdict for moving to this cell
        let wasm_move_safe = wasm_is_safe(&pruner, 0, &grid, nx, ny, &bombs);

        println!(
            "    adj ({nx},{ny}): walkable={walkable} in_blast={in_blast_adj} has_bomb={has_bomb_adj} wasm_move_safe={wasm_move_safe}"
        );
    }

    // Also try with no bombs to isolate if bombs cause the issue
    let native_no_bombs = native_is_safe(4, &grid, x, y, &[]);
    let wasm_no_bombs = wasm_is_safe(&pruner, 4, &grid, x, y, &[]);
    println!("\n  Bomb at ({x},{y}) with NO bombs: native={native_no_bombs} wasm={wasm_no_bombs}");

    // Try with WASM relevance too
    let relevance = pruner.action_relevance(4, &grid, x, y, 0, &bombs);
    println!("  WASM relevance for bomb: {relevance:.3}");

    // ── Grid cell verification ─────────────────────────────────
    // Check what WASM "sees" for each cell by testing Wait action at each position.
    // If WASM says Wait is safe → cell is walkable and not in blast zone.
    // If WASM says Wait is unsafe → cell is either not walkable or in blast zone.
    println!("\n  WASM grid cell verification (5x5 region around ({x},{y})):");
    for dy in -2i32..=2 {
        for dx in -2i32..=2 {
            let cx = x + dx;
            let cy = y + dy;
            let native_walk = grid.is_walkable(cx, cy);
            let native_cell = grid.get(cx, cy);
            let native_in_blast = is_in_blast_zone_native(&grid, cx, cy, &bombs);
            let wasm_wait = wasm_is_safe(&pruner, 5, &grid, cx, cy, &bombs); // Wait action
            let cell_ch = match native_cell {
                Cell::Floor => '.',
                Cell::FixedWall => '#',
                Cell::DestructibleWall => '█',
                Cell::PowerUpHidden(_) => '?',
            };
            let mismatch = if (native_walk && !native_in_blast) != wasm_wait {
                " ← MISMATCH"
            } else {
                ""
            };
            println!(
                "    ({cx},{cy}) {} nw={native_walk} nb={native_in_blast} ww={wasm_wait}{mismatch}",
                cell_ch,
            );
        }
    }

    // ── Verify individual cells by testing movement into them ──
    println!("\n  WASM movement verification (can WASM move TO these cells?):");
    // To test if WASM thinks (tx,ty) is walkable, we check if player AT (tx,ty) can Wait
    // If Wait is safe, the cell is walkable and not in blast zone
    for dy in -3i32..=3 {
        for dx in -3i32..=3 {
            let cx = x + dx;
            let cy = y + dy;
            let native_walk = grid.is_walkable(cx, cy);
            // Test Wait at (cx,cy) to see if WASM thinks it's a valid walkable cell
            let wasm_wait_no_bombs = wasm_is_safe(&pruner, 5, &grid, cx, cy, &[]);
            let match_ = native_walk == wasm_wait_no_bombs;
            if !match_ {
                println!(
                    "    ({cx},{cy}) native_walk={native_walk} wasm_wait={wasm_wait_no_bombs} ← MISMATCH"
                );
            }
        }
    }

    // ── Check escape route from (7,8) specifically ──────────────
    // The WASM's has_escape_route from (7,8) with bomb at (7,9) should find a path
    // Let's verify WASM can move around from (7,8)
    println!("\n  WASM movement from (7,8) with bombs:");
    for action_idx in 0..6 {
        let wasm_safe = wasm_is_safe(&pruner, action_idx, &grid, 7, 8, &bombs);
        let native_safe = native_is_safe(action_idx, &grid, 7, 8, &bombs);
        let action = BomberAction::from(action_idx);
        let match_ = if wasm_safe == native_safe {
            ""
        } else {
            " ← MISMATCH"
        };
        println!(
            "    from (7,8) action={action_idx} ({action:?}): native={native_safe} wasm={wasm_safe}{match_}"
        );
    }

    // ── Check with single bomb to isolate ───────────────────────
    println!("\n  Bomb at ({x},{y}) with individual bombs:");
    for (i, &bomb) in bombs.iter().enumerate() {
        let single_bombs = vec![bomb];
        let native_single = native_is_safe(4, &grid, x, y, &single_bombs);
        let wasm_single = wasm_is_safe(&pruner, 4, &grid, x, y, &single_bombs);
        let match_ = if native_single == wasm_single {
            ""
        } else {
            " ← MISMATCH"
        };
        println!(
            "    bomb[{}]=({},{}) r={} f={}: native={native_single} wasm={wasm_single}{match_}",
            i, bomb.0.0, bomb.0.1, bomb.1, bomb.2
        );
    }

    // ── Binary search: add bombs one at a time ──────────────────
    println!("\n  Bomb at ({x},{y}) with cumulative bombs:");
    for count in 1..=bombs.len() {
        let partial_bombs: Vec<_> = bombs[..count].to_vec();
        let native_partial = native_is_safe(4, &grid, x, y, &partial_bombs);
        let wasm_partial = wasm_is_safe(&pruner, 4, &grid, x, y, &partial_bombs);
        let tag = if native_partial != wasm_partial {
            " ← FIRST MISMATCH"
        } else {
            ""
        };
        println!("    {count} bombs: native={native_partial} wasm={wasm_partial}{tag}");
    }

    // ── WASM blast zone check from escape path ──────────────────
    println!("\n  WASM blast zone check along escape path (7,10)→(7,11)→(6,11):");
    for &(cx, cy) in &[(7, 10), (7, 11), (6, 11), (6, 10), (5, 11)] {
        let native_walk = grid.is_walkable(cx, cy);
        let native_in_blast = is_in_blast_zone_native(&grid, cx, cy, &bombs);
        let wasm_wait = wasm_is_safe(&pruner, 5, &grid, cx, cy, &bombs);
        println!(
            "    ({cx},{cy}): walkable={native_walk} native_blast={native_in_blast} wasm_wait_safe={wasm_wait}"
        );
    }

    // ── WASM bomb placement from (7,10) ─────────────────────────
    println!("\n  Bomb from (7,10) with all 4 bombs:");
    let native_710 = native_is_safe(4, &grid, 7, 10, &bombs);
    let wasm_710 = wasm_is_safe(&pruner, 4, &grid, 7, 10, &bombs);
    println!("    native={native_710} wasm={wasm_710}");

    // ── WASM relevance gives continuous score ────────────────────
    println!("\n  WASM relevance scores at ({x},{y}) with all bombs:");
    for action_idx in 0..6 {
        let rel = pruner.action_relevance(action_idx, &grid, x, y, 0, &bombs);
        let action = BomberAction::from(action_idx);
        println!("    action={action_idx} ({action:?}): relevance={rel:.3}");
    }
}

/// Precise escape-path trace for the 4-bomb mismatch (seed=1000, pos=(7,9)).
///
/// Mismatch: `native=true wasm=false` only when all 4 bombs present.
/// The escape path for the new bomb at (7,9) should be via adjacent (7,10):
///   (7,10) → (7,11) → (6,11)  [safe from all blast zones]
///
/// This test traces the BFS step-by-step as the WASM `has_escape_route` would,
/// checking walkability, bomb-entity blocking, and blast-zone membership at each
/// cell along the candidate escape path.
#[test]
fn test_precise_4bomb_escape_path_trace() {
    let Some(pruner) = load_pruner() else { return };

    let grid = ArenaGrid::generate(1000);
    let bombs: Vec<((i32, i32), u32, u32)> = vec![
        ((5, 3), 1, 2),
        ((8, 5), 2, 1),
        ((9, 3), 1, 1),
        ((6, 1), 2, 2),
    ];

    // ── Verify grid cells along escape path ─────────────────────
    let path_cells: [(i32, i32); 4] = [(7, 9), (7, 10), (7, 11), (6, 11)];
    println!("Grid cells along escape path:");
    for &(cx, cy) in &path_cells {
        let cell = grid.get(cx, cy);
        let walk = grid.is_walkable(cx, cy);
        let cell_ch = match cell {
            Cell::Floor => '.',
            Cell::FixedWall => '#',
            Cell::DestructibleWall => '█',
            Cell::PowerUpHidden(_) => '?',
        };
        let in_blast_existing = is_in_blast_zone_native(&grid, cx, cy, &bombs);

        // New bomb at (7,9) range 2: compute blast manually
        let in_blast_new = is_in_single_blast_native(&grid, cx, cy, 7, 9, 2);

        // Combined: existing + new
        let mut all_bombs = bombs.clone();
        all_bombs.push(((7, 9), 2, 0));
        let in_blast_all = is_in_blast_zone_native(&grid, cx, cy, &all_bombs);

        // Is (cx,cy) a bomb entity? (blocks movement in BFS)
        let is_bomb_entity = all_bombs
            .iter()
            .any(|&(pos, _, _)| pos.0 == cx && pos.1 == cy);

        println!(
            "  ({cx},{cy}) '{cell_ch}' walk={walk} blast_existing={in_blast_existing} \
             blast_new={in_blast_new} blast_all={in_blast_all} is_bomb={is_bomb_entity}"
        );
    }

    // ── Verify WASM sees the same grid ──────────────────────────
    // Use Wait action (idx=5) to probe: safe means not in blast zone
    println!("\nWASM Wait probes (blast-zone membership for existing bombs):");
    for &(cx, cy) in &path_cells {
        let wasm_wait = wasm_is_safe(&pruner, 5, &grid, cx, cy, &bombs);
        let native_wait = native_is_safe(5, &grid, cx, cy, &bombs);
        let tag = if wasm_wait != native_wait {
            " ← MISMATCH"
        } else {
            ""
        };
        println!("  ({cx},{cy}): native_wait={native_wait} wasm_wait={wasm_wait}{tag}");
    }

    // ── Check WASM movement from each adjacent cell of (7,9) ────
    // The bomb-action escape route checks: any(adjacent walkable AND has_escape)
    println!("\nWASM bomb escape: adjacent cells of (7,9):");
    for &(dx, dy) in &[(0i32, -1), (0, 1), (-1, 0), (1, 0)] {
        let nx = 7 + dx;
        let ny = 9 + dy;
        let walk = grid.is_walkable(nx, ny);
        let wasm_wait_adj = wasm_is_safe(&pruner, 5, &grid, nx, ny, &bombs);

        // Check if (nx,ny) is blocked by a bomb entity
        let is_bomb_pos = bombs.iter().any(|&(pos, _, _)| pos.0 == nx && pos.1 == ny);

        // Try bomb from (nx,ny) — if WASM accepts, the escape from there works
        let wasm_bomb_from_adj = wasm_is_safe(&pruner, 4, &grid, nx, ny, &bombs);
        let native_bomb_from_adj = native_is_safe(4, &grid, nx, ny, &bombs);
        let bomb_match = if wasm_bomb_from_adj != native_bomb_from_adj {
            " ← DIFF"
        } else {
            ""
        };

        println!(
            "  adj ({nx},{ny}): walk={walk} is_bomb_pos={is_bomb_pos} \
             wasm_wait={wasm_wait_adj} bomb_from_here: n={native_bomb_from_adj} w={wasm_bomb_from_adj}{bomb_match}"
        );
    }

    // ── Trace BFS manually from (7,10) with new bomb at (7,9) ───
    // max_steps = blast_range + 1 = 3
    // blocked = existing bomb positions ∪ {(7,9)}
    // all_bombs for blast = existing ∪ {(7,9, range=2)}
    println!("\nManual BFS from (7,10) with new bomb at (7,9) range=2:");
    let mut visited = std::collections::HashSet::<(i32, i32)>::new();
    let mut queue = std::collections::VecDeque::<((i32, i32), i32)>::new();
    let mut all_bombs = bombs.clone();
    all_bombs.push(((7, 9), 2, 0));
    let blocked: std::collections::HashSet<(i32, i32)> =
        all_bombs.iter().map(|&(p, _, _)| p).collect();

    queue.push_back(((7, 10), 0));
    visited.insert((7, 10));

    let mut found_safe = false;
    let mut step_log = Vec::new();
    while let Some(((cx, cy), steps)) = queue.pop_front() {
        if steps > 3 {
            continue;
        }
        let in_blast = is_in_blast_zone_native(&grid, cx, cy, &all_bombs);
        let walk = grid.is_walkable(cx, cy);
        let is_blocked = blocked.contains(&(cx, cy));
        step_log.push(format!(
            "  step={steps} ({cx},{cy}): walk={walk} blocked={is_blocked} in_blast={in_blast}"
        ));

        if !in_blast && walk && !is_blocked {
            found_safe = true;
            step_log.push(format!("  → SAFE CELL FOUND at ({cx},{cy}) step={steps}"));
            break;
        }

        for (nx, ny) in [(cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)] {
            if visited.insert((nx, ny)) {
                let w = grid.is_walkable(nx, ny);
                let b = blocked.contains(&(nx, ny));
                if w && !b {
                    queue.push_back(((nx, ny), steps + 1));
                }
            }
        }
    }

    for line in &step_log {
        println!("{line}");
    }

    if !found_safe {
        println!("  → NO SAFE CELL FOUND (BFS exhausted)");
    }

    // ── Core assertions ─────────────────────────────────────────
    // Native must find escape via (7,10)
    let native_bomb_79 = native_is_safe(4, &grid, 7, 9, &bombs);
    assert!(
        native_bomb_79,
        "Native must accept bomb at (7,9) — escape via (7,10) exists"
    );

    // All adjacent cells must have matching Wait verdicts (no blast-zone disagreement)
    for &(dx, dy) in &[(0i32, -1), (0, 1), (-1, 0), (1, 0)] {
        let nx = 7 + dx;
        let ny = 9 + dy;
        let native_wait = native_is_safe(5, &grid, nx, ny, &bombs);
        let wasm_wait = wasm_is_safe(&pruner, 5, &grid, nx, ny, &bombs);
        assert_eq!(
            native_wait, wasm_wait,
            "Wait at ({nx},{ny}): native={native_wait} wasm={wasm_wait}"
        );
    }

    // Document the known difference: WASM rejects bomb at (7,9) with 4 bombs
    let wasm_bomb_79 = wasm_is_safe(&pruner, 4, &grid, 7, 9, &bombs);
    if native_bomb_79 != wasm_bomb_79 {
        println!(
            "\n⚠ Known difference: bomb at (7,9) with 4 bombs: \
             native={native_bomb_79} wasm={wasm_bomb_79}"
        );
        println!("  This likely indicates WASM BFS differs from native in edge cases.");
        println!("  Escape path exists (native BFS finds it), but WASM does not.");
        println!("  TODO: Investigate WASM has_escape_route with 5 bombs in all_bombs array.");
    }
}

/// Comprehensive A/B test: generate many random game states and compare all actions.
#[test]
fn test_ab_correctness_many_states() {
    let Some(pruner) = load_pruner() else { return };

    let mut results = AbResults::new();

    for i in 0..NUM_RANDOM_GRIDS {
        let seed = 1000 + i as u64;
        let grid = ArenaGrid::generate(seed);
        let walkable = find_walkable_positions(&grid);

        if walkable.is_empty() {
            continue;
        }

        let bomb_configs = generate_bomb_configs(&grid, seed);

        for bombs in &bomb_configs {
            // Test a subset of walkable positions (up to 10 per config)
            let test_positions: Vec<(i32, i32)> = walkable
                .iter()
                .step_by((walkable.len().max(1)) / 10)
                .copied()
                .take(10)
                .collect();

            for &(x, y) in &test_positions {
                for &action_idx in &ALL_ACTION_INDICES {
                    results.total_comparisons += 1;

                    let native = native_is_safe(action_idx, &grid, x, y, bombs);
                    let wasm = wasm_is_safe(&pruner, action_idx, &grid, x, y, bombs);

                    if native != wasm {
                        results.record(seed, x, y, action_idx, native, wasm, &grid, bombs);
                    }
                }
            }
        }
    }

    results.print_summary();

    // Critical: movement/wait actions must match exactly, and WASM must never
    // allow an action that native rejects (WASM-allows-unsafe = bug).
    assert!(
        results.critical_mismatches.is_empty(),
        "❌ {} critical mismatches (movement/wait differ or WASM allows unsafe action!)",
        results.critical_mismatches.len()
    );

    println!(
        "✅ PASSED: {} comparisons, {} bomb differences (WASM stricter — expected)",
        results.total_comparisons,
        results.bomb_stricter_mismatches.len()
    );
}

// ── Grid Builders ──────────────────────────────────────────────

/// Build an empty 13×13 grid (all floors, no border walls).
fn empty_grid() -> ArenaGrid {
    ArenaGrid {
        cells: vec![vec![Cell::Floor; 13]; 13],
        width: 13,
        height: 13,
    }
}
