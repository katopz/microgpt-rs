//! WASM bomber validator game state serialization.
//!
//! Serializes the 13×13 arena grid, player position, and active bombs
//! into a u32-aligned token buffer for the WASM bomber validator ABI.
//!
//! # ABI Contract
//!
//! The WASM SDK's `read_parent_tokens(ptr, len)` reads `len × 4` bytes from
//! memory and converts each 4-byte chunk (little-endian u32) into a `usize`
//! token. So every value must be stored as a 4-byte u32, and the `len`
//! parameter passed to `is_valid` must be the **token count** (not byte count).
//!
//! # Token Buffer Layout
//!
//! | Token Index | Value      | Description                                        |
//! |-------------|------------|----------------------------------------------------|
//! | 0..168      | cell byte  | grid: 13×13 cells, 1 token each (row-major)        |
//! | 169         | player_x   | player X coordinate (u8)                           |
//! | 170         | player_y   | player Y coordinate (u8)                           |
//! | 171         | player_id  | player ID (u8)                                     |
//! | 172         | bomb_count | number of bombs N (u8, max 16)                     |
//! | 173..       | N×4 tokens | bombs: N × (x, y, range, fuse)                     |
//!
//! # Cell → Token Mapping
//!
//! | Cell Variant            | Token Value |
//! |-------------------------|-------------|
//! | Floor                   | 0           |
//! | FixedWall               | 1           |
//! | DestructibleWall        | 2           |
//! | PowerUpHidden(_)        | 3           |

use super::{ARENA_H, ARENA_W, ArenaGrid, Cell};

/// Maximum number of bombs that can be serialized.
const MAX_BOMBS: usize = 16;

/// Number of grid tokens (13×13 = 169).
const GRID_TOKENS: usize = ARENA_W * ARENA_H;

/// Token index: player_x.
#[allow(dead_code)]
const OFF_PLAYER_X: usize = GRID_TOKENS;

/// Token index: player_y.
#[allow(dead_code)]
const OFF_PLAYER_Y: usize = GRID_TOKENS + 1;

/// Token index: player_id.
#[allow(dead_code)]
const OFF_PLAYER_ID: usize = GRID_TOKENS + 2;

/// Token index: bomb_count.
#[allow(dead_code)]
const OFF_BOMB_COUNT: usize = GRID_TOKENS + 3;

/// Token index: first bomb (x, y, range, fuse × N).
#[allow(dead_code)]
const OFF_BOMBS: usize = GRID_TOKENS + 4;

/// Header size in tokens: grid + player_x + player_y + player_id + bomb_count.
const HEADER_TOKENS: usize = OFF_BOMBS; // 173

/// Bytes per u32 token.
const BYTES_PER_TOKEN: usize = 4;

/// Cell token value for [`Cell::Floor`].
const CELL_FLOOR: u8 = 0;

/// Cell token value for [`Cell::FixedWall`].
const CELL_FIXED_WALL: u8 = 1;

/// Cell token value for [`Cell::DestructibleWall`].
const CELL_DESTRUCTIBLE: u8 = 2;

/// Cell token value for [`Cell::PowerUpHidden`].
const CELL_POWERUP: u8 = 3;

/// Convert a [`Cell`] to its token value for the WASM ABI.
fn cell_to_token(cell: &Cell) -> u8 {
    match cell {
        Cell::Floor => CELL_FLOOR,
        Cell::FixedWall => CELL_FIXED_WALL,
        Cell::DestructibleWall => CELL_DESTRUCTIBLE,
        Cell::PowerUpHidden(_) => CELL_POWERUP,
    }
}

/// Clamp an `i32` coordinate to `u8` range [0, 255].
fn clamp_to_u8(val: i32) -> u8 {
    val.clamp(0, u8::MAX as i32) as u8
}

/// Append a u8 value as a u32 LE token (4 bytes) to the buffer.
fn push_token(buf: &mut Vec<u8>, value: u8) {
    let token = value as u32;
    buf.extend_from_slice(&token.to_le_bytes());
}

/// Serialize game state as u32-aligned token buffer for WASM bomber validator ABI.
///
/// Each value (grid cell, coordinate, bomb field) is encoded as a 4-byte
/// little-endian u32 token, matching the WASM SDK's `read_parent_tokens`
/// which reads `len × 4` bytes and converts each chunk to a u32.
///
/// Returns `(byte_buffer, token_count)` where:
/// - `byte_buffer` contains the serialized state (`token_count × 4` bytes)
/// - `token_count` should be passed as the `len` parameter to WASM `is_valid`
///
/// Player coordinates are clamped to `u8` range. Bomb count is capped at 16.
///
/// # Panics
///
/// Does not panic — out-of-bounds coordinates are clamped, bomb count is capped.
pub fn serialize_game_state(
    grid: &ArenaGrid,
    player_x: i32,
    player_y: i32,
    player_id: u8,
    bombs: &[((i32, i32), u32, u32)],
) -> (Vec<u8>, u32) {
    let bomb_count = bombs.len().min(MAX_BOMBS);
    let token_count = (HEADER_TOKENS + bomb_count * 4) as u32;
    let total_bytes = token_count as usize * BYTES_PER_TOKEN;
    let mut buf = Vec::with_capacity(total_bytes);

    // Grid: 13×13 cells, each as one u32 token (row-major: y outer, x inner)
    for y in 0..ARENA_H {
        for x in 0..ARENA_W {
            push_token(&mut buf, cell_to_token(&grid.cells[y][x]));
        }
    }

    // Player position (clamped to u8)
    push_token(&mut buf, clamp_to_u8(player_x));
    push_token(&mut buf, clamp_to_u8(player_y));

    // Player ID
    push_token(&mut buf, player_id);

    // Bomb count (capped at MAX_BOMBS)
    push_token(&mut buf, bomb_count as u8);

    // Bombs: each bomb is 4 tokens (x, y, range, fuse)
    for &((bx, by), blast_range, fuse) in &bombs[..bomb_count] {
        push_token(&mut buf, clamp_to_u8(bx));
        push_token(&mut buf, clamp_to_u8(by));
        push_token(&mut buf, blast_range as u8);
        push_token(&mut buf, fuse as u8);
    }

    debug_assert_eq!(buf.len(), total_bytes);
    (buf, token_count)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::bomber::PowerUpKind;

    /// Read a u32 LE token from the buffer at the given token index.
    fn read_token(buf: &[u8], token_idx: usize) -> u32 {
        let offset = token_idx * BYTES_PER_TOKEN;
        u32::from_le_bytes([
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
        ])
    }

    /// Build an empty grid (all [`Cell::Floor`]).
    fn empty_grid() -> ArenaGrid {
        ArenaGrid {
            cells: vec![vec![Cell::Floor; ARENA_W]; ARENA_H],
            width: ARENA_W,
            height: ARENA_H,
        }
    }

    /// Build a grid full of [`Cell::FixedWall`].
    fn full_wall_grid() -> ArenaGrid {
        ArenaGrid {
            cells: vec![vec![Cell::FixedWall; ARENA_W]; ARENA_H],
            width: ARENA_W,
            height: ARENA_H,
        }
    }

    #[test]
    fn empty_grid_no_bombs_player_at_1_1() {
        let grid = empty_grid();
        let (buf, token_count) = serialize_game_state(&grid, 1, 1, 0, &[]);

        // 173 tokens × 4 bytes = 692 bytes
        assert_eq!(token_count, 173);
        assert_eq!(buf.len(), 173 * BYTES_PER_TOKEN);

        // Grid should be all zeros (Floor)
        for i in 0..GRID_TOKENS {
            assert_eq!(read_token(&buf, i), 0, "grid token {i} should be Floor(0)");
        }

        // Player at (1, 1)
        assert_eq!(read_token(&buf, OFF_PLAYER_X), 1);
        assert_eq!(read_token(&buf, OFF_PLAYER_Y), 1);
        assert_eq!(read_token(&buf, OFF_PLAYER_ID), 0);
        assert_eq!(read_token(&buf, OFF_BOMB_COUNT), 0);
    }

    #[test]
    fn test_full_wall_grid() {
        let grid = full_wall_grid();
        let (buf, token_count) = serialize_game_state(&grid, 5, 5, 2, &[]);

        assert_eq!(token_count, 173);
        assert_eq!(buf.len(), 173 * BYTES_PER_TOKEN);

        // Grid should be all 1s (FixedWall)
        for i in 0..GRID_TOKENS {
            assert_eq!(
                read_token(&buf, i),
                1,
                "grid token {i} should be FixedWall(1)"
            );
        }

        assert_eq!(read_token(&buf, OFF_PLAYER_X), 5);
        assert_eq!(read_token(&buf, OFF_PLAYER_Y), 5);
        assert_eq!(read_token(&buf, OFF_PLAYER_ID), 2);
        assert_eq!(read_token(&buf, OFF_BOMB_COUNT), 0);
    }

    #[test]
    fn grid_with_all_cell_types() {
        let mut grid = empty_grid();
        grid.cells[0][0] = Cell::Floor;
        grid.cells[1][0] = Cell::FixedWall;
        grid.cells[2][0] = Cell::DestructibleWall;
        grid.cells[3][0] = Cell::PowerUpHidden(PowerUpKind::BombUp);
        grid.cells[4][0] = Cell::PowerUpHidden(PowerUpKind::FireUp);
        grid.cells[5][0] = Cell::PowerUpHidden(PowerUpKind::SpeedUp);

        let (buf, _) = serialize_game_state(&grid, 0, 0, 0, &[]);

        // Token indices for grid[y][x] = y * ARENA_W + x
        assert_eq!(read_token(&buf, 0 * ARENA_W + 0), 0); // Floor at (0,0)
        assert_eq!(read_token(&buf, 1 * ARENA_W + 0), 1); // FixedWall at (0,1)
        assert_eq!(read_token(&buf, 2 * ARENA_W + 0), 2); // DestructibleWall at (0,2)
        assert_eq!(read_token(&buf, 3 * ARENA_W + 0), 3); // PowerUpHidden(BombUp) at (0,3)
        assert_eq!(read_token(&buf, 4 * ARENA_W + 0), 3); // PowerUpHidden(FireUp) at (0,4)
        assert_eq!(read_token(&buf, 5 * ARENA_W + 0), 3); // PowerUpHidden(SpeedUp) at (0,5)
    }

    #[test]
    fn multiple_bombs() {
        let grid = empty_grid();
        let bombs: [((i32, i32), u32, u32); 3] = [((3, 4), 2, 3), ((5, 6), 3, 1), ((7, 8), 1, 4)];

        let (buf, token_count) = serialize_game_state(&grid, 1, 1, 0, &bombs);

        // 173 header + 3×4 bomb tokens = 185 tokens × 4 = 740 bytes
        assert_eq!(token_count, 185);
        assert_eq!(buf.len(), 185 * BYTES_PER_TOKEN);
        assert_eq!(read_token(&buf, OFF_BOMB_COUNT), 3);

        // First bomb: (3, 4, 2, 3)
        assert_eq!(read_token(&buf, OFF_BOMBS), 3); // x
        assert_eq!(read_token(&buf, OFF_BOMBS + 1), 4); // y
        assert_eq!(read_token(&buf, OFF_BOMBS + 2), 2); // range
        assert_eq!(read_token(&buf, OFF_BOMBS + 3), 3); // fuse

        // Second bomb: (5, 6, 3, 1)
        assert_eq!(read_token(&buf, OFF_BOMBS + 4), 5); // x
        assert_eq!(read_token(&buf, OFF_BOMBS + 5), 6); // y
        assert_eq!(read_token(&buf, OFF_BOMBS + 6), 3); // range
        assert_eq!(read_token(&buf, OFF_BOMBS + 7), 1); // fuse

        // Third bomb: (7, 8, 1, 4)
        assert_eq!(read_token(&buf, OFF_BOMBS + 8), 7); // x
        assert_eq!(read_token(&buf, OFF_BOMBS + 9), 8); // y
        assert_eq!(read_token(&buf, OFF_BOMBS + 10), 1); // range
        assert_eq!(read_token(&buf, OFF_BOMBS + 11), 4); // fuse
    }

    #[test]
    fn max_bombs_16() {
        let grid = empty_grid();
        let bombs: Vec<((i32, i32), u32, u32)> = (0..20).map(|i| ((i, i), 2, 4)).collect();

        let (buf, token_count) = serialize_game_state(&grid, 0, 0, 0, &bombs);

        // Should cap at 16 bombs: 173 + 16×4 = 237 tokens × 4 = 948 bytes
        assert_eq!(token_count, 237);
        assert_eq!(buf.len(), 237 * BYTES_PER_TOKEN);
        assert_eq!(read_token(&buf, OFF_BOMB_COUNT), 16); // bomb_count capped

        // First bomb
        assert_eq!(read_token(&buf, OFF_BOMBS), 0); // x
        assert_eq!(read_token(&buf, OFF_BOMBS + 1), 0); // y
        assert_eq!(read_token(&buf, OFF_BOMBS + 2), 2); // range
        assert_eq!(read_token(&buf, OFF_BOMBS + 3), 4); // fuse

        // 16th bomb (last serialized)
        let last_base = OFF_BOMBS + 15 * 4;
        assert_eq!(read_token(&buf, last_base), 15); // x
        assert_eq!(read_token(&buf, last_base + 1), 15); // y
        assert_eq!(read_token(&buf, last_base + 2), 2); // range
        assert_eq!(read_token(&buf, last_base + 3), 4); // fuse
    }

    #[test]
    fn out_of_bounds_player_coordinates_clamped() {
        let grid = empty_grid();

        // Negative coordinates → clamped to 0
        let (buf, _) = serialize_game_state(&grid, -5, -10, 0, &[]);
        assert_eq!(read_token(&buf, OFF_PLAYER_X), 0);
        assert_eq!(read_token(&buf, OFF_PLAYER_Y), 0);

        // Beyond u8 max → clamped to 255
        let (buf, _) = serialize_game_state(&grid, 300, 500, 0, &[]);
        assert_eq!(read_token(&buf, OFF_PLAYER_X), 255);
        assert_eq!(read_token(&buf, OFF_PLAYER_Y), 255);

        // Zero edge
        let (buf, _) = serialize_game_state(&grid, 0, 0, 0, &[]);
        assert_eq!(read_token(&buf, OFF_PLAYER_X), 0);
        assert_eq!(read_token(&buf, OFF_PLAYER_Y), 0);

        // u8::MAX edge
        let (buf, _) = serialize_game_state(&grid, 255, 255, 0, &[]);
        assert_eq!(read_token(&buf, OFF_PLAYER_X), 255);
        assert_eq!(read_token(&buf, OFF_PLAYER_Y), 255);
    }

    #[test]
    fn token_count_matches_buffer_size() {
        let grid = empty_grid();

        // No bombs
        let (buf, token_count) = serialize_game_state(&grid, 0, 0, 0, &[]);
        assert_eq!(buf.len(), token_count as usize * BYTES_PER_TOKEN);

        // 1 bomb
        let (buf, token_count) = serialize_game_state(&grid, 0, 0, 0, &[((1, 1), 2, 3)]);
        assert_eq!(buf.len(), token_count as usize * BYTES_PER_TOKEN);

        // 16 bombs
        let bombs: Vec<((i32, i32), u32, u32)> = (0..16).map(|i| ((i, i), 2, 4)).collect();
        let (buf, token_count) = serialize_game_state(&grid, 0, 0, 0, &bombs);
        assert_eq!(buf.len(), token_count as usize * BYTES_PER_TOKEN);
    }
}
