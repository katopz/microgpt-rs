//! Go board state — port from `alpha_go/go.py:FastGoBoard` + `go_game.h:GoBoard`.
//!
//! Implements the [`GameState`] trait for MCTS compatibility.
//! Uses simple ko (not positional superko) and Tromp-Taylor area scoring.
//!
//! ## Board Layout
//!
//! Flat array `Vec<GoCell>` with `row * size + col` indexing (matches C++ `GoBoard`).
//! Pre-computed neighbor offsets avoid per-access bounds checking.

use std::collections::HashSet;
use std::fmt;

use super::types::{GoAction, GoCell};
use crate::pruners::game_state::{GameState, StateHeuristic};

/// Default komi: 7.5 (AI standard, matches `go_game.h:GoBoard::KOMI`).
pub const DEFAULT_KOMI: f32 = 7.5;

// ── GoState ────────────────────────────────────────────────────

/// Lightweight Go state snapshot. Port from `go.py:FastGoBoard` + `go_game.h:GoBoard`.
///
/// 9×9: 81 cells ≈ 500 bytes total. Clone < 100ns.
/// 19×19: 361 cells ≈ 1.8KB total. Clone < 500ns.
#[derive(Clone)]
pub struct GoState {
    /// Board cells: flat array, `row * size + col`. 0=empty, 1=black, 2=white.
    pub board: Vec<GoCell>,
    /// Board dimension (9, 13, or 19).
    pub size: usize,
    /// Current player to move.
    pub to_play: GoCell,
    /// Flat index of forbidden recapture (simple ko), or `None`.
    pub ko_point: Option<usize>,
    /// Consecutive pass count. Game ends at ≥ 2.
    pub consecutive_passes: u8,
    /// Total moves played (including passes).
    pub move_count: u32,
    /// Komi compensation for White (default 7.5).
    pub komi: f32,
    /// Stones captured BY Black (removed from White).
    pub captured_black: u32,
    /// Stones captured BY White (removed from Black).
    pub captured_white: u32,
    /// Pre-computed neighbor flat indices per cell (2–4 entries each).
    neighbor_offsets: Vec<Vec<usize>>,
}

impl GoState {
    /// Create a new empty board of the given size.
    ///
    /// Default komi = 7.5, Black plays first.
    /// Panics if `size` is 0.
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "Go board size must be > 0");
        Self {
            board: vec![GoCell::Empty; size * size],
            size,
            to_play: GoCell::Black,
            ko_point: None,
            consecutive_passes: 0,
            move_count: 0,
            komi: DEFAULT_KOMI,
            captured_black: 0,
            captured_white: 0,
            neighbor_offsets: Self::init_neighbors(size),
        }
    }

    /// Create a board with custom komi.
    pub fn with_komi(size: usize, komi: f32) -> Self {
        let mut state = Self::new(size);
        state.komi = komi;
        state
    }

    /// Create a board from an existing position (for testing / API sync).
    ///
    /// `board` is a flat array of `i8` values (0=empty, 1=black, 2=white).
    /// Panics if `board.len() != size * size`.
    pub fn from_flat(size: usize, board: &[i8], to_play: GoCell) -> Self {
        assert_eq!(board.len(), size * size, "Board array size mismatch");
        let mut state = Self::new(size);
        for (i, &v) in board.iter().enumerate() {
            state.board[i] = GoCell::from_i8(v).unwrap_or(GoCell::Empty);
        }
        state.to_play = to_play;
        state
    }

    // ── Neighbor Cache ─────────────────────────────────────────

    /// Pre-compute neighbor flat indices for every cell.
    ///
    /// Corner cells have 2 neighbors, edge cells have 3, interior cells have 4.
    fn init_neighbors(size: usize) -> Vec<Vec<usize>> {
        let mut neighbors = vec![Vec::with_capacity(4); size * size];
        for r in 0..size {
            for c in 0..size {
                let idx = r * size + c;
                for (dr, dc) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr >= 0 && nr < size as i32 && nc >= 0 && nc < size as i32 {
                        neighbors[idx].push(nr as usize * size + nc as usize);
                    }
                }
            }
        }
        neighbors
    }

    /// Get pre-computed neighbor flat indices for a cell.
    #[inline]
    fn neighbors(&self, idx: usize) -> &[usize] {
        &self.neighbor_offsets[idx]
    }

    /// Convert (row, col) to flat index.
    #[inline]
    pub fn flat_index(&self, row: usize, col: usize) -> usize {
        row * self.size + col
    }

    /// Convert flat index to (row, col).
    #[inline]
    pub fn row_col(&self, flat: usize) -> (usize, usize) {
        (flat / self.size, flat % self.size)
    }

    /// Read cell at (row, col).
    #[inline]
    pub fn at(&self, row: usize, col: usize) -> GoCell {
        self.board[self.flat_index(row, col)]
    }

    // ── Core Queries ───────────────────────────────────────────

    /// BFS flood fill to find a connected group and its liberties.
    ///
    /// Port from `go.py:FastGoBoard._get_group_and_liberties`.
    /// Returns `(group_indices, liberty_indices)`. Both empty if `board[idx]` is Empty.
    fn get_group_and_liberties(&self, idx: usize) -> (Vec<usize>, Vec<usize>) {
        let color = match self.board[idx] {
            GoCell::Empty => return (Vec::new(), Vec::new()),
            c => c,
        };

        let mut group = Vec::new();
        let mut liberties = Vec::new();
        let mut visited = vec![false; self.size * self.size];
        let mut stack = vec![idx];

        while let Some(pos) = stack.pop() {
            if visited[pos] {
                continue;
            }
            visited[pos] = true;

            match self.board[pos] {
                c if c == color => {
                    group.push(pos);
                    for &n in self.neighbors(pos) {
                        if !visited[n] {
                            stack.push(n);
                        }
                    }
                }
                GoCell::Empty => {
                    liberties.push(pos);
                }
                _ => {} // Opponent color — boundary, don't follow
            }
        }

        (group, liberties)
    }

    /// Check if placing `color` at `idx` would be suicide.
    ///
    /// A move is suicide if the resulting group has 0 liberties AND no captures.
    /// Uses analytical checks (no temporary board mutation).
    fn would_be_suicide(&self, idx: usize, color: GoCell) -> bool {
        let opponent = color.opponent();

        for &n in self.neighbors(idx) {
            match self.board[n] {
                GoCell::Empty => return false, // Has liberty → not suicide
                c if c == color => {
                    // Same-color neighbor: if its group has a liberty OTHER than idx → not suicide
                    let (_, liberties) = self.get_group_and_liberties(n);
                    if liberties.iter().any(|&l| l != idx) {
                        return false;
                    }
                }
                c if c == opponent => {
                    // Opponent neighbor: if its group's ONLY liberty is idx → we capture → not suicide
                    let (_, liberties) = self.get_group_and_liberties(n);
                    if liberties.len() == 1 && liberties[0] == idx {
                        return false;
                    }
                }
                _ => unreachable!(),
            }
        }

        // No liberty, no capture → suicide
        true
    }

    /// Check if placing at (row, col) is legal for the current player.
    ///
    /// Order: bounds → occupied → ko → suicide.
    /// Port from `go.py:FastGoBoard.is_legal`.
    pub fn is_legal(&self, row: usize, col: usize) -> bool {
        if row >= self.size || col >= self.size {
            return false;
        }

        let idx = self.flat_index(row, col);

        if self.board[idx] != GoCell::Empty {
            return false;
        }

        if self.ko_point == Some(idx) {
            return false;
        }

        if self.would_be_suicide(idx, self.to_play) {
            return false;
        }

        true
    }

    /// Get all legal moves as (row, col) pairs.
    ///
    /// Does NOT include pass — use [`GoAction::Pass`] separately.
    pub fn legal_moves(&self) -> Vec<(usize, usize)> {
        let mut moves = Vec::new();
        for r in 0..self.size {
            for c in 0..self.size {
                if self.is_legal(r, c) {
                    moves.push((r, c));
                }
            }
        }
        moves
    }

    /// Count legal moves without allocating a vec.
    pub fn legal_move_count(&self) -> usize {
        let mut count = 0;
        for r in 0..self.size {
            for c in 0..self.size {
                if self.is_legal(r, c) {
                    count += 1;
                }
            }
        }
        count
    }

    // ── Mutating Operations ────────────────────────────────────

    /// Place a stone at (row, col). Returns `false` if illegal (no state change).
    ///
    /// Port from `go.py:FastGoBoard.play`. Handles:
    /// - Stone placement
    /// - Capture resolution (remove opponent groups with 0 liberties)
    /// - Simple ko detection
    /// - Turn switching
    pub fn play_move(&mut self, row: usize, col: usize) -> bool {
        if !self.is_legal(row, col) {
            return false;
        }

        let idx = self.flat_index(row, col);
        let color = self.to_play;
        let opponent = color.opponent();

        // Place stone
        self.board[idx] = color;
        self.consecutive_passes = 0;
        self.move_count += 1;

        // Resolve captures: collect groups first, then remove (avoids borrow conflict)
        let mut captured_count = 0usize;
        let mut captured_point = None;
        let mut groups_to_remove: Vec<Vec<usize>> = Vec::new();

        for &n in self.neighbors(idx) {
            if self.board[n] == opponent {
                let (group, liberties) = self.get_group_and_liberties(n);
                if liberties.is_empty() {
                    if group.len() == 1 {
                        captured_point = Some(group[0]);
                    }
                    captured_count += group.len();
                    groups_to_remove.push(group);
                }
            }
        }

        for group in &groups_to_remove {
            self.remove_group(group);
        }

        // Simple ko detection:
        // Set ko_point when exactly 1 stone captured AND capturing group has exactly 1 liberty
        if captured_count == 1 {
            if let Some(cp) = captured_point {
                let (_, my_liberties) = self.get_group_and_liberties(idx);
                self.ko_point = if my_liberties.len() == 1 {
                    Some(cp)
                } else {
                    None
                };
            }
        } else {
            self.ko_point = None;
        }

        // Switch player
        self.to_play = opponent;
        true
    }

    /// Pass turn. Increments consecutive passes (game ends at 2).
    /// Clears ko point. Switches player.
    pub fn play_pass(&mut self) {
        self.consecutive_passes += 1;
        self.move_count += 1;
        self.ko_point = None;
        self.to_play = self.to_play.opponent();
    }

    /// Remove a group of stones from the board, returning the count removed.
    /// Updates capture counters.
    fn remove_group(&mut self, group: &[usize]) -> usize {
        for &idx in group {
            self.board[idx] = GoCell::Empty;
        }
        match self.to_play {
            GoCell::Black => self.captured_black += group.len() as u32,
            GoCell::White => self.captured_white += group.len() as u32,
            GoCell::Empty => unreachable!(),
        }
        group.len()
    }

    // ── Scoring ────────────────────────────────────────────────

    /// Tromp-Taylor area scoring. Returns `black_score - white_score` (including komi).
    ///
    /// Positive = Black wins. Negative = White wins.
    /// Port from `go.py:FastGoBoard.score` + `go_game.h:GoBoard::score`.
    pub fn score(&self) -> f32 {
        let mut black_score = 0.0_f32;
        let mut white_score = self.komi;
        let mut counted = vec![false; self.size * self.size];

        for idx in 0..self.size * self.size {
            if counted[idx] {
                continue;
            }

            match self.board[idx] {
                GoCell::Black => {
                    black_score += 1.0;
                    counted[idx] = true;
                }
                GoCell::White => {
                    white_score += 1.0;
                    counted[idx] = true;
                }
                GoCell::Empty => {
                    let (territory, borders) = self.flood_empty(idx, &mut counted);
                    if borders.len() == 1 {
                        if borders.contains(&GoCell::Black) {
                            black_score += territory.len() as f32;
                        } else if borders.contains(&GoCell::White) {
                            white_score += territory.len() as f32;
                        }
                    }
                    // Mixed borders → neutral (dame), no points
                }
            }
        }

        black_score - white_score
    }

    /// Flood-fill an empty region to determine territory ownership.
    ///
    /// Returns `(territory_cells, border_colors)`.
    /// `border_colors` is a subset of `{Black, White}` — empty only if board is entirely empty.
    fn flood_empty(&self, start: usize, counted: &mut [bool]) -> (Vec<usize>, HashSet<GoCell>) {
        let mut territory = Vec::new();
        let mut borders = HashSet::new();
        let mut stack = vec![start];

        while let Some(pos) = stack.pop() {
            if counted[pos] {
                continue;
            }

            match self.board[pos] {
                GoCell::Empty => {
                    counted[pos] = true;
                    territory.push(pos);
                    for &n in self.neighbors(pos) {
                        if !counted[n] {
                            stack.push(n);
                        }
                    }
                }
                color => {
                    borders.insert(color);
                }
            }
        }

        (territory, borders)
    }

    /// Get the winner, if the game is over.
    ///
    /// Returns `None` for draw (jigo — extremely unlikely with fractional komi).
    /// Returns `None` if game is not yet terminal.
    pub fn get_winner(&self) -> Option<GoCell> {
        if !self.is_terminal() {
            return None;
        }
        let s = self.score();
        match s {
            s if s > 0.0 => Some(GoCell::Black),
            s if s < 0.0 => Some(GoCell::White),
            _ => None, // Draw
        }
    }

    /// Is the game over? (Two consecutive passes.)
    pub fn is_terminal(&self) -> bool {
        self.consecutive_passes >= 2
    }

    /// Count stones of a given color on the board.
    pub fn stone_count(&self, color: GoCell) -> usize {
        self.board.iter().filter(|&&c| c == color).count()
    }
}

// ── Display ────────────────────────────────────────────────────

impl fmt::Display for GoState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Column headers
        write!(f, "   ")?;
        for c in 0..self.size {
            write!(f, "{c:2}")?;
        }
        writeln!(f)?;

        // Board rows
        for r in 0..self.size {
            write!(f, "{r:2} ")?;
            for c in 0..self.size {
                let idx = self.flat_index(r, c);
                if self.ko_point == Some(idx) {
                    write!(f, " □")?;
                } else {
                    write!(f, " {}", self.board[idx])?;
                }
            }
            writeln!(f)?;
        }

        // Status line
        let player_name = match self.to_play {
            GoCell::Black => "Black",
            GoCell::White => "White",
            GoCell::Empty => "?",
        };
        write!(
            f,
            "To play: {player_name} | Move: {} | Passes: {} | Komi: {}",
            self.move_count, self.consecutive_passes, self.komi
        )
    }
}

// ── GameState Trait ────────────────────────────────────────────

impl GameState for GoState {
    type Action = GoAction;

    fn available_actions(&self, _player_id: u8) -> Vec<GoAction> {
        // Go is alternating-turn: only current player (to_play) has legal moves.
        // player_id is ignored — MCTS must call this for the correct turn.
        let mut actions: Vec<GoAction> = self
            .legal_moves()
            .into_iter()
            .map(|(r, c)| GoAction::Place(r, c))
            .collect();
        // Pass is always legal
        actions.push(GoAction::Pass);
        actions
    }

    fn advance(&self, action: &GoAction, _player_id: u8) -> Self {
        let mut next = self.clone();
        match action {
            GoAction::Place(row, col) => {
                let ok = next.play_move(*row, *col);
                debug_assert!(ok, "advance() called with illegal move {action}");
            }
            GoAction::Pass => {
                next.play_pass();
            }
        }
        next
    }

    fn is_terminal(&self) -> bool {
        self.consecutive_passes >= 2
    }

    fn reward(&self, player_id: u8) -> f32 {
        // 1.0 = win, 0.5 = draw, 0.0 = loss
        if !self.is_terminal() {
            return 0.5; // Game not over — neutral
        }
        let s = self.score();
        let we_are_black = player_id == 0;
        match s {
            s if s > 0.0 => {
                if we_are_black {
                    1.0
                } else {
                    0.0
                }
            }
            s if s < 0.0 => {
                if we_are_black {
                    0.0
                } else {
                    1.0
                }
            }
            _ => 0.5, // Draw
        }
    }

    fn tick(&self) -> u32 {
        self.move_count
    }
}

// ── GoHeuristic ────────────────────────────────────────────────

/// Pluggable heuristic for evaluating non-terminal Go states.
///
/// Uses a weighted combination of:
/// - Liberty advantage (40%) — total liberties of our groups vs opponent's
/// - Capture delta (30%) — stones we've captured vs stones we've lost
/// - Influence (20%) — empty cells closer to our stones
/// - Center preference (10%) — stones in the center region
///
/// Returns a value in roughly [-1, 1]. Positive = good for `player_id`.
pub struct GoHeuristic;

impl GoHeuristic {
    /// Liberty advantage: (our_total_liberties - opp_total_liberties) / total_cells.
    fn liberty_advantage(&self, state: &GoState, color: GoCell) -> f32 {
        let opponent = color.opponent();
        let total = state.size * state.size;
        let mut our_libs = 0usize;
        let mut opp_libs = 0usize;
        let mut visited = vec![false; total];

        for idx in 0..total {
            if visited[idx] {
                continue;
            }
            if state.board[idx] == GoCell::Empty {
                continue;
            }
            let (group, liberties) = state.get_group_and_liberties(idx);
            for &g in &group {
                visited[g] = true;
            }
            if state.board[idx] == color {
                our_libs += liberties.len();
            } else if state.board[idx] == opponent {
                opp_libs += liberties.len();
            }
        }

        (our_libs as f32 - opp_libs as f32) / total as f32
    }

    /// Capture delta: (our_captures - opp_captures) / total_cells.
    fn capture_delta(&self, state: &GoState, color: GoCell) -> f32 {
        let (our_captures, opp_captures) = match color {
            GoCell::Black => (state.captured_black, state.captured_white),
            GoCell::White => (state.captured_white, state.captured_black),
            GoCell::Empty => unreachable!(),
        };
        let total = state.size * state.size;
        (our_captures as f32 - opp_captures as f32) / total as f32
    }

    /// Influence: count empty cells whose closest stone is ours.
    fn influence(&self, state: &GoState, color: GoCell) -> f32 {
        let opponent = color.opponent();
        let mut our_influence = 0usize;
        let mut total_empty = 0usize;

        for idx in 0..state.size * state.size {
            if state.board[idx] != GoCell::Empty {
                continue;
            }
            total_empty += 1;

            // BFS to find closest stone of each color
            let mut our_dist = usize::MAX;
            let mut opp_dist = usize::MAX;
            let mut visited = vec![false; state.size * state.size];
            let mut queue = std::collections::VecDeque::new();
            queue.push_back((idx, 0usize));
            visited[idx] = true;

            while let Some((pos, dist)) = queue.pop_front() {
                if state.board[pos] == color && dist < our_dist {
                    our_dist = dist;
                } else if state.board[pos] == opponent && dist < opp_dist {
                    opp_dist = dist;
                }
                // Early exit if both found at same distance
                if our_dist <= dist && opp_dist <= dist {
                    break;
                }
                if dist > our_dist.min(opp_dist) {
                    break;
                }
                for &n in state.neighbors(pos) {
                    if !visited[n] {
                        visited[n] = true;
                        queue.push_back((n, dist + 1));
                    }
                }
            }

            if our_dist < opp_dist {
                our_influence += 1;
            }
        }

        if total_empty == 0 {
            return 0.0;
        }
        (our_influence as f32 / total_empty as f32) * 2.0 - 1.0 // Normalize to [-1, 1]
    }

    /// Center preference: count our stones in center third vs opponent's.
    fn center_preference(&self, state: &GoState, color: GoCell) -> f32 {
        let opponent = color.opponent();
        let third = state.size / 3;
        let two_third = state.size - third;
        let mut our_center = 0usize;
        let mut opp_center = 0usize;

        for r in third..two_third {
            for c in third..two_third {
                let idx = state.flat_index(r, c);
                match state.board[idx] {
                    GoCell::Empty => {}
                    c if c == color => our_center += 1,
                    c if c == opponent => opp_center += 1,
                    _ => {}
                }
            }
        }

        let total = (two_third - third) * (two_third - third);
        if total == 0 {
            return 0.0;
        }
        (our_center as f32 - opp_center as f32) / total as f32
    }
}

impl StateHeuristic<GoState> for GoHeuristic {
    fn evaluate(&self, state: &GoState, player_id: u8) -> f32 {
        if state.is_terminal() {
            return state.reward(player_id);
        }
        let color = GoCell::from_player_id(player_id);
        let liberty = self.liberty_advantage(state, color);
        let capture = self.capture_delta(state, color);
        let influence = self.influence(state, color);
        let center = self.center_preference(state, color);
        liberty * 0.4 + capture * 0.3 + influence * 0.2 + center * 0.1
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ───────────────────────────────────────────

    #[test]
    fn new_9x9_initial_state() {
        let state = GoState::new(9);
        assert_eq!(state.size, 9);
        assert_eq!(state.board.len(), 81);
        assert_eq!(state.to_play, GoCell::Black);
        assert_eq!(state.ko_point, None);
        assert_eq!(state.consecutive_passes, 0);
        assert_eq!(state.move_count, 0);
        assert!((state.komi - 7.5).abs() < f32::EPSILON);
        assert!(state.board.iter().all(|&c| c == GoCell::Empty));
    }

    #[test]
    fn legal_moves_empty_9x9() {
        let state = GoState::new(9);
        let moves = state.legal_moves();
        assert_eq!(moves.len(), 81, "Empty 9×9 should have 81 legal moves");
    }

    #[test]
    fn legal_moves_empty_19x19() {
        let state = GoState::new(19);
        assert_eq!(state.legal_move_count(), 361);
    }

    // ── Single Stone ───────────────────────────────────────────

    #[test]
    fn play_single_stone() {
        let mut state = GoState::new(9);
        assert!(state.play_move(4, 4));
        assert_eq!(state.at(4, 4), GoCell::Black);
        assert_eq!(state.to_play, GoCell::White);
        assert_eq!(state.move_count, 1);
        assert_eq!(state.consecutive_passes, 0);
        // 80 remaining empty cells
        assert_eq!(state.legal_move_count(), 80);
    }

    #[test]
    fn turn_alternation() {
        let mut state = GoState::new(9);
        assert_eq!(state.to_play, GoCell::Black);
        state.play_move(0, 0);
        assert_eq!(state.to_play, GoCell::White);
        state.play_move(1, 0);
        assert_eq!(state.to_play, GoCell::Black);
    }

    #[test]
    fn illegal_move_occupied() {
        let mut state = GoState::new(9);
        assert!(state.play_move(4, 4));
        assert!(!state.play_move(4, 4), "Cannot play on occupied cell");
    }

    #[test]
    fn illegal_move_out_of_bounds() {
        let state = GoState::new(9);
        assert!(!state.is_legal(9, 0));
        assert!(!state.is_legal(0, 9));
    }

    // ── Pass ───────────────────────────────────────────────────

    #[test]
    fn pass_switches_turn() {
        let mut state = GoState::new(9);
        state.play_pass();
        assert_eq!(state.to_play, GoCell::White);
        assert_eq!(state.consecutive_passes, 1);
        assert_eq!(state.move_count, 1);
    }

    #[test]
    fn two_passes_end_game() {
        let mut state = GoState::new(9);
        state.play_pass();
        assert!(!state.is_terminal());
        state.play_pass();
        assert!(state.is_terminal());
    }

    #[test]
    fn pass_move_pass_does_not_end() {
        let mut state = GoState::new(9);
        state.play_pass(); // passes=1
        state.play_move(0, 0); // passes=0
        state.play_pass(); // passes=1
        assert!(
            !state.is_terminal(),
            "Non-consecutive passes should not end game"
        );
    }

    #[test]
    fn pass_clears_ko() {
        let mut state = GoState::new(9);
        // Set up a ko situation
        state.play_move(0, 0); // B
        state.play_move(0, 2); // W
        state.play_move(1, 1); // B
        state.play_move(2, 0); // W — now (0,1) area
        state.play_move(0, 1); // B — might trigger ko depending on state
        // Regardless, a pass should clear ko
        let ko_before = state.ko_point;
        state.play_pass();
        assert_eq!(state.ko_point, None, "Pass must clear ko_point");
        let _ = ko_before; // Suppress unused warning
    }

    // ── Capture ────────────────────────────────────────────────

    #[test]
    fn simple_capture_center() {
        // Black surrounds White at (4,4)
        let mut state = GoState::new(9);
        // Place White at center
        state.play_move(4, 4); // B at (4,4) — actually let's set this up properly
        state.play_pass(); // W passes
        // Now B has (4,4). Let's restart with a cleaner setup.
        let mut state = GoState::new(9);
        // Place White at (4,4)
        state.play_move(0, 0); // B at (0,0)
        state.play_move(4, 4); // W at (4,4)
        // Black surrounds: (3,4), (5,4), (4,3), (4,5)
        state.play_move(3, 4); // B
        state.play_pass(); // W
        state.play_move(5, 4); // B
        state.play_pass(); // W
        state.play_move(4, 3); // B
        state.play_pass(); // W
        state.play_move(4, 5); // B — captures W at (4,4)

        assert_eq!(
            state.at(4, 4),
            GoCell::Empty,
            "Captured stone should be removed"
        );
        assert_eq!(
            state.captured_black, 1,
            "Black should have captured 1 White stone"
        );
    }

    #[test]
    fn corner_capture() {
        // Black captures White at (0,0)
        let mut state = GoState::new(9);
        state.play_move(1, 1); // B at (1,1)
        state.play_move(0, 0); // W at (0,0)
        state.play_move(0, 1); // B at (0,1)
        state.play_pass(); // W passes
        state.play_move(1, 0); // B at (1,0) — captures W at (0,0)

        assert_eq!(
            state.at(0, 0),
            GoCell::Empty,
            "Corner capture removes stone"
        );
        assert_eq!(state.captured_black, 1);
    }

    #[test]
    fn group_capture() {
        // Capture a group of 2 White stones
        let mut state = GoState::new(9);
        // W at (0,0) and (0,1) — captured by B at (1,0), (1,1), (0,2)
        state.play_move(1, 0); // B
        state.play_move(0, 0); // W
        state.play_move(1, 1); // B
        state.play_move(0, 1); // W
        state.play_move(0, 2); // B — now W group {(0,0),(0,1)} has liberties:
        // (0,0) neighbors: (0,1)=W, (1,0)=B → no empty
        // (0,1) neighbors: (0,0)=W, (0,2)=B, (1,1)=B → no empty
        // Captured!
        assert_eq!(state.at(0, 0), GoCell::Empty);
        assert_eq!(state.at(0, 1), GoCell::Empty);
        assert_eq!(state.captured_black, 2, "Group of 2 captured");
    }

    // ── Suicide ────────────────────────────────────────────────

    #[test]
    fn single_stone_suicide_illegal() {
        // Black at (3,4), (5,4), (4,3), (4,5) — White cannot play (4,4)
        let mut state = GoState::new(9);
        state.play_move(3, 4); // B
        state.play_pass(); // W
        state.play_move(5, 4); // B
        state.play_pass(); // W
        state.play_move(4, 3); // B
        state.play_pass(); // W
        state.play_move(4, 5); // B — surrounds (4,4) for White

        // Now White tries (4,4) — surrounded by 4 Black, no captures
        assert!(
            !state.is_legal(4, 4),
            "Suicide at (4,4) should be illegal for White"
        );
    }

    #[test]
    fn capture_not_suicide() {
        // White stone at (0,0), Black at (0,1) and (1,0).
        // White plays at (0,1) — wait, let's set up properly.
        // B at (0,2), (1,1). W at (0,1).
        // If W plays (0,0): neighbors are (0,1)=W and (1,0)=empty.
        // That has a liberty, so it's legal. Not suicide.

        // W surrounded except for one B stone it captures
        let mut state = GoState::new(9);
        // Place B at (1,0), (0,2), (1,1). W at (0,1).
        // Now B plays (0,0): neighbors (0,1)=W, (1,0)=B.
        // B's group at (0,0)+(1,0) has liberties? (0,0) has no empty neighbors except what W occupies.
        // (1,0) has neighbors (0,0)=B, (1,1)=B, (2,0)=empty. So liberty at (2,0).
        // Also check: does B at (0,0) capture W at (0,1)?
        // W at (0,1): neighbors (0,0)=B, (0,2)=B, (1,1)=B → no empty → captured!
        // So B at (0,0) captures W at (0,1), then B's group has liberty. Legal!
        state.play_move(1, 0); // B
        state.play_move(0, 1); // W
        state.play_move(0, 2); // B
        state.play_pass(); // W
        state.play_move(1, 1); // B
        state.play_pass(); // W

        // Now B plays (0,0) — captures W at (0,1)
        assert!(state.is_legal(0, 0), "Capture-not-suicide should be legal");
        assert!(state.play_move(0, 0));
        assert_eq!(
            state.at(0, 1),
            GoCell::Empty,
            "W at (0,1) should be captured"
        );
        assert_eq!(state.at(0, 0), GoCell::Black);
    }

    // ── Ko ─────────────────────────────────────────────────────

    #[test]
    fn ko_prevents_immediate_recapture() {
        // Classic ko shape:
        // . X O .
        // X O . O
        // . X O .
        // Setup: after Black captures at (1,2), White cannot recapture at (1,1)
        let mut state = GoState::new(9);
        // Build the ko shape around (1,1) and (1,2)
        state.play_move(0, 1); // B
        state.play_move(0, 2); // W
        state.play_move(1, 0); // B
        state.play_move(1, 3); // W
        state.play_move(2, 1); // B
        state.play_move(2, 2); // W
        state.play_move(1, 2); // B — this is Black's setup move... let me redo

        // Actually let me build the shape more carefully.
        // Position:
        //   0 1 2 3
        // 0 . B W .
        // 1 B W . W
        // 2 . B W .
        //
        // Then B plays (1,2), capturing W at (1,1):
        //   0 1 2 3
        // 0 . B W .
        // 1 B . B W
        // 2 . B W .
        //
        // Now W cannot play (1,1) — it's ko.
        let mut state = GoState::new(9);
        // Build: B at (0,1), (1,0), (2,1). W at (0,2), (1,1), (1,3), (2,2).
        state.play_move(0, 1); // B
        state.play_move(0, 2); // W
        state.play_move(1, 0); // B
        state.play_move(1, 1); // W
        state.play_move(2, 1); // B
        state.play_move(1, 3); // W
        state.play_pass(); // B passes
        state.play_move(2, 2); // W — now W is at (0,2), (1,1), (1,3), (2,2)
        // Now B plays (1,2) — should capture W at (1,1)
        assert!(state.play_move(1, 2)); // B captures W at (1,1)
        assert_eq!(state.at(1, 1), GoCell::Empty, "W captured at (1,1)");
        assert_eq!(
            state.ko_point,
            Some(state.flat_index(1, 1)),
            "Ko point should be (1,1)"
        );

        // White cannot recapture at (1,1)
        assert!(
            !state.is_legal(1, 1),
            "Ko recapture at (1,1) should be illegal"
        );
    }

    #[test]
    fn ko_cleared_after_other_move() {
        // After the ko situation above, play elsewhere, then ko is cleared
        let mut state = GoState::new(9);
        state.play_move(0, 1); // B
        state.play_move(0, 2); // W
        state.play_move(1, 0); // B
        state.play_move(1, 1); // W
        state.play_move(2, 1); // B
        state.play_move(1, 3); // W
        state.play_pass(); // B
        state.play_move(2, 2); // W
        state.play_move(1, 2); // B captures at (1,1) → ko

        assert_eq!(state.ko_point, Some(state.flat_index(1, 1)));

        // White plays elsewhere
        state.play_move(8, 8); // W plays far away
        assert_eq!(
            state.ko_point, None,
            "Ko should be cleared after other move"
        );
        assert!(state.is_legal(1, 1), "Ko point should now be legal");
    }

    // ── Scoring ────────────────────────────────────────────────

    #[test]
    fn score_empty_board() {
        let mut state = GoState::new(9);
        state.play_pass();
        state.play_pass();
        let score = state.score();
        // No stones, no territory. Score = 0 - 7.5 = -7.5 (White wins by komi)
        assert!(
            (score - (-7.5)).abs() < 0.01,
            "Empty board score should be -7.5, got {score}"
        );
    }

    #[test]
    fn score_simple_territory() {
        // B fills left half, W fills right half of a 3x3 board (simpler)
        let mut state = GoState::with_komi(3, 0.0);
        // B: (0,0), (1,0), (2,0)
        // W: (0,2), (1,2), (2,2)
        // Middle column (0,1), (1,1), (2,1) empty — bordered by both → neutral
        state.play_move(0, 0); // B
        state.play_move(0, 2); // W
        state.play_move(1, 0); // B
        state.play_move(1, 2); // W
        state.play_move(2, 0); // B
        state.play_move(2, 2); // W
        state.play_pass();
        state.play_pass();

        let score = state.score();
        // B: 3 stones + 0 territory (middle is neutral)
        // W: 3 stones + 0 territory
        // Score = 3.0 - 3.0 = 0.0
        assert!(
            score.abs() < 0.01,
            "Score should be 0.0 with neutral territory, got {score}"
        );
    }

    #[test]
    fn score_black_territory() {
        // 3x3, B surrounds all of W's territory
        let mut state = GoState::with_komi(3, 0.0);
        // B fills entire top row and sides
        state.play_move(0, 0); // B
        state.play_pass(); // W
        state.play_move(0, 1); // B
        state.play_pass(); // W
        state.play_move(0, 2); // B
        state.play_pass(); // W
        state.play_move(1, 0); // B
        state.play_pass(); // W
        state.play_move(1, 2); // B
        state.play_pass(); // W
        state.play_move(2, 0); // B
        state.play_pass(); // W
        state.play_move(2, 1); // B
        state.play_pass(); // W
        state.play_move(2, 2); // B
        state.play_pass(); // W passes

        // Board: all Black except (1,1) which is empty, surrounded by B
        state.play_pass(); // B passes → game over

        let score = state.score();
        // B: 8 stones + 1 territory (1,1) = 9
        // W: 0 stones + 0 territory = 0
        // Score = 9.0 - 0.0 = 9.0
        assert!(
            (score - 9.0).abs() < 0.01,
            "Score should be 9.0, got {score}"
        );
    }

    // ── GameState Trait ────────────────────────────────────────

    #[test]
    fn game_state_advance_is_immutable() {
        let state = GoState::new(9);
        let original_moves = state.legal_move_count();
        let _next = state.advance(&GoAction::Place(4, 4), 0);
        assert_eq!(
            state.legal_move_count(),
            original_moves,
            "advance() must not mutate self"
        );
    }

    #[test]
    fn game_state_available_actions_includes_pass() {
        let state = GoState::new(9);
        let actions = state.available_actions(0);
        assert_eq!(actions.len(), 82, "81 places + 1 pass");
        assert!(actions.contains(&GoAction::Pass));
    }

    #[test]
    fn game_state_reward_black_wins() {
        let mut state = GoState::with_komi(3, 0.0);
        // B fills entire board
        for r in 0..3 {
            for c in 0..3 {
                state.play_move(r, c); // B
                if !(r == 2 && c == 2) {
                    state.play_pass(); // W passes
                }
            }
        }
        state.play_pass(); // W passes
        state.play_pass(); // B passes → game over

        assert!(state.is_terminal());
        assert!(
            state.reward(0) > 0.9,
            "Black should win, reward={}",
            state.reward(0)
        );
        assert!(
            state.reward(1) < 0.1,
            "White should lose, reward={}",
            state.reward(1)
        );
    }

    // ── Property: Random Game ──────────────────────────────────

    #[test]
    fn random_game_9x9_completes_without_panic() {
        let mut rng = fastrand::Rng::new();
        for seed in 0..50u64 {
            rng.seed(seed);
            let mut state = GoState::new(9);
            for _ in 0..300 {
                if state.is_terminal() {
                    break;
                }
                let legal = state.legal_moves();
                if legal.is_empty() || rng.f32() < 0.02 {
                    // Occasional pass to test game end
                    state.play_pass();
                } else {
                    let idx = rng.usize(..legal.len());
                    let (r, c) = legal[idx];
                    let ok = state.play_move(r, c);
                    assert!(ok, "Legal move ({r},{c}) failed in game seed={seed}");
                }
            }
            // Force game end
            if !state.is_terminal() {
                state.play_pass();
                state.play_pass();
            }
            assert!(state.is_terminal(), "Game seed={seed} should terminate");
            let _score = state.score(); // Should not panic
        }
    }

    #[test]
    fn random_game_invariants() {
        let mut rng = fastrand::Rng::new();
        for seed in 0..20u64 {
            rng.seed(seed);
            let mut state = GoState::new(9);
            for _ in 0..200 {
                if state.is_terminal() {
                    break;
                }
                let legal = state.legal_moves();
                // Invariant: legal moves are actually legal
                for &(r, c) in &legal {
                    assert!(
                        state.is_legal(r, c),
                        "legal_moves contains illegal ({r},{c}) seed={seed}"
                    );
                }

                if legal.is_empty() || rng.f32() < 0.05 {
                    state.play_pass();
                } else {
                    let idx = rng.usize(..legal.len());
                    let (r, c) = legal[idx];
                    let _pre_board = state.board.clone();
                    let ok = state.play_move(r, c);
                    assert!(ok);
                    // Invariant: stone was placed (cell was empty, now has current player's color)
                    assert!(
                        state.at(r, c).is_stone(),
                        "Cell ({r},{c}) should have a stone after play_move"
                    );
                    // Invariant: stones on board never exceed moves played
                    // (each move places at most 1 stone; captures only remove stones)
                    let black_stones = state.stone_count(GoCell::Black);
                    let white_stones = state.stone_count(GoCell::White);
                    assert!(
                        black_stones + white_stones <= state.move_count as usize,
                        "Stones on board ({black_stones}+{white_stones}={}) exceed move_count={} in seed={seed}",
                        black_stones + white_stones,
                        state.move_count
                    );
                }
            }
        }
    }

    // ── Display ────────────────────────────────────────────────

    #[test]
    fn display_shows_board() {
        let state = GoState::new(3);
        let display = format!("{state}");
        assert!(
            display.contains("X") || display.contains("·"),
            "Display should show board"
        );
        assert!(display.contains("To play: Black"));
    }
}
