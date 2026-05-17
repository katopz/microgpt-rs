//! Plan 065 Phase 2 T24: Go Player Tournament — 9×9.
//!
//! Pits different Go player types against each other to measure relative strength.
//! Configurable via environment variables:
//!
//! ```sh
//! # Default: 10 games per matchup, 9×9 board
//! cargo run --features go --example go_02_tournament
//!
//! # Custom: 20 games per matchup
//! GO_GAMES=20 cargo run --features go --example go_02_tournament
//! ```

use std::env;
use std::time::Instant;

use fastrand::Rng;
use microgpt_rs::pruners::go::{
    DEFAULT_KOMI, GoAction, GoCell, GoGZeroPlayer, GoGreedyPlayer, GoHLPlayer, GoMctsPlayer,
    GoPlayer, GoRandomPlayer, GoReplay, GoState, GoValidatorPlayer,
};

// ── Constants ──────────────────────────────────────────────────

/// Default number of games per matchup.
const DEFAULT_NUM_GAMES: usize = 10;

/// Default board size.
const DEFAULT_BOARD_SIZE: usize = 9;

/// Max moves before forcing game end (safety limit).
const MAX_MOVES: usize = 300;

// ── Player Factory ─────────────────────────────────────────────

/// Create a player by name.
fn make_player(name: &str) -> Box<dyn GoPlayer> {
    match name {
        "random" => Box::new(GoRandomPlayer),
        "greedy" => Box::new(GoGreedyPlayer),
        "validator" => Box::new(GoValidatorPlayer),
        "hl" => Box::new(GoHLPlayer::new()),
        "mcts" => Box::new(GoMctsPlayer::new(200, 50)),
        "gzero" => Box::new(GoGZeroPlayer::new()),
        _ => panic!("Unknown player: {name}"),
    }
}

// ── Outcome Update Helper ──────────────────────────────────────

/// Call `update_outcome(won)` on bandit-based players (HL, GZero).
fn update_player_outcome(player: &mut dyn GoPlayer, won: bool) {
    if let Some(hl) = player.as_any_mut().downcast_mut::<GoHLPlayer>() {
        hl.update_outcome(won);
    }
    if let Some(gz) = player.as_any_mut().downcast_mut::<GoGZeroPlayer>() {
        gz.update_outcome(won);
    }
}

// ── Game Result ────────────────────────────────────────────────

/// Result of a single game.
#[allow(dead_code)]
struct GameResult {
    /// Did the first player (the one being tested) win?
    first_player_won: bool,
    /// Final score (Black perspective). Positive = Black wins.
    score: f32,
    /// Total moves played.
    moves: usize,
    /// Time spent on this game.
    duration: std::time::Duration,
    /// Which color the first player was.
    first_player_color: GoCell,
}

/// Result of a matchup (multiple games).
#[allow(dead_code)]
struct MatchupResult {
    /// Name of the first player.
    first_player: String,
    /// Name of the second player.
    second_player: String,
    /// Individual game results.
    games: Vec<GameResult>,
}

impl MatchupResult {
    fn wins(&self) -> usize {
        self.games.iter().filter(|g| g.first_player_won).count()
    }

    fn losses(&self) -> usize {
        self.games.len() - self.wins()
    }

    fn win_rate(&self) -> f64 {
        self.wins() as f64 / self.games.len() as f64 * 100.0
    }

    fn avg_moves(&self) -> f64 {
        let total: usize = self.games.iter().map(|g| g.moves).sum();
        total as f64 / self.games.len() as f64
    }
}

// ── Game Loop ──────────────────────────────────────────────────

/// Play a single game between two players, returning the result from player_a's perspective.
fn play_game(
    player_a: &mut dyn GoPlayer,
    player_b: &mut dyn GoPlayer,
    player_a_color: GoCell,
    board_size: usize,
    rng: &mut Rng,
) -> GameResult {
    let start = Instant::now();
    let mut state = GoState::new(board_size);
    let mut replay = GoReplay::new(board_size, DEFAULT_KOMI);
    let mut moves = 0usize;

    for _ in 0..MAX_MOVES {
        if state.is_terminal() {
            break;
        }

        let legal = state.legal_moves();
        let legal_count = state.legal_move_count();

        let action = if state.to_play == player_a_color {
            player_a.select_move(&state, &legal, rng)
        } else {
            player_b.select_move(&state, &legal, rng)
        };

        // Apply action
        match &action {
            GoAction::Place(row, col) => {
                let ok = state.play_move(*row, *col);
                debug_assert!(ok, "Player selected illegal move ({row},{col})");
            }
            GoAction::Pass => {
                state.play_pass();
            }
        }

        replay.record(&action, state.to_play.opponent(), legal_count);
        moves += 1;
    }

    // Force game end if not terminal
    if !state.is_terminal() {
        state.play_pass();
        state.play_pass();
        moves += 2;
    }

    let score = state.score();
    let winner = state.get_winner();
    let first_player_won = winner == Some(player_a_color);

    replay.finalize(winner, score);

    let duration = start.elapsed();

    GameResult {
        first_player_won,
        score,
        moves,
        duration,
        first_player_color: player_a_color,
    }
}

/// Run a matchup: `first_player_name` vs `second_player_name` for `num_games` games.
/// Colors alternate each game.
fn run_matchup(
    first_player_name: &str,
    second_player_name: &str,
    num_games: usize,
    board_size: usize,
    rng: &mut Rng,
) -> MatchupResult {
    let mut player_a = make_player(first_player_name);
    let mut player_b = make_player(second_player_name);

    let mut games = Vec::with_capacity(num_games);

    for i in 0..num_games {
        // Swap colors each game
        let player_a_color = match i % 2 {
            0 => GoCell::Black,
            _ => GoCell::White,
        };

        let color_label = match player_a_color {
            GoCell::Black => "B",
            GoCell::White => "W",
            _ => unreachable!(),
        };

        print!(
            "  [{:>2}/{}] {}({}) ",
            i + 1,
            num_games,
            player_a.name(),
            color_label
        );
        let _ = std::io::Write::flush(&mut std::io::stdout());

        let result = play_game(
            player_a.as_mut(),
            player_b.as_mut(),
            player_a_color,
            board_size,
            rng,
        );

        let outcome = if result.first_player_won { "W" } else { "L" };
        let score_display = if result.score > 0.0 {
            format!("B+{:.1}", result.score)
        } else {
            format!("W+{:.1}", result.score.abs())
        };
        println!(
            "{outcome} {:>8} {:>3} moves ({:.1}s)",
            score_display,
            result.moves,
            result.duration.as_secs_f64()
        );

        // Update bandit-based players
        update_player_outcome(player_a.as_mut(), result.first_player_won);
        update_player_outcome(player_b.as_mut(), !result.first_player_won);

        games.push(result);
    }

    // Reset players between matchups (not between games — bandit learning persists)
    player_a.reset();
    player_b.reset();

    MatchupResult {
        first_player: first_player_name.to_string(),
        second_player: second_player_name.to_string(),
        games,
    }
}

// ── Matchup Definitions ────────────────────────────────────────

/// A matchup definition: first player vs second player.
struct MatchupDef {
    first: &'static str,
    second: &'static str,
}

impl MatchupDef {
    const fn new(first: &'static str, second: &'static str) -> Self {
        Self { first, second }
    }

    fn label(&self) -> String {
        format!("{} vs {}", self.first, self.second)
    }
}

/// All matchups to run.
const MATCHUPS: &[MatchupDef] = &[
    MatchupDef::new("random", "random"),
    MatchupDef::new("greedy", "random"),
    MatchupDef::new("validator", "random"),
    MatchupDef::new("hl", "random"),
    MatchupDef::new("mcts", "random"),
];

// ── Output Formatting ─────────────────────────────────────────

/// Print the tournament header.
fn print_header(num_games: usize, board_size: usize) {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║           Go Player Tournament — {board_size}×{board_size}                    ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Games per matchup: {num_games:<4}                                ║");
    println!("║  Komi: {DEFAULT_KOMI}                                          ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
}

/// Print a single matchup's summary line.
fn print_matchup_summary(result: &MatchupResult) {
    let wins = result.wins();
    let losses = result.losses();
    let win_rate = result.win_rate();
    let avg_moves = result.avg_moves();
    println!(
        "  Result: {} {}W / {}L ({:.1}%) avg {:.1} moves",
        result.first_player, wins, losses, win_rate, avg_moves
    );
    println!();
}

/// Print the final results table.
fn print_final_table(results: &[MatchupResult]) {
    println!("════════════════════════════════════════════════════════════");
    println!("  FINAL RESULTS");
    println!("════════════════════════════════════════════════════════════");
    println!("  {:<14} {:<12} {:>6}", "Player", "vs Random", "Win%");
    println!("  {}  {}  {}", "──────────────", "───────────", "─────");

    for result in results {
        let wins = result.wins();
        let losses = result.losses();
        let win_rate = result.win_rate();
        let wl = format!("{wins}W/{losses}L");
        println!(
            "  {:<14} {:<10} {:>5.1}%",
            result.first_player, wl, win_rate
        );
    }

    println!("════════════════════════════════════════════════════════════");
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    let num_games: usize = env::var("GO_GAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NUM_GAMES);

    let board_size: usize = env::var("GO_BOARD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BOARD_SIZE);

    print_header(num_games, board_size);

    let mut rng = fastrand::Rng::with_seed(42);
    let mut all_results: Vec<MatchupResult> = Vec::with_capacity(MATCHUPS.len());

    for (idx, matchup) in MATCHUPS.iter().enumerate() {
        println!(
            "Matchup {}/{}: {}",
            idx + 1,
            MATCHUPS.len(),
            matchup.label()
        );

        let result = run_matchup(
            matchup.first,
            matchup.second,
            num_games,
            board_size,
            &mut rng,
        );
        print_matchup_summary(&result);
        all_results.push(result);
    }

    print_final_table(&all_results);
}
