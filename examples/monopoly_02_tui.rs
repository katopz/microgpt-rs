//! Monopoly HL Arena — Animated TUI Replay with Walk Effect
//!
//! Pre-runs a full Monopoly game, then replays with a visual board perimeter
//! and step-by-step walk animation for player movement.
//!
//! Board Layout (40 squares around perimeter):
//!   Top row:    squares 30→20 (left to right on screen)
//!   Right col:  squares 19→11 (top to bottom on screen)
//!   Bottom row: squares 0→10  (left to right on screen)
//!   Left col:   squares 31→39 (top to bottom on screen)
//!
//! Interior shows player stats and current action.
//!
//! Controls:
//!   Space / →      — Next frame      ← / Backspace — Previous frame
//!   A              — Toggle auto-play  F              — Fast forward
//!   Home / End     — Jump start/end    Q / Esc        — Quit
//!
//! Run: `cargo run --example monopoly_02_tui --features monopoly`

use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use fastrand::Rng;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::{Frame, Terminal};

use microgpt_rs::pruners::monopoly::{
    CardEffect, GameEvent, GreedyPlayer, HLPlayer, JailReason, MonopolyPlayer, PropertyGroup,
    RandomPlayer, ReleaseMethod, SquareKind, TaxKind, ValidatorPlayer, run_game, square_kind,
    square_name,
};

// ── Constants ──────────────────────────────────────────────────

const P_EMOJI: [&str; 4] = ["🐰", "🐱", "🐶", "🐵"];
const P_NAMES: [&str; 4] = ["Random", "Greedy", "Validator", "HL"];
const MAX_TURNS: u32 = 200;
const WALK_STEP_MS: u64 = 80;
const EVENT_STEP_MS: u64 = 300;
const FAST_FWD_MS: u64 = 10;
const CELL_W: usize = 6; // " XX🐰 " per cell

/// 2-char abbreviations for all 40 squares
const SQ_ABBREV: [&str; 40] = [
    "GO", "ME", "CC", "BA", "Tx", "RR", "OR", "Ch", "VE", "CT", // 0-9
    "JL", "SC", "EU", "ST", "VI", "PR", "SJ", "CC", "TN", "NY", // 10-19
    "FP", "KY", "Ch", "IN", "IL", "BO", "AT", "VN", "WW", "MG", // 20-29
    "GJ", "PC", "NC", "CC", "PA", "RR", "Ch", "PP", "Tx", "BW", // 30-39
];

// ── Types ──────────────────────────────────────────────────────

#[derive(Clone)]
struct PlayerVisual {
    cash: u32,
    position: u8,
    properties: Vec<u8>,
    is_bankrupt: bool,
    in_jail: bool,
}

impl PlayerVisual {
    fn new() -> Self {
        Self {
            cash: 1500,
            position: 0,
            properties: Vec::new(),
            is_bankrupt: false,
            in_jail: false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FrameType {
    Walk,
    Event,
}

struct ReplayFrame {
    players: [PlayerVisual; 4],
    owners: [Option<u8>; 40],
    _houses: [u8; 40],
    active_player: Option<u8>,
    highlight: Option<u8>,
    event_text: Option<String>,
    last_event_text: String,
    dice_info: Option<(u8, u8, bool)>, // (die1, die2, doubles)
    frame_type: FrameType,
}

struct RecordedGame {
    frames: Vec<ReplayFrame>,
    events: Vec<String>,
    winner: u8,
    total_turns: u32,
    seed: u64,
}

// ── Grid Mapping ───────────────────────────────────────────────

/// Map square index (0-39) to grid position (col, row) on an 11×11 grid.
fn _sq_to_grid(sq: u8) -> (usize, usize) {
    match sq {
        0..=10 => (sq as usize, 10),
        11..=19 => (10, (20 - sq) as usize),
        20..=30 => ((30 - sq) as usize, 0),
        31..=39 => (0, (sq - 30) as usize),
        _ => (0, 10),
    }
}

/// Map grid position to square index. Returns None for interior cells.
fn grid_to_sq(row: usize, col: usize) -> Option<u8> {
    match (row, col) {
        (10, 0..=10) => Some(col as u8),
        (1..=9, 10) => Some((20 - row) as u8),
        (0, 0..=10) => Some((30 - col) as u8),
        (1..=9, 0) => Some((30 + row) as u8),
        _ => None,
    }
}

// ── Color Helpers ──────────────────────────────────────────────

fn group_bg(g: &PropertyGroup) -> Color {
    match g {
        PropertyGroup::Brown => Color::Rgb(100, 60, 30),
        PropertyGroup::LightBlue => Color::Rgb(50, 100, 140),
        PropertyGroup::Pink => Color::Rgb(130, 50, 90),
        PropertyGroup::Orange => Color::Rgb(150, 90, 30),
        PropertyGroup::Red => Color::Rgb(140, 30, 30),
        PropertyGroup::Yellow => Color::Rgb(140, 140, 30),
        PropertyGroup::Green => Color::Rgb(30, 110, 30),
        PropertyGroup::DarkBlue => Color::Rgb(30, 30, 140),
    }
}

fn group_fg(g: &PropertyGroup) -> Color {
    match g {
        PropertyGroup::Brown => Color::Rgb(180, 120, 70),
        PropertyGroup::LightBlue => Color::Rgb(130, 190, 220),
        PropertyGroup::Pink => Color::Rgb(220, 130, 180),
        PropertyGroup::Orange => Color::Rgb(240, 170, 80),
        PropertyGroup::Red => Color::Rgb(230, 100, 100),
        PropertyGroup::Yellow => Color::Rgb(230, 230, 100),
        PropertyGroup::Green => Color::Rgb(100, 200, 100),
        PropertyGroup::DarkBlue => Color::Rgb(100, 100, 230),
    }
}

fn square_bg(sq: u8, owner: Option<u8>) -> Color {
    match square_kind(sq) {
        SquareKind::Property(g) if owner.is_some() => group_bg(&g),
        SquareKind::Property(_) => Color::Rgb(35, 35, 35),
        SquareKind::Go => Color::Rgb(50, 50, 25),
        SquareKind::FreeParking => Color::Rgb(25, 50, 25),
        SquareKind::Jail => Color::Rgb(50, 40, 25),
        SquareKind::GoToJail => Color::Rgb(50, 25, 25),
        SquareKind::Railroad => Color::Rgb(35, 35, 45),
        SquareKind::Utility => Color::Rgb(35, 45, 35),
        SquareKind::Tax(_) => Color::Rgb(45, 35, 35),
        SquareKind::Chance => Color::Rgb(50, 50, 25),
        SquareKind::CommunityChest => Color::Rgb(25, 50, 50),
    }
}

// ── State Tracking ─────────────────────────────────────────────

fn apply_event(
    event: &GameEvent,
    ps: &mut [PlayerVisual; 4],
    owners: &mut [Option<u8>; 40],
    houses: &mut [u8; 40],
) {
    match event {
        GameEvent::PlayerMoved { player, to, .. } => {
            ps[*player as usize].position = *to;
            // Salary is handled by SalaryCollected event, not here
        }
        GameEvent::SalaryCollected { .. } => {
            // Handled by deferral logic in record_game
        }
        GameEvent::PropertyBought {
            player,
            square,
            price,
        } => {
            let p = *player as usize;
            ps[p].cash = ps[p].cash.saturating_sub(*price);
            ps[p].properties.push(*square);
            owners[*square as usize] = Some(*player);
        }
        GameEvent::PropertyAuctioned {
            square,
            winner,
            price,
        } => {
            let w = *winner as usize;
            ps[w].cash = ps[w].cash.saturating_sub(*price);
            ps[w].properties.push(*square);
            owners[*square as usize] = Some(*winner);
        }
        GameEvent::RentPaid {
            payer,
            payee,
            amount,
            ..
        } => {
            ps[*payer as usize].cash = ps[*payer as usize].cash.saturating_sub(*amount);
            ps[*payee as usize].cash += amount;
        }
        GameEvent::TaxPaid { player, amount, .. } => {
            ps[*player as usize].cash = ps[*player as usize].cash.saturating_sub(*amount);
        }
        GameEvent::CardDrawn { player, effect, .. } => match effect {
            CardEffect::CollectMoney(a) | CardEffect::CollectFromEachPlayer(a) => {
                ps[*player as usize].cash += a;
            }
            CardEffect::PayMoney(a) => {
                ps[*player as usize].cash = ps[*player as usize].cash.saturating_sub(*a);
            }
            CardEffect::PayEachPlayer(a) => {
                ps[*player as usize].cash = ps[*player as usize].cash.saturating_sub(*a * 3);
            }
            CardEffect::GoToJail => {
                ps[*player as usize].position = 10;
                ps[*player as usize].in_jail = true;
            }
            CardEffect::GetOutOfJailFree => {
                ps[*player as usize].in_jail = false;
            }
            _ => {}
        },
        GameEvent::HouseBuilt {
            player,
            square,
            houses: new_total,
        } => {
            let cost = match square_kind(*square) {
                SquareKind::Property(g) => match g {
                    PropertyGroup::Brown | PropertyGroup::LightBlue => 50,
                    PropertyGroup::Pink | PropertyGroup::Orange => 100,
                    PropertyGroup::Red | PropertyGroup::Yellow => 150,
                    PropertyGroup::Green | PropertyGroup::DarkBlue => 200,
                },
                _ => 0,
            };
            let prev = houses[*square as usize];
            let built = new_total.saturating_sub(prev);
            ps[*player as usize].cash = ps[*player as usize]
                .cash
                .saturating_sub(cost * built as u32);
            houses[*square as usize] = *new_total;
        }
        GameEvent::PropertyMortgaged { player, amount, .. } => {
            ps[*player as usize].cash += amount;
        }
        GameEvent::PropertyUnmortgaged { player, cost, .. } => {
            ps[*player as usize].cash = ps[*player as usize].cash.saturating_sub(*cost);
        }
        GameEvent::PlayerJailed { player, .. } => {
            ps[*player as usize].in_jail = true;
            ps[*player as usize].position = 10;
        }
        GameEvent::PlayerReleasedFromJail { player, method } => {
            let p = *player as usize;
            ps[p].in_jail = false;
            if let ReleaseMethod::PaidFine = method {
                ps[p].cash = ps[p].cash.saturating_sub(50);
            }
        }
        GameEvent::PlayerBankrupt { player, creditor } => {
            let p = *player as usize;
            ps[p].is_bankrupt = true;
            let props = ps[p].properties.clone();
            for sq in &props {
                owners[*sq as usize] = *creditor;
            }
            if let Some(c) = creditor {
                ps[*c as usize].properties.extend_from_slice(&props);
            }
            ps[p].cash = 0;
        }
        _ => {}
    }
}

// ── Walk Steps ─────────────────────────────────────────────────

fn walk_steps(from: u8, to: u8, passed_go: bool) -> Vec<u8> {
    if from == to {
        return vec![];
    }
    let mut steps = Vec::new();
    if passed_go || to > from {
        let mut pos = from;
        loop {
            pos = (pos + 1) % 40;
            steps.push(pos);
            if pos == to {
                break;
            }
        }
    } else {
        let mut pos = from;
        loop {
            pos = (pos + 39) % 40;
            steps.push(pos);
            if pos == to {
                break;
            }
        }
    }
    steps
}

fn active_player_from(event: &GameEvent) -> Option<u8> {
    match event {
        GameEvent::TurnStarted { player }
        | GameEvent::DiceRolled { player, .. }
        | GameEvent::PlayerMoved { player, .. }
        | GameEvent::SalaryCollected { player, .. }
        | GameEvent::PropertyBought { player, .. }
        | GameEvent::PropertyDeclined { player, .. }
        | GameEvent::TaxPaid { player, .. }
        | GameEvent::CardDrawn { player, .. }
        | GameEvent::HouseBuilt { player, .. }
        | GameEvent::PropertyMortgaged { player, .. }
        | GameEvent::PropertyUnmortgaged { player, .. }
        | GameEvent::PlayerJailed { player, .. }
        | GameEvent::PlayerReleasedFromJail { player, .. }
        | GameEvent::PlayerBankrupt { player, .. } => Some(*player),
        GameEvent::RentPaid { payer, .. } => Some(*payer),
        GameEvent::PropertyAuctioned { winner, .. } => Some(*winner),
        GameEvent::TradeOffered { proposer, .. } => Some(*proposer),
        GameEvent::TradeAccepted { responder, .. } => Some(*responder),
        GameEvent::TradeDeclined { responder, .. } => Some(*responder),
        GameEvent::AuctionBid { player, .. } | GameEvent::AuctionWon { player, .. } => {
            Some(*player)
        }
        GameEvent::AuctionStarted { .. } | GameEvent::GameOver { .. } => None,
    }
}

// ── Event Formatting ───────────────────────────────────────────

fn fmt_card(effect: &CardEffect) -> String {
    match effect {
        CardEffect::CollectMoney(a) => format!("Collect ${a}"),
        CardEffect::PayMoney(a) => format!("Pay ${a}"),
        CardEffect::PayPerHouse { house, hotel } => {
            format!("Pay ${house}/house, ${hotel}/hotel")
        }
        CardEffect::MoveTo(pos) => format!("Move to {}", square_name(*pos)),
        CardEffect::MoveBack(s) => format!("Move back {s} spaces"),
        CardEffect::MoveToNearest { is_railroad } => {
            if *is_railroad {
                "Move to nearest Railroad".into()
            } else {
                "Move to nearest Utility".into()
            }
        }
        CardEffect::GoToJail => "Go to Jail!".into(),
        CardEffect::GetOutOfJailFree => "Get Out of Jail Free".into(),
        CardEffect::PayEachPlayer(a) => format!("Pay each player ${a}"),
        CardEffect::CollectFromEachPlayer(a) => format!("Collect ${a} from each player"),
    }
}

fn fmt_event(evt: &GameEvent) -> String {
    match evt {
        GameEvent::TurnStarted { player } => format!("P{player}'s turn"),
        GameEvent::DiceRolled {
            player,
            die1,
            die2,
            doubles,
        } => {
            let sum = die1 + die2;
            let d = if *doubles { " doubles!" } else { "" };
            format!("P{player} rolled {die1}+{die2}={sum}{d}")
        }
        GameEvent::PlayerMoved {
            player,
            to,
            passed_go,
            ..
        } => {
            let go = if *passed_go { " (passed GO)" } else { "" };
            format!("P{player} -> {}{go}", square_name(*to))
        }
        GameEvent::SalaryCollected { player, amount } => {
            format!("P{player} collected ${amount} salary")
        }
        GameEvent::PropertyBought {
            player,
            square,
            price,
        } => {
            format!("P{player} bought {} for ${price}", square_name(*square))
        }
        GameEvent::PropertyAuctioned {
            square,
            winner,
            price,
        } => {
            format!(
                "P{winner} won auction for {} at ${price}",
                square_name(*square)
            )
        }
        GameEvent::PropertyDeclined { player, square } => {
            format!("P{player} declined {}", square_name(*square))
        }
        GameEvent::RentPaid {
            payer,
            payee,
            amount,
            square,
        } => {
            format!(
                "P{payer} paid ${amount} rent to P{payee} ({})",
                square_name(*square)
            )
        }
        GameEvent::TaxPaid {
            player,
            amount,
            tax_kind,
        } => {
            let k = match tax_kind {
                TaxKind::Income => "Income Tax",
                TaxKind::Luxury => "Luxury Tax",
            };
            format!("P{player} paid ${amount} {k}")
        }
        GameEvent::CardDrawn {
            player,
            is_chance,
            effect,
        } => {
            let deck = if *is_chance {
                "Chance"
            } else {
                "Community Chest"
            };
            format!("P{player} drew {deck}: {}", fmt_card(effect))
        }
        GameEvent::HouseBuilt {
            player,
            square,
            houses,
        } => {
            format!(
                "P{player} built on {} (now {houses} houses)",
                square_name(*square)
            )
        }
        GameEvent::PropertyMortgaged {
            player,
            square,
            amount,
        } => {
            format!("P{player} mortgaged {} for ${amount}", square_name(*square))
        }
        GameEvent::PropertyUnmortgaged {
            player,
            square,
            cost,
        } => {
            format!("P{player} unmortgaged {} for ${cost}", square_name(*square))
        }
        GameEvent::TradeOffered {
            proposer,
            responder,
        } => {
            format!("P{proposer} offered trade to P{responder}")
        }
        GameEvent::TradeAccepted {
            proposer,
            responder,
        } => {
            format!("P{responder} accepted trade from P{proposer}")
        }
        GameEvent::TradeDeclined {
            proposer,
            responder,
        } => {
            format!("P{responder} declined trade from P{proposer}")
        }
        GameEvent::PlayerJailed { player, reason } => {
            let why = match reason {
                JailReason::LandedOnGoToJail => "landed on Go To Jail",
                JailReason::Speeding => "speeding (3 doubles)",
                JailReason::CardEffect => "card effect",
            };
            format!("P{player} sent to jail ({why})")
        }
        GameEvent::PlayerReleasedFromJail { player, method } => {
            let how = match method {
                ReleaseMethod::PaidFine => "paid $50 fine",
                ReleaseMethod::UsedCard => "used GOOJF card",
                ReleaseMethod::RolledDoubles => "rolled doubles",
                ReleaseMethod::MaxTurnsExceeded => "served max turns",
            };
            format!("P{player} released ({how})")
        }
        GameEvent::PlayerBankrupt { player, creditor } => {
            let c = creditor.map_or(String::new(), |c| format!(" (debts to P{c})"));
            format!("P{player} BANKRUPT{c}")
        }
        GameEvent::GameOver { winner } => {
            format!(
                "🏆 Game Over! P{winner} ({}) wins!",
                P_NAMES[*winner as usize]
            )
        }
        GameEvent::AuctionStarted { square } => {
            format!("Auction: {}", square_name(*square))
        }
        GameEvent::AuctionBid { player, amount } => format!("P{player} bid ${amount}"),
        GameEvent::AuctionWon {
            player,
            square,
            amount,
        } => {
            format!(
                "P{player} won {} at auction for ${amount}",
                square_name(*square)
            )
        }
    }
}

// ── Recording ──────────────────────────────────────────────────

fn record_game(seed: u64) -> RecordedGame {
    let mut rng = Rng::with_seed(seed);
    let mut players: [Box<dyn MonopolyPlayer>; 4] = [
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(ValidatorPlayer::new(2)),
        Box::new(HLPlayer::new(3)),
    ];
    let result = run_game(seed, &mut players, &mut rng, MAX_TURNS);

    let mut frames: Vec<ReplayFrame> = Vec::new();
    let mut all_events: Vec<String> = Vec::new();
    let mut ps = [
        PlayerVisual::new(),
        PlayerVisual::new(),
        PlayerVisual::new(),
        PlayerVisual::new(),
    ];
    let mut owners: [Option<u8>; 40] = [None; 40];
    let mut houses: [u8; 40] = [0; 40];
    let mut pending_salary: Option<(u8, u32)> = None;
    let mut last_text = String::new();
    let mut last_dice: Option<(u8, u8, bool)> = None;

    for event in &result.events {
        let active = active_player_from(event);

        // Track dice rolls
        if let GameEvent::DiceRolled {
            die1,
            die2,
            doubles,
            ..
        } = event
        {
            last_dice = Some((*die1, *die2, *doubles));
        }

        // Defer SalaryCollected to apply during walk
        if let GameEvent::SalaryCollected { player, amount } = event {
            pending_salary = Some((*player, *amount));
            continue;
        }

        // PlayerMoved: generate walk frames
        if let GameEvent::PlayerMoved {
            player,
            from,
            to,
            passed_go,
        } = event
        {
            let steps = walk_steps(*from, *to, *passed_go);
            for pos in steps {
                // Apply pending salary when visually passing GO
                if let Some((sp, sa)) = pending_salary
                    && sp == *player
                    && pos == 0
                {
                    ps[*player as usize].cash += sa;
                    pending_salary = None;
                }

                let mut walk_ps = ps.clone();
                walk_ps[*player as usize].position = pos;
                frames.push(ReplayFrame {
                    players: walk_ps,
                    owners,
                    _houses: houses,
                    active_player: Some(*player),
                    highlight: Some(pos),
                    event_text: None,
                    last_event_text: last_text.clone(),
                    dice_info: last_dice,
                    frame_type: FrameType::Walk,
                });
            }

            // Apply final position
            ps[*player as usize].position = *to;

            let text = fmt_event(event);
            last_text = text.clone();
            all_events.push(text.clone());

            frames.push(ReplayFrame {
                players: ps.clone(),
                owners,
                _houses: houses,
                active_player: Some(*player),
                highlight: None,
                event_text: Some(text),
                last_event_text: last_text.clone(),
                dice_info: last_dice,
                frame_type: FrameType::Event,
            });
            continue;
        }

        // Apply any lingering pending salary
        if let Some((sp, sa)) = pending_salary.take() {
            ps[sp as usize].cash += sa;
        }

        // Apply event to state
        apply_event(event, &mut ps, &mut owners, &mut houses);

        let text = fmt_event(event);
        last_text = text.clone();
        all_events.push(text.clone());

        frames.push(ReplayFrame {
            players: ps.clone(),
            owners,
            _houses: houses,
            active_player: active,
            highlight: None,
            event_text: Some(text),
            last_event_text: last_text.clone(),
            dice_info: last_dice,
            frame_type: FrameType::Event,
        });
    }

    RecordedGame {
        frames,
        events: all_events,
        winner: result.winner,
        total_turns: result.total_turns,
        seed,
    }
}

// ── Rendering Helpers ──────────────────────────────────────────

/// Pad or truncate string to exact display width.
fn pad_to(s: &str, width: usize) -> String {
    let mut w = 0usize;
    let mut end = 0;
    for (i, c) in s.char_indices() {
        let cw = if c.is_ascii() { 1 } else { 2 };
        if w + cw > width {
            break;
        }
        w += cw;
        end = i + c.len_utf8();
    }
    let truncated = &s[..end];
    let padding = width.saturating_sub(w);
    format!("{truncated}{}", " ".repeat(padding))
}

fn cell_content(sq: u8, snap: &ReplayFrame) -> String {
    // Show abbreviation + active player emoji if on this square
    for (i, player) in snap.players.iter().enumerate() {
        if !player.is_bankrupt && player.position == sq && snap.active_player == Some(i as u8) {
            return format!("{}{}", SQ_ABBREV[sq as usize], P_EMOJI[i]);
        }
    }
    for (i, player) in snap.players.iter().enumerate() {
        if !player.is_bankrupt && player.position == sq {
            return format!("{}{}", SQ_ABBREV[sq as usize], P_EMOJI[i]);
        }
    }
    format!("{}  ", SQ_ABBREV[sq as usize])
}

fn cell_style(sq: u8, owner: Option<u8>, highlight: bool, has_player: bool) -> Style {
    if highlight {
        return Style::default().fg(Color::Black).bg(Color::Yellow).bold();
    }

    let bg = square_bg(sq, owner);
    let fg = match square_kind(sq) {
        SquareKind::Property(g) if owner.is_some() => Color::White,
        SquareKind::Property(g) => group_fg(&g),
        _ if has_player => Color::White,
        _ => Color::Gray,
    };

    Style::default().fg(fg).bg(bg)
}

fn make_cell(sq: u8, snap: &ReplayFrame) -> Span<'static> {
    let owner = snap.owners[sq as usize];
    let is_highlight = snap.highlight == Some(sq);
    let has_player = (0..4).any(|i| !snap.players[i].is_bankrupt && snap.players[i].position == sq);
    let content = cell_content(sq, snap);
    let style = cell_style(sq, owner, is_highlight, has_player);
    Span::styled(format!(" {} ", pad_to(&content, 4)), style)
}

fn player_stat_line(idx: usize, snap: &ReplayFrame) -> Span<'static> {
    let p = &snap.players[idx];
    let is_active = snap.active_player == Some(idx as u8);

    let status = if p.is_bankrupt {
        "BANKRUPT".to_string()
    } else if p.in_jail {
        "Jail".to_string()
    } else {
        square_name(p.position).to_string()
    };

    let _cash_color = if p.cash < 200 {
        Color::Red
    } else if p.cash < 500 {
        Color::Yellow
    } else {
        Color::Green
    };

    let style = if is_active {
        Style::default().add_modifier(Modifier::BOLD)
    } else if p.is_bankrupt {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default()
    };

    let line = format!(
        "⋮ P{} {:<8} ${:<5} {}p {}",
        idx + 1,
        P_NAMES[idx],
        p.cash,
        p.properties.len(),
        status,
    );

    Span::styled(line, style)
}

fn interior_content(row: usize, snap: &ReplayFrame, width: usize) -> Span<'static> {
    match row {
        1..=4 => {
            let stat = player_stat_line(row - 1, snap);
            let content = stat.content.to_string();
            Span::styled(pad_to(&content, width), stat.style)
        }
        6 => Span::raw(pad_to("⋮ ", width)),
        7 => {
            // Dice display
            if let Some((d1, d2, doubles)) = snap.dice_info {
                let active = snap
                    .active_player
                    .map(|p| format!("P{}", p + 1))
                    .unwrap_or_default();
                let dice_str = format!(
                    "{} rolled {}+{}={}{}",
                    active,
                    d1,
                    d2,
                    d1 + d2,
                    if doubles { " ⚡" } else { "" }
                );
                Span::styled(
                    pad_to(&dice_str, width),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                Span::raw(pad_to("⋮ ", width))
            }
        }
        8 => {
            // Event text
            let text = if snap.event_text.is_some() {
                snap.event_text.clone().unwrap_or_default()
            } else {
                snap.last_event_text.clone()
            };
            if text.is_empty() {
                Span::raw(pad_to("⋮ ", width))
            } else {
                let style = if snap.highlight.is_some() {
                    Style::default().fg(Color::DarkGray)
                } else {
                    Style::default().fg(Color::White)
                };
                Span::styled(pad_to(&format!("⋮ {text}"), width), style)
            }
        }
        9 => Span::raw(pad_to("⋮ ", width)),
        _ => Span::raw(pad_to("⋮ ", width)),
    }
}

// ── Render Functions ───────────────────────────────────────────

fn render_board(f: &mut Frame, snap: &ReplayFrame, game: &RecordedGame, area: Rect) {
    let inner_w = area.width.saturating_sub(2) as usize;
    let board_w = 11 * CELL_W;
    let interior_w = inner_w.saturating_sub(2 * CELL_W);

    let mut lines: Vec<Line> = Vec::new();

    for row in 0..11usize {
        let mut spans: Vec<Span> = Vec::new();

        if row == 0 || row == 10 {
            // Top or bottom row: 11 cells
            for col in 0..11 {
                if let Some(sq) = grid_to_sq(row, col) {
                    spans.push(make_cell(sq, snap));
                }
            }
            // Right padding if terminal is wider than board
            let pad = inner_w.saturating_sub(board_w);
            if pad > 0 {
                spans.push(Span::raw(" ".repeat(pad)));
            }
        } else {
            // Side row: left cell + interior + right cell
            if let Some(sq) = grid_to_sq(row, 0) {
                spans.push(make_cell(sq, snap));
            }
            spans.push(interior_content(row, snap, interior_w));
            if let Some(sq) = grid_to_sq(row, 10) {
                spans.push(make_cell(sq, snap));
            }
        }

        lines.push(Line::from(spans));
    }

    let title = format!(
        " Monopoly — seed:{} turns:{} winner:P{} ({}) ",
        game.seed, game.total_turns, game.winner, P_NAMES[game.winner as usize]
    );
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .title_style(Style::default().fg(Color::Cyan).bold());
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_events(f: &mut Frame, game: &RecordedGame, cursor: usize, area: Rect) {
    let vis = area.height.saturating_sub(2) as usize;
    if vis == 0 {
        return;
    }

    // Find current event index from cursor
    let cur_event_idx = game.frames[..=cursor.min(game.frames.len().saturating_sub(1))]
        .iter()
        .rev()
        .find_map(|frame| frame.event_text.as_ref())
        .map(|_| {
            let mut count: usize = 0;
            for (i, frame) in game.frames.iter().enumerate() {
                if frame.event_text.is_some() {
                    count += 1;
                }
                if i == cursor {
                    break;
                }
            }
            count.saturating_sub(1)
        })
        .unwrap_or(0);

    let total_events = game.events.len();
    let start = cur_event_idx.saturating_sub(vis / 2);
    let end = (start + vis).min(total_events);

    let mut lines: Vec<Line> = Vec::new();
    for i in start..end {
        let txt = &game.events[i];
        if i == cur_event_idx {
            lines.push(Line::from(vec![
                Span::styled("▸ ", Style::default().fg(Color::Yellow).bold()),
                Span::styled(txt, Style::default().fg(Color::White).bold()),
            ]));
        } else if i < cur_event_idx {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(txt, Style::default().fg(Color::DarkGray)),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(txt, Style::default().fg(Color::Gray)),
            ]));
        }
    }
    while lines.len() < vis {
        lines.push(Line::from(""));
    }

    // Count walk frames vs total
    let walk_count = game.frames[..=cursor.min(game.frames.len().saturating_sub(1))]
        .iter()
        .filter(|f| f.frame_type == FrameType::Walk)
        .count();
    let event_count = cursor.saturating_sub(walk_count);

    let title = format!(
        " Events ({}/{}) Frame:{}/{} ",
        event_count,
        total_events,
        cursor,
        game.frames.len().saturating_sub(1)
    );
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .title_style(Style::default().fg(Color::Cyan).bold());
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_controls(f: &mut Frame, auto_play: bool, fast_forward: bool, area: Rect) {
    let auto_str = if fast_forward {
        "FF▶▶"
    } else if auto_play {
        "ON ▶"
    } else {
        "OFF ⏸"
    };

    let line = Line::from(vec![
        Span::styled(
            format!(" Auto:{auto_str} "),
            Style::default().fg(if auto_play || fast_forward {
                Color::Yellow
            } else {
                Color::DarkGray
            }),
        ),
        Span::styled("Space/→", Style::default().fg(Color::White)),
        Span::styled(" Next ", Style::default().fg(Color::DarkGray)),
        Span::styled("←", Style::default().fg(Color::White)),
        Span::styled(" Prev ", Style::default().fg(Color::DarkGray)),
        Span::styled("A", Style::default().fg(Color::White)),
        Span::styled(" Auto ", Style::default().fg(Color::DarkGray)),
        Span::styled("F", Style::default().fg(Color::White)),
        Span::styled(" FF ", Style::default().fg(Color::DarkGray)),
        Span::styled("Home/End", Style::default().fg(Color::White)),
        Span::styled(" Jump ", Style::default().fg(Color::DarkGray)),
        Span::styled("Q", Style::default().fg(Color::White)),
        Span::styled(" Quit", Style::default().fg(Color::DarkGray)),
    ]);
    let block = Block::default().borders(Borders::ALL);
    f.render_widget(Paragraph::new(vec![line]).block(block), area);
}

// ── Main ───────────────────────────────────────────────────────

fn main() -> io::Result<()> {
    let seed: u64 = 42;
    let game = record_game(seed);
    println!(
        "\n═══ Monopoly TUI (seed={seed}) ═══ Turns:{} Frames:{} Winner:P{} ({})\n",
        game.total_turns,
        game.frames.len(),
        game.winner,
        P_NAMES[game.winner as usize]
    );

    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    let total = game.frames.len().saturating_sub(1);
    let mut cursor = 0usize;
    let mut auto_play = false;
    let mut fast_forward = false;

    loop {
        let snap = game
            .frames
            .get(cursor)
            .unwrap_or_else(|| game.frames.last().unwrap());

        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(13),   // Board
                    Constraint::Min(5),    // Events
                    Constraint::Length(3), // Controls
                ])
                .split(f.area());

            render_board(f, snap, &game, chunks[0]);
            render_events(f, &game, cursor, chunks[1]);
            render_controls(f, auto_play, fast_forward, chunks[2]);
        })?;

        // Auto-play / fast-forward
        if auto_play || fast_forward {
            let dur = if fast_forward {
                Duration::from_millis(FAST_FWD_MS)
            } else {
                let step_ms = match snap.frame_type {
                    FrameType::Walk => WALK_STEP_MS,
                    FrameType::Event => EVENT_STEP_MS,
                };
                Duration::from_millis(step_ms)
            };

            if event::poll(dur)? {
                if let Event::Key(key) = event::read()?
                    && key.kind == KeyEventKind::Press
                {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char('a') => {
                            auto_play = !auto_play;
                            fast_forward = false;
                        }
                        KeyCode::Char('f') => fast_forward = !fast_forward,
                        KeyCode::Char(' ') => {
                            auto_play = false;
                            fast_forward = false;
                        }
                        _ => {}
                    }
                }
            } else if cursor < total {
                cursor += 1;
            } else {
                auto_play = false;
                fast_forward = false;
            }
            continue;
        }

        // Manual input
        if let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => break,
                KeyCode::Char(' ') | KeyCode::Right | KeyCode::Enter => {
                    cursor = (cursor + 1).min(total);
                }
                KeyCode::Left | KeyCode::Backspace => {
                    cursor = cursor.saturating_sub(1);
                }
                KeyCode::Char('f') => fast_forward = !fast_forward,
                KeyCode::Char('a') => auto_play = !auto_play,
                KeyCode::Home => cursor = 0,
                KeyCode::End => cursor = total,
                _ => {}
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    // Final standings
    if let Some(final_frame) = game.frames.last() {
        println!("\n═══ Final Standings ═══");
        for i in 0..4 {
            let p = &final_frame.players[i];
            let s = if p.is_bankrupt { "BANKRUPT" } else { "Active" };
            println!(
                "  {} P{} {:<10} ${:<6} {} props  {}",
                P_EMOJI[i],
                i,
                P_NAMES[i],
                p.cash,
                p.properties.len(),
                s
            );
        }
        println!(
            "  🏆 Winner: P{} ({})\n",
            game.winner, P_NAMES[game.winner as usize]
        );
    }

    Ok(())
}
