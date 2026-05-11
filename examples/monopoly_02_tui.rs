//! Monopoly HL Arena — Animated TUI Replay (Plan 034)
//!
//! Pre-runs a full Monopoly game, then replays events with ratatui + crossterm.
//! Three-panel layout: board overview, player stats, event log.
//!
//! Controls:
//!   Space / →      — Next event       ← / Backspace — Previous event
//!   F              — Fast forward      A              — Toggle auto-play
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
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::{Frame, Terminal};

use microgpt_rs::pruners::monopoly::{
    CardEffect, GameEvent, GreedyPlayer, HLPlayer, JailReason, MonopolyPlayer, PropertyGroup,
    RandomPlayer, ReleaseMethod, SquareKind, TaxKind, ValidatorPlayer, run_game, square_kind,
    square_name,
};

const P_EMOJI: [&str; 4] = ["🎲", "💰", "🛡️", "🧠"];
const P_NAMES: [&str; 4] = ["Random", "Greedy", "Validator", "HL"];
const MAX_TURNS: u32 = 200;
const AUTO_STEP_MS: u64 = 250;

#[derive(Clone)]
struct PlayerState {
    cash: u32,
    position: u8,
    properties: Vec<u8>,
    is_bankrupt: bool,
    in_jail: bool,
}

impl PlayerState {
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

struct AppState {
    events: Vec<GameEvent>,
    event_index: usize,
    seed: u64,
    auto_play: bool,
    fast_forward: bool,
    winner: u8,
    total_turns: u32,
}

impl AppState {
    fn new(seed: u64) -> Self {
        let mut rng = Rng::with_seed(seed);
        let mut players: [Box<dyn MonopolyPlayer>; 4] = [
            Box::new(RandomPlayer::new(0)),
            Box::new(GreedyPlayer::new(1)),
            Box::new(ValidatorPlayer::new(2)),
            Box::new(HLPlayer::new(3)),
        ];
        let result = run_game(seed, &mut players, &mut rng, MAX_TURNS);
        Self {
            events: result.events,
            event_index: 0,
            seed,
            auto_play: false,
            fast_forward: false,
            winner: result.winner,
            total_turns: result.total_turns,
        }
    }

    /// Replay events up to current index, return player states and square owners.
    fn replay_to(&self, up_to: usize) -> ([PlayerState; 4], [Option<u8>; 40]) {
        let mut ps = [
            PlayerState::new(),
            PlayerState::new(),
            PlayerState::new(),
            PlayerState::new(),
        ];
        let mut owners: [Option<u8>; 40] = [None; 40];
        for i in 0..up_to.min(self.events.len()) {
            match &self.events[i] {
                GameEvent::PlayerMoved {
                    player,
                    to,
                    passed_go,
                    ..
                } => {
                    let p = *player as usize;
                    ps[p].position = *to;
                    if *passed_go {
                        ps[p].cash += 200;
                    }
                }
                GameEvent::SalaryCollected { player, amount } => {
                    ps[*player as usize].cash += amount;
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
                        ps[*player as usize].cash =
                            ps[*player as usize].cash.saturating_sub(*a * 3);
                    }
                    CardEffect::MoveTo(pos) => ps[*player as usize].position = *pos,
                    CardEffect::GoToJail => {
                        ps[*player as usize].position = 10;
                        ps[*player as usize].in_jail = true;
                    }
                    CardEffect::GetOutOfJailFree => ps[*player as usize].in_jail = false,
                    _ => {}
                },
                GameEvent::HouseBuilt {
                    player,
                    square,
                    houses,
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
                    let p = *player as usize;
                    ps[p].cash = ps[p].cash.saturating_sub(cost * *houses as u32);
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
        (ps, owners)
    }
}

fn fmt_card(effect: &CardEffect) -> String {
    match effect {
        CardEffect::CollectMoney(a) => format!("Collect ${a}"),
        CardEffect::PayMoney(a) => format!("Pay ${a}"),
        CardEffect::PayPerHouse { house, hotel } => format!("Pay ${house}/house, ${hotel}/hotel"),
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
        GameEvent::TurnStarted { player } => format!("P{player}'s turn started"),
        GameEvent::DiceRolled {
            player,
            die1,
            die2,
            doubles,
        } => {
            let sum = die1 + die2;
            let d = if *doubles { " (doubles!)" } else { "" };
            format!("P{player} rolled {die1}+{die2}={sum}{d}")
        }
        GameEvent::PlayerMoved {
            player,
            to,
            passed_go,
            ..
        } => {
            let go = if *passed_go { " (passed GO)" } else { "" };
            format!("P{player} moved to {}{go}", square_name(*to))
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
        } => format!(
            "P{winner} won auction for {} at ${price}",
            square_name(*square)
        ),
        GameEvent::PropertyDeclined { player, square } => {
            format!("P{player} declined {}", square_name(*square))
        }
        GameEvent::RentPaid {
            payer,
            payee,
            amount,
            square,
        } => format!(
            "P{payer} paid ${amount} rent to P{payee} ({})",
            square_name(*square)
        ),
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
        } => format!(
            "P{player} built on {} (now {houses} houses)",
            square_name(*square)
        ),
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
            ..
        } => {
            format!("P{responder} accepted trade from P{proposer}")
        }
        GameEvent::TradeDeclined {
            proposer,
            responder,
            ..
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
            format!("P{player} released from jail ({how})")
        }
        GameEvent::PlayerBankrupt { player, creditor } => {
            let c = creditor.map_or(String::new(), |c| format!(" (debts to P{c})"));
            format!("P{player} went BANKRUPT{c}")
        }
        GameEvent::GameOver { winner } => format!(
            "🏆 Game Over! P{winner} ({}) wins!",
            P_NAMES[*winner as usize]
        ),
        GameEvent::AuctionStarted { square } => {
            format!("Auction started for {}", square_name(*square))
        }
        GameEvent::AuctionBid { player, amount } => format!("P{player} bid ${amount}"),
        GameEvent::AuctionWon {
            player,
            square,
            amount,
        } => format!(
            "P{player} won {} at auction for ${amount}",
            square_name(*square)
        ),
    }
}

fn group_color(group: &PropertyGroup) -> Color {
    match group {
        PropertyGroup::Brown => Color::Rgb(139, 90, 43),
        PropertyGroup::LightBlue => Color::Cyan,
        PropertyGroup::Pink => Color::Magenta,
        PropertyGroup::Orange => Color::Rgb(255, 165, 0),
        PropertyGroup::Red => Color::Red,
        PropertyGroup::Yellow => Color::Yellow,
        PropertyGroup::Green => Color::Green,
        PropertyGroup::DarkBlue => Color::Blue,
    }
}

fn render_board(f: &mut Frame, owners: &[Option<u8>; 40], area: Rect) {
    let mut spans: Vec<Span> = Vec::new();
    for i in 0..40u8 {
        if i > 0 {
            spans.push(Span::raw(" → "));
        }
        let name = square_name(i);
        let style = match (&square_kind(i), owners[i as usize]) {
            (SquareKind::Property(g), Some(_)) => Style::default().fg(group_color(g)).bold(),
            (SquareKind::Property(g), None) => Style::default().fg(group_color(g)),
            (SquareKind::Go, _) => Style::default().fg(Color::Yellow).bold(),
            (SquareKind::FreeParking, _) => Style::default().fg(Color::Green),
            (SquareKind::GoToJail, _) => Style::default().fg(Color::Red),
            (SquareKind::Jail, _) | (SquareKind::Railroad, _) | (SquareKind::Utility, _) => {
                Style::default().fg(Color::White)
            }
            _ => Style::default().fg(Color::DarkGray),
        };
        let emoji = match owners[i as usize] {
            Some(o) => P_EMOJI[o as usize],
            None => "",
        };
        spans.push(Span::styled(format!("{name}{emoji}"), style));
    }

    // Break spans into lines that fit width
    let mut lines: Vec<Line> = Vec::new();
    let mut cur: Vec<Span> = Vec::new();
    let mut w = 0usize;
    let max_w = area.width.saturating_sub(2) as usize;
    for span in spans {
        let sw = span.width();
        if w + sw > max_w && !cur.is_empty() {
            lines.push(Line::from(cur.clone()));
            cur.clear();
            w = 0;
        }
        cur.push(span);
        w += sw;
    }
    if !cur.is_empty() {
        lines.push(Line::from(cur));
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Board ")
        .title_style(Style::default().fg(Color::Cyan).bold());
    f.render_widget(
        Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn render_players(f: &mut Frame, ps: &[PlayerState; 4], area: Rect) {
    let mut lines: Vec<Line> = Vec::new();
    for i in 0..4 {
        let p = &ps[i];
        let status = if p.is_bankrupt {
            Span::styled("BANKRUPT", Style::default().fg(Color::Red).bold())
        } else if p.in_jail {
            Span::styled("In Jail", Style::default().fg(Color::Yellow))
        } else {
            Span::styled(
                format!("At {}", square_name(p.position)),
                Style::default().fg(Color::Green),
            )
        };
        let cash_fg = if p.cash < 200 {
            Color::Red
        } else if p.cash < 500 {
            Color::Yellow
        } else {
            Color::Green
        };
        lines.push(Line::from(vec![
            Span::styled(format!("{} P{} ", P_EMOJI[i], i), Style::default().bold()),
            Span::styled(
                format!("{:<10}", P_NAMES[i]),
                Style::default().fg(Color::White),
            ),
            Span::styled(
                format!("${:<6} ", p.cash),
                Style::default().fg(cash_fg).bold(),
            ),
            Span::styled(
                format!("{} props  ", p.properties.len()),
                Style::default().fg(Color::Cyan),
            ),
            status,
        ]));
    }
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Players ")
        .title_style(Style::default().fg(Color::Cyan).bold());
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_events(f: &mut Frame, state: &AppState, area: Rect) {
    let total = state.events.len();
    let cur = state.event_index;
    let vis = area.height.saturating_sub(2) as usize;
    let start = cur.saturating_sub(vis / 2);
    let end = (start + vis).min(total);

    let mut lines: Vec<Line> = Vec::new();
    for i in start..end {
        let txt = fmt_event(&state.events[i]);
        if i == cur {
            lines.push(Line::from(vec![
                Span::styled("▸ ", Style::default().fg(Color::Yellow).bold()),
                Span::styled(txt, Style::default().fg(Color::White).bold()),
            ]));
        } else {
            let fg = if i < cur {
                Color::DarkGray
            } else {
                Color::Gray
            };
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(txt, Style::default().fg(fg)),
            ]));
        }
    }
    while lines.len() < vis {
        lines.push(Line::from(""));
    }

    let title = format!(" Events ({cur}/{total}) ");
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .title_style(Style::default().fg(Color::Cyan).bold());
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_controls(f: &mut Frame, area: Rect, state: &AppState) {
    let a = if state.auto_play { "ON" } else { "OFF" };
    let line = Line::from(vec![
        Span::styled(format!(" Auto:{a} "), Style::default().fg(Color::Yellow)),
        Span::styled("Space/→", Style::default().fg(Color::White)),
        Span::styled(" Next ", Style::default().fg(Color::DarkGray)),
        Span::styled("←", Style::default().fg(Color::White)),
        Span::styled(" Prev ", Style::default().fg(Color::DarkGray)),
        Span::styled("F", Style::default().fg(Color::White)),
        Span::styled(" End ", Style::default().fg(Color::DarkGray)),
        Span::styled("A", Style::default().fg(Color::White)),
        Span::styled(" Toggle ", Style::default().fg(Color::DarkGray)),
        Span::styled("Home/End", Style::default().fg(Color::White)),
        Span::styled(" Jump ", Style::default().fg(Color::DarkGray)),
        Span::styled("Q", Style::default().fg(Color::White)),
        Span::styled(" Quit", Style::default().fg(Color::DarkGray)),
    ]);
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(
            " Seed:{} Turns:{} Winner:P{} ",
            state.seed, state.total_turns, state.winner
        ))
        .title_style(Style::default().fg(Color::DarkGray));
    f.render_widget(Paragraph::new(vec![line]).block(block), area);
}

fn main() -> io::Result<()> {
    let seed: u64 = 42;
    let state = AppState::new(seed);
    println!(
        "\n═══ Monopoly (seed={seed}) ═══ Turns:{} Events:{} Winner:P{} ({})\n",
        state.total_turns,
        state.events.len(),
        state.winner,
        P_NAMES[state.winner as usize]
    );

    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    let mut state = state;
    let total = state.events.len().saturating_sub(1);
    loop {
        let (ps, owners) = state.replay_to(state.event_index);
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(7),    // Board
                    Constraint::Length(6), // Players
                    Constraint::Min(10),   // Events
                    Constraint::Length(3), // Controls
                ])
                .split(f.area());
            render_board(f, &owners, chunks[0]);
            render_players(f, &ps, chunks[1]);
            render_events(f, &state, chunks[2]);
            render_controls(f, chunks[3], &state);
        })?;

        // Auto-play / fast-forward
        if state.auto_play || state.fast_forward {
            let dur = if state.fast_forward {
                Duration::from_millis(10)
            } else {
                Duration::from_millis(AUTO_STEP_MS)
            };
            if event::poll(dur)? {
                if let Event::Key(key) = event::read()?
                    && key.kind == KeyEventKind::Press
                {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char('a') => {
                            state.auto_play = !state.auto_play;
                            state.fast_forward = false;
                        }
                        KeyCode::Char('f') => state.fast_forward = !state.fast_forward,
                        KeyCode::Char(' ') => {
                            state.auto_play = false;
                            state.fast_forward = false;
                        }
                        _ => {}
                    }
                }
            } else if state.event_index < total {
                state.event_index += 1;
            } else {
                state.auto_play = false;
                state.fast_forward = false;
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
                    state.event_index = (state.event_index + 1).min(total);
                }
                KeyCode::Left | KeyCode::Backspace => {
                    state.event_index = state.event_index.saturating_sub(1)
                }
                KeyCode::Char('f') => state.event_index = total,
                KeyCode::Char('a') => state.auto_play = !state.auto_play,
                KeyCode::Home => state.event_index = 0,
                KeyCode::End => state.event_index = total,
                _ => {}
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    let (final_ps, _) = state.replay_to(state.events.len());
    println!("\n═══ Final Standings ═══");
    for i in 0..4 {
        let p = &final_ps[i];
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
        state.winner, P_NAMES[state.winner as usize]
    );

    Ok(())
}
