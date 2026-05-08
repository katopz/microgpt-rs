//! Blue Bear Tactical Puzzle — Interactive TUI Solver
//!
//! Demonstrates using Speculative Decoding (DDTree) with ConstraintPruner
//! as a heavily constrained state-space solver.
//!
//! Controls:
//!   ← / Backspace / P  — Previous step
//!   → / Enter / N      — Next step
//!   Home                — Jump to start
//!   End                 — Jump to end
//!   R                   — Restart solver
//!   Q / Esc             — Quit
//!
//! Run: `cargo run --example blue_bear_tui`

use std::io::{self, Stdout};
use std::time::Instant;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::{Frame, Terminal};

use microgpt_rs::pruners::tactical_pruner::{GameState, TacticalPruner};
use microgpt_rs::speculative::{build_dd_tree_pruned, extract_parent_tokens};
use microgpt_rs::types::Config;

// ── Emoji Map ──────────────────────────────────────────────────

const BEAR: &str = "🐻";
const MONSTER_LIVE: &str = "👹";
const MONSTER_DEAD: &str = "💀";
const TREASURE: &str = "💎";
const GOAL: &str = "🚪";
const GOAL_OPEN: &str = "🏆";
const WALL: &str = "🧱";
const FLOOR: &str = "⬜";
const ITEM: &str = "🔑";
const CHECK: &str = "✓";
const CROSS: &str = "✗";

// ── Action Helpers ─────────────────────────────────────────────

fn action_icon(action: usize) -> &'static str {
    match action {
        0 => "↑",
        1 => "↓",
        2 => "←",
        3 => "→",
        4 => "⚔",
        _ => "?",
    }
}

fn action_name(action: usize) -> &'static str {
    match action {
        0 => "Up",
        1 => "Down",
        2 => "Left",
        3 => "Right",
        4 => "Attack",
        _ => "???",
    }
}

// ── Puzzle Map ─────────────────────────────────────────────────
//
// B X T    Bear at (0,0), X = Monster+Treasure at (0,1), Treasure at (0,2)
// # M G    Wall at (1,0), Monster at (1,1), Goal at (1,2)
//
// Solution: → ⚔ ↓ ⚔ ↑ → ↓  (7 steps)
// Note: DDTree packs 16 bits/token into u128 → max lookahead = 8.

const MAP: &str = "\
B X T
# M G";

// ── App State ──────────────────────────────────────────────────

struct App {
    pruner: TacticalPruner,
    solution: Vec<usize>,
    states: Vec<GameState>,
    current: usize,
    solved: bool,
    solve_time_ms: u64,
    tree_nodes: usize,
}

impl App {
    fn new() -> Self {
        let pruner = TacticalPruner::new(MAP);

        let mut config = Config::draft();
        config.vocab_size = 5; // Up, Down, Left, Right, Attack
        config.draft_lookahead = 8; // u128/16 = 8 tokens max
        config.tree_budget = 10000;

        // Uniform marginals: DDTree operates as BFS / Best-First search
        let marginals = vec![vec![0.2f32; 5]; config.draft_lookahead];
        let refs: Vec<&[f32]> = marginals.iter().map(|v| v.as_slice()).collect();

        let start = Instant::now();
        let tree = build_dd_tree_pruned(&refs, &config, &pruner, false);
        let solve_time = start.elapsed();

        // Find first path that reaches the goal
        let mut solution_path = None;
        for node in &tree {
            let path = extract_parent_tokens(node.parent_path, node.depth + 1);
            if let Some(state) = pruner.replay_state(&path)
                && (state.r, state.c) == pruner.goal
            {
                solution_path = Some(path);
                break;
            }
        }

        let solution = match solution_path {
            Some(p) => p,
            None => panic!("Puzzle should be solvable within lookahead"),
        };

        // Structural assertions
        assert!(!tree.is_empty(), "Tree should contain nodes after pruning");
        assert_eq!(
            solution.len(),
            7,
            "Expected 7-step solution for BXT/SMG map"
        );

        // Verify expected solution: → ⚔ ↓ ⚔ ↑ → ↓
        let expected = [3, 4, 1, 4, 0, 3, 1]; // Right, Attack, Down, Attack, Up, Right, Down
        assert_eq!(
            solution, expected,
            "Solution should match expected action sequence"
        );

        // Pre-compute all intermediate states: states[0] = initial, states[n] = after action n-1
        let mut states = vec![pruner.initial_state()];
        let mut state = pruner.initial_state();
        for &action in &solution {
            state = pruner.apply_action(&state, action).unwrap();
            states.push(state.clone());
        }

        // Verify final state
        let all_treasures = (1 << pruner.treasures.len()) - 1;
        let final_state = states.last().unwrap();
        assert_eq!(
            (final_state.r, final_state.c),
            pruner.goal,
            "Bear must be at goal"
        );
        assert_eq!(
            final_state.collected_treasures, all_treasures,
            "All treasures must be collected"
        );
        assert_eq!(
            final_state.killed_monsters, all_treasures,
            "All monsters must be killed"
        );
        assert_eq!(
            final_state.inventory, 0,
            "Inventory should be empty at goal"
        );

        Self {
            pruner,
            solution,
            states,
            current: 0,
            solved: true,
            solve_time_ms: solve_time.as_millis() as u64,
            tree_nodes: tree.len(),
        }
    }

    fn restart(&mut self) {
        *self = Self::new();
    }

    fn current_state(&self) -> &GameState {
        &self.states[self.current]
    }

    fn total_steps(&self) -> usize {
        self.solution.len()
    }

    fn is_at_start(&self) -> bool {
        self.current == 0
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.total_steps()
    }
}

// ── Entry Point ────────────────────────────────────────────────

fn main() -> io::Result<()> {
    let mut terminal = setup()?;
    let res = run(&mut terminal);
    teardown(&mut terminal)?;
    res
}

fn setup() -> io::Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    Terminal::new(CrosstermBackend::new(stdout))
}

fn teardown(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn run(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
    let mut app = App::new();

    loop {
        terminal.draw(|f| draw(f, &app))?;

        if let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                KeyCode::Char('r') => app.restart(),
                KeyCode::Right | KeyCode::Enter | KeyCode::Char('n') => {
                    if !app.is_at_end() {
                        app.current += 1;
                    }
                }
                KeyCode::Left | KeyCode::Backspace | KeyCode::Char('p') => {
                    if !app.is_at_start() {
                        app.current -= 1;
                    }
                }
                KeyCode::Home => app.current = 0,
                KeyCode::End => app.current = app.total_steps(),
                _ => {}
            }
        }
    }
}

// ── Drawing ────────────────────────────────────────────────────

fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // title bar
            Constraint::Min(12),   // main content
            Constraint::Length(3), // navigation bar
        ])
        .split(area);

    draw_title(f, chunks[0], app);
    draw_content(f, chunks[1], app);
    draw_nav(f, chunks[2], app);
}

fn draw_title(f: &mut Frame, area: Rect, app: &App) {
    let status = if app.solved {
        format!(
            "Solved in {} steps · {}ms · {} nodes",
            app.solution.len(),
            app.solve_time_ms,
            thousands(app.tree_nodes),
        )
    } else {
        "No solution found".into()
    };

    let line = Line::from(vec![
        Span::styled(
            " 🐻 Blue Bear ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!(" {status} "), Style::default().fg(Color::Cyan)),
        Span::styled(
            " ← → Step · R Restart · Q Quit ",
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    let para = Paragraph::new(line).block(Block::default().borders(Borders::ALL));
    f.render_widget(para, area);
}

fn draw_content(f: &mut Frame, area: Rect, app: &App) {
    let grid_rows = app.pruner.grid.len() as u16;

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(26), Constraint::Min(34)])
        .split(area);

    // Left column: map (fits grid exactly) + legend
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(grid_rows + 2), Constraint::Min(6)])
        .split(cols[0]);

    draw_map(f, left[0], app);
    draw_legend(f, left[1]);

    // Right column: state + solution
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(9), Constraint::Min(8)])
        .split(cols[1]);

    draw_state(f, right[0], app);
    draw_solution(f, right[1], app);
}

fn draw_map(f: &mut Frame, area: Rect, app: &App) {
    let state = app.current_state();
    let pruner = &app.pruner;

    let mut lines: Vec<Line> = Vec::new();
    for (r, row) in pruner.grid.iter().enumerate() {
        let mut spans: Vec<Span> = Vec::new();
        for c in 0..row.len() {
            let emoji = cell_emoji(pruner, state, r, c);
            spans.push(Span::raw(format!("{emoji} ")));
        }
        lines.push(Line::from(spans));
    }

    let para = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" 🗺 Map "));
    f.render_widget(para, area);
}

fn cell_emoji(pruner: &TacticalPruner, state: &GameState, r: usize, c: usize) -> String {
    // Player always renders on top
    if state.r == r && state.c == c {
        return BEAR.into();
    }

    // Live monster
    for (i, &(mr, mc)) in pruner.monsters.iter().enumerate() {
        if (mr, mc) == (r, c) && (state.killed_monsters & (1 << i)) == 0 {
            return MONSTER_LIVE.into();
        }
    }

    // Dropped item on floor (from killed monster, not yet picked up)
    for (i, &(mr, mc)) in pruner.monsters.iter().enumerate() {
        if (mr, mc) == (r, c) && (state.dropped_items & (1 << i)) != 0 {
            return ITEM.into();
        }
    }

    // Uncollected treasure
    for (i, &(tr, tc)) in pruner.treasures.iter().enumerate() {
        if (tr, tc) == (r, c) && (state.collected_treasures & (1 << i)) == 0 {
            return TREASURE.into();
        }
    }

    // Goal / Exit
    if pruner.goal == (r, c) {
        let all = (1 << pruner.treasures.len()) - 1;
        return if state.collected_treasures == all {
            GOAL_OPEN.into()
        } else {
            GOAL.into()
        };
    }

    // Wall
    if pruner.grid[r][c] == '#' {
        return WALL.into();
    }

    // Dead monster (killed, item already picked up)
    for (i, &(mr, mc)) in pruner.monsters.iter().enumerate() {
        if (mr, mc) == (r, c) && (state.killed_monsters & (1 << i)) != 0 {
            return MONSTER_DEAD.into();
        }
    }

    FLOOR.into()
}

fn draw_legend(f: &mut Frame, area: Rect) {
    let lines = vec![
        Line::from(vec![
            Span::raw(format!(" {BEAR} You    ")),
            Span::raw(format!("{MONSTER_LIVE} Monster ")),
        ]),
        Line::from(vec![
            Span::raw(format!(" {TREASURE} Treasure")),
            Span::raw(format!(" {GOAL} Exit    ")),
        ]),
        Line::from(vec![
            Span::raw(format!(" {WALL} Wall    ")),
            Span::raw(format!("{FLOOR} Floor   ")),
        ]),
        Line::from(vec![
            Span::raw(format!(" {ITEM} Item    ")),
            Span::raw(format!("{MONSTER_DEAD} Dead    ")),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            " Kill 👹 → 🔑 → unlock 💎",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let para =
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" Legend "));
    f.render_widget(para, area);
}

fn draw_state(f: &mut Frame, area: Rect, app: &App) {
    let state = app.current_state();
    let step_label = if app.is_at_start() {
        "Start".into()
    } else {
        format!("{}/{}", app.current, app.total_steps())
    };

    let inv_display: String = if state.inventory == 0 {
        "(empty)".into()
    } else {
        (0..state.inventory).map(|_| ITEM).collect()
    };

    let monsters: String = app
        .pruner
        .monsters
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if (state.killed_monsters & (1 << i)) != 0 {
                format!("{MONSTER_LIVE}{CHECK} ")
            } else {
                format!("{MONSTER_LIVE}{CROSS} ")
            }
        })
        .collect();

    let treasures: String = app
        .pruner
        .treasures
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if (state.collected_treasures & (1 << i)) != 0 {
                format!("{TREASURE}{CHECK} ")
            } else {
                format!("{TREASURE}{CROSS} ")
            }
        })
        .collect();

    let lines = vec![
        Line::from(vec![
            Span::styled("  Step:      ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                step_label,
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Position:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("({}, {})", state.r, state.c),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Inventory: ", Style::default().fg(Color::DarkGray)),
            Span::styled(inv_display, Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled("  Monsters:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(monsters, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  Treasures: ", Style::default().fg(Color::DarkGray)),
            Span::styled(treasures, Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            format!("  Slots: {}/2", state.inventory),
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let para =
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" 📊 State "));
    f.render_widget(para, area);
}

fn draw_solution(f: &mut Frame, area: Rect, app: &App) {
    let mut lines: Vec<Line> = Vec::new();

    for (i, &action) in app.solution.iter().enumerate() {
        let step_num = i + 1;
        let icon = action_icon(action);
        let name = action_name(action);
        let is_current = step_num == app.current;
        let is_done = step_num < app.current;

        let style = if is_current {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else if is_done {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        };

        let marker = if is_current { "▶" } else { " " };

        lines.push(Line::from(vec![
            Span::styled(format!("{marker} "), style),
            Span::styled(format!("{step_num:>2}. "), style),
            Span::styled(format!("{icon} "), style),
            Span::styled(name, style),
        ]));
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "  No solution found.",
            Style::default().fg(Color::Red),
        )));
    }

    let title = format!(" 📋 Solution ({} steps) ", app.total_steps());
    let para = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

fn draw_nav(f: &mut Frame, area: Rect, app: &App) {
    let total = app.total_steps();
    let cur = app.current;

    let back_style = if app.is_at_start() {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD)
    };

    let next_style = if app.is_at_end() {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD)
    };

    let center = if total == 0 {
        "No solution".into()
    } else if app.is_at_start() {
        "▶ Start ◀".into()
    } else if app.is_at_end() {
        format!("🎉 Step {cur}/{total} — Solved!")
    } else {
        let icon = action_icon(app.solution[cur - 1]);
        let name = action_name(app.solution[cur - 1]);
        format!("Step {cur}/{total} — {icon} {name}")
    };

    let line = Line::from(vec![
        Span::styled(" ◀ Back ", back_style),
        Span::styled(
            format!("   {center}   "),
            Style::default().fg(Color::Yellow),
        ),
        Span::styled(" Next ▶ ", next_style),
    ]);

    let para = Paragraph::new(line).block(Block::default().borders(Borders::ALL));
    f.render_widget(para, area);
}

// ── Helpers ────────────────────────────────────────────────────

fn thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}
