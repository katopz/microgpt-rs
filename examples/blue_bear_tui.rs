//! Blue Bear Tactical Puzzle — Animated TUI Solver
//!
//! Features:
//! - Animated movement with terrain-cost-based speed
//! - Larger map tiles with cost indicators and walking corridors
//! - Next/Back navigation with smooth animation
//! - Auto-play mode (Space)
//!
//! Controls:
//!   ← / Backspace / P  — Previous step (instant)
//!   → / Enter / N      — Next step (animated)
//!   . (period)          — Skip to next (instant)
//!   Space               — Toggle auto-play
//!   Home                — Jump to start
//!   End                 — Jump to end
//!   R                   — Restart solver
//!   Q / Esc             — Quit
//!
//! Run: `cargo run --example blue_bear_tui`

use std::io::{self, Stdout};
use std::time::{Duration, Instant};

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

// ── Emoji Constants ────────────────────────────────────────────

const BEAR: &str = "🐻";
const MONSTER_LIVE: &str = "👹";
const MONSTER_DEAD: &str = "💀";
const TREASURE: &str = "💎";
const GOAL: &str = "🚪";
const GOAL_OPEN: &str = "🏆";
const WALL: &str = "🧱";
const FLOOR: &str = "⬜";
const SAND: &str = "🟨";
const WATER: &str = "🟦";
const ITEM: &str = "🔑";
const SWORD: &str = "⚔";
const CHECK: &str = "✓";
const CROSS: &str = "✗";

// ── Timing Constants ───────────────────────────────────────────

const TICK_MS: u64 = 50; // 20 FPS animation tick
const MS_PER_COST: u64 = 150; // milliseconds per terrain cost unit
const ATTACK_MS: u64 = 200; // fixed attack animation duration

// ── Helper Functions ───────────────────────────────────────────

fn action_icon(action: usize) -> &'static str {
    match action {
        0 => "↑",
        1 => "↓",
        2 => "←",
        3 => "→",
        4 => SWORD,
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

fn terrain_emoji(grid: &[Vec<char>], r: usize, c: usize) -> &'static str {
    match grid[r][c] {
        '#' => WALL,
        '~' => SAND,
        'w' => WATER,
        _ => FLOOR,
    }
}

fn terrain_label(grid: &[Vec<char>], r: usize, c: usize) -> &'static str {
    match grid[r][c] {
        '~' => "Sand",
        'w' => "Water",
        '#' => "Wall",
        _ => "Grass",
    }
}

// ── Puzzle Map ─────────────────────────────────────────────────
//
// B . ~ T    Bear(0,0), Sand(0,2) cost 2, Treasure(0,3)
// . M # .    Monster(1,1), Wall(1,2)
// . . w G    Water(2,2) cost 3, Goal(2,3)
//
// Solution: → ↓ ⚔ ↑ → → ↓ ↓  (8 steps, total cost 8)
// Action ids: [3, 1, 4, 0, 3, 3, 1, 1]

const MAP: &str = "\
B . ~ T
. M # .
. . w G";

// ── Animation State ────────────────────────────────────────────

struct AnimState {
    from: (usize, usize),
    to: (usize, usize),
    action: usize,
    start: Instant,
    duration_ms: u64,
}

// ── Application State ──────────────────────────────────────────

struct App {
    pruner: TacticalPruner,
    solution: Vec<usize>,
    states: Vec<GameState>,
    current: usize,
    anim: Option<AnimState>,
    auto_play: bool,
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

        let solution = solution_path.expect("Puzzle should be solvable within lookahead");

        // Pre-compute all intermediate states: states[0]=initial, states[n]=after action n-1
        let mut states = vec![pruner.initial_state()];
        let mut state = pruner.initial_state();
        for &action in &solution {
            state = pruner.apply_action(&state, action).unwrap();
            states.push(state.clone());
        }

        // Verify final state
        let final_state = states.last().unwrap();
        assert_eq!(
            (final_state.r, final_state.c),
            pruner.goal,
            "Bear must be at goal"
        );

        Self {
            pruner,
            solution,
            states,
            current: 0,
            anim: None,
            auto_play: false,
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

    /// Begin animating the next step.
    fn start_animation(&mut self) {
        if self.is_at_end() || self.anim.is_some() {
            return;
        }

        let action = self.solution[self.current];
        let from_state = &self.states[self.current];
        let to_state = &self.states[self.current + 1];

        let duration_ms = if action == 4 {
            ATTACK_MS
        } else {
            let cost = self.pruner.terrain_cost(to_state.r, to_state.c);
            cost as u64 * MS_PER_COST
        };

        self.anim = Some(AnimState {
            from: (from_state.r, from_state.c),
            to: (to_state.r, to_state.c),
            action,
            start: Instant::now(),
            duration_ms,
        });
    }

    /// Advance animation by one tick. Returns true when animation completes.
    fn tick_animation(&mut self) -> bool {
        let Some(ref anim) = self.anim else {
            return false;
        };

        if anim.start.elapsed().as_millis() as u64 >= anim.duration_ms {
            self.anim = None;
            self.current += 1;
            return true;
        }
        false
    }

    /// Returns animated bear position as (row, col, progress) during animation.
    fn bear_anim_position(&self) -> Option<(f32, f32, f32)> {
        let anim = self.anim.as_ref()?;
        let elapsed = anim.start.elapsed().as_millis() as f32;
        let progress = (elapsed / anim.duration_ms as f32).min(1.0);

        let row = anim.from.0 as f32 + (anim.to.0 as f32 - anim.from.0 as f32) * progress;
        let col = anim.from.1 as f32 + (anim.to.1 as f32 - anim.from.1 as f32) * progress;

        Some((row, col, progress))
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

// ── Main Loop ──────────────────────────────────────────────────

fn run(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
    let mut app = App::new();

    loop {
        terminal.draw(|f| draw(f, &app))?;

        // Tick animation forward
        let completed = app.tick_animation();
        if completed && app.auto_play && !app.is_at_end() {
            app.start_animation();
        }
        // Stop auto-play when reaching the end
        if app.is_at_end() {
            app.auto_play = false;
        }

        // Poll for input (faster during animation, slower when idle)
        let timeout = if app.anim.is_some() || app.auto_play {
            Duration::from_millis(TICK_MS)
        } else {
            Duration::from_millis(100)
        };

        if event::poll(timeout)?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            handle_key(&mut app, key.code);
        }
    }
}

fn handle_key(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Char('q') | KeyCode::Esc => std::process::exit(0),
        KeyCode::Char('r') => app.restart(),

        // Next step (animated)
        KeyCode::Right | KeyCode::Enter | KeyCode::Char('n') => {
            if app.anim.is_none() && !app.is_at_end() {
                app.start_animation();
            }
        }

        // Skip (instant)
        KeyCode::Char('.') => {
            if app.anim.is_none() && !app.is_at_end() {
                app.current += 1;
            }
        }

        // Previous step (instant)
        KeyCode::Left | KeyCode::Backspace | KeyCode::Char('p') => {
            if app.anim.is_none() && !app.is_at_start() {
                app.current -= 1;
            }
        }

        // Toggle auto-play
        KeyCode::Char(' ') => {
            app.auto_play = !app.auto_play;
            if app.auto_play && app.anim.is_none() && !app.is_at_end() {
                app.start_animation();
            }
        }

        KeyCode::Home => {
            app.anim = None;
            app.auto_play = false;
            app.current = 0;
        }
        KeyCode::End => {
            app.anim = None;
            app.auto_play = false;
            app.current = app.total_steps();
        }
        _ => {}
    }
}

// ── Drawing ────────────────────────────────────────────────────

fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // title bar
            Constraint::Min(10),   // main content
            Constraint::Length(3), // navigation bar
        ])
        .split(area);

    draw_title(f, chunks[0], app);
    draw_content(f, chunks[1], app);
    draw_nav(f, chunks[2], app);
}

fn draw_title(f: &mut Frame, area: Rect, app: &App) {
    let final_cost = app.states.last().map_or(0, |s| s.total_cost);
    let status = if app.solved {
        format!(
            "Solved: {} steps · Cost {} · {}ms · {} nodes",
            app.solution.len(),
            final_cost,
            app.solve_time_ms,
            thousands(app.tree_nodes),
        )
    } else {
        "No solution found".into()
    };

    let auto_indicator = if app.auto_play { " ⏵AUTO" } else { "" };

    let line = Line::from(vec![
        Span::styled(
            " 🐻 Blue Bear ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" {status}{auto_indicator} "),
            Style::default().fg(Color::Cyan),
        ),
        Span::styled(
            " ← → · Space Auto · Q Quit ",
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    let para = Paragraph::new(line).block(Block::default().borders(Borders::ALL));
    f.render_widget(para, area);
}

fn draw_content(f: &mut Frame, area: Rect, app: &App) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(30), Constraint::Min(36)])
        .split(area);

    // Left column: map + legend
    let grid_rows = app.pruner.grid.len() as u16;
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(grid_rows + 2), Constraint::Min(6)])
        .split(cols[0]);

    draw_map(f, left[0], app);
    draw_legend(f, left[1], app);

    // Right column: state + solution
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(10), Constraint::Min(8)])
        .split(cols[1]);

    draw_state(f, right[0], app);
    draw_solution(f, right[1], app);
}

// ── Map Rendering with Animation ──────────────────────────────

fn draw_map(f: &mut Frame, area: Rect, app: &App) {
    let state = app.current_state();
    let pruner = &app.pruner;
    let anim_pos = app.bear_anim_position();

    let mut lines = Vec::new();

    for r in 0..pruner.grid.len() {
        let mut spans: Vec<Span> = Vec::new();

        for c in 0..pruner.grid[0].len() {
            let (bear_here, bear_in_gap) = bear_render_state(anim_pos, state, r, c);

            // Cell content: bear or base entity/terrain
            let emoji = if bear_here {
                // During attack animation, show bear with sword indicator
                if app.anim.as_ref().is_some_and(|a| a.action == 4) {
                    SWORD
                } else {
                    BEAR
                }
            } else {
                cell_base_emoji(pruner, state, r, c)
            };

            // Cost indicator (only for terrain cost > 1, hidden when bear is here)
            let cost = pruner.terrain_cost(r, c);
            let cost_str = if bear_here || cost <= 1 {
                " "
            } else {
                // Cost as digit
                match cost {
                    2 => "2",
                    3 => "3",
                    _ => "?",
                }
            };

            // Gap between cells: show bear if animating through it, else spaces
            let gap = if bear_in_gap { BEAR } else { "  " };

            spans.push(Span::raw(format!("{emoji}{cost_str}{gap}")));
        }

        lines.push(Line::from(spans));
    }

    let para = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" 🗺 Map "));
    f.render_widget(para, area);
}

/// Determines bear rendering: at cell (r,c) or in the gap after cell (r,c).
fn bear_render_state(
    anim_pos: Option<(f32, f32, f32)>,
    state: &GameState,
    r: usize,
    c: usize,
) -> (bool, bool) {
    match anim_pos {
        Some((anim_r, anim_c, _progress)) => {
            // Determine which row the bear is on (round for hop effect)
            let bear_row = anim_r.round() as usize;
            if bear_row != r {
                return (false, false);
            }

            // Determine horizontal position
            let cell_idx = anim_c.floor() as usize;
            let frac = anim_c - cell_idx as f32;

            if cell_idx == c && frac < 0.35 {
                // Bear is at this cell
                (true, false)
            } else if cell_idx == c && (0.35..0.65).contains(&frac) {
                // Bear is in the gap after this cell
                (false, true)
            } else if cell_idx + 1 == c && frac >= 0.65 {
                // Bear has arrived at next cell
                (true, false)
            } else {
                (false, false)
            }
        }
        None => {
            // Idle: bear at current state position
            let here = state.r == r && state.c == c;
            (here, false)
        }
    }
}

/// Returns the emoji for a cell WITHOUT the bear (entities, terrain).
fn cell_base_emoji(pruner: &TacticalPruner, state: &GameState, r: usize, c: usize) -> &'static str {
    // Dead monster (killed, item already picked up)
    for (i, &(mr, mc)) in pruner.monsters.iter().enumerate() {
        if (mr, mc) == (r, c)
            && (state.killed_monsters & (1 << i)) != 0
            && (state.dropped_items & (1 << i)) == 0
        {
            return MONSTER_DEAD;
        }
    }

    // Live monster
    for (i, &(mr, mc)) in pruner.monsters.iter().enumerate() {
        if (mr, mc) == (r, c) && (state.killed_monsters & (1 << i)) == 0 {
            return MONSTER_LIVE;
        }
    }

    // Dropped item on floor
    for (i, &(mr, mc)) in pruner.monsters.iter().enumerate() {
        if (mr, mc) == (r, c) && (state.dropped_items & (1 << i)) != 0 {
            return ITEM;
        }
    }

    // Uncollected treasure
    for (i, &(tr, tc)) in pruner.treasures.iter().enumerate() {
        if (tr, tc) == (r, c) && (state.collected_treasures & (1 << i)) == 0 {
            return TREASURE;
        }
    }

    // Goal / Exit
    if pruner.goal == (r, c) {
        let all = (1 << pruner.treasures.len()) - 1;
        return if state.collected_treasures == all {
            GOAL_OPEN
        } else {
            GOAL
        };
    }

    // Terrain
    terrain_emoji(&pruner.grid, r, c)
}

// ── Legend ──────────────────────────────────────────────────────

fn draw_legend(f: &mut Frame, area: Rect, app: &App) {
    let pruner = &app.pruner;
    let has_sand = pruner.grid.iter().any(|row| row.contains(&'~'));
    let has_water = pruner.grid.iter().any(|row| row.contains(&'w'));

    let mut lines = vec![
        Line::from(vec![
            Span::raw(format!(" {BEAR} You      ")),
            Span::raw(format!("{MONSTER_LIVE} Monster  ")),
        ]),
        Line::from(vec![
            Span::raw(format!(" {TREASURE} Treasure ")),
            Span::raw(format!("{GOAL} Exit     ")),
        ]),
        Line::from(vec![
            Span::raw(format!(" {FLOOR} Grass(1) ")),
            if has_sand {
                Span::raw(format!("{SAND} Sand(2)"))
            } else {
                Span::raw("         ")
            },
        ]),
    ];

    if has_water {
        lines.push(Line::from(vec![
            Span::raw(format!(" {WATER} Water(3) ")),
            Span::raw(format!("{WALL} Wall     ")),
        ]));
    } else {
        lines.push(Line::from(vec![
            Span::raw(format!(" {WALL} Wall     ")),
            Span::raw("          "),
        ]));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!(" Kill {MONSTER_LIVE} → {ITEM} → unlock {TREASURE}"),
        Style::default().fg(Color::DarkGray),
    )));

    let para =
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" Legend "));
    f.render_widget(para, area);
}

// ── State Panel ────────────────────────────────────────────────

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

    let cost_style = if state.total_cost > 0 {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };

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
            Span::styled("  Cost:      ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", state.total_cost), cost_style),
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

// ── Solution List ──────────────────────────────────────────────

fn draw_solution(f: &mut Frame, area: Rect, app: &App) {
    let mut lines: Vec<Line> = Vec::new();

    for (i, &action) in app.solution.iter().enumerate() {
        let step_num = i + 1;
        let icon = action_icon(action);
        let name = action_name(action);
        let is_current = step_num == app.current;
        let is_done = step_num < app.current;

        // Cost delta for this step
        let prev_cost = app.states[i].total_cost;
        let next_cost = app.states[step_num].total_cost;
        let cost_delta = next_cost - prev_cost;
        let cost_info = if cost_delta > 0 {
            format!(" (+{})", cost_delta)
        } else {
            String::new()
        };

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
            Span::styled(format!("{name:<6}"), style),
            Span::styled(cost_info, Style::default().fg(Color::Yellow)),
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

// ── Navigation Bar ─────────────────────────────────────────────

fn draw_nav(f: &mut Frame, area: Rect, app: &App) {
    let total = app.total_steps();
    let cur = app.current;

    let back_style = if app.is_at_start() || app.anim.is_some() {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD)
    };

    let next_style = if app.is_at_end() || app.anim.is_some() {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD)
    };

    let center = if let Some(anim) = &app.anim {
        let action = anim.action;
        let icon = action_icon(action);
        let name = action_name(action);
        format!("⟳ {icon} {name}...")
    } else if total == 0 {
        "No solution".into()
    } else if app.is_at_start() {
        "▶ Start ◀".into()
    } else if app.is_at_end() {
        let final_cost = app.states.last().map_or(0, |s| s.total_cost);
        format!("🎉 Step {cur}/{total} · Cost {final_cost} — Solved!")
    } else {
        let action = app.solution[cur - 1];
        let icon = action_icon(action);
        let name = action_name(action);
        let next_state = &app.states[cur];
        let cost = app.pruner.terrain_cost(next_state.r, next_state.c);
        let terrain = terrain_label(&app.pruner.grid, next_state.r, next_state.c);
        let cost_extra = if cost > 1 {
            format!(" · {terrain}(+{cost})")
        } else {
            String::new()
        };
        format!("Step {cur}/{total} — {icon} {name}{cost_extra}")
    };

    let auto_str = if app.auto_play { " ⏵" } else { "" };

    let line = Line::from(vec![
        Span::styled(" ◀ Back ", back_style),
        Span::styled(
            format!("   {center}{auto_str}   "),
            Style::default().fg(Color::Yellow),
        ),
        Span::styled(" Next ▶ ", next_style),
    ]);

    let para = Paragraph::new(line).block(Block::default().borders(Borders::ALL));
    f.render_widget(para, area);
}

// ── Utility ────────────────────────────────────────────────────

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
