//! Dungeon TUI — Multi-Floor Animated Dungeon Solver
//!
//! Controls:
//!   ← / Backspace  — Previous step
//!   → / Enter      — Next step
//!   Space           — Toggle auto-play
//!   PageUp/Down     — Peek other floors
//!   Home/End        — Jump to start/end
//!   Q / Esc         — Quit
//!
//! Run: `cargo run --example dungeon_tui`

use std::collections::HashMap;
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
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::{Frame, Terminal};

use microgpt_rs::pruners::dungeon_pathfinder::{
    DungeonAction, MultiFloorBlocked, MultiFloorTarget, enumerate_multifloor_targets,
    find_path_multifloor,
};
use microgpt_rs::pruners::dungeon_pruner::{
    DungeonMap, DungeonPruner, DungeonState, StairConnection,
};
use microgpt_rs::speculative::types::ConstraintPruner;
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
const ITEM: &str = "🔑";
const STAIRS: &str = "🪜";
const STAIRS_DOWN: &str = "🪜⬇";
const STAIRS_UP: &str = "🪜⬆";
const CHECK: &str = "✓";
const CROSS: &str = "✗";

// ── Timing ─────────────────────────────────────────────────────

const TICK_MS: u64 = 50;
const AUTO_STEP_MS: u64 = 400;

// ── Helpers ────────────────────────────────────────────────────

fn action_icon(action: usize) -> &'static str {
    match action {
        0 => "↑",
        1 => "↓",
        2 => "←",
        3 => "→",
        4 => "⚔",
        5 => "🪜",
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
        5 => "Stairs",
        _ => "???",
    }
}

fn is_stair_on_floor(map: &DungeonMap, floor: usize, r: usize, c: usize) -> bool {
    map.stairs.iter().any(|s| {
        (s.from.0 == floor && s.from.1 == r && s.from.2 == c)
            || (s.to.0 == floor && s.to.1 == r && s.to.2 == c)
    })
}

fn cell_emoji(
    pruner: &DungeonPruner,
    state: &DungeonState,
    df: usize,
    r: usize,
    c: usize,
) -> &'static str {
    if state.floor == df && state.r == r && state.c == c {
        return BEAR;
    }
    for (i, &(f, mr, mc)) in pruner.map.monsters.iter().enumerate() {
        if (f, mr, mc) == (df, r, c) && (state.killed_monsters & (1 << i)) == 0 {
            return MONSTER_LIVE;
        }
    }
    for (i, &(f, mr, mc)) in pruner.map.monsters.iter().enumerate() {
        if (f, mr, mc) == (df, r, c) && (state.dropped_items & (1 << i)) != 0 {
            return ITEM;
        }
    }
    for (i, &(f, tr, tc)) in pruner.map.treasures.iter().enumerate() {
        if (f, tr, tc) == (df, r, c) && (state.collected_treasures & (1 << i)) == 0 {
            return TREASURE;
        }
    }
    if pruner.map.goal == (df, r, c) {
        let all = (1 << pruner.map.treasures.len()) - 1;
        return if state.collected_treasures == all {
            GOAL_OPEN
        } else {
            GOAL
        };
    }
    if is_stair_on_floor(&pruner.map, df, r, c) {
        return STAIRS;
    }
    for (i, &(f, mr, mc)) in pruner.map.monsters.iter().enumerate() {
        if (f, mr, mc) == (df, r, c)
            && (state.killed_monsters & (1 << i)) != 0
            && (state.dropped_items & (1 << i)) == 0
        {
            return MONSTER_DEAD;
        }
    }
    match pruner.map.floors[df][r][c] {
        '#' => WALL,
        _ => FLOOR,
    }
}

// ── Dungeon Map Definition ─────────────────────────────────────
// Floor 0 (7×7): Start(1,1), Monster(3,3), Treasure(5,2), Stairs↓(5,5)
// Floor 1 (7×7): Monster(2,2), Treasure(4,4), Goal(5,5), Stairs↑(1,1)

const FLOOR0_MAP: &str = "\
# # # # # # #
# B . . . . #
# . . . . . #
# . . M . . #
# . . . . . #
# . T . . . #
# # # # # # #";

const FLOOR1_MAP: &str = "\
# # # # # # #
# . . . . . #
# . M . . . #
# . . . . . #
# . . . T . #
# . . . . G #
# # # # # # #";

fn dungeon_stairs() -> Vec<StairConnection> {
    vec![StairConnection {
        from: (0, 5, 5),
        to: (1, 1, 1),
    }]
}

fn dungeon_action_to_usize(action: &DungeonAction) -> usize {
    match action {
        DungeonAction::Move(n) => *n,
        DungeonAction::Attack => 4,
        DungeonAction::UseStairs(_) => 5,
    }
}

// ── Strategic Pruner ───────────────────────────────────────────

struct StrategicPruner<'a> {
    pruner: &'a DungeonPruner,
    targets: Vec<MultiFloorTarget>,
}

impl<'a> StrategicPruner<'a> {
    fn new(pruner: &'a DungeonPruner) -> Self {
        let targets =
            enumerate_multifloor_targets(pruner.map.monsters.len(), pruner.map.treasures.len());
        Self { pruner, targets }
    }

    fn blocked_for_target(
        &self,
        state: &DungeonState,
        target: &MultiFloorTarget,
    ) -> MultiFloorBlocked {
        let mut blocked: MultiFloorBlocked = HashMap::new();
        let all_t = (1 << self.pruner.map.treasures.len()) - 1;

        if state.collected_treasures != all_t {
            match target {
                MultiFloorTarget::Goal => {}
                _ => {
                    let (f, r, c) = self.pruner.map.goal;
                    blocked.entry(f).or_default().insert((r, c));
                }
            }
        }
        if state.inventory == 0 {
            for (i, &(f, r, c)) in self.pruner.map.treasures.iter().enumerate() {
                if (state.collected_treasures & (1 << i)) == 0 {
                    match target {
                        MultiFloorTarget::Treasure(j) if *j == i => continue,
                        _ => {
                            blocked.entry(f).or_default().insert((r, c));
                        }
                    }
                }
            }
        }
        blocked
    }

    fn replay_targets(
        &self,
        parent_tokens: &[usize],
        start_state: &DungeonState,
    ) -> Option<DungeonState> {
        let mut state = start_state.clone();
        for &token_idx in parent_tokens {
            let target = self.targets.get(token_idx)?;
            let target_pos = target.pos(
                &self.pruner.map.monsters,
                &self.pruner.map.treasures,
                self.pruner.map.goal,
            );
            let blocked = self.blocked_for_target(&state, target);
            let path = find_path_multifloor(
                &self.pruner.map,
                (state.floor, state.r, state.c),
                target_pos,
                &blocked,
            )?;
            for action in &path {
                state = self
                    .pruner
                    .apply_action(&state, dungeon_action_to_usize(action))?;
            }
            if let MultiFloorTarget::Monster(_) = target {
                state = self.pruner.apply_action(&state, 4)?;
            }
        }
        Some(state)
    }
}

impl ConstraintPruner for StrategicPruner<'_> {
    fn is_valid(&self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        let Some(target) = self.targets.get(token_idx) else {
            return false;
        };
        if parent_tokens.contains(&token_idx) {
            return false;
        }
        let start_state = self.pruner.initial_state();
        let Some(state) = self.replay_targets(parent_tokens, &start_state) else {
            return false;
        };
        let blocked = self.blocked_for_target(&state, target);
        match target {
            MultiFloorTarget::Monster(i) => {
                if (state.killed_monsters & (1 << i)) != 0 {
                    return false;
                }
                find_path_multifloor(
                    &self.pruner.map,
                    (state.floor, state.r, state.c),
                    self.pruner.map.monsters[*i],
                    &blocked,
                )
                .is_some()
            }
            MultiFloorTarget::Treasure(j) => {
                if (state.collected_treasures & (1 << j)) != 0 {
                    return false;
                }
                if state.inventory == 0 {
                    return false;
                }
                let pos = self.pruner.map.treasures[*j];
                for (i, &m_pos) in self.pruner.map.monsters.iter().enumerate() {
                    if m_pos == pos && (state.killed_monsters & (1 << i)) == 0 {
                        return false;
                    }
                }
                find_path_multifloor(
                    &self.pruner.map,
                    (state.floor, state.r, state.c),
                    pos,
                    &blocked,
                )
                .is_some()
            }
            MultiFloorTarget::Goal => {
                let all_t = (1 << self.pruner.map.treasures.len()) - 1;
                if state.collected_treasures != all_t {
                    return false;
                }
                find_path_multifloor(
                    &self.pruner.map,
                    (state.floor, state.r, state.c),
                    self.pruner.map.goal,
                    &blocked,
                )
                .is_some()
            }
        }
    }
}

// ── Solve & Expand ─────────────────────────────────────────────

fn solve_dungeon(pruner: &DungeonPruner) -> Option<(Vec<usize>, usize)> {
    let strategic = StrategicPruner::new(pruner);
    let num_targets = strategic.targets.len();
    if num_targets == 0 {
        return None;
    }

    let mut config = Config::draft();
    config.vocab_size = num_targets;
    config.draft_lookahead = num_targets;
    config.tree_budget = 10000;

    let marginals = vec![vec![1.0f32 / num_targets as f32; num_targets]; config.draft_lookahead];
    let refs: Vec<&[f32]> = marginals.iter().map(|v| v.as_slice()).collect();
    let tree = build_dd_tree_pruned(&refs, &config, &strategic, false);
    let tree_nodes = tree.len();

    for node in &tree {
        let target_seq = extract_parent_tokens(node.parent_path, node.depth + 1);
        let start_state = pruner.initial_state();
        let Some(final_state) = strategic.replay_targets(&target_seq, &start_state) else {
            continue;
        };
        if (final_state.floor, final_state.r, final_state.c) == pruner.map.goal {
            let actions = expand_targets(pruner, &target_seq)?;
            return Some((actions, tree_nodes));
        }
    }
    None
}

fn expand_targets(pruner: &DungeonPruner, target_seq: &[usize]) -> Option<Vec<usize>> {
    let strategic = StrategicPruner::new(pruner);
    let mut state = pruner.initial_state();
    let mut all_actions = Vec::new();

    for &token_idx in target_seq {
        let target = &strategic.targets[token_idx];
        let target_pos = target.pos(&pruner.map.monsters, &pruner.map.treasures, pruner.map.goal);
        let blocked = strategic.blocked_for_target(&state, target);
        let path = find_path_multifloor(
            &pruner.map,
            (state.floor, state.r, state.c),
            target_pos,
            &blocked,
        )?;
        for action in &path {
            let a = dungeon_action_to_usize(action);
            state = pruner.apply_action(&state, a)?;
            all_actions.push(a);
        }
        if let MultiFloorTarget::Monster(_) = target {
            state = pruner.apply_action(&state, 4)?;
            all_actions.push(4);
        }
    }
    Some(all_actions)
}

// ── Application State ──────────────────────────────────────────

struct App {
    pruner: DungeonPruner,
    solution: Vec<usize>,
    states: Vec<DungeonState>,
    current: usize,
    peek_floor: Option<usize>,
    auto_play: bool,
    solved: bool,
    solve_time_ms: u64,
    tree_nodes: usize,
}

impl App {
    fn new() -> Self {
        let map = DungeonMap::new(&[FLOOR0_MAP, FLOOR1_MAP], dungeon_stairs());
        let pruner = DungeonPruner::new(map);
        let start = Instant::now();
        let solve_result = solve_dungeon(&pruner);
        let solve_time = start.elapsed();
        let (solution, tree_nodes) = match solve_result {
            Some((actions, nodes)) => (actions, nodes),
            None => (Vec::new(), 0),
        };
        let solved = !solution.is_empty();
        let mut states = vec![pruner.initial_state()];
        if solved {
            let mut state = pruner.initial_state();
            for &action in &solution {
                state = pruner.apply_action(&state, action).unwrap();
                states.push(state.clone());
            }
        }
        Self {
            pruner,
            solution,
            states,
            current: 0,
            peek_floor: None,
            auto_play: false,
            solved,
            solve_time_ms: solve_time.as_millis() as u64,
            tree_nodes,
        }
    }

    fn current_state(&self) -> &DungeonState {
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
    fn num_floors(&self) -> usize {
        self.pruner.map.floors.len()
    }
    fn display_floor(&self) -> usize {
        self.peek_floor.unwrap_or(self.current_state().floor)
    }

    fn step_forward(&mut self) {
        if !self.is_at_end() {
            self.current += 1;
            self.peek_floor = None;
        }
    }
    fn step_back(&mut self) {
        if !self.is_at_start() {
            self.current -= 1;
            self.peek_floor = None;
        }
    }
    fn jump_to_start(&mut self) {
        self.current = 0;
        self.peek_floor = None;
        self.auto_play = false;
    }
    fn jump_to_end(&mut self) {
        self.current = self.total_steps();
        self.peek_floor = None;
        self.auto_play = false;
    }
    fn toggle_auto_play(&mut self) {
        self.auto_play = !self.auto_play;
        if self.auto_play && self.is_at_end() {
            self.auto_play = false;
        }
    }
    fn peek_up(&mut self) {
        let f = self.display_floor();
        if f > 0 {
            self.peek_floor = Some(f - 1);
        }
    }
    fn peek_down(&mut self) {
        let f = self.display_floor();
        if f < self.num_floors() - 1 {
            self.peek_floor = Some(f + 1);
        }
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
    let mut last_step = Instant::now();

    loop {
        terminal.draw(|f| draw(f, &app))?;

        if app.auto_play
            && !app.is_at_end()
            && last_step.elapsed() >= Duration::from_millis(AUTO_STEP_MS)
        {
            app.step_forward();
            last_step = Instant::now();
        }
        if app.is_at_end() {
            app.auto_play = false;
        }

        let timeout = if app.auto_play {
            Duration::from_millis(TICK_MS)
        } else {
            Duration::from_millis(100)
        };

        if event::poll(timeout)?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            handle_key(&mut app, key.code);
            last_step = Instant::now();
        }
    }
}

fn handle_key(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Char('q') | KeyCode::Esc => std::process::exit(0),
        KeyCode::Right | KeyCode::Enter | KeyCode::Char('n') | KeyCode::Char('.') => {
            if !app.is_at_end() {
                app.step_forward();
            }
        }
        KeyCode::Left | KeyCode::Backspace | KeyCode::Char('p') => {
            if !app.is_at_start() {
                app.step_back();
            }
        }
        KeyCode::Char(' ') => app.toggle_auto_play(),
        KeyCode::Home => app.jump_to_start(),
        KeyCode::End => app.jump_to_end(),
        KeyCode::PageUp => app.peek_up(),
        KeyCode::PageDown => app.peek_down(),
        _ => {}
    }
}

// ── Drawing ────────────────────────────────────────────────────

fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(3),
        ])
        .split(area);
    draw_title(f, chunks[0], app);
    draw_content(f, chunks[1], app);
    draw_nav(f, chunks[2], app);
}

fn draw_title(f: &mut Frame, area: Rect, app: &App) {
    let final_cost = app.states.last().map_or(0, |s| s.total_cost);
    let status = if app.solved {
        let steps = app.solution.len();
        let ms = app.solve_time_ms;
        let nodes = thousands(app.tree_nodes);
        format!("Solved: {steps} steps · Cost {final_cost} · {ms}ms · {nodes} nodes")
    } else {
        "No solution found".into()
    };
    let auto = if app.auto_play { " ⏵AUTO" } else { "" };

    let line = Line::from(vec![
        Span::styled(
            " 🏰 Dungeon ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" {status}{auto} "),
            Style::default().fg(Color::Cyan),
        ),
        Span::styled(
            " ← → · Space · PgUp/Dn · Q Quit ",
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    f.render_widget(
        Paragraph::new(line).block(Block::default().borders(Borders::ALL)),
        area,
    );
}

fn draw_content(f: &mut Frame, area: Rect, app: &App) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(16),
            Constraint::Min(24),
            Constraint::Length(30),
        ])
        .split(area);
    draw_floors(f, cols[0], app);
    draw_map(f, cols[1], app);
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(12), Constraint::Min(7)])
        .split(cols[2]);
    draw_state(f, right[0], app);
    draw_legend(f, right[1], app);
}

// ── Floor Sidebar ──────────────────────────────────────────────

fn draw_floors(f: &mut Frame, area: Rect, app: &App) {
    let state = app.current_state();
    let bear_floor = state.floor;
    let peek = app.display_floor();
    let map = &app.pruner.map;
    let mut lines = Vec::new();

    for floor in (0..map.floors.len()).rev() {
        let floor_num = floor + 1;
        let is_bear = floor == bear_floor;
        let is_peek = floor == peek && !is_bear;
        let (marker, icon) = match (is_bear, is_peek) {
            (true, _) => ("◀", BEAR),
            (false, true) => ("👁", ""),
            _ => (" ", ""),
        };
        let style = if is_bear {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else if is_peek {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        lines.push(Line::from(Span::styled(
            format!("{marker} {floor_num}F {icon}"),
            style,
        )));

        for stair in &map.stairs {
            if stair.from.0 == floor {
                let (r, c) = (stair.from.1, stair.from.2);
                let dir = if stair.to.0 > floor { "↓" } else { "↑" };
                lines.push(Line::from(Span::styled(
                    format!("  {dir}({r},{c})"),
                    Style::default().fg(Color::DarkGray),
                )));
            }
            if stair.to.0 == floor {
                let (r, c) = (stair.to.1, stair.to.2);
                let dir = if stair.from.0 > floor { "↓" } else { "↑" };
                lines.push(Line::from(Span::styled(
                    format!("  {dir}({r},{c})"),
                    Style::default().fg(Color::DarkGray),
                )));
            }
        }

        if map.goal.0 == floor {
            let (r, c) = (map.goal.1, map.goal.2);
            let all_t = (1 << map.treasures.len()) - 1;
            let g = if state.collected_treasures == all_t {
                GOAL_OPEN
            } else {
                GOAL
            };
            lines.push(Line::from(Span::styled(
                format!("  {g}({r},{c})"),
                Style::default().fg(Color::DarkGray),
            )));
        }
        if floor > 0 {
            lines.push(Line::from(Span::styled(
                "  ──────",
                Style::default().fg(Color::DarkGray),
            )));
        }
    }
    f.render_widget(
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" Floors ")),
        area,
    );
}

// ── Map Rendering ──────────────────────────────────────────────

fn draw_map(f: &mut Frame, area: Rect, app: &App) {
    let display_floor = app.display_floor();
    let is_peeking = display_floor != app.current_state().floor;
    let state = app.current_state();
    let grid = &app.pruner.map.floors[display_floor];
    let mut lines = Vec::new();

    for (r, row) in grid.iter().enumerate() {
        let mut spans: Vec<Span> = Vec::new();
        for c in 0..row.len() {
            let emoji = cell_emoji(&app.pruner, state, display_floor, r, c);
            let gap = if c < row.len() - 1 { " " } else { "" };
            spans.push(Span::raw(format!("{emoji}{gap}")));
        }
        lines.push(Line::from(spans));
    }

    let fnum = display_floor + 1;
    let title = if is_peeking {
        format!(" 👁 Map F{fnum}")
    } else {
        format!(" 🗺 Map F{fnum}")
    };
    f.render_widget(
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(title)),
        area,
    );
}

// ── State Panel ────────────────────────────────────────────────

fn draw_state(f: &mut Frame, area: Rect, app: &App) {
    let state = app.current_state();
    let peek_floor = app.display_floor();
    let is_peeking = peek_floor != state.floor;

    let step_label = if app.is_at_start() {
        "Start".to_string()
    } else {
        format!("{}/{}", app.current, app.total_steps())
    };
    let floor_label = format!("{}/{}", state.floor + 1, app.num_floors());
    let pos_label = format!("({}, {})", state.r, state.c);
    let inv: String = if state.inventory == 0 {
        "(empty)".into()
    } else {
        (0..state.inventory).map(|_| ITEM).collect()
    };
    let monsters: String = app
        .pruner
        .map
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
        .map
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

    let dim = Style::default().fg(Color::DarkGray);
    let white = Style::default().fg(Color::White);
    let bold = Style::default()
        .fg(Color::White)
        .add_modifier(Modifier::BOLD);

    let mut lines = vec![
        Line::from(vec![
            Span::styled("  Step:      ", dim),
            Span::styled(step_label, bold),
        ]),
        Line::from(vec![
            Span::styled("  Floor:     ", dim),
            Span::styled(floor_label, white),
        ]),
        Line::from(vec![
            Span::styled("  Position:  ", dim),
            Span::styled(pos_label, white),
        ]),
    ];
    if is_peeking {
        let vf = peek_floor + 1;
        lines.push(Line::from(vec![
            Span::styled("  Viewing:   ", dim),
            Span::styled(format!("F{vf}"), Style::default().fg(Color::Cyan)),
        ]));
    }
    lines.push(Line::from(vec![
        Span::styled("  Inventory: ", dim),
        Span::styled(inv, Style::default().fg(Color::Yellow)),
    ]));
    lines.push(Line::from(vec![
        Span::styled("  Cost:      ", dim),
        Span::styled(format!("{}", state.total_cost), cost_style),
    ]));
    lines.push(Line::from(vec![
        Span::styled("  Monsters:  ", dim),
        Span::styled(monsters, white),
    ]));
    lines.push(Line::from(vec![
        Span::styled("  Treasures: ", dim),
        Span::styled(treasures, white),
    ]));

    f.render_widget(
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" 📊 State ")),
        area,
    );
}

// ── Legend ──────────────────────────────────────────────────────

fn draw_legend(f: &mut Frame, area: Rect, app: &App) {
    let has_stairs = !app.pruner.map.stairs.is_empty();
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
            Span::raw(format!(" {WALL} Wall      ")),
            Span::raw(format!("{FLOOR} Floor    ")),
        ]),
        Line::from(vec![
            Span::raw(format!(" {ITEM} Item      ")),
            Span::raw(format!("{MONSTER_DEAD} Dead     ")),
        ]),
    ];
    if has_stairs {
        lines.push(Line::from(vec![
            Span::raw(format!(" {STAIRS_DOWN} Stairs↓ ")),
            Span::raw(format!("{STAIRS_UP} Stairs↑  ")),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!(" Kill {MONSTER_LIVE} → {ITEM} → unlock {TREASURE}"),
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" Legend ")),
        area,
    );
}

// ── Navigation Bar ─────────────────────────────────────────────

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
        let fc = app.states.last().map_or(0, |s| s.total_cost);
        format!("🎉 Step {cur}/{total} · Cost {fc} — Solved!")
    } else {
        let action = app.solution[cur - 1];
        let icon = action_icon(action);
        let name = action_name(action);
        let ns = &app.states[cur];
        let fl = ns.floor + 1;
        format!("Step {cur}/{total}: {icon} {name} → F{fl}")
    };
    let peek_str = match app.peek_floor {
        Some(pf) => format!(" 👁F{}", pf + 1),
        None => String::new(),
    };
    let auto_str = if app.auto_play { " ⏵" } else { "" };

    let line = Line::from(vec![
        Span::styled(" ◀ Back ", back_style),
        Span::styled(
            format!("   {center}{peek_str}{auto_str}   "),
            Style::default().fg(Color::Yellow),
        ),
        Span::styled(" Next ▶ ", next_style),
    ]);
    f.render_widget(
        Paragraph::new(line).block(Block::default().borders(Borders::ALL)),
        area,
    );
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
