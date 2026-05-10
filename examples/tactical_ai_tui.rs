//! Hierarchical Tactical AI — Interactive TUI
//!
//! 16×16 dungeon with DDTree (strategic) + A* (tactical) hierarchical solver.
//! Shows strategic plan sidebar, A* path overlay, and animated movement.
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
//! Run: `cargo run --example tactical_ai_tui`

use std::collections::HashSet;
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

use microgpt_rs::pruners::pathfinder::{Target, enumerate_targets, find_path};
use microgpt_rs::pruners::tactical_pruner::{GameState, TacticalPruner};
use microgpt_rs::speculative::types::ConstraintPruner;
use microgpt_rs::speculative::{build_dd_tree_pruned, extract_parent_tokens};
use microgpt_rs::types::Config;

// ── Constants ──────────────────────────────────────────────────

const BEAR: &str = "🐻";
const MONSTER_LIVE: &str = "👹";
const MONSTER_DEAD: &str = "💀";
const TREASURE: &str = "💎";
const GOAL: &str = "🚪";
const GOAL_OPEN: &str = "🏆";
const WALL: &str = "🧱";
const FLOOR: &str = "⬜";
const ITEM: &str = "🔑";
const SWORD: &str = "⚔";
const CHECK: &str = "✓";
const ARROW: &str = "▸";

const TICK_MS: u64 = 50;
const MOVE_MS: u64 = 100;
const ATTACK_MS: u64 = 150;

const DIR_DELTA: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

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

fn target_icon(target: &Target) -> &'static str {
    match target {
        Target::Monster(_) => MONSTER_LIVE,
        Target::Treasure(_) => TREASURE,
        Target::Goal => GOAL,
    }
}

fn target_label(target: &Target) -> String {
    match target {
        Target::Monster(i) => format!("M({i})"),
        Target::Treasure(j) => format!("T({j})"),
        Target::Goal => "Goal".into(),
    }
}

// ── 16×16 Dungeon Map ──────────────────────────────────────────

const MAP: &str = "\
# # # # # # # # # # # # # # # #
# B . . . . . # . . . . . . . #
# . # # # # . # . # # # # . . #
# . . . . # . # . # . . T . . #
# . M . . # . # . # . # # # . #
# # # # . # . # . # . . . . . #
# . . . . # . . . # . . . . . #
# . # # # # # # # # . # # # . #
# . # . . . . . . . # . . . G #
# . # . # # # . # # # . # # . #
# T . . # . # . M . # . . # . #
# # # . # . # . # . . # # . . #
# . . . # . # . # . # # . . . #
# . # # # . # . # . . . . # . #
# . . . . . # . # # # # . # . #
# . . . . M . . . . . . # T . #
# # # # # # # # # # # # # # # #";

// ── StrategicPruner ────────────────────────────────────────────

struct StrategicPruner<'a> {
    tactical: &'a TacticalPruner,
    targets: Vec<Target>,
}

impl<'a> StrategicPruner<'a> {
    fn new(tactical: &'a TacticalPruner) -> Self {
        let targets = enumerate_targets(tactical.monsters.len(), tactical.treasures.len());
        Self { tactical, targets }
    }

    fn blocked_set(&self, state: &GameState) -> HashSet<(usize, usize)> {
        let mut blocked = HashSet::new();
        for (i, &pos) in self.tactical.monsters.iter().enumerate() {
            if (state.killed_monsters & (1 << i)) == 0 {
                blocked.insert(pos);
            }
        }
        let all_treasures = (1 << self.tactical.treasures.len()) - 1;
        if state.collected_treasures != all_treasures {
            blocked.insert(self.tactical.goal);
        }
        blocked
    }

    fn blocked_for_target(&self, state: &GameState, target: &Target) -> HashSet<(usize, usize)> {
        let mut blocked = self.blocked_set(state);
        if let Target::Monster(i) = target {
            blocked.remove(&self.tactical.monsters[*i]);
        }
        if let Target::Goal = target {
            blocked.remove(&self.tactical.goal);
        }
        blocked
    }

    fn replay_targets(
        &self,
        parent_tokens: &[usize],
        start_state: &GameState,
    ) -> Option<GameState> {
        let mut state = start_state.clone();
        for &token_idx in parent_tokens {
            let target = self.targets.get(token_idx)?;
            let target_pos = target.pos(
                &self.tactical.monsters,
                &self.tactical.treasures,
                self.tactical.goal,
            );
            let blocked = self.blocked_for_target(&state, target);
            let path = find_path(
                &self.tactical.grid,
                (state.r, state.c),
                target_pos,
                &blocked,
            )?;
            for &action in &path {
                state = self.tactical.apply_action(&state, action)?;
            }
            if let Target::Monster(_) = target {
                state = self.tactical.apply_action(&state, 4)?;
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
        let start_state = self.tactical.initial_state();
        let Some(state) = self.replay_targets(parent_tokens, &start_state) else {
            return false;
        };
        let blocked = self.blocked_for_target(&state, target);
        match target {
            Target::Monster(i) => {
                if (state.killed_monsters & (1 << i)) != 0 {
                    return false;
                }
                let pos = self.tactical.monsters[*i];
                find_path(&self.tactical.grid, (state.r, state.c), pos, &blocked).is_some()
            }
            Target::Treasure(j) => {
                if (state.collected_treasures & (1 << j)) != 0 || state.inventory == 0 {
                    return false;
                }
                let pos = self.tactical.treasures[*j];
                for (i, &m_pos) in self.tactical.monsters.iter().enumerate() {
                    if m_pos == pos && (state.killed_monsters & (1 << i)) == 0 {
                        return false;
                    }
                }
                find_path(&self.tactical.grid, (state.r, state.c), pos, &blocked).is_some()
            }
            Target::Goal => {
                let all_treasures = (1 << self.tactical.treasures.len()) - 1;
                if state.collected_treasures != all_treasures {
                    return false;
                }
                find_path(
                    &self.tactical.grid,
                    (state.r, state.c),
                    self.tactical.goal,
                    &blocked,
                )
                .is_some()
            }
        }
    }
}

// ── Data Types ─────────────────────────────────────────────────

struct TargetSegment {
    #[allow(dead_code)]
    target_idx: usize,
    target: Target,
    path_actions: Vec<usize>,
    path_positions: Vec<(usize, usize)>,
    has_attack: bool,
    #[allow(dead_code)]
    start_step: usize,
}

impl TargetSegment {
    fn step_count(&self) -> usize {
        self.path_actions.len() + if self.has_attack { 1 } else { 0 }
    }
}

struct Solution {
    target_sequence: Vec<usize>,
    segments: Vec<TargetSegment>,
    flat_actions: Vec<usize>,
    states: Vec<GameState>,
    solve_time_ms: u64,
    tree_nodes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Moving,
    Attacking,
    Done,
}

struct AnimState {
    from: (usize, usize),
    to: (usize, usize),
    action: usize,
    start: Instant,
    duration_ms: u64,
}

// ── Solver ─────────────────────────────────────────────────────

fn compute_path_positions(start: (usize, usize), actions: &[usize]) -> Vec<(usize, usize)> {
    let mut positions = vec![start];
    let (mut r, mut c) = start;
    for &action in actions {
        let (dr, dc) = DIR_DELTA[action];
        r = (r as isize + dr) as usize;
        c = (c as isize + dc) as usize;
        positions.push((r, c));
    }
    positions
}

fn expand_to_segments(pruner: &TacticalPruner, target_seq: &[usize]) -> Option<Vec<TargetSegment>> {
    let targets = enumerate_targets(pruner.monsters.len(), pruner.treasures.len());
    let strategic = StrategicPruner::new(pruner);
    let mut state = pruner.initial_state();
    let mut segments = Vec::new();
    let mut start_step = 0;

    for &token_idx in target_seq {
        let target = targets[token_idx].clone();
        let target_pos = target.pos(&pruner.monsters, &pruner.treasures, pruner.goal);
        let blocked = strategic.blocked_for_target(&state, &target);

        let path = find_path(&pruner.grid, (state.r, state.c), target_pos, &blocked)?;
        let positions = compute_path_positions((state.r, state.c), &path);

        for &action in &path {
            state = pruner.apply_action(&state, action)?;
        }

        let has_attack = matches!(target, Target::Monster(_));
        if has_attack {
            state = pruner.apply_action(&state, 4)?;
        }

        segments.push(TargetSegment {
            target_idx: token_idx,
            target,
            path_actions: path,
            path_positions: positions,
            has_attack,
            start_step,
        });

        start_step += segments.last().unwrap().step_count();
    }

    Some(segments)
}

fn solve(pruner: &TacticalPruner) -> Solution {
    let state = pruner.initial_state();
    let strategic = StrategicPruner::new(pruner);
    let num_targets = strategic.targets.len();

    let mut config = Config::draft();
    config.vocab_size = num_targets;
    config.draft_lookahead = num_targets;
    config.tree_budget = 10000;

    let marginals = vec![vec![1.0f32 / num_targets as f32; num_targets]; config.draft_lookahead];
    let refs: Vec<&[f32]> = marginals.iter().map(|v| v.as_slice()).collect();

    let start = Instant::now();
    let tree = build_dd_tree_pruned(&refs, &config, &strategic, false);
    let solve_time = start.elapsed();

    // Find target sequence that reaches goal
    let mut found_seq = None;
    for node in &tree {
        let target_seq = extract_parent_tokens(node.parent_path, node.depth + 1);
        if let Some(final_state) = strategic.replay_targets(&target_seq, &state)
            && (final_state.r, final_state.c) == pruner.goal
        {
            found_seq = Some(target_seq);
            break;
        }
    }

    let target_sequence = found_seq.expect("16×16 dungeon should be solvable");
    let segments =
        expand_to_segments(pruner, &target_sequence).expect("Segment expansion should succeed");

    // Flatten actions
    let mut flat_actions = Vec::new();
    for seg in &segments {
        flat_actions.extend_from_slice(&seg.path_actions);
        if seg.has_attack {
            flat_actions.push(4);
        }
    }

    // Pre-compute states
    let mut states = vec![pruner.initial_state()];
    let mut st = pruner.initial_state();
    for &action in &flat_actions {
        st = pruner.apply_action(&st, action).unwrap();
        states.push(st.clone());
    }

    Solution {
        target_sequence,
        segments,
        flat_actions,
        states,
        solve_time_ms: solve_time.as_millis() as u64,
        tree_nodes: tree.len(),
    }
}

// ── App ────────────────────────────────────────────────────────

struct App {
    pruner: TacticalPruner,
    solution: Solution,
    current: usize,
    anim: Option<AnimState>,
    auto_play: bool,
}

impl App {
    fn new() -> Self {
        let pruner = TacticalPruner::new(MAP);
        let solution = solve(&pruner);

        // Verify
        let final_state = solution.states.last().unwrap();
        assert_eq!((final_state.r, final_state.c), pruner.goal);

        Self {
            pruner,
            solution,
            current: 0,
            anim: None,
            auto_play: false,
        }
    }

    fn current_state(&self) -> &GameState {
        &self.solution.states[self.current]
    }

    fn total_steps(&self) -> usize {
        self.solution.flat_actions.len()
    }

    fn is_at_start(&self) -> bool {
        self.current == 0
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.total_steps()
    }

    // ── Segment Helpers ────────────────────────────────────────

    fn current_segment(&self) -> usize {
        let mut offset = 0;
        for (i, seg) in self.solution.segments.iter().enumerate() {
            if self.current < offset + seg.step_count() {
                return i;
            }
            offset += seg.step_count();
        }
        self.solution.segments.len().saturating_sub(1)
    }

    fn step_in_segment(&self) -> usize {
        let seg_idx = self.current_segment();
        let offset: usize = self.solution.segments[..seg_idx]
            .iter()
            .map(|s| s.step_count())
            .sum();
        self.current - offset
    }

    fn phase(&self) -> Phase {
        if self.is_at_end() {
            return Phase::Done;
        }
        let seg = &self.solution.segments[self.current_segment()];
        let step = self.step_in_segment();
        if seg.has_attack && step == seg.path_actions.len() {
            Phase::Attacking
        } else {
            Phase::Moving
        }
    }

    fn remaining_path(&self) -> HashSet<(usize, usize)> {
        if self.is_at_end() {
            return HashSet::new();
        }
        let seg = &self.solution.segments[self.current_segment()];
        let step = self.step_in_segment();
        if step < seg.path_actions.len() {
            seg.path_positions[step + 1..].iter().copied().collect()
        } else {
            HashSet::new()
        }
    }

    fn current_target_pos(&self) -> Option<(usize, usize)> {
        if self.is_at_end() {
            return None;
        }
        let seg = &self.solution.segments[self.current_segment()];
        seg.path_positions.last().copied()
    }

    fn bear_pos(&self) -> (usize, usize) {
        if let Some(ref anim) = self.anim {
            let progress = anim.start.elapsed().as_millis() as f32 / anim.duration_ms as f32;
            if progress < 0.5 { anim.from } else { anim.to }
        } else {
            let state = self.current_state();
            (state.r, state.c)
        }
    }

    fn is_attacking_anim(&self) -> bool {
        self.anim.as_ref().is_some_and(|a| a.action == 4)
    }

    // ── Animation ──────────────────────────────────────────────

    fn start_animation(&mut self) {
        if self.is_at_end() || self.anim.is_some() {
            return;
        }
        let action = self.solution.flat_actions[self.current];
        let from_state = &self.solution.states[self.current];
        let to_state = &self.solution.states[self.current + 1];

        self.anim = Some(AnimState {
            from: (from_state.r, from_state.c),
            to: (to_state.r, to_state.c),
            action,
            start: Instant::now(),
            duration_ms: if action == 4 { ATTACK_MS } else { MOVE_MS },
        });
    }

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

        let completed = app.tick_animation();
        if completed && app.auto_play && !app.is_at_end() {
            app.start_animation();
        }
        if app.is_at_end() {
            app.auto_play = false;
        }

        let timeout = if app.anim.is_some() || app.auto_play {
            Duration::from_millis(TICK_MS)
        } else {
            Duration::from_millis(100)
        };

        if event::poll(timeout)?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
            && handle_key(&mut app, key.code)
        {
            return Ok(());
        }
    }
}

fn handle_key(app: &mut App, code: KeyCode) -> bool {
    match code {
        KeyCode::Char('q') | KeyCode::Esc => return true,
        KeyCode::Char('r') => app.restart(),
        KeyCode::Right | KeyCode::Enter | KeyCode::Char('n') => {
            if app.anim.is_none() && !app.is_at_end() {
                app.start_animation();
            }
        }
        KeyCode::Char('.') => {
            if app.anim.is_none() && !app.is_at_end() {
                app.current += 1;
            }
        }
        KeyCode::Left | KeyCode::Backspace | KeyCode::Char('p') => {
            if app.anim.is_none() && !app.is_at_start() {
                app.current -= 1;
            }
        }
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
    false
}

impl App {
    fn restart(&mut self) {
        *self = Self::new();
    }
}

// ── Drawing ────────────────────────────────────────────────────

fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // title
            Constraint::Min(18),   // content
            Constraint::Length(3), // nav
        ])
        .split(area);

    draw_title(f, chunks[0], app);
    draw_content(f, chunks[1], app);
    draw_nav(f, chunks[2], app);
}

fn draw_title(f: &mut Frame, area: Rect, app: &App) {
    let final_cost = app.solution.states.last().map_or(0, |s| s.total_cost);
    let auto = if app.auto_play { " ⏵AUTO" } else { "" };
    let line = Line::from(vec![
        Span::styled(
            " 🐻 Tactical AI ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(
                " {} steps · Cost {} · {}ms · {} nodes{auto} ",
                app.total_steps(),
                final_cost,
                app.solution.solve_time_ms,
                app.solution.tree_nodes,
            ),
            Style::default().fg(Color::Cyan),
        ),
        Span::styled(" ← → Space · Q Quit ", Style::default().fg(Color::DarkGray)),
    ]);
    let para = Paragraph::new(line).block(Block::default().borders(Borders::ALL));
    f.render_widget(para, area);
}

fn draw_content(f: &mut Frame, area: Rect, app: &App) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(49), Constraint::Min(28)])
        .split(area);

    draw_map(f, cols[0], app);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(10), Constraint::Min(8)])
        .split(cols[1]);

    draw_strategy(f, right[0], app);
    draw_phase(f, right[1], app);
}

// ── Map ────────────────────────────────────────────────────────

fn draw_map(f: &mut Frame, area: Rect, app: &App) {
    let state = app.current_state();
    let pruner = &app.pruner;
    let (bear_r, bear_c) = app.bear_pos();
    let path_set = app.remaining_path();
    let target_pos = app.current_target_pos();

    let mut lines = Vec::new();
    for r in 0..pruner.grid.len() {
        let mut spans = Vec::new();
        for c in 0..pruner.grid[0].len() {
            let is_bear = bear_r == r && bear_c == c;
            let on_path = path_set.contains(&(r, c));
            let is_target = target_pos == Some((r, c));

            let (emoji, style) = if is_bear {
                let e = if app.is_attacking_anim() { SWORD } else { BEAR };
                (e.into(), Style::default())
            } else {
                cell_render(pruner, state, r, c, on_path, is_target)
            };

            spans.push(Span::styled(format!("{emoji} "), style));
        }
        lines.push(Line::from(spans));
    }

    let para = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" 🗺 Map "));
    f.render_widget(para, area);
}

fn cell_render(
    pruner: &TacticalPruner,
    state: &GameState,
    r: usize,
    c: usize,
    on_path: bool,
    is_target: bool,
) -> (String, Style) {
    let style = if is_target {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else if on_path {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default()
    };

    // Live monster
    for (i, &(mr, mc)) in pruner.monsters.iter().enumerate() {
        if (mr, mc) == (r, c) && (state.killed_monsters & (1 << i)) == 0 {
            return (MONSTER_LIVE.into(), style);
        }
    }

    // Dropped item
    for (i, &(mr, mc)) in pruner.monsters.iter().enumerate() {
        if (mr, mc) == (r, c) && (state.dropped_items & (1 << i)) != 0 {
            return (ITEM.into(), style);
        }
    }

    // Uncollected treasure
    for (i, &(tr, tc)) in pruner.treasures.iter().enumerate() {
        if (tr, tc) == (r, c) && (state.collected_treasures & (1 << i)) == 0 {
            return (TREASURE.into(), style);
        }
    }

    // Goal
    if pruner.goal == (r, c) {
        let all = (1 << pruner.treasures.len()) - 1;
        let emoji = if state.collected_treasures == all {
            GOAL_OPEN
        } else {
            GOAL
        };
        return (emoji.into(), style);
    }

    // Dead monster
    for (i, &(mr, mc)) in pruner.monsters.iter().enumerate() {
        if (mr, mc) == (r, c) && (state.killed_monsters & (1 << i)) != 0 {
            return (MONSTER_DEAD.into(), style);
        }
    }

    // Terrain
    let emoji = if pruner.grid[r][c] == '#' {
        WALL
    } else {
        FLOOR
    };
    (emoji.into(), style)
}

// ── Strategy Panel ─────────────────────────────────────────────

fn draw_strategy(f: &mut Frame, area: Rect, app: &App) {
    let targets = enumerate_targets(app.pruner.monsters.len(), app.pruner.treasures.len());
    let cur_seg = if app.is_at_end() {
        app.solution.segments.len()
    } else {
        app.current_segment()
    };

    let mut lines = Vec::new();
    for (visit_idx, &token_idx) in app.solution.target_sequence.iter().enumerate() {
        let target = &targets[token_idx];
        let seg = &app.solution.segments[visit_idx];
        let icon = target_icon(target);
        let label = target_label(target);
        let pos = target.pos(&app.pruner.monsters, &app.pruner.treasures, app.pruner.goal);
        let steps = seg.path_actions.len();

        let (status, status_style) = if visit_idx < cur_seg {
            (CHECK, Style::default().fg(Color::Green))
        } else if visit_idx == cur_seg && !app.is_at_end() {
            (
                ARROW,
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            ("·", Style::default().fg(Color::DarkGray))
        };

        lines.push(Line::from(vec![
            Span::styled(format!(" {status} "), status_style),
            Span::styled(
                format!("{icon} {label:>5}"),
                Style::default().fg(Color::White),
            ),
            Span::styled(
                format!(" ({},{})", pos.0, pos.1),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(
                format!(" {steps:>2}s"),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "  No targets",
            Style::default().fg(Color::DarkGray),
        )));
    }

    let title = format!(" Strategy ({}) ", app.solution.target_sequence.len());
    let para = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(title));
    f.render_widget(para, area);
}

// ── Phase Panel ────────────────────────────────────────────────

fn draw_phase(f: &mut Frame, area: Rect, app: &App) {
    let state = app.current_state();
    let phase = app.phase();
    let seg_idx = app.current_segment();

    let (phase_label, phase_color) = match phase {
        Phase::Moving => ("Moving", Color::Cyan),
        Phase::Attacking => ("Attacking", Color::Red),
        Phase::Done => ("Done", Color::Green),
    };

    let target_info = if app.is_at_end() {
        format!("{GOAL_OPEN} Goal reached")
    } else {
        let seg = &app.solution.segments[seg_idx];
        let icon = target_icon(&seg.target);
        let label = target_label(&seg.target);
        format!("{icon} {label}")
    };

    let inv_display = if state.inventory == 0 {
        "(empty)".into()
    } else {
        (0..state.inventory).map(|_| ITEM).collect::<String>()
    };

    let num_monsters = app.pruner.monsters.len();
    let num_treasures = app.pruner.treasures.len();
    let killed_count = (0..num_monsters)
        .filter(|i| (state.killed_monsters & (1 << i)) != 0)
        .count();
    let collected_count = (0..num_treasures)
        .filter(|j| (state.collected_treasures & (1 << j)) != 0)
        .count();

    let lines = vec![
        Line::from(vec![
            Span::styled("  Phase:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(phase_label, Style::default().fg(phase_color)),
        ]),
        Line::from(vec![
            Span::styled("  Target:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(target_info, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  Step:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}/{}", app.current, app.total_steps()),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Cost:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", state.total_cost),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Pos:     ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("({}, {})", state.r, state.c),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Inv:     ", Style::default().fg(Color::DarkGray)),
            Span::styled(inv_display, Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![Span::styled(
            format!(
                "  {MONSTER_LIVE} {killed_count}/{num_monsters}  {TREASURE} {collected_count}/{num_treasures}"
            ),
            Style::default().fg(Color::DarkGray),
        )]),
    ];

    let para = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" Phase "));
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
        let icon = action_icon(anim.action);
        let name = action_name(anim.action);
        format!("⟳ {icon} {name}...")
    } else if total == 0 {
        "No solution".into()
    } else if app.is_at_start() {
        format!("{ARROW} Start {ARROW}")
    } else if app.is_at_end() {
        let cost = app.solution.states.last().map_or(0, |s| s.total_cost);
        format!("🎉 {total} steps · Cost {cost}")
    } else {
        let action = app.solution.flat_actions[cur - 1];
        let icon = action_icon(action);
        let name = action_name(action);
        let phase = match app.phase() {
            Phase::Moving => "Moving",
            Phase::Attacking => "Attacking",
            Phase::Done => "Done",
        };
        format!("Step {cur}/{total} — {icon} {name} · {phase}")
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
