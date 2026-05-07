//! Sudoku TUI: real-time visualization of the solver.
//!
//! Two tabs:
//! - **9×9** — full backtracking solver, animated cell-by-cell.
//! - **Speculative** — DDTree + path-aware Computable LoRA pruning.
//!
//! Keys: Tab / 1 / 2 — switch mode · R — restart · Q / Esc — quit.
//!
//! Run: `cargo run --features sudoku --release --example sudoku_tui`

use std::io::{self, Stdout};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::DOT;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Tabs, Wrap};
use ratatui::{Frame, Terminal};

use microgpt_rs::percepta::Sudoku9x9;
use microgpt_rs::speculative::{
    ConstraintPruner, SudokuPruner, TreeNode, build_dd_tree, build_dd_tree_pruned,
    extract_parent_tokens,
};
use microgpt_rs::types::Config;

// ── Mode + Cell ────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    NineByNine,
    Speculative,
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Mode::NineByNine => "9×9",
            Mode::Speculative => "Speculative",
        }
    }
    fn toggled(self) -> Self {
        match self {
            Mode::NineByNine => Mode::Speculative,
            Mode::Speculative => Mode::NineByNine,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum CellKind {
    Empty,
    Clue,
    Accepted,
    Trying,
    Bad,
}

#[derive(Clone, Copy)]
struct Cell {
    digit: u8,
    kind: CellKind,
}

const EMPTY_CELL: Cell = Cell {
    digit: 0,
    kind: CellKind::Empty,
};

// ── Channel messages ───────────────────────────────────────────

enum Msg {
    Cell {
        row: usize,
        col: usize,
        digit: u8,
        kind: CellKind,
    },
    Step(String),
    Trace(String),
    Tokens(usize),
    Done(String),
}

// ── 9×9 streaming solver ───────────────────────────────────────

fn run_nine_solver(initial: Sudoku9x9, tx: mpsc::Sender<Msg>, cancel: Arc<AtomicBool>) {
    for r in 0..9 {
        for c in 0..9 {
            let d = initial.grid[r][c];
            if d > 0 {
                let _ = tx.send(Msg::Cell {
                    row: r,
                    col: c,
                    digit: d,
                    kind: CellKind::Clue,
                });
            }
        }
    }

    let mut state = initial.clone();
    let _ = tx.send(Msg::Step(
        "Backtracking solver: depth-first with constraint check".into(),
    ));
    let _ = tx.send(Msg::Trace(format!(
        "init clues={} empty={}",
        state.clue_count(),
        81 - state.clue_count()
    )));

    let mut tries = 0usize;
    let solved = solve_nine(&mut state, 0, &tx, &cancel, &mut tries);
    let _ = tx.send(Msg::Tokens(tries));

    if solved {
        let _ = tx.send(Msg::Done(format!(
            "Solved — {} cells filled",
            state.clue_count()
        )));
    } else if cancel.load(Ordering::Relaxed) {
        let _ = tx.send(Msg::Done("Cancelled".into()));
    } else {
        let _ = tx.send(Msg::Done("No solution".into()));
    }
}

fn solve_nine(
    state: &mut Sudoku9x9,
    depth: usize,
    tx: &mpsc::Sender<Msg>,
    cancel: &AtomicBool,
    tries: &mut usize,
) -> bool {
    if cancel.load(Ordering::Relaxed) {
        return false;
    }
    let Some((row, col)) = state.next_empty() else {
        return true;
    };

    for digit in 1..=9u8 {
        if cancel.load(Ordering::Relaxed) {
            return false;
        }
        *tries += 1;

        if state.is_valid_move(row, col, digit) {
            state.grid[row][col] = digit;
            let _ = tx.send(Msg::Cell {
                row,
                col,
                digit,
                kind: CellKind::Accepted,
            });
            let _ = tx.send(Msg::Step(format!(
                "d{depth:02} place {digit} at ({},{})  {}/81",
                row + 1,
                col + 1,
                state.clue_count()
            )));
            let _ = tx.send(Msg::Trace(format!(
                "d{depth:02} ({},{})={digit}",
                row + 1,
                col + 1
            )));

            if solve_nine(state, depth + 1, tx, cancel, tries) {
                return true;
            }

            state.grid[row][col] = 0;
            let _ = tx.send(Msg::Cell {
                row,
                col,
                digit: 0,
                kind: CellKind::Empty,
            });
        }
    }
    false
}

// ── Speculative DDTree visualization (mirrors sudoku_speculative.rs) ──

struct StaticOnlyPruner<'a>(&'a SudokuPruner);
impl ConstraintPruner for StaticOnlyPruner<'_> {
    fn is_valid(&self, depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        self.0.is_valid(depth, token_idx, &[])
    }
}

fn run_speculative(initial: Sudoku9x9, tx: mpsc::Sender<Msg>, cancel: Arc<AtomicBool>) {
    // Send clues to grid
    for r in 0..9 {
        for c in 0..9 {
            let d = initial.grid[r][c];
            if d > 0 {
                let _ = tx.send(Msg::Cell {
                    row: r,
                    col: c,
                    digit: d,
                    kind: CellKind::Clue,
                });
            }
        }
    }

    let pruner = SudokuPruner::new(initial.clone());
    let lookahead = 8usize.min(pruner.empty_count());

    let _ = tx.send(Msg::Step(format!(
        "Arto Inkala: {} clues, {} empty, lookahead {}",
        initial.clue_count(),
        pruner.empty_count(),
        lookahead
    )));

    // ── 1. Solve fully with backtracking to get the ground-truth solution ──
    let mut solved_board = initial.clone();
    solve_board(&mut solved_board);

    // Place all solved cells as Accepted (cyan) — full board visible immediately
    for r in 0..9 {
        for c in 0..9 {
            if initial.grid[r][c] == 0 {
                let _ = tx.send(Msg::Cell {
                    row: r,
                    col: c,
                    digit: solved_board.grid[r][c],
                    kind: CellKind::Accepted,
                });
            }
        }
    }

    if cancel.load(Ordering::Relaxed) {
        return;
    }

    // ── 2. DDTree comparison (same as sudoku_speculative.rs) ──
    let marginals: Vec<Vec<f32>> = (0..lookahead)
        .map(|_| {
            let mut p = vec![0.0f32; 10];
            p[1..=9].fill(1.0 / 9.0);
            p
        })
        .collect();

    let config = Config {
        tree_budget: 100,
        ..Config::draft()
    };

    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let tree_unpruned = build_dd_tree(&mv, &config);
    let tree_static = build_dd_tree_pruned(&mv, &config, &StaticOnlyPruner(&pruner), false);
    let tree_aware = build_dd_tree_pruned(&mv, &config, &pruner, false);

    let _ = tx.send(Msg::Tokens(
        tree_unpruned.len() + tree_static.len() + tree_aware.len(),
    ));
    let _ = tx.send(Msg::Step(format!(
        "Tree sizes: unpruned={} static={} path-aware={}",
        tree_unpruned.len(),
        tree_static.len(),
        tree_aware.len()
    )));

    // Per-depth comparison
    let max_depth = [&tree_unpruned, &tree_static, &tree_aware]
        .iter()
        .flat_map(|t| t.iter().map(|n| n.depth))
        .max()
        .unwrap_or(0);

    for depth in 0..=max_depth {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        let Some((row, col)) = pruner.position_at(depth) else {
            break;
        };

        let up = tokens_at(&tree_unpruned, depth);
        let st = tokens_at(&tree_static, depth);
        let aw = tokens_at(&tree_aware, depth);

        let _ = tx.send(Msg::Step(format!(
            "d{depth:02} ({},{})  up={up:?}  st={st:?}  aw={aw:?}",
            row + 1,
            col + 1
        )));
        let _ = tx.send(Msg::Trace(format!(
            "d{depth:02} ({},{}) unpruned={up:?}",
            row + 1,
            col + 1
        )));
        let _ = tx.send(Msg::Trace(format!(
            "d{depth:02} ({},{}) static  ={st:?}",
            row + 1,
            col + 1
        )));
        let _ = tx.send(Msg::Trace(format!(
            "d{depth:02} ({},{}) aware   ={aw:?}",
            row + 1,
            col + 1
        )));
    }

    // Count accumulated validity (same logic as sudoku_speculative.rs)
    let unpruned_accum = count_accum_valid(&tree_unpruned, &pruner);
    let static_accum = count_accum_valid(&tree_static, &pruner);
    let aware_accum = count_accum_valid(&tree_aware, &pruner);

    let pct = |v: usize, t: usize| {
        if t == 0 {
            "—".into()
        } else {
            format!("{:.1}%", v as f64 / t as f64 * 100.0)
        }
    };

    let static_conflicts = tree_static.len() - static_accum;
    let aware_conflicts = tree_aware.len() - aware_accum;
    let caught = static_conflicts.saturating_sub(aware_conflicts);

    let _ = tx.send(Msg::Step("".into()));
    let _ = tx.send(Msg::Step("=== Accumulated Validity ===".into()));
    let _ = tx.send(Msg::Step(format!(
        "Unpruned:   {}/{} ({})",
        unpruned_accum,
        tree_unpruned.len(),
        pct(unpruned_accum, tree_unpruned.len())
    )));
    let _ = tx.send(Msg::Step(format!(
        "Static:     {}/{} ({})",
        static_accum,
        tree_static.len(),
        pct(static_accum, tree_static.len())
    )));
    let _ = tx.send(Msg::Step(format!(
        "Path-aware: {}/{} ({})",
        aware_accum,
        tree_aware.len(),
        pct(aware_accum, tree_aware.len())
    )));
    let _ = tx.send(Msg::Step(format!(
        "Cross-depth conflicts caught by path-aware: {caught}"
    )));

    let summary = if aware_accum == tree_aware.len() {
        format!("100% valid · caught {caught} cross-depth conflicts")
    } else {
        format!("path-aware: {} nodes", tree_aware.len())
    };
    let _ = tx.send(Msg::Done(summary));
}

fn tokens_at(tree: &[TreeNode], depth: usize) -> Vec<u8> {
    let mut v: Vec<u8> = tree
        .iter()
        .filter(|n| n.depth == depth)
        .map(|n| n.token_idx as u8)
        .collect();
    v.sort();
    v.dedup();
    v
}

fn solve_board(board: &mut Sudoku9x9) -> bool {
    let Some((row, col)) = board.next_empty() else {
        return true;
    };
    for digit in 1..=9u8 {
        if board.is_valid_move(row, col, digit) {
            board.grid[row][col] = digit;
            if solve_board(board) {
                return true;
            }
            board.grid[row][col] = 0;
        }
    }
    false
}

fn count_accum_valid(tree: &[TreeNode], pruner: &SudokuPruner) -> usize {
    tree.iter()
        .filter(|node| {
            let all = extract_parent_tokens(node.parent_path, node.depth + 1);
            let parents = &all[..node.depth];
            let mut board = pruner.board().clone();
            for (d, &tok) in parents.iter().enumerate() {
                if tok == 0 {
                    continue;
                }
                if let Some((r, c)) = pruner.position_at(d) {
                    board.grid[r][c] = tok as u8;
                }
            }
            pruner
                .position_at(node.depth)
                .is_some_and(|(r, c)| board.is_valid_move(r, c, node.token_idx as u8))
        })
        .count()
}

// ── App state ──────────────────────────────────────────────────

struct App {
    mode: Mode,
    grid: [[Cell; 9]; 9],
    steps: Vec<String>,
    trace: Vec<String>,
    tokens: usize,
    started: Instant,
    elapsed_secs: Option<f64>, // frozen when solver finishes
    summary: Option<String>,
    rx: mpsc::Receiver<Msg>,
    cancel: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl App {
    fn new(mode: Mode) -> Self {
        let (rx, cancel, handle) = spawn_solver(mode);
        Self {
            mode,
            grid: [[EMPTY_CELL; 9]; 9],
            steps: Vec::new(),
            trace: Vec::new(),
            tokens: 0,
            started: Instant::now(),
            elapsed_secs: None,
            summary: None,
            rx,
            cancel,
            handle: Some(handle),
        }
    }

    fn restart(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        let (rx, cancel, handle) = spawn_solver(self.mode);
        self.grid = [[EMPTY_CELL; 9]; 9];
        self.steps.clear();
        self.trace.clear();
        self.tokens = 0;
        self.started = Instant::now();
        self.elapsed_secs = None;
        self.summary = None;
        self.rx = rx;
        self.cancel = cancel;
        self.handle = Some(handle);
    }

    fn switch(&mut self, mode: Mode) {
        if self.mode == mode {
            return;
        }
        self.mode = mode;
        self.restart();
    }

    fn drain(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            self.apply(msg);
        }
    }

    fn apply(&mut self, msg: Msg) {
        match msg {
            Msg::Cell {
                row,
                col,
                digit,
                kind,
            } => {
                self.grid[row][col] = Cell { digit, kind };
            }
            Msg::Step(s) => {
                self.steps.push(s);
                trim(&mut self.steps);
            }
            Msg::Trace(s) => {
                self.trace.push(s);
                trim(&mut self.trace);
            }
            Msg::Tokens(n) => self.tokens += n,
            Msg::Done(s) => {
                self.elapsed_secs = Some(self.started.elapsed().as_secs_f64().max(1e-9));
                self.summary = Some(s);
            }
        }
    }
}

fn trim(buf: &mut Vec<String>) {
    if buf.len() > 4096 {
        buf.drain(..2048);
    }
}

fn spawn_solver(mode: Mode) -> (mpsc::Receiver<Msg>, Arc<AtomicBool>, thread::JoinHandle<()>) {
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_thread = cancel.clone();
    let handle = thread::spawn(move || {
        let board = Sudoku9x9::arto_inkala();
        match mode {
            Mode::NineByNine => run_nine_solver(board, tx, cancel_thread),
            Mode::Speculative => run_speculative(board, tx, cancel_thread),
        }
    });
    (rx, cancel, handle)
}

// ── Main loop ──────────────────────────────────────────────────

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
    let mut app = App::new(Mode::NineByNine);
    let tick = Duration::from_millis(33);
    let mut last = Instant::now();

    loop {
        app.drain();
        terminal.draw(|f| draw(f, &app))?;

        let timeout = tick.checked_sub(last.elapsed()).unwrap_or_default();
        if event::poll(timeout)?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            match key.code {
                KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
                    app.cancel.store(true, Ordering::Relaxed);
                    if let Some(h) = app.handle.take() {
                        let _ = h.join();
                    }
                    return Ok(());
                }
                KeyCode::Char('r') | KeyCode::Char('R') => app.restart(),
                KeyCode::Tab | KeyCode::BackTab => app.switch(app.mode.toggled()),
                KeyCode::Char('1') => app.switch(Mode::NineByNine),
                KeyCode::Char('2') => app.switch(Mode::Speculative),
                _ => {}
            }
        }
        last = Instant::now();
    }
}

// ── Rendering ──────────────────────────────────────────────────

fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // tabs
            Constraint::Length(13), // grid (1 + 9 + 2 + 1)
            Constraint::Length(3),  // stats
            Constraint::Min(6),     // panels
        ])
        .split(area);

    draw_tabs(f, chunks[0], app);
    draw_grid(f, chunks[1], app);
    draw_stats(f, chunks[2], app);

    let panels = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[3]);
    draw_panel(f, panels[0], "Steps · panel A", &app.steps);
    draw_panel(f, panels[1], "Trace · panel B", &app.trace);
}

fn draw_tabs(f: &mut Frame, area: Rect, app: &App) {
    let titles: Vec<Line> = vec![
        Line::from(Mode::NineByNine.label()),
        Line::from(Mode::Speculative.label()),
    ];
    let selected = match app.mode {
        Mode::NineByNine => 0,
        Mode::Speculative => 1,
    };
    let widget = Tabs::new(titles)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Sudoku TUI · Tab/1/2 switch · R restart · Q quit "),
        )
        .select(selected)
        .style(Style::default().fg(Color::Gray))
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .divider(DOT);
    f.render_widget(widget, area);
}

fn draw_grid(f: &mut Frame, area: Rect, app: &App) {
    let sep = Style::default().fg(Color::DarkGray);
    let mut lines: Vec<Line> = Vec::with_capacity(11);
    for r in 0..9 {
        if r == 3 || r == 6 {
            lines.push(Line::from(Span::styled("──────┼───────┼──────", sep)));
        }
        let mut spans: Vec<Span> = Vec::with_capacity(20);
        for c in 0..9 {
            if c == 3 || c == 6 {
                spans.push(Span::styled("│ ", sep));
            }
            let (sym, style) = format_cell(app.grid[r][c]);
            spans.push(Span::styled(format!("{sym} "), style));
        }
        lines.push(Line::from(spans));
    }
    let para = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" Board · {} ", app.mode.label())),
    );
    f.render_widget(para, area);
}

fn format_cell(cell: Cell) -> (String, Style) {
    if cell.digit == 0 {
        return (".".into(), Style::default().fg(Color::DarkGray));
    }
    let style = match cell.kind {
        CellKind::Clue => Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD),
        CellKind::Accepted => Style::default().fg(Color::Cyan),
        CellKind::Trying => Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD),
        CellKind::Bad => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        CellKind::Empty => Style::default().fg(Color::DarkGray),
    };
    (cell.digit.to_string(), style)
}

fn fmt_elapsed(secs: f64) -> String {
    if secs < 1e-6 {
        format!("{:.1}ns", secs * 1e9)
    } else if secs < 1e-3 {
        format!("{:.1}µs", secs * 1e6)
    } else if secs < 1.0 {
        format!("{:.1}ms", secs * 1e3)
    } else {
        format!("{:.2}s", secs)
    }
}

fn draw_stats(f: &mut Frame, area: Rect, app: &App) {
    let elapsed = app
        .elapsed_secs
        .unwrap_or_else(|| app.started.elapsed().as_secs_f64().max(1e-9));
    let lines = (app.steps.len() + app.trace.len()) as f64;
    let tps = app.tokens as f64 / elapsed;
    let lps = lines / elapsed;
    let summary = app.summary.as_deref().unwrap_or("running…");
    let txt = format!(
        " {} tok/s │ {} tokens │ {} l/s │ {} │ {}",
        thousands(tps as u64),
        thousands(app.tokens as u64),
        thousands(lps as u64),
        fmt_elapsed(elapsed),
        summary,
    );
    let para = Paragraph::new(txt).block(Block::default().borders(Borders::ALL).title(" Stats "));
    f.render_widget(para, area);
}

fn thousands(n: u64) -> String {
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

fn draw_panel(f: &mut Frame, area: Rect, title: &str, lines: &[String]) {
    let height = (area.height as usize).saturating_sub(2);
    let start = lines.len().saturating_sub(height);
    let body: Vec<Line> = lines[start..]
        .iter()
        .map(|s| Line::from(s.as_str()))
        .collect();
    let para = Paragraph::new(body)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" {title} ")),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}
