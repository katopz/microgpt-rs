//! T10: Performance Benchmarks — Native vs WASM `is_safe_action` overhead (Plan 034)
//!
//! Measures per-call overhead of WASM validation vs native Rust.
//! Targets: is_safe_action < 10µs, relevance < 20µs, full game < 50ms.
//!
//! Run: `cargo test --test bomber_wasm_bench --features bomber-wasm --release -- --nocapture`
//!
//! Results are saved to `.benchmarks/003_bomber_wasm_validator.md`.
//!
//! # Prerequisites
//!
//! Build the WASM validator first (release mode for accurate perf):
//! ```sh
//! cd riir-ai && cargo build --example bomber_validator --target wasm32-unknown-unknown --release
//! ```

#![cfg(feature = "bomber-wasm")]

use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use microgpt_rs::pruners::bomber::wasm_pruner::BomberWasmPruner;
use microgpt_rs::pruners::bomber::wasm_state::serialize_game_state;
use microgpt_rs::pruners::bomber::{ArenaGrid, BomberAction, Cell, GridPos, is_safe_action};

// ── Config ──────────────────────────────────────────────────────

const WASM_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../riir-ai/target/wasm32-unknown-unknown/release/examples/bomber_validator.wasm"
);

const WARMUP: u64 = 100;
const ITERS: u64 = 10_000;
const GAME_TICKS: u32 = 200;
const GAME_ROUNDS: u32 = 5;

// ── Helpers ─────────────────────────────────────────────────────

fn wasm_available() -> bool {
    Path::new(WASM_PATH).exists()
}

fn skip_msg() -> String {
    format!(
        "Skipping: WASM not found at {WASM_PATH}\n  Build with: cd riir-ai && cargo build --example bomber_validator --target wasm32-unknown-unknown --release"
    )
}

/// Simple grid with known layout for repeatable benchmarks.
fn bench_grid() -> ArenaGrid {
    ArenaGrid::generate(42)
}

/// Grid with bombs placed for complex-state benchmarks.
fn bench_bombs() -> Vec<((i32, i32), u32, u32)> {
    vec![((5, 5), 2, 4), ((7, 3), 2, 3), ((3, 7), 2, 2)]
}

/// Micro-benchmark: time a closure over N iterations, return ns/iter.
fn bench_ns(label: &str, warmup: u64, iters: u64, f: impl Fn()) -> f64 {
    for _ in 0..warmup {
        black_box(f());
    }
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    let elapsed = start.elapsed();
    let ns_per = elapsed.as_nanos() as f64 / iters as f64;
    let us_per = ns_per / 1000.0;
    println!(
        "  {:<45} {:>8.2} µs/call  ({:.0} ops/sec)",
        label,
        us_per,
        1_000_000_000.0 / ns_per
    );
    ns_per
}

// ── Benchmarks ──────────────────────────────────────────────────

#[test]
fn bench_wasm_instantiation() {
    if !wasm_available() {
        println!("{}", skip_msg());
        return;
    }

    println!("\n═══ Benchmark: WASM Instantiation Overhead ═══");

    let bytes = std::fs::read(WASM_PATH).expect("read wasm");
    println!("  WASM size: {:.1} KB", bytes.len() as f64 / 1024.0);

    let ns = bench_ns("BomberWasmPruner::load(bytes)", 5, 20, || {
        let _ = BomberWasmPruner::load(&bytes);
    });
    println!("  → One-time cost: {:.2} ms", ns / 1_000_000.0);
}

#[test]
fn bench_serialization() {
    if !wasm_available() {
        println!("{}", skip_msg());
        return;
    }

    println!("\n═══ Benchmark: Game State Serialization ═══");

    let grid = bench_grid();
    let no_bombs: Vec<((i32, i32), u32, u32)> = vec![];
    let bombs = bench_bombs();

    bench_ns(
        "serialize_game_state (no bombs, 13×13)",
        WARMUP,
        ITERS,
        || {
            let _ = serialize_game_state(&grid, 3, 3, 0, &no_bombs);
        },
    );

    bench_ns(
        "serialize_game_state (3 bombs, 13×13)",
        WARMUP,
        ITERS,
        || {
            let _ = serialize_game_state(&grid, 3, 3, 0, &bombs);
        },
    );
}

#[test]
fn bench_is_safe_action_empty_grid() {
    if !wasm_available() {
        println!("{}", skip_msg());
        return;
    }

    println!("\n═══ Benchmark: is_safe_action — Empty Grid, No Bombs ═══");

    let grid = bench_grid();
    let no_bombs: Vec<((i32, i32), u32, u32)> = vec![];
    let pos = GridPos { x: 3, y: 3 };
    let pruner = BomberWasmPruner::load_from_file(WASM_PATH).unwrap();

    // Native
    let native_ns = bench_ns("Native is_safe_action (Up)", WARMUP, ITERS, || {
        let _ = is_safe_action(&BomberAction::Up, &grid, pos, &no_bombs);
    });

    // WASM
    let wasm_ns = bench_ns("WASM  is_safe_action (Up)", WARMUP, ITERS, || {
        let _ = pruner.is_safe_action(0, &grid, 3, 3, 0, &no_bombs);
    });

    let overhead = wasm_ns / native_ns;
    println!("  → WASM overhead: {:.1}×", overhead);
    assert!(
        wasm_ns < 10_000.0,
        "WASM is_safe_action should be < 10µs, got {:.2}µs",
        wasm_ns / 1000.0
    );

    // All 6 actions
    println!();
    println!("  ── All actions (native vs WASM) ──");
    let actions = [
        (BomberAction::Up, 0, "Up"),
        (BomberAction::Down, 1, "Down"),
        (BomberAction::Left, 2, "Left"),
        (BomberAction::Right, 3, "Right"),
        (BomberAction::Bomb, 4, "Bomb"),
        (BomberAction::Wait, 5, "Wait"),
    ];

    for (action, idx, name) in &actions {
        let native_ns = bench_ns(
            &format!("Native is_safe_action ({})", name),
            WARMUP / 10,
            ITERS / 10,
            || {
                let _ = is_safe_action(action, &grid, pos, &no_bombs);
            },
        );
        let wasm_ns = bench_ns(
            &format!("WASM  is_safe_action ({})", name),
            WARMUP / 10,
            ITERS / 10,
            || {
                let _ = pruner.is_safe_action(*idx, &grid, 3, 3, 0, &no_bombs);
            },
        );
        println!(
            "    {}: native={:.0}ns wasm={:.0}ns overhead={:.1}×",
            name,
            native_ns,
            wasm_ns,
            wasm_ns / native_ns
        );
    }
}

#[test]
fn bench_is_safe_action_with_bombs() {
    if !wasm_available() {
        println!("{}", skip_msg());
        return;
    }

    println!("\n═══ Benchmark: is_safe_action — With Bombs ═══");

    let grid = bench_grid();
    let bombs = bench_bombs();
    let pos = GridPos { x: 5, y: 3 };
    let pruner = BomberWasmPruner::load_from_file(WASM_PATH).unwrap();

    // Native
    let native_ns = bench_ns("Native is_safe_action (Up, 3 bombs)", WARMUP, ITERS, || {
        let _ = is_safe_action(&BomberAction::Up, &grid, pos, &bombs);
    });

    // WASM
    let wasm_ns = bench_ns("WASM  is_safe_action (Up, 3 bombs)", WARMUP, ITERS, || {
        let _ = pruner.is_safe_action(0, &grid, 5, 3, 0, &bombs);
    });

    let overhead = wasm_ns / native_ns;
    println!("  → WASM overhead with bombs: {:.1}×", overhead);
}

#[test]
fn bench_relevance_scoring() {
    if !wasm_available() {
        println!("{}", skip_msg());
        return;
    }

    println!("\n═══ Benchmark: relevance (Q16.16 scoring) ═══");

    let grid = bench_grid();
    let bombs = bench_bombs();
    let pruner = BomberWasmPruner::load_from_file(WASM_PATH).unwrap();

    bench_ns("WASM action_relevance (Up, 3 bombs)", WARMUP, ITERS, || {
        let _ = pruner.action_relevance(0, &grid, 5, 3, 0, &bombs);
    });

    bench_ns(
        "WASM action_relevance (Bomb, 3 bombs)",
        WARMUP,
        ITERS,
        || {
            let _ = pruner.action_relevance(4, &grid, 5, 3, 0, &bombs);
        },
    );
}

#[test]
fn bench_full_game_simulation() {
    if !wasm_available() {
        println!("{}", skip_msg());
        return;
    }

    println!("\n═══ Benchmark: Full Game Simulation (200 ticks × 6 actions) ═══");

    let pruner = BomberWasmPruner::load_from_file(WASM_PATH).unwrap();
    let no_bombs: Vec<((i32, i32), u32, u32)> = vec![];
    let bombs = bench_bombs();

    // Simulate: for each tick, check all 6 actions for all players
    let simulate_game_native = |seed: u64| {
        let grid = ArenaGrid::generate(seed);
        let mut total_checks = 0u64;
        for _tick in 0..GAME_TICKS {
            for player_id in 0..4u8 {
                let px = 1 + (player_id as i32) * 3;
                let py = 1 + (player_id as i32) * 3;
                let pos = GridPos { x: px, y: py };
                for action in BomberAction::all() {
                    let _ = black_box(is_safe_action(&action, &grid, pos, &bombs));
                    total_checks += 1;
                }
            }
        }
        total_checks
    };

    let simulate_game_wasm = |seed: u64, pruner: &BomberWasmPruner| {
        let grid = ArenaGrid::generate(seed);
        let mut total_checks = 0u64;
        for _tick in 0..GAME_TICKS {
            for player_id in 0..4u8 {
                let px = 1 + (player_id as i32) * 3;
                let py = 1 + (player_id as i32) * 3;
                for action_idx in 0..6usize {
                    let _ = black_box(
                        pruner.is_safe_action(action_idx, &grid, px, py, player_id, &bombs),
                    );
                    total_checks += 1;
                }
            }
        }
        total_checks
    };

    // Warmup
    for seed in 0..2u64 {
        black_box(simulate_game_native(seed));
        black_box(simulate_game_wasm(seed, &pruner));
    }

    // Native game
    let start = Instant::now();
    let mut native_total_checks = 0u64;
    for round in 0..GAME_ROUNDS {
        native_total_checks += simulate_game_native(round as u64);
    }
    let native_elapsed = start.elapsed();

    // WASM game
    let start = Instant::now();
    let mut wasm_total_checks = 0u64;
    for round in 0..GAME_ROUNDS {
        wasm_total_checks += simulate_game_wasm(round as u64, &pruner);
    }
    let wasm_elapsed = start.elapsed();

    let native_per_game = native_elapsed.as_micros() as f64 / GAME_ROUNDS as f64;
    let wasm_per_game = wasm_elapsed.as_micros() as f64 / GAME_ROUNDS as f64;
    let overhead = wasm_elapsed.as_secs_f64() / native_elapsed.as_secs_f64();
    let checks_per_game = native_total_checks / GAME_ROUNDS as u64;

    println!(
        "  Checks per game: {} ({} ticks × 4 players × 6 actions)",
        checks_per_game, GAME_TICKS
    );
    println!("  Native per game:  {:.2} ms", native_per_game / 1000.0);
    println!("  WASM   per game:  {:.2} ms", wasm_per_game / 1000.0);
    println!("  WASM overhead:    {:.1}×", overhead);
    println!(
        "  Native per check: {:.0} ns",
        native_elapsed.as_nanos() as f64 / native_total_checks as f64
    );
    println!(
        "  WASM   per check: {:.0} ns",
        wasm_elapsed.as_nanos() as f64 / wasm_total_checks as f64
    );

    assert!(
        wasm_per_game < 50_000.0,
        "Full WASM game should be < 50ms, got {:.2}ms",
        wasm_per_game / 1000.0
    );
}

#[test]
fn bench_summary() {
    if !wasm_available() {
        println!("{}", skip_msg());
        return;
    }

    println!("\n═══ Benchmark Summary ═══");
    println!("  Run individual tests for detailed results:");
    println!(
        "    cargo test --test bomber_wasm_bench --features bomber-wasm --release -- --nocapture"
    );
    println!();
    println!("  Targets (from Plan 034):");
    println!("    is_safe_action:  < 10µs");
    println!("    relevance:       < 20µs");
    println!("    Full game:       < 50ms (200 ticks × 6 calls)");
    println!();
    println!("  Results to be recorded in .benchmarks/003_bomber_wasm_validator.md");
}
