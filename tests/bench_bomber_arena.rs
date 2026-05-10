//! Bomberman HL Arena benchmarks — run with: cargo test --features bomber bench_bomber_arena -- --nocapture

#[cfg(feature = "bomber")]
use std::time::Instant;

#[cfg(feature = "bomber")]
use fastrand::Rng;

#[cfg(feature = "bomber")]
use microgpt_rs::pruners::bomber::{
    ArenaGrid, BomberAction, BomberPlayer, GameEvent, GreedyPlayer, GridPos, HLPlayer,
    RandomPlayer, ValidatorPlayer, init_world, run_tick, spawn_players,
};

#[cfg(feature = "bomber")]
fn random_actions(rng: &mut Rng) -> [Option<BomberAction>; 4] {
    let variants = [
        BomberAction::Up,
        BomberAction::Down,
        BomberAction::Left,
        BomberAction::Right,
        BomberAction::Wait,
    ];
    std::array::from_fn(|_| Some(variants[rng.usize(0..variants.len())]))
}

#[cfg(feature = "bomber")]
#[test]
fn bench_arena_generation() {
    let n: u64 = 1000;
    let start = Instant::now();
    for seed in 0..n {
        std::hint::black_box(ArenaGrid::generate(seed));
    }
    let elapsed = start.elapsed();
    let per_gen = elapsed / n as u32;

    println!("\n🧪 Arena Generation ({n} iterations)");
    println!("{}", "═".repeat(60));
    println!("Total: {elapsed:?}");
    println!("Per generation: {per_gen:?}");

    assert!(per_gen.as_micros() < 100, "Too slow: {per_gen:?} >= 100µs");
}

#[cfg(feature = "bomber")]
#[test]
fn bench_single_tick() {
    let n: u64 = 1000;
    let mut rng = Rng::new();
    let mut world = init_world(0);
    spawn_players(&mut world);
    let start = Instant::now();
    for _ in 0..n {
        if !run_tick(&mut world, random_actions(&mut rng)) {
            // Game ended — reset world outside hot path
            world = init_world(rng.u64(..));
            spawn_players(&mut world);
        }
    }
    let elapsed = start.elapsed();
    let per_tick = elapsed / n as u32;

    println!("\n🧪 Single Tick ({n} iterations, 4 players)");
    println!("{}", "═".repeat(60));
    println!("Total: {elapsed:?}");
    println!("Per tick: {per_tick:?}");

    assert!(per_tick.as_micros() < 50, "Too slow: {per_tick:?} >= 50µs");
}

#[cfg(feature = "bomber")]
#[test]
fn bench_full_game() {
    let n: u64 = 100;
    let mut rng = Rng::new();
    let start = Instant::now();
    for seed in 0..n {
        let mut world = init_world(seed);
        spawn_players(&mut world);
        for _ in 0..200u32 {
            if !run_tick(&mut world, random_actions(&mut rng)) {
                break;
            }
        }
    }
    let elapsed = start.elapsed();
    let per_game = elapsed / n as u32;

    println!("\n🧪 Full Game ({n} games, 200 ticks, 4 players)");
    println!("{}", "═".repeat(60));
    println!("Total: {elapsed:?}");
    println!("Per game: {per_game:?}");

    assert!(per_game.as_millis() < 10, "Too slow: {per_game:?} >= 10ms");
}

#[cfg(feature = "bomber")]
#[test]
fn bench_player_select_action() {
    let n: u64 = 1000;
    let mut rng = Rng::new();
    let grid = ArenaGrid::generate(42);
    let pos = GridPos { x: 1, y: 1 };
    let events: &[GameEvent] = &[];

    let mut p1 = RandomPlayer::new(0);
    let t1 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(p1.select_action(&grid, pos, events, &mut rng));
    }
    let t1 = t1.elapsed() / n as u32;

    let mut p2 = GreedyPlayer::new(1);
    let t2 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(p2.select_action(&grid, pos, events, &mut rng));
    }
    let t2 = t2.elapsed() / n as u32;

    let mut p3 = ValidatorPlayer::new(2);
    let t3 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(p3.select_action(&grid, pos, events, &mut rng));
    }
    let t3 = t3.elapsed() / n as u32;

    let mut p4 = HLPlayer::new(3);
    let t4 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(p4.select_action(&grid, pos, events, &mut rng));
    }
    let t4 = t4.elapsed() / n as u32;

    println!("\n🧪 Player select_action ({n} calls each)");
    println!("{}", "═".repeat(60));
    println!("P1 Random:    {t1:?}");
    println!("P2 Greedy:    {t2:?}");
    println!("P3 Validator: {t3:?}");
    println!("P4 HL:        {t4:?}");

    assert!(t4.as_micros() < 200, "HLPlayer too slow: {t4:?} >= 200µs");
}
