# Optimization Skill

Hot-path Rust optimization patterns. Apply to any microsecond-sensitive code.

## When to Optimize

1. Profile first — never optimize without numbers
2. Identify the top 3 bottlenecks (80% of time is in 20% of code)
3. Measure after each change — some "optimizations" make things worse
4. Run both debug (reveals algorithmic cost) and release (reveals compiler optimizations)

## Do

### Profiling

- Break down complex functions into per-component micro-benchmarks
- Use `std::hint::black_box()` to prevent dead-code elimination
- Warm up before measuring (100+ iterations) to prime CPU caches
- Run 10,000+ iterations for stable results
- Print component-level breakdowns with `--nocapture` test harness
- Compare same-commit, back-to-back runs to isolate feature impact from system noise

### Data Structures

- Use fixed-size arrays `[T; N]` when domain is bounded
- Pre-compute lookup tables once, store in config/context — O(1) reads beat O(n) scans
- Track per-slot aggregates during insert/evict instead of scanning on read
- Cache allocations: `Vec::with_capacity()` once, `clear()` + reuse across calls
- Pass pre-allocated scratch buffers as `&mut [T]` parameters instead of allocating inside hot loops

### SIMD / Auto-vectorization

- Write chunked loops (4 or 8 elements at a time) to help LLVM auto-vectorize
- Cast `usize` → `u64` slices (same layout on 64-bit) for wider SIMD lanes
- Use `u64` equality comparison — compiler maps to `_mm256_cmpeq_epi64` on AVX2
- Keep inner loops branch-free (use `bool as usize` instead of `if`)
- Verify with release build — SIMD benefits only appear with optimizations enabled

### Parallelism

- Only parallelize when per-task work exceeds thread-pool overhead (~5μs for rayon)
- Benchmark serial vs parallel at actual workload size before committing
- Rule of thumb: parallelism wins only when per-iteration work > 10μs or count > 1000

### Allocation

- Pre-build lookup tables and cached data in config structs via builder pattern
- Reuse scratch buffers across loop iterations instead of allocating per-iteration
- Pre-allocate output arrays upfront, write in-place instead of collecting per-iteration
- Reorder struct fields to eliminate padding (group by alignment: u64 → u32 → u8)
- Use `#[repr(u8)]` on field-less enums to guarantee 1-byte size

### WASM FFI / Sandboxed Execution

- **Batch API**: Serialize shared state once, validate all N×M combinations in one FFI call (amortize ~250ns FFI floor across 24 pairs → 5–6× speedup)
- **Zero-copy serialization**: Use fixed-size stack buffers (`[u8; 1024]`) instead of `Vec::with_capacity()` — eliminates allocator overhead in tight loops (~3.6× faster)
- **Fuel budgeting**: Set WASM fuel proportional to worst-case algorithmic complexity, not average case. BFS on bounded grids with N entities can spike 4–5× above typical. Fuzz-test with max inputs to find the ceiling.
- **Lock-free instance pools**: Use `papaya::HashMap<ThreadId, T>` for per-thread WASM stores — lock-free reads on existing entries, uncontended `Mutex` per thread. Better than a single global `Mutex` for multi-threaded servers.
- **Batch state layout**: Omit per-item data from shared state; pass player/entity arrays separately. Grid(169) + bombs(N×4) shared once, then per-entity `(id, x, y)` array alongside action indices and output results buffer.
- **TypedFunc clone**: `wasmtime::TypedFunc` is cheap to clone (handle index). Clone it to release `&self` borrow before calling mutable `Store` methods — avoids borrow-checker conflicts with zero cost.

### Caching

- Compute once per position/context, not per-sample-per-position (N×M → M calls)
- Pre-compute values that don't change across samples (entropy, base path, threshold)

## Don't

### Don't: Rayon for tiny workloads

```text
// BAD: rayon overhead (~5μs) >> computation (~0.1μs per row)
let counts: Vec<_> = (0..10).into_par_iter()
    .map(|i| compute_row(variants, i))
    .collect();

// GOOD: serial for small m
let counts: Vec<_> = (0..10)
    .map(|i| compute_row(variants, i))
    .collect();
```

Threshold: rayon wins only at m ≥ 64 with μs/row work, or m ≥ 1000 with ns/row work.

### Don't: GPU for microsecond workloads

GPU kernel launch overhead is ~50μs. If your computation is 2-5μs, GPU is a net negative.
GPU wins only for: batched matmul, large tensor ops, or when you can amortize launch across many ops.

### Don't: Allocate inside hot loops

```text
// BAD: allocates every call, every sample, every position
for sample in 0..10 {
    let support = rule.support(vocab_size); // Vec allocation!
}

// GOOD: pre-compute once, reuse
let config = Config::default().with_cached_data(size);
for sample in 0..10 {
    let support = config.data_for(rule); // &[T] — zero alloc
}
```

### Don't: Linear scan for hot-path queries

```text
// BAD: O(n) scan per query
fn query(&self, key: usize) -> f32 {
    for item in &self.items {
        if item.key == key { ... }
    }
}

// GOOD: O(1) precomputed index
struct Store {
    stats: [SlotStats; MAX_SLOTS], // updated on insert/evict
}
fn query(&self, key: usize) -> f32 {
    self.stats[key].rate() // O(1)
}
```

### Don't: Recompute unchanged values

```text
// BAD: same value recomputed N×M times
for sample in 0..N {
    for &pos in &positions {
        let h = expensive_calc(data[pos]); // SAME value every sample!
    }
}

// GOOD: compute once per position
let cache: Vec<f32> = positions.iter()
    .map(|&pos| expensive_calc(data[pos]))
    .collect(); // M calls instead of N×M
```

### Don't: Parallelize without measuring

Always benchmark before AND after adding parallelism. If the serial version is faster, keep serial.
Parallel overhead: thread wake (~2μs) + work stealing (~3μs) + synchronization.
If your total work is < 10μs, parallelism will make it slower.

### Don't: Ignore binary bloat from feature flags

Adding code behind a feature flag still affects the entire binary when enabled:
- Larger binary → more icache misses → slower hot loops in unrelated code
- Feature-gated code in the same crate affects code layout and branch prediction

Mitigation:
- Isolate feature-gated benchmarks into separate binaries (`[[bin]]`) or test files
- Compare no-feature vs with-feature on the same commit, back-to-back
- If regressions appear only with feature enabled and code is properly gated, it's binary bloat, not a bug

### Don't: Under-budget WASM fuel for complex algorithms

WASM fuel limits prevent infinite loops but can silently trap legitimate computation.
Complex BFS/graph algorithms with N entities on bounded domains can spike well above average:

```text
// BAD: fuel based on average case, traps on worst case
const FUEL_PER_CALL: u64 = 10_000;  // sufficient for 1–2 bombs
// BFS with 4+ bombs × 4 directions × range × 169 cells = ~40K ops → SILENT TRAP

// GOOD: fuel based on worst-case analysis + headroom
const FUEL_PER_CALL: u64 = 50_000;  // 16 bombs × 4 dirs × range 3 × 169 cells ≈ 40K + margin
```

Symptom: WASM returns `false` for valid inputs that should return `true`. Only manifests with complex inputs. Batch APIs may mask this if they use higher fuel multipliers. Fuzz-test with maximum entity counts to catch fuel traps.

### Don't: Serialize per-item when state is shared across a batch

When validating N items against the same state (e.g., N players on one game grid), serializing the state N times wastes both allocation and FFI overhead:

```text
// BAD: 24 × (serialize + FFI + compute) = ~12µs/tick
for player in 0..4 {
    for action in 0..6 {
        let state = serialize(grid, player, action);  // 24 serializations!
        wasm.is_valid(state);                          // 24 FFI calls!
    }
}

// GOOD: 1 × (serialize + FFI + batch compute) = ~1.7µs/tick
let state = serialize_grid(grid, bombs);               // 1 serialization
wasm.batch_validate(state, players, actions, results);  // 1 FFI call
```

The batch API turns N×M individual calls into 1 call. The WASM module internally loops over all combinations, reusing the parsed state. For 4 players × 6 actions, this gives ~5.8× speedup.

### Don't: Compare benchmarks across different CPU thermal states

Laptop CPUs throttle aggressively. A 30% "regression" may just be heat.
Always compare same-commit, back-to-back runs to isolate feature impact from system noise.

## Profiling Template

```text
// tests/prof_bench.rs — run with: cargo test --features X prof_bench -- --nocapture
#[cfg(feature = "X")]
#[test]
fn prof_components() {
    let warmup = 100;
    let iters = 10000;
    
    for _ in 0..warmup { black_box(component_a()); }
    let start = Instant::now();
    for _ in 0..iters { black_box(component_a()); }
    let t_a = start.elapsed();
    
    // ... same pattern for component_b, component_c ...
    
    println!("  Component A: {:.2} μs", t_a.as_micros() as f64 / iters as f64);
    println!("  Total Δ:     {:.2} μs", total.as_micros() as f64 / iters as f64);
}
```

## WASM FFI Batch Template

```text
// Pattern: batch validate N items × M actions in one FFI call
//
// Memory layout written to WASM:
//   [0..state_end)         shared state (grid + bombs, no per-entity data)
//   [players_off..+N×12)   entity array: N × (id, x, y) as u32 LE
//   [actions_off..+M×4)    action indices as u32 LE
//   [results_off..+N×M×4)  output: u32 LE results (0/1 or Q16.16)
//
// WASM export signature:
//   batch_is_valid(state_ptr, state_len, players_ptr, player_count,
//                  actions_ptr, action_count, results_ptr) -> u32

const MAX_ENTITIES: usize = 4;
const ACTION_COUNT: usize = 6;
const ACTIONS_BYTES: [u8; ACTION_COUNT * 4] = [0,0,0,0, 1,0,0,0, 2,0,0,0, 3,0,0,0, 4,0,0,0, 5,0,0,0];

fn batch_validate(&self, grid: &Grid, players: &[(u8,i32,i32)], bombs: &[Bomb]) -> BatchResult {
    self.with_inner(|inner| {
        // 1. Serialize shared state once (zero-copy stack buffer)
        let (state_bytes, state_tokens) = inner.state_buf.serialize_grid(grid, bombs);
        let mut tmp = [0u8; 1024];
        tmp[..state_bytes].copy_from_slice(inner.state_buf.as_bytes(state_bytes));

        // 2. Compute aligned offsets
        let players_off = (state_bytes + 7) & !7;  // align8
        let actions_off = players_off + players.len() * 12;
        let results_off = actions_off + ACTION_COUNT * 4;

        // 3. Write to WASM memory
        inner.write_memory(0, &tmp[..state_bytes])?;
        inner.write_memory(players_off, &players_to_bytes(players))?;
        inner.write_memory(actions_off, &ACTIONS_BYTES)?;

        // 4. Call batch export
        let batch_fn = inner.batch_fn.as_ref()?.clone();
        batch_fn.call(&mut inner.store, (0, state_tokens, players_off as u32,
            players.len() as u32, actions_off as u32, ACTION_COUNT as u32,
            results_off as u32))?;

        // 5. Read results
        Some(BatchResult::from_memory(inner, results_off, players.len(), ACTION_COUNT))
    })
}