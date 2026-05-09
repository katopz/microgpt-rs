# Optimization Skill

Micro-benchmarking, profiling, and optimization patterns for hot-path Rust code.

## When to Optimize

1. Profile first — never optimize without numbers
2. Identify the top 3 bottlenecks (typically 80% of time is in 20% of code)
3. Measure after each change — some "optimizations" make things worse
4. Run both debug (development) and release (production) benchmarks

## Do

### Profiling

- Break down complex functions into per-component micro-benchmarks
- Use `std::hint::black_box()` to prevent dead-code elimination
- Warm up before measuring (100+ iterations) to prime CPU caches
- Run 10,000+ iterations for stable results
- Print component-level breakdowns with `--nocapture` test harness
- Measure in both debug and release: release reveals compiler optimizations, debug reveals algorithmic cost

### Data Structures

- Use fixed-size arrays `[T; N]` when domain is bounded (e.g., positions 0..16 for speculative lookahead)
- Pre-compute lookup tables once, store in config/context — `O(1)` reads beat `O(n)` scans
- Use `PositionStats` pattern: track per-slot aggregates during insert/evict instead of scanning on read
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
- Rayon `into_par_iter()` overhead: ~5μs for thread wake + scheduling
- Rule of thumb: parallelism wins only when per-iteration work > 10μs or count > 1000
- For m=10 variants with 0.1μs/row computation: serial is 10x faster than rayon

### Allocation

- Pre-build support sets, lookup tables, and cached data in config structs
- Use builder pattern: `Config::default().with_cached_support(vocab_size)` — compute once, use N times
- Reuse scratch buffers across loop iterations instead of allocating per-iteration
- Batch writes: collect into temp buffer → single `record_batch()` call instead of N individual calls
- `TokenRule::support(vocab_size)` was allocating `Vec<usize>` 10x per rescue — caching cut 34%

### Entropy / Math

- Cache repeated computations: if `token_entropy(pos)` is needed for N samples × M positions, compute once per position (N×M → M calls)
- Pre-compute values that don't change across samples (entropy, base path, threshold)

## Don't

### Don't: Rayon for tiny workloads

```
// BAD: rayon overhead (~5μs) >> computation (~0.1μs per row)
let counts: Vec<_> = (0..10).into_par_iter()
    .map(|i| compute_row(variants, i))
    .collect();

// GOOD: serial for m <= 64
let counts: Vec<_> = (0..10)
    .map(|i| compute_row(variants, i))
    .collect();
```

Threshold: rayon wins only at m ≥ 64 with μs/row work, or m ≥ 1000 with ns/row work.

### Don't: GPU for microsecond workloads

GPU kernel launch overhead is ~50μs. If your computation is 2-5μs, GPU is a net negative.
GPU wins only for: batched matmul, large tensor ops, or when you can amortize launch across many ops.

### Don't: Allocate inside hot loops

```
// BAD: allocates Vec every call, every sample, every position
for sample in 0..10 {
    let support = rule.support(vocab_size); // Vec allocation!
    // ...
}

// GOOD: pre-compute once, reuse
let config = PpotConfig::default().with_cached_support(vocab_size);
for sample in 0..10 {
    let support = config.support_for(rule); // &[usize] — zero alloc
    // ...
}
```

### Don't: Linear scan for hot-path queries

```
// BAD: O(n) scan over all 64 insights per position per query
fn position_affinity(&self, pos: usize) -> f32 {
    let mut accepted = 0;
    for insight in &self.insights { // scans ALL 64 entries
        if insight.position == pos { ... }
    }
}

// GOOD: O(1) precomputed index
struct SessionKnowledge {
    position_stats: [PositionStats; 16], // precomputed, updated on insert
}
fn position_affinity(&self, pos: usize) -> f32 {
    self.position_stats[pos].success_rate() // O(1)
}
```

### Don't: Recompute unchanged values

```
// BAD: entropy recomputed for each of 10 samples × 3 positions = 30 calls
for sample in 0..10 {
    for &pos in &positions {
        let h = token_entropy(marginals[pos]); // SAME value every time!
    }
}

// GOOD: compute once per position
let entropy_cache: Vec<f32> = positions.iter()
    .map(|&pos| token_entropy(marginals[pos]))
    .collect(); // 3 calls instead of 30
```

### Don't: Parallelize without measuring

Always benchmark before AND after adding parallelism. If the serial version is faster, keep serial.
Parallel overhead: thread wake (~2μs) + work stealing (~3μs) + synchronization.
If your total work is < 10μs, parallelism will make it slower.

## Profiling Template

```rust
// tests/prof_bench.rs — run with: cargo test --features X prof_bench -- --nocapture
#[cfg(feature = "X")]
#[test]
fn prof_components() {
    // Setup
    let warmup = 100;
    let iters = 10000;
    
    // Component A
    for _ in 0..warmup { black_box(component_a()); }
    let start = Instant::now();
    for _ in 0..iters { black_box(component_a()); }
    let t_a = start.elapsed();
    
    // Component B
    // ... same pattern ...
    
    // Summary
    println!("  Component A: {:.2} μs/rescue", t_a.as_micros() as f64 / iters as f64);
    println!("  Component B: {:.2} μs/rescue", t_b.as_micros() as f64 / iters as f64);
    println!("  Total Δ:     {:.2} μs", (t_a + t_b).as_micros() as f64 / iters as f64);
}
```

## Real-World Results (PPoT Plan 027)

| Optimization | Before | After | Technique |
|---|---|---|---|
| Knowledge queries | 2.67 μs | 0.58 μs | Precomputed `[PositionStats; 16]` |
| position_affinity | 0.75 μs | 0.04 μs | O(1) array lookup vs O(n) scan |
| should_skip | 0.77 μs | 0.03 μs | O(1) accepted/rejected counters |
| Multi-strategy gen | 8.05 μs | 5.33 μs | Cached support sets in config |
| Rank consistency | 7.42 μs | 6.10 μs | Chunked u64 compare (4-at-a-time) |
| Rayon parallel rank | 6.10 μs | 39.2 μs | REMOVED — overhead dominates m≤16 |
| Plan 027 total (debug) | 88.6 μs | 75.9 μs | -14% overall |
| Plan 027 total (release) | — | 4.5 μs | Only 2.6 μs over greedy baseline |