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