# Plan 011: Systems Optimization — Paged KV Cache + Grouped-Query Attention

## Objective

Optimize the transformer inference engine for production workloads: paged KV cache to prevent memory fragmentation from DDTree branching, and Grouped-Query Attention (GQA) to shrink KV cache size by sharing K/V heads across Q heads.

## The Problem

### Paged KV Cache

Current `KVCache` is a flat `Vec<f32>` of `[block_size, n_embd]`. When DDTree branches explore different token paths, each branch needs its own cache state. Current approach:
- Clone the entire cache for each branch → O(n_layer * block_size * n_embd * 4) bytes per branch
- With `tree_budget=32`, `n_embd=256`, `block_size=256`, that's 32 × 4 × 256 × 256 × 4 = 336 MB of cache clones
- Most of this memory is unused (only `pos` slots are filled)

### GQA

Current multi-head attention has `n_head` Q heads and `n_head` K/V heads (1:1 ratio). For `n_head=8, n_embd=256`, the KV cache per layer is `2 × block_size × 256` floats. GQA shares K/V heads:
- `n_kv_head < n_head` — e.g., `n_head=8, n_kv_head=2` → 4× KV cache reduction
- Used in Llama-2, Mistral, and most modern LLMs

## Architecture

### Paged KV Cache

```rust
// transformer.rs — paged KV cache

/// Page size in tokens (tuneable, must be power of 2).
const PAGE_SIZE: usize = 16;

/// Paged KV cache: allocates memory in fixed-size pages.
/// Each sequence (branch) gets its own page table.
/// Pages are shared between branches that share a prefix.
pub struct PagedKVCache {
    /// Pool of pages. Each page: [PAGE_SIZE, n_kv_embd] where n_kv_embd = n_kv_head * head_dim.
    pages: Vec<Vec<f32>>,
    /// Per-layer page tables. layer_page_tables[layer][seq_idx] = vec of page indices.
    layer_page_tables: Vec<Vec<Vec<usize>>>,
    /// Free list of page indices for reuse.
    free_pages: Vec<usize>,
    /// Dimension of each KV entry (n_kv_head * head_dim).
    kv_dim: usize,
    /// Total pages allocated.
    total_pages: usize,
}

impl PagedKVCache {
    pub fn new(config: &Config, max_sequences: usize) -> Self {
        let kv_dim = config.n_head * config.head_dim; // or n_kv_head * head_dim with GQA
        let initial_pages = config.block_size / PAGE_SIZE;
        
        Self {
            pages: (0..initial_pages * config.n_layer)
                .map(|_| vec![0.0; PAGE_SIZE * kv_dim])
                .collect(),
            layer_page_tables: (0..config.n_layer)
                .map(|_| (0..max_sequences).map(|_| Vec::new()).collect())
                .collect(),
            free_pages: Vec::new(),
            kv_dim,
            total_pages: initial_pages * config.n_layer,
        }
    }
    
    /// Allocate a new page. Reuse from free list or grow the pool.
    fn alloc_page(&mut self) -> usize {
        if let Some(idx) = self.free_pages.pop() {
            self.pages[idx].fill(0.0);
            idx
        } else {
            self.pages.push(vec![0.0; PAGE_SIZE * self.kv_dim]);
            let idx = self.total_pages;
            self.total_pages += 1;
            idx
        }
    }
    
    /// Copy-on-write: fork a sequence's page table at a given position.
    /// Shares prefix pages, allocates new pages for divergent suffix.
    pub fn fork(&mut self, seq_idx: usize, fork_at_pos: usize) -> usize {
        let new_seq = self.layer_page_tables[0].len();
        for layer_tables in &mut self.layer_page_tables {
            let source = &layer_tables[seq_idx];
            let fork_page = fork_at_pos / PAGE_SIZE;
            let mut new_table = source[..fork_page].to_vec();
            // Remaining pages will be allocated as needed
            layer_tables.push(new_table);
        }
        new_seq
    }
}
```

### Grouped-Query Attention

```rust
// types.rs — GQA config

pub struct Config {
    // ... existing fields ...
    pub n_kv_head: usize,  // NEW: number of K/V heads (≤ n_head)
}

impl Config {
    pub fn micro() -> Self {
        Self {
            // ... existing ...
            n_head: 4,
            head_dim: 4,
            n_kv_head: 4,  // standard MHA (1:1)
        }
    }
    
    /// GQA config: 8 Q heads, 2 KV heads (4:1 ratio).
    pub fn gqa_draft() -> Self {
        Self {
            vocab_size: 4096,
            block_size: 256,
            n_embd: 64,
            n_head: 8,
            head_dim: 8,
            n_kv_head: 2,     // 4× KV cache reduction
            mlp_hidden: 256,
            n_layer: 4,
            bos_token: 0,
            temperature: 0.8,
            draft_lookahead: 5,
            tree_budget: 32,
        }
    }
}
```

```rust
// transformer.rs — GQA forward

/// Multi-head attention with GQA support.
/// When n_kv_head < n_head, each KV head is shared by (n_head / n_kv_head) Q heads.
fn attention_head_gqa(
    q: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    attn_out: &mut [f32],
    scores_buf: &mut [f32],
    q_head: usize,       // which Q head
    kv_group: usize,     // which KV group (q_head * n_kv_head / n_head)
    n: usize,
    hd: usize,
    kv_dim: usize,       // n_kv_head * head_dim
    t_n: usize,
    scale: f32,
) {
    let q_off = q_head * hd;
    let kv_off = kv_group * hd;
    
    // Q head q_off, K/V head kv_off
    // ... same attention logic, but K/V offset uses kv_group instead of q_head ...
}
```

### KV Cache Size Comparison

| Config | n_head | n_kv_head | head_dim | KV per token (per layer) | Reduction |
|--------|--------|-----------|----------|-------------------------|-----------|
| `micro` | 4 | 4 | 4 | 2 × 4 × 4 = 32 floats | — |
| `bpe` | 4 | 4 | 8 | 2 × 4 × 8 = 64 floats | — |
| `gqa_draft` | 8 | 2 | 8 | 2 × 2 × 8 = 32 floats | 4× smaller |
| cLoRA target | 8 | 2 | 32 | 2 × 2 × 32 = 128 floats | 4× smaller |

For a 4-layer model with `block_size=256`:
- MHA: 4 × 2 × 256 × 8 × 32 = 524 KB
- GQA: 4 × 2 × 256 × 2 × 32 = 131 KB → **4× reduction**

## Tasks

### Phase 1: GQA Config
- [ ] 1.1 Add `n_kv_head: usize` to `Config`
- [ ] 1.2 Add `n_kv_head` to all existing Config constructors (set = n_head for MHA)
- [ ] 1.3 Add `Config::gqa_draft()` with `n_kv_head: 2`
- [ ] 1.4 Add validation: `n_head % n_kv_head == 0`
- [ ] 1.5 Run `cargo test` — all pass (n_kv_head = n_head = same behavior)

### Phase 2: GQA Forward Pass
- [ ] 2.1 Add `kv_dim = n_kv_head * head_dim` to forward()
- [ ] 2.2 Compute KV group: `kv_group = q_head * n_kv_head / n_head`
- [ ] 2.3 Resize KV cache to `[block_size, kv_dim]` instead of `[block_size, n_embd]`
- [ ] 2.4 Update `attention_head` → `attention_head_gqa` with kv_group parameter
- [ ] 2.5 Add test: GQA produces valid (finite) logits
- [ ] 2.6 Add test: `n_kv_head == n_head` produces identical results to old code
- [ ] 2.7 Add benchmark: MHA vs GQA throughput

### Phase 3: Paged KV Cache
- [ ] 3.1 Create `PagedKVCache` struct in `transformer.rs`
- [ ] 3.2 Implement page allocation, fork, and free
- [ ] 3.3 Add `forward_paged()` variant that uses `PagedKVCache`
- [ ] 3.4 Add test: paged cache produces same results as flat cache for linear sequence
- [ ] 3.5 Add test: fork + write doesn't corrupt parent sequence
- [ ] 3.6 Add test: page reuse after free

### Phase 4: DDTree Integration
- [ ] 4.1 Create `PagedMultiLayerKVCache` (paged + multi-layer)
- [ ] 4.2 Update DDTree branch exploration to use paged cache
- [ ] 4.3 Benchmark: memory usage with `tree_budget=32` — flat clone vs paged
- [ ] 4.4 Benchmark: DDTree build throughput — flat vs paged

### Phase 5: Validation
- [ ] 5.1 Run `cargo test --all-features`
- [ ] 5.2 Run `cargo clippy --all-features`
- [ ] 5.3 Run `cargo run --release` — benchmark unchanged for micro config
- [ ] 5.4 Add benchmark suite: MHA, GQA, flat cache, paged cache

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Paged cache slower for linear generation | Page table indirection overhead | Benchmark; use flat cache for non-DDTree paths |
| GQA quality reduction | Slightly worse acceptance rate | Configurable: MHA for target, GQA for draft |
| Page fragmentation | Memory waste | Compact pages periodically; use power-of-2 page sizes |
| Complex fork semantics | Bugs in branch management | Extensive testing with DDTree stress tests |

## Expected Outcomes

1. `Config.n_kv_head` — configurable KV head count
2. GQA support in `forward()` — 4× KV cache reduction for draft models
3. `PagedKVCache` — copy-on-write fork for DDTree branches
4. Memory usage: O(tree_budget × pos_used) instead of O(tree_budget × block_size)
5. No performance regression for existing `micro`/`draft` configs (MHA mode)

## Files to Create/Modify

| File | Action | Phase |
|------|--------|-------|
| `src/types.rs` | Add `n_kv_head` to Config | 1 |
| `src/transformer.rs` | GQA forward, PagedKVCache | 2-3 |
| `src/speculative/dd_tree.rs` | Paged cache integration | 4 |
| `src/benchmark.rs` | GQA + paged cache benchmarks | 5 |

## References

- `.research/01_Advanced Neuro-Symbolic Rust Translation.md` — §Paged Memory, §GQA
- `.research/00_Neuro-Symbolic LLM Architecture.md` — §Hardware Improvements
- [GQA: Efficient Inference with GQA](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
- [vLLM PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al., 2023