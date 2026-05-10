# Handover 006: Systems Optimization (GQA + Paged KV Cache)

## What Happened

Implemented Plan 011: Systems Optimization with two major features:

1. **Grouped-Query Attention (GQA)** — Allows `n_kv_head < n_head` to shrink KV cache. Added `n_kv_head` field to `Config`, updated the full forward pass to support GQA with `kv_dim` stride and KV group mapping, and created `Config::gqa_draft()` (8 Q heads, 2 KV heads = 4× KV cache reduction).

2. **Paged KV Cache** — Copy-on-write page allocation for DDTree branch exploration. `PagedKVCache` struct with `fork()` for prefix sharing, `write_kv`/`read_kv` for per-layer per-position access, and `reset()` for page reuse.

**Key invariant**: When `n_kv_head == n_head`, behavior is **identical** to the old MHA code (verified by existing tests still passing with zero regressions).

## Where Is the Plan/Code/Test

- **Plan**: `.plans/011_systems_optimization.md`
- **Code**:
  - `src/types.rs` — `n_kv_head` field, `kv_dim()` helper, `gqa_draft()`, `validate()`
  - `src/transformer.rs` — GQA-aware `attention_head()`, updated `forward()`, `KVCache`, `ForwardContext`, `TransformerWeights`, new `PagedKVCache`
  - `tests/integration.rs` — Updated `test_forward_cache_populated` to use `kv_dim`
- **Tests**: 10 new tests in `src/transformer.rs`:
  - `test_gqa_produces_valid_logits`
  - `test_gqa_mha_backward_compat`
  - `test_gqa_kv_cache_smaller`
  - `test_gqa_generate_valid_tokens`
  - `test_config_validate_gqa`
  - `test_paged_cache_write_read_roundtrip`
  - `test_paged_cache_linear_matches_flat`
  - `test_paged_cache_fork_no_corruption`
  - `test_paged_cache_fork_shares_prefix`
  - `test_paged_cache_reset_frees_pages`

## Reflection: Struggling/Solved

- **Borrow checker in `ensure_pages`**: The original `for layer_tables in &mut self.layer_page_tables` + `self.alloc_page()` caused double mutable borrow. Solved by splitting into three phases: (1) grow sequence slots, (2) compute deficits and allocate pages, (3) assign pages to tables.
- **Backward compatibility**: Carefully verified that all existing configs have `n_kv_head == n_head`, ensuring `kv_dim == n_embd` and `kv_group_offset == q_head_offset` — identical behavior to the old code. All 168 existing tests pass unchanged.
- **File size**: `transformer.rs` is at 994 lines (under 1024 limit).

## Remain Work

From Plan 011, still TODO:
- [ ] 2.7 Add benchmark: MHA vs GQA throughput
- [ ] 3.3 Add `forward_paged()` variant that uses `PagedKVCache` directly
- [ ] 4.1-4.4 DDTree integration with PagedKVCache (replace flat cache clones)
- [ ] 5.3 Run `cargo run --release` — verify benchmark unchanged for micro config
- [ ] 5.4 Add benchmark suite: MHA, GQA, flat cache, paged cache

## Issues Ref

No issues created yet. No blocking issues.

## How to Dev/Test

```bash
# Run all tests (178 total: 98 lib + 80 integration)
cargo test --quiet

# Run clippy (zero warnings)
cargo clippy --quiet

# Run specific GQA tests
cargo test --quiet gqa
cargo test --quiet paged_cache

# Run release benchmark (verify no regression)
cargo run --release
```
