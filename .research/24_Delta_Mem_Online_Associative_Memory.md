# Research: δ-Mem Online Associative Memory — Modelless Distillation (24)

> Source: [δ-Mem: Online Associative Memory](https://arxiv.org/abs/2605.12357)
> Date: 2026-05, distilled 2026-06
> Code: `.raw/delta-Mem/deltamem/` (local source audit)
> **Verdict: HIGH VALUE — The delta-rule update mechanism is self-contained and doesn't require neural training. Modelless distillation to feature-hashed associative memory in Rust.**

## TL;DR

The paper introduces an Online Stable Associative Memory (OSAM) module that attaches to any transformer attention layer. It maintains a low-rank square state matrix S ∈ ℝ^{r×r} per layer, updated via a delta-rule: read the current prediction (S·k), subtract the old prediction's contribution, add the new value's contribution. The state persists across tokens and across conversation turns, giving the model a "scratchpad" memory that grows with context without increasing KV-cache size.

**Our verdict:** The core delta-rule update formula is mathematically simple and completely independent of backpropagation. We can implement it with feature hashes instead of learned projections, heuristic gates instead of trained sigmoid gates, and per-domain states instead of multi-head states. No neural training required.

---

## Core Theorem (What We Actually Need)

**Equations 10-12 (Paper) — The Delta-Rule Update:**

```
S_t = λ·S_{t-1} − β·(S_{t-1}·k_t)⊗k_t + β·v_t⊗k_t
```

Where:
- `S_t` ∈ ℝ^{r×r} is the associative memory state (low-rank square matrix)
- `k_t` ∈ ℝ^r is the key vector (what to look up)
- `v_t` ∈ ℝ^r is the value scalar (what to write) — NOTE: in the paper this is a scalar per row
- `λ` = keep gate (how much to retain old state)
- `β` = write/erase gate (how aggressively to update)
- `(S_{t-1}·k_t)⊗k_t` = predicted outer product (what the old state predicted for this key)
- `v_t⊗k_t` = new outer product (what we want to write for this key)

**The key insight:** The "delta" in delta-rule is literally `v_t − S_{t-1}·k_t` — the difference between what we want and what we predicted. The update subtracts the old prediction and adds the new value. This is a second-order gradient descent step on a rank-r approximation of the full associative memory.

**Gate coupling (default):** `λ = 1 − β`, meaning if you write aggressively (high β), you keep less (low λ). This couples retention and writing into a single gate, inspired by gated retention in Qwen-Next.

**Reading:** `read_t = S·q_t` — a simple matrix-vector multiply to retrieve from memory.

---

## Paper Architecture (What We DON'T Need)

| Component | Paper | Why We Skip |
|-----------|-------|-------------|
| Learned memory projections W_mq, W_mk, W_mv | `nn.Parameter` [state_read_dim, hidden_size] | We use feature hashing (LSH random projection) |
| Learned delta projections W∆q, W∆o | `nn.Parameter` [out_features, state_read_dim] | We hash to scalar, no output correction needed |
| Learned gate projections W_β, W_λ | `nn.Parameter` + bias + sigmoid | We compute gates heuristically from δ statistics |
| Triton GPU kernel for affine_scan | CUDA-parallel row scan | CPU sequential scan is fine (r=8 → 8 iterations) |
| Base model (Qwen3/SmolLM3) | 3B+ parameter LLM | We don't attach to a transformer; standalone memory |
| SFT training loop | DeepSpeed ZeRO-2, 8×A800, bf16 | No training needed for modelless approach |
| Multiple training losses | Contrastive, KL, margin, anchor, recovery, sparsity | Not applicable without neural training |
| Session past_key_values | KV-cache persistence | We serialize state to JSONL |

---

## Source Code Verification (`.raw/delta-Mem/deltamem/`)

### 1. File Structure

| File | Lines | Purpose |
|------|-------|---------|
| `core/delta_impl.py` | 2803 | Core: `DeltaMemAttention` class, all projections, scan kernels |
| `core/delta.py` | ~140 | Config re-exports, public API surface |
| `core/write_segmentation.py` | — | Sentence/message chunking for write granularity |
| `kernels/affine_scan.py` | ~370 | Triton kernel + torch fallback for state scan |
| `train/delta_sft_experimental.py` | ~2900 | `DeltaMemTrainer` with multi-objective loss |
| `runtime/session.py` | ~794 | `DeltaMemChatSession` with snapshot/load |
| `runtime/chat.py` | — | Inference wrapper |
| `model_loading.py` | — | Model loading with attention implementation resolution |

### 2. Exact Delta-Rule Implementation

The torch fallback scan in `delta_impl.py` L1895-1938 is the clearest reference implementation:

```python
# delta_impl.py L1917-1929 (verified)
for token_idx in range(seq_len):
    q_t = memory_q_seq[:, token_idx, :]
    k_t = memory_k_seq[:, token_idx, :]
    v_t = memory_v_seq[:, token_idx, :]
    keep_t = keep_seq[:, token_idx, :].unsqueeze(-1)    # λ
    erase_t = erase_seq[:, token_idx, :].unsqueeze(-1)   # β for erase
    write_t = write_seq[:, token_idx, :].unsqueeze(-1)   # β for write

    read_t = torch.einsum("bij,bj->bi", current_state, q_t)      # S·q
    pred_t = torch.einsum("bij,bj->bi", current_state, k_t)      # S·k
    write_outer = v_t.unsqueeze(-1) * k_t.unsqueeze(1)           # v⊗k
    pred_outer = pred_t.unsqueeze(-1) * k_t.unsqueeze(1)         # (S·k)⊗k
    next_state = keep_t * current_state - erase_t * pred_outer + write_t * write_outer
```

This is EXACTLY the paper's Equations 10-12:
- `keep_t * current_state` = `λ·S_{t-1}` (retain)
- `erase_t * pred_outer` = `β·(S_{t-1}·k_t)⊗k_t` (erase old prediction)
- `write_t * write_outer` = `β·v_t⊗k_t` (write new value)

### 3. Config Defaults (Verified `delta_impl.py` L186-250)

```python
# delta_impl.py L187-196 (verified)
class HFDeltaMemConfig:
    rank: int = 8
    alpha: float = 16.0
    beta_bias_init: float = -1.5
    normalize_qk: bool = True
    couple_lambda: bool = True
    state_update_mode: str = "standard"
    rankwise_gates: bool = True
    output_init: str = "zero"           # config default; training script overrides
    base_slice_ref_width: int = 8
    online_gain: float = 0.05
    num_state_heads: int = 1
    num_memory_partitions: int = 1
    delta_heads: tuple[str, ...] = VALID_DELTA_HEADS  # ("q", "k", "v", "o")
```

**Note:** The training script `delta_sft_experimental.py` L1694 overrides `--delta-heads` default to `"q,k,v,o"` (all 4). The training script also sets `--output-init` default to `"base_slice"` (L1689), and `--max-length` default to 1024 (L1747).

### 4. Gate Coupling (`delta_impl.py` L917-928, verified)

```python
# delta_impl.py L925-928 (verified)
if self.couple_lambda:
    lam = 1.0 - beta                    # λ = 1 - β (coupled)
else:
    lam = torch.sigmoid(                # λ = separate learned gate
        gate_splits[1] + self.lambda_bias.view(...)
    ).unsqueeze(-1)
```

When `couple_lambda=True` (default), there is no separate `lambda_proj` — the keep gate is simply `1 − β`. The `lambda_proj` and `lambda_bias` are only allocated when `couple_lambda=False` (L556-561).

### 5. Projections (`delta_impl.py` L588-599, verified)

```python
# delta_impl.py L588-599 (verified)
self.memory_q_proj = nn.Parameter(torch.empty(self.state_read_dim, hidden_size))
self.memory_k_proj = nn.Parameter(torch.empty(self.state_read_dim, hidden_size))
self.memory_v_proj = nn.Parameter(torch.empty(self.state_read_dim, hidden_size))

self.delta_q_proj = nn.Parameter(torch.empty(base.q_proj.out_features, self.state_read_dim))
self.delta_k_proj = nn.Parameter(torch.empty(base.k_proj.out_features, self.state_read_dim))
self.delta_v_proj = nn.Parameter(torch.empty(self.base_v_out_features, self.state_read_dim))
self.delta_o_proj = nn.Parameter(torch.empty(base.o_proj.out_features, self.state_read_dim))
```

All projections are bare `nn.Parameter` (not `nn.Linear`) — just weight matrices, no bias. Memory projections map hidden→state_dim, delta projections map state_dim→output_features.

### 6. Delta Head Application (`delta_impl.py` L855-885, verified)

```python
# delta_impl.py L873-885 (verified)
query_states = self.base.q_proj(hidden_states)
if delta_q is not None:
    query_states = query_states + delta_q.to(hidden_states.dtype)
key_states = self.base.k_proj(hidden_states)
if delta_k is not None:
    key_states = key_states + delta_k.to(hidden_states.dtype)
value_states = self.base.v_proj(hidden_states) if self.base.v_proj is not None else key_states
if delta_v is not None:
    value_states = value_states + delta_v.to(hidden_states.dtype)
```

Only active delta heads contribute corrections. Inactive heads return `None` from `_project_delta_head`, so the addition is skipped. The final output correction is: `attn_output = self.base.o_proj(attn_output) + delta_o`.

### 7. Output Initialization (`delta_impl.py` L667-687, verified)

```python
# delta_impl.py L671-687 (verified)
def _init_delta_head(self, head: nn.Parameter, base_weight: torch.Tensor) -> None:
    if self.output_init == "zero":
        nn.init.zeros_(head)
        return
    # ... base_slice / base_slice_fixed: initializes from normalized slice of base weight
    base_slice = base_weight[:, :slice_width].detach().clone().float()
    base_slice = F.normalize(base_slice, dim=0, eps=1e-6)
    head[:, :slice_width].copy_((base_slice * self.online_gain).to(head.dtype))
```

With `output_init="base_slice"` (training default), delta heads start from a *meaningful initialization* — a normalized slice of the base attention projection weights, scaled by `online_gain=0.05`. This is NOT zero init.

### 8. Multi-Head State / MSW (`delta_impl.py` L580-583, L795-803, verified)

```python
# delta_impl.py L580-583 (verified)
self.num_state_heads = config.num_state_heads   # default=1
self.state_read_dim = self.rank * self.num_state_heads
self.multi_head_state = self.num_state_heads > 1
```

With `num_state_heads=4` (MSW config): state shape is `[batch, 4, rank, rank]`. Each head scans independently. Reads are concatenated across heads to form `state_read_dim = 4 × rank = 32`.

### 9. Write Granularity (`delta_impl.py` L932-938, verified)

Three modes controlled by `memory_write_granularity`:
- `"token"`: write at every token (TSW — token-level state write)
- `"message_mean"`: average hidden states per message span, write once (SSW — segment-level state write)
- `"sentence_mean"`: average per sentence chunk, write once (SSW variant)

Message/sentence averaging uses `write_message_ids` / `write_sentence_ids` tensors passed through the forward pass. This dramatically reduces write frequency for long conversations.

### 10. Training Losses (`delta_sft_experimental.py` L90-116, verified)

Multiple loss components with configurable weights (defaults from constructor):

| Loss | Default Weight | Purpose |
|------|---------------|---------|
| `memory_contrast_weight` | 0.1 | Contrastive between context/no-context |
| `memory_kl_weight` | 0.1 | KL divergence between memory and no-memory outputs |
| `memory_margin` | 0.1 | Margin between context and no-context CE losses |
| `memory_causal_weight` | 1.0 | Main SFT causal loss |
| `memory_anchor_weight` | 1.0 | Anchor loss (keep close to teacher) |
| `memory_anchor_margin` | 0.005 | Anchor margin |
| `memory_recover_weight` | 0.25 | Recovery from corrupted state |
| `write_sparsity_weight` | 0.0 | Encourage sparse β (disabled by default) |
| `memory_dropout_no_memory_prob` | 0.0 | Ablation: drop memory entirely |
| `memory_dropout_state_only_prob` | 0.0 | Ablation: drop KV cache, keep state |

The `memory_loss_mode` parameter selects which combination of these losses to use. The `"state_causal_anchor"` mode (L598-603) combines margin + causal + anchor losses.

### 11. Session Management (`session.py` L211-297, verified)

```python
# session.py L211-216 (verified)
@dataclass(frozen=True)
class DeltaMemSessionSnapshot:
    messages: list[dict[str, str]]
    processed_input_ids: list[int]
    delta_state: dict[str, torch.Tensor]
    past_key_values: object | None = None
    write_message_ids: list[int] = field(default_factory=list)
    write_sentence_ids: list[int] = field(default_factory=list)
```

- `snapshot()` (L252-263): saves messages + delta_state + past_key_values + write IDs
- `load_snapshot()` (L265-297): restores full conversation state, calls `reset_delta_mem_states` then `load_delta_mem_online_state`
- `reset()` (L236-247): clears messages + calls `reset_delta_mem_states(model)`
- State persists across turns within a session

### 12. Triton Kernel (`affine_scan.py` L104-172, verified)

Per-row parallelism: each Triton program handles one row of the state matrix. The grid is `(batch_size * rank,)`, so for rank=8 there are 8 parallel programs per batch element.

```python
# affine_scan.py L115-116 (verified)
pid = tl.program_id(axis=0)
batch_idx = pid // rank
row_idx = pid % rank
```

Sequential scan over tokens within the kernel (sequential dependency — can't parallelize across time). Stores state history for backward pass in `state_hist` tensor of shape `[batch, seq, rank, rank]`.

### 13. Training Hyperparameters (`delta_sft_experimental.py`, verified)

From argparse defaults in the training script:

| Parameter | Default | Source Line |
|-----------|---------|-------------|
| `rank` | 8 | config |
| `alpha` | 16.0 | config |
| `beta_bias_init` | -1.5 | config |
| `couple_lambda` | True | config |
| `state_update_mode` | "standard" | config |
| `rankwise_gates` | True | config |
| `online_gain` | 0.05 | config |
| `delta_heads` | "q,k,v,o" | L1694 |
| `output_init` | "base_slice" | L1689 |
| `base_slice_ref_width` | 8 | L1690 |
| `max_length` | 1024 | L1747 |
| `max_write_length` | 1024 | L1761 |
| `learning_rate` | 2e-4 | L1782 |
| `lr_scheduler_type` | "cosine" | L1784 |
| `warmup_ratio` | 0.10 | L1785 |
| `num_train_epochs` | 1.0 | L1787 |
| `per_device_train_batch_size` | 1 | L1779 |

---

## Mapping to Our Stack

```
Paper (δ-Mem)                           Our Stack (Modelless)
─────────────────────                    ─────────────────────────
OSAM state S [batch, heads, r, r]  ←→   DeltaMemoryState: Vec<f32> (r×r flat)
Memory projections W_mq/W_mk/W_mv  ←→   FeatureHasher (LSH random projection)
Delta projections W∆q/W∆o          ←→   Fixed random projection (hash to scalar)
Gate projections W_β, W_λ          ←→   Heuristic from δ statistics
Triton affine_scan                  ←→   CPU sequential scan (r=8 → 8 row iterations)
Multi-head state (MSW)              ←→   Per-domain states from PromptRouter
Message write (SSW)                 ←→   Average per DDTree build
Session snapshot (pickle)           ←→   Serialize state to JSONL
beta_bias_init=-1.5                 ←→   sigmoid(-1.5) ≈ 0.18, initial write rate
normalize_qk (tanh + L2 norm)      ←→   blake3 hash → normalize to unit vector
```

### Architecture Mapping Table

| Paper Component | Paper Implementation | Our Modelless Equivalent |
|---|---|---|
| OSAM state S | `[batch, heads, rank, rank]` float tensor | `DeltaMemoryState` r×r `Vec<f32>` |
| Memory projections W_mq/W_mk/W_mv | Learned `nn.Parameter` [state_read_dim, hidden] | `FeatureHasher` (LSH random projection) |
| Delta projections W∆q/W∆o | Learned `nn.Parameter` [out_features, state_read_dim] | Fixed random projection (hash to scalar) |
| Gate projections W_β, W_λ | Learned `nn.Parameter` + bias + sigmoid | Heuristic from δ statistics |
| Triton affine_scan | GPU-parallel row scan | CPU sequential scan (r=8 → 8 iterations) |
| Multi-head state (MSW) | 4 parallel `[rank, rank]` states | Per-domain states from `PromptRouter` |
| Message write (SSW) | Average per message span | Average per DDTree build |
| Session snapshot | pickle state dict | Serialize state to JSONL |
| KV-cache persistence | `DynamicCache` in snapshot | Not needed (stateless per request) |

---

## Modelless Distillations

### D1: DeltaMemoryState — Core State Container

A `Vec<f32>` of length r×r (default r=8, so 64 floats = 256 bytes). Implements:
- `read(q: &[f32]) -> Vec<f32>`: matrix-vector multiply S·q
- `write(k: &[f32], v: f32, beta: f32)`: delta-rule update S' = (1-β)S - β(S·k)⊗k + β·v·k⊗k
- `reset()`: zero the state
- `serialize()/deserialize()`: JSONL-friendly

The entire state for one domain is 256 bytes. For 100 domains, that's 25 KB — negligible.

### D2: FeatureHashed Projections

Instead of learned `nn.Parameter` projections:
- `hash_to_key(input: &str) -> [f32; 8]`: blake3 hash → normalize to unit vector
- `hash_to_query(input: &str) -> [f32; 8]`: separate blake3 hash → normalize
- `hash_to_value(outcome: &Outcome) -> f32`: blake3 hash → map to [-1, 1] range

No learning. The random projection acts as a locality-sensitive hash — similar inputs produce similar keys, dissimilar inputs produce orthogonal keys (with high probability).

### D3: Heuristic Gates

Instead of learned sigmoid gates with `beta_bias_init=-1.5`:
- `compute_beta(delta: f64) -> f32`: `sigmoid(a * delta + b)` where a, b are tuned constants
- `compute_lambda(beta: f32) -> f32`: `1.0 - beta` (coupled, same as paper default)
- High δ → high β (write aggressively when outcomes are surprising)
- Low δ → low β (don't overwrite stable knowledge)

### D4: Per-Domain Multi-Head State

Instead of `num_state_heads=4` with learned routing:
- Each domain from `PromptRouter` gets its own `DeltaMemoryState`
- Reading: look up domain, read from that state
- Writing: write to the domain's state
- No routing needed — the domain is determined by the request context

### D5: Segment-Level Writes (SSW)

Instead of per-token writes:
- Average all feature hashes for a complete DDTree build
- Write once per build with the averaged key/value
- Reduces write frequency by 10-100× for typical builds
- Mirrors the paper's `message_mean` write granularity

---

## Experimental Results (Paper)

### Core Metrics

The paper evaluates on multi-turn conversation benchmarks:

| Benchmark | Metric | δ-Mem | Baseline (no memory) |
|-----------|--------|-------|---------------------|
| QASPER | F1 | +significant | — |
| LocoMT | Score | +significant | — |
| Memory Agent Bench | Score | +significant | — |

### Key Findings from Code

1. **State is tiny**: rank=8, so state is 8×8=64 floats per layer. Even with all layers, total memory overhead is KB-scale vs MB-scale for KV-cache.

2. **Gate initialization matters**: `beta_bias_init=-1.5` means `sigmoid(-1.5) ≈ 0.18` — the model starts conservative (low write rate) and learns to write more aggressively.

3. **Output init matters**: `base_slice` initialization (training default) starts delta corrections from a meaningful projection of the base model's weights, not zero. This gives faster convergence than zero init.

4. **Write granularity tradeoff**: Token-level writes (TSW) are most expressive but expensive. Message-level writes (SSW) are much cheaper with small quality loss. The `memory_write_granularity` parameter lets you trade off.

5. **Coupled gates work well**: `couple_lambda=True` (default) with `λ = 1 - β` is sufficient — separate learned gates don't help enough to justify the extra parameters.

6. **State normalization**: `normalize_qk=True` (default) applies tanh + L2 normalization to queries and keys, keeping them on the unit sphere. This prevents state explosion.

---

## Relationship to Existing Work

| This Paper | Our Existing |
|------------|-------------|
| OSAM state S | LoRA adapter state (weight matrices) |
| Delta corrections to Q/O | ScreeningPruner relevance adjustments |
| Gate β from sigmoid(W·h+b) | BanditPruner δ-based gating |
| Multi-head state per layer | Per-domain routing from PromptRouter |
| Session snapshot/load | Artifact JSONL serialization |
| Message-level write averaging | DDTree build aggregation |
| KL loss between memory/no-memory | DeltaBanditPruner δ divergence signal |

### What Won't Transfer

- The learned projections (W_mq, W_mk, W_mv, W∆q, W∆o) — these are specific to the base model's hidden dimension and require training. We replace with feature hashes.
- The Triton GPU kernel — our state is small enough for CPU.
- The full training pipeline with 6+ loss terms — we don't train.
- Attachment to transformer attention layers — our memory is standalone, not patching Q/K/V/O.
- KV-cache persistence — our system is stateless per request; we serialize state separately.

### Key Insight for Modelless

The delta-rule update `S' = (1-β)S - β(S·k)⊗k + β·v⊗k` is a **second-order gradient descent step** on a rank-r approximation. It converges to the correct association in O(1/β) steps for stationary distributions. For non-stationary distributions (which is our use case — game balance changes), the `λ = 1-β` decay ensures old associations fade naturally.

The paper proves that with `normalize_qk=True` (keys on unit sphere) and coupled gates, the state norm remains bounded. This is critical for our modelless version — we must normalize our feature hashes to prevent state explosion.

## Plan 053 Implementation Results

**Status:** Feature-gated (`delta_mem = ["bandit"]`), **NOT in default features.**

### What We Built (Plan 053)

All distillations D1–D5 implemented in `src/pruners/delta_mem/`:
- `state.rs` — DeltaMemoryState (r×r associative matrix, delta-rule read/write)
- `hash.rs` — FeatureHasher (LSH projection replacing learned W_mq/W_mk/W_mv)
- `pruner.rs` — MemorySteeredPruner<P> (wraps any ScreeningPruner with memory corrections)
- `multi.rs` — MultiDomainMemory (per-domain isolated states, MSW adaptation)
- `multi_pruner.rs` — MultiDomainMemoryPruner<P> (domain-routed pruner)

### DDTree Proof Test Results

| Pruner | Avg Nodes | Time | vs Baseline |
|--------|-----------|------|-------------|
| NoScreeningPruner | 256.0 | 44ms | baseline |
| Fresh MemorySteeredPruner | 256.0 | 1.13s | +2470% |
| Trained MemorySteeredPruner (200 obs) | 256.0 | 1.14s | +2492% |

### What Works

- ✅ Delta-rule math converges (cosine similarity ≤0.20 alignment error after 200 updates)
- ✅ Domain isolation is perfect (0% cross-interference in MultiDomainMemory)
- ✅ Fresh memory = zero corruption (identical to NoScreeningPruner baseline)
- ✅ State norm bounded with L2-normalized keys (<200 after 500 writes)
- ✅ Coupled gates (λ=1-β) produce smaller state norm than uncoupled
- ✅ Snapshots serialize/deserialize correctly for persistence

### What Doesn't Work

- ❌ No DDTree quality improvement (corrections too small to flip branch ordering)
- ❌ 26× latency overhead (FeatureHasher + matmul per `relevance()` call, ~682 calls/build)
- ❌ Tree fills budget regardless of relevance adjustments

### Why No Gain for DDTree

The paper corrects **attention Q/O projections across all layers** of a 4B+ parameter model — that's hundreds of dimensions being nudged in a learned representation space. We correct a **single scalar relevance score** in a tree search. The correction surface is too simple — there's not enough "model" to correct. The value proposition is specifically for Transformer attention correction, not for tree-based relevance scoring.

### Path Forward

The infrastructure is correct and may have value in other contexts:
- **Transformer integration** — if riir-gpu gets a Transformer, these same primitives apply corrections where they matter (attention layers, not tree scoring)
- **Caching** — cache relevance per (depth, token_idx) to amortize hashing cost
- **Offline analysis** — use memory state to analyze blind spots without hot-path overhead
- **Not recommended for DDTree hot-path** — the 26× overhead is unacceptable for per-build use

**See also:**
- Research 21 (G-Zero) — δ is the same signal used to gate writes in our modelless delta memory
- Research 20 (TurboQuant) — online vector quantization for feature hashing
- Research 22 (Lighthouse Attention) — alternative long-context attention mechanism
- Research 07 (Screening Absolute Relevance) — the relevance signal that could feed into memory reads
- Plan 053 (`.plans/053_delta_mem_modelless.md`) — full implementation details and benchmarks