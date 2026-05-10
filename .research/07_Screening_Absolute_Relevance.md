# Research: Screening Is Enough — Absolute Relevance & Hard Rejection

**Date:** 2025-06
**Status:** Research → Verdict
**Context:** microgpt-rs + anyrag + riir-validator-sdk neuro-symbolic architecture
**Paper:** "Screening Is Enough" (arXiv:2604.01178) — Skipping Top-K for Absolute Relevance

---

## TL;DR

Standard attention/RAG systems use softmax, which **forces competition** — it must distribute probability mass that sums to 1.0. If you provide 5 garbage documents, the LLM *has* to pay attention to at least one. "Screening Is Enough" replaces this with **absolute relevance thresholds**: each item independently passes or fails a screening filter, bypassing softmax competition entirely.

Applied to microgpt-rs: upgrade the binary `ConstraintPruner` (`is_valid -> bool`) into a continuous `ScreeningPruner` (`relevance -> f32 ∈ [0.0, 1.0]`). This blends LLM semantic probabilities with deterministic relevance scores in log-probability space.

---

## The Problem: Softmax Forces Competition

### In RAG (anyrag)
- Vector DB returns Top-K chunks by cosine similarity (softmax-equivalent)
- If K=5 but only 2 chunks are actually relevant, the LLM *must* attend to all 5
- The irrelevant 3 dilute context, cause hallucination, waste inference budget

### In Speculative Decoding (DDTree)
- Current `ConstraintPruner` is binary: `is_valid() -> bool`
- A move can be *physically valid* but *tactically terrible* (e.g., walking away from goal)
- Binary pruning can't express "valid but suboptimal" — it's either kept at full score or killed

### In the "Screening Is Enough" Paper
- Softmax's normalization constraint means irrelevant items steal probability mass
- The fix: **independent screening** — each item gets an absolute relevance score
- Items below threshold are **trimmed** (hard rejection), not softly penalized via softmax

---

## The Mathematical Core

### Current: Binary Pruning in DDTree

```
Score = Σ ln(P_llm(token_i))     if is_valid() == true
        -∞ (rejected)             if is_valid() == false
```

The LLM's log-prob is either kept untouched (valid) or the branch is killed (invalid).

### Proposed: Graded Screening

```
Score = Σ [ln(P_llm(token_i)) + ln(R(token_i))]
```

Where `R ∈ [0.0, 1.0]` is the absolute relevance from the deterministic screener.

| Relevance R | ln(R) | Effect |
|---|---|---|
| 1.0 | 0.0 | No penalty. LLM score untouched. Perfect match. |
| 0.8 | -0.22 | Slight penalty. Good match. |
| 0.5 | -0.69 | Moderate penalty. Mediocre match. |
| 0.1 | -2.30 | Heavy penalty. Poor match, unlikely to win. |
| 0.0 | -∞ | **Hard Trim.** Branch instantly destroyed. |

### Why This Works

The beauty is that **Screening subsumes Binary Pruning** as a special case:
- `relevance() = 1.0` everywhere → identical to `NoPruner`
- `relevance() = 0.0` or `1.0` only → identical to current `ConstraintPruner`
- Any intermediate value → new graded behavior

The log-space addition means:
1. **Hard rejection at R=0.0**: `ln(0) = -∞`, mathematically identical to current pruning
2. **Graded softmask at 0.0 < R < 1.0**: proportional penalty in probability space
3. **Perfect pass-through at R=1.0**: `ln(1) = 0`, no distortion

---

## Integration Points Across Repos

### microgpt-rs: DDTree Upgrade
- `ConstraintPruner` trait in `src/speculative/types.rs` gains `relevance()` method
- `build()` in `src/speculative/dd_tree.rs` adds `relevance.ln()` to score
- All existing pruners (sudoku, blue bear, dungeon, tactical) get backward-compat blanket impl

### anyrag: Document Screening
- Replace Top-K retrieval with **Screening-based retrieval**
- Each document chunk gets absolute relevance from metadata (date, author, source quality)
- Chunks below threshold are trimmed regardless of cosine similarity score
- Solves the "garbage in, garbage out" RAG problem

### riir-validator-sdk: WASM ABI Extension
- Current ABI: `is_valid(depth, token_idx, ptr, len) -> u32` (0 or 1)
- Proposed ABI: add `relevance(depth, token_idx, ptr, len) -> u32` (fixed-point Q16.16)
- Backward compat: if `relevance` export missing, fall back to `is_valid` (1.0 or 0.0)
- **Constraint**: WASM validators forbid floating-point for determinism → use fixed-point encoding

---

## The Synergy Triangle

```
                    Raven (Fixed Memory Slots)
                    O(1) routing, sparse write
                           │
                           ▼
              ┌─────────────────────────┐
              │   microgpt-rs DDTree    │
              │   Speculative Decoding  │
              │   + Screening Scorer    │
              └─────────────────────────┘
                           ▲
                           │
                    Screening (Absolute Relevance)
                    Bypasses softmax competition
```

- **Raven**: Provides O(1) memory slots for long-context recall (the *memory*)
- **DDTree**: Provides speculative drafting with tree search (the *reasoning*)
- **Screening**: Provides absolute relevance scoring (the *judgment*)

Together: an inference engine that uses LLMs for **semantic creativity** but strictly controls them with **deterministic absolute reality**.

---

## Key Design Decisions

### 1. Fixed-Point Relevance in WASM (not f32)
WASM validators must be deterministic across platforms. IEEE 754 floating-point can vary. Use Q16.16 fixed-point:
- `0x00000000` = 0.0 (hard trim)
- `0x00010000` = 1.0 (perfect match)
- `0x00008000` = 0.5 (50% relevance)
- `0x00FFFFFF` ≈ 0.999985 (nearly perfect)

Host-side conversion: `relevance_f32 = raw_u32 as f32 / 65536.0`

### 2. Backward Compatibility via Blanket Impl
```rust
pub trait ScreeningPruner: Send + Sync {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32;
}

// Auto-impl: any ConstraintPruner is a ScreeningPruner with binary relevance
impl<T: ConstraintPruner + Send + Sync> ScreeningPruner for T {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        if self.is_valid(depth, token_idx, parent_tokens) { 1.0 } else { 0.0 }
    }
}
```

### 3. Threshold Configuration
Add `screening_threshold: f32` to `Config`:
- Default: `0.0` (all non-zero relevance passes — pure softmask mode)
- Can set to e.g. `0.3` to aggressively trim low-relevance branches
- When threshold > 0.0, acts as hard filter *and* soft penalizer

### 4. NaN Safety
`0.0f32.ln()` returns `f32::NEG_INFINITY` in Rust (not NaN). This is correct — it kills the branch. But we should clamp explicitly:
```rust
let relevance = screener.relevance(depth, i, parent_tokens);
if relevance <= 0.0 { continue; } // hard trim, skip branch
let penalty = relevance.ln(); // guaranteed finite since relevance > 0.0
```

---

## Concrete Example: Blue Bear Solver

Current binary pruner:
- Move Up toward monster: `is_valid = true` → score unchanged
- Move Left away from monster: `is_valid = true` → score unchanged  
- Move into wall: `is_valid = false` → branch killed

With ScreeningPruner:
- Move Up toward monster: `relevance = 1.0` → no penalty
- Move Left away from monster: `relevance = 0.3` → -1.20 penalty
- Move into wall: `relevance = 0.0` → hard trim

The DDTree now naturally prefers moves that advance toward the goal, without needing a separate A* heuristic layer. The relevance score *is* the heuristic, unified with the constraint system.

---

## Concrete Example: RAG Document Selection

LLM softmax says:
- Chunk A (Jira ticket, perfect match): 10% probability
- Chunk B (old Confluence doc): 10% probability
- Chunk C (4-year-old Slack message): 80% probability (!!)

Screener metadata:
- Chunk A: `relevance = 1.0` (recent, authoritative)
- Chunk B: `relevance = 0.5` (slightly outdated)
- Chunk C: `relevance = 0.0` (completely irrelevant)

Blended scores:
- Chunk A: `ln(0.1) + ln(1.0) = -2.30 + 0.0 = -2.30`
- Chunk B: `ln(0.1) + ln(0.5) = -2.30 - 0.69 = -2.99`
- Chunk C: `ln(0.8) + ln(0.0) = -∞` → **TRIMMED**

Result: The 80%-probability garbage document is mathematically eliminated.

---

## Risks & Caveats

1. **Score calibration**: Relevance scores must be calibrated. If screener always returns 0.9+, the penalty is negligible. If it returns 0.01 for "decent" items, it over-prunes.
2. **WASM ABI break**: Adding `relevance` export to SDK requires version bump. Host must handle missing export gracefully.
3. **Performance**: `relevance()` is called ~100 times per decoding step (same as `is_valid`). If it's more expensive than a boolean check, latency increases.
4. **Over-constraining**: If screener is too aggressive, the tree becomes sparse and speculative decoding loses its advantage (low acceptance rate).

---

## Verdict: Adopt

The Screening Pruner is a strict superset of the current binary ConstraintPruner. It:
- **Preserves all existing behavior** via backward-compat blanket impl
- **Enables new capabilities** (graded relevance, metadata-aware RAG, heuristic search)
- **Has clean math** (log-space addition, no special cases)
- **Cross-cuts all three repos** (microgpt-rs engine, anyrag RAG, riir-validator-sdk WASM)

The WASM fixed-point encoding and backward compatibility layer make this adoptable incrementally without breaking existing validators.

---

## References

- "Screening Is Enough" (arXiv:2604.01178)
- microgpt-rs DDTree: `src/speculative/dd_tree.rs`, `src/speculative/types.rs`
- riir-validator-sdk: `src/validator.rs`, `src/exports.rs`
- Raven RSM: `.research/06_Raven_Routing_Slot_Memories.md`
