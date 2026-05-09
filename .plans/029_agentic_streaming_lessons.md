# Plan 029: Agentic Streaming Lessons from NVIDIA Dynamo

> Source: [Streaming Tokens and Tools: Multi-Turn Agentic Harness Support in NVIDIA Dynamo](https://developer.nvidia.com/blog/streaming-tokens-and-tools-multi-turn-agentic-harness-support-in-nvidia-dynamo/)
> Research: `.research/13_nvidia_dynamo_agentic_lessons.md`

## Tasks

- [ ] 1. Benchmark: measure TTFT with stable vs unstable prefix on speculative pipeline
- [ ] 2. Generalize `SolveEvent` → `DraftEvent` enum for streaming speculative steps
- [ ] 3. Add per-request agent hints to REST module (`latency_sensitivity`, `speculative_prefill`)
- [ ] 4. Add `/v1/tokenize` endpoint to REST module
- [ ] 5. Add domain-level truncation policy to `domains.toml`
- [ ] 6. Document DDTree branch ordering preserves reasoning sequence
- [ ] 7. Ensure `ScreeningPruner` + `ConstraintPruner` don't compete for same decision
- [ ] 8. Update README with production lessons from Dynamo

---

## Context

NVIDIA Dynamo hardened agentic inference for Claude Code, Codex, and OpenClaw. Key findings map directly to our speculative decoding stack:

### 1. Prompt Stability → 5× TTFT Penalty

Varying prefix at position zero poisons KV cache reuse. Dynamo measured 912ms vs 168ms TTFT on a 52K-token prompt (B200) just from a per-session billing header. Our `PagedKVCache` has prefix reuse but we haven't measured the impact.

**Action:** Benchmark prefix stability. If our REST module prepends per-request metadata, that's the same class of bug.

### 2. Streaming Tool Dispatch

Dynamo added `event: tool_call_dispatch` — fires when tool call is structurally complete, not when stream ends. Our `SolveEvent` already does this for Sudoku. Generalize to speculative decoding steps.

**Pattern:**
```
Buffered:  tool-call withheld until stream end → harness waits
Dispatch:  typed event at structural completion → harness acts immediately
```

### 3. Interleaved Reasoning Must Be Preserved

Dynamo found grouped ordering (reasoning × N then tool_calls × N) lost sequence meaning. Correct is interleaved (reasoning_0, tool_call_0, reasoning_1, tool_call_1). Our `extract_parent_tokens()` already preserves path ordering in DDTree branches — document this.

### 4. Single Parser Ownership

Competing parser layers caused silent malformation. Our `SpeculativeVerifier` trait already follows single-owner pattern. Verify `ScreeningPruner` and `ConstraintPruner` don't overlap in decision-making.

### 5. Catalog Metadata Shapes Agent Behavior

Dynamo showed wrong catalog = 50% fewer tool calls (41.7 vs 21.0 per task). Truncation policy (`tokens` vs `bytes`) changed what the model could inspect after failures. Our `domains.toml` needs truncation policy per domain.

### 6. Agent Hints (Per-Turn Intent)

Dynamo added `nvext.agent_hints: latency_sensitivity, priority, osl, speculative_prefill`. A user-waiting session ≠ background tool chain. Our REST module should accept per-request hints that control speculative behavior.

---

## Implementation Notes

### Task 1: Prefix Stability Benchmark

Add benchmark variant that:
- Runs speculative pipeline with stable prefix (same system tokens each step)
- Runs with varying prefix (prepend random/request-specific tokens at position 0)
- Compare: TTFT, acceptance rate, throughput

Expected: varying prefix should show measurable regression if PagedKVCache relies on prefix match.

### Task 2: `DraftEvent` Enum

```rust
pub enum DraftEvent {
    Drafting { pos: usize, candidates: usize },
    Pruned { pos: usize, kept: usize, rejected: usize },
    Verified { pos: usize, accepted: usize, bonus: bool },
    BranchRejected { pos: usize, reason: RejectionReason },
    StepComplete { tokens_accepted: usize, latency_us: u64 },
}
```

Emit during `speculative_step()`, consumed by REST streaming endpoint and TUI.

### Task 3: Agent Hints

```rust
pub struct AgentHints {
    pub latency_sensitivity: f32,   // 0.0=background, 1.0=interactive
    pub speculative_prefill: bool,  // enable prompt compression
    pub priority: u8,               // scheduling priority
}
```

Passed via REST request header, forwarded to `SpeculativeContext`.

### Task 4: `/v1/tokenize`

Simple endpoint wrapping existing BPE tokenizer. Returns token count for context accounting — harnesses use this to decide when to compact conversation.

### Task 5: Domain Truncation Policy

```toml
[[domain]]
name = "py2rs"
keywords = ["python", "rewrite", "translate"]
pruner = "syn_validator.wasm"
reader_lora = "python_reader.bin"
writer_lora = "rust_writer.bin"
truncation = { mode = "tokens", limit = 10000 }  # NEW
```

### Task 7: Pruner Overlap Audit

Verify `ScreeningPruner::relevance()` and `ConstraintPruner::is_valid()` make independent decisions:
- `ConstraintPruner` = hard structural validity (brackets, keywords)
- `ScreeningPruner` = graded semantic relevance (domain match)
- `BinaryScreeningPruner` adapter = bridge, should not add logic

If both prune the same token for different reasons, that's fine. If both claim ownership of the same decision type, that's a bug.

---

## Files to Modify

| File | Change |
|------|--------|
| `src/speculative/types.rs` | Add `DraftEvent` enum |
| `src/speculative/step.rs` | Emit `DraftEvent` during speculative step |
| `src/rest/` | Add agent hints parsing, `/v1/tokenize` endpoint |
| `src/types.rs` | Add `AgentHints` struct, domain truncation config |
| `src/router/` | Load truncation policy from domain config |
| `README.md` | Add "Production Lessons" section, Dynamo reference |

---

## Dependencies

- Task 2 depends on Task 1 results (if prefix stability is a problem, events help diagnose)
- Task 3, 4, 5 are independent
- Task 8 depends on all others complete

## Estimated Effort

| Task | Size | Notes |
|------|------|-------|
| 1. Prefix benchmark | S | Add 2 benchmark variants |
| 2. DraftEvent enum | M | New enum + emit points in step.rs |
| 3. Agent hints | S | Parse header, pass through |
| 4. /v1/tokenize | S | Wrap existing BPE |
| 5. Domain truncation | S | Config field + loader |
| 6. Documentation | S | README section |
| 7. Pruner audit | S | Code review, no new code |
| 8. README update | S | Final documentation |