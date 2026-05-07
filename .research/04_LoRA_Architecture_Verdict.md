# Verdict: Advanced LoRA Architecture in Neuro-Symbolic Inference

**Date:** 2025-06
**Status:** Refined Strategy
**Context:** microgpt-rs (MIT) + anyrag + Compiler-in-the-Loop
**Terminology:** See `.research/05_Artifact_Definition.md` — "Computable LoRA" is deprecated.

---

## Terminology Correction

Before discussing architecture, the naming must be clean:

| Old Term | New Term | Why |
|----------|----------|-----|
| Computable LoRA / cLoRA | **Deterministic Validator** (`validator.wasm` / `rules.rs`) | It's executable logic, not weights. No matrix multiplication. No learning. |
| Traditional LoRA | **Neural Adapter** (`lora.bin` / `lora.safetensors`) | It IS LoRA — low-rank weight matrices that modify output distribution. |

The codebase currently has `ComputableLora` in `percepta.rs` which is actually a constraint filter (zeros out invalid logits). This should be renamed to reflect what it is — a deterministic rules engine. Reserve "LoRA" exclusively for actual Low-Rank Adaptation.

---

## Assessment: Component by Component

### 1. Draft-Target Alignment — Highest Priority ✅

**What it says:** Train draft model via knowledge distillation from target so acceptance rate goes from 1.18 → 12-15 tokens/step.

**Verdict: Correct and should be Phase 1.**

Current benchmark reality:
```
LeviathanVerifier: ~1.18 tokens accepted per step
```

Why: The draft model's weights are random/generic — it doesn't agree with the target model on which tokens are likely. This is the #1 bottleneck in the entire pipeline.

The fix:
1. Train `lora.bin` (Neural Adapter) on the target model — learns idiomatic Rust
2. Distill a smaller `draft_lora.bin` from the target's distribution — learns to mimic target's "vibes"
3. DFlash marginals now align with target expectations → acceptance rate jumps

This is well-established in speculative decoding literature (Leviathan et al., 2022; Liu et al., 2023). The math is sound. The implementation path is clear.

**Current gap:** No LoRA implementation exists in microgpt-rs yet. The forward pass uses `TransformerWeights` directly. Adding LoRA means:
- `lora_a: Vec<Vec<f32>>` and `lora_b: Vec<Vec<f32>>` per layer
- Modified forward: `output = base_forward(x) + lora_a * lora_b * x`
- Loading from `.safetensors` or custom binary format

**Execution order:**
1. Implement LoRA weight loading + modified forward pass
2. Train initial `lora.bin` on verified Python→Rust pairs (even 1000 pairs to start)
3. Distill draft adapter from target
4. Re-benchmark — expect 5-10× acceptance rate improvement

### 2. Multi-LoRA (Stacking) — Valid, Phase 2 ✅

**What it says:** Apply multiple LoRAs simultaneously: `Output = Base + LoRA_1 + LoRA_2 + ...`

**Verdict: Mathematically correct. LoRA is additive by design.**

```
W' = W + A₁B₁ + A₂B₂
```

This works because LoRA doesn't modify base weights — it adds low-rank deltas. Multiple LoRAs = multiple deltas, all additive.

**Use case is real:** A Python file that uses `requests` + `json` + `sqlite3` benefits from simultaneous `reqwest_lora.bin` + `serde_lora.bin` + `rusqlite_lora.bin`.

**Current gap:** Trivial to implement once single-LoRA works. Just accumulate deltas before applying. The hard part is training the individual adapters, not the math.

### 3. S-LoRA (Multi-Tenant Serving) — Correct but Premature ⚠️

**What it says:** One frozen 7B base model on GPU, 50 concurrent users each with their own tiny LoRA, batched forward pass with per-user LoRA applied at the last millisecond.

**Verdict: Architecturally correct (S-LoRA, Sheng et al., 2023). But requires GPU infra + real model.**

Current state:
- microgpt-rs runs on CPU
- Model: 27 vocab, 16 embd, 1 layer — a toy
- No GPU inference path (wgpu feature exists but is minimal)
- No safetensors loading for real model weights

S-LoRA is Phase 3-4 infrastructure. It's the right end state for a SaaS serving thousands of concurrent translations, but you need:
1. A real base model (7B+ parameters, not 16-embd micro-transformer)
2. GPU inference pipeline (CUDA/Metal/Rocm)
3. Batched KV cache management
4. LoRA weight paging (unified memory for thousands of adapters)

**Don't build this until single-LoRA on a single model is working in production.**

### 4. LoRA-MoE (Curator Experts) — The Real Insight 🔥

**What it says:** Curator `.bin` files = MoE experts. Router dynamically selects which LoRA to apply based on file context.

**Verdict: This is the novel contribution. This is the marketplace's technical moat.**

The connection between Curator deliverables and MoE experts is the strongest idea in the entire strategy:

```
requirements.txt: numpy, flask, pydantic
         ↓
Router: pull numpy→ndarray.bin, flask→axum.bin, pydantic→serde.bin
         ↓
DDTree: translate with domain-specific intelligence
```

**However, token-by-token routing is premature.** Start with per-file routing:

| Routing Level | What | Complexity | When |
|---------------|------|-----------|------|
| Per-file | One LoRA set per file, selected by imports/dependencies | Low | Phase 1-2 |
| Per-span | Switch LoRA at function boundaries | Medium | Phase 3 |
| Per-token | Router shifts weight per token during generation | High | Phase 4+ |

Per-token routing requires swapping adapter weights mid-forward-pass. Production MoE (Mixtral 8x7B) routes at the layer level with hardcoded experts. Per-token dynamic LoRA switching has massive overhead unless you batch very carefully.

**Recommendation:** Start with per-file routing (trivial once multi-LoRA stacking works). Graduate to finer granularity later.

---

## The Symbiotic Loop (Self-Improving Cycle)

This is the flywheel that makes everything work:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  1. Deterministic Validator forces "dumb" base LLM     │
│     to produce valid Rust by pruning invalid tokens     │
│                        ↓                                │
│  2. Valid outputs saved to anyrag (Turso) as episodes   │
│                        ↓                                │
│  3. Episodes become training data for lora.bin          │
│                        ↓                                │
│  4. lora.bin makes LLM permanently smarter              │
│                        ↓                                │
│  5. Smarter LLM needs less pruning → validator relaxes  │
│                        ↓                                │
│  └─────────────── back to 1 ────────────────────────────│
│                                                         │
│  Key insight: Validator AUTO-GENERATES the training     │
│  data needed to make itself obsolete per-domain.        │
└─────────────────────────────────────────────────────────┘
```

This loop is honest — it doesn't claim the validator IS LoRA. The validator is the bootstrap mechanism that produces the data to train the real LoRA.

---

## The Two Artifacts in Practice

### Artifact 1: Deterministic Validator (OSS)

What exists now:
- `SynPruner` — two-tier Rust syntax validation (bracket DFA + syn parse)
- `SudokuPruner` — path-aware constraint satisfaction
- `PartialParser` — O(n) bracket balancing
- `CompilerFeedback` — extracts suggestions from parse errors
- `ConstraintPruner` trait — the extension point

What needs building:
- `CargoCheckPruner` — runs `cargo check` in sandbox, feeds errors back as constraints
- Per-domain validators from Curators (`domain_validator.wasm`)

### Artifact 2: Neural Adapter (Closed/SaaS)

What exists now:
- Nothing. No LoRA implementation in the codebase.

What needs building:
- LoRA weight matrices (A, B) per transformer layer
- Modified forward pass with LoRA injection
- `.safetensors` or custom binary loader
- Training pipeline (fine-tune on verified Python→Rust pairs)
- Knowledge distillation for draft model
- Curator `domain_lora.bin` upload + management

---

## Execution Priority (Honest Order)

| Priority | What | Why | Depends On |
|----------|------|-----|-----------|
| **P0** | Rename `ComputableLora` → `ConstraintFilter` or similar | Terminology clarity before anything else | Nothing |
| **P1** | Implement LoRA forward pass (single adapter) | Without this, nothing else matters | Weight matrix structs, forward pass modification |
| **P2** | Train initial `lora.bin` on ~1000 Python→Rust pairs | Bootstrap the Neural Adapter | P1, training data (anyrag episodes or manual curation) |
| **P3** | Draft-Target distillation | Biggest perf win: 1.18 → 12+ tokens/step | P2 (need trained target LoRA first) |
| **P4** | Multi-LoRA stacking | Enable domain composition | P1 (trivial once single-LoRA works) |
| **P5** | Per-file LoRA routing (MoE-lite) | Curator marketplace technical foundation | P4 |
| **P6** | GPU inference + real model | Scale from toy to production | External: need GPU infra + base model |
| **P7** | S-LoRA multi-tenant | SaaS scale | P6 |
| **P8** | Per-token LoRA routing | Maximum intelligence density | P5 + research |

---

## Key Risks

| Risk | Mitigation |
|------|-----------|
| LoRA training requires more data than expected | Validator auto-generates training data. Cold-start with manually curated pairs. |
| Draft distillation doesn't improve acceptance rate enough | Measure before/after. Even 3× improvement (1.18 → 3.5) is significant. |
| Per-token routing overhead kills throughput | Don't do it yet. Per-file routing is 90% of the value at 10% of the cost. |
| Base model obsolescence (new GPT every 6 months) | LoRA is model-agnostic. Swap base model, retrain adapters. The validator is model-free. |
| Naming confusion persists | P0: rename `ComputableLora` immediately. Use "Deterministic Validator" and "Neural Adapter" everywhere. |

---

## Final Verdict

The LoRA architecture strategy is correct at every level — draft-target alignment, multi-LoRA stacking, S-LoRA serving, and Curator-as-MoE-expert. The MoE/Curator connection is the strongest strategic insight.

**The critical path is:**
1. Clean up terminology (P0)
2. Implement actual LoRA (P1)
3. Bootstrap training data via validator-forced outputs (P2)
4. Distill draft model (P3)

Everything else (S-LoRA, MoE routing, per-token switching) follows naturally once the foundation exists.

**The validator doesn't need to be smart. It needs to be strict.** Strict validators produce clean training data. Clean training data produces smart LoRAs. Smart LoRAs make the validator's job easier. Flywheel.