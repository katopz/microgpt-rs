# Plan 031: Slot Machine Bandit Demo — Rules-Based Speculative Decoding

## Goal

Replace the "real transformer model" requirement in `bandit_demo.rs`'s disclaimer with a
rules-based slot machine that closes the full speculative decoding loop:

```
Episode N:   Reel weights → build_dd_tree_screened() → extract path → payline verify → reward
Episode N+1: Bandit learned which symbols pay → higher relevance for paying combos → better tree
```

No disclaimer needed — this proves **actual value**, not just mechanical compatibility.

## Context

`bandit_demo.rs` (lines 22-48) states the demo is "proof of mechanical compatibility, not proof of value"
because it uses `BernoulliEnv::pull()` coin flips. `dungeon_multifloor.rs` proved the pattern works
with deterministic constraints. A slot machine bridges the gap: **structured stochastic marginals**
(like a real draft model) + **rules-based verification** (like a real target model) = meaningful
bandit learning.

### Analogy Table

| Speculative Decoding | Slot Machine |
|---------------------|--------------|
| Draft model marginals P(token\|context) | Reel weights P(symbol\|reel) |
| Target model verification | Payline rules (combo valid?) |
| Accept → reward 1.0, Reject → 0.0 | Payout table (graded rewards) |
| BanditPruner screens branches | Bandit learns which symbols pay |

## Architecture

### Symbol Enum (6 symbols = vocab_size)
- Cherry, Lemon, Orange, Bell, Diamond, Seven

### SlotReels (provides marginals)
- 3 reels (positions = lookahead depth = 3)
- Each reel has weighted probabilities (non-uniform marginals)
- Reel 0: Cherry(0.30) Lemon(0.25) Orange(0.20) Bell(0.15) Diamond(0.07) Seven(0.03)
- Reel 1: Cherry(0.25) Lemon(0.20) Orange(0.20) Bell(0.15) Diamond(0.10) Seven(0.10)
- Reel 2: Cherry(0.20) Lemon(0.20) Orange(0.20) Bell(0.15) Diamond(0.15) Seven(0.10)

### PaylineRules (verification + reward)
- 3×Seven = JACKPOT (reward 1.0)
- 3×Diamond = BIG_WIN (reward 0.8)
- 3×Bell = NICE (reward 0.6)
- 3× same = WIN (reward 0.5) — catches Cherry/Lemon/Orange triples
- 2× same (any position) = PAIR (reward 0.2)
- Nothing = MISS (reward 0.0)

### SlotScreeningPruner (implements ScreeningPruner)
- Provides payline-aware domain relevance
- If parent symbols suggest a paying combo → boost completion symbol relevance
- If combo is already broken (e.g., Cherry+Seven) → reduce relevance
- This is the "domain knowledge" the bandit adapts on top of

### Demo Flow
```
1. SlotReels.marginals() → [reel0_weights, reel1_weights, reel2_weights]
2. BanditPruner<SlotScreeningPruner>.prepare_episode(rng)
3. build_dd_tree_screened(marginals, config, &pruner, true) → tree
4. extract_best_path(tree) → [symbol0, symbol1, symbol2]
5. PaylineRules.evaluate(path) → (combo_name, reward)
6. pruner.update(symbol0, reward) for each symbol in path
7. pruner.decay_epsilon()
8. Repeat for N episodes → bandit converges toward paying combos
```

### Metrics Printed
- Cumulative reward per strategy (UCB1, ε-greedy, Thompson)
- Average reward convergence plot (ASCII)
- Best combo found per strategy
- Bandit Q-values heatmap (which symbols learned high value)
- Comparison: bandit-assisted vs random reel spinning

## Tasks

- [x] 1. Create `examples/slot_bandit_demo.rs` with Symbol enum, Display impl
- [x] 2. Add SlotReels struct with per-reel weights and marginals() method
- [x] 3. Add PaylineRules with evaluate() returning (ComboName, f32 reward)
- [x] 4. Add SlotScreeningPruner implementing ScreeningPruner with payline awareness
- [x] 5. Add episode runner: DDTree build → path extract → verify → reward → update
- [x] 6. Add ASCII visualization: convergence plot, Q-value heatmap, combo table
- [x] 7. Add main() comparing UCB1, ε-greedy, Thompson Sampling + random baseline
- [x] 8. Add `[[example]] slot_bandit_demo` to Cargo.toml gated by `bandit` feature
- [x] 9. Test: `cargo run --example bandit_03_slot --features bandit`
- [x] 10. Update README.md with slot machine demo section

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `examples/slot_bandit_demo.rs` | Create | Slot machine demo: Symbol, SlotReels, PaylineRules, SlotScreeningPruner, main |
| `Cargo.toml` | Edit | Add `[[example]] slot_bandit_demo` gated by `bandit` |
| `README.md` | Edit | Add slot bandit demo section |

## Design Decisions

1. **Self-contained example** — all slot logic in one file, no new library modules needed.
   The point is demonstrating DDTree + BanditPruner with rules-based verification,
   not adding a slot machine library to microgpt-rs.

2. **Graded rewards (not binary)** — unlike speculative decoding's accept/reject (1.0/0.0),
   the slot uses graded rewards (0.0, 0.2, 0.5, 0.6, 0.8, 1.0). This better exercises
   the bandit's ability to distinguish good from great, not just good from bad.

3. **SlotScreeningPruner provides domain knowledge** — like a draft model knows syntax,
   the pruner knows "Cherry+Cherry needs another Cherry for triple". The bandit layer
   learns on top of this which high-value combos are reachable.

4. **3 reels = small tree** — vocab_size=6, lookahead=3 → tiny tree, fast episodes.
   Good for demo clarity. The bandit converges in ~200 episodes.

5. **Random baseline** — spin reels without DDTree/bandit, compare cumulative reward.
   Proves the bandit+tree pipeline adds value beyond random chance.

## Success Criteria

- [x] Bandit-assisted strategies outperform random baseline (higher cumulative reward)
- [x] Q-values converge: Seven/Diamond > Bell > Cherry/Lemon/Orange
- [x] Accept rate improves across episodes (better combos found over time)
- [x] No disclaimer needed — full loop: marginals → tree → verify → reward → learn

## Out of Scope

- Multi-line slots (multiple paylines per spin)
- Wild symbols / scatter symbols
- Bonus rounds / free spins
- Graphical UI (ASCII only)