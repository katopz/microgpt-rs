https://arxiv.org/pdf/2211.17192

I'll create a completely self-contained, heavily annotated Rust module that implements the exact algorithm described in **"Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2022)**. 

This implementation is designed to plug directly into your `microgpt-rs` ecosystem, using the existing `TransformerWeights`, `KVCache`, and `Config` structures. I've mapped the code directly to **Algorithm 1** from the paper, explaining the mathematics (like $p(x)$ vs $q(x)$ and the residual distribution) inline.


http://googleusercontent.com/immersive_entry_chip/0

### Key Educational Notes on the Implementation:

1. **The Batched Target Pass Guarantee:** I placed a comment in Phase 2 explaining a critical implementation detail. In production systems (like `vLLM` or `TensorRT`), the target model does *not* loop autoregressively to verify. It takes the sequence `[last_token, draft_1, draft_2, draft_3]` and runs a single `forward` pass, yielding all 4 logits simultaneously. Because `microgpt-rs` uses a single-token `forward` signature, this code simulates it sequentially, but the *mathematics* and *cache states* match the paper perfectly.
2. **The Residual Distribution**: The `sample_residual_distribution` function is the actual "magic" of the paper. Without it, speculative decoding changes the output distribution. By sampling the replacement token from $\max(0, p - q)$, Leviathan et al. proved that the final generated sequence is statistically identical to generating directly from the large model.
3. **The Bonus Token**: Notice that even if the target model accepts 100% of the drafted tokens, we still get $N+1$ tokens out of the step (Phase 4). The target model had to calculate the logits for the final drafted token anyway, meaning we get the *next* token for free!

```rust
//! Speculative Decoding implementation based on https://arxiv.org/pdf/2211.17192
//! "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2022)
//!
//! This module implements the vanilla speculative decoding algorithm, bridging
//! a smaller, faster "Draft" model (M_q) with a larger, slower "Target" model (M_p).
//! 
//! Key Concept: We can generate tokens faster than strictly autoregressive decoding 
//! by guessing several tokens at once and validating them in a single batched pass 
//! of the large model, using a specific statistical rejection sampling method to 
//! guarantee identical output distributions to the target model.

use crate::transformer::{forward, ForwardContext, KVCache, TransformerWeights};
use crate::types::{Config, Rng, softmax};

/// Performs a single step of Speculative Decoding (Algorithm 1 from the paper).
///
/// Returns a vector of accepted tokens. This will always contain at least 1 token,
/// and at most `gamma + 1` tokens.
///
/// # Arguments
/// * `gamma` - The number of tokens to draft per step (denoted as $\gamma$ in paper).
/// * `pos` - The current sequence length (used for KV cache positional indices).
/// * `last_token` - The last token generated in the previous step (or prompt).
pub fn speculative_decode_step(
    target_weights: &TransformerWeights,
    target_ctx: &mut ForwardContext,
    target_cache: &mut KVCache,
    draft_weights: &TransformerWeights,
    draft_ctx: &mut ForwardContext,
    draft_cache: &mut KVCache,
    rng: &mut Rng,
    gamma: usize,
    pos: usize,
    last_token: usize,
) -> Vec<usize> {
    
    // ─────────────────────────────────────────────────────────
    // Phase 1: DRAFTING (Run M_q autoregressively)
    // "Sample \tilde{x}_1, ..., \tilde{x}_\gamma ~ M_q(x | x_{<i})"
    // ─────────────────────────────────────────────────────────
    
    let mut q_distributions = Vec::with_capacity(gamma);
    let mut draft_tokens = Vec::with_capacity(gamma);
    let mut current_draft_token = last_token;
    
    for i in 0..gamma {
        // Run the small draft model
        let logits = forward(
            draft_ctx, draft_weights, draft_cache, 
            pos + i, current_draft_token
        );
        
        softmax(logits);
        let q_dist = logits.to_vec(); // $q(x)$
        
        // Sample the next draft token from $q(x)$
        let next_token = sample_from_distribution(&q_dist, rng);
        
        draft_tokens.push(next_token);
        q_distributions.push(q_dist);
        current_draft_token = next_token;
    }

    // ─────────────────────────────────────────────────────────
    // Phase 2: TARGET SCORING (Run M_p in parallel)
    // "Compute M_p(x | x_{<i}) for all i"
    // ─────────────────────────────────────────────────────────
    // Note: In a production engine (like vLLM/FlashAttention), this is 
    // executed as a *single batched forward pass* over the entire drafted sequence. 
    // For educational mapping to `microgpt-rs`, we simulate the parallel pass 
    // by evaluating them sequentially without sampling.
    
    let mut p_distributions = Vec::with_capacity(gamma + 1);
    let mut current_target_token = last_token;
    
    // We run gamma + 1 times to get the probability distribution for the 
    // extra "bonus" token if all drafted tokens are accepted.
    for i in 0..=gamma { 
        let logits = forward(
            target_ctx, target_weights, target_cache, 
            pos + i, current_target_token
        );
        
        softmax(logits);
        p_distributions.push(logits.to_vec()); // $p(x)$
        
        if i < gamma {
            // Feed the drafted tokens into the target model to score them
            current_target_token = draft_tokens[i];
        }
    }

    // ─────────────────────────────────────────────────────────
    // Phase 3: MODIFIED REJECTION SAMPLING
    // Evaluate drafted tokens left-to-right.
    // ─────────────────────────────────────────────────────────
    
    let mut accepted_tokens = Vec::new();
    let mut all_accepted = true;
    
    for i in 0..gamma {
        let p_dist = &p_distributions[i];
        let q_dist = &q_distributions[i];
        let drafted_token = draft_tokens[i];
        
        let p_i = p_dist[drafted_token];
        let q_i = q_dist[drafted_token];
        
        // Equation 1 & 2: Acceptance Criterion
        // If the target model likes this token more (or equally) than the draft model, 
        // accept it! If it likes it less, accept it with probability p / q.
        let acceptance_prob = (p_i / q_i).min(1.0);
        let r = rng.uniform();
        
        if r <= acceptance_prob {
            // ✅ Accepted!
            accepted_tokens.push(drafted_token);
        } else {
            // ❌ Rejected!
            // The draft model "oversampled" this token compared to the target model.
            // We must correct the distribution by sampling a replacement token from
            // the Residual Distribution (Equation 3).
            
            let replacement_token = sample_residual_distribution(p_dist, q_dist, rng);
            accepted_tokens.push(replacement_token);
            all_accepted = false;
            
            // Critical: Because we rejected at step `i`, the remaining drafted tokens
            // are invalid. The KV caches for both models have advanced `gamma` steps, 
            // but the true sequence diverged here. 
            // (In `microgpt-rs`, the caller will adjust the `pos` counter backwards 
            // based on `accepted_tokens.len()` on the next loop, effectively 
            // overwriting the stale KV cache states).
            break;
        }
    }
    
    // ─────────────────────────────────────────────────────────
    // Phase 4: THE BONUS TOKEN
    // "if all x_i are accepted, sample an extra token from M_p"
    // ─────────────────────────────────────────────────────────
    if all_accepted {
        let final_p_dist = &p_distributions[gamma];
        let extra_token = sample_from_distribution(final_p_dist, rng);
        accepted_tokens.push(extra_token);
    }
    
    accepted_tokens
}

// ── Helper Math Functions ──────────────────────────────────────────────

/// Standard cumulative distribution function (CDF) sampling.
fn sample_from_distribution(probs: &[f32], rng: &mut Rng) -> usize {
    let r = rng.uniform();
    let mut cdf = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cdf += p;
        if r <= cdf {
            return i;
        }
    }
    probs.len() - 1 // Fallback to last token to avoid panics on rounding errors
}

/// Implements Equation 3 from the paper:
/// $p'(x) = \frac{\max(0, p(x) - q(x))}{\sum \max(0, p(x) - q(x))}$
///
/// If a draft token is rejected, we sample the replacement from this distribution.
/// It intuitively means: "Sample from the tokens that the target model liked 
/// *more* than the draft model did, weighted by how much more it liked them."
fn sample_residual_distribution(p: &[f32], q: &[f32], rng: &mut Rng) -> usize {
    let mut residual: Vec<f32> = p.iter().zip(q.iter()).map(|(&p_val, &q_val)| {
        (p_val - q_val).max(0.0) // Clip negatives to 0
    }).collect();

    // Normalize the residual distribution so it sums to 1.0
    let sum: f32 = residual.iter().sum();
    
    if sum > 0.0 {
        for val in &mut residual {
            *val /= sum;
        }
    } else {
        // Fallback: If distributions are numerically identical but rejected due to 
        // float inaccuracies, just sample from the target distribution.
        return sample_from_distribution(p, rng);
    }

    sample_from_distribution(&residual, rng)
}
```
