// Cross-entropy loss per-sample: softmax + per-position loss.
// Dispatch one invocation per sample position (batch_seq total).

struct LossParams {
    batch_seq: u32,
    vocab_size: u32,
    total_tokens: u32,
}

@group(0) @binding(0) var<storage, read>       logits: array<f32>;           // [batch * seq * vocab]
@group(0) @binding(1) var<storage, read>       targets: array<u32>;          // [batch * seq]
@group(0) @binding(2) var<storage, read_write> per_sample_loss: array<f32>;  // [batch_seq] output
@group(0) @binding(3) var<storage, read_write> log_probs: array<f32>;        // [batch * seq * vocab] for backward
@group(0) @binding(4) var<uniform>             params: LossParams;

@compute @workgroup_size(64, 1, 1)
fn cross_entropy_per_sample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.batch_seq) { return; }

    let offset = i * params.vocab_size;
    let tgt = targets[i];

    // Find max for numerical stability
    var max_logit: f32 = logits[offset];
    for (var v = 1u; v < params.vocab_size; v = v + 1u) {
        let val = logits[offset + v];
        if (val > max_logit) { max_logit = val; }
    }

    // Compute sum of exp(logit - max) + normalize
    var sum_exp: f32 = 0.0;
    for (var v = 0u; v < params.vocab_size; v = v + 1u) {
        let exp_val = exp(logits[offset + v] - max_logit);
        log_probs[offset + v] = exp_val;
        sum_exp = sum_exp + exp_val;
    }

    // Normalize + compute loss
    var target_prob: f32 = 0.0;
    for (var v = 0u; v < params.vocab_size; v = v + 1u) {
        log_probs[offset + v] = log_probs[offset + v] / sum_exp;
        if (v == tgt) {
            target_prob = log_probs[offset + v];
        }
    }

    per_sample_loss[i] = -log(target_prob + 1e-10);
}
