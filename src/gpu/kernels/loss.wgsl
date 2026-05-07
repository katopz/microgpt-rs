// Cross-entropy loss with softmax.
// Dispatch 1: per-sample softmax + loss (one invocation per sample position).
// Dispatch 2: tree reduction to compute mean loss.

// Dispatch 1
@group(0) @binding(0) var<storage, read>       logits: array<f32>;      // [batch * seq * vocab]
@group(0) @binding(1) var<storage, read>       targets: array<u32>;     // [batch * seq]
@group(0) @binding(2) var<storage, read_write> per_sample_loss: array<f32>;  // [batch_seq] output
@group(0) @binding(3) var<storage, read_write> log_probs: array<f32>;   // [batch * seq * vocab] for backward
@group(0) @binding(4) var<uniform>             params: LossParams;

struct LossParams {
    batch_seq: u32,
    vocab_size: u32,
    total_tokens: u32,
}

@compute @workgroup_size(64, 1, 1)
fn cross_entropy_per_sample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.batch_seq) { return; }

    let offset = i * params.vocab_size;
    let target = targets[i];

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
        if (v == target) {
            target_prob = log_probs[offset + v];
        }
    }

    per_sample_loss[i] = -log(target_prob + 1e-10);
}

// Dispatch 2: Tree reduction for mean loss
@group(0) @binding(0) var<storage, read>       reduce_per_sample_loss: array<f32>;
@group(0) @binding(1) var<storage, read_write> loss: array<f32>;        // [1] output
@group(0) @binding(2) var<uniform>             reduce_params: LossParams;

var<workgroup> shared_loss: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn cross_entropy_reduce(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = gid.x;
    var val: f32 = 0.0;
    if (i < reduce_params.batch_seq) {
        val = reduce_per_sample_loss[i];
    }
    shared_loss[lid.x] = val;
    workgroupBarrier();

    // Tree reduction
    var stride = 128u;
    while (stride > 0u) {
        if (lid.x < stride) {
            shared_loss[lid.x] = shared_loss[lid.x] + shared_loss[lid.x + stride];
        }
        stride = stride >> 1u;
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        loss[0] = shared_loss[0] / f32(reduce_params.total_tokens);
    }
}
