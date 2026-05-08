// Cross-entropy loss reduce: tree reduction to compute mean loss.
// Dispatch one workgroup (256 threads) for the reduction.

struct LossParams {
    batch_seq: u32,
    vocab_size: u32,
    total_tokens: u32,
}

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
