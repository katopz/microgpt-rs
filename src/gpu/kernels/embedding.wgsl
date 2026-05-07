// Embedding lookup: output = wte[token] + wpe[position]
// One invocation per embedding dimension element.

@group(0) @binding(0) var<storage, read>       wte: array<f32>;      // [vocab_size, n_embd]
@group(0) @binding(1) var<storage, read>       wpe: array<f32>;      // [block_size, n_embd]
@group(0) @binding(2) var<storage, read>       tokens: array<u32>;   // [batch * seq]
@group(0) @binding(3) var<storage, read_write> output: array<f32>;   // [batch * seq * n_embd]
@group(0) @binding(4) var<uniform>             params: EmbeddingParams;

struct EmbeddingParams {
    batch_seq: u32,  // batch_size * seq_len
    n_embd: u32,
    vocab_size: u32,
    block_size: u32,
}

@compute @workgroup_size(256, 1, 1)
fn embedding_lookup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_idx = gid.x;
    if (flat_idx >= params.batch_seq * params.n_embd) { return; }

    let token_idx = flat_idx / params.n_embd;
    let dim = flat_idx % params.n_embd;

    let token = tokens[token_idx];
    let pos = token_idx % params.block_size;

    let wte_val = wte[token * params.n_embd + dim];
    let wpe_val = wpe[pos * params.n_embd + dim];

    output[flat_idx] = wte_val + wpe_val;
}
