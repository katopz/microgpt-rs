use microgpt_rs::speculative;
use microgpt_rs::transformer;
use microgpt_rs::types;

// ── types ──────────────────────────────────────────────────────

#[test]
fn test_config_micro_defaults() {
    let config = types::Config::micro();
    assert_eq!(config.vocab_size, 27);
    assert_eq!(config.block_size, 16);
    assert_eq!(config.n_embd, 16);
    assert_eq!(config.n_head, 4);
    assert_eq!(config.head_dim, 4);
    assert_eq!(config.mlp_hidden, 64);
    assert_eq!(config.bos_token, 26);
    assert!((config.temperature - 0.5).abs() < 1e-6);
    assert_eq!(config.draft_lookahead, 8);
    assert_eq!(config.tree_budget, 16);
}

#[test]
fn test_config_draft_defaults() {
    let config = types::Config::draft();
    assert_eq!(config.vocab_size, 27);
    assert_eq!(config.block_size, 16);
    assert_eq!(config.n_embd, 4);
    assert_eq!(config.n_head, 2);
    assert_eq!(config.head_dim, 2);
    assert_eq!(config.mlp_hidden, 16);
    assert_eq!(config.bos_token, 26);
    assert_eq!(config.draft_lookahead, 8);
    assert_eq!(config.tree_budget, 16);
}

#[test]
fn test_config_default_is_micro() {
    let default = types::Config::default();
    let micro = types::Config::micro();
    assert_eq!(default.vocab_size, micro.vocab_size);
    assert_eq!(default.block_size, micro.block_size);
    assert_eq!(default.n_embd, micro.n_embd);
}

#[test]
fn test_rng_deterministic() {
    let mut a = types::Rng::new(42);
    let mut b = types::Rng::new(42);
    for _ in 0..200 {
        assert_eq!(a.next(), b.next());
    }
}

#[test]
fn test_rng_different_seeds_diverge() {
    let mut a = types::Rng::new(1);
    let mut b = types::Rng::new(2);
    let mut same = 0;
    for _ in 0..100 {
        if a.next() == b.next() {
            same += 1;
        }
    }
    assert!(
        same < 10,
        "different seeds should produce different sequences"
    );
}

#[test]
fn test_rng_zero_seed_remapped() {
    let mut rng = types::Rng::new(0);
    let val = rng.next();
    assert_ne!(val, 0, "rng with seed 0 should still produce output");
}

#[test]
fn test_rng_uniform_range() {
    let mut rng = types::Rng::new(42);
    for _ in 0..2000 {
        let v = rng.uniform();
        assert!(
            (0.0..1.0).contains(&v),
            "uniform should be in [0,1), got {v}"
        );
    }
}

#[test]
fn test_rng_normal_finite() {
    let mut rng = types::Rng::new(42);
    for _ in 0..500 {
        let v = rng.normal();
        assert!(
            v.is_finite(),
            "normal should produce finite values, got {v}"
        );
    }
}

#[test]
fn test_softmax_basic() {
    let mut x = vec![1.0_f32, 2.0, 3.0];
    types::softmax(&mut x);
    let sum: f32 = x.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax sum should be 1.0, got {sum}"
    );
    assert!(
        x.iter().all(|&v| v > 0.0),
        "all softmax values should be positive"
    );
    assert!(x[0] < x[1] && x[1] < x[2], "softmax should preserve order");
}

#[test]
fn test_softmax_empty() {
    let mut x: Vec<f32> = vec![];
    types::softmax(&mut x);
    assert!(x.is_empty());
}

#[test]
fn test_softmax_uniform() {
    let mut x = vec![5.0_f32; 10];
    types::softmax(&mut x);
    let expected = 1.0 / 10.0;
    for &v in &x {
        assert!(
            (v - expected).abs() < 1e-5,
            "uniform softmax should give equal values"
        );
    }
}

#[test]
fn test_softmax_large_values_no_overflow() {
    let mut x = vec![1000.0_f32, 1001.0, 1002.0];
    types::softmax(&mut x);
    let sum: f32 = x.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "should handle large values, sum={sum}"
    );
    assert!(x.iter().all(|v| v.is_finite()));
}

#[test]
fn test_rmsnorm_unit_vector() {
    let mut x = vec![1.0_f32; 16];
    types::rmsnorm(&mut x);
    let ms: f32 = x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32;
    assert!(
        (ms - 1.0).abs() < 1e-4,
        "rmsnorm should normalize to unit variance, ms={ms}"
    );
}

#[test]
fn test_rmsnorm_empty() {
    let mut x: Vec<f32> = vec![];
    types::rmsnorm(&mut x);
    assert!(x.is_empty());
}

#[test]
fn test_matmul_identity() {
    let config = types::Config::micro();
    let n = config.n_embd;
    let mut identity = vec![0.0; n * n];
    for i in 0..n {
        identity[i * n + i] = 1.0;
    }
    let input = vec![2.0; n];
    let mut output = vec![0.0; n];
    types::matmul(&mut output, &identity, &input, n, n);
    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v - 2.0).abs() < 1e-5,
            "identity matmul at {i}: expected 2.0, got {v}"
        );
    }
}

#[test]
fn test_matmul_zero_weight() {
    let config = types::Config::micro();
    let n = config.n_embd;
    let weight = vec![0.0; n * n];
    let input = vec![42.0; n];
    let mut output = vec![0.0; n];
    types::matmul(&mut output, &weight, &input, n, n);
    assert!(
        output.iter().all(|&v| v == 0.0),
        "zero weight should give zero output"
    );
}

#[test]
fn test_sample_token_valid() {
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let mut probs = vec![0.0; config.vocab_size];
    probs[5] = 1.0;
    for _ in 0..100 {
        let token = types::sample_token(&probs, &mut rng);
        assert_eq!(token, 5, "should always sample token 5");
    }
}

// ── transformer ────────────────────────────────────────────────

#[test]
fn test_forward_output_size() {
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    let mut ctx = transformer::ForwardContext::new(&config);
    let mut cache = transformer::MultiLayerKVCache::new(&config);
    let logits = transformer::forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_forward_logits_finite() {
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    let mut ctx = transformer::ForwardContext::new(&config);
    let mut cache = transformer::MultiLayerKVCache::new(&config);
    let logits = transformer::forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "logit {i} is not finite: {l}");
    }
}

#[test]
fn test_forward_cache_populated() {
    let config = types::Config::micro();
    let kvd = types::kv_dim(&config);
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    let mut ctx = transformer::ForwardContext::new(&config);
    let mut cache = transformer::MultiLayerKVCache::new(&config);
    transformer::forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
    let key_sum: f32 = cache.layers[0].key[..kvd].iter().sum();
    let val_sum: f32 = cache.layers[0].value[..kvd].iter().sum();
    assert!(key_sum != 0.0, "K cache at pos 0 should be populated");
    assert!(val_sum != 0.0, "V cache at pos 0 should be populated");
}

#[test]
fn test_forward_positions_differ() {
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    let mut ctx = transformer::ForwardContext::new(&config);
    let mut cache = transformer::MultiLayerKVCache::new(&config);
    let logits_0 = transformer::forward(&mut ctx, &weights, &mut cache, 0, 0, &config).to_vec();
    let logits_1 = transformer::forward(&mut ctx, &weights, &mut cache, 0, 1, &config);
    let different = logits_0.iter().zip(logits_1).any(|(&a, b)| a != *b);
    assert!(different, "logits at different positions should differ");
}

#[test]
fn test_forward_draft_model() {
    let draft_config = types::Config::draft();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&draft_config, &mut rng);
    let mut ctx = transformer::ForwardContext::new(&draft_config);
    let mut cache = transformer::MultiLayerKVCache::new(&draft_config);
    let logits = transformer::forward(&mut ctx, &weights, &mut cache, 0, 0, &draft_config);
    assert_eq!(logits.len(), draft_config.vocab_size);
    assert!(logits.iter().all(|&l| l.is_finite()));
}

#[test]
fn test_generate_deterministic() {
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);

    let mut rng1 = types::Rng::new(100);
    let t1 = transformer::generate(&weights, &config, &mut rng1, 16);

    let mut rng2 = types::Rng::new(100);
    let t2 = transformer::generate(&weights, &config, &mut rng2, 16);

    assert_eq!(t1, t2, "same seed must produce identical tokens");
}

#[test]
fn test_generate_valid_tokens() {
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    let tokens = transformer::generate(&weights, &config, &mut rng, 64);
    assert_eq!(tokens.len(), 64);
    for &t in &tokens {
        assert!(
            t < config.vocab_size,
            "token {t} out of range [0,{})",
            config.vocab_size
        );
    }
}

#[test]
fn test_generate_different_seeds_diverge() {
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);

    let mut rng1 = types::Rng::new(1);
    let t1 = transformer::generate(&weights, &config, &mut rng1, 16);

    let mut rng2 = types::Rng::new(999);
    let t2 = transformer::generate(&weights, &config, &mut rng2, 16);

    assert_ne!(t1, t2, "different seeds should produce different output");
}

#[test]
fn test_generate_exact_length() {
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    for len in [1, 8, 16, 32, 50] {
        let tokens = transformer::generate(&weights, &config, &mut rng, len);
        assert_eq!(
            tokens.len(),
            len,
            "generate should return exactly {len} tokens"
        );
    }
}

#[test]
fn test_tokens_to_string_roundtrip() {
    let tokens = vec![0, 1, 2, 25, 26];
    let s = transformer::tokens_to_string(&tokens);
    assert_eq!(s, "abcz_");
}

#[test]
fn test_tokens_to_string_empty() {
    let tokens: Vec<usize> = vec![];
    let s = transformer::tokens_to_string(&tokens);
    assert_eq!(s, "");
}

#[test]
fn test_kv_cache_reset() {
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    let mut ctx = transformer::ForwardContext::new(&config);
    let mut cache = transformer::MultiLayerKVCache::new(&config);
    transformer::forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
    cache.reset();
    for (i, layer) in cache.layers.iter().enumerate() {
        assert!(
            layer.key.iter().all(|&v| v == 0.0),
            "layer {i} cache key should be zeroed after reset"
        );
        assert!(
            layer.value.iter().all(|&v| v == 0.0),
            "layer {i} cache value should be zeroed after reset"
        );
    }
}

#[test]
fn test_forward_context_reuse() {
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    let mut ctx = transformer::ForwardContext::new(&config);
    let mut cache = transformer::MultiLayerKVCache::new(&config);

    let l1 = transformer::forward(&mut ctx, &weights, &mut cache, 0, 0, &config).to_vec();
    let l2 = transformer::forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
    // Second call reuses buffers — results may differ due to cache accumulation
    for &v in l2.iter() {
        assert!(v.is_finite(), "reused context produced non-finite: {v}");
    }
    drop(l1);
}

// ── speculative ────────────────────────────────────────────────

fn make_draft() -> (transformer::TransformerWeights, types::Config) {
    let config = types::Config::draft();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    (weights, config)
}

#[test]
fn test_dflash_produces_marginals() {
    let (weights, config) = make_draft();
    let marginals = speculative::dflash_predict(&weights, &config, 0, 0);
    assert!(
        !marginals.is_empty(),
        "should produce at least one marginal"
    );
    assert!(marginals.len() <= config.draft_lookahead);
    for (i, row) in marginals.iter().enumerate() {
        assert_eq!(row.len(), config.vocab_size, "row {i} wrong size");
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "row {i} sum = {sum}, expected 1.0"
        );
    }
}

#[test]
fn test_dflash_parallel_matches_count() {
    let (weights, config) = make_draft();
    let seq = speculative::dflash_predict(&weights, &config, 0, 0);
    let par = speculative::dflash_predict_parallel(&weights, &config, 0, 0);
    assert_eq!(seq.len(), par.len(), "parallel should produce same count");
}

#[test]
fn test_dflash_parallel_valid_probs() {
    let (weights, config) = make_draft();
    let marginals = speculative::dflash_predict_parallel(&weights, &config, 0, 0);
    for (i, row) in marginals.iter().enumerate() {
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "parallel row {i} sum = {sum}, expected 1.0"
        );
    }
}

#[test]
fn test_dflash_positions_differ() {
    let (weights, config) = make_draft();
    let m0 = speculative::dflash_predict(&weights, &config, 0, 0);
    let m1 = speculative::dflash_predict(&weights, &config, 0, 1);
    assert_ne!(
        m0[0], m1[0],
        "marginals at different positions should differ"
    );
}

#[test]
fn test_ddtree_respects_budget() {
    let (weights, config) = make_draft();
    let marginals = speculative::dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let tree = speculative::build_dd_tree(&mv, &config);
    assert!(
        tree.len() <= config.tree_budget,
        "tree size {} exceeds budget {}",
        tree.len(),
        config.tree_budget
    );
    assert!(!tree.is_empty(), "tree should have at least one node");
}

#[test]
fn test_ddtree_scores_descending() {
    let (weights, config) = make_draft();
    let marginals = speculative::dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let tree = speculative::build_dd_tree(&mv, &config);
    for window in tree.windows(2) {
        assert!(
            window[0].score >= window[1].score,
            "scores not descending: {} >= {}",
            window[0].score,
            window[1].score
        );
    }
}

#[test]
fn test_ddtree_depth_within_lookahead() {
    let (weights, config) = make_draft();
    let marginals = speculative::dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let tree = speculative::build_dd_tree(&mv, &config);
    for node in &tree {
        assert!(
            node.depth < config.draft_lookahead,
            "depth {} should be < lookahead {}",
            node.depth,
            config.draft_lookahead
        );
    }
}

#[test]
fn test_ddtree_valid_token_indices() {
    let (weights, config) = make_draft();
    let marginals = speculative::dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let tree = speculative::build_dd_tree(&mv, &config);
    for node in &tree {
        assert!(
            node.token_idx < config.vocab_size,
            "token_idx {} out of range",
            node.token_idx
        );
    }
}

#[test]
fn test_ddtree_empty_marginals() {
    let config = types::Config::draft();
    let tree = speculative::build_dd_tree(&[], &config);
    assert!(tree.is_empty(), "empty marginals should produce empty tree");
}

#[test]
fn test_speculative_step_accepts_at_least_one() {
    let (weights, config) = make_draft();
    for seed in [0, 42, 100, 999] {
        let mut step_rng = types::Rng::new(seed);
        let (accepted, accept_len) =
            speculative::speculative_step(&weights, &config, 0, 0, &mut step_rng);
        assert!(
            !accepted.is_empty(),
            "seed {seed}: should accept at least 1 token"
        );
        assert!(accept_len >= 1, "seed {seed}: accept_len should be >= 1");
        for &t in &accepted {
            assert!(t < config.vocab_size, "seed {seed}: token {t} out of range");
        }
    }
}

#[test]
fn test_speculative_step_consistent_for_same_seed() {
    let (weights, config) = make_draft();

    let mut rng1 = types::Rng::new(77);
    let (a1, l1) = speculative::speculative_step(&weights, &config, 0, 0, &mut rng1);

    let mut rng2 = types::Rng::new(77);
    let (a2, l2) = speculative::speculative_step(&weights, &config, 0, 0, &mut rng2);

    assert_eq!(a1, a2, "same seed should produce same accepted tokens");
    assert_eq!(l1, l2, "same seed should produce same acceptance length");
}

// ── Percepta 2D Convex Hull Attention Tests ────────────────────────

use microgpt_rs::percepta::{KVCache2D, Vec2};

#[test]
fn test_percepta_vec2_dot() {
    let a = Vec2::new(3.0, 4.0);
    let b = Vec2::new(1.0, 0.0);
    assert!((a.dot(&b) - 3.0).abs() < 1e-6, "dot product wrong");
}

#[test]
fn test_percepta_cross_z_signs() {
    let origin = Vec2::new(0.0, 0.0);
    let right = Vec2::new(1.0, 0.0);
    let up = Vec2::new(0.0, 1.0);
    // Left turn: origin → right → up
    assert!(Vec2::cross_z(&origin, &right, &up) > 0.0);
    // Right turn: origin → up → right
    assert!(Vec2::cross_z(&origin, &up, &right) < 0.0);
}

#[test]
fn test_percepta_empty_cache() {
    let cache = KVCache2D::new();
    assert!(cache.is_empty());
    let (s, v) = cache.fast_attention(&Vec2::new(1.0, 0.0));
    assert_eq!(s, f32::NEG_INFINITY);
    assert_eq!(v, 0);
}

#[test]
fn test_percepta_single_key() {
    let mut cache = KVCache2D::new();
    cache.append(Vec2::new(5.0, 10.0), 42);
    let (lin_s, lin_v) = cache.linear_attention(&Vec2::new(1.0, 1.0));
    let (fast_s, fast_v) = cache.fast_attention(&Vec2::new(1.0, 1.0));
    assert!((lin_s - fast_s).abs() < 1e-6);
    assert_eq!(lin_v, fast_v);
    assert_eq!(lin_v, 42);
}

#[test]
fn test_percepta_two_keys_picks_max() {
    let mut cache = KVCache2D::new();
    cache.append(Vec2::new(0.0, 100.0), 0);
    cache.append(Vec2::new(100.0, 0.0), 1);

    // Query with positive x, zero y → should pick key (100, 0)
    let q = Vec2::new(1.0, 0.0);
    let (_, fast_v) = cache.fast_attention(&q);
    assert_eq!(fast_v, 1, "should pick x-heavy key");

    // Query with zero x, positive y → should pick key (0, 100)
    let q = Vec2::new(0.0, 1.0);
    let (_, fast_v) = cache.fast_attention(&q);
    assert_eq!(fast_v, 0, "should pick y-heavy key");
}

#[test]
fn test_percepta_linear_fast_match_parabolic() {
    let mut cache = KVCache2D::new();
    for i in 0..1000u32 {
        let x = i as f32;
        let y = -((x - 500.0) / 50.0).powi(2);
        cache.append(Vec2::new(x, y), i as usize);
    }

    let queries = [
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(-1.0, 1.0),
        Vec2::new(5.0, 10.0),
        Vec2::new(-3.0, 7.0),
    ];

    for query in &queries {
        let (lin_s, lin_v) = cache.linear_attention(query);
        let (fast_s, fast_v) = cache.fast_attention(query);
        assert!(
            (lin_s - fast_s).abs() < 1e-3,
            "score mismatch for query ({}, {}): linear={lin_s}, fast={fast_s}",
            query.x,
            query.y
        );
        assert_eq!(
            lin_v, fast_v,
            "value mismatch for query ({}, {})",
            query.x, query.y
        );
    }
}

#[test]
fn test_percepta_linear_fast_match_100k() {
    let mut cache = KVCache2D::new();
    for i in 0..100_000u32 {
        let x = i as f32;
        let y = -((x - 50000.0) / 1000.0).powi(2);
        cache.append(Vec2::new(x, y), i as usize);
    }

    let query = Vec2::new(5.0, 10.0);
    let (lin_s, lin_v) = cache.linear_attention(&query);
    let (fast_s, fast_v) = cache.fast_attention(&query);

    assert!((lin_s - fast_s).abs() < 1e-3, "100k trace: score mismatch");
    assert_eq!(lin_v, fast_v, "100k trace: value mismatch");
}

#[test]
fn test_percepta_hull_compression_collinear() {
    let mut cache = KVCache2D::new();
    for i in 0..500 {
        cache.append(Vec2::new(i as f32, i as f32 * 2.0), i);
    }
    // Collinear points compress to 2 endpoints
    assert!(
        cache.hull_len() <= 2,
        "collinear hull should be <= 2, got {}",
        cache.hull_len()
    );
    assert_eq!(cache.len(), 500, "total keys preserved");
}

#[test]
fn test_percepta_hull_keeps_convex() {
    let mut cache = KVCache2D::new();
    // Concave-down: every point is a hull vertex
    for i in 0..50u32 {
        let x = i as f32;
        let y = -(x - 25.0).powi(2);
        cache.append(Vec2::new(x, y), i as usize);
    }
    assert_eq!(cache.hull_len(), 50, "concave-down should keep all points");
}

#[test]
fn test_percepta_reset() {
    let mut cache = KVCache2D::new();
    cache.append(Vec2::new(1.0, 2.0), 0);
    cache.append(Vec2::new(3.0, 4.0), 1);
    assert!(!cache.is_empty());
    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.hull_len(), 0);
}

#[test]
fn test_percepta_hull_is_subset_of_keys() {
    let mut cache = KVCache2D::new();
    for i in 0..200u32 {
        let x = i as f32;
        let y = (x * 0.1).sin() - (x * 0.05).cos();
        cache.append(Vec2::new(x, y), i as usize);
    }

    // All hull indices must be valid key indices
    for &idx in cache.hull_indices() {
        assert!(idx < cache.len(), "hull index {idx} out of range");
    }
    // Hull must be smaller than total keys (unless all convex)
    assert!(
        cache.hull_len() <= cache.len(),
        "hull can't be larger than keys"
    );
}

#[test]
fn test_percepta_multiple_queries_correctness() {
    let mut cache = KVCache2D::new();
    // Sinusoidal distribution — hull compresses to peaks only
    for i in 0..5000u32 {
        let x = i as f32;
        let y = (x * 0.01).sin();
        cache.append(Vec2::new(x, y), i as usize);
    }

    let hull_ratio = cache.hull_len() as f64 / cache.len() as f64;
    assert!(
        hull_ratio < 0.5,
        "sinusoidal hull should compress (< 50%), got {:.1}%",
        hull_ratio * 100.0
    );

    // Verify every hull point is correct
    for query in [
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(-1.0, 0.5),
    ] {
        let (lin_s, lin_v) = cache.linear_attention(&query);
        let (fast_s, fast_v) = cache.fast_attention(&query);
        assert!(
            (lin_s - fast_s).abs() < 1e-3,
            "sinusoidal score mismatch: lin={lin_s}, fast={fast_s}"
        );
        assert_eq!(lin_v, fast_v, "sinusoidal value mismatch");
    }
}

// ── Percepta: Adversarial, Computation & Geometry Tests ───────────

#[test]
fn test_percepta_adversarial_v_shape_fast_wrong() {
    let mut cache = KVCache2D::new();
    // V-shape: valley at index 2
    cache.append(Vec2::new(0.0, 10.0), 0);
    cache.append(Vec2::new(1.0, 5.0), 1);
    cache.append(Vec2::new(2.0, 0.0), 2); // valley bottom
    cache.append(Vec2::new(3.0, 5.0), 3);
    cache.append(Vec2::new(4.0, 10.0), 4);

    // Upper hull = only the two peaks (indices 0 and 4)
    assert_eq!(cache.hull_len(), 2, "V-shape hull should be 2 endpoints");

    // Query pointing DOWN: linear finds valley bottom, fast misses it
    let query = Vec2::new(0.0, -1.0);
    let (lin_score, lin_val) = cache.linear_attention(&query);
    let (fast_score, fast_val) = cache.fast_attention(&query);

    assert_eq!(lin_val, 2, "linear should find valley bottom");
    assert!((lin_score - 0.0).abs() < 1e-6, "linear score should be 0");
    assert_ne!(fast_val, lin_val, "fast should disagree on valley query");
    assert!(fast_score < lin_score, "fast score should be worse");
}

#[test]
fn test_percepta_adversarial_v_shape_positive_correct() {
    let mut cache = KVCache2D::new();
    cache.append(Vec2::new(0.0, 10.0), 0);
    cache.append(Vec2::new(1.0, 5.0), 1);
    cache.append(Vec2::new(2.0, 0.0), 2);
    cache.append(Vec2::new(3.0, 5.0), 3);
    cache.append(Vec2::new(4.0, 10.0), 4);

    // Query pointing UP: optimum IS on hull → fast matches linear
    let query = Vec2::new(0.0, 1.0);
    let (lin_score, _) = cache.linear_attention(&query);
    let (fast_score, fast_val) = cache.fast_attention(&query);

    assert!(
        (lin_score - fast_score).abs() < 1e-6,
        "scores should match for hull-optimal query"
    );
    assert!(
        fast_val == 0 || fast_val == 4,
        "fast should find a peak, got {fast_val}"
    );
}

#[test]
fn test_percepta_dfa_divisible_by_3() {
    // DFA: binary strings divisible by 3
    // States: 0 (accept), 1, 2
    // Transition: δ(state, bit) = (2*state + bit) % 3
    let input = [1, 1, 0, 1, 1, 0]; // 54 in binary, 54 % 3 == 0
    let mut state = 0usize;
    let mut cache = KVCache2D::new();

    for (step, &bit) in input.iter().enumerate() {
        let next_state = (state * 2 + bit) % 3;
        cache.append(
            Vec2::new(step as f32, state as f32 * 100.0 + bit as f32 * 10.0),
            next_state,
        );
        state = next_state;
    }

    assert_eq!(state, 0, "54 should be divisible by 3");
    assert_eq!(cache.len(), 6, "should have 6 trace entries");
}

#[test]
fn test_percepta_dfa_stress_all_integers() {
    for n in 0..1000u32 {
        let bits: Vec<u8> = (0..16)
            .rev()
            .map(|shift| ((n >> shift) & 1) as u8)
            .collect();

        let mut state = 0usize;
        let mut cache = KVCache2D::new();

        for (step, &bit) in bits.iter().enumerate() {
            let next_state = (state * 2 + bit as usize) % 3;
            cache.append(Vec2::new(step as f32, state as f32 * 100.0), next_state);
            state = next_state;
        }

        assert_eq!(
            state == 0,
            n % 3 == 0,
            "DFA wrong for n={n}: expected state={}, got state={state}",
            if n % 3 == 0 { 0 } else { 1 }
        );

        if !cache.is_empty() {
            let query = Vec2::new(0.0, 1.0);
            let (lin_s, _) = cache.linear_attention(&query);
            let (fast_s, _) = cache.fast_attention(&query);
            assert!(
                (lin_s - fast_s).abs() < 1e-3,
                "DFA trace attention mismatch for n={n}"
            );
        }
    }
}

#[test]
fn test_percepta_fibonacci_trace() {
    let mut cache = KVCache2D::with_capacity(50);
    let mut fib = vec![0u64, 1u64];
    cache.append(Vec2::new(0.0, 0.0), 0);
    cache.append(Vec2::new(1.0, 1.0), 1);

    for i in 2..45u32 {
        let next = fib[i as usize - 1] + fib[i as usize - 2];
        fib.push(next);
        cache.append(Vec2::new(i as f32, next as f32), i as usize);
    }

    // Exponential growth is concave-UP → hull compresses to 2
    assert!(
        cache.hull_len() <= 2,
        "exponential growth should compress hull, got {}",
        cache.hull_len()
    );

    // Endpoint queries still work correctly
    let query = Vec2::new(1.0, 0.0);
    let (lin_s, lin_v) = cache.linear_attention(&query);
    let (fast_s, fast_v) = cache.fast_attention(&query);
    assert!((lin_s - fast_s).abs() < 1e-3);
    assert_eq!(lin_v, fast_v);

    assert_eq!(fib[10], 55);
    assert_eq!(fib[20], 6765);
    assert_eq!(fib[44], 701408733);
}

#[test]
fn test_percepta_counter_collinear() {
    let mut cache = KVCache2D::new();
    for i in 0..10000 {
        cache.append(Vec2::new(i as f32, i as f32), i);
    }

    assert!(
        cache.hull_len() <= 2,
        "counter trace should compress to 2, got {}",
        cache.hull_len()
    );

    let query = Vec2::new(1.0, 1.0);
    let (lin_s, lin_v) = cache.linear_attention(&query);
    let (fast_s, fast_v) = cache.fast_attention(&query);

    assert!((lin_s - fast_s).abs() < 1e-3);
    assert_eq!(lin_v, fast_v);
    assert_eq!(lin_v, 9999, "should pick the last counter value");
}

#[test]
fn test_percepta_unimodality_proof() {
    let mut cache = KVCache2D::new();
    // Concave-down parabola: all points on hull
    for i in 0..100u32 {
        let x = i as f32;
        let y = -(x - 50.0).powi(2) / 100.0;
        cache.append(Vec2::new(x, y), i as usize);
    }
    assert_eq!(cache.hull_len(), 100);

    let queries = [
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(-1.0, 1.0),
        Vec2::new(2.0, -1.0),
    ];

    for query in &queries {
        let hull = cache.hull_indices();
        let scores: Vec<f32> = hull
            .iter()
            .map(|&idx| query.dot(&cache.keys()[idx]))
            .collect();

        let max_pos = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        for i in 1..=max_pos {
            assert!(
                scores[i] >= scores[i - 1] - 1e-6,
                "not unimodal before max at i={i}: {} < {}",
                scores[i],
                scores[i - 1]
            );
        }
        for i in max_pos + 1..scores.len() {
            assert!(
                scores[i] <= scores[i - 1] + 1e-6,
                "not unimodal after max at i={i}: {} > {}",
                scores[i],
                scores[i - 1]
            );
        }
    }
}

#[test]
fn test_percepta_supporting_point_360_degrees() {
    let mut cache = KVCache2D::new();
    // Concave-down parabola: all points on upper hull
    for i in 0..500u32 {
        let x = i as f32;
        let y = -(x - 250.0).powi(2) / 100.0 + 100.0;
        cache.append(Vec2::new(x, y), i as usize);
    }

    for deg in 0..360 {
        let rad = (deg as f32).to_radians();
        let query = Vec2::new(rad.cos(), rad.sin());

        let (lin_s, lin_v) = cache.linear_attention(&query);
        let (fast_s, fast_v) = cache.fast_attention(&query);

        assert!(
            (lin_s - fast_s).abs() < 1e-3,
            "supporting point violated at deg={deg}: lin={lin_s}, fast={fast_s}"
        );
        assert_eq!(lin_v, fast_v, "value mismatch at deg={deg}");
    }
}

#[test]
fn test_percepta_random_convex_stress() {
    let mut seed = 12345u64;
    let next_seed = |s: &mut u64| -> f32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 33) as f32) / (1u64 << 31) as f32
    };

    let mut cache = KVCache2D::with_capacity(10000);
    let center = next_seed(&mut seed) * 5000.0;
    let scale = 100.0 + next_seed(&mut seed) * 900.0;
    let offset = next_seed(&mut seed) * 50.0;

    for i in 0..10000u32 {
        let x = i as f32;
        let y = -(x - center).powi(2) / scale + offset;
        cache.append(Vec2::new(x, y), i as usize);
    }

    for _ in 0..100 {
        let qx = (next_seed(&mut seed) - 0.5) * 20.0;
        let qy = next_seed(&mut seed) * 20.0; // qy >= 0 for unimodal guarantee
        let query = Vec2::new(qx, qy);

        let (lin_s, lin_v) = cache.linear_attention(&query);
        let (fast_s, fast_v) = cache.fast_attention(&query);

        assert!(
            (lin_s - fast_s).abs() < 1e-2,
            "random stress: score mismatch for ({qx:.2}, {qy:.2}): lin={lin_s:.2}, fast={fast_s:.2}"
        );
        assert_eq!(
            lin_v, fast_v,
            "random stress: value mismatch for ({qx:.2}, {qy:.2})"
        );
    }
}

// ── Percepta: Arithmetic Computation via Attention ─────────────────

#[test]
fn test_percepta_arithmetic_addition() {
    let mut cache = KVCache2D::new();
    let query = Vec2::new(1.0, 0.0);

    cache.append(Vec2::new(0.0, 42.0), 42);

    for step in 1..=17 {
        let (lin_s, lin_v) = cache.linear_attention(&query);
        let (fast_s, fast_v) = cache.fast_attention(&query);
        assert!((lin_s - fast_s).abs() < 1e-3, "step {step}: score mismatch");
        assert_eq!(lin_v, fast_v, "step {step}: value mismatch");
        let next = fast_v + 1;
        cache.append(Vec2::new(step as f32, next as f32), next);
    }

    let (_, result) = cache.fast_attention(&query);
    assert_eq!(result, 59, "42 + 17 = 59");
}

#[test]
fn test_percepta_arithmetic_subtraction() {
    let mut cache = KVCache2D::new();
    let query = Vec2::new(1.0, 0.0);

    cache.append(Vec2::new(0.0, 100.0), 100);

    for step in 1..=37 {
        let (_, prev) = cache.fast_attention(&query);
        let next = prev - 1;
        cache.append(Vec2::new(step as f32, next as f32), next);
    }

    let (_, result) = cache.fast_attention(&query);
    assert_eq!(result, 63, "100 - 37 = 63");

    let (lin_s, lin_v) = cache.linear_attention(&query);
    let (fast_s, fast_v) = cache.fast_attention(&query);
    assert!((lin_s - fast_s).abs() < 1e-3);
    assert_eq!(lin_v, fast_v);
}

#[test]
fn test_percepta_arithmetic_multiplication() {
    let mut cache = KVCache2D::new();
    let query = Vec2::new(1.0, 0.0);

    cache.append(Vec2::new(0.0, 0.0), 0);

    for step in 1..=8 {
        let (_, prev) = cache.fast_attention(&query);
        let next = prev + 7;
        cache.append(Vec2::new(step as f32, next as f32), next);
    }

    let (_, result) = cache.fast_attention(&query);
    assert_eq!(result, 56, "7 × 8 = 56");
}

#[test]
fn test_percepta_arithmetic_division() {
    let mut cache = KVCache2D::new();
    let query = Vec2::new(1.0, 0.0);

    cache.append(Vec2::new(0.0, 100.0), 100);

    let mut quotient = 0usize;
    for step in 1.. {
        let (_, prev) = cache.fast_attention(&query);
        if prev < 7 {
            break;
        }
        let next = prev - 7;
        cache.append(Vec2::new(step as f32, next as f32), next);
        quotient += 1;
    }

    let (_, remainder) = cache.fast_attention(&query);
    assert_eq!(quotient, 14, "100 ÷ 7 = 14");
    assert_eq!(remainder, 2, "100 % 7 = 2");
}

#[test]
fn test_percepta_arithmetic_power() {
    let mut cache = KVCache2D::new();
    let query = Vec2::new(1.0, 0.0);

    cache.append(Vec2::new(0.0, 1.0), 1);

    for step in 1..=10 {
        let (_, prev) = cache.fast_attention(&query);
        let next = prev * 2;
        cache.append(Vec2::new(step as f32, next as f32), next);
    }

    let (_, result) = cache.fast_attention(&query);
    assert_eq!(result, 1024, "2^10 = 1024");
    assert!(cache.hull_len() <= 2, "exponential trace compresses to 2");
}

#[test]
fn test_percepta_arithmetic_combined_vm() {
    let mut cache = KVCache2D::new();
    let query = Vec2::new(1.0, 0.0);

    // (3 + 5) * 2 - 2 = 14
    let program: Vec<(&str, usize)> = vec![("LOAD", 3), ("ADD", 5), ("MUL", 2), ("SUB", 2)];

    let mut expected = 0usize;
    for (step, (opcode, operand)) in program.iter().enumerate() {
        let acc = match *opcode {
            "LOAD" => *operand,
            "ADD" => {
                let (_, prev) = cache.fast_attention(&query);
                prev + operand
            }
            "SUB" => {
                let (_, prev) = cache.fast_attention(&query);
                prev - operand
            }
            "MUL" => {
                let (_, prev) = cache.fast_attention(&query);
                prev * operand
            }
            _ => panic!("unknown opcode: {opcode}"),
        };
        cache.append(Vec2::new(step as f32, acc as f32), acc);
        expected = acc;
    }

    let (_, result) = cache.fast_attention(&query);
    assert_eq!(result, 14, "(3 + 5) × 2 - 2 = 14");
    assert_eq!(result, expected);

    let (lin_s, lin_v) = cache.linear_attention(&query);
    let (fast_s, fast_v) = cache.fast_attention(&query);
    assert!((lin_s - fast_s).abs() < 1e-3);
    assert_eq!(lin_v, fast_v);
}

// ── Percepta: Backtracking Computation (Sudoku & N-Queens) ────────
//
// The Percepta blog solved the Arto Inkala Sudoku (hardest in the world)
// inside a transformer at 32K tok/s — NO training needed. They COMPILED
// a C solver into transformer weights. The model executes deterministically.
//
// These tests prove our 2D attention mechanism correctly tracks
// backtracking search: forward placements, dead-end detection, undos,
// and alternative branches.

fn sudoku4_check(board: &[u8; 16], pos: usize, digit: u8) -> bool {
    let row = pos / 4;
    let col = pos % 4;
    for c in 0..4 {
        if board[row * 4 + c] == digit {
            return false;
        }
    }
    for r in 0..4 {
        if board[r * 4 + col] == digit {
            return false;
        }
    }
    let br = (row / 2) * 2;
    let bc = (col / 2) * 2;
    for r in br..br + 2 {
        for c in bc..bc + 2 {
            if board[r * 4 + c] == digit {
                return false;
            }
        }
    }
    true
}

fn sudoku4_valid(board: &[u8; 16]) -> bool {
    for pos in 0..16 {
        let d = board[pos];
        if d == 0 {
            return false;
        }
        let mut tmp = *board;
        tmp[pos] = 0;
        if !sudoku4_check(&tmp, pos, d) {
            return false;
        }
    }
    true
}

fn sudoku4_solve(board: &mut [u8; 16], cache: &mut KVCache2D, step: &mut usize) -> bool {
    let filled = board.iter().filter(|&&v| v > 0).count();
    cache.append(Vec2::new(*step as f32, filled as f32 * 10.0), *step);
    *step += 1;

    let pos = match board.iter().position(|&v| v == 0) {
        Some(p) => p,
        None => return true,
    };

    for digit in 1..=4u8 {
        if sudoku4_check(board, pos, digit) {
            board[pos] = digit;
            if sudoku4_solve(board, cache, step) {
                return true;
            }
            board[pos] = 0;
        }
    }
    false
}

fn nqueens_check(queens: &[i32], row: usize, col: i32) -> bool {
    for (r, &c) in queens.iter().enumerate().take(row) {
        if c == col || (c - col).abs() == (r as i32 - row as i32).abs() {
            return false;
        }
    }
    true
}

fn nqueens_solve(
    queens: &mut [i32],
    row: usize,
    n: usize,
    cache: &mut KVCache2D,
    step: &mut usize,
) -> bool {
    let placed = queens.iter().filter(|&&q| q >= 0).count();
    cache.append(Vec2::new(*step as f32, placed as f32 * 10.0), *step);
    *step += 1;

    if row >= n {
        return true;
    }

    for col in 0..n {
        if nqueens_check(queens, row, col as i32) {
            queens[row] = col as i32;
            if nqueens_solve(queens, row + 1, n, cache, step) {
                return true;
            }
            queens[row] = -1;
        }
    }
    false
}

#[test]
fn test_percepta_backtracking_pattern() {
    let mut cache = KVCache2D::new();

    // Forward: depth 0→1→2→3→4 (peak)
    cache.append(Vec2::new(0.0, 10.0), 0);
    cache.append(Vec2::new(1.0, 20.0), 1);
    cache.append(Vec2::new(2.0, 30.0), 2);
    cache.append(Vec2::new(3.0, 40.0), 3);
    cache.append(Vec2::new(4.0, 50.0), 4); // peak

    // Dead end → backtrack to depth 2
    cache.append(Vec2::new(5.0, 30.0), 5); // valley

    // New branch from depth 2 → deeper
    cache.append(Vec2::new(6.0, 40.0), 6);
    cache.append(Vec2::new(7.0, 50.0), 7);
    cache.append(Vec2::new(8.0, 60.0), 8);
    cache.append(Vec2::new(9.0, 70.0), 9); // solution

    let query = Vec2::new(1.0, 0.0);
    let (_, result) = cache.fast_attention(&query);
    assert_eq!(result, 9, "should return final state");

    // Hull captures peaks (not valleys)
    let hull = cache.hull_indices();
    assert!(
        hull.len() <= 3,
        "hull should compress to ~3, got {}",
        hull.len()
    );
    assert!(hull.contains(&9), "hull should contain solution");
    assert!(
        !hull.contains(&5),
        "hull should NOT contain backtrack valley"
    );

    let (lin_s, lin_v) = cache.linear_attention(&query);
    let (fast_s, fast_v) = cache.fast_attention(&query);
    assert!((lin_s - fast_s).abs() < 1e-3);
    assert_eq!(lin_v, fast_v);
}

#[test]
fn test_percepta_sudoku_4x4() {
    // 4×4 Sudoku: 1 _ _ _ / _ _ 2 _ / _ 3 _ _ / _ _ _ 4
    let mut board: [u8; 16] = [1, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 4];
    let mut cache = KVCache2D::new();
    let mut step = 0usize;

    let solved = sudoku4_solve(&mut board, &mut cache, &mut step);

    assert!(solved, "4×4 Sudoku should be solvable");
    assert!(board.iter().all(|&v| v > 0), "all cells filled");
    assert!(sudoku4_valid(&board), "solution should satisfy constraints");

    let query = Vec2::new(1.0, 0.0);
    let (lin_s, lin_v) = cache.linear_attention(&query);
    let (fast_s, fast_v) = cache.fast_attention(&query);
    assert!((lin_s - fast_s).abs() < 1e-3);
    assert_eq!(lin_v, fast_v);
    assert_eq!(fast_v, step - 1, "should return final step");

    // Hull should compress the backtracking trace
    assert!(
        cache.hull_len() < cache.len(),
        "hull should compress: hull={}, total={}",
        cache.hull_len(),
        cache.len()
    );
}

#[test]
fn test_percepta_nqueens_8() {
    let mut queens: [i32; 8] = [-1; 8];
    let mut cache = KVCache2D::new();
    let mut step = 0usize;

    let solved = nqueens_solve(&mut queens, 0, 8, &mut cache, &mut step);

    assert!(solved, "8-Queens should have a solution");
    assert!(queens.iter().all(|&q| q >= 0), "all queens placed");

    for i in 0..8 {
        for j in i + 1..8 {
            assert_ne!(queens[i], queens[j], "queens {i},{j} same column");
            assert_ne!(
                (queens[i] - queens[j]).abs(),
                (j - i) as i32,
                "queens {i},{j} same diagonal"
            );
        }
    }

    let query = Vec2::new(1.0, 0.0);
    let (lin_s, lin_v) = cache.linear_attention(&query);
    let (fast_s, fast_v) = cache.fast_attention(&query);
    assert!((lin_s - fast_s).abs() < 1e-3);
    assert_eq!(lin_v, fast_v);

    assert!(
        cache.hull_len() < cache.len(),
        "8-Queens hull should compress: hull={}, total={}",
        cache.hull_len(),
        cache.len()
    );
}

// ── 9×9 Sudoku + Computable LoRA Tests ───────────────────────────

#[test]
fn test_sudoku9x9_arto_inkala_clues() {
    let puzzle = microgpt_rs::percepta::Sudoku9x9::arto_inkala();
    assert_eq!(puzzle.clue_count(), 21, "Arto Inkala should have 21 clues");
    assert_eq!(puzzle.grid[0][0], 8);
    assert_eq!(puzzle.grid[8][1], 9);
}

#[test]
fn test_sudoku9x9_is_valid_move() {
    let puzzle = microgpt_rs::percepta::Sudoku9x9::arto_inkala();
    // Row 0 has 8, so placing 8 at (0,1) is invalid (row)
    assert!(!puzzle.is_valid_move(0, 1, 8));
    // Col 1 has 5 (at row 3), 7 (row 2), 9 (row 8), so those are invalid for col
    assert!(!puzzle.is_valid_move(0, 1, 5));
    assert!(!puzzle.is_valid_move(0, 1, 7));
    // Box (0,0)-(2,2) contains 8,3,7 — placing those is invalid
    assert!(!puzzle.is_valid_move(1, 0, 3));
    // 0 is never valid
    assert!(!puzzle.is_valid_move(0, 1, 0));
    // Something valid
    assert!(puzzle.is_valid_move(0, 1, 1));
}

#[test]
fn test_sudoku9x9_display_format() {
    let puzzle = microgpt_rs::percepta::Sudoku9x9::arto_inkala();
    let display = puzzle.display();
    assert!(display.contains("8 . . | . . . | . . "));
    assert!(display.contains("------+-------+------"));
}

#[test]
fn test_sudoku9x9_solve_arto_inkala() {
    let mut puzzle = microgpt_rs::percepta::Sudoku9x9::arto_inkala();
    let mut cache = microgpt_rs::percepta::KVCache2D::new();
    let mut step = 0usize;

    let solved = puzzle.solve(&mut cache, &mut step);

    assert!(solved, "Arto Inkala should be solvable");
    assert!(puzzle.is_solved(), "board should be fully solved");
    assert!(step > 0, "should take at least one step");

    // All cells filled
    for r in 0..9 {
        for c in 0..9 {
            assert!(puzzle.grid[r][c] > 0, "cell ({r},{c}) should be filled");
        }
    }
}

#[test]
fn test_sudoku9x9_solve_hull_compression() {
    let mut puzzle = microgpt_rs::percepta::Sudoku9x9::arto_inkala();
    let mut cache = microgpt_rs::percepta::KVCache2D::new();
    let mut step = 0usize;

    puzzle.solve(&mut cache, &mut step);

    assert!(
        cache.hull_len() < cache.len(),
        "9×9 hull should compress: hull={}, total={}",
        cache.hull_len(),
        cache.len()
    );

    // Attention retrieves final state
    let query = microgpt_rs::percepta::Vec2::new(1.0, 0.0);
    let (lin_s, lin_v) = cache.linear_attention(&query);
    let (fast_s, fast_v) = cache.fast_attention(&query);
    assert!((lin_s - fast_s).abs() < 1e-3, "scores should match");
    assert_eq!(lin_v, fast_v, "values should match");
    assert_eq!(fast_v, step - 1, "should return final step");
}

#[test]
fn test_computable_lora_prune_drafts() {
    use microgpt_rs::percepta::{ComputableLora, Sudoku9x9};

    let puzzle = Sudoku9x9::arto_inkala();

    // Cell (0,1): row 0 has 8, col 1 has 5/7/9, box has 8/3/7
    // Valid digits for (0,1): 1, 2, 4, 6
    let drafts: Vec<(u8, f32)> = vec![
        (8, -0.1), // Invalid: in row and box
        (5, -0.3), // Invalid: in col
        (7, -0.5), // Invalid: in col and box
        (3, -0.7), // Invalid: in box
        (2, -1.0), // Valid
        (1, -1.2), // Valid
    ];

    let pruned = ComputableLora::prune_drafts(&puzzle, 0, 1, &drafts);

    assert_eq!(pruned.len(), 2, "should have 2 valid moves");
    assert_eq!(pruned[0].0, 2, "highest prob valid = 2");
    assert_eq!(pruned[1].0, 1, "next valid = 1");

    // All pruned are valid
    for (digit, _) in &pruned {
        assert!(
            puzzle.is_valid_move(0, 1, *digit),
            "digit {digit} should be valid"
        );
    }
}

#[test]
fn test_computable_lora_prune_all_invalid() {
    use microgpt_rs::percepta::{ComputableLora, Sudoku9x9};

    let puzzle = Sudoku9x9::arto_inkala();

    // Cell (0,0) already has 8 — all drafts for that cell should be pruned
    // because any digit placed there would conflict with the existing 8
    let _drafts: Vec<(u8, f32)> = vec![(1, -0.1), (2, -0.2), (3, -0.3)];

    // Actually (0,0) is filled, so let's test a cell where all proposed digits are invalid
    // Cell (1,2): already has 3, so placing anything there should be "invalid"
    // because the cell is already filled. But is_valid_move only checks if the digit
    // conflicts with neighbors. Let's pick a filled cell and verify all neighbors block it.
    // Better: cell (0,1) where we propose only invalid digits
    let all_invalid: Vec<(u8, f32)> = vec![
        (8, -0.1), // in row + box
        (3, -0.2), // in box
        (7, -0.3), // in col + box
    ];

    let pruned = ComputableLora::prune_drafts(&puzzle, 0, 1, &all_invalid);
    assert!(pruned.is_empty(), "all drafts should be pruned");
}

#[test]
fn test_streaming_solver_arto_inkala() {
    use microgpt_rs::percepta::{SolveEvent, StreamingSolver};

    let mut solver = StreamingSolver::new(microgpt_rs::percepta::Sudoku9x9::arto_inkala().grid);

    let solved = solver.solve_streaming();

    assert!(solved, "should solve");
    assert!(solver.state.is_solved(), "board should be solved");
    assert!(!solver.events.is_empty(), "should have events");

    // Should have at least one Solved event
    let has_solved = solver
        .events
        .iter()
        .any(|e| matches!(e, SolveEvent::Solved { .. }));
    assert!(has_solved, "should have Solved event");

    // Should have Try and Accepted events
    let has_try = solver
        .events
        .iter()
        .any(|e| matches!(e, SolveEvent::Try { .. }));
    let has_accepted = solver
        .events
        .iter()
        .any(|e| matches!(e, SolveEvent::Accepted { .. }));
    assert!(has_try, "should have Try events");
    assert!(has_accepted, "should have Accepted events");

    // Format should produce non-empty output
    let output = solver.format_events();
    assert!(!output.is_empty(), "format should produce output");
    assert!(output.contains("Solved"), "should mention solving");
    assert!(
        output.contains("Hull compression"),
        "should show hull stats"
    );
}

#[test]
fn test_sudoku9x9_next_empty() {
    let puzzle = microgpt_rs::percepta::Sudoku9x9::arto_inkala();
    let empty = puzzle.next_empty();
    assert!(empty.is_some(), "should find empty cell");
    let (r, c) = empty.unwrap();
    assert_eq!(puzzle.grid[r][c], 0, "returned cell should be empty");
}
