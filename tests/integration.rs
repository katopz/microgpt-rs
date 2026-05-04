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
    // Should not panic or loop forever
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
    // Should be monotonically increasing since input was [1,2,3]
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
    probs[5] = 1.0; // all mass on token 5
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
    let mut cache = transformer::KVCache::new(&config);
    let logits = transformer::forward(&weights, &mut cache, 0, 0, &config);
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_forward_logits_finite() {
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    let mut cache = transformer::KVCache::new(&config);
    let logits = transformer::forward(&weights, &mut cache, 0, 0, &config);
    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "logit {i} is not finite: {l}");
    }
}

#[test]
fn test_forward_cache_populated() {
    let config = types::Config::micro();
    let n = config.n_embd;
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    let mut cache = transformer::KVCache::new(&config);
    transformer::forward(&weights, &mut cache, 0, 0, &config);
    let key_sum: f32 = cache.key[..n].iter().sum();
    let val_sum: f32 = cache.value[..n].iter().sum();
    assert!(key_sum != 0.0, "K cache at pos 0 should be populated");
    assert!(val_sum != 0.0, "V cache at pos 0 should be populated");
}

#[test]
fn test_forward_positions_differ() {
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    let mut cache = transformer::KVCache::new(&config);
    let logits_0 = transformer::forward(&weights, &mut cache, 0, 0, &config);
    let logits_1 = transformer::forward(&weights, &mut cache, 0, 1, &config);
    let different = logits_0.iter().zip(&logits_1).any(|(&a, &b)| a != b);
    assert!(different, "logits at different positions should differ");
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
    let mut cache = transformer::KVCache::new(&config);
    transformer::forward(&weights, &mut cache, 0, 0, &config);
    cache.reset();
    assert!(
        cache.key.iter().all(|&v| v == 0.0),
        "cache key should be zeroed after reset"
    );
    assert!(
        cache.value.iter().all(|&v| v == 0.0),
        "cache value should be zeroed after reset"
    );
}

// ── speculative ────────────────────────────────────────────────

#[test]
fn test_dflash_produces_marginals() {
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    let marginals = speculative::dflash_predict(&weights, 0, 0, &config);
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
fn test_dflash_positions_differ() {
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    let m0 = speculative::dflash_predict(&weights, 0, 0, &config);
    let m1 = speculative::dflash_predict(&weights, 0, 1, &config);
    assert_ne!(
        m0[0], m1[0],
        "marginals at different positions should differ"
    );
}

#[test]
fn test_ddtree_respects_budget() {
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    let marginals = speculative::dflash_predict(&weights, 0, 0, &config);
    let tree = speculative::build_dd_tree(&marginals, &config);
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
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    let marginals = speculative::dflash_predict(&weights, 0, 0, &config);
    let tree = speculative::build_dd_tree(&marginals, &config);
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
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    let marginals = speculative::dflash_predict(&weights, 0, 0, &config);
    let tree = speculative::build_dd_tree(&marginals, &config);
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
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    let marginals = speculative::dflash_predict(&weights, 0, 0, &config);
    let tree = speculative::build_dd_tree(&marginals, &config);
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
    let config = types::Config::micro();
    let tree = speculative::build_dd_tree(&[], &config);
    assert!(tree.is_empty(), "empty marginals should produce empty tree");
}

#[test]
fn test_speculative_step_accepts_at_least_one() {
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);
    for seed in [0, 42, 100, 999] {
        let mut step_rng = types::Rng::new(seed);
        let (accepted, accept_len) =
            speculative::speculative_step(&weights, 0, 0, &config, &mut step_rng);
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
    let config = types::Config::micro();
    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);

    let mut rng1 = types::Rng::new(77);
    let (a1, l1) = speculative::speculative_step(&weights, 0, 0, &config, &mut rng1);

    let mut rng2 = types::Rng::new(77);
    let (a2, l2) = speculative::speculative_step(&weights, 0, 0, &config, &mut rng2);

    assert_eq!(a1, a2, "same seed should produce same accepted tokens");
    assert_eq!(l1, l2, "same seed should produce same acceptance length");
}
