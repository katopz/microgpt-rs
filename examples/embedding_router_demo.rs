//! Embedding Router Demo — demonstrates KV cache priming via embedding retrieval (Plan 024).
//!
//! Shows how to:
//! 1. Build an `EmbeddingRouter` with `TruncatePadProjector`
//! 2. Use sync `route()` as keyword fallback
//! 3. Use async `route_async()` with three-tier fallback
//! 4. Project embeddings to the draft model's hidden dimension
//! 5. Demonstrate the full pipeline with simulated embeddings
//!
//! # Run
//!
//! ```sh
//! cargo run --example embedding_router_demo --features embedding_router
//! ```
//!
//! # Prerequisites
//!
//! For full embedding retrieval, a running anyrag server is needed.
//! The demo gracefully falls back to keyword routing when the server is down.

use std::path::Path;

use microgpt_rs::router::{
    DomainConfig, EmbeddingRouteDecision, EmbeddingRouter, EmbeddingRouterConfig, ExpertRegistry,
    PromptRouter, RouterConfig, TruncatePadProjector,
};

fn main() {
    println!("=== microgpt-rs Embedding Router Demo (Plan 024) ===\n");

    // -----------------------------------------------------------------------
    // 1. Build domain config (in production, loaded from domains.toml)
    // -----------------------------------------------------------------------

    let config = RouterConfig {
        domain: vec![
            DomainConfig {
                name: "sudoku".into(),
                keywords: vec![
                    "sudoku".into(),
                    "puzzle".into(),
                    "grid".into(),
                    "9x9".into(),
                    "digit".into(),
                ],
                pruner: None,
                lora: None,
                native_pruner: Some("sudoku".into()),
            },
            DomainConfig {
                name: "pathfinding".into(),
                keywords: vec![
                    "path".into(),
                    "maze".into(),
                    "bear".into(),
                    "terrain".into(),
                    "tactical".into(),
                    "grid".into(),
                ],
                pruner: None,
                lora: None,
                native_pruner: Some("tactical".into()),
            },
            DomainConfig {
                name: "rust_code".into(),
                keywords: vec![
                    "rust".into(),
                    "cargo".into(),
                    "axum".into(),
                    "tokio".into(),
                    "trait".into(),
                    "impl".into(),
                    "compile".into(),
                ],
                pruner: Some("syn_validator.wasm".into()),
                lora: None,
                native_pruner: None,
            },
            DomainConfig {
                name: "py2rs".into(),
                keywords: vec![
                    "python".into(),
                    "rewrite".into(),
                    "fastapi".into(),
                    "flask".into(),
                    "translate".into(),
                ],
                pruner: Some("syn_validator.wasm".into()),
                lora: Some("py2rs_lora.bin".into()),
                native_pruner: None,
            },
            DomainConfig {
                name: "general".into(),
                keywords: vec![],
                pruner: None,
                lora: None,
                native_pruner: Some("no_pruner".into()),
            },
        ],
    };

    let pruner_dir = Path::new("./pruners");

    // -----------------------------------------------------------------------
    // 2. Build EmbeddingRouter with TruncatePadProjector
    // -----------------------------------------------------------------------

    let embedding_config = EmbeddingRouterConfig {
        anyrag_url: "http://localhost:9090".into(),
        timeout_ms: 200,
        classify_domain: true,
        auth_token: None,
    };

    let router = EmbeddingRouter::new(
        embedding_config,
        config.domain.clone(),
        Box::new(TruncatePadProjector),
    );
    let registry = ExpertRegistry::from_config(&config, pruner_dir);

    let domain_count = config.domain.len();
    println!("Configured {domain_count} domains with TruncatePadProjector\n");

    // -----------------------------------------------------------------------
    // 3. Sync routing (keyword fallback — no network needed)
    // -----------------------------------------------------------------------

    println!("=== Sync Routing (Keyword Fallback) ===\n");

    let prompts = [
        "solve this sudoku puzzle with a 9x9 grid",
        "write Rust code for an HTTP server using axum and tokio",
        "find the shortest path through the maze for the blue bear",
        "translate this FastAPI python code to Rust axum",
        "what is the meaning of life?",
    ];

    for prompt in &prompts {
        let decision = router.route(prompt);
        let expert = registry.get_expert(&decision.domain);

        let pruner_type = if decision.domain == "general" {
            "NoScreeningPruner".to_string()
        } else {
            match &expert.lora_path {
                Some(lora) => format!("ScreeningPruner + LoRA({})", lora.display()),
                None => "ScreeningPruner".to_string(),
            }
        };

        let domain = &decision.domain;
        let confidence = decision.confidence;
        println!("Prompt: \"{prompt}\"");
        println!("  → domain:     {domain}");
        println!("  → confidence: {confidence:.3}");
        println!("  → pruner:     {pruner_type}");
        println!("  → embedding:  None (sync mode)");
        println!();
    }

    // -----------------------------------------------------------------------
    // 4. Async routing with three-tier fallback
    // -----------------------------------------------------------------------

    println!("=== Async Routing (Three-Tier Fallback) ===\n");
    println!("Attempting to connect to anyrag at http://localhost:9090...");
    println!("(This will fall back to keyword routing if server is down)\n");

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");

    for prompt in &prompts {
        let decision: EmbeddingRouteDecision = rt.block_on(router.route_async(prompt));

        let embedding_status = match &decision.embedding {
            Some(emb) => {
                let dims = emb.len();
                format!("Some({dims}") + " dims)"
            }
            None => "None (fallback)".to_string(),
        };

        let domain = &decision.route.domain;
        let confidence = decision.route.confidence;
        println!("Prompt: \"{prompt}\"");
        println!("  → domain:           {domain}");
        println!("  → confidence:       {confidence:.3}");
        println!("  → embedding:        {embedding_status}");
        println!("  → embedding_source: {:?}", decision.embedding_source);
        println!();
    }

    // -----------------------------------------------------------------------
    // 5. Embedding projection demonstration
    // -----------------------------------------------------------------------

    println!("=== Embedding Projection (768 → 64 dims) ===\n");

    // Simulate a 768-dim embedding from a retrieval model (e.g., BERT)
    let simulated_embedding: Vec<f32> = (0..768).map(|i| (i as f32 * 0.001).sin()).collect();

    // Project to draft model's n_embd (e.g., 64)
    let draft_n_embd = 64;
    let projected = router.project_embedding(&simulated_embedding, draft_n_embd);

    let input_dims = simulated_embedding.len();
    let output_dims = projected.len();
    let nonzero = projected.iter().filter(|&&v| v != 0.0).count();
    println!("Input embedding:  {input_dims} dims");
    println!("Projected output: {output_dims} dims");
    println!("First 8 values:   {:?}", &projected[..8]);
    println!("Non-zero count:   {nonzero}/{output_dims}");
    println!();

    // Also demonstrate pad case (32 → 64)
    let small_embedding: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
    let padded = router.project_embedding(&small_embedding, draft_n_embd);

    let small_dims = small_embedding.len();
    let padded_dims = padded.len();
    println!("Pad case:  {small_dims} → {padded_dims} dims");
    println!("Last 8 values (should be zeros): {:?}", &padded[56..64]);
    println!();

    // -----------------------------------------------------------------------
    // 6. Full pipeline: route → project → conditioned draft (conceptual)
    // -----------------------------------------------------------------------

    println!("=== Full Pipeline (Conceptual) ===\n");

    let prompt = "fn validate_token(";
    let decision: EmbeddingRouteDecision = rt.block_on(router.route_async(prompt));

    let domain = &decision.route.domain;
    println!("Step 1: Route prompt \"{prompt}\"");
    println!("  → domain:     {domain}");
    let embedding_info: Option<String> = decision.embedding.as_ref().map(|e: &Vec<f32>| {
        let dims = e.len();
        format!("{dims}") + " dims"
    });
    println!("  → embedding:  {embedding_info:?}");

    if let Some(ref embedding_vec) = decision.embedding {
        println!("\nStep 2: Project embedding to draft model dim");
        let projected_for_draft = router.project_embedding(embedding_vec, draft_n_embd);
        let projected_dims = projected_for_draft.len();
        println!("  → projected: {projected_dims} dims");
        println!("  → ready for: speculative_step_embedding_conditioned()");
    } else {
        println!("\nStep 2: No embedding available");
        println!("  → using: speculative_step() (unconditioned)");
    }

    println!("\nStep 3: Run speculative decoding");
    println!("  → If embedding available: dflash_predict_conditioned_with()");
    println!("     seeds KV cache with projected embedding at position 0");
    println!("  → If no embedding: dflash_predict()");
    println!("     standard unconditioned draft");
    println!("  → Draft tokens biased toward retrieved code patterns");

    // -----------------------------------------------------------------------
    // 7. Pipeline architecture summary
    // -----------------------------------------------------------------------

    println!("\n=== Pipeline Architecture ===\n");
    println!("IDE Context (\"editing auth.rs, typing fn validate_token(\")");
    println!("    │");
    println!("    ▼");
    println!("EmbeddingRouter::route_async(prompt, Some(file_context))");
    println!("    │");
    println!("    ├─► HTTP POST anyrag /search/embedding      (Tier 1, ~200ms)");
    println!("    ├─► HTTP POST anyrag /classify/domain        (Tier 2, ~100ms)");
    println!("    └─► KeywordRouter::route(prompt)             (Tier 3, <1ms)");
    println!("    │");
    println!("    ▼");
    println!("EmbeddingRouteDecision {{ domain, embedding: Some(vec![...]) }}");
    println!("    │");
    println!("    ▼");
    println!("EmbeddingProjector::project(&embedding, n_embd)");
    println!("    │");
    println!("    ▼");
    println!("speculative_step_embedding_conditioned(draft, token, pos, &projected, rng)");
    println!("    │");
    println!("    ▼");
    println!("Draft tokens with semantic bias toward retrieved code patterns");
}
