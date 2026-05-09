//! Router Demo — demonstrates the full prompt routing pipeline.
//!
//! Shows how to:
//! 1. Define domains via `RouterConfig` (normally loaded from `domains.toml`)
//! 2. Build a `KeywordRouter` + `ExpertRegistry`
//! 3. Route prompts to domains
//! 4. Fetch expert bundles for DDTree construction
//!
//! # Run
//!
//! ```sh
//! cargo run --example router_demo --features router
//! ```

use std::path::Path;

use microgpt_rs::router::{
    DomainConfig, ExpertRegistry, KeywordRouter, PromptRouter, RouterConfig,
};

fn main() {
    println!("=== microgpt-rs Prompt Router Demo (Plan 023) ===\n");

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
                reader_lora: None,
                writer_lora: None,
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
                reader_lora: None,
                writer_lora: None,
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
                reader_lora: None,
                writer_lora: None,
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
                reader_lora: None,
                writer_lora: None,
                native_pruner: None,
            },
            DomainConfig {
                name: "general".into(),
                keywords: vec![],
                pruner: None,
                lora: None,
                reader_lora: None,
                writer_lora: None,
                native_pruner: Some("no_pruner".into()),
            },
        ],
    };

    let pruner_dir = Path::new("./pruners");

    // -----------------------------------------------------------------------
    // 2. Build router + registry
    // -----------------------------------------------------------------------

    let router = KeywordRouter::new(config.domain.clone());
    let registry = ExpertRegistry::from_config(&config, pruner_dir);

    println!("Configured {} domains:\n", config.domain.len());
    for domain in &config.domain {
        let kw_display = if domain.keywords.is_empty() {
            "(fallback)".to_string()
        } else {
            domain.keywords.join(", ")
        };
        println!("  {} — keywords: {kw_display}", domain.name);
    }
    println!();

    // -----------------------------------------------------------------------
    // 3. Route sample prompts
    // -----------------------------------------------------------------------

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

        println!("Prompt: \"{prompt}\"");
        println!("  → domain:     {}", decision.domain);
        println!("  → confidence: {:.3}", decision.confidence);
        println!("  → pruner:     {pruner_type}");
        println!();
    }

    // -----------------------------------------------------------------------
    // 4. EMO concept — expert pool is locked for the generation
    // -----------------------------------------------------------------------

    println!("=== EMO Concept ===");
    println!("Once the router selects a domain, the ExpertBundle is locked");
    println!("for the entire DDTree generation. The draft model cannot");
    println!("domain-drift because its pruner is fixed.");
    println!();

    let decision = router.route("solve this sudoku puzzle");
    let expert = registry.get_expert(&decision.domain);

    println!(
        "Locked expert: domain=\"{}\" lora={:?}",
        expert.domain, expert.lora_path,
    );
    println!("This expert's pruner will score every token in the DDTree.");
    println!("Tokens that drift away from the domain get low relevance scores");
    println!("and are naturally pruned by the screening mechanism.");
}
