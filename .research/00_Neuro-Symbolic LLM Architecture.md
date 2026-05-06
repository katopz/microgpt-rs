# **Research Context: Advanced Neuro-Symbolic Inference Architecture**

**Project Focus:** "Rewrite it in Rust" (RIIR) Translation Service

**Core Technologies:** microgpt-rs, anyrag, Turso SQLite, Percepta In-Model Execution

## **Abstract**

This document outlines a novel architecture for an edge-capable, highly optimized Large Language Model (LLM) inference engine. The goal is to build a "Rewrite it in Rust" (RIIR) service that guarantees syntactically correct output with extremely low latency. The architecture merges three cutting-edge AI paradigms:

1. **Speculative Decoding** via micro-transformers (microgpt-rs).  
2. **Retrieval-Based Speculative Decoding & Self-Improving RAG** using vector databases (anyrag \+ Turso).  
3. **In-Model Computation / Computable LoRA** via deterministic execution traces and ![][image1] attention (Percepta concept).

## **Part 1: The Engine Foundation (microgpt-rs)**

The base of the architecture is inspired by microgpt-rs, a high-performance Rust implementation of a micro-Transformer utilizing speculative decoding.

### **Core Mechanics**

* **Zero-Allocation Forward Pass:** Memory buffers (ForwardContext) are pre-allocated, avoiding dynamic heap allocations during the inference loop, ensuring predictable and extremely fast execution in Rust.  
* **DFlash (Dynamic Flash):** A block-parallel drafting mechanism that bypasses strict causal masking. It predicts ![][image2] future tokens simultaneously via independent marginal distributions.  
* **DDTree (Dynamic Draft Tree):** Instead of a linear draft chain, it uses a Best-First Search algorithm (via a max-heap) to build a tree of the most probable token paths. The target model then verifies these branches in parallel.

### **Known Bottlenecks & Hardware Improvements**

Currently, a raw microgpt-rs implementation yields a \~0.91x speedup because an untrained draft model only has a \~75% acceptance rate. To push this to \>1.0x, the following optimizations are required:

* **Hardware:** Implementation of SIMD intrinsics (ARM NEON / AVX2), Apple Metal Performance Shaders (MPS), or WGPU compute shaders.  
* **Architecture:** Introduction of Grouped-Query Attention (GQA) to shrink the KV cache size, INT8/4-bit quantization, and a Paged KV Cache to manage the DDTree branching without memory fragmentation.

## **Part 2: Database-as-a-Shim & Self-Improving RAG (anyrag \+ Turso)**

Instead of relying solely on a mathematically trained draft model, the architecture leverages **Retrieval-Based Speculative Decoding** using Turso SQLite (with vector support) and the anyrag framework.

### **The "Free" Embedding**

During target model inference, the last hidden state (prior to the unembedding layer) is extracted. Because this dense vector is already computed, it serves as a "free" embedding to query the Turso database for historical token continuations.

### **Addressing Latency via anyrag**

To ensure database retrieval does not become the bottleneck, the anyrag framework handles:

1. **Semantic Caching:** High-frequency queries (e.g., "How to implement Send \+ Sync") bypass the embedding step and return exact cached contexts.  
2. **Concept Sharding:** Queries are classified by anyrag's Query Analysis (e.g., "Lifetimes", "Macros") and routed to specific, smaller shards in Turso, ensuring sub-millisecond HNSW vector lookups.  
3. **Knowledge Graph Routing (IndraDB):** Mapping relationships between concepts (e.g., Arc \-\> Mutex \-\> Threads) to pull relevant context instantly without relying purely on fuzzy vector similarity.

### **The Self-Improving "Runtime LoRA" Pipeline**

This system creates a continuous learning loop without backpropagation:

1. **Day 1 (RAG Phase):** The system answers RIIR queries by pulling chunks from ingested sources (The Rust Book, standard library, GitHub repos via Code RAG).  
2. **Day 30 (Synthesis):** anyrag synthesizes the most common and successful code translation patterns into structured Q\&A pairs.  
3. **Day 31 (Export & Fine-tune):** The /knowledge/export endpoint generates JSONL files used to train a LoRA on the base model.  
4. **Day 32 (Base Model Upgrade):** The base model natively understands the translations. The SQLite episodic memory is cleared to begin learning new edge cases.

*Note on Context Pollution:* anyrag must heavily weight the Code RAG (/search/examples) over standard documentation when doing RIIR to prevent the LLM from generating explanatory text instead of executable code.

## **Part 3: LLMs as Computers & Computable LoRA (Percepta Concept)**

To guarantee the syntactical correctness of the generated Rust code, the architecture integrates the concept of **In-Model Execution**.

### **The Concept**

LLMs typically rely on external interpreters (tool-use) to run or verify code. The Percepta architecture proves that a Transformer can execute compiled code (like WebAssembly) *internally* by generating an execution trace token-by-token.

* **The Technical Unlock:** By restricting lookup heads to a dimension of 2 (2D heads), the Transformer's attention lookups shift from ![][image3] linear scans to ![][image1] convex-hull queries. This allows the model to execute millions of steps without performance degradation as context grows.

### **The "Computable LoRA"**

We adapt this into a **Computable LoRA**: an adapter whose weights represent a deterministic state-machine or lightweight Rust parser.

* Instead of predicting the next text token based on semantic likelihood, the Computable LoRA predicts the next *deterministic execution state*.  
* As the base model generates code, the Computable LoRA runs alongside it, checking the syntax tree.

## **Part 4: The Grand Unification (Neuro-Symbolic RIIR Architecture)**

How a request ("Rewrite this Python class in Rust") flows through the completed system:

1. **Prompt Ingestion & RAG:** The user submits Python code. anyrag queries Turso/GitHub to retrieve 3 examples of idiomatic Rust structs and traits, appending them to the system prompt.  
2. **Drafting (The Shim):**  
   * The microgpt-rs engine begins drafting tokens.  
   * Simultaneously, the engine queries Turso using the current hidden state to retrieve highly probable token sequences from past successful compilations.  
   * These sequences populate the DDTree (Dynamic Draft Tree).  
3. **Rule Pruning (Computable LoRA):**  
   * Before the large target model verifies the DDTree, the **Computable LoRA** (utilizing ![][image1] fast attention) evaluates the branches.  
   * If a drafted branch violates Rust syntax (e.g., missing semicolons, invalid mutable borrows), the Computable LoRA immediately outputs a halt/error signal internally, pruning that branch from the tree.  
4. **Target Verification:** The large target model evaluates the remaining, syntactically perfect branches, accepting the longest valid sequence.  
5. **Feedback Loop:** Once the generated code is compiled and verified by the user, the sequence and its hidden states are INSERTed back into Turso, instantly upgrading the "Runtime LoRA" for the next user.

## **Conclusion**

This neuro-symbolic architecture offsets the weaknesses of LLMs (hallucination, strict rule adherence) with the strengths of deterministic state machines and graph databases, all while maintaining inference speed through speculative decoding and ![][image1] execution traces.
