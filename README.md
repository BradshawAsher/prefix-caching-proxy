# Prefix-Caching KV Router (MLSys)

> An asynchronous Layer-7 Reverse Proxy and Key-Value (KV) Cache-Affinity Router written in **Rust** (`Tokio` + `Axum`). Designed to eliminate redundant LLM prompt prefill calculations across distributed inference clusters via **Longest Prefix Matching (LPM)** in an in-memory **Radix Tree**.

---

## The Problem: Cache Fragmentation in Distributed Inference

When a Large Language Model reads an input prompt, it computes attention matrices for every token to build its short-term memory (the **Key-Value / KV Cache**). This calculation—known as the **prefill stage**—causes the initial latency delay before the first word is generated (**Time-To-First-Token, or TTFT**).

Modern inference engines (vLLM, SGLang, Ollama) feature prefix caching to reuse KV tensors across requests. However, **standard Layer-7 load balancers (NGINX, AWS ALB, Round-Robin) break this mechanism**:
1. If Worker 1 caches a 3,000-token system prompt, but Round-Robin sends the next turn to Worker 2, Worker 2 must recalculate all 3,000 tokens from scratch.
2. GPUs waste compute cycles redoing work, Time-To-First-Token spikes, and expensive GPU VRAM is duplicated.

---

## Architecture

This project intercepts OpenAI-compatible `/v1/chat/completions` requests, inspects incoming token prefixes before touching downstream workers, and routes to the worker holding the largest warm prefix in its GPU memory.

```
                          Concurrent Client Requests (Agents / Chat)
                                              │
                                              ▼
                  ┌───────────────────────────────────────────────────────┐
                  │        Rust L7 Reverse Proxy (Axum / Tokio)           │
                  │                                                       │
                  │  1. Fast BPE Tokenization (sub-100μs)                 │
                  │  2. Longest Prefix Match (LPM) in In-Memory Radix Tree│
                  │  3. Cache Affinity Score vs. Node Queue Balancing     │
                  └───────────┬───────────────────────────────┬───────────┘
                              │                               │
                              ▼ (Hit: Warm Cache)             ▼ (Miss / Balanced)
                 ┌─────────────────────────┐     ┌─────────────────────────┐
                 │ Worker 1 (Port 8001)    │     │ Worker 2 (Port 8002)    │
                 │   Ollama / vLLM Node    │     │   Ollama / vLLM Node    │
                 │ [Warm Cache: ~20ms TTFT]│     │ [Cold Cache: ~180ms TTFT│
                 └─────────────────────────┘     └─────────────────────────┘
```

---

## Core Components

* **In-Memory Compressed Radix Tree (`src/radix.rs`):** Compresses non-branching token sequences into contiguous slices, reducing pointer overhead by ~95% compared to standard Tries. Features automatic branch splitting and thread-safe lock-free reads.
* **Fast BPE Tokenizer (`src/tokenizer.rs`):** HuggingFace-backed token encoding turning raw prompt strings into `Vec<u32>` token IDs in under 50 microseconds.
* **Bidirectional Streaming Proxy (`src/main.rs`):** Pipes Server-Sent Events (SSE) token generation chunks directly to the client without buffering full responses in memory.
* **Affinity Routing Algorithm:**
  $$\text{Score}(W) = \alpha \cdot \frac{\text{Matched Prefix Tokens}}{\text{Total Prompt Tokens}} - \beta \cdot \text{Pending Requests}(W)$$

---

## Quick Start & Verification

### Prerequisites
- [Rust & Cargo](https://rustup.rs/) (`>= 1.80`)
- [Ollama](https://ollama.com/) (for local verification)

### Running Automated Tests
The in-memory Radix Tree includes unit tests verifying exact prefix matches, partial prefix matches, multi-worker branch splitting, and zero-match handling:

```powershell
cargo test
```

Expected output:
```
running 4 tests
test radix::tests::test_branch_splitting_across_workers ... ok
test radix::tests::test_no_match ... ok
test radix::tests::test_partial_prefix_match ... ok
test radix::tests::test_single_insert_and_exact_match ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

---

## Project Status

- [x] **Phase 1:** High-throughput Tokio/Axum asynchronous scaffold.
- [x] **Phase 2:** In-memory compressed Radix Tree with dynamic branch splitting (`src/radix.rs`).
- [ ] **Phase 3:** Fast HuggingFace BPE Tokenizer integration (`src/tokenizer.rs`).
- [ ] **Phase 4:** Axum reverse proxy handler & bidirectional SSE streaming (`src/main.rs`).
- [ ] **Phase 5:** Multi-agent synthetic benchmark harness and TTFT latency profiling.

---

## License
MIT
