# Systems & MLSys Interview Preparation Guide: Prefix-Caching KV Router

This guide prepares you to defend this project in **Machine Learning Systems (MLSys)**, **High-Performance Infrastructure**, and **Distributed Systems** engineering interviews (e.g., OpenAI, Anthropic, Databricks, Meta, Cloudflare, Anyscale).

---

## Table of Contents
1. [Core Pitch & High-Level Problem](#1-core-pitch--high-level-problem)
2. [LLM Inference Engine Fundamentals (Prefill vs. Decode)](#2-llm-inference-engine-fundamentals-prefill-vs-decode)
3. [Data Structures: Compressed Radix Tree Mechanics](#3-data-structures-compressed-radix-tree-mechanics)
4. [Tokenizer Pipeline: Why Token IDs vs. Strings](#4-tokenizer-pipeline-why-token-ids-vs-strings)
5. [Concurrency, Async Runtime & Thread-Safety Invariants](#5-concurrency-async-runtime--thread-safety-invariants)
6. [Routing Policy: Cache Locality vs. Load Balancing](#6-routing-policy-cache-locality-vs-load-balancing)
7. [Networking, Zero-Copy SSE Streaming & Backpressure](#7-networking-zero-copy-sse-streaming--backpressure)
8. [Empirical Benchmarks & Profiling Defense](#8-empirical-benchmarks--profiling-defense)
9. [Failure Modes, Cache Invalidation & Distributed Scaling](#9-failure-modes-cache-invalidation--distributed-scaling)
10. [Rapid-Fire Questions & Cheat Sheet](#10-rapid-fire-questions--cheat-sheet)

---

## 1. Core Pitch & High-Level Problem

### Q1: Can you give a 60-second elevator pitch of this project?
> **Interviewer Intent:** Assessing communication clarity, ability to articulate business/systems value, and problem-space ownership.

**Answer:**  
"In distributed LLM serving (e.g., clusters running vLLM, SGLang, or Ollama), processing the prompt—the **prefill stage**—is computationally heavy because attention has quadratic complexity with prompt length. To avoid recomputing these attention tensors for shared system prompts or multi-turn agent conversations, modern engines support **prefix caching**.

However, standard Layer-7 load balancers like NGINX, AWS ALB, or basic Round-Robin are **cache-agnostic**. They spray identical prefix requests across different workers, causing severe **KV cache fragmentation**, cold prefill recalculations, and GPU VRAM duplication.

I built a high-performance Layer-7 Reverse Proxy in **Rust** using **Tokio** and **Axum**. It intercepts incoming `/v1/chat/completions` requests, extracts and tokenizes prompt prefixes via HuggingFace's BPE engine in under 50 microseconds, matches them against an in-memory **Compressed Radix Tree** using **Longest Prefix Matching (LPM)**, and routes the request to the worker holding the largest warm KV cache slice. In benchmarks across multi-agent workflows, this slashed **P95 Time-To-First-Token (TTFT) by 88.3%** and achieved a **100% cache hit rate** with sub-2.5ms proxy routing overhead."

---

### Q2: Why can't you just use standard HTTP sticky sessions or consistent hashing?
> **Interviewer Intent:** Testing whether you understand the fundamental difference between standard web routing and LLM prefix structures.

**Answer:**  
- **Sticky Sessions (e.g., Cookie or IP-based):** Sticky sessions pin a specific *client* or *session* to a worker. But in multi-agent workflows, microservices, or shared enterprise workloads, hundreds of distinct users share the *exact same 2,000-token system prompt* or document context (e.g., RAG knowledge base). Sticky sessions cannot detect shared prompt substrings across independent clients.
- **Consistent Hashing (e.g., Hashing the Prompt String):** Consistent hashing hashes the *entire* request payload. If User A asks `"Summarize doc X. Focus on finances."` and User B asks `"Summarize doc X. Focus on legal."`, hashing the whole text yields completely different hash digests, routing them to separate workers even though 95% of their prompt (`doc X`) is identical.
- **Prefix-Aware Routing via Radix Tree:** Prefix caching requires finding the **Longest Common Subsequence starting at index 0 (Prefix)**. Only a hierarchical spatial data structure like a Trie or Radix Tree can evaluate prefix overlap dynamically across arbitrary prompt lengths.

---

## 2. LLM Inference Engine Fundamentals (Prefill vs. Decode)

### Q3: Why is KV cache reuse such a critical optimization? What happens under the hood during prefill vs. decode?
> **Interviewer Intent:** Verifying your deep knowledge of Transformer mechanics and GPU hardware constraints.

**Answer:**  
Transformer generation operates in two distinct phases:

```
Prompt Tokens [T_0, T_1, ..., T_N] ──► [ PREFILL PHASE ] ──► Compute Q, K, V for all N tokens
                                                                (Compute-bound, Tensor Cores saturated)
                                                                Saves K, V matrices to VRAM (KV Cache)
                                                                           │
                                                                           ▼
Generated Token T_{N+1}            ◄── [ DECODE PHASE ]  ◄── Reads KV Cache of T_0..N from HBM
                                                                (Memory-bandwidth bound, Arithmetic intensity < 1)
```

1. **Prefill Stage (Prompt Processing):**
   - **Characteristics:** The model ingests all $N$ prompt tokens simultaneously. The self-attention operation computes $Q \cdot K^T$ across all tokens, which is $O(N^2)$ in compute and memory footprint without optimizations.
   - **Hardware Bottleneck:** **Compute-bound**. It maximizes GPU Tensor Core arithmetic intensity (TFLOPS).
   - **Impact on Latency:** Directly dictates **Time-To-First-Token (TTFT)**. If a prompt is 4,000 tokens, prefill can take 200ms to 800ms on an A100/H100.
   - **Prefix Cache Value:** If tokens $0 \dots M$ are already in the worker's GPU memory, the engine skips computing attention for those $M$ tokens and only prefills $M+1 \dots N$. This drops TTFT from hundreds of milliseconds to tens of milliseconds.

2. **Decode Stage (Autoregressive Generation):**
   - **Characteristics:** Generates one token at a time ($T_{k} \to T_{k+1}$).
   - **Hardware Bottleneck:** **Memory-bandwidth bound**. In each forward pass, the model must read all previous Key and Value projection matrices from High-Bandwidth Memory (HBM) into SRAM just to calculate attention for a single new token.
   - **Impact on Latency:** Dictates **Inter-Token Latency (ITL)** or tokens-per-second throughput.

---

## 3. Data Structures: Compressed Radix Tree Mechanics

### Q4: Walk through the implementation of your Radix Tree (`src/radix.rs`). Why a Radix Tree instead of a standard Trie?
> **Interviewer Intent:** Evaluating data structure fundamentals, algorithmic optimization, and memory efficiency.

**Answer:**  
In a standard Trie, each node represents a single token, and edges point to subsequent single tokens. For an LLM prompt with 2,048 tokens, an uncompressed Trie requires allocating **2,048 individual heap nodes**, each containing pointers/maps to child nodes.
- **Pointers Overhead:** On 64-bit architectures, pointer overhead and cache-line misses during traversal are massive ($O(N)$ node hops).

In our **Compressed Radix Tree (Patricia Trie)** (`src/radix.rs`):
- Non-branching contiguous token sequences are compressed into a single node holding `Vec<u32>` (`tokens: Vec<TokenId>`).
- If a 1,500-token system prompt has no intermediate branches, it occupies **a single node** with a 1,500-element contiguous vector.
- **Traversal & Cache Locality:** Traversing the prefix requires $O(\text{branches})$ node jumps instead of $O(N)$ node jumps—reducing pointer traversals and node allocations by **>95%**. Contiguous slices are also CPU cache-friendly (L1/L2 data cache prefetching).

```
Standard Trie:   (Root) ─► [Token 1] ─► [Token 2] ─► [Token 3] ... ─► [Token 1500]  (1,500 node allocations)

Radix Tree:      (Root) ─► [Token 1 ... Token 1500]                                 (1 node allocation)
```

---

### Q5: How does dynamic branch splitting work when two prompts diverge?
> **Interviewer Intent:** Testing whether you understand the core algorithmic logic of Radix Tree insertion.

**Answer:**  
*(Reference: `src/radix.rs` lines 90-136)*

When inserting a token slice `remaining` into `current`:
1. Find if a child exists starting with `remaining[0]`.
2. If found, calculate the common prefix length (`common_len`) between `child.tokens` and `remaining`:
   ```rust
   let common_len = child.tokens.iter().zip(remaining.iter()).take_while(|(a, b)| a == b).count();
   ```
3. **Case 1: Exact / Full Edge Match (`common_len == child.tokens.len()`):**
   - The entire child slice matches. Recurse into this child with `&remaining[common_len..]`.
4. **Case 2: Divergence / Partial Match (`common_len < child.tokens.len()`):**
   - We must split the existing edge into three pieces:
     - **Split Node (Shared Prefix):** Holds `child.tokens[..common_len]`. It inherits worker IDs from both the existing branch and the new inserting worker.
     - **Existing Child (Truncated):** Holds `child.tokens[common_len..]`. It retains its original workers and existing children (`std::mem::take(&mut child.children)`).
     - **New Sibling Child:** Holds `remaining[common_len..]`, tagged only with the new `worker_id`.

```
Before Split:
(Root) ──────────► [ 1, 2, 3, 4 ] (Worker 1)

Insert [ 1, 2, 5, 6 ] (Worker 2):
(Root) ──────────► [ 1, 2 ] (Workers: {1, 2})
                     ├──► [ 3, 4 ] (Worker: {1})
                     └──► [ 5, 6 ] (Worker: {2})
```

---

### Q6: What is the time and space complexity of `insert` and `find_longest_prefix`?
> **Interviewer Intent:** Standard CS fundamentals applied to systems software.

**Answer:**  
Let:
- $L$ = length of prompt in tokens ($L \le 4096$ or $32\text{k}$)
- $B$ = maximum branching factor at any node (number of unique next tokens, bounded by vocabulary size $V \approx 150\text{k}$, but practically $< 100$ in common prompt prefixes)
- $K$ = number of downstream workers

- **Time Complexity:**
  - **`find_longest_prefix`:** $O(L)$ comparisons in the worst case. At each node, child lookup is $O(1)$ amortized via `HashMap<TokenId, RadixNode>`. Slices are compared via SIMD/vectorized equality checks. In practice, search executes in **< 15 microseconds**.
  - **`insert`:** $O(L)$ token comparisons. Splitting a vector is $O(M)$ where $M \le L$ is the slice length. Total insertion time is sub-30 microseconds.
- **Space Complexity:**
  - $O(U)$ where $U$ is the total number of unique prefix tokens registered across all active prompts. Memory consumption is minimal: 100,000 cached tokens consume $\approx 100{,}000 \times 4\text{ bytes} \approx 400\text{ KB}$ of raw token data, plus node metadata overhead ($\approx 10\text{--}20\text{ MB}$ total in RAM).

---

## 4. Tokenizer Pipeline: Why Token IDs vs. Strings

### Q7: Why do you tokenize the prompt into `u32` IDs in the proxy rather than matching UTF-8 string prefixes?
> **Interviewer Intent:** This is a classic trap question. It separates surface-level engineers from genuine MLSys engineers.

**Answer:**  
"Matching raw UTF-8 strings for prefix caching causes subtle, critical bugs due to **Byte-Pair Encoding (BPE) tokenization properties**:

1. **Token Boundary Ambiguity (Merge Rules):**
   - In BPE tokenizers, tokens are formed by merging frequent character pairs.
   - For example, the string `" apple"` might be a single token (ID: `15200`). But `" apples"` might tokenize into `[" apples"]` (ID: `38901`) or `[" apple", "s"]`.
   - Even worse, spaces at word boundaries merge with following words. If Prompt A ends with `"Hello "` and Prompt B continues `"Hello world"`, string prefix matching says they share `"Hello "`. But BPE tokenization might merge `" "` and `"w"` into `" w"`, completely shifting token IDs.
2. **GPU Prefix Cache Alignment:**
   - Inference engines (vLLM PagedAttention, SGLang) allocate KV cache in fixed-size blocks (typically **16 or 32 tokens**).
   - The GPU never sees text; it only checks if block hash $H = \text{hash}(\text{token\_id}_0, \dots, \text{token\_id}_{15})$ matches. If token IDs don't match exactly from position 0, the GPU has a 0% cache hit rate.
3. **Chat Template Formatting:**
   - OpenAI API calls send structured JSON: `[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]`.
   - In `src/tokenizer.rs`, we format messages with ChatML delimiters (`<|im_start|>system\n...<|im_end|>`). Tokenizing this exact sequence guarantees that our prefix IDs mirror the exact token stream ingested by the worker's LLM engine."

---

### Q8: Doesn't running tokenization on the proxy add unacceptable latency overhead?
> **Interviewer Intent:** Testing latency budget understanding.

**Answer:**  
"No. HuggingFace's Rust `tokenizers` library is heavily optimized:
- It uses multithreaded Rayon parallelism and SIMD byte matching.
- For a typical 1,500-token prompt, tokenization takes **35 to 80 microseconds (0.035 - 0.080 ms)** on a modern CPU.
- In contrast, a cold GPU prefill takes **150ms to 500ms**.
- Spending 0.05ms in the proxy to save 200ms on the GPU yields an efficiency gain of roughly **4,000x**.
- Furthermore, proxy tokenization is done *before* network socket dispatch, adding negligible jitter."

---

## 5. Concurrency, Async Runtime & Thread-Safety Invariants

### Q9: Explain your concurrency model. How do you prevent lock contention on the Radix Tree?
> **Interviewer Intent:** Probing Rust systems programming, multithreading, and synchronization primitives.

**Answer:**  
*(Reference: `src/main.rs` lines 71-126, `src/radix.rs`)*

1. **`parking_lot::RwLock` over `std::sync::RwLock`:**
   - `parking_lot` locks do not allocate OS mutex primitives (pthread mutex / Windows SRW locks); they use 1-word futexes.
   - They provide **writer-fairness** (preventing writer starvation) and have significantly lower uncontended and read-heavy acquisition latency.
2. **Read vs. Write Path Separation:**
   - **Read Path (Hot Path):** During incoming request routing (`chat_completions_handler`), we acquire a shared read lock:
     ```rust
     let match_result = state.tree.read().find_longest_prefix(&token_ids);
     ```
     Multiple concurrent requests read the Radix Tree simultaneously with **zero lock contention**. The read lock is held for only $\sim 10\text{--}15\mu\text{s}$ during traversal and dropped immediately before making downstream HTTP calls.
   - **Write Path (Background Async Task):** Updating the tree is decoupled from the client request path. When a worker is assigned a prompt, we spawn a non-blocking Tokio task:
     ```rust
     tokio::spawn(async move {
         tree_ref.write().insert(&tree_tokens, worker_id);
     });
     ```
     The write lock is acquired asynchronously in the background, never blocking client I/O or delaying token streaming.

---

### Q10: Why use `AtomicUsize` with `SeqCst` / `Relaxed` for tracking worker queues?
> **Interviewer Intent:** Testing low-level atomic memory ordering and lock-free concurrency.

**Answer:**  
*(Reference: `src/main.rs` lines 229-239, 301-306)*

- Every `WorkerNode` holds `active_requests: Arc<AtomicUsize>`.
- **Increment/Decrement:** When dispatching a request to a worker, we perform:
  ```rust
  target_worker.active_requests.fetch_add(1, Ordering::SeqCst);
  // ... downstream request executes ...
  active_counter.fetch_sub(1, Ordering::SeqCst);
  ```
  `Ordering::SeqCst` ensures immediate sequential consistency across all CPU cores, guaranteeing that two concurrent requests arriving at the same millisecond observe an accurate queue count.
- **Reading Queue Depth for Load Balancing:**
  In `select_least_loaded_worker`, we read with `Ordering::Relaxed`:
  ```rust
  w.active_requests.load(Ordering::Relaxed)
  ```
  Since routing choices don't require strict memory synchronization barriers (an off-by-one difference during an ultra-tight race does not violate memory safety), `Relaxed` avoids expensive CPU memory fence instructions.

---

## 6. Routing Policy: Cache Locality vs. Load Balancing

### Q11: What is the "Thundering Herd / Hotspot" problem in cache-affinity routing, and how do you mitigate it?
> **Interviewer Intent:** Testing distributed systems trade-offs. If one prompt prefix is very popular, won't one worker get overwhelmed while others sit idle?

**Answer:**  
"This is the fundamental tension between **Cache Locality** and **Load Balancing**.

If 1,000 concurrent requests share the exact same system prompt (e.g., a viral customer service bot):
- **Pure Cache Routing:** Sends 100% of requests to Worker 1. Worker 1's queue explodes, its GPU request queue saturates, and TTFT degrades because requests wait in line. Meanwhile, Worker 2 sits at 0% GPU utilization.
- **Pure Load Balancing (Round-Robin):** Distributes requests evenly, but destroys cache locality, forcing every worker to redundantly prefill.

**Our Hybrid Scoring Function:**
$$\text{Score}(W) = \alpha \cdot \frac{\text{Matched Prefix Tokens}}{\text{Total Prompt Tokens}} - \beta \cdot \text{Pending Requests}(W)$$

In our implementation (`src/main.rs`):
1. **Prefix Threshold (`min_prefix_match_ratio = 0.20`):** We only route for affinity if at least 20% of the prompt matches a cached prefix. Weak matches fall back to the least-loaded worker.
2. **Queue Penalty / Shedding:** If a worker's `active_requests` exceeds a queue ceiling, the penalty term $\beta \cdot \text{queue}$ overrides the cache affinity bonus $\alpha \cdot \text{match\_ratio}$.
3. **Cache Replication on Overload:** When Worker 1 is overloaded, routing the overflow request to Worker 2 causes Worker 2 to prefill the prompt *once*. The proxy's post-response background update registers the prefix for Worker 2 as well (`workers.insert(2)`). Now, subsequent requests can route to *either* Worker 1 or Worker 2 based on queue depth!"

---

## 7. Networking, Zero-Copy SSE Streaming & Backpressure

### Q12: How do you handle Server-Sent Events (SSE) streaming without buffering full responses in memory?
> **Interviewer Intent:** Assessing production backend networking skills, async I/O, and memory leak prevention.

**Answer:**  
*(Reference: `src/main.rs` lines 258-271)*

"In LLM generation, responses are streamed token-by-token over HTTP SSE (`text/event-stream`).
If a proxy collects all chunks in memory before replying (`res.bytes().await`), two critical failures occur:
1. **Destroys TTFT:** The user receives zero tokens until the entire 500-token completion finishes (30+ seconds).
2. **Memory Exhaustion:** Buffering concurrent 1,000 streams with large generations will blow up proxy heap memory.

**Our Implementation:**
We establish a **zero-copy asynchronous stream pipe**:
```rust
let stream = res.bytes_stream();
let body = Body::from_stream(stream);
(status, headers, body).into_response()
```
- `reqwest::Response::bytes_stream()` yields an asynchronous `Stream<Item = Result<Bytes, ...>>`.
- Axum wraps this directly in `axum::body::Body::from_stream()`.
- **Kernel-level TCP Flow:** As TCP packets arrive from Worker 1, Tokio forwards them directly down the client socket without deserializing or buffering them in application memory.
- **Backpressure Handling:** If a client on a mobile device has a slow network connection, TCP window flow control automatically slows down the proxy's read from the worker, preventing unbounded buffer growth in memory."

---

## 8. Empirical Benchmarks & Profiling Defense

### Q13: Walk me through your benchmark setup and numbers (`benchmarks/RESULTS.md`). How did you simulate the experiment?
> **Interviewer Intent:** Validating that your numbers are real, mathematically sound, and reproducible.

**Answer:**  
"To generate deterministic, reproducible benchmarks:
1. **Workload:** 15 multi-agent analytical queries simulating an M&A financial due diligence copilot. All 15 requests shared a standardized **1,500-token system prompt** detailing GAAP rules, but had unique user queries.
2. **Cluster Setup:** 2 downstream model worker endpoints (`Worker-1:8001`, `Worker-2:8002`) and 1 proxy instance (`Proxy:8000`).
3. **Comparison Baselines:**
   - **Baseline (Naive Round-Robin):** Alternating requests strictly: Request 1 $\to$ W1, Request 2 $\to$ W2, Request 3 $\to$ W1, etc.
   - **Prefix-Caching Proxy:** All traffic directed to the Rust reverse proxy.

**Results:**
| Metric | Naive Round-Robin | Prefix-Caching Proxy | Delta |
| :--- | :--- | :--- | :--- |
| **Cache Hit Rate** | 86.7% | **100.0%** | **+13.3%** |
| **Median TTFT (P50)** | 23.2 ms | **21.8 ms** | **-6.0%** |
| **Tail Latency (P95 TTFT)** | 287.8 ms | **33.7 ms** | **-88.3% tail drop** |
| **Average TTFT** | 53.2 ms | **29.4 ms** | **-44.7% faster** |
| **Proxy Routing Overhead** | N/A | **< 2.5 ms (P99)** | **Sub-100µs CPU time** |

**Why P95 Latency Dropped 88.3%:**
Under Round-Robin, alternating requests caused cold prefill cache misses on the alternating worker, causing severe tail-latency spikes ($\approx 287.8\text{ms}$). The Prefix-Caching Proxy pinned identical prefix queries to the warm worker, maintaining warm GPU cache residency and flat-lining tail latency at 33.7ms."

---

### Q14: Where does the 2.5ms proxy overhead come from? Break down the latency budget.
> **Interviewer Intent:** Probing systems profiling skills and microsecond-level latency breakdown.

**Answer:**  
"Profiling our request pipeline reveals where the time is spent:
1. **JSON Deserialization (Axum / Serde):** $\approx 150\text{--}250\mu\text{s}$ to parse the incoming HTTP body into `ChatCompletionRequest`.
2. **BPE Tokenization (`tokenizers`):** $\approx 50\text{--}90\mu\text{s}$ for 1,500 characters.
3. **Radix Tree Read Lock & LPM Traversal:** $\approx 10\text{--}25\mu\text{s}$.
4. **Proxy Internal Logic Total:** **$\approx 300\mu\text{s}$ (0.3ms)** of pure CPU time.
5. **Remaining $\approx 1.5\text{--}2.0\text{ms}$:** Local loopback TCP socket handshakes, kernel network stack transitions, and Tokio task scheduling between client $\to$ proxy $\to$ worker.

The routing engine itself accounts for less than 15% of the total 2.5ms overhead."

---

## 9. Failure Modes, Cache Invalidation & Distributed Scaling

### Q15: What happens when a worker GPU runs out of VRAM and evicts a cached prefix (e.g., via LRU)? How does your proxy stay in sync?
> **Interviewer Intent:** Senior/Staff-level question on distributed state drift and cache consistency.

**Answer:**  
"In production, inference engines like vLLM use **PagedAttention** with an LRU block eviction policy when KV memory pressure is high. If Worker 1 evicts a prefix, the proxy could suffer from **cache state drift** (thinking Worker 1 has it when it doesn't).

**Strategies to Solve This:**
1. **TTL and LRU in Proxy Radix Tree:**
   In `src/radix.rs`, each `RadixNode` already has `last_accessed: Instant`. A background cleanup loop evicts prefix nodes not accessed within a configurable TTL (e.g., 10 minutes) or enforces a max node cap matching worker VRAM limits.
2. **Asynchronous Feedback / Header Echoing:**
   Downstream engines (or custom middleware) return headers in the response:
   - `x-cache-hit-tokens: 1500`
   - `x-kv-cache-usage: 88%`
   If the proxy routed assuming a 1,500-token hit but the worker returns `x-cache-hit-tokens: 0`, the proxy immediately updates its Radix Tree branch to remove that `worker_id`.
3. **Explicit Eviction Webhooks / Prometheus Scraping:**
   vLLM exposes metrics via `/metrics`. The proxy can periodically poll worker KV cache usage and trigger pruning."

---

### Q16: What if a worker crashes or reboots?
> **Interviewer Intent:** Resiliency, fault tolerance, and circuit breakers.

**Answer:**  
1. **Immediate Failure Recovery:** If `state.client.post(&target_url).send().await` returns an I/O error or timeout, the proxy intercepts the `Err(err)` (lines 285-296), marks the worker as unhealthy, and transparently retries the request against the least-loaded healthy worker.
2. **Cache Invalidation on Restart:** When a worker crashes, its GPU VRAM is completely wiped. When the health check detects a worker restart (or incremented boot ID), the proxy purges that `worker_id` from all nodes in the Radix Tree in a single pass."

---

### Q17: How would you scale this proxy horizontally from 1 instance to 10 instances?
> **Interviewer Intent:** Distributed systems architecture at scale.

**Answer:**  
"If you run 10 proxy instances behind an AWS NLB, they each need to know which worker holds which cache.
Three architectures exist:

1. **Partitioned Routing (Worker Sharding):**
   - Partition workers into pools by model or tenant. Proxy 1 handles Tenant A; Proxy 2 handles Tenant B.
2. **Distributed Prefix Index via Redis / Key-Value Store:**
   - Instead of in-memory Radix Trees, store prefix hashes in Redis.
   - *Drawback:* Adds a 1-2ms network roundtrip to Redis on every request, eroding the latency savings.
3. **Decentralized Gossip / Consistent Hashing on Prefix Clusters (Recommended):**
   - Keep Radix Trees local in-memory on each proxy instance.
   - Use an L4 NLB with **Maglev or Rendezvous Hashing** on the client/system prompt hash so requests with the same system prompt usually land on the same proxy instance.
   - Proxies broadcast major prefix warm events to peer proxies over a lightweight UDP/gRPC gossip protocol."

---

## 10. Rapid-Fire Questions & Cheat Sheet

| Question | Short Answer / Punchline | Code Reference |
| :--- | :--- | :--- |
| **Why Rust?** | Zero-cost abstractions, memory safety without GC pauses, predictable sub-millisecond P99 tail latency. | `Cargo.toml` |
| **What web framework?** | **Axum** built on **Tokio** and **Tower**. Non-blocking async I/O with work-stealing thread pools. | `src/main.rs:122-127` |
| **Why not standard Trie?** | Trie allocates 1 node per token. Radix compresses sequential edges, saving 95% pointer overhead and CPU cache misses. | `src/radix.rs:8-18` |
| **Why `parking_lot`?** | Uses 1-byte futexes, writer starvation prevention, significantly lower overhead than `std::sync::RwLock`. | `src/main.rs:12` |
| **Why not regex or string prefix?** | BPE token merge rules mean string prefixes do not map 1:1 to token ID prefixes. GPU cache keys are token IDs. | `src/tokenizer.rs:27-30` |
| **How is streaming piped?** | Zero-copy byte stream from `reqwest::Response::bytes_stream()` directly into `axum::body::Body`. Zero memory buffering. | `src/main.rs:268-271` |
| **Empirical TTFT reduction?** | **-88.3% P95 TTFT drop** (287.8ms down to 33.7ms) on 1,500-token prompts. | `benchmarks/RESULTS.md` |
| **Proxy latency overhead?** | **< 2.5 ms P99 total**, with only $\approx 300\mu\text{s}$ CPU routing time. | `src/main.rs:223-227` |
