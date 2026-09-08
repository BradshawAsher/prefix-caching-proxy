# Prefix-Caching KV Router: Conversational & Practical Interview Prep

> **Goal**: Plain-English, conversational answers for real human interviewers. No academic posturing—just clear, engaging engineering stories that demonstrate technical ownership and practical problem-solving.

---

## 1. "Explain this project like I'm 5 (or a non-technical manager)"

> *"When an AI like ChatGPT or Claude reads a prompt, it's like a person reading a 50-page legal contract before answering your question. Doing that reading—called the 'prefill stage'—takes most of the time and burns huge GPU power.*
> 
> *In enterprise systems, hundreds of users ask questions using the **exact same 50-page document or system prompt**.*
> 
> *Normally, load balancers (like AWS ALB or NGINX) are dumb: they send User 1 to GPU A, and User 2 to GPU B. So both GPUs waste time reading the exact same 50 pages from scratch.*
> 
> *My project is an intelligent traffic cop in Rust. It reads the first few sentences of an incoming request, checks which GPU already read that document in its memory, and routes the request directly to that GPU. The GPU skips the entire reading phase and responds almost instantly—slashing tail latency by **88%**."*

---

## 2. "Tell me about a difficult bug you ran into and how you fixed it"

Have **two distinct bugs** ready to share:

### Bug #1: The "Token Boundary Mismatch" Bug (MLSys & NLP Mechanics)
* **Situation**: "Early in the project, we were matching prompt prefixes using raw text strings (like `string.starts_with()`). But during our benchmarks, our cache hit rate mysteriously dropped by 30% on prompts that looked virtually identical."
* **Task**: "I had to find out why the router was reporting cache hits, but the backend GPUs were still doing cold recalculations."
* **Action**: "I dug into how Large Language Models actually process text. GPUs don't read words or characters; they read **Token IDs** produced by a Byte-Pair Encoding (BPE) tokenizer. I discovered that slight changes in whitespace, capitalization, or contractions (like `'I am'` versus `'I\'m'`) alter the exact token boundaries. String matching thought it was an 80% match, but on the GPU, the token numbers were completely different after the first contraction, causing the GPU's KV cache to miss entirely!"
* **Result**: "I brought the HuggingFace BPE tokenizer directly into our Rust proxy. Before querying our Radix tree, we convert the prompt into raw integer token IDs in microseconds. Matching by token IDs instead of raw strings brought our cache hit rate to a verified 100% on shared prompts."

---

### Bug #2: The "Overloaded Worker" / Cache Hotspotting Bug (Load Balancing)
* **Situation**: "When we started stress-testing the router with high traffic, one of our GPUs suddenly spiked to 100% queue utilization and requests began timing out, while the other GPU sat nearly idle."
* **Task**: "I had to diagnose why traffic wasn't distributing properly across the cluster."
* **Action**: "I realized our router was blindly routing requests to whatever worker had *any* cache match—even if it was tiny! If Worker 1 had cached just 10 tokens out of a 1,000-token prompt (a 1% match), the router would send it there to save 1% of compute, even though Worker 1 was already drowning in traffic! Saving 2 milliseconds of GPU compute caused 500 milliseconds of queueing delay."
* **Result**: "I introduced a minimum cache affinity threshold (`min_prefix_match_ratio: 0.20`). If a match is less than 20%, the router decides the compute savings aren't worth overloading a busy worker, and defaults to **least-connections load balancing**. This completely eliminated worker hotspotting and smoothed out our latency curves."

---

## 3. "What part of this project are you most proud of?"

### Proud Story #1: Slashing Tail Latency by 88% on Real LLM Engines (vLLM)
> *"I'm most proud of seeing the real-world latency impact.*
> 
> *In distributed systems, average latency looks fine, but the **P95 tail latency** is what hurts users. In multi-turn chat and agentic workflows, users often wait 300 to 500 milliseconds just for the first token to appear.*
> 
> *When we hooked our Rust router up to real vLLM workers and ran benchmark workloads, P95 Time-To-First-Token dropped from **287ms down to 33ms**—an **88.3% drop**. Seeing the GPU completely bypass the expensive prefill phase and stream the first word instantly felt like magic."*

---

### Proud Story #2: Designing the In-Memory Compressed Radix Tree
> *"I'm really proud of our custom Radix tree implementation in Rust.*
> 
> *A standard Trie creates a new node for every single token. If a prompt has 2,000 tokens, you'd have 2,000 heap-allocated nodes, which wastes tons of RAM on pointers and causes CPU cache misses.*
> 
> *I built a compressed Radix Tree that collapses non-branching token sequences into contiguous slices. It handles dynamic branch splitting on the fly when new prompts branch off. That cut our memory overhead by ~95% and allowed our prefix lookup to run in under 10 microseconds."*

---

## 4. Common "Easy English" Questions & Natural Answers

### "Why did you use Rust instead of Python?"
* **Conversational Answer**:
  > *"Most of the AI world uses Python, but a reverse proxy sits on the critical network path of every single request.*
  > 
  > *If the proxy adds 50 milliseconds of overhead because of Python's Global Interpreter Lock (GIL) or garbage collection, you've defeated the whole purpose of caching.*
  > 
  > *In Rust using Tokio and Axum, our total proxy routing overhead is **under 2.5 milliseconds**. We can parse HTTP requests, tokenize text, search the tree, and stream responses with zero garbage collection pauses."*

---

### "What is the difference between a normal cache and a KV cache?"
* **Conversational Answer**:
  > *"A normal web cache (like Redis or Cloudflare) saves the **final response** (e.g. the final answer or web page).*
  > 
  > *A **KV cache** is different: it saves the **intermediate mathematical memory** of the neural network's attention layers for previous tokens.*
  > 
  > *This means two users can have different questions at the end, but if they share the first 2,000 tokens, the neural network doesn't have to re-read those first 2,000 tokens."*

---

### "If you had another month to work on this, what would you build next?"
* **Conversational Answer**:
  > *"Two features:*
  > 1. *Right now, our Radix tree lives in memory on a single proxy node. If you have multiple proxy instances behind a DNS load balancer, I'd implement a lightweight gossip protocol or Redis sync so all proxies share prefix awareness.*
  > 2. *I'd add **GPU VRAM telemetry**: dynamically querying vLLM workers to know when they evict KV blocks due to memory pressure, so our Radix tree can automatically prune cold branches."*
