# The Relay: Project Engineering Log & Results Write-up

This write-up covers what I built, the empirical measurements from real Groq probes and local stand-in load runs, how the pipeline handles rate limits and latency under concurrency, and what broke along the way. All numbers in the tables below are computed directly from [`results.csv`](results.csv) (213 rows) and the step timer logs in [`raw/timing_breakdown.jsonl`](raw/timing_breakdown.jsonl).

---

## 1. What I Built

### System Architecture
- **Multi-Step Pipeline** ([`src/pipeline.py`](src/pipeline.py)):
  - Built an end-to-end question-answering pipeline grounded exclusively in 3 specific arXiv papers:
    1. *Attention Is All You Need* (`1706.03762v7`) — 15 pages, 24 sections
    2. *Moshi: a speech-text foundation model for real-time dialogue* (`2410.00037v1`) — 66 pages, 36 sections
    3. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (`2507.07955v2`) — 39 pages, 24 sections
  - Total corpus: 120 pages, 55,815 words, 362,929 characters extracted via [`src/ingest.py`](src/ingest.py).
- **Five Discrete Pipeline Steps** (with independent latency timers):
  1. `step_1_understand`: Query reformulation, synonym expansion, and technical concept extraction.
  2. `step_2_retrieve`: Non-blocking tool call executing BM25 lexical search, Dense BGE (`BAAI/bge-small-en-v1.5`), Hybrid Reciprocal Rank Fusion (RRF), or Cross-Encoder re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
  3. `step_3_decide` (Branch/Loop): Evaluates evidence sufficiency; loops with rewritten query (up to 2 times) or branches to truthful refusal.
  4. `step_4_draft`: Synthesizes cited answer strictly from retrieved passages.
  5. `step_5_verify` (Check & Redraft): Verifies factual claims and page citations against raw source chunks; triggers self-correction redraft if hallucination or unsupported claim is detected; skippable under degraded mode.
- **Core Technology Stack**:
  - Python 3.12, `asyncio`, `httpx` (streaming SSE client), `FastAPI` + `uvicorn` (stand-in server), `pymupdf` (PDF layout extraction), `rank_bm25`, `sentence-transformers`, `pandas`.
  - **No High-Level Agent Wrappers**: I avoided LangChain, LlamaIndex, or AutoGen so that every queue, retry loop, step timer, and cancellation boundary is directly visible and debuggable in plain Python.
- **Grounding & Citation Rules**:
  - Every answer must carry page-anchored citations: `[Paper Title, Sec X, p. N]`.
  - Truthful refusal: When evidence is missing or the question is unanswerable, the system outputs `"The papers do not say."` It never hallucinates or extrapolates to appear helpful.

---

## 2. Real Service and Stand-In

### Real Groq Service Probes (`qwen/qwen3.8-27b`)
- Probed using [`src/probe_groq.py`](src/probe_groq.py); results saved in [`raw/probe_sequential.json`](raw/probe_sequential.json), [`raw/probe_concurrent.json`](raw/probe_concurrent.json), [`raw/probe_429.json`](raw/probe_429.json), and [`raw/probe_recovery_refill_curve.json`](raw/probe_recovery_refill_curve.json):
  - **Median Internal TTFT**: `801–887 ms` on multi-thousand token prompts.
  - **Generation Speed**: `~300 tokens/sec`.
  - **Concurrency Lower Bound**: 3 concurrent in-flight requests finished in `470.4 ms` with zero queuing penalty (lower bound $\ge 3$).
  - **Rate Limit Headers**: Gateway header reports `x-ratelimit-limit-requests: 1000` (RPD) and `x-ratelimit-limit-tokens: 8000`.
  - **Enforced Token Cap**: The HTTP 429 error response body reveals the true enforced limit: `Limit 7000, Used ..., Requested ...`. The binding limit is **7,000 Input Tokens Per Minute (ITPM)**.
  - **Token Refill Dynamics**: Measured continuously in [`raw/probe_recovery_refill_curve.json`](raw/probe_recovery_refill_curve.json) across multiple intervals ($t+8s$, $t+20s$). The bucket refills at $116.7\text{ to }122.3\text{ tokens/sec}$, precisely matching continuous refill ($\frac{7,000}{60\text{s}} = 116.67\text{ tok/s}$).
  - **Single-Request Oversize Limit**: An 8,512-token prompt returned **HTTP 413 Payload Too Large** ([`raw/probe_429_deliberate.json`](raw/probe_429_deliberate.json)), proving that single-request context overflow is rejected with 413, whereas cumulative rate exhaustion produces 429.

### Calibrated Stand-In Server ([`standin/server.py`](standin/server.py))
- **TTFT Calibration with Prefill Penalty**: Base TTFT sampled from log-normal distribution ($\mu = 5.90, \sigma = 0.23$) with an added linear prompt prefill term ($0.04\text{ ms} \times \text{tokens\_in}$):
  $$\text{TTFT} = \text{base\_ms} + 0.04 \times \text{tokens\_in}$$
  This reproduces realistic 800–1,200 ms latency when ingesting 2,000–3,000 token retrieved contexts.
- **Streaming Emulation**: Emulates chunk-by-chunk SSE streaming at 300 tokens/sec.
- **Continuous Token Refill & Daily Rolling Pool**:
  - Implements a continuous token bucket recharging at $\frac{7,000}{60} = 116.67\text{ tokens/sec}$ with an enforced 7,000 ITPM cap.
  - Implements a rolling daily pool of 1,000 requests recharging at 1 request every 86.4 seconds ($86,400\text{s} / 1,000$).
- **Single-Request Guard**: Prompts $> 8,000$ tokens immediately return HTTP 413.
- **Fault Injection**: Hooks for `error_rate` (500s), `timeout_rate` (hangs), and `slow_rate` (tail latency).
- **Known Differences from Real Service**:
  1. *Compute Footprint*: Mock runs locally with $<1\%$ CPU footprint; does not emulate GPU memory contention or thermal throttling.
  2. *Network Jitter*: Localhost socket communication eliminates cloud load balancer routing delays and WAN packet retransmissions.
  3. *Token Tokenizer*: Stand-in uses whitespace tokenization calibrated against GPT/Qwen BPE tokenizers ($\approx 1.3 \times \text{words}$).

---

## 3. Benchmark Results (Tables 1, 2a, 2b, 3)

All numbers below are generated directly from [`results.csv`](results.csv) by [`analysis/tables.py`](analysis/tables.py) and verified by [`analysis/verify_tables.py`](analysis/verify_tables.py):

### Table 1: Real Model Service & Rate Limit Parameters (`qwen/qwen3.8-27b`)

| Parameter | Value | Confidence | How Found / Observed |
| :--- | :--- | :---: | :--- |
| **Model ID** | `qwen/qwen3.8-27b` | High | Direct API query to `/v1/models` & successful test generation |
| **Daily Request Pool (RPD)** | 1,000 requests | High | Observed directly in `x-ratelimit-limit-requests: 1000` header |
| **Request Refill Rate** | 1 req / 1m26.4s (rolling) | High | Observed in `x-ratelimit-reset-requests: 1m26.4s` ($86,400\text{s} / 1,000$) |
| **Requests / Minute (RPM)** | 30 req/min | Medium | Official Groq Docs (Nov 2026); unobserved as response header |
| **Input Tokens / Minute (ITPM)** | 7,000 ITPM (enforced) | High | Observed in 429 error payload body (`Limit 7000`); gateway header reports 8000 |
| **Token Refill Rate** | ~116.7 tokens/sec (continuous) | High | Empirically measured across intervals in `raw/probe_recovery_refill_curve.json` |
| **Tokens / Day (TPD)** | 200,000 tokens/day (enforced) | High | Observed directly in live HTTP 429 error body (`Limit 200000, Used 199399`); official Groq docs claim 500,000 |
| **Single-Request Limit** | HTTP 413 Payload Too Large | High | Observed in `raw/probe_429_deliberate.json` on 8,512 token prompt |
| **Cumulative Limit Behavior** | HTTP 429 Too Many Requests | High | Observed in `raw/probe_429.json` when 60s sliding window exceeded 7,000 tokens |
| **Concurrency Lower Bound** | >= 3 in-flight | High | Probed with 3 parallel calls in Phase 4 (finished in 470.4ms); upper cliff unmeasured |

---

### Table 2a: Latency Distribution Across Traffic Scenarios

| Scenario | Offered Rate | Duration (s) | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99 (ms) | Total p50 (ms) | Total p95 (ms) | Total p99 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`steady`** | 2.8 req/m | 325.9s | 1331.4 | 1575.9 | 1606.6 | 3062.1 | 5941.0 | 6457.4 |
| **`burst`** | 72.0 req/m | 20.8s | 1212.4 | 1212.4 | 1212.4 | 8100.0 | 8345.9 | 8466.6 |
| **`sustained_overload`** | 39.9 req/m | 30.1s | 0.0 | 0.0 | 0.0 | 8100.0 | 8374.8 | 8415.5 |
| **`limit_halved`** | 22.2 req/m | 27.1s | 2529.8 | 3866.7 | 3985.5 | 8100.0 | 8305.4 | 8439.9 |
| **`daily_exhausted`** | 30000.0 req/m | 0.0s | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

---

### Table 2b: Reliability, Queue & Cost Metrics Across Traffic Scenarios

| Scenario | Success % | Rate Limited % | Timeout/Drop % | Retries / Req | Peak Queue | Tokens / Ans | Cost / 1k Ans (Commercial) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`steady`** | 100.0% | 0.0% | 0.0% | 0.00 | 1 | 1806.9 | $0.0915 |
| **`burst`** | 4.0% | 0.0% | 96.0% | 0.00 | 20 | 1082.0 | $0.0549 |
| **`sustained_overload`** | 0.0% | 0.0% | 100.0% | 0.00 | 20 | 11.0 | $0.0006 |
| **`limit_halved`** | 30.0% | 0.0% | 70.0% | 0.10 | 4 | 1104.3 | $0.0561 |
| **`daily_exhausted`** | 0.0% | 100.0% | 0.0% | 0.00 | 0 | 0.0 | $0.0000 |

> *Note on Cost*: Free tier actual cost is **$0.00** (Groq terms). Commercial costs are calculated at list price ($0.05/1M prompt tokens, $0.08/1M completion tokens).

---

### Table 3: Retrieval Setting Quality, Token Cost & Latency Comparison

| Retrieval Setting | Configuration | Quality % | Tokens / Ans | TTFT p50 (ms) | Total p50 (ms) | Low-Limit / Degraded Fit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Setting A (`bm25_k3_c256`)** | 256 tokens × top-3 (BM25 Lexical) | **56.2%** | 2944.2 | 685.3 ms | 2501.3 ms | **Yes (Fast / Cheap)** |
| **Setting B (`dense_k3_c256`)** | 256 tokens × top-3 (Dense BGE Semantic) | **68.8%** | 3139.1 | 592.3 ms | 2549.6 ms | **Yes (Fast / Cheap)** |
| **Setting C (`dense_k3_c512`)** | 512 tokens × top-3 (Dense BGE Context) | **68.8%** | 4178.2 | 705.1 ms | 2473.2 ms | **No (High tokens; dilutes tables)** |

> *Notes on Table 3*:
> - All 48 benchmark runs (16 per setting) completed successfully (`status=ok`) by pacing queries to respect Groq's remaining token headroom ($\ge 4,500$ tokens) before dispatching the next request.
> - **Sample Size ($N=16$)**: Setting B answered **11/16** correctly vs Setting A's **9/16** (+12.6 percentage points, i.e. 11/16 vs 9/16: a two-question gap, too small to call a real difference). Their 95% Wilson score confidence intervals overlap heavily ([44.4%, 85.8%] for B vs [33.2%, 76.9%] for A; McNemar's exact test $p = 0.50$). This cannot be claimed as a statistically proven difference; rather, it highlights qualitative retrieval differences on specific multi-concept queries.
> - **Qualitative Breakdown (`q01` and `q10`)**: Dense BGE retrieved relevant passages for two questions where BM25 failed on vocabulary mismatch: `q01` (base Transformer parameters, where dimensions are split across paragraphs) and `q10` (Moshi Inner Monologue, phrased with descriptive concepts lacking verbatim query keywords). On the remaining 14 of 16 questions, BM25 and Dense BGE produced identical outcomes (including all 5 unanswerable questions truthfully refused).
> - **Chunk Inefficiency (512 vs 256 tokens)**: Setting C consumed **+33.1% more tokens** (4,178.2 vs 3,139.1 tokens/answer) and achieved **11/16 (68.8%)**, tying Setting B in overall accuracy. Individual question analysis reveals trade-offs: 512-token chunks gained `q11` (Moshi text tokenizer description) but lost `q10` (Inner Monologue, where extra narrative diluted the model's extraction focus). Larger chunks increased token burn rate by one-third without delivering net accuracy gains.
> - **Truthful Refusal**: All 5 near-miss unanswerable questions (`q12`–`q16`) were truthfully refused across all 3 settings without hardcoded keyword shortcuts.

---

## 4. Where the Time Goes (Slow Request Latency Decomposition)

### Case Study: Real Slow Request `eval_bm25_k3_c256_q02` (Source: `real`, Scenario: `retrieval_sweep`)
- **Total Latency (`total_ms`)**: **4,794.0 ms**
- **Internal First Token Time (`first_token_ms`)**: **1,261.3 ms**
- **Tokens In**: 4,214 | **Tokens Out**: 160
- **Retries (`retries`)**: 0
- **Retrieval Setting**: `bm25_k3_c256` | **Answer Grade**: `correct`

#### 1. Waiting vs Working vs Retrying Decomposition (from `raw/timing_breakdown.jsonl`)
- **Waiting (Queue Wait)**: **0.0 ms** (**0.0%**) — Dispatched immediately without queue contention.
- **Working (Inference & Retrieval)**: **4,793.9 ms** (**100.0%**) across 5 sequential stages:
  - **Step 1 (Query Understanding & Deconstruction)**: `928.3 ms` (**19.4%**) — LLM decomposes question into targeted sub-queries.
  - **Step 2 (Passage Retrieval & Scoring)**: `2.7 ms` (**0.1%**) — BM25 top-3 chunk retrieval executed via worker thread (`asyncio.to_thread`).
  - **Step 3 (Context Sufficiency Decision)**: `947.4 ms` (**19.8%**) — LLM evaluates retrieved passages for answerability.
  - **Step 4 (Draft Generation)**: `1,404.8 ms` (**29.3%**) — LLM synthesizes structured response with bracketed citations.
  - **Step 5 (Self-Correction Verification Gate)**: `1,510.8 ms` (**31.5%**) — LLM audits draft claims against source chunk text.
- **Retrying**: **0.0 ms** (**0.0%**) — Verification passed on initial attempt.

#### 2. Pre- vs Post-First-Token Delay & User-Visible TTFT
- **Pre-First-Token Delay (Internal TTFT)**: `1,261.3 ms` (**26.3%** of total duration)
- **Post-First-Token Delay (`total_ms - first_token_ms`)**: `3,532.7 ms` (**73.7%** of total duration)
- **User-Visible TTFT**: **4,794.0 ms** (**100.0%** of total duration)

> **Why I Chose Not to Stream Draft Tokens**:
> - In an agentic pipeline, `first_token_ms` measures only the first token emitted by Step 1. The remaining 73.7% of total time is spent executing the remaining 4 sequential reasoning passes.
> - I deliberately chose **not** to stream draft tokens from Step 4 before Step 5 verification finishes. In a voice or technical assistant, streaming an unverified draft means the user reads or hears a potential hallucination before the verification gate can catch it. Holding output until verification adds perceived initial latency, but ensures that whatever text is presented to the user is verified and cited.

---

## 5. When a Limit is Hit

### The Real Capacity Knee: Tokens, Not Requests
- Groq advertises a **30 RPM** rate limit, which might suggest a system can handle ~30 requests/minute.
- However, each verified answer requires 4–5 sequential LLM passes consuming $\approx 2,400\text{--}2,650$ input tokens.
- Under Groq's enforced **7,000 ITPM limit**, sustainable continuous throughput is strictly:
  $$\text{Capacity} = \frac{7,000\text{ ITPM}}{2,500\text{ tokens/answer}} \approx 2.8\text{ answers/minute}$$
- In the stand-in capacity curve sweep ([`data/capacity_curve_summary.json`](data/capacity_curve_summary.json)):
  - At **1.0 to 2.5 APM**, token inflow rises from 2,354.7 to 5,489.8 tok/min, with 100% success and p50 latency under 2.3s.
  - At **3.0 APM** ($N=10$, 20s interval), token inflow reaches **6,473.1 tok/min** (**92.5%** of the 7,000 ITPM cap).
  - At **4.0 APM** (15s interval), token inflow reaches **8,347.7 tok/min** (exceeding the 7,000 cap). The system hits rate-limit backoff, pushing p95 latency from 2.99s to **5.83s** (`token_limit_bound: true`).
  - At **6.0 APM** (10s interval), offered load (9,151.4 tok/min) overwhelms the bucket: success drops to **75.0%**, effective throughput plateaus at **4.60 APM**, and requests encounter 429 backoff and 8.0s deadline timeouts.
- **Real Groq 3.0 APM Validation** ([`data/capacity_real_groq_check.json`](data/capacity_real_groq_check.json)):
  To test this limit on the live service, I sent 4 consecutive questions at 3.0 APM (20s interval). By request 3, `x-ratelimit-remaining-tokens` dropped from full down to **1,283 tokens**. At 20s intervals, Groq only refills $\sim 2,334$ tokens while complex multi-step queries consume $\sim 4,000$ tokens, creating an unsustainable net deficit of $\sim 1,700$ tokens per turn.
- **Token capacity, not request count, is the binding operational constraint.** This elevates retrieval chunk size from an internal tuning knob into an external system capacity lever.

### How This Library Misleads a Naive Retriever (3 Real Failure Modes)
Cataloged in [`data/papers/extraction_report.json`](data/papers/extraction_report.json):
1. **Linearized Table Grid Collapse (14 tables across corpus)**:
   - *Example*: Transformer Table 2 BLEU scores (Question `q06`). PyMuPDF linearizes 2D table columns into continuous 1D text (`28.4`, `41.8`, `2.3 · 1019`), stripping column headers. Naive retrievers pull numbers without labels, causing hallucinated language mappings or false refusals (`refused_wrongly`).
2. **Mangled Multi-Line Equations & Formatting (3 major equation clusters)**:
   - *Example*: Attention scaling factor $\frac{1}{\sqrt{d_k}}$ (Question `q15`). Fractions split across line breaks (`1 \n √dk`) and superscripts flatten ($10^{20} \to 1020$). Lexical tokenizers fail to match queries due to character breaks.
3. **Diagram Vector Glyph Stacking (13 clusters, max run 43 chars)**:
   - *Example*: Dynamic Chunking Figure 1 (Question `q03`). Vector diagrams extract as vertical character runs (`n \n d \n o \n f \n t \n o \n k \n e \n n \n s`) polluted with disjoint glyphs (`!`, `/0`, `ℳ`). These chunks pollute embedding space, causing dense retrievers to rank noisy diagram text above actual methodology paragraphs.

---

## 6. Retries & Resilience Architecture

Implemented in [`src/resilience.py`](src/resilience.py):

### Resilience Design
1. **Exponential Backoff with Full Jitter**:
   - AWS Full Jitter: $\text{delay} = \text{Uniform}(0, \min(4.0s, 0.5s \times 2^{\text{attempt}}))$.
   - Obeys upstream `Retry-After` headers, adding minor random jitter ($+50\text{--}250\text{ms}$) to prevent synchronization storms.
2. **Safety Rules & Global Retry Budget**:
   - Only idempotent status codes `{429, 502, 503, 504, 408}` are retried. Non-retryable client errors (`400, 401, 403, 404, 422`) fail fast. Max 3 retries per request.
   - Retries capped at **25%** of lifetime requests seen.
3. **Per-Request Deadline (8.0s Justification)**:
   - The 8.0s deadline is derived directly from the pipeline's attempt-backoff-retry runway:
     $$\text{Runway} = \text{Attempt 1 }(2.0\text{s}) + \text{Backoff Wait }(1.5\text{s}) + \text{Attempt 2 Retry }(2.0\text{s}) + \text{Network Buffer }(2.5\text{s}) = 8.0\text{s}$$
   - Any request waiting in queue for $>3.0\text{s}$ cannot complete two attempts before 8.0s. Inter-step `check_deadline()` assertions drop doomed requests immediately to prevent wasting GPU quota.
4. **Bounded Queue & Load Shedding**:
   - Queue capacity: 20 requests. At $\ge 75\%$ capacity (15 requests), incoming requests automatically switch to **Degraded Mode**. At 100% capacity (20 requests), incoming requests are **shed immediately** (`retrieval_setting="shed"`) with 0ms delay.

### Retry Amplification Experiment ([`raw/retry_amplification_demo.json`](raw/retry_amplification_demo.json))
Tested with 20 concurrent requests offered against a strict 10 RPM server limit:
- **Blind Retries (No Jitter, No Budget, Infinite Deadline)**: 20 offered requests hammered the server with **50 HTTP calls** (**2.5x amplification**), causing cascading 429 lockouts and a **50.0% failure rate**.
- **Resilient Retries (Deadline-capped, Jittered Backoff, 25% Budget)**: Controlled total calls to **20** (**60.0% traffic reduction**), preventing server collapse and recovering cleanly.

---

## 7. What the Person Sees

The system delivers 4 distinct, transparent user experiences:
1. **Normal Cited Response**:
   `"In the base Transformer model, d_model = 512 and h = 8 [Attention Is All You Need, Sec 3.2.2 Multi-Head Attention, p. 5]."`
2. **Truthful Refusal (Unsupported Questions)**:
   `"The papers do not say."` (Maintained strictly even in degraded mode).
3. **Daily Budget Exhaustion Notification**:
   > *"The Relay research assistant has reached its daily free-tier computation budget (1,000 requests / 200,000 tokens). To ensure strict cost controls, no further automated calls can be made today. Quota refills continuously at 1 request every 86.4 seconds. Please try again later."*
4. **Overload Shedding Notification**:
   > *"Server is currently overloaded. Request was shed to protect system stability."*

---

## 8. Unsure / Left Out / Learned

### Deliberate Omissions & Trade-Offs
- **OCR / Vision Table Parsing**: Linearized table collapse was mitigated by isolating tables into 256-token chunks rather than introducing heavy multimodal vision models (e.g. Nougat, Surya). This preserved lightweight, sub-second local ingestion.
- **Speculative Query Pre-fetching**: I avoided speculative parallel retrieval on rewritten queries to preserve our strict 7,000 ITPM rate limit runway.

### Core Engineering Learnings
1. **The Compounding Penalty of Multi-Step Pipelines**:
   In single-turn generation, latency is just network time plus token emission. In verified RAG, latency compounds across 4–5 sequential reasoning steps. A modest 300 ms slowdown per step accumulates into a 1.5 s increase per turn, turning queuing contention into instant deadline timeouts.
2. **Token Limits Govern Throughput, Not RPM**:
   Free-tier LLM rate limits cannot be modeled by request counts alone. Systems must track cumulative token consumption and scale retrieval context down to maintain operational throughput.
3. **A/B Isolation of Steady Workload & Event-Loop Health**:
   I ran a controlled A/B test ([`data/ab_steady_isolation_summary.json`](data/ab_steady_isolation_summary.json)) holding offered rate ($2.6\text{ req/min}$, 23.08s interval) and request count ($N=15$) constant:
   - *Pre-fix Architecture (Synchronous CPU retrieval on event loop + lazy model loading)*: Experienced a **5,791.2 ms maximum event loop lag** during cold-start model initialization and synchronous retrieval passes. This pushed p95 total latency to **7,842.1 ms** (on the brink of the 8.0s deadline) and produced an initial timeout (93.3% success).
   - *Post-fix Architecture (`asyncio.to_thread` + preloaded singleton models)*: Reduced maximum event loop lag by **81.3%** down to **1,084.1 ms**, cut p95 total latency from 7,842 ms down to **5,941.0 ms**, and achieved **100.0% success** across all 15 requests.
   - *Root Cause of Original 53% Failure*: Auditing the git history of the original steady run (`commit 795370c`) revealed that requests were arriving with an 8.0s/clustered interval ($\sim 9.7\text{ req/min}$), generating $\sim 25,000\text{ tokens/min}$ against the 7,000 ITPM cap. The original collapse was therefore **both token bucket exhaustion under rapid arrivals AND event-loop blocking during cold-start/retrieval**. Isolating the rate at 2.6 req/min removed token exhaustion, allowing us to isolate and prove the 1,900 ms p95 latency reduction delivered by the async fix.
4. **Chunk Inefficiency & Dilution (512 vs 256 tokens)**:
   In our 3-setting real Groq benchmark, doubling chunk size from 256 to 512 tokens while keeping the dense BGE retriever constant (`Setting C` vs `Setting B`) caused a **+33.1% surge in prompt tokens** (4,178.2 vs 3,139.1 tokens/answer) with zero net accuracy gain (both tied at 11/16, 68.8%). While 512-token chunks helped on `q11`, it caused extraction dilution on `q10`, proving that larger chunks burn daily quotas significantly faster without improving overall precision.

---

## 9. Vaani Reflections: Conversational Interfaces & Latency Boundaries

In exploring conversational voice dialogue modeling (as exemplified by the *Moshi* paper included in our library), the interaction between latency boundaries and verified grounding presents a fundamental systems challenge:
- **The Human Turn-Taking Boundary vs. Multi-Step Verification**: Human conversational turn-taking operates on a strict **200–300 ms boundary**. The *Moshi* foundation model achieves real-time spoken dialogue by streaming 12.5 Hz discrete audio tokens directly from an asymmetric multi-stream transformer. In contrast, an inspectable, hallucination-free RAG assistant requires query understanding, vector retrieval, evidence sufficiency filtering, draft generation, and post-generation verification—consuming **2,400–2,700 ms** even on ultra-low-latency LPUs.
- **Why Naive Voice Streaming Fails**: In audio interfaces, you cannot un-say words. If an agent naively streams intermediate drafting tokens as spoken audio and the subsequent verification gate catches a factual hallucination or missing citation, the system cannot retract what the user has already heard.
- **The Voice-First RAG Architecture**:
  1. *Immediate Acoustic Acknowledgment (0–250 ms)*: Emitting conversational fillers (*"Checking the papers..."*, *"Looking at Table 2..."*) or prosodic backchannels preserves the conversational floor without violating turn-taking expectations.
  2. *Asynchronous Intent Streaming*: Running an intent classification model directly on incoming audio frames while retrieval executes in background worker threads.
  3. *Atomic Verified Payload Delivery*: Holding the synthesis stream until Step 5 verification passes, ensuring every spoken technical fact is 100% grounded in source literature.

---

## 10. How I Worked with the Assistant

### Pair-Programming Dynamic & Critical Engineering Catches
Throughout this project, I used the AI assistant as a rapid implementation accelerator while maintaining absolute empirical skepticism, architectural control, and code-level verification:

1. **Catching Table 1 Hallucinations & Header Mismatches**:
   When the assistant initially populated Table 1, it claimed Groq enforced **30 RPM observed in response headers** and an **8,000 TPM limit**, citing an **18.0s recovery time** from `probe_recovery.json`.
   Upon auditing the raw HTTP headers in `raw/probe_sequential.json`, I caught multiple severe hallucinations:
   - Every response header stated `x-ratelimit-limit-requests: 1000`, not 30. The 30 RPM was an unobserved figure copied from online documentation.
   - The reset countdown header (`x-ratelimit-reset-requests: 1m26.4s`) exactly equaled $86,400\text{s} / 1,000$, proving a continuous rolling daily refill rather than a static midnight reset.
   - The HTTP 429 response body explicitly cited an enforced **7,000 ITPM limit**, contradicting the gateway's 8,000 TPM header.
   - `probe_recovery.json` was timestamped at 20:48:14—*before* the 429 probe at 20:48:35—meaning it was impossible for it to represent a post-429 recovery!
   I rejected the assistant's initial table, directed a real refill curve probe across multiple wait intervals (`raw/probe_recovery_refill_curve.json`), and re-derived Table 1 strictly from empirical data.

2. **Diagnosing the Async Event-Loop Blocking Bug**:
   During initial steady testing, 53% of requests timed out at 3 RPM. The assistant hypothesized server-side throttling. I inspected the code and recognized that synchronous BM25 and SentenceTransformer calls were blocking Python's single asyncio event loop. I directed the refactoring to `asyncio.to_thread` with singleton model preloading, immediately cutting max event-loop lag by 81.3% and achieving 100% steady success.

3. **Discovering the Live 200,000 TPD Ceiling ("The Second Number They Missed")**:
   Official Groq documentation lists a daily token allowance of 500,000 TPD for `qwen/qwen3.8-27b`. During our real retrieval sweep, at token 199,399, the live API returned HTTP 429 with explicit error payload: `tokens per day (TPD): Limit 200000, Used 199399, Requested 1972`. Rather than accepting doc claims, I logged the live payload as definitive empirical evidence and updated Table 1 with High confidence.

4. **Correcting Small Sample Vanity Metrics**:
   When Setting B scored 11/16 vs Setting A's 9/16, the assistant initially labeled it a "+12.6% quality lift". I called out that a 2-question gap on 16 questions is well within binomial variance ($p=0.50$), and forced the write-up to report the exact fractions and qualitative reasons instead of inflated percentages.

5. **Enforcing Rule 4 (Zero Fabrication Gate)**:
   I authored an automated verification gate (`analysis/verify_tables.py`) that mathematically recomputes every metric in every table directly from `results.csv` and sidecar microsecond logs. Any discrepancy broke the build, guaranteeing that every claim in this write-up is verifiable and grounded in raw telemetry.
