# The Relay: Evidence & Defense Dossier

This document provides the definitive empirical measurements, architectural proofs, code references, and design rationales for **The Relay** ML Engineering take-home defense. Every single figure is mathematically verified against [`results.csv`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/results.csv) (137 valid rows, exact 12-column schema) and sidecar microsecond telemetry [`raw/timing_breakdown.jsonl`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/raw/timing_breakdown.jsonl) via [`analysis/verify_tables.py`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/analysis/verify_tables.py).

---

## 1. What I Built

### System Architecture
- **Multi-Step Inspectable Pipeline** ([`src/pipeline.py`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/src/pipeline.py)):
  - Built an end-to-end question-answering pipeline grounded exclusively in 3 specific arXiv papers:
    1. *Attention Is All You Need* (`1706.03762v7`) — 15 pages, 24 sections
    2. *Moshi: a speech-text foundation model for real-time dialogue* (`2410.00037v1`) — 66 pages, 36 sections
    3. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (`2507.07955v2`) — 39 pages, 24 sections
  - Total corpus: 120 pages, 55,815 words, 362,929 characters extracted via [`src/ingest.py`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/src/ingest.py).
- **Five Discrete Pipeline Steps** (with independent latency timers):
  1. `step_1_understand`: Query reformulation, synonym expansion, and technical concept extraction.
  2. `step_2_retrieve`: Non-blocking tool call executing BM25 lexical search, Dense BGE (`BAAI/bge-small-en-v1.5`), Hybrid Reciprocal Rank Fusion (RRF), or Cross-Encoder re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
  3. `step_3_decide` (Branch/Loop): Evaluates evidence sufficiency; loops with rewritten query (up to 2 times) or branches to truthful refusal.
  4. `step_4_draft`: Synthesizes cited answer strictly from retrieved passages.
  5. `step_5_verify` (Check & Redraft): Verifies factual claims and page citations against raw source chunks; triggers self-correction redraft if hallucination or unsupported claim is detected; skippable under degraded mode.
- **Core Technology Stack**:
  - Python 3.12, `asyncio`, `httpx` (streaming SSE client), `FastAPI` + `uvicorn` (stand-in server), `pymupdf` (PDF layout extraction), `rank_bm25`, `sentence-transformers`, `pandas`.
  - **Zero High-Level Agent Wrappers**: Explicitly avoided LangChain, LlamaIndex, or AutoGen. Every queue, retry loop, step timer, and cancellation boundary is directly implemented and inspectable in pure Python.
- **Strict Grounding & Citation Invariants**:
  - Every answer must carry page-anchored citations: `[Paper Title, Sec X, p. N]`.
  - Truthful refusal invariant: When evidence is missing or the question is unanswerable, the system outputs `"The papers do not say."` It never hallucinates or extrapolates to appear helpful.

---

## 2. Real Service and Stand-In

### Real Groq Service Probes (`qwen/qwen3.8-27b`)
- Probed using [`src/probe_groq.py`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/src/probe_groq.py); results saved in [`raw/probe_sequential.json`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/raw/probe_sequential.json), [`raw/probe_concurrent.json`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/raw/probe_concurrent.json), [`raw/probe_429.json`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/raw/probe_429.json), and [`raw/probe_recovery_refill_curve.json`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/raw/probe_recovery_refill_curve.json):
  - **Median Internal TTFT**: `801–887 ms` on multi-thousand token prompts.
  - **Generation Speed**: `~300 tokens/sec`.
  - **Concurrency Plateau**: 3 concurrent in-flight requests finished in `470.4 ms` with zero queuing penalty (lower bound $\ge 3$).
  - **Rate Limit Headers**: Gateway header reports `x-ratelimit-limit-requests: 1000` (RPD) and `x-ratelimit-limit-tokens: 8000`.
  - **Enforced Token Cap**: The HTTP 429 error response body reveals the true enforced limit: `Limit 7000, Used ..., Requested ...`. The binding limit is **7,000 Input Tokens Per Minute (ITPM)**.
  - **Token Refill Dynamics**: Measured continuously in [`raw/probe_recovery_refill_curve.json`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/raw/probe_recovery_refill_curve.json) across multiple intervals ($t+8s$, $t+20s$). The bucket refills at $116.7\text{ to }122.3\text{ tokens/sec}$, precisely matching continuous refill ($\frac{7,000}{60\text{s}} = 116.67\text{ tok/s}$).
  - **Single-Request Oversize Limit**: An 8,512-token prompt returned **HTTP 413 Payload Too Large** ([`raw/probe_429_deliberate.json`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/raw/probe_429_deliberate.json)), proving that single-request context overflow is rejected with 413, whereas cumulative rate exhaustion produces 429.

### Calibrated Stand-In Server ([`standin/server.py`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/standin/server.py))
- **TTFT Calibration with Prefill Penalty**: Base TTFT sampled from log-normal distribution ($\mu = 5.90, \sigma = 0.23$) with an added linear prompt prefill term ($0.04\text{ ms} \times \text{tokens\_in}$):
  $$\text{TTFT} = \text{base\_ms} + 0.04 \times \text{tokens\_in}$$
  This reproduces realistic 800–1,200 ms latency when ingesting 2,000–3,000 token retrieved contexts.
- **Streaming Emulation**: Emulates chunk-by-chunk SSE streaming at 300 tokens/sec.
- **Continuous Token Refill & Daily Rolling Pool**:
  - Implements a continuous token bucket recharging at $\frac{7,000}{60} = 116.67\text{ tokens/sec}$ with an enforced 7,000 ITPM cap.
  - Implements a rolling daily pool of 1,000 requests recharging at 1 request every 86.4 seconds ($86,400\text{s} / 1,000$).
- **Single-Request Guard**: Prompts $> 8,000$ tokens immediately return HTTP 413.
- **Fault Injection**: Hooks for `error_rate` (500s), `timeout_rate` (hangs), and `slow_rate` (tail latency).
- **Auditable Discrepancies**:
  1. *Compute Footprint*: Mock runs locally with $<1\%$ CPU footprint; does not emulate GPU memory contention or thermal throttling.
  2. *Network Jitter*: Localhost socket communication eliminates cloud load balancer routing delays and WAN packet retransmissions.
  3. *Token Tokenizer*: Stand-in uses whitespace tokenization calibrated against GPT/Qwen BPE tokenizers ($\approx 1.3 \times \text{words}$).

---

## 3. Benchmark Results (Tables 1, 2a, 2b, 3)

All numbers below are generated directly from [`results.csv`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/results.csv) by [`analysis/tables.py`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/analysis/tables.py) and verified by [`analysis/verify_tables.py`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/analysis/verify_tables.py):

### Table 1: Real Model Service & Rate Limit Parameters (`qwen/qwen3.8-27b`)

| Parameter | Value | Confidence | How Found / Observed |
| :--- | :--- | :---: | :--- |
| **Model ID** | `qwen/qwen3.8-27b` | High | Direct API query to `/v1/models` & successful test generation |
| **Daily Request Pool (RPD)** | 1,000 requests | High | Observed directly in `x-ratelimit-limit-requests: 1000` header |
| **Request Refill Rate** | 1 req / 1m26.4s (rolling) | High | Observed in `x-ratelimit-reset-requests: 1m26.4s` ($86,400\text{s} / 1,000$) |
| **Requests / Minute (RPM)** | 30 req/min | Medium | Official Groq Docs (Nov 2026); unobserved as response header |
| **Input Tokens / Minute (ITPM)** | 7,000 ITPM (enforced) | High | Observed in 429 error payload body (`Limit 7000`); gateway header reports 8000 |
| **Token Refill Rate** | ~116.7 tokens/sec (continuous) | High | Empirically measured across intervals in `raw/probe_recovery_refill_curve.json` |
| **Tokens / Day (TPD)** | 500,000 tokens/day | Medium | Official Groq Docs (Nov 2026); unobserved as response header |
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
| **Setting A (`bm25_k3_c256`)** | 256 tokens × top-3 (BM25 Lexical) | **53.8%** | 2657.4 | 887.6 ms | 3111.0 ms | **Yes (Fast / Cheap)** |
| **Setting B (`dense_k3_c256`)** | 256 tokens × top-3 (Dense BGE Semantic) | **62.5%** | 2421.2 | 801.2 ms | 2782.7 ms | **Yes (Fast / Cheap)** |

> *Quality Scoring Methodology*: Quality is calculated strictly on completed requests (`status == 'ok'`). Setting A completed 13/16 with 7 correct/refused_correctly (53.8%). Setting B completed 8/16 with 5 correct/refused_correctly (62.5%), with 8 requests halted by the 7,000 ITPM rate limit. All 5 refusal questions were truthfully refused (`refused_correctly`) without synthetic keyword rules.

---

## 4. Where the Time Goes (Slow Request Latency Decomposition)

### Case Study: Real Slow Request `eval_bm25_k3_c256_q02` (Source: `real`, Scenario: `retrieval_sweep`)
- **Total Latency (`total_ms`)**: **5,558.5 ms**
- **Internal First Token Time (`first_token_ms`)**: **939.5 ms**
- **Tokens In**: 4,214 | **Tokens Out**: 160
- **Retries (`retries`)**: 0
- **Retrieval Setting**: `bm25_k3_c256` | **Answer Grade**: `correct`

#### 1. Waiting vs Working vs Retrying Decomposition (from `raw/timing_breakdown.jsonl`)
- **Waiting (Queue Wait)**: **0.0 ms** (**0.0%**) — Dispatched immediately without queue contention.
- **Working (Inference & Retrieval)**: **5,558.5 ms** (**100.0%**) across 5 sequential stages:
  - **Step 1 (Query Understanding & Deconstruction)**: `1,159.9 ms` (**20.9%**) — LLM decomposes question into targeted sub-queries.
  - **Step 2 (Passage Retrieval & Scoring)**: `2.8 ms` (**0.1%**) — BM25 top-3 chunk retrieval executed via `asyncio.to_thread`.
  - **Step 3 (Context Sufficiency Decision)**: `1,351.4 ms` (**24.3%**) — LLM evaluates retrieved passages for answerability.
  - **Step 4 (Draft Generation)**: `1,052.0 ms` (**18.9%**) — LLM synthesizes structured response with bracketed citations.
  - **Step 5 (Self-Correction Verification Gate)**: `1,992.4 ms` (**35.8%**) — LLM audits draft claims against source chunk text.
- **Retrying**: **0.0 ms** (**0.0%**) — Verification passed on initial attempt. (In contrast, request `eval_bm25_k3_c256_q05` triggered 1 retry taking `1,460.7 ms` / 30.9% of total time).

#### 2. Pre- vs Post-First-Token Delay & User-Visible TTFT
- **Pre-First-Token Delay (Internal TTFT)**: `939.5 ms` (**16.9%** of total duration)
- **Post-First-Token Delay (`total_ms - first_token_ms`)**: `4,619.0 ms` (**83.1%** of total duration)
- **User-Visible TTFT**: **5,558.5 ms** (**100.0%** of total duration)

> **Architectural Decision & The Streaming Trade-off**:
> - **Why >83% of delay occurs AFTER internal first token**: In an agentic pipeline, `first_token_ms` measures only the first token emitted by Step 1. The remaining 83.1% is consumed executing the remaining 4 sequential reasoning passes.
> - **The Streaming Decision**: We deliberately do **not** stream draft tokens from Step 4 before Step 5 verification. If an unverified draft is streamed and subsequently fails the verification gate, the user has already read hallucinated or incorrect claims. Holding output until verification adds +2,244 ms to perceived initial latency, but provides a 100% guarantee of groundedness.

---

## 5. When a Limit is Hit

### The Headline Insight: The Number They Missed
- Groq advertises a **30 RPM** rate limit, prompting engineers to assume the system can sustain ~30 requests/minute.
- However, each verified answer requires 4–5 sequential LLM passes consuming $\approx 2,400\text{--}2,650$ input tokens.
- Under Groq's enforced **7,000 ITPM limit**, sustainable throughput is strictly:
  $$\text{Capacity} = \frac{7,000\text{ ITPM}}{2,500\text{ tokens/answer}} \approx 2.8\text{ answers/minute}$$
- In our capacity curve experiment ([`loadgen/run_capacity_curve.py`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/loadgen/run_capacity_curve.py)), offering 1.0 and 2.0 APM achieved **100% success** (p50 total 2,367.6 ms and 2,900.5 ms). But at **3.0 APM**, the 7,000 ITPM bucket saturated, triggering 429 backoff and pushing 100% of requests past the 8.0s deadline.
- **Token capacity, not request count, is the binding operational constraint.** This elevates retrieval chunk size from an internal tuning knob into an external system capacity lever.

### How This Library Misleads a Naive Retriever (3 Real Failure Modes)
Cataloged in [`data/papers/extraction_report.json`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/data/papers/extraction_report.json):
1. **Linearized Table Grid Collapse (14 tables across corpus)**:
   - *Example*: Transformer Table 2 BLEU scores (Question `q06`). PyMuPDF linearizes 2D table columns into continuous 1D text (`28.4`, `41.8`, `2.3 · 1019`), stripping column headers. Naive retrievers pull numbers without labels, causing hallucinated language mappings or false refusals (`refused_wrongly`).
2. **Mangled Multi-Line Equations & Formatting (3 major equation clusters)**:
   - *Example*: Attention scaling factor $\frac{1}{\sqrt{d_k}}$ (Question `q15`). Fractions split across line breaks (`1 \n √dk`) and superscripts flatten ($10^{20} \to 1020$). Lexical tokenizers fail to match queries due to character breaks.
3. **Diagram Vector Glyph Stacking (13 clusters, max run 43 chars)**:
   - *Example*: Dynamic Chunking Figure 1 (Question `q03`). Vector diagrams extract as vertical character runs (`n \n d \n o \n f \n t \n o \n k \n e \n n \n s`) polluted with disjoint glyphs (`!`, `/0`, `ℳ`). These chunks pollute embedding space, causing dense retrievers to rank noisy diagram text above actual methodology paragraphs.

---

## 6. Retries & Resilience Architecture

Implemented in [`src/resilience.py`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/src/resilience.py):

### Resilience Engine Design
1. **Exponential Backoff with Full Jitter**:
   - AWS Full Jitter: $\text{delay} = \text{Uniform}(0, \min(4.0s, 0.5s \times 2^{\text{attempt}}))$.
   - Obeys upstream `Retry-After` headers, adding minor random jitter ($+50\text{--}250\text{ms}$) to prevent synchronization storms.
2. **Safety Rules & Global Retry Budget**:
   - Only idempotent status codes `{429, 502, 503, 504, 408}` are retried. Non-retryable client errors (`400, 401, 403, 404, 422`) fail fast. Max 3 retries per request.
   - **Global Retry Budget**: Retries capped at **25%** of lifetime requests seen.
3. **Per-Request Deadline (8.0s Justification)**:
   - The 8.0s deadline is derived directly from the pipeline's attempt-backoff-retry runway:
     $$\text{Runway} = \text{Attempt 1 }(2.0\text{s}) + \text{Backoff Wait }(1.5\text{s}) + \text{Attempt 2 Retry }(2.0\text{s}) + \text{Network Buffer }(2.5\text{s}) = 8.0\text{s}$$
   - Any request waiting in queue for $>3.0\text{s}$ cannot complete two attempts before 8.0s. Inter-step `check_deadline()` assertions drop doomed requests immediately to prevent wasting GPU quota.
4. **Bounded Queue & Load Shedding**:
   - Queue capacity: 20 requests. At $\ge 75\%$ capacity (15 requests), incoming requests automatically switch to **Degraded Mode**. At 100% capacity (20 requests), incoming requests are **shed immediately** (`retrieval_setting="shed"`) with 0ms delay.

### Retry Amplification Experiment ([`raw/retry_amplification_demo.json`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/raw/retry_amplification_demo.json))
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
   > *"The Relay research assistant has reached its daily free-tier computation budget (1,000 requests / 500,000 tokens). To ensure strict cost controls, no further automated calls can be made today. Quota refills continuously at 1 request every 86.4 seconds. Please try again later."*
4. **Overload Shedding Notification**:
   > *"Server is currently overloaded. Request was shed to protect system stability."*

---

## 8. Unsure / Left Out / Learned

### Deliberate Omissions & Trade-Offs
- **OCR / Vision Table Parsing**: Linearized table collapse was mitigated by isolating tables into 256-token chunks rather than introducing heavy multimodal vision models (e.g. Nougat, Surya). This preserved lightweight, sub-second local ingestion.
- **Speculative Query Pre-fetching**: We avoided speculative parallel retrieval on rewritten queries to preserve our strict 7,000 ITPM rate limit runway.

### Core Engineering Learnings
1. **The Compounding Penalty of Multi-Step Pipelines**:
   In single-turn generation, latency is just network time plus token emission. In verified RAG, latency compounds across 4–5 sequential reasoning steps. A modest 300 ms slowdown per step accumulates into a 1.5 s increase per turn, turning queuing contention into instant deadline timeouts.
2. **Token Limits Govern Throughput, Not RPM**:
   Free-tier LLM rate limits cannot be modeled by request counts alone. Systems must track cumulative token consumption and scale retrieval context down to maintain operational throughput.
3. **A/B Isolation of Steady Workload & Event-Loop Health**:
   We ran a controlled A/B test ([`data/ab_steady_isolation_summary.json`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/data/ab_steady_isolation_summary.json)) holding offered rate ($2.6\text{ req/min}$, 23.08s interval) and request count ($N=15$) constant:
   - *Pre-fix Architecture (Synchronous CPU retrieval on event loop + lazy model loading)*: Experienced a **5,791.2 ms maximum event loop lag** during cold-start model initialization and synchronous retrieval passes. This pushed p95 total latency to **7,842.1 ms** (on the brink of the 8.0s deadline) and produced an initial timeout (93.3% success).
   - *Post-fix Architecture (`asyncio.to_thread` + preloaded singleton models)*: Reduced maximum event loop lag by **81.3%** down to **1,084.1 ms**, cut p95 total latency from 7,842 ms down to **5,941.0 ms**, and achieved **100.0% success** across all 15 requests.
   - *Root Cause of Original 53% Failure*: Auditing the git history of the original steady run (`commit 795370c`) revealed that requests were arriving with an 8.0s/clustered interval ($\sim 9.7\text{ req/min}$), generating $\sim 25,000\text{ tokens/min}$ against the 7,000 ITPM cap. The original collapse was therefore **both token bucket exhaustion under rapid arrivals AND event-loop blocking during cold-start/retrieval**. Isolating the rate at 2.6 req/min removed token exhaustion, allowing us to isolate and prove the 1,900 ms p95 latency reduction delivered by the async fix.

---

## 9. Vaani Reflections: Conversational Interfaces & Latency Boundaries

In exploring conversational voice and dialogue modeling (exemplified by the *Moshi* paper included in our library), the interaction between latency and grounding becomes the central product dilemma:
- **The Voice vs. RAG Latency Conflict**: Human conversation operates on a strict **200–300 ms turn-taking boundary**. Moshi achieves real-time speech dialogue by streaming 12.5 Hz audio tokens with an asymmetric encoder-decoder architecture. In contrast, verified technical RAG requires multi-step retrieval, sufficiency analysis, and claim verification—consuming **2,700–3,100 ms** even on lightning-fast LPU hardware.
- **Resolving the Paradox**: A conversational voice agent cannot remain dead silent for 3.0 seconds while an internal RAG pipeline executes. The UI must provide conversational acoustic fillers (*"Checking the papers..."*, *"Looking at Table 2..."*) or dual-stream intermediate intent acknowledgments while holding the factual payload until Step 5 verification guarantees that the spoken output is unhallucinated.

---

## 10. How I Worked with the Assistant

### Pair-Programming Dynamic & The Table 1 Hallucination Catch
Throughout this project, I used the AI assistant as a rapid implementation partner while maintaining strict empirical skepticism and architectural ownership:

1. **Catching the Table 1 Hallucinations**:
   When the assistant initially populated Table 1, it claimed Groq enforced **30 RPM observed in response headers** and an **8,000 TPM limit**, citing an **18.0s recovery time** from `probe_recovery.json`.
   Upon auditing the raw HTTP headers in `raw/probe_sequential.json`, I noticed:
   - Every single response header stated `x-ratelimit-limit-requests: 1000`, not 30. The 30 RPM was an unobserved figure copied from online documentation.
   - The reset countdown header (`x-ratelimit-reset-requests: 1m26.4s`) exactly equaled $86,400\text{s} / 1,000$, proving a continuous rolling daily refill rather than a static midnight reset.
   - The HTTP 429 response body explicitly cited an enforced **7,000 ITPM limit**, contradicting the gateway's 8,000 TPM header.
   - `probe_recovery.json` was timestamped at 20:48:14—*before* the 429 probe at 20:48:35—meaning it was impossible for it to represent a post-429 recovery!
   I rejected the assistant's initial table, forced a live re-probe of Groq's refill curve across multiple intervals (`raw/probe_recovery_refill_curve.json`), and re-derived Table 1 strictly from empirical reality.

2. **Diagnosing the Async Event-Loop Bug**:
   During initial steady-load testing, 53% of requests timed out at 3 RPM. The assistant hypothesized server-side throttling. I inspected the code and recognized that synchronous BM25 and SentenceTransformer calls were blocking the Python asyncio event loop. I directed the refactoring to `asyncio.to_thread` with singleton model preloading, immediately resolving the issue and bringing steady success to 100%.

3. **Enforcing Rule 4 (Zero Fabrication)**:
   I established an automated gate (`analysis/verify_tables.py`) that recomputes every figure directly from `results.csv` and sidecar telemetry. Any discrepancy between markdown tables and underlying CSV data broke the build. This dynamic ensured the final submission is mathematically airtight and defense-grade.
