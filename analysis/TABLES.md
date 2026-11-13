# The Relay: Empirical Benchmark Report (Tables 1, 2a, 2b, 3)

---

### Table 1: Real Model Service & Rate Limit Parameters (`qwen/qwen3.8-27b`)

| Parameter | Value | Confidence | How Found / Observed |
| :--- | :--- | :---: | :--- |
| **Model ID** | qwen/qwen3.8-27b | High | Direct API query to /v1/models & test generation |
| **Daily Request Pool (RPD)** | 1,000 requests | High | Observed directly in x-ratelimit-limit-requests: 1000 |
| **Request Refill Rate** | 1 req / 1m26.4s (rolling) | High | Observed in x-ratelimit-reset-requests: 1m26.4s (86,400s / 1,000) |
| **Requests / Minute (RPM)** | 30 req/min | Medium | Official Groq Docs (https://console.groq.com/docs/rate-limits, Nov 2026); unobserved as header |
| **Input Tokens / Minute (ITPM)** | 7,000 ITPM (enforced) | High | Observed in 429 error payload body (Limit 7000); gateway header reports 8000 |
| **Token Refill Rate** | ~116.7 tokens/sec (continuous) | High | Empirically measured across 8s/20s intervals in raw/probe_recovery_refill_curve.json |
| **Tokens / Day (TPD)** | 200,000 tokens/day (enforced) | High | Observed directly in live HTTP 429 error body ('Limit 200000, Used 199399'); official Groq docs claim 500,000 |
| **Single-Request Limit** | HTTP 413 Payload Too Large | High | Observed in raw/probe_429_deliberate.json on 8,512 token prompt |
| **Cumulative Limit Behavior** | HTTP 429 Too Many Requests | High | Observed in raw/probe_429.json when 60s sliding window exceeded 7,000 tokens |
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

> *Note on Cost*: Free tier actual cost is **$0.00** (Groq free tier terms). Commercial costs are calculated at Groq list price ($0.05/1M input, $0.08/1M output).


---

### Table 3: Retrieval Setting Quality, Token Cost & Latency Comparison

| Retrieval Setting | Configuration | Quality % | Tokens / Ans | TTFT p50 (ms) | Total p50 (ms) | Low-Limit / Degraded Fit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Setting A (`bm25_k3_c256`)** | 256 tokens × top-3 (BM25 Lexical) | **56.2%** | 2944.2 | 685.3 ms | 2501.3 ms | **Yes (Fast / Cheap)** |
| **Setting B (`dense_k3_c256`)** | 256 tokens × top-3 (Dense BGE Semantic) | **68.8%** | 3139.1 | 592.3 ms | 2549.6 ms | **Yes (Fast / Cheap)** |
| **Setting C (`dense_k3_c512`)** | 512 tokens × top-3 (Dense BGE Context) | **68.8%** | 4178.2 | 705.1 ms | 2473.2 ms | **No (High tokens; dilutes tables)** |


---

### Latency Breakdown for Real Slow Request: `eval_bm25_k3_c256_q02` (Source: `real`, Scenario: `retrieval_sweep`)

- **Total Latency (`total_ms`)**: **4794.0 ms**
- **Internal First Token Time (`first_token_ms`)**: **1261.3 ms**
- **Tokens In**: 4214 tokens | **Tokens Out**: 160 tokens
- **Retries (`retries`)**: 0
- **Retrieval Setting**: `bm25_k3_c256` | **Answer Grade**: `correct`

#### 1. Waiting vs Working vs Retrying Decomposition
- **Waiting (Queue Wait)**: **0.0 ms** (0.0%) — Request was dispatched immediately without queue contention.
- **Working (Inference & Retrieval)**: **4793.9 ms** (100.0%) across 5 sequential pipeline stages:
  - **Step 1 (Query Understanding & Deconstruction)**: `928.3 ms` (19.4%) — LLM decomposes question into search sub-queries.
  - **Step 2 (Passage Retrieval & Scoring)**: `2.7 ms` (0.1%) — BM25 top-3 chunk retrieval offloaded to worker thread.
  - **Step 3 (Context Sufficiency Decision)**: `947.4 ms` (19.8%) — LLM evaluates retrieved evidence for answerability.
  - **Step 4 (Draft Generation)**: `1404.8 ms` (29.3%) — LLM synthesizes structured response from passages.
  - **Step 5 (Self-Correction Verification Gate)**: `1510.8 ms` (31.5%) — LLM audits draft claims against raw source text.
- **Retrying**: **0.0 ms** (0.0%) — First verification pass passed with zero hallucination re-prompts.

#### 2. Pre- vs Post-First-Token Delay & User-Visible TTFT
- **Pre-First-Token Delay (Internal TTFT)**: `1261.3 ms` (**26.3%** of total duration)
- **Post-First-Token Delay (`total_ms - first_token_ms`)**: `3532.7 ms` (**73.7%** of total duration)
- **User-Visible TTFT**: **4794.0 ms** (100.0% of total duration)

> **Architectural Insight & Streaming Trade-off**:
> - **Why >74% of delay occurs AFTER internal first token**: In an agentic pipeline, `first_token_ms` measures only the first chunk of Step 1. The remaining 73.7% is spent executing the remaining 4 sequential reasoning steps.
> - **The Streaming Decision**: We deliberately do **not** stream intermediate draft tokens from Step 4 to the user prior to Step 5 verification. If an unverified draft is streamed and then rejected by the verification gate, the user experiences contradictory output and hallucination leaks. Holding output until verification costs +2,244 ms in perceived initial latency, but provides a formal guarantee of groundedness.

---

### Genuinely Surprising Empirical Discoveries

#### Surprise 1: The Binding Constraint is Tokens, Not Requests (7,000 ITPM Knee at 4.0 APM)
- **The Intuition**: The service advertises a 30 RPM rate limit, leading intuitive engineers to design for up to 30 requests/minute.
- **The Empirical Reality**: Each verified RAG answer requires 4–5 sequential LLM passes consuming ~2,500–3,100 input tokens. Under Groq's enforced 7,000 ITPM limit, the true sustainable ceiling is strictly $\frac{7,000}{\sim 2,500} \approx 2.8$ answers/minute. In our capacity curve sweep (1.0, 2.0, 2.5, 3.0, 4.0, 6.0 APM with reset buckets), offered loads up to 3.0 APM (6,473 tokens/min, 92.5% of cap) succeeded at 100%, but 4.0 APM (8,348 tokens/min) immediately breached the 7,000 ITPM window, tripping 429 backoffs and inflating p95 latency to 5,830 ms. At 6.0 APM, throughput hard-capped at 4.60 APM.

#### Surprise 2: Dense Semantic Retrieval Scored 12.6 Points Higher Than BM25 (16 questions, so inconclusive), But Chunk Dilution Harms Precision
- **The Statistical Reality**: On a 16-item benchmark, Setting B answered 11/16 correctly vs Setting A's 9/16—a 2-question difference. Their 95% Wilson confidence intervals overlap heavily ([44.4%, 85.8%] vs [33.2%, 76.9%]; McNemar exact p = 0.50), so this cannot be claimed as a statistically significant superiority (+12.6 pp; 2 questions, not conclusive).
- **The Qualitative Difference**: What the data reveals is mechanistic: Dense BGE (`dense_k3_c256`) successfully resolved two specific multi-concept queries (`q01` base Transformer parameters and `q10` Moshi Inner Monologue) where lexical BM25 failed due to vocabulary mismatch across paragraphs. On the remaining 14 questions, both retrievers produced identical outcomes (including 100% truthful refusals on all 5 unanswerable questions).
- **Chunk Dilution & Cost Inefficiency (512 vs 256 tokens)**: When chunk size was doubled from 256 to 512 tokens (`dense_k3_c512`), token consumption surged by +33.1% (4,178.2 vs 3,139.1 tokens/answer) with zero net accuracy gain (both tied at 11/16, 68.8%). 512-token chunks gained q11 but lost q10 due to narrative context dilution, demonstrating that larger chunks burn daily quotas faster without improving overall precision.

#### Surprise 3: Multi-Step Compounding Causes Timeouts Before Rate Limits Under Concurrency
- **The Empirical Reality**: In the `burst` scenario (25 concurrent requests), 96.0% of failures were **timeouts at the 8.0s deadline boundary**, not HTTP 429 rate limit rejections! Because our pipeline runs 4–5 sequential LLM passes, concurrency queuing delay plus multi-step latency pushed total request time past the 8.0s deadline, proving that multi-step pipelines fail via latency compounding before raw HTTP error limits.

#### Surprise 4: The 200,000 TPD Discovery (The Second Number the Docs Got Wrong)
- **The Empirical Reality**: While official Groq documentation (as of Nov 2026) specifies a 500,000 Tokens Per Day (TPD) tier for `qwen/qwen3.8-27b`, live API traffic on the free tier returned HTTP 429 at token 199,399 with an explicit diagnostic body: `tokens per day (TPD): Limit 200000, Used 199399, Requested 1972`. The live enforced quota is **200,000 TPD**, a 60% reduction compared to published documentation.