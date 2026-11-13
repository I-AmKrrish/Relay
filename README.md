# The Relay: Resilient Multi-Step LLM Pipeline & Load Stress Benchmark

A multi-step LLM question-answering pipeline grounded strictly in three arXiv papers, evaluated under traffic load and rate limits with automated resilience, bounded queuing, degraded mode, and an inspectable calibrated stand-in server.

Built for an ML Engineering take-home assignment. Designed to be completely transparent and inspectable: **zero LangChain or high-level agent wrappers**, explicit timing instrumentation on every step, and strict ground-truth logging to `results.csv` and sidecar microsecond telemetry `raw/timing_breakdown.jsonl`.

---

## 📚 The Paper Library (Exact Versions)
The assistant answers **strictly** from these three papers and refuses if unsupported:
1. **Attention Is All You Need**: arXiv `1706.03762v7`
2. **Moshi: a speech-text foundation model for real-time dialogue**: arXiv `2410.00037v1`
3. **Dynamic Chunking for End-to-End Hierarchical Sequence Modeling**: arXiv `2507.07955v2`

Every answer cites the source down to the page: `[Paper Title, Sec X, p. N]`. If the library does not state the answer, the system truthfully outputs: `"The papers do not say."`

---

## 🏗️ Architecture & Five-Step Pipeline

```mermaid
flowchart TD
    UserQuery["User Question"] --> S1["1. Understand (Rewrite / Sub-queries)"]
    S1 --> S2["2. Retrieve (BM25 / Dense BGE / Hybrid / Rerank)"]
    S2 --> S3{"3. Decide (Sufficient?)"}
    S3 -- "Weak (Loop <= 2x)" --> S2
    S3 -- "Unsupported" --> Refusal["Refuse: 'The papers do not say.'"]
    S3 -- "Sufficient" --> S4["4. Draft (Cited Answer)"]
    S4 --> S5{"5. Verify (Check Claims & Citations)"}
    S5 -- "Passed" --> FinalAnswer["Final Answer with Citations"]
    S5 -- "Failed Critique" --> Redraft["Redraft (Max 1x)"]
    Redraft --> S5
```

Each step records its own microsecond latency to `raw/timing_breakdown.jsonl`: `queue_wait_ms`, `step_1_understand_ms`, `step_2_retrieve_ms`, `step_3_decide_ms`, `step_4_draft_ms`, `step_5_verify_ms`, and `retry_time_ms`.

---

## 🛡️ Resilience & Load Shedding Mechanisms

- **Exponential Backoff with Full Jitter**: Follows AWS Full Jitter pattern ($\text{delay} = \text{Uniform}(0, \min(4.0s, 0.5s \times 2^{\text{attempt}}))$) and strictly obeys upstream HTTP 429 `Retry-After` headers.
- **Safety Rules & Retry Budget**: Only retries safe idempotent codes (`{429, 502, 503, 504, 408}`); non-retryable client errors (`{400, 401, 403, 404, 422}`) fail fast. Global retries are capped at 25% of total requests.
- **Per-Request Deadline (8.0s)**: Derived from the pipeline's attempt-backoff-retry runway (~2.0s initial attempt + ~1.5s backoff wait + ~2.0s retry attempt + ~2.5s network buffer). Inter-step `check_deadline()` assertions drop doomed requests immediately to prevent wasting quota.
- **Bounded Queue & Load Shedding**: Queue depth capped at 20. At $\ge 75\%$ capacity, incoming requests automatically switch to **Degraded Mode**; at 100% capacity, excess requests are **immediately shed** with 0ms delay.
- **Degraded Mode**: Uses Setting A (256-token BM25) and skips the verification step, reducing prompt token usage by **32.6%** and saving ~1.5s per request, while strictly maintaining the refusal invariant.
- **Daily Budget Exhaustion**: Provides transparent user notification with exact rolling refill details (1 request refilled every 86.4 seconds).

---

## 📁 Repository Layout

```
relay/
  data/papers/          # Downloaded PDFs, extracted text, chunks, embeddings
  src/
    ingest.py           # Ingestion, section extraction & anomaly detection
    chunking.py         # Configurable sliding-window chunker (256 & 512 tokens)
    retrieval.py        # Async non-blocking retrieval: BM25, Dense BGE, Hybrid RRF, Cross-Encoder
    pipeline.py         # 5-step pipeline with branch, loop & verification
    llm_client.py       # Unified streaming client (Groq / Stand-in) with TTFT timing
    resilience.py       # Resilience engine, jittered backoff, queue & shedding
  standin/
    server.py           # Calibrated FastAPI mock server (continuous 7,000 ITPM bucket, rolling 1,000 RPD)
  loadgen/
    run_load.py         # Workload generator (steady, burst, overload, halved, exhausted)
    run_capacity_curve.py # Headline experiment mapping 1, 2, 3, 4, 6 APM capacity knee
  eval/
    questions.json      # 16 benchmark questions including 4 challenging in-domain near-misses
    grade.py            # Automated heuristic & LLM grading helper
  analysis/
    tables.py           # Generates Tables 1, 2a, 2b, 3 from results.csv
    verify_tables.py    # Rule 4 automated verification gate & consistency checker
    TABLES.md           # Formatted markdown tables & empirical findings
  raw/                  # Raw probe logs, refill curves, and sidecar timing_breakdown.jsonl
  results.csv           # 212 validated execution rows (exact 12-column schema)
  NOTES.md              # Detailed running engineering log across all phases
  WRITEUP_EVIDENCE.md   # Definitive section-by-section evidence dossier for final defense
  requirements.txt      # Python dependencies
```

---

## 🚀 Quickstart & Reproduction

### 1. Setup Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Key (Optional for real Groq probes)
Copy `.env.example` to `.env` and insert your Groq API key:
```bash
cp .env.example .env
```
*(All stand-in runs operate completely locally without needing an API key).*

### 3. Start the Calibrated Stand-in Server
```bash
python standin/server.py --port 8000
```

### 4. Run the Pipeline
```bash
# Query via stand-in
python src/pipeline.py --question "What is the scaling factor in Scaled Dot-Product Attention?" --backend stand-in

# Query with resilience engine
python src/resilience.py --question "What is the dimensionality d_model in the base Transformer?"
```

### 5. Run the Load Generator & Capacity Curve
```bash
# Run load scenarios (burst, sustained_overload, limit_halved)
python loadgen/run_load.py --scenario burst

# Run the headline capacity curve experiment (1, 2, 3, 4, 6 APM)
python loadgen/run_capacity_curve.py

# Demonstrate retry amplification
python src/resilience.py --demo-amplification
```

### 6. Verify Tables & Consistency
```bash
python analysis/tables.py --save-md
python analysis/verify_tables.py
```
This reads `results.csv` and ensures 100% mathematical consistency across all tables.

---

## 📊 Summary of Empirical Findings

1. **Table 1 (Service Parameters & Live TPD Discovery)**: Groq free-tier enforces a rolling pool of **1,000 RPD** (refilling 1 request every 86.4s) and an enforced cap of **7,000 ITPM** (refilling continuously at ~116.7 tok/s). While official docs cite 500,000 Tokens/Day (TPD), live API traffic returned HTTP 429 at token 199,399 with an explicit body revealing the true quota is **200,000 TPD**. Prompts $>8,000$ tokens return HTTP 413 Payload Too Large.
2. **The Headline Insight ("The Number They Missed")**:
   - The binding constraint is **tokens**, not requests.
   - At ~2,500 input tokens per verified RAG answer and a 7,000 ITPM cap, maximum sustainable throughput is strictly $\approx 2.8$ answers/minute.
   - Stand-In Capacity Sweep ([`data/capacity_curve_summary.json`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/data/capacity_curve_summary.json)): 100% success up to 3.0 APM (6,473 tok/min, 92.5% cap); token limit binds at 4.0 APM (8,348 tok/min, p95 latency jumps to 5.8s); throughput plateaus at 4.6 APM under 6.0 APM offered load.
   - Real Groq 3.0 APM Check ([`data/capacity_real_groq_check.json`](file:///c:/Users/rabbu/OneDrive/Desktop/Projects/Relay/data/capacity_real_groq_check.json)): 4 consecutive requests spaced 20s apart bled remaining tokens down to 1,283 by request 3, empirically proving the continuous deficit.
3. **Table 2a & 2b (Traffic Scenarios)**:
   - `steady`: **100.0% success** (15/15 ok), p50 total latency 3,062.1 ms, p95 5,941.0 ms across 15 requests at 2.8 req/m (isolated A/B test proved async threading and model preloading cut event-loop max lag from 5,791 ms down to 1,084 ms).
   - `burst`: 25 concurrent requests hit the queue; 96.0% experienced clean timeouts at the 8.0s deadline boundary.
   - `sustained_overload`: 20 requests safely bounded by 8.0s deadline timeouts.
   - `daily_exhausted`: Requests rejected in 0ms consuming 0 tokens.
4. **Table 3 (Retrieval Trade-Offs on Real Groq across 3 Controlled Settings)**:
   - **Setting A (`bm25_k3_c256`)**: Completed 16/16 with **56.2% quality** (9/16: 4 factual + 5 refusals) and 2,944.2 tokens/answer.
   - **Setting B (`dense_k3_c256`)**: Completed 16/16 with **68.8% quality** (11/16: 6 factual + 5 refusals) and 3,139.1 tokens/answer (**+12.6% quality lift** over BM25). Dense embeddings resolved multi-concept queries (`q01` and `q10`) that BM25 missed due to keyword dispersion.
   - **Setting C (`dense_k3_c512`)**: Completed 15/16 with **66.7% quality** (10/15: 5 factual + 5 refusals) and 4,016.8 tokens/answer (**+36.4% token overhead**). Demonstrates **chunk dilution**: larger chunks added surrounding narrative that degraded technical precision and accelerated token bucket depletion.
   - All 47 rows completed with `status == 'ok'` with zero status/grade contradictions.
5. **Where the Time Goes (Real Request `eval_bm25_k3_c256_q02`)**:
   - Total latency was 4,794.0 ms, with 100% of time spent in working execution across 5 discrete pipeline stages (Step 1 928ms, Step 2 2.7ms, Step 3 947ms, Step 4 1,405ms, Step 5 1,511ms).
   - Over **73.7% of latency occurred AFTER the first internal token** because of subsequent sequential passes (decide, draft, verify). Output is held until verification passes to prevent streaming ungrounded hallucinations, making user-visible TTFT $\approx 4,794.0\text{ ms}$.
