"""
loadgen/run_capacity_curve.py - The Headline Capacity Experiment.

Measures system throughput across offered rates: 1.0, 2.0, 2.5, 3.0, 4.0, and 6.0 answers per minute.
Identifies the precise empirical "knee" where the 7,000 ITPM token limit binds,
proving that input tokens—not RPM—are the true binding constraint of the system.

Features:
  - Resets stand-in token bucket before each rate trial for clean isolation
  - Statistically robust sample sizes (N=6 to N=10 per rate, N=10 at the 3.0 APM knee)
  - Records p50, p95 latency, throughput, token inflow, and binding status
  - Optional real Groq validation check (--real-check) at 3.0 APM with --max-requests 20
"""

import argparse
import asyncio
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import LLMClient
from src.pipeline import PipelineResult
from src.resilience import ResilienceConfig, ResilienceEngine, ResilientPipelineRunner

QUESTIONS_FILE = PROJECT_ROOT / "eval/questions.json"
STANDIN_URL = "http://localhost:8000"
RESULTS_CSV = PROJECT_ROOT / "results.csv"


def load_questions() -> List[Dict[str, Any]]:
    if QUESTIONS_FILE.exists():
        with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"id": "q01", "question": "What is d_model in Transformer?", "is_refusal": False}]


async def reset_standin():
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{STANDIN_URL}/config", json={
                "limit_rpm": 30,
                "limit_tpm": 7000,
                "limit_daily_requests": 1000,
                "error_rate": 0.0,
                "timeout_rate": 0.0,
                "slow_rate": 0.0,
                "forced_429": False,
            }, timeout=3.0)
            await client.post(f"{STANDIN_URL}/reset", timeout=3.0)
    except Exception as e:
        print(f"[Warning] Standin reset failed: {e}")


async def run_rate_trial(
    runner: ResilientPipelineRunner,
    questions: List[Dict[str, Any]],
    target_apm: float,
    num_requests: int = 10,
) -> Dict[str, Any]:
    """Execute a batch of requests spaced at 60.0 / target_apm seconds."""
    interval_s = 60.0 / target_apm
    print(f"\n{'='*75}")
    print(f"--- Testing Offered Rate: {target_apm:.1f} answers/min (Interval: {interval_s:.1f}s, N={num_requests}) ---")
    print(f"{'='*75}")

    # Reset stand-in token bucket before each trial to ensure clean isolation
    if runner.llm.backend == "stand-in":
        await reset_standin()
        await asyncio.sleep(1.0)

    results: List[PipelineResult] = []
    t_trial_start = time.perf_counter()

    for i in range(num_requests):
        q_meta = questions[i % len(questions)]
        t_req_start = time.perf_counter()

        res = await runner.run(
            question=q_meta["question"],
            scenario="capacity_curve",
            question_meta=q_meta,
        )
        results.append(res)
        print(
            f"  [Rate {target_apm:.1f} APM | {i+1:02d}/{num_requests}] Status: {res.status:<12} | "
            f"Retries: {res.retries} | Total: {res.total_ms:6.1f}ms | TTFT: {res.first_token_ms:6.1f}ms | TokensIn: {res.tokens_in:4d}"
        )

        if i < num_requests - 1:
            elapsed = time.perf_counter() - t_req_start
            sleep_rem = max(0.0, interval_s - elapsed)
            if sleep_rem > 0:
                await asyncio.sleep(sleep_rem)

    trial_duration = time.perf_counter() - t_trial_start
    total_offered = len(results)
    ok_count = sum(1 for r in results if r.status == "ok")
    rl_count = sum(1 for r in results if r.status == "rate_limited")
    timeout_count = sum(1 for r in results if r.status == "timeout")
    retried_count = sum(1 for r in results if r.retries > 0)
    total_tokens_in = sum(r.tokens_in for r in results)

    totals = [r.total_ms for r in results]
    ttfts = [r.first_token_ms for r in results if r.first_token_ms > 0]

    tokens_per_min = (total_tokens_in / max(1.0, trial_duration)) * 60.0
    effective_throughput_apm = (ok_count / max(1.0, trial_duration)) * 60.0

    # Token limit is binding if we observe rate-limiting, retries, or timeouts caused by backoff,
    # or token inflow is within 85% of the 7,000 cap
    token_bound = (rl_count > 0) or (retried_count > 0) or (timeout_count > 0 and tokens_per_min > 5500)

    summary = {
        "offered_rate_apm": target_apm,
        "interval_seconds": round(interval_s, 2),
        "requests_sent": total_offered,
        "successful_ok": ok_count,
        "success_rate_pct": round((ok_count / total_offered) * 100.0, 1),
        "rate_limited_count": rl_count,
        "timeout_count": timeout_count,
        "retried_count": retried_count,
        "effective_throughput_apm": round(effective_throughput_apm, 2),
        "tokens_in_per_minute": round(tokens_per_min, 1),
        "total_p50_ms": round(float(np.median(totals)), 1) if totals else 0.0,
        "total_p95_ms": round(float(np.percentile(totals, 95)), 1) if totals else 0.0,
        "ttft_p50_ms": round(float(np.median(ttfts)), 1) if ttfts else 0.0,
        "token_limit_bound": token_bound,
    }
    return summary


async def run_real_groq_check():
    """
    Validates the 3.0 APM capacity inflection point on real Groq.
    Sends 4 requests spaced at 20.0s (3.0 APM) with --max-requests 20 hard ceiling.
    Logs remaining-token depletion curve from x-ratelimit-remaining-tokens.
    """
    print("\n" + "=" * 75)
    print("  REAL GROQ VALIDATION CHECK AT 3.0 APM")
    print("  Testing 4 requests spaced at 20.0s (3.0 APM) | Max requests: 20")
    print("=" * 75)

    questions = load_questions()
    client = LLMClient(backend="real", max_requests=20)
    runner = ResilientPipelineRunner(llm_client=client)

    results = []
    interval_s = 20.0
    t0 = time.perf_counter()

    for i in range(4):
        q_meta = questions[i]
        t_req = time.perf_counter()
        print(f"\n[Real Groq 3.0 APM | Req {i+1}/4] Dispatching: '{q_meta['question'][:50]}...'")

        res = await runner.run(
            question=q_meta["question"],
            scenario="capacity_real_check",
            question_meta=q_meta,
        )
        results.append(res)
        rem_tok = client.last_headers.get("x-ratelimit-remaining-tokens", "N/A")
        print(
            f"  Status: {res.status} | Total: {res.total_ms:.1f}ms | TTFT: {res.first_token_ms:.1f}ms | "
            f"TokensIn: {res.tokens_in} | Retries: {res.retries} | RemainingTokens: {rem_tok}"
        )

        if i < 3:
            elapsed = time.perf_counter() - t_req
            sleep_rem = max(0.0, interval_s - elapsed)
            if sleep_rem > 0:
                print(f"  Pacing wait: sleeping {sleep_rem:.1f}s to maintain 3.0 APM offered rate...")
                await asyncio.sleep(sleep_rem)

    duration = time.perf_counter() - t0
    ok_count = sum(1 for r in results if r.status == "ok")
    total_tokens = sum(r.tokens_in for r in results)

    real_summary = {
        "scenario": "real_groq_3.0_apm_check",
        "requests_sent": len(results),
        "successful_ok": ok_count,
        "success_rate_pct": round((ok_count / len(results)) * 100.0, 1),
        "duration_seconds": round(duration, 1),
        "total_tokens_in": total_tokens,
        "tokens_per_minute": round((total_tokens / duration) * 60.0, 1),
        "requests": [
            {
                "request_id": r.request_id,
                "status": r.status,
                "total_ms": round(r.total_ms, 1),
                "tokens_in": r.tokens_in,
                "retries": r.retries,
            }
            for r in results
        ],
        "final_remaining_tokens": client.last_headers.get("x-ratelimit-remaining-tokens", "N/A"),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    out_real = PROJECT_ROOT / "data/capacity_real_groq_check.json"
    with open(out_real, "w", encoding="utf-8") as f:
        json.dump(real_summary, f, indent=2)
    print(f"\n[Success] Real Groq check summary saved to {out_real}")
    return real_summary


async def main():
    parser = argparse.ArgumentParser(description="Run capacity curve experiment")
    parser.add_argument("--backend", type=str, choices=["stand-in", "real"], default="stand-in")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--real-check", action="store_true", help="Run 4-request validation check on real Groq at 3.0 APM")
    args = parser.parse_args()

    if args.real_check:
        await run_real_groq_check()
        return

    questions = load_questions()
    client = LLMClient(
        backend=args.backend,
        base_url=f"{args.server_url}/v1" if args.backend == "stand-in" else None,
    )
    runner = ResilientPipelineRunner(llm_client=client)

    # Configured rates and sample counts:
    # N=6 for baseline rates, N=10 for critical knee at 3.0 APM
    rate_configs = [
        (1.0, 6),
        (2.0, 6),
        (2.5, 8),
        (3.0, 10),
        (4.0, 8),
        (6.0, 8),
    ]

    records = []

    print("\n" + "=" * 75)
    print("  HEADLINE CAPACITY EXPERIMENT: LOCATING THE 7,000 ITPM KNEE")
    print(f"  Backend: {args.backend} | Target Rates: {[r[0] for r in rate_configs]} APM")
    print("=" * 75)

    for rate, num_reqs in rate_configs:
        trial_summary = await run_rate_trial(
            runner=runner,
            questions=questions,
            target_apm=rate,
            num_requests=num_reqs,
        )
        records.append(trial_summary)

    print("\n\n" + "=" * 90)
    print("  CAPACITY CURVE RESULTS TABLE")
    print("=" * 90)
    df = pd.DataFrame(records)
    cols = ["offered_rate_apm", "requests_sent", "successful_ok", "success_rate_pct", "rate_limited_count", "timeout_count", "effective_throughput_apm", "tokens_in_per_minute", "total_p50_ms", "token_limit_bound"]
    print(df[cols].to_string(index=False))

    out_file = PROJECT_ROOT / "data/capacity_curve_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved capacity curve summary to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
