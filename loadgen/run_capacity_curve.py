"""
loadgen/run_capacity_curve.py - The Headline Capacity Experiment.

Measures system throughput across offered rates: 1, 2, 3, 4, and 6 answers per minute.
Identifies the precise empirical "knee" where the 7,000 ITPM token limit binds,
proving that input tokens—not RPM—are the true binding constraint of the system.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from eval.grade import grade_answer_heuristic
from src.llm_client import LLMClient
from src.pipeline import PipelineResult
from src.resilience import ResilienceConfig, ResilienceEngine, ResilientPipelineRunner


QUESTIONS_FILE = PROJECT_ROOT / "eval/questions.json"


def load_questions() -> List[Dict[str, Any]]:
    if QUESTIONS_FILE.exists():
        with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"id": "q01", "question": "What is d_model in Transformer?", "is_refusal": False}]


async def run_rate_trial(
    runner: ResilientPipelineRunner,
    questions: List[Dict[str, Any]],
    target_apm: float,
    num_requests: int = 5,
) -> Dict[str, Any]:
    """Execute a batch of requests spaced at 60.0 / target_apm seconds."""
    interval_s = 60.0 / target_apm
    print(f"\n--- Testing Offered Rate: {target_apm:.1f} answers/min (Interval: {interval_s:.1f}s, N={num_requests}) ---")

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
            f"  [Rate {target_apm:.1f} APM | {i+1}/{num_requests}] Status: {res.status} | "
            f"Retries: {res.retries} | Total: {res.total_ms:.1f}ms | TokensIn: {res.tokens_in}"
        )

        if i < num_requests - 1:
            elapsed = time.perf_counter() - t_req_start
            sleep_rem = max(0.0, interval_s - elapsed)
            if sleep_rem > 0:
                await asyncio.sleep(sleep_rem)

    trial_duration = time.perf_counter() - t_trial_start
    total_offered = len(results)
    ok_count = sum(1 for r in results if r.status == "ok")
    rl_count = sum(1 for r in results if r.status == "rate_limited" or r.retries > 0)
    timeout_count = sum(1 for r in results if r.status == "timeout")
    total_tokens_in = sum(r.tokens_in for r in results)

    tokens_per_min = (total_tokens_in / max(1.0, trial_duration)) * 60.0
    effective_throughput_apm = (ok_count / max(1.0, trial_duration)) * 60.0

    return {
        "offered_rate_apm": target_apm,
        "requests_sent": total_offered,
        "successful_ok": ok_count,
        "success_rate_pct": round((ok_count / total_offered) * 100.0, 1),
        "rate_limited_or_retried_count": rl_count,
        "rate_limited_pct": round((rl_count / total_offered) * 100.0, 1),
        "timeout_count": timeout_count,
        "effective_throughput_apm": round(effective_throughput_apm, 2),
        "tokens_in_per_minute": round(tokens_per_min, 1),
        "mean_latency_ms": round(sum(r.total_ms for r in results) / total_offered, 1),
        "token_limit_bound": tokens_per_min > 6800,
    }


async def main():
    parser = argparse.ArgumentParser(description="Run capacity curve experiment")
    parser.add_argument("--backend", type=str, choices=["stand-in", "real"], default="stand-in")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--requests-per-rate", type=int, default=5)
    args = parser.parse_args()

    questions = load_questions()
    client = LLMClient(
        backend=args.backend,
        base_url=f"{args.server_url}/v1" if args.backend == "stand-in" else None,
    )
    runner = ResilientPipelineRunner(llm_client=client)

    rates = [1.0, 2.0, 3.0, 4.0, 6.0]
    records = []

    print("\n" + "=" * 75)
    print("  HEADLINE CAPACITY EXPERIMENT: LOCATING THE 7,000 ITPM KNEE")
    print(f"  Backend: {args.backend} | Target Rates: {rates} APM")
    print("=" * 75)

    for rate in rates:
        trial_summary = await run_rate_trial(
            runner=runner,
            questions=questions,
            target_apm=rate,
            num_requests=args.requests_per_rate,
        )
        records.append(trial_summary)

    print("\n\n" + "=" * 80)
    print("  CAPACITY CURVE RESULTS TABLE")
    print("=" * 80)
    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    out_file = PROJECT_ROOT / "data/capacity_curve_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved capacity curve summary to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())

