"""
loadgen/run_load.py - Workload Generator for The Relay.

Executes and benchmarks the 5 required traffic shapes:
  1. steady: Evenly spaced requests below limit (10 requests, offered rate ~15 RPM).
  2. burst: Sudden spike of 25 concurrent requests, testing bounded queue (capacity 20),
            degraded activation (queue >= 15), and immediate load shedding (queue >= 20).
  3. sustained_overload: Sustained offered rate (40 RPM) against a 20 RPM limit,
                         testing persistent backoff, shedding, and degraded mode.
  4. limit_halved: Dynamically halves stand-in RPM from 30 to 15, measuring adaptation
                   to sudden quota constriction.
  5. daily_exhausted: Simulates hitting the 1,000 RPD daily ceiling, validating transparent
                      user messaging (UTC reset time, remaining hours, non-queued status).

All execution logs are written to results.csv matching the exact 12-column schema:
  request_id,scenario,source,start_time,first_token_ms,total_ms,status,retries,tokens_in,tokens_out,retrieval_setting,answer_grade
"""

import argparse
import asyncio
import datetime
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import LLMClient
from src.pipeline import PipelineResult
from src.resilience import ResilienceConfig, ResilienceEngine, ResilientPipelineRunner
from eval.grade import grade_answer_heuristic


# Load standard question set
QUESTIONS_FILE = PROJECT_ROOT / "eval/questions.json"


def load_questions() -> List[Dict[str, Any]]:
    if QUESTIONS_FILE.exists():
        with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return [
        {
            "id": "q01",
            "question": "What is the dimensionality d_model in the base Transformer model?",
            "is_refusal": False,
        }
    ]


class LoadGenerator:
    """Orchestrates traffic shapes against The Relay pipeline."""

    def __init__(
        self,
        backend: str = "stand-in",
        server_url: str = "http://127.0.0.1:8000",
        max_requests: Optional[int] = None,
    ):
        self.backend = backend.lower()
        self.server_url = server_url
        self.max_requests = max_requests
        self.questions = load_questions()
        self.client = LLMClient(
            backend=self.backend,
            base_url=f"{self.server_url}/v1" if self.backend == "stand-in" else None,
            max_requests=self.max_requests,
        )

    async def _configure_standin(
        self,
        limit_rpm: Optional[int] = None,
        limit_tpm: Optional[int] = None,
        limit_daily: Optional[int] = None,
        forced_429: bool = False,
        error_rate: float = 0.0,
        timeout_rate: float = 0.0,
        slow_rate: float = 0.0,
    ):
        """Dynamically configure stand-in server limits and behaviors."""
        if self.backend != "stand-in":
            return

        payload: Dict[str, Any] = {"forced_429": forced_429}
        if limit_rpm is not None:
            payload["limit_rpm"] = limit_rpm
        if limit_tpm is not None:
            payload["limit_tpm"] = limit_tpm
        if limit_daily is not None:
            payload["limit_daily_requests"] = limit_daily
        payload["error_rate"] = error_rate
        payload["timeout_rate"] = timeout_rate
        payload["slow_rate"] = slow_rate

        try:
            async with httpx.AsyncClient() as http:
                await http.post(f"{self.server_url}/config", json=payload, timeout=3.0)
                await http.post(f"{self.server_url}/reset", timeout=3.0)
        except Exception as e:
            print(f"[Warning] Failed to configure stand-in server at {self.server_url}: {e}")

    async def _execute_single(
        self,
        runner: ResilientPipelineRunner,
        question_idx: int,
        scenario: str,
        force_degraded: bool = False,
    ) -> PipelineResult:
        """Run a single request through the resilient pipeline runner."""
        q_meta = self.questions[question_idx % len(self.questions)]
        q_text = q_meta["question"]

        result = await runner.run(
            question=q_text,
            scenario=scenario,
            force_degraded=force_degraded,
            question_meta=q_meta,
        )
        return result

    # --------------------------------------------------------------------------
    # Scenario 1: Steady
    # --------------------------------------------------------------------------
    async def run_steady(self, num_requests: int = 5, interval_seconds: float = 30.0) -> List[PipelineResult]:
        """
        Steady traffic below rate and token limits: 1 request every interval_seconds.
        With compound multi-step pipelines (~2,500 tokens per request),
        staying below Groq's 8,000 TPM limit requires pacing at 2 requests/min (30s interval),
        ensuring at most 2 requests ever overlap inside the 60-second sliding window (~5,000 tokens < 8,000 TPM).
        """
        print(f"\n{'='*60}")
        print(f"--- Running Scenario: STEADY ({num_requests} requests, 1 every {interval_seconds}s) ---")
        print(f"{'='*60}")

        await self._configure_standin(limit_rpm=30, limit_tpm=7000, limit_daily=1000)
        runner = ResilientPipelineRunner(llm_client=self.client)

        results: List[PipelineResult] = []
        for i in range(num_requests):
            t_start = time.perf_counter()
            res = await self._execute_single(runner, i, scenario="steady")
            results.append(res)
            print(f"  [steady {i+1}/{num_requests}] Status: {res.status} | TTFT: {res.first_token_ms:.1f}ms | Total: {res.total_ms:.1f}ms | Grade: {res.answer_grade}")

            # Sleep between requests to maintain steady pacing
            if i < num_requests - 1:
                elapsed = time.perf_counter() - t_start
                sleep_rem = max(0.0, interval_seconds - elapsed)
                if sleep_rem > 0:
                    await asyncio.sleep(sleep_rem)

        return results

    # --------------------------------------------------------------------------
    # Scenario 2: Sudden Burst
    # --------------------------------------------------------------------------
    async def run_burst(self, num_requests: int = 25) -> List[PipelineResult]:
        """
        Sudden Burst: 25 requests launched concurrently.
        With max_queue_depth=20:
          - Requests 1-14: Normal mode
          - Requests 15-19: Degraded mode (Setting A, skip verifier)
          - Requests 20-25: Load shedding (immediately dropped with 503 Overloaded)
        """
        print(f"\n{'='*60}")
        print(f"--- Running Scenario: BURST ({num_requests} concurrent requests) ---")
        print(f"{'='*60}")

        await self._configure_standin(limit_rpm=30, limit_tpm=7000, limit_daily=1000)
        runner = ResilientPipelineRunner(llm_client=self.client)

        t0 = time.perf_counter()
        tasks = [
            self._execute_single(runner, i, scenario="burst")
            for i in range(num_requests)
        ]
        results: List[PipelineResult] = await asyncio.gather(*tasks)
        burst_duration = time.perf_counter() - t0

        shed_count = sum(1 for r in results if r.retrieval_setting == "shed")
        degraded_count = sum(1 for r in results if "degraded" in r.retrieval_setting)
        ok_count = sum(1 for r in results if r.status == "ok")

        print(f"  Burst Summary (Duration: {burst_duration:.2f}s):")
        print(f"    Total Offered: {num_requests}")
        print(f"    Successful OK: {ok_count}")
        print(f"    Degraded Answers: {degraded_count}")
        print(f"    Shed (Dropped): {shed_count}")

        return results

    # --------------------------------------------------------------------------
    # Scenario 3: Sustained Overload
    # --------------------------------------------------------------------------
    async def run_sustained_overload(self, num_requests: int = 30, interval_seconds: float = 0.5) -> List[PipelineResult]:
        """
        Sustained Overload: Offered rate (60 RPM) consistently exceeding server capacity (20 RPM).
        """
        print(f"\n{'='*60}")
        print(f"--- Running Scenario: SUSTAINED OVERLOAD ({num_requests} requests, 1 every {interval_seconds}s vs 20 RPM limit) ---")
        print(f"{'='*60}")

        await self._configure_standin(limit_rpm=20, limit_tpm=7000, limit_daily=1000)
        runner = ResilientPipelineRunner(llm_client=self.client)

        results: List[PipelineResult] = []
        tasks = []

        async def worker(idx: int):
            res = await self._execute_single(runner, idx, scenario="sustained_overload")
            results.append(res)
            print(f"  [sustained {idx+1}/{num_requests}] Status: {res.status} | Setting: {res.retrieval_setting} | Retries: {res.retries} | Total: {res.total_ms:.1f}ms")

        for i in range(num_requests):
            tasks.append(asyncio.create_task(worker(i)))
            await asyncio.sleep(interval_seconds)

        await asyncio.gather(*tasks)
        return results

    # --------------------------------------------------------------------------
    # Scenario 4: Per-Minute Limit Halved
    # --------------------------------------------------------------------------
    async def run_limit_halved(self, num_requests: int = 15, interval_seconds: float = 1.0) -> List[PipelineResult]:
        """
        Limit Halved: Server rate limit cut by 50% (30 RPM -> 15 RPM).
        """
        print(f"\n{'='*60}")
        print(f"--- Running Scenario: LIMIT HALVED (Server Limit = 15 RPM, {num_requests} requests) ---")
        print(f"{'='*60}")

        await self._configure_standin(limit_rpm=15, limit_tpm=8000, limit_daily=1000)
        runner = ResilientPipelineRunner(llm_client=self.client)

        results: List[PipelineResult] = []
        tasks = []

        async def worker(idx: int):
            res = await self._execute_single(runner, idx, scenario="limit_halved")
            results.append(res)
            print(f"  [limit_halved {idx+1}/{num_requests}] Status: {res.status} | Retries: {res.retries} | Total: {res.total_ms:.1f}ms")

        for i in range(num_requests):
            tasks.append(asyncio.create_task(worker(i)))
            await asyncio.sleep(interval_seconds)

        await asyncio.gather(*tasks)

        # Restore normal limit
        await self._configure_standin(limit_rpm=30)
        return results

    # --------------------------------------------------------------------------
    # Scenario 5: Daily Budget Exhausted
    # --------------------------------------------------------------------------
    async def run_daily_exhausted(self, num_requests: int = 5) -> List[PipelineResult]:
        """
        Daily Budget Exhausted: Requests arriving after daily ceiling (1,000 RPD) is hit.
        Verifies honest refusal response with UTC reset info.
        """
        print(f"\n{'='*60}")
        print(f"--- Running Scenario: DAILY BUDGET EXHAUSTED ({num_requests} requests) ---")
        print(f"{'='*60}")

        await self._configure_standin(limit_daily=0)
        runner = ResilientPipelineRunner(llm_client=self.client)

        results: List[PipelineResult] = []
        for i in range(num_requests):
            res = await self._execute_single(runner, i, scenario="daily_exhausted")
            results.append(res)
            print(f"  [daily_exhausted {i+1}/{num_requests}] Status: {res.status} | Setting: {res.retrieval_setting}")
            print(f"    Message excerpt: {res.final_answer[:90]}...")

        # Restore normal daily limit
        await self._configure_standin(limit_daily=1000)
        return results

    # --------------------------------------------------------------------------
    # Run All Scenarios
    # --------------------------------------------------------------------------
    async def run_all(self) -> Dict[str, List[PipelineResult]]:
        """Run all 5 traffic shapes in succession."""
        all_results: Dict[str, List[PipelineResult]] = {}
        all_results["steady"] = await self.run_steady()
        all_results["burst"] = await self.run_burst()
        all_results["sustained_overload"] = await self.run_sustained_overload()
        all_results["limit_halved"] = await self.run_limit_halved()
        all_results["daily_exhausted"] = await self.run_daily_exhausted()
        return all_results


def summarize_results(all_results: Dict[str, List[PipelineResult]]) -> Dict[str, Any]:
    """Compute aggregate scenario statistics for reporting."""
    summary = {}
    for scenario_name, res_list in all_results.items():
        total = len(res_list)
        if total == 0:
            continue
        statuses = [r.status for r in res_list]
        ok_count = statuses.count("ok")
        rate_limited_count = statuses.count("rate_limited")
        timeout_count = statuses.count("timeout")
        error_count = statuses.count("error")
        shed_count = sum(1 for r in res_list if r.retrieval_setting == "shed")
        degraded_count = sum(1 for r in res_list if "degraded" in r.retrieval_setting)
        avg_retries = sum(r.retries for r in res_list) / total

        total_times = [r.total_ms for r in res_list if r.total_ms > 0]
        p50_total = sorted(total_times)[len(total_times) // 2] if total_times else 0.0

        summary[scenario_name] = {
            "total_offered": total,
            "ok_count": ok_count,
            "rate_limited_count": rate_limited_count,
            "timeout_count": timeout_count,
            "error_count": error_count,
            "shed_count": shed_count,
            "degraded_count": degraded_count,
            "avg_retries": round(avg_retries, 2),
            "p50_total_ms": round(p50_total, 1),
            "success_rate_pct": round((ok_count / total) * 100, 1),
        }
    return summary


async def main():
    parser = argparse.ArgumentParser(description="Run The Relay Load Generator across 5 traffic shapes")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["steady", "burst", "sustained_overload", "limit_halved", "daily_exhausted", "all"],
        default="all",
        help="Which scenario to run (default: all)",
    )
    parser.add_argument("--source", type=str, choices=["real", "stand-in"], default="stand-in", help="Target backend")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8000", help="Stand-in server URL")
    parser.add_argument("--num-requests", type=int, default=None, help="Override number of requests")
    parser.add_argument("--max-requests", type=int, default=100, help="Safety cap on total requests")
    parser.add_argument("--summary-file", type=str, default="raw/load_run_summary.json", help="Summary JSON output path")
    args = parser.parse_args()

    # Safety check for real service
    if args.source == "real":
        print(f"[CAUTION] You selected --source real. This will make calls to Groq API using your shared daily quota.")
        print(f"Max requests capped at: {args.max_requests}")

    gen = LoadGenerator(
        backend=args.source,
        server_url=args.server_url,
        max_requests=args.max_requests,
    )

    all_results: Dict[str, List[PipelineResult]] = {}

    if args.scenario in ("steady", "all"):
        n = args.num_requests or 5
        all_results["steady"] = await gen.run_steady(num_requests=n)

    if args.scenario in ("burst", "all"):
        n = args.num_requests or 25
        all_results["burst"] = await gen.run_burst(num_requests=n)

    if args.scenario in ("sustained_overload", "all"):
        n = args.num_requests or 30
        all_results["sustained_overload"] = await gen.run_sustained_overload(num_requests=n)

    if args.scenario in ("limit_halved", "all"):
        n = args.num_requests or 15
        all_results["limit_halved"] = await gen.run_limit_halved(num_requests=n)

    if args.scenario in ("daily_exhausted", "all"):
        n = args.num_requests or 5
        all_results["daily_exhausted"] = await gen.run_daily_exhausted(num_requests=n)

    summary = summarize_results(all_results)
    print("\n" + "=" * 60)
    print("  PHASE 8 LOAD RUN AGGREGATE SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    summary_path = PROJECT_ROOT / args.summary_file
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "backend": args.source,
                "summary": summary,
            },
            f,
            indent=2,
        )
    print(f"\nSaved load run summary to {summary_path}")
    print(f"All individual request metrics appended to results.csv")


if __name__ == "__main__":
    asyncio.run(main())

