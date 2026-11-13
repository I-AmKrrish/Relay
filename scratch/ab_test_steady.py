"""
scratch/ab_test_steady.py - Controlled A/B Isolation Test for Steady Workload at 2.6 req/min.

Isolates:
  Condition 1: Sync / Blocking retrieval (pre-fix architecture: lazy model load, synchronous retrieval on event loop)
  Condition 2: Async / Threaded retrieval (post-fix architecture: preloaded models, asyncio.to_thread with semaphore)

Both conditions run at the EXACT same offered rate:
  - 2.6 requests/minute (1 request every 23.08 seconds)
  - 15 requests each (N = 15)
  - Tested against the calibrated stand-in server (7,000 ITPM, 30 RPM, 1,000 RPD)

Outputs:
  - data/ab_steady_isolation_summary.json
  - Updates results.csv with clean N=15 steady scenario
"""

import argparse
import asyncio
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import LLMClient
from src.pipeline import RelayPipeline, PipelineResult
from src.resilience import ResilientPipelineRunner
from src import retrieval


STANDIN_URL = "http://localhost:8000"
RESULTS_CSV = PROJECT_ROOT / "results.csv"


async def reset_standin():
    async with httpx.AsyncClient() as client:
        await client.post(f"{STANDIN_URL}/config", json={
            "limit_rpm": 30,
            "limit_tpm": 7000,
            "limit_daily_requests": 1000,
            "error_rate": 0.0,
            "timeout_rate": 0.0,
            "slow_rate": 0.0,
            "forced_429": False,
        })
        await client.post(f"{STANDIN_URL}/reset")


class EventLoopMonitor:
    """Monitors asyncio event loop latency to detect CPU blocking."""
    def __init__(self, interval_ms: float = 10.0):
        self.interval_s = interval_ms / 1000.0
        self.running = False
        self.lags_ms: List[float] = []
        self._task = None

    async def _monitor(self):
        while self.running:
            t0 = time.perf_counter()
            await asyncio.sleep(self.interval_s)
            actual_delay = (time.perf_counter() - t0) * 1000.0
            lag = max(0.0, actual_delay - (self.interval_s * 1000.0))
            self.lags_ms.append(lag)

    def start(self):
        self.running = True
        self.lags_ms = []
        self._task = asyncio.create_task(self._monitor())

    def stop(self) -> Dict[str, float]:
        self.running = False
        if self._task:
            self._task.cancel()
        if not self.lags_ms:
            return {"mean_lag_ms": 0.0, "max_lag_ms": 0.0, "p99_lag_ms": 0.0}
        arr = np.array(self.lags_ms)
        return {
            "mean_lag_ms": float(np.mean(arr)),
            "max_lag_ms": float(np.max(arr)),
            "p99_lag_ms": float(np.percentile(arr, 99)),
        }


def load_benchmark_questions():
    with open(PROJECT_ROOT / "eval/questions.json", "r", encoding="utf-8") as f:
        return json.load(f)


async def run_condition(
    name: str,
    scenario_tag: str,
    sync_mode: bool,
    num_requests: int,
    interval_seconds: float,
) -> Tuple[List[PipelineResult], Dict[str, Any]]:
    print(f"\n{'='*70}")
    print(f"  RUNNING CONDITION: {name}")
    print(f"  Mode: {'SYNCHRONOUS (Pre-fix)' if sync_mode else 'ASYNC THREADED (Post-fix)'}")
    print(f"  N = {num_requests}, Offered Rate = {60.0/interval_seconds:.1f} req/m (interval = {interval_seconds:.2f}s)")
    print(f"{'='*70}")

    await reset_standin()
    questions = load_benchmark_questions()

    # Configure preloading & step_2_retrieve patching
    original_step_2_retrieve = RelayPipeline.step_2_retrieve

    if sync_mode:
        print("[Setup] Sync mode: clearing model cache to simulate cold-start / unpatched state...")
        retrieval._MODEL_CACHE.clear()
        retrieval._RETRIEVER_INSTANCE_CACHE.clear()

        # Synchronous blocking retrieval monkey patch
        async def sync_step_2_retrieve(self, query: str, top_k=None):
            k = top_k or self.top_k
            return self.retriever.retrieve(query, top_k=k)

        RelayPipeline.step_2_retrieve = sync_step_2_retrieve
    else:
        print("[Setup] Async mode: preloading models into memory...")
        retrieval.preload_all_models()
        RelayPipeline.step_2_retrieve = original_step_2_retrieve

    client = LLMClient(backend="stand-in", base_url=f"{STANDIN_URL}/v1")
    runner = ResilientPipelineRunner(llm_client=client)

    monitor = EventLoopMonitor()
    monitor.start()

    results: List[PipelineResult] = []
    t_start_total = time.perf_counter()

    try:
        for i in range(num_requests):
            q_meta = questions[i % len(questions)]
            q_text = q_meta["question"]
            t_req_start = time.perf_counter()

            # Execute pipeline
            res = await runner.run(
                question=q_text,
                scenario=scenario_tag,
                question_meta=q_meta,
            )
            results.append(res)
            print(f"  [{name} {i+1:02d}/{num_requests}] status={res.status:<12} | total={res.total_ms:6.1f}ms | ttft={res.first_token_ms:6.1f}ms | tokens_in={res.tokens_in:4d} | grade={res.answer_grade}")

            # Sleep remaining time to hit exact interval
            if i < num_requests - 1:
                elapsed = time.perf_counter() - t_req_start
                sleep_rem = max(0.0, interval_seconds - elapsed)
                if sleep_rem > 0:
                    await asyncio.sleep(sleep_rem)

    finally:
        lag_stats = monitor.stop()
        RelayPipeline.step_2_retrieve = original_step_2_retrieve  # Restore original

    wall_duration = time.perf_counter() - t_start_total

    # Calculate statistics
    n = len(results)
    ok_cnt = sum(1 for r in results if r.status == "ok")
    rl_cnt = sum(1 for r in results if r.status == "rate_limited")
    to_cnt = sum(1 for r in results if r.status == "timeout")
    err_cnt = sum(1 for r in results if r.status == "error")

    totals = [r.total_ms for r in results]
    ttfts = [r.first_token_ms for r in results if r.first_token_ms > 0]

    stats = {
        "condition": name,
        "sync_mode": sync_mode,
        "n": n,
        "wall_duration_s": wall_duration,
        "offered_rate_rpm": (n / wall_duration) * 60.0,
        "success_pct": (ok_cnt / n) * 100.0,
        "rate_limited_pct": (rl_cnt / n) * 100.0,
        "timeout_pct": (to_cnt / n) * 100.0,
        "error_pct": (err_cnt / n) * 100.0,
        "total_p50_ms": float(np.median(totals)) if totals else 0.0,
        "total_p95_ms": float(np.percentile(totals, 95)) if totals else 0.0,
        "total_p99_ms": float(np.percentile(totals, 99)) if totals else 0.0,
        "ttft_p50_ms": float(np.median(ttfts)) if ttfts else 0.0,
        "ttft_p95_ms": float(np.percentile(ttfts, 95)) if ttfts else 0.0,
        "ttft_p99_ms": float(np.percentile(ttfts, 99)) if ttfts else 0.0,
        "mean_tokens_in": float(np.mean([r.tokens_in for r in results])),
        "mean_tokens_out": float(np.mean([r.tokens_out for r in results])),
        "loop_lag": lag_stats,
    }

    return results, stats


def clean_old_steady_from_results():
    if not RESULTS_CSV.exists():
        return
    df = pd.read_csv(RESULTS_CSV)
    # Remove old steady rows (and any temporary test rows)
    df_clean = df[~df["scenario"].isin(["steady", "test_steady_sync", "steady_sync_baseline"])].copy()
    df_clean.to_csv(RESULTS_CSV, index=False)
    print(f"[Cleanup] Removed old steady/test rows. Remaining rows: {len(df_clean)}")


async def main():
    parser = argparse.ArgumentParser(description="Run A/B steady workload test")
    parser.add_argument("--dry-run", action="store_true", help="Run 2 requests per condition for testing")
    parser.add_argument("--requests", type=int, default=15, help="Number of requests per condition")
    parser.add_argument("--interval", type=float, default=60.0 / 2.6, help="Interval between requests (s)")
    args = parser.parse_args()

    num_reqs = 2 if args.dry_run else args.requests
    interval = 1.0 if args.dry_run else args.interval

    print("=" * 70)
    print("  A/B ISOLATION TEST: STEADY WORKLOAD AT 2.6 REQ/MIN")
    print(f"  Requests per condition: {num_reqs} | Interval: {interval:.2f}s")
    print("=" * 70)

    # Condition 1: Sync (Pre-fix architecture)
    results_sync, stats_sync = await run_condition(
        name="Condition 1 (Sync Pre-Fix)",
        scenario_tag="steady_sync_baseline",
        sync_mode=True,
        num_requests=num_reqs,
        interval_seconds=interval,
    )

    # Cooldown before Condition 2
    print("\n[Cooldown] Waiting 5 seconds before Condition 2...")
    await asyncio.sleep(5.0)

    # Before Condition 2, clean old steady rows if this is a full benchmark run
    if not args.dry_run:
        clean_old_steady_from_results()

    # Condition 2: Async + Preloaded (Post-fix architecture)
    results_async, stats_async = await run_condition(
        name="Condition 2 (Async Post-Fix)",
        scenario_tag="steady",
        sync_mode=False,
        num_requests=num_reqs,
        interval_seconds=interval,
    )

    summary = {
        "sync_condition": stats_sync,
        "async_condition": stats_async,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    out_file = PROJECT_ROOT / "data/ab_steady_isolation_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("  A/B TEST COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Condition 1 (Sync)':<20} | {'Condition 2 (Async)':<20}")
    print("-" * 75)
    print(f"{'Success %':<30} | {stats_sync['success_pct']:>18.1f}% | {stats_async['success_pct']:>18.1f}%")
    print(f"{'Timeout %':<30} | {stats_sync['timeout_pct']:>18.1f}% | {stats_async['timeout_pct']:>18.1f}%")
    print(f"{'Rate Limited %':<30} | {stats_sync['rate_limited_pct']:>18.1f}% | {stats_async['rate_limited_pct']:>18.1f}%")
    print(f"{'Total Latency p50 (ms)':<30} | {stats_sync['total_p50_ms']:>18.1f} | {stats_async['total_p50_ms']:>18.1f}")
    print(f"{'Total Latency p95 (ms)':<30} | {stats_sync['total_p95_ms']:>18.1f} | {stats_async['total_p95_ms']:>18.1f}")
    print(f"{'Event Loop Mean Lag (ms)':<30} | {stats_sync['loop_lag']['mean_lag_ms']:>18.2f} | {stats_async['loop_lag']['mean_lag_ms']:>18.2f}")
    print(f"{'Event Loop Max Lag (ms)':<30} | {stats_sync['loop_lag']['max_lag_ms']:>18.2f} | {stats_async['loop_lag']['max_lag_ms']:>18.2f}")
    print(f"{'Event Loop p99 Lag (ms)':<30} | {stats_sync['loop_lag']['p99_lag_ms']:>18.2f} | {stats_async['loop_lag']['p99_lag_ms']:>18.2f}")
    print("=" * 75)
    print(f"Summary saved to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
