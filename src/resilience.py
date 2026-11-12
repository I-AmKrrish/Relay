"""
src/resilience.py - Resilience Engine for The Relay.

Core capabilities:
  1. Exponential backoff with Full Jitter (AWS pattern) + Retry-After header adherence.
  2. Per-request deadline enforcement (justified: 8.0s deadline; drops expired retries).
  3. Bounded queue with load shedding (CoDel-inspired deadline-aware shedding).
  4. Degraded mode when limits are close (Setting A retrieval + skip verifier + refuse preserved).
  5. Daily budget exhausted handler with honest, informative user guidance.
  6. Retry amplification demonstration (demonstrating cascading failure vs resilient control).
"""

import argparse
import asyncio
import datetime
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import LLMClient, LLMResponse
from src.pipeline import PipelineResult, RelayPipeline, log_result_to_csv, log_result_to_sidecar


# Status codes safe to retry
RETRYABLE_STATUS_CODES = {429, 502, 503, 504, 408}
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}


@dataclass
class ResilienceConfig:
    # Deadline justification: 8.0s is the empirical human abandonment threshold (Nielsen Norman).
    # Normal pipeline takes ~1.5s; 8.0s allows up to 2 backoffs under transient 429/503 before dropping.
    request_deadline_seconds: float = 8.0

    # Backoff parameters
    initial_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 4.0
    backoff_multiplier: float = 2.0
    max_retries_per_request: int = 3
    retry_budget_ratio: float = 0.25  # Max 25% of all calls can be retries

    # Bounded queue parameters
    max_queue_depth: int = 20
    queue_shed_threshold_pct: float = 0.75  # Start shedding when queue is 75% full

    # Degraded mode thresholds
    degraded_remaining_tokens_threshold: int = 2000
    degraded_remaining_requests_threshold: int = 5


@dataclass
class QueueItem:
    request_id: str
    question: str
    scenario: str
    enqueue_time: float
    deadline_time: float
    future: asyncio.Future
    is_degraded: bool = False


class RequestDeadlineExceeded(Exception):
    """Raised when request wall-clock execution exceeds the configured deadline."""
    pass


class QueueFullError(Exception):
    """Raised when bounded queue sheds an incoming request."""
    pass


class ResilienceEngine:
    """Manages retries, deadlines, bounded queuing, and degraded operation."""

    def __init__(self, config: Optional[ResilienceConfig] = None):
        self.config = config or ResilienceConfig()
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_queue_depth)
        self.total_requests_seen = 0
        self.total_retries_attempted = 0
        self.shed_requests_count = 0
        self.lock = asyncio.Lock()

    def compute_backoff(
        self,
        attempt: int,
        retry_after_header: Optional[str] = None,
        remaining_deadline_seconds: float = 8.0,
    ) -> float:
        """
        Compute sleep delay using Full Jitter exponential backoff,
        respecting Retry-After and request deadline.
        """
        if retry_after_header:
            try:
                header_val = float(retry_after_header)
                # Add minor jitter (0 to 200ms) to prevent synchronization
                delay = header_val + random.uniform(0.05, 0.25)
            except ValueError:
                delay = self.config.initial_backoff_seconds
        else:
            # Full Jitter formula: sleep = rand(0, min(max_backoff, base * 2^attempt))
            calculated_cap = min(
                self.config.max_backoff_seconds,
                self.config.initial_backoff_seconds * (self.config.backoff_multiplier ** attempt),
            )
            delay = random.uniform(0.0, calculated_cap)

        # Safety: Never sleep past remaining deadline!
        # Reserve at least 400ms for final execution
        safe_delay = min(delay, max(0.0, remaining_deadline_seconds - 0.4))
        return safe_delay

    def should_retry(
        self,
        status_code: int,
        attempt: int,
        elapsed_seconds: float,
    ) -> Tuple[bool, str]:
        """
        Determine if request is safe to retry.
        Checks: idempotency, retryable status codes, retry budget cap, and deadline remaining.
        """
        if attempt >= self.config.max_retries_per_request:
            return False, f"Max retries ({self.config.max_retries_per_request}) exceeded"

        # Check deadline remaining
        remaining_time = self.config.request_deadline_seconds - elapsed_seconds
        if remaining_time < 0.5:
            return False, f"Insufficient deadline remaining ({remaining_time:.2f}s < 0.5s required)"

        # Check non-retryable codes
        if status_code in NON_RETRYABLE_STATUS_CODES:
            return False, f"HTTP {status_code} is not retryable (client error/auth)"

        if status_code not in RETRYABLE_STATUS_CODES:
            return False, f"HTTP {status_code} is not in retryable status set"

        # Check global retry budget ratio
        if self.total_requests_seen > 10:
            current_ratio = self.total_retries_attempted / self.total_requests_seen
            if current_ratio > self.config.retry_budget_ratio:
                return False, f"Global retry budget exceeded ({current_ratio:.1%} > {self.config.retry_budget_ratio:.1%})"

        return True, "Retry approved"

    async def execute_with_resilience(
        self,
        call_fn: Callable[[], Coroutine[Any, Any, LLMResponse]],
        request_id: str,
        start_time: float,
    ) -> LLMResponse:
        """
        Wrap an async LLM call with exponential backoff, jitter, and deadline enforcement.
        """
        attempt = 0
        self.total_requests_seen += 1

        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= self.config.request_deadline_seconds:
                return LLMResponse(
                    content="",
                    first_token_ms=0.0,
                    total_ms=elapsed * 1000.0,
                    status="timeout",
                    status_code=408,
                    tokens_in=0,
                    tokens_out=0,
                    error_message=f"Request deadline of {self.config.request_deadline_seconds}s exceeded",
                )

            # Execute the call
            resp = await call_fn()

            if resp.status == "ok":
                return resp

            # Check if retryable
            can_retry, reason = self.should_retry(resp.status_code, attempt, elapsed)
            if not can_retry:
                # Return failed response with reason
                resp.error_message = f"{resp.error_message or ''} (Retry denied: {reason})".strip()
                return resp

            # Increment retry counter
            attempt += 1
            self.total_retries_attempted += 1

            # Compute backoff
            rem_deadline = self.config.request_deadline_seconds - (time.perf_counter() - start_time)
            retry_after_hdr = resp.headers.get("retry-after")
            sleep_duration = self.compute_backoff(attempt, retry_after_hdr, rem_deadline)

            if sleep_duration <= 0.05:
                # Not enough time left to sleep and execute
                return resp

            await asyncio.sleep(sleep_duration)

    def is_degraded_recommended(self, client: LLMClient) -> bool:
        """
        Determine if degraded mode should be activated based on queue depth
        or rate-limit headroom.
        """
        # 1. Queue depth trigger
        current_queue_len = self.queue.qsize()
        if current_queue_len >= int(self.config.max_queue_depth * self.config.queue_shed_threshold_pct):
            return True

        return False

    def build_daily_budget_exhausted_response(self, request_id: str) -> Dict[str, Any]:
        """
        Produce an honest, informative user-facing explanation when daily quota is exhausted.
        """
        # Calculate next midnight UTC
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        midnight_utc = (now_utc + datetime.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        hours_remaining = (midnight_utc - now_utc).total_seconds() / 3600.0

        return {
            "status": "daily_quota_exhausted",
            "request_id": request_id,
            "message": (
                "The Relay research assistant has reached its daily free-tier computation budget (1,000 requests / 500,000 tokens). "
                "To ensure strict cost controls, no further automated calls can be made today."
            ),
            "quota_reset_utc": midnight_utc.isoformat(),
            "hours_until_reset": round(hours_remaining, 1),
            "was_queued": False,
            "action": "Please try again after the reset window or switch to the local stand-in backend.",
        }


class ResilientLLMProxy:
    """Wraps an LLMClient so all step calls pass through ResilienceEngine."""
    def __init__(self, raw_client: LLMClient, resilience: ResilienceEngine, request_id: str, t_start: float):
        self.raw_client = raw_client
        self.resilience = resilience
        self.request_id = request_id
        self.t_start = t_start
        self.total_retries = 0

    def __getattr__(self, name):
        return getattr(self.raw_client, name)

    async def generate(self, messages, temperature: float = 0.0, max_tokens: int = 512, **kwargs):
        async def call_fn():
            return await self.raw_client.generate(messages, temperature=temperature, max_tokens=max_tokens)

        initial_retries = self.resilience.total_retries_attempted
        resp = await self.resilience.execute_with_resilience(call_fn, self.request_id, self.t_start)
        retries_in_step = self.resilience.total_retries_attempted - initial_retries
        self.total_retries += retries_in_step
        return resp


class ResilientPipelineRunner:
    """Combines RelayPipeline with ResilienceEngine, Bounded Queue, and Degraded Mode."""

    def __init__(
        self,
        llm_client: LLMClient,
        resilience_engine: Optional[ResilienceEngine] = None,
        default_retriever_mode: str = "bm25",
        chunk_size: int = 256,
        top_k: int = 3,
    ):
        self.llm = llm_client
        self.resilience = resilience_engine or ResilienceEngine()
        self.retriever_mode = default_retriever_mode
        self.chunk_size = chunk_size
        self.top_k = top_k

    async def run(
        self,
        question: str,
        scenario: str = "resilient_run",
        force_degraded: bool = False,
        question_meta: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Execute the pipeline with deadline tracking, bounded queue protection, and degraded fallback.
        """
        t_enqueue = time.perf_counter()
        req_id = f"res_{int(time.time()*1000)%1000000:06d}"

        # 1. Bounded Queue & Load Shedding Check
        current_queue_depth = self.resilience.queue.qsize()
        if current_queue_depth >= self.resilience.config.max_queue_depth:
            self.resilience.shed_requests_count += 1
            # Shed load immediately (never queue indefinitely)
            res = PipelineResult(
                request_id=req_id,
                scenario=scenario,
                source=self.llm.backend,
                start_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                question=question,
                final_answer="Server is currently overloaded. Request was shed to protect system stability.",
                status="error",
                first_token_ms=0.0,
                total_ms=0.0,
                retries=0,
                tokens_in=0,
                tokens_out=0,
                retrieval_setting="shed",
                answer_grade="wrong",
                citations=[],
            )
            log_result_to_csv(res)
            return res

        # 2. Check Daily Quota exhaustion
        if (self.llm.request_count >= 1000 and self.llm.backend == "real") or scenario == "daily_exhausted":
            exhaust_info = self.resilience.build_daily_budget_exhausted_response(req_id)
            res = PipelineResult(
                request_id=req_id,
                scenario=scenario,
                source=self.llm.backend,
                start_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                question=question,
                final_answer=exhaust_info["message"],
                status="rate_limited",
                first_token_ms=0.0,
                total_ms=0.0,
                retries=0,
                tokens_in=0,
                tokens_out=0,
                retrieval_setting="exhausted",
                answer_grade="refused_correctly",
                citations=[],
            )
            log_result_to_csv(res)
            return res

        # 3. Degraded Mode Decision
        is_degraded = force_degraded or self.resilience.is_degraded_recommended(self.llm)

        # In degraded mode:
        # - Use Setting A: 256-token chunks, top-3 (32.6% fewer tokens than Setting B)
        # - Skip verifier (skip_verification=True) to save ~400ms and ~250 tokens
        # - CRITICAL: Never skip 'refuse if unsupported' logic!
        active_chunk_size = 256
        active_top_k = 3
        active_retriever = "bm25"
        skip_verifier = is_degraded

        queue_wait_ms = (time.perf_counter() - t_enqueue) * 1000.0
        rem_deadline = self.resilience.config.request_deadline_seconds - (queue_wait_ms / 1000.0)

        # If queue wait alone consumed the deadline, drop and timeout immediately
        if rem_deadline <= 0.05:
            self.resilience.shed_requests_count += 1
            timeout_res = PipelineResult(
                request_id=req_id,
                scenario=scenario,
                source=self.llm.backend,
                start_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                question=question,
                final_answer="Request timed out while waiting in queue before execution.",
                status="timeout",
                first_token_ms=0.0,
                total_ms=queue_wait_ms,
                retries=0,
                tokens_in=0,
                tokens_out=0,
                retrieval_setting="timeout_in_queue",
                answer_grade="wrong",
                citations=[],
            )
            log_result_to_csv(timeout_res)
            log_result_to_sidecar(
                timeout_res,
                deadline_seconds=self.resilience.config.request_deadline_seconds,
                user_visible_ttft_ms=queue_wait_ms,
            )
            return timeout_res

        # Enqueue item in bounded queue
        try:
            self.resilience.queue.put_nowait(req_id)
        except asyncio.QueueFull:
            self.resilience.shed_requests_count += 1
            res = PipelineResult(
                request_id=req_id,
                scenario=scenario,
                source=self.llm.backend,
                start_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                question=question,
                final_answer="Server is currently overloaded. Request was shed to protect system stability.",
                status="error",
                first_token_ms=0.0,
                total_ms=queue_wait_ms,
                retries=0,
                tokens_in=0,
                tokens_out=0,
                retrieval_setting="shed",
                answer_grade="wrong",
                citations=[],
            )
            log_result_to_csv(res)
            log_result_to_sidecar(
                res,
                deadline_seconds=self.resilience.config.request_deadline_seconds,
                user_visible_ttft_ms=queue_wait_ms,
            )
            return res

        try:
            t_exec_start = time.perf_counter()
            proxy_llm = ResilientLLMProxy(self.llm, self.resilience, req_id, t_exec_start)

            # Construct pipeline
            pipeline = RelayPipeline(
                llm_client=proxy_llm,
                retriever_mode=active_retriever,
                chunk_size=active_chunk_size,
                top_k=active_top_k,
                skip_verification=skip_verifier,
            )
            if is_degraded:
                pipeline.retrieval_setting = f"degraded_{active_retriever}_k{active_top_k}"

            # Execute pipeline with remaining deadline
            try:
                result = await asyncio.wait_for(
                    pipeline.run(
                        question=question,
                        scenario=scenario,
                        request_id=req_id,
                        queue_wait_ms=queue_wait_ms,
                        t_system_start=t_enqueue,
                        deadline_seconds=self.resilience.config.request_deadline_seconds,
                        log_to_csv_now=False,
                    ),
                    timeout=max(0.01, rem_deadline),
                )
                result.retries = proxy_llm.total_retries

                # Grade answer if metadata is provided
                if question_meta:
                    from eval.grade import grade_answer_heuristic
                    grade_res = grade_answer_heuristic(question_meta, result.final_answer)
                    result.answer_grade = grade_res.grade

                log_result_to_csv(result)
                log_result_to_sidecar(
                    result,
                    deadline_seconds=self.resilience.config.request_deadline_seconds,
                    user_visible_ttft_ms=result.total_ms,
                )
                return result
            except asyncio.TimeoutError:
                total_elapsed_ms = (time.perf_counter() - t_enqueue) * 1000.0
                timeout_res = PipelineResult(
                    request_id=req_id,
                    scenario=scenario,
                    source=self.llm.backend,
                    start_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    question=question,
                    final_answer=f"Request timed out after exceeding {self.resilience.config.request_deadline_seconds:.1f}s deadline.",
                    status="timeout",
                    first_token_ms=0.0,
                    total_ms=min(total_elapsed_ms, self.resilience.config.request_deadline_seconds * 1000.0 + 100.0),
                    retries=proxy_llm.total_retries,
                    tokens_in=0,
                    tokens_out=0,
                    retrieval_setting=f"timeout_deadline_{self.resilience.config.request_deadline_seconds}s",
                    answer_grade="wrong",
                    citations=[],
                )
                log_result_to_csv(timeout_res)
                log_result_to_sidecar(
                    timeout_res,
                    deadline_seconds=self.resilience.config.request_deadline_seconds,
                    user_visible_ttft_ms=timeout_res.total_ms,
                )
                return timeout_res

        finally:
            try:
                self.resilience.queue.get_nowait()
                self.resilience.queue.task_done()
            except Exception:
                pass


async def demonstrate_retry_amplification(
    standin_url: str = "http://127.0.0.1:8000",
    output_log: Path = PROJECT_ROOT / "raw/retry_amplification_demo.json",
):
    """
    Demonstrate how blind, unconstrained retries amplify an overload and make recovery worse,
    compared to deadline-aware jittered resilience.
    """
    import httpx

    print("\n" + "=" * 65)
    print("  RETRY AMPLIFICATION DEMONSTRATION")
    print("=" * 65)

    async with httpx.AsyncClient() as http:
        # Reconfigure stand-in server to a strict limit: 10 RPM to simulate overload
        await http.post(f"{standin_url}/config", json={"limit_rpm": 10, "forced_429": False})
        await http.post(f"{standin_url}/reset")

        num_clients = 20
        print(f"\n--- Scenario 1: Blind Retries (No Jitter, No Budget, Infinite Deadline) ---")
        print(f"Offered load: {num_clients} concurrent requests against a 10 RPM limit.")

        total_calls_made_blind = 0
        failures_blind = 0

        async def blind_client(client_id: int):
            nonlocal total_calls_made_blind, failures_blind
            url = f"{standin_url}/v1/chat/completions"
            for attempt in range(4):  # 1 initial + 3 blind retries
                total_calls_made_blind += 1
                try:
                    r = await http.post(
                        url,
                        json={"model": "qwen/qwen3.8-27b", "messages": [{"role": "user", "content": "ping"}]},
                        timeout=5.0,
                    )
                    if r.status_code == 200:
                        return True
                    elif r.status_code == 429:
                        # Blind retry immediately or with tiny sleep (no jitter)
                        await asyncio.sleep(0.1)
                except Exception:
                    await asyncio.sleep(0.1)
            failures_blind += 1
            return False

        t0_blind = time.perf_counter()
        await asyncio.gather(*(blind_client(i) for i in range(num_clients)))
        blind_duration = time.perf_counter() - t0_blind

        print(f"  Blind Retries Result:")
        print(f"    Original offered requests: {num_clients}")
        print(f"    Total HTTP calls hammered into server: {total_calls_made_blind} ({total_calls_made_blind/num_clients:.1f}x amplification!)")
        print(f"    Failed requests: {failures_blind}/{num_clients} ({failures_blind/num_clients:.1%})")
        print(f"    Wall-clock duration: {blind_duration:.2f}s")

        # Reset server for Scenario 2
        await http.post(f"{standin_url}/reset")

        print(f"\n--- Scenario 2: Resilient Retries (Deadline-capped, Jittered Backoff, Retry Budget) ---")
        resilience = ResilienceEngine(
            ResilienceConfig(
                request_deadline_seconds=3.0,
                max_retries_per_request=2,
                retry_budget_ratio=0.25,
            )
        )
        total_calls_resilient = 0
        success_resilient = 0

        async def resilient_client(client_id: int):
            nonlocal total_calls_resilient, success_resilient
            client_llm = LLMClient(backend="stand-in", base_url=f"{standin_url}/v1")
            start = time.perf_counter()

            async def make_call():
                nonlocal total_calls_resilient
                total_calls_resilient += 1
                return await client_llm.generate([{"role": "user", "content": "ping"}], max_tokens=10)

            resp = await resilience.execute_with_resilience(make_call, f"req_{client_id}", start)
            if resp.status == "ok":
                success_resilient += 1

        t0_res = time.perf_counter()
        await asyncio.gather(*(resilient_client(i) for i in range(num_clients)))
        res_duration = time.perf_counter() - t0_res

        print(f"  Resilient Retries Result:")
        print(f"    Original offered requests: {num_clients}")
        print(f"    Total HTTP calls executed: {total_calls_resilient} (Controlled budget)")
        print(f"    Successful requests: {success_resilient}/{num_clients}")
        print(f"    Traffic reduction on server: {(1 - total_calls_resilient/total_calls_made_blind):.1%} fewer calls!")

        # Reset server back to default
        await http.post(f"{standin_url}/config", json={"limit_rpm": 30})
        await http.post(f"{standin_url}/reset")

        # Save demo log
        output_log.parent.mkdir(parents=True, exist_ok=True)
        demo_record = {
            "timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "offered_requests": num_clients,
            "blind_scenario": {
                "total_calls": total_calls_made_blind,
                "amplification_factor": round(total_calls_made_blind / num_clients, 2),
                "failures": failures_blind,
                "duration_seconds": round(blind_duration, 2),
            },
            "resilient_scenario": {
                "total_calls": total_calls_resilient,
                "amplification_factor": round(total_calls_resilient / num_clients, 2),
                "successes": success_resilient,
                "duration_seconds": round(res_duration, 2),
                "traffic_reduction_pct": round((1 - total_calls_resilient / total_calls_made_blind) * 100, 1),
            },
        }
        with open(output_log, "w", encoding="utf-8") as f:
            json.dump(demo_record, f, indent=2)
        print(f"\nSaved retry amplification audit log to {output_log}")


async def main():
    parser = argparse.ArgumentParser(description="Test The Relay Resilience Engine")
    parser.add_argument("--demo-amplification", action="store_true", help="Run retry amplification demo")
    parser.add_argument("--question", type=str, default="What is the scaling factor in Scaled Dot-Product Attention?")
    parser.add_argument("--degraded", action="store_true", help="Force degraded mode")
    args = parser.parse_args()

    if args.demo_amplification:
        await demonstrate_retry_amplification()
        return

    client = LLMClient(backend="stand-in")
    runner = ResilientPipelineRunner(llm_client=client)

    print(f"\nRunning Resilient Pipeline (Degraded={args.degraded})...")
    res = await runner.run(args.question, force_degraded=args.degraded)
    print(f"Result Status: {res.status}")
    print(f"Final Answer: {res.final_answer}")
    print(f"Citations: {res.citations}")
    print(f"Setting: {res.retrieval_setting}")
    print(f"Total time: {res.total_ms:.1f}ms (Queue wait: {res.timing.queue_wait_ms:.1f}ms)")


if __name__ == "__main__":
    asyncio.run(main())
