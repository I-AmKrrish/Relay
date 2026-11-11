"""
standin/server.py - Calibrated Stand-in Model Server for The Relay.

Simulates the Groq model service (qwen/qwen3.8-27b) using FastAPI and uvicorn.
Features:
  1. Calibrated Latency: TTFT sampled from log-normal distribution fitted to Phase 4 real Groq probes (mu=5.95, sigma=0.15).
  2. Streaming Tokens: SSE chunks (data: {...}) simulating token-by-token generation at ~300 tok/s.
  3. Dynamic Concurrency Slowdown: Adds latency penalty when concurrent in-flight requests exceed 3.
  4. Fault Injection: Configurable error_rate (500s), timeout_rate (hanging), and slow_rate (tail latency).
  5. Realistic Rate Limiting: Enforces RPM, TPM, and daily limits, returning 429 + Retry-After + Groq-format error JSON.
  6. Configurable via CLI arguments and runtime /config endpoint.
"""

import argparse
import asyncio
import json
import math
import random
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, AsyncGenerator, Deque, Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Global Server Configuration
class ServerConfig:
    # Latency calibration (fitted to Phase 4 empirical Groq measurements)
    ttft_lognorm_mu: float = 5.95  # exp(5.95) ≈ 384 ms
    ttft_lognorm_sigma: float = 0.15  # range roughly 290 - 510 ms
    token_gen_rate_tok_per_sec: float = 300.0  # ~3.3 ms per token
    concurrency_slowdown_alpha: float = 0.15  # 15% latency penalty per in-flight request beyond 3

    # Rate limits (calibrated to Groq free tier)
    limit_rpm: int = 30
    limit_tpm: int = 8000
    limit_daily_requests: int = 1000

    # Fault injection
    error_rate: float = 0.0  # Probability of HTTP 500
    timeout_rate: float = 0.0  # Probability of hanging >30s
    slow_rate: float = 0.0  # Probability of 3x-5x latency spike
    forced_429: bool = False  # Always 429 if True


config = ServerConfig()

# In-flight concurrency and rate limit state
in_flight_count = 0
in_flight_lock = asyncio.Lock()
request_history: Deque[Tuple[float, int]] = deque()  # (timestamp, token_count)
daily_request_history: Deque[float] = deque()  # timestamps in last 24h


app = FastAPI(title="The Relay Stand-in Model Service", version="1.0.0")


class ChatCompletionRequest(BaseModel):
    model: str = "qwen/qwen3.8-27b"
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None


def check_rate_limits(estimated_tokens: int) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Check current request against RPM, TPM, and daily request limits.
    Returns: (is_limited, retry_after_seconds, reason_string)
    """
    now = time.time()

    if config.forced_429:
        return True, 15, "Forced 429 testing mode enabled."

    # Clean history older than 60 seconds
    while request_history and request_history[0][0] < now - 60.0:
        request_history.popleft()

    # Clean daily history older than 24 hours (86400 seconds)
    while daily_request_history and daily_request_history[0] < now - 86400.0:
        daily_request_history.popleft()

    # 1. Daily limit check
    if len(daily_request_history) >= config.limit_daily_requests:
        oldest = daily_request_history[0]
        retry_after = max(1, int(math.ceil(oldest + 86400.0 - now)))
        return True, retry_after, f"Daily request limit reached ({config.limit_daily_requests} RPD)."

    # 2. RPM check
    current_rpm = len(request_history)
    if current_rpm >= config.limit_rpm:
        oldest = request_history[0][0]
        retry_after = max(1, int(math.ceil(oldest + 60.0 - now)))
        return True, retry_after, f"Rate limit reached on requests per minute (RPM): Limit {config.limit_rpm}, Used {current_rpm}."

    # 3. TPM check
    current_tpm = sum(tok for _, tok in request_history)
    if current_tpm + estimated_tokens > config.limit_tpm:
        oldest = request_history[0][0]
        retry_after = max(1, int(math.ceil(oldest + 60.0 - now)))
        return True, retry_after, (
            f"Rate limit reached on tokens per minute (TPM): "
            f"Limit {config.limit_tpm}, Used {current_tpm}, Requested {estimated_tokens}."
        )

    # Record accepted request
    request_history.append((now, estimated_tokens))
    daily_request_history.append(now)
    return False, None, None


def synthesize_completion_content(messages: List[Dict[str, str]]) -> str:
    """Generate grounded, realistic text matching the pipeline prompt."""
    prompt = messages[-1].get("content", "")
    sys_prompt = messages[0].get("content", "") if messages else ""

    # Pipeline Step 1: Understand
    if "Extract the core technical concepts" in sys_prompt:
        # Extract meaningful terms from user question
        terms = re.findall(r"\b[A-Za-z0-9_\-]+\b", prompt)
        filtered = [t for t in terms if len(t) > 3 and t.lower() not in ("what", "which", "where", "explain", "model")][:6]
        return " ".join(filtered) if filtered else "transformer sequence modeling"

    # Pipeline Step 3: Decide
    if "Determine if the provided passages" in sys_prompt:
        unanswerable_kws = ["llama", "whisper", "gpu hours", "point cloud", "chinese", "mars", "weather", "capital"]
        if any(k in prompt.lower() for k in unanswerable_kws):
            return json.dumps({
                "decision": "unsupported",
                "reason": "Topic is not covered in the library papers.",
                "alternate_query": ""
            })
        return json.dumps({
            "decision": "sufficient",
            "reason": "Retrieved passages contain the required facts.",
            "alternate_query": ""
        })

    # Pipeline Step 5: Verify
    if "strict fact-checker" in sys_prompt:
        return json.dumps({"passed": True, "critique": ""})

    # Pipeline Step 4: Draft
    unanswerable_kws = ["llama", "whisper", "gpu hours", "point cloud", "chinese", "mars", "weather", "capital"]
    if any(k in prompt.lower() for k in unanswerable_kws) or "unsupported" in prompt.lower():
        return "The papers do not say."

    # Extract citations present in prompt passages
    cits = re.findall(r"\[[^\]]+?,\s*p\.\s*\d+\]", prompt)
    cit = cits[0] if cits else "[Attention Is All You Need, Sec 1 Introduction, p. 2]"

    # Extract first informative technical sentence from passages
    passage_lines = [
        l.strip()
        for l in prompt.split("\n")
        if len(l.strip()) > 35 and not l.startswith("Passage") and not l.startswith("[")
    ]
    fact_sentence = passage_lines[0] if passage_lines else "The architectural parameters and evaluation metrics are described in the library."
    return f"{fact_sentence} {cit}."


async def stream_generator(
    content: str,
    ttft_ms: float,
    model: str,
    tokens_in: int,
    tokens_out: int,
    req_id: str,
    include_usage: bool = True,
) -> AsyncGenerator[str, None]:
    """Yield simulated SSE streaming tokens."""
    # 1. Initial TTFT delay
    await asyncio.sleep(ttft_ms / 1000.0)

    words = content.split(" ")
    created_ts = int(time.time())

    # 2. Token-by-token streaming
    token_delay = 1.0 / config.token_gen_rate_tok_per_sec  # ~3.3 ms per token
    for idx, word in enumerate(words):
        token_piece = word + (" " if idx < len(words) - 1 else "")
        chunk_obj = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token_piece},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk_obj)}\n\n"
        await asyncio.sleep(token_delay)

    # 3. Final chunk with finish_reason and optional usage
    final_chunk = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    if include_usage:
        final_chunk["usage"] = {
            "prompt_tokens": tokens_in,
            "completion_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out,
        }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, raw_req: Request):
    global in_flight_count

    # Calculate input tokens
    prompt_words = sum(len(m.get("content", "").split()) for m in req.messages)
    tokens_in = max(10, int(prompt_words * 1.3))
    tokens_out_est = req.max_tokens or 64

    # 1. Rate Limit Gate
    is_limited, retry_after, reason = check_rate_limits(tokens_in + tokens_out_est)
    if is_limited:
        headers = {
            "retry-after": str(retry_after),
            "x-ratelimit-limit-requests": str(config.limit_rpm),
            "x-ratelimit-limit-tokens": str(config.limit_tpm),
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-requests": f"{retry_after}s",
            "x-ratelimit-reset-tokens": f"{retry_after}s",
            "content-type": "application/json",
        }
        error_body = {
            "error": {
                "message": reason,
                "type": "tokens" if "tokens" in (reason or "").lower() else "requests",
                "code": "rate_limit_exceeded",
            }
        }
        return JSONResponse(status_code=429, content=error_body, headers=headers)

    # 2. Fault Injection: On-demand HTTP 500 error
    if config.error_rate > 0.0 and random.random() < config.error_rate:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Simulated internal server error (fault injection)", "type": "server_error"}},
        )

    # 3. Fault Injection: On-demand Timeout (hangs past client deadline)
    if config.timeout_rate > 0.0 and random.random() < config.timeout_rate:
        await asyncio.sleep(40.0)  # Exceeds default 30s client timeout
        return JSONResponse(status_code=408, content={"error": {"message": "Simulated server timeout"}})

    # Track in-flight concurrency
    async with in_flight_lock:
        in_flight_count += 1
        current_concurrency = in_flight_count

    try:
        # 4. Latency sampling (Log-normal distribution calibrated to real Groq probes)
        base_ttft_ms = max(150.0, random.lognormvariate(config.ttft_lognorm_mu, config.ttft_lognorm_sigma))

        # Fault Injection: "Succeeds slowly" tail latency spike
        if config.slow_rate > 0.0 and random.random() < config.slow_rate:
            base_ttft_ms *= random.uniform(3.0, 5.0)

        # Dynamic Concurrency Slowdown (for in-flight > 3)
        excess = max(0, current_concurrency - 3)
        slowdown_multiplier = 1.0 + (excess * config.concurrency_slowdown_alpha)
        calibrated_ttft_ms = base_ttft_ms * slowdown_multiplier

        # Synthesize output content
        content = synthesize_completion_content(req.messages)
        tokens_out = max(1, len(content.split()))

        req_id = f"chatcmpl-standin-{time.time_ns()}"
        include_usage = bool(req.stream_options and req.stream_options.get("include_usage"))

        # Remaining header calculations
        now = time.time()
        rem_requests = max(0, config.limit_rpm - len(request_history))
        rem_tokens = max(0, config.limit_tpm - sum(t for _, t in request_history))
        headers = {
            "x-ratelimit-limit-requests": str(config.limit_rpm),
            "x-ratelimit-limit-tokens": str(config.limit_tpm),
            "x-ratelimit-remaining-requests": str(rem_requests),
            "x-ratelimit-remaining-tokens": str(rem_tokens),
            "x-ratelimit-reset-requests": "2.5s",
            "x-ratelimit-reset-tokens": "500ms",
        }

        if req.stream:
            return StreamingResponse(
                stream_generator(
                    content=content,
                    ttft_ms=calibrated_ttft_ms,
                    model=req.model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    req_id=req_id,
                    include_usage=include_usage,
                ),
                media_type="text/event-stream",
                headers=headers,
            )
        else:
            # Non-streaming response
            total_delay = calibrated_ttft_ms + (tokens_out / config.token_gen_rate_tok_per_sec * 1000.0)
            await asyncio.sleep(total_delay / 1000.0)
            body = {
                "id": req_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": tokens_in,
                    "completion_tokens": tokens_out,
                    "total_tokens": tokens_in + tokens_out,
                },
            }
            return JSONResponse(status_code=200, content=body, headers=headers)

    finally:
        async with in_flight_lock:
            in_flight_count = max(0, in_flight_count - 1)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "in_flight": in_flight_count,
        "config": {
            "limit_rpm": config.limit_rpm,
            "limit_tpm": config.limit_tpm,
            "limit_daily_requests": config.limit_daily_requests,
            "error_rate": config.error_rate,
            "timeout_rate": config.timeout_rate,
            "slow_rate": config.slow_rate,
            "forced_429": config.forced_429,
        },
    }


class ConfigUpdateRequest(BaseModel):
    limit_rpm: Optional[int] = None
    limit_tpm: Optional[int] = None
    limit_daily_requests: Optional[int] = None
    error_rate: Optional[float] = None
    timeout_rate: Optional[float] = None
    slow_rate: Optional[float] = None
    forced_429: Optional[bool] = None


@app.post("/config")
def update_config(update: ConfigUpdateRequest):
    """Runtime dynamic reconfiguration of limits and fault injection."""
    for field_name, value in update.model_dump(exclude_unset=True).items():
        if hasattr(config, field_name):
            setattr(config, field_name, value)
    return {"status": "updated", "config": health()["config"]}


@app.post("/reset")
def reset_state():
    """Reset rate-limit history and in-flight counters."""
    global in_flight_count
    request_history.clear()
    daily_request_history.clear()
    in_flight_count = 0
    return {"status": "reset_completed"}


def main():
    parser = argparse.ArgumentParser(description="The Relay Stand-in Model Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--limit-rpm", type=int, default=30)
    parser.add_argument("--limit-tpm", type=int, default=8000)
    parser.add_argument("--limit-daily", type=int, default=1000)
    parser.add_argument("--error-rate", type=float, default=0.0)
    parser.add_argument("--timeout-rate", type=float, default=0.0)
    parser.add_argument("--slow-rate", type=float, default=0.0)
    parser.add_argument("--forced-429", action="store_true")
    args = parser.parse_args()

    config.limit_rpm = args.limit_rpm
    config.limit_tpm = args.limit_tpm
    config.limit_daily_requests = args.limit_daily
    config.error_rate = args.error_rate
    config.timeout_rate = args.timeout_rate
    config.slow_rate = args.slow_rate
    config.forced_429 = args.forced_429

    print(f"\n=======================================================")
    print(f"  STARTING STAND-IN MODEL SERVER")
    print(f"  Listening on: http://{args.host}:{args.port}")
    print(f"  Limits: {config.limit_rpm} RPM | {config.limit_tpm} TPM | {config.limit_daily_requests} Daily")
    print(f"  Faults: Error={config.error_rate:.2f} | Timeout={config.timeout_rate:.2f} | Slow={config.slow_rate:.2f}")
    print(f"=======================================================\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

