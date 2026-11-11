"""
src/probe_groq.py - Controlled empirical probe of the real Groq API.

Objectives:
  1. Measure empirical latency distribution (TTFT and total_ms) under streaming.
  2. Inspect and record x-ratelimit-* and retry-after response headers.
  3. Measure concurrency behavior (3 in-flight parallel calls).
  4. Trigger a controlled 429 Too Many Requests response to capture status code, body, headers, and recovery time.
  5. Log every call to results.csv and save raw payloads to raw/ folder.
  6. Enforce strict --max-requests cap (default: 10).
"""

import argparse
import asyncio
import csv
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


def log_probe_to_csv(
    request_id: str,
    scenario: str,
    first_token_ms: float,
    total_ms: float,
    status: str,
    tokens_in: int,
    tokens_out: int,
    csv_path: Path = PROJECT_ROOT / "results.csv",
):
    """Log an execution row to results.csv matching the exact 12-column schema."""
    start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    row = [
        request_id,
        scenario,
        "real",
        start_iso,
        f"{first_token_ms:.1f}",
        f"{total_ms:.1f}",
        status,
        0,  # retries
        tokens_in,
        tokens_out,
        "none_probe",
        "not_graded",
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


async def send_groq_streaming_probe(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int = 50,
    request_id: str = "probe_0",
    scenario: str = "probe_seq",
    raw_dir: Path = PROJECT_ROOT / "raw",
) -> Dict[str, Any]:
    """Execute a single streaming call to Groq and capture all headers and timings."""
    endpoint = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "TheRelayProbe/1.0",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = time.perf_counter()
    first_token_time: Optional[float] = None
    collected_tokens: List[str] = []
    tokens_in = 0
    tokens_out = 0
    captured_headers: Dict[str, str] = {}
    status_code = 0
    raw_error = None
    status = "ok"

    try:
        async with client.stream("POST", endpoint, headers=headers, json=payload, timeout=20.0) as resp:
            status_code = resp.status_code
            for k, v in resp.headers.items():
                k_lower = k.lower()
                if k_lower.startswith("x-ratelimit-") or k_lower in ("retry-after", "content-type", "date"):
                    captured_headers[k_lower] = v

            if status_code == 429:
                status = "rate_limited"
                raw_bytes = await resp.aread()
                raw_error = raw_bytes.decode("utf-8", errors="replace")
            elif status_code != 200:
                status = "error"
                raw_bytes = await resp.aread()
                raw_error = raw_bytes.decode("utf-8", errors="replace")
            else:
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if "usage" in chunk and chunk["usage"]:
                        u = chunk["usage"]
                        tokens_in = u.get("prompt_tokens", tokens_in)
                        tokens_out = u.get("completion_tokens", tokens_out)

                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        piece = delta.get("content", "")
                        if piece:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            collected_tokens.append(piece)

    except httpx.TimeoutException:
        status = "timeout"
        status_code = 408
    except Exception as exc:
        status = "error"
        raw_error = str(exc)

    t_end = time.perf_counter()
    total_ms = (t_end - t0) * 1000.0
    first_token_ms = ((first_token_time - t0) * 1000.0) if first_token_time else total_ms
    full_text = "".join(collected_tokens)

    # Save raw log
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_record = {
        "request_id": request_id,
        "scenario": scenario,
        "timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "status_code": status_code,
        "status": status,
        "first_token_ms": first_token_ms,
        "total_ms": total_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "headers": captured_headers,
        "response_text": full_text,
        "raw_error": raw_error,
    }
    raw_file = raw_dir / f"{request_id}.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(raw_record, f, indent=2)

    # Log to results.csv
    log_probe_to_csv(
        request_id=request_id,
        scenario=scenario,
        first_token_ms=first_token_ms,
        total_ms=total_ms,
        status=status,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )

    return raw_record


async def run_probes(model: str = "qwen/qwen3.8-27b", max_requests: int = 10):
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("ERROR: GROQ_API_KEY is not set in environment or .env. Aborting.")
        return

    print(f"\n=======================================================")
    print(f"  GROQ REAL SERVICE PROBE (Model: {model})")
    print(f"  Hard request cap: {max_requests}")
    print(f"=======================================================\n")

    requests_sent = 0
    raw_dir = PROJECT_ROOT / "raw"
    all_results = []

    async with httpx.AsyncClient() as client:
        # PART 1: Sequential Baseline Probes (3 requests)
        print("--- [Part 1/3] Sequential Latency & Rate Limit Headers ---")
        seq_prompts = [
            "Explain self-attention mechanism in 15 words.",
            "What is an audio codec in 15 words?",
            "What is byte-level language modeling in 15 words?",
        ]
        for idx, prompt in enumerate(seq_prompts, start=1):
            if requests_sent >= max_requests:
                print("Cap reached, stopping.")
                break
            req_id = f"probe_seq_{idx}"
            print(f"Sending sequential probe {idx}/3: '{prompt}'...")
            res = await send_groq_streaming_probe(
                client, api_key, model, prompt, max_tokens=35, request_id=req_id, scenario="probe_seq"
            )
            requests_sent += 1
            all_results.append(res)
            print(
                f"  -> Status: {res['status_code']} | TTFT: {res['first_token_ms']:.1f}ms | "
                f"Total: {res['total_ms']:.1f}ms | Tokens: {res['tokens_in']}in/{res['tokens_out']}out"
            )
            # Print rate limit headers
            h = res["headers"]
            rl_rem_req = h.get("x-ratelimit-remaining-requests", "N/A")
            rl_rem_tok = h.get("x-ratelimit-remaining-tokens", "N/A")
            rl_reset_req = h.get("x-ratelimit-reset-requests", "N/A")
            rl_reset_tok = h.get("x-ratelimit-reset-tokens", "N/A")
            print(f"  -> Headers: Rem-Req: {rl_rem_req}, Rem-Tok: {rl_rem_tok}, Reset-Req: {rl_reset_req}, Reset-Tok: {rl_reset_tok}")
            await asyncio.sleep(0.5)

        # PART 2: Concurrency Probes (3 parallel requests)
        print("\n--- [Part 2/3] Concurrency Probing (3 parallel in-flight) ---")
        concur_prompts = [
            "Define multi-head attention concisely.",
            "Define residual vector quantization concisely.",
            "Define hierarchical sequence modeling concisely.",
        ]
        tasks = []
        for idx, prompt in enumerate(concur_prompts, start=1):
            if requests_sent >= max_requests:
                break
            req_id = f"probe_concur_{idx}"
            tasks.append(
                send_groq_streaming_probe(
                    client, api_key, model, prompt, max_tokens=30, request_id=req_id, scenario="probe_concurrent"
                )
            )
            requests_sent += 1

        print(f"Dispatching {len(tasks)} requests concurrently...")
        t_batch_start = time.perf_counter()
        concur_results = await asyncio.gather(*tasks)
        batch_duration_ms = (time.perf_counter() - t_batch_start) * 1000.0
        all_results.extend(concur_results)

        for r in concur_results:
            print(
                f"  [{r['request_id']}] Status: {r['status_code']} | TTFT: {r['first_token_ms']:.1f}ms | "
                f"Total: {r['total_ms']:.1f}ms"
            )
        print(f"  -> Concurrency batch wall-clock time: {batch_duration_ms:.1f} ms")

        # PART 3: Controlled 429 Rate-Limit Trigger
        print("\n--- [Part 3/3] Controlled 429 Limit Probe ---")
        if requests_sent < max_requests:
            req_id = "probe_429_burst"
            print("Triggering controlled token limit probe (requesting max_tokens=8000 when free TPM is 6000)...")
            res_429 = await send_groq_streaming_probe(
                client,
                api_key,
                model,
                "List numbers from 1 to 5000",
                max_tokens=8000,  # Exceeds 6000 TPM limit
                request_id=req_id,
                scenario="probe_429",
            )
            requests_sent += 1
            all_results.append(res_429)
            print(f"  -> Status Code: {res_429['status_code']}")
            print(f"  -> Status: {res_429['status']}")
            print(f"  -> Headers captured: {res_429['headers']}")
            if res_429["raw_error"]:
                print(f"  -> 429 Body excerpt: {res_429['raw_error'][:250]}...")

            # Check recovery time with a small follow-up request
            if requests_sent < max_requests:
                retry_after_str = res_429["headers"].get("retry-after", "1")
                try:
                    wait_sec = float(retry_after_str)
                except ValueError:
                    wait_sec = 1.0
                print(f"  Waiting {wait_sec:.1f}s according to Retry-After / recovery window...")
                await asyncio.sleep(min(wait_sec, 3.0))

                req_id_rec = "probe_recovery"
                print("Sending recovery verification request...")
                res_rec = await send_groq_streaming_probe(
                    client,
                    api_key,
                    model,
                    "Ping: answer with 'pong'.",
                    max_tokens=10,
                    request_id=req_id_rec,
                    scenario="probe_recovery",
                )
                requests_sent += 1
                all_results.append(res_rec)
                print(f"  -> Recovery Status: {res_rec['status_code']} ({res_rec['status']}) in {res_rec['total_ms']:.1f}ms")

    print(f"\n=======================================================")
    print(f"  PROBE COMPLETE. Total requests used: {requests_sent}/{max_requests}")
    print(f"  Raw logs saved in: {raw_dir}")
    print(f"  Results appended to: {PROJECT_ROOT / 'results.csv'}")
    print(f"=======================================================\n")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Empirical Probe of Groq Service")
    parser.add_argument("--model", type=str, default="qwen/qwen3.8-27b")
    parser.add_argument("--max-requests", type=int, default=10, help="Hard request cap (default: 10)")
    args = parser.parse_args()

    asyncio.run(run_probes(model=args.model, max_requests=args.max_requests))


if __name__ == "__main__":
    main()

