"""
Empirical Token Bucket Refill Curve Probe against Real Groq API.
Measures continuous token restoration over multiple time intervals.
"""
import asyncio
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.environ.get("GROQ_API_KEY", "")
assert API_KEY, "GROQ_API_KEY must be set in .env"

MODEL = "qwen/qwen3.8-27b"
URL = "https://api.groq.com/openai/v1/chat/completions"


async def send_ping(client: httpx.AsyncClient, prompt: str, max_tokens: int = 5):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    resp = await client.post(URL, headers=headers, json=payload, timeout=20.0)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    rl_headers = {
        k.lower(): v for k, v in resp.headers.items() 
        if k.lower().startswith("x-ratelimit-") or k.lower() == "retry-after"
    }
    
    return {
        "status_code": resp.status_code,
        "latency_ms": round(latency_ms, 1),
        "headers": rl_headers,
        "body": resp.json() if resp.status_code == 200 or resp.status_code == 429 else resp.text[:200]
    }


async def main():
    print("=== STARTING EMPIRICAL REFILL CURVE PROBE (Max 4 requests) ===")
    curve_data = []
    
    async with httpx.AsyncClient() as client:
        # 1. Baseline ping to inspect current quota headroom
        print("1. Sending initial baseline ping to measure current token headroom...")
        p1 = await send_ping(client, "Reply with 'ready'.", max_tokens=5)
        rem_tok_1 = p1["headers"].get("x-ratelimit-remaining-tokens")
        reset_tok_1 = p1["headers"].get("x-ratelimit-reset-tokens")
        print(f"   Status: {p1['status_code']} | Remaining Tokens: {rem_tok_1} | Reset Window: {reset_tok_1}")
        curve_data.append({"step": "initial_baseline", "elapsed_s": 0.0, "result": p1})
        
        # 2. Draw down tokens using a moderately large prompt (~3,000 tokens)
        print("\n2. Sending heavy payload (~3,500 prompt tokens) to draw down token bucket...")
        heavy_prompt = "Analyze the following text tokens repeatedly: " + (" attention sequence layer hidden dimension " * 450)
        p2 = await send_ping(client, heavy_prompt, max_tokens=10)
        rem_tok_2 = p2["headers"].get("x-ratelimit-remaining-tokens")
        reset_tok_2 = p2["headers"].get("x-ratelimit-reset-tokens")
        retry_after = p2["headers"].get("retry-after")
        print(f"   Status: {p2['status_code']} | Remaining Tokens: {rem_tok_2} | Reset Window: {reset_tok_2} | Retry-After: {retry_after}")
        curve_data.append({"step": "drawdown_probe", "elapsed_s": 0.0, "result": p2})
        
        # 3. Wait 8 seconds to measure mid-flight refill
        wait_1 = 8.0
        print(f"\n3. Waiting {wait_1}s to sample continuous token refill rate...")
        await asyncio.sleep(wait_1)
        p3 = await send_ping(client, "Reply with 'ping'.", max_tokens=5)
        rem_tok_3 = p3["headers"].get("x-ratelimit-remaining-tokens")
        reset_tok_3 = p3["headers"].get("x-ratelimit-reset-tokens")
        print(f"   Status: {p3['status_code']} | Remaining Tokens: {rem_tok_3} | Reset Window: {reset_tok_3}")
        curve_data.append({"step": f"refill_sample_{wait_1}s", "elapsed_s": wait_1, "result": p3})
        
        # 4. Wait another 12 seconds (total 20s elapsed) to verify further refill
        wait_2 = 12.0
        print(f"\n4. Waiting another {wait_2}s (total 20s) to verify continuous refill progress...")
        await asyncio.sleep(wait_2)
        p4 = await send_ping(client, "Reply with 'done'.", max_tokens=5)
        rem_tok_4 = p4["headers"].get("x-ratelimit-remaining-tokens")
        reset_tok_4 = p4["headers"].get("x-ratelimit-reset-tokens")
        print(f"   Status: {p4['status_code']} | Remaining Tokens: {rem_tok_4} | Reset Window: {reset_tok_4}")
        curve_data.append({"step": f"refill_sample_{wait_1 + wait_2}s", "elapsed_s": wait_1 + wait_2, "result": p4})
        
    out_file = PROJECT_ROOT / "raw/probe_recovery_refill_curve.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(curve_data, f, indent=2)
    print(f"\n[SUCCESS] Empirical refill curve saved to: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())

