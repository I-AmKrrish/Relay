import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(".")
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

from src.llm_client import LLMClient
from src.pipeline import RelayPipeline, log_result_to_csv, log_result_to_sidecar
from eval.grade import grade_answer_heuristic

async def finish_q06():
    print("Checking if Groq token budget has refilled...")
    # Check with a 1-token ping
    while True:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "qwen/qwen3.8-27b", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
            )
            if resp.status_code == 200:
                print("Groq is READY! Status 200 OK.")
                print("Remaining tokens:", resp.headers.get("x-ratelimit-remaining-tokens"))
                break
            elif resp.status_code == 429:
                err_text = resp.text
                retry_after = float(resp.headers.get("retry-after", "15"))
                print(f"Still waiting on TPD refill. Retry-After: {retry_after}s. Waiting 15s...")
                await asyncio.sleep(min(15.0, retry_after))
            else:
                print(f"Unexpected status: {resp.status_code}")
                await asyncio.sleep(10.0)

    # Now execute q06 under Setting C
    with open("eval/questions.json", "r", encoding="utf-8") as f:
        questions = {q["id"]: q for q in json.load(f)}

    q_meta = questions["q06"]
    req_id = "eval_dense_k3_c512_q06"

    client = LLMClient(backend="real", model="qwen/qwen3.8-27b", max_requests=10)
    pipeline = RelayPipeline(
        llm_client=client,
        retriever_mode="embedding",
        chunk_size=512,
        overlap=64,
        top_k=3,
    )
    pipeline.retrieval_setting = "dense_k3_c512"

    print(f"\nDispatching q06 for Setting C: '{q_meta['question']}'")
    res = await pipeline.run(
        question=q_meta["question"],
        scenario="retrieval_sweep",
        request_id=req_id,
        log_to_csv_now=False,
    )

    grade_res = grade_answer_heuristic(q_meta, res.final_answer)
    res.answer_grade = grade_res.grade

    print(f"Result q06: status={res.status} | grade={res.answer_grade} | total={res.total_ms:.1f}ms | tokens={res.tokens_in+res.tokens_out}")
    print(f"Answer: {res.final_answer[:150]}")

    if res.status == "ok":
        log_result_to_csv(res)
        log_result_to_sidecar(res, deadline_seconds=8.0, user_visible_ttft_ms=res.total_ms)

        # Update detailed answers JSON
        with open("data/table3_detailed_answers.json", "r", encoding="utf-8") as f:
            runs = json.load(f)

        for run in runs:
            if run["setting_id"] == "dense_k3_c512":
                for item in run["results"]:
                    if item["id"] == "q06":
                        item["status"] = res.status
                        item["grade"] = res.answer_grade
                        item["first_token_ms"] = res.first_token_ms
                        item["total_ms"] = res.total_ms
                        item["tokens_in"] = res.tokens_in
                        item["tokens_out"] = res.tokens_out
                        item["tokens_total"] = res.tokens_in + res.tokens_out
                        item["final_answer"] = res.final_answer

        with open("data/table3_detailed_answers.json", "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2)
        print("Updated data/table3_detailed_answers.json for q06")

if __name__ == "__main__":
    asyncio.run(finish_q06())

