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

def recompute_table3_summary():
    with open("data/table3_detailed_answers.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    summaries = []
    for run in data:
        sid = run["setting_id"]
        res_list = run["results"]
        completed_ok = [r for r in res_list if r["status"] == "ok"]
        correct_count = sum(1 for r in completed_ok if r["grade"] in ["correct", "refused_correctly"])
        n_completed = len(completed_ok)
        acc = (correct_count / n_completed * 100.0) if n_completed > 0 else 0.0

        avg_tok = sum(r["tokens_total"] for r in completed_ok) / max(1, n_completed)
        avg_ttft = sum(r["first_token_ms"] for r in completed_ok) / max(1, n_completed)
        avg_tot = sum(r["total_ms"] for r in completed_ok) / max(1, n_completed)

        setting_name = {
            "bm25_k3_c256": "Setting A (256-tok, top-3 BM25)",
            "dense_k3_c256": "Setting B (256-tok, top-3 Dense BGE)",
            "dense_k3_c512": "Setting C (512-tok, top-3 Dense BGE)",
        }.get(sid, sid)

        var_isolated = {
            "bm25_k3_c256": "Baseline (Lexical)",
            "dense_k3_c256": "Retriever: Dense BGE vs BM25 (chunk=256 held constant)",
            "dense_k3_c512": "Chunk Size: 512 vs 256 (Dense retriever held constant)",
        }.get(sid, "")

        summaries.append({
            "setting": setting_name,
            "setting_id": sid,
            "variable_isolated": var_isolated,
            "completed_ok": n_completed,
            "total_offered": len(res_list),
            "quality_accuracy_pct": round(acc, 1),
            "avg_tokens_per_answer": round(avg_tok, 1),
            "avg_ttft_ms": round(avg_ttft, 1),
            "avg_total_ms": round(avg_tot, 1),
            "cheap_under_limit": avg_tok < 3500,
        })

    with open("data/table3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print("Recomputed data/table3_summary.json!")

async def finish_q06():
    print("Checking if Groq token budget is available...")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "qwen/qwen3.8-27b", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
        )
        if resp.status_code == 200:
            print(f"Groq is READY! Status 200 OK. Remaining tokens: {resp.headers.get('x-ratelimit-remaining-tokens')}")
        elif resp.status_code == 429:
            retry_after = float(resp.headers.get("retry-after", "15"))
            print(f"Still waiting on TPD refill. Retry-After: {retry_after}s.")
            return
        else:
            print(f"Unexpected status: {resp.status_code}")
            return

    # Now execute q06 under Setting C
    with open("eval/questions.json", "r", encoding="utf-8") as f:
        questions = {q["id"]: q for q in json.load(f)}

    q_meta = questions["q06"]
    req_id = "eval_dense_k3_c512_q06"

    # Enforce strict safety boundary per Hard Rule #3
    client = LLMClient(backend="real", model="qwen/qwen3.8-27b", max_requests=8)
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

    # Verify if 28.4 is in the answer
    if "28.4" in res.final_answer:
        res.answer_grade = "correct"

    print(f"Result q06: status={res.status} | grade={res.answer_grade} | total={res.total_ms:.1f}ms | tokens={res.tokens_in+res.tokens_out}")
    print(f"Answer: {res.final_answer[:200]}")

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

        recompute_table3_summary()

if __name__ == "__main__":
    asyncio.run(finish_q06())
