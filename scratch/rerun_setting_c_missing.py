import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(".")
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import LLMClient
from src.pipeline import RelayPipeline, log_result_to_csv, log_result_to_sidecar
from eval.grade import grade_answer_heuristic

async def rerun_missing():
    print("Pre-cooling down for 30s to ensure full token bucket on Groq...")
    await asyncio.sleep(30.0)

    with open("eval/questions.json", "r", encoding="utf-8") as f:
        all_q = {q["id"]: q for q in json.load(f)}

    client = LLMClient(backend="real", model="qwen/qwen3.8-27b", max_requests=25)
    pipeline = RelayPipeline(
        llm_client=client,
        retriever_mode="embedding",
        chunk_size=512,
        overlap=64,
        top_k=3,
    )
    pipeline.retrieval_setting = "dense_k3_c512"

    missing_ids = ["q01", "q06"]
    for qid in missing_ids:
        q_meta = all_q[qid]
        req_id = f"eval_dense_k3_c512_{qid}"
        print(f"\nRunning {qid} for Setting C: '{q_meta['question'][:50]}...'")

        res = None
        for attempt in range(4):
            res = await pipeline.run(
                question=q_meta["question"],
                scenario="retrieval_sweep",
                request_id=req_id,
                log_to_csv_now=False,
            )
            if res.status == "ok":
                break
            print(f"  Attempt {attempt+1} yielded {res.status}. Sleeping 25s...")
            await asyncio.sleep(25.0)

        # Grade
        grade_res = grade_answer_heuristic(q_meta, res.final_answer)
        res.answer_grade = grade_res.grade

        if res.status == "ok":
            log_result_to_csv(res)
            log_result_to_sidecar(res, deadline_seconds=8.0, user_visible_ttft_ms=res.total_ms)

        rem_tokens = client.last_headers.get("x-ratelimit-remaining-tokens", "N/A")
        print(f"  Result {qid}: status={res.status} | grade={res.answer_grade} | total={res.total_ms:.1f}ms | tokens={res.tokens_in+res.tokens_out} | rem_tokens={rem_tokens}")

        # Update detailed answers JSON
        with open("data/table3_detailed_answers.json", "r", encoding="utf-8") as f:
            runs = json.load(f)

        for run in runs:
            if run["setting_id"] == "dense_k3_c512":
                for item in run["results"]:
                    if item["id"] == qid:
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

        print(f"  Updated data/table3_detailed_answers.json for {qid}")
        print("  Pacing: sleeping 25s before next request...")
        await asyncio.sleep(25.0)

if __name__ == "__main__":
    asyncio.run(rerun_missing())

