"""
eval/run_retrieval_benchmark.py - Run retrieval experiments across 3 settings to populate Table 3.

Settings tested:
  1. Setting A: 256 tokens x top-3 (BM25, no rerank) -> bm25_k3_c256 (Fastest & cheapest)
  2. Setting B: 512 tokens x top-5 (Dense BGE, no rerank) -> dense_k5_c512 (Broad context)
  3. Setting C: 256 tokens x top-3 (Cross-Encoder Re-ranked from top-8) -> rerank_k3_c256 (High precision)

Measures and outputs Table 3:
  - Quality (Accuracy %: correct + refused_correctly)
  - Tokens per answer (tokens_in + tokens_out)
  - TTFT (first_token_ms)
  - Total time (total_ms)
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from eval.grade import grade_answer_heuristic, grade_answer_llm
from src.llm_client import LLMClient
from src.pipeline import PipelineResult, RelayPipeline, log_result_to_csv


SETTINGS = [
    {
        "name": "Setting A (256-tok, top-3 BM25)",
        "setting_id": "bm25_k3_c256",
        "retriever_mode": "bm25",
        "chunk_size": 256,
        "overlap": 32,
        "top_k": 3,
        "rerank": False,
    },
    {
        "name": "Setting B (512-tok, top-5 Dense)",
        "setting_id": "dense_k5_c512",
        "retriever_mode": "embedding",
        "chunk_size": 512,
        "overlap": 64,
        "top_k": 5,
        "rerank": False,
    },
    {
        "name": "Setting C (256-tok, top-8 Rerank)",
        "setting_id": "rerank_k3_c256",
        "retriever_mode": "hybrid",
        "chunk_size": 256,
        "overlap": 32,
        "top_k": 3,
        "rerank": True,
    },
]


async def run_benchmark(
    questions_file: Path = PROJECT_ROOT / "eval/questions.json",
    backend: str = "stand-in",
    model: str = "qwen/qwen3.8-27b",
    max_questions: Optional[int] = None,
    max_requests: int = 200,
):
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if max_questions:
        questions = questions[:max_questions]

    print(f"\n=======================================================")
    print(f"  RETRIEVAL EXPERIMENT SWEEP (Table 3)")
    print(f"  Backend: {backend} | Model: {model} | Questions: {len(questions)}")
    print(f"=======================================================\n")

    summary_records = []

    for cfg in SETTINGS:
        print(f"\n--- Testing {cfg['name']} [{cfg['setting_id']}] ---")
        client = LLMClient(
            backend=backend,
            model=model,
            max_requests=max_requests,
        )
        pipeline = RelayPipeline(
            llm_client=client,
            retriever_mode=cfg["retriever_mode"],
            chunk_size=cfg["chunk_size"],
            overlap=cfg["overlap"],
            top_k=cfg["top_k"],
        )
        # Override setting id for exact tracking
        pipeline.retrieval_setting = cfg["setting_id"]

        results_for_setting: List[Dict[str, Any]] = []

        for idx, q_meta in enumerate(questions, start=1):
            q_text = q_meta["question"]
            req_id = f"eval_{cfg['setting_id']}_{q_meta['id']}"

            # Execute pipeline
            pipe_res = await pipeline.run(
                question=q_text,
                scenario="retrieval_sweep",
                request_id=req_id,
            )

            # Grade the answer
            grade_res = grade_answer_heuristic(q_meta, pipe_res.final_answer)
            pipe_res.answer_grade = grade_res.grade

            # Re-log to results.csv with updated grade
            log_result_to_csv(pipe_res)

            print(
                f"  [{idx}/{len(questions)}] {q_meta['id']}: Grade={pipe_res.answer_grade} | "
                f"TTFT={pipe_res.first_token_ms:.1f}ms | Total={pipe_res.total_ms:.1f}ms | "
                f"Tokens={pipe_res.tokens_in + pipe_res.tokens_out}"
            )

            results_for_setting.append({
                "id": q_meta["id"],
                "grade": pipe_res.answer_grade,
                "first_token_ms": pipe_res.first_token_ms,
                "total_ms": pipe_res.total_ms,
                "tokens_in": pipe_res.tokens_in,
                "tokens_out": pipe_res.tokens_out,
                "tokens_total": pipe_res.tokens_in + pipe_res.tokens_out,
            })

        # Calculate metrics for this setting
        total_eval = len(results_for_setting)
        correct_count = sum(1 for r in results_for_setting if r["grade"] in ("correct", "refused_correctly"))
        accuracy_pct = (correct_count / max(1, total_eval)) * 100.0
        avg_tokens = sum(r["tokens_total"] for r in results_for_setting) / max(1, total_eval)
        avg_ttft = sum(r["first_token_ms"] for r in results_for_setting) / max(1, total_eval)
        avg_total_ms = sum(r["total_ms"] for r in results_for_setting) / max(1, total_eval)

        summary_records.append({
            "setting": cfg["name"],
            "setting_id": cfg["setting_id"],
            "quality_accuracy_pct": accuracy_pct,
            "avg_tokens_per_answer": avg_tokens,
            "avg_ttft_ms": avg_ttft,
            "avg_total_ms": avg_total_ms,
            "cheap_under_limit": cfg["chunk_size"] <= 256 and cfg["top_k"] <= 3,
        })

    # Print Table 3
    print("\n\n" + "=" * 70)
    print("  TABLE 3: RETRIEVAL SETTINGS COMPARISON (Quality vs. Cost)")
    print("=" * 70)
    summary_df = pd.DataFrame(summary_records)
    print(summary_df.to_string(index=False))

    table3_file = PROJECT_ROOT / "data/table3_summary.json"
    table3_file.parent.mkdir(parents=True, exist_ok=True)
    with open(table3_file, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)
    print(f"\nSaved Table 3 data to {table3_file}")


def main():
    parser = argparse.ArgumentParser(description="Run retrieval experiments for Table 3")
    parser.add_argument("--backend", type=str, choices=["real", "stand-in"], default="stand-in")
    parser.add_argument("--model", type=str, default="qwen/qwen3.8-27b")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-requests", type=int, default=200)
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            backend=args.backend,
            model=args.model,
            max_questions=args.max_questions,
            max_requests=args.max_requests,
        )
    )


if __name__ == "__main__":
    main()
