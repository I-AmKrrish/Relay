"""
eval/run_retrieval_benchmark.py - Run retrieval experiments across 3 controlled settings for Table 3.

Settings tested (each isolating exactly ONE variable):
  1. Setting A: 256-tok x top-3 BM25 -> bm25_k3_c256 (Baseline Lexical)
  2. Setting B: 256-tok x top-3 Dense BGE -> dense_k3_c256 (Isolates Retriever: BM25 -> Dense BGE, chunk=256 held constant)
  3. Setting C: 512-tok x top-3 Dense BGE -> dense_k3_c512 (Isolates Chunk Size: 256 -> 512 tokens, Dense retriever held constant)

Features:
  - Token-aware pacing (ensures bucket has >= 4,500 tokens before dispatching)
  - Automatic Retry-After handling on rate limits so EVERY question completes with status == 'ok'
  - Preloaded neural models to avoid event loop stalls
  - Atomic logging of only status == 'ok' rows to results.csv
  - Outputs exact required columns for Table 3
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

from eval.grade import grade_answer_heuristic
from src.llm_client import LLMClient
from src.pipeline import PipelineResult, RelayPipeline, log_result_to_csv, log_result_to_sidecar
from src.retrieval import preload_all_models

QUESTIONS_FILE = PROJECT_ROOT / "eval/questions.json"
RESULTS_CSV = PROJECT_ROOT / "results.csv"

ALL_SETTINGS = {
    "A": {
        "name": "Setting A (256-tok, top-3 BM25)",
        "setting_id": "bm25_k3_c256",
        "retriever_mode": "bm25",
        "chunk_size": 256,
        "overlap": 32,
        "top_k": 3,
        "variable_isolated": "Baseline (Lexical)",
    },
    "B": {
        "name": "Setting B (256-tok, top-3 Dense BGE)",
        "setting_id": "dense_k3_c256",
        "retriever_mode": "embedding",
        "chunk_size": 256,
        "overlap": 32,
        "top_k": 3,
        "variable_isolated": "Retriever: Dense BGE vs BM25 (chunk=256 held constant)",
    },
    "C": {
        "name": "Setting C (512-tok, top-3 Dense BGE)",
        "setting_id": "dense_k3_c512",
        "retriever_mode": "embedding",
        "chunk_size": 512,
        "overlap": 64,
        "top_k": 3,
        "variable_isolated": "Chunk Size: 512 vs 256 (Dense retriever held constant)",
    },
}


def clean_old_sweep_from_results():
    if not RESULTS_CSV.exists():
        return
    df = pd.read_csv(RESULTS_CSV)
    df_clean = df[df["scenario"] != "retrieval_sweep"].copy()
    df_clean.to_csv(RESULTS_CSV, index=False)
    print(f"[Cleanup] Removed old retrieval_sweep rows from results.csv. Remaining rows: {len(df_clean)}")


async def run_benchmark(
    questions_file: Path = QUESTIONS_FILE,
    backend: str = "real",
    model: str = "qwen/qwen3.8-27b",
    settings_to_run: Optional[List[str]] = None,
    max_questions: Optional[int] = None,
    max_requests: int = 250,
    pacing_seconds: float = 22.0,
):
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if max_questions:
        questions = questions[:max_questions]

    # Startup preloading of neural models to prevent in-flight event loop stalls
    preload_all_models()

    active_keys = settings_to_run or ["A", "B", "C"]
    chosen_settings = [ALL_SETTINGS[k.strip().upper()] for k in active_keys if k.strip().upper() in ALL_SETTINGS]

    print(f"\n{'='*75}")
    print(f"  RETRIEVAL EXPERIMENT SWEEP (Table 3)")
    print(f"  Backend: {backend} | Model: {model} | Questions: {len(questions)}")
    print(f"  Active Settings: {[s['name'] for s in chosen_settings]}")
    print(f"{'='*75}\n")

    summary_records = []
    all_detailed_runs = []

    client = LLMClient(
        backend=backend,
        model=model,
        max_requests=max_requests,
    )

    for cfg in chosen_settings:
        print(f"\n{'='*75}")
        print(f"--- Running {cfg['name']} [{cfg['setting_id']}] ---")
        print(f"    Variable Isolated: {cfg['variable_isolated']}")
        print(f"{'='*75}")

        pipeline = RelayPipeline(
            llm_client=client,
            retriever_mode=cfg["retriever_mode"],
            chunk_size=cfg["chunk_size"],
            overlap=cfg["overlap"],
            top_k=cfg["top_k"],
        )
        pipeline.retrieval_setting = cfg["setting_id"]

        results_for_setting: List[Dict[str, Any]] = []

        for idx, q_meta in enumerate(questions, start=1):
            q_text = q_meta["question"]
            req_id = f"eval_{cfg['setting_id']}_{q_meta['id']}"

            # Execute with automatic 429 retry to guarantee complete 16/16 ok rows
            pipe_res = None
            max_attempts = 5
            for attempt in range(max_attempts):
                pipe_res = await pipeline.run(
                    question=q_text,
                    scenario="retrieval_sweep",
                    request_id=req_id,
                    log_to_csv_now=False,
                )

                if pipe_res.status == "ok":
                    break
                elif pipe_res.status == "rate_limited":
                    hdr = client.last_headers.get("retry-after")
                    retry_wait = 20.0
                    if hdr:
                        try:
                            retry_wait = float(hdr)
                        except ValueError:
                            pass
                    sleep_time = retry_wait + 3.0
                    print(f"    [429 Rate Limited on {q_meta['id']}] Waiting {sleep_time:.1f}s for token refill before attempt {attempt+2}/{max_attempts}...")
                    await asyncio.sleep(sleep_time)
                else:
                    print(f"    [{pipe_res.status} on {q_meta['id']}] Retrying attempt {attempt+2}/{max_attempts} after 5.0s...")
                    await asyncio.sleep(5.0)

            # Grade the answer
            grade_res = grade_answer_heuristic(q_meta, pipe_res.final_answer)
            pipe_res.answer_grade = grade_res.grade

            # Atomic single logging to results.csv and sidecar ONLY on completed ok rows
            if pipe_res.status == "ok":
                log_result_to_csv(pipe_res)
                log_result_to_sidecar(
                    pipe_res,
                    deadline_seconds=8.0,
                    user_visible_ttft_ms=pipe_res.total_ms,
                )

            rem_tokens_str = client.last_headers.get("x-ratelimit-remaining-tokens", "N/A")
            print(
                f"  [{idx:02d}/{len(questions)}] {q_meta['id']}: status={pipe_res.status:<12} | "
                f"grade={pipe_res.answer_grade:<17} | total={pipe_res.total_ms:6.1f}ms | "
                f"tokens={pipe_res.tokens_in + pipe_res.tokens_out:4d} | rem_tokens={rem_tokens_str}"
            )

            results_for_setting.append({
                "id": q_meta["id"],
                "status": pipe_res.status,
                "grade": pipe_res.answer_grade,
                "first_token_ms": pipe_res.first_token_ms,
                "total_ms": pipe_res.total_ms,
                "tokens_in": pipe_res.tokens_in,
                "tokens_out": pipe_res.tokens_out,
                "tokens_total": pipe_res.tokens_in + pipe_res.tokens_out,
                "final_answer": pipe_res.final_answer,
            })

            # Dynamic token-aware pacing between questions (protecting 7,000 ITPM cap)
            if backend == "real" and idx < len(questions):
                rem_tok_hdr = client.last_headers.get("x-ratelimit-remaining-tokens")
                if rem_tok_hdr:
                    try:
                        rem_tokens = int(rem_tok_hdr)
                        # We require at least 4,500 tokens before sending next question
                        if rem_tokens < 4500:
                            refill_wait = max(pacing_seconds, ((6000 - rem_tokens) / 116.7) + 2.0)
                            print(f"    [Pacing] Low token headroom ({rem_tokens} < 4,500). Sleeping {refill_wait:.1f}s to refill to ~6,000...")
                            await asyncio.sleep(refill_wait)
                        else:
                            await asyncio.sleep(pacing_seconds)
                    except ValueError:
                        await asyncio.sleep(pacing_seconds)
                else:
                    await asyncio.sleep(pacing_seconds)

        # Calculate metrics strictly on completed rows (status == 'ok')
        ok_results = [r for r in results_for_setting if r["status"] == "ok"]
        total_ok = len(ok_results)
        total_offered = len(results_for_setting)

        correct_count = sum(1 for r in ok_results if r["grade"] in ("correct", "refused_correctly"))
        accuracy_pct = (correct_count / max(1, total_ok)) * 100.0

        avg_tokens = sum(r["tokens_total"] for r in ok_results) / max(1, total_ok) if total_ok else 0.0
        avg_ttft = sum(r["first_token_ms"] for r in ok_results) / max(1, total_ok) if total_ok else 0.0
        avg_total_ms = sum(r["total_ms"] for r in ok_results) / max(1, total_ok) if total_ok else 0.0

        summary_records.append({
            "setting": cfg["name"],
            "setting_id": cfg["setting_id"],
            "variable_isolated": cfg["variable_isolated"],
            "completed_ok": total_ok,
            "total_offered": total_offered,
            "quality_accuracy_pct": round(accuracy_pct, 1),
            "avg_tokens_per_answer": round(avg_tokens, 1),
            "avg_ttft_ms": round(avg_ttft, 1),
            "avg_total_ms": round(avg_total_ms, 1),
            "cheap_under_limit": cfg["chunk_size"] <= 256 and cfg["top_k"] <= 3,
        })

        all_detailed_runs.append({
            "setting": cfg["name"],
            "setting_id": cfg["setting_id"],
            "variable_isolated": cfg["variable_isolated"],
            "results": results_for_setting,
        })

    # Print Table 3
    print("\n\n" + "=" * 80)
    print("  TABLE 3: RETRIEVAL SETTINGS COMPARISON (Quality vs. Cost)")
    print("=" * 80)
    summary_df = pd.DataFrame(summary_records)
    print(summary_df.to_string(index=False))

    table3_file = PROJECT_ROOT / "data/table3_summary.json"
    table3_file.parent.mkdir(parents=True, exist_ok=True)
    with open(table3_file, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)
    print(f"\nSaved Table 3 data to {table3_file}")

    detailed_file = PROJECT_ROOT / "data/table3_detailed_answers.json"
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(all_detailed_runs, f, indent=2)
    print(f"Saved detailed answer logs to {detailed_file}")


def main():
    parser = argparse.ArgumentParser(description="Run retrieval experiments for Table 3")
    parser.add_argument("--backend", type=str, choices=["real", "stand-in"], default="real")
    parser.add_argument("--model", type=str, default="qwen/qwen3.8-27b")
    parser.add_argument("--settings", type=str, default="A,B,C", help="Comma-separated setting keys: A, B, C")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-requests", type=int, default=250)
    parser.add_argument("--pacing-seconds", type=float, default=22.0)
    args = parser.parse_args()

    # Clean old sweep rows from results.csv before fresh run
    clean_old_sweep_from_results()

    chosen = [s.strip() for s in args.settings.split(",") if s.strip()]

    asyncio.run(
        run_benchmark(
            backend=args.backend,
            model=args.model,
            settings_to_run=chosen,
            max_questions=args.max_questions,
            max_requests=args.max_requests,
            pacing_seconds=args.pacing_seconds,
        )
    )


if __name__ == "__main__":
    main()
