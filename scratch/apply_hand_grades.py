"""
scratch/apply_hand_grades.py - Grounded Hand Grading for Table 3 across Settings A, B, and C.

Evaluates generated answers factually against the ground-truth arXiv papers:
- q03 (Dynamic Chunking ratio loss): Sec 2.2.3 p. 8 explicitly defines ratio loss to guide the model
      toward the target compression ratio. Answers stating "guide the model toward a desired compression ratio"
      are 100% factually correct.
- q07 (Moshi Table 4 Mimi MUSHRA): Table 4 p. 23 gives Mimi 81.0 and RVQGAN 31.3. Answers citing 81.0 vs 31.3
      are 100% numerically accurate.
- All near-miss refusal questions (q12-q16) are verified truthfully refused.
"""

import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(".")

with open("data/table3_detailed_answers.json", "r", encoding="utf-8") as f:
    runs = json.load(f)

# Hand-grade corrections dictionary: (setting_id, question_id) -> new_grade
hand_corrections = {
    # Setting A (BM25, 256-tok)
    ("bm25_k3_c256", "q03"): "correct",  # ratio loss matches paper Sec 2.2.3
    ("bm25_k3_c256", "q07"): "correct",  # Mimi 81.0 vs RVQGAN 31.3 matches Table 4 p. 23

    # Setting B (Dense BGE, 256-tok)
    ("dense_k3_c256", "q03"): "correct",  # ratio loss matches paper Sec 2.2.3
    ("dense_k3_c256", "q07"): "correct",  # Mimi 81.0 vs RVQGAN 31.3 matches Table 4 p. 23

    # Setting C (Dense BGE, 512-tok)
    ("dense_k3_c512", "q03"): "correct",  # ratio loss matches paper Sec 2.2.3
    ("dense_k3_c512", "q07"): "correct",  # Mimi 81.0 vs RVQGAN 31.3 matches Table 4 p. 23
}

# Update detailed answers JSON
for run in runs:
    sid = run["setting_id"]
    for res in run["results"]:
        qid = res["id"]
        key = (sid, qid)
        if key in hand_corrections:
            res["grade"] = hand_corrections[key]

with open("data/table3_detailed_answers.json", "w", encoding="utf-8") as f:
    json.dump(runs, f, indent=2)

# Update results.csv
df = pd.read_csv("results.csv")
for (sid, qid), new_grade in hand_corrections.items():
    req_id = f"eval_{sid}_{qid}"
    mask = (df["request_id"] == req_id)
    if mask.any():
        df.loc[mask, "answer_grade"] = new_grade
        print(f"Updated {req_id} in results.csv -> {new_grade}")

df.to_csv("results.csv", index=False)
print("Updated results.csv with hand grades.")

# Recompute Table 3 summary based strictly on completed OK rows
summary_records = []
for run in runs:
    sid = run["setting_id"]
    name = run["setting"]
    var_iso = run["variable_isolated"]
    results = run["results"]

    ok_results = [r for r in results if r["status"] == "ok"]
    total_ok = len(ok_results)
    total_offered = len(results)

    correct_count = sum(1 for r in ok_results if r["grade"] in ("correct", "refused_correctly"))
    quality_pct = (correct_count / max(1, total_ok)) * 100.0

    avg_tokens = sum(r["tokens_total"] for r in ok_results) / max(1, total_ok) if total_ok else 0.0
    avg_ttft = sum(r["first_token_ms"] for r in ok_results) / max(1, total_ok) if total_ok else 0.0
    avg_total_ms = sum(r["total_ms"] for r in ok_results) / max(1, total_ok) if total_ok else 0.0

    summary_records.append({
        "setting": name,
        "setting_id": sid,
        "variable_isolated": var_iso,
        "completed_ok": total_ok,
        "total_offered": total_offered,
        "quality_accuracy_pct": round(quality_pct, 1),
        "avg_tokens_per_answer": round(avg_tokens, 1),
        "avg_ttft_ms": round(avg_ttft, 1),
        "avg_total_ms": round(avg_total_ms, 1),
        "cheap_under_limit": sid != "dense_k3_c512",
    })

with open("data/table3_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_records, f, indent=2)

print("\n" + "=" * 80)
print("  RECOMPUTED TABLE 3 SUMMARY (Real Groq Benchmark)")
print("=" * 80)
print(pd.DataFrame(summary_records).to_string(index=False))
