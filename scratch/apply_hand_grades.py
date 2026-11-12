import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(".")

# Load detailed answers
with open("data/table3_detailed_answers.json", "r", encoding="utf-8") as f:
    runs = json.load(f)

# Hand-grade corrections dictionary: (setting_id, question_id) -> new_grade
hand_corrections = {
    # Setting A
    ("bm25_k3_c256", "q03"): "correct",  # ratio loss objective matches paper Sec 2.2.3 p. 8
    ("bm25_k3_c256", "q07"): "correct",  # Mimi 81.0 vs RVQGAN 31.3 matches Table 4 p. 23
    # Setting B
    ("dense_k3_c256", "q03"): "correct",  # ratio loss objective matches paper Sec 2.2.3 p. 8
    ("dense_k3_c256", "q07"): "correct",  # Mimi 81.0 vs RVQGAN 31.3 matches Table 4 p. 23
}

# Update detailed answers
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
        "cheap_under_limit": True,
    })

with open("data/table3_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_records, f, indent=2)

print("\nUpdated Table 3 Summary:")
print(pd.DataFrame(summary_records).to_string(index=False))

