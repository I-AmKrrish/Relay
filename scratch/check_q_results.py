import json

with open("data/table3_detailed_answers.json", "r", encoding="utf-8") as f:
    data = json.load(f)

res = {s["setting_id"]: {q["id"]: (q["status"], q["grade"]) for q in s["results"]} for s in data}

all_q_ids = sorted(list(res["bm25_k3_c256"].keys()))
print(f"{'QID':<6} | {'BM25 (A)':<22} | {'Dense 256 (B)':<22} | {'Dense 512 (C)':<22}")
print("-" * 75)
for qid in all_q_ids:
    a_stat, a_grd = res["bm25_k3_c256"].get(qid, ("-", "-"))
    b_stat, b_grd = res["dense_k3_c256"].get(qid, ("-", "-"))
    c_stat, c_grd = res["dense_k3_c512"].get(qid, ("-", "-"))
    print(f"{qid:<6} | {a_stat + ':' + a_grd:<22} | {b_stat + ':' + b_grd:<22} | {c_stat + ':' + c_grd:<22}")
