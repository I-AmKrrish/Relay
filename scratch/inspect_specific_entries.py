import json

with open("raw/timing_breakdown.jsonl", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f if line.strip()]

for e in entries:
    if e["request_id"] in ["eval_bm25_k3_c256_q02", "eval_bm25_k3_c256_q05", "res_324124"]:
        print(json.dumps(e, indent=2))
        print("="*50)

