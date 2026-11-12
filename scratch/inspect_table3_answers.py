import json

with open("data/table3_detailed_answers.json", "r", encoding="utf-8") as f:
    runs = json.load(f)

for run in runs:
    print(f"\n{'='*70}\nSetting: {run['setting']} ({run['setting_id']})\n{'='*70}")
    for res in run["results"]:
        ans = res.get("final_answer", "").replace("\n", " ")
        print(f"[{res['id']}] Status: {res['status']:<12} | Grade: {res['grade']:<18} | Tokens: {res['tokens_total']}")
        print(f"     Answer: {ans[:140]}...")

