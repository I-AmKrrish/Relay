import json

with open("data/table3_detailed_answers.json", "r", encoding="utf-8") as f:
    runs = json.load(f)

for run in runs:
    print("=" * 80)
    print(f"SETTING: {run['setting']} ({run['setting_id']})")
    print(f"Completed OK: {sum(1 for x in run['results'] if x['status'] == 'ok')}/16")
    print("=" * 80)
    for x in run["results"]:
        qid = x["id"]
        status = x["status"]
        grade = x["grade"]
        ans = x["final_answer"].replace("\n", " ").strip()
        print(f"[{qid}] status={status:<12} | heuristic_grade={grade:<17} | {ans[:100]}...")

