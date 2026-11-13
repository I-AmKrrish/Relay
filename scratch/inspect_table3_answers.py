import json

with open("data/table3_detailed_answers.json", encoding="utf-8") as f:
    runs = json.load(f)

for r in runs:
    print("=" * 60)
    ok_items = [x for x in r["results"] if x["status"] == "ok"]
    print(r["setting"], "| Completed OK:", len(ok_items), "out of", len(r["results"]))
    for x in r["results"]:
        qid = x["id"]
        status = x["status"]
        grade = x["grade"]
        tokens = x["tokens_total"]
        ans_preview = x["final_answer"].replace("\n", " ")[:60]
        print(f"  {qid}: status={status:<12} | grade={grade:<17} | tok={tokens:4d} | ans={ans_preview}...")
