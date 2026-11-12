import json

with open("data/table3_detailed_answers.json", "r", encoding="utf-8") as f:
    runs = json.load(f)

for run in runs:
    print(f"\n=======================================================")
    print(f"Hand-Grading Setting: {run['setting']} ({run['setting_id']})")
    print(f"=======================================================")
    for res in run["results"]:
        if res["grade"] == "refused_correctly":
            continue
        print(f"\n--- Question {res['id']} (Status: {res['status']}, Auto-Grade: {res['grade']}) ---")
        print(f"Final Answer:\n{res['final_answer']}")

