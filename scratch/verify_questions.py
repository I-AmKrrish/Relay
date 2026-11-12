import json

with open("data/papers/extracted_pages.json", "r", encoding="utf-8") as f:
    pages = json.load(f)

with open("eval/questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

print(f"Total questions: {len(questions)}")
for q in questions:
    print(f"\n--- {q['id']} ({q['category']}) Refusal={q['is_refusal']} ---")
    print(f"Q: {q['question']}")
    print(f"Expected: {q['expected_answer']}")
    print(f"Citations: {q['expected_citations']}")

