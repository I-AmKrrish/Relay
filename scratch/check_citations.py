import json

with open("data/papers/extracted_pages.json", "r", encoding="utf-8") as f:
    pages = json.load(f)

def search_corpus(query, paper_key=None):
    results = []
    q_lower = query.lower()
    for p in pages:
        if paper_key and p["paper_key"] != paper_key:
            continue
        text = p.get("clean_text", "")
        if q_lower in text.lower():
            results.append((p["paper_key"], p.get("active_section", ""), p["page_number"], text))
    return results

print("=== Checking q01: Transformer d_model and heads ===")
res = search_corpus("d_model", "attention")
for r in res[:2]:
    print(f"[{r[0]}, {r[1]}, p. {r[2]}]:\n{r[3][:250]}\n")

print("=== Checking q02: Mimi 24 kHz and 12.5 Hz ===")
res = search_corpus("12.5", "moshi")
for r in res[:2]:
    print(f"[{r[0]}, {r[1]}, p. {r[2]}]:\n{r[3][:250]}\n")

print("=== Checking q03: Dynamic Chunking ratio loss ===")
res = search_corpus("ratio loss", "dynamic_chunking")
for r in res[:2]:
    print(f"[{r[0]}, {r[1]}, p. {r[2]}]:\n{r[3][:250]}\n")

print("=== Checking q06: Transformer big BLEU 28.4 ===")
res = search_corpus("28.4", "attention")
for r in res[:2]:
    print(f"[{r[0]}, {r[1]}, p. {r[2]}]:\n{r[3][:250]}\n")

print("=== Checking q07: Moshi MUSHRA ===")
res = search_corpus("mushra", "moshi")
for r in res[:2]:
    print(f"[{r[0]}, {r[1]}, p. {r[2]}]:\n{r[3][:250]}\n")

print("=== Checking q08: Dynamic Chunking Table 1 ===")
res = search_corpus("Table 1", "dynamic_chunking")
for r in res[:2]:
    print(f"[{r[0]}, {r[1]}, p. {r[2]}]:\n{r[3][:250]}\n")
