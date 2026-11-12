import json

pages = json.load(open("data/papers/extracted_pages.json", encoding="utf-8"))
for p in pages:
    if p["paper_key"] == "attention" and "512" in p["clean_text"]:
        print(f"Page {p['page_number']} [{p.get('active_section')}]:")
        for line in p["clean_text"].split("\n"):
            if "512" in line or "head" in line.lower() or "d_model" in line.lower() or "dmodel" in line.lower():
                print("  ", line[:120])

