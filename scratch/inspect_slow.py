import json

with open("raw/timing_breakdown.jsonl", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f if line.strip()]

ok_entries = [e for e in entries if e.get("status") == "ok"]
print(f"Total ok entries in sidecar: {len(ok_entries)}")

slowest = sorted(ok_entries, key=lambda x: x.get("total_ms", 0), reverse=True)[:10]
for s in slowest:
    req_id = s["request_id"]
    scenario = s["scenario"]
    source = s["source"]
    total = s["total_ms"]
    q_wait = s["queue_wait_ms"]
    s1 = s["step_1_understand_ms"]
    s2 = s["step_2_retrieve_ms"]
    s3 = s["step_3_decide_ms"]
    s4 = s["step_4_draft_ms"]
    s5 = s["step_5_verify_ms"]
    retries = s["retries"]
    print(f"{req_id} | scenario: {scenario} | source: {source} | total: {total:.1f}ms | q_wait: {q_wait:.1f}ms | s1: {s1:.1f}ms | s2: {s2:.1f}ms | s3: {s3:.1f}ms | s4: {s4:.1f}ms | s5: {s5:.1f}ms | retries: {retries}")
