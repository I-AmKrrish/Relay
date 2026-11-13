import asyncio
import sys
from pathlib import Path
PROJECT_ROOT = Path(".")
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import LLMClient
from src.retrieval import get_retriever

async def test_q06_call():
    r, chunks = get_retriever(mode="embedding", chunk_size=512, overlap=64)
    docs = await r.aretrieve("What BLEU score was achieved by the Transformer (big) model on the WMT 2014 English-to-German translation test in Table 2?", top_k=3)
    client = LLMClient(backend="real", model="qwen/qwen3.8-27b")

    context = "\n\n".join(f"[{d.chunk_id}]: {d.text}" for d in docs)
    prompt = f"Given context:\n{context}\n\nWhat BLEU score was achieved by the Transformer (big) model on the WMT 2014 English-to-German translation test in Table 2?"
    print(f"Prompt length: {len(prompt)} characters")

    resp = await client.generate(messages=[{"role": "user", "content": prompt}])
    print("Status:", resp.status)
    print("Status code:", resp.status_code)
    print("Headers:", resp.headers)
    print("Raw body:", resp.raw_body)
    print("Content preview:", resp.content[:200])

if __name__ == "__main__":
    asyncio.run(test_q06_call())
