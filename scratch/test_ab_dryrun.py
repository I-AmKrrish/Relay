import asyncio
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import QuestionAnsweringPipeline
from src.llm_client import LLMClient
from src import retrieval

async def dryrun():
    print("Testing pipeline initialization...")
    client = LLMClient(backend="stand-in", base_url="http://localhost:8000/v1")
    p = QuestionAnsweringPipeline(llm_client=client)
    res = await p.run("What is d_model in the base Transformer?", scenario="test_dryrun")
    print(f"Status: {res.status} | Total: {res.total_ms:.1f}ms | Tokens: {res.tokens_in}+{res.tokens_out}")

asyncio.run(dryrun())
