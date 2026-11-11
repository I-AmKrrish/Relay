"""
src/pipeline.py - Multi-step RAG Pipeline for The Relay.

Pipeline Architecture:
  Step 1: Understand (classify intent and refine search query)
  Step 2: Retrieve (tool call over the arXiv library)
  Step 3: Decide (BRANCH/LOOP: check sufficiency; re-retrieve if weak; refuse if unsupported)
  Step 4: Draft (synthesize cited answer; state 'the papers do not say' if missing)
  Step 5: Verify (confirm claims/citations; redraft once on error; skippable in degraded mode)

Instruments every step with exact latencies:
  - queue_wait_ms
  - model_time_ms (split into understand, draft, verify)
  - retry_time_ms (loops and redrafts)
  - first_token_ms (TTFT) and total_ms
"""

import argparse
import asyncio
import csv
import datetime
import json
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import LLMClient, LLMResponse
from src.retrieval import SearchResult, get_retriever


@dataclass
class TimingBreakdown:
    queue_wait_ms: float = 0.0
    understand_ms: float = 0.0
    retrieve_ms: float = 0.0
    decide_ms: float = 0.0
    draft_ms: float = 0.0
    verify_ms: float = 0.0
    retry_time_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class PipelineResult:
    request_id: str
    scenario: str
    source: str  # 'real' | 'stand-in'
    start_time: str
    question: str
    final_answer: str
    status: str  # 'ok' | 'rate_limited' | 'timeout' | 'error'
    first_token_ms: float
    total_ms: float
    retries: int
    tokens_in: int
    tokens_out: int
    retrieval_setting: str
    answer_grade: str  # 'correct' | 'wrong' | 'refused_correctly' | 'refused_wrongly'
    citations: List[str] = field(default_factory=list)
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)
    verification_passed: bool = True
    loop_count: int = 0


def log_result_to_csv(result: PipelineResult, csv_path: Path = PROJECT_ROOT / "results.csv"):
    """
    Append an execution row to results.csv matching the exact required columns:
      request_id,scenario,source,start_time,first_token_ms,total_ms,status,retries,tokens_in,tokens_out,retrieval_setting,answer_grade
    """
    file_exists = csv_path.exists()
    headers = [
        "request_id",
        "scenario",
        "source",
        "start_time",
        "first_token_ms",
        "total_ms",
        "status",
        "retries",
        "tokens_in",
        "tokens_out",
        "retrieval_setting",
        "answer_grade",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([
            result.request_id,
            result.scenario,
            result.source,
            result.start_time,
            f"{result.first_token_ms:.1f}",
            f"{result.total_ms:.1f}",
            result.status,
            result.retries,
            result.tokens_in,
            result.tokens_out,
            result.retrieval_setting,
            result.answer_grade,
        ])


class RelayPipeline:
    """The 5-step RAG pipeline."""

    def __init__(
        self,
        llm_client: LLMClient,
        retriever_mode: str = "bm25",
        chunk_size: int = 256,
        overlap: int = 32,
        top_k: int = 3,
        max_re_retrieve_loops: int = 1,
        skip_verification: bool = False,
    ):
        self.llm = llm_client
        self.retriever_mode = retriever_mode
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.max_loops = max_re_retrieve_loops
        self.skip_verification = skip_verification

        # Initialize retriever
        chunks_file = PROJECT_ROOT / f"data/papers/chunks_{chunk_size}_{overlap}.json"
        self.retriever, self.chunks = get_retriever(
            mode=retriever_mode,
            chunk_size=chunk_size,
            overlap=overlap,
            chunks_path=chunks_file,
        )
        self.retrieval_setting = f"{retriever_mode}_k{top_k}_c{chunk_size}"

    async def step_1_understand(self, question: str) -> Tuple[str, LLMResponse]:
        """
        Step 1: Understand question and formulate an optimized keyword search query.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert ML researcher. Your corpus consists ONLY of three papers:\n"
                    "1. 'Attention Is All You Need' (Transformer)\n"
                    "2. 'Moshi' (Speech-text foundation model for real-time dialogue)\n"
                    "3. 'Dynamic Chunking for End-to-End Hierarchical Sequence Modeling' (H-Net)\n\n"
                    "Extract the core technical concepts and produce a concise 3-8 word search query. "
                    "Output ONLY the search query, nothing else."
                ),
            },
            {"role": "user", "content": f"User Question: {question}"},
        ]
        resp = await self.llm.generate(messages, temperature=0.0, max_tokens=32)
        search_query = resp.content.strip().strip('"').strip("'")
        if not search_query or len(search_query) < 3:
            search_query = question
        return search_query, resp

    def step_2_retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Step 2: Tool call over the paper library.
        """
        k = top_k or self.top_k
        return self.retriever.retrieve(query, top_k=k)

    async def step_3_decide(
        self, question: str, results: List[SearchResult]
    ) -> Tuple[str, str, LLMResponse]:
        """
        Step 3: Decide (Branch/Loop)
        Returns:
          decision: 'sufficient' | 'weak' | 'unsupported'
          rewrite_query: alternate query if weak
        """
        if not results or all(r.score <= 0.0 for r in results):
            return "unsupported", "", LLMResponse(content="", first_token_ms=0, total_ms=0, status="ok", status_code=200, tokens_in=0, tokens_out=0)

        passages_text = "\n\n".join(
            f"{r.citation}:\n{r.text}" for r in results[:self.top_k]
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Determine if the provided passages contain enough factual evidence to directly answer the user's question.\n"
                    "Respond with a JSON object:\n"
                    "{\n"
                    '  "decision": "sufficient" | "weak" | "unsupported",\n'
                    '  "reason": "brief reason",\n'
                    '  "alternate_query": "better search query if weak, else empty"\n'
                    "}\n"
                    "If the question cannot be answered by the library, select 'unsupported'. Output valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nPassages:\n{passages_text}",
            },
        ]
        resp = await self.llm.generate(messages, temperature=0.0, max_tokens=100)
        decision = "sufficient"
        alternate_query = ""
        try:
            # Extract JSON from output
            match = re.search(r"\{.*\}", resp.content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                decision = data.get("decision", "sufficient").lower()
                alternate_query = data.get("alternate_query", "")
        except Exception:
            decision = "sufficient"

        return decision, alternate_query, resp

    async def step_4_draft(
        self, question: str, results: List[SearchResult], critique: Optional[str] = None
    ) -> Tuple[str, LLMResponse]:
        """
        Step 4: Draft a strictly cited answer using only retrieved passages.
        If unsupported, states 'The papers do not say'.
        """
        if not results:
            return "The papers do not say.", LLMResponse(
                content="The papers do not say.",
                first_token_ms=0,
                total_ms=0,
                status="ok",
                status_code=200,
                tokens_in=0,
                tokens_out=6,
            )

        passages_text = "\n\n".join(
            f"Passage {idx+1} {r.citation}:\n{r.text}" for idx, r in enumerate(results)
        )
        system_prompt = (
            "You are a strict, factual ML research assistant answering questions ONLY from three papers:\n"
            "1. 'Attention Is All You Need'\n"
            "2. 'Moshi: a speech-text foundation model for real-time dialogue'\n"
            "3. 'Dynamic Chunking for End-to-End Hierarchical Sequence Modeling'\n\n"
            "RULES:\n"
            "1. Base your answer EXCLUSIVELY on the provided passages.\n"
            "2. Every claim must cite its source using the exact format: [Paper Title, Sec X, p. N].\n"
            "3. If the provided passages do NOT contain the answer, you must respond: 'The papers do not say.'\n"
            "4. NEVER guess or hallucinate."
        )
        if critique:
            system_prompt += f"\n\nNOTE: A previous draft failed verification: {critique}. Redraft carefully."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nPassages:\n{passages_text}"},
        ]
        resp = await self.llm.generate(messages, temperature=0.0, max_tokens=512)
        return resp.content.strip(), resp

    async def step_5_verify(
        self, question: str, draft: str, results: List[SearchResult]
    ) -> Tuple[bool, str, LLMResponse]:
        """
        Step 5: Verify that every claim and citation is faithfully grounded.
        Returns: (passed: bool, critique: str, LLMResponse)
        """
        if self.skip_verification:
            return True, "", LLMResponse(content="", first_token_ms=0, total_ms=0, status="ok", status_code=200, tokens_in=0, tokens_out=0)

        # If answer is an honest refusal, verification passes immediately
        if "the papers do not say" in draft.lower():
            return True, "", LLMResponse(content="", first_token_ms=0, total_ms=0, status="ok", status_code=200, tokens_in=0, tokens_out=0)

        passages_text = "\n\n".join(
            f"{r.citation}:\n{r.text}" for r in results
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict fact-checker verifying a generated research answer.\n"
                    "Verify that:\n"
                    "1. Every factual statement is directly supported by the provided passages.\n"
                    "2. Every citation [Paper, Sec X, p. N] accurately corresponds to the text.\n"
                    "3. No external knowledge or extrapolation was used.\n\n"
                    "Respond with a JSON object:\n"
                    "{\n"
                    '  "passed": true | false,\n'
                    '  "critique": "brief explanation if failed, else empty"\n'
                    "}\n"
                    "Output valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nDraft Answer:\n{draft}\n\nPassages:\n{passages_text}",
            },
        ]
        resp = await self.llm.generate(messages, temperature=0.0, max_tokens=150)
        passed = True
        critique = ""
        try:
            match = re.search(r"\{.*\}", resp.content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                passed = bool(data.get("passed", True))
                critique = data.get("critique", "")
        except Exception:
            passed = True

        return passed, critique, resp

    async def run(
        self,
        question: str,
        scenario: str = "interactive",
        request_id: Optional[str] = None,
        queue_wait_ms: float = 0.0,
        log_to_csv_now: bool = True,
    ) -> PipelineResult:
        """
        Execute the full 5-step pipeline end-to-end with timing instrumentation.
        """
        req_id = request_id or f"req_{uuid.uuid4().hex[:8]}"
        start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        t0 = time.perf_counter()

        timing = TimingBreakdown(queue_wait_ms=queue_wait_ms)
        tokens_in = 0
        tokens_out = 0
        retries = 0
        loop_count = 0
        first_token_ms: float = 0.0
        final_status = "ok"

        # Step 1: Understand
        t_s1 = time.perf_counter()
        search_query, u_resp = await self.step_1_understand(question)
        timing.understand_ms = (time.perf_counter() - t_s1) * 1000.0
        tokens_in += u_resp.tokens_in
        tokens_out += u_resp.tokens_out
        if u_resp.status != "ok":
            final_status = u_resp.status

        # Step 2: Retrieve (Initial)
        t_s2 = time.perf_counter()
        retrieved_docs = self.step_2_retrieve(search_query)
        timing.retrieve_ms = (time.perf_counter() - t_s2) * 1000.0

        # Step 3: Decide (Branch / Loop)
        t_s3 = time.perf_counter()
        decision, alternate_query, d_resp = await self.step_3_decide(question, retrieved_docs)
        timing.decide_ms = (time.perf_counter() - t_s3) * 1000.0
        tokens_in += d_resp.tokens_in
        tokens_out += d_resp.tokens_out
        if d_resp.status != "ok":
            final_status = d_resp.status

        # Loop: Re-retrieve if decision is weak and loops allowed
        while decision == "weak" and loop_count < self.max_loops:
            loop_count += 1
            retries += 1
            t_loop_start = time.perf_counter()
            query_to_use = alternate_query if alternate_query else f"{question} details"
            more_docs = self.step_2_retrieve(query_to_use)
            # Merge unique chunks
            seen_cids = {d.chunk_id for d in retrieved_docs}
            for d in more_docs:
                if d.chunk_id not in seen_cids:
                    retrieved_docs.append(d)
                    seen_cids.add(d.chunk_id)

            decision, alternate_query, loop_d_resp = await self.step_3_decide(question, retrieved_docs)
            timing.retry_time_ms += (time.perf_counter() - t_loop_start) * 1000.0
            tokens_in += loop_d_resp.tokens_in
            tokens_out += loop_d_resp.tokens_out

        # Branch to Refusal if unsupported
        if decision == "unsupported":
            final_answer = "The papers do not say."
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing.total_ms = total_ms
            res = PipelineResult(
                request_id=req_id,
                scenario=scenario,
                source=self.llm.backend,
                start_time=start_iso,
                question=question,
                final_answer=final_answer,
                status=final_status,
                first_token_ms=timing.understand_ms,
                total_ms=total_ms,
                retries=retries,
                tokens_in=tokens_in,
                tokens_out=tokens_out + 6,
                retrieval_setting=self.retrieval_setting,
                answer_grade="not_graded",
                citations=[],
                timing=timing,
                verification_passed=True,
                loop_count=loop_count,
            )
            log_result_to_csv(res)
            return res

        # Step 4: Draft Answer
        t_s4 = time.perf_counter()
        draft, draft_resp = await self.step_4_draft(question, retrieved_docs[:self.top_k])
        timing.draft_ms = (time.perf_counter() - t_s4) * 1000.0
        tokens_in += draft_resp.tokens_in
        tokens_out += draft_resp.tokens_out
        first_token_ms = draft_resp.first_token_ms
        if draft_resp.status != "ok":
            final_status = draft_resp.status

        # Step 5: Verify
        t_s5 = time.perf_counter()
        passed, critique, v_resp = await self.step_5_verify(question, draft, retrieved_docs[:self.top_k])
        timing.verify_ms = (time.perf_counter() - t_s5) * 1000.0
        tokens_in += v_resp.tokens_in
        tokens_out += v_resp.tokens_out
        if v_resp.status != "ok":
            final_status = v_resp.status

        final_answer = draft
        # Redraft once on failure
        if not passed and not self.skip_verification:
            retries += 1
            t_redraft = time.perf_counter()
            redraft, redraft_resp = await self.step_4_draft(
                question, retrieved_docs[:self.top_k], critique=critique
            )
            timing.retry_time_ms += (time.perf_counter() - t_redraft) * 1000.0
            tokens_in += redraft_resp.tokens_in
            tokens_out += redraft_resp.tokens_out

            # Second verification
            t_v2 = time.perf_counter()
            passed2, _, v2_resp = await self.step_5_verify(question, redraft, retrieved_docs[:self.top_k])
            timing.retry_time_ms += (time.perf_counter() - t_v2) * 1000.0
            tokens_in += v2_resp.tokens_in
            tokens_out += v2_resp.tokens_out

            if passed2:
                final_answer = redraft
            else:
                final_answer = "The papers do not say."
                passed = False

        total_ms = (time.perf_counter() - t0) * 1000.0
        timing.total_ms = total_ms

        # Extract citations like [Paper, Sec X, p. N]
        found_citations = re.findall(r"\[[^\]]+?,\s*p\.\s*\d+\]", final_answer)

        result = PipelineResult(
            request_id=req_id,
            scenario=scenario,
            source=self.llm.backend,
            start_time=start_iso,
            question=question,
            final_answer=final_answer,
            status=final_status,
            first_token_ms=first_token_ms,
            total_ms=total_ms,
            retries=retries,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            retrieval_setting=self.retrieval_setting,
            answer_grade="not_graded",
            citations=found_citations,
            timing=timing,
            verification_passed=passed,
            loop_count=loop_count,
        )

        if log_to_csv_now:
            log_result_to_csv(result)
        return result


async def main():
    parser = argparse.ArgumentParser(description="Run The Relay pipeline on a single question")
    parser.add_argument("--question", type=str, required=True, help="Question to ask the pipeline")
    parser.add_argument("--backend", type=str, choices=["real", "stand-in"], default="real")
    parser.add_argument("--model", type=str, default="qwen/qwen3.8-27b")
    parser.add_argument("--retriever", type=str, choices=["bm25", "embedding", "hybrid"], default="bm25")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--skip-verification", action="store_true", help="Skip verifier (degraded mode)")
    parser.add_argument("--max-requests", type=int, default=10, help="Hard cap on LLM calls")
    args = parser.parse_args()

    client = LLMClient(
        backend=args.backend,
        model=args.model,
        max_requests=args.max_requests,
    )
    pipeline = RelayPipeline(
        llm_client=client,
        retriever_mode=args.retriever,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        skip_verification=args.skip_verification,
    )

    print(f"\n--- Running Pipeline (Backend: {args.backend}, Model: {args.model}) ---")
    print(f"Question: {args.question}")
    result = await pipeline.run(args.question)
    print(f"\nStatus: {result.status}")
    print(f"Total time: {result.total_ms:.1f} ms (TTFT: {result.first_token_ms:.1f} ms)")
    print(f"Tokens in: {result.tokens_in} | Tokens out: {result.tokens_out} | Retries: {result.retries}")
    print(f"Timing Breakdown: Queue: {result.timing.queue_wait_ms:.1f}ms | Understand: {result.timing.understand_ms:.1f}ms | Retrieve: {result.timing.retrieve_ms:.1f}ms | Decide: {result.timing.decide_ms:.1f}ms | Draft: {result.timing.draft_ms:.1f}ms | Verify: {result.timing.verify_ms:.1f}ms | Retry/Loop: {result.timing.retry_time_ms:.1f}ms")
    print(f"\nAnswer:\n{result.final_answer}")


if __name__ == "__main__":
    asyncio.run(main())
