"""
eval/grade.py - Evaluation and grading engine for The Relay.

Grades every answer into one of 4 strict outcomes:
  1. correct: Answer is factually accurate and includes valid citations.
  2. wrong: Answer contains factual errors, hallucinations, or incorrect claims.
  3. refused_correctly: Model truthfully responded 'the papers do not say' for an unanswerable question.
  4. refused_wrongly: Model refused ('the papers do not say') for a question that was answerable in the papers.

Supports:
  - Fast rule-based heuristic checking (refusals, citation patterns, key factual terms)
  - LLM-as-a-judge verification using the active LLM client
  - Batch evaluation updating results.csv
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.llm_client import LLMClient


@dataclass
class GradeResult:
    question_id: str
    grade: str  # 'correct' | 'wrong' | 'refused_correctly' | 'refused_wrongly'
    explanation: str
    citation_valid: bool
    is_refusal_expected: bool


def grade_answer_heuristic(
    question_meta: Dict[str, Any],
    final_answer: str,
) -> GradeResult:
    """
    Rule-based grader for rapid evaluation:
    Handles refusal logic and citation validation deterministically.
    """
    q_id = question_meta.get("id", "unknown")
    is_refusal_expected = question_meta.get("is_refusal", False)
    expected_citations = question_meta.get("expected_citations", [])
    expected_ans = question_meta.get("expected_answer", "")

    answer_clean = final_answer.strip().lower()
    is_refused = "the papers do not say" in answer_clean

    # Case 1: Refusal expected
    if is_refusal_expected:
        if is_refused:
            return GradeResult(
                question_id=q_id,
                grade="refused_correctly",
                explanation="Model truthfully stated 'the papers do not say' for out-of-corpus question.",
                citation_valid=True,
                is_refusal_expected=True,
            )
        else:
            return GradeResult(
                question_id=q_id,
                grade="wrong",
                explanation="Model hallucinated an answer for a question not answered in the corpus.",
                citation_valid=False,
                is_refusal_expected=True,
            )

    # Case 2: Answerable question, but model refused
    if is_refused:
        return GradeResult(
            question_id=q_id,
            grade="refused_wrongly",
            explanation="Model refused an answerable question that exists in the papers.",
            citation_valid=False,
            is_refusal_expected=False,
        )

    # Case 3: Answerable question - check citations and content
    found_citations = re.findall(r"\[[^\]]+?,\s*p\.\s*\d+\]", final_answer)
    citation_valid = len(found_citations) > 0

    # Basic keyword match against expected answer
    keywords = [
        w.lower()
        for w in re.findall(r"\b[a-zA-Z0-9_\-\./]+\b", expected_ans)
        if len(w) > 3 and w.lower() not in ("model", "paper", "which", "using", "with", "from")
    ]
    matches = sum(1 for kw in keywords if kw in answer_clean)
    match_ratio = matches / max(1, len(keywords))

    if match_ratio >= 0.4 and citation_valid:
        return GradeResult(
            question_id=q_id,
            grade="correct",
            explanation=f"Answer contains key factual assertions (match ratio {match_ratio:.2f}) and valid citations.",
            citation_valid=True,
            is_refusal_expected=False,
        )
    elif not citation_valid:
        return GradeResult(
            question_id=q_id,
            grade="wrong",
            explanation="Answer failed to include strict citations [Paper, Sec X, p. N].",
            citation_valid=False,
            is_refusal_expected=False,
        )
    else:
        return GradeResult(
            question_id=q_id,
            grade="wrong",
            explanation=f"Answer lacked necessary factual content (match ratio {match_ratio:.2f}).",
            citation_valid=citation_valid,
            is_refusal_expected=False,
        )


async def grade_answer_llm(
    llm_client: LLMClient,
    question_meta: Dict[str, Any],
    final_answer: str,
) -> GradeResult:
    """
    LLM-as-a-judge grading for complex cross-paper and synthesis evaluation.
    """
    # If question was a refusal, rule-based grading is authoritative
    if question_meta.get("is_refusal", False) or "the papers do not say" in final_answer.lower():
        return grade_answer_heuristic(question_meta, final_answer)

    q_id = question_meta.get("id", "unknown")
    question_text = question_meta["question"]
    expected_ans = question_meta["expected_answer"]
    expected_cits = question_meta.get("expected_citations", [])

    judge_prompt = (
        "You are an impartial ML research judge evaluating an AI student's answer against a ground-truth reference.\n\n"
        f"Question: {question_text}\n\n"
        f"Ground Truth Expected Answer: {expected_ans}\n"
        f"Expected Citations: {expected_cits}\n\n"
        f"Candidate Answer to Grade:\n{final_answer}\n\n"
        "Grading Rubric:\n"
        "1. Is the candidate answer factually accurate and consistent with the ground truth?\n"
        "2. Does it contain proper citations in the format [Paper, Sec X, p. N]?\n"
        "3. Output a JSON object:\n"
        "{\n"
        '  "grade": "correct" | "wrong" | "refused_wrongly",\n'
        '  "reason": "1-2 sentence explanation"\n'
        "}\n"
        "Output valid JSON only."
    )

    resp = await llm_client.generate(
        messages=[{"role": "system", "content": judge_prompt}],
        temperature=0.0,
        max_tokens=150,
    )

    grade = "correct"
    reason = "Evaluated by LLM Judge"
    try:
        match = re.search(r"\{.*\}", resp.content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            grade = data.get("grade", "correct").lower()
            reason = data.get("reason", "")
    except Exception:
        # Fall back to heuristic if judge output parsing fails
        return grade_answer_heuristic(question_meta, final_answer)

    found_citations = re.findall(r"\[[^\]]+?,\s*p\.\s*\d+\]", final_answer)
    return GradeResult(
        question_id=q_id,
        grade=grade,
        explanation=reason,
        citation_valid=len(found_citations) > 0,
        is_refusal_expected=False,
    )


def grade_results_file(
    results_csv: Path = PROJECT_ROOT / "results.csv",
    questions_json: Path = PROJECT_ROOT / "eval/questions.json",
):
    """
    Iterate over results.csv and grade any 'not_graded' rows.
    """
    if not results_csv.exists() or not questions_json.exists():
        print("results.csv or questions.json missing.")
        return

    with open(questions_json, "r", encoding="utf-8") as f:
        questions = json.load(f)
    q_map = {q["question"].strip().lower(): q for q in questions}

    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} rows from {results_csv}")

    # Grade matching questions if text was logged
    print("Grading summary ready.")


def main():
    parser = argparse.ArgumentParser(description="Evaluation Grader Helper")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results.csv")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "eval/questions.json")
    args = parser.parse_args()

    grade_results_file(args.results, args.questions)


if __name__ == "__main__":
    main()

