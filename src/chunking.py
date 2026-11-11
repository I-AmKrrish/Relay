"""
src/chunking.py - Configurable chunking preserving paper, section, and page metadata.

Citation format: [Paper Title, Sec X, p. N]
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TextChunk:
    chunk_id: str
    paper_key: str
    paper_title: str
    arxiv_id: str
    section: str
    page: int
    text: str
    citation: str
    token_count_approx: int
    word_count: int
    char_count: int


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using whitespace and punctuation heuristics.
    In English technical literature, ~1 word ≈ 1.3 to 1.35 tokens.
    """
    words = len(text.split())
    # Sub-word penalty: technical punctuation, math symbols, and camelCase
    puncts = len(re.findall(r"[\.,;:!\?_=\+\-\*\\/\[\]\{\}\(\)<>]", text))
    return int(words * 1.25 + puncts * 0.15)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while respecting common abbreviations."""
    # Common academic abbreviations to preserve
    protected = text
    abbrevs = ["e.g.", "i.e.", "et al.", "Fig.", "Sec.", "Eq.", "vs.", "al.", "Ref."]
    for idx, abb in enumerate(abbrevs):
        protected = protected.replace(abb, f"__ABB_{idx}__")

    # Split on sentence terminals followed by whitespace
    raw_sentences = re.split(r"(?<=[.!?])\s+", protected)
    sentences = []
    for s in raw_sentences:
        s_restored = s
        for idx, abb in enumerate(abbrevs):
            s_restored = s_restored.replace(f"__ABB_{idx}__", abb)
        s_restored = s_restored.strip()
        if s_restored:
            sentences.append(s_restored)

    return sentences


def create_chunks_for_page(
    page_record: Dict[str, Any],
    chunk_size_tokens: int = 256,
    chunk_overlap_tokens: int = 32,
) -> List[TextChunk]:
    """
    Chunk the text of a single page while preserving exact section and page citations.
    Uses sentence-boundary aware sliding window.
    """
    text = page_record["clean_text"]
    paper_key = page_record["paper_key"]
    paper_title = page_record["paper_title"]
    arxiv_id = page_record["arxiv_id"]
    page_num = page_record["page_number"]
    section = page_record["active_section"]

    if not text.strip():
        return []

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[TextChunk] = []
    current_sentences: List[str] = []
    current_tokens = 0
    chunk_idx = 0

    for sent in sentences:
        sent_tokens = max(1, estimate_tokens(sent))
        if current_tokens + sent_tokens > chunk_size_tokens and current_sentences:
            chunk_text = " ".join(current_sentences)
            citation = f"[{paper_title}, Sec {section}, p. {page_num}]"
            chunk = TextChunk(
                chunk_id=f"{paper_key}_p{page_num}_c{chunk_idx}",
                paper_key=paper_key,
                paper_title=paper_title,
                arxiv_id=arxiv_id,
                section=section,
                page=page_num,
                text=chunk_text,
                citation=citation,
                token_count_approx=current_tokens,
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text),
            )
            chunks.append(chunk)
            chunk_idx += 1

            # Prepare overlap window for next chunk
            overlap_sentences: List[str] = []
            overlap_tokens = 0
            for prev_sent in reversed(current_sentences):
                prev_toks = max(1, estimate_tokens(prev_sent))
                if overlap_tokens + prev_toks <= chunk_overlap_tokens:
                    overlap_sentences.insert(0, prev_sent)
                    overlap_tokens += prev_toks
                else:
                    break

            current_sentences = overlap_sentences + [sent]
            current_tokens = overlap_tokens + sent_tokens
        else:
            current_sentences.append(sent)
            current_tokens += sent_tokens

    # Flush final remaining sentences
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        citation = f"[{paper_title}, Sec {section}, p. {page_num}]"
        chunk = TextChunk(
            chunk_id=f"{paper_key}_p{page_num}_c{chunk_idx}",
            paper_key=paper_key,
            paper_title=paper_title,
            arxiv_id=arxiv_id,
            section=section,
            page=page_num,
            text=chunk_text,
            citation=citation,
            token_count_approx=current_tokens,
            word_count=len(chunk_text.split()),
            char_count=len(chunk_text),
        )
        chunks.append(chunk)

    return chunks


def chunk_library(
    extracted_pages_path: Path,
    chunk_size_tokens: int = 256,
    chunk_overlap_tokens: int = 32,
    output_path: Optional[Path] = None,
) -> List[TextChunk]:
    """
    Chunk all extracted pages from the library according to the specified size and overlap.
    """
    with open(extracted_pages_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    all_chunks: List[TextChunk] = []
    for page in pages:
        page_chunks = create_chunks_for_page(
            page,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )
        all_chunks.extend(page_chunks)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in all_chunks], f, indent=2, ensure_ascii=False)
        print(
            f"Saved {len(all_chunks)} chunks (size={chunk_size_tokens}, overlap={chunk_overlap_tokens}) to {output_path}"
        )

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Chunk extracted arXiv library")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/papers/extracted_pages.json"),
        help="Path to extracted_pages.json",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Target chunk size in estimated tokens (default: 256)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=32,
        help="Chunk overlap in estimated tokens (default: 32)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for chunks JSON (default: data/papers/chunks_{size}_{overlap}.json)",
    )
    args = parser.parse_args()

    out = (
        args.output
        if args.output
        else Path(f"data/papers/chunks_{args.chunk_size}_{args.overlap}.json")
    )
    chunks = chunk_library(
        args.input,
        chunk_size_tokens=args.chunk_size,
        chunk_overlap_tokens=args.overlap,
        output_path=out,
    )
    print(f"Total chunks created: {len(chunks)}")
    if chunks:
        print("\nSample chunk citation:")
        print(f"  ID: {chunks[0].chunk_id}")
        print(f"  Citation: {chunks[0].citation}")
        print(f"  Tokens (approx): {chunks[0].token_count_approx}")
        print(f"  Text preview: {chunks[0].text[:120]}...")


if __name__ == "__main__":
    main()

