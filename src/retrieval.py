"""
src/retrieval.py - Retrieval engine for The Relay.

Supported retrieval modes:
  1. BM25 (Rank-BM25 Okapi) - Baseline lexical retriever
  2. Dense Embedding (BAAI/bge-small-en-v1.5) - Semantic retriever with local disk caching
  3. Hybrid (Reciprocal Rank Fusion of BM25 + Dense)
  4. Optional Cross-Encoder Re-ranker (cross-encoder/ms-marco-MiniLM-L-6-v2)

All retrieved results preserve structured metadata and strict citations:
  [Paper Title, Sec X, p. N]
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from rank_bm25 import BM25Okapi

from src.chunking import TextChunk, chunk_library


@dataclass
class SearchResult:
    chunk_id: str
    paper_key: str
    paper_title: str
    arxiv_id: str
    section: str
    page: int
    text: str
    citation: str
    score: float
    rank: int


def tokenize_bm25(text: str) -> List[str]:
    """Tokenize and normalize text for BM25 matching."""
    text = text.lower()
    # Replace math symbols / hyphens with spaces or alphanumeric tokens
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text)
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


class BM25Retriever:
    """Pure lexical retriever based on BM25Okapi."""

    def __init__(self, chunks: List[TextChunk]):
        self.chunks = chunks
        self.corpus_tokens = [tokenize_bm25(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        query_tokens = tokenize_bm25(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(doc_scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[idx]
            score = float(doc_scores[idx])
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    paper_key=chunk.paper_key,
                    paper_title=chunk.paper_title,
                    arxiv_id=chunk.arxiv_id,
                    section=chunk.section,
                    page=chunk.page,
                    text=chunk.text,
                    citation=chunk.citation,
                    score=score,
                    rank=rank,
                )
            )
        return results


class EmbeddingRetriever:
    """
    Dense semantic retriever using BAAI/bge-small-en-v1.5.
    Embeddings are cached to disk to eliminate redundant computation.
    """

    def __init__(
        self,
        chunks: List[TextChunk],
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_path: Optional[Path] = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for EmbeddingRetriever. "
                "Install via: pip install sentence-transformers"
            )

        self.chunks = chunks
        self.model_name = model_name
        self.cache_path = cache_path
        self.model = SentenceTransformer(model_name)
        self.chunk_embeddings = self._get_or_compute_embeddings()

    def _get_or_compute_embeddings(self) -> np.ndarray:
        if self.cache_path and self.cache_path.exists():
            print(f"Loading cached embeddings from {self.cache_path}...")
            embeddings = np.load(self.cache_path)
            if embeddings.shape[0] == len(self.chunks):
                return embeddings
            print("Cached embeddings count mismatch. Recomputing...")

        print(f"Computing embeddings for {len(self.chunks)} chunks using {self.model_name}...")
        texts = [c.text for c in self.chunks]
        # BGE models benefit from normalized embeddings for inner product
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_path, embeddings)
            print(f"Saved embeddings cache to {self.cache_path}")

        return embeddings

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        # BGE models recommend query prefix for retrieval
        query_instruction = "Represent this sentence for searching relevant passages: "
        formatted_query = f"{query_instruction}{query.strip()}"
        query_emb = self.model.encode(
            formatted_query, normalize_embeddings=True, show_progress_bar=False
        )
        query_emb = np.array(query_emb, dtype=np.float32)

        # Cosine similarity is dot product when embeddings are L2 normalized
        similarities = np.dot(self.chunk_embeddings, query_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[idx]
            score = float(similarities[idx])
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    paper_key=chunk.paper_key,
                    paper_title=chunk.paper_title,
                    arxiv_id=chunk.arxiv_id,
                    section=chunk.section,
                    page=chunk.page,
                    text=chunk.text,
                    citation=chunk.citation,
                    score=score,
                    rank=rank,
                )
            )
        return results


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and Dense BGE retrieval
    via Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, bm25_retriever: BM25Retriever, dense_retriever: EmbeddingRetriever, rrf_k: int = 60):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k
        self.chunks_by_id = {c.chunk_id: c for c in bm25_retriever.chunks}

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        # Fetch candidate pools from both retrievers
        candidate_k = max(top_k * 3, 20)
        bm25_res = self.bm25.retrieve(query, top_k=candidate_k)
        dense_res = self.dense.retrieve(query, top_k=candidate_k)

        rrf_scores: Dict[str, float] = {}
        for rank, item in enumerate(bm25_res, start=1):
            rrf_scores[item.chunk_id] = rrf_scores.get(item.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, item in enumerate(dense_res, start=1):
            rrf_scores[item.chunk_id] = rrf_scores.get(item.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        results = []
        for rank, cid in enumerate(sorted_ids, start=1):
            chunk = self.chunks_by_id[cid]
            score = rrf_scores[cid]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    paper_key=chunk.paper_key,
                    paper_title=chunk.paper_title,
                    arxiv_id=chunk.arxiv_id,
                    section=chunk.section,
                    page=chunk.page,
                    text=chunk.text,
                    citation=chunk.citation,
                    score=score,
                    rank=rank,
                )
            )
        return results


class CrossEncoderReRanker:
    """
    Optional Cross-Encoder re-ranker.
    Takes candidate search results and re-scores query-document pairs.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
        if not candidates:
            return []

        pairs = [[query, c.text] for c in candidates]
        scores = self.model.predict(pairs)

        indexed_scores = list(zip(candidates, scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for rank, (cand, score) in enumerate(indexed_scores[:top_k], start=1):
            reranked.append(
                SearchResult(
                    chunk_id=cand.chunk_id,
                    paper_key=cand.paper_key,
                    paper_title=cand.paper_title,
                    arxiv_id=cand.arxiv_id,
                    section=cand.section,
                    page=cand.page,
                    text=cand.text,
                    citation=cand.citation,
                    score=float(score),
                    rank=rank,
                )
            )
        return reranked


def get_retriever(
    mode: str = "bm25",
    chunk_size: int = 256,
    overlap: int = 32,
    chunks_path: Optional[Path] = None,
    extracted_pages_path: Path = Path("data/papers/extracted_pages.json"),
) -> Tuple[Any, List[TextChunk]]:
    """Factory function for initializing retrievers."""
    if chunks_path and chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
            chunks = [TextChunk(**rc) for rc in raw_chunks]
    else:
        chunks = chunk_library(
            extracted_pages_path,
            chunk_size_tokens=chunk_size,
            chunk_overlap_tokens=overlap,
            output_path=chunks_path,
        )

    if mode == "bm25":
        retriever = BM25Retriever(chunks)
    elif mode == "embedding":
        cache_file = Path(f"data/papers/bge_embeddings_{chunk_size}_{overlap}.npy")
        retriever = EmbeddingRetriever(chunks, cache_path=cache_file)
    elif mode == "hybrid":
        cache_file = Path(f"data/papers/bge_embeddings_{chunk_size}_{overlap}.npy")
        bm25_r = BM25Retriever(chunks)
        dense_r = EmbeddingRetriever(chunks, cache_path=cache_file)
        retriever = HybridRetriever(bm25_r, dense_r)
    else:
        raise ValueError(f"Unknown retriever mode: {mode}. Choose bm25, embedding, or hybrid.")

    return retriever, chunks


def main():
    parser = argparse.ArgumentParser(description="Test retrieval on the arXiv library")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["bm25", "embedding", "hybrid"],
        default="bm25",
        help="Retrieval mode (default: bm25)",
    )
    parser.add_argument("--chunk-size", type=int, default=256, help="Chunk size in tokens")
    parser.add_argument("--overlap", type=int, default=32, help="Chunk overlap in tokens")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to return")
    parser.add_argument("--rerank", action="store_true", help="Apply cross-encoder re-ranker")
    args = parser.parse_args()

    chunks_file = Path(f"data/papers/chunks_{args.chunk_size}_{args.overlap}.json")
    retriever, _ = get_retriever(
        mode=args.mode,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        chunks_path=chunks_file,
    )

    fetch_k = args.top_k * 2 if args.rerank else args.top_k
    results = retriever.retrieve(args.query, top_k=fetch_k)

    if args.rerank:
        print("Applying CrossEncoder re-ranker...")
        reranker = CrossEncoderReRanker()
        results = reranker.rerank(args.query, results, top_k=args.top_k)

    print(f"\n=== Top {len(results)} Results for: '{args.query}' (Mode: {args.mode}) ===")
    for r in results:
        print(f"\n[Rank {r.rank} | Score: {r.score:.4f}] {r.citation}")
        print(f"Snippet: {r.text[:200]}...")


if __name__ == "__main__":
    main()
