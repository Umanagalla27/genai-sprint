import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25 keyword matching."""
    return re.findall(r"\w+", text.lower())


class HybridSearcher:
    """
    Combines Dense Semantic Retrieval with Sparse BM25 Keyword Search
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None

    def fit_corpus(self, chunks: List[Dict[str, Any]]) -> None:
        """Fit BM25 index on chunk corpus."""
        self.corpus_chunks = chunks
        tokenized_corpus = [tokenize(c["content"]) for c in chunks]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)

    def search_sparse(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Run BM25 keyword search."""
        if not self.bm25 or not self.corpus_chunks:
            return []
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in ranked_indices:
            if scores[idx] > 0:  # Only include if there was at least 1 keyword match
                results.append({
                    **self.corpus_chunks[idx],
                    "sparse_score": float(scores[idx]),
                })
        return results

    def fuse_rrf(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Merge dense and sparse search rankings using Reciprocal Rank Fusion.
        RRF Score = 1 / (k + dense_rank) + 1 / (k + sparse_rank)
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # 1. Process dense rankings
        for rank, chunk in enumerate(dense_results):
            key = f"{chunk['doc_id']}_{chunk['chunk_id']}"
            chunk_map[key] = chunk
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (self.rrf_k + (rank + 1)))

        # 2. Process sparse rankings
        for rank, chunk in enumerate(sparse_results):
            key = f"{chunk['doc_id']}_{chunk['chunk_id']}"
            if key not in chunk_map:
                chunk_map[key] = chunk
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (self.rrf_k + (rank + 1)))

        # 3. Sort by fused score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:top_k]

        fused_results = []
        for key in sorted_keys:
            chunk = chunk_map[key]
            fused_results.append({
                **chunk,
                "rrf_score": round(rrf_scores[key], 5),
            })

        return fused_results
