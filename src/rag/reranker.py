from typing import List, Dict, Any
from flashrank import Ranker, RerankRequest


class RerankerService:
    """
    Lightweight Cross-Encoder Reranking Service using FlashRank (ONNX runtime).
    Default model: ms-marco-TinyBERT-L-2-v2 (fast, low memory footprint).
    """

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.ranker = Ranker(model_name=model_name, cache_dir="/tmp/flashrank")

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Takes candidates from first-stage retrieval and scores them jointly with the query
        using full cross-attention. Returns top_k highest-scoring chunks.
        """
        if not candidate_chunks:
            return []

        # Prepare FlashRank input schema: list of dicts with 'id' and 'text'
        passages = []
        chunk_map = {}
        for idx, chunk in enumerate(candidate_chunks):
            passage_id = f"{chunk['doc_id']}_{chunk['chunk_id']}_{idx}"
            passages.append({"id": passage_id, "text": chunk["content"]})
            chunk_map[passage_id] = chunk

        rerank_request = RerankRequest(query=query, passages=passages)
        ranked_results = self.ranker.rerank(rerank_request)

        # Map back to original chunk metadata with new rerank score
        reranked_chunks = []
        for res in ranked_results[:top_k]:
            original_chunk = chunk_map[res["id"]]
            reranked_chunks.append({
                **original_chunk,
                "rerank_score": round(float(res["score"]), 5),
            })

        return reranked_chunks
