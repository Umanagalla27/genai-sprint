from src.rag.reranker import RerankerService


def test_reranker_service_orders_by_relevance():
    reranker = RerankerService()

    candidates = [
        {"doc_id": "doc1", "chunk_id": 0, "content": "Bananas are yellow fruits packed with potassium and vitamin B6."},
        {"doc_id": "doc2", "chunk_id": 0, "content": "Backpropagation computes gradients through the chain rule in neural networks."},
        {"doc_id": "doc3", "chunk_id": 0, "content": "Apples and oranges are popular orchard fruits grown worldwide."},
    ]

    query = "How do artificial neural networks calculate gradients?"

    # The reranker must score the backpropagation chunk highest
    reranked = reranker.rerank(query=query, candidate_chunks=candidates, top_k=1)
    
    assert len(reranked) == 1
    assert reranked[0]["doc_id"] == "doc2"
    assert "rerank_score" in reranked[0]
