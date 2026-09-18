import pytest
from src.rag.hybrid_search import HybridSearcher, tokenize


def test_tokenize():
    tokens = tokenize("Hello, World! RAG 123.")
    assert tokens == ["hello", "world", "rag", "123"]


def test_hybrid_searcher_sparse():
    searcher = HybridSearcher(rrf_k=60)
    chunks = [
        {"doc_id": "doc1", "chunk_id": 0, "content": "FastAPI is a modern web framework for Python."},
        {"doc_id": "doc2", "chunk_id": 0, "content": "Qdrant is a vector similarity search engine."},
        {"doc_id": "doc3", "chunk_id": 0, "content": "Docker containers run isolated environments."},
    ]
    searcher.fit_corpus(chunks)

    results = searcher.search_sparse("FastAPI web", top_k=2)
    assert len(results) > 0
    assert results[0]["doc_id"] == "doc1"
    assert "sparse_score" in results[0]


def test_hybrid_searcher_fuse_rrf():
    searcher = HybridSearcher(rrf_k=60)
    dense_results = [
        {"doc_id": "doc1", "chunk_id": 0, "content": "Chunk A", "score": 0.95},
        {"doc_id": "doc2", "chunk_id": 0, "content": "Chunk B", "score": 0.85},
    ]
    sparse_results = [
        {"doc_id": "doc2", "chunk_id": 0, "content": "Chunk B", "sparse_score": 1.5},
        {"doc_id": "doc3", "chunk_id": 0, "content": "Chunk C", "sparse_score": 1.2},
    ]

    fused = searcher.fuse_rrf(dense_results, sparse_results, top_k=3)
    assert len(fused) == 3
    # doc2 was present in both, so its RRF score should be highest
    assert fused[0]["doc_id"] == "doc2"
    assert "rrf_score" in fused[0]
