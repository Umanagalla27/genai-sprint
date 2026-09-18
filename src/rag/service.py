import os
from typing import List, Dict, Any
from groq import Groq

from src.rag.chunker import RecursiveChunker
from src.rag.embeddings import EmbeddingService
from src.rag.vector_store import QdrantVectorStore
from src.rag.hybrid_search import HybridSearcher
from src.rag.reranker import RerankerService


class RAGService:
    """End-to-end RAG orchestrator for document indexing, retrieval, and answer synthesis."""

    def __init__(self, collection_name: str = "production_rag_docs"):
        self.chunker = RecursiveChunker(chunk_size=300, chunk_overlap=40)
        self.embedder = EmbeddingService()
        self.vector_store = QdrantVectorStore(collection_name=collection_name)
        self.hybrid_searcher = HybridSearcher()
        self.raw_chunks_cache: List[Dict[str, Any]] = []
        self.reranker = RerankerService()

    def ingest_document(self, text: str, doc_id: str, metadata: Dict[str, Any] | None = None) -> int:
        """Ingest, chunk, embed, and store document in vector DB, updating BM25 index."""
        chunks = self.chunker.chunk_text(text, doc_id=doc_id, extra_metadata=metadata)
        if not chunks:
            return 0

        # Cache for BM25
        for c in chunks:
            self.raw_chunks_cache.append({
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "content": c.content,
                "metadata": c.metadata,
            })
        self.hybrid_searcher.fit_corpus(self.raw_chunks_cache)

        vectors = self.embedder.embed_batch([c.content for c in chunks])
        return self.vector_store.upsert_chunks(chunks, vectors)

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Embed query and search vector store for top-k chunks."""
        query_vector = self.embedder.embed_text(query)
        return self.vector_store.search(query_vector=query_vector, limit=top_k)

    def retrieve_hybrid(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve using both Dense Vectors and Sparse BM25, fused via RRF."""
        # Dense retrieval
        dense_hits = self.retrieve_context(query, top_k=top_k * 2)
        # Sparse retrieval
        sparse_hits = self.hybrid_searcher.search_sparse(query, top_k=top_k * 2)
        # Fuse
        return self.hybrid_searcher.fuse_rrf(dense_hits, sparse_hits, top_k=top_k)

    def retrieve_and_rerank(self, query: str, initial_top_k: int = 10, final_top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Two-stage retrieval pipeline:
        Stage 1: Hybrid Search (Dense + BM25) retrieves top candidate pool (e.g. 10 chunks).
        Stage 2: Cross-Encoder jointly scores candidates against query to return top_k (e.g. 3 chunks).
        """
        # Stage 1: Candidate retrieval
        candidates = self.retrieve_hybrid(query, top_k=initial_top_k)
        if not candidates:
            return []

        # Stage 2: Cross-Encoder reranking
        return self.reranker.rerank(query=query, candidate_chunks=candidates, top_k=final_top_k)

    def answer_query(
        self,
        query: str,
        llm_client: Groq,
        model: str = "openai/gpt-oss-20b",
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Retrieve relevant context and generate a grounded answer."""
        retrieved_chunks = self.retrieve_and_rerank(query, initial_top_k=10, final_top_k=top_k)

        if not retrieved_chunks:
            return {
                "query": query,
                "answer": "No relevant documents found in the knowledge base.",
                "context_used": "",
                "sources": [],
            }

        # Build grounded context string
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            score = chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score", 0.0)))
            context_blocks.append(f"[Source {i+1} - Doc: {chunk['doc_id']} (Score: {score:.3f})]:\n{chunk['content']}")
        context_str = "\n\n".join(context_blocks)

        system_prompt = (
            "You are a helpful and strictly grounded technical assistant. "
            "Answer the user's question using ONLY the provided context below. "
            "If the answer cannot be found in the context, explicitly say: "
            "'I do not have enough information in the provided context to answer this.' "
            "Do not hallucinate or use outside knowledge."
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"

        completion = llm_client.chat.completions.create(
            model=model,
            temperature=0.1,  # Low temperature for strict factual grounding
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = completion.choices[0].message.content

        return {
            "query": query,
            "answer": answer,
            "context_used": context_str,
            "sources": [
                {
                    "doc_id": c["doc_id"],
                    "chunk_id": c["chunk_id"],
                    "score": round(c.get("rerank_score", c.get("rrf_score", c.get("score", 0.0))), 4),
                    "preview": c["content"][:80] + "...",
                }
                for c in retrieved_chunks
            ],
        }
