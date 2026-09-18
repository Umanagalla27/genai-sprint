from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from src.rag.chunker import DocumentChunk


class QdrantVectorStore:
    """
    Manages collection lifecycle and vector upserts in Qdrant.
    Supports in-memory mode for fast testing and disk persistence for production.
    """

    def __init__(
        self,
        collection_name: str = "production_rag_docs",
        location: str = ":memory:",
        vector_dim: int = 384,
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.client = QdrantClient(location=location)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it does not exist with Cosine distance."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )

    def upsert_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> int:
        """
        Store chunks with vector embeddings and full payload metadata.
        Returns the number of points inserted.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks count must match embeddings count.")

        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=idx + 1,
                vector=vector,
                payload={
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "char_count": chunk.char_count,
                    **chunk.metadata,
                },
            )
            points.append(point)

        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def count(self) -> int:
        """Return total number of points in the collection."""
        res = self.client.count(collection_name=self.collection_name)
        return res.count

    def search(self, query_vector: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        """
        Search for top-k closest vectors using Cosine similarity.
        Returns list of payload dictionaries with relevance score.
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        )
        hits = []
        for point in results.points:
            hits.append({
                "score": point.score,
                "doc_id": point.payload.get("doc_id"),
                "chunk_id": point.payload.get("chunk_id"),
                "content": point.payload.get("content"),
                "metadata": {k: v for k, v in point.payload.items() if k not in ["content", "doc_id", "chunk_id"]},
            })
        return hits

