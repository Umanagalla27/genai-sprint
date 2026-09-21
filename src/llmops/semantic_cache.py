from typing import Dict, Any, List, Optional, Callable, Tuple
import time
import uuid
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from src.rag.embeddings import EmbeddingService


class CacheStats(BaseModel):
    total_lookups: int = 0
    total_queries: int = 0
    hits: int = 0
    misses: int = 0
    hit_rate_pct: float = 0.0
    cached_entries: int = 0
    similarity_threshold: float = 0.85
    total_saved_latency_ms: float = 0.0


class SemanticCache:
    """
    Semantic Caching layer for LLM applications.
    Uses vector embeddings to identify semantically similar incoming prompts
    and returns cached responses, bypassing expensive and slow LLM generation.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        collection_name: str = "llm_semantic_cache",
        location: str = ":memory:",
        embedder: Optional[EmbeddingService] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.collection_name = collection_name
        self.embedder = embedder or EmbeddingService()
        self.client = QdrantClient(location=location)

        # Performance & telemetry counters
        self.hits: int = 0
        self.misses: int = 0
        self.total_saved_latency_ms: float = 0.0

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ensure Qdrant collection exists with Cosine distance."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedder.dimension,
                    distance=Distance.COSINE,
                ),
            )

    def lookup(self, query: str) -> Tuple[bool, str, float]:
        """
        Search for a semantically similar cached query.
        Returns: (hit: bool, response: str, score: float)
        If hit: (True, cached_response, similarity_score)
        If miss: (False, "", 0.0)
        """
        if not query.strip():
            self.misses += 1
            return False, "", 0.0

        query_vector = self.embedder.embed_text(query)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=1,
        )

        if not results.points:
            self.misses += 1
            return False, "", 0.0

        top_point = results.points[0]
        similarity_score = float(top_point.score)

        if similarity_score >= self.similarity_threshold:
            self.hits += 1
            resp = top_point.payload.get("response", "")
            return True, resp, round(similarity_score, 4)

        self.misses += 1
        return False, "", round(similarity_score, 4)

    def set(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a query-response pair in the semantic cache.
        Returns the generated entry ID.
        """
        if not query.strip() or not response.strip():
            raise ValueError("Query and response must not be empty.")

        query_vector = self.embedder.embed_text(query)
        point_id = str(uuid.uuid4())

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=query_vector,
                    payload={
                        "query": query,
                        "response": response,
                        "timestamp": time.time(),
                        "metadata": metadata or {},
                    },
                )
            ],
        )
        return point_id

    # Alias for set
    store = set

    def get_stats(self) -> CacheStats:
        """Return cache telemetry and efficiency statistics as a CacheStats object."""
        total_lookups = self.hits + self.misses
        hit_rate = (self.hits / total_lookups * 100.0) if total_lookups > 0 else 0.0
        count_res = self.client.count(collection_name=self.collection_name)

        return CacheStats(
            total_lookups=total_lookups,
            total_queries=total_lookups,
            hits=self.hits,
            misses=self.misses,
            hit_rate_pct=round(hit_rate, 2),
            cached_entries=count_res.count,
            similarity_threshold=self.similarity_threshold,
            total_saved_latency_ms=round(self.total_saved_latency_ms, 2),
        )

    def stats(self) -> Dict[str, Any]:
        """Dictionary representation of stats for backward compatibility."""
        return self.get_stats().model_dump()

    def query_or_compute(
        self,
        query: str,
        compute_fn: Callable[[str], str],
        metadata: Optional[Dict[str, Any]] = None,
        estimated_llm_latency_ms: float = 1200.0,
    ) -> Tuple[str, bool, float, float]:
        """
        Convenience wrapper:
        1. Checks cache for semantically matching query.
        2. If HIT: returns cached response immediately.
        3. If MISS: computes response via `compute_fn`, caches it, and returns it.
        Returns: (response, is_hit, latency_ms, similarity_score)
        """
        start_time = time.perf_counter()
        hit, resp, score = self.lookup(query)

        if hit:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.total_saved_latency_ms += max(0.0, estimated_llm_latency_ms - elapsed_ms)
            return resp, True, elapsed_ms, score

        # Compute fresh response
        response = compute_fn(query)
        self.set(query, response, metadata=metadata)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return response, False, elapsed_ms, score

    def clear(self) -> None:
        """Clear all cached entries."""
        self.client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection()
        self.hits = 0
        self.misses = 0
        self.total_saved_latency_ms = 0.0
