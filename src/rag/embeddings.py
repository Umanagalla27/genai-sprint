from typing import List
from fastembed import TextEmbedding


class EmbeddingService:
    """
    Production-grade local embedding service using FastEmbed (ONNX runtime).
    Default model: BAAI/bge-small-en-v1.5 (384 dimensions, normalized).
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.model = TextEmbedding(model_name=self.model_name)
        self.dimension = 384

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single query string."""
        if not text.strip():
            raise ValueError("Cannot embed empty text.")
        embeddings = list(self.model.embed([text]))
        return embeddings[0].tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of strings efficiently."""
        if not texts:
            return []
        embeddings = list(self.model.embed(texts))
        return [emb.tolist() for emb in embeddings]
