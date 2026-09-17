from src.rag.chunker import DocumentChunk
from src.rag.embeddings import EmbeddingService
from src.rag.vector_store import QdrantVectorStore


def test_embedding_service_dimensions():
    embedder = EmbeddingService()
    vec = embedder.embed_text("Test semantic embedding vector.")
    assert len(vec) == 384
    assert isinstance(vec[0], float)


def test_vector_store_lifecycle():
    store = QdrantVectorStore(collection_name="test_collection", location=":memory:")
    assert store.count() == 0

    chunk = DocumentChunk(
        doc_id="doc_1",
        chunk_id=0,
        content="Testing vector upsert in memory.",
        char_count=32,
        metadata={"source": "unit_test"},
    )
    fake_vector = [0.1] * 384

    inserted = store.upsert_chunks([chunk], [fake_vector])
    assert inserted == 1
    assert store.count() == 1
