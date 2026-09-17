import pytest
from src.rag.chunker import RecursiveChunker

def test_empty_text():
    chunker = RecursiveChunker()
    chunks = chunker.chunk_text("", doc_id="test_0")
    assert chunks == []

def test_invalid_overlap():
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=100, chunk_overlap=150)

def test_chunk_metadata_assignment():
    text = "Sentence one. Sentence two. Sentence three."
    chunker = RecursiveChunker(chunk_size=30, chunk_overlap=5)
    chunks = chunker.chunk_text(text, doc_id="doc_xyz", extra_metadata={"author": "Vaswani"})
    assert len(chunks) > 0
    assert chunks[0].doc_id == "doc_xyz"
    assert chunks[0].metadata["author"] == "Vaswani"
    assert "chunk_index" in chunks[0].metadata
