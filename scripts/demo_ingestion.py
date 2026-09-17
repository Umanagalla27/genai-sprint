import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.chunker import RecursiveChunker
from src.rag.embeddings import EmbeddingService
from src.rag.vector_store import QdrantVectorStore

sample_doc = """
Retrieval-Augmented Generation (RAG) is an architectural pattern that combines an information retrieval system with a generative Large Language Model.
Instead of relying solely on the static knowledge stored in model weights during pretraining, RAG fetches relevant context dynamically from an external knowledge base.

A typical production RAG system has three phases:
Phase 1 is Document Ingestion, where raw files (PDFs, Markdown, Confluence) are parsed, segmented into semantic chunks, and embedded into high-dimensional vector spaces.
Phase 2 is Retrieval, where incoming user questions are converted to embeddings and queried against a vector database using approximate nearest neighbor algorithms like HNSW.
Phase 3 is Generation, where the most relevant chunks are synthesized into an augmented prompt and passed to the LLM with strict grounding instructions to eliminate hallucinations.
"""

# 1. Chunk document
chunker = RecursiveChunker(chunk_size=200, chunk_overlap=30)
chunks = chunker.chunk_text(sample_doc, doc_id="rag_overview_doc")
print(f"Generated {len(chunks)} chunks.")

# 2. Generate embeddings
embedder = EmbeddingService()
contents = [c.content for c in chunks]
print("Generating FastEmbed vectors...")
vectors = embedder.embed_batch(contents)
print(f"Generated {len(vectors)} vectors of dimension {len(vectors[0])}.")

# 3. Store in Qdrant (in-memory)
store = QdrantVectorStore()
inserted_count = store.upsert_chunks(chunks, vectors)
print(f"Successfully upserted {inserted_count} points into Qdrant collection '{store.collection_name}'.")
print(f"Total points in collection: {store.count()}")
