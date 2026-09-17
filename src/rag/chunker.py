from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import re


@dataclass
class DocumentChunk:
    doc_id: str
    chunk_id: int
    content: str
    char_count: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RecursiveChunker:
    """
    Production recursive chunker that splits text hierarchically:
    Paragraphs -> Sentences -> Words -> Characters, while preserving overlap.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] | None = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk_text(self, text: str, doc_id: str, extra_metadata: Dict[str, Any] | None = None) -> List[DocumentChunk]:
        """Splits raw text into structured DocumentChunk objects with overlap and metadata."""
        if not text.strip():
            return []

        raw_chunks = self._split_text(text, self.separators)
        chunks_with_overlap = self._merge_with_overlap(raw_chunks)

        document_chunks = []
        for idx, chunk_content in enumerate(chunks_with_overlap):
            doc_chunk = DocumentChunk(
                doc_id=doc_id,
                chunk_id=idx,
                content=chunk_content.strip(),
                char_count=len(chunk_content.strip()),
                metadata={
                    **(extra_metadata or {}),
                    "chunk_index": idx,
                    "total_chunks_expected": len(chunks_with_overlap),
                },
            )
            document_chunks.append(doc_chunk)

        return document_chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using hierarchy of separators."""
        final_pieces = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if re.search(re.escape(sep), text):
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits = []
        for piece in splits:
            if len(piece) < self.chunk_size:
                good_splits.append(piece)
            else:
                if new_separators:
                    other_splits = self._split_text(piece, new_separators)
                    good_splits.extend(other_splits)
                else:
                    good_splits.append(piece)

        return good_splits

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        """Greedily combine small splits into chunks up to chunk_size, carrying over overlap."""
        chunks = []
        current_chunk = []
        current_len = 0

        for piece in pieces:
            piece_len = len(piece)
            if current_len + piece_len <= self.chunk_size:
                current_chunk.append(piece)
                current_len += piece_len
            else:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Compute overlap buffer from the end of current chunk
                    overlap_buffer = []
                    overlap_len = 0
                    for prev_piece in reversed(current_chunk):
                        if overlap_len + len(prev_piece) <= self.chunk_overlap:
                            overlap_buffer.insert(0, prev_piece)
                            overlap_len += len(prev_piece)
                        else:
                            break
                    current_chunk = overlap_buffer
                    current_len = overlap_len
                current_chunk.append(piece)
                current_len += piece_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
