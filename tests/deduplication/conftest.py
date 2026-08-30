"""Shared fixtures/helpers for deduplication tests."""

from __future__ import annotations

from rag_pipeline.chunking.models import Chunk, build_chunk
from rag_pipeline.config import ChunkingStrategy


def make_chunk(
    *,
    document_id: str = "d" * 64,
    chunk_index: int = 0,
    text: str = "content",
    source_file: str = "doc.md",
) -> Chunk:
    return build_chunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        source_file=source_file,
        section_heading=None,
        page_number=None,
        strategy=ChunkingStrategy.RECURSIVE,
    )
