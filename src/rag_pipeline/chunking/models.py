"""Canonical, strategy-independent representation of a retrieval chunk.

A `Chunk` is produced from an ingestion `Segment` by exactly one chunking
strategy. It carries the provenance (document id, source file, section
heading, page number, originating strategy) needed by future indexing,
retrieval, citation, and evaluation steps, without those steps having to
re-derive it from the source document.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from ..config import ChunkingStrategy


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single retrieval-oriented chunk of text, tied back to its source document."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    source_file: str
    section_heading: str | None
    page_number: int | None
    chunking_strategy: ChunkingStrategy
    character_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "source_file": self.source_file,
            "section_heading": self.section_heading,
            "page_number": self.page_number,
            "chunking_strategy": self.chunking_strategy.value,
            "character_count": self.character_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            chunk_index=data["chunk_index"],
            text=data["text"],
            source_file=data["source_file"],
            section_heading=data.get("section_heading"),
            page_number=data.get("page_number"),
            chunking_strategy=ChunkingStrategy(data["chunking_strategy"]),
            character_count=data["character_count"],
        )


def compute_chunk_id(
    document_id: str, strategy: ChunkingStrategy, chunk_index: int, text: str
) -> str:
    """Deterministic chunk identity: SHA-256 over (document, strategy, index, text).

    The same document processed with the same strategy always produces the
    same ordered chunk IDs. Including `strategy` in the payload means two
    different strategies never collide merely because a chunk index matches.
    """
    payload = f"{document_id}|{strategy.value}|{chunk_index}|{text}".encode()
    return hashlib.sha256(payload).hexdigest()


def build_chunk(
    *,
    document_id: str,
    chunk_index: int,
    text: str,
    source_file: str,
    section_heading: str | None,
    page_number: int | None,
    strategy: ChunkingStrategy,
) -> Chunk:
    """Construct a `Chunk`, guaranteeing `chunk_id`/`character_count` stay consistent."""
    return Chunk(
        chunk_id=compute_chunk_id(document_id, strategy, chunk_index, text),
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        source_file=source_file,
        section_heading=section_heading,
        page_number=page_number,
        chunking_strategy=strategy,
        character_count=len(text),
    )
