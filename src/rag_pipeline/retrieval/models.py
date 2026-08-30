"""Typed, immutable models for dense retrieval results."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ChunkingStrategy


@dataclass(frozen=True, slots=True)
class DenseRetrievalResult:
    """One ranked dense-retrieval hit, with full provenance back to its source chunk.

    `rank` starts at 1 and reflects Chroma's own nearest-neighbor ordering
    (never independently re-sorted). `distance` is the raw cosine distance
    Chroma returned; `similarity` is the derived, human-friendly
    `1.0 - distance` -- both are kept, since smaller distance and larger
    similarity mean the same thing ("better match") but are not
    interchangeable values.
    """

    chunk_id: str
    rank: int
    text: str
    distance: float
    similarity: float
    document_id: str
    chunk_index: int
    source_file: str
    section_heading: str | None
    page_number: int | None
    chunking_strategy: ChunkingStrategy
