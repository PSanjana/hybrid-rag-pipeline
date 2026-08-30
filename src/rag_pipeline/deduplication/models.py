"""Typed, immutable models for near-duplicate chunk detection results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..chunking.models import Chunk

DEDUP_ALGORITHM_VERSION = "cosine_v1"


class DuplicateType(StrEnum):
    """Why a chunk was skipped: identical final text, or cosine similarity above threshold."""

    EXACT = "exact"
    NEAR = "near"


@dataclass(frozen=True, slots=True)
class DuplicateRecord:
    """One skipped chunk, and the canonical (kept) chunk it duplicates.

    `similarity` is always `1.0` for an `EXACT` record -- identical final
    text implies canonical similarity by definition, so it is never
    computed via cosine comparison for that case (see
    `rag_pipeline.deduplication.detector`).
    """

    skipped_chunk_id: str
    canonical_chunk_id: str
    duplicate_type: DuplicateType
    similarity: float
    skipped_document_id: str
    canonical_document_id: str
    skipped_chunk_index: int
    canonical_chunk_index: int
    skipped_source_file: str
    canonical_source_file: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "skipped_chunk_id": self.skipped_chunk_id,
            "canonical_chunk_id": self.canonical_chunk_id,
            "duplicate_type": self.duplicate_type.value,
            "similarity": self.similarity,
            "skipped_document_id": self.skipped_document_id,
            "canonical_document_id": self.canonical_document_id,
            "skipped_chunk_index": self.skipped_chunk_index,
            "canonical_chunk_index": self.canonical_chunk_index,
            "skipped_source_file": self.skipped_source_file,
            "canonical_source_file": self.canonical_source_file,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DuplicateRecord:
        return cls(
            skipped_chunk_id=data["skipped_chunk_id"],
            canonical_chunk_id=data["canonical_chunk_id"],
            duplicate_type=DuplicateType(data["duplicate_type"]),
            similarity=data["similarity"],
            skipped_document_id=data["skipped_document_id"],
            canonical_document_id=data["canonical_document_id"],
            skipped_chunk_index=data["skipped_chunk_index"],
            canonical_chunk_index=data["canonical_chunk_index"],
            skipped_source_file=data["skipped_source_file"],
            canonical_source_file=data["canonical_source_file"],
        )


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    """Output of one deduplication run: the surviving corpus, plus a full audit trail.

    `kept_embeddings` stays aligned index-for-index with `kept_chunks` --
    the same invariant `index_chunks()` needs to build the dense index from
    precomputed vectors.
    """

    kept_chunks: tuple[Chunk, ...]
    kept_embeddings: tuple[tuple[float, ...], ...]
    duplicates: tuple[DuplicateRecord, ...]
    algorithm_version: str
    similarity_threshold: float
