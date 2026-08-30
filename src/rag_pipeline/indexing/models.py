"""Canonical chunk ordering/validation, and the schema-versioned index manifest model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..chunking.models import Chunk
from ..config import ChunkingStrategy
from .exceptions import InvalidChunkCorpusError, ManifestError

MANIFEST_SCHEMA_VERSION = 1


def canonical_order(chunks: Sequence[Chunk]) -> tuple[Chunk, ...]:
    """Validate `chunks` and return them in one deterministic canonical order.

    Order is `(document_id, chunk_index, chunk_id)` — stable regardless of
    incidental filesystem/iteration order the caller assembled `chunks` in.
    The `chunk_id` tie-breaker only matters for ordering two chunks from
    different documents that happen to share a `chunk_index`; within one
    document, `(document_id, chunk_index)` is required to be unique (see
    below), so there's never actually a tie to break there.

    Never mutates `chunks`; always returns a new tuple.

    Raises `InvalidChunkCorpusError` for an empty corpus, a duplicate
    `chunk_id`, a duplicate `(document_id, chunk_index)` position (even
    with distinct chunk_ids -- two chunks can't both legitimately claim
    the same position within one document), empty/whitespace-only chunk
    text, or chunks spanning more than one chunking strategy.
    """
    if not chunks:
        raise InvalidChunkCorpusError("Cannot index an empty chunk corpus.")

    seen_ids: set[str] = set()
    seen_positions: set[tuple[str, int]] = set()
    strategies: set[ChunkingStrategy] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen_ids:
            raise InvalidChunkCorpusError(f"Duplicate chunk_id: {chunk.chunk_id!r}.")
        seen_ids.add(chunk.chunk_id)

        position = (chunk.document_id, chunk.chunk_index)
        if position in seen_positions:
            raise InvalidChunkCorpusError(
                f"Duplicate (document_id, chunk_index) position: {position!r}."
            )
        seen_positions.add(position)

        if not chunk.text.strip():
            raise InvalidChunkCorpusError(
                f"Chunk {chunk.chunk_id!r} has empty/whitespace-only text."
            )
        strategies.add(chunk.chunking_strategy)

    if len(strategies) > 1:
        raise InvalidChunkCorpusError(
            "All chunks must belong to one chunking strategy; found: "
            f"{sorted(strategy.value for strategy in strategies)}."
        )

    return tuple(
        sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.chunk_index, chunk.chunk_id))
    )


@dataclass(frozen=True, slots=True)
class IndexManifest:
    """The authoritative record of which dense+sparse snapshot is active for one strategy.

    `created_at` is operational metadata only — it deliberately does not
    participate in `snapshot_id` (see `fingerprint.compute_snapshot_id`).
    """

    schema_version: int
    snapshot_id: str
    chunking_strategy: ChunkingStrategy
    embedding_model: str
    embedding_dimension: int
    bm25_tokenizer_version: str
    chunk_count: int
    chunk_ids: tuple[str, ...]
    chroma_collection_name: str
    sparse_snapshot_path: str
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "chunking_strategy": self.chunking_strategy.value,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "bm25_tokenizer_version": self.bm25_tokenizer_version,
            "chunk_count": self.chunk_count,
            "chunk_ids": list(self.chunk_ids),
            "chroma_collection_name": self.chroma_collection_name,
            "sparse_snapshot_path": self.sparse_snapshot_path,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexManifest:
        version = data.get("schema_version")
        if version != MANIFEST_SCHEMA_VERSION:
            raise ManifestError(
                f"Unsupported manifest schema_version={version!r}; expected "
                f"{MANIFEST_SCHEMA_VERSION}."
            )
        return cls(
            schema_version=version,
            snapshot_id=data["snapshot_id"],
            chunking_strategy=ChunkingStrategy(data["chunking_strategy"]),
            embedding_model=data["embedding_model"],
            embedding_dimension=data["embedding_dimension"],
            bm25_tokenizer_version=data["bm25_tokenizer_version"],
            chunk_count=data["chunk_count"],
            chunk_ids=tuple(data["chunk_ids"]),
            chroma_collection_name=data["chroma_collection_name"],
            sparse_snapshot_path=data["sparse_snapshot_path"],
            created_at=data.get("created_at"),
        )
