"""Chroma-backed dense vector index.

Uses a local persistent Chroma client with PRECOMPUTED embeddings from our
own `EmbeddingProvider` — Chroma is never asked to call OpenAI or generate
embeddings itself (`embedding_function=None` on every collection).
"""

from __future__ import annotations

from collections.abc import Sequence

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import ChromaError, NotFoundError

from ..chunking.models import Chunk
from ..config import ChunkingStrategy, Settings
from .exceptions import DenseIndexError

_DEFAULT_FINGERPRINT_LENGTH = 12


def get_chroma_client(settings: Settings) -> ClientAPI:
    """A local persistent Chroma client rooted at `settings.chroma_dir`."""
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def build_collection_name(
    strategy: ChunkingStrategy,
    snapshot_id: str,
    prefix: str = "rag",
    fingerprint_length: int = _DEFAULT_FINGERPRINT_LENGTH,
) -> str:
    """`<prefix>-<strategy>-<fingerprint-prefix>`, e.g. `rag-recursive-a1b2c3d4e5f6`.

    A fresh, snapshot-specific collection per index build — never a mutable
    collection shared across different corpora — so old and new snapshots
    for the same strategy can coexist during a rebuild, and different
    strategies never collide.
    """
    return f"{prefix}-{strategy.value}-{snapshot_id[:fingerprint_length]}"


def _chunk_metadata(chunk: Chunk) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "source_file": chunk.source_file,
        "chunking_strategy": chunk.chunking_strategy.value,
        "character_count": chunk.character_count,
    }
    # Chroma's metadata values must be non-None scalars, so optional fields
    # are omitted entirely rather than stored as null when absent.
    if chunk.section_heading is not None:
        metadata["section_heading"] = chunk.section_heading
    if chunk.page_number is not None:
        metadata["page_number"] = chunk.page_number
    return metadata


def build_dense_index(
    client: ClientAPI,
    collection_name: str,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
    batch_size: int,
) -> Collection:
    """Create a fresh `collection_name` (cosine space) and write one record per chunk.

    `chunks` and `embeddings` must be the same canonical order used for the
    BM25 sparse snapshot.

    Any pre-existing collection under `collection_name` is deleted first.
    This matters for rebuilds: `collection_name` is derived from the
    snapshot fingerprint, so a rebuild only ever targets the same name when
    the corpus itself is unchanged (the fingerprint didn't move) but the
    *previously stored* collection was judged invalid (e.g. reuse
    validation found corrupted content) -- `.add()` silently no-ops for an
    ID that already exists, so writing into a stale collection by ID would
    leave corrupted records exactly as corrupted. Deleting first guarantees
    every rebuild ends up in a genuinely clean, fully-correct state.
    """
    if len(chunks) != len(embeddings):
        raise DenseIndexError(
            f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})."
        )

    try:
        client.delete_collection(name=collection_name)
    except NotFoundError:
        pass  # No prior collection under this name -- the common, first-build case.
    except ChromaError as exc:
        raise DenseIndexError(
            f"Failed to clear pre-existing Chroma collection {collection_name!r}: {exc}"
        ) from exc

    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )
    except ChromaError as exc:
        raise DenseIndexError(
            f"Failed to create Chroma collection {collection_name!r}: {exc}"
        ) from exc

    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start : start + batch_size]
        batch_embeddings = embeddings[start : start + batch_size]
        # Explicitly typed as list[Sequence[float]] (not inferred) so mypy
        # checks each element against Sequence[float] rather than against
        # an invariant list[float], which Collection.add()'s signature
        # would otherwise reject.
        embeddings_arg: list[Sequence[float]] = [list(vector) for vector in batch_embeddings]
        try:
            collection.add(
                ids=[chunk.chunk_id for chunk in batch_chunks],
                embeddings=embeddings_arg,
                documents=[chunk.text for chunk in batch_chunks],
                metadatas=[_chunk_metadata(chunk) for chunk in batch_chunks],
            )
        except ChromaError as exc:
            raise DenseIndexError(
                f"Failed to write chunk batch to Chroma collection {collection_name!r}: {exc}"
            ) from exc

    return collection


def get_dense_collection_ids(collection: Collection) -> list[str]:
    """The full set of record IDs currently stored in `collection`."""
    try:
        result = collection.get(include=[])
    except ChromaError as exc:
        raise DenseIndexError(
            f"Failed to read Chroma collection {collection.name!r}: {exc}"
        ) from exc
    return list(result["ids"])


def verify_dense_collection(collection: Collection, expected_chunk_ids: Sequence[str]) -> None:
    """Raise `DenseIndexError` unless `collection` holds exactly `expected_chunk_ids`."""
    try:
        count = collection.count()
    except ChromaError as exc:
        raise DenseIndexError(
            f"Failed to count Chroma collection {collection.name!r}: {exc}"
        ) from exc

    if count != len(expected_chunk_ids):
        raise DenseIndexError(
            f"Chroma collection {collection.name!r} has {count} records, expected "
            f"{len(expected_chunk_ids)}."
        )

    stored_ids = set(get_dense_collection_ids(collection))
    if stored_ids != set(expected_chunk_ids):
        missing = set(expected_chunk_ids) - stored_ids
        extra = stored_ids - set(expected_chunk_ids)
        raise DenseIndexError(
            f"Chroma collection {collection.name!r} stored IDs do not match the canonical "
            f"chunk ID set (missing={len(missing)}, extra={len(extra)})."
        )
