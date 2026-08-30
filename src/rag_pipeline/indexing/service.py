"""Synchronization/orchestration: build one dense + sparse snapshot from one chunk corpus.

    chunks
    -> validation (models.canonical_order)
    -> deterministic ordering
    -> snapshot fingerprint (fingerprint.compute_snapshot_id)
    -> [idempotence check: reuse a valid existing snapshot if it matches]
    -> generate embeddings
    -> build new Chroma snapshot collection (dense.py)
    -> build/persist new BM25 corpus snapshot (sparse.py)
    -> verify synchronization
    -> atomically activate manifest (manifest.py)

If either index build fails, the previous active manifest (if any) is left
untouched, and newly-created (now-orphaned) snapshot artifacts are removed
on a best-effort basis. This is a snapshot/activation model, not a
database transaction: Chroma and the filesystem cannot be committed
together atomically, so the manifest write is the single point that
"activates" a snapshot, and it only happens after everything else has been
built and cross-checked.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from chromadb.errors import ChromaError

from ..chunking.models import Chunk
from ..config import Settings
from ..embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from .dense import (
    build_collection_name,
    build_dense_index,
    get_chroma_client,
    get_dense_collection_ids,
    verify_dense_collection,
)
from .exceptions import DenseIndexError, IndexingError, SparseIndexError, SynchronizationError
from .fingerprint import compute_snapshot_id
from .manifest import load_manifest, write_manifest
from .models import MANIFEST_SCHEMA_VERSION, IndexManifest, canonical_order
from .sparse import load_sparse_snapshot, sparse_snapshot_dir, write_sparse_snapshot
from .tokenizer import TOKENIZER_VERSION

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexingResult:
    manifest: IndexManifest
    reused_existing: bool


def verify_synchronization(
    canonical_chunk_ids: Sequence[str],
    dense_chunk_ids: Sequence[str],
    sparse_chunk_ids: Sequence[str],
) -> None:
    """Raise `SynchronizationError` unless dense, sparse, and canonical ID sets agree exactly.

    Also requires the sparse corpus's *order* to match the canonical order,
    since BM25's corpus-position -> chunk_id mapping depends on it.
    """
    canonical_set = set(canonical_chunk_ids)
    dense_set = set(dense_chunk_ids)
    sparse_set = set(sparse_chunk_ids)

    if dense_set != canonical_set:
        missing = canonical_set - dense_set
        extra = dense_set - canonical_set
        raise SynchronizationError(
            f"Dense index ID set does not match the canonical chunk corpus "
            f"(missing={len(missing)}, extra={len(extra)})."
        )

    if sparse_set != canonical_set:
        missing = canonical_set - sparse_set
        extra = sparse_set - canonical_set
        raise SynchronizationError(
            f"Sparse index ID set does not match the canonical chunk corpus "
            f"(missing={len(missing)}, extra={len(extra)})."
        )

    if tuple(sparse_chunk_ids) != tuple(canonical_chunk_ids):
        raise SynchronizationError(
            "Sparse snapshot chunk order does not match the canonical chunk order."
        )


def _existing_snapshot_is_valid(
    settings: Settings, manifest: IndexManifest, ordered_chunks: Sequence[Chunk]
) -> bool:
    """Best-effort check that a previously-built snapshot still matches `ordered_chunks`.

    `ordered_chunks` is the freshly-computed canonical corpus for *this*
    call (already known to share `manifest`'s snapshot_id, and therefore
    its chunk_ids, since the fingerprint is a SHA-256 hash of them). Beyond
    ID/count agreement, this also checks that the stored *content* wasn't
    corrupted out from under an unchanged snapshot_id:

    - dense: record count, ID set, cosine vector-space config, and stored
      document text for every ID matches the expected chunk text exactly;
    - sparse: chunk_id order matches, tokenizer_version matches the
      current tokenizer, and persisted texts match the expected chunk text
      exactly.

    Deliberately NOT re-verified: the actual embedding *vector values*
    stored in Chroma. Recomputing or fetching+comparing every vector would
    mean re-embedding (or at least re-fetching in full) the whole corpus,
    which defeats the point of reuse; ID/text/config integrity is treated
    as a sufficient, much cheaper signal that a snapshot is still valid.
    """
    expected_text_by_id = {chunk.chunk_id: chunk.text for chunk in ordered_chunks}
    try:
        client = get_chroma_client(settings)
        collection = client.get_collection(name=manifest.chroma_collection_name)
        verify_dense_collection(collection, manifest.chunk_ids)

        if collection.configuration_json.get("hnsw", {}).get("space") != "cosine":
            return False

        stored = collection.get(ids=list(manifest.chunk_ids), include=["documents"])
        stored_documents = stored["documents"]
        if stored_documents is None:
            return False
        for chunk_id, document in zip(stored["ids"], stored_documents, strict=True):
            if document != expected_text_by_id.get(chunk_id):
                return False

        sparse_snapshot = load_sparse_snapshot(settings, manifest.snapshot_id)
        if tuple(sparse_snapshot.chunk_ids) != manifest.chunk_ids:
            return False
        for chunk_id, text in zip(sparse_snapshot.chunk_ids, sparse_snapshot.texts, strict=True):
            if text != expected_text_by_id.get(chunk_id):
                return False
    except (DenseIndexError, SparseIndexError, ChromaError):
        return False
    return True


def _cleanup_failed_snapshot(settings: Settings, collection_name: str, snapshot_id: str) -> None:
    """Best-effort removal of partially-built artifacts after a failed index build.

    Never raises: a cleanup failure must not mask the original error, and
    the previously-active manifest (untouched by this whole attempt) is
    what keeps the system usable regardless of whether cleanup succeeds.
    Each cleanup step is independently isolated so a failure in one (e.g.
    obtaining the Chroma client) can't prevent the other (filesystem
    cleanup) from still being attempted. Only a warning is logged — never
    the failed exception's context beyond its own message, and never any
    document text or secrets, since none of that is ever in scope here.
    """
    try:
        client = get_chroma_client(settings)
        client.delete_collection(name=collection_name)
    except Exception as exc:
        logger.warning("cleanup: failed to delete Chroma collection %s: %s", collection_name, exc)

    try:
        snapshot_dir = sparse_snapshot_dir(settings, snapshot_id)
        shutil.rmtree(snapshot_dir, ignore_errors=True)
    except Exception as exc:
        logger.warning("cleanup: failed to remove sparse snapshot dir for %s: %s", snapshot_id, exc)


def index_chunks(
    chunks: Sequence[Chunk],
    settings: Settings,
    embedding_provider: EmbeddingProvider | None = None,
) -> IndexingResult:
    """Build (or reuse) one synchronized dense+sparse snapshot for `chunks`.

    Idempotence: if the exact same corpus/strategy/embedding model/
    tokenizer version was already indexed and its active manifest's
    snapshot still validates, that snapshot is reused rather than rebuilt
    — no embedding provider is even constructed in that case, so an
    already-indexed corpus can be "reindexed" without an API key. A new
    snapshot is built (and the old one left in place until the new one is
    fully validated) whenever reuse isn't safely verifiable.
    """
    ordered = canonical_order(chunks)
    strategy = ordered[0].chunking_strategy
    chunk_ids = tuple(chunk.chunk_id for chunk in ordered)

    snapshot_id = compute_snapshot_id(
        chunk_ids, strategy, settings.embedding_model, TOKENIZER_VERSION
    )

    existing_manifest = load_manifest(settings, strategy)
    if existing_manifest is not None and existing_manifest.snapshot_id == snapshot_id:
        if _existing_snapshot_is_valid(settings, existing_manifest, ordered):
            logger.info(
                "indexing: reusing valid existing snapshot_id=%s strategy=%s", snapshot_id, strategy
            )
            return IndexingResult(manifest=existing_manifest, reused_existing=True)
        logger.warning(
            "indexing: existing manifest for strategy=%s matched snapshot_id=%s but failed "
            "validation; rebuilding",
            strategy,
            snapshot_id,
        )

    if embedding_provider is None:
        embedding_provider = OpenAIEmbeddingProvider(settings)

    collection_name = build_collection_name(
        strategy, snapshot_id, settings.chroma_collection_prefix
    )

    logger.info(
        "indexing started strategy=%s snapshot_id=%s chunk_count=%d",
        strategy,
        snapshot_id,
        len(ordered),
    )

    try:
        embeddings = embedding_provider.embed([chunk.text for chunk in ordered])
        if len(embeddings) != len(ordered):
            raise DenseIndexError(
                f"Embedding provider returned {len(embeddings)} vectors for {len(ordered)} chunks."
            )
        embedding_dimension = len(embeddings[0]) if embeddings else 0

        client = get_chroma_client(settings)
        collection = build_dense_index(
            client, collection_name, ordered, embeddings, settings.index_batch_size
        )
        verify_dense_collection(collection, chunk_ids)
        dense_ids = get_dense_collection_ids(collection)

        sparse_path = write_sparse_snapshot(settings, ordered, snapshot_id, TOKENIZER_VERSION)
        # Re-read what was actually persisted, rather than trusting the
        # in-memory `ordered` list, so a write/read corruption is caught
        # here rather than silently activating a broken manifest.
        persisted_sparse_snapshot = load_sparse_snapshot(settings, snapshot_id)

        verify_synchronization(chunk_ids, dense_ids, persisted_sparse_snapshot.chunk_ids)

        # Manifest activation is part of the protected build: if writing it
        # fails (disk full, permissions, ...), the dense+sparse artifacts
        # just created are orphaned and must be cleaned up exactly like any
        # other build failure -- the previously-active manifest (if any)
        # must never be replaced by a snapshot that failed to activate.
        manifest = IndexManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            snapshot_id=snapshot_id,
            chunking_strategy=strategy,
            embedding_model=settings.embedding_model,
            embedding_dimension=embedding_dimension,
            bm25_tokenizer_version=TOKENIZER_VERSION,
            chunk_count=len(ordered),
            chunk_ids=chunk_ids,
            chroma_collection_name=collection_name,
            sparse_snapshot_path=str(sparse_path),
            created_at=datetime.now(UTC).isoformat(),
        )
        write_manifest(settings, manifest)
    except Exception as exc:
        _cleanup_failed_snapshot(settings, collection_name, snapshot_id)
        raise IndexingError(
            f"Failed to build synchronized index snapshot for strategy={strategy!r}: {exc}"
        ) from exc

    logger.info(
        "indexing completed strategy=%s snapshot_id=%s chunk_count=%d",
        strategy,
        snapshot_id,
        len(ordered),
    )
    return IndexingResult(manifest=manifest, reused_existing=False)
