"""Synchronization/orchestration: build one dense + sparse snapshot from one chunk corpus.

    chunks
    -> validation (models.canonical_order)
    -> deterministic ordering
    -> pre-dedup request fingerprint (fingerprint.compute_snapshot_id over RAW chunk_ids)
    -> [idempotence check: reuse a valid existing snapshot if the request
       fingerprint and a content-integrity check both pass -- no embedding
       call needed]
    -> generate embeddings ONCE, over the full raw canonical corpus
    -> exact + near-duplicate detection (deduplication.deduplicate_chunks)
    -> kept chunks + kept embeddings
    -> final snapshot_id (fingerprint.compute_snapshot_id over the KEPT chunk_ids)
    -> build new Chroma snapshot collection (dense.py), from kept chunks/embeddings
    -> build/persist new BM25 corpus snapshot (sparse.py), from kept chunks only
    -> verify synchronization (kept chunk_ids == dense ids == sparse ids)
    -> persist the duplicate report (dedup_report.py)
    -> atomically activate manifest (manifest.py)

Deduplication runs once, *before* either index is built -- never after --
so Chroma and BM25 are always built from the exact same post-deduplication
corpus; there is no window where one index could see a chunk the other
doesn't.

If deduplication, either index build, duplicate-report persistence, or
manifest activation fails, the previous active manifest (if any) is left
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
from ..deduplication import DEDUP_ALGORITHM_VERSION, deduplicate_chunks
from ..embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from .dedup_report import dedup_report_dir, load_dedup_report, write_dedup_report
from .dense import (
    build_collection_name,
    build_dense_index,
    get_chroma_client,
    get_dense_collection_ids,
    verify_dense_collection,
)
from .exceptions import (
    DedupReportError,
    DenseIndexError,
    IndexingError,
    SparseIndexError,
    SynchronizationError,
)
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

    `canonical_chunk_ids` here is the *post-dedup, kept* corpus -- the
    caller is responsible for passing kept IDs, not the raw pre-dedup ones.

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

    `ordered_chunks` is the freshly-computed, *pre-dedup* canonical corpus
    for *this* call -- a superset of `manifest.chunk_ids` (the kept,
    post-dedup corpus) whenever the raw request is genuinely unchanged
    (already established by the caller matching `manifest.request_fingerprint`
    before calling this). Text for each of `manifest.chunk_ids` is looked
    up from it, so this never needs to know in advance which chunks
    deduplication would keep -- and therefore never needs to re-run
    embeddings or deduplication just to validate reuse.

    Beyond ID/count agreement, this also checks that the stored *content*
    wasn't corrupted out from under an unchanged snapshot_id:

    - dense: record count, ID set, cosine vector-space config, and stored
      document text for every ID matches the expected chunk text exactly;
    - sparse: chunk_id order matches, tokenizer_version matches the
      current tokenizer, and persisted texts match the expected chunk text
      exactly;
    - dedup report: a duplicate report exists for this snapshot_id and
      records the same dedup algorithm version and threshold as the
      manifest (existence + config consistency, not a re-verification of
      which specific chunks were flagged as duplicates).

    Deliberately NOT re-verified: the actual embedding *vector values*
    stored in Chroma, and the specific per-chunk duplicate decisions
    recorded in the dedup report. Recomputing or fetching+comparing every
    vector would mean re-embedding (or at least re-fetching in full) the
    whole corpus, which defeats the point of reuse; re-deriving duplicate
    decisions would mean re-running deduplication, which needs those same
    embeddings. ID/text/config integrity is treated as a sufficient, much
    cheaper signal that a snapshot is still valid.
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

        dedup_report = load_dedup_report(settings, manifest.snapshot_id)
        if dedup_report.dedup_algorithm_version != manifest.dedup_algorithm_version:
            return False
        if dedup_report.dedup_similarity_threshold != manifest.dedup_similarity_threshold:
            return False
    except (DenseIndexError, SparseIndexError, DedupReportError, ChromaError):
        return False
    return True


def _cleanup_failed_snapshot(
    settings: Settings, collection_name: str | None, snapshot_id: str | None
) -> None:
    """Best-effort removal of partially-built artifacts after a failed index build.

    `collection_name`/`snapshot_id` are `None` when the failure happened
    before either was even computed (embedding generation or
    deduplication itself failing) -- nothing was created yet in that case,
    so the corresponding cleanup step is simply skipped rather than
    attempted against a name that was never used.

    Never raises: a cleanup failure must not mask the original error, and
    the previously-active manifest (untouched by this whole attempt) is
    what keeps the system usable regardless of whether cleanup succeeds.
    Each cleanup step is independently isolated so a failure in one (e.g.
    obtaining the Chroma client) can't prevent the others (sparse/dedup
    filesystem cleanup) from still being attempted. Only a warning is
    logged — never the failed exception's context beyond its own message,
    and never any document text or secrets, since none of that is ever in
    scope here.
    """
    if collection_name is not None:
        try:
            client = get_chroma_client(settings)
            client.delete_collection(name=collection_name)
        except Exception as exc:
            logger.warning(
                "cleanup: failed to delete Chroma collection %s: %s", collection_name, exc
            )

    if snapshot_id is not None:
        try:
            shutil.rmtree(sparse_snapshot_dir(settings, snapshot_id), ignore_errors=True)
        except Exception as exc:
            logger.warning(
                "cleanup: failed to remove sparse snapshot dir for %s: %s", snapshot_id, exc
            )

        try:
            shutil.rmtree(dedup_report_dir(settings, snapshot_id), ignore_errors=True)
        except Exception as exc:
            logger.warning(
                "cleanup: failed to remove dedup report dir for %s: %s", snapshot_id, exc
            )


def index_chunks(
    chunks: Sequence[Chunk],
    settings: Settings,
    embedding_provider: EmbeddingProvider | None = None,
) -> IndexingResult:
    """Build (or reuse) one synchronized dense+sparse snapshot for `chunks`.

    Deduplication (near-duplicate filtering, see `rag_pipeline.deduplication`)
    runs exactly once per build: embeddings are generated for the full raw
    canonical corpus, then `deduplicate_chunks()` decides what survives,
    and only the surviving ("kept") chunks/embeddings are ever written to
    Chroma or BM25.

    Idempotence without unnecessary re-embedding: reuse is gated by a
    *pre-dedup request fingerprint* -- a hash over the raw canonical
    chunk_ids, chunking strategy, embedding model, tokenizer version,
    dedup algorithm version, and dedup threshold (see
    `fingerprint.compute_snapshot_id`). If an active manifest's
    `request_fingerprint` matches and its snapshot still validates (see
    `_existing_snapshot_is_valid`), it's reused with no embedding provider
    call at all: deduplication is deterministic given identical inputs, so
    an unchanged request is assumed to produce the same kept corpus
    without re-running embeddings or deduplication just to confirm it. The
    *final* `snapshot_id` (recorded once a rebuild completes) is the same
    kind of fingerprint, computed over the post-dedup *kept* chunk_ids
    instead of the raw ones -- so a threshold or algorithm change always
    changes snapshot identity, even if by coincidence the same chunks
    would still survive.
    """
    ordered = canonical_order(chunks)
    strategy = ordered[0].chunking_strategy
    raw_chunk_ids = tuple(chunk.chunk_id for chunk in ordered)

    request_fingerprint = compute_snapshot_id(
        raw_chunk_ids,
        strategy,
        settings.embedding_model,
        TOKENIZER_VERSION,
        DEDUP_ALGORITHM_VERSION,
        settings.dedup_similarity_threshold,
    )

    existing_manifest = load_manifest(settings, strategy)
    if (
        existing_manifest is not None
        and existing_manifest.request_fingerprint == request_fingerprint
    ):
        if _existing_snapshot_is_valid(settings, existing_manifest, ordered):
            logger.info(
                "indexing: reusing valid existing snapshot_id=%s strategy=%s",
                existing_manifest.snapshot_id,
                strategy,
            )
            return IndexingResult(manifest=existing_manifest, reused_existing=True)
        logger.warning(
            "indexing: existing manifest for strategy=%s matched the request fingerprint but "
            "failed content validation; rebuilding",
            strategy,
        )

    if embedding_provider is None:
        embedding_provider = OpenAIEmbeddingProvider(settings)

    logger.info(
        "indexing started strategy=%s pre_dedup_chunk_count=%d",
        strategy,
        len(ordered),
    )

    collection_name: str | None = None
    snapshot_id: str | None = None
    try:
        embeddings = embedding_provider.embed([chunk.text for chunk in ordered])
        if len(embeddings) != len(ordered):
            raise DenseIndexError(
                f"Embedding provider returned {len(embeddings)} vectors for {len(ordered)} chunks."
            )
        embedding_dimension = len(embeddings[0]) if embeddings else 0

        dedup_result = deduplicate_chunks(ordered, embeddings, settings.dedup_similarity_threshold)
        kept_chunks = dedup_result.kept_chunks
        kept_embeddings = dedup_result.kept_embeddings
        kept_chunk_ids = tuple(chunk.chunk_id for chunk in kept_chunks)

        snapshot_id = compute_snapshot_id(
            kept_chunk_ids,
            strategy,
            settings.embedding_model,
            TOKENIZER_VERSION,
            DEDUP_ALGORITHM_VERSION,
            settings.dedup_similarity_threshold,
        )
        collection_name = build_collection_name(
            strategy, snapshot_id, settings.chroma_collection_prefix
        )

        client = get_chroma_client(settings)
        collection = build_dense_index(
            client, collection_name, kept_chunks, kept_embeddings, settings.index_batch_size
        )
        verify_dense_collection(collection, kept_chunk_ids)
        dense_ids = get_dense_collection_ids(collection)

        sparse_path = write_sparse_snapshot(settings, kept_chunks, snapshot_id, TOKENIZER_VERSION)
        # Re-read what was actually persisted, rather than trusting the
        # in-memory `kept_chunks` list, so a write/read corruption is
        # caught here rather than silently activating a broken manifest.
        persisted_sparse_snapshot = load_sparse_snapshot(settings, snapshot_id)

        verify_synchronization(kept_chunk_ids, dense_ids, persisted_sparse_snapshot.chunk_ids)

        written_dedup_report_path = write_dedup_report(settings, snapshot_id, dedup_result)

        # Manifest activation is part of the protected build: if writing it
        # fails (disk full, permissions, ...), the dense+sparse+dedup-report
        # artifacts just created are orphaned and must be cleaned up exactly
        # like any other build failure -- the previously-active manifest (if
        # any) must never be replaced by a snapshot that failed to activate.
        manifest = IndexManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            snapshot_id=snapshot_id,
            request_fingerprint=request_fingerprint,
            chunking_strategy=strategy,
            embedding_model=settings.embedding_model,
            embedding_dimension=embedding_dimension,
            bm25_tokenizer_version=TOKENIZER_VERSION,
            dedup_algorithm_version=DEDUP_ALGORITHM_VERSION,
            dedup_similarity_threshold=settings.dedup_similarity_threshold,
            pre_dedup_chunk_count=len(ordered),
            chunk_count=len(kept_chunks),
            duplicate_count=len(dedup_result.duplicates),
            chunk_ids=kept_chunk_ids,
            chroma_collection_name=collection_name,
            sparse_snapshot_path=str(sparse_path),
            dedup_report_path=str(written_dedup_report_path),
            created_at=datetime.now(UTC).isoformat(),
        )
        write_manifest(settings, manifest)
    except Exception as exc:
        _cleanup_failed_snapshot(settings, collection_name, snapshot_id)
        raise IndexingError(
            f"Failed to build synchronized index snapshot for strategy={strategy!r}: {exc}"
        ) from exc

    logger.info(
        "indexing completed strategy=%s snapshot_id=%s pre_dedup_chunk_count=%d "
        "kept_chunk_count=%d duplicate_count=%d",
        strategy,
        snapshot_id,
        manifest.pre_dedup_chunk_count,
        manifest.chunk_count,
        manifest.duplicate_count,
    )
    return IndexingResult(manifest=manifest, reused_existing=False)
