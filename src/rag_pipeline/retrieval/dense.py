"""Dense (embedding-based) retrieval over the active Chroma snapshot for one chunking strategy.

    question
    -> query validation (non-empty)
    -> resolve top_k (explicit override or settings.dense_top_k)
    -> load the active manifest for the requested strategy
    -> verify embedding-model compatibility (settings.embedding_model == manifest.embedding_model)
    -> embed the question exactly once; validate the resulting vector
    -> open the manifest's exact Chroma collection (never guessed, never scanned)
    -> query it for n_results = min(top_k, manifest.chunk_count)
    -> parse/validate the raw Chroma response into typed, ranked results

Read-only end to end: only `Collection.query()` is ever called against
Chroma, and the manifest/sparse-snapshot/dedup-report on disk are never
written. Sparse (BM25) retrieval, hybrid fusion, reranking, and generation
are later pipeline stages and are not implemented here.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

from chromadb.api.types import QueryResult
from chromadb.errors import ChromaError

from ..config import ChunkingStrategy, Settings
from ..embeddings import EmbeddingProvider, EmbeddingProviderError, OpenAIEmbeddingProvider
from ..embeddings.validation import validate_vector
from ..indexing.dense import get_chroma_client
from ..indexing.models import IndexManifest
from ._shared import (
    load_active_manifest,
    optional_int,
    optional_str,
    parse_chunking_strategy,
    require_int,
    require_str,
    resolve_top_k,
    validate_query,
)
from .exceptions import DenseRetrievalError, EmbeddingModelMismatchError
from .models import DenseRetrievalResult

logger = logging.getLogger(__name__)


def _check_embedding_model_compatibility(settings: Settings, manifest: IndexManifest) -> None:
    if settings.embedding_model != manifest.embedding_model:
        raise EmbeddingModelMismatchError(
            f"Configured embedding_model={settings.embedding_model!r} does not match the "
            f"active index's embedding_model={manifest.embedding_model!r} for strategy="
            f"{manifest.chunking_strategy.value!r}."
        )


def _embed_query(
    embedding_provider: EmbeddingProvider, query: str, expected_dimension: int
) -> list[float]:
    try:
        vectors = embedding_provider.embed([query])
    except EmbeddingProviderError as exc:
        raise DenseRetrievalError(f"Failed to embed the query: {exc}") from exc
    if len(vectors) != 1:
        raise DenseRetrievalError(
            f"Embedding provider returned {len(vectors)} vectors for a single query; "
            "expected exactly 1."
        )
    vector = vectors[0]
    try:
        validate_vector(vector)
    except EmbeddingProviderError as exc:
        raise DenseRetrievalError(f"Invalid query embedding: {exc}") from exc
    if len(vector) != expected_dimension:
        raise DenseRetrievalError(
            f"Query embedding has dimension {len(vector)}, but the active index has "
            f"dimension {expected_dimension}."
        )
    return vector


def _parse_query_response(
    raw: QueryResult, requested_strategy: ChunkingStrategy
) -> list[DenseRetrievalResult]:
    """Turn one raw Chroma `query()` response into ranked, typed results.

    Never trusts the nested response structure: batch count, per-field
    presence, array-length agreement, and required metadata fields are all
    checked explicitly, raising `DenseRetrievalError` rather than
    constructing a partially-corrupted result.
    """
    documents_batches = raw.get("documents")
    metadatas_batches = raw.get("metadatas")
    distances_batches = raw.get("distances")
    ids_batches = raw.get("ids")

    if ids_batches is None or documents_batches is None:
        raise DenseRetrievalError("Chroma query response is missing required result fields.")
    if metadatas_batches is None or distances_batches is None:
        raise DenseRetrievalError("Chroma query response is missing required result fields.")

    # Each field is independently batched by Chroma (one batch per query in
    # query_embeddings); a single query must get exactly one batch in
    # *every* field, not just `ids` -- indexing straight into `[0]` without
    # checking documents/metadatas/distances too would either raise a raw
    # IndexError on zero batches or silently drop extra ones.
    for field_name, batches in (
        ("ids", ids_batches),
        ("documents", documents_batches),
        ("metadatas", metadatas_batches),
        ("distances", distances_batches),
    ):
        if len(batches) != 1:
            raise DenseRetrievalError(
                f"Expected exactly one query result batch for {field_name!r}, got {len(batches)}."
            )

    ids = ids_batches[0]
    documents = documents_batches[0]
    metadatas = metadatas_batches[0]
    distances = distances_batches[0]

    if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
        raise DenseRetrievalError(
            "Chroma query response arrays have mismatched lengths: "
            f"ids={len(ids)}, documents={len(documents)}, metadatas={len(metadatas)}, "
            f"distances={len(distances)}."
        )

    results: list[DenseRetrievalResult] = []
    for rank, (chunk_id, text, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances, strict=True), start=1
    ):
        if not isinstance(chunk_id, str) or not chunk_id:
            raise DenseRetrievalError(
                f"Chroma query response contains an invalid id: {chunk_id!r}."
            )
        if not isinstance(text, str):
            raise DenseRetrievalError(
                f"Chroma query response has no document text for id {chunk_id!r}."
            )
        if metadata is None:
            raise DenseRetrievalError(f"Chroma query response has no metadata for id {chunk_id!r}.")
        if not isinstance(distance, int | float) or isinstance(distance, bool):
            raise DenseRetrievalError(
                f"Chroma query response has a non-numeric distance for id {chunk_id!r}: "
                f"{distance!r}."
            )
        if not math.isfinite(float(distance)):
            raise DenseRetrievalError(
                f"Chroma query response has a non-finite distance for id {chunk_id!r}: "
                f"{distance!r}."
            )

        document_id = require_str(metadata, "document_id", chunk_id, DenseRetrievalError)
        chunk_index = require_int(metadata, "chunk_index", chunk_id, DenseRetrievalError)
        source_file = require_str(metadata, "source_file", chunk_id, DenseRetrievalError)
        chunking_strategy = parse_chunking_strategy(
            metadata, chunk_id, requested_strategy, DenseRetrievalError
        )

        distance_value = float(distance)
        results.append(
            DenseRetrievalResult(
                chunk_id=chunk_id,
                rank=rank,
                text=text,
                distance=distance_value,
                similarity=1.0 - distance_value,
                document_id=document_id,
                chunk_index=chunk_index,
                source_file=source_file,
                section_heading=optional_str(
                    metadata, "section_heading", chunk_id, DenseRetrievalError
                ),
                page_number=optional_int(metadata, "page_number", chunk_id, DenseRetrievalError),
                chunking_strategy=chunking_strategy,
            )
        )
    return results


def retrieve_dense(
    query: str,
    strategy: ChunkingStrategy,
    settings: Settings,
    embedding_provider: EmbeddingProvider | None = None,
    top_k: int | None = None,
) -> list[DenseRetrievalResult]:
    """Return the top-k dense nearest-neighbor chunks for `query` under `strategy`'s active index.

    The active Chroma collection is resolved solely from the strategy's
    active manifest (`chroma_collection_name`) -- never guessed, never
    scanned across all collections. Raises `IndexNotReadyError` if no
    manifest is active for `strategy`, `EmbeddingModelMismatchError` if the
    configured embedding model doesn't match the one the index was built
    with, and `DenseRetrievalError` for any embedding- or Chroma-response
    problem that can't be trusted.
    """
    validate_query(query)
    resolved_top_k = resolve_top_k(top_k, settings.dense_top_k)
    manifest = load_active_manifest(settings, strategy)
    _check_embedding_model_compatibility(settings, manifest)

    if embedding_provider is None:
        embedding_provider = OpenAIEmbeddingProvider(settings)

    query_vector = _embed_query(embedding_provider, query, manifest.embedding_dimension)

    try:
        client = get_chroma_client(settings)
        collection = client.get_collection(name=manifest.chroma_collection_name)
    except ChromaError as exc:
        raise DenseRetrievalError(
            f"Failed to open Chroma collection {manifest.chroma_collection_name!r} for "
            f"strategy={strategy.value!r}: {exc}"
        ) from exc

    n_results = min(resolved_top_k, manifest.chunk_count)

    # Explicitly typed as list[Sequence[float]] (not inferred) so mypy
    # checks the element against Sequence[float] rather than against an
    # invariant list[float], which Collection.query()'s signature would
    # otherwise reject (see the identical pattern in indexing/dense.py).
    query_embeddings_arg: list[Sequence[float]] = [query_vector]
    try:
        raw = collection.query(
            query_embeddings=query_embeddings_arg,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except ChromaError as exc:
        raise DenseRetrievalError(f"Chroma query failed: {exc}") from exc

    results = _parse_query_response(raw, strategy)

    logger.info(
        "dense retrieval: strategy=%s snapshot_id=%s top_k=%d returned=%d embedding_model=%s "
        "query_length=%d",
        strategy.value,
        manifest.snapshot_id,
        resolved_top_k,
        len(results),
        manifest.embedding_model,
        len(query),
    )
    return results
