"""Sparse (BM25) retrieval over the active sparse snapshot for one chunking strategy.

    question
    -> query validation (non-empty, tokenizes to at least one token)
    -> resolve top_k (explicit override or settings.sparse_top_k)
    -> load the active manifest for the requested strategy
    -> verify tokenizer-version compatibility (manifest.bm25_tokenizer_version == TOKENIZER_VERSION)
    -> reconstruct BM25 from the persisted sparse snapshot (load_bm25_index -- never
       rebuilt from scratch here)
    -> score every corpus position with BM25Okapi.get_scores(), unnormalized
    -> rank deterministically: score DESC, ties broken by corpus position ASC
    -> hydrate the top-k chunk IDs' text/metadata from Chroma via Collection.get()
       (never Collection.query() -- no vector search happens in this module)
    -> cross-check hydrated text against the sparse snapshot's own text,
       reorder into BM25 rank order (Chroma's return order is not trusted),
       and validate metadata into typed, ranked results

Read-only end to end: only `Collection.get()` is ever called against
Chroma, and the manifest/sparse-snapshot/dedup-report/dense collection
contents on disk are never written. Reciprocal Rank Fusion, hybrid
dense+sparse merging, reranking, and generation are later pipeline stages
and are not implemented here.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence

from chromadb.api.types import GetResult, Metadata
from chromadb.errors import ChromaError

from ..config import ChunkingStrategy, Settings
from ..indexing.dense import get_chroma_client
from ..indexing.exceptions import SparseIndexError
from ..indexing.models import IndexManifest
from ..indexing.sparse import load_bm25_index, load_sparse_snapshot
from ..indexing.tokenizer import TOKENIZER_VERSION, tokenize
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
from .exceptions import InvalidQueryError, SparseRetrievalError, TokenizerVersionMismatchError
from .models import SparseRetrievalResult

logger = logging.getLogger(__name__)

_RankedCandidate = tuple[int, str, float]  # (corpus_position, chunk_id, bm25_score)


def _check_tokenizer_compatibility(manifest: IndexManifest) -> None:
    if manifest.bm25_tokenizer_version != TOKENIZER_VERSION:
        raise TokenizerVersionMismatchError(
            f"Active index for strategy={manifest.chunking_strategy.value!r} was built with "
            f"bm25_tokenizer_version={manifest.bm25_tokenizer_version!r}, but the runtime "
            f"tokenizer is {TOKENIZER_VERSION!r}."
        )


def _score_candidates(
    settings: Settings, manifest: IndexManifest, query_tokens: list[str]
) -> tuple[list[_RankedCandidate], dict[str, str]]:
    """Reconstruct BM25 from the persisted snapshot and score every corpus position.

    Returns the scored `(position, chunk_id, score)` candidates (unranked,
    in canonical corpus order) plus a `chunk_id -> text` map from the
    sparse snapshot, used later both for hydration lookups and the
    stored-text integrity check.
    """
    try:
        snapshot = load_sparse_snapshot(settings, manifest.snapshot_id)
        reconstructed = load_bm25_index(settings, manifest.snapshot_id)
    except SparseIndexError as exc:
        raise SparseRetrievalError(
            f"Failed to reconstruct the sparse BM25 index for strategy="
            f"{manifest.chunking_strategy.value!r}: {exc}"
        ) from exc

    # Exact, order-sensitive equality -- not merely a count comparison. A
    # snapshot/reconstruction with the right chunk *count* but a
    # substituted or reordered chunk_id would otherwise silently score and
    # return the wrong corpus under an apparently-valid manifest.
    if snapshot.chunk_ids != manifest.chunk_ids:
        raise SparseRetrievalError(
            f"Sparse snapshot chunk_ids do not match the active manifest's chunk_ids for "
            f"strategy={manifest.chunking_strategy.value!r}; the sparse snapshot may be "
            "corrupted or out of sync with the manifest."
        )
    if reconstructed.chunk_ids != manifest.chunk_ids:
        raise SparseRetrievalError(
            f"Reconstructed BM25 chunk_ids do not match the active manifest's chunk_ids for "
            f"strategy={manifest.chunking_strategy.value!r}; the sparse snapshot may be "
            "corrupted or out of sync with the manifest."
        )

    scores = reconstructed.bm25.get_scores(query_tokens)
    if len(scores) != len(reconstructed.chunk_ids):
        raise SparseRetrievalError(
            f"BM25 returned {len(scores)} scores for {len(reconstructed.chunk_ids)} corpus "
            "positions."
        )

    candidates: list[_RankedCandidate] = []
    for position, (chunk_id, score) in enumerate(zip(reconstructed.chunk_ids, scores, strict=True)):
        score_value = float(score)
        if not math.isfinite(score_value):
            raise SparseRetrievalError(
                f"BM25 produced a non-finite score for chunk_id={chunk_id!r}: {score_value!r}."
            )
        candidates.append((position, chunk_id, score_value))

    texts_by_id = dict(zip(snapshot.chunk_ids, snapshot.texts, strict=True))
    return candidates, texts_by_id


def _rank_candidates(candidates: Sequence[_RankedCandidate], top_k: int) -> list[_RankedCandidate]:
    """Sort by score DESC, ties broken by canonical corpus position ASC -- never random."""
    ranked = sorted(candidates, key=lambda candidate: (-candidate[2], candidate[0]))
    return ranked[:top_k]


def _hydrate_results(
    stored: GetResult,
    ranked_candidates: Sequence[_RankedCandidate],
    snapshot_texts_by_id: Mapping[str, str],
    requested_strategy: ChunkingStrategy,
) -> list[SparseRetrievalResult]:
    """Turn a Chroma `get()` response into ranked, typed results, in BM25 rank order.

    Chroma's return order is never assumed to match the requested ID
    order: an id -> (text, metadata) map is built first, then results are
    reconstructed by walking `ranked_candidates` (the BM25-determined
    order) -- Chroma's own ordering never influences the final ranking.

    Response integrity is checked in a fixed sequence, before that map is
    ever built: every returned id must be a non-empty string; no id may be
    returned more than once (a duplicate would otherwise silently
    overwrite an entry in the map below, masking a corrupted or
    substituted record); the returned record count must equal the number
    of requested *unique* ids (catches a same-set-but-duplicated response
    that pure set comparison alone would miss); and only then is the
    returned id set checked against what was requested.
    """
    ids = stored.get("ids")
    documents = stored.get("documents")
    metadatas = stored.get("metadatas")
    if ids is None or documents is None or metadatas is None:
        raise SparseRetrievalError("Chroma get() response is missing required result fields.")
    if not (len(ids) == len(documents) == len(metadatas)):
        raise SparseRetrievalError(
            "Chroma get() response arrays have mismatched lengths: "
            f"ids={len(ids)}, documents={len(documents)}, metadatas={len(metadatas)}."
        )

    for raw_id in ids:
        if not isinstance(raw_id, str) or not raw_id:
            raise SparseRetrievalError(f"Chroma get() response contains an invalid id: {raw_id!r}.")

    id_counts = Counter(ids)
    duplicate_ids = sorted(chunk_id for chunk_id, count in id_counts.items() if count > 1)
    if duplicate_ids:
        raise SparseRetrievalError(
            f"Chroma get() response contains duplicate id(s): {duplicate_ids}."
        )

    requested_ids = {chunk_id for _position, chunk_id, _score in ranked_candidates}
    if len(ids) != len(requested_ids):
        raise SparseRetrievalError(
            f"Chroma get() returned {len(ids)} record(s) for {len(requested_ids)} requested "
            "unique id(s)."
        )

    returned_ids = set(ids)
    missing_ids = requested_ids - returned_ids
    if missing_ids:
        raise SparseRetrievalError(
            f"Chroma get() did not return {len(missing_ids)} requested id(s): "
            f"{sorted(missing_ids)}."
        )
    extra_ids = returned_ids - requested_ids
    if extra_ids:
        raise SparseRetrievalError(
            f"Chroma get() returned {len(extra_ids)} unexpected id(s) that were not "
            f"requested: {sorted(extra_ids)}."
        )

    record_by_id: dict[str, tuple[str, Metadata]] = {}
    for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=True):
        record_by_id[chunk_id] = (text, metadata)

    results: list[SparseRetrievalResult] = []
    for rank, (_position, chunk_id, score) in enumerate(ranked_candidates, start=1):
        text, metadata = record_by_id[chunk_id]
        if not isinstance(text, str):
            raise SparseRetrievalError(
                f"Chroma get() response has no document text for id {chunk_id!r}."
            )
        if metadata is None:
            raise SparseRetrievalError(
                f"Chroma get() response has no metadata for id {chunk_id!r}."
            )

        expected_text = snapshot_texts_by_id.get(chunk_id)
        if text != expected_text:
            raise SparseRetrievalError(
                f"Stored Chroma document text for id {chunk_id!r} does not match the sparse "
                "corpus snapshot text -- dense/sparse synchronization may be broken."
            )

        document_id = require_str(metadata, "document_id", chunk_id, SparseRetrievalError)
        chunk_index = require_int(metadata, "chunk_index", chunk_id, SparseRetrievalError)
        source_file = require_str(metadata, "source_file", chunk_id, SparseRetrievalError)
        chunking_strategy = parse_chunking_strategy(
            metadata, chunk_id, requested_strategy, SparseRetrievalError
        )

        results.append(
            SparseRetrievalResult(
                chunk_id=chunk_id,
                rank=rank,
                text=text,
                bm25_score=score,
                document_id=document_id,
                chunk_index=chunk_index,
                source_file=source_file,
                section_heading=optional_str(
                    metadata, "section_heading", chunk_id, SparseRetrievalError
                ),
                page_number=optional_int(metadata, "page_number", chunk_id, SparseRetrievalError),
                chunking_strategy=chunking_strategy,
            )
        )
    return results


def retrieve_sparse(
    query: str,
    strategy: ChunkingStrategy,
    settings: Settings,
    top_k: int | None = None,
) -> list[SparseRetrievalResult]:
    """Return the top-k BM25 lexical matches for `query` under `strategy`'s active index.

    The active sparse snapshot is resolved solely from the strategy's
    active manifest (`snapshot_id`) -- never guessed, never scanned across
    all persisted BM25 directories. Raises `IndexNotReadyError` if no
    manifest is active for `strategy`, `TokenizerVersionMismatchError` if
    the active index's BM25 tokenizer version doesn't match the runtime
    tokenizer, `InvalidQueryError` if the query is empty/whitespace-only or
    tokenizes to nothing, and `SparseRetrievalError` for any BM25-
    reconstruction, scoring, or Chroma-hydration problem that can't be
    trusted. No API key or embedding provider is required.
    """
    validate_query(query)
    query_tokens = tokenize(query)
    if not query_tokens:
        raise InvalidQueryError(
            f"Query {query!r} produced no usable tokens under tokenizer_version="
            f"{TOKENIZER_VERSION!r}."
        )
    resolved_top_k = resolve_top_k(top_k, settings.sparse_top_k)
    manifest = load_active_manifest(settings, strategy)
    _check_tokenizer_compatibility(manifest)

    candidates, snapshot_texts_by_id = _score_candidates(settings, manifest, query_tokens)
    n_results = min(resolved_top_k, manifest.chunk_count)
    ranked_candidates = _rank_candidates(candidates, n_results)

    top_chunk_ids = [chunk_id for _position, chunk_id, _score in ranked_candidates]
    try:
        client = get_chroma_client(settings)
        collection = client.get_collection(name=manifest.chroma_collection_name)
        stored = collection.get(ids=top_chunk_ids, include=["documents", "metadatas"])
    except ChromaError as exc:
        raise SparseRetrievalError(
            f"Failed to hydrate sparse results from Chroma collection "
            f"{manifest.chroma_collection_name!r} for strategy={strategy.value!r}: {exc}"
        ) from exc

    results = _hydrate_results(stored, ranked_candidates, snapshot_texts_by_id, strategy)

    logger.info(
        "sparse retrieval: strategy=%s snapshot_id=%s top_k=%d token_count=%d returned=%d "
        "tokenizer_version=%s",
        strategy.value,
        manifest.snapshot_id,
        resolved_top_k,
        len(query_tokens),
        len(results),
        TOKENIZER_VERSION,
    )
    return results
