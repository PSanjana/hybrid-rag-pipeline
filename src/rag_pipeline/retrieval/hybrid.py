"""Hybrid retrieval orchestration: dense + sparse + weighted RRF fusion.

    question
    -> retrieve_dense(...)   [the authoritative dense search, unmodified]
    -> retrieve_sparse(...)  [the authoritative sparse search, unmodified]
    -> fuse_rankings(...)    [pure RRF fusion, no I/O -- see retrieval.fusion]

Both channels must succeed: if either `retrieve_dense()` or
`retrieve_sparse()` raises, `retrieve_hybrid()` wraps it as
`HybridRetrievalError` (original exception chained via `__cause__`)
rather than silently degrading to a single-channel result. Query
validation, embedding, and tokenization are never duplicated here --
`retrieve_dense()`/`retrieve_sparse()` remain the sole authoritative
implementations of those steps; this module only orchestrates and fuses
their already-computed, already-ranked outputs.
"""

from __future__ import annotations

import logging

from ..config import ChunkingStrategy, Settings
from ..embeddings import EmbeddingProvider
from ._shared import resolve_top_k
from .dense import retrieve_dense
from .exceptions import HybridRetrievalError, RetrievalError
from .fusion import fuse_rankings
from .models import HybridRetrievalResult
from .sparse import retrieve_sparse

logger = logging.getLogger(__name__)


def retrieve_hybrid(
    query: str,
    strategy: ChunkingStrategy,
    settings: Settings,
    embedding_provider: EmbeddingProvider | None = None,
    dense_top_k: int | None = None,
    sparse_top_k: int | None = None,
    hybrid_top_k: int | None = None,
) -> list[HybridRetrievalResult]:
    """Return the top `hybrid_top_k` RRF-fused chunks for `query` under `strategy`'s active index.

    `dense_top_k`/`sparse_top_k` independently override each channel's own
    candidate depth (falling back to `settings.dense_top_k`/
    `settings.sparse_top_k` when omitted, exactly as `retrieve_dense()`/
    `retrieve_sparse()` already do) -- they are not forced to match each
    other or `hybrid_top_k`. Fusion weights/rank constant always come from
    `settings.rrf_dense_weight`/`settings.rrf_sparse_weight`/
    `settings.rrf_rank_constant`.

    Raises `HybridRetrievalError` if either retrieval channel fails (the
    original exception is chained as `__cause__`), or `FusionError` if the
    two channels' results cannot be safely fused (see `retrieval.fusion`).
    """
    resolved_hybrid_top_k = resolve_top_k(hybrid_top_k, settings.hybrid_top_k)

    try:
        dense_results = retrieve_dense(
            query,
            strategy,
            settings,
            embedding_provider=embedding_provider,
            top_k=dense_top_k,
        )
    except RetrievalError as exc:
        raise HybridRetrievalError(f"Dense retrieval failed during hybrid fusion: {exc}") from exc

    try:
        sparse_results = retrieve_sparse(query, strategy, settings, top_k=sparse_top_k)
    except RetrievalError as exc:
        raise HybridRetrievalError(f"Sparse retrieval failed during hybrid fusion: {exc}") from exc

    results = fuse_rankings(
        dense_results,
        sparse_results,
        dense_weight=settings.rrf_dense_weight,
        sparse_weight=settings.rrf_sparse_weight,
        rank_constant=settings.rrf_rank_constant,
        top_k=resolved_hybrid_top_k,
    )

    logger.info(
        "hybrid retrieval: strategy=%s dense_count=%d sparse_count=%d fused_count=%d "
        "hybrid_top_k=%d dense_weight=%.3f sparse_weight=%.3f rank_constant=%.1f",
        strategy.value,
        len(dense_results),
        len(sparse_results),
        len(results),
        resolved_hybrid_top_k,
        settings.rrf_dense_weight,
        settings.rrf_sparse_weight,
        settings.rrf_rank_constant,
    )
    return results
