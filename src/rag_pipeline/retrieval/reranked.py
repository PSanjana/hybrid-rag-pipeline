"""Reranked retrieval orchestration: hybrid RRF candidates -> reranker -> final top-k.

    query
    -> retrieve_hybrid(top_k=rerank_candidate_k)  [wide candidate pool, recall-focused]
    -> rerank_candidates(...)                     [pure cross-encoder-style reranking]
    -> final RerankedRetrievalResult list          [narrow, precision-focused]

`retrieve_hybrid()` remains the sole authoritative hybrid-retrieval
implementation -- this module only requests a wider candidate depth from
it (`settings.rerank_candidate_k`, deliberately independent of the
general-purpose `settings.hybrid_top_k` default) and reranks its output.
A hybrid retrieval failure, or a reranker *provider* failure, is wrapped
as `RerankedRetrievalError` (original exception chained via `__cause__`)
rather than silently falling back to the unreranked hybrid ordering. A
malformed-candidate/malformed-score `RerankError` from the pure
`rerank_candidates()` layer is not wrapped further, exactly mirroring how
`retrieve_hybrid()` lets a `FusionError` from `fuse_rankings()` propagate
unwrapped.
"""

from __future__ import annotations

import logging

from ..config import ChunkingStrategy, Settings
from ..embeddings import EmbeddingProvider
from ..reranking.base import Reranker
from ..reranking.exceptions import RerankingError
from ._shared import resolve_top_k
from .exceptions import RerankedRetrievalError, RetrievalError
from .hybrid import retrieve_hybrid
from .models import RerankedRetrievalResult
from .rerank import rerank_candidates

logger = logging.getLogger(__name__)


def retrieve_reranked(
    query: str,
    strategy: ChunkingStrategy,
    settings: Settings,
    reranker: Reranker,
    embedding_provider: EmbeddingProvider | None = None,
    dense_top_k: int | None = None,
    sparse_top_k: int | None = None,
    candidate_k: int | None = None,
    final_top_k: int | None = None,
) -> list[RerankedRetrievalResult]:
    """Return the top `final_top_k` reranked chunks for `query` under `strategy`'s active index.

    Requests `candidate_k` (defaulting to `settings.rerank_candidate_k`,
    NOT `settings.hybrid_top_k`) fused hybrid candidates from
    `retrieve_hybrid()`, then reranks exactly that candidate set down to
    `final_top_k` (defaulting to `settings.rerank_top_k`) via
    `rerank_candidates()`. `dense_top_k`/`sparse_top_k` pass straight
    through to `retrieve_hybrid()`, independent of `candidate_k`.

    Raises `RerankedRetrievalError` if hybrid retrieval fails or the
    reranker provider fails (original exception chained as `__cause__`),
    or `RerankError` if the candidate set/reranker output cannot be
    trusted (see `retrieval.rerank`).
    """
    resolved_candidate_k = resolve_top_k(candidate_k, settings.rerank_candidate_k)
    resolved_final_top_k = resolve_top_k(final_top_k, settings.rerank_top_k)
    if resolved_final_top_k > resolved_candidate_k:
        raise RetrievalError(
            f"final_top_k ({resolved_final_top_k}) must not exceed candidate_k "
            f"({resolved_candidate_k})."
        )

    try:
        candidates = retrieve_hybrid(
            query,
            strategy,
            settings,
            embedding_provider=embedding_provider,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            hybrid_top_k=resolved_candidate_k,
        )
    except RetrievalError as exc:
        raise RerankedRetrievalError(f"Hybrid retrieval failed during reranking: {exc}") from exc

    try:
        results = rerank_candidates(query, candidates, reranker, top_k=resolved_final_top_k)
    except RerankingError as exc:
        raise RerankedRetrievalError(f"Reranker provider failed: {exc}") from exc

    logger.info(
        "reranked retrieval: strategy=%s candidate_count=%d final_count=%d "
        "candidate_k=%d final_top_k=%d",
        strategy.value,
        len(candidates),
        len(results),
        resolved_candidate_k,
        resolved_final_top_k,
    )
    return results
