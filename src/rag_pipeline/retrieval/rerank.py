"""Pure reranking of RRF-fused hybrid candidates via a `Reranker`.

No I/O beyond the injected `Reranker.score()` call: this module never
runs dense/sparse retrieval, never invokes RRF fusion, and never writes
index artifacts -- given an already-fused `HybridRetrievalResult`
candidate list and a `Reranker`, it validates input integrity, requests
one relevance score per candidate (in candidate order), validates the
returned scores, and returns a deterministically sorted/truncated
`RerankedRetrievalResult` list. This keeps reranking math and
provider-output validation independently unit-testable, without any
index, embedding, or BM25 fixture.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..reranking.base import Reranker
from ._shared import validate_query
from .exceptions import RerankError
from .models import HybridRetrievalResult, RerankedRetrievalResult


def _validate_candidates(candidates: Sequence[HybridRetrievalResult]) -> None:
    """A valid candidate list has unique chunk_ids and contiguous hybrid ranks 1..N.

    Contiguity is enforced for the same reason `fuse_rankings()` enforces
    it on its own inputs: `retrieve_hybrid()` always emits exactly that
    (hybrid ranks assigned by `enumerate(..., start=1)` over its own
    already-sorted output), so any gap or out-of-order rank is a sign of
    a hand-built or corrupted candidate list that must not be silently
    repaired.
    """
    seen_chunk_ids: set[str] = set()
    seen_hybrid_ranks: set[int] = set()
    hybrid_ranks: list[int] = []
    for candidate in candidates:
        if candidate.rank <= 0:
            raise RerankError(f"candidate hybrid rank must be positive, got {candidate.rank!r}.")
        if candidate.chunk_id in seen_chunk_ids:
            raise RerankError(f"duplicate candidate chunk_id: {candidate.chunk_id!r}.")
        if candidate.rank in seen_hybrid_ranks:
            raise RerankError(f"duplicate candidate hybrid rank: {candidate.rank!r}.")
        seen_chunk_ids.add(candidate.chunk_id)
        seen_hybrid_ranks.add(candidate.rank)
        hybrid_ranks.append(candidate.rank)

    if hybrid_ranks != sorted(hybrid_ranks):
        raise RerankError("candidates are not ordered consistently by ascending hybrid rank.")

    expected = list(range(1, len(candidates) + 1))
    if hybrid_ranks != expected:
        raise RerankError(
            f"candidate hybrid ranks are not contiguous starting at 1: got {hybrid_ranks}, "
            f"expected {expected}."
        )


def _validate_scores(scores: Sequence[float], expected_count: int) -> list[float]:
    if len(scores) != expected_count:
        raise RerankError(
            f"reranker returned {len(scores)} scores for {expected_count} candidates."
        )
    validated: list[float] = []
    for index, score in enumerate(scores):
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise RerankError(f"reranker score at position {index} is not numeric: {score!r}.")
        value = float(score)
        if not math.isfinite(value):
            raise RerankError(f"reranker score at position {index} is not finite: {value!r}.")
        validated.append(value)
    return validated


def _sort_key(scored: tuple[float, HybridRetrievalResult]) -> tuple[float, int, str]:
    """reranker_score DESC, then previous hybrid_rank ASC, then chunk_id ASC.

    Both tie-break levels are total orders over distinct candidates
    (hybrid_rank and chunk_id are each unique within one candidate list,
    per `_validate_candidates`), so this always resolves deterministically.
    """
    score, candidate = scored
    return (-score, candidate.rank, candidate.chunk_id)


def rerank_candidates(
    query: str,
    candidates: Sequence[HybridRetrievalResult],
    reranker: Reranker,
    *,
    top_k: int,
) -> list[RerankedRetrievalResult]:
    """Rerank already-fused hybrid `candidates` by `reranker`-assigned relevance.

    Ordering: `reranker_score` DESC, ties broken by `hybrid_rank` ASC then
    `chunk_id` ASC -- same inputs and scores always produce identical
    output. Final ranks are assigned 1..N only after sorting and
    truncating to `top_k`; fewer than `top_k` candidates returns all of
    them, and an empty candidate list returns an empty list without
    calling the reranker.

    Raises `RerankError` for an empty/whitespace query, malformed
    candidate integrity, a non-positive `top_k`, or a malformed reranker
    score count/value (wrong count, non-numeric, bool, non-finite).
    Any exception the reranker itself raises (e.g. `RerankerError`)
    propagates unwrapped -- see `retrieval.reranked` for orchestration-
    level wrapping.
    """
    validate_query(query)
    if top_k <= 0:
        raise RerankError(f"top_k must be positive, got {top_k!r}.")

    _validate_candidates(candidates)

    if not candidates:
        return []

    raw_scores = reranker.score(query, [candidate.text for candidate in candidates])
    scores = _validate_scores(raw_scores, len(candidates))

    ordered = sorted(zip(scores, candidates, strict=True), key=_sort_key)
    truncated = ordered[:top_k]

    results: list[RerankedRetrievalResult] = []
    for final_rank, (score, candidate) in enumerate(truncated, start=1):
        results.append(
            RerankedRetrievalResult(
                chunk_id=candidate.chunk_id,
                rank=final_rank,
                reranker_score=score,
                hybrid_rank=candidate.rank,
                rrf_score=candidate.rrf_score,
                dense_rank=candidate.dense_rank,
                sparse_rank=candidate.sparse_rank,
                dense_contribution=candidate.dense_contribution,
                sparse_contribution=candidate.sparse_contribution,
                dense_distance=candidate.dense_distance,
                dense_similarity=candidate.dense_similarity,
                bm25_score=candidate.bm25_score,
                text=candidate.text,
                document_id=candidate.document_id,
                chunk_index=candidate.chunk_index,
                source_file=candidate.source_file,
                section_heading=candidate.section_heading,
                page_number=candidate.page_number,
                chunking_strategy=candidate.chunking_strategy,
            )
        )
    return results
