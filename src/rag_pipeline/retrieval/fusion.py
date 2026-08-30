"""Pure weighted Reciprocal Rank Fusion (RRF) of independent dense and sparse rankings.

No I/O anywhere in this module: it never calls an embedding provider,
Chroma, or BM25, and depends only on already-computed
`DenseRetrievalResult`/`SparseRetrievalResult` lists plus fusion
parameters -- so it is exhaustively unit-testable without any index or
corpus fixture.

Formula, for a chunk_id appearing at dense rank `r_d` and/or sparse rank
`r_s` (both 1-based; a retriever that never returned the chunk at all
contributes exactly 0, not a penalty score):

    dense_contribution  = dense_weight  / (rank_constant + r_d)   -- 0.0 if chunk absent from dense
    sparse_contribution = sparse_weight / (rank_constant + r_s)   -- 0.0 if chunk absent from sparse
    rrf_score = dense_contribution + sparse_contribution

Deliberately never used in the score: dense cosine similarity/distance,
raw BM25 score. Those two scales are not comparable to each other (cosine
is bounded, BM25 is unbounded and corpus/query-dependent) -- RRF sidesteps
that problem entirely by fusing rank *positions*, never raw scores.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .exceptions import FusionError
from .models import DenseRetrievalResult, HybridRetrievalResult, SparseRetrievalResult

_PROVENANCE_FIELDS = (
    "text",
    "document_id",
    "chunk_index",
    "source_file",
    "section_heading",
    "page_number",
    "chunking_strategy",
)


@dataclass(frozen=True, slots=True)
class _FusedCandidate:
    chunk_id: str
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None
    dense_contribution: float
    sparse_contribution: float
    dense_result: DenseRetrievalResult | None
    sparse_result: SparseRetrievalResult | None


def _validate_ranking(
    results: Sequence[DenseRetrievalResult] | Sequence[SparseRetrievalResult], label: str
) -> None:
    """A valid single-retriever ranking has unique chunk_ids and contiguous ranks 1..N.

    Contiguity (not merely positive+unique ranks) is enforced because
    `retrieve_dense()`/`retrieve_sparse()` always emit exactly that --
    `rank = enumerate(results, start=1)` over their own already-ordered
    output -- so any gap, skip, or non-ascending order is a sign of a
    hand-built or corrupted input that must not be silently repaired.
    """
    seen_chunk_ids: set[str] = set()
    ranks: list[int] = []
    for result in results:
        if result.rank <= 0:
            raise FusionError(f"{label} ranking contains a non-positive rank: {result.rank!r}.")
        if result.chunk_id in seen_chunk_ids:
            raise FusionError(
                f"{label} ranking contains chunk_id {result.chunk_id!r} more than once."
            )
        seen_chunk_ids.add(result.chunk_id)
        ranks.append(result.rank)

    if ranks != sorted(ranks):
        raise FusionError(f"{label} ranking is not ordered consistently by ascending rank.")

    expected = list(range(1, len(results) + 1))
    if ranks != expected:
        raise FusionError(
            f"{label} ranking ranks are not contiguous starting at 1: got {ranks}, expected "
            f"{expected}."
        )


def _verify_overlap_provenance(
    dense_result: DenseRetrievalResult, sparse_result: SparseRetrievalResult
) -> None:
    """A chunk_id present in both rankings must describe the same canonical chunk.

    Neither side is preferred/trusted over the other on a mismatch --
    dense and sparse are supposed to read the same synchronized corpus, so
    disagreement here means something is corrupted, not that one side is
    "more right."
    """
    mismatched_fields = [
        field
        for field in _PROVENANCE_FIELDS
        if getattr(dense_result, field) != getattr(sparse_result, field)
    ]
    if mismatched_fields:
        raise FusionError(
            f"Dense and sparse results for chunk_id={dense_result.chunk_id!r} disagree on "
            f"{mismatched_fields}; the two retrievers must operate on the same synchronized "
            "corpus."
        )


def _sort_key(candidate: _FusedCandidate) -> tuple[float, float, float, float, str]:
    """rrf_score DESC, then deterministic evidence-aware tie-breaking.

    Tie-break order (each step only applies once every prior step ties
    exactly):
      1. rrf_score DESC
      2. best available individual rank ASC (min of dense/sparse rank,
         missing treated as +inf)
      3. dense rank ASC, missing treated as +inf
      4. sparse rank ASC, missing treated as +inf
      5. chunk_id ASC (a total order: every candidate's chunk_id is
         unique, so this always breaks any remaining tie deterministically)

    Never depends on set/dict iteration order: candidates are gathered
    from dicts keyed by chunk_id (order-irrelevant), and this key alone
    determines the final sequence.
    """
    dense_sort = candidate.dense_rank if candidate.dense_rank is not None else math.inf
    sparse_sort = candidate.sparse_rank if candidate.sparse_rank is not None else math.inf
    best_rank = min(dense_sort, sparse_sort)
    return (-candidate.rrf_score, best_rank, dense_sort, sparse_sort, candidate.chunk_id)


def fuse_rankings(
    dense_results: Sequence[DenseRetrievalResult],
    sparse_results: Sequence[SparseRetrievalResult],
    *,
    dense_weight: float,
    sparse_weight: float,
    rank_constant: float,
    top_k: int,
) -> list[HybridRetrievalResult]:
    """Fuse independently-ranked dense/sparse results into one deterministic hybrid ranking.

    Candidate identity is `chunk_id` alone; the candidate set is the
    UNION of dense and sparse chunk_ids (never the intersection) -- a
    chunk returned by only one retriever remains eligible, contributing
    only that retriever's term to `rrf_score`. See the module docstring
    for the exact formula and `_sort_key` for the tie-break rule.

    Raises `FusionError` for: a non-positive or non-contiguous rank in
    either input ranking, a duplicate chunk_id within one ranking, a
    provenance mismatch on a chunk_id present in both rankings, an
    invalid weight/rank_constant/top_k.
    """
    if dense_weight < 0:
        raise FusionError(f"dense_weight must not be negative, got {dense_weight!r}.")
    if sparse_weight < 0:
        raise FusionError(f"sparse_weight must not be negative, got {sparse_weight!r}.")
    if dense_weight + sparse_weight <= 0:
        raise FusionError("At least one of dense_weight/sparse_weight must be positive.")
    if rank_constant < 0:
        raise FusionError(f"rank_constant must not be negative, got {rank_constant!r}.")
    if top_k <= 0:
        raise FusionError(f"top_k must be positive, got {top_k!r}.")

    _validate_ranking(dense_results, "dense")
    _validate_ranking(sparse_results, "sparse")

    dense_by_id = {result.chunk_id: result for result in dense_results}
    sparse_by_id = {result.chunk_id: result for result in sparse_results}

    for chunk_id in dense_by_id.keys() & sparse_by_id.keys():
        _verify_overlap_provenance(dense_by_id[chunk_id], sparse_by_id[chunk_id])

    candidates: list[_FusedCandidate] = []
    for chunk_id in dense_by_id.keys() | sparse_by_id.keys():
        dense_result = dense_by_id.get(chunk_id)
        sparse_result = sparse_by_id.get(chunk_id)

        dense_rank = dense_result.rank if dense_result is not None else None
        sparse_rank = sparse_result.rank if sparse_result is not None else None

        dense_contribution = (
            dense_weight / (rank_constant + dense_rank) if dense_rank is not None else 0.0
        )
        sparse_contribution = (
            sparse_weight / (rank_constant + sparse_rank) if sparse_rank is not None else 0.0
        )

        candidates.append(
            _FusedCandidate(
                chunk_id=chunk_id,
                rrf_score=dense_contribution + sparse_contribution,
                dense_rank=dense_rank,
                sparse_rank=sparse_rank,
                dense_contribution=dense_contribution,
                sparse_contribution=sparse_contribution,
                dense_result=dense_result,
                sparse_result=sparse_result,
            )
        )

    ordered = sorted(candidates, key=_sort_key)
    truncated = ordered[:top_k]

    results: list[HybridRetrievalResult] = []
    for hybrid_rank, candidate in enumerate(truncated, start=1):
        source = (
            candidate.dense_result
            if candidate.dense_result is not None
            else candidate.sparse_result
        )
        assert source is not None  # every candidate has at least one contributing side

        results.append(
            HybridRetrievalResult(
                chunk_id=candidate.chunk_id,
                rank=hybrid_rank,
                rrf_score=candidate.rrf_score,
                dense_rank=candidate.dense_rank,
                sparse_rank=candidate.sparse_rank,
                dense_contribution=candidate.dense_contribution,
                sparse_contribution=candidate.sparse_contribution,
                dense_distance=candidate.dense_result.distance
                if candidate.dense_result is not None
                else None,
                dense_similarity=candidate.dense_result.similarity
                if candidate.dense_result is not None
                else None,
                bm25_score=candidate.sparse_result.bm25_score
                if candidate.sparse_result is not None
                else None,
                text=source.text,
                document_id=source.document_id,
                chunk_index=source.chunk_index,
                source_file=source.source_file,
                section_heading=source.section_heading,
                page_number=source.page_number,
                chunking_strategy=source.chunking_strategy,
            )
        )
    return results
