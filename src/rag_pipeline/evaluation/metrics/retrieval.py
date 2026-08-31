"""Deterministic retrieval-relevance metrics for one golden case (offline, no LLM).

`evaluate_retrieval()` operates on nothing but each result's `source_file`,
`text`, and position in the list. It never reads a native retrieval score
(cosine similarity/distance, BM25 score, RRF score, reranker score), so the
same function scores dense, sparse, hybrid, and reranked result lists
identically -- score scales from different channels are not comparable and
are not evaluation truth.

No `chunk_precision_at_k` is computed. The golden benchmark labels required
and acceptable *source documents*, not every relevant chunk, so treating
"every chunk from an expected document" as relevant would claim a stronger
relevance labelling than the dataset actually contains. Likewise an
unlisted document is not assumed irrelevant.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ..exceptions import MetricInputError
from ..models import Answerability, GoldenQACase
from .models import RetrievalMetrics


@runtime_checkable
class RetrievedChunk(Protocol):
    """The minimal structural view of a retrieval result this metric needs.

    Every Phase 2 result type -- `DenseRetrievalResult`,
    `SparseRetrievalResult`, `HybridRetrievalResult`,
    `RerankedRetrievalResult` -- already exposes these three attributes,
    so the metric couples to this shape, never to a concrete class or a
    channel-specific relevance score.
    """

    chunk_id: str
    text: str
    source_file: str


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MetricInputError(f"{label} must be a positive int, got {value!r}.")
    return value


def evaluate_retrieval(
    case: GoldenQACase,
    results: Sequence[RetrievedChunk],
    k: int,
) -> RetrievalMetrics:
    """Score one ordered retrieval result sequence against `case` at cut-off `k`.

    For an **unanswerable** case every source signal is `None` (there is
    no expected-source truth -- N/A is never silently turned into 0.0).

    For an **answerable** case, over the top `k` results:

    * ``required_source_hit_at_k`` -- 1.0 iff at least one
      ``expected_source_files`` document appears, else 0.0.
    * ``required_source_recall_at_k`` -- distinct required documents seen
      / number of ``expected_source_files``. Repeated chunks from one
      document count once.
    * ``complete_required_source_retrieval_at_k`` -- ``True`` iff *every*
      ``expected_source_files`` document appears in the top ``k``.
    * ``reciprocal_rank`` -- ``1 / rank`` of the first result (1-based,
      over the *whole* sequence, not just the top ``k``) whose
      ``source_file`` is a required document; 0.0 if none.
    * ``identifier_recall_at_k`` -- distinct ``expected_identifiers`` that
      occur (case-insensitive substring, no tokenisation/expansion) in
      the text of at least one top-``k`` chunk / number of identifiers;
      ``None`` when the case lists no identifiers.

    Raises `MetricInputError` if ``k`` is not a positive int.
    """
    k = _require_positive_int(k, "k")

    ordered = list(results)
    top_k = ordered[:k]

    if case.answerability is not Answerability.ANSWERABLE:
        return RetrievalMetrics(
            k=k,
            required_source_hit_at_k=None,
            required_source_recall_at_k=None,
            complete_required_source_retrieval_at_k=None,
            reciprocal_rank=None,
            identifier_recall_at_k=None,
        )

    required = case.expected_source_files
    required_set = set(required)
    top_k_sources = {chunk.source_file for chunk in top_k}

    found = tuple(name for name in required if name in top_k_sources)
    missing = tuple(name for name in required if name not in top_k_sources)

    hit = 1.0 if found else 0.0
    recall = len(found) / len(required)
    complete = not missing

    reciprocal_rank = 0.0
    for rank, chunk in enumerate(ordered, start=1):
        if chunk.source_file in required_set:
            reciprocal_rank = 1.0 / rank
            break

    identifiers = case.expected_identifiers
    if identifiers:
        lowered_texts = [chunk.text.lower() for chunk in top_k]
        ids_found = tuple(
            identifier
            for identifier in identifiers
            if any(identifier.lower() in text for text in lowered_texts)
        )
        ids_missing = tuple(identifier for identifier in identifiers if identifier not in ids_found)
        identifier_recall: float | None = len(ids_found) / len(identifiers)
    else:
        ids_found = ()
        ids_missing = ()
        identifier_recall = None

    return RetrievalMetrics(
        k=k,
        required_source_hit_at_k=hit,
        required_source_recall_at_k=recall,
        complete_required_source_retrieval_at_k=complete,
        reciprocal_rank=reciprocal_rank,
        identifier_recall_at_k=identifier_recall,
        required_sources_found=found,
        required_sources_missing=missing,
        identifiers_found=ids_found,
        identifiers_missing=ids_missing,
    )
