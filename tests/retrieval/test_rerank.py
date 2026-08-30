"""Tests for rag_pipeline.retrieval.rerank (pure candidate reranking, no I/O)."""

from __future__ import annotations

import math

import pytest

from rag_pipeline.config import ChunkingStrategy
from rag_pipeline.retrieval.exceptions import InvalidQueryError, RerankError
from rag_pipeline.retrieval.models import RerankedRetrievalResult
from rag_pipeline.retrieval.rerank import rerank_candidates

from .conftest import FakeReranker, make_hybrid_result

_QUERY = "a query"


def _rerank(candidates, reranker, **overrides):
    params = {"top_k": 10, **overrides}
    return rerank_candidates(_QUERY, candidates, reranker, **params)


# --- reranker invocation -----------------------------------------------------------


def test_one_candidate_gets_one_reranker_score() -> None:
    candidates = [make_hybrid_result(chunk_id="a", rank=1)]
    reranker = FakeReranker({"text for a": 3.0})
    results = _rerank(candidates, reranker)
    assert len(results) == 1
    assert results[0].reranker_score == pytest.approx(3.0)


def test_reranker_receives_query_correctly() -> None:
    candidates = [make_hybrid_result(chunk_id="a", rank=1)]
    reranker = FakeReranker({"text for a": 1.0})
    _rerank(candidates, reranker)
    assert reranker.calls[0][0] == _QUERY


def test_documents_passed_in_hybrid_rank_order() -> None:
    candidates = [
        make_hybrid_result(chunk_id="a", rank=1, text="text-a"),
        make_hybrid_result(chunk_id="b", rank=2, text="text-b"),
        make_hybrid_result(chunk_id="c", rank=3, text="text-c"),
    ]
    reranker = FakeReranker({"text-a": 1.0, "text-b": 2.0, "text-c": 3.0})
    _rerank(candidates, reranker)
    assert reranker.calls[0][1] == ["text-a", "text-b", "text-c"]


def test_one_score_required_per_candidate() -> None:
    candidates = [
        make_hybrid_result(chunk_id="a", rank=1, text="text-a"),
        make_hybrid_result(chunk_id="b", rank=2, text="text-b"),
    ]
    reranker = FakeReranker({"text-a": 1.0, "text-b": 2.0})
    results = _rerank(candidates, reranker)
    assert {r.chunk_id for r in results} == {"a", "b"}


# --- score validation ----------------------------------------------------------------


def test_too_few_scores_rejected() -> None:
    candidates = [
        make_hybrid_result(chunk_id="a", rank=1),
        make_hybrid_result(chunk_id="b", rank=2),
    ]
    reranker = FakeReranker(override_scores=[1.0])
    with pytest.raises(RerankError):
        _rerank(candidates, reranker)


def test_too_many_scores_rejected() -> None:
    candidates = [make_hybrid_result(chunk_id="a", rank=1)]
    reranker = FakeReranker(override_scores=[1.0, 2.0])
    with pytest.raises(RerankError):
        _rerank(candidates, reranker)


def test_non_numeric_score_rejected() -> None:
    candidates = [make_hybrid_result(chunk_id="a", rank=1)]
    reranker = FakeReranker(override_scores=["not-a-number"])
    with pytest.raises(RerankError):
        _rerank(candidates, reranker)


def test_bool_score_rejected() -> None:
    candidates = [make_hybrid_result(chunk_id="a", rank=1)]
    reranker = FakeReranker(override_scores=[True])
    with pytest.raises(RerankError):
        _rerank(candidates, reranker)


def test_nan_score_rejected() -> None:
    candidates = [make_hybrid_result(chunk_id="a", rank=1)]
    reranker = FakeReranker(override_scores=[math.nan])
    with pytest.raises(RerankError):
        _rerank(candidates, reranker)


def test_positive_infinity_score_rejected() -> None:
    candidates = [make_hybrid_result(chunk_id="a", rank=1)]
    reranker = FakeReranker(override_scores=[math.inf])
    with pytest.raises(RerankError):
        _rerank(candidates, reranker)


def test_negative_infinity_score_rejected() -> None:
    candidates = [make_hybrid_result(chunk_id="a", rank=1)]
    reranker = FakeReranker(override_scores=[-math.inf])
    with pytest.raises(RerankError):
        _rerank(candidates, reranker)


def test_finite_negative_score_accepted() -> None:
    candidates = [make_hybrid_result(chunk_id="a", rank=1)]
    reranker = FakeReranker(override_scores=[-5.5])
    results = _rerank(candidates, reranker)
    assert results[0].reranker_score == pytest.approx(-5.5)


# --- ranking / ordering ----------------------------------------------------------


def test_highest_reranker_score_becomes_rank_one() -> None:
    candidates = [
        make_hybrid_result(chunk_id="a", rank=1, text="text-a"),
        make_hybrid_result(chunk_id="b", rank=2, text="text-b"),
    ]
    reranker = FakeReranker({"text-a": 1.0, "text-b": 9.0})
    results = _rerank(candidates, reranker)
    assert results[0].chunk_id == "b"
    assert results[0].rank == 1
    assert results[1].chunk_id == "a"
    assert results[1].rank == 2


def test_original_hybrid_rank_does_not_force_final_order() -> None:
    # "a" is hybrid rank 1 (best pre-rerank) but scores lowest; "c" is
    # hybrid rank 3 (worst pre-rerank) but scores highest.
    candidates = [
        make_hybrid_result(chunk_id="a", rank=1, text="text-a"),
        make_hybrid_result(chunk_id="b", rank=2, text="text-b"),
        make_hybrid_result(chunk_id="c", rank=3, text="text-c"),
    ]
    reranker = FakeReranker({"text-a": 0.1, "text-b": 0.5, "text-c": 9.9})
    results = _rerank(candidates, reranker)
    assert [r.chunk_id for r in results] == ["c", "b", "a"]


def test_tie_uses_lower_hybrid_rank_first() -> None:
    candidates = [
        make_hybrid_result(chunk_id="a", rank=1, text="text-a"),
        make_hybrid_result(chunk_id="b", rank=2, text="text-b"),
    ]
    reranker = FakeReranker({"text-a": 5.0, "text-b": 5.0})
    results = _rerank(candidates, reranker)
    assert [r.chunk_id for r in results] == ["a", "b"]


def test_final_chunk_id_fallback_deterministic() -> None:
    # Same reranker_score AND same hybrid_rank is impossible for two
    # distinct candidates through the validated input (hybrid ranks are
    # unique per _validate_candidates); directly exercise the sort key's
    # final chunk_id-ASC fallback in isolation instead.
    from rag_pipeline.retrieval.rerank import _sort_key

    candidate_a = make_hybrid_result(chunk_id="zzz", rank=1)
    candidate_b = make_hybrid_result(chunk_id="aaa", rank=1)
    ordered = sorted([(5.0, candidate_a), (5.0, candidate_b)], key=_sort_key)
    assert [c.chunk_id for _, c in ordered] == ["aaa", "zzz"]


def test_final_ranks_start_at_one() -> None:
    candidates = [
        make_hybrid_result(chunk_id="a", rank=1, text="text-a"),
        make_hybrid_result(chunk_id="b", rank=2, text="text-b"),
    ]
    reranker = FakeReranker({"text-a": 2.0, "text-b": 1.0})
    results = _rerank(candidates, reranker)
    assert [r.rank for r in results] == [1, 2]


def test_top_k_truncates_after_reranking() -> None:
    candidates = [
        make_hybrid_result(chunk_id=f"c{i}", rank=i, text=f"text-{i}") for i in range(1, 6)
    ]
    reranker = FakeReranker({f"text-{i}": float(i) for i in range(1, 6)})
    results = _rerank(candidates, reranker, top_k=2)
    assert len(results) == 2
    # highest scores are text-5, text-4.
    assert [r.chunk_id for r in results] == ["c5", "c4"]
    assert [r.rank for r in results] == [1, 2]


def test_top_k_greater_than_candidate_count_returns_all() -> None:
    candidates = [make_hybrid_result(chunk_id="a", rank=1)]
    reranker = FakeReranker({"text for a": 1.0})
    results = _rerank(candidates, reranker, top_k=1000)
    assert len(results) == 1


def test_same_inputs_and_scores_always_produce_identical_output() -> None:
    candidates = [
        make_hybrid_result(chunk_id=f"c{i}", rank=i, text=f"text-{i}") for i in range(1, 5)
    ]
    reranker = FakeReranker({f"text-{i}": float(5 - i) for i in range(1, 5)})
    first = _rerank(candidates, reranker)
    second = _rerank(candidates, reranker)
    assert [r.chunk_id for r in first] == [r.chunk_id for r in second]
    assert [r.reranker_score for r in first] == [r.reranker_score for r in second]


# --- candidate integrity -----------------------------------------------------------


def test_duplicate_candidate_chunk_id_rejected() -> None:
    candidates = [
        make_hybrid_result(chunk_id="a", rank=1),
        make_hybrid_result(chunk_id="a", rank=2),
    ]
    with pytest.raises(RerankError):
        _rerank(candidates, FakeReranker())


def test_duplicate_hybrid_rank_rejected() -> None:
    candidates = [
        make_hybrid_result(chunk_id="a", rank=1),
        make_hybrid_result(chunk_id="b", rank=1),
    ]
    with pytest.raises(RerankError):
        _rerank(candidates, FakeReranker())


def test_non_positive_hybrid_rank_rejected() -> None:
    candidates = [make_hybrid_result(chunk_id="a", rank=0)]
    with pytest.raises(RerankError):
        _rerank(candidates, FakeReranker())


def test_non_contiguous_hybrid_ranking_rejected() -> None:
    candidates = [
        make_hybrid_result(chunk_id="a", rank=1),
        make_hybrid_result(chunk_id="b", rank=3),  # gap: rank 2 missing
    ]
    with pytest.raises(RerankError):
        _rerank(candidates, FakeReranker())


def test_malformed_ordering_rejected() -> None:
    candidates = [
        make_hybrid_result(chunk_id="a", rank=2),
        make_hybrid_result(chunk_id="b", rank=1),
    ]
    with pytest.raises(RerankError):
        _rerank(candidates, FakeReranker())


def test_non_positive_top_k_rejected() -> None:
    candidates = [make_hybrid_result(chunk_id="a", rank=1)]
    with pytest.raises(RerankError):
        _rerank(candidates, FakeReranker(), top_k=0)


def test_empty_query_rejected_before_reranker_is_called() -> None:
    candidates = [make_hybrid_result(chunk_id="a", rank=1)]
    reranker = FakeReranker()
    with pytest.raises(InvalidQueryError):
        rerank_candidates("   ", candidates, reranker, top_k=10)
    assert reranker.calls == []


def test_empty_candidate_list_returns_empty_without_calling_reranker() -> None:
    reranker = FakeReranker()
    results = _rerank([], reranker)
    assert results == []
    assert reranker.calls == []


def test_reranker_provider_error_propagates_unwrapped() -> None:
    from rag_pipeline.reranking.exceptions import RerankerError

    candidates = [make_hybrid_result(chunk_id="a", rank=1)]
    reranker = FakeReranker(error=RerankerError("boom"))
    with pytest.raises(RerankerError):
        _rerank(candidates, reranker)


# --- provenance/diagnostics preservation --------------------------------------------


def test_all_hybrid_and_provenance_fields_are_retained() -> None:
    candidate = make_hybrid_result(
        chunk_id="a",
        rank=1,
        rrf_score=0.0123,
        text="the candidate text",
        dense_rank=2,
        sparse_rank=5,
        dense_contribution=0.011,
        sparse_contribution=0.0013,
        dense_distance=0.2,
        dense_similarity=0.8,
        bm25_score=4.5,
        document_id="e" * 64,
        chunk_index=7,
        source_file="doc.md",
        section_heading="Setup",
        page_number=3,
        chunking_strategy=ChunkingStrategy.FIXED,
    )
    reranker = FakeReranker({"the candidate text": 1.0})
    result = _rerank([candidate], reranker)[0]

    assert isinstance(result, RerankedRetrievalResult)
    assert result.hybrid_rank == 1
    assert result.rrf_score == pytest.approx(0.0123)
    assert result.dense_rank == 2
    assert result.sparse_rank == 5
    assert result.dense_contribution == pytest.approx(0.011)
    assert result.sparse_contribution == pytest.approx(0.0013)
    assert result.dense_distance == pytest.approx(0.2)
    assert result.dense_similarity == pytest.approx(0.8)
    assert result.bm25_score == pytest.approx(4.5)
    assert result.text == "the candidate text"
    assert result.document_id == "e" * 64
    assert result.chunk_index == 7
    assert result.source_file == "doc.md"
    assert result.section_heading == "Setup"
    assert result.page_number == 3
    assert result.chunking_strategy == ChunkingStrategy.FIXED


def test_missing_dense_or_sparse_side_is_retained_as_none() -> None:
    candidate = make_hybrid_result(
        chunk_id="a",
        rank=1,
        dense_rank=None,
        sparse_rank=3,
        dense_contribution=0.0,
        dense_distance=None,
        dense_similarity=None,
        bm25_score=1.2,
    )
    reranker = FakeReranker({"text for a": 1.0})
    result = _rerank([candidate], reranker)[0]
    assert result.dense_rank is None
    assert result.dense_distance is None
    assert result.dense_similarity is None
    assert result.sparse_rank == 3
    assert result.bm25_score == pytest.approx(1.2)
