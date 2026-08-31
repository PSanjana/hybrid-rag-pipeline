"""Offline tests for rag_pipeline.evaluation.metrics.retrieval (no LLM, no network)."""

from __future__ import annotations

import pytest

from rag_pipeline.evaluation.exceptions import MetricInputError
from rag_pipeline.evaluation.metrics.retrieval import evaluate_retrieval
from rag_pipeline.evaluation.models import Answerability, QuestionType

from .conftest import make_chunk, make_golden_case


def test_required_source_at_rank_one_gives_hit_at_1() -> None:
    case = make_golden_case(expected_source_files=("a.md",))
    results = [make_chunk(source_file="a.md"), make_chunk(source_file="b.md")]

    metrics = evaluate_retrieval(case, results, k=1)

    assert metrics.required_source_hit_at_k == 1.0
    assert metrics.required_source_recall_at_k == 1.0
    assert metrics.complete_required_source_retrieval_at_k is True
    assert metrics.required_sources_found == ("a.md",)


def test_no_required_source_in_top_k_gives_hit_zero() -> None:
    case = make_golden_case(expected_source_files=("a.md",))
    results = [make_chunk(source_file="b.md"), make_chunk(source_file="c.md")]

    metrics = evaluate_retrieval(case, results, k=2)

    assert metrics.required_source_hit_at_k == 0.0
    assert metrics.required_source_recall_at_k == 0.0
    assert metrics.complete_required_source_retrieval_at_k is False
    assert metrics.required_sources_missing == ("a.md",)


def test_repeated_chunks_from_one_required_source_do_not_inflate_recall() -> None:
    case = make_golden_case(
        requires_multi_document_reasoning=True,
        question_type=QuestionType.MULTI_DOCUMENT_REASONING,
        expected_source_files=("a.md", "b.md"),
    )
    results = [
        make_chunk(source_file="a.md", chunk_id="1"),
        make_chunk(source_file="a.md", chunk_id="2"),
        make_chunk(source_file="c.md", chunk_id="3"),
        make_chunk(source_file="a.md", chunk_id="4"),
        make_chunk(source_file="b.md", chunk_id="5"),
    ]

    metrics = evaluate_retrieval(case, results, k=5)

    # a.md + b.md both present, each counted once -> 2/2
    assert metrics.required_source_recall_at_k == 1.0


def test_two_required_sources_one_found_gives_recall_half_and_incomplete() -> None:
    case = make_golden_case(
        requires_multi_document_reasoning=True,
        question_type=QuestionType.MULTI_DOCUMENT_REASONING,
        expected_source_files=("a.md", "b.md"),
    )
    results = [make_chunk(source_file="a.md"), make_chunk(source_file="z.md")]

    metrics = evaluate_retrieval(case, results, k=5)

    assert metrics.required_source_hit_at_k == 1.0
    assert metrics.required_source_recall_at_k == 0.5
    assert metrics.complete_required_source_retrieval_at_k is False
    assert metrics.required_sources_missing == ("b.md",)


def test_both_multi_doc_required_sources_found_gives_complete_true() -> None:
    case = make_golden_case(
        requires_multi_document_reasoning=True,
        question_type=QuestionType.MULTI_DOCUMENT_REASONING,
        expected_source_files=("a.md", "b.md"),
    )
    results = [make_chunk(source_file="b.md"), make_chunk(source_file="a.md")]

    metrics = evaluate_retrieval(case, results, k=2)

    assert metrics.complete_required_source_retrieval_at_k is True
    assert metrics.required_source_recall_at_k == 1.0


def test_only_one_multi_doc_source_found_gives_complete_false_though_hit_true() -> None:
    case = make_golden_case(
        requires_multi_document_reasoning=True,
        question_type=QuestionType.MULTI_DOCUMENT_REASONING,
        expected_source_files=("a.md", "b.md"),
    )
    results = [make_chunk(source_file="a.md")]

    metrics = evaluate_retrieval(case, results, k=5)

    assert metrics.required_source_hit_at_k == 1.0
    assert metrics.required_source_recall_at_k == 0.5
    assert metrics.complete_required_source_retrieval_at_k is False


def test_reciprocal_rank_is_one_over_first_required_rank() -> None:
    case = make_golden_case(expected_source_files=("a.md",))
    results = [
        make_chunk(source_file="x.md"),
        make_chunk(source_file="y.md"),
        make_chunk(source_file="a.md"),
    ]

    metrics = evaluate_retrieval(case, results, k=5)

    assert metrics.reciprocal_rank == pytest.approx(1.0 / 3.0)


def test_reciprocal_rank_zero_when_no_required_source_anywhere() -> None:
    case = make_golden_case(expected_source_files=("a.md",))
    results = [make_chunk(source_file="x.md"), make_chunk(source_file="y.md")]

    metrics = evaluate_retrieval(case, results, k=5)

    assert metrics.reciprocal_rank == 0.0


def test_expected_identifier_found_in_chunk_text() -> None:
    case = make_golden_case(
        question_type=QuestionType.EXACT_IDENTIFIER,
        expected_identifiers=("ERR_DB_1042",),
    )
    results = [make_chunk(source_file="a.md", text="... ERR_DB_1042 means ...")]

    metrics = evaluate_retrieval(case, results, k=5)

    assert metrics.identifier_recall_at_k == 1.0
    assert metrics.identifiers_found == ("ERR_DB_1042",)


def test_identifier_match_is_case_insensitive() -> None:
    case = make_golden_case(
        question_type=QuestionType.EXACT_IDENTIFIER,
        expected_identifiers=("ERR_DB_1042",),
    )
    results = [make_chunk(source_file="a.md", text="the code err_db_1042 appeared")]

    metrics = evaluate_retrieval(case, results, k=5)

    assert metrics.identifier_recall_at_k == 1.0


def test_repeated_identifier_across_chunks_counts_once() -> None:
    case = make_golden_case(
        question_type=QuestionType.EXACT_IDENTIFIER,
        expected_identifiers=("ERR_DB_1042",),
    )
    results = [
        make_chunk(source_file="a.md", text="ERR_DB_1042", chunk_id="1"),
        make_chunk(source_file="a.md", text="ERR_DB_1042 again", chunk_id="2"),
    ]

    metrics = evaluate_retrieval(case, results, k=5)

    assert metrics.identifier_recall_at_k == 1.0


def test_some_identifiers_missing_gives_partial_recall() -> None:
    case = make_golden_case(
        question_type=QuestionType.EXACT_IDENTIFIER,
        expected_identifiers=("ERR_DB_1042", "AUTH_TOKEN_TTL", "DATABASE_POOL_TIMEOUT"),
    )
    results = [make_chunk(source_file="a.md", text="only ERR_DB_1042 here")]

    metrics = evaluate_retrieval(case, results, k=5)

    assert metrics.identifier_recall_at_k == pytest.approx(1.0 / 3.0)
    assert metrics.identifiers_found == ("ERR_DB_1042",)
    assert metrics.identifiers_missing == ("AUTH_TOKEN_TTL", "DATABASE_POOL_TIMEOUT")


def test_no_expected_identifiers_makes_identifier_recall_not_applicable() -> None:
    case = make_golden_case(expected_identifiers=())
    results = [make_chunk(source_file="authentication-api.md", text="anything")]

    metrics = evaluate_retrieval(case, results, k=5)

    assert metrics.identifier_recall_at_k is None
    assert metrics.identifiers_found == ()
    assert metrics.identifiers_missing == ()


def test_unanswerable_case_has_all_source_and_identifier_metrics_not_applicable() -> None:
    case = make_golden_case(
        id="absent-1",
        answerability=Answerability.UNANSWERABLE,
        question_type=QuestionType.UNANSWERABLE_ABSENT,
        expected_answer=None,
        expected_facts=(),
        expected_source_files=(),
        expected_identifiers=(),
    )
    results = [make_chunk(source_file="a.md", text="ERR_DB_1042")]

    metrics = evaluate_retrieval(case, results, k=5)

    assert metrics.required_source_hit_at_k is None
    assert metrics.required_source_recall_at_k is None
    assert metrics.complete_required_source_retrieval_at_k is None
    assert metrics.reciprocal_rank is None
    assert metrics.identifier_recall_at_k is None


def test_native_retrieval_scores_do_not_affect_metrics() -> None:
    case = make_golden_case(
        question_type=QuestionType.EXACT_IDENTIFIER,
        expected_identifiers=("ERR_DB_1042",),
        expected_source_files=("a.md",),
    )
    high = [
        make_chunk(source_file="z.md", text="noise", similarity=0.99, chunk_id="1"),
        make_chunk(source_file="a.md", text="ERR_DB_1042", similarity=0.98, chunk_id="2"),
    ]
    low_and_sparse = [
        make_chunk(source_file="z.md", text="noise", bm25_score=-42.0, sparse=True, chunk_id="1"),
        make_chunk(
            source_file="a.md", text="ERR_DB_1042", bm25_score=0.001, sparse=True, chunk_id="2"
        ),
    ]

    m_high = evaluate_retrieval(case, high, k=5)
    m_low = evaluate_retrieval(case, low_and_sparse, k=5)

    assert m_high == m_low
    assert m_high.reciprocal_rank == pytest.approx(0.5)
    assert m_high.identifier_recall_at_k == 1.0


def test_k_must_be_a_positive_int() -> None:
    case = make_golden_case()
    results = [make_chunk(source_file="authentication-api.md")]

    for bad in (0, -1, True):
        with pytest.raises(MetricInputError, match="k must be a positive int"):
            evaluate_retrieval(case, results, k=bad)  # type: ignore[arg-type]


def test_hit_at_k_respects_the_cut_off() -> None:
    case = make_golden_case(expected_source_files=("a.md",))
    results = [
        make_chunk(source_file="x.md", chunk_id="1"),
        make_chunk(source_file="y.md", chunk_id="2"),
        make_chunk(source_file="a.md", chunk_id="3"),
    ]

    at_2 = evaluate_retrieval(case, results, k=2)
    at_3 = evaluate_retrieval(case, results, k=3)

    assert at_2.required_source_hit_at_k == 0.0
    assert at_3.required_source_hit_at_k == 1.0
    # reciprocal_rank ranges over the whole sequence, independent of k
    assert at_2.reciprocal_rank == pytest.approx(1.0 / 3.0)
