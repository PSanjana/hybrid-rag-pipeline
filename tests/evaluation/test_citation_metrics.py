"""Offline tests for rag_pipeline.evaluation.metrics.citations (deterministic, no LLM)."""

from __future__ import annotations

import pytest

from rag_pipeline.evaluation.exceptions import MetricInputError
from rag_pipeline.evaluation.metrics.citations import evaluate_citation_accuracy
from rag_pipeline.evaluation.models import Answerability, QuestionType
from rag_pipeline.generation.models import (
    AnswerDecision,
    CitationVerdict,
    CitationVerificationReport,
)

from .conftest import (
    make_final_answer,
    make_golden_case,
    make_grounded_answer,
    make_verification_report,
)

_SUP = CitationVerdict.SUPPORTED
_PARTIAL = CitationVerdict.PARTIALLY_SUPPORTED
_UNSUP = CitationVerdict.UNSUPPORTED


def _final(*, answer_text: str, sources: list[str], verdicts: list[CitationVerdict], case_kw=None):
    grounded = make_grounded_answer(answer_text=answer_text, sources=sources)
    report = make_verification_report(grounded, verdicts)
    case = make_golden_case(**(case_kw or {}))
    final = make_final_answer(decision=AnswerDecision.ANSWERED, grounded=grounded, report=report)
    return case, final


def test_fully_supported_citations_give_support_score_one() -> None:
    case, final = _final(
        answer_text="A [1]. B [2].",
        sources=["authentication-api.md", "authentication-api.md"],
        verdicts=[_SUP, _SUP],
        case_kw={"expected_source_files": ("authentication-api.md",)},
    )

    metrics = evaluate_citation_accuracy(case=case, final_answer=final)

    assert metrics.applicable is True
    assert metrics.semantic_citation_support_score == 1.0
    assert metrics.fully_supported_citation_rate == 1.0


def test_mixed_verdicts_give_correct_support_score() -> None:
    case, final = _final(
        answer_text="A [1]. B [2]. C [3].",
        sources=["authentication-api.md"] * 3,
        verdicts=[_SUP, _PARTIAL, _UNSUP],
        case_kw={"expected_source_files": ("authentication-api.md",)},
    )

    metrics = evaluate_citation_accuracy(case=case, final_answer=final)

    assert metrics.semantic_citation_support_score == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_fully_supported_rate_counts_only_supported_over_total() -> None:
    case, final = _final(
        answer_text="A [1]. B [2]. C [3]. D [4].",
        sources=["authentication-api.md"] * 4,
        verdicts=[_SUP, _SUP, _PARTIAL, _UNSUP],
        case_kw={"expected_source_files": ("authentication-api.md",)},
    )

    metrics = evaluate_citation_accuracy(case=case, final_answer=final)

    assert metrics.fully_supported_citation_rate == pytest.approx(0.5)
    assert metrics.total_citation_occurrences == 4
    assert metrics.supported_count == 2


def test_repeated_citations_to_one_source_are_deduplicated_for_source_metrics() -> None:
    case, final = _final(
        answer_text="A [1]. B [1]. C [1].",
        sources=["authentication-api.md"],
        verdicts=[_SUP, _SUP, _SUP],
        case_kw={"expected_source_files": ("authentication-api.md",)},
    )

    metrics = evaluate_citation_accuracy(case=case, final_answer=final)

    assert metrics.unique_cited_source_files == ("authentication-api.md",)
    assert metrics.cited_source_golden_match_rate == 1.0
    assert metrics.required_source_citation_recall == 1.0


def test_expected_source_cited_counts_as_a_golden_match() -> None:
    case, final = _final(
        answer_text="A [1].",
        sources=["authentication-api.md"],
        verdicts=[_SUP],
        case_kw={"expected_source_files": ("authentication-api.md",)},
    )

    metrics = evaluate_citation_accuracy(case=case, final_answer=final)

    assert metrics.matched_cited_source_files == ("authentication-api.md",)
    assert metrics.cited_source_golden_match_rate == 1.0


def test_acceptable_source_cited_also_counts_as_a_golden_match() -> None:
    case, final = _final(
        answer_text="A [1].",
        sources=["production-runbook.txt"],
        verdicts=[_SUP],
        case_kw={
            "expected_source_files": ("authentication-api.md",),
            "acceptable_source_files": ("production-runbook.txt",),
        },
    )

    metrics = evaluate_citation_accuracy(case=case, final_answer=final)

    assert metrics.matched_cited_source_files == ("production-runbook.txt",)
    assert metrics.cited_source_golden_match_rate == 1.0
    # required source was NOT cited, so required-source recall is 0.0 (applicable)
    assert metrics.required_source_citation_recall == 0.0
    assert metrics.required_sources_not_cited == ("authentication-api.md",)


def test_unmatched_source_lowers_golden_source_match_rate() -> None:
    case, final = _final(
        answer_text="A [1]. B [2].",
        sources=["authentication-api.md", "unrelated-doc.md"],
        verdicts=[_SUP, _SUP],
        case_kw={"expected_source_files": ("authentication-api.md",)},
    )

    metrics = evaluate_citation_accuracy(case=case, final_answer=final)

    assert metrics.cited_source_golden_match_rate == pytest.approx(0.5)
    assert metrics.unmatched_cited_source_files == ("unrelated-doc.md",)


def test_multi_document_required_citation_recall_partial() -> None:
    case, final = _final(
        answer_text="A [1]. B [2].",
        sources=["api-error-codes.txt", "unrelated-doc.md"],
        verdicts=[_SUP, _SUP],
        case_kw={
            "requires_multi_document_reasoning": True,
            "question_type": QuestionType.MULTI_DOCUMENT_REASONING,
            "expected_source_files": ("api-error-codes.txt", "database-operations.md"),
        },
    )

    metrics = evaluate_citation_accuracy(case=case, final_answer=final)

    assert metrics.required_source_citation_recall == pytest.approx(0.5)
    assert metrics.required_sources_cited == ("api-error-codes.txt",)
    assert metrics.required_sources_not_cited == ("database-operations.md",)


def test_multi_document_required_citation_recall_complete() -> None:
    case, final = _final(
        answer_text="A [1]. B [2].",
        sources=["api-error-codes.txt", "database-operations.md"],
        verdicts=[_SUP, _SUP],
        case_kw={
            "requires_multi_document_reasoning": True,
            "question_type": QuestionType.MULTI_DOCUMENT_REASONING,
            "expected_source_files": ("api-error-codes.txt", "database-operations.md"),
        },
    )

    metrics = evaluate_citation_accuracy(case=case, final_answer=final)

    assert metrics.required_source_citation_recall == 1.0
    assert metrics.required_sources_not_cited == ()


def test_abstained_answer_makes_citation_metrics_not_applicable() -> None:
    grounded = make_grounded_answer(answer_text="draft [1].", sources=["authentication-api.md"])
    report = make_verification_report(grounded, [_UNSUP])
    final = make_final_answer(
        decision=AnswerDecision.ABSTAINED_UNSUPPORTED_CITATION,
        grounded=grounded,
        report=report,
        abstention_reason="reason",
    )

    metrics = evaluate_citation_accuracy(case=make_golden_case(), final_answer=final)

    assert metrics.applicable is False
    assert metrics.semantic_citation_support_score is None
    assert metrics.cited_source_golden_match_rate is None
    assert "abstained" in (metrics.not_applicable_reason or "")


def test_answered_unanswerable_case_has_required_recall_na_and_zero_golden_match() -> None:
    case = make_golden_case(
        id="absent-1",
        answerability=Answerability.UNANSWERABLE,
        question_type=QuestionType.UNANSWERABLE_ABSENT,
        expected_answer=None,
        expected_facts=(),
        expected_source_files=(),
    )
    grounded = make_grounded_answer(answer_text="wrongly answered [1].", sources=["some-doc.md"])
    report = make_verification_report(grounded, [_SUP])
    final = make_final_answer(decision=AnswerDecision.ANSWERED, grounded=grounded, report=report)

    metrics = evaluate_citation_accuracy(case=case, final_answer=final)

    assert metrics.applicable is True
    # no required-source truth for an unanswerable case -> N/A, not 0.0
    assert metrics.required_source_citation_recall is None
    # nothing is a golden-legitimate source -> applicable and zero
    assert metrics.cited_source_golden_match_rate == 0.0
    assert metrics.unmatched_cited_source_files == ("some-doc.md",)


def test_malformed_answered_with_zero_citation_occurrences_rejected() -> None:
    grounded = make_grounded_answer(
        answer_text="no citations here", sources=["authentication-api.md"]
    )
    empty_report = CitationVerificationReport(
        grounded_answer=grounded, occurrences=(), verifications=()
    )
    final = make_final_answer(
        decision=AnswerDecision.ANSWERED, grounded=grounded, report=empty_report
    )

    with pytest.raises(MetricInputError, match="zero citation occurrences"):
        evaluate_citation_accuracy(case=make_golden_case(), final_answer=final)
