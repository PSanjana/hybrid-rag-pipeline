"""Tests for CitationVerdict/CitationVerification/CitationVerificationReport (models only)."""

from __future__ import annotations

from rag_pipeline.generation.models import (
    CitationVerdict,
    CitationVerification,
    CitationVerificationReport,
)

from .conftest import make_grounded_answer


def _verification(
    occurrence_id: int, citation_number: int, verdict: CitationVerdict
) -> CitationVerification:
    return CitationVerification(
        occurrence_id=occurrence_id,
        citation_number=citation_number,
        verdict=verdict,
        rationale="because the evidence says so",
        chunk_id="chunk-1",
    )


def test_supported_verdict_accepted() -> None:
    v = _verification(1, 1, CitationVerdict.SUPPORTED)
    assert v.verdict == CitationVerdict.SUPPORTED


def test_partially_supported_verdict_accepted() -> None:
    v = _verification(1, 1, CitationVerdict.PARTIALLY_SUPPORTED)
    assert v.verdict == CitationVerdict.PARTIALLY_SUPPORTED


def test_unsupported_verdict_accepted() -> None:
    v = _verification(1, 1, CitationVerdict.UNSUPPORTED)
    assert v.verdict == CitationVerdict.UNSUPPORTED


def test_contradicted_verdict_accepted() -> None:
    v = _verification(1, 1, CitationVerdict.CONTRADICTED)
    assert v.verdict == CitationVerdict.CONTRADICTED


def test_exactly_four_verdict_values_exist() -> None:
    assert {v.value for v in CitationVerdict} == {
        "supported",
        "partially_supported",
        "unsupported",
        "contradicted",
    }


def _report(verdicts: list[CitationVerdict]) -> CitationVerificationReport:
    grounded_answer = make_grounded_answer(answer_text="A [1]. B [1]. C [1].")
    verifications = tuple(
        _verification(i, 1, verdict) for i, verdict in enumerate(verdicts, start=1)
    )
    occurrences = ()  # not exercised by these count/aggregate tests
    return CitationVerificationReport(
        grounded_answer=grounded_answer, occurrences=occurrences, verifications=verifications
    )


def test_derived_counts_are_correct() -> None:
    report = _report(
        [
            CitationVerdict.SUPPORTED,
            CitationVerdict.SUPPORTED,
            CitationVerdict.PARTIALLY_SUPPORTED,
            CitationVerdict.UNSUPPORTED,
            CitationVerdict.CONTRADICTED,
        ]
    )
    assert report.total_occurrences == 5
    assert report.supported_count == 2
    assert report.partially_supported_count == 1
    assert report.unsupported_count == 1
    assert report.contradicted_count == 1


def test_all_supported_true_only_when_every_occurrence_is_supported() -> None:
    all_supported_report = _report([CitationVerdict.SUPPORTED, CitationVerdict.SUPPORTED])
    assert all_supported_report.all_supported is True

    mixed_report = _report([CitationVerdict.SUPPORTED, CitationVerdict.PARTIALLY_SUPPORTED])
    assert mixed_report.all_supported is False


def test_all_supported_is_vacuously_true_for_zero_occurrences() -> None:
    report = _report([])
    assert report.all_supported is True
    assert report.total_occurrences == 0


def test_repeated_citation_number_can_receive_different_verdicts_at_different_occurrences() -> None:
    grounded_answer = make_grounded_answer(answer_text="A [1]. B [1].")
    verifications = (
        _verification(1, 1, CitationVerdict.SUPPORTED),
        _verification(2, 1, CitationVerdict.CONTRADICTED),
    )
    report = CitationVerificationReport(
        grounded_answer=grounded_answer, occurrences=(), verifications=verifications
    )
    assert report.verifications[0].citation_number == report.verifications[1].citation_number == 1
    assert report.verifications[0].verdict != report.verifications[1].verdict
    assert report.supported_count == 1
    assert report.contradicted_count == 1
    assert report.all_supported is False
