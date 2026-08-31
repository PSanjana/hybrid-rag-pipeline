"""Tests for rag_pipeline.generation.confidence.score_confidence (pure, no I/O)."""

from __future__ import annotations

import dataclasses
import math

import pytest

from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.generation.confidence import score_confidence
from rag_pipeline.generation.exceptions import (
    CitationValidationError,
    ConfidenceInputError,
    UncitedAnswerError,
)
from rag_pipeline.generation.models import (
    CitationOccurrence,
    CitationVerdict,
    CitationVerification,
    CitationVerificationReport,
    Evidence,
)
from rag_pipeline.generation.verification import verify_grounded_answer

from .conftest import FakeCitationJudge, make_grounded_answer, make_reranked_result

_TWO_RESULTS = [
    make_reranked_result(chunk_id="a", rank=1, text="Access tokens expire after 60 minutes."),
    make_reranked_result(chunk_id="b", rank=2, text="Production access requires MFA."),
]


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def _build(
    answer_text: str,
    reranked_results: list | None = None,
    verdicts_by_occurrence: dict | None = None,
):
    results = (
        reranked_results
        if reranked_results is not None
        else [make_reranked_result(chunk_id="a", rank=1)]
    )
    answer = make_grounded_answer(answer_text=answer_text, reranked_results=results)
    judge = FakeCitationJudge(verdicts_by_occurrence or {})
    report = verify_grounded_answer("q", answer, judge)
    return answer, report, results


# --- verdict component ---------------------------------------------------------------


def test_one_supported_occurrence_gives_citation_score_one() -> None:
    answer, report, results = _build("A [1].", verdicts_by_occurrence={1: (1, "supported", "ok")})
    assessment = score_confidence(answer, report, results, _settings())
    assert assessment.citation_support_score == pytest.approx(1.0)
    assert assessment.supported_count == 1


def test_one_partially_supported_occurrence_gives_citation_score_half() -> None:
    answer, report, results = _build(
        "A [1].", verdicts_by_occurrence={1: (1, "partially_supported", "ok")}
    )
    assessment = score_confidence(answer, report, results, _settings())
    assert assessment.citation_support_score == pytest.approx(0.5)
    assert assessment.partially_supported_count == 1


def test_one_unsupported_occurrence_gives_citation_score_zero() -> None:
    answer, report, results = _build("A [1].", verdicts_by_occurrence={1: (1, "unsupported", "ok")})
    assessment = score_confidence(answer, report, results, _settings())
    assert assessment.citation_support_score == pytest.approx(0.0)
    assert assessment.unsupported_count == 1


def test_one_contradicted_occurrence_gives_citation_score_zero() -> None:
    answer, report, results = _build(
        "A [1].", verdicts_by_occurrence={1: (1, "contradicted", "ok")}
    )
    assessment = score_confidence(answer, report, results, _settings())
    assert assessment.citation_support_score == pytest.approx(0.0)
    assert assessment.contradicted_count == 1


def test_mixed_verdict_average_is_correct() -> None:
    answer, report, results = _build(
        "A [1]. B [2].",
        reranked_results=_TWO_RESULTS,
        verdicts_by_occurrence={1: (1, "supported", "ok"), 2: (2, "unsupported", "ok")},
    )
    assessment = score_confidence(answer, report, results, _settings())
    assert assessment.citation_support_score == pytest.approx(0.5)


def test_contradiction_count_retained_even_though_it_maps_to_zero() -> None:
    answer, report, results = _build(
        "A [1]. B [2].",
        reranked_results=_TWO_RESULTS,
        verdicts_by_occurrence={1: (1, "contradicted", "ok"), 2: (2, "supported", "ok")},
    )
    assessment = score_confidence(answer, report, results, _settings())
    assert assessment.contradicted_count == 1
    assert assessment.citation_support_score == pytest.approx(0.5)


def test_has_contradiction_true_when_a_contradiction_exists() -> None:
    answer, report, results = _build(
        "A [1].", verdicts_by_occurrence={1: (1, "contradicted", "ok")}
    )
    assessment = score_confidence(answer, report, results, _settings())
    assert assessment.has_contradiction is True


def test_has_contradiction_false_when_no_contradiction_exists() -> None:
    answer, report, results = _build("A [1].", verdicts_by_occurrence={1: (1, "supported", "ok")})
    assessment = score_confidence(answer, report, results, _settings())
    assert assessment.has_contradiction is False


# --- retrieval agreement ---------------------------------------------------------------


def test_dual_channel_cited_chunk_gives_agreement_one() -> None:
    results = [make_reranked_result(chunk_id="a", rank=1, dense_rank=1, sparse_rank=1)]
    answer, report, results = _build(
        "A [1].", reranked_results=results, verdicts_by_occurrence={1: (1, "supported", "ok")}
    )
    assessment = score_confidence(answer, report, results, _settings())
    assert assessment.retrieval_agreement_score == pytest.approx(1.0)
    assert assessment.dual_channel_cited_evidence_count == 1
    assert assessment.unique_cited_evidence_count == 1


def test_dense_only_cited_chunk_gives_agreement_zero() -> None:
    results = [make_reranked_result(chunk_id="a", rank=1, dense_rank=1, sparse_rank=None)]
    answer, report, results = _build(
        "A [1].", reranked_results=results, verdicts_by_occurrence={1: (1, "supported", "ok")}
    )
    assessment = score_confidence(answer, report, results, _settings())
    assert assessment.retrieval_agreement_score == pytest.approx(0.0)


def test_sparse_only_cited_chunk_gives_agreement_zero() -> None:
    results = [make_reranked_result(chunk_id="a", rank=1, dense_rank=None, sparse_rank=1)]
    answer, report, results = _build(
        "A [1].", reranked_results=results, verdicts_by_occurrence={1: (1, "supported", "ok")}
    )
    assessment = score_confidence(answer, report, results, _settings())
    assert assessment.retrieval_agreement_score == pytest.approx(0.0)


def test_mixed_dual_and_single_channel_gives_correct_fraction() -> None:
    results = [
        make_reranked_result(chunk_id="a", rank=1, dense_rank=1, sparse_rank=1),
        make_reranked_result(chunk_id="b", rank=2, dense_rank=2, sparse_rank=None),
    ]
    answer, report, results = _build(
        "A [1]. B [2].",
        reranked_results=results,
        verdicts_by_occurrence={1: (1, "supported", "ok"), 2: (2, "supported", "ok")},
    )
    assessment = score_confidence(answer, report, results, _settings())
    assert assessment.unique_cited_evidence_count == 2
    assert assessment.dual_channel_cited_evidence_count == 1
    assert assessment.retrieval_agreement_score == pytest.approx(0.5)


def test_repeated_citation_to_same_evidence_counted_once() -> None:
    results = [make_reranked_result(chunk_id="a", rank=1, dense_rank=1, sparse_rank=1)]
    answer, report, results = _build(
        "A [1]. B [1].",
        reranked_results=results,
        verdicts_by_occurrence={1: (1, "supported", "ok"), 2: (1, "supported", "ok")},
    )
    assessment = score_confidence(answer, report, results, _settings())
    assert assessment.total_citation_occurrences == 2
    assert assessment.unique_cited_evidence_count == 1
    assert assessment.dual_channel_cited_evidence_count == 1
    assert assessment.retrieval_agreement_score == pytest.approx(1.0)


def test_uncited_evidence_does_not_affect_retrieval_agreement() -> None:
    results = [
        make_reranked_result(chunk_id="a", rank=1, dense_rank=1, sparse_rank=1),
        make_reranked_result(chunk_id="b", rank=2, dense_rank=2, sparse_rank=None),
    ]
    # Only chunk "a" (citation [1]) is cited; chunk "b" ([2]) is supplied
    # as evidence but never cited, so it must not affect agreement.
    answer, report, results = _build(
        "A [1].", reranked_results=results, verdicts_by_occurrence={1: (1, "supported", "ok")}
    )
    assessment = score_confidence(answer, report, results, _settings())
    assert assessment.unique_cited_evidence_count == 1
    assert assessment.retrieval_agreement_score == pytest.approx(1.0)


def test_native_retrieval_scores_have_no_impact_when_rank_presence_is_identical() -> None:
    results_low = [
        make_reranked_result(
            chunk_id="a",
            rank=1,
            dense_rank=1,
            sparse_rank=1,
            bm25_score=0.001,
            dense_distance=0.9,
            rrf_score=0.0001,
            reranker_score=-5.0,
        )
    ]
    results_high = [
        make_reranked_result(
            chunk_id="a",
            rank=1,
            dense_rank=1,
            sparse_rank=1,
            bm25_score=500.0,
            dense_distance=0.01,
            rrf_score=0.5,
            reranker_score=99.0,
        )
    ]
    answer_low, report_low, results_low = _build(
        "A [1].", reranked_results=results_low, verdicts_by_occurrence={1: (1, "supported", "ok")}
    )
    answer_high, report_high, results_high = _build(
        "A [1].", reranked_results=results_high, verdicts_by_occurrence={1: (1, "supported", "ok")}
    )
    low = score_confidence(answer_low, report_low, results_low, _settings())
    high = score_confidence(answer_high, report_high, results_high, _settings())
    assert low.retrieval_agreement_score == pytest.approx(high.retrieval_agreement_score)


# --- composite formula ---------------------------------------------------------------


def test_default_weights_are_0_9_and_0_1() -> None:
    settings = _settings()
    assert settings.confidence_citation_weight == pytest.approx(0.9)
    assert settings.confidence_retrieval_agreement_weight == pytest.approx(0.1)


def test_formula_correct_with_default_weights() -> None:
    results = [make_reranked_result(chunk_id="a", rank=1, dense_rank=1, sparse_rank=None)]
    answer, report, results = _build(
        "A [1].",
        reranked_results=results,
        verdicts_by_occurrence={1: (1, "partially_supported", "ok")},
    )
    assessment = score_confidence(answer, report, results, _settings())
    expected = (0.9 * 0.5 + 0.1 * 0.0) / (0.9 + 0.1)
    assert assessment.score == pytest.approx(expected)


def test_custom_weight_ratio_works() -> None:
    results = [make_reranked_result(chunk_id="a", rank=1, dense_rank=1, sparse_rank=1)]
    answer, report, results = _build(
        "A [1].", reranked_results=results, verdicts_by_occurrence={1: (1, "unsupported", "ok")}
    )
    settings = _settings(confidence_citation_weight=0.2, confidence_retrieval_agreement_weight=0.8)
    assessment = score_confidence(answer, report, results, settings)
    expected = (0.2 * 0.0 + 0.8 * 1.0) / (0.2 + 0.8)
    assert assessment.score == pytest.approx(expected)
    assert assessment.citation_weight == pytest.approx(0.2)
    assert assessment.retrieval_agreement_weight == pytest.approx(0.8)


def test_weights_need_not_sum_to_one() -> None:
    settings = _settings(confidence_citation_weight=3.0, confidence_retrieval_agreement_weight=1.0)
    assert (
        settings.confidence_citation_weight + settings.confidence_retrieval_agreement_weight == 4.0
    )


def test_result_is_normalized_by_the_weight_sum() -> None:
    results = [make_reranked_result(chunk_id="a", rank=1, dense_rank=1, sparse_rank=1)]
    answer, report, results = _build(
        "A [1].", reranked_results=results, verdicts_by_occurrence={1: (1, "supported", "ok")}
    )
    settings = _settings(confidence_citation_weight=3.0, confidence_retrieval_agreement_weight=1.0)
    assessment = score_confidence(answer, report, results, settings)
    # Both components are 1.0 here, so regardless of the weight ratio the
    # normalized score must still land at exactly 1.0.
    assert assessment.score == pytest.approx(1.0)


def test_negative_citation_weight_rejected() -> None:
    with pytest.raises(ValueError, match="confidence_citation_weight must not be negative"):
        _settings(confidence_citation_weight=-0.1)


def test_negative_retrieval_agreement_weight_rejected() -> None:
    with pytest.raises(
        ValueError, match="confidence_retrieval_agreement_weight must not be negative"
    ):
        _settings(confidence_retrieval_agreement_weight=-0.1)


def test_both_weights_zero_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _settings(confidence_citation_weight=0.0, confidence_retrieval_agreement_weight=0.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_citation_weight_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="confidence_citation_weight must be finite"):
        _settings(confidence_citation_weight=bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_retrieval_agreement_weight_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="confidence_retrieval_agreement_weight must be finite"):
        _settings(confidence_retrieval_agreement_weight=bad)


@pytest.mark.parametrize(
    ("verdict", "dense_rank", "sparse_rank"),
    [
        ("supported", 1, 1),
        ("supported", 1, None),
        ("partially_supported", None, 1),
        ("unsupported", 1, 1),
        ("contradicted", None, None),
    ],
)
def test_final_score_always_within_zero_one(
    verdict: str, dense_rank: int | None, sparse_rank: int | None
) -> None:
    results = [
        make_reranked_result(chunk_id="a", rank=1, dense_rank=dense_rank, sparse_rank=sparse_rank)
    ]
    answer, report, results = _build(
        "A [1].", reranked_results=results, verdicts_by_occurrence={1: (1, verdict, "ok")}
    )
    assessment = score_confidence(answer, report, results, _settings())
    assert 0.0 <= assessment.score <= 1.0
    assert 0.0 <= assessment.citation_support_score <= 1.0
    assert 0.0 <= assessment.retrieval_agreement_score <= 1.0


# --- insufficient evidence -------------------------------------------------------------


def test_insufficient_evidence_answer_gives_score_zero() -> None:
    answer = make_grounded_answer(
        answer_text=(
            "The supplied documents do not provide enough information to answer this question."
        )
    )
    judge = FakeCitationJudge(error=RuntimeError("judge must never be called"))
    report = verify_grounded_answer("q", answer, judge)
    assessment = score_confidence(answer, report, [], _settings())
    assert assessment.score == pytest.approx(0.0)
    assert judge.calls == []


def test_insufficient_evidence_sets_is_insufficient_evidence_true() -> None:
    answer = make_grounded_answer(
        answer_text=(
            "The supplied documents do not provide enough information to answer this question."
        )
    )
    report = verify_grounded_answer("q", answer, FakeCitationJudge())
    assessment = score_confidence(answer, report, [], _settings())
    assert assessment.is_insufficient_evidence is True


def test_insufficient_evidence_citation_counts_are_zero() -> None:
    answer = make_grounded_answer(
        answer_text=(
            "The supplied documents do not provide enough information to answer this question."
        )
    )
    report = verify_grounded_answer("q", answer, FakeCitationJudge())
    assessment = score_confidence(answer, report, [], _settings())
    assert assessment.supported_count == 0
    assert assessment.partially_supported_count == 0
    assert assessment.unsupported_count == 0
    assert assessment.contradicted_count == 0
    assert assessment.total_citation_occurrences == 0
    assert assessment.unique_cited_evidence_count == 0


def test_insufficient_evidence_retrieval_agreement_is_zero() -> None:
    answer = make_grounded_answer(
        answer_text=(
            "The supplied documents do not provide enough information to answer this question."
        )
    )
    report = verify_grounded_answer("q", answer, FakeCitationJudge())
    assessment = score_confidence(answer, report, [], _settings())
    assert assessment.retrieval_agreement_score == pytest.approx(0.0)
    assert assessment.dual_channel_cited_evidence_count == 0


def test_vacuous_all_supported_is_not_treated_as_confidence() -> None:
    answer = make_grounded_answer(
        answer_text=(
            "The supplied documents do not provide enough information to answer this question."
        )
    )
    report = verify_grounded_answer("q", answer, FakeCitationJudge())
    assert report.all_supported is True  # vacuously true for zero occurrences
    assessment = score_confidence(answer, report, [], _settings())
    assert assessment.score == pytest.approx(0.0)  # NOT high confidence


# --- integrity ---------------------------------------------------------------------


def test_normal_zero_citation_answer_rejected() -> None:
    answer = make_grounded_answer(answer_text="A substantive claim with no citations at all.")
    report = CitationVerificationReport(grounded_answer=answer, occurrences=(), verifications=())
    with pytest.raises(UncitedAnswerError):
        score_confidence(answer, report, [make_reranked_result(chunk_id="a", rank=1)], _settings())


def test_occurrence_verification_count_mismatch_rejected() -> None:
    answer = make_grounded_answer(
        answer_text="A [1]. B [1].", reranked_results=[make_reranked_result(chunk_id="a", rank=1)]
    )
    # Offsets must match the deterministic parse of "A [1]. B [1]." exactly
    # so the answer/report binding passes and the count check downstream is
    # what actually rejects the report.
    occurrences = (
        CitationOccurrence(occurrence_id=1, citation_number=1, start_offset=2, end_offset=5),
        CitationOccurrence(occurrence_id=2, citation_number=1, start_offset=9, end_offset=12),
    )
    verifications = (
        CitationVerification(
            occurrence_id=1,
            citation_number=1,
            verdict=CitationVerdict.SUPPORTED,
            rationale="ok",
            chunk_id="a",
        ),
    )
    report = CitationVerificationReport(
        grounded_answer=answer, occurrences=occurrences, verifications=verifications
    )
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, [make_reranked_result(chunk_id="a", rank=1)], _settings())


def test_occurrence_id_mismatch_rejected() -> None:
    answer = make_grounded_answer(
        answer_text="A [1].", reranked_results=[make_reranked_result(chunk_id="a", rank=1)]
    )
    occurrences = (
        CitationOccurrence(occurrence_id=1, citation_number=1, start_offset=2, end_offset=5),
    )
    verifications = (
        CitationVerification(
            occurrence_id=2,
            citation_number=1,
            verdict=CitationVerdict.SUPPORTED,
            rationale="ok",
            chunk_id="a",
        ),
    )
    report = CitationVerificationReport(
        grounded_answer=answer, occurrences=occurrences, verifications=verifications
    )
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, [make_reranked_result(chunk_id="a", rank=1)], _settings())


def test_citation_number_mismatch_between_occurrence_and_verification_rejected() -> None:
    answer = make_grounded_answer(answer_text="A [1]. B [2].", reranked_results=_TWO_RESULTS)
    # Offsets match the real parse of "A [1]. B [2]." so binding passes and
    # the occurrence/verification citation-number check is what rejects it.
    occurrences = (
        CitationOccurrence(occurrence_id=1, citation_number=1, start_offset=2, end_offset=5),
        CitationOccurrence(occurrence_id=2, citation_number=2, start_offset=9, end_offset=12),
    )
    verifications = (
        CitationVerification(
            occurrence_id=1,
            citation_number=2,  # wrong: should be 1
            verdict=CitationVerdict.SUPPORTED,
            rationale="ok",
            chunk_id="a",
        ),
        CitationVerification(
            occurrence_id=2,
            citation_number=2,
            verdict=CitationVerdict.SUPPORTED,
            rationale="ok",
            chunk_id="b",
        ),
    )
    report = CitationVerificationReport(
        grounded_answer=answer, occurrences=occurrences, verifications=verifications
    )
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, _TWO_RESULTS, _settings())


def test_report_with_correct_occurrences_but_zero_verifications_rejected() -> None:
    # Occurrences exactly match the answer text (binding passes), but the
    # report carries no verifications at all -- a cited answer with nothing
    # verified must not be scored.
    answer = make_grounded_answer(
        answer_text="A [1].", reranked_results=[make_reranked_result(chunk_id="a", rank=1)]
    )
    report = CitationVerificationReport(
        grounded_answer=answer,
        occurrences=(
            CitationOccurrence(occurrence_id=1, citation_number=1, start_offset=2, end_offset=5),
        ),
        verifications=(),
    )
    with pytest.raises(UncitedAnswerError):
        score_confidence(answer, report, [make_reranked_result(chunk_id="a", rank=1)], _settings())


def test_verification_chunk_id_evidence_mismatch_rejected() -> None:
    answer = make_grounded_answer(
        answer_text="A [1].", reranked_results=[make_reranked_result(chunk_id="a", rank=1)]
    )
    occurrences = (
        CitationOccurrence(occurrence_id=1, citation_number=1, start_offset=2, end_offset=5),
    )
    verifications = (
        CitationVerification(
            occurrence_id=1,
            citation_number=1,
            verdict=CitationVerdict.SUPPORTED,
            rationale="ok",
            chunk_id="not-the-real-chunk-id",
        ),
    )
    report = CitationVerificationReport(
        grounded_answer=answer, occurrences=occurrences, verifications=verifications
    )
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, [make_reranked_result(chunk_id="a", rank=1)], _settings())


def test_duplicate_retrieval_chunk_ids_rejected() -> None:
    answer, report, _results = _build("A [1].", verdicts_by_occurrence={1: (1, "supported", "ok")})
    duplicated_results = [
        make_reranked_result(chunk_id="a", rank=1),
        make_reranked_result(chunk_id="a", rank=2),
    ]
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, duplicated_results, _settings())


def test_cited_chunk_missing_from_reranked_results_rejected() -> None:
    answer, report, _results = _build("A [1].", verdicts_by_occurrence={1: (1, "supported", "ok")})
    with pytest.raises(ConfidenceInputError):
        score_confidence(
            answer,
            report,
            [make_reranked_result(chunk_id="totally-different", rank=1)],
            _settings(),
        )


def test_malformed_evidence_numbering_rejected_via_existing_validation() -> None:
    malformed_evidence = (
        Evidence(
            citation_number=1,
            chunk_id="a",
            text="text",
            source_file="doc.md",
            document_id="d" * 64,
            chunk_index=0,
            section_heading=None,
            page_number=None,
            chunking_strategy=ChunkingStrategy.RECURSIVE,
            reranked_rank=1,
        ),
        Evidence(
            citation_number=1,  # duplicate citation_number -- malformed
            chunk_id="b",
            text="text",
            source_file="doc.md",
            document_id="d" * 64,
            chunk_index=0,
            section_heading=None,
            page_number=None,
            chunking_strategy=ChunkingStrategy.RECURSIVE,
            reranked_rank=2,
        ),
    )
    answer = make_grounded_answer(answer_text="A [1].")
    # Swap in malformed evidence directly (bypassing build_evidence()).
    answer = dataclasses.replace(answer, evidence=malformed_evidence)
    occurrences = (
        CitationOccurrence(occurrence_id=1, citation_number=1, start_offset=2, end_offset=5),
    )
    verifications = (
        CitationVerification(
            occurrence_id=1,
            citation_number=1,
            verdict=CitationVerdict.SUPPORTED,
            rationale="ok",
            chunk_id="a",
        ),
    )
    report = CitationVerificationReport(
        grounded_answer=answer, occurrences=occurrences, verifications=verifications
    )
    with pytest.raises(CitationValidationError):
        score_confidence(
            answer,
            report,
            [
                make_reranked_result(chunk_id="a", rank=1),
                make_reranked_result(chunk_id="b", rank=2),
            ],
            _settings(),
        )


def test_report_built_against_a_different_grounded_answer_rejected() -> None:
    answer_a, report_a, results = _build(
        "A [1].", verdicts_by_occurrence={1: (1, "supported", "ok")}
    )
    # A structurally-valid report, but for a different answer object.
    answer_b = make_grounded_answer(
        answer_text="A different [1].",
        reranked_results=[make_reranked_result(chunk_id="a", rank=1)],
    )
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer_b, report_a, results, _settings())


def test_duplicate_occurrence_id_rejected() -> None:
    # A report whose occurrences carry a duplicate occurrence_id can never
    # match the deterministic (contiguous, unique-id) parse of the answer
    # text, so answer/report binding rejects it -- still a ConfidenceInputError.
    answer = make_grounded_answer(
        answer_text="A [1]. B [1].", reranked_results=[make_reranked_result(chunk_id="a", rank=1)]
    )
    occurrences = (
        CitationOccurrence(occurrence_id=1, citation_number=1, start_offset=2, end_offset=5),
        CitationOccurrence(occurrence_id=1, citation_number=1, start_offset=9, end_offset=12),
    )
    verifications = (
        CitationVerification(
            occurrence_id=1,
            citation_number=1,
            verdict=CitationVerdict.SUPPORTED,
            rationale="ok",
            chunk_id="a",
        ),
        CitationVerification(
            occurrence_id=1,
            citation_number=1,
            verdict=CitationVerdict.SUPPORTED,
            rationale="ok",
            chunk_id="a",
        ),
    )
    report = CitationVerificationReport(
        grounded_answer=answer, occurrences=occurrences, verifications=verifications
    )
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, [make_reranked_result(chunk_id="a", rank=1)], _settings())


def test_duplicate_verification_occurrence_id_rejected() -> None:
    answer = make_grounded_answer(
        answer_text="A [1]. B [1].", reranked_results=[make_reranked_result(chunk_id="a", rank=1)]
    )
    # Occurrences match the real parse of "A [1]. B [1]." (binding passes);
    # the duplicate lives only in the verifications.
    occurrences = (
        CitationOccurrence(occurrence_id=1, citation_number=1, start_offset=2, end_offset=5),
        CitationOccurrence(occurrence_id=2, citation_number=1, start_offset=9, end_offset=12),
    )
    # Two verifications, both claiming occurrence_id=1.
    verifications = tuple(
        CitationVerification(
            occurrence_id=1,
            citation_number=1,
            verdict=CitationVerdict.SUPPORTED,
            rationale="ok",
            chunk_id="a",
        )
        for _ in range(2)
    )
    report = CitationVerificationReport(
        grounded_answer=answer, occurrences=occurrences, verifications=verifications
    )
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, [make_reranked_result(chunk_id="a", rank=1)], _settings())


def test_occurrence_citation_number_outside_supplied_evidence_range_rejected() -> None:
    # The answer text genuinely cites [2] (so answer/report binding passes),
    # but only a single evidence item ([1]) was supplied -- exercising the
    # `_validate_report_integrity` "citation number not in evidence range"
    # branch specifically, not the earlier answer-binding check.
    answer = make_grounded_answer(
        answer_text="A [2].", reranked_results=[make_reranked_result(chunk_id="a", rank=1)]
    )
    occurrences = (
        CitationOccurrence(occurrence_id=1, citation_number=2, start_offset=2, end_offset=5),
    )
    verifications = (
        CitationVerification(
            occurrence_id=1,
            citation_number=2,
            verdict=CitationVerdict.SUPPORTED,
            rationale="ok",
            chunk_id="a",
        ),
    )
    report = CitationVerificationReport(
        grounded_answer=answer, occurrences=occurrences, verifications=verifications
    )
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, [make_reranked_result(chunk_id="a", rank=1)], _settings())


def test_empty_chunk_id_in_reranked_results_rejected() -> None:
    answer, report, _results = _build("A [1].", verdicts_by_occurrence={1: (1, "supported", "ok")})
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, [make_reranked_result(chunk_id="", rank=1)], _settings())


# --- answer/report binding -------------------------------------------------------------


def _hand_report(
    answer: object,
    occurrences: tuple[CitationOccurrence, ...],
    verifications: tuple[CitationVerification, ...],
) -> CitationVerificationReport:
    return CitationVerificationReport(
        grounded_answer=answer,  # type: ignore[arg-type]
        occurrences=occurrences,
        verifications=verifications,
    )


def _supported(occurrence_id: int, citation_number: int, chunk_id: str) -> CitationVerification:
    return CitationVerification(
        occurrence_id=occurrence_id,
        citation_number=citation_number,
        verdict=CitationVerdict.SUPPORTED,
        rationale="ok",
        chunk_id=chunk_id,
    )


def test_binding_rejects_report_occurrence_citing_a_different_number_than_the_answer() -> None:
    # Answer text actually cites [2]; the report claims the occurrence is [1].
    answer = make_grounded_answer(answer_text="A [2].", reranked_results=_TWO_RESULTS)
    report = _hand_report(
        answer,
        (CitationOccurrence(occurrence_id=1, citation_number=1, start_offset=2, end_offset=5),),
        (_supported(1, 1, "a"),),
    )
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, _TWO_RESULTS, _settings())


def test_binding_rejects_altered_occurrence_offsets() -> None:
    # IDs and citation numbers are right, but the offsets don't match the
    # deterministic parse of "A [1]." (which is start=2, end=5).
    answer = make_grounded_answer(
        answer_text="A [1].", reranked_results=[make_reranked_result(chunk_id="a", rank=1)]
    )
    report = _hand_report(
        answer,
        (CitationOccurrence(occurrence_id=1, citation_number=1, start_offset=0, end_offset=3),),
        (_supported(1, 1, "a"),),
    )
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, [make_reranked_result(chunk_id="a", rank=1)], _settings())


def test_binding_rejects_report_that_omits_an_actual_occurrence() -> None:
    answer = make_grounded_answer(answer_text="A [1]. B [2].", reranked_results=_TWO_RESULTS)
    # Real answer has two occurrences; report only carries the first.
    report = _hand_report(
        answer,
        (CitationOccurrence(occurrence_id=1, citation_number=1, start_offset=2, end_offset=5),),
        (_supported(1, 1, "a"),),
    )
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, _TWO_RESULTS, _settings())


def test_binding_rejects_report_that_invents_an_occurrence_absent_from_the_answer() -> None:
    answer = make_grounded_answer(
        answer_text="A [1].", reranked_results=[make_reranked_result(chunk_id="a", rank=1)]
    )
    report = _hand_report(
        answer,
        (
            CitationOccurrence(occurrence_id=1, citation_number=1, start_offset=2, end_offset=5),
            CitationOccurrence(occurrence_id=2, citation_number=1, start_offset=6, end_offset=9),
        ),
        (_supported(1, 1, "a"), _supported(2, 1, "a")),
    )
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, [make_reranked_result(chunk_id="a", rank=1)], _settings())


def test_binding_rejects_cited_numbers_disagreeing_with_actual_answer_citations() -> None:
    answer = make_grounded_answer(
        answer_text="A [1].", reranked_results=[make_reranked_result(chunk_id="a", rank=1)]
    )
    # answer_text yields cited numbers (1,), but the model object claims (1, 2).
    answer = dataclasses.replace(answer, cited_numbers=(1, 2))
    report = _hand_report(
        answer,
        (CitationOccurrence(occurrence_id=1, citation_number=1, start_offset=2, end_offset=5),),
        (_supported(1, 1, "a"),),
    )
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, [make_reranked_result(chunk_id="a", rank=1)], _settings())


def test_binding_allows_valid_repeated_citations() -> None:
    results = [make_reranked_result(chunk_id="a", rank=1, dense_rank=1, sparse_rank=1)]
    answer, report, results = _build(
        "A [1]. B [1].",
        reranked_results=results,
        verdicts_by_occurrence={1: (1, "supported", "ok"), 2: (1, "supported", "ok")},
    )
    assessment = score_confidence(answer, report, results, _settings())
    assert assessment.total_citation_occurrences == 2
    assert assessment.unique_cited_evidence_count == 1
    assert assessment.citation_support_score == pytest.approx(1.0)


# --- insufficient-evidence short-circuit hardening ------------------------------------


def test_insufficiency_phrase_with_citations_is_scored_as_a_normal_answer() -> None:
    text = (
        "The supplied documents do not provide enough information on exact timing, but the "
        "token limit itself is documented [1]."
    )
    results = [make_reranked_result(chunk_id="a", rank=1, dense_rank=1, sparse_rank=1)]
    answer, report, results = _build(
        text, reranked_results=results, verdicts_by_occurrence={1: (1, "supported", "ok")}
    )
    assessment = score_confidence(answer, report, results, _settings())
    assert assessment.is_insufficient_evidence is False
    assert assessment.total_citation_occurrences == 1
    assert assessment.citation_support_score == pytest.approx(1.0)
    assert assessment.score > 0.0


def test_zero_citation_insufficiency_answer_with_non_empty_report_rejected() -> None:
    answer = make_grounded_answer(
        answer_text=(
            "The supplied documents do not provide enough information to answer this question."
        ),
        reranked_results=[make_reranked_result(chunk_id="a", rank=1)],
    )
    # occurrences correctly empty (matches the answer text), but verifications
    # are not -- a malformed report that must not be silently ignored.
    report = _hand_report(answer, (), (_supported(1, 1, "a"),))
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, [make_reranked_result(chunk_id="a", rank=1)], _settings())


# --- runtime field-type hardening ----------------------------------------------------


def test_occurrence_id_true_rejected() -> None:
    answer = make_grounded_answer(
        answer_text="A [1].", reranked_results=[make_reranked_result(chunk_id="a", rank=1)]
    )
    report = _hand_report(
        answer,
        (
            CitationOccurrence(
                occurrence_id=True,  # type: ignore[arg-type]
                citation_number=1,
                start_offset=2,
                end_offset=5,
            ),
        ),
        (_supported(1, 1, "a"),),
    )
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, [make_reranked_result(chunk_id="a", rank=1)], _settings())


def test_occurrence_citation_number_true_rejected() -> None:
    answer = make_grounded_answer(
        answer_text="A [1].", reranked_results=[make_reranked_result(chunk_id="a", rank=1)]
    )
    report = _hand_report(
        answer,
        (
            CitationOccurrence(
                occurrence_id=1,
                citation_number=True,  # type: ignore[arg-type]
                start_offset=2,
                end_offset=5,
            ),
        ),
        (_supported(1, 1, "a"),),
    )
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, [make_reranked_result(chunk_id="a", rank=1)], _settings())


def test_verification_occurrence_id_true_rejected() -> None:
    answer = make_grounded_answer(
        answer_text="A [1].", reranked_results=[make_reranked_result(chunk_id="a", rank=1)]
    )
    report = _hand_report(
        answer,
        (CitationOccurrence(occurrence_id=1, citation_number=1, start_offset=2, end_offset=5),),
        (
            CitationVerification(
                occurrence_id=True,  # type: ignore[arg-type]
                citation_number=1,
                verdict=CitationVerdict.SUPPORTED,
                rationale="ok",
                chunk_id="a",
            ),
        ),
    )
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, [make_reranked_result(chunk_id="a", rank=1)], _settings())


def test_verification_verdict_arbitrary_string_rejected() -> None:
    answer = make_grounded_answer(
        answer_text="A [1].", reranked_results=[make_reranked_result(chunk_id="a", rank=1)]
    )
    report = _hand_report(
        answer,
        (CitationOccurrence(occurrence_id=1, citation_number=1, start_offset=2, end_offset=5),),
        (
            CitationVerification(
                occurrence_id=1,
                citation_number=1,
                verdict="totally-bogus-verdict",  # type: ignore[arg-type]
                rationale="ok",
                chunk_id="a",
            ),
        ),
    )
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, [make_reranked_result(chunk_id="a", rank=1)], _settings())


def test_verification_empty_chunk_id_rejected() -> None:
    answer = make_grounded_answer(
        answer_text="A [1].", reranked_results=[make_reranked_result(chunk_id="a", rank=1)]
    )
    report = _hand_report(
        answer,
        (CitationOccurrence(occurrence_id=1, citation_number=1, start_offset=2, end_offset=5),),
        (_supported(1, 1, ""),),
    )
    with pytest.raises(ConfidenceInputError):
        score_confidence(answer, report, [make_reranked_result(chunk_id="a", rank=1)], _settings())


def test_malformed_report_fields_do_not_leak_raw_key_or_type_error() -> None:
    answer = make_grounded_answer(
        answer_text="A [1].", reranked_results=[make_reranked_result(chunk_id="a", rank=1)]
    )
    report = _hand_report(
        answer,
        (CitationOccurrence(occurrence_id=1, citation_number=1, start_offset=2, end_offset=5),),
        (
            CitationVerification(
                occurrence_id=1,
                citation_number=1,
                verdict="nonsense",  # type: ignore[arg-type]
                rationale="ok",
                chunk_id="a",
            ),
        ),
    )
    with pytest.raises(ConfidenceInputError):
        try:
            score_confidence(
                answer, report, [make_reranked_result(chunk_id="a", rank=1)], _settings()
            )
        except (KeyError, TypeError, AttributeError) as exc:  # pragma: no cover - must not happen
            raise AssertionError(f"leaked a raw {type(exc).__name__}: {exc!r}") from exc
