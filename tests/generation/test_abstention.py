"""Tests for rag_pipeline.generation.abstention.apply_abstention_policy (pure, no I/O)."""

from __future__ import annotations

import dataclasses
import math

import pytest

from rag_pipeline.config import Settings
from rag_pipeline.generation.abstention import (
    ABSTENTION_TEXT,
    apply_abstention_policy,
)
from rag_pipeline.generation.confidence import score_confidence
from rag_pipeline.generation.exceptions import AbstentionPolicyInputError
from rag_pipeline.generation.models import (
    AnswerDecision,
    CitationVerificationReport,
    ConfidenceAssessment,
    FinalAnswer,
    GroundedAnswer,
)
from rag_pipeline.generation.verification import verify_grounded_answer

from .conftest import FakeCitationJudge, make_grounded_answer, make_reranked_result

_INSUFFICIENT_TEXT = (
    "The supplied documents do not provide enough information to answer this question."
)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def _dual(chunk_id: str, rank: int) -> object:
    return make_reranked_result(chunk_id=chunk_id, rank=rank, dense_rank=rank, sparse_rank=rank)


def _trio(
    answer_text: str,
    verdicts_by_occurrence: dict[int, tuple[int, str, str]],
    *,
    reranked_results: list | None = None,
    settings: Settings | None = None,
) -> tuple[GroundedAnswer, CitationVerificationReport, ConfidenceAssessment]:
    """Build a genuine (answer, report, confidence) trio through the real Step 2/3 code."""
    results = reranked_results if reranked_results is not None else [_dual("a", 1)]
    answer = make_grounded_answer(answer_text=answer_text, reranked_results=results)
    report = verify_grounded_answer("q", answer, FakeCitationJudge(verdicts_by_occurrence))
    confidence = score_confidence(answer, report, results, settings or _settings())
    return answer, report, confidence


def _insufficient_trio() -> tuple[GroundedAnswer, CitationVerificationReport, ConfidenceAssessment]:
    answer = make_grounded_answer(answer_text=_INSUFFICIENT_TEXT)
    report = verify_grounded_answer("q", answer, FakeCitationJudge())
    confidence = score_confidence(answer, report, [], _settings())
    return answer, report, confidence


# --- policy precedence --------------------------------------------------------------


def test_insufficient_evidence_yields_insufficient_decision() -> None:
    answer, report, confidence = _insufficient_trio()
    final = apply_abstention_policy(answer, report, confidence, _settings())
    assert final.decision is AnswerDecision.ABSTAINED_INSUFFICIENT_EVIDENCE
    assert final.abstained is True
    assert final.answer_text == ABSTENTION_TEXT


def test_contradiction_yields_contradiction_decision() -> None:
    answer, report, confidence = _trio("A [1].", {1: (1, "contradicted", "conflicts")})
    final = apply_abstention_policy(answer, report, confidence, _settings())
    assert final.decision is AnswerDecision.ABSTAINED_CONTRADICTION


def test_unsupported_citation_yields_unsupported_decision() -> None:
    answer, report, confidence = _trio("A [1].", {1: (1, "unsupported", "not established")})
    final = apply_abstention_policy(answer, report, confidence, _settings())
    assert final.decision is AnswerDecision.ABSTAINED_UNSUPPORTED_CITATION


def test_low_score_yields_low_confidence_decision() -> None:
    # One partially-supported occurrence: score ~= 0.55 < 0.8, no contradiction/unsupported.
    answer, report, confidence = _trio("A [1].", {1: (1, "partially_supported", "partial")})
    assert confidence.score < 0.8
    final = apply_abstention_policy(answer, report, confidence, _settings())
    assert final.decision is AnswerDecision.ABSTAINED_LOW_CONFIDENCE


def test_healthy_assessment_is_answered() -> None:
    answer, report, confidence = _trio("A [1].", {1: (1, "supported", "matches")})
    assert confidence.score >= 0.8
    final = apply_abstention_policy(answer, report, confidence, _settings())
    assert final.decision is AnswerDecision.ANSWERED
    assert final.abstained is False


def test_contradiction_plus_low_score_resolves_to_contradiction() -> None:
    answer, report, confidence = _trio(
        "A [1]. B [2].",
        {1: (1, "contradicted", "conflicts"), 2: (2, "partially_supported", "partial")},
        reranked_results=[_dual("a", 1), _dual("b", 2)],
    )
    assert confidence.has_contradiction is True
    assert confidence.score < 0.8
    final = apply_abstention_policy(answer, report, confidence, _settings())
    assert final.decision is AnswerDecision.ABSTAINED_CONTRADICTION


def test_unsupported_plus_low_score_resolves_to_unsupported() -> None:
    answer, report, confidence = _trio(
        "A [1]. B [2].",
        {1: (1, "unsupported", "not established"), 2: (2, "partially_supported", "partial")},
        reranked_results=[_dual("a", 1), _dual("b", 2)],
    )
    assert confidence.has_contradiction is False
    assert confidence.unsupported_count == 1
    assert confidence.score < 0.8
    final = apply_abstention_policy(answer, report, confidence, _settings())
    assert final.decision is AnswerDecision.ABSTAINED_UNSUPPORTED_CITATION


def test_insufficient_flag_with_contradiction_like_counts_is_rejected_by_integrity() -> None:
    # Hand-built: insufficiency state must carry all-zero counts. A stray
    # contradiction count here is malformed, not "a contradiction that
    # also happens to be insufficient".
    answer, report, confidence = _insufficient_trio()
    malformed = dataclasses.replace(confidence, has_contradiction=True, contradicted_count=1)
    with pytest.raises(AbstentionPolicyInputError):
        apply_abstention_policy(answer, report, malformed, _settings())


# --- threshold behavior ------------------------------------------------------------


def test_default_confidence_threshold_is_0_8() -> None:
    assert _settings().confidence_threshold == pytest.approx(0.8)


def test_score_exactly_at_threshold_is_accepted() -> None:
    answer, report, confidence = _trio("A [1].", {1: (1, "supported", "ok")})
    at_threshold = dataclasses.replace(confidence, score=0.8)
    final = apply_abstention_policy(answer, report, at_threshold, _settings())
    assert final.decision is AnswerDecision.ANSWERED


def test_score_just_below_threshold_abstains() -> None:
    answer, report, confidence = _trio("A [1].", {1: (1, "supported", "ok")})
    below = dataclasses.replace(confidence, score=0.7999)
    final = apply_abstention_policy(answer, report, below, _settings())
    assert final.decision is AnswerDecision.ABSTAINED_LOW_CONFIDENCE


def test_custom_threshold_changes_the_boundary() -> None:
    answer, report, confidence = _trio("A [1].", {1: (1, "supported", "ok")})
    mid = dataclasses.replace(confidence, score=0.6)
    assert (
        apply_abstention_policy(answer, report, mid, _settings(confidence_threshold=0.5)).decision
        is AnswerDecision.ANSWERED
    )
    assert (
        apply_abstention_policy(answer, report, mid, _settings(confidence_threshold=0.7)).decision
        is AnswerDecision.ABSTAINED_LOW_CONFIDENCE
    )


def test_threshold_zero_is_valid() -> None:
    assert _settings(confidence_threshold=0.0).confidence_threshold == 0.0


def test_threshold_one_is_valid() -> None:
    assert _settings(confidence_threshold=1.0).confidence_threshold == 1.0


def test_negative_threshold_rejected() -> None:
    with pytest.raises(ValueError, match="confidence_threshold must be between 0.0 and 1.0"):
        _settings(confidence_threshold=-0.01)


def test_threshold_above_one_rejected() -> None:
    with pytest.raises(ValueError, match="confidence_threshold must be between 0.0 and 1.0"):
        _settings(confidence_threshold=1.01)


def test_threshold_nan_rejected() -> None:
    with pytest.raises(ValueError, match="confidence_threshold must be finite"):
        _settings(confidence_threshold=math.nan)


@pytest.mark.parametrize("bad", [math.inf, -math.inf])
def test_threshold_infinity_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="confidence_threshold must be finite"):
        _settings(confidence_threshold=bad)


# --- partial support ----------------------------------------------------------------


def test_partial_support_alone_does_not_force_abstention() -> None:
    answer, report, confidence = _trio(
        "A [1]. B [2].",
        {1: (1, "supported", "ok"), 2: (2, "partially_supported", "partial")},
        reranked_results=[_dual("a", 1), _dual("b", 2)],
    )
    assert confidence.partially_supported_count == 1
    # Well clear of a 0.7 threshold, so only the threshold rule could have
    # abstained here -- and it does not.
    final = apply_abstention_policy(answer, report, confidence, _settings(confidence_threshold=0.7))
    assert final.decision is AnswerDecision.ANSWERED


def test_partial_answer_above_threshold_is_answerable() -> None:
    answer, report, confidence = _trio(
        "A [1]. B [2].",
        {1: (1, "supported", "ok"), 2: (2, "partially_supported", "partial")},
        reranked_results=[_dual("a", 1), _dual("b", 2)],
    )
    final = apply_abstention_policy(answer, report, confidence, _settings(confidence_threshold=0.5))
    assert final.decision is AnswerDecision.ANSWERED
    assert final.confidence.partially_supported_count == 1


def test_partial_answer_below_threshold_abstains_through_low_confidence() -> None:
    answer, report, confidence = _trio(
        "A [1]. B [2].",
        {1: (1, "supported", "ok"), 2: (2, "partially_supported", "partial")},
        reranked_results=[_dual("a", 1), _dual("b", 2)],
    )
    final = apply_abstention_policy(
        answer, report, confidence, _settings(confidence_threshold=0.95)
    )
    assert final.decision is AnswerDecision.ABSTAINED_LOW_CONFIDENCE


# --- result model -----------------------------------------------------------------


def test_answered_preserves_original_answer_text_verbatim() -> None:
    answer, report, confidence = _trio("A [1].", {1: (1, "supported", "ok")})
    final = apply_abstention_policy(answer, report, confidence, _settings())
    assert final.answer_text == answer.answer_text
    assert final.answer_text is answer.answer_text


def test_abstention_replaces_user_facing_text_and_hides_the_draft() -> None:
    answer, report, confidence = _trio("A [1].", {1: (1, "contradicted", "conflicts")})
    final = apply_abstention_policy(answer, report, confidence, _settings())
    assert final.answer_text == ABSTENTION_TEXT
    assert final.answer_text != answer.answer_text
    assert answer.answer_text not in final.answer_text


def test_original_grounded_answer_retained_internally_when_abstaining() -> None:
    answer, report, confidence = _trio("A [1].", {1: (1, "unsupported", "no")})
    final = apply_abstention_policy(answer, report, confidence, _settings())
    assert final.abstained is True
    assert final.grounded_answer is answer


def test_confidence_assessment_is_preserved() -> None:
    answer, report, confidence = _trio("A [1].", {1: (1, "supported", "ok")})
    final = apply_abstention_policy(answer, report, confidence, _settings())
    assert final.confidence is confidence


def test_verification_report_is_preserved() -> None:
    answer, report, confidence = _trio("A [1].", {1: (1, "supported", "ok")})
    final = apply_abstention_policy(answer, report, confidence, _settings())
    assert final.verification_report is report


def test_decision_enum_and_abstained_bool_are_consistent() -> None:
    answered = apply_abstention_policy(*_trio("A [1].", {1: (1, "supported", "ok")}), _settings())
    assert answered.decision is AnswerDecision.ANSWERED
    assert answered.abstained is False
    assert answered.abstention_reason is None

    abstained = apply_abstention_policy(
        *_trio("A [1].", {1: (1, "contradicted", "x")}), _settings()
    )
    assert abstained.decision is AnswerDecision.ABSTAINED_CONTRADICTION
    assert abstained.abstained is True
    assert isinstance(abstained.abstention_reason, str) and abstained.abstention_reason


def test_final_answer_is_immutable() -> None:
    final = apply_abstention_policy(*_trio("A [1].", {1: (1, "supported", "ok")}), _settings())
    assert isinstance(final, FinalAnswer)
    with pytest.raises(dataclasses.FrozenInstanceError):
        final.answer_text = "mutated"  # type: ignore[misc]


# --- integrity ------------------------------------------------------------------


def test_confidence_score_nan_rejected() -> None:
    answer, report, confidence = _trio("A [1].", {1: (1, "supported", "ok")})
    with pytest.raises(AbstentionPolicyInputError, match="score"):
        apply_abstention_policy(
            answer, report, dataclasses.replace(confidence, score=math.nan), _settings()
        )


@pytest.mark.parametrize("bad", [1.5, -0.1, math.inf])
def test_confidence_score_outside_unit_interval_rejected(bad: float) -> None:
    answer, report, confidence = _trio("A [1].", {1: (1, "supported", "ok")})
    with pytest.raises(AbstentionPolicyInputError, match="score"):
        apply_abstention_policy(
            answer, report, dataclasses.replace(confidence, score=bad), _settings()
        )


def test_unsupported_count_mismatch_rejected() -> None:
    answer, report, confidence = _trio("A [1].", {1: (1, "supported", "ok")})
    with pytest.raises(AbstentionPolicyInputError, match="unsupported_count"):
        apply_abstention_policy(
            answer, report, dataclasses.replace(confidence, unsupported_count=3), _settings()
        )


def test_contradicted_count_mismatch_rejected() -> None:
    answer, report, confidence = _trio("A [1].", {1: (1, "supported", "ok")})
    with pytest.raises(AbstentionPolicyInputError, match="contradicted_count"):
        apply_abstention_policy(
            answer,
            report,
            dataclasses.replace(confidence, contradicted_count=2, has_contradiction=True),
            _settings(),
        )


def test_has_contradiction_flag_mismatch_rejected() -> None:
    answer, report, confidence = _trio("A [1].", {1: (1, "supported", "ok")})
    with pytest.raises(AbstentionPolicyInputError, match="has_contradiction"):
        apply_abstention_policy(
            answer, report, dataclasses.replace(confidence, has_contradiction=True), _settings()
        )


def test_total_citation_count_mismatch_rejected() -> None:
    answer, report, confidence = _trio("A [1].", {1: (1, "supported", "ok")})
    with pytest.raises(AbstentionPolicyInputError, match="total_citation_occurrences"):
        apply_abstention_policy(
            answer,
            report,
            dataclasses.replace(confidence, total_citation_occurrences=9),
            _settings(),
        )


def test_insufficient_flag_true_but_answer_is_substantive_rejected() -> None:
    answer, report, confidence = _trio("A [1].", {1: (1, "supported", "ok")})
    with pytest.raises(AbstentionPolicyInputError, match="insufficiency"):
        apply_abstention_policy(
            answer,
            report,
            dataclasses.replace(confidence, is_insufficient_evidence=True),
            _settings(),
        )


def test_insufficient_flag_false_but_answer_is_canonical_insufficiency_rejected() -> None:
    answer, report, confidence = _insufficient_trio()
    with pytest.raises(AbstentionPolicyInputError, match="insufficiency"):
        apply_abstention_policy(
            answer,
            report,
            dataclasses.replace(confidence, is_insufficient_evidence=False),
            _settings(),
        )


def test_insufficient_flag_true_but_verification_report_is_non_empty_rejected() -> None:
    # Consistent flag/answer, but the report should be empty for this state.
    answer, _, confidence = _insufficient_trio()
    _, non_empty_report, _ = _trio("A [1].", {1: (1, "supported", "ok")})
    with pytest.raises(AbstentionPolicyInputError, match="not empty"):
        apply_abstention_policy(answer, non_empty_report, confidence, _settings())


def test_threshold_invalid_at_policy_time_is_rejected() -> None:
    # A Settings that bypassed field validation (model_construct) must still
    # be caught by the policy's own trust-boundary check.
    answer, report, confidence = _trio("A [1].", {1: (1, "supported", "ok")})
    bad_settings = Settings.model_construct(confidence_threshold=1.5)
    with pytest.raises(AbstentionPolicyInputError, match="confidence_threshold"):
        apply_abstention_policy(answer, report, confidence, bad_settings)
