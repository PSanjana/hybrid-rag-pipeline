"""Trust-boundary tests: metrics reject inconsistent GoldenQACase / FinalAnswer input.

Every deterministic metric re-checks the handful of fields it relies on and
raises `MetricInputError` (never a raw KeyError/TypeError) rather than emit a
misleading number from data that was not produced together.
"""

from __future__ import annotations

import dataclasses

import pytest

from rag_pipeline.evaluation.exceptions import EvaluationJudgeError, MetricInputError
from rag_pipeline.evaluation.metrics.abstention import evaluate_abstention
from rag_pipeline.evaluation.metrics.citations import evaluate_citation_accuracy
from rag_pipeline.evaluation.metrics.correctness import evaluate_correctness
from rag_pipeline.evaluation.metrics.faithfulness import evaluate_faithfulness
from rag_pipeline.generation.models import AnswerDecision, CitationVerdict

from .conftest import (
    FakeCorrectnessJudge,
    FakeFaithfulnessJudge,
    make_final_answer,
    make_golden_case,
    make_grounded_answer,
    make_verification_report,
)


def _answered_final(answer_text: str = "X [1]."):
    grounded = make_grounded_answer(answer_text=answer_text, sources=["authentication-api.md"])
    report = make_verification_report(grounded, [CitationVerdict.SUPPORTED])
    return make_final_answer(decision=AnswerDecision.ANSWERED, grounded=grounded, report=report)


def test_correctness_rejects_answered_final_whose_text_diverges_from_its_draft() -> None:
    final = _answered_final()
    tampered = dataclasses.replace(final, answer_text="a rewrite that is not the draft")

    with pytest.raises(MetricInputError, match="inconsistent with its grounded answer"):
        evaluate_correctness(
            case=make_golden_case(expected_facts=("f1",)),
            final_answer=tampered,
            judge=FakeCorrectnessJudge([(1, "correct", "r")]),
        )


def test_correctness_rejects_answerable_case_with_no_expected_facts() -> None:
    case = make_golden_case(expected_facts=())  # only possible via direct construction

    with pytest.raises(MetricInputError, match="no expected_facts"):
        evaluate_correctness(
            case=case,
            final_answer=_answered_final(),
            judge=FakeCorrectnessJudge([(1, "correct", "r")]),
        )


def test_correctness_passes_through_a_judge_raised_evaluation_error_unwrapped() -> None:
    original = EvaluationJudgeError("judge already normalised this")

    with pytest.raises(EvaluationJudgeError) as exc_info:
        evaluate_correctness(
            case=make_golden_case(expected_facts=("f1",)),
            final_answer=_answered_final(),
            judge=FakeCorrectnessJudge(error=original),
        )
    assert exc_info.value is original


def test_faithfulness_rejects_answered_final_with_no_evidence() -> None:
    grounded = make_grounded_answer(answer_text="no citations", sources=[])
    final = make_final_answer(decision=AnswerDecision.ANSWERED, grounded=grounded)

    with pytest.raises(MetricInputError, match="carries no evidence"):
        evaluate_faithfulness(
            question="q",
            final_answer=final,
            judge=FakeFaithfulnessJudge([(1, "c", "supported", "r")]),
        )


def test_faithfulness_rejects_answered_final_whose_text_diverges_from_its_draft() -> None:
    final = _answered_final()
    tampered = dataclasses.replace(final, answer_text="not the draft")

    with pytest.raises(MetricInputError, match="inconsistent with its grounded answer"):
        evaluate_faithfulness(
            question="q",
            final_answer=tampered,
            judge=FakeFaithfulnessJudge([(1, "c", "supported", "r")]),
        )


def test_citation_metrics_reject_report_that_belongs_to_a_different_grounded_answer() -> None:
    final = _answered_final()
    other = make_grounded_answer(answer_text="Y [1].", sources=["database-operations.md"])
    other_report = make_verification_report(other, [CitationVerdict.SUPPORTED])
    tampered = dataclasses.replace(final, verification_report=other_report)

    with pytest.raises(MetricInputError, match="does not match"):
        evaluate_citation_accuracy(case=make_golden_case(), final_answer=tampered)


def test_abstention_rejects_self_inconsistent_final_answer() -> None:
    final = _answered_final()
    # decision says ANSWERED but abstained flag flipped
    tampered = dataclasses.replace(final, abstained=True)

    with pytest.raises(MetricInputError, match="inconsistent with decision"):
        evaluate_abstention(case=make_golden_case(), final_answer=tampered)
