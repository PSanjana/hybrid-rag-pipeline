"""Offline tests for rag_pipeline.evaluation.metrics.abstention (deterministic, no LLM)."""

from __future__ import annotations

from rag_pipeline.evaluation.metrics.abstention import aggregate_abstention, evaluate_abstention
from rag_pipeline.evaluation.metrics.models import AbstentionMetrics
from rag_pipeline.evaluation.models import Answerability, QuestionType
from rag_pipeline.generation.models import AnswerDecision

from .conftest import make_final_answer, make_golden_case, make_grounded_answer


def _answerable_case() -> object:
    return make_golden_case()


def _unanswerable_case() -> object:
    return make_golden_case(
        id="absent-1",
        answerability=Answerability.UNANSWERABLE,
        question_type=QuestionType.UNANSWERABLE_ABSENT,
        expected_answer=None,
        expected_facts=(),
        expected_source_files=(),
    )


def _answered() -> object:
    grounded = make_grounded_answer(answer_text="X [1].", sources=["authentication-api.md"])
    return make_final_answer(decision=AnswerDecision.ANSWERED, grounded=grounded)


def _abstained(decision: AnswerDecision = AnswerDecision.ABSTAINED_LOW_CONFIDENCE) -> object:
    grounded = make_grounded_answer(answer_text="X [1].", sources=["authentication-api.md"])
    return make_final_answer(decision=decision, grounded=grounded, abstention_reason="reason")


def test_answerable_and_answered_is_a_correct_decision() -> None:
    m = evaluate_abstention(case=_answerable_case(), final_answer=_answered())

    assert m.expected_abstain is False
    assert m.actual_abstain is False
    assert m.decision_correct is True
    assert m.false_abstention is False
    assert m.false_answer is False


def test_answerable_and_abstained_is_a_false_abstention() -> None:
    m = evaluate_abstention(case=_answerable_case(), final_answer=_abstained())

    assert m.decision_correct is False
    assert m.false_abstention is True
    assert m.false_answer is False


def test_unanswerable_and_abstained_is_a_correct_decision() -> None:
    m = evaluate_abstention(
        case=_unanswerable_case(),
        final_answer=_abstained(AnswerDecision.ABSTAINED_INSUFFICIENT_EVIDENCE),
    )

    assert m.expected_abstain is True
    assert m.actual_abstain is True
    assert m.decision_correct is True
    assert m.false_abstention is False
    assert m.false_answer is False


def test_unanswerable_and_answered_is_a_false_answer() -> None:
    m = evaluate_abstention(case=_unanswerable_case(), final_answer=_answered())

    assert m.decision_correct is False
    assert m.false_answer is True
    assert m.false_abstention is False


def _mix() -> list[AbstentionMetrics]:
    # 3 answerable: 2 answered, 1 false-abstention.
    # 2 unanswerable: 1 abstained, 1 false-answer.
    return [
        AbstentionMetrics(False, False, True, False, False),
        AbstentionMetrics(False, False, True, False, False),
        AbstentionMetrics(False, True, False, True, False),
        AbstentionMetrics(True, True, True, False, False),
        AbstentionMetrics(True, False, False, False, True),
    ]


def test_aggregate_decision_accuracy() -> None:
    agg = aggregate_abstention(_mix())

    assert agg.total == 5
    assert agg.decision_accuracy == 3 / 5


def test_aggregate_answerable_coverage() -> None:
    agg = aggregate_abstention(_mix())

    assert agg.total_answerable == 3
    assert agg.answerable_coverage == 2 / 3


def test_aggregate_false_abstention_rate() -> None:
    agg = aggregate_abstention(_mix())

    assert agg.false_abstention_rate == 1 / 3


def test_aggregate_unanswerable_abstention_recall_and_false_answer_rate() -> None:
    agg = aggregate_abstention(_mix())

    assert agg.total_unanswerable == 2
    assert agg.unanswerable_abstention_recall == 1 / 2
    assert agg.false_answer_rate == 1 / 2


def test_aggregate_zero_denominators_are_not_applicable() -> None:
    only_answerable = [AbstentionMetrics(False, False, True, False, False)]
    agg = aggregate_abstention(only_answerable)

    assert agg.unanswerable_abstention_recall is None
    assert agg.false_answer_rate is None
    assert agg.answerable_coverage == 1.0

    empty = aggregate_abstention([])
    assert empty.decision_accuracy is None
    assert empty.answerable_coverage is None
    assert empty.false_abstention_rate is None
