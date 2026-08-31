"""Deterministic abstention-behaviour metrics: did the policy answer/abstain as expected?

`evaluate_abstention()` compares the actual policy result
(`FinalAnswer.abstained`) against the golden expectation
(`Answerability.UNANSWERABLE` => expected abstention). It never infers the
decision from the confidence score -- it grades the policy's real output.

`aggregate_abstention()` is a pure roll-up over many per-case results.
Every rate has an explicit zero-denominator guard and reports `None`
(not applicable) rather than dividing by zero.
"""

from __future__ import annotations

from collections.abc import Iterable

from ...generation.models import AnswerDecision, FinalAnswer
from ..exceptions import MetricInputError
from ..models import Answerability, GoldenQACase
from .models import AbstentionAggregate, AbstentionMetrics


def evaluate_abstention(
    *,
    case: GoldenQACase,
    final_answer: FinalAnswer,
) -> AbstentionMetrics:
    """Grade one case's answer/abstain decision against golden expectation.

    * ``expected_abstain`` -- golden case is `UNANSWERABLE`.
    * ``actual_abstain`` -- ``final_answer.abstained``.
    * ``decision_correct`` -- the two agree.
    * ``false_abstention`` -- golden answerable but the policy abstained.
    * ``false_answer`` -- golden unanswerable but the policy answered.

    Raises `MetricInputError` if ``final_answer.abstained`` disagrees with
    ``final_answer.decision`` (a self-inconsistent `FinalAnswer`).
    """
    if final_answer.abstained != (final_answer.decision is not AnswerDecision.ANSWERED):
        raise MetricInputError(
            f"FinalAnswer.abstained={final_answer.abstained} is inconsistent with "
            f"decision={final_answer.decision.value}."
        )

    expected_abstain = case.answerability is Answerability.UNANSWERABLE
    actual_abstain = final_answer.abstained
    return AbstentionMetrics(
        expected_abstain=expected_abstain,
        actual_abstain=actual_abstain,
        decision_correct=expected_abstain == actual_abstain,
        false_abstention=(not expected_abstain) and actual_abstain,
        false_answer=expected_abstain and (not actual_abstain),
    )


def aggregate_abstention(items: Iterable[AbstentionMetrics]) -> AbstentionAggregate:
    """Roll per-case `AbstentionMetrics` up into corpus-level rates.

    * ``decision_accuracy`` -- correct decisions / total (``None`` if no cases).
    * ``answerable_coverage`` -- answered answerable / total answerable.
    * ``false_abstention_rate`` -- false abstentions / total answerable.
    * ``unanswerable_abstention_recall`` -- correctly abstained unanswerable
      / total unanswerable.
    * ``false_answer_rate`` -- answered unanswerable / total unanswerable.

    Each rate with a zero denominator is ``None``, never ``0.0``.
    """
    metrics = list(items)
    total = len(metrics)
    total_answerable = sum(1 for m in metrics if not m.expected_abstain)
    total_unanswerable = sum(1 for m in metrics if m.expected_abstain)

    correct = sum(1 for m in metrics if m.decision_correct)
    answered_answerable = sum(1 for m in metrics if not m.expected_abstain and not m.actual_abstain)
    false_abstentions = sum(1 for m in metrics if m.false_abstention)
    abstained_unanswerable = sum(1 for m in metrics if m.expected_abstain and m.actual_abstain)
    answered_unanswerable = sum(1 for m in metrics if m.false_answer)

    return AbstentionAggregate(
        total=total,
        total_answerable=total_answerable,
        total_unanswerable=total_unanswerable,
        decision_accuracy=(correct / total if total else None),
        answerable_coverage=(answered_answerable / total_answerable if total_answerable else None),
        false_abstention_rate=(false_abstentions / total_answerable if total_answerable else None),
        unanswerable_abstention_recall=(
            abstained_unanswerable / total_unanswerable if total_unanswerable else None
        ),
        false_answer_rate=(
            answered_unanswerable / total_unanswerable if total_unanswerable else None
        ),
    )
