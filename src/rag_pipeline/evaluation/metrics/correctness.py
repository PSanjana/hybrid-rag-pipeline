"""Semantic answer-correctness evaluation against golden expected facts.

`evaluate_correctness()` measures two orthogonal things, both judged from
the golden truth alone (never from retrieved evidence -- that is
faithfulness):

1. **Expected-fact coverage.** One verdict + rationale per numbered
   golden `expected_fact`. Deterministic Python maps verdicts to scores
   (`FACT_VERDICT_SCORES`) and takes the mean -> `expected_fact_score`.
   The judge never returns the number.
2. **Answer-level golden contradiction.** Whether the *complete* answer
   contains one or more MATERIAL claims that DIRECTLY CONFLICT with the
   supplied golden truth. Per-fact scoring cannot catch this: an answer
   can state every expected fact correctly and still add a contradicting
   claim. An extra statement merely *absent* from the (non-exhaustive)
   golden facts is NOT a contradiction.

No numeric penalty for contradictions is applied here -- `score` /
`expected_fact_score` stay the pure coverage mean, and
`has_golden_contradiction` is a separate signal for later analysis to
combine.

Applicability (deliberately narrow): correctness is computed only for an
ANSWERABLE golden case whose final policy decision was `ANSWERED`. An
answerable case that abstained is N/A (the false abstention is the
abstention metric's job); an unanswerable case is N/A (no golden facts).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ...generation.models import AnswerDecision, FinalAnswer
from ..exceptions import EvaluationJudgeError, EvaluationJudgeOutputError, MetricInputError
from ..models import Answerability, GoldenQACase
from .models import (
    FACT_VERDICT_SCORES,
    CorrectnessReport,
    FactAssessment,
    FactVerdict,
    GoldenContradiction,
)
from .prompts import CORRECTNESS_SYSTEM_PROMPT, build_correctness_user_prompt

_VALID_FACT_VERDICTS: dict[str, FactVerdict] = {v.value: v for v in FactVerdict}


@dataclass(frozen=True, slots=True)
class RawFactVerdict:
    """One unvalidated per-fact verdict from a `CorrectnessJudge`, before validation.

    `verdict` is a raw `str`, not yet a `FactVerdict` -- a judge's output
    (real or fake) is never trusted until `evaluate_correctness()` has
    validated the whole set against the exact expected fact IDs.
    """

    fact_id: int
    verdict: str
    rationale: str


@dataclass(frozen=True, slots=True)
class RawGoldenContradiction:
    """One unvalidated answer-level golden-contradiction item from a `CorrectnessJudge`."""

    contradiction_id: int
    claim_text: str
    rationale: str
    conflicting_fact_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class RawCorrectnessAssessment:
    """A `CorrectnessJudge`'s full unvalidated result: per-fact verdicts + contradictions.

    A single structured return keeps the two judged concerns together and
    leaves room to grow without another protocol change. Both lists are
    validated -- against the exact expected fact-id set and for
    contiguous, well-formed contradiction ids -- before any
    `CorrectnessReport` is built. An empty `golden_contradictions` list is
    valid and means "no contradiction found".
    """

    fact_verdicts: list[RawFactVerdict]
    golden_contradictions: list[RawGoldenContradiction] = field(default_factory=list)


@runtime_checkable
class CorrectnessJudge(Protocol):
    def assess_correctness(self, system_prompt: str, user_prompt: str) -> RawCorrectnessAssessment:
        """Return per-fact verdicts plus any answer-level golden contradictions.

        Mirrors `generation.CitationJudge.judge()`'s (system prompt, user
        prompt) shape. Must not filter, deduplicate, reorder, or add
        facts; must return an empty contradiction list when none exist.
        All set/shape validation happens in `evaluate_correctness()`.
        """
        ...


def _validate_raw_fact_verdicts(
    raw_verdicts: Sequence[RawFactVerdict], fact_count: int
) -> dict[int, RawFactVerdict]:
    """Strictly validate raw judge output against the exact fact-id set `1..fact_count`.

    Rejects (via `EvaluationJudgeOutputError`): a non-int / bool fact_id,
    a duplicate id, an id outside `1..fact_count`, a missing id, a verdict
    that is not one of the four `FactVerdict` values, and a blank
    rationale. Never fills in, drops, or repairs a bad result.
    """
    expected_ids = set(range(1, fact_count + 1))
    by_id: dict[int, RawFactVerdict] = {}
    for raw in raw_verdicts:
        if isinstance(raw.fact_id, bool) or not isinstance(raw.fact_id, int):
            raise EvaluationJudgeOutputError(
                f"Correctness judge returned a non-integer fact_id: {raw.fact_id!r}."
            )
        if raw.fact_id in by_id:
            raise EvaluationJudgeOutputError(
                f"Correctness judge returned duplicate fact_id={raw.fact_id!r}."
            )
        if raw.fact_id not in expected_ids:
            raise EvaluationJudgeOutputError(
                f"Correctness judge returned unexpected fact_id={raw.fact_id!r}; "
                f"expected one of 1..{fact_count}."
            )
        if not isinstance(raw.verdict, str) or raw.verdict not in _VALID_FACT_VERDICTS:
            raise EvaluationJudgeOutputError(
                f"Correctness judge returned an invalid verdict {raw.verdict!r} for "
                f"fact_id={raw.fact_id!r}; expected one of {sorted(_VALID_FACT_VERDICTS)}."
            )
        if not isinstance(raw.rationale, str) or not raw.rationale.strip():
            raise EvaluationJudgeOutputError(
                f"Correctness judge returned an empty rationale for fact_id={raw.fact_id!r}."
            )
        by_id[raw.fact_id] = raw

    missing = sorted(expected_ids - set(by_id))
    if missing:
        raise EvaluationJudgeOutputError(
            f"Correctness judge did not return a verdict for fact_id(s) {missing}."
        )
    return by_id


def _validate_raw_contradictions(
    raw_contradictions: Sequence[RawGoldenContradiction], fact_count: int
) -> tuple[GoldenContradiction, ...]:
    """Validate the answer-level contradiction list: empty is fine; otherwise strict.

    Rejects (via `EvaluationJudgeOutputError`): a non-int / bool
    contradiction_id, a duplicate id, ids that are not exactly `1..N`, a
    blank `claim_text`, a blank `rationale`, and any
    `conflicting_fact_ids` entry that is not a real int in
    `1..fact_count` or that repeats. Returns `()` for an empty list --
    zero contradictions is a valid, common result.
    """
    if not raw_contradictions:
        return ()

    by_id: dict[int, RawGoldenContradiction] = {}
    for raw in raw_contradictions:
        if isinstance(raw.contradiction_id, bool) or not isinstance(raw.contradiction_id, int):
            raise EvaluationJudgeOutputError(
                f"Correctness judge returned a non-integer contradiction_id: "
                f"{raw.contradiction_id!r}."
            )
        if raw.contradiction_id in by_id:
            raise EvaluationJudgeOutputError(
                f"Correctness judge returned duplicate contradiction_id={raw.contradiction_id!r}."
            )
        if not isinstance(raw.claim_text, str) or not raw.claim_text.strip():
            raise EvaluationJudgeOutputError(
                f"Correctness judge returned an empty claim_text for "
                f"contradiction_id={raw.contradiction_id!r}."
            )
        if not isinstance(raw.rationale, str) or not raw.rationale.strip():
            raise EvaluationJudgeOutputError(
                f"Correctness judge returned an empty rationale for "
                f"contradiction_id={raw.contradiction_id!r}."
            )
        seen_fact_ids: set[int] = set()
        for fact_id in raw.conflicting_fact_ids:
            if (
                isinstance(fact_id, bool)
                or not isinstance(fact_id, int)
                or not 1 <= fact_id <= fact_count
            ):
                raise EvaluationJudgeOutputError(
                    f"Correctness judge returned an invalid conflicting_fact_id {fact_id!r} for "
                    f"contradiction_id={raw.contradiction_id!r}; expected an int in "
                    f"1..{fact_count}."
                )
            if fact_id in seen_fact_ids:
                raise EvaluationJudgeOutputError(
                    f"Correctness judge repeated conflicting_fact_id={fact_id!r} for "
                    f"contradiction_id={raw.contradiction_id!r}."
                )
            seen_fact_ids.add(fact_id)
        by_id[raw.contradiction_id] = raw

    if set(by_id) != set(range(1, len(by_id) + 1)):
        raise EvaluationJudgeOutputError(
            f"Correctness judge contradiction_ids must be contiguous 1..{len(by_id)}; "
            f"got {sorted(by_id)}."
        )

    return tuple(
        GoldenContradiction(
            contradiction_id=cid,
            claim_text=by_id[cid].claim_text,
            rationale=by_id[cid].rationale,
            conflicting_fact_ids=tuple(by_id[cid].conflicting_fact_ids),
        )
        for cid in range(1, len(by_id) + 1)
    )


def _not_applicable(reason: str) -> CorrectnessReport:
    return CorrectnessReport(
        applicable=False, score=None, expected_fact_score=None, not_applicable_reason=reason
    )


def evaluate_correctness(
    *,
    case: GoldenQACase,
    final_answer: FinalAnswer,
    judge: CorrectnessJudge,
) -> CorrectnessReport:
    """Score golden-fact coverage and detect answer-level golden contradictions.

    Returns a non-applicable `CorrectnessReport` (and never calls
    `judge`) when the golden case is unanswerable, or when the final
    policy decision is anything other than `ANSWERED`.

    Otherwise: builds the judge prompt from the question, the substantive
    answer, the numbered golden facts, and (as context only) the
    reference `expected_answer`; asks `judge` for (a) one verdict per fact
    and (b) any material answer claims that directly conflict with the
    golden truth; strictly validates both; then computes
    ``expected_fact_score = mean(FACT_VERDICT_SCORES[verdict])`` over all
    facts (also mirrored as `score`), the per-verdict counts, and
    `golden_contradiction_count` / `has_golden_contradiction` /
    `golden_contradictions`. No numeric penalty is applied for
    contradictions.

    Raises `MetricInputError` if the case/answer pair is internally
    inconsistent (answerable case with no `expected_facts`; an `ANSWERED`
    `FinalAnswer` whose text does not match its grounded answer);
    `EvaluationJudgeError` if the judge itself fails (cause preserved);
    `EvaluationJudgeOutputError` if the judge's output does not match the
    expected fact-id set or its contradiction list is malformed.
    """
    if case.answerability is not Answerability.ANSWERABLE:
        return _not_applicable(
            "golden case is unanswerable; there are no golden expected facts to score "
            "answer correctness against"
        )
    if final_answer.decision is not AnswerDecision.ANSWERED:
        return _not_applicable(
            "golden case is answerable but the final policy abstained "
            f"({final_answer.decision.value}); correctness is not applicable to an abstention "
            "(a false abstention is measured by the abstention metric)"
        )

    facts = case.expected_facts
    if not facts:
        raise MetricInputError(
            f"{case.id}: answerable golden case has no expected_facts; cannot evaluate correctness."
        )
    if (
        final_answer.abstained
        or final_answer.answer_text != final_answer.grounded_answer.answer_text
    ):
        raise MetricInputError(
            f"{case.id}: FinalAnswer.decision is ANSWERED but its answer_text/abstained fields "
            "are inconsistent with its grounded answer."
        )

    user_prompt = build_correctness_user_prompt(
        question=case.question,
        answer_text=final_answer.answer_text,
        expected_facts=facts,
        expected_answer=case.expected_answer,
    )
    try:
        raw = judge.assess_correctness(CORRECTNESS_SYSTEM_PROMPT, user_prompt)
    except EvaluationJudgeError:
        raise
    except Exception as exc:  # normalise any provider failure to our error type
        raise EvaluationJudgeError(f"Correctness judge failed: {exc}") from exc

    by_id = _validate_raw_fact_verdicts(raw.fact_verdicts, len(facts))
    contradictions = _validate_raw_contradictions(raw.golden_contradictions, len(facts))

    assessments = tuple(
        FactAssessment(
            fact_id=fact_id,
            fact_text=facts[fact_id - 1],
            verdict=_VALID_FACT_VERDICTS[by_id[fact_id].verdict],
            rationale=by_id[fact_id].rationale,
        )
        for fact_id in range(1, len(facts) + 1)
    )
    verdicts = [a.verdict for a in assessments]
    expected_fact_score = sum(FACT_VERDICT_SCORES[v] for v in verdicts) / len(verdicts)

    return CorrectnessReport(
        applicable=True,
        score=expected_fact_score,
        expected_fact_score=expected_fact_score,
        fact_assessments=assessments,
        correct_count=sum(1 for v in verdicts if v is FactVerdict.CORRECT),
        partially_correct_count=sum(1 for v in verdicts if v is FactVerdict.PARTIALLY_CORRECT),
        missing_count=sum(1 for v in verdicts if v is FactVerdict.MISSING),
        contradicted_count=sum(1 for v in verdicts if v is FactVerdict.CONTRADICTED),
        golden_contradictions=contradictions,
        golden_contradiction_count=len(contradictions),
        has_golden_contradiction=bool(contradictions),
    )
