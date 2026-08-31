"""Semantic faithfulness evaluation: are the answer's claims supported by its evidence?

`evaluate_faithfulness()` asks whether the material factual claims in the
substantive answer are supported by the evidence that was supplied to
generation (`GroundedAnswer.evidence`). It deliberately does NOT compare
against the golden expected answer -- an answer can be faithful to
(incorrect) evidence yet wrong, or correct yet unfaithful, and those two
measurements must stay separate.

The injected `FaithfulnessJudge` identifies each material claim and
classifies it; deterministic Python maps verdicts to scores
(`CLAIM_VERDICT_SCORES`) and takes the mean. Zero claims for a
substantive answer is rejected, never scored as perfect faithfulness.

Applicability: computed only when the final policy decision was
`ANSWERED` (the rejected draft behind an abstention is not the user-facing
result). Independent of golden answerability -- an erroneously-answered
*unanswerable* case is still evaluated against its supplied evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...generation.context import format_evidence_block
from ...generation.models import AnswerDecision, FinalAnswer
from ..exceptions import EvaluationJudgeError, EvaluationJudgeOutputError, MetricInputError
from .models import CLAIM_VERDICT_SCORES, ClaimAssessment, ClaimVerdict, FaithfulnessReport
from .prompts import FAITHFULNESS_SYSTEM_PROMPT, build_faithfulness_user_prompt

_VALID_CLAIM_VERDICTS: dict[str, ClaimVerdict] = {v.value: v for v in ClaimVerdict}


@dataclass(frozen=True, slots=True)
class RawClaimVerdict:
    """One unvalidated material-claim verdict from a `FaithfulnessJudge`, before validation."""

    claim_id: int
    claim_text: str
    verdict: str
    rationale: str


@runtime_checkable
class FaithfulnessJudge(Protocol):
    def assess_faithfulness(self, system_prompt: str, user_prompt: str) -> list[RawClaimVerdict]:
        """Return an ordered list of every material factual claim in the answer, each judged.

        Mirrors `generation.CitationJudge.judge()`'s (system prompt, user
        prompt) shape. Claim ids must be contiguous `1..M`; the caller
        validates that.
        """
        ...


def _validate_raw_claim_verdicts(
    raw_claims: Sequence[RawClaimVerdict],
) -> tuple[ClaimAssessment, ...]:
    """Validate raw judge output: >=1 claim, contiguous `1..M` ids, well-formed fields.

    Rejects (via `EvaluationJudgeOutputError`): zero claims for a
    substantive answer, a non-int / bool claim_id, a duplicate id, ids
    that are not exactly `1..M`, a blank claim_text, a verdict outside the
    four `ClaimVerdict` values, and a blank rationale. Zero claims is
    never silently treated as perfect faithfulness.
    """
    if not raw_claims:
        raise EvaluationJudgeOutputError(
            "Faithfulness judge returned zero claims for a substantive ANSWERED response; it "
            "must identify at least one material factual claim (zero claims is never treated "
            "as perfect faithfulness)."
        )

    by_id: dict[int, RawClaimVerdict] = {}
    for raw in raw_claims:
        if isinstance(raw.claim_id, bool) or not isinstance(raw.claim_id, int):
            raise EvaluationJudgeOutputError(
                f"Faithfulness judge returned a non-integer claim_id: {raw.claim_id!r}."
            )
        if raw.claim_id in by_id:
            raise EvaluationJudgeOutputError(
                f"Faithfulness judge returned duplicate claim_id={raw.claim_id!r}."
            )
        if not isinstance(raw.claim_text, str) or not raw.claim_text.strip():
            raise EvaluationJudgeOutputError(
                f"Faithfulness judge returned an empty claim_text for claim_id={raw.claim_id!r}."
            )
        if not isinstance(raw.verdict, str) or raw.verdict not in _VALID_CLAIM_VERDICTS:
            raise EvaluationJudgeOutputError(
                f"Faithfulness judge returned an invalid verdict {raw.verdict!r} for "
                f"claim_id={raw.claim_id!r}; expected one of {sorted(_VALID_CLAIM_VERDICTS)}."
            )
        if not isinstance(raw.rationale, str) or not raw.rationale.strip():
            raise EvaluationJudgeOutputError(
                f"Faithfulness judge returned an empty rationale for claim_id={raw.claim_id!r}."
            )
        by_id[raw.claim_id] = raw

    if set(by_id) != set(range(1, len(by_id) + 1)):
        raise EvaluationJudgeOutputError(
            f"Faithfulness judge claim_ids must be contiguous 1..{len(by_id)}; got {sorted(by_id)}."
        )

    return tuple(
        ClaimAssessment(
            claim_id=claim_id,
            claim_text=by_id[claim_id].claim_text,
            verdict=_VALID_CLAIM_VERDICTS[by_id[claim_id].verdict],
            rationale=by_id[claim_id].rationale,
        )
        for claim_id in range(1, len(by_id) + 1)
    )


def _not_applicable(reason: str) -> FaithfulnessReport:
    return FaithfulnessReport(applicable=False, score=None, not_applicable_reason=reason)


def evaluate_faithfulness(
    *,
    question: str,
    final_answer: FinalAnswer,
    judge: FaithfulnessJudge,
) -> FaithfulnessReport:
    """Score whether `final_answer`'s claims are supported by its supplied evidence.

    Returns a non-applicable `FaithfulnessReport` (and never calls
    `judge`) when the final policy decision is anything other than
    `ANSWERED`.

    Otherwise: renders the supplied `GroundedAnswer.evidence` with the
    same `format_evidence_block()` used at generation time, asks `judge`
    to identify and classify every material claim, validates the result,
    then computes ``score = mean(CLAIM_VERDICT_SCORES[verdict])`` and
    exposes the per-verdict counts.

    Raises `MetricInputError` if an `ANSWERED` `FinalAnswer` carries no
    evidence or its text does not match its grounded answer;
    `EvaluationJudgeError` if the judge fails (cause preserved);
    `EvaluationJudgeOutputError` for malformed judge output.
    """
    if final_answer.decision is not AnswerDecision.ANSWERED:
        return _not_applicable(
            f"the final policy abstained ({final_answer.decision.value}); faithfulness of the "
            "user-facing result is not applicable (the rejected draft is not the final output)"
        )

    evidence = final_answer.grounded_answer.evidence
    if not evidence:
        raise MetricInputError(
            "FinalAnswer.decision is ANSWERED but its grounded answer carries no evidence; "
            "cannot evaluate faithfulness."
        )
    if (
        final_answer.abstained
        or final_answer.answer_text != final_answer.grounded_answer.answer_text
    ):
        raise MetricInputError(
            "FinalAnswer.decision is ANSWERED but its answer_text/abstained fields are "
            "inconsistent with its grounded answer."
        )

    user_prompt = build_faithfulness_user_prompt(
        question=question,
        answer_text=final_answer.answer_text,
        evidence_block=format_evidence_block(evidence),
    )
    try:
        raw_claims = judge.assess_faithfulness(FAITHFULNESS_SYSTEM_PROMPT, user_prompt)
    except EvaluationJudgeError:
        raise
    except Exception as exc:  # normalise any provider failure to our error type
        raise EvaluationJudgeError(f"Faithfulness judge failed: {exc}") from exc

    assessments = _validate_raw_claim_verdicts(raw_claims)
    verdicts = [a.verdict for a in assessments]
    score = sum(CLAIM_VERDICT_SCORES[v] for v in verdicts) / len(verdicts)

    return FaithfulnessReport(
        applicable=True,
        score=score,
        claim_assessments=assessments,
        supported_count=sum(1 for v in verdicts if v is ClaimVerdict.SUPPORTED),
        partially_supported_count=sum(1 for v in verdicts if v is ClaimVerdict.PARTIALLY_SUPPORTED),
        unsupported_count=sum(1 for v in verdicts if v is ClaimVerdict.UNSUPPORTED),
        contradicted_count=sum(1 for v in verdicts if v is ClaimVerdict.CONTRADICTED),
    )
