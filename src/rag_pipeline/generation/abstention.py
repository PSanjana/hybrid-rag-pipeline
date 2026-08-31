"""Deterministic graceful-abstention ("I don't know") policy -- Phase 3 Step 4.

`apply_abstention_policy()` is POLICY ONLY. It consumes the already
computed `GroundedAnswer` (Step 1), `CitationVerificationReport`
(Step 2), and `ConfidenceAssessment` (Step 3) and makes one deterministic
final decision: return the grounded substantive answer unchanged, or
replace it with the fixed `ABSTENTION_TEXT`. It performs NO retrieval,
generation, judging, or confidence recomputation, and mutates nothing.

Why consume Step 3 rather than recompute confidence here: `score_confidence()`
is the single, hardened, independently tested source of the heuristic
`ConfidenceAssessment` (verdict aggregation + retrieval-agreement join +
its own trust-boundary integrity checks). Recomputing any of that here
would duplicate logic, risk the two implementations drifting, and blur
the layering the earlier steps deliberately established. Step 4 instead
reads a handful of already-derived fields (`is_insufficient_evidence`,
`has_contradiction`, `unsupported_count`, `score`) and applies a fixed
precedence plus a single configurable threshold.

The threshold (`Settings.confidence_threshold`, default 0.8) is an
initial UNCALIBRATED heuristic; Phase 4 evaluation is expected to tune
it. It is a separate policy setting and never reuses the Step 3
`confidence_*` component weights.

`answer_question_with_policy()` is a thin composition: it calls the
existing `retrieve_generate_verify_and_score()` exactly once, then
`apply_abstention_policy()`. It duplicates no retrieval, generation,
verification, or confidence-scoring logic.
"""

from __future__ import annotations

import math

from ..config import ChunkingStrategy, Settings
from ..embeddings import EmbeddingProvider
from ..reranking.base import Reranker
from .base import CitationJudge, Generator
from .citations import extract_citations
from .confidence import retrieve_generate_verify_and_score
from .exceptions import AbstentionPolicyInputError
from .models import (
    AnswerDecision,
    CitationVerificationReport,
    ConfidenceAssessment,
    FinalAnswer,
    GroundedAnswer,
)
from .prompt import is_insufficient_evidence_answer

ABSTENTION_TEXT = (
    "I don't have enough reliable information in the supplied documents to answer that confidently."
)
"""The single, stable, user-facing graceful-abstention response.

Version-controlled constant. Deliberately says nothing about internal
confidence scores, retrieval algorithms (RRF/BM25/dense/reranker), or
judge verdicts/failures, and never exposes any unsupported generated
claim.
"""

_ABSTENTION_REASONS: dict[AnswerDecision, str] = {
    AnswerDecision.ABSTAINED_INSUFFICIENT_EVIDENCE: (
        "The supplied documents did not contain enough information to answer."
    ),
    AnswerDecision.ABSTAINED_CONTRADICTION: (
        "A cited source contradicts a claim in the drafted answer."
    ),
    AnswerDecision.ABSTAINED_UNSUPPORTED_CITATION: (
        "A cited source does not support a claim in the drafted answer."
    ),
    AnswerDecision.ABSTAINED_LOW_CONFIDENCE: (
        "The drafted answer did not meet the confidence threshold."
    ),
}
"""Short enum-derived display/debug text -- never a score or algorithm name."""


def _validate_policy_inputs(
    grounded_answer: GroundedAnswer,
    verification_report: CitationVerificationReport,
    confidence: ConfidenceAssessment,
    settings: Settings,
) -> None:
    """Re-check only the fields the policy decision depends on (trust boundary).

    Deliberately NOT a re-run of `score_confidence()`'s full validation
    suite: it checks that the configured threshold is valid, that
    `confidence.score` is finite and in `[0, 1]`, that the
    citation-verdict counts on `confidence` match the
    `verification_report`, that `has_contradiction` agrees with both
    count fields, and that `is_insufficient_evidence` is consistent with
    whether `grounded_answer` is the canonical zero-citation
    insufficiency response. Raises `AbstentionPolicyInputError` on any
    inconsistency rather than deciding from a malformed trio.
    """
    threshold = settings.confidence_threshold
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise AbstentionPolicyInputError(
            f"Settings.confidence_threshold={threshold!r} is not a finite value in [0, 1]."
        )

    if not math.isfinite(confidence.score) or not 0.0 <= confidence.score <= 1.0:
        raise AbstentionPolicyInputError(
            f"ConfidenceAssessment.score={confidence.score!r} is not a finite value in [0, 1]."
        )

    canonical_insufficient = is_insufficient_evidence_answer(
        grounded_answer.answer_text
    ) and not extract_citations(grounded_answer.answer_text)
    if confidence.is_insufficient_evidence != canonical_insufficient:
        raise AbstentionPolicyInputError(
            f"ConfidenceAssessment.is_insufficient_evidence={confidence.is_insufficient_evidence} "
            f"is inconsistent with the grounded answer's canonical insufficiency state "
            f"({canonical_insufficient})."
        )

    if confidence.is_insufficient_evidence:
        # Step 3 emits an all-zero assessment and an empty report for this
        # state; a non-empty report or any non-zero count means the trio
        # was not produced together (this is also where a hand-built
        # "insufficient + contradiction" assessment is rejected).
        if verification_report.occurrences or verification_report.verifications:
            raise AbstentionPolicyInputError(
                "ConfidenceAssessment.is_insufficient_evidence is True but the "
                "CitationVerificationReport is not empty."
            )
        nonzero = (
            confidence.supported_count
            or confidence.partially_supported_count
            or confidence.unsupported_count
            or confidence.contradicted_count
            or confidence.total_citation_occurrences
            or confidence.has_contradiction
        )
        if nonzero:
            raise AbstentionPolicyInputError(
                "ConfidenceAssessment.is_insufficient_evidence is True but its citation "
                "counts / has_contradiction are not all zero/false."
            )
        return

    expected = {
        "supported_count": verification_report.supported_count,
        "partially_supported_count": verification_report.partially_supported_count,
        "unsupported_count": verification_report.unsupported_count,
        "contradicted_count": verification_report.contradicted_count,
        "total_citation_occurrences": verification_report.total_occurrences,
    }
    for field, expected_value in expected.items():
        actual_value = getattr(confidence, field)
        if actual_value != expected_value:
            raise AbstentionPolicyInputError(
                f"ConfidenceAssessment.{field}={actual_value!r} does not match the "
                f"CitationVerificationReport ({expected_value!r})."
            )

    # `has_contradiction` must agree with its own count; combined with the
    # `contradicted_count == report.contradicted_count` check above this
    # also pins it to the verification report, so no separate check needed.
    if confidence.has_contradiction != (confidence.contradicted_count > 0):
        raise AbstentionPolicyInputError(
            f"ConfidenceAssessment.has_contradiction={confidence.has_contradiction} is "
            f"inconsistent with contradicted_count={confidence.contradicted_count}."
        )


def apply_abstention_policy(
    grounded_answer: GroundedAnswer,
    verification_report: CitationVerificationReport,
    confidence: ConfidenceAssessment,
    settings: Settings,
) -> FinalAnswer:
    """Decide, deterministically, whether to return the grounded answer or abstain.

    Pure: no retrieval, generation, judging, or confidence recomputation,
    and mutates nothing. After `_validate_policy_inputs()`, applies this
    fixed precedence (order matters -- a contradicted, low-scoring answer
    is `ABSTAINED_CONTRADICTION`, not `ABSTAINED_LOW_CONFIDENCE`):

    1. `confidence.is_insufficient_evidence` -> ABSTAINED_INSUFFICIENT_EVIDENCE
    2. `confidence.has_contradiction`        -> ABSTAINED_CONTRADICTION
    3. `confidence.unsupported_count > 0`    -> ABSTAINED_UNSUPPORTED_CITATION
    4. `confidence.score < settings.confidence_threshold`
                                            -> ABSTAINED_LOW_CONFIDENCE
    5. otherwise                             -> ANSWERED

    A merely `partially_supported_count > 0` never triggers abstention on
    its own: a partial verdict already pulls down the Step 3
    citation-support component, so the configured threshold (rule 4) is
    what decides whether such an answer still passes, unless rule 1-3
    fires for another reason.

    For ANSWERED, `FinalAnswer.answer_text is grounded_answer.answer_text`
    verbatim (no rewrite, no citation edits, no appended metadata). For
    any abstention, `FinalAnswer.answer_text` is `ABSTENTION_TEXT` and
    the rejected substantive answer is retained only on
    `FinalAnswer.grounded_answer` for debugging/evaluation.
    """
    _validate_policy_inputs(grounded_answer, verification_report, confidence, settings)

    if confidence.is_insufficient_evidence:
        decision = AnswerDecision.ABSTAINED_INSUFFICIENT_EVIDENCE
    elif confidence.has_contradiction:
        decision = AnswerDecision.ABSTAINED_CONTRADICTION
    elif confidence.unsupported_count > 0:
        decision = AnswerDecision.ABSTAINED_UNSUPPORTED_CITATION
    elif confidence.score < settings.confidence_threshold:
        decision = AnswerDecision.ABSTAINED_LOW_CONFIDENCE
    else:
        decision = AnswerDecision.ANSWERED

    if decision is AnswerDecision.ANSWERED:
        return FinalAnswer(
            answer_text=grounded_answer.answer_text,
            decision=decision,
            grounded_answer=grounded_answer,
            verification_report=verification_report,
            confidence=confidence,
            abstained=False,
            abstention_reason=None,
        )

    return FinalAnswer(
        answer_text=ABSTENTION_TEXT,
        decision=decision,
        grounded_answer=grounded_answer,
        verification_report=verification_report,
        confidence=confidence,
        abstained=True,
        abstention_reason=_ABSTENTION_REASONS[decision],
    )


def answer_question_with_policy(
    question: str,
    strategy: ChunkingStrategy,
    settings: Settings,
    reranker: Reranker,
    generator: Generator,
    judge: CitationJudge,
    embedding_provider: EmbeddingProvider | None = None,
    dense_top_k: int | None = None,
    sparse_top_k: int | None = None,
    candidate_k: int | None = None,
    final_top_k: int | None = None,
) -> FinalAnswer:
    """question -> retrieve_generate_verify_and_score() (once) -> apply_abstention_policy().

    A thin composition. `retrieve_generate_verify_and_score()` is the
    single Step 1-3 orchestration (retrieval + generation + verification +
    confidence scoring, each underlying stage called exactly once); this
    function calls it exactly once and feeds its `(GroundedAnswer,
    CitationVerificationReport, ConfidenceAssessment)` tuple straight
    into the pure policy. No retrieval/generation/verification/confidence
    logic is duplicated here, and upstream failures propagate unchanged.
    """
    grounded_answer, verification_report, confidence = retrieve_generate_verify_and_score(
        question,
        strategy,
        settings,
        reranker,
        generator,
        judge,
        embedding_provider=embedding_provider,
        dense_top_k=dense_top_k,
        sparse_top_k=sparse_top_k,
        candidate_k=candidate_k,
        final_top_k=final_top_k,
    )
    return apply_abstention_policy(grounded_answer, verification_report, confidence, settings)
