"""Semantic citation-support verification: per-occurrence LLM-judge verdicts.

`verify_grounded_answer()` is independently testable from generation:
given an already-produced `GroundedAnswer` and a `CitationJudge`, it
never re-runs retrieval, reranking, or generation itself.
`retrieve_generate_and_verify()` is a thin orchestration on top that
calls the existing `retrieve_and_generate()` unmodified, then verifies
its output -- retrieval/generation logic is never duplicated here.

Verification is strictly read-only: it never mutates `GroundedAnswer`,
rewrites answer text, adds/removes citations, or reorders evidence. It
only ever returns a separate `CitationVerificationReport`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from ..config import ChunkingStrategy, Settings
from ..embeddings import EmbeddingProvider
from ..reranking.base import Reranker
from .base import CitationJudge, Generator, RawJudgeVerdict
from .citations import extract_citation_occurrences, validate_citations, validate_evidence_numbering
from .context import format_evidence_block
from .exceptions import CitationJudgeError, CitationJudgeOutputError, UncitedAnswerError
from .judge_prompt import JUDGE_SYSTEM_PROMPT, annotate_answer, build_judge_user_prompt
from .models import (
    CitationOccurrence,
    CitationVerdict,
    CitationVerification,
    CitationVerificationReport,
    Evidence,
    GroundedAnswer,
)
from .prompt import is_insufficient_evidence_answer
from .service import retrieve_and_generate

logger = logging.getLogger(__name__)

_VALID_VERDICTS: dict[str, CitationVerdict] = {v.value: v for v in CitationVerdict}


def _validate_raw_verdict_shape(raw: RawJudgeVerdict) -> None:
    """Validate one raw verdict's field types/positivity, before any set/dict/string operation.

    A malformed field must always surface as `CitationJudgeOutputError`,
    never as a raw `TypeError`/`AttributeError`/`KeyError` from an
    unchecked hash or string operation downstream (e.g. hashing an
    unhashable `occurrence_id`, or calling `.strip()` on a `None`
    rationale). `bool` is rejected explicitly for `occurrence_id`/
    `citation_number` -- Python's `bool` is an `int` subclass, so an
    unchecked `isinstance(x, int)` alone would silently accept `True`/
    `False` as valid identifiers.
    """
    if isinstance(raw.occurrence_id, bool) or not isinstance(raw.occurrence_id, int):
        raise CitationJudgeOutputError(
            f"Judge returned a non-integer occurrence_id: {raw.occurrence_id!r}."
        )
    if raw.occurrence_id <= 0:
        raise CitationJudgeOutputError(
            f"Judge returned a non-positive occurrence_id: {raw.occurrence_id!r}."
        )
    if isinstance(raw.citation_number, bool) or not isinstance(raw.citation_number, int):
        raise CitationJudgeOutputError(
            f"Judge returned a non-integer citation_number: {raw.citation_number!r}."
        )
    if raw.citation_number <= 0:
        raise CitationJudgeOutputError(
            f"Judge returned a non-positive citation_number: {raw.citation_number!r}."
        )
    if not isinstance(raw.verdict, str):
        raise CitationJudgeOutputError(f"Judge returned a non-string verdict: {raw.verdict!r}.")
    if not isinstance(raw.rationale, str):
        raise CitationJudgeOutputError(f"Judge returned a non-string rationale: {raw.rationale!r}.")
    if not raw.rationale.strip():
        raise CitationJudgeOutputError("Judge returned an empty or whitespace-only rationale.")


def _validate_and_build_verifications(
    raw_verdicts: Sequence[RawJudgeVerdict],
    occurrences: Sequence[CitationOccurrence],
    evidence_by_number: dict[int, Evidence],
) -> list[CitationVerification]:
    """Strictly validate raw judge output against the exact expected occurrence set.

    Every item's fields are validated in isolation first (see
    `_validate_raw_verdict_shape`) -- only once every field is known to
    be a well-formed type does this function perform any set-membership
    or dict-key lookup with judge-supplied values. Then rejects (via
    `CitationJudgeOutputError`): a missing occurrence, a duplicate
    occurrence_id, an occurrence_id the judge invented, or a
    citation_number inconsistent with the deterministic occurrence.
    Never silently fills in, drops, or repairs a bad result.
    """
    for raw in raw_verdicts:
        _validate_raw_verdict_shape(raw)

    expected_by_id = {occurrence.occurrence_id: occurrence for occurrence in occurrences}

    seen_ids: set[int] = set()
    by_id: dict[int, RawJudgeVerdict] = {}
    for raw in raw_verdicts:
        if raw.occurrence_id in seen_ids:
            raise CitationJudgeOutputError(
                f"Judge returned duplicate occurrence_id={raw.occurrence_id!r}."
            )
        seen_ids.add(raw.occurrence_id)
        if raw.occurrence_id not in expected_by_id:
            raise CitationJudgeOutputError(
                f"Judge returned unexpected occurrence_id={raw.occurrence_id!r}; expected one "
                f"of {sorted(expected_by_id)}."
            )
        by_id[raw.occurrence_id] = raw

    missing = sorted(set(expected_by_id) - seen_ids)
    if missing:
        raise CitationJudgeOutputError(
            f"Judge did not return a verdict for occurrence_id(s) {missing}."
        )

    verifications: list[CitationVerification] = []
    for occurrence_id in sorted(expected_by_id):
        expected = expected_by_id[occurrence_id]
        raw = by_id[occurrence_id]

        if raw.citation_number != expected.citation_number:
            raise CitationJudgeOutputError(
                f"Judge returned citation_number={raw.citation_number!r} for "
                f"occurrence_id={occurrence_id!r}, but the deterministic occurrence has "
                f"citation_number={expected.citation_number!r}."
            )
        if raw.verdict not in _VALID_VERDICTS:
            raise CitationJudgeOutputError(
                f"Judge returned an invalid verdict {raw.verdict!r} for "
                f"occurrence_id={occurrence_id!r}; expected one of {sorted(_VALID_VERDICTS)}."
            )

        evidence = evidence_by_number[expected.citation_number]
        verifications.append(
            CitationVerification(
                occurrence_id=occurrence_id,
                citation_number=expected.citation_number,
                verdict=_VALID_VERDICTS[raw.verdict],
                rationale=raw.rationale,
                chunk_id=evidence.chunk_id,
            )
        )
    return verifications


def verify_grounded_answer(
    question: str,
    grounded_answer: GroundedAnswer,
    judge: CitationJudge,
) -> CitationVerificationReport:
    """Verify semantic citation support for every citation occurrence in `grounded_answer`.

    Extracts deterministic `CitationOccurrence`s from
    `grounded_answer.answer_text` (never an LLM), resolves each
    occurrence's citation number to its `Evidence`, asks `judge` for one
    verdict per occurrence via the fixed judge system prompt plus a
    judge-only annotated copy of the answer (`grounded_answer.answer_text`
    itself is never altered), strictly validates the judge's raw output
    against the exact expected occurrence set, and returns a
    `CitationVerificationReport`.

    A zero-citation-occurrence `grounded_answer` is accepted without
    calling `judge` ONLY when its text is the recognized fixed
    insufficient-evidence response (`is_insufficient_evidence_answer()`)
    -- this is not semantic abstention detection, just recognizing the
    one literal response form `generate_grounded_answer()` itself
    permits to go uncited. Any other zero-citation, non-empty answer is
    rejected outright: this service is independently callable, so it
    cannot assume every `GroundedAnswer` it receives already passed
    `generate_grounded_answer()`'s own uncited-answer check.

    Raises `UncitedAnswerError` if `grounded_answer` has zero citation
    occurrences and its text is not the recognized insufficient-evidence
    response; `CitationValidationError` if `grounded_answer.evidence`'s
    own citation numbering is malformed, or an occurrence's citation
    number falls outside the supplied evidence range (should not happen
    for a `GroundedAnswer` produced by `generate_grounded_answer`, but is
    re-checked here rather than trusted); `CitationJudgeError` if the
    judge itself raises (cause preserved; a `CitationJudgeError` the
    judge already raised is passed through unchanged, never
    double-wrapped -- mirroring `generate_grounded_answer()`'s treatment
    of `GenerationProviderError`); or `CitationJudgeOutputError` if the
    judge's output doesn't exactly match the expected occurrence set.
    """
    occurrences = extract_citation_occurrences(grounded_answer.answer_text)

    if not occurrences:
        if is_insufficient_evidence_answer(grounded_answer.answer_text):
            return CitationVerificationReport(
                grounded_answer=grounded_answer, occurrences=(), verifications=()
            )
        raise UncitedAnswerError(
            "GroundedAnswer has zero citation occurrences and is not the recognized "
            "insufficient-evidence response; cannot verify an uncited substantive answer."
        )

    validate_citations(
        [occurrence.citation_number for occurrence in occurrences], len(grounded_answer.evidence)
    )

    evidence_by_number = validate_evidence_numbering(grounded_answer.evidence)
    annotated_answer = annotate_answer(grounded_answer.answer_text, occurrences)
    evidence_block = format_evidence_block(grounded_answer.evidence)
    user_prompt = build_judge_user_prompt(question, annotated_answer, evidence_block)

    try:
        raw_verdicts = judge.judge(JUDGE_SYSTEM_PROMPT, user_prompt)
    except CitationJudgeError:
        raise
    except Exception as exc:
        raise CitationJudgeError(f"Citation judge failed: {exc}") from exc

    verifications = _validate_and_build_verifications(raw_verdicts, occurrences, evidence_by_number)

    logger.info(
        "citation verification: occurrence_count=%d supported=%d partial=%d unsupported=%d "
        "contradicted=%d",
        len(verifications),
        sum(1 for v in verifications if v.verdict is CitationVerdict.SUPPORTED),
        sum(1 for v in verifications if v.verdict is CitationVerdict.PARTIALLY_SUPPORTED),
        sum(1 for v in verifications if v.verdict is CitationVerdict.UNSUPPORTED),
        sum(1 for v in verifications if v.verdict is CitationVerdict.CONTRADICTED),
    )
    return CitationVerificationReport(
        grounded_answer=grounded_answer,
        occurrences=tuple(occurrences),
        verifications=tuple(verifications),
    )


def retrieve_generate_and_verify(
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
) -> tuple[GroundedAnswer, CitationVerificationReport]:
    """question -> retrieve_and_generate() -> verify_grounded_answer() -> (answer, report).

    Never duplicates retrieval/generation internals -- calls the
    existing `retrieve_and_generate()` unmodified, then verifies its
    output with `verify_grounded_answer()`. Both stages already raise
    their own clear, cause-preserving `GenerationError` subclasses
    (`RetrieveAndGenerateError`, `GenerationProviderError`,
    `CitationJudgeError`, `CitationJudgeOutputError`, ...); this thin
    composition adds no new exception type and does not catch anything.
    """
    grounded_answer = retrieve_and_generate(
        question,
        strategy,
        settings,
        reranker,
        generator,
        embedding_provider=embedding_provider,
        dense_top_k=dense_top_k,
        sparse_top_k=sparse_top_k,
        candidate_k=candidate_k,
        final_top_k=final_top_k,
    )
    report = verify_grounded_answer(question, grounded_answer, judge)
    return grounded_answer, report
