"""Deterministic, decomposable confidence scoring for a verified GroundedAnswer.

`score_confidence()` combines two signals into one heuristic quality
score:

1. semantic citation-support verdicts (from `CitationVerificationReport`,
   the dominant component, default weight 0.9), and
2. weak corroborating dual-channel retrieval agreement -- whether each
   UNIQUE cited evidence chunk was found by both the dense and sparse
   retrieval channels (default weight 0.1), joined back via the stable
   `Evidence.chunk_id` <-> `RerankedRetrievalResult.chunk_id` identity
   established in Phase 1/2.

This is a HEURISTIC QUALITY SIGNAL, not a calibrated probability, not a
percentage chance of correctness, and not an accept/reject decision --
deciding what to DO with a low score (abstain, warn, etc.) is Phase 3
Step 4's job, not this module's.

Deliberately excludes every raw retrieval-diagnostic score (dense cosine
similarity/distance, BM25 score, RRF score, reranker score): those are
uncalibrated, scale-incompatible native scores from earlier pipeline
stages, never combined with each other there either (see
`retrieval.fusion`/`retrieval.rerank`) -- confidence scoring continues
that same discipline by using only rank *presence*
(`dense_rank is not None`), never any of those underlying magnitudes.

`score_confidence()` is pure and side-effect-free: no network call, no
retrieval, no generation, no semantic judging, and no mutation of any
input. `retrieve_generate_verify_and_score()` is a thin orchestration on
top that composes the existing `retrieval.retrieve_reranked()`,
`service.generate_grounded_answer()`, and
`verification.verify_grounded_answer()` -- see its docstring for why it
calls those directly rather than the composite
`retrieve_and_generate()`/`retrieve_generate_and_verify()` wrappers.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..config import ChunkingStrategy, Settings
from ..embeddings import EmbeddingProvider
from ..reranking.base import Reranker
from ..retrieval import RetrievalError, retrieve_reranked
from ..retrieval.models import RerankedRetrievalResult
from .base import CitationJudge, Generator
from .citations import (
    extract_citation_occurrences,
    extract_citations,
    validate_evidence_numbering,
)
from .exceptions import ConfidenceInputError, RetrieveAndGenerateError, UncitedAnswerError
from .models import (
    CitationOccurrence,
    CitationVerdict,
    CitationVerificationReport,
    ConfidenceAssessment,
    Evidence,
    GroundedAnswer,
)
from .prompt import is_insufficient_evidence_answer
from .service import generate_grounded_answer
from .verification import verify_grounded_answer

_VERDICT_SCORES: dict[CitationVerdict, float] = {
    CitationVerdict.SUPPORTED: 1.0,
    CitationVerdict.PARTIALLY_SUPPORTED: 0.5,
    CitationVerdict.UNSUPPORTED: 0.0,
    CitationVerdict.CONTRADICTED: 0.0,
}


def _require_positive_int(value: object, label: str) -> None:
    """Reject `value` unless it is a real positive `int` (never `bool`).

    Python's `bool` is an `int` subclass and `True == 1`/`False == 0`, so
    a stray `True` occurrence/citation id would otherwise pass silently
    into a dict key or `range`-style check. `bool` is rejected explicitly
    and *before* the `isinstance(value, int)` test.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfidenceInputError(f"{label} must be a positive int, got {value!r}.")


def _validate_report_field_types(report: CitationVerificationReport) -> None:
    """Runtime-harden manually-constructed report values before any dict/set/lookup op.

    Confidence scoring is an independent trust boundary: a hand-built
    `CitationVerificationReport` carrying a `bool` `occurrence_id`, a
    non-`CitationVerdict` `verdict`, an empty `chunk_id`, etc. must raise
    a clear `ConfidenceInputError` here, never surface later as a raw
    `KeyError` (dict key / `_VERDICT_SCORES` lookup), `TypeError` (set
    membership, arithmetic), or `AttributeError`. This checks field
    *types/shapes* only; it never re-judges citation support.
    """
    for occurrence in report.occurrences:
        _require_positive_int(occurrence.occurrence_id, "CitationOccurrence.occurrence_id")
        _require_positive_int(occurrence.citation_number, "CitationOccurrence.citation_number")

    for verification in report.verifications:
        _require_positive_int(verification.occurrence_id, "CitationVerification.occurrence_id")
        _require_positive_int(verification.citation_number, "CitationVerification.citation_number")
        if not isinstance(verification.verdict, CitationVerdict):
            raise ConfidenceInputError(
                f"CitationVerification for occurrence_id={verification.occurrence_id!r} has a "
                f"verdict that is not a CitationVerdict: {verification.verdict!r}."
            )
        if not isinstance(verification.chunk_id, str) or not verification.chunk_id:
            raise ConfidenceInputError(
                f"CitationVerification for occurrence_id={verification.occurrence_id!r} has an "
                f"invalid chunk_id: {verification.chunk_id!r}."
            )


def _validate_answer_binding(
    grounded_answer: GroundedAnswer, report: CitationVerificationReport
) -> tuple[CitationOccurrence, ...]:
    """Prove the report describes the citations *actually present* in the answer text.

    `_validate_report_integrity` only checks the report's *internal*
    occurrence/verification relationships; it cannot tell whether those
    occurrences are the ones a deterministic parse of
    `grounded_answer.answer_text` yields. This re-extracts both
    structures with the existing fixed-regex parsers and requires an
    exact match -- occurrence IDs, citation numbers, and offsets all
    included -- so a hand-built report cannot score citations different
    from those really in the answer. Raises `ConfidenceInputError` on
    any disagreement; never repairs either side.
    """
    actual_occurrences = tuple(extract_citation_occurrences(grounded_answer.answer_text))
    actual_cited_numbers = tuple(extract_citations(grounded_answer.answer_text))

    if tuple(grounded_answer.cited_numbers) != actual_cited_numbers:
        raise ConfidenceInputError(
            "GroundedAnswer.cited_numbers "
            f"{tuple(grounded_answer.cited_numbers)!r} does not match the deterministic "
            f"deduplicated citation-number sequence extracted from answer_text "
            f"{actual_cited_numbers!r}."
        )

    if tuple(report.occurrences) != actual_occurrences:
        raise ConfidenceInputError(
            "CitationVerificationReport.occurrences do not exactly match the citation "
            "occurrences deterministically extracted from GroundedAnswer.answer_text "
            "(occurrence ids, citation numbers, and offsets must all agree)."
        )

    return actual_occurrences


def _validate_report_integrity(
    grounded_answer: GroundedAnswer, report: CitationVerificationReport
) -> dict[int, Evidence]:
    """Validate `report`'s internal occurrence/verification relationships before scoring.

    Assumes `score_confidence()` has already run `_validate_report_field_types()`
    (field types/shapes), the `report.grounded_answer == grounded_answer`
    check, and `_validate_answer_binding()` (occurrences match the answer
    text). Reuses `validate_evidence_numbering()` rather than
    re-implementing evidence-numbering validation. Raises
    `ConfidenceInputError` for a mismatched occurrence/verification count
    or ID set, an occurrence citing a number outside the evidence range,
    or a verification whose `chunk_id` doesn't match the `Evidence` its
    citation number resolves to. Never silently scores a malformed
    hand-built report.
    """
    if len(report.occurrences) != len(report.verifications):
        raise ConfidenceInputError(
            f"CitationVerificationReport has {len(report.occurrences)} occurrence(s) but "
            f"{len(report.verifications)} verification(s); they must align 1:1."
        )

    # Belt-and-suspenders: `_validate_answer_binding()` already guarantees
    # `report.occurrences` is the contiguous, unique-id parse of the answer
    # text, so this can't fire in the current call path -- kept so the
    # helper stays correct if ever called on its own.
    occurrences_by_id = {occurrence.occurrence_id: occurrence for occurrence in report.occurrences}
    if len(occurrences_by_id) != len(report.occurrences):
        raise ConfidenceInputError(
            "CitationVerificationReport contains duplicate occurrence_id(s)."
        )

    verifications_by_id = {v.occurrence_id: v for v in report.verifications}
    if len(verifications_by_id) != len(report.verifications):
        raise ConfidenceInputError(
            "CitationVerificationReport contains duplicate verification occurrence_id(s)."
        )

    if set(occurrences_by_id) != set(verifications_by_id):
        raise ConfidenceInputError(
            "CitationVerificationReport occurrence IDs and verification occurrence IDs do not "
            "align 1:1."
        )

    evidence_by_number = validate_evidence_numbering(grounded_answer.evidence)

    for occurrence_id, occurrence in occurrences_by_id.items():
        verification = verifications_by_id[occurrence_id]
        if verification.citation_number != occurrence.citation_number:
            raise ConfidenceInputError(
                f"Verification for occurrence_id={occurrence_id!r} has citation_number="
                f"{verification.citation_number!r}, but the occurrence has citation_number="
                f"{occurrence.citation_number!r}."
            )
        if occurrence.citation_number not in evidence_by_number:
            raise ConfidenceInputError(
                f"Occurrence_id={occurrence_id!r} cites citation_number="
                f"{occurrence.citation_number!r}, which is not present in the supplied evidence."
            )
        expected_evidence = evidence_by_number[occurrence.citation_number]
        if verification.chunk_id != expected_evidence.chunk_id:
            raise ConfidenceInputError(
                f"Verification for occurrence_id={occurrence_id!r} has chunk_id="
                f"{verification.chunk_id!r}, but citation_number="
                f"{occurrence.citation_number!r} resolves to evidence chunk_id="
                f"{expected_evidence.chunk_id!r}."
            )

    return evidence_by_number


def _validate_retrieval_results(
    reranked_results: Sequence[RerankedRetrievalResult],
) -> dict[str, RerankedRetrievalResult]:
    """Build `chunk_id -> RerankedRetrievalResult`, validating chunk_id integrity first.

    Requires every `chunk_id` to be a non-empty string and unique across
    `reranked_results` -- never builds this mapping via a bare dict
    comprehension, which would let a duplicate `chunk_id` silently
    overwrite an earlier result instead of being rejected.
    """
    by_chunk_id: dict[str, RerankedRetrievalResult] = {}
    for result in reranked_results:
        chunk_id = result.chunk_id
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ConfidenceInputError(f"Retrieval result has an invalid chunk_id: {chunk_id!r}.")
        if chunk_id in by_chunk_id:
            raise ConfidenceInputError(f"Duplicate chunk_id in reranked_results: {chunk_id!r}.")
        by_chunk_id[chunk_id] = result
    return by_chunk_id


def score_confidence(
    grounded_answer: GroundedAnswer,
    verification_report: CitationVerificationReport,
    reranked_results: Sequence[RerankedRetrievalResult],
    settings: Settings,
) -> ConfidenceAssessment:
    """Compute a deterministic heuristic confidence assessment for `grounded_answer`.

    Trust-boundary order of checks (each raises before any scoring):

    1. `_validate_report_field_types()` -- runtime-harden manually-built
       report field values (positive-int ids/citation numbers, real
       `CitationVerdict` verdicts, non-empty `chunk_id`s) so a malformed
       value cannot later surface as a raw `KeyError`/`TypeError`.
    2. `verification_report.grounded_answer == grounded_answer` -- the
       report must belong to the answer being scored.
    3. `_validate_answer_binding()` -- `grounded_answer.cited_numbers`
       and `verification_report.occurrences` must exactly match what the
       deterministic parsers (`extract_citations()` /
       `extract_citation_occurrences()`) yield from
       `grounded_answer.answer_text`. A report cannot score citations
       different from those actually in the answer text.

    Then, if `grounded_answer` is the recognized insufficient-evidence
    response (`is_insufficient_evidence_answer()`) AND has zero actual
    citation occurrences: the report must also be empty (else
    `ConfidenceInputError`), and a valid all-zero `ConfidenceAssessment`
    with `is_insufficient_evidence=True` is returned -- WITHOUT
    interpreting `verification_report.all_supported == True` (vacuously
    true at zero occurrences) as high confidence. If the insufficiency
    phrase appears in an answer that *also* contains citations, those
    citations are NOT discarded: it is scored as an ordinary cited
    answer.

    For an ordinary cited answer: requires at least one citation
    occurrence (`UncitedAnswerError` if not), then `_validate_report_integrity()`
    (`ConfidenceInputError`) and `_validate_retrieval_results()`
    (`ConfidenceInputError` -- clean, unique `chunk_id`s covering every
    cited evidence chunk).

    `citation_support_score` is the mean of each verification's mapped
    verdict value (SUPPORTED=1.0, PARTIALLY_SUPPORTED=0.5,
    UNSUPPORTED=0.0, CONTRADICTED=0.0) over ALL citation occurrences
    (not deduplicated by citation number -- a claim repeated twice with
    different verdicts contributes both). `retrieval_agreement_score` is
    computed only over UNIQUE cited evidence chunks (repeated citations
    to the same chunk count once): the fraction with both
    `dense_rank is not None` and `sparse_rank is not None`. `score` is
    the weighted average of the two, normalized by the configured
    weights' sum (`settings.confidence_citation_weight`/
    `settings.confidence_retrieval_agreement_weight`), so it stays in
    `[0, 1]` regardless of how the weights are tuned.
    """
    citation_weight = settings.confidence_citation_weight
    retrieval_weight = settings.confidence_retrieval_agreement_weight

    _validate_report_field_types(verification_report)

    if verification_report.grounded_answer != grounded_answer:
        raise ConfidenceInputError(
            "CitationVerificationReport.grounded_answer does not match the GroundedAnswer "
            "being scored."
        )

    actual_occurrences = _validate_answer_binding(grounded_answer, verification_report)

    if is_insufficient_evidence_answer(grounded_answer.answer_text) and not actual_occurrences:
        if verification_report.occurrences or verification_report.verifications:
            raise ConfidenceInputError(
                "GroundedAnswer is the recognized zero-citation insufficient-evidence "
                "response, but its CitationVerificationReport is non-empty."
            )
        return ConfidenceAssessment(
            score=0.0,
            citation_support_score=0.0,
            retrieval_agreement_score=0.0,
            supported_count=0,
            partially_supported_count=0,
            unsupported_count=0,
            contradicted_count=0,
            total_citation_occurrences=0,
            unique_cited_evidence_count=0,
            dual_channel_cited_evidence_count=0,
            has_contradiction=False,
            is_insufficient_evidence=True,
            citation_weight=citation_weight,
            retrieval_agreement_weight=retrieval_weight,
        )

    if not actual_occurrences:
        raise UncitedAnswerError(
            "Cannot score confidence: GroundedAnswer is not the recognized "
            "insufficient-evidence response but has zero citation occurrences."
        )
    if not verification_report.verifications:
        raise UncitedAnswerError(
            "Cannot score confidence: GroundedAnswer has citation occurrences but the "
            "verification report has zero verifications."
        )

    evidence_by_number = _validate_report_integrity(grounded_answer, verification_report)
    by_chunk_id = _validate_retrieval_results(reranked_results)

    verdicts = [v.verdict for v in verification_report.verifications]
    supported_count = sum(1 for v in verdicts if v is CitationVerdict.SUPPORTED)
    partially_supported_count = sum(1 for v in verdicts if v is CitationVerdict.PARTIALLY_SUPPORTED)
    unsupported_count = sum(1 for v in verdicts if v is CitationVerdict.UNSUPPORTED)
    contradicted_count = sum(1 for v in verdicts if v is CitationVerdict.CONTRADICTED)
    total_occurrences = len(verdicts)

    citation_support_score = sum(_VERDICT_SCORES[v] for v in verdicts) / total_occurrences

    cited_citation_numbers = {
        occurrence.citation_number for occurrence in verification_report.occurrences
    }
    # Every cited evidence chunk_id feeds a dict lookup / set op in the
    # retrieval-agreement join below, so it must be a non-empty str.
    # `_validate_report_field_types()` + `_validate_report_integrity()`
    # already guarantee this transitively (each equals a validated
    # verification.chunk_id), but the join is an explicit trust boundary
    # and re-checks rather than assuming.
    cited_chunk_ids: set[str] = set()
    for number in cited_citation_numbers:
        chunk_id = evidence_by_number[number].chunk_id
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ConfidenceInputError(
                f"Cited evidence for citation_number={number!r} has an invalid chunk_id: "
                f"{chunk_id!r}."
            )
        cited_chunk_ids.add(chunk_id)

    dual_channel_count = 0
    for chunk_id in cited_chunk_ids:
        result = by_chunk_id.get(chunk_id)
        if result is None:
            raise ConfidenceInputError(
                f"Cited evidence chunk_id={chunk_id!r} was not found among the supplied "
                "reranked_results; cannot compute retrieval agreement."
            )
        if result.dense_rank is not None and result.sparse_rank is not None:
            dual_channel_count += 1

    unique_cited_evidence_count = len(cited_chunk_ids)
    retrieval_agreement_score = dual_channel_count / unique_cited_evidence_count

    weight_sum = citation_weight + retrieval_weight
    score = (
        citation_weight * citation_support_score + retrieval_weight * retrieval_agreement_score
    ) / weight_sum

    return ConfidenceAssessment(
        score=score,
        citation_support_score=citation_support_score,
        retrieval_agreement_score=retrieval_agreement_score,
        supported_count=supported_count,
        partially_supported_count=partially_supported_count,
        unsupported_count=unsupported_count,
        contradicted_count=contradicted_count,
        total_citation_occurrences=total_occurrences,
        unique_cited_evidence_count=unique_cited_evidence_count,
        dual_channel_cited_evidence_count=dual_channel_count,
        has_contradiction=contradicted_count > 0,
        is_insufficient_evidence=False,
        citation_weight=citation_weight,
        retrieval_agreement_weight=retrieval_weight,
    )


def retrieve_generate_verify_and_score(
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
) -> tuple[GroundedAnswer, CitationVerificationReport, ConfidenceAssessment]:
    """question -> retrieve_reranked() -> generate -> verify -> score_confidence().

    A thin composition, not a duplication: it calls the exact same
    authoritative pure functions `retrieve_and_generate()`/
    `retrieve_generate_and_verify()` already call --
    `retrieval.retrieve_reranked()`, `service.generate_grounded_answer()`,
    and `verification.verify_grounded_answer()` -- each exactly once. It
    deliberately does NOT call `retrieve_and_generate()`/
    `retrieve_generate_and_verify()` themselves: those composite wrappers
    intentionally return only their final result and discard the
    intermediate `RerankedRetrievalResult` list, but confidence scoring
    specifically needs that list for its retrieval-agreement component.
    Both composite wrappers remain available, unmodified, for callers
    that don't need confidence scoring.

    A retrieval failure is wrapped as `RetrieveAndGenerateError` (cause
    preserved), exactly mirroring `retrieve_and_generate()`'s own
    treatment; every other failure (generation, verification, or
    confidence-scoring integrity) propagates unwrapped.
    """
    try:
        reranked_results = retrieve_reranked(
            question,
            strategy,
            settings,
            reranker,
            embedding_provider=embedding_provider,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            candidate_k=candidate_k,
            final_top_k=final_top_k,
        )
    except RetrievalError as exc:
        raise RetrieveAndGenerateError(f"Retrieval failed during generation: {exc}") from exc

    grounded_answer = generate_grounded_answer(question, reranked_results, generator)
    report = verify_grounded_answer(question, grounded_answer, judge)
    assessment = score_confidence(grounded_answer, report, reranked_results, settings)
    return grounded_answer, report, assessment
