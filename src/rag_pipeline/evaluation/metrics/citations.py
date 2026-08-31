"""Deterministic citation-accuracy metrics from the production CitationVerificationReport.

No second LLM is used here. The Phase 3 citation judge already produced a
per-occurrence `SUPPORTED`/`PARTIALLY_SUPPORTED`/`UNSUPPORTED`/`CONTRADICTED`
verdict for each citation; `evaluate_citation_accuracy()` only maps and
aggregates those, and resolves the unique cited evidence chunks back to
their `source_file`s to compare against the golden legitimate source set.

This is independent of Step 3's `ConfidenceAssessment` -- the confidence
score is never read here, and `semantic_citation_support_score` is its own
metric even though it shares the verdict->number mapping.

Applicability: N/A when the final policy abstained (the rejected draft's
citations are not the user-facing result).
"""

from __future__ import annotations

from ...generation.citations import resolve_citation
from ...generation.exceptions import CitationValidationError
from ...generation.models import AnswerDecision, CitationVerdict, FinalAnswer
from ..exceptions import MetricInputError
from ..models import GoldenQACase
from .models import CitationMetrics

_CITATION_VERDICT_SCORES: dict[CitationVerdict, float] = {
    CitationVerdict.SUPPORTED: 1.0,
    CitationVerdict.PARTIALLY_SUPPORTED: 0.5,
    CitationVerdict.UNSUPPORTED: 0.0,
    CitationVerdict.CONTRADICTED: 0.0,
}


def _not_applicable(reason: str) -> CitationMetrics:
    return CitationMetrics(
        applicable=False,
        semantic_citation_support_score=None,
        fully_supported_citation_rate=None,
        cited_source_golden_match_rate=None,
        required_source_citation_recall=None,
        not_applicable_reason=reason,
    )


def evaluate_citation_accuracy(
    *,
    case: GoldenQACase,
    final_answer: FinalAnswer,
) -> CitationMetrics:
    """Compute deterministic citation-accuracy metrics for an `ANSWERED` `FinalAnswer`.

    Returns a non-applicable `CitationMetrics` when the final decision was
    an abstention.

    For an `ANSWERED` result:

    * ``semantic_citation_support_score`` -- mean of the verdict->score
      map over all citation *occurrences*.
    * ``fully_supported_citation_rate`` -- ``supported_count /
      total_citation_occurrences``.
    * ``cited_source_golden_match_rate`` -- of the DISTINCT source files
      among the cited evidence chunks, the fraction that are in
      ``expected_source_files`` ∪ ``acceptable_source_files``. Named a
      GOLDEN-SOURCE match, not universal precision.
    * ``required_source_citation_recall`` -- distinct
      ``expected_source_files`` represented among the cited evidence /
      number of ``expected_source_files``; ``None`` if the case lists no
      required sources (an erroneously-answered unanswerable case).
      Repeated citations to one source never inflate either source metric.

    Raises `MetricInputError` if an `ANSWERED` `FinalAnswer` has zero
    citation occurrences, its verification report does not belong to its
    grounded answer, or a cited number cannot be resolved to evidence.
    """
    if final_answer.decision is not AnswerDecision.ANSWERED:
        return _not_applicable(
            f"the final policy abstained ({final_answer.decision.value}); citation accuracy of "
            "the user-facing result is not applicable"
        )

    grounded = final_answer.grounded_answer
    report = final_answer.verification_report
    verifications = report.verifications
    if not verifications:
        raise MetricInputError(
            "FinalAnswer.decision is ANSWERED but its verification report has zero citation "
            "occurrences; an answered response must cite at least once."
        )
    if report.grounded_answer != grounded:
        raise MetricInputError(
            "FinalAnswer.verification_report.grounded_answer does not match "
            "FinalAnswer.grounded_answer."
        )
    if not grounded.cited_numbers:
        raise MetricInputError(
            "FinalAnswer has citation occurrences but GroundedAnswer.cited_numbers is empty."
        )

    total = len(verifications)
    supported = report.supported_count
    support_score = sum(_CITATION_VERDICT_SCORES[v.verdict] for v in verifications) / total
    fully_supported_rate = supported / total

    unique_sources: list[str] = []
    seen: set[str] = set()
    for number in grounded.cited_numbers:
        try:
            evidence = resolve_citation(grounded.evidence, number)
        except CitationValidationError as exc:
            raise MetricInputError(
                f"cannot resolve cited number {number!r} to supplied evidence: {exc}"
            ) from exc
        if evidence.source_file not in seen:
            seen.add(evidence.source_file)
            unique_sources.append(evidence.source_file)

    golden_sources = set(case.expected_source_files) | set(case.acceptable_source_files)
    matched = tuple(name for name in unique_sources if name in golden_sources)
    unmatched = tuple(name for name in unique_sources if name not in golden_sources)
    golden_match_rate = len(matched) / len(unique_sources)

    required = case.expected_source_files
    if required:
        cited_set = set(unique_sources)
        required_cited = tuple(name for name in required if name in cited_set)
        required_not_cited = tuple(name for name in required if name not in cited_set)
        required_recall: float | None = len(required_cited) / len(required)
    else:
        required_cited = ()
        required_not_cited = ()
        required_recall = None

    return CitationMetrics(
        applicable=True,
        semantic_citation_support_score=support_score,
        fully_supported_citation_rate=fully_supported_rate,
        cited_source_golden_match_rate=golden_match_rate,
        required_source_citation_recall=required_recall,
        total_citation_occurrences=total,
        supported_count=supported,
        partially_supported_count=report.partially_supported_count,
        unsupported_count=report.unsupported_count,
        contradicted_count=report.contradicted_count,
        unique_cited_source_files=tuple(unique_sources),
        matched_cited_source_files=matched,
        unmatched_cited_source_files=unmatched,
        required_sources_cited=required_cited,
        required_sources_not_cited=required_not_cited,
    )
