"""Typed, immutable models for grounded generation and citation verification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..config import ChunkingStrategy


@dataclass(frozen=True, slots=True)
class Evidence:
    """One numbered piece of evidence handed to the generator, derived 1:1 from a reranked chunk.

    `citation_number` is the *only* identifier the model is allowed to
    cite (`[1]`, `[2]`, ...) -- assigned deterministically from reranked
    order: reranked rank 1 -> citation_number 1, rank 2 -> 2, and so on.
    The underlying SHA-256 `chunk_id` is retained for internal
    provenance/debugging but is never the citation syntax shown to the
    model or the user. `reranked_rank` is kept as a distinct, explicitly
    named field (even though it is numerically identical to
    `citation_number` today) because it names a *retrieval* concept,
    while `citation_number` names a *presentation* concept -- keeping
    them named separately leaves room for the two to diverge later
    (e.g. citation renumbering) without an ambiguous single field.

    Deliberately excludes every retrieval-diagnostic score (rrf_score,
    reranker_score, dense/sparse ranks, bm25/dense similarity) -- those
    are ranking diagnostics, not factual evidence, and must never reach
    the generation prompt; they remain available on the original
    `RerankedRetrievalResult` for logs/internal use.
    """

    citation_number: int
    chunk_id: str
    text: str
    source_file: str
    document_id: str
    chunk_index: int
    section_heading: str | None
    page_number: int | None
    chunking_strategy: ChunkingStrategy
    reranked_rank: int


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """A generated answer grounded in, and cited against, supplied `Evidence`.

    `evidence` is the full ordered evidence set that was supplied to the
    generator (index i corresponds to citation number i+1) -- not just
    the subset the model happened to cite -- so any citation number the
    model could have used can always be resolved back to its provenance
    (see `generation.citations.resolve_citation`). `cited_numbers` is the
    set of citation numbers the model actually used, in first-appearance
    order, already validated to fall within the supplied evidence range.
    Both are tuples (not lists) so the result is genuinely immutable, not
    just non-reassignable.
    """

    answer_text: str
    evidence: tuple[Evidence, ...]
    cited_numbers: tuple[int, ...]
    generation_model: str | None = None


@dataclass(frozen=True, slots=True)
class CitationOccurrence:
    """One concrete appearance of a citation bracket in generated answer text.

    Distinct from `GroundedAnswer.cited_numbers` (which is deduplicated
    to unique citation numbers): the same `citation_number` can appear
    in multiple occurrences (`[1]` used twice), and each use is verified
    independently since the surrounding claim can differ each time.
    `occurrence_id` is 1-based, contiguous, and assigned strictly in
    left-to-right appearance order in the answer text -- it is the
    identity `CitationVerification` results are matched back against,
    never the (repeatable) `citation_number` alone. `start_offset`/
    `end_offset` delimit exactly the bracket substring itself (e.g.
    `answer_text[start_offset:end_offset] == "[1]"`), so the raw text
    can be recovered without re-parsing.
    """

    occurrence_id: int
    citation_number: int
    start_offset: int
    end_offset: int


class CitationVerdict(StrEnum):
    """A citation judge's verdict for one citation occurrence's factual claim.

    Never a calibrated confidence probability -- these are discrete,
    qualitative categories, and no numeric score is derived from them
    anywhere in this phase.
    """

    SUPPORTED = "supported"
    """The cited evidence directly supports all material factual details
    associated with this citation occurrence."""

    PARTIALLY_SUPPORTED = "partially_supported"
    """The evidence supports part of the associated claim, but at least
    one material detail goes beyond what the evidence establishes."""

    UNSUPPORTED = "unsupported"
    """The evidence does not establish the associated claim."""

    CONTRADICTED = "contradicted"
    """The evidence explicitly conflicts with one or more material
    details of the associated claim."""


@dataclass(frozen=True, slots=True)
class CitationVerification:
    """A citation judge's verdict for exactly one `CitationOccurrence`.

    `occurrence_id`/`citation_number` mirror the occurrence being judged
    -- they are taken from the deterministic occurrence, never from
    whatever a judge freely returns (see
    `verification.verify_grounded_answer`, which validates a judge's raw
    output against the expected occurrence set before a
    `CitationVerification` is ever constructed). `rationale` is
    required, non-empty free text explaining the verdict; it is never a
    confidence probability.
    """

    occurrence_id: int
    citation_number: int
    verdict: CitationVerdict
    rationale: str
    chunk_id: str
    judge_model: str | None = None


@dataclass(frozen=True, slots=True)
class CitationVerificationReport:
    """The full per-occurrence citation-support verification result for one `GroundedAnswer`.

    `occurrences` and `verifications` are aligned 1:1 by
    `occurrence_id` -- for N citation occurrences there are exactly N of
    each, or both are empty (the fixed insufficient-evidence response
    has zero citations, so it gets an empty report without ever calling
    a judge; see `verification.verify_grounded_answer`).

    The derived counts and `all_supported` below are a factual TALLY of
    verdicts, not a confidence score or an accept/reject policy decision
    -- deciding what to do with these numbers (abstain, warn the user,
    etc.) is explicitly a later phase, not this one. In particular,
    `all_supported` is vacuously `True` whenever `total_occurrences` is
    zero (including a genuinely empty/malformed report as well as the
    recognized insufficient-evidence response), so `all_supported` alone
    is NOT sufficient for any future confidence/acceptance judgment --
    that would also need to inspect `total_occurrences` and whether the
    answer is the recognized insufficiency response (see
    `prompt.is_insufficient_evidence_answer`).
    """

    grounded_answer: GroundedAnswer
    occurrences: tuple[CitationOccurrence, ...]
    verifications: tuple[CitationVerification, ...]

    @property
    def total_occurrences(self) -> int:
        return len(self.verifications)

    @property
    def supported_count(self) -> int:
        return sum(1 for v in self.verifications if v.verdict is CitationVerdict.SUPPORTED)

    @property
    def partially_supported_count(self) -> int:
        return sum(
            1 for v in self.verifications if v.verdict is CitationVerdict.PARTIALLY_SUPPORTED
        )

    @property
    def unsupported_count(self) -> int:
        return sum(1 for v in self.verifications if v.verdict is CitationVerdict.UNSUPPORTED)

    @property
    def contradicted_count(self) -> int:
        return sum(1 for v in self.verifications if v.verdict is CitationVerdict.CONTRADICTED)

    @property
    def all_supported(self) -> bool:
        """True iff every occurrence verified as SUPPORTED (vacuously True for zero occurrences).

        A factual aggregate over verdicts only -- not a confidence score
        or a final accept/reject decision. Because it is vacuously
        `True` when `total_occurrences == 0`, callers must not treat
        `all_supported` alone as "this answer is trustworthy": a report
        with zero occurrences could be the recognized
        insufficient-evidence response (fine) or, in principle, an
        otherwise-empty report from some other caller-constructed
        `GroundedAnswer` -- always check `total_occurrences` (and, where
        relevant, `prompt.is_insufficient_evidence_answer`) alongside
        this property.
        """
        return all(v.verdict is CitationVerdict.SUPPORTED for v in self.verifications)
