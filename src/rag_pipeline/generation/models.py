"""Typed, immutable models for grounded generation."""

from __future__ import annotations

from dataclasses import dataclass

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
