"""Typed, immutable models for dense and sparse retrieval results."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ChunkingStrategy


@dataclass(frozen=True, slots=True)
class DenseRetrievalResult:
    """One ranked dense-retrieval hit, with full provenance back to its source chunk.

    `rank` starts at 1 and reflects Chroma's own nearest-neighbor ordering
    (never independently re-sorted). `distance` is the raw cosine distance
    Chroma returned; `similarity` is the derived, human-friendly
    `1.0 - distance` -- both are kept, since smaller distance and larger
    similarity mean the same thing ("better match") but are not
    interchangeable values.
    """

    chunk_id: str
    rank: int
    text: str
    distance: float
    similarity: float
    document_id: str
    chunk_index: int
    source_file: str
    section_heading: str | None
    page_number: int | None
    chunking_strategy: ChunkingStrategy


@dataclass(frozen=True, slots=True)
class SparseRetrievalResult:
    """One ranked BM25 sparse-retrieval hit, with full provenance back to its source chunk.

    `rank` starts at 1 and reflects descending `bm25_score` order, with
    ties broken by ascending canonical sparse-corpus position -- never an
    independent re-sort by chunk_id or any other field. `bm25_score` is
    the raw, unnormalized score from `BM25Okapi.get_scores()`: higher is
    better, negative finite scores are valid, and it is never combined
    with or compared against dense cosine similarity/distance.
    """

    chunk_id: str
    rank: int
    text: str
    bm25_score: float
    document_id: str
    chunk_index: int
    source_file: str
    section_heading: str | None
    page_number: int | None
    chunking_strategy: ChunkingStrategy


@dataclass(frozen=True, slots=True)
class HybridRetrievalResult:
    """One ranked hybrid hit, fusing independent dense and sparse rankings via weighted RRF.

    `rank` starts at 1 and reflects final RRF ordering (`rrf_score` DESC,
    then deterministic rank-based tie-breaking -- see `retrieval.fusion`);
    it is assigned only after fusion and truncation, and is independent of
    `dense_rank`/`sparse_rank`, which are `None` when the chunk was not
    returned by that channel at all.

    `rrf_score` is computed purely from `dense_rank`/`sparse_rank`
    positions (see `dense_contribution`/`sparse_contribution`) -- never
    from `dense_distance`/`dense_similarity`/`bm25_score`, which are
    retained here only for diagnostics/provenance and are never summed,
    normalized, or compared against each other.
    """

    chunk_id: str
    rank: int
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None
    dense_contribution: float
    sparse_contribution: float
    dense_distance: float | None
    dense_similarity: float | None
    bm25_score: float | None
    text: str
    document_id: str
    chunk_index: int
    source_file: str
    section_heading: str | None
    page_number: int | None
    chunking_strategy: ChunkingStrategy
