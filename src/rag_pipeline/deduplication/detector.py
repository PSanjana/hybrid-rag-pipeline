"""Deterministic exact + near-duplicate chunk detection.

Algorithm, given one canonically-ordered chunk corpus and one embedding per
chunk (same order):

  1. Walk chunks in canonical order. For each candidate:
     a. Exact check: if its `.text` exactly matches an already-*kept*
        chunk's text, skip it as an `EXACT` duplicate of that chunk. No
        cosine comparison is performed for this case -- text equality is
        cheap, unambiguous, and doesn't depend on embedding quality, so
        checking it first also avoids wasted cosine work for the common
        "identical paragraph copy-pasted across documents" case.
     b. Otherwise, compare its embedding against every currently-*kept*
        chunk's embedding via cosine similarity, and find the maximum.
        If that maximum is strictly greater than `threshold`, skip the
        candidate as a `NEAR` duplicate of whichever kept chunk produced
        it; otherwise keep the candidate.
  2. Skipped chunks are never added to the kept set and never compared
     against by later candidates -- only kept chunks anchor future
     comparisons, so which chunk is "canonical" for a duplicate chain is
     always the first-accepted chunk under canonical order, and stays
     stable regardless of how many duplicates follow.

Determinism: given the same chunks, order, embeddings, and threshold, the
kept/skipped sets and every recorded similarity are always identical --
there is no concurrency or randomness anywhere in this module.

Scalability: comparing each candidate against every previously-kept chunk
is worst-case O(N^2) cosine comparisons for a corpus of N chunks. This is
an explicit, accepted tradeoff for this project's current corpus size
(low hundreds of chunks) in exchange for a simple, fully-deterministic,
dependency-free implementation. It would not scale to a corpus of
hundreds of thousands of chunks without an approximate-nearest-neighbor
index (out of scope here -- see the project roadmap).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..chunking.models import Chunk
from ..embeddings.exceptions import EmbeddingProviderError
from ..embeddings.similarity import cosine_similarity
from ..embeddings.validation import validate_consistent_dimensionality, validate_vector
from .exceptions import DeduplicationError
from .models import DEDUP_ALGORITHM_VERSION, DeduplicationResult, DuplicateRecord, DuplicateType


def _validate_input(
    chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]], threshold: float
) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise DeduplicationError(
            f"dedup similarity threshold must be in [0.0, 1.0], got {threshold!r}."
        )
    if len(chunks) != len(embeddings):
        raise DeduplicationError(
            f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})."
        )
    try:
        for vector in embeddings:
            validate_vector(vector)
        validate_consistent_dimensionality(embeddings)
    except EmbeddingProviderError as exc:
        raise DeduplicationError(f"Invalid embedding input to deduplication: {exc}") from exc


def _build_record(
    skipped: Chunk, canonical: Chunk, duplicate_type: DuplicateType, similarity: float
) -> DuplicateRecord:
    return DuplicateRecord(
        skipped_chunk_id=skipped.chunk_id,
        canonical_chunk_id=canonical.chunk_id,
        duplicate_type=duplicate_type,
        similarity=similarity,
        skipped_document_id=skipped.document_id,
        canonical_document_id=canonical.document_id,
        skipped_chunk_index=skipped.chunk_index,
        canonical_chunk_index=canonical.chunk_index,
        skipped_source_file=skipped.source_file,
        canonical_source_file=canonical.source_file,
    )


def deduplicate_chunks(
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
    threshold: float,
) -> DeduplicationResult:
    """Filter `chunks` (in the given order) against exact + near-duplicate rules.

    `embeddings[i]` must be the embedding for `chunks[i]`; ordering is
    preserved and neither input is mutated. Uses strict `similarity >
    threshold` semantics, so a similarity exactly equal to `threshold` is
    kept, not flagged.
    """
    _validate_input(chunks, embeddings, threshold)

    kept_chunks: list[Chunk] = []
    kept_embeddings: list[list[float]] = []
    duplicates: list[DuplicateRecord] = []
    kept_text_to_chunk: dict[str, Chunk] = {}

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        exact_match = kept_text_to_chunk.get(chunk.text)
        if exact_match is not None:
            duplicates.append(_build_record(chunk, exact_match, DuplicateType.EXACT, 1.0))
            continue

        best_match: Chunk | None = None
        best_similarity = 0.0
        for kept_chunk, kept_embedding in zip(kept_chunks, kept_embeddings, strict=True):
            similarity = cosine_similarity(embedding, kept_embedding)
            if best_match is None or similarity > best_similarity:
                best_match = kept_chunk
                best_similarity = similarity

        if best_match is not None and best_similarity > threshold:
            duplicates.append(_build_record(chunk, best_match, DuplicateType.NEAR, best_similarity))
            continue

        kept_chunks.append(chunk)
        kept_embeddings.append(list(embedding))
        kept_text_to_chunk[chunk.text] = chunk

    return DeduplicationResult(
        kept_chunks=tuple(kept_chunks),
        kept_embeddings=tuple(tuple(vector) for vector in kept_embeddings),
        duplicates=tuple(duplicates),
        algorithm_version=DEDUP_ALGORITHM_VERSION,
        similarity_threshold=threshold,
    )
