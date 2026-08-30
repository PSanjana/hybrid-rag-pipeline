"""Shared embedding-vector validation.

Used by `OpenAIEmbeddingProvider.embed()` to validate raw API responses,
and by `rag_pipeline.deduplication` to validate precomputed embeddings
passed into `deduplicate_chunks()` -- one set of checks, not two.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .exceptions import EmbeddingProviderError


def validate_vector(vector: Sequence[float]) -> None:
    """Raise `EmbeddingProviderError` unless `vector` is non-empty and every value is finite."""
    if not vector:
        raise EmbeddingProviderError("Embedding provider returned an empty vector.")
    if not all(math.isfinite(value) for value in vector):
        raise EmbeddingProviderError("Embedding provider returned a non-finite value.")


def validate_consistent_dimensionality(vectors: Sequence[Sequence[float]]) -> None:
    """Raise `EmbeddingProviderError` unless every vector in `vectors` shares one dimension."""
    if not vectors:
        return
    expected_dim = len(vectors[0])
    for vector in vectors:
        if len(vector) != expected_dim:
            raise EmbeddingProviderError(
                f"Inconsistent embedding dimensionality within one call: expected "
                f"{expected_dim}, got {len(vector)}."
            )
