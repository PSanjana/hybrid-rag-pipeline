"""Shared fixtures/helpers for retrieval tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from rag_pipeline.chunking.models import Chunk, build_chunk
from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.retrieval.models import (
    DenseRetrievalResult,
    HybridRetrievalResult,
    SparseRetrievalResult,
)

_HASH_EMBEDDING_DIM = 32


class HashEmbeddingProvider:
    """Deterministic, network-free embedding stub for tests that don't care about vector values.

    Used to build a valid dense+sparse index for sparse-retrieval tests
    (`index_chunks()` always builds both sides), even though sparse
    retrieval itself never touches embeddings. Centered to [-1.0, 1.0]
    (not [0.0, 1.0]) and using the full 32-byte digest for the same reason
    as `tests/indexing/conftest.py`'s `FakeEmbeddingProvider`: an
    all-positive embedding space gives unrelated vectors a high baseline
    cosine similarity, which could make deduplication spuriously trigger
    during indexing.
    """

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([(digest[i] - 127.5) / 127.5 for i in range(_HASH_EMBEDDING_DIM)])
        return vectors


class DictEmbeddingProvider:
    """Deterministic embedding provider driven by an explicit text -> vector mapping.

    Every text passed to `embed()` must have an exact entry in `vectors` --
    a `KeyError` makes a missing/mistyped test fixture obvious rather than
    silently falling back to an unrelated vector. Used so retrieval tests
    can engineer exact, intentional nearest-neighbor relationships instead
    of relying on incidental hash-based similarity.
    """

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [list(self._vectors[text]) for text in texts]


@pytest.fixture
def index_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, index_root_dir=tmp_path / "indexes")


def make_chunk(
    *,
    document_id: str = "d" * 64,
    chunk_index: int = 0,
    text: str = "content",
    source_file: str = "doc.md",
    section_heading: str | None = None,
    page_number: int | None = None,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
) -> Chunk:
    return build_chunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        source_file=source_file,
        section_heading=section_heading,
        page_number=page_number,
        strategy=strategy,
    )


def make_dense_result(
    *,
    chunk_id: str,
    rank: int,
    text: str | None = None,
    distance: float = 0.1,
    document_id: str = "d" * 64,
    chunk_index: int = 0,
    source_file: str = "doc.md",
    section_heading: str | None = None,
    page_number: int | None = None,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
) -> DenseRetrievalResult:
    return DenseRetrievalResult(
        chunk_id=chunk_id,
        rank=rank,
        text=text if text is not None else f"text for {chunk_id}",
        distance=distance,
        similarity=1.0 - distance,
        document_id=document_id,
        chunk_index=chunk_index,
        source_file=source_file,
        section_heading=section_heading,
        page_number=page_number,
        chunking_strategy=chunking_strategy,
    )


def make_sparse_result(
    *,
    chunk_id: str,
    rank: int,
    text: str | None = None,
    bm25_score: float = 1.0,
    document_id: str = "d" * 64,
    chunk_index: int = 0,
    source_file: str = "doc.md",
    section_heading: str | None = None,
    page_number: int | None = None,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
) -> SparseRetrievalResult:
    return SparseRetrievalResult(
        chunk_id=chunk_id,
        rank=rank,
        text=text if text is not None else f"text for {chunk_id}",
        bm25_score=bm25_score,
        document_id=document_id,
        chunk_index=chunk_index,
        source_file=source_file,
        section_heading=section_heading,
        page_number=page_number,
        chunking_strategy=chunking_strategy,
    )


def make_hybrid_result(
    *,
    chunk_id: str,
    rank: int,
    rrf_score: float = 0.01,
    text: str | None = None,
    dense_rank: int | None = 1,
    sparse_rank: int | None = None,
    dense_contribution: float = 0.01,
    sparse_contribution: float = 0.0,
    dense_distance: float | None = 0.1,
    dense_similarity: float | None = 0.9,
    bm25_score: float | None = None,
    document_id: str = "d" * 64,
    chunk_index: int = 0,
    source_file: str = "doc.md",
    section_heading: str | None = None,
    page_number: int | None = None,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
) -> HybridRetrievalResult:
    return HybridRetrievalResult(
        chunk_id=chunk_id,
        rank=rank,
        rrf_score=rrf_score,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
        dense_contribution=dense_contribution,
        sparse_contribution=sparse_contribution,
        dense_distance=dense_distance,
        dense_similarity=dense_similarity,
        bm25_score=bm25_score,
        text=text if text is not None else f"text for {chunk_id}",
        document_id=document_id,
        chunk_index=chunk_index,
        source_file=source_file,
        section_heading=section_heading,
        page_number=page_number,
        chunking_strategy=chunking_strategy,
    )


class FakeReranker:
    """Deterministic, network-free `Reranker` double for offline reranking tests.

    `scores_by_text` maps document text -> the score to return for it,
    looked up in the exact order `documents` is received (so callers can
    assert on `.calls` to verify candidate/hybrid-rank ordering was
    preserved). `override_scores`, if given, is returned verbatim instead
    of consulting `scores_by_text` -- lets tests simulate malformed
    provider output (too few/many scores, non-numeric, NaN/inf) without
    needing a matching text entry for every candidate. `error`, if given,
    is raised instead of scoring at all (simulating a provider failure).
    """

    def __init__(
        self,
        scores_by_text: dict[str, float] | None = None,
        *,
        override_scores: list[object] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._scores_by_text = scores_by_text or {}
        self._override_scores = override_scores
        self._error = error
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        if self._error is not None:
            raise self._error
        if self._override_scores is not None:
            return cast(list[float], self._override_scores)
        return [self._scores_by_text[document] for document in documents]
