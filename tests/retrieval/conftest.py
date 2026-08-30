"""Shared fixtures/helpers for retrieval tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_pipeline.chunking.models import Chunk, build_chunk
from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.retrieval.models import DenseRetrievalResult, SparseRetrievalResult

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
