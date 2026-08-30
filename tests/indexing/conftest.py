"""Shared fixtures for indexing tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_pipeline.chunking.models import Chunk, build_chunk
from rag_pipeline.config import ChunkingStrategy, Settings

_EMBEDDING_DIM = 32


def _hash_vector(text: str, dim: int = _EMBEDDING_DIM) -> list[float]:
    """Deterministic pseudo-embedding for `text`, centered in [-1.0, 1.0].

    Centered (not [0.0, 1.0]) and using the full 32-byte SHA-256 digest
    deliberately: an all-positive embedding space gives *unrelated*
    vectors a high baseline cosine similarity purely from sharing a
    positive "DC" component (measured mean ~0.76 for an 8-dim, [0, 1]
    scheme across distinct short texts, with some pairs already exceeding
    0.95 by chance) -- which would make near-duplicate detection trigger
    spuriously against these fixtures once deduplication is wired into
    `index_chunks()`. Centering + the larger dimension keeps observed
    similarity between distinct texts comfortably below the default 0.95
    dedup threshold (measured max ~0.65 across 300 distinct synthetic
    texts).
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [(digest[i] - 127.5) / 127.5 for i in range(dim)]


class FakeEmbeddingProvider:
    """Deterministic, network-free stub embedding provider.

    Each text maps to a vector derived from its SHA-256 digest (see
    `_hash_vector`) — deterministic, reproducible across calls/processes,
    with enough spread that distinct texts get distinct, low-similarity
    vectors.
    """

    def __init__(self, dim: int = _EMBEDDING_DIM) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [_hash_vector(text, self.dim) for text in texts]


class ForcedSimilarityEmbeddingProvider:
    """A `FakeEmbeddingProvider` that forces a chosen set of texts to share one embedding.

    Used to deterministically engineer a near-duplicate scenario in
    integration tests: every text in `forced_group` gets the exact same
    vector (cosine similarity 1.0 between them), while every other text
    gets its normal, independent hash-based vector.
    """

    def __init__(self, forced_group: set[str], dim: int = _EMBEDDING_DIM) -> None:
        self._forced_group = forced_group
        self._forced_vector = _hash_vector(sorted(forced_group)[0], dim)
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [
            list(self._forced_vector) if text in self._forced_group else _hash_vector(text)
            for text in texts
        ]


@pytest.fixture
def index_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, index_root_dir=tmp_path / "indexes")


def make_chunks(
    count: int,
    *,
    document_id: str = "d" * 64,
    source_file: str = "doc.md",
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    text_prefix: str = "Content about ERR_DB_1042 in chunk",
) -> list[Chunk]:
    return [
        build_chunk(
            document_id=document_id,
            chunk_index=index,
            text=f"{text_prefix} number {index}.",
            source_file=source_file,
            section_heading=None,
            page_number=None,
            strategy=strategy,
        )
        for index in range(count)
    ]
