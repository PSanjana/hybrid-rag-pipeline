"""Shared fixtures for indexing tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_pipeline.chunking.models import Chunk, build_chunk
from rag_pipeline.config import ChunkingStrategy, Settings

_EMBEDDING_DIM = 8


class FakeEmbeddingProvider:
    """Deterministic, network-free stub embedding provider.

    Each text maps to an 8-dimensional vector derived from its SHA-256
    digest — deterministic, reproducible across calls/processes, and with
    enough spread that distinct texts get distinct vectors.
    """

    def __init__(self, dim: int = _EMBEDDING_DIM) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([digest[i] / 255.0 for i in range(self.dim)])
        return vectors


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
