"""Shared fixtures/helpers for retrieval tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_pipeline.chunking.models import Chunk, build_chunk
from rag_pipeline.config import ChunkingStrategy, Settings


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
