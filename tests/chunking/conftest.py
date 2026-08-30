"""Shared fixtures for chunking tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from rag_pipeline.config import Settings
from rag_pipeline.ingestion.models import NormalizedDocument, Segment


def make_document(
    *,
    document_id: str = "d" * 64,
    source_file: str = "doc.txt",
    file_type: str = "txt",
    segments: tuple[Segment, ...],
) -> NormalizedDocument:
    return NormalizedDocument(
        document_id=document_id,
        source_file=source_file,
        file_type=file_type,
        raw_path=f"{document_id}/{source_file}",
        segments=segments,
    )


class FakeEmbeddingProvider:
    """Deterministic stub embedding provider — no network, no API key.

    Each text is mapped to a 2D vector via `vector_for`. By default, every
    text maps to the same vector (forcing similarity == 1.0, i.e. no
    boundary); tests override `vector_for` to force specific boundaries.
    """

    def __init__(self, vector_for: Callable[[str], list[float]] | None = None) -> None:
        self._vector_for = vector_for or (lambda _text: [1.0, 0.0])
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector_for(text) for text in texts]


@pytest.fixture
def test_settings() -> Settings:
    return Settings(_env_file=None, chunk_size=100, chunk_overlap=20)
