"""Tests for rag_pipeline.chunking.dispatcher."""

import pytest

from rag_pipeline.chunking.dispatcher import chunk_document
from rag_pipeline.chunking.exceptions import UnsupportedChunkingStrategyError
from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.ingestion.models import Segment

from .conftest import FakeEmbeddingProvider, make_document


def test_fixed_strategy_resolves_correctly() -> None:
    document = make_document(
        segments=(Segment(text="x" * 250, section_heading=None, page_number=None),)
    )
    settings = Settings(_env_file=None, chunk_size=100, chunk_overlap=20)
    chunks = chunk_document(document, strategy=ChunkingStrategy.FIXED, settings=settings)
    assert chunks
    assert all(c.chunking_strategy == ChunkingStrategy.FIXED for c in chunks)


def test_recursive_strategy_resolves_correctly() -> None:
    document = make_document(
        segments=(
            Segment(
                text="\n\n".join(f"Paragraph {i}." for i in range(20)),
                section_heading=None,
                page_number=None,
            ),
        )
    )
    settings = Settings(_env_file=None, chunk_size=100, chunk_overlap=20)
    chunks = chunk_document(document, strategy=ChunkingStrategy.RECURSIVE, settings=settings)
    assert chunks
    assert all(c.chunking_strategy == ChunkingStrategy.RECURSIVE for c in chunks)


def test_semantic_strategy_resolves_correctly() -> None:
    document = make_document(
        segments=(Segment(text="Alpha.\n\nBeta.", section_heading=None, page_number=None),)
    )
    settings = Settings(_env_file=None, chunk_size=1000)
    chunks = chunk_document(
        document,
        strategy=ChunkingStrategy.SEMANTIC,
        settings=settings,
        embedding_provider=FakeEmbeddingProvider(),
    )
    assert chunks
    assert all(c.chunking_strategy == ChunkingStrategy.SEMANTIC for c in chunks)


def test_default_strategy_comes_from_settings() -> None:
    document = make_document(
        segments=(Segment(text="x" * 250, section_heading=None, page_number=None),)
    )
    settings = Settings(
        _env_file=None, chunk_strategy=ChunkingStrategy.FIXED, chunk_size=100, chunk_overlap=20
    )
    chunks = chunk_document(document, settings=settings)
    assert all(c.chunking_strategy == ChunkingStrategy.FIXED for c in chunks)


def test_invalid_strategy_cannot_silently_fall_through() -> None:
    document = make_document(
        segments=(Segment(text="x" * 50, section_heading=None, page_number=None),)
    )
    settings = Settings(_env_file=None)
    with pytest.raises(UnsupportedChunkingStrategyError):
        chunk_document(document, strategy="not-a-real-strategy", settings=settings)  # type: ignore[arg-type]
