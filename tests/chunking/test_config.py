"""Tests for chunking-related fields on rag_pipeline.config.Settings."""

import pytest

from rag_pipeline.config import ChunkingStrategy, Settings


def test_default_strategy_is_valid() -> None:
    settings = Settings(_env_file=None)
    assert settings.chunk_strategy in set(ChunkingStrategy)


def test_default_chunk_size_and_overlap() -> None:
    settings = Settings(_env_file=None)
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200


def test_strategy_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_STRATEGY", "semantic")
    settings = Settings(_env_file=None)
    assert settings.chunk_strategy == ChunkingStrategy.SEMANTIC


def test_chunk_size_and_overlap_are_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")
    settings = Settings(_env_file=None)
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50


def test_non_positive_chunk_size_rejected() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        Settings(_env_file=None, chunk_size=0)


def test_negative_chunk_overlap_rejected() -> None:
    with pytest.raises(ValueError, match="chunk_overlap must not be negative"):
        Settings(_env_file=None, chunk_overlap=-1)


def test_overlap_equal_to_chunk_size_rejected() -> None:
    with pytest.raises(ValueError, match="strictly smaller"):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_overlap_larger_than_chunk_size_rejected() -> None:
    with pytest.raises(ValueError, match="strictly smaller"):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=150)


def test_similarity_threshold_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="chunk_semantic_similarity_threshold"):
        Settings(_env_file=None, chunk_semantic_similarity_threshold=1.5)
