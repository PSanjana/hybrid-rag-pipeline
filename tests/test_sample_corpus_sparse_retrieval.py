"""Sparse (BM25) retrieval over the synthetic Acme Cloud sample corpus (offline).

data/sample -> ingest -> recursive chunking -> indexing -> retrieve_sparse.

Unlike dense retrieval, BM25 scores real lexical/token overlap, so no
intentionally-engineered fake embeddings are needed here for the ranking
itself -- sparse retrieval never touches embeddings. A generic deterministic
hash-based embedding provider is used only to satisfy `index_chunks()`'s
requirement of building a dense side too. No network/OpenAI calls.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_pipeline.chunking import ChunkingStrategy, chunk_document
from rag_pipeline.config import Settings
from rag_pipeline.indexing import index_chunks
from rag_pipeline.ingestion import ingest_document
from rag_pipeline.retrieval import retrieve_sparse

SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "data" / "sample"
_SUPPORTED_EXTENSIONS = {".md", ".txt", ".html", ".pdf"}


class FakeEmbeddingProvider:
    """Deterministic, network-free embedding stub -- values are irrelevant to sparse retrieval."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [(byte - 127.5) / 127.5 for byte in hashlib.sha256(text.encode()).digest()]
            for text in texts
        ]


@pytest.fixture
def pipeline_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        index_root_dir=tmp_path / "indexes",
    )


def _sample_files() -> list[Path]:
    return sorted(
        f
        for f in SAMPLE_ROOT.rglob("*")
        if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTENSIONS
    )


def _index_sample_corpus(settings: Settings) -> None:
    chunks = []
    for path in _sample_files():
        document = ingest_document(path, settings=settings)
        chunks.extend(
            chunk_document(document, strategy=ChunkingStrategy.RECURSIVE, settings=settings)
        )
    index_chunks(chunks, settings, embedding_provider=FakeEmbeddingProvider())


def test_err_db_1042_query_ranks_database_material_at_the_top(pipeline_settings: Settings) -> None:
    _index_sample_corpus(pipeline_settings)

    results = retrieve_sparse("ERR_DB_1042", ChunkingStrategy.RECURSIVE, pipeline_settings, top_k=5)

    assert results
    assert results[0].bm25_score > 0.0
    assert "err_db_1042" in results[0].text.lower()
    scores = [r.bm25_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_auth_token_ttl_query_ranks_authentication_material_highly(
    pipeline_settings: Settings,
) -> None:
    _index_sample_corpus(pipeline_settings)

    results = retrieve_sparse(
        "AUTH_TOKEN_TTL", ChunkingStrategy.RECURSIVE, pipeline_settings, top_k=5
    )

    assert results
    assert results[0].bm25_score > 0.0
    assert "auth_token_ttl" in results[0].text.lower()


def test_result_provenance_points_to_real_sample_source_files(pipeline_settings: Settings) -> None:
    _index_sample_corpus(pipeline_settings)

    results = retrieve_sparse("ERR_DB_1042", ChunkingStrategy.RECURSIVE, pipeline_settings, top_k=5)

    sample_filenames = {f.name for f in _sample_files()}
    assert results
    for result in results:
        assert result.source_file in sample_filenames
