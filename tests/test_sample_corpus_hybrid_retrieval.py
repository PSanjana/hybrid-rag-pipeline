"""Hybrid (RRF-fused) retrieval over the synthetic Acme Cloud sample corpus (offline).

data/sample -> ingest -> recursive chunking -> indexing -> retrieve_hybrid.

Reuses the deterministic, intentionally-engineered bag-of-terms fake
embedding provider from test_sample_corpus_retrieval.py so dense
contributions are meaningful (not just incidental hash similarity) even
for a semantically-phrased query that never contains the exact
identifier. Sparse (BM25) scoring runs against the real corpus text
unmodified. No network/OpenAI calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_pipeline.chunking import ChunkingStrategy, chunk_document
from rag_pipeline.config import Settings
from rag_pipeline.indexing import index_chunks
from rag_pipeline.ingestion import ingest_document
from rag_pipeline.retrieval import retrieve_hybrid

from .test_sample_corpus_retrieval import (
    _SUPPORTED_EXTENSIONS,
    SAMPLE_ROOT,
    ConceptEmbeddingProvider,
)


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


def _index_sample_corpus(settings: Settings, provider: ConceptEmbeddingProvider) -> None:
    chunks = []
    for path in _sample_files():
        document = ingest_document(path, settings=settings)
        chunks.extend(
            chunk_document(document, strategy=ChunkingStrategy.RECURSIVE, settings=settings)
        )
    index_chunks(chunks, settings, embedding_provider=provider)


def test_err_db_1042_query_ranks_database_content_strongly(pipeline_settings: Settings) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    results = retrieve_hybrid(
        "ERR_DB_1042", ChunkingStrategy.RECURSIVE, pipeline_settings, embedding_provider=provider
    )

    assert results
    top = results[0]
    assert top.rrf_score > 0
    assert "err_db_1042" in top.text.lower() or "database" in top.source_file.lower()


def test_semantically_phrased_database_question_still_ranks_database_content(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    query = "What causes the database connection pool to become exhausted?"
    assert "ERR_DB_1042".lower() not in query.lower()

    results = retrieve_hybrid(
        query,
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        embedding_provider=provider,
        hybrid_top_k=5,
    )

    assert results
    top_sources = [r.source_file for r in results[:3]]
    assert any("database" in source.lower() for source in top_sources)
    # Dense contribution specifically must be doing real work here -- the
    # query never contains the literal ERR_DB_1042 identifier, so a strong
    # top result relies on the engineered term-vector "semantic" signal
    # (see ConceptEmbeddingProvider), not lexical overlap alone.
    assert results[0].dense_contribution > 0


def test_auth_token_ttl_query_gets_strong_lexical_contribution(pipeline_settings: Settings) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    # "AUTH_TOKEN_TTL" is a single underscore-joined identifier, so it
    # never matches any of ConceptEmbeddingProvider's individual
    # whole-word terms (e.g. "token") -- its *dense* vector is therefore
    # near-noise for this query. With dense_weight (0.7) > sparse_weight
    # (0.3), a full dense top-10 list can numerically outrank even a
    # sparse-rank-1 hit (0.7/61 per dense-only chunk vs 0.3/61 here), so
    # this chunk isn't guaranteed to land in a small top-k window -- a
    # generously large hybrid_top_k is used so it's captured regardless,
    # and what's actually asserted is the *strength of its lexical
    # contribution*, not its final position.
    results = retrieve_hybrid(
        "AUTH_TOKEN_TTL",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        embedding_provider=provider,
        hybrid_top_k=100,
    )

    assert results
    identifier_hits = [r for r in results if "auth_token_ttl" in r.text.lower()]
    assert identifier_hits, "expected at least one result whose text contains AUTH_TOKEN_TTL"
    hit = identifier_hits[0]
    assert hit.sparse_contribution > 0
    assert hit.bm25_score is not None
    assert hit.bm25_score > 0


def test_at_least_one_result_receives_contributions_from_both_retrievers(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    results = retrieve_hybrid(
        "database connection pool ERR_DB_1042",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        embedding_provider=provider,
        hybrid_top_k=10,
    )

    assert any(r.dense_rank is not None and r.sparse_rank is not None for r in results)


def test_all_result_provenance_points_to_real_sample_sources(pipeline_settings: Settings) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    results = retrieve_hybrid(
        "database connection pool",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        embedding_provider=provider,
        hybrid_top_k=10,
    )

    sample_filenames = {f.name for f in _sample_files()}
    assert results
    for result in results:
        assert result.source_file in sample_filenames
