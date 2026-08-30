"""Reranked retrieval over the synthetic Acme Cloud sample corpus (offline).

data/sample -> ingest -> recursive chunking -> indexing -> retrieve_hybrid
(top rerank_candidate_k) -> retrieve_reranked (controlled fake reranker).

Reuses the deterministic, intentionally-engineered bag-of-terms fake
embedding provider from test_sample_corpus_retrieval.py, and the
deterministic FakeReranker test double from tests/retrieval/conftest.py.
No network/OpenAI calls, and no real cross-encoder model is loaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_pipeline.chunking import ChunkingStrategy, chunk_document
from rag_pipeline.config import Settings
from rag_pipeline.indexing import index_chunks
from rag_pipeline.ingestion import ingest_document
from rag_pipeline.retrieval import retrieve_hybrid, retrieve_reranked

from .retrieval.conftest import FakeReranker
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


def _uniform_low_score_reranker(hybrid_results, boosted_chunk_id: str, boosted_score: float):
    scores_by_text = {r.text: 0.01 for r in hybrid_results}
    boosted = next(r for r in hybrid_results if r.chunk_id == boosted_chunk_id)
    scores_by_text[boosted.text] = boosted_score
    return FakeReranker(scores_by_text)


def test_controlled_reranker_promotes_most_relevant_db_chunk_to_final_rank_one(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    query = "database connection pool ERR_DB_1042"

    # Establish the pre-rerank hybrid candidate pool (top rerank_candidate_k,
    # i.e. 20) to identify the chunk that actually documents the error code
    # (its section heading names ERR_DB_1042 directly) but is *not* RRF's
    # top pick -- RRF fuses rank positions only, so it has no notion of
    # "this chunk is the canonical explanation."
    candidates = retrieve_hybrid(
        query,
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        embedding_provider=provider,
        hybrid_top_k=pipeline_settings.rerank_candidate_k,
    )
    canonical = next(
        c for c in candidates if c.section_heading and "ERR_DB_1042" in c.section_heading
    )
    assert canonical.rank > 1, "expected the canonical chunk to not already be RRF's top pick"

    reranker = _uniform_low_score_reranker(candidates, canonical.chunk_id, boosted_score=99.0)
    results = retrieve_reranked(
        query,
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        reranker,
        embedding_provider=provider,
    )

    assert results[0].chunk_id == canonical.chunk_id
    assert results[0].rank == 1
    assert results[0].hybrid_rank == canonical.rank


def test_candidate_set_is_drawn_from_top_rerank_candidate_k_hybrid_results(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    query = "database connection pool ERR_DB_1042"
    candidates = retrieve_hybrid(
        query,
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        embedding_provider=provider,
        hybrid_top_k=pipeline_settings.rerank_candidate_k,
    )
    reranker = FakeReranker({c.text: float(c.rank) for c in candidates})
    results = retrieve_reranked(
        query,
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        reranker,
        embedding_provider=provider,
    )

    candidate_chunk_ids = {c.chunk_id for c in candidates}
    assert {r.chunk_id for r in results} <= candidate_chunk_ids
    assert all(r.hybrid_rank <= pipeline_settings.rerank_candidate_k for r in results)


def test_final_result_count_is_at_most_five(pipeline_settings: Settings) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    query = "database connection pool ERR_DB_1042"
    candidates = retrieve_hybrid(
        query,
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        embedding_provider=provider,
        hybrid_top_k=pipeline_settings.rerank_candidate_k,
    )
    reranker = FakeReranker({c.text: float(c.rank) for c in candidates})
    results = retrieve_reranked(
        query,
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        reranker,
        embedding_provider=provider,
    )

    assert 0 < len(results) <= 5
    assert results[0].rank == 1
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_prior_hybrid_ranks_remain_observable_on_final_results(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    query = "database connection pool ERR_DB_1042"
    candidates = retrieve_hybrid(
        query,
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        embedding_provider=provider,
        hybrid_top_k=pipeline_settings.rerank_candidate_k,
    )
    hybrid_rank_by_id = {c.chunk_id: c.rank for c in candidates}
    reranker = FakeReranker({c.text: float(c.rank) for c in candidates})
    results = retrieve_reranked(
        query,
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        reranker,
        embedding_provider=provider,
    )

    for result in results:
        assert result.hybrid_rank == hybrid_rank_by_id[result.chunk_id]


def test_final_result_provenance_points_to_real_sample_sources(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    query = "database connection pool ERR_DB_1042"
    candidates = retrieve_hybrid(
        query,
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        embedding_provider=provider,
        hybrid_top_k=pipeline_settings.rerank_candidate_k,
    )
    reranker = FakeReranker({c.text: float(c.rank) for c in candidates})
    results = retrieve_reranked(
        query,
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        reranker,
        embedding_provider=provider,
    )

    sample_filenames = {f.name for f in _sample_files()}
    assert results
    for result in results:
        assert result.source_file in sample_filenames


def test_auth_token_ttl_is_outside_default_hybrid_top_k_but_inside_rerank_candidate_k(
    pipeline_settings: Settings,
) -> None:
    # Regression demonstration: the ordinary hybrid_top_k default (10)
    # would silently drop this chunk (see
    # test_sample_corpus_hybrid_retrieval.py's
    # test_auth_token_ttl_query_gets_strong_lexical_contribution -- its
    # dense signal is near-noise for this single underscore-glued
    # identifier, so it lands at hybrid rank 11-12, just past the
    # ordinary top-10 cutoff). rerank_candidate_k (20) is wide enough to
    # keep it eligible for reranking regardless.
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    candidates = retrieve_hybrid(
        "AUTH_TOKEN_TTL",
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        embedding_provider=provider,
        hybrid_top_k=pipeline_settings.rerank_candidate_k,
    )
    identifier_hits = [c for c in candidates if "auth_token_ttl" in c.text.lower()]
    assert identifier_hits, "expected AUTH_TOKEN_TTL chunk(s) within rerank_candidate_k"
    for hit in identifier_hits:
        assert hit.rank > 10, "expected the identifier chunk to fall outside the ordinary top 10"
        assert hit.rank <= pipeline_settings.rerank_candidate_k


def test_controlled_reranker_promotes_auth_token_ttl_chunk_into_final_top_five(
    pipeline_settings: Settings,
) -> None:
    provider = ConceptEmbeddingProvider()
    _index_sample_corpus(pipeline_settings, provider)

    query = "AUTH_TOKEN_TTL"
    candidates = retrieve_hybrid(
        query,
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        embedding_provider=provider,
        hybrid_top_k=pipeline_settings.rerank_candidate_k,
    )
    target = next(c for c in candidates if "auth_token_ttl" in c.text.lower())
    assert target.rank > pipeline_settings.hybrid_top_k

    reranker = _uniform_low_score_reranker(candidates, target.chunk_id, boosted_score=50.0)
    results = retrieve_reranked(
        query,
        ChunkingStrategy.RECURSIVE,
        pipeline_settings,
        reranker,
        embedding_provider=provider,
    )

    assert results[0].chunk_id == target.chunk_id
    assert results[0].rank == 1
    assert results[0].hybrid_rank == target.rank
    assert results[0].hybrid_rank > pipeline_settings.hybrid_top_k
