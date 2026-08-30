"""Tests for rag_pipeline.retrieval.reranked (retrieve_reranked orchestration)."""

from __future__ import annotations

import pytest

from rag_pipeline.chunking.models import Chunk
from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.indexing import index_chunks
from rag_pipeline.reranking.exceptions import RerankerError
from rag_pipeline.retrieval import reranked as reranked_module
from rag_pipeline.retrieval.exceptions import (
    HybridRetrievalError,
    RerankedRetrievalError,
)
from rag_pipeline.retrieval.models import RerankedRetrievalResult
from rag_pipeline.retrieval.reranked import retrieve_reranked

from .conftest import DictEmbeddingProvider, FakeReranker, make_chunk, make_hybrid_result


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    hybrid_results: list | None = None,
) -> dict[str, list]:
    calls: dict[str, list] = {"hybrid": []}

    def fake_retrieve_hybrid(
        query,
        strategy,
        settings,
        embedding_provider=None,
        dense_top_k=None,
        sparse_top_k=None,
        hybrid_top_k=None,
    ):
        calls["hybrid"].append(
            {
                "query": query,
                "strategy": strategy,
                "dense_top_k": dense_top_k,
                "sparse_top_k": sparse_top_k,
                "hybrid_top_k": hybrid_top_k,
            }
        )
        return (
            hybrid_results
            if hybrid_results is not None
            else [make_hybrid_result(chunk_id="a", rank=1)]
        )

    monkeypatch.setattr(reranked_module, "retrieve_hybrid", fake_retrieve_hybrid)
    return calls


# --- orchestration: retrieve_hybrid is called, with the right candidate depth -----


def test_retrieve_reranked_invokes_retrieve_hybrid(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fakes(monkeypatch)
    retrieve_reranked(
        "a query", ChunkingStrategy.RECURSIVE, index_settings, FakeReranker({"text for a": 1.0})
    )
    assert len(calls["hybrid"]) == 1
    assert calls["hybrid"][0]["query"] == "a query"


def test_default_candidate_depth_requested_is_rerank_candidate_k(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fakes(monkeypatch)
    retrieve_reranked(
        "a query", ChunkingStrategy.RECURSIVE, index_settings, FakeReranker({"text for a": 1.0})
    )
    assert calls["hybrid"][0]["hybrid_top_k"] == index_settings.rerank_candidate_k == 20


def test_ordinary_hybrid_top_k_is_not_used_as_candidate_depth(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert index_settings.hybrid_top_k == 10
    calls = _install_fakes(monkeypatch)
    retrieve_reranked(
        "a query", ChunkingStrategy.RECURSIVE, index_settings, FakeReranker({"text for a": 1.0})
    )
    assert calls["hybrid"][0]["hybrid_top_k"] != index_settings.hybrid_top_k
    assert calls["hybrid"][0]["hybrid_top_k"] == 20


def test_exactly_the_hybrid_candidates_returned_are_sent_to_reranker(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    hybrid_results = [
        make_hybrid_result(chunk_id="x", rank=1, text="text-x"),
        make_hybrid_result(chunk_id="y", rank=2, text="text-y"),
    ]
    _install_fakes(monkeypatch, hybrid_results=hybrid_results)
    reranker = FakeReranker({"text-x": 1.0, "text-y": 2.0})
    retrieve_reranked("a query", ChunkingStrategy.RECURSIVE, index_settings, reranker)
    assert reranker.calls[0][1] == ["text-x", "text-y"]


def test_final_default_top_k_is_five(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    hybrid_results = [
        make_hybrid_result(chunk_id=f"c{i}", rank=i, text=f"text-{i}") for i in range(1, 8)
    ]
    _install_fakes(monkeypatch, hybrid_results=hybrid_results)
    reranker = FakeReranker({f"text-{i}": float(i) for i in range(1, 8)})
    results = retrieve_reranked("a query", ChunkingStrategy.RECURSIVE, index_settings, reranker)
    assert index_settings.rerank_top_k == 5
    assert len(results) == 5


def test_candidate_depth_override_is_honored(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fakes(monkeypatch)
    retrieve_reranked(
        "a query",
        ChunkingStrategy.RECURSIVE,
        index_settings,
        FakeReranker({"text for a": 1.0}),
        candidate_k=42,
        final_top_k=5,
    )
    assert calls["hybrid"][0]["hybrid_top_k"] == 42


def test_final_depth_override_is_honored(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    hybrid_results = [
        make_hybrid_result(chunk_id=f"c{i}", rank=i, text=f"text-{i}") for i in range(1, 5)
    ]
    _install_fakes(monkeypatch, hybrid_results=hybrid_results)
    reranker = FakeReranker({f"text-{i}": float(i) for i in range(1, 5)})
    results = retrieve_reranked(
        "a query", ChunkingStrategy.RECURSIVE, index_settings, reranker, final_top_k=2
    )
    assert len(results) == 2


def test_retrieve_reranked_returns_reranked_retrieval_results(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fakes(monkeypatch)
    results = retrieve_reranked(
        "a query", ChunkingStrategy.RECURSIVE, index_settings, FakeReranker({"text for a": 1.0})
    )
    assert results
    assert all(isinstance(r, RerankedRetrievalResult) for r in results)


# --- failure semantics --------------------------------------------------------------


def test_hybrid_retrieval_failure_is_wrapped_with_cause_preserved(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args: object, **_kwargs: object):
        raise HybridRetrievalError("simulated hybrid failure")

    monkeypatch.setattr(reranked_module, "retrieve_hybrid", _boom)

    with pytest.raises(RerankedRetrievalError) as exc_info:
        retrieve_reranked("a query", ChunkingStrategy.RECURSIVE, index_settings, FakeReranker({}))
    assert isinstance(exc_info.value.__cause__, HybridRetrievalError)


def test_reranker_provider_failure_is_wrapped_with_cause_preserved(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fakes(monkeypatch)
    reranker = FakeReranker(error=RerankerError("simulated provider failure"))

    with pytest.raises(RerankedRetrievalError) as exc_info:
        retrieve_reranked("a query", ChunkingStrategy.RECURSIVE, index_settings, reranker)
    assert isinstance(exc_info.value.__cause__, RerankerError)


def test_no_extra_dense_or_sparse_retrieval_or_embedding_beyond_hybrid_retrieval(
    index_settings: Settings,
) -> None:
    # A real (small) index, with a tracking embedding provider shared by
    # both indexing and dense retrieval: if retrieve_reranked() ran any
    # extra dense/sparse retrieval or embedded the query itself (beyond
    # retrieve_hybrid()'s own single embedding call), this provider would
    # see more than one `.embed([query])` call.
    chunks: list[Chunk] = [
        make_chunk(chunk_index=0, text="alpha content", source_file="alpha.md"),
        make_chunk(chunk_index=1, text="beta content", source_file="beta.md"),
    ]
    vectors = {
        "alpha content": [1.0, 0.0],
        "beta content": [0.0, 1.0],
        "alpha query": [1.0, 0.0],
    }
    index_provider = DictEmbeddingProvider(vectors)
    index_chunks(chunks, index_settings, embedding_provider=index_provider)

    query_provider = DictEmbeddingProvider(vectors)
    reranker = FakeReranker({"alpha content": 1.0, "beta content": 1.0})
    retrieve_reranked(
        "alpha query",
        ChunkingStrategy.RECURSIVE,
        index_settings,
        reranker,
        embedding_provider=query_provider,
    )

    assert query_provider.calls == [["alpha query"]]


# --- configuration -----------------------------------------------------------------


def test_default_rerank_candidate_k_is_20() -> None:
    settings = Settings(_env_file=None)
    assert settings.rerank_candidate_k == 20


def test_default_rerank_top_k_is_5() -> None:
    settings = Settings(_env_file=None)
    assert settings.rerank_top_k == 5


def test_non_positive_rerank_candidate_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="rerank_candidate_k must be positive"):
        Settings(_env_file=None, rerank_candidate_k=0)


def test_non_positive_rerank_top_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="rerank_top_k must be positive"):
        Settings(_env_file=None, rerank_top_k=0)


def test_rerank_top_k_greater_than_candidate_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="rerank_top_k must not exceed rerank_candidate_k"):
        Settings(_env_file=None, rerank_candidate_k=5, rerank_top_k=10)


def test_hybrid_top_k_remains_independently_configurable() -> None:
    settings = Settings(_env_file=None, hybrid_top_k=3)
    assert settings.hybrid_top_k == 3
    assert settings.rerank_candidate_k == 20
    assert settings.rerank_top_k == 5


def test_default_hybrid_top_k_is_still_10() -> None:
    settings = Settings(_env_file=None)
    assert settings.hybrid_top_k == 10
