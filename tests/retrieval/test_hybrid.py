"""Tests for rag_pipeline.retrieval.hybrid (retrieve_hybrid orchestration)."""

from __future__ import annotations

import pytest

from rag_pipeline.chunking.models import Chunk
from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.indexing import index_chunks
from rag_pipeline.retrieval import hybrid as hybrid_module
from rag_pipeline.retrieval.exceptions import (
    DenseRetrievalError,
    HybridRetrievalError,
    IndexNotReadyError,
    RetrievalError,
    SparseRetrievalError,
)
from rag_pipeline.retrieval.hybrid import retrieve_hybrid
from rag_pipeline.retrieval.models import HybridRetrievalResult

from .conftest import DictEmbeddingProvider, make_chunk, make_dense_result, make_sparse_result


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    dense_results: list | None = None,
    sparse_results: list | None = None,
) -> dict[str, list]:
    calls: dict[str, list] = {"dense": [], "sparse": [], "fuse": []}

    def fake_retrieve_dense(query, strategy, settings, embedding_provider=None, top_k=None):
        calls["dense"].append((query, strategy, top_k))
        return (
            dense_results
            if dense_results is not None
            else [make_dense_result(chunk_id="a", rank=1)]
        )

    def fake_retrieve_sparse(query, strategy, settings, top_k=None):
        calls["sparse"].append((query, strategy, top_k))
        return (
            sparse_results
            if sparse_results is not None
            else [make_sparse_result(chunk_id="a", rank=1)]
        )

    real_fuse_rankings = hybrid_module.fuse_rankings

    def tracking_fuse_rankings(dense, sparse, **kwargs):
        calls["fuse"].append({"dense": dense, "sparse": sparse, **kwargs})
        return real_fuse_rankings(dense, sparse, **kwargs)

    monkeypatch.setattr(hybrid_module, "retrieve_dense", fake_retrieve_dense)
    monkeypatch.setattr(hybrid_module, "retrieve_sparse", fake_retrieve_sparse)
    monkeypatch.setattr(hybrid_module, "fuse_rankings", tracking_fuse_rankings)
    return calls


# --- orchestration: dense/sparse are called with the actual outputs used ---------


def test_retrieve_hybrid_calls_dense_retriever(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fakes(monkeypatch)
    retrieve_hybrid("a query", ChunkingStrategy.RECURSIVE, index_settings)
    assert len(calls["dense"]) == 1
    assert calls["dense"][0][0] == "a query"
    assert calls["dense"][0][1] == ChunkingStrategy.RECURSIVE


def test_retrieve_hybrid_calls_sparse_retriever(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fakes(monkeypatch)
    retrieve_hybrid("a query", ChunkingStrategy.RECURSIVE, index_settings)
    assert len(calls["sparse"]) == 1
    assert calls["sparse"][0][0] == "a query"
    assert calls["sparse"][0][1] == ChunkingStrategy.RECURSIVE


def test_fusion_receives_the_actual_dense_and_sparse_ranked_outputs(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    dense_results = [make_dense_result(chunk_id="x", rank=1)]
    sparse_results = [make_sparse_result(chunk_id="y", rank=1)]
    calls = _install_fakes(monkeypatch, dense_results=dense_results, sparse_results=sparse_results)

    retrieve_hybrid("a query", ChunkingStrategy.RECURSIVE, index_settings)

    assert len(calls["fuse"]) == 1
    assert calls["fuse"][0]["dense"] is dense_results
    assert calls["fuse"][0]["sparse"] is sparse_results


def test_fusion_receives_configured_weights_and_rank_constant(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    settings = Settings(
        _env_file=None,
        index_root_dir=tmp_path / "indexes",
        rrf_dense_weight=0.4,
        rrf_sparse_weight=0.6,
        rrf_rank_constant=17,
        hybrid_top_k=3,
    )
    calls = _install_fakes(monkeypatch)

    retrieve_hybrid("a query", ChunkingStrategy.RECURSIVE, settings)

    assert len(calls["fuse"]) == 1
    fuse_call = calls["fuse"][0]
    assert fuse_call["dense_weight"] == pytest.approx(0.4)
    assert fuse_call["sparse_weight"] == pytest.approx(0.6)
    assert fuse_call["rank_constant"] == 17
    assert fuse_call["top_k"] == 3


def test_retrieve_hybrid_returns_hybrid_retrieval_results(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fakes(monkeypatch)
    results = retrieve_hybrid("a query", ChunkingStrategy.RECURSIVE, index_settings)
    assert results
    assert all(isinstance(r, HybridRetrievalResult) for r in results)


# --- failure semantics: both channels must succeed --------------------------------


def test_dense_failure_does_not_silently_degrade_to_sparse_only(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom_dense(*_args: object, **_kwargs: object):
        raise DenseRetrievalError("simulated dense failure")

    def fake_retrieve_sparse(query, strategy, settings, top_k=None):
        raise AssertionError("sparse retrieval should not run its own logic after dense failed")

    monkeypatch.setattr(hybrid_module, "retrieve_dense", _boom_dense)
    # sparse doesn't need to be reached at all if dense is called first,
    # but guard against a silent single-channel fallback regardless.

    with pytest.raises(HybridRetrievalError) as exc_info:
        retrieve_hybrid("a query", ChunkingStrategy.RECURSIVE, index_settings)

    assert isinstance(exc_info.value.__cause__, DenseRetrievalError)


def test_dense_failure_via_index_not_ready_is_wrapped(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom_dense(*_args: object, **_kwargs: object):
        raise IndexNotReadyError("simulated missing manifest")

    monkeypatch.setattr(hybrid_module, "retrieve_dense", _boom_dense)

    with pytest.raises(HybridRetrievalError) as exc_info:
        retrieve_hybrid("a query", ChunkingStrategy.RECURSIVE, index_settings)

    assert isinstance(exc_info.value.__cause__, IndexNotReadyError)


def test_sparse_failure_does_not_silently_degrade_to_dense_only(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_retrieve_dense(query, strategy, settings, embedding_provider=None, top_k=None):
        return [make_dense_result(chunk_id="a", rank=1)]

    def _boom_sparse(*_args: object, **_kwargs: object):
        raise SparseRetrievalError("simulated sparse failure")

    monkeypatch.setattr(hybrid_module, "retrieve_dense", fake_retrieve_dense)
    monkeypatch.setattr(hybrid_module, "retrieve_sparse", _boom_sparse)

    with pytest.raises(HybridRetrievalError) as exc_info:
        retrieve_hybrid("a query", ChunkingStrategy.RECURSIVE, index_settings)

    assert isinstance(exc_info.value.__cause__, SparseRetrievalError)


def test_hybrid_top_k_validated_before_calling_either_retriever(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args: object, **_kwargs: object):
        raise AssertionError("retrieval should not run for an invalid hybrid_top_k")

    monkeypatch.setattr(hybrid_module, "retrieve_dense", _boom)
    monkeypatch.setattr(hybrid_module, "retrieve_sparse", _boom)

    with pytest.raises(RetrievalError):
        retrieve_hybrid("a query", ChunkingStrategy.RECURSIVE, index_settings, hybrid_top_k=0)


# --- query is not independently embedded/tokenized by the hybrid layer -----------


def test_query_is_not_independently_embedded_by_the_hybrid_layer(index_settings: Settings) -> None:
    # A real (small) index, with a tracking embedding provider shared by
    # both indexing and dense retrieval: if retrieve_hybrid() embedded the
    # query itself (in addition to retrieve_dense()'s own embedding call),
    # this provider would see two `.embed([query])` calls instead of one.
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
    retrieve_hybrid(
        "alpha query",
        ChunkingStrategy.RECURSIVE,
        index_settings,
        embedding_provider=query_provider,
    )

    assert query_provider.calls == [["alpha query"]]
