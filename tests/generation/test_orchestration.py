"""Tests for rag_pipeline.generation.service.retrieve_and_generate (retrieval + generation)."""

from __future__ import annotations

import pytest

from rag_pipeline.chunking.models import Chunk
from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.generation import service as service_module
from rag_pipeline.generation.exceptions import RetrieveAndGenerateError
from rag_pipeline.generation.models import GroundedAnswer
from rag_pipeline.generation.service import retrieve_and_generate
from rag_pipeline.indexing import index_chunks
from rag_pipeline.retrieval.exceptions import RerankedRetrievalError

from ..retrieval.conftest import DictEmbeddingProvider, FakeReranker, make_chunk
from .conftest import FakeGenerator, make_reranked_result


def _install_fake_retrieve_reranked(
    monkeypatch: pytest.MonkeyPatch, results: list | None = None
) -> dict[str, list]:
    calls: dict[str, list] = {"retrieve_reranked": []}

    def fake_retrieve_reranked(
        query,
        strategy,
        settings,
        reranker,
        embedding_provider=None,
        dense_top_k=None,
        sparse_top_k=None,
        candidate_k=None,
        final_top_k=None,
    ):
        calls["retrieve_reranked"].append({"query": query, "strategy": strategy})
        return results if results is not None else [make_reranked_result(chunk_id="a", rank=1)]

    monkeypatch.setattr(service_module, "retrieve_reranked", fake_retrieve_reranked)
    return calls


def test_retrieve_reranked_called_exactly_once(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_retrieve_reranked(monkeypatch)
    generator = FakeGenerator("Answer text [1].")
    retrieve_and_generate(
        "a question",
        ChunkingStrategy.RECURSIVE,
        index_settings,
        FakeReranker({}),
        generator,
    )
    assert len(calls["retrieve_reranked"]) == 1


def test_final_reranked_results_become_evidence(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = [
        make_reranked_result(chunk_id="a", rank=1, text="alpha"),
        make_reranked_result(chunk_id="b", rank=2, text="beta"),
    ]
    _install_fake_retrieve_reranked(monkeypatch, results=results)
    generator = FakeGenerator("alpha claim [1]. beta claim [2].")
    answer = retrieve_and_generate(
        "a question",
        ChunkingStrategy.RECURSIVE,
        index_settings,
        FakeReranker({}),
        generator,
    )
    assert [e.chunk_id for e in answer.evidence] == ["a", "b"]


def test_same_question_flows_into_retrieval_and_generation(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_retrieve_reranked(monkeypatch)
    generator = FakeGenerator("Answer text [1].")
    retrieve_and_generate(
        "what is the token TTL?",
        ChunkingStrategy.RECURSIVE,
        index_settings,
        FakeReranker({}),
        generator,
    )
    assert calls["retrieve_reranked"][0]["query"] == "what is the token TTL?"
    system_prompt, user_prompt = generator.calls[0]
    assert "what is the token TTL?" in user_prompt


def test_retrieved_text_and_provenance_preserved_in_citations(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = [
        make_reranked_result(
            chunk_id="a",
            rank=1,
            text="Access tokens expire after 60 minutes.",
            source_file="authentication-api.md",
            section_heading="Token Lifetime",
        )
    ]
    _install_fake_retrieve_reranked(monkeypatch, results=results)
    generator = FakeGenerator("Access tokens expire after 60 minutes [1].")
    answer = retrieve_and_generate(
        "q", ChunkingStrategy.RECURSIVE, index_settings, FakeReranker({}), generator
    )
    assert answer.evidence[0].source_file == "authentication-api.md"
    assert answer.evidence[0].section_heading == "Token Lifetime"


def test_returns_grounded_answer(index_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_retrieve_reranked(monkeypatch)
    generator = FakeGenerator("Answer text [1].")
    answer = retrieve_and_generate(
        "q", ChunkingStrategy.RECURSIVE, index_settings, FakeReranker({}), generator
    )
    assert isinstance(answer, GroundedAnswer)


def test_retrieval_failure_is_surfaced_rather_than_generating_without_evidence(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args: object, **_kwargs: object):
        raise RerankedRetrievalError("simulated retrieval failure")

    monkeypatch.setattr(service_module, "retrieve_reranked", _boom)
    generator = FakeGenerator("should not be reached")

    with pytest.raises(RetrieveAndGenerateError) as exc_info:
        retrieve_and_generate(
            "q", ChunkingStrategy.RECURSIVE, index_settings, FakeReranker({}), generator
        )
    assert isinstance(exc_info.value.__cause__, RerankedRetrievalError)
    assert generator.calls == []


def test_no_extra_dense_sparse_or_rrf_work_beyond_retrieve_reranked(
    index_settings: Settings,
) -> None:
    # A real (small) index, with a tracking embedding provider: if
    # retrieve_and_generate() ran any retrieval work of its own (beyond
    # the single retrieve_reranked() call), this provider would see more
    # than one `.embed([query])` call.
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
    generator = FakeGenerator("alpha content claim [1].")

    retrieve_and_generate(
        "alpha query",
        ChunkingStrategy.RECURSIVE,
        index_settings,
        reranker,
        generator,
        embedding_provider=query_provider,
    )

    assert query_provider.calls == [["alpha query"]]
