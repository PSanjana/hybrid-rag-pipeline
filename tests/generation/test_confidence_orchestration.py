"""Tests for rag_pipeline.generation.confidence.retrieve_generate_verify_and_score."""

from __future__ import annotations

import pytest

from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.generation import confidence as confidence_module
from rag_pipeline.generation.confidence import retrieve_generate_verify_and_score
from rag_pipeline.generation.exceptions import RetrieveAndGenerateError
from rag_pipeline.generation.models import (
    CitationVerificationReport,
    ConfidenceAssessment,
    GroundedAnswer,
)
from rag_pipeline.retrieval.exceptions import RerankedRetrievalError

from ..retrieval.conftest import FakeReranker
from .conftest import FakeCitationJudge, FakeGenerator, make_reranked_result


def _install_fake_retrieve_reranked(
    monkeypatch: pytest.MonkeyPatch, results: list | None = None
) -> dict[str, list]:
    calls: dict[str, list] = {"retrieve_reranked": []}

    default_results = [
        make_reranked_result(chunk_id="a", rank=1, text="Access tokens expire after 60 minutes.")
    ]

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
        return results if results is not None else default_results

    monkeypatch.setattr(confidence_module, "retrieve_reranked", fake_retrieve_reranked)
    return calls


def test_retrieve_reranked_called_exactly_once(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_retrieve_reranked(monkeypatch)
    generator = FakeGenerator("Access tokens expire after 60 minutes [1].")
    judge = FakeCitationJudge({1: (1, "supported", "matches")})
    retrieve_generate_verify_and_score(
        "a question", ChunkingStrategy.RECURSIVE, index_settings, FakeReranker({}), generator, judge
    )
    assert len(calls["retrieve_reranked"]) == 1


def test_returns_answer_report_and_assessment(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_retrieve_reranked(monkeypatch)
    generator = FakeGenerator("Access tokens expire after 60 minutes [1].")
    judge = FakeCitationJudge({1: (1, "supported", "matches")})
    answer, report, assessment = retrieve_generate_verify_and_score(
        "a question", ChunkingStrategy.RECURSIVE, index_settings, FakeReranker({}), generator, judge
    )
    assert isinstance(answer, GroundedAnswer)
    assert isinstance(report, CitationVerificationReport)
    assert isinstance(assessment, ConfidenceAssessment)
    assert assessment.citation_support_score == pytest.approx(1.0)


def test_same_question_flows_through_every_stage(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_retrieve_reranked(monkeypatch)
    generator = FakeGenerator("Access tokens expire after 60 minutes [1].")
    judge = FakeCitationJudge({1: (1, "supported", "matches")})
    retrieve_generate_verify_and_score(
        "what is the token TTL?",
        ChunkingStrategy.RECURSIVE,
        index_settings,
        FakeReranker({}),
        generator,
        judge,
    )
    assert calls["retrieve_reranked"][0]["query"] == "what is the token TTL?"
    system_prompt, user_prompt = generator.calls[0]
    assert "what is the token TTL?" in user_prompt


def test_retrieval_failure_is_wrapped_with_cause_preserved(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args: object, **_kwargs: object):
        raise RerankedRetrievalError("simulated retrieval failure")

    monkeypatch.setattr(confidence_module, "retrieve_reranked", _boom)
    generator = FakeGenerator("should not be reached")
    judge = FakeCitationJudge()

    with pytest.raises(RetrieveAndGenerateError) as exc_info:
        retrieve_generate_verify_and_score(
            "q", ChunkingStrategy.RECURSIVE, index_settings, FakeReranker({}), generator, judge
        )
    assert isinstance(exc_info.value.__cause__, RerankedRetrievalError)
    assert generator.calls == []
    assert judge.calls == []


def test_confidence_module_never_imports_the_composite_wrapper() -> None:
    # retrieve_generate_verify_and_score() must go through
    # retrieve_reranked()/generate_grounded_answer()/verify_grounded_answer()
    # directly (to keep the intermediate RerankedRetrievalResult list for
    # scoring), never through the retrieve_and_generate()/
    # retrieve_generate_and_verify() composite wrappers -- this asserts
    # that structurally, since confidence_module never imports those names.
    assert not hasattr(confidence_module, "retrieve_and_generate")
    assert not hasattr(confidence_module, "retrieve_generate_and_verify")


def test_insufficient_evidence_answer_gets_zero_confidence(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_retrieve_reranked(monkeypatch)
    generator = FakeGenerator(
        "The supplied documents do not provide enough information to answer this question."
    )
    judge = FakeCitationJudge(error=RuntimeError("judge must never be called"))
    answer, report, assessment = retrieve_generate_verify_and_score(
        "q", ChunkingStrategy.RECURSIVE, index_settings, FakeReranker({}), generator, judge
    )
    assert assessment.is_insufficient_evidence is True
    assert assessment.score == pytest.approx(0.0)
    assert judge.calls == []
