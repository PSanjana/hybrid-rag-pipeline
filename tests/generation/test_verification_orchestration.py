"""Tests for rag_pipeline.generation.verification.retrieve_generate_and_verify."""

from __future__ import annotations

import pytest

from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.generation import verification as verification_module
from rag_pipeline.generation.exceptions import RetrieveAndGenerateError
from rag_pipeline.generation.models import CitationVerdict, GroundedAnswer
from rag_pipeline.generation.verification import retrieve_generate_and_verify

from ..retrieval.conftest import FakeReranker
from .conftest import FakeCitationJudge, FakeGenerator, make_grounded_answer


def _install_fake_retrieve_and_generate(
    monkeypatch: pytest.MonkeyPatch, answer: GroundedAnswer | None = None
) -> dict[str, list]:
    calls: dict[str, list] = {"retrieve_and_generate": []}

    default_answer = make_grounded_answer(answer_text="Tokens expire after 60 minutes [1].")

    def fake_retrieve_and_generate(
        question,
        strategy,
        settings,
        reranker,
        generator,
        embedding_provider=None,
        dense_top_k=None,
        sparse_top_k=None,
        candidate_k=None,
        final_top_k=None,
    ):
        calls["retrieve_and_generate"].append({"question": question})
        return answer if answer is not None else default_answer

    monkeypatch.setattr(verification_module, "retrieve_and_generate", fake_retrieve_and_generate)
    return calls


def test_retrieve_and_generate_called_exactly_once(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_retrieve_and_generate(monkeypatch)
    judge = FakeCitationJudge({1: (1, "supported", "matches")})
    retrieve_generate_and_verify(
        "a question",
        ChunkingStrategy.RECURSIVE,
        index_settings,
        FakeReranker({}),
        FakeGenerator("unused"),
        judge,
    )
    assert len(calls["retrieve_and_generate"]) == 1


def test_returns_answer_and_verification_report(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_retrieve_and_generate(monkeypatch)
    judge = FakeCitationJudge({1: (1, "supported", "matches")})
    answer, report = retrieve_generate_and_verify(
        "a question",
        ChunkingStrategy.RECURSIVE,
        index_settings,
        FakeReranker({}),
        FakeGenerator("unused"),
        judge,
    )
    assert isinstance(answer, GroundedAnswer)
    assert report.total_occurrences == 1
    assert report.verifications[0].verdict == CitationVerdict.SUPPORTED


def test_same_question_flows_into_generation_and_verification(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_retrieve_and_generate(monkeypatch)
    judge = FakeCitationJudge({1: (1, "supported", "matches")})
    retrieve_generate_and_verify(
        "what is the token TTL?",
        ChunkingStrategy.RECURSIVE,
        index_settings,
        FakeReranker({}),
        FakeGenerator("unused"),
        judge,
    )
    assert calls["retrieve_and_generate"][0]["question"] == "what is the token TTL?"
    _, user_prompt = judge.calls[0]
    assert "what is the token TTL?" in user_prompt


def test_retrieval_or_generation_failure_is_surfaced(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args: object, **_kwargs: object):
        raise RetrieveAndGenerateError("simulated failure upstream")

    monkeypatch.setattr(verification_module, "retrieve_and_generate", _boom)
    judge = FakeCitationJudge()

    with pytest.raises(RetrieveAndGenerateError):
        retrieve_generate_and_verify(
            "q",
            ChunkingStrategy.RECURSIVE,
            index_settings,
            FakeReranker({}),
            FakeGenerator("unused"),
            judge,
        )
    assert judge.calls == []


def test_no_verification_occurs_for_insufficient_evidence_answer(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    insufficient_answer = GroundedAnswer(
        answer_text=(
            "The supplied documents do not provide enough information to answer this question."
        ),
        evidence=tuple(),
        cited_numbers=(),
    )
    _install_fake_retrieve_and_generate(monkeypatch, answer=insufficient_answer)
    judge = FakeCitationJudge()

    answer, report = retrieve_generate_and_verify(
        "q",
        ChunkingStrategy.RECURSIVE,
        index_settings,
        FakeReranker({}),
        FakeGenerator("unused"),
        judge,
    )

    assert report.total_occurrences == 0
    assert judge.calls == []
