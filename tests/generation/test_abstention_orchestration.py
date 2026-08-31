"""Tests for rag_pipeline.generation.abstention.answer_question_with_policy (offline)."""

from __future__ import annotations

import pytest

from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.generation import abstention as abstention_module
from rag_pipeline.generation import confidence as confidence_module
from rag_pipeline.generation.abstention import answer_question_with_policy
from rag_pipeline.generation.confidence import score_confidence
from rag_pipeline.generation.exceptions import RetrieveAndGenerateError
from rag_pipeline.generation.models import AnswerDecision, FinalAnswer
from rag_pipeline.generation.verification import verify_grounded_answer

from ..retrieval.conftest import FakeReranker
from .conftest import (
    FakeCitationJudge,
    FakeGenerator,
    make_grounded_answer,
    make_reranked_result,
)


def _valid_trio(settings: Settings) -> tuple[object, object, object]:
    results = [make_reranked_result(chunk_id="a", rank=1, dense_rank=1, sparse_rank=1)]
    answer = make_grounded_answer(answer_text="A [1].", reranked_results=results)
    report = verify_grounded_answer("q", answer, FakeCitationJudge({1: (1, "supported", "ok")}))
    confidence = score_confidence(answer, report, results, settings)
    return answer, report, confidence


def _install_spy(
    monkeypatch: pytest.MonkeyPatch, trio: tuple[object, object, object]
) -> list[tuple[tuple, dict]]:
    calls: list[tuple[tuple, dict]] = []

    def spy(*args: object, **kwargs: object) -> tuple[object, object, object]:
        calls.append((args, kwargs))
        return trio

    monkeypatch.setattr(abstention_module, "retrieve_generate_verify_and_score", spy)
    return calls


def test_step3_orchestration_is_called_exactly_once(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    trio = _valid_trio(index_settings)
    calls = _install_spy(monkeypatch, trio)
    generator = FakeGenerator("unused")
    judge = FakeCitationJudge()

    answer_question_with_policy(
        "q", ChunkingStrategy.RECURSIVE, index_settings, FakeReranker({}), generator, judge
    )

    assert len(calls) == 1


def test_policy_layer_does_not_retrieve_generate_or_judge_itself(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    trio = _valid_trio(index_settings)
    _install_spy(monkeypatch, trio)
    generator = FakeGenerator("unused")
    judge = FakeCitationJudge()

    answer_question_with_policy(
        "q", ChunkingStrategy.RECURSIVE, index_settings, FakeReranker({}), generator, judge
    )

    # Step 1-3 orchestration is entirely stubbed; the policy layer touches
    # no provider on its own.
    assert generator.calls == []
    assert judge.calls == []


def test_abstention_module_does_not_import_lower_stage_functions() -> None:
    # answer_question_with_policy must go strictly through
    # retrieve_generate_verify_and_score(); it never re-invokes an
    # individual retrieval/generation/verification/confidence stage.
    for name in (
        "retrieve_reranked",
        "generate_grounded_answer",
        "verify_grounded_answer",
        "score_confidence",
    ):
        assert not hasattr(abstention_module, name)


def test_returned_step3_aggregate_is_passed_straight_into_the_policy(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer, report, confidence = _valid_trio(index_settings)
    _install_spy(monkeypatch, (answer, report, confidence))
    generator = FakeGenerator("unused")

    final = answer_question_with_policy(
        "q",
        ChunkingStrategy.RECURSIVE,
        index_settings,
        FakeReranker({}),
        generator,
        FakeCitationJudge(),
    )

    assert isinstance(final, FinalAnswer)
    assert final.grounded_answer is answer
    assert final.verification_report is report
    assert final.confidence is confidence
    assert final.decision is AnswerDecision.ANSWERED


def test_upstream_failure_propagates_unchanged(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> object:
        raise RetrieveAndGenerateError("simulated upstream failure")

    monkeypatch.setattr(abstention_module, "retrieve_generate_verify_and_score", boom)

    with pytest.raises(RetrieveAndGenerateError, match="simulated upstream failure"):
        answer_question_with_policy(
            "q",
            ChunkingStrategy.RECURSIVE,
            index_settings,
            FakeReranker({}),
            FakeGenerator("unused"),
            FakeCitationJudge(),
        )


def test_end_to_end_through_real_step3_with_only_retrieval_stubbed(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercise the real generate -> verify -> score -> policy chain; only
    # the retrieval boundary is faked. Each lower stage is invoked exactly
    # once by Step 3, and the policy layer adds nothing.
    reranked = [
        make_reranked_result(
            chunk_id="a",
            rank=1,
            dense_rank=1,
            sparse_rank=1,
            text="Access tokens expire after 60 minutes.",
        )
    ]
    retrieval_calls: list[str] = []

    def fake_retrieve_reranked(query, *_a: object, **_k: object) -> list:
        retrieval_calls.append(query)
        return reranked

    monkeypatch.setattr(confidence_module, "retrieve_reranked", fake_retrieve_reranked)
    generator = FakeGenerator("Access tokens expire after 60 minutes [1].")
    judge = FakeCitationJudge({1: (1, "supported", "matches")})

    final = answer_question_with_policy(
        "how long do tokens last?",
        ChunkingStrategy.RECURSIVE,
        index_settings,
        FakeReranker({}),
        generator,
        judge,
    )

    assert retrieval_calls == ["how long do tokens last?"]
    assert len(generator.calls) == 1
    assert len(judge.calls) == 1
    assert isinstance(final, FinalAnswer)
    assert final.decision is AnswerDecision.ANSWERED
    assert final.answer_text == "Access tokens expire after 60 minutes [1]."


def test_end_to_end_insufficient_evidence_abstains_without_calling_judge(
    index_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        confidence_module,
        "retrieve_reranked",
        lambda *_a, **_k: [make_reranked_result(chunk_id="a", rank=1)],
    )
    generator = FakeGenerator(
        "The supplied documents do not provide enough information to answer this question."
    )
    judge = FakeCitationJudge(error=RuntimeError("judge must never be called"))

    final = answer_question_with_policy(
        "q",
        ChunkingStrategy.RECURSIVE,
        index_settings,
        FakeReranker({}),
        generator,
        judge,
    )

    assert final.decision is AnswerDecision.ABSTAINED_INSUFFICIENT_EVIDENCE
    assert final.abstained is True
    assert judge.calls == []
