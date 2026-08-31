"""Tests for rag_pipeline.evaluation.metrics.openai_judge.OpenAIEvaluationJudge.

No network: the constructor makes no network call, and every ``.assess_*``
call here monkeypatches the underlying OpenAI client's
``chat.completions.parse`` with a local fake -- no HTTP request is made and
no API key needs to be valid.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from openai import OpenAIError

from rag_pipeline.config import Settings
from rag_pipeline.evaluation.exceptions import EvaluationJudgeError
from rag_pipeline.evaluation.metrics.openai_judge import OpenAIEvaluationJudge


def _response(parsed: Any, *, refusal: str | None = None) -> SimpleNamespace:
    message = SimpleNamespace(parsed=parsed, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _make_judge(settings: Settings | None = None) -> OpenAIEvaluationJudge:
    settings = settings or Settings(_env_file=None, openai_api_key="sk-test-not-real")
    return OpenAIEvaluationJudge(settings)


def _patch_parse(judge: OpenAIEvaluationJudge, fn: Any) -> None:
    judge._client.chat.completions.parse = fn  # type: ignore[method-assign]


def test_missing_api_key_raises_clear_configuration_error() -> None:
    settings = Settings(_env_file=None)
    assert settings.openai_api_key is None
    with pytest.raises(EvaluationJudgeError, match="OPENAI_API_KEY"):
        OpenAIEvaluationJudge(settings)


def test_constructs_with_api_key_present_without_network_call() -> None:
    assert _make_judge() is not None


def test_uses_configured_evaluation_judge_model() -> None:
    settings = Settings(
        _env_file=None, openai_api_key="sk-test-not-real", evaluation_judge_model="eval-model-x"
    )
    judge = OpenAIEvaluationJudge(settings)
    captured: dict[str, Any] = {}

    def fake_parse(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return _response(SimpleNamespace(verdicts=[], golden_contradictions=[]))

    _patch_parse(judge, fake_parse)
    judge.assess_correctness("system", "user")

    assert captured["model"] == "eval-model-x"
    assert captured["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]


def test_assess_correctness_maps_parsed_verdicts_to_raw_fact_verdicts() -> None:
    judge = _make_judge()
    _patch_parse(
        judge,
        lambda **_kw: _response(
            SimpleNamespace(
                verdicts=[
                    SimpleNamespace(fact_id=1, verdict="correct", rationale="ok"),
                    SimpleNamespace(fact_id=2, verdict="missing", rationale="absent"),
                ],
                golden_contradictions=[],
            )
        ),
    )

    result = judge.assess_correctness("system", "user")

    assert [(r.fact_id, r.verdict, r.rationale) for r in result.fact_verdicts] == [
        (1, "correct", "ok"),
        (2, "missing", "absent"),
    ]
    assert result.golden_contradictions == []


def test_assess_correctness_maps_parsed_golden_contradictions() -> None:
    judge = _make_judge()
    _patch_parse(
        judge,
        lambda **_kw: _response(
            SimpleNamespace(
                verdicts=[SimpleNamespace(fact_id=1, verdict="correct", rationale="ok")],
                golden_contradictions=[
                    SimpleNamespace(
                        contradiction_id=1,
                        claim_text="tokens live 24h",
                        rationale="golden fact says 60 minutes",
                        conflicting_fact_ids=[1],
                    )
                ],
            )
        ),
    )

    result = judge.assess_correctness("system", "user")

    assert len(result.golden_contradictions) == 1
    contradiction = result.golden_contradictions[0]
    assert contradiction.contradiction_id == 1
    assert contradiction.claim_text == "tokens live 24h"
    assert contradiction.rationale == "golden fact says 60 minutes"
    assert contradiction.conflicting_fact_ids == (1,)


def test_assess_faithfulness_maps_parsed_claims_to_raw_claim_verdicts() -> None:
    judge = _make_judge()
    _patch_parse(
        judge,
        lambda **_kw: _response(
            SimpleNamespace(
                claims=[
                    SimpleNamespace(
                        claim_id=1, claim_text="c1", verdict="supported", rationale="ev"
                    )
                ]
            )
        ),
    )

    result = judge.assess_faithfulness("system", "user")

    assert [(r.claim_id, r.claim_text, r.verdict, r.rationale) for r in result] == [
        (1, "c1", "supported", "ev")
    ]


def test_wraps_openai_errors_with_cause_preserved() -> None:
    judge = _make_judge()
    original = OpenAIError("simulated failure")

    def _boom(**_kw: Any) -> SimpleNamespace:
        raise original

    _patch_parse(judge, _boom)

    with pytest.raises(EvaluationJudgeError) as exc_info:
        judge.assess_correctness("system", "user")
    assert exc_info.value.__cause__ is original


def test_rejects_no_choices_refusal_and_none_parsed() -> None:
    judge = _make_judge()

    _patch_parse(judge, lambda **_kw: SimpleNamespace(choices=[]))
    with pytest.raises(EvaluationJudgeError, match="no choices"):
        judge.assess_faithfulness("s", "u")

    _patch_parse(judge, lambda **_kw: _response(None, refusal="cannot help"))
    with pytest.raises(EvaluationJudgeError, match="refused"):
        judge.assess_faithfulness("s", "u")

    _patch_parse(judge, lambda **_kw: _response(None))
    with pytest.raises(EvaluationJudgeError, match="no parsed"):
        judge.assess_faithfulness("s", "u")
