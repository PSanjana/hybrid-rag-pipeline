"""Tests for rag_pipeline.generation.openai_judge.OpenAICitationJudge.

No network access: the constructor never makes a network call, and every
`.judge()` call here monkeypatches the underlying OpenAI client's
`chat.completions.parse` method with a local fake response -- no real
HTTP request is ever made, and no API key needs to be valid.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from openai import OpenAIError

from rag_pipeline.config import Settings
from rag_pipeline.generation.exceptions import CitationJudgeError
from rag_pipeline.generation.openai_judge import OpenAICitationJudge


def _fake_verdict_item(
    occurrence_id: int, citation_number: int, verdict: str, rationale: str
) -> SimpleNamespace:
    return SimpleNamespace(
        occurrence_id=occurrence_id,
        citation_number=citation_number,
        verdict=verdict,
        rationale=rationale,
    )


def _fake_response(
    verdicts: list[SimpleNamespace] | None, *, refusal: str | None = None
) -> SimpleNamespace:
    parsed = SimpleNamespace(verdicts=verdicts) if verdicts is not None else None
    message = SimpleNamespace(parsed=parsed, refusal=refusal)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _make_judge(settings: Settings | None = None) -> OpenAICitationJudge:
    settings = settings or Settings(_env_file=None, openai_api_key="sk-test-not-real")
    return OpenAICitationJudge(settings)


def _patch_parse(judge: OpenAICitationJudge, fn: Any) -> None:
    judge._client.chat.completions.parse = fn  # type: ignore[method-assign]


def test_missing_api_key_raises_clear_configuration_error() -> None:
    settings = Settings(_env_file=None)
    assert settings.openai_api_key is None
    with pytest.raises(CitationJudgeError, match="OPENAI_API_KEY"):
        OpenAICitationJudge(settings)


def test_judge_constructs_with_api_key_present_without_network_call() -> None:
    judge = _make_judge()
    assert judge is not None


def test_judge_returns_raw_verdicts_from_parsed_response() -> None:
    judge = _make_judge()
    _patch_parse(
        judge,
        lambda **_kw: _fake_response(
            [_fake_verdict_item(1, 1, "supported", "matches the evidence")]
        ),
    )

    results = judge.judge("system", "user")

    assert len(results) == 1
    assert results[0].occurrence_id == 1
    assert results[0].citation_number == 1
    assert results[0].verdict == "supported"
    assert results[0].rationale == "matches the evidence"


def test_judge_preserves_order_and_count_of_multiple_verdicts() -> None:
    judge = _make_judge()
    _patch_parse(
        judge,
        lambda **_kw: _fake_response(
            [
                _fake_verdict_item(1, 1, "supported", "a"),
                _fake_verdict_item(2, 2, "unsupported", "b"),
            ]
        ),
    )

    results = judge.judge("system", "user")

    assert [r.occurrence_id for r in results] == [1, 2]


def test_judge_passes_system_and_user_prompt_as_messages() -> None:
    judge = _make_judge()
    captured: dict[str, Any] = {}

    def fake_parse(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return _fake_response([])

    _patch_parse(judge, fake_parse)
    judge.judge("the judge system prompt", "the judge user prompt")

    assert captured["messages"] == [
        {"role": "system", "content": "the judge system prompt"},
        {"role": "user", "content": "the judge user prompt"},
    ]


def test_judge_uses_configured_model() -> None:
    settings = Settings(
        _env_file=None, openai_api_key="sk-test-not-real", citation_judge_model="custom-judge"
    )
    judge = OpenAICitationJudge(settings)
    captured: dict[str, Any] = {}

    def fake_parse(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return _fake_response([])

    _patch_parse(judge, fake_parse)
    judge.judge("system", "user")

    assert captured["model"] == "custom-judge"


def test_judge_wraps_openai_errors_with_cause_preserved() -> None:
    judge = _make_judge()
    original = OpenAIError("simulated connection failure")

    def _boom(**_kw: Any) -> SimpleNamespace:
        raise original

    _patch_parse(judge, _boom)

    with pytest.raises(CitationJudgeError) as exc_info:
        judge.judge("system", "user")
    assert exc_info.value.__cause__ is original


def test_judge_rejects_no_choices() -> None:
    judge = _make_judge()
    _patch_parse(judge, lambda **_kw: SimpleNamespace(choices=[]))

    with pytest.raises(CitationJudgeError, match="no choices"):
        judge.judge("system", "user")


def test_judge_rejects_a_refusal() -> None:
    judge = _make_judge()
    _patch_parse(judge, lambda **_kw: _fake_response([], refusal="cannot help with this"))

    with pytest.raises(CitationJudgeError, match="refused"):
        judge.judge("system", "user")


def test_judge_rejects_none_parsed() -> None:
    judge = _make_judge()
    _patch_parse(judge, lambda **_kw: _fake_response(None))

    with pytest.raises(CitationJudgeError, match="no parsed"):
        judge.judge("system", "user")
