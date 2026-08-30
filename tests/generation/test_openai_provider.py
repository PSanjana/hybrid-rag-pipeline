"""Tests for rag_pipeline.generation.openai.OpenAIGenerator.

No network access: the constructor never makes a network call, and every
`.generate()` call here monkeypatches the underlying OpenAI client's
`chat.completions.create` method with a local fake response -- no real
HTTP request is ever made, and no API key needs to be valid.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from openai import OpenAIError

from rag_pipeline.config import Settings
from rag_pipeline.generation.exceptions import GenerationProviderError
from rag_pipeline.generation.openai import OpenAIGenerator


def _fake_response(content: str | None) -> SimpleNamespace:
    """Build a duck-typed stand-in for OpenAI's ChatCompletion response."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _make_provider(settings: Settings | None = None) -> OpenAIGenerator:
    settings = settings or Settings(_env_file=None, openai_api_key="sk-test-not-real")
    return OpenAIGenerator(settings)


def _patch_create(provider: OpenAIGenerator, fn: Any) -> None:
    provider._client.chat.completions.create = fn  # type: ignore[method-assign]


def test_missing_api_key_raises_clear_configuration_error() -> None:
    settings = Settings(_env_file=None)
    assert settings.openai_api_key is None
    with pytest.raises(GenerationProviderError, match="OPENAI_API_KEY"):
        OpenAIGenerator(settings)


def test_provider_constructs_with_api_key_present_without_network_call() -> None:
    provider = _make_provider()
    assert provider is not None


def test_generate_returns_text_content() -> None:
    provider = _make_provider()
    _patch_create(provider, lambda **_kw: _fake_response("Access tokens expire in 60m [1]."))

    result = provider.generate("system", "user")

    assert result == "Access tokens expire in 60m [1]."


def test_generate_passes_system_and_user_prompt_as_messages() -> None:
    provider = _make_provider()
    captured: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return _fake_response("answer")

    _patch_create(provider, fake_create)
    provider.generate("the system prompt", "the user prompt")

    assert captured["messages"] == [
        {"role": "system", "content": "the system prompt"},
        {"role": "user", "content": "the user prompt"},
    ]


def test_generate_uses_configured_model() -> None:
    settings = Settings(
        _env_file=None, openai_api_key="sk-test-not-real", generation_model="custom-model"
    )
    provider = OpenAIGenerator(settings)
    captured: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return _fake_response("answer")

    _patch_create(provider, fake_create)
    provider.generate("system", "user")

    assert captured["model"] == "custom-model"


def test_generate_wraps_openai_errors_with_cause_preserved() -> None:
    provider = _make_provider()
    original = OpenAIError("simulated connection failure")

    def _boom(**_kw: Any) -> SimpleNamespace:
        raise original

    _patch_create(provider, _boom)

    with pytest.raises(GenerationProviderError) as exc_info:
        provider.generate("system", "user")
    assert exc_info.value.__cause__ is original


def test_generate_rejects_no_choices() -> None:
    provider = _make_provider()
    _patch_create(provider, lambda **_kw: SimpleNamespace(choices=[]))

    with pytest.raises(GenerationProviderError, match="no choices"):
        provider.generate("system", "user")


def test_generate_rejects_none_content() -> None:
    provider = _make_provider()
    _patch_create(provider, lambda **_kw: _fake_response(None))

    with pytest.raises(GenerationProviderError, match="empty"):
        provider.generate("system", "user")


def test_generate_rejects_whitespace_only_content() -> None:
    provider = _make_provider()
    _patch_create(provider, lambda **_kw: _fake_response("   \n  "))

    with pytest.raises(GenerationProviderError, match="empty"):
        provider.generate("system", "user")
