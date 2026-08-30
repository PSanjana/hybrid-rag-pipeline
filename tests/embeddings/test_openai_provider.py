"""Tests for rag_pipeline.embeddings.openai.OpenAIEmbeddingProvider.

No network access: the constructor never makes a network call, and every
`.embed()` call here monkeypatches the underlying OpenAI client's
`embeddings.create` method with a local fake response — no real HTTP
request is ever made, and no API key needs to be valid.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from rag_pipeline.config import Settings
from rag_pipeline.embeddings.exceptions import EmbeddingProviderError
from rag_pipeline.embeddings.openai import OpenAIEmbeddingProvider


def _fake_response(items: list[tuple[int, list[float]]]) -> SimpleNamespace:
    """Build a duck-typed stand-in for OpenAI's CreateEmbeddingResponse."""
    data = [SimpleNamespace(index=index, embedding=vector) for index, vector in items]
    return SimpleNamespace(data=data)


def _make_provider(
    settings: Settings | None = None, batch_size: int = 128
) -> OpenAIEmbeddingProvider:
    settings = settings or Settings(_env_file=None, openai_api_key="sk-test-not-real")
    return OpenAIEmbeddingProvider(settings, batch_size=batch_size)


def _patch_create(provider: OpenAIEmbeddingProvider, fn: Any) -> None:
    provider._client.embeddings.create = fn  # type: ignore[method-assign]


def test_missing_api_key_raises_clear_configuration_error() -> None:
    settings = Settings(_env_file=None)
    assert settings.openai_api_key is None
    with pytest.raises(EmbeddingProviderError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider(settings)


def test_provider_constructs_with_api_key_present() -> None:
    provider = _make_provider()
    assert provider is not None


def test_empty_input_returns_empty_list_without_calling_the_client() -> None:
    provider = _make_provider()

    def _should_not_be_called(**_kwargs: Any) -> SimpleNamespace:
        raise AssertionError("embeddings.create should not be called for empty input")

    _patch_create(provider, _should_not_be_called)
    assert provider.embed([]) == []


def test_embed_preserves_input_count_and_order_via_response_index() -> None:
    provider = _make_provider()

    def fake_create(*, model: str, input: list[str]) -> SimpleNamespace:
        # Return items in a SHUFFLED order relative to the request, each
        # tagged with its true input index -- the provider must not assume
        # `response.data` is already in request order.
        vectors = {i: [float(i), float(i) + 0.5] for i in range(len(input))}
        shuffled = list(reversed(range(len(input))))
        return _fake_response([(i, vectors[i]) for i in shuffled])

    _patch_create(provider, fake_create)

    result = provider.embed(["alpha", "beta", "gamma"])

    assert len(result) == 3
    assert result == [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]]


def test_embed_rejects_too_few_response_items() -> None:
    # batch of 2, only one item returned (index 1 entirely missing).
    provider = _make_provider()
    _patch_create(provider, lambda **_kw: _fake_response([(0, [1.0, 2.0])]))

    with pytest.raises(EmbeddingProviderError):
        provider.embed(["alpha", "beta"])


def test_embed_rejects_too_many_response_items_with_valid_but_duplicated_index() -> None:
    # The bug this guards against: 3 items for 2 inputs, where the *set* of
    # indexes present ({0, 1}) looks complete because index 0 is
    # duplicated -- a naive `set(by_index) == set(range(len(batch)))`
    # check alone does not catch this, since a dict comprehension silently
    # collapses the duplicate before the set comparison ever happens.
    provider = _make_provider()
    _patch_create(provider, lambda **_kw: _fake_response([(0, [1.0]), (0, [2.0]), (1, [3.0])]))

    with pytest.raises(EmbeddingProviderError):
        provider.embed(["alpha", "beta"])


def test_embed_rejects_duplicated_index_with_missing_index() -> None:
    # 2 items for 2 inputs (count matches), but both claim index 0;
    # index 1 is never provided.
    provider = _make_provider()
    _patch_create(provider, lambda **_kw: _fake_response([(0, [1.0]), (0, [2.0])]))

    with pytest.raises(EmbeddingProviderError):
        provider.embed(["alpha", "beta"])


def test_embed_rejects_missing_index() -> None:
    # 2 items for 2 inputs, but indexes are {0, 2} instead of {0, 1}.
    provider = _make_provider()
    _patch_create(provider, lambda **_kw: _fake_response([(0, [1.0]), (2, [2.0])]))

    with pytest.raises(EmbeddingProviderError):
        provider.embed(["alpha", "beta"])


def test_embed_rejects_out_of_range_index() -> None:
    provider = _make_provider()
    _patch_create(provider, lambda **_kw: _fake_response([(0, [1.0]), (5, [2.0])]))

    with pytest.raises(EmbeddingProviderError):
        provider.embed(["alpha", "beta"])


def test_embed_restores_input_order_for_a_valid_permuted_response() -> None:
    provider = _make_provider()
    # All three indexes present exactly once, but out of order.
    _patch_create(
        provider,
        lambda **_kw: _fake_response([(2, [2.0]), (0, [0.0]), (1, [1.0])]),
    )

    result = provider.embed(["alpha", "beta", "gamma"])

    assert result == [[0.0], [1.0], [2.0]]


def test_embed_rejects_empty_vector() -> None:
    provider = _make_provider()
    _patch_create(provider, lambda **_kw: _fake_response([(0, [])]))

    with pytest.raises(EmbeddingProviderError, match="empty"):
        provider.embed(["alpha"])


def test_embed_rejects_non_finite_values() -> None:
    provider = _make_provider()
    _patch_create(provider, lambda **_kw: _fake_response([(0, [1.0, float("nan")])]))

    with pytest.raises(EmbeddingProviderError, match="non-finite"):
        provider.embed(["alpha"])


def test_embed_rejects_inconsistent_dimensionality_across_batches() -> None:
    # batch_size=1 forces one HTTP call per input, so dimensionality can
    # legitimately differ *between* batches even though each individual
    # batch response is internally consistent.
    provider = _make_provider(batch_size=1)
    calls = {"n": 0}

    def fake_create(*, model: str, input: list[str]) -> SimpleNamespace:
        calls["n"] += 1
        dim = 3 if calls["n"] == 1 else 4
        return _fake_response([(0, [1.0] * dim)])

    _patch_create(provider, fake_create)

    with pytest.raises(EmbeddingProviderError, match="[Dd]imension"):
        provider.embed(["alpha", "beta"])


def test_embedding_dimension_is_not_hard_coded() -> None:
    provider = _make_provider()
    _patch_create(provider, lambda **_kw: _fake_response([(0, [1.0] * 7)]))
    result = provider.embed(["alpha"])
    assert len(result[0]) == 7
