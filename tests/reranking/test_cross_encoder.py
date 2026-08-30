"""Tests for rag_pipeline.reranking.cross_encoder (import/lazy-load safety only).

No test here downloads a model or calls the network -- `sentence-transformers`
is an optional extra and may not even be installed in this environment;
that is exactly the contract under test: importing this module, and
constructing `CrossEncoderReranker`, must never require it.
"""

from __future__ import annotations

import pytest

from rag_pipeline.reranking.cross_encoder import CrossEncoderReranker
from rag_pipeline.reranking.exceptions import RerankerError


def test_importing_module_does_not_require_sentence_transformers() -> None:
    # If this module's import triggered a `sentence_transformers` import
    # (or a model download), it would already have failed/hung before
    # reaching this line.
    reranker = CrossEncoderReranker()
    assert reranker is not None


def test_constructing_reranker_does_not_load_a_model() -> None:
    reranker = CrossEncoderReranker(model_name="some/model")
    assert reranker._model is None


def test_scoring_empty_documents_returns_empty_without_loading_a_model() -> None:
    reranker = CrossEncoderReranker(model_name="some/model")
    assert reranker.score("a query", []) == []
    assert reranker._model is None


def test_model_name_and_batch_size_are_configurable() -> None:
    reranker = CrossEncoderReranker(model_name="custom/model", batch_size=7)
    assert reranker._model_name == "custom/model"
    assert reranker._batch_size == 7


def test_scoring_without_sentence_transformers_installed_raises_reranker_error() -> None:
    try:
        import sentence_transformers  # noqa: F401

        pytest.skip("sentence-transformers is installed in this environment")
    except ImportError:
        pass

    reranker = CrossEncoderReranker()
    with pytest.raises(RerankerError, match="sentence-transformers is required"):
        reranker.score("a query", ["some document"])
