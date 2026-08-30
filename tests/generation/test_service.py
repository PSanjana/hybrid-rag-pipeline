"""Tests for rag_pipeline.generation.service.generate_grounded_answer (pure, no retrieval)."""

from __future__ import annotations

import copy

import pytest

from rag_pipeline.generation.exceptions import (
    CitationValidationError,
    GenerationProviderError,
    InvalidGenerationInputError,
    UncitedAnswerError,
)
from rag_pipeline.generation.models import GroundedAnswer
from rag_pipeline.generation.service import generate_grounded_answer

from .conftest import FakeGenerator, make_reranked_result

_RESULTS = [
    make_reranked_result(chunk_id="a", rank=1, text="Access tokens expire after 60 minutes."),
    make_reranked_result(chunk_id="b", rank=2, text="Production access requires MFA."),
]


def test_question_passed_correctly_into_prompt() -> None:
    generator = FakeGenerator("Tokens expire in 60 minutes [1].")
    generate_grounded_answer("How long do access tokens last?", _RESULTS, generator)
    system_prompt, user_prompt = generator.calls[0]
    assert "How long do access tokens last?" in user_prompt


def test_provider_receives_system_and_evidence_context() -> None:
    generator = FakeGenerator("Tokens expire in 60 minutes [1].")
    generate_grounded_answer("q", _RESULTS, generator)
    system_prompt, user_prompt = generator.calls[0]
    assert "grounded question-answering" in system_prompt.lower()
    assert "Access tokens expire after 60 minutes." in user_prompt
    assert "Production access requires MFA." in user_prompt


def test_generated_cited_answer_accepted() -> None:
    generator = FakeGenerator("Access tokens expire after 60 minutes [1].")
    answer = generate_grounded_answer("q", _RESULTS, generator)
    assert isinstance(answer, GroundedAnswer)
    assert answer.cited_numbers == (1,)
    assert answer.answer_text == "Access tokens expire after 60 minutes [1]."
    assert len(answer.evidence) == 2


def test_multiple_citations_accepted_in_first_appearance_order() -> None:
    generator = FakeGenerator("Tokens expire [2]. MFA is required [1].")
    answer = generate_grounded_answer("q", _RESULTS, generator)
    assert answer.cited_numbers == (2, 1)


def test_zero_citation_factual_answer_rejected() -> None:
    generator = FakeGenerator("Access tokens expire after 60 minutes.")
    with pytest.raises(UncitedAnswerError):
        generate_grounded_answer("q", _RESULTS, generator)


def test_explicit_insufficient_evidence_response_does_not_require_citations() -> None:
    generator = FakeGenerator(
        "The supplied documents do not provide enough information to answer this question."
    )
    answer = generate_grounded_answer("q", _RESULTS, generator)
    assert answer.cited_numbers == ()


def test_zero_citation_answer_accepted_when_no_evidence_supplied() -> None:
    generator = FakeGenerator("There is nothing to answer from.")
    answer = generate_grounded_answer("q", [], generator)
    assert answer.cited_numbers == ()
    assert answer.evidence == ()


def test_invalid_citation_number_rejected() -> None:
    generator = FakeGenerator("Tokens expire after 60 minutes [7].")
    with pytest.raises(CitationValidationError):
        generate_grounded_answer("q", _RESULTS, generator)


def test_zero_citation_number_rejected() -> None:
    generator = FakeGenerator("Tokens expire after 60 minutes [0].")
    with pytest.raises(CitationValidationError):
        generate_grounded_answer("q", _RESULTS, generator)


def test_empty_provider_response_rejected() -> None:
    generator = FakeGenerator("   ")
    with pytest.raises(GenerationProviderError):
        generate_grounded_answer("q", _RESULTS, generator)


def test_provider_failure_wrapped_with_cause_preserved() -> None:
    original = RuntimeError("simulated provider crash")
    generator = FakeGenerator(error=original)
    with pytest.raises(GenerationProviderError) as exc_info:
        generate_grounded_answer("q", _RESULTS, generator)
    assert exc_info.value.__cause__ is original


def test_generation_provider_error_from_provider_is_not_double_wrapped() -> None:
    from rag_pipeline.generation.exceptions import GenerationProviderError as GPE

    original = GPE("provider-specific failure")
    generator = FakeGenerator(error=original)
    with pytest.raises(GenerationProviderError) as exc_info:
        generate_grounded_answer("q", _RESULTS, generator)
    assert exc_info.value is original


def test_empty_question_rejected_before_calling_provider() -> None:
    generator = FakeGenerator("should not be reached")
    with pytest.raises(InvalidGenerationInputError):
        generate_grounded_answer("   ", _RESULTS, generator)
    assert generator.calls == []


def test_no_api_key_required_with_fake_provider() -> None:
    # FakeGenerator never touches OPENAI_API_KEY / the network at all --
    # this test's mere success is the assertion.
    generator = FakeGenerator("Tokens expire after 60 minutes [1].")
    generate_grounded_answer("q", _RESULTS, generator)


def test_input_reranked_results_not_mutated() -> None:
    before = copy.deepcopy(_RESULTS)
    generator = FakeGenerator("Tokens expire after 60 minutes [1].")
    generate_grounded_answer("q", _RESULTS, generator)
    assert _RESULTS == before
