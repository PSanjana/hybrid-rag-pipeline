"""Grounded-generation service: evidence -> prompt -> provider -> validated GroundedAnswer.

`generate_grounded_answer()` is independently testable from retrieval:
given an already-final `RerankedRetrievalResult` list (the same
authoritative output `retrieve_reranked()` produces) and a `Generator`,
it never touches dense/sparse/RRF/reranking/embedding machinery itself.
`retrieve_and_generate()` is a thin orchestration on top that calls the
existing `retrieve_reranked()` unmodified, then feeds its output into
`generate_grounded_answer()` -- retrieval logic is never duplicated here.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from ..config import ChunkingStrategy, Settings
from ..embeddings import EmbeddingProvider
from ..reranking.base import Reranker
from ..retrieval import RetrievalError, retrieve_reranked
from ..retrieval.models import RerankedRetrievalResult
from .base import Generator
from .citations import extract_citations, validate_citations
from .context import build_evidence, format_evidence_block
from .exceptions import (
    GenerationProviderError,
    InvalidGenerationInputError,
    RetrieveAndGenerateError,
    UncitedAnswerError,
)
from .models import GroundedAnswer
from .prompt import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

# Must stay in sync with rule 6 of `prompt.SYSTEM_PROMPT` -- a literal,
# case-insensitive substring match, not a semantic/LLM judgment. This is
# deliberately narrow: it only recognizes the exact response form the
# system prompt asks the model to use when evidence is insufficient, and
# is not a general-purpose confidence/abstention system (see module and
# package docstrings).
_INSUFFICIENT_EVIDENCE_MARKER = "the supplied documents do not provide enough information"


def _is_explicit_insufficient_evidence_response(answer_text: str) -> bool:
    return _INSUFFICIENT_EVIDENCE_MARKER in answer_text.lower()


def generate_grounded_answer(
    question: str,
    reranked_results: Sequence[RerankedRetrievalResult],
    provider: Generator,
) -> GroundedAnswer:
    """Generate a grounded, bracket-cited answer from already-reranked evidence.

    Builds numbered `Evidence` from `reranked_results` (citation number
    == reranked rank; see `context.build_evidence`), asks `provider` for
    an answer grounded in that evidence plus the fixed grounding/
    anti-injection system prompt, extracts and range-validates its
    bracket citations, and returns a `GroundedAnswer`. `reranked_results`
    is read-only and is never mutated.

    Raises `InvalidGenerationInputError` for an empty/whitespace
    question; `GenerationProviderError` if the provider itself raises or
    returns an empty response (cause preserved for the former);
    `CitationValidationError` if any cited number falls outside
    `[1, len(evidence)]`; or `UncitedAnswerError` if evidence was
    supplied and the answer contains zero bracket citations while not
    reading as the explicit insufficient-evidence response form.
    """
    if not question.strip():
        raise InvalidGenerationInputError("Question must not be empty or whitespace-only.")

    evidence = build_evidence(reranked_results)
    evidence_block = format_evidence_block(evidence)
    user_prompt = build_user_prompt(question, evidence_block)

    try:
        answer_text = provider.generate(SYSTEM_PROMPT, user_prompt)
    except GenerationProviderError:
        raise
    except Exception as exc:
        raise GenerationProviderError(f"Generation provider failed: {exc}") from exc

    if not answer_text.strip():
        raise GenerationProviderError("Generation provider returned an empty response.")

    cited_numbers = extract_citations(answer_text)
    validate_citations(cited_numbers, len(evidence))

    if (
        evidence
        and not cited_numbers
        and not _is_explicit_insufficient_evidence_response(answer_text)
    ):
        raise UncitedAnswerError(
            "Generated answer contains no bracket citations despite supplied evidence, and "
            "is not the explicit insufficient-evidence response form."
        )

    logger.info(
        "grounded generation: evidence_count=%d cited_count=%d",
        len(evidence),
        len(cited_numbers),
    )
    return GroundedAnswer(
        answer_text=answer_text,
        evidence=tuple(evidence),
        cited_numbers=tuple(cited_numbers),
    )


def retrieve_and_generate(
    question: str,
    strategy: ChunkingStrategy,
    settings: Settings,
    reranker: Reranker,
    generator: Generator,
    embedding_provider: EmbeddingProvider | None = None,
    dense_top_k: int | None = None,
    sparse_top_k: int | None = None,
    candidate_k: int | None = None,
    final_top_k: int | None = None,
) -> GroundedAnswer:
    """question -> retrieve_reranked() -> generate_grounded_answer() -> GroundedAnswer.

    `retrieve_reranked()` remains the sole authoritative retrieval
    implementation -- this function never duplicates dense, sparse, RRF,
    or reranking logic, only composes the existing pipeline with
    generation. A retrieval failure is wrapped as
    `RetrieveAndGenerateError` (cause preserved) rather than silently
    generating an answer without evidence; a generation-side failure
    (`GenerationProviderError`, `CitationValidationError`,
    `UncitedAnswerError`) propagates unwrapped, exactly mirroring how
    `retrieve_reranked()` lets a `RerankError` from `rerank_candidates()`
    propagate unwrapped.
    """
    try:
        reranked_results = retrieve_reranked(
            question,
            strategy,
            settings,
            reranker,
            embedding_provider=embedding_provider,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            candidate_k=candidate_k,
            final_top_k=final_top_k,
        )
    except RetrievalError as exc:
        raise RetrieveAndGenerateError(f"Retrieval failed during generation: {exc}") from exc

    return generate_grounded_answer(question, reranked_results, generator)
