"""Grounded answer generation with bracketed citations, downstream of retrieve_reranked().

Given a question and the final reranked evidence chunks,
`generate_grounded_answer()` asks a `Generator` provider for an answer
that uses only the supplied evidence and cites it by number (`[1]`,
`[2]`, ...), then validates that every citation refers to supplied
evidence and that a substantive answer is not left uncited. Evidence
blocks are treated as untrusted data (see `prompt.SYSTEM_PROMPT`), not
instructions -- a document containing text like "ignore previous
instructions" is never followed. `retrieve_and_generate()` composes this
with the existing `retrieve_reranked()` retrieval pipeline, duplicating
none of its dense/sparse/RRF/reranking logic.

Citation semantic verification, confidence scoring, and a final
abstention ("I don't know") policy are later pipeline stages and are not
implemented here -- the `_INSUFFICIENT_EVIDENCE_MARKER` string match in
`service.py` only recognizes the literal response form the system prompt
asks for; it is not a semantic judge.
"""

from .base import Generator
from .citations import extract_citations, resolve_citation, validate_citations
from .context import build_evidence, format_evidence_block
from .exceptions import (
    CitationValidationError,
    GenerationError,
    GenerationProviderError,
    InvalidGenerationInputError,
    RetrieveAndGenerateError,
    UncitedAnswerError,
)
from .models import Evidence, GroundedAnswer
from .openai import OpenAIGenerator
from .prompt import SYSTEM_PROMPT, build_user_prompt
from .service import generate_grounded_answer, retrieve_and_generate

__all__ = [
    "SYSTEM_PROMPT",
    "CitationValidationError",
    "Evidence",
    "GenerationError",
    "GenerationProviderError",
    "Generator",
    "GroundedAnswer",
    "InvalidGenerationInputError",
    "OpenAIGenerator",
    "RetrieveAndGenerateError",
    "UncitedAnswerError",
    "build_evidence",
    "build_user_prompt",
    "extract_citations",
    "format_evidence_block",
    "generate_grounded_answer",
    "resolve_citation",
    "retrieve_and_generate",
    "validate_citations",
]
