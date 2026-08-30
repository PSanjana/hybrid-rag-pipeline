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

`verify_grounded_answer()` goes one step further: for every citation
*occurrence* in a `GroundedAnswer` (not just each unique citation
number -- a repeated `[1]` is judged separately each time it appears),
a `CitationJudge` provider semantically verifies whether the cited
evidence actually supports the associated claim, returning a
`CitationVerificationReport` of `SUPPORTED`/`PARTIALLY_SUPPORTED`/
`UNSUPPORTED`/`CONTRADICTED` verdicts. `retrieve_generate_and_verify()`
composes retrieval, generation, and verification together, duplicating
none of the underlying stages.

Verdicts are a factual tally, never a calibrated confidence probability.
`CitationVerificationReport.all_supported` is a factual aggregate over
verdicts only -- it is vacuously `True` for an empty report (e.g. the
zero-citation insufficient-evidence response), so it is NOT by itself
sufficient for any future confidence/acceptance judgment: that would
also need to inspect `total_occurrences` and whether the answer is the
recognized insufficiency response (`is_insufficient_evidence_answer()`).
Confidence scoring and a final abstention ("I don't know") policy are
later pipeline stages and are not implemented here --
`is_insufficient_evidence_answer()` only recognizes the literal response
form the system prompt asks for; it is not a semantic judge, and
citation verification here checks *support*, not overall answer
trustworthiness or acceptance.
"""

from .base import CitationJudge, Generator, RawJudgeVerdict
from .citations import (
    extract_citation_occurrences,
    extract_citations,
    resolve_citation,
    validate_citations,
    validate_evidence_numbering,
)
from .context import build_evidence, format_evidence_block
from .exceptions import (
    CitationJudgeError,
    CitationJudgeOutputError,
    CitationValidationError,
    GenerationError,
    GenerationProviderError,
    InvalidGenerationInputError,
    RetrieveAndGenerateError,
    UncitedAnswerError,
)
from .judge_prompt import JUDGE_SYSTEM_PROMPT, annotate_answer, build_judge_user_prompt
from .models import (
    CitationOccurrence,
    CitationVerdict,
    CitationVerification,
    CitationVerificationReport,
    Evidence,
    GroundedAnswer,
)
from .openai import OpenAIGenerator
from .openai_judge import OpenAICitationJudge
from .prompt import SYSTEM_PROMPT, build_user_prompt, is_insufficient_evidence_answer
from .service import generate_grounded_answer, retrieve_and_generate
from .verification import retrieve_generate_and_verify, verify_grounded_answer

__all__ = [
    "JUDGE_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "CitationJudge",
    "CitationJudgeError",
    "CitationJudgeOutputError",
    "CitationOccurrence",
    "CitationValidationError",
    "CitationVerdict",
    "CitationVerification",
    "CitationVerificationReport",
    "Evidence",
    "GenerationError",
    "GenerationProviderError",
    "Generator",
    "GroundedAnswer",
    "InvalidGenerationInputError",
    "OpenAICitationJudge",
    "OpenAIGenerator",
    "RawJudgeVerdict",
    "RetrieveAndGenerateError",
    "UncitedAnswerError",
    "annotate_answer",
    "build_evidence",
    "build_judge_user_prompt",
    "build_user_prompt",
    "extract_citation_occurrences",
    "extract_citations",
    "format_evidence_block",
    "generate_grounded_answer",
    "is_insufficient_evidence_answer",
    "resolve_citation",
    "retrieve_and_generate",
    "retrieve_generate_and_verify",
    "validate_citations",
    "validate_evidence_numbering",
    "verify_grounded_answer",
]
