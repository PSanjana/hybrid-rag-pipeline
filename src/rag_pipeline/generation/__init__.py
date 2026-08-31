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
`score_confidence()` goes one step further: it combines the
verification report's verdicts (dominant, default weight 0.9) with weak
corroborating dense+sparse dual-channel retrieval agreement for the
evidence actually cited (default weight 0.1) into one deterministic,
decomposable `ConfidenceAssessment`. This is a HEURISTIC QUALITY SIGNAL
-- explicitly not a calibrated probability, not a percentage chance of
correctness, and not an accept/reject decision. `is_insufficient_evidence_answer()`
only recognizes the literal response form the system prompt asks for;
it is not a semantic judge, and neither citation verification nor
confidence scoring here decide overall answer trustworthiness or
acceptance -- that final abstention ("I don't know") policy is Phase 3
Step 4's job, not this package's.
"""

from .base import CitationJudge, Generator, RawJudgeVerdict
from .citations import (
    extract_citation_occurrences,
    extract_citations,
    resolve_citation,
    validate_citations,
    validate_evidence_numbering,
)
from .confidence import retrieve_generate_verify_and_score, score_confidence
from .context import build_evidence, format_evidence_block
from .exceptions import (
    CitationJudgeError,
    CitationJudgeOutputError,
    CitationValidationError,
    ConfidenceInputError,
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
    ConfidenceAssessment,
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
    "ConfidenceAssessment",
    "ConfidenceInputError",
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
    "retrieve_generate_verify_and_score",
    "score_confidence",
    "validate_citations",
    "validate_evidence_numbering",
    "verify_grounded_answer",
]
