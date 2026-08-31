"""Domain-specific exceptions for grounded generation."""


class GenerationError(Exception):
    """Base class for all generation-related errors."""


class InvalidGenerationInputError(GenerationError):
    """Raised when a question is empty or whitespace-only."""


class GenerationProviderError(GenerationError):
    """Raised when a generation provider is misconfigured, fails, or returns an empty response."""


class CitationValidationError(GenerationError):
    """Raised when a generated answer cites a bracket number outside the supplied evidence."""


class UncitedAnswerError(GenerationError):
    """Raised when a factual answer has supplied evidence but zero bracket citations."""


class RetrieveAndGenerateError(GenerationError):
    """Raised when retrieval fails during retrieve_and_generate orchestration."""


class CitationJudgeError(GenerationError):
    """Raised when a citation-judge provider is misconfigured, fails, or returns nothing."""


class CitationJudgeOutputError(GenerationError):
    """Raised when a judge's output doesn't exactly match the expected occurrence set."""


class ConfidenceInputError(GenerationError):
    """Raised when a GroundedAnswer/CitationVerificationReport/retrieval-result set is malformed.

    Confidence scoring is a trust boundary: it never assumes its inputs
    were produced by the pipeline's own earlier stages, and fails
    clearly rather than compute a misleading score from inconsistent or
    incomplete data.
    """


class AbstentionPolicyInputError(GenerationError):
    """Raised when the abstention policy's inputs are mutually inconsistent or out of range.

    The Phase 3 Step 4 policy layer is another trust boundary: it
    re-checks only the `ConfidenceAssessment`/`CitationVerificationReport`/
    `GroundedAnswer` fields the decision actually depends on (score
    finiteness/range, citation counts vs the report, contradiction
    flag/count, insufficiency flag vs the answer's canonical form, and
    the configured threshold) and refuses to decide from a hand-built
    contradictory trio rather than emit a misleading FinalAnswer.
    """
