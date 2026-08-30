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
