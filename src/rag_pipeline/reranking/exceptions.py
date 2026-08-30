"""Domain-specific exceptions for the reranking abstraction."""


class RerankingError(Exception):
    """Base class for all reranking-related errors."""


class RerankerError(RerankingError):
    """Raised when a reranker provider's underlying model/runtime call fails or misbehaves."""
