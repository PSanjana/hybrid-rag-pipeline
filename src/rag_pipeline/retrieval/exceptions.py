"""Domain-specific exceptions for the retrieval pipeline."""


class RetrievalError(Exception):
    """Base class for all retrieval-related errors."""


class InvalidQueryError(RetrievalError):
    """Raised when a query string is empty or whitespace-only."""


class IndexNotReadyError(RetrievalError):
    """Raised when no active manifest/snapshot exists for the requested strategy."""


class EmbeddingModelMismatchError(RetrievalError):
    """Raised when the configured embedding model does not match the active index's model."""


class DenseRetrievalError(RetrievalError):
    """Raised when a query embedding or a Chroma query response cannot be trusted/parsed."""
