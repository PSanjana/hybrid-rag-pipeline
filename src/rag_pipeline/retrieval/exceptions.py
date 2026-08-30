"""Domain-specific exceptions for the retrieval pipeline."""


class RetrievalError(Exception):
    """Base class for all retrieval-related errors."""


class InvalidQueryError(RetrievalError):
    """Raised when a query is empty/whitespace-only, or (sparse) tokenizes to nothing."""


class IndexNotReadyError(RetrievalError):
    """Raised when no active manifest/snapshot exists for the requested strategy."""


class EmbeddingModelMismatchError(RetrievalError):
    """Raised when the configured embedding model does not match the active index's model."""


class TokenizerVersionMismatchError(RetrievalError):
    """Raised when the active index's BM25 tokenizer version doesn't match the runtime tokenizer."""


class DenseRetrievalError(RetrievalError):
    """Raised when a query embedding or a Chroma query response cannot be trusted/parsed."""


class SparseRetrievalError(RetrievalError):
    """Raised when BM25 reconstruction/scoring or Chroma result hydration cannot be trusted."""
