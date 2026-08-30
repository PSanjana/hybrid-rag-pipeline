"""Domain-specific exceptions for the indexing pipeline."""


class IndexingError(Exception):
    """Base class for all indexing-related errors."""


class InvalidChunkCorpusError(IndexingError):
    """Raised when the input chunk corpus is malformed.

    Covers duplicate `chunk_id`s, empty/whitespace-only chunk text, chunks
    spanning more than one chunking strategy, and an empty corpus.
    """


class DenseIndexError(IndexingError):
    """Raised when building or verifying the Chroma dense index fails."""


class SparseIndexError(IndexingError):
    """Raised when building, persisting, or reconstructing the BM25 sparse index fails."""


class ManifestError(IndexingError):
    """Raised when reading or writing the index manifest fails."""


class SynchronizationError(IndexingError):
    """Raised when the dense and sparse indexes do not represent the same chunk corpus."""


class DedupReportError(IndexingError):
    """Raised when persisting or reading the deduplication duplicate report fails."""
