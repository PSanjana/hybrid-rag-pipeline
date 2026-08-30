"""Domain-specific exceptions for the ingestion pipeline."""


class IngestionError(Exception):
    """Base class for all ingestion-related errors."""


class SourceNotFoundError(IngestionError):
    """Raised when the given source path does not exist or is not a regular file."""


class UnsupportedFileTypeError(IngestionError):
    """Raised when a source file's extension has no registered loader."""


class DocumentExtractionError(IngestionError):
    """Raised when a loader fails to extract usable content from a source file."""


class NoExtractableTextError(DocumentExtractionError):
    """Raised when a source yields no extractable text at all (e.g. a scanned PDF)."""


class PersistenceError(IngestionError):
    """Raised when raw or processed document data cannot be persisted or read reliably."""
