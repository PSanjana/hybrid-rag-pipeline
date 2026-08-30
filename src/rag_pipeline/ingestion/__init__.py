"""Multi-format document ingestion and normalization.

Turns a local source file (.txt/.md/.markdown/.html/.htm/.pdf) into a
canonical `NormalizedDocument` of provenance metadata plus normalized
content `Segment`s, and persists both the raw source and the processed
representation. Chunking, embedding, and retrieval are separate, later
pipeline stages and are not implemented here.
"""

from .exceptions import (
    DocumentExtractionError,
    IngestionError,
    NoExtractableTextError,
    PersistenceError,
    SourceNotFoundError,
    UnsupportedFileTypeError,
)
from .loader import supported_extensions
from .models import NormalizedDocument, Segment
from .service import ingest_document

__all__ = [
    "DocumentExtractionError",
    "IngestionError",
    "NoExtractableTextError",
    "NormalizedDocument",
    "PersistenceError",
    "Segment",
    "SourceNotFoundError",
    "UnsupportedFileTypeError",
    "ingest_document",
    "supported_extensions",
]
