"""Near-duplicate chunk detection: exact-text and cosine-similarity based, pre-indexing.

Operates purely on canonical `Chunk` objects and their precomputed
embeddings -- no dependency on Chroma or BM25. Used by
`rag_pipeline.indexing.service.index_chunks` to filter the chunk corpus
before either index is built, so both indexes are always built from the
same post-deduplication corpus.
"""

from .detector import deduplicate_chunks
from .exceptions import DeduplicationError
from .models import DEDUP_ALGORITHM_VERSION, DeduplicationResult, DuplicateRecord, DuplicateType

__all__ = [
    "DEDUP_ALGORITHM_VERSION",
    "DeduplicationError",
    "DeduplicationResult",
    "DuplicateRecord",
    "DuplicateType",
    "deduplicate_chunks",
]
