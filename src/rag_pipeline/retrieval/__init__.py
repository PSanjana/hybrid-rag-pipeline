"""Dense (embedding-based) retrieval over an active per-strategy Chroma snapshot.

Given a question, `retrieve_dense()` embeds it with the same shared
`EmbeddingProvider` used at indexing time, queries the requested chunking
strategy's active Chroma collection (resolved via its manifest, never
guessed), and returns ranked, provenance-carrying results. Sparse (BM25)
retrieval, hybrid fusion, reranking, and generation are later pipeline
stages and are not implemented here.
"""

from .dense import retrieve_dense
from .exceptions import (
    DenseRetrievalError,
    EmbeddingModelMismatchError,
    IndexNotReadyError,
    InvalidQueryError,
    RetrievalError,
)
from .models import DenseRetrievalResult

__all__ = [
    "DenseRetrievalError",
    "DenseRetrievalResult",
    "EmbeddingModelMismatchError",
    "IndexNotReadyError",
    "InvalidQueryError",
    "RetrievalError",
    "retrieve_dense",
]
