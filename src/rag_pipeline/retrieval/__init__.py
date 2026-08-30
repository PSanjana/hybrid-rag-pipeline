"""Dense (embedding) and sparse (BM25) retrieval over an active per-strategy index snapshot.

Given a question, `retrieve_dense()` embeds it with the same shared
`EmbeddingProvider` used at indexing time and queries the requested
chunking strategy's active Chroma collection via cosine nearest-neighbor
search; `retrieve_sparse()` tokenizes it with the same shared BM25
tokenizer used at indexing time and scores the active BM25 sparse corpus.
Both resolve their active snapshot solely via its manifest (never
guessed), and return ranked, provenance-carrying results. Reciprocal Rank
Fusion, hybrid dense+sparse merging, reranking, and generation are later
pipeline stages and are not implemented here.
"""

from .dense import retrieve_dense
from .exceptions import (
    DenseRetrievalError,
    EmbeddingModelMismatchError,
    IndexNotReadyError,
    InvalidQueryError,
    RetrievalError,
    SparseRetrievalError,
    TokenizerVersionMismatchError,
)
from .models import DenseRetrievalResult, SparseRetrievalResult
from .sparse import retrieve_sparse

__all__ = [
    "DenseRetrievalError",
    "DenseRetrievalResult",
    "EmbeddingModelMismatchError",
    "IndexNotReadyError",
    "InvalidQueryError",
    "RetrievalError",
    "SparseRetrievalError",
    "SparseRetrievalResult",
    "TokenizerVersionMismatchError",
    "retrieve_dense",
    "retrieve_sparse",
]
