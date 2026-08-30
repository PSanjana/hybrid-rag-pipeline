"""Dense (embedding), sparse (BM25), RRF-fused hybrid, and reranked retrieval over an active index.

Given a question, `retrieve_dense()` embeds it with the same shared
`EmbeddingProvider` used at indexing time and queries the requested
chunking strategy's active Chroma collection via cosine nearest-neighbor
search; `retrieve_sparse()` tokenizes it with the same shared BM25
tokenizer used at indexing time and scores the active BM25 sparse corpus.
Both resolve their active snapshot solely via its manifest (never
guessed), and return ranked, provenance-carrying results.

`retrieve_hybrid()` orchestrates both channels and fuses their rankings
via pure weighted Reciprocal Rank Fusion (`fuse_rankings()`, see
`retrieval.fusion`) -- combining rank *positions*, never the incompatible
raw cosine/BM25 score scales.

`retrieve_reranked()` requests a wider hybrid candidate pool
(`rerank_candidate_k`, default 20) and reorders it with a cross-encoder-
style `Reranker` (see `rag_pipeline.reranking`) via pure
`rerank_candidates()`, returning the final top `rerank_top_k` (default 5)
chunks. Answer generation and citations are later pipeline stages and are
not implemented here.
"""

from .dense import retrieve_dense
from .exceptions import (
    DenseRetrievalError,
    EmbeddingModelMismatchError,
    FusionError,
    HybridRetrievalError,
    IndexNotReadyError,
    InvalidQueryError,
    RerankedRetrievalError,
    RerankError,
    RetrievalError,
    SparseRetrievalError,
    TokenizerVersionMismatchError,
)
from .fusion import fuse_rankings
from .hybrid import retrieve_hybrid
from .models import (
    DenseRetrievalResult,
    HybridRetrievalResult,
    RerankedRetrievalResult,
    SparseRetrievalResult,
)
from .rerank import rerank_candidates
from .reranked import retrieve_reranked
from .sparse import retrieve_sparse

__all__ = [
    "DenseRetrievalError",
    "DenseRetrievalResult",
    "EmbeddingModelMismatchError",
    "FusionError",
    "HybridRetrievalError",
    "HybridRetrievalResult",
    "IndexNotReadyError",
    "InvalidQueryError",
    "RerankError",
    "RerankedRetrievalError",
    "RerankedRetrievalResult",
    "RetrievalError",
    "SparseRetrievalError",
    "SparseRetrievalResult",
    "TokenizerVersionMismatchError",
    "fuse_rankings",
    "rerank_candidates",
    "retrieve_dense",
    "retrieve_hybrid",
    "retrieve_reranked",
    "retrieve_sparse",
]
