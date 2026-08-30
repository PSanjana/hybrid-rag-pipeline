"""Reranker abstraction: score (query, document) pairs for relevance, provider-agnostic.

`Reranker` (see `base.py`) is the only thing the retrieval layer depends
on; a production cross-encoder (`CrossEncoderReranker`) and offline test
doubles are interchangeable implementations of it. This package has no
dependency on `rag_pipeline.retrieval` or its chunk-level models -- it
only ever deals in raw `(query, documents) -> scores`.
"""

from .base import Reranker
from .cross_encoder import CrossEncoderReranker
from .exceptions import RerankerError, RerankingError

__all__ = [
    "CrossEncoderReranker",
    "Reranker",
    "RerankerError",
    "RerankingError",
]
