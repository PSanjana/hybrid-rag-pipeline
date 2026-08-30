"""Shared embedding-provider abstraction, used by both semantic chunking and dense indexing."""

from .base import EmbeddingProvider
from .exceptions import EmbeddingProviderError
from .openai import OpenAIEmbeddingProvider
from .similarity import cosine_similarity
from .validation import validate_consistent_dimensionality, validate_vector

__all__ = [
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "OpenAIEmbeddingProvider",
    "cosine_similarity",
    "validate_consistent_dimensionality",
    "validate_vector",
]
