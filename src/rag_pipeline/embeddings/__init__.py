"""Shared embedding-provider abstraction, used by both semantic chunking and dense indexing."""

from .base import EmbeddingProvider
from .exceptions import EmbeddingProviderError
from .openai import OpenAIEmbeddingProvider

__all__ = ["EmbeddingProvider", "EmbeddingProviderError", "OpenAIEmbeddingProvider"]
