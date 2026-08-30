"""Exceptions for the embeddings package.

Standalone (not a `ChunkingError`/`IndexingError` subclass): embeddings are
a shared concern used by both chunking and indexing, not owned by either.
"""


class EmbeddingProviderError(Exception):
    """Raised when an embedding provider is misconfigured or an embedding request fails."""
