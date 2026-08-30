"""Domain-specific exceptions for the chunking pipeline.

`EmbeddingProviderError` used to live here, but embeddings are a shared
concern (used by both chunking and indexing) — it now lives in
`rag_pipeline.embeddings`. Import it from there.
"""


class ChunkingError(Exception):
    """Base class for all chunking-related errors."""


class UnsupportedChunkingStrategyError(ChunkingError):
    """Raised when a requested chunking strategy has no registered implementation."""
