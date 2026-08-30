"""Configurable chunking: `NormalizedDocument` -> retrieval-oriented `Chunk` objects.

Three switchable strategies are available (see
`rag_pipeline.config.ChunkingStrategy`): fixed-size with overlap,
recursive/structure-aware, and embedding-similarity-based semantic
chunking. This package stops at producing `Chunk` objects — indexing,
retrieval, and reranking are later pipeline stages and are not implemented
here.
"""

from ..config import ChunkingStrategy
from ..embeddings import EmbeddingProvider, EmbeddingProviderError, OpenAIEmbeddingProvider
from .dispatcher import chunk_document
from .exceptions import ChunkingError, UnsupportedChunkingStrategyError
from .models import Chunk, build_chunk, compute_chunk_id

__all__ = [
    "Chunk",
    "ChunkingError",
    "ChunkingStrategy",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "OpenAIEmbeddingProvider",
    "UnsupportedChunkingStrategyError",
    "build_chunk",
    "chunk_document",
    "compute_chunk_id",
]
