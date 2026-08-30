"""Public entry point for dispatching to the configured chunking strategy."""

from __future__ import annotations

import logging
from collections.abc import Callable

from ..config import ChunkingStrategy, Settings
from ..embeddings import EmbeddingProvider
from ..ingestion.models import NormalizedDocument
from .exceptions import UnsupportedChunkingStrategyError
from .fixed import chunk_fixed
from .models import Chunk
from .recursive import chunk_recursive
from .semantic import chunk_semantic

logger = logging.getLogger(__name__)

_STRATEGIES: dict[ChunkingStrategy, Callable[[NormalizedDocument, Settings], list[Chunk]]] = {
    ChunkingStrategy.FIXED: chunk_fixed,
    ChunkingStrategy.RECURSIVE: chunk_recursive,
}


def chunk_document(
    document: NormalizedDocument,
    strategy: ChunkingStrategy | None = None,
    settings: Settings | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[Chunk]:
    """Chunk `document` using the requested (or configured default) strategy.

    `embedding_provider` is only consulted for the semantic strategy; it is
    ignored (and no embedding provider is constructed) for fixed/recursive.
    """
    settings = settings or Settings()
    resolved_strategy = strategy or settings.chunk_strategy

    logger.info(
        "chunking started document_id=%s strategy=%s", document.document_id, resolved_strategy
    )

    if resolved_strategy == ChunkingStrategy.SEMANTIC:
        chunks = chunk_semantic(document, settings, embedding_provider=embedding_provider)
    else:
        chunker = _STRATEGIES.get(resolved_strategy)
        if chunker is None:
            raise UnsupportedChunkingStrategyError(
                f"Unsupported chunking strategy: {resolved_strategy!r}."
            )
        chunks = chunker(document, settings)

    logger.info(
        "chunking completed document_id=%s strategy=%s chunk_count=%d",
        document.document_id,
        resolved_strategy,
        len(chunks),
    )
    return chunks
