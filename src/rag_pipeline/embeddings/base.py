"""Shared embedding-provider abstraction.

Used by both semantic chunking (`rag_pipeline.chunking.semantic`) and the
dense indexing layer (`rag_pipeline.indexing.dense`), so the project has
exactly one embedding abstraction and one production implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order."""
        ...
