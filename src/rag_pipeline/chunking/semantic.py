"""Semantic (embedding-similarity) chunking.

Algorithm, applied independently per ingestion `Segment`:

  1. Split the segment into semantic units (see `_split_semantic_units`).
  2. Embed all units in one batched call via the configured
     `EmbeddingProvider`.
  3. Walk consecutive units, computing cosine similarity between each pair.
     A topic boundary is introduced whenever similarity drops below
     `settings.chunk_semantic_similarity_threshold`.
  4. Units between boundaries are combined (joined by a blank line) into
     one chunk.
  5. Size safeguard: `settings.chunk_size` is treated as a hard maximum.
     Adding the next unit to the current group is refused (forcing a
     boundary there regardless of similarity) if it would exceed that
     maximum. If a single unit — or a forced group — is still larger than
     `chunk_size`, it is further split via the deterministic structural
     packer from `recursive.py` so no chunk can grow unbounded.

Semantic unit granularity — paragraphs: a segment is split on blank lines
first, falling back to single-newline lines when there are no blank-line
breaks, and finally treated as one unit if there's no line structure at
all. Sentence-level splitting was deliberately avoided: naive `.`-based
sentence boundaries are unreliable for technical documentation full of
version numbers ("v1.2.3"), abbreviations ("e.g."), URLs ("example.com"),
and config keys. Paragraph-level units sidestep that ambiguity entirely, at
the cost of coarser topic-boundary granularity — an acceptable tradeoff for
this domain.
"""

from __future__ import annotations

from ..config import ChunkingStrategy, Settings
from ..embeddings import (
    EmbeddingProvider,
    EmbeddingProviderError,
    OpenAIEmbeddingProvider,
    cosine_similarity,
)
from ..ingestion.models import NormalizedDocument
from .models import Chunk, build_chunk
from .recursive import pack_structural


def _split_semantic_units(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    units = [unit.strip() for unit in text.split("\n\n") if unit.strip()]
    if len(units) > 1:
        return units

    units = [unit.strip() for unit in text.split("\n") if unit.strip()]
    if len(units) > 1:
        return units

    return [text]


def _group_units(
    units: list[str],
    embeddings: list[list[float]],
    similarity_threshold: float,
    max_size: int,
) -> list[str]:
    """Group semantic units into chunk texts using similarity + a size safeguard."""
    groups: list[list[str]] = [[units[0]]]
    group_lengths = [len(units[0])]

    for index in range(1, len(units)):
        unit = units[index]
        similarity = cosine_similarity(embeddings[index - 1], embeddings[index])
        combined_length = group_lengths[-1] + 2 + len(unit)  # +2 for the "\n\n" join

        if similarity < similarity_threshold or combined_length > max_size:
            groups.append([unit])
            group_lengths.append(len(unit))
        else:
            groups[-1].append(unit)
            group_lengths[-1] = combined_length

    joined = ["\n\n".join(group) for group in groups]

    # Safeguard: a group that's still oversized (e.g. one enormous unit) is
    # split deterministically so no chunk exceeds max_size.
    result: list[str] = []
    for text in joined:
        if len(text) <= max_size:
            result.append(text)
        else:
            result.extend(pack_structural(text, max_size))
    return result


def chunk_semantic(
    document: NormalizedDocument,
    settings: Settings,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[Chunk]:
    """Chunk each segment independently using embedding-similarity topic boundaries."""
    if embedding_provider is None:
        embedding_provider = OpenAIEmbeddingProvider(settings)

    chunks: list[Chunk] = []
    chunk_index = 0
    for segment in document.segments:
        units = _split_semantic_units(segment.text)
        if not units:
            continue

        if len(units) == 1:
            unit = units[0]
            pieces = (
                [unit]
                if len(unit) <= settings.chunk_size
                else pack_structural(unit, settings.chunk_size)
            )
        else:
            embeddings = embedding_provider.embed(units)
            if len(embeddings) != len(units):
                raise EmbeddingProviderError(
                    f"Embedding provider returned {len(embeddings)} vectors for "
                    f"{len(units)} semantic units."
                )
            pieces = _group_units(
                units, embeddings, settings.chunk_semantic_similarity_threshold, settings.chunk_size
            )

        for piece in pieces:
            stripped = piece.strip()
            if not stripped:
                continue
            chunks.append(
                build_chunk(
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    text=stripped,
                    source_file=document.source_file,
                    section_heading=segment.section_heading,
                    page_number=segment.page_number,
                    strategy=ChunkingStrategy.SEMANTIC,
                )
            )
            chunk_index += 1
    return chunks
