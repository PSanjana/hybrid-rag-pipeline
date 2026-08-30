"""Fixed-size chunking with overlap — the position-based baseline strategy.

`chunk_size` is a hard MAXIMUM number of characters per chunk. Windows are
sliced from a fixed stride sequence:

    start_i = i * (chunk_size - overlap)
    end_i   = min(start_i + chunk_size, len(text))

Deliberately no boundary-awareness: windows are raw character slices, never
adjusted toward whitespace. This keeps the two configuration guarantees
exact and independently verifiable:

  * every emitted chunk satisfies `len(chunk.text) <= chunk_size`,
  * every pair of consecutive (non-final) windows overlaps in *exactly*
    `chunk_overlap` characters of the source text.

This is intentionally the simplest of the three strategies — a clean,
position-only baseline to compare against `recursive.py` (structure-aware)
and `semantic.py` (topic-aware). It never falls back through progressively
smaller separators; see `recursive.py` for that behavior.
"""

from __future__ import annotations

from ..config import ChunkingStrategy, Settings
from ..ingestion.models import NormalizedDocument
from .models import Chunk, build_chunk


def split_fixed_windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split `text` into fixed-size, non-empty character windows.

    Outer whitespace is stripped exactly once, here, at the public entry
    point (see `_fixed_windows` for the non-stripping core used internally
    by `recursive.py`'s character-level fallback, where `text` may carry a
    meaningful trailing separator reattached by a higher-level split).

    Raises `ValueError` for a non-positive `chunk_size` or an `overlap` that
    isn't strictly smaller than `chunk_size` — both would make the window
    stride non-positive and loop forever otherwise.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and strictly smaller than chunk_size.")

    return _fixed_windows(text.strip(), chunk_size, overlap)


def _fixed_windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Core windowing logic. Assumes `chunk_size`/`overlap` are already valid."""
    length = len(text)
    if length == 0:
        return []

    step = chunk_size - overlap
    pieces: list[str] = []
    start = 0
    while start < length:
        end = min(start + chunk_size, length)
        piece = text[start:end]
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        start += step

    return pieces


def chunk_fixed(document: NormalizedDocument, settings: Settings) -> list[Chunk]:
    """Chunk each segment independently using fixed-size windows with overlap."""
    chunks: list[Chunk] = []
    chunk_index = 0
    for segment in document.segments:
        for piece in split_fixed_windows(segment.text, settings.chunk_size, settings.chunk_overlap):
            chunks.append(
                build_chunk(
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    text=piece,
                    source_file=document.source_file,
                    section_heading=segment.section_heading,
                    page_number=segment.page_number,
                    strategy=ChunkingStrategy.FIXED,
                )
            )
            chunk_index += 1
    return chunks
