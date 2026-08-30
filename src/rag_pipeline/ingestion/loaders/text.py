"""Plain-text (.txt) loader."""

from __future__ import annotations

from pathlib import Path

from ..exceptions import DocumentExtractionError
from ..models import Segment
from ..normalization import normalize_text


def load_text(path: Path, raw_bytes: bytes) -> list[Segment]:
    """Load a plain-text file into a single normalized segment.

    Decodes as UTF-8, transparently handling a leading UTF-8 BOM if present.
    """
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentExtractionError(f"File {path.name!r} is not valid UTF-8 text.") from exc

    normalized = normalize_text(text)
    return [Segment(text=normalized, section_heading=None, page_number=None)]
