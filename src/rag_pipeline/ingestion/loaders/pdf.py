"""Text-based PDF loader.

Extracts embedded text per page using `pypdf`. Scanned/image-only PDFs have
no embedded text layer, so OCR would be required to extract content from
them; that is explicitly out of scope, and such files raise
`NoExtractableTextError` rather than being silently ingested as empty.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ..exceptions import DocumentExtractionError, NoExtractableTextError
from ..models import Segment
from ..normalization import normalize_text


def load_pdf(path: Path, raw_bytes: bytes) -> list[Segment]:
    try:
        reader = PdfReader(BytesIO(raw_bytes))
    except (PdfReadError, ValueError) as exc:
        raise DocumentExtractionError(
            f"Could not read PDF {path.name!r}: file may be corrupt or invalid."
        ) from exc

    if reader.is_encrypted:
        raise DocumentExtractionError(
            f"PDF {path.name!r} is encrypted and cannot be read without a password."
        )

    segments: list[Segment] = []
    # Pages with no extractable text (e.g. blank pages) are skipped: they
    # contribute no segment, but surrounding pages keep their true 1-based
    # page_number rather than being renumbered.
    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text()
        normalized = normalize_text(raw_text or "")
        if normalized:
            segments.append(Segment(text=normalized, section_heading=None, page_number=page_number))

    if not segments:
        raise NoExtractableTextError(
            f"No extractable text found in {path.name!r}. The file may be a "
            "scanned/image-only PDF; OCR is not implemented."
        )

    return segments
