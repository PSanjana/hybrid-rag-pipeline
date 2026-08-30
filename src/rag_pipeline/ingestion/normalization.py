"""Conservative text normalization shared by all loaders.

This intentionally does *not* perform retrieval-specific preprocessing
(lowercasing, stemming, stopword removal, whitespace collapsing across
lines). It only fixes extraction/encoding artifacts while preserving
paragraph structure, code text, and technical content such as URLs and
identifiers.
"""

from __future__ import annotations

import re

_CRLF_RE = re.compile(r"\r\n?")
_NUL_RE = re.compile(r"\x00")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalize extracted text while preserving paragraph structure.

    - Normalizes CRLF/CR line endings to LF.
    - Removes NUL characters (a common PDF/encoding extraction artifact).
    - Strips trailing whitespace from each line.
    - Collapses runs of 3+ blank lines down to a single blank line.
    - Strips leading/trailing blank lines from the whole text.
    """
    if not text:
        return ""

    text = _NUL_RE.sub("", text)
    text = _CRLF_RE.sub("\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _EXCESS_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip("\n").strip()
