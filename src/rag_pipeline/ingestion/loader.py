"""Public entry point for dispatching to the correct format-specific loader."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .exceptions import UnsupportedFileTypeError
from .loaders.html import load_html
from .loaders.markdown import load_markdown
from .loaders.pdf import load_pdf
from .loaders.text import load_text
from .models import Segment

_LOADERS: dict[str, Callable[[Path, bytes], list[Segment]]] = {
    ".txt": load_text,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".html": load_html,
    ".htm": load_html,
    ".pdf": load_pdf,
}

_FILE_TYPES: dict[str, str] = {
    ".txt": "txt",
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
}


def supported_extensions() -> frozenset[str]:
    """Return the set of file extensions with a registered loader."""
    return frozenset(_LOADERS)


def load_segments(path: Path, raw_bytes: bytes) -> tuple[list[Segment], str]:
    """Dispatch to the loader matching `path`'s extension (case-insensitive).

    Returns the extracted segments and a normalized file_type label.
    Raises `UnsupportedFileTypeError` for unregistered extensions.
    """
    extension = path.suffix.lower()
    loader = _LOADERS.get(extension)
    if loader is None:
        raise UnsupportedFileTypeError(
            f"Unsupported file extension {path.suffix!r} for {path.name!r}. "
            f"Supported extensions: {sorted(_LOADERS)}."
        )
    segments = loader(path, raw_bytes)
    return segments, _FILE_TYPES[extension]
