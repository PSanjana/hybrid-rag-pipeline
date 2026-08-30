"""Local HTML (.html, .htm) loader.

Only parses HTML content supplied directly to the ingestion system; this is
not a web crawler and never fetches remote resources.
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup, Comment, Tag

from ..exceptions import DocumentExtractionError
from ..models import Segment
from ..normalization import normalize_text

_NOISE_TAGS = ("script", "style", "noscript", "head", "template")
_HEADING_LEVELS = {f"h{level}": level for level in range(1, 7)}
_CONTENT_TAGS = (*_HEADING_LEVELS, "p", "li", "pre")


def load_html(path: Path, raw_bytes: bytes) -> list[Segment]:
    try:
        html_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentExtractionError(f"File {path.name!r} is not valid UTF-8 text.") from exc

    soup = BeautifulSoup(html_text, "html.parser")

    for tag_name in _NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for comment in soup.find_all(string=lambda node: isinstance(node, Comment)):
        comment.extract()

    root = soup.body or soup

    sections: list[tuple[str | None, list[str]]] = []
    heading_stack: list[tuple[int, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_lines:
            sections.append((current_heading, list(current_lines)))
            current_lines.clear()

    for element in root.find_all(list(_CONTENT_TAGS)):
        if not isinstance(element, Tag):
            continue
        # Skip elements already contained in another selected content element
        # (e.g. a nested <li>, or <p> inside <li>) to avoid duplicate text.
        if element.find_parent(list(_CONTENT_TAGS)) is not None:
            continue

        text = element.get_text(" ", strip=True)
        if not text:
            continue

        if element.name in _HEADING_LEVELS:
            flush()
            level = _HEADING_LEVELS[element.name]
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
            current_heading = " > ".join(heading_text for _, heading_text in heading_stack)
        else:
            current_lines.append(text)

    flush()

    if not sections:
        raise DocumentExtractionError(f"No extractable content found in {path.name!r}.")

    return [
        Segment(
            text=normalize_text("\n\n".join(lines)),
            section_heading=heading,
            page_number=None,
        )
        for heading, lines in sections
    ]
