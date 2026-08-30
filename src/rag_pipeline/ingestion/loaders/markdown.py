"""Markdown (.md, .markdown) loader.

Markdown structure is parsed (not treated as arbitrary text) so that headings
and code blocks are identified explicitly rather than left as raw markup.

Heading representation: each segment's `section_heading` is the *full
ancestor path* of active headings at that point in the document, joined with
" > " (e.g. "Authentication > JWT Tokens"). This is more informative than
just the closest heading because it disambiguates same-named subsections
that live under different parents.
"""

from __future__ import annotations

from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from ..exceptions import DocumentExtractionError
from ..models import Segment
from ..normalization import normalize_text

_HEADING_SPAN = 3  # heading_open, inline, heading_close


def _render_inline_text(token: Token) -> str:
    """Render the visible text of an inline token, dropping markup markers."""
    if not token.children:
        return token.content

    parts: list[str] = []
    for child in token.children:
        if child.type in ("text", "code_inline"):
            parts.append(child.content)
        elif child.type in ("softbreak", "hardbreak"):
            parts.append("\n")
    return "".join(parts)


def load_markdown(path: Path, raw_bytes: bytes) -> list[Segment]:
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentExtractionError(f"File {path.name!r} is not valid UTF-8 text.") from exc

    md = MarkdownIt("commonmark")
    tokens = md.parse(text)

    sections: list[tuple[str | None, str]] = []
    heading_stack: list[tuple[int, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if not current_lines:
            return
        content = "\n\n".join(current_lines)
        if content.strip():
            sections.append((current_heading, content))
        current_lines.clear()

    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token.type == "heading_open":
            flush()
            level = int(token.tag[1:])
            inline_token = tokens[index + 1]
            heading_text = _render_inline_text(inline_token).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            current_heading = " > ".join(text for _, text in heading_stack)
            index += _HEADING_SPAN
            continue

        if token.type == "inline":
            rendered = _render_inline_text(token).strip()
            if rendered:
                current_lines.append(rendered)
        elif token.type in ("fence", "code_block"):
            code = token.content.rstrip("\n")
            if code:
                current_lines.append(code)

        index += 1

    flush()

    if not sections:
        raise DocumentExtractionError(f"No extractable content found in {path.name!r}.")

    return [
        Segment(text=normalize_text(content), section_heading=heading, page_number=None)
        for heading, content in sections
    ]
