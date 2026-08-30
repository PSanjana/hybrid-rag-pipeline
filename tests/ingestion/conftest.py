"""Shared fixtures and a minimal hand-built PDF generator for ingestion tests.

The PDF builder avoids depending on a heavyweight PDF-generation library
(e.g. reportlab) purely for tests: it hand-assembles a tiny valid PDF with
a real, correctly-offset xref table and real text-drawing content streams,
so PDF loader tests exercise genuine `pypdf` extraction rather than mocks.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from rag_pipeline.config import Settings


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_minimal_pdf(pages_text: list[str | None]) -> bytes:
    """Build a minimal valid single/multi-page PDF.

    Each entry in `pages_text` becomes one page; `None` produces a page with
    an empty content stream (i.e. a genuinely blank page with no text).
    """
    page_count = len(pages_text)
    font_obj = 3
    page_objs = [4 + i for i in range(page_count)]
    content_objs = [4 + page_count + i for i in range(page_count)]

    objects: dict[int, bytes] = {}
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{num} 0 R" for num in page_objs)
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode()
    objects[font_obj] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for index, text in enumerate(pages_text):
        page_obj = page_objs[index]
        content_obj = content_objs[index]
        objects[page_obj] = (
            f"<< /Type /Page /Parent 2 0 R "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> "
            f"/MediaBox [0 0 300 300] /Contents {content_obj} 0 R >>"
        ).encode()

        if text:
            stream_body = f"BT /F1 12 Tf 72 200 Td ({_escape_pdf_text(text)}) Tj ET".encode()
        else:
            stream_body = b""
        objects[content_obj] = (
            f"<< /Length {len(stream_body)} >>\nstream\n".encode() + stream_body + b"\nendstream"
        )

    ordered_nums = sorted(objects)
    max_obj_num = ordered_nums[-1]

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num in ordered_nums:
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode()
        out += objects[num]
        out += b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {max_obj_num + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for num in range(1, max_obj_num + 1):
        offset = offsets.get(num)
        if offset is None:
            out += b"0000000000 00000 f \n"
        else:
            out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {max_obj_num + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref_offset}\n".encode()
    out += b"%%EOF"
    return bytes(out)


@pytest.fixture
def build_pdf_bytes() -> Callable[[list[str | None]], bytes]:
    return build_minimal_pdf


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
    )
