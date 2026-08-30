"""Tests for rag_pipeline.ingestion.loaders.text."""

from pathlib import Path

import pytest

from rag_pipeline.ingestion.exceptions import DocumentExtractionError
from rag_pipeline.ingestion.loaders.text import load_text


def test_loads_plain_utf8_text() -> None:
    segments = load_text(Path("notes.txt"), b"Hello, world.\n\nSecond paragraph.")
    assert len(segments) == 1
    assert segments[0].text == "Hello, world.\n\nSecond paragraph."
    assert segments[0].section_heading is None
    assert segments[0].page_number is None


def test_loads_utf8_with_bom() -> None:
    raw = "Hello with BOM.".encode("utf-8-sig")
    segments = load_text(Path("notes.txt"), raw)
    assert segments[0].text == "Hello with BOM."
    assert not segments[0].text.startswith("﻿")


def test_invalid_utf8_raises_document_extraction_error() -> None:
    invalid_bytes = b"\xff\xfe\x00not valid utf-8 alone \x80\x81"
    with pytest.raises(DocumentExtractionError):
        load_text(Path("bad.txt"), invalid_bytes)
