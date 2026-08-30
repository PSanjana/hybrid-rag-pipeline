"""Tests for rag_pipeline.ingestion.loader dispatch behavior."""

from pathlib import Path

import pytest

from rag_pipeline.ingestion.exceptions import UnsupportedFileTypeError
from rag_pipeline.ingestion.loader import load_segments, supported_extensions


def test_supported_extensions_are_explicit() -> None:
    assert supported_extensions() == frozenset(
        {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}
    )


def test_dispatches_txt_to_text_loader() -> None:
    segments, file_type = load_segments(Path("notes.txt"), b"hello world")
    assert file_type == "txt"
    assert segments[0].text == "hello world"


def test_dispatch_is_case_insensitive() -> None:
    segments, file_type = load_segments(Path("NOTES.TXT"), b"hello world")
    assert file_type == "txt"
    assert segments[0].text == "hello world"


def test_unsupported_extension_raises() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        load_segments(Path("archive.zip"), b"binary data")


def test_unsupported_extension_message_names_extension() -> None:
    with pytest.raises(UnsupportedFileTypeError, match=r"\.zip"):
        load_segments(Path("archive.zip"), b"binary data")
