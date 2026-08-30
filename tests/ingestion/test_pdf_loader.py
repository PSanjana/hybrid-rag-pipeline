"""Tests for rag_pipeline.ingestion.loaders.pdf."""

from collections.abc import Callable
from pathlib import Path

import pytest

from rag_pipeline.ingestion.exceptions import DocumentExtractionError, NoExtractableTextError
from rag_pipeline.ingestion.loaders.pdf import load_pdf

PdfBuilder = Callable[[list[str | None]], bytes]


def test_text_based_pdf_extraction(build_pdf_bytes: PdfBuilder) -> None:
    data = build_pdf_bytes(["Hello from page one."])
    segments = load_pdf(Path("doc.pdf"), data)
    assert len(segments) == 1
    assert segments[0].text == "Hello from page one."


def test_pages_retain_correct_1_based_page_numbers(build_pdf_bytes: PdfBuilder) -> None:
    data = build_pdf_bytes(["Page one.", "Page two.", "Page three."])
    segments = load_pdf(Path("doc.pdf"), data)
    assert [s.page_number for s in segments] == [1, 2, 3]
    assert [s.text for s in segments] == ["Page one.", "Page two.", "Page three."]


def test_empty_pages_are_skipped_but_page_numbers_preserved(build_pdf_bytes: PdfBuilder) -> None:
    data = build_pdf_bytes(["Page one text.", None, "Page three text."])
    segments = load_pdf(Path("doc.pdf"), data)
    assert [s.page_number for s in segments] == [1, 3]
    assert [s.text for s in segments] == ["Page one text.", "Page three text."]


def test_no_extractable_text_raises_dedicated_error(build_pdf_bytes: PdfBuilder) -> None:
    data = build_pdf_bytes([None, None])
    with pytest.raises(NoExtractableTextError):
        load_pdf(Path("scanned.pdf"), data)


def test_corrupt_pdf_raises_document_extraction_error() -> None:
    with pytest.raises(DocumentExtractionError):
        load_pdf(Path("corrupt.pdf"), b"%PDF-1.4\nthis is not a real pdf body")
