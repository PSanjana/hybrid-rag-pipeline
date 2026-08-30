"""Tests for rag_pipeline.ingestion.normalization."""

from rag_pipeline.ingestion.normalization import normalize_text


def test_preserves_paragraph_boundaries() -> None:
    text = "First paragraph.\n\nSecond paragraph."
    assert normalize_text(text) == "First paragraph.\n\nSecond paragraph."


def test_normalizes_crlf_line_endings() -> None:
    text = "Line one.\r\nLine two.\r\n\r\nParagraph two."
    result = normalize_text(text)
    assert "\r" not in result
    assert result == "Line one.\nLine two.\n\nParagraph two."


def test_normalizes_bare_cr_line_endings() -> None:
    text = "Line one.\rLine two."
    assert normalize_text(text) == "Line one.\nLine two."


def test_strips_trailing_whitespace_per_line() -> None:
    text = "Line one.   \nLine two.\t\n"
    result = normalize_text(text)
    assert result == "Line one.\nLine two."


def test_collapses_excessive_blank_lines() -> None:
    text = "Paragraph one.\n\n\n\n\nParagraph two."
    assert normalize_text(text) == "Paragraph one.\n\nParagraph two."


def test_strips_leading_and_trailing_blank_lines() -> None:
    text = "\n\n  Content here.  \n\n"
    assert normalize_text(text) == "Content here."


def test_removes_nul_characters() -> None:
    text = "Some\x00text"
    assert normalize_text(text) == "Sometext"


def test_preserves_urls_and_identifiers() -> None:
    text = "See https://example.com/path?x=1 or CONFIG_KEY_NAME or ERR-4042."
    assert normalize_text(text) == text


def test_empty_string_returns_empty_string() -> None:
    assert normalize_text("") == ""
