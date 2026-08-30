"""Tests for rag_pipeline.ingestion.loaders.markdown."""

from pathlib import Path

from rag_pipeline.ingestion.loaders.markdown import load_markdown

MARKDOWN = """\
# Authentication

Intro paragraph about authentication.

## JWT Tokens

JWT content goes here.

```python
def verify(token):
    return True
```

## API Keys

API key content.

# Deployment

Deployment intro.
"""


def test_headings_are_detected() -> None:
    segments = load_markdown(Path("doc.md"), MARKDOWN.encode())
    headings = [s.section_heading for s in segments]
    assert "Authentication" in headings
    assert "Authentication > JWT Tokens" in headings
    assert "Authentication > API Keys" in headings
    assert "Deployment" in headings


def test_section_content_is_associated_with_headings() -> None:
    segments = load_markdown(Path("doc.md"), MARKDOWN.encode())
    jwt_segment = next(s for s in segments if s.section_heading == "Authentication > JWT Tokens")
    assert "JWT content goes here." in jwt_segment.text


def test_code_block_contents_are_preserved() -> None:
    segments = load_markdown(Path("doc.md"), MARKDOWN.encode())
    jwt_segment = next(s for s in segments if s.section_heading == "Authentication > JWT Tokens")
    assert "def verify(token):" in jwt_segment.text
    assert "return True" in jwt_segment.text


def test_page_number_is_none_for_all_segments() -> None:
    segments = load_markdown(Path("doc.md"), MARKDOWN.encode())
    assert all(s.page_number is None for s in segments)


def test_sibling_headings_reset_the_ancestor_path() -> None:
    segments = load_markdown(Path("doc.md"), MARKDOWN.encode())
    api_keys = next(s for s in segments if s.section_heading == "Authentication > API Keys")
    deployment = next(s for s in segments if s.section_heading == "Deployment")
    assert "API key content." in api_keys.text
    assert "Deployment intro." in deployment.text


def test_content_before_first_heading_has_no_section_heading() -> None:
    text = "Just a plain paragraph with no heading."
    segments = load_markdown(Path("doc.md"), text.encode())
    assert len(segments) == 1
    assert segments[0].section_heading is None
    assert segments[0].text == text
