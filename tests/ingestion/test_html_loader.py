"""Tests for rag_pipeline.ingestion.loaders.html."""

from pathlib import Path

from rag_pipeline.ingestion.loaders.html import load_html

HTML = """\
<html>
<head><title>Doc</title></head>
<body>
<script>var x = "should not appear";</script>
<style>.noise { color: red; }</style>
<h1>Getting Started</h1>
<p>Intro paragraph text.</p>
<h2>Installation</h2>
<p>Run the installer.</p>
<ul>
<li>Step one</li>
<li>Step two</li>
</ul>
<pre>code --flag value</pre>
</body>
</html>
"""


def test_headings_are_detected() -> None:
    segments = load_html(Path("doc.html"), HTML.encode())
    headings = [s.section_heading for s in segments]
    assert "Getting Started" in headings
    assert "Getting Started > Installation" in headings


def test_scripts_and_styles_are_excluded() -> None:
    segments = load_html(Path("doc.html"), HTML.encode())
    full_text = "\n".join(s.text for s in segments)
    assert "should not appear" not in full_text
    assert "color: red" not in full_text


def test_paragraph_and_list_content_is_retained() -> None:
    segments = load_html(Path("doc.html"), HTML.encode())
    full_text = "\n".join(s.text for s in segments)
    assert "Intro paragraph text." in full_text
    assert "Run the installer." in full_text
    assert "Step one" in full_text
    assert "Step two" in full_text
    assert "code --flag value" in full_text


def test_page_number_is_none() -> None:
    segments = load_html(Path("doc.html"), HTML.encode())
    assert all(s.page_number is None for s in segments)


def test_no_duplicate_extraction_from_nested_elements() -> None:
    nested_html = """
    <html><body>
    <ul>
      <li>Outer item
        <ul><li>Inner item</li></ul>
      </li>
    </ul>
    </body></html>
    """
    segments = load_html(Path("doc.html"), nested_html.encode())
    full_text = "\n".join(s.text for s in segments)
    assert full_text.count("Inner item") == 1
