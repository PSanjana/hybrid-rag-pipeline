"""Tests for rag_pipeline.chunking.fixed."""

import pytest

from rag_pipeline.chunking.fixed import chunk_fixed, split_fixed_windows
from rag_pipeline.config import ChunkingStrategy, Settings
from rag_pipeline.ingestion.models import Segment

from .conftest import make_document


def test_short_segment_produces_one_chunk() -> None:
    pieces = split_fixed_windows("short text", chunk_size=100, overlap=20)
    assert pieces == ["short text"]


def test_exact_boundary_length_produces_one_chunk() -> None:
    text = "x" * 100
    pieces = split_fixed_windows(text, chunk_size=100, overlap=20)
    assert pieces == [text]


def test_long_text_splits_into_multiple_pieces() -> None:
    text = "x" * 250
    pieces = split_fixed_windows(text, chunk_size=100, overlap=20)
    assert len(pieces) > 1


def test_no_empty_chunks() -> None:
    text = "word " * 500
    pieces = split_fixed_windows(text, chunk_size=50, overlap=10)
    assert all(piece.strip() for piece in pieces)


def test_overlap_is_directly_inspectable_on_no_whitespace_text() -> None:
    # No spaces/newlines near the cut points, so hard windows apply exactly:
    # start_i = i * (chunk_size - overlap); consecutive windows share exactly
    # `overlap` raw characters.
    text = "".join(str(i % 10) for i in range(300))
    chunk_size, overlap = 100, 20
    pieces = split_fixed_windows(text, chunk_size=chunk_size, overlap=overlap)

    step = chunk_size - overlap
    for index in range(len(pieces) - 1):
        expected_overlap = text[index * step + chunk_size - overlap : index * step + chunk_size]
        assert pieces[index].endswith(expected_overlap)
        assert pieces[index + 1].startswith(expected_overlap)


def test_deterministic_results() -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 20
    first = split_fixed_windows(text, chunk_size=80, overlap=15)
    second = split_fixed_windows(text, chunk_size=80, overlap=15)
    assert first == second


def test_smallest_valid_overlap_step_terminates() -> None:
    # overlap == chunk_size - 1 is the smallest valid step (step == 1); must terminate.
    text = "x" * 500
    pieces = split_fixed_windows(text, chunk_size=10, overlap=9)
    assert pieces  # terminates and produces output


def test_overlap_equal_to_chunk_size_raises_instead_of_looping() -> None:
    with pytest.raises(ValueError, match="overlap"):
        split_fixed_windows("x" * 500, chunk_size=10, overlap=10)


def test_overlap_greater_than_chunk_size_raises_instead_of_looping() -> None:
    with pytest.raises(ValueError, match="overlap"):
        split_fixed_windows("x" * 500, chunk_size=10, overlap=15)


def test_chunk_fixed_preserves_provenance_for_txt() -> None:
    document = make_document(
        segments=(Segment(text="x" * 250, section_heading=None, page_number=None),)
    )
    settings = Settings(_env_file=None, chunk_size=100, chunk_overlap=20)
    chunks = chunk_fixed(document, settings)
    assert all(chunk.source_file == document.source_file for chunk in chunks)
    assert all(chunk.document_id == document.document_id for chunk in chunks)
    assert all(chunk.chunking_strategy == ChunkingStrategy.FIXED for chunk in chunks)


def test_chunk_fixed_preserves_pdf_page_number() -> None:
    document = make_document(
        file_type="pdf",
        segments=(
            Segment(text="x" * 250, section_heading=None, page_number=1),
            Segment(text="y" * 250, section_heading=None, page_number=2),
        ),
    )
    settings = Settings(_env_file=None, chunk_size=100, chunk_overlap=20)
    chunks = chunk_fixed(document, settings)
    page_one_chunks = [c for c in chunks if c.text.startswith("x")]
    page_two_chunks = [c for c in chunks if c.text.startswith("y")]
    assert page_one_chunks and all(c.page_number == 1 for c in page_one_chunks)
    assert page_two_chunks and all(c.page_number == 2 for c in page_two_chunks)


def test_chunk_fixed_preserves_section_heading() -> None:
    document = make_document(
        file_type="markdown",
        segments=(Segment(text="x" * 250, section_heading="Intro > Details", page_number=None),),
    )
    settings = Settings(_env_file=None, chunk_size=100, chunk_overlap=20)
    chunks = chunk_fixed(document, settings)
    assert all(chunk.section_heading == "Intro > Details" for chunk in chunks)


def test_chunk_fixed_empty_segment_produces_no_chunks() -> None:
    document = make_document(segments=(Segment(text="", section_heading=None, page_number=None),))
    settings = Settings(_env_file=None, chunk_size=100, chunk_overlap=20)
    assert chunk_fixed(document, settings) == []


def test_max_size_invariant_holds_for_prose_text() -> None:
    # Realistic prose (with spaces/newlines) — no boundary trimming means the
    # hard chunk_size cap must still hold exactly for every emitted window.
    text = "The quick brown fox jumps over the lazy dog.\n\n" * 30
    for chunk_size, overlap in [(50, 10), (137, 33), (1000, 200)]:
        pieces = split_fixed_windows(text, chunk_size=chunk_size, overlap=overlap)
        assert all(len(piece) <= chunk_size for piece in pieces)


def test_chunk_fixed_respects_max_size_invariant() -> None:
    document = make_document(
        segments=(
            Segment(
                text="The quick brown fox jumps over the lazy dog. " * 40,
                section_heading=None,
                page_number=None,
            ),
        )
    )
    settings = Settings(_env_file=None, chunk_size=137, chunk_overlap=33)
    chunks = chunk_fixed(document, settings)
    assert chunks
    assert all(len(chunk.text) <= settings.chunk_size for chunk in chunks)


def test_chunk_fixed_chunk_index_is_sequential_across_segments() -> None:
    document = make_document(
        segments=(
            Segment(text="a" * 50, section_heading=None, page_number=1),
            Segment(text="b" * 50, section_heading=None, page_number=2),
        )
    )
    settings = Settings(_env_file=None, chunk_size=100, chunk_overlap=20)
    chunks = chunk_fixed(document, settings)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
