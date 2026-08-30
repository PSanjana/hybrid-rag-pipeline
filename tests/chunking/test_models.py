"""Tests for rag_pipeline.chunking.models."""

from rag_pipeline.chunking.models import Chunk, build_chunk, compute_chunk_id
from rag_pipeline.config import ChunkingStrategy

DOCUMENT_ID = "c" * 64


def test_build_chunk_contains_required_metadata() -> None:
    chunk = build_chunk(
        document_id=DOCUMENT_ID,
        chunk_index=0,
        text="Some chunk text.",
        source_file="notes.txt",
        section_heading="Intro",
        page_number=None,
        strategy=ChunkingStrategy.FIXED,
    )
    assert chunk.chunk_id
    assert chunk.document_id == DOCUMENT_ID
    assert chunk.chunk_index == 0
    assert chunk.text == "Some chunk text."
    assert chunk.source_file == "notes.txt"
    assert chunk.section_heading == "Intro"
    assert chunk.page_number is None
    assert chunk.chunking_strategy == ChunkingStrategy.FIXED
    assert chunk.character_count == len("Some chunk text.")


def test_character_count_matches_text() -> None:
    text = "Unicode café résumé naïve — 42 chars-ish."
    chunk = build_chunk(
        document_id=DOCUMENT_ID,
        chunk_index=0,
        text=text,
        source_file="notes.txt",
        section_heading=None,
        page_number=None,
        strategy=ChunkingStrategy.RECURSIVE,
    )
    assert chunk.character_count == len(text)


def test_to_dict_and_from_dict_round_trip() -> None:
    chunk = build_chunk(
        document_id=DOCUMENT_ID,
        chunk_index=2,
        text="Round trip me.",
        source_file="notes.txt",
        section_heading="Heading > Sub",
        page_number=3,
        strategy=ChunkingStrategy.SEMANTIC,
    )
    data = chunk.to_dict()
    assert data["chunking_strategy"] == "semantic"
    reloaded = Chunk.from_dict(data)
    assert reloaded == chunk


def test_chunk_ids_are_stable_for_identical_inputs() -> None:
    first = compute_chunk_id(DOCUMENT_ID, ChunkingStrategy.FIXED, 0, "same text")
    second = compute_chunk_id(DOCUMENT_ID, ChunkingStrategy.FIXED, 0, "same text")
    assert first == second


def test_chunk_ids_differ_by_strategy_even_with_same_index_and_text() -> None:
    fixed_id = compute_chunk_id(DOCUMENT_ID, ChunkingStrategy.FIXED, 0, "same text")
    recursive_id = compute_chunk_id(DOCUMENT_ID, ChunkingStrategy.RECURSIVE, 0, "same text")
    assert fixed_id != recursive_id


def test_chunk_ids_differ_by_text() -> None:
    a = compute_chunk_id(DOCUMENT_ID, ChunkingStrategy.FIXED, 0, "text a")
    b = compute_chunk_id(DOCUMENT_ID, ChunkingStrategy.FIXED, 0, "text b")
    assert a != b


def test_chunk_ids_differ_by_index() -> None:
    a = compute_chunk_id(DOCUMENT_ID, ChunkingStrategy.FIXED, 0, "same text")
    b = compute_chunk_id(DOCUMENT_ID, ChunkingStrategy.FIXED, 1, "same text")
    assert a != b


def test_chunk_ids_differ_by_document() -> None:
    a = compute_chunk_id("a" * 64, ChunkingStrategy.FIXED, 0, "same text")
    b = compute_chunk_id("b" * 64, ChunkingStrategy.FIXED, 0, "same text")
    assert a != b
