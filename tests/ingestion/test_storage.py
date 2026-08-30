"""Tests for rag_pipeline.ingestion.storage."""

import json
from pathlib import Path

import pytest

from rag_pipeline.ingestion.exceptions import PersistenceError
from rag_pipeline.ingestion.models import NormalizedDocument, Segment
from rag_pipeline.ingestion.storage import (
    load_processed,
    processed_document_path,
    raw_document_path,
    write_processed,
    write_raw,
)

DOCUMENT_ID = "a" * 64


def _sample_document() -> NormalizedDocument:
    return NormalizedDocument(
        document_id=DOCUMENT_ID,
        source_file="notes.txt",
        file_type="txt",
        raw_path=f"{DOCUMENT_ID}/notes.txt",
        segments=(Segment(text="Hello world.", section_heading=None, page_number=None),),
    )


def test_write_raw_preserves_exact_bytes(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    original = b"\x00binary-ish\xffbytes\n\r\n"
    target = write_raw(raw_root, DOCUMENT_ID, "notes.txt", original)
    assert target.read_bytes() == original


def test_write_processed_creates_json_file(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    target = write_processed(processed_root, _sample_document())
    assert target.exists()
    assert target == processed_document_path(processed_root, DOCUMENT_ID)


def test_processed_json_contains_schema_version(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    target = write_processed(processed_root, _sample_document())
    data = json.loads(target.read_text())
    assert data["schema_version"] == 1


def test_processed_json_contains_document_and_segment_metadata(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    target = write_processed(processed_root, _sample_document())
    data = json.loads(target.read_text())
    assert data["document_id"] == DOCUMENT_ID
    assert data["source_file"] == "notes.txt"
    assert data["file_type"] == "txt"
    assert data["raw_path"] == f"{DOCUMENT_ID}/notes.txt"
    assert data["segments"] == [
        {"text": "Hello world.", "section_heading": None, "page_number": None}
    ]


def test_load_processed_reconstructs_equivalent_document(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    original = _sample_document()
    write_processed(processed_root, original)
    loaded = load_processed(processed_root, DOCUMENT_ID)
    assert loaded == original


def test_load_processed_missing_document_raises(tmp_path: Path) -> None:
    with pytest.raises(PersistenceError):
        load_processed(tmp_path / "processed", "missing-id")


@pytest.mark.parametrize(
    "malicious_filename",
    ["../../etc/passwd", "../escape.txt", "/etc/passwd", "..\\..\\escape.txt"],
)
def test_filename_cannot_escape_raw_data_directory(tmp_path: Path, malicious_filename: str) -> None:
    raw_root = tmp_path / "raw"
    with pytest.raises(PersistenceError):
        raw_document_path(raw_root, DOCUMENT_ID, malicious_filename)


def test_safe_filename_stays_within_raw_data_directory(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    target = raw_document_path(raw_root, DOCUMENT_ID, "notes.txt")
    assert target.resolve().parent == (raw_root / DOCUMENT_ID).resolve()
