"""Tests for rag_pipeline.ingestion.service — full ingestion orchestration."""

import hashlib
from pathlib import Path

import pytest

from rag_pipeline.config import Settings
from rag_pipeline.ingestion.exceptions import SourceNotFoundError
from rag_pipeline.ingestion.service import compute_document_id, ingest_document
from rag_pipeline.ingestion.storage import load_processed, raw_document_path


def test_document_id_is_deterministic_sha256() -> None:
    data = b"hello world"
    assert compute_document_id(data) == hashlib.sha256(data).hexdigest()
    assert compute_document_id(data) == compute_document_id(data)


def test_different_bytes_produce_different_document_ids() -> None:
    assert compute_document_id(b"hello") != compute_document_id(b"hello!")


def test_ingest_missing_file_raises(test_settings: Settings) -> None:
    with pytest.raises(SourceNotFoundError):
        ingest_document(Path("/nonexistent/path/to/file.txt"), settings=test_settings)


def test_ingest_directory_raises(tmp_path: Path, test_settings: Settings) -> None:
    with pytest.raises(SourceNotFoundError):
        ingest_document(tmp_path, settings=test_settings)


def test_ingest_txt_end_to_end(tmp_path: Path, test_settings: Settings) -> None:
    source = tmp_path / "notes.txt"
    content = b"Hello, ingestion pipeline."
    source.write_bytes(content)

    document = ingest_document(source, settings=test_settings)

    assert document.document_id == hashlib.sha256(content).hexdigest()
    assert document.source_file == "notes.txt"
    assert document.file_type == "txt"
    assert document.segments[0].text == "Hello, ingestion pipeline."


def test_ingest_persists_raw_bytes_exactly(tmp_path: Path, test_settings: Settings) -> None:
    source = tmp_path / "notes.txt"
    content = b"Exact bytes must round-trip."
    source.write_bytes(content)

    document = ingest_document(source, settings=test_settings)

    raw_path = raw_document_path(test_settings.raw_data_dir, document.document_id, "notes.txt")
    assert raw_path.read_bytes() == content


def test_ingest_persists_processed_representation(tmp_path: Path, test_settings: Settings) -> None:
    source = tmp_path / "notes.txt"
    source.write_bytes(b"Some content.")

    document = ingest_document(source, settings=test_settings)

    reloaded = load_processed(test_settings.processed_data_dir, document.document_id)
    assert reloaded == document


def test_reingesting_identical_content_is_deterministic(
    tmp_path: Path, test_settings: Settings
) -> None:
    source = tmp_path / "notes.txt"
    source.write_bytes(b"Idempotent content.")

    first = ingest_document(source, settings=test_settings)
    second = ingest_document(source, settings=test_settings)

    assert first.document_id == second.document_id
    assert first == second


def test_processed_document_usable_after_source_file_removed(
    tmp_path: Path, test_settings: Settings
) -> None:
    source = tmp_path / "notes.txt"
    source.write_bytes(b"Will be deleted after ingestion.")

    document = ingest_document(source, settings=test_settings)
    source.unlink()

    reloaded = load_processed(test_settings.processed_data_dir, document.document_id)
    assert reloaded == document
    assert reloaded.segments[0].text == "Will be deleted after ingestion."
