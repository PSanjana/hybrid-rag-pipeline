"""Tests for rag_pipeline.indexing.dedup_report (duplicate report persistence)."""

from __future__ import annotations

import pytest

from rag_pipeline.config import Settings
from rag_pipeline.deduplication.models import DeduplicationResult, DuplicateRecord, DuplicateType
from rag_pipeline.indexing.dedup_report import (
    dedup_report_path,
    load_dedup_report,
    write_dedup_report,
)
from rag_pipeline.indexing.exceptions import DedupReportError

from .conftest import make_chunks


def _result(
    kept: list,
    duplicates: list[DuplicateRecord] | None = None,
    threshold: float = 0.95,
    version: str = "cosine_v1",
) -> DeduplicationResult:
    return DeduplicationResult(
        kept_chunks=tuple(kept),
        kept_embeddings=tuple((0.0, 0.0) for _ in kept),
        duplicates=tuple(duplicates or ()),
        algorithm_version=version,
        similarity_threshold=threshold,
    )


def test_report_persists_and_reloads(index_settings: Settings) -> None:
    chunks = make_chunks(2)
    write_dedup_report(index_settings, "snap-a", _result(chunks))

    report = load_dedup_report(index_settings, "snap-a")

    assert report.snapshot_id == "snap-a"
    assert report.dedup_algorithm_version == "cosine_v1"
    assert report.dedup_similarity_threshold == 0.95
    assert report.duplicates == ()


def test_empty_duplicate_report_is_valid(index_settings: Settings) -> None:
    chunks = make_chunks(1)
    write_dedup_report(index_settings, "snap-empty", _result(chunks, duplicates=[]))

    report = load_dedup_report(index_settings, "snap-empty")

    assert report.duplicates == ()


def test_exact_and_near_records_serialize_correctly(index_settings: Settings) -> None:
    chunks = make_chunks(2)
    record_exact = DuplicateRecord(
        skipped_chunk_id="skip1",
        canonical_chunk_id="keep1",
        duplicate_type=DuplicateType.EXACT,
        similarity=1.0,
        skipped_document_id="d" * 64,
        canonical_document_id="d" * 64,
        skipped_chunk_index=1,
        canonical_chunk_index=0,
        skipped_source_file="a.md",
        canonical_source_file="a.md",
    )
    record_near = DuplicateRecord(
        skipped_chunk_id="skip2",
        canonical_chunk_id="keep1",
        duplicate_type=DuplicateType.NEAR,
        similarity=0.97,
        skipped_document_id="e" * 64,
        canonical_document_id="d" * 64,
        skipped_chunk_index=2,
        canonical_chunk_index=0,
        skipped_source_file="b.md",
        canonical_source_file="a.md",
    )
    write_dedup_report(
        index_settings, "snap-b", _result(chunks, duplicates=[record_exact, record_near])
    )

    report = load_dedup_report(index_settings, "snap-b")

    assert report.duplicates == (record_exact, record_near)


def test_load_missing_report_raises(index_settings: Settings) -> None:
    with pytest.raises(DedupReportError):
        load_dedup_report(index_settings, "does-not-exist")


def test_load_corrupt_report_raises(index_settings: Settings) -> None:
    path = dedup_report_path(index_settings, "snap-corrupt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json", encoding="utf-8")
    with pytest.raises(DedupReportError):
        load_dedup_report(index_settings, "snap-corrupt")


def test_snapshot_id_mismatch_is_rejected(index_settings: Settings) -> None:
    chunks = make_chunks(1)
    write_dedup_report(index_settings, "snap-c", _result(chunks))
    with pytest.raises(DedupReportError):
        load_dedup_report(index_settings, "snap-c-typo")


def test_write_is_atomic_no_tmp_file_left_behind(index_settings: Settings) -> None:
    chunks = make_chunks(1)
    path = write_dedup_report(index_settings, "snap-d", _result(chunks))

    tmp_path = path.with_name(f"{path.name}.tmp")
    assert path.exists()
    assert not tmp_path.exists()
